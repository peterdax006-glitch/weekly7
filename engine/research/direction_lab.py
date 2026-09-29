"""Direction research laboratory (Research Brain contract C66 section 10; canons C24, C56, C63, C64, C66). IMPLEMENTED - NOT VALIDATED.

Objective 2 of the contract: among stocks the VOLATILITY model would really have selected, is there any side signal? The lab is
deliberately built so that the honest answer 'no reliable direction signal' is a first-class outcome, reached by the same code path
as a positive one (the earlier work found direction at ~51-52%, so a lab that can only say yes would be a defect).

Pipeline (all research side, MATURED_RESEARCH_STATE; a result reaches a decision only through MaturedRecord.gate(now)):
  universe -> P(vol) -> predicted movers -> P(up | mover) -> risk / payoff -> verdict
  1. VolatilityEvidence / DirectionGate: direction is only studied once the volatility stage has out-of-sample evidence, and the
     gate is an object the priority engine can read (section 10 title; RESEARCH_MAPPING: 'the volatility-evidence gate must be an object').
  2. ConditionalUniverse: the pool is built from the champion's WALK-FORWARD scores, never from realised movers. A canary
     (score_leak_canary / realised_mover_canary) refuses a pool that is suspiciously identical to the realised movers.
  3. Fifteen HYPOTHESES (the section 10 list, one HypothesisSpec each) become feature families; missing inputs are reported as
     UNAVAILABLE, never silently dropped.
  4. Every hypothesis must beat SIX comparators: base rate, random scores, a model fitted on label-shuffled (within-week) data, a
     no-learning fixed-sign rule, the best simple one-feature baseline and the existing production direction model. 'Beat' means a
     paired week-block bootstrap on Brier score AND a max-T permutation p-value that prices having tried every hypothesis.
  5. Controls: a planted signal must be found, a look-ahead canary must be caught by the leakage screen, shuffled labels must read ~50%.
     If any control misbehaves the whole run is CONTROL_FAILURE and no hypothesis may be reported as a candidate.
  6. Candidates are then stress-tested: per-year / sector / regime stability, independent coverage bands, held-out-ticker transfer
     (memorisation), decay, and the payoff after costs (an 80%-accurate call on a small win with a large loss is not an edge).
  7. A power report states the smallest accuracy edge this sample could have seen, so 'no signal' is never confused with 'no power'.

BUILDS ON (imports, never copies): engine.direction_features (walk_forward protocol, labels, frontier, eighty_question, controls,
audit_point_in_time, plant_signal, DirModel), engine.direction (Brier / ECE / Wilson), engine.direction_ablate (week bootstrap),
engine.direction_calib (compare_gates), engine.learning.calibration (change_scan, minimum_detectable_gap, platt_slope), engine.fv_pipeline
(the shared 0.80 target and certify_threshold), engine.learning.trader_view (blind-safe release), engine.research.core.
Public entry: step(state, now, inputs) -> (state, LabReport); DirectionLab.run(inputs, now) for one-shot use."""
from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from engine import direction as D
from engine import direction_ablate as DA
from engine import direction_calib as DC
from engine import direction_features as DF
from engine.learning import calibration as LC
from engine.learning import trader_view as TV
from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, current_code_hash, require_past,
                                  stable_hash)
from engine.research.core import (Availability, DecisionEffect, ExperimentValue, GateVerdict, MaturedRecord, Namespace, Problem,
                                  ResearchQuestion, ResearchState, Stage)

EIGHTY = 0.80                          # the section 10/11 target; fv_pipeline.FVConfig.gate must agree (see shared_target)
EPS = 1e-9
TARGETS = ("up_sign", "up_first", "up_rel")   # up_sign: finished the week above entry; up_first: the +10% barrier came first; up_rel: above the week's pool median
FORBIDDEN_RANK_COLUMNS = ("mover", "up_first", "up_sign", "fwd", "amb", "end_date", "entry_date")


class Outcome(_StrEnum):
    """What the lab may say about one hypothesis. NO_RELIABLE_SIGNAL is the expected, first-class answer."""
    NO_RELIABLE_SIGNAL = "NO_RELIABLE_SIGNAL"
    WEAK_UNREPLICATED = "WEAK_UNREPLICATED"          # beat some comparators or one era, but not all the gates
    CANDIDATE = "CANDIDATE"                          # beat every comparator with family-wise control; still needs fresh holdout
    CONTROL_FAILURE = "CONTROL_FAILURE"              # a control misbehaved: the run cannot vouch for any hypothesis
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"
    LEAK_SUSPECT = "LEAK_SUSPECT"                    # the feature predicts too well to be point-in-time
    VOLATILITY_GATE_CLOSED = "VOLATILITY_GATE_CLOSED"


class Comparator(_StrEnum):
    BASE_RATE = "base_rate"
    RANDOM = "random"
    SHUFFLED = "shuffled_labels"
    NO_LEARNING = "no_learning"
    SIMPLE_BASELINE = "simple_baseline"
    EXISTING_MODEL = "existing_model"


ALL_COMPARATORS = tuple(Comparator)
TAG_BASE, TAG_RANDOM, TAG_SHUF, TAG_SIMPLE, TAG_EXISTING = ("base:prior", "ctl_random", "ctl_shuffled_week", "simple:baseline",
                                                            "existing:prod")


@dataclasses.dataclass(frozen=True)
class LabConfig:
    """Every knob of the lab in one validated, hashable object (the config hash is part of each result's provenance)."""
    target: str = "up_sign"
    test_years: tuple[int, ...] = ()
    n_pick: int = 30
    n_pool: int = 100
    coverages: tuple[float, ...] = DF.COVERAGES
    models: tuple[str, ...] = ("linear", "gbm")
    screen_models: tuple[str, ...] = ("linear",)
    n_boot: int = 600
    n_perm: int = 400
    alpha: float = 0.05
    gate: float = EIGHTY
    min_test_rows: int = 400
    min_weeks: int = 26
    min_eras: int = 3
    era_share: float = 0.6                    # a candidate must beat the base rate in at least this share of test years
    edge_floor: float = 0.002                 # Brier skill (1 - Brier/Brier_base) below this is not worth acting on
    screen_p: float = 0.25                    # stage-1 permutation p above this ends a hypothesis cheaply
    embargo_days: int = 14
    min_train: int = 2000
    min_calib: int = 200
    cost_bps: float = 7.0                     # fee + slippage per side, same order as fv_pipeline's defaults
    leak_auc: float = 0.80                    # single-feature AUC above this is not a point-in-time feature
    use_miner: bool = False
    seed: int = 0

    def validate(self) -> "LabConfig":
        errs = []
        if self.target not in TARGETS:
            errs.append(f"target {self.target!r} not in {TARGETS}")
        if not 0 < self.n_pick <= self.n_pool:
            errs.append("need 0 < n_pick <= n_pool")
        if not 0 < self.alpha < 0.5:
            errs.append("alpha must be in (0, 0.5)")
        if not self.coverages or any(not 0 < c <= 1 for c in self.coverages):
            errs.append("coverages must lie in (0, 1]")
        if not set(self.models) <= {"linear", "gbm"} or not set(self.screen_models) <= {"linear", "gbm"}:
            errs.append("models must be 'linear' and/or 'gbm'")
        if self.n_boot < 50 or self.n_perm < 20:
            errs.append("n_boot >= 50 and n_perm >= 20 are needed for any interval")
        if not 0.5 <= self.gate < 1:
            errs.append("gate must lie in [0.5, 1)")
        if not 0 < self.era_share <= 1:
            errs.append("era_share must lie in (0, 1]")
        if errs:
            raise ValueError("LabConfig: " + "; ".join(errs))
        return self

    def hash(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


def shared_target() -> float:
    """The one definition of the 80% target: the fv pipeline's certification gate. A different number here would make the lab and
    the pipeline talk about different things (RESEARCH_MAPPING, section 10-11 note)."""
    from engine.fv_pipeline import FVConfig
    return float(FVConfig().gate)


# ---------------------------------------------------------------------------------------------- hypothesis catalogue
@dataclasses.dataclass(frozen=True)
class HypothesisSpec:
    """One section-10 research topic. `columns` are the candidate inputs (any subset present is used); `rule_columns` feed the
    fixed-sign NO-LEARNING comparator (prior_sign * mean sign of these), which has no fitted parameter at all."""
    name: str
    topic: str
    columns: tuple[str, ...]
    mechanism: str
    prior_sign: int                                   # +1 continuation, -1 reversal, 0 the literature gives no sign
    rule_columns: tuple[str, ...] = ()
    min_columns: int = 1

    def validate(self) -> "HypothesisSpec":
        errs = []
        if not self.name or not self.columns:
            errs.append("name and columns are required")
        if self.prior_sign not in (-1, 0, 1):
            errs.append("prior_sign must be -1, 0 or 1")
        if self.prior_sign != 0 and not self.rule_columns:
            errs.append("a signed prior needs rule_columns for the no-learning comparator")
        if not set(self.rule_columns) <= set(self.columns):
            errs.append("rule_columns must be a subset of columns")
        if self.min_columns < 1 or self.min_columns > len(self.columns):
            errs.append("min_columns out of range")
        if errs:
            raise ValueError(f"HypothesisSpec {self.name}: " + "; ".join(errs))
        return self


def _spec(name, topic, columns, mechanism, sign, rules=(), min_columns=1) -> HypothesisSpec:
    return HypothesisSpec(name, topic, tuple(columns), mechanism, sign, tuple(rules), min_columns).validate()


HYPOTHESES: dict[str, HypothesisSpec] = {s.name: s for s in (
    _spec("momentum", "momentum", ["r20", "r60", "r120", "mom_12_1", "mom_skip"],
          "Intermediate-term winners keep winning (Jegadeesh-Titman); may reverse inside a predicted-mover week.", 1,
          ["r60", "mom_12_1"]),
    _spec("reversal", "reversal", ["r1", "r5", "r5_nonews", "rev5_news_adj", "rev_strength"],
          "Short-term overreaction: last week's no-news move is partly undone (Chan 2003).", -1, ["r5_nonews", "r5"]),
    _spec("event_reaction", "event reaction", ["news5", "ev_offering", "ev_shelf", "ev_activist", "ev_activist_amend",
                                              "ev_agreement", "ev_red_flag", "evt_filing", "event_net"],
          "Filings carry a signed reaction; offerings/red flags skew down, activists/agreements skew up.", 0),
    _spec("earnings_reaction", "earnings reaction", ["ear", "ear_recent", "ear_sign_recent", "ear_x_vol", "ear_volsurge",
                                                     "days_since_earn", "days_to_earn", "earn_in_week", "evt_earn"],
          "Post-earnings-announcement drift: the surprise direction persists for weeks (Bernard-Thomas).", 1,
          ["ear_sign_recent"]),
    _spec("insider", "insider behavior", ["ins_buyers30", "ins_value30", "ins_officer30", "ins_opportunistic30"],
          "Opportunistic insider buying clusters precede positive drift (Cohen-Malloy-Pomorski).", 1,
          ["ins_buyers30", "ins_opportunistic30"]),
    _spec("range_52w", "52-week positioning", ["dist_52wh", "dist_ma50", "dist_ma200", "near_high_flag", "range_pos"],
          "Anchoring on the 52-week high: stocks near it keep rising, stocks far below it drift (George-Hwang).", 1,
          ["dist_52wh"]),
    _spec("relative_strength", "relative strength", ["rel_ind20", "rel_ind60", "xs_r20", "xs_r60", "rs_spread"],
          "Cross-sectional winners against peers persist for the week.", 1, ["rel_ind20", "xs_r20"]),
    _spec("sector_behavior", "sector behavior", ["sector_mom", "sector_rev", "rel_sector_r5", "rel_ind20"],
          "Industry momentum leads individual stocks (Moskowitz-Grinblatt); needs a sector map, else UNAVAILABLE.", 1,
          ["sector_mom"], min_columns=1),
    _spec("market_regime", "market regime", ["m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_breadth", "reg_x_r5", "reg_x_mom"],
          "The market's state changes the sign of short-horizon effects (bear markets reverse harder).", 0),
    _spec("volume_structure", "volume structure", ["ear_volsurge", "vol_surge", "vol_ratio5", "dvol_ratio", "obv_slope",
                                                   "vol_x_r5"],
          "Volume-confirmed moves continue, low-volume moves reverse (Gervais et al., Lee-Swaminathan).", 1, ["vol_x_r5"]),
    _spec("intraday_path", "intraday path", ["close_loc", "intraday20", "overnight20", "path_efficiency"],
          "Closing near the day's high signals buying pressure that carries into the next session.", 1, ["close_loc"]),
    _spec("gap_behavior", "gap behavior", ["gap_today", "overnight20", "gap_x_closeloc", "gap_fill"],
          "Gaps either continue (news) or fill (noise); the sign depends on the close location.", 0),
    _spec("pattern_combinations", "pattern combinations", ["combo_rev_mom", "combo_ear_range", "combo_gap_vol",
                                                           "combo_news_reg", "miner"],
          "Pairwise conjunctions of the simple signals; the space is large so it carries the heaviest multiplicity price.", 0),
    _spec("volatility_regime", "volatility regime", ["m_vix", "m_vix_chg5", "m_vix_term", "m_dispersion", "atr_pct",
                                                     "vol20", "vix_x_r5"],
          "High-volatility regimes raise reversal and lower continuation.", 0),
    _spec("cross_sectional", "cross-sectional relationships", ["xs_r5", "xs_r20", "xs_ear", "xs_dist_52wh", "xs_close_loc",
                                                                "xs_rel_ind20", "xs_gap_today"],
          "Where a name ranks among the week's other predicted movers, not its raw value.", 0),
)}
FAMILY_ORDER = tuple(HYPOTHESES)
CONTROL_KEYS = ("planted", "leak", "ctl_")


def resolve_columns(available: Sequence[str], spec: HypothesisSpec) -> tuple[str, ...]:
    """The subset of the spec's candidate inputs that exist; a spec with fewer than min_columns is UNAVAILABLE."""
    have = set(available)
    got = tuple(c for c in spec.columns if c in have)
    return got if len(got) >= spec.min_columns else ()


def availability_table(available: Sequence[str], specs: Mapping[str, HypothesisSpec] | None = None) -> pd.DataFrame:
    """Which hypotheses can be tested at all, and which of their inputs are missing. Missing inputs are a finding, not a skip."""
    rows = []
    for name, spec in (specs or HYPOTHESES).items():
        got = resolve_columns(available, spec)
        rows.append(dict(hypothesis=name, topic=spec.topic, n_inputs=len(got), n_missing=len(spec.columns) - len(got),
                         missing=",".join(c for c in spec.columns if c not in got),
                         availability=(Availability.KNOWN_BEFORE_EVENT if got else Availability.UNAVAILABLE).value))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- inputs
@dataclasses.dataclass
class LabInputs:
    """Everything the lab reads. X and score are point-in-time (the score MUST be the champion's walk-forward P(vol), NaN where the
    champion had no fitted model yet); labels are matured outcomes (research side only)."""
    X: pd.DataFrame
    score: pd.Series
    labels: pd.DataFrame
    existing_p: pd.Series | None = None
    sector: pd.Series | None = None                    # ticker -> sector, or (date, ticker) -> sector
    volatility_evidence: "VolatilityEvidence | None" = None
    pattern_score: pd.Series | None = None             # optional signed pattern-miner score (point-in-time, walk-forward)

    LABEL_COLUMNS = ("entry_date", "end_date", "fwd", "mover", "up_sign", "up_first")

    def validate(self, now) -> list[str]:
        """Fail-closed structural checks; returns warnings, raises FirewallBreach / ValueError on anything that would leak."""
        errs, warn = [], []
        for nm, obj in (("X", self.X), ("score", self.score), ("labels", self.labels)):
            if not isinstance(obj.index, pd.MultiIndex) or obj.index.nlevels != 2:
                errs.append(f"{nm} must be indexed by (date, ticker)")
            elif obj.index.has_duplicates:
                errs.append(f"{nm} has duplicated (date, ticker) rows")
        missing = [c for c in self.LABEL_COLUMNS if c not in self.labels.columns]
        if missing:
            errs.append(f"labels lack columns {missing}")
        if errs:
            raise ValueError("LabInputs: " + "; ".join(errs))
        dates = pd.DatetimeIndex(self.labels.index.get_level_values(0))
        if not (pd.DatetimeIndex(self.labels["entry_date"]) > dates).all():
            raise FirewallBreach("a label enters on or before its decision date: the labels are not next-open fills (rule 3)")
        sd = pd.DatetimeIndex(self.score.index.get_level_values(0))
        n_future = int((sd >= pd.Timestamp(as_date(now))).sum())
        if n_future:
            warn.append(f"{n_future} score rows are dated on/after now and will be ignored")
        if self.existing_p is not None and (self.existing_p.dropna().lt(0).any() or self.existing_p.dropna().gt(1).any()):
            raise ValueError("existing_p must be a probability")
        if self.X.columns.duplicated().any():
            raise ValueError("X has duplicated feature columns")
        bad = [c for c in self.X.columns if str(c).lower() in FORBIDDEN_RANK_COLUMNS]
        if bad:
            raise FirewallBreach(f"X carries outcome-named columns {bad}: refusing to build features from them")
        return warn


# ---------------------------------------------------------------------------------------------- volatility evidence + gate
@dataclasses.dataclass(frozen=True)
class VolatilityEvidence:
    """Out-of-sample proof that the volatility stage can find movers: mean weekly AUC of P(vol) against the realised mover flag and
    the lift of the top-n picks over the universe mover rate, both with week-bootstrap lower bounds."""
    n_weeks: int
    n_rows: int
    auc: float
    auc_lo: float
    top_lift: float
    top_lift_lo: float
    mover_rate: float
    pick_rate: float
    source: str = "computed"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _weekly_auc(score: np.ndarray, flag: np.ndarray) -> float:
    n1 = int(flag.sum())
    n0 = len(flag) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[flag.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def assess_volatility_evidence(score: pd.Series, mover: pd.Series, n_pick: int = 30, n_boot: int = 400, seed: int = 0) -> VolatilityEvidence:
    """Weekly AUC and top-pick lift of the champion volatility score, judged by resampling WEEKS (rows of one week share a market)."""
    df = pd.DataFrame({"s": score, "m": mover.reindex(score.index)}).dropna()
    if df.empty:
        return VolatilityEvidence(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), "empty")
    df["m"] = df["m"].astype(float)
    df["wk"] = df.index.get_level_values(0)
    aucs, hits, picks, movers, uni = [], [], [], [], []
    for _, g in df.groupby("wk", sort=True):
        a = _weekly_auc(g["s"].to_numpy(), g["m"].to_numpy())
        if not np.isfinite(a):
            continue
        top = g["s"].rank(ascending=False, method="first") <= n_pick
        aucs.append(a)
        hits.append(float(g.loc[top, "m"].sum()))
        picks.append(float(top.sum()))
        movers.append(float(g["m"].sum()))
        uni.append(float(len(g)))
    if not aucs:
        return VolatilityEvidence(0, len(df), float("nan"), float("nan"), float("nan"), float("nan"), float(df["m"].mean()), float("nan"), "one_class")
    aucs, hits, picks, movers, uni = (np.asarray(v) for v in (aucs, hits, picks, movers, uni))
    rng = np.random.default_rng(seed)
    W = len(aucs)
    idx = rng.integers(0, W, size=(n_boot, W))
    base = movers.sum() / uni.sum()
    lift = (hits.sum() / picks.sum()) / base if base > 0 else float("nan")
    b_auc = aucs[idx].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b_lift = (hits[idx].sum(1) / picks[idx].sum(1)) / (movers[idx].sum(1) / uni[idx].sum(1))
    b_lift = b_lift[np.isfinite(b_lift)]
    return VolatilityEvidence(W, int(uni.sum()), float(aucs.mean()), float(np.quantile(b_auc, 0.05)), float(lift),
                              float(np.quantile(b_lift, 0.05)) if len(b_lift) else float("nan"), float(base),
                              float(hits.sum() / picks.sum()))


def score_leak_canary(ev: VolatilityEvidence, max_auc: float = 0.90) -> list[str]:
    """A volatility score that ranks realised movers near-perfectly is not a forecast: it was built from the outcome. The best
    real mover models sit near 0.6-0.7 (state/research); 0.90 is a tripwire, not a target."""
    out = []
    if np.isfinite(ev.auc) and ev.auc > max_auc:
        out.append(f"volatility score AUC {ev.auc:.3f} > {max_auc}: too good to be point-in-time")
    if np.isfinite(ev.pick_rate) and ev.n_rows >= 500 and ev.pick_rate >= 0.98:
        out.append(f"{ev.pick_rate:.1%} of the picks are realised movers: the pool looks like the realised movers")
    return out


@dataclasses.dataclass(frozen=True)
class GateDecision:
    is_open: bool
    verdict: GateVerdict
    reasons: tuple[str, ...]
    evidence: VolatilityEvidence

    def priority_multiplier(self, closed_floor: float = 0.05) -> float:
        """What the research-priority engine multiplies the direction target by: full weight when open, a small floor when closed
        (a closed gate must not starve the check that re-opens it, but must stop expensive direction runs)."""
        return 1.0 if self.is_open else closed_floor


class DirectionGate:
    """'Direction only after volatility has evidence' as an object. Open requires: enough weeks, AUC lower bound above chance by a
    margin, top-pick lift lower bound above 1, and no leak canary."""

    def __init__(self, min_weeks: int = 52, min_auc_lo: float = 0.52, min_lift_lo: float = 1.15):
        if min_weeks < 1 or not 0.5 <= min_auc_lo < 1 or min_lift_lo < 1:
            raise ValueError("DirectionGate thresholds are out of range")
        self.min_weeks, self.min_auc_lo, self.min_lift_lo = min_weeks, min_auc_lo, min_lift_lo

    def assess(self, ev: VolatilityEvidence) -> GateDecision:
        why = []
        if ev.n_weeks < self.min_weeks:
            why.append(f"only {ev.n_weeks} evaluated weeks < {self.min_weeks}")
        if not (np.isfinite(ev.auc_lo) and ev.auc_lo >= self.min_auc_lo):
            why.append(f"AUC lower bound {ev.auc_lo:.3f} < {self.min_auc_lo}")
        if not (np.isfinite(ev.top_lift_lo) and ev.top_lift_lo >= self.min_lift_lo):
            why.append(f"top-pick lift lower bound {ev.top_lift_lo:.2f} < {self.min_lift_lo}")
        leak = score_leak_canary(ev)
        if leak:
            return GateDecision(False, GateVerdict.QUARANTINED, tuple(leak), ev)
        if why:
            return GateDecision(False, GateVerdict.NEEDS_MORE_EVIDENCE if ev.n_weeks < self.min_weeks else GateVerdict.FAILED,
                                tuple(why), ev)
        return GateDecision(True, GateVerdict.PROMOTE, ("volatility stage has out-of-sample evidence",), ev)


# ---------------------------------------------------------------------------------------------- the conditional universe
@dataclasses.dataclass
class Universe:
    """pool: one row per (date, ticker) in the champion's top n_pool, in a stable order (row position is the join key everywhere
    downstream). funnel: what fell out at each stage, per year."""
    pool: pd.DataFrame
    funnel: pd.DataFrame
    dropped_unmatured: int
    warnings: list[str]


def _segments(Xp: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    need = {"days_since_earn", "earn_in_week", "news5"}
    seg = DF.event_type(Xp) if need <= set(Xp.columns) else pd.Series("na", index=Xp.index, name="seg")
    reg = DF.regime_type(Xp) if "m_spy_ma200" in Xp.columns else pd.Series("na", index=Xp.index, name="reg")
    return seg, reg


def _sector_of(idx: pd.MultiIndex, sector: pd.Series | None) -> pd.Series:
    if sector is None:
        return pd.Series("na", index=idx)
    if isinstance(sector.index, pd.MultiIndex):
        return sector.reindex(idx).fillna("na")
    return pd.Series(idx.get_level_values(1).map(sector).fillna("na"), index=idx)


class ConditionalUniverse:
    """score -> predicted movers -> labelled direction pool. Ranking reads the score only: DirectionLab.run refuses a score named
    like an outcome column, and the pool keeps realised-mover flags in a separately named diagnostic column (`realised_mover`) that
    the feature builders never read (screen_feature_leakage and name_leak_reasons reject outcome-named columns)."""

    def __init__(self, cfg: LabConfig):
        self.cfg = cfg

    def build(self, inputs: LabInputs, now) -> Universe:
        cfg, warn = self.cfg, inputs.validate(now)
        now_ts = pd.Timestamp(as_date(now))
        lab = inputs.labels
        matured = pd.DatetimeIndex(lab["end_date"]) < now_ts
        dropped = int((~matured).sum())
        lab = lab[matured]
        sc = inputs.score.dropna()
        sc = sc[pd.DatetimeIndex(sc.index.get_level_values(0)) < now_ts]
        if sc.empty:
            return Universe(pd.DataFrame(), pd.DataFrame(), dropped, warn + ["no scored rows before now"])
        picks = DF.select_picks(sc, cfg.n_pick, cfg.n_pool)
        pool = picks[picks["pool"]].copy()
        pool.index = pool.index.set_names([None, None])     # avoid index-level/column ambiguity with the date column
        pool["score"] = sc.reindex(pool.index).to_numpy()
        L = lab.reindex(pool.index)
        pool["fwd"] = L["fwd"].to_numpy(dtype=float)
        pool["up"] = np.nan if cfg.target == "up_rel" else L[cfg.target].to_numpy(dtype=float)
        pool["entry_date"] = L["entry_date"].to_numpy()
        pool["end_date"] = L["end_date"].to_numpy()
        pool["realised_mover"] = L["mover"].to_numpy(dtype=float)
        Xp = inputs.X.reindex(pool.index)
        pool["seg"], pool["reg"] = (s.to_numpy() for s in _segments(Xp))
        pool["sector"] = _sector_of(pool.index, inputs.sector).to_numpy()
        pool["date"] = pd.DatetimeIndex(pool.index.get_level_values(0))
        pool["ticker"] = pool.index.get_level_values(1)
        pool["year"] = pool["date"].dt.year
        pool = pool.sort_values(["date", "rank"], kind="stable")
        if cfg.target == "up_rel":
            pool["up"] = add_relative_target(pool).to_numpy()
        funnel = self._funnel(sc, pool)
        if pool["up"].notna().mean() < 0.5:
            warn.append("fewer than half of the pool rows are labelled: check the label alignment")
        canary = realised_mover_canary(pool)
        warn += canary
        return Universe(pool, funnel, dropped, warn)

    @staticmethod
    def _funnel(sc: pd.Series, pool: pd.DataFrame) -> pd.DataFrame:
        yr = pd.DatetimeIndex(sc.index.get_level_values(0)).year
        scored = pd.Series(1, index=sc.index).groupby(yr).sum().rename("scored_rows")
        g = pool.groupby("year")
        out = pd.DataFrame({"scored_rows": scored, "pool_rows": g.size(), "pick_rows": g["pick"].sum(),
                            "labelled_pool": g["up"].apply(lambda s: int(s.notna().sum())),
                            "labelled_picks": pool[pool["pick"]].groupby("year")["up"].apply(lambda s: int(s.notna().sum())),
                            "weeks": g["date"].nunique()})
        return out.fillna(0).astype(int).reset_index().rename(columns={"index": "year"})


def realised_mover_canary(pool: pd.DataFrame, min_rows: int = 100) -> list[str]:
    """Warns when the pool is (almost) exactly the realised movers, which would make direction an easier, irrelevant problem."""
    m = pool.loc[pool["pick"], "realised_mover"].dropna()
    if len(m) < min_rows:
        return []
    msgs = []
    if m.mean() >= 0.98:
        msgs.append(f"{m.mean():.1%} of picks are realised movers: the pool is built from outcomes")
    rest = pool.loc[~pool["pick"], "realised_mover"].dropna()
    if len(rest) >= min_rows and rest.mean() <= 0.02 and m.mean() >= 0.90:
        msgs.append("picks are realised movers and the rest of the pool is not: perfect separation")
    return msgs


def universe_ease_check(pool: pd.DataFrame, labels: pd.DataFrame, cfg: LabConfig) -> pd.DataFrame:
    """Why direction must be judged on predicted movers only: compares the up-rate and the accuracy of a trivial reversal rule on
    (a) the whole labelled universe, (b) realised movers (NOT a decision universe), (c) the pool, (d) the picks. If (b) looks easier
    than (d) the older 'realised movers' direction results are optimistic and must not be quoted."""
    rows = []
    lab = labels.dropna(subset=[cfg.target])
    subsets = {"all_labelled": lab[cfg.target], "realised_movers": lab.loc[lab["mover"].astype(bool), cfg.target],
               "pool": pool.dropna(subset=["up"])["up"], "picks": pool[pool["pick"]].dropna(subset=["up"])["up"]}
    for name, s in subsets.items():
        n = len(s)
        up = float(s.mean()) if n else float("nan")
        lo, hi = LC.wilson(float(s.sum()), n) if n else (float("nan"), float("nan"))
        rows.append(dict(subset=name, n=n, up_rate=up, majority_acc=max(up, 1 - up) if n else float("nan"), up_lo=lo, up_hi=hi,
                         decision_universe=name in ("pool", "picks")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- features
def _f(X: pd.DataFrame, name: str) -> pd.Series | None:
    return X[name].astype("float64") if name in X.columns else None


def _sgn(s: pd.Series | None) -> pd.Series | None:
    return None if s is None else np.sign(s.fillna(0.0))


def _leave_one_out_mean(values: pd.Series, dates: pd.Series, groups: pd.Series) -> pd.Series:
    """Mean of `values` over the OTHER rows of the same (date, group); NaN for groups of one or the 'na' group. Leave-one-out matters:
    including the row itself would leak its own return into the sector feature."""
    df = pd.DataFrame({"v": values.to_numpy(), "d": dates.to_numpy(), "g": groups.to_numpy()})
    key = df.groupby(["d", "g"])["v"]
    s, n = key.transform("sum"), key.transform("count")
    own = df["v"].fillna(0.0)
    cnt = n - df["v"].notna().astype(int)
    out = (s - own) / cnt.where(cnt > 0)
    out[df["g"] == "na"] = np.nan
    return pd.Series(out.to_numpy(), index=values.index)


def derive_features(Xp: pd.DataFrame, pool: pd.DataFrame, pattern_score: pd.Series | None = None) -> pd.DataFrame:
    """Signed, interpretable derived inputs computed on the POOL rows only (cross-sectional ranks are ranks among the week's predicted
    movers, which is the population a direction bet is drawn from). Everything reads columns dated at the decision close; a derived
    column is simply absent when its inputs are (availability_table then reports it)."""
    d = DF.add_derived(Xp) if {"ear", "days_since_earn", "ear_volsurge", "r5_nonews", "r5_news"} <= set(Xp.columns) else Xp.copy()
    out: dict[str, pd.Series] = {}
    r5, r20, r60 = _f(d, "r5"), _f(d, "r20"), _f(d, "r60")
    if r60 is not None and r5 is not None:
        out["mom_skip"] = r60 - r5                                     # 12-1 style: momentum with the latest week skipped
    if r5 is not None and r20 is not None:
        out["rev_strength"] = (-r5 / (r20.abs() + 0.02)).clip(-10, 10)  # reversal relative to the prevailing trend
    ev = [c for c in ("ev_activist", "ev_activist_amend", "ev_agreement") if c in d]
    bad = [c for c in ("ev_offering", "ev_shelf", "ev_red_flag") if c in d]
    if ev or bad:
        out["event_net"] = (d[ev].sum(axis=1) if ev else 0.0) - (d[bad].sum(axis=1) if bad else 0.0)
    dist = _f(d, "dist_52wh")
    if dist is not None:
        out["near_high_flag"] = (dist > -0.05).astype("float64")
    d50, d200 = _f(d, "dist_ma50"), _f(d, "dist_ma200")
    if d50 is not None and d200 is not None:
        out["range_pos"] = d50 - d200
    xs_cols = [c for c in ("r5", "r20", "r60", "ear", "dist_52wh", "close_loc", "rel_ind20", "gap_today", "ear_recent") if c in d]
    if xs_cols:
        R = DF.xs_rank(d, xs_cols)
        for c in xs_cols:
            out["xs_" + c] = R[c].astype("float64")
    if "xs_r60" in out and "xs_r5" in out:
        out["rs_spread"] = out["xs_r60"] - out["xs_r5"]
    dates, sect = pool["date"], pool["sector"]
    if r20 is not None and (sect != "na").any():
        m20 = _leave_one_out_mean(r20, dates, sect)
        out["sector_mom"] = m20
        if r5 is not None:
            m5 = _leave_one_out_mean(r5, dates, sect)
            out["sector_rev"] = -m5
            out["rel_sector_r5"] = r5 - m5
    reg = _sgn(_f(d, "m_spy_ma200"))
    if reg is not None and r5 is not None:
        out["reg_x_r5"] = reg * r5
    if reg is not None and r60 is not None:
        out["reg_x_mom"] = reg * r60
    vs = _f(d, "ear_volsurge") if "ear_volsurge" in d else _f(d, "vol_surge")
    if vs is not None:
        lv = np.log1p(vs.clip(0, 20).fillna(0))
        if r5 is not None:
            out["vol_x_r5"] = lv * r5
        gap = _f(d, "gap_today")
        if gap is not None:
            out["combo_gap_vol"] = gap * lv
    cl, ic = _f(d, "close_loc"), _f(d, "intraday20")
    if cl is not None and ic is not None:
        out["path_efficiency"] = (cl - 0.5) * ic                 # fixed midpoint: a panel median would read later dates
    gap = _f(d, "gap_today")
    if gap is not None:
        if "xs_close_loc" in out:
            out["gap_x_closeloc"] = gap * out["xs_close_loc"] * 2
        out["gap_fill"] = -gap * (gap.abs() > 0.03)
    if "xs_r5" in out and "xs_r60" in out:
        out["combo_rev_mom"] = -out["xs_r5"] * out["xs_r60"]
    ess = _f(d, "ear_sign_recent")
    if ess is not None and "xs_dist_52wh" in out:
        out["combo_ear_range"] = ess * out["xs_dist_52wh"]
    n5 = _f(d, "news5")
    if n5 is not None and reg is not None:
        out["combo_news_reg"] = n5 * reg
    vix = _f(d, "m_vix")
    if vix is not None and r5 is not None:
        out["vix_x_r5"] = (vix - 20.0) * r5                        # fixed long-run VIX level, not a panel median (that reads later dates)
    if pattern_score is not None:
        out["miner"] = pattern_score.reindex(d.index).astype("float64")
    ext = pd.DataFrame(out, index=d.index)
    return pd.concat([d.drop(columns=[c for c in ext.columns if c in d.columns]), ext], axis=1)


LEAK_NAME_TOKENS = ("fwd", "future", "next", "label", "target", "outcome", "up_first", "up_sign", "realised", "realized")


def name_leak_reasons(cols: Sequence[str]) -> dict[str, str]:
    """Feature names that describe an outcome. Cheap, and it catches the copy-paste leaks that no statistic needs to find."""
    out = {}
    for c in cols:
        low = str(c).lower()
        hit = [t for t in LEAK_NAME_TOKENS if t in low]
        if hit:
            out[c] = f"name contains {hit[0]!r}"
    return out


def screen_feature_leakage(M: pd.DataFrame, up: np.ndarray, fwd: np.ndarray, max_auc: float = 0.80, min_rows: int = 200) -> pd.DataFrame:
    """Single-feature look-ahead screen. A point-in-time feature almost never separates next week's direction by more than a few
    AUC points, so a pooled AUC above `max_auc` (or below 1 - max_auc: an inverted leak) or a rank correlation with the forward
    return above 0.5 says the column was built from the outcome. One column per row, flag = reason string or ''."""
    rows = []
    ok_y = np.isfinite(up)
    for c in M.columns:
        x = M[c].to_numpy(dtype=float)
        m = ok_y & np.isfinite(x)
        n = int(m.sum())
        if n < min_rows or len(np.unique(up[m])) < 2 or np.ptp(x[m]) == 0:
            rows.append(dict(column=c, n=n, auc=float("nan"), rho_fwd=float("nan"), flag=""))
            continue
        a = _weekly_auc(x[m], up[m].astype(bool))
        mf = m & np.isfinite(fwd)
        rho = float(stats.spearmanr(x[mf], fwd[mf])[0]) if mf.sum() >= min_rows else float("nan")
        why = []
        if a > max_auc or a < 1 - max_auc:
            why.append(f"AUC {a:.3f} vs next-week direction")
        if np.isfinite(rho) and abs(rho) > 0.5:
            why.append(f"rank correlation {rho:.2f} with the forward return")
        rows.append(dict(column=c, n=n, auc=a, rho_fwd=rho, flag="; ".join(why)))
    return pd.DataFrame(rows, columns=["column", "n", "auc", "rho_fwd", "flag"])


@dataclasses.dataclass
class FamilyMatrix:
    """One hypothesis' feature block, row-aligned with the pool. `outcome` is set (and the matrix empty) when the family cannot be
    tested; dropped maps a candidate column to why it was excluded."""
    name: str
    columns: tuple[str, ...]
    matrix: np.ndarray
    dropped: dict[str, str]
    outcome: Outcome | None = None

    @property
    def testable(self) -> bool:
        return self.outcome is None and self.matrix.shape[1] > 0


def build_family_matrices(Xp: pd.DataFrame, pool: pd.DataFrame, cfg: LabConfig, specs: Mapping[str, HypothesisSpec] | None = None,
                          min_coverage: float = 0.05) -> tuple[dict[str, FamilyMatrix], pd.DataFrame]:
    """Feature blocks for every hypothesis plus 'combined' (union of every kept column, the existing-model stand-in). Columns that
    are mostly missing, constant, name an outcome or fail the leakage screen are dropped WITH a recorded reason; a family left with
    nothing is UNAVAILABLE_INPUT (or LEAK_SUSPECT when the screen removed all of it)."""
    specs = specs or HYPOTHESES
    up = pool["up"].to_numpy(dtype=float)
    fwd = pool["fwd"].to_numpy(dtype=float)
    candidates = sorted({c for s in specs.values() for c in s.columns if c in Xp.columns})
    screen = screen_feature_leakage(Xp[candidates], up, fwd, cfg.leak_auc) if candidates else pd.DataFrame(columns=["column", "n", "auc", "rho_fwd", "flag"])
    flagged = dict(zip(screen["column"], screen["flag"]))
    flagged = {k: v for k, v in flagged.items() if v}
    flagged.update(name_leak_reasons(candidates))
    fams: dict[str, FamilyMatrix] = {}
    kept_all: list[str] = []
    for name, spec in specs.items():
        got = resolve_columns(Xp.columns, spec)
        if not got:
            fams[name] = FamilyMatrix(name, (), np.empty((len(Xp), 0), np.float32), {c: "not in the panel" for c in spec.columns},
                                      Outcome.UNAVAILABLE_INPUT)
            continue
        keep, dropped = [], {c: "not in the panel" for c in spec.columns if c not in got}
        for c in got:
            x = Xp[c].to_numpy(dtype=float)
            if c in flagged:
                dropped[c] = "leak screen: " + flagged[c]
            elif np.isfinite(x).mean() < min_coverage:
                dropped[c] = f"only {np.isfinite(x).mean():.1%} non-missing"
            elif np.nanstd(x) == 0:
                dropped[c] = "constant"
            else:
                keep.append(c)
        if not keep:
            leak_only = dropped and all(v.startswith("leak") for k, v in dropped.items() if k in got)
            fams[name] = FamilyMatrix(name, (), np.empty((len(Xp), 0), np.float32), dropped,
                                      Outcome.LEAK_SUSPECT if leak_only else Outcome.UNAVAILABLE_INPUT)
            continue
        fams[name] = FamilyMatrix(name, tuple(keep), Xp[keep].to_numpy(dtype=np.float32), dropped)
        kept_all += keep
    union = tuple(dict.fromkeys(kept_all))
    fams["combined"] = FamilyMatrix("combined", union, Xp[list(union)].to_numpy(dtype=np.float32) if union else np.empty((len(Xp), 0), np.float32),
                                    {}, None if union else Outcome.UNAVAILABLE_INPUT)
    return fams, screen


def add_controls(fams: dict[str, FamilyMatrix], pool: pd.DataFrame, cfg: LabConfig, planted_share: float = 0.10, planted_acc: float = 0.90) -> dict[str, Any]:
    """Positive controls appended as pseudo-families: `planted` states the true direction with prob planted_acc on a random
    planted_share of rows (must be found at that coverage), `leak` states it exactly (must be caught by the leakage screen, and the
    lab must never report it as a hypothesis). Returns the ground truth needed to score them."""
    rng = np.random.default_rng(cfg.seed + 101)
    up = pool["up"].to_numpy(dtype=float)
    mask = rng.uniform(size=len(pool)) < planted_share
    pl = DF.plant_signal(up, mask, acc=planted_acc, seed=cfg.seed + 102)
    lk = np.where(np.nan_to_num(up, nan=0.5) > 0.5, 1.0, -1.0).astype("float32")
    fams["planted"] = FamilyMatrix("planted", ("planted",), pl[:, None], {})
    fams["leak"] = FamilyMatrix("leak", ("leak",), lk[:, None], {})
    return dict(mask=mask, planted_share=planted_share, planted_acc=planted_acc)


def control_leak_verdict(fams: Mapping[str, FamilyMatrix], pool: pd.DataFrame, cfg: LabConfig) -> dict[str, Any]:
    """Runs the leakage screen on the planted/leak controls themselves: the leak MUST be flagged and the (honest, 90%-on-10%) plant
    must NOT be, otherwise the screen is either blind or trigger-happy."""
    up, fwd = pool["up"].to_numpy(dtype=float), pool["fwd"].to_numpy(dtype=float)
    M = pd.DataFrame({k: fams[k].matrix[:, 0] for k in ("planted", "leak") if k in fams})
    sc = screen_feature_leakage(M, up, fwd, cfg.leak_auc)
    flag = dict(zip(sc["column"], sc["flag"]))
    return dict(leak_caught=bool(flag.get("leak")), plant_wrongly_flagged=bool(flag.get("planted")), screen=sc)


# ---------------------------------------------------------------------------------------------- competitors and the walk-forward
SIMPLE_RULES = (("r5", -1), ("r20", 1), ("mom_12_1", 1), ("dist_52wh", 1), ("close_loc", 1), ("ear_sign_recent", 1))


def _shuffled_week_hook(Fc: np.ndarray, date: np.ndarray, yv: np.ndarray, seed: int) -> Callable:
    """Control: a model fitted on labels shuffled WITHIN each decision week (the week's drift and the marginal are preserved, only
    the feature-label link is destroyed). It must read ~50% on the test block; if it does not the protocol leaks."""
    def fn(tr, ca, te):
        rng = np.random.default_rng(seed)
        y = yv[tr].copy()
        d = date[tr]
        order = np.argsort(d, kind="stable")
        ds = d[order]
        cuts = np.flatnonzero(ds[1:] != ds[:-1]) + 1
        for blk in np.split(np.arange(len(ds)), cuts):
            idx = order[blk]
            y[idx] = rng.permutation(y[idx])
        mdl = DF.DirModel("linear", seed).fit(Fc[tr], y)
        return mdl.prob(Fc[ca]), mdl.prob(Fc[te]), True
    return fn


def _base_hook(yv: np.ndarray) -> Callable:
    def fn(tr, ca, te):
        base = float(np.mean(yv[tr]))
        return np.full(len(ca), base), np.full(len(te), base), True
    return fn


def _nolearn_hook(M: np.ndarray, sign: int) -> Callable:
    """Fixed-sign rule with no fitted parameter: sign * the mean sign of the rule columns. Only the shared Platt map (two numbers,
    fitted on the calibration block like every other model) is applied afterwards."""
    S = sign * np.sign(np.nan_to_num(M.astype(float), nan=0.0)).mean(axis=1)

    def fn(tr, ca, te):
        return S[ca], S[te], False
    return fn


def simple_candidates(Xp: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray]:
    """The fixed one-feature rules of the literature (short-term reversal, momentum, 52-week high, close location, PEAD sign)."""
    names, cols = [], []
    for c, s in SIMPLE_RULES:
        if c in Xp.columns:
            names.append(c)
            cols.append(s * Xp[c].to_numpy(dtype=float))
    return tuple(names), (np.column_stack(cols) if cols else np.empty((len(Xp), 0)))


def _simple_baseline_hook(S: np.ndarray, yv: np.ndarray) -> Callable:
    """The strongest SIMPLE rival: among the fixed rules, take the one with the best calibration-block accuracy (one discrete
    choice, no training rows). The hypothesis has to beat what a person could write in one line."""
    def fn(tr, ca, te):
        best, acc_best = None, -1.0
        yc = yv[ca] > 0.5
        for j in range(S.shape[1]):
            x = S[ca, j]
            ok = np.isfinite(x)
            if ok.sum() < 30:
                continue
            med = np.nanmedian(x)
            acc = float(((x[ok] > med) == yc[ok]).mean())
            if acc > acc_best:
                best, acc_best = j, acc
        if best is None:
            return np.zeros(len(ca)), np.zeros(len(te)), False
        med = np.nanmedian(S[ca, best])
        return np.nan_to_num(S[ca, best] - med), np.nan_to_num(S[te, best] - med), False
    return fn


def _existing_hook(p: np.ndarray) -> Callable:
    """The production direction model's point-in-time P(up), scored as given (missing = 0.5), calibrated like every other model."""
    q = np.where(np.isfinite(p), np.clip(p, 1e-4, 1 - 1e-4), 0.5)

    def fn(tr, ca, te):
        return q[ca], q[te], True
    return fn


def make_extra_hooks(pool: pd.DataFrame, fams: Mapping[str, FamilyMatrix], Xp: pd.DataFrame, cfg: LabConfig, names: Sequence[str],
                     existing_p: pd.Series | None = None, specs: Mapping[str, HypothesisSpec] | None = None) -> dict[str, Callable]:
    """The `extra` models for direction_features.walk_forward: base rate, within-week-shuffled control, simple baseline, the existing
    production model when supplied, and one fixed-sign no-learning rule per signed hypothesis in `names`."""
    specs = specs or HYPOTHESES
    yv = pool["up"].to_numpy(dtype=float)
    date = pd.DatetimeIndex(pool["date"]).to_numpy()
    hooks: dict[str, Callable] = {TAG_BASE: _base_hook(yv)}
    Fc = fams["combined"].matrix
    if Fc.shape[1]:
        hooks[TAG_SHUF] = _shuffled_week_hook(Fc, date, yv, cfg.seed + 7)
    sn, S = simple_candidates(Xp)
    if S.shape[1]:
        hooks[TAG_SIMPLE] = _simple_baseline_hook(S, yv)
    if existing_p is not None:
        hooks[TAG_EXISTING] = _existing_hook(existing_p.reindex(pool.index).to_numpy(dtype=float))
    fam_r = fams.get("market_regime")
    if fam_r is not None and fam_r.testable and "market_regime" in names and len(np.unique(pool["reg"])) >= 2:
        hooks["regime_switch:market_regime"] = regime_switch_hook(Fc, pool["reg"].to_numpy(), yv, cfg.seed)
    for h in names:
        spec, fam = specs.get(h), fams.get(h)
        if spec is None or fam is None or not fam.testable or spec.prior_sign == 0:
            continue
        use = [fam.columns.index(c) for c in spec.rule_columns if c in fam.columns]
        if use:
            hooks[f"nolearn:{h}"] = _nolearn_hook(fam.matrix[:, use], spec.prior_sign)
    return hooks


@dataclasses.dataclass
class WalkForwardResult:
    pred: pd.DataFrame
    blocks: pd.DataFrame
    models: tuple[str, ...]
    stage: Stage
    tags: dict[str, str]


def run_walk_forward(pool: pd.DataFrame, fams: Mapping[str, FamilyMatrix], Xp: pd.DataFrame, cfg: LabConfig, names: Sequence[str],
                     kinds: Sequence[str], stage: Stage, existing_p: pd.Series | None = None,
                     extra_hooks: Mapping[str, Callable] | None = None) -> WalkForwardResult:
    """One pass of the shared protocol (direction_features.walk_forward: train before the embargo, calibrate on last year's picks,
    test on this year's picks, thresholds from the calibration block only) over the requested families + combined + the two
    controls and every comparator hook. 'blend' rows are dropped: the lab never lets an ensemble stand in for a hypothesis."""
    use = [n for n in names if fams[n].testable]
    F = {n: fams[n].matrix for n in use}
    F["combined"] = fams["combined"].matrix
    for ctl in ("planted", "leak"):
        if ctl in fams:
            F[ctl] = fams[ctl].matrix
    if F["combined"].shape[1] == 0 or not cfg.test_years:
        return WalkForwardResult(pd.DataFrame(columns=["row", "date", "year", "model", "p", "up", "q", "p0"]), pd.DataFrame(), tuple(), stage, {})
    hooks = dict(extra_hooks) if extra_hooks is not None else make_extra_hooks(pool, fams, Xp, cfg, use, existing_p)
    segs = ("seg", "reg") + (("sector",) if "sector" in pool and pool["sector"].nunique() > 1 else ())
    pred, blocks = DF.walk_forward(pool, F, "up", cfg.test_years, models=tuple(kinds), extra=hooks, seg_cols=segs,
                                   embargo_days=cfg.embargo_days, min_train=cfg.min_train, min_calib=cfg.min_calib, seed=cfg.seed)
    if len(pred):
        pred = pred[pred["model"] != "blend"].reset_index(drop=True)
    return WalkForwardResult(pred, blocks, tuple(sorted(pred["model"].unique())) if len(pred) else tuple(), stage,
                             {k: k for k in hooks})


def comparator_tags(hyp: str, kind: str, models: Sequence[str]) -> dict[Comparator, tuple[str, str]]:
    """Comparator -> (model tag, provenance). When a comparator has no dedicated model (unsigned hypothesis has no no-learning rule;
    no production model supplied) the fallback is named, so the report never implies a comparison that did not happen."""
    have = set(models)
    out: dict[Comparator, tuple[str, str]] = {Comparator.BASE_RATE: (TAG_BASE, "dedicated"), Comparator.RANDOM: (TAG_RANDOM, "dedicated"),
                                              Comparator.SHUFFLED: (TAG_SHUF, "dedicated")}
    out[Comparator.NO_LEARNING] = (f"nolearn:{hyp}", "dedicated") if f"nolearn:{hyp}" in have else (TAG_BASE, "fallback_base_rate:unsigned_prior")
    out[Comparator.SIMPLE_BASELINE] = (TAG_SIMPLE, "dedicated") if TAG_SIMPLE in have else (TAG_BASE, "fallback_base_rate:no_simple_inputs")
    if TAG_EXISTING in have:
        out[Comparator.EXISTING_MODEL] = (TAG_EXISTING, "dedicated")
    else:
        out[Comparator.EXISTING_MODEL] = (f"combined:{kind}", "fallback_combined_proxy")
    return {k: v for k, v in out.items() if v[0] in have}


# ---------------------------------------------------------------------------------------------- statistics
def wide(pred: pd.DataFrame, value: str = "p") -> pd.DataFrame:
    """Rows x models matrix of `value`; only rows scored by EVERY model are kept so all comparisons are paired on identical rows."""
    if pred.empty:
        return pd.DataFrame()
    w = pred.pivot(index="row", columns="model", values=value)
    return w.dropna(axis=0, how="any")


def row_frame(pred: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """date, year, up and the train base rate p0 of the given test rows (identical for every model)."""
    first = pred.drop_duplicates("row").set_index("row")
    return first.loc[rows, ["date", "year", "up", "p0"]]


def brier_skill(p: np.ndarray, y: np.ndarray, p0: np.ndarray) -> float:
    """1 - Brier(model) / Brier(train base rate): positive means the probabilities carry information the base rate does not."""
    b, b0 = float(np.mean((p - y) ** 2)), float(np.mean((p0 - y) ** 2))
    return 1.0 - b / b0 if b0 > 0 else float("nan")


def paired_test(pred: pd.DataFrame, a: str, b: str, n_boot: int = 600, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Is model `a` better than `b`? Per-row Brier difference (a - b; negative favours a), week-cluster bootstrap. `better` needs the
    whole one-sided (1-alpha) interval below zero. p_one is the normal-approximation one-sided p from the bootstrap standard error."""
    W = wide(pred[pred["model"].isin([a, b])])
    empty = dict(n=0, weeks=0, mean_diff=float("nan"), lo=float("nan"), hi=float("nan"), p_one=1.0, better=False, skill_a=float("nan"),
                 skill_b=float("nan"))
    if W.empty or a not in W or b not in W:
        return empty
    rf = row_frame(pred, W.index.to_numpy())
    y = rf["up"].to_numpy(float)
    diff = pd.Series((W[a].to_numpy() - y) ** 2 - (W[b].to_numpy() - y) ** 2,
                     index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(rf["date"]), W.index.to_numpy()], names=["date", "row"]))
    bs = DA.week_bootstrap(diff, n_boot, seed, alpha=2 * alpha)
    se = (bs["hi"] - bs["lo"]) / (2 * stats.norm.ppf(1 - alpha)) if np.isfinite(bs["hi"]) and bs["hi"] > bs["lo"] else float("nan")
    p_one = float(stats.norm.cdf(bs["mean"] / se)) if se and np.isfinite(se) else 1.0
    p0 = rf["p0"].to_numpy(float)
    return dict(n=len(diff), weeks=bs["n_weeks"], mean_diff=bs["mean"], lo=bs["lo"], hi=bs["hi"], p_one=p_one,
                better=bool(np.isfinite(bs["hi"]) and bs["hi"] < 0), skill_a=brier_skill(W[a].to_numpy(), y, p0),
                skill_b=brier_skill(W[b].to_numpy(), y, p0))


def _within_group_perms(y: np.ndarray, gid: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """count label vectors (N x count), each a permutation of y inside every group (decision week): the null keeps each week's
    drift and the marginal up-rate and destroys only the link between the prediction and the outcome."""
    base = np.argsort(gid, kind="stable")
    out = np.empty((len(y), count), np.float32)
    for j in range(count):
        order = np.lexsort((rng.random(len(y)), gid))
        col = np.empty(len(y), np.float32)
        col[base] = y[order]
        out[:, j] = col
    return out


def max_t_permutation(P: np.ndarray, y: np.ndarray, p0: np.ndarray, gid: np.ndarray, n_perm: int, seed: int = 0,
                      chunk: int = 50) -> dict[str, np.ndarray]:
    """Studentised max-T (Westfall-Young) test of Brier skill for M candidate cells at once. Observed skill of column m is compared
    with its distribution when the test labels are shuffled within week; p_raw is per cell, p_maxT compares each cell with the
    LARGEST standardised null skill across all cells in the same shuffle, so it prices having tried M ideas (the multiplicity the
    honest 80% question also has to pay). P is N x M calibrated probabilities on the shared test rows."""
    N, M = P.shape
    if N == 0 or M == 0:
        return dict(skill=np.zeros(M), p_raw=np.ones(M), p_maxT=np.ones(M), null_sd=np.ones(M), n_perm=0)
    rng = np.random.default_rng(seed)
    Pf, yf = P.astype(np.float32), y.astype(np.float32)
    m2 = (Pf.astype(np.float64) ** 2).mean(axis=0)
    p02 = float(np.mean(p0 ** 2))
    b_obs = ((Pf - yf[:, None]) ** 2).mean(axis=0)
    b0_obs = float(np.mean((p0 - yf) ** 2))
    skill = 1.0 - b_obs / b0_obs
    null = np.empty((M, n_perm))
    done = 0
    while done < n_perm:
        c = min(chunk, n_perm - done)
        L = _within_group_perms(yf, gid, c, rng)
        ybar = L.mean(axis=0)
        b = m2[:, None] - 2.0 * (Pf.T @ L) / N + ybar[None, :]
        b0 = p02 - 2.0 * (p0.astype(np.float32) @ L) / N + ybar
        null[:, done:done + c] = 1.0 - b / b0[None, :]
        done += c
    sd = null.std(axis=1)
    sd[~(sd > 0)] = 1.0
    t_obs, t_null = skill / sd, null / sd[:, None]
    p_raw = (1 + (null >= skill[:, None]).sum(axis=1)) / (n_perm + 1)
    mx = t_null.max(axis=0)
    p_max = (1 + (mx[None, :] >= t_obs[:, None]).sum(axis=1)) / (n_perm + 1)
    return dict(skill=skill, p_raw=p_raw, p_maxT=p_max, null_sd=sd, n_perm=n_perm)


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (strong family-wise control, uniformly better than Bonferroni)."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """BH adjusted p-values (false-discovery-rate control); reported beside Holm because discovery, not proof, is the point of a screen."""
    p = np.asarray(p, float)
    if p.size == 0:
        return p
    o = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    run = 1.0
    for k, i in enumerate(o):
        rank = len(p) - k
        run = min(run, p[i] * len(p) / rank)
        adj[i] = min(1.0, run)
    return adj


def week_design_effect(correct: np.ndarray, week: np.ndarray) -> float:
    """Kish design effect 1 + (m-1)*ICC for outcomes clustered in decision weeks (one-way ANOVA ICC, floored at 0). A pool of 30
    picks a week does not carry 30 independent observations: they share the market."""
    if len(correct) < 4:
        return 1.0
    df = pd.DataFrame({"c": correct.astype(float), "w": week})
    g = df.groupby("w")["c"]
    k, n = g.ngroups, len(df)
    if k < 2 or n <= k:
        return 1.0
    m = n / k
    grand = df["c"].mean()
    msb = (g.count() * (g.mean() - grand) ** 2).sum() / (k - 1)
    msw = ((df["c"] - g.transform("mean")) ** 2).sum() / (n - k)
    icc = max((msb - msw) / (msb + (m - 1) * msw), 0.0) if (msb + (m - 1) * msw) > 0 else 0.0
    return float(1.0 + (m - 1) * icc)


def power_report(n: int, weeks: int, base_rate: float, deff: float, alpha: float = 0.05, power: float = 0.8, target: float = EIGHTY) -> dict[str, float]:
    """What this sample could have detected. n_eff = n / design effect; mde_edge is the smallest accuracy edge over the base rate
    detectable at the stated power; n_bets_for_target is the number of independent bets whose Wilson 95% lower bound would clear
    `target` if the true accuracy were target+0.05 (a 'no 80% region' verdict below that count is silence, not evidence)."""
    n_eff = max(n / max(deff, 1.0), 1.0)
    mde = LC.minimum_detectable_gap(int(round(n_eff)), max(base_rate, 1 - base_rate), alpha, power) if n_eff >= 2 else float("inf")
    need = float("inf")
    acc = min(target + 0.05, 0.99)
    for m in range(10, 20001, 5):
        if LC.wilson(acc * m, m, z=stats.norm.ppf(0.975))[0] >= target:
            need = float(m)
            break
    return dict(n=n, weeks=weeks, design_effect=deff, n_eff=n_eff, mde_edge=float(mde), n_bets_for_target=need)


def coverage_bands(pred: pd.DataFrame, model: str, edges: Sequence[float] = (0.0, 0.01, 0.05, 0.10, 0.25, 1.0)) -> pd.DataFrame:
    """Accuracy in DISJOINT confidence bands. The nested frontier (top 5% inside top 25%) repeats rows, so its cells are not
    independent; the bands are, and a real signal must show accuracy falling as the band widens."""
    d = pred[pred["model"] == model]
    if d.empty:
        return pd.DataFrame(columns=["band_lo", "band_hi", "n", "acc", "wilson_lo", "wilson_hi", "weeks"])
    ok = ((d["p"] >= 0.5) == (d["up"] > 0.5)).to_numpy()
    q = d["q"].to_numpy()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q > lo + 1e-12) & (q <= hi + 1e-12) if lo > 0 else (q <= hi + 1e-12)
        n = int(m.sum())
        wl, wh = LC.wilson(float(ok[m].sum()), n, z=1.96) if n else (float("nan"), float("nan"))
        rows.append(dict(band_lo=lo, band_hi=hi, n=n, acc=float(ok[m].mean()) if n else float("nan"), wilson_lo=wl, wilson_hi=wh,
                         weeks=int(d.loc[m, "date"].nunique()) if n else 0))
    return pd.DataFrame(rows)


def confidence_monotonicity(pred: pd.DataFrame, model: str, n_boot: int = 300, seed: int = 0) -> dict[str, float]:
    """Does confidence rank correctness? Spearman(-q, correct) with a week-bootstrap interval. A model whose most confident calls are
    not its most accurate ones has no usable frontier no matter what its headline accuracy is."""
    d = pred[pred["model"] == model]
    if len(d) < 50:
        return dict(rho=float("nan"), lo=float("nan"), hi=float("nan"), n=len(d))
    ok = ((d["p"] >= 0.5) == (d["up"] > 0.5)).to_numpy().astype(float)
    conf = -d["q"].to_numpy()
    wk, uniq = pd.factorize(d["date"])
    rho = float(stats.spearmanr(conf, ok)[0]) if np.ptp(conf) > 0 and np.ptp(ok) > 0 else 0.0
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(wk == w) for w in range(len(uniq))]
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        if np.ptp(conf[pick]) > 0 and np.ptp(ok[pick]) > 0:
            boots.append(stats.spearmanr(conf[pick], ok[pick])[0])
    lo, hi = (np.quantile(boots, [0.05, 0.95]) if len(boots) > 20 else (float("nan"), float("nan")))
    return dict(rho=rho, lo=float(lo), hi=float(hi), n=len(d))


def era_stability(pred: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Brier skill and accuracy per test year, plus the Fama-MacBeth style summary: mean yearly skill, its t statistic across years,
    the share of years with positive skill and the longest run of negative years."""
    d = pred[pred["model"] == model]
    rows = []
    for y, g in d.groupby("year"):
        p, up, p0 = g["p"].to_numpy(), g["up"].to_numpy(), g["p0"].to_numpy()
        acc = float(((p >= 0.5) == (up > 0.5)).mean())
        base = float(max(up.mean(), 1 - up.mean()))
        rows.append(dict(year=int(y), n=len(g), skill=brier_skill(p, up, p0), acc=acc, majority_acc=base, acc_edge=acc - base))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(years=0, mean_skill=float("nan"), t_stat=float("nan"), share_positive=float("nan"), longest_negative_run=0)
    s = t["skill"].to_numpy()
    k = len(s)
    tstat = float(s.mean() / (s.std(ddof=1) / math.sqrt(k))) if k > 2 and s.std(ddof=1) > 0 else float("nan")
    run = best = 0
    for v in s:
        run = run + 1 if v <= 0 else 0
        best = max(best, run)
    return t, dict(years=k, mean_skill=float(s.mean()), t_stat=tstat, share_positive=float((s > 0).mean()), longest_negative_run=int(best))


def segment_stability(pred: pd.DataFrame, model: str, segcol: str, min_n: int = 100) -> tuple[pd.DataFrame, dict[str, float]]:
    """Skill inside each segment value (sector, market regime, event type). A signal that lives in one segment only is a
    conditional claim and is reported as such; one whose sign flips between segments is a regime artefact."""
    d = pred[pred["model"] == model]
    if d.empty or segcol not in d:
        return pd.DataFrame(), dict(segments=0, share_positive=float("nan"), sign_flips=0)
    rows = []
    for s, g in d.groupby(segcol):
        if len(g) < min_n:
            continue
        p, up, p0 = g["p"].to_numpy(), g["up"].to_numpy(), g["p0"].to_numpy()
        ok = (p >= 0.5) == (up > 0.5)
        lo, hi = LC.wilson(float(ok.sum()), len(g), z=1.96)
        rows.append(dict(segment=s, n=len(g), skill=brier_skill(p, up, p0), acc=float(ok.mean()), wilson_lo=lo, wilson_hi=hi,
                         base_acc=float(max(up.mean(), 1 - up.mean()))))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(segments=0, share_positive=float("nan"), sign_flips=0)
    pos = (t["skill"] > 0)
    return t, dict(segments=len(t), share_positive=float(pos.mean()), sign_flips=int(pos.any() and (~pos).any()))


def signal_decay(pred: pd.DataFrame, model: str) -> dict[str, Any]:
    """Has the edge weakened or broken? Weekly Brier gain over the base rate feeds the calibration module's change-point scan (a
    stated false-alarm level, not a hand-set CUSUM); the Spearman trend of yearly skill says whether the drift is monotone."""
    d = pred[pred["model"] == model]
    if len(d) < 200:
        return dict(break_week=None, stat=0.0, trend_rho=float("nan"), direction="unknown", weeks=0)
    d = d.assign(gain=(d["p0"] - d["up"]) ** 2 - (d["p"] - d["up"]) ** 2)
    wk = d.groupby("date")["gain"].mean().sort_index()
    stat, idx = LC.change_scan(wk.to_numpy())
    yr = d.groupby("year").apply(lambda g: brier_skill(g["p"].to_numpy(), g["up"].to_numpy(), g["p0"].to_numpy()))
    rho = float(stats.spearmanr(yr.index, yr.to_numpy())[0]) if len(yr) >= 4 and yr.nunique() > 1 else float("nan")
    direction = "unknown"
    if idx is not None:
        direction = "weakening" if wk.iloc[idx:].mean() < wk.iloc[:idx].mean() else "strengthening"
    return dict(break_week=str(wk.index[idx].date()) if idx is not None else None, stat=float(stat), trend_rho=rho, direction=direction,
                weeks=len(wk))


def payoff_table(pred: pd.DataFrame, pool: pd.DataFrame, model: str, coverages: Sequence[float], cost_bps: float, n_boot: int = 300,
                 seed: int = 0) -> pd.DataFrame:
    """What the calls earn, not just how often they are right: signed forward return net of a round-trip cost, its week-bootstrap
    interval, the mean win, mean loss and worst loss. Direction accuracy is only a means to a payoff (section 43: do not
    optimise the wrong metric)."""
    d = pred[pred["model"] == model]
    cols = ["cov_target", "n", "hit", "mean_net_bp", "lo_bp", "hi_bp", "avg_win_bp", "avg_loss_bp", "worst_bp"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    fwd = pool["fwd"].to_numpy(dtype=float)[d["row"].to_numpy()]
    side = np.where(d["p"].to_numpy() >= 0.5, 1.0, -1.0)
    net = (side * fwd - 2 * cost_bps / 1e4) * 1e4
    q = d["q"].to_numpy()
    wk, uniq = pd.factorize(d["date"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    rows = []
    for c in coverages:
        m = (q <= c + 1e-12) & np.isfinite(net)
        n = int(m.sum())
        if n == 0:
            continue
        S = np.bincount(wk[m], weights=net[m], minlength=len(uniq))
        N = np.bincount(wk[m], minlength=len(uniq)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = S[idx].sum(1) / N[idx].sum(1)
        bs = bs[np.isfinite(bs)]
        lo, hi = (np.quantile(bs, [0.05, 0.95]) if len(bs) > 20 else (float("nan"), float("nan")))
        x = net[m]
        rows.append(dict(cov_target=c, n=n, hit=float((x > 0).mean()), mean_net_bp=float(x.mean()), lo_bp=float(lo), hi_bp=float(hi),
                         avg_win_bp=float(x[x > 0].mean()) if (x > 0).any() else 0.0,
                         avg_loss_bp=float(x[x <= 0].mean()) if (x <= 0).any() else 0.0, worst_bp=float(x.min())))
    return pd.DataFrame(rows, columns=cols)


def ticker_transfer(pool: pd.DataFrame, M: np.ndarray, cfg: LabConfig, kind: str = "linear") -> dict[str, float]:
    """Memorisation probe. Tickers are split in two halves by a stable hash. A model is fitted on half A's past rows only and scored
    on the test-year picks of half A (names it has seen) and of half B (names it has never seen). A real, transferable signal has
    a gap near zero; a stock-identity memoriser shows skill on A and none on B. Reports the mean yearly gap and its t statistic."""
    if M.shape[1] == 0 or not cfg.test_years:
        return dict(years=0, skill_seen=float("nan"), skill_unseen=float("nan"), gap=float("nan"), gap_t=float("nan"))
    half = np.array([int(stable_hash(str(t), 8), 16) % 2 for t in pool["ticker"]])
    date = pd.DatetimeIndex(pool["date"]).to_numpy()
    yv, year, pick = pool["up"].to_numpy(float), pool["year"].to_numpy(), pool["pick"].to_numpy(bool)
    have = np.isfinite(yv)
    seen, unseen = [], []
    for Y in cfg.test_years:
        cut = (pd.Timestamp(year=int(Y), month=1, day=1) - pd.Timedelta(days=cfg.embargo_days + DF.LABEL_SPAN_DAYS)).to_datetime64()
        tr = np.flatnonzero(have & (half == 0) & (date < cut))
        if len(tr) < cfg.min_train or len(np.unique(yv[tr])) < 2:
            continue
        mdl = DF.DirModel(kind, cfg.seed).fit(M[tr], yv[tr])
        base = float(yv[tr].mean())
        for grp, sink in ((0, seen), (1, unseen)):
            te = np.flatnonzero(have & pick & (year == Y) & (half == grp))
            if len(te) >= 50:
                sink.append(brier_skill(mdl.prob(M[te]), yv[te], np.full(len(te), base)))
    k = min(len(seen), len(unseen))
    if k == 0:
        return dict(years=0, skill_seen=float("nan"), skill_unseen=float("nan"), gap=float("nan"), gap_t=float("nan"))
    a, b = np.asarray(seen[:k]), np.asarray(unseen[:k])
    g = a - b
    t = float(g.mean() / (g.std(ddof=1) / math.sqrt(k))) if k > 2 and g.std(ddof=1) > 0 else float("nan")
    return dict(years=k, skill_seen=float(a.mean()), skill_unseen=float(b.mean()), gap=float(g.mean()), gap_t=t)


# ---------------------------------------------------------------------------------------------- per-hypothesis assessment
@dataclasses.dataclass
class ComparatorResult:
    """One head-to-head: the hypothesis' calibrated probabilities against one comparator on identical rows."""
    comparator: Comparator
    against: str
    source: str
    n: int
    mean_diff: float
    lo: float
    hi: float
    p_one: float
    passed: bool


@dataclasses.dataclass
class HypothesisResult:
    """Everything the lab knows about one hypothesis after the staged tests. `reasons` is the human-readable trail behind the
    outcome; an empty trail on a NO_RELIABLE_SIGNAL is impossible by construction (decide_outcome always names what failed)."""
    name: str
    topic: str
    outcome: Outcome
    state: ResearchState
    stage: Stage
    best_tag: str = ""
    n_columns: int = 0
    n: int = 0
    weeks: int = 0
    skill: float = float("nan")
    acc: float = float("nan")
    acc_edge: float = float("nan")
    p_raw: float = 1.0
    p_maxT: float = 1.0
    p_holm: float = 1.0
    p_bh: float = 1.0
    comparators: dict[str, ComparatorResult] = dataclasses.field(default_factory=dict)
    era: dict[str, float] = dataclasses.field(default_factory=dict)
    regime: dict[str, float] = dataclasses.field(default_factory=dict)
    sector: dict[str, float] = dataclasses.field(default_factory=dict)
    event: dict[str, float] = dataclasses.field(default_factory=dict)
    transfer: dict[str, float] = dataclasses.field(default_factory=dict)
    decay: dict[str, Any] = dataclasses.field(default_factory=dict)
    monotonic: dict[str, float] = dataclasses.field(default_factory=dict)
    payoff_full: dict[str, float] = dataclasses.field(default_factory=dict)
    learn_curve: dict[str, float] = dataclasses.field(default_factory=dict)
    downside: dict[str, float] = dataclasses.field(default_factory=dict)
    portfolio: dict[str, Any] = dataclasses.field(default_factory=dict)
    calibration: dict[str, Any] = dataclasses.field(default_factory=dict)
    mechanism: dict[str, float] = dataclasses.field(default_factory=dict)
    posterior_skill: float = float("nan")
    reasons: list[str] = dataclasses.field(default_factory=list)
    dropped: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def comparators_passed(self) -> int:
        return sum(1 for c in self.comparators.values() if c.passed)

    def brief(self) -> dict[str, Any]:
        return dict(hypothesis=self.name, outcome=str(self.outcome), state=str(self.state), stage=str(self.stage), skill=self.skill,
                    acc=self.acc, p_raw=self.p_raw, p_maxT=self.p_maxT, comparators=f"{self.comparators_passed}/{len(ALL_COMPARATORS)}",
                    n=self.n, best=self.best_tag)


def compare_all(pred: pd.DataFrame, hyp: str, tag: str, kind: str, cfg: LabConfig, seed: int) -> dict[Comparator, ComparatorResult]:
    """The six section-10 comparators, each a paired week-cluster test of Brier score on identical rows. A comparator with no
    dedicated model uses a named fallback (see comparator_tags); one that cannot be scored at all is simply absent, and
    decide_outcome refuses to call a hypothesis a CANDIDATE unless all six are present and passed."""
    out: dict[Comparator, ComparatorResult] = {}
    models = tuple(pred["model"].unique())
    for comp, (other, src) in comparator_tags(hyp, kind, models).items():
        t = paired_test(pred, tag, other, cfg.n_boot, cfg.alpha, seed)
        out[comp] = ComparatorResult(comp, other, src, t["n"], t["mean_diff"], t["lo"], t["hi"], t["p_one"],
                                     bool(t["better"]))
    return out


def decide_outcome(res: HypothesisResult, cfg: LabConfig, n_dedicated: int) -> tuple[Outcome, list[str]]:
    """Fail-closed verdict. CANDIDATE needs EVERY gate; the first failure sets the reasons. NO_RELIABLE_SIGNAL is the default: a
    hypothesis is guilty of having no signal until it clears the family-wise test, all six comparators, era stability, transfer to
    unseen tickers and a positive net payoff. WEAK_UNREPLICATED marks an unadjusted or partial win that deserves a replication run."""
    why: list[str] = []
    if res.n < cfg.min_test_rows or res.weeks < cfg.min_weeks:
        return Outcome.INSUFFICIENT_DATA, [f"{res.n} test rows / {res.weeks} weeks < {cfg.min_test_rows} / {cfg.min_weeks}"]
    if res.era.get("years", 0) < cfg.min_eras:
        why.append(f"only {int(res.era.get('years', 0))} test years < {cfg.min_eras}")
    if not res.p_maxT < cfg.alpha:
        why.append(f"family-wise p {res.p_maxT:.3f} >= {cfg.alpha} (raw p {res.p_raw:.3f}, {len(ALL_COMPARATORS)} comparators, "
                   f"{n_dedicated} cells tried)")
    if not res.skill >= cfg.edge_floor:
        why.append(f"Brier skill {res.skill:+.4f} below the floor {cfg.edge_floor}")
    if np.isfinite(res.posterior_skill) and res.posterior_skill <= 0:
        why.append("empirical-Bayes shrinkage across all hypotheses tried leaves no edge (winner's curse)")
    missing = [c.value for c in ALL_COMPARATORS if c not in res.comparators]
    if missing:
        why.append("comparators unavailable: " + ",".join(missing))
    failed = [c.value for c, r in res.comparators.items() if not r.passed]
    if failed:
        why.append("did not beat: " + ",".join(failed))
    if res.era.get("years", 0) >= cfg.min_eras and res.era.get("share_positive", 0) < cfg.era_share:
        why.append(f"positive in {res.era.get('share_positive', 0):.0%} of years < {cfg.era_share:.0%}")
    t = res.era.get("t_stat", float("nan"))
    if np.isfinite(t) and t < 1.645:
        why.append(f"yearly skill t = {t:.2f} < 1.645")
    tr = res.transfer
    if tr.get("years", 0) and (tr.get("skill_unseen", 0) <= 0 or (np.isfinite(tr.get("gap_t", np.nan)) and tr["gap_t"] > 2.0)):
        why.append(f"does not transfer to unseen tickers (seen {tr.get('skill_seen', float('nan')):+.4f}, unseen {tr.get('skill_unseen', float('nan')):+.4f})")
    if res.payoff_full and not res.payoff_full.get("mean_net_bp", -1.0) > 0:
        why.append(f"net payoff {res.payoff_full.get('mean_net_bp', float('nan')):+.1f}bp <= 0 after costs")
    if not why:
        return Outcome.CANDIDATE, ["passed every gate; still needs a fresh holdout (Stage 4) before it may influence a decision"]
    if res.p_raw < cfg.alpha and res.skill >= cfg.edge_floor:
        return Outcome.WEAK_UNREPLICATED, why
    return Outcome.NO_RELIABLE_SIGNAL, why


def state_for(outcome: Outcome, powered: bool) -> ResearchState:
    """Where a verdict leaves the hypothesis in the research queue. 'No signal' is FAILED only when the sample was powerful enough
    to have seen a modest edge; otherwise the branch is DORMANT, waiting for data."""
    if outcome is Outcome.CANDIDATE:
        return ResearchState.VALIDATING
    if outcome is Outcome.WEAK_UNREPLICATED:
        return ResearchState.REPLICATING
    if outcome is Outcome.NO_RELIABLE_SIGNAL:
        return ResearchState.FAILED if powered else ResearchState.DORMANT
    if outcome is Outcome.UNAVAILABLE_INPUT:
        return ResearchState.QUEUED
    if outcome in (Outcome.LEAK_SUSPECT, Outcome.CONTROL_FAILURE):
        return ResearchState.CANCELLED
    return ResearchState.DORMANT


def stress_test(res: HypothesisResult, pred: pd.DataFrame, pool: pd.DataFrame, M: np.ndarray, tag: str, kind: str, cfg: LabConfig) -> None:
    """Fills the stability / transfer / decay / payoff fields of `res` for the best cell (run only for stage-2 survivors, because
    these tests are the expensive ones)."""
    _, res.era = era_stability(pred, tag)
    _, res.regime = segment_stability(pred, tag, "reg")
    _, res.event = segment_stability(pred, tag, "seg")
    if "sector" in pred:
        _, res.sector = segment_stability(pred, tag, "sector")
    res.transfer = ticker_transfer(pool, M, cfg, kind)
    res.decay = signal_decay(pred, tag)
    res.monotonic = confidence_monotonicity(pred, tag, min(cfg.n_boot, 300), cfg.seed)
    pay = payoff_table(pred, pool, tag, cfg.coverages, cfg.cost_bps, min(cfg.n_boot, 300), cfg.seed)
    if len(pay):
        full = pay[pay["cov_target"] == pay["cov_target"].max()].iloc[0]
        res.payoff_full = {k: float(full[k]) for k in pay.columns}


# ---------------------------------------------------------------------------------------------- controls
@dataclasses.dataclass
class ControlReport:
    """The protocol's own health. Any False here voids the run: a lab that cannot find a planted signal, cannot catch a leak or
    finds skill in shuffled labels has no standing to say anything about a real hypothesis (in either direction)."""
    planted_found: bool
    planted_acc: float
    planted_lo: float
    leak_caught: bool
    plant_wrongly_flagged: bool
    shuffled_null: bool
    random_null: bool
    audit_ok: bool
    audit_violations: list[str]
    notes: list[str]

    @property
    def ok(self) -> bool:
        return (self.planted_found and self.leak_caught and not self.plant_wrongly_flagged and self.shuffled_null and self.random_null
                and self.audit_ok)

    def failures(self) -> list[str]:
        f = []
        if not self.planted_found:
            f.append("planted signal not found")
        if not self.leak_caught:
            f.append("look-ahead canary not caught by the leakage screen")
        if self.plant_wrongly_flagged:
            f.append("honest planted feature was flagged as a leak")
        if not self.shuffled_null:
            f.append("shuffled-label control shows skill")
        if not self.random_null:
            f.append("random control shows skill")
        if not self.audit_ok:
            f.append("point-in-time audit failed: " + "; ".join(self.audit_violations[:3]))
        return f


def evaluate_controls(pred: pd.DataFrame, blocks: pd.DataFrame, leak: Mapping[str, Any], cfg: LabConfig, planted_share: float = 0.10,
                      strict_alpha: float | None = None) -> ControlReport:
    """Scores the planted / shuffled / random / leak controls of one walk-forward pass. Null controls are judged at a stricter alpha
    (alpha/5) so honest sampling noise does not void one run in twenty."""
    a = strict_alpha or cfg.alpha / 5
    notes: list[str] = []
    planted = "planted:linear" if "planted:linear" in set(pred["model"]) else next((m for m in pred["model"].unique() if m.startswith("planted")), None)
    acc = lo = float("nan")
    found = False
    if planted is not None:
        fr = DF.frontier(pred, planted, coverages=(min(0.25, max(planted_share * 1.5, 0.05)),), B=min(cfg.n_boot, 400), seed=cfg.seed)
        if len(fr) and fr["n"].iloc[0] > 0:
            acc, lo = float(fr["acc"].iloc[0]), float(fr["lo"].iloc[0])
            found = bool(np.isfinite(lo) and lo >= 0.70)
    nulls = {}
    for key, tag in (("shuffled", TAG_SHUF), ("random", TAG_RANDOM)):
        if tag not in set(pred["model"]) or TAG_BASE not in set(pred["model"]):
            nulls[key] = True
            notes.append(f"{tag} not run")
            continue
        t = paired_test(pred, tag, TAG_BASE, cfg.n_boot, a, cfg.seed)
        nulls[key] = not (t["better"] and t["skill_a"] > 0)
        if not nulls[key]:
            notes.append(f"{tag} beat the base rate (skill {t['skill_a']:+.4f})")
    audit = DF.audit_point_in_time(blocks, embargo_days=cfg.embargo_days) if len(blocks) and "fitted" in blocks else dict(ok=True, violations=[])
    return ControlReport(found, acc, lo, bool(leak.get("leak_caught")), bool(leak.get("plant_wrongly_flagged")), nulls["shuffled"],
                         nulls["random"], bool(audit["ok"]), list(audit["violations"]), notes)


def honest_statement(eighty: Mapping[str, Any], power: Mapping[str, float], gate: float = EIGHTY) -> str:
    """The sentence the contract demands when the answer is negative, qualified by whether the sample had the power to say so."""
    if eighty.get("reached_lo_adj", 0) > 0 and eighty.get("control_hits", 0) == 0:
        return (f"{eighty['reached_lo_adj']} cell(s) reach {gate:.0%} with the multiplicity-adjusted lower bound; treat as a lead for "
                f"fresh-holdout replication, not as a result.")
    if eighty.get("control_hits", 0) > 0:
        return f"Control cells reached {gate:.0%}: the protocol is broken, no directional claim of any kind can be made."
    base = f"No reliable {gate:.0%} directional region has been found."
    need = power.get("n_bets_for_target", float("inf"))
    n_eff = power.get("n_eff", 0.0)
    if np.isfinite(need) and n_eff < need:
        return base + f" The sample ({n_eff:.0f} effective bets) is smaller than the {need:.0f} needed to confirm {gate:.0%} even if it existed: absence of proof."
    return base + f" The sample ({n_eff:.0f} effective bets) was large enough to have confirmed it."


# ---------------------------------------------------------------------------------------------- single-feature diagnostics
def _date_codes(dates: Sequence) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(pd.DatetimeIndex(dates))
    return codes, len(uniq)


def feature_ic_table(M: pd.DataFrame, pool: pd.DataFrame, target: str = "up", min_rows: int = 20, family_of: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Weekly cross-sectional information coefficient of every raw or derived column with the outcome, among the week's predicted
    movers. IC_w is the Spearman correlation of the column with `target` inside week w (mean-centred ranks, vectorised); the
    table reports the mean IC, its t statistic across weeks, hit rate, information ratio, the two half-sample means (a real effect
    keeps its sign) and Benjamini-Hochberg adjusted p over ALL columns tried. This is the cheapest test in the lab and the one
    every later complexity has to beat."""
    cols = ["column", "family", "weeks", "mean_ic", "sd_ic", "t", "hit_rate", "ir", "ic_first", "ic_second", "sign_stable", "p_two",
            "p_bh"]
    y = pool[target].to_numpy(dtype=float)
    dc, nd = _date_codes(pool["date"])
    rows = []
    for c in M.columns:
        x = M[c].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5 * min_rows or nd < 4:
            continue
        g = dc[m]
        cnt = np.bincount(g, minlength=nd)
        xr = pd.Series(x[m]).groupby(g).rank(pct=True).to_numpy()
        yr = pd.Series(y[m]).groupby(g).rank(pct=True).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            xc = xr - (np.bincount(g, weights=xr, minlength=nd) / cnt)[g]
            yc = yr - (np.bincount(g, weights=yr, minlength=nd) / cnt)[g]
            sxy = np.bincount(g, weights=xc * yc, minlength=nd)
            sxx = np.bincount(g, weights=xc * xc, minlength=nd)
            syy = np.bincount(g, weights=yc * yc, minlength=nd)
            ic = sxy / np.sqrt(sxx * syy)
        ic = ic[(cnt >= min_rows) & np.isfinite(ic)]
        if len(ic) < 8:
            continue
        mid = len(ic) // 2
        mean, sd = float(ic.mean()), float(ic.std(ddof=1))
        t = mean / (sd / math.sqrt(len(ic))) if sd > 0 else 0.0
        p = float(2 * stats.t.sf(abs(t), len(ic) - 1))
        a, b = float(ic[:mid].mean()), float(ic[mid:].mean())
        rows.append(dict(column=c, family=(family_of or {}).get(c, ""), weeks=len(ic), mean_ic=mean, sd_ic=sd, t=t,
                         hit_rate=float((np.sign(ic) == np.sign(mean)).mean()), ir=mean / sd if sd > 0 else 0.0, ic_first=a, ic_second=b,
                         sign_stable=bool(np.sign(a) == np.sign(b) == np.sign(mean)), p_two=p, p_bh=1.0))
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out["p_bh"] = benjamini_hochberg(out["p_two"].to_numpy())
    return out.sort_values("p_two").reset_index(drop=True) if len(out) else out


def quantile_spread(x: np.ndarray, up: np.ndarray, dates: Sequence, n_bins: int = 5, n_boot: int = 400, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Up-rate by within-week quantile of a feature and the top-minus-bottom spread with a week-cluster bootstrap. A monotone rise
    from the bottom to the top bin is the classic signature of a real ranking signal; a spread made of one extreme bin is not."""
    x, up = np.asarray(x, float), np.asarray(up, float)
    m = np.isfinite(x) & np.isfinite(up)
    dc, nd = _date_codes(np.asarray(dates)[m])
    empty = (pd.DataFrame(columns=["bin", "n", "up_rate", "wilson_lo", "wilson_hi"]),
             dict(spread=float("nan"), lo=float("nan"), hi=float("nan"), rho=float("nan"), weeks=0))
    if m.sum() < n_bins * 30 or nd < 8:
        return empty
    r = pd.Series(x[m]).groupby(dc).rank(pct=True, method="first").to_numpy()
    b = np.minimum((r * n_bins).astype(int), n_bins - 1)
    yy = up[m]
    rows = []
    for k in range(n_bins):
        s = b == k
        n = int(s.sum())
        lo, hi = LC.wilson(float(yy[s].sum()), n, z=1.96) if n else (float("nan"), float("nan"))
        rows.append(dict(bin=k, n=n, up_rate=float(yy[s].mean()) if n else float("nan"), wilson_lo=lo, wilson_hi=hi))
    t = pd.DataFrame(rows)
    top, bot = b == n_bins - 1, b == 0
    Kt, Nt = np.bincount(dc[top], weights=yy[top], minlength=nd), np.bincount(dc[top], minlength=nd).astype(float)
    Kb, Nb = np.bincount(dc[bot], weights=yy[bot], minlength=nd), np.bincount(dc[bot], minlength=nd).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = Kt[idx].sum(1) / Nt[idx].sum(1) - Kb[idx].sum(1) / Nb[idx].sum(1)
    bs = bs[np.isfinite(bs)]
    spread = float(Kt.sum() / max(Nt.sum(), 1) - Kb.sum() / max(Nb.sum(), 1))
    lo, hi = (np.quantile(bs, [0.05, 0.95]) if len(bs) > 20 else (float("nan"), float("nan")))
    rho = float(stats.spearmanr(t["bin"], t["up_rate"])[0]) if t["up_rate"].nunique() > 1 else 0.0
    return t, dict(spread=spread, lo=float(lo), hi=float(hi), rho=rho, weeks=nd)


def null_max_accuracy(sizes: Sequence[int], p: float = 0.5, n_sim: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """How accurate the BEST of many cells looks by luck alone: for cells of the given sizes with true accuracy p, the `level`
    quantile of the maximum observed accuracy over all cells. An 'accuracy 0.82 in a cell of 40' is only interesting if it beats
    this number, which grows with the number of cells tried. Also returns the chance any cell reaches 0.80 by luck."""
    s = np.asarray([int(n) for n in sizes if n > 0])
    if s.size == 0:
        return dict(q_max=float("nan"), p_any_80=float("nan"), cells=0)
    rng = np.random.default_rng(seed)
    acc = rng.binomial(s[None, :], p, size=(n_sim, len(s))) / s[None, :]
    mx = acc.max(axis=1)
    return dict(q_max=float(np.quantile(mx, level)), p_any_80=float((mx >= 0.80).mean()), cells=int(len(s)))


def conditional_cells(M: pd.DataFrame, pool: pd.DataFrame, columns: Sequence[str], conditions: Sequence[str] = ("all", "reg", "seg"),
                      n_tier: int = 3, min_n: int = 40, n_sim: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exhaustive small-cell search, done honestly. A cell = (feature tercile) x (condition value). The side of each cell is set on
    the FIRST half of the dates and its accuracy is measured on the SECOND half only; every cell tried is counted, and the best
    out-of-sample cell is compared with the maximum accuracy that many cells produce by pure chance (null_max_accuracy). This is
    how 'an 80% pocket' is manufactured, and this function is what proves whether one is real."""
    d = pd.DatetimeIndex(pool["date"])
    if len(d) < 4:
        return pd.DataFrame(), dict(cells=0, best_acc=float("nan"), null_q95=float("nan"), beats_null=False, p_any_80=float("nan"), reaching_80=0)
    cut = d[np.argsort(d.values)[len(d) // 2]]
    up = pool["up"].to_numpy(dtype=float)
    first, second = np.asarray(d < cut), np.asarray(d >= cut)
    cells = []
    for c in columns:
        x = M[c].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(up)
        if ok.sum() < 6 * min_n:
            continue
        edges = np.nanquantile(x[ok & first], np.linspace(0, 1, n_tier + 1)[1:-1]) if (ok & first).sum() > n_tier else None
        if edges is None or len(np.unique(edges)) < len(edges):
            continue
        tier = np.digitize(x, edges)
        for cond in conditions:
            vals = ["all"] if cond == "all" else sorted(pool[cond].dropna().unique())
            for v in vals:
                cm = np.ones(len(pool), bool) if cond == "all" else (pool[cond].to_numpy() == v)
                for k in range(n_tier):
                    m = ok & cm & (tier == k)
                    a, b = m & first, m & second
                    if a.sum() < min_n or b.sum() < min_n:
                        continue
                    side = 1.0 if up[a].mean() >= 0.5 else 0.0
                    hit = float((up[b] == side).mean())
                    lo, hi = LC.wilson(float((up[b] == side).sum()), int(b.sum()), z=1.96)
                    cells.append(dict(column=c, condition=cond, value=str(v), tier=k, n_first=int(a.sum()), n_second=int(b.sum()), side=side,
                                      acc_second=hit, wilson_lo=lo, first_up=float(up[a].mean())))
    t = pd.DataFrame(cells)
    if t.empty:
        return t, dict(cells=0, best_acc=float("nan"), null_q95=float("nan"), beats_null=False, p_any_80=float("nan"), reaching_80=0)
    base = float(np.nanmean(up[second]))
    nm = null_max_accuracy(t["n_second"].tolist(), max(base, 1 - base), n_sim, seed)
    best = t.sort_values("acc_second", ascending=False).iloc[0]
    return t, dict(cells=len(t), best_acc=float(best["acc_second"]), best_n=int(best["n_second"]), best_cell=f"{best['column']}|{best['condition']}={best['value']}|t{best['tier']}",
                   null_q95=nm["q_max"], beats_null=bool(best["acc_second"] > nm["q_max"]), p_any_80=nm["p_any_80"],
                   reaching_80=int((t["acc_second"] >= EIGHTY).sum()), reaching_80_lo=int((t["wilson_lo"] >= EIGHTY).sum()))


def interaction_scan(M: pd.DataFrame, pool: pd.DataFrame, columns: Sequence[str], top_k: int = 5, min_rows: int = 500) -> tuple[pd.DataFrame, dict[str, float]]:
    """Do pairwise interactions replicate? On the first half of the dates every product of two z-scored columns is ranked by |IC|
    with the outcome; the top_k are then re-measured on the second half. Under no structure about top_k * 0.05 of them keep
    |t| > 1.96 and half keep their sign; a real interaction keeps both. The count of pairs searched is reported."""
    cols = [c for c in columns if c in M.columns]
    d = pd.DatetimeIndex(pool["date"])
    if len(cols) < 2 or len(d) < 2 * min_rows:
        return pd.DataFrame(columns=["a", "b", "t_first", "t_second", "sign_kept", "replicates"]), dict(pairs=0, selected=0, replicated=0, expected_by_chance=0.0)
    cut = d[np.argsort(d.values)[len(d) // 2]]
    first = np.asarray(d < cut)
    Z = M[cols].to_numpy(dtype=float)
    mu, sd = np.nanmean(Z[first], axis=0), np.nanstd(Z[first], axis=0)
    sd[~(sd > 0)] = 1.0
    Z = np.nan_to_num((np.clip((Z - mu) / sd, -4, 4)), nan=0.0)
    up = pool["up"].to_numpy(dtype=float)
    ok = np.isfinite(up)

    def t_of(v, m):
        m = m & ok
        if m.sum() < min_rows or np.ptp(v[m]) == 0:
            return 0.0, 0
        r = float(np.corrcoef(v[m], up[m])[0, 1])
        n = int(m.sum())
        return r * math.sqrt((n - 2) / max(1 - r * r, 1e-12)), n
    scored = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = Z[:, i] * Z[:, j]
            t1, _ = t_of(v, first)
            scored.append((abs(t1), t1, i, j))
    scored.sort(reverse=True)
    rows = []
    for _, t1, i, j in scored[:top_k]:
        t2, _ = t_of(Z[:, i] * Z[:, j], ~first)
        kept = bool(np.sign(t1) == np.sign(t2) and t2 != 0)
        rows.append(dict(a=cols[i], b=cols[j], t_first=t1, t_second=t2, sign_kept=kept, replicates=bool(kept and abs(t2) > 1.96)))
    t = pd.DataFrame(rows)
    return t, dict(pairs=len(scored), selected=len(t), replicated=int(t["replicates"].sum()) if len(t) else 0,
                   expected_by_chance=len(t) * 0.025)


def sample_size_curve(pool: pd.DataFrame, M: np.ndarray, cfg: LabConfig, kind: str = "linear",
                      fractions: Sequence[float] = (0.25, 0.5, 1.0)) -> tuple[pd.DataFrame, dict[str, float]]:
    """Is there anything to learn? Skill on each test year when the model sees only the most recent 25%/50%/100% of the available
    history. If skill does not rise with more data the feature set has no structure that experience can reveal (or it is
    non-stationary); a slope significantly above zero says accumulating data helps, which is the case for waiting."""
    if M.shape[1] == 0 or not cfg.test_years:
        return pd.DataFrame(columns=["fraction", "years", "mean_skill", "se"]), dict(slope=float("nan"), rho=float("nan"))
    date = pd.DatetimeIndex(pool["date"]).to_numpy()
    yv, year, pick = pool["up"].to_numpy(float), pool["year"].to_numpy(), pool["pick"].to_numpy(bool)
    have = np.isfinite(yv)
    res: dict[float, list[float]] = {f: [] for f in fractions}
    for Y in cfg.test_years:
        cut = (pd.Timestamp(year=int(Y), month=1, day=1) - pd.Timedelta(days=cfg.embargo_days + DF.LABEL_SPAN_DAYS)).to_datetime64()
        past = np.flatnonzero(have & (date < cut))
        te = np.flatnonzero(have & pick & (year == Y))
        if len(past) < cfg.min_train or len(te) < 50:
            continue
        past = past[np.argsort(date[past], kind="stable")]
        for f in fractions:
            tr = past[-max(int(len(past) * f), 1):]
            if len(np.unique(yv[tr])) < 2 or len(tr) < 200:
                continue
            mdl = DF.DirModel(kind, cfg.seed).fit(M[tr], yv[tr])
            res[f].append(brier_skill(mdl.prob(M[te]), yv[te], np.full(len(te), float(yv[tr].mean()))))
    rows = [dict(fraction=f, years=len(v), mean_skill=float(np.mean(v)), se=float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))
            for f, v in res.items() if v]
    t = pd.DataFrame(rows, columns=["fraction", "years", "mean_skill", "se"])
    if len(t) < 3:
        return t, dict(slope=float("nan"), rho=float("nan"))
    slope = float(np.polyfit(np.log(t["fraction"]), t["mean_skill"], 1)[0])
    return t, dict(slope=slope, rho=float(stats.spearmanr(t["fraction"], t["mean_skill"])[0]))


def winners_curse(skills: Sequence[float], null_sd: Sequence[float]) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of the observed skills. The standardised skills z_i = skill_i / null_sd_i have variance
    1 + tau^2 when true effects have variance tau^2; the shrinkage factor tau^2 / (1 + tau^2) is what the best-looking hypothesis
    should be discounted by (method of moments, floored at zero). With no real effect tau^2 = 0 and every posterior skill is 0, so
    the ranking that the raw numbers suggest is refused. Also gives the expected maximum |z| among M nulls, sqrt(2 ln M)."""
    s, sd = np.asarray(skills, float), np.asarray(null_sd, float)
    if s.size == 0:
        return pd.DataFrame(columns=["z", "posterior_skill", "shrink", "tau2", "null_max_z"])
    sd = np.where(sd > 0, sd, 1.0)
    z = s / sd
    tau2 = max(float(np.var(z, ddof=1)) - 1.0, 0.0) if len(z) > 1 else 0.0
    b = tau2 / (tau2 + 1.0)
    return pd.DataFrame({"z": z, "posterior_skill": b * z * sd, "shrink": b, "tau2": tau2,
                         "null_max_z": math.sqrt(2 * math.log(max(len(z), 2)))})


# ---------------------------------------------------------------------------------------------- risk of the calls
def downside_profile(pred: pd.DataFrame, pool: pd.DataFrame, model: str, coverage: float = 1.0, cap: float = -0.20, cost_bps: float = 7.0,
                     n_random: int = 200, seed: int = 0) -> dict[str, float]:
    """The risk half of 'direction probability -> risk assessment'. For the bets at `coverage`: net signed return quantiles, expected
    shortfall, the share of calls that lose more than the owner's cap (-20%), the loss when wrong versus the win when right, and the
    same statistics for RANDOM sides on the same rows. A directional model earns its keep only if it improves the tail, not just
    the hit rate; the random-side row is the no-skill floor for every one of these numbers."""
    d = pred[(pred["model"] == model) & (pred["q"] <= coverage + 1e-12)]
    if len(d) < 30:
        return dict(n=len(d))
    fwd = pool["fwd"].to_numpy(dtype=float)[d["row"].to_numpy()]
    ok = np.isfinite(fwd)
    fwd = fwd[ok]
    side = np.where(d["p"].to_numpy()[ok] >= 0.5, 1.0, -1.0)
    net = side * fwd - 2 * cost_bps / 1e4
    rng = np.random.default_rng(seed)
    rnd = np.array([(rng.choice([-1.0, 1.0], len(fwd)) * fwd - 2 * cost_bps / 1e4) for _ in range(n_random)])
    q05, q01 = np.quantile(net, 0.05), np.quantile(net, 0.01)
    wrong = net < 0
    out = dict(n=int(len(net)), mean=float(net.mean()), q05=float(q05), q01=float(q01), es05=float(net[net <= q05].mean()),
               cap_breach=float((net <= cap).mean()), loss_when_wrong=float(net[wrong].mean()) if wrong.any() else 0.0,
               win_when_right=float(net[~wrong].mean()) if (~wrong).any() else 0.0, worst=float(net.min()),
               random_mean=float(rnd.mean()), random_q05=float(np.quantile(rnd, 0.05)), random_cap_breach=float((rnd <= cap).mean()))
    out["tail_improvement"] = out["q05"] - out["random_q05"]
    out["breakeven_hit"] = (abs(out["loss_when_wrong"]) / (abs(out["loss_when_wrong"]) + max(out["win_when_right"], 1e-9))) if wrong.any() else 0.0
    return out


def rank_gradient(pool: pd.DataFrame, n_bins: int = 3, n_boot: int = 300, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Does the volatility model's own ranking carry a direction? Up-rate by within-week tercile of P(vol) among the pool, and the
    Spearman IC of P(vol) with the outcome. A significant gradient would mean the mover model already leaks a side and the
    direction stage must be judged AFTER removing it; a null gradient means the two problems really are separate."""
    d = pool.dropna(subset=["up", "score"])
    if len(d) < 60 * n_bins:
        return pd.DataFrame(), dict(ic=float("nan"), lo=float("nan"), hi=float("nan"))
    t, sp = quantile_spread(d["score"].to_numpy(), d["up"].to_numpy(), d["date"].to_numpy(), n_bins, n_boot, seed)
    ic = feature_ic_table(pd.DataFrame({"score": d["score"].to_numpy()}), d, "up")
    return t, dict(ic=float(ic["mean_ic"].iloc[0]) if len(ic) else float("nan"), t=float(ic["t"].iloc[0]) if len(ic) else float("nan"),
                   spread=sp["spread"], lo=sp["lo"], hi=sp["hi"])


def market_component(pool: pd.DataFrame) -> dict[str, float]:
    """How much of 'up vs down' is the market rather than the stock? ICC of the outcome inside decision weeks (share of variance
    explained by the week), the mean of the weekly up-rate and its dispersion. A high ICC says direction among movers is mostly
    market timing: a stock-picking model earns nothing from it, and a per-stock accuracy figure is inflated by the week's drift."""
    d = pool.dropna(subset=["up"])
    if len(d) < 100:
        return dict(icc=float("nan"), weeks=0, weekly_up_mean=float("nan"), weekly_up_sd=float("nan"), excess_dispersion=float("nan"))
    wk = d.groupby("date")["up"].agg(["mean", "count"])
    icc = max((week_design_effect(d["up"].to_numpy(), pd.factorize(d["date"])[0]) - 1) / max(wk["count"].mean() - 1, 1), 0.0)
    p = float(d["up"].mean())
    exp_sd = math.sqrt(p * (1 - p) / max(wk["count"].mean(), 1))
    return dict(icc=float(icc), weeks=len(wk), weekly_up_mean=float(wk["mean"].mean()), weekly_up_sd=float(wk["mean"].std(ddof=1)),
                excess_dispersion=float(wk["mean"].std(ddof=1) / exp_sd) if exp_sd > 0 else float("nan"))


def add_relative_target(pool: pd.DataFrame) -> pd.Series:
    """Market-neutral direction: 1 when the stock ended above the median forward return of the week's pool, else 0 (NaN where the
    return is missing). Balanced by construction, so it isolates stock selection from market timing. Research-side outcome only."""
    fwd = pool["fwd"]
    med = fwd.groupby(pool["date"]).transform("median")
    return pd.Series(np.where(fwd.notna(), (fwd > med).astype(float), np.nan), index=pool.index, name="up_rel")


# ---------------------------------------------------------------------------------------------- mechanism checks
@dataclasses.dataclass(frozen=True)
class MechanismCheck:
    """A prediction the hypothesis' stated mechanism makes about SHAPE, independent of headline skill. kind: ic_sign (the column's
    weekly IC has sign `expect`), ic_order (IC of cols[0] exceeds IC of cols[1] when expect=+1), dose (up-rate trends with the
    column, Cochran-Armitage), cond_ic (IC of cols[0] is higher in the 'hi' state of `cond` than the 'lo' state when expect=+1),
    spread (top-minus-bottom quintile up-rate has sign `expect`). A real effect should reproduce the shape the literature gives it;
    a coincidence usually does not, which is evidence the skill test alone cannot supply."""
    hypothesis: str
    kind: str
    cols: tuple[str, ...]
    expect: int
    text: str
    cond: str = ""


MECHANISM_CHECKS: tuple[MechanismCheck, ...] = (
    MechanismCheck("momentum", "ic_order", ("mom_12_1", "r5"), 1, "long-horizon momentum carries a higher IC than the 1-week move"),
    MechanismCheck("momentum", "ic_sign", ("r60",), 1, "intermediate momentum continues"),
    MechanismCheck("reversal", "ic_sign", ("r5",), -1, "last week's move partly reverses"),
    MechanismCheck("reversal", "ic_order", ("r5_news", "r5_nonews"), 1, "reversal is stronger without news (no-news moves reverse, news moves continue)"),
    MechanismCheck("event_reaction", "dose", ("event_net",), 1, "up-rate rises with the net signed filing content"),
    MechanismCheck("earnings_reaction", "ic_sign", ("ear_recent",), 1, "a recent positive earnings reaction drifts up"),
    MechanismCheck("earnings_reaction", "cond_ic", ("ear_recent",), -1, "the drift fades as the release ages", "release_age"),
    MechanismCheck("insider", "dose", ("ins_buyers30",), 1, "more insider buyers, higher up-rate"),
    MechanismCheck("range_52w", "spread", ("dist_52wh",), 1, "names nearer their 52-week high rise more than names far below it"),
    MechanismCheck("relative_strength", "spread", ("rel_ind20",), 1, "industry-relative winners keep winning"),
    MechanismCheck("sector_behavior", "ic_sign", ("sector_mom",), 1, "sector momentum leads its members"),
    MechanismCheck("market_regime", "cond_ic", ("r5",), 1, "short-term reversal is stronger in bear markets", "regime"),
    MechanismCheck("volume_structure", "cond_ic", ("r5",), 1, "volume-confirmed moves continue more than thin-volume moves", "volume"),
    MechanismCheck("intraday_path", "ic_sign", ("close_loc",), 1, "closing near the high carries into the next week"),
    MechanismCheck("gap_behavior", "cond_ic", ("gap_today",), 1, "gaps continue when the close is strong", "close_strength"),
    MechanismCheck("volatility_regime", "cond_ic", ("r5",), -1, "reversal strengthens when the VIX is high", "vix"),
    MechanismCheck("cross_sectional", "ic_sign", ("xs_r20",), 1, "rank among the week's movers carries the same sign as the level"),
)


@dataclasses.dataclass(frozen=True)
class MechanismResult:
    check: MechanismCheck
    status: str                      # consistent / inconsistent / inconclusive / untestable
    stat: float
    detail: str


def cochran_armitage(counts: np.ndarray, successes: np.ndarray, scores: np.ndarray | None = None, deff: float = 1.0) -> float:
    """Cochran-Armitage trend z for a monotone change of a proportion across ordered groups, divided by sqrt(design effect) so that
    rows clustered in weeks are not counted as independent. Positive z = the proportion rises with the score."""
    n, r = np.asarray(counts, float), np.asarray(successes, float)
    ok = n > 0
    n, r = n[ok], r[ok]
    if len(n) < 3:
        return 0.0
    s = np.arange(len(n), dtype=float) if scores is None else np.asarray(scores, float)[ok]
    N, R = n.sum(), r.sum()
    p = R / N
    if p <= 0 or p >= 1:
        return 0.0
    t = np.sum(s * (r - n * p))
    var = p * (1 - p) * (np.sum(n * s * s) - np.sum(n * s) ** 2 / N)
    return float(t / math.sqrt(var * max(deff, 1.0))) if var > 0 else 0.0


def _condition_labels(Xd: pd.DataFrame, pool: pd.DataFrame, cond: str) -> pd.Series | None:
    """'lo'/'hi' state of a conditioning variable at the decision close (never an outcome)."""
    def split(v: pd.Series, thr: float | None = None):
        t = float(v.median()) if thr is None else thr
        return pd.Series(np.where(v.isna(), None, np.where(v > t, "hi", "lo")), index=v.index)
    if cond == "regime":
        r = pool["reg"]
        return pd.Series(np.where(r == "bull", "hi", np.where(r == "bear", "lo", None)), index=pool.index) if (r != "na").any() else None
    if cond == "vix" and "m_vix" in Xd:
        return split(Xd["m_vix"].astype(float))
    if cond == "volume":
        c = "ear_volsurge" if "ear_volsurge" in Xd else "vol_surge" if "vol_surge" in Xd else None
        return split(Xd[c].astype(float)) if c else None
    if cond == "close_strength" and "close_loc" in Xd:
        return split(Xd["close_loc"].astype(float))
    if cond == "release_age" and "days_since_earn" in Xd:
        return split(Xd["days_since_earn"].astype(float), 20.0)                 # 'hi' = older than 20 sessions
    return None


def evaluate_mechanism(check: MechanismCheck, Xd: pd.DataFrame, pool: pd.DataFrame, z_crit: float = 1.96, seed: int = 0) -> MechanismResult:
    """Runs one check; status is 'inconclusive' unless |z| clears z_crit, so the mechanism scorecard cannot manufacture agreement."""
    if any(c not in Xd.columns for c in check.cols):
        return MechanismResult(check, "untestable", float("nan"), "missing " + ",".join(c for c in check.cols if c not in Xd.columns))
    Xd = Xd.reset_index(drop=True)
    pl = pool.reset_index(drop=True)

    def verdict(z: float, detail: str) -> MechanismResult:
        if not np.isfinite(z) or abs(z) < z_crit:
            return MechanismResult(check, "inconclusive", float(z), detail)
        return MechanismResult(check, "consistent" if np.sign(z) == check.expect else "inconsistent", float(z), detail)
    if check.kind == "ic_sign":
        t = feature_ic_table(Xd[[check.cols[0]]], pl, "up")
        return verdict(float(t["t"].iloc[0]), f"mean IC {t['mean_ic'].iloc[0]:+.4f}") if len(t) else MechanismResult(check, "untestable", float("nan"), "too few weeks")
    if check.kind == "ic_order":
        t = feature_ic_table(Xd[list(check.cols)], pl, "up").set_index("column")
        if len(t) < 2:
            return MechanismResult(check, "untestable", float("nan"), "too few weeks")
        a, b = t.loc[check.cols[0]], t.loc[check.cols[1]]
        se = math.sqrt(a["sd_ic"] ** 2 / a["weeks"] + b["sd_ic"] ** 2 / b["weeks"])
        return verdict(float((a["mean_ic"] - b["mean_ic"]) / se) if se > 0 else float("nan"), f"IC {a['mean_ic']:+.4f} vs {b['mean_ic']:+.4f}")
    if check.kind == "dose":
        x, up = Xd[check.cols[0]].to_numpy(float), pl["up"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(up)
        if m.sum() < 200 or len(np.unique(x[m])) < 3:
            return MechanismResult(check, "untestable", float("nan"), "no dose variation")
        edges = np.unique(x[m]) if len(np.unique(x[m])) <= 6 else np.unique(np.quantile(x[m], np.linspace(0, 1, 6)[1:-1]))
        b = np.digitize(x[m], edges) if len(np.unique(x[m])) > 6 else np.searchsorted(edges, x[m])
        k = int(b.max()) + 1
        n = np.bincount(b, minlength=k)
        r = np.bincount(b, weights=up[m], minlength=k)
        deff = week_design_effect(up[m], pd.factorize(pl["date"].to_numpy()[m])[0])
        return verdict(cochran_armitage(n, r, deff=deff), f"{k} dose levels, design effect {deff:.1f}")
    if check.kind == "spread":
        _, s = quantile_spread(Xd[check.cols[0]].to_numpy(float), pl["up"].to_numpy(float), pl["date"].to_numpy(), 5, 300, seed)
        if not np.isfinite(s["spread"]):
            return MechanismResult(check, "untestable", float("nan"), "too few rows")
        se = (s["hi"] - s["lo"]) / (2 * 1.645) if np.isfinite(s["hi"]) and s["hi"] > s["lo"] else float("nan")
        return verdict(s["spread"] / se if se and np.isfinite(se) else float("nan"), f"top-minus-bottom up-rate {s['spread']:+.4f}")
    lab = _condition_labels(Xd.set_index(pool.index), pool, check.cond)
    if lab is None:
        return MechanismResult(check, "untestable", float("nan"), f"condition {check.cond!r} unavailable")
    lab = lab.reset_index(drop=True)
    parts = {}
    for st in ("lo", "hi"):
        m = (lab == st).to_numpy()
        if m.sum() < 200:
            return MechanismResult(check, "untestable", float("nan"), "thin condition state")
        parts[st] = feature_ic_table(Xd.loc[m, [check.cols[0]]].reset_index(drop=True), pl.loc[m].reset_index(drop=True), "up")
    if not all(len(v) for v in parts.values()):
        return MechanismResult(check, "untestable", float("nan"), "too few weeks in a state")
    a, b = parts["hi"].iloc[0], parts["lo"].iloc[0]
    se = math.sqrt(a["sd_ic"] ** 2 / a["weeks"] + b["sd_ic"] ** 2 / b["weeks"])
    return verdict(float((a["mean_ic"] - b["mean_ic"]) / se) if se > 0 else float("nan"), f"IC hi {a['mean_ic']:+.4f} / lo {b['mean_ic']:+.4f}")


def mechanism_report(Xd: pd.DataFrame, pool: pd.DataFrame, checks: Sequence[MechanismCheck] = MECHANISM_CHECKS, seed: int = 0) -> dict[str, list[MechanismResult]]:
    """All mechanism checks grouped by hypothesis."""
    out: dict[str, list[MechanismResult]] = {}
    for ch in checks:
        out.setdefault(ch.hypothesis, []).append(evaluate_mechanism(ch, Xd, pool, seed=seed))
    return out


def mechanism_summary(results: Sequence[MechanismResult]) -> dict[str, float]:
    """Counts by status and the share of TESTABLE, conclusive checks that agree with the stated mechanism."""
    c = {s: sum(1 for r in results if r.status == s) for s in ("consistent", "inconsistent", "inconclusive", "untestable")}
    conclusive = c["consistent"] + c["inconsistent"]
    return dict(**c, agreement=float(c["consistent"] / conclusive) if conclusive else float("nan"))


# ---------------------------------------------------------------------------------------------- calibration and gates
def calibration_report(pred: pd.DataFrame, model: str, seed: int = 0, n_boot: int = 200) -> dict[str, Any]:
    """Is P(up) trustworthy, not just ranked? Murphy decomposition (reliability vs resolution), logistic recalibration slope with its
    interval (slope < 1 = overconfident), equal-mass ECE with bootstrap interval and its sampling floor, sharpness and AUC."""
    d = pred[pred["model"] == model]
    if len(d) < 100:
        return dict(n=len(d))
    p, y = np.clip(d["p"].to_numpy(float), 1e-4, 1 - 1e-4), d["up"].to_numpy(float)
    ece_pt, ece_lo, ece_hi = LC.bootstrap_ece_ci(p, y, np.random.default_rng(seed), n_boot)
    return dict(n=len(d), decomposition=LC.brier_decomposition(p, y), slope=LC.platt_slope(p, y), ece=ece_pt, ece_lo=ece_lo, ece_hi=ece_hi,
                ece_floor=D.ece_noise_floor(p), sharpness=LC.sharpness(p), auc=LC.discrimination(p, y), overconfidence=LC.overconfidence(p, y))


def gate_alternatives(pred: pd.DataFrame, model: str, gate: float = EIGHTY, min_rows: int = 400) -> pd.DataFrame:
    """The existing gate routes (raw, Platt, isotonic, Venn-Abers, conformal) applied to this model's test rows: the earlier half of
    the rows calibrates and the later half is judged, so no route sees its own evaluation rows. Empty when the sample is small."""
    d = pred[pred["model"] == model].sort_values("date")
    if len(d) < min_rows:
        return pd.DataFrame()
    cut = len(d) // 2
    a, b = d.iloc[:cut], d.iloc[cut:]
    return DC.compare_gates(a["p"].to_numpy(), a["up"].to_numpy(), b["p"].to_numpy(), b["up"].to_numpy(), gate)


# ---------------------------------------------------------------------------------------------- what the calls earn, week by week
def weekly_portfolio(pred: pd.DataFrame, pool: pd.DataFrame, model: str, coverage: float = 1.0, cost_bps: float = 7.0, n_boot: int = 400,
                     seed: int = 0) -> dict[str, Any]:
    """Equal-weight book of the model's calls, one book per decision week (long the up calls, short the down calls, round-trip cost
    charged), compared with two no-skill books on the same names: always-long and random sides. Reports the mean and dispersion of
    weekly return, the share of positive weeks (the project's third objective), the worst week, the maximum drawdown of the summed
    weekly returns and a week-bootstrap interval on the gain over always-long."""
    d = pred[(pred["model"] == model) & (pred["q"] <= coverage + 1e-12)]
    if len(d) < 60:
        return dict(weeks=0)
    fwd = pool["fwd"].to_numpy(float)[d["row"].to_numpy()]
    ok = np.isfinite(fwd)
    d, fwd = d[ok], fwd[ok]
    cost = 2 * cost_bps / 1e4
    side = np.where(d["p"].to_numpy() >= 0.5, 1.0, -1.0)
    rng = np.random.default_rng(seed)
    books = {"model": side * fwd - cost, "always_long": fwd - cost, "random": rng.choice([-1.0, 1.0], len(fwd)) * fwd - cost}
    wk = pd.factorize(d["date"].to_numpy(), sort=True)[0]
    nw = wk.max() + 1
    W = {k: np.bincount(wk, weights=v, minlength=nw) / np.bincount(wk, minlength=nw) for k, v in books.items()}
    out: dict[str, Any] = dict(weeks=int(nw), coverage=coverage)
    for k, w in W.items():
        cum = np.cumsum(w)
        out[k] = dict(mean=float(w.mean()), sd=float(w.std(ddof=1)) if nw > 1 else float("nan"), share_positive=float((w > 0).mean()),
                      worst=float(w.min()), max_drawdown=float((np.maximum.accumulate(cum) - cum).max()),
                      in_band=float(((np.abs(w) >= 0.05) & (np.abs(w) <= 0.10)).mean()),
                      sharpe=float(w.mean() / w.std(ddof=1) * math.sqrt(52)) if nw > 2 and w.std(ddof=1) > 0 else float("nan"))
    gain = W["model"] - W["always_long"]
    idx = rng.integers(0, nw, size=(n_boot, nw))
    bs = gain[idx].mean(axis=1)
    out["gain_over_long"] = dict(mean=float(gain.mean()), lo=float(np.quantile(bs, 0.05)), hi=float(np.quantile(bs, 0.95)))
    return out


def abstention_value(pred: pd.DataFrame, pool: pd.DataFrame, model: str, n_bins: int = 5, cost_bps: float = 7.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Should the lab ever abstain? Mean net return of the calls in confidence bins (bin 0 = the most confident fifth). If abstaining
    on the least confident calls does not raise the average, the confidence measure carries no information about payoff, whatever
    its calibration plots look like. Reports the Spearman correlation between confidence rank and net return."""
    d = pred[pred["model"] == model]
    cols = ["bin", "n", "hit", "mean_net_bp", "mean_conf"]
    if len(d) < n_bins * 30:
        return pd.DataFrame(columns=cols), dict(rho=float("nan"), gain_top_vs_all_bp=float("nan"))
    fwd = pool["fwd"].to_numpy(float)[d["row"].to_numpy()]
    ok = np.isfinite(fwd)
    d, fwd = d[ok], fwd[ok]
    net = (np.where(d["p"].to_numpy() >= 0.5, 1.0, -1.0) * fwd - 2 * cost_bps / 1e4) * 1e4
    conf = np.abs(d["p"].to_numpy() - 0.5)
    b = np.minimum(((1 - d["q"].to_numpy()) * n_bins).astype(int), n_bins - 1)
    b = n_bins - 1 - b
    rows = [dict(bin=k, n=int((b == k).sum()), hit=float((net[b == k] > 0).mean()), mean_net_bp=float(net[b == k].mean()),
                 mean_conf=float(conf[b == k].mean())) for k in range(n_bins) if (b == k).any()]
    rho = float(stats.spearmanr(conf, net)[0]) if np.ptp(conf) > 0 else 0.0
    t = pd.DataFrame(rows, columns=cols)
    return t, dict(rho=rho, gain_top_vs_all_bp=float(t["mean_net_bp"].iloc[0] - net.mean()) if len(t) else float("nan"))


def wrong_call_anatomy(pred: pd.DataFrame, pool: pd.DataFrame, model: str, min_n: int = 50) -> tuple[pd.DataFrame, dict[str, float]]:
    """Study the losers (section 5 applied to direction): where are the wrong calls concentrated? Wrong-call rate by event type,
    market regime, sector, move size (terciles of |return|) and by the week's market direction, with an excess-over-overall column
    and a Wilson interval. Also splits the wrong calls into 'the whole week went the other way' (market-driven) and
    'this stock went against its week' (stock-specific), because the remedies differ (regime awareness vs selection)."""
    d = pred[pred["model"] == model]
    if len(d) < 2 * min_n:
        return pd.DataFrame(columns=["factor", "level", "n", "wrong", "wrong_rate", "excess", "wilson_lo", "wilson_hi"]), dict(market_driven=float("nan"), stock_specific=float("nan"))
    rows_ix = d["row"].to_numpy()
    fwd = pool["fwd"].to_numpy(float)[rows_ix]
    wrong = ((d["p"].to_numpy() >= 0.5) != (d["up"].to_numpy() > 0.5))
    dates = pd.DatetimeIndex(d["date"])
    wk_med = pd.Series(fwd).groupby(dates.to_numpy()).transform("median").to_numpy()
    size = pd.Series(np.abs(fwd)).rank(pct=True).to_numpy()
    factors = {"event": d["seg"].to_numpy(), "regime": d["reg"].to_numpy(),
               "size": np.where(size > 2 / 3, "large", np.where(size > 1 / 3, "mid", "small")),
               "week": np.where(wk_med > 0, "market_up", "market_down")}
    if "sector" in d:
        factors["sector"] = d["sector"].to_numpy()
    overall = float(wrong.mean())
    rows = []
    for f, lv in factors.items():
        for v in pd.unique(lv):
            m = lv == v
            if m.sum() < min_n:
                continue
            lo, hi = LC.wilson(float(wrong[m].sum()), int(m.sum()), z=1.96)
            rows.append(dict(factor=f, level=str(v), n=int(m.sum()), wrong=int(wrong[m].sum()), wrong_rate=float(wrong[m].mean()),
                             excess=float(wrong[m].mean() - overall), wilson_lo=lo, wilson_hi=hi))
    side = np.where(d["p"].to_numpy() >= 0.5, 1.0, -1.0)
    market_against = np.sign(wk_med) == -side
    n_wrong = max(int(wrong.sum()), 1)
    return pd.DataFrame(rows), dict(market_driven=float((wrong & market_against).sum() / n_wrong), stock_specific=float((wrong & ~market_against).sum() / n_wrong),
                                    overall_wrong=overall)


def ic_by_year(M: pd.DataFrame, pool: pd.DataFrame, target: str = "up") -> pd.DataFrame:
    """Mean weekly IC per calendar year for each column plus the share of years whose sign agrees with the pooled sign: an IC that
    flips sign between years is a regime artefact however large its pooled t statistic is."""
    parts = []
    for y in sorted(pool["year"].unique()):
        m = (pool["year"] == y).to_numpy()
        t = feature_ic_table(M[m].reset_index(drop=True), pool[m].reset_index(drop=True), target, min_rows=15)
        if len(t):
            parts.append(t.set_index("column")["mean_ic"].rename(int(y)))
    if not parts:
        return pd.DataFrame()
    W = pd.concat(parts, axis=1)
    yrs = list(W.columns)
    pooled = np.sign(W[yrs].mean(axis=1))
    agree = W[yrs].apply(lambda col: np.sign(col) == pooled)
    W["sign_agreement"] = agree.where(W[yrs].notna()).sum(axis=1) / W[yrs].notna().sum(axis=1)
    return W.reset_index()


# ---------------------------------------------------------------------------------------------- the lab checks itself
def null_size_check(n_rows: int = 3000, n_models: int = 12, n_weeks: int = 60, n_perm: int = 150, trials: int = 120, seed: int = 0) -> dict[str, float]:
    """Size of the testing machinery under the null: random calibrated probabilities unrelated to the labels are pushed through
    max_t_permutation `trials` times. The share of trials with ANY cell below 0.05 (family-wise, p_maxT) must be about 0.05 or less,
    and the raw p-values must look uniform (KS). A test that rejects a true null far more often than alpha would make every 'lead'
    the lab reports meaningless."""
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_weeks), n_rows // n_weeks)
    n = len(gid)
    fw, raw = 0, []
    for t in range(trials):
        y = (rng.uniform(size=n) < 0.5).astype(float)
        P = np.clip(0.5 + rng.normal(0, 0.03, (n, n_models)), 0.02, 0.98)
        r = max_t_permutation(P, y, np.full(n, 0.5), gid, n_perm, seed=int(rng.integers(1 << 30)))
        fw += int((r["p_maxT"] < 0.05).any())
        raw += list(r["p_raw"])
    raw = np.asarray(raw)
    return dict(family_wise_size=fw / trials, raw_size=float((raw < 0.05).mean()), ks_p=float(stats.kstest(raw, "uniform")[1]), trials=trials)


def determinism_check(inputs: LabInputs, now, cfg: LabConfig, gate: DirectionGate | None = None) -> dict[str, Any]:
    """Two runs, same inputs, same seed: the per-hypothesis outcome table must be identical (rule 4: every draw is seeded). Returns
    the differing hypotheses, if any."""
    a = DirectionLab(cfg, gate).run(inputs, now).frame()
    b = DirectionLab(cfg, gate).run(inputs, now).frame()
    diff = [] if a.equals(b) else sorted(set(a.loc[(a != b).any(axis=1), "hypothesis"]) if len(a) == len(b) else set(a["hypothesis"]) ^ set(b["hypothesis"]))
    return dict(identical=not diff, differing=diff)


def lifetime_trials(state: LabState) -> int:
    """Total cells (hypothesis x model) ever tested across all fresh looks: the running price of the whole research programme, which
    the sequential alpha already charges through time and which the report states so no run looks free."""
    return int(sum(h.get("n_cells", 0) for h in state.history if h.get("fresh", True)))


# ---------------------------------------------------------------------------------------------- the lab
@dataclasses.dataclass
class LabReport:
    """One run of the lab. `statement` is the contract sentence; `headline()` never claims more than the outcomes support."""
    now: str
    config_hash: str
    code_hash: str
    gate: GateDecision
    results: dict[str, HypothesisResult]
    universe_funnel: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    ease: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    availability: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    leak_screen: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    frontier: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    controls: ControlReport | None = None
    eighty: dict[str, Any] = dataclasses.field(default_factory=dict)
    power: dict[str, float] = dataclasses.field(default_factory=dict)
    statement: str = ""
    warnings: list[str] = dataclasses.field(default_factory=list)
    stages_run: list[Stage] = dataclasses.field(default_factory=list)
    n_cells: int = 0
    existing_source: str = "none"
    years: tuple[int, ...] = ()
    latest_label_end: str = ""
    ic: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    ic_years: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    rank_gradient: dict[str, float] = dataclasses.field(default_factory=dict)
    market: dict[str, float] = dataclasses.field(default_factory=dict)
    cells_search: dict[str, Any] = dataclasses.field(default_factory=dict)
    interactions: dict[str, float] = dataclasses.field(default_factory=dict)
    curse: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    mechanisms: dict[str, list] = dataclasses.field(default_factory=dict)
    wrong_calls: dict[str, float] = dataclasses.field(default_factory=dict)
    gates: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    dossiers: dict[str, dict] = dataclasses.field(default_factory=dict)
    stability: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    profiles: dict[str, dict] = dataclasses.field(default_factory=dict)
    placebo: dict[str, float] = dataclasses.field(default_factory=dict)
    baselines: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    disclosures: dict[str, Any] = dataclasses.field(default_factory=dict)
    adequacy: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    blend: dict[str, Any] = dataclasses.field(default_factory=dict)
    vol_buckets: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    return_rank: dict[str, float] = dataclasses.field(default_factory=dict)
    truncation: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def protocol_ok(self) -> bool:
        return self.controls is not None and self.controls.ok

    def outcomes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results.values():
            out[str(r.outcome)] = out.get(str(r.outcome), 0) + 1
        return out

    def candidates(self) -> list[str]:
        return [n for n, r in self.results.items() if r.outcome is Outcome.CANDIDATE]

    def found_anything(self) -> bool:
        return any(r.outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED) for r in self.results.values())

    def headline(self) -> str:
        if not self.gate.is_open and not any(r.stage is not Stage.CHEAP_SCREEN for r in self.results.values()):
            return "Direction not studied: the volatility stage has no out-of-sample evidence yet (" + "; ".join(self.gate.reasons) + ")"
        if self.controls is not None and not self.controls.ok:
            return "Run voided by its own controls: " + "; ".join(self.controls.failures())
        c = self.candidates()
        if c:
            return f"{len(c)} candidate(s) {c} beat every comparator with family-wise control; a fresh holdout is required before use."
        if self.found_anything():
            return "Weak, unreplicated leads only; nothing beats every comparator. " + self.statement
        return "No reliable direction signal among predicted movers. " + self.statement

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.brief() for r in self.results.values()])


class DirectionLab:
    """Runs the section-10 programme. All arithmetic is seeded from cfg.seed; nothing here reads a row dated at or after `now`."""

    def __init__(self, cfg: LabConfig | None = None, gate: DirectionGate | None = None, specs: Mapping[str, HypothesisSpec] | None = None):
        self.cfg = (cfg or LabConfig()).validate()
        self.gate = gate or DirectionGate()
        self.specs = dict(specs or HYPOTHESES)
        for s in self.specs.values():
            s.validate()

    # -- helpers -------------------------------------------------------------------------------------------------
    def _years(self, pool: pd.DataFrame) -> tuple[int, ...]:
        if self.cfg.test_years:
            return tuple(self.cfg.test_years)
        ys = sorted(pool.loc[pool["up"].notna(), "year"].unique())
        return tuple(int(y) for y in ys[2:])              # the first two years are train and calibration by construction

    def _evidence(self, inputs: LabInputs, now) -> VolatilityEvidence:
        if inputs.volatility_evidence is not None:
            return inputs.volatility_evidence
        lab = inputs.labels[pd.DatetimeIndex(inputs.labels["end_date"]) < pd.Timestamp(as_date(now))]
        sc = inputs.score.dropna()
        sc = sc[sc.index.isin(lab.index)]
        return assess_volatility_evidence(sc, lab["mover"].astype(float), self.cfg.n_pick, min(self.cfg.n_boot, 400), self.cfg.seed)

    def _closed_report(self, now, decision: GateDecision, warn: list[str]) -> LabReport:
        res = {n: HypothesisResult(n, s.topic, Outcome.VOLATILITY_GATE_CLOSED, ResearchState.DORMANT, Stage.CHEAP_SCREEN,
                                   reasons=list(decision.reasons)) for n, s in self.specs.items()}
        return LabReport(str(as_date(now)), self.cfg.hash(), current_code_hash(), decision, res, warnings=warn)

    @staticmethod
    def _cells(names: Sequence[str], kinds: Sequence[str], cols: Sequence[str]) -> list[str]:
        return [f"{h}:{k}" for h in names for k in kinds if f"{h}:{k}" in set(cols)]

    def _max_t(self, P: pd.DataFrame, rows_info: pd.DataFrame, n_perm: int) -> dict[str, np.ndarray]:
        gid = pd.factorize(rows_info["date"])[0]
        return max_t_permutation(P.to_numpy(np.float64), rows_info["up"].to_numpy(float), rows_info["p0"].to_numpy(float), gid,
                                 n_perm, self.cfg.seed + 31)

    # -- the run -------------------------------------------------------------------------------------------------
    def run(self, inputs: LabInputs, now, ignore_gate: bool = False) -> LabReport:
        cfg = self.cfg
        warn = inputs.validate(now)
        if inputs.score.name in FORBIDDEN_RANK_COLUMNS:
            raise FirewallBreach(f"the volatility score is named {inputs.score.name!r}: the pool must be ranked by a forecast, not an outcome")
        ev = self._evidence(inputs, now)
        decision = self.gate.assess(ev)
        if not decision.is_open and not ignore_gate:
            return self._closed_report(now, decision, warn)
        if not decision.is_open:
            warn.append("gate closed but ignore_gate=True: results are a dry run and must not reach a decision")
        if abs(cfg.gate - shared_target()) > 1e-12:
            warn.append(f"lab gate {cfg.gate} differs from the fv pipeline gate {shared_target()}: the two 80% claims are not comparable")
        uni = ConditionalUniverse(cfg).build(inputs, now)
        warn += uni.warnings
        pool = uni.pool
        report = LabReport(str(as_date(now)), cfg.hash(), current_code_hash(), decision, {}, universe_funnel=uni.funnel, warnings=warn,
                           existing_source="production" if inputs.existing_p is not None else "combined_proxy")
        if len(pool) == 0:
            report.results = {n: HypothesisResult(n, s.topic, Outcome.INSUFFICIENT_DATA, ResearchState.DORMANT, Stage.CHEAP_SCREEN,
                                                  reasons=["empty pool"]) for n, s in self.specs.items()}
            return report
        audit = audit_pool(pool)
        if not audit["ok"]:
            raise FirewallBreach("direction pool failed its structural audit: " + "; ".join(audit["violations"]))
        years = self._years(pool)
        lc = dataclasses.replace(cfg, test_years=years)
        report.years = years
        report.latest_label_end = str(pd.Timestamp(pool["end_date"].max()).date()) if pool["end_date"].notna().any() else ""
        lab = inputs.labels
        report.ease = universe_ease_check(pool, lab[pd.DatetimeIndex(lab["end_date"]) < pd.Timestamp(as_date(now))], lc)
        Xp = inputs.X.reindex(pool.index)
        Xd = derive_features(Xp, pool, inputs.pattern_score)
        fams, screen = build_family_matrices(Xd, pool, lc, self.specs)
        report.availability = availability_table(Xd.columns, self.specs)
        report.leak_screen = screen
        truth = add_controls(fams, pool, lc)
        leak = control_leak_verdict(fams, pool, lc)
        names = [n for n in self.specs if fams[n].testable]
        report.results = {n: self._blank(n, fams[n]) for n in self.specs}
        if not names or not years:
            for n in names:
                report.results[n].outcome, report.results[n].reasons = Outcome.INSUFFICIENT_DATA, ["no test years or no testable inputs"]
            return report
        wf1 = run_walk_forward(pool, fams, Xd, lc, names, lc.screen_models, Stage.CHEAP_SCREEN, inputs.existing_p)
        report.stages_run.append(Stage.CHEAP_SCREEN)
        W1 = wide(wf1.pred)
        if W1.empty:
            for n in names:
                report.results[n].outcome, report.results[n].reasons = Outcome.INSUFFICIENT_DATA, ["no fitted walk-forward block"]
            return report
        info = row_frame(wf1.pred, W1.index.to_numpy())
        cells1 = self._cells(names, lc.screen_models, W1.columns)
        mt1 = self._max_t(W1[cells1], info, lc.n_perm)
        p_screen = dict(zip(cells1, mt1["p_raw"]))
        survivors = sorted({c.split(":")[0] for c in cells1 if p_screen[c] < lc.screen_p})
        for n in names:
            r = report.results[n]
            r.n, r.weeks = len(info), int(info["date"].nunique())
        pred_final, W_final, cells_final = wf1.pred, W1, cells1
        final_info, wf2 = info, None
        if survivors:
            hooks2 = make_extra_hooks(pool, fams, Xd, lc, [n for n in survivors if fams[n].testable], inputs.existing_p, self.specs)
            hooks2.update(miner_extra(inputs, pool, lc))
            wf2 = run_walk_forward(pool, fams, Xd, lc, survivors, lc.models, Stage.STRONGER_TESTS, inputs.existing_p, hooks2)
            report.stages_run.append(Stage.STRONGER_TESTS)
            W2 = wide(wf2.pred)
            common = W1.index.intersection(W2.index)
            new_cells = [c for c in self._cells(survivors, lc.models, W2.columns) if c not in cells1]
            if len(common) and new_cells:
                W_final = pd.concat([W1.loc[common, cells1], W2.loc[common, new_cells]], axis=1)
                cells_final = list(W_final.columns)
                final_info = row_frame(wf1.pred, common.to_numpy())
            pred_final = wf2.pred
        mt = self._max_t(W_final[cells_final], final_info, lc.n_perm)
        holm_p, bh_p = holm(mt["p_raw"]), benjamini_hochberg(mt["p_raw"])
        cell_stats = {c: dict(skill=float(mt["skill"][i]), p_raw=float(mt["p_raw"][i]), p_maxT=float(mt["p_maxT"][i]), p_holm=float(holm_p[i]),
                              p_bh=float(bh_p[i])) for i, c in enumerate(cells_final)}
        report.n_cells = len(cells_final)
        report.power = self._power(pred_final, W_final, final_info, lc)
        powered = report.power.get("mde_edge", 1.0) <= 0.03
        cu = winners_curse([cell_stats[c]["skill"] for c in cells_final], mt["null_sd"])
        cu.index = cells_final
        report.curse = cu
        for n in names:
            mine = [c for c in cells_final if c.split(":")[0] == n]
            if not mine:
                continue
            best = max(mine, key=lambda c: cell_stats[c]["skill"])
            r, cs = report.results[n], cell_stats[best]
            r.best_tag, r.skill, r.p_raw, r.p_maxT, r.p_holm, r.p_bh = best, cs["skill"], cs["p_raw"], cs["p_maxT"], cs["p_holm"], cs["p_bh"]
            y, pp = final_info["up"].to_numpy(float), W_final[best].to_numpy(float)
            r.acc = float(((pp >= 0.5) == (y > 0.5)).mean())
            r.acc_edge = r.acc - float(max(y.mean(), 1 - y.mean()))
            r.posterior_skill = float(cu.loc[best, "posterior_skill"])
            if n not in survivors:
                r.outcome, r.state = Outcome.NO_RELIABLE_SIGNAL, state_for(Outcome.NO_RELIABLE_SIGNAL, powered)
                r.reasons = [f"stage-1 screen: permutation p {r.p_raw:.3f} >= {lc.screen_p}; not escalated (compute saved)"]
                continue
            kind = best.split(":")[1]
            r.stage = Stage.STRONGER_TESTS
            r.comparators = compare_all(wf2.pred, n, best, kind, lc, lc.seed + 5)
            stress_test(r, wf2.pred, pool, fams[n].matrix, best, kind, lc)
            _, r.learn_curve = sample_size_curve(pool, fams[n].matrix, lc, kind)
            r.downside = downside_profile(wf2.pred, pool, best, 1.0, cost_bps=lc.cost_bps, seed=lc.seed)
            r.portfolio = weekly_portfolio(wf2.pred, pool, best, 1.0, lc.cost_bps, min(lc.n_boot, 300), lc.seed)
            r.calibration = calibration_report(wf2.pred, best, lc.seed)
            r.outcome, r.reasons = decide_outcome(r, lc, len(cells_final))
            r.state = state_for(r.outcome, powered)
            if r.outcome is Outcome.CANDIDATE:
                r.stage = Stage.CROSS_YEAR
        self._deep(report, Xd, pool, fams, lc)
        report.controls = evaluate_controls(pred_final, wf2.blocks if wf2 is not None else wf1.blocks, leak, lc, truth["planted_share"])
        if not report.controls.ok:
            for r in report.results.values():
                if r.outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED, Outcome.NO_RELIABLE_SIGNAL):
                    r.reasons = report.controls.failures() + r.reasons
                    r.outcome, r.state = Outcome.CONTROL_FAILURE, ResearchState.CANCELLED
        self._frontier(report, pred_final, cells_final, names, lc)
        report.statement = honest_statement(report.eighty, report.power, lc.gate)
        leads = [r for r in report.results.values() if r.stage is not Stage.CHEAP_SCREEN and r.best_tag]
        self._robustness(report, pred_final, W_final, final_info, pool, Xd, fams, lc)
        extras_for_report(report, inputs, pool, Xd, pred_final, W_final, final_info, lc)
        if leads:
            top = max(leads, key=lambda r: r.skill)
            report.gates = gate_alternatives(pred_final, top.best_tag, lc.gate)
            report.wrong_calls = wrong_call_anatomy(pred_final, pool, top.best_tag)[1]
        return report

    def _robustness(self, report: LabReport, pred: pd.DataFrame, W: pd.DataFrame, info: pd.DataFrame, pool: pd.DataFrame, Xd: pd.DataFrame,
                    fams: Mapping[str, FamilyMatrix], cfg: LabConfig) -> None:
        """Second-opinion checks on every hypothesis that reached stage 2 and is not already a null: dossier extras, the
        robust_verdict (a CANDIDATE that fails it is demoted to WEAK_UNREPLICATED with the failed checks named), the lead-lag
        profile of its strongest input, and the win-share stability of the winning cell among all cells."""
        report.stability = selection_stability(W, info, min(cfg.n_boot, 200), cfg.seed)
        for n, r in report.results.items():
            if r.stage is Stage.CHEAP_SCREEN or not r.best_tag or r.outcome not in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED):
                continue
            dz = evidence_dossier(report, n, pred, pool)
            rv = robust_verdict(dz)
            dz["robust"] = rv
            report.dossiers[n] = dz
            fam = fams[n]
            if fam.columns:
                ranked = report.ic[report.ic["column"].isin(fam.columns)] if len(report.ic) else report.ic
                col = str(ranked["column"].iloc[0]) if len(ranked) else fam.columns[0]
                report.profiles[n] = profile_verdict(lead_lag_profile(Xd[[col]], pool, col))
            if r.outcome is Outcome.CANDIDATE and not rv["robust"]:
                r.outcome, r.state = Outcome.WEAK_UNREPLICATED, state_for(Outcome.WEAK_UNREPLICATED, True)
                r.reasons = ["robustness failed: " + ",".join(rv["failed"])] + r.reasons
            if report.profiles.get(n, {}).get("kind") == "contaminated" and r.outcome is not Outcome.LEAK_SUSPECT:
                r.outcome, r.state = Outcome.LEAK_SUSPECT, ResearchState.CANCELLED
                r.reasons = ["lead-lag profile: the feature predicts outcomes that were already known: " + report.profiles[n]["detail"]] + r.reasons

    def _deep(self, report: LabReport, Xd: pd.DataFrame, pool: pd.DataFrame, fams: Mapping[str, FamilyMatrix], cfg: LabConfig) -> None:
        """Model-free diagnostics that need no walk-forward: single-feature IC tables (pooled and by year), the volatility model's
        own directional gradient, the market-vs-stock decomposition, the exhaustive small-cell search against its chance maximum,
        the interaction replication scan and the mechanism checks of every hypothesis."""
        owner = {c: n for n, f in fams.items() if n in self.specs for c in f.columns}
        cols = [c for c in owner if c in Xd.columns]
        if not cols:
            return
        M = Xd[cols]
        report.ic = feature_ic_table(M, pool, "up", family_of=owner)
        report.ic_years = ic_by_year(M[report.ic["column"].head(30).tolist()] if len(report.ic) else M.iloc[:, :30], pool)
        report.rank_gradient = rank_gradient(pool, seed=cfg.seed)[1]
        report.market = market_component(pool)
        top = report.ic["column"].head(10).tolist() if len(report.ic) else cols[:10]
        _, report.cells_search = conditional_cells(M, pool, top[:8], seed=cfg.seed)
        _, report.interactions = interaction_scan(M, pool, top)
        report.mechanisms = mechanism_report(Xd, pool, seed=cfg.seed)
        for name, res in report.mechanisms.items():
            if name in report.results:
                report.results[name].mechanism = mechanism_summary(res)

    def _blank(self, name: str, fam: FamilyMatrix) -> HypothesisResult:
        spec = self.specs[name]
        r = HypothesisResult(name, spec.topic, fam.outcome or Outcome.NO_RELIABLE_SIGNAL, ResearchState.QUEUED, Stage.CHEAP_SCREEN,
                             n_columns=len(fam.columns), dropped=dict(fam.dropped))
        if fam.outcome is not None:
            r.state = state_for(fam.outcome, False)
            r.reasons = [f"{c}: {why}" for c, why in list(fam.dropped.items())[:6]] or ["no usable inputs"]
        return r

    def _power(self, pred: pd.DataFrame, W: pd.DataFrame, info: pd.DataFrame, cfg: LabConfig) -> dict[str, float]:
        y = info["up"].to_numpy(float)
        proxy = next((c for c in ("combined:linear", "combined:gbm") if c in pred["model"].unique()), None)
        if proxy is not None:
            pw = pred[pred["model"] == proxy].set_index("row").loc[info.index.to_numpy(), "p"].to_numpy(float)
            correct = ((pw >= 0.5) == (y > 0.5)).astype(float)
        else:
            correct = (y > 0.5).astype(float)
        wk = pd.factorize(info["date"])[0]
        deff = week_design_effect(correct, wk)
        return power_report(len(y), int(info["date"].nunique()), float(y.mean()), deff, cfg.alpha, target=cfg.gate)

    def _frontier(self, report: LabReport, pred: pd.DataFrame, cells: Sequence[str], names: Sequence[str], cfg: LabConfig) -> None:
        keep = [m for m in pred["model"].unique() if m in set(cells) or m.startswith(("combined", "planted", "leak", "ctl_"))]
        sub = pred[pred["model"].isin(keep)]
        if sub.empty:
            report.eighty = dict(cells=0, reached_lo_adj=0, reached_lo=0, control_hits=0)
            return
        n_seg = sum(sub[c].nunique() for c in ("seg", "reg") if c in sub)
        n_cells = max(len(names) * len(cfg.models) * (1 + n_seg) * len(cfg.coverages), 1)
        ft = DF.frontier_table(sub, cfg.coverages, seg_cols=("seg", "reg"), B=min(cfg.n_boot, 400), seed=cfg.seed, n_cells=n_cells)
        report.frontier = ft
        report.eighty = DF.eighty_question(ft, gate=cfg.gate)


# ---------------------------------------------------------------------------------------------- persistent state and the step entry
@dataclasses.dataclass
class LabState:
    """What survives between runs (trusted side). `null_streak` counts consecutive powered runs that found no lead, which is the
    evidence the compute manager needs before moving effort from direction to volatility (RESEARCH_MAPPING: 'direction yields
    nothing' is a valid answer, not a stall). `info_prev` is the information clock (matured weeks) at the last FRESH look, the
    input of the alpha-spending rule; `last_alpha` is reused when the very same data are looked at again."""
    states: dict[str, ResearchState] = dataclasses.field(default_factory=dict)
    null_streak: int = 0
    runs: int = 0
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    seen_keys: set = dataclasses.field(default_factory=set)
    ledger: "EvidenceLedger | None" = None
    info_prev: int = 0
    last_alpha: float = 0.0
    alpha_spent: float = 0.0

    def digest(self) -> str:
        return stable_hash({"states": {k: str(v) for k, v in sorted(self.states.items())}, "streak": self.null_streak, "runs": self.runs,
                            "info": self.info_prev}, 12)

    def validate(self) -> list[str]:
        """Consistency of the persisted state (a corrupted state must not silently steer the research queue)."""
        errs = []
        if self.runs < len(self.history):
            errs.append("more history entries than runs")
        if self.null_streak < 0 or self.null_streak > self.runs:
            errs.append("null streak outside [0, runs]")
        if self.info_prev < 0 or not 0 <= self.alpha_spent <= 1:
            errs.append("information clock or alpha spent out of range")
        bad = [k for k, v in self.states.items() if not isinstance(v, ResearchState)]
        if bad:
            errs.append(f"non-ResearchState values for {bad[:3]}")
        return errs


def data_fingerprint(inputs: LabInputs) -> str:
    """Cheap content fingerprint (shapes, column names, boundary index rows, checksum of a subsample) so an identical rerun is
    recognised without hashing gigabytes."""
    def edge(idx):
        n = len(idx)
        return [str(idx[i]) for i in (0, n // 2, n - 1)] if n else []
    sc = inputs.score.dropna()
    return stable_hash({"x": [inputs.X.shape, list(map(str, inputs.X.columns)), edge(inputs.X.index)],
                        "s": [len(sc), round(float(sc.sum()), 6) if len(sc) else 0.0, edge(sc.index)],
                        "l": [inputs.labels.shape, edge(inputs.labels.index),
                              round(float(inputs.labels["fwd"].fillna(0).sum()), 6)]}, 16)


def compute_shift(state: LabState, report: LabReport, min_streak: int = 3) -> dict[str, Any]:
    """Should effort move from direction to volatility? Only when repeated runs were powered, controls passed, the gate was open and
    nothing survived. An underpowered or void run never counts as a null (absence of evidence)."""
    powered = report.power.get("mde_edge", 1.0) <= 0.03
    valid = report.gate.is_open and report.protocol_ok and powered
    shift = bool(valid and not report.found_anything() and state.null_streak >= min_streak)
    why = ("repeated powered runs found no direction lead" if shift else
           "keep direction on the queue: " + ("gate closed" if not report.gate.is_open else "run void" if not report.protocol_ok
                                               else "underpowered" if not powered else "leads exist" if report.found_anything()
                                               else f"null streak {state.null_streak} < {min_streak}"))
    return dict(shift_to_volatility=shift, reason=why, null_streak=state.null_streak,
                priority_multiplier=report.gate.priority_multiplier() * (0.25 if shift else 1.0))


def step(state: LabState | None, now, inputs: LabInputs, cfg: LabConfig | None = None, gate: DirectionGate | None = None,
         ignore_gate: bool = False, spending: AlphaSpending | None = None) -> tuple[LabState, LabReport]:
    """The lab's ONE public entry for the research loop: run once at `now`, fold the outcome into the persistent state.
    Sequential looks: each fresh look on more matured weeks is charged the Lan-DeMets alpha increment (AlphaSpending) instead of
    the full alpha, so re-running the lab every month cannot manufacture a discovery. A rerun on identical inputs and code is
    recorded, reuses the previous alpha and does not extend the null streak (repetition is not replication)."""
    state = state or LabState()
    bad = state.validate()
    if bad:
        raise ValueError("LabState is inconsistent: " + "; ".join(bad))
    base = (cfg or LabConfig()).validate()
    spending = spending or AlphaSpending(base.alpha)
    key = stable_hash([data_fingerprint(inputs), current_code_hash(), str(as_date(now))], 16)
    fresh = key not in state.seen_keys
    weeks = information_weeks(inputs, now)
    alpha = max(spending.increment(state.info_prev, max(weeks, state.info_prev)), 1e-4) if fresh else state.last_alpha
    lab = DirectionLab(dataclasses.replace(base, alpha=min(alpha, 0.49)), gate)
    report = lab.run(inputs, now, ignore_gate=ignore_gate)
    ledger = copy.deepcopy(state.ledger) if state.ledger is not None else EvidenceLedger(base.alpha)
    new = LabState(dict(state.states), state.null_streak, state.runs + 1, list(state.history), set(state.seen_keys), ledger, state.info_prev,
                   state.last_alpha, state.alpha_spent)
    feed_ledger(new.ledger, report)
    for name, r in report.results.items():
        new.states[name] = r.state
    powered = report.power.get("mde_edge", 1.0) <= 0.03
    counts = report.gate.is_open and report.protocol_ok and powered
    if fresh:
        new.null_streak = (state.null_streak + 1) if counts and not report.found_anything() else (0 if report.found_anything() else state.null_streak)
        new.info_prev, new.last_alpha = max(weeks, state.info_prev), alpha
        new.alpha_spent = min(state.alpha_spent + alpha, 1.0)
    new.seen_keys.add(key)
    new.history.append(dict(key=key, fresh=fresh, outcomes=report.outcomes(), gate_open=report.gate.is_open, protocol_ok=report.protocol_ok,
                            null_streak=new.null_streak, config=base.hash(), alpha=alpha, weeks=weeks, n_cells=report.n_cells))
    return new, report


# ---------------------------------------------------------------------------------------------- sequential looks (alpha spending)
class AlphaSpending:
    """The lab is run again and again as weeks mature; each look at overlapping data is another chance for a false discovery. Lan-
    DeMets spending spreads a total alpha over information time t = weeks_now / target_weeks: 'obf' spends almost nothing early
    (O'Brien-Fleming shape), 'pocock' spends evenly. The alpha available at a look is the INCREMENT of the spending function since
    the previous look, a conservative (Bonferroni-style) level that keeps the total below `total` whatever the number of looks."""

    def __init__(self, total: float = 0.05, target_weeks: int = 520, kind: str = "obf"):
        if not 0 < total < 0.5 or target_weeks < 8 or kind not in ("obf", "pocock"):
            raise ValueError("AlphaSpending: total in (0, 0.5), target_weeks >= 8, kind in {'obf', 'pocock'}")
        self.total, self.target_weeks, self.kind = total, int(target_weeks), kind

    def spent(self, weeks: float) -> float:
        t = min(max(weeks / self.target_weeks, 0.0), 1.0)
        if t <= 0:
            return 0.0
        if self.kind == "pocock":
            return float(self.total * math.log(1 + (math.e - 1) * t))
        return float(min(2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - self.total / 2) / math.sqrt(t)), self.total))

    def increment(self, weeks_prev: float, weeks_now: float) -> float:
        """Alpha to use at this look; never negative, never above what remains."""
        if weeks_now < weeks_prev:
            raise ValueError("information cannot shrink between looks (would re-use evidence)")
        return max(self.spent(weeks_now) - self.spent(weeks_prev), 0.0)


def information_weeks(inputs: LabInputs, now) -> int:
    """Matured, scored decision weeks available at `now` (the information clock of the sequential test)."""
    lab = inputs.labels
    ok = pd.DatetimeIndex(lab["end_date"]) < pd.Timestamp(as_date(now))
    d = pd.DatetimeIndex(lab.index.get_level_values(0))[ok]
    sc = inputs.score.dropna()
    sd = pd.DatetimeIndex(sc.index.get_level_values(0))
    return int(len(d.unique().intersection(sd.unique())))


# ---------------------------------------------------------------------------------------------- synthetic worlds
SYNTH_COLUMNS = ("r1", "r5", "r20", "r60", "r120", "mom_12_1", "dist_52wh", "dist_ma50", "dist_ma200", "close_loc", "intraday20",
                 "gap_today", "overnight20", "ear", "ear_volsurge", "days_since_earn", "days_to_earn", "earn_in_week", "news5",
                 "ins_buyers30", "ins_value30", "rel_ind20", "rel_ind60", "r5_nonews", "r5_news", "m_spy_ma50", "m_spy_ma200",
                 "m_spy_r5", "m_vix", "m_vix_chg5", "m_breadth", "atr_pct")


def synthetic_inputs(seed: int = 0, n_tickers: int = 150, n_weeks: int = 416, world: str = "null", strength: float = 0.20,
                     start: str = "2010-01-01", sectors: int = 6, vol_skill: float = 1.3) -> LabInputs:
    """A small market with known truth, for tests and for the lab's own health probe (self_check).
    world 'null'    : P(up) = 0.5 + a shared weekly drift; no feature relates to direction.
    world 'planted' : P(up) = 0.5 - strength * tanh(z(r5)) (a short-term reversal of size `strength`).
    world 'leaky'   : as null, but column r1 is built from the realised outcome (a look-ahead the lab must refuse).
    The volatility score is informative in every world (the gate must open), the first two years are unscored (a walk-forward that
    has not been fitted yet), and the first-touch labels follow the contract's next-open convention (entry after the decision)."""
    if world not in ("null", "planted", "leaky"):
        raise ValueError("world must be null, planted or leaky")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_weeks, freq="W-FRI")
    tickers = [f"S{i:03d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    X = {c: rng.standard_normal(n).astype("float32") * 0.05 for c in SYNTH_COLUMNS}
    tick_sector = np.array([f"SEC{i % sectors}" for i in range(n_tickers)])
    v_t = rng.standard_normal(n_tickers)                                    # persistent volatility trait per ticker
    v = np.tile(v_t, n_weeks) * 0.8 + rng.standard_normal(n) * 0.6
    X["atr_pct"] = np.abs(v).astype("float32")
    X["days_since_earn"] = rng.integers(1, 90, n).astype("float32")
    X["days_to_earn"] = rng.integers(1, 90, n).astype("float32")
    X["earn_in_week"] = (rng.uniform(size=n) < 0.08).astype("float32")
    X["news5"] = (rng.uniform(size=n) < 0.10).astype("float32")
    X["ear_volsurge"] = np.abs(rng.standard_normal(n)).astype("float32") * 2
    X["close_loc"] = rng.uniform(0, 1, n).astype("float32")
    wk_mkt = rng.standard_normal(n_weeks)
    for c, sc_ in (("m_spy_ma50", 0.04), ("m_spy_ma200", 0.08), ("m_spy_r5", 0.02), ("m_vix", 5.0), ("m_vix_chg5", 0.05), ("m_breadth", 0.15)):
        base = 20.0 if c == "m_vix" else 0.0
        X[c] = np.repeat(base + wk_mkt * sc_ + rng.standard_normal(n_weeks) * sc_ * 0.3, n_tickers).astype("float32")
    z5 = (X["r5"] - X["r5"].mean()) / X["r5"].std()
    p_up = 0.5 - strength * np.tanh(z5) if world == "planted" else np.full(n, 0.5)
    drift = np.repeat(0.04 * np.tanh(wk_mkt * 0.8), n_tickers)               # week-level up/down tilt shared by all names
    up = rng.uniform(size=n) < np.clip(p_up + drift, 0.02, 0.98)
    mover = rng.uniform(size=n) < 1 / (1 + np.exp(-(-1.4 + vol_skill * v)))
    mag = np.where(mover, 0.10 + np.abs(rng.standard_normal(n)) * 0.05, np.abs(rng.standard_normal(n)) * 0.02)
    fwd = np.where(up, 1.0, -1.0) * mag
    score = pd.Series(v + rng.standard_normal(n) * 0.9, index=idx, name="p_vol")
    score[np.repeat(dates.year < dates[0].year + 2, n_tickers)] = np.nan
    ent = pd.DatetimeIndex(np.repeat(dates + pd.Timedelta(days=3), n_tickers))
    end = pd.DatetimeIndex(np.repeat(dates + pd.Timedelta(days=8), n_tickers))
    labels = pd.DataFrame({"entry_date": ent, "end_date": end, "fwd": fwd.astype("float32"), "mover": mover, "amb": False,
                           "up_first": np.where(mover, up.astype(float), np.nan), "up_sign": (fwd > 0).astype(float)}, index=idx)
    if world == "leaky":
        X["r1"] = (np.where(fwd > 0, 1.0, -1.0) * 0.05 + rng.standard_normal(n) * 0.01).astype("float32")
    Xf = pd.DataFrame(X, index=idx)
    sector = pd.Series(tick_sector, index=tickers, name="sector")
    return LabInputs(Xf, score, labels, sector=sector)


def synthetic_config(**kw) -> LabConfig:
    """A fast configuration for synthetic worlds: linear models only, few permutations, three test years."""
    base = dict(models=("linear",), screen_models=("linear",), n_boot=150, n_perm=120, min_test_rows=300, min_weeks=20, min_eras=3,
                min_train=1500, min_calib=150, test_years=(2014, 2015, 2016, 2017), n_pick=30, n_pool=100, seed=3)
    base.update(kw)
    return LabConfig(**base).validate()


def self_check(seed: int = 0, strength: float = 0.25, cfg: LabConfig | None = None) -> dict[str, Any]:
    """The lab tests itself on worlds whose truth is known: it must call the null world 'no reliable signal' and find the planted
    reversal in the planted world, and its controls must pass in both. Returns a dict of booleans plus the two reports; a False
    anywhere means the lab must not be trusted on real data until fixed (research-brain health input)."""
    cfg = cfg or synthetic_config()
    lab = DirectionLab(cfg, DirectionGate(min_weeks=30))
    now = "2019-06-01"
    null = lab.run(synthetic_inputs(seed, world="null"), now)
    planted = lab.run(synthetic_inputs(seed + 1, world="planted", strength=strength), now)
    found = planted.results["reversal"].outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED)
    false_pos = null.candidates()
    return dict(null_no_candidates=not false_pos, null_false_positives=false_pos, planted_found=bool(found),
                controls_ok_null=null.protocol_ok, controls_ok_planted=planted.protocol_ok, ok=bool(not false_pos and found and null.protocol_ok
                                                                                                       and planted.protocol_ok),
                null_report=null, planted_report=planted)


# ---------------------------------------------------------------------------------------------- robustness of the conclusion
DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {}, "barrier_label": {"target": "up_first"}, "market_neutral": {"target": "up_rel"},
    "narrow_picks": {"n_pick": 15, "n_pool": 60}, "wide_picks": {"n_pick": 60, "n_pool": 140}}


def robustness_sweep(inputs: LabInputs, now, cfg: LabConfig, variants: Mapping[str, Mapping[str, Any]] | None = None,
                     gate: DirectionGate | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-runs the lab under label, market-neutral and universe-size variants. A finding that appears under one definition of
    'direction' or of 'predicted mover' and vanishes under the others is a property of the definition, not of the market. Returns
    the per-variant outcome table and an agreement summary: the share of hypotheses whose outcome CLASS (signal / no signal / other)
    is the same in every variant, and the conclusion drawn (stable_null, stable_signal, definition_dependent)."""
    rows = []
    for name, kw in (variants or DEFAULT_VARIANTS).items():
        rep = DirectionLab(dataclasses.replace(cfg, **kw), gate).run(inputs, now)
        for h, r in rep.results.items():
            rows.append(dict(variant=name, hypothesis=h, outcome=str(r.outcome), skill=r.skill, p_maxT=r.p_maxT,
                             cls="signal" if r.outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED) else
                             "null" if r.outcome is Outcome.NO_RELIABLE_SIGNAL else "other"))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, dict(agreement=float("nan"), conclusion="empty", definition_dependent=[])
    per = t.groupby("hypothesis")["cls"].nunique()
    dep = sorted(per[per > 1].index.tolist())
    signal_any = bool((t["cls"] == "signal").any())
    conclusion = "definition_dependent" if dep else ("stable_signal" if signal_any else "stable_null")
    return t, dict(agreement=float((per == 1).mean()), conclusion=conclusion, definition_dependent=dep, variants=int(t["variant"].nunique()))


# ---------------------------------------------------------------------------------------------- research-brain integration
def _bucket(n: float) -> str:
    """Opaque size class ('n_1e3' = thousands of rows): exact counts in the 1900-2100 range look like years to the blind-side scrub."""
    return f"n_1e{int(math.log10(max(float(n), 1.0)))}"


def decision_effect_of(outcome: Outcome) -> DecisionEffect:
    """A CANDIDATE may, after a fresh holdout, inform DIRECTION; everything else is research knowledge that may not reach production."""
    return DecisionEffect.DIRECTION if outcome is Outcome.CANDIDATE else DecisionEffect.NONE


def _record_payload(name: str, r: HypothesisResult, years: Sequence[int]) -> dict[str, Any]:
    return dict(kind="direction_hypothesis", hypothesis=name, topic=r.topic, outcome=str(r.outcome), skill=round(float(r.skill), 5)
                if np.isfinite(r.skill) else None, p_family=round(float(r.p_maxT), 4), beaten=r.comparators_passed,
                comparators=len(ALL_COMPARATORS), size=_bucket(r.n), effect=str(decision_effect_of(r.outcome)),
                filed_under=[int(y) for y in years])


def to_matured_records(report: LabReport, run_id: str = "", extra_years: Sequence[int] = ()) -> list[MaturedRecord]:
    """Every hypothesis verdict as a MATURED_RESEARCH_STATE record. matured_at is the last label close used (nothing can be
    released before every outcome behind it has closed); `filed_under` carries the real years the evidence comes from so the
    same-year rerun guard can refuse it, and is stripped before anything crosses to the trader."""
    matured = report.latest_label_end or str(report.now)
    years = sorted(set(int(y) for y in report.years) | set(int(y) for y in extra_years))
    prov = Provenance(created_real=str(report.now), learned_at=matured, code_hash=report.code_hash or current_code_hash(),
                      config_hash=report.config_hash, experiment_id="direction_lab", run_id=run_id, outcomes_seen_through=matured)
    out = []
    for name, r in report.results.items():
        pl = _record_payload(name, r, years)
        out.append(MaturedRecord("DL" + stable_hash([name, report.config_hash, matured, str(r.outcome)], 12), matured, pl, prov))
    summary = dict(kind="direction_summary", statement=report.statement, gate_open=report.gate.is_open, protocol_ok=report.protocol_ok,
                   candidates=len(report.candidates()), filed_under=[int(y) for y in years])
    out.append(MaturedRecord("DL" + stable_hash(["summary", report.config_hash, matured], 12), matured, summary, prov))
    return out


def replay_year_guard(payload: Mapping[str, Any], replaying_years: Sequence[int]) -> None:
    """Research filed under a real year must never be released while that same year is being replayed in disguise (the same-year
    rerun leak). Fails closed on overlap."""
    overlap = sorted(set(int(y) for y in payload.get("filed_under", ())) & set(int(y) for y in replaying_years))
    if overlap:
        raise FirewallBreach(f"direction research filed under {len(overlap)} year(s) that are currently being replayed: refusing release")


def release_to_trader(records: Sequence[MaturedRecord], now, replaying_years: Sequence[int] = ()) -> list[dict[str, Any]]:
    """The only road from the lab to a decision: each record must have matured strictly before `now` (MaturedRecord.gate), must not
    come from a replayed year, must be a CANDIDATE with a DIRECTION effect (leads and nulls stay on the research side), and its
    payload, stripped of the filing years, must survive the blind-side scan. Anything else raises or is withheld."""
    out = []
    for rec in records:
        payload = dict(rec.gate(now))
        replay_year_guard(payload, replaying_years)
        if payload.get("kind") != "direction_hypothesis" or payload.get("effect") != str(DecisionEffect.DIRECTION):
            continue
        safe = {k: v for k, v in payload.items() if k not in ("filed_under", "topic")}
        TV.assert_trader_safe(safe, what=f"direction record {rec.record_id}")
        out.append(safe)
    return out


def questions_from_report(report: LabReport, created_real: str) -> list[ResearchQuestion]:
    """Section 40: the lab's own follow-ups as research objects. Text is identity-free (no tickers, dates or years). A closed gate
    asks about volatility, a void run asks about the protocol, leads ask for replication, a powered null asks whether the compute
    should move on, and an unavailable input asks for data."""
    ev_through = report.latest_label_end or str(report.now)
    qs = []

    def q(text, source, success, failure, **kw):
        qs.append(ResearchQuestion.make(text, source, Problem.DIRECTION, created_real, ev_through, success, failure, **kw))
    if report.results and all(r.outcome is Outcome.VOLATILITY_GATE_CLOSED for r in report.results.values()):
        qs.append(ResearchQuestion.make("Which volatility improvements would open the direction gate?", "gate", Problem.VOLATILITY, created_real,
                                        ev_through, "AUC lower bound and top-pick lift lower bound clear the gate thresholds",
                                        "No improvement after a powered volatility experiment"))
        return qs
    if report.controls is not None and not report.controls.ok:
        q("Which protocol control failed and why: " + "; ".join(report.controls.failures()), "contradiction",
          "the failing control is explained and the run repeated with all controls passing", "the failure is not reproducible")
    for name, r in report.results.items():
        if r.outcome is Outcome.WEAK_UNREPLICATED:
            q(f"Does the {r.topic} lead replicate on a fresh, unseen period among predicted movers?", "surprise",
              "family-wise p below alpha on a period that no earlier run touched", "skill within noise on the fresh period",
              parents=(name,))
        elif r.outcome is Outcome.CANDIDATE:
            q(f"Does the {r.topic} candidate survive the fresh holdout and transfer to unseen names, sectors and regimes?", "discovery",
              "beats every comparator on the holdout with positive net payoff", "any comparator not beaten or payoff <= 0", parents=(name,))
        elif r.outcome is Outcome.UNAVAILABLE_INPUT:
            q(f"Can the missing {r.topic} inputs be built point-in-time from raw data?", "discovery",
              "a leak-free column exists for at least one input", "no point-in-time source exists", parents=(name,))
    if report.power.get("mde_edge", 0) > 0.03 and report.controls is not None and report.controls.ok:
        q("How much more history is needed before a 3-point accuracy edge could be detected among predicted movers?", "regime",
          "the sample reaches the effective size that gives 80% power for a 3-point edge", "the edge cannot be resolved with available history")
    if report.cells_search and report.cells_search.get("beats_null"):
        q("Does the best small conditional pocket found by the exhaustive cell search survive a fresh period?", "surprise",
          "same side and accuracy above the chance maximum on new data", "accuracy inside the chance maximum")
    return qs


def experiment_value(report: LabReport, name: str, specs: Mapping[str, HypothesisSpec] | None = None) -> ExperimentValue:
    """Value vector of re-running (or extending) one hypothesis, for the priority engine. Unknowns stay None. direction_value grows
    with the strength of the evidence for a lead and shrinks to ~0 for a powered null; overfit_risk grows with the number of cells
    tried; redundancy is the share of the hypothesis' inputs that another hypothesis also uses."""
    specs = specs or HYPOTHESES
    r = report.results[name]
    lead = r.outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED)
    dv = float(max(0.0, 1.0 - r.p_maxT)) if lead else 0.0
    others = set().union(*[set(s.columns) for k, s in specs.items() if k != name]) if len(specs) > 1 else set()
    mine = set(specs[name].columns)
    unc = float(min(report.power.get("mde_edge", 1.0), 1.0)) if report.power else None
    cost = {Stage.CHEAP_SCREEN: 2.0, Stage.STRONGER_TESTS: 15.0, Stage.CROSS_YEAR: 30.0}.get(r.stage, 60.0)
    return ExperimentValue(information_gain=dv if lead else 0.05 * (1.0 if report.gate.is_open else 0.2), decision_value=dv,
                           uncertainty_reduction=unc, transfer_potential=float(r.era.get("share_positive")) if r.era else None,
                           direction_value=dv, compute_cost=cost, overfit_risk=float(1 - math.exp(-max(report.n_cells, 1) / 20.0)),
                           redundancy=float(len(mine & others) / max(len(mine), 1)))


def priority_terms(report: LabReport, state: LabState | None = None) -> dict[str, Any]:
    """What research_policy.PriorityFunction needs from the direction lab: the direction target's value, the gate multiplier and the
    compute-shift recommendation (INTEGRATION: research_policy adds a DIRECTION target and reads these three numbers)."""
    state = state or LabState()
    shift = compute_shift(state, report)
    best = max((experiment_value(report, n).direction_value or 0.0 for n in report.results), default=0.0)
    return dict(direction_value=best, gate_multiplier=report.gate.priority_multiplier(), shift=shift["shift_to_volatility"],
                reason=shift["reason"], multiplier=shift["priority_multiplier"])


# ---------------------------------------------------------------------------------------------- reporting
def format_report(report: LabReport, top: int = 15) -> str:
    """Plain-text report: headline first, then the funnel, the controls, one line per hypothesis (with the reason it failed), the
    honest 80% statement, the power, the small-cell search and the market-vs-stock decomposition."""
    L = [f"DIRECTION LAB  now={report.now}  config={report.config_hash}  code={report.code_hash}", "STATUS: IMPLEMENTED - NOT VALIDATED",
         "", "HEADLINE: " + report.headline(), ""]
    ev = report.gate.evidence
    L.append(f"volatility gate: {'OPEN' if report.gate.is_open else 'CLOSED'} (weeks={ev.n_weeks}, AUC={ev.auc:.3f} lo={ev.auc_lo:.3f}, "
             f"top-pick lift={ev.top_lift:.2f} lo={ev.top_lift_lo:.2f}); " + "; ".join(report.gate.reasons))
    if len(report.universe_funnel):
        L += ["", "universe funnel (per year):", report.universe_funnel.to_string(index=False)]
    if report.controls is not None:
        c = report.controls
        L += ["", f"controls: {'PASS' if c.ok else 'FAIL'}  planted found={c.planted_found} (acc {c.planted_acc:.3f}, lo {c.planted_lo:.3f})  "
                  f"leak caught={c.leak_caught}  shuffled null={c.shuffled_null}  random null={c.random_null}  audit={c.audit_ok}"]
        L += ["  note: " + n for n in c.notes]
    L += ["", f"hypotheses ({len(report.results)}; comparators: {', '.join(c.value for c in ALL_COMPARATORS)}; existing model = {report.existing_source}):"]
    for name, r in list(report.results.items())[:top]:
        L.append(f"  {name:22s} {str(r.outcome):24s} skill={r.skill:+.4f} p_raw={r.p_raw:.3f} p_family={r.p_maxT:.3f} "
                 f"beat={r.comparators_passed}/{len(ALL_COMPARATORS)}  " + ("; ".join(r.reasons[:2]) if r.reasons else ""))
    if report.power:
        p = report.power
        L += ["", f"power: n={p['n']} weeks={p['weeks']} design effect={p['design_effect']:.1f} effective n={p['n_eff']:.0f} "
                  f"minimum detectable edge={p['mde_edge']:.3f} bets needed to confirm {EIGHTY:.0%}={p['n_bets_for_target']:.0f}"]
    if report.eighty:
        e = report.eighty
        L += [f"80% question: cells={e.get('cells')} reached (point)={e.get('reached_acc')} reached (lower bound)={e.get('reached_lo')} "
              f"reached (multiplicity-adjusted)={e.get('reached_lo_adj')} control hits={e.get('control_hits')}", "  " + report.statement]
    if report.cells_search:
        s = report.cells_search
        L.append(f"small-cell search: {s.get('cells')} cells, best out-of-sample {s.get('best_acc', float('nan')):.3f} vs chance maximum "
                 f"{s.get('null_q95', float('nan')):.3f} -> {'BEATS chance' if s.get('beats_null') else 'inside chance'}")
    if report.market:
        m = report.market
        L.append(f"market vs stock: week explains {m.get('icc', float('nan')):.1%} of direction variance (excess dispersion {m.get('excess_dispersion', float('nan')):.2f}x)")
    d = report.disclosures
    if d:
        s = d["survivorship"]
        L.append(f"survivorship: {s['tickers']} tickers, {s['ended_early']} ended early -> "
                 + ("SURVIVOR-ONLY SUSPECTED; " + s["note"] if s["survivor_only_suspected"] else "delistings present"))
        L.append(f"chance of a lead: expected {d['false_lead_arithmetic']['expected_weak']:.2f} unadjusted leads among {len(HYPOTHESES)} hypotheses by luck alone")
    if len(report.baselines):
        b = report.baselines.sort_values("edge", ascending=False).iloc[0]
        L.append(f"best fixed rule: {b['rule']} accuracy {b['acc']:.3f} (majority {b['majority']:.3f}, edge {b['edge']:+.3f})")
    if report.truncation:
        L.append(f"future-read test: {'PASS' if report.truncation['ok'] else 'FAIL'} ({report.truncation['columns']} derived columns, {report.truncation['rows']} rows)")
    if report.warnings:
        L += ["", "warnings:"] + ["  " + w for w in report.warnings]
    return "\n".join(L)


def save_report(report: LabReport, out_dir, extra: Mapping[str, Any] | None = None):
    """Writes summary.json, hypotheses.csv, frontier.csv, ic.csv and report.txt with a provenance stamp (code hash, config hash, seed)
    under out_dir. The runner script (wave 2) calls this; the tests call it on a temporary directory."""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(now=report.now, config_hash=report.config_hash, code_hash=report.code_hash, gate_open=report.gate.is_open,
                   protocol_ok=report.protocol_ok, outcomes=report.outcomes(), candidates=report.candidates(), statement=report.statement,
                   power=report.power, eighty={k: v for k, v in report.eighty.items() if k not in ("hits", "best")},
                   warnings=report.warnings, stages=[str(s) for s in report.stages_run], extra=dict(extra or {}))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    report.frame().to_csv(out / "hypotheses.csv", index=False)
    if len(report.frontier):
        report.frontier.to_csv(out / "frontier.csv", index=False)
    if len(report.ic):
        report.ic.to_csv(out / "ic.csv", index=False)
    (out / "report.txt").write_text(format_report(report), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------- placebo and timing controls
def lead_lag_profile(M: pd.DataFrame, pool: pd.DataFrame, column: str, lags: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3), min_rows: int = 300) -> pd.DataFrame:
    """Mean weekly IC of one feature against the outcome of the SAME ticker `lag` weeks later (lag 0 = the real target, lag < 0 = an
    outcome that was already known when the feature was formed, lag > 0 = a later week). A genuine one-week-ahead predictor peaks at
    0 and is near zero elsewhere; a column that predicts outcomes of weeks before it existed is contaminated, and a column that
    predicts every lag equally is a slow-moving trait of the stock, not a weekly forecast. Rows are matched by (ticker, date order)."""
    cols = ["lag", "n", "mean_ic", "t"]
    if column not in M.columns or len(pool) < min_rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({"x": M[column].to_numpy(dtype=float), "up": pool["up"].to_numpy(dtype=float), "ticker": pool["ticker"].to_numpy(),
                       "date": pool["date"].to_numpy()}).sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    rows = []
    for lag in lags:
        shifted = df.groupby("ticker")["up"].shift(-lag)
        sub = pd.DataFrame({"x": df["x"]}, index=df.index)
        tmp = pd.DataFrame({"date": pd.DatetimeIndex(df["date"]), "up": shifted.to_numpy()})
        ok = np.isfinite(shifted.to_numpy()) & np.isfinite(df["x"].to_numpy())
        if ok.sum() < min_rows:
            rows.append(dict(lag=lag, n=int(ok.sum()), mean_ic=float("nan"), t=float("nan")))
            continue
        t = feature_ic_table(sub[ok].reset_index(drop=True), tmp[ok].reset_index(drop=True), "up", min_rows=10)
        rows.append(dict(lag=lag, n=int(ok.sum()), mean_ic=float(t["mean_ic"].iloc[0]) if len(t) else float("nan"),
                         t=float(t["t"].iloc[0]) if len(t) else float("nan")))
    return pd.DataFrame(rows, columns=cols)


def profile_verdict(profile: pd.DataFrame, t_crit: float = 2.0, t_crit_neg: float = 3.5) -> dict[str, Any]:
    """Reads a lead_lag_profile: 'forecast' when lag 0 is significant and no negative lag is, 'contaminated' when a negative lag is
    significant with the same sign (the feature knows the past outcome), 'trait' when every lag carries the same sign and size,
    'none' when nothing is significant."""
    if profile.empty or 0 not in set(profile["lag"]):
        return dict(kind="unknown", detail="no lag-0 estimate")
    p = profile.set_index("lag")
    t0 = p.loc[0, "t"]
    neg = [l for l in p.index if l < 0 and np.isfinite(p.loc[l, "t"]) and abs(p.loc[l, "t"]) > t_crit_neg]      # stricter: several negative lags are tried
    pos = [l for l in p.index if l > 0 and np.isfinite(p.loc[l, "t"]) and abs(p.loc[l, "t"]) > t_crit]
    if not np.isfinite(t0) or abs(t0) <= t_crit:
        return dict(kind="none" if not neg else "contaminated", detail=f"lag-0 t {t0:.2f}", lags=neg)
    if neg and all(np.sign(p.loc[l, "mean_ic"]) == np.sign(p.loc[0, "mean_ic"]) for l in neg):
        return dict(kind="contaminated", detail=f"significant at negative lags {neg}", lags=neg)
    if len(pos) >= 2 and all(np.sign(p.loc[l, "mean_ic"]) == np.sign(p.loc[0, "mean_ic"]) for l in pos):
        return dict(kind="trait", detail=f"persists at lags {pos}: a stock characteristic, not a weekly forecast", lags=pos)
    return dict(kind="forecast", detail=f"lag-0 t {t0:.2f}, clean elsewhere", lags=[])


def time_shuffle_control(M: pd.DataFrame, pool: pd.DataFrame, seed: int = 0, n_rep: int = 5) -> dict[str, float]:
    """Placebo: permute each feature's cross-sections across WEEKS (the week's whole feature vector is given to another week,
    cross-sectional structure preserved, link to that week's outcome destroyed) and recompute the IC table. The largest |t| over
    all columns and repetitions is what the IC screen reports for features that cannot possibly predict; it calibrates the
    threshold that real columns must beat and shows the screen's size."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pool["date"])
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < 20 or M.shape[1] == 0:
        return dict(max_abs_t=float("nan"), share_significant=float("nan"), n_tests=0)
    pos = {d: np.flatnonzero(dates == d) for d in uniq}
    maxt, sig, tot = [], 0, 0
    for _ in range(n_rep):
        perm = rng.permutation(len(uniq))
        idx = np.arange(len(pool))
        for a, b in zip(uniq, uniq[perm]):
            ra, rb = pos[a], pos[b]
            k = min(len(ra), len(rb))
            idx[ra[:k]] = rb[:k]
        t = feature_ic_table(M.iloc[idx].reset_index(drop=True), pool.reset_index(drop=True), "up")
        if len(t):
            maxt.append(float(t["t"].abs().max()))
            sig += int((t["p_two"] < 0.05).sum())
            tot += len(t)
    return dict(max_abs_t=float(np.max(maxt)) if maxt else float("nan"), share_significant=sig / tot if tot else float("nan"), n_tests=tot)


def placebo_date_shift(pred: pd.DataFrame, model: str, shift: int = 1, seed: int = 0, n_boot: int = 200) -> dict[str, float]:
    """Predictions from week w scored against the outcomes of week w+shift (same row order inside the week). A model that has
    learned the week's drift, or one whose outcomes are contaminated by a persistent stock trait, keeps its skill; a model that
    forecasts THIS week's direction loses it. Returns skill at shift 0 and at `shift`, with the bootstrap interval of the drop."""
    d = pred[pred["model"] == model].sort_values(["date", "row"])
    if len(d) < 200 or d["date"].nunique() < shift + 5:
        return dict(skill_true=float("nan"), skill_placebo=float("nan"), drop=float("nan"), lo=float("nan"), hi=float("nan"))
    wk = pd.factorize(d["date"], sort=True)[0]
    p, up, p0 = d["p"].to_numpy(), d["up"].to_numpy(), d["p0"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(wk)) + 1]
    ends = np.r_[starts[1:], len(wk)]
    plc = np.full(len(d), np.nan)
    for i, (a, b) in enumerate(zip(starts, ends)):
        if i + shift < len(starts):
            a2, b2 = starts[i + shift], ends[i + shift]
            k = min(b - a, b2 - a2)
            plc[a:a + k] = up[a2:a2 + k]
    ok = np.isfinite(plc)
    if ok.sum() < 100:
        return dict(skill_true=float("nan"), skill_placebo=float("nan"), drop=float("nan"), lo=float("nan"), hi=float("nan"))
    gain_true = (p0[ok] - up[ok]) ** 2 - (p[ok] - up[ok]) ** 2
    gain_pl = (p0[ok] - plc[ok]) ** 2 - (p[ok] - plc[ok]) ** 2
    diff = pd.Series(gain_true - gain_pl, index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(d["date"].to_numpy()[ok]), np.arange(ok.sum())]))
    bs = DA.week_bootstrap(diff, n_boot, seed)
    return dict(skill_true=brier_skill(p[ok], up[ok], p0[ok]), skill_placebo=brier_skill(p[ok], plc[ok], p0[ok]), drop=bs["mean"], lo=bs["lo"], hi=bs["hi"])


# ---------------------------------------------------------------------------------------------- pooling across years
def random_effects(effects: Sequence[float], variances: Sequence[float]) -> dict[str, float]:
    """DerSimonian-Laird random-effects pooling of per-year effects. Reports the pooled effect and its interval, tau^2 (real
    between-year variation), I^2 (share of variation that is heterogeneity) and a 95% PREDICTION interval for the effect in a new
    year: for a trading rule the question is not the average of the past but whether next year's effect can be negative."""
    e, v = np.asarray(effects, float), np.asarray(variances, float)
    ok = np.isfinite(e) & np.isfinite(v) & (v > 0)
    e, v = e[ok], v[ok]
    k = len(e)
    if k < 2:
        return dict(k=k, pooled=float(e[0]) if k else float("nan"), lo=float("nan"), hi=float("nan"), tau2=float("nan"), i2=float("nan"),
                    pred_lo=float("nan"), pred_hi=float("nan"), p_negative_year=float("nan"))
    w = 1 / v
    fixed = float((w * e).sum() / w.sum())
    q = float((w * (e - fixed) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max((q - (k - 1)) / c, 0.0) if c > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = float((ws * e).sum() / ws.sum())
    se = float(math.sqrt(1 / ws.sum()))
    i2 = max((q - (k - 1)) / q, 0.0) if q > 0 else 0.0
    tcrit = stats.t.ppf(0.975, max(k - 2, 1))
    psd = math.sqrt(tau2 + se ** 2)
    return dict(k=k, pooled=mu, lo=mu - 1.96 * se, hi=mu + 1.96 * se, tau2=float(tau2), i2=float(i2), pred_lo=float(mu - tcrit * psd),
                pred_hi=float(mu + tcrit * psd), p_negative_year=float(stats.norm.cdf(-mu / psd)) if psd > 0 else float(mu < 0))


def yearly_skill_effects(pred: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-year mean Brier gain over the base rate and the variance of that mean (week-cluster aware: the variance is taken over
    weekly means), the inputs of random_effects."""
    d = pred[pred["model"] == model]
    eff, var = [], []
    for y, g in d.groupby("year"):
        gain = (g["p0"] - g["up"]) ** 2 - (g["p"] - g["up"]) ** 2
        wk = gain.groupby(g["date"]).mean()
        if len(wk) >= 8:
            eff.append(float(wk.mean()))
            var.append(float(wk.var(ddof=1) / len(wk)))
    return np.asarray(eff), np.asarray(var)


def beta_posterior(k: float, n: float, prior: tuple[float, float] = (1.0, 1.0)) -> dict[str, float]:
    """Beta posterior of a hit rate with a flat prior: mean and 90% credible interval. Unlike a bare percentage it says how little a
    handful of bets can tell (5 for 5 has a 90% lower bound near 0.6, not 1.0)."""
    a, b = prior[0] + k, prior[1] + max(n - k, 0)
    return dict(mean=float(a / (a + b)), lo=float(stats.beta.ppf(0.05, a, b)), hi=float(stats.beta.ppf(0.95, a, b)), n=float(n))


def prob_accuracy_above(k: float, n: float, threshold: float = EIGHTY, prior: tuple[float, float] = (1.0, 1.0)) -> float:
    """Posterior probability that the true hit rate exceeds `threshold`. For the 80% question this is the honest single number for one
    cell; the multiplicity price is applied on top by the lab (family-wise tests), not hidden here."""
    return float(stats.beta.sf(threshold, prior[0] + k, prior[1] + max(n - k, 0)))


def breakeven_cost_bp(pred: pd.DataFrame, pool: pd.DataFrame, model: str, coverage: float = 1.0) -> dict[str, float]:
    """Round-trip cost at which the calls' mean net return crosses zero, and the mean gross edge. A directional edge smaller than the
    cost of acting on it is not an edge; this puts the two on one scale (basis points per bet)."""
    d = pred[(pred["model"] == model) & (pred["q"] <= coverage + 1e-12)]
    if len(d) < 50:
        return dict(gross_bp=float("nan"), breakeven_round_trip_bp=float("nan"), n=len(d))
    fwd = pool["fwd"].to_numpy(float)[d["row"].to_numpy()]
    ok = np.isfinite(fwd)
    fwd, side = fwd[ok], np.where(d["p"].to_numpy()[ok] >= 0.5, 1.0, -1.0)
    gross = float((side * fwd).mean() * 1e4)
    return dict(gross_bp=gross, breakeven_round_trip_bp=gross, n=int(len(fwd)))


def label_noise_attenuation(skill: float, flip: float) -> float:
    """Skill left after a fraction `flip` of labels is wrong: probabilities move (1 - 2*flip) of the way, and Brier gain scales with
    the square of that. Says how much of an observed null could be the label rather than the market (barrier labels on ambiguous
    same-session touches, for instance)."""
    if not 0 <= flip < 0.5:
        raise ValueError("flip must lie in [0, 0.5)")
    return float(skill * (1 - 2 * flip) ** 2)


# ---------------------------------------------------------------------------------------------- anytime-valid evidence
class EvidenceLedger:
    """Sequential evidence per hypothesis that stays valid however often the lab is re-run. Each fresh look contributes a p-value;
    it is turned into an e-value with the calibrator e(p) = kappa * p^(kappa-1) (kappa = 0.5) and e-values MULTIPLY across looks
    (a test supermartingale under the null). Ville's inequality then gives a rule that never exceeds level alpha in total, even with
    optional stopping: reject when the running product exceeds 1/alpha. Looks that are the same data again (same key) are ignored,
    because repeating a test on identical data is not new evidence."""

    def __init__(self, alpha: float = 0.05, kappa: float = 0.5):
        if not 0 < kappa < 1 or not 0 < alpha < 0.5:
            raise ValueError("EvidenceLedger: kappa in (0,1), alpha in (0, 0.5)")
        self.alpha, self.kappa = alpha, kappa
        self.log_e: dict[str, float] = {}
        self.looks: dict[str, int] = {}
        self.seen: set = set()

    def e_value(self, p: float) -> float:
        p = min(max(float(p), 1e-12), 1.0)
        return float(self.kappa * p ** (self.kappa - 1))

    def update(self, hypothesis: str, p: float, key: str) -> float:
        """Fold one look in; returns the running e-value. A repeated key is a no-op."""
        tag = (hypothesis, key)
        if tag in self.seen:
            return self.value(hypothesis)
        self.seen.add(tag)
        self.log_e[hypothesis] = self.log_e.get(hypothesis, 0.0) + math.log(self.e_value(p))
        self.looks[hypothesis] = self.looks.get(hypothesis, 0) + 1
        return self.value(hypothesis)

    def value(self, hypothesis: str) -> float:
        return float(math.exp(min(self.log_e.get(hypothesis, 0.0), 700.0)))

    def rejects(self, hypothesis: str) -> bool:
        return self.value(hypothesis) >= 1 / self.alpha

    def family_rejects(self) -> list[str]:
        """Hypotheses rejected with the family's alpha split evenly (Bonferroni over the hypotheses seen so far), still anytime-valid."""
        m = max(len(self.log_e), 1)
        return sorted(h for h in self.log_e if self.value(h) >= m / self.alpha)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([dict(hypothesis=h, looks=self.looks[h], e_value=self.value(h), rejects=self.rejects(h))
                             for h in sorted(self.log_e)], columns=["hypothesis", "looks", "e_value", "rejects"])


def feed_ledger(ledger: EvidenceLedger, report: LabReport) -> EvidenceLedger:
    """One report into the ledger, keyed by the report's own content so a rerun is not double counted. Hypotheses that were not
    tested (unavailable, gate closed, insufficient data) contribute nothing rather than a p of 1."""
    for name, r in report.results.items():
        if r.outcome in (Outcome.UNAVAILABLE_INPUT, Outcome.VOLATILITY_GATE_CLOSED, Outcome.INSUFFICIENT_DATA, Outcome.CONTROL_FAILURE, Outcome.LEAK_SUSPECT):
            continue
        ledger.update(name, r.p_maxT if np.isfinite(r.p_maxT) else 1.0, stable_hash([report.now, report.latest_label_end, report.config_hash, name], 12))
    return ledger


def weeks_needed(target_edge: float, base_rate: float = 0.5, picks_per_week: int = 15, design_effect: float = 1.0, alpha: float = 0.05,
                 power: float = 0.8) -> int:
    """Decision weeks required before an accuracy edge of `target_edge` over the majority-class rate could be detected: the planning
    number that turns 'underpowered' into a date. Uses the normal approximation on effective independent bets."""
    if target_edge <= 0:
        raise ValueError("target_edge must be positive")
    sd = math.sqrt(max(base_rate * (1 - base_rate), 1e-9))
    n_eff = ((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * sd / target_edge) ** 2
    return int(math.ceil(n_eff * design_effect / max(picks_per_week, 1)))


# ---------------------------------------------------------------------------------------------- robustness of a lead
def jackknife_weeks(pred: pd.DataFrame, model: str, drop_best: Sequence[int] = (1, 3, 10)) -> pd.DataFrame:
    """Skill after removing the k BEST weeks (by weekly Brier gain) and after removing the k best tickers' rows. An edge that
    disappears without its three luckiest weeks is a few events, not a rule; a healthy edge loses only a little. Returns one row
    per k with the remaining skill and the share of the original skill kept."""
    d = pred[pred["model"] == model]
    cols = ["k", "weeks_left", "skill_left", "share_kept"]
    if len(d) < 200:
        return pd.DataFrame(columns=cols)
    d = d.assign(gain=(d["p0"] - d["up"]) ** 2 - (d["p"] - d["up"]) ** 2)
    wk = d.groupby("date")["gain"].sum().sort_values(ascending=False)
    full = brier_skill(d["p"].to_numpy(), d["up"].to_numpy(), d["p0"].to_numpy())
    rows = []
    for k in drop_best:
        keep = ~d["date"].isin(wk.index[:k])
        g = d[keep]
        s = brier_skill(g["p"].to_numpy(), g["up"].to_numpy(), g["p0"].to_numpy()) if len(g) > 50 else float("nan")
        rows.append(dict(k=int(k), weeks_left=int(g["date"].nunique()), skill_left=s, share_kept=s / full if full and np.isfinite(s) else float("nan")))
    return pd.DataFrame(rows, columns=cols)


def pick_overlap(pool: pd.DataFrame) -> dict[str, float]:
    """Week-to-week persistence of the picks (share of this week's picks that were also picks last week) and the concentration of
    pick-weeks in the most frequent tickers. High persistence means successive weeks are not independent draws, so week-cluster
    intervals are still optimistic and the ticker-transfer probe matters more."""
    pk = pool[pool["pick"]]
    if pk["date"].nunique() < 3:
        return dict(overlap=float("nan"), top10_share=float("nan"), tickers=int(pk["ticker"].nunique()))
    by = {d: set(g["ticker"]) for d, g in pk.groupby("date")}
    ds = sorted(by)
    ov = [len(by[b] & by[a]) / max(len(by[b]), 1) for a, b in zip(ds[:-1], ds[1:])]
    cnt = pk["ticker"].value_counts()
    return dict(overlap=float(np.mean(ov)), top10_share=float(cnt.head(10).sum() / cnt.sum()), tickers=int(len(cnt)))


def column_ablation(pool: pd.DataFrame, M: np.ndarray, names: Sequence[str], cfg: LabConfig, top: int = 6, kind: str = "linear") -> pd.DataFrame:
    """Which INPUTS of a lead carry it? Refits the walk-forward for the full input set and for each single-input drop, and pairs
    the Brier differences over weeks (the direction_ablate week bootstrap). An input whose removal does not hurt is passenger; a
    lead whose skill lives in one input is that input's hypothesis, not the family's."""
    cols = ["column", "delta_brier", "lo", "hi", "verdict", "n"]
    if M.shape[1] < 2 or not cfg.test_years:
        return pd.DataFrame(columns=cols)
    F = {"full": M}
    keep = list(range(M.shape[1]))[:top]
    for j in keep:
        F[f"drop:{names[j]}"] = np.delete(M, j, axis=1)
    pred, _ = DF.walk_forward(pool, F, "up", cfg.test_years, models=(kind,), extra=None, seg_cols=(), embargo_days=cfg.embargo_days,
                              min_train=cfg.min_train, min_calib=cfg.min_calib, seed=cfg.seed, controls=False)
    if pred.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for j in keep:
        t = paired_test(pred, f"full:{kind}", f"drop:{names[j]}:{kind}", cfg.n_boot, cfg.alpha, cfg.seed)
        if t["n"] == 0:
            continue
        verdict = "carries" if t["hi"] < 0 else "harms" if t["lo"] > 0 else "passenger"
        rows.append(dict(column=names[j], delta_brier=t["mean_diff"], lo=t["lo"], hi=t["hi"], verdict=verdict, n=t["n"]))
    return pd.DataFrame(rows, columns=cols)


def segment_ic_matrix(M: pd.DataFrame, pool: pd.DataFrame, segcol: str = "seg", min_rows: int = 300) -> pd.DataFrame:
    """Mean weekly IC of each column inside each segment value (event type, regime or sector), with BH adjustment over the whole
    matrix. The interaction question 'does this signal work only somewhere?' asked with a multiplicity price attached."""
    rows = []
    for v in sorted(pool[segcol].dropna().unique()):
        m = (pool[segcol] == v).to_numpy()
        if m.sum() < min_rows:
            continue
        t = feature_ic_table(M[m].reset_index(drop=True), pool[m].reset_index(drop=True), "up", min_rows=10)
        for r in t.itertuples():
            rows.append(dict(segment=str(v), column=r.column, weeks=r.weeks, mean_ic=r.mean_ic, t=r.t, p_two=r.p_two))
    out = pd.DataFrame(rows, columns=["segment", "column", "weeks", "mean_ic", "t", "p_two"])
    out["p_bh"] = benjamini_hochberg(out["p_two"].to_numpy()) if len(out) else []
    return out


def rolling_ic(x: np.ndarray, up: np.ndarray, dates: Sequence, window: int = 26) -> pd.DataFrame:
    """Weekly IC of one feature and its rolling mean: does the signal come and go? Returns week, ic, rolling mean and the lag-1
    autocorrelation of the weekly IC (slow-moving regimes give positive autocorrelation, which also inflates naive t statistics)."""
    x, up = np.asarray(x, float), np.asarray(up, float)
    codes, uniq = pd.factorize(pd.DatetimeIndex(dates), sort=True)
    ic = np.full(len(uniq), np.nan)
    for w in range(len(uniq)):
        m = (codes == w) & np.isfinite(x) & np.isfinite(up)
        if m.sum() >= 10 and np.ptp(x[m]) > 0 and np.ptp(up[m]) > 0:
            ic[w] = stats.spearmanr(x[m], up[m])[0]
    s = pd.Series(ic, index=uniq)
    out = pd.DataFrame({"week": uniq, "ic": ic, "rolling": s.rolling(window, min_periods=max(window // 2, 4)).mean().to_numpy()})
    v = s.dropna()
    out.attrs["ic_autocorr"] = float(v.autocorr(1)) if len(v) > 8 else float("nan")
    return out


def evidence_dossier(report: LabReport, name: str, pred: pd.DataFrame | None = None, pool: pd.DataFrame | None = None) -> dict[str, Any]:
    """One hypothesis, everything known, in a plain dict ready for a report or a research question: the verdict and its reasons,
    each comparator's paired difference, stability numbers, payoff and calibration. With `pred`/`pool` it adds the robustness
    extras (random-effects pooling over years, jackknife, placebo shift, breakeven cost)."""
    r = report.results[name]
    d: dict[str, Any] = dict(hypothesis=name, topic=r.topic, outcome=str(r.outcome), state=str(r.state), stage=str(r.stage), reasons=list(r.reasons),
                             skill=r.skill, posterior_skill=r.posterior_skill, p_raw=r.p_raw, p_family=r.p_maxT, p_holm=r.p_holm, p_bh=r.p_bh,
                             comparators={c.value: dict(against=x.against, source=x.source, mean_diff=x.mean_diff, hi=x.hi, passed=x.passed)
                                          for c, x in r.comparators.items()}, era=r.era, regime=r.regime, event=r.event, sector=r.sector,
                             transfer=r.transfer, decay=r.decay, payoff=r.payoff_full, mechanism=r.mechanism, dropped=r.dropped)
    if pred is not None and r.best_tag:
        eff, var = yearly_skill_effects(pred, r.best_tag)
        d["random_effects"] = random_effects(eff, var)
        d["jackknife"] = jackknife_weeks(pred, r.best_tag).to_dict("records")
        d["placebo_shift"] = placebo_date_shift(pred, r.best_tag)
        if pool is not None:
            d["breakeven"] = breakeven_cost_bp(pred, pool, r.best_tag)
    return d


# ---------------------------------------------------------------------------------------------- open hypothesis registry
class HypothesisRegistry:
    """The catalogue is open (contract: 'H10+ open'): new hypotheses can be added by the research loop, but each one is validated,
    given a stable id, checked for redundancy with what exists (a new spec that shares most of its inputs with an old one is the
    same hypothesis under another name, and would silently inflate the multiplicity count while adding no information), and can be
    retired without deletion. The family size used for the multiplicity price is `active_count`, not len(HYPOTHESES)."""

    def __init__(self, base: Mapping[str, HypothesisSpec] | None = None, max_jaccard: float = 0.6):
        if not 0 < max_jaccard <= 1:
            raise ValueError("max_jaccard must lie in (0, 1]")
        self.max_jaccard = max_jaccard
        self.specs: dict[str, HypothesisSpec] = {}
        self.retired: dict[str, str] = {}
        self.log: list[dict[str, Any]] = []
        for s in (base or HYPOTHESES).values():
            self.specs[s.name] = s.validate()

    @staticmethod
    def jaccard(a: HypothesisSpec, b: HypothesisSpec) -> float:
        x, y = set(a.columns), set(b.columns)
        return len(x & y) / len(x | y) if x | y else 0.0

    def redundant_with(self, spec: HypothesisSpec) -> list[tuple[str, float]]:
        out = [(n, self.jaccard(spec, s)) for n, s in self.specs.items() if n not in self.retired]
        return sorted([(n, j) for n, j in out if j >= self.max_jaccard], key=lambda t: -t[1])

    def add(self, spec: HypothesisSpec, reason: str = "") -> str:
        """Register a new hypothesis; returns its id. Refuses duplicates by name and by near-identical input sets."""
        spec.validate()
        if spec.name in self.specs:
            raise ValueError(f"hypothesis {spec.name!r} already registered")
        if not reason:
            raise ValueError("a new hypothesis needs a stated reason (which question or surprise motivated it)")
        dup = self.redundant_with(spec)
        if dup:
            raise ValueError(f"hypothesis {spec.name!r} is redundant with {dup[0][0]} (Jaccard {dup[0][1]:.2f}); extend that one instead")
        self.specs[spec.name] = spec
        hid = "H" + stable_hash([spec.name, spec.columns, spec.prior_sign], 8)
        self.log.append(dict(action="add", name=spec.name, id=hid, reason=reason))
        return hid

    def retire(self, name: str, why: str) -> None:
        if name not in self.specs:
            raise KeyError(name)
        self.retired[name] = why
        self.log.append(dict(action="retire", name=name, reason=why))

    def active(self) -> dict[str, HypothesisSpec]:
        return {n: s for n, s in self.specs.items() if n not in self.retired}

    @property
    def active_count(self) -> int:
        return len(self.active())

    def coverage_gaps(self, available: Sequence[str]) -> list[str]:
        """Active hypotheses that cannot be tested with the panel's columns: what to build next."""
        return [n for n, s in self.active().items() if not resolve_columns(available, s)]


# ---------------------------------------------------------------------------------------------- pre-registration
@dataclasses.dataclass(frozen=True)
class PreRegistration:
    """A success criterion fixed BEFORE the confirming data are looked at. The hash covers the hypothesis, the model tag, the
    comparators, the alpha, the minimum edge and the period, so a replication cannot quietly move its own goalposts after seeing the
    result: verify() recomputes the hash and refuses a mismatch."""
    hypothesis: str
    best_tag: str
    comparators: tuple[str, ...]
    alpha: float
    min_skill: float
    fresh_after: str
    min_weeks: int
    registered_real: str
    digest: str = ""

    @staticmethod
    def make(hypothesis: str, best_tag: str, alpha: float, min_skill: float, fresh_after, min_weeks: int, registered_real,
             comparators: Sequence[Comparator] = ALL_COMPARATORS) -> "PreRegistration":
        if not (0 < alpha < 0.5 and min_skill >= 0 and min_weeks >= 8):
            raise ValueError("PreRegistration: alpha in (0, 0.5), min_skill >= 0, min_weeks >= 8")
        if as_date(fresh_after) >= as_date(registered_real):
            raise ValueError("fresh_after must precede the registration date so the period is genuinely after the lead's data")
        body = dict(h=hypothesis, t=best_tag, c=sorted(str(c) for c in comparators), a=alpha, m=min_skill, f=str(as_date(fresh_after)),
                    w=min_weeks, r=str(as_date(registered_real)))
        return PreRegistration(hypothesis, best_tag, tuple(sorted(str(c) for c in comparators)), alpha, min_skill, str(as_date(fresh_after)),
                               min_weeks, str(as_date(registered_real)), stable_hash(body, 16))

    def verify(self) -> bool:
        body = dict(h=self.hypothesis, t=self.best_tag, c=list(self.comparators), a=self.alpha, m=self.min_skill, f=self.fresh_after,
                    w=self.min_weeks, r=self.registered_real)
        return self.digest == stable_hash(body, 16)

    def evaluate(self, result: HypothesisResult, weeks_seen: int, latest_data_start: str) -> dict[str, Any]:
        """Judge a replication result against the frozen criterion. The data must start on/after `fresh_after`, cover min_weeks, be
        the same model, clear the alpha and skill floor, and beat every registered comparator."""
        if not self.verify():
            raise FirewallBreach("pre-registration digest mismatch: the criterion was edited after registration")
        why = []
        if as_date(latest_data_start) < as_date(self.fresh_after):
            why.append("replication data start before the fresh period: not independent of the lead")
        if weeks_seen < self.min_weeks:
            why.append(f"{weeks_seen} weeks < {self.min_weeks}")
        if result.best_tag != self.best_tag:
            why.append("a different model was evaluated than the one registered")
        if not result.p_raw < self.alpha:
            why.append(f"p {result.p_raw:.3f} >= {self.alpha}")
        if not result.skill >= self.min_skill:
            why.append(f"skill {result.skill:+.4f} < {self.min_skill}")
        lost = [c for c in self.comparators if c not in {k.value for k, v in result.comparators.items() if v.passed}]
        if lost:
            why.append("comparators not beaten: " + ",".join(lost))
        return dict(replicated=not why, reasons=why, digest=self.digest)


def plan_replication(report: LabReport, name: str, registered_real) -> PreRegistration | None:
    """For a WEAK_UNREPLICATED or CANDIDATE hypothesis: the pre-registered replication on data after the current sample. None for
    anything else (a null needs no replication; it needs more power, which the question generator asks for)."""
    r = report.results[name]
    if r.outcome not in (Outcome.WEAK_UNREPLICATED, Outcome.CANDIDATE) or not r.best_tag or not report.latest_label_end:
        return None
    need = max(int(report.power.get("weeks", 0)) // 2, 26)
    return PreRegistration.make(name, r.best_tag, 0.05, max(0.5 * r.skill, 0.002), report.latest_label_end, need, registered_real)


# ---------------------------------------------------------------------------------------------- payoff-relevant tests
def return_rank_test(pred: pd.DataFrame, pool: pd.DataFrame, model: str, n_boot: int = 300, seed: int = 0) -> dict[str, float]:
    """Does the model's probability rank the forward RETURN, not just the sign? Weekly Spearman of P(up) with the forward return
    among the week's picks, its t across weeks, and the top-minus-bottom-third return spread in basis points. A model can be
    right about the sign on the small moves and wrong on the large ones; rank IC against the return is the payoff-weighted view."""
    d = pred[pred["model"] == model]
    if len(d) < 200:
        return dict(ic=float("nan"), t=float("nan"), spread_bp=float("nan"), lo_bp=float("nan"), hi_bp=float("nan"), weeks=0)
    fwd = pool["fwd"].to_numpy(float)[d["row"].to_numpy()]
    df = pd.DataFrame({"w": d["date"].to_numpy(), "p": d["p"].to_numpy(), "r": fwd}).dropna()
    ics, top, bot = [], [], []
    for _, g in df.groupby("w"):
        if len(g) < 12 or g["p"].nunique() < 3:
            continue
        ics.append(float(stats.spearmanr(g["p"], g["r"])[0]))
        rk = g["p"].rank(pct=True, method="first")
        top.append(float(g.loc[rk > 2 / 3, "r"].mean()))
        bot.append(float(g.loc[rk <= 1 / 3, "r"].mean()))
    if len(ics) < 8:
        return dict(ic=float("nan"), t=float("nan"), spread_bp=float("nan"), lo_bp=float("nan"), hi_bp=float("nan"), weeks=len(ics))
    ics, sp = np.asarray(ics), (np.asarray(top) - np.asarray(bot)) * 1e4
    rng = np.random.default_rng(seed)
    bs = sp[rng.integers(0, len(sp), size=(n_boot, len(sp)))].mean(axis=1)
    sd = ics.std(ddof=1)
    return dict(ic=float(ics.mean()), t=float(ics.mean() / (sd / math.sqrt(len(ics)))) if sd > 0 else 0.0, spread_bp=float(sp.mean()),
                lo_bp=float(np.quantile(bs, 0.05)), hi_bp=float(np.quantile(bs, 0.95)), weeks=len(ics))


def selection_stability(W: pd.DataFrame, info: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """How often is each candidate the best when the weeks are resampled? W is rows x candidate probabilities on shared rows. With no
    real winner the crown is spread thinly over many cells; a real one wins most resamples. The best-looking cell's win share is
    the cheapest guard against reporting a lucky leader. Also gives each cell's mean rank."""
    if W.empty or W.shape[1] < 2:
        return pd.DataFrame(columns=["cell", "win_share", "mean_rank"])
    y = info["up"].to_numpy(float)
    wk, uniq = pd.factorize(info["date"].to_numpy())
    loss = (W.to_numpy(float) - y[:, None]) ** 2
    S = np.zeros((len(uniq), W.shape[1]))
    np.add.at(S, wk, loss)
    N = np.bincount(wk, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    wins, ranks = np.zeros(W.shape[1]), np.zeros(W.shape[1])
    for _ in range(n_boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        m = S[idx].sum(0) / N[idx].sum()
        wins[np.argmin(m)] += 1
        ranks += stats.rankdata(m)
    return pd.DataFrame({"cell": W.columns, "win_share": wins / n_boot, "mean_rank": ranks / n_boot}).sort_values("win_share", ascending=False).reset_index(drop=True)


def frontier_monotone_check(ft: pd.DataFrame, model: str, min_n: int = 30) -> dict[str, Any]:
    """A usable frontier has accuracy that does not fall as coverage shrinks. Reads the overall (segment-free) cells for one model
    and reports the coverage-accuracy Spearman, the largest violation, and whether the tightest cell is the most accurate. A
    frontier that rises and falls at random is noise; one that is flat at the base rate says confidence carries nothing."""
    d = ft[(ft["model"] == model) & ft["segcol"].isna() & (ft["n"] >= min_n)].sort_values("cov_target")
    if len(d) < 3:
        return dict(rho=float("nan"), worst_violation=float("nan"), tightest_is_best=False, cells=len(d), flat=True)
    rho = float(stats.spearmanr(d["cov_target"], d["acc"])[0]) if d["acc"].nunique() > 1 else 0.0
    acc = d["acc"].to_numpy()
    viol = float(max(np.max(acc[1:] - acc[:-1]) if len(acc) > 1 else 0.0, 0.0))
    return dict(rho=rho, worst_violation=viol, tightest_is_best=bool(acc[0] >= acc.max() - 1e-12), cells=len(d),
                flat=bool(np.ptp(acc) < 0.02))


def compare_reports(a: LabReport, b: LabReport) -> pd.DataFrame:
    """Outcome drift between two looks (earlier a, later b): which hypotheses changed class, and by how much their skill and family
    p moved. Flips from lead to null on more data are the normal fate of a false lead and are recorded as such."""
    rows = []
    for h in sorted(set(a.results) | set(b.results)):
        x, y = a.results.get(h), b.results.get(h)
        rows.append(dict(hypothesis=h, before=str(x.outcome) if x else "absent", after=str(y.outcome) if y else "absent",
                         d_skill=(y.skill - x.skill) if x and y else float("nan"), d_p_family=(y.p_maxT - x.p_maxT) if x and y else float("nan"),
                         lead_lost=bool(x and y and x.outcome is Outcome.WEAK_UNREPLICATED and y.outcome is Outcome.NO_RELIABLE_SIGNAL),
                         lead_gained=bool(x and y and x.outcome is Outcome.NO_RELIABLE_SIGNAL and y.outcome in (Outcome.WEAK_UNREPLICATED, Outcome.CANDIDATE))))
    return pd.DataFrame(rows)


def miner_extra(inputs: LabInputs, pool: pd.DataFrame, cfg: LabConfig) -> dict[str, Callable]:
    """The pattern miner as a direction model for the pattern_combinations hypothesis: a signed, past-only score from
    direction_features.miner_hook (fit on the TRAIN rows of each block with `now` = the day before calibration). Only built when
    cfg.use_miner is set, because it is the slowest model in the lab; returns {} otherwise."""
    if not cfg.use_miner:
        return {}
    Xm = inputs.X.reindex(pool.index)
    ydir = np.where(np.isfinite(pool["up"].to_numpy(float)), np.where(pool["up"].to_numpy(float) > 0.5, 1.0, -1.0), np.nan)
    return {"miner:pattern": DF.miner_hook(Xm, ydir, pd.DatetimeIndex(pool["date"]).to_numpy())}


def robust_verdict(dossier: Mapping[str, Any], min_kept: float = 0.5) -> dict[str, Any]:
    """Second opinion on a lead from the robustness extras in evidence_dossier: the random-effects prediction interval must not
    include a negative next-year effect with high probability, the jackknife must keep at least half the skill, the placebo shift
    must cost skill, and the mechanism checks must not contradict. Returns per-check pass/fail and an overall 'robust'."""
    re_ = dossier.get("random_effects", {})
    jk = dossier.get("jackknife", [])
    pl = dossier.get("placebo_shift", {})
    mech = dossier.get("mechanism", {})
    checks = {
        "next_year_positive": bool(re_) and np.isfinite(re_.get("p_negative_year", np.nan)) and re_["p_negative_year"] < 0.2,
        "survives_dropping_best_weeks": bool(jk) and all(np.isfinite(r["share_kept"]) and r["share_kept"] >= min_kept for r in jk),
        "placebo_shift_costs_skill": bool(pl) and np.isfinite(pl.get("lo", np.nan)) and pl["lo"] > 0,
        "mechanism_not_contradicted": not mech or mech.get("inconsistent", 0) <= mech.get("consistent", 0),
    }
    return dict(checks=checks, robust=all(checks.values()), failed=[k for k, v in checks.items() if not v])


# ---------------------------------------------------------------------------------------------- point-in-time audit of the lab's own features
def truncation_test(inputs: LabInputs, cfg: LabConfig, now, cut, tol: float = 1e-6) -> dict[str, Any]:
    """The strongest leak test a feature pipeline can take: build the derived features from data truncated at `cut` and from the
    full panel, and compare every value dated before `cut`. If a feature at a past date changes when later rows are removed, it
    read the future (a cross-sectional rank across dates, a rolling window centred on the row, a leave-one-out that spans weeks).
    Returns the offending columns with the largest absolute change; an empty list is the only passing answer."""
    cut_ts = pd.Timestamp(as_date(cut))
    uni = ConditionalUniverse(cfg).build(inputs, now)
    pool = uni.pool
    if pool.empty:
        return dict(ok=True, offenders=[], rows=0, columns=0)
    dates = pd.DatetimeIndex(pool["date"])
    early = (dates < cut_ts)
    X_full = inputs.X.reindex(pool.index)
    full = derive_features(X_full, pool, inputs.pattern_score)
    p_cut = pool[early]
    part = derive_features(inputs.X.reindex(p_cut.index), p_cut, inputs.pattern_score)
    common = [c for c in part.columns if c in full.columns]
    a, b = full.loc[p_cut.index, common].to_numpy(float), part[common].to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    delta = np.where(both, np.abs(a - b), 0.0)
    nan_mismatch = (np.isfinite(a) != np.isfinite(b)).sum(axis=0)
    worst = delta.max(axis=0) if len(delta) else np.zeros(len(common))
    off = [dict(column=c, max_abs_change=float(w), nan_mismatch=int(m)) for c, w, m in zip(common, worst, nan_mismatch) if w > tol or m > 0]
    return dict(ok=not off, offenders=sorted(off, key=lambda r: -r["max_abs_change"]), rows=int(early.sum()), columns=len(common))


def audit_pool(pool: pd.DataFrame) -> dict[str, Any]:
    """Structural audit of the direction pool: unique (date, ticker), entry strictly after the decision, outcome window closed
    before use, picks a subset of the pool, at most one pick-set per week and consistent ranks."""
    bad = []
    if pool.empty:
        return dict(ok=True, violations=[], rows=0)
    key = pd.MultiIndex.from_arrays([pool["date"], pool["ticker"]])
    if key.has_duplicates:
        bad.append("duplicated (date, ticker) rows")
    lab = pool.dropna(subset=["entry_date"])
    if len(lab) and not (pd.DatetimeIndex(lab["entry_date"]) > pd.DatetimeIndex(lab["date"])).all():
        bad.append("an entry session is not after its decision date")
    if len(lab) and not (pd.DatetimeIndex(lab["end_date"]) > pd.DatetimeIndex(lab["entry_date"])).all():
        bad.append("an outcome window ends before it begins")
    if (pool["pick"] & ~pool["pool"]).any():
        bad.append("a pick outside the pool")
    g = pool.groupby("date")
    if (g["rank"].min() != 1).any():
        bad.append("a week whose best rank is not 1")
    if (g["pick"].sum() > g["pool"].sum()).any():
        bad.append("more picks than pool rows in a week")
    unl = pool["up"].isna() & pool["end_date"].notna()
    if unl.any():
        bad.append(f"{int(unl.sum())} rows have a closed window but no label")
    return dict(ok=not bad, violations=bad, rows=len(pool))


def sector_neutralize(M: pd.DataFrame, pool: pd.DataFrame, min_group: int = 3) -> pd.DataFrame:
    """Demean each column within (week, sector) among the pool rows (groups smaller than `min_group`, and the 'na' sector, are
    demeaned against the whole week). Separates 'this name beats its industry' from 'its industry moved', which the raw
    hypotheses mix; a signal that vanishes here was a sector bet."""
    d = pd.DataFrame({"d": pool["date"].to_numpy(), "s": pool["sector"].to_numpy()}, index=M.index)
    out = M.copy()
    grp = d.groupby(["d", "s"])["s"].transform("size")
    for c in M.columns:
        x = M[c].astype(float)
        by_grp = x - x.groupby([d["d"], d["s"]]).transform("mean")
        by_wk = x - x.groupby(d["d"]).transform("mean")
        use = (grp >= min_group) & (d["s"] != "na")
        out[c] = np.where(use, by_grp, by_wk)
    return out


def regime_switch_hook(M: np.ndarray, reg: np.ndarray, yv: np.ndarray, seed: int = 0, min_rows: int = 400) -> Callable:
    """A regime-conditional model for walk_forward's `extra`: one linear model per market regime (bull/bear) fitted on the TRAIN rows
    of that regime only; rows of a regime with too little training data fall back to the pooled model. It is the direct test of
    'the sign of the effect depends on the regime', a claim the pooled models cannot express."""
    def fn(tr, ca, te):
        pooled = DF.DirModel("linear", seed).fit(M[tr], yv[tr])
        models = {}
        for v in np.unique(reg[tr]):
            m = tr[reg[tr] == v]
            if len(m) >= min_rows and len(np.unique(yv[m])) == 2:
                models[v] = DF.DirModel("linear", seed).fit(M[m], yv[m])

        def score(rows):
            out = pooled.prob(M[rows])
            for v, mdl in models.items():
                sel = reg[rows] == v
                if sel.any():
                    out[sel] = mdl.prob(M[rows[sel]])
            return out
        return score(ca), score(te), True
    return fn


def honest_blend(W: pd.DataFrame, info: pd.DataFrame, cells: Sequence[str] | None = None) -> dict[str, Any]:
    """Would combining the leaders help? Non-negative weights are fitted on the FIRST half of the rows (by date) by projected least
    squares on the Brier loss and judged on the SECOND half only, against the equal-weight blend and the best single cell chosen on
    the first half. A blend that only wins in-sample is over-fitting the winner's curse."""
    cells = list(cells or W.columns)
    if len(cells) < 2:
        return dict(ok=False, reason="need at least two cells")
    order = np.argsort(pd.DatetimeIndex(info["date"]).to_numpy(), kind="stable")
    P, y = W[cells].to_numpy(float)[order], info["up"].to_numpy(float)[order]
    h = len(y) // 2
    A, b = P[:h], y[:h]
    w = np.full(len(cells), 1 / len(cells))
    lr = 1.0 / max(np.linalg.norm(A, 2) ** 2 / len(A), 1e-9)
    for _ in range(500):
        w = np.clip(w - lr * (A.T @ (A @ w - b)) / len(A), 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(len(cells), 1 / len(cells))
    best = int(np.argmin(((A - b[:, None]) ** 2).mean(axis=0)))
    te_y, te_P = y[h:], P[h:]
    loss = lambda p: float(np.mean((p - te_y) ** 2))
    base = float(np.mean((np.full(len(te_y), b.mean()) - te_y) ** 2))
    fitted, equal, single = loss(te_P @ w), loss(te_P.mean(axis=1)), loss(te_P[:, best])
    return dict(ok=True, weights=dict(zip(cells, np.round(w, 4))), loss_fitted=fitted, loss_equal=equal, loss_best_single=single, loss_base=base,
                skill_fitted=1 - fitted / base, skill_equal=1 - equal / base, skill_single=1 - single / base, best_single=cells[best],
                fitted_beats_equal=bool(fitted < equal), rows_test=int(len(te_y)))


# ---------------------------------------------------------------------------------------------- persistence and health
def report_to_dict(report: LabReport) -> dict[str, Any]:
    """Plain, JSON-safe summary of a run (no frames): enough to compare looks later (compare_summaries) and to rebuild the outcome
    table. Numbers stay numbers; enums become their string values."""
    def num(x):
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x
    return dict(now=report.now, config_hash=report.config_hash, code_hash=report.code_hash, gate_open=report.gate.is_open,
                gate_reasons=list(report.gate.reasons), protocol_ok=report.protocol_ok, n_cells=report.n_cells, years=list(report.years),
                latest_label_end=report.latest_label_end, statement=report.statement, headline=report.headline(),
                power={k: num(float(v)) for k, v in report.power.items()},
                hypotheses={n: dict(outcome=str(r.outcome), state=str(r.state), stage=str(r.stage), best=r.best_tag, skill=num(r.skill),
                                    p_raw=num(r.p_raw), p_family=num(r.p_maxT), beat=r.comparators_passed, reasons=list(r.reasons))
                            for n, r in report.results.items()})


def compare_summaries(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The same drift question as compare_reports but from stored summaries, so a look from last month can be compared with today's
    without keeping its frames."""
    out = []
    for h in sorted(set(old.get("hypotheses", {})) | set(new.get("hypotheses", {}))):
        a, b = old.get("hypotheses", {}).get(h), new.get("hypotheses", {}).get(h)
        out.append(dict(hypothesis=h, before=a["outcome"] if a else "absent", after=b["outcome"] if b else "absent",
                        changed=bool(a and b and a["outcome"] != b["outcome"]),
                        d_skill=(b["skill"] - a["skill"]) if a and b and a["skill"] is not None and b["skill"] is not None else None))
    return out


def lab_health(report: LabReport, cfg: LabConfig | None = None) -> dict[str, Any]:
    """The lab's own vital signs for the research-brain health system (contract 37): did its controls pass, was it powered, did it
    leave any hypothesis untestable, how many cells did it price, and did the honest negative statement get produced. A lab that
    is unhealthy must not have its nulls counted as evidence of anything."""
    n = max(len(report.results), 1)
    untestable = sum(1 for r in report.results.values() if r.outcome in (Outcome.UNAVAILABLE_INPUT, Outcome.INSUFFICIENT_DATA))
    powered = report.power.get("mde_edge", 1.0) <= 0.03 if report.power else False
    flags = dict(gate_open=report.gate.is_open, controls_pass=report.protocol_ok, powered=bool(powered), untestable_share=untestable / n,
                 leak_suspects=sum(1 for r in report.results.values() if r.outcome is Outcome.LEAK_SUSPECT), cells_priced=report.n_cells,
                 statement_present=bool(report.statement), comparators_full=all(len(r.comparators) == len(ALL_COMPARATORS)
                                                                                 for r in report.results.values() if r.stage is not Stage.CHEAP_SCREEN and r.comparators))
    ok = flags["gate_open"] and flags["controls_pass"] and flags["untestable_share"] <= 0.5 and flags["statement_present"]
    flags["healthy"] = bool(ok)
    flags["nulls_count_as_evidence"] = bool(ok and powered)
    return flags


# ---------------------------------------------------------------------------------------------- baselines and false-lead arithmetic
def baseline_table(Xp: pd.DataFrame, pool: pd.DataFrame, years: Sequence[int], rules: Sequence[tuple[str, int]] = SIMPLE_RULES) -> pd.DataFrame:
    """Every fixed one-line rule of the literature scored on the same test-year picks as the lab: accuracy of 'sign * feature above its
    week median => up', its Wilson interval, the majority-class rate, and its edge. This is the bar a fitted model has to clear
    (the SIMPLE_BASELINE comparator picks the best of these on the calibration block; this table shows all of them on the test block).
    No parameter is fitted, so there is nothing to over-fit and the multiplicity is just the number of rules."""
    cols = ["rule", "sign", "n", "acc", "wilson_lo", "wilson_hi", "majority", "edge", "years"]
    te = pool["pick"].to_numpy(bool) & pool["year"].isin(list(years)).to_numpy() & pool["up"].notna().to_numpy()
    if not te.any():
        return pd.DataFrame(columns=cols)
    up = pool["up"].to_numpy(float)
    rows = []
    for c, s in rules:
        if c not in Xp.columns:
            continue
        x = pd.Series(s * Xp[c].to_numpy(float))
        med = x.groupby(pool["date"].to_numpy()).transform("median").to_numpy()
        ok = te & np.isfinite(x.to_numpy()) & np.isfinite(med)
        if ok.sum() < 100:
            continue
        call = (x.to_numpy()[ok] > med[ok]).astype(float)
        hits = float((call == up[ok]).sum())
        lo, hi = LC.wilson(hits, int(ok.sum()), z=1.96)
        maj = float(max(up[ok].mean(), 1 - up[ok].mean()))
        rows.append(dict(rule=c, sign=s, n=int(ok.sum()), acc=hits / ok.sum(), wilson_lo=lo, wilson_hi=hi, majority=maj, edge=hits / ok.sum() - maj,
                         years=int(pool.loc[ok, "year"].nunique())))
    return pd.DataFrame(rows, columns=cols)


def expected_false_leads(n_hypotheses: int, screen_p: float, alpha: float) -> dict[str, float]:
    """Arithmetic of the two-stage funnel under the null: a hypothesis survives the screen with probability `screen_p`, and then
    shows an unadjusted lead with probability `alpha`; the family-wise rule cuts the second stage to alpha / m at worst. Gives the
    expected number of WEAK leads and the chance of at least one, so a reader knows one lead in fifteen is roughly what luck
    delivers and is not a discovery by itself."""
    if n_hypotheses < 1 or not 0 < screen_p <= 1 or not 0 < alpha < 1:
        raise ValueError("expected_false_leads: need n >= 1, screen_p in (0,1], alpha in (0,1)")
    p_lead = screen_p * alpha
    p_cand = screen_p * alpha / n_hypotheses
    return dict(expected_weak=n_hypotheses * p_lead, p_any_weak=1 - (1 - p_lead) ** n_hypotheses, expected_candidate=n_hypotheses * p_cand,
                p_any_candidate=1 - (1 - p_cand) ** n_hypotheses)


def accuracy_by_vol_bucket(pred: pd.DataFrame, pool: pd.DataFrame, model: str, n_bins: int = 3) -> pd.DataFrame:
    """Direction accuracy inside terciles of the volatility score. If direction is easier where the volatility model is most
    confident (bigger, cleaner moves) the two stages interact and the gate should weight direction bets by P(vol); if it is
    flat, the stages are independent and can be tuned separately (section 35: two prediction problems, one pipeline)."""
    d = pred[pred["model"] == model]
    cols = ["bucket", "n", "acc", "wilson_lo", "wilson_hi", "up_rate", "mean_abs_return_bp"]
    if len(d) < n_bins * 60:
        return pd.DataFrame(columns=cols)
    sc = pool["score"].to_numpy(float)[d["row"].to_numpy()]
    fwd = pool["fwd"].to_numpy(float)[d["row"].to_numpy()]
    rk = pd.Series(sc).groupby(d["date"].to_numpy()).rank(pct=True, method="first").to_numpy()
    b = np.minimum((np.nan_to_num(rk, nan=0.0) * n_bins).astype(int), n_bins - 1)
    ok = ((d["p"].to_numpy() >= 0.5) == (d["up"].to_numpy() > 0.5))
    rows = []
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        lo, hi = LC.wilson(float(ok[m].sum()), int(m.sum()), z=1.96)
        rows.append(dict(bucket=k, n=int(m.sum()), acc=float(ok[m].mean()), wilson_lo=lo, wilson_hi=hi, up_rate=float(d["up"].to_numpy()[m].mean()),
                         mean_abs_return_bp=float(np.nanmean(np.abs(fwd[m])) * 1e4)))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------------- what to do next
STAGE_MINUTES = {Stage.CHEAP_SCREEN: 2.0, Stage.STRONGER_TESTS: 15.0, Stage.CROSS_YEAR: 30.0, Stage.FRESH_HOLDOUT: 45.0, Stage.INTEGRATION: 90.0}


def next_experiments(report: LabReport, state: LabState | None = None) -> list[dict[str, Any]]:
    """The lab's own suggestions to the priority engine, ranked by direction value per compute minute. A closed gate suggests only
    the volatility experiment that would reopen it; leads suggest a pre-registered replication; unavailable inputs suggest building
    them; powered nulls suggest moving on; underpowered ones suggest waiting a stated number of weeks."""
    out: list[dict[str, Any]] = []
    if not report.gate.is_open:
        return [dict(action="improve_volatility_evidence", target=None, why="; ".join(report.gate.reasons), value=1.0 * report.gate.priority_multiplier(),
                     minutes=STAGE_MINUTES[Stage.STRONGER_TESTS], stage=str(Stage.STRONGER_TESTS))]
    ledger = state.ledger if state is not None else None
    for n, r in report.results.items():
        ev = experiment_value(report, n)
        if r.outcome is Outcome.CANDIDATE:
            act, nxt = "fresh_holdout", Stage.FRESH_HOLDOUT
        elif r.outcome is Outcome.WEAK_UNREPLICATED:
            act, nxt = "preregistered_replication", Stage.CROSS_YEAR
        elif r.outcome is Outcome.UNAVAILABLE_INPUT:
            act, nxt = "build_point_in_time_input", Stage.CHEAP_SCREEN
        elif r.outcome is Outcome.LEAK_SUSPECT:
            act, nxt = "find_the_leak", Stage.CHEAP_SCREEN
        elif r.outcome is Outcome.NO_RELIABLE_SIGNAL and r.state is ResearchState.DORMANT:
            act, nxt = "wait_for_power", Stage.CHEAP_SCREEN
        else:
            continue
        val = float(ev.direction_value or 0.0) + (0.1 if act == "build_point_in_time_input" else 0.0)
        if ledger is not None and ledger.looks.get(n, 0) > 0:
            val += 0.2 * min(math.log10(max(ledger.value(n), 1.0)), 2.0)
        out.append(dict(action=act, target=n, why=(r.reasons[0] if r.reasons else ""), value=val, minutes=STAGE_MINUTES[nxt], stage=str(nxt),
                        per_minute=val / STAGE_MINUTES[nxt]))
    return sorted(out, key=lambda r: -r["per_minute"])


def format_dossier(d: Mapping[str, Any]) -> str:
    """One hypothesis as readable text: verdict first, then the evidence for and against."""
    L = [f"{d['hypothesis']} ({d['topic']}): {d['outcome']} [{d['state']}] stage {d['stage']}",
         f"  skill {d['skill']:+.4f} (shrunk {d['posterior_skill']:+.4f}); p raw {d['p_raw']:.3f}, family {d['p_family']:.3f}, Holm {d['p_holm']:.3f}, BH {d['p_bh']:.3f}"]
    L += ["  because: " + r for r in d.get("reasons", [])[:5]]
    for c, v in d.get("comparators", {}).items():
        L.append(f"  vs {c:16s} {'BEAT' if v['passed'] else 'not beaten'}  (against {v['against']}, {v['source']}, mean diff {v['mean_diff']:+.4f})")
    if d.get("random_effects"):
        re_ = d["random_effects"]
        L.append(f"  years: pooled {re_['pooled']:+.4f} [{re_['lo']:+.4f}, {re_['hi']:+.4f}], tau2 {re_['tau2']:.2e}, I2 {re_['i2']:.0%}, "
                 f"P(a new year is negative) {re_['p_negative_year']:.0%}")
    if d.get("robust"):
        L.append("  robustness: " + ("PASS" if d["robust"]["robust"] else "FAIL " + ",".join(d["robust"]["failed"])))
    if d.get("dropped"):
        L.append("  dropped inputs: " + "; ".join(f"{k} ({v})" for k, v in list(d["dropped"].items())[:4]))
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- honest limits of the sample
def survivorship_disclosure(pool: pd.DataFrame, labels: pd.DataFrame | None = None) -> dict[str, Any]:
    """The price panel is survivor-only (memory: all backtests are survivorship-biased until delisted history exists). A panel built
    from survivors shows no ticker disappearing mid-sample, so this counts tickers whose last pool row is well before the sample
    end. Zero disappearing names over many years is itself the warning: real universes lose several percent of names a year."""
    if pool.empty:
        return dict(tickers=0, ended_early=0, ended_early_share=float("nan"), years=0, survivor_only_suspected=True)
    last = pool.groupby("ticker")["date"].max()
    first = pool.groupby("ticker")["date"].min()
    end = pool["date"].max()
    span_years = max((end - pool["date"].min()).days / 365.25, 0.0)
    ended = int((last < end - pd.Timedelta(days=90)).sum())
    late = int((first > pool["date"].min() + pd.Timedelta(days=365)).sum())
    share = ended / max(len(last), 1)
    return dict(tickers=int(len(last)), ended_early=ended, ended_early_share=float(share), started_late=late, years=float(span_years),
                survivor_only_suspected=bool(span_years >= 3 and share < 0.01),
                note="direction skill on a survivor-only panel is biased upward for long calls; treat every lead as conditional on this")


def label_ambiguity_report(labels: pd.DataFrame) -> dict[str, float]:
    """How much of the label is defined? Share of decision rows that are movers, same-session double touches (direction undefined),
    windows with missing bars, and the up-rate among the defined movers. A high ambiguous share means `up_first` is being decided
    on the rows where the market did the most, which biases every direction number in one direction."""
    if labels.empty:
        return dict(rows=0, mover_rate=float("nan"), ambiguous_share=float("nan"), undefined_share=float("nan"), up_rate_movers=float("nan"))
    mover = labels["mover"].astype(bool)
    amb = labels["amb"].astype(bool) if "amb" in labels else pd.Series(False, index=labels.index)
    defined = labels["up_first"].notna() if "up_first" in labels else pd.Series(False, index=labels.index)
    return dict(rows=int(len(labels)), mover_rate=float(mover.mean()), ambiguous_share=float(amb.sum() / max(mover.sum(), 1)),
                undefined_share=float((mover & ~defined).sum() / max(mover.sum(), 1)),
                up_rate_movers=float(labels.loc[defined, "up_first"].mean()) if defined.any() else float("nan"))


def segment_adequacy(pool: pd.DataFrame, target_edge: float = 0.05, picks_per_week: int = 15) -> pd.DataFrame:
    """Which segments (event type, regime, sector) are large enough for the lab to say anything? For each segment: rows, weeks,
    the minimum detectable accuracy edge on that many effective bets and the weeks still needed for `target_edge`. Small segments
    are where 80% pockets get 'found'; this table shows in advance that no negative or positive claim can be made there."""
    rows = []
    for col in ("seg", "reg", "sector"):
        if col not in pool:
            continue
        for v, g in pool[pool["pick"] & pool["up"].notna()].groupby(col):
            n = len(g)
            base = float(g["up"].mean())
            deff = week_design_effect(g["up"].to_numpy(), pd.factorize(g["date"].to_numpy())[0]) if n > 20 else 1.0
            mde = LC.minimum_detectable_gap(int(max(n / deff, 2)), max(base, 1 - base)) if n >= 4 else float("inf")
            rows.append(dict(factor=col, level=str(v), rows=n, weeks=int(g["date"].nunique()), mde_edge=float(mde), design_effect=float(deff),
                             weeks_needed=weeks_needed(target_edge, max(base, 1 - base), picks_per_week, deff), adequate=bool(mde <= target_edge)))
    return pd.DataFrame(rows, columns=["factor", "level", "rows", "weeks", "mde_edge", "design_effect", "weeks_needed", "adequate"])


def extras_for_report(report: LabReport, inputs: LabInputs, pool: pd.DataFrame, Xd: pd.DataFrame, pred: pd.DataFrame, W: pd.DataFrame,
                      info: pd.DataFrame, cfg: LabConfig) -> None:
    """The remaining descriptive blocks of a run, attached to the report: baselines, survivorship and label disclosures, segment
    adequacy, the honest blend of the leading cells, volatility-bucket accuracy and payoff-rank of the best lead, and the
    truncation (future-read) test of the derived features."""
    report.baselines = baseline_table(Xd, pool, cfg.test_years)
    report.disclosures = dict(survivorship=survivorship_disclosure(pool), labels=label_ambiguity_report(inputs.labels),
                              false_lead_arithmetic=expected_false_leads(len(HYPOTHESES), cfg.screen_p, cfg.alpha))
    report.adequacy = segment_adequacy(pool)
    lead = [r for r in report.results.values() if r.best_tag and r.stage is not Stage.CHEAP_SCREEN]
    if len(lead) >= 2 and not W.empty:
        top = [r.best_tag for r in sorted(lead, key=lambda r: -r.skill)[:4] if r.best_tag in W.columns]
        report.blend = honest_blend(W, info, top) if len(top) >= 2 else {}
    if lead:
        best = max(lead, key=lambda r: r.skill)
        report.vol_buckets = accuracy_by_vol_bucket(pred, pool, best.best_tag)
        report.return_rank = return_rank_test(pred, pool, best.best_tag, min(cfg.n_boot, 300), cfg.seed)
    cut = pd.Timestamp(pool["date"].quantile(0.6, interpolation="nearest")) if len(pool) else None
    report.truncation = truncation_test(inputs, cfg, report.now, cut) if cut is not None else {}
    if report.truncation and not report.truncation["ok"]:
        report.warnings.append("truncation test: derived features at past dates change when later rows are removed: "
                               + ",".join(o["column"] for o in report.truncation["offenders"][:5]))


# ---------------------------------------------------------------------------------------------- multiplicity price and the final answer
def multiplicity_price(n_cells: int, alpha: float = 0.05) -> dict[str, float]:
    """What having looked at n_cells cells costs: the Bonferroni per-cell alpha, the critical z it implies, the expected largest null
    z among n_cells (sqrt(2 ln n)) and the chance that at least one null cell clears an unadjusted 5%. The 80% frontier is read
    against these numbers, not against a single-cell interval."""
    if n_cells < 1 or not 0 < alpha < 1:
        raise ValueError("multiplicity_price: n_cells >= 1 and alpha in (0, 1)")
    a = alpha / n_cells
    return dict(n_cells=n_cells, alpha_per_cell=a, z_crit=float(stats.norm.isf(a)), expected_null_max_z=float(math.sqrt(2 * math.log(max(n_cells, 2)))),
                p_any_unadjusted=float(1 - (1 - alpha) ** n_cells))


def cell_evidence(hits: int, n: int, design_effect: float = 1.0, gate: float = EIGHTY, n_cells: int = 1) -> dict[str, float]:
    """One frontier cell judged five ways: raw rate, Wilson lower bound, Wilson bound on the design-effect-deflated sample (weeks are
    not independent), the multiplicity-adjusted lower bound (Wilson at alpha/n_cells on the deflated sample), and the posterior
    probability that the true rate exceeds the gate. Only the adjusted bound may be called a reach of the gate."""
    if n <= 0 or hits < 0 or hits > n:
        raise ValueError("cell_evidence: need 0 <= hits <= n and n > 0")
    n_eff = max(n / max(design_effect, 1.0), 1.0)
    k_eff = hits * n_eff / n
    z_adj = float(stats.norm.isf(0.05 / max(n_cells, 1) / 2))
    return dict(rate=hits / n, wilson_lo=LC.wilson(hits, n, 1.96)[0], deflated_lo=LC.wilson(k_eff, n_eff, 1.96)[0],
                adjusted_lo=LC.wilson(k_eff, n_eff, z_adj)[0], p_above_gate=prob_accuracy_above(k_eff, n_eff, gate), n_eff=float(n_eff),
                reaches_gate=bool(LC.wilson(k_eff, n_eff, z_adj)[0] >= gate))


class Recommendation(_StrEnum):
    NOT_USABLE = "NOT_USABLE"                  # nothing to use; the direction stage should abstain
    RESEARCH_ONLY = "RESEARCH_ONLY"            # leads exist but none may influence a decision
    NEEDS_HOLDOUT = "NEEDS_HOLDOUT"            # a candidate passed every gate; a fresh holdout is the next step
    VOID = "VOID"                              # the run's own controls or gate failed; say nothing about direction


def recommendation(report: LabReport) -> tuple[Recommendation, str]:
    """The lab's single answer to 'may direction influence a decision?'. There is deliberately no 'USE' value: a candidate becomes
    usable only after the fresh-holdout stage (pre-registered replication) and the promotion gate, neither of which this module
    can grant."""
    if not report.gate.is_open:
        return Recommendation.VOID, "volatility stage lacks out-of-sample evidence: " + "; ".join(report.gate.reasons)
    if not report.protocol_ok:
        return Recommendation.VOID, "controls failed: " + "; ".join(report.controls.failures() if report.controls else ["no controls ran"])
    if report.candidates():
        return Recommendation.NEEDS_HOLDOUT, f"candidate(s) {report.candidates()} passed every gate; a fresh, pre-registered holdout is required"
    if report.found_anything():
        return Recommendation.RESEARCH_ONLY, "weak, unreplicated leads only"
    return Recommendation.NOT_USABLE, report.statement or "no reliable direction signal"


def blind_summary(report: LabReport) -> dict[str, Any]:
    """The only thing about a run that may leave the research side without a matured-record gate: counts and a recommendation, with
    no dates, years, tickers or hypothesis-level numbers. Passed through the blind-side scan before it is returned, so a future
    edit that adds an identifying field fails here instead of in a live replay."""
    rec, _ = recommendation(report)
    out = dict(recommendation=str(rec), hypotheses=len(report.results), candidates=len(report.candidates()),
               leads=sum(1 for r in report.results.values() if r.outcome is Outcome.WEAK_UNREPLICATED),
               gate_open=bool(report.gate.is_open), controls_pass=bool(report.protocol_ok))
    TV.assert_trader_safe(out, what="direction blind summary")
    return out


def state_to_dict(state: LabState) -> dict[str, Any]:
    """JSON-safe LabState (the ledger's e-values are stored as logs so the running product survives a round trip exactly)."""
    return dict(states={k: str(v) for k, v in state.states.items()}, null_streak=state.null_streak, runs=state.runs, history=state.history,
                seen_keys=sorted(state.seen_keys, key=str), info_prev=state.info_prev, last_alpha=state.last_alpha, alpha_spent=state.alpha_spent,
                ledger=None if state.ledger is None else dict(alpha=state.ledger.alpha, kappa=state.ledger.kappa, log_e=state.ledger.log_e,
                                                              looks=state.ledger.looks, seen=sorted([list(t) for t in state.ledger.seen])))


def state_from_dict(d: Mapping[str, Any]) -> LabState:
    """Inverse of state_to_dict; the result is validated, and an inconsistent record raises instead of steering the queue."""
    led = None
    if d.get("ledger"):
        L = d["ledger"]
        led = EvidenceLedger(L["alpha"], L["kappa"])
        led.log_e, led.looks, led.seen = dict(L["log_e"]), dict(L["looks"]), {tuple(t) for t in L["seen"]}
    st = LabState({k: ResearchState(v) for k, v in d["states"].items()}, int(d["null_streak"]), int(d["runs"]), list(d["history"]),
                  set(d["seen_keys"]), led, int(d["info_prev"]), float(d["last_alpha"]), float(d["alpha_spent"]))
    bad = st.validate()
    if bad:
        raise ValueError("stored LabState is inconsistent: " + "; ".join(bad))
    return st


OPEN_SPECS: tuple[HypothesisSpec, ...] = (
    _spec("overnight_drift", "overnight versus intraday returns", ["overnight20", "intraday20", "atr_pct", "m_vix_chg5", "rel_ind20"],
          "Overnight returns carry the informed order flow and intraday returns the noise; their difference may sign the next week.", 0),
    _spec("earnings_proximity", "earnings proximity", ["days_to_earn", "earn_in_week", "days_since_earn", "ear_volsurge"],
          "The days around a release change both the size and the sign asymmetry of the coming move.", 0, min_columns=2),
    _spec("insider_conviction", "insider conviction", ["ins_value30", "ins_officer30", "ins_opportunistic30"],
          "Large officer purchases, not the count of buyers, are the literature's informative subset.", 1, ["ins_officer30"], min_columns=2),
    _spec("breadth_dispersion", "breadth and dispersion", ["m_breadth", "m_dispersion", "m_vix_term", "m_spy_r5"],
          "When few names carry the index, single-name direction depends on the leaders rather than the market.", 0, min_columns=2),
    _spec("trend_quality", "trend quality", ["dist_ma50", "dist_ma200", "r20", "r60", "atr_pct"],
          "Trends with low volatility relative to their slope persist; the same slope with high volatility does not.", 1, ["dist_ma50"], min_columns=3),
)


def load_open_specs(registry: HypothesisRegistry, available: Sequence[str]) -> dict[str, str]:
    """Offers each open spec to the registry: added when its inputs exist and it is not redundant, otherwise the reason is returned.
    New hypotheses therefore enter through the same validation and multiplicity accounting as the original fifteen."""
    out = {}
    for s in OPEN_SPECS:
        if not resolve_columns(available, s):
            out[s.name] = "inputs not in the panel"
            continue
        try:
            out[s.name] = registry.add(s, "open hypothesis: " + s.mechanism[:60])
        except ValueError as e:
            out[s.name] = str(e)
    return out


def by_outcome(report: LabReport) -> dict[str, list[str]]:
    """Hypothesis names grouped by outcome, in catalogue order (what a reader scans first)."""
    out: dict[str, list[str]] = {}
    for n, r in report.results.items():
        out.setdefault(str(r.outcome), []).append(n)
    return out


def summary_line(report: LabReport) -> str:
    """One line for logs: recommendation, gate, controls, counts by outcome and the number of cells priced."""
    rec, _ = recommendation(report)
    counts = ", ".join(f"{k}={len(v)}" for k, v in by_outcome(report).items())
    return f"direction_lab {report.now}: {rec} | gate={'open' if report.gate.is_open else 'closed'} controls={'ok' if report.protocol_ok else 'FAIL'} | {counts} | cells={report.n_cells}"
