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
from engine.research.core import (Availability, ExperimentValue, GateVerdict, MaturedRecord, Namespace, Problem,
                                  ResearchQuestion, ResearchState, Stage)

EIGHTY = 0.80                          # the section 10/11 target; fv_pipeline.FVConfig.gate must agree (see shared_target)
EPS = 1e-9
TARGETS = ("up_sign", "up_first")      # up_sign: finished the holding week above entry; up_first: the +10% barrier came first
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
    df["date"] = df.index.get_level_values(0)
    aucs, hits, picks, movers, uni = [], [], [], [], []
    for _, g in df.groupby("date", sort=True):
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
    """score -> predicted movers -> labelled direction pool. Ranking reads the score only: the constructor refuses any score whose
    name or origin is an outcome column, and the pool keeps realised-mover flags in a separately named diagnostic column that the
    feature builders refuse to read."""

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
        pool["score"] = sc.reindex(pool.index).to_numpy()
        L = lab.reindex(pool.index)
        pool["up"] = L[cfg.target].to_numpy(dtype=float)
        pool["fwd"] = L["fwd"].to_numpy(dtype=float)
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
        out["path_efficiency"] = (cl - cl.median()) * ic
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
        out["vix_x_r5"] = (vix - vix.median()) * r5
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
    pred, blocks = DF.walk_forward(pool, F, "up", cfg.test_years, models=tuple(kinds), extra=hooks, seg_cols=("seg", "reg"),
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
                                     bool(t["better"]) and not src.startswith("fallback_base_rate:unsigned") or bool(t["better"]))
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
    partial = res.p_raw < cfg.alpha or (Comparator.BASE_RATE in res.comparators and res.comparators[Comparator.BASE_RATE].passed)
    if partial and res.skill > 0:
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
            wf2 = run_walk_forward(pool, fams, Xd, lc, survivors, lc.models, Stage.STRONGER_TESTS, inputs.existing_p)
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
            if n not in survivors:
                r.outcome, r.state = Outcome.NO_RELIABLE_SIGNAL, state_for(Outcome.NO_RELIABLE_SIGNAL, powered)
                r.reasons = [f"stage-1 screen: permutation p {r.p_raw:.3f} >= {lc.screen_p}; not escalated (compute saved)"]
                continue
            kind = best.split(":")[1]
            r.stage = Stage.STRONGER_TESTS
            r.comparators = compare_all(wf2.pred, n, best, kind, lc, lc.seed + 5)
            stress_test(r, wf2.pred, pool.assign(**{}), fams[n].matrix, best, kind, lc)
            r.outcome, r.reasons = decide_outcome(r, lc, len(cells_final))
            r.state = state_for(r.outcome, powered)
            if r.outcome is Outcome.CANDIDATE:
                r.stage = Stage.CROSS_YEAR
        report.controls = evaluate_controls(pred_final, wf2.blocks if wf2 is not None else wf1.blocks, leak, lc, truth["planted_share"])
        if not report.controls.ok:
            for r in report.results.values():
                if r.outcome in (Outcome.CANDIDATE, Outcome.WEAK_UNREPLICATED, Outcome.NO_RELIABLE_SIGNAL):
                    r.reasons = report.controls.failures() + r.reasons
                    r.outcome, r.state = Outcome.CONTROL_FAILURE, ResearchState.CANCELLED
        self._frontier(report, pred_final, cells_final, names, lc)
        report.statement = honest_statement(report.eighty, report.power, lc.gate)
        return report

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
    nothing' is a valid answer, not a stall)."""
    states: dict[str, ResearchState] = dataclasses.field(default_factory=dict)
    null_streak: int = 0
    runs: int = 0
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    seen_keys: set = dataclasses.field(default_factory=set)

    def digest(self) -> str:
        return stable_hash({"states": {k: str(v) for k, v in sorted(self.states.items())}, "streak": self.null_streak, "runs": self.runs}, 12)


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
         ignore_gate: bool = False) -> tuple[LabState, LabReport]:
    """The lab's ONE public entry for the research loop: run once at `now`, fold the outcome into the persistent state. A rerun on
    identical inputs and code is recorded but does not extend the null streak (repetition is not replication)."""
    state = state or LabState()
    lab = DirectionLab(cfg, gate)
    report = lab.run(inputs, now, ignore_gate=ignore_gate)
    key = stable_hash([data_fingerprint(inputs), current_code_hash(), lab.cfg.hash(), str(as_date(now))], 16)
    fresh = key not in state.seen_keys
    new = LabState(dict(state.states), state.null_streak, state.runs + 1, list(state.history), set(state.seen_keys))
    for name, r in report.results.items():
        new.states[name] = r.state
    powered = report.power.get("mde_edge", 1.0) <= 0.03
    counts = report.gate.is_open and report.protocol_ok and powered
    if fresh:
        new.null_streak = (state.null_streak + 1) if counts and not report.found_anything() else (0 if report.found_anything() else state.null_streak)
    new.seen_keys.add(key)
    new.history.append(dict(key=key, fresh=fresh, outcomes=report.outcomes(), gate_open=report.gate.is_open, protocol_ok=report.protocol_ok,
                            null_streak=new.null_streak, config=report.config_hash))
    return new, report
