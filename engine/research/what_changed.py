"""'What changed?' investigation tree, knowability classes and testable discoveries (C68 checklists J, K, V; Bible phases 9/10).
IMPLEMENTED - NOT VALIDATED.

A significant prediction error opens a structured ten-level investigation (market, sector, cross-section, stock, pattern,
interaction, timing, exit, external information, model failure). Levels 1-4 are a NESTED decomposition of the error (market shock,
then sector beyond the market, then peers beyond both, then what is left for the stock), so their shares cannot add up to more
than the error. Each level reports whether it fired, how much of the error it explains, and whether that could have been
detected BEFORE the outcome (the difference between an explanation and a signal). Every investigation ends in the fixed chain

    CAUSE -> EVIDENCE -> PRE-OUTCOME DETECTABILITY -> REPEATABILITY -> NEW KNOWLEDGE -> MODEL CHANGE -> TEST.

The outcome is placed in exactly one of five knowability classes (checklist K). Nothing is forced predictable: "currently
unexplained" and "genuinely unknowable" are answers, and an unknowable outcome is preserved as knowledge. "Predictable only under a
newly discovered condition" is granted only when a condition has already survived replication.

Checklist V: every "I figured out why" becomes a DiscoveryClaim. It is knowledge only after eleven steps pass; before that it
stays an explanation hypothesis. The steps reuse engine.research.replication (fresh periods, stocks, regimes, control, transfer,
permanent hash-chained record) and engine.research.quality_gate (its replication gate); the competing explanations go into the
existing hypothesis tree. Model changes are PROPOSALS with applied=False and are gated on the test; the exit level measures the
path and never sets an exit or a hold (exits come from the learned exit policy only; the +/-1pp target is evaluation).

Builds on: pattern_change (checklist H) -> engine.learning.lifecycle/reliability/health; engine.research.knowability (assessment),
hypothesis_tree, replication, quality_gate. Public entry: `step(state, now, cases, ...)`."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import DecisionEffect, FirewallBreach, Subsystem, _StrEnum, as_date, require_past, stable_hash
from engine.learning.experiment_memory import Hypothesis
from engine.learning.research_priority import identity_leak
from engine.research import hypothesis_tree as HT
from engine.research import pattern_change as PC
from engine.research import quality_gate as QG
from engine.research import replication as RP
from engine.research.core import Availability, Knowability

PARAMS: dict[str, Any] = {
    "min_abs_error": 0.01,      # an error under this many return points is never an investigation
    "sig_z": 1.0,               # ... nor is one under this many stock-vol units over the horizon
    "share_fire": 0.30,         # a level must explain at least this share of the error to be a cause
    "z_fire": 1.5, "z_stock": 2.0, "z_combo": 2.0,
    "min_peers": 5, "peer_agree": 0.70, "min_hist": 30, "min_combo": 10, "combo_recent": 8,
    "lead_step": 4, "lead_max": 32,          # look back this many rows in steps to find when a pattern change became detectable
    "external_residual": 0.5,                # external information is the cause only when levels 1-4 leave at least this much unexplained
    "model_residual": 0.4,                   # ... likewise for model failure
    "repeat_min": 3, "repeat_rate": 0.30,
}


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


class Level(_StrEnum):
    MARKET = "L1_MARKET"
    SECTOR = "L2_SECTOR"
    CROSS_SECTION = "L3_CROSS_SECTION"
    STOCK = "L4_STOCK"
    PATTERN = "L5_PATTERN"
    INTERACTION = "L6_INTERACTION"
    TIMING = "L7_TIMING"
    EXIT = "L8_EXIT"
    EXTERNAL = "L9_EXTERNAL_INFORMATION"
    MODEL_FAILURE = "L10_MODEL_FAILURE"

    @property
    def number(self) -> int:
        return int(self.value[1:].split("_")[0])


LEVELS = tuple(Level)


class FiveWay(_StrEnum):
    """Checklist K: exactly these five."""
    PREDICTABLE_BUT_MISSED = "PREDICTABLE_BUT_MISSED"
    PARTIALLY_PREDICTABLE = "PARTIALLY_PREDICTABLE"
    PREDICTABLE_UNDER_NEW_CONDITION = "PREDICTABLE_ONLY_UNDER_NEWLY_DISCOVERED_CONDITION"
    CURRENTLY_UNEXPLAINED = "CURRENTLY_UNEXPLAINED"
    GENUINELY_UNKNOWABLE = "GENUINELY_UNKNOWABLE_FROM_AVAILABLE_INFORMATION"


# ------------------------------------------------------------------------------------------------ the case
@dataclasses.dataclass
class ErrorCase:
    """One significant prediction error, with only what the investigation may read. Histories end at or before the decision date;
    paths are fill..maturity; `after_path` (optional, HINDSIGHT) runs past maturity and is used by the timing level alone.
    Prediction and realisation are fractional returns over the horizon. Pattern effects are in the same units."""
    case_id: str
    decision_date: str
    matured_at: str
    predicted: float
    realised: float
    stock_hist: pd.Series
    stock_path: pd.Series
    market_hist: pd.Series | None = None
    market_path: pd.Series | None = None
    sector_hist: pd.Series | None = None
    sector_path: pd.Series | None = None
    peers: pd.DataFrame | None = None                          # columns: realised, optional predicted
    peer_baseline: float = 0.0
    pattern_frames: Mapping[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    combos: Mapping[str, pd.Series] = dataclasses.field(default_factory=dict)       # "a|b" -> effect of that pair firing together
    after_path: pd.Series | None = None
    policy_exit_return: float | None = None                    # what the LEARNED exit policy realised, if it exited early
    knowability: Any = None                                    # engine.research.knowability.KnowabilityAssessment
    confidence: float | None = None

    @property
    def error(self) -> float:
        return float(self.realised - self.predicted)

    @property
    def horizon(self) -> int:
        return int(len(self.stock_path))

    def validate(self) -> list[str]:
        errs = []
        if not (math.isfinite(self.predicted) and math.isfinite(self.realised)):
            errs.append("predicted and realised must be finite")
        if as_date(self.matured_at) <= as_date(self.decision_date):
            errs.append("the outcome cannot mature on or before the decision date")
        if self.horizon < 1:
            errs.append("empty stock path")
        elif not np.isfinite(self.stock_path.to_numpy(float)).all():
            errs.append("non-finite returns in the stock path: NaN is not a zero return")
        if identity_leak(self.case_id):
            errs.append("case_id carries a date or year")
        for name in ("market_path", "sector_path"):
            p = getattr(self, name)
            if p is not None and len(p) != self.horizon:
                errs.append(f"{name} must cover the same sessions as stock_path")
        return errs


def _cum(r) -> float:
    a = np.asarray(r, float)
    return float(np.prod(1.0 + a) - 1.0) if len(a) else 0.0


def _upto(s: pd.Series | None, decision_date: str) -> pd.Series | None:
    """History as of the decision: the close of the decision date is known, nothing later is."""
    if s is None:
        return None
    return s.loc[s.index <= pd.Timestamp(as_date(decision_date))].astype(float)


def _beta(y: pd.Series, x: pd.Series, min_n: int) -> tuple[float, pd.Series]:
    """OLS beta of y on x over the shared dates (1.0 when there are too few) and the residual series."""
    j = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(j) < min_n or float(j.iloc[:, 1].var()) < 1e-14:
        return 1.0, (j.iloc[:, 0] - j.iloc[:, 1]) if len(j) else pd.Series(dtype=float)
    b = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    return b, j.iloc[:, 0] - b * j.iloc[:, 1]


# ------------------------------------------------------------------------------------------------ findings
@dataclasses.dataclass(frozen=True)
class LevelFinding:
    level: Level
    measured: bool                    # False = the inputs for this level were not supplied; that is not "no cause"
    fired: bool
    component: float                  # signed part of the error attributed to this level
    share: float                      # component / error, clipped to [0, 1]
    z: float
    availability: Availability       # could the cause have been known BEFORE the outcome?
    lead: int | None                  # rows before the decision at which it was already detectable (pattern level)
    evidence: tuple
    qualifier: str = ""               # identity-free label used in the cause key (direction, pattern class, ...)


def _unmeasured(level: Level, why: str) -> LevelFinding:
    return LevelFinding(level, False, False, 0.0, 0.0, 0.0, Availability.UNCERTAIN, None, (why,))


def _share(comp: float, err: float) -> float:
    return float(min(1.0, max(0.0, comp / err))) if abs(err) > 1e-12 else 0.0


def is_significant(case: ErrorCase, cfg=None) -> tuple[bool, float]:
    """(significant, z): the error must clear both an absolute bar and the stock's own horizon volatility."""
    P = _cfg(cfg)
    h = _upto(case.stock_hist, case.decision_date)
    sd = float(h.std()) * math.sqrt(max(case.horizon, 1)) if h is not None and len(h) >= 10 else float("nan")
    z = abs(case.error) / sd if sd and sd > 1e-12 else float("nan")
    return bool(abs(case.error) >= P["min_abs_error"] and (not math.isfinite(z) or z >= P["sig_z"])), z


def _decompose(case: ErrorCase, P: Mapping) -> dict:
    """Nested attribution of the error: market, then sector beyond market, then peers beyond both, then the stock's residual."""
    err, h = case.error, case.horizon
    sh = _upto(case.stock_hist, case.decision_date)
    mh, sc = _upto(case.market_hist, case.decision_date), _upto(case.sector_hist, case.decision_date)
    out = {"market": 0.0, "sector": 0.0, "cross": 0.0, "stock": err, "beta": 1.0, "z_market": 0.0, "z_sector": 0.0, "n_peers": 0,
           "market_measured": False, "sector_measured": False, "cross_measured": False}
    if mh is not None and case.market_path is not None and len(mh) >= P["min_hist"]:
        beta, _ = _beta(sh, mh, P["min_hist"])
        shock = _cum(case.market_path) - float(mh.mean()) * h
        sd = float(mh.std()) * math.sqrt(h)
        out.update(market=beta * shock, beta=beta, z_market=shock / sd if sd > 1e-12 else 0.0, market_measured=True)
    if sc is not None and case.sector_path is not None and mh is not None and case.market_path is not None and len(sc) >= P["min_hist"]:
        bs, res = _beta(sc, mh, P["min_hist"])
        shock = _cum(case.sector_path) - bs * _cum(case.market_path) - float(res.mean()) * h
        sd = float(res.std()) * math.sqrt(h) if len(res) > 2 else 0.0
        out.update(sector=shock, z_sector=shock / sd if sd > 1e-12 else 0.0, sector_measured=True)
    if case.peers is not None and len(case.peers) >= P["min_peers"]:
        pe = case.peers["realised"].astype(float) - (case.peers["predicted"].astype(float) if "predicted" in case.peers.columns
                                                     else case.peer_baseline)
        pe = pe.dropna()
        if len(pe) >= P["min_peers"]:
            rem = err - out["market"] - out["sector"]
            raw = float(pe.median()) - out["market"] - out["sector"]
            # peers may explain what the market and sector left, never more, and never in the opposite direction
            cross = math.copysign(min(abs(raw), abs(rem)), rem) if raw * rem > 0 else 0.0
            out.update(cross=cross, n_peers=len(pe), cross_measured=True, peer_agree=float((np.sign(pe) == np.sign(err)).mean()))
    out["stock"] = err - out["market"] - out["sector"] - out["cross"]
    return out


def _level_market(case, d, P) -> LevelFinding:
    if not d["market_measured"]:
        return _unmeasured(Level.MARKET, "no market history/path supplied")
    sh = _share(d["market"], case.error)
    fired = sh >= P["share_fire"] and abs(d["z_market"]) >= P["z_fire"]
    return LevelFinding(Level.MARKET, True, fired, d["market"], sh, d["z_market"], Availability.KNOWN_ONLY_AFTER_EVENT, None,
                        (f"market shock {d['market']:.3f} of the error (beta {d['beta']:.2f}, z {d['z_market']:.1f})",),
                        "up" if d["market"] > 0 else "down")


def _level_sector(case, d, P) -> LevelFinding:
    if not d["sector_measured"]:
        return _unmeasured(Level.SECTOR, "no sector history/path supplied")
    sh = _share(d["sector"], case.error)
    fired = sh >= P["share_fire"] and abs(d["z_sector"]) >= P["z_fire"]
    return LevelFinding(Level.SECTOR, True, fired, d["sector"], sh, d["z_sector"], Availability.KNOWN_ONLY_AFTER_EVENT, None,
                        (f"sector moved {d['sector']:.3f} beyond the market (z {d['z_sector']:.1f})",), "up" if d["sector"] > 0 else "down")


def _level_cross(case, d, P) -> LevelFinding:
    if not d["cross_measured"]:
        return _unmeasured(Level.CROSS_SECTION, f"fewer than {P['min_peers']} comparable stocks")
    sh = _share(d["cross"], case.error)
    fired = sh >= P["share_fire"] and d["peer_agree"] >= P["peer_agree"]
    return LevelFinding(Level.CROSS_SECTION, True, fired, d["cross"], sh, d["peer_agree"], Availability.KNOWN_ONLY_AFTER_EVENT, None,
                        (f"{d['n_peers']} comparable stocks erred the same way in {d['peer_agree']:.0%} of cases; median excess {d['cross']:.3f}",),
                        "up" if d["cross"] > 0 else "down")


def _level_stock(case, d, P) -> LevelFinding:
    sh = _upto(case.stock_hist, case.decision_date)
    if sh is None or len(sh) < P["min_hist"]:
        return _unmeasured(Level.STOCK, "too little stock history for its own volatility")
    mh = _upto(case.market_hist, case.decision_date)
    _, res = _beta(sh, mh, P["min_hist"]) if mh is not None else (1.0, sh)
    sd = float(res.std()) * math.sqrt(case.horizon) if len(res) > 2 else float(sh.std()) * math.sqrt(case.horizon)
    z = d["stock"] / sd if sd > 1e-12 else 0.0
    day_sd = float(sh.std())
    worst = float(np.max(np.abs(case.stock_path.to_numpy(float))))
    extreme = day_sd > 1e-12 and worst >= 5.0 * day_sd
    share_ = _share(d["stock"], case.error)
    fired = share_ >= P["share_fire"] and (abs(z) >= P["z_stock"] or extreme)
    return LevelFinding(Level.STOCK, True, fired, d["stock"], share_, z, Availability.KNOWN_ONLY_AFTER_EVENT, None,
                        (f"stock residual {d['stock']:.3f} (z {z:.1f}); largest single day {worst / day_sd:.1f} daily sd" if day_sd > 1e-12
                         else f"stock residual {d['stock']:.3f}",), "extreme_day" if extreme else ("up" if d["stock"] > 0 else "down"))


def detectable_lead(frame: pd.DataFrame, decision_date: str, P: Mapping) -> int | None:
    """How many rows before the decision the pattern was already classed as failing, consecutively (causal at each step).
    0 = only at the decision itself, None = not failing at the decision."""
    f = PC.causal_slice(frame, decision_date)
    if len(f) < P["min_hist"]:
        return None
    if PC.classify("lead", frame, decision_date).change not in PC.FAILING:
        return None
    lead = 0
    for k in range(P["lead_step"], P["lead_max"] + 1, P["lead_step"]):
        if len(f) - k < P["min_hist"]:
            break
        if PC.classify("lead", f.iloc[:len(f) - k], f.index[len(f) - k - 1] + pd.Timedelta(days=1)).change not in PC.FAILING:
            break
        lead = k
    return lead


def _level_pattern(case, d, P) -> LevelFinding:
    if not case.pattern_frames:
        return _unmeasured(Level.PATTERN, "no pattern histories supplied")
    best, ev, cls = None, [], []
    for pid in sorted(case.pattern_frames):
        v = PC.classify(pid, case.pattern_frames[pid], case.decision_date)
        cls.append(v.change.value)
        ev.append(f"{pid}: {v.change.value} (z {v.profile.z:.1f}, error {v.profile.error_direction})")
        shortfall = v.profile.recent_mean - v.profile.hist_mean if math.isfinite(v.profile.recent_mean) else 0.0
        if v.change in PC.FAILING and (best is None or shortfall < best[1]):
            best = (pid, shortfall, v)
    dirn = 1.0 if case.predicted >= 0 else -1.0             # pattern effects are direction-adjusted; the error is raw
    if best is None:
        return LevelFinding(Level.PATTERN, True, False, 0.0, 0.0, 0.0, Availability.KNOWN_BEFORE_EVENT, None, tuple(ev), "healthy")
    pid, shortfall, v = best
    shortfall *= dirn
    sh = _share(shortfall, case.error)
    lead = detectable_lead(case.pattern_frames[pid], case.decision_date, P)
    return LevelFinding(Level.PATTERN, True, sh >= P["share_fire"], shortfall, sh, v.profile.z, Availability.KNOWN_BEFORE_EVENT, lead,
                        tuple(ev) + (f"pattern change was detectable {lead} rows before the decision" if lead is not None else "",),
                        v.change.value.lower())


def _level_interaction(case, d, P, pattern_fired: bool) -> LevelFinding:
    if not case.combos:
        return _unmeasured(Level.INTERACTION, "no combination histories supplied")
    best, ev = None, []
    for key in sorted(case.combos):
        s = _upto(case.combos[key], case.decision_date)
        if s is None or len(s) < P["min_combo"] + P["combo_recent"]:
            continue
        hist, rec = s.iloc[:-P["combo_recent"]].to_numpy(float), s.iloc[-P["combo_recent"]:].to_numpy(float)
        parts = [case.pattern_frames[k]["effect"].astype(float) for k in key.split("|") if k in case.pattern_frames]
        parts = [p.loc[p.index <= pd.Timestamp(as_date(case.decision_date))] for p in parts]
        base = float(np.mean([p.mean() for p in parts])) if parts else float(hist.mean())
        sd = float(hist.std(ddof=1)) if len(hist) > 2 else 0.0
        z = (float(rec.mean()) - base) / (sd / math.sqrt(len(rec))) if sd > 1e-12 else 0.0
        ev.append(f"{key}: recent {float(rec.mean()):.4f} vs parts {base:.4f} (z {z:.1f})")
        if best is None or z < best[1]:
            best = (key, z, float(rec.mean()) - base)
    if best is None:
        return _unmeasured(Level.INTERACTION, "no combination has enough history")
    key, z, gap = best
    gap *= 1.0 if case.predicted >= 0 else -1.0
    sh = _share(gap, case.error)
    fired = sh >= P["share_fire"] and z <= -P["z_combo"] and not pattern_fired
    return LevelFinding(Level.INTERACTION, True, fired, gap, sh, z, Availability.KNOWN_BEFORE_EVENT, None, tuple(ev), "combination")


def _dir_cum(path: np.ndarray, sign: float) -> np.ndarray:
    return sign * (np.cumprod(1.0 + path) - 1.0)


def _level_timing(case, d, P) -> LevelFinding:
    if case.after_path is None:
        return _unmeasured(Level.TIMING, "no post-maturity path supplied (timing needs hindsight and is flagged as such)")
    sign = 1.0 if case.predicted >= 0 else -1.0
    target = abs(case.predicted)
    inside = _dir_cum(case.stock_path.to_numpy(float), sign)
    if len(inside) and float(inside.max()) >= target:
        return LevelFinding(Level.TIMING, True, False, 0.0, 0.0, 0.0, Availability.KNOWN_ONLY_AFTER_EVENT, None,
                            ("the predicted move was reached inside the window; timing is not the issue",), "on_time")
    full = _dir_cum(np.concatenate([case.stock_path.to_numpy(float), case.after_path.to_numpy(float)]), sign)
    hit = np.flatnonzero(full >= target)
    if not len(hit):
        return LevelFinding(Level.TIMING, True, False, 0.0, 0.0, 0.0, Availability.KNOWN_ONLY_AFTER_EVENT, None,
                            ("the predicted move never arrived in the observed window or after it: not a timing error",), "never")
    late = int(hit[0]) + 1 - case.horizon
    return LevelFinding(Level.TIMING, True, True, case.error, 1.0, float(late), Availability.KNOWN_ONLY_AFTER_EVENT, None,
                        (f"the predicted move arrived {late} sessions after maturity (direction right, timing late)",), "late")


def _level_exit(case, d, P) -> LevelFinding:
    sign = 1.0 if case.predicted >= 0 else -1.0
    path = _dir_cum(case.stock_path.to_numpy(float), sign)
    if not len(path):
        return _unmeasured(Level.EXIT, "empty path")
    target, end, peak = abs(case.predicted), float(path[-1]), float(path.max())
    reached = peak >= target
    shortfall = max(0.0, min(peak, target) - end) if reached else 0.0
    comp = -sign * shortfall                                   # error = sign * (end - target) in return terms
    sh = _share(comp, case.error) if reached and end < target else 0.0
    evidence = (f"path peaked at {peak:.3f} of a predicted {target:.3f} and ended at {end:.3f}; evaluation only, the target sets no exit",)
    if case.policy_exit_return is not None:
        evidence += (f"the learned exit policy realised {case.policy_exit_return:.3f}",)
    return LevelFinding(Level.EXIT, True, bool(reached and end < target and sh >= P["share_fire"]), comp, sh, peak - end,
                        Availability.UNCERTAIN, None, evidence, "peak_given_back" if reached else "not_reached")


def _level_external(case, prior_share: float, P) -> LevelFinding:
    k = case.knowability
    if k is None:
        return _unmeasured(Level.EXTERNAL, "no knowability assessment supplied")
    cls = k.classification
    fired = cls in (Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE) and (1.0 - prior_share) >= P["external_residual"]
    return LevelFinding(Level.EXTERNAL, True, fired, case.error * max(0.0, 1.0 - prior_share) if fired else 0.0,
                        max(0.0, 1.0 - prior_share) if fired else 0.0, float(k.anticipation), Availability.UNAVAILABLE if fired else Availability.UNCERTAIN,
                        None, (f"knowability {cls.value}; coverage {k.coverage:.2f}; {len(k.information_that_was_unavailable)} unavailable items",),
                        cls.value.lower())


def _level_model(case, prior_share: float, P) -> LevelFinding:
    k = case.knowability
    if k is None:
        return _unmeasured(Level.MODEL_FAILURE, "no knowability assessment supplied")
    predictable = k.classification in (Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE)
    fired = predictable and (1.0 - prior_share) >= P["model_residual"]
    return LevelFinding(Level.MODEL_FAILURE, True, fired, case.error * max(0.0, 1.0 - prior_share) if fired else 0.0,
                        max(0.0, 1.0 - prior_share) if fired else 0.0, float(k.anticipation), Availability.KNOWN_BEFORE_EVENT if predictable else Availability.UNCERTAIN,
                        None, (f"information existed before the decision ({k.n_present_channels} channels, anticipation {k.anticipation:.2f}) "
                               f"and the model did not act on it",) if predictable else ("no pre-outcome information found for the model to miss",),
                        k.classification.value.lower())


# ------------------------------------------------------------------------------------------------ five-way knowability
@dataclasses.dataclass(frozen=True)
class KnowabilityVerdict:
    klass: FiveWay
    reasons: tuple
    excluded_data_failure: bool         # a data defect is not a market event; it is excluded from market conclusions
    basis: str                          # which evidence decided: assessment / findings / condition


def classify_knowability(findings: Sequence[LevelFinding], assessment: Any = None, confirmed_conditions: Mapping[str, Any] | None = None,
                         condition_key: str = "") -> KnowabilityVerdict:
    """Map an outcome to the five checklist-K classes. Order matters: unknowable evidence is never overridden by a story, and the
    'newly discovered condition' class needs a condition that has ALREADY been replicated (never a fresh claim)."""
    by = {f.level: f for f in findings}
    cond_ok = bool(condition_key) and confirmed_conditions is not None and \
        getattr(confirmed_conditions.get(condition_key), "status", None) == RP.Status.REPLICATED
    if assessment is not None:
        c = assessment.classification
        if c == Knowability.DATA_FAILURE:
            return KnowabilityVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("the move traces to a data defect, not market behaviour",), True, "assessment")
        if c in (Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE):
            return KnowabilityVerdict(FiveWay.GENUINELY_UNKNOWABLE, (f"assessment {c.value}: nothing available beforehand carried it",), False, "assessment")
        if c == Knowability.PREDICTABLE:
            return KnowabilityVerdict(FiveWay.PREDICTABLE_BUT_MISSED, ("the information was available and sufficient before the decision",), False, "assessment")
        if c in (Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE):
            if cond_ok:
                return KnowabilityVerdict(FiveWay.PREDICTABLE_UNDER_NEW_CONDITION,
                                          (f"a replicated condition ({condition_key}) makes it predictable where it was only weakly so",), False, "condition")
            return KnowabilityVerdict(FiveWay.PARTIALLY_PREDICTABLE, (f"assessment {c.value}: part of the move was foreseeable",), False, "assessment")
        if cond_ok:
            return KnowabilityVerdict(FiveWay.PREDICTABLE_UNDER_NEW_CONDITION,
                                      (f"a replicated condition ({condition_key}) explains what the assessment could not",), False, "condition")
        return KnowabilityVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("the assessment found no explanation; unknown stays unknown",), False, "assessment")
    pat = by.get(Level.PATTERN)
    if pat is not None and pat.fired and pat.lead is not None:
        return KnowabilityVerdict(FiveWay.PREDICTABLE_BUT_MISSED, (f"the pattern was already failing {pat.lead} rows before the decision",), False, "findings")
    if cond_ok:
        return KnowabilityVerdict(FiveWay.PREDICTABLE_UNDER_NEW_CONDITION, (f"replicated condition {condition_key}",), False, "condition")
    if any(f.fired and f.availability == Availability.KNOWN_BEFORE_EVENT for f in findings):
        return KnowabilityVerdict(FiveWay.PARTIALLY_PREDICTABLE, ("a cause with pre-outcome evidence fired, but no knowability assessment confirms it",), False, "findings")
    return KnowabilityVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("no level explained the error with pre-outcome evidence; no forced explanation",), False, "findings")


# ------------------------------------------------------------------------------------------------ the chain
@dataclasses.dataclass(frozen=True)
class Detectability:
    availability: Availability
    lead_rows: int | None
    basis: str


@dataclasses.dataclass(frozen=True)
class Repeatability:
    verdict: str                      # REPEATED / ONE_OFF_SO_FAR / UNTESTED
    n_same: int
    n_similar: int
    rate: float | None


@dataclasses.dataclass(frozen=True)
class ModelChange:
    subsystem: Subsystem | None
    effect: DecisionEffect
    proposal: str
    applied: bool = False              # a proposal is never applied by an investigation
    gated_on: str = "TEST"


@dataclasses.dataclass(frozen=True)
class ValidationPlan:
    kind: str                          # HYPOTHESIS_TREE / NONE
    tree_id: str
    steps: tuple                       # the eleven checklist-V steps this cause must pass before it is knowledge
    question: str


V_STEPS = ("formulate the hypothesis", "identify supporting evidence", "identify contradictory evidence", "create a historical test",
           "out-of-sample evaluation", "multiple periods", "multiple stocks", "multiple regimes", "controls and placebos",
           "does the discovery transfer", "record the result permanently")


@dataclasses.dataclass(frozen=True)
class Conclusion:
    cause: str
    level: Level | None
    cause_key: str
    evidence: tuple
    detectability: Detectability
    repeatability: Repeatability
    new_knowledge: str
    model_change: ModelChange
    test: ValidationPlan

    def chain(self) -> tuple:
        return ("CAUSE", "EVIDENCE", "PRE-OUTCOME DETECTABILITY", "REPEATABILITY", "NEW KNOWLEDGE", "MODEL CHANGE", "TEST")


@dataclasses.dataclass(frozen=True)
class Investigation:
    case_id: str
    now: str
    significant: bool
    error: float
    error_z: float
    findings: tuple
    knowability: KnowabilityVerdict | None
    conclusion: Conclusion | None
    explained_share: float             # levels 1-8 together, capped at 1
    code_note: str = "IMPLEMENTED - NOT VALIDATED"

    def finding(self, level: Level) -> LevelFinding:
        return next(f for f in self.findings if f.level == level)

    def fired(self) -> tuple:
        return tuple(f.level for f in self.findings if f.fired)

    def validate(self) -> list[str]:
        errs = []
        if not self.significant:
            return errs if self.conclusion is None else ["an insignificant error must not carry a conclusion"]
        if len(self.findings) != len(LEVELS) or [f.level for f in self.findings] != list(LEVELS):
            errs.append("a significant error needs all ten levels in order")
        c = self.conclusion
        if c is None:
            return errs + ["significant error without a conclusion chain"]
        for name in ("cause", "cause_key", "new_knowledge"):
            if not getattr(c, name):
                errs.append(f"chain link {name} is empty")
            elif identity_leak(getattr(c, name)):
                errs.append(f"chain link {name} carries a date or year")
        if not c.evidence:
            errs.append("chain link EVIDENCE is empty")
        if self.knowability is not None and self.knowability.klass == FiveWay.GENUINELY_UNKNOWABLE:
            if c.model_change.effect != DecisionEffect.NONE or c.test.kind != "NONE":
                errs.append("an unknowable outcome must not propose a model change or a test: it is preserved as knowledge")
        if c.model_change.applied:
            errs.append("an investigation may only PROPOSE a model change")
        if c.test.kind == "HYPOTHESIS_TREE" and len(c.test.steps) != len(V_STEPS):
            errs.append("a testable cause carries all eleven checklist-V steps")
        return errs


class CaseLibrary:
    """Prior conclusions, for repeatability. A case is only visible to a later investigation once it has matured before that one."""

    def __init__(self):
        self.rows: list[tuple] = []           # (case_id, matured_at, level, cause_key, five_way)

    def add(self, inv: Investigation) -> None:
        if inv.conclusion is None or any(r[0] == inv.case_id for r in self.rows):
            return
        c = inv.conclusion
        self.rows.append((inv.case_id, inv.now, c.level.value if c.level else "NONE", c.cause_key,
                          inv.knowability.klass.value if inv.knowability else ""))

    def stats(self, level: str, cause_key: str, before) -> tuple[int, int]:
        prior = [r for r in self.rows if as_date(r[1]) < as_date(before)]
        similar = [r for r in prior if r[2] == level]
        return sum(1 for r in similar if r[3] == cause_key), len(similar)


def repeatability(lib: CaseLibrary | None, level: str, cause_key: str, before, P: Mapping) -> Repeatability:
    if lib is None or not lib.rows:
        return Repeatability("UNTESTED", 0, 0, None)
    same, sim = lib.stats(level, cause_key, before)
    rate = same / sim if sim else None
    if sim == 0:
        return Repeatability("UNTESTED", same, sim, None)
    verdict = "REPEATED" if same >= P["repeat_min"] and rate is not None and rate >= P["repeat_rate"] else "ONE_OFF_SO_FAR"
    return Repeatability(verdict, same, sim, rate)


CAUSE_TEXT = {
    Level.MARKET: "a market-wide move ({q}) drove the error",
    Level.SECTOR: "the sector moved differently from the market ({q})",
    Level.CROSS_SECTION: "comparable stocks all erred the same way ({q})",
    Level.STOCK: "the stock behaved unusually on its own ({q})",
    Level.PATTERN: "the pattern behind the prediction had changed ({q})",
    Level.INTERACTION: "a combination of patterns behaved differently from its parts ({q})",
    Level.TIMING: "the prediction was right but arrived {q}",
    Level.EXIT: "the move was reached and then given back ({q})",
    Level.EXTERNAL: "information that could not have been known drove the outcome ({q})",
    Level.MODEL_FAILURE: "the information was available and the model failed to use it ({q})",
}
CHANGE_OF = {
    Level.MARKET: (Subsystem.RISK, DecisionEffect.POSITION_SIZE, "condition exposure on the market state that produced this shock"),
    Level.SECTOR: (Subsystem.RISK, DecisionEffect.POSITION_SIZE, "condition sector exposure on sector-relative state"),
    Level.CROSS_SECTION: (Subsystem.SELECTION, DecisionEffect.RANKING, "rank against comparable stocks, not in isolation"),
    Level.STOCK: (Subsystem.RISK, DecisionEffect.ABSTENTION, "abstain on stocks with the unusual-behaviour signature"),
    Level.PATTERN: (Subsystem.SELECTION, DecisionEffect.PATTERN_WEIGHTING, "down-weight the pattern in the condition where it changed"),
    Level.INTERACTION: (Subsystem.SELECTION, DecisionEffect.RANKING, "score the combination on its own history, not as a sum"),
    Level.TIMING: (Subsystem.TIMING, DecisionEffect.TIMING, "study entry timing for this pattern"),
    Level.EXIT: (Subsystem.EXIT, DecisionEffect.EXIT, "feed the path to the learned exit policy as training data; the target sets nothing"),
    Level.EXTERNAL: (None, DecisionEffect.NONE, "none: the outcome was unknowable"),
    Level.MODEL_FAILURE: (Subsystem.SELECTION, DecisionEffect.CONFIDENCE, "add the missed information channel and recheck calibration"),
}


def choose_primary(findings: Sequence[LevelFinding]) -> LevelFinding | None:
    """The fired level with the largest explained share; ties go to the broader (lower) level."""
    fired = [f for f in findings if f.fired]
    return min(fired, key=lambda f: (-round(f.share, 6), f.level.number)) if fired else None


def _fmt(x: float) -> str:
    return f"{x:.2f}" if abs(x) < 1000 else f"{x:.0f}"


def build_conclusion(case: ErrorCase, findings: Sequence[LevelFinding], kv: KnowabilityVerdict, lib: CaseLibrary | None, now, P) -> Conclusion:
    prim = choose_primary(findings)
    if prim is None:
        cause = "no level explained the error; the cause is currently unknown"
        key, level = "L0:unexplained", None
        evidence = tuple(e for f in findings for e in f.evidence[:1]) or ("no evidence",)
        det = Detectability(Availability.UNCERTAIN, None, "nothing explained the error, so nothing can be called detectable")
    else:
        level = prim.level
        cause = CAUSE_TEXT[level].format(q=prim.qualifier.replace("_", " "))
        key = f"L{level.number}:{prim.qualifier}"
        evidence = prim.evidence + (f"explains {prim.share:.0%} of an error of {case.error:.3f}",)
        det = Detectability(prim.availability, prim.lead, "pattern change measured only from outcomes before the decision" if level == Level.PATTERN
                            else "the cause is measured from outcomes after the decision; a pre-outcome signal has not been shown")
    rep = repeatability(lib, level.value if level else "NONE", key, now, P)
    kl = kv.klass
    if kl == FiveWay.GENUINELY_UNKNOWABLE:
        knowledge = "UNKNOWABLE_PRESERVED: this outcome could not have been known from available information"
        mc, test = ModelChange(None, DecisionEffect.NONE, "none: an unknowable outcome changes no model"), ValidationPlan("NONE", "", (), "")
    elif kl == FiveWay.CURRENTLY_UNEXPLAINED and prim is None:
        knowledge = "UNEXPLAINED_KEPT_OPEN: no explanation has support; the unknown is kept as unknown"
        mc = ModelChange(None, DecisionEffect.NONE, "none until an explanation survives tests")
        test = ValidationPlan("HYPOTHESIS_TREE", "WC-" + stable_hash([case.case_id, key], 8), V_STEPS,
                              "what explains an error that no level accounted for?")
    else:
        sub, eff, prop = CHANGE_OF[level] if level else (None, DecisionEffect.NONE, "none")
        knowledge = f"CANDIDATE_EXPLANATION: {cause}; not knowledge until it survives the eleven tests"
        mc = ModelChange(sub, eff, prop)
        test = ValidationPlan("HYPOTHESIS_TREE", "WC-" + stable_hash([case.case_id, key], 8), V_STEPS, f"does it hold that {cause}?")
    return Conclusion(cause, level, key, evidence, det, rep, knowledge, mc, test)


# ------------------------------------------------------------------------------------------------ the investigation
def investigate(case: ErrorCase, now, cfg=None, library: CaseLibrary | None = None,
                confirmed_conditions: Mapping[str, Any] | None = None) -> Investigation:
    """Run the ten levels on one error as of `now`. Refuses a case that has not matured before `now`. Histories are cut at the
    decision date, so the level findings cannot change when data after the decision changes (pattern and market detection are
    forward in time); only the timing level reads post-maturity data and marks itself KNOWN_ONLY_AFTER_EVENT."""
    P = _cfg(cfg)
    errs = case.validate()
    if errs:
        raise ValueError(f"cannot investigate {case.case_id}: {errs}")
    require_past(case.matured_at, now, f"error case {case.case_id}")
    sig, z = is_significant(case, P)
    if not sig:
        return Investigation(case.case_id, str(as_date(now)), False, case.error, z if math.isfinite(z) else 0.0, tuple(), None, None, 0.0)
    d = _decompose(case, P)
    f = {Level.MARKET: _level_market(case, d, P), Level.SECTOR: _level_sector(case, d, P), Level.CROSS_SECTION: _level_cross(case, d, P),
         Level.STOCK: _level_stock(case, d, P)}
    f[Level.PATTERN] = _level_pattern(case, d, P)
    f[Level.INTERACTION] = _level_interaction(case, d, P, f[Level.PATTERN].fired)
    f[Level.TIMING] = _level_timing(case, d, P)
    f[Level.EXIT] = _level_exit(case, d, P)
    # timing and exit are alternatives to the four broad levels: they explain the SAME shortfall, so they only count as causes
    # when the nested levels left something unexplained
    nested = sum(f[l].share for l in (Level.MARKET, Level.SECTOR, Level.CROSS_SECTION, Level.STOCK) if f[l].fired)
    for lv in (Level.TIMING, Level.EXIT):
        if f[lv].fired and nested >= 1.0 - P["share_fire"] and not (f[Level.STOCK].fired and f[Level.STOCK].qualifier == "extreme_day"):
            f[lv] = dataclasses.replace(f[lv], fired=False, evidence=f[lv].evidence + ("the broad levels already explain the error",))
    broad = min(1.0, sum(f[l].share for l in LEVELS[:8] if f[l].fired))
    f[Level.EXTERNAL] = _level_external(case, broad, P)
    f[Level.MODEL_FAILURE] = _level_model(case, broad, P)
    findings = tuple(f[l] for l in LEVELS)
    kv = classify_knowability(findings, case.knowability, confirmed_conditions,
                              condition_key=(lambda p: f"L{p.level.number}:{p.qualifier}" if p else "")(choose_primary(findings)))
    concl = build_conclusion(case, findings, kv, library, now, P)
    return Investigation(case.case_id, str(as_date(now)), True, case.error, z if math.isfinite(z) else 0.0, findings, kv, concl, broad)


def plan_test(inv: Investigation, created_real: str) -> HT.HypothesisTree | None:
    """The hypothesis tree for an investigation's cause: the primary cause, every other fired level as a rival, an isolated
    one-off (noise) and the unknown. Unknowable outcomes get no tree. Priors come from the explained shares, never from hope."""
    c = inv.conclusion
    if c is None or c.test.kind != "HYPOTHESIS_TREE":
        return None
    hyps = []
    for f in inv.findings:
        if f.fired and c.level is not None:
            hyps.append(Hypothesis(f"h_{f.level.value.split('_')[0].lower()}", CAUSE_TEXT[f.level].format(q=f.qualifier.replace("_", " ")),
                                   min(max(f.share, 0.05), 0.9), kind="explanation"))
    hyps.append(Hypothesis("h_oneoff", "an isolated failure with no repeatable cause", 0.20, kind="noise"))
    hyps.append(Hypothesis(HT.UNKNOWN_HID, "none of the named explanations", 0.10, kind="explanation"))
    text = c.test.question if not identity_leak(c.test.question) else "what explains this error?"
    return HT.build_tree(c.test.tree_id, text, inv.now, hyps, problem="RESEARCH_PROCESS")


# ------------------------------------------------------------------------------------------------ checklist V
@dataclasses.dataclass(frozen=True)
class DiscoveryClaim:
    """'I figured out why I was wrong.' `discovery` is the original historical test in replication's own vocabulary."""
    claim_id: str
    statement: str
    condition_key: str
    supporting: tuple
    contradicting: tuple
    contradiction_search_done: bool
    discovery: RP.Discovery | None
    origin_cases: tuple = ()


class ClaimStatus(_StrEnum):
    EXPLANATION_HYPOTHESIS = "EXPLANATION_HYPOTHESIS"
    KNOWLEDGE = "KNOWLEDGE"
    REFUTED = "REFUTED"


@dataclasses.dataclass(frozen=True)
class StepResult:
    number: int
    name: str
    passed: bool
    detail: str


@dataclasses.dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    status: ClaimStatus
    steps: tuple
    assessment: Any
    gate_ok: bool

    @property
    def is_knowledge(self) -> bool:
        return self.status == ClaimStatus.KNOWLEDGE

    def failed_steps(self) -> tuple:
        return tuple(s.number for s in self.steps if not s.passed)


def claim_from_investigation(inv: Investigation, discovery: RP.Discovery | None, supporting: Sequence[str] = (),
                             contradicting: Sequence[str] = (), searched: bool = False) -> DiscoveryClaim | None:
    """Turn a candidate explanation into a claim. Unknowable and unexplained outcomes produce none: nothing to test."""
    c = inv.conclusion
    if c is None or c.test.kind != "HYPOTHESIS_TREE" or c.level is None:
        return None
    return DiscoveryClaim("CL-" + stable_hash([inv.case_id, c.cause_key], 8), c.cause, c.cause_key, tuple(supporting) or c.evidence,
                          tuple(contradicting), searched, discovery, (inv.case_id,))


def evaluate_claim(claim: DiscoveryClaim, runs: Sequence[RP.ReplicationRun], now, ledger: RP.ReplicationLedger | None = None,
                   policy: RP.ReplicationPolicy = RP.DEFAULT_POLICY) -> ClaimVerdict:
    """Run the eleven checklist-V steps as of `now`. KNOWLEDGE only when every step passes; a claim that fails a powered
    replication is REFUTED (kept, with the reason); anything else stays an EXPLANATION_HYPOTHESIS."""
    d = claim.discovery
    steps: list[StepResult] = []

    def add(n: int, ok: bool, detail: str) -> None:
        steps.append(StepResult(n, V_STEPS[n - 1], bool(ok), detail))

    leak = identity_leak(claim.statement)
    add(1, bool(claim.statement.strip()) and bool(claim.condition_key) and not leak,
        leak or ("a stated, identity-free claim with a condition" if claim.statement.strip() else "empty statement"))
    add(2, len(claim.supporting) > 0, f"{len(claim.supporting)} supporting item(s)")
    add(3, claim.contradiction_search_done, "contradictory evidence was searched for" if claim.contradiction_search_done
        else "no search for contradictory evidence was made: silence is not absence")
    if d is None:
        for n in range(4, 12):
            add(n, False, "no historical test (Discovery) exists yet")
        return ClaimVerdict(claim.claim_id, ClaimStatus.EXPLANATION_HYPOTHESIS, tuple(steps), None, False)
    derr = d.validate()
    add(4, not derr, "a historical test with its search size recorded" if not derr else "; ".join(derr))
    mine = [r for r in runs if r.discovery_id == d.discovery_id]
    for r in mine:
        require_past(r.matured_at, now, f"replication run {r.run_id}")
    a = RP.assess(d, mine, now, policy) if not derr else None
    fresh_after = [r for r in mine if as_date(r.window[0]) > as_date(d.window[1])]
    add(5, bool(fresh_after) and a is not None and a.n_supporting >= 1, f"{len(fresh_after)} run(s) on data after the original window")
    axes = set(a.axes_covered) if a else set()
    add(6, RP.Axis.PERIOD.value in axes and a is not None and a.n_supporting >= policy.min_independent,
        f"{a.n_supporting if a else 0} independent supporting period(s), {policy.min_independent} required")
    add(7, RP.Axis.STOCKS.value in axes, "fresh stocks covered" if RP.Axis.STOCKS.value in axes else "no independent run on fresh stocks")
    add(8, RP.Axis.REGIME.value in axes, "fresh regimes covered" if RP.Axis.REGIME.value in axes else "no independent run in a fresh regime")
    with_ctrl = [r for r in mine if r.control_effects is not None]
    add(9, bool(with_ctrl) and a is not None and a.requirements.get("control_beaten", False),
        f"{len(with_ctrl)} run(s) carry a matched control; control beaten: {a.requirements.get('control_beaten') if a else False}")
    add(10, a is not None and a.status == RP.Status.REPLICATED, f"replication status {a.status.value if a else 'NONE'}")
    gate_ok = False
    if a is not None:
        gate_ok = QG.gate_replication(a, QG.QualityPolicy(code_hash=d.code_hash)).ok
    recorded = False
    if ledger is not None and a is not None:
        try:
            if d.discovery_id not in ledger.discoveries():
                ledger.add_discovery(d)
            known = {r.run_id for r in ledger.runs_for(d.discovery_id, now)}
            for r in mine:
                if r.run_id not in known:
                    ledger.add_run(r, now)
            ledger.record_assessment(a)
            recorded = not ledger.verify()
        except (ValueError, KeyError, FileExistsError) as e:
            add(11, False, f"could not record permanently: {e}")
    if not any(s.number == 11 for s in steps):
        add(11, recorded, "recorded on the hash-chained replication ledger" if recorded else "no ledger given: the result was not recorded")
    if a is not None and a.status == RP.Status.FAILED:
        status = ClaimStatus.REFUTED
    elif all(s.passed for s in steps) and gate_ok:
        status = ClaimStatus.KNOWLEDGE
    else:
        status = ClaimStatus.EXPLANATION_HYPOTHESIS
    return ClaimVerdict(claim.claim_id, status, tuple(steps), a, gate_ok)


# ------------------------------------------------------------------------------------------------ daily entry
@dataclasses.dataclass
class WhatChangedState:
    library: CaseLibrary = dataclasses.field(default_factory=CaseLibrary)
    investigations: dict = dataclasses.field(default_factory=dict)
    trees: dict = dataclasses.field(default_factory=dict)
    claims: dict = dataclasses.field(default_factory=dict)
    verdicts: dict = dataclasses.field(default_factory=dict)
    ledger: RP.ReplicationLedger = dataclasses.field(default_factory=RP.ReplicationLedger)
    unknowable: dict = dataclasses.field(default_factory=dict)         # case_id -> preserved conclusion (never reclassified silently)

    def confirmed_conditions(self) -> dict:
        return {self.claims[k].condition_key: v.assessment for k, v in self.verdicts.items() if v.is_knowledge and k in self.claims}


@dataclasses.dataclass(frozen=True)
class StepReport:
    now: str
    investigated: tuple
    skipped_insignificant: tuple
    by_class: Mapping[str, int]
    new_trees: tuple
    preserved_unknowable: tuple
    knowledge: tuple


def step(state: WhatChangedState, now, cases: Sequence[ErrorCase], cfg=None,
         discoveries: Mapping[str, RP.Discovery] | None = None, runs: Sequence[RP.ReplicationRun] = ()) -> StepReport:
    """The research loop's daily entry: investigate every matured, significant, not-yet-investigated error, open a hypothesis tree
    for each testable cause, preserve unknowable outcomes, and re-evaluate open claims against the runs that have matured."""
    done, skipped, trees, unk = [], [], [], []
    counts: dict[str, int] = {}
    for c in sorted(cases, key=lambda c: c.case_id):
        if c.case_id in state.investigations or as_date(c.matured_at) >= as_date(now):
            continue
        inv = investigate(c, now, cfg, state.library, state.confirmed_conditions())
        bad = inv.validate()
        if bad:
            raise ValueError(f"malformed investigation {c.case_id}: {bad}")
        state.investigations[c.case_id] = inv
        if not inv.significant:
            skipped.append(c.case_id)
            continue
        state.library.add(inv)
        done.append(c.case_id)
        counts[inv.knowability.klass.value] = counts.get(inv.knowability.klass.value, 0) + 1
        if inv.knowability.klass == FiveWay.GENUINELY_UNKNOWABLE:
            state.unknowable[c.case_id] = inv.conclusion
            unk.append(c.case_id)
        tree = plan_test(inv, str(as_date(now)))
        if tree is not None:
            state.trees[tree.tree_id] = tree
            trees.append(tree.tree_id)
            cl = claim_from_investigation(inv, (discoveries or {}).get(c.case_id))
            if cl is not None:
                state.claims[cl.claim_id] = cl
    knowledge = []
    for cid, cl in sorted(state.claims.items()):
        v = evaluate_claim(cl, runs, now, state.ledger)
        state.verdicts[cid] = v
        if v.is_knowledge:
            knowledge.append(cid)
    return StepReport(str(as_date(now)), tuple(done), tuple(skipped), counts, tuple(trees), tuple(unk), tuple(knowledge))


def reclassify_unknowable(state: WhatChangedState, case_id: str, new_evidence_claim: str) -> bool:
    """An unknowable conclusion can only be reopened by a claim that has become KNOWLEDGE; otherwise it stays preserved."""
    v = state.verdicts.get(new_evidence_claim)
    if case_id not in state.unknowable or v is None or not v.is_knowledge:
        return False
    del state.unknowable[case_id]
    return True


def investigation_table(state: WhatChangedState) -> pd.DataFrame:
    rows = []
    for cid, inv in sorted(state.investigations.items()):
        c = inv.conclusion
        rows.append({"case": cid, "significant": inv.significant, "error": inv.error,
                     "class": inv.knowability.klass.value if inv.knowability else "", "cause_key": c.cause_key if c else "",
                     "fired": ",".join(l.value for l in inv.fired()), "explained": inv.explained_share,
                     "repeat": c.repeatability.verdict if c else ""})
    return pd.DataFrame(rows)


def render(inv: Investigation) -> str:
    """Plain-language report of one investigation, chain last."""
    if not inv.significant:
        return f"{inv.case_id}: error {inv.error:.3f} is within normal variation; no investigation."
    lines = [f"{inv.case_id}: error {inv.error:.3f} (z {inv.error_z:.1f}); {inv.knowability.klass.value}"]
    for f in inv.findings:
        state = "FIRED" if f.fired else ("clear" if f.measured else "not measured")
        lines.append(f"  {f.level.value:<26} {state:<12} share {f.share:.0%}  {f.evidence[0] if f.evidence else ''}")
    c = inv.conclusion
    lines += [f"  CAUSE        {c.cause}", f"  EVIDENCE     {'; '.join(c.evidence[:2])}",
              f"  DETECTABLE   {c.detectability.availability.value} ({c.detectability.basis})",
              f"  REPEATABLE   {c.repeatability.verdict} ({c.repeatability.n_same}/{c.repeatability.n_similar})",
              f"  NEW KNOWLEDGE {c.new_knowledge}", f"  MODEL CHANGE {c.model_change.proposal} (proposal, gated on TEST)",
              f"  TEST         {c.test.kind} {c.test.tree_id}"]
    return "\n".join(lines)
