"""Prediction-error engine (C68 checklist C; extends engine.learning.surprise, which supplies the z / bits / cell machinery).

A frozen expectation (engine.research.expectations) meets its matured outcome (engine.research.outcomes) and the engine computes
TEN SEPARATE errors, each with its own scale and z - return, directional, volatility, timing, exit, confidence, market-regime,
sector-regime, pattern-strength, interaction - and never one collapsed score (`ErrorReport` has no total; `severity` only
RANKS components). The +10% / +15,+11,+7 case from the checklist is reproduced by `diagnose`: direction right, exit return
below the prediction, the opportunity larger than predicted, the best exit earlier than the actual one, and the exit error
computed from the path alone so the exit model is judged independently of the return predictor.

Rules kept here: an outcome is used only if it matured strictly before `now`; the ±1pp figure is MEASURED (`honest_tolerance`,
Wilson interval, unscored expectations counted against it) and never fed back to any exit or selection decision - no function in
this module returns anything that an exit could read, and the exit error does not read `predicted_return`; a regime label that
is not in the labeler's vocabulary is left UNSCORED rather than scored as wrong. Feeding the SurpriseTracker uses identity-free
cells so repeated similar errors escalate research priority through the existing tracker. LABEL = IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.calibration import wilson                       # noqa: F401 - re-exported; the one Wilson interval
from engine.learning.core import as_date, canonical_json, require_past, stable_hash
from engine.learning.surprise import SurpriseTracker, binary_z
from engine.research import calibration_target as CT
from engine.research.expectations import Expectation, LedgerTampered, SealedLane
from engine.research.outcomes import OutcomeLedger, OutcomeReconstruction

LABEL = "IMPLEMENTED - NOT VALIDATED"
LANE_ERRORS = "err68"
COMPONENTS = ("return", "direction", "volatility", "timing", "exit", "confidence", "market_regime", "sector_regime",
              "pattern_strength", "interaction")
DIRECTIONS = ("UP", "DOWN", "FLAT")
VOL_STATES = ("CALM", "VOLATILE")


@dataclasses.dataclass(frozen=True)
class ErrorConfig:
    tol: float = 0.01                    # the +-1 percentage-point evaluation target (measurement only)
    target_rate: float = 0.80
    min_scored: int = 50                 # below this the +-1pp rate is reported but NOT judged
    scale_floor: float = 0.005           # smallest return scale, so a tiny predicted volatility cannot manufacture huge z
    vol_rel_scale: float = 0.35          # a realised/predicted volatility ratio within +-35% is within normal
    timing_rel_scale: float = 0.25       # timing error scale as a share of the predicted holding period (>= 1 session)
    confident: float = 0.70              # confidence at/above which a wrong call is 'confident'
    wrong_z: float = 2.0                 # |return z| at/above which the miss is 'substantial'
    move_vol_mult: float = 0.5           # regime UP/DOWN needs a move of this many hold-period vols (floor 0.5%)
    vol_calm_max: float = 0.015          # daily context volatility at/above this is VOLATILE
    regime_scale: float = 0.5            # categorical mismatch (0..1) is standardised by this
    pattern_scale_floor: float = 0.10

    def validate(self) -> list[str]:
        e = []
        if not 0 < self.tol < 0.5:
            e.append("tol outside (0, 0.5)")
        if not 0.5 < self.target_rate < 1:
            e.append("target_rate outside (0.5, 1)")
        if self.scale_floor <= 0 or self.regime_scale <= 0 or self.pattern_scale_floor <= 0:
            e.append("scales must be > 0")
        return e


@dataclasses.dataclass(frozen=True)
class ErrorComponent:
    """One error, on its own scale. error = actual - expected (or a 0..1 miss for categorical ones); None = not scorable, and
    `applicable=False` says why in `note` - an unscorable component is never read as zero error."""
    name: str
    expected: float | None
    actual: float | None
    error: float | None
    scale: float | None
    z: float | None
    applicable: bool
    note: str = ""
    detail: Mapping = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "detail", MappingProxyType(dict(sorted(self.detail.items()))))


def _comp(name, expected, actual, scale, detail=None, note="") -> ErrorComponent:
    err = float(actual - expected)
    return ErrorComponent(name, float(expected), float(actual), err, float(scale), err / float(scale), True, note, detail or {})


def _na(name: str, note: str, detail: Mapping | None = None) -> ErrorComponent:
    return ErrorComponent(name, None, None, None, None, None, False, note, detail or {})


@dataclasses.dataclass(frozen=True)
class ErrorReport:
    prediction_id: str
    decided_at: str
    matured_at: str
    components: Mapping                  # name -> ErrorComponent, exactly COMPONENTS
    diagnosis: tuple                     # tags describing what the separate errors jointly say
    confident_wrong: bool
    expectation_hash: str
    outcome_hash: str

    def __post_init__(self):
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "diagnosis", tuple(self.diagnosis))

    def __getitem__(self, name: str) -> ErrorComponent:
        return self.components[name]

    def vector(self) -> dict:
        """Signed z per component (None where not scorable). A vector, deliberately not a scalar."""
        return {n: c.z for n, c in self.components.items()}

    def severity(self, k: int | None = None) -> list[tuple[str, float]]:
        """Components ranked by |z|, largest first. A ranking to decide where to look first; no number here sums the errors."""
        rows = sorted(((n, abs(c.z)) for n, c in self.components.items() if c.applicable and c.z is not None),
                      key=lambda r: (-r[1], r[0]))
        return rows[:k] if k else rows

    def content(self) -> dict:
        return json.loads(canonical_json(self))

    @property
    def report_hash(self) -> str:
        return stable_hash(self.content(), 32)

    @classmethod
    def from_record(cls, d: Mapping) -> "ErrorReport":
        d = dict(d)
        d["components"] = {k: ErrorComponent(**v) for k, v in d["components"].items()}
        return cls(**d)


# ------------------------------------------------------------------------------------------------ regime vocabulary
def realized_regime(move: float | None, vol: float | None, hold: int, cfg: ErrorConfig) -> str | None:
    """Label a realised context window as '<UP|DOWN|FLAT>_<CALM|VOLATILE>' from what happened over the hold only."""
    if move is None or vol is None or not math.isfinite(vol):
        return None
    thr = max(0.005, cfg.move_vol_mult * vol * math.sqrt(max(hold, 1)))
    d = "UP" if move >= thr else "DOWN" if move <= -thr else "FLAT"
    return f"{d}_{'VOLATILE' if vol >= cfg.vol_calm_max else 'CALM'}"


def parse_regime(label: str) -> dict[str, str]:
    """The dimensions an expected regime string names (subset of direction / volatility state). Unknown words are ignored."""
    toks = {t for t in str(label).upper().replace("-", "_").replace(" ", "_").split("_") if t}
    out = {}
    for t in toks:
        if t in DIRECTIONS:
            out["direction"] = t
        elif t in VOL_STATES:
            out["vol"] = t
    return out


def regime_error(name: str, expected_label: str, move, vol, hold: int, cfg: ErrorConfig) -> ErrorComponent:
    actual = realized_regime(move, vol, hold, cfg)
    exp = parse_regime(expected_label)
    if actual is None:
        return _na(name, "no realised context series: regime error unknowable", {"expected_label": expected_label})
    if not exp:
        return _na(name, f"expected label {expected_label!r} names no direction/volatility state: left unscored",
                   {"actual_label": actual})
    act = parse_regime(actual)
    miss = [k for k in exp if exp[k] != act[k]]
    frac = len(miss) / len(exp)
    return _comp(name, 0.0, frac, cfg.regime_scale, {"expected_label": expected_label, "actual_label": actual,
                                                     "mismatched": miss, "move": float(move), "vol": float(vol)})


# ------------------------------------------------------------------------------------------------ component builders
def _return_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    scale = max(exp.predicted_volatility * math.sqrt(out.holding_days), cfg.scale_floor)
    err = out.exit_return - exp.predicted_return
    detail = {"exit_return_error": err, "peak_vs_prediction": out.max_return - exp.predicted_return,
              "mfe_error": out.mfe - exp.mfe, "mae_error": out.mae - exp.mae, "pit": exp.distribution.cdf(out.exit_return),
              "pinball_loss": exp.distribution.pinball(out.exit_return), "within_tol": CT.within(out.exit_return, exp.predicted_return, cfg.tol),
              "in_band_predicted": exp.prob_in_band(), "in_band_actual": bool(0.05 <= out.exit_return <= 0.10)}
    return _comp("return", exp.predicted_return, out.exit_return, scale, detail)


def _direction_error(exp: Expectation, out: OutcomeReconstruction) -> ErrorComponent:
    correct = int(out.exit_return > 0)
    z = binary_z(exp.confidence, correct)
    return ErrorComponent("direction", float(exp.confidence), float(correct), float(1 - correct), 1.0, float(z), True,
                          "flat outcome counts as a miss" if out.actual_direction == 0 else "",
                          {"brier": (exp.confidence - correct) ** 2, "ever_favourable": out.max_return > 0,
                           "actual_direction": out.actual_direction, "predicted_direction": exp.direction})


def _volatility_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    if not math.isfinite(out.realized_vol) or out.realized_vol <= 0:
        return _na("volatility", "realised volatility not estimable (flat or too short)", {"method": out.vol_method})
    ratio = out.realized_vol / exp.predicted_volatility
    return _comp("volatility", exp.predicted_volatility, out.realized_vol, exp.predicted_volatility * cfg.vol_rel_scale,
                 {"log_ratio": math.log(ratio), "ratio": ratio, "method": out.vol_method})


def _timing_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    scale = max(1.0, cfg.timing_rel_scale * exp.holding_period)
    lo, hi = exp.exit_window
    return _comp("timing", exp.time_to_peak, float(out.t_max), scale,
                 {"peak_in_predicted_window": bool(lo <= out.t_max <= hi), "peak_at_exit": out.t_max == out.holding_days,
                  "peak_before_exit": out.t_max < out.holding_days})


def _exit_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    """Judges the exit alone: what the taken exit earned against the best exit that existed on the realised path. It reads the
    predicted exit window and holding period but NOT predicted_return, direction or confidence, so a good return forecast cannot
    hide a bad exit (or the reverse)."""
    best = out.best_exit["return"]
    scale = max((out.realized_vol if math.isfinite(out.realized_vol) else 0.0) * math.sqrt(out.holding_days), cfg.scale_floor)
    lo, hi = exp.exit_window
    miss = 0.0 if lo <= out.holding_days <= hi else (out.holding_days - lo if out.holding_days < lo else out.holding_days - hi)
    post = out.post_exit_best
    detail = {"regret": out.regret, "best_t": out.best_exit["t"], "hold": out.holding_days, "window_miss_sessions": miss,
              "captured_fraction": (out.exit_return / out.max_return) if out.max_return > 0 else None,
              "post_exit_gain": None if post is None else post["return"] - out.exit_return,
              "intraday_upper_bound": out.best_exit["intraday_upper_bound"]}
    return _comp("exit", best, out.exit_return, scale, detail, "expected = best exit on the realised path (hindsight benchmark)")


def _confidence_error(exp: Expectation, out: OutcomeReconstruction) -> ErrorComponent:
    correct = float(out.exit_return > 0)
    c = min(max(exp.confidence, 0.001), 0.999)
    return _comp("confidence", exp.confidence, correct, math.sqrt(c * (1 - c)),
                 {"overconfident": bool(correct < exp.confidence), "brier": (exp.confidence - correct) ** 2})


def _mean_gap(expected: Mapping, realized: Mapping) -> tuple[list, dict, dict]:
    keys = sorted(k for k in expected if k in realized)
    rows = {k: {"expected": float(expected[k]), "realized": float(realized[k]), "error": float(realized[k] - expected[k])} for k in keys}
    missing = sorted(k for k in expected if k not in realized)
    unexpected = {k: float(v) for k, v in realized.items() if k not in expected}
    return keys, rows, {"unobserved": missing, "unexpected": unexpected}


def _pattern_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    keys, rows, extra = _mean_gap(exp.pattern_strengths, out.pattern_realized)
    changed = sorted(c["pattern"] for c in out.pattern_changes)
    if not keys:
        return _na("pattern_strength", "no expected pattern has a realised strength: unknowable, not zero",
                   {**extra, "changed_in_hold": changed})
    e = np.array([rows[k]["expected"] for k in keys])
    r = np.array([rows[k]["realized"] for k in keys])
    scale = max(float(np.mean(np.abs(e))), cfg.pattern_scale_floor)
    worst = max(keys, key=lambda k: abs(rows[k]["error"]))
    return _comp("pattern_strength", float(e.mean()), float(r.mean()), scale,
                 {"per_pattern": rows, "rms": float(np.sqrt(np.mean((r - e) ** 2))), "worst_pattern": worst,
                  "changed_in_hold": changed, **extra})


def _interaction_error(exp: Expectation, out: OutcomeReconstruction, cfg: ErrorConfig) -> ErrorComponent:
    keys, rows, extra = _mean_gap(exp.interactions, out.interaction_realized)
    if keys:
        e = np.array([rows[k]["expected"] for k in keys])
        r = np.array([rows[k]["realized"] for k in keys])
        scale = max(float(np.mean(np.abs(e))), cfg.pattern_scale_floor)
        return _comp("interaction", float(e.mean()), float(r.mean()), scale, {"per_interaction": rows, **extra})
    if extra["unexpected"] and not exp.interactions:
        mag = float(np.mean(np.abs(list(extra["unexpected"].values()))))
        return _comp("interaction", 0.0, mag, cfg.pattern_scale_floor, {**extra}, "interactions appeared that were not expected")
    if not exp.interactions and not out.interaction_realized:
        return _comp("interaction", 0.0, 0.0, cfg.pattern_scale_floor, {}, "none expected, none realised")
    return _na("interaction", "expected interactions have no realised strength: unknowable, not zero", extra)


# ------------------------------------------------------------------------------------------------ joint reading
def diagnose(exp: Expectation, out: OutcomeReconstruction, comps: Mapping, cfg: ErrorConfig) -> list[str]:
    """What the separate errors say TOGETHER, as tags. The checklist example (+10% predicted; path +15,+11,+7): DIRECTION_CORRECT,
    EXIT_RETURN_BELOW_PREDICTION, OPPORTUNITY_LARGER_THAN_PREDICTED, EXIT_LATE, plus MARKET_TAILWIND when the market helped."""
    tags = []
    tol = cfg.tol
    tags.append("DIRECTION_CORRECT" if out.exit_return > 0 else "DIRECTION_WRONG")
    d = comps["return"].detail
    if CT.within(out.exit_return, exp.predicted_return, tol):
        tags.append("EXIT_RETURN_WITHIN_TOLERANCE")
    elif out.exit_return < exp.predicted_return:
        tags.append("EXIT_RETURN_BELOW_PREDICTION")
    else:
        tags.append("EXIT_RETURN_ABOVE_PREDICTION")
    if d["peak_vs_prediction"] > tol:
        tags.append("OPPORTUNITY_LARGER_THAN_PREDICTED")
    elif out.max_return < exp.predicted_return - tol:
        tags.append("OPPORTUNITY_NEVER_REACHED_PREDICTION")
    if out.regret > tol and out.best_exit["t"] < out.holding_days:
        tags.append("EXIT_LATE")
    post = out.post_exit_best
    if post is not None and post["return"] - out.exit_return > tol:
        tags.append("EXIT_EARLY")
    if out.market_move is not None and abs(out.market_move) >= tol:
        tags.append("MARKET_TAILWIND" if out.market_move * out.side > 0 else "MARKET_HEADWIND")
    if out.pattern_changes:
        tags.append("PATTERN_CHANGED_IN_HOLD")
    if out.data_gaps:
        tags.append("DATA_GAPS")
    if out.external_events:
        tags.append("EXTERNAL_EVENT_IN_HOLD")
    return tags


def compute_errors(exp: Expectation, exp_hash: str, out: OutcomeReconstruction, now, cfg: ErrorConfig | None = None) -> ErrorReport:
    """PUBLIC pure function: the ten separate errors of one matured prediction. Refuses an outcome that has not matured before
    `now`, or that belongs to a different prediction / expectation content."""
    cfg = cfg or ErrorConfig()
    require_past(out.matured_at, now, f"outcome {out.prediction_id} maturity")
    if out.prediction_id != exp.prediction_id or out.expectation_hash != exp_hash:
        raise ValueError("outcome does not belong to this frozen expectation")
    comps = {"return": _return_error(exp, out, cfg), "direction": _direction_error(exp, out),
             "volatility": _volatility_error(exp, out, cfg), "timing": _timing_error(exp, out, cfg),
             "exit": _exit_error(exp, out, cfg), "confidence": _confidence_error(exp, out),
             "market_regime": regime_error("market_regime", exp.market_regime, out.market_move, out.market_vol, out.holding_days, cfg),
             "sector_regime": regime_error("sector_regime", exp.sector_regime, out.sector_move, out.sector_vol, out.holding_days, cfg),
             "pattern_strength": _pattern_error(exp, out, cfg), "interaction": _interaction_error(exp, out, cfg)}
    wrong = out.exit_return <= 0
    cw = bool(exp.confidence >= cfg.confident and wrong and comps["return"].z <= -cfg.wrong_z)
    tags = diagnose(exp, out, comps, cfg) + (["CONFIDENT_WRONG"] if cw else [])
    return ErrorReport(exp.prediction_id, exp.decided_at, out.matured_at, comps, tuple(tags), cw, exp_hash, out.outcome_hash)


# ------------------------------------------------------------------------------------------------ surprise integration
def situation_cell(exp: Expectation, component: str) -> str:
    """Identity-free context signature for the SurpriseTracker: repeated errors in SIMILAR situations escalate together."""
    conf = "high" if exp.confidence >= 0.7 else "mid" if exp.confidence >= 0.6 else "low"
    return "|".join((f"err={component}", f"side={'long' if exp.direction > 0 else 'short'}", f"mkt={exp.market_regime}",
                     f"sec={exp.sector_regime}", f"conf={conf}", f"hold={'short' if exp.holding_period <= 3 else 'long'}"))


def feed_surprise(tracker: SurpriseTracker, exp: Expectation, rep: ErrorReport, now) -> int:
    """Register each scorable component with the EXISTING surprise tracker (never a second one). Continuous components go in
    as expected/actual on their own scale; direction goes in as a binary against stated confidence; `confidence` is skipped
    (it is the same evidence as direction and would count it twice). Returns records added."""
    n = 0
    kids = (rep.prediction_id,)
    for name, c in rep.components.items():
        if not c.applicable or name == "confidence":
            continue
        cell = situation_cell(exp, name)
        try:
            if name == "direction":
                tracker.observe_binary(cell, c.expected, int(c.actual), rep.decided_at, rep.matured_at, now, kids)
            else:
                tracker.observe(cell, c.expected, c.actual, rep.decided_at, rep.matured_at, now, scale=c.scale, knowledge_ids=kids)
            n += 1
        except ValueError as ex:
            if "duplicate" not in str(ex):
                raise
    return n


# ------------------------------------------------------------------------------------------------ record for error research
ERROR_RECORD_SCHEMA = MappingProxyType({
    "obs_id": "prediction id of the frozen expectation (str)",
    "decided_at": "ISO date the expectation was formed",
    "matured_at": "ISO date the position exited (outcome first knowable), strictly after decided_at",
    "expected": "predicted return (fraction), the frozen expectation's predicted_return",
    "realised": "realised position return at exit (fraction)",
    "confidence": "stated P(direction correct) in [0.5, 1]",
    "scale": "error sd the prediction claimed: predicted daily volatility * sqrt(holding sessions), floored",
    "pattern": "identity-free key of the expected pattern with the largest |strength| ('' if none)",
    "sector": "the expectation's sector regime label (identity-free)",
    "regime": "the expectation's market regime label (identity-free)",
    "stock_type": "predicted-volatility bucket: vol_low (<1.5%/day), vol_mid (<3%), vol_high",
    "exit_regret": "gain left on the table by the exit as a fraction of |predicted return| (None if no predicted return)",
})
# consumed by engine.research.error_research.obs_from_record (by these names). market_weight and transfer are NOT produced here:
# nothing in an expectation or outcome measures them, so they are left to the consumer's stated defaults rather than invented.


@dataclasses.dataclass(frozen=True)
class ErrorRecord:
    """The flat, identity-free view of one matured prediction error for the research world. Field names are ERROR_RECORD_SCHEMA."""
    obs_id: str
    decided_at: str
    matured_at: str
    expected: float
    realised: float
    confidence: float
    scale: float
    pattern: str
    sector: str
    regime: str
    stock_type: str
    exit_regret: float | None

    def validate(self) -> list[str]:
        e = []
        if set(dataclasses.asdict(self)) != set(ERROR_RECORD_SCHEMA):
            e.append("record fields differ from ERROR_RECORD_SCHEMA")
        if as_date(self.matured_at) <= as_date(self.decided_at):
            e.append("matured_at must be after decided_at")
        if not (0.0 <= self.confidence <= 1.0 and self.scale > 0):
            e.append("confidence outside [0,1] or scale <= 0")
        return e


def vol_bucket(daily_vol: float) -> str:
    return "vol_low" if daily_vol < 0.015 else "vol_mid" if daily_vol < 0.03 else "vol_high"


def error_record(exp: Expectation, out: OutcomeReconstruction, rep: ErrorReport) -> ErrorRecord:
    """Flatten (expectation, outcome, report) into the record error research reads. Every value comes from the frozen
    expectation, the matured outcome or the report - nothing is estimated here."""
    r = rep["return"]
    top = max(exp.pattern_strengths, key=lambda k: (abs(exp.pattern_strengths[k]), k)) if exp.pattern_strengths else ""
    regret = out.regret / abs(exp.predicted_return) if abs(exp.predicted_return) > 1e-12 else None
    rec = ErrorRecord(rep.prediction_id, rep.decided_at, rep.matured_at, float(r.expected), float(r.actual), float(exp.confidence),
                      float(r.scale), top, exp.sector_regime, exp.market_regime, vol_bucket(exp.predicted_volatility), regret)
    errs = rec.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return rec


# ------------------------------------------------------------------------------------------------ aggregation
def error_profile(reports: Sequence[ErrorReport], component: str) -> dict:
    """Bias, spread and consistency of one component over many predictions - the input for 'systematic error' questions."""
    cs = [r.components[component] for r in reports if r.components[component].applicable]
    if not cs:
        return {"component": component, "n": 0, "unscored": len(reports)}
    e = np.array([c.error for c in cs])
    z = np.array([c.z for c in cs])
    sd = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
    tstat = float(np.mean(e) / (sd / math.sqrt(len(e)))) if len(e) > 1 and sd > 0 else float("nan")
    return {"component": component, "n": len(cs), "unscored": len(reports) - len(cs), "bias": float(e.mean()),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "sd": sd, "t_bias": tstat,
            "mean_z": float(z.mean()), "share_positive": float((e > 0).mean()), "share_big": float((np.abs(z) >= 2).mean())}


def all_profiles(reports: Sequence[ErrorReport]) -> dict:
    return {c: error_profile(reports, c) for c in COMPONENTS}


def confidence_reliability(reports: Sequence[ErrorReport], bins: int = 5) -> dict:
    """Stated confidence against realised hit rate, equal-count bins; ECE and mean over-confidence."""
    rows = sorted(((r.components["confidence"].expected, r.components["confidence"].actual) for r in reports
                   if r.components["confidence"].applicable))
    if not rows:
        return {"n": 0, "bins": [], "ece": None, "overconfidence": None}
    chunks = np.array_split(np.arange(len(rows)), min(bins, len(rows)))
    out, ece = [], 0.0
    for idx in chunks:
        conf, hit = np.array([rows[i][0] for i in idx]), np.array([rows[i][1] for i in idx])
        out.append({"n": len(idx), "mean_confidence": float(conf.mean()), "hit_rate": float(hit.mean())})
        ece += len(idx) / len(rows) * abs(conf.mean() - hit.mean())
    allc, allh = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    return {"n": len(rows), "bins": out, "ece": float(ece), "overconfidence": float(allc.mean() - allh.mean())}


def honest_tolerance(reports: Sequence[ErrorReport], n_unscored: int = 0, cfg: ErrorConfig | None = None) -> dict:
    """ADAPTER over the canonical +-1pp statistic (engine.research.calibration_target.within / share_verdict): the hit flags were
    set by CT.within when each report was computed, and the share, Wilson interval, worst case (`n_unscored` matured-but-
    unreconstructable predictions count as misses) and verdict all come from CT.share_verdict. Only the verdict names are this
    module's own. The anti-gaming headline (commitments, suppressed losers, cherry-picking) is calibration_target.evaluate."""
    cfg = cfg or ErrorConfig()
    sv = CT.share_verdict([bool(r.components["return"].detail["within_tol"]) for r in reports], n_unscored, cfg.target_rate,
                          cfg.min_scored, 0.95)
    names = {CT.Status.EMPTY: "INSUFFICIENT_SAMPLE", CT.Status.INSUFFICIENT_SAMPLE: "INSUFFICIENT_SAMPLE", CT.Status.ACHIEVED: "TARGET_MET"}
    verdict = names.get(sv["status"]) or ("TARGET_NOT_MET" if sv["clear_miss"] else "NOT_DEMONSTRATED")
    return {"hits": sv["hits"], "n": sv["n"], "rate": sv["share"], "wilson_low": sv["lo"], "wilson_high": sv["hi"], "n_unscored": n_unscored,
            "worst_case_rate": sv["worst_case"], "tol": cfg.tol, "target": cfg.target_rate, "verdict": verdict,
            "canonical": "engine.research.calibration_target.share_verdict"}


def by_group(reports: Sequence[ErrorReport], exps: Mapping[str, Expectation], key, component: str = "return") -> dict:
    """Error profile of one component split by any property of the expectation (regime, sector regime, confidence bucket...)."""
    groups: dict[str, list[ErrorReport]] = {}
    for r in reports:
        groups.setdefault(str(key(exps[r.prediction_id])), []).append(r)
    return {g: error_profile(rs, component) for g, rs in sorted(groups.items())}


# ------------------------------------------------------------------------------------------------ the engine
class ErrorEngine:
    """PUBLIC ENTRY `step(outcomes, now)`: evaluate every matured, not yet evaluated prediction, store the reports on a hash
    chain, and feed the existing surprise tracker. Deterministic: same ledgers and `now` give the same reports."""

    def __init__(self, cfg: ErrorConfig | None = None, tracker: SurpriseTracker | None = None, root=None):
        self.cfg = cfg or ErrorConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid ErrorConfig: " + "; ".join(errs))
        self.tracker = tracker
        self.lane = SealedLane(root, LANE_ERRORS)
        self._reports: dict[str, ErrorReport] = {}
        for ln in self.lane.lines():
            rep = ErrorReport.from_record(ln["body"]["report"])
            self._reports[rep.prediction_id] = rep

    def __len__(self) -> int:
        return len(self._reports)

    def report(self, pid: str) -> ErrorReport:
        return self._reports[pid]

    def reports(self, now=None) -> list[ErrorReport]:
        rs = sorted(self._reports.values(), key=lambda r: (r.matured_at, r.prediction_id))
        return rs if now is None else [r for r in rs if as_date(r.matured_at) < as_date(now)]

    def step(self, outcomes: OutcomeLedger, now) -> list[ErrorReport]:
        new = []
        for exp, out in outcomes.matured(now):
            if exp.prediction_id in self._reports:
                continue
            h = outcomes.expectations.meta(exp.prediction_id)["content_hash"]
            rep = compute_errors(exp, h, out, now, self.cfg)
            self.lane.append({"prediction_id": rep.prediction_id, "report_hash": rep.report_hash, "report": rep.content()})
            self._reports[rep.prediction_id] = rep
            if self.tracker is not None:
                feed_surprise(self.tracker, exp, rep, now)
            new.append(rep)
        return new

    def records(self, outcomes: OutcomeLedger, now) -> list[ErrorRecord]:
        """ErrorRecords (the error-research feed) for every evaluated prediction that matured strictly before `now`."""
        return [error_record(e, o, self._reports[e.prediction_id]) for e, o in outcomes.matured(now)
                if e.prediction_id in self._reports]

    def confident_failures(self, now=None) -> list[ErrorReport]:
        return [r for r in self.reports(now) if r.confident_wrong]

    def summary(self, now, n_unscored: int = 0) -> dict:
        rs = self.reports(now)
        return {"n": len(rs), "profiles": all_profiles(rs), "confidence": confidence_reliability(rs),
                "tolerance": honest_tolerance(rs, n_unscored, self.cfg), "confident_wrong": len(self.confident_failures(now))}

    def verify(self) -> dict:
        def check(body: dict) -> str | None:
            try:
                rep = ErrorReport.from_record(body["report"])
            except (KeyError, TypeError, ValueError) as ex:
                return f"unreadable error report: {ex}"
            return None if rep.report_hash == body.get("report_hash") else f"error report {body.get('prediction_id')} does not match its hash"
        return self.lane.verify(body_check=check)

    def assert_intact(self) -> dict:
        rep = self.verify()
        if not rep["ok"]:
            raise LedgerTampered("; ".join(rep["problems"]))
        return rep
