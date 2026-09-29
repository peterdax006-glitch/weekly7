"""Pattern change detection (C68 checklist H; builds on C62 sections 10-12 and 46, Bible phases 9/10). IMPLEMENTED - NOT VALIDATED.

Every important pattern carries a continuously updated reliability profile: historical vs recent reliability, reliability by regime,
volatility, sector, holding period and confidence bin, recent error, error direction and magnitude, degradation rate and recovery
rate. From it each pattern is classified as exactly one of: temporary NOISE, NORMAL_VARIANCE, WEAKENING, STRENGTHENING,
REGIME_SPECIFIC_FAILURE, STRUCTURAL_CHANGE, OBSOLESCENCE, RETURNING (or INSUFFICIENT_EVIDENCE, never guessed).

A failing pattern is never deleted: the verdict's action is INVESTIGATE until an investigation record exists, and only an
OBSOLESCENCE verdict that has been investigated may PROPOSE retirement (the gate that retires stays in engine.learning.retirement).

Everything is forward-in-time: `classify` slices the observation frame to rows strictly before `now`, so its verdict at t cannot
change when the data after t changes (tested by scrambling the future). Mechanisms are the existing ones, not copies:
engine.learning.reliability (AR(1) deflation, discounted stats, posterior), lifecycle (causal stage trace, deterioration shape by
BIC + block permutation, recovery bar) and health (nine-state monitor, reported next to ours for comparison).
Public entry: `step(state, now, frames)`."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import health as HL
from engine.learning import lifecycle as LC
from engine.learning import reliability as RL
from engine.learning.core import FirewallBreach, Health, Lifecycle, _StrEnum, as_date, stable_hash

DIMENSIONS = ("regime", "volatility", "sector", "holding_period", "confidence")
PARAMS: dict[str, Any] = {
    "recent_n": 26,            # rows that count as "recent"
    "min_hist": 26,            # historical rows needed before recent-vs-history means anything
    "min_cell": 6,             # rows a dimension level needs in the recent window before it may accuse or clear a pattern
    "z_weak": 1.0,             # |z| below this = ordinary variance
    "z_strong": 2.0,           # |z| at or above this = a real change (before the outlier check)
    "t_cell_fail": 2.0,        # a recent dimension level with t at or below -this is failing
    "trim_frac": 1 / 13,       # share of the recent window dropped (worst first) for the outlier-robust re-test
    "shape_span": 78,          # rows handed to lifecycle.classify_deterioration
    "obsolete_n": 104,         # consecutive failing rows (~2 years weekly), unrecovered, before OBSOLESCENCE may be said
    "return_window": 104,      # an established pattern that began failing within this many rows and now delivers again is RETURNING
    "t_return": 2.0,           # ... and its recent window must clear this t
    "conf_bins": 3, "slope_window": 78, "seed": 0,     # slope over ~1.5 years: 26 noisy rows cannot resolve a decay rate
}


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


class ChangeClass(_StrEnum):
    NOISE = "NOISE"
    NORMAL_VARIANCE = "NORMAL_VARIANCE"
    WEAKENING = "WEAKENING"
    STRENGTHENING = "STRENGTHENING"
    REGIME_SPECIFIC_FAILURE = "REGIME_SPECIFIC_FAILURE"
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
    OBSOLESCENCE = "OBSOLESCENCE"
    RETURNING = "RETURNING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Action(_StrEnum):
    KEEP = "KEEP"
    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"                         # the default answer to any failure: find out why first
    REDUCE_IN_CONTEXT = "REDUCE_IN_CONTEXT"             # regime-specific: down-weight only where it fails
    PROPOSE_RETIREMENT = "PROPOSE_RETIREMENT"           # obsolete AND already investigated; a gate still decides
    COLLECT_DATA = "COLLECT_DATA"


FAILING = (ChangeClass.WEAKENING, ChangeClass.REGIME_SPECIFIC_FAILURE, ChangeClass.STRUCTURAL_CHANGE, ChangeClass.OBSOLESCENCE)
HEALTH_OF = {ChangeClass.NOISE: Health.HEALTHY, ChangeClass.NORMAL_VARIANCE: Health.HEALTHY, ChangeClass.STRENGTHENING: Health.HEALTHY,
             ChangeClass.WEAKENING: Health.DEGRADING, ChangeClass.REGIME_SPECIFIC_FAILURE: Health.UNSTABLE,
             ChangeClass.STRUCTURAL_CHANGE: Health.BROKEN, ChangeClass.OBSOLESCENCE: Health.BROKEN,
             ChangeClass.RETURNING: Health.RECOVERING, ChangeClass.INSUFFICIENT_EVIDENCE: Health.INSUFFICIENT_EVIDENCE}


# ------------------------------------------------------------------------------------------------ observations
def causal_slice(frame: pd.DataFrame, now, strict: bool = False) -> pd.DataFrame:
    """Rows whose outcome matured strictly before `now`. strict=True raises FirewallBreach when a later row is present."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("observation frame needs a DatetimeIndex (the date each outcome matured)")
    cut = pd.Timestamp(as_date(now))
    late = frame.index >= cut
    if strict and late.any():
        raise FirewallBreach(f"observation dated {frame.index[late][0].date()} is not strictly before now={as_date(now)}")
    return frame.loc[~late].sort_index()


def validate_frame(frame: pd.DataFrame) -> list[str]:
    """Structural problems that would make a profile lie: no effect column, duplicated dates, non-finite effects."""
    errs = []
    if "effect" not in frame.columns:
        return ["frame needs an 'effect' column (signed outcome, higher = the pattern worked)"]
    if not isinstance(frame.index, pd.DatetimeIndex):
        return ["frame index must be a DatetimeIndex"]
    if frame.index.has_duplicates:
        errs.append("duplicate dates: one outcome per matured date")
    if np.isinf(frame["effect"].astype(float).to_numpy()).any():
        errs.append("effect contains infinities")
    if "predicted" in frame.columns and np.isinf(frame["predicted"].astype(float).to_numpy()).any():
        errs.append("predicted contains infinities")
    return errs


def _finite(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def _t(v: np.ndarray) -> float:
    if len(v) < 3:
        return 0.0
    sd = float(v.std(ddof=1))
    return float(v.mean() / (sd / math.sqrt(len(v)))) if sd > 1e-12 else (math.copysign(9.0, v.mean()) if abs(v.mean()) > 1e-12 else 0.0)


# ------------------------------------------------------------------------------------------------ the reliability profile
@dataclasses.dataclass(frozen=True)
class Cell:
    """Reliability of a pattern inside one level of one dimension, over all history and over the recent window."""
    dimension: str
    level: str
    n: int
    mean: float
    hit_rate: float
    t: float
    recent_n: int
    recent_mean: float
    recent_t: float
    mean_error: float | None      # realised minus predicted, when predictions were recorded


@dataclasses.dataclass(frozen=True)
class ReliabilityProfile:
    pattern_id: str
    as_of: str
    n: int
    hist_n: int
    recent_n: int
    hist_mean: float
    hist_sd: float
    recent_mean: float
    hist_reliability: float | None          # P(long-run effect > 0)
    recent_reliability: float | None        # P(recent effect >= a share of what it used to deliver)
    z: float                                # recent vs historical, AR(1) deflated
    z_trimmed: float                        # same after dropping the most extreme recent rows in the deviation's direction
    error_mean: float | None                # recent realised - predicted
    error_direction: str                    # OVERPREDICTED / UNDERPREDICTED / BALANCED / UNMEASURED
    error_mae: float | None
    error_over_share: float | None          # share of recent rows where the pattern predicted more than it delivered
    degradation_rate: float                 # slope of the trailing effect per row over the slope window (negative = decaying)
    recovery_rate: float                    # recoveries per failure spell in the lifecycle trace (nan = never failed)
    cells: tuple
    confidence_gap: float | None            # mean stated confidence minus hit rate in the recent window

    def cell(self, dimension: str, level: str) -> Cell | None:
        return next((c for c in self.cells if c.dimension == dimension and c.level == level), None)


def _ar1_se(sd: float, rho: float, n: int) -> float:
    return sd * math.sqrt(max((1.0 + rho) / (1.0 - rho), 0.2) / max(n, 1))


def _trim_z(recent: np.ndarray, hist_mean: float, se_h: float, sd_h: float, rho: float, frac: float, sign: float) -> float:
    k = max(1, int(round(len(recent) * frac)))
    order = np.argsort(-sign * recent)                      # drop the rows that push hardest in the deviation's direction
    keep = np.delete(recent, order[:k]) if len(recent) > k + 2 else recent
    se = math.sqrt(_ar1_se(sd_h, rho, len(keep)) ** 2 + se_h ** 2)
    return float((keep.mean() - hist_mean) / se) if se > 1e-12 else 0.0


def _confidence_bin(conf: pd.Series, hist_index: pd.Index, bins: int) -> pd.Series:
    """Bin edges from HISTORY only, so a recent row cannot move the edges that judge it."""
    h = conf.loc[hist_index].dropna()
    if len(h) < bins * 3:
        return pd.Series("ALL", index=conf.index)
    edges = np.unique(np.quantile(h, np.linspace(0, 1, bins + 1)[1:-1]))
    lab = np.array([f"C{i}" for i in range(len(edges) + 1)])
    return pd.Series(lab[np.searchsorted(edges, conf.to_numpy(float), side="right")], index=conf.index).where(conf.notna(), "NA")


def dimension_cells(frame: pd.DataFrame, recent_n: int, conf_bins: int = 3) -> list[Cell]:
    """Reliability of the pattern by regime / volatility / sector / holding period / confidence bin."""
    out: list[Cell] = []
    if not len(frame):
        return out
    recent_idx = frame.index[-recent_n:]
    cols = {"regime": "regime", "volatility": "volatility", "sector": "sector", "holding_period": "hold"}
    keyed = {d: frame[c].astype(str) for d, c in cols.items() if c in frame.columns}
    if "confidence" in frame.columns:
        keyed["confidence"] = _confidence_bin(frame["confidence"].astype(float), frame.index.difference(recent_idx), conf_bins)
    eff = frame["effect"].astype(float)
    err = (frame["effect"] - frame["predicted"]).astype(float) if "predicted" in frame.columns else None
    for dim, lab in keyed.items():
        for level in sorted(lab.unique()):
            m = (lab == level).to_numpy()
            v = _finite(eff[m])
            r = _finite(eff[m & frame.index.isin(recent_idx)])
            e = _finite(err[m]) if err is not None else np.array([])
            out.append(Cell(dim, str(level), len(v), float(v.mean()) if len(v) else float("nan"),
                            float((v > 0).mean()) if len(v) else float("nan"), _t(v), len(r),
                            float(r.mean()) if len(r) else float("nan"), _t(r), float(e.mean()) if len(e) else None))
    return out


def profile(pattern_id: str, frame: pd.DataFrame, now, cfg=None) -> ReliabilityProfile:
    """The full checklist-H profile of one pattern as of `now` (rows before `now` only)."""
    P = _cfg(cfg)
    errs = validate_frame(frame)
    if errs:
        raise ValueError(f"pattern {pattern_id}: {errs}")
    f = causal_slice(frame, now)
    x = f["effect"].astype(float).to_numpy()
    n = len(x)
    rn = min(P["recent_n"], max(n // 3, 1))
    hist, recent = _finite(x[:n - rn]), _finite(x[n - rn:])
    nan = float("nan")
    if n < P["recent_n"] + P["min_cell"] or len(hist) < 3 or len(recent) < 3:
        return ReliabilityProfile(pattern_id, str(as_date(now)), n, len(hist), len(recent), nan, nan, nan, None, None, 0.0, 0.0,
                                  None, "UNMEASURED", None, None, 0.0, nan, tuple(), None)
    rho = RL.ar1_rho(hist)
    sd_h = float(hist.std(ddof=1))
    se_h = _ar1_se(sd_h, rho, len(hist))
    se = math.sqrt(_ar1_se(sd_h, rho, len(recent)) ** 2 + se_h ** 2)
    z = float((recent.mean() - hist.mean()) / se) if se > 1e-12 else 0.0
    zt = _trim_z(recent, float(hist.mean()), se_h, sd_h, rho, P["trim_frac"], -1.0 if z < 0 else 1.0)
    truth, td = RL.truth_confidence(hist)
    cur, _ = RL.current_reliability(x[np.isfinite(x)], td) if truth is not None else (None, {})
    e_mean = e_mae = over = None
    direction = "UNMEASURED"
    conf_gap = None
    if "predicted" in f.columns:
        rf = f.iloc[n - rn:]
        e = (rf["effect"] - rf["predicted"]).astype(float).to_numpy()
        e = e[np.isfinite(e)]
        if len(e) >= 3:
            e_mean, e_mae, over = float(e.mean()), float(np.abs(e).mean()), float((e < 0).mean())
            band = 0.5 * float(e.std(ddof=1)) / math.sqrt(len(e))
            direction = "BALANCED" if abs(e_mean) <= max(band, 1e-12) else ("UNDERPREDICTED" if e_mean > 0 else "OVERPREDICTED")
    if "confidence" in f.columns:
        c = f["confidence"].astype(float).iloc[n - rn:]
        ok = c.notna().to_numpy() & np.isfinite(f["effect"].astype(float).iloc[n - rn:].to_numpy())
        if ok.sum() >= 3:
            conf_gap = float(c.to_numpy()[ok].mean() - (f["effect"].astype(float).iloc[n - rn:].to_numpy()[ok] > 0).mean())
    return ReliabilityProfile(pattern_id, str(as_date(now)), n, len(hist), len(recent), float(hist.mean()), sd_h, float(recent.mean()),
                              truth, cur, z, zt, e_mean, direction, e_mae, over, degradation_rate(x, P["slope_window"]),
                              recovery_rate(x, P), tuple(dimension_cells(f, rn, P["conf_bins"])), conf_gap)


def degradation_rate(x: np.ndarray, window: int) -> float:
    """OLS slope of the last `window` outcomes per row (negative = decaying). 0 when there are too few finite rows."""
    v = np.asarray(x, float)[-window:]
    v = v[np.isfinite(v)]
    if len(v) < 6:
        return 0.0
    t = np.arange(len(v), dtype=float)
    t -= t.mean()
    return float((t @ (v - v.mean())) / (t @ t))


def recovery_rate(x: np.ndarray, P: Mapping) -> float:
    """Recoveries per failure spell in the causal lifecycle trace; NaN when the pattern never failed."""
    v = np.asarray(x, float)
    if len(v) < 2 * P["recent_n"]:
        return float("nan")
    tr = LC.trace(v)
    fails = sum(c.to == Lifecycle.FAILURE.value for c in tr.changes)
    recov = sum(c.to == Lifecycle.RECOVERY.value for c in tr.changes)
    return float(recov / fails) if fails else float("nan")


# ------------------------------------------------------------------------------------------------ the verdict
@dataclasses.dataclass(frozen=True)
class PatternVerdict:
    pattern_id: str
    as_of: str
    change: ChangeClass
    action: Action
    health: Health
    reasons: tuple
    culprit: tuple | None                     # (dimension, level) for a regime-specific failure
    deterioration: str | None                 # lifecycle.Deterioration name when a shape was fitted
    profile: ReliabilityProfile
    lifecycle_stage: str
    failing_rows: int
    investigated: bool

    delete_allowed = False                    # class-level: no verdict ever authorises deletion (checklist H)

    @property
    def needs_investigation(self) -> bool:
        return self.change in FAILING and not self.investigated

    def record_id(self) -> str:
        return stable_hash([self.pattern_id, self.as_of, self.change, self.action, self.culprit])


def _failing_run(stage: np.ndarray) -> int:
    """Length of the trailing run of FAILURE / DECAY / DEGRADED stages."""
    bad = {Lifecycle.FAILURE.value, Lifecycle.DECAY.value, Lifecycle.DEGRADED.value}
    n = 0
    for s in stage[::-1]:
        if str(s) not in bad:
            break
        n += 1
    return n


def _accuse_cells(prof: ReliabilityProfile, frame: pd.DataFrame, P: Mapping) -> tuple | None:
    """A dimension level failing recently while the REST of the pattern's recent rows still deliver: (dimension, level)."""
    if not len(frame):
        return None
    rn = prof.recent_n
    recent = frame.iloc[-rn:]
    cols = {"regime": "regime", "volatility": "volatility", "sector": "sector", "holding_period": "hold"}
    best = None
    for dim, c in cols.items():
        if c not in recent.columns:
            continue
        lab = recent[c].astype(str)
        for level in sorted(lab.unique()):
            inn = _finite(recent["effect"].astype(float)[(lab == level).to_numpy()])
            out = _finite(recent["effect"].astype(float)[(lab != level).to_numpy()])
            if len(inn) < P["min_cell"] or len(out) < P["min_cell"]:
                continue
            ti = _t(inn)
            gap_out = (out.mean() - prof.hist_mean) / max(prof.hist_sd / math.sqrt(len(out)), 1e-12)
            if ti <= -P["t_cell_fail"] and gap_out > -P["z_weak"] and (best is None or ti < best[0]):
                best = (ti, dim, level)
    return None if best is None else (best[1], best[2])


def classify(pattern_id: str, frame: pd.DataFrame, now, cfg=None, families: Mapping[str, pd.DataFrame | None] | None = None,
             investigated: bool = False, seed: int | None = None) -> PatternVerdict:
    """Decide which of the checklist-H classes a pattern is in at `now`. `families` (optional context frames aligned to `frame`) lets
    lifecycle.classify_deterioration link a decline to regime / volatility / sector; `investigated` records that a real
    investigation of the failure exists (required before retirement can even be proposed)."""
    P = _cfg(cfg)
    f = causal_slice(frame, now)
    prof = profile(pattern_id, f, now, P)
    asof = str(as_date(now))
    if prof.hist_n < P["min_hist"] or prof.recent_n < P["min_cell"] or not math.isfinite(prof.z):
        return PatternVerdict(pattern_id, asof, ChangeClass.INSUFFICIENT_EVIDENCE, Action.COLLECT_DATA, Health.INSUFFICIENT_EVIDENCE,
                              (f"only {prof.hist_n} historical and {prof.recent_n} recent outcomes",), None, None, prof,
                              Lifecycle.BIRTH.value, 0, investigated)
    x = f["effect"].astype(float).to_numpy()
    tr = LC.trace(x)
    stage = tr.stage
    cur_stage = str(tr.current)
    failing = _failing_run(stage)
    why: list[str] = [f"recent-vs-history z={prof.z:.2f} (outlier-trimmed {prof.z_trimmed:.2f}); lifecycle stage {cur_stage}"]
    culprit = _accuse_cells(prof, f, P)
    det: str | None = None
    ff = {k: (None if v is None else v.loc[v.index.isin(f.index)]) for k, v in (families or {}).items()}
    est = (Lifecycle.ACTIVE.value, Lifecycle.PEAK.value, Lifecycle.DECAY.value, Lifecycle.DEGRADED.value, Lifecycle.RECOVERY.value)
    had_failure = any(c.to == Lifecycle.FAILURE.value and c.frm in est for c in tr.changes)      # an ESTABLISHED pattern failed
    fail_at = max([c.at for c in tr.changes if c.to == Lifecycle.FAILURE.value and c.frm in est] or [-10 ** 9])
    returned = (had_failure and len(x) - fail_at <= P["return_window"] and prof.recent_mean > 0 and prof.z > -P["z_strong"]
                and fail_at < len(x) - prof.recent_n and _t(_finite(x[-prof.recent_n:])) >= P["t_return"])
    if culprit is not None:
        cls = ChangeClass.REGIME_SPECIFIC_FAILURE
        why.append(f"failing only where {culprit[0]}={culprit[1]}; the other recent rows still deliver")
    elif returned:
        cls = ChangeClass.RETURNING
        why.append("a failure spell ended recently and the pattern now delivers again")
    elif prof.z >= P["z_strong"]:
        cls = ChangeClass.STRENGTHENING if prof.z_trimmed >= P["z_weak"] else ChangeClass.NOISE
        why.append("recent gain survives dropping the best rows" if cls == ChangeClass.STRENGTHENING else "gain rests on a few outlier rows")
    elif prof.z <= -P["z_weak"]:
        span = min(len(x), P["shape_span"])
        dv = LC.classify_deterioration(x, ff or None, start=len(x) - span, cfg=None, seed=P["seed"] if seed is None else seed)
        det = dv.kind.value
        why.append(f"deterioration shape {det}")
        rec = LC.assess_recovery(x, len(x) - failing, len(x)) if failing else None
        if prof.z_trimmed <= -P["z_weak"] and prof.recent_mean <= 0 and failing >= P["obsolete_n"] and rec is not None and not rec.recovered:
            cls = ChangeClass.OBSOLESCENCE
            why.append(f"failing for {failing} rows with no recovery: {rec.why}")
        elif dv.linked_family in ("regime", "volatility", "sector", "liquidity"):
            cls = ChangeClass.REGIME_SPECIFIC_FAILURE
            culprit = (dv.linked_family, "linked")
            why.append(f"decline explained by the {dv.linked_family} family")
        elif prof.z_trimmed > -P["z_weak"]:
            cls = ChangeClass.NOISE
            why.append("decline disappears when the worst rows are dropped")
        elif prof.z > -P["z_strong"] and dv.kind.value in ("RANDOM", "INSUFFICIENT"):
            cls = ChangeClass.NOISE
            why.append("modest decline that no trend or step explains beyond chance")
        elif dv.kind.value == "ABRUPT" and prof.z <= -P["z_strong"]:
            cls = ChangeClass.STRUCTURAL_CHANGE
            why.append(f"step of {dv.shape.step_size:.4g} at row {dv.shape.step_at}")
        else:
            cls = ChangeClass.WEAKENING
    else:
        cls = ChangeClass.NORMAL_VARIANCE
        why.append("recent behaviour is within the historical band")
    if cls in FAILING:
        act = Action.PROPOSE_RETIREMENT if (cls == ChangeClass.OBSOLESCENCE and investigated) else \
            Action.REDUCE_IN_CONTEXT if (cls == ChangeClass.REGIME_SPECIFIC_FAILURE and investigated) else Action.INVESTIGATE
    else:
        act = Action.MONITOR if cls in (ChangeClass.NOISE, ChangeClass.RETURNING) else Action.KEEP
    return PatternVerdict(pattern_id, asof, cls, act, HEALTH_OF[cls], tuple(why), culprit, det, prof, cur_stage, failing, investigated)


def health_cross_check(pattern_id: str, frame: pd.DataFrame, now, verdict: PatternVerdict, cfg=None) -> dict:
    """Run the C62 health monitor on the same outcomes and say whether the two agree that the pattern is failing. A disagreement is
    reported, not resolved: neither is the judge."""
    f = causal_slice(frame, now)
    if len(f) < 3:
        return {"health": Health.UNKNOWN.value, "agree_failing": None, "ours": verdict.health.value}
    rec = HL.assess([HL.HealthInput(pattern_id, f["effect"].astype(float))], now, cfg)[0]
    theirs = HL.is_failing(rec.state)
    ours = verdict.change in FAILING
    return {"health": rec.state.value, "ours": verdict.health.value, "agree_failing": theirs == ours,
            "health_failing": theirs, "our_failing": ours}


# ------------------------------------------------------------------------------------------------ state and the daily entry
@dataclasses.dataclass
class PatternChangeState:
    """Everything the monitor remembers between days: the latest verdict, the class history and which patterns have been
    investigated. History is append-only; a class is only debounced (reported as 'stable') after two consecutive equal days."""
    latest: dict = dataclasses.field(default_factory=dict)
    history: dict = dataclasses.field(default_factory=dict)
    investigated: set = dataclasses.field(default_factory=set)

    def mark_investigated(self, pattern_id: str) -> None:
        self.investigated.add(pattern_id)

    def stable_class(self, pattern_id: str) -> ChangeClass | None:
        h = self.history.get(pattern_id, [])
        return h[-1][1] if len(h) >= 2 and h[-1][1] == h[-2][1] else None

    def flap_count(self, pattern_id: str) -> int:
        h = [c for _, c in self.history.get(pattern_id, [])]
        return sum(1 for a, b in zip(h, h[1:]) if a != b)


@dataclasses.dataclass(frozen=True)
class ChangeReport:
    now: str
    verdicts: tuple
    investigations_needed: tuple           # pattern ids the what-changed tree must open
    proposed_retirements: tuple
    counts: Mapping[str, int]


def step(state: PatternChangeState, now, frames: Mapping[str, pd.DataFrame], cfg=None,
         families: Mapping[str, Mapping[str, pd.DataFrame | None]] | None = None) -> ChangeReport:
    """The research loop's daily entry: classify every pattern as of `now` and hand back who must be investigated."""
    out, need, retire = [], [], []
    for pid in sorted(frames):
        v = classify(pid, frames[pid], now, cfg, (families or {}).get(pid), investigated=pid in state.investigated)
        state.latest[pid] = v
        h = state.history.setdefault(pid, [])
        if not h or h[-1][0] != v.as_of:
            h.append((v.as_of, v.change))
        out.append(v)
        if v.needs_investigation:
            need.append(pid)
        if v.action == Action.PROPOSE_RETIREMENT:
            retire.append(pid)
    counts: dict[str, int] = {}
    for v in out:
        counts[v.change.value] = counts.get(v.change.value, 0) + 1
    return ChangeReport(str(as_date(now)), tuple(out), tuple(need), tuple(retire), counts)


def verdict_table(report: ChangeReport) -> pd.DataFrame:
    rows = [{"pattern": v.pattern_id, "change": v.change.value, "action": v.action.value, "health": v.health.value,
             "z": v.profile.z, "z_trimmed": v.profile.z_trimmed, "recent": v.profile.recent_mean, "hist": v.profile.hist_mean,
             "error_direction": v.profile.error_direction, "degradation": v.profile.degradation_rate, "stage": v.lifecycle_stage,
             "culprit": "" if v.culprit is None else "/".join(v.culprit)} for v in report.verdicts]
    return pd.DataFrame(rows)


def false_alarm_rate(n_patterns: int, n_rows: int, seed: int, cfg=None, now_offset: int = 0) -> dict:
    """Null calibration: patterns with a constant true effect and iid noise. The share classed as failing is the false-alarm rate a
    reader must expect from this classifier on patterns that did not change at all."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=n_rows + 1)
    failing = 0
    for _ in range(n_patterns):
        fr = pd.DataFrame({"effect": rng.normal(0.004, 0.02, n_rows)}, index=idx[:n_rows])
        v = classify("null", fr, idx[n_rows - now_offset], cfg)
        failing += v.change in FAILING
    return {"patterns": n_patterns, "failing": failing, "rate": failing / max(n_patterns, 1)}
