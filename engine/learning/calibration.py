"""Calibration (contract C62 section 54; checklist J08 calibration monitoring, supports B14 dynamic influence). Bible phase
serving: PHASE 13 direction calibration generalised to any confidence the learner states ("confidence must correspond to
reality"; if the learner becomes overconfident, reduce influence).

Tracks predicted confidence versus actual success and turns the gap into behaviour:
  reliability diagram (Wilson intervals), ECE / MCE / Brier with the Murphy decomposition, log loss, overconfidence,
  logistic recalibration slope with a confidence interval (slope < 1 = predictions too extreme), an ECE null distribution by
  seeded simulation (is this gap bigger than sampling noise?), Hosmer-Lemeshow, confidence drift (rolling ECE / signed gap
  trend plus a change-point scan), per-context calibration with empirical-Bayes shrinkage and FDR control, and the
  INFLUENCE-REDUCTION RULE: significant overconfidence scales a knowledge item's influence by (real edge / claimed edge),
  never above 1 (underconfidence does not buy extra influence) and never below a floor (silencing is the retirement gate's job).

Built on: engine.pattern_reliability (brier, equal-mass ece), engine.pattern_stats (calibration_table under every binned
diagram, bh_reject for FDR). engine/direction_calib.py fits per-type Platt/Venn-Abers maps for the direction model; this
module is the monitor that judges any stated confidence after the fact (the Newton logistic slope with a CI is new:
direction_calib's fit has no standard errors). Every function that reads records takes `now` and only
sees records whose outcome matured strictly before it."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps
from scipy.special import expit

from engine import pattern_reliability, pattern_stats

from .core import FirewallBreach, Health, as_date, canonical_json, require_past, stable_hash
from .surprise import benjamini_hochberg


# ------------------------------------------------------------------------------------------------- basic metrics

@dataclasses.dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_p: float
    freq: float
    ci_lo: float
    ci_hi: float

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive = predicted more than happened (over-prediction)."""
        return self.mean_p - self.freq


def _check_py(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-d arrays of equal length")
    if len(p) and (np.isnan(p).any() or p.min() < 0 or p.max() > 1):
        raise ValueError("probabilities must lie in [0, 1] and not be NaN")
    if len(y) and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def reliability_diagram(p, y, n_bins: int = 10, strategy: str = "uniform") -> list[Bin]:
    """Bins of predicted probability with the observed frequency of success in each, built on
    engine.pattern_stats.calibration_table (this module adds Wilson intervals, it does not add another table). 'quantile'
    gives equal-count edges, the honest choice when predictions cluster; empty bins are omitted, never zero-filled."""
    p, y = _check_py(p, y)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"unknown strategy {strategy}")
    if len(p) == 0:
        return []
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges = np.array([0.0, 1.0]) if len(edges) < 2 else edges
        edges[0], edges[-1] = 0.0, 1.0
    edges = edges.copy()
    edges[-1] = 1.0000001                                      # calibration_table bins are [lo, hi): keep p == 1.0 in the top bin
    tab = pattern_stats.calibration_table(p, y.astype(bool), tuple(float(e) for e in edges))
    out = []
    for row in tab.itertuples():
        if row.n == 0:
            continue
        lo, hi = wilson(row.real_share * row.n, row.n)
        out.append(Bin(float(row.lo), float(row.hi), int(row.n), float(row.mean_p), float(row.real_share), lo, hi))
    return out


def ece(bins: Sequence[Bin]) -> float:
    n = sum(b.n for b in bins)
    return float(sum(b.n * abs(b.gap) for b in bins) / n) if n else float("nan")


def mce(bins: Sequence[Bin], min_n: int = 5) -> float:
    g = [abs(b.gap) for b in bins if b.n >= min_n]
    return float(max(g)) if g else float("nan")


def signed_gap(p, y) -> float:
    p, y = _check_py(p, y)
    return float(p.mean() - y.mean()) if len(p) else float("nan")


def brier(p, y) -> float:
    """Brier score; the arithmetic is engine.pattern_reliability.brier, this adds input validation."""
    p, y = _check_py(p, y)
    return pattern_reliability.brier(p, y)


def ece_equal_mass(p, y, n_bins: int = 10) -> float:
    """ECE over equal-mass bins - engine.pattern_reliability.ece, validated. NaN when there are fewer than 3 rows per bin."""
    p, y = _check_py(p, y)
    return pattern_reliability.ece(p, y, n_bins)


def brier_decomposition(p, y, n_bins: int = 10) -> dict[str, float]:
    """Murphy: Brier ~ reliability - resolution + uncertainty. Reliability is what calibration repairs; resolution is skill.
    The identity is exact only for constant-within-bin forecasts, so `residual` reports the within-bin remainder."""
    p, y = _check_py(p, y)
    bins = reliability_diagram(p, y, n_bins)
    n, base = len(p), float(y.mean())
    rel = sum(b.n * (b.mean_p - b.freq) ** 2 for b in bins) / n
    res = sum(b.n * (b.freq - base) ** 2 for b in bins) / n
    unc = base * (1 - base)
    bs = brier(p, y)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc, "residual": bs - (rel - res + unc)}


def log_loss(p, y, eps: float = 1e-9) -> float:
    p, y = _check_py(p, y)
    q = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))) if len(p) else float("nan")


def overconfidence(p, y) -> dict[str, float]:
    """Confidence in the predicted side vs how often that side was right. excess > 0 means overconfident. `edge_ratio` is
    realised edge over claimed edge (accuracy-0.5)/(confidence-0.5), the quantity the influence rule scales by."""
    p, y = _check_py(p, y)
    if len(p) == 0:
        return {"confidence": float("nan"), "accuracy": float("nan"), "excess": float("nan"), "edge_ratio": float("nan")}
    side = p >= 0.5
    conf = np.where(side, p, 1 - p)
    ok = (y == 1.0) == side
    c, a = float(conf.mean()), float(ok.mean())
    claimed = c - 0.5
    ratio = max(a - 0.5, 0.0) / claimed if claimed > 1e-9 else 1.0
    return {"confidence": c, "accuracy": a, "excess": c - a, "edge_ratio": min(ratio, 1.0)}


def _logit(p, eps=1e-6):
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


def platt_slope(p, y, ridge: float = 1e-6, iters: int = 60) -> dict[str, float]:
    """Logistic recalibration  P(y=1) = expit(a + b * logit(p)).  b < 1: predictions are too extreme (overconfident);
    b > 1: too timid. Newton-Raphson; standard errors from the inverse Fisher information."""
    p, y = _check_py(p, y)
    n = len(p)
    if n < 10 or y.min() == y.max():
        return {"a": float("nan"), "b": float("nan"), "se_a": float("nan"), "se_b": float("nan"), "b_lo": float("nan"),
                "b_hi": float("nan"), "n": n}
    z = _logit(p)
    a, b = 0.0, 1.0
    H = np.eye(2)
    for _ in range(iters):
        q = expit(a + b * z)
        wv = np.clip(q * (1 - q), 1e-9, None)
        g = np.array([np.sum(q - y) + ridge * a, np.sum((q - y) * z) + ridge * (b - 1.0)])
        H = np.array([[wv.sum() + ridge, (wv * z).sum()], [(wv * z).sum(), (wv * z * z).sum() + ridge]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.pinv(H)
    sa, sb = math.sqrt(max(cov[0, 0], 0)), math.sqrt(max(cov[1, 1], 0))
    return {"a": float(a), "b": float(b), "se_a": sa, "se_b": sb, "b_lo": float(b - 1.96 * sb), "b_hi": float(b + 1.96 * sb),
            "n": n}


def ece_null_pvalue(p, n_bins: int, observed: float, rng: np.random.Generator, n_sim: int = 400) -> float:
    """P(equal-mass ECE >= observed) if the forecasts were perfectly calibrated: outcomes redrawn as Bernoulli(p). ECE is biased upward on
    small samples, so a raw ECE of 0.05 means nothing until compared with this distribution."""
    p = np.asarray(p, float)
    if len(p) == 0 or not math.isfinite(observed):
        return float("nan")                                  # nothing to test: callers must not read this as "calibrated"
    sims = np.empty(n_sim)
    for i in range(n_sim):
        ys = (rng.random(len(p)) < p).astype(float)
        sims[i] = pattern_reliability.ece(p, ys, n_bins)
    return float((1 + np.sum(sims >= observed - 1e-12)) / (n_sim + 1))


def hosmer_lemeshow(bins: Sequence[Bin]) -> tuple[float, float, int]:
    """Chi-square goodness of calibration over bins. Returns (statistic, p, df)."""
    use = [b for b in bins if 0 < b.mean_p < 1 and b.n >= 5]
    if len(use) < 3:
        return float("nan"), float("nan"), 0
    stat = sum(b.n * (b.freq - b.mean_p) ** 2 / (b.mean_p * (1 - b.mean_p)) for b in use)
    df = len(use) - 2
    return float(stat), float(sps.chi2.sf(stat, df)), df


# ------------------------------------------------------------------------------------------------- recalibration maps

class IsotonicCalibrator:
    """Monotone recalibration by weighted pool-adjacent-violators, written out so it is deterministic and dependency-free."""

    def __init__(self):
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def fit(self, p, y) -> "IsotonicCalibrator":
        p, y = _check_py(p, y)
        if len(p) == 0:
            raise ValueError("cannot fit on no data")
        o = np.argsort(p, kind="stable")
        xs, ys = p[o], y[o]
        # blocks of (value, weight, right-edge x); merge backwards while monotonicity is violated
        bv, bw, bx = [], [], []
        for v, w, x in zip(ys, np.ones(len(ys)), xs):
            bv.append(v), bw.append(w), bx.append(x)
            while len(bv) > 1 and bv[-2] > bv[-1]:
                w2 = bw[-2] + bw[-1]
                bv[-2] = (bv[-2] * bw[-2] + bv[-1] * bw[-1]) / w2
                bw[-2], bx[-2] = w2, bx[-1]
                bv.pop(), bw.pop(), bx.pop()
        self.x, self.y = np.array(bx), np.array(bv)
        return self

    def predict(self, p) -> np.ndarray:
        if self.x.size == 0:
            raise RuntimeError("calibrator not fitted")
        return np.interp(np.asarray(p, float), self.x, self.y)


class PlattCalibrator:
    """Logistic map expit(a + b * logit p) fitted on past records; identity when there is too little data to fit."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b

    def fit(self, p, y) -> "PlattCalibrator":
        r = platt_slope(p, y)
        if math.isfinite(r["b"]):
            self.a, self.b = r["a"], r["b"]
        return self

    def predict(self, p) -> np.ndarray:
        return expit(self.a + self.b * _logit(np.asarray(p, float)))


# ------------------------------------------------------------------------------------------------- drift

@dataclasses.dataclass(frozen=True)
class DriftReport:
    n_windows: int
    ece_by_window: tuple[float, ...]
    gap_by_window: tuple[float, ...]
    ece_slope: float                       # per window
    ece_slope_p: float
    gap_slope: float
    gap_slope_p: float
    scan_stat: float                       # largest standardised mean shift over all candidate change points
    change_index: int | None               # record index (matured order) where the shift is estimated to begin; None = no alarm
    change_date: str | None
    shift_direction: str | None            # "overconfident" (confidence exceeds accuracy more than before) or "underconfident"


def _confidence_residual(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-forecast standardised residual in the CONFIDENCE frame: (confidence - correct) / sd. Mean zero and unit variance
    when calibrated; a persistently positive level means overconfident."""
    side = p >= 0.5
    conf = np.where(side, p, 1 - p)
    correct = ((y == 1.0) == side).astype(float)
    return (conf - correct) / np.sqrt(np.clip(conf * (1 - conf), 0.05, None))


def change_scan(r: np.ndarray, trim: float = 0.1, threshold: float = 3.5) -> tuple[float, int | None]:
    """Standardised mean-shift scan over all split points (Brownian-bridge style statistic). Unlike a Page CUSUM it has a
    stated false-alarm level: under pure noise the maximum exceeds 3.5 about 1% of the time at n=800 (checked by simulation
    in the tests), whereas a k=0.5,h=5 CUSUM raised false alarms on ~90% of honest 800-forecast histories."""
    n = len(r)
    lo = max(int(trim * n), 2)
    if n < 20 or n - lo <= lo:
        return 0.0, None
    cs = np.cumsum(r)
    tau = np.arange(lo, n - lo + 1)
    before = cs[tau - 1] / tau
    after = (cs[-1] - cs[tau - 1]) / (n - tau)
    sd = float(r.std()) or 1.0
    stat = np.abs(after - before) * np.sqrt(tau * (n - tau) / n) / sd
    i = int(np.argmax(stat))
    return float(stat[i]), (int(tau[i]) if stat[i] > threshold else None)


def confidence_drift(p, y, dates: Sequence[Any], window: int = 60, step: int = 30, threshold: float = 3.5) -> DriftReport:
    """Does calibration get worse (or shift sign) over time? Rolling windows in matured order give ECE and signed-gap trends;
    a change-point scan on the standardised confidence residual flags a sustained shift and says when it began."""
    p, y = _check_py(p, y)
    order = sorted(range(len(p)), key=lambda i: (as_date(dates[i]), i))
    p, y = p[order], y[order]
    ds = [as_date(dates[i]) for i in order]
    eces, gaps = [], []
    for s in range(0, max(len(p) - window + 1, 0), step):
        pw, yw = p[s:s + window], y[s:s + window]
        eces.append(pattern_reliability.ece(pw, yw, 5))
        gaps.append(float(pw.mean() - yw.mean()))

    def slope(v):
        v = [x for x in v if math.isfinite(x)]
        if len(v) < 4:
            return float("nan"), float("nan")
        r = sps.linregress(np.arange(len(v)), v)
        return float(r.slope), float(r.pvalue)
    es, ep = slope(eces)
    gs, gp = slope(gaps)
    res = _confidence_residual(p, y) if len(p) else np.array([])
    stat, idx = change_scan(res, threshold=threshold)
    direction = None
    if idx is not None:
        direction = "overconfident" if res[idx:].mean() > res[:idx].mean() else "underconfident"
    return DriftReport(len(eces), tuple(eces), tuple(gaps), es, ep, gs, gp, stat, idx,
                       ds[idx].isoformat() if idx is not None else None, direction)


# ------------------------------------------------------------------------------------------------- records and log

@dataclasses.dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    predicted: float
    outcome: int
    decided_at: str
    matured_at: str
    context: str = ""
    knowledge_id: str = ""

    def validate(self) -> list[str]:
        errs = []
        if not (isinstance(self.predicted, float) and 0.0 <= self.predicted <= 1.0):
            errs.append("predicted outside [0,1]")
        if self.outcome not in (0, 1):
            errs.append("outcome must be 0 or 1")
        if as_date(self.decided_at) > as_date(self.matured_at):
            errs.append("decided_at after matured_at")
        return errs


@dataclasses.dataclass(frozen=True)
class InfluencePolicy:
    """The overconfidence rule. `alpha`: significance of the ECE null test. `min_excess`: smallest confidence-minus-accuracy
    that matters. `floor`: influence never goes below this here (removal belongs to the retirement gate)."""
    min_n: int = 60
    n_bins: int = 10
    alpha: float = 0.05
    min_excess: float = 0.02
    floor: float = 0.2
    drift_penalty: float = 0.85
    n_sim: int = 400
    min_context_n: int = 30
    shrink_k: float = 30.0
    fdr_q: float = 0.10
    context_min_gap: float = 0.03           # a context is flagged only if significant AND its raw gap is practically large

    def validate(self) -> list[str]:
        errs = []
        if self.min_n < 3 * self.n_bins:
            errs.append("min_n < 3 * n_bins makes equal-mass ECE undefined")
        if not 0 <= self.floor < 1:
            errs.append("floor outside [0,1)")
        if not 0 < self.alpha < 1:
            errs.append("alpha outside (0,1)")
        if not 0 < self.drift_penalty <= 1:
            errs.append("drift_penalty outside (0,1]")
        return errs


@dataclasses.dataclass(frozen=True)
class ContextCalibration:
    context: str
    n: int
    gap: float                             # raw signed gap (predicted - realised)
    shrunk_gap: float                      # empirical-Bayes shrinkage toward the global gap
    ece: float
    hl_p: float
    flagged: bool
    direction: str                         # "overconfident" | "underconfident" | "ok"
    edge_ratio: float


@dataclasses.dataclass(frozen=True)
class CalibrationAssessment:
    as_of: str
    n: int
    health: str
    reason: str
    ece: float
    ece_p: float
    mce: float
    brier: float
    log_loss: float
    confidence: float
    accuracy: float
    excess: float
    slope: float
    slope_lo: float
    slope_hi: float
    influence: float
    overconfident: bool
    drift: DriftReport | None
    bins: tuple[Bin, ...]
    contexts: tuple[ContextCalibration, ...]


def influence_factor(excess: float, edge_ratio: float, ece_p: float, slope_hi: float, policy: InfluencePolicy,
                     drift_overconfident: bool = False) -> float:
    """Pure rule (testable on its own). Significant overconfidence -> influence = realised edge / claimed edge, tightened by
    the recalibration slope when its CI excludes 1, times the drift penalty if a change-scan alarm points the same way;
    clipped to [floor, 1]. Not significant, or underconfident -> exactly 1.0."""
    if not (math.isfinite(excess) and math.isfinite(ece_p)):
        return 1.0
    significant = ece_p < policy.alpha and excess > policy.min_excess
    if not significant:
        return 1.0
    f = edge_ratio
    if math.isfinite(slope_hi) and slope_hi < 1.0:
        f = min(f, max(slope_hi, 0.0))
    if drift_overconfident:
        f *= policy.drift_penalty
    return float(min(1.0, max(policy.floor, f)))


class CalibrationMonitor:
    """J08. Feed resolved (confidence, outcome) pairs; ask for an assessment and apply the influence rule to new confidence."""

    def __init__(self, policy: InfluencePolicy | None = None):
        self.policy = policy or InfluencePolicy()
        errs = self.policy.validate()
        if errs:
            raise ValueError("invalid InfluencePolicy: " + "; ".join(errs))
        self._recs: list[CalibrationRecord] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._recs)

    def add(self, predicted: float, outcome: int, decided_at, matured_at, now, context: str = "",
            knowledge_id: str = "") -> CalibrationRecord:
        require_past(matured_at, now, "calibration outcome matured_at")
        rid = stable_hash([round(float(predicted), 10), int(outcome), str(as_date(decided_at)), str(as_date(matured_at)),
                           context, knowledge_id, len(self._recs)], 16)
        rec = CalibrationRecord(rid, float(predicted), int(outcome), as_date(decided_at).isoformat(),
                                as_date(matured_at).isoformat(), context, knowledge_id)
        errs = rec.validate()
        if errs:
            raise ValueError("invalid calibration record: " + "; ".join(errs))
        self._recs.append(rec)
        self._ids.add(rid)
        return rec

    def records(self, now, context: str | None = None, knowledge_id: str | None = None) -> list[CalibrationRecord]:
        cut = as_date(now)
        rs = [r for r in self._recs if as_date(r.matured_at) < cut and (context is None or r.context == context)
              and (knowledge_id is None or r.knowledge_id == knowledge_id)]
        return sorted(rs, key=lambda r: (r.matured_at, r.record_id))

    @staticmethod
    def _arrays(rs: Sequence[CalibrationRecord]):
        return (np.array([r.predicted for r in rs], float), np.array([r.outcome for r in rs], float),
                [r.matured_at for r in rs])

    def context_report(self, now, global_gap: float | None = None, knowledge_id: str | None = None) -> tuple[ContextCalibration, ...]:
        pol = self.policy
        rs = self.records(now, knowledge_id=knowledge_id)
        if not rs:
            return ()
        p, y, _ = self._arrays(rs)
        g0 = float(p.mean() - y.mean()) if global_gap is None else global_gap
        ctxs = sorted({r.context for r in rs if r.context})
        rows, pvals = [], []
        for c in ctxs:
            sub = [r for r in rs if r.context == c]
            if len(sub) < pol.min_context_n:
                continue
            pc, yc, _ = self._arrays(sub)
            bins = reliability_diagram(pc, yc, 5, "quantile")
            _, hp, _ = hosmer_lemeshow(bins)
            oc = overconfidence(pc, yc)
            gap = float(pc.mean() - yc.mean())
            k = pol.shrink_k
            shrunk = len(sub) / (len(sub) + k) * gap + k / (len(sub) + k) * g0
            rows.append((c, len(sub), gap, shrunk, ece(bins), hp, oc))
            pvals.append(1.0 if not math.isfinite(hp) else hp)
        keep = benjamini_hochberg(pvals, pol.fdr_q)
        out = []
        for (c, n, gap, shrunk, e, hp, oc), fl in zip(rows, keep):
            fl = bool(fl and abs(gap) >= pol.context_min_gap)
            direction = "ok" if not fl else ("overconfident" if oc["excess"] > 0 else "underconfident")
            out.append(ContextCalibration(c, n, gap, shrunk, e, hp, bool(fl), direction, oc["edge_ratio"]))
        return tuple(out)

    def assess(self, now, seed: int, knowledge_id: str | None = None) -> CalibrationAssessment:
        pol = self.policy
        rs = self.records(now, knowledge_id=knowledge_id)
        n = len(rs)
        nan = float("nan")
        if n < pol.min_n:
            return CalibrationAssessment(as_date(now).isoformat(), n, Health.INSUFFICIENT_EVIDENCE.value,
                                         f"only {n} resolved forecasts (need {pol.min_n})", nan, nan, nan, nan, nan, nan, nan,
                                         nan, nan, nan, nan, 1.0, False, None, (), ())
        p, y, ds = self._arrays(rs)
        bins = reliability_diagram(p, y, pol.n_bins, "quantile")
        e = ece_equal_mass(p, y, pol.n_bins)
        ep = ece_null_pvalue(p, pol.n_bins, e, np.random.default_rng(seed), pol.n_sim)
        oc = overconfidence(p, y)
        ps = platt_slope(p, y)
        drift = confidence_drift(p, y, ds, window=max(30, n // 4), step=max(10, n // 12)) if n >= 90 else None
        drift_over = bool(drift and drift.shift_direction == "overconfident")
        infl = influence_factor(oc["excess"], oc["edge_ratio"], ep, ps["b_hi"], pol, drift_over)
        over = infl < 1.0
        if over:
            health, why = Health.DEGRADING, (f"overconfident: claims {oc['confidence']:.3f}, achieves {oc['accuracy']:.3f} "
                                             f"(ECE {e:.3f}, null p={ep:.3f}); influence x{infl:.2f}")
        elif drift and drift.shift_direction:
            health, why = Health.UNSTABLE, f"calibration drift ({drift.shift_direction}) beginning about {drift.change_date}"
        elif ep < pol.alpha:
            health, why = Health.UNSTABLE, f"miscalibrated but not overconfident (ECE {e:.3f}, null p={ep:.3f})"
        else:
            health, why = Health.HEALTHY, f"ECE {e:.3f} within sampling noise (null p={ep:.3f})"
        ctx = self.context_report(now, float(p.mean() - y.mean()), knowledge_id)
        return CalibrationAssessment(as_date(now).isoformat(), n, health.value, why, e, ep, mce(bins), brier(p, y),
                                     log_loss(p, y), oc["confidence"], oc["accuracy"], oc["excess"], ps["b"], ps["b_lo"],
                                     ps["b_hi"], infl, over, drift, tuple(bins), ctx)

    def apply(self, confidence: float, now, seed: int, context: str | None = None,
              knowledge_id: str | None = None) -> tuple[float, float]:
        """Reduce a fresh confidence toward 0.5 by the influence factor (the more overconfident the record, the less the
        stated confidence is allowed to move the decision). Returns (adjusted confidence, factor used). The context-level
        factor applies when that context is flagged overconfident with enough data, and the smaller of the two wins."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence outside [0,1]")
        a = self.assess(now, seed, knowledge_id)
        f = a.influence
        if context:
            for c in a.contexts:
                if c.context == context and c.flagged and c.direction == "overconfident":
                    f = min(f, max(self.policy.floor, min(1.0, c.edge_ratio)))
        return 0.5 + f * (confidence - 0.5), f

    def per_knowledge(self, now, seed: int, min_n: int | None = None) -> dict[str, CalibrationAssessment]:
        """Assess each knowledge id separately (a well-calibrated item must not be punished for a badly calibrated neighbour)."""
        need = self.policy.min_n if min_n is None else min_n
        ids = sorted({r.knowledge_id for r in self.records(now) if r.knowledge_id})
        return {k: self.assess(now, seed, k) for k in ids if len(self.records(now, knowledge_id=k)) >= need}

    def dump(self, path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(canonical_json(dataclasses.asdict(r)) for r in self._recs) + ("\n" if self._recs else ""),
                     encoding="utf-8")
        return len(self._recs)

    @classmethod
    def load(cls, path, policy: InfluencePolicy | None = None) -> "CalibrationMonitor":
        m = cls(policy)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = CalibrationRecord(**json.loads(line))
                    if r.validate():
                        raise FirewallBreach(f"stored calibration record {r.record_id} invalid: {r.validate()}")
                    m._recs.append(r)
                    m._ids.add(r.record_id)
        return m


def diagram_text(bins: Sequence[Bin], width: int = 30) -> str:
    """Plain-text reliability diagram for reports: predicted vs observed, with the Wilson interval."""
    lines = ["  predicted   observed  [95% CI]        n   gap"]
    for b in bins:
        bar = "#" * int(round(b.freq * width))
        lines.append(f"  {b.mean_p:8.3f}  {b.freq:8.3f}  [{b.ci_lo:.2f},{b.ci_hi:.2f}] {b.n:5d}  {b.gap:+.3f}  {bar}")
    return "\n".join(lines)


def report(a: CalibrationAssessment) -> str:
    lines = [f"CALIBRATION as of {a.as_of} - IMPLEMENTED - NOT VALIDATED", f"health: {a.health} - {a.reason}", f"n={a.n}"]
    if a.n and math.isfinite(a.ece):
        lines += [f"ECE {a.ece:.4f} (null p {a.ece_p:.3f}), MCE {a.mce:.3f}, Brier {a.brier:.4f}, log loss {a.log_loss:.4f}",
                  f"confidence {a.confidence:.3f} vs accuracy {a.accuracy:.3f}; recalibration slope {a.slope:.2f} "
                  f"[{a.slope_lo:.2f}, {a.slope_hi:.2f}]; influence factor {a.influence:.2f}", diagram_text(a.bins)]
        if a.drift:
            lines.append(f"drift: ECE slope {a.drift.ece_slope:+.4f}/window (p {a.drift.ece_slope_p:.3f}); "
                         f"change-scan {a.drift.shift_direction or 'quiet'} (stat {a.drift.scan_stat:.2f})")
        for c in a.contexts:
            if c.flagged:
                lines.append(f"  context {c.context}: {c.direction}, gap {c.gap:+.3f} (shrunk {c.shrunk_gap:+.3f}), n={c.n}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- diagnostics and comparison

def bootstrap_ece_ci(p, y, rng: np.random.Generator, n_boot: int = 300, n_bins: int = 10, level: float = 0.90) -> tuple[float, float, float]:
    """(point, lo, hi) equal-mass ECE with a percentile bootstrap interval. The bootstrap is biased upward on small samples
    (ECE is a positive statistic), which is why influence decisions use the null-simulation p-value, not this interval."""
    p, y = _check_py(p, y)
    point = pattern_reliability.ece(p, y, n_bins)
    if not math.isfinite(point):
        return point, float("nan"), float("nan")
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(p), len(p))
        vals[i] = pattern_reliability.ece(p[idx], y[idx], n_bins)
    a = (1 - level) / 2
    return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def minimum_detectable_gap(n: int, p_mean: float = 0.7, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest average confidence-minus-accuracy gap a sample of n forecasts could reveal at the given power. Used to
    say 'insufficient evidence' with a number: below this gap the monitor is blind, so silence is not reassurance."""
    if n < 2:
        return float("inf")
    sd = math.sqrt(max(p_mean * (1 - p_mean), 1e-9))
    return float((sps.norm.ppf(1 - alpha) + sps.norm.ppf(power)) * sd / math.sqrt(n))


def sharpness(p) -> dict[str, float]:
    """How decisive the forecasts are, independent of whether they are right: mean |p-0.5| and mean entropy in bits. Calibration
    is cheap for a forecaster that always says 0.5; sharpness shows whether it earned its calibration by committing."""
    p = np.asarray(p, float)
    if len(p) == 0:
        return {"mean_abs_dev": float("nan"), "entropy_bits": float("nan")}
    q = np.clip(p, 1e-9, 1 - 1e-9)
    h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return {"mean_abs_dev": float(np.abs(p - 0.5).mean()), "entropy_bits": float(h.mean())}


def discrimination(p, y) -> float:
    """AUC (engine.pattern_reliability.auc): can the forecasts rank successes above failures at all? A calibrated map cannot
    add discrimination, so influence is only worth restoring when this is above 0.5."""
    p, y = _check_py(p, y)
    return pattern_reliability.auc(p, y)


def confidence_bands(p, y, edges: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> list[dict[str, float]]:
    """Accuracy of the predicted SIDE by stated-confidence band: what happens when the system says '80% sure'? Rows: band,
    n, mean stated confidence, hit rate, and the hit rate's Wilson interval."""
    p, y = _check_py(p, y)
    conf = np.maximum(p, 1 - p)
    hit = ((y == 1.0) == (p >= 0.5)).astype(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        l, h = wilson(float(hit[m].sum()), int(m.sum()))
        rows.append({"lo": float(lo), "hi": float(min(hi, 1.0)), "n": int(m.sum()), "stated": float(conf[m].mean()),
                     "hit_rate": float(hit[m].mean()), "ci_lo": l, "ci_hi": h})
    return rows


def compare_recalibrators(p_fit, y_fit, p_test, y_test) -> tuple[dict[str, dict[str, float]], str]:
    """Fit identity / Platt / isotonic on EARLIER forecasts and score them on LATER ones (the caller supplies the time split;
    a recalibrator judged on its own fitting data always looks good). Lower Brier and log loss are better.
    Returns (scores by method, name of the method with the lowest test Brier)."""
    p_fit, y_fit = _check_py(p_fit, y_fit)
    p_test, y_test = _check_py(p_test, y_test)
    if len(p_fit) < 30 or len(p_test) < 30:
        raise ValueError("need at least 30 forecasts on each side of the split")
    maps = {"identity": np.asarray(p_test, float), "platt": PlattCalibrator().fit(p_fit, y_fit).predict(p_test),
            "isotonic": IsotonicCalibrator().fit(p_fit, y_fit).predict(p_test)}
    out = {}
    for name, q in maps.items():
        q = np.clip(q, 0.0, 1.0)
        out[name] = {"brier": brier(q, y_test), "log_loss": log_loss(q, y_test), "ece": ece(reliability_diagram(q, y_test, 5, "quantile"))}
    return out, min(out, key=lambda k: (out[k]["brier"], k))


def shrink_confidence(conf, factor: float):
    """Apply an influence factor to stated probabilities (vectorised): 0.5 + factor * (p - 0.5). factor 1 leaves them alone,
    0 reduces every forecast to a coin flip. The direction of every call is preserved."""
    if not 0.0 <= factor <= 1.0:
        raise ValueError("factor must be within [0, 1]")
    c = np.asarray(conf, float)
    return 0.5 + factor * (c - 0.5)


class GapTracker:
    """Fast online monitor of the signed confidence gap (an EWMA), for the days between full assessments. It cannot lower
    influence by itself; it says 'run the full assessment now'."""

    def __init__(self, lam: float = 0.05, alarm: float = 0.06, warmup: int = 30):
        if not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        self.lam, self.alarm, self.warmup = lam, alarm, warmup
        self.value = 0.0
        self.n = 0
        self.last_date: dt.date | None = None

    def update(self, predicted: float, outcome: int, matured_at, now) -> float:
        require_past(matured_at, now, "gap tracker outcome")
        d = as_date(matured_at)
        if self.last_date is not None and d < self.last_date:
            raise FirewallBreach("gap tracker updates must arrive in matured order")
        conf = max(predicted, 1 - predicted)
        correct = float((outcome == 1) == (predicted >= 0.5))
        self.value = (1 - self.lam) * self.value + self.lam * (conf - correct)
        self.n += 1
        self.last_date = d
        return self.value

    @property
    def alarmed(self) -> bool:
        return self.n >= self.warmup and abs(self.value) > self.alarm


def era_report(monitor: "CalibrationMonitor", now, seed: int = 0) -> list[dict[str, Any]]:
    """Calibration by calendar year of the outcome (the per-era breakdown): is the problem everywhere or one year?"""
    rs = monitor.records(now)
    years = sorted({as_date(r.matured_at).year for r in rs})
    rows = []
    for yr in years:
        sub = [r for r in rs if as_date(r.matured_at).year == yr]
        p = np.array([r.predicted for r in sub])
        y = np.array([r.outcome for r in sub], float)
        oc = overconfidence(p, y)
        rows.append({"year": yr, "n": len(sub), "brier": brier(p, y), "ece": pattern_reliability.ece(p, y, 5),
                     "confidence": oc["confidence"], "accuracy": oc["accuracy"], "excess": oc["excess"]})
    return rows


# ------------------------------------------------------------------------------------------------- composing influence (B14)

@dataclasses.dataclass(frozen=True)
class InfluenceBreakdown:
    knowledge_id: str
    lifecycle: float                       # retirement ledger: 1 ACTIVE, <1 DEGRADED, 0 DORMANT/RETIRED/unregistered
    calibration: float                     # overconfidence rule
    temporal: float                        # learned time behaviour (decay / conditional class)
    total: float
    limiting: str                          # which factor is the smallest


def combined_influence(kid: str, now, ledger, monitor: "CalibrationMonitor | None" = None, profile=None,
                       context: Mapping[str, Any] | None = None, seed: int = 0,
                       reliability: float | None = None) -> InfluenceBreakdown:
    """Dynamic influence of one knowledge item = lifecycle x calibration x temporal, each in [0, 1]. Multiplying (not
    averaging) means any single reason to distrust the item cannot be diluted by the others; a dormant item is exactly 0
    whatever its calibration says. `ledger` is a retirement.RetirementLedger, `profile` a temporal.TemporalProfile."""
    from .temporal import expected_influence
    life = ledger.influence(kid, now, reliability)
    cal = 1.0
    if monitor is not None and life > 0.0:
        cal = monitor.assess(now, seed, kid).influence
    tmp = 1.0 if profile is None or life == 0.0 else expected_influence(profile, now, context)
    parts = {"lifecycle": life, "calibration": cal, "temporal": tmp}
    return InfluenceBreakdown(kid, life, cal, tmp, float(life * cal * tmp), min(parts, key=lambda k: (parts[k], k)))
