"""Learning curve and multi-dimensional learning delta (contract C62 sections 48 and 65; checklist L08-L12 support, K-series curve).

STATUS: IMPLEMENTED - NOT VALIDATED (planted-curve unit tests only; no real archive was touched).

Section 48: the graph that matters is EXPERIENCE -> FUTURE IMPROVEMENT, never experience -> amount of stored memory. A LearningCurve
records, per learner version, the experience it had, the knowledge it holds (and how much of it is validated), and the gains it
earned: same-year gain, transfer gain, risk, and the memorisation gap. The analysis is built to tell four curves apart:
    RISING       transfer gain climbs with experience (the only curve that is learning)
    MEMORISING   same-year gain climbs and the memorisation gap grows while transfer does not
    HOARDING     the memory grows and nothing improves (experience -> stored memory, the graph the owner said NOT to read)
    DEGRADING    more experience makes the future worse
plus PLATEAUED (rose, then stopped), FLAT and INSUFFICIENT_POINTS. Regressions (a fall from the best level so far, i.e. forgetting)
and the marginal return of further experience are computed explicitly.

Section 65: learning_delta = post_learning_performance - no_learning_performance, reported across eight dimensions that are never
collapsed. A positive delta in one dimension is not proof that the whole system works; a higher-tier gain cannot buy back
damage to a lower-numbered tier (engine/objective.py); LearningDelta.whole_system_claim_allowed is False unless every dimension
was measured and none got worse and the tier-1 dimensions are all significantly better."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .. import learning_delta as LD
from .core import ValidationLabel, _StrEnum, as_date, stable_hash
from . import transfer_score as TS


class CurveVerdict(_StrEnum):
    RISING = "RISING"
    PLATEAUED = "PLATEAUED"
    FLAT = "FLAT"
    DEGRADING = "DEGRADING"
    MEMORISING = "MEMORISING"
    HOARDING = "HOARDING"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"


@dataclass(frozen=True)
class CurvePoint:
    """One learner version's position on the curve. NaN in a gain/risk field means 'not measured' (never zero)."""
    step: int
    experience_count: int          # independent experiences the learner has processed (weeks/decisions with outcomes, not raw rows)
    knowledge_count: int
    validated_knowledge_count: int
    same_year_gain: float = float("nan")
    transfer_gain: float = float("nan")
    risk: float = float("nan")     # tier-2 risk of the learner's portfolio, higher is better
    memorization_gap: float = float("nan")
    as_of: str = ""                # ISO date the newest experience in it had matured

    def check(self) -> list[str]:
        errs = []
        for f in ("step", "experience_count", "knowledge_count", "validated_knowledge_count"):
            v = getattr(self, f)
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
                errs.append(f"{f}={v!r} must be a non-negative integer")
        if not errs and self.validated_knowledge_count > self.knowledge_count:
            errs.append("validated knowledge exceeds total knowledge")
        for f in ("same_year_gain", "transfer_gain", "risk", "memorization_gap"):
            v = getattr(self, f)
            if isinstance(v, float) and math.isinf(v):
                errs.append(f"{f} is infinite")
        if self.as_of:
            try:
                as_date(self.as_of)
            except ValueError:
                errs.append(f"as_of {self.as_of!r} is not a date")
        return errs


class LearningCurve:
    """Append-only sequence of CurvePoints for one learner lineage. History is never edited (section 49)."""

    def __init__(self, name: str = "learner", points: Sequence[CurvePoint] = ()):
        self.name = name
        self._pts: list[CurvePoint] = []
        for p in points:
            self.add(p)

    def __len__(self):
        return len(self._pts)

    @property
    def points(self) -> tuple:
        return tuple(self._pts)

    def add(self, p: CurvePoint) -> "LearningCurve":
        errs = p.check()
        if self._pts:
            last = self._pts[-1]
            if p.step <= last.step:
                errs.append(f"step {p.step} does not follow step {last.step}")
            if p.experience_count < last.experience_count:
                errs.append(f"experience fell from {last.experience_count} to {p.experience_count}: experience cannot be un-had")
            if p.as_of and last.as_of and as_date(p.as_of) < as_date(last.as_of):
                errs.append("as_of moved backwards")
        if errs:
            raise ValueError("; ".join(errs))
        self._pts.append(p)
        return self

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(p, name) for p in self._pts], float)

    def frame(self):
        import pandas as pd
        return pd.DataFrame([dataclasses.asdict(p) for p in self._pts])

    def fingerprint(self) -> str:
        return stable_hash([dataclasses.asdict(p) for p in self._pts])

    def future_improvement(self) -> tuple[np.ndarray, np.ndarray]:
        """(experience, transfer gain): THE graph. Points with an unmeasured transfer gain are omitted, not zero-filled."""
        x, y = self.column("experience_count"), self.column("transfer_gain")
        ok = np.isfinite(y)
        return x[ok], y[ok]

    def memory_growth(self) -> tuple[np.ndarray, np.ndarray]:
        """(experience, knowledge count): the graph NOT to read as learning; used only to detect hoarding."""
        return self.column("experience_count"), self.column("knowledge_count")


def curve_point_from_transfer(step: int, experience_count: int, report, knowledge_count: int, validated_count: int, *, risk: float = float("nan"),
                              as_of: str = "", forward_axes: Sequence[str] = ("YEAR",)) -> CurvePoint:
    """Build a point from a transfer.TransferReport: same_year_gain from the YEAR axis, transfer_gain = mean cross-context gain over
    `forward_axes` (default the unseen-year axis), memorisation_gap from the YEAR axis's replay-vs-forward gap. Untested -> NaN."""
    from .transfer import Axis
    ax = report.axes
    yr = ax.get(Axis.YEAR)
    tg = [ax[a].cross.mean for a in ax if a.value in forward_axes and ax[a].tested]
    mem = float("nan") if yr is None or yr.memorization is None else yr.memorization.gap
    return CurvePoint(step, experience_count, knowledge_count, validated_count, float(report.same_year_gain),
                      float(np.mean(tg)) if tg else float("nan"), risk, mem, as_of)


# ---------------------------------------------------------------------------------------------------------------
# trend statistics
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Trend:
    n: int
    slope: float                   # Theil-Sen, per 100 experiences
    lo: float
    hi: float
    ols_slope: float
    perm_p: float                  # slope permutation p (order of points shuffled)
    first: float                   # mean of the first `edge` points
    last: float

    @property
    def rising(self) -> bool:
        return self.n >= 4 and self.lo > 0

    @property
    def falling(self) -> bool:
        return self.n >= 4 and self.hi < 0


def theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Median of pairwise slopes (robust to a single collapsed point). NaN with fewer than two distinct x."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx > 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else float("nan")


def trend(x, y, *, seed: int = 0, n_boot: int = 300, level: float = 0.95, edge: int = 3, n_perm: int = 4000) -> Trend:
    """Slope of y against experience x per 100 experiences, with a bootstrap interval over points and a permutation p-value.
    Fewer than 4 points give an honest infinite interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 or np.ptp(x) == 0:
        return Trend(n, float("nan"), float("-inf"), float("inf"), float("nan"), 1.0, float("nan"), float("nan"))
    s = theil_sen(x, y) * 100
    ols = float(np.polyfit(x, y, 1)[0]) * 100 if y.std() > 0 else 0.0
    e = max(1, min(edge, n // 2))
    first, last = float(y[:e].mean()), float(y[-e:].mean())
    if n < 4:
        return Trend(n, s, float("-inf"), float("inf"), ols, 1.0, first, last)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) > 0:
            boots.append(theil_sen(x[idx], y[idx]) * 100)
    q = (1 - level) / 2
    lo, hi = (float(np.quantile(boots, q)), float(np.quantile(boots, 1 - q))) if len(boots) >= 20 else (float("-inf"), float("inf"))
    p = LD.slope_perm_p(y, n_perm=n_perm, seed=LD.derive_seed(seed, "curve") % (2 ** 31))
    return Trend(n, s, lo, hi, ols, p, first, last)


@dataclass(frozen=True)
class Saturation:
    asymptote: float               # fitted level as experience -> infinity, measured from the first point
    tau: float                     # experience scale of the approach
    r2_saturating: float
    r2_linear: float
    preferred: str                 # 'saturating' or 'linear'
    headroom: float                # asymptote - current fitted level (how much more experience could add)
    start: float = 0.0             # fitted level at the first experience (c)
    x0: float = 0.0                # experience at the first point


def saturation_fit(x, y, *, min_points: int = 6) -> Saturation | None:
    """Fit y = c + a (1 - exp(-(x - x0)/tau)) by grid search over tau with least squares for (c, a); compare with a straight line.
    'saturating' is preferred only when it removes at least 30% of the linear fit's squared error. Returns None below min_points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_points or np.ptp(x) == 0 or y.std() == 0:
        return None
    x0, span = x.min(), np.ptp(x)
    sst = float(((y - y.mean()) ** 2).sum())
    lin = np.polyfit(x, y, 1)
    sse_lin = float(((y - np.polyval(lin, x)) ** 2).sum())
    best = None
    for tau in np.geomspace(span / 25, span * 4, 60):
        b = 1 - np.exp(-(x - x0) / tau)
        A = np.column_stack([np.ones_like(x), b])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, coef)
    sse, tau, (c, a) = best
    fitted_now = c + a * (1 - math.exp(-span / tau))
    pref = "saturating" if a > 0 and sse < 0.7 * sse_lin else "linear"
    return Saturation(float(c + a), float(tau), 1 - sse / sst, 1 - sse_lin / sst, pref, float(c + a - fitted_now), float(c), float(x0))


def change_point(y, *, min_seg: int = 3, t_min: float = 3.0):
    """Best single split of a series by the two-sample t statistic of the segment means. Returns (index of first point of the
    second segment, shift, t) or None when no split reaches t_min. Detects a step up (a real lesson landed) or down (forgetting)."""
    y = np.asarray([v for v in y if np.isfinite(v)], float)
    n = len(y)
    if n < 2 * min_seg:
        return None
    best = None
    for s in range(min_seg, n - min_seg + 1):
        a, b = y[:s], y[s:]
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(1, n - 2)) if n > 2 else 0.0
        if sp <= 0:
            continue
        t = (b.mean() - a.mean()) / (sp * math.sqrt(1 / len(a) + 1 / len(b)))
        if best is None or abs(t) > abs(best[2]):
            best = (s, float(b.mean() - a.mean()), float(t))
    return best if best and abs(best[2]) >= t_min else None


def regressions(x, y, *, smooth: int = 3, k_noise: float = 3.0) -> list[dict]:
    """Points where the trailing-mean level fell below the best level so far by more than k_noise x the noise of the series
    (noise = MAD-based sd of first differences / sqrt 2). These are candidate forgetting events, listed with how far they fell."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < smooth + 3:
        return []
    sm = np.convolve(y, np.ones(smooth) / smooth, mode="valid")
    xs = x[smooth - 1:]
    d = np.diff(y)
    noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2) if len(d) else 0.0
    thr = max(k_noise * noise, 1e-9)
    best, out = -np.inf, []
    for i, v in enumerate(sm):
        best = max(best, v)
        if best - v > thr:
            out.append({"experience": float(xs[i]), "level": float(v), "best_so_far": float(best), "drop": float(best - v)})
    return out


def marginal_returns(x, y, *, n_bins: int = 4):
    """Mean gain in each equal-count experience bin and the change between successive bins per 100 experiences: what the
    next unit of experience is worth right now. A shrinking or negative last entry means more experience no longer helps."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 * n_bins:
        return []
    order = np.argsort(x)
    chunks = np.array_split(order, n_bins)
    rows, prev = [], None
    for c in chunks:
        cur = {"x_mid": float(x[c].mean()), "gain": float(y[c].mean()), "n": int(len(c)), "per_100": float("nan")}
        if prev is not None and cur["x_mid"] > prev["x_mid"]:
            cur["per_100"] = (cur["gain"] - prev["gain"]) / (cur["x_mid"] - prev["x_mid"]) * 100
        rows.append(cur)
        prev = cur
    return rows


@dataclass(frozen=True)
class CurveAnalysis:
    verdict: CurveVerdict
    why: str
    transfer: Trend
    same_year: Trend
    memory: Trend
    memorization: Trend
    risk: Trend
    saturation: Saturation | None
    step: tuple | None
    regressions: tuple
    marginal: tuple
    validated_share: float         # share of held knowledge that is validated, at the last point
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def as_record(self) -> dict:
        rec = {"verdict": self.verdict.value, "why": self.why, "label": self.label.value, "validated_share": self.validated_share,
               "transfer": dataclasses.asdict(self.transfer), "same_year": dataclasses.asdict(self.same_year),
               "memory": dataclasses.asdict(self.memory), "memorization": dataclasses.asdict(self.memorization),
               "risk": dataclasses.asdict(self.risk), "saturation": dataclasses.asdict(self.saturation) if self.saturation else None,
               "step": list(self.step) if self.step else None, "regressions": list(self.regressions), "marginal": list(self.marginal)}
        rec["id"] = stable_hash(rec)
        return rec


def analyse_curve(curve: LearningCurve, *, seed: int = 0, n_boot: int = 300, min_points: int = 6) -> CurveAnalysis:
    """Classify the curve. The verdict is about EXPERIENCE -> FUTURE IMPROVEMENT; memory growth is used only to recognise hoarding.
    Order: too few points, degrading, memorising, hoarding, plateaued, rising, flat. Never returns a validated label."""
    x, tg = curve.future_improvement()
    xs, sy = curve.column("experience_count"), curve.column("same_year_gain")
    _, kc = curve.memory_growth()
    tr = trend(x, tg, seed=seed, n_boot=n_boot)
    sm = trend(xs, sy, seed=seed + 1, n_boot=n_boot)
    mem = trend(xs, kc, seed=seed + 2, n_boot=n_boot)
    mg = trend(xs, curve.column("memorization_gap"), seed=seed + 3, n_boot=n_boot)
    rk = trend(xs, curve.column("risk"), seed=seed + 4, n_boot=n_boot)
    sat = saturation_fit(x, tg, min_points=min_points)
    step = change_point(tg)
    reg = tuple(regressions(x, tg))
    marg = tuple(marginal_returns(x, tg))
    last = curve.points[-1] if len(curve) else None
    vshare = (last.validated_knowledge_count / last.knowledge_count) if last and last.knowledge_count else float("nan")

    def out(v, why):
        return CurveAnalysis(v, why, tr, sm, mem, mg, rk, sat, step, reg, marg, vshare)

    if len(x) < min_points:
        return out(CurveVerdict.INSUFFICIENT_POINTS, f"only {len(x)} points with a measured transfer gain (< {min_points})")
    if tr.falling:
        return out(CurveVerdict.DEGRADING, f"transfer gain falls {tr.slope:+.4g} per 100 experiences (CI {tr.lo:+.3g}..{tr.hi:+.3g})")
    if sm.rising and mg.rising and not tr.rising:
        return out(CurveVerdict.MEMORISING, f"same-year gain rises ({sm.slope:+.3g}/100) and the memorisation gap rises ({mg.slope:+.3g}/100) "
                                            f"while transfer does not (CI {tr.lo:+.3g}..{tr.hi:+.3g})")
    if mem.rising and not tr.rising:
        return out(CurveVerdict.HOARDING, f"knowledge count grows {mem.slope:+.3g}/100 experiences and the future does not improve "
                                          f"(transfer CI {tr.lo:+.3g}..{tr.hi:+.3g})")
    if tr.rising:
        plateau = LD.plateau_index(tg, win=min(5, max(2, len(tg) // 4)))
        if sat and sat.preferred == "saturating" and plateau is not None:
            return out(CurveVerdict.PLATEAUED, f"transfer rose then flattened near {sat.asymptote:+.4g} (tau {sat.tau:.0f} experiences, headroom {sat.headroom:+.3g})")
        extra = f"; {len(reg)} regression point(s)" if reg else ""
        return out(CurveVerdict.RISING, f"transfer gain rises {tr.slope:+.4g} per 100 experiences (CI {tr.lo:+.3g}..{tr.hi:+.3g}, perm p {tr.perm_p:.3g}){extra}")
    return out(CurveVerdict.FLAT, f"no significant trend (slope {tr.slope:+.3g}/100, CI {tr.lo:+.3g}..{tr.hi:+.3g})")


def render_curve(curve: LearningCurve, analysis: CurveAnalysis | None = None) -> str:
    """Text table. The headline is experience -> future improvement; the memory column is labelled as a diagnostic."""
    a = analysis or analyse_curve(curve)
    L = [f"LEARNING CURVE {curve.name}  points={len(curve)}  [{a.label.value}]", f"verdict: {a.verdict.value}: {a.why}",
         f"{'step':>4} {'experience':>10} {'FUTURE IMPROVEMENT (transfer)':>30} {'same-year':>10} {'mem-gap':>9} {'risk':>8} | diagnostic: knowledge (validated)"]
    f = lambda v: "     n/a" if not math.isfinite(v) else f"{v:+8.4f}"
    for p in curve.points:
        L.append(f"{p.step:>4} {p.experience_count:>10} {f(p.transfer_gain):>30} {f(p.same_year_gain):>10} {f(p.memorization_gap):>9} {f(p.risk):>8} | "
                 f"{p.knowledge_count} ({p.validated_knowledge_count})")
    if a.step:
        L.append(f"step change at point {a.step[0]}: shift {a.step[1]:+.4g} (t={a.step[2]:+.1f})")
    for r in a.regressions[:5]:
        L.append(f"regression at experience {r['experience']:.0f}: level {r['level']:+.4g} vs best {r['best_so_far']:+.4g}")
    if a.marginal:
        L.append("marginal return per 100 experiences: " + ", ".join("n/a" if not math.isfinite(m["per_100"]) else f"{m['per_100']:+.3g}" for m in a.marginal))
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# section 65: multi-dimensional learning delta
# ---------------------------------------------------------------------------------------------------------------
DIMENSIONS = ("movement_delta", "selection_delta", "direction_delta", "risk_delta", "drawdown_delta", "band_share_delta",
              "transfer_delta", "calibration_delta")
TIER_OF_DIM = {"movement_delta": 1, "selection_delta": 1, "band_share_delta": 1, "risk_delta": 2, "drawdown_delta": 2,
               "direction_delta": 3, "transfer_delta": None, "calibration_delta": None}
SOURCE_METRIC = {"movement_delta": ("mover_hit", 1.0), "selection_delta": ("mean_week", 1.0), "direction_delta": ("dir_hit", 1.0),
                 "risk_delta": ("worst5", 1.0), "drawdown_delta": ("max_dd", 1.0), "band_share_delta": ("in_band", 1.0),
                 "calibration_delta": ("brier", -1.0)}      # every dimension is oriented so that a positive delta is better


class DeltaState(_StrEnum):
    BETTER = "BETTER"
    WORSE = "WORSE"
    NO_CHANGE = "NO_CHANGE"
    INSUFFICIENT = "INSUFFICIENT"
    UNTESTED = "UNTESTED"


@dataclass(frozen=True)
class DeltaValue:
    name: str
    tier: int | None
    delta: float                   # after - before, oriented so that positive is better
    lo: float
    hi: float
    n: int
    p_signflip: float
    state: DeltaState
    source: str

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["state"] = self.state.value
        return d


def _delta_value(name, diffs, source, min_pairs, seed, n_boot) -> DeltaValue:
    v = np.asarray([x for x in diffs if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return DeltaValue(name, TIER_OF_DIM[name], float("nan"), float("nan"), float("nan"), 0, float("nan"), DeltaState.UNTESTED, source)
    bm = TS.cluster_bootstrap_mean(v, None, n_boot=n_boot, seed=seed)
    p = LD.signflip_p(v, seed=seed)
    if len(v) < min_pairs:
        st = DeltaState.INSUFFICIENT
    elif bm.lo > 0:
        st = DeltaState.BETTER
    elif bm.hi < 0:
        st = DeltaState.WORSE
    else:
        st = DeltaState.NO_CHANGE
    return DeltaValue(name, TIER_OF_DIM[name], bm.mean, bm.lo, bm.hi, int(len(v)), p, st, source)


@dataclass(frozen=True)
class LearningDelta:
    values: Mapping[str, DeltaValue]
    n_pairs: int
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def __getitem__(self, name):
        return self.values[name]

    def states(self) -> dict:
        return {n: v.state for n, v in self.values.items()}

    def count(self, state: DeltaState) -> int:
        return sum(v.state == state for v in self.values.values())

    @property
    def whole_system_claim_allowed(self) -> bool:
        """True only if every dimension was measured with enough pairs, none is WORSE, all tier-1 dimensions are BETTER and the
        transfer delta is BETTER. One positive dimension is never enough (section 65)."""
        s = self.states()
        if any(st in (DeltaState.UNTESTED, DeltaState.INSUFFICIENT, DeltaState.WORSE) for st in s.values()):
            return False
        return all(s[n] == DeltaState.BETTER for n in ("movement_delta", "selection_delta", "band_share_delta", "transfer_delta"))

    def tier_conflicts(self) -> list[str]:
        """Statements where a lower-priority dimension improved while a higher-priority one got worse (it cannot buy it back)."""
        out = []
        for hi in self.values.values():
            if hi.state != DeltaState.WORSE or hi.tier is None:
                continue
            for lo in self.values.values():
                if lo.state == DeltaState.BETTER and (lo.tier is None or lo.tier > hi.tier):
                    out.append(f"{lo.name} improved but {hi.name} (tier {hi.tier}) got worse: the gain cannot buy the damage back")
        return out

    def decided_by(self) -> str:
        """The lexicographic view: the first tier with a significant change decides direction. 'tier N: names' or 'undecided'."""
        for t in (1, 2, 3):
            ch = [v for v in self.values.values() if v.tier == t and v.state in (DeltaState.BETTER, DeltaState.WORSE)]
            if ch:
                worse = [v.name for v in ch if v.state == DeltaState.WORSE]
                return f"tier {t}: " + ("WORSE " + ", ".join(worse) if worse else "BETTER " + ", ".join(v.name for v in ch))
        return "undecided"

    def statement(self) -> str:
        """The only sentence this object will produce about the whole: counts, conflicts, and what is untested. It never says the
        system 'improved'."""
        s = f"{self.count(DeltaState.BETTER)} of {len(self.values)} dimensions better, {self.count(DeltaState.WORSE)} worse, " \
            f"{self.count(DeltaState.NO_CHANGE)} unchanged, {self.count(DeltaState.UNTESTED) + self.count(DeltaState.INSUFFICIENT)} untested/insufficient. "
        s += f"Decided by {self.decided_by()}. "
        for c in self.tier_conflicts():
            s += c + ". "
        s += "A gain in one dimension is not proof that the learning system works." if not self.whole_system_claim_allowed else \
            "All dimensions measured, none worse, tier-1 and transfer better (still not validated without the scorecard controls)."
        return s

    def as_record(self) -> dict:
        rec = {"n_pairs": self.n_pairs, "label": self.label.value, "values": {n: v.as_dict() for n, v in self.values.items()},
               "whole_system_claim_allowed": self.whole_system_claim_allowed, "decided_by": self.decided_by(), "statement": self.statement()}
        rec["id"] = stable_hash(rec)
        return rec


def compute_learning_delta(pairs: Sequence[Mapping], *, transfer_deltas: Sequence[float] | None = None, min_pairs: int = 3, seed: int = 0,
                           n_boot: int = 1000) -> LearningDelta:
    """learning_delta = post_learning_performance - no_learning_performance, per dimension, over windows.
    `pairs`: one mapping per window {'before': metrics, 'after': metrics} (metrics as from learning_delta.run_metrics plus an
    optional 'brier'). Each dimension reads one metric (SOURCE_METRIC) and is oriented so positive is better: drawdown and risk
    are negative numbers (less negative is better), Brier is lower-is-better (sign flipped). A window missing the metric on either
    side is skipped for that dimension only. `transfer_deltas` are already per-window deltas (learner with the lesson minus
    without, on windows the lesson never saw)."""
    vals = {}
    for i, name in enumerate(DIMENSIONS):
        sd = LD.derive_seed(seed, name) % (2 ** 31)
        if name == "transfer_delta":
            vals[name] = _delta_value(name, list(transfer_deltas or []), "supplied transfer deltas", min_pairs, sd, n_boot)
            continue
        metric, sign = SOURCE_METRIC[name]
        diffs = []
        for pr in pairs:
            b, a = pr.get("before", {}), pr.get("after", {})
            if metric in b and metric in a and b[metric] is not None and a[metric] is not None:
                d = float(a[metric]) - float(b[metric])
                diffs.append(sign * d if np.isfinite(d) else None)
        vals[name] = _delta_value(name, diffs, metric, min_pairs, sd, n_boot)
    return LearningDelta(vals, len(pairs))


def delta_by_group(pairs: Sequence[Mapping], groups: Sequence[str], **kw) -> dict:
    """compute_learning_delta separately per group label (era, year, regime), in sorted label order."""
    if len(pairs) != len(groups):
        raise ValueError("one group label per pair")
    return {g: compute_learning_delta([p for p, l in zip(pairs, groups) if l == g], **kw) for g in sorted(set(groups), key=str)}


# ---------------------------------------------------------------------------------------------------------------
# time safety, persistence, comparison against a control, extrapolation
# ---------------------------------------------------------------------------------------------------------------
def known_at(curve: LearningCurve, now, *, drop: bool = False) -> LearningCurve:
    """The part of the curve that could exist at `now`. Points whose newest experience matured at/after `now` are a future leak:
    by default that raises FirewallBreach (fail closed); drop=True returns the earlier points explicitly instead. Points with no
    as_of date cannot be placed in time and are refused the same way."""
    from .core import FirewallBreach
    keep, bad = [], []
    for p in curve.points:
        if not p.as_of or as_date(p.as_of) >= as_date(now):
            bad.append(p)
        else:
            keep.append(p)
    if bad and not drop:
        raise FirewallBreach(f"{len(bad)} curve point(s) undated or dated at/after now={as_date(now)} (first step {bad[0].step})")
    return LearningCurve(curve.name, keep)


def curve_to_records(curve: LearningCurve) -> list[dict]:
    return [dataclasses.asdict(p) for p in curve.points]


def curve_from_records(name: str, records: Sequence[Mapping]) -> LearningCurve:
    """Rebuild a curve from stored records, re-running every validation (a corrupted or reordered history is refused)."""
    return LearningCurve(name, [CurvePoint(**{k: r[k] for k in (f.name for f in dataclasses.fields(CurvePoint)) if k in r}) for r in records])


def forgetting_score(curve: LearningCurve) -> dict:
    """How much of the best transfer gain was later lost: the largest fall from a running peak, as a share of that peak, and where.
    0 means the curve never gave anything back. NaN when nothing positive was ever reached."""
    x, y = curve.future_improvement()
    if len(y) == 0:
        return {"score": float("nan"), "at_experience": None, "peak": float("nan")}
    peak, worst, at, pk = -np.inf, 0.0, None, float("nan")
    for xv, yv in zip(x, y):
        if yv > peak:
            peak = yv
        if peak > 0 and (peak - yv) / peak > worst:
            worst, at, pk = (peak - yv) / peak, float(xv), float(peak)
    return {"score": float(worst) if peak > 0 else float("nan"), "at_experience": at, "peak": pk if at is not None else float(peak)}


@dataclass(frozen=True)
class CurveComparison:
    grid: tuple                    # experience values compared
    diff: tuple                    # test - control transfer gain on that grid
    trend: Trend                   # trend of the difference against experience
    separates: bool                # the learner's curve climbs away from the control's
    why: str


def compare_curves(test: LearningCurve, control: LearningCurve, *, n_grid: int = 12, seed: int = 0, n_boot: int = 300) -> CurveComparison:
    """Learner minus control (no learning, or the random learner) on a common experience grid (linear interpolation inside the range
    both cover; no extrapolation). The learner has learned something only if the DIFFERENCE climbs; a learner and a control that
    both rise together (a drifting market, an easier era) leave a flat difference."""
    xt, yt = test.future_improvement()
    xc, yc = control.future_improvement()
    if len(xt) < 3 or len(xc) < 3:
        return CurveComparison((), (), trend([], []), False, "fewer than 3 measured points on one of the curves")
    lo, hi = max(xt.min(), xc.min()), min(xt.max(), xc.max())
    if hi <= lo:
        return CurveComparison((), (), trend([], []), False, "the curves cover disjoint experience ranges")
    grid = np.linspace(lo, hi, n_grid)
    diff = np.interp(grid, xt, yt) - np.interp(grid, xc, yc)
    tr = trend(grid, diff, seed=seed, n_boot=n_boot)
    sep = tr.rising
    return CurveComparison(tuple(float(g) for g in grid), tuple(float(v) for v in diff), tr, sep,
                           f"difference slope {tr.slope:+.4g} per 100 experiences (CI {tr.lo:+.3g}..{tr.hi:+.3g})"
                           + ("" if sep else "; the learner does not climb away from the control"))


def experience_needed(curve: LearningCurve, target_gain: float, *, seed: int = 0) -> dict:
    """How much more experience the curve says it would take to reach `target_gain` of transfer gain. Uses the saturating fit when
    it is preferred (and says 'unreachable' if its asymptote is below the target), otherwise the Theil-Sen line (refusing to
    extrapolate a non-positive slope). The answer is an extrapolation and is labelled as one; None means no honest answer."""
    x, y = curve.future_improvement()
    if len(x) < 4 or np.ptp(x) == 0:
        return {"method": None, "experience": None, "reachable": None, "note": "fewer than 4 measured points"}
    if y[-3:].mean() >= target_gain if len(y) >= 3 else y[-1] >= target_gain:
        return {"method": "observed", "experience": float(x[-1]), "reachable": True, "note": "already reached in the last points"}
    sat = saturation_fit(x, y)
    if sat and sat.preferred == "saturating":
        if sat.asymptote <= target_gain:
            return {"method": "saturating", "experience": None, "reachable": False, "note": f"fitted asymptote {sat.asymptote:+.4g} is below the target {target_gain:+.4g}"}
        a = sat.asymptote - sat.start
        frac = (target_gain - sat.start) / a
        return {"method": "saturating", "experience": float(sat.x0 - sat.tau * math.log(max(1e-9, 1 - frac))), "reachable": True,
                "note": f"extrapolated from a saturating fit (tau {sat.tau:.0f})"}
    slope = theil_sen(x, y)
    if not slope > 0:
        return {"method": "linear", "experience": None, "reachable": False, "note": f"trend slope {slope:+.3g} is not positive"}
    icpt = float(np.median(y - slope * x))
    return {"method": "linear", "experience": float((target_gain - icpt) / slope), "reachable": True, "note": "linear extrapolation of the Theil-Sen line"}


# ---------------------------------------------------------------------------------------------------------------
# adapters from the existing learning-delta harness records
# ---------------------------------------------------------------------------------------------------------------
def pairs_from_delta_records(recs: Sequence[Mapping]) -> tuple[list[dict], list[float]]:
    """Convert learning_delta.aggregate-style pair records into (before/after pairs, per-window transfer deltas):
    run1 -> run2 is before/after on the same disguised year; transfer_s0 -> transfer_s1 is the lesson's effect on a different year.
    The transfer delta is taken on mean_week, the harness's primary metric."""
    pairs, transfer = [], []
    for r in recs:
        if "run1" in r and "run2" in r:
            pairs.append({"before": r["run1"], "after": r["run2"]})
        if "transfer_s0" in r and "transfer_s1" in r and not r.get("transfer_anachronistic", True):
            a, b = r["transfer_s0"].get(LD.PRIMARY), r["transfer_s1"].get(LD.PRIMARY)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                transfer.append(float(b) - float(a))
    return pairs, transfer


def delta_from_records(recs: Sequence[Mapping], **kw) -> LearningDelta:
    """compute_learning_delta straight from learning_delta harness records; only past-only transfer pairs feed transfer_delta."""
    pairs, transfer = pairs_from_delta_records(recs)
    return compute_learning_delta(pairs, transfer_deltas=transfer, **kw)


def curve_from_play_records(recs: Sequence[Mapping], *, tag: str = "main", name: str = "same-year", metric: str = LD.PRIMARY,
                            experience_per_run: int = 1, knowledge_of=None) -> LearningCurve:
    """LearningCurve from learning_delta.play_curve records (one per repeated run): experience = runs so far x experience_per_run,
    same_year_gain = metric relative to the first run. There is no transfer measurement in a same-year chain, so transfer_gain stays
    NaN and the verdict of such a curve is INSUFFICIENT_POINTS by design: same-year improvement alone is never the learning curve."""
    sel = [r for r in recs if r.get("tag") == tag]
    c = LearningCurve(name)
    first = sel[0][metric] if sel else float("nan")
    for i, r in enumerate(sel):
        k = int(knowledge_of(r)) if knowledge_of else 0
        c.add(CurvePoint(i, (i + 1) * experience_per_run, k, 0, same_year_gain=float(r[metric] - first)))
    return c
