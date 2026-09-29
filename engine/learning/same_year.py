"""Same-year rerun harness (contract C62 section 25 with 29, 65, 66, 87; checklist E01-E07, L01-L05; canon C54, C55, C57, C58).
STATUS: IMPLEMENTED — NOT VALIDATED (planted-world unit tests only; C63: no real-data run was made).

What it does. The same year is played again and again, each time freshly disguised (new codes, a date shift, rows re-ordered,
plus identity-preserving perturbations) while five frozen controls (engine.learning.controls) carry memory forward through the C58
date filter. The learning curve of each control over runs 1..N is measured. The legitimate learner (B) must beat the no-learning
floor (A) without resembling the identity memoriser (C): its memorisation gap (gain with identities kept minus gain with identities
disguised) must be about zero and C's must be large. The random learner (D) must show no curve; the leaky learner (E) must be
caught by the future firewall (LeakGuard) and must be distinguishable from B by skill at run 1. If any of those sanity conditions
fails the harness declares itself VOID rather than report a learning result -- a harness that cannot see memorisation or a leak
cannot vouch for learning.

The RERUN-FINGERPRINT PROBLEM (mapping E02, contradictory). A disguised rerun shows the SAME numbers, so a numeric memory can
recognise the year (section 1.2 forbids identifying a disguised rerun) -- but a pattern is a regularity in those numbers, so any
perturbation that removes the fingerprint also removes some pattern information. This module does not pretend to dissolve that: it
builds the perturbations (sub-universe sampling, per-name scaling inside volatility buckets, week-offset jitter, feature noise,
market-context noise, common-mode return noise), a RECOGNITION PROBE (a classifier asked whether run k's panel is the same year as run
j's, reported as an AUC with a bootstrap interval, at two adversary tiers: tier 1 = exact/serial numeric memory, tier 2 = fuzzy row
matching) and an INFORMATION-COST table (what each perturbation costs the planted patterns). Where tier 2 is still recognisable that is
reported as the honest residual, not hidden.

Reuses: engine.learning.planted_world (reidentify = the disguise; make_world/year_swap = the year and the other years),
engine.learning.learning_curve (LearningCurve, trend), engine.learning.controls (the five frozen controls; which wrap
engine.pattern_memory), engine.learning.core (FirewallBreach, require_past, stable_hash)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import controls as CT
from engine.learning.core import FirewallBreach, ValidationLabel, _StrEnum, current_code_hash, require_past, stable_hash
from engine.learning.learning_curve import CurvePoint, LearningCurve, Trend, trend
from engine.learning.planted_world import CANARY_PREFIX, PlantedWorld, assert_no_canary, reidentify

GUARD_GATE = CT.MemoryGate("guard")
MAX_SHIFT_YEARS = 60
MODES = ("fresh_perturbed", "fresh_plain", "kept")


# ---------------------------------------------------------------------------------------------------------------
# perturbations that break exact-number recognition
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PerturbConfig:
    """Strength of each identity-preserving perturbation; 0 / 1.0 = off. Every draw is seeded by (seed, run, salt)."""
    sub_universe: float = 1.0        # fraction of names kept per run
    vol_sd: float = 0.0              # sd of the log scale applied to each name's excess return (mean-zero inside its volatility bucket)
    week_jitter: int = 0             # up to this many weeks trimmed from each end
    feature_noise: float = 0.0       # sd of iid noise added to every feature (features are ~unit variance)
    market_noise: float = 0.0        # sd of noise added to each market-context column, per week
    common_mode_sd: float = 0.0      # sd of an additive per-week shock shared by every name (cancels in excess returns)
    vol_buckets: int = 3

    def validate(self) -> list[str]:
        errs = []
        if not 0.2 <= self.sub_universe <= 1.0:
            errs.append(f"sub_universe {self.sub_universe} outside [0.2, 1]")
        for f in ("vol_sd", "feature_noise", "market_noise", "common_mode_sd"):
            if getattr(self, f) < 0:
                errs.append(f"{f} is negative")
        if self.week_jitter < 0 or self.vol_buckets < 1:
            errs.append("week_jitter/vol_buckets out of range")
        return errs

    @property
    def is_off(self) -> bool:
        return self == PerturbConfig(vol_buckets=self.vol_buckets)

    @staticmethod
    def standard() -> "PerturbConfig":
        return PerturbConfig(sub_universe=0.7, vol_sd=0.5, week_jitter=3, feature_noise=0.5, market_noise=1.5, common_mode_sd=0.05)

    def scaled(self, s: float) -> "PerturbConfig":
        """Every perturbation at s times its standard strength (s=0 -> off); the frontier sweeps this."""
        if s < 0:
            raise ValueError("strength must be >= 0")
        st = PerturbConfig.standard()
        return PerturbConfig(1.0 - s * (1.0 - st.sub_universe) if s <= 2 else 0.2, st.vol_sd * s, int(round(st.week_jitter * s)),
                             st.feature_noise * s, st.market_noise * s, st.common_mode_sd * s, self.vol_buckets)

    def only(self, name: str) -> "PerturbConfig":
        """This config with just one perturbation switched on (the cost table's per-perturbation rows)."""
        keep = {"sub_universe": ("sub_universe",), "vol_scaling": ("vol_sd",), "week_jitter": ("week_jitter",),
                "feature_noise": ("feature_noise",), "market_noise": ("market_noise",), "common_mode": ("common_mode_sd",)}
        if name not in keep:
            raise KeyError(name)
        base = PerturbConfig(vol_buckets=self.vol_buckets)
        return dataclasses.replace(base, **{f: getattr(self, f) for f in keep[name]})


PERTURBATIONS = ("sub_universe", "vol_scaling", "week_jitter", "feature_noise", "market_noise", "common_mode")


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng([int(p) & 0xFFFFFFFF for p in parts])


@dataclass
class RunPanel:
    """One disguised presentation of the year. `weeks[i] = (disguised date, real date)`; the real date is trusted-side only."""
    X: pd.DataFrame                  # (date, ticker) x features + market context; no canary
    y: pd.Series                     # raw forward return on the same index
    weeks: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        bad = [c for c in self.X.columns if str(c).startswith(CANARY_PREFIX)]
        if bad:
            raise FirewallBreach(f"canary column(s) {bad} in a presentation")
        self._slices = {}
        for d, g in self.X.groupby(level=0, sort=True):
            self._slices[pd.Timestamp(d)] = g.droplevel(0)

    def __len__(self):
        return len(self.weeks)

    def week(self, i: int) -> tuple:
        disg, real = self.weeks[i]
        Xw = self._slices[disg]
        yw = self.y.xs(disg, level=0).reindex(Xw.index)
        return Xw, yw, CT.Moment(disg, i, real)

    def excess(self, i: int) -> pd.Series:
        Xw, yw, _ = self.week(i)
        return yw - float(yw.mean())


def _clean_frames(world: PlantedWorld) -> tuple:
    keep = world.feature_columns() + world.market_columns()
    return world.X[keep].copy(), world.y_raw.copy()


def make_run_panel(world: PlantedWorld, run: int, seed: int, cfg: PerturbConfig | None = None, mode: str = "fresh_perturbed") -> RunPanel:
    """Disguise the world for run `run` (mode fresh_*: new codes, whole-block date shift, shuffled rows) and, for fresh_perturbed, apply
    the identity-preserving perturbations. mode 'kept' returns the year exactly as it is (the memoriser's paradise)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    cfg = cfg or PerturbConfig.standard()
    errs = cfg.validate()
    if errs:
        raise ValueError("; ".join(errs))
    real_dates = list(world.dates)
    if mode == "kept":
        X, y = _clean_frames(world)
        return RunPanel(X, y, [(pd.Timestamp(d), pd.Timestamp(d)) for d in real_dates], {"mode": mode, "run": run})
    shift_years = 1 + int(_rng(seed, 1).permutation(MAX_SHIFT_YEARS)[run % MAX_SHIFT_YEARS])       # distinct across runs (audit_disguises checks)
    re = reidentify(world, seed=int(_rng(seed, run, 2).integers(0, 2 ** 31)), tickers=True, shift_years=shift_years, shuffle_rows=True)
    X, y = _clean_frames(re.world)
    disguised = list(re.world.dates)
    lo, hi = 0, len(disguised)
    meta = {"mode": mode, "run": run, "shift_days": re.date_shift_days, "names": int(X.index.get_level_values(1).nunique())}
    if mode == "fresh_perturbed" and not cfg.is_off:
        X, y, lo, hi, pm = _perturb(X, y, cfg, seed, run)
        meta.update(pm)
    weeks = [(pd.Timestamp(disguised[i]), pd.Timestamp(real_dates[i])) for i in range(lo, hi)]
    return RunPanel(X, y, weeks, meta)


def _perturb(X: pd.DataFrame, y: pd.Series, cfg: PerturbConfig, seed: int, run: int) -> tuple:
    dates = X.index.get_level_values(0)
    tick = X.index.get_level_values(1)
    W = dates.nunique()
    meta = {}
    if cfg.sub_universe < 1.0:
        names = np.array(sorted(set(tick)))
        k = max(int(round(len(names) * cfg.sub_universe)), min(len(names), 12))
        keep = set(_rng(seed, run, 10).choice(names, size=k, replace=False))
        m = np.array([t in keep for t in tick])
        X, y = X[m], y[m]
        dates, tick = X.index.get_level_values(0), X.index.get_level_values(1)
        meta["names"] = k
    lo = int(_rng(seed, run, 11).integers(0, cfg.week_jitter + 1)) if cfg.week_jitter else 0
    hi_cut = int(_rng(seed, run, 12).integers(0, cfg.week_jitter + 1)) if cfg.week_jitter else 0
    if lo + hi_cut >= W - 8:
        lo = hi_cut = 0
    y = y.copy()
    wk_mean = y.groupby(level=0).transform("mean")
    if cfg.vol_sd > 0:
        dev = y - wk_mean
        vol = dev.groupby(level=1).std().fillna(dev.std())
        nb = max(1, min(cfg.vol_buckets, len(vol)))
        edges = np.quantile(vol.to_numpy(), np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.array([])
        bucket = pd.Series(np.searchsorted(edges, vol.to_numpy()), index=vol.index)
        logf = pd.Series(_rng(seed, run, 13).normal(0.0, cfg.vol_sd, len(vol)), index=vol.index)
        logf = logf - logf.groupby(bucket).transform("mean")             # mean-zero inside each volatility bucket
        scaled = dev * np.exp(logf.reindex(tick).to_numpy())
        y = wk_mean + (scaled - scaled.groupby(level=0).transform("mean"))     # the market's weekly return is left exactly as it was
        meta["vol_sd"] = cfg.vol_sd
    if cfg.common_mode_sd > 0:
        shock = pd.Series(_rng(seed, run, 14).normal(0.0, cfg.common_mode_sd, W), index=sorted(set(dates)))
        y = y + shock.reindex(dates).to_numpy()
    X = X.copy()
    feats = [c for c in X.columns if str(c).startswith("f")]
    if cfg.feature_noise > 0 and feats:
        X[feats] = X[feats].to_numpy() + _rng(seed, run, 15).standard_normal((len(X), len(feats))) * cfg.feature_noise
    mcols = [c for c in X.columns if str(c).startswith("m_")]
    if cfg.market_noise > 0 and mcols:
        ud = sorted(set(dates))
        noise = pd.DataFrame(_rng(seed, run, 16).standard_normal((len(ud), len(mcols))) * cfg.market_noise, index=ud, columns=mcols)
        X[mcols] = X[mcols].to_numpy() + noise.reindex(dates).to_numpy()
    if lo or hi_cut:
        ud = sorted(set(dates))
        keep_d = set(ud[lo:len(ud) - hi_cut])
        m = np.array([d in keep_d for d in dates])
        X, y = X[m], y[m]
    return X, y, lo, W - hi_cut, meta


# ---------------------------------------------------------------------------------------------------------------
# the future firewall for controls
# ---------------------------------------------------------------------------------------------------------------
def rank_ic(scores: np.ndarray, excess: np.ndarray) -> float:
    if len(scores) < 6 or np.ptp(scores) == 0 or np.ptp(excess) == 0:
        return 0.0
    a = pd.Series(scores).rank().to_numpy()
    b = pd.Series(excess).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class GuardRecord:
    letter: str
    provenance_breaches: list = field(default_factory=list)
    ic_mean: float = 0.0
    ic_leak: bool = False
    canary: bool = False

    @property
    def flagged(self) -> bool:
        return bool(self.provenance_breaches) or self.ic_leak or self.canary


class LeakGuard:
    """The future-information firewall as the harness applies it to every control at every decision (contract sections 30, 55):
    (1) no canary/future column may reach a control; (2) the newest outcome a control has used must be dated strictly before the real
    decision time (core.require_past); (3) a run's mean rank-IC against realised excess returns above `ic_threshold` is not skill in
    a world where real signal is a few percent of a standard deviation -- it is a leak (or, for C, a memorised answer). The IC test
    catches a leak that lies about its provenance."""

    def __init__(self, ic_threshold: float = 0.25):
        self.ic_threshold = float(ic_threshold)

    def check_frame(self, X: pd.DataFrame) -> None:
        assert_no_canary(X.columns)

    def check_decision(self, control: CT.Control, moment: CT.Moment) -> str | None:
        seen = control.seen_through()
        if seen is None:
            return None
        try:
            require_past(seen, moment.real(GUARD_GATE), f"control {control.letter} evidence")
        except FirewallBreach as e:
            return str(e)
        return None

    def summarise(self, letter: str, breaches: list, ics: np.ndarray) -> GuardRecord:
        m = float(np.mean(ics)) if len(ics) else 0.0
        return GuardRecord(letter, list(breaches), m, bool(m > self.ic_threshold))


# ---------------------------------------------------------------------------------------------------------------
# playing a run
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ControlRun:
    letter: str
    run: int
    gains: np.ndarray              # per week: mean excess return of the top-k picks
    ics: np.ndarray
    hits: np.ndarray               # per week: share of picks that beat the universe
    state_size: int
    guard: GuardRecord
    real_reads: int = 0

    @property
    def gain(self) -> float:
        return float(self.gains.mean()) if len(self.gains) else 0.0

    @property
    def first_third(self) -> float:
        n = max(len(self.gains) // 3, 1)
        return float(self.gains[:n].mean()) if len(self.gains) else 0.0


def pick_top(scores: pd.Series, k: int, tie_seed: tuple) -> np.ndarray:
    """Positions of the top-k scores; exact ties are broken by a seeded random permutation, never by name or row order."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, int)
    tie = _rng(*tie_seed).permutation(n)
    order = np.lexsort((tie, -scores.to_numpy(dtype=float)))
    return order[:min(k, n)]


def play_run(control: CT.Control, panel: RunPanel, run: int, guard: LeakGuard, top_k: int = 8, seed: int = 0) -> ControlRun:
    """One pass of one control through one presentation: at each week it first learns from the outcome that has just matured (last
    week's features and realised return), then decides on this week's features. The final week's outcome is observed after the run."""
    control.begin_run(run)
    n = len(panel)
    gains, ics, hits, breaches = np.zeros(n), np.zeros(n), np.zeros(n), []
    reads0 = 0
    prev = None
    if control.letter == "E":
        control.attach_oracle(lambda m: (panel.excess(m.week), m.real_ts))
    for i in range(n):
        Xw, yw, mom = panel.week(i)
        guard.check_frame(Xw)
        if prev is not None:
            control.observe(prev[0], prev[1], prev[2], mom)
        scores = control.decide(Xw, mom)
        msg = guard.check_decision(control, mom)
        if msg:
            if not control.expects_breach:
                raise FirewallBreach(msg)
            breaches.append(msg)
        ex = (yw - float(yw.mean())).to_numpy()
        pos = pick_top(scores, top_k, (seed, run, i, 0x7C))
        gains[i] = float(ex[pos].mean()) if len(pos) else 0.0
        hits[i] = float((ex[pos] > 0).mean()) if len(pos) else 0.0
        ics[i] = rank_ic(scores.to_numpy(dtype=float), ex)
        reads0 += len(mom.reads)
        prev = (Xw, yw, mom)
    if prev is not None:
        end = CT.Moment(prev[2].disguised + pd.Timedelta(days=CT.HORIZON_DAYS), n, prev[2].real_ts + pd.Timedelta(days=CT.HORIZON_DAYS))
        control.observe(prev[0], prev[1], prev[2], end)
    control.end_run(run)
    return ControlRun(control.letter, run, gains, ics, hits, control.state_size(), guard.summarise(control.letter, breaches, ics), reads0)


# ---------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gap:
    """A mean difference with a bootstrap interval; NaN mean = not measurable."""
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def positive(self) -> bool:
        return self.n >= 4 and self.lo > 0

    @property
    def includes_zero(self) -> bool:
        return self.n >= 4 and self.lo <= 0 <= self.hi


def block_bootstrap_mean(x: np.ndarray, *, seed: int = 0, n_boot: int = 400, block: int = 4, level: float = 0.95) -> Gap:
    """Mean of a weekly series with a moving-block bootstrap interval (weekly gains are not independent draws)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return Gap(float("nan"), float("-inf"), float("inf"), n)
    rng = np.random.default_rng(seed)
    b = max(1, min(block, n))
    starts = np.arange(n - b + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(starts, size=int(math.ceil(n / b)))
        means[i] = np.concatenate([x[s:s + b] for s in picks])[:n].mean()
    q = (1 - level) / 2
    return Gap(float(x.mean()), float(np.quantile(means, q)), float(np.quantile(means, 1 - q)), n)


def late_runs(runs: Sequence[ControlRun], share: float = 0.4) -> list:
    k = max(int(round(len(runs) * share)), 1)
    return list(runs[-k:])


def stacked(runs: Sequence[ControlRun], field_name: str = "gains") -> np.ndarray:
    """(runs x weeks) matrix; runs of different length (week jitter) are cut to the shortest so runs pair by position."""
    n = min(len(getattr(r, field_name)) for r in runs)
    return np.array([getattr(r, field_name)[:n] for r in runs])


def paired_gap(a_runs: Sequence[ControlRun], b_runs: Sequence[ControlRun], *, seed: int = 0) -> Gap:
    """mean(a - b) over matching runs and weeks. Run j of a is paired with run j of b: they played the same presentation."""
    m = min(len(a_runs), len(b_runs))
    if m == 0:
        return Gap(float("nan"), float("-inf"), float("inf"), 0)
    A, B = stacked(a_runs[:m]), stacked(b_runs[:m])
    n = min(A.shape[1], B.shape[1])
    return block_bootstrap_mean((A[:, :n] - B[:, :n]).mean(0), seed=seed)


def slope_ci(G: np.ndarray, *, seed: int = 0, n_boot: int = 400, block: int = 4, level: float = 0.95) -> Gap:
    """Slope of per-run mean gain against run index (per run), with a moving-block bootstrap over WEEKS (the same weeks are resampled
    for every run, so the run-to-run pairing is kept). `G` is runs x weeks. Needs >= 4 runs for an interval."""
    G = np.asarray(G, float)
    R, n = G.shape if G.ndim == 2 else (0, 0)
    x = np.arange(R, dtype=float)

    def slope(m):
        return float(np.polyfit(x, m, 1)[0]) if R >= 2 and m.std() > 0 else 0.0
    if R < 4 or n < 4:
        return Gap(slope(G.mean(1)) if R >= 2 else float("nan"), float("-inf"), float("inf"), R)
    rng = np.random.default_rng(seed)
    b = max(1, min(block, n))
    starts = np.arange(n - b + 1)
    out = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(starts, size=int(math.ceil(n / b)))
        idx = np.concatenate([np.arange(s, s + b) for s in picks])[:n]
        out[i] = slope(G[:, idx].mean(1))
    q = (1 - level) / 2
    return Gap(slope(G.mean(1)), float(np.quantile(out, q)), float(np.quantile(out, 1 - q)), R)


# ---------------------------------------------------------------------------------------------------------------
# recognition probe (can a classifier tell two panels are the same year?)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PanelSignature:
    m: np.ndarray                  # (weeks, market columns), standardised
    ret: np.ndarray                # weekly mean raw return, standardised
    cum: np.ndarray                # sorted per-name cumulative return (standardised)
    row_hashes: frozenset
    feat: np.ndarray               # sampled standardised feature rows
    codes: np.ndarray              # their quintile codes (per-week cross-sections)


def signature(panel: RunPanel, seed: int = 0, max_rows: int = 1200, decimals: int = 4) -> PanelSignature:
    X, y = panel.X, panel.y
    feats = [c for c in X.columns if str(c).startswith("f")]
    mcols = [c for c in X.columns if str(c).startswith("m_")]
    z = lambda a: (a - a.mean(0)) / np.where(a.std(0) > 1e-12, a.std(0), 1.0)
    m = X[mcols].groupby(level=0).first().sort_index().to_numpy(dtype=float) if mcols else np.zeros((len(panel), 1))
    ret = y.groupby(level=0).mean().sort_index().to_numpy(dtype=float)
    cum = np.sort(y.groupby(level=1).sum().to_numpy(dtype=float))
    arr = X[feats].to_numpy(dtype=np.float64)
    hashes = frozenset(hash(r.tobytes()) for r in np.round(arr, decimals))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=min(max_rows, len(arr)), replace=False)
    q = X[feats].groupby(level=0).rank(pct=True).fillna(0.5).to_numpy()
    codes = np.minimum((q * 5).astype(np.int8), 4)[idx]
    return PanelSignature(z(m), z(ret[:, None])[:, 0], z(cum[:, None])[:, 0], hashes, arr[idx], codes)


def max_lagged_corr(a: np.ndarray, b: np.ndarray, min_overlap: float = 0.6) -> float:
    """Best |correlation| between two series over every relative shift that keeps `min_overlap` of the shorter one (the adversary does
    not know the date shift or the trimmed weeks)."""
    a, b = np.atleast_2d(a.T).T if a.ndim == 1 else a, np.atleast_2d(b.T).T if b.ndim == 1 else b
    need = int(math.ceil(min(len(a), len(b)) * min_overlap))
    best = 0.0
    for lag in range(-(len(b) - need), len(a) - need + 1):
        a0, b0 = max(lag, 0), max(-lag, 0)
        n = min(len(a) - a0, len(b) - b0)
        if n < need:
            continue
        for j in range(min(a.shape[1], b.shape[1])):
            u, v = a[a0:a0 + n, j], b[b0:b0 + n, j]
            if u.std() > 1e-12 and v.std() > 1e-12:
                best = max(best, abs(float(np.corrcoef(u, v)[0, 1])))
    return best


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a) - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


TIER1 = ("row_exact", "m_corr", "ret_corr", "cum_ks")
TIER2 = ("nn_dist", "nn_hamming")


def pair_features(sa: PanelSignature, sb: PanelSignature) -> dict:
    """What an adversary can compute from two presentations without names, dates or labels."""
    from scipy.spatial import cKDTree
    ha, hb = sa.row_hashes, sb.row_hashes
    out = {"row_exact": len(ha & hb) / max(min(len(ha), len(hb)), 1), "m_corr": max_lagged_corr(sa.m, sb.m),
           "ret_corr": max_lagged_corr(sa.ret, sb.ret), "cum_ks": -ks_distance(sa.cum, sb.cum)}
    out["nn_dist"] = -float(cKDTree(sb.feat).query(sa.feat[:400], k=1)[0].mean())
    out["nn_hamming"] = -float(cKDTree(sb.codes.astype(float)).query(sa.codes[:400].astype(float), k=1, p=1)[0].mean())
    return out


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted half; 0.5 when a class is empty."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    p, n = scores[labels], scores[~labels]
    if len(p) == 0 or len(n) == 0:
        return 0.5
    ranks = pd.Series(np.concatenate([p, n])).rank().to_numpy()
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


@dataclass(frozen=True)
class ProbeResult:
    tier: str
    auc: float
    lo: float
    hi: float
    n_pos: int
    n_neg: int
    single_best: float             # best single-feature AUC oriented above 0.5 (an optimistic adversary who picks the feature by hindsight)
    features: tuple

    @property
    def consistent_with_chance(self) -> bool:
        return self.n_pos >= 4 and self.lo <= 0.5 <= self.hi


def oof_auc(F: np.ndarray, labels: np.ndarray, seed: int = 0, folds: int = 3) -> np.ndarray:
    """Out-of-fold logistic-regression scores: the adversary is scored on pairs it was not trained on."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(labels))
    k = max(2, min(folds, int(labels.sum()), int((~labels).sum())))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(F, labels):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(F[tr]), labels[tr])
        oof[te] = clf.decision_function(sc.transform(F[te]))
    return oof


def recognition_probe(same: Sequence[RunPanel], others: Sequence[Sequence[RunPanel]], *, seed: int = 0, n_boot: int = 300,
                      max_neg: int = 60) -> dict:
    """AUC of an adversary asked 'are these two presentations the same year?'. Positives = pairs from `same` (i<j); negatives =
    pairs (one panel of `same`, one panel of another year). Tier 1 = numeric fingerprints (exact rows, market series, return series,
    per-name return distribution); tier 2 adds fuzzy row matching. Returns {'tier1': ProbeResult, 'tier2': ProbeResult}."""
    if len(same) < 3 or not others:
        raise ValueError("the probe needs >= 3 presentations of the year and at least one other year")
    rng = np.random.default_rng(seed)
    sig_same = [signature(p, seed + i) for i, p in enumerate(same)]
    sig_oth = [[signature(p, seed + 100 + i) for i, p in enumerate(o)] for o in others]
    pos = [(i, j) for i in range(len(same)) for j in range(i + 1, len(same))]
    neg = [(i, (o, j)) for i in range(len(same)) for o in range(len(others)) for j in range(len(others[o]))]
    if len(neg) > max_neg:
        neg = [neg[k] for k in rng.choice(len(neg), max_neg, replace=False)]
    rows, labels = [], []
    for i, j in pos:
        rows.append(pair_features(sig_same[i], sig_same[j])); labels.append(True)
    for i, (o, j) in neg:
        rows.append(pair_features(sig_same[i], sig_oth[o][j])); labels.append(False)
    labels = np.array(labels)
    out = {}
    for tier, cols in (("tier1", TIER1), ("tier2", TIER1 + TIER2)):
        F = np.array([[r[c] for c in cols] for r in rows], float)
        keep = [k for k in range(F.shape[1]) if np.ptp(F[:, k]) > 1e-12]
        F = F[:, keep] if keep else np.zeros((len(F), 1))
        oof = oof_auc(F, labels, seed)
        auc = auc_score(labels, oof)
        boots = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(labels), len(labels))
            if labels[ix].any() and (~labels[ix]).any():
                boots.append(auc_score(labels[ix], oof[ix]))
        lo, hi = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if len(boots) > 20 else (0.0, 1.0)
        singles = [max(a, 1 - a) for a in (auc_score(labels, F[:, k]) for k in range(F.shape[1]))]
        out[tier] = ProbeResult(tier, auc, lo, hi, int(labels.sum()), int((~labels).sum()), float(max(singles)), tuple(cols[k] for k in keep))
    return out


def presentations(world: PlantedWorld, n: int, seed: int, cfg: PerturbConfig | None, mode: str) -> list:
    return [make_run_panel(world, k, seed, cfg, mode) for k in range(n)]


# ---------------------------------------------------------------------------------------------------------------
# information cost of each perturbation (what it does to the planted patterns)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PerturbationCost:
    name: str
    t_retention: float             # mean signed t of the planted single-key items, perturbed / unperturbed
    detect_rate: float             # share of planted single-key items still detected (|t| >= bar, right sign)
    key_auc: float                 # AUC of |t| separating planted keys from unplanted ones
    gate_agreement: float          # share of weeks where sign(market context) equals the true one (regime items)
    rows_ratio: float              # rows kept / rows in the unperturbed panel

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def planted_single_keys(world: PlantedWorld) -> dict:
    """{key name: true sign} for planted items a single (feature, quintile) ledger can express."""
    out = {}
    for it in world.spec.items:
        if it.kind in ("strong", "weak", "negative") and len(it.conds) == 1 and it.unless is None and it.gate is None and abs(it.effect) > 0:
            (f, q), = it.conds
            out[f"{f} q{q}"] = 1 if it.effect > 0 else -1
    return out


def key_t_stats(panel: RunPanel) -> pd.Series:
    """t of every (feature, quintile) key over the whole presentation (owner-side analysis, not a learner)."""
    feats = [c for c in panel.X.columns if str(c).startswith("f")]
    names = CT.key_names(feats)
    N = np.zeros(len(names)); S = np.zeros(len(names)); SS = np.zeros(len(names))
    for i in range(len(panel)):
        Xw, yw, _ = panel.week(i)
        n, s, ss, _ = CT.week_key_stats(Xw, yw, feats)
        N += n; S += s; SS += ss
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(N > 0, S / N, 0.0)
        var = np.maximum(np.where(N > 0, SS / N - mean ** 2, 0.0), 1e-12)
        t = np.where(N > 1, mean / np.sqrt(var / N), 0.0)
    return pd.Series(t, index=names)


def gate_sign_agreement(base: RunPanel, pert: RunPanel) -> float:
    """Share of surviving weeks where the perturbed market-context sign matches the original's (the price of market noise for regime
    items). Weeks are matched through the real date the harness kept."""
    col = next((c for c in base.X.columns if str(c).startswith("m_")), None)
    if col is None:
        return 1.0
    b = base.X[col].groupby(level=0).first().sort_index()
    p = pert.X[col].groupby(level=0).first().sort_index()
    bmap = {r: v for (_, r), v in zip(base.weeks, b.to_numpy())}
    agree = [np.sign(v) == np.sign(bmap[r]) for (_, r), v in zip(pert.weeks, p.to_numpy()) if r in bmap]
    return float(np.mean(agree)) if agree else 1.0


def information_cost(world: PlantedWorld, cfg: PerturbConfig, *, seeds: Sequence[int] = (0, 1, 2), t_bar: float = 2.5) -> list:
    """One PerturbationCost row per single perturbation and one for all of them together, averaged over `seeds`."""
    planted = planted_single_keys(world)
    if not planted:
        raise ValueError("the world has no single-key planted item to measure recovery on")
    base = make_run_panel(world, 0, 0, cfg, "kept")
    t0 = key_t_stats(base)
    n0 = len(base.X)
    rows = []
    for name in PERTURBATIONS + ("all",):
        c = cfg if name == "all" else cfg.only(name)
        ret, det, auc, gate, ratio = [], [], [], [], []
        for s in seeds:
            pan = make_run_panel(world, 1, s, c, "fresh_perturbed")
            t1 = key_t_stats(pan)
            signed0 = np.array([sg * t0[k] for k, sg in planted.items()])
            signed1 = np.array([sg * t1[k] for k, sg in planted.items()])
            ret.append(float(signed1.mean() / signed0.mean()) if signed0.mean() != 0 else float("nan"))
            det.append(float(np.mean(signed1 >= t_bar)))
            mask = t1.index.isin(list(planted))
            auc.append(auc_score(mask, t1.abs().to_numpy()))
            gate.append(gate_sign_agreement(base, pan))
            ratio.append(len(pan.X) / n0)
        rows.append(PerturbationCost(name, float(np.nanmean(ret)), float(np.mean(det)), float(np.mean(auc)), float(np.mean(gate)), float(np.mean(ratio))))
    return rows


def perturbation_frontier(world: PlantedWorld, others: Sequence[PlantedWorld], levels: Sequence[float] = (0.0, 0.5, 1.0, 1.5), *,
                          n_runs: int = 5, seed: int = 0, base: PerturbConfig | None = None) -> pd.DataFrame:
    """Recognisability against recoverability as perturbation strength grows: one row per level with tier-1/tier-2 AUC and the
    t-retention of the planted keys. The honest way to choose a level is to read this table, not to pick one that looks good."""
    base = base or PerturbConfig.standard()
    rows = []
    for s in levels:
        cfg = base.scaled(s)
        same = presentations(world, n_runs, seed, cfg, "fresh_perturbed" if s > 0 else "fresh_plain")
        oth = [presentations(o, max(n_runs // 2, 3), seed + 50 + k, cfg, "fresh_perturbed" if s > 0 else "fresh_plain") for k, o in enumerate(others)]
        pr = recognition_probe(same, oth, seed=seed)
        cost = information_cost(world, cfg, seeds=(seed, seed + 1))[-1]
        rows.append({"strength": s, "tier1_auc": pr["tier1"].auc, "tier1_lo": pr["tier1"].lo, "tier1_hi": pr["tier1"].hi,
                     "tier2_auc": pr["tier2"].auc, "t_retention": cost.t_retention, "detect_rate": cost.detect_rate,
                     "gate_agreement": cost.gate_agreement})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------------------------------------------
class Verdict(_StrEnum):
    LEARNING = "LEARNING"                 # B beats A, its curve rises, memorisation gap ~ 0, and the harness passed its own checks
    NO_LEARNING = "NO_LEARNING"           # B does not beat A
    MEMORISING = "MEMORISING"             # B beats A but its gain depends on identities being kept
    VOID_HARNESS = "VOID_HARNESS"         # a control did not behave as designed: no learning claim may be made
    INSUFFICIENT_RUNS = "INSUFFICIENT_RUNS"


@dataclass(frozen=True)
class HarnessConfig:
    n_runs: int = 6
    top_k: int = 8
    seed: int = 0
    perturb: PerturbConfig = field(default_factory=PerturbConfig.standard)
    ic_threshold: float = 0.25
    late_share: float = 0.4
    gap_tolerance: float = 0.25           # B's memorisation gap must be under this share of B's own gain
    min_runs: int = 5

    def validate(self) -> list[str]:
        errs = list(self.perturb.validate())
        if self.n_runs < 1 or self.top_k < 1:
            errs.append("n_runs and top_k must be >= 1")
        return errs


@dataclass
class ModeResult:
    mode: str
    runs: dict                              # letter -> [ControlRun]
    metas: list

    def gains(self, letter: str) -> np.ndarray:
        return np.array([r.gain for r in self.runs[letter]])

    def late(self, letter: str, share: float = 0.4) -> list:
        return late_runs(self.runs[letter], share)


@dataclass
class Judgement:
    verdict: Verdict
    reasons: list
    facts: dict
    label: str = ValidationLabel.NOT_VALIDATED.value

    def as_record(self) -> dict:
        return {"verdict": str(self.verdict), "reasons": list(self.reasons), "facts": self.facts, "label": self.label}


class SameYearHarness:
    """Plays one planted year N times per mode with the five frozen controls and judges the result."""

    def __init__(self, world: PlantedWorld, controls: Mapping[str, CT.FrozenControl] | None = None, cfg: HarnessConfig | None = None,
                 registry: CT.ControlRegistry | None = None):
        self.world = world
        self.cfg = cfg or HarnessConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.frozen = dict(controls or CT.standard_controls())
        if sorted(self.frozen) != sorted(CT.LETTERS):
            raise ValueError(f"the harness needs exactly the controls {CT.LETTERS}, got {sorted(self.frozen)}")
        self.registry = registry or CT.ControlRegistry()
        for fc in self.frozen.values():
            self.registry.register(fc)
        self.guard = LeakGuard(self.cfg.ic_threshold)
        self.results: dict = {}

    def run_mode(self, mode: str) -> ModeResult:
        cfg = self.cfg
        for fc in self.frozen.values():
            self.registry.verify(fc)                           # refuse to run on a changed control
        controls = {L: fc.build(cfg.seed) for L, fc in self.frozen.items()}
        runs = {L: [] for L in controls}
        metas = []
        for k in range(cfg.n_runs):
            panel = make_run_panel(self.world, k, cfg.seed, cfg.perturb, mode)
            metas.append(panel.meta)
            for L, c in controls.items():
                runs[L].append(play_run(c, panel, k, self.guard, cfg.top_k, cfg.seed))
        res = ModeResult(mode, runs, metas)
        self.results[mode] = res
        return res

    def run_all(self, modes: Sequence[str] = MODES) -> dict:
        for m in modes:
            self.run_mode(m)
        return self.results

    # -- curves
    def curve(self, mode: str, letter: str) -> LearningCurve:
        c = LearningCurve(f"{letter}:{mode}")
        exp = 0
        for r in self.results[mode].runs[letter]:
            exp += len(r.gains)
            c.add(CurvePoint(r.run + 1, exp, r.state_size, 0, same_year_gain=r.gain))
        return c

    def curve_trend(self, mode: str, letter: str) -> Gap:
        """Slope of the control's per-run gain over runs (per run) with a week-block bootstrap interval; .positive means rising."""
        return slope_ci(stacked(self.results[mode].runs[letter]), seed=self.cfg.seed)

    def library_trend(self, mode: str, letter: str) -> Trend:
        """The same curve through learning_curve.trend (Theil-Sen over points), for the shared report format."""
        cv = self.curve(mode, letter)
        return trend(cv.column("experience_count"), cv.column("same_year_gain"), seed=self.cfg.seed, n_boot=200, n_perm=1000)

    # -- gaps
    def memorisation_gap(self, letter: str) -> Gap:
        """Gain with identities kept minus gain with identities disguised but numbers untouched (late runs, paired by run). Comparing
        against fresh_plain, not fresh_perturbed, isolates identity: the perturbations cost information for everybody."""
        return paired_gap(self.results["kept"].late(letter, self.cfg.late_share), self.results["fresh_plain"].late(letter, self.cfg.late_share),
                          seed=self.cfg.seed)

    def perturbation_price(self, letter: str) -> Gap:
        """What the perturbations cost a control: gain under fresh_plain minus gain under fresh_perturbed (late runs)."""
        return paired_gap(self.results["fresh_plain"].late(letter, self.cfg.late_share), self.results["fresh_perturbed"].late(letter, self.cfg.late_share),
                          seed=self.cfg.seed)

    def identity_gap(self, mode: str = "kept") -> Gap:
        """Gain of the identity memoriser over the legitimate learner (late runs): how much of a gain identities alone can buy."""
        r = self.results[mode]
        return paired_gap(r.late("C", self.cfg.late_share), r.late("B", self.cfg.late_share), seed=self.cfg.seed)

    def beats_baseline(self, mode: str = "fresh_perturbed") -> Gap:
        r = self.results[mode]
        return paired_gap(r.late("B", self.cfg.late_share), r.late("A", self.cfg.late_share), seed=self.cfg.seed)

    def skill_at_run_one(self, mode: str, letter: str) -> float:
        return float(self.results[mode].runs[letter][0].ics.mean())

    # -- judgement
    def judge(self) -> Judgement:
        cfg = self.cfg
        need = set(MODES)
        if not need <= set(self.results):
            raise RuntimeError(f"judge() needs modes {sorted(need)} to have been run")
        fp, kept = self.results["fresh_perturbed"], self.results["kept"]
        if cfg.n_runs < cfg.min_runs:
            return Judgement(Verdict.INSUFFICIENT_RUNS, [f"{cfg.n_runs} runs < {cfg.min_runs}"], {"n_runs": cfg.n_runs})
        void, facts = [], {}
        e_recs = [r.guard for r in fp.runs["E"]]
        facts["E_flagged_share"] = float(np.mean([g.flagged for g in e_recs]))
        facts["E_ic_run1"], facts["B_ic_run1"] = self.skill_at_run_one("fresh_perturbed", "E"), self.skill_at_run_one("fresh_perturbed", "B")
        if facts["E_flagged_share"] < 1.0:
            void.append("the leaky learner E was not caught by the future firewall in every run")
        if facts["E_ic_run1"] <= facts["B_ic_run1"] + 0.1:
            void.append("E is not distinguishable from B by skill at run 1")
        for L in ("A", "B", "C", "D"):
            unexpected = [r.guard for r in fp.runs[L] if r.guard.provenance_breaches]
            if unexpected:
                void.append(f"control {L} breached the future firewall")
        gap_c = self.memorisation_gap("C")
        facts["C_memorisation_gap"] = dataclasses.asdict(gap_c)
        if not gap_c.positive:
            void.append("the harness cannot see the identity memoriser (C's kept-vs-disguised gap is not positive)")
        trend_d = self.curve_trend("fresh_perturbed", "D")
        facts["D_trend"] = dataclasses.asdict(trend_d)
        if trend_d.positive:
            void.append("the random learner D shows a rising curve")
        a_gain = block_bootstrap_mean(stacked(fp.runs["A"]).mean(0), seed=cfg.seed, level=0.99)      # 99%: a pure lottery must not void the harness by chance
        facts["A_gain"] = dataclasses.asdict(a_gain)
        if a_gain.n >= 4 and not a_gain.includes_zero:
            void.append("the no-learning control A has a non-zero gain")
        if void:
            return Judgement(Verdict.VOID_HARNESS, void, facts)
        beat = self.beats_baseline()
        tr_b = self.curve_trend("fresh_perturbed", "B")
        gap_b = self.memorisation_gap("B")
        facts.update({"B_beats_A": dataclasses.asdict(beat), "B_trend": dataclasses.asdict(tr_b), "B_memorisation_gap": dataclasses.asdict(gap_b),
                      "identity_gap_C_minus_B": dataclasses.asdict(self.identity_gap("kept")),
                      "B_perturbation_price": dataclasses.asdict(self.perturbation_price("B"))})
        reasons = []
        if not beat.positive:
            return Judgement(Verdict.NO_LEARNING, ["B does not beat the no-learning floor with a positive interval"], facts)
        tol = cfg.gap_tolerance * max(beat.mean, 1e-9)
        if gap_b.positive and gap_b.mean > tol:
            return Judgement(Verdict.MEMORISING, [f"B's memorisation gap {gap_b.mean:.5f} exceeds {cfg.gap_tolerance:.0%} of its gain"], facts)
        if not tr_b.positive:
            reasons.append("B beats A but its curve over runs is not (yet) rising with a positive interval")
        return Judgement(Verdict.LEARNING, reasons or ["B beats A, rises over runs and does not need identities"], facts)

    # -- report
    def render(self) -> str:
        lines = ["# Same-year rerun harness (IMPLEMENTED — NOT VALIDATED)", ""]
        for mode, res in self.results.items():
            lines.append(f"## mode {mode}")
            lines.append("| control | " + " | ".join(f"run{k + 1}" for k in range(self.cfg.n_runs)) + " | trend/100wk | leak-flagged |")
            lines.append("|---|" + "---|" * (self.cfg.n_runs + 2))
            for L in CT.LETTERS:
                g = res.gains(L)
                tr = self.curve_trend(mode, L) if len(g) >= 4 else None
                lines.append(f"| {L} | " + " | ".join(f"{v:+.4f}" for v in g) + f" | {tr.mean:+.5f} | {sum(r.guard.flagged for r in res.runs[L])}/{len(g)} |"
                             if tr else f"| {L} | " + " | ".join(f"{v:+.4f}" for v in g) + " | n/a | - |")
            lines.append("")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return stable_hash({"code": current_code_hash(), "controls": {L: fc.record.as_dict() for L, fc in self.frozen.items()},
                            "cfg": dataclasses.asdict(self.cfg), "world": self.world.content_hash()})


# ---------------------------------------------------------------------------------------------------------------
# section 65: the multi-dimensional learning delta of the rerun chain
# ---------------------------------------------------------------------------------------------------------------
BAND_PROXY = 0.005      # planted-world stand-in for 'a week inside the target band': a top-k excess gain of at least +0.5%


def run_metrics(run: ControlRun) -> dict:
    """Metrics of one pass in the vocabulary of learning_curve.SOURCE_METRIC (planted-world proxies, not the 7% band)."""
    g = run.gains
    if len(g) == 0:
        return {}
    cum = np.cumsum(g)
    return {"mean_week": float(g.mean()), "in_band": float((g >= BAND_PROXY).mean()), "worst5": float(np.quantile(g, 0.05)),
            "max_dd": float((cum - np.maximum.accumulate(cum)).min()), "pos_share": float((g > 0).mean()),
            "dir_hit": float((run.ics > 0).mean()), "mover_hit": float(run.hits.mean())}


def harness_learning_delta(h: "SameYearHarness", letter: str = "B", mode: str = "fresh_perturbed", *, seed: int = 0, n_boot: int = 400):
    """learning_delta = later runs - run 1 of the same control, on every section-65 dimension (transfer and calibration stay UNTESTED:
    a same-year chain measures neither, and a positive same-year delta alone never proves learning). Pairs share their run-1 'before',
    so they are not independent; the interval is therefore wide by design of the bootstrap, and still only a same-year statement."""
    from engine.learning.learning_curve import compute_learning_delta
    runs = h.results[mode].runs[letter]
    pairs = [{"before": run_metrics(runs[0]), "after": run_metrics(r)} for r in runs[1:]]
    return compute_learning_delta(pairs, seed=seed, n_boot=n_boot, min_pairs=max(2, len(pairs) // 2))


def delta_statements(h: "SameYearHarness", mode: str = "fresh_perturbed") -> dict:
    """One honest sentence per control about its same-year delta across all dimensions."""
    return {L: harness_learning_delta(h, L, mode).statement() for L in CT.LETTERS}


# ---------------------------------------------------------------------------------------------------------------
# the disguised-rerun mechanism audit (E02) and the serialisable record
# ---------------------------------------------------------------------------------------------------------------
def audit_disguises(panels: Sequence[RunPanel]) -> list[str]:
    """Problems that would let a player recognise a rerun BY IDENTITY (the numeric tier is the probe's job): a code reused between
    runs, a date shift repeated, a disguised date equal to a real one, or the row order following code order."""
    problems = []
    seen_codes: dict = {}
    shifts: dict = {}
    for k, p in enumerate(panels):
        codes = set(p.X.index.get_level_values(1))
        for prev, c in seen_codes.items():
            over = codes & c
            if over:
                problems.append(f"runs {prev} and {k} share {len(over)} ticker codes")
        seen_codes[k] = codes
        sh = p.meta.get("shift_days")
        if sh is not None:
            if sh in shifts:
                problems.append(f"runs {shifts[sh]} and {k} use the same date shift ({sh} days)")
            shifts[sh] = k
        same_date = sum(d == r for d, r in p.weeks)
        if same_date:
            problems.append(f"run {k}: {same_date} disguised dates equal their real dates")
        first = p.X.groupby(level=0, sort=True).head(1).index.get_level_values(1)
        if len(first) > 4 and list(first) == sorted(first):
            problems.append(f"run {k}: row order follows code order (a tie-break would leak identity)")
    return problems


def record_of(h: "SameYearHarness") -> dict:
    """JSON-able summary of a finished harness: per mode and control the per-run gains, the guard flags, the verdict and hashes."""
    rec = {"label": ValidationLabel.NOT_VALIDATED.value, "fingerprint": h.fingerprint(), "config": dataclasses.asdict(h.cfg), "modes": {}}
    for mode, res in h.results.items():
        rec["modes"][mode] = {L: {"gains": [round(float(r.gain), 8) for r in runs], "ic": [round(float(r.ics.mean()), 6) for r in runs],
                                  "flagged": [bool(r.guard.flagged) for r in runs], "state_size": [int(r.state_size) for r in runs]}
                              for L, runs in res.runs.items()}
    if set(MODES) <= set(h.results):
        j = h.judge()
        rec["judgement"] = j.as_record()
    rec["id"] = stable_hash(rec)
    return rec
