"""Precursor research on mover episodes and the always-on sweep (canon C67; contract C66 sections 4-6, 9, 13, 21-23). IMPLEMENTED - NOT VALIDATED.

C67: find even the smallest patterns that could have been predicted BEFORE a 5-10% (or >10%) move, and before what the mover did next. This module
owns the second half of the machine (episodes.py detects, episode_paths.py labels the future):

  PRECURSORS   a registry of strictly past-only features (volume expansion, compression, prior ranges, gaps, candle structure, prior movers,
               market-relative and sector-relative moves, plus externally supplied event/insider/filing panels that MUST carry a publication
               lag). Each feature is computed on the whole (sessions x names) grid and read at an ANCHOR: 'pre' = the close before the move day,
               'post' = the close of the move day (before the next session's outcome). Every feature is audited on registration by truncating
               the future and comparing (future-invariance): a feature that reads the move day's own close from a pre anchor is refused.
  CONTRASTS    an episode type (lens x band x side) against MATCHED non-episodes (same date, same volatility cohort), and one path class
               against another inside a type (reversals vs spikes, consolidation vs expansion...). The statistic is a stratified difference of
               cross-sectional feature ranks, clustered by calendar quarter (or month), so hundreds of movers on one day count as one day.
  EVIDENCE     sufficient statistics per (type, comparison, anchor, feature, cluster) are merged unit by unit and kept, together with the same
               statistics for label-SHUFFLED copies of every comparison (labels permuted within strata). The rows themselves are dropped after each
               unit (mapping rule 27: only exception rows, streamed year by year).
  EVALUATION   every test ever computed is entered in ONE cumulative multiple-testing ledger (engine.research.discovery.TrialLedger, not a copy).
               A result is a HYPOTHESIS until it (1) has a cumulative q-value under the bar, (2) beats every shuffled-label copy in its family and
               the pooled shuffled null, (3) survives walk-forward (discover on the past, confirm on the next block), and (4) keeps its sign in
               every era. Only then is it a CANDIDATE - a research question, still never a trading rule.
  ALWAYS-ON    `step(state, now, loader)` sweeps unit after unit (year x universe slice x lens), least-covered first, with atomic checkpoints;
               finished work is never redone, an interrupted unit leaves no trace, a wider feature set extends old units instead of repeating them.
  FIREWALL     everything here is MATURED_RESEARCH_STATE. `release` returns MaturedRecord objects whose maturity is strictly before `now` and never
               anything filed under a year that is being replayed in disguise; candidate text carries no date, year or ticker.
Built on, not copied: engine.candles (candle structure), engine.features (_days_since, _event_calendar), engine.research.discovery (TrialLedger),
engine.learning.trader_view (identity screen), engine.learning.checkpoints (atomic write), engine.learning.core (hash, provenance)."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import candles as CANDLES
from engine import features as FEATURES
from engine.learning.checkpoints import _atomic_text
from engine.learning.core import Provenance, current_code_hash
from engine.learning.trader_view import string_reasons
from engine.research import episode_paths as EP
from engine.research import episodes as EPI
from engine.research.core import (FirewallBreach, GateVerdict, MaturedRecord, Namespace, Problem, ResearchQuestion, _StrEnum, as_date,
                                  require_past, stable_hash)
from engine.research.discovery import TrialLedger
from engine.research.episodes import (CoverageBook, EpisodeConfig, EpisodeError, Grid, Unit, UnitRecord, build_grid, era_of, lens_band_side)

NAMESPACE = Namespace.MATURED_RESEARCH
ANCHORS = ("pre", "post")
ANCHOR_OFFSET = {"pre": 1, "post": 0}          # sessions between the move day and the anchor row: pre reads row t-1, post reads row t


class PrecursorLeak(FirewallBreach):
    """A precursor's value at some session changed when the future was removed. It is refused, not repaired."""


class LabError(ValueError):
    """A lab configuration, store or unit that cannot be trusted. Never swallowed."""


# ------------------------------------------------------------------------------------------------------------------ configuration
@dataclasses.dataclass(frozen=True)
class LabConfig:
    """Everything the sweep varies. Two runs with equal configs and equal data give identical evidence (all randomness is seeded from `seed`,
    the unit id and the session date)."""
    n_slices: int = 4
    slice_salt: int = 0
    lenses: tuple[str, ...] = ("c2c", "rng", "o2c", "gap")
    n_perm: int = 6                                    # label-shuffled copies of every comparison
    max_controls: int = 30                             # matched controls sampled per (date, cohort) stratum
    cluster_months: int = 3                            # evidence is clustered by calendar block of this many months
    class_stratum_days: int = 7                        # class-vs-class comparisons are stratified by blocks of this many calendar days
    min_type_rows: int = 20                            # a type with fewer episodes in a unit is skipped for that unit
    pairs: tuple[tuple[str, str], ...] = (("REVERSED_NEXT_DAY", "SPIKED_NEXT_DAY"), ("CONSOLIDATED", "EXPANDED"), ("STOPPED", "EXPANDED"))
    pair_horizons: tuple[int, ...] = (1,)
    seed: int = 0
    era_edges: tuple[int, ...] = EPI.ERA_EDGES

    def validate(self) -> list[str]:
        errs = []
        if self.n_slices < 1 or self.n_perm < 1 or self.max_controls < 1:
            errs.append("n_slices, n_perm and max_controls must be >= 1")
        if 12 % self.cluster_months != 0:
            errs.append("cluster_months must divide 12")
        if not self.lenses or any(l not in {m.value for m in EPI.Measure} for l in self.lenses):
            errs.append("unknown lens")
        bad = [c for p in self.pairs for c in p if c not in EP.CLASS_ORDER]
        if bad:
            errs.append(f"unknown path classes {bad}")
        if any(a == b for a, b in self.pairs):
            errs.append("a comparison needs two different classes")
        if self.class_stratum_days < 1:
            errs.append("class_stratum_days must be >= 1")
        return errs

    def require_valid(self) -> "LabConfig":
        errs = self.validate()
        if errs:
            raise LabError("; ".join(errs))
        return self

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


@dataclasses.dataclass(frozen=True)
class EvidenceRules:
    """The bar between HYPOTHESIS and CANDIDATE. Deliberately strict on process (ledger, shuffles, walk-forward, eras) and modest on effect size:
    the owner asked for even the smallest patterns, so a real 2-point rank shift is welcome and a lucky 20-point one is not."""
    min_n1: int = 40
    min_n0: int = 40
    min_clusters: int = 4
    screen_q: float = 0.25                             # tests under this q are kept as tracked hypotheses
    alpha_q: float = 0.05                              # cumulative q needed for CANDIDATE
    alpha_perm: float = 0.01                           # pooled shuffled-null p needed for CANDIDATE
    min_effect: float = 0.02                           # in cross-sectional rank units (0..1)
    wf_folds: int = 4
    wf_screen_t: float = 1.5
    wf_confirm_t: float = 1.0
    wf_pass_frac: float = 0.6
    wf_min_tested: int = 2
    era_min_clusters: int = 2
    era_min_eras: int = 2
    era_t: float = 0.75

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


# ------------------------------------------------------------------------------------------------------------------ primitives
def _roll(a: np.ndarray, w: int, kind: str, minp: int | None = None) -> np.ndarray:
    r = pd.DataFrame(a).rolling(w, min_periods=minp if minp is not None else max(3, w // 2))
    return getattr(r, kind)().to_numpy()


def shift_down(a: np.ndarray, k: int) -> np.ndarray:
    """Row t of the result is row t-k of `a` (k >= 0); the first k rows are NaN. Past-only by construction."""
    if k < 0:
        raise LabError("shift_down only looks backward")
    out = np.full(a.shape, np.nan)
    if k == 0:
        return a.copy()
    if k < a.shape[0]:
        out[k:] = a[:-k]
    return out


def xs_rank(a: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank within each session (NaN stays NaN). Ranks make features comparable across eras and remove the market level,
    which is constant within a date and cannot separate names."""
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy(np.float32)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1.0), np.nan)


class Ctx:
    """Everything a feature function may read: the grid (whose rows may extend past the anchor - which is exactly what the audit checks), optional
    sector codes and externally supplied panels. Base arrays are memoised."""

    def __init__(self, grid: Grid, sector_codes: np.ndarray | None = None, externals: Mapping[str, Any] | None = None):
        self.g = grid
        self.sector_codes = sector_codes
        self.externals = dict(externals or {})
        self._m: dict[str, np.ndarray] = {}

    def memo(self, key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]

    @property
    def tr(self) -> np.ndarray:
        g = self.g
        return self.memo("tr", lambda: _safe_div(np.maximum(g.H, g.prev_c) - np.minimum(g.L, g.prev_c), g.prev_c))

    @property
    def lv(self) -> np.ndarray:
        g = self.g
        return self.memo("lv", lambda: np.log1p(g.V) if g.V is not None else np.full(g.C.shape, np.nan))

    @property
    def V(self) -> np.ndarray:
        return self.g.V if self.g.V is not None else np.full(self.g.C.shape, np.nan)

    def _candles(self) -> dict[str, np.ndarray]:
        if "candles" not in self._m:
            g = self.g
            frames = {n: pd.DataFrame(getattr(g, k), index=g.dates, columns=g.tickers)
                      for n, k in (("Open", "O"), ("High", "H"), ("Low", "L"), ("Close", "C"))}
            self._m["candles"] = {k: v.to_numpy(np.float64) for k, v in CANDLES.build(frames).items()}
        return self._m["candles"]

    def market(self, key: str) -> np.ndarray:
        """Cross-sectional statistics of the whole grid by session, broadcast over names: mean return, breadth, dispersion."""
        def build():
            c = np.where(self.g.eligible(), self.g.c2c, np.nan)
            ok = np.isfinite(c)
            n = ok.sum(1).astype(float)
            z = np.where(ok, c, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, z.sum(1) / n, np.nan)
                up = np.where(n > 0, (z > 0).sum(1) / n, np.nan)
                var = np.where(n > 1, (z ** 2).sum(1) / n - mean ** 2, np.nan)
            return {"ret": mean, "breadth": up, "disp": np.sqrt(np.maximum(var, 0.0))}
        return self.memo("market", build)[key]


@dataclasses.dataclass(frozen=True)
class PrecursorSpec:
    """A named past-only feature. `lags` are how many extra sessions BEFORE the anchor to read (0 = the anchor row itself); a feature is
    exposed once per lag as `name@lag`. `fn(ctx)` returns a (sessions x names) array whose row t may use data through the close of t only."""
    name: str
    family: str
    fn: Callable[[Ctx], np.ndarray]
    lags: tuple[int, ...] = (0,)
    doc: str = ""

    def columns(self) -> list[str]:
        return [f"{self.name}@{l}" for l in self.lags]

    def fingerprint(self) -> str:
        try:
            code = self.fn.__code__
            blob = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode()
        except AttributeError:
            blob = repr(self.fn).encode()
        return hashlib.sha256(self.name.encode() + blob + repr(self.lags).encode()).hexdigest()[:16]


_BUILTIN: list[PrecursorSpec] = []


def _reg(name: str, family: str, lags: Sequence[int] = (0,)):
    def deco(fn):
        _BUILTIN.append(PrecursorSpec(name, family, fn, tuple(lags), (fn.__doc__ or "").strip()))
        return fn
    return deco


# ------------------------------------------------------------------------------------------------------------------ built-in precursors
@_reg("vol_z", "volume", (0, 1, 2))
def _vol_z(c: Ctx):
    """Log volume against its own 20-session mean and spread: is today's turnover unusual for this name?"""
    lv = c.lv
    return _safe_div(lv - _roll(lv, 20, "mean"), _roll(lv, 20, "std"))


@_reg("vol_ratio_5_60", "volume")
def _vol_ratio_5_60(c: Ctx):
    """Five-session average volume over sixty-session average: sustained volume expansion."""
    return _safe_div(_roll(c.V, 5, "mean", 3), _roll(c.V, 60, "mean", 30))


@_reg("vol_burst", "volume", (0, 1))
def _vol_burst(c: Ctx):
    """Volume today over the previous twenty sessions' mean (today excluded): the raw spike."""
    return _safe_div(c.V, shift_down(_roll(c.V, 20, "mean", 10), 1))


@_reg("vol_max_5", "volume")
def _vol_max_5(c: Ctx):
    """Largest of the last five sessions' volume over the sixty-session mean."""
    return _safe_div(_roll(c.V, 5, "max", 3), _roll(c.V, 60, "mean", 30))


@_reg("dollar_vol_level", "volume")
def _dollar_vol_level(c: Ctx):
    """Log of average dollar volume over twenty sessions: liquidity level (a rank, so a size proxy)."""
    return np.log1p(_roll(c.g.C * c.V, 20, "mean", 10))


@_reg("dollar_vol_trend", "volume")
def _dollar_vol_trend(c: Ctx):
    dv = c.g.C * c.V
    return np.log(_safe_div(_roll(dv, 5, "mean", 3), _roll(dv, 60, "mean", 30)))


@_reg("tr_now", "range", (0, 1, 2))
def _tr_now(c: Ctx):
    """True range (including the opening gap) as a fraction of the previous close."""
    return c.tr


@_reg("tr_z", "range", (0, 1))
def _tr_z(c: Ctx):
    """Today's true range against the mean of the previous twenty sessions' true ranges (today excluded)."""
    return _safe_div(c.tr, shift_down(_roll(c.tr, 20, "mean", 10), 1))


@_reg("tr_ratio_5_20", "compression")
def _tr_ratio_5_20(c: Ctx):
    """Short against medium true-range average: below one is compression."""
    return _safe_div(_roll(c.tr, 5, "mean", 3), _roll(c.tr, 20, "mean", 10))


@_reg("tr_ratio_20_60", "compression")
def _tr_ratio_20_60(c: Ctx):
    return _safe_div(_roll(c.tr, 20, "mean", 10), _roll(c.tr, 60, "mean", 30))


@_reg("narrow_days_5", "compression")
def _narrow_days_5(c: Ctx):
    """How many of the last five sessions had a true range under 60% of the prior twenty-session mean."""
    narrow = (c.tr < 0.6 * shift_down(_roll(c.tr, 20, "mean", 10), 1)).astype(float)
    narrow[~np.isfinite(c.tr)] = np.nan
    return _roll(narrow, 5, "sum", 3)


@_reg("ret_1", "return", (0, 1, 2))
def _ret_1(c: Ctx):
    return c.g.c2c


@_reg("ret_5", "return")
def _ret_5(c: Ctx):
    return _safe_div(c.g.C, shift_down(c.g.C, 5)) - 1.0


@_reg("ret_20", "return")
def _ret_20(c: Ctx):
    return _safe_div(c.g.C, shift_down(c.g.C, 20)) - 1.0


@_reg("ret_60", "return")
def _ret_60(c: Ctx):
    return _safe_div(c.g.C, shift_down(c.g.C, 60)) - 1.0


@_reg("abs_ret_5", "return")
def _abs_ret_5(c: Ctx):
    """Sum of absolute daily returns over five sessions: recent realised movement whatever its sign."""
    return _roll(np.abs(c.g.c2c), 5, "sum", 3)


@_reg("up_share_10", "return")
def _up_share_10(c: Ctx):
    up = np.where(np.isfinite(c.g.c2c), (c.g.c2c > 0).astype(float), np.nan)
    return _roll(up, 10, "mean", 6)


@_reg("gap_now", "gap", (0, 1))
def _gap_now(c: Ctx):
    return c.g.gap


@_reg("gap_abs_5", "gap")
def _gap_abs_5(c: Ctx):
    return _roll(np.abs(c.g.gap), 5, "mean", 3)


@_reg("gap_fill_5", "gap")
def _gap_fill_5(c: Ctx):
    """Share of the last five sessions whose opening gap was filled intraday (candle module's gap_filled)."""
    return _roll(c._candles()["gap_filled"], 5, "mean", 3)


@_reg("dist_hi_60", "structure")
def _dist_hi_60(c: Ctx):
    return _safe_div(c.g.C, _roll(c.g.H, 60, "max", 30)) - 1.0


@_reg("dist_lo_60", "structure")
def _dist_lo_60(c: Ctx):
    return _safe_div(c.g.C, _roll(c.g.L, 60, "min", 30)) - 1.0


@_reg("dist_hi_252", "structure")
def _dist_hi_252(c: Ctx):
    return _safe_div(c.g.C, _roll(c.g.H, 252, "max", 120)) - 1.0


@_reg("price_level", "structure")
def _price_level(c: Ctx):
    return np.log(c.g.C)


@_reg("sigma_20", "volatility")
def _sigma_20(c: Ctx):
    return _roll(c.g.c2c, 20, "std", 12)


@_reg("sigma_ratio_5_60", "volatility")
def _sigma_ratio_5_60(c: Ctx):
    return _safe_div(_roll(c.g.c2c, 5, "std", 4), _roll(c.g.c2c, 60, "std", 40))


@_reg("sigma_ratio_20_252", "volatility")
def _sigma_ratio_20_252(c: Ctx):
    return _safe_div(_roll(c.g.c2c, 20, "std", 12), _roll(c.g.c2c, 252, "std", 120))


@_reg("max_abs_20", "mover_history")
def _max_abs_20(c: Ctx):
    return _roll(np.abs(c.g.c2c), 20, "max", 12)


@_reg("n_movers_20", "mover_history")
def _n_movers_20(c: Ctx):
    """How many of the last twenty sessions were themselves moves of at least the lower band edge."""
    big = np.where(np.isfinite(c.g.c2c), (np.abs(c.g.c2c) >= c.g.cfg.lo).astype(float), np.nan)
    return _roll(big, 20, "sum", 12)


@_reg("days_since_mover", "mover_history")
def _days_since_mover(c: Ctx):
    flag = pd.DataFrame((np.nan_to_num(np.abs(c.g.c2c)) >= c.g.cfg.lo).astype(float))
    return FEATURES._days_since(flag, cap=60).to_numpy(np.float64)


@_reg("last_mover_side", "mover_history")
def _last_mover_side(c: Ctx):
    """Sign of the most recent 5%+ move (forward-filled): did the name last jump or crash?"""
    mv = np.where(np.nan_to_num(np.abs(c.g.c2c)) >= c.g.cfg.lo, np.sign(c.g.c2c), np.nan)
    return pd.DataFrame(mv).ffill().to_numpy()


@_reg("loc_now", "candle", (0, 1))
def _loc_now(c: Ctx):
    """Where the close sat in the day's high-low range."""
    return c.g.loc


@_reg("loc_mean_5", "candle")
def _loc_mean_5(c: Ctx):
    return _roll(c.g.loc, 5, "mean", 3)


for _k in ("body", "upwick", "lowwick", "pos", "streak", "reversal_1d", "inside_day", "outside_day", "engulf", "week_vs_month_pos",
           "day_vs_week_body"):
    _key = _k if _k in ("streak", "reversal_1d", "inside_day", "outside_day", "engulf", "week_vs_month_pos", "day_vs_week_body") else "cd_" + _k
    _BUILTIN.append(PrecursorSpec("candle_" + _k, "candle", (lambda key: (lambda c: c._candles()[key]))(_key), (0,), f"engine.candles {_key}"))


@_reg("resid_ret_1", "relative")
def _resid_ret_1(c: Ctx):
    """Today's return minus the cross-sectional mean return that day: the name's own move once the market is removed."""
    return c.g.c2c - c.market("ret")[:, None]


@_reg("resid_ret_5", "relative")
def _resid_ret_5(c: Ctx):
    m5 = _roll(c.market("ret")[:, None], 5, "sum", 3)
    return _roll(c.g.c2c, 5, "sum", 3) - m5


@_reg("sector_rel_ret_1", "sector")
def _sector_rel_ret_1(c: Ctx):
    """Return relative to the mean of the name's sector that day (NaN without a sector map)."""
    return _sector_relative(c, c.g.c2c)


@_reg("sector_ret_5", "sector")
def _sector_ret_5(c: Ctx):
    """Mean five-session return of the name's sector: is the whole group moving?"""
    if c.sector_codes is None:
        return np.full(c.g.C.shape, np.nan)
    r5 = _safe_div(c.g.C, shift_down(c.g.C, 5)) - 1.0
    return _sector_mean(c, r5)


def _sector_mean(c: Ctx, a: np.ndarray) -> np.ndarray:
    codes = c.sector_codes
    out = np.full(a.shape, np.nan)
    for s in np.unique(codes[codes >= 0]):
        cols = np.flatnonzero(codes == s)
        if len(cols) < 3:
            continue
        blk = a[:, cols]
        n = np.isfinite(blk).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, cols] = np.where(n > 0, np.where(np.isfinite(blk), blk, 0.0).sum(1) / n, np.nan)[:, None]
    return out


def _sector_relative(c: Ctx, a: np.ndarray) -> np.ndarray:
    return a - _sector_mean(c, a) if c.sector_codes is not None else np.full(a.shape, np.nan)


def external_spec(name: str, family: str, getter: Callable[[Ctx], np.ndarray], pub_lag: int, lags: Sequence[int] = (0,)) -> PrecursorSpec:
    """Wrap an externally supplied panel (events, insider trades, filings, an existing pattern's score) so that it is read only `pub_lag`
    sessions after its own date. A negative lag is refused outright; a zero lag means 'known at the same close' and the caller vouches for it."""
    if pub_lag < 0:
        raise PrecursorLeak(f"external precursor {name}: publication lag {pub_lag} would read the future")

    def fn(c: Ctx):
        return shift_down(np.asarray(getter(c), dtype=np.float64), int(pub_lag))
    fn.__doc__ = f"external panel {name} read {pub_lag} sessions after its own date"
    return PrecursorSpec(name, family, fn, tuple(lags), fn.__doc__)


def event_days_spec(name: str, events: pd.DataFrame, kind: str, pub_lag: int = 0, cap: int = 60) -> PrecursorSpec:
    """Sessions since the last public filing of one kind, via engine.features' own calendar (which already places after-hours filings on the
    NEXT session). `events` needs columns kind / accepted (tz-aware timestamp) / ticker."""
    def getter(c: Ctx):
        cal = FEATURES._event_calendar(events, kind, c.g.dates, list(c.g.tickers))
        return FEATURES._days_since(cal, cap=cap).to_numpy(np.float64)
    return external_spec(name, "event", getter, pub_lag)


# ------------------------------------------------------------------------------------------------------------------ audit and registry
@dataclasses.dataclass
class AuditReport:
    name: str
    ok: bool
    cuts: tuple[int, ...]
    compared: int
    max_diff: float
    all_nan: bool
    reasons: list[str]

    def require(self) -> None:
        if not self.ok:
            raise PrecursorLeak(f"precursor {self.name} refused: " + "; ".join(self.reasons))


_AUDIT_CACHE: dict[str, AuditReport] = {}


def _probe_ctx(bars: Mapping[str, pd.DataFrame], cut: int | None) -> Ctx:
    b = {k: (v if cut is None else v.iloc[:cut + 1]) for k, v in bars.items()}
    g = build_grid(b, EpisodeConfig(warm=80))
    return Ctx(g, sector_codes=(np.arange(len(g.tickers)) % 5).astype(int))


def audit_spec(spec: PrecursorSpec, seed: int = 7, n_names: int = 40, n_days: int = 150, cuts: Sequence[int] = (70, 100, 130)) -> AuditReport:
    """Future-invariance test: the panel computed on bars truncated at each cut must equal the panel computed with the real future present, at
    every session up to the cut. A negative shift, a centred window, a full-sample normalisation or any read of a later close changes the tail
    values and is caught. Cached by the function's own fingerprint."""
    key = spec.fingerprint()
    if key in _AUDIT_CACHE:
        return _AUDIT_CACHE[key]
    bars, _ = EPI.synthetic_bars(n_names, n_days, seed=seed)
    reasons: list[str] = []
    compared, worst = 0, 0.0
    try:
        full = np.asarray(spec.fn(_probe_ctx(bars, None)), dtype=np.float64)
    except Exception as e:                                    # a feature that cannot even run is refused, not skipped
        rep = AuditReport(spec.name, False, tuple(cuts), 0, float("nan"), True, [f"raised {type(e).__name__}: {e}"])
        _AUDIT_CACHE[key] = rep
        return rep
    if full.shape != (n_days, n_names):
        reasons.append(f"shape {full.shape} != {(n_days, n_names)}")
    else:
        for cut in cuts:
            part = np.asarray(spec.fn(_probe_ctx(bars, cut)), dtype=np.float64)
            a, b = part[:cut + 1], full[:cut + 1]
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=False)
            compared += a.size
            if not same.all():
                d = np.abs(np.where(same, 0.0, a - b))
                worst = max(worst, float(np.nanmax(np.where(np.isfinite(d), d, np.inf))))
                reasons.append(f"value at or before session {cut} changes when later sessions are removed ({int((~same).sum())} cells)")
    all_nan = bool(full.size and not np.isfinite(full).any())
    rep = AuditReport(spec.name, not reasons, tuple(cuts), compared, worst, all_nan, reasons)
    _AUDIT_CACHE[key] = rep
    return rep


class PrecursorRegistry:
    """The set of features the sweep computes. `register` audits before admitting; a spec that fails the audit never enters."""

    def __init__(self, specs: Iterable[PrecursorSpec] = (), audit: bool = True):
        self.specs: dict[str, PrecursorSpec] = {}
        for s in specs:
            self.register(s, audit=audit)

    def register(self, spec: PrecursorSpec, audit: bool = True) -> AuditReport | None:
        if spec.name in self.specs:
            raise LabError(f"precursor {spec.name} already registered")
        if any(l < 0 for l in spec.lags) or not spec.lags:
            raise LabError(f"precursor {spec.name}: lags must be non-negative and non-empty")
        rep = None
        if audit:
            rep = audit_spec(spec)
            rep.require()
        self.specs[spec.name] = spec
        return rep

    def columns(self) -> list[str]:
        return [c for s in self.specs.values() for c in s.columns()]

    def family_of(self, column: str) -> str:
        return self.specs[column.split("@")[0]].family

    def digest(self) -> str:
        return stable_hash({n: s.fingerprint() for n, s in sorted(self.specs.items())}, 12)

    def extra_warm(self) -> int:
        return 260 + max((max(s.lags) for s in self.specs.values()), default=0)


def default_registry(audit: bool = True) -> PrecursorRegistry:
    return PrecursorRegistry(list(_BUILTIN), audit=audit)


# ==END==
