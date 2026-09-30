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
F29 (C75 1C) changes: a unit now computes evidence for its OWN lens only (it used to compute every lens, which with the default four lenses
added each lens's evidence four times); the sweep WIDENS - once every unit is covered it adds the next built-in precursor family and keeps
going - and it is judged once more when exhausted (the research loop's three-year planted feed used to finish before its first scheduled
evaluation and so never produced a candidate); EvalReport.promoted is what the loop turns into questions; grouped steps read each
(year, slice) once for all lenses; `holdout_check` (untouched later years), `raise_questions` (engine.research.questions.generate) and
`truncation_audit` (the F09 method at episode level) are the out-of-sample, question and point-in-time checks of the real-data run
(scripts/c67_mover_sweep_real.py). Real-data results from that run are SURVIVOR_ONLY.
Built on, not copied: engine.candles (candle structure), engine.features (_days_since, _event_calendar), engine.research.discovery (TrialLedger),
engine.learning.trader_view (identity screen), engine.learning.checkpoints (atomic write), engine.learning.core (hash, provenance)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import candles as CANDLES
from engine import features as FEATURES
from engine.learning.checkpoints import _atomic_text
from engine.learning.core import Health, Provenance, current_code_hash
from engine.learning.trader_view import assert_trader_safe, string_reasons
from engine.research import episode_paths as EP
from engine.research import episodes as EPI
from engine.research.core import (FirewallBreach, Knowability, MaturedRecord, Namespace, Problem, ResearchQuestion, _StrEnum, as_date,
                                  require_past, stable_hash)
from engine.research.discovery import TrialLedger
from engine.research.episodes import (CoverageBook, EpisodeConfig, Grid, Unit, UnitRecord, build_grid, era_of, lens_band_side)

NAMESPACE = Namespace.MATURED_RESEARCH
ANCHORS = ("pre", "post")
ANCHOR_OFFSET = {"pre": 1, "post": 0}          # sessions between the move day and the anchor row: pre reads row t-1, post reads row t


class PrecursorLeak(FirewallBreach):
    """A precursor's value at some session changed when the future was removed. It is refused, not repaired."""


class LabError(ValueError):
    """A lab configuration, store or unit that cannot be trusted. Never swallowed."""


class DefinitionDrift(LabError):
    """A stored sweep was built under different episode / path / lab definitions than the ones supplied now."""


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
        self._m: dict[str, Any] = {}

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
            free = repr([c.cell_contents for c in (self.fn.__closure__ or ())]).encode()     # closures share bytecode: their contents tell them apart
            blob = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode() + free
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
    if codes is None:
        raise ValueError("sector means need sector codes")
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


# ------------------------------------------------------------------------------------------------------------------ row sets
def type_code(band: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Integer id of an episode type within a lens: band (1 or 2) * 10 + side + 1 (side -1/0/+1)."""
    return (band.astype(np.int64) * 10 + side.astype(np.int64) + 1)


def type_label(code: int) -> str:
    band, side = divmod(int(code), 10)
    return f"{EPI.BAND_LABEL.get(band, str(band))}_{'up' if side == 2 else 'dn' if side == 0 else 'flat'}"


def cluster_of(dates, months: int) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return ((d.year.to_numpy() * 12 + d.month.to_numpy() - 1) // months).astype(np.int64)


def cluster_year(cluster: int, months: int) -> int:
    return int(cluster * months // 12)


@dataclasses.dataclass
class RowSet:
    """Episodes of one lens plus their matched controls, as index arrays into the grid. Controls are non-episodes (no band under ANY measure),
    eligible, of the same date and volatility cohort as some episode; at most `max_controls` per stratum, sampled with a seed from (unit, date, cohort)."""
    lens: str
    ti: np.ndarray
    nj: np.ndarray
    is_ep: np.ndarray
    tcode: np.ndarray                       # -1 for controls
    stratum: np.ndarray                     # ti * n_cohorts + cohort, increasing with time
    ep_row: np.ndarray                      # row in the episode frame, -1 for controls

    def __len__(self) -> int:
        return len(self.ti)

    def types(self, min_rows: int = 1) -> list[int]:
        u, n = np.unique(self.tcode[self.is_ep], return_counts=True)
        return [int(a) for a, b in zip(u, n) if b >= min_rows]


def unit_seed(uid: str) -> int:
    return int(stable_hash(uid, 8), 16)


def build_rowset(grid: Grid, eps: pd.DataFrame, lens: str, cfg: LabConfig, useed: int) -> RowSet:
    band, side = lens_band_side(eps, lens)
    keep = (band > 0) & (side != 0) & (eps["cohort"].to_numpy() >= 0)
    idx = np.flatnonzero(keep)
    nco = max(grid.cfg.n_cohorts, 1)
    ti, nj = eps["ti"].to_numpy()[idx], eps["nj"].to_numpy()[idx]
    coh = eps["cohort"].to_numpy()[idx].astype(np.int64)
    tcode = type_code(band[idx], side[idx])
    empty = np.zeros(0, np.int64)
    ctl_t, ctl_n, ctl_s = [], [], []
    pool = grid.eligible() & ~grid.any_band() & (grid.cohort >= 0)
    ordinals = np.array([d.toordinal() for d in grid.dates], dtype=np.int64)
    for key in np.unique(ti * nco + coh):
        t, c = divmod(int(key), nco)
        cand = np.flatnonzero(pool[t] & (grid.cohort[t] == c))
        if len(cand) > cfg.max_controls:
            rng = np.random.default_rng([cfg.seed & 0xFFFFFFFF, useed, int(ordinals[t]), int(c)])
            cand = np.sort(rng.choice(cand, cfg.max_controls, replace=False))
        ctl_t.append(np.full(len(cand), t, np.int64))
        ctl_n.append(cand.astype(np.int64))
        ctl_s.append(np.full(len(cand), key, np.int64))
    ct = np.concatenate(ctl_t) if ctl_t else empty
    cn = np.concatenate(ctl_n) if ctl_n else empty
    cs = np.concatenate(ctl_s) if ctl_s else empty
    strat = np.concatenate([ti * nco + coh, cs])
    order = np.argsort(strat, kind="stable")
    return RowSet(lens, np.concatenate([ti, ct])[order], np.concatenate([nj, cn])[order],
                  np.concatenate([np.ones(len(ti), bool), np.zeros(len(ct), bool)])[order],
                  np.concatenate([tcode, np.full(len(ct), -1, np.int64)])[order], strat[order],
                  np.concatenate([idx, np.full(len(ct), -1, np.int64)])[order])


def ranked_panels(ctx: Ctx, registry: PrecursorRegistry, names: Iterable[str] | None = None) -> tuple[dict[str, np.ndarray], list[str]]:
    """Cross-sectional rank of every requested base feature, plus the names whose panel is entirely missing (e.g. sector features with no map)."""
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for n in (registry.specs if names is None else names):
        spec = registry.specs[n]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            a = np.asarray(spec.fn(ctx), dtype=np.float64)
        if a.shape != ctx.g.C.shape:
            raise LabError(f"precursor {n} returned shape {a.shape}, expected {ctx.g.C.shape}")
        a = np.where(np.isfinite(a), a, np.nan)
        if not np.isfinite(a).any():
            missing.append(n)
            continue
        out[n] = xs_rank(a)
    return out, missing


def gather_features(panels: Mapping[str, np.ndarray], columns: Sequence[str], ti: np.ndarray, nj: np.ndarray, anchor: str) -> np.ndarray:
    """(n rows x n columns) ranks read at the anchor. A row whose anchor precedes the grid, or a column with no panel, is NaN."""
    if anchor not in ANCHOR_OFFSET:
        raise LabError(f"unknown anchor {anchor!r}")
    off = ANCHOR_OFFSET[anchor]
    out = np.full((len(ti), len(columns)), np.nan, dtype=np.float32)
    for j, col in enumerate(columns):
        base, lag = col.split("@")
        R = panels.get(base)
        if R is None:
            continue
        r = ti - off - int(lag)
        ok = r >= 0
        out[ok, j] = R[r[ok], nj[ok]]
    return out


# ------------------------------------------------------------------------------------------------------------------ stratified contrast
class Contrast:
    """Weighted difference of feature ranks between a labelled group and its controls inside strata (a stratum is one date and volatility cohort,
    or one calendar block for class comparisons). For every stratum with both groups present: diff = mean(group) - mean(other), weight
    w = n1 n0 / (n1 + n0). Strata are summed into clusters. Rows are pre-sorted once so a label-shuffled copy costs two reductions."""

    def __init__(self, F: np.ndarray, stratum: np.ndarray, cluster: np.ndarray):
        n = len(stratum)
        self.n = n
        order = np.argsort(stratum, kind="stable")
        self.order = order
        st = stratum[order]
        self.F = F[order]
        cl = cluster[order]
        self.starts = np.flatnonzero(np.r_[True, st[1:] != st[:-1]]) if n else np.zeros(0, np.int64)
        self.block = np.repeat(np.arange(len(self.starts)), np.diff(np.r_[self.starts, n])) if n else np.zeros(0, np.int64)
        self.valid = np.isfinite(self.F).astype(np.float64)
        self.Fz = np.where(np.isfinite(self.F), self.F, 0.0).astype(np.float64)
        self.stot = np.add.reduceat(self.Fz, self.starts, axis=0) if n else np.zeros((0, F.shape[1]))
        self.ntot = np.add.reduceat(self.valid, self.starts, axis=0) if n else np.zeros((0, F.shape[1]))
        bc = cl[self.starts] if n else np.zeros(0, np.int64)
        self.cluster_ids, self.inv = np.unique(bc, return_inverse=True)

    def sorted_labels(self, g: np.ndarray) -> np.ndarray:
        return g[self.order].astype(np.float64)

    def base_for(self, active_sorted: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Per-stratum totals over the rows that take part (all rows, or only the active ones)."""
        if active_sorted is None or self.n == 0:
            return self.stot, self.ntot
        a = active_sorted[:, None]
        return np.add.reduceat(self.Fz * a, self.starts, axis=0), np.add.reduceat(self.valid * a, self.starts, axis=0)

    def shuffled_labels(self, g_sorted: np.ndarray, rng: np.random.Generator, active_sorted: np.ndarray | None = None) -> np.ndarray:
        """Permute labels within strata among the active rows (rows are contiguous per stratum after the sort); inactive rows keep label 0."""
        u = rng.random(self.n)
        if active_sorted is None:
            return g_sorted[np.lexsort((u, self.block))]
        inactive = active_sorted == 0
        order_a = np.lexsort((u, inactive, self.block))
        base_a = np.lexsort((np.arange(self.n), inactive, self.block))
        out = np.zeros(self.n)
        out[order_a] = g_sorted[base_a]
        return out

    def sums(self, g_sorted: np.ndarray, base: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """(clusters x features x 4) of [sum w*diff, sum w, sum n1, sum n0]."""
        f = self.F.shape[1]
        out = np.zeros((len(self.cluster_ids), f, 4))
        if self.n == 0:
            return out
        stot, ntot = base if base is not None else (self.stot, self.ntot)
        g = g_sorted[:, None]
        s1 = np.add.reduceat(self.Fz * g, self.starts, axis=0)
        n1 = np.add.reduceat(self.valid * g, self.starts, axis=0)
        s0, n0 = stot - s1, ntot - n1
        both = (n1 > 0) & (n0 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = np.where(both, s1 / np.where(n1 > 0, n1, 1.0) - s0 / np.where(n0 > 0, n0, 1.0), 0.0)
            w = np.where(both, n1 * n0 / np.where(both, n1 + n0, 1.0), 0.0)
        for k, arr in enumerate((w * diff, w, np.where(both, n1, 0.0), np.where(both, n0, 0.0))):
            np.add.at(out[:, :, k], self.inv, arr)
        return out

    def versions(self, g: np.ndarray, n_perm: int, rng: np.random.Generator, active: np.ndarray | None = None) -> np.ndarray:
        """(clusters x features x (1 + n_perm) x 4): the real labelling first, then n_perm within-stratum shuffles among the active rows."""
        gs = self.sorted_labels(g)
        act = None if active is None else active[self.order].astype(np.float64)
        if act is not None:
            gs = gs * act
        base = self.base_for(act)
        parts = [self.sums(gs, base)]
        for _ in range(n_perm):
            parts.append(self.sums(self.shuffled_labels(gs, rng, act), base))
        return np.stack(parts, axis=2)


def tensor_stats(T: np.ndarray) -> dict[str, np.ndarray]:
    """Cluster-robust effect and t from a (clusters x features x 4) block. effect = sum A / sum W; the variance treats each cluster's residual
    (A_c - effect W_c) as one observation, so hundreds of movers on the same days do not masquerade as independent evidence."""
    A, W, n1, n0 = T[:, :, 0], T[:, :, 1], T[:, :, 2], T[:, :, 3]
    Wt = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(Wt > 0, A.sum(0) / np.where(Wt > 0, Wt, 1.0), np.nan)
        used = W > 0
        C = used.sum(0).astype(float)
        resid = np.where(used, A - eff[None, :] * W, 0.0)
        var = np.where(C > 1, (resid ** 2).sum(0) * C / np.maximum(C - 1.0, 1.0), np.nan) / np.where(Wt > 0, Wt, 1.0) ** 2
        se = np.sqrt(var)
        t = np.where(se > 0, eff / se, np.nan)
    return {"effect": eff, "se": se, "t": t, "df": C - 1.0, "clusters": C, "n1": n1.sum(0), "n0": n0.sum(0)}


def t_pvalue(t: np.ndarray, df: np.ndarray) -> np.ndarray:
    p = np.ones_like(t, dtype=float)
    ok = np.isfinite(t) & (df >= 1)
    p[ok] = 2.0 * sps.t.sf(np.abs(t[ok]), df[ok])
    return p


# ------------------------------------------------------------------------------------------------------------------ evidence store
class Cell:
    """Sufficient statistics of one comparison key across every feature and cluster ever seen, real labels and shuffles side by side."""

    def __init__(self, n_versions: int):
        self.V = n_versions
        self.features: list[str] = []
        self.clusters: dict[int, np.ndarray] = {}
        self.last_ordinal = 0
        self.units = 0

    def _col(self, feats: Sequence[str]) -> np.ndarray:
        for f in feats:
            if f not in self.features:
                self.features.append(f)
                for c, a in self.clusters.items():
                    self.clusters[c] = np.concatenate([a, np.zeros((1, self.V, 4))], axis=0)
        return np.array([self.features.index(f) for f in feats])

    def add(self, feats: Sequence[str], ids: np.ndarray, tensor: np.ndarray, last_ordinal: int) -> None:
        if tensor.shape[2] != self.V:
            raise LabError(f"tensor has {tensor.shape[2]} versions, cell expects {self.V}")
        cols = self._col(feats)
        for i, cid in enumerate(ids):
            cur = self.clusters.setdefault(int(cid), np.zeros((len(self.features), self.V, 4)))
            cur[cols] += tensor[i]
        self.last_ordinal = max(self.last_ordinal, int(last_ordinal))
        self.units += 1

    def stacked(self, cluster_subset: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids = sorted(self.clusters if cluster_subset is None else [c for c in cluster_subset if c in self.clusters])
        if not ids:
            return np.zeros(0, np.int64), np.zeros((0, len(self.features), self.V, 4))
        return np.array(ids, np.int64), np.stack([self.clusters[c] for c in ids])


class EvidenceStore:
    """All comparison cells. Keys read `lens|type|comparison|anchor`. Merging is the ONLY write path, and a unit's delta is merged exactly once."""

    def __init__(self, n_perm: int):
        self.n_perm = int(n_perm)
        self.cells: dict[str, Cell] = {}

    def merge(self, delta: "EvidenceDelta") -> None:
        for key, part in delta.parts.items():
            cell = self.cells.setdefault(key, Cell(1 + self.n_perm))
            cell.add(part.features, part.ids, part.tensor, part.last_ordinal)

    def digest(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.cells):
            c = self.cells[k]
            h.update(k.encode())
            h.update(json.dumps(c.features).encode())
            for cid in sorted(c.clusters):
                h.update(np.int64(cid).tobytes())
                h.update(np.round(c.clusters[cid], 9).tobytes())
        return h.hexdigest()[:20]

    def n_tests(self) -> int:
        return sum(len(c.features) for c in self.cells.values())

    def save(self, path) -> str:
        """Write an .npz atomically and return the sha256 of the file (recorded in the manifest that commits it)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}            # Any: numpy's savez stub mistypes **kwds as bool
        meta: dict[str, Any] = {"n_perm": self.n_perm, "cells": []}
        for i, k in enumerate(sorted(self.cells)):
            c = self.cells[k]
            ids, t = c.stacked()
            arrays[f"ids{i}"], arrays[f"t{i}"] = ids, t
            meta["cells"].append({"key": k, "features": c.features, "last": c.last_ordinal, "units": c.units})
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return hashlib.sha256(p.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, expect_sha: str | None = None) -> "EvidenceStore":
        p = Path(path)
        raw = p.read_bytes()
        if expect_sha is not None and hashlib.sha256(raw).hexdigest() != expect_sha:
            raise LabError(f"evidence file {p.name} fails its checksum")
        z = np.load(p, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        st = cls(meta["n_perm"])
        for i, m in enumerate(meta["cells"]):
            c = Cell(1 + st.n_perm)
            c.features = list(m["features"])
            for cid, arr in zip(z[f"ids{i}"], z[f"t{i}"]):
                c.clusters[int(cid)] = np.array(arr)
            c.last_ordinal, c.units = int(m["last"]), int(m["units"])
            st.cells[m["key"]] = c
        return st


@dataclasses.dataclass
class DeltaPart:
    features: list[str]
    ids: np.ndarray
    tensor: np.ndarray
    last_ordinal: int


@dataclasses.dataclass
class EvidenceDelta:
    """What one unit adds to the store. Built without touching the store, so a unit that dies half way leaves nothing behind."""
    parts: dict[str, DeltaPart] = dataclasses.field(default_factory=dict)
    n_rows: int = 0

    def add(self, key: str, part: DeltaPart) -> None:
        if key in self.parts:
            raise LabError(f"duplicate evidence key {key} inside one unit")
        self.parts[key] = part


def evidence_key(lens: str, tcode: int, comp: str, anchor: str) -> str:
    return f"{lens}|{type_label(tcode)}|{comp}|{anchor}"


def pair_comp(a: str, b: str, h: int) -> str:
    return f"{a}~{b}@h{h}"


def split_key(key: str) -> tuple[str, str, str, str]:
    lens, typ, comp, anchor = key.split("|")
    return lens, typ, comp, anchor


# ------------------------------------------------------------------------------------------------------------------ one unit's evidence
MASK32 = 0xFFFFFFFF


def unit_delta(grid: Grid, epp: pd.DataFrame, registry: PrecursorRegistry, columns: Sequence[str], cfg: LabConfig, uid: str,
               sector_codes: np.ndarray | None = None, externals: Mapping[str, Any] | None = None, lenses: Sequence[str] | None = None,
               memo: dict | None = None) -> tuple[EvidenceDelta, dict[str, Any]]:
    """Comparison statistics for one unit. `epp` is the unit's episode frame joined to its path labels. Two families of comparison per lens and
    episode type: (a) 'ctl' - episodes against matched non-episodes, read at the PRE anchor (what could be known before the move); (b) class
    pairs - e.g. reversals against spikes among the same type, read at BOTH anchors (before the move; and at the move day's close, before the
    outcome). Every comparison also gets `cfg.n_perm` label-shuffled copies. Nothing here touches shared state.
    `lenses` restricts the comparisons to the unit's own lens: a unit is (year, slice, LENS), and computing every lens in every unit would
    add each lens's evidence once per lens (a 4x double count with the default four lenses). `memo` reuses ranked panels of the same grid
    across the lens units of one (year, slice)."""
    useed = unit_seed(uid)
    delta = EvidenceDelta()
    info: dict[str, Any] = {"missing": [], "rows": 0, "comparisons": 0, "skipped_thin": 0}
    if len(epp) == 0 or not columns:
        return delta, info
    ctx = Ctx(grid, sector_codes, externals)
    want = sorted({c.split("@")[0] for c in columns})
    if memo is not None and memo.get("panels_grid") is grid:
        have, miss_old = memo["panels"], memo["panels_missing"]
        todo = [n for n in want if n not in have and n not in miss_old]
        new, new_miss = ranked_panels(ctx, registry, todo) if todo else ({}, [])
        have.update(new)
        miss_old.extend(new_miss)
        panels = {n: have[n] for n in want if n in have}
        missing = [n for n in want if n in miss_old]
    else:
        panels, missing = ranked_panels(ctx, registry, want)
        if memo is not None:
            memo.update(panels_grid=grid, panels=dict(panels), panels_missing=list(missing))
    cols = [c for c in columns if c.split("@")[0] in panels]
    info["missing"] = missing
    if not cols:
        return delta, info
    ordinals = np.array([d.toordinal() for d in grid.dates], dtype=np.int64)
    for lens in (cfg.lenses if lenses is None else lenses):
        rows = build_rowset(grid, epp, lens, cfg, useed)
        types = rows.types(cfg.min_type_rows)
        info["skipped_thin"] += len(rows.types(1)) - len(types)
        if not types:
            continue
        info["rows"] += len(rows)
        con = Contrast(gather_features(panels, cols, rows.ti, rows.nj, "pre"), rows.stratum, cluster_of(grid.dates[rows.ti], cfg.cluster_months))
        for tc in types:
            key = evidence_key(lens, tc, "ctl", "pre")
            grp = rows.is_ep & (rows.tcode == tc)
            rng = np.random.default_rng([cfg.seed & MASK32, useed, int(stable_hash(key, 8), 16)])
            delta.add(key, DeltaPart(list(cols), con.cluster_ids, con.versions(grp, cfg.n_perm, rng, (~rows.is_ep) | grp),
                                     int(ordinals[rows.ti[grp]].max())))
            info["comparisons"] += 1
        band, side = lens_band_side(epp, lens)
        tall = np.where((band > 0) & (side != 0), type_code(band, side), -1)
        for tc in types:
            for a, b in cfg.pairs:
                for h in cfg.pair_horizons:
                    if f"cls_{h}" not in epp:
                        continue
                    lab = epp[f"cls_{h}"].to_numpy()
                    sel = (tall == tc) & epp[f"ok_{h}"].to_numpy() & np.isin(lab, (a, b))
                    ga = lab[sel] == a
                    if sel.sum() < cfg.min_type_rows or ga.sum() < 10 or (~ga).sum() < 10:
                        info["skipped_thin"] += 1
                        continue
                    ti, nj = epp["ti"].to_numpy()[sel], epp["nj"].to_numpy()[sel]
                    strat = ordinals[ti] // cfg.class_stratum_days
                    cl = cluster_of(grid.dates[ti], cfg.cluster_months)
                    matured = int(pd.Timestamp(pd.to_datetime(epp["matured_at"].to_numpy()[sel]).max()).toordinal())
                    for anchor in ANCHORS:
                        key = evidence_key(lens, tc, pair_comp(a, b, h), anchor)
                        rng = np.random.default_rng([cfg.seed & MASK32, useed, int(stable_hash(key, 8), 16)])
                        c2 = Contrast(gather_features(panels, cols, ti, nj, anchor), strat, cl)
                        delta.add(key, DeltaPart(list(cols), c2.cluster_ids, c2.versions(ga, cfg.n_perm, rng), matured))
                        info["comparisons"] += 1
    delta.n_rows = info["rows"]
    return delta, info


# ------------------------------------------------------------------------------------------------------------------ controls: walk-forward and eras
def _one(T: np.ndarray, fj: int, sel: np.ndarray | None = None) -> dict[str, float]:
    block = T[:, fj:fj + 1, 0, :] if sel is None else T[sel][:, fj:fj + 1, 0, :]
    st = tensor_stats(block)
    return {k: float(v[0]) for k, v in st.items()}


def walk_forward(T: np.ndarray, fj: int, rules: EvidenceRules) -> dict[str, Any]:
    """Discover on the past, confirm on the next block. Clusters (time-ordered) are cut into `wf_folds` contiguous folds; for each fold k >= 1 the
    effect is estimated on folds before it, and only if that past estimate was already visible (signed t >= wf_screen_t) is fold k a test:
    it passes when the same sign shows up there with signed t >= wf_confirm_t. A fold with a single cluster has no variance and is not a test."""
    C = T.shape[0]
    parts = [p for p in np.array_split(np.arange(C), rules.wf_folds) if len(p)]
    tested = passed = 0
    detail = []
    for k in range(1, len(parts)):
        train, test = np.concatenate(parts[:k]), parts[k]
        if len(train) < 2 or len(test) < 2:
            continue
        a, b = _one(T, fj, train), _one(T, fj, test)
        if not (math.isfinite(a["t"]) and math.isfinite(b["t"])):
            continue
        s = 1.0 if a["effect"] >= 0 else -1.0
        if s * a["t"] < rules.wf_screen_t:
            continue
        tested += 1
        ok = s * b["t"] >= rules.wf_confirm_t
        passed += int(ok)
        detail.append({"fold": k, "train_t": a["t"], "test_t": b["t"], "test_effect": b["effect"], "ok": ok})
    enough = tested >= rules.wf_min_tested
    return {"tested": tested, "passed": passed, "enough": enough, "ok": bool(enough and passed / tested >= rules.wf_pass_frac), "detail": detail}


def era_check(ids: np.ndarray, T: np.ndarray, fj: int, cfg: LabConfig, rules: EvidenceRules) -> dict[str, Any]:
    """The effect must keep its sign, with signed t >= era_t, in EVERY era that has enough clusters, and at least `era_min_eras` eras must have
    enough clusters. An effect that lives in one era is a regime, not a precursor."""
    yrs = np.array([cluster_year(int(c), cfg.cluster_months) for c in ids])
    eras = era_of(pd.DatetimeIndex([pd.Timestamp(year=int(y), month=6, day=30) for y in yrs]), cfg.era_edges)
    overall = _one(T, fj)
    s = 1.0 if overall["effect"] >= 0 else -1.0
    rows = []
    for e in np.unique(eras):
        sel = eras == e
        if sel.sum() < rules.era_min_clusters:
            continue
        r = _one(T, fj, sel)
        rows.append({"era": int(e), "effect": r["effect"], "t": r["t"], "clusters": int(sel.sum())})
    enough = len(rows) >= rules.era_min_eras
    ok = enough and all(math.isfinite(r["t"]) and s * r["t"] >= rules.era_t and s * r["effect"] > 0 for r in rows)
    return {"eras": rows, "enough": enough, "ok": bool(ok)}


# ------------------------------------------------------------------------------------------------------------------ candidates
class CandidateStatus(_StrEnum):
    HYPOTHESIS = "HYPOTHESIS"                        # tracked; has not earned anything
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # passes the shuffles and the ledger but walk-forward or eras cannot be judged yet
    FAILED_SHUFFLE = "FAILED_SHUFFLE"
    FAILED_WALK_FORWARD = "FAILED_WALK_FORWARD"
    FAILED_ERA = "FAILED_ERA"
    CANDIDATE = "CANDIDATE"                          # survived every control; a research question, never a trading rule


LENS_WORDS = {"c2c": "close-to-close", "o2c": "open-to-close", "rng": "intraday-range", "gap": "opening-gap"}
FAMILY_WORDS = {"volume": "volume", "range": "range", "compression": "range compression", "return": "prior return", "gap": "gap",
                "structure": "price structure", "volatility": "volatility", "mover_history": "prior-mover history", "candle": "candle shape",
                "relative": "market-relative move", "sector": "sector move", "event": "filing/event timing", "external": "external"}


@dataclasses.dataclass(frozen=True)
class Candidate:
    candidate_id: str
    key: str
    lens: str
    type: str
    comparison: str
    anchor: str
    feature: str
    family: str
    direction: str                                   # higher / lower rank in the labelled group
    effect: float
    t: float
    p_param: float
    p_perm: float
    q: float
    n1: int
    n0: int
    clusters: int
    wf_tested: int
    wf_passed: int
    eras: tuple[tuple[int, float, float], ...]
    status: CandidateStatus
    reasons: tuple[str, ...]
    matured_ordinal: int
    years: tuple[int, ...]
    eval_seq: int
    text: str

    @property
    def matured_at(self) -> str:
        import datetime as _dt
        return _dt.date.fromordinal(self.matured_ordinal).isoformat() if self.matured_ordinal > 0 else ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = str(self.status)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Candidate":
        d = dict(d)
        d["status"] = CandidateStatus(d["status"])
        d["eras"] = tuple(tuple(x) for x in d["eras"])
        d["reasons"], d["years"] = tuple(d["reasons"]), tuple(d["years"])
        return Candidate(**d)


def alpha_id(prefix: str, obj: Any, n: int = 12) -> str:
    """Content-derived identifier written in letters only. A hex hash can contain digit runs that read as a year or a date to the trader-view
    screen (about one id in fifty), so ids are re-encoded a-p."""
    return prefix + "".join(chr(ord("a") + int(ch, 16)) for ch in stable_hash(obj, n))


def problem_of(comp: str) -> Problem:
    """Which of the owner's objectives a comparison serves: knowing WHETHER it moves (volatility), which way it goes next (direction), what
    the stock will cost (loss avoidance)."""
    if comp == "ctl":
        return Problem.VOLATILITY
    if comp.startswith("REVERSED_NEXT_DAY") or comp.startswith("SPIKED_NEXT_DAY"):
        return Problem.DIRECTION
    if comp.startswith("STOPPED"):
        return Problem.LOSS_AVOIDANCE
    return Problem.VOLATILITY


def describe(key: str, feature: str, family: str, effect: float, t: float) -> str:
    """Identity-free sentence: no date, no year, no ticker. Numbers are ranks and t statistics only."""
    lens, typ, comp, anchor = split_key(key)
    band, direction = typ.rsplit("_", 1)
    base, lag = feature.split("@")
    when = "at the close before the move day" if anchor == "pre" else "at the close of the move day, before the next session"
    lagtxt = "" if lag == "0" else f" ({lag} sessions earlier still)"
    hl = "higher" if effect >= 0 else "lower"
    if comp == "ctl":
        vs = "matched stocks of the same volatility that did not move"
        what = f"Among {direction} movers ({LENS_WORDS.get(lens, lens)} measure, band {band.replace('_', ' to ')})"
    else:
        cls, h = comp.split("@h")
        a, b = cls.split("~")
        vs = f"the movers whose path was {b.lower().replace('_', ' ')} at horizon {h}"
        what = f"Among {direction} movers ({LENS_WORDS.get(lens, lens)} measure, band {band.replace('_', ' to ')}) whose path was {a.lower().replace('_', ' ')}"
    return (f"{what}, the {FAMILY_WORDS.get(family, family)} precursor {base}{lagtxt} {when} ranks {hl} than {vs}; "
            f"cross-sectional rank shift {effect:+.3f}, t {t:.1f}.")


@dataclasses.dataclass
class EvalReport:
    run_id: str
    evidence_digest: str
    n_cells: int
    n_tests: int
    n_eligible: int
    n_tracked: int
    pool_size: int
    ledger_trials: int
    ledger_distinct: int
    expected_best_null_t: float
    by_status: dict[str, int]
    skipped: bool = False
    promoted: tuple = ()                             # CANDIDATE ids, strongest first (engine.research.loop turns these into questions)
    tracked_top: tuple = ()                          # the strongest tracked ids of any status (for reports and question raising)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_evidence(store: EvidenceStore, ledger: TrialLedger, rules: EvidenceRules, cfg: LabConfig, registry: PrecursorRegistry, now,
                      run_id: str, eval_seq: int) -> tuple[dict[str, Candidate], EvalReport]:
    """Turn accumulated sufficient statistics into tracked hypotheses and candidates. Every eligible test is entered in the cumulative ledger
    (so a result is judged against the whole search history, not this call); its p-value is the larger of the parametric cluster-robust p and
    the pooled shuffled-null p; a tracked test earns CANDIDATE only by beating every shuffle in its own family AND surviving walk-forward AND
    keeping its sign in every era."""
    tests: list[dict[str, Any]] = []
    pool: list[np.ndarray] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in sorted(store.cells):
        cell = store.cells[key]
        ids, T = cell.stacked()
        if len(ids) < rules.min_clusters or T.shape[1] == 0:
            continue
        cache[key] = (ids, T)
        real = tensor_stats(T[:, :, 0, :])
        elig = (real["n1"] >= rules.min_n1) & (real["n0"] >= rules.min_n0) & (real["clusters"] >= rules.min_clusters) & np.isfinite(real["t"])
        if not elig.any():
            continue
        nulls = np.stack([tensor_stats(T[:, :, v, :])["t"] for v in range(1, T.shape[2])])
        nl = np.abs(nulls[:, elig])
        nl = nl[np.isfinite(nl)]
        pool.append(nl)
        kmax = float(nl.max()) if nl.size else 0.0
        pp = t_pvalue(real["t"], real["df"])
        for j in np.flatnonzero(elig):
            tests.append({"key": key, "j": int(j), "feature": cell.features[j], "effect": float(real["effect"][j]), "t": float(real["t"][j]),
                          "p_param": float(pp[j]), "n1": int(real["n1"][j]), "n0": int(real["n0"][j]), "clusters": int(real["clusters"][j]),
                          "kmax": kmax, "last": cell.last_ordinal})
    pool_all = np.sort(np.concatenate(pool)) if pool else np.zeros(0)
    n_all = sum(len(c.features) for c in store.cells.values())
    digest = store.digest()
    if not tests:
        rep = EvalReport(run_id, digest, len(store.cells), n_all, 0, 0, len(pool_all), ledger.total_trials, ledger.distinct_trials,
                         ledger.expected_best_null_t(), {})
        return {}, rep
    absT = np.abs(np.array([x["t"] for x in tests]))
    ge = len(pool_all) - np.searchsorted(pool_all, absT, side="left")
    p_perm = (1.0 + ge) / (1.0 + len(pool_all))
    p_use = np.maximum(np.array([x["p_param"] for x in tests]), p_perm)
    fams = ["precursor:" + registry.family_of(x["feature"]) if x["feature"].split("@")[0] in registry.specs else "precursor:external" for x in tests]
    q = ledger.register(run_id, now, [f"{x['key']}|{x['feature']}" for x in tests], p_use, fams, digest)
    out: dict[str, Candidate] = {}
    counts: dict[str, int] = {}
    for i, x in enumerate(tests):
        if q[i] > rules.screen_q:
            continue
        ids, T = cache[x["key"]]
        fam = fams[i].split(":")[1]
        reasons: list[str] = []
        shuffle_bad = p_perm[i] > rules.alpha_perm or absT[i] <= x["kmax"]
        if p_perm[i] > rules.alpha_perm:
            reasons.append("pooled shuffled-null p above bar")
        if absT[i] <= x["kmax"]:
            reasons.append("does not beat every shuffled label in its family")
        if q[i] > rules.alpha_q:
            reasons.append("cumulative q above bar")
        if abs(x["effect"]) < rules.min_effect:
            reasons.append("effect below the minimum rank shift")
        wf: dict[str, Any] = {"tested": 0, "passed": 0, "ok": False, "enough": False}
        er: dict[str, Any] = {"eras": [], "ok": False, "enough": False}
        if not shuffle_bad:
            wf, er = walk_forward(T, x["j"], rules), era_check(ids, T, x["j"], cfg, rules)
            if wf["enough"] and not wf["ok"]:
                reasons.append("failed walk-forward")
            if er["enough"] and not er["ok"]:
                reasons.append("sign or size not held in every era")
            if not wf["enough"]:
                reasons.append("too few walk-forward tests to judge")
            if not er["enough"]:
                reasons.append("too few eras to judge")
        if shuffle_bad:
            status = CandidateStatus.FAILED_SHUFFLE
        elif wf["enough"] and not wf["ok"]:
            status = CandidateStatus.FAILED_WALK_FORWARD
        elif er["enough"] and not er["ok"]:
            status = CandidateStatus.FAILED_ERA
        elif not (wf["enough"] and er["enough"]):
            status = CandidateStatus.INSUFFICIENT_EVIDENCE
        elif q[i] > rules.alpha_q or abs(x["effect"]) < rules.min_effect:
            status = CandidateStatus.HYPOTHESIS
        else:
            status = CandidateStatus.CANDIDATE
        cid = alpha_id("pc", {"k": x["key"], "f": x["feature"]})
        lens, typ, comp, anchor = split_key(x["key"])
        years = tuple(sorted({cluster_year(int(c), cfg.cluster_months) for c in ids}))
        text = describe(x["key"], x["feature"], fam, x["effect"], x["t"])
        out[cid] = Candidate(cid, x["key"], lens, typ, comp, anchor, x["feature"], fam, "higher" if x["effect"] >= 0 else "lower", x["effect"], x["t"],
                             x["p_param"], float(p_perm[i]), float(q[i]), x["n1"], x["n0"], x["clusters"], int(wf["tested"]), int(wf["passed"]),
                             tuple((e["era"], e["effect"], e["t"]) for e in er["eras"]), status, tuple(reasons), int(x["last"]), years, eval_seq, text)
        counts[str(status)] = counts.get(str(status), 0) + 1
    order = sorted(out, key=lambda k: (-abs(out[k].t), k))
    rep = EvalReport(run_id, digest, len(store.cells), n_all, len(tests), len(out), len(pool_all), ledger.total_trials, ledger.distinct_trials,
                     ledger.expected_best_null_t(), counts, promoted=tuple(k for k in order if out[k].status == CandidateStatus.CANDIDATE),
                     tracked_top=tuple(order[:25]))
    return out, rep


def assert_identity_free(c: Candidate, tickers: Iterable[str] = ()) -> None:
    """A candidate's words may not carry a date, a year or a stock name."""
    why = string_reasons(c.text) + string_reasons(c.candidate_id)
    if why:
        raise FirewallBreach(f"candidate {c.candidate_id} text carries {why}")
    banned = {str(t).upper() for t in tickers}
    if banned and any(tok in banned for tok in c.text.replace(",", " ").replace(";", " ").split()):
        raise FirewallBreach(f"candidate {c.candidate_id} names a stock")


def to_question(c: Candidate, created_real: str) -> ResearchQuestion:
    """The candidate as a research question (R10-style knowledge objects are built downstream from it; here it stays a plain record)."""
    comp = c.comparison
    return ResearchQuestion.make(
        text=f"Is this precursor real and stable? {c.text}", source="mover_episode", problem=problem_of(comp), created_real=created_real,
        evidence_through=c.matured_at or created_real,
        success=f"the rank shift keeps its sign with signed t above the bar on a fresh block of years (now t {c.t:.1f}, shift {c.effect:+.3f})",
        failure="sign flips or the shift falls under the minimum on fresh years, or it is explained by another registered precursor")


def as_matured_record(c: Candidate, created_real: str, code_hash: str, data_hash: str, config_hash: str, seed: int) -> MaturedRecord:
    """Wrap a candidate for the only road to the trader (MaturedRecord.gate(now)). Payload is identity-free."""
    assert_identity_free(c)
    prov = Provenance(created_real=created_real, learned_at=c.matured_at, code_hash=code_hash, data_hash=data_hash, config_hash=config_hash,
                      experiment_id=c.candidate_id, seed=seed, outcomes_seen_through=c.matured_at)
    payload = {"kind": "mover_precursor", "candidate_id": c.candidate_id, "status": str(c.status), "lens": c.lens, "type": c.type,
               "comparison": c.comparison, "anchor": c.anchor, "feature": c.feature, "direction": c.direction, "effect": c.effect, "t": c.t,
               "q": c.q, "decision_effect": "NONE", "text": c.text}
    assert_trader_safe(payload, f"candidate {c.candidate_id} payload")
    return MaturedRecord(c.candidate_id, c.matured_at, payload, prov)


def release_candidates(cands: Mapping[str, Candidate], now, replay_windows: Sequence[tuple[Any, Any]] = (), created_real: str = "",
                       code_hash: str = "", data_hash: str = "", config_hash: str = "", seed: int = 0) -> tuple[list[MaturedRecord], dict[str, str]]:
    """Records safe to hand to the curator at `now`: status CANDIDATE, maturity strictly before now, and nothing built from a year that is being
    replayed in disguise (same-year rerun leak, mapping rule 27). Returns (records, {candidate_id: why it was withheld})."""
    replay_years: set[int] = set()
    for a, b in replay_windows:
        replay_years |= set(range(as_date(a).year, as_date(b).year + 1))
    out, held = [], {}
    for cid in sorted(cands):
        c = cands[cid]
        if c.status != CandidateStatus.CANDIDATE:
            held[cid] = f"status {c.status}"
            continue
        if not c.matured_at:
            held[cid] = "no maturity date"
            continue
        try:
            require_past(c.matured_at, now, f"candidate {cid}")
        except FirewallBreach:
            held[cid] = "not yet matured"
            continue
        if replay_years & set(c.years):
            held[cid] = "built from a year currently replayed in disguise"
            continue
        rec = as_matured_record(c, created_real or c.matured_at, code_hash, data_hash, config_hash, seed)
        rec.gate(now)                                     # fail closed: the gate itself must accept it
        out.append(rec)
    return out, held


# ------------------------------------------------------------------------------------------------------------------ market-state scan
def _lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 4 or np.nanstd(x) == 0:
        return 0.0
    a, b = x[:-1] - np.nanmean(x), x[1:] - np.nanmean(x)
    d = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def market_state_scan(counts: pd.DataFrame, ctx: Ctx, lag: int = 1, min_days: int = 30) -> list[dict[str, Any]]:
    """Date-level precursors: does the NUMBER of movers on a day depend on the market's state the day before? Cross-sectional ranks cannot see
    a quantity that is constant across names on a date, so this is its own scan. State is read `lag` >= 1 sessions before the count. The
    effective sample size is deflated by the product of the two series' lag-1 autocorrelations, so slow-moving states are not over-counted."""
    if lag < 1:
        raise LabError("market state must be read at least one session before the count")
    if len(counts) < min_days:
        return []
    idx = ctx.g.dates.get_indexer(counts.index)
    if (idx < 0).any():
        raise LabError("counts dates are not on the grid")
    rows = []
    for sname in ("ret", "breadth", "disp"):
        x_full = shift_down(ctx.market(sname)[:, None], lag)[:, 0]
        x = x_full[idx]
        for col in [c for c in counts.columns if c.split("_")[0] in {m.value for m in EPI.Measure}]:
            y = counts[col].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_days or np.std(y[ok]) == 0 or np.std(x[ok]) == 0:
                continue
            xs, ys = x[ok], y[ok]
            r = float(np.corrcoef(xs, ys)[0, 1])
            rho = _lag1_autocorr(xs) * _lag1_autocorr(ys)
            n_eff = float(np.clip(len(xs) * (1 - rho) / (1 + rho) if rho > -0.99 else len(xs), 3.0, len(xs)))
            t = r * math.sqrt((n_eff - 2.0) / max(1.0 - r * r, 1e-12))
            rows.append({"count": col, "state": sname, "lag": int(lag), "r": r, "t": float(t), "n": int(len(xs)), "n_eff": n_eff,
                         "p": float(2.0 * sps.t.sf(abs(t), max(n_eff - 2.0, 1.0)))})
    return rows


# ------------------------------------------------------------------------------------------------------------------ lab state
@dataclasses.dataclass
class LabState:
    """Everything the sweep knows. Mutated only by `commit_unit` and `evaluate_state`; persisted only by `LabStore.commit`."""
    cfg: LabConfig
    ecfg: EpisodeConfig
    pcfg: EP.PathConfig
    rules: EvidenceRules
    registry: PrecursorRegistry
    book: CoverageBook
    evidence: EvidenceStore
    ledger: TrialLedger
    candidates: dict[str, Candidate] = dataclasses.field(default_factory=dict)
    history: dict[str, list[list]] = dataclasses.field(default_factory=dict)
    market: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    daily: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)      # per-session mover counts of each slice-year (first lens only)
    notes: list[str] = dataclasses.field(default_factory=list)
    eval_seq: int = 0
    commit_seq: int = 0
    last_eval_digest: str = ""
    next_eval_at: int = 8
    widen: bool = False                                                          # grow the registry family by family when coverage is complete
    widened: list[str] = dataclasses.field(default_factory=list)                 # families added by widening, in order (persisted, replayed)
    labels: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)  # per unit: episode_paths.label_counts of its lens
    units_run: int = 0                                                           # unit computations committed (widened re-runs included)

    def digest(self) -> str:
        return stable_hash({"book": self.book.digest(), "ev": self.evidence.digest(), "cand": sorted(self.candidates),
                            "led": self.ledger.total_trials, "eval": self.eval_seq}, 16)

    def units_done(self) -> int:
        return len(self.book.records)


def new_state(years: Sequence[int], cfg: LabConfig = LabConfig(), ecfg: EpisodeConfig = EpisodeConfig(), pcfg: EP.PathConfig = EP.PathConfig(),
              rules: EvidenceRules = EvidenceRules(), registry: PrecursorRegistry | None = None, first_eval_units: int = 8,
              widen: bool = True) -> LabState:
    """A fresh sweep. `widen` (default on): when every unit is covered for the current registry, the sweep adds the next precursor family
    of the built-in registry and keeps going (the planted loop world went idle after 3 units because nothing ever widened it)."""
    cfg.require_valid()
    ecfg.require_valid()
    pcfg.require_valid()
    reg = registry if registry is not None else default_registry()
    return LabState(cfg, ecfg, pcfg, rules, reg, CoverageBook(years, cfg.n_slices, cfg.lenses), EvidenceStore(cfg.n_perm), TrialLedger(),
                    next_eval_at=int(first_eval_units), widen=bool(widen))


class LabStore:
    """Directory layout: manifest.json (the single commit point; embeds a hash of its own body and the sha256 of each data file) plus
    evidence_<seq>.npz, ledger_<seq>.json, candidates_<seq>.json. Data files are written first, the manifest last, atomically; a crash before
    the manifest replaces nothing. Files of older sequences are deleted only after the new manifest is in place."""

    def __init__(self, directory):
        self.dir = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def commit(self, st: LabState) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        seq = st.commit_seq + 1
        ev_sha = st.evidence.save(self.dir / f"evidence_{seq}.npz")
        led = json.dumps(st.ledger.to_dict(), sort_keys=True)
        _atomic_text(self.dir / f"ledger_{seq}.json", led)
        cand = json.dumps({"candidates": {k: v.to_dict() for k, v in sorted(st.candidates.items())}, "history": st.history}, sort_keys=True)
        _atomic_text(self.dir / f"candidates_{seq}.json", cand)
        cnt_sha = self._save_counts(st, self.dir / f"counts_{seq}.npz")
        lab = json.dumps(st.labels, sort_keys=True)
        _atomic_text(self.dir / f"labels_{seq}.json", lab)
        body = {"seq": seq, "cfg": st.cfg.digest(), "ecfg": st.ecfg.digest(), "pcfg": st.pcfg.digest(), "rules": st.rules.digest(),
                "coverage": st.book.to_dict(), "market": st.market, "notes": st.notes[-200:], "eval_seq": st.eval_seq,
                "last_eval_digest": st.last_eval_digest, "next_eval_at": st.next_eval_at, "widen": st.widen, "widened": list(st.widened),
                "units_run": st.units_run,
                "files": {"labels": [f"labels_{seq}.json", hashlib.sha256(lab.encode()).hexdigest()],
                          "evidence": [f"evidence_{seq}.npz", ev_sha], "ledger": [f"ledger_{seq}.json", hashlib.sha256(led.encode()).hexdigest()],
                          "candidates": [f"candidates_{seq}.json", hashlib.sha256(cand.encode()).hexdigest()],
                          "counts": [f"counts_{seq}.npz", cnt_sha]}}
        _atomic_text(self.manifest_path, json.dumps({"hash": stable_hash(body, 24), "body": body}, sort_keys=True))
        st.commit_seq = seq
        for p in list(self.dir.glob("evidence_*.npz")) + list(self.dir.glob("ledger_*.json")) + list(self.dir.glob("candidates_*.json"))                 + list(self.dir.glob("counts_*.npz")) + list(self.dir.glob("labels_*.json")):
            n = int(p.stem.split("_")[1])
            if n != seq:
                p.unlink(missing_ok=True)
        for p in self.dir.glob("*.tmp*"):
            p.unlink(missing_ok=True)
        return seq

    @staticmethod
    def _save_counts(st: LabState, path: Path) -> str:
        arrays: dict[str, Any] = {}            # Any: numpy's savez stub mistypes **kwds as bool
        meta: dict[str, Any] = {"units": []}
        for i, uid in enumerate(sorted(st.daily)):
            d = st.daily[uid]
            arrays[f"d{i}"] = d.index.values.astype("datetime64[D]").astype(np.int64)
            arrays[f"v{i}"] = d.to_numpy(np.int64)
            meta["units"].append({"uid": uid, "columns": list(d.columns)})
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _load_counts(path: Path, sha: str) -> dict[str, pd.DataFrame]:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise LabError(f"{path.name} fails its checksum")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        return {m["uid"]: pd.DataFrame(z[f"v{i}"], index=pd.DatetimeIndex(z[f"d{i}"].astype("datetime64[D]").astype("datetime64[ns]")),
                                       columns=m["columns"]) for i, m in enumerate(meta["units"])}

    def load(self, cfg: LabConfig = LabConfig(), ecfg: EpisodeConfig = EpisodeConfig(), pcfg: EP.PathConfig = EP.PathConfig(),
             rules: EvidenceRules = EvidenceRules(), registry: PrecursorRegistry | None = None) -> LabState:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            body, h = raw["body"], raw["hash"]
        except (OSError, ValueError, KeyError) as e:
            raise LabError(f"manifest unreadable: {e}") from e
        if stable_hash(body, 24) != h:
            raise LabError("manifest fails its content hash")
        for name, have in (("cfg", cfg.digest()), ("ecfg", ecfg.digest()), ("pcfg", pcfg.digest())):
            if body[name] != have:
                raise DefinitionDrift(f"stored {name} differs from the one supplied: a sweep may not change its definitions midway")
        files = body["files"]
        for k in ("ledger", "candidates"):
            name, sha = files[k]
            if hashlib.sha256((self.dir / name).read_bytes()).hexdigest() != sha:
                raise LabError(f"{name} fails its checksum")
        ev = EvidenceStore.load(self.dir / files["evidence"][0], files["evidence"][1])
        st = LabState(cfg, ecfg, pcfg, rules, registry if registry is not None else default_registry(), CoverageBook.from_dict(body["coverage"]), ev,
                      TrialLedger.from_dict(json.loads((self.dir / files["ledger"][0]).read_text(encoding="utf-8"))))
        c = json.loads((self.dir / files["candidates"][0]).read_text(encoding="utf-8"))
        st.daily = self._load_counts(self.dir / files["counts"][0], files["counts"][1])
        st.candidates = {k: Candidate.from_dict(v) for k, v in c["candidates"].items()}
        st.history = c["history"]
        st.market, st.notes = body["market"], list(body["notes"])
        st.eval_seq, st.commit_seq = body["eval_seq"], body["seq"]
        st.last_eval_digest, st.next_eval_at = body["last_eval_digest"], body["next_eval_at"]
        st.widen = bool(body.get("widen", False))
        st.units_run = int(body.get("units_run", len(st.book.records)))
        for fam in body.get("widened", []):                    # replay the widening so a resumed sweep has the registry it had
            add_family(st, fam)
        st.widened = list(body.get("widened", []))
        if "labels" in files:
            name, sha = files["labels"]
            raw_l = (self.dir / name).read_bytes()
            if hashlib.sha256(raw_l).hexdigest() != sha:
                raise LabError(f"{name} fails its checksum")
            st.labels = json.loads(raw_l.decode("utf-8"))
        return st


# ------------------------------------------------------------------------------------------------------------------ loaders
Loader = Callable[[Unit, EpisodeConfig, int, int], Mapping[str, pd.DataFrame]]
ContextFn = Callable[[Unit, Grid], "tuple[np.ndarray | None, dict[str, Any]]"]


def cache_loader(n_slices: int, salt: int = 0, cache_dir=None, names: Sequence[str] = ("stocks_pre2000", "stocks")) -> Loader:
    """Loader over the daily parquet caches (the only place the sweep touches disk data). Reads only the columns of the unit's slice."""
    universe: list[str] = []

    def load(unit: Unit, ecfg: EpisodeConfig, extra_warm: int, extra_future: int) -> Mapping[str, pd.DataFrame]:
        if not universe:
            universe.extend(EPI.cache_tickers(cache_dir, names))
        a, b = EPI.year_window(unit.year, ecfg, extra_warm, extra_future)
        return EPI.load_bars(a, b, EPI.slice_tickers(universe, unit.slice_id, n_slices, salt), cache_dir, names=names)
    return load


def frame_loader(bars: Mapping[str, pd.DataFrame]) -> Loader:
    """Loader over an in-memory block (tests and small experiments): applies the same window rule as the real one."""
    def load(unit: Unit, ecfg: EpisodeConfig, extra_warm: int, extra_future: int) -> Mapping[str, pd.DataFrame]:
        return EPI.year_bars(bars, unit.year, ecfg, extra_warm, extra_future)
    return load


# ------------------------------------------------------------------------------------------------------------------ running a unit
@dataclasses.dataclass
class UnitOutcome:
    unit: Unit
    waiting: bool
    reason: str = ""
    record: UnitRecord | None = None
    delta: EvidenceDelta | None = None
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    market_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    counts: pd.DataFrame | None = None


def assert_before(bars: Mapping[str, pd.DataFrame], now) -> None:
    """Fail closed if a loader handed over sessions at or after `now`: the research world studies matured history only."""
    idx = bars["Close"].index if "Close" in bars else pd.Index([])
    if len(idx) and as_date(idx.max()) >= as_date(now):
        raise FirewallBreach(f"loader returned a session dated {as_date(idx.max())}, not strictly before now={as_date(now)}")


def _memo_loader(loader: Loader, memo: dict) -> Loader:
    """One-entry cache over a loader: the lens units of one (year, slice) read the same bars, so a grouped step reads them once."""
    def load(unit: Unit, ecfg: EpisodeConfig, extra_warm: int, extra_future: int) -> Mapping[str, pd.DataFrame]:
        key = (unit.year, unit.slice_id, ecfg.digest(), int(extra_warm), int(extra_future))
        if memo.get("raw_key") != key:
            memo.clear()
            memo["raw_key"], memo["raw"] = key, loader(unit, ecfg, extra_warm, extra_future)
        return memo["raw"]
    return load


def run_unit(state: LabState, unit: Unit, loader: Loader, now, context_fn: ContextFn | None = None, memo: dict | None = None) -> UnitOutcome:
    """Compute one unit's contribution WITHOUT touching state. Returns waiting=True (nothing recorded) if the year's sessions are not all final
    yet - a year needs its full calendar plus the longest look-forward horizon beyond it - so a partial year is never half-counted. Evidence is
    computed for the unit's OWN lens only. `memo` (a grouped step's cache) reuses the grid, the labelled episodes and the ranked panels of the
    same (year, slice) across its lens units; results are identical with or without it."""
    cols_all = state.registry.columns()
    old = state.book.records.get(unit.uid)
    pending = [c for c in cols_all if old is None or c not in old.features_done]
    if not pending:
        return UnitOutcome(unit, True, "nothing pending for this unit")
    raw = loader(unit, state.ecfg, state.registry.extra_warm(), 0)
    assert_before(raw, now)
    done_cols = tuple(sorted(set(cols_all) | set(old.features_done if old else ())))
    if "Close" not in raw or len(raw["Close"].index) == 0 or raw["Close"].shape[1] == 0:
        rec = UnitRecord(unit.uid, "", True, 0, 0, 0, 0, {}, done_cols, state.ecfg.digest(), "", str(as_date(now)))
        return UnitOutcome(unit, False, "empty", rec, EvidenceDelta(), {"empty": True}, [], pd.DataFrame())
    gkey = ("grid", unit.year, unit.slice_id, id(raw))
    if memo is not None and memo.get("grid_key") == gkey:
        keep, grid, eps, counts, epp, lo, yr_end = memo["grid_val"]
    else:
        keep = EPI.slice_tickers(raw["Close"].columns, unit.slice_id, state.cfg.n_slices, state.cfg.slice_salt)
        bars = {k: v[keep] for k, v in raw.items()}
        grid = build_grid(bars, state.ecfg)
        year_rows = np.flatnonzero(grid.dates.year == unit.year)
        if len(year_rows) == 0:
            rec = UnitRecord(unit.uid, "", True, 0, len(keep), 0, 0, {}, done_cols, state.ecfg.digest(), "", str(as_date(now)))
            return UnitOutcome(unit, False, "no sessions in year", rec, EvidenceDelta(), {"empty": True}, [], pd.DataFrame())
        lo, yr_end = int(year_rows[0]), int(year_rows[-1]) + 1
        if not (yr_end <= grid.shape[0] - state.ecfg.max_horizon and grid.dates[-1].year > unit.year):
            return UnitOutcome(unit, True, "year not final: data ends before the year plus the look-forward horizon")
        eps = EPI.episode_frame(grid, lo, yr_end)
        counts = EPI.day_counts(grid, lo, yr_end)
        epp = EP.with_paths(eps, EP.label_paths(grid, eps, state.pcfg))
        if memo is not None:
            memo["grid_key"], memo["grid_val"] = gkey, (keep, grid, eps, counts, epp, lo, yr_end)
    sector, ext = context_fn(unit, grid) if context_fn else (None, {})
    delta, info = unit_delta(grid, epp, state.registry, pending, state.cfg, unit.uid, sector, ext, lenses=(unit.lens,), memo=memo)
    first_lens = unit.lens == state.cfg.lenses[0]
    market_rows = market_state_scan(counts, Ctx(grid, sector, ext)) if (old is None and first_lens) else []
    if old is None:
        info["labels"] = EP.label_counts(epp, unit.lens)
    tot = {k: int(v) for k, v in EPI.band_totals(counts).items()}
    rec = UnitRecord(unit.uid, str(grid.dates[yr_end - 1].date()), True, int(len(counts)), int(grid.eligible()[lo:yr_end].any(0).sum()), int(len(eps)),
                     int(counts["n_suspect"].sum()), tot, done_cols, state.ecfg.digest(), current_code_hash(), str(as_date(now)))
    return UnitOutcome(unit, False, "", rec, delta, info, market_rows, counts)


def commit_unit(state: LabState, out: UnitOutcome) -> None:
    """Apply a finished unit: merge its evidence, mark it covered, enter its date-level tests in the ledger, keep its label ledger. All-or-nothing:
    a failure while merging puts the coverage record back as it was."""
    if out.record is None or out.delta is None:
        raise LabError(f"unit {out.unit.uid} has nothing to commit ({out.reason})")
    prev = state.book.records.get(out.unit.uid)
    state.book.mark(out.record)
    try:
        state.evidence.merge(out.delta)
        if out.market_rows:
            uid = out.unit.uid
            state.market[uid] = out.market_rows
            state.ledger.register(f"mkt-{uid}", out.record.finished_real, [f"{uid}|{r['count']}|{r['state']}" for r in out.market_rows],
                                  [r["p"] for r in out.market_rows], ["market_state"] * len(out.market_rows), f"unit:{uid}")
        if out.counts is not None and len(out.counts) and out.unit.lens == state.cfg.lenses[0]:
            state.daily[out.unit.uid] = out.counts.astype(np.int64)
        if "labels" in out.info:
            state.labels[out.unit.uid] = out.info["labels"]
        if out.info.get("missing"):
            state.notes.append(f"{out.unit.uid}: features without data: {','.join(sorted(out.info['missing']))}")
        state.units_run += 1
    except Exception:
        if prev is None:
            state.book.records.pop(out.unit.uid, None)
        else:
            state.book.records[out.unit.uid] = prev
        raise


def evaluate_state(state: LabState, now, force: bool = False) -> EvalReport | None:
    """Judge everything accumulated so far. Repeated looks are themselves multiple testing, so unless `force` an evaluation happens only when
    the amount of unit work (units run, widened re-runs included) has doubled since the last one (first at `next_eval_at`), and never twice on
    identical evidence."""
    n = state.units_run
    if not force and n < state.next_eval_at:
        return None
    digest = state.evidence.digest()
    if digest == state.last_eval_digest:
        return None
    seq = state.eval_seq + 1
    cands, rep = evaluate_evidence(state.evidence, state.ledger, state.rules, state.cfg, state.registry, now, f"eval{seq}-{digest[:10]}", seq)
    merged: dict[str, Candidate] = {}
    for cid, c in cands.items():
        merged[cid] = c
        state.history.setdefault(cid, []).append([seq, str(c.status), round(float(c.t), 4)])
    for cid, old in state.candidates.items():
        if cid not in merged:                              # nothing is deleted: a finding that fell out of screening is kept, demoted
            merged[cid] = dataclasses.replace(old, status=CandidateStatus.HYPOTHESIS, eval_seq=seq,
                                              reasons=old.reasons + ("no longer under the screening q",))
            state.history.setdefault(cid, []).append([seq, "DEMOTED", round(float(old.t), 4)])
    state.candidates = merged
    state.eval_seq, state.last_eval_digest = seq, digest
    state.next_eval_at = max(n * 2, state.next_eval_at)
    return rep


# ------------------------------------------------------------------------------------------------------------------ widening (never idle while work exists)
def family_order() -> list[str]:
    """Built-in precursor families in registration order: the ladder a widening sweep climbs."""
    seen: list[str] = []
    for s in _BUILTIN:
        if s.family not in seen:
            seen.append(s.family)
    return seen


def add_family(state: LabState, family: str) -> list[str]:
    """Register every built-in precursor of `family` that the state's registry lacks (each passes the future-invariance audit first)."""
    added = []
    for s in _BUILTIN:
        if s.family == family and s.name not in state.registry.specs:
            state.registry.register(s, audit=True)
            added.append(s.name)
    return added


def widen(state: LabState) -> str | None:
    """Add the next built-in family the registry does not fully contain. Returns the family, or None when the registry already holds every
    built-in precursor (then the sweep is truly exhausted until new years arrive)."""
    for fam in family_order():
        if add_family(state, fam):
            state.widened.append(fam)
            state.notes.append(f"widened: coverage complete, added the {fam} family")
            return fam
    return None


@dataclasses.dataclass
class StepReport:
    done: list[str]
    waiting: list[str]
    remaining: int
    evaluation: EvalReport | None
    coverage: dict[str, Any]
    candidates: int
    widened: list[str] = dataclasses.field(default_factory=list)
    exhausted: bool = False                          # nothing runnable is left and nothing can be widened: idle until new data
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"done": self.done, "waiting": self.waiting, "remaining": self.remaining,
                "evaluation": self.evaluation.to_dict() if self.evaluation else None, "coverage": self.coverage, "candidates": self.candidates,
                "widened": self.widened, "exhausted": self.exhausted, "seconds": round(self.seconds, 3)}


def _group_first(todo: list[Unit]) -> list[Unit]:
    """Keep the least-covered unit first, then pull forward the other lens units of its (year, slice) so their bars are read once."""
    head = todo[0]
    same = [u for u in todo[1:] if (u.year, u.slice_id) == (head.year, head.slice_id)]
    rest = [u for u in todo[1:] if (u.year, u.slice_id) != (head.year, head.slice_id)]
    return [head] + same + rest


def step(state: LabState, now, loader: Loader, max_units: int = 1, store: LabStore | None = None, context_fn: ContextFn | None = None,
         years_before: int | None = None, force_eval: bool = False, budget_s: float | None = None, grouped: bool = False) -> StepReport:
    """THE public entry (the research loop schedules this through compute_manager). Runs up to `max_units` unfinished units, least-covered
    first, and stops starting new ones once `budget_s` seconds have passed; commits after each so a killed job loses at most the unit in flight.
    When coverage is complete and the state widens, the next precursor family is added and the sweep carries on in the same call. Evaluates on
    the doubling schedule, and once more when the sweep is exhausted on evidence that was never judged (a small world - three years, one
    slice - used to finish its units before the first scheduled evaluation and so never produced a single candidate). Years from `years_before`
    (default: the year of `now`) onward are not touched. `grouped` runs the lens units of one (year, slice) together on one read of the bars."""
    t0 = time.monotonic()
    ybefore = as_date(now).year if years_before is None else years_before
    done: list[str] = []
    waiting: list[str] = []
    widened: list[str] = []
    memo: dict | None = {} if grouped else None
    ld = _memo_loader(loader, memo) if memo is not None else loader
    over = lambda: budget_s is not None and time.monotonic() - t0 >= budget_s
    while True:
        cols = state.registry.columns()
        todo = [u for u in state.book.pending(cols, years_before=ybefore) if u.uid not in waiting]
        if grouped and todo:
            todo = _group_first(todo)
        for u in todo:
            if len(done) >= max_units or over():
                break
            out = run_unit(state, u, ld, now, context_fn, memo)
            if out.waiting:
                waiting.append(u.uid)
                continue
            commit_unit(state, out)
            done.append(u.uid)
            if store is not None:
                store.commit(state)
        remaining = len([u for u in state.book.pending(cols, years_before=ybefore) if u.uid not in waiting])
        if remaining == 0 and state.widen and len(done) < max_units and not over():
            fam = widen(state)
            if fam is not None:
                widened.append(fam)
                continue
        break
    exhausted = remaining == 0 and not (state.widen and any(s.name not in state.registry.specs for s in _BUILTIN))
    unjudged = bool(state.evidence.cells) and state.evidence.digest() != state.last_eval_digest
    rep = None
    if done or force_eval or (exhausted and unjudged):
        rep = evaluate_state(state, now, force=force_eval or (exhausted and unjudged))
    if store is not None and (rep is not None or (widened and not done)):
        store.commit(state)
    return StepReport(done, waiting, remaining, rep, state.book.report(state.registry.columns()), len(state.candidates), widened, exhausted,
                      time.monotonic() - t0)


def sweep(state: LabState, now, loader: Loader, store: LabStore | None = None, context_fn: ContextFn | None = None,
          max_units: int | None = None, budget_s: float | None = None, grouped: bool = False) -> list[StepReport]:
    """Run `step` repeatedly until nothing is runnable (or `max_units` units have run, or `budget_s` seconds have passed). An always-on job
    calls this again whenever new years, new sessions or new precursors appear; with nothing new it returns immediately."""
    t0 = time.monotonic()
    reports, ran = [], 0
    per = max(1, len(state.cfg.lenses)) if grouped else 1
    while max_units is None or ran < max_units:
        left = None if budget_s is None else budget_s - (time.monotonic() - t0)
        if left is not None and left <= 0:
            break
        r = step(state, now, loader, per if max_units is None else min(per, max_units - ran), store, context_fn, budget_s=left, grouped=grouped)
        reports.append(r)
        if not r.done:
            break
        ran += len(r.done)
    return reports


def open_state(directory, years: Sequence[int], cfg: LabConfig = LabConfig(), ecfg: EpisodeConfig = EpisodeConfig(),
               pcfg: EP.PathConfig = EP.PathConfig(), rules: EvidenceRules = EvidenceRules(), registry: PrecursorRegistry | None = None,
               widen: bool = True, on_drift: str = "archive") -> tuple[LabState, LabStore]:
    """Resume from a directory if it holds a sweep, else start one. New years are appended to the coverage book, never replacing finished work.
    A stored sweep built under DIFFERENT definitions (e.g. the path taxonomy gained a class) cannot be continued: with on_drift='archive' the
    old directory is renamed aside, intact, and a fresh sweep starts (a note says so); with 'raise' the DefinitionDrift propagates. Corruption
    always raises."""
    if on_drift not in ("archive", "raise"):
        raise LabError("on_drift must be 'archive' or 'raise'")
    store = LabStore(directory)
    note = ""
    if store.exists():
        try:
            st = store.load(cfg, ecfg, pcfg, rules, registry)
            st.book.extend_years(years)
            st.widen = bool(widen)
            return st, store
        except DefinitionDrift as e:
            if on_drift == "raise":
                raise
            old = Path(directory)
            aside = old.with_name(f"{old.name}.superseded-{stable_hash(str(e), 8)}")
            k = 0
            while aside.exists():
                k += 1
                aside = old.with_name(f"{old.name}.superseded-{stable_hash(str(e), 8)}-{k}")
            os.replace(old, aside)
            note = f"previous sweep archived as {aside.name}: {e}"
    st = new_state(years, cfg, ecfg, pcfg, rules, registry, widen=widen)
    if note:
        st.notes.append(note)
    return st, store


def candidates_table(state: LabState, statuses: Sequence[str] | None = None) -> pd.DataFrame:
    """Tracked hypotheses and candidates as a table, strongest first."""
    rows = [c.to_dict() for c in state.candidates.values() if statuses is None or str(c.status) in statuses]
    if not rows:
        return pd.DataFrame(columns=["candidate_id", "key", "feature", "status", "effect", "t", "q"])
    d = pd.DataFrame(rows)
    return d.assign(abs_t=d["t"].abs()).sort_values("abs_t", ascending=False).drop(columns="abs_t").reset_index(drop=True)


def lab_report(state: LabState, top: int = 12) -> str:
    """Plain-text summary for the research page: coverage, the multiple-testing account, and the leading candidates. Identity-free."""
    cov = state.book.report(state.registry.columns())
    lines = [f"mover-episode lab: {cov['units_done']}/{cov['units_total']} units ({cov['fraction_done']:.0%}), evaluations {state.eval_seq}",
             f"ledger: {state.ledger.total_trials} looks, {state.ledger.distinct_trials} distinct tests, best-by-luck |t| {state.ledger.expected_best_null_t():.2f}",
             f"evidence: {len(state.evidence.cells)} comparison cells, {state.evidence.n_tests()} feature tests, {state.evidence.n_perm} shuffled copies each"]
    by: dict[str, int] = {}
    for c in state.candidates.values():
        by[str(c.status)] = by.get(str(c.status), 0) + 1
    lines.append("tracked: " + (", ".join(f"{k}={v}" for k, v in sorted(by.items())) or "none"))
    t = candidates_table(state)
    for _, r in t.head(top).iterrows():
        lines.append(f"  [{r['status']}] {r['feature']:<24} {r['key']:<44} shift {r['effect']:+.3f} t {r['t']:.1f} q {r['q']:.3g}")
    if cov["thin_cells"]:
        lines.append(f"thin band cells (<200 episodes): {len(cov['thin_cells'])}")
    lines.append(EPI.firewall_note())
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ feature diagnostics
def feature_health(F: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Per column: share of finite values, spread and whether it is constant. A precursor with a third of its values missing, or none varying,
    is a data problem to be fixed before any statistic is read."""
    if F.shape[1] != len(columns):
        raise LabError("columns do not match the feature matrix")
    fin = np.isfinite(F)
    with np.errstate(invalid="ignore"):
        sd = np.array([np.std(F[fin[:, j], j]) if fin[:, j].sum() > 1 else np.nan for j in range(F.shape[1])])
    return pd.DataFrame({"column": list(columns), "finite_share": fin.mean(axis=0) if len(F) else np.zeros(len(columns)), "std": sd,
                         "constant": ~(sd > 1e-9)})


def redundancy_groups(F: np.ndarray, columns: Sequence[str], threshold: float = 0.9, sample: int = 4000, seed: int = 0) -> list[list[str]]:
    """Groups of columns whose ranks correlate at least `threshold` in absolute value (connected components). Members of a group are one
    finding seen several times, not several findings; the ledger already counts them all, and candidates from one group are shown together."""
    n, f = F.shape
    if f < 2 or n < 30:
        return []
    rng = np.random.default_rng(seed)
    X = F[rng.choice(n, min(sample, n), replace=False)] if n > sample else F
    X = np.where(np.isfinite(X), X, 0.5)
    keep = X.std(axis=0) > 1e-9
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return []
    R = np.corrcoef(X[:, idx], rowvar=False)
    parent = list(range(len(idx)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if abs(R[a, b]) >= threshold:
                parent[find(a)] = find(b)
    groups: dict[int, list[str]] = {}
    for a in range(len(idx)):
        groups.setdefault(find(a), []).append(columns[idx[a]])
    return sorted([sorted(g) for g in groups.values() if len(g) > 1])


DECOY_PREFIX = "nullprobe_"                        # not "decoy_": the trader-view screen reads "dec" + digit as a month-day date


def _hash_uniform(ordinals: np.ndarray, tick_codes: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic pseudo-random uniform in [0, 1) for each (session, name): an integer hash of the date ordinal and a per-ticker code."""
    x = (ordinals.astype(np.uint64)[:, None] * np.uint64(2654435761) + tick_codes.astype(np.uint64)[None, :] * np.uint64(40503)
         + np.uint64((seed + 1) * 97531)) & np.uint64(0xFFFFFFFF)
    for shift, mul in ((15, 2246822519), (13, 3266489917)):
        x ^= x >> np.uint64(shift)
        x = (x * np.uint64(mul)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return x.astype(np.float64) / 4294967296.0


def decoy_spec(idx: int, seed: int = 0, lags: Sequence[int] = (0, 1, 2)) -> PrecursorSpec:
    """A precursor that is pure noise by construction (a hash of session and ticker, independent of everything). It travels through the sweep,
    the ledger, the shuffles and the candidate rules exactly like a real feature, so the number of decoys that ever look like discoveries is a
    direct measurement of the false-discovery rate of the whole machine."""
    def fn(c: Ctx):
        o = np.array([d.toordinal() for d in c.g.dates], dtype=np.int64)
        k = np.array([zlib.crc32(str(t).encode()) for t in c.g.tickers], dtype=np.int64)
        return _hash_uniform(o, k, seed * 1009 + idx)
    fn.__doc__ = f"decoy {idx}: noise, must never be a discovery"
    return PrecursorSpec(f"{DECOY_PREFIX}{idx}", "decoy", fn, tuple(lags), fn.__doc__)


def registry_with_decoys(n_decoys: int = 3, seed: int = 0, audit: bool = True) -> PrecursorRegistry:
    reg = default_registry(audit=audit)
    for i in range(n_decoys):
        reg.register(decoy_spec(i, seed), audit=audit)
    return reg


def decoy_report(state: LabState) -> dict[str, Any]:
    """How many decoy tests were eligible, tracked, and (worst case) promoted. With honest controls the promoted count is zero and the tracked
    count is near screen_q times the eligible count."""
    tests = tracked = promoted = 0
    for key, cell in state.evidence.cells.items():
        tests += sum(1 for f in cell.features if f.startswith(DECOY_PREFIX))
    for c in state.candidates.values():
        if c.feature.startswith(DECOY_PREFIX):
            tracked += 1
            promoted += int(c.status == CandidateStatus.CANDIDATE)
    return {"decoy_tests": tests, "decoy_tracked": tracked, "decoy_promoted": promoted,
            "tracked_share": tracked / tests if tests else float("nan"), "screen_q": state.rules.screen_q}


# ------------------------------------------------------------------------------------------------------------------ candidate follow-up
def _cell_of(state: LabState, c: Candidate):
    cell = state.evidence.cells.get(c.key)
    if cell is None or c.feature not in cell.features:
        raise LabError(f"candidate {c.candidate_id} has no evidence cell in this state")
    return cell, cell.features.index(c.feature)


def candidate_timeline(state: LabState, cid: str) -> pd.DataFrame:
    """The candidate's rank shift year by year (clusters within a year pooled): the picture a human wants before believing 'stable across eras'."""
    c = state.candidates[cid]
    cell, fj = _cell_of(state, c)
    ids, T = cell.stacked()
    yrs = np.array([cluster_year(int(i), state.cfg.cluster_months) for i in ids])
    rows = []
    for y in sorted(set(yrs.tolist())):
        sel = yrs == y
        st = tensor_stats(T[sel][:, fj:fj + 1, 0, :])
        rows.append({"year": int(y), "clusters": int(sel.sum()), "n1": int(st["n1"][0]), "effect": float(st["effect"][0]), "t": float(st["t"][0])})
    return pd.DataFrame(rows, columns=["year", "clusters", "n1", "effect", "t"])


def candidate_health(state: LabState, cid: str, recent_years: int = 3) -> dict[str, Any]:
    """Health of one finding: HEALTHY when the most recent years still show its sign at the confirm bar, DEGRADING when the sign holds but the
    evidence is gone, BROKEN when the sign reversed, INSUFFICIENT_EVIDENCE when there are too few clusters to say."""
    c = state.candidates[cid]
    cell, fj = _cell_of(state, c)
    ids, T = cell.stacked()
    overall = _one(T, fj) if len(ids) >= 2 else None
    yrs = np.array([cluster_year(int(i), state.cfg.cluster_months) for i in ids])
    recent_sel = yrs >= (yrs.max() - recent_years + 1) if len(yrs) else np.zeros(0, bool)
    if overall is None or recent_sel.sum() < 2 or (~recent_sel).sum() < 2:
        return {"candidate_id": cid, "health": Health.INSUFFICIENT_EVIDENCE, "overall": overall, "recent": None}
    recent = _one(T, fj, recent_sel)
    s = 1.0 if overall["effect"] >= 0 else -1.0
    if s * recent["effect"] <= 0:
        h = Health.BROKEN
    elif s * recent["t"] >= state.rules.wf_confirm_t and abs(recent["effect"]) >= 0.5 * abs(overall["effect"]):
        h = Health.HEALTHY
    else:
        h = Health.DEGRADING
    return {"candidate_id": cid, "health": h, "overall": overall, "recent": recent}


def health_check_all(state: LabState, recent_years: int = 3) -> dict[str, Health]:
    """Every tracked finding that has not failed outright gets a health verdict (C58-C61: nothing is left unchecked)."""
    out: dict[str, Health] = {}
    for cid, c in sorted(state.candidates.items()):
        if c.status in (CandidateStatus.FAILED_SHUFFLE,):
            continue
        try:
            out[cid] = candidate_health(state, cid, recent_years)["health"]
        except LabError:
            out[cid] = Health.UNKNOWN
    return out


def knowability_of(c: Candidate) -> Knowability:
    """What a finding licenses us to say about predictability. This module NEVER returns PREDICTABLE: a rank shift in a precursor is evidence that
    a question is worth asking, not proof that the move could have been called (that needs decision value on fresh data downstream)."""
    if c.status == CandidateStatus.CANDIDATE:
        return Knowability.WEAKLY_PREDICTABLE if abs(c.effect) < 0.05 else Knowability.POTENTIALLY_PREDICTABLE
    return Knowability.UNKNOWN


def family_representatives(state: LabState) -> dict[str, str]:
    """Within one comparison and one feature family, the strongest finding represents the others (they are the same idea at neighbouring lags or
    windows). Returns {candidate_id: representative_id}; a representative maps to itself."""
    best: dict[tuple[str, str, str], Candidate] = {}
    for c in state.candidates.values():
        k = (c.key, c.family, str(c.status == CandidateStatus.CANDIDATE))
        if k not in best or abs(c.t) > abs(best[k].t) or (abs(c.t) == abs(best[k].t) and c.candidate_id < best[k].candidate_id):
            best[k] = c
    return {c.candidate_id: best[(c.key, c.family, str(c.status == CandidateStatus.CANDIDATE))].candidate_id for c in state.candidates.values()}


def unknown_map(state: LabState) -> pd.DataFrame:
    """Per comparison cell: how many feature tests were eligible, how many were tracked and how many became candidates. A cell with eligible
    tests and no candidate is an explicit 'no precursor found' - the honest answer for that episode type - and is listed, not hidden."""
    rows = []
    by_key: dict[str, list[Candidate]] = {}
    for c in state.candidates.values():
        by_key.setdefault(c.key, []).append(c)
    for key, cell in sorted(state.evidence.cells.items()):
        ids, T = cell.stacked()
        if len(ids) == 0:
            continue
        real = tensor_stats(T[:, :, 0, :])
        elig = int(((real["n1"] >= state.rules.min_n1) & (real["n0"] >= state.rules.min_n0) & np.isfinite(real["t"])).sum())
        cs = by_key.get(key, [])
        n_c = sum(1 for c in cs if c.status == CandidateStatus.CANDIDATE)
        lens, typ, comp, anchor = split_key(key)
        rows.append({"lens": lens, "type": typ, "comparison": comp, "anchor": anchor, "eligible": elig, "tracked": len(cs), "candidates": n_c,
                     "verdict": "UNKNOWN: no precursor found" if elig and n_c == 0 else "candidate(s)" if n_c else "too thin"})
    return pd.DataFrame(rows, columns=["lens", "type", "comparison", "anchor", "eligible", "tracked", "candidates", "verdict"])


def precursor_catalog(registry: PrecursorRegistry) -> pd.DataFrame:
    """The registry as a table (name, family, lags, first line of its docstring)."""
    return pd.DataFrame([{"name": s.name, "family": s.family, "lags": list(s.lags), "columns": len(s.columns()),
                          "doc": (s.doc.splitlines()[0] if s.doc else "")} for s in registry.specs.values()])


def replay_guard(state: LabState, replay_windows: Sequence[tuple[Any, Any]]) -> dict[str, list[int]]:
    """Which finished units are filed under years that a disguised replay is currently using. Their candidates are withheld by `release`; this
    lists the units so a scheduler can also avoid learning anything new from those years while the replay runs."""
    years: set[int] = set()
    for a, b in replay_windows:
        years |= set(range(as_date(a).year, as_date(b).year + 1))
    hit: dict[str, list[int]] = {}
    for uid in state.book.records:
        u = Unit.parse(uid)
        if u.year in years:
            hit.setdefault(u.lens, []).append(u.year)
    return {k: sorted(set(v)) for k, v in hit.items()}


def release(state: LabState, now, replay_windows: Sequence[tuple[Any, Any]] = (), created_real: str = "") -> tuple[list[MaturedRecord], dict[str, str]]:
    """Public release path (sole road to the trader): candidates that survived every control, matured strictly before `now`, not built from any
    year being replayed in disguise. Wraps release_candidates with this state's provenance."""
    return release_candidates(state.candidates, now, replay_windows, created_real or str(as_date(now)), current_code_hash(), state.evidence.digest(),
                              state.cfg.digest(), state.cfg.seed)


# ------------------------------------------------------------------------------------------------------------------ the always-on job
@dataclasses.dataclass
class TickReport:
    tick: int
    now: str
    done: list[str]
    waiting: list[str]
    idle: bool
    new_years: int
    fraction_done: float
    candidates: int
    ledger_looks: int
    evaluated: bool
    widened: list[str] = dataclasses.field(default_factory=list)
    exhausted: bool = False
    seconds: float = 0.0


class AlwaysOn:
    """The job the wave-2 research loop schedules through compute_manager: every tick it (1) learns of any newly available years, (2) runs up to
    `units_per_tick` unfinished units, least-covered first, committing after each, and (3) evaluates on the doubling schedule. A tick that has
    nothing to do says so (idle=True) and costs almost nothing, so the job can simply be called forever; new years, new sessions or a wider
    precursor registry make it busy again. It never redoes finished work and never runs a unit on a year that is not final."""

    def __init__(self, state: LabState, loader: Loader, store: LabStore | None = None, context_fn: ContextFn | None = None,
                 years_fn: Callable[[Any], Iterable[int]] | None = None, units_per_tick: int = 1, budget_s: float | None = None,
                 grouped: bool = False):
        if units_per_tick < 1:
            raise LabError("units_per_tick must be >= 1")
        self.state, self.loader, self.store, self.context_fn = state, loader, store, context_fn
        self.years_fn, self.units_per_tick = years_fn, int(units_per_tick)
        self.budget_s, self.grouped = budget_s, bool(grouped)
        self.ticks = 0
        self.idle_ticks = 0

    def tick(self, now) -> TickReport:
        self.ticks += 1
        new = self.state.book.extend_years(self.years_fn(now)) if self.years_fn is not None else 0
        rep = step(self.state, now, self.loader, self.units_per_tick, self.store, self.context_fn, budget_s=self.budget_s, grouped=self.grouped)
        idle = not rep.done
        self.idle_ticks = self.idle_ticks + 1 if idle else 0
        return TickReport(self.ticks, str(as_date(now)), rep.done, rep.waiting, idle, new, rep.coverage["fraction_done"], rep.candidates,
                          self.state.ledger.total_trials, rep.evaluation is not None, rep.widened, rep.exhausted, rep.seconds)

    def run(self, now_fn: Callable[[], Any], max_ticks: int | None = None, stop: Callable[[], bool] = lambda: False,
            stop_after_idle: int | None = None) -> Iterator[TickReport]:
        """Yield one TickReport per tick until `stop()`, `max_ticks`, or `stop_after_idle` consecutive idle ticks."""
        n = 0
        while (max_ticks is None or n < max_ticks) and not stop():
            r = self.tick(now_fn())
            n += 1
            yield r
            if stop_after_idle is not None and self.idle_ticks >= stop_after_idle:
                return

    def coverage(self) -> dict[str, Any]:
        return self.state.book.report(self.state.registry.columns())


# ------------------------------------------------------------------------------------------------------------------ the 'hundreds a day' ledger
def universe_counts(state: LabState, year: int) -> pd.DataFrame:
    """Whole-universe mover counts per session for one year: the slices' counts added up (each name lives in exactly one slice)."""
    parts = [d for uid, d in sorted(state.daily.items()) if Unit.parse(uid).year == year]
    return EPI.merge_counts(parts)


def hundreds_by_year(state: LabState, minimum: int = 100, measure: str = "c2c") -> pd.DataFrame:
    """For each finished year: median daily movers per direction in the 5-10% band, the share of sessions with at least `minimum` of them, and
    the same for the >10% band. The owner's premise (hundreds a day) is a number here, per year, on the universe actually swept."""
    rows = []
    for y in state.book.years:
        c = universe_counts(state, y)
        if len(c) == 0:
            continue
        r = EPI.hundreds_report(c, minimum, measure)
        rows.append({"year": y, "sessions": r["days"], "universe_median": r["universe_median"],
                     "median_up_5_10": r["median"].get(f"{measure}_up_5_10", float("nan")), "median_dn_5_10": r["median"].get(f"{measure}_dn_5_10", float("nan")),
                     "share_days_both_sides_at_least": r["both_sides_share_days_at_least"],
                     "median_gt10": float(c[[k for k in c.columns if k.startswith(measure + "_") and k.endswith("gt10")]].sum(axis=1).median())})
    return pd.DataFrame(rows, columns=["year", "sessions", "universe_median", "median_up_5_10", "median_dn_5_10", "share_days_both_sides_at_least", "median_gt10"])


# ------------------------------------------------------------------------------------------------------------------ label ledger by year
def labels_by_year(state: LabState, lens: str = "c2c") -> dict[int, dict[str, float]]:
    """The units' outcome-label ledgers (episode_paths.label_counts) added up per year for one lens (all universe slices together)."""
    out: dict[int, dict[str, float]] = {}
    for uid, cnt in sorted(state.labels.items()):
        u = Unit.parse(uid)
        if u.lens == lens:
            out[u.year] = EP.merge_label_counts([out.get(u.year, {}), cnt])
    return out


def label_year_table(state: LabState, lens: str = "c2c", horizon: int = 1, band: str | None = "5_10") -> pd.DataFrame:
    """Rows = year, columns = count of each path class at `horizon` (episode types of `band` pooled over direction; None = every band), plus
    the collapse / acceleration flag counts and the observable total. Counts, not shares, so a thin year shows as thin."""
    rows = []
    for y, cnt in sorted(labels_by_year(state, lens).items()):
        r: dict[str, float] = {"year": y}
        for k, v in cnt.items():
            typ, what, val = k.split("|", 2)
            if band is not None and not typ.endswith(":" + band):
                continue
            if what == f"cls_{horizon}":
                r[val] = r.get(val, 0.0) + v
            elif what == "n" and val == f"ok_{horizon}":
                r["n_ok"] = r.get("n_ok", 0.0) + v
            elif what in (f"coll_{horizon}", f"accel_{horizon}"):
                r[what] = r.get(what, 0.0) + v
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).fillna(0.0).set_index("year")
    order = ["n_ok"] + [c for c in EP.CLASS_ORDER if c in t.columns] + [c for c in (f"coll_{horizon}", f"accel_{horizon}") if c in t.columns]
    return t[order].astype(np.int64)


# ------------------------------------------------------------------------------------------------------------------ questions into the research brain
def question_events(state: LabState, now, statuses: Sequence[str] = ("CANDIDATE", "INSUFFICIENT_EVIDENCE"), min_abs_t: float = 3.0,
                    max_n: int = 25, include_decoys: bool = False) -> list:
    """Tracked findings as engine.research.questions events (source 'new_discovery'), strongest first. Only findings whose evidence matured
    strictly before `now`; decoy features are left out unless asked for (a decoy that reaches here is itself a false-positive alarm)."""
    from engine.research import questions as Q
    evs = []
    for c in sorted(state.candidates.values(), key=lambda c: (-abs(c.t), c.candidate_id)):
        if len(evs) >= max_n:
            break
        if str(c.status) not in statuses or abs(c.t) < min_abs_t or not c.matured_at:
            continue
        if c.feature.startswith(DECOY_PREFIX) and not include_decoys:
            continue
        if as_date(c.matured_at) >= as_date(now):
            continue
        evs.append(Q.QuestionEvent("new_discovery", c.candidate_id, c.matured_at, float(min(1.0, abs(c.t) / 6.0)),
                                   stake=0.7 if c.status == CandidateStatus.CANDIDATE else 0.4, problem=problem_of(c.comparison),
                                   n_obs=int(c.clusters), p_isolate=0.6, p_actionable=0.4 if c.anchor == "pre" else 0.5, detail=c.text))
    return evs


def raise_questions(state: LabState, now, ledger=None, **kw) -> tuple[Any, Any]:
    """Push tracked findings through the research brain's question generator (engine.research.questions.generate): each becomes a question
    object with hypotheses (including chance), a test plan, success/failure criteria and a priority, recorded in the QuestionLedger.
    Returns (GenerationReport, ledger)."""
    from engine.research import questions as Q
    led = ledger if ledger is not None else Q.QuestionLedger()
    rep = Q.generate(question_events(state, now, **kw), now, led)
    return rep, led


# ------------------------------------------------------------------------------------------------------------------ held-out years
def _restricted_store(store: EvidenceStore, keep: Callable[[int], bool], last_ordinal: int) -> EvidenceStore:
    out = EvidenceStore(store.n_perm)
    for key, cell in store.cells.items():
        ids, T = cell.stacked()
        sel = [i for i, c in enumerate(ids) if keep(int(c))]
        if not sel:
            continue
        c2 = Cell(cell.V)
        c2.features = list(cell.features)
        c2.clusters = {int(ids[i]): T[i].copy() for i in sel}
        c2.last_ordinal, c2.units = min(cell.last_ordinal, last_ordinal), cell.units
        out.cells[key] = c2
    return out


def holdout_check(state: LabState, split_year: int, now, statuses: Sequence[str] = ("CANDIDATE", "INSUFFICIENT_EVIDENCE", "HYPOTHESIS",
                  "FAILED_WALK_FORWARD", "FAILED_ERA")) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A true out-of-sample check. Discovery sees only clusters before `split_year` (every test it makes is entered in the state's ledger,
    because a look is a look); each tracked finding is then measured on the untouched clusters from `split_year` on. A finding is CONFIRMED
    when the held-out effect keeps its sign with signed t >= the walk-forward confirm bar. Decoys go through the same path, so the decoy
    confirmation rate is the chance rate of this very check."""
    import datetime as _dt
    m = state.cfg.cluster_months
    disc = _restricted_store(state.evidence, lambda c: cluster_year(c, m) < split_year, _dt.date(split_year - 1, 12, 31).toordinal())
    cands, rep = evaluate_evidence(disc, state.ledger, state.rules, state.cfg, state.registry, now, f"holdout{split_year}-{disc.digest()[:10]}", -1)
    rows = []
    for cid, c in cands.items():
        if str(c.status) not in statuses:
            continue
        cell = state.evidence.cells[c.key]
        ids, T = cell.stacked()
        sel = np.array([cluster_year(int(i), m) >= split_year for i in ids])
        fj = cell.features.index(c.feature)
        h = _one(T, fj, sel) if sel.sum() >= 2 else {"effect": float("nan"), "t": float("nan"), "clusters": float(sel.sum()), "n1": 0.0}
        s = 1.0 if c.effect >= 0 else -1.0
        kept = bool(math.isfinite(h["effect"]) and s * h["effect"] > 0)
        rows.append({"candidate_id": cid, "status_discovery": str(c.status), "key": c.key, "feature": c.feature, "family": c.family,
                     "decoy": c.feature.startswith(DECOY_PREFIX), "effect_disc": c.effect, "t_disc": c.t, "q_disc": c.q, "wf": f"{c.wf_passed}/{c.wf_tested}",
                     "effect_hold": h["effect"], "t_hold": h["t"], "clusters_hold": int(h["clusters"]), "n1_hold": int(h["n1"]),
                     "sign_kept": kept, "confirmed": bool(kept and math.isfinite(h["t"]) and s * h["t"] >= state.rules.wf_confirm_t),
                     "text": c.text})
    cols = ["candidate_id", "status_discovery", "key", "feature", "family", "decoy", "effect_disc", "t_disc", "q_disc", "wf", "effect_hold", "t_hold",
            "clusters_hold", "n1_hold", "sign_kept", "confirmed", "text"]
    t = pd.DataFrame(rows, columns=cols)
    if len(t):
        t = t.assign(a=t["t_disc"].abs()).sort_values("a", ascending=False).drop(columns="a").reset_index(drop=True)
    real, dec = t[~t["decoy"]] if len(t) else t, t[t["decoy"]] if len(t) else t
    summary = {"split_year": int(split_year), "discovery_tests": rep.n_eligible, "tracked": int(len(t)),
               "by_status": dict(pd.Series([str(x) for x in t["status_discovery"]]).value_counts()) if len(t) else {},
               "real_sign_kept": float(real["sign_kept"].mean()) if len(real) else float("nan"),
               "real_confirmed": float(real["confirmed"].mean()) if len(real) else float("nan"),
               "candidates_confirmed": int(real.loc[real["status_discovery"] == "CANDIDATE", "confirmed"].sum()) if len(real) else 0,
               "candidates": int((real["status_discovery"] == "CANDIDATE").sum()) if len(real) else 0,
               "decoy_tracked": int(len(dec)), "decoy_confirmed": int(dec["confirmed"].sum()) if len(dec) else 0,
               "ledger_looks_after": state.ledger.total_trials}
    return t, summary


# ------------------------------------------------------------------------------------------------------------------ truncation audit (F09 method)
def truncation_audit(bars: Mapping[str, pd.DataFrame], registry: PrecursorRegistry, ecfg: EpisodeConfig = EpisodeConfig(),
                     pcfg: EP.PathConfig = EP.PathConfig(), n_cuts: int = 4, seed: int = 0, cuts: Sequence[int] | None = None,
                     sector_codes: np.ndarray | None = None) -> dict[str, Any]:
    """Point-in-time audit of the whole precursor path on real or synthetic bars (the F09 truncation method, applied at episode level): for
    sessions t that contain episodes, rebuild the grid from bars TRUNCATED after t and require that (1) the episodes of day t are identical and
    (2) every precursor read at the 'pre' anchor (close of t-1) and the 'post' anchor (close of t) is identical to the full-data value. Also
    checks that every path label matures strictly after its entry day (the future is used only as the outcome). Clean = zero mismatches."""
    full = build_grid(bars, ecfg)
    T = full.shape[0]
    eps = EPI.episode_frame(full)
    ctx = Ctx(full, sector_codes)
    names = sorted(registry.specs)
    pf, _ = ranked_panels(ctx, registry, names)
    cols = [c for c in registry.columns() if c.split("@")[0] in pf]
    lo_ok = max(ecfg.warm, 2)
    days = sorted(set(int(t) for t in eps["ti"]) & set(range(lo_ok, T - 1))) if len(eps) else []
    if cuts is None:
        rng = np.random.default_rng(seed)
        cuts = sorted(rng.choice(days, min(n_cuts, len(days)), replace=False).tolist()) if days else []
    mism: dict[str, int] = {}
    compared = checked = 0
    ep_ok = True
    for t in cuts:
        trunc = {k: v.iloc[:t + 1] for k, v in bars.items()}
        g2 = build_grid(trunc, ecfg)
        e_full = eps[eps["ti"] == t]
        e_tr = EPI.episode_frame(g2, t, t + 1)
        if EPI.episode_digest(e_full) != EPI.episode_digest(e_tr):
            ep_ok = False
        p2, _ = ranked_panels(Ctx(g2, sector_codes), registry, names)
        ti, nj = e_full["ti"].to_numpy(), e_full["nj"].to_numpy()
        checked += len(ti)
        for anchor in ANCHORS:
            a = gather_features(pf, cols, ti, nj, anchor)
            b = gather_features(p2, cols, ti, nj, anchor)
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-6)
            compared += same.size
            for j in np.flatnonzero(~same.all(axis=0)):
                mism[f"{cols[j]}@{anchor}"] = mism.get(f"{cols[j]}@{anchor}", 0) + int((~same[:, j]).sum())
    labels_after = True
    if len(eps):
        lab = EP.label_paths(full, eps, pcfg)
        m = pd.to_datetime(lab["matured_at"])
        ok = m.notna()
        labels_after = bool((m[ok].to_numpy() > pd.to_datetime(eps.loc[ok.to_numpy(), "date"]).to_numpy()).all())
    return {"cuts": [str(full.dates[t].date()) for t in cuts], "episodes_checked": int(checked), "cells_compared": int(compared),
            "mismatches": mism, "episodes_identical": ep_ok, "labels_mature_after_entry": labels_after,
            "clean": bool(not mism and ep_ok and labels_after), "features": len(cols)}
