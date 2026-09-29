"""Mover-episode detector, streaming driver, coverage book and synthetic worlds (canon C67; contract C66 sections 4, 5, 21-23).
IMPLEMENTED - NOT VALIDATED (C63: unit tests on planted and null worlds only; no real-data run).

C67, the owner's words: every simulated day look at HUNDREDS of stocks that moved 5-10% (and more than 10% as its own band) and at what they
did next, and study them continuously across tons of stocks for even the smallest pattern that could have been predicted beforehand.

This module owns the first half of that machine:
  * measures      per name-day, from daily OHLC only: close-to-close return, open-to-close, intraday range (high-low)/prev close, gap (open vs
                  prev close), and where the close sits in the range. Each is banded into 5-10% and >10%, up and down (range: by body sign).
  * Grid          one (sessions x names) block of arrays: the measures, the name's OWN trailing volatility (through the PREVIOUS close, so the
                  move never inflates its own yardstick), a volatility cohort per date, liquidity and data-glitch flags. Halted or missing bars
                  give NaN measures - a two-day return is never read as a one-day move.
  * episodes      only the exception rows are kept (mapping rule 27: ~38M name-days will not fit in RAM). Per-day counts per measure and band
                  are recorded for every session, because "hundreds a day" is a claim that has to be checked, not assumed.
  * EpisodeStream year-by-year / chunk-by-chunk driver. It keeps a bounded tail of sessions so a rolling window never sees a truncated past and
                  a name-day is emitted only once its longest look-forward horizon exists. Output is identical to the one-shot in-memory run.
  * CoverageBook  what has been swept (year x universe slice x lens x feature set), so an always-on sweep never redoes finished work and
                  always moves to the least-covered area; atomic, hash-verified persistence so a killed job costs one unit, not the sweep.
  * loaders       read the daily caches through engine.data's file layout (pre-2000 and modern), restricted to a hash slice of the universe.
  * synthetic_bars  a seeded OHLCV world with planted precursors and planted follow-through classes, for the tests.
Everything here is MATURED_RESEARCH_STATE by construction (an episode is defined by a move that has already happened). Nothing reaches the blind
trader except through MaturedRecord.gate(now); see precursors.py for the release path."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.checkpoints import _atomic_text
from engine.research.core import FirewallBreach, Namespace, _StrEnum, as_date, stable_hash

NAMESPACE = Namespace.MATURED_RESEARCH
PRICE_KEYS = ("Open", "High", "Low", "Close")
FIELD_KEYS = PRICE_KEYS + ("Volume",)


class EpisodeError(ValueError):
    """Malformed bars, a bad configuration, or a stream fed out of order. Never swallowed: a silent repair here would corrupt every count."""


class Measure(_StrEnum):
    C2C = "c2c"             # close / previous close - 1
    O2C = "o2c"             # close / open - 1
    RANGE = "rng"           # (high - low) / previous close (unsigned; side taken from the candle body)
    GAP = "gap"             # open / previous close - 1


SIGNED = (Measure.C2C, Measure.O2C, Measure.GAP)
BAND_LABEL = {1: "5_10", 2: "gt10"}


# ------------------------------------------------------------------------------------------------------------------ configuration
@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    """All thresholds are parameters (C67); nothing is a magic number inside the detector. `lo`/`hi` bound the two bands; the trailing-volatility
    yardstick and every liquidity gate use data through the previous close only."""
    lo: float = 0.05
    hi: float = 0.10
    max_move: float = 0.50                  # a c2c or gap beyond this is a split/data glitch until proven otherwise: counted, never an episode
    vol_window: int = 60
    vol_min: int = 40
    sigma_floor: float = 0.004              # keeps a flat stock's z-score finite
    dv_window: int = 20
    min_price: float = 1.0
    min_dollar_vol: float = 250_000.0       # average dollar volume over the prior dv_window sessions
    n_cohorts: int = 5                      # volatility cohorts per date (used for matched controls)
    horizons: tuple[int, ...] = (1, 2, 3, 5)
    warm: int = 90                          # sessions of past kept so every trailing window is complete
    measures: tuple[str, ...] = ("c2c", "o2c", "rng", "gap")

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 < self.lo < self.hi < 1.0):
            errs.append("need 0 < lo < hi < 1")
        if self.max_move <= self.hi:
            errs.append("max_move must exceed hi or every >10% band is empty")
        if self.vol_min < 5 or self.vol_min > self.vol_window:
            errs.append("vol_min must lie in [5, vol_window]")
        if not self.horizons or any(int(h) < 1 for h in self.horizons) or list(self.horizons) != sorted(set(self.horizons)):
            errs.append("horizons must be strictly increasing positive integers")
        if self.n_cohorts < 1:
            errs.append("n_cohorts must be >= 1")
        if self.warm < self.vol_window + 2:
            errs.append("warm must cover vol_window + 2 sessions")
        bad = [m for m in self.measures if m not in {x.value for x in Measure}]
        if bad or not self.measures:
            errs.append(f"unknown or empty measures {bad}")
        return errs

    def require_valid(self) -> "EpisodeConfig":
        errs = self.validate()
        if errs:
            raise EpisodeError("; ".join(errs))
        return self

    @property
    def max_horizon(self) -> int:
        return int(max(self.horizons))

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


# ------------------------------------------------------------------------------------------------------------------ the grid
@dataclasses.dataclass
class Grid:
    """One (sessions x names) block. Row t of every trailing quantity uses data through the close of t-1 unless the name says otherwise."""
    cfg: EpisodeConfig
    dates: pd.DatetimeIndex
    tickers: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray | None
    prev_c: np.ndarray
    c2c: np.ndarray
    o2c: np.ndarray
    rng: np.ndarray
    gap: np.ndarray
    loc: np.ndarray                          # (close - low) / (high - low); 0.5 when the bar is flat
    sigma: np.ndarray                        # std of c2c over the vol_window sessions ending at t-1, floored
    dv_prev: np.ndarray                      # mean dollar volume over the dv_window sessions ending at t-1 (NaN when no volume)
    vol_prev: np.ndarray                     # mean share volume over the same window
    valid: np.ndarray                        # OHLC finite and positive, previous close finite
    suspect: np.ndarray                      # a glitch-sized move or an impossible bar
    liquid: np.ndarray
    cohort: np.ndarray                       # int8 volatility cohort per date, -1 when sigma is unknown
    band: dict[str, np.ndarray]              # measure value -> int8 codes (signed for c2c/o2c/gap: +-1, +-2; range: 1, 2)

    @property
    def shape(self) -> tuple[int, int]:
        return self.C.shape

    def eligible(self) -> np.ndarray:
        """Name-days whose measures may count: real bar, no glitch, tradable by the liquidity gate."""
        return self.valid & ~self.suspect & self.liquid

    def body_sign(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.sign(self.C - self.O)

    def any_band(self) -> np.ndarray:
        m = np.zeros(self.shape, bool)
        for k in self.cfg.measures:
            m |= self.band[k] != 0
        return m


def _wide(bars: Mapping[str, pd.DataFrame], key: str) -> pd.DataFrame:
    if key not in bars:
        raise EpisodeError(f"bars missing {key}")
    return bars[key]


def check_bars(bars: Mapping[str, pd.DataFrame]) -> list[str]:
    """Problems that would silently corrupt every count (empty list = clean). Frames must be aligned wide (date x ticker) blocks."""
    bad = []
    for k in PRICE_KEYS:
        if k not in bars:
            bad.append(f"missing {k}")
    if bad:
        return bad
    ref = bars["Close"]
    for k in FIELD_KEYS:
        if k not in bars:
            continue
        f = bars[k]
        if f.shape != ref.shape or not f.index.equals(ref.index) or not f.columns.equals(ref.columns):
            bad.append(f"{k} not aligned with Close")
    if ref.index.has_duplicates:
        bad.append("duplicate sessions")
    if not ref.index.is_monotonic_increasing:
        bad.append("sessions not increasing")
    if ref.columns.has_duplicates:
        bad.append("duplicate tickers")
    return bad


def _band_codes(x: np.ndarray, ok: np.ndarray, cfg: EpisodeConfig, signed: bool) -> np.ndarray:
    a = np.abs(np.where(np.isfinite(x), x, 0.0))
    code = np.where(a >= cfg.hi, 2, np.where(a >= cfg.lo, 1, 0)).astype(np.int8)
    code = np.where(ok, code, 0).astype(np.int8)
    return (code * np.sign(np.where(np.isfinite(x), x, 0.0)).astype(np.int8)).astype(np.int8) if signed else code


def _empty_grid(cfg: EpisodeConfig, dates, tickers) -> Grid:
    z = np.zeros((len(dates), len(tickers)))
    b = np.zeros(z.shape, bool)
    return Grid(cfg, pd.DatetimeIndex(dates), np.asarray(tickers, dtype=object), z, z, z, z, None, z, z, z, z, z, z, z, z, z, b, b, b,
                np.full(z.shape, -1, np.int8), {m: np.zeros(z.shape, np.int8) for m in cfg.measures})


def build_grid(bars: Mapping[str, pd.DataFrame], cfg: EpisodeConfig = EpisodeConfig()) -> Grid:
    """Compute every measure and yardstick for a block of bars. Columns are sorted so that chunked and one-shot runs order rows identically."""
    cfg.require_valid()
    bad = check_bars(bars)
    if bad:
        raise EpisodeError("; ".join(bad))
    cols = sorted(bars["Close"].columns)
    fr = {k: bars[k][cols] for k in FIELD_KEYS if k in bars}
    if fr["Close"].shape[0] == 0 or len(cols) == 0:
        return _empty_grid(cfg, fr["Close"].index, cols)
    O, H, L, C = (fr[k].to_numpy(np.float64) for k in PRICE_KEYS)
    V = fr["Volume"].to_numpy(np.float64) if "Volume" in fr else None
    prev_c = np.full_like(C, np.nan)
    prev_c[1:] = C[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (O > 0) & (H > 0) & (L > 0) & (C > 0) & np.isfinite(O + H + L + C)
        valid = pos & np.isfinite(prev_c) & (prev_c > 0)
        c2c = np.where(valid, C / prev_c - 1.0, np.nan)
        o2c = np.where(valid, C / O - 1.0, np.nan)
        gap = np.where(valid, O / prev_c - 1.0, np.nan)
        rng = np.where(valid, (H - L) / prev_c, np.nan)
        span = H - L
        loc = np.where(valid, np.where(span > 0, (C - L) / np.where(span > 0, span, 1.0), 0.5), np.nan)
        broken = valid & ((H < L - 1e-9) | (H < np.maximum(O, C) - 1e-9) | (L > np.minimum(O, C) + 1e-9))
        suspect = valid & (broken | (np.abs(c2c) > cfg.max_move) | (np.abs(gap) > cfg.max_move) | (rng > 2 * cfg.max_move)
                          | split_like_mask(c2c, o2c, gap))
        sig = pd.DataFrame(c2c).rolling(cfg.vol_window, min_periods=cfg.vol_min).std().to_numpy()
        sigma = np.full_like(sig, np.nan)
        sigma[1:] = sig[:-1]
        sigma = np.where(np.isfinite(sigma), np.maximum(sigma, cfg.sigma_floor), np.nan)
    if V is not None:
        dvp = pd.DataFrame(C * V).rolling(cfg.dv_window, min_periods=max(5, cfg.dv_window // 2)).mean().to_numpy()
        vlp = pd.DataFrame(V).rolling(cfg.dv_window, min_periods=max(5, cfg.dv_window // 2)).mean().to_numpy()
        dv_prev, vol_prev = np.full_like(dvp, np.nan), np.full_like(vlp, np.nan)
        dv_prev[1:], vol_prev[1:] = dvp[:-1], vlp[:-1]
        liquid = (prev_c >= cfg.min_price) & (dv_prev >= cfg.min_dollar_vol)
    else:
        dv_prev = vol_prev = np.full_like(C, np.nan)
        liquid = prev_c >= cfg.min_price
    liquid = np.where(np.isfinite(prev_c), liquid, False)
    cohort = np.full(C.shape, -1, np.int8)
    if cfg.n_cohorts > 1:
        pct = pd.DataFrame(sigma).rank(axis=1, pct=True, method="first").to_numpy()
        cohort = np.where(np.isfinite(pct), np.minimum((np.nan_to_num(pct) * cfg.n_cohorts).astype(np.int64), cfg.n_cohorts - 1), -1).astype(np.int8)
    else:
        cohort = np.where(np.isfinite(sigma), 0, -1).astype(np.int8)
    ok = valid & ~suspect & liquid
    src = {"c2c": c2c, "o2c": o2c, "gap": gap, "rng": rng}
    band = {m: _band_codes(src[m], ok, cfg, m != Measure.RANGE.value) for m in cfg.measures}
    return Grid(cfg, fr["Close"].index, np.asarray(cols, dtype=object), O, H, L, C, V, prev_c, c2c, o2c, rng, gap, loc, sigma, dv_prev,
                vol_prev, valid, suspect, liquid, cohort, band)


# ------------------------------------------------------------------------------------------------------------------ episodes and counts
EPISODE_COLUMNS = ("date", "ticker", "c2c", "o2c", "rng", "gap", "loc", "b_c2c", "b_o2c", "b_rng", "b_gap", "body", "side", "sigma",
                   "vol_z", "cohort", "prev_c", "dv_prev", "vol_ratio", "gap_share", "ti", "nj")


def episode_frame(grid: Grid, lo: int = 0, hi: int | None = None) -> pd.DataFrame:
    """Every eligible banded name-day in rows [lo, hi) of the grid. `ti`/`nj` are the row/column of the grid (block-local, for gathering
    forward windows and precursor panels); the identity of a row is (date, ticker)."""
    T = grid.shape[0]
    hi = T if hi is None else hi
    if not (0 <= lo <= hi <= T):
        raise EpisodeError(f"row range [{lo},{hi}) outside a grid of {T} sessions")
    cfg = grid.cfg
    m = np.zeros(grid.shape, bool)
    m[lo:hi] = grid.any_band()[lo:hi] & grid.eligible()[lo:hi]
    ti, nj = np.nonzero(m)
    if len(ti) == 0:
        return pd.DataFrame({c: pd.Series(dtype=object if c == "ticker" else "float64") for c in EPISODE_COLUMNS}).astype(
            {"date": "datetime64[ns]", "ti": "int64", "nj": "int64"})
    c2c, o2c = grid.c2c[ti, nj], grid.o2c[ti, nj]
    body = np.sign(grid.C[ti, nj] - grid.O[ti, nj])
    side = np.sign(np.where(c2c != 0, c2c, np.where(o2c != 0, o2c, grid.gap[ti, nj])))
    sig = grid.sigma[ti, nj]
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = grid.V[ti, nj] / grid.vol_prev[ti, nj] if grid.V is not None else np.full(len(ti), np.nan)
        gap_share = np.clip(grid.gap[ti, nj] / np.where(np.abs(c2c) > 1e-9, c2c, np.nan), -2.0, 2.0)
    out = {"date": grid.dates[ti], "ticker": grid.tickers[nj], "c2c": c2c, "o2c": o2c, "rng": grid.rng[ti, nj], "gap": grid.gap[ti, nj],
           "loc": grid.loc[ti, nj]}
    for k in ("c2c", "o2c", "rng", "gap"):
        out["b_" + k] = grid.band[k][ti, nj] if k in grid.band else np.zeros(len(ti), np.int8)
    out.update({"body": body.astype(np.int8), "side": side.astype(np.int8), "sigma": sig, "vol_z": np.abs(c2c) / sig,
                "cohort": grid.cohort[ti, nj], "prev_c": grid.prev_c[ti, nj], "dv_prev": grid.dv_prev[ti, nj], "vol_ratio": vol_ratio,
                "gap_share": gap_share, "ti": ti.astype(np.int64), "nj": nj.astype(np.int64)})
    return pd.DataFrame(out)


def lens_band_side(eps: pd.DataFrame, lens: str) -> tuple[np.ndarray, np.ndarray]:
    """(absolute band 0/1/2, side -1/0/+1) of each episode under one lens. Signed measures take the side from the measure itself, the range
    lens from the candle body: a 12% high-low day that closed down is a DOWN range episode."""
    if lens not in {m.value for m in Measure}:
        raise EpisodeError(f"unknown lens {lens!r}")
    b = eps["b_" + lens].to_numpy()
    if lens == Measure.RANGE.value:
        return np.abs(b).astype(np.int8), eps["body"].to_numpy().astype(np.int8)
    return np.abs(b).astype(np.int8), np.sign(b).astype(np.int8)


def type_name(band: int, side: int, lens: str) -> str:
    return f"{lens}:{'up' if side > 0 else 'dn' if side < 0 else 'flat'}:{BAND_LABEL.get(int(band), 'none')}"


def day_counts(grid: Grid, lo: int = 0, hi: int | None = None) -> pd.DataFrame:
    """Per session: how many names moved into each band, per measure and direction, plus the size of the tradable universe that day and
    how many bars were discarded as glitches or as illiquid. This is the ledger behind the 'hundreds a day' claim."""
    T = grid.shape[0]
    hi = T if hi is None else hi
    sl = slice(lo, hi)
    ok = grid.eligible()[sl]
    cols: dict[str, np.ndarray] = {"n_names": (grid.valid & grid.liquid)[sl].sum(1), "n_suspect": grid.suspect[sl].sum(1),
                                   "n_illiquid": (grid.valid & ~grid.liquid)[sl].sum(1)}
    body = grid.body_sign()[sl]
    for k in grid.cfg.measures:
        b = grid.band[k][sl]
        for code in (1, 2):
            lab = BAND_LABEL[code]
            if k == Measure.RANGE.value:
                cols[f"{k}_up_{lab}"] = ((b == code) & ok & (body > 0)).sum(1)
                cols[f"{k}_dn_{lab}"] = ((b == code) & ok & (body < 0)).sum(1)
            else:
                cols[f"{k}_up_{lab}"] = ((b == code) & ok).sum(1)
                cols[f"{k}_dn_{lab}"] = ((b == -code) & ok).sum(1)
    return pd.DataFrame(cols, index=grid.dates[sl])


def hundreds_report(counts: pd.DataFrame, minimum: int = 100, measure: str = "c2c") -> dict[str, Any]:
    """Does the universe deliver 'hundreds of 5-10% movers a day'? Reports the share of days at or above `minimum` and the distribution, per
    direction. An honest answer to a small universe is 'no', and the caller sees the number rather than an assumption."""
    if len(counts) == 0:
        return {"days": 0, "share_days_at_least": {}, "median": {}, "universe_median": 0}
    out: dict[str, Any] = {"days": int(len(counts)), "minimum": int(minimum), "share_days_at_least": {}, "median": {}, "p90": {}}
    for col in [c for c in counts.columns if c.startswith(measure + "_") and c.endswith("5_10")]:
        out["share_days_at_least"][col] = float((counts[col] >= minimum).mean())
        out["median"][col] = float(counts[col].median())
        out["p90"][col] = float(counts[col].quantile(0.9))
    both = sum(counts[c] for c in counts.columns if c.startswith(measure + "_") and c.endswith("5_10"))
    out["both_sides_share_days_at_least"] = float((both >= minimum).mean())
    out["universe_median"] = float(counts["n_names"].median())
    return out


def band_totals(counts: pd.DataFrame) -> dict[str, int]:
    """Total episodes per (measure, direction, band) column of a counts table."""
    return {c: int(counts[c].sum()) for c in counts.columns if c.split("_")[0] in {m.value for m in Measure}}


# ------------------------------------------------------------------------------------------------------------------ streaming driver
@dataclasses.dataclass
class Block:
    """What the stream emits for one finalized run of sessions. `payload` is whatever the caller's `on_block` returned (paths, precursors...)."""
    lo_date: pd.Timestamp
    hi_date: pd.Timestamp
    episodes: pd.DataFrame
    counts: pd.DataFrame
    payload: Any = None


BlockFn = Callable[[Grid, pd.DataFrame, int, int], Any]


class EpisodeStream:
    """Feed bars in date order, in any chunking; get episodes only for sessions whose longest look-forward horizon is already present
    (`flush` finalizes the rest with whatever future exists, which the path labeller marks as unlabelled). The buffer keeps `warm` sessions
    of past, so a chunked run equals a one-shot run.  `on_block(grid, episodes, lo, hi)` is called with block-local rows; its return value is
    the Block payload. `extra_warm`/`extra_future` let downstream feature windows widen the buffer."""

    def __init__(self, cfg: EpisodeConfig = EpisodeConfig(), on_block: BlockFn | None = None, extra_warm: int = 0, extra_future: int = 0):
        self.cfg = cfg.require_valid()
        self.on_block = on_block
        self.warm = max(cfg.warm, int(extra_warm))
        self.future = cfg.max_horizon + int(extra_future)
        self.buf: dict[str, pd.DataFrame] | None = None
        self.emitted: pd.Timestamp | None = None
        self.fed: pd.Timestamp | None = None
        self.sessions_seen = 0

    def feed(self, chunk: Mapping[str, pd.DataFrame]) -> list[Block]:
        bad = check_bars(chunk)
        if bad:
            raise EpisodeError("chunk: " + "; ".join(bad))
        idx = chunk["Close"].index
        if len(idx) == 0:
            return []
        if self.fed is not None and idx[0] <= self.fed:
            raise EpisodeError(f"chunk starts {idx[0].date()} at/before the last fed session {self.fed.date()}")
        keys = [k for k in FIELD_KEYS if k in chunk]
        if self.buf is None:
            self.buf = {k: chunk[k].copy() for k in keys}
        else:
            if set(keys) != set(self.buf):
                raise EpisodeError("chunk fields differ from the stream's fields")
            cols = self.buf["Close"].columns.union(chunk["Close"].columns)
            self.buf = {k: pd.concat([self.buf[k].reindex(columns=cols), chunk[k].reindex(columns=cols)]) for k in keys}
        self.fed = idx[-1]
        self.sessions_seen += len(idx)
        return self._process(final=False)

    def flush(self) -> list[Block]:
        return self._process(final=True) if self.buf is not None else []

    def _process(self, final: bool) -> list[Block]:
        grid = build_grid(self.buf, self.cfg)
        T = grid.shape[0]
        lo = 0 if self.emitted is None else int(grid.dates.searchsorted(self.emitted, side="right"))
        hi = T if final else T - self.future
        if hi <= lo:
            return []
        eps = episode_frame(grid, lo, hi)
        blk = Block(grid.dates[lo], grid.dates[hi - 1], eps, day_counts(grid, lo, hi),
                    self.on_block(grid, eps, lo, hi) if self.on_block else None)
        self.emitted = grid.dates[hi - 1]
        keep_from = max(0, hi - self.warm)
        self.buf = {k: v.iloc[keep_from:] for k, v in self.buf.items()}
        return [blk]


def run_in_memory(bars: Mapping[str, pd.DataFrame], cfg: EpisodeConfig = EpisodeConfig(), on_block: BlockFn | None = None) -> Block:
    """The reference implementation the stream is tested against: one grid over everything, no buffer trimming."""
    grid = build_grid(bars, cfg)
    T = grid.shape[0]
    if T == 0:
        return Block(pd.NaT, pd.NaT, episode_frame(grid), day_counts(grid), None)
    eps = episode_frame(grid, 0, T)
    return Block(grid.dates[0], grid.dates[-1], eps, day_counts(grid, 0, T), on_block(grid, eps, 0, T) if on_block else None)


def concat_blocks(blocks: Sequence[Block]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(episodes, counts) of a list of blocks; empty list gives two empty frames."""
    if not blocks:
        return pd.DataFrame(columns=list(EPISODE_COLUMNS)), pd.DataFrame()
    return (pd.concat([b.episodes for b in blocks], ignore_index=True), pd.concat([b.counts for b in blocks]))


def episodes_by_date(eps: pd.DataFrame, lens: str = "c2c") -> pd.DataFrame:
    """Tidy per-day table: number of episodes per absolute band and side under one lens (input for coverage and reports)."""
    if len(eps) == 0:
        return pd.DataFrame(columns=["date", "band", "side", "n"])
    band, side = lens_band_side(eps, lens)
    t = pd.DataFrame({"date": eps["date"].to_numpy(), "band": band, "side": side})
    t = t[t["band"] > 0]
    return t.groupby(["date", "band", "side"]).size().rename("n").reset_index()


# ------------------------------------------------------------------------------------------------------------------ data loading
def slice_of(ticker: str, n_slices: int, salt: int = 0) -> int:
    """Deterministic universe slice of a ticker. A stable hash, never Python's per-process hash(), so a resumed sweep sees the same slices."""
    if n_slices < 1:
        raise EpisodeError("n_slices must be >= 1")
    return int(stable_hash({"t": str(ticker), "s": int(salt)}, 8), 16) % n_slices


def slice_tickers(tickers: Iterable[str], slice_id: int, n_slices: int, salt: int = 0) -> list[str]:
    if not (0 <= slice_id < n_slices):
        raise EpisodeError(f"slice {slice_id} outside 0..{n_slices - 1}")
    return sorted(t for t in tickers if slice_of(t, n_slices, salt) == slice_id)


def _cache_dir(cache_dir) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    from engine import config as K
    return Path(K.CACHE)


def cache_tickers(cache_dir=None, names: Sequence[str] = ("stocks_pre2000", "stocks")) -> list[str]:
    """Union of ticker columns across the daily caches, read from the parquet schema only (no data pages)."""
    import pyarrow.parquet as pq
    root, seen = _cache_dir(cache_dir), set()
    for n in names:
        p = root / f"{n}_close.parquet"
        if p.exists():
            seen |= {c for c in pq.ParquetFile(p).schema.names if not c.startswith("__")}
    return sorted(seen)


def _read_field(path: Path, tickers: Sequence[str] | None, start, end) -> pd.DataFrame:
    import pyarrow.parquet as pq
    cols = None
    if tickers is not None:
        have = set(pq.ParquetFile(path).schema.names)
        cols = [t for t in tickers if t in have]
        if not cols:
            return pd.DataFrame(index=pd.DatetimeIndex([]))
    df = pd.read_parquet(path, columns=cols)
    df.index = pd.DatetimeIndex(df.index)
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)].astype("float64")


def load_bars(start, end, tickers: Sequence[str] | None = None, cache_dir=None, with_volume: bool = True,
              names: Sequence[str] = ("stocks_pre2000", "stocks")) -> dict[str, pd.DataFrame]:
    """Wide OHLCV blocks for [start, end] from the cache files (`<name>_<field>.parquet`, the layout engine.data writes). The pre-2000 file
    supplies the early sessions and the modern file the rest; where both contain a session the modern file wins. NOTE: this panel is
    survivor-only (canon: price panel is survivor-biased) - the episode counts it yields overstate quality, never hazard."""
    root = _cache_dir(cache_dir)
    fields = FIELD_KEYS if with_volume else PRICE_KEYS
    out: dict[str, pd.DataFrame] = {}
    for f in fields:
        parts = []
        for n in names:
            p = root / f"{n}_{f.lower()}.parquet"
            if p.exists():
                d = _read_field(p, tickers, start, end)
                if len(d):
                    parts.append(d)
        if not parts:
            out[f] = pd.DataFrame(index=pd.DatetimeIndex([]))
            continue
        merged = parts[0]
        for d in parts[1:]:
            cols = merged.columns.union(d.columns)
            merged = pd.concat([merged.reindex(columns=cols), d.reindex(columns=cols)])
            merged = merged[~merged.index.duplicated(keep="last")]
        out[f] = merged.sort_index()
    cols = out["Close"].columns
    idx = out["Close"].index
    return {f: out[f].reindex(index=idx, columns=cols) for f in fields}


def year_window(year: int, cfg: EpisodeConfig, extra_warm: int = 0, extra_future: int = 0) -> tuple[str, str]:
    """Calendar span to load for a year: enough past for every trailing window and enough future for the longest horizon."""
    back = int((max(cfg.warm, extra_warm) + 10) * 1.6) + 10
    fwd = int((cfg.max_horizon + extra_future + 3) * 1.6) + 6
    return (str((pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=back)).date()),
            str((pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=fwd)).date()))


# ------------------------------------------------------------------------------------------------------------------ coverage book
@dataclasses.dataclass(frozen=True)
class Unit:
    """One piece of the sweep: a calendar year of one universe slice studied through one lens."""
    year: int
    slice_id: int
    lens: str

    @property
    def uid(self) -> str:
        return f"{self.year}|{self.slice_id}|{self.lens}"

    @staticmethod
    def parse(uid: str) -> "Unit":
        y, s, l = uid.split("|")
        return Unit(int(y), int(s), l)


@dataclasses.dataclass
class UnitRecord:
    """What a finished unit produced. `features_done` lets a later, wider feature set extend a unit without redoing its old features;
    `through` is the last session of the year that was analysed. A unit is only ever run on a year whose sessions are all final (complete)."""
    uid: str
    through: str
    complete: bool
    n_days: int
    n_names: int
    n_episodes: int
    n_suspect: int
    band_counts: dict[str, int]
    features_done: tuple[str, ...]
    config_digest: str
    code_hash: str = ""
    finished_real: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["features_done"] = sorted(self.features_done)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "UnitRecord":
        return UnitRecord(**{**d, "features_done": tuple(d["features_done"]), "band_counts": dict(d["band_counts"])})


class CoverageBook:
    """The sweep's memory of what is done. `pending` orders unfinished units least-covered first: by how much of that year is done, then how
    much of that lens, then that slice, then a stable hash (so the order never depends on set iteration or a random seed). A unit that is
    done for the requested features is never returned again."""

    def __init__(self, years: Sequence[int], n_slices: int = 4, lenses: Sequence[str] = ("c2c", "rng", "o2c", "gap")):
        if not years or n_slices < 1 or not lenses:
            raise EpisodeError("coverage needs years, slices and lenses")
        self.years = sorted({int(y) for y in years})
        self.n_slices = int(n_slices)
        self.lenses = tuple(lenses)
        self.records: dict[str, UnitRecord] = {}

    def all_units(self) -> list[Unit]:
        return [Unit(y, s, l) for y in self.years for s in range(self.n_slices) for l in self.lenses]

    def extend_years(self, years: Iterable[int]) -> int:
        new = sorted(set(int(y) for y in years) - set(self.years))
        self.years = sorted(set(self.years) | set(new))
        return len(new)

    def is_done(self, u: Unit, features: Sequence[str] = ()) -> bool:
        r = self.records.get(u.uid)
        return r is not None and set(features) <= set(r.features_done)

    def pending(self, features: Sequence[str] = (), limit: int | None = None, years_before: int | None = None) -> list[Unit]:
        """Unfinished units, least-covered first. `years_before` keeps only years strictly earlier than that year (a year whose data is
        still arriving is not yet a unit)."""
        done_y, done_l, done_s = {}, {}, {}
        for uid in self.records:
            u = Unit.parse(uid)
            if self.is_done(u, features):
                done_y[u.year] = done_y.get(u.year, 0) + 1
                done_l[u.lens] = done_l.get(u.lens, 0) + 1
                done_s[u.slice_id] = done_s.get(u.slice_id, 0) + 1
        todo = [u for u in self.all_units() if not self.is_done(u, features) and (years_before is None or u.year < years_before)]
        todo.sort(key=lambda u: (done_y.get(u.year, 0), done_l.get(u.lens, 0), done_s.get(u.slice_id, 0), stable_hash(u.uid, 8)))
        return todo[:limit] if limit else todo

    def mark(self, rec: UnitRecord) -> None:
        u = Unit.parse(rec.uid)
        if u.year not in self.years or not (0 <= u.slice_id < self.n_slices) or u.lens not in self.lenses:
            raise EpisodeError(f"unit {rec.uid} is outside this coverage book")
        old = self.records.get(rec.uid)
        if old is not None and set(rec.features_done) <= set(old.features_done):
            raise EpisodeError(f"unit {rec.uid} would be recorded again with no new feature")
        self.records[rec.uid] = rec

    def fraction_done(self, features: Sequence[str] = ()) -> float:
        us = self.all_units()
        return sum(self.is_done(u, features) for u in us) / len(us) if us else 1.0

    def report(self, features: Sequence[str] = (), thin: int = 200) -> dict[str, Any]:
        by_year: dict[int, dict[str, int]] = {}
        by_lens: dict[str, int] = {}
        band_by_year: dict[int, dict[str, int]] = {}
        for uid, r in self.records.items():
            u = Unit.parse(uid)
            by_year.setdefault(u.year, {"units": 0, "episodes": 0})
            by_year[u.year]["units"] += 1
            by_year[u.year]["episodes"] += r.n_episodes
            by_lens[u.lens] = by_lens.get(u.lens, 0) + 1
            if u.lens == "c2c":
                acc = band_by_year.setdefault(u.year, {})
                for k, v in r.band_counts.items():
                    acc[k] = acc.get(k, 0) + int(v)
        thin_cells = sorted((y, k, v) for y, d in band_by_year.items() for k, v in d.items() if k.startswith("c2c_") and v < thin)
        return {"units_total": len(self.all_units()), "units_done": sum(self.is_done(u, features) for u in self.all_units()),
                "fraction_done": self.fraction_done(features), "by_year": by_year, "by_lens": by_lens, "thin_cells": thin_cells,
                "years_untouched": [y for y in self.years if y not in by_year]}

    def digest(self) -> str:
        return stable_hash({"y": self.years, "s": self.n_slices, "l": self.lenses, "r": {k: v.to_dict() for k, v in sorted(self.records.items())}}, 16)

    def to_dict(self) -> dict:
        return {"years": self.years, "n_slices": self.n_slices, "lenses": list(self.lenses),
                "records": {k: v.to_dict() for k, v in sorted(self.records.items())}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CoverageBook":
        b = cls(d["years"], d["n_slices"], d["lenses"])
        b.records = {k: UnitRecord.from_dict(v) for k, v in d["records"].items()}
        return b

    def save(self, path) -> str:
        """Atomic write with an embedded content hash; `load` refuses a truncated or edited file."""
        body = self.to_dict()
        h = stable_hash(body, 20)
        _atomic_text(Path(path), json.dumps({"hash": h, "body": body}, sort_keys=True))
        return h

    @classmethod
    def load(cls, path) -> "CoverageBook":
        p = Path(path)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            body, h = d["body"], d["hash"]
        except (OSError, ValueError, KeyError) as e:
            raise EpisodeError(f"coverage file {p.name} unreadable: {e}") from e
        if stable_hash(body, 20) != h:
            raise EpisodeError(f"coverage file {p.name} fails its content hash")
        return cls.from_dict(body)


# ------------------------------------------------------------------------------------------------------------------ eras
ERA_EDGES = (1990, 2000, 2008, 2016)


def era_of(dates: pd.DatetimeIndex | np.ndarray, edges: Sequence[int] = ERA_EDGES) -> np.ndarray:
    """Era index of each date (0 = before the first edge). Precursors must hold in every era, not on average."""
    years = pd.DatetimeIndex(dates).year.to_numpy()
    return np.searchsorted(np.asarray(edges), years, side="right").astype(np.int8)


# ------------------------------------------------------------------------------------------------------------------ synthetic worlds
@dataclasses.dataclass(frozen=True)
class Plant:
    """What a synthetic world hides in its bars.
    kind 'none'                a null world: movers exist but nothing precedes them.
    kind 'volume_before'       a share `frac` of movers is preceded, `lag` sessions earlier, by a volume spike of `strength` times normal.
    kind 'class_separator'     movers continue the next day when their own move-day volume is high and reverse when it is not.
    kind 'compression_before'  a share `frac` of movers is preceded by two quiet sessions (very narrow ranges)."""
    kind: str = "none"
    frac: float = 0.6
    lag: int = 2
    strength: float = 8.0
    follow: float = 0.045


def synthetic_bars(n_names: int = 120, n_days: int = 260, seed: int = 0, start: str = "2019-01-02", plant: Plant = Plant(),
                   mover_rate: float = 0.012, sigma_range: tuple[float, float] = (0.010, 0.018)) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Seeded OHLCV world. Returns (bars, truth); truth lists every planted (date index, name index, side) so a test can score a detector against
    exact ground truth. Movers arrive at random (rate `mover_rate` per name-day, 5.5-9.5% both ways) whether or not anything is planted."""
    if n_names < 2 or n_days < 30:
        raise EpisodeError("synthetic world needs >= 2 names and >= 30 sessions")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"S{j:04d}" for j in range(n_names)]
    sig = rng.uniform(*sigma_range, n_names)
    r = rng.normal(0.0, 1.0, (n_days, n_names)) * sig
    r = np.clip(r, -0.045, 0.045)                      # ordinary noise never makes a mover by accident
    base_v = np.exp(rng.normal(13.8, 0.6, n_names))
    V = base_v * np.exp(rng.normal(0.0, 0.25, (n_days, n_names)))
    quiet = np.zeros((n_days, n_names), bool)
    last_event = np.full(n_names, -100)
    events = []
    for t in range(12, n_days - 8):
        for j in np.flatnonzero(rng.random(n_names) < mover_rate):
            if t - last_event[j] < 9:
                continue
            side = 1 if rng.random() < 0.5 else -1
            mag = rng.uniform(0.055, 0.095)
            last_event[j] = t
            r[t, j] = side * mag
            V[t, j] *= 2.0
            planted_pre = False
            high_vol = None
            if plant.kind == "volume_before" and rng.random() < plant.frac:
                V[t - plant.lag, j] *= plant.strength
                planted_pre = True
            elif plant.kind == "compression_before" and rng.random() < plant.frac:
                quiet[t - 2:t, j] = True
                planted_pre = True
            elif plant.kind == "class_separator":
                high_vol = bool(rng.random() < 0.5)
                V[t, j] *= 3.0 if high_vol else 0.6
                r[t + 1, j] = (side if high_vol else -side) * rng.uniform(0.7, 1.3) * plant.follow
                V[t + 1, j] *= 1.5
            events.append((t, int(j), side, planted_pre, high_vol))
    r = np.where(quiet, r * 0.15, r)
    C = 40.0 * np.exp(np.cumsum(r, axis=0) + rng.normal(0, 0.5, n_names))
    prev = np.vstack([C[:1] / (1 + r[:1]), C[:-1]])
    gap_w = rng.uniform(-0.15, 0.6, (n_days, n_names))
    gap_w = np.where(np.abs(r) > 0.05, rng.uniform(0.0, 0.6, (n_days, n_names)), gap_w * 0.3)
    O = prev * (1.0 + r * gap_w)
    wick = rng.exponential(0.5, (2, n_days, n_names)) * sig * np.where(quiet, 0.15, 1.0)
    wick = wick * (1.0 + 4.0 * np.abs(r))
    H = np.maximum(O, C) * (1.0 + wick[0])
    L = np.minimum(O, C) * (1.0 - np.minimum(wick[1], 0.5))
    bars = {"Open": O, "High": H, "Low": L, "Close": C, "Volume": V}
    frames = {k: pd.DataFrame(v, index=dates, columns=tickers) for k, v in bars.items()}
    truth = {"events": events, "dates": dates, "tickers": tickers, "plant": dataclasses.asdict(plant), "seed": seed}
    return frames, truth


def year_bars(bars: Mapping[str, pd.DataFrame], year: int, cfg: EpisodeConfig = EpisodeConfig(), extra_warm: int = 0,
              extra_future: int = 0) -> dict[str, pd.DataFrame]:
    """Slice an in-memory block down to a unit's loading window (used by tests that stand in for the cache)."""
    a, b = year_window(year, cfg, extra_warm, extra_future)
    return {k: v.loc[a:b] for k, v in bars.items()}


def merge_counts(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Add day-count tables from different universe slices (same dates, disjoint names) into whole-universe counts."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    return sum(p.reindex(parts[0].index).fillna(0) for p in parts[1:]) + parts[0] if len(parts) > 1 else parts[0]


def firewall_note() -> str:
    """One-line statement of the namespace rule, attached to every report this package writes."""
    return ("episodes and path labels are MATURED_RESEARCH_STATE (they use the future by definition); they reach the blind trader only "
            "through MaturedRecord.gate(now) with maturity strictly before now, never with a date, year or ticker attached")


def assert_no_future(eps: pd.DataFrame, now) -> None:
    """Fail closed if any episode row is dated at or after `now`: an episode is only known once its move day has closed."""
    if len(eps) and as_date(pd.Timestamp(eps["date"].max())) >= as_date(now):
        raise FirewallBreach(f"episode dated {pd.Timestamp(eps['date'].max()).date()} is not strictly before now={as_date(now)}")


# ------------------------------------------------------------------------------------------------------------------ data health
SPLIT_RATIOS = (2, 3, 4, 5, 10, 20)


def split_like_mask(c2c: np.ndarray, o2c: np.ndarray, gap: np.ndarray, tol: float = 0.02, body_max: float = 0.05) -> np.ndarray:
    """Corporate-action signature: the previous close is followed by a bar that sits at (almost exactly) 1/k or k times that close for a whole-number
    k, and the move happened at the open (gap ~ close move, small body). Such a bar is a stock split or reverse split in an unadjusted feed, not a
    mover. Ratios like 3-for-2 are left alone: they are indistinguishable from a genuine 33% move."""
    hit = np.zeros(c2c.shape, bool)
    with np.errstate(invalid="ignore"):
        at_open = (np.abs(o2c) < body_max) & (np.abs(gap - c2c) < body_max)
        for k in SPLIT_RATIOS:
            for r in (1.0 / k - 1.0, k - 1.0):
                hit |= np.abs(c2c - r) <= tol
    return hit & at_open


def bars_health(bars: Mapping[str, pd.DataFrame], cfg: EpisodeConfig = EpisodeConfig(), stale_run: int = 5, gap_days: int = 6) -> dict[str, Any]:
    """Data-quality census of a block of bars, run before a unit's counts are trusted. Reports (and lists as issues) misaligned frames, missing
    closes, stale prices (a close that does not change for `stale_run` sessions with no volume), impossible bars, split-like jumps, and calendar
    holes wider than `gap_days` days. It counts; it never repairs."""
    bad = check_bars(bars)
    out: dict[str, Any] = {"issues": list(bad), "sessions": 0, "names": 0}
    if bad:
        return out
    C = bars["Close"]
    out["sessions"], out["names"] = int(C.shape[0]), int(C.shape[1])
    if C.shape[0] == 0 or C.shape[1] == 0:
        out["issues"].append("empty block")
        return out
    g = build_grid(bars, cfg)
    out["missing_close_share"] = float(1.0 - np.isfinite(g.C).mean())
    out["glitch_bars"] = int(g.suspect.sum())
    out["split_like_bars"] = int(split_like_mask(g.c2c, g.o2c, g.gap).sum())
    out["nonpositive_bars"] = int(((g.O <= 0) | (g.H <= 0) | (g.L <= 0) | (g.C <= 0)).sum())
    same = np.zeros(g.C.shape, bool)
    same[1:] = (g.C[1:] == g.C[:-1]) & np.isfinite(g.C[1:])
    run = pd.DataFrame(same.astype(float)).rolling(stale_run).sum().to_numpy()
    quiet = (g.V == 0) if g.V is not None else np.ones(g.C.shape, bool)
    out["stale_bars"] = int(((run >= stale_run) & quiet).sum())
    if len(g.dates) > 1:
        span = np.diff(g.dates.values).astype("timedelta64[D]").astype(int)
        out["calendar_holes"] = int((span > gap_days).sum())
        out["longest_hole_days"] = int(span.max())
    out["volume_present_share"] = float(np.isfinite(g.V).mean()) if g.V is not None else 0.0
    for name, ok in (("missing closes above 20%", out["missing_close_share"] > 0.2), ("glitch bars present", out["glitch_bars"] > 0),
                     ("split-like jumps present", out["split_like_bars"] > 0), ("calendar holes present", out.get("calendar_holes", 0) > 0),
                     ("no volume", out["volume_present_share"] == 0.0)):
        if ok:
            out["issues"].append(name)
    return out


def episode_digest(eps: pd.DataFrame, decimals: int = 9) -> str:
    """Order-independent content hash of an episode frame (block-local `ti`/`nj` excluded), for run-equality checks."""
    if len(eps) == 0:
        return stable_hash([], 16)
    cols = [c for c in eps.columns if c not in ("ti", "nj")]
    d = eps[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(decimals)
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    return stable_hash([[str(v) for v in row] for row in d.itertuples(index=False, name=None)], 16)


# ------------------------------------------------------------------------------------------------------------------ what the episodes look like
def repeat_movers(eps: pd.DataFrame, lens: str = "c2c", window: int = 5) -> dict[str, float]:
    """How often a mover was itself preceded by another episode of the same name within `window` sessions. If the answer is large, episodes are
    not independent draws and every count of them must be clustered by name as well as by date."""
    if len(eps) == 0:
        return {"episodes": 0, "repeat_share": float("nan"), "median_gap": float("nan")}
    band, _ = lens_band_side(eps, lens)
    d = eps[band > 0][["date", "ticker"]].copy()
    if len(d) == 0:
        return {"episodes": 0, "repeat_share": float("nan"), "median_gap": float("nan")}
    sess = {v: i for i, v in enumerate(sorted(d["date"].unique()))}
    d["s"] = d["date"].map(sess)
    d = d.sort_values(["ticker", "s"])
    gap = d.groupby("ticker")["s"].diff()
    return {"episodes": int(len(d)), "repeat_share": float((gap <= window).mean()), "median_gap": float(gap.dropna().median()) if gap.notna().any() else float("nan")}


def concentration_report(eps: pd.DataFrame, lens: str = "c2c", top: int = 10) -> dict[str, float]:
    """Are the episodes spread across the universe or produced by a few names? Herfindahl index of episodes across tickers (1/N is perfectly
    even), the share owned by the top-k names and the number of distinct names."""
    if len(eps) == 0:
        return {"names": 0, "herfindahl": float("nan"), "top_share": float("nan"), "even_herfindahl": float("nan")}
    band, _ = lens_band_side(eps, lens)
    n = eps.loc[band > 0, "ticker"].value_counts()
    if n.empty:
        return {"names": 0, "herfindahl": float("nan"), "top_share": float("nan"), "even_herfindahl": float("nan")}
    p = n / n.sum()
    return {"names": int(len(n)), "herfindahl": float((p ** 2).sum()), "top_share": float(p.head(top).sum()), "even_herfindahl": float(1.0 / len(n))}


def rate_by_era(counts: pd.DataFrame, columns: Sequence[str] | None = None, edges: Sequence[int] = ERA_EDGES) -> pd.DataFrame:
    """Episodes per 1,000 tradable names per day, by era. A band whose rate swings by an order of magnitude between eras is a different phenomenon
    in each, and a pooled precursor estimate for it is suspect."""
    if len(counts) == 0:
        return pd.DataFrame()
    cols = list(columns) if columns is not None else [c for c in counts.columns if c.split("_")[0] in {m.value for m in Measure}]
    era = era_of(counts.index, edges)
    rate = counts[cols].div(counts["n_names"].replace(0, np.nan), axis=0) * 1000.0
    rate["era"] = era
    return rate.groupby("era").mean()


def year_summary(bars: Mapping[str, pd.DataFrame], cfg: EpisodeConfig = EpisodeConfig(), minimum: int = 100) -> dict[str, Any]:
    """One-shot descriptive summary of a block: data health, the 'hundreds a day' check, band totals, repeat and concentration measures. Pure
    counting: nothing here is a finding."""
    blk = run_in_memory(bars, cfg)
    return {"health": bars_health(bars, cfg), "hundreds": hundreds_report(blk.counts, minimum), "band_totals": band_totals(blk.counts),
            "repeat": repeat_movers(blk.episodes), "concentration": concentration_report(blk.episodes), "episodes": int(len(blk.episodes))}


# ------------------------------------------------------------------------------------------------------------------ resumable stream
def stream_state(st: EpisodeStream) -> dict[str, Any]:
    """Serialisable snapshot of a stream (buffer, emitted and fed dates) so an interrupted long run resumes without re-reading old sessions."""
    buf = None
    if st.buf is not None:
        buf = {k: {"index": [str(x.date()) for x in v.index], "columns": [str(c) for c in v.columns], "values": v.to_numpy(np.float64).tolist()}
               for k, v in st.buf.items()}
    return {"cfg": dataclasses.asdict(st.cfg), "emitted": None if st.emitted is None else str(st.emitted.date()),
            "fed": None if st.fed is None else str(st.fed.date()), "sessions_seen": st.sessions_seen, "buf": buf}


def restore_stream(state: Mapping[str, Any], on_block: BlockFn | None = None, extra_warm: int = 0, extra_future: int = 0) -> EpisodeStream:
    """Rebuild a stream from `stream_state`. The configuration must be the one the snapshot was taken under."""
    cfg = EpisodeConfig(**{**state["cfg"], "horizons": tuple(state["cfg"]["horizons"]), "measures": tuple(state["cfg"]["measures"])})
    st = EpisodeStream(cfg, on_block, extra_warm, extra_future)
    if state["buf"] is not None:
        st.buf = {k: pd.DataFrame(np.array(v["values"], dtype=np.float64).reshape(len(v["index"]), len(v["columns"])),
                                  index=pd.DatetimeIndex(v["index"]), columns=v["columns"]) for k, v in state["buf"].items()}
    st.emitted = None if state["emitted"] is None else pd.Timestamp(state["emitted"])
    st.fed = None if state["fed"] is None else pd.Timestamp(state["fed"])
    st.sessions_seen = int(state["sessions_seen"])
    return st


def save_stream(st: EpisodeStream, path) -> str:
    """Atomic snapshot to disk with an embedded content hash; returns the hash."""
    body = stream_state(st)
    h = stable_hash(body, 20)
    _atomic_text(Path(path), json.dumps({"hash": h, "body": body}))
    return h


def load_stream(path, on_block: BlockFn | None = None, extra_warm: int = 0, extra_future: int = 0) -> EpisodeStream:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        body, h = raw["body"], raw["hash"]
    except (OSError, ValueError, KeyError) as e:
        raise EpisodeError(f"stream snapshot unreadable: {e}") from e
    if stable_hash(body, 20) != h:
        raise EpisodeError("stream snapshot fails its content hash")
    return restore_stream(body, on_block, extra_warm, extra_future)
