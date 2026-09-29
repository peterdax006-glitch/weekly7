"""Source families for continual pattern discovery (contract C66 section 13; canons C56, C62, C63, C66).

Section 13 lists every place a pattern may come from: price, returns, ranges, gaps, volume, relative volume, volatility,
cross-sectional / sector / market rank, moving relationships, trend structure, mean reversion, event timing, earnings timing,
filings, insider activity, macro state, breadth, correlations, dispersion, liquidity, regime and existing learned patterns.
This module turns raw inputs into point-in-time feature columns for each of those families, and PROVES the point-in-time
property instead of asserting it.

  * SourceInputs   bars + optional sectors / earnings / filings / insiders / macro. `upto(date)` cuts every table to what was
                   knowable then (future-dated fields of scheduled events are blanked, not merely dropped).
  * FamilySpec     one family = one builder over a pivoted price panel (`Wide`) + the inputs it `requires` + its declared
                   information delay. `FAMILIES` is the registry; a family whose inputs are absent is SKIPPED WITH A REASON.
  * build_features one call -> FeatureBuild (float32 panel indexed (date, ticker), the family and availability class of every
                   column, skipped families). Columns starting `m_` are market context (one value per date), the repo convention.
  * audit_pit      recomputes every family on inputs truncated at earlier dates and compares the overlapping rows: a feature
                   that peeks forward, or that depends on the whole sample, differs and is named. This is the gate that makes a
                   new source admissible; it is run per family, not once for the package.
  * forward_labels the outcome: enter at the NEXT session's open, exit at the close `horizon` sessions after the signal, minus
                   the day's universe mean. Rows whose exit session is not strictly before `now` are not returned.
  * year_chunks    the streaming reader (mapping rule 27): one year of rows at a time with a look-back warm-up, so a market-wide
                   run never holds ~38M name-days at once.

Nothing here reads a cache or the network; every table is handed in. Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.research.core import Availability, FirewallBreach, as_date, stable_hash

ANNUAL = math.sqrt(252.0)
BAR_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


class SourceError(ValueError):
    """Inputs that cannot be turned into features (malformed table, duplicated bar, empty universe)."""


# ------------------------------------------------------------------------------------------------------ configuration
@dataclasses.dataclass(frozen=True)
class SourceConfig:
    """Windows and delays. Delays are in SESSIONS and only ever make information later, never earlier."""
    filing_lag: int = 1              # an SEC filing dated d is treated as known from the second session after d
    insider_lag: int = 1
    earnings_lag: int = 0            # an announcement dated d is known from the first session after d (after-close safe)
    macro_lag: int = 1               # macro prints are revised and released late: one extra session
    assume_lead_sessions: int = 10   # an earnings date with no announced_at is assumed public this many sessions ahead
    event_window: int = 63
    min_price: float = 1.0           # bars below this close are masked (sub-dollar prints dominate ranks with noise)
    warmup: int = 63                 # sessions dropped from the front of every build (rolling windows are not yet valid)
    breadth_min_names: int = 5

    def validate(self) -> list[str]:
        errs = []
        for f in ("filing_lag", "insider_lag", "earnings_lag", "macro_lag", "assume_lead_sessions"):
            if getattr(self, f) < 0:
                errs.append(f"{f} must be >= 0 (a negative delay is a look-ahead)")
        if self.event_window < 5:
            errs.append("event_window must be >= 5")
        if self.warmup < 0 or self.breadth_min_names < 1 or self.min_price < 0:
            errs.append("warmup, breadth_min_names and min_price must be non-negative (breadth_min_names >= 1)")
        return errs


DEFAULT_SOURCE_CONFIG = SourceConfig()


# ---------------------------------------------------------------------------------------------------------- inputs
@dataclasses.dataclass(frozen=True)
class LearnedSignal:
    """An already-learned pattern offered as a source (section 13, 'existing learned patterns'): a pattern_identity expression
    text and the signed weight it carries. Its features must exist in the panel it is scored on."""
    name: str
    expression: str
    weight: float = 1.0


@dataclasses.dataclass
class SourceInputs:
    """bars: long table date,ticker,open,high,low,close,volume. earnings: ticker,date[,announced_at,surprise]. filings:
    ticker,filed_at[,form]. insiders: ticker,filed_at,value[,insider]. macro: DataFrame indexed by date."""
    bars: pd.DataFrame
    sectors: Mapping[str, str] | None = None
    earnings: pd.DataFrame | None = None
    filings: pd.DataFrame | None = None
    insiders: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    learned: tuple[LearnedSignal, ...] = ()

    def validate(self) -> list[str]:
        errs = []
        b = self.bars
        miss = [c for c in BAR_COLUMNS if c not in b.columns]
        if miss:
            return [f"bars missing columns {miss}"]
        if b.empty:
            errs.append("bars is empty")
            return errs
        if b.duplicated(["date", "ticker"]).any():
            errs.append("bars has duplicate (date, ticker) rows")
        if (b[["open", "high", "low", "close"]].astype(float) <= 0).any().any():
            errs.append("bars has non-positive prices")
        hi, lo = b["high"].astype(float), b["low"].astype(float)
        if (hi < lo - 1e-9).any():
            errs.append("bars has high < low")
        if (b["volume"].astype(float) < 0).any():
            errs.append("bars has negative volume")
        for name, cols in (("earnings", ("ticker", "date")), ("filings", ("ticker", "filed_at")),
                           ("insiders", ("ticker", "filed_at", "value"))):
            t = getattr(self, name)
            if t is not None and [c for c in cols if c not in t.columns]:
                errs.append(f"{name} missing columns {[c for c in cols if c not in t.columns]}")
        if self.macro is not None and not isinstance(self.macro.index, pd.DatetimeIndex):
            errs.append("macro must be indexed by date")
        return errs

    def upto(self, date) -> "SourceInputs":
        """Everything knowable at the close of `date` (inclusive). A scheduled event dated after `date` keeps only its date
        (if it had been announced) - its realised fields (surprise) are blanked, they did not exist yet."""
        d = pd.Timestamp(as_date(date))
        b = self.bars[pd.to_datetime(self.bars["date"]) <= d]
        earn = None
        if self.earnings is not None:
            e = self.earnings.copy()
            e["date"] = pd.to_datetime(e["date"])
            ann = pd.to_datetime(e["announced_at"]) if "announced_at" in e.columns else pd.Series(pd.NaT, index=e.index)
            known_later = (e["date"] > d) & (ann <= d)
            keep = (e["date"] <= d) | known_later
            e = e[keep].copy()
            fut = e["date"] > d
            for col in e.columns:
                if col not in ("ticker", "date", "announced_at"):
                    e.loc[fut, col] = np.nan
            earn = e
        fil = None if self.filings is None else self.filings[pd.to_datetime(self.filings["filed_at"]) <= d]
        ins = None if self.insiders is None else self.insiders[pd.to_datetime(self.insiders["filed_at"]) <= d]
        mac = None if self.macro is None else self.macro[self.macro.index <= d]
        return dataclasses.replace(self, bars=b, earnings=earn, filings=fil, insiders=ins, macro=mac)

    def before(self, now) -> "SourceInputs":
        """Strictly before `now`: an observation dated on `now` is not yet known when a decision is made at `now`."""
        d = pd.Timestamp(as_date(now))
        dates = pd.to_datetime(self.bars["date"])
        prior = dates[dates < d]
        if prior.empty:
            raise SourceError(f"no bars strictly before {d.date()}")
        return self.upto(prior.max())

    def tickers(self) -> list[str]:
        return sorted(self.bars["ticker"].astype(str).unique())

    def fingerprint(self) -> str:
        b = self.bars
        return stable_hash({"n": len(b), "t": self.tickers(), "d0": str(pd.to_datetime(b["date"]).min().date()),
                            "d1": str(pd.to_datetime(b["date"]).max().date()),
                            "px": round(float(b["close"].astype(float).sum()), 4),
                            "aux": [None if t is None else len(t) for t in (self.earnings, self.filings, self.insiders, self.macro)]})


class Wide:
    """The pivoted price panel every family reads: dates x tickers. Built once per build_features call."""

    def __init__(self, bars: pd.DataFrame, cfg: SourceConfig):
        b = bars.copy()
        b["date"] = pd.to_datetime(b["date"])
        b["ticker"] = b["ticker"].astype(str)
        piv = {c: b.pivot(index="date", columns="ticker", values=c).sort_index().astype(float) for c in
               ("open", "high", "low", "close", "volume")}
        bad = piv["close"] < cfg.min_price
        for c in piv:
            piv[c] = piv[c].mask(bad)
        self.open, self.high, self.low, self.close, self.volume = (piv[c] for c in ("open", "high", "low", "close", "volume"))
        self.dates = self.close.index
        self.tickers = self.close.columns
        self.n = len(self.dates)
        self.ret = self.close.pct_change(fill_method=None)
        self.mkt_ret = self.ret.mean(axis=1)
        self.mkt_level = (1.0 + self.mkt_ret.fillna(0.0)).cumprod()
        self.cfg = cfg

    def event_positions(self, event_dates, lag: int) -> np.ndarray:
        """First session at which an event dated `event_dates` is known: strictly after its date, plus `lag`."""
        ts = pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize()
        return self.dates.searchsorted(ts, side="right") + lag


# ---------------------------------------------------------------------------------------------------- small kernels
def _consecutive(cond: pd.DataFrame) -> pd.DataFrame:
    """Length of the current run of True per column (0 where False), vectorised across tickers."""
    c = cond.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(c.shape, dtype=np.float64)
    for i in range(c.shape[0]):
        prev = out[i - 1] if i else 0.0
        out[i] = np.where(c[i], prev + 1.0, 0.0)
    return pd.DataFrame(out, index=cond.index, columns=cond.columns)


def _rank_within(df: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """Percentile rank of each ticker among the tickers of its own group, per date."""
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    by: dict[str, list[str]] = {}
    for t in df.columns:
        by.setdefault(groups.get(t, "?"), []).append(t)
    for _, cols in by.items():
        if len(cols) >= 3:
            out[cols] = df[cols].rank(axis=1, pct=True)
    return out


def _group_mean(df: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    by: dict[str, list[str]] = {}
    for t in df.columns:
        by.setdefault(groups.get(t, "?"), []).append(t)
    for _, cols in by.items():
        out[cols] = np.repeat(df[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1)
    return out


def _roll_slope_r2(ly: pd.DataFrame, w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OLS slope and R^2 of a series on its own time index, by window sums (no per-window refit)."""
    idx = np.arange(len(ly), dtype=np.float64)[:, None]
    end = idx
    sy = ly.rolling(w).sum()
    sxy = (ly * idx).rolling(w).sum() - end * sy               # sum of (t - t_end) * y over the window
    syy = (ly * ly).rolling(w).sum()
    k = np.arange(w, dtype=np.float64)
    sx, sxx = float((k - (w - 1)).sum()), float(((k - (w - 1)) ** 2).sum())
    den = w * sxx - sx * sx
    slope = (w * sxy - sx * sy) / den
    varx = den / (w * w)
    vary = (syy - sy * sy / w) / w
    r2 = (slope ** 2 * varx) / vary.where(vary > 1e-14)
    return slope, r2.clip(0.0, 1.0)


def _expanding_like(s: pd.Series, window: int = 504) -> pd.Series:
    """Point-in-time percentile of a market series against its own trailing `window` (min half a window)."""
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


class EventIndex:
    """Per-ticker sorted session positions of events (with optional values and ids): the vectorised answer to 'how long since',
    'how many in the last W sessions' and 'what did they sum to', all evaluated at every session at once."""

    def __init__(self, w: Wide, table: pd.DataFrame | None, time_col: str, lag: int, value_col: str | None = None,
                 id_col: str | None = None, mask: pd.Series | None = None):
        self.w = w
        self.pos: dict[str, np.ndarray] = {}
        self.val: dict[str, np.ndarray] = {}
        self.ids: dict[str, np.ndarray] = {}
        if table is None or table.empty:
            return
        t = table if mask is None else table[mask.reindex(table.index, fill_value=False)]
        for tick, g in t.groupby(t["ticker"].astype(str)):
            if tick not in w.tickers:
                continue
            p = w.event_positions(g[time_col], lag)
            order = np.argsort(p, kind="mergesort")
            keep = order[p[order] < w.n]                      # events not yet effective inside the data are invisible
            if not len(keep):
                continue
            self.pos[tick] = p[keep]
            if value_col is not None:
                self.val[tick] = g[value_col].astype(float).to_numpy()[keep]
            if id_col is not None:
                self.ids[tick] = g[id_col].astype(str).to_numpy()[keep]

    def _frame(self, fn: Callable[[str, np.ndarray], np.ndarray], fill: float = np.nan) -> pd.DataFrame:
        out = np.full((self.w.n, len(self.w.tickers)), fill, dtype=np.float64)
        s = np.arange(self.w.n)
        for j, tick in enumerate(self.w.tickers):
            if tick in self.pos:
                out[:, j] = fn(tick, s)
        return pd.DataFrame(out, index=self.w.dates, columns=self.w.tickers)

    def since(self) -> pd.DataFrame:
        def f(tick, s):
            p = self.pos[tick]
            i = np.searchsorted(p, s, side="right") - 1
            return np.where(i >= 0, s - p[np.maximum(i, 0)], np.nan)
        return self._frame(f)

    def count(self, window: int) -> pd.DataFrame:
        def f(tick, s):
            p = self.pos[tick]
            return (np.searchsorted(p, s, side="right") - np.searchsorted(p, s - window, side="right")).astype(float)
        return self._frame(f, fill=0.0)

    def total(self, window: int) -> pd.DataFrame:
        def f(tick, s):
            p, v = self.pos[tick], self.val[tick]
            cs = np.concatenate([[0.0], np.cumsum(v)])
            return cs[np.searchsorted(p, s, side="right")] - cs[np.searchsorted(p, s - window, side="right")]
        return self._frame(f, fill=0.0)

    def last_value(self, values: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
        src = self.val if values is None else values

        def f(tick, s):
            p = self.pos[tick]
            i = np.searchsorted(p, s, side="right") - 1
            return np.where(i >= 0, src[tick][np.maximum(i, 0)], np.nan)
        return self._frame(f)

    def distinct_active(self, window: int) -> pd.DataFrame:
        """Number of distinct ids with an event in the last `window` sessions: intervals per id are merged, then a difference
        array counts the coverage (exact, no per-session set building)."""
        def f(tick, s):
            diff = np.zeros(self.w.n + window + 2)
            byid: dict[str, list[int]] = {}
            for p, i in zip(self.pos[tick], self.ids[tick]):
                byid.setdefault(i, []).append(int(p))
            for ps in byid.values():
                start, end = ps[0], ps[0] + window
                for p in ps[1:]:
                    if p < end:
                        end = p + window
                    else:
                        diff[start] += 1
                        diff[end] -= 1
                        start, end = p, p + window
                diff[start] += 1
                diff[end] -= 1
            return np.cumsum(diff)[: self.w.n]
        return self._frame(f, fill=0.0)


# ------------------------------------------------------------------------------------------------------ the families
Feature = "pd.DataFrame | pd.Series"


def f_price(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    c = w.close
    return {"ret_1": w.ret, "ret_5": c.pct_change(5, fill_method=None), "ret_21": c.pct_change(21, fill_method=None),
            "ret_63": c.pct_change(63, fill_method=None),
            "mom_21_skip5": c.shift(5) / c.shift(26) - 1.0,
            "dist_high_252": c / c.rolling(252, min_periods=126).max() - 1.0,
            "dist_low_252": c / c.rolling(252, min_periods=126).min() - 1.0,
            "log_price": np.log(c)}


def f_returns(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    r = w.ret
    valid = r.notna()
    up = ((r > 0) & valid).astype(float)
    dn = ((r < 0) & valid).astype(float)
    return {"skew_21": r.rolling(21, min_periods=15).skew(),
            "down_share_21": dn.rolling(21).sum() / valid.astype(float).rolling(21).sum().where(lambda x: x > 0),
            "max_ret_21": r.rolling(21, min_periods=15).max(), "min_ret_21": r.rolling(21, min_periods=15).min(),
            "up_streak": _consecutive(r > 0), "down_streak": _consecutive(r < 0),
            "autocorr_21": r.rolling(21, min_periods=15).corr(r.shift(1)),
            "up_minus_down_21": (up.rolling(21).sum() - dn.rolling(21).sum()) / 21.0}


def f_ranges(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    h, l, c = w.high, w.low, w.close
    pc = c.shift(1)
    tr = pd.DataFrame(np.fmax(np.fmax((h - l).to_numpy(), (h - pc).abs().to_numpy()), (l - pc).abs().to_numpy()),
                      index=c.index, columns=c.columns)
    rng = (h - l) / c
    span = (h - l).where((h - l) > 0)
    return {"range_pct": rng, "atr_14_pct": tr.rolling(14, min_periods=10).mean() / c,
            "range_expansion": rng / rng.rolling(20, min_periods=15).mean().shift(1),
            "close_loc": ((c - l) / span).fillna(0.5).where(c.notna()),
            "inside_day": ((h < h.shift(1)) & (l > l.shift(1))).astype(float).where(c.notna()),
            "narrow_range_7": (rng <= rng.rolling(7).min()).astype(float).where(c.notna() & rng.rolling(7).min().notna()),
            "range_trend": rng.rolling(5).mean() / rng.rolling(63, min_periods=40).mean()}


def f_gaps(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    o, c = w.open, w.close
    pc = c.shift(1)
    gap = o / pc - 1.0
    span = (o - pc).where((o - pc).abs() > 0.002 * pc)
    return {"gap_open": gap, "gap_fill": ((o - c) / span).clip(-2, 2),
            "gap_abs_mean_21": gap.abs().rolling(21, min_periods=15).mean(),
            "gap_up_count_21": (gap > 0.02).astype(float).where(gap.notna()).rolling(21, min_periods=15).sum(),
            "overnight_21": gap.rolling(21, min_periods=15).sum(),
            "intraday_21": (c / o - 1.0).rolling(21, min_periods=15).sum(),
            "overnight_minus_intraday": gap.rolling(21, min_periods=15).sum() - (c / o - 1.0).rolling(21, min_periods=15).sum()}


def f_volume(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    v, c, r = w.volume, w.close, w.ret
    sgn = np.sign(r)
    obv = (sgn * v).fillna(0.0).cumsum()
    v21 = v.rolling(21, min_periods=15).sum()
    upv = v.where(r > 0, 0.0).rolling(21, min_periods=15).sum()
    return {"log_volume_21": np.log1p(v.rolling(21, min_periods=15).mean()),
            "volume_trend": v.rolling(5).mean() / v.rolling(63, min_periods=40).mean(),
            "log_dollar_volume_21": np.log1p((c * v).rolling(21, min_periods=15).mean()),
            "up_volume_share_21": upv / v21.where(v21 > 0),
            "obv_slope_21": (obv - obv.shift(21)) / v21.where(v21 > 0)}


def f_relvol(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    v = w.volume
    base = v.rolling(20, min_periods=15).mean().shift(1)
    rv = v / base.where(base > 0)
    return {"rvol_1": rv, "rvol_5": v.rolling(5).mean() / v.rolling(63, min_periods=40).mean().shift(5),
            "rvol_max_10": rv.rolling(10, min_periods=7).max(), "rvol_signed": rv * np.sign(w.ret),
            "spike_days_21": (rv > 2.0).astype(float).where(rv.notna()).rolling(21, min_periods=15).sum(),
            "quiet_days_10": (rv < 0.6).astype(float).where(rv.notna()).rolling(10, min_periods=7).sum()}


def f_volatility(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    r = w.ret
    rv5 = r.rolling(5, min_periods=4).std() * ANNUAL
    rv21 = r.rolling(21, min_periods=15).std() * ANNUAL
    rv63 = r.rolling(63, min_periods=40).std() * ANNUAL
    hl = np.log(w.high / w.low) ** 2
    semi = (r.clip(upper=0.0) ** 2).rolling(21, min_periods=15).mean()
    return {"rv_5": rv5, "rv_21": rv21, "rv_63": rv63, "vol_ratio_5_63": rv5 / rv63.where(rv63 > 0),
            "vol_of_vol": rv5.rolling(63, min_periods=40).std() / rv63.where(rv63 > 0),
            "parkinson_21": np.sqrt(hl.rolling(21, min_periods=15).mean() / (4.0 * math.log(2.0))) * ANNUAL,
            "downside_vol_21": np.sqrt(semi) * ANNUAL,
            "vol_pct_own_252": rv21.rolling(252, min_periods=126).rank(pct=True)}


def f_xsrank(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    """Cross-sectional rank features: where a stock sits among all stocks today, and how that rank has moved."""
    r5 = w.close.pct_change(5, fill_method=None)
    r21 = w.close.pct_change(21, fill_method=None)
    rv21 = w.ret.rolling(21, min_periods=15).std()
    dv = (w.close * w.volume).rolling(21, min_periods=15).mean()
    rk = {"xs_rank_ret_5": r5.rank(axis=1, pct=True), "xs_rank_ret_21": r21.rank(axis=1, pct=True),
          "xs_rank_rv_21": rv21.rank(axis=1, pct=True), "xs_rank_dollar_vol": dv.rank(axis=1, pct=True)}
    rk["xs_rank_ret_21_change_5"] = rk["xs_rank_ret_21"] - rk["xs_rank_ret_21"].shift(5)
    return rk


def f_sectorrank(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    sec = dict(inp.sectors or {})
    r21 = w.close.pct_change(21, fill_method=None)
    r5 = w.close.pct_change(5, fill_method=None)
    rv21 = w.ret.rolling(21, min_periods=15).std()
    sm21 = _group_mean(r21, sec)
    return {"sec_rank_ret_21": _rank_within(r21, sec), "sec_rank_rv_21": _rank_within(rv21, sec),
            "sec_rel_ret_21": r21 - sm21, "sec_rel_ret_5": r5 - _group_mean(r5, sec), "sector_ret_21": sm21}


def f_marketrank(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    m = w.mkt_level
    m5, m21, m63 = m.pct_change(5), m.pct_change(21), m.pct_change(63)
    r5, r21 = w.close.pct_change(5, fill_method=None), w.close.pct_change(21, fill_method=None)
    r63 = w.close.pct_change(63, fill_method=None)
    rel21 = r21.sub(m21, axis=0)
    return {"mkt_rel_ret_5": r5.sub(m5, axis=0), "mkt_rel_ret_21": rel21, "mkt_rel_ret_63": r63.sub(m63, axis=0),
            "mkt_rank_rel_ret_21": rel21.rank(axis=1, pct=True),
            "mkt_rel_persistence": (w.ret.sub(w.mkt_ret, axis=0) > 0).astype(float).where(w.ret.notna()).rolling(21, min_periods=15).mean()}


def f_movavg(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    c = w.close
    s10, s50 = c.rolling(10, min_periods=8).mean(), c.rolling(50, min_periods=40).mean()
    s200 = c.rolling(200, min_periods=150).mean()
    e12, e26 = c.ewm(span=12, adjust=False, ignore_na=True).mean(), c.ewm(span=26, adjust=False, ignore_na=True).mean()
    sign = np.sign(s50 - s200)
    same = (sign == sign.shift(1)) & sign.notna() & sign.shift(1).notna()
    return {"dist_sma10": c / s10 - 1.0, "dist_sma50": c / s50 - 1.0, "dist_sma200": c / s200 - 1.0,
            "sma10_over_50": s10 / s50 - 1.0, "sma50_over_200": s50 / s200 - 1.0, "slope_sma50_10": s50 / s50.shift(10) - 1.0,
            "macd_pct": (e12 - e26) / c,
            "ma_stack": ((c > s10).astype(float) + (s10 > s50).astype(float) + (s50 > s200).astype(float)).where(s200.notna()),
            "days_since_cross_50_200": _consecutive(same).clip(upper=250.0).where(s200.notna())}


def f_trend(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    c, h = w.close, w.high
    ly = np.log(c)
    slope, r2 = _roll_slope_r2(ly, 30)
    net = (c - c.shift(21)).abs()
    path = c.diff().abs().rolling(21, min_periods=15).sum()
    sma20 = c.rolling(20, min_periods=15).mean()
    return {"efficiency_21": net / path.where(path > 0),
            "breakout_20": (c > h.rolling(20, min_periods=15).max().shift(1)).astype(float).where(sma20.notna()),
            "breakdown_20": (c < w.low.rolling(20, min_periods=15).min().shift(1)).astype(float).where(sma20.notna()),
            "trend_slope_30": slope, "trend_r2_30": r2,
            "higher_highs_20": (c > c.rolling(5).max().shift(1)).astype(float).where(sma20.notna()).rolling(20, min_periods=15).sum(),
            "days_above_sma20": (c > sma20).astype(float).where(sma20.notna()).rolling(20, min_periods=15).mean()}


def f_meanrev(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    c, r = w.close, w.ret
    m20, s20 = c.rolling(20, min_periods=15).mean(), c.rolling(20, min_periods=15).std()
    gain = r.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
    loss = (-r).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.where(loss > 0))
    rv63 = r.rolling(63, min_periods=40).std()
    return {"zscore_20": (c - m20) / s20.where(s20 > 0), "rsi_14": rsi.where(c.notna()),
            "pullback_20": c / c.rolling(20, min_periods=15).max() - 1.0,
            "bounce_20": c / c.rolling(20, min_periods=15).min() - 1.0,
            "ret_5_z": c.pct_change(5, fill_method=None) / (rv63 * math.sqrt(5.0)).where(rv63 > 0),
            "reversal_hits_21": ((np.sign(r) != np.sign(r.shift(1))) & r.notna() & r.shift(1).notna()).astype(float)
            .where(r.notna()).rolling(21, min_periods=15).mean()}


def f_eventtiming(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    """Calendar structure known in advance for every date (month turn, quarter end, options-expiry week, weekday, post-gap)."""
    d = w.dates
    d64 = d.values.astype("datetime64[D]")
    month_start = d.to_period("M").to_timestamp().values.astype("datetime64[D]")
    month_end = (d.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)).values.astype("datetime64[D]")
    pos_in_month = np.busday_count(month_start, d64)                     # weekdays elapsed this month: a pure calendar count,
    to_month_end = np.busday_count(d64, month_end)                       # so a truncated history cannot change it
    third_fri = ((d.day >= 15) & (d.day <= 21) & (d.dayofweek == 4))
    week_key = d.year * 100 + d.isocalendar().week.to_numpy()
    opex_weeks = set(week_key[third_fri])
    gaps = np.diff(d.values.astype("datetime64[D]"), prepend=d.values.astype("datetime64[D]")[:1]).astype(int)
    idx = d
    return {"m_cal_weekday": pd.Series(d.dayofweek.to_numpy(dtype=float), index=idx),
            "m_cal_month": pd.Series(d.month.to_numpy(dtype=float), index=idx),
            "m_cal_turn_of_month": pd.Series(((pos_in_month < 3) | (to_month_end <= 2)).astype(float), index=idx),
            "m_cal_quarter_end_week": pd.Series(((d.month % 3 == 0) & (to_month_end <= 5)).astype(float), index=idx),
            "m_cal_opex_week": pd.Series(np.array([float(k in opex_weeks) for k in week_key]), index=idx),
            "m_cal_post_gap_days": pd.Series(np.clip(gaps, 0, 7).astype(float), index=idx)}


def f_earnings(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    e = inp.earnings.copy()
    e["date"] = pd.to_datetime(e["date"])
    past = e[e["date"] < pd.Timestamp(w.dates[-1]) + pd.Timedelta(days=1)]
    idx = EventIndex(w, past, "date", cfg.earnings_lag)
    since = idx.since()
    c = w.close.to_numpy()
    react, drift, surp = {}, {}, {}
    for tick, p in idx.pos.items():
        j = w.tickers.get_loc(tick)
        col = c[:, j]
        r = np.full(len(p), np.nan)
        ok = (p >= 2) & (p < w.n)
        r[ok] = col[p[ok]] / col[p[ok] - 2] - 1.0
        react[tick] = r
        drift[tick] = col[np.minimum(p, w.n - 1)]
    if "surprise" in past.columns:
        for tick, g in past.assign(_p=w.event_positions(past["date"], cfg.earnings_lag)).groupby(past["ticker"].astype(str)):
            if tick in idx.pos:
                gg = g[g["_p"] < w.n].sort_values("_p", kind="mergesort")
                surp[tick] = gg["surprise"].astype(float).to_numpy()
    last_react = idx.last_value(react)
    last_px = idx.last_value(drift)
    out = {"days_since_earnings": since, "earn_reaction_last": last_react, "earn_drift_since": w.close / last_px - 1.0,
           "earn_recent_5": (since <= 5).astype(float).where(since.notna(), 0.0)}
    if surp:
        out["earn_surprise_last"] = idx.last_value(surp)
    # days to the next earnings date, from dates that were public at the time (announced_at, else a fixed lead)
    to_next = np.full((w.n, len(w.tickers)), np.nan)
    ann = pd.to_datetime(e["announced_at"]) if "announced_at" in e.columns else pd.Series(pd.NaT, index=e.index)
    d64 = w.dates.values.astype("datetime64[D]")
    for (tick, ed, an) in zip(e["ticker"].astype(str), e["date"], ann):
        if tick not in w.tickers:
            continue
        j = w.tickers.get_loc(tick)
        ed_pos = int(w.dates.searchsorted(ed, side="left"))            # first session on/after the event date
        if pd.isna(an):
            first_known = max(ed_pos - cfg.assume_lead_sessions, 0) if ed_pos < w.n else None
            if first_known is None:
                continue                                                # unannounced future event: invisible
        else:
            first_known = int(w.dates.searchsorted(pd.Timestamp(an), side="right"))
        hi = min(ed_pos, w.n)
        if first_known >= hi:
            continue
        span = np.arange(first_known, hi)
        dn = np.busday_count(d64[span], np.datetime64(ed.date(), "D")).astype(float)
        cur = to_next[span, j]
        to_next[span, j] = np.where(np.isnan(cur), dn, np.minimum(cur, dn))
    tn = pd.DataFrame(to_next, index=w.dates, columns=w.tickers)
    out["days_to_earnings"] = tn
    out["earn_in_window_5"] = (tn <= 5).astype(float).where(tn.notna(), 0.0)
    return out


def f_filings(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    f = inp.filings
    allf = EventIndex(w, f, "filed_at", cfg.filing_lag)
    cnt = allf.count(cfg.event_window)
    out = {"days_since_filing": allf.since(), "filings_63": cnt,
           "filings_z_xs": (cnt.sub(cnt.mean(axis=1), axis=0)).div(cnt.std(axis=1).where(lambda x: x > 0), axis=0)}
    if "form" in f.columns:
        form = f["form"].astype(str).str.upper()
        k8 = EventIndex(w, f, "filed_at", cfg.filing_lag, mask=form.str.startswith("8-K"))
        out["days_since_8k"] = k8.since()
        out["filings_8k_63"] = k8.count(cfg.event_window)
        periodic = EventIndex(w, f, "filed_at", cfg.filing_lag, mask=form.str.startswith(("10-K", "10-Q")))
        out["days_since_periodic"] = periodic.since()
    return out


def f_insider(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    t = inp.insiders
    buys = EventIndex(w, t, "filed_at", cfg.insider_lag, "value", mask=t["value"].astype(float) > 0,
                      id_col="insider" if "insider" in t.columns else None)
    sells = EventIndex(w, t, "filed_at", cfg.insider_lag, "value", mask=t["value"].astype(float) < 0)
    net = EventIndex(w, t, "filed_at", cfg.insider_lag, "value").total(cfg.event_window)
    dv = (w.close * w.volume).rolling(21, min_periods=15).mean()
    out = {"insider_net_63": np.sign(net) * np.log1p(net.abs()), "insider_net_over_dv": net / dv.where(dv > 0),
           "insider_buys_63": buys.count(cfg.event_window), "insider_sells_63": sells.count(cfg.event_window),
           "days_since_insider_buy": buys.since()}
    if "insider" in t.columns:
        out["insider_buyers_63"] = buys.distinct_active(cfg.event_window)
    return out


def f_macro(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    m = inp.macro.sort_index().reindex(w.dates.union(inp.macro.index)).ffill().shift(cfg.macro_lag).reindex(w.dates)
    out = {}
    for col in inp.macro.columns:
        s = m[col].astype(float)
        out[f"m_macro_{col}_pct"] = _expanding_like(s)
        out[f"m_macro_{col}_chg5"] = s.diff(5)
        out[f"m_macro_{col}_chg21"] = s.diff(21)
        out[f"m_macro_{col}_z252"] = (s - s.rolling(252, min_periods=126).mean()) / s.rolling(252, min_periods=126).std()
    return out


def f_breadth(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    c = w.close
    have = c.notna().sum(axis=1)
    ok = have >= cfg.breadth_min_names

    def share(cond: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
        return (cond & base.notna()).sum(axis=1).div(base.notna().sum(axis=1).where(lambda x: x > 0)).where(ok)

    s50, s200 = c.rolling(50, min_periods=40).mean(), c.rolling(200, min_periods=150).mean()
    adv = share(w.ret > 0, w.ret)
    dec = share(w.ret < 0, w.ret)
    net = (adv - dec)
    mc = net.ewm(span=19, adjust=False, ignore_na=True).mean() - net.ewm(span=39, adjust=False, ignore_na=True).mean()
    b50 = share(c > s50, s50)
    return {"m_breadth_sma50": b50, "m_breadth_sma200": share(c > s200, s200), "m_adv_share": adv,
            "m_new_high_share": share(c >= c.rolling(252, min_periods=126).max(), c.rolling(252, min_periods=126).max()),
            "m_new_low_share": share(c <= c.rolling(252, min_periods=126).min(), c.rolling(252, min_periods=126).min()),
            "m_breadth_thrust_10": b50 - b50.shift(10), "m_mcclellan": mc,
            "m_gap_up_share": share((w.open / c.shift(1) - 1.0) > 0.01, c.shift(1))}


def f_correlations(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    r, m = w.ret, w.mkt_ret
    sd_i = r.rolling(63, min_periods=40).std()
    sd_m = m.rolling(63, min_periods=40).std()
    corr = r.rolling(63, min_periods=40).corr(m)
    beta = corr.mul(sd_i).div(sd_m.where(sd_m > 0), axis=0)
    n = r.notna().sum(axis=1).astype(float)
    s1, s2 = sd_i.sum(axis=1), (sd_i ** 2).sum(axis=1)
    denom = ((s1 ** 2 - s2) / n ** 2).where(n > 1)
    avg_corr = ((sd_m ** 2 - s2 / n ** 2) / denom.where(denom > 0)).clip(-1.0, 1.0)
    out = {"corr_to_mkt_63": corr, "beta_63": beta, "idio_vol_63": sd_i * np.sqrt((1.0 - corr ** 2).clip(lower=0.0)) * ANNUAL,
           "corr_change_21_63": r.rolling(21, min_periods=15).corr(m) - corr, "m_avg_corr_63": avg_corr,
           "m_avg_corr_change_21": avg_corr - avg_corr.shift(21)}
    if inp.sectors:
        sm = _group_mean(r, dict(inp.sectors))
        out["corr_to_sector_63"] = r.rolling(63, min_periods=40).corr(sm)
    return out


def f_dispersion(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    r = w.ret
    disp = r.std(axis=1)
    d21 = disp.rolling(21, min_periods=15).mean()
    zx = r.sub(r.mean(axis=1), axis=0).div(disp.where(disp > 0), axis=0)
    q90, q10 = r.quantile(0.9, axis=1), r.quantile(0.1, axis=1)
    return {"m_disp_1": disp, "m_disp_21": d21, "m_disp_ratio_252": d21 / d21.rolling(252, min_periods=126).mean(),
            "m_xs_skew": r.skew(axis=1), "m_xs_spread_90_10": q90 - q10, "ret_z_xs": zx,
            "abs_ret_z_xs": zx.abs()}


def f_liquidity(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    dv = w.close * w.volume
    adv21, adv126 = dv.rolling(21, min_periods=15).mean(), dv.rolling(126, min_periods=80).mean()
    amihud = (w.ret.abs() / dv.where(dv > 0)).rolling(21, min_periods=15).mean() * 1e6
    spread = (2.0 * (w.high - w.low) / (w.high + w.low)).rolling(21, min_periods=15).mean()
    zero = (w.volume <= 0).astype(float).where(w.volume.notna()).rolling(63, min_periods=40).mean()
    lt = np.log1p(adv21) - np.log1p(adv126)
    return {"log_adv_21": np.log1p(adv21), "amihud_21": np.log1p(amihud), "hl_spread_21": spread,
            "zero_volume_share_63": zero, "adv_trend": lt, "m_liq_median_adv_chg": lt.median(axis=1),
            "m_liq_median_amihud": np.log1p(amihud).median(axis=1)}


def f_regime(w: Wide, inp: SourceInputs, cfg: SourceConfig) -> dict:
    lvl = w.mkt_level
    trend = (lvl > lvl.rolling(200, min_periods=150).mean()).astype(float).where(lvl.rolling(200, min_periods=150).mean().notna())
    vol = w.mkt_ret.rolling(21, min_periods=15).std() * ANNUAL
    volpct = _expanding_like(vol)
    hi = (volpct > 0.66).astype(float).where(volpct.notna())
    state = (trend * 2.0 + hi)
    age = _consecutive(pd.DataFrame({"s": state == state.shift(1)}))["s"]
    switches = (state != state.shift(1)).astype(float).where(state.notna() & state.shift(1).notna()).rolling(63, min_periods=40).sum()
    return {"m_regime_trend": trend, "m_regime_vol_pct": volpct, "m_regime_drawdown": lvl / lvl.rolling(252, min_periods=126).max() - 1.0,
            "m_regime_state": state, "m_regime_age": age.where(state.notna()), "m_regime_switches_63": switches}


# ------------------------------------------------------------------------------------------------------- the registry
@dataclasses.dataclass(frozen=True)
class FamilySpec:
    name: str
    fn: Callable[[Wide, SourceInputs, SourceConfig], dict]
    requires: tuple[str, ...]                    # optional SourceInputs attributes that must be non-empty
    availability: Availability
    doc: str
    lag_sessions: int = 0                        # information delay already applied inside the builder (documentation + audit)

    def usable(self, inp: SourceInputs) -> str | None:
        """None if the inputs support this family, else the reason it is skipped."""
        for r in self.requires:
            t = getattr(inp, r, None)
            if t is None or (hasattr(t, "empty") and t.empty) or (isinstance(t, Mapping) and not t):
                return f"needs {r}"
        return None


_A = Availability
FAMILIES: dict[str, FamilySpec] = {s.name: s for s in (
    FamilySpec("price", f_price, (), _A.KNOWN_BEFORE_EVENT, "trailing returns and distance to 252-day extremes"),
    FamilySpec("returns", f_returns, (), _A.KNOWN_BEFORE_EVENT, "shape of the trailing return distribution and streaks"),
    FamilySpec("ranges", f_ranges, (), _A.KNOWN_BEFORE_EVENT, "true range, expansion, close location, inside/narrow days"),
    FamilySpec("gaps", f_gaps, (), _A.KNOWN_BEFORE_EVENT, "opening gaps, fills, overnight vs intraday returns"),
    FamilySpec("volume", f_volume, (), _A.KNOWN_BEFORE_EVENT, "volume level, trend and directional volume"),
    FamilySpec("relvol", f_relvol, (), _A.KNOWN_BEFORE_EVENT, "volume relative to the stock's own trailing average"),
    FamilySpec("volatility", f_volatility, (), _A.KNOWN_BEFORE_EVENT, "realised volatility at three horizons, vol-of-vol, semivariance"),
    FamilySpec("xsrank", f_xsrank, (), _A.KNOWN_BEFORE_EVENT, "cross-sectional ranks of return, volatility, liquidity"),
    FamilySpec("sectorrank", f_sectorrank, ("sectors",), _A.KNOWN_BEFORE_EVENT, "rank and return relative to the stock's sector"),
    FamilySpec("marketrank", f_marketrank, (), _A.KNOWN_BEFORE_EVENT, "return relative to the equal-weight market"),
    FamilySpec("movavg", f_movavg, (), _A.KNOWN_BEFORE_EVENT, "moving-average relationships and crossovers"),
    FamilySpec("trend", f_trend, (), _A.KNOWN_BEFORE_EVENT, "trend efficiency, breakouts, regression slope and R2"),
    FamilySpec("meanrev", f_meanrev, (), _A.KNOWN_BEFORE_EVENT, "z-score, RSI, pullback / bounce, reversal frequency"),
    FamilySpec("eventtiming", f_eventtiming, (), _A.KNOWN_BEFORE_EVENT, "calendar structure known in advance"),
    FamilySpec("earnings", f_earnings, ("earnings",), _A.UNCERTAIN, "time since / to earnings, reaction, drift, surprise"),
    FamilySpec("filings", f_filings, ("filings",), _A.UNCERTAIN, "filing recency and density by form", lag_sessions=1),
    FamilySpec("insider", f_insider, ("insiders",), _A.UNCERTAIN, "net insider value, buy clusters, recency", lag_sessions=1),
    FamilySpec("macro", f_macro, ("macro",), _A.UNCERTAIN, "macro level percentile, changes and z-scores", lag_sessions=1),
    FamilySpec("breadth", f_breadth, (), _A.KNOWN_BEFORE_EVENT, "share above averages, advance/decline, new highs, McClellan"),
    FamilySpec("correlations", f_correlations, (), _A.KNOWN_BEFORE_EVENT, "market correlation, beta, idiosyncratic volatility"),
    FamilySpec("dispersion", f_dispersion, (), _A.KNOWN_BEFORE_EVENT, "cross-sectional return dispersion and skew"),
    FamilySpec("liquidity", f_liquidity, (), _A.KNOWN_BEFORE_EVENT, "dollar volume, Amihud, high-low spread, zero-volume days"),
    FamilySpec("regime", f_regime, (), _A.KNOWN_BEFORE_EVENT, "market trend / volatility state, drawdown, regime age"),
)}
DERIVED_FAMILIES = ("learned",)                   # built from other families' columns, never from raw inputs
ALL_FAMILY_NAMES = tuple(FAMILIES) + DERIVED_FAMILIES


def register_family(spec: FamilySpec) -> None:
    """Add a family; the name is the prefix, so it must be new and free of the separator."""
    if spec.name in FAMILIES or spec.name in DERIVED_FAMILIES or "__" in spec.name or not spec.name.isidentifier():
        raise SourceError(f"family name {spec.name!r} is taken or malformed")
    FAMILIES[spec.name] = spec


# --------------------------------------------------------------------------------------------------------- building
@dataclasses.dataclass
class FeatureBuild:
    X: pd.DataFrame
    family_of: dict[str, str]
    availability: dict[str, Availability]
    skipped: dict[str, str]
    last_date: pd.Timestamp | None
    fingerprint: str

    def columns_of(self, family: str) -> list[str]:
        return [c for c, f in self.family_of.items() if f == family]

    def families(self) -> list[str]:
        return sorted(set(self.family_of.values()))


def _stack(w: Wide, feats: Mapping[str, "pd.DataFrame | pd.Series"], keep: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    nt = len(w.tickers)
    for name, v in feats.items():
        if isinstance(v, pd.Series):
            if not name.startswith("m_"):
                raise SourceError(f"per-date feature {name!r} must be named m_*")
            arr = np.repeat(v.reindex(w.dates).to_numpy(dtype=np.float32)[:, None], nt, axis=1)
        else:
            if name.startswith("m_"):
                raise SourceError(f"per-stock feature {name!r} must not be named m_*")
            arr = v.reindex(index=w.dates, columns=w.tickers).to_numpy(dtype=np.float32)
        arr = arr.reshape(-1)
        arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
        out[name] = arr[keep]
    return out


def build_features(inp: SourceInputs, families: Sequence[str] | None = None, cfg: SourceConfig = DEFAULT_SOURCE_CONFIG,
                   as_of=None, before=None) -> FeatureBuild:
    """Point-in-time feature panel. `as_of` cuts the inputs at that close (inclusive); `before` cuts strictly before a moment.
    Families whose inputs are missing are listed in `skipped` with the reason - a missing source is never an empty column."""
    errs = cfg.validate() + inp.validate()
    if errs:
        raise SourceError("; ".join(errs))
    if as_of is not None and before is not None:
        raise SourceError("give as_of or before, not both")
    if as_of is not None:
        inp = inp.upto(as_of)
    if before is not None:
        inp = inp.before(before)
    names = list(families) if families is not None else list(ALL_FAMILY_NAMES)
    unknown = [n for n in names if n not in ALL_FAMILY_NAMES and n not in FAMILIES]
    if unknown:
        raise SourceError(f"unknown families {unknown}")
    w = Wide(inp.bars, cfg)
    skipped: dict[str, str] = {}
    built: dict[str, dict] = {}
    for n in names:
        if n in DERIVED_FAMILIES:
            continue
        why = FAMILIES[n].usable(inp)
        if why:
            skipped[n] = why
            continue
        built[n] = FAMILIES[n].fn(w, inp, cfg)
    keep = w.close.notna().to_numpy().reshape(-1)
    if cfg.warmup:
        keep = keep & np.repeat(np.arange(w.n) >= cfg.warmup, len(w.tickers))
    idx = pd.MultiIndex.from_product([w.dates, w.tickers], names=["date", "ticker"])[keep]
    cols: dict[str, np.ndarray] = {}
    fam_of: dict[str, str] = {}
    avail: dict[str, Availability] = {}
    for n, feats in built.items():
        named = {(f"{n}__{k}" if not k.startswith("m_") else f"m_{n}__{k[2:]}"): v for k, v in feats.items()}
        for col, arr in _stack(w, named, keep).items():
            cols[col] = arr
            fam_of[col] = n
            avail[col] = FAMILIES[n].availability
    X = pd.DataFrame(cols, index=idx)
    if "learned" in names:
        if inp.learned:
            extra = learned_features(X, inp.learned)
            for c in extra.columns:
                fam_of[c] = "learned"
                avail[c] = Availability.KNOWN_BEFORE_EVENT
            X = pd.concat([X, extra], axis=1)
        else:
            skipped["learned"] = "needs learned"
    last = pd.Timestamp(w.dates[-1]) if w.n else None
    return FeatureBuild(X, fam_of, avail, skipped, last, stable_hash({"in": inp.fingerprint(), "f": sorted(fam_of), "c": cfg}))


def learned_features(X: pd.DataFrame, signals: Sequence[LearnedSignal]) -> pd.DataFrame:
    """Existing learned patterns as a source: the summed signed weight of the patterns firing on a row, how many fire, and
    whether patterns of opposite sign fire together (a conflict is information, not noise). Patterns whose features are
    absent from X are skipped and named in `attrs['unscorable']`."""
    from engine.pattern_identity import Expression
    cols = [c for c in X.columns if not c.startswith("learned__")]
    Q = quantise_panel(X[cols])
    score = np.zeros(len(X))
    pos = np.zeros(len(X))
    neg = np.zeros(len(X))
    bad = []
    for s in signals:
        expr = Expression.parse(s.expression)
        if any(f not in Q.features for f in expr.features):
            bad.append(s.name)
            continue
        m = expr.mask(Q.codes, Q.features)
        score += s.weight * m
        pos += m & (s.weight > 0)
        neg += m & (s.weight < 0)
    out = pd.DataFrame({"learned__known_score": score.astype(np.float32), "learned__known_count": (pos + neg).astype(np.float32),
                        "learned__known_conflict": ((pos > 0) & (neg > 0)).astype(np.float32)}, index=X.index)
    out.attrs["unscorable"] = bad
    return out


# ------------------------------------------------------------------------------------------------------- quantiles
N_LEVELS = 5


@dataclasses.dataclass(frozen=True)
class Quantiles:
    """Quantile level per row and feature. -1 = unknown (missing value, too few names that day, time-series warm-up): an
    unknown matches no level and is no exception either, exactly as pattern_identity.Expression.mask treats it."""
    features: tuple[str, ...]
    codes: np.ndarray

    def column(self, feature: str) -> np.ndarray:
        return self.codes[:, self.features.index(feature)]


def quantise_panel(X: pd.DataFrame, min_names: int = 5, min_history: int = 60) -> Quantiles:
    """Stock features: cross-sectional quintile within each date (NaN stays unknown, unlike engine.candidates, which maps NaN
    to the middle level and would pool every missing value into q2). m_* features: expanding point-in-time quintile."""
    from engine.candidates import ts_quintile_series
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise SourceError("X must be indexed by (date, ticker)")
    cols = list(X.columns)
    codes = np.full((len(X), len(cols)), -1, dtype=np.int8)
    dates = X.index.get_level_values(0)
    names = pd.Series(1, index=X.index).groupby(level=0).transform("sum").to_numpy()
    for j, c in enumerate(cols):
        if c.startswith("m_"):
            per = X[c].groupby(level=0).first().sort_index()
            lv = ts_quintile_series(per, min_history)
            codes[:, j] = lv.reindex(dates).fillna(-1).to_numpy(dtype=np.int8)
        else:
            r = X[c].groupby(level=0).rank(pct=True).to_numpy()
            lv = np.minimum(np.nan_to_num(r, nan=-1.0) * N_LEVELS, N_LEVELS - 1).astype(np.int8)
            lv[~np.isfinite(r)] = -1
            lv[names < min_names] = -1
            codes[:, j] = lv
    return Quantiles(tuple(cols), codes)


# ------------------------------------------------------------------------------------------------------------ labels
def forward_labels(bars: pd.DataFrame, horizon: int, now=None) -> tuple[pd.Series, pd.Series]:
    """Excess forward return and the date it matures. Signal at the close of t; fill at the open of t+1; exit at the close of
    t+horizon; minus the universe mean for the same signal date. With `now`, only rows whose exit session is STRICTLY before
    now are returned (an outcome maturing on `now` is not yet known). Returns (y, matured_at) indexed (date, ticker)."""
    if horizon < 1:
        raise SourceError("horizon must be >= 1 session")
    b = bars.copy()
    b["date"] = pd.to_datetime(b["date"])
    b["ticker"] = b["ticker"].astype(str)
    op = b.pivot(index="date", columns="ticker", values="open").sort_index().astype(float)
    cl = b.pivot(index="date", columns="ticker", values="close").sort_index().astype(float)
    raw = cl.shift(-horizon) / op.shift(-1) - 1.0
    exit_date = pd.Series(cl.index, index=cl.index).shift(-horizon)
    excess = raw.sub(raw.mean(axis=1), axis=0)
    y = excess.stack().dropna().rename("y")
    y.index.names = ["date", "ticker"]
    mat = exit_date.reindex(y.index.get_level_values(0)).to_numpy()
    matured = pd.Series(mat, index=y.index, name="matured_at")
    if now is not None:
        cut = pd.Timestamp(as_date(now))
        ok = (matured < cut).to_numpy()
        y, matured = y[ok], matured[ok]
    return y.sort_index(), matured.sort_index()


# --------------------------------------------------------------------------------------------------- the PIT audit
@dataclasses.dataclass(frozen=True)
class PitFinding:
    family: str
    column: str
    cut: str
    max_abs_diff: float
    rows_compared: int

    def __str__(self):
        return f"{self.family}.{self.column}: differs by {self.max_abs_diff:.3g} on {self.rows_compared} rows when data after {self.cut} is removed"


def audit_pit(inp: SourceInputs, families: Sequence[str] | None = None, cfg: SourceConfig = DEFAULT_SOURCE_CONFIG,
              cuts: Sequence | None = None, tol: float = 1e-6, check_last: int = 30) -> list[PitFinding]:
    """Truncation invariance: features on rows dated <= cut must be identical whether or not later data exists. Anything
    that differs used information from after its own date (a shifted-negative window, a full-sample normalisation, an event
    counted before it was public). Returns the offending (family, column) pairs; empty means every checked family passed."""
    full = build_features(inp, families, cfg)
    dates = np.sort(pd.to_datetime(inp.bars["date"]).unique())
    if cuts is None:
        cuts = [dates[int(len(dates) * q)] for q in (0.6, 0.8)]
    out: list[PitFinding] = []
    for cut in cuts:
        cut = pd.Timestamp(cut)
        part = build_features(inp.upto(cut), families, cfg)
        lo = cut - pd.Timedelta(days=int(check_last * 1.6))
        fd = full.X.index.get_level_values(0)
        pd_ = part.X.index.get_level_values(0)
        a = full.X[(fd <= cut) & (fd >= lo)]
        b = part.X[(pd_ <= cut) & (pd_ >= lo)]
        for col in a.columns:
            if col not in b.columns:
                out.append(PitFinding(full.family_of[col], col, str(cut.date()), float("inf"), len(a)))
                continue
            x, y = a[col].to_numpy(dtype=np.float64), b[col].reindex(a.index).to_numpy(dtype=np.float64)
            both = np.isfinite(x) & np.isfinite(y)
            miss = np.isfinite(x) != np.isfinite(y)
            diff = np.where(both, np.abs(x - y), 0.0)
            scale = np.maximum(1.0, np.where(both, np.abs(x), 1.0))
            worst = float(np.max(diff / scale)) if len(diff) else 0.0
            if miss.any():
                worst = float("inf")
            if worst > tol:
                out.append(PitFinding(full.family_of[col], col, str(cut.date()), worst, int(len(a))))
    return out


def admissible_families(inp: SourceInputs, families: Sequence[str] | None = None, cfg: SourceConfig = DEFAULT_SOURCE_CONFIG,
                        **kw) -> tuple[list[str], dict[str, list[PitFinding]]]:
    """Families that pass the PIT audit on these inputs, and the findings for the ones that do not. A family that fails is
    excluded from discovery entirely (its columns cannot be trusted to be knowable at their own date)."""
    names = [n for n in (families or ALL_FAMILY_NAMES) if n in FAMILIES and FAMILIES[n].usable(inp) is None]
    good, bad = [], {}
    for n in names:
        found = audit_pit(inp, [n], cfg, **kw)
        (bad.__setitem__(n, found) if found else good.append(n))
    return good, bad


# --------------------------------------------------------------------------------------------------- streaming reader
def year_chunks(loader: Callable[[pd.Timestamp, pd.Timestamp], SourceInputs], years: Sequence[int], horizon: int,
                families: Sequence[str] | None = None, cfg: SourceConfig = DEFAULT_SOURCE_CONFIG, now=None,
                warmup_days: int = 420, cushion_days: int = 21) -> Iterator[tuple[int, pd.DataFrame, pd.Series]]:
    """Yield (year, X, y) one calendar year at a time. The loader is asked for [year start - warmup_days, year end +
    cushion_days] so rolling windows are valid on the first row of the year and the last rows still get their exit bar; the
    rows are then trimmed to the year. With `now`, years that start on/after it are refused and labels maturing on/after it
    are dropped, so a streamed run has exactly the point-in-time property of a one-shot run."""
    for yr in years:
        start, end = pd.Timestamp(year=yr, month=1, day=1), pd.Timestamp(year=yr, month=12, day=31)
        if now is not None and start >= pd.Timestamp(as_date(now)):
            raise FirewallBreach(f"year {yr} starts on/after now={as_date(now)}")
        inp = loader(start - pd.Timedelta(days=warmup_days), end + pd.Timedelta(days=cushion_days))
        if now is not None:
            inp = inp.before(now)
        fb = build_features(inp, families, cfg)
        y, _ = forward_labels(inp.bars, horizon, now)
        d = fb.X.index.get_level_values(0)
        X = fb.X[(d >= start) & (d <= end)]
        yy = y.reindex(X.index)
        yield yr, X, yy
