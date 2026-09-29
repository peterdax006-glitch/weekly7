"""Stock-level analog engine (Bible PHASE 8.6; canon C41): "where data suffices".

stock_fingerprint()    price state (returns, acceleration, drawdown, MA200 distance, vol, vol ratio, relative strength,
                       correlation, beta) + volume surge, return skew, up-day share, largest recent one-day move, and
                       lagged market-context columns. Outcomes: forward return, vol, drawdown, return vs the market.
data_sufficient()      a stock qualifies for its OWN-history analogs only when, at the query, at least `min_pool` valid
                       analog rows sit >= gap sessions back; short-history names never get a made-up forecast.
StockAnalogs           own-history analogs (KNNEngine per ticker), with walk-forward weights and data-sufficiency gating.
PooledStockAnalogs     cross-stock analogs for names with short history: fingerprints are standardised across the
                       universe ON EACH DATE (uses only that day's cross-section, so point-in-time by construction), the
                       pool is every (ticker, date) at least `gap` sessions back, picks are >= sep sessions apart per
                       ticker, and at most `max_per_window` picks may share one sep-window (one crash day must not
                       fill all k slots with the same market event)."""
import numpy as np
import pandas as pd

from . import analog_weighting as W
from .analogs_sector import hash_seed


def stock_fingerprint(close, volume, market, horizon=5, context=None, min_obs=60, sector_px=None):
    """(F, O) for one stock. close/volume: Series on the stock's own trading days; market: benchmark price Series;
    sector_px: optional price series of the stock's sector index (adds relative strength and correlation to it)."""
    close = close.astype(float).where(close > 0)
    F = W.price_state(close, market, min_obs)
    r = np.log(close / close.shift(1))
    dv = (close * volume.reindex(close.index)).rolling(21, min_periods=10).mean()
    F["vol_surge"] = np.log(dv / (close * volume.reindex(close.index)).rolling(252, min_periods=100).mean())
    F["skew_63"] = r.rolling(63, min_periods=40).skew()
    F["up_share_21"] = (r > 0).where(r.notna()).rolling(21, min_periods=15).mean()
    F["max_move_63"] = r.abs().rolling(63, min_periods=40).max()
    if sector_px is not None:
        sp = sector_px.reindex(close.index).ffill().astype(float)
        sr = np.log(sp / sp.shift(1))
        for n, lab in ((21, "1m"), (63, "3m")):
            F[f"rel_sector_{lab}"] = F[f"ret_{lab}"] - np.log(sp / sp.shift(n))
        F["corr_sector"] = r.rolling(63, min_periods=40).corr(sr)
    if context is not None and len(context.columns):
        F = F.join(context.add_prefix("m_").reindex(close.index).ffill())
    F = F.replace([np.inf, -np.inf], np.nan)
    return F, W.forward_outcomes(close, horizon, market)


def data_sufficient(F, O, target, gap=63, min_hist=250, min_pool=120, as_of=None):
    """True when the stock has enough usable past at `as_of` (default: last row): >= min_pool rows with <= 1/3 of the
    features missing and a known outcome, all >= gap sessions back, and >= min_hist rows of history."""
    if as_of is not None:
        F, O = F.loc[:as_of], O.loc[:as_of]
    if len(F) - gap < min_hist:
        return False
    F, O = F.iloc[: len(F) - gap], O.iloc[: len(O) - gap]
    good = (F.isna().mean(axis=1) <= W.MAX_NAN_FRAC) & O[target].notna()
    return int(good.sum()) >= min_pool


class StockAnalogs:
    """Own-history analogs. fp: {ticker: (F, O)}. Tickers that fail data_sufficient at build time are listed in .insufficient
    (re-checked per query, so a name qualifies from the first date it has enough history)."""

    def __init__(self, fp, horizon=5, gap=63, sep=21, metric="euclid", target=None, min_hist=250, min_pool=120):
        self.horizon, self.gap, self.sep, self.min_pool = horizon, gap, sep, min_pool
        self.target = target or f"fwd_ret_{horizon}d"
        self.engines, self.insufficient = {}, []
        for tk, (F, O) in fp.items():
            if not data_sufficient(F, O, self.target, gap, min_hist, min_pool):
                self.insufficient.append(tk)
                continue
            self.engines[tk] = W.KNNEngine(F, O, self.target, horizon, gap, sep, metric, min_hist)

    def find(self, ticker, t, k=5, uniform=False):
        eng = self.engines.get(ticker)
        if eng is None:
            return None
        i = int(eng.F.index.searchsorted(pd.Timestamp(t), side="right")) - 1
        if i < 0 or not data_sufficient(eng.F.iloc[: i + 1], eng.O.iloc[: i + 1], self.target, self.gap, eng.min_hist, self.min_pool):
            return None
        return eng.query(i, k, uniform=uniform)

    def learn(self, tickers=None, refit_every=126, **kw):
        out = {}
        for tk in (tickers or list(self.engines)):
            eng = self.engines[tk]
            lo = eng.min_hist + eng.gap + eng.horizon + 8 * eng.sep
            pos = list(range(lo, len(eng.F.index), refit_every))
            out[tk] = W.learn_weights_walk_forward(eng, pos, seed=hash_seed(tk), **kw).diagnostics if pos else pd.DataFrame()
        return out

    def ablation(self, ticker, start=None, end=None, step=None, k=5, seed=0, n_random=10):
        eng = self.engines[ticker]
        pos = W.eval_positions(eng, start, end, step or max(self.horizon, 5))
        table, preds = W.compare_methods(eng, pos, k, seed, n_random)
        return table, preds, W.weighting_verdict(table, preds, seed=seed)


class PooledStockAnalogs:
    """Cross-stock analogs. Memory is T x N x F float32 - build it for the working universe, not the whole cache."""

    def __init__(self, fp, target=None, horizon=5, gap=63, sep=21, stride=1, min_tickers=8, min_hist=250,
                 max_per_window=2, metric="euclid"):
        if gap < horizon:
            raise ValueError("gap must be >= horizon")
        self.horizon, self.gap, self.sep, self.stride = horizon, gap, sep, max(1, int(stride))
        self.min_hist, self.max_per_window, self.metric = min_hist, max_per_window, metric
        self.target = target or f"fwd_ret_{horizon}d"
        self.tickers = sorted(fp)
        feats = sorted(set.intersection(*[set(F.columns) for F, _ in fp.values()])) if fp else []
        ocols = sorted(set.intersection(*[set(O.columns) for _, O in fp.values()])) if fp else []
        if self.target not in ocols:
            raise KeyError(f"target {self.target!r} missing from outcomes")
        self.feats, self.ocols = feats, ocols
        self.index = pd.DatetimeIndex(sorted(set().union(*[F.index for F, _ in fp.values()]))) if fp else pd.DatetimeIndex([])
        T, N, nf = len(self.index), len(self.tickers), len(feats)
        Z = np.full((T, N, nf), np.nan, np.float32)
        Y = np.full((T, N, len(ocols)), np.nan, np.float32)
        for j, tk in enumerate(self.tickers):
            F, O = fp[tk]
            Z[:, j, :] = F[feats].reindex(self.index).values
            Y[:, j, :] = O[ocols].reindex(self.index).values
        self.Z = self._cross_section(Z, min_tickers)
        self.Y = Y
        self.ti = ocols.index(self.target)

    @staticmethod
    def _cross_section(Z, min_tickers):
        """Per date and feature: (x - mean) / std across tickers, clipped. Dates with fewer than min_tickers values -> NaN."""
        cnt = np.isfinite(Z).sum(axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            mu = np.nanmean(Z, axis=1, keepdims=True)
            sd = np.nanstd(Z, axis=1, keepdims=True)
            out = np.clip((Z - mu) / np.where(sd > 1e-12, sd, np.nan), -W.ZCLIP, W.ZCLIP)
        return np.where(cnt >= min_tickers, out, np.nan).astype(np.float32)

    def query(self, ticker, t, k=5, w=None):
        """Analogs for (ticker, t) drawn from ALL tickers' pasts >= gap sessions back; None if the stock has no valid
        fingerprint at t or the pool is thin."""
        if ticker not in self.tickers:
            return None
        i = int(self.index.searchsorted(pd.Timestamp(t), side="right")) - 1
        lim = i - self.gap
        if i < 0 or lim < self.min_hist:
            return None
        j = self.tickers.index(ticker)
        zt = self.Z[i, j]
        use = np.flatnonzero(np.isfinite(zt))
        if len(use) < 4:
            return None
        rows = np.arange(0, lim + 1, self.stride)
        block = self.Z[rows][:, :, use]                                       # (nd, N, nu)
        miss = np.isnan(block)
        ok = (miss.mean(axis=2) <= W.MAX_NAN_FRAC) & np.isfinite(self.Y[rows][:, :, self.ti])
        di, ni = np.nonzero(ok)
        if len(di) < 3 * self.sep:
            return None
        Zp = np.where(miss[di, ni], 0.0, block[di, ni]).astype(float)
        wv = np.ones(len(use)) if w is None else pd.Series(w).reindex([self.feats[u] for u in use]).fillna(1.0).values
        d = W.weighted_distance(Zp, zt[use].astype(float), wv, self.metric)
        order = np.argsort(d, kind="stable")
        chosen, per_win = [], {}
        for o in order:
            pos, tk = int(rows[di[o]]), int(ni[o])
            win = pos // self.sep
            if any(tk == c[1] and abs(pos - c[0]) < self.sep for c in chosen):
                continue
            if per_win.get(win, 0) >= self.max_per_window:
                continue
            chosen.append((pos, tk, o))
            per_win[win] = per_win.get(win, 0) + 1
            if len(chosen) == k:
                break
        if not chosen:
            return None
        pick = np.array([c[2] for c in chosen])
        O = self.Y[[c[0] for c in chosen], [c[1] for c in chosen]].astype(float)
        res = W.summarise(self.index[[c[0] for c in chosen]], self.index[i], d[pick], d, O, self.ocols, self.target, k,
                          self.sep, len(use), len(di), age_sessions=i - np.array([c[0] for c in chosen]))
        res["analogs"].insert(1, "ticker", [self.tickers[c[1]] for c in chosen])
        return res

    def forecast_matrix(self, dates, tickers=None, k=5):
        """(dates x tickers) pooled-analog forecasts as of each date and the realised target (scoring only)."""
        tk = tickers or self.tickers
        P = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates), columns=tk)
        Y = P.copy()
        for t in P.index:
            i = int(self.index.searchsorted(t, side="right")) - 1
            for name in tk:
                r = self.query(name, t, k)
                if r is not None:
                    P.loc[t, name] = r["forecast_return"]
                    Y.loc[t, name] = self.Y[i, self.tickers.index(name), self.ti]
        return P, Y

    def rank_ic(self, dates, tickers=None, k=5, seed=0):
        """Cross-sectional rank IC of the pooled analog forecast vs the realised forward return (per date, summary)."""
        P, Y = self.forecast_matrix(dates, tickers, k)
        return W.cross_sectional_ic(P, Y, min_n=5, seed=seed)

    def panel_features(self, dates, tickers=None, k=5):
        """(date, ticker) -> pooled analog forecast / confidence / uniqueness for the pattern miner, as of each date."""
        rows = {}
        for t in dates:
            for tk in (tickers or self.tickers):
                r = self.query(tk, t, k)
                rows[(pd.Timestamp(t), tk)] = ({"an_stk_fc": r["forecast_return"], "an_stk_conf": r["confidence"],
                                                "an_stk_uniq": r["uniqueness"]} if r else
                                               {"an_stk_fc": np.nan, "an_stk_conf": np.nan, "an_stk_uniq": np.nan})
        X = pd.DataFrame.from_dict(rows, orient="index")
        X.index = pd.MultiIndex.from_tuples(X.index, names=["date", "ticker"])
        return X
