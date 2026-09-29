"""Sector analog engine (Bible PHASE 8.5; canon C41): when a sector last looked like this, what did it do next?

sector_groups()        two-digit SIC (or any ticker->label map) -> member lists, dropping sectors too thin to index.
sector_index()         equal-weight index of the members alive each day (entries/exits handled; needs >= min_names live).
sector_fingerprints()  per sector: price state (returns, acceleration, drawdown, MA200 distance, vol, vol ratio, relative
                       strength, correlation, beta) + internal breadth, dispersion, dollar-volume share and its 63-session
                       change (crowding) + lagged market context columns (m_*). Outcomes: forward return, vol, drawdown
                       and return relative to the market.
SectorAnalogs          one KNNEngine per sector (point-in-time z-scores, analog end >= gap sessions before the query,
                       analogs >= sep apart, one per episode) with walk-forward learned weights, a cross-sector ranking
                       for a day, and an ablation against no-analog / random / shuffled / random-weight controls.
Survivorship caveat: the stock cache holds today's survivors, so historical sector indices are optimistic; the analog
forecast inherits that bias and only relative comparisons between sectors should be trusted until a delisted-inclusive
universe exists."""
import numpy as np
import pandas as pd

from . import analog_weighting as W


def sector_groups(labels, min_members=5, level=2):
    """labels: Series ticker -> code (e.g. SIC). Codes are truncated to `level` characters. Returns {sector: [tickers]}
    for sectors with >= min_members names; unknown labels are excluded rather than lumped into a fake sector."""
    s = pd.Series(labels).dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")].str[:level]
    out = {}
    for code, tk in s.groupby(s).groups.items():
        if len(tk) >= min_members:
            out[str(code)] = sorted(tk)
    return out


def sector_index(close, members, min_names=3):
    """Equal-weight index (start 100) built from mean daily log-returns of members with a price today and yesterday.
    Days with fewer than min_names contributors carry no return (NaN index until enough names exist)."""
    cols = [c for c in members if c in close.columns]
    if not cols:
        return pd.Series(dtype=float)
    r = np.log(close[cols] / close[cols].shift(1))
    r = r.replace([np.inf, -np.inf], np.nan)
    n = r.notna().sum(axis=1)
    mean = r.mean(axis=1).where(n >= min_names)
    ok = mean.first_valid_index()
    if ok is None:
        return pd.Series(np.nan, index=close.index)
    return 100 * np.exp(mean.fillna(0.0).cumsum()).where(mean.index >= ok)


def sector_fingerprints(close, volume, groups, market, context=None, horizon=21, min_names=3, min_obs=60):
    """{sector: (F, O)}. close/volume: date x ticker. market: benchmark price Series (e.g. SPY). context: optional frame of
    market fingerprints already lagged to publication (e.g. analogs.fingerprints()[0]); its columns arrive as m_<name>."""
    mkt = market.astype(float)
    dv = (close * volume)
    dv21 = dv.rolling(21, min_periods=10).mean()
    tot = dv21.sum(axis=1).replace(0, np.nan)
    ctx = None
    if context is not None and len(context.columns):
        ctx = context.add_prefix("m_").reindex(close.index).ffill()
    out = {}
    for sec, members in groups.items():
        idx = sector_index(close, members, min_names)
        if idx.notna().sum() < min_obs + horizon:
            continue
        cols = [c for c in members if c in close.columns]
        F = W.price_state(idx, mkt, min_obs)
        px = close[cols]
        F["breadth_50"] = (px > px.rolling(50, min_periods=30).mean()).where(px.notna()).sum(axis=1) / px.notna().sum(axis=1).replace(0, np.nan)
        rr = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan)
        F["dispersion"] = rr.std(axis=1).rolling(21, min_periods=10).mean()
        sdv = dv21[cols].sum(axis=1)
        share = sdv / tot
        F["dv_share"] = share
        F["dv_share_chg"] = share - share.shift(63)
        w = dv21[cols].div(sdv.replace(0, np.nan), axis=0)
        F["top_heavy"] = (w ** 2).sum(axis=1)                 # Herfindahl inside the sector: is one name carrying it?
        if ctx is not None:
            F = F.join(ctx)
        F = F.replace([np.inf, -np.inf], np.nan)
        O = W.forward_outcomes(idx, horizon, mkt)
        out[sec] = (F, O)
    return out


def breadth_dispersion_crowding(close, volume, labels, level=2, ma=50):
    """Market-wide fingerprint columns from the stock universe (Bible 8.1): breadth (share of names above their 50-session
    mean), dispersion (21-session mean of the cross-sectional std of daily returns), sector crowding (Herfindahl of dollar volume
    by sector) and top-sector share change over 63 sessions. Row t uses data <= t; names without a label are left out of the
    sector shares (they still count for breadth and dispersion)."""
    live = close.notna().sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=close.index)
    out["breadth"] = (close > close.rolling(ma, min_periods=int(ma * 0.6)).mean()).where(close.notna()).sum(axis=1) / live
    rr = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    out["dispersion"] = rr.std(axis=1).rolling(21, min_periods=10).mean()
    lab = pd.Series(labels).dropna().astype(str).str.strip().str[:level]
    lab = lab[~lab.isin(["", "nan"])]
    dv = (close * volume).rolling(21, min_periods=10).mean()
    cols = [c for c in dv.columns if c in lab.index]
    if cols:
        by = dv[cols].T.groupby(lab.reindex(cols).values).sum().T
        share = by.div(by.sum(axis=1).replace(0, np.nan), axis=0)
        out["sector_crowding"] = (share ** 2).sum(axis=1)
        top = share.max(axis=1)
        out["top_sector_share_chg"] = top - top.shift(63)
    return out


class SectorAnalogs:
    """Sector analog forecasts. fp: {sector: (F, O)} from sector_fingerprints (or any frames with the same layout)."""

    def __init__(self, fp, horizon=21, gap=63, sep=21, metric="euclid", target=None, min_hist=250):
        self.horizon, self.gap, self.sep = horizon, gap, sep
        self.target = target or f"fwd_rel_{horizon}d"
        self.engines = {}
        self.skipped = {}
        for sec, (F, O) in fp.items():
            tgt = self.target if self.target in O.columns else f"fwd_ret_{horizon}d"
            try:
                self.engines[sec] = W.KNNEngine(F, O, tgt, horizon, gap, sep, metric, min_hist)
            except (KeyError, ValueError) as e:
                self.skipped[sec] = str(e)

    def sectors(self):
        return sorted(self.engines)

    def _pos(self, eng, t):
        i = int(eng.F.index.searchsorted(pd.Timestamp(t), side="right")) - 1     # last row on/before t: never after
        return i if i >= 0 else None

    def find(self, sector, t, k=5, uniform=False):
        """Analog result dict (dates, distances, ages, forecast return/vol/drawdown, uniqueness, count, confidence) for
        `sector` as of t, or None when history/features are insufficient."""
        eng = self.engines.get(sector)
        if eng is None:
            return None
        i = self._pos(eng, t)
        return None if i is None else eng.query(i, k, uniform=uniform)

    def rank(self, t, k=5):
        """One row per sector as of t: forecast, confidence, uniqueness, count. Sorted by confidence-scaled forecast so
        the sectors the analogs are both bullish on AND sure about come first."""
        rows = []
        for sec in self.sectors():
            r = self.find(sec, t, k)
            if r is None:
                continue
            rows.append({"sector": sec, "forecast": r["forecast_return"], "forecast_vol": r["forecast_vol"],
                         "forecast_drawdown": r["forecast_drawdown"], "confidence": r["confidence"],
                         "uniqueness": r["uniqueness"], "n_analogs": r["n_analogs"],
                         "score": r["forecast_return"] * r["confidence"]})
        cols = ["sector", "forecast", "forecast_vol", "forecast_drawdown", "confidence", "uniqueness", "n_analogs", "score"]
        return pd.DataFrame(rows, columns=cols).sort_values("score", ascending=False).reset_index(drop=True)

    def learn(self, refit_every=126, start=None, **kw):
        """Walk-forward feature weights per sector (8.4). Refits every `refit_every` sessions from the first row that has
        enough history; returns {sector: diagnostics frame}."""
        out = {}
        for sec, eng in self.engines.items():
            lo = eng.min_hist + eng.gap + eng.horizon + 8 * eng.sep
            if start is not None:
                lo = max(lo, int(eng.F.index.searchsorted(pd.Timestamp(start))))
            pos = list(range(lo, len(eng.F.index), refit_every))
            out[sec] = W.learn_weights_walk_forward(eng, pos, seed=hash_seed(sec), **kw).diagnostics if pos else pd.DataFrame()
        return out

    def ablation(self, sector, start=None, end=None, step=21, k=5, seed=0, n_random=10):
        """8.8 for one sector: (table, predictions, verdict). Requires learn() first if 'weighted' should differ from uniform."""
        eng = self.engines[sector]
        pos = W.eval_positions(eng, start, end, step)
        table, preds = W.compare_methods(eng, pos, k, seed, n_random)
        return table, preds, W.weighting_verdict(table, preds, seed=seed)

    def forecast_matrix(self, dates, k=5, realised="fwd_rel"):
        """(dates x sectors) analog forecasts as of each date, and the matching realised outcome frame (used only for scoring)."""
        P, Y = {}, {}
        for sec in self.sectors():
            eng = self.engines[sec]
            rcol = next((c for c in eng.O.columns if c.startswith(realised)), eng.target)
            fc, re = {}, {}
            for t in dates:
                r = self.find(sec, t, k)
                i = self._pos(eng, t)
                if r is not None:
                    fc[pd.Timestamp(t)], re[pd.Timestamp(t)] = r["forecast_return"], eng.O[rcol].iloc[i]
            P[sec], Y[sec] = pd.Series(fc, dtype=float), pd.Series(re, dtype=float)
        return pd.DataFrame(P), pd.DataFrame(Y)

    def rank_ic(self, dates, k=5, seed=0):
        """Does ranking sectors by analog forecast order their realised relative return? (IC per date, summary)."""
        P, Y = self.forecast_matrix(dates, k)
        return W.cross_sectional_ic(P, Y, min_n=min(5, max(3, len(self.engines))), seed=seed)

    def panel_features(self, dates, sector_of, k=5):
        """Rows (date, ticker) -> sector analog forecast/confidence/uniqueness for the pattern miner (as-of each date).
        sector_of: ticker -> sector code. Tickers without an analog sector get NaN, never a fabricated value."""
        rows = {}
        for t in dates:
            per = {sec: self.find(sec, t, k) for sec in self.sectors()}
            for tk, sec in sector_of.items():
                r = per.get(sec)
                rows[(pd.Timestamp(t), tk)] = ({"an_sec_fc": r["forecast_return"], "an_sec_conf": r["confidence"],
                                                "an_sec_uniq": r["uniqueness"]} if r else
                                               {"an_sec_fc": np.nan, "an_sec_conf": np.nan, "an_sec_uniq": np.nan})
        X = pd.DataFrame.from_dict(rows, orient="index")
        X.index = pd.MultiIndex.from_tuples(X.index, names=["date", "ticker"])
        return X


def hash_seed(text):
    """Stable small integer from a label (Python's hash() is salted per process, so it cannot seed anything)."""
    import zlib
    return zlib.crc32(str(text).encode()) % 2 ** 31
