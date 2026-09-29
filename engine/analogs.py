"""Rare-analog engine (canon C41): notice when today resembles the only other time something happened.

fingerprints()  one row per trading day since 1962: market state at several scales (drawdown, speed of rise and its
                acceleration, breadth, dispersion, fear and its term structure, rates, curve, credit, stress,
                recession, sector crowding). Macro series are lagged to their publication delay: point-in-time.
Analogs.find()  for a day t: the k most similar PAST days (at least `gap` sessions before t, so an analog is a
                separate episode, not last week), their similarity, what followed them, and how unique t is.
Every row keeps its real date (C41: "know what year all data points occurred"); the blind feed shifts all dates by
the same amount, so relative ages - which is all the engine uses - survive the disguise."""
import numpy as np
import pandas as pd

from . import config as K, data

# publication lags (sessions) so a value is only used once it was public
# publication lag in SESSIONS after the observation date. Audited 2026-09-28 (scripts/pit_audit_real.py): UNRATE at 25
# and UMCSENT at 20 were ~1 session early on 10-20% of months; USREC is set retroactively when NBER dates a turning
# point 6-21 months later, so a 120-session lag leaked future recession calls - 460 sessions (~22 months) is safe.
LAG = {"CPIAUCSL": 35, "UNRATE": 30, "INDPRO": 35, "UMCSENT": 25, "USREC": 460, "NFCI": 7, "STLFSI4": 7}


def fingerprints():
    mk = data.load("market")["Close"]
    try:
        ix = data.load("index_hist")["Close"]
        spx = mk["SPY"].combine_first(ix["^GSPC"] * (mk["SPY"].dropna().iloc[0] / ix["^GSPC"].reindex(mk["SPY"].dropna().index).dropna().iloc[0]))
        vix = mk["^VIX"].combine_first(ix["^VIX"]) if "^VIX" in ix else mk["^VIX"]
    except FileNotFoundError:
        spx, vix = mk["SPY"], mk["^VIX"]
    spx = spx.dropna()
    idx = spx.index
    r = np.log(spx / spx.shift(1))
    F = pd.DataFrame(index=idx)
    F["drawdown"] = spx / spx.rolling(252, min_periods=60).max() - 1
    for n, lab in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        F[f"ret_{lab}"] = np.log(spx / spx.shift(n))
    F["accel"] = F["ret_6m"] - F["ret_12m"] / 2                     # speed-up of the rise (bubble-like when high)
    F["dist_ma200"] = spx / spx.rolling(200, min_periods=100).mean() - 1
    F["vol_1m"] = r.rolling(21).std() * np.sqrt(252)
    F["vol_ratio"] = r.rolling(21).std() / r.rolling(252, min_periods=100).std()
    F["vix"] = vix.reindex(idx).ffill()
    if "^VIX3M" in mk:
        F["vix_term"] = (vix / mk["^VIX3M"]).reindex(idx).ffill()
    # breadth / dispersion / sector crowding from the stock universe (post-1962 survivors)
    try:
        old = data.load("stocks_pre2000")["Close"]
        new = data.load("stocks")["Close"]
        C = pd.concat([old, new.loc["2000-01-01":]]).sort_index()
        C = C.loc[~C.index.duplicated(keep="last")]
        V = pd.concat([data.load("stocks_pre2000")["Volume"], data.load("stocks")["Volume"].loc["2000-01-01":]]).sort_index()
        V = V.loc[~V.index.duplicated(keep="last")]
    except FileNotFoundError:
        C, V = data.load("stocks")["Close"], data.load("stocks")["Volume"]
    C, V = C.reindex(idx), V.reindex(idx)
    live = C.notna().sum(axis=1)
    F["breadth"] = ((C > C.rolling(50, min_periods=30).mean()).sum(axis=1) / live.replace(0, np.nan))
    rr = np.log(C / C.shift(1))
    F["dispersion"] = rr.std(axis=1).rolling(21, min_periods=10).mean()
    sic = pd.read_parquet(K.CACHE / "sic.parquet").set_index("ticker")["sic"].astype(str).str[:2]
    dv = (C * V).rolling(21, min_periods=10).mean()
    grp = sic.reindex(dv.columns).fillna("99").values
    by_sector = dv.T.groupby(grp).sum().T
    share = by_sector.div(by_sector.sum(axis=1), axis=0)
    F["sector_crowding"] = (share ** 2).sum(axis=1)                  # Herfindahl of trading by sector
    F["top_sector_share_chg"] = share.max(axis=1) - share.max(axis=1).shift(63)
    # macro, point-in-time
    try:
        M = pd.read_parquet(K.CACHE / "macro.parquet")
        for c in M.columns:
            s = M[c].dropna()
            s = s.reindex(idx.union(s.index)).ffill().reindex(idx)
            F[f"mac_{c}"] = s.shift(LAG.get(c, 1))
        if "mac_DGS10" in F:
            F["rate_chg_1y"] = F["mac_DGS10"] - F["mac_DGS10"].shift(252)
        if "mac_BAMLH0A0HYM2" in F:
            F["credit_chg_3m"] = F["mac_BAMLH0A0HYM2"] - F["mac_BAMLH0A0HYM2"].shift(63)
    except FileNotFoundError:
        pass
    # what followed each day (outcomes; used only for PAST analogs, never as inputs)
    O = pd.DataFrame(index=idx)
    O["fwd_ret_1m"] = np.log(spx.shift(-21) / spx)
    O["fwd_vol_1m"] = r[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252)
    O["fwd_maxdd_1m"] = (spx.shift(-21).rolling(21).min() / spx - 1)
    return F, O


class Analogs:
    def __init__(self, F, O, gap=63, min_hist=250):
        self.F, self.O, self.gap, self.min_hist = F, O, gap, min_hist
        self.w = pd.Series(1.0, index=F.columns)                      # feature weights (learned by .learn_weights)

    def find(self, t, k=5):
        """k nearest past days to day t, using only rows that end >= gap sessions before t (plus their outcomes,
        which by then are fully known: outcomes span 21 sessions < gap)."""
        idx = self.F.index
        i = idx.get_loc(t)
        if i - self.gap < self.min_hist:
            return None
        past = self.F.iloc[: i - self.gap]
        cols = [c for c in self.F.columns if self.F[c].iloc[:i].notna().mean() > 0.5 and pd.notna(self.F.iloc[i][c])]
        if len(cols) < 4:
            return None
        P = past[cols]
        mu, sd = P.mean(), P.std().replace(0, np.nan)
        Z = ((P - mu) / sd).fillna(0.0)
        zt = ((self.F.iloc[i][cols] - mu) / sd).fillna(0.0)
        w = self.w.reindex(cols).fillna(1.0).values
        d = np.sqrt((((Z.values - zt.values) ** 2) * w).sum(axis=1) / w.sum())
        order = np.argsort(d)
        # one analog per episode: skip days within 21 sessions of an already-chosen analog
        chosen = []
        for j in order:
            if all(abs(j - c) > 21 for c in chosen):
                chosen.append(j)
            if len(chosen) == k:
                break
        rows = []
        for j in chosen:
            rows.append({"date": P.index[j], "distance": float(d[j]), "age_years": (t - P.index[j]).days / 365.25,
                         **self.O.iloc[j].to_dict()})
        A = pd.DataFrame(rows)
        uniq = float(d[order[0]])                                     # how unusual today is
        sim = np.exp(-A["distance"] ** 2)
        pred = {c: float((sim * A[c]).sum() / sim.sum()) for c in self.O.columns if A[c].notna().all()}
        return {"analogs": A, "uniqueness": uniq, "prediction": pred, "n_close": int((d < 1.0).sum())}
