"""Point-in-time feature panel (Blueprint Part B).

Every feature for decision date t uses only information available by ~15:40 ET on t:
bars through t-1 close plus t's bar (the decision is made at the close), and filings
accepted before 15:30 ET on t. Output: long DataFrame indexed (date, ticker), float32.

Options-chain features (B2) have no free history, so they are live-only and enter through
the evidence composite; the learning loop measures them live (Part M)."""
import numpy as np
import pandas as pd

from .config import CACHE, MIN_PRICE, MIN_DOLLAR_VOL

ET = "America/New_York"


def _rank(df):  # cross-sectional percentile rank per date, centred on 0
    return df.rank(axis=1, pct=True) - 0.5


def _event_calendar(ev: pd.DataFrame, kind: str, dates, tickers) -> pd.DataFrame:
    """1 on the first session at which a filing of this kind was public, else 0."""
    e = ev[ev["kind"] == kind].copy()
    if e.empty:
        return pd.DataFrame(0.0, index=dates, columns=tickers, dtype="float32")
    loc = e["accepted"].dt.tz_convert(ET)
    day = loc.dt.tz_localize(None).dt.normalize()
    after = (loc.dt.hour * 60 + loc.dt.minute) >= 15 * 60 + 30
    idx = dates.searchsorted(day.values)
    idx = np.where(after.values & (idx < len(dates)) & (dates[np.minimum(idx, len(dates) - 1)] == day.values), idx + 1, idx)
    e = e.assign(i=idx)
    e = e[(e["i"] < len(dates)) & e["ticker"].isin(tickers)]
    m = pd.DataFrame(0.0, index=dates, columns=tickers, dtype="float32")
    for t, grp in e.groupby("ticker"):
        m.iloc[grp["i"].unique(), m.columns.get_loc(t)] = 1.0
    return m


def _days_since(flag: pd.DataFrame, cap=400) -> pd.DataFrame:
    pos = np.arange(len(flag), dtype="float32")[:, None]
    last = np.where(flag.values > 0, pos, np.nan)
    last = pd.DataFrame(last, index=flag.index, columns=flag.columns).ffill()
    return (pos - last).clip(upper=cap).fillna(cap).astype("float32")


def build(stocks: dict, market: dict, ev: pd.DataFrame, ins: pd.DataFrame, sic: pd.DataFrame,
          start="2013-01-01", chunk=500, relative=False, rel_q=(0.2, 0.4)):
    """Cross-sectional pieces (industry momentum, regime) on the whole universe, then the
    per-stock features in ticker chunks so memory stays bounded (~50 wide frames per chunk)."""
    C, H, L, V = stocks["Close"], stocks["High"], stocks["Low"], stocks["Volume"]
    dv20 = (C * V).rolling(20, min_periods=15).median()
    if relative:
        # any-era filters (canon C10: 1976-2025): split-adjusted prices and 1980s dollar volumes make fixed
        # thresholds meaningless, so keep names above the day's 20th price and 40th dollar-volume percentile
        tradable = (C.rank(axis=1, pct=True) >= rel_q[0]) & (dv20.rank(axis=1, pct=True) >= rel_q[1]) & C.notna()
    else:
        tradable = (C >= MIN_PRICE) & (dv20 >= MIN_DOLLAR_VOL) & C.notna()
    ever = tradable.loc[pd.Timestamp(start) - pd.Timedelta(days=10):].any()
    tickers = ever[ever].index
    r = np.log(C / C.shift(1))
    grp = sic.set_index("ticker")["sic"].astype(str).str[:2].reindex(C.columns).fillna("99")
    ind = {}
    for n in (20, 60):
        base = np.log(C / C.shift(n)).where(tradable)
        ind[n] = base.T.groupby(grp.values).transform("mean").T.astype("float32")
    reg = regime_frame(market, C.where(tradable), r.where(tradable))
    C1 = C.shift(1)
    atr = np.maximum(H - L, np.maximum((H - C1).abs(), (L - C1).abs())).rolling(14, min_periods=10).mean()
    del r, C1
    from .edgar import EQUITY_424
    ev = ev[~((ev["kind"] == "OFFERING") & ~ev["form"].isin(EQUITY_424))]
    parts = []
    for i in range(0, len(tickers), chunk):
        cols = tickers[i:i + chunk]
        sub = {k: v[cols] for k, v in stocks.items()}
        ins_c = None if ins is None else ins[ins["symbol"].isin(cols)]
        parts.append(_block(sub, market, ev[ev["ticker"].isin(cols)], ins_c,
                            {n: ind[n][cols] for n in ind}, tradable[cols], start))
    X = pd.concat(parts).sort_index()
    regs = reg.reindex(X.index.get_level_values(0))
    for c in reg.columns:
        X[c] = regs[c].values.astype("float32")
    X.index.names = ["date", "ticker"]
    return X, atr


def _block(stocks, market, ev, ins, ind, tradable, start):
    O, H, L, C, V = (stocks[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    dates, tickers = C.index, C.columns
    r = np.log(C / C.shift(1))
    dv = (C * V)
    dv20 = dv.rolling(20, min_periods=15).median()

    F = {}
    for n in (1, 5, 20, 60, 120):
        F[f"r{n}"] = np.log(C / C.shift(n))
    F["mom_12_1"] = np.log(C.shift(21) / C.shift(252))
    F["dist_52wh"] = C / H.rolling(252, min_periods=120).max() - 1
    F["dist_ma50"] = C / C.rolling(50).mean() - 1
    F["dist_ma200"] = C / C.rolling(200, min_periods=150).mean() - 1
    vol20 = r.rolling(20, min_periods=15).std()
    F["vol20"] = vol20
    F["vol_ratio"] = vol20 / r.rolling(120, min_periods=60).std()
    C1 = C.shift(1)
    tr = np.maximum(H - L, np.maximum((H - C1).abs(), (L - C1).abs()))
    atr = tr.rolling(14, min_periods=10).mean()
    F["atr_pct"] = atr / C
    F["max20"] = r.rolling(20, min_periods=15).max()                         # lottery penalty (B5)
    F["min20"] = r.rolling(20, min_periods=15).min()
    F["skew60"] = r.rolling(60, min_periods=40).skew()
    F["log_dv"] = np.log1p(dv20)
    F["vol_surge1"] = V / V.rolling(50, min_periods=30).mean()
    F["vol_surge5"] = V.rolling(5).mean() / V.rolling(50, min_periods=30).mean()
    on = np.log(O / C.shift(1))
    F["overnight20"] = on.rolling(20, min_periods=15).sum()
    F["intraday20"] = np.log(C / O).rolling(20, min_periods=15).sum()
    pos_d = (r > 0).astype("float32").rolling(60, min_periods=40).mean()
    neg_d = (r < 0).astype("float32").rolling(60, min_periods=40).mean()
    F["frog"] = -np.sign(F["r60"]) * (neg_d - pos_d)          # Da-Gurun-Warachka: high = continuous momentum
    F["range_compress"] = (H.rolling(10).max() / L.rolling(10).min()) / (H.rolling(60).max() / L.rolling(60).min())
    F["gap_today"] = on
    F["close_loc"] = (C - L) / (H - L).replace(0, np.nan)                     # where in today's range it closed

    # industry (SIC 2-digit) momentum, computed on the whole universe by build()
    for n in (20, 60):
        F[f"ind_mom{n}"] = ind[n]
        F[f"rel_ind{n}"] = F[f"r{n}"] - ind[n]

    # ---- events (B1, B3) ----
    kinds = {k: _event_calendar(ev, k, dates, tickers) for k in
             ("EARN", "OFFERING", "SHELF", "ACTIVIST", "ACTIVIST_AMEND", "RESTATEMENT", "DELIST_NOTICE",
              "BANKRUPTCY", "LATE_FILING", "AGREEMENT", "AUDITOR_CHANGE", "UNREG_SALE")}
    earn = kinds["EARN"]
    ds_earn = _days_since(earn)
    F["days_since_earn"] = ds_earn
    # earnings-announcement return: move from the close before the release to 1 session after, vs SPY
    spy = market["Close"]["SPY"].reindex(dates)
    ab = r.sub(np.log(spy / spy.shift(1)), axis=0)
    ear_now = (ab + ab.shift(-1)).where(earn > 0)      # uses t+1 -> shift forward below so it's known
    ear_known = ear_now.shift(1)                        # available one session after the release session
    F["ear"] = ear_known.ffill(limit=60)
    earn_vs = (V / V.rolling(50, min_periods=30).mean()).where(earn > 0).shift(1)
    F["ear_volsurge"] = earn_vs.ffill(limit=60)
    # expected next earnings from quarterly cadence (~63 sessions): days until
    F["days_to_earn"] = (63 - ds_earn).where(ds_earn < 120)
    F["earn_in_week"] = ((F["days_to_earn"] >= 0) & (F["days_to_earn"] <= 6)).astype("float32")
    for k, w in (("OFFERING", 10), ("SHELF", 30), ("ACTIVIST", 20), ("ACTIVIST_AMEND", 20), ("AGREEMENT", 10)):
        F[f"ev_{k.lower()}"] = kinds[k].rolling(w, min_periods=1).max()
    bad = sum(kinds[k] for k in ("RESTATEMENT", "DELIST_NOTICE", "BANKRUPTCY", "LATE_FILING", "AUDITOR_CHANGE"))
    F["ev_red_flag"] = bad.rolling(60, min_periods=1).max()
    news = sum(kinds.values())
    news5 = news.rolling(5, min_periods=1).max()
    F["news5"] = news5
    # Chan (2003): moves without news reverse; moves with news continue
    F["r5_nonews"] = F["r5"] * (1 - news5)
    F["r5_news"] = F["r5"] * news5

    # ---- insider purchases (B2) ----
    F.update(_insider_features(ins, dates, tickers, dv20))

    panel = []
    keep = dates[dates >= pd.Timestamp(start)]
    for name, w in F.items():
        w = w.reindex(index=keep).where(tradable.reindex(keep))
        panel.append(w.stack(future_stack=True).rename(name).astype("float32"))
    X = pd.concat(panel, axis=1)
    return X[X["r1"].notna()]


def _insider_features(ins, dates, tickers, dv20):
    out = {}
    z = pd.DataFrame(0.0, index=dates, columns=tickers, dtype="float32")
    if ins is None or ins.empty:
        return {"ins_buyers30": z, "ins_value30": z, "ins_officer30": z, "ins_opportunistic30": z}
    d = ins[ins["symbol"].isin(tickers)].copy()
    d = d[(d["value"] > 1000) & (d["value"] < 5e8)]          # drop data-entry errors (e.g. $7e15 rows)
    d["day"] = d["filed"].dt.normalize() + pd.Timedelta(days=1)       # filing time unknown -> next day
    # Cohen-Malloy-Pomorski: routine = same calendar month in each of the prior 3 years
    d["ym"] = d["tdate"].dt.year * 12 + d["tdate"].dt.month
    # point-in-time: a past trade only counts once its OWN filing was public (late filings exist). Without
    # this, the routine check could see a report filed after the decision day - the leak the parity test caught.
    first_public = d.groupby([d["owner_cik"], d["tdate"].dt.year, d["tdate"].dt.month])["day"].min().to_dict()
    y, mth = d["tdate"].dt.year.values, d["tdate"].dt.month.values
    d["routine"] = [all(first_public.get((o, yy - k, mm), pd.Timestamp.max) <= day for k in (1, 2, 3))
                    for o, yy, mm, day in zip(d["owner_cik"], y, mth, d["day"])]
    rel = d["relation"].fillna("").str.lower() + " " + d["title"].fillna("").str.lower()
    d["officer"] = rel.str.contains("officer|ceo|cfo|chief|president").astype("float32")
    d["i"] = dates.searchsorted(d["day"].values)
    d = d[d["i"] < len(dates)]

    def mat(sub, col=None, how="sum"):
        m = pd.DataFrame(0.0, index=dates, columns=tickers, dtype="float64")
        g = sub.groupby(["i", "symbol"])
        s = g[col].sum() if col else g["owner_cik"].nunique()
        for (i, t), v in s.items():
            m.iat[i, m.columns.get_loc(t)] += v
        return m

    opp = d[~d["routine"]]
    buyers = mat(opp)                          # distinct opportunistic buyers per day
    out["ins_buyers30"] = buyers.rolling(21, min_periods=1).sum()
    out["ins_value30"] = mat(opp, "value").rolling(21, min_periods=1).sum() / (dv20 * 21)
    out["ins_officer30"] = mat(opp, "officer").rolling(21, min_periods=1).sum()
    out["ins_opportunistic30"] = (out["ins_buyers30"] >= 2).astype("float32")  # cluster
    return out


def regime_frame(market, Ct, rt):
    mc = market["Close"]
    spy = mc["SPY"]
    df = pd.DataFrame(index=Ct.index)
    s = spy.reindex(df.index)
    df["m_spy_ma50"] = s / s.rolling(50).mean() - 1
    df["m_spy_ma200"] = s / s.rolling(200).mean() - 1
    df["m_spy_r5"] = np.log(s / s.shift(5))
    vix = mc["^VIX"].reindex(df.index).ffill()
    df["m_vix"] = vix
    df["m_vix_chg5"] = np.log(vix / vix.shift(5))
    if "^VIX3M" in mc:
        df["m_vix_term"] = vix / mc["^VIX3M"].reindex(df.index).ffill()
    df["m_breadth"] = (Ct > Ct.rolling(50).mean()).sum(axis=1) / Ct.notna().sum(axis=1)
    df["m_dispersion"] = rt.std(axis=1).rolling(5).mean()
    return df.astype("float32")


def labels(stocks: dict, atr: pd.DataFrame, horizon=5, target=0.07, stop_atr=2.0):
    """Triple-barrier (Part C1) from entry at close t over sessions t+1..t+horizon.
    y_bar: 1 = +target touched first, -1 = stop touched first (ties -> stop), 0 = neither.
    fwd: close-to-close log return over the horizon."""
    C, H, L = stocks["Close"], stocks["High"], stocks["Low"]
    up = C * (1 + target)
    dn = C - stop_atr * atr
    first_up = pd.DataFrame(np.inf, index=C.index, columns=C.columns)
    first_dn = pd.DataFrame(np.inf, index=C.index, columns=C.columns)
    for k in range(horizon, 0, -1):
        hk, lk = H.shift(-k), L.shift(-k)
        first_up = first_up.mask(hk >= up, k)
        first_dn = first_dn.mask(lk <= dn, k)
    y = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    y = y.mask(first_up < first_dn, 1.0).mask(first_dn <= first_up, -1.0).where(np.isfinite(first_dn) | np.isfinite(first_up), 0.0)
    fwd = np.log(C.shift(-horizon) / C)
    y = y.where(fwd.notna())
    return y, fwd
