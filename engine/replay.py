"""Live-as-if replays of any year 1976-2025 (canon C9, C10).

prepare_year(Y)  builds point-in-time features for Y-7..Y, trains the model ONLY on data whose labels end
                 before Y starts, and saves one prediction row per stock per trading day of Y.
simulate(Y, cfg) walks Y one trading day at a time with the live policy rules and era-appropriate costs.
The first call is the honest out-of-sample replay; later calls with other configs are how the loop
tests adjustments on years it has already seen."""
import json
import numpy as np
import pandas as pd

from . import config as K, data, features, model, policy

DIR = K.STATE / "replays"
PRED = K.CACHE / "replay_preds"
DIR.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

CHAMPION = {"k": 4, "exit_q": 0.8, "brake": 0.08, "max_per_sector": 2, "w_model": 0.5, "pick": "top",
            "pool_q": 0.95, "liq_q": 0.5, "vol_filter": True}
_cache = {}


def _all_prices():
    if "stocks" in _cache:
        return _cache["stocks"], _cache["market"]
    new = data.load("stocks")
    try:
        old = data.load("stocks_pre2000")
        stocks = {f: pd.concat([old[f], new[f].loc["2000-01-01":]]).sort_index() for f in new}
    except FileNotFoundError:
        stocks = new
    stocks = {f: v.loc[~v.index.duplicated(keep="last")] for f, v in stocks.items()}
    mk = data.load("market")
    try:
        ix = data.load("index_hist")
    except FileNotFoundError:
        _cache["stocks"], _cache["market"] = stocks, mk
        return stocks, mk
    # market proxy: SPY where it exists (1993+), the S&P 500 index before that
    close = mk["Close"].reindex(ix["Close"].index.union(mk["Close"].index))
    g = ix["Close"]["^GSPC"].reindex(close.index)
    spy = close["SPY"]
    first = spy.first_valid_index()
    scale = spy.loc[first] / g.loc[first]
    close["SPY"] = spy.where(spy.notna(), g * scale)
    for c in ("^VIX",):
        close[c] = close[c].where(close[c].notna(), ix["Close"][c].reindex(close.index))
    market = {f: (close if f == "Close" else mk[f].reindex(close.index)) for f in mk}
    _cache["stocks"], _cache["market"] = stocks, market
    return stocks, market


def cost_bps(year):
    """Era costs: 1/8-dollar ticks before 1997 and 1/16 until decimalisation in 2001 meant far wider spreads."""
    return 40 if year < 1997 else 20 if year < 2001 else 10


def prepare_year(Y, log=print):
    out = PRED / f"{Y}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    stocks, market = _all_prices()
    lo_data = pd.Timestamp(f"{Y - 8}-01-01")
    sl = {f: v.loc[lo_data:f"{Y}-12-31"].dropna(axis=1, how="all") for f, v in stocks.items()}
    mk = {f: v.loc[lo_data:f"{Y}-12-31"] for f, v in market.items()}
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    ins = pd.read_parquet(K.CACHE / "insider.parquet") if (K.CACHE / "insider.parquet").exists() else None
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    X, atr = features.build(sl, mk, ev, ins, sic, start=f"{Y - 7}-01-01", relative=True)
    for c in ("ev_activist", "ev_activist_amend"):      # 13D attribution still under repair; absent pre-2001 anyway
        if c in X:
            X[c] = 0.0
    udates = pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
    ydays = udates[udates.year == Y]
    first_day = ydays[0]
    cut = udates[udates.searchsorted(first_day) - model.EMBARGO]
    train_days = udates[(udates >= first_day - pd.DateOffset(years=model.TRAIN_YEARS)) & (udates < cut)][::2]
    last_label = udates[udates.searchsorted(train_days.max()) + 5]
    assert last_label < first_day, f"leak in {Y}: labels reach {last_label.date()}"
    yb, fw = features.labels(sl, atr)
    d = X.index.get_level_values(0)
    Xtr = X[d.isin(train_days)]
    y = yb.stack(future_stack=True).reindex(Xtr.index)
    f = fw.stack(future_stack=True).reindex(Xtr.index)
    ok = y.notna().values & f.notna().values
    m = model.fit_models(model.normalise(Xtr)[ok], y[ok], f[ok])
    Xy = X[d.year == Y]
    R = model.normalise(Xy).reindex(columns=m["cols"])
    ew = {k: v for k, v in model.EVIDENCE.items() if k != "ev_activist"}
    P = pd.DataFrame({"mu_raw": m["reg"].predict(R), "evidence": model.evidence_score(R, ew)}, index=R.index)
    for c in ("vol20", "max20", "log_dv", "ev_red_flag", "ev_offering"):
        P[c] = Xy[c].values
    P.attrs = {}
    P.to_parquet(out)
    (DIR / f"prep_{Y}.json").write_text(json.dumps({"year": Y, "train_from": str(train_days.min().date()),
                                                    "train_to": str(train_days.max().date()),
                                                    "labels_end": str(last_label.date()), "rows": int(ok.sum()),
                                                    "stocks_in_year": int(Xy.index.get_level_values(1).nunique())}))
    log(f"  {Y}: trained on {int(ok.sum()):,} rows through {train_days.max().date()} "
        f"(labels end {last_label.date()}), {Xy.index.get_level_values(1).nunique()} stocks tradable")
    return P


def _eligible(p, cfg):
    ok = policy.not_crypto(p.index) & ~((p["ev_red_flag"] > 0) | ((p["ev_offering"] > 0) & (p["log_dv"].rank(pct=True) < 0.5)))
    if cfg["vol_filter"]:
        ok &= ~((p["vol20"].rank(pct=True) > 0.9) | (p["max20"].rank(pct=True) > 0.9))
    ok &= p["log_dv"].rank(pct=True) >= cfg["liq_q"]
    return ok


def simulate(Y, cfg=None, P=None, detail=False):
    cfg = {**CHAMPION, **(cfg or {})}
    stocks, market = _all_prices()
    P = prepare_year(Y) if P is None else P
    C = stocks["Close"]
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    divs = {t: policy.sic_division(c) for t, c in zip(sic["ticker"], sic["sic"])}
    days = pd.DatetimeIndex(sorted(P.index.get_level_values(0).unique()))
    bps = cost_bps(Y)
    cash, pos, week_start, capped = K.START_CASH, {}, K.START_CASH, False
    eq, weeks, trades = {}, [], []
    spy = market["Close"]["SPY"]
    prev_week_spy = spy.loc[:days[0]].iloc[-2]
    for i, d in enumerate(days):
        px = C.loc[d]
        val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        week_end = i + 1 >= len(days) or days[i + 1].isocalendar().week != d.isocalendar().week
        wr = val / week_start - 1
        target = None
        if week_end or not pos:
            p = P.xs(d, level=0)
            s = policy.score(p, cfg["w_model"])
            ok = _eligible(p, cfg)
            target = policy.topk_targets(s[ok], list(pos), cfg["k"], cfg["exit_q"],
                                         divs if cfg["max_per_sector"] else None, cfg["max_per_sector"],
                                         pick=cfg["pick"], vol=p["vol20"], pool_q=cfg["pool_q"])
        elif not capped and cfg["brake"] and wr <= -cfg["brake"]:
            target = pd.Series({t: q * px[t] / val for t, q in pos.items()}) * policy.TOPK["brake_exposure"]
            capped = True
        if target is not None:
            target = target * 0.985
            for t in set(pos) | set(target.index):
                pr = px.get(t, np.nan)
                if not np.isfinite(pr):
                    continue
                dv = target.get(t, 0.0) * val - pos.get(t, 0.0) * pr
                if abs(dv) < 1.0:
                    continue
                cash -= dv + abs(dv) * bps / 1e4
                pos[t] = pos.get(t, 0.0) + dv / pr
                if detail:
                    trades.append({"date": str(d.date()), "ticker": t, "dollars": round(dv, 2), "price": round(float(pr), 2)})
                if abs(pos[t]) * pr < 0.5:
                    pos.pop(t)
            val = cash + sum(q * px[t] for t, q in pos.items() if np.isfinite(px.get(t, np.nan)))
        eq[d] = val
        if week_end:
            sp = spy.loc[:d].iloc[-1]
            weeks.append({"week_end": str(d.date()), "ret": val / week_start - 1, "spy": sp / prev_week_spy - 1,
                          "holdings": sorted(pos) if detail else None})
            week_start, capped, prev_week_spy = val, False, sp
    e = pd.Series(eq)
    w = pd.Series([x["ret"] for x in weeks])
    ws = pd.Series([x["spy"] for x in weeks])
    res = {"year": Y, "end": float(e.iloc[-1]), "year_return": float(e.iloc[-1] / K.START_CASH - 1),
           "spy_year_return": float((1 + ws).prod() - 1), "mean_week": float(w.mean()), "spy_mean_week": float(ws.mean()),
           "weeks": len(w), "weeks_ge_7": int((w >= 0.07).sum()), "weeks_le_m7": int((w <= -0.07).sum()),
           "best_week": float(w.max()), "worst_week": float(w.min()), "max_drawdown": float((e / e.cummax() - 1).min()),
           "cost_bps": bps}
    if detail:
        res.update({"weekly": weeks, "trades": trades})
    return res
