"""Live jobs (Blueprint Part F). Entry: python -m engine.live <job>

  decide   15:40 ET  full pipeline -> target weights -> orders; logs every prediction
  risk     every 30 min in session: stops, partial takes, bank/brake, red-flag filings
  close    16:15 ET  equity snapshot, shadow baselines, prediction scoring, dashboard data
  weekly   Saturday  retrain + learning loop (engine.improve)

All jobs reconcile against the broker first and are safe to run twice."""
import json, pickle, sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as K, data, features, model, portfolio, options, edgar, policy
from .broker import get_broker
from .backtest import regime_of

ET = ZoneInfo("America/New_York")
MODELS = K.STATE / "models"
PRED_DIR = K.STATE / "predictions"
META = K.STATE / "positions_meta.json"     # stop, entry, opened, reasons per holding
EQUITY = K.STATE / "equity.json"
DECISIONS = K.STATE / "decisions.jsonl"
for p in (MODELS, PRED_DIR):
    p.mkdir(parents=True, exist_ok=True)


def jload(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p, obj):
    p.write_text(json.dumps(obj, indent=1, default=float))


# ---------------- models ----------------
def load_models():
    m = {"cols": json.loads((MODELS / "cols.json").read_text())}
    for k in ("clf", "reg"):
        m[k] = lgb.Booster(model_file=str(MODELS / f"{k}.txt"))
    m["iso"] = pickle.loads((MODELS / "iso.pkl").read_bytes()) if (MODELS / "iso.pkl").exists() else None
    m["meta"] = jload(MODELS / "meta.json", {})
    return m


def predict_live(m, R):
    R = R.reindex(columns=m["cols"])
    out = pd.DataFrame(index=R.index)
    out["rank_raw"] = np.nan
    p = m["clf"].predict(R)
    out["p_stop"], out["p_target_raw"] = p[:, 0], p[:, 2]
    out["p_target"] = m["iso"].predict(out["p_target_raw"]) if m["iso"] is not None else out["p_target_raw"]
    out["mu_raw"] = m["reg"].predict(R)
    contrib = m["reg"].predict(R, pred_contrib=True)[:, :-1]             # SHAP values per feature
    return out, pd.DataFrame(contrib, index=R.index, columns=m["cols"])


# ---------------- data ----------------
def fresh_frames(lookback_days=420):
    data.KEEP_PARTIAL = True          # decide runs ~15:40 ET: today's bar so far ~= the close we trade at
    stocks = data.update("stocks")
    market = data.update("market")
    data.KEEP_PARTIAL = False
    cut = stocks["Close"].index[-1] - pd.Timedelta(days=lookback_days)
    return {k: v.loc[cut:] for k, v in stocks.items()}, {k: v.loc[cut:] for k, v in market.items()}


def todays_features(stocks, market):
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    ins = pd.read_parquet(K.CACHE / "insider.parquet") if (K.CACHE / "insider.parquet").exists() else None
    start = stocks["Close"].index[-1].strftime("%Y-%m-%d")
    X, atr = features.build(stocks, market, ev, ins, sic, start=start)
    return X.xs(X.index.get_level_values(0).max(), level=0, drop_level=False), sic


def explain(contrib_row, evid_row, top=3):
    names = {"ear": "strong earnings reaction", "ins_buyers30": "insider buying cluster",
             "ins_officer30": "officers buying", "dist_52wh": "near 52-week high", "ind_mom60": "strong industry",
             "frog": "steady (not spiky) uptrend", "ev_activist": "activist stake filed", "overnight20": "overnight strength",
             "r5_nonews": "no-news pullback (tends to revert)", "mom_12_1": "12-month momentum",
             "vol_surge5": "rising volume", "rel_ind60": "beating its industry", "max20": "no lottery spike",
             "vol_spread": "options: calls pricier than puts", "cp_volume": "options: call-heavy volume"}
    s = contrib_row.sort_values(ascending=False).head(top)
    return [names.get(k, k.replace("_", " ")) for k in s.index if s[k] > 0]


# ---------------- week state ----------------
def week_state(equity_now):
    hist = jload(EQUITY, [])
    now = datetime.now(ET)
    monday = (now - pd.Timedelta(days=now.weekday())).strftime("%Y-%m-%d")
    before = [h for h in hist if h["date"] < monday]
    week_start = before[-1]["equity"] if before else K.START_CASH
    days_left = max(0, 4 - now.weekday())
    return equity_now / week_start - 1, days_left, week_start


# ---------------- jobs ----------------
def decide():
    broker = get_broker()
    import os
    try:
        if os.environ.get("W7_SKIP_FILINGS"):
            raise RuntimeError("skipped by W7_SKIP_FILINGS")
        n_co, n_ins = edgar.refresh_recent(pd.read_csv(K.CACHE / "universe.csv"))
        print(f"filings refreshed: {n_co} companies, {n_ins} new insider purchases")
    except Exception as e:
        print("filings refresh failed, using cached filings:", e)
    stocks, market = fresh_frames()
    X, sic = todays_features(stocks, market)
    d = X.index.get_level_values(0)[0]
    xr = X.xs(d, level=0)
    R = model.normalise(X).xs(d, level=0)
    m = load_models()
    P, contrib = predict_live(m, R)
    evid = model.evidence_score(model.normalise(X), m["meta"].get("evidence_weights")).xs(d, level=0)
    P["evidence"] = evid
    params = {k: m["meta"][k] for k in policy.PARAMS if k in m["meta"]}
    P["score"] = policy.score(P, params.get("w_model"))
    # options tilt on the top 60 eligible names (live-only signal family, Part M measures it)
    spots = stocks["Close"].iloc[-1].to_dict()
    ok = policy.eligible(xr.reindex(P.index), params)
    top = P.loc[ok[ok].index, "score"].nlargest(60).index
    opt = options.snapshot(list(top), spots)
    tilt = options.evidence_adjustment(opt).reindex(P.index).fillna(0)
    P["score"] = P["score"] + m["meta"].get("w_options", 0.15) * tilt
    # store every component so any re-weighting challenger can be scored on live data (Part M3)
    keep = [c for c in model.EVIDENCE if c in R.columns]
    comp = R.loc[P.index, keep].add_prefix("f_")
    raw = xr.loc[P.index, ["r20", "vol_surge5", "vol20", "atr_pct"]].add_prefix("x_")
    o = opt.add_prefix("o_") if not opt.empty else pd.DataFrame(index=P.index)
    P.join(comp).join(raw).join(o).assign(tilt=tilt, date=str(d.date())).to_parquet(PRED_DIR / f"{d.date()}.parquet")

    prices = stocks["Close"].iloc[-1].to_dict()
    acct = broker.account(prices)
    held = list(broker.positions())
    week_ret, days_left, _ = week_state(acct["equity"])
    friday = days_left == 0
    need = K.WEEKLY_TARGET if friday else (1 + K.WEEKLY_TARGET) / (1 + week_ret) - 1
    horizon = 5 if friday else days_left
    regime = regime_of(xr.iloc[0])
    gross, mode = portfolio.control(0 if friday else week_ret, days_left, True, regime)
    info = {}
    week_end = policy.last_session_of_week(datetime.now(ET).date())
    braked = mode == "brake" or week_ret <= -policy.TOPK["brake"]
    if week_end or not held:
        ok = policy.eligible(xr.reindex(P.index), params).fillna(False)
        target = policy.topk_targets(P.loc[ok[ok].index, "score"], held)
        if braked and not week_end:
            target = target * policy.TOPK["brake_exposure"]
        mode = "rebalance" if week_end else "initial build"
        execute(broker, target, prices, xr, acct["equity"], mode, contrib, P)
        gross = float(target.sum())
    else:
        target = pd.Series(dtype=float)
        mode = "hold (brake)" if braked else "hold"
        print(f"mid-week: holding the week's plan ({mode})")
    rec = {"t": datetime.now(ET).isoformat(timespec="seconds"), "date": str(d.date()), "regime": regime,
           "mode": mode, "gross": gross, "week_ret": week_ret, "need": need, "horizon": horizon,
           "p_target_week": info.get("p_target"), "cvar": info.get("cvar"), "weights": target.round(4).to_dict(),
           "broker": broker.name}
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, indent=1, default=float))


def execute(broker, target, prices, xr, equity, mode, contrib, P):
    pos = broker.positions()
    meta = jload(META, {})
    cur_w = {t: p["qty"] * prices.get(t, p["avg"]) / equity for t, p in pos.items()}
    # sells first (free cash), then buys
    plan = []
    for t in set(cur_w) | set(target.index):
        dw = target.get(t, 0.0) - cur_w.get(t, 0.0)
        if t in cur_w and t in target.index and abs(dw) < K.REBALANCE_BAND:
            continue
        if abs(dw) * equity >= 1.0 and prices.get(t):
            plan.append((dw, t))
    for dw, t in sorted(plan):
        qty = dw * equity / prices[t]
        if dw < 0 and t in pos and target.get(t, 0) == 0:
            qty = -pos[t]["qty"]
        reason = f"rebalance ({mode})" if t in target.index else "dropped from plan"
        broker.order(t, qty, prices[t], reason)
        if dw > 0:
            atr = float(xr.loc[t, "atr_pct"]) * prices[t] if t in xr.index else 0.05 * prices[t]
            meta.setdefault(t, {})
            meta[t].update({"entry": prices[t], "stop": prices[t] - K.STOP_ATR * atr, "took": False,
                            "opened": meta[t].get("opened", datetime.now(ET).strftime("%Y-%m-%d")),
                            "why": explain(contrib.loc[t], P.loc[t]) if t in contrib.index else [],
                            "p_target": float(P.loc[t, "p_target"]) if t in P.index else None})
        elif target.get(t, 0) == 0:
            meta.pop(t, None)
    jsave(META, meta)


def risk():
    """Intraday (champion v1.1): only the -8% weekly brake. Per-stock stops and the +7% bank rule
    were removed after they lost money in the 2017-2026 backtest (see the experiment registry)."""
    broker = get_broker()
    pos = broker.positions()
    if not pos:
        return
    import yfinance as yf
    q = yf.download(list(pos), period="1d", interval="5m", progress=False, auto_adjust=True)
    if not len(q):
        return
    last = q["Close"].ffill().iloc[-1].to_dict()
    acct = broker.account(last)
    week_ret, _, _ = week_state(acct["equity"])
    gross_now = sum(p["qty"] * last.get(t, p["avg"]) for t, p in pos.items()) / acct["equity"]
    cap = policy.TOPK["brake_exposure"]
    if week_ret <= -policy.TOPK["brake"] and gross_now > cap + 0.02:
        scale = cap / gross_now
        for t, p in pos.items():
            broker.order(t, -p["qty"] * (1 - scale), last.get(t, p["avg"]), f"weekly brake ({week_ret:.1%})")


def close():
    from .shadows import update_shadows
    from .scoring import score_predictions
    from .site_data import write_site
    broker = get_broker()
    stocks = data.load("stocks")
    prices = stocks["Close"].iloc[-1].to_dict()
    acct = broker.account(prices)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    hist = [h for h in jload(EQUITY, []) if h["date"] != today]
    hist.append({"date": today, "equity": round(acct["equity"], 2), "cash": round(acct["cash"], 2)})
    jsave(EQUITY, hist)
    update_shadows(stocks)
    score_predictions(stocks)
    write_site(broker, prices)


if __name__ == "__main__":
    job = sys.argv[1]
    {"decide": decide, "risk": risk, "close": close,
     "weekly": lambda: __import__("engine.improve", fromlist=["x"]).weekly()}[job]()
