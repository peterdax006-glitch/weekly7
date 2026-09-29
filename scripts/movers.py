"""Canon C23 step 1: every week, pick the 10 stocks most likely to move 10% (up OR down) within the next week,
and measure how often that is right. Blind windows (sealed random 12-month periods, disguised dates/tickers);
the model trains only on warm-up data whose labels end >= 5 sessions before the window; outcomes are read
only after each week has passed (they are stored for evaluation, never shown to the model).

usage: movers.py run <window_id> [variant_json]     one blind window -> state/movers/<id>/
       movers.py report                            hit rates across all windows"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import lightgbm as lgb
from engine import config as K, livesim

OUT = K.STATE / "movers"
OUT.mkdir(parents=True, exist_ok=True)
H, MOVE = 5, 0.10                                   # next 5 sessions, +/-10%
DROP = ("ev_activist",)                             # insider leak fixed 28 Sep; 13D attribution still being repaired


def labels(stocks):
    """Canon C33: a pick decided after day t's close can only be bought at day t+1's OPEN. Moves are measured
    from that entry over sessions t+1..t+5 (regular hours only)."""
    C, Hh, L, O = stocks["Close"], stocks["High"], stocks["Low"], stocks["Open"]
    entry = O.shift(-1)
    hi = pd.concat([Hh.shift(-k) for k in range(1, H + 1)]).groupby(level=0).max()
    lo = pd.concat([L.shift(-k) for k in range(1, H + 1)]).groupby(level=0).min()
    up, dn = hi / entry - 1, lo / entry - 1
    close = C.shift(-H) / entry - 1
    return up, dn, close


def week_ends(idx):
    return [d for i, d in enumerate(idx[:-1]) if idx[i + 1].isocalendar().week != d.isocalendar().week]


def run_window(rid, variant=None, scramble_after=None, save=True):
    v = {"n_estimators": 300, "num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.05, **(variant or {})}
    model_v = {k: v[k] for k in ("n_estimators", "num_leaves", "min_child_samples", "learning_rate")}
    t0 = time.time()
    feed = livesim.Feed(livesim.SealedYear(rid))
    if v.get("f64") or scramble_after is not None:     # same numeric precision on both sides of the scramble test
        for f in ("Open", "High", "Low", "Close"):
            feed._stocks[f] = feed._stocks[f].astype("float64")
    if scramble_after is not None:                     # anti-cheat: noise after the cut must not change earlier picks
        cut_day = feed.sessions[feed.sessions.get_loc(feed.first_live) + scramble_after]
        m = feed._stocks["Close"].index > cut_day
        rng = np.random.default_rng(11)
        for f in ("Open", "High", "Low", "Close"):
            arr = feed._stocks[f].astype("float64")
            arr.loc[m] = arr.loc[m].values * np.exp(rng.normal(0, 0.3, arr.loc[m].shape))
            feed._stocks[f] = arr
        run_window.cut_day = cut_day
    feed.precompute_features(rel_q=tuple(v.get("rel_q", (0.2, 0.4))))
    X = feed._X
    if v.get("stock_type"):                             # C27: explicit stock-type inputs (sector group, size, price level)
        from engine import policy
        div = {t: policy.sic_division(c) for t, c in zip(feed.sic["ticker"], feed.sic["sic"])}
        tick = X.index.get_level_values(1)
        X = X.assign(sector_div=[float(ord(div.get(t, "?")[0]) - 64) for t in tick],
                     price_level=feed._stocks["Close"].stack(future_stack=True).reindex(X.index).values)
    if v.get("absfeat"):                               # magnitude signals: a +/-10% mover can go either way
        X = X.assign(abs_ear=X["ear"].abs(), abs_r5=X["r5"].abs(), abs_r1=X["r1"].abs(),
                     abs_gap=X["gap_today"].abs(), abs_r20=X["r20"].abs(), range_pct=X["max20"] - X["min20"])
    cols = [c for c in X.columns if not c.startswith(DROP)]
    S = feed._stocks
    up, dn, cl = labels(S)
    touch = ((up >= MOVE) | (dn <= -MOVE)).astype(float)
    sessions = feed.sessions
    first = feed.first_live
    # ---- train on warm-up week-ends whose 5-day label window closes before the hidden window ----
    warm = [d for d in week_ends(sessions[sessions < first])]
    cutoff = sessions[max(0, sessions.get_loc(first) - H - 1)]
    if v.get("all_days"):                               # every session, not only week-ends (~5x the rows)
        warm = list(sessions[(sessions < first) & (sessions >= sessions[0] + pd.Timedelta(days=300))])
    warm = [d for d in warm if d <= cutoff]
    dates = X.index.get_level_values(0)
    Xt = X[dates.isin(warm)][cols]
    y = touch.stack(future_stack=True).reindex(Xt.index)
    ok = y.notna().values
    Rt = Xt[ok].groupby(level=0).rank(pct=True)                  # cross-sectional ranks: era-free
    for c in cols:
        if c.startswith("m_"):
            Rt[c] = Xt[ok][c]
    clf = lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                             random_state=7, verbose=-1, **model_v)
    clf.fit(Rt, y[ok])
    base_rate_train = float(y[ok].mean())
    reg = None
    if v.get("magnitude"):                              # also predict HOW FAR it swings (max of |high|, |low| move)
        swing = np.maximum(up, -dn).clip(upper=1.0)
        ys = swing.stack(future_stack=True).reindex(Xt.index)[ok]
        reg = lgb.LGBMRegressor(objective="huber", alpha=0.1, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                random_state=7, verbose=-1, **model_v)
        reg.fit(Rt, ys.fillna(0))
    # ---- hidden window: predict each week, evaluate after the week ----
    rows = []
    refit = int(v.get("refit_weeks", 0))
    hidden_weeks = week_ends(sessions[sessions >= first])
    for wi, d in enumerate(hidden_weeks):
        if refit and wi and wi % refit == 0:
            # C15/C16/C29: re-train on everything whose 5-session label window has CLOSED by today
            closed_to = sessions[max(0, sessions.get_loc(d) - H - 1)]
            train_days = [x for x in week_ends(sessions[sessions <= closed_to]) if x >= warm[0]]
            Xt2 = X[dates.isin(train_days)][cols]
            y2 = touch.stack(future_stack=True).reindex(Xt2.index)
            ok2 = y2.notna().values
            R2 = Xt2[ok2].groupby(level=0).rank(pct=True)
            for c in cols:
                if c.startswith("m_"):
                    R2[c] = Xt2[ok2][c]
            clf = lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                     random_state=7, verbose=-1, **model_v)
            clf.fit(R2, y2[ok2])
        Xd = X.xs(d, level=0)[cols]
        Rd = Xd.rank(pct=True)
        for c in cols:
            if c.startswith("m_"):
                Rd[c] = Xd[c]
        p = pd.Series(clf.predict_proba(Rd)[:, 1], index=Rd.index)
        if reg is not None:                             # blend: rank of P(touch) and rank of predicted swing
            mag = pd.Series(reg.predict(Rd), index=Rd.index)
            p = 0.5 * p.rank(pct=True) + 0.5 * mag.rank(pct=True)
            p = p * clf.predict_proba(Rd)[:, 1].max() if False else p
        # outcomes (read only for evaluation, after the week)
        o = pd.DataFrame({"prob": p, "up": up.loc[d].reindex(p.index), "dn": dn.loc[d].reindex(p.index),
                          "close": cl.loc[d].reindex(p.index)})
        for c in ("vol20", "atr_pct", "earn_in_week", "days_to_earn", "log_dv", "max20", "r5", "gap_today", "vol_surge1"):
            if c in Xd:
                o[c] = Xd[c].values
        o["date"] = str(d.date())
        rows.append(o.dropna(subset=["up"]))
    D = pd.concat(rows)
    if not save:
        return D
    wdir = OUT / (rid + v.get("tag", ""))
    wdir.mkdir(exist_ok=True)
    D.to_parquet(wdir / "weeks.parquet")
    res = summarize(D)
    res.update({"id": rid, "train_base_rate": base_rate_train, "variant": v, "seconds": round(time.time() - t0)})
    (wdir / "result.json").write_text(json.dumps(res, default=float))
    return res


def pick(D, n=10, rule=None):
    rule = rule or {}
    out = []
    for d, g in D.groupby("date"):
        if rule.get("earnings_only"):
            g = g[g["earn_in_week"] > 0] if (g["earn_in_week"] > 0).sum() >= n else g
        out.append(g.nlargest(n, "prob"))
    return pd.concat(out)


def summarize(D, n=10, rule=None):
    P = pick(D, n, rule)
    touch = ((P["up"] >= MOVE) | (P["dn"] <= -MOVE))
    closed = P["close"].abs() >= MOVE
    all_touch = ((D["up"] >= MOVE) | (D["dn"] <= -MOVE)).mean()
    return {"weeks": int(P["date"].nunique()), "picks": int(len(P)), "hit_touch": float(touch.mean()),
            "hit_close": float(closed.mean()), "base_rate_touch": float(all_touch),
            "weeks_all_10_right": float(touch.groupby(P["date"]).mean().eq(1).mean()),
            "up_touch": float((P["up"] >= MOVE).mean()), "down_touch": float((P["dn"] <= -MOVE).mean())}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        r = run_window(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
        print(json.dumps(r, default=float), flush=True)
    elif sys.argv[1] == "scramble":
        rid = sys.argv[2]
        a = run_window(rid, {"f64": True}, save=False)
        b = run_window(rid, {"f64": True}, scramble_after=120, save=False)
        dates = sorted(a["date"].unique())
        cut = str(run_window.cut_day.date())                # decisions on or before the last clean day
        pa = pick(a[a["date"] <= cut]); pb = pick(b[b["date"] <= cut])
        same = sorted(zip(pa["date"], pa.index)) == sorted(zip(pb["date"], pb.index))
        pa2 = pick(a[a["date"] > dates[-5]]); pb2 = pick(b[b["date"] > dates[-5]])
        print(f"{rid} future-scramble: picks before the cut identical: {same} | after the cut differ (sanity): "
              f"{sorted(zip(pa2['date'], pa2.index)) != sorted(zip(pb2['date'], pb2.index))}")
    elif sys.argv[1] == "report":
        tag = sys.argv[2] if len(sys.argv) > 2 else ""
        rs = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"m??{tag}/result.json"))]
        for r in rs:
            print(f"{r['id']}: hit (touched +/-10%) {r['hit_touch']:.1%} | closed +/-10% {r['hit_close']:.1%} | "
                  f"all stocks {r['base_rate_touch']:.1%} | weeks with all 10 right {r['weeks_all_10_right']:.0%}")
        if rs:
            print(f"AVERAGE over {len(rs)} windows: hit {np.mean([r['hit_touch'] for r in rs]):.1%} "
                  f"(closed {np.mean([r['hit_close'] for r in rs]):.1%}) vs all-stock base {np.mean([r['base_rate_touch'] for r in rs]):.1%}")
