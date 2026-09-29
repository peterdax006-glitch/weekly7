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
DROP = ("ins_", "ev_activist")                      # insider data is off in sims (C18 fail-safe)


def labels(stocks):
    C, Hh, L = stocks["Close"], stocks["High"], stocks["Low"]
    hi = pd.concat([Hh.shift(-k) for k in range(1, H + 1)]).groupby(level=0).max()
    lo = pd.concat([L.shift(-k) for k in range(1, H + 1)]).groupby(level=0).min()
    up, dn = hi / C - 1, lo / C - 1
    close = C.shift(-H) / C - 1
    return up, dn, close


def week_ends(idx):
    return [d for i, d in enumerate(idx[:-1]) if idx[i + 1].isocalendar().week != d.isocalendar().week]


def run_window(rid, variant=None):
    v = {"n_estimators": 300, "num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.05, **(variant or {})}
    t0 = time.time()
    feed = livesim.Feed(livesim.SealedYear(rid))
    feed.precompute_features()
    X = feed._X
    cols = [c for c in X.columns if not c.startswith(DROP)]
    S = feed._stocks
    up, dn, cl = labels(S)
    touch = ((up >= MOVE) | (dn <= -MOVE)).astype(float)
    sessions = feed.sessions
    first = feed.first_live
    # ---- train on warm-up week-ends whose 5-day label window closes before the hidden window ----
    warm = [d for d in week_ends(sessions[sessions < first])]
    cutoff = sessions[max(0, sessions.get_loc(first) - H - 1)]
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
                             random_state=7, verbose=-1, **v)
    clf.fit(Rt, y[ok])
    base_rate_train = float(y[ok].mean())
    # ---- hidden window: predict each week, evaluate after the week ----
    rows = []
    for d in week_ends(sessions[sessions >= first]):
        Xd = X.xs(d, level=0)[cols]
        Rd = Xd.rank(pct=True)
        for c in cols:
            if c.startswith("m_"):
                Rd[c] = Xd[c]
        p = pd.Series(clf.predict_proba(Rd)[:, 1], index=Rd.index)
        # outcomes (read only for evaluation, after the week)
        o = pd.DataFrame({"prob": p, "up": up.loc[d].reindex(p.index), "dn": dn.loc[d].reindex(p.index),
                          "close": cl.loc[d].reindex(p.index)})
        for c in ("vol20", "atr_pct", "earn_in_week", "days_to_earn", "log_dv", "max20", "r5", "gap_today", "vol_surge1"):
            if c in Xd:
                o[c] = Xd[c].values
        o["date"] = str(d.date())
        rows.append(o.dropna(subset=["up"]))
    D = pd.concat(rows)
    wdir = OUT / rid
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
    elif sys.argv[1] == "report":
        rs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/result.json"))]
        for r in rs:
            print(f"{r['id']}: hit (touched +/-10%) {r['hit_touch']:.1%} | closed +/-10% {r['hit_close']:.1%} | "
                  f"all stocks {r['base_rate_touch']:.1%} | weeks with all 10 right {r['weeks_all_10_right']:.0%}")
        if rs:
            print(f"AVERAGE over {len(rs)} windows: hit {np.mean([r['hit_touch'] for r in rs]):.1%} "
                  f"(closed {np.mean([r['hit_close'] for r in rs]):.1%}) vs all-stock base {np.mean([r['base_rate_touch'] for r in rs]):.1%}")
