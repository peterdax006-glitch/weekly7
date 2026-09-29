"""Fast weekly frontier: equal-weight top-k portfolios with hysteresis, after costs.
Answers: how much of the ~0.2%/week edge survives trading, and what it takes to get +7% weeks."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K
from engine.improve import log_experiment

P = pd.read_parquet(K.CACHE / "oos_preds.parquet", columns=["mu_raw", "evidence"])
X = pd.read_parquet(K.CACHE / "panel.parquet", columns=["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering"])
C = pd.read_parquet(K.CACHE / "stocks_close.parquet")
fri = C.index[(C.index.to_series().shift(-1).dt.isocalendar().week != C.index.to_series().dt.isocalendar().week).values]
fri = fri[(fri >= P.index.get_level_values(0).min())]
Pf = P[P.index.get_level_values(0).isin(fri)]
Xf = X.reindex(Pf.index)
g = Pf.groupby(level=0)
score = 0.5 * g["mu_raw"].rank(pct=True) + 0.5 * Pf["evidence"]   # == policy.score
vr = Xf.groupby(level=0)["vol20"].rank(pct=True); mr = Xf.groupby(level=0)["max20"].rank(pct=True)
ok = (vr <= 0.9) & (mr <= 0.9) & (Xf["log_dv"] >= np.log1p(2e7)) & (Xf["ev_red_flag"] <= 0)
score = score[ok]
ret = (C.shift(-1, freq=None).reindex(C.index) * 0)  # placeholder
nxt = pd.Series(fri[1:], index=fri[:-1])
R = {}
for d, n in nxt.items():
    R[d] = (C.loc[n] / C.loc[d] - 1)

def run(k, exit_q, cost_bps, pick="top", start_cash=1000.0):
    held, val, rows, turn = [], start_cash, [], 0.0
    for d in nxt.index:
        s = score.xs(d, level=0) if d in score.index.get_level_values(0) else pd.Series(dtype=float)
        if s.empty: continue
        q = s.rank(pct=True)
        keep = [t for t in held if t in q.index and q[t] >= exit_q]
        pool = s.sort_values(ascending=False)
        if pick == "hivol":   # among the top 5% by score, prefer the more volatile (still <= 90th pct)
            top5 = pool.index[: max(k, int(len(pool) * 0.05))]
            v = Xf.loc[(d, slice(None)), "vol20"].droplevel(0).reindex(top5)
            pool = v.sort_values(ascending=False)
        new = [t for t in pool.index if t not in keep][: max(0, k - len(keep))]
        port = keep + new
        changed = len(set(port) ^ set(held)) / max(1, k)
        turn += changed / 2
        r = R[d].reindex(port).fillna(0).mean() - changed * cost_bps / 1e4
        val *= 1 + r
        rows.append((d, r)); held = port
    w = pd.Series(dict(rows))
    eq = (1 + w).cumprod() * start_cash
    yrs = len(w) / 52
    return {"k": k, "exit_q": exit_q, "pick": pick, "cost_bps": cost_bps, "final": round(eq.iloc[-1], 0),
            "cagr": round((eq.iloc[-1] / start_cash) ** (1 / yrs) - 1, 4), "mean_week": round(w.mean(), 5),
            "pct_ge_7": round((w >= 0.07).mean(), 4), "pct_le_m7": round((w <= -0.07).mean(), 4),
            "max_dd": round((eq / eq.cummax() - 1).min(), 3), "turnover_per_week": round(turn / len(w), 3)}

spy = market = pd.read_parquet(K.CACHE / "market_close.parquet")["SPY"]
sw = pd.Series({d: spy.loc[n] / spy.loc[d] - 1 for d, n in nxt.items()})
print("SPY", {"mean_week": round(sw.mean(), 5), "pct_ge_7": round((sw >= .07).mean(), 4), "final": round(1000 * (1 + sw).prod())})
out = []
for pick in ("top", "hivol"):
    for k in (4, 8, 15):
        for ex in (0.99, 0.95, 0.90, 0.80):
            r = run(k, ex, 10, pick); out.append(r); print(r, flush=True)
log_experiment({"event": "frontier", "rows": out})
(K.STATE / "research" / "frontier.json").write_text(json.dumps(out, indent=1, default=float))
