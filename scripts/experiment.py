"""Run one named backtest variant on the cached out-of-sample predictions (Part M3 historical test).
Every run is appended to the experiment registry so the trial count stays honest."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import config as K, data, backtest, model
from engine.improve import log_experiment

VARIANTS = {
    "v9_goal_nostop": dict(score="new", objective="goal", vol_filter=True, stop_atr=50.0),
    "v10_goal_stop3": dict(score="new", objective="goal", vol_filter=True, stop_atr=3.0),
    "v11_growth_nostop": dict(score="new", objective="growth", vol_filter=True, stop_atr=50.0),
    "v12_goal_stop2": dict(score="new", objective="goal", vol_filter=True, stop_atr=2.0),
}
name = sys.argv[1]
t0 = time.time()
stocks, market = data.load("stocks"), data.load("market")
X = pd.read_parquet(K.CACHE / "panel.parquet")
P = pd.read_parquet(K.CACHE / "oos_preds.parquet")
sic = pd.read_parquet(K.CACHE / "sic.parquet")
g = P.groupby(level=0)
from engine import policy
new_score = policy.score(P)
start = P.index.get_level_values(0).min()
out = K.STATE / "research"; out.mkdir(parents=True, exist_ok=True)

def summ(e):
    w = backtest.weekly(e)
    return {"final": float(e.iloc[-1]), "weeks": int(len(w)), "mean_week": float(w.mean()), "median_week": float(w.median()),
            "pct_ge_7": float((w >= 0.07).mean()), "pct_le_m7": float((w <= -0.07).mean()), "win_weeks": float((w > 0).mean()),
            "worst_week": float(w.min()), "best_week": float(w.max()), "max_dd": float((e / e.cummax() - 1).min())}

if name == "controls":
    xr = X
    hot = (X["vol20"].groupby(level=0).rank(pct=True) > 0.9) | (X["max20"].groupby(level=0).rank(pct=True) > 0.9)
    res = {}
    for n, e in [("eq_new_top8", backtest.baseline_score(new_score.where(~hot.reindex(new_score.index).fillna(False)), stocks["Close"], start=start, name="eq_new_top8")),
                 ("eq_evidence_top8", backtest.baseline_score(P["evidence"], stocks["Close"], start=start, name="eq_evidence_top8")),
                 ("trend_chaser", backtest.baseline_trend(X, stocks["Close"], start=start)),
                 ("random_vol", backtest.baseline_random(X, stocks["Close"], start=start))]:
        res[n] = summ(e); e.to_csv(out / f"eq_{n}.csv")
    spy = market["Close"]["SPY"].loc[start:]; spy = spy / spy.iloc[0] * K.START_CASH
    res["spy"] = summ(spy); spy.to_csv(out / "eq_spy.csv")
else:
    v = VARIANTS[name]
    sc = new_score if v["score"] == "new" else P["score"]
    eq, dl = backtest.run(P, sc, X, stocks, market, sic, start=start, name=name, objective=v["objective"],
                          vol_filter=v["vol_filter"], stop_atr=v.get("stop_atr"), log=lambda *a: print(*a, flush=True))
    eq.to_csv(out / f"eq_{name}.csv"); dl.to_csv(out / f"dec_{name}.csv", index=False)
    res = {name: summ(eq) | {"cost_total": float(dl["cost"].sum()), "avg_names": float(dl["n"].mean()),
                             **{k: float(v) for k, v in backtest.run.stats.items()}}}
for k, r in res.items():
    log_experiment({"event": "historical_test", "variant": k, **r})
(out / f"res_{name}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), f"\n{time.time() - t0:.0f}s")
