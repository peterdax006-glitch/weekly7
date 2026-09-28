"""Dashboard backtest panel: the live champion's 2017-2026 out-of-sample record vs the baselines."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import config as K, data, backtest
T = K.STATE / "research" / "tuning"
champ = pd.read_parquet(T / "eq_CHAMPION_v1.2.parquet").iloc[:, 0]          # the live champion
P = pd.read_parquet(K.CACHE / "oos_preds.parquet", columns=["evidence"])
X = pd.read_parquet(K.CACHE / "panel.parquet", columns=["r20", "vol_surge5", "vol20"])
C = data.load("stocks")["Close"]
start = champ.index[0]
trend = backtest.baseline_trend(X, C, start=start)
rnd = backtest.baseline_random(X, C, start=start)
spy = data.load("market")["Close"]["SPY"].loc[start:]
spy = spy / spy.iloc[0] * 1000
def summ(e):
    w = backtest.weekly(e)
    return {"final": float(e.iloc[-1]), "mean_week": float(w.mean()), "pct_ge_7": float((w >= .07).mean()),
            "pct_le_m7": float((w <= -.07).mean()), "win_weeks": float((w > 0).mean()),
            "worst_week": float(w.min()), "max_dd": float((e / e.cummax() - 1).min())}
ev_only = backtest.baseline_score(P["evidence"], C, k=8, start=start, name="evidence_only")
out = {"weekly7": summ(champ), "evidence_only": summ(ev_only), "trend_chaser": summ(trend), "random_vol": summ(rnd), "spy": summ(spy),
       "note": "weekly7 = live champion v1.2; evidence_only = top-8 by the literature prior, weekly"}
(K.STATE / "research" / "backtest_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
