"""Dashboard backtest panel: the live champion's 2017-2026 out-of-sample record vs the baselines."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import config as K, data, backtest
T = K.STATE / "research" / "tuning"
champ = pd.read_parquet(T / "eq_cfg_3_0.9_0.08_None_0.5_50000000.0_False_hivol_0.95.parquet").iloc[:, 0]
v12 = pd.read_parquet(T / "eq_CHAMPION_v1.2.parquet").iloc[:, 0]
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
out = {"weekly7": summ(champ), "evidence_only": summ(v12), "trend_chaser": summ(trend), "random_vol": summ(rnd), "spy": summ(spy),
       "note": "weekly7 = live v1.3; the 'Evidence-Only' row shows v1.2 (top-4, sector cap) for comparison"}
(K.STATE / "research" / "backtest_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
