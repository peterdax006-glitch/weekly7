"""Does the weekly +7% machinery (bank / brake) help the champion constructor? Daily simulation."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from engine import config as K, data, backtest, policy
from engine.improve import log_experiment
stocks = data.load("stocks")
X = pd.read_parquet(K.CACHE / "panel.parquet", columns=["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering"])
P = pd.read_parquet(K.CACHE / "oos_preds.parquet", columns=["mu_raw", "evidence"])
s = policy.score(P)
out = {}
def summ(e):
    w = backtest.weekly(e)
    return {"final": round(float(e.iloc[-1])), "cagr": round(float((e.iloc[-1] / 1000) ** (52 / len(w)) - 1), 4),
            "mean_week": round(float(w.mean()), 5), "pct_ge_7": round(float((w >= .07).mean()), 4),
            "pct_le_m7": round(float((w <= -.07).mean()), 4), "max_dd": round(float((e / e.cummax() - 1).min()), 3)}
for name, kw in [("k4_plain", {}), ("k4_bank7", {"bank": 0.07}), ("k4_brake8", {"brake": 0.08}),
                 ("k4_bank7_brake8", {"bank": 0.07, "brake": 0.08}), ("k3_plain", {"k": 3}), ("k5_plain", {"k": 5})]:
    e, st = backtest.run_topk(s, X, stocks, **{"k": 4, "exit_q": 0.8, **kw}, name=name)
    out[name] = summ(e) | {k: round(float(v), 2) for k, v in st.items()}
    print(name, out[name], flush=True)
    e.to_csv(K.STATE / "research" / f"eq_{name}.csv")
log_experiment({"event": "topk_rules", "rows": out})
(K.STATE / "research" / "topk_rules.json").write_text(json.dumps(out, indent=1))
