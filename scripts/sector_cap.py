"""Challenger: enforce the blueprint's 40% sector cap (<= 1 name per SIC division at k=4)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import config as K, data, backtest, policy
from engine.improve import log_experiment
stocks = data.load("stocks")
X = pd.read_parquet(K.CACHE / "panel.parquet", columns=["vol20", "max20", "log_dv", "ev_red_flag", "ev_offering"])
P = pd.read_parquet(K.CACHE / "oos_preds.parquet", columns=["mu_raw", "evidence"])
sic = pd.read_parquet(K.CACHE / "sic.parquet")
sec = {t: policy.sic_division(c) for t, c in zip(sic["ticker"], sic["sic"])}
s = policy.score(P)
out = {}
for name, kw in [("k4_brake8_champion", {}), ("k4_brake8_sectorcap1", {"sectors": sec, "max_per_sector": 1}),
                 ("k4_brake8_sectorcap2", {"sectors": sec, "max_per_sector": 2})]:
    e, st = backtest.run_topk(s, X, stocks, k=4, exit_q=0.8, brake=0.08, name=name, **kw)
    w = backtest.weekly(e)
    out[name] = {"final": round(float(e.iloc[-1])), "cagr": round(float((e.iloc[-1] / 1000) ** (52 / len(w)) - 1), 4),
                 "pct_ge_7": round(float((w >= .07).mean()), 4), "pct_le_m7": round(float((w <= -.07).mean()), 4),
                 "worst_week": round(float(w.min()), 4), "max_dd": round(float((e / e.cummax() - 1).min()), 3)}
    print(name, out[name], flush=True)
log_experiment({"event": "challenger_historical", "id": "sector_cap", "rows": out})
