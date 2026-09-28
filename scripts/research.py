"""Full offline research run: features -> walk-forward models -> backtests -> report JSON.
This is the Part H1 ship gate and the first entry in the experiment registry (Part M5)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

import os
from engine import config as K, data, features, model, backtest
START = os.environ.get("W7_PANEL_START", "2013-01-01")          # extended run: 2002-01-01
FIRST = int(os.environ.get("W7_FIRST_YEAR", "2017"))            # extended run: 2008
SUFFIX = os.environ.get("W7_SUFFIX", "")                         # extended run: _ext

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

stocks, market = data.load("stocks"), data.load("market")
ev = pd.read_parquet(K.CACHE / "events.parquet")
sic = pd.read_parquet(K.CACHE / "sic.parquet")
ins_p = K.CACHE / "insider.parquet"
ins = pd.read_parquet(ins_p) if ins_p.exists() else None
log("loaded", stocks["Close"].shape, len(ev), "events", 0 if ins is None else len(ins), "insider rows")

X, atr = features.build(stocks, market, ev, ins, sic, start=START)
y_bar, fwd = features.labels(stocks, atr)
y = y_bar.stack(future_stack=True).reindex(X.index)
f = fwd.stack(future_stack=True).reindex(X.index)
log("panel", X.shape)
X.to_parquet(K.CACHE / f"panel{SUFFIX}.parquet")

P, last_models, iso = model.walk_forward(X, y, f, first_year=FIRST, log=log)
from engine import policy
P["score"] = policy.score(P)
P.to_parquet(K.CACHE / f"oos_preds{SUFFIX}.parquet")
ic_blend = model.daily_ic(P["score"], f)
ic_evid = model.daily_ic(P["evidence"], f)
ic_model = model.daily_ic(P["mu_raw"], f)
log(f"IC blend {ic_blend.mean():.4f} (t={ic_blend.mean() / ic_blend.std() * np.sqrt(len(ic_blend)):.1f})"
    f"  evidence {ic_evid.mean():.4f}  model {ic_model.mean():.4f}")

# calibration of P(+7% before stop) on held-out rows
cal = pd.DataFrame({"p": P["p_target"], "y": (y.reindex(P.index) == 1).astype(float)}).dropna()
cal["bin"] = pd.cut(cal["p"], [0, .05, .1, .15, .2, .3, .5, 1])
calib = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
log("calibration\n" + calib.to_string())

res = {"ic": {"blend": float(ic_blend.mean()), "evidence": float(ic_evid.mean()), "model": float(ic_model.mean()),
       "blend_t": float(ic_blend.mean() / ic_blend.std() * np.sqrt(len(ic_blend)))},
       "calibration": calib.reset_index().astype({"bin": str}).to_dict("records")}
(K.STATE / "research").mkdir(parents=True, exist_ok=True)
(K.STATE / "research" / f"model_summary{SUFFIX}.json").write_text(json.dumps(res, indent=2, default=float))
from engine.improve import log_experiment
log_experiment({"event": "walk_forward", "note": "v1.1: insider features, excess-return regressor, ranker retired, 2 priors off", **res["ic"]})
log(json.dumps(res["ic"]))
