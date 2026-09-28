import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from engine import config as K, data, backtest
stocks, market = data.load("stocks"), data.load("market")
X = pd.read_parquet(K.CACHE / "panel.parquet")
P = pd.read_parquet(K.CACHE / "oos_preds.parquet")
sic = pd.read_parquet(K.CACHE / "sic.parquet")
g = P.groupby(level=0)
sc = 0.5 * g["mu_raw"].rank(pct=True) + 0.5 * P["evidence"]
t = time.time()
eq, dl = backtest.run(P, sc, X, stocks, market, sic, start="2017-01-01", end="2017-06-30", n_scen=2000,
                      objective=sys.argv[1] if len(sys.argv) > 1 else "growth", vol_filter=True, log=lambda *a: None)
print("final", eq.iloc[-1], "days", len(eq), backtest.run.stats, f"{time.time()-t:.0f}s")
print(dl[["mode","n","need","p_target"]].describe(include="all").T)
print(dl["mode"].value_counts())
# gross (pre-cost) equity path check: what did the top-8 by score return equal-weight over the same period?
