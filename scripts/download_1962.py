"""Price history 1962-1999 for today's universe (survivors) + S&P 500 index as the pre-1993 market."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import config as K, data
t = time.time()
u = pd.read_csv(K.CACHE / "universe.csv")
old = data.download(u["ticker"].tolist(), start="1962-01-01", chunk=150)
for f in old:
    old[f] = old[f].loc[: "1999-12-31"].dropna(axis=1, how="all")
data.save(old, "stocks_pre2000")
mk = data.download(["^GSPC", "^VIX", "^IRX", "^TNX"], start="1962-01-01")
data.save(mk, "index_hist")
print("done", old["Close"].shape, round(time.time() - t), flush=True)
