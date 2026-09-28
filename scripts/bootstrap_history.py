"""One-off: build the full daily-bar history for the universe and market series."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from engine.universe import load_universe
from engine import data

t = time.time()
data.save(data.download(data.MARKET), "market")
print("market done", round(time.time() - t))
u = load_universe()
u.to_csv(data.CACHE / "universe.csv", index=False)
data.save(data.download(u["ticker"].tolist()), "stocks")
print("stocks done", round(time.time() - t))
