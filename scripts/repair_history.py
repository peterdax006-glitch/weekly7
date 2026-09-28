"""Re-download tickers whose stored history is entirely empty (Yahoo timeouts during bootstrap)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import data

for name in ("market", "stocks"):
    fr = data.load(name)
    empty = [c for c in fr["Close"].columns if fr["Close"][c].isna().all()]
    print(name, "empty:", len(empty))
    if not empty:
        continue
    new = data.download(empty, chunk=40, pause=2.0)
    got = [c for c in new["Close"].columns if new["Close"][c].notna().any()]
    print("recovered", len(got), "of", len(empty))
    for f in fr:
        fr[f] = fr[f].drop(columns=got, errors="ignore").join(new[f][got], how="left")
    data.save(fr, name)
