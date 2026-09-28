"""Extend the research history back to 2000 (canon C7: 'a random year from a random decade').
Bars 2000+, SEC events since 2001, DERA insider data 2006-2008 prepended. Survivorship bias
grows the further back we go (only today's listed companies) - reported with every result."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import config as K, data, edgar
t = time.time()
u = pd.read_csv(K.CACHE / "universe.csv")
m = data.download(data.MARKET, start="2000-01-01"); data.save(m, "market")
print("market", round(time.time() - t), flush=True)
s = data.download(u["ticker"].tolist(), start="2000-01-01"); data.save(s, "stocks")
print("stocks", s["Close"].shape, round(time.time() - t), flush=True)
ev = edgar.fetch_events(u, since="2001-01-01")
print("events", len(ev), round(time.time() - t), flush=True)
edgar.fix_activist(u, since="2001-01-01")
print("13D fixed", round(time.time() - t), flush=True)
old = pd.read_parquet(K.CACHE / "insider.parquet")
early = edgar.fetch_insider(first_year=2006, last=(2008, 4))
pd.concat([early, old]).drop_duplicates(["acc", "owner_cik", "tdate", "shares"]).to_parquet(K.CACHE / "insider.parquet")
print("insider", round(time.time() - t), flush=True)
