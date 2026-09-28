"""Small end-to-end smoke test on ~40 tickers (no SEC data) to catch code errors fast."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import data, features, model, backtest

T = ("AAPL MSFT NVDA AMD TSLA AAL DAL UAL JPM BAC XOM CVX PFE MRNA GME AMC PLTR SOFI RIOT MARA "
     "F GM INTC MU CRM ORCL NFLX DIS NKE SBUX KO PEP WMT TGT HD LOW BA CAT DE UBER").split()
t = time.time()
stocks = data.download(T, start="2016-01-01")
market = data.download(data.MARKET, start="2016-01-01")
ev = pd.DataFrame({"ticker": ["AAPL", "AAL"], "form": ["8-K", "424B5"],
                   "accepted": pd.to_datetime(["2024-05-02T20:30:00Z", "2024-06-03T13:00:00Z"], utc=True),
                   "kind": ["EARN", "OFFERING"]})
sic = pd.DataFrame({"ticker": T, "sic": ["3571"] * 20 + ["4512"] * 20})
X, atr = features.build(stocks, market, ev, None, sic, start="2017-01-01")
yb, fw = features.labels(stocks, atr)
y = yb.stack(future_stack=True).reindex(X.index); f = fw.stack(future_stack=True).reindex(X.index)
print("panel", X.shape, "label balance", y.value_counts(normalize=True).round(3).to_dict(), round(time.time() - t))
print(X.isna().mean().sort_values(ascending=False).head(8).round(2).to_dict())
P, m, iso = model.walk_forward(X, y, f, first_year=2023)
P["score"] = model.blend(P)
print("IC", model.daily_ic(P["score"], f).mean())
eq, dl = backtest.run(P, P["score"], X, stocks, market, sic, n_scen=1000)
print(eq.tail(3)); print(dl["mode"].value_counts().to_dict()); print(backtest.weekly(eq).describe().round(4).to_dict())
