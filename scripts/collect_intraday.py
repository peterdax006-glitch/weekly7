"""Forward minute-candle collector (canon C35/C38). Free sources keep ~7 days of 1-minute bars and ~60 days of
5-minute bars, so this saves them continuously; minute-level patterns accumulate from today onward.
Collects the most liquid names (plus anything the engine holds). Appends to data/intraday/<interval>/<date>.parquet."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, yfinance as yf
from engine import config as K, data

OUT = K.DATA / "intraday"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
C, V = data.load("stocks")["Close"], data.load("stocks")["Volume"]
dv = (C * V).iloc[-20:].median().sort_values(ascending=False)
tickers = list(dv.index[:N])
for interval, period in (("1m", "7d"), ("5m", "60d")):
    out = OUT / interval
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        try:
            d = yf.download(chunk, period=period, interval=interval, progress=False, auto_adjust=False,
                            group_by="column", threads=True)
        except Exception as e:
            print("retry later:", e); time.sleep(5); continue
        if d.empty:
            continue
        st = d.stack(level=1, future_stack=True)
        st.index.names = ["ts", "ticker"]
        parts.append(st)
        time.sleep(1)
    if parts:
        D = pd.concat(parts)
        D.index = D.index.set_levels(D.index.levels[0].tz_convert("America/New_York").tz_localize(None), level=0)
        for day, g in D.groupby(D.index.get_level_values(0).normalize()):
            f = out / f"{day.date()}.parquet"
            if f.exists():
                g = pd.concat([pd.read_parquet(f), g])
                g = g[~g.index.duplicated(keep="last")]
            g.to_parquet(f)
        print(f"{interval}: {len(D):,} bars across {D.index.get_level_values(0).normalize().nunique()} days", flush=True)
