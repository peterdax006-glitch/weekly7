"""More data for the Algorithm (canon C38): free macro history from FRED's public CSV endpoint (no API key).
Saved to data/cache/macro.parquet (daily, forward-filled only from PAST values when used)."""
import io, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, requests
from engine import config as K

SERIES = {
    "DGS10": "10-year Treasury yield", "DGS2": "2-year Treasury yield", "DGS3MO": "3-month Treasury yield",
    "T10Y2Y": "10y minus 2y (yield curve)", "T10Y3M": "10y minus 3m", "DFF": "Fed funds rate",
    "BAMLH0A0HYM2": "High-yield credit spread", "BAMLC0A0CM": "Investment-grade credit spread",
    "T10YIE": "10y breakeven inflation", "VIXCLS": "VIX (long history)", "DCOILWTICO": "WTI oil price",
    "DTWEXBGS": "Trade-weighted dollar", "UNRATE": "Unemployment rate", "CPIAUCSL": "CPI",
    "INDPRO": "Industrial production", "UMCSENT": "Consumer sentiment", "USREC": "Recession indicator",
    "NFCI": "Chicago Fed financial conditions", "STLFSI4": "St. Louis Fed financial stress", "GOLDAMGBD228NLBM": "Gold price",
}
frames, got = [], {}
for sid, name in SERIES.items():
    for attempt in range(3):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60,
                             headers={"User-Agent": "Weekly7 research"})
            if r.status_code == 200 and r.text.startswith(("observation_date", "DATE")):
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = ["date", sid]
                df["date"] = pd.to_datetime(df["date"])
                df[sid] = pd.to_numeric(df[sid], errors="coerce")
                frames.append(df.set_index("date"))
                got[sid] = (str(df["date"].min().date()), len(df))
                break
        except requests.RequestException:
            time.sleep(3)
    print(sid, got.get(sid, "FAILED"), flush=True)
M = pd.concat(frames, axis=1).sort_index()
M.to_parquet(K.CACHE / "macro.parquet")
print(f"saved {M.shape[1]} series, {len(M):,} dates, from {M.index.min().date()}")
