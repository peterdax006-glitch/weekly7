"""Daily bar store. Wide parquet frames (date x ticker) per field, split-and-dividend adjusted.
Source: yfinance (history + fallback). Alpaca replaces it for live bars once keys exist."""
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .config import CACHE, HISTORY_START

FIELDS = ["Open", "High", "Low", "Close", "Volume"]
MARKET = ["SPY", "QQQ", "IWM", "^VIX", "^VIX3M",
          "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
ET = ZoneInfo("America/New_York")
KEEP_PARTIAL = False   # live decide sets this; everything else drops unfinished bars


def _session_closed_today() -> bool:
    now = datetime.now(ET)
    return now.hour > 16 or (now.hour == 16 and now.minute >= 5)


def _drop_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Never let an unfinished intraday bar into a daily frame (look-ahead / noise)."""
    today = pd.Timestamp(datetime.now(ET).date())
    if len(df) and df.index[-1] >= today and not _session_closed_today() and not KEEP_PARTIAL:
        df = df[df.index < today]
    return df


def download(tickers, start=HISTORY_START, chunk=150, pause=1.0) -> dict:
    frames = {f: [] for f in FIELDS}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                d = yf.download(part, start=start, auto_adjust=True, threads=True,
                                progress=False, group_by="column")
                break
            except Exception as e:  # rate limit / transient
                print("retry", i, e); time.sleep(10 * (attempt + 1))
        else:
            continue
        if not isinstance(d.columns, pd.MultiIndex):
            d.columns = pd.MultiIndex.from_product([d.columns, part])
        for f in FIELDS:
            frames[f].append(d[f])
        print(f"  bars {min(i + chunk, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(pause)
    # repair pass: Yahoo times out silently -> retry any ticker that came back empty, in small batches
    got = pd.concat(frames["Close"], axis=1) if frames["Close"] else pd.DataFrame()
    missing = [t for t in tickers if t not in got.columns or got[t].isna().all()]
    if missing and len(missing) < len(tickers):
        print(f"  repair: retrying {len(missing)} empty tickers", flush=True)
        for i in range(0, len(missing), 20):
            part = missing[i:i + 20]
            time.sleep(2)
            try:
                d = yf.download(part, start=start, auto_adjust=True, threads=False, progress=False, group_by="column")
            except Exception:
                continue
            if not isinstance(d.columns, pd.MultiIndex):
                d.columns = pd.MultiIndex.from_product([d.columns, part])
            for f in FIELDS:
                frames[f].append(d[f].dropna(axis=1, how="all"))
    out = {}
    for f in FIELDS:
        w = pd.concat(frames[f], axis=1)
        # keep the non-empty copy when a ticker was re-downloaded
        w = w.loc[:, ~(w.isna().all() & w.columns.duplicated(keep=False))]
        w = w.loc[:, ~w.columns.duplicated()]
        w.index = pd.to_datetime(w.index).tz_localize(None)
        out[f] = _drop_partial(w.sort_index()).astype("float32")
    return out


def save(frames: dict, name: str):
    for f, w in frames.items():
        w.columns = w.columns.astype(str)
        w.to_parquet(CACHE / f"{name}_{f.lower()}.parquet")


def load(name: str) -> dict:
    return {f: pd.read_parquet(CACHE / f"{name}_{f.lower()}.parquet") for f in FIELDS}


def update(name: str, tickers=None):
    """Incremental refresh: re-pull the last 10 sessions and append (fixes late revisions)."""
    old = load(name)
    tickers = tickers or list(old["Close"].columns)
    start = (old["Close"].index[-1] - timedelta(days=15)).strftime("%Y-%m-%d")
    new = download(tickers, start=start)
    merged = {}
    for f in FIELDS:
        a, b = old[f], new[f]
        merged[f] = pd.concat([a[a.index < b.index[0]], b]).sort_index()
        merged[f] = merged[f].loc[:, ~merged[f].columns.duplicated()]
    save(merged, name)
    return merged
