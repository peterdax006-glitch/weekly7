"""Forward minute-candle collector (canon C35/C38). Free sources keep ~7 days of 1-minute bars and ~60 days of
5-minute bars, so this saves them continuously; minute-level patterns accumulate from today onward.
Collects the most liquid names (plus anything the engine holds). Appends to data/intraday/<interval>/<date>.parquet.

Usage: python scripts/collect_intraday.py [N=400] [--recent DAYS] [--manifest PATH]
  --recent DAYS   only write the last DAYS trading days (the scheduled job uses 7; a weekly backfill uses 60)
  --manifest PATH write the list of files this run created or changed (the workflow uploads exactly those)
Only completed sessions are written: a run during market hours never saves a half day."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, yfinance as yf
from engine import config as K, data

OUT = K.DATA / "intraday"


def pick_tickers(n):
    st = data.load("stocks")
    C, V = st["Close"], st["Volume"]
    dv = (C * V).iloc[-20:].median().sort_values(ascending=False)
    held = []
    pos = K.STATE / "positions_meta.json"
    if pos.exists():
        import json
        try:
            held = list(json.loads(pos.read_text()).keys())
        except Exception as e:                                   # a corrupt positions file must not stop collection
            print("positions_meta unreadable:", e)
    return list(dict.fromkeys(held + list(dv.index[:n])))


def fetch(tickers, interval, period):
    parts = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period=period, interval=interval, progress=False, auto_adjust=False,
                                group_by="column", threads=True)
                break
            except Exception as e:
                print(f"retry {attempt + 1}/3:", e); time.sleep(5 * (attempt + 1))
        else:
            continue
        if d.empty:
            continue
        st = d.stack(level=1, future_stack=True)
        st.index.names = ["ts", "ticker"]
        parts.append(st.dropna(how="all"))
        time.sleep(1)
    if not parts:
        return None
    D = pd.concat(parts)
    D.index = D.index.set_levels(D.index.levels[0].tz_convert("America/New_York").tz_localize(None), level=0)
    return D


def completed_days(D, now_ny):
    """Session days that are over: before today, or today once the 16:00 close has passed."""
    days = sorted(D.index.get_level_values(0).normalize().unique())
    today = now_ny.normalize()
    return [d for d in days if d < today or (d == today and (now_ny.hour, now_ny.minute) >= (16, 15))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=400)
    ap.add_argument("--recent", type=int, default=0)
    ap.add_argument("--manifest", default="")
    a = ap.parse_args()
    tickers = pick_tickers(a.n)
    now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    written = []
    for interval, period in (("1m", "7d"), ("5m", "60d")):
        out = OUT / interval
        out.mkdir(parents=True, exist_ok=True)
        D = fetch(tickers, interval, period)
        if D is None:
            print(f"{interval}: no data returned"); continue
        days = completed_days(D, now_ny)
        if a.recent:
            days = days[-a.recent:]
        norm = D.index.get_level_values(0).normalize()
        for day in days:
            g = D[norm == day]
            f = out / f"{day.date()}.parquet"
            if f.exists():
                old = pd.read_parquet(f)
                g = pd.concat([old, g])
                g = g[~g.index.duplicated(keep="last")].sort_index()
                if len(g) == len(old):
                    continue                                     # nothing new for this day
            g.to_parquet(f)
            written.append(f)
        print(f"{interval}: {len(D):,} bars, {len(days)} completed days considered, "
              f"{sum(1 for w in written if w.parent.name == interval)} files written", flush=True)
    if a.manifest:
        Path(a.manifest).write_text("\n".join(str(w) for w in written) + ("\n" if written else ""))


if __name__ == "__main__":
    main()
