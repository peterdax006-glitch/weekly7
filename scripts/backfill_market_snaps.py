"""Add the market-wide readings (m_*) to decision snapshots of already-played hidden years, rebuilt through each
year's own sealed feed (exactly what the blind system could see that day). Lets the re-tester try the
stress-rebound and trend-cash abilities on every past year."""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from engine import livesim

for a in sorted(glob.glob(str(livesim.DIR / "r*"))):
    a = Path(a)
    if not (a / "result.json").exists():
        continue
    snaps = sorted(a.glob("snap_*.parquet"))
    if snaps and any(c.startswith("m_") for c in pd.read_parquet(snaps[0]).columns):
        continue
    feed = livesim.Feed(livesim.SealedYear(a.name))
    feed.precompute_features()
    mcols = [c for c in feed._X.columns if c.startswith("m_")]
    for f in snaps:
        d = pd.Timestamp(f.stem[5:])
        p = pd.read_parquet(f)
        row = feed._X.xs(d, level=0)[mcols].iloc[0]
        for c in mcols:
            p[c] = float(row[c])
        p.to_parquet(f)
    print(a.name, "backfilled", len(snaps), "snapshots", flush=True)
