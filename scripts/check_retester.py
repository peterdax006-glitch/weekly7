"""The re-tester must reproduce each live-clock run exactly, or adjustments are judged on a different system."""
import json, sys, glob, pandas as pd
sys.path.insert(0, ".")
src = open("scripts/livesim_cycle.py", encoding="utf-8").read()
src = src[:src.index("def worker(")]
ns = {"__file__": "scripts/livesim_cycle.py"}; sys.argv = ["x"]
exec(compile(src, "lc", "exec"), ns)
from engine import policy
bad = 0
for a in sorted(glob.glob("state/livesim/r*")):
    if not glob.glob(f"{a}/result.json"):
        continue
    res = json.load(open(f"{a}/result.json"))
    snaps = {f.split("snap_")[1][:-8]: pd.read_parquet(f) for f in sorted(glob.glob(f"{a}/snap_*.parquet"))}
    sc = pd.read_parquet(f"{a}/sic.parquet"); dv = {t: policy.sic_division(c) for t, c in zip(sc["ticker"], sc["sic"])}
    v = ns["replay_variant"](res["config"], snaps, pd.read_parquet(f"{a}/closes.parquet"), json.load(open(f"{a}/meta.json"))["cost_bps"], dv)
    live, re_ = res["diagnosis"]["year_return"], v["year_return"]
    ok = abs(live - re_) < 0.005
    bad += not ok
    print(a[-4:], f"live {live:+.4f}  re-tester {re_:+.4f}  {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
