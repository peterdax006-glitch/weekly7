"""Phase 23 gate: the re-tester must reproduce each live-clock run within 0.5% (RELATIVE), or adjustments would be
judged on a different system.

For every archived run (state/livesim/r*/result.json):
  1. Code provenance FIRST. If the code the run was produced by has changed on disk since (engine.provenance.stale on the
     archived stamp, or files edited during the run), the verdict is STALE_CODE - not a parity failure - and nothing else
     is compared. A run archived before stamps existed is UNSTAMPED: it is still compared, reported as such, and neither
     counts as verified nor fails the gate (its code cannot be shown to be the code on disk).
  2. Otherwise engine.retester.compare_runs on the headline numbers (year return, mean week, weeks >= +7%, max drawdown)
     and, when the archive kept them, every weekly return; any divergence beyond 0.5% is a FAIL.

usage: check_retester.py [archive_dir=state/livesim] [run_glob=r*]
exit 0 = nothing failed and something was verified; 1 = at least one FAIL; 2 = nothing was verified (all stale/unstamped)."""
import json, sys, glob, pandas as pd
sys.path.insert(0, ".")
src = open("scripts/livesim_cycle.py", encoding="utf-8").read()
src = src[:src.index("def _heartbeat(")]
ns = {"__file__": "scripts/livesim_cycle.py"}; sys.argv_in, sys.argv = list(sys.argv), ["x"]
exec(compile(src, "lc", "exec"), ns)
from engine import policy, adaptive, retester as R


def replay_of(a):
    """Re-trade the archived year from its decision snapshots (no new information). With the archived opens
    (opens_v2.parquet, canon C33) this drives the SAME adaptive.Session the live run used, filling at the next open.
    Archives without opens fall back to the legacy close-fill replay_variant, which cannot reproduce a next-open run;
    those are labelled so and never count as verified."""
    res = json.load(open(f"{a}/result.json"))
    snaps = {f.split("snap_")[1][:-8]: pd.read_parquet(f) for f in sorted(glob.glob(f"{a}/snap_*.parquet"))}
    sc = pd.read_parquet(f"{a}/sic.parquet"); dv = {t: policy.sic_division(c) for t, c in zip(sc["ticker"], sc["sic"])}
    closes, bps = pd.read_parquet(f"{a}/closes.parquet"), json.load(open(f"{a}/meta.json"))["cost_bps"]
    if glob.glob(f"{a}/opens_v2.parquet"):
        S = adaptive.replay(res["config"], snaps, closes, bps, dv, opens=pd.read_parquet(f"{a}/opens_v2.parquet"))
        v = S.result()
        v["weekly_returns"] = {f"w{i:03d}": float(x) for i, x in enumerate(S.weeks)}
        return res, v, "session"
    return res, ns["replay_variant"](res["config"], snaps, closes, bps, dv), "legacy-close-fill"


if __name__ == "__main__":
    tally = {"PASS": 0, "FAIL": 0, "STALE_CODE": 0, "UNSTAMPED_PASS": 0, "UNSTAMPED_FAIL": 0}
    root, pat = (sys.argv_in[1] if len(sys.argv_in) > 1 else "state/livesim"), (sys.argv_in[2] if len(sys.argv_in) > 2 else "r*")
    for a in sorted(glob.glob(f"{root}/{pat}")):
        if not glob.glob(f"{a}/result.json"):
            continue
        res, v, how = replay_of(a)
        stamped, verdict = R.judge_archived(res, v)
        stamped = stamped and how == "session"                # a close-fill replay of a next-open run proves nothing
        live, re_ = res["diagnosis"]["year_return"], v["year_return"]
        label = verdict.status if stamped else f"UNSTAMPED_{verdict.status}"
        tally[label] += 1
        print(a[-4:], f"live {live:+.4f}  re-tester {re_:+.4f}  {label} [{how}]" + (f"  {verdict.summary()}" if label != "PASS" else ""))
    print("tally", json.dumps(tally))
    sys.exit(1 if tally["FAIL"] else 2 if tally["PASS"] == 0 else 0)
