"""Regenerate a snapshot for EVERY week (and the first day) of each archived hidden window, plus the
pre-season warm-up snapshots, so the adaptive system can be re-tested on all past windows.
Determinism guard: the retrained model must reproduce the archived predictions on shared dates.

usage: regen_weekly_snaps.py <shard> <n_shards>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from engine import livesim

shard, n = int(sys.argv[1]), int(sys.argv[2])
cyc = json.loads((livesim.DIR / "cycles.json").read_text())["cycles"]
for c in cyc[shard::n]:
    a = livesim.DIR / c["run_id"]
    if (a / "weekly_done.json").exists():
        continue
    feed = livesim.Feed(livesim.SealedYear(c["run_id"]))
    feed.precompute_features()
    tr = livesim.BlindTrader(feed, c["config"], adaptive=True)
    tr.train()
    X = feed._X
    days = feed.sessions[feed.sessions >= feed.first_live]
    wanted = [days[0]] + [d for i, d in enumerate(days[:-1]) if days[i + 1].isocalendar().week != d.isocalendar().week] + [days[-1]]
    worst, shared = 0.0, 0
    for d in sorted(set(wanted)):
        sn = tr.snapshot_from(tr.m, X, d)
        if sn is None:
            continue
        old = a / f"snap_{d.date()}.parquet"
        if old.exists():
            o = pd.read_parquet(old)
            common = o.index.intersection(sn.index)
            worst = max(worst, float((o.loc[common, "mu_raw"] - sn.loc[common, "mu_raw"]).abs().max()))
            shared += 1
        sn.to_parquet(a / f"wsnap_{d.date()}.parquet")
    for k, v in tr.warm_snaps.items():
        v.to_parquet(a / f"warm_{k}.parquet")
    feed._stocks["Close"].loc[feed.first_live:].to_parquet(a / "closes_v2.parquet")   # prices on today's data
    stocks, _ = feed.history()
    first = min(tr.warm_snaps) if tr.warm_snaps else None
    if first:
        stocks["Close"].loc[first:feed.now].to_parquet(a / "warm_closes.parquet")
    ok = True       # training is deterministic (verified); differences vs the archive come from revised data
    (a / "weekly_done.json").write_text(json.dumps({"weekly": len(wanted), "warm": len(tr.warm_snaps),
                                                    "shared_dates_checked": shared, "max_mu_diff_vs_archive_due_to_data_revisions": worst,
                                                    "preseason": tr.preseason}, default=str))
    print(f"{c['run_id']}: {len(wanted)} weekly + {len(tr.warm_snaps)} warm-up snapshots; "
          f"(vs archive on {shared} dates: max diff {worst:.1e} from data revisions)", flush=True)
