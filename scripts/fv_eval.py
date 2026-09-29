"""Real-cache runner for the Find-volatility pipeline (Bible Phase 38 V1-V5; canons C23/C24/C33).

Loads a SEEDED ticker sample from data/cache/stocks_*.parquet (column subset, float32; never the whole cache), builds the
weekly point-in-time panel, walks forward (refit every `--refit` weeks from data that had finished), and writes the
ablation (M / M+D / M+D+E / M+D+S / M+D+E+S) with week-block bootstrap CIs, per-era and per-type breakdowns, the mover
report, the no-look-ahead scramble audit and provenance to state/research/fv/<tag>/.

usage: fv_eval.py [--tickers 350] [--seed 7] [--start 2010-01-01] [--end 2026-09-25] [--tag run1] [--refit 13]
                  [--min-train 156] [--refill] [--strict] [--no-short] [--audit] [--pool 30] [--wait-ram 20]
Run in the background; the walk-forward checkpoints after every block (--tag reuses it), so a kill costs minutes.
Never kill python by image name - only the PID this script printed."""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from engine import config as K
from engine import fv_pipeline as F

OUT = K.STATE / "research" / "fv"
NEED_GB = 2.5


def free_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


def wait_for_ram(minutes):
    """Rule 10: below 2.5 GB free, poll every 60 s and give up after `minutes`."""
    t0 = time.time()
    while free_gb() < NEED_GB:
        if time.time() - t0 > minutes * 60:
            print(f"gave up: only {free_gb():.2f} GB free after {minutes} min", flush=True)
            sys.exit(3)
        print(f"waiting for RAM: {free_gb():.2f} GB free", flush=True)
        time.sleep(60)


def load_sample(n, seed, start, end, min_days=600):
    """Seeded sample of tickers with enough traded days and a dollar-volume floor, columns read one field at a time."""
    C = pd.read_parquet(K.CACHE / "stocks_close.parquet")
    V = pd.read_parquet(K.CACHE / "stocks_volume.parquet")
    C, V = C.loc[start:end], V.loc[start:end]
    ok = (C.notna().sum() >= min_days) & ((C * V).median() >= 2e6) & (C.median() >= 3)
    pool = np.array(sorted(ok[ok].index))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False))
    bars = {"Close": C[pick].astype("float32"), "Volume": V[pick].astype("float32")}
    del C, V
    for k in ("Open", "High", "Low"):
        f = pd.read_parquet(K.CACHE / f"stocks_{k.lower()}.parquet", columns=pick).loc[start:end]
        bars[k] = f.astype("float32")
    return bars, pick, len(pool)


def load_calendar():
    """Earnings-filing calendar for gaprisk (point-in-time use is inside gaprisk); None if unavailable."""
    try:
        from engine.gaprisk import EventCalendar
        ev = pd.read_parquet(K.CACHE / "events.parquet")
        return EventCalendar(ev)
    except Exception as e:
        print("no event calendar:", repr(e)[:120], flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=350)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-09-25")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--refit", type=int, default=13)
    ap.add_argument("--min-train", type=int, default=156)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--refill", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--audit", action="store_true", help="also run the future-scramble audit (2 extra short runs)")
    ap.add_argument("--wait-ram", type=int, default=20)
    ap.add_argument("--boot", type=int, default=600)
    a = ap.parse_args()
    print("pid", os.getpid(), flush=True)
    wait_for_ram(a.wait_ram)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bars, tickers, n_pool = load_sample(a.tickers, a.seed, a.start, a.end)
    print(f"sample {len(tickers)} of {n_pool} eligible tickers, {len(bars['Close'])} sessions, "
          f"{sum(v.memory_usage().sum() for v in bars.values()) / 1e6:.0f} MB, free {free_gb():.1f} GB", flush=True)
    cfg = F.FVConfig(refit_every=a.refit, min_train_weeks=a.min_train, pool=a.pool, refill=a.refill, strict_movers=a.strict,
                     allow_short=not a.no_short, seed=a.seed)
    panel = F.build_panel(bars, cfg)
    print(f"panel {len(panel.X)} rows, {len(panel.dates)} decision weeks, features {panel.X.shape[1]}", flush=True)
    run = F.walk_forward(panel, cfg, load_calendar(), ckpt=out / "checkpoint.pkl", log=lambda m: print(m, flush=True))
    print(f"walk-forward done in {time.time() - t0:.0f}s: {len(run.cands)} candidate rows over {len(run.week_dates)} weeks", flush=True)

    tab, diffs = F.ablation(run, boot=a.boot, seed=a.seed)
    full = F.summarize(run, "MDES", boot=a.boot, seed=a.seed)
    verdict = F.goal_verdict(full)
    mover = F.mover_report(run, boot=a.boot, seed=a.seed)
    eras = F.era_table(run, "MDES")
    types = F.type_table(run, "MDES")
    extra = {"sample_tickers": tickers, "eligible_pool": n_pool, "fill_timing": F.audit_fill_timing(run),
             "seconds": round(time.time() - t0), "sample_seed": a.seed, "window": [a.start, a.end]}
    for name, spec in (("md_only", "MD"),):
        F.era_table(run, spec).to_csv(out / f"eras_{name}.csv", index=False)
    if a.audit:
        cut = panel.dates[a.min_train + a.refit + 20]
        sub = {k: v.iloc[: panel.sessions.get_loc(cut) + 1 + 200] for k, v in bars.items()}
        sub_cfg = replace(cfg, lgb_trees=60, pool=15)
        extra["scramble_audit"] = F.audit_no_lookahead(sub, sub_cfg, cut, last_block=1)
        print("scramble audit:", extra["scramble_audit"], flush=True)
    F.save_results(out, run, tab, diffs, mover, eras, types, verdict, extra)
    print((out / "report.md").read_text(encoding="utf-8"), flush=True)
    (out / "DONE").write_text(json.dumps({"seconds": extra["seconds"], "verdict": verdict}, default=float))


if __name__ == "__main__":
    main()
