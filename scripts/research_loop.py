"""Runnable entry of the autonomous research loop (contract C66 sections 3, 47, 50, 51; canon C63, C64, C66, C67).
IMPLEMENTED - NOT VALIDATED: under C63 only the planted source has been run; a real-data run is the next wave's job.

    python scripts/research_loop.py --once                           one cycle on a planted world (default source)
    python scripts/research_loop.py --cycles 20 --mode thread         twenty cycles, experiments in a worker pool
    python scripts/research_loop.py --forever --wall-hours 10         keep researching until the feed or the wall budget ends
    python scripts/research_loop.py --source frame --frame F.pkl      a prepared lab frame (volatility_lab.frame_from_panel layout)
    python scripts/research_loop.py --source world --cycles 10        the W02 planted world: every stage fed from bars and events
    python scripts/research_loop.py --status                          print the newest checkpoint's cycle report and exit

Everything is written under --root (default state/research/research_loop/<run_id>/): checkpoints (resumable: the loop continues
from the next unfinished stage after a kill; stale code is refused unless --allow-code-change), per-cycle reports, the compute
ledger and isolated experiment folders, and a provenance stamp. A real-data source waits for >= 2.5 GB free RAM (CONTEXT rule 10)
before it loads anything."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import config as K                                                    # noqa: E402
from engine.research import error_loop as C68                                     # noqa: E402,F401 - registers the C68 stages
from engine.research import loop as LP                                            # noqa: E402
from engine.research import two_stage as TS                                       # noqa: E402

DEFAULT_ROOT = K.STATE / "research" / "research_loop"


def wait_for_memory(min_gb: float = 2.5, poll_s: float = 60.0, give_up_s: float = 1200.0) -> float:
    """CONTEXT rule 10: poll free RAM every minute, give up after twenty (the shared machine runs long experiments)."""
    import psutil
    t0 = time.monotonic()
    while True:
        free = psutil.virtual_memory().available / 1e9
        if free >= min_gb:
            return free
        if time.monotonic() - t0 > give_up_s:
            raise SystemExit(f"only {free:.1f} GB free after {give_up_s / 60:.0f} min of waiting (< {min_gb} GB): not starting")
        print(f"[research_loop] {free:.1f} GB free < {min_gb} GB; waiting", flush=True)
        time.sleep(poll_s)


def build_feed(a: argparse.Namespace):
    if a.source == "world":                                   # W02: the rich planted world, every stage fed from bars/events
        return LP.world_feed("planted")
    if a.source == "real":                                    # W02 real-cache adapter (C63: only once the foundation allows it)
        wait_for_memory()
        return LP.world_feed("real", years=tuple(range(a.sweep_from, a.sweep_to + 1)), sample=a.real_sample)
    if a.source == "planted":
        from engine.research import volatility_lab as VL
        F = VL.planted_frame(a.planted_truth, n_dates=a.planted_dates, n_tickers=a.planted_names, seed=a.seed, effect=a.planted_effect,
                             dir_signal=a.planted_direction)
    elif a.source == "frame":
        if not a.frame:
            raise SystemExit("--source frame needs --frame <path to a pickled lab frame>")
        wait_for_memory()
        import pandas as pd
        F = pd.read_pickle(a.frame)
    else:
        raise SystemExit(f"unknown source {a.source!r}")
    import pandas as pd
    dates = sorted(pd.unique(F.index.get_level_values(0)))
    start = max(0, min(len(dates) - 1, a.warmup_dates))
    return LP.FrameFeed(F, dates=dates[start:])


def build_sweeps(a: argparse.Namespace, feed=None) -> list:
    """C67 always-on sweeps over the real caches (only with --sweeps; they read data, so the RAM rule applies first). A W02 world
    feed brings its own point-in-time precursor sweep."""
    if not a.sweeps:
        return []
    if hasattr(feed, "sweeps"):
        return feed.sweeps()
    wait_for_memory()
    from engine.research import precursors as PC
    years = tuple(range(a.sweep_from, a.sweep_to + 1))
    return [LP.precursor_sweep(years, PC.cache_loader(a.sweep_slices, a.seed))]


def config(a: argparse.Namespace) -> LP.LoopConfig:
    return LP.LoopConfig(run_id=a.run_id, seed=a.seed, budget_cpu_min=a.budget_cpu, period_cpu_min=a.period_cpu, mode=a.mode,
                         max_workers=a.workers, checkpoint=a.checkpoint, allow_code_change=a.allow_code_change,
                         disabled=tuple(a.disable or ()), sweep_units=a.sweep_units,
                         two_stage=TS.TwoStageConfig(gate_min_weeks=a.gate_min_weeks))


def record_run(a: argparse.Namespace, summary: dict, root: Path) -> dict:
    """B12 (INTEGRATION): the finished loop leaves a write-once checkpoint bundle (checkpoint.write_checkpoint) and a Phase 36
    run report (run_report.build_report / write_report). The loop produces knowledge, not a weekly return series, so the
    report's Tier 1-3 fields stay null and its decision is CONTINUE TESTING - it is a record, never an adoption."""
    from engine.backtest import record_major_run
    try:
        out = record_major_run("research_loop", a.run_id, {k: v for k, v in vars(a).items()}, summary.get("counters", {}), summary,
                               a.seed, unproven=["research loop: no weekly return series; Tier 1-3 are not applicable"],
                               logs={"summary.json": (root / "summary.json").read_text(encoding="utf-8")},
                               ckpt_root=a.checkpoint_root or None, report_dir=a.report_dir or None)
    except Exception as e:                                  # noqa: BLE001 - the loop's own outputs are already on disk
        out = {"error": f"{type(e).__name__}: {e}"}
    (root / "run_record.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"[research_loop] run record: {out}", flush=True)
    return out


def status(root: Path) -> int:
    reps = sorted((root / "reports").glob("cycle_*.json"))
    if not reps:
        print(f"no cycle reports under {root}")
        return 1
    rep = json.loads(reps[-1].read_text(encoding="utf-8"))
    print(LP.render_report(rep))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default="")
    ap.add_argument("--run-id", default="research")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true")
    g.add_argument("--forever", action="store_true")
    g.add_argument("--cycles", type=int, default=0)
    g.add_argument("--status", action="store_true")
    ap.add_argument("--wall-hours", type=float, default=0.0)
    ap.add_argument("--source", default="planted", choices=("planted", "frame", "world", "real"))
    ap.add_argument("--frame", default="")
    ap.add_argument("--planted-truth", default="H1")
    ap.add_argument("--planted-dates", type=int, default=120)
    ap.add_argument("--planted-names", type=int, default=80)
    ap.add_argument("--planted-effect", type=float, default=1.3)
    ap.add_argument("--planted-direction", type=float, default=0.0)
    ap.add_argument("--warmup-dates", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-cpu", type=float, default=240.0)
    ap.add_argument("--period-cpu", type=float, default=1200.0)
    ap.add_argument("--mode", default="inline", choices=("inline", "thread", "process"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--checkpoint", default="stage", choices=("stage", "cycle", "off"))
    ap.add_argument("--allow-code-change", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints and start a new run")
    ap.add_argument("--disable", nargs="*")
    ap.add_argument("--gate-min-weeks", type=int, default=26)
    ap.add_argument("--sweeps", action="store_true")
    ap.add_argument("--sweep-units", type=int, default=1)
    ap.add_argument("--sweep-slices", type=int, default=8)
    ap.add_argument("--real-sample", type=int, default=300)
    ap.add_argument("--sweep-from", type=int, default=1990)
    ap.add_argument("--sweep-to", type=int, default=2020)
    ap.add_argument("--checkpoint-root", default="", help="checkpoint bundles (default state/checkpoints)")
    ap.add_argument("--report-dir", default="", help="run reports (default state/reports/runs)")
    ap.add_argument("--no-record", action="store_true", help="skip the B12 checkpoint bundle and run report")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else DEFAULT_ROOT / a.run_id
    if a.status:
        return status(root)
    cfg = config(a)
    root.mkdir(parents=True, exist_ok=True)
    from engine.learning import wiring                    # hub sinks persist under state/learning/hub/<lane>/ (F03, C69 ledger)
    wiring.configure_from_env(a.run_id)
    try:
        from engine import provenance
        stamp = provenance.stamp(vars(a), a.seed)
    except Exception as e:                                  # noqa: BLE001 - provenance failure is recorded, never silent
        stamp = {"error": f"{type(e).__name__}: {e}"}
    (root / "provenance.json").write_text(json.dumps({"args": vars(a), "stamp": stamp, "label": LP.LABEL}, indent=1, default=str),
                                          encoding="utf-8")
    feed = build_feed(a)
    max_cycles = None if a.forever else (1 if a.once or not a.cycles else a.cycles)
    t0 = time.monotonic()
    state, reports = LP.run(feed, root, cfg, max_cycles=max_cycles, sweeps=build_sweeps(a, feed), fresh=a.fresh,
                            wall_budget_s=a.wall_hours * 3600 if a.wall_hours else None)
    for rep in reports:
        print(LP.render_report(rep), flush=True)
    summary = {"cycles": len(reports), "seconds": round(time.monotonic() - t0, 1), "questions": len(state.questions),
               "knowledge": len(state.knowledge), "section47_chains": len(state.lineage.section47_chains()),
               "knowledge_chains": len(state.lineage.knowledge_chains()), "counters": {k: v for k, v in state.counters.items()
                                                                                   if not k.startswith("_")}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, default=str))
    if not a.no_record:
        record_run(a, summary, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
