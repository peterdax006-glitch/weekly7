"""Run N recursion steps of the Creator on the REAL workload (devbench DEV split only; C77 sec 53-54, CR196-198).

    python scripts/creator_recurse.py --steps 2 --seed 1
Each step finds a weakness in the recursion ledger (or, with --probe, tries the next untried process parameter), A/B tests the
changed process on a bounded subset of dev tasks, adopts it only on IMPROVEMENT, and writes the process in force to
state/creator/process.json, which scripts/creator_swarm.py make_worker() reads. A later step uses seed+iteration, so it runs under
different conditions. The devbench holdout is never touched. Never pushes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import process_levers as PL  # noqa: E402
from creator import recursion as R  # noqa: E402
from creator.ledger import Ledger  # noqa: E402

STATE = ROOT / "state" / "creator"


def probe_change(process: R.ProcessConfig, tried: set[tuple[str, int]]) -> R.Change | None:
    for param in R.PARAMS:
        cur = getattr(process, param)
        if cur + 1 <= R.ME.BOUNDS[param][1] and (param, cur + 1) not in tried:
            return R.Change(param, cur, cur + 1, f"probe: raise {param} {cur} -> {cur + 1}")
    return None


def tune_worker(a: argparse.Namespace) -> int:
    """K19: propose and measure single-field changes of the own worker's configuration (creator.autotune.trial: devbench dev
    replicates + frozen holdout); an IMPROVEMENT is written to state/creator/worker_config.json, which creator.process_levers
    .build_worker (the swarm's worker) reads. Needs the local model server."""
    from creator import autotune as AT
    from creator import generator as G
    led = Ledger(a.ledger, evidence_root=a.ledger.parent)
    with G.LocalModel() as llm:
        for _ in range(a.tune):
            t = AT.trial(led, llm)
            if t is None:
                break
            print(json.dumps({"tune": t.field, "verdict": t.verdict, "adopted": t.adopted, "seconds": t.seconds}), flush=True)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--confirm", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="when the ledger shows no weakness, try the next untried parameter")
    ap.add_argument("--tune", type=int, default=0, help="run N creator.autotune worker-configuration trials instead of process steps")
    ap.add_argument("--ledger", type=Path, default=STATE / "recursion_ledger.jsonl")
    ap.add_argument("--process-file", type=Path, default=PL.DEFAULT_PATH)
    a = ap.parse_args(argv)
    if a.tune:
        return tune_worker(a)
    led = Ledger(a.ledger, evidence_root=a.ledger.parent)
    for _ in range(a.steps):
        it = len(R.recorded_steps(led)) + 1
        wl = PL.DevWorkload(seed=a.seed + it - 1, chunk=a.chunk, reps=a.reps, confirm=a.confirm)
        forced = None
        if a.probe and not any(R.design_change(w, R.current_process(led), R.tried_changes(led)) for w in R.find_weaknesses(led)):
            forced = probe_change(R.current_process(led), R.tried_changes(led))
        rep = R.step(led, wl, forced=forced)
        PL.save_process(R.current_process(led), a.process_file)
        print(json.dumps({"iteration": rep.iteration, "seed": a.seed + it - 1, "change": rep.change and rep.change.rationale,
                          "verdict": rep.verdict, "adopted": rep.adopted, "process": rep.process_after.__dict__,
                          "detail": {k: v for k, v in (rep.detail or {}).items() if k in ("diff", "se", "why", "holdout_diff")},
                          "task_runs": wl.runs}), flush=True)
        if rep.change is None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
