"""Measure the system's OWN search worker (no AI) on the sealed HOLDOUT split - the tasks nobody tuned against.

Run by the orchestrator, never by a worker: the configuration is frozen first (devbench.freeze_config), results go to
state/creator/self_search_holdout.json. Compares the current search with the pre-push generic mutations (--baseline equivalent:
targeted families disabled) so a gain that only exists on the dev split shows up as overfitting."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import devbench as D  # noqa: E402
from creator import generator as G  # noqa: E402

OUT = ROOT / "state" / "creator" / "self_search_holdout.json"
FROZEN = ROOT / "state" / "creator" / "frozen_configs.json"


def run(targeted: bool, workers: int) -> dict:
    tasks = [t for t in D.load_tasks() if t.split == "holdout"]
    m = D.load_manifest()
    config = {"solver": "self-search", "targeted": targeted, "budget": 120}
    D.freeze_config(config, FROZEN)
    original = G.mutations
    if not targeted:
        G.mutations = G.generic_mutations                               # type: ignore[assignment]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(lambda t: D.run_task(t, G.SearchSolver(), m, solver_name="self-search"), tasks))
    finally:
        G.mutations = original                                          # type: ignore[assignment]
    suite = D.SuiteScore("holdout", "self-search", tuple(scores))
    return {"config": config, "summary": suite.summary(), "tasks": {s.task_id: s.outcome for s in scores}}


def main(argv: list[str]) -> int:
    workers = int(argv[0]) if argv else 4
    t0 = time.time()
    out: dict[str, dict] = {"targeted": run(True, workers), "generic_only": run(False, workers)}
    OUT.write_text(json.dumps({**out, "seconds": round(time.time() - t0)}, indent=1), encoding="utf-8")
    for k in ("generic_only", "targeted"):
        s = out[k]["summary"]
        print(k, f"{s['SOLVED']}/{s['n']} solved", "by category", s["by_category"], "false completions", s["FALSE_COMPLETION"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
