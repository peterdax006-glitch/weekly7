"""Measure the Creator's OWN worker (local model + search, no other AI) on the sealed devbench, and whether its experience helps.

    1. holdout, no experience          (baseline; configuration frozen first)
    2. dev split, learning ON          (experience recorded from dev tasks only)
    3. holdout, using dev experience   (learning OFF - holdout tasks are never recorded)
Writes state/creator/generator_bench.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import devbench as D  # noqa: E402
from creator import generator as G  # noqa: E402

OUT = ROOT / "state" / "creator" / "generator_bench.json"
FROZEN = ROOT / "state" / "creator" / "frozen_configs.json"


def main() -> int:
    m = D.load_manifest()
    tasks = D.load_tasks()
    run_id = time.strftime("%Y%m%dT%H%M%S")
    mem = G.ExperienceMemory(ROOT / "state" / "creator" / f"generator_memory_{run_id}.jsonl")
    out: dict = {"run": run_id, "model": str(G.DEFAULT_MODEL.name), "phases": {}}
    t0 = time.time()
    with G.LocalModel() as llm:
        def phase(name: str, solver: G.GeneratorSolver, split: str) -> None:
            if split == "holdout":
                D.freeze_config(solver.config() | {"phase": name}, FROZEN)
                sc = D.score_holdout(solver, solver.config() | {"phase": name}, FROZEN, solver.name, tasks=tasks, manifest=m)
            else:
                sc = D.score_dev(solver, solver.name, tasks=tasks, manifest=m)
            out["phases"][name] = {"summary": sc.summary(), "tasks": {s.task_id: s.outcome for s in sc.scores},
                                   "log": solver.log}
            print(name, json.dumps(sc.summary()), flush=True)
            OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        phase("1_holdout_no_experience", G.GeneratorSolver(llm, memory=None, learn=False), "holdout")
        phase("2_dev_learning", G.GeneratorSolver(llm, memory=mem, learn=True), "dev")
        phase("3_holdout_with_experience", G.GeneratorSolver(llm, memory=mem, learn=False), "holdout")
        out["model_calls"], out["model_seconds"] = llm.calls, round(llm.seconds, 1)
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("DONE", out["seconds"], "s", out["model_calls"], "model calls", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
