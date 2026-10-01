"""Measure the non-AI SearchSolver on the DEV split only: solve rate by category."""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creator import devbench as D  # noqa: E402
from creator import generator as G  # noqa: E402


def main() -> int:
    if "--baseline" in sys.argv:                      # measure the pre-targeted search for the before/after comparison
        G.targeted_mutations = lambda tree, per_family=40: iter(())  # type: ignore[assignment]
    manifest = D.load_manifest()
    tasks = [t for t in D.load_tasks() if t.split == "dev"]
    with ThreadPoolExecutor(4) as ex:
        scores = []
        for sc in ex.map(lambda t: D.run_task(t, G.SearchSolver(rank="--no-rank" not in sys.argv), manifest), tasks):
            scores.append(sc)
            print(f"{sc.task_id} {sc.category} {sc.outcome}", flush=True)
    by: dict[str, list[bool]] = defaultdict(list)
    for s in scores:
        by[s.category].append(s.outcome == "SOLVED")
    for c in sorted(by):
        print(f"{c:20s} {sum(by[c]):3d}/{len(by[c]):3d}")
    tot = sum(s.outcome == "SOLVED" for s in scores)
    print(f"{'TOTAL':20s} {tot:3d}/{len(scores):3d} = {tot / max(1, len(scores)):.3f}")
    for s in scores:
        if s.outcome != "SOLVED":
            print("  unsolved", s.task_id, s.category, s.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
