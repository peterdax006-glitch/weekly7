"""ONE real, budgeted agent call on one devbench dev task: proves the CLI flags, confinement and scoring work end to end."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import agents as A  # noqa: E402
from creator import devbench as D  # noqa: E402


def main(task_id: str = "D01") -> int:
    os.environ["CREATOR_AGENT_CALLS"] = "1"
    os.environ["CREATOR_AGENT_JOB_CALLS"] = "2"
    budget = A.Budget()
    solver = A.AgentSolver(budget, attempts=1)
    task = next(t for t in D.load_tasks() if t.id == task_id)
    if task.split != "dev":
        raise SystemExit("smoke runs only on the dev split")
    score = D.run_task(task, solver, D.load_manifest())
    run = solver.runs[-1] if solver.runs else None
    out = {"task": task_id, "outcome": score.outcome, "claimed": score.claimed_done, "hidden": score.hidden.__dict__,
           "visible_after": score.visible_after.__dict__, "notes": score.notes, "seconds": score.seconds,
           "run": None if run is None else {k: getattr(run, k) for k in ("run_id", "outcome", "usd", "turns", "contamination")},
           "budget_today": budget.today()}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
