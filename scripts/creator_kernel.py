"""Run the Creator's self-development kernel on this repository.

    python scripts/creator_kernel.py --cycles 3                 # self-first: the system's own workers, the Claude session last
    python scripts/creator_kernel.py --worker self-only         # never hand off (unattended runs)
    python scripts/creator_kernel.py --worker session           # straight to the Claude session

Owner, 1 Oct 2026: "it should work toward the level where it works on itself more than a third party works on it" (and: no local
LLM - it uses the Claude session for what it cannot do). Default worker: creator.selfworkers.SelfFirst - the system's own rule
transforms and test-guided search first, the Claude session only for what they could not do. self_share counts who did the work.
The kernel plans, measures, decides, merges or rejects; it never pushes."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import kernel as K  # noqa: E402

HANDOFF = ROOT / "state" / "creator" / "HANDOFF.json"


def announce(workdir: Path, package_id: str) -> None:
    HANDOFF.write_text(json.dumps({"package": package_id, "sandbox": str(workdir), "task": str(workdir / ".creator_task.md"),
                                   "answer_with": str(workdir / ".creator_done.json"),
                                   "at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
    print(f"HANDOFF {package_id} -> {workdir}", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--steps", default=",".join(K.P.WORKER_STEPS), help="comma list of requirement steps to plan")
    ap.add_argument("--mode", choices=("auto", "gaps", "efficiency"), default="auto")
    ap.add_argument("--worker", choices=("self", "self-only", "session"), default="self")
    ap.add_argument("--handoff-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    session = K.HandoffWorker(timeout_s=a.handoff_hours * 3600, notify=announce)
    if a.worker == "session":
        worker: K.Worker = session
    else:
        from creator import selfworkers as SW                       # loaded only when the system's own workers are used
        worker = SW.SelfFirst([SW.RuleWorker(), SW.SearchWorker()], None if a.worker == "self-only" else session)
    cfg = K.KernelConfig(repo=ROOT, state=ROOT / "state" / "creator", steps=tuple(s for s in a.steps.split(",") if s),
                         mode=a.mode)
    try:
        reps = K.run(cfg, worker, a.cycles, on_cycle=lambda r: print(json.dumps(dataclasses.asdict(r), default=str)[:3000],
                                                                       flush=True))
    finally:
        HANDOFF.unlink(missing_ok=True)
    print("SUMMARY", json.dumps(K.summary(reps)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
