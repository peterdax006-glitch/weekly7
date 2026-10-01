"""Run the Creator's self-development kernel on this repository, with the Claude session as its ONLY worker.

    python scripts/creator_kernel.py --cycles 1 [--steps integrated]

Owner, 1 Oct 2026: "the creator shouldn't hire a claude worker, you should be the only claude worker working on it". So the
kernel plans, opens a sandbox and HANDS the package over: it writes <sandbox>/.creator_task.md and announces the handoff in
state/creator/HANDOFF.json; the session implements the package inside that sandbox and writes <sandbox>/.creator_done.json
({"claimed_done": true, "notes": "..."}); the kernel then measures, decides, merges or rejects exactly as before. No LLM calls are
made by this process. The kernel never pushes."""
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
    ap.add_argument("--handoff-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    worker = K.HandoffWorker(timeout_s=a.handoff_hours * 3600, notify=announce)
    cfg = K.KernelConfig(repo=ROOT, state=ROOT / "state" / "creator", steps=tuple(s for s in a.steps.split(",") if s))
    try:
        reps = K.run(cfg, worker, a.cycles, on_cycle=lambda r: print(json.dumps(dataclasses.asdict(r), default=str)[:3000],
                                                                       flush=True))
    finally:
        HANDOFF.unlink(missing_ok=True)
    print("SUMMARY", json.dumps(K.summary(reps)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
