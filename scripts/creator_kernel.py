"""Run the Creator's self-development kernel on this repository.

    python scripts/creator_kernel.py --cycles 3                 # the Creator's OWN worker (local model, no other AI) - default
    python scripts/creator_kernel.py --worker session           # hand the package to the Claude session instead (only when the
                                                                # owner directs it; that adoption counts toward claude_dependence)

Owner, 1 Oct 2026: "get it to the point where the system knows the goal and then it can start working on building itself and then
you just direct". The default worker is creator.localworker.LocalWorker: the local model (llama.cpp + open weights in
~/creator_runtime) is started only when a package arrives. The kernel plans, measures, decides, merges or rejects; it never pushes."""
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
    ap.add_argument("--worker", choices=("local", "session"), default="local")
    ap.add_argument("--handoff-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    if a.worker == "session":
        worker: K.Worker = K.HandoffWorker(timeout_s=a.handoff_hours * 3600, notify=announce)
    else:
        from creator.localworker import LocalWorker                 # loaded only when the local worker is used
        worker = LocalWorker()
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
