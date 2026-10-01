"""Run the Creator as a swarm: keep adding the system's own workers while this computer has free RAM, pull them back when it gets
tight, and send what needs thinking to the Claude session one package at a time.

    python scripts/creator_swarm.py --rounds 0                  # 0 = keep going round after round
Owner, 1 Oct 2026: "keep adding agents to work on it until the memory capcity of this device is full" (with a safety margin: the
host kills processes when RAM is critically low), and "I want you to do the thinking until the ai/system is capable of doing it
for itself" - the session handoff is serialised (one thinker), the own workers go first. Announces handoffs in
state/creator/HANDOFF.json; heartbeat in state/creator/swarm_heartbeat.json. Never pushes."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import kernel as K  # noqa: E402
from creator import process_levers as PL  # noqa: E402
from creator import selfworkers as SW  # noqa: E402
from creator import swarm as W  # noqa: E402

STATE = ROOT / "state" / "creator"
HANDOFFS = STATE / "handoffs"                            # one file per package waiting for Claude (all at once, 2 Oct)
LOG = STATE / "swarm_log.jsonl"


def announce(workdir: Path, package_id: str) -> None:
    HANDOFFS.mkdir(parents=True, exist_ok=True)
    (HANDOFFS / f"{package_id}.json").write_text(json.dumps(
        {"package": package_id, "sandbox": str(workdir), "task": str(workdir / ".creator_task.md"),
         "answer_with": str(workdir / ".creator_done.json"), "at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1),
        encoding="utf-8")


class SerialSession:
    """The Claude session as the last worker. Owner, 2 Oct: 'if we have the available memory then make sure to have all the
    projects that need to be handled by you as claude running' - every package that needs Claude is handed off AT ONCE (one
    file each in state/creator/handoffs/), never queued behind the others."""
    name = "claude-session"

    def __init__(self, hours: float) -> None:
        self.inner = K.HandoffWorker(timeout_s=hours * 3600, notify=announce)

    def __call__(self, plan, package, workdir):                       # type: ignore[no-untyped-def]
        with W.waiting_on_thinker(plan.package_id):                       # waiting for me holds no worker slot
            try:
                res = self.inner(plan, package, workdir)
                if res.claimed_done:
                    W.HANDED_BACK.add(plan.package_id)                    # finished thinking: pull this one back last
                return res
            finally:
                (HANDOFFS / f"{plan.package_id}.json").unlink(missing_ok=True)


def make_process_worker(session, process_file: Path = PL.DEFAULT_PATH):      # type: ignore[no-untyped-def]
    """The worker built from the process the recursion adopted (defaults when none): the read path of creator_recurse.py."""
    return PL.build_worker(PL.load_process(process_file), session)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--floor-fraction", type=float, default=0.07)   # reserve = this share of the machine's RAM (>= 0.8 GB)
    ap.add_argument("--filler", type=int, default=40)               # leftover-memory jobs per round (own-worker benchmarks)
    ap.add_argument("--test-parallel", type=int, default=6)
    ap.add_argument("--packages", type=int, default=12)
    ap.add_argument("--steps", default=",".join(K.P.WORKER_STEPS))
    ap.add_argument("--mode", choices=("auto", "gaps", "efficiency"), default="auto")
    ap.add_argument("--no-session", action="store_true")
    ap.add_argument("--process-file", type=Path, default=PL.DEFAULT_PATH)   # the process the recursion adopted (CR196-198)
    ap.add_argument("--handoff-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    session = None if a.no_session else SerialSession(a.handoff_hours)

    def make_worker() -> SW.SelfFirst:                                   # own workers first: rules, generated tests, search
        return make_process_worker(session, a.process_file)
    gov = W.Governor(floor_fraction=a.floor_fraction, max_workers=a.max_workers)
    cfg = K.KernelConfig(repo=ROOT, state=STATE, steps=tuple(s for s in a.steps.split(",") if s), mode=a.mode,
                         test_parallel=a.test_parallel)
    n = 0
    with K._KernelLock(STATE):                                           # no single-kernel run at the same time
        while a.rounds == 0 or n < a.rounds:
            n += 1
            rnd = W.run_round(cfg, make_worker, gov, max_packages=a.packages, filler_budget=a.filler,
                              filler=W.self_bench_filler(STATE / "self_bench.jsonl"),
                              on_report=lambda r: print(json.dumps({"package": r.package, "req": r.requirement,
                                                                    "outcome": r.outcome, "reason": r.reason[:200],
                                                                    "by": r.details.get("worker", {}).get("by")}), flush=True))
            line = {"round": n, "outcome": rnd.outcome, "packages": len(rnd.reports), "peak_parallel": rnd.peak_parallel,
                    "pulled_back": rnd.pulled_back, "by_outcome": K.summary(rnd.reports)["by_outcome"],
                    "free_gb": round(W.free_ram_gb(), 2), "at": dt.datetime.now().isoformat(timespec="seconds")}
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            print("ROUND", json.dumps(line), flush=True)
            if rnd.outcome == "RAM_TIGHT":
                time.sleep(60)                                           # memory may free up soon: look again in a minute
            elif rnd.outcome != "WORKED":
                time.sleep(600)                                          # nothing to do / red audit: look again later
    print("SWARM DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
