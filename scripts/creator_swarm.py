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
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from typing import TYPE_CHECKING  # noqa: E402

from creator import kernel as K  # noqa: E402   (the core; everything else is loaded on demand through the registry)
from creator import registry as REG  # noqa: E402
from creator import swarm as W  # noqa: E402

if TYPE_CHECKING:
    from creator import curriculum as CUR
    from creator import selfworkers as SW

STATE = ROOT / "state" / "creator"
HANDOFFS = STATE / "handoffs"                            # one file per package waiting for Claude (all at once, 2 Oct)
LOG = STATE / "swarm_log.jsonl"
HEARTBEAT = STATE / "teacher_heartbeat"                  # touched by the teacher's (Claude's) monitor while it is present
DRAIN = STATE / "NUPEN_DRAIN"                           # deploy pause: plan nothing new, let in-flight cycles finish, exit cleanly
DRAIN_MAX_S = float(os.environ.get("NUPEN_DRAIN_MAX_S", "3600"))     # in-flight cycles take 35-45 min
LESSONS = STATE / "lessons.jsonl"                        # the curriculum: every handoff and student attempt as a lesson


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

    def __init__(self, hours: float, presence: Optional[Path] = None, fresh_s: float = 900.0) -> None:
        self.inner = K.HandoffWorker(timeout_s=hours * 3600, notify=announce)
        self.presence, self.fresh_s = presence, fresh_s

    def teacher_present(self) -> bool:
        """Owner, 1 Oct: Nupen runs whenever the computer is on, without needing me. With a presence file, a package goes to the
        teacher only while the teacher's heartbeat is fresh; otherwise it is DEFERRED (no attempt used, retried later) instead of
        waiting hours for nobody."""
        if self.presence is None:
            return True
        try:
            return time.time() - self.presence.stat().st_mtime < self.fresh_s
        except OSError:
            return False

    def __call__(self, plan, package, workdir):                       # type: ignore[no-untyped-def]
        if not self.teacher_present():
            return K.WorkResult(False, "teacher absent: deferred until the teacher is back", deferred=True)
        with W.waiting_on_thinker(plan.package_id):                       # waiting for me holds no worker slot
            try:
                res = self.inner(plan, package, workdir)
                if res.claimed_done:
                    W.HANDED_BACK.add(plan.package_id)                    # finished thinking: pull this one back last
                return res
            finally:
                (HANDOFFS / f"{plan.package_id}.json").unlink(missing_ok=True)


PROCESS_FILE = STATE / "process.json"                    # process_levers.DEFAULT_PATH, without importing the module to learn it


def make_process_worker(session, process_file: Path = PROCESS_FILE):      # type: ignore[no-untyped-def]
    """The worker built from the process the recursion adopted (defaults when none): the read path of creator_recurse.py."""
    PL = REG.get("process_levers")
    return PL.build_worker(PL.load_process(process_file), session)


def make_students(model_student: bool = True) -> list:      # type: ignore[type-arg]
    """The curriculum's students: creator.student.LessonStudent when that module exists (absent = no students), then the
    model-backed ModelStudent (local qwen, server started lazily per attempt) unless model_student is False."""
    LessonStudent = REG.optional("lesson_student")
    if LessonStudent is None:
        return []
    students: list = [LessonStudent(LESSONS)]                # type: ignore[type-arg]
    ReplayStudent = REG.optional("replay_student")
    if ReplayStudent is not None:
        students.insert(0, ReplayStudent(LESSONS))           # an unmeasured teacher solution first: measure it, don't redo it
    ResumeStudent = REG.optional("resume_student")
    if ResumeStudent is not None:
        students.insert(1 if students and students[0].name == "claude-replay" else 0,
                        ResumeStudent(LESSONS))              # then pending diffs / unmeasured lessons, re-applied to the current tree
    if model_student:
        ModelStudent = REG.optional("model_student")
        if ModelStudent is not None:
            students.append(ModelStudent(LESSONS))
        try:
            ActionStudent = REG.optional("action_student")
            if ActionStudent is not None:
                students.append(ActionStudent(LESSONS))
        except Exception:                                     # noqa: BLE001 - a missing student never crashes the swarm
            pass
    return students


def make_curriculum(lessons: Path = LESSONS, students=None, model_student: bool = True) -> "CUR.Curriculum":    # type: ignore[no-untyped-def]
    return REG.get("curriculum").Curriculum(lessons, make_students(model_student) if students is None else students)


THINK_EVERY_S = 600.0                                     # live predictions + resolutions (thinking.run)
BLUEPRINT_EVERY_S = 3600.0                                # a dated blueprint snapshot (anticipation needs a time series of them)
_THINK_LAST = {"run": 0.0, "blueprint": 0.0}


def make_filler():                                        # type: ignore[no-untyped-def]
    """Leftover-capacity work. In THINKING focus (state/creator/focus.json; owner, 2 Oct 2026: 'have Nupen work souly on thinking
    until its opinion is trust worthy') it is the thinking drills over all local data, the live predictions every THINK_EVERY_S and
    a blueprint every BLUEPRINT_EVERY_S - as independent jobs the Governor admits until CPU or RAM is full. Otherwise the self-bench."""
    try:
        focus = REG.optional("focus")
        if focus is not None and focus.current(STATE) == "thinking":
            drills = REG.get("drillsources").drill_filler(STATE, ROOT, Path.home() / "Masterstock" / "JOURNAL.md",
                                                          ROOT / "state" / "research")

            def next_job():                               # type: ignore[no-untyped-def]
                now = time.time()
                if now - _THINK_LAST["run"] >= THINK_EVERY_S:
                    _THINK_LAST["run"] = now
                    return lambda: REG.get("thinking").run(STATE)
                if now - _THINK_LAST["blueprint"] >= BLUEPRINT_EVERY_S:
                    _THINK_LAST["blueprint"] = now
                    import subprocess
                    return lambda: subprocess.run([sys.executable, str(ROOT / "scripts" / "nupen_blueprint.py"), "--state", str(STATE)],
                                                  capture_output=True, timeout=1800, cwd=ROOT)
                return drills()
            return next_job
    except Exception as e:                                # noqa: BLE001 - a broken thinking module never stops the swarm
        print(f"THINKING filler unavailable: {type(e).__name__}: {e}", flush=True)
    return W.self_bench_filler(STATE / "self_bench.jsonl")


def main(argv: list[str]) -> int:
    DEV = REG.get("device")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=int(DEV.settings()["max_workers"]))   # RAM-derived (32 on 16 GB)
    ap.add_argument("--floor-fraction", type=float, default=float(DEV.settings()["governor_floor_fraction"]))   # reserve = this share of RAM
    ap.add_argument("--floor-min-gb", type=float, default=float(DEV.settings()["governor_floor_min_gb"]))      # ... but never less than this
    ap.add_argument("--filler", type=int, default=40)               # leftover-memory jobs per round (own-worker benchmarks)
    ap.add_argument("--test-parallel", type=int, default=int(DEV.settings()["test_slots"]))   # machine-derived (4 on 16 GB); creator.testslots caps across all evaluations
    ap.add_argument("--packages", type=int, default=12)
    ap.add_argument("--steps", default=",".join(K.P.WORKER_STEPS))
    ap.add_argument("--mode", choices=("auto", "gaps", "efficiency"), default="auto")
    ap.add_argument("--no-session", action="store_true")
    ap.add_argument("--no-model-student", action="store_true")      # skip the local-model student (nupen-model-v1)
    ap.add_argument("--process-file", type=Path, default=PROCESS_FILE)   # the process the recursion adopted (CR196-198)
    ap.add_argument("--handoff-hours", type=float, default=6.0)
    ap.add_argument("--teacher-presence", action="store_true")      # hand off only while the teacher's heartbeat is fresh
    ap.add_argument("--user-aware", action="store_true")            # keep 25% of RAM free while the owner is at the keyboard
    a = ap.parse_args(argv)
    DEV.write_snapshot(log=print)                                   # state/creator/device.json; a changed machine shows in the log
    session = None if a.no_session else SerialSession(a.handoff_hours, HEARTBEAT if a.teacher_presence else None)

    cur = make_curriculum(model_student=not a.no_model_student)

    def make_worker() -> "SW.SelfFirst":                                   # students, own workers, then (recorded) the session
        return cur.install(make_process_worker(session, a.process_file))
    gov = W.Governor(floor_fraction=a.floor_fraction, floor_min_gb=a.floor_min_gb, max_workers=a.max_workers,
                     user_active_floor_fraction=0.25 if a.user_aware else None, test_parallel=a.test_parallel,
                     disk_floor_gb=float(DEV.settings().get("disk_floor_gb", 1.0)))
    cfg = K.KernelConfig(repo=ROOT, state=STATE, steps=tuple(s for s in a.steps.split(",") if s), mode=a.mode,
                         test_parallel=a.test_parallel)
    n = 0
    with K._KernelLock(STATE):                                           # no single-kernel run at the same time
        try:                                                             # nothing of an earlier process still runs: release
            freed = REG.get("swarmops").release_orphans(K.Ledger(cfg.ledger_path, evidence_root=cfg.repo), "swarm restarted, no worker survives")
            if freed:
                print("ORPHANS RELEASED " + json.dumps(freed), flush=True)
        except Exception as e:                                           # noqa: BLE001 - never block the start on bookkeeping
            print(f"ORPHANS error: {type(e).__name__}: {e}", flush=True)
        try:                                                             # independent verdicts that are current reach the ledger
            print("REVALIDATE " + REG.get("swarmops").revalidate(cfg.repo), flush=True)
        except Exception as e:                                           # noqa: BLE001 - bookkeeping never blocks the start
            print(f"REVALIDATE error: {type(e).__name__}: {e}", flush=True)
        while (a.rounds == 0 or n < a.rounds) and not DRAIN.exists():
            n += 1
            try:
                rec = cur.reconcile()                                    # settle lessons left pending by a stopped swarm (never raises)
                if rec.get("settled"):
                    print("RECONCILED", json.dumps(rec), flush=True)
            except Exception:                                            # noqa: BLE001
                pass
            def on_report(r: K.CycleReport) -> None:
                cur.resolve(r)
                print(json.dumps({"package": r.package, "req": r.requirement, "outcome": r.outcome, "reason": r.reason[:200],
                                  "by": r.details.get("worker", {}).get("by")}), flush=True)
            rnd = W.run_round(cfg, make_worker, gov, max_packages=a.packages, filler_budget=a.filler,
                              filler=make_filler(), on_report=on_report,
                              drain=DRAIN.exists, drain_max_s=DRAIN_MAX_S)
            line = {"round": n, "outcome": rnd.outcome, "packages": len(rnd.reports), "peak_parallel": rnd.peak_parallel,
                    "pulled_back": rnd.pulled_back, "by_outcome": K.summary(rnd.reports)["by_outcome"],
                    "free_gb": round(W.free_ram_gb(), 2), "at": dt.datetime.now().isoformat(timespec="seconds")}
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
            print("ROUND", json.dumps(line), flush=True)
            if rnd.outcome == "DRAINED":
                break
            if rnd.outcome == "RAM_TIGHT":
                time.sleep(60)                                           # memory may free up soon: look again in a minute
            elif rnd.outcome != "WORKED":
                for _ in range(600):                                     # nothing to do / red audit: look again later (a drain ends the wait)
                    if DRAIN.exists():
                        break
                    time.sleep(1)
    print("SWARM DONE" + (" (drained)" if DRAIN.exists() else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
