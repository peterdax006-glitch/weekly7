"""Nupen is the student learning to think for itself; Claude is the teacher who thinks for it until it can.

Handoffs as a curriculum: every solved work package becomes a lesson, students try before Claude, and a student that has
shown it can do a kind of task takes it over (owner-endorsed: Claude must be needed less and less).

* Lesson   - one attempt at one package by one solver (claude or a student): task, context, files before/after, reasoning,
             and (once the kernel's cycle outcome is known) adopted + verdict. Stored append-only in a JSONL ledger.
* Student  - protocol: name, can_attempt(task) and __call__(plan, package, workdir) -> kernel WorkResult.
* Scoring  - student_scores(lessons) per (solver, task_kind) and the overall claude_share.
* Routing  - Router.handed_over(kind): a student with >= min_attempts attempts at adopted rate >= threshold owns the kind;
             its packages are never handed to Claude (a failure is queued in deferred.jsonl and retried by the kernel later).
The kernel's own measurement (sandbox tests + improvement verdict) decides adoption; nothing here judges."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

from creator import kernel as K
from creator import reasoning as RE

CLAUDE = "claude"
_APPEND_LOCK = threading.Lock()                     # swarm workers (threads) share one lessons.jsonl and deferred.jsonl


def _append_line(path: Path, text: str) -> None:
    """One whole line per append: lessons carry whole files, far beyond the write buffer, so unlocked appends interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (text + "\n").encode("utf-8")
    with _APPEND_LOCK, path.open("ab") as fh:
        fh.write(data)
        fh.flush()


ADJUDICATED = {"ADOPTED": True, "REJECTED": False, "ROLLED_BACK": False}      # cycle outcomes that decide a lesson
# Cycles that ended without judging the change at all (RAM pull-back, kernel crash). The lesson gets a TERMINAL outcome
# (adopted=False, verdict "cancelled: ..." / "error: ...") so it is never left unscored forever, but it says nothing about
# skill, so it is EXCLUDED from student success rates (attempts/rate) and from teacher_share, and never counts as a rejection.
NO_SIGNAL = {"CANCELLED": "cancelled", "ERROR": "error"}
REPLAY = "claude-replay"                            # creator.replay_student: re-applies an unmeasured teacher solution
TEACHER = frozenset({"claude", REPLAY})             # both are the teacher's work: never a student's skill, never an owner


def is_skill_signal(les: "Lesson") -> bool:
    """True when the lesson's outcome is a real judgement of the work (not a cancelled/errored cycle)."""
    return not str(les.verdict).lower().startswith(tuple(v + ":" for v in NO_SIGNAL.values()))


@dataclasses.dataclass
class Lesson:
    lesson_id: str
    package_id: str
    component: str
    task_kind: str                                  # gap | bugfix | shrink | activation | tests
    objective: str                                  # the task text (the package objective)
    context: str = ""                               # what the system knew: the .creator_task.md text
    files_before: dict[str, str] = dataclasses.field(default_factory=dict)   # changed files only: path -> text before ("" if new)
    files_after: dict[str, str] = dataclasses.field(default_factory=dict)    # changed files only: path -> text after
    reasoning: str = ""                             # free text from the solver
    solver: str = CLAUDE                            # "claude" or the student's name
    adopted: Optional[bool] = None                  # None until the kernel's cycle outcome is known
    verdict: str = ""                               # outcome / reason once known
    claimed_done: bool = False
    at: str = ""
    predicted: dict[str, float] = dataclasses.field(default_factory=dict)   # the solver's predicted effect, e.g. {'size_delta': -12}
    measured: dict[str, float] = dataclasses.field(default_factory=dict)    # the same effect measured on the change (creator.reasoning)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Lesson":
        names = {f.name for f in dataclasses.fields(Lesson)}
        return Lesson(**{k: v for k, v in d.items() if k in names})


@runtime_checkable
class Student(Protocol):
    name: str

    def can_attempt(self, task: Lesson) -> bool:
        """Cheap check on a lesson-like task (task_kind, component, objective, context filled; files empty)."""
        ...

    def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult: ...


def task_kind(plan: Any) -> str:
    """Derive the kind from the plan: efficiency work is shrink (activation for the activation requirement), a 'tested' step fixes
    code under test (bugfix), everything else closes a gap."""
    step = str(getattr(plan, "step", ""))
    key = str(getattr(plan, "requirement_key", ""))
    if step == "coverage":
        return "tests"
    if step == "efficiency":
        return "activation" if key.endswith("activation") else "shrink"
    if step == "tested":
        return "bugfix"
    return "gap"


# ------------------------------------------------------------------------------------------------ storage

class LessonLog:
    """Append-only JSONL. Two line kinds: a lesson, and {"event": "outcome", lesson_id, adopted, verdict} that fills it in later."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _append(self, rec: dict[str, Any]) -> None:
        _append_line(self.path, json.dumps(rec, sort_keys=True))

    def add(self, lesson: Lesson) -> str:
        self._append(lesson.to_dict())
        return lesson.lesson_id

    def outcome(self, lesson_id: str, adopted: Optional[bool], verdict: str) -> None:
        self._append({"event": "outcome", "lesson_id": lesson_id, "adopted": adopted, "verdict": verdict[:2000],
                      "at": dt.datetime.now().isoformat(timespec="seconds")})

    def lessons(self) -> list[Lesson]:
        out: dict[str, Lesson] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for ln in lines:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("event") == "outcome":
                les = out.get(str(d.get("lesson_id")))
                if les is not None:
                    les.adopted, les.verdict = d.get("adopted"), str(d.get("verdict", ""))
            elif "lesson_id" in d:
                out[d["lesson_id"]] = Lesson.from_dict(d)
        return list(out.values())

    def defer(self, package_id: str, kind: str, why: str) -> None:
        dp = self.path.with_name("deferred.jsonl")
        _append_line(dp, json.dumps({"package": package_id, "task_kind": kind, "why": why,
                                     "at": dt.datetime.now().isoformat(timespec="seconds")}))


# ------------------------------------------------------------------------------------------------ scoring and routing

def student_scores(lessons: Iterable[Lesson]) -> dict[str, Any]:
    """Several students are scored side by side, each per task_kind. Per (solver, task_kind): attempts (lessons with a known outcome), adopted, rate; plus claude_share of adopted changes."""
    cells: dict[tuple[str, str], list[int]] = {}
    adopted_by: dict[str, int] = {}
    for les in lessons:
        if les.adopted is None or not is_skill_signal(les):
            continue
        c = cells.setdefault((les.solver, les.task_kind), [0, 0])
        c[0] += 1
        if les.adopted:
            c[1] += 1
            adopted_by[les.solver] = adopted_by.get(les.solver, 0) + 1
    total = sum(adopted_by.values())
    share = round(sum(adopted_by.get(t, 0) for t in TEACHER) / total, 3) if total else None   # a replay is the teacher's work
    return {"cells": {f"{s}|{k}": {"solver": s, "task_kind": k, "attempts": a, "adopted": d, "rate": round(d / a, 3)}
                      for (s, k), (a, d) in sorted(cells.items())},
            "adopted_total": total, "adopted_by": adopted_by,
            "teacher_share": share, "claude_share": share}          # claude_share: alias of teacher_share (Claude is the teacher)


@dataclasses.dataclass(frozen=True)
class Router:
    min_attempts: int = 5
    threshold: float = 0.8

    def owners(self, lessons: Iterable[Lesson], kind: str) -> list[str]:
        """Students (never claude) that own `kind`: enough attempts at a high enough adopted rate."""
        out = []
        for c in student_scores(lessons)["cells"].values():
            if (c["task_kind"] == kind and c["solver"] not in TEACHER and c["attempts"] >= self.min_attempts
                    and c["rate"] >= self.threshold):
                out.append(c["solver"])
        return out

    def handed_over(self, lessons: Iterable[Lesson], kind: str) -> bool:
        return bool(self.owners(lessons, kind))


# ------------------------------------------------------------------------------------------------ the capturing workers

def _git(workdir: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def snapshot_change(workdir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(files_before, files_after) for the files changed in the sandbox right now (before = committed text, '' when new)."""
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for p in K.git_changed_files(workdir):
        rel = p.relative_to(workdir).as_posix()
        if rel in K.S.HANDOFF_FILES:
            continue
        try:
            after[rel] = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        before[rel] = _git(workdir, "show", f"HEAD:{rel}")
    return before, after


def _restore(workdir: Path, before: dict[str, str], after: dict[str, str]) -> None:
    """Undo a change snapshot_change described: new files are removed, changed files get their committed text back."""
    for rel in after:
        f = workdir / rel
        if before.get(rel):
            f.write_bytes(before[rel].encode("utf-8"))
        else:
            f.unlink(missing_ok=True)


class Curriculum:
    """Wires lessons into the kernel loop. `student_steps()` and `claude_step(session)` are workers for creator.selfworkers.SelfFirst
    (students as own workers, the capturing session as fallback); `resolve(report)` is the kernel's on_report callback."""

    def __init__(self, lessons_path: Path, students: Sequence[Student] = (), router: Optional[Router] = None) -> None:
        self.log = LessonLog(lessons_path)
        self.students = list(students)
        self.router = router or Router()
        self._open: dict[str, list[str]] = {}             # package_id -> lessons waiting for the cycle outcome

    # --- recording
    def _draft(self, plan: Any, package: Any, solver: str) -> Lesson:
        try:
            ctx = K.render_package(plan, package)
        except Exception:                                   # noqa: BLE001 - a lesson is never worth a crash
            ctx = ""
        return Lesson(lesson_id=uuid.uuid4().hex[:12], package_id=str(getattr(plan, "package_id", "")),
                      component=str(getattr(plan, "component", "")), task_kind=task_kind(plan),
                      objective=str(getattr(package, "objective", "")), context=ctx, solver=solver,
                      at=dt.datetime.now().isoformat(timespec="seconds"))

    def _record(self, les: Lesson, res: K.WorkResult, workdir: Path) -> None:
        les.claimed_done, les.reasoning = res.claimed_done, (res.reasoning or res.notes)
        if res.claimed_done:
            les.files_before, les.files_after = snapshot_change(workdir)
            les.predicted = {k: float(v) for k, v in (getattr(res, "predicted", None) or {}).items()}
            les.measured = {"size_delta": RE.size_delta(les.files_before, les.files_after)}
        self.log.add(les)
        if res.claimed_done:
            self._open.setdefault(les.package_id, []).append(les.lesson_id)
        else:
            self.log.outcome(les.lesson_id, False, f"not claimed done: {res.notes[:300]}")

    def resolve(self, report: Any) -> None:
        """Fill adopted/verdict of the package's pending lessons from the kernel's cycle outcome (CycleReport)."""
        ids = self._open.pop(str(getattr(report, "package", "")), [])
        outcome = str(getattr(report, "outcome", ""))
        for lid in ids:
            if outcome in ADJUDICATED:
                self.log.outcome(lid, ADJUDICATED[outcome], f"{outcome}: {str(getattr(report, 'reason', ''))[:500]}")
            elif outcome in NO_SIGNAL:
                self.log.outcome(lid, False, f"{NO_SIGNAL[outcome]}: {str(getattr(report, 'reason', ''))[:500]}")
            else:
                self.log.outcome(lid, None, f"{outcome}: undecided")

    # --- workers
    def student_steps(self) -> list["_StudentStep"]:
        return [_StudentStep(self, s) for s in self.students]

    def claude_step(self, session: Any) -> "_ClaudeStep":
        return _ClaudeStep(self, session)

    def install(self, selffirst: Any) -> Any:
        """A SelfFirst whose own workers are the students (first) then the existing ones, and whose fallback records Claude."""
        from creator import selfworkers as SW
        fb = selffirst.fallback
        return SW.SelfFirst([*self.student_steps(), *selffirst.own], self.claude_step(fb) if fb is not None else None)


class _StudentStep:
    def __init__(self, cur: Curriculum, student: Student) -> None:
        self.cur, self.student, self.name = cur, student, student.name

    def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult:
        les = self.cur._draft(plan, package, self.student.name)
        if (les.task_kind == "tests" and self.name not in TEACHER          # a replay IS the teacher's example (2 Oct, CP0091)
                and not any(p.task_kind == "tests" and p.adopted for p in self.cur.log.lessons())):
            return K.WorkResult(False, f"{self.name}: no adopted example of writing tests yet - the teacher shows first")
        if not self.student.can_attempt(les):
            return K.WorkResult(False, f"{self.name} cannot attempt this task")
        past = self.cur.log.lessons()
        if not self.cur.router.handed_over(past, les.task_kind) and any(
                p.package_id == les.package_id and p.solver == self.name and p.adopted is False and is_skill_signal(p) for p in past):
            return K.WorkResult(False, f"{self.name} was already rejected on {les.package_id}")      # Claude's turn
        res = self.student(plan, package, workdir)
        if res.claimed_done:                                    # stop rule: an identical change to a rejected one is never retried
            before, after = snapshot_change(workdir)
            dup = RE.identical_failed(past, les.component, before, after)
            if dup is not None:
                _restore(workdir, before, after)
                res = K.WorkResult(False, f"{self.name}: identical to rejected lesson {dup.lesson_id} ({dup.package_id}): not retried; "
                                          f"{RE.NEEDS_TEACHER}", calls=res.calls)
        self.cur._record(les, res, workdir)
        return dataclasses.replace(res, by=res.by or self.name)


class _ClaudeStep:
    name = CLAUDE + "-session"

    def __init__(self, cur: Curriculum, session: Any) -> None:
        self.cur, self.session = cur, session
        self.name = getattr(session, "name", self.name)

    def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult:
        les = self.cur._draft(plan, package, CLAUDE)
        owners = self.cur.router.owners(self.cur.log.lessons(), les.task_kind)
        if owners:
            why = f"{les.task_kind} is handed over to {owners}; not sent to Claude, retry later"
            self.cur.log.defer(les.package_id, les.task_kind, why)
            return K.WorkResult(False, why, deferred=True)
        res = self.session(plan, package, workdir)
        self.cur._record(les, res, workdir)
        return res
