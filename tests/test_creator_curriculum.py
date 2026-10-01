"""Curriculum: lessons from handoffs, students before Claude, scoring, and handover (temp repos and ledgers, fake kernel loop)."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from creator import curriculum as CUR
from creator import kernel as K
from creator import selfworkers as SW


def sh(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


def make_repo(tmp: Path) -> Path:
    repo = tmp / "wd"
    repo.mkdir()
    sh(repo, "init", "-q")
    (repo / "m.py").write_text("x = 1\n", encoding="utf-8")
    sh(repo, "add", ".")
    sh(repo, "commit", "-qm", "base")
    return repo


def plan(pid: str = "P1", step: str = "exists", key: str = "R.a") -> Any:
    return SimpleNamespace(package_id=pid, component="C1", step=step, requirement_key=key)


PKG: Any = SimpleNamespace(package_id="P1", objective="make m.py say y", why_it_exists="w", implementation_requirements=("do it",),
                           test_requirements=("test it",), interfaces=(), expected_failure_modes=(), completion_criteria=("done",))


class FakeStudent:
    def __init__(self, name: str = "stu", succeed: bool = True, can: bool = True) -> None:
        self.name, self.succeed, self.can, self.calls = name, succeed, can, 0

    def can_attempt(self, task: CUR.Lesson) -> bool:
        return self.can

    def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult:
        self.calls += 1
        if self.succeed:
            (workdir / "m.py").write_text("x = 2\n", encoding="utf-8")
        return K.WorkResult(self.succeed, "student notes", reasoning="because", by=self.name)


class FakeClaude:
    name = "claude-session"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult:
        self.calls += 1
        (workdir / "m.py").write_text("x = 3\n", encoding="utf-8")
        return K.WorkResult(True, "claude notes", reasoning="I changed x", by=self.name)


def report(pid: str, outcome: str) -> Any:
    return SimpleNamespace(package=pid, outcome=outcome, reason="r")


def answer_later(repo: Path, body: dict[str, Any]) -> threading.Thread:
    def go() -> None:
        for _ in range(400):
            if (repo / ".creator_task.md").exists():
                break
            time.sleep(0.025)
        (repo / "m.py").write_text("x = 3\n", encoding="utf-8")
        (repo / ".creator_done.json").write_text(json.dumps(body), encoding="utf-8")
    t = threading.Thread(target=go)
    t.start()
    return t


def test_lesson_captured_with_reasoning_from_done_file_and_adopted_filled(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    cur = CUR.Curriculum(tmp_path / "lessons.jsonl")
    t = answer_later(repo, {"claimed_done": True, "notes": "n", "reasoning": "why x=3"})
    res = cur.claude_step(K.HandoffWorker(poll_s=0.05, timeout_s=20))(plan(), PKG, repo)
    t.join()
    assert res.claimed_done and res.reasoning == "why x=3"
    (les,) = cur.log.lessons()
    assert les.reasoning == "why x=3" and les.solver == "claude" and les.adopted is None and les.task_kind == "gap"
    assert les.files_before == {"m.py": "x = 1\n"} and les.files_after == {"m.py": "x = 3\n"}
    assert les.objective == "make m.py say y" and les.package_id == "P1" and "make m.py say y" in les.context
    cur.resolve(report("P1", "ADOPTED"))
    (les,) = cur.log.lessons()
    assert les.adopted is True and "ADOPTED" in les.verdict
    assert len((tmp_path / "lessons.jsonl").read_text().splitlines()) == 2        # append-only: an outcome line, no rewrite


def test_old_done_file_without_reasoning_still_works(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    t = answer_later(repo, {"claimed_done": True, "notes": "old"})
    res = K.HandoffWorker(poll_s=0.05, timeout_s=20)(plan(), PKG, repo)
    t.join()
    assert res.claimed_done and res.notes == "old" and res.reasoning == ""


def test_student_tried_before_claude_and_claude_skipped_when_it_claims(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    stu, claude = FakeStudent(), FakeClaude()
    cur = CUR.Curriculum(tmp_path / "l.jsonl", [stu])
    res = cur.install(SW.SelfFirst([], claude))(plan(), PKG, repo)
    assert res.claimed_done and res.by == "stu" and stu.calls == 1 and claude.calls == 0
    cur.resolve(report("P1", "ADOPTED"))
    (les,) = cur.log.lessons()
    assert les.solver == "stu" and les.adopted is True and les.reasoning == "because"


def test_claude_gets_it_when_all_students_fail_and_failure_is_a_lesson(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    stu, claude = FakeStudent(succeed=False), FakeClaude()
    cur = CUR.Curriculum(tmp_path / "l.jsonl", [stu, FakeStudent("cant", can=False)])
    res = cur.install(SW.SelfFirst([], claude))(plan(), PKG, repo)
    assert res.claimed_done and res.by == "claude-session" and stu.calls == 1 and claude.calls == 1
    cur.resolve(report("P1", "REJECTED"))
    ls = {les.solver: les for les in cur.log.lessons()}
    assert set(ls) == {"stu", "claude"} and ls["stu"].adopted is False and ls["claude"].adopted is False


def test_student_rejected_by_kernel_is_not_retried_on_same_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    stu, claude = FakeStudent(), FakeClaude()
    cur = CUR.Curriculum(tmp_path / "l.jsonl", [stu])
    w = cur.install(SW.SelfFirst([], claude))
    w(plan(), PKG, repo)
    cur.resolve(report("P1", "REJECTED"))
    SW.reset(repo)
    res = w(plan(), PKG, repo)                       # the retry goes to Claude
    assert stu.calls == 1 and claude.calls == 1 and res.by == "claude-session"


def lessons_for(solver: str, kind: str, adopted: list[bool]) -> list[CUR.Lesson]:
    return [CUR.Lesson(lesson_id=f"{solver}{kind}{i}", package_id=f"p{i}", component="c", task_kind=kind, objective="o",
                       solver=solver, adopted=a) for i, a in enumerate(adopted)]


def test_scores_and_claude_share() -> None:
    ls = lessons_for("stu", "gap", [True, True, False]) + lessons_for("claude", "gap", [True]) + \
        lessons_for("claude", "shrink", [False])
    ls.append(CUR.Lesson("u", "pu", "c", "gap", "o", solver="stu", adopted=None))        # undecided: not counted
    sc = CUR.student_scores(ls)
    assert sc["cells"]["stu|gap"] == {"solver": "stu", "task_kind": "gap", "attempts": 3, "adopted": 2, "rate": 0.667}
    assert sc["cells"]["claude|shrink"]["adopted"] == 0
    assert sc["adopted_total"] == 3 and sc["teacher_share"] == sc["claude_share"] == 0.333


def test_routing_hands_over_only_after_threshold_and_never_drops(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    router = CUR.Router(min_attempts=5, threshold=0.8)
    assert not router.handed_over(lessons_for("stu", "gap", [True] * 4), "gap")                  # not before N attempts
    assert not router.handed_over(lessons_for("stu", "gap", [True, True, True, False, False]), "gap")   # rate too low
    five = lessons_for("stu", "gap", [True] * 5)
    assert router.handed_over(five, "gap") and not router.handed_over(five, "shrink")
    assert not router.handed_over(lessons_for("claude", "gap", [True] * 9), "gap")                # Claude never owns a kind
    log = CUR.LessonLog(tmp_path / "l.jsonl")
    for les in five:
        log.add(les)
        log.outcome(les.lesson_id, True, "ADOPTED")
    stu, claude = FakeStudent(succeed=False), FakeClaude()
    cur = CUR.Curriculum(tmp_path / "l.jsonl", [stu], router)
    res = cur.install(SW.SelfFirst([], claude))(plan("P9"), PKG, repo)
    assert not res.claimed_done and res.deferred and claude.calls == 0 and stu.calls == 1        # student only, Claude skipped, handed over not failed
    deferred = [json.loads(x) for x in (tmp_path / "deferred.jsonl").read_text().splitlines()]
    assert deferred[0]["package"] == "P9" and deferred[0]["task_kind"] == "gap"                  # queued, not dropped
    other = cur.install(SW.SelfFirst([], claude))(plan("P10", step="efficiency", key="EFF.x"), PKG, repo)
    assert other.claimed_done and claude.calls == 1                                              # other kinds still go to Claude


def test_task_kind_from_plan() -> None:
    assert CUR.task_kind(plan(step="efficiency", key="EFF.activation")) == "activation"
    assert CUR.task_kind(plan(step="efficiency", key="EFF.size")) == "shrink"
    assert CUR.task_kind(plan(step="tested")) == "bugfix" and CUR.task_kind(plan(step="exists")) == "gap"


def test_several_students_scored_separately_per_kind() -> None:
    ls = lessons_for("nupen", "gap", [True] * 5) + lessons_for("small_model", "gap", [True, False] * 3)
    sc = CUR.student_scores(ls)
    assert sc["cells"]["nupen|gap"]["rate"] == 1.0 and sc["cells"]["small_model|gap"]["rate"] == 0.5
    assert CUR.Router().owners(ls, "gap") == ["nupen"]


def test_concurrent_appends_never_interleave_or_lose_lessons(tmp_path: Path) -> None:
    """Regression (validator round 3): swarm workers share one lessons.jsonl; a lesson carrying whole files is far larger than the
    write buffer, so unlocked appends interleaved and corrupt lines were silently dropped by lessons()."""
    log = CUR.LessonLog(tmp_path / "l.jsonl")
    big = "x = 1\n" * 40000

    def work(w: int) -> None:
        for i in range(12):
            les = CUR.Lesson(f"w{w}i{i}", f"P{w}", "C", "gap", "o", files_before={"m.py": big}, files_after={"m.py": big + "y"})
            log.add(les)
            log.outcome(les.lesson_id, True, "ADOPTED")
            log.defer(f"P{w}", "gap", "why " + big[:30000])
    ts = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for name in ("l.jsonl", "deferred.jsonl"):
        for ln in (tmp_path / name).read_text(encoding="utf-8").splitlines():
            json.loads(ln)                                                  # every line is whole
    assert len(log.lessons()) == 96 and all(x.adopted is True for x in log.lessons())
