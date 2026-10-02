"""The replay student re-applies a teacher solution that was never measured, and only then (2 Oct 2026)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from creator import curriculum as C
from creator.replay_student import ReplayStudent

OBJ = "cover the untested public names of app/x.py"


def _lesson(log: C.LessonLog, before: dict, after: dict, adopted=None, verdict: str = "", solver: str = "claude",
            objective: str = OBJ) -> str:
    les = C.Lesson(lesson_id=f"L{len(log.lessons())}", package_id="CP1", component="EFF.coverage", task_kind="tests",
                   objective=objective, context="", files_before=before, files_after=after, reasoning="why it is right",
                   solver=solver, claimed_done=True)
    lid = log.add(les)
    if adopted is not None or verdict:
        log.outcome(lid, adopted, verdict)
    return lid


def _work(tmp_path: Path, text: str = "def f():\n    return 1\n") -> Path:
    wd = tmp_path / "wd"
    (wd / "app").mkdir(parents=True)
    (wd / "app" / "x.py").write_text(text, encoding="utf-8")
    return wd


def test_an_unmeasured_teacher_solution_is_replayed_and_attributed_to_the_teacher(tmp_path: Path) -> None:
    log = C.LessonLog(tmp_path / "lessons.jsonl")
    _lesson(log, {"tests/test_x.py": ""}, {"tests/test_x.py": "def test_f():\n    assert f() == 1\n"},
            adopted=False, verdict="cancelled: pulled back before evaluation (RAM tight)")
    st = ReplayStudent(tmp_path / "lessons.jsonl")
    wd = _work(tmp_path)
    assert st.can_attempt(SimpleNamespace(objective=OBJ)) and not st.can_attempt(SimpleNamespace(objective="other"))
    r = st(None, SimpleNamespace(objective=OBJ), wd)
    assert r.claimed_done and r.by == "claude-replay" and r.reasoning == "why it is right"
    assert (wd / "tests" / "test_x.py").read_text(encoding="utf-8").startswith("def test_f")
    assert "claude-replay" in C.TEACHER                                          # its adoptions are the teacher's work


def test_never_replays_a_measured_solution_or_onto_changed_files(tmp_path: Path) -> None:
    log = C.LessonLog(tmp_path / "lessons.jsonl")
    _lesson(log, {"app/x.py": "def f():\n    return 1\n"}, {"app/x.py": "def f():\n    return 2\n"})   # no verdict yet
    st = ReplayStudent(tmp_path / "lessons.jsonl")
    changed = _work(tmp_path, "def f():\n    return 99\n")                      # the file moved on since the teacher's start
    assert not st(None, SimpleNamespace(objective=OBJ), changed).claimed_done
    assert changed.joinpath("app/x.py").read_text(encoding="utf-8").endswith("99\n")    # untouched
    log2 = C.LessonLog(tmp_path / "l2.jsonl")
    _lesson(log2, {"app/x.py": "def f():\n    return 1\n"}, {"app/x.py": "def f():\n    return 2\n"}, adopted=False,
            verdict="REJECTED: claim REGRESSION")                               # measured and rejected: never again
    assert not ReplayStudent(tmp_path / "l2.jsonl").can_attempt(SimpleNamespace(objective=OBJ))
    log3 = C.LessonLog(tmp_path / "l3.jsonl")
    _lesson(log3, {"tests/t.py": ""}, {"tests/t.py": "x = 1\n"}, solver="nupen-model-v2")   # a student's, not the teacher's
    assert not ReplayStudent(tmp_path / "l3.jsonl").can_attempt(SimpleNamespace(objective=OBJ))


def test_teacher_share_counts_replays_and_a_replay_never_owns_a_kind(tmp_path: Path) -> None:
    lessons = [C.Lesson(lesson_id=f"L{i}", package_id=f"P{i}", component="c", task_kind="tests", objective="o", context="",
                        solver=s, adopted=True, verdict="ADOPTED: merged") for i, s in enumerate(["claude-replay"] * 6 + ["nupen-x"] * 2)]
    sc = C.student_scores(lessons)
    assert sc["teacher_share"] == 0.75
    assert "claude-replay" not in C.Router(min_attempts=5, threshold=0.8).owners(lessons, "tests")
