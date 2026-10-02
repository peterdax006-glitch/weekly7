"""Reasoning procedure (creator.reasoning): it reaches the work package with recalled lessons, predictions are stored against measured
effects and scored, stop rules refuse repeated failures, and missing optional modules never break planning."""
from __future__ import annotations

import datetime as dt
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from creator import constraints as CON
from creator import curriculum as CUR
from creator import gaps as G
from creator import kernel as K
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import reasoning as RE
from creator import selfmodel as SM
from creator.ledger import Ledger
from tests.test_creator_curriculum import FakeStudent, make_repo, plan as cplan, PKG, sh
from tests.test_creator_planner import SPECS, put

SPECS3 = [SPECS[0], SM.CapabilitySpec("K05", "user", ("pkg/user.py",), ("tests/test_user.py",), 3),
          SM.CapabilitySpec("K06", "third", ("pkg/third.py",), ("tests/test_third.py",), 3)]


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Ledger, SM.SelfModel, Path]:
    r = tmp_path / "proj"
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n")
    put(r, "pkg/app.py", "from pkg.base import one\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")
    led = Ledger(r / "dev.jsonl", evidence_root=r)
    O.compile_capabilities(led, O.self_objective(led), SPECS3)
    model = SM.build(r, scope=("pkg", "tests"), capabilities=SPECS3,
                     test_evidence=SM.collect_test_evidence(r, ["tests/test_base.py"], r / "ev.json"), include_versions=False)
    G.sync(led, model)
    return led, model, r


def text_of(led: Ledger, plan: P.Plan) -> str:
    return "\n".join(led.get(plan.work_package_id).implementation_requirements)


def test_the_package_carries_the_procedure_and_the_recalled_lessons(world) -> None:
    led, model, root = world
    log = CUR.LessonLog(root / "lessons.jsonl")                      # planted: one adopted and one rejected lesson for this kind
    log.add(CUR.Lesson("L1", "CP0900", "K05", "gap", "o", files_after={"pkg/user.py": "x=1\n"}, solver="claude", adopted=True, verdict="ADOPTED: ok"))
    log.add(CUR.Lesson("L2", "CP0901", "K05", "gap", "o", solver="nupen-model-v1", adopted=False, verdict="REJECTED: left a stub body"))
    plan = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert plan is not None
    text = text_of(led, plan)
    assert text.startswith("How to approach this: Thinking procedure v1") and all(f"{i}." in text for i in range(1, 10))
    assert "what worked before: claude on K05" in text and "what failed before: nupen-model-v1 on K05" in text and "left a stub body" in text
    assert text.index("what worked before") < text.index("what failed before")                       # worked first
    assert "Heuristics:" in text and "Prefer reversible actions" in text


def test_optional_modules_are_used_when_present_and_skipped_when_absent_or_broken(world, monkeypatch) -> None:
    led, model, _ = world
    monkeypatch.setitem(sys.modules, "creator.fundamentals", None)       # simulate absence: both modules now exist in the tree
    monkeypatch.setitem(sys.modules, "creator.registry", None)
    plan = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert plan is not None and "principles:" not in text_of(led, plan) and "sparse rule:" not in text_of(led, plan)     # absent
    fund = types.ModuleType("creator.fundamentals")
    fund.principles_for = lambda kind: [f"P-{kind}: keep it small"]                              # type: ignore[attr-defined]
    reg = types.ModuleType("creator.registry")
    reg.sparse_rule_text = lambda: (_ for _ in ()).throw(RuntimeError("broken"))                  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "creator.fundamentals", fund)
    monkeypatch.setitem(sys.modules, "creator.registry", reg)
    P.record_outcome(led, plan, False, "first failure")
    again = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert again is not None
    t = text_of(led, again)
    assert "principles: P-gap: keep it small" in t and "sparse rule:" not in t                  # broken module skipped, planning intact


def test_the_same_failure_twice_without_new_evidence_is_not_retried_and_goes_to_the_teacher(world) -> None:
    led, model, _ = world
    first = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert first is not None
    P.record_outcome(led, first, False, "worker produced a stub")
    second = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert second is not None and second.gap_id == first.gap_id
    P.record_outcome(led, second, False, "worker produced a stub")
    third = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert third is None or third.gap_id != first.gap_id
    assert led.view.status[first.gap_id] is M.Status.BLOCKED
    why = [t.record.reason for t in led.about(first.gap_id) if t.rtype == "Transition" and t.record.to_state is M.Status.BLOCKED][-1]
    assert RE.NEEDS_TEACHER in why and "worker produced a stub" in why
    again = P.plan_next(led, model, "b", SPECS3, steps=("exists",), max_attempts=9)
    assert again is None or again.gap_id != first.gap_id
    assert led.view.status[first.gap_id] is M.Status.BLOCKED                                     # stays blocked: no third identical try
    P.unblock(led, first.gap_id, "new strategy")                                                 # new evidence reopens it
    assert P.plan_next(led, model, "b", SPECS3, steps=("exists",), max_attempts=9) is not None


def test_different_reasons_are_not_a_stall(world) -> None:
    led, model, _ = world
    a = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert a is not None
    P.record_outcome(led, a, False, "worker produced a stub")
    b = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert b is not None and b.gap_id == a.gap_id
    P.record_outcome(led, b, False, "tests failed")
    assert RE.stalled(led, P.attempts_for(led, a.gap_id)) == ""


def test_a_student_change_identical_to_a_rejected_one_is_not_evaluated_again(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    cur = CUR.Curriculum(tmp_path / "lessons.jsonl", [FakeStudent("stu")])
    CUR.LessonLog(tmp_path / "lessons.jsonl").add(CUR.Lesson("L0", "P0", "C1", "gap", "o", files_before={"m.py": "x = 1\n"},
                                                            files_after={"m.py": "x = 2\n"}, solver="claude", adopted=False,
                                                            verdict="REJECTED: regression"))
    step = cur.student_steps()[0]
    res = step(cplan("P1"), PKG, repo)
    assert not res.claimed_done and "identical to rejected lesson L0" in res.notes and RE.NEEDS_TEACHER in res.notes
    assert (repo / "m.py").read_text(encoding="utf-8") == "x = 1\n"                              # sandbox restored, nothing to evaluate
    assert RE.identical_failed([], "C1", {}, {"m.py": "x"}) is None


def test_prediction_and_outcome_are_stored_per_lesson_and_calibration_is_computed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)

    class Predicting(FakeStudent):
        def __call__(self, plan: Any, package: Any, workdir: Path) -> K.WorkResult:
            r = super().__call__(plan, package, workdir)
            return K.WorkResult(r.claimed_done, r.notes, by=self.name, predicted={"size_delta": 0.0})

    cur = CUR.Curriculum(tmp_path / "lessons.jsonl", [Predicting("pred")])
    cur.student_steps()[0](cplan("P1"), PKG, repo)
    les = CUR.LessonLog(tmp_path / "lessons.jsonl").lessons()[0]
    assert les.predicted == {"size_delta": 0.0} and les.measured == {"size_delta": 0.0}          # x = 1 -> x = 2: same node count
    mk = lambda i, s, p, m: CUR.Lesson(f"c{i}", f"P{i}", "C1", "shrink", "o", solver=s, claimed_done=True, adopted=False, verdict="REJECTED: x",
                                       predicted=p, measured=m, at=dt.datetime(2026, 10, 2, 11, 0).isoformat())
    lessons = [mk(1, "good", {"size_delta": -10.0}, {"size_delta": -9.0}), mk(2, "good", {"size_delta": -10.0}, {"size_delta": -12.0}),
               mk(3, "bad", {"size_delta": -50.0}, {"size_delta": 7.0}), mk(4, "bad", {}, {"size_delta": -3.0}),
               mk(5, "bad", {"size_delta": 5.0}, {"size_delta": 5.0})]
    cal = RE.calibration(lessons)
    assert cal["good"]["score"] == 1.0 and cal["good"]["coverage"] == 1.0
    assert cal["bad"]["claimed"] == 3 and cal["bad"]["predicted"] == 2 and cal["bad"]["well_calibrated"] == 1 and cal["bad"]["score"] == 0.333
    state = tmp_path / "st"
    for x in lessons:
        CUR.LessonLog(state / "lessons.jsonl").add(x)
    rep = CON.measure_all(state, dt.datetime(2026, 10, 2, 12, 0), 24.0)
    m = {x["name"]: x for x in rep["ranked"] + rep["drivers"]}["calibration"]
    assert m["loss"] == pytest.approx(1 - 3 / 5) and set(m["detail"]["per_student"]) == {"good", "bad"}


def test_size_delta_and_diff_hash_basics() -> None:
    assert RE.size_delta({"a.py": "x = 1\ny = 2\n"}, {"a.py": "x = 1\n"}) < 0
    assert RE.nodes("    return 1") > 0                                                           # a snippet that does not parse alone
    assert RE.diff_hash({"a.py": "1"}, {"a.py": "2 "}) == RE.diff_hash({"a.py": "1"}, {"a.py": "2"})
    assert "Thinking procedure v1" in RE.procedure_text() and "Record the lesson" in RE.procedure_text(compact=False)


def test_action_student_fills_predicted_from_prescreen(tmp_path):
    from creator import action_student as AS
    from creator.kernel import WorkResult
    assert WorkResult(True, "x").predicted is None
    p = AS.parse_prediction(f"{AS.PREDICT_TAG} metric=size size_delta=-4 act_delta=na")
    assert p is not None and p["size_delta"] == -4
