"""CR09: the planner (C77 secs 17-19, 63, 64)."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import selfmodel as SM
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 3),
         SM.CapabilitySpec("K02", "user", ("pkg/user.py",), ("tests/test_user.py",), 3)]


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Ledger, SM.SelfModel]:
    r = tmp_path / "proj"
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n")
    put(r, "pkg/app.py", "from pkg.base import one\n\n\ndef main():\n    return one()\n")      # production use: K01 integrated
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")
    led = Ledger(r / "dev.jsonl", evidence_root=r)
    O.compile_capabilities(led, O.self_objective(led), SPECS)
    ev = SM.collect_test_evidence(r, ["tests/test_base.py"], r / "ev.json")
    model = SM.build(r, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=ev, include_versions=False)
    G.sync(led, model)
    return led, model


def test_plan_takes_the_top_unblocked_gap_and_writes_the_whole_chain(world) -> None:
    led, model = world
    top = next(g for g in G.ranked(led) if not g.blocked_by)
    plan = P.plan_next(led, model, "abc123", SPECS)
    assert plan is not None and plan.gap_id == top.gap_id and plan.package_id == "CP0001"
    wp = led.get(plan.work_package_id)
    assert wp.parents == (plan.gap_id,) and wp.completion_criteria[0].startswith(f"check:{plan.step}:{plan.component}")
    assert any("STATUS: DONE is a claim" in s for s in wp.anti_premature_completion)
    cp, ex = led.get(plan.change_proposal_id), led.get(plan.experiment_id)
    assert cp.originating_task == plan.work_package_id and ex.parents == (plan.change_proposal_id,)
    assert ex.baseline_ref == "abc123" and ex.candidate_ref == "sandbox:CP0001"
    assert led.view.status[plan.gap_id] is M.Status.IN_PROGRESS


def test_planned_gaps_are_not_planned_twice_and_ids_are_sequential(world) -> None:
    led, model = world
    a = P.plan_next(led, model, "b", SPECS)
    b = P.plan_next(led, model, "b", SPECS)
    assert a and b and a.gap_id != b.gap_id and (a.package_id, b.package_id) == ("CP0001", "CP0002")


def test_validated_gaps_go_to_the_validator_not_a_worker(world) -> None:
    led, model = world
    plan = P.plan_next(led, model, "b", SPECS, steps=P.VALIDATOR_STEPS)
    assert plan is not None and plan.step == "validated" and plan.role is M.Role.VALIDATOR
    worker = P.plan_next(led, model, "b", SPECS, steps=P.WORKER_STEPS)
    assert worker is not None and worker.role is M.Role.IMPLEMENTER


def test_failures_replan_with_reasons_then_block(world) -> None:
    led, model = world
    first = P.plan_next(led, model, "b", SPECS, steps=("exists",))
    assert first is not None
    P.record_outcome(led, first, False, "worker produced a stub")
    assert led.view.status[first.gap_id] is M.Status.FAILED
    second = P.plan_next(led, model, "b", SPECS, steps=("exists",))
    assert second is not None and second.gap_id == first.gap_id and second.attempt == 2
    assert "worker produced a stub" in led.get(second.work_package_id).why_it_exists
    assert led.get(second.work_package_id).dependencies == ("CP0001",)
    P.record_outcome(led, second, False, "tests failed")
    third = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
    assert third is None or third.gap_id != first.gap_id
    assert led.view.status[first.gap_id] is M.Status.BLOCKED
    P.unblock(led, first.gap_id, "strategy changed")
    again = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=5)
    assert again is not None and again.gap_id == first.gap_id and again.attempt == 3
    with pytest.raises(P.PlanningError):
        P.unblock(led, again.gap_id, "not blocked")


def test_success_leaves_the_gap_for_sync_to_prove(world) -> None:
    led, model = world
    plan = P.plan_next(led, model, "b", SPECS)
    assert plan is not None
    P.record_outcome(led, plan, True, "adopted")
    assert led.view.status[plan.work_package_id] is M.Status.IMPLEMENTED
    assert led.view.status[plan.gap_id] is M.Status.IN_PROGRESS            # not closed by the planner's say-so


def test_nothing_plannable(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    m = SM.build(tmp_path, scope=("none",), capabilities=[], include_versions=False)
    assert P.plan_next(led, m, "b", []) is None



def test_the_package_states_the_current_check_not_the_gap_text(world) -> None:
    """Regression (1 Oct): a gap opened before its module existed kept saying 'nothing to integrate' to the planner."""
    led, model = world
    root = Path(model.root)
    put(root, "pkg/user.py", "def use():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n")
    put(root, "tests/test_user.py", "from pkg.user import use\n\n\ndef test_use():\n    assert use() == 1\n")
    ev = SM.collect_test_evidence(root, ["tests/test_base.py", "tests/test_user.py"], root / "ev.json")
    now = SM.build(root, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=ev, include_versions=False)
    G.sync(led, now)
    plan = P.plan_next(led, now, "b", SPECS, steps=("integrated",))
    assert plan is not None and plan.requirement_key == "K02.integrated"
    assert "nothing to integrate" in led.get(plan.gap_id).description          # the frozen text
    why = led.get(plan.work_package_id).why_it_exists
    assert "not imported by production code" in why and "nothing to integrate" not in why
