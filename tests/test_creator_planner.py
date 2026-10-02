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


def _stop(led: Ledger, plan: P.Plan, reason: str) -> None:
    P.record_outcome(led, plan, False, reason)


def test_interruptions_are_not_attempts_so_a_gap_is_never_blocked_by_stops(world) -> None:
    """1 Oct: K07/K08/K12/K19 were BLOCKED after three host stops / pull-backs / pauses although no work of theirs was judged."""
    led, model = world
    for why in ("interrupted: host stopped the swarm", "stopped by the owner during evaluation", "paused: owner restarting",
                "pulled back before work (RAM tight)"):
        plan = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
        assert plan is not None
        _stop(led, plan, why)
    assert led.view.status[plan.gap_id] is M.Status.FAILED
    assert len(P.attempts_for(led, plan.gap_id)) == 0
    real = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
    assert real is not None and real.gap_id == plan.gap_id and real.attempt == 1       # stops left the count at zero


def test_a_gap_blocked_by_stops_is_reopened_but_a_really_exhausted_one_stays_blocked(world) -> None:
    led, model = world
    plan = P.plan_next(led, model, "b", SPECS, steps=("exists",))
    assert plan is not None
    _stop(led, plan, "interrupted: host stop")
    led.transition(plan.gap_id, M.Status.BLOCKED, "3 packages failed: (old count included stops)", M.Role.KERNEL)
    again = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
    assert again is not None and again.gap_id == plan.gap_id and led.view.status[again.gap_id] is M.Status.IN_PROGRESS
    _stop(led, again, "tests failed")
    third = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
    assert third is not None and third.gap_id == plan.gap_id
    _stop(led, third, "tests failed again")
    other = P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2)
    assert other is None or other.gap_id != plan.gap_id
    assert led.view.status[plan.gap_id] is M.Status.BLOCKED                  # two real failures: blocked, not retried forever
    assert P.plan_next(led, model, "b", SPECS, steps=("exists",), max_attempts=2) is None


def test_efficiency_falls_back_to_the_other_kind_when_the_preferred_one_has_no_target(monkeypatch, tmp_path: Path) -> None:
    """1 Oct: the preferred kind's only target (kernel.py activation) was held, so the round ended at 2 packages."""
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(P, "_plan_activation", lambda *a, **k: calls.append("activation"))
    monkeypatch.setattr(P, "_plan_size", lambda *a, **k: calls.append("size") or "SIZE-PLAN")
    assert P.plan_efficiency(led, tmp_path, "b") == "SIZE-PLAN" and calls == ["size"]
    calls.clear()
    monkeypatch.setattr(P, "_plan_size", lambda *a, **k: calls.append("size"))
    monkeypatch.setattr(P, "_plan_activation", lambda *a, **k: calls.append("activation") or "ACT-PLAN")
    assert P.plan_efficiency(led, tmp_path, "b") == "ACT-PLAN" and calls == ["size", "activation"]
    calls.clear()
    assert P.plan_efficiency(led, tmp_path, "b", kind="size") is None and calls == ["size"]       # an explicit kind never falls back


def test_a_pulled_back_target_is_not_skipped_but_a_failed_one_is(world) -> None:
    """recent_failed_targets skips targets whose shrink FAILED; a RAM pull-back judged nothing, so it must stay plannable."""
    led, model = world
    objective_id = next(e.id for e in led.of_type("Objective") if e.record.statement == O.SELF_STATEMENT)
    req = P.efficiency_requirement(led, objective_id)
    out = {}
    for path, why in (("pkg/a.py", "pulled back before evaluation (RAM tight)"), ("pkg/b.py", "claim REGRESSION")):
        gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(req,), kind=M.GapKind.ARCHITECTURE, description=f"shrink {path}",
                               importance=0.3))
        wp = led.append(M.WorkPackage(created_by=M.Role.KERNEL, parents=(gap,), package_id=P.next_package_id(led),
                                      objective=f"shrink {path} without losing capability", why_it_exists="t", prerequisites=("none",),
                                      inputs=(path,), outputs=(path,), implementation_requirements=("x",), interfaces=("x",),
                                      data_flow="x", dependencies=(), test_requirements=("x",), validation_requirements=("x",),
                                      expected_failure_modes=("x",), evidence_requirements=("x",), failure_conditions=("x",),
                                      rollback_requirements=("x",), completion_criteria=("x",), anti_premature_completion=("x",),
                                      meaningful_code_depth=0))
        led.transition(wp, M.Status.IN_PROGRESS, "go", M.Role.KERNEL)
        led.transition(wp, M.Status.FAILED, why, M.Role.KERNEL)
        out[path] = wp
    assert P.recent_failed_targets(led) == ["pkg/b.py"]


def test_the_coverage_planner_picks_the_module_with_the_biggest_test_gap(tmp_path: Path) -> None:
    put(tmp_path, "creator/__init__.py", "")
    put(tmp_path, "creator/a.py", "def one():\n    return 1\n")
    put(tmp_path, "creator/b.py", "def two():\n    return 2\n\n\ndef three():\n    return 3\n")
    put(tmp_path, "tests/test_a.py", "def test_x():\n    assert True\n")
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    O.self_objective(led)
    plan = P._plan_coverage(led, tmp_path, "base")
    assert plan is not None and plan.step == "coverage" and plan.requirement_key == P.COVERAGE_KEY
    wp = led.get(plan.work_package_id)
    assert plan.component == "creator/b.py" and "two, three" in wp.implementation_requirements[0]
    assert P._plan_coverage(led, tmp_path, "base", avoid=("creator/b.py",)).component == "creator/a.py"
    put(tmp_path, "tests/test_all.py", "import creator\none, two, three\n")
    assert P._plan_coverage(led, tmp_path, "base") is None                    # nothing left to cover


# ------------------------------------------------------------------------------------------------ K08 / K12 / K07 in the loop

SPECS3 = [SPECS[0], SM.CapabilitySpec("K05", "user", ("pkg/user.py",), ("tests/test_user.py",), 3),
          SM.CapabilitySpec("K06", "third", ("pkg/third.py",), ("tests/test_third.py",), 3)]


@pytest.fixture()
def world3(tmp_path: Path) -> tuple[Ledger, SM.SelfModel]:
    """Two components (K05, K06) whose modules do not exist yet: two parallel 'exists' gaps. A creator/ file mentions user.py."""
    r = tmp_path / "proj"
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n")
    put(r, "pkg/app.py", "from pkg.base import one\n")
    put(r, "creator/caller.py", "from pkg import user  # import user\nfrom pkg import third  # import third\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")
    led = Ledger(r / "dev.jsonl", evidence_root=r)
    O.compile_capabilities(led, O.self_objective(led), SPECS3)
    model = SM.build(r, scope=("pkg", "tests"), capabilities=SPECS3,
                     test_evidence=SM.collect_test_evidence(r, ["tests/test_base.py"], r / "ev.json"), include_versions=False)
    G.sync(led, model)
    return led, model


def test_design_selects_an_approach_records_every_option_and_changes_with_failure(world3) -> None:
    led, model = world3
    first = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert first is not None
    text = " ".join(led.get(first.work_package_id).implementation_requirements)
    assert "Approach chosen by design.select" in text and ": direct - " in text and "rejected: " in text
    opts = [e for e in led.of_type("DesignOption") if e.record.parents == (first.work_package_id,)]
    assert {o.record.summary.split(":")[0] for o in opts} == {"direct", "test-first", "diagnose-first"}
    rejects = [d for d in led.of_type("Decision") if d.record.verdict is M.DecisionVerdict.REJECT]
    assert len(rejects) == 2 and all("scored" in d.record.reason for d in rejects)
    P.record_outcome(led, first, False, "worker produced a stub")
    again = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert again is not None and again.gap_id == first.gap_id and again.attempt == 2
    assert ": diagnose-first - " in " ".join(led.get(again.work_package_id).implementation_requirements)


def test_memory_puts_what_failed_before_for_another_component_into_the_package(world3) -> None:
    led, model = world3
    a = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert a is not None
    P.record_outcome(led, a, False, "worker produced a stub")
    b = P.plan_next(led, model, "b", SPECS3, steps=("exists",), exclude_components=(a.component,))
    assert b is not None and b.gap_id != a.gap_id and b.component != a.component
    modes = led.get(b.work_package_id).expected_failure_modes
    assert any(m.startswith("Memory - failed before (CP0001): worker produced a stub") for m in modes)


def test_a_blocked_gap_becomes_researched_and_the_findings_reach_the_next_package(world3) -> None:
    led, model = world3
    p1 = P.plan_next(led, model, "b", SPECS3, steps=("exists",))
    assert p1 is not None and p1.component in ("K05", "K06")
    cid = p1.component
    P.record_outcome(led, p1, False, "worker produced a stub")
    p2 = P.plan_next(led, model, "b", SPECS3, steps=("exists",), max_attempts=2)
    assert p2 is not None and p2.gap_id == p1.gap_id
    P.record_outcome(led, p2, False, "tests failed")
    other = P.plan_next(led, model, "b", SPECS3, steps=("exists",), max_attempts=2)       # the block happens while planning
    assert led.view.status[p1.gap_id] is M.Status.BLOCKED
    rqs = [c for c in led.view.children[p1.gap_id] if led.view.by_id[c].rtype == "ResearchQuestion"]
    assert len(rqs) == 1 and "How should we close" in led.get(rqs[0]).question
    finding = led.get(next(c for c in led.view.children[rqs[0]] if led.view.by_id[c].rtype == "Finding"))
    assert cid in finding.statement and finding.sources and finding.sources[0].startswith("creator/")
    blocked = [t for t in led.about(p1.gap_id) if t.rtype == "Transition" and t.record.to_state is M.Status.BLOCKED]
    assert rqs[0] in blocked[-1].record.reason
    P.unblock(led, p1.gap_id, "strategy changed")
    again = P.plan_next(led, model, "b", SPECS3, steps=("exists",), max_attempts=5)
    assert again is not None and again.gap_id == p1.gap_id
    assert any(s.startswith("Research - ") and finding.statement in s for s in led.get(again.work_package_id).implementation_requirements)
    assert other is None or other.gap_id != p1.gap_id
