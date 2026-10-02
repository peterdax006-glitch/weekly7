"""K27: goal generation - proposals from planted evidence, duplicates refused, approval compiles real work, rejection is permanent.
Temporary ledgers and project trees only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import curriculum as CUR
from creator import gaps as G
from creator import goals as GO
from creator import model as M
from creator import selfmodel as SM
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-02T00:00:00+00:00")
    monkeypatch.setattr("creator.ledger.current_provenance", lambda *a, **k: prov)


SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 0)]


@pytest.fixture()
def world(tmp_path: Path) -> tuple[Ledger, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "creator").mkdir(parents=True)
    (repo / "creator" / "capabilities.json").write_text(json.dumps({"capabilities": [
        {"id": "K01", "name": "base", "modules": ["pkg/base.py"], "tests": ["tests/test_base.py"], "floor": 0}]}))
    state = repo / "state" / "creator"
    state.mkdir(parents=True)
    return Ledger(state / "ledger.jsonl", evidence_root=repo), state, repo


def plant_failures(led: Ledger, cls: str, n: int, diagnose: str = "") -> list[str]:
    ids = []
    subject = led.append(M.Objective(created_by=M.Role.OWNER, statement=f"subject {cls} {len(led.of_type('Objective'))}",
                                     acceptance_criteria=("x",)))
    for i in range(n):
        fid = led.append(M.Failure(created_by=M.Role.TESTER, subject_id=subject, symptom=f"s{i}", classification=cls, reproduction="r"))
        ids.append(fid)
        if diagnose:
            led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid, hypotheses=("h",), root_cause=diagnose))
    return ids


def plant_checklist(repo: Path) -> None:
    (repo / "state" / "build").mkdir(parents=True)
    items = [{"id": f"CR{i}", "group": "SECTION / 9. TELEPATHY LINK", "status": "NOT_STARTED", "code_paths": [],
              "description": "build the telepathy link"} for i in range(4)]
    items += [{"id": f"CR5{i}", "group": f"SECTION / {i}. DOCTRINE", "status": "NOT_STARTED", "code_paths": [],
               "description": f"SECTION REQUIREMENT: {i}. DOCTRINE - every requirement in this section"} for i in range(28)]
    items.append({"id": "CR9", "group": "SECTION / 9. TELEPATHY LINK", "status": "TESTING", "code_paths": ["x.py"]})
    items.append({"id": "CR10", "group": "SECTION / mapped", "status": "NOT_STARTED", "code_paths": ["y.py"]})
    (repo / "state" / "build" / "CREATOR_MASTER_CHECKLIST.json").write_text(json.dumps({"items": items}))


def plant_lessons(state: Path, kind: str, solver: str, n: int, adopted: bool) -> None:
    log = CUR.LessonLog(state / "lessons.jsonl")
    for i in range(n):
        les = CUR.Lesson(lesson_id=f"{solver}{kind}{i}", package_id="p", component="K01", task_kind=kind, objective="o", solver=solver)
        log.add(les)
        log.outcome(les.lesson_id, adopted, "ADOPT" if adopted else "REJECT: no")


def by_source(props: list[GO.Proposal]) -> dict[str, GO.Proposal]:
    return {p.source: p for p in props}


def test_failure_source_proposes_an_uncovered_recurring_class(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    ids = plant_failures(led, "FLAKY_NETWORK", 4)
    plant_failures(led, "RARE", 1)                                     # below the recurrence threshold
    p = by_source(GO.propose(led, state, repo, SPECS))["failure"]
    assert p.key == "failure:FLAKY_NETWORK" and set(p.evidence) == set(ids) and "4 of 5" in p.rationale
    assert p.spec["modules"] and p.spec["tests"] and p.spec["ladder"][0] == "exists" and 0 < p.value <= 1 and p.cost["packages"] == 5


def test_a_failure_class_a_capability_already_names_is_not_proposed(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_failures(led, "BASE", 5)
    assert GO.propose(led, state, repo, SPECS) == []


def test_weakness_source_needs_exhausted_remedies(world: tuple[Ledger, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    led, state, repo = world
    obj = led.append(M.Objective(created_by=M.Role.OWNER, statement="plant", acceptance_criteria=("x",)))
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="a", candidate_ref="b"))
    for _ in range(4):
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.REJECT,
                              reason="NO_EFFECT: nothing"))
    assert "weakness" not in by_source(GO.propose(led, state, repo, SPECS))          # remedies still open: recursion owns it
    every = {(p, v) for p in GO.R.PARAMS for v in range(0, 20)}                       # every parameter step already tried
    monkeypatch.setattr(GO.R, "tried_changes", lambda _led: every)
    p = by_source(GO.propose(led, state, repo, SPECS))["weakness"]
    assert p.key == "weakness:NO_EFFECT" and len(p.evidence) == 4 and all(i in led.view.by_id for i in p.evidence)


def test_section_source_uses_open_unmapped_checklist_items(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_checklist(repo)
    props = GO.propose(led, state, repo, SPECS)
    assert [p.key for p in props] == ["section:9. TELEPATHY LINK"]               # mapped and doctrine meta sections are not gaps
    assert set(props[0].evidence) == {"CR0", "CR1", "CR2", "CR3"} and "4 of 5" in props[0].rationale


def test_student_source_needs_every_student_to_fail(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_lessons(state, "tests", "alpha", 3, False)
    plant_lessons(state, "tests", "beta", 2, False)
    plant_lessons(state, "gap", "alpha", 3, False)
    plant_lessons(state, "gap", "beta", 1, True)                                      # one student succeeds: no proposal
    plant_lessons(state, "shrink", CUR.CLAUDE, 5, False)                              # the teacher alone does not count
    props = GO.propose(led, state, repo, SPECS)
    assert [p.key for p in props] == ["student:tests"] and "5 attempts" in props[0].rationale


def test_knowledge_source_counts_unexplained_diagnoses(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_failures(led, "UNKNOWN", 3, diagnose="undetermined: no hypothesis fits")
    plant_failures(led, "EXPLAINED", 4, diagnose="a real cause")
    keys = {p.key for p in GO.propose(led, state, repo, SPECS)}
    assert "knowledge:UNKNOWN" in keys and "knowledge:EXPLAINED" not in keys


def test_no_duplicates_and_rejection_is_permanent(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_failures(led, "FLAKY_NETWORK", 4)
    first = GO.propose(led, state, repo, SPECS)
    assert len(first) == 1 and GO.propose(led, state, repo, SPECS) == []              # same evidence: nothing new
    with pytest.raises(GO.GoalError):
        GO.reject(state, first[0].id, "  ")
    GO.reject(state, first[0].id, "not worth it")
    plant_failures(led, "FLAKY_NETWORK", 6)                                           # more evidence, still rejected
    assert GO.propose(led, state, repo, SPECS) == []
    assert GO.listing(state)[0]["status"] == "REJECTED" and GO.listing(state)[0]["reason"] == "not worth it"
    with pytest.raises(GO.GoalError):
        GO.approve(state, repo, led, first[0].id, "owner")                            # a rejection cannot be reversed


def test_a_proposal_of_an_existing_module_is_refused(world: tuple[Ledger, Path, Path]) -> None:
    led, state, repo = world
    plant_failures(led, "FLAKY_NETWORK", 4)
    clash = [SM.CapabilitySpec("K01", "other", ("creator/goal_failure_flaky_network.py",), (), 0)]
    assert GO.propose(led, state, repo, clash) == []


def test_unapproved_proposal_creates_no_work_and_approved_one_is_planned(world: tuple[Ledger, Path, Path], tmp_path: Path) -> None:
    led, state, repo = world
    plant_failures(led, "FLAKY_NETWORK", 4)
    p = GO.propose(led, state, repo, SPECS)[0]
    before = len(led.of_type("Requirement"))
    assert before == 0 and SM.load_capabilities(GO.approved_path(repo)) == []          # pending proposals add no requirement
    with pytest.raises(GO.GoalError):
        GO.approve(state, repo, led, p.id, "kernel")                                  # only owner or teacher may approve
    cap = GO.approve(state, repo, led, p.id, "teacher")
    assert cap == "K02"                                                               # next id after the declared K01
    assert led.of_type("Objective")[-1].record.created_by is M.Role.BUILDER
    assert "APPROVED GOAL " + p.id in led.of_type("Objective")[-1].record.statement
    assert [s.id for s in SM.load_capabilities(GO.approved_path(repo))] == ["K02"]
    assert led.view.unique.get(("Requirement", "K02.exists")) and led.view.unique.get(("Requirement", "K02.validated"))
    # the existing gap machinery now has work for it, with no other change
    proj = tmp_path / "proj"
    for rel, text in (("pkg/__init__.py", ""), ("pkg/base.py", "def one():\n    return 1\n"), ("tests/__init__.py", ""),
                      ("tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text(text)
    specs = SPECS + SM.load_capabilities(GO.approved_path(repo))
    model = SM.build(proj, scope=("pkg", "tests"), capabilities=specs, test_evidence={}, include_versions=False)
    G.sync(led, model)
    assert any(r.requirement_key == "K02.exists" for r in G.ranked(led))             # a Gap the planner will plan
    with pytest.raises(GO.GoalError):
        GO.approve(state, repo, led, p.id, "owner")                                   # decided once
    assert GO.pending(state) == [] and GO.listing(state)[0]["capability_id"] == "K02"


def test_maybe_propose_runs_at_most_once_a_day(world: tuple[Ledger, Path, Path]) -> None:
    import datetime as dt
    led, state, repo = world
    plant_failures(led, "FLAKY_NETWORK", 4)
    t0 = dt.datetime(2026, 10, 2, 8, 0, 0)
    assert len(GO.maybe_propose(led, state, repo, t0)) == 1
    plant_failures(led, "OTHER_CLASS", 4)
    assert GO.maybe_propose(led, state, repo, t0 + dt.timedelta(hours=5)) == []        # too soon
    assert len(GO.maybe_propose(led, state, repo, t0 + dt.timedelta(hours=25))) == 1
