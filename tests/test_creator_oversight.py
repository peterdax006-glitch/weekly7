"""K23 oversight: goal drift, knowledge gaps, critical path (C77 CR057, CR059, CR083)."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import model as M
from creator import objective as O
from creator import oversight as OV
from creator import selfmodel as SM
from creator.ledger import Ledger

K = M.Role.KERNEL


@pytest.fixture()
def led(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)


def comp() -> M.ComputationRef:
    return M.ComputationRef(function=M.VERDICT_FUNCTION, code_hash="ab" * 8, inputs_sha256="cd" * 8, output_sha256="ef" * 8)


def adopted_world(led: Ledger, objective: str, tmp: Path, key: str = "K01.exists", top: float = 1.0) -> str:
    req = led.append(M.Requirement(created_by=K, parents=(objective,), key=key, description="d", priority=M.Priority.HIGH,
                                   acceptance_test="t", measurement_method="m", failure_condition="f", validation_method="v",
                                   evidence_location="e"))
    gap = led.append(M.Gap(created_by=K, parents=(req,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=K, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="b", candidate_ref="c"))
    (tmp / "ev.txt").write_text("x", encoding="utf-8")
    ev = (M.EvidenceRef.of(tmp / "ev.txt", tmp),)

    def ms(v: float, split: M.Split = M.Split.DEV, metric: str = "m") -> str:
        return led.append(M.Measurement(created_by=M.Role.VALIDATOR, parents=(ex,), metric=metric, value=v, stderr=0.0, n=1,
                                        population="p", conditions="c", higher_is_better=True, split=split,
                                        computation=M.ComputationRef("creator.evaluate:metrics_of", "ab" * 8, "cd" * 8, "ef" * 8),
                                        evidence=ev))
    b, c = [ms(0), ms(0)], [ms(top), ms(top)]
    gb, gc = ms(1, metric="g"), ms(1, metric="g")
    hb, hc = ms(0, M.Split.HOLDOUT), ms(1, M.Split.HOLDOUT)
    claim = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(ex,), subject_id=ex, baseline_ids=tuple(b),
                                          candidate_ids=tuple(c), verdict=M.Verdict.IMPROVEMENT, computation=comp(),
                                          regression_baseline_ids=(gb,), regression_candidate_ids=(gc,),
                                          holdout_baseline_id=hb, holdout_candidate_id=hc))
    return led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="r",
                                 claim_id=claim))


def test_adoptions_that_trace_to_an_owner_objective_are_not_drift(led: Ledger, tmp_path: Path) -> None:
    adopted_world(led, O.self_objective(led), tmp_path)
    r = OV.goal_drift(led)
    assert r.adopted == 1 and r.traced_to_owner == 1 and not r.drifting


def test_an_adoption_under_a_non_owner_objective_is_drift(led: Ledger, tmp_path: Path) -> None:
    rogue = led.append(M.Objective(created_by=K, statement="something the kernel invented", acceptance_criteria=("x",)))
    dec = adopted_world(led, rogue, tmp_path)
    r = OV.goal_drift(led)
    assert r.drifting and r.untraced == (dec,) and "do not trace" in r.why


def test_concentrated_work_is_flagged(led: Ledger, tmp_path: Path) -> None:
    obj = O.self_objective(led)
    for i in range(10):
        adopted_world(led, obj, tmp_path, key=f"K{i:02d}.exists")
    r = OV.goal_drift(led, recent=10, concentration=0.8)
    assert r.concentrated == "exists" and "recent adoptions" in r.why and not r.drifting


def test_unexplained_failures_become_knowledge_gaps_once(led: Ledger) -> None:
    obj = O.self_objective(led)
    fid = led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(obj,), subject_id=obj, symptom="weird crash in planner",
                               classification="UNKNOWN", reproduction="pytest x"))
    led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid, hypotheses=("no idea",),
                           root_cause="undetermined: failure does not match a known class"))
    explained = led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(obj,), subject_id=obj, symptom="assert 1 == 2",
                                     classification="ASSERTION", reproduction="pytest y"))
    led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(explained,), failure_id=explained, hypotheses=("h",),
                           root_cause="ASSERTION failure originating at a.py:1"))
    new = OV.knowledge_gaps(led)
    assert len(new) == 1 and led.get(new[0]).kind is M.GapKind.KNOWLEDGE
    assert led.of_type("ResearchQuestion") and "weird crash" in led.of_type("ResearchQuestion")[0].record.question
    assert OV.knowledge_gaps(led) == []                                   # idempotent


def test_critical_path_puts_the_most_blocking_gap_first(led: Ledger, tmp_path: Path) -> None:
    from creator import gaps as G
    specs = [SM.CapabilitySpec("K01", "a", ("pkg/a.py",), ("tests/t.py",), 0),
             SM.CapabilitySpec("K02", "b", ("pkg/b.py",), ("tests/u.py",), 0)]
    O.compile_capabilities(led, O.self_objective(led), specs)
    (tmp_path / "pkg").mkdir()
    model = SM.build(tmp_path, scope=("pkg",), capabilities=specs, include_versions=False)
    G.sync(led, model)
    path = OV.critical_path(led)
    assert path[0].requirement_key in ("K01.exists",) and path[0].blocks >= 4             # K01 exists blocks its ladder + K02
    assert all(path[i].blocks >= path[i + 1].blocks for i in range(len(path) - 1))


def test_concentrated_work_with_a_flat_metric_level_is_flagged_as_stalled(led: Ledger, tmp_path: Path) -> None:
    obj = O.self_objective(led)
    for i in range(10):
        adopted_world(led, obj, tmp_path, key=f"K{i:02d}.exists")
    r = OV.goal_drift(led, recent=10, concentration=0.8)
    assert r.concentrated == "exists" and r.metrics_stalled and "did not rise" in r.why and not r.drifting


def test_concentrated_work_with_a_rising_metric_level_is_not_stalled(led: Ledger, tmp_path: Path) -> None:
    obj = O.self_objective(led)
    for i in range(10):
        adopted_world(led, obj, tmp_path, key=f"K{i:02d}.exists", top=1.0 + i)
    r = OV.goal_drift(led, recent=10, concentration=0.8)
    assert r.concentrated == "exists" and not r.metrics_stalled and "did not rise" not in r.why
