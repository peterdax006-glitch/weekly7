"""CR151 (recursive-improvement audit) and CR204 (no critical failure stays hidden): the audit checks added next to the existing ones.
Every test here fails without creator/audit/checks.py check_recursion / check_unresolved_failures. Temporary ledgers only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import model as M
from creator import process_levers as PL
from creator import recursion as R
from creator.audit import checks as A
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-02T00:00:00+00:00")
    monkeypatch.setattr("creator.ledger.current_provenance", lambda *a, **k: prov)


@pytest.fixture()
def led(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", evidence_root=tmp_path)


def plant_weakness(led: Ledger) -> None:
    obj = led.append(M.Objective(created_by=M.Role.OWNER, statement="plant", acceptance_criteria=("x",)))
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="a", candidate_ref="b"))
    for why in ["INSUFFICIENT_EVIDENCE: no holdout"] * 4:
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.REJECT, reason=why))


def real_step(led: Ledger) -> R.StepReport:
    plant_weakness(led)
    rep = R.step(led, R.make_workload(), min_n=3)
    assert rep.adopted and rep.weakness is not None, (rep.verdict, rep.detail)
    return rep


def crit(found: list[A.AuditFinding]) -> list[str]:
    return [f.detail for f in found if f.severity == "CRITICAL"]


def test_a_real_adopted_step_with_matching_process_json_is_clean(led: Ledger, tmp_path: Path) -> None:
    real_step(led)
    pj = tmp_path / "process.json"
    PL.save_process(R.current_process(led), pj)
    assert A.check_recursion(led, process_path=pj) == []


def test_a_ledger_without_recursion_and_without_process_json_is_clean(led: Ledger, tmp_path: Path) -> None:
    plant_weakness(led)
    assert A.check_recursion(led, process_path=tmp_path / "process.json") == []


def test_process_json_changed_by_hand_is_critical(led: Ledger, tmp_path: Path) -> None:
    real_step(led)
    pj = tmp_path / "process.json"
    PL.save_process(R.current_process(led).with_("max_retries", 4), pj)                # not what the evidence adopted
    found = A.check_recursion(led, process_path=pj)
    assert any("not the last adopted process" in d for d in crit(found))


def test_process_json_without_any_adopted_step_is_critical(led: Ledger, tmp_path: Path) -> None:
    plant_weakness(led)
    pj = tmp_path / "process.json"
    pj.write_text(json.dumps({"max_retries": 5}), encoding="utf-8")
    assert any("no adopted, measured step" in d for d in crit(A.check_recursion(led, process_path=pj)))


def test_an_adopted_step_citing_a_rejected_claim_is_critical(led: Ledger) -> None:
    """A step adopted although its claim is not an IMPROVEMENT: recorded straight into a Finding."""
    plant_weakness(led)
    rep = R.step(led, R.make_workload(), min_n=3, forced=R.Change("reviewer_depth", 1, 1, "no-op"))      # same config: NO_EFFECT
    assert not rep.adopted
    step = R.recorded_steps(led)[-1]
    step.update(adopted=True, process_after={**step["process_before"], "reviewer_depth": 1})
    forged = {**step, "change": {**step["change"], "new": 1}}
    led.append(M.Finding(created_by=M.Role.KERNEL, parents=(step["experiment"],), uncertainty=M.Uncertainty.LIKELY,
                         statement=R.STEP_TAG + json.dumps(forged, sort_keys=True)))
    found = crit(A.check_recursion(led, process_path=Path("absent.json")))
    assert any("not IMPROVEMENT" in d or "does not change the process" in d for d in found), found


def test_an_adopted_step_without_decision_or_claim_is_critical(led: Ledger) -> None:
    rep = real_step(led)
    step = R.recorded_steps(led)[-1]
    forged = {**step, "iteration": 2, "claim": "IC-doesnotexist", "process_before": step["process_after"],
              "process_after": {**step["process_after"], "max_retries": step["process_after"]["max_retries"] + 1},
              "change": {"param": "max_retries", "old": 3, "new": 4, "rationale": "x"}}
    led.append(M.Finding(created_by=M.Role.KERNEL, parents=(rep.experiment_id or "",), uncertainty=M.Uncertainty.LIKELY,
                         statement=R.STEP_TAG + json.dumps(forged, sort_keys=True)))
    assert any("no ImprovementClaim" in d for d in crit(A.check_recursion(led, process_path=Path("absent.json"))))


def test_a_step_that_starts_from_a_process_nobody_adopted_is_critical(led: Ledger) -> None:
    rep = real_step(led)
    step = R.recorded_steps(led)[-1]
    forged = {**step, "iteration": 2, "process_before": {**step["process_after"], "reviewer_depth": 3}}
    led.append(M.Finding(created_by=M.Role.KERNEL, parents=(rep.experiment_id or "",), uncertainty=M.Uncertainty.LIKELY,
                         statement=R.STEP_TAG + json.dumps(forged, sort_keys=True)))
    assert any("lineage broken" in d for d in crit(A.check_recursion(led, process_path=Path("absent.json"))))


def test_a_forced_step_without_a_weakness_is_flagged_high_not_hidden(led: Ledger) -> None:
    plant_weakness(led)
    rep = R.step(led, R.make_workload(), forced=R.Change("max_retries", 2, 3, "probe"))
    assert rep.adopted
    found = A.check_recursion(led, process_path=Path("absent.json"))
    assert [(f.severity, f.check) for f in found if "weakness" in f.detail] == [("HIGH", "recursion")]


def test_the_recursion_checks_run_inside_the_audit_and_the_adversary_attacks_are_caught(led: Ledger) -> None:
    assert {"recursion", "unresolved_failures"} <= set(A.CHECKS)
    plant_weakness(led)
    assert A.audit(led, only=("recursion", "unresolved_failures")).clean
    res = {a.name: a for a in A.adversary(tuple(x for x in A.ATTACKS if x[0] in ("unevidenced_process_change", "hidden_critical_failure")))}
    assert set(res) == {"unevidenced_process_change", "hidden_critical_failure"} and all(a.caught for a in res.values()), res


def test_the_attacks_are_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch each defence off and the matching attack must stop being caught (the adversary is not asserting by default)."""
    monkeypatch.setattr(A, "check_recursion", lambda led, **_: [])
    monkeypatch.setattr(A, "check_unresolved_failures", lambda led, **_: [])
    res = {a.name: a.caught for a in A.adversary(tuple(x for x in A.ATTACKS if x[0] in ("unevidenced_process_change", "hidden_critical_failure")))}
    assert res == {"unevidenced_process_change": False, "hidden_critical_failure": False}


# ------------------------------------------------------------------------------------------------ CR204

def world(led: Ledger) -> tuple[str, str]:
    obj = led.append(M.Objective(created_by=M.Role.OWNER, statement="w", acceptance_criteria=("x",)))
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="a", candidate_ref="b"))
    return gap, ex


def rollback(led: Ledger, ex: str) -> str:
    return led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ROLLBACK, reason="audit red after merge"))


def failure(led: Ledger, subject: str, cls: str = "post-merge rollback") -> str:
    return led.append(M.Failure(created_by=M.Role.DEBUGGER, parents=(subject,), subject_id=subject, symptom="s", classification=cls,
                                reproduction="r"))


def diagnose(led: Ledger, fid: str) -> str:
    return led.append(M.Diagnosis(created_by=M.Role.DEBUGGER, parents=(fid,), failure_id=fid, hypotheses=("h",), root_cause="rc"))


def test_an_unresolved_rollback_is_critical(led: Ledger) -> None:
    _gap, ex = world(led)
    rb = rollback(led, ex)
    found = A.check_unresolved_failures(led)
    assert [(f.severity, f.subject) for f in found] == [("CRITICAL", rb)]
    rep = A.failure_report(led)
    assert rep["critical"] == 1 and rep["critical_unresolved"][0]["id"] == rb          # reported in STATUS


def test_diagnosis_plus_repair_resolves_but_diagnosis_alone_does_not(led: Ledger) -> None:
    _gap, ex = world(led)
    rollback(led, ex)
    fid = failure(led, ex)
    assert A.check_unresolved_failures(led)                                            # failure with no diagnosis
    did = diagnose(led, fid)
    assert {f.subject for f in A.check_unresolved_failures(led)}                        # a diagnosis without a fix is not a resolution
    led.append(M.Repair(created_by=M.Role.IMPLEMENTER, parents=(did,), diagnosis_id=did, description="reverted"))
    assert A.check_unresolved_failures(led) == []
    assert A.failure_report(led)["critical_unresolved"] == []


def test_a_diagnosis_of_another_failure_does_not_resolve(led: Ledger) -> None:
    gap, ex = world(led)
    rollback(led, ex)
    failure(led, ex)
    decoy = failure(led, gap, "flaky")
    did = diagnose(led, decoy)
    led.append(M.Repair(created_by=M.Role.IMPLEMENTER, parents=(did,), diagnosis_id=did, description="fixed the decoy"))
    assert len(A.check_unresolved_failures(led)) == 2                                    # the rollback AND its failure


def test_only_the_owner_can_accept_a_critical_failure(led: Ledger) -> None:
    _gap, ex = world(led)
    rb = rollback(led, ex)
    led.append(M.Finding(created_by=M.Role.KERNEL, parents=(rb, ex), uncertainty=M.Uncertainty.LIKELY, statement="ACCEPTED: fine"))
    assert A.check_unresolved_failures(led)                                              # the kernel cannot accept its own failure
    led.append(M.Finding(created_by=M.Role.OWNER, parents=(rb, ex), uncertainty=M.Uncertainty.LIKELY, statement="not an acceptance"))
    assert A.check_unresolved_failures(led)
    led.append(M.Finding(created_by=M.Role.OWNER, parents=(rb, ex), uncertainty=M.Uncertainty.LIKELY, statement="ACCEPTED: known risk"))
    assert A.check_unresolved_failures(led) == []


def test_an_integrity_failure_is_critical_an_ordinary_one_is_not(led: Ledger) -> None:
    gap, _ex = world(led)
    ordinary = failure(led, gap, "ASSERTION")
    integrity = failure(led, gap, "ledger integrity breach")
    found = A.check_unresolved_failures(led)
    assert [f.subject for f in found] == [integrity]
    rep = A.failure_report(led)
    assert rep["critical"] == 1 and rep["ordinary_open"] == 1 and ordinary not in {x["id"] for x in rep["critical_unresolved"]}


def test_failed_gaps_are_listed_but_are_not_critical(led: Ledger) -> None:
    gap, _ex = world(led)
    led.transition(gap, M.Status.IN_PROGRESS, "start", M.Role.KERNEL)
    led.transition(gap, M.Status.FAILED, "attempt rejected", M.Role.KERNEL)
    items = A.enumerate_failures(led)
    assert [(i.kind, i.critical) for i in items] == [("FAILED", False)]
    assert A.check_unresolved_failures(led) == []


def test_a_rolled_back_record_is_critical_until_resolved(led: Ledger) -> None:
    gap, ex = world(led)
    for to in (M.Status.IN_PROGRESS, M.Status.FAILED, M.Status.ROLLED_BACK):
        led.transition(gap, to, "x", M.Role.KERNEL)
    assert [f.subject for f in A.check_unresolved_failures(led)] == [gap]
    led.append(M.Finding(created_by=M.Role.OWNER, parents=(gap, ex), uncertainty=M.Uncertainty.LIKELY, statement="ACCEPTED: abandoned"))
    assert A.check_unresolved_failures(led) == []
