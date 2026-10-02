"""K20: recursion over the development process (CR191-198). Temporary ledgers; a fixed deterministic workload."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import model as M
from creator import recursion as R
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Computing real provenance (git + tree hashes) per append would take minutes for the hundreds of records here."""
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-01T00:00:00+00:00")
    monkeypatch.setattr("creator.ledger.current_provenance", lambda *a, **k: prov)


@pytest.fixture()
def led(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", evidence_root=tmp_path)


def plant(led: Ledger) -> None:
    """Evidence of a weak process: mostly INSUFFICIENT_EVIDENCE rejections, one REGRESSION, a strategy failing one class."""
    obj = led.append(M.Objective(created_by=M.Role.OWNER, statement="plant", acceptance_criteria=("x",)))
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="a", candidate_ref="b"))
    for why in ["INSUFFICIENT_EVIDENCE: no holdout"] * 4 + ["REGRESSION: guard"] + ["INSUFFICIENT_EVIDENCE: benefit not reproduced"]:
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.REJECT, reason=why))
    for i in range(5):
        led.append(M.StrategyOutcome(created_by=M.Role.KERNEL, parents=(ex,), strategy_id="s1", problem_class="bugfix",
                                     subject_id=ex, success=i == 0, cost=1.0, duration_s=1.0))
        led.append(M.StrategyOutcome(created_by=M.Role.KERNEL, parents=(ex,), strategy_id="s1", problem_class="feature",
                                     subject_id=ex, success=True, cost=1.0, duration_s=1.0))


def test_weakness_is_found_from_planted_evidence(led: Ledger) -> None:
    plant(led)
    ws = R.find_weaknesses(led)
    kinds = {w.kind: w for w in ws}
    assert kinds["rejection:INSUFFICIENT_EVIDENCE"].count == 5 and kinds["rejection:INSUFFICIENT_EVIDENCE"].share == pytest.approx(5 / 6)
    assert kinds["worker"].key == "s1/bugfix" and kinds["worker"].count == 4          # lowest success rate, not the healthy class
    assert ws[0].kind == "rejection:INSUFFICIENT_EVIDENCE"                              # worst first
    assert all(i in led.view.by_id for w in ws for i in w.evidence_ids)                  # cites real records
    ch = R.design_change(ws[0], R.ProcessConfig(), set())
    assert ch is not None and (ch.param, ch.old, ch.new) == ("max_retries", 2, 3)
    assert R.design_change(ws[0], R.ProcessConfig(), {("max_retries", 3)}).param == "reviewer_depth"   # type: ignore[union-attr]


def test_no_evidence_no_weakness_no_change(led: Ledger) -> None:
    assert R.find_weaknesses(led) == []
    rep = R.step(led, R.make_workload())
    assert rep.change is None and not rep.adopted and rep.process_after == rep.process_before


def test_a_non_improving_change_is_rejected(led: Ledger) -> None:
    wl = R.make_workload()
    rep = R.step(led, wl, forced=R.Change("research_budget", 3, 4, "no effect on this workload"))   # not read by the workload
    assert rep.verdict in ("NO_EFFECT", "INSUFFICIENT_EVIDENCE") and not rep.adopted
    assert rep.process_after == R.ProcessConfig()
    decs = [e.record for e in led.of_type("Decision")]
    assert [d.verdict for d in decs] == [M.DecisionVerdict.REJECT]                      # type: ignore[union-attr]
    assert R.current_process(led) == R.ProcessConfig()
    worse = R.step(led, wl, forced=R.Change("max_retries", 2, 0, "fewer retries"))
    assert worse.verdict == "REGRESSION" and not worse.adopted


def test_an_improving_change_is_adopted_and_used_by_the_next_iteration(led: Ledger) -> None:
    plant(led)
    wl = R.make_workload()
    reps = R.run(led, wl, iterations=2)
    first, second = reps
    assert first.change is not None and (first.change.param, first.change.new) == ("max_retries", 3)
    assert first.verdict == "IMPROVEMENT" and first.adopted
    assert first.process_after.max_retries == 3
    assert second.process_before == first.process_after                                  # iteration 2 starts from iteration 1's adoption
    assert second.change is not None
    assert (second.change.param, second.change.new) != ("max_retries", 3)               # never re-proposes a tried change
    assert second.process_after.max_retries >= 3
    adopt = [e.record for e in led.of_type("Decision") if e.record.verdict is M.DecisionVerdict.ADOPT]   # type: ignore[union-attr]
    assert adopt and all(d.claim_id for d in adopt)                                      # type: ignore[union-attr]
    assert R.current_process(led) == second.process_after                                # rebuilt from the ledger alone
    assert led.verify() > 0


def test_the_adoption_gate_cannot_be_bypassed(led: Ledger) -> None:
    from creator.ledger import LedgerError
    obj = led.append(M.Objective(created_by=M.Role.OWNER, statement="o", acceptance_criteria=("x",)))
    gap = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(obj,), kind=M.GapKind.CAPABILITY, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(gap,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="a", candidate_ref="b"))
    claim, verdict, _ = R.ab_test(led, ex, R.ProcessConfig(), R.ProcessConfig(research_budget=9), R.make_workload())
    assert verdict is not M.Verdict.IMPROVEMENT
    with pytest.raises(LedgerError):
        led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, claim_id=claim, reason="x"))


def test_min_effect_makes_adoption_stricter_and_is_recorded_on_the_claim(led: Ledger, tmp_path: Path) -> None:
    """Validator round 5: `step(min_effect=)` was untested. The same improving change that is adopted at 0 must NOT be adopted when the
    smallest gain that matters exceeds what it delivers, and the claim carries the value the verdict was computed with."""
    change = R.Change("max_retries", 2, 3, "more retries")
    ok = R.step(led, R.make_workload(), forced=change)
    assert ok.adopted and ok.verdict == "IMPROVEMENT"
    led2 = Ledger(tmp_path / "strict.jsonl", evidence_root=tmp_path)
    strict = R.step(led2, R.make_workload(), forced=change, min_effect=0.99)
    assert not strict.adopted and strict.verdict != "IMPROVEMENT" and strict.process_after == R.ProcessConfig()
    claims = [e.record for e in led2.of_type("ImprovementClaim")]
    assert len(claims) == 1 and claims[0].min_effect == 0.99                             # type: ignore[union-attr]
