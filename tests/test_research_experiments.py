"""Tests for engine.research.experiments (C66 s15/s32) and engine.research.failed_lab (C66 s17). Synthetic only, no network.
Every mechanism has a planted case it must catch, a null case it must leave alone, and the empty case."""
import dataclasses

import pytest

from engine.learning import experiment_memory as em
from engine.learning import failed_learners as fl_mod
from engine.learning.core import DecisionEffect, FailureCause, FirewallBreach
from engine.learning.experiment_memory import (DesignSpec, ExperimentLedger, ExperimentResult, ExperimentStatus, Hypothesis, Prediction,
                                               ResultKind)
from engine.research import experiments as ex
from engine.research import failed_lab as lab_mod
from engine.research.core import ExperimentValue, GateVerdict
from engine.research.experiments import (AnswerGrade, Complexity, DecisionImpact, DesignExtension, LaunchDecision, LaunchRefused,
                                         MemorizationResult, OutcomeExtension, ReplicationReason, ResearchExperimentMemory, TestState,
                                         TimePeriod, TransferResult)
from engine.research.failed_lab import (Decision, FailedLearnerLab, Finding, LabProposal, ReasonBucket)

Q = "Does the volatility rank feature improve out of sample movement forecasts"
Q_OTHER = "Which sector calendar effect explains earnings drift"


def mem_new(tmp_path=None):
    return ResearchExperimentMemory(ExperimentLedger(path=(tmp_path / "led.jsonl") if tmp_path else None),
                                    ex.ExtensionStore((tmp_path / "ext.jsonl") if tmp_path else None))


def record(eid, q=Q, seed=1, cfg=None, controls=("label_shuffle",), code="c1", data="d1", now="2026-03-01", cost=5.0):
    hyps = [Hypothesis("h1", "the feature helps", 0.5), Hypothesis("h_null", "no effect beyond noise", 0.5, kind="null")]
    exp = em.uniform_expected(hyps, ["adopt", "reject"], {"h1": "adopt", "h_null": "reject"})
    design = DesignSpec(config=cfg if cfg is not None else {"depth": 3, "lr": 0.1}, windows=("w1",), seed=seed, controls=tuple(controls),
                        cost_minutes=cost, code_hash=code, data_hash=data)
    return em.new_record(eid, q, "we believe the feature is noise", hyps, Prediction("it helps", confidence=0.6), design, exp, now)


def design(period=("2015-01-01", "2020-12-31"), cutoff="2020-12-31", universe="U1", regime="R1", algo="gbm", feats=("vol_rank", "ret5"),
           rep="raw", hp=None, choices=1):
    return DesignExtension("panel_a", TimePeriod(*period), cutoff, tuple(feats), rep, algo, hp if hp is not None else {"depth": 3},
                           universe, regime, Complexity(10, len(feats), choices, 5000),
                           ExperimentValue(information_gain=0.5), DecisionImpact((DecisionEffect.RANKING,), 0.01, True))


def outcome(kind_ok=True, transfer=TestState.PASSED, mem_state=TestState.PASSED, failure=None, cost=7.0, impact=(DecisionEffect.RANKING,)):
    return OutcomeExtension(failure, TransferResult(transfer, 0.02, (0.01, 0.03) if transfer != TestState.NOT_TESTED else (), 4 if transfer != TestState.NOT_TESTED else 0),
                            MemorizationResult(mem_state, 0.03, 0.02, "disguised rerun" if mem_state != TestState.NOT_TESTED else ""),
                            ExperimentValue(information_gain=0.4), DecisionImpact(impact, 0.01, impact != (DecisionEffect.NONE,)), cost)


def result(kind=ResultKind.CONFIRMED, day="2026-03-05", ci=(0.1, 0.3), n=200):
    out = "adopt" if kind in (ResultKind.CONFIRMED,) else "reject"
    return ExperimentResult(kind, out, {"score": 0.2}, n, ci, "power for 0.1 effect: 0.9" if kind == ResultKind.NULL else "", f"{day}T00:00:00", "s")


def run(mem, eid, now="2026-03-01", close_now="2026-03-06", kind=ResultKind.CONFIRMED, ci=(0.1, 0.3), out=None, ext=None, rec=None, claims=(), stmt="",
        day="2026-03-05", n=200):
    rec = rec or record(eid, now=now)
    ext = ext or design()
    mem.register(rec, ext, now, claims, stmt)
    mem.start(eid, now)
    failure = FailureCause.FALSE_PATTERN if kind == ResultKind.REFUTED else None
    return mem.close(eid, result(kind, day, ci, n), out or outcome(failure=failure), close_now, ["it held" if kind == ResultKind.CONFIRMED else "it did not"],
                     ["mechanism not isolated"], "replicate on a fresh period", "posterior")


# ------------------------------------------------------------------ design extension

def test_design_check_catches_cutoff_problems():
    assert design().check("2026-03-01") == []
    late = design(period=("2015-01-01", "2021-06-30"), cutoff="2020-12-31")
    assert any("after information_cutoff" in e for e in late.check("2026-03-01"))
    assert any("not strictly before now" in e for e in design(cutoff="2026-03-01").check("2026-03-01"))
    assert any("features empty" in e for e in design(feats=()).check())
    assert any("n_free_choices" in e for e in Complexity(1, 1, 0, 1).check())


def test_configuration_hash_is_stable_and_sensitive():
    a, b = design(), design()
    assert a.configuration_hash == b.configuration_hash
    assert a.configuration_hash != design(algo="ridge").configuration_hash
    assert a.configuration_hash != design(hp={"depth": 4}).configuration_hash


def test_decision_impact_rules():
    assert DecisionImpact((DecisionEffect.NONE,), None, True).check()           # research-only cannot reach production
    assert DecisionImpact((DecisionEffect.NONE, DecisionEffect.RANKING)).check()
    assert DecisionImpact(()).check()
    assert DecisionImpact((DecisionEffect.RANKING,), 0.1, True).check() == []


def test_transfer_and_memorization_results_need_their_evidence():
    assert TransferResult(TestState.PASSED, 0.1, (), 0).check()
    assert TransferResult(TestState.PASSED, 0.1, (-0.1, 0.2), 3).check()         # interval includes zero cannot PASS
    assert TransferResult(TestState.NOT_TESTED).check() == []
    assert MemorizationResult(TestState.PASSED, 0.1, 0.09, "").check()           # no control named
    assert MemorizationResult(TestState.FAILED, 0.3, 0.0, "shuffle").gap() == pytest.approx(0.3)


def test_complexity_pressure_rises_with_forking_paths():
    lo = Complexity(5, 5, 1, 10000).overfit_pressure()
    hi = Complexity(5, 5, 150, 10000).overfit_pressure()
    assert hi > lo
    assert Complexity(50, 5, 1, 100).overfit_pressure() > lo


# ------------------------------------------------------------------ the four launch questions

def test_empty_memory_launches_novel():
    m = mem_new()
    a = m.launch_gate(record("e1"), design(), "2026-03-01")
    assert a.decision == LaunchDecision.LAUNCH_NOVEL and not a.have_we_done_this and a.decision.permits_launch


def test_invalid_proposals_are_refused_with_reasons():
    m = mem_new()
    no_counter = record("e1")
    hyps = tuple(dataclasses.replace(h, kind="explanation") for h in no_counter.competing_hypotheses)
    a = m.launch_gate(no_counter.with_(competing_hypotheses=hyps), design(), "2026-03-01")
    assert a.decision == LaunchDecision.INVALID_PROPOSAL and any("counter-hypothesis" in p for p in a.problems)
    a2 = m.launch_gate(record("e2"), design(cutoff="2026-03-02", period=("2015-01-01", "2020-12-31")), "2026-03-01")
    assert a2.decision == LaunchDecision.INVALID_PROPOSAL


def test_register_refuses_and_writes_nothing_when_gate_says_no():
    m = mem_new()
    bad = record("e1").with_(competing_hypotheses=record("e1").competing_hypotheses[:1])
    with pytest.raises(LaunchRefused):
        m.register(bad, design(), "2026-03-01")
    assert len(m.ledger) == 0 and len(m.store) == 0


def test_same_design_after_solid_answer_is_refused():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01")
    assert a.decision == LaunchDecision.REFUSE_SAME_DESIGN and a.did_it_answer and a.have_we_done_this
    with pytest.raises(LaunchRefused):
        m.register(record("e2", now="2026-04-01"), design(), "2026-04-01")
    assert len(m.ledger) == 1


def test_reworded_question_with_same_config_is_still_caught():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", q="Is there an unseen year gain from the volatility rank feature in movement forecast", now="2026-04-01"),
                      design(), "2026-04-01")
    assert a.have_we_done_this and not a.decision.permits_launch


def test_unrelated_question_is_not_blocked_by_history():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", q=Q_OTHER, cfg={"window": 9, "k": 4}, seed=2, now="2026-04-01"), design(algo="tree", feats=("cal",)), "2026-04-01")
    assert a.decision == LaunchDecision.LAUNCH_NOVEL


def test_different_design_needs_a_reason_and_lists_the_true_ones():
    m = mem_new()
    run(m, "e1")
    fresh = design(period=("2021-01-01", "2025-06-30"), cutoff="2025-06-30", universe="U2")
    a = m.launch_gate(record("e2", seed=2, now="2026-04-01"), fresh, "2026-04-01")
    assert a.decision == LaunchDecision.NEEDS_REASON
    assert ReplicationReason.FRESH_PERIOD in a.available_reasons and ReplicationReason.FRESH_STOCKS in a.available_reasons
    b = m.launch_gate(record("e2", seed=2, now="2026-04-01"), fresh, "2026-04-01", claims=[ReplicationReason.FRESH_PERIOD, ReplicationReason.FRESH_STOCKS])
    assert b.decision == LaunchDecision.LAUNCH_REPLICATION and set(b.verified_reasons) == {ReplicationReason.FRESH_PERIOD, ReplicationReason.FRESH_STOCKS}
    stamped, _ = m.register(record("e2", seed=2, now="2026-04-01"), fresh, "2026-04-01", claims=[ReplicationReason.FRESH_PERIOD])
    assert "FRESH_PERIOD" in stamped.repeat_reason


def test_false_replication_claim_is_rejected():
    m = mem_new()
    run(m, "e1")
    overlapping = design(period=("2018-01-01", "2022-12-31"), cutoff="2022-12-31")
    a = m.launch_gate(record("e2", seed=2, now="2026-04-01"), overlapping, "2026-04-01", claims=[ReplicationReason.FRESH_PERIOD])
    assert not a.decision.permits_launch
    assert any(r == ReplicationReason.FRESH_PERIOD and "overlap" in why for r, why in a.rejected_reasons)
    same_seed = m.launch_gate(record("e2", seed=1, now="2026-04-01"), design(hp={"depth": 6}), "2026-04-01", claims=[ReplicationReason.FRESH_SEED])
    assert any(r == ReplicationReason.FRESH_SEED for r, _ in same_seed.rejected_reasons)


def test_new_seed_cannot_overturn_a_refutation():
    m = mem_new()
    run(m, "e1", kind=ResultKind.REFUTED, ci=(-0.2, -0.05))
    a = m.launch_gate(record("e2", seed=9, now="2026-04-01"), design(), "2026-04-01", claims=[ReplicationReason.FRESH_SEED])
    assert not a.decision.permits_launch
    assert any("cannot overturn" in why for _, why in a.rejected_reasons)


def test_confirmed_claim_may_be_reseeded():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", seed=9, now="2026-04-01"), design(hp={"depth": 3, "extra": 1}), "2026-04-01", claims=[ReplicationReason.FRESH_SEED])
    assert a.decision == LaunchDecision.LAUNCH_REPLICATION


def test_weak_answer_does_not_count_as_answering():
    m = mem_new()
    run(m, "e1", out=outcome(transfer=TestState.NOT_TESTED, mem_state=TestState.NOT_TESTED))
    prior = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01")
    assert not prior.did_it_answer and prior.priors[0].grade == AnswerGrade.WEAK
    assert any("transfer never tested" in r for r in prior.priors[0].grade_reasons)
    b = m.launch_gate(record("e2", now="2026-04-01"), design(hp={"depth": 3, "cv": "blocked"}), "2026-04-01",
                      difference_statement="this run adds the transfer and memorisation checks the first run skipped")
    assert b.decision == LaunchDecision.LAUNCH_AFTER_EXPLANATION


def test_unanswered_prior_requires_an_explanation_of_the_difference():
    m = mem_new()
    m.register(record("e1"), design(), "2026-03-01")
    m.start("e1", "2026-03-01")
    m.ledger.fail("e1", "2026-03-02", "out of memory")
    bare = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01")
    assert bare.decision == LaunchDecision.NEEDS_EXPLANATION
    ok = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01", difference_statement="rerun with the memory leak in the loader fixed")
    assert ok.decision == LaunchDecision.LAUNCH_AFTER_EXPLANATION
    junk = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01", difference_statement="different")
    assert junk.decision == LaunchDecision.NEEDS_EXPLANATION


def test_experiment_in_flight_blocks_a_duplicate():
    m = mem_new()
    m.register(record("e1"), design(), "2026-03-01")
    a = m.launch_gate(record("e2", now="2026-03-02"), design(), "2026-03-02", claims=[ReplicationReason.FRESH_PERIOD])
    assert a.decision == LaunchDecision.WAIT_IN_FLIGHT


def test_code_change_justifies_a_repeat_of_the_same_design():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", code="c2", now="2026-04-01"), design(), "2026-04-01", claims=[ReplicationReason.CODE_CHANGED])
    assert a.decision == LaunchDecision.LAUNCH_REPLICATION
    b = m.launch_gate(record("e2", code="c1", now="2026-04-01"), design(), "2026-04-01", claims=[ReplicationReason.CODE_CHANGED])
    assert not b.decision.permits_launch                                            # the code hash did not change


def test_design_diff_seed_only_is_the_same_design():
    r1, r2 = record("a", seed=1), record("b", seed=2)
    d = ex.design_diff(r2, design(), r1, design())
    assert d.is_same_design and "seed" in d.changed
    e = ex.design_diff(r2, design(algo="ridge", feats=("other",)), r1, design())
    assert not e.is_same_design and e.method_changed()
    f = ex.design_diff(r2, design(), r1, None)
    assert f.unknown and "prior did not record" in " ".join(f.lines())


def test_render_launch_states_the_answer():
    m = mem_new()
    run(m, "e1")
    a = m.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01")
    txt = ex.render_launch(a, Q)
    assert "REFUSE_SAME_DESIGN" in txt and "prior e1" in txt


# ------------------------------------------------------------------ closing and the 24 fields

def test_close_enforces_the_outcome_rules():
    m = mem_new()
    m.register(record("e1"), design(), "2026-03-01")
    m.start("e1", "2026-03-01")
    with pytest.raises(ValueError, match="failure mode"):
        m.close("e1", result(ResultKind.REFUTED, ci=(-0.2, -0.1)), outcome(failure=None), "2026-03-06", ["x"], ["y"], "z")
    with pytest.raises(ValueError, match="confidence interval"):
        m.close("e1", result(ResultKind.CONFIRMED, ci=()), outcome(), "2026-03-06", ["x"], ["y"], "z")
    with pytest.raises(FirewallBreach):
        m.close("e1", result(ResultKind.CONFIRMED, day="2020-12-31"), outcome(), "2026-03-06", ["x"], ["y"], "z")
    assert m.ledger.get("e1", "2026-03-07").status == ExperimentStatus.RUNNING     # nothing was half-written
    assert m.store.outcome("e1", "2026-03-07") is None


def test_dossier_has_all_24_fields_and_marks_the_missing():
    assert len(ex.SECTION15_FIELDS) == 24
    m = mem_new()
    run(m, "e1")
    d = m.dossier("e1", "2026-03-07")
    assert set(d) == set(ex.SECTION15_FIELDS) and ex.MISSING not in d.values()
    assert d["counter_hypotheses"] == ["no effect beyond noise"] and d["transfer_result"] == "PASSED" and d["failure_mode"] == "none"
    m2 = mem_new()
    m2.register(record("e2"), design(), "2026-03-01")
    open_d = m2.dossier("e2", "2026-03-02")
    assert open_d["result"] == ex.MISSING and open_d["transfer_result"] == ex.MISSING and open_d["confidence_interval"] == ex.MISSING
    cov = m.field_coverage("2026-03-07")
    assert cov["complete"] == 1 and cov["n"] == 1


def test_dossier_is_time_aware():
    m = mem_new()
    run(m, "e1")
    with pytest.raises(KeyError):
        m.dossier("e1", "2026-03-01")                       # not yet recorded strictly before this instant
    still_running = m.dossier("e1", "2026-03-04")
    assert still_running["result"] == ex.MISSING           # result observed 03-05 is invisible on 03-04


def test_legacy_record_reads_as_missing_not_invented():
    m = mem_new()
    em.import_legacy(m.ledger, [{"experiment_id": "old1", "question": "did knob k help", "outcome": "reject", "reason": "no gain", "t": "2025-01-01T00:00:00",
                                 "config": {"k": 3}}])
    d = m.dossier("old1", "2026-01-01")
    assert d["algorithm"] == ex.MISSING and d["dataset"] == ex.MISSING and d["transfer_result"] == ex.MISSING
    assert m.field_coverage("2026-01-01")["complete"] == 0
    assert ex.memory_integrity(m, "2026-01-01")["ok"] and ex.memory_integrity(m, "2026-01-01")["legacy"] == 1


def test_persistence_round_trip(tmp_path):
    m = mem_new(tmp_path)
    run(m, "e1")
    before = m.dossier("e1", "2026-03-07")
    m2 = mem_new(tmp_path)
    assert m2.dossier("e1", "2026-03-07") == before
    assert m2.launch_gate(record("e2", now="2026-04-01"), design(), "2026-04-01").decision == LaunchDecision.REFUSE_SAME_DESIGN
    with open(tmp_path / "ext.jsonl", "a") as f:
        f.write("{not json\n")
    assert mem_new(tmp_path).store.unparseable == 1


# ------------------------------------------------------------------ section 32 replication

def confirmed_series(m, periods, universes=("U1", "U1"), seeds=(1, 2), kinds=(ResultKind.CONFIRMED, ResultKind.CONFIRMED), cis=((0.1, 0.3), (0.12, 0.32))):
    for i, (p, u, s, k, ci) in enumerate(zip(periods, universes, seeds, kinds, cis)):
        now = f"2026-0{3 + i}-01"
        close = f"2026-0{3 + i}-20"
        ext = design(period=p, cutoff=p[1], universe=u, hp={"depth": 3, "i": i})
        run(m, f"r{i}", now=now, close_now=close, kind=k, ci=ci, ext=ext, rec=record(f"r{i}", seed=s, cfg={"depth": 3, "i": i}, now=now), day=f"2026-0{3 + i}-15",
            claims=[ReplicationReason.FRESH_PERIOD] if i else ())


def test_single_confirmation_is_never_enough():
    m = mem_new()
    run(m, "e1")
    s = ex.replication_status(m, Q, "2026-05-01")
    assert s.gate == GateVerdict.NEEDS_MORE_EVIDENCE and "one lucky experiment" in s.message


def test_replication_needs_independent_axes():
    m = mem_new()
    confirmed_series(m, [("2010-01-01", "2014-12-31"), ("2015-01-01", "2020-12-31")], universes=("U1", "U2"))
    s = ex.replication_status(m, Q, "2026-09-01")
    assert s.gate == GateVerdict.PROMOTE and "fresh_period" in s.axes_covered and "fresh_stocks" in s.axes_covered


def test_replication_with_same_period_only_is_thin():
    m = mem_new()
    m2_period = ("2010-01-01", "2014-12-31")
    run(m, "a", now="2026-03-01", close_now="2026-03-20", ext=design(period=m2_period, cutoff="2014-12-31"), day="2026-03-15")
    run(m, "b", now="2026-04-01", close_now="2026-04-20", ext=design(period=m2_period, cutoff="2014-12-31", hp={"depth": 3, "z": 1}),
        rec=record("b", seed=5, cfg={"depth": 3, "z": 1}, now="2026-04-01"), day="2026-04-15", claims=[ReplicationReason.FRESH_SEED])
    s = ex.replication_status(m, Q, "2026-09-01")
    assert s.gate == GateVerdict.NEEDS_MORE_EVIDENCE and "fresh_period" in s.axes_missing


def test_discordant_replicate_quarantines():
    m = mem_new()
    confirmed_series(m, [("2010-01-01", "2014-12-31"), ("2015-01-01", "2020-12-31")], kinds=(ResultKind.CONFIRMED, ResultKind.REFUTED),
                     cis=((0.1, 0.3), (-0.3, -0.1)))
    s = ex.replication_status(m, Q, "2026-09-01")
    assert s.gate == GateVerdict.QUARANTINED and s.discordant == ("r1",)
    assert m.ledger.contradictory_answers("2026-09-01")
    nxt = ex.suggest_next(m, "2026-09-01")
    assert any(q.origin == "contradiction" for q in nxt)


def test_memorised_original_fails_replication():
    m = mem_new()
    run(m, "e1", out=outcome(mem_state=TestState.FAILED))
    s = ex.replication_status(m, Q, "2026-09-01")
    assert s.gate == GateVerdict.FAILED and "memorisation" in s.message


def test_replication_empty_and_refuted_only():
    m = mem_new()
    assert ex.replication_status(m, Q, "2026-09-01").gate == GateVerdict.UNKNOWN
    run(m, "e1", kind=ResultKind.REFUTED, ci=(-0.2, -0.1))
    assert ex.replication_status(m, Q, "2026-09-01").gate == GateVerdict.FAILED


# ------------------------------------------------------------------ multiplicity, accounting, audits

def test_multiplicity_deflation():
    assert ex.deflated_threshold(0.05, 1) == pytest.approx(0.05)
    assert ex.deflated_threshold(0.05, 20) < 0.0026
    assert ex.deflated_p(0.01, 50) > 0.39
    assert ex.expected_max_null_z(1) == pytest.approx(0.0, abs=1e-9)
    assert ex.expected_max_null_z(1000) > 3.0
    with pytest.raises(ValueError):
        ex.deflated_threshold(1.5, 3)


def test_family_trials_counts_refuted_variants():
    m = mem_new()
    run(m, "a", kind=ResultKind.REFUTED, ci=(-0.2, -0.1), ext=design(choices=12))
    run(m, "b", now="2026-04-01", close_now="2026-04-20", kind=ResultKind.REFUTED, ci=(-0.2, -0.1), ext=design(choices=8, algo="ridge", feats=("q",)),
        rec=record("b", cfg={"alpha": 1}, now="2026-04-01"), day="2026-04-15", claims=[ReplicationReason.INDEPENDENT_METHOD])
    assert ex.family_trials(m, Q, "2026-09-01") == 20
    assert ex.family_trials(m, Q_OTHER, "2026-09-01") == 0


def test_value_accounting_flags_wasted_compute():
    m = mem_new()
    run(m, "a", out=outcome(cost=10.0))
    run(m, "b", now="2026-04-01", close_now="2026-04-20", kind=ResultKind.REFUTED, ci=(-0.2, -0.1), ext=design(algo="ridge", feats=("q",)),
        out=outcome(failure=FailureCause.FALSE_PATTERN, cost=30.0, impact=(DecisionEffect.NONE,)), rec=record("b", cfg={"a": 1}, now="2026-04-01"),
        day="2026-04-15", claims=[ReplicationReason.INDEPENDENT_METHOD])
    acc = ex.value_accounting(m, "2026-09-01")
    assert acc["total_cost_minutes"] == pytest.approx(40.0) and acc["share_wasted"] == pytest.approx(0.75)
    assert ex.value_accounting(mem_new(), "2026-09-01")["n"] == 0


def test_audit_cutoffs_catches_a_planted_breach():
    m = mem_new()
    run(m, "ok")
    assert ex.audit_cutoffs(m, "2026-09-01") == []
    bad = design(cutoff="2026-02-27")
    rec = record("bad", q=Q_OTHER, cfg={"w": 1}, now="2026-02-28")
    m.register(rec, bad, "2026-02-28")
    m.store._rows[-1]["payload"]["information_cutoff"] = "2026-03-15"       # corrupt the sidecar after the fact
    probs = ex.audit_cutoffs(m, "2026-09-01")
    assert any(p["experiment_id"] == "bad" for p in probs)


def test_memory_integrity_flags_orphans():
    m = mem_new()
    run(m, "e1")
    assert ex.memory_integrity(m, "2026-09-01")["ok"]
    m.store.put_design("ghost", design(), "2026-03-02")
    rep = ex.memory_integrity(m, "2026-09-01")
    assert not rep["ok"] and any("ghost" in p for p in rep["problems"])


def test_duplicate_clusters_find_unjustified_repeats():
    m = mem_new()
    run(m, "a", out=outcome(cost=9.0))
    # bypass the gate to plant a repeat that already happened
    twin = record("b", now="2026-04-01")
    stamped = dataclasses.replace(twin, version=1, recorded_at="2026-04-01T00:00:00")
    m.ledger._rows.append(stamped)
    m.store._append("b", "design", design(), "2026-04-01")
    cl = ex.duplicate_clusters(m, "2026-09-01")
    assert len(cl) == 1 and cl[0]["unjustified"] == ["b"] and cl[0]["wasted_minutes"] == pytest.approx(5.0)
    assert ex.duplicate_clusters(mem_new(), "2026-09-01") == []


def test_matured_record_gate_refuses_the_present():
    m = mem_new()
    run(m, "e1")
    rec = ex.matured_record(m, "e1", "2026-03-07")
    with pytest.raises(FirewallBreach):
        rec.gate("2026-03-05")                                # result matured on 03-05: not strictly before
    assert rec.gate("2026-03-06")["result"] == "CONFIRMED"
    with pytest.raises(ValueError):
        ex.matured_record(m, "nope", "2026-03-07")


def test_suggest_next_uses_the_memory_itself():
    m = mem_new()
    run(m, "e1")
    qs = ex.suggest_next(m, "2026-09-01")
    origins = {q.origin for q in qs}
    assert "not_learned" in origins and "replication" in origins
    assert ex.suggest_next(mem_new(), "2026-09-01") == []


def test_evidence_summary_grades_matter():
    m = mem_new()
    run(m, "e1", out=outcome(transfer=TestState.NOT_TESTED))
    s = ex.evidence_summary(m, Q, "2026-09-01")
    assert s["stance"] == "UNSETTLED" and "CONFIRMED/WEAK" in s["by_outcome_and_grade"]
    assert ex.evidence_summary(m, Q_OTHER, "2026-09-01")["stance"] == "UNTESTED"


def test_step_on_empty_and_on_populated_memory():
    empty = ex.step(mem_new(), "2026-09-01")
    assert empty.n_experiments == 0 and empty.healthy and "healthy=True" in ex.render_step(empty)
    m = mem_new()
    run(m, "e1")
    rep = ex.step(m, "2026-09-01", current_code="c9")
    assert rep.stale and rep.needs_replication and rep.coverage["complete"] == 1
    assert "needs replication" in ex.render_step(rep)


# ================================================================== failed lab

def learner(name, mode, tags, hypo, regimes=("real A",), failed=None, gen=fl_mod.Generalization.SINGLE_REGIME, ts="2026-01-01T00:00:00",
            val=None, reason="the gain vanished on unseen years"):
    vr = val or {"transfer_ci_lo": -0.1, "transfer_ci_hi": 0.1, "same_year_ci_lo": 0.01, "pairs": 12}
    return fl_mod.FailedLearner(name, hypo, f"engine/x.py::{name}", mode, tuple(regimes), vr, reason, gen, ts, tuple(tags), "memory",
                                tuple(failed if failed is not None else regimes))


REG = ("alpha", "bravo", "charlie")
WORDS = ("clustering", "kernel smoothing", "hashing", "quantised buckets", "decayed averages", "rank transforms", "graph walk",
         "bayesian shrinkage", "wavelet features", "quantile maps", "time warping", "sparse coding", "gated recall", "prototype pruning")


def plant_class(lab, n_over=11, n_transfer=2, n_risk=1):
    """The contract's example: 14 attempts, 11 overfit, 2 did not transfer, 1 risk concentration; hypotheses and regimes vary."""
    i = 0
    for mode, k in ((fl_mod.FailureMode.OVERFIT, n_over), (fl_mod.FailureMode.NO_TRANSFER, n_transfer), (fl_mod.FailureMode.TRADEOFF_HARM, n_risk)):
        for _ in range(k):
            tags = ("stored_numeric_memory", f"variant_{i % 5}")
            val = {"transfer_ci_lo": -0.1, "transfer_ci_hi": 0.1, "same_year_ci_lo": 0.01 if mode == fl_mod.FailureMode.OVERFIT else -0.01,
                   "guard_worse": mode == fl_mod.FailureMode.TRADEOFF_HARM}
            lab.record_failure(learner(f"m{i}", mode, tags, f"{WORDS[i % len(WORDS)]} of stored memory for window outcomes, variant {i}",
                                       regimes=(f"real {REG[i % 3]}",), val=val, ts=f"2026-01-{1 + i:02d}T00:00:00",
                                       gen=fl_mod.Generalization.GENERALIZED, failed=(f"real {REG[i % 3]}", f"real {REG[(i + 1) % 3]}")), None)
            i += 1


def test_seeded_lab_audits_clean_and_has_written_autopsies():
    lab = FailedLearnerLab()
    out = lab_mod.seed_lab(lab)
    assert out["added"] == 13 and out["written_autopsies"] == 13
    a = lab_mod.audit_lab(lab, "2026-10-01")
    assert a["ok"] and a["n_autopsies"] == 13 and not a["derived_only"]
    assert lab_mod.seed_lab(lab)["added"] == 0                                   # idempotent
    mb = lab.autopsy("memory_bank", 1, "2026-10-01")
    assert mb.memorized == Finding.YES and mb.overfit == Finding.YES and not mb.derived
    assert lab.autopsy("band_cfg", 1, "2026-10-01").increased_risk == Finding.YES


def test_class_statement_reproduces_the_contract_sentence():
    lab = FailedLearnerLab()
    plant_class(lab)
    p = LabProposal("new", "recall stored memory for window outcomes with a fresh idea", ("stored_numeric_memory",), data_regime=("real alpha",))
    c = lab.consult(p, "2026-06-01")
    s = c.statement
    assert "attempted this class of learner 14 times" in s
    assert "11 failed because of overfitting" in s and "2 failed because the signal did not transfer" in s and "1 failed because of risk concentration" in s
    assert s.endswith("A new attempt requires a materially different hypothesis.")
    assert c.history.n == 14 and sum(c.history.by_bucket.values()) == 14
    assert c.verdict == lab_mod.ClassVerdict.DEAD_CLASS and c.decision == Decision.BLOCKED
    assert c.reopen_conditions                                                    # a dead class always says how it reopens


def test_empty_lab_and_unrelated_idea_proceed():
    lab = FailedLearnerLab()
    p = LabProposal("x", "quantum stuff", ("causal_discovery",))
    c = lab.consult(p, "2026-06-01")
    assert c.decision == Decision.PROCEED and c.statement.startswith("No learner") and c.history.n == 0
    lab_mod.seed_lab(lab)
    unrelated = LabProposal("y", "granger causality between sector returns", ("causal_discovery",), data_regime=("real",))
    assert lab.consult(unrelated, "2026-10-01", log=False).decision == Decision.PROCEED


def test_renamed_idea_is_recognised_by_class():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    sneaky = LabProposal("totally_new_name", "remember what worked in past windows and replay it", ("window_replay",), data_regime=("real",))
    assert lab_mod.classes_of(("window_replay",)) == {}                            # a brand-new tag has no declared class ...
    same = lab_mod.proposal_from_description("renamed", "a stored memory bank of past window numbers to recall what worked", regime=["real"])
    c = lab.consult(same, "2026-10-01", log=False)
    assert c.history.n >= 3 and not c.decision.permits
    assert sneaky.mechanism_tags == ("window_replay",)


def test_material_difference_needs_new_mechanism_and_controls():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    controls = ("memoriser_control", "past_only_transfer", "disguised_rerun", "noise_control", "walk_forward", "base_rate_baseline", "guard_metrics",
                "no_learning_control")
    p = LabProposal("gnn", "message passing over sector graph to carry information across stocks not windows", ("stored_numeric_memory", "graph_message_passing"),
                    information_sources=("sector graph",), controls_planned=controls, data_regime=("real 2027",))
    c = lab.consult(p, "2026-10-01", log=False)
    assert c.difference.material and c.decision == Decision.PROCEED_WITH_CONTROLS and "graph_message_passing" in c.difference.novel_tags
    bare = dataclasses.replace(p, controls_planned=())
    c2 = lab.consult(bare, "2026-10-01", log=False)
    assert not c2.difference.material and c2.difference.controls_missing and not c2.decision.permits
    restated = dataclasses.replace(p, hypothesis="Storing window-level numeric memories lets the system recall what worked and improve when the same market recurs",
                                   controls_planned=controls)
    assert lab.consult(restated, "2026-10-01", log=False).difference.hypothesis_restated


def test_consult_is_time_aware():
    lab = FailedLearnerLab()
    plant_class(lab)                                                             # recorded January 2026
    p = LabProposal("x", "recall stored memory for window outcomes", ("stored_numeric_memory",))
    assert lab.consult(p, "2025-06-01", log=False).decision == Decision.PROCEED   # before any failure existed
    assert lab.consult(p, "2026-06-01", log=False).history.n == 14
    assert lab.consult(p, "2026-01-05", log=False).history.n == 4                 # only failures recorded strictly before the 5th


def test_bad_proposal_and_require():
    lab = FailedLearnerLab()
    with pytest.raises(ValueError):
        lab.consult(LabProposal("", "", ()), "2026-06-01")
    with pytest.raises(ValueError):
        lab.consult(LabProposal("a", "b", ("t",), addresses=("not a bucket",)), "2026-06-01")
    plant_class(lab)
    with pytest.raises(lab_mod.MateriallySameAttempt) as e:
        lab.require(LabProposal("x", "recall stored memory for window outcomes", ("stored_numeric_memory",)), "2026-06-01")
    assert e.value.consultation.decision == Decision.BLOCKED


def test_autopsy_refuses_opinion_and_contradiction():
    lab = FailedLearnerLab()
    f = learner("bad", fl_mod.FailureMode.MEMORISATION, ("memory_bank",), "store numbers")
    site = lab_mod.FailureSite(lab_mod.FailureStage.HOLDOUT)
    no_basis = lab_mod.Autopsy("a", "b", "c", site, memorized=Finding.YES)
    with pytest.raises(ValueError, match="no basis"):
        lab.record_failure(f, no_basis)
    contradicts = lab_mod.Autopsy("a", "b", "c", site, memorized=Finding.NO, basis={"memorized": "measured"})
    with pytest.raises(ValueError, match="did not memorise"):
        lab.record_failure(f, contradicts)
    ok = lab_mod.Autopsy("a", "b", "c", site, memorized=Finding.YES, basis={"memorized": "gap CI above zero"})
    lab.record_failure(f, ok)
    assert len(lab.registry) == 1 and lab.autopsy("bad", 1, "2027-01-01").memorized == Finding.YES
    both = lab_mod.Autopsy("a", "b", "c", site, transferred=Finding.YES, memorized=Finding.YES, basis={"transferred": "x", "memorized": "y"})
    assert any("cannot both" in e for e in both.validate())


def test_unknown_stays_unknown_in_derived_autopsies():
    f = learner("plain", fl_mod.FailureMode.NO_SKILL, ("direction_classifier",), "predict direction", val={"accuracy": 0.518})
    a = lab_mod.derive_autopsy(f)
    assert a.derived and a.overfit == Finding.UNKNOWN and a.increased_risk == Finding.UNKNOWN and set(a.unknown_fields()) >= {"overfit", "increased_risk"}
    idle = learner("idle", fl_mod.FailureMode.NO_EFFECT, ("pool_rank_rules",), "pool", val={"adopted": 0})
    assert lab_mod.derive_autopsy(idle).overfit == Finding.NOT_APPLICABLE


def test_retest_counts_as_another_attempt():
    lab = FailedLearnerLab()
    lab.record_failure(learner("m0", fl_mod.FailureMode.OVERFIT, ("stored_numeric_memory",), "recall past windows numerically"), None)
    lab.record_failure(learner("m0", fl_mod.FailureMode.OVERFIT, ("stored_numeric_memory",), "recall past windows numerically"), None, retest=True)
    p = LabProposal("x", "recall past windows numerically", ("stored_numeric_memory",))
    assert lab.consult(p, "2027-01-01", log=False).history.n == 2 and len(lab.registry) == 1


def test_a_recorded_pass_reopens_the_class():
    lab = FailedLearnerLab()
    plant_class(lab)
    p = LabProposal("x", "recall stored memory for window outcomes", ("stored_numeric_memory",))
    assert lab.consult(p, "2026-06-01", log=False).verdict == lab_mod.ClassVerdict.DEAD_CLASS
    with pytest.raises(ValueError):
        lab.record_pass(lab_mod.PassRecord("winner", "h", ("stored_numeric_memory",), "real 2030", "2026-05-01T00:00:00"))
    lab.record_pass(lab_mod.PassRecord("winner", "h", ("stored_numeric_memory",), "real 2030", "2026-05-01T00:00:00", ("state/x/report.md",)))
    c = lab.consult(p, "2026-06-01", log=False)
    assert c.verdict == lab_mod.ClassVerdict.CONTESTED and "1 attempt(s) passed" in c.statement
    assert lab.consult(p, "2026-04-01", log=False).history.passes == ()           # the pass did not exist yet


def test_class_verdict_needs_enough_attempts_and_regimes():
    lab = FailedLearnerLab()
    for i in range(2):
        lab.record_failure(learner(f"a{i}", fl_mod.FailureMode.OVERFIT, ("stored_numeric_memory",), f"idea {i} about storing numbers"), None)
    p = LabProposal("x", "storing numbers idea", ("stored_numeric_memory",))
    assert lab.consult(p, "2027-01-01", log=False).verdict == lab_mod.ClassVerdict.UNDER_EXPLORED
    for i in range(2, 6):
        lab.record_failure(learner(f"a{i}", fl_mod.FailureMode.OVERFIT, ("stored_numeric_memory",), f"idea {i} about storing numbers differently"), None)
    v = lab.consult(p, "2027-01-01", log=False)
    assert v.verdict == lab_mod.ClassVerdict.CONTESTED and "regimes" in v.verdict_why     # six attempts, one regime: cannot call it dead


def test_consultation_log_measures_prevented_rediscovery(tmp_path):
    lab = FailedLearnerLab(path=tmp_path / "lab.jsonl")
    assert lab.rediscovery_prevented("2026-06-01")["share_redirected"] == 0.0
    plant_class(lab)
    lab.consult(LabProposal("x", "recall stored memory for window outcomes", ("stored_numeric_memory",)), "2026-06-01")
    lab.consult(LabProposal("y", "granger causality across sectors", ("causal_discovery",)), "2026-06-02")
    r = lab.rediscovery_prevented("2026-07-01")
    assert r["consultations"] == 2 and r["redirected"] == 1 and r["share_redirected"] == 0.5
    reload = FailedLearnerLab(FailedLearnerRegistry_from(lab), path=tmp_path / "lab.jsonl")
    assert reload.rediscovery_prevented("2026-07-01")["consultations"] == 2


def FailedLearnerRegistry_from(lab):
    reg = fl_mod.FailedLearnerRegistry()
    reg._rows = list(lab.registry._rows)
    return reg


def test_lab_health_reports_derivation_and_unknowns():
    lab = FailedLearnerLab()
    for i in range(3):
        lab.record_failure(learner(f"d{i}", fl_mod.FailureMode.NO_SKILL, ("direction_classifier",), f"direction idea {i}", val={"accuracy": 0.5}), None)
    h = lab_mod.lab_health(lab, "2027-01-01")
    assert h.derived_autopsies == 3 and any("derived from summary numbers" in n for n in h.notes)
    assert any("failures only" in n for n in h.notes) and h.unknown_findings["increased_risk"] == 3
    empty = lab_mod.lab_health(FailedLearnerLab(), "2027-01-01")
    assert empty.n_learners == 0 and empty.ok


def test_attempt_trend_detects_infinite_rediscovery():
    lab = FailedLearnerLab()
    for i in range(4):
        lab.record_failure(learner(f"r{i}", fl_mod.FailureMode.OVERFIT, ("stored_numeric_memory",), "store window numbers and recall what worked before",
                                   ts=f"2026-01-0{i + 1}T00:00:00"), None)
    p = LabProposal("x", "store window numbers and recall what worked", ("stored_numeric_memory",))
    t = lab_mod.attempt_trend(lab, p, "2026-06-01")
    assert t["rediscovery_rate"] == 1.0 and len(t["rediscoveries"]) == 3
    assert lab_mod.attempt_trend(FailedLearnerLab(), p, "2026-06-01")["attempts"] == 0


def test_plan_retry_lists_what_must_change():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    p = LabProposal("x", "recall stored memory of windows", ("stored_numeric_memory",), controls_planned=("noise_control",))
    plan = lab_mod.plan_retry(lab, p, "2026-10-01", universe_regimes=["synthetic planted world", "real archive windows 2001+ and pre-1997"],
                              candidate_tags=["graph_message_passing", "stored_numeric_memory"])
    assert "memoriser_control" in plan.add_controls and "noise_control" not in plan.add_controls
    assert plan.regimes_to_use == ("synthetic planted world",) and plan.untried_tags == ("graph_message_passing",)
    assert plan.hypothesis_guidance


def test_stage_profile_and_control_gaps():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    sp = lab_mod.stage_profile(lab, "2026-10-01")
    assert sp["n"] == 13 and sp["dominant_stage"] == "DESIGN" and sp["by_subsystem"]["SELECTION"] >= 10
    gaps = lab_mod.control_gaps(lab, "2026-10-01")
    assert list(gaps)[0] == "noise_control" or gaps[list(gaps)[0]] >= 2
    assert lab_mod.stage_profile(FailedLearnerLab(), "2026-10-01")["n"] == 0


def test_bucket_to_cause_and_reason_buckets():
    assert lab_mod.failure_cause_of(ReasonBucket.UNKNOWN) == FailureCause.UNKNOWN
    assert lab_mod.failure_cause_of(ReasonBucket.RISK_CONCENTRATION) == FailureCause.RISK_ERROR
    tradeoff = learner("t", fl_mod.FailureMode.TRADEOFF_HARM, ("band_targeting",), "band")
    assert lab_mod.reason_bucket(tradeoff) == ReasonBucket.RISK_CONCENTRATION
    assert lab_mod.reason_bucket(learner("q", fl_mod.FailureMode.LEAKAGE, ("x",), "h")) == ReasonBucket.LEAKAGE


def test_lab_matured_record_and_persistence(tmp_path):
    lab = FailedLearnerLab(path=tmp_path / "lab.jsonl")
    lab_mod.seed_lab(lab)
    rec = lab_mod.matured_record(lab, "memory_bank", "2026-10-01")
    with pytest.raises(FirewallBreach):
        rec.gate("2026-09-29")
    assert rec.gate("2026-09-30")["bucket"] == "overfitting"
    reg = FailedLearnerRegistry_from(lab)
    again = FailedLearnerLab(reg, path=tmp_path / "lab.jsonl")
    assert again.autopsy("memory_bank", 1, "2026-10-01").memorized == Finding.YES
    with pytest.raises(KeyError):
        lab_mod.matured_record(lab, "nope", "2026-10-01")


def test_lab_step_reports_dead_classes_and_reopen_paths():
    lab = FailedLearnerLab()
    plant_class(lab)
    st = lab_mod.step(lab, "2026-06-01")
    assert "memory_recall" in st.dead_classes
    assert any(r["class"] == "memory_recall" and r["conditions"] for r in st.reopenable)
    assert "memory_recall" in lab_mod.render_lab(lab, "2026-06-01") and "# Failed-learner lab" in lab_mod.to_markdown(lab, "2026-06-01")
    empty = lab_mod.step(FailedLearnerLab(), "2026-06-01")
    assert empty.dead_classes == () and empty.health.n_learners == 0


# ------------------------------------------------------------------ the two memories together

def test_lab_entry_from_a_refuted_experiment_and_joint_gate():
    m = mem_new()
    bad = outcome(failure=FailureCause.FALSE_PATTERN, mem_state=TestState.FAILED)
    run(m, "e1", kind=ResultKind.REFUTED, ci=(-0.2, -0.05), out=bad)
    lab = FailedLearnerLab()
    fl = lab_mod.lab_entry_from_experiment(m, lab, "e1", "vol_rank_gbm", ("stored_numeric_memory", "feature_rank"), "real panel A", "engine/x.py::G", "2026-09-01")
    assert fl.failure_mode == fl_mod.FailureMode.MEMORISATION
    aut = lab.autopsy("vol_rank_gbm", 1, "2026-09-02")
    assert aut.memorized == Finding.YES and aut.basis["memorized"]
    ok_exp = run(mem_new(), "z1")
    with pytest.raises(ValueError):
        lab_mod.lab_entry_from_experiment(m, lab, "nope", "n", ("t",), "r", "i", "2026-09-01")
    p = LabProposal("retry", "recall a stored feature ranking of windows", ("stored_numeric_memory", "feature_rank"), data_regime=("real panel A",))
    both = lab_mod.learner_launch_gate(m, lab, record("e2", now="2026-10-01"), design(), p, "2026-10-01")
    assert not both["permitted"] and both["reasons"]
    novel = lab_mod.learner_launch_gate(mem_new(), FailedLearnerLab(), record("e3"), design(), p, "2026-10-01")
    assert novel["permitted"] and ok_exp.status == ExperimentStatus.ANSWERED


# ------------------------------------------------------------------ batches, annotation, discovery

def test_screen_batch_stops_two_workers_launching_the_same_test():
    m = mem_new()
    a = (record("b1", now="2026-04-01"), design(), (), "")
    b = (record("b2", now="2026-04-01"), design(), (), "")
    c = (record("b3", q=Q_OTHER, cfg={"k": 1}, seed=3, now="2026-04-01"), design(algo="tree", feats=("cal",)), (), "")
    res = ex.screen_batch(m, [a, b, c], "2026-04-01")
    assert [r["launchable"] for r in res] == [True, False, True]
    assert res[1]["decision"] == LaunchDecision.WAIT_IN_FLIGHT and "b1" in res[1]["message"]
    assert ex.screen_batch(m, [], "2026-04-01") == []
    assert len(m.ledger) == 0                                    # read-only


def test_annotate_legacy_only_fills_legacy_rows():
    m = mem_new()
    em.import_legacy(m.ledger, [{"experiment_id": "old1", "question": "did knob k help", "outcome": "reject", "reason": "no gain", "t": "2025-01-01T00:00:00",
                                 "config": {"k": 3}}])
    assert m.dossier("old1", "2026-01-01")["algorithm"] == ex.MISSING
    ex.annotate_legacy(m, "old1", design(), "2026-02-01")
    assert m.dossier("old1", "2026-03-01")["algorithm"] == "gbm"
    assert m.dossier("old1", "2026-01-15")["algorithm"] == ex.MISSING        # the annotation does not rewrite the past
    with pytest.raises(ValueError):
        ex.annotate_legacy(m, "old1", design(), "2026-03-01")               # already written
    run(m, "new1", rec=record("new1", q=Q_OTHER, cfg={"zz": 1}, seed=7), ext=design(algo="tree", feats=("cal",)))
    with pytest.raises(ValueError, match="not a legacy"):
        ex.annotate_legacy(m, "new1", design(), "2026-09-01")


def test_markdown_report_lists_stance_and_completeness():
    m = mem_new()
    run(m, "e1")
    txt = ex.to_markdown(m, "2026-09-01")
    assert "SUPPORTED" in txt or "UNSETTLED" in txt
    assert "| e1 | ANSWERED | CONFIRMED | SOLID | 24/24 |" in txt
    assert "0 experiments" in ex.to_markdown(mem_new(), "2026-09-01")


def test_discover_classes_finds_groups_and_taxonomy_gaps():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    lab.record_failure(learner("odd", fl_mod.FailureMode.NO_SKILL, ("hologram_projection",), "project holograms of order flow onto price bands"), None)
    groups = lab_mod.discover_classes(lab, "2026-10-01")
    assert sum(g["attempts"] for g in groups) == 14
    assert any(g["unclassified"] and g["learners"] == ["odd"] for g in groups)
    memory_group = next(g for g in groups if "memory_bank" in g["learners"])
    assert "basis_learner" in memory_group["learners"] and "memory_recall" in memory_group["declared_classes"]
    assert lab_mod.discover_classes(FailedLearnerLab(), "2026-10-01") == []


def test_screen_proposals_flags_batch_twins_and_dead_ends():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    good = LabProposal("g", "granger causality between sector returns", ("causal_discovery",), data_regime=("real",))
    twin = LabProposal("g2", "granger causality between sector returns", ("causal_discovery",), data_regime=("real",))
    dead = lab_mod.proposal_from_description("d", "a stored memory bank of past window numbers to recall what worked", regime=["real"])
    res = lab_mod.screen_proposals(lab, [good, twin, dead], "2026-10-01")
    assert [r["permitted"] for r in res] == [True, False, False]
    assert res[1]["duplicate_of_batch_member"] == "g"
    assert lab.consultations("2026-10-02") == []                                  # screening does not log


def test_explain_reads_like_a_person_wrote_it():
    lab = FailedLearnerLab()
    lab_mod.seed_lab(lab)
    c = lab.consult(lab_mod.proposal_from_description("d", "a stored memory bank of past window numbers to recall what worked", regime=["real"]),
                    "2026-10-01", log=False)
    txt = lab_mod.explain(c)
    assert "already attempted this class" in txt and "Before any retry, include" in txt and "worth reopening" in txt
