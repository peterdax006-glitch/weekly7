"""Tests for engine.learning.experiment_memory / research_policy / research_priority / failed_learners (contract sections
31, 35, 36, 38, 51). Synthetic only. Every mechanism has a planted case it must catch plus the empty/degenerate case."""
import itertools
import json
import math

import numpy as np
import pytest

from engine.learning import experiment_memory as em
from engine.learning import failed_learners as fl
from engine.learning import research_policy as rp
from engine.learning import research_priority as rpri
from engine.learning.core import FailureCause, FirewallBreach


# ------------------------------------------------------------------------------------------------ helpers

def mk_record(eid="e1", q="Does the volume filter improve out of sample returns?", cfg=None, now="2026-01-01", seed=1,
              code="c1", data="d1", windows=("w1", "w2")):
    hyps = (em.Hypothesis("h_real", "the filter helps", 0.5), em.Hypothesis("h_null", "no effect", 0.5))
    exp = em.uniform_expected(hyps, ("better", "same"), {"h_real": "better", "h_null": "same"}, 0.8)
    return em.new_record(eid, q, "probably no effect", hyps, em.Prediction("no effect", "mean_week", "none", None, 0.6),
                         em.DesignSpec(config=cfg if cfg is not None else {"vol_q": 0.8, "k": 4}, windows=windows, seed=seed,
                                       controls=("shuffle",), cost_minutes=10, subsystem="SELECTION", code_hash=code, data_hash=data),
                         exp, now)


def answer(led, eid, outcome="same", kind=em.ResultKind.REFUTED, when="2026-01-05", now="2026-01-06"):
    res = em.ExperimentResult(kind=kind, outcome=outcome, metrics={"score": 0.1}, n=100, ci=(-0.1, 0.1),
                              observed_at=when, summary="s", power_note="power 0.8 at 0.05" if kind == em.ResultKind.NULL else "")
    return led.record_result(eid, res, now, ["the filter does not help"], ["did not test other years"], "stop tuning the filter")


# ------------------------------------------------------------------------------------------------ experiment memory

def test_record_validation_catches_planted_defects():
    good = mk_record()
    assert good.validate("proposal") == []
    one = good.with_(competing_hypotheses=good.competing_hypotheses[:1])
    assert any("fewer than two" in e for e in one.validate())
    bad_prior = good.with_(competing_hypotheses=(em.Hypothesis("h_real", "a", 0.9), em.Hypothesis("h_null", "b", 0.5)))
    assert any("priors sum" in e for e in bad_prior.validate())
    no_exp = good.with_(expected_outcomes=good.expected_outcomes[:2])
    assert any("no expected outcome" in e for e in no_exp.validate())
    assert any("WHAT WAS LEARNED empty" in e for e in good.validate("answered"))
    assert any("no seed" in e for e in good.with_(experiment=em.DesignSpec(config={"a": 1})).validate())


def test_ledger_round_trip_and_computed_belief_update(tmp_path):
    led = em.ExperimentLedger(tmp_path / "ledger.jsonl")
    led.propose(mk_record(), "2026-01-02")
    rec = answer(led, "e1", "same")
    bu = rec.belief_update
    assert abs(sum(bu.posterior.values()) - 1) < 1e-9
    assert bu.posterior["h_null"] > 0.75 and bu.moved > 0.2          # computed from expected outcomes, not typed in
    again = em.ExperimentLedger(tmp_path / "ledger.jsonl")
    assert again.get("e1", "2026-02-01").status == em.ExperimentStatus.ANSWERED
    assert len(again) == 1 and again.unparseable == 0
    with open(tmp_path / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write("{torn\n")
    assert em.ExperimentLedger(tmp_path / "ledger.jsonl").unparseable == 1     # counted, not dropped


def test_we_already_tested_this_planted_repeat_blocks():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    answer(led, "e1")
    with pytest.raises(em.DuplicateExperiment) as ex:
        led.propose(mk_record("e2", now="2026-02-01"), "2026-02-02")
    assert ex.value.verdict.status == em.DuplicateStatus.SAME_CONFIG_REPEAT
    assert "we already tested this" in str(ex.value)


def test_reworded_question_same_answer_still_blocks():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    answer(led, "e1")
    v = led.already_tested("Is there an out of sample improvement in returns from the volume filter?",
                           em.DesignSpec(config={"vol_q": 0.5, "k": 8}, seed=1), "2026-03-01")
    assert v.status == em.DuplicateStatus.SAME_QUESTION_ANSWERED and v.blocking


def test_different_question_different_config_is_novel():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    answer(led, "e1")
    v = led.already_tested("Which sector concentration limit reduces drawdown?", em.DesignSpec(config={"sector_cap": 0.3}, seed=1), "2026-03-01")
    assert v.status == em.DuplicateStatus.NOVEL and not v.blocking


def test_code_or_data_change_makes_a_repeat_a_justified_retest():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    answer(led, "e1")
    rec = mk_record("e2", now="2026-02-01", code="c2")
    stored = led.propose(rec, "2026-02-02")
    assert "code changed" in stored.repeat_reason


def test_in_flight_blocks_and_invalid_and_failed_do_not():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    v = led.already_tested(mk_record("e2").question, mk_record("e2").experiment, "2026-01-03")
    assert v.status == em.DuplicateStatus.IN_FLIGHT and v.blocking
    led.fail("e1", "2026-01-04", "crashed")
    v = led.already_tested(mk_record("e2").question, mk_record("e2").experiment, "2026-01-05")
    assert not v.blocking                                  # a crashed run answered nothing


def test_known_failure_nearby_blocks_a_tiny_variation():
    led = em.ExperimentLedger()
    led.propose(mk_record(cfg={"vol_q": 0.80, "k": 4}), "2026-01-02")
    answer(led, "e1")
    v = led.already_tested("Totally different wording about alpha decay", em.DesignSpec(config={"vol_q": 0.81, "k": 4}, seed=2), "2026-03-01")
    assert v.status == em.DuplicateStatus.KNOWN_FAILURE_NEARBY and v.blocking


def test_repeat_reason_unlocks_a_blocked_repeat_but_not_in_flight():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    with pytest.raises(em.DuplicateExperiment):
        led.propose(mk_record("e2", now="2026-01-03"), "2026-01-03", allow_repeat_reason="rerun")
    answer(led, "e1")
    led.propose(mk_record("e3", now="2026-02-01"), "2026-02-02", allow_repeat_reason="the filter code was rewritten")


def test_time_firewall_hides_the_future():
    led = em.ExperimentLedger()
    led.propose(mk_record(), "2026-01-02")
    answer(led, "e1", when="2026-01-05", now="2026-01-06")
    assert led.get("e1", "2026-01-05").status == em.ExperimentStatus.RUNNING or led.get("e1", "2026-01-05") is None or \
        led.get("e1", "2026-01-05").result is None
    assert led.get("e1", "2026-01-01") is None                       # not yet recorded
    assert led.get("e1", "2026-01-07").result is not None
    led2 = em.ExperimentLedger()
    led2.propose(mk_record(), "2026-01-02")
    res = em.ExperimentResult(em.ResultKind.REFUTED, "same", observed_at="2026-03-01")
    with pytest.raises(FirewallBreach):
        led2.record_result("e1", res, "2026-01-06", ["x"], ["y"], "z")


def test_empty_ledger_and_reports_and_calibration():
    led = em.ExperimentLedger()
    assert led.already_tested("anything", em.DesignSpec(config={"a": 1}, seed=1), "2026-01-01").status == em.DuplicateStatus.NOVEL
    assert led.answered("2026-01-01") == [] and led.contradictory_answers("2026-01-01") == []
    assert led.prediction_calibration("2026-01-01")["verdict"] == "insufficient"
    assert "experiments: 0" in em.render_report(led, "2026-01-01")
    assert em.coverage_report([])["n"] == 0


def test_not_learned_register_and_contradictory_answers():
    led = em.ExperimentLedger()
    for i, (kind, outc) in enumerate([(em.ResultKind.CONFIRMED, "better"), (em.ResultKind.REFUTED, "same")]):
        rec = mk_record(f"e{i}", cfg={"vol_q": 0.2 * (i + 1) + i, "k": 4 + 3 * i}, windows=(f"w{i}",), now=f"2026-01-0{i + 1}")
        led.propose(rec.with_(question="Does the volume filter improve out of sample returns?"), f"2026-0{1 + 2 * i}-02",
                    allow_repeat_reason="second window" if i else "")
        answer(led, f"e{i}", outc, kind, when=f"2026-0{2 + 2 * i}-01", now=f"2026-0{2 + 2 * i}-10")
    now = "2026-06-01"
    assert len(led.contradictory_answers(now)) == 1
    reg = led.not_learned_register(now)
    assert list(reg.values())[0]["count"] == 2                         # both left the same thing open
    sig = rpri.signals_from_ledger(led, now)
    kinds = {s.kind for s in sig}
    assert rpri.SignalKind.CONTRADICTION in kinds and rpri.SignalKind.UNKNOWN_AREA in kinds


def test_bayes_and_robust_update_misfit_guard():
    prior = {"a": 0.5, "b": 0.5, "u": 0.0001}
    post = em.bayes_update({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1})
    assert post["a"] == pytest.approx(0.9)
    p2, flag = em.robust_update({"a": 0.45, "b": 0.45, "u": 0.10}, {"a": 0.05, "b": 0.03, "u": 0.05}, "u")
    assert flag and p2["u"] > 0.3                                      # nothing explains it: mass moves to UNKNOWN
    p3, flag3 = em.robust_update({"a": 0.5, "b": 0.5, "u": 0.0}, {"a": 0.6, "b": 0.1, "u": 0.3}, "u")
    assert not flag3
    with pytest.raises(ValueError):
        em.normalise({"a": 0.0})


def test_legacy_import_is_honest_about_missing_fields(tmp_path):
    rows = [{"t": "2026-09-28T18:30:24", "event": "frontier", "k": 4, "exit_q": 0.9},
            {"experiment_id": "x1", "timestamp": "2026-01-01T00:00:00", "outcome": "reject", "reason": "no gain", "config": {"k": 4}, "seed": 3}]
    led = em.ExperimentLedger()
    out = em.import_legacy(led, rows)
    assert out["added"] == 2 and out["mean_completeness"] < 0.5
    assert em.import_legacy(led, rows)["added"] == 1 or em.import_legacy(led, rows)["skipped"] >= 1
    rec = led.get("x1", "2027-01-01")
    assert rec.legacy and "prediction" in rec.legacy_missing and rec.completeness() < 0.5
    assert em.NOT_RECORDED in rec.current_belief


def test_facade_over_tried_index_and_registry_memory(tmp_path):
    from engine.experiment_memory import Space, TriedIndex
    from engine.registry import ExperimentMemory
    space = Space({"vol_q": (0.0, 1.0, 0.05), "k": (1, 20, 1)})
    tried = TriedIndex(tmp_path / "tried.jsonl", space)
    tried.add("old1", {"vol_q": 0.8, "k": 4}, "reject", 0.1, "no gain", now="2025-01-01")
    reg_mem = ExperimentMemory(tmp_path / "mem.jsonl")
    led = em.ExperimentLedger(bridge=em.LegacyBridge(tried, reg_mem))
    # ledger is EMPTY but the older store already knows this configuration: the façade must still block it
    with pytest.raises(em.DuplicateExperiment):
        led.propose(mk_record(cfg={"vol_q": 0.8, "k": 4}), "2026-01-02")
    led.propose(mk_record("e9", cfg={"vol_q": 0.3, "k": 12}), "2026-01-02")
    answer(led, "e9", "better", em.ResultKind.CONFIRMED)
    assert any(r["experiment_id"] == "e9" for r in tried.rows)         # mirrored into TriedIndex
    assert any(e["experiment_id"] == "e9" for e in reg_mem.entries)    # and into registry.ExperimentMemory
    assert reg_mem.already_tried({"vol_q": 0.3, "k": 12})["tried"] == 1


def test_question_similarity_orders_sensibly():
    a = "Does the volume filter improve out of sample returns?"
    assert em.question_similarity(a, "Is there an out of sample improvement in returns from the volume filter?") > 0.72
    assert em.question_similarity(a, "Which stops reduce drawdown in bear markets?") < 0.3
    assert em.question_similarity("", a) == 0.0


# ------------------------------------------------------------------------------------------------ information theory

def test_mutual_information_planted_cases():
    prior = {"a": 0.5, "b": 0.5}
    sharp = {"a": {"x": 1.0}, "b": {"y": 1.0}}
    flat = {"a": {"x": 0.5, "y": 0.5}, "b": {"x": 0.5, "y": 0.5}}
    assert rp.mutual_information(prior, sharp) == pytest.approx(1.0)
    assert rp.mutual_information(prior, flat) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        rp.mutual_information(prior, {"a": {"x": 0.9}, "b": {"y": 1.0}})
    assert rp.mutual_information({"a": 1.0}, {"a": {"x": 1.0}}) == 0.0


def test_beta_and_normal_eig_are_monotone_and_nonnegative():
    vals = [rp.eig_beta_binomial(1, 1, m) for m in (0, 1, 5, 20, 80)]
    assert vals[0] == 0.0 and all(b >= a for a, b in zip(vals, vals[1:])) and vals[-1] > 1.0
    assert rp.eig_beta_binomial(30, 30, 10) < rp.eig_beta_binomial(1, 1, 10)         # a settled belief learns less
    assert rp.eig_normal_mean(10, 10) == pytest.approx(0.5)
    assert rp.eig_normal_mean(10, 0) == 0.0
    assert rp.entropy_bits([]) == 0.0
    with pytest.raises(ValueError):
        rp.entropy_bits([0.5, 0.2])


def test_evsi_is_zero_when_decision_is_settled_and_grows_with_data():
    settled = rp.evsi_normal(1.0, 0.01, 0.1, 100)
    open_q = rp.evsi_normal(0.0, 0.05, 0.1, 100)
    assert settled < 1e-9 < open_q
    assert rp.evsi_normal(0.0, 0.05, 0.1, 400) > open_q
    assert rp.evsi_normal(0.0, 0.0, 0.1, 100) == 0.0


# ------------------------------------------------------------------------------------------------ priority function

def base_cand(**kw):
    d = dict(cid="c1", question="q", target=rp.ResearchTarget.FAILURE, created_at="2026-01-01", info=rp.InfoModel("normal", {"n_now": 10, "n_new": 30}))
    d.update(kw)
    return rp.Candidate(**d)


def test_informative_experiment_outranks_uninformative_one():
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-02-01")
    mk = lambda cid, p: base_cand(cid=cid, info=rp.InfoModel("discrete", {"hypotheses": [("a", .5), ("b", .5)],
                                  "expected": [("a", "x", p), ("a", "y", 1 - p), ("b", "x", 1 - p), ("b", "y", p)]}))
    assert pol.priority.score(mk("sharp", 0.95), ctx).priority > pol.priority.score(mk("flat", 0.5), ctx).priority


def test_penalties_block_or_reduce_as_planted():
    pol = rp.ResearchPolicy()
    dup = rp.PolicyContext(now="2026-02-01", duplicate_check=lambda c: em.DuplicateVerdict(em.DuplicateStatus.SAME_CONFIG_REPEAT, "we already tested this"))
    s = pol.priority.score(base_cand(), dup)
    assert s.priority == 0 and "already tested" in s.blocked
    s = pol.priority.score(base_cand(data_needs=("delisted",)), rp.PolicyContext(now="2026-02-01", available_data=frozenset({"panel"})))
    assert s.blocked.startswith("data not available")
    ctx = rp.PolicyContext(now="2026-02-01")
    base = pol.priority.score(base_cand(), ctx).priority
    assert pol.priority.score(base_cand(overfit_hint=0.9), ctx).priority < base * 0.6
    assert pol.priority.score(base_cand(prior_tests_on_window=10, free_parameters=10), ctx).priority < base * 0.5
    big = pol.priority.score(base_cand(cost=rp.ComputeCost(cpu_minutes=90)), ctx, rp.ComputeBudget(cpu_minutes=100, ram_gb_free=8)).priority
    small = pol.priority.score(base_cand(cost=rp.ComputeCost(cpu_minutes=5)), ctx, rp.ComputeBudget(cpu_minutes=100, ram_gb_free=8)).priority
    assert big < small
    over = pol.priority.score(base_cand(cost=rp.ComputeCost(cpu_minutes=500)), ctx, rp.ComputeBudget(cpu_minutes=100, ram_gb_free=8))
    assert over.blocked and "cpu-min" in over.blocked
    ram = pol.priority.score(base_cand(cost=rp.ComputeCost(ram_gb=6)), ctx, rp.ComputeBudget(cpu_minutes=100, ram_gb_free=4))
    assert ram.blocked and "GB" in ram.blocked


def test_known_low_value_family_is_penalised_only_with_evidence():
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-06-01")
    c = base_cand(family="tuning:grid")
    fresh = pol.priority.score(c, ctx).priority
    for i in range(6):
        pol.observe(rp.RealisedGain(f"g{i}", rp.ResearchTarget.KNOWN_RELIABLE, "tuning:grid", 0.3, 0.0, False, 10, f"2026-03-0{i + 1}"))
    after = pol.priority.score(c, rp.PolicyContext(now="2026-06-01", history=tuple(pol.history))).priority
    assert after < fresh * 0.6
    other = pol.priority.score(base_cand(family="new:family"), rp.PolicyContext(now="2026-06-01", history=tuple(pol.history))).priority
    assert other > after


def test_priority_is_deterministic_and_rejects_the_future_and_bad_inputs():
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-02-01")
    assert pol.priority.score(base_cand(), ctx).priority == pol.priority.score(base_cand(), ctx).priority
    with pytest.raises(FirewallBreach):
        pol.priority.score(base_cand(created_at="2026-02-01"), ctx)
    with pytest.raises(ValueError):
        pol.priority.score(base_cand(factors=rp.Factors(uncertainty=1.5)), ctx)
    with pytest.raises(ValueError):
        rp.PriorityFunction(rp.PolicyConfig(exploit_cap=0.0))


def test_eig_calibrator_learns_overclaiming():
    cal = rp.EigCalibrator(prior_strength=2.0)
    for i in range(20):
        cal.observe(rp.RealisedGain(f"g{i}", rp.ResearchTarget.FAILURE, "f", 1.0, 0.1, False, 5, "2026-01-01"))
    assert cal.scale("FAILURE") < 0.3 and cal.report()["FAILURE"]["verdict"] == "overclaims"
    assert cal.scale("UNKNOWN_AREA") == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ allocation (section 35)

ALL_OPEN = {t.value: 1.0 for t in rp.TARGETS}


def test_allocation_respects_floors_cap_and_sums_to_one():
    a = rp.TargetAllocator().allocate(600.0, ALL_OPEN, seed=3)
    assert abs(sum(a.shares.values()) - 1) < 1e-9 and a.check() == []
    assert all(s >= rp.PolicyConfig().explore_floor - 1e-9 for s in a.shares.values())
    assert a.shares["KNOWN_RELIABLE"] <= rp.PolicyConfig().exploit_cap + 1e-9
    assert sum(a.minutes.values()) == pytest.approx(600.0)


def test_allocation_is_seed_deterministic_and_seed_sensitive():
    a1 = rp.TargetAllocator().allocate(100, ALL_OPEN, seed=5).shares
    a2 = rp.TargetAllocator().allocate(100, ALL_OPEN, seed=5).shares
    a3 = rp.TargetAllocator().allocate(100, ALL_OPEN, seed=6).shares
    assert a1 == a2 and a1 != a3


def test_closed_targets_get_nothing_but_exploration_targets_stay_open():
    a = rp.TargetAllocator().allocate(100, {"FAILURE": 2.0}, seed=1)
    assert a.shares["CONTRADICTION"] == 0 and a.shares["FAILURE"] > 0
    assert a.shares["UNKNOWN_AREA"] >= rp.PolicyConfig().explore_floor - 1e-9      # always open
    only = rp.TargetAllocator().allocate(100, {"KNOWN_RELIABLE": 1.0}, seed=1)
    assert sum(only.shares.values()) == pytest.approx(1.0)


def test_bandit_concentrates_on_the_planted_best_target_and_beats_exploit_only():
    y = {"KNOWN_RELIABLE": 0.03, "FAILURE": 0.6, "UNKNOWN_AREA": 0.15}
    bandit = rp.simulate_allocation(y, 40, 100, seed=4, policy="bandit")
    exploit = rp.simulate_allocation(y, 40, 100, seed=4, policy="exploit_only")
    uniform = rp.simulate_allocation(y, 40, 100, seed=4, policy="uniform")
    assert bandit["total_bits"] > uniform["total_bits"] > exploit["total_bits"]
    assert bandit["experiments_by_target"]["FAILURE"] > bandit["experiments_by_target"]["KNOWN_RELIABLE"] * 3
    assert bandit["experiments_by_target"]["UNKNOWN_AREA"] > 0                     # still exploring


def test_exploit_only_spend_is_flagged_by_the_audit():
    hist = [rp.RealisedGain(f"g{i}", rp.ResearchTarget.KNOWN_RELIABLE, "f", .1, .0, False, 10, f"2026-01-{i + 1:02d}") for i in range(20)]
    assert rp.audit_spend(hist)["violation"]
    mixed = [rp.RealisedGain(f"g{i}", rp.TARGETS[i % 10], "f", .1, .1, True, 10, f"2026-01-{i + 1:02d}") for i in range(20)]
    assert not rp.audit_spend(mixed)["violation"]
    assert rp.audit_spend([])["n"] == 0
    assert rp.epsilon_schedule(10_000) == pytest.approx(0.10)                     # never stops exploring


def test_discounted_allocator_forgets_and_validates():
    d = rp.DiscountedAllocator(gamma=0.5)
    for _ in range(10):
        d.observe("FAILURE", True, 1.0, 5)
    hi = d.f_useful[rp.ResearchTarget.FAILURE]
    for _ in range(20):
        d.age()
    assert d.f_useful[rp.ResearchTarget.FAILURE] < hi * 1e-3
    with pytest.raises(ValueError):
        rp.DiscountedAllocator(gamma=0.0)


def test_box_projection_and_infeasible_bounds():
    s = rp.project_box_simplex({"a": 10, "b": 1, "c": 1}, {"a": 0, "b": 0.1, "c": 0.1}, {"a": 0.5, "b": 1, "c": 1})
    assert s["a"] == pytest.approx(0.5) and s["b"] == pytest.approx(0.25) and sum(s.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rp.project_box_simplex({"a": 1, "b": 1}, {"a": 0.6, "b": 0.6}, {"a": 1, "b": 1})


# ------------------------------------------------------------------------------------------------ selection & compute

def test_knapsack_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        items = [(float(rng.uniform(1, 10)), int(rng.integers(1, 9))) for _ in range(7)]
        cap = int(rng.integers(5, 25))
        best, chosen = rp.knapsack(items, cap)
        brute = max(sum(items[i][0] for i in sub) for r in range(8) for sub in itertools.combinations(range(7), r)
                    if sum(items[i][1] for i in sub) <= cap)
        assert best == pytest.approx(brute) and sum(items[i][1] for i in chosen) <= cap


def make_cands(n=8, real=0):
    return [base_cand(cid=f"c{i}", cost=rp.ComputeCost(cpu_minutes=10 + 4 * i, ram_gb=1.0, real_data=i < real),
                      target=rp.TARGETS[i % 10], info=rp.InfoModel("normal", {"n_now": 10, "n_new": 10 + 5 * i})) for i in range(n)]


def test_plan_respects_budget_ram_and_real_data_slots():
    pol = rp.ResearchPolicy()
    budget = rp.ComputeBudget(cpu_minutes=60, ram_gb_free=8, real_data_slots=1)
    plan = pol.plan(make_cands(10, real=4), rp.PolicyContext(now="2026-02-01"), budget, seed=2)
    assert rp.audit_plan(plan, pol.cfg, budget) == []
    assert plan.selection.used_minutes <= 60 + 1e-9
    assert sum(1 for s in plan.selection.selected if s.candidate.cost.real_data) <= 1
    assert "Research plan" in plan.explain() and plan_ok_rows(plan)


def plan_ok_rows(plan):
    rows = rp.plan_rows(plan)
    return len(rows) == len(plan.scored) and all("priority" in r for r in rows)


def test_plan_empty_and_duplicate_ids_and_future_history():
    pol = rp.ResearchPolicy()
    budget = rp.ComputeBudget(cpu_minutes=60, ram_gb_free=8)
    empty = pol.plan([], rp.PolicyContext(now="2026-02-01"), budget, seed=1)
    assert empty.selection.selected == () and abs(sum(empty.allocation.shares.values()) - 1) < 1e-9
    with pytest.raises(ValueError):
        pol.plan([base_cand(), base_cand()], rp.PolicyContext(now="2026-02-01"), budget, seed=1)
    pol.observe(rp.RealisedGain("g", rp.ResearchTarget.FAILURE, "f", .1, .1, True, 5, "2026-03-01"))
    with pytest.raises(FirewallBreach):
        pol.plan([base_cand()], rp.PolicyContext(now="2026-02-01"), budget, seed=1)
    with pytest.raises(FirewallBreach):
        rp.ResearchPolicy().observe(rp.RealisedGain("g", rp.ResearchTarget.FAILURE, "f", .1, .1, True, 5, "2026-03-01"), now="2026-02-01")


def test_blocked_candidates_are_deferred_with_reasons_never_selected():
    pol = rp.ResearchPolicy()
    budget = rp.ComputeBudget(cpu_minutes=60, ram_gb_free=8)
    cands = make_cands(3) + [base_cand(cid="needs", data_needs=("missing",))]
    plan = pol.plan(cands, rp.PolicyContext(now="2026-02-01", available_data=frozenset()), budget, seed=1)
    assert "needs" not in plan.selection.ids()
    assert any(s.candidate.cid == "needs" and "data not available" in why for s, why in plan.selection.deferred)


def test_family_caps_dependencies_and_frontier():
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-02-01")
    scored = [pol.priority.score(base_cand(cid=f"f{i}", family="same", cost=rp.ComputeCost(cpu_minutes=10)), ctx) for i in range(4)]
    scored += [pol.priority.score(base_cand(cid="other", family="other", cost=rp.ComputeCost(cpu_minutes=10)), ctx)]
    keep, deferred = rp.apply_family_caps(scored, max_share=0.5)
    assert len([s for s in keep if s.candidate.family == "same"]) < 4 and deferred
    order, dfr = rp.schedule_with_dependencies(scored[:3], {"f1": ["f0"], "f2": ["f1"]})
    ids = [s.candidate.cid for s in order]
    assert ids.index("f0") < ids.index("f1") < ids.index("f2")
    order2, dfr2 = rp.schedule_with_dependencies(scored[:2], {"f1": ["ghost"]})
    assert [s.candidate.cid for s in order2] == ["f0"] and "prerequisite" in dfr2[0][1]
    with pytest.raises(ValueError):
        rp.schedule_with_dependencies(scored[:2], {"f0": ["f1"], "f1": ["f0"]})
    front = rp.pareto_frontier(scored)
    assert front and all(a.candidate.cost.cpu_minutes <= b.candidate.cost.cpu_minutes or a.priority > b.priority for a, b in zip(front, front[1:]))


def test_joint_information_diversifies_redundant_experiments():
    prior = {"a": 0.5, "b": 0.5}
    lk = {"a": {"x": 0.8, "y": 0.2}, "b": {"x": 0.2, "y": 0.8}}
    one = rp.joint_information(prior, [lk])
    two = rp.joint_information(prior, [lk, lk])
    assert one < two < 2 * one                                                     # second copy adds less than the first
    ind = {"a": {"p": 0.8, "q": 0.2}, "b": {"p": 0.2, "q": 0.8}}
    assert rp.joint_information(prior, [lk, ind]) == pytest.approx(two)
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-02-01")
    mk = lambda cid: base_cand(cid=cid, info=rp.InfoModel("discrete", {"hypotheses": [("a", .5), ("b", .5)],
                               "expected": [("a", "x", .8), ("a", "y", .2), ("b", "x", .2), ("b", "y", .8)]}))
    other = base_cand(cid="third", info=rp.InfoModel("discrete", {"hypotheses": [("c", .5), ("d", .5)],
                      "expected": [("c", "x", .8), ("c", "y", .2), ("d", "x", .2), ("d", "y", .8)]}))
    sc = [pol.priority.score(c, ctx) for c in (mk("r1"), mk("r2"), other)]
    picks = rp.greedy_diverse_batch(sc, 2)
    assert {p[0].candidate.cid for p in picks} & {"third"}                         # the independent one beats the redundant copy


# ------------------------------------------------------------------------------------------------ misc policy mechanisms

def test_sprt_stops_early_and_respects_bounds():
    hi = rp.sprt_bernoulli([1] * 30, 0.3, 0.7)
    assert hi.decision == "ACCEPT_H1" and hi.n < 30
    lo = rp.sprt_bernoulli([0] * 30, 0.3, 0.7)
    assert lo.decision == "ACCEPT_H0"
    assert rp.sprt_bernoulli([1, 0, 1, 0], 0.3, 0.7).decision == "CONTINUE"
    with pytest.raises(ValueError):
        rp.sprt_bernoulli([1], 0.7, 0.3)
    assert rp.expected_sample_saving(0.3, 0.7, 40)["saving_h1"] > 0.3


def test_cost_model_learns_overruns():
    cm = rp.CostModel(prior_strength=1.0)
    for _ in range(10):
        cm.observe("t", 10, 30)
    assert cm.multiplier("t") > 2.0 and cm.multiplier("unseen") == pytest.approx(1.0)
    assert cm.adjusted("t", rp.ComputeCost(cpu_minutes=10)).cpu_minutes > 20
    with pytest.raises(ValueError):
        cm.observe("t", 0, 5)


def test_power_required_n_and_design_choice():
    n = rp.required_n(0.02, 0.1)
    assert rp.power_of(n, 0.02, 0.1) >= 0.79 and rp.power_of(n // 2, 0.02, 0.1) < 0.6
    opts = [rp.DesignOption("small", 20, rp.ComputeCost(cpu_minutes=5), 0.1), rp.DesignOption("big", 400, rp.ComputeCost(cpu_minutes=200), 0.1),
            rp.DesignOption("mid", 300, rp.ComputeCost(cpu_minutes=40), 0.1)]
    choice = rp.best_design(opts, 0.05, rp.ComputeBudget(cpu_minutes=100, ram_gb_free=8), effect_of_interest=0.02, min_power=0.5)
    assert choice["chosen"] == "mid"                                               # small underpowered, big does not fit
    assert rp.best_design(opts[:1], 0.05, effect_of_interest=0.02)["chosen"] is None
    with pytest.raises(ValueError):
        rp.required_n(0.0, 0.1)


def test_cusum_detects_planted_shift_and_is_quiet_otherwise():
    rng = np.random.default_rng(0)
    quiet = rng.normal(1, 0.1, 30)
    assert rp.cusum_shift(quiet)["verdict"] == "STABLE"
    shifted = np.concatenate([rng.normal(1, 0.1, 15), rng.normal(2, 0.1, 15)])
    r = rp.cusum_shift(shifted)
    assert r["verdict"] == "SHIFT" and r["direction"] == "up" and 15 <= r["alarm"] <= 20
    assert rp.cusum_shift([1, 2])["verdict"] == "INSUFFICIENT"


def test_ips_off_policy_estimate_and_zero_propensity_refusal():
    rng = np.random.default_rng(1)
    log = []
    for _ in range(400):
        t = rp.ResearchTarget.FAILURE if rng.random() < 0.5 else rp.ResearchTarget.KNOWN_RELIABLE
        log.append(rp.LoggedChoice(t, 0.5, 1.0 if t == rp.ResearchTarget.FAILURE else 0.0))
    good = {"FAILURE": 0.9, "KNOWN_RELIABLE": 0.1}
    bad = {"FAILURE": 0.1, "KNOWN_RELIABLE": 0.9}
    assert rp.ips_estimate(log, good)["snips"] > rp.ips_estimate(log, bad)["snips"]
    assert rp.compare_allocations(log, good, bad, seed=0)["verdict"] == "A_BETTER"
    with pytest.raises(ValueError):
        rp.ips_estimate([rp.LoggedChoice(rp.ResearchTarget.FAILURE, 0.0, 1.0)], good)
    assert rp.ips_estimate([], good)["n"] == 0


def test_rank_stability_and_elasticity_run_and_flag_fragile_rankings():
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-02-01")
    a = base_cand(cid="a", factors=rp.Factors(0.9, 0.9, 0.9, 1.0, 0.9))
    b = base_cand(cid="b", factors=rp.Factors(0.2, 0.2, 0.2, 1.0, 0.2))
    r = rp.rank_stability(pol, [a, b], ctx, seed=1, n_draws=50)
    assert r["stable"] and r["top_k_prob"]["a"] > 0.9
    near = rp.rank_stability(pol, [a, base_cand(cid="c", factors=rp.Factors(0.88, 0.9, 0.9, 1.0, 0.9))], ctx, seed=1, n_draws=80, noise=0.5)
    assert near["tau_mean"] < r["tau_mean"]
    el = rp.factor_elasticity(pol, base_cand(factors=rp.Factors(0.5, 0.5, 0.5, 0.5, 0.5)), ctx)
    assert all(v == pytest.approx(1.0, abs=0.05) for v in el.values())              # a pure product: elasticity 1
    assert rp.rank_stability(pol, [], ctx, seed=1)["n"] == 0


def test_horizon_plan_sums_and_validates():
    h = rp.plan_horizon(1000, 5, front_load=0.5)
    assert sum(m for m, _ in h.rounds) == pytest.approx(1000) and h.rounds[0][0] > h.rounds[-1][0]
    with pytest.raises(ValueError):
        rp.plan_horizon(100, 0)


def test_seed_from_log_uses_only_the_past_and_infers_targets():
    rows = [{"t": "2026-01-01T00:00:00", "event": "tuning grid", "outcome": "reject"}, {"t": "2026-01-02T00:00:00", "event": "regime era test", "outcome": "adopt"},
            {"t": "2026-09-01T00:00:00", "event": "data audit", "outcome": "adopt"}, {"t": "2026-01-03T00:00:00", "event": "x", "outcome": "continue_testing"}]
    pol = rp.ResearchPolicy()
    counts = rp.seed_from_log(pol, rows, "2026-06-01")
    assert counts == {"KNOWN_RELIABLE": 1, "REGIME_TRANSITION": 1}                    # the September row is the future
    assert rp.infer_target({"event": "zzz"}) == rp.ResearchTarget.UNKNOWN_AREA


def test_meta_advice_moves_the_overfit_penalty_and_factors():
    pol = rp.ResearchPolicy()
    c = base_cand(family="calendar_like")
    plain = pol.priority.score(c, rp.PolicyContext(now="2026-02-01")).priority
    advice = rp.MetaAdvice(family_overfit={"calendar_like": 0.95}, family_survival={"calendar_like": 0.1}, n_observations=500)
    assert advice.trust() > 0.4 and advice.check() == []
    with_meta = pol.priority.score(c, rp.PolicyContext(now="2026-02-01", meta=advice)).priority
    assert with_meta < plain * 0.75                                    # unvalidated advice is trusted at half weight
    validated = rp.MetaAdvice(family_overfit={"calendar_like": 0.95}, n_observations=500, oos_label="VALIDATED")
    assert pol.priority.score(c, rp.PolicyContext(now="2026-02-01", meta=validated)).priority < with_meta * 0.7
    adj = rp.meta_adjusted_factors(c, advice)
    assert adj.factors.transfer_potential < c.factors.transfer_potential
    assert rp.meta_adjusted_factors(c, rp.MetaAdvice.empty()) == c
    assert rp.MetaAdvice(family_overfit={"x": 1.5}).check()


def test_policy_self_check_and_persistence_round_trip(tmp_path):
    assert rp.self_check(3)["all_passed"]
    pol = rp.ResearchPolicy()
    pol.observe(rp.RealisedGain("g1", rp.ResearchTarget.FAILURE, "f", .3, .2, True, 10, "2026-01-01"))
    pol.save(tmp_path / "h.jsonl")
    back = rp.ResearchPolicy.load(tmp_path / "h.jsonl")
    assert len(back.history) == 1 and back.allocator.stats[rp.ResearchTarget.FAILURE].useful == 1
    assert "calibration" in rp.report(back)
    cfg = rp.config_from_dict(rp.config_to_dict(rp.PolicyConfig()))
    assert cfg == rp.PolicyConfig()
    with pytest.raises(ValueError):
        rp.config_from_dict({"nonsense": 1})
    c = base_cand(family="x", config={"a": (1, 2)})
    assert rp.candidate_from_dict(json.loads(json.dumps(rp.candidate_to_dict(c)))).cid == "c1"


# ------------------------------------------------------------------------------------------------ research priority engine

def failure_signal(subject="kn_1", when="2026-01-01", **kw):
    return rpri.make_signal("FAILURE", when, subject, kw.pop("mag", 0.4), subsystem="SELECTION", profile=kw.pop("profile", {}), **kw)


def test_signal_identity_firewall_and_question_generation():
    with pytest.raises(ValueError):
        rpri.QuestionGenerator().generate([failure_signal(subject="AAPL_2008-09-15")], "2026-02-01")
    s1, s2 = failure_signal(when="2026-01-01"), failure_signal(when="2026-01-05", mag=0.3)
    qs = rpri.QuestionGenerator().generate([s1, s2], "2026-02-01")
    assert len(qs) == 1 and len(qs[0].signals) == 2 and qs[0].magnitude > 0.4         # merged, magnitudes combined
    assert rpri.QuestionGenerator().generate([failure_signal(explained=True)], "2026-02-01") == []
    with pytest.raises(FirewallBreach):
        rpri.QuestionGenerator().generate([failure_signal(when="2026-03-01")], "2026-02-01")
    assert rpri.QuestionGenerator().generate([], "2026-02-01") == []
    assert rpri.identity_leak("what happened in 2008") and rpri.identity_leak("is it real") == ""


def test_failure_hypotheses_always_include_coincidence_and_unknown_and_follow_evidence():
    base = rpri.failure_hypotheses({})
    causes = {h.cause for h in base}
    assert "FALSE_PATTERN" in causes and "UNKNOWN" in causes
    assert sum(h.prior for h in base) == pytest.approx(1.0) and all(h.prior > 0 for h in base)
    ctx = rpri.failure_hypotheses({"concentrated_in_context": 1})
    lead = max(ctx, key=lambda h: h.prior)
    assert lead.cause == "WRONG_CONTEXT" and lead.prior > max(h.prior for h in base)
    anomaly = rpri.failure_hypotheses({"data_anomaly": 1})
    assert max(anomaly, key=lambda h: h.prior).cause == "MEASUREMENT_ERROR"
    assert min(h.prior for h in rpri.failure_hypotheses({"data_anomaly": 1, "concentrated_in_context": 1} ) if h.cause == "UNKNOWN") >= 0.04


def test_expected_outcome_tables_are_proper_distributions_and_discriminate():
    for c, row in rpri.SPLIT_TABLE.items():
        assert sum(row) == pytest.approx(1.0), c
    hyps = rpri.failure_hypotheses({})
    exp = rpri.split_expected(hyps)
    bits = rp.eig_from_expected(hyps, exp)
    assert 0.2 < bits < math.log2(len(hyps))
    same = rpri.SPLIT_TABLE[FailureCause.UNKNOWN]
    flat_hyps = [em.Hypothesis("a", "x", 0.5, cause="UNKNOWN"), em.Hypothesis("b", "y", 0.5, cause="UNKNOWN")]
    assert rp.eig_from_expected(flat_hyps, rpri.split_expected(flat_hyps)) == pytest.approx(0.0, abs=1e-12)    # identical rows cannot discriminate
    for kind in rpri.SignalKind:
        hs = rpri.generic_hypotheses(kind, "s")
        assert sum(h.prior for h in hs) == pytest.approx(1.0) and any(h.kind == "noise" or h.kind == "measurement" for h in hs)


def test_planted_diagnosis_finds_the_true_cause_and_admits_unknown():
    hits = [rpri.simulate_diagnosis(FailureCause.WRONG_CONTEXT, s, 8)["diagnosed"] for s in range(30)]
    assert np.mean(hits) > 0.8
    prior = rpri.simulate_diagnosis(FailureCause.WRONG_CONTEXT, 1, 0)["prior"]
    assert np.mean([rpri.simulate_diagnosis(FailureCause.WRONG_CONTEXT, s, 8)["posterior"] for s in range(30)]) > prior + 0.3
    outside = [rpri.simulate_diagnosis(FailureCause.MEASUREMENT_ERROR, s, 8) for s in range(30)]
    assert not outside[0]["in_hypothesis_set"]
    assert np.mean([o["confident_wrong"] for o in outside]) < 0.15                     # an incomplete set falls to UNKNOWN, not to a wrong answer


def test_queue_merge_aging_staleness_and_closure():
    q = rpri.ResearchQueue()
    b = rpri.CandidateBuilder()
    qs = rpri.QuestionGenerator().generate([failure_signal()], "2026-02-01")
    cand = b.build(qs[0], "2026-02-01").candidate
    q.enqueue(cand, "2026-02-01")
    q.enqueue(cand, "2026-02-02")                                                          # merge, not duplicate
    assert len(q.items) == 1 and q.items[cand.cid].last_evidence_at.startswith("2026-02-02")
    pol = rp.ResearchPolicy()
    fresh = q.rescore(pol, rp.PolicyContext(now="2026-02-03"))[0].adjusted
    old = q.rescore(pol, rp.PolicyContext(now="2026-06-03"))[0].adjusted                   # 4 months without new evidence
    assert old < fresh
    assert q.close_explained(cand.evidence, "2026-06-04") == [cand.cid]
    assert q.top(3) == [] and q.summary()["by_status"]["OBSOLETE"] == 1
    with pytest.raises(FirewallBreach):
        q.enqueue(rp.Candidate("late", "q", rp.ResearchTarget.FAILURE, "2027-01-01"), "2026-01-01")


def test_queue_persistence_round_trip_and_followups(tmp_path):
    q = rpri.ResearchQueue(log_path=tmp_path / "q.jsonl")
    b = rpri.CandidateBuilder()
    bq = b.build(rpri.QuestionGenerator().generate([failure_signal()], "2026-02-01")[0], "2026-02-01")
    q.enqueue(bq.candidate, "2026-02-01")
    q.start(bq.candidate.cid, "2026-02-02")
    with pytest.raises(ValueError):
        q.start(bq.candidate.cid, "2026-02-02")
    fu = rp.Candidate("fu1", "follow", rp.ResearchTarget.WEAK_PATTERN, "2026-02-03", factors=rp.Factors(uncertainty=1.0))
    made = q.complete(bq.candidate.cid, "2026-02-04", [fu])
    assert made[0].parent == bq.candidate.cid and made[0].candidate.factors.uncertainty == pytest.approx(0.8)
    back = rpri.queue_restore(rpri.queue_snapshot(q))
    assert set(back.items) == set(q.items) and back.items[bq.candidate.cid].status == rpri.ItemStatus.DONE
    log = rpri.replay_log(tmp_path / "q.jsonl")
    assert log["counts"]["enqueue"] == 2 and log["counts"]["complete"] == 1 and log["unparseable"] == 0


def test_engine_step_end_to_end_with_ledger_signals_and_result_update():
    led = em.ExperimentLedger()
    eng = rpri.ResearchPriorityEngine()
    signals = [failure_signal(f"kn_{i}", mag=0.3 + 0.1 * i, profile={"concentrated_in_context": 1} if i == 0 else {}) for i in range(3)]
    signals.append(rpri.make_signal("DATA_QUALITY", "2026-01-02", "channel_x", 0.9, stake=0.9))
    budget = rp.ComputeBudget(cpu_minutes=120, ram_gb_free=8)
    step = eng.step(signals, led, budget, "2026-02-01", seed=1)
    assert len(step.questions) == 4 and step.plan.selection.selected and rp.audit_plan(step.plan, eng.policy.cfg, budget) == []
    assert step.queue_summary["by_status"]["OPEN"] == 4
    chosen = step.plan.selection.selected[0].candidate
    bq = eng.built(chosen.cid)
    rec = eng.builder.to_experiment_record(bq, "2026-02-01", seed=3)
    led.propose(rec, "2026-02-02")
    res = em.ExperimentResult(em.ResultKind.CONFIRMED, "context_split_explains" if "context_split_explains" in {o.outcome for o in rec.expected_outcomes} else rec.expected_outcomes[0].outcome,
                              observed_at="2026-02-05", summary="s")
    done = led.record_result(rec.experiment_id, res, "2026-02-06", ["found it"], ["other eras"], "map the boundary")
    out = eng.update_from_result(done, chosen.cid, step.plan.selection.selected[0].eig_bits, "2026-02-06")
    assert out["realised_bits"] >= 0 and eng.queue.items[chosen.cid].status == rpri.ItemStatus.DONE
    assert len(eng.policy.history) == 1
    # answered questions do not come back
    step2 = eng.step(signals, led, budget, "2026-02-10", seed=2)
    assert all(rpri.question_key(q.text) != rpri.question_key(rec.question) for q in step2.questions)
    assert "Research queue" in rpri.render_queue(eng)


def test_adapters_reject_the_future_and_map_health():
    with pytest.raises(FirewallBreach):
        rpri.signals_from_failure_rows([{"knowledge_id": "k", "when": "2026-03-01"}], "2026-02-01")
    sig = rpri.signals_from_health([{"knowledge_id": "k1", "when": "2026-01-01", "health": "BROKEN"}, {"knowledge_id": "k2", "when": "2026-01-01", "health": "HEALTHY"},
                                    {"knowledge_id": "k3", "when": "2026-01-01", "health": "HEALTHY", "reliability_drop": 0.3}], "2026-02-01")
    assert {s.subject for s in sig} == {"k1", "k3"}
    audit = rpri.signals_from_data_audit({"a": "LEAK", "b": "OK", "c": "QUARANTINED"}, "2026-01-01", "2026-02-01")
    assert len(audit) == 2 and max(s.magnitude for s in audit) > 0.9
    mw = rpri.signals_from_missed_winners([{"situation_key": "sit_ab", "when": "2026-01-01", "gain_share": 0.2}], "2026-02-01")
    assert mw[0].kind == rpri.SignalKind.MISSED_WINNER
    merged = rpri.merge_similar_questions(rpri.QuestionGenerator().generate([failure_signal("kn_a"), failure_signal("kn_b")], "2026-02-01"))
    assert len(merged) >= 1


# ------------------------------------------------------------------------------------------------ failed learners

def test_seeded_registry_holds_the_real_history_and_audits_clean():
    reg = fl.FailedLearnerRegistry()
    out = fl.seed_registry(reg)
    assert out["added"] == 13 and len(reg) == 13
    names = {r.learner for r in reg.as_of("2026-10-01")}
    assert {"memory_bank", "basis_learner", "episodic_memory_retrieval", "missed_winner_detector", "lessons_from_other_windows",
            "cross_year_learners_all", "band_cfg", "direction_on_movers"} <= names
    mb = reg.get("memory_bank", "2026-10-01")
    assert mb.failure_mode == fl.FailureMode.MEMORISATION and mb.validation_result["transfer_mean_week"] < 0
    assert reg.get("episodic_memory_retrieval", "2026-10-01").validation_result["walk_forward_skill"] == -0.16
    assert reg.get("lessons_from_other_windows", "2026-10-01").validation_result["weekly_effect"] == pytest.approx(-0.00188)
    assert fl.audit_registry(reg, "2026-10-01")["ok"]
    assert fl.seed_registry(reg)["added"] == 0                                        # idempotent
    assert reg.as_of("2026-01-01") == []                                              # nothing existed before it was recorded


def test_dead_end_gate_blocks_a_renamed_memory_bank_and_clears_a_novel_idea():
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    renamed = fl.LearnerProposal("shiny_new_memory", "store window numeric memories to recall what worked before", ("stored_numeric_memory", "memory_bank"), "memory", ("real",))
    v = reg.check_proposal(renamed, "2026-10-01")
    assert v.blocked and "memory_bank" in v.message and "memoriser_control" in v.required_controls
    with pytest.raises(fl.DeadEndRepeat):
        reg.require_clear(renamed, "2026-10-01")
    novel = fl.LearnerProposal("causal_graph", "infer causal structure between macro drivers and regimes", ("causal_discovery", "macro_graph"), "structure", ("real",))
    assert reg.check_proposal(novel, "2026-10-01").status == fl.DeadEndStatus.CLEAR


def test_justification_must_add_a_new_mechanism_and_the_missing_controls():
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    p = fl.LearnerProposal("mem2", "store window numeric memories to recall what worked before", ("stored_numeric_memory", "memory_bank", "abstraction_layer"), "memory", ("real",))
    weak = fl.Justification(("abstraction_layer",), "it abstracts", ("noise_control",))
    assert reg.check_proposal(p, "2026-10-01", weak).blocked                                # controls for MEMORISATION missing
    same = fl.Justification(("memory_bank",), "it is better", ("memoriser_control", "past_only_transfer", "disguised_rerun"))
    assert reg.check_proposal(p, "2026-10-01", same).blocked                                # no NEW mechanism
    ok = fl.Justification(("abstraction_layer",), "stores rules not paths", ("memoriser_control", "past_only_transfer", "disguised_rerun"))
    assert reg.check_proposal(p, "2026-10-01", ok).status == fl.DeadEndStatus.JUSTIFIED


def test_single_regime_failure_allows_retry_in_a_new_regime():
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    p = fl.LearnerProposal("dir2", "predict the direction of a mover next week above the base rate", ("direction_classifier", "mover_features"), "direction", ("intraday panel",))
    v = reg.check_proposal(p, "2026-10-01")
    assert v.status == fl.DeadEndStatus.RETRY_ALLOWED_NEW_REGIME


def test_retest_supersedes_and_history_is_kept(tmp_path):
    reg = fl.FailedLearnerRegistry(tmp_path / "fl.jsonl")
    fl.seed_registry(reg)
    old = reg.get("basis_learner", "2026-10-01")
    new = reg.retest("basis_learner", fl.replace(old, recorded_at="2026-11-01T00:00:00", reason="re-run on 2027 windows: still no transfer"))
    assert new.version == 2 and new.supersedes == old.key
    assert reg.get("basis_learner", "2026-10-15").version == 1 and reg.get("basis_learner", "2026-12-01").version == 2
    assert len(reg.history("basis_learner")) == 2
    again = fl.FailedLearnerRegistry(tmp_path / "fl.jsonl")
    assert len(again) == 13 and again.get("basis_learner", "2026-12-01").version == 2
    with pytest.raises(ValueError):
        reg.add(old)                                                                       # versions only go forward
    with pytest.raises(KeyError):
        reg.retest("ghost", old)


def test_validation_refuses_incomplete_failures():
    reg = fl.FailedLearnerRegistry()
    seed = fl.seed_history()[0]
    for kw in ({"reason": ""}, {"data_regime": ()}, {"validation_result": {}}, {"mechanism_tags": ()},
               {"generalization": fl.Generalization.GENERALIZED, "regimes_failed": ("only one",)},
               {"regimes_failed": ("r",), "regimes_passed": ("r",)}):
        with pytest.raises(ValueError):
            reg.add(fl.replace(seed, **kw))


def test_failure_mode_classification_from_harness_numbers():
    C = fl.classify_failure_mode
    assert C(0.002, (0.0001, 0.005), -0.0003, (-0.0009, 0.0002)) == fl.FailureMode.MEMORISATION
    assert C(0.0, (-0.001, 0.001), 0.0, (-0.001, 0.001)) == fl.FailureMode.NO_TRANSFER
    assert C(None, None, -0.01, (-0.02, -0.005)) == fl.FailureMode.HARMFUL
    assert C(0.01, (0.001, 0.02), 0.01, (0.002, 0.02), guard_worse=True) == fl.FailureMode.TRADEOFF_HARM
    assert C(0.01, (0.001, 0.02), 0.01, (0.002, 0.02)) is None                              # a real pass is not a failure
    assert C(0.0, (0, 0), 0.0, (0, 0), adopted_anything=False) == fl.FailureMode.NO_EFFECT
    assert C(0.05, (0.01, 0.1), 0.05, (0.01, 0.1), leak_found=True) == fl.FailureMode.LEAKAGE
    summary = {"metrics": {"mean_week": {"same": {"mean": 0.002, "lo": 0.0001, "hi": 0.005, "n": 12}, "transfer": {"mean": -0.0003, "lo": -0.0009, "hi": 0.0002, "n": 12}}}}
    f = fl.from_harness_summary("x", summary, "2026-10-01T00:00:00", "m::X", "h", ("t1",), "fam", "reg", ("p",))
    assert f.failure_mode == fl.FailureMode.MEMORISATION and f.validation_result["pairs"] == 12
    passing = {"metrics": {"mean_week": {"same": {"mean": 0.01, "lo": 0.005, "hi": 0.02, "n": 12}, "transfer": {"mean": 0.01, "lo": 0.004, "hi": 0.02, "n": 12}}}}
    assert fl.from_harness_summary("y", passing, "2026-10-01T00:00:00", "m::Y", "h", ("t",)) is None
    with pytest.raises(ValueError):
        fl.from_harness_summary("z", {}, "2026-10-01T00:00:00", "m", "h", ("t",))


def test_generalization_classifier_advice_and_evidence_verification():
    G = fl.classify_generalization
    assert G(("a", "b"), ()) == fl.Generalization.GENERALIZED and G(("a",), ("b",)) == fl.Generalization.REGIME_SPECIFIC
    assert G(("a",), ()) == fl.Generalization.SINGLE_REGIME and G((), ()) == fl.Generalization.UNKNOWN and G(("a",), (), 2) == fl.Generalization.GENERALIZED
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    adv = fl.advice_for_next_learner(reg, fl.LearnerProposal("p", "veto rules from lessons in other windows", ("veto_rules",), "lessons", ("real",)), "2026-10-01", ["real", "synthetic planted"])
    assert "veto_rules" in adv["tags_that_keep_failing"] and adv["required_controls"]
    ev = reg.verify_evidence(".", "2026-10-01")
    assert set(ev) == {"present", "missing", "verified_learners", "unverified_learners"}
    assert fl.jaccard([], []) == 0.0 and "modes" in fl.render_report(reg, "2026-10-01")


# ------------------------------------------------------------------------------------------------ part 2 mechanisms

def test_starvation_explanation_and_plan_outcome_distribution():
    hist = [rp.RealisedGain(f"g{i}", rp.ResearchTarget.FAILURE, "f", .1, .1, True, 5, "2026-01-01") for i in range(3)]
    st = rp.starving_targets(hist, "2026-06-01", open_targets=["FAILURE", "CONTRADICTION"])
    assert {s["target"] for s in st} == {"FAILURE", "CONTRADICTION"} and [s for s in st if s["target"] == "CONTRADICTION"][0]["never_run"]
    alloc = rp.TargetAllocator()
    a = alloc.allocate(100, ALL_OPEN, 7)
    rows = rp.explain_allocation(a, alloc, 7, ALL_OPEN)
    assert abs(sum(r["share"] for r in rows) - 1) < 1e-9 and rows[0]["share"] >= rows[-1]["share"]
    pol = rp.ResearchPolicy()
    plan = pol.plan(make_cands(6), rp.PolicyContext(now="2026-02-01"), rp.ComputeBudget(cpu_minutes=80, ram_gb_free=8), 1)
    d = rp.plan_outcome_distribution(plan, pol.allocator, seed=1)
    assert 0 < d["p_at_least_one"] <= 1 and d["p_nothing"] == pytest.approx(1 - d["p_at_least_one"])
    empty = pol.plan([], rp.PolicyContext(now="2026-02-01"), rp.ComputeBudget(cpu_minutes=10, ram_gb_free=8), 1)
    assert rp.plan_outcome_distribution(empty)["p_nothing"] == 1.0


def test_second_pass_fills_idle_minutes_and_respects_limits():
    pol = rp.ResearchPolicy()
    budget = rp.ComputeBudget(cpu_minutes=40, ram_gb_free=8, real_data_slots=1)
    plan = pol.plan(make_cands(8, real=8), rp.PolicyContext(now="2026-02-01"), budget, 1)
    assert plan.selection.deferred
    extra = rp.second_pass(plan, minutes_actually_used=5.0, budget=budget)
    assert extra.used_minutes <= 35 + 1e-9
    room = max(0, budget.real_data_slots - sum(1 for s in plan.selection.selected if s.candidate.cost.real_data))
    assert sum(1 for s in extra.selected if s.candidate.cost.real_data) <= room
    assert rp.second_pass(plan, minutes_actually_used=40.0, budget=budget).selected == ()


def test_expiry_and_planted_world_tuning_and_baseline_comparison():
    kept, gone = rp.expire_candidates([base_cand(cid="old", created_at="2025-01-01"), base_cand(cid="new", created_at="2026-01-20")], "2026-02-01")
    assert [c.cid for c in kept] == ["new"] and [c.cid for c in gone] == ["old"]
    worlds = [{"KNOWN_RELIABLE": 0.03, "FAILURE": 0.6}, {"KNOWN_RELIABLE": 0.5, "FAILURE": 0.05, "UNKNOWN_AREA": 0.05}]
    rows = rp.tune_on_planted(worlds, [rp.PolicyConfig(), rp.PolicyConfig(exploit_cap=0.9)], [1, 2], rounds=20)
    assert len(rows) == 2 and rows[0]["mean_efficiency"] >= rows[1]["mean_efficiency"]
    cmp_ = rp.policy_regret_vs_baselines({"KNOWN_RELIABLE": 0.03, "FAILURE": 0.6}, 30, 100, [1, 2, 3])
    assert cmp_["bandit_beats_both_every_seed"]
    with pytest.raises(ValueError):
        rp.tune_on_planted([], [rp.PolicyConfig()], [1])


def test_propensities_round_ledger_and_ips_join(tmp_path):
    pol = rp.ResearchPolicy()
    prop = rp.estimate_propensities(pol.allocator, ALL_OPEN, seed=1, n_draws=40)
    assert sum(prop.values()) == pytest.approx(1.0) and min(prop.values()) >= 0
    budget = rp.ComputeBudget(cpu_minutes=80, ram_gb_free=8)
    plan = pol.plan(make_cands(6), rp.PolicyContext(now="2026-02-01"), budget, 1)
    led = rp.RoundLedger(tmp_path / "rounds.jsonl")
    row = led.log(plan, 1, 80, prop, pol.cfg)
    with pytest.raises(ValueError):
        led.log(plan, 1, 80, prop, pol.cfg)
    assert led.as_of("2026-01-01") == [] and len(led.as_of("2026-03-01")) == 1
    cid = plan.selection.selected[0].candidate.cid
    gain = rp.RealisedGain(cid, plan.selection.selected[0].candidate.target, "f", .2, .3, True, 10, "2026-02-05")
    other = rp.RealisedGain("unlogged", rp.ResearchTarget.FAILURE, "f", .1, .1, True, 5, "2026-02-06")
    choices = led.logged_choices([gain, other], "2026-03-01")
    assert len(choices) == 1 and choices[0].propensity >= 0.02
    assert len(rp.RoundLedger(tmp_path / "rounds.jsonl").rows) == 1 and row.round_id


def test_marginal_value_of_compute_is_nondecreasing_and_policy_health_flags_problems():
    pol = rp.ResearchPolicy()
    rows = rp.marginal_value_of_compute(pol, make_cands(10), rp.PolicyContext(now="2026-02-01"), rp.ComputeBudget(cpu_minutes=40, ram_gb_free=8))
    bits = [r["total_bits"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(bits, bits[1:])) and rows[-1]["selected"] >= rows[0]["selected"]
    assert rp.policy_health(pol, "2026-06-01")["healthy"]
    sick = rp.ResearchPolicy()
    for i in range(25):
        sick.observe(rp.RealisedGain(f"g{i}", rp.ResearchTarget.KNOWN_RELIABLE, "tuning", 1.0, 0.0, False, 10, f"2026-01-{i + 1:02d}"))
    sick.observe(rp.RealisedGain("z", rp.ResearchTarget.FAILURE, "d", .5, 1.0, True, 10, "2026-02-01"))
    h = rp.policy_health(sick, "2026-06-01")
    assert not h["healthy"] and any("known-reliable" in p for p in h["problems"])


def test_maintenance_backlog_pressure_and_machine_budget():
    old = rp.ReliableItem("old", "2025-01-01", 0.6, 0.9)
    fresh = rp.ReliableItem("fresh", "2026-01-30", 0.9, 0.9)
    lowstake = rp.ReliableItem("lowstake", "2025-01-01", 0.6, 0.02)
    assert rp.verification_urgency(old, "2026-02-01") > rp.verification_urgency(fresh, "2026-02-01") * 10
    sched = rp.maintenance_schedule([old, fresh, lowstake], "2026-02-01", minutes_available=8.0)
    assert [s["knowledge_id"] for s in sched["schedule"]] == ["old"] and "fresh" in sched["skipped_not_urgent"]
    two = rp.maintenance_schedule([old, rp.ReliableItem("old2", "2025-03-01", 0.6, 0.8)], "2026-02-01", minutes_available=8.0)
    assert len(two["schedule"]) == 1 and len(two["skipped_no_minutes"]) == 1
    with pytest.raises(FirewallBreach):
        rp.verification_urgency(rp.ReliableItem("k", "2027-01-01", .5, .5), "2026-02-01")
    with pytest.raises(ValueError):
        rp.maintenance_schedule([rp.ReliableItem("bad", "2025-01-01", 1.5, .5)], "2026-02-01", 8.0)
    p = rp.backlog_pressure([{"target": "CONTRADICTION", "magnitude": 0.5, "since": "2025-01-01"},
                             {"target": "FAILURE", "magnitude": 0.5, "since": "2026-01-31"}], "2026-02-01")
    assert p["CONTRADICTION"] > p["FAILURE"]
    assert rp.contradiction_debt([{"strength": 1.0, "since": "2026-01-02"}], "2026-02-01") == pytest.approx(1.0, abs=0.05)
    assert rp.budget_from_machine(60, ram_free_gb=1.0).real_data_slots == 0             # too little free RAM: no real-data job
    assert rp.budget_from_machine(60, ram_free_gb=9.0).real_data_slots == 1


def test_successive_halving_finds_the_best_arm_within_budget():
    quality = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.4, "e": 0.1, "f": 0.6, "g": 0.3, "h": 0.05}
    res = rp.successive_halving(list(quality), lambda arm, budget: quality[arm] + 0.1 / budget, min_budget=1, total_budget=40)
    assert res.survivors == ("b",) and res.total_cost <= 40 and len(res.eliminated) == 7
    with pytest.raises(ValueError):
        rp.successive_halving(["a", "b"], lambda a, b: 0.0, min_budget=5, total_budget=6)


def test_engine_helpers_proposing_pressure_stability_and_relevance():
    led = em.ExperimentLedger()
    eng = rpri.ResearchPriorityEngine()
    signals = [failure_signal(f"kn_{i}", mag=0.4) for i in range(3)]
    step = eng.step(signals, led, rp.ComputeBudget(cpu_minutes=120, ram_gb_free=8), "2026-02-01", seed=1)
    out = rpri.propose_selected(step, eng, led, "2026-02-02", seed=4)
    assert out["proposed"] and not out["refused"] and len(led.open_questions("2026-02-03")) == len(out["proposed"])
    again = rpri.propose_selected(step, eng, led, "2026-02-03", seed=4)                 # the ledger now knows these are in flight
    assert again["refused"] and not again["proposed"]
    assert rpri.queue_health(eng.queue, "2026-02-04")["open"] <= 3
    assert all(v >= 0 for v in rpri.queue_pressure(eng.queue, "2026-03-01").values())
    assert "Research agenda" in rpri.agenda_markdown(eng) and "Research step" in rpri.explain_step(step)
    # hysteresis: a 5 percent lead does not displace the incumbent, a 50 percent lead does
    assert rpri.stable_order(["a", "b"], {"a": 1.0, "b": 1.05}) == ["a", "b"]
    assert rpri.stable_order(["a", "b"], {"a": 1.0, "b": 1.5}) == ["b", "a"]
    assert rpri.stable_order(["gone", "a"], {"a": 1.0, "new": 2.0}) == ["new", "a"]
    assert rpri.decision_relevance(["NONE"]) < rpri.decision_relevance(["SELECTION"]) and rpri.decision_relevance([]) == 0.0
    with pytest.raises(ValueError):
        rpri.decision_relevance(["NOT_A_DECISION"])
    assert rpri.relevance_from_subsystem("SELECTION", 1.0) > rpri.relevance_from_subsystem("", 1.0)


def test_signal_ageing_dedupe_and_surprise_adapter():
    s = failure_signal(when="2026-01-01", mag=0.8)
    assert rpri.age_weighted_magnitude(s, "2026-04-01", half_life_days=90) == pytest.approx(0.4, abs=0.02)
    with pytest.raises(FirewallBreach):
        rpri.age_weighted_magnitude(s, "2026-01-01")
    uniq, dropped = rpri.dedupe_signals([s, s, failure_signal(when="2026-01-01", mag=0.8, subject="other")])
    assert len(uniq) == 2 and dropped == 1
    rows = [{"subject": "pat_1", "when": "2026-01-01", "expected": 0.01, "observed": -0.05, "sd": 0.01},
            {"subject": "pat_2", "when": "2026-01-01", "expected": 0.01, "observed": 0.012, "sd": 0.01}]
    sig = rpri.signals_from_surprise_rows(rows, "2026-02-01")
    assert [x.subject for x in sig] == ["pat_1"] and sig[0].magnitude == pytest.approx(6.0)
    with pytest.raises(ValueError):
        rpri.signals_from_surprise_rows([{**rows[0], "sd": 0.0}], "2026-02-01")


def test_ledger_lineage_staleness_and_unanswered_hypotheses():
    led = em.ExperimentLedger()
    led.propose(mk_record("root"), "2026-01-02")
    answer(led, "root")
    child = mk_record("child", q="Which sector explains the volume filter failure?", cfg={"sector": "tech"}, now="2026-02-01").with_(parent_ids=("root",))
    led.propose(child, "2026-02-02")
    lin = em.lineage(led, "child", "2026-03-01")
    assert lin["ancestors"] == ["root"] and em.lineage(led, "root", "2026-03-01")["descendants"] == ["child"]
    ghost = mk_record("orph", q="Completely different topic about exits", cfg={"exit_q": 0.9}, now="2026-02-01").with_(parent_ids=("nope",))
    led.propose(ghost, "2026-02-03")
    assert em.lineage(led, "orph", "2026-03-01")["missing_parents"] == ["nope"]
    assert em.stale_answers(led, "2026-03-01", "c1") == [] and em.stale_answers(led, "2026-03-01", "c2")[0]["reasons"] == ["code changed"]
    assert em.unanswered_hypotheses(led, "2026-03-01", threshold=0.05)
    with pytest.raises(KeyError):
        em.lineage(led, "root", "2026-01-01")


def test_failed_learner_text_tagging_and_markdown():
    tags = fl.infer_tags("We will remember what worked: a memory bank of window-level memories with veto lessons from other windows")
    assert {"memory_bank", "veto_rules", "lessons_from_other_windows"} <= set(tags)
    p = fl.proposal_from_text("renamed", "Recall past situations from a memory bank of similar past situations", "memory", ("real",))
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    assert reg.check_proposal(p, "2026-10-01").blocked                                   # a rename does not escape the record
    with pytest.raises(ValueError):
        fl.proposal_from_text("mystery", "something wholly unrecognisable xyzzy")
    assert fl.failure_summary_for(reg, ["direction_classifier"], "2026-10-01")[0]["learner"] == "direction_on_movers"
    md = fl.to_markdown(reg, "2026-10-01")
    assert md.startswith("# Failed learners") and "A retry must include" in md


# ------------------------------------------------------------------------------------------------ measured costs, registry overfit, inventory, experience

def test_measured_costs_replace_priors_and_ignore_the_future_and_junk():
    rows = [{"t": "2026-01-01T00:00:00", "event": "split_by_context_and_era", "runtime_seconds": 3600},
            {"t": "2026-01-02T00:00:00", "event": "split_by_context_and_era", "runtime_seconds": 4200},
            {"t": "2026-01-03T00:00:00", "event": "split_by_context_and_era", "runtime_seconds": 0},          # artefact
            {"t": "2026-09-01T00:00:00", "event": "split_by_context_and_era", "runtime_seconds": 999999},     # future
            {"t": "2026-01-04T00:00:00", "event": "no_duration"}]
    mc = rp.MeasuredCosts(prior_strength=1.0)
    out = mc.add_rows(rows, "2026-06-01")
    assert out["used"] == 2 and out["skipped"] == 3
    est, n = mc.estimate("split_by_context_and_era", prior_minutes=15)
    assert n == 2 and 40 < est < 70                                    # pulled from the 15-min prior toward the measured ~65
    assert mc.estimate("unseen", 15) == (15, 0)
    c = base_cand(config={"template": "split_by_context_and_era"}, cost=rp.ComputeCost(cpu_minutes=15, wall_minutes=10))
    assert mc.apply(c).cost.cpu_minutes == pytest.approx(est) and mc.apply(base_cand()).cost.cpu_minutes == 10
    assert rp.row_minutes({"duration_s": 120}) == 2.0 and rp.row_minutes({"seconds": float("nan")}) is None
    # a measured overrun now makes the plan defer what the prior thought cheap
    pol = rp.ResearchPolicy()
    budget = rp.ComputeBudget(cpu_minutes=30, ram_gb_free=8)
    ctx = rp.PolicyContext(now="2026-06-01")
    assert pol.priority.score(c, ctx, budget).blocked == ""
    assert "cpu-min" in pol.priority.score(mc.apply(c), ctx, budget).blocked


def test_failed_learner_registry_drives_the_overfit_penalty():
    reg = fl.FailedLearnerRegistry()
    fl.seed_registry(reg)
    risk, why = rp.registry_overfit_risk(("stored_numeric_memory", "memory_bank"), reg, "2026-10-01")
    assert risk > 0.4 and any("memory_bank" in w for w in why)
    assert rp.registry_overfit_risk(("causal_discovery",), reg, "2026-10-01") == (0.0, [])
    assert rp.registry_overfit_risk(("memory_bank",), reg, "2026-01-01")[0] == 0.0              # nothing was known yet
    pol = rp.ResearchPolicy()
    ctx = rp.PolicyContext(now="2026-10-01")
    plain = base_cand(created_at="2026-09-01")
    tagged = rp.with_registry_overfit(plain, ("stored_numeric_memory", "memory_bank"), reg, "2026-10-01")
    assert tagged.overfit_hint > 0.4 and pol.priority.score(tagged, ctx).priority < pol.priority.score(plain, ctx).priority * 0.7
    assert rp.with_registry_overfit(plain, ("causal_discovery",), reg, "2026-10-01") == plain
    hinted = base_cand(created_at="2026-09-01", overfit_hint=0.95)
    assert rp.with_registry_overfit(hinted, ("memory_bank",), reg, "2026-10-01").overfit_hint == 0.95   # never lowers


def test_cache_inventory_gates_unavailable_data(tmp_path):
    (tmp_path / "delisted_prices.parquet").write_bytes(b"x")
    (tmp_path / "events.parquet").write_bytes(b"x")
    (tmp_path / "market_context_daily.csv").write_bytes(b"x")
    inv = rp.cache_inventory(tmp_path)
    assert "delisted" in inv and "events" in inv and rp.cache_inventory(tmp_path / "missing") == frozenset()
    have = rp.available_datasets(["price_panel", "market_context", "delisted_history", "sentiment_feed"], inv)
    assert have == frozenset({"price_panel", "market_context", "delisted_history"})
    ctx = rp.context_with_inventory(rp.PolicyContext(now="2026-02-01"), tmp_path, ["price_panel", "sentiment_feed"])
    pol = rp.ResearchPolicy()
    assert pol.priority.score(base_cand(data_needs=("price_panel",)), ctx).blocked == ""
    s = pol.priority.score(base_cand(data_needs=("sentiment_feed",)), ctx)
    assert s.priority == 0 and "sentiment_feed" in s.blocked


def test_queue_reranks_by_realised_gain_and_decays_barren_targets():
    eng = rpri.ResearchPriorityEngine()
    mk = lambda cid, fam, tgt: rp.Candidate(cid, "q " + cid, tgt, "2026-01-01", family=fam, magnitude=0.5,
                                            info=rp.InfoModel("normal", {"n_now": 10, "n_new": 30}))
    for c in (mk("good", "fam_good", rp.ResearchTarget.FAILURE), mk("bad", "fam_bad", rp.ResearchTarget.KNOWN_RELIABLE),
              mk("bad2", "fam_bad", rp.ResearchTarget.KNOWN_RELIABLE)):
        eng.queue.enqueue(c, "2026-01-02")
    for i in range(6):
        eng.policy.observe(rp.RealisedGain(f"g{i}", rp.ResearchTarget.FAILURE, "fam_good", .3, 0.5, True, 10, f"2026-02-0{i + 1}"))
        eng.policy.observe(rp.RealisedGain(f"b{i}", rp.ResearchTarget.KNOWN_RELIABLE, "fam_bad", .3, 0.0, False, 10, f"2026-02-0{i + 1}"))
    assert rpri.barren_streak(eng.policy.history, rp.ResearchTarget.KNOWN_RELIABLE) == 6
    assert rpri.barren_streak(eng.policy.history, rp.ResearchTarget.FAILURE) == 0
    eng.queue.rescore(eng.policy, rp.PolicyContext(now="2026-03-01"))
    before = {i.candidate.cid: i.adjusted for i in eng.queue.items.values()}
    order = rpri.experience_rerank(eng)
    after = {i.candidate.cid: i.adjusted for i in eng.queue.items.values()}
    assert after["good"] >= before["good"] and after["bad"] < before["bad"] * 0.4
    assert order[0] == "good" and "barren streak 6" in eng.queue.items["bad"].note
    item = eng.queue.items["good"]
    assert rpri.experience_factor(item, [])[0] == 1.0                                     # no experience, no change
    # one useful run in the barren target resets the streak
    eng.policy.observe(rp.RealisedGain("rescue", rp.ResearchTarget.KNOWN_RELIABLE, "fam_bad", .3, 0.4, True, 10, "2026-02-20"))
    assert rpri.barren_streak(eng.policy.history, rp.ResearchTarget.KNOWN_RELIABLE) == 0
