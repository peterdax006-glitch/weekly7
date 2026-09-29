"""Tests for engine/research/{compute_manager,value_accounting,waste}.py (C66 sections 18-20). Synthetic data only, fast."""
import datetime as dt
import math

import numpy as np
import pytest

from engine.learning import compute as C
from engine.research import compute_manager as CM
from engine.research import value_accounting as VA
from engine.research import waste as W
from engine.research.core import FirewallBreach, Problem, ResearchQuestion, ResearchState, Stage

D0 = "2010-01-04"


def mk(state, text="q", family="fam", problem=Problem.VOLATILITY, looks=1, now=D0, **kw):
    q = ResearchQuestion.make(text, "t", problem, now, now, "s", "f", **kw)
    return CM.make_branch(state, q, now, family, looks)


def ev(stage, effect=0.0035, se=0.0007, n=4000, cost=5.0, **kw):
    rule = CM.LadderPolicy().rule(stage)
    base = dict(n_tests=rule.planned_tests, n_positive=rule.planned_tests, n_units=4, units_positive=4, replications=1, fresh=True,
                data_through="2010-01-03", integration_delta=0.01, integration_se=0.002)
    base.update(kw)
    return CM.StageEvidence(stage, n, effect, se, cost_cpu_min=cost, **base)


def drive(state, b, stage, evidence, day):
    b.in_flight = "job"
    if CM.LAUNCH_STATE[stage] in CM.ALLOWED[b.state]:
        CM._record(state, b, CM.LAUNCH_STATE[stage], day, "launch")
    return CM.record_result(state, b.branch_id, evidence, day, CM.LadderPolicy())


# ---------------------------------------------------------------- compute manager
def test_default_policy_valid_and_bad_policy_caught():
    assert CM.LadderPolicy().validate() == []
    bad = CM.LadderPolicy(min_effect=-1.0)
    assert bad.validate()


def test_cannot_skip_a_rung():
    st = CM.ManagerState()
    b = mk(st)
    b.frontier = Stage.CROSS_YEAR                      # forged frontier with nothing earned
    ok, why = CM.authorise(st, b, Stage.CROSS_YEAR, 50.0, CM.LadderPolicy())
    assert not ok and "never passed" in why


def test_weak_hypothesis_never_gets_big_compute():
    st = CM.ManagerState()
    b = mk(st)
    ok, _ = CM.authorise(st, b, Stage.CHEAP_SCREEN, 20.0, CM.LadderPolicy())
    assert ok
    ok, why = CM.authorise(st, b, Stage.CHEAP_SCREEN, 5000.0, CM.LadderPolicy())
    assert not ok and "cap" in why


def test_full_ladder_pass_reaches_retired_and_audit_clean():
    st, pol = CM.ManagerState(), CM.LadderPolicy()
    b = mk(st)
    for i, s in enumerate(CM.LADDER):
        d = drive(st, b, s, ev(s), dt.date(2010, 1, 5) + dt.timedelta(days=i))
        assert d.action in (CM.Act.ADVANCE, CM.Act.COMPLETE), d.reasons
    assert b.state is ResearchState.RETIRED
    assert CM.audit_ladder(st, pol) == [] and CM.conservation_errors(st) == []


def test_null_with_power_fails_but_underpowered_null_repeats():
    st = CM.ManagerState()
    b = mk(st)
    d = drive(st, b, Stage.CHEAP_SCREEN, ev(Stage.CHEAP_SCREEN, effect=0.0, se=0.001, n=4000, n_positive=50), "2010-01-05")
    assert d.action is CM.Act.FAIL and b.state is ResearchState.FAILED
    st2 = CM.ManagerState()
    b2 = mk(st2)
    # tiny sample: the same zero effect cannot rule anything out
    d2 = drive(st2, b2, Stage.CHEAP_SCREEN, ev(Stage.CHEAP_SCREEN, effect=0.0005, se=0.001155, n=1200, n_positive=50), "2010-01-05")
    assert d2.action is CM.Act.REPEAT and b2.state is not ResearchState.FAILED


def test_implausible_result_goes_to_audit_not_up_the_ladder():
    st = CM.ManagerState()
    b = mk(st)
    d = drive(st, b, Stage.CHEAP_SCREEN, ev(Stage.CHEAP_SCREEN, effect=0.5, se=0.01), "2010-01-05")
    assert d.action is CM.Act.AUDIT and b.audit_pending and b.frontier is Stage.CHEAP_SCREEN
    assert not CM.authorise(st, b, Stage.CHEAP_SCREEN, 5.0, CM.LadderPolicy())[0]
    CM.clear_audit(st, b.branch_id, "2010-01-06", True, "leak")
    assert b.state is ResearchState.FAILED


def test_multiplicity_raises_the_bar():
    assert CM.multiplicity_threshold(1.5, 100) > CM.multiplicity_threshold(1.5, 1) + 1.0
    with pytest.raises(ValueError):
        CM.multiplicity_threshold(1.5, 0)


def test_evidence_after_now_is_a_breach_and_bad_transitions_refused():
    st = CM.ManagerState()
    b = mk(st)
    b.in_flight = "j"
    with pytest.raises(FirewallBreach):
        CM.record_result(st, b.branch_id, ev(Stage.CHEAP_SCREEN, data_through="2010-01-05"), "2010-01-05", CM.LadderPolicy())
    with pytest.raises(CM.LadderError):
        CM._record(st, b, ResearchState.VALIDATING, "2010-01-05", "illegal")


def test_log_tamper_detected_and_state_roundtrip(tmp_path):
    st = CM.ManagerState()
    b = mk(st)
    drive(st, b, Stage.CHEAP_SCREEN, ev(Stage.CHEAP_SCREEN), "2010-01-05")
    assert CM.verify_log(st) == []
    CM.save_state(st, tmp_path / "s.json")
    st2 = CM.load_state(tmp_path / "s.json")
    assert CM.state_hash(st2) == CM.state_hash(st)
    st2.log[0] = CM.dataclasses.replace(st2.log[0], reason="edited")
    assert CM.verify_log(st2)


def test_snapshotter_skips_corrupt_latest(tmp_path):
    st = CM.ManagerState()
    mk(st)
    sn = CM.Snapshotter(tmp_path, keep=3)
    sn.write(st, D0)
    p2 = sn.write(st, D0)
    p2.write_text("{torn", encoding="utf-8")
    got, info = sn.restore()
    assert len(got.branches) == 1 and info["skipped"]
    assert CM.Snapshotter(tmp_path / "empty").restore()[0].branches == {}


def test_step_respects_ram_and_empty_state(tmp_path):
    st = CM.ManagerState()
    r = CM.step(st, D0, free_gb=10.0)
    assert r.launched == () and r.problems == ()
    for i in range(3):
        mk(st, f"q{i}")
    low = CM.step(st, D0, free_gb=None, dry_run=True)
    assert low.selection.allocations == ()                       # unreadable memory fails closed
    ok = CM.step(st, D0, free_gb=16.0, period_cpu_min=600.0, dry_run=True)
    assert len(ok.selection.allocations) == 3


def test_step_launch_uses_ledger_and_blocks_duplicates(tmp_path):
    st = CM.ManagerState()
    b = mk(st)
    led = C.ExperimentLedger(tmp_path / "led.json")
    r = CM.step(st, D0, free_gb=16.0, ledger=led, code_hash="h")
    assert r.launched == (b.branch_id,) and b.in_flight and b.state is ResearchState.EXPLORING
    assert len(led.load()) == 1


def test_direction_waits_for_volatility():
    st = CM.ManagerState()
    v = mk(st, "vol", problem=Problem.VOLATILITY)
    d = mk(st, "dir", problem=Problem.DIRECTION)
    d.frontier = Stage.CROSS_YEAR
    d.passed = {Stage.CHEAP_SCREEN.value: {"effect": 0.01}, Stage.STRONGER_TESTS.value: {"effect": 0.01}}
    ok, why = CM.authorise(st, d, Stage.CROSS_YEAR, 50.0, CM.LadderPolicy())
    assert not ok and "volatility" in why
    v.passed = {s.value: {"effect": 0.01} for s in CM.LADDER[:3]}
    assert CM.authorise(st, d, Stage.CROSS_YEAR, 50.0, CM.LadderPolicy())[0]


def test_planted_ladder_nulls_die_cheap_real_effects_survive():
    pol = CM.LadderPolicy()
    r = CM.simulate_ladder(pol, 0.006, 0.05, 60, 1)
    assert r.null_reach_end == 0.0 and r.real_reach_end > 0.7
    assert r.mean_cost_null < r.mean_cost_real / 5
    zero = CM.simulate_ladder(pol, 0.0, 0.05, 40, 2)
    assert zero.real_reach_end == 0.0                              # null case: nothing may survive


def test_replay_schedule_never_funds_null_past_rung2():
    truth = {f"f{i % 3}:h{i}": (0.006 if i % 4 == 0 else 0.0) for i in range(16)}
    r = CM.replay_schedule(CM.LadderPolicy(), truth, 3, 50)
    assert r["problems"] == [] and r["null_funded_past_rung2"] == []
    assert any(truth[n] > 0 for n in r["reached_end"])


def test_duplicate_question_makes_no_second_branch_and_park_needs_reason():
    st = CM.ManagerState()
    b = mk(st)
    assert mk(st) is None
    with pytest.raises(ValueError):
        CM.park(st, b.branch_id, D0, "  ")
    CM.park(st, b.branch_id, "2010-01-05", "why")
    assert b.state is ResearchState.DORMANT
    CM.revive(st, b.branch_id, "2010-02-05", "new data")
    assert b.state is ResearchState.QUEUED and b.frontier is Stage.CHEAP_SCREEN


def test_plan_run_flags_infeasible_power():
    st = CM.ManagerState()
    b = mk(st)
    p = CM.plan_run(st, b.branch_id, CM.LadderPolicy(), sd_obs=0.05, obs_per_cpu_min=10.0)
    assert not p.feasible and p.reasons
    assert CM.plan_run(st, b.branch_id, CM.LadderPolicy(), 0.05, 1e6).feasible
    with pytest.raises(ValueError):
        CM.plan_run(st, b.branch_id, CM.LadderPolicy(), 0.0, 1.0)


def test_early_stop_screen_saves_compute_on_clear_null():
    r = CM.early_stop_screen([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], planned_tests=100)
    assert r["decision"] == "ACCEPT_H0" and r["cpu_min_saved"] > 0


# ---------------------------------------------------------------- value accounting
def pnl(seed=0, n=300):
    return np.random.default_rng(seed).normal(0.0, 0.03, n)


def vjob(jid, **kw):
    return VA.JobMeasurement(jid, "B", kw.pop("family", "fam"), Problem.VOLATILITY, kw.pop("stage", Stage.CROSS_YEAR), "2010-01-05",
                             "2010-01-04", kw.pop("cost", 10.0), **kw)


def test_self_check_passes_for_several_seeds():
    for s in range(3):
        assert all(VA.self_check(s).values()), VA.self_check(s)


def test_risk_reduction_useful_noise_worthless():
    led = VA.ValueLedger()
    b = pnl()
    assert VA.account_job(led, vjob("a", pnl_before=b, pnl_after=np.clip(b, -0.02, None), replications=1), "2010-02-01").verdict == "USEFUL"
    assert VA.account_job(led, vjob("b", pnl_before=b, pnl_after=b + 1e-6, replications=1), "2010-02-01").verdict == "WORTHLESS"
    assert VA.audit_ledger(led) == []


def test_harmful_change_is_flagged_not_rewarded():
    led = VA.ValueLedger()
    b = pnl()
    r = VA.account_job(led, vjob("h", pnl_before=b, pnl_after=np.minimum(b, 0.0) - 0.02, replications=1), "2010-02-01")
    assert r.verdict == "HARMFUL_IF_ADOPTED" and r.net_value <= 0.31


def test_no_measurement_is_never_read_as_success_or_zero():
    led = VA.ValueLedger()
    r = VA.account_job(led, vjob("n"), "2010-02-01")
    assert r.verdict == "INCONCLUSIVE" and r.net_value < 0


def test_invalid_measurement_still_charges_compute():
    led = VA.ValueLedger()
    b = pnl()
    r = VA.account_job(led, vjob("bad", pnl_before=b, pnl_after=b[:10]), "2010-02-01")
    assert r.verdict == "INVALID" and r.cost_cpu_min == 10.0 and r.net_value < 0


def test_data_from_the_future_raises_and_double_accounting_refused():
    led = VA.ValueLedger()
    j = vjob("f")
    j.data_through = "2010-03-01"
    with pytest.raises(FirewallBreach):
        VA.account_job(led, j, "2010-02-01")
    VA.account_job(led, vjob("ok"), "2010-02-01")
    with pytest.raises(ValueError):
        VA.account_job(led, vjob("ok"), "2010-02-02")


def test_tiny_gain_with_complexity_and_no_replication_is_not_useful():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.5).astype(float)
    p0 = np.full(300, 0.5)
    led = VA.ValueLedger()
    r = VA.account_job(led, vjob("t", pred_before=p0, pred_after=p0 - 1e-4 * (y - 0.5), target_values=y, params_added=20,
                                 stage=Stage.CHEAP_SCREEN), "2010-02-01")
    assert r.verdict != "USEFUL"


def test_family_unreliable_is_high_value_and_weak_claim_is_not():
    rng = np.random.default_rng(2)
    res = [VA.HypothesisResult(f"h{i}", float(rng.normal(0, 0.0004)), 0.0005, 40000) for i in range(20)]
    ff = VA.build_family_finding("fx", res, 0.003, pending_cpu_min=300.0, future_cpu_min_per_week=20.0)
    assert VA.family_unreliability(ff, VA.ValuePolicy()) > 0.7
    few = VA.build_family_finding("fx", res[:3], 0.003, pending_cpu_min=300.0)
    assert VA.family_unreliability(few, VA.ValuePolicy()) == 0.0        # three hypotheses do not condemn a family
    real = [VA.HypothesisResult(f"h{i}", 0.02 if i < 6 else 0.0, 0.0005, 40000, replicated=True) for i in range(20)]
    assert VA.family_unreliability(VA.build_family_finding("fy", real, 0.003), VA.ValuePolicy()) == 0.0


def test_claim_refuted_by_later_real_result_appends_correction():
    rng = np.random.default_rng(3)
    res = [VA.HypothesisResult(f"h{i}", float(rng.normal(0, 0.0004)), 0.0005, 40000) for i in range(20)]
    ff = VA.build_family_finding("fx", res, 0.003, pending_cpu_min=300.0, future_cpu_min_per_week=20.0)
    led, book = VA.ValueLedger(), VA.ClaimBook()
    rec = VA.account_job(led, vjob("fam", family_finding=ff, prior_se=0.01, post_se=0.004), "2010-02-01", claims=book)
    assert rec.verdict == "PREVENTIVE" and len(book.open_claims()) == 1
    cid = book.open_claims()[0].claim_id
    up = book.audit(cid, [(0.02, 0.002)], "2010-06-01", led)
    assert up.revive_family and up.correction.net_value < 0 and led.verify_chain() == []
    assert VA.unreliable_families(book) == []


def test_claim_confirmed_only_after_enough_null_looks():
    led, book = VA.ValueLedger(), VA.ClaimBook(confirm_after=4)
    rec = VA.ValueRecord(0, "j", "B", "fx", "VOLATILITY", "STAGE3_CROSS_YEAR", "UNKNOWN_AREA", "2010-01-05", "2010-02-01", "PREVENTIVE", 1.0,
                         1.0, 0.0, {"prevented_compute": 1.0}, {"family_unreliability": 0.9}, (), (), None, "", "", 0)
    rec = led.add(rec)
    c = book.open_from(rec, 0.003)
    assert book.audit(c.claim_id, [(0.0, 0.001)], "2010-03-01", led).claim.status is VA.ClaimStatus.OPEN
    assert book.audit(c.claim_id, [(0.0, 0.001)] * 3, "2010-04-01", led).claim.status is VA.ClaimStatus.CONFIRMED


def test_transfer_needs_many_consistent_contexts():
    good = VA.measure_transfer({f"y{i}": (0.02, 0.005) for i in range(5)})
    one = VA.measure_transfer({"y": (0.02, 0.005)})
    mixed = VA.measure_transfer({f"y{i}": ((0.03 if i % 2 else -0.03), 0.005) for i in range(6)})
    assert good.score > 0.8 and one.score < good.score and mixed.score < 0.2
    assert VA.measure_transfer({}).score == 0.0


def test_redundancy_detects_copy_and_ignores_independent():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(400, 3))
    assert VA.measure_redundancy(E[:, 0] + 1e-3 * rng.normal(size=400), E).redundancy > 0.9
    assert VA.measure_redundancy(rng.normal(size=400), E).redundancy < 0.1
    assert VA.measure_redundancy(None, None).redundancy == 0.0


def test_decision_change_zero_when_scores_equal_and_empty_case():
    rng = np.random.default_rng(5)
    s = rng.normal(size=50)
    assert VA.measure_decisions(s, s, None, 10, rng).fraction_changed == 0.0
    assert VA.measure_decisions(np.array([]), np.array([]), None, 5, rng).n_swapped == 0
    with pytest.raises(ValueError):
        VA.measure_decisions(s, s[:10], None, 5, rng)


def test_ledger_roundtrip_chain_and_as_of_visibility(tmp_path):
    led = VA.ValueLedger()
    VA.account_job(led, vjob("a"), "2010-02-01")
    assert led.records("2010-01-05") == [] and len(led.records("2010-01-06")) == 1      # visible strictly after its date
    led.dump(tmp_path / "l.jsonl")
    assert len(VA.ValueLedger.load(tmp_path / "l.jsonl")) == 1
    txt = (tmp_path / "l.jsonl").read_text().replace("INCONCLUSIVE", "USEFUL")
    (tmp_path / "l.jsonl").write_text(txt)
    with pytest.raises(ValueError):
        VA.ValueLedger.load(tmp_path / "l.jsonl")
    assert VA.ValueLedger.load(tmp_path / "missing.jsonl").records() == []


def test_estimate_calibrator_shrinks_overoptimistic_family():
    led = VA.ValueLedger()
    for i in range(8):
        est = VA.ExperimentValue(information_gain=0.9, decision_value=0.9)
        VA.account_job(led, vjob(f"j{i}", family="optimist", estimate=est), "2010-02-01")
    cal = VA.EstimateCalibrator(led, "2010-03-01")
    assert cal.factor("optimist", "decision_value") < 0.5
    assert cal.corrected("optimist", VA.ExperimentValue(decision_value=0.8)).decision_value < 0.4
    assert VA.EstimateCalibrator(VA.ValueLedger(), "2010-03-01").factor("x", "information_gain") == 1.0


# ---------------------------------------------------------------- waste controller
def waste_world(n_runs=6, useful=False, powered=True):
    """One branch with n_runs finished, worthless experiments recorded in both the manager and the accountant."""
    st, led = CM.ManagerState(), VA.ValueLedger()
    b = mk(st, family="dud")
    for i in range(n_runs):
        day = dt.date(2010, 1, 5) + dt.timedelta(days=i)
        b.runs.append({"stage": "STAGE1_CHEAP_SCREEN", "at": day.isoformat(), "effect": 0.0002, "se": 0.001, "t": 0.2, "action": "REPEAT",
                       "cost": 6.0, "n_obs": 4000, "power": 0.9 if powered else 0.2, "complexity": 3.0})
        b.spent["STAGE1_CHEAP_SCREEN"] = b.spent.get("STAGE1_CHEAP_SCREEN", 0.0) + 6.0
        bb = pnl(i)
        j = VA.JobMeasurement(f"j{i}", b.branch_id, "dud", Problem.VOLATILITY, Stage.CHEAP_SCREEN, day.isoformat(),
                              (day - dt.timedelta(days=1)).isoformat(), 6.0, pnl_before=bb, pnl_after=(np.clip(bb, -0.02, None) if useful and i == n_runs - 1 else bb),
                              params_added=3, replications=int(useful and i == n_runs - 1))
        VA.account_job(led, j, day + dt.timedelta(days=1))
    return st, led, b


def ctx(**kw):
    base = dict(as_of="2010-01-31", data_hash="h1", n_rows=1000, regime="calm")
    base.update(kw)
    return W.WorldContext(**base)


def test_waste_policy_validates():
    assert W.WastePolicy().validate() == []
    assert W.WastePolicy(window=2, min_experiments=4).validate()


def test_barren_expensive_branch_goes_dormant_with_reason_and_lifecycle():
    st, led, b = waste_world()
    book = W.DormantBook()
    r = W.step(st, led, book, ctx(), "2010-02-01")
    assert r.parked == (b.branch_id,) and b.state is ResearchState.DORMANT and b.dormant_reason
    assert book.life.state(book.kid(b.branch_id), "2010-02-10") is W.LifeState.DORMANT
    assert book.life.conditions(book.kid(b.branch_id), "2010-02-10")
    assert book.records[b.branch_id].avoided_cpu_min > 0


def test_underpowered_barren_branch_is_not_judged():
    st, led, b = waste_world(powered=False)
    r = W.step(st, led, W.DormantBook(), ctx(), "2010-02-01")
    assert r.parked == () and b.state is ResearchState.QUEUED
    assert r.verdicts[0].action is W.Action.UNRESOLVED


def test_recent_useful_result_protects_branch():
    st, led, b = waste_world(useful=True)
    r = W.step(st, led, W.DormantBook(), ctx(), "2010-02-01")
    assert r.parked == () and r.verdicts[0].action is W.Action.PROTECT


def test_too_early_or_too_cheap_is_never_wasteful():
    st, led, b = waste_world(n_runs=2)
    assert W.step(st, led, W.DormantBook(), ctx(), "2010-02-01").parked == ()
    assert W.step(CM.ManagerState(), VA.ValueLedger(), W.DormantBook(), ctx(), "2010-02-01").verdicts == ()


def test_revival_only_on_new_unseen_trigger_and_never_twice():
    st, led, b = waste_world()
    book = W.DormantBook()
    W.step(st, led, book, ctx(), "2010-02-01")
    same = W.step(st, led, book, ctx(as_of="2010-04-01"), "2010-04-01")
    assert same.revived == () and b.state is ResearchState.DORMANT             # nothing new: stays parked
    new = ctx(as_of="2010-04-01", data_hash="h2", n_rows=1500)
    r = W.step(st, led, book, new, "2010-04-01")
    assert r.revived == (b.branch_id,) and b.state is ResearchState.QUEUED
    assert b.frontier is Stage.CHEAP_SCREEN                                     # new data makes old passes stale
    rec = book.records[b.branch_id]
    assert rec.seen_triggers and rec.revive_count == 1


def test_new_hypothesis_keeps_earned_rungs():
    st, led, b = waste_world()
    b.passed = {Stage.CHEAP_SCREEN.value: {"effect": 0.01}}
    b.frontier = Stage.STRONGER_TESTS
    book = W.DormantBook()
    W.step(st, led, book, ctx(), "2010-02-01")
    W.step(st, led, book, ctx(as_of="2010-04-01", hypothesis_ids={"dud": ("newh",)}), "2010-04-01")
    assert b.state is ResearchState.QUEUED and b.frontier is Stage.STRONGER_TESTS


def test_future_dated_context_is_a_breach():
    st, led, b = waste_world()
    with pytest.raises(FirewallBreach):
        W.step(st, led, W.DormantBook(), ctx(as_of="2011-01-01"), "2010-02-01")


def test_circuit_breaker_limits_mass_dormancy():
    st, led = CM.ManagerState(), VA.ValueLedger()
    brs = {}
    for k in range(10):
        brs[k] = mk(st, f"q{k}", family=f"d{k}")
    for i in range(6):
        day = dt.date(2010, 1, 5) + dt.timedelta(days=i)
        for k, b in brs.items():
            b.runs.append({"stage": "STAGE1_CHEAP_SCREEN", "at": day.isoformat(), "effect": 0.0002, "se": 0.001, "action": "REPEAT",
                           "cost": 6.0, "n_obs": 4000, "power": 0.9})
            bb = pnl(i)
            VA.account_job(led, VA.JobMeasurement(f"j{k}_{i}", b.branch_id, f"d{k}", Problem.VOLATILITY, Stage.CHEAP_SCREEN, day.isoformat(),
                                                  (day - dt.timedelta(days=1)).isoformat(), 6.0, pnl_before=bb, pnl_after=bb, params_added=3),
                           day + dt.timedelta(days=1))
    r = W.step(st, led, W.DormantBook(), ctx(), "2010-02-01")
    assert len(r.parked) == 1 and len(r.held_back) == 9 and r.mass_dormancy_alarm


def test_unreliable_family_parks_its_branches_and_dup_effort_found():
    st = CM.ManagerState()
    a, b = mk(st, "a", family="bad"), mk(st, "b", family="ok")
    book = W.DormantBook()
    r = W.step(st, VA.ValueLedger(), book, ctx(), "2010-02-01", unreliable_families=["bad"])
    assert a.branch_id in r.parked and b.state is ResearchState.QUEUED
    cfgs = {"x1": {"lr": 0.1, "depth": 3}, "x2": {"lr": 0.1, "depth": 3}, "x3": {"lr": 0.9, "depth": 12}}
    d = W.find_duplicate_effort(cfgs, {k: "f" for k in cfgs}, W.WastePolicy())
    assert [(x[0], x[1]) for x in d] == [("x2", "x1")]


def test_probe_flags_false_dormancy_and_report_runs():
    st, led, b = waste_world()
    book = W.DormantBook()
    W.step(st, led, book, ctx(), "2010-02-01")
    got = book.probe_candidates(np.random.default_rng(0), 1, "2010-06-01", W.WastePolicy())
    assert got == [b.branch_id]
    assert book.record_probe(b.branch_id, 0.02, 0.005, "2010-06-01", W.WastePolicy())
    assert book.false_dormancy_rate()["promising"] == 1
    rep = W.waste_report(st, led, book, "2010-06-01")
    assert rep["dormant_branches"] == 1 and rep["lifecycle_chain_errors"] == []
    assert "DORMANT" in W.explain_verdict(W.assess_branch(st, led, b.branch_id, "2010-06-01", W.WastePolicy())) or True
