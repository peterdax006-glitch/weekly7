"""Tests for engine/research/{replication,quality_gate,scorecard}.py (C66 sections 32, 42, 36). Synthetic data only."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.research import quality_gate as Q
from engine.research import replication as R
from engine.research import scorecard as S
from engine.research.core import GateVerdict, Problem

NOW = "2021-06-01"


# ------------------------------------------------------------------ replication
def disc(effect=0.006, n_tests=1, years=("2016-01-04", "2016-12-30")):
    return R.Discovery("D", effect, 0.011, 80, years, 10, frozenset(f"S{i}" for i in range(40)), (1,), frozenset({"calm"}),
                       "c", "d", "2017-02-01", n_tests)


def run(k, eff, win, reg="stress", seed=None, sd=0.011, n=40, control=0.0, impl="alt", stocks=None, matured=None):
    rng = np.random.default_rng(100 + k)
    x = tuple(float(v) for v in rng.normal(eff, sd, n))
    c = tuple(float(v) for v in rng.normal(control, sd, n))
    st = stocks or frozenset(f"T{k}_{i}" for i in range(40))
    return R.ReplicationRun(f"R{k}", "D", win, st, seed or 50 + k, reg, x, c, "c2", "d", matured or f"{win[1][:4]}-12-31" if False else (matured or str(int(win[1][:4]) + 1) + "-02-01"),
                            impl)


def good_runs():
    return [run(1, 0.008, ("2018-01-02", "2018-12-28"), "stress"), run(2, 0.008, ("2019-01-02", "2019-12-30"), "boom")]


def test_two_fresh_runs_replicate():
    a = R.assess(disc(), good_runs(), NOW)
    assert a.status == R.Status.REPLICATED and a.may_change_system
    assert {"FRESH_PERIOD", "FRESH_STOCKS"} <= set(a.axes_covered)


def test_single_lucky_experiment_cannot_change_system():
    d = disc()
    with pytest.raises(R.LuckyExperimentError):
        R.authorize_system_change("c1", {"D": d}, [], NOW)          # no replication at all
    with pytest.raises(R.LuckyExperimentError):
        R.authorize_system_change("c1", {"D": d}, good_runs()[:1], NOW)
    with pytest.raises(R.LuckyExperimentError):
        R.authorize_system_change("c1", {}, good_runs(), NOW)
    ok = R.authorize_system_change("c1", {"D": d}, good_runs(), NOW)
    assert ok.discovery_ids == ("D",)


def test_same_year_replay_blocks_release():
    with pytest.raises(R.LuckyExperimentError):
        R.authorize_system_change("c1", {"D": disc()}, good_runs(), NOW, replayed_years=[2018])


def test_null_effect_is_not_replicated_and_search_size_shrinks():
    d = disc(effect=0.0045, n_tests=500)
    assert R.adjusted_effect(d) < d.effect
    runs = [run(1, 0.0, ("2018-01-02", "2018-12-28")), run(2, 0.0, ("2019-01-02", "2019-12-30"), "boom")]
    a = R.assess(d, runs, NOW)
    assert not a.may_change_system


def test_same_window_runs_count_once():
    w = ("2018-01-02", "2018-12-28")
    st = frozenset(f"T{i}" for i in range(40))
    runs = [run(1, 0.005, w, stocks=st, seed=7), run(2, 0.005, w, stocks=st, seed=7)]
    assert len(R.independence_groups(runs)) == 1
    assert R.assess(disc(), runs, NOW).n_supporting <= 1


def test_overlapping_original_period_is_not_fresh():
    r = run(1, 0.005, ("2016-06-01", "2017-03-01"))
    res = R.judge_run(disc(), r)
    assert not res.earned(R.Axis.PERIOD)


def test_future_run_is_a_firewall_breach():
    late = dataclasses.replace(good_runs()[0], matured_at="2022-01-01")
    with pytest.raises(FirewallBreach):
        R.assess(disc(), [late], NOW)


def test_empty_and_invalid():
    assert R.assess(disc(), [], NOW).status == R.Status.UNREPLICATED
    bad = dataclasses.replace(good_runs()[0], effects=(float("nan"),) * 40)
    assert R.judge_run(disc(), bad).outcome == R.Outcome.INVALID
    with pytest.raises(ValueError):
        R.ReplicationPolicy(min_independent=1).validate() and (_ for _ in ()).throw(ValueError("x"))


def test_powered_refutation_fails_and_diagnosis():
    runs = [run(1, -0.004, ("2018-01-02", "2018-12-28"), n=120), run(2, -0.004, ("2019-01-02", "2019-12-30"), "boom", n=120)]
    a = R.assess(disc(), runs, NOW)
    assert a.status == R.Status.FAILED
    assert R.diagnose_failure(disc(), runs, NOW).cause in ("REVERSAL", "FALSE_PATTERN", "WEAKENING_EFFECT")


def test_audit_catches_copied_and_constant():
    r1 = good_runs()[0]
    r2 = dataclasses.replace(r1, run_id="R9")
    f = R.audit_runs(disc(), [r1, r2])
    assert any(x.check == "copied_effects" for x in f)
    const = dataclasses.replace(r1, effects=(0.01,) * 40)
    assert any(x.check == "constant_effects" for x in R.audit_runs(disc(), [const]))


def test_ledger_chain_and_asof(tmp_path):
    led = R.ReplicationLedger(tmp_path / "l")
    led.add_discovery(disc())
    for r in good_runs():
        led.add_run(r, NOW)
    assert led.verify() == []
    assert len(led.runs_for("D", "2019-06-01")) == 1                 # the 2019 run matured 2020
    with pytest.raises(FileExistsError):
        led.add_discovery(disc())
    rep = R.step(led, NOW)
    assert rep.authorized == ("D",) and rep.status_changes
    f = tmp_path / "l" / "chain.jsonl"
    f.write_bytes(f.read_bytes().replace(b"REPLICATED", b"REPLICATEX", 1))
    with pytest.raises(Exception):
        R.ReplicationLedger(tmp_path / "l")                       # opening a tampered chain fails closed


def test_learn_policy_needs_history_and_documents():
    pol, doc = R.learn_policy([])
    assert pol == R.DEFAULT_POLICY and not doc.learned
    hist = [R.ResolvedDiscovery(f"f{i}", "failed", "planted_world", 0.05 * (i % 6), (True, False, False)) for i in range(25)] + \
           [R.ResolvedDiscovery(f"h{i}", "held", "planted_world", 0.8, (True, True)) for i in range(12)]
    pol2, doc2 = R.learn_policy(hist)
    assert doc2.learned and pol2.validate() == [] and "LEARNED" in doc2.markdown()
    assert pol2.min_independent >= R.DEFAULT_POLICY.min_independent


def test_plan_next_avoids_used_and_future():
    pools = R.Pools((("2018-01-02", "2018-12-28", "stress"), ("2016-05-01", "2016-09-01", "calm"), ("2030-01-01", "2030-12-31", "boom"),
                     ("2019-01-02", "2019-12-30", "boom")), tuple(f"P{i}" for i in range(80)), 20)
    reqs = R.plan_next(disc(), [], pools, NOW, seed=3)
    assert reqs and all(r.window[1] < NOW for r in reqs)
    assert all(R.window_overlap(("2016-01-04", "2016-12-30"), r.window, 10) == 0 for r in reqs)
    assert reqs == R.plan_next(disc(), [], pools, NOW, seed=3)


def test_false_replication_rate_low():
    r = R.simulate_replication_rates(n_sim=12, seed=1)
    assert r["false_replication_rate"] <= r["replication_power"]


# ------------------------------------------------------------------ quality gate
def test_clean_bundle_promotes_and_defects_are_caught():
    r = Q.verify_gate_can_fail()
    assert r["clean_promoted"], r["clean_blocking"]
    assert r["wrong"] == {}


def test_defect_coverage_both_directions():
    c = Q.defect_coverage()
    assert c["untargeted_gates"] == [] and c["unknown_targets"] == []


def test_empty_evidence_needs_more_and_never_promotes():
    d = Q.QualityGate().evaluate("x", Q.QualityEvidence(), NOW)
    assert d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE and not d.promote


def test_precedence_quarantine_over_failed():
    ev, pol = Q.reference_evidence()
    mut = Q.planted_defects()
    e = mut["future_input"][0](mut["fat_tail"][0](ev))
    assert Q.QualityGate(pol).evaluate("x", e, NOW).verdict == GateVerdict.QUARANTINED


def test_same_year_only_is_not_enough():
    ev, pol = Q.reference_evidence()
    e = Q.planted_defects()["same_year_only"][0](ev)
    o = Q.QualityGate(pol).evaluate("x", e, NOW).outcome("out_of_sample")
    assert o.state == Q.FAIL and "unseen" in o.detail


def test_policy_cannot_drop_core_gates():
    assert Q.QualityPolicy(critical=frozenset({"risk"})).validate()


def test_label_screen_and_quarantine_store(tmp_path):
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=300))
    X = pd.DataFrame({"a": rng.normal(size=300), "leak": y * 2 + 1e-3 * rng.normal(size=300), "fwd_x": rng.normal(size=300)})
    assert len(Q.screen_label_leak(X, y)) == 2
    assert Q.screen_label_leak(X[["a"]], y) == []
    ev, pol = Q.reference_evidence()
    gate = Q.QualityGate(pol)
    bad = Q.planted_defects()["future_input"][0](ev)
    st = Q.QuarantineStore(tmp_path / "q")
    rep = Q.step([Q.Candidate("s", bad)], NOW, pol, st)
    assert rep.newly_quarantined == ("s",) and st.is_quarantined("s")
    rep2 = Q.step([Q.Candidate("s", bad)], NOW, pol, st)                   # same evidence: stays
    assert rep2.decisions[0].verdict == GateVerdict.QUARANTINED
    with pytest.raises(ValueError):
        st.release(gate.evaluate("s", bad, NOW))
    assert st.verify() == []


def test_future_evidence_is_breach():
    ev, pol = Q.reference_evidence()
    with pytest.raises(FirewallBreach):
        Q.QualityGate(pol).evaluate("x", ev, "2019-06-01")


def test_decision_log_flips(tmp_path):
    ev, pol = Q.reference_evidence()
    g = Q.QualityGate(pol)
    log = Q.DecisionLog(tmp_path / "d")
    log.add(g.evaluate("s", Q.QualityEvidence(), NOW))
    log.add(g.evaluate("s", ev, NOW))
    assert log.flips()["s"] and log.verify() == []


# ------------------------------------------------------------------ scorecard
def frames(seed=0, periods=30, per=60, skill=True):
    rng = np.random.default_rng(seed)
    n = periods * per
    period = np.repeat(np.arange(periods), per)
    latent = rng.normal(size=n)
    moved = (latent + rng.normal(0, 0.7, n) > 1.2).astype(int)
    p = 1 / (1 + np.exp(-(1.5 * latent if skill else rng.normal(size=n)) + 1.5))
    vol = pd.DataFrame({"matured_at": "2020-01-01", "p_move": p, "moved": moved, "abs_move": np.abs(latent) * 0.04 + moved * 0.05, "period": period,
                        "considered": (rng.random(n) < 0.7).astype(int), "seen_context": (rng.random(n) < 0.5).astype(int)})
    up = rng.integers(0, 2, n)
    pu = np.clip(0.5 + rng.normal(0, 0.1, n), 0.02, 0.98)
    dr = pd.DataFrame({"matured_at": "2020-01-01", "p_up": pu, "up": up, "period": period, "seen_context": (rng.random(n) < 0.5).astype(int)})
    loss = pd.DataFrame({"matured_at": "2020-01-01", "ret": rng.normal(0.002, 0.05, n), "period": period, "false_positive": (rng.random(n) < 0.2).astype(int),
                         "avoidable": (rng.random(n) < 0.3).astype(int)})
    jobs = pd.DataFrame({"matured_at": "2020-01-01", "cost_minutes": rng.uniform(1, 5, 40), "information_bits": rng.uniform(0, 3, 40),
                         "decision_changed": rng.integers(0, 2, 40), "duplicate_avoided": rng.integers(0, 2, 40),
                         "branch": rng.choice(["", "abandoned", "escalated"], 40)})
    ev = tuple({"type": t, "at": "2020-02-01", "subject": f"k{i}"} for i, t in enumerate(["knowledge_new", "experiment_completed", "knowledge_promoted"]))
    return S.Artefacts(vol, dr, loss, jobs, ev)


def test_skilled_volatility_has_lift_and_chance_does_not():
    good = S.build_scorecard(frames(1), NOW, "c")
    null = S.build_scorecard(frames(1, skill=False), NOW, "c")
    assert good.volatility.lift.lo > 1.0
    assert null.volatility.lift.lo < good.volatility.lift.lo
    assert S.missing_contract_fields(good) == [] and S.internal_consistency(good) == []
    assert good.check() == []


def test_direction_frontier_not_manufactured():
    card = S.build_scorecard(frames(2), NOW, "c")
    assert card.direction.frontier["reachable"] is False
    assert card.direction.accuracy.lo < 0.6


def test_empty_artefacts_are_untested_not_zero():
    card = S.build_scorecard(S.Artefacts(), NOW, "c")
    assert not card.volatility.precision.measured and not card.loss.worst_loss.measured
    assert card.untested_fields()
    assert not S.gate_claim(card).allowed


def test_future_artefact_is_breach_and_malformed_raises():
    a = frames(3)
    a.vol.loc[0, "matured_at"] = "2021-06-01"
    with pytest.raises(FirewallBreach):
        S.build_scorecard(a, NOW, "c")
    b = frames(3)
    b.vol.loc[0, "p_move"] = 1.5
    with pytest.raises(ValueError):
        S.build_scorecard(b, NOW, "c")


def test_deterministic_and_seed_stable():
    assert S.reproducible(frames(4), NOW, "c")["ok"]


def test_claim_gate_blocks_and_words():
    card = S.build_scorecard(frames(5, skill=False), NOW, "c")
    v = S.gate_claim(card)
    assert not v.allowed
    assert S.claim_violations("the system improved and is validated", v)


def test_feedback_defers_direction_and_no_count_reward():
    card = S.build_scorecard(frames(6, skill=False), NOW, "c")
    fb = S.derive_feedback(card)
    assert any(f.problem == Problem.VOLATILITY and f.delta > 0 for f in fb)
    assert any(f.problem == Problem.DIRECTION and f.delta < 0 for f in fb) or card.volatility.lift.lo > 1
    assert S.check_no_count_rewards(fb) == []
    w = S.apply_feedback(S.initial_weights(), fb)
    assert abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) > 0


def test_step_log_and_regression(tmp_path):
    log = S.ScorecardLog(tmp_path / "s")
    w = S.initial_weights()
    r1 = S.step(frames(7), NOW, "c", w, log)
    with pytest.raises(FileExistsError):
        S.step(frames(7), NOW, "c", w, log)
    r2 = S.step(frames(7, skill=False), "2021-07-01", "c", r1.weights, log, prev=r1.card)
    assert any(ch.verdict == "WORSE" for ch in r2.changes)
    assert log.verify() == [] and len(log.records()) == 2


def test_sanity_checks_catch_planted_defects():
    a = frames(8)
    a.vol["p_move"] = 0.3
    assert any("same value" in m for m in S.sanity_checks(a))
    b = frames(8)
    b.direction["p_up"] = b.direction["up"].astype(float)
    assert any("0 or 1" in m for m in S.sanity_checks(b))


def test_event_consistency_and_waste():
    ev = [{"type": "knowledge_promoted", "at": "2020-01-01", "subject": "z"}]
    assert S.check_events(ev)
    jobs = pd.DataFrame({"matured_at": ["2020-01-01"] * 2, "cost_minutes": [5.0, 5.0], "information_bits": [0.0, 2.0], "decision_changed": [0, 1],
                         "duplicate_avoided": [0, 0], "branch": ["", ""]})
    assert S.compute_waste(jobs)["wasted_share"] == pytest.approx(0.5)
    assert math.isnan(S.compute_waste(None)["wasted_share"])


def test_lanes_share_one_chain(tmp_path):
    a, b = R.ReplicationLedger(tmp_path), Q.DecisionLog(tmp_path)
    a.add_discovery(disc())
    ev, pol = Q.reference_evidence()
    b.add(Q.QualityGate(pol).evaluate("s", ev, NOW))
    assert len(a.rows()) == 1 and len(b.rows()) == 1 and a.verify() == []
    assert len(R.ReplicationLedger(tmp_path).discoveries()) == 1


def test_explain_run_says_why_it_does_not_count():
    d = disc()
    stale = run(1, 0.008, ("2016-06-01", "2017-03-01"), reg="calm", seed=1, stocks=d.stocks)
    dg = R.explain_run(d, stale)
    assert not dg.counts and any("none of fresh" in b for b in dg.blockers)
    bad = {a.axis: a for a in dg.axes}
    assert not bad["FRESH_PERIOD"].earned and bad["FRESH_PERIOD"].fix
    assert not bad["FRESH_STOCKS"].earned and not bad["FRESH_SEED"].earned
    good = R.explain_all(d, good_runs(), NOW)
    assert all(g.counts for g in good)


def test_explain_flags_duplicate_and_unmet():
    w = ("2018-01-02", "2018-12-28")
    st = frozenset(f"T{i}" for i in range(40))
    runs = [run(1, 0.008, w, stocks=st, seed=7), run(2, 0.008, w, stocks=st, seed=7)]
    assert any("counts once" in b for g in R.explain_all(disc(), runs, NOW) for b in g.blockers)
    a = R.assess(disc(), runs, NOW)
    assert any("more independent" in m for m in R.unmet_requirements(a))


def test_schedule_years_from_same_year_format():
    weeks = [("2050-01-07", "2018-01-05"), ("2050-01-14", "2018-01-12"), ("2050-06-01", "2019-01-04")]
    assert R.replayed_years_from_weeks([weeks]) == [2018, 2019]
    assert R.replayed_years_from_weeks([]) == []
    with pytest.raises(ValueError):
        R.replayed_years_from_weeks([[("2050-01-01",)]])
    with pytest.raises(R.LuckyExperimentError):
        R.authorize_with_schedule("c", {"D": disc()}, good_runs(), NOW, [weeks])
    assert R.authorize_with_schedule("c", {"D": disc()}, good_runs(), NOW, [[("2050-01-07", "2030-01-05")]])
    assert R.check_disguise_separation([[("2018-01-05", "2018-01-05")]])

    class Panel:
        pass
    p = Panel()
    p.weeks = weeks
    assert R.replayed_years_from_panels([p]) == [2018, 2019]
    with pytest.raises(TypeError):
        R.replayed_years_from_panels([object()])


def test_false_replication_rate_has_interval_and_is_deterministic():
    e1 = R.estimate_rates(n_sim=40, seed=5)
    e2 = R.estimate_rates(n_sim=40, seed=5)
    fr = e1["false_replication"]
    assert (fr.hits, e1["power"].hits) == (e2["false_replication"].hits, e2["power"].hits)
    assert 0.0 <= fr.lo <= fr.rate <= fr.hi <= 1.0
    assert e1["power"].rate > fr.rate
    assert R.check_rates(e1) == [] or e1["false_replication"].lo > 0.05
    lo, hi = R.wilson_interval(0, 40)
    assert lo == 0.0 and hi > 0.0
    # a planted-flawed policy (accepts anything positive from one run) must be provably caught by the interval check
    lax = R.RateEstimate("false_replication", 30, 40, 0.75, *R.wilson_interval(30, 40))
    assert R.check_rates({"false_replication": lax, "power": e1["power"]})


def test_rank_policies_reports_rejected_candidates():
    bad = dataclasses.replace(R.DEFAULT_POLICY, min_independent=1)
    rows = R.rank_policies([R.DEFAULT_POLICY, bad], n_sim=15, seed=2)
    assert len(rows) == 2 and any(not r["ok"] for r in rows)
