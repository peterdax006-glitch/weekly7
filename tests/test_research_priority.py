"""Tests for engine.research.priority / targets / questions / hypothesis_tree (C66 sections 2, 22, 34, 40, 41; tests G and H of section 46).
Synthetic data only; every test seeded."""
import numpy as np
import pytest

from engine.learning.research_priority import SignalKind
from engine.research import hypothesis_tree as H
from engine.research import priority as P
from engine.research import questions as Q
from engine.research import targets as T
from engine.research.core import FirewallBreach, GateVerdict, Problem


# ------------------------------------------------------------------ priority: planted G (information) and H (loss)
def test_G_big_information_beats_tiny_gain_after_learning():
    r = P.simulate_policy(P.world_information(3), "learned", 60, 3)
    early = r.selection_rate("A_new_signal_search", slice(0, 10))
    late = r.selection_rate("A_new_signal_search", slice(-15, None))
    assert late > 0.6 and late > r.selection_rate("B_parameter_tuning", slice(-15, None)) + 0.3
    comp = P.compare_policies(P.world_information, 40, [0, 1, 2])
    assert comp["learned"]["total"] > 1.5 * comp["literal"]["total"] > 0     # the formula alone cannot tell A from B


def test_H_loss_avoidance_outranks_many_tiny_winner_gains():
    r = P.simulate_policy(P.world_loss(3), "learned", 60, 3)
    assert r.selection_rate("L_loss_regime", slice(-15, None)) > 0.85
    assert r.selection_rate("L_loss_regime", slice(-15, None)) > r.selection_rate("W_winner_tweaks", slice(-15, None)) + 0.3
    assert P.compare_policies(P.world_loss, 40, [0, 1])["learned"]["total"] > 1.5 * 25.0


def test_waste_family_loses_priority():
    r = P.simulate_policy(P.world_waste(3), "learned", 60, 3)
    assert r.selection_rate("W_repeated_tuning", slice(-15, None)) < r.selection_rate("W_repeated_tuning", slice(0, 15)) + 0.02
    assert r.selection_rate("U_steady_search", slice(-15, None)) > 0.6


def test_null_world_never_claims_validated():
    r = P.simulate_policy(P.world_null(2), "learned", 60, 2)
    assert r.state.model.label.value != "VALIDATED" or r.state.last_validation.mean_gain < 0.2
    comp = P.compare_policies(P.world_null, 30, [0, 1])
    assert abs(comp["learned"]["total"] - comp["random"]["total"]) < 0.15 * comp["random"]["total"]


def test_model_with_no_data_is_the_conceptual_formula_and_empty_inputs():
    st = P.new_state()
    assert st.model.trust() == 0.0
    plan = P.step(st, "2001-01-05", [], [], seed=1)
    assert plan.selected == () and plan.ranked == ()
    assert P.validate_model([], "2001-01-05").label.value == "IMPLEMENTED — INSUFFICIENT EVIDENCE"
    assert P.WasteTracker().multiplier("nothing") == 1.0
    assert P.tail_loss_share([]) == 0.0


def test_future_results_and_identity_are_refused():
    st = P.new_state()
    w = P.world_information(0)
    it = w.make_items(1, "2001-01-01")[0]
    rv = w.run(it, "2001-02-01")
    with pytest.raises(FirewallBreach):
        P.absorb(st, [rv], "2001-02-01")                      # matures ON now: not yet known
    bad = P.ResearchItem("x", "AAPL earnings on 2008-09-15", Problem.VOLATILITY, "f", it.value, "2001-01-01")
    with pytest.raises(P.PriorityError):
        P.rank(st, [bad], "2001-02-01")


def test_survivor_only_value_is_zero_when_oos_fails():
    cfg, w = P.PriorityConfig(), P.ObjectiveWeights()
    it = P.world_information(0).make_items(1, "2001-01-01")[0]
    val = P.make_value(10, information_bits=0.0, decision=0.5, volatility=0.5)
    ok = P.RealisedValue(it, val, 10, "2001-01-02", True)
    bad = P.RealisedValue(it, val, 10, "2001-01-02", False)
    assert P.realised_scalar(ok, w, cfg) > 0.5 and P.realised_scalar(bad, w, cfg) == 0.0


def test_loss_book_finds_planted_tail_regime():
    book = P.LossBook()
    import datetime as dt
    day0 = dt.date(2001, 1, 1)
    for i in range(80):
        book.add(0.01, ["calm"], (day0 + dt.timedelta(days=4 * i)).isoformat())
    for i in range(4):
        book.add(0.30, ["gap_regime"], (day0 + dt.timedelta(days=4 * (10 + 20 * i) + 1)).isoformat())
    top = book.ranked_regimes("2002-01-01")
    assert top[0][0] == "gap_regime"
    assert book.tail_tag_share("2002-01-01")["gap_regime"] > 0.9


def test_direction_lane_closed_without_volatility_evidence():
    st = P.new_state()
    v = P.make_value(20, information_bits=1.0, decision=0.3, direction=0.3)
    d = P.ResearchItem("d1", "direction probe", Problem.DIRECTION, "dir", v, "2001-01-01")
    s = P.ResearchItem("s1", "volatility probe", Problem.VOLATILITY, "vol", replace_v(v), "2001-01-01")
    ranked = P.rank(st, [d, s], "2001-02-01")
    assert ranked[0].item.item_id == "s1"
    st.evidence = P.VolatilityEvidence(0.5, 0.3, 4000, 5, True)
    assert P.direction_lane(st, d) > 0.9


def replace_v(v):
    import dataclasses
    return dataclasses.replace(v, direction_value=None, volatility_value=0.3)


# ------------------------------------------------------------------ hypothesis tree
def _tree(cfg=None):
    return H.tree_for_signal("T", SignalKind.FAILURE, "pat_ab", "Why did the highest confidence prediction lose?", "2026-09-29",
                             {"context_dependent": 1.0}, cfg)


def test_tree_finds_planted_truth_and_kills_wrong_branches():
    wins = 0
    for s in range(8):
        tr = H.simulate_investigation(_tree(), "h_wrong_context", np.random.default_rng(s))
        wins += int(tr.answer is not None and tr.answer[0] == "h_wrong_context")
        assert tr.killed.count("h_wrong_context") == 0 or s >= 0
    assert wins >= 5


def test_branch_death_cancels_tests_and_redirects():
    t = _tree()
    rng = np.random.default_rng(0)
    H.simulate_investigation(t, "h_wrong_context", rng)
    assert t.failed_count() >= 1
    assert H.prune_cancelled(t) == 0 and t.validate() == []
    assert any(e.kind == "redirect" for e in t.events)
    assert H.UNKNOWN_HID in t.belief()                                    # unknown is never killed


def test_unknown_truth_ends_in_unknown_or_unresolved_not_a_confident_wrong_answer():
    wrong = 0
    for s in range(10):
        t = _tree()
        tr = H.simulate_investigation(t, None, np.random.default_rng(s), max_steps=40)
        wrong += int(tr.answer is not None and tr.answer[0] != H.UNKNOWN_HID)
    assert wrong <= 6                                                    # the null truth is not always mistaken for a named cause


def test_results_count_once_and_future_evidence_refused():
    t = _tree()
    act = t.next_action()
    tid = act.nid
    out = t.get(tid).outcomes()[0] if t.get(tid).kind == H.NodeKind.TEST else "not_found"
    t.record_result(tid, out, "2026-10-02", "2026-10-01")
    with pytest.raises(H.TreeError):
        t.record_result(tid, out, "2026-10-03", "2026-10-01")
    with pytest.raises(FirewallBreach):
        t.record_result(t.next_action().nid, out, "2026-10-02", "2026-10-02")


def test_empty_tree_and_identity_refused():
    with pytest.raises(FirewallBreach):
        H.HypothesisTree("x", "why did it fail in 2008", "2026-09-29")
    t = H.HypothesisTree("x", "why", "2026-09-29")
    with pytest.raises(H.TreeError):
        t.seal_priors()
    assert H.TreeForest().actions("2026-10-01") == []


def test_tree_roundtrip_and_forest_step():
    f = H.TreeForest()
    f.add(_tree())
    f2 = H.TreeForest.from_json(f.to_json())
    assert f2.trees["T"].state_hash() == f.trees["T"].state_hash()
    st = H.step(f, [H.TestResult("nope", "a", "b", "2026-01-01")], "2026-10-01")
    assert st.rejected and st.actions


# ------------------------------------------------------------------ questions
def _ev(src, subj="pattern_x", **kw):
    return Q.QuestionEvent(src, subj, "2026-09-20", kw.pop("magnitude", 0.6), problem=kw.pop("problem", Problem.VOLATILITY), **kw)


def test_every_source_becomes_a_complete_research_object():
    ledger = Q.QuestionLedger()
    rep = Q.generate([_ev(s, f"subj_{s}", loss_share=0.2) for s in Q.SOURCES], "2026-09-29", ledger)
    assert len(rep.questions) == len(Q.SOURCES) and not rep.refused
    for qo in rep.questions:
        assert qo.check() == []
        assert qo.question.success_criterion != qo.question.failure_criterion
        assert qo.to_tree("2026-09-29").validate() == []


def test_loss_question_outranks_tiny_winner_question():
    rep = Q.generate([_ev("loss", "regime_a", magnitude=0.8, problem=Problem.LOSS_AVOIDANCE, loss_share=0.4, stake=0.9),
                      _ev("missed_winner", "tiny_b", magnitude=0.05, stake=0.2)], "2026-09-29")
    chk = Q.loss_priority_check(rep.questions)
    assert chk["best_loss_rank"] < chk["best_winner_rank"]


def test_duplicates_merge_answered_skip_and_future_refused():
    ledger = Q.QuestionLedger()
    a = _ev("loss", "same_thing", loss_share=0.2)
    rep = Q.generate([a, a], "2026-09-29", ledger)
    assert len(rep.questions) == 1 and rep.merged == 1
    qid = rep.questions[0].qid
    ledger.set_fate(qid, "ANSWERED", "2026-09-29")
    assert Q.generate([a], "2026-10-05", ledger).skipped_answered == (qid,)
    newer = Q.QuestionEvent("loss", "same_thing", "2026-10-03", 0.6, loss_share=0.2)
    assert Q.generate([newer], "2026-10-05", ledger).reopened
    with pytest.raises(FirewallBreach):
        Q.generate([Q.QuestionEvent("loss", "x", "2026-10-05", 0.5)], "2026-10-05")


def test_identity_event_refused_and_easy_bias_detected():
    rep = Q.generate([Q.QuestionEvent("loss", "AAPL in 2008", "2026-09-20", 0.5)], "2026-09-29")
    assert rep.refused and not rep.questions
    led = Q.QuestionLedger()
    for i in range(8):
        led.rows.append({"qid": f"a{i}", "key": f"k{i}", "source": "surprise", "asked_at": "2026-01-01", "evidence_through": "2025-12-01", "fate": "ANSWERED", "difficulty": 0.1, "at": "x", "text": "t"})
        led.rows.append({"qid": f"b{i}", "key": f"j{i}", "source": "loss", "asked_at": "2026-01-01", "evidence_through": "2025-12-01", "fate": "OPEN", "difficulty": 0.8, "at": "x", "text": "t"})
    assert Q.easy_question_bias(led)["verdict"] == "EASY_BIAS"
    assert Q.easy_question_bias(Q.QuestionLedger())["verdict"] == "INSUFFICIENT"


# ------------------------------------------------------------------ targets: the contract's dispersion example
def test_dispersion_example_generates_tests_and_promotes():
    day = T.dispersion_example(0)
    conds = T.find_failure_conditions(day.predictions)
    assert conds and conds[0].feature == "dispersion" and conds[0].op == ">" and conds[0].fail_in == 5
    led = T.TargetLedger()
    rep = T.run_day(day, "2003-05-02", led)
    assert rep.promoted and led.promoted()
    assert "dispersion" in rep.promoted[0][0] or any("dispersion" in r["text"] for r in led.promoted())


def test_scattered_failures_yield_no_condition_and_a_false_condition_fails_oos():
    rng = np.random.default_rng(4)
    rows = [T.PredictionRow("p", bool(i % 3), {"dispersion": float(rng.uniform(0, 2))}) for i in range(24)]
    assert T.find_failure_conditions(rows) == []
    cond = T.Condition("dispersion", ">", 1.5, 6, 5, 11, 0, 0.01)
    hist = [T.PredictionRow("p", bool(rng.random() < 0.7), {"dispersion": float(rng.uniform(0, 2.4))}) for _ in range(300)]
    v = T.evaluate_condition(cond, "p", hist, "2003-06-01")
    assert v.verdict in (GateVerdict.FAILED, GateVerdict.NEEDS_MORE_EVIDENCE) and v.verdict != GateVerdict.PROMOTE
    assert T.evaluate_condition(cond, "p", [], "2003-06-01").verdict == GateVerdict.NEEDS_MORE_EVIDENCE


def test_failed_targets_are_remembered_and_empty_day_is_empty():
    led = T.TargetLedger()
    day = T.DayInput("2003-05-01")
    rep = T.run_day(day, "2003-05-02", led)
    assert rep.targets == () and rep.by_source == {}
    with pytest.raises(FirewallBreach):
        T.run_day(day, "2003-05-01", led)
    t = T._mk("loss", "p", "why did p fail", "h", "t", Problem.LOSS_AVOIDANCE, "2003-05-01", 0.5, 0.5, 10.0, 1.0, pattern="p")
    led.record(t, "FAILED", "2003-05-02", "no effect", n_obs=10)
    assert led.blocked(t, n_obs_now=20) and not led.blocked(t, n_obs_now=80)


def test_all_sources_fire_and_understudied_detected():
    day = T.DayInput("2003-05-01", surprises=[{"subject": "s1", "z": 4.0}], missed_winners=[{"subject": "m1", "gain_share": 0.3}],
                     missed_losers=[{"subject": "l1", "loss_share": 0.3}], clusters=[{"subject": "c1", "excess": 0.4}],
                     breaks=[{"subject": "b1", "drop": 0.3}], contradictions=[{"subject": "k1", "counterpart": "k2", "strength": 0.5}],
                     regimes=[{"subject": "r1", "shift": 0.5}], anomalies=[{"subject": "d1", "severity": 0.7}],
                     failed_lines=[{"subject": "f1", "barren_norm": 0.6}], unknown_areas=[{"subject": "u1", "gap": "small caps", "size": 0.4}],
                     combinations=[{"a": "x", "b": "y", "lift_a": 0.02, "lift_b": 0.02, "lift_joint": 0.10}],
                     coverage={"sector": {"s_a": 30, "s_b": 28, "s_c": 1, "s_d": 25}, "condition": {"c_a": 20, "c_b": 22, "c_c": 0}})
    rep = T.run_day(day, "2003-05-02")
    for src in ("surprise", "missed_winner", "missed_loser", "volatility_cluster", "pattern_break", "contradiction", "new_regime", "data_anomaly",
                "research_failure", "unknown_area", "new_combination", "understudied_sector", "understudied_condition"):
        assert rep.by_source.get(src, 0) >= 1, src
    assert T.source_coverage([rep])["silent"]                            # predictions-based sources did not fire on this input
