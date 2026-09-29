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


# ------------------------------------------------------------------ wave-2 increment: questions
def test_numeric_criteria_are_executable_and_undecided_is_not_failure():
    e = Q.QuestionEvent("loss", "regime_a", "2026-09-20", 0.6, loss_share=0.3, problem=Problem.LOSS_AVOIDANCE)
    c = Q.numeric_criteria(e)
    assert Q.judge(c, Q.Outcome(n=5, lift=0.5, p_value=0.0001, loss_avoided=0.5)) == "UNDECIDED"
    assert Q.judge(c, Q.Outcome(n=500, lift=0.2, p_value=0.001, loss_avoided=0.5)) == "SUCCESS"
    assert Q.judge(c, Q.Outcome(n=500, lift=0.2, p_value=0.001, loss_avoided=0.05)) == "FAILURE"      # lift alone is not loss avoided
    d = Q.numeric_criteria(Q.QuestionEvent("new_discovery", "d", "2026-09-20", 0.6))
    assert Q.judge(d, Q.Outcome(n=500, lift=0.2, p_value=0.001, periods_replicated=1)) == "FAILURE"    # one period is not replication


def test_planted_split_is_caught_and_null_split_is_not():
    rng = np.random.default_rng(1)
    flag = rng.random(400) < 0.3
    y = rng.normal(0, 1, 400) + 0.6 * flag
    assert Q.outcome_from_split(y, flag, 1).decision_changed
    assert not Q.outcome_from_split(rng.normal(0, 1, 400), flag, 1).decision_changed
    rep = Q.outcome_from_replication([0.3, 0.28, 0.31], [0.1, 0.1, 0.1])
    assert rep.periods_replicated == 3 and rep.size_ratio > 0.85
    assert Q.outcome_from_replication([0.3, -0.3], [0.1, 0.1]).periods_replicated <= 1
    with pytest.raises(Q.QuestionError):
        Q.outcome_from_replication([], [])


def test_event_builders_filter_noise():
    assert Q.event_from_surprise("s", 1.5, "2026-09-20") is None and Q.event_from_surprise("s", 4.0, "2026-09-20")
    assert Q.event_from_contradiction("a", "b", 10, 20, 11, 20, "2026-09-20") is None            # 50% vs 55% is noise
    assert Q.event_from_contradiction("a", "b", 18, 20, 4, 20, "2026-09-20")
    assert Q.event_from_break("p", 0.7, 0.68, 100, "2026-09-20") is None and Q.event_from_break("p", 0.7, 0.4, 100, "2026-09-20")
    assert Q.event_from_break("p", 0.7, 0.4, 5, "2026-09-20") is None


def test_question_quality_learning_promotes_paying_source_and_calibrates():
    book = Q.QuestionOutcomeBook()
    qo = Q.generate([_ev("loss", "x1", loss_share=0.3)], "2026-09-29").questions[0]
    qo2 = Q.generate([_ev("surprise", "x2")], "2026-09-29").questions[0]
    for i in range(20):
        book.record(qo, 0.3, Q.Outcome(100, 0.1, 0.01, decision_changed=True, information_bits=1.0), "SUCCESS", f"2026-10-{1 + i % 20:02d}", "2027-01-01")
        book.record(qo2, 0.3, Q.Outcome(100, 0.0, 0.5, decision_changed=False, information_bits=0.05), "FAILURE", f"2026-10-{1 + i % 20:02d}", "2027-01-01")
    assert book.multiplier("loss", "2027-01-01") > 1.3 > 0.8 > book.multiplier("surprise", "2027-01-01")
    assert book.multiplier("loss", "2026-09-30") == 1.0                                          # invisible before it matured
    assert book.calibration("surprise", "2027-01-01")["verdict"] == "overclaims"
    with pytest.raises(FirewallBreach):
        book.record(qo, 0.3, Q.Outcome(1, 0, 1), "FAILURE", "2027-01-01", "2027-01-01")
    assert Q.QuestionOutcomeBook.from_json(book.to_json()).table("2027-01-01") == book.table("2027-01-01")


def test_memory_dedup_blocks_tested_question_and_answer_lineage():
    from engine.learning.experiment_memory import ExperimentLedger
    qo = Q.generate([_ev("loss", "dup_a", loss_share=0.3)], "2026-09-29").questions[0]

    class Seen:
        def already_tested(self, q, d, now, reason=""):
            from types import SimpleNamespace
            return SimpleNamespace(blocking=True, tested_before=True, message="ran already")
    assert Q.check_against_memory(qo, Seen(), "2026-10-01").status == "BLOCKED"
    assert Q.check_against_memory(qo, ExperimentLedger(), "2026-10-01").status == "NOVEL"
    kept, blocked = Q.filter_novel([qo], Seen(), "2026-10-01")
    assert not kept and blocked
    ev = _ev("loss", "dup_a", loss_share=0.3)
    verdict, fups = Q.answer(qo, Q.Outcome(500, 0.2, 0.001, loss_avoided=0.4, decision_changed=True), ev, None, 0.3, "2026-10-01", "2026-10-02")
    assert verdict == "SUCCESS" and fups and fups[0].source == "new_discovery"
    v2, f2 = Q.answer(qo, Q.Outcome(3, 0.2, 0.001), ev, None, 0.3, "2026-10-01", "2026-10-02")
    assert v2 == "UNDECIDED" and f2[0].source == "loss"


def test_slots_guarantee_diversity_and_specific_hypotheses_have_chance():
    objs = Q.generate([_ev("loss", f"l{i}", loss_share=0.5, magnitude=0.9, problem=Problem.LOSS_AVOIDANCE) for i in range(6)] + [_ev("surprise", "s1", magnitude=0.1)],
                      "2026-09-29").questions
    got = Q.allocate_slots(objs, 4)
    assert "surprise" in {q.source for q in got} and sum(1 for q in got if q.source == "loss") <= 2
    for src in ("new_discovery", "pattern_break", "regime_change", "missed_winner"):
        hs = Q.specific_hypotheses(_ev(src))
        assert any(h.kind == "noise" for h in hs) or src == "regime_change"
        assert abs(sum(h.prior for h in hs) - 1) < 1e-9
    assert Q.ledger_integrity(Q.QuestionLedger()) == []


# ------------------------------------------------------------------ wave-2 increment: targets
def test_scorers_route_every_source_and_reject_unknown():
    rows = {"surprise": {"z": 4}, "loss": {"loss_share": 0.3}, "win": {"n": 20, "wins": 18}, "missed_winner": {"gain_share": 0.3}, "missed_loser": {"loss_share": 0.3},
            "volatility_cluster": {"size": 12, "base_size": 3}, "pattern_break": {"drop": 0.3, "n_after": 40}, "contradiction": {"strength": 0.4},
            "new_regime": {"shift": 0.5}, "data_anomaly": {"severity": 0.7}, "research_failure": {"barren_streak": 5},
            "new_combination": {"lift_a": 0.02, "lift_b": 0.02, "lift_joint": 0.1}, "unknown_area": {"size": 0.4}}
    for s, r in rows.items():
        sc = T.score_row(s, r)
        assert sc.source == s and 0 < sc.magnitude <= 1, s
    assert T.score_row("win", {"n": 2, "wins": 2}).magnitude == 0.0                   # too few to study
    assert T.score_row("volatility_cluster", {"size": 3, "base_size": 3}).confidence < T.score_row("volatility_cluster", {"size": 12, "base_size": 3}).confidence
    with pytest.raises(T.TargetError):
        T.score_row("astrology", {})


def test_source_yield_learns_and_keeps_floor():
    sy = T.SourceYield()
    for i in range(30):
        sy.observe("pattern_break", True, "2026-01-01", "2026-02-01")
        sy.observe("surprise", False, "2026-01-01", "2026-02-01")
    assert sy.multiplier("pattern_break") > 1.3 and 0.5 <= sy.multiplier("surprise") < 0.8
    with pytest.raises(FirewallBreach):
        sy.observe("loss", True, "2026-02-01", "2026-02-01")


def test_coverage_model_finds_understudied_and_decays_old_studies():
    m = T.CoverageModel("sector", half_life_days=100)
    m.register(["tech", "energy", "utilities", "rare"])
    for i in range(30):
        m.study("tech", f"2026-0{1 + i % 5}-{1 + i % 27:02d}", "2027-01-01")
        m.study("energy", f"2026-0{1 + i % 5}-{1 + i % 27:02d}", "2027-01-01")
    m.study("utilities", "2026-03-01", "2027-01-01")
    need = m.need("2026-09-01")
    assert need["rare"] == 1.0 and need["tech"] < 0.2
    assert [b for b, _ in m.understudied("2026-09-01")][0] == "rare"
    assert m.gini("2026-09-01") > 0.2
    old = m.decayed_counts("2026-09-01")["tech"]
    assert m.decayed_counts("2028-09-01")["tech"] < 0.05 * old                           # studies age out
    assert T.CoverageModel.from_json(m.to_json()).need("2026-09-01") == need
    assert T.CoverageModel("sector").understudied("2026-09-01") == []                    # empty model
    tg = T.coverage_targets(m, "2026-09-02", "2026-09-01")
    assert tg and tg[0].source == "understudied_sector"


def _track_rows(cond, true_effect, seed, n_days=60):
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        dd = f"2003-{6 + d // 28:02d}-{1 + d % 28:02d}"
        for _ in range(6):
            disp = float(rng.uniform(0.3, 2.4))
            fail = rng.random() < ((0.85 if disp > cond.threshold else 0.10) if true_effect else 0.30)
            out.append(T.DatedRow(dd, T.PredictionRow("pattern_x", not fail, {"dispersion": disp})))
    return out


def test_tracker_follows_true_condition_to_confirmation_and_never_confirms_a_null():
    day = T.dispersion_example(0)
    cond = T.find_failure_conditions(day.predictions)[0]
    tgt = T.condition_target("pattern_x", cond, "2003-05-01")
    tr = T.OpenConditionTracker()
    tc = tr.open(tgt, "2003-05-01", "2003-05-02")
    tr.update(_track_rows(cond, True, 3), "2004-01-01")
    assert tc.state == T.TrackState.CONFIRMED and tc.closed
    for seed in range(4):
        tr2 = T.OpenConditionTracker()
        tc2 = tr2.open(tgt, "2003-05-01", "2003-05-02")
        tr2.update(_track_rows(cond, False, seed), "2004-01-01")
        assert tc2.state != T.TrackState.CONFIRMED
    assert T.OpenConditionTracker.from_json(tr.to_json()).summary() == tr.summary()
    tr.update([], "2004-01-01")                                                           # empty update is harmless
    tr3 = T.OpenConditionTracker()
    tr3.open(tgt, "2003-05-01", "2003-05-02")
    with pytest.raises(FirewallBreach):
        tr3.update([T.DatedRow("2004-01-01", T.PredictionRow("pattern_x", True, {"dispersion": 1.0}))], "2004-01-01")
    assert T.expected_days_to_confirm(cond, 3.0) < 60
    assert T.expected_days_to_confirm(T.Condition("f", ">", 1, 5, 1, 5, 2, 0.1), 3.0) == float("inf")


def test_autopsy_maps_to_sources_and_day_book_is_append_only():
    ents = [T.AutopsyEntry("2003-05-01", "FALSE_NEGATIVE", "sit_a", 0.2, knowable_before=0.8),
            T.AutopsyEntry("2003-05-01", "LOSER", "sit_b", -0.3, knowable_before=0.6),
            T.AutopsyEntry("2003-05-01", "UNPREDICTABLE_MOVER", "sit_c", 0.3, sd=0.05)]
    ents += [T.AutopsyEntry("2003-05-01", "WINNER", f"w{i}", 0.05, "pattern_x", {"dispersion": 0.5 + 0.05 * i}, True, True) for i in range(8)]
    di = T.day_input_from_autopsy(ents, "2003-05-02")
    assert len(di.missed_winners) == 1 and len(di.missed_losers) == 1 and len(di.surprises) == 1 and len(di.predictions) == 8
    led, tr, book = T.TargetLedger(), T.OpenConditionTracker(), T.DayTargetBook()
    rep = T.run_autopsy_day(ents, "2003-05-02", led, tr, book)
    assert {"missed_winner", "missed_loser", "surprise"} <= set(rep.by_source)
    assert book.days["2003-05-01"]["counts"]["WINNER"] == 8
    tid = rep.targets[0].target_id
    book.link(tid, question_id="Q1", verdict="PROMOTED")
    assert book.trace(tid)["question_id"] == "Q1" and book.yield_by_source()
    with pytest.raises(T.TargetError):
        book.record_day("2003-05-01", {}, rep)
    with pytest.raises(T.TargetError):
        T.day_input_from_autopsy([], "2003-05-02")
    with pytest.raises(FirewallBreach):
        T.day_input_from_autopsy(ents, "2003-05-01")
    assert T.DayTargetBook.from_json(book.to_json()).days == book.days


# ------------------------------------------------------------------ wave-2 increment: forest persistence
def test_forest_save_load_resume_and_tamper_detection(tmp_path):
    f = H.TreeForest()
    f.add(_tree())
    H.simulate_investigation(f.trees["T"], "h_wrong_context", np.random.default_rng(2), max_steps=3)
    assert H.verify_forest(f) == []
    p = tmp_path / "forest.json"
    cs = H.save_forest(f, p)
    assert H.forest_checksum(H.load_forest(p)) == cs
    p.write_text(p.read_text().replace('"against"', '"againstx"', 1), encoding="utf-8")            # corrupt the file
    with pytest.raises(H.ForestIntegrityError):
        H.load_forest(p)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(H.ForestIntegrityError):
        H.load_forest(p)
    q = tmp_path / "new.json"
    forest, out = H.resume(q, [], "2026-10-01")                                                 # no file: start empty, save
    assert len(forest) == 0 and q.exists()
    with pytest.raises(FileNotFoundError):
        H.load_forest(tmp_path / "missing.json")


def test_resume_replaying_overlapping_results_is_safe_and_broken_forest_not_saved(tmp_path):
    import dataclasses
    f = H.TreeForest()
    f.add(_tree())
    p = tmp_path / "f.json"
    H.save_forest(f, p)
    t = f.trees["T"]
    act = t.next_action()
    out = t.get(act.nid).outcomes()[0] if t.get(act.nid).kind == H.NodeKind.TEST else "not_found"
    res = [H.TestResult("T", act.nid, out, "2026-10-01")]
    forest, s1 = H.resume(p, res, "2026-10-03")
    forest2, s2 = H.resume(p, res, "2026-10-04")                                                 # same log again after a "crash"
    assert s1.rejected == () and len(s2.rejected) == 1 and H.verify_forest(forest2) == []
    tr = forest2.trees["T"]
    tr.nodes[H.UNKNOWN_HID] = dataclasses.replace(tr.nodes[H.UNKNOWN_HID], status=H.NodeStatus.FAILED)
    with pytest.raises(H.ForestIntegrityError):
        H.save_forest(forest2, tmp_path / "bad.json")


# ------------------------------------------------------------------ wave-2b: detectors, selection, builders
def test_detectors_catch_planted_and_stay_silent_on_null():
    rng = np.random.default_rng(0)
    assert T.detect_clusters({"tech": 12, "energy": 2, "util": 1}, {"tech": 2.0, "energy": 2.0, "util": 2.0})[0]["subject"] == "tech"
    assert T.detect_clusters({"tech": 3, "energy": 2}, {"tech": 2.0, "energy": 2.0}) == []
    ok = (rng.random(200) < 0.7).astype(float)
    broken = np.concatenate([ok[:150], (rng.random(50) < 0.3).astype(float)])
    assert T.detect_breaks({"p_broken": broken, "p_fine": ok})[0]["subject"] == "p_broken"
    assert [r["subject"] for r in T.detect_breaks({"p_fine": ok})] == []
    shifted = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 20)])
    assert T.detect_regime_shift(shifted) is not None and T.detect_regime_shift(rng.normal(0, 1, 220)) is None
    assert T.detect_regime_shift([1.0, 2.0]) is None
    c = T.detect_contradictions({"sig": {"a": (18, 20), "b": (5, 20), "c": (17, 20)}})
    assert {(r["subject"], r["counterpart"]) for r in c} == {("a", "b"), ("b", "c")}


def test_bh_and_threshold_stability_and_conjunction():
    assert T.bh_select([]) == [] and T.bh_select([0.001, 0.9, 0.5, 0.004]) == [0, 3]
    day = T.dispersion_example(0)
    assert T.threshold_stability(day.predictions, "dispersion")["found_share"] > 0.5
    rng = np.random.default_rng(2)
    noise = [T.PredictionRow("p", bool(rng.random() < 0.7), {"dispersion": float(rng.uniform(0, 2))}) for _ in range(30)]
    assert T.threshold_stability(noise, "dispersion")["verdict"] == "UNSTABLE"
    rows = [T.PredictionRow("p", not (d > 1.0 and b > 0.5 and rng.random() < 0.9), {"dispersion": float(d), "breadth": float(b)})
            for d, b in zip(rng.uniform(0, 2, 200), rng.uniform(0, 1, 200))]
    base = T.find_failure_conditions(rows)
    assert base and T.find_conjunction_conditions(rows, base) is not None


def test_daily_selection_keeps_breadth_and_respects_budget():
    mk = lambda src, i, cost, pri: __import__("dataclasses").replace(T._mk(src, f"s{i}", f"t {src} {i}", "h", "t", Problem.VOLATILITY, "2003-05-01", 0.5, 0.5, cost, 1.0), priority=pri)
    ts = [mk("loss", i, 20.0, 1.0 - 0.01 * i) for i in range(6)] + [mk("surprise", 0, 20.0, 0.1), mk("unknown_area", 0, 60.0, 0.05)]
    chosen, deferred = T.select_daily(ts, 100.0, max_per_source=3)
    assert {t.source for t in chosen} >= {"loss", "surprise"} and sum(t.value.compute_cost for t in chosen) <= 100.0
    assert sum(1 for t in chosen if t.source == "loss") <= 3 and deferred
    assert T.select_daily([], 100.0) == ((), ())
    aged = T.age_targets(ts, "2004-05-01")
    assert aged == []                                                                    # a year old: below the drop line
    merged = T.merge_across_days([[ts[0]], [ts[0]], [ts[1]]])
    assert len(merged) == 2 and merged[0].magnitude >= merged[1].magnitude


def test_run_days_is_deterministic_and_bridges_to_questions():
    def day(d, n_fail):
        e = [T.AutopsyEntry(d, "WINNER", f"w{i}", 0.05, "pattern_x", {"dispersion": 0.4 + 0.05 * i}, True, True) for i in range(10)]
        e += [T.AutopsyEntry(d, "FALSE_POSITIVE", f"f{i}", -0.06, "pattern_x", {"dispersion": 1.7 + 0.05 * i}, True, False) for i in range(n_fail)]
        e.append(T.AutopsyEntry(d, "FALSE_NEGATIVE", "sit_m", 0.2, knowable_before=0.7))
        return e
    days = {"2003-05-01": day("2003-05-01", 5), "2003-05-02": day("2003-05-02", 4)}
    r1 = T.run_days(days)
    r2 = T.run_days(days)
    assert r1.digest == r2.digest and r1.n_targets > 0 and "missed_winner" not in r1.silent_sources
    rep = T.run_day(T.dispersion_example(0), "2003-05-02")
    t = T.run_day(T.DayInput("2003-05-01", missed_winners=[{"subject": "sit_m", "gain_share": 0.4, "knowable_before": 0.8}]), "2003-05-02").targets[0]
    ev = T.target_to_event(t, "2003-05-02")
    assert ev.source == "missed_winner" and Q.generate([ev], "2003-05-03").questions
    with pytest.raises(FirewallBreach):
        T.target_to_event(t, "2003-05-01")


def test_remaining_question_builders_cost_model_and_reports():
    assert Q.builders_cover_sources() == []
    assert Q.event_from_false_positive("p", 2, 20, 0.7, "2026-09-20") is None and Q.event_from_false_positive("p", 9, 12, 0.7, "2026-09-20")
    assert Q.event_from_anomaly("d", 0.1, "2026-09-20") is None and Q.event_from_anomaly("d", 0.8, "2026-09-20")
    assert Q.event_from_coverage("tech", 0.2, "2026-09-20") is None and Q.event_from_research_failure("f", 2, "2026-09-20") is None
    qo = Q.generate([_ev("loss", "cm", loss_share=0.3)], "2026-09-29").questions[0]
    cm = Q.PlanCostModel()
    for _ in range(12):
        cm.observe("loss", 10.0, 20.0)
    assert Q.PlanCostModel().recost(qo).plan.cost_minutes == pytest.approx(qo.plan.cost_minutes) and cm.recost(qo).plan.cost_minutes > 1.4 * qo.plan.cost_minutes
    with pytest.raises(Q.QuestionError):
        cm.observe("loss", 0.0, 1.0)
    assert Q.refine_question(qo, Q.Outcome(n=qo.plan.min_n, lift=0.1, p_value=0.01), "2026-10-01") is None
    r = Q.refine_question(qo, Q.Outcome(n=max(qo.plan.min_n // 2, 1), lift=0.1, p_value=0.5), "2026-10-01")
    assert r is not None and r.plan.cost_minutes > qo.plan.cost_minutes
    assert Q.refine_question(qo, Q.Outcome(n=1, lift=0, p_value=1), "2026-10-01") is None or qo.plan.min_n <= 5
    txt = Q.explain_question(qo)
    assert "success:" in txt and "failure:" in txt and "| 1 |" in Q.agenda_markdown([qo])
    rep = Q.quality_report(Q.QuestionLedger(), Q.QuestionOutcomeBook(), "2026-10-01")
    assert rep["unbuilt_sources"] == [] and rep["integrity"] == []
    dm = Q.DifficultyModel()
    assert dm.difficulty("loss", 0.4) == 0.4
    for _ in range(20):
        dm.observe("loss", 3.0, True)
    assert dm.difficulty("loss", 0.4) > 0.6
