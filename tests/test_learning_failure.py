"""S05 tests: failure classifier (D01-D10), selection/timing/risk separation (C09), loss postmortems (section 24).
Synthetic data only; every test plants a known cause/defect and proves the code catches it, plus the empty cases."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.learning import failure as F
from engine.learning import postmortem as PM
from engine.learning import separation as S
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle, Promotion,
                                  Provenance, Subsystem, Unknown)

FC = FailureCause
NOW = "2020-06-01"


def trade(**kw):
    base = dict(rid="t1", decided_at="2019-12-20", resolved_at="2019-12-30", side=1, pnl=-0.0515, signal_ret=-0.05, entry_gap=0.001,
                end_ret_from_fill=-0.051, exit_ret=-0.051, exp_move=0.07, exp_vol=0.03)
    base.update(kw)
    return F.TradeRecord(**base)


def know(kid="K1", contexts=None, anti=None, epistemic=Epistemic.SUPPORTED, promotion=Promotion.CHAMPION, learned="2019-01-01",
         reliability=0.8, failure_risk=0.1):
    prov = Provenance(created_real="2019-01-02", learned_at=learned, code_hash="abc", outcomes_seen_through=learned)
    return SimpleNamespace(knowledge_id=kid, version=1, epistemic=epistemic, lifecycle=Lifecycle.ACTIVE, promotion=promotion,
                           confidence=Confidence(current_reliability=reliability, failure_risk=failure_risk), provenance=prov,
                           contexts=contexts or {}, anti_contexts=anti or {}, decision_effect=(DecisionEffect.SELECTION,))


# ------------------------------------------------------------------------------------------------ D01-D10 classifier
@pytest.mark.parametrize("cause", list(F.CAUSES_EXPLAINING))
def test_planted_cause_is_named(cause):
    """Plant each cause with three seeds; the classifier must name exactly that cause every time."""
    clf = F.LossClassifier()
    for seed in range(3):
        t, env = F.planted_case(cause, seed)
        c = clf.classify(t, env, NOW)
        assert c.cause == cause, f"seed {seed}: {c.cause} {c.note} {c.secondary}"
        assert c.named and 0 < c.confidence <= 1


def test_planted_unknown_is_not_forced():
    clf = F.LossClassifier()
    t, env = F.planted_case(FC.UNKNOWN, 0)
    c = clf.classify(t, env, NOW)
    assert c.cause == FC.UNKNOWN and c.unknown_state == Unknown.UNKNOWN and c.confidence == 0.0
    assert c.top_subsystem() == Subsystem.DIRECTION             # the subsystem is still known even though no cause is named


def test_battery_confusion_has_no_wrong_named():
    r = F.planted_battery(seeds=(0, 1, 2, 3))
    assert r["wrong_named"] == 0 and r["acc_named"] == 1.0
    assert r["matrix"]["UNKNOWN"] == {"UNKNOWN": 4}


def test_insufficient_when_detectors_cannot_run():
    t = trade(signal_ret=None, entry_gap=None, end_ret_from_fill=None, exit_ret=None, exp_move=None, exp_vol=None, pnl=-0.05)
    c = F.LossClassifier().classify(t, None, NOW)
    assert c.cause == FC.INSUFFICIENT_EVIDENCE and c.unknown_state == Unknown.INSUFFICIENT_DATA
    assert c.coverage < 0.375 and "pattern" in c.missing


def test_conflict_is_unknown_not_a_coin_flip():
    def d1(t, env, p):
        return F.DetectorResult("a", True, (), (F.Evidence("a", FC.SELECTION_ERROR, 0.8),))

    def d2(t, env, p):
        return F.DetectorResult("b", True, (), (F.Evidence("b", FC.TIMING_ERROR, 0.78),))
    c = F.LossClassifier(detectors={"a": d1, "b": d2}).classify(trade(), None, NOW)
    assert c.cause == FC.UNKNOWN and c.unknown_state == Unknown.CONFLICTED
    assert {k for k, _ in c.secondary} == {FC.SELECTION_ERROR, FC.TIMING_ERROR}


def test_contradicting_evidence_lowers_a_cause():
    ev_for = [F.Evidence("x", FC.RISK_ERROR, 0.7)]
    ev_against = ev_for + [F.Evidence("y", FC.RISK_ERROR, 0.9, supports=False)]
    a = F.combine_evidence(ev_for)[FC.RISK_ERROR].score
    b = F.combine_evidence(ev_against)[FC.RISK_ERROR].score
    assert a == pytest.approx(0.7) and b < 0.25


def test_non_losses_and_noise_are_not_analysed():
    clf = F.LossClassifier()
    assert clf.classify(trade(pnl=0.03), None, NOW).meaningful is False
    assert clf.classify(trade(pnl=-0.004), None, NOW).meaningful is False
    inside = clf.classify(trade(pnl=-0.02, exp_vol=0.05), None, NOW)
    assert not inside.meaningful and "volatility" in inside.note


def test_fail_closed_on_time():
    clf = F.LossClassifier()
    with pytest.raises(FirewallBreach):
        clf.classify(trade(), None, "2019-12-30")             # resolves ON now: not yet known
    t, env = F.planted_case(FC.WEAKENING_EFFECT, 0)
    late = dataclasses.replace(env.patterns["P1"], dates=env.patterns["P1"].dates[:-1] + ("2021-01-01",))
    with pytest.raises(FirewallBreach):
        clf.classify(t, dataclasses.replace(env, patterns={"P1": late}), NOW)


def test_invalid_records_are_rejected():
    with pytest.raises(ValueError):
        trade(side=0).require_valid()
    with pytest.raises(ValueError):
        trade(context={"ticker": 1.0}).require_valid()
    with pytest.raises(ValueError):
        trade(resolved_at="2019-12-10").require_valid()
    with pytest.raises(ValueError):
        F.LossClassifier(detectors={})


def test_classification_is_deterministic():
    t, env = F.planted_case(FC.REGIME_CHANGE, 3)
    a = F.LossClassifier().classify(t, env, NOW).to_dict()
    b = F.LossClassifier().classify(t, env, NOW).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_context_condition_semantics():
    assert F.condition_holds((0, 1), 0.5) is True and F.condition_holds((0, 1), 2) is False
    assert F.condition_holds((3, None), 4) is True
    assert F.condition_holds({"in": ["a", "b"]}, "c") is False
    assert F.condition_holds((0, 1), None) is None              # unknown is never "satisfied"
    v = F.evaluate_contexts(SimpleNamespace(contexts={"m_x": (0, 1), "m_y": (0, 1)}, anti_contexts={"m_z": (5, None)}),
                            {"m_x": 3.0, "m_z": 9.0})
    assert v.failed == ("m_x",) and v.anti_hit == ("m_z",) and v.unknown == ("m_y",) and not v.holds_all


def test_pattern_detectors_do_not_fire_on_a_healthy_pattern():
    t, env = F.planted_case(FC.UNKNOWN, 5)
    r = F.detect_pattern(t, env, F.FailureParams(), NOW)
    assert r.ran and not [e for e in r.evidence if e.supports]


def test_pattern_history_too_short_is_reported_not_guessed():
    t, env = F.planted_case(FC.UNKNOWN, 0)
    short = F.PatternHistory("P1", tuple([0.01] * 5))
    r = F.detect_pattern(t, dataclasses.replace(env, patterns={"P1": short}), F.FailureParams())
    assert not r.ran and any("too short" in m for m in r.missing)


def test_direction_surprise_bits():
    assert F.surprise_bits(0.5, True) == pytest.approx(1.0)
    assert F.surprise_bits(0.9, False) == pytest.approx(np.log2(10), rel=1e-6)
    assert F.surprise_bits(None, True) is None


def test_measurement_detector_catches_unreconciled_books():
    t = trade(signal_ret=-0.06, entry_gap=0.0, end_ret_from_fill=0.30, exit_ret=-0.05, pnl=-0.05)
    r = F.detect_measurement(t, F.FailureEnv(), F.FailureParams())
    assert any(e.facts.get("reconcile_error", 0) > 0.2 for e in r.evidence if e.supports)
    clean = F.detect_measurement(trade(), F.FailureEnv(), F.FailureParams())
    assert clean.ran and not [e for e in clean.evidence if e.supports]


def test_robustness_of_a_clear_case_and_a_borderline_one():
    clf = F.LossClassifier()
    t, env = F.planted_case(FC.REVERSAL, 0)
    assert F.robustness(clf, t, env, NOW, n=20, seed=1)["stability"] >= 0.9
    t2, env2 = F.planted_case(FC.SELECTION_ERROR, 0)
    r = F.robustness(clf, t2, env2, NOW, n=20, rel_noise=0.15, seed=2)
    assert r["n"] == 20 and sum(r["distribution"].values()) == 20


def test_ablating_a_detector_changes_answers_that_needed_it():
    cases = [F.planted_case(c, 0) for c in (FC.FALSE_PATTERN, FC.REGIME_CHANGE, FC.SELECTION_ERROR)]
    ab = F.ablate_detectors(None, cases, NOW)
    assert ab["pattern"]["changed"] >= 1 and ab["regime"]["changed"] >= 1 and ab["selection"]["changed"] >= 1
    assert ab["measurement"]["changed"] == 0                   # nothing here needed the data-integrity detector


def test_confidence_is_calibrated_on_planted_cases():
    clf = F.LossClassifier()
    pairs = []
    for c in F.CAUSES_EXPLAINING:
        for s in range(3):
            t, env = F.planted_case(c, s)
            pairs.append((clf.classify(t, env, NOW), c))
    cal = F.confidence_calibration(pairs)
    assert cal["accuracy"] == 1.0 and cal["verdict"] in ("UNDERCONFIDENT", "CALIBRATED")
    assert F.confidence_calibration([])["verdict"] == "NOTHING_NAMED"


def test_ledger_counts_and_refuses_duplicates():
    clf, led = F.LossClassifier(), F.FailureLedger()
    for i, c in enumerate([FC.RISK_ERROR, FC.RISK_ERROR, FC.UNKNOWN, FC.REVERSAL]):
        t, env = F.planted_case(c, i, rid=f"r{i}")
        led.add(clf.classify(t, env, NOW), {"era": "a" if i < 2 else "b"})
    assert led.cause_counts()["RISK_ERROR"] == 2 and led.unknown_rate() == pytest.approx(0.25)
    assert led.by_tag("era")["b"] == {"REVERSAL": 1, "UNKNOWN": 1} or led.by_tag("era")["b"] == {"UNKNOWN": 1, "REVERSAL": 1}
    with pytest.raises(ValueError):
        led.add(clf.classify(*F.planted_case(FC.RISK_ERROR, 0, rid="r0"), NOW))
    assert "IMPLEMENTED - NOT VALIDATED" in led.report()
    assert F.FailureLedger().unknown_rate() != F.FailureLedger().unknown_rate()     # NaN on the empty ledger


def test_explain_classification_reads_both_ways():
    clf = F.LossClassifier()
    named = F.explain_classification(clf.classify(*F.planted_case(FC.RISK_ERROR, 0), NOW))
    assert "RISK_ERROR" in named and "+ risk" in named
    unk = F.explain_classification(clf.classify(trade(signal_ret=None, entry_gap=None, end_ret_from_fill=None, exit_ret=None,
                                                      exp_move=None, exp_vol=None), None, NOW))
    assert "INSUFFICIENT_EVIDENCE" in unk and "could not run" in unk


def test_trade_dict_roundtrip_and_unknown_field():
    t, _ = F.planted_case(FC.TIMING_ERROR, 0)
    assert F.trade_from_dict(F.trade_to_dict(t)).rid == t.rid
    with pytest.raises(ValueError):
        F.trade_from_dict({**F.trade_to_dict(t), "ticker": "AAPL"})


def test_records_from_lessons_frame_respects_now_and_hides_identity():
    dates = pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-13"])
    idx = pd.MultiIndex.from_arrays([dates, ["AAA", "BBB", "AAA"]], names=["date", "ticker"])
    frame = pd.DataFrame({"score": [0.9, -0.5, 0.7], "y": [-0.06, 0.02, -0.03], "taken": [True, True, True],
                          "resolved": pd.to_datetime(["2020-01-13", "2020-01-13", "2020-01-20"])}, index=idx)
    X = pd.DataFrame({"m_vol": [1.0, 1.0, 2.0], "f": [0.1, 0.2, 0.3]}, index=idx)
    recs = F.records_from_lessons_frame(frame, X, "2020-01-14")
    assert len(recs) == 2 and all("AAA" not in json.dumps(F.trade_to_dict(r)) for r in recs)
    assert recs[0].context == {"m_vol": 1.0} and recs[0].pnl == pytest.approx(-0.0605)
    assert F.records_from_lessons_frame(frame.iloc[0:0], X, NOW) == []


def test_pattern_histories_are_fail_closed():
    eff = pd.DataFrame({"pA": [0.01, 0.02, 0.01]}, index=pd.to_datetime(["2020-01-06", "2020-01-13", "2020-02-03"]))
    pats = pd.DataFrame({"key_named": ["pA"], "p_real": [0.9], "t_disc": [4.0], "t_conf": [2.5]})
    with pytest.raises(FirewallBreach):
        F.pattern_histories_from_effects(eff, pats, "2020-01-20")
    h = F.pattern_histories_from_effects(eff, pats, "2020-01-20", on_future="drop")
    assert h["pA"].effects == (0.01, 0.02) and h["pA"].p_real == 0.9


# ------------------------------------------------------------------------------------------------ producers: existing lesson systems
def test_lessons_lesson_and_memory_lesson_become_hypotheses():
    from engine import lessons, memory
    L = lessons.Lesson(lid="L1", conds=[("m_vix", ">", 0.7)], direction=-1, factor=0.5, category="regime_failure", n=80, n_weeks=12,
                       n_tickers=20, delta=-0.01, t=-3.0, p=0.01, val_delta=-0.008, a=8, b=2, born_tick=1, ttl=400, kind="regime_misread")
    ml = memory.Lesson(arm="A1", fingerprint="fp", context={}, features={}, outcome_bin="lo", error_type="false_positive",
                       source_experiment="e", date="2019Q1", era="e1", relevance=0.5, reliability=0.6, shock_state="pre_break")
    noise = dataclasses.replace(ml, error_type="noise")
    hs = F.hypotheses_from_lessons([L, ml, noise, L])
    assert len(hs) == 2                                        # duplicates merged, noise produced nothing
    h_l, h_m = hs
    assert h_l.cause == FC.REGIME_CHANGE and h_l.source == "lessons.Lesson" and h_l.basis["n_weeks"] == 12
    assert h_m.cause == FC.REGIME_CHANGE and h_m.source == "memory.Lesson"
    assert all(h.epistemic == Epistemic.HYPOTHESIS and not h.production_effect and not h.validate() for h in hs)
    L.status = "retired"
    assert F.hypothesis_from_lessons_lesson(L) is None
    with pytest.raises(TypeError):
        F.hypotheses_from_lessons([object()])


# ------------------------------------------------------------------------------------------------ C09 separation
def test_decomposition_is_exactly_additive_on_random_trades():
    rng = np.random.default_rng(0)
    for i in range(300):
        side = int(rng.choice([-1, 1]))
        sig = float(rng.normal(0, 0.08))
        gap = float(rng.normal(0, 0.02))
        end = (1 + sig) / (1 + gap) - 1
        ex = float(end + rng.normal(0, 0.02))
        t = trade(rid=f"r{i}", side=side, signal_ret=sig, entry_gap=gap, end_ret_from_fill=end, exit_ret=ex, pnl=side * ex - 0.0005,
                  exp_move=float(abs(rng.normal(0.07, 0.02))))
        d = S.decompose(t)
        assert abs(d.checksum()) < 1e-12 and abs(d.residual) < 1e-12
        d2 = S.decompose(t, order="direction_first")
        assert abs(d2.checksum()) < 1e-12
        assert d2.parts["timing"] == pytest.approx(d.parts["timing"]) and d2.parts["exit"] == pytest.approx(d.parts["exit"])


def test_well_selected_badly_timed_teaches_timing_not_selection():
    """The section-23 example: the stock moved more than promised in the right direction; the overnight gap ate it."""
    t = trade(rid="wt", signal_ret=0.06, entry_gap=0.08, end_ret_from_fill=1.06 / 1.08 - 1, exit_ret=1.06 / 1.08 - 1,
              pnl=1.06 / 1.08 - 1 - 0.0005, exp_move=0.05, dir_prob=0.6, exp_vol=0.01)
    clf = F.LossClassifier()
    cls = clf.classify(t, F.FailureEnv(), NOW)
    naive = F.Classification(**{**cls.__dict__, "cause": FC.SELECTION_ERROR, "confidence": 0.9, "unknown_state": None})   # a label that blames selection
    att = S.attribute(t, cls)
    assert att.primary == Subsystem.TIMING and Subsystem.SELECTION in att.protected and att.teach == (Subsystem.TIMING,)
    sig = S.route(t, naive, att)
    assert sig.overridden and sig.weight(Subsystem.SELECTION) == 0.0 and sig.weight(Subsystem.TIMING) > 0.5
    assert "SELECTION" in sig.blocked


def test_wrong_side_teaches_direction_and_wrong_move_teaches_selection():
    wrong = trade(rid="w", signal_ret=-0.09, end_ret_from_fill=-0.09, exit_ret=-0.09, entry_gap=0.0, pnl=-0.0905, dir_prob=0.7, exp_move=0.07)
    assert S.attribute(wrong).primary == Subsystem.DIRECTION
    stalled = trade(rid="s", signal_ret=0.004, end_ret_from_fill=0.004, exit_ret=-0.02, entry_gap=0.0, pnl=-0.0205, exp_move=0.07)
    a = S.attribute(stalled)
    assert a.primary == Subsystem.SELECTION and Subsystem.TIMING in a.protected


def test_stop_gap_is_risk_and_giveback_is_exit():
    gapped = trade(rid="g", signal_ret=-0.18, end_ret_from_fill=-0.18, exit_ret=-0.18, entry_gap=0.0, pnl=-0.1805, stop=0.05, stop_hit=True,
                   stop_fill_ret=-0.18, exp_move=0.18, dir_prob=None)
    d = S.decompose(gapped)
    assert d.risk == pytest.approx(-0.13) and d.exit > 0
    assert S.attribute(gapped).primary in (Subsystem.RISK, Subsystem.SELECTION, Subsystem.DIRECTION)
    assert Subsystem.RISK in S.attribute(gapped).teach
    giveback = trade(rid="gb", signal_ret=0.01, end_ret_from_fill=0.01, exit_ret=-0.04, entry_gap=0.0, pnl=-0.0405, exp_move=0.01, mfe=0.06)
    assert S.attribute(giveback).primary == Subsystem.EXIT


def test_oversizing_is_charged_to_risk():
    small = trade(rid="a", weight=0.1, target_weight=0.1)
    big = trade(rid="b", weight=0.3, target_weight=0.1)
    assert S.decompose(small).sizing_excess == 0.0 and S.decompose(big).sizing_excess < -0.09
    assert S.attribute(big).share(Subsystem.RISK) > 0.3


def test_unreconciled_books_refuse_to_assign_blame():
    t = trade(rid="bad", signal_ret=-0.05, end_ret_from_fill=-0.05, exit_ret=-0.05, pnl=-0.5)
    a = S.attribute(t)
    assert a.basis == "insufficient" and a.teach == () and "reconcile" in a.conflict


def test_no_loss_no_blame():
    a = S.attribute(trade(rid="w", signal_ret=0.08, end_ret_from_fill=0.08, exit_ret=0.08, entry_gap=0.0, pnl=0.0795, exp_move=0.07))
    assert a.basis == "no_loss" and a.primary is None and a.blame == {}


def test_detector_votes_that_contradict_the_arithmetic_are_reported():
    t = trade(rid="c", signal_ret=-0.09, end_ret_from_fill=-0.09, exit_ret=-0.09, entry_gap=0.0, pnl=-0.0905, dir_prob=0.7, exp_move=0.07)
    fake = F.Classification(rid="c", cause=FC.SELECTION_ERROR, score=0.9, confidence=0.9, unknown_state=None, meaningful=True, coverage=1.0,
                            ran=(), missing={}, secondary=(), scores=(), subsystem_votes={"SELECTION": 0.9}, surprise_bits=None)
    a = S.attribute(t, fake)
    assert a.basis == "reconciled" and "detectors point at SELECTION" in a.conflict


def test_misattribution_audit_counts_what_a_naive_rule_gets_wrong():
    rng = np.random.default_rng(1)
    trades = []
    for i in range(60):     # right idea, adverse gap: decided_by SELECTION, true blame TIMING
        gap = 0.06 + rng.random() * 0.02
        end = 1.05 / (1 + gap) - 1
        trades.append(trade(rid=f"t{i}", signal_ret=0.05, entry_gap=gap, end_ret_from_fill=end, exit_ret=end, pnl=end - 0.0005, exp_move=0.04))
    r = S.misattribution_audit(trades)
    assert r["n_losses"] == 60 and r["misattribution_rate"] == 1.0 and r["right_idea_wrong_execution"] == 60
    assert S.misattribution_audit([])["misattribution_rate"] != S.misattribution_audit([])["misattribution_rate"]      # NaN


def test_direction_noise_check_separates_coin_from_edge():
    rng = np.random.default_rng(3)

    def mk(hit_p, n=200):
        out = []
        for i in range(n):
            up = rng.random() < 0.5
            right = rng.random() < hit_p
            side = 1 if (up == right) else -1
            out.append(trade(rid=f"d{i}", side=side, signal_ret=0.05 if up else -0.05, dir_prob=0.6 if side > 0 else 0.4))
        return out
    assert S.direction_noise_check(mk(0.5))["noise"] is True
    edge = S.direction_noise_check(mk(0.65))
    assert edge["verdict"] in ("EDGE", "OVERCLAIMS") and edge["hit_rate"] > 0.58
    assert S.direction_noise_check(mk(0.5, 5))["verdict"] == "INSUFFICIENT_DATA"


def test_subsystem_ledger_recovers_the_planted_dominant_failure():
    rng = np.random.default_rng(4)
    led = S.SubsystemLedger()
    for i in range(80):      # 80 timing losses (gap) and 20 clean trades
        gap = 0.05 + rng.random() * 0.03
        end = 1.05 / (1 + gap) - 1
        led.add(trade(rid=f"l{i}", signal_ret=0.05, entry_gap=gap, end_ret_from_fill=end, exit_ret=end, pnl=end - 0.0005, exp_move=0.04))
    for i in range(20):
        led.add(trade(rid=f"w{i}", signal_ret=0.06, entry_gap=0.0, end_ret_from_fill=0.06, exit_ret=0.06, pnl=0.0595, exp_move=0.05))
    assert led.error_rates()["TIMING"] == pytest.approx(0.8) and led.error_rates()["SELECTION"] == 0.0
    ci = led.blame_share_ci(seed=0)
    assert ci["TIMING"][0] > 0.99 and ci["TIMING"][1] > 0.95
    assert led.taught_counts()["TIMING"] == 80 and led.taught_counts()["SELECTION"] == 0
    with pytest.raises(ValueError):
        led.add(trade(rid="l0", signal_ret=0.05))
    assert S.SubsystemLedger().error_rates() == {} and S.SubsystemLedger().blame_share_ci() == {}


def test_order_sensitivity_flags_ambiguous_blame():
    small_wrong = trade(rid="sw", signal_ret=-0.01, end_ret_from_fill=-0.01, exit_ret=-0.01, entry_gap=0.0, pnl=-0.0105, dir_prob=0.6, exp_move=0.07)
    r = S.order_sensitivity([small_wrong])
    assert r["n"] == 1 and r["agreement"] == 0.0 and r["flips"]
    assert S.attribute(small_wrong).ambiguous is True
    assert S.order_sensitivity([])["n"] == 0


def test_diagnostics_on_planted_trades():
    rng = np.random.default_rng(5)
    ts = [trade(rid=f"s{i}", rank_pct=float(rng.random()), exp_move=0.07, signal_ret=float(0.07 * (0.1 + 1.0 * rng.random()) * (1 if rng.random() < .5 else -1)),
                dir_prob=float(0.4 + 0.2 * rng.random())) for i in range(80)]
    sel = S.selection_report(ts)
    assert sel["verdict"] == "OVERPROMISES" and len(sel["table"]) == 5
    assert S.direction_calibration(ts)["n"] == 80
    stops = S.stop_effectiveness([trade(rid="z", stop=0.05, stop_hit=True, stop_fill_ret=-0.12, end_ret_from_fill=0.02)])
    assert stops["gapped_share"] == 1.0 and stops["verdict"] == "STOPS_LEAK"
    assert S.stop_effectiveness([trade()])["verdict"] == "NO_STOPS_HIT"
    assert S.exit_report([trade(mfe=0.05, pnl=-0.02)])["verdict"] == "LEAKY_EXITS" and S.exit_report([trade()])["verdict"] == "NEVER_AHEAD"
    assert S.risk_report([trade(pnl=-0.2, exp_vol=0.03)] * 10)["verdict"] == "TAILS_TOO_FAT"
    curve = S.timing_policy_curve([trade(rid=f"g{i}", entry_gap=0.05, pnl=-0.05) for i in range(5)] + [trade(rid="k", entry_gap=0.0, pnl=0.02)], gaps=(0.02,))
    assert curve[0]["skipped"] == 5 and curve[0]["net_change"] == pytest.approx(0.25)
    assert S.subsystem_scorecard(ts)["n"] == 80


def test_teaching_ledger_needs_independent_periods():
    led = S.TeachingLedger()
    t = trade()
    sig = S.TeachingSignal("r", {"TIMING": 1.0}, ("SELECTION",), False, "x")
    for i in range(30):
        led.add(sig, "p1")                                     # thirty losses, ONE period
    assert led.ready() == []
    for i in range(6):
        led.add(sig, f"q{i}")
    assert led.ready() == [Subsystem.TIMING] and led.snapshot()["label_overrides"] == 0
    with pytest.raises(ValueError):
        S.TeachingLedger(decay=0.0)


# ------------------------------------------------------------------------------------------------ section 24 postmortem
def build(cause=FC.RISK_ERROR, seed=0, uses=(), **kw):
    t, env = F.planted_case(cause, seed)
    return t, env, PM.PostmortemBuilder(code_hash="test").build(t, env, NOW, uses)


@pytest.mark.parametrize("cause", list(F.CAUSES_EXPLAINING) + [FC.UNKNOWN])
def test_every_section24_field_is_populated(cause):
    t, env, pm = build(cause)
    assert pm is not None and not pm.validate() and not PM.audit_postmortem(pm, t)
    for f in PM.REQUIRED_FIELDS:
        v = getattr(pm, f)
        assert v not in (None, "", (), {}) or (f == "what_should_change" and cause == FC.UNKNOWN)
    assert pm.cause == cause


def test_named_cause_says_what_invalidated_the_belief_and_unknown_says_unknown():
    _, _, named = build(FC.REVERSAL)
    assert any("reversed" in s for s in named.invalidating_condition) and named.confidence_in_explanation > 0.3
    _, _, unk = build(FC.UNKNOWN)
    assert unk.invalidating_condition == (PM.UNKNOWN_MARK,) and unk.confidence_in_explanation == 0.0


def test_a_postmortem_only_emits_hypotheses():
    _, _, pm = build(FC.WEAKENING_EFFECT)
    assert pm.what_should_change and all(h.epistemic == Epistemic.HYPOTHESIS and not h.production_effect for h in pm.what_should_change)
    PM.assert_hypothesis_only(pm)
    bad = dataclasses.replace(pm.what_should_change[0], production_effect=True)
    tampered = dataclasses.replace(pm, what_should_change=(bad,))
    with pytest.raises(PM.ProductionWrite):
        PM.assert_hypothesis_only(tampered)
    with pytest.raises(PM.ProductionWrite):
        PM.PostmortemStore().append(tampered)
    with pytest.raises(TypeError):
        PM.assert_hypothesis_only({"apply": "now"})


def test_future_knowledge_is_a_firewall_breach_not_a_story():
    t, env = F.planted_case(FC.RISK_ERROR, 0)
    leak = know("KFUT", learned="2020-03-01")                  # learned after the 2019-12-20 decision
    with pytest.raises(FirewallBreach):
        PM.PostmortemBuilder(code_hash="x").build(t, env, NOW, [PM.KnowledgeUse(leak)])


def test_knowledge_that_should_not_have_influenced_is_flagged_and_gated():
    t, env = F.planted_case(FC.RISK_ERROR, 0)
    uses = [PM.KnowledgeUse(know("KOK")), PM.KnowledgeUse(know("KGATED", epistemic=Epistemic.GATED)),
            PM.KnowledgeUse(know("KSHADOW", promotion=Promotion.SHADOW), weight=0.5),
            PM.KnowledgeUse(know("KANTI", anti={"m_trend": (-1, 1)})), PM.KnowledgeUse(know("KWEAK", reliability=0.1), direction=-1)]
    pm = PM.PostmortemBuilder(code_hash="x").build(t, env, NOW, uses)
    flagged = {m.knowledge_id: m.severity for m in pm.knowledge_should_not_have}
    assert flagged["KGATED"] == "BLOCK" and flagged["KSHADOW"] == "BLOCK" and flagged["KANTI"] == "BLOCK" and flagged["KWEAK"] == "WARN"
    assert "KOK" not in flagged
    assert sum("gate knowledge" in h.statement for h in pm.what_should_change) == 3
    assert not PM.audit_postmortem(pm, t)
    assert set(pm.knowledge_influencing) == {"KOK", "KGATED", "KSHADOW", "KANTI", "KWEAK"}


def test_missing_conditions_come_from_unrecorded_and_ignored_dimensions():
    t, env = F.planted_case(FC.REGIME_CHANGE, 0)
    k = know("KREG", contexts={"m_vol": (-9, 9), "m_breadth": (0, 1)})
    miss = PM.missing_conditions(t, env, [PM.KnowledgeUse(k)])
    assert any("m_breadth" in m and "not recorded" in m for m in miss)
    assert any("m_trend" in m and "no rule conditions on" in m for m in miss)
    assert not any("m_vol" in m and "no rule conditions on" in m for m in miss)


def test_non_meaningful_losses_write_no_postmortem():
    b = PM.PostmortemBuilder(code_hash="x")
    assert b.build(trade(pnl=0.03), None, NOW) is None and b.build(trade(pnl=-0.002), None, NOW) is None
    assert sum(b.skipped.values()) == 2
    cls = F.LossClassifier().classify(trade(pnl=0.03), None, NOW)
    with pytest.raises(ValueError):
        PM.build_postmortem(trade(pnl=0.03), cls, S.attribute(trade(pnl=0.03)), None, NOW, code_hash="x")


def test_store_is_append_only_hash_chained_and_identity_free(tmp_path):
    path = tmp_path / "pm.jsonl"
    store = PM.PostmortemStore(path)
    for c in (FC.RISK_ERROR, FC.REVERSAL, FC.UNKNOWN):
        store.append(build(c)[2])
    assert len(store) == 3 and path.read_text(encoding="utf-8").count("\n") == 3
    store.verify_chain()
    again = PM.PostmortemStore(path)
    assert len(again) == 3
    with pytest.raises(ValueError):
        store.append(build(FC.RISK_ERROR)[2])                  # same pid twice
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text(rows[0].replace("RISK_ERROR", "SELECTION_ERROR") + "\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
    with pytest.raises(FirewallBreach):
        PM.PostmortemStore(path)
    assert store.audit_identity(["AAPL", "2019-12-20"]) == []
    assert store.field_coverage()["believed"] == 1.0 and PM.PostmortemStore().field_coverage()["believed"] == 0.0


def test_hypotheses_are_counted_in_distinct_periods():
    book = PM.HypothesisBook()
    _, _, pm = build(FC.WEAKENING_EFFECT)
    for i in range(30):                                        # thirty postmortems, all from ONE period
        book.add(dataclasses.replace(pm, pid=f"p{i}"), loss=0.05)
    hid = pm.what_should_change[0].hid
    assert book.support(hid) == (30, 1) and book.ready() == []
    for i in range(6):
        book.add(dataclasses.replace(pm, pid=f"q{i}", period=f"per{i}"), loss=0.05)
    assert book.support(hid) == (36, 7) and [r.hypothesis.hid for r in book.ready()] == [hid]
    assert "ready for out-of-sample" in book.summary() and len(PM.HypothesisBook()) == 0


def test_postmortem_roundtrip_and_audit_catches_incoherence():
    t, _, pm = build(FC.REVERSAL)
    back = PM.load_postmortem(pm.to_dict())
    assert back.pid == pm.pid and back.cause == pm.cause and back.confidence_in_explanation == pytest.approx(pm.confidence_in_explanation)
    assert back.what_should_change[0].hid == pm.what_should_change[0].hid
    broken = dataclasses.replace(pm, invalidating_condition=(PM.UNKNOWN_MARK,))
    assert any("invalidated" in e for e in PM.audit_postmortem(broken))
    assert any("different trade" in e for e in PM.audit_postmortem(pm, trade(rid="other")))
    with pytest.raises(Exception):
        PM.load_postmortem({**pm.to_dict(), "cause": "MADE_UP"})


def test_counterfactuals_rank_the_levers():
    t = trade(signal_ret=0.06, entry_gap=0.06, end_ret_from_fill=0.06 / 1.06 * 1.0, exit_ret=-0.03, pnl=-0.0305, mfe=0.05, weight=0.3, target_weight=0.1, stop=0.03, stop_hit=True)
    cfs = PM.counterfactuals(t)
    assert [c.delta for c in cfs] == sorted((c.delta for c in cfs), reverse=True)
    names = {c.name for c in cfs}
    assert {"skip_trade", "no_adverse_gap", "stop_honoured", "sized_to_plan", "take_half_profit_at_mfe", "hold_to_horizon_end"} <= names
    assert all(not c.validate() for c in cfs)


def test_knowledge_involvement_flags_the_guilty_item_only():
    rng = np.random.default_rng(6)
    inv = PM.KnowledgeInvolvement()
    good, bad = know("GOOD"), know("BAD")
    for i in range(400):
        uses = [PM.KnowledgeUse(good)] + ([PM.KnowledgeUse(bad)] if i % 2 == 0 else [])
        lost = rng.random() < (0.7 if i % 2 == 0 else 0.3)
        inv.add(uses, lost)
    tab = {r["knowledge"]: r for r in inv.table()}
    assert tab["BAD"]["flag"] is True and tab["GOOD"]["flag"] is False
    assert PM.KnowledgeInvolvement().table() == []


def test_summary_and_render_over_many_postmortems():
    bodies = [build(c, s)[2].to_dict() for c in (FC.RISK_ERROR, FC.REVERSAL, FC.UNKNOWN) for s in range(2)]
    s = PM.summarize(bodies)
    assert s["n"] == 6 and s["causes"]["RISK_ERROR"]["n"] == 2 and s["unnamed_share"] == pytest.approx(1 / 3)
    txt = PM.render_summary(s)
    assert "IMPLEMENTED - NOT VALIDATED" in txt and "RISK_ERROR" in txt
    assert PM.summarize([]) == {"n": 0} and "no postmortems" in PM.render_summary({"n": 0})
    assert "POSTMORTEM" in build(FC.RISK_ERROR)[2].render()


def test_lesson_kind_bridge_uses_the_existing_vocabulary():
    assert PM.lesson_kind(build(FC.REGIME_CHANGE)[2]) == "regime_misread"
    assert PM.lesson_kind(build(FC.RISK_ERROR)[2]) == "oversized_loser"
    assert PM.lesson_kind(build(FC.MEASUREMENT_ERROR)[2]) is None
    assert PM.lesson_kind(build(FC.FALSE_PATTERN)[2]) == "bad_entry"


# ------------------------------------------------------------------------------------------------ later additions
def test_timing_detector_reads_an_adverse_gap_correctly_for_both_sides():
    """A long is hurt by an upward gap (it pays more); a short by a downward one. Getting the sign wrong blames the wrong trade."""
    long_adverse = trade(side=1, signal_ret=0.06, entry_gap=0.08, end_ret_from_fill=1.06 / 1.08 - 1, exit_ret=1.06 / 1.08 - 1, pnl=-0.0195)
    short_adverse = trade(side=-1, signal_ret=-0.06, entry_gap=-0.08, end_ret_from_fill=0.94 / 0.92 - 1, exit_ret=0.94 / 0.92 - 1, pnl=-0.0225)
    long_fine = trade(side=1, signal_ret=0.06, entry_gap=-0.08, end_ret_from_fill=1.06 / 0.92 - 1, exit_ret=-0.02, pnl=-0.0205)
    for t, expect in ((long_adverse, True), (short_adverse, True), (long_fine, False)):
        r = F.detect_timing(t, F.FailureEnv(), F.FailureParams())
        assert any("gap ate" in e.note for e in r.evidence) is expect


def test_threshold_sensitivity_finds_the_decision_boundary():
    t, env = F.planted_case(FC.SELECTION_ERROR, 0)
    sweep = F.threshold_sensitivity(t, env, NOW, "accept", [0.3, 0.6, 0.9, 0.99])
    assert sweep[0]["cause"] == "SELECTION_ERROR" and sweep[-1]["cause"] == "UNKNOWN" and any(r["flipped"] for r in sweep)
    with pytest.raises(ValueError):
        F.threshold_sensitivity(t, env, NOW, "no_such_field", [1])
    b = F.boundary_cases([F.planted_case(FC.SELECTION_ERROR, 0), F.planted_case(FC.UNKNOWN, 0)], NOW, "accept", [0.3, 0.6, 0.9, 0.99])
    assert b["boundary_share"] == 0.5 and F.boundary_cases([], NOW, "accept", [0.5])["n"] == 0


def test_audit_detectors_gives_one_row_per_detector():
    t, env = F.planted_case(FC.RISK_ERROR, 0)
    rows = F.audit_detectors(F.LossClassifier(), t, env, NOW)
    assert [r["id"] for r in rows] == ["D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09"]
    risk = next(r for r in rows if r["detector"] == "risk")
    assert risk["ran"] and "RISK_ERROR" in risk["causes"] and risk["max_support"] > 0.9
    with pytest.raises(FirewallBreach):
        F.audit_detectors(F.LossClassifier(), t, env, "2019-12-30")


def test_averaged_decomposition_is_order_free_and_still_additive():
    t = trade(signal_ret=-0.01, end_ret_from_fill=-0.01, exit_ret=-0.01, entry_gap=0.0, pnl=-0.0105, dir_prob=0.6, exp_move=0.07)
    av = S.decompose_averaged(t)
    d = S.decompose(t)
    assert sum(v for k, v in av.items() if k != "residual") + d.promise + av["residual"] == pytest.approx(d.pnl)
    assert av["selection"] == pytest.approx((S.decompose(t).selection + S.decompose(t, order="direction_first").selection) / 2)


def test_error_rate_drift_detects_a_subsystem_that_breaks_midway():
    rng = np.random.default_rng(11)
    ts = []
    for i in range(120):
        broken = i >= 60 and rng.random() < 0.8
        if broken:
            gap = 0.06
            end = 1.05 / (1 + gap) - 1
            ts.append(trade(rid=f"d{i}", signal_ret=0.05, entry_gap=gap, end_ret_from_fill=end, exit_ret=end, pnl=end - 0.0005, exp_move=0.04))
        else:
            ts.append(trade(rid=f"d{i}", signal_ret=0.06, entry_gap=0.0, end_ret_from_fill=0.06, exit_ret=0.06, pnl=0.0595, exp_move=0.05))
    r = S.error_rate_drift(ts, seed=1)
    assert r["subsystems"]["TIMING"]["drifting"] and r["subsystems"]["TIMING"]["diff"] > 0.6 and not r["subsystems"]["SELECTION"]["drifting"]
    assert S.error_rate_drift(ts[:10])["verdict"] == "INSUFFICIENT_DATA"


def test_blame_by_context_shows_conditional_failure():
    rng = np.random.default_rng(12)
    ts = []
    for i in range(90):
        hi = i % 3 == 2                                         # timing losses only when volatility is high
        vol = float(3.0 + rng.random()) if hi else float(rng.random())
        if hi:
            gap = 0.07
            end = 1.05 / (1 + gap) - 1
            ts.append(trade(rid=f"c{i}", signal_ret=0.05, entry_gap=gap, end_ret_from_fill=end, exit_ret=end, pnl=end - 0.0005, exp_move=0.04, context={"m_vol": vol}))
        else:
            ts.append(trade(rid=f"c{i}", signal_ret=0.004, entry_gap=0.0, end_ret_from_fill=0.004, exit_ret=-0.02, pnl=-0.0205, exp_move=0.07, context={"m_vol": vol}))
    r = S.blame_by_context(ts, "m_vol", bins=3)
    lead = [max(b["blame"].items(), key=lambda kv: kv[1])[0] for b in r["table"]]
    assert r["primary_changes_with_context"] and lead[-1] == "TIMING" and lead[0] == "SELECTION"
    assert S.blame_by_context(ts, "m_nothing")["verdict"] == "INSUFFICIENT_DATA"


def test_co_failure_and_improvement_potential_and_focus():
    ts = [S.planted_trade(f, s) for f in (Subsystem.TIMING, Subsystem.SELECTION, Subsystem.EXIT) for s in range(12)]
    ts = [dataclasses.replace(t, rid=f"{t.rid}-{i}") for i, t in enumerate(ts)]
    cf = S.co_failure(ts)
    assert cf["n"] == 36 and 0.0 <= cf["max_abs"] <= 1.0
    ip = S.improvement_potential(ts)
    assert ip["largest"] in ("TIMING", "SELECTION", "EXIT") and all(v >= 0 for v in ip["uplift"].values())
    focus = S.subsystem_focus(ts, seed=1)
    assert focus and focus[0]["vs_mean"] >= focus[-1]["vs_mean"]
    assert S.subsystem_focus(ts[:5]) == [] and S.improvement_potential([])["verdict"] == "INSUFFICIENT_DATA"
    assert "subsystem scorecard" in S.render_scorecard(S.subsystem_scorecard(ts))


def test_the_decomposition_recovers_every_planted_fault():
    r = S.decomposition_selfcheck(seeds=(0, 1, 2, 3, 4))
    assert r["recovery"] == 1.0 and set(r["by_fault"]) == {s.value for s in Subsystem}
    with pytest.raises(ValueError):
        S.planted_trade("NOPE")


def test_exit_alternatives_sizing_and_reports():
    ts = [trade(rid=f"x{i}", mfe=0.08, pnl=-0.02) for i in range(10)] + [trade(rid=f"y{i}", mfe=0.01, pnl=0.01) for i in range(10)]
    alt = S.exit_alternatives(ts, targets=(0.05,))
    assert alt[0]["touched"] == 10 and alt[0]["uplift_upper_bound"] > 0.02 and S.exit_alternatives([trade()]) == []
    sized = [trade(rid=f"z{i}", weight=0.05 + 0.01 * i, target_weight=0.1, pnl=-0.003 * i if i > 10 else 0.002) for i in range(30)]
    rep = S.sizing_report(sized)
    assert rep["weight_pnl_corr"] < 0 and rep["n"] == 30 and S.sizing_report(sized[:5])["verdict"] == "INSUFFICIENT_DATA"


def test_params_validation_and_explain_and_tables():
    assert S.validate_params(S.SeparationParams()) == []
    assert len(S.validate_params(S.SeparationParams(min_blame=0.0, noise_alpha=0.9, boot=5, min_direction_n=2))) == 4
    d = S.decompose(S.planted_trade(Subsystem.TIMING, 0))
    txt = S.explain_decomposition(d)
    assert "the fill (overnight gap) cost money" in txt and "promised" in txt
    tab = S.attribution_table([S.planted_trade(f, 0) for f in Subsystem])
    assert list(tab["primary"]) == [S.attribute(S.planted_trade(f, 0)).primary.value for f in Subsystem]
    led = S.SubsystemLedger()
    for i, f in enumerate(list(Subsystem) * 4):
        led.add(dataclasses.replace(S.planted_trade(f, i), rid=f"led{i}"))
    assert 0.2 <= S.blame_concentration(led)["herfindahl"] <= 1.0 and S.learning_targets(led, min_taught=2)
    empty = S.blame_concentration(S.SubsystemLedger())["herfindahl"]
    assert empty != empty


def test_label_vs_arithmetic_measures_wrong_teaching():
    clf = F.LossClassifier()
    pairs = []
    for i in range(10):    # an adverse-gap loss that a label blames on SELECTION: label says SELECTION, arithmetic says TIMING
        t = S.planted_trade(Subsystem.TIMING, i)
        c = clf.classify(t, F.FailureEnv(), NOW)
        pairs.append((t, F.Classification(**{**c.__dict__, "cause": FC.SELECTION_ERROR, "unknown_state": None, "confidence": 0.8})))
    r = S.label_vs_arithmetic(pairs)
    assert r["n"] == 10 and r["agreement"] == 0.0 and r["table"]["SELECTION_ERROR"]["TIMING"] == 10


def test_selection_of_losses_for_review_prefers_big_and_novel():
    big_common = [trade(rid=f"bc{i}", pnl=-0.10, pattern_ids=("P",)) for i in range(5)]
    small_novel = trade(rid="sn", pnl=-0.06, pattern_ids=("Q",))
    picked = PM.select_for_postmortem(big_common + [small_novel, trade(rid="win", pnl=0.05)], budget=3)
    assert "sn" in [t.rid for t in picked] and "win" not in [t.rid for t in picked] and len(picked) == 3
    assert PM.select_for_postmortem([], 5) == [] and PM.select_for_postmortem(big_common, 0) == []


def test_recurrence_separates_a_systematic_mode_from_one_event():
    _, _, base_pm = build(FC.WEAKENING_EFFECT)
    systematic = [dataclasses.replace(base_pm, pid=f"s{i}", period=f"p{i}") for i in range(5)]
    one_event = [dataclasses.replace(build(FC.RISK_ERROR)[2], pid=f"e{i}", period="only") for i in range(5)]
    rec = PM.recurrence(systematic + one_event, min_periods=3)
    assert len(rec) == 1 and rec[0]["cause"] == "WEAKENING_EFFECT" and rec[0]["n_periods"] == 5 and rec[0]["streak"] == 5
    assert PM.recurrence([]) == []


def test_hypothesis_outcomes_keep_failures_and_block_reproposal():
    h = build(FC.WEAKENING_EFFECT)[2].what_should_change[0]
    out = PM.HypothesisOutcomes()
    assert out.propose(h) and out.state(h.hid) == "PROPOSED"
    out.record(h.hid, "FAIL", "oos lift 0.9 on 12 periods")
    assert out.state(h.hid) == "TESTED_FAIL" and out.propose(h) is False
    with pytest.raises(ValueError):
        out.reopen(h.hid, "")
    out.reopen(h.hid, "new regime data")
    assert out.state(h.hid) == "PROPOSED" and out.history(h.hid)[-1].startswith("REOPENED")
    with pytest.raises(KeyError):
        out.record("nope", "PASS", "x")
    with pytest.raises(ValueError):
        out.record(h.hid, "MAYBE", "x")
    with pytest.raises(ValueError):
        out.record(h.hid, "PASS", "")
    assert out.counts()["PROPOSED"] == 1


def test_export_replay_coverage_and_conflicts():
    t, env, pm = build(FC.REVERSAL)
    md = PM.export_markdown(pm)
    for heading in ("What was believed", "Supporting evidence", "Knowledge that should not have", "Condition that invalidated the belief", "Confidence in this explanation"):
        assert heading in md
    assert PM.replay_check(pm, t, env, NOW) == []
    assert PM.replay_check(pm, dataclasses.replace(t, pnl=-0.0501), env, NOW)          # a different trade cannot reproduce it
    cov = PM.loss_coverage([t, trade(rid="unreviewed", pnl=-0.5)], [pm])
    assert cov["n_losses"] == 2 and cov["coverage"] == 0.5 and cov["by_size"][0]["coverage"] == 0.0
    book = PM.HypothesisBook()
    a = F.Hypothesis("h1", "gate pattern P1 while flat", Subsystem.SELECTION, DecisionEffect.PATTERN_WEIGHTING, FC.REVERSAL, ("t",), target="P1")
    b = F.Hypothesis("h2", "promote pattern P1 in calm regimes", Subsystem.SELECTION, DecisionEffect.PATTERN_WEIGHTING, FC.REVERSAL, ("t",), target="P1")
    for h in (a, b):
        book._h[h.hid] = PM.HypothesisRecord(h)
    assert PM.hypothesis_conflicts(book) == [("P1", "h1", "h2")]


def test_store_views(tmp_path):
    store = PM.PostmortemStore(tmp_path / "s.jsonl")
    uses = [PM.KnowledgeUse(know("KG", epistemic=Epistemic.GATED))]
    for i, c in enumerate((FC.RISK_ERROR, FC.REVERSAL, FC.RISK_ERROR)):
        t, env = F.planted_case(c, i, rid=f"sv{i}")
        store.append(PM.PostmortemBuilder(code_hash="x").build(t, env, NOW, uses))
    assert len(PM.query_store(store, cause="RISK_ERROR")) == 2 and len(PM.query_store(store, min_confidence=0.99)) == 0
    hs = PM.hypotheses_in_store(store)
    assert hs and all(v["n_periods"] >= 1 for v in hs.values())
    assert PM.knowledge_flag_history(store)["KG"]["BLOCK"] == 3
    assert PM.severity_counts(store)["BLOCK"] == 3 and sum(PM.period_counts(store).values()) == 3
    assert set(PM.confidence_by_cause(store)) == {"RISK_ERROR", "REVERSAL"} and PM.hypothesis_effect_mix(store)
    first = PM.load_postmortem(store.bodies()[0]).what_should_change[0]
    assert len(PM.dedupe_hypotheses([first] * 3)) == 1
