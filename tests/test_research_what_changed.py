"""Tests for engine/research/pattern_change.py (C68 checklist H) and engine/research/what_changed.py (checklists J, K, V).
Synthetic data only. Every mechanism has a planted case it must catch, a null case where it must find nothing, and an empty case."""
import copy
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import DecisionEffect, FirewallBreach, Health, Lifecycle
from engine.research import knowability as KN
from engine.research import pattern_change as PC
from engine.research import replication as RP
from engine.research import what_changed as WC
from engine.research.core import Availability, Knowability

IDX = pd.bdate_range("2001-01-01", periods=420)


def frame(eff, **cols):
    d = pd.DataFrame({"effect": np.asarray(eff, float)}, index=IDX[:len(eff)])
    for k, v in cols.items():
        d[k] = v
    return d


def rng_(seed):
    return np.random.default_rng(seed)


# ==================================================================================================== checklist H
def test_normal_pattern_is_not_flagged():
    f = frame(rng_(1).normal(0.01, 0.02, 260))
    v = PC.classify("p", f, IDX[260])
    assert v.change == PC.ChangeClass.NORMAL_VARIANCE and v.action == PC.Action.KEEP and v.health == Health.HEALTHY
    assert not v.needs_investigation


def test_structural_step_is_caught_and_never_deleted():
    r = rng_(2)
    f = frame(np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.02, 0.02, 60)])
    v = PC.classify("p", f, IDX[260])
    assert v.change == PC.ChangeClass.STRUCTURAL_CHANGE and v.deterioration == "ABRUPT"
    assert v.action == PC.Action.INVESTIGATE and v.needs_investigation
    assert v.delete_allowed is False and v.health == Health.BROKEN


def test_gradual_decay_is_weakening():
    r = rng_(3)
    f = frame(np.r_[r.normal(0.01, 0.02, 150), np.linspace(0.01, -0.03, 110) + r.normal(0, 0.02, 110)])
    v = PC.classify("p", f, IDX[260])
    assert v.change == PC.ChangeClass.WEAKENING and v.profile.degradation_rate < 0
    assert v.action == PC.Action.INVESTIGATE


def test_strengthening_pattern():
    r = rng_(4)
    f = frame(np.r_[r.normal(0.0, 0.02, 200), r.normal(0.03, 0.02, 60)])
    v = PC.classify("p", f, IDX[260])
    assert v.change == PC.ChangeClass.STRENGTHENING and v.profile.z > 2 and v.action == PC.Action.KEEP


def test_regime_specific_failure_names_the_regime():
    r = rng_(5)
    reg = np.where(np.arange(260) % 2 == 0, "bull", "bear")
    e = r.normal(0.01, 0.02, 260)
    e[200:][reg[200:] == "bear"] -= 0.05
    v = PC.classify("p", frame(e, regime=reg), IDX[260])
    assert v.change == PC.ChangeClass.REGIME_SPECIFIC_FAILURE and v.culprit == ("regime", "bear")
    assert v.action == PC.Action.INVESTIGATE
    v2 = PC.classify("p", frame(e, regime=reg), IDX[260], investigated=True)
    assert v2.action == PC.Action.REDUCE_IN_CONTEXT              # only after investigation, and only in context


def test_obsolescence_needs_long_unrecovered_failure_and_investigation_before_retirement():
    r = rng_(6)
    f = frame(np.r_[r.normal(0.01, 0.02, 120), r.normal(-0.02, 0.02, 140)])
    v = PC.classify("p", f, IDX[260])
    assert v.change == PC.ChangeClass.OBSOLESCENCE and v.failing_rows >= 104
    assert v.action == PC.Action.INVESTIGATE                        # never straight to retirement
    assert PC.classify("p", f, IDX[260], investigated=True).action == PC.Action.PROPOSE_RETIREMENT
    short = frame(np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.02, 0.02, 60)])
    assert PC.classify("p", short, IDX[260]).change != PC.ChangeClass.OBSOLESCENCE     # a recent step is not obsolescence yet


def test_returning_pattern_after_a_failure_spell():
    r = rng_(7)
    f = frame(np.r_[r.normal(0.012, 0.02, 120), r.normal(-0.03, 0.02, 45), r.normal(0.02, 0.02, 45)])
    v = PC.classify("p", f, IDX[210])
    assert v.change == PC.ChangeClass.RETURNING and v.action == PC.Action.MONITOR and v.health == Health.RECOVERING


def test_isolated_outlier_is_noise_not_failure():
    r = rng_(8)
    e = r.normal(0.01, 0.02, 260)
    e[250] = -0.35
    v = PC.classify("p", frame(e), IDX[260])
    assert v.change in (PC.ChangeClass.NOISE, PC.ChangeClass.NORMAL_VARIANCE) and v.change not in PC.FAILING


def test_empty_and_short_frames_are_insufficient_not_guessed():
    empty = pd.DataFrame({"effect": pd.Series(dtype=float)}, index=pd.DatetimeIndex([]))
    assert PC.classify("p", empty, IDX[100]).change == PC.ChangeClass.INSUFFICIENT_EVIDENCE
    assert PC.classify("p", frame(rng_(9).normal(0, 0.02, 20)), IDX[20]).action == PC.Action.COLLECT_DATA
    with pytest.raises(ValueError):
        PC.classify("p", pd.DataFrame({"x": [1.0]}, index=IDX[:1]), IDX[10])
    assert PC.step(PC.PatternChangeState(), IDX[100], {}).verdicts == ()


def test_verdict_at_t_is_identical_when_the_future_is_scrambled():
    r = rng_(10)
    f = frame(np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.02, 0.02, 60)])
    now = IDX[200]
    base = PC.classify("p", f, now)
    g = f.copy()
    fut = g.index >= now
    g.loc[fut, "effect"] = rng_(99).permutation(g.loc[fut, "effect"].to_numpy()) * 40 + 3.0
    again = PC.classify("p", g, now)
    assert base.record_id() == again.record_id() and base.profile.z == again.profile.z
    with pytest.raises(FirewallBreach):
        PC.causal_slice(g, now, strict=True)
    assert len(PC.causal_slice(g, IDX[260])) == 260 and IDX[259] in PC.causal_slice(g, IDX[260]).index


def test_profile_carries_every_checklist_dimension():
    r = rng_(11)
    n = 260
    f = frame(r.normal(0.01, 0.02, n), regime=np.where(np.arange(n) % 3 == 0, "bull", "bear"), volatility=np.where(np.arange(n) % 2 == 0, "lo", "hi"),
              sector=np.where(np.arange(n) % 4 < 2, "tech", "util"), hold=np.where(np.arange(n) % 5 < 3, 5, 10),
              confidence=r.uniform(0.3, 0.9, n), predicted=np.full(n, 0.03))
    p = PC.profile("p", f, IDX[n])
    assert {c.dimension for c in p.cells} == set(PC.DIMENSIONS)
    assert p.cell("regime", "bull") is not None and p.cell("holding_period", "5") is not None
    assert p.error_direction == "OVERPREDICTED" and p.error_mean < 0 and 0.5 < p.error_over_share <= 1.0
    assert p.confidence_gap is not None and p.error_mae > 0
    assert p.hist_reliability is not None and p.recent_reliability is not None
    assert PC.profile("p", f.iloc[:10], IDX[10]).error_direction == "UNMEASURED"


def test_degradation_and_recovery_rates():
    assert PC.degradation_rate(np.linspace(0.02, -0.02, 40), 26) < -0.001
    assert PC.degradation_rate(np.ones(3), 26) == 0.0
    r = rng_(12)
    up = np.r_[r.normal(0.012, 0.02, 120), r.normal(-0.03, 0.02, 45), r.normal(0.02, 0.02, 60)]
    assert PC.recovery_rate(up, PC.PARAMS) > 0
    assert math.isnan(PC.recovery_rate(r.normal(0.01, 0.02, 200), PC.PARAMS))


def test_null_patterns_rarely_false_alarm():
    res = PC.false_alarm_rate(50, 260, seed=21)
    assert res["rate"] <= 0.10, res


def test_state_reports_needed_investigations_and_tracks_flapping():
    r = rng_(13)
    frames = {"ok": frame(r.normal(0.01, 0.02, 260)), "broken": frame(np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.02, 0.02, 60)])}
    st = PC.PatternChangeState()
    rep = PC.step(st, IDX[260], frames)
    assert rep.investigations_needed == ("broken",) and rep.counts["STRUCTURAL_CHANGE"] == 1
    st.mark_investigated("broken")
    rep2 = PC.step(st, IDX[261], frames)
    assert rep2.investigations_needed == () and st.stable_class("broken") == PC.ChangeClass.STRUCTURAL_CHANGE
    assert st.flap_count("broken") == 0 and not PC.verdict_table(rep2).empty


def test_health_monitor_cross_check_agrees_on_a_clear_break():
    r = rng_(14)
    f = frame(np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.03, 0.02, 60)])
    v = PC.classify("p", f, IDX[260])
    chk = PC.health_cross_check("p", f, IDX[260], v)
    assert chk["our_failing"] is True and "health" in chk


# ==================================================================================================== checklist J
D0 = pd.Timestamp("2005-06-01")
H_IDX = pd.bdate_range(end=D0, periods=300)
P_IDX = pd.bdate_range(D0 + pd.Timedelta(days=1), periods=5)
A_IDX = pd.bdate_range(P_IDX[-1] + pd.Timedelta(days=1), periods=12)
DECISION, MATURED = str(D0.date()), str(P_IDX[-1].date())
PAT_IDX = pd.bdate_range(end=D0 - pd.Timedelta(days=1), periods=260)


def series(idx, vals):
    return pd.Series(np.asarray(vals, float), index=idx)


def make_case(kind="null", seed=0, predicted=0.05):
    r = rng_(seed)
    mkt_h = series(H_IDX, r.normal(0.0004, 0.008, len(H_IDX)))
    sec_h = series(H_IDX, mkt_h.to_numpy() + r.normal(0, 0.005, len(H_IDX)))
    stk_h = series(H_IDX, 1.0 * mkt_h.to_numpy() + r.normal(0, 0.012, len(H_IDX)))
    mkt_p = series(P_IDX, r.normal(0.0004, 0.008, 5))
    sec_p = series(P_IDX, mkt_p.to_numpy() + r.normal(0, 0.003, 5))
    stk_p = series(P_IDX, mkt_p.to_numpy() + r.normal(0, 0.004, 5))
    peers = pd.DataFrame({"realised": r.normal(0.0, 0.01, 12) + _cum(stk_p) * 0.0})
    pats, combos, after, policy_exit = {}, {}, None, None
    realised_shift = 0.0
    if kind == "market":
        mkt_p = series(P_IDX, np.full(5, -0.03))
        sec_p = series(P_IDX, mkt_p.to_numpy() + r.normal(0, 0.002, 5))
        stk_p = series(P_IDX, mkt_p.to_numpy() + r.normal(0, 0.002, 5))
        peers = pd.DataFrame({"realised": _cum(stk_p) + r.normal(0, 0.004, 12)})
    elif kind == "sector":
        sec_p = series(P_IDX, np.full(5, -0.03))
        stk_p = series(P_IDX, sec_p.to_numpy() * 0.9 + r.normal(0, 0.001, 5))
        peers = pd.DataFrame({"realised": r.normal(0.0, 0.01, 12)})
    elif kind == "cross":
        stk_p = series(P_IDX, np.full(5, -0.02))
        peers = pd.DataFrame({"realised": r.normal(-0.10, 0.01, 12)})
    elif kind == "stock":
        v = np.zeros(5)
        v[2] = -0.16
        stk_p = series(P_IDX, mkt_p.to_numpy() + v)
        peers = pd.DataFrame({"realised": r.normal(0.0, 0.01, 12)})
    elif kind == "pattern":
        eff = np.r_[r.normal(0.05, 0.02, 200), r.normal(0.0, 0.02, 60)]
        pats["pat_a"] = pd.DataFrame({"effect": eff}, index=PAT_IDX)
        stk_p = series(P_IDX, mkt_p.to_numpy() * 0.5 + np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        stk_p = series(P_IDX, np.full(5, (1.0 + 0.0) ** (1 / 5) - 1))
    elif kind in ("timing", "exit"):
        pass
    if kind == "timing":
        stk_p = series(P_IDX, np.full(5, 0.002))
        after = series(A_IDX, np.r_[np.full(3, 0.0), np.full(9, 0.012)])
    if kind == "exit":
        stk_p = series(P_IDX, np.array([0.03, 0.04, 0.03, -0.04, -0.06]))
    realised = _cum(stk_p)
    return WC.ErrorCase(f"case-{kind}-{seed}", DECISION, MATURED, predicted, realised, stk_h, stk_p, mkt_h, mkt_p, sec_h, sec_p, peers, 0.0,
                        pats, combos, after, policy_exit)


def _cum(s):
    return float(np.prod(1.0 + np.asarray(s, float)) - 1.0)


NOW = "2005-07-15"


def test_insignificant_error_opens_no_investigation():
    c = make_case("null", 1, predicted=0.0)
    c.realised = 0.002
    inv = WC.investigate(c, NOW)
    assert not inv.significant and inv.conclusion is None and inv.validate() == []
    assert "no investigation" in WC.render(inv)


def test_market_shock_is_found_at_level_one():
    inv = WC.investigate(make_case("market", 2), NOW)
    assert inv.significant and WC.Level.MARKET in inv.fired()
    assert inv.conclusion.level == WC.Level.MARKET and inv.finding(WC.Level.MARKET).share > 0.5
    assert not inv.finding(WC.Level.STOCK).fired
    assert inv.validate() == [] and len(inv.findings) == 10


def test_sector_shock_beyond_the_market_is_found_at_level_two():
    inv = WC.investigate(make_case("sector", 3), NOW)
    assert WC.Level.SECTOR in inv.fired() and not inv.finding(WC.Level.MARKET).fired
    assert inv.conclusion.level == WC.Level.SECTOR


def test_cross_section_peers_erring_together_is_level_three():
    inv = WC.investigate(make_case("cross", 4), NOW)
    assert inv.finding(WC.Level.CROSS_SECTION).fired and inv.finding(WC.Level.CROSS_SECTION).z >= 0.7
    assert inv.conclusion.level in (WC.Level.CROSS_SECTION, WC.Level.STOCK)


def test_single_stock_extreme_day_is_level_four_not_market():
    inv = WC.investigate(make_case("stock", 5), NOW)
    assert inv.finding(WC.Level.STOCK).fired and inv.finding(WC.Level.STOCK).qualifier == "extreme_day"
    assert not inv.finding(WC.Level.MARKET).fired and inv.conclusion.level == WC.Level.STOCK


def test_pattern_change_is_found_and_was_detectable_before_the_outcome():
    inv = WC.investigate(make_case("pattern", 6), NOW)
    f = inv.finding(WC.Level.PATTERN)
    assert f.fired and f.availability == Availability.KNOWN_BEFORE_EVENT and f.lead is not None
    assert inv.knowability.klass == WC.FiveWay.PREDICTABLE_BUT_MISSED
    assert inv.conclusion.detectability.lead_rows == f.lead


def test_pattern_detection_uses_only_data_before_the_decision():
    c = make_case("pattern", 6)
    base = WC.investigate(c, NOW)
    c2 = copy.deepcopy(c)
    for k, fr in c2.pattern_frames.items():
        extra = pd.DataFrame({"effect": np.full(20, 9.0)}, index=pd.bdate_range(D0 + pd.Timedelta(days=1), periods=20))
        c2.pattern_frames[k] = pd.concat([fr, extra])
    c2.stock_hist = pd.concat([c2.stock_hist, pd.Series(50.0, index=P_IDX)])
    c2.market_hist = pd.concat([c2.market_hist, pd.Series(-50.0, index=P_IDX)])
    again = WC.investigate(c2, NOW)
    assert [dataclasses.replace(f, evidence=()) for f in base.findings] == [dataclasses.replace(f, evidence=()) for f in again.findings]
    assert base.conclusion.cause_key == again.conclusion.cause_key


def _combo_case(recent_mean, seed=40):
    r = rng_(seed)
    c = make_case("null", seed)
    for k in ("pat_a", "pat_b"):
        c.pattern_frames[k] = pd.DataFrame({"effect": r.normal(0.05, 0.02, 260)}, index=PAT_IDX)
    cidx = pd.bdate_range(end=D0 - pd.Timedelta(days=1), periods=40)
    c.combos = {"pat_a|pat_b": pd.Series(np.r_[r.normal(0.05, 0.02, 32), r.normal(recent_mean, 0.01, 8)], index=cidx)}
    c.realised = 0.0
    return c


def test_interaction_level_catches_a_combination_that_stopped_working_while_its_parts_are_healthy():
    inv = WC.investigate(_combo_case(-0.03), NOW)
    f = inv.finding(WC.Level.INTERACTION)
    assert f.fired and f.z <= -2 and not inv.finding(WC.Level.PATTERN).fired
    assert inv.conclusion.level == WC.Level.INTERACTION and "combination" in inv.conclusion.cause
    null = WC.investigate(_combo_case(0.05, 41), NOW)
    assert null.finding(WC.Level.INTERACTION).measured and not null.finding(WC.Level.INTERACTION).fired
    assert not WC.investigate(make_case("market", 2), NOW).finding(WC.Level.INTERACTION).measured        # no combos supplied


def test_timing_needs_hindsight_and_is_marked_as_such():
    c = make_case("timing", 7)
    inv = WC.investigate(c, NOW)
    f = inv.finding(WC.Level.TIMING)
    assert f.fired and f.qualifier == "late" and f.availability == Availability.KNOWN_ONLY_AFTER_EVENT
    c.after_path = None
    assert not WC.investigate(c, NOW).finding(WC.Level.TIMING).measured


def test_exit_level_measures_the_path_and_changes_nothing():
    c = make_case("exit", 8)
    before = copy.deepcopy(c)
    inv = WC.investigate(c, NOW)
    f = inv.finding(WC.Level.EXIT)
    assert f.fired and f.qualifier == "peak_given_back" and "sets no exit" in f.evidence[0]
    assert inv.conclusion.model_change.applied is False
    assert c.predicted == before.predicted and c.stock_path.equals(before.stock_path)      # the investigation mutates nothing
    hi = dataclasses.replace(c, predicted=0.09)                   # perturbing the target moves only the evaluation, never a decision object
    assert WC.investigate(hi, NOW).conclusion.model_change.applied is False


def test_unexplained_error_stays_unexplained():
    inv = WC.investigate(dataclasses.replace(make_case("null", 9), realised=0.01), NOW)          # error -0.04: significant, no level fires
    assert inv.significant and inv.fired() == ()
    assert inv.conclusion.level is None and inv.conclusion.cause_key == "L0:unexplained"
    assert inv.knowability.klass == WC.FiveWay.CURRENTLY_UNEXPLAINED and inv.conclusion.model_change.effect == DecisionEffect.NONE
    assert inv.conclusion.new_knowledge.startswith("UNEXPLAINED_KEPT_OPEN") and inv.validate() == []
    big = WC.investigate(dataclasses.replace(make_case("null", 9), realised=-0.06), NOW)          # a huge residual is described, not called knowable
    assert big.conclusion.level == WC.Level.STOCK and big.knowability.klass == WC.FiveWay.CURRENTLY_UNEXPLAINED


# ==================================================================================================== checklist K
def planted(kind, seed=0):
    return KN.classify_move(KN.planted_inputs(kind, seed))


@pytest.mark.parametrize("kind,want", [
    ("PREDICTABLE", WC.FiveWay.PREDICTABLE_BUT_MISSED),
    ("POTENTIALLY_PREDICTABLE", WC.FiveWay.PARTIALLY_PREDICTABLE),
    ("WEAKLY_PREDICTABLE", WC.FiveWay.PARTIALLY_PREDICTABLE),
    ("EXTERNALLY_CAUSED", WC.FiveWay.GENUINELY_UNKNOWABLE),
    ("INFORMATIONALLY_UNAVAILABLE", WC.FiveWay.GENUINELY_UNKNOWABLE),
    ("UNKNOWN", WC.FiveWay.CURRENTLY_UNEXPLAINED),
])
def test_five_way_mapping_follows_the_assessment(kind, want):
    a = planted(kind)
    kv = WC.classify_knowability([], a)
    assert a.classification.value == kind                       # the planted generator produced the class we asked for
    assert kv.klass == want


def test_data_failure_is_excluded_not_explained():
    a = planted("DATA_FAILURE")
    kv = WC.classify_knowability([], a)
    assert a.classification == Knowability.DATA_FAILURE and kv.excluded_data_failure and kv.klass == WC.FiveWay.CURRENTLY_UNEXPLAINED


def test_new_condition_class_needs_an_already_replicated_condition():
    a = planted("WEAKLY_PREDICTABLE")
    fake = lambda st: type("A", (), {"status": st})()
    assert WC.classify_knowability([], a, {"L5:x": fake(RP.Status.PARTIAL)}, "L5:x").klass != WC.FiveWay.PREDICTABLE_UNDER_NEW_CONDITION
    ok = WC.classify_knowability([], a, {"L5:x": fake(RP.Status.REPLICATED)}, "L5:x")
    assert a.classification == Knowability.WEAKLY_PREDICTABLE and ok.klass == WC.FiveWay.PREDICTABLE_UNDER_NEW_CONDITION
    unk = planted("EXTERNALLY_CAUSED")
    assert WC.classify_knowability([], unk, {"L5:x": fake(RP.Status.REPLICATED)}, "L5:x").klass == WC.FiveWay.GENUINELY_UNKNOWABLE


def test_unknowable_outcome_is_preserved_and_proposes_nothing():
    a = planted("EXTERNALLY_CAUSED")
    c = make_case("null", 10)
    c.realised, c.knowability = 0.005, a
    inv = WC.investigate(c, NOW)
    assert a.classification == Knowability.EXTERNALLY_CAUSED and inv.finding(WC.Level.EXTERNAL).fired
    assert inv.finding(WC.Level.EXTERNAL).availability == Availability.UNAVAILABLE and inv.conclusion.level == WC.Level.EXTERNAL
    assert inv.knowability.klass == WC.FiveWay.GENUINELY_UNKNOWABLE
    cc = inv.conclusion
    assert cc.new_knowledge.startswith("UNKNOWABLE_PRESERVED") and cc.model_change.effect == DecisionEffect.NONE and cc.test.kind == "NONE"
    assert WC.plan_test(inv, NOW) is None and inv.validate() == []


def test_model_failure_level_fires_when_information_existed():
    a = planted("PREDICTABLE")
    c = make_case("null", 11)
    c.realised, c.knowability = 0.005, a
    inv = WC.investigate(c, NOW)
    assert a.classification == Knowability.PREDICTABLE and inv.finding(WC.Level.MODEL_FAILURE).fired
    assert inv.finding(WC.Level.MODEL_FAILURE).availability == Availability.KNOWN_BEFORE_EVENT


# ==================================================================================================== the chain
def test_every_investigation_ends_in_the_full_chain():
    inv = WC.investigate(make_case("market", 12), NOW)
    c = inv.conclusion
    assert c.chain() == ("CAUSE", "EVIDENCE", "PRE-OUTCOME DETECTABILITY", "REPEATABILITY", "NEW KNOWLEDGE", "MODEL CHANGE", "TEST")
    assert c.cause and c.evidence and c.detectability and c.repeatability and c.new_knowledge and c.model_change and c.test
    assert c.model_change.applied is False and c.model_change.gated_on == "TEST" and len(c.test.steps) == 11
    text = WC.render(inv)
    assert all(k in text for k in ("CAUSE", "EVIDENCE", "DETECTABLE", "REPEATABLE", "NEW KNOWLEDGE", "MODEL CHANGE", "TEST"))


def test_validate_catches_planted_chain_defects():
    inv = WC.investigate(make_case("market", 12), NOW)
    bad = dataclasses.replace(inv, conclusion=dataclasses.replace(inv.conclusion, evidence=()))
    assert any("EVIDENCE" in e for e in bad.validate())
    leaky = dataclasses.replace(inv, conclusion=dataclasses.replace(inv.conclusion, cause="it happened on 2008-09-15"))
    assert any("date or year" in e for e in leaky.validate())
    applied = dataclasses.replace(inv, conclusion=dataclasses.replace(inv.conclusion, model_change=dataclasses.replace(inv.conclusion.model_change, applied=True)))
    assert any("PROPOSE" in e for e in applied.validate())
    unk = WC.investigate(dataclasses.replace(make_case("null", 10), realised=-0.07, knowability=planted("EXTERNALLY_CAUSED")), NOW)
    forced = dataclasses.replace(unk, conclusion=dataclasses.replace(unk.conclusion, model_change=WC.ModelChange(None, DecisionEffect.EXIT, "change exits")))
    assert any("unknowable" in e for e in forced.validate())
    assert dataclasses.replace(inv, conclusion=None).validate()


def test_immature_or_malformed_cases_are_refused():
    c = make_case("market", 13)
    with pytest.raises(FirewallBreach):
        WC.investigate(c, MATURED)                              # matures ON now: not yet known
    bad = dataclasses.replace(c, stock_path=c.stock_path * np.nan)
    with pytest.raises(ValueError):
        WC.investigate(bad, NOW)
    with pytest.raises(ValueError):
        WC.investigate(dataclasses.replace(c, matured_at=DECISION), NOW)
    with pytest.raises(ValueError):
        WC.investigate(dataclasses.replace(c, case_id="AAPL-2008-09-15"), NOW)


def test_investigation_is_deterministic():
    a, b = WC.investigate(make_case("pattern", 6), NOW), WC.investigate(make_case("pattern", 6), NOW)
    assert a == b


def test_null_world_does_not_hallucinate_market_causes():
    quiet_fired, rand_fired, n = 0, 0, 0
    for s in range(60):
        c = make_case("null", 100 + s)
        c.realised = float(np.random.default_rng(s).normal(0, 0.04))
        inv = WC.investigate(c, NOW)
        if inv.significant:
            n += 1
            rand_fired += inv.finding(WC.Level.MARKET).fired
            q = dataclasses.replace(c, market_path=c.market_path * 0.0, stock_path=c.stock_path)
            quiet_fired += WC.investigate(q, NOW).finding(WC.Level.MARKET).fired
    assert n >= 20
    assert quiet_fired == 0                                    # a market that did nothing can never be the cause
    assert rand_fired / n <= 0.20                              # a market that truly moved may coincide with the error, but not often


# ==================================================================================================== repeatability, trees
def test_repeatability_counts_only_prior_matured_cases():
    lib = WC.CaseLibrary()
    P = WC._cfg(None)
    assert WC.repeatability(lib, "L1_MARKET", "L1:down", NOW, P).verdict == "UNTESTED"
    for i in range(4):
        inv = WC.investigate(make_case("market", 20 + i), f"2005-07-{10 + i:02d}")
        inv = dataclasses.replace(inv, case_id=f"m{i}")
        lib.add(inv)
    r = WC.repeatability(lib, "L1_MARKET", "L1:down", "2005-08-01", P)
    assert r.verdict == "REPEATED" and r.n_same >= 3
    early = WC.repeatability(lib, "L1_MARKET", "L1:down", "2005-07-11", P)
    assert early.n_similar < r.n_similar                        # later cases are invisible to an earlier decision
    inv = WC.investigate(make_case("market", 30), NOW, library=lib)
    assert inv.conclusion.repeatability.verdict in ("REPEATED", "ONE_OFF_SO_FAR")


def test_plan_test_builds_a_tree_with_rivals_noise_and_unknown():
    inv = WC.investigate(make_case("pattern", 6), NOW)
    t = WC.plan_test(inv, NOW)
    ids = set(t.belief())
    assert t is not None and "h_oneoff" in ids and WC.HT.UNKNOWN_HID in ids and t.tree_id == inv.conclusion.test.tree_id
    assert abs(sum(t.belief().values()) - 1.0) < 1e-6
    u = WC.investigate(dataclasses.replace(make_case("null", 14), realised=0.01), NOW)
    ut = WC.plan_test(u, NOW)
    assert u.conclusion.level is None and ut is not None and "h_oneoff" in ut.belief()    # unexplained is researched, never forced


# ==================================================================================================== checklist V
def discovery(seed=0, n=40):
    return RP.Discovery("D-1", 0.008, 0.02, n, ("2003-01-01", "2003-12-31"), 5, frozenset({f"S{i}" for i in range(30)}), (1,), frozenset({"bull"}),
                        "code0", "data0", "2004-01-15", n_tests_searched=3)


def run(i, eff=0.010, n=40, ctrl=True, window=None, stocks=None, regime=None, implementation="primary"):
    r = rng_(50 + i)
    e = tuple(r.normal(eff, 0.02, n))
    c = tuple(np.asarray(e) - 0.006 - r.normal(0, 0.005, n)) if ctrl else None
    return RP.ReplicationRun(f"R{i}", "D-1", window or (f"{2005 + 2 * i}-01-01", f"{2005 + 2 * i}-12-31"),
                             stocks or frozenset({f"T{i}_{j}" for j in range(30)}), 100 + i, regime or ("bear", "sideways", "crash")[i % 3], e, c,
                             f"code{i + 1}", f"data{i + 1}", f"{2006 + 2 * i}-02-01", implementation)


def claim(searched=True, disc=True, supporting=("s",)):
    return WC.DiscoveryClaim("CL-1", "the pattern fails when the market is falling", "L5:regime_specific_failure", supporting, (), searched,
                             discovery() if disc else None)


CLAIM_NOW = "2012-01-01"


def test_claim_without_a_historical_test_is_only_an_explanation_hypothesis():
    v = WC.evaluate_claim(claim(disc=False), [], CLAIM_NOW)
    assert v.status == WC.ClaimStatus.EXPLANATION_HYPOTHESIS and not v.is_knowledge and v.failed_steps() == tuple(range(4, 12))


def test_claim_that_replicates_on_every_axis_becomes_knowledge():
    ledger = RP.ReplicationLedger()
    v = WC.evaluate_claim(claim(), [run(0), run(1), run(2)], CLAIM_NOW, ledger)
    assert len(v.steps) == 11
    assert v.status == WC.ClaimStatus.KNOWLEDGE and v.failed_steps() == () and v.gate_ok
    assert ledger.verify() == [] and ledger.status_history("D-1")                 # step 11: recorded permanently, chain intact


def test_planted_defects_each_block_knowledge():
    runs = [run(0), run(1), run(2)]
    assert not WC.evaluate_claim(claim(searched=False), runs, CLAIM_NOW).is_knowledge           # never looked for contradictions
    assert not WC.evaluate_claim(claim(supporting=()), runs, CLAIM_NOW).is_knowledge             # no supporting evidence
    assert not WC.evaluate_claim(claim(), runs[:1], CLAIM_NOW).is_knowledge                      # one replication is not enough
    no_ctrl = [run(i, ctrl=False) for i in range(3)]
    v = WC.evaluate_claim(claim(), no_ctrl, CLAIM_NOW)
    assert not v.is_knowledge and 9 in v.failed_steps()                                          # no control/placebo
    same_period = [run(0, window=("2003-01-01", "2003-12-31"), stocks=discovery().stocks)]
    v2 = WC.evaluate_claim(claim(), same_period, CLAIM_NOW)
    assert not v2.is_knowledge and 5 in v2.failed_steps()                                        # in-sample is not out-of-sample
    same_regime = [run(i, regime="bull") for i in range(3)]
    assert 8 in WC.evaluate_claim(claim(), same_regime, CLAIM_NOW).failed_steps()                # one regime only


def test_claim_that_fails_replication_is_refuted_and_kept():
    bad = [run(i, eff=-0.012, n=80) for i in range(3)]
    v = WC.evaluate_claim(claim(), bad, CLAIM_NOW)
    assert v.status == WC.ClaimStatus.REFUTED and not v.is_knowledge


def test_future_runs_are_a_firewall_breach_not_ignored():
    with pytest.raises(FirewallBreach):
        WC.evaluate_claim(claim(), [run(0), run(1), run(2)], "2006-01-01")


def test_claim_from_investigation_only_for_testable_causes():
    inv = WC.investigate(make_case("pattern", 6), NOW)
    cl = WC.claim_from_investigation(inv, discovery())
    assert cl is not None and cl.condition_key == inv.conclusion.cause_key and not cl.contradiction_search_done
    unk = WC.investigate(dataclasses.replace(make_case("null", 10), realised=-0.07, knowability=planted("EXTERNALLY_CAUSED")), NOW)
    assert WC.claim_from_investigation(unk, None) is None


# ==================================================================================================== the daily entry
def test_daily_step_investigates_once_and_preserves_the_unknowable():
    st = WC.WhatChangedState()
    unk = dataclasses.replace(make_case("null", 10), case_id="case-unk", realised=-0.07, knowability=planted("EXTERNALLY_CAUSED"))
    quiet = dataclasses.replace(make_case("null", 15, predicted=0.0), case_id="case-quiet", realised=0.001)
    cases = [make_case("market", 16), make_case("pattern", 6), unk, quiet]
    rep = WC.step(st, NOW, cases)
    assert quiet.case_id in rep.skipped_insignificant and unk.case_id in rep.preserved_unknowable
    assert len(rep.investigated) == 3 and rep.new_trees and st.claims
    assert all(v.status == WC.ClaimStatus.EXPLANATION_HYPOTHESIS for v in st.verdicts.values())      # nothing is knowledge yet
    again = WC.step(st, NOW, cases)
    assert again.investigated == () and not WC.investigation_table(st).empty
    assert WC.reclassify_unknowable(st, unk.case_id, "nope") is False and unk.case_id in st.unknowable
    early = WC.step(WC.WhatChangedState(), MATURED, cases)
    assert early.investigated == ()                                # nothing has matured strictly before the day it matures


def test_daily_step_on_no_cases_is_empty_and_safe():
    rep = WC.step(WC.WhatChangedState(), NOW, [])
    assert rep.investigated == () and rep.knowledge == () and dict(rep.by_class) == {}
