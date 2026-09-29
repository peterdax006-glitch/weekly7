"""Tests for engine.research.error_research (C68 checklists D, E, S, T, U). Synthetic data only, every draw seeded.
Each mechanism has a planted case it must catch, a null case where it must find nothing, and an empty/degenerate case."""
from types import SimpleNamespace

import numpy as np
import pytest

from engine.research import error_research as E
from engine.research import priority as P
from engine.research.core import FirewallBreach, Knowability, Problem

CFG = E.ErrorConfig()
NOW = "2021-06-30"


def day(i: int) -> str:
    return str((np.datetime64("2020-01-01") + int(i)).astype("datetime64[D]"))


def mk(i, expected=0.09, realised=0.045, conf=0.6, scale=0.02, pattern="p1", sector="tech", regime="calm", stock_type="mid", **kw):
    return {"record_id": f"o{i}", "decision_date": day(i), "outcome_date": day(i + 5), "expected": expected, "realised": realised,
            "confidence": conf, "scale": scale, "pattern": pattern, "sector": sector, "regime": regime, "stock_type": stock_type, **kw}


def noise_book(n=40, seed=0, sd=0.02, pattern="p1"):
    rng = np.random.default_rng(seed)
    return [mk(i, expected=0.06, realised=0.06 + rng.normal(0, sd), pattern=pattern) for i in range(n)]


def filled(records, cfg=CFG, now=NOW):
    st = E.new_state(cfg)
    for r in records:
        st.book.add(E.obs_from_record(r), now)
    return st


# ------------------------------------------------------------------ D: error-size research
def test_tiny_errors_stay_cheap_even_when_statistically_large():
    recs = [mk(i, expected=0.06, realised=0.062, conf=0.95, scale=0.0005) for i in range(40)]      # z = 4 but a 0.2pp miss
    st = E.new_state()
    rep = E.step(st, NOW, recs, with_self_questions=False)
    assert rep.tiers["DEEP"] == 0 and rep.tiers["STANDARD"] == 0
    assert rep.tiers["NONE"] == 40 and not rep.items and rep.minutes_requested == 0.0
    assert not rep.confident_wrong and len(rep.tiny_ids) == 40


def test_intensity_grows_with_every_factor_and_zero_error_is_zero():
    base = E.obs_from_record(mk(0, expected=0.09, realised=0.03, conf=0.5))
    i0 = E.intensity(base, CFG).value
    assert i0 > 0
    import dataclasses as dc
    assert E.intensity(dc.replace(base, confidence=0.9), CFG).value > i0
    assert E.intensity(dc.replace(base, market_weight=0.8), CFG).value > i0
    assert E.intensity(dc.replace(base, transfer=0.95), CFG).value > i0
    assert E.intensity(dc.replace(base, realised=-0.05), CFG).value > i0
    assert E.intensity(dc.replace(base, realised=base.expected), CFG).value == 0.0
    assert E.intensity(base, CFG, escalation=3.0).value > i0
    with pytest.raises(E.ErrorResearchError):
        E.intensity(base, CFG, escalation=0.5)


def test_confident_wrong_gets_far_more_than_unsure_and_slightly_wrong():
    fail = E.obs_from_record(mk(0, expected=0.09, realised=-0.05, conf=0.92))
    mild = E.obs_from_record(mk(1, expected=0.09, realised=0.075, conf=0.3))
    a, b = E.intensity(fail, CFG), E.intensity(mild, CFG)
    assert a.confident_wrong and a.tier == E.Tier.DEEP and not b.confident_wrong
    assert a.value > 20 * max(b.value, 1e-9) and TIER(b) < TIER(a)
    unsure_big_miss = E.obs_from_record(mk(2, expected=0.09, realised=-0.05, conf=0.2))
    assert not E.is_confident_wrong(unsure_big_miss, CFG)
    assert E.intensity(unsure_big_miss, CFG).value < a.value / 3


def TIER(it):
    return E.TIER_ORDER.index(it.tier)


def test_value_vector_none_for_tiny_and_valid_for_deep():
    tiny = E.obs_from_record(mk(0, expected=0.06, realised=0.0605, scale=0.0001))
    assert E.value_from_intensity(tiny, E.intensity(tiny, CFG), CFG) is None
    fail = E.obs_from_record(mk(1, expected=0.09, realised=-0.05, conf=0.92))
    it = E.intensity(fail, CFG)
    item = E.item_for_error(fail, it, CFG, NOW)
    assert item.check() == [] and item.value.compute_cost == CFG.tier_minutes[3] and item.real_data
    assert item.value.failure_reduction_value > 0.5 and item.problem == Problem.DIRECTION == E.item_for_error(fail, it, CFG, NOW).problem


# ------------------------------------------------------------------ E: the confident-wrong investigation
def kn(classification, avail=()):
    return SimpleNamespace(classification=classification, information_that_would_have_been_available=tuple(avail), explanations=())


def cw_obs():
    return E.obs_from_record(mk(0, expected=0.09, realised=-0.05, conf=0.92))


def test_planted_confident_failure_triggers_deep_investigation_with_all_15_questions():
    st = E.new_state()
    recs = noise_book(30, seed=1) + [mk(40, expected=0.09, realised=-0.05, conf=0.92, pattern="p2")]
    ctx = E.InvestigationContext(feature_shifts={"rsi_z": 0.3, "vol_z": 4.2}, pattern_reliability_before=0.7, pattern_reliability_after=0.35,
                                 overriding_pattern="p9", vol_ratio=2.4, corr_change=0.4, sector_error_share=0.8, timing_shift_days=2,
                                 exit_regret=0.5, model_inputs=["rsi_z"], knowability=kn(Knowability.POTENTIALLY_PREDICTABLE, ["vix_jump"]),
                                 precursor_hits=None, interaction_change_z=3.0)
    rep = E.step(st, NOW, recs, contexts={"o40": ctx}, with_self_questions=False)
    assert rep.confident_wrong == ("o40",) and rep.tiers["DEEP"] >= 1
    inv = st.investigations["o40"]
    assert len(inv.findings) == 15 and inv.check() == []
    assert "vol_z" in inv.by_number(2).answer
    assert inv.by_number(4).answer.startswith("yes") and inv.by_number(5).answer.endswith("overrode it")
    assert inv.by_number(12).answer.startswith("yes") and "vix_jump" in inv.by_number(13).answer
    assert inv.by_number(14).status == E.FindingStatus.NOT_APPLICABLE
    assert inv.by_number(15).status == E.FindingStatus.OPEN and inv.follow_ups           # a precursor question for the discovery lab
    assert inv.causes and inv.by_number(1).status == E.FindingStatus.ANSWERED
    assert any(i.family == "error_research/confident_wrong" and i.value.compute_cost == CFG.tier_minutes[3] for i in rep.items)


def test_unknowable_cause_is_classified_not_invented_and_deprioritised():
    st = E.new_state()
    o = cw_obs()
    inv = E.investigate(o, E.InvestigationContext(knowability=kn(Knowability.EXTERNALLY_CAUSED)), CFG, NOW)
    assert inv.by_number(12).answer.startswith("no")
    assert inv.by_number(14).status == E.FindingStatus.UNKNOWABLE and inv.by_number(15).status == E.FindingStatus.NOT_APPLICABLE
    assert inv.by_number(13).status == E.FindingStatus.NOT_APPLICABLE
    st.depth.mark("pattern=p1", inv)
    assert st.depth.multiplier("pattern=p1") == pytest.approx(CFG.unknowable_mult)
    inv2 = E.investigate(o, E.InvestigationContext(knowability=kn(Knowability.PREDICTABLE, ["x"]), precursor_hits=["p_a"]), CFG, NOW)
    st.depth.mark("pattern=p1", inv2)                         # later evidence reopens the cell
    assert st.depth.multiplier("pattern=p1") == 1.0


def test_empty_context_leaves_questions_unanswered_never_guessed():
    inv = E.investigate(cw_obs(), E.InvestigationContext(), CFG, NOW)
    assert [f.number for f in inv.findings] == list(range(1, 16))
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
        assert inv.by_number(n).status == E.FindingStatus.UNANSWERED, n
    assert inv.by_number(15).status == E.FindingStatus.OPEN and len(inv.follow_ups) >= 10 and inv.answered_share == 0.0


def test_investigation_of_a_future_outcome_is_refused():
    o = cw_obs()
    with pytest.raises(FirewallBreach):
        E.investigate(o, E.InvestigationContext(), CFG, o.matured_at)


def test_null_context_says_possibly_isolated():
    ctx = E.InvestigationContext(feature_shifts={"a": 0.2}, pattern_reliability_before=0.6, pattern_reliability_after=0.58, vol_ratio=1.05,
                                 corr_change=0.02, sector_error_share=0.1, timing_shift_days=0, exit_regret=0.02, interaction_change_z=0.4,
                                 overriding_pattern="", knowability=kn(Knowability.UNKNOWN))
    inv = E.investigate(cw_obs(), ctx, CFG, NOW)
    assert not inv.causes and "isolated" in inv.by_number(1).answer
    assert inv.by_number(14).status == E.FindingStatus.UNKNOWABLE and inv.knowable == Knowability.UNKNOWN


# ------------------------------------------------------------------ S: repeated error escalation
def shrink_series(n, seed=3):
    rng = np.random.default_rng(seed)
    return [mk(i, expected=rng.uniform(0.08, 0.10), realised=rng.uniform(0.03, 0.06), conf=0.7) for i in range(n)]


def test_repeated_expected_8_10_realised_3_6_escalates_monotonically():
    prev, mults = 1.0, []
    for n in range(6, 41, 2):
        st = filled(shrink_series(n))
        gs = st.book.stats(NOW, "all")
        assert gs.kind == E.BiasKind.SHRINK and gs.ratio < 0.75
        assert gs.multiplier >= prev - 1e-9
        prev = gs.multiplier
        mults.append(gs.multiplier)
    assert mults[0] > 1.5 and mults[-1] > mults[0] + 1.0 and mults[-1] <= CFG.max_mult
    it = [E.intensity(E.obs_from_record(mk(0, 0.09, 0.045, 0.6)), CFG, m).value for m in mults]
    assert all(b >= a for a, b in zip(it, it[1:])) and it[-1] > it[0]


def test_isolated_noise_and_single_outlier_do_not_escalate():
    fired = 0
    for seed in range(12):
        st = filled(noise_book(40, seed=seed))
        fired += any(s.systematic for s in st.book.all_stats(NOW).values())
    assert fired <= 1                                       # false-alarm rate near the nominal level, not a stampede
    recs = noise_book(30, seed=5) + [mk(31, expected=0.06, realised=-0.10, conf=0.9)]
    st = filled(recs)
    assert st.book.stats(NOW, "all").multiplier == 1.0
    assert E.intensity(E.obs_from_record(recs[-1]), CFG, st.book.escalation(E.obs_from_record(recs[-1]), NOW)[0]).confident_wrong


def test_underprediction_is_detected_and_worded_as_a_missing_factor():
    rng = np.random.default_rng(4)
    recs = [mk(i, expected=0.05, realised=0.05 + rng.uniform(0.02, 0.06)) for i in range(20)]
    gs = filled(recs).book.stats(NOW, "all")
    assert gs.kind == E.BiasKind.UNDER and gs.mean_z > 0 and gs.run >= 10
    assert "missing factor" in E.hypothesis_for(gs)
    assert "consistently suppressing" in E.hypothesis_for(filled(shrink_series(20)).book.stats(NOW, "all"))


def test_pattern_level_escalation_isolated_to_the_failing_group():
    good = noise_book(30, seed=8, pattern="good")
    bad = [dict(r, record_id="b" + r["record_id"], pattern="bad") for r in shrink_series(30)]
    st = filled(good + bad)
    assert st.book.stats(NOW, "pattern=bad").systematic and not st.book.stats(NOW, "pattern=good").systematic
    groups = [s.group for s in st.book.escalated(NOW)]
    assert any("pattern=bad" in g for g in groups) and not any("pattern=good" in g for g in groups)
    esc, gs = st.book.escalation(E.obs_from_record(bad[-1]), NOW)
    assert esc > 1.5 and gs.systematic


def test_half_life_reported_for_persistent_error_streams():
    gs = filled(shrink_series(40)).book.stats(NOW, "all")
    assert gs.half_life is None or gs.half_life > 0            # persistence estimate is optional but never negative


def test_pattern_stats_ignore_records_that_mature_after_now():
    recs = shrink_series(20)
    st = filled(recs, now="2030-01-01")
    cut = recs[9]["outcome_date"]
    a = st.book.stats(cut, "all")
    st2 = filled(recs[:9], now="2030-01-01")
    b = st2.book.stats(cut, "all")
    assert (a.n, a.t, a.multiplier) == (b.n, b.t, b.multiplier)
    scrambled = filled(recs[:9] + [dict(r, realised=-r["realised"] * 7) for r in recs[10:]], now="2030-01-01")
    c = scrambled.book.stats(cut, "all")
    assert (a.n, a.t, a.multiplier) == (c.n, c.t, c.multiplier)         # the future cannot change what was known at `cut`


def test_book_refuses_future_outcomes_and_rewrites():
    st = E.new_state()
    o = E.obs_from_record(mk(0))
    with pytest.raises(FirewallBreach):
        st.book.add(o, o.matured_at)
    assert st.book.add(o, NOW) is True and st.book.add(o, NOW) is False
    import dataclasses as dc
    with pytest.raises(E.ErrorResearchError):
        st.book.add(dc.replace(o, realised=0.5), NOW)


def test_empty_book_is_neutral():
    st = E.new_state()
    assert st.book.all_stats(NOW) == {} and st.book.escalated(NOW) == []
    gs = st.book.stats(NOW, "all")
    assert gs.multiplier == 1.0 and gs.kind == E.BiasKind.NONE and E.item_for_pattern(gs, CFG, NOW) is None


# ------------------------------------------------------------------ U: allocation through the existing priority engine
def test_step_feeds_priority_engine_and_ranks_failure_above_noise():
    recs = noise_book(30, seed=2) + [mk(40, expected=0.09, realised=-0.05, conf=0.92, pattern="p2", market_weight=0.5, transfer=0.8)]
    ps = P.new_state()
    st = E.new_state()
    rep = E.step(st, NOW, recs, priority_state=ps, budget=E.ComputeBudget(cpu_minutes=400, ram_gb_free=8), seed=1)
    assert rep.plan is not None and rep.plan.selected
    top = rep.plan.top(3)
    assert any(r.item.family == "error_research/confident_wrong" for r in top)
    assert len(rep.self_questions) == 11
    rep2 = E.step(E.new_state(), NOW, recs, priority_state=P.new_state(), budget=E.ComputeBudget(cpu_minutes=400, ram_gb_free=8), seed=1)
    assert rep2.plan.selected == rep.plan.selected                                # deterministic


def test_unknowable_and_barren_cells_lose_priority_multiplier():
    dl = E.DepthLedger(CFG)
    assert dl.multiplier("c") == 1.0
    mults = []
    for _ in range(5):
        dl.observe("c", 0.0)
        mults.append(dl.multiplier("c"))
    assert all(b < a for a, b in zip(mults, mults[1:])) and mults[-1] >= CFG.barren_floor
    dl.observe("c", 0.5)
    assert dl.multiplier("c") == 1.0                                                # one good result reopens the line
    with pytest.raises(E.ErrorResearchError):
        dl.observe("c", -1)
    assert dl.redundancy("c") > dl.redundancy("fresh") == 0.0


def test_finished_results_teach_the_depth_ledger():
    st = E.new_state()
    recs = [mk(40, expected=0.09, realised=-0.05, conf=0.92, pattern="p2")]
    rep = E.step(st, NOW, recs, with_self_questions=False)
    item = rep.items[0]
    assert st.item_cells[item.item_id] == "pattern=p2|sector=tech|regime=calm|stock_type=mid"
    rv = P.RealisedValue(item, E.ExperimentValue(decision_value=0.0, compute_cost=5.0), 5.0, "2021-07-15", survived_oos=False)
    E.step(st, "2021-08-01", [], results=[rv], with_self_questions=False)
    assert st.depth.barren_streak(st.item_cells[item.item_id]) == 1


# ------------------------------------------------------------------ T: self-research questions
def test_self_questions_are_eleven_identity_free_and_flag_thin_evidence():
    qs = E.self_research_questions(E.new_state().book, NOW, NOW)
    assert [q.number for q in qs] == list(range(1, 12)) and all(q.insufficient for q in qs)
    from engine.learning.research_priority import identity_leak
    assert all(identity_leak(q.question.text) == "" and q.item.check() == [] for q in qs)
    assert all(q.item.value.compute_cost <= 3.0 for q in qs)                       # nothing to research yet: cheap


def test_self_questions_find_the_planted_degradation_and_rising_errors():
    rng = np.random.default_rng(7)
    recs = []
    for i in range(60):
        sd = 0.008 + 0.0012 * i                                                     # errors grow with time in pattern 'rot'
        recs.append(mk(i, expected=0.06, realised=0.06 + rng.normal(0, sd), pattern="rot", regime="stress" if i % 2 else "calm"))
        recs.append(dict(mk(i, expected=0.06, realised=0.06 + rng.normal(0, 0.012), pattern="steady", regime="calm"), record_id=f"s{i}"))
    st = filled(recs, now="2030-01-01")
    fnd = E.self_findings(st.book, "2030-01-01")
    assert not fnd["errors_rising"]["insufficient"] and fnd["errors_rising"]["p"] < 0.05
    assert [p for p, _ in fnd["patterns_degrading"]["degrading"]] == ["rot"]
    qs = {q.key: q for q in E.self_research_questions(st.book, "2030-01-01", "2030-01-01")}
    assert qs["errors_rising"].item.value.information_gain > qs["confidence_miscalibrated"].item.value.information_gain
    assert not qs["errors_rising"].insufficient and not qs["patterns_degrading"].insufficient
    assert qs["what_next"].finding["next"] in {"errors_rising", "patterns_degrading", "worst_regime", "systematic_bias"}


def test_self_questions_on_pure_noise_find_nothing_significant():
    st = filled(noise_book(60, seed=11), now="2030-01-01")
    fnd = E.self_findings(st.book, "2030-01-01")
    assert fnd["errors_rising"]["p"] > 0.05 and not fnd["patterns_degrading"]["degrading"]
    assert not fnd["confidence_miscalibrated"].get("informative", False)


def test_discovery_and_barren_findings_read_priority_state():
    ps = P.new_state()
    item = P.ResearchItem("i1", "a probe", Problem.VOLATILITY, "fam", P.make_value(10, 0.1, 0.1, 0.1), "2020-01-01")
    for k in range(6):
        ps.waste.observe("dead_family", 0.0, f"2020-02-0{k + 1}")
    res = [SimpleNamespace(survived_oos=v) for v in (False, False, False, False, True)]
    fnd = E.self_findings(E.new_state().book, NOW, res, ps)
    assert fnd["barren_approaches"]["barren"] == ["dead_family"]
    assert fnd["discoveries_oos"]["rate"] == pytest.approx(0.2) and fnd["discoveries_oos"]["p"] < 0.5


# ------------------------------------------------------------------ validation and adaptation
def test_record_adapter_validates():
    with pytest.raises(E.ErrorResearchError):
        E.obs_from_record({"record_id": "x", "expected": 0.1})
    with pytest.raises(E.ErrorResearchError):
        E.obs_from_record(mk(0, conf=1.4))
    with pytest.raises(E.ErrorResearchError):
        E.obs_from_record(mk(0, pattern="pattern seen 2015-03-04"))                # a date in a label breaks the identity firewall
    o = E.obs_from_record(SimpleNamespace(**{"prediction_id": "z", "decision_date": "2020-01-01", "matured_at": "2020-01-06",
                                             "expected_return": 0.1, "realized": 0.04, "conf": 0.5}))
    assert o.obs_id == "z" and o.error == pytest.approx(-0.06)


def test_config_check_and_empty_step():
    assert E.ErrorConfig().check() == []
    assert E.ErrorConfig(tier_cut=(0.1, 0.05, 0.2)).check()
    with pytest.raises(E.ErrorResearchError):
        E.new_state(E.ErrorConfig(min_n=1))
    rep = E.step(E.new_state(), NOW, [], with_self_questions=True)
    assert rep.ingested == 0 and not rep.items and rep.plan is None and rep.self_questions == ()
