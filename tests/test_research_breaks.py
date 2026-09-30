"""Tests for engine.research.break_research (C66 section 14): planted precursor it must explain, breaks with no precursor it must
leave UNKNOWN, the empty/degenerate cases, the firewalls and the scheduler. Synthetic data only."""
import dataclasses
import functools
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import break_detection as bd
from engine.learning.core import FirewallBreach
from engine.research import break_research as BR
from engine.research.core import ResearchState

FAST = {"n_perm": 150, "n_random": 80, "boot": 150}


def _ar1(rng, T, phi):
    x = np.zeros(T)
    e = rng.normal(0, 1, T)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + math.sqrt(1 - phi * phi) * e[i]
    return x


@functools.lru_cache(maxsize=None)
def make_item(kind="precursor", T=1000, seed=0, name="pat_a", peek=False):
    """precursor: a stress column ramps up in the 8 rows BEFORE every break and stays high during it; random: the same kind of
    breaks with no observable link; healthy: never breaks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-04", periods=T, freq="W-FRI")
    cols = {f"m_n{k}": _ar1(rng, T, 0.8) for k in range(6)}
    stress = 0.6 * _ar1(rng, T, 0.7)
    sign = np.ones(T)
    onsets, t, length, lead = [], 45, 22, 8
    while t < T - length - 5:
        onsets.append(t)
        sign[t:t + length] = -1.0
        t += length + 22 + int(rng.exponential(24))
    if kind == "precursor":
        for o in onsets:
            stress[o - lead:o] += 2.0 * (np.arange(lead) + 1) / lead
            stress[o:o + length] += 2.0
    if kind == "healthy":
        sign[:] = 1.0
    cols["m_stress"] = stress
    value = sign * 0.008 + rng.normal(0, 0.010, T)
    fr = pd.DataFrame({"value": value, **cols}, index=idx)
    if peek:
        fr["m_peek"] = fr["value"] + rng.normal(0, 0.0005, T)
    dims = ["liquidity", "trend", "breadth", "macro", "feature_distribution", "regime"]
    spec = [bd.ContextColumn("m_stress", "volatility")] + [bd.ContextColumn(f"m_n{k}", d) for k, d in enumerate(dims)]
    if peek:
        spec.append(bd.ContextColumn("m_peek", "macro"))
    return bd.ItemSeries(name, fr, tuple(spec))


def last(item):
    return item.frame.index[-1]


# ------------------------------------------------------------------------------------------------- the six questions

def test_planted_precursor_is_explained_with_all_six_questions():
    item = make_item("precursor", seed=0)
    F = BR.build_frame(item, last(item), FAST)
    res, rule = BR.run_ladder(F, 1, stop_early=False)
    assert [r.qid for r in res] == list(BR.QUESTION_ORDER)
    assert all(r.passed for r in res), [r.brief() for r in res]
    assert rule.column == "m_stress" and rule.direction == 1
    assert BR.adjudicate(res)[0] == BR.Verdict.EXPLAINED


def test_breaks_without_a_precursor_are_never_explained():
    n_unknown = 0
    for seed in range(10):                                         # F22: all seeds (was (0, 2, 3): seed 1 was left out, it passes)
        item = make_item("random", seed=seed)
        F = BR.build_frame(item, last(item), FAST)
        res, rule = BR.run_ladder(F, seed, stop_early=True)
        verdict = BR.adjudicate(res)[0]
        assert verdict != BR.Verdict.EXPLAINED, [r.brief() for r in res]
        n_unknown += verdict == BR.Verdict.UNKNOWN
    assert n_unknown >= 7


def test_null_calibration_counts_placebos_and_none_are_explained():
    item = make_item("precursor", seed=0)
    cal = BR.null_calibration([item], last(item), FAST, seed=3, shifts=(37, 83))
    assert cal["placebos"] == 2 and cal["explained"] == 0
    assert cal["reached"][BR.QuestionId.PREEXISTING_PREDICTOR.value] == 2


def test_planted_recall_is_reported_on_the_mirror_case():
    item = make_item("precursor", seed=0)
    rec = BR.planted_recall([item], last(item), FAST)
    assert rec["items"] == 1 and rec["explained"] == 1 and rec["recall"] == 1.0


def test_lookahead_predictor_is_refused_not_explained():
    item = make_item("precursor", seed=0, peek=True)
    with pytest.raises(FirewallBreach):
        BR.build_frame(item, last(item), FAST)


def test_gate_that_fires_on_losses_reduces_them_and_a_random_gate_does_not():
    rng = np.random.default_rng(5)
    v = rng.normal(0.004, 0.01, 400)
    lossy = v < -0.006
    good = BR.excess_loss_avoided(v, lossy)
    rand = np.array([BR.excess_loss_avoided(v, rng.random(400) < lossy.mean()) for _ in range(40)])
    assert good > 0 and good > np.quantile(rand, 0.99)
    assert abs(rand.mean()) < 0.5 * good


def test_random_gate_shift_null_is_not_significant():
    rng = np.random.default_rng(7)
    v = rng.normal(0.003, 0.01, 300)
    g = BR.markov_gate(0.25, 5.0, 300, rng)
    obs = BR.expectancy_gain(v, g)
    null = BR.shift_null(lambda gg: BR.expectancy_gain(v, gg), g, 200, rng)
    assert BR.plus_one_p(obs, null) > 0.05


# ------------------------------------------------------------------------------------------------- empty and degenerate cases

def test_short_history_is_insufficient_not_a_crash():
    item = make_item("precursor", seed=0)
    short = bd.ItemSeries("s", item.frame.iloc[:60], item.columns)
    F = BR.build_frame(short, last(short), FAST)
    res, rule = BR.run_ladder(F, 0, stop_early=True)
    assert rule is None and BR.adjudicate(res)[0] == BR.Verdict.INSUFFICIENT_EVIDENCE
    assert res[0].outcome == BR.Outcome.NOT_TESTABLE


def test_one_row_history_raises_a_clear_error():
    item = make_item("precursor", seed=0)
    tiny = bd.ItemSeries("t", item.frame.iloc[:2], item.columns)
    with pytest.raises(BR.BreakResearchError):
        BR.build_frame(tiny, last(tiny), FAST)


def test_healthy_item_opens_no_investigation():
    item = make_item("healthy", seed=1)
    assert BR.detect_events(item, last(item), FAST) == []
    st = BR.new_state()
    rep = BR.step(st, last(item), {"h": item}, cfg=FAST)
    assert rep.opened == () and not st.investigations and st.unknown_share() != st.unknown_share()


def test_step_with_no_items_and_empty_adjudication():
    st = BR.new_state()
    rep = BR.step(st, "2020-01-03", {}, cfg=FAST)
    assert rep.opened == () and rep.ran == () and st.verify() == []
    assert BR.adjudicate([])[0] == BR.Verdict.INSUFFICIENT_EVIDENCE


def test_helpers_on_empty_inputs():
    assert BR.runs(np.zeros(5, bool)) == [] and BR.runs(np.array([1, 1, 0, 1], bool)) == [2, 1]
    assert math.isnan(BR.cvar(np.array([])) ) and BR.max_drawdown(np.array([])) == 0.0
    assert BR.block_resample(0, 4, np.random.default_rng(0)).size == 0
    assert BR.plus_one_p(1.0, np.array([])) == 1.0
    assert BR.fdr_across_investigations(BR.new_state())["n"] == 0
    assert BR.pooled_predictor([], "x")["n_frames"] == 0


# ------------------------------------------------------------------------------------------------- statistics helpers

def test_drawdown_cvar_and_runs():
    v = np.array([1.0, -2.0, -1.0, 3.0, -0.5])
    assert BR.max_drawdown(v) == pytest.approx(3.0)
    assert BR.cvar(np.array([-5.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), 0.10) == -5.0


def test_markov_gate_matches_requested_coverage_and_run_length():
    g = BR.markov_gate(0.3, 6.0, 20000, np.random.default_rng(1))
    assert g.mean() == pytest.approx(0.3, abs=0.03)
    assert np.mean(BR.runs(g)) == pytest.approx(6.0, rel=0.2)


def test_gate_with_hold_is_causal_and_holds():
    raw = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    held = BR.gate_with_hold(raw, 3)
    assert held.tolist() == [False, False, True, True, True, False, False, True, True]
    changed = raw.copy()
    changed[-1] = True
    assert BR.gate_with_hold(changed, 3)[:-1].tolist() == held[:-1].tolist()


def test_deff_grows_with_persistence():
    rng = np.random.default_rng(0)
    iid = (rng.random(400) < 0.3).astype(float)
    blocky = np.repeat((rng.random(40) < 0.3).astype(float), 10)
    assert BR.deff_of(blocky) > BR.deff_of(iid) >= 1.0


def test_auc_columns_matches_reference():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(200, 3)), rng.random(200) < 0.4
    X[y, 1] += 1.0
    a = BR.fast_auc_columns(X, y)
    assert a[1] == pytest.approx(BR.PR.auc(X[:, 1], y)) and a[1] > 0.7 and abs(a[0] - 0.5) < 0.12


def test_fdr_across_investigations_pays_for_many_patterns():
    st = BR.new_state()
    base = make_item("precursor", seed=0)
    ev = BR.detect_events(base, last(base), FAST)[0]
    val = BR.ExperimentValue()
    for k, p in enumerate([0.001, 0.04, 0.5, 0.9]):
        inv = BR.open_investigation(dataclasses.replace(ev, item_key=f"k{k}"), f"i{k}", "2020-01-01", BR.Trigger("episode", 0.5), val, BR.PARAMS)
        r = BR.QuestionResult(BR.QuestionId.PREEXISTING_PREDICTOR, BR.Outcome.PASSED, 0.7, p, 30, {}, "x")
        st.investigations[inv.inv_id] = dataclasses.replace(inv, results=(r,), state=ResearchState.PROMISING)
    out = BR.fdr_across_investigations(st, 0.10)
    assert out["n"] == 4 and sum(out["survives"].values()) == 2


# ------------------------------------------------------------------------------------------------- adjudication and validation

def _r(qid, outcome):
    return BR.QuestionResult(qid, outcome, 0.5, 0.5, 10, {}, "because")


def test_adjudication_rules():
    allp = [_r(q, BR.Outcome.PASSED) for q in BR.QUESTION_ORDER]
    assert BR.adjudicate(allp)[0] == BR.Verdict.EXPLAINED
    one_fail = allp[:3] + [_r(BR.QUESTION_ORDER[3], BR.Outcome.FAILED)]
    assert BR.adjudicate(one_fail)[0] == BR.Verdict.UNKNOWN
    untestable = allp[:2] + [_r(BR.QUESTION_ORDER[2], BR.Outcome.NOT_TESTABLE)]
    assert BR.adjudicate(untestable)[0] == BR.Verdict.INSUFFICIENT_EVIDENCE
    fail_wins = untestable + [_r(BR.QUESTION_ORDER[3], BR.Outcome.FAILED)]
    assert BR.adjudicate(fail_wins)[0] == BR.Verdict.UNKNOWN


def test_investigation_cannot_claim_a_cause_without_six_passes():
    item = make_item("precursor", seed=0)
    ev = BR.detect_events(item, last(item), FAST)[0]
    inv = BR.open_investigation(ev, "x", "2020-01-01", BR.Trigger("episode", 0.5), BR.ExperimentValue(), BR.PARAMS)
    bad = dataclasses.replace(inv, verdict=BR.Verdict.EXPLAINED)
    assert any("all six" in e for e in bad.validate())
    sneaky = dataclasses.replace(inv, cause=BR.BreakCause.REGIME_CHANGE, verdict=BR.Verdict.UNKNOWN)
    assert any("only be claimed" in e for e in sneaky.validate())


def test_break_event_validation_and_text_firewall():
    ev = BR.BreakEvent("k", 10, 5, None, "a", "b", 1.0, 3, 100)
    assert ev.validate()
    with pytest.raises(BR.BreakResearchError):
        BR.open_investigation(ev, "x", "2020-01-01", BR.Trigger("episode", 0.5), BR.ExperimentValue(), BR.PARAMS)
    with pytest.raises(FirewallBreach):
        BR.safe_text("why did it break in 2008")
    with pytest.raises(FirewallBreach):
        BR.safe_text("on 2009-03-06 it failed")


# ------------------------------------------------------------------------------------------------- hypotheses

def test_hypothesis_priors_sum_to_one_with_unknown_floor():
    item = make_item("random", seed=0)
    F = BR.build_frame(item, last(item), FAST)
    ev = BR.detect_events(item, last(item), FAST)[0]
    pr = BR.hypothesis_priors(F, ev)
    assert sum(pr.values()) == pytest.approx(1.0) and pr[BR.BreakCause.UNKNOWN_CAUSE] >= BR.UNKNOWN_FLOOR - 1e-9
    hyps = BR.assess_hypotheses(F, ev, seed=1)
    assert {h.cause for h in hyps} == set(BR.BreakCause)
    assert sum(h.posterior for h in hyps) == pytest.approx(1.0)
    assert all(not h.validate() for h in hyps)
    assert next(h for h in hyps if h.cause == BR.BreakCause.UNKNOWN_CAUSE).posterior >= BR.UNKNOWN_FLOOR - 1e-9


def test_planted_recurring_calendar_breaks_support_event_environment():
    T = 700
    idx = pd.date_range("2005-01-07", periods=T, freq="W-FRI")
    rng = np.random.default_rng(4)
    v = 0.006 + rng.normal(0, 0.01, T)
    for y in range(2005, 2018):
        m = (idx.year == y) & idx.month.isin([8, 9, 10])
        v[m] -= 0.016
    fr = pd.DataFrame({"value": v, "m_a": _ar1(rng, T, 0.8), "m_b": _ar1(rng, T, 0.8)}, index=idx)
    item = bd.ItemSeries("cal", fr, (bd.ContextColumn("m_a", "regime"), bd.ContextColumn("m_b", "liquidity")))
    F = BR.build_frame(item, last(item), FAST)
    got = BR.test_event_environment(F)
    assert got.supported and got.p < 0.05
    rnd = BR.test_event_environment(BR.build_frame(make_item("random", seed=0), last(make_item("random", seed=0)), FAST))
    assert rnd.supported is not True or rnd.p > 0.001


def test_random_variation_is_live_for_a_mild_drop_and_dead_for_a_deep_one():
    rng = np.random.default_rng(9)
    T = 600
    idx = pd.date_range("2008-01-04", periods=T, freq="W-FRI")

    def frame(drop):
        v = 0.006 + rng.normal(0, 0.01, T)
        v[300:330] -= drop
        f = pd.DataFrame({"value": v, "m_a": _ar1(rng, T, 0.8)}, index=idx)
        return BR.build_frame(bd.ItemSeries("r", f, (bd.ContextColumn("m_a", "regime"),)), idx[-1], FAST)

    deep, mild = BR.test_random_variation(frame(0.03), 1), BR.test_random_variation(frame(0.0), 1)
    assert deep.supported is False and mild.supported is True


# ------------------------------------------------------------------------------------------------- scheduler, firewalls, outputs

@pytest.fixture(scope="module")
def stepped():
    item = make_item("precursor", seed=0)
    st = BR.new_state()
    now = last(item)
    rep = BR.step(st, now, {"pat_a": item}, cfg={**FAST, "max_per_step": 2}, seed=5, created_real="2026-09-29")
    return st, rep, item, now


def test_step_opens_one_investigation_per_break_and_runs_within_budget(stepped):
    st, rep, item, now = stepped
    n_breaks = len(BR.detect_events(item, now, FAST))
    assert len(rep.opened) == n_breaks == len(st.investigations)
    assert len(rep.ran) == 2 and len(rep.deferred) == n_breaks - 2
    assert all(st.investigations[i].state != ResearchState.QUEUED for i in rep.ran)
    assert st.verify() == [] and rep.minutes > 0


def test_step_is_idempotent_for_the_same_evidence(stepped):
    st, rep, item, now = stepped
    n = len(st.investigations)
    rep2 = BR.step(st, now, {"pat_a": item}, cfg={**FAST, "max_per_step": 0}, seed=5)
    assert rep2.opened == () and len(st.investigations) == n


def test_time_only_moves_forward(stepped):
    st, rep, item, now = stepped
    with pytest.raises(FirewallBreach):
        BR.step(st, item.frame.index[100], {"pat_a": item}, cfg=FAST)


def test_outputs_are_identity_free_and_valid(stepped):
    st, rep, item, now = stepped
    assert rep.questions
    for q in rep.questions:
        assert BR.RP.identity_leak(q.text) == "" and "pat_a" not in q.text
    for s in rep.signals:
        assert not s.check()
    for rec in rep.records:
        assert "pat_a" not in str(rec.payload) and "2008" not in str(rec.payload)
    for inv in st.investigations.values():
        assert not inv.validate()


def test_record_release_needs_matured_time_and_respects_same_year_replay(stepped):
    st, rep, item, now = stepped
    rec = rep.records[0]
    assert BR.release(st, rec.record_id, "2030-01-01")["item"]
    with pytest.raises(FirewallBreach):
        BR.release(st, rec.record_id, rec.matured_at)
    year = st.filed_years[rec.record_id][0]
    with pytest.raises(FirewallBreach):
        BR.release(st, rec.record_id, "2030-01-01", replaying_years=[year])
    with pytest.raises(BR.BreakResearchError):
        BR.release(st, "nope", "2030-01-01")


def test_records_are_withheld_from_the_report_while_their_year_is_replayed():
    item = make_item("precursor", seed=0)
    st = BR.new_state()
    yrs = tuple(sorted({int(y) for y in item.frame.index.year}))
    rep = BR.step(st, last(item), {"pat_a": item}, cfg={**FAST, "max_per_step": 1}, replaying_years=yrs)
    assert rep.ran and rep.records == () and len(rep.withheld) >= 1


def test_leaking_item_is_refused_and_its_open_work_is_cancelled():
    good = make_item("precursor", seed=0)
    bad = make_item("precursor", seed=0, name="pat_leak", peek=True)
    st = BR.new_state()
    rep = BR.step(st, last(good), {"pat_a": good, "pat_leak": bad}, cfg={**FAST, "max_per_step": 0})
    assert len(rep.refused) == 1 and st.refused and rep.opened
    assert all(i.item_id == "pat_a" for i in st.investigations.values())


def test_export_verifies_and_detects_tampering(stepped):
    st, *_ = stepped
    snap = BR.export_state(st)
    assert BR.verify_export(snap) == []
    snap["ledger"][0]["payload"]["rows"] = 1
    assert BR.verify_export(snap)
    st.ledger[0]["payload"]["rows"] = 1
    assert st.verify()


def test_reopen_needs_new_evidence_and_retire_exhausted():
    item = make_item("precursor", seed=0)
    st = BR.new_state()
    now = last(item)
    BR.step(st, now, {"a": item}, cfg={**FAST, "max_per_step": 1}, seed=1)
    inv = next(i for i in st.investigations.values() if i.attempts)
    st.investigations[inv.inv_id] = dataclasses.replace(inv, state=ResearchState.DORMANT, attempts=4)
    assert BR.retire_exhausted(st, now, FAST) == [inv.inv_id]
    assert st.investigations[inv.inv_id].state == ResearchState.RETIRED and st.verify() == []


def test_prefix_stability_of_detected_breaks():
    item = make_item("precursor", seed=0)
    idx = item.frame.index
    assert BR.events_prefix_stable(item, idx[500], idx[-1], FAST) == []


def test_report_funnel_and_dossier_render(stepped):
    st, rep, item, now = stepped
    text = BR.render_report(st)
    assert BR.LABEL in text and "unknown share" in text
    f = BR.question_funnel(st)
    assert list(f["question"]) == [q.value for q in BR.QUESTION_ORDER] and f["asked"].iloc[0] >= 1
    assert len(BR.cause_table(st)) == len(BR.BreakCause)
    inv = st.investigations[rep.ran[0]]
    assert "Q1_PREEXISTING_PREDICTOR" in BR.dossier(inv)
    assert BR.compute_accounting(st)["investigations"] == len(st.investigations)
    assert len(BR.investigation_table(st)) == len(st.investigations)


def test_health_row_dated_now_or_later_is_refused():
    item = make_item("precursor", seed=0)
    st = BR.new_state()
    with pytest.raises(FirewallBreach):
        BR.step(st, "2020-01-03", {"a": item}, health=[{"knowledge_id": "a", "when": "2020-01-03", "health": "BROKEN"}], cfg=FAST)
    assert BR.health_triggers([{"knowledge_id": "a", "when": "2020-01-02", "health": "BROKEN"}], "2020-01-03") == {"a": pytest.approx(0.9)}


def test_holding_period_and_pooled_predictor_on_planted_precursor():
    item = make_item("precursor", seed=0)
    F = BR.build_frame(item, last(item), FAST)
    res, rule = BR.run_ladder(F, 1, stop_early=True)
    table = BR.holding_period_check(F, rule)
    assert list(table["hold"]) == [1, 3, 6] and table["coverage"].is_monotonic_increasing
    pooled = BR.pooled_predictor([F, BR.build_frame(make_item("precursor", seed=1), last(make_item("precursor", seed=1)), FAST)], "m_stress")
    assert pooled["n_frames"] == 2 and pooled["p"] < 0.01 and pooled["consistent_sign"]
    null = BR.pooled_predictor([BR.build_frame(make_item("random", seed=s), last(make_item("random", seed=s)), FAST) for s in (0, 3)], "m_n2")
    assert null["p"] > 0.001


def test_shared_breaks_pool_evidence_and_find_the_common_precursor():
    base = make_item("precursor", seed=0)
    rng = np.random.default_rng(11)
    items = {}
    for k in range(3):
        fr = base.frame.copy()
        fr["value"] = fr["value"] + rng.normal(0, 0.004, len(fr))
        items[f"p{k}"] = bd.ItemSeries(f"p{k}", fr, base.columns)
    out = BR.shared_break_investigation(items, last(base), FAST)
    assert out["groups"] and out["question"] is not None
    assert out["leaders"] and out["leaders"][0]["column"] == "m_stress"
    assert BR.RP.identity_leak(out["question"].text) == ""


def test_shared_break_investigation_finds_nothing_for_independent_healthy_items():
    items = {f"h{k}": make_item("healthy", seed=k, name=f"h{k}") for k in range(3)}
    out = BR.shared_break_investigation(items, last(items["h0"]), FAST)
    assert out["groups"] == [] and out["leaders"] == [] and out["question"] is None


def test_unknown_anatomy_separates_negative_from_underpowered(stepped):
    st, rep, item, now = stepped
    inv = next(iter(st.investigations.values()))
    neg = dataclasses.replace(inv, verdict=BR.Verdict.UNKNOWN, results=(_r(BR.QUESTION_ORDER[0], BR.Outcome.PASSED), _r(BR.QUESTION_ORDER[1], BR.Outcome.FAILED)))
    blind = dataclasses.replace(inv, verdict=BR.Verdict.INSUFFICIENT_EVIDENCE, results=(_r(BR.QUESTION_ORDER[0], BR.Outcome.NOT_TESTABLE),))
    a, b = BR.unknown_anatomy(neg), BR.unknown_anatomy(blind)
    assert a["kind"] == "negative" and not a["more_data_could_help"] and a["blocking"] == BR.QUESTION_ORDER[1].value
    assert b["kind"] == "underpowered" and b["more_data_could_help"] and len(b["unasked"]) == 5


def test_queue_preview_respects_the_step_budget():
    item = make_item("precursor", seed=0)
    st = BR.new_state()
    BR.step(st, last(item), {"a": item}, cfg={**FAST, "max_per_step": 0})
    prev = BR.queue_preview(st, {"max_per_step": 3})
    assert len(prev) == len(st.investigations) and sum(not p["deferred"] for p in prev) == 3
    assert [p["priority"] for p in prev] == sorted((p["priority"] for p in prev), reverse=True)


def test_warning_profile_gives_notice_for_the_precursor_and_none_for_a_symptom():
    item = make_item("precursor", seed=0)
    F = BR.build_frame(item, last(item), FAST)
    res, rule = BR.run_ladder(F, 1, stop_early=True)
    prof = BR.warning_profile(F, rule)
    assert prof["n_breaks"] >= 3 and prof["warned"] >= prof["n_breaks"] // 2 and prof["median_lead"] >= 1
    # a symptom gate: fires on the realised loss itself (lagged one row), so it closes only after the break began
    v = pd.Series(F.v)
    X2 = F.X.copy()
    X2["sym"] = (-v.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0).values
    sym = BR.ColumnRule("sym", 1, float(np.quantile(X2["sym"], 0.8)))
    F2 = dataclasses.replace(F, X=X2)
    late = BR.warning_profile(F2, sym)
    assert late["median_lead"] <= prof["median_lead"]


def test_actionable_fails_closed_on_undefined_numbers():
    ok, why = BR.actionable({"n_breaks": 0, "warned": 0, "median_lead": float("nan"), "false_alarm_share": float("nan")})
    assert not ok and len(why) >= 2
    ok, _ = BR.actionable({"n_breaks": 4, "warned": 4, "median_lead": 5.0, "false_alarm_share": 0.1})
    assert ok


def test_queue_handoff_lists_next_actions_and_nothing_is_ready_before_replication(stepped):
    st, rep, item, now = stepped
    out = BR.queue_handoff(st, now)
    assert len(out["investigations"]) == len(st.investigations)
    assert out["ready_for_champion_challenger"] == []
    assert {r["next"] for r in out["investigations"]} <= {"run", "collect more evidence", "replicate on fresh data",
                                                          "retire or try a different predictor family", "monitor"}
    with pytest.raises(FirewallBreach):
        BR.queue_handoff(st, item.frame.index[10])


def test_failure_cause_bridge_never_guesses(stepped):
    st, *_ = stepped
    for inv in st.investigations.values():
        fc = BR.failure_cause_of(inv)
        assert (fc == BR.FailureCause.UNKNOWN or fc == BR.FailureCause.INSUFFICIENT_EVIDENCE) or inv.verdict == BR.Verdict.EXPLAINED
    assert sum(BR.unknown_kinds(st).values()) == len(st.investigations)


# ---------------------------------------------------------------- W-05: the cross-check uses the placebo-controlled engine
class _Inv:
    """Just what agreement_with_engine reads from an Investigation."""
    rule = None
    verdict = "UNKNOWN"


def test_w05_agreement_check_refuses_a_placebo_only_explanation(monkeypatch):
    item = make_item("precursor", seed=0)
    cfg = {"bd": FAST}
    bare = bd.explain_break(item, last(item), {**bd.PARAMS, **FAST}, 0)
    assert bare.explained, "planted precursor not explained by the bare engine under FAST settings"   # F22: was a pytest.skip that hid a regression
    ok = BR.agreement_with_engine(_Inv(), item, last(item), cfg, 0)
    assert ok["engine_status"] == "EXPLAINED" and ok["engine_columns"]              # a real link survives the placebo bar
    orig = bd.explain_break

    def rigged(it, as_of=None, cfg=None, seed=0):
        out = orig(it, as_of, cfg, seed)
        if it is not item:                                                            # every shifted-context placebo looks as strong as the real one
            return dataclasses.replace(out, best_t=bare.oos.t_diff + 1.0)
        return out
    monkeypatch.setattr(bd, "explain_break", rigged)
    assert bd.explain_break(item, last(item), {**bd.PARAMS, **FAST}, 0).explained     # the OLD call site would still have said EXPLAINED
    ag = BR.agreement_with_engine(_Inv(), item, last(item), cfg, 0)
    assert ag["engine_status"] == "UNKNOWN" and ag["engine_columns"] == () and not ag["both_explain"]
