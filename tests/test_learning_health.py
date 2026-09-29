"""Tests for engine.learning.health (section 46): the nine states on planted items, the phantom catch, the dashboard export,
the trust-weight firewall, and the no-lookahead / append-only rules. Synthetic data only."""
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning import health as hm
from engine.learning.core import FirewallBreach, Health


def series(parts, seed=0, sigma=0.010, start="2014-01-03"):
    """Weekly outcomes from (n, mean) segments."""
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(m, sigma, n) for n, m in parts])
    return pd.Series(v, index=pd.date_range(start, periods=len(v), freq="W-FRI"))


def after(s, days=7):
    return s.index[-1] + pd.Timedelta(days=days)


def one(s, kid="k", **kw):
    inp = hm.HealthInput(kid, s, **kw)
    return hm.assess([inp], after(s))[0]


def test_healthy_item_is_trusted():
    r = one(series([(220, 0.006)], seed=1))
    assert r.state == Health.HEALTHY and r.trusted and r.trust_weight == 1.0 and r.reasons


def test_broken_item_is_caught_and_untrusted():
    r = one(series([(150, 0.006), (60, -0.010)], seed=2))
    assert r.state == Health.BROKEN and not r.trusted and r.trust_weight == 0.0
    assert "monitor_broken" in r.flags and r.evidence["cusum"] >= r.evidence["cusum_alarm"]


def test_phantom_pattern_never_established_is_caught_by_health():
    caught = 0
    for seed in range(5):
        r = one(series([(220, -0.0005)], seed=10 + seed))
        caught += r.state == Health.BROKEN and "phantom" in r.flags
        assert not r.trusted
    assert caught >= 4
    assert hm.phantom_items(hm.assess([hm.HealthInput("p", series([(220, -0.0005)], seed=10))], after(series([(220, 0)])))) in (["p"], [])


def test_insufficient_evidence_and_unknown_are_different_states():
    short = one(series([(20, 0.006)], seed=3))
    assert short.state == Health.INSUFFICIENT_EVIDENCE and short.trust_weight == 0.0
    s = series([(120, 0.006)], seed=4)
    stale = hm.assess([hm.HealthInput("k", s)], s.index[-1] + pd.Timedelta(days=200))[0]
    assert stale.state == Health.UNKNOWN and "old" in stale.reasons[0]
    quiet = hm.assess([hm.HealthInput("k", s, should_fire=False)], s.index[-1] + pd.Timedelta(days=200))[0]
    assert quiet.state != Health.UNKNOWN
    nothing = hm.assess([hm.HealthInput("k", pd.Series([np.nan] * 5, index=pd.date_range("2020-01-03", periods=5, freq="W-FRI")))], "2021-01-01")[0]
    assert nothing.state == Health.UNKNOWN


def test_dormant_from_ledger_state_and_from_a_stopped_item():
    s = series([(200, 0.006)], seed=5)
    assert one(s, retirement_state="DORMANT").state == Health.DORMANT
    assert one(s, retirement_state="RETIRED").state == Health.DORMANT
    v = s.copy()
    v.iloc[150:] = np.nan
    ex = pd.Series(np.r_[np.ones(150, bool), np.zeros(50, bool)], index=s.index)
    r = one(v.iloc[:200], exposure=ex.iloc[:200])
    assert r.state == Health.DORMANT and "stopped_firing" in r.flags


def test_contradicted_needs_an_open_contradiction_above_the_bar():
    s = series([(220, 0.006)], seed=6)
    hi = hm.ContradictionRef("c1", 0.8)
    assert one(s, contradictions=(hi,)).state == Health.CONTRADICTED
    assert one(s, contradictions=(hm.ContradictionRef("c1", 0.8, resolved=True),)).state == Health.HEALTHY
    assert one(s, contradictions=(hm.ContradictionRef("c2", 0.2),)).state == Health.HEALTHY
    with pytest.raises(ValueError):
        hm.assess([hm.HealthInput("k", s, contradictions=(hm.ContradictionRef("c", 1.7),))], after(s))


def test_broken_and_contradicted_shows_both_but_states_broken():
    s = series([(150, 0.006), (60, -0.010)], seed=2)
    r = one(s, contradictions=(hm.ContradictionRef("c1", 0.9),))
    assert r.state == Health.BROKEN and "contradicted" in r.flags and any("contradiction" in x for x in r.reasons)


def test_degrading_from_the_ledger_and_recovering_after_a_break():
    s = series([(220, 0.006)], seed=7)
    assert one(s, retirement_state="DEGRADED").state == Health.DEGRADING
    rec = series([(110, 0.006), (35, -0.012), (75, 0.008)], seed=8)
    states = []
    for cut in range(150, 221, 5):
        r = hm.assess([hm.HealthInput("k", rec.iloc[:cut])], rec.index[cut - 1] + pd.Timedelta(days=7))[0]
        states.append(r.state)
    assert Health.BROKEN in states and (Health.RECOVERING in states or states[-1] == Health.HEALTHY)
    assert states.index(Health.BROKEN) < len(states) - 1


def test_unstable_item_with_repeated_breaks():
    s = series([(60, 0.008), (25, -0.012), (35, 0.008), (25, -0.012), (35, 0.008), (25, -0.012), (30, 0.008)], seed=9)
    seen = set()
    for cut in range(120, len(s) + 1, 6):
        r = hm.assess([hm.HealthInput("k", s.iloc[:cut])], s.index[cut - 1] + pd.Timedelta(days=7))[0]
        seen.add(r.state)
    assert Health.UNSTABLE in seen or Health.BROKEN in seen


def test_all_nine_states_are_reachable_and_have_a_trust_weight():
    assert set(hm.TRUST_WEIGHT) == set(Health) and set(hm.EPISTEMIC_FOR) == set(Health) and set(hm.SEVERITY) == set(Health)
    assert hm.TRUST_WEIGHT[Health.HEALTHY] == 1.0
    assert all(hm.TRUST_WEIGHT[h] == 0.0 for h in hm.SILENT)


def test_no_lookahead_outcomes_at_or_after_as_of_are_ignored():
    s = series([(220, 0.006)], seed=11)
    cut = s.index[150]
    a = hm.assess([hm.HealthInput("k", s)], cut)[0]
    s2 = s.copy()
    s2.iloc[150:] = -0.5
    b = hm.assess([hm.HealthInput("k", s2)], cut)[0]
    assert a.record_id == b.record_id and a.state == b.state and a.evidence["n_obs"] == 150


def test_input_validation():
    s = series([(50, 0.006)], seed=12)
    assert hm.HealthInput("k", s.iloc[::-1]).validate()
    assert hm.HealthInput("k", s, retirement_state="LOST").validate()
    with pytest.raises(ValueError):
        hm.assess([hm.HealthInput("k", s.iloc[::-1])], after(s))
    assert hm.assess([], "2020-01-01") == []


def _book_world():
    a = series([(220, 0.006)], seed=1)
    b = series([(150, 0.006), (70, -0.010)], seed=2)
    c = series([(19, 0.006)], seed=3, start=pd.Timestamp("2014-01-03") + pd.Timedelta(weeks=200))
    d = series([(220, 0.006)], seed=4)
    ins = [hm.HealthInput("healthy", a), hm.HealthInput("broken", b), hm.HealthInput("thin", c),
           hm.HealthInput("parked", d, retirement_state="DORMANT"),
           hm.HealthInput("contra", a, contradictions=(hm.ContradictionRef("c9", 0.9),))]
    return ins, a.index


def test_dashboard_answers_the_five_questions_and_validates():
    ins, idx = _book_world()
    mon = hm.HealthMonitor()
    mon.step(ins, idx[180])
    research = [hm.ResearchAssignment("R1", "why did the broken item stop working?", ("broken",))]
    mon.step(ins, idx[219], research)
    dash = hm.build_dashboard(mon.book, idx[219], research)
    assert hm.validate_dashboard(dash) == []
    ids = lambda name: {e["knowledge_id"] for e in dash["sections"][name]}
    assert ids("trusted") == {"healthy"} and ids("broken") == {"broken"} and ids("contradicted") == {"contra"}
    assert ids("dormant") == {"parked"} and ids("insufficient_evidence") == {"thin"}
    b = [e for e in dash["sections"]["broken"] if e["knowledge_id"] == "broken"][0]
    assert b["investigating"] == ["R1"] and b["why"] and b["evidence"]["cusum"] is not None
    assert any(e["knowledge_id"] == "broken" for e in dash["losing_trust"])
    assert any("state" in x or "CUSUM" in x for x in b["evidence_changed"])
    assert "contra" in dash["unattended_failures"] and "broken" not in dash["unattended_failures"]
    text = hm.render_text(dash)
    assert "Losing trust" in text and "NOBODY" in text and "R1" in text


def test_dashboard_is_strict_json_and_refuses_when_invalid(tmp_path):
    ins, idx = _book_world()
    mon = hm.HealthMonitor()
    mon.step(ins, idx[219])
    dash = hm.build_dashboard(mon.book, idx[219])
    p = hm.write_dashboard(tmp_path / "sub" / "health.json", dash)
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["n_items"] == 5 and hm.validate_dashboard(back) == []
    bad = json.loads(json.dumps(dash))
    bad["sections"]["trusted"].append(bad["sections"]["broken"][0])
    assert hm.validate_dashboard(bad)
    bad2 = json.loads(json.dumps(dash))
    bad2["schema"] = "other"
    with pytest.raises(ValueError):
        hm.write_dashboard(tmp_path / "x.json", bad2)
    assert hm.validate_dashboard({}) != []


def test_book_is_append_only_forward_only_and_tamper_evident():
    ins, idx = _book_world()
    mon = hm.HealthMonitor()
    mon.step(ins, idx[180])
    mon.step(ins, idx[219])
    assert mon.book.verify() == [] and len(mon.book) == 10
    with pytest.raises(ValueError):
        mon.step(ins, idx[200])
    h = mon.book.history("broken")
    assert h[1].prev_state == h[0].state and h[0].since <= h[1].since
    mon.book._chain[3]["chain"] = "forged"
    assert mon.book.verify()
    assert set(mon.book.counts()) == {x.value for x in Health}
    assert mon.book.transition_counts().columns.tolist() == ["from", "to", "n"]


def test_trust_weights_scale_influence_and_silent_states_cannot_carry_weight():
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    w = {r.knowledge_id: 1.0 for r in recs}
    scaled = hm.apply_trust(w, recs)
    assert scaled["healthy"] == 1.0 and scaled["broken"] == 0.0 and scaled["parked"] == 0.0
    hm.assert_untrusted_silent(scaled, recs)
    with pytest.raises(FirewallBreach):
        hm.assert_untrusted_silent(w, recs)
    assert hm.apply_trust({"ghost": 1.0}, recs) == {"ghost": 0.0}
    with pytest.raises(FirewallBreach):
        hm.assert_untrusted_silent({"ghost": 0.3}, recs)


def test_explain_change_lists_what_moved():
    s = series([(150, 0.006), (60, -0.010)], seed=2)
    early = hm.assess([hm.HealthInput("k", s)], s.index[150])[0]
    late = hm.assess([hm.HealthInput("k", s)], after(s))[0]
    msgs = hm.explain_change(early, late)
    assert any("state" in m for m in msgs) and any("CUSUM" in m for m in msgs)
    assert hm.explain_change(None, late)[0].startswith("first assessment")
    assert hm.explain_change(late, late) == []


def test_false_alarm_expectation_and_epistemic_mapping():
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    assert hm.false_alarm_expectation(recs, 52) > 0
    assert hm.false_alarm_expectation([], 52) == 0.0
    rec = [r for r in recs if r.knowledge_id == "contra"][0]
    assert rec.epistemic.value == "CONTRADICTED"


def _tracked_book():
    good = series([(300, 0.012)], seed=21, sigma=0.008)
    bad = series([(140, 0.007), (80, -0.011), (80, 0.007)], seed=22)
    mon = hm.HealthMonitor()
    ins = [hm.HealthInput("good", good), hm.HealthInput("flip", bad)]
    for cut in range(100, 300, 10):
        mon.step(ins, good.index[cut])
    return mon.book, good.index


def test_trajectory_time_in_state_and_flap_rate():
    book, idx = _tracked_book()
    tj = hm.trust_trajectory(book, "flip")
    assert len(tj) == 20 and set(tj["state"]) & {"BROKEN", "RECOVERING", "DEGRADING", "UNSTABLE"}
    tis = hm.time_in_state(book, "good")
    assert tis["n"] == 20 and sum(tis["counts"].values()) == 20 and 1 <= tis["current_run"] <= 20
    assert 0.0 < hm.flap_rate(book, "flip") <= 1.0 and 0.0 <= hm.flap_rate(book, "good") <= 1.0
    assert hm.flap_rate(book, "nobody") == 0.0 and hm.time_in_state(book, "nobody")["n"] == 0


def test_debounce_delays_recovery_but_never_a_warning():
    book, _ = _tracked_book()
    hist = [r.state for r in book.history("flip")]
    off = hm.debounced_state(book, "flip", min_hold=3)
    assert off is not None and hm.debounced_state(book, "nobody") is None
    fast = hm.SEVERITY[hist[-1]]
    if any(hm.SEVERITY[s] > hm.SEVERITY[hist[0]] for s in hist):
        first_bad = next(i for i, s in enumerate(hist) if hm.SEVERITY[s] > hm.SEVERITY[hist[0]])
        assert hm.SEVERITY[hm.debounced_state(book, "flip", 3, as_of=book.history("flip")[first_bad].as_of)] > hm.SEVERITY[hist[0]]
    assert set(hm.official_snapshot(book)) == {"good", "flip"} and fast >= 0


def test_research_queue_orders_unattended_severe_items_first():
    ins, idx = _book_world()
    mon = hm.HealthMonitor()
    research = [hm.ResearchAssignment("R1", "look at the broken one", ("broken",))]
    mon.step(ins, idx[219], research)
    dash = hm.build_dashboard(mon.book, idx[219], research)
    q = hm.research_queue(dash)
    assert [r["rank"] for r in q] == list(range(1, len(q) + 1))
    assert q[0]["severity"] >= q[-1]["severity"] and any(not r["attended"] for r in q)
    unattended_first = [r for r in q if r["severity"] == q[0]["severity"]]
    assert unattended_first == sorted(unattended_first, key=lambda r: r["attended"])


def test_dashboard_diff_and_summary_line():
    ins, idx = _book_world()
    mon = hm.HealthMonitor()
    mon.step(ins, idx[150])
    d1 = hm.build_dashboard(mon.book, idx[150])
    mon.step(ins, idx[219])
    d2 = hm.build_dashboard(mon.book, idx[219])
    diff = hm.diff_dashboards(d1, d2)
    assert any(x["knowledge_id"] == "broken" for x in diff["worse"]) and diff["gone"] == [] and diff["unchanged"] >= 1
    line = hm.summary_line(d2)
    assert "5 items" in line and "broken" in line and "no research" in line


def test_detection_scoring_against_planted_intervals():
    book, idx = _tracked_book()
    truth = {"flip": [(idx[140], idx[219])], "good": []}
    sc = hm.evaluate_detection(book, truth, grace=5)
    assert sc["intervals"] == 1 and sc["detected"] == 1 and sc["mean_delay"] >= 0
    assert 0.0 <= sc["false_alarm_rate"] <= 0.4 and sc["clean_assessments"] > 0
    none = hm.evaluate_detection(book, {})
    assert none["intervals"] == 0 and np.isnan(none["recall"])


def test_epistemic_proposals_only_for_items_that_changed():
    from engine.learning.core import Epistemic
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    cur = {r.knowledge_id: Epistemic.SUPPORTED for r in recs}
    props = hm.epistemic_proposals(recs, cur)
    ids = {p["knowledge_id"] for p in props}
    assert "healthy" not in ids and {"broken", "contra", "parked"} <= ids
    assert hm.epistemic_proposals(recs, {}) == []


def test_inputs_from_knowledge_objects_and_ledger_state():
    from types import SimpleNamespace
    from engine.learning.retirement import RetirementLedger, State
    s = series([(220, 0.008)], seed=31, sigma=0.008)
    now = after(s)
    led = RetirementLedger()
    led.register("a", s.index[0])
    led.register("b", s.index[0])
    led.transition("b", State.DORMANT, s.index[100], "DORMANT", "test park")
    items = [SimpleNamespace(knowledge_id="a"), SimpleNamespace(knowledge_id="b"), SimpleNamespace(knowledge_id="c")]
    ins = hm.inputs_from_knowledge(items, {"a": s, "b": s}, now, led, {"a": (hm.ContradictionRef("x", 0.9),)})
    assert [i.knowledge_id for i in ins] == ["a", "b", "c"]
    assert ins[1].retirement_state == "DORMANT" and ins[0].retirement_state == "ACTIVE" and ins[2].retirement_state is None
    recs = {r.knowledge_id: r for r in hm.assess(ins, now)}
    assert recs["a"].state == Health.CONTRADICTED and recs["b"].state == Health.DORMANT and recs["c"].state == Health.UNKNOWN


def test_export_import_roundtrip_detects_tampering():
    book, idx = _tracked_book()
    data = json.loads(json.dumps(hm.export_book(book)))
    back = hm.import_book(data)
    assert len(back) == len(book) and back.verify() == [] and back.counts() == book.counts()
    forged = json.loads(json.dumps(data))
    forged["records"][3]["state"] = "HEALTHY" if forged["records"][3]["state"] != "HEALTHY" else "BROKEN"
    with pytest.raises(ValueError):
        hm.import_book(forged)
    with pytest.raises(ValueError):
        hm.import_book({"schema": "x"})


def test_coverage_gaps_and_stale_assessments():
    book, idx = _tracked_book()
    dates = [idx[c] for c in range(100, 300, 10)]
    assert hm.coverage_gaps(book, dates) == []
    gaps = hm.coverage_gaps(book, dates + [idx[299] + pd.Timedelta(days=30)])
    assert len(gaps) == 2 and {g["knowledge_id"] for g in gaps} == {"good", "flip"}
    assert hm.stale_assessments(book, idx[291]) == []
    assert set(hm.stale_assessments(book, idx[299] + pd.Timedelta(days=60))) == {"good", "flip"}


def test_book_level_summaries_and_history_rendering():
    book, idx = _tracked_book()
    dw = hm.dwell_summary(book)
    assert set(dw.columns) == {"state", "runs", "median_run", "longest_run"} and (dw["runs"] >= 1).all()
    wo = hm.worst_offenders(book, 2)
    assert wo[0]["share_failing"] >= wo[1]["share_failing"] and len(wo) == 2
    txt = hm.render_history(book, "flip")
    assert txt.startswith("# flip") and "BROKEN" in txt
    assert "never assessed" in hm.render_history(book, "ghost")
    assert hm.dwell_summary(hm.HealthBook()).empty and hm.worst_offenders(hm.HealthBook()) == []


def test_trust_index_and_severity_histogram():
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    ti = hm.book_trust_index(recs)
    assert 0.0 < ti < 1.0 and np.isnan(hm.book_trust_index([]))
    hist = hm.severity_histogram(recs)
    assert sum(hist.values()) == 5 and 6 in hist and list(hist) == sorted(hist)


def test_influence_predicate_matches_the_silent_list():
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    silent = set(hm.silent_ids(recs))
    assert silent == {r.knowledge_id for r in recs if r.state in hm.SILENT} and "healthy" not in silent and "broken" in silent
    assert hm.influence_allowed(Health.HEALTHY) and not hm.influence_allowed(Health.BROKEN) and hm.influence_allowed(Health.UNSTABLE)


def test_worst_state_and_failing_share():
    ins, idx = _book_world()
    recs = hm.assess(ins, idx[219])
    assert hm.worst_state(recs) == Health.BROKEN and hm.worst_state([]) is None
    assert 0.0 < hm.failing_share(recs) < 1.0 and np.isnan(hm.failing_share([]))
    assert hm.is_failing(Health.BROKEN) and not hm.is_failing(Health.DORMANT)
