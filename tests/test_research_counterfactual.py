"""Tests for engine/research/counterfactual.py (C66 section 8, "could I have known?"). Synthetic data only.

Every mechanism has a planted case it must catch, a null case where it must find nothing, and an empty case."""
import dataclasses
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from engine import pit
from engine.learning.core import FirewallBreach, Provenance
from engine.research import counterfactual as cf
from engine.research.core import Availability, Knowability

warnings.filterwarnings("ignore")
CFG = cf.CounterfactualConfig()
_CASES: dict = {}


def case(kind, seed=1):
    if (kind, seed) not in _CASES:
        _CASES[(kind, seed)] = cf.make_planted_case(kind, seed)
    return _CASES[(kind, seed)]


def report(kind, seed=1):
    key = ("rep", kind, seed)
    if key not in _CASES:
        c = case(kind, seed)
        _CASES[key] = cf.step(c.store, [c.event], c.now, c.providers)[0]
    return _CASES[key]


# ---------------------------------------------------------------------------------------------------- availability classes

T0, T1 = "2021-03-01", "2021-03-02"


@pytest.mark.parametrize("eff,av,want", [
    ("2021-02-20", "2021-02-22", Availability.KNOWN_BEFORE_EVENT),
    ("2021-03-01", "2021-03-01", Availability.KNOWN_BEFORE_EVENT),       # public at the close itself
    ("2021-03-01", "2021-03-02", Availability.SIMULTANEOUS),             # filed after the close, public at the fill session
    ("2021-03-02", "2021-03-02", Availability.SIMULTANEOUS),
    ("2021-02-25", "2021-03-05", Availability.KNOWN_ONLY_AFTER_EVENT),   # existed, published late
    ("2021-03-04", "2021-03-04", Availability.UNAVAILABLE),              # came into being after the boundary
    ("2021-03-04", "2021-03-01", Availability.UNCERTAIN),                # published before it happened: impossible
    ("", "2021-03-01", Availability.UNCERTAIN),
    ("2021-02-25", "", Availability.UNCERTAIN),
])
def test_availability_classes(eff, av, want):
    assert cf.classify_availability(eff, av, T0, T1)[0] == want


def test_availability_lag_and_provenance_and_boundary():
    k, lag = cf.classify_availability("2021-02-22", "2021-02-24", T0, T1)
    assert k == Availability.KNOWN_BEFORE_EVENT and lag == -3
    assert cf.classify_availability("2021-02-22", "2021-02-24", T0, T1, provenance_ok=False)[0] == Availability.UNCERTAIN
    assert cf.classify_availability("2021-02-25", "2021-03-03", T0, T1, boundary_sessions=1)[0] == Availability.SIMULTANEOUS
    assert cf.classify_availability("2021-02-25", "2021-03-03", T0, T1, boundary_sessions=0)[0] == Availability.KNOWN_ONLY_AFTER_EVENT
    assert cf.classify_availability("garbage", "2021-03-03", T0, T1)[0] == Availability.UNCERTAIN


def test_config_check_catches_bad_thresholds():
    assert CFG.check() == []
    assert cf.CounterfactualConfig(weak=0.5, moderate=0.4).check()
    assert cf.CounterfactualConfig(sample_rate=1.5).check()
    assert cf.CounterfactualConfig(min_history=500).check()
    with pytest.raises(ValueError):
        cf.build_snapshot(case("unknown").store, case("unknown").event.decision_ts, cfg=cf.CounterfactualConfig(weak=0.9))


def test_event_spec_validation():
    c = case("unknown")
    e = c.event
    assert e.validate() == []
    assert cf.EventSpec.make("A", "2021-03-01", "2021-03-05", 1, model_score=0.5, score_asof="2021-03-04").validate()      # future score
    assert cf.EventSpec.make("A", "2021-03-01", "2021-03-05", 1, model_score=0.5).validate()                               # undated score
    assert dataclasses.replace(e, direction=0).validate()
    assert dataclasses.replace(e, realized_return=-0.1).validate()                                                         # contradicts direction
    assert dataclasses.replace(e, event_start=e.decision_ts).validate()
    assert dataclasses.replace(e, ticker="").validate()


# ---------------------------------------------------------------------------------------------------- planted worlds


@pytest.mark.parametrize("kind", cf.PLANTED_KINDS)
def test_planted_cause_is_classified(kind):
    r = report(kind)
    assert r.knowability == case(kind).truth, (kind, r.knowability, r.scores, r.notes)
    assert r.validate() == []
    assert r.confidence_in_classification.check() == []


def test_precursor_reports_available_evidence_and_lead_time():
    r = report("precursor")
    keys = {i.key for i in r.information_that_would_have_been_available}
    assert "TECHNICAL.squeeze_rank" in keys
    assert all(i.availability == Availability.KNOWN_BEFORE_EVENT for i in r.information_that_would_have_been_available)
    assert r.scores["combined"] >= CFG.strong and r.scores["n_supporting_domains"] >= 2
    lts = cf.lead_times(r)
    assert lts and all(lt.sessions_before >= 0 for lt in lts)
    assert cf.earliest_warning(r) is not None


def test_news_cause_was_unavailable_not_predictable():
    r = report("news")
    earn = [i for i in r.information_that_was_unavailable if i.domain == cf.Domain.EVENT]
    assert earn and earn[0].availability == Availability.UNAVAILABLE
    assert any(u.used_for == "cause" and u.relevance >= CFG.min_cause_relevance for u in r.future_information_used_by_auditor)
    assert "PREDICTABLE" not in r.knowability.value.replace("INFORMATIONALLY_UNAVAILABLE", "")
    assert r.attribution.idio_share >= CFG.idio_share


def test_external_shock_is_attributed_to_market():
    r = report("external")
    assert r.attribution.complete and r.attribution.market_share + r.attribution.peer_share >= CFG.external_share
    assert r.event.direction == -1 and r.attribution.realized < 0
    assert any("market" in n for n in r.notes)


def test_unknown_stays_unknown_with_capped_confidence():
    r = report("unknown")
    assert r.knowability == Knowability.UNKNOWN
    assert r.confidence_in_classification.overall <= CFG.unknown_conf_cap + 1e-9
    assert not r.information_that_would_have_been_available or r.scores["strongest"] < CFG.min_pointer


def test_null_world_never_finds_predictability_from_noise():
    """Ten seeds of a world with an untraceable jump: the pipeline must not conjure a predictable move out of noise."""
    labels = []
    for seed in range(10, 20):
        labels.append(report("unknown", seed).knowability)
    assert Knowability.PREDICTABLE not in labels
    assert sum(k == Knowability.UNKNOWN for k in labels) >= 3
    null_combined = np.mean([report("unknown", sd).scores["combined"] for sd in range(10, 20)])
    assert null_combined < report("precursor").scores["combined"]         # noise scores below a planted precursor


def test_data_failure_is_not_read_as_unknowable():
    r = report("data_failure")
    assert r.knowability == Knowability.DATA_FAILURE
    assert "VOLUME" in r.knowledge_state_at_decision.missing_domains()
    assert r.confidence_in_classification.caps


def test_all_ten_domains_reconstructed_in_planted_world():
    ks = report("precursor").knowledge_state_at_decision
    assert ks.missing_domains() == () and set(ks.coverage()) == {d.value for d in cf.ALL_DOMAINS}


# ---------------------------------------------------------------------------------------------------- the state never sees the future


def test_knowledge_state_holds_nothing_after_the_decision():
    r = report("news")
    ks = r.knowledge_state_at_decision
    assert ks.validate() == []
    assert all(i.availability == Availability.KNOWN_BEFORE_EVENT and i.available <= ks.decision_ts for i in ks.items)
    assert ks.guard_max_available <= ks.decision_ts and ks.guard_denied == 0
    assert not any(i.key == "EVENT.EARN" for i in ks.items)      # the post-decision filing is in the auditor's list only


def test_later_learned_knowledge_is_unavailable_not_in_state():
    r = report("precursor")
    names = {i.key for i in r.information_that_was_unavailable}
    assert "PATTERN.learned_later_squeeze" in names and "MEMORY.pattern:M_late" in names
    state_keys = {i.key for i in r.knowledge_state_at_decision.items}
    assert "PATTERN.learned_later_squeeze" not in state_keys and "MEMORY.pattern:M_late" not in state_keys
    assert "PATTERN.quiet_pattern" in state_keys and "MEMORY.pattern:M_old" in state_keys


def test_state_rejects_a_planted_future_item():
    r = report("precursor")
    ks = r.knowledge_state_at_decision
    late = cf.InfoItem.make(cf.Domain.PRICE, "peek", 1.0, r.event.event_end, r.event.event_end, "x", ks.decision_ts, r.event.event_start)
    assert dataclasses.replace(ks, items=ks.items + (late,)).validate()
    assert dataclasses.replace(ks, guard_max_available=r.event.event_end).validate()
    assert dataclasses.replace(ks, guard_denied=2).validate()


def test_future_invariance_holds_for_the_real_pipeline():
    c = case("precursor")
    res = cf.verify_future_invariance(c.store, c.event, c.providers)
    assert res.invariant and res.log_clean, res.detail


def test_future_invariance_catches_a_provider_that_peeks_through_the_store():
    """A feature function that bypasses the Guard and reads the raw store sees the real future: the digest must change."""
    c = case("precursor")
    raw = c.store.source("prices").frames["Close"]

    def peeking(g, ticker, asof):
        fr = g._store.source("prices").frames["Close"]           # reflects into the Guard's private store: the fence is bypassed
        return {"f_peek": float(fr[ticker].iloc[c.decision_index + 3] / fr[ticker].iloc[c.decision_index] - 1.0)}

    res = cf.verify_future_invariance(c.store, c.event, cf.Providers(feature_fn=peeking))
    assert not res.invariant and "reads the future" in res.detail


def test_guard_stops_a_provider_that_asks_for_the_future():
    c = case("precursor")

    def greedy(g, ticker, asof):
        return {"x": float(g.wide("prices", "Close", tickers=[ticker], end=asof + pd.Timedelta(days=9)).iloc[-1, 0])}

    with pytest.raises(pit.LookAheadError):
        cf.step(c.store, [c.event], c.now, cf.Providers(feature_fn=greedy))


def test_selfcheck_all_defences_fire():
    c = case("news")
    out = cf.selfcheck(c.store, c.event, c.providers)
    assert out and all(out.values()), out


def test_log_problems_flags_a_leaking_log():
    log = pit.AuditLog()
    log.record(op="wide", source="prices", as_of="2021-03-01", requested="x", verdict="ok", rows=3, max_effective="2021-03-03",
               max_avail="2021-03-03")
    assert cf.log_problems(log, "2021-03-01")
    clean = pit.AuditLog()
    clean.record(op="wide", source="prices", as_of="2021-03-01", requested="x", verdict="ok", rows=3, max_effective="2021-03-01",
                 max_avail="2021-03-01")
    assert cf.log_problems(clean, "2021-03-01") == []


def test_outcome_not_matured_is_not_assessed():
    c = case("precursor")
    assert cf.step(c.store, [c.event], pd.Timestamp(c.event.event_end), c.providers) == []          # ends ON now: not known yet
    assert cf.step(c.store, [c.event], pd.Timestamp(c.event.event_start), c.providers) == []
    assert cf.auditor_as_of_for(c.event, pd.Timestamp(c.event.event_end)) is None
    assert cf.auditor_as_of_for(c.event, c.now) is not None


def test_auditor_guard_refuses_a_window_that_is_not_after_the_event():
    c = case("precursor")
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    state, side = cf.reconstruct_state(snap, c.event, c.providers)
    with pytest.raises(FirewallBreach):
        cf.audit_event(c.store, snap, c.event, state, side, c.event.event_end, c.providers)


def test_snapshot_for_another_day_is_refused():
    c = case("precursor")
    snap = cf.build_snapshot(c.store, pd.Timestamp(c.event.decision_ts) - pd.Timedelta(days=3))
    with pytest.raises(FirewallBreach):
        cf.reconstruct_state(snap, c.event, c.providers)


# ---------------------------------------------------------------------------------------------------- the gate


def test_gate_refuses_before_maturity_and_releases_after():
    r = report("precursor")
    g = cf.ClassificationGate()
    rid = g.add(r)
    with pytest.raises(FirewallBreach):
        g.release_one(rid, r.matured_at)                 # ON the maturity date: not yet
    with pytest.raises(FirewallBreach):
        g.release_one(rid, r.event.decision_ts)
    row = g.release_one(rid, pd.Timestamp(r.matured_at) + pd.Timedelta(days=1))
    assert row["label"] == r.knowability.value
    assert g.release(r.matured_at) == [] and g.withheld[-1]["why"] == "not_matured"
    assert len(g.release(pd.Timestamp(r.matured_at) + pd.Timedelta(days=1))) == 1


def test_gate_withholds_same_year_reruns():
    r = report("precursor")
    g = cf.ClassificationGate()
    rid = g.add(r)
    later = pd.Timestamp(r.matured_at) + pd.Timedelta(days=5)
    with pytest.raises(FirewallBreach):
        g.release_one(rid, later, replaying_years=[r.event.real_year])
    assert g.release(later, replaying_years=[r.event.real_year]) == []
    assert g.withheld[-1]["why"] == "same_year_rerun"
    assert g.training_labels(later, replaying_years=[r.event.real_year]).empty
    assert len(g.release(later, replaying_years=[r.event.real_year + 1])) == 1


@pytest.mark.parametrize("effect", ["SELECTION", "RANKING", "POSITION_SIZE", "DIRECTION", "TIMING", "EXIT", "STOP"])
def test_classification_may_not_drive_a_live_decision(effect):
    r = report("precursor")
    g = cf.ClassificationGate()
    g.add(r)
    with pytest.raises(FirewallBreach):
        g.release(pd.Timestamp(r.matured_at) + pd.Timedelta(days=5), effect=effect)


def test_released_rows_are_identity_free():
    from engine.learning.trader_view import find_violations
    r = report("news")
    g = cf.ClassificationGate()
    g.add(r)
    rows = g.release(pd.Timestamp(r.matured_at) + pd.Timedelta(days=5))
    assert rows and find_violations(rows[0]) == []
    blob = json.dumps(rows)
    assert r.event.ticker not in blob and r.event.decision_ts[:4] not in blob and r.report_id not in blob and r.event.event_id not in blob


def test_feeding_the_classification_back_is_refused():
    r = report("precursor")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback(r, "model input")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback(r.knowability, "label")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback({"features": {"x": 1.0}, "knowability": "PREDICTABLE"}, "row")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback(pd.DataFrame({"x": [1.0], "cf_confidence": [0.4]}), "frame")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback([{"a": 1}, [r.confidence_in_classification]], "nested")
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback(r.knowledge_state_at_decision, "state")
    cf.refuse_feedback(pd.DataFrame({"x": [1.0], "y": [2.0]}), "clean frame")
    cf.refuse_feedback({"features": {"x": 1.0}}, "clean row")


def _fabricated(n=24):
    base = report("precursor")
    labels = list(cf._ORDER)
    out = []
    for i in range(n):
        k = labels[i % len(labels)]
        e = dataclasses.replace(base.event, event_id=f"E{i:03d}")
        out.append(dataclasses.replace(base, event=e, knowability=k, scores={**base.scores, "combined": (i % 7) / 7.0}))
    return out


def test_renamed_copy_of_the_classification_is_detected():
    reps = _fabricated()
    ids = [r.event.event_id for r in reps]
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"noise_a": rng.normal(size=len(ids)), "noise_b": rng.normal(size=len(ids))}, index=ids)
    cf.refuse_disguised_feedback(X, reps)                        # null: unrelated columns pass
    X["momentum_score"] = [cf._ordinal(r.knowability) * 1.7 + 3.0 for r in reps]   # the label under another name
    assert "momentum_score" in cf.disguised_feedback_columns(X, reps)
    with pytest.raises(FirewallBreach):
        cf.refuse_disguised_feedback(X, reps)
    assert cf.disguised_feedback_columns(X.iloc[:3], reps) == {}


def test_training_labels_are_dated_at_maturity():
    r = report("precursor")
    g = cf.ClassificationGate()
    g.add(r)
    assert g.training_labels(r.matured_at).empty
    lab = g.training_labels(pd.Timestamp(r.matured_at) + pd.Timedelta(days=1))
    assert list(lab.columns) == ["report_id", "cf_knowability", "cf_confidence", "cf_matured_at"] and len(lab) == 1
    cf.assert_label_consumer_safe(lab, pd.Timestamp(r.matured_at) + pd.Timedelta(days=1))
    with pytest.raises(FirewallBreach):
        cf.assert_label_consumer_safe(lab, r.event.decision_ts)
    with pytest.raises(FirewallBreach):
        cf.refuse_feedback(lab, "training frame handed to the trader")
    cf.assert_label_consumer_safe(pd.DataFrame(columns=lab.columns), "2021-01-01")


def test_matured_record_carries_only_labels():
    rec = report("news").matured_record()
    assert set(rec.payload) == {"knowability", "confidence", "category", "direction", "n_available", "n_unavailable"}
    with pytest.raises(FirewallBreach):
        rec.gate(rec.matured_at)


# ---------------------------------------------------------------------------------------------------- report integrity


def test_report_validate_catches_planted_defects():
    r = report("precursor")
    assert r.validate() == []
    bad_unavail = dataclasses.replace(r, information_that_was_unavailable=r.information_that_would_have_been_available[:1])
    assert any("was in fact known" in e for e in bad_unavail.validate())
    bad_avail = dataclasses.replace(r, information_that_would_have_been_available=r.information_that_was_unavailable[:1])
    assert bad_avail.validate()
    assert dataclasses.replace(r, information_that_would_have_been_available=()).validate()
    early = dataclasses.replace(r, auditor_as_of=r.event.event_end)
    assert any("not matured" in e for e in early.validate())
    unk = report("unknown")
    over = dataclasses.replace(unk, confidence_in_classification=dataclasses.replace(unk.confidence_in_classification, overall=0.99))
    assert any("near-certainty" in e for e in over.validate())


def test_timestamp_audit_detects_a_misclassified_item():
    r = report("precursor")
    assert cf.audit_report_timestamps(r) == []
    ks = r.knowledge_state_at_decision
    forged = dataclasses.replace(ks.items[0], availability=Availability.KNOWN_ONLY_AFTER_EVENT)
    r2 = dataclasses.replace(r, knowledge_state_at_decision=dataclasses.replace(ks, items=(forged,) + ks.items[1:]))
    assert cf.audit_report_timestamps(r2)


def test_report_is_deterministic():
    c = case("news")
    assert cf.replay_matches(c.store, c.event, c.now, c.providers)
    a, b = report("news"), report("news")
    assert cf.compare_reports(a, b) == {}
    other = dataclasses.replace(a, knowability=Knowability.UNKNOWN)
    assert "knowability" in cf.compare_reports(a, other)
    with pytest.raises(ValueError):
        cf.compare_reports(a, dataclasses.replace(report("precursor"), event=dataclasses.replace(a.event, event_id="other")))


def test_to_dict_is_json_and_round_trips_key_fields():
    d = report("news").to_dict()
    text = json.dumps(d)
    assert "NaN" not in text
    back = json.loads(text)
    assert back["knowability"] == "INFORMATIONALLY_UNAVAILABLE" and back["namespace"] == "MATURED_RESEARCH_STATE"
    for k in ("knowledge_state_at_decision", "future_information_used_by_auditor", "information_that_would_have_been_available",
              "information_that_was_unavailable", "confidence_in_classification"):
        assert k in back


def test_confidence_components_and_shared_dimensions():
    conf = report("precursor").confidence_in_classification
    c = conf.as_confidence()
    assert c.check() == [] and c.truth == conf.overall and c.usefulness is None and c.transfer is None
    dq = report("data_failure").confidence_in_classification
    assert dq.coverage < conf.coverage


# ---------------------------------------------------------------------------------------------------- attribution


def test_attribution_arithmetic():
    a = cf.attribute_move(0.10, 1.0, 100, 0.10, 0.10)
    assert a.complete and a.market_share == pytest.approx(1.0) and a.idio_share == pytest.approx(0.0)
    b = cf.attribute_move(0.10, 1.0, 100, 0.02, 0.02)
    assert b.idio_share == pytest.approx(0.08 / 0.10) and b.check() == []
    opp = cf.attribute_move(0.10, 1.0, 100, -0.05, None)       # market opposed the move: it explains none of it
    assert opp.market_share == 0.0 and opp.idio_share == pytest.approx(1.0)
    inc = cf.attribute_move(0.10, None, 0, 0.05, 0.05)
    assert not inc.complete and inc.idio_share == 1.0 and inc.market_component is None
    assert not cf.attribute_move(None, 1.0, 50, 0.01, 0.0).complete


def test_incomplete_attribution_never_becomes_external_or_unavailable():
    r = report("external")
    att = cf.attribute_move(r.attribution.realized, None, 0, None, None)
    audit = cf.AuditResult(att.realized, (), (), att, (), r.auditor_as_of, "", "")
    kn, notes = cf.classify_knowability(r.knowledge_state_at_decision, {"combined": 0.0, "direction": 0.0, "opposing": 0.0, "agreeing": 0.0,
                                                                        "n_supporting_domains": 0.0, "strongest": 0.0}, audit, CFG, ())
    assert kn == Knowability.UNKNOWN and any("incomplete" in n for n in notes)


def test_beta_is_past_only():
    c = case("external")
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    beta, n = cf.market_beta(snap, "T00", 120)
    assert n == 120 and 0.7 < beta < 1.7
    assert cf.market_beta(snap, "NOPE", 120) == (None, 0)


# ---------------------------------------------------------------------------------------------------- classifier ladder (unit)


def _audit(kn_att=None, uses=()):
    att = kn_att or cf.attribute_move(0.1, 1.0, 100, 0.0, 0.0)
    return cf.AuditResult(att.realized, tuple(uses), (), att, (), "2021-04-01", "", "")


def _scores(**kw):
    base = {"combined": 0.0, "direction": 0.0, "opposing": 0.0, "agreeing": 0.0, "n_supporting_domains": 0.0, "strongest": 0.0, "magnitude": 0.0}
    base.update(kw)
    return base


def test_ladder_thresholds():
    st = report("precursor").knowledge_state_at_decision
    au = _audit()
    k = lambda **kw: cf.classify_knowability(st, _scores(strongest=0.6, **kw), au, CFG, ())[0]
    assert k(combined=0.8, direction=0.4, n_supporting_domains=3) == Knowability.PREDICTABLE
    assert k(combined=0.8, direction=0.0, n_supporting_domains=3) == Knowability.POTENTIALLY_PREDICTABLE      # no direction evidence
    assert k(combined=0.8, direction=0.4, n_supporting_domains=1) == Knowability.POTENTIALLY_PREDICTABLE      # one domain only
    assert k(combined=0.5) == Knowability.POTENTIALLY_PREDICTABLE
    assert k(combined=0.25) == Knowability.WEAKLY_PREDICTABLE
    assert k(combined=0.05) == Knowability.UNKNOWN


def test_conflicting_evidence_is_unknown():
    st = report("precursor").knowledge_state_at_decision
    kn, notes = cf.classify_knowability(st, _scores(combined=0.1, opposing=0.6, strongest=0.05), _audit(), CFG, ())
    assert kn == Knowability.UNKNOWN and any("other way" in n for n in notes)


def test_unprovable_timestamps_force_unknown():
    st = report("precursor").knowledge_state_at_decision
    unc = [cf.InfoItem(cf.Domain.EVENT, "x", 1.0, "", "", "s", Availability.UNCERTAIN)] * 3
    kn, notes = cf.classify_knowability(st, _scores(), _audit(), CFG, unc)
    assert kn == Knowability.UNKNOWN and "timestamps" in notes[0]


def test_sub_threshold_pointers_are_noise():
    st = report("precursor").knowledge_state_at_decision
    kn, notes = cf.classify_knowability(st, _scores(combined=0.5, strongest=0.15, direction=0.3), _audit(), CFG, ())
    assert kn == Knowability.UNKNOWN and any("sub-threshold" in n for n in notes)


def test_boundary_cause_is_flagged_and_capped():
    r = report("news")
    item = cf.InfoItem(cf.Domain.EVENT, "EARN", 1.0, r.event.event_start, r.event.event_start, "events", Availability.SIMULTANEOUS, 1)
    au = _audit(uses=[cf.FutureUse(item, "cause", 1, 0.9)])
    kn, notes = cf.classify_knowability(r.knowledge_state_at_decision, _scores(), au, CFG, ())
    assert kn == Knowability.INFORMATIONALLY_UNAVAILABLE and any("boundary" in n for n in notes)


def test_evidence_directions():
    st = report("precursor").knowledge_state_at_decision
    st = dataclasses.replace(st, items=tuple(dataclasses.replace(i, value=-0.01) if i.key == "TECHNICAL.dist_52wh" else i for i in st.items))
    up = cf.EventSpec.make("T00", st.decision_ts, "2021-03-08", 1)
    down = cf.EventSpec.make("T00", st.decision_ts, "2021-03-08", -1)
    eu = {e.name for e in cf.collect_evidence(st, up) if e.agrees}
    ed = {e.name for e in cf.collect_evidence(st, down) if e.agrees}
    assert "near_52w_high" in eu and "near_52w_high" not in ed
    scored = cf.collect_evidence(st, dataclasses.replace(up, model_score=-0.9, score_asof=st.decision_ts))
    assert any(e.name == "model_score" and e.agrees is False for e in scored)
    assert cf.collect_evidence(dataclasses.replace(st, items=()), up) == []


def test_score_evidence_uses_best_per_domain_and_subtracts_opposition():
    E = cf.Evidence
    D = cf.Domain
    ch = cf.Channel
    same_domain = [E(D.TECHNICAL, "a", 0.5, ch.MAGNITUDE, None), E(D.TECHNICAL, "b", 0.5, ch.MAGNITUDE, None)]
    two_domains = [E(D.TECHNICAL, "a", 0.5, ch.MAGNITUDE, None), E(D.VOLUME, "b", 0.5, ch.MAGNITUDE, None)]
    assert cf.score_evidence(same_domain)["magnitude"] == pytest.approx(0.5)
    assert cf.score_evidence(two_domains)["magnitude"] == pytest.approx(0.75)
    agree = [E(D.PATTERN, "p", 0.6, ch.DIRECTION, True)]
    fight = agree + [E(D.MEMORY, "m", 0.6, ch.DIRECTION, False)]
    assert cf.score_evidence(fight)["direction"] < cf.score_evidence(agree)["direction"]
    assert cf.score_evidence([])["combined"] == 0.0


# ---------------------------------------------------------------------------------------------------- stability, missed info, what became known


def test_label_stability_reports_fragility():
    firm = cf.label_stability(report("precursor"))
    assert firm.label == Knowability.PREDICTABLE and 0.0 <= firm.stable_share <= 1.0
    r = report("precursor")
    borderline = dataclasses.replace(r, scores={**r.scores}, evidence=tuple(dataclasses.replace(e, strength=e.strength * 0.5)
                                                                              for e in r.evidence))
    fb = cf.label_stability(borderline)
    assert fb.label != Knowability.PREDICTABLE
    ev_domains = {e.domain for e in r.evidence}
    assert firm.decisive_domains or len(ev_domains) > 1
    assert len(cf.perturbed_configs(CFG)) >= 6 and all(c.check() == [] for c in cf.perturbed_configs(CFG))


def test_reclassify_without_a_domain_can_change_the_label():
    r = report("precursor")
    assert cf.reclassify(r, CFG) == Knowability.PREDICTABLE
    assert cf.reclassify(r, CFG, cf.Domain.VOLUME) == Knowability.DATA_FAILURE        # core domain removed: reconstruction broken
    assert cf.reclassify(r, CFG, cf.Domain.TECHNICAL) != Knowability.PREDICTABLE       # the squeeze carried the label


def test_missed_information_lists_evidence_the_model_did_not_use():
    reps = [report("precursor"), report("unknown"), report("news")]
    mi = cf.missed_information(reps, min_strength=0.3)
    assert "range_squeeze" in set(mi["evidence"]) and (mi["weighted_share"] > 0).all()
    assert cf.missed_information([report("unknown")]).empty
    assert cf.missed_information([]).empty


def test_domain_contribution_and_coverage_tables():
    reps = [report("precursor"), report("news"), report("unknown")]
    dc = cf.domain_contribution(reps)
    assert dc.loc["TECHNICAL", "PREDICTABLE"] > 0 and list(dc.index) == [d.value for d in cf.ALL_DOMAINS]
    cov = cf.coverage_report(reps)
    assert (cov["present_share"] <= 1.0).all() and cov.set_index("domain").loc["PRICE", "present_share"] == 1.0
    assert cf.coverage_report([]).empty and cf.domain_contribution([]).empty
    ub = cf.unavailable_breakdown(reps)
    assert {"EVENT", "PATTERN", "MEMORY"} <= set(ub["domain"]) and cf.unavailable_breakdown([]).empty


def test_what_became_known_finds_the_later_filing():
    c = case("news")
    diff = cf.what_became_known(c.store, c.event, horizons=(1, 5), providers=c.providers)
    assert {"horizon", "key", "domain", "change"} <= set(diff.columns)
    assert ((diff["key"] == "EVENT.EARN") & (diff["change"] == "new") & (diff["horizon"] == 5)).any()
    assert not ((diff["key"] == "EVENT.EARN") & (diff["horizon"] == 1)).any()


def test_partial_knowledge_pairs_a_scheduled_catalyst_with_its_content():
    r = report("news")
    sched = cf.InfoItem.make(cf.Domain.EVENT, "scheduled:EARN", 1.0, "2021-01-05", "2021-01-05", "schedule", r.event.decision_ts,
                             r.event.event_start)
    ks = dataclasses.replace(r.knowledge_state_at_decision, items=r.knowledge_state_at_decision.items + (sched,))
    pk = cf.partial_knowledge(dataclasses.replace(r, knowledge_state_at_decision=ks))
    assert len(pk) == 1 and pk[0]["content_class"] == "UNAVAILABLE"
    assert cf.partial_knowledge(report("unknown")) == []


# ---------------------------------------------------------------------------------------------------- selection, batches, checkpoints


def _specs(n, day="2021-03-01", cal=None):
    out = []
    for i in range(n):
        out.append(cf.EventSpec.make(f"T{i:02d}", day, "2021-03-05", 1, cal, "mover", realized_return=0.30 - 0.001 * i))
    return out


def test_select_events_caps_and_weights():
    cfg = cf.CounterfactualConfig(top_k_per_day=10, sample_rate=0.25)
    evs = _specs(200)
    chosen = cf.select_events(evs, cfg, seed=3)
    top = [c for c in chosen if c[1] == "top_k"]
    smp = [c for c in chosen if c[1] == "sampled"]
    assert len(top) == 10 and all(w == 1.0 for _, _, w in top)
    assert {e.event_id for e, _, _ in top} == {e.event_id for e in evs[:10]}            # the largest moves, in full
    assert all(w == pytest.approx(4.0) for _, _, w in smp) and 25 <= len(smp) <= 75
    shuffled = cf.select_events(list(reversed(evs)), cfg, seed=3)
    assert {e.event_id for e, _, _ in shuffled} == {e.event_id for e, _, _ in chosen}       # order-independent
    other = cf.select_events(evs, cfg, seed=4)
    assert {e.event_id for e, _, _ in other} != {e.event_id for e, _, _ in chosen}
    assert cf.select_events([], cfg) == []
    assert all(s != "sampled" for _, s, _ in cf.select_events(evs, dataclasses.replace(cfg, sample_rate=0.0)))


def test_sampling_audit_weights_rebuild_the_population():
    cfg = cf.CounterfactualConfig(top_k_per_day=20, sample_rate=0.2)
    evs = []
    for d in ("2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04"):
        evs += _specs(300, d)
    chosen = cf.select_events(evs, cfg, seed=0)
    rows = pd.DataFrame({"weight": [w for _, _, w in chosen]})
    a = cf.sampling_audit(rows, len(evs))
    assert not a["biased"] and 0.8 < a["ratio"] < 1.2 and a["ess"] < len(rows)
    assert cf.sampling_audit(pd.DataFrame({"weight": [1.0] * 10}), 1000)["biased"]
    assert cf.sampling_audit(pd.DataFrame(columns=["weight"]), 0)["n"] == 0


def test_run_batch_checkpoint_resume_and_manifest(tmp_path):
    c = case("precursor")
    r1 = cf.run_batch(c.store, [c.event], c.now, c.providers, out_dir=tmp_path)
    assert len(r1.rows) == 1 and r1.n_snapshots == 1 and r1.resumed == 0 and r1.n_selected == 1
    year_file = tmp_path / f"cf_{c.event.real_year}.jsonl"
    assert year_file.exists() and (tmp_path / "cf_manifest.json").exists()
    r2 = cf.run_batch(c.store, [c.event], c.now, c.providers, out_dir=tmp_path)
    assert r2.resumed == 1 and r2.rows.empty and r2.n_snapshots == 0
    assert len(year_file.read_text().splitlines()) == 1                               # nothing written twice
    loaded = cf.load_rows(tmp_path)
    assert len(loaded) == 1 and loaded.iloc[0]["knowability"] == r1.rows.iloc[0]["knowability"]
    with open(year_file, "a", encoding="utf-8") as f:
        f.write('{"torn": tru')                                                          # a kill mid-write
    assert cf.load_rows(tmp_path).attrs["torn"] == 1 and len(cf.load_rows(tmp_path)) == 1
    with pytest.raises(ValueError):
        cf.run_batch(c.store, [c.event], c.now, c.providers, cfg=cf.CounterfactualConfig(strong=0.7), out_dir=tmp_path)
    with pytest.raises(ValueError):
        cf.run_batch(c.store, [c.event], c.now, c.providers, out_dir=tmp_path, seed=99)


def test_run_batch_skips_unmatured_and_records_bad_events():
    c = case("precursor")
    res = cf.run_batch(c.store, [c.event], pd.Timestamp(c.event.event_end), c.providers)
    assert res.rows.empty and res.skipped and "not matured" in res.skipped[0]["why"]
    ghost = cf.EventSpec.make("NOPE", c.event.decision_ts, c.event.event_end, 1, c.store.cal)
    res2 = cf.run_batch(c.store, [ghost, c.event], c.now, c.providers)
    assert len(res2.rows) + len(res2.errors) >= 1 and res2.n_snapshots == 1              # one snapshot serves both events of the day


def test_run_batch_empty_and_shared_snapshot():
    c = case("precursor")
    empty = cf.run_batch(c.store, [], c.now)
    assert empty.rows.empty and empty.n_snapshots == 0 and empty.summary()["n"] == 0
    assert cf.step(c.store, [], c.now) == []
    two = cf.run_batch(c.store, [c.event, dataclasses.replace(c.event, ticker="T01", event_id="CFother", peers=())], c.now, c.providers)
    assert two.n_snapshots == 1


def test_run_year_streams_and_resumes(tmp_path):
    c = case("precursor")
    d = pd.Timestamp(c.event.decision_ts)
    res = cf.run_year(c.store, d.year, c.now, tmp_path, c.providers, stride=1, horizon_sessions=4, min_abs=0.05, tail=0.05,
                      max_days=1)
    assert isinstance(res, cf.BatchResult)
    with pytest.raises(ValueError):
        cf.run_year(c.store, d.year, c.now, tmp_path / "x", cf.Providers(patterns=[object()]))


def test_discover_events_finds_the_planted_mover_after_maturity():
    c = case("precursor")
    found = cf.discover_events(c.store, c.event.decision_ts, c.now, horizon_sessions=4, tail=0.05, min_abs=0.05)
    assert any(e.ticker == "T00" and e.direction == 1 and e.realized_return > 0.05 for e in found)
    assert all(e.validate() == [] for e in found)
    assert cf.discover_events(c.store, c.event.decision_ts, pd.Timestamp(c.event.event_end), horizon_sessions=4) == []
    assert cf.discover_events(c.store, c.event.decision_ts, c.now, horizon_sessions=4, min_abs=5.0) == []


def test_events_from_frame_validates():
    df = pd.DataFrame({"ticker": ["A", "B"], "decision_ts": ["2021-03-01"] * 2, "event_end": ["2021-03-05"] * 2, "direction": [1, -1]})
    assert len(cf.events_from_frame(df)) == 2
    bad = df.assign(event_end="2021-02-01")
    with pytest.raises(ValueError):
        cf.events_from_frame(bad)
    assert cf.events_from_frame(df.iloc[:0]) == []


# ---------------------------------------------------------------------------------------------------- summaries and honesty


def _rows(labels, conf=0.7, n_unc=0):
    return pd.DataFrame({"report_id": range(len(labels)), "event_id": range(len(labels)), "year": 2021, "category": "mover", "direction": 1,
                         "knowability": labels, "confidence": conf, "n_available": 1, "n_unavailable": 1, "n_future_used": 1,
                         "n_uncertain": n_unc, "coverage_domains": 9, "market_share": 0.1, "idio_share": 0.5, "combined": 0.3,
                         "selection": "top_k", "weight": 1.0, "matured_at": "2021-04-01"})


def test_honesty_flags_catch_forced_labelling():
    forced = _rows(["WEAKLY_PREDICTABLE"] * 40, conf=0.15)
    flags = cf.honesty_flags(forced)
    assert any("no UNKNOWN" in f for f in flags) and any("forced labelling" in f for f in flags)
    leaky = _rows(["PREDICTABLE"] * 40 + ["UNKNOWN"] * 5)
    assert any("PREDICTABLE" in f for f in cf.honesty_flags(leaky))
    broken = _rows(["DATA_FAILURE"] * 20 + ["UNKNOWN"] * 20)
    assert any("DATA_FAILURE" in f for f in cf.honesty_flags(broken))
    healthy = _rows(["UNKNOWN"] * 15 + ["EXTERNALLY_CAUSED"] * 15 + ["WEAKLY_PREDICTABLE"] * 10 + ["INFORMATIONALLY_UNAVAILABLE"] * 10)
    assert cf.honesty_flags(healthy) == []
    assert cf.honesty_flags(_rows(["UNKNOWN"] * 3)) == ["too few reports for distribution checks"]
    assert any("timestamps" in f for f in cf.honesty_flags(_rows(["UNKNOWN"] * 40, n_unc=2)))


def test_summarize_weights_and_empty():
    rows = _rows(["UNKNOWN", "UNKNOWN", "PREDICTABLE"])
    rows["weight"] = [1.0, 1.0, 4.0]
    s = cf.summarize(rows)
    assert s["by_class"]["PREDICTABLE"] == pytest.approx(4 / 6, abs=1e-3) and s["n"] == 3 and s["weighted_n"] == 6.0
    e = cf.summarize(_rows([]))
    assert e["n"] == 0 and e["unknown_rate"] is None
    md_empty = cf.report_markdown(_rows([]))
    assert "No reports" in md_empty
    md = cf.report_markdown(_rows(["UNKNOWN"] * 5 + ["EXTERNALLY_CAUSED"] * 5))
    assert "Distribution of knowability" in md and "Honesty flags" in md and "NOT VALIDATED" in md


def test_share_table_wilson_and_ess():
    lo, hi = cf.wilson_interval(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61 and cf.wilson_interval(0, 0) == (0.0, 1.0)
    assert cf.wilson_interval(0, 50)[0] == 0.0 and cf.wilson_interval(50, 50)[1] == 1.0
    assert cf.effective_sample_size([1, 1, 1, 1]) == pytest.approx(4.0)
    assert cf.effective_sample_size([10, 1, 1, 1]) < 4.0 and cf.effective_sample_size([]) == 0.0
    t = cf.share_table(_rows(["UNKNOWN"] * 30 + ["PREDICTABLE"] * 10))
    u = t[t["knowability"] == "UNKNOWN"].iloc[0]
    assert u["share"] == pytest.approx(0.75) and u["lo"] < 0.75 < u["hi"]
    assert cf.share_table(_rows([])).empty


def test_explain_mentions_the_reasons():
    txt = cf.explain(report("precursor"))
    assert "T00" in txt and "PREDICTABLE" in txt and "pointer" in txt
    assert "Not knowable then" in cf.explain(report("news"))
    assert "nothing in the available information" in cf.explain(dataclasses.replace(report("unknown"), evidence=()))
    assert "could not be measured" in cf.explain(dataclasses.replace(report("unknown"),
                                                                    attribution=cf.attribute_move(None, None, 0, None, None)))


# ---------------------------------------------------------------------------------------------------- providers and the store builder


def test_external_classifier_is_used_but_never_hides_data_failure():
    c = case("unknown")
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    seen = {}

    def clf(bundle):
        seen["keys"] = sorted(bundle)
        return "WEAKLY_PREDICTABLE", "r02 thinks so"

    p = dataclasses.replace(c.providers, classifier=cf.adapt_classifier(clf))
    r = cf.assess_event(c.store, snap, c.event, c.now, p)
    assert r.knowability == Knowability.WEAKLY_PREDICTABLE and any("in-module ladder said UNKNOWN" in n for n in r.notes)
    assert {"state", "scores", "evidence", "attribution", "baseline"} <= set(seen["keys"])
    dc = case("data_failure")
    snap2 = cf.build_snapshot(dc.store, dc.event.decision_ts)
    r2 = cf.assess_event(dc.store, snap2, dc.event, dc.now, dataclasses.replace(dc.providers, classifier=cf.adapt_classifier(clf)))
    assert r2.knowability == Knowability.DATA_FAILURE
    broken = dataclasses.replace(c.providers, classifier=lambda b: (_ for _ in ()).throw(ValueError("boom")))
    r3 = cf.assess_event(c.store, snap, c.event, c.now, broken)
    assert r3.knowability == Knowability.UNKNOWN and any("unusable" in n for n in r3.notes)
    with pytest.raises(TypeError):
        cf.adapt_classifier(object())


def test_adapt_classifier_accepts_objects_and_enums():
    class R02:
        def classify(self, bundle):
            return type("V", (), {"knowability": Knowability.EXTERNALLY_CAUSED})()
    kn, why = cf.adapt_classifier(R02())({})
    assert kn == Knowability.EXTERNALLY_CAUSED and why == ""


def test_validate_providers_catches_bad_parts():
    c = case("unknown")
    assert cf.validate_providers(c.providers) == []
    assert cf.validate_providers(cf.Providers(patterns=[object()]))
    bad_prov = Provenance("", "", "")
    p = cf.PlantedPattern("x", "x", 0.1, 0.5, bad_prov, lambda f: True)
    assert any("provenance" in e for e in cf.validate_providers(cf.Providers(patterns=[p])))
    assert cf.validate_providers(cf.Providers(memory=object()))
    inverted = cf.PlantedMemory("m", "k", {}, 0.1, "2021-05-01", "2021-04-01")
    assert any("matured before" in e for e in cf.validate_providers(cf.Providers(memory=cf.PlantedMemoryStore([inverted]))))
    with pytest.raises(ValueError):
        cf.reconstruct_state(cf.build_snapshot(c.store, c.event.decision_ts), c.event, cf.Providers(patterns=[object()]))


def test_pattern_learned_after_the_decision_is_not_in_the_state_even_if_it_fires():
    c = case("precursor")
    late = Provenance("2030-01-01T00:00:00", "2021-06-01", "x", outcomes_seen_through="2021-06-01")
    p = cf.PlantedPattern("late", "late_fire", 0.9, 0.99, late, lambda f: True)
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    st, side = cf.reconstruct_state(snap, c.event, cf.Providers(patterns=[p]))
    assert not any(i.name == "late_fire" for i in st.items) and side["later_patterns"]


def test_store_from_feed_data_sets_filing_availability():
    idx = pd.bdate_range("2021-01-04", periods=30)
    close = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    stocks = {f: close.copy() for f in ("Close", "Open", "High", "Low", "Volume")}
    market = {f: pd.DataFrame(1.0, index=idx, columns=["SPY"]) for f in ("Close", "Open")}
    ev = pd.DataFrame({"ticker": ["A", "A"], "kind": ["EARN", "OFFERING"],
                       "accepted": pd.to_datetime(["2021-01-12 14:00", "2021-01-12 22:00"], utc=True)})     # 09:00 and 17:00 New York
    store = cf.store_from_feed_data((stocks, market, ev, None, None))
    rec = store.source("events").df.set_index("kind")
    assert rec.loc["EARN", "available"] == pd.Timestamp("2021-01-12") and rec.loc["OFFERING", "available"] == pd.Timestamp("2021-01-13")
    assert store.validate().ok
    empty = cf.store_from_feed_data((stocks, market, ev.iloc[:0], None, None))
    assert len(empty.source("events").df) == 0
    with pytest.raises(ValueError):
        cf.store_from_feed_data(({"Open": close}, market, ev))


def test_technical_state_needs_history_and_leaves_gaps_unknown():
    c = case("precursor")
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    tech, last = cf.technical_state(snap, "T00", CFG)
    assert last == c.event.decision_ts and "squeeze_rank" in tech and tech["squeeze_rank"] > 0.7 and "rsi14" in tech
    assert cf.technical_state(snap, "NOPE", CFG) == ({}, "")
    short = cf.build_snapshot(c.store, pd.Timestamp(c.store.source("prices").frames["Close"].index[20]), cfg=cf.CounterfactualConfig(min_history=10))
    t2, _ = cf.technical_state(short, "T00", cf.CounterfactualConfig(min_history=30))
    assert "vol20" not in t2 and "r5" in t2 and "dist_ma200" not in t2


def test_snapshot_shares_one_day_and_logs_every_read():
    c = case("precursor")
    snap = cf.build_snapshot(c.store, c.event.decision_ts)
    assert snap.log.verify() and len(snap.log) >= 5 and cf.log_problems(snap.log, c.event.decision_ts) == []
    assert snap.event_start == pd.Timestamp(c.event.event_start) and "m_vix" in snap.market_row and len(snap.xs) == 40
    assert snap.market_row["m_breadth"] <= 1.0


def test_manifest_detects_a_changed_config(tmp_path):
    m1 = cf.check_manifest(tmp_path, CFG, cf.SourceMap(), 0)
    assert cf.check_manifest(tmp_path, CFG, cf.SourceMap(), 0) == m1
    with pytest.raises(ValueError):
        cf.check_manifest(tmp_path, dataclasses.replace(CFG, weak=0.2), cf.SourceMap(), 0)
    assert cf.check_manifest(tmp_path, dataclasses.replace(CFG, weak=0.2), cf.SourceMap(), 0, allow_change=True) == m1


def test_entrypoints_are_exposed():
    assert set(cf.ENTRYPOINTS) >= {"step", "run_batch", "run_year", "assess_event", "release", "selfcheck"}
    assert "ClassificationGate" in cf.__all__ and "step" in cf.__all__


def test_unknown_planted_kind_is_refused():
    with pytest.raises(ValueError):
        cf.make_planted_case("nonsense")
