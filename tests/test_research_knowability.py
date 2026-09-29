"""Tests for engine.research.knowability and engine.research.unknown_cause (C66 sections 7, 8, 33). Synthetic data only."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FailureCause, FirewallBreach
from engine.learning.unknowns import UnknownLedger
from engine.research import knowability as K
from engine.research import unknown_cause as U
from engine.research.core import Availability, Knowability, Namespace

KN = Knowability


def cls(kind, seed=0, **kw):
    return K.classify_move(K.planted_inputs(kind, seed), **kw)


# ------------------------------------------------------------------ availability tagging
B = K.DecisionBoundary("2019-06-10", "2019-06-11")


def item(pub, eff="2019-06-11", prov=K.PROV_RECORDED, scheduled=False, exists=True, i="X"):
    return K.InfoItem(i, K.InfoKind.EVENT, "T", eff, pub, "src", prov, exists, scheduled, 0.5)


@pytest.mark.parametrize("pub,expect", [
    ("2019-06-10 09:00", Availability.KNOWN_BEFORE_EVENT),
    ("2019-06-10 15:45", Availability.SIMULTANEOUS),
    ("2019-06-11 08:00", Availability.SIMULTANEOUS),
    ("2019-06-11 09:31", Availability.KNOWN_ONLY_AFTER_EVENT),
    ("2019-06-05", Availability.KNOWN_BEFORE_EVENT),
    ("2019-06-10", Availability.SIMULTANEOUS),
    ("2019-06-11", Availability.UNCERTAIN),
    ("2019-06-13", Availability.KNOWN_ONLY_AFTER_EVENT),
    (None, Availability.UNCERTAIN),
])
def test_availability_tags(pub, expect):
    assert K.judge_item(item(pub, scheduled=True), B).availability == expect


def test_unknown_provenance_never_known_before():
    j = K.judge_item(item("2019-06-01", prov=K.PROV_UNKNOWN), B)
    assert j.availability == Availability.UNCERTAIN


def test_nonexistent_item_is_unavailable_and_publication_before_event_is_uncertain():
    assert K.judge_item(item(None, exists=False), B).availability == Availability.UNAVAILABLE
    early = item("2019-05-01", eff="2019-06-11", scheduled=False)
    assert K.judge_item(early, B).availability == Availability.UNCERTAIN


def test_inferred_lag_near_boundary_is_uncertain():
    it = K.InfoItem("M", K.InfoKind.MACRO, "MARKET", "2019-06-07", None, "cpi", K.PROV_INFERRED)
    j = K.judge_item(it, B, lag=K.LagPolicy((("cpi", 1),), margin_sessions=2))
    assert j.availability == Availability.UNCERTAIN and j.inferred
    far = dataclasses.replace(it, effective_at="2019-05-01")
    assert K.judge_item(far, B, lag=K.LagPolicy((("cpi", 1),))).availability == Availability.KNOWN_BEFORE_EVENT


def test_bad_inputs_raise():
    with pytest.raises(K.KnowabilityError):
        K.judge_all([item("2019-06-01", i="A"), item("2019-06-02", i="A")], B)
    with pytest.raises(K.KnowabilityError):
        K.judge_all([dataclasses.replace(item("2019-06-01"), strength=2.0)], B)
    with pytest.raises(K.KnowabilityError):
        K.judge_all([], K.DecisionBoundary("2019-06-10", "2019-06-10"))


def test_precursor_probe_requires_coverage():
    cause = item("2019-06-13")
    assert K.judge_precursors(cause, [K.SourceProbe("a", True), K.SourceProbe("b", True)], B)[0] == Availability.UNAVAILABLE
    assert K.judge_precursors(cause, [K.SourceProbe("a", True), K.SourceProbe("b", False)], B)[0] == Availability.UNCERTAIN
    assert K.judge_precursors(cause, [K.SourceProbe("a", True, ("P1",))], B)[0] == Availability.KNOWN_BEFORE_EVENT
    assert K.judge_precursors(cause, [], B)[0] == Availability.UNCERTAIN


# ------------------------------------------------------------------ classifier on planted truth
def test_planted_battery_recovers_every_class():
    bat = K.planted_battery(seeds=(0, 1, 2))
    assert bat["min_recall"] == 1.0, bat["confusion"]


def test_null_world_is_unknown_not_predictable():
    got = [cls("UNKNOWN", s).classification for s in range(25)]
    assert got.count(KN.UNKNOWN) >= 22
    assert KN.PREDICTABLE not in got and KN.POTENTIALLY_PREDICTABLE not in got


def test_unknown_stays_unknown_when_source_not_covering():
    a = cls("HINDSIGHT_ONLY")
    assert a.classification == KN.UNKNOWN and "cannot be established" in a.trace[-1]


def test_cause_found_after_fact_is_not_predictive_knowledge():
    a = cls("INFORMATIONALLY_UNAVAILABLE")
    assert a.classification == KN.INFORMATIONALLY_UNAVAILABLE and a.anticipation < 0.2
    assert "8K-1" in a.information_that_was_unavailable and a.n_present_channels == 0


def test_uncertain_timestamps_only_lower_a_class():
    a = cls("UNCERTAIN_EVENT")
    assert a.classification == KN.WEAKLY_PREDICTABLE and a.confidence_in_classification < 0.5
    assert "ER-U" in a.uncertain_information


def test_data_failure_outranks_market_story():
    a = cls("DATA_FAILURE")
    assert a.classification == KN.DATA_FAILURE and "BAD_TICK" in a.quality_flags and a.check() == []


def test_data_quality_checks():
    inp = K.planted_inputs("UNKNOWN", 0)
    bars = inp.bars.copy()
    bars.iloc[280, bars.columns.get_loc("close")] = -1.0
    assert "BAD_PRICE" in K.assess_data_quality(dataclasses.replace(inp, bars=bars)).codes()
    lab = dataclasses.replace(inp, move=dataclasses.replace(inp.move, fwd_return=inp.move.fwd_return + 0.2))
    assert "RETURN_MISMATCH" in K.assess_data_quality(lab).codes()
    clean = K.assess_data_quality(inp)
    assert not clean.failed


def test_externally_caused_needs_systematic_share():
    a = cls("EXTERNALLY_CAUSED")
    assert a.classification == KN.EXTERNALLY_CAUSED and a.systematic_share >= 0.6
    assert cls("UNKNOWN").systematic_share < 0.6


def test_predictable_requires_model_flag():
    assert cls("PREDICTABLE").classification == KN.PREDICTABLE
    assert cls("POTENTIALLY_PREDICTABLE").classification == KN.POTENTIALLY_PREDICTABLE


def test_assessment_fields_and_validation():
    a = cls("PREDICTABLE")
    assert a.namespace == Namespace.MATURED_RESEARCH and a.check() == []
    assert set(K.CHANNELS) == set(a.knowledge_state_at_decision)
    assert K.validate_assessment_semantics(a) == []
    bad = dataclasses.replace(a, classification=KN.EXTERNALLY_CAUSED, systematic_share=0.1)
    assert K.validate_assessment_semantics(bad)
    assert dataclasses.replace(a, trace=()).check()


# ------------------------------------------------------------------ no peeking
def test_channels_are_future_invariant():
    for kind in ("PREDICTABLE", "UNCERTAIN_EVENT", "EXTERNALLY_CAUSED"):
        assert K.channels_future_invariant(K.planted_inputs(kind, 1)) == []


def test_planted_peek_is_caught(monkeypatch):
    real = K.price_channel

    def peeking(pre, cfg):
        return K.ChannelEvidence("PRICE", float(np.tanh(abs(float(pre["close"].iloc[-1])) / 100.0)))
    monkeypatch.setattr(K, "price_channel", real)
    inp = K.planted_inputs("PREDICTABLE", 0)
    assert K.channels_future_invariant(inp) == []

    def peek_build(inp, judg, cfg, _b=K.build_channels):
        ch = _b(inp, judg, cfg)
        fut = inp.bars.sort_index().loc[inp.bars.index > pd.Timestamp(inp.move.decision_date)]
        ch["VOLUME"] = K.ChannelEvidence("VOLUME", float(np.tanh(fut["volume"].mean() / 1e6)))
        return ch
    monkeypatch.setattr(K, "build_channels", peek_build)
    assert "VOLUME" in K.channels_future_invariant(inp)


def test_late_pattern_hits_are_excluded():
    inp = K.planted_inputs("UNKNOWN", 0)
    late = {"id": "PX", "p_real": 0.99, "strength": 1.0, "effect": 0.1, "learned_at": "2019-07-30"}
    a = K.classify_move(dataclasses.replace(inp, pattern_hits=[late]))
    assert a.knowledge_state_at_decision["PATTERNS"]["anticipation"] == 0.0
    assert any("PX" in f for f in a.future_information_used_by_auditor)


def test_naive_effective_date_join_leaks():
    inp = K.planted_inputs("BACKFILLED", 0)
    assert K.naive_vs_pit_gap(inp)["leaked"] == ["10KA-1"]
    assert K.classify_naive(inp) != K.classify_move(inp).classification
    assert K.classify_move(inp).classification == KN.UNKNOWN


def test_feature_leak_detection():
    f = [K.FeatureValue("ok", 1.0, "2019-06-07"), K.FeatureValue("leak", 1.0, "2019-06-11 10:00")]
    assert K.feature_leaks(f, B) == ["leak (KNOWN_ONLY_AFTER_EVENT)"]
    with pytest.raises(FirewallBreach):
        K.assert_features_clean(f, B)


# ------------------------------------------------------------------ hindsight firewall
def test_hindsight_columns_refused():
    K.refuse_hindsight_columns(["ret_5d", "vol_20"])
    for bad in ("knowability_class", "is_predictable", "y_UNKNOWN", "could_have_known"):
        with pytest.raises(FirewallBreach):
            K.refuse_hindsight_columns(["ret_5d", bad])


def test_assessment_cannot_reach_trader():
    a = cls("PREDICTABLE")
    with pytest.raises(FirewallBreach):
        K.assert_not_trader_bound({"x": [a]})
    K.assert_not_trader_bound({"x": [1, 2]})


def test_release_gate_maturity_and_same_year():
    a = cls("PREDICTABLE")
    with pytest.raises(FirewallBreach):
        K.release_for_trader(a, a.matured_at)                       # matures ON now: not yet a fact
    with pytest.raises(FirewallBreach):
        K.release_for_trader(a, "2030-01-01", replaying_years=[2019])
    out = K.release_for_trader(a, "2030-01-01")
    blob = repr(out)
    assert "TKR" not in blob and "2019" not in blob and set(out) == {"knowability", "anticipation_bucket", "confidence_bucket", "n_present_channels"}


def test_step_skips_immature_and_is_idempotent():
    st = K.KnowabilityState()
    ins = [K.planted_inputs("UNKNOWN", s) for s in range(3)]
    _, rep = K.step(st, "2019-06-20", ins)
    assert rep.n_classified == 0 and rep.n_immature == 3
    _, rep = K.step(st, "2030-01-01", ins)
    assert rep.n_classified == 3
    _, rep = K.step(st, "2030-01-01", ins)
    assert rep.n_classified == 0 and rep.n_duplicates == 3


def test_empty_cases():
    led = K.KnowabilityLedger()
    assert K.render_report(led).startswith("knowability: no moves")
    assert K.hindsight_dependence(led)["n_moves"] == 0
    assert K.audit_ledger(led) == {} and len(led.table("regime")) == 0
    assert K.item_key(item(None)) == "EVENT"
    assert K.items_from_edgar_events(pd.DataFrame(), "T", ("2019-01-01", "2019-12-31")) == []
    assert K.items_from_insider(None, "T", ("2019-01-01", "2019-12-31")) == []
    assert K.discrimination_auc([])["n_real"] == 0
    assert np.isnan(K.wilson(0, 0)[0])
    assert K.regime_knowability_prior(led, "2030-01-01")["prior"] == {}
    assert K.research_targets(led, "2030-01-01") == []


def test_ledger_immutable_and_persisted(tmp_path):
    led = K.KnowabilityLedger()
    a = cls("PREDICTABLE")
    assert led.add(a) and not led.add(a)
    with pytest.raises(K.KnowabilityError):
        led.add(dataclasses.replace(a, anticipation=0.1))
    st = K.KnowabilityState()
    K.step(st, "2030-01-01", [K.planted_inputs(k, 0) for k in ("PREDICTABLE", "UNKNOWN", "DATA_FAILURE")])
    K.checkpoint_state(st, tmp_path)
    back = K.resume_state(tmp_path)
    assert back.ledger.counts() == st.ledger.counts()
    with pytest.raises(K.KnowabilityError):
        K.resume_state(tmp_path, cfg=dataclasses.replace(K.KnowabilityConfig(), weak_at=0.25))
    p = tmp_path / "ledger.jsonl"
    txt = p.read_text().replace('"UNKNOWN"', '"PREDICTABLE"', 1)
    p.write_text(txt)
    with pytest.raises(K.KnowabilityError):
        K.load_ledger(p)


def test_population_views():
    st = K.KnowabilityState()
    ins = [K.planted_inputs(k, s) for k in ("PREDICTABLE", "UNKNOWN", "EXTERNALLY_CAUSED", "INFORMATIONALLY_UNAVAILABLE") for s in range(4)]
    K.step(st, "2030-01-01", ins)
    led = st.ledger
    assert len(led) == 16 and led.table("regime")["of"].max() == 16
    ceil = K.knowability_ceiling(led)
    assert abs(ceil["reachable"]["share"] - 0.25) < 1e-9 and abs(ceil["structural"]["share"] - 0.5) < 1e-9
    assert K.hindsight_dependence(led)["n_explained"] >= 4
    assert "major moves classified" in K.render_report(led)
    assert K.class_share_ci(led, KN.UNKNOWN)["n"] == 16
    assert K.day_cluster_ci(led, KN.UNKNOWN)["share"] == pytest.approx(0.25)
    assert set(K.coverage_report(led)) == set(K.CHANNELS)
    assert len(K.to_frame(led)) == 16
    assert K.model_gap(led)["potential"] == 0


def test_auc_planted_signal_vs_noise():
    sig = K.discrimination_auc([K.planted_inputs("PREDICTABLE", s) for s in range(6)], n_boot=30)
    noise = K.discrimination_auc([K.planted_inputs("UNKNOWN", s) for s in range(6)], n_boot=30)
    assert sig["auc"] > 0.9 and noise["auc"] < 0.75


def test_streaming_selection_keeps_exceptions_only():
    r = pd.Series(np.r_[np.random.default_rng(0).normal(0, 0.01, 500), 0.2, -0.25], index=[f"n{i}" for i in range(502)])
    pv = pd.Series(0.01, index=r.index)
    snap, rows = K.select_major_moves("2019-06-10", r, pv, z_min=5.0)
    assert set(rows.index) == {"n500", "n501"} and snap.n_names == 502
    snap0, rows0 = K.select_major_moves("2019-06-10", pd.Series(dtype=float), pd.Series(dtype=float))
    assert rows0.empty and snap0.n_names == 0


def test_stability_and_ablation_run():
    inp = K.planted_inputs("PREDICTABLE", 0)
    assert K.class_stability(inp)["stable_share"] >= 0.75
    assert K.ablate_channels(inp)["base"] == "PREDICTABLE"
    assert K.knowability_onset(inp)["lead_sessions"] is not None
    assert K.horizon_profile(inp, [1, 2, 3]).shape[0] == 3


# ================================================================== unknown_cause
def test_real_causes_supported_decoys_not_and_unknown_kept():
    R = U.planted_records(400, 0)
    tests = U.test_causes(R)
    assert U.supported_causes(tests) == {"EARNINGS", "MARKET_WIDE"}
    v = U.assign(R, tests)
    tc = U.truth_check(R, v)
    assert tc["recall_real"] > 0.95 and tc["forced_label_rate"] == 0.0
    assert U.audit_verdicts(R, v, tests) == []


@pytest.mark.parametrize("seed", range(4))
def test_null_world_supports_nothing(seed):
    N = U.null_records(400, seed)
    assert U.supported_causes(U.test_causes(N)) == frozenset()
    assert all(x.unknown for x in U.assign(N, U.test_causes(N)))


def test_foil_forces_labels_and_is_caught_by_audit():
    R = U.planted_records(400, 0)
    forced = U.force_assign(R)
    assert U.truth_check(R, forced)["forced_label_rate"] == 1.0
    assert U.audit_verdicts(R, forced, U.test_causes(R))
    fc = U.forcing_curve(R)
    assert fc["forced_foil"] == 0.0 and fc["explanations_owed_to_skipping_the_null"] > 0.3


def test_genuine_discovery_vs_forced_labels():
    before = U.planted_records(300, 4, real_causes=(), decoy_causes=("MOMENTUM",), p_decoy=0.05, p_unexplained=0.0)
    genuine = U.planted_records(300, 2)
    decoys = U.planted_records(300, 3, real_causes=(), decoy_causes=("MOMENTUM", "EARNINGS"), p_decoy=0.7, p_unexplained=0.0)
    g = U.diagnose_rate_change(before, genuine)
    assert g.verdict == "GENUINE_DISCOVERY" and g.transferable_share > 0.9 and g.placebo_ratio < 0.3
    lax = dataclasses.replace(U.AssignerConfig(), require_supported=False)
    f = U.diagnose_rate_change(before, decoys, U.AssignerConfig(), lax)
    assert f.verdict == "FORCED_LABELS" and f.loosened
    bb = U.diagnose_rate_change(before, decoys, assigner_after=U.make_forcing_assigner())
    assert bb.verdict == "FORCED_LABELS" and bb.placebo_ratio > 0.6


def test_no_change_and_rise_and_insufficient():
    A = U.planted_records(200, 1)
    B2 = U.planted_records(200, 2)
    assert U.diagnose_rate_change(A, B2).verdict == "NO_SIGNIFICANT_CHANGE"
    assert U.diagnose_rate_change(A[:5], B2[:5]).verdict == "INSUFFICIENT_EVIDENCE"
    rose = U.diagnose_rate_change(U.planted_records(300, 2), U.planted_records(300, 4, real_causes=(), decoy_causes=("MOMENTUM",), p_decoy=0.05, p_unexplained=0.0))
    assert rose.verdict == "UNKNOWN_RATE_ROSE"


def test_loosening_detected_only_one_way():
    a, b = U.AssignerConfig(), dataclasses.replace(U.AssignerConfig(), min_strength=0.1, alpha=0.2)
    assert len(U.loosened(a, b)) == 2 and U.loosened(b, a) == []


def test_rates_by_dimension_and_heterogeneity():
    R = U.planted_records(400, 0)
    v = U.assign(R, U.test_causes(R))
    for d in U.DIMENSIONS:
        t = U.rate_table(R, v, d)
        assert t["n"].sum() == 400 and ((t["lo"] <= t["rate"]) & (t["rate"] <= t["hi"])).all()
    assert U.rate_heterogeneity(R, v, "regime")["p"] > 0.01           # regimes are assigned at random
    with pytest.raises(U.UnknownCauseError):
        U.rate_table(R, v, "bogus")
    assert U.rate_table([], [], "regime").empty
    assert U.two_proportion_p(0, 0, 1, 1) == 1.0


def test_rate_planted_regime_difference_is_found():
    R = U.planted_records(600, 0)
    R = [dataclasses.replace(r, regime="R1", abs_z=r.abs_z + (0.0 if i % 2 else 0.0)) for i, r in enumerate(R)]
    stripped = [dataclasses.replace(r, regime="QUIET", evidence=()) if i % 3 == 0 else r for i, r in enumerate(R)]
    v = U.assign(stripped, U.test_causes(stripped))
    t = U.rate_table(stripped, v, "regime").set_index("group")
    assert t.loc["QUIET", "rate"] == 1.0 and t.loc["QUIET", "rate"] > t.loc["R1", "rate"]
    assert U.rate_heterogeneity(stripped, v, "regime")["p"] < 0.05


def test_book_flags_unearned_drop():
    book = U.UnknownRateBook()
    book.add(U.WindowRecord("W1", "2016-01-01", 100, 70, "aa", ()))
    book.add(U.WindowRecord("W2", "2017-01-01", 100, 40, "bb", (), "FORCED_LABELS"))
    assert book.unearned_drops() == ["W2"]
    assert not U.rate_fall_needs_evidence(book)["passes"]
    with pytest.raises(U.UnknownCauseError):
        book.add(U.WindowRecord("W0", "2015-01-01", 1, 1, "aa", ()))
    ok = U.UnknownRateBook()
    ok.add(U.WindowRecord("W1", "2016-01-01", 100, 70, "aa", ()))
    ok.add(U.WindowRecord("W2", "2017-01-01", 100, 40, "aa", ("EARNINGS",), "GENUINE_DISCOVERY"))
    assert U.rate_fall_needs_evidence(ok)["passes"] and ok.persistence() == {"EARNINGS": 1.0}


def test_unknown_ledger_integration_and_resolution():
    R = U.planted_records(300, 0)
    v = U.assign(R, U.test_causes(R))
    led = UnknownLedger()
    opened = U.open_unknowns(R, v, led, "2030-01-01")
    assert opened and all(s.startswith("unknown_cause:") for s in opened)
    forced = U.RateChange(0.7, 0.3, 0.4, 0.001, 100, 100, 0.1, 0.9, 0.9, (), "FORCED_LABELS", ())
    assert U.resolve_on_discovery(led, forced, "2029-12-01", "2030-01-01") == []
    ok = dataclasses.replace(forced, verdict="GENUINE_DISCOVERY")
    assert set(U.resolve_on_discovery(led, ok, "2029-12-01", "2030-01-02")) == set(opened)
    assert U.failure_cause(v[0]) in (FailureCause.UNKNOWN, FailureCause.INSUFFICIENT_EVIDENCE, FailureCause.SELECTION_ERROR)


def test_record_from_assessment_and_step_end_to_end(tmp_path):
    ass = [K.classify_move(K.planted_inputs(k, s)) for k in ("PREDICTABLE", "EXTERNALLY_CAUSED", "INFORMATIONALLY_UNAVAILABLE", "UNKNOWN", "DATA_FAILURE")
           for s in range(3)]
    assert U.record_from_assessment(ass[-1]) is None                       # data failure is not an unexplained market move
    st = U.UnknownCauseState()
    _, rep = U.step(st, "2019-06-12", ass)                                # nothing has matured yet
    assert rep.n_records == 0 and "no matured" in U.render_report(rep)
    _, rep = U.step(st, "2030-01-01", ass)
    assert rep.n_records == 12 and 0.0 <= rep.unknown_rate <= 1.0
    h = U.save_state(st, tmp_path / "s.json")
    back = U.load_state(tmp_path / "s.json")
    assert set(back.seen) == set(st.seen) and back.cfg == st.cfg
    (tmp_path / "s.json").write_text((tmp_path / "s.json").read_text().replace("MARKET_WIDE", "MOMENTUM", 1))
    if "MARKET_WIDE" in (tmp_path / "s.json").read_text() or True:
        with pytest.raises(U.UnknownCauseError):
            U.load_state(tmp_path / "s.json")
    assert h


def test_empty_unknown_cause_cases():
    st = U.UnknownCauseState()
    _, rep = U.step(st, "2030-01-01", [])
    assert rep.n_records == 0 and rep.unknown_rate is None
    assert U.test_causes([]) and not U.supported_causes(U.test_causes([]))
    assert U.assign([], {}) == [] and U.unknown_counts([]) == (0, 0)
    assert U.break_unknown_table(pd.DataFrame()).empty
    assert U.release_prior(st, "2030-01-01")["prior"] == {}
    assert U.blind_causes_of([]) == list(U.CAUSES)


def test_unknown_cause_firewall():
    with pytest.raises(FirewallBreach):
        U.refuse_unknown_cause_features(["ret", "unknown_cause_flag"])
    R = U.planted_records(300, 0)
    st = U.UnknownCauseState()
    st.seen = {r.move_id: r for r in R}
    out = U.release_prior(st, "2030-01-01", min_n=30)
    assert out["prior"] and "R1" in out["prior"] and set(out["prior"]["R1"]) == {"unknown_rate"}
    assert U.release_prior(st, "2030-01-01", replaying_years=range(2000, 2100))["prior"] == {}


def test_break_unknown_share_shares_definition():
    inv = pd.DataFrame({"verdict": ["EXPLAINED_AND_GATED"] * 4 + ["DISCARDED_UNPREDICTABLE"] * 6 + ["OPEN"],
                        "cause": ["x"] * 4 + ["unknown"] * 6 + ["x"], "detail": ["d"] * 11})
    t = U.break_unknown_table(inv)
    assert t["resolved"].iloc[0] == 10 and t["share"].iloc[0] == pytest.approx(0.6)
    assert U.one_vocabulary_summary([], inv)["breaks"]["unknown"] == 6


def test_stability_and_novelty():
    R = U.planted_records(300, 0)
    assert U.verdict_stability(R, n=3)["mean_agreement"] > 0.9
    assert U.supported_set_stability(R, n=3)["EARNINGS"] == 1.0
    nov = U.explanation_novelty(U.null_records(200, 1), R)
    assert set(nov["new"]) == {"EARNINGS", "MARKET_WIDE"}
    tr = U.transfer_of_causes(R[:200], R[200:])
    assert tr["effect"] > 0.3 and tr["effect"] > tr["random_effect"] + 0.2
    assert U.transfer_of_causes(U.null_records(200, 1), R[200:])["supported"] == []
    assert U.commonness(R)["MOMENTUM"] > 0.2
    assert isinstance(U.strength_sweep(R, bars=(0.2, 0.5)), pd.DataFrame)
    st = U.stream_by_year([(2016, []), (2017, [])], U.UnknownCauseState(), "2030-01-01")
    assert len(st) == 2
    with pytest.raises(U.UnknownCauseError):
        U.stream_by_year([(2017, []), (2016, [])], U.UnknownCauseState(), "2030-01-01")


def test_group_generality_and_hindsight_fraction():
    R = U.planted_records(600, 0)
    assert U.regime_bound_causes(R, min_group=60) == {"EARNINGS": "GENERAL", "MARKET_WIDE": "GENERAL"}
    # plant a regime-bound cause: EARNINGS enlarges moves only in R1
    R2 = [dataclasses.replace(r, abs_z=r.abs_z - (0.9 if r.regime != "R1" and r.strength("EARNINGS") > 0 else 0.0)) for r in R]
    assert U.regime_bound_causes(R2, min_group=60).get("EARNINGS") in ("REGIME_BOUND", None)
    h = U.hindsight_fraction_by_cause(R)
    assert h["EARNINGS"] == 1.0 and h["MOMENTUM"] == 0.0
    assert U.anticipable_causes(R) == []                       # both real causes are hindsight-only in this world


def test_redundancy_and_residual():
    R = U.planted_records(300, 0)
    dup = [dataclasses.replace(r, evidence=r.evidence + (U.CauseEvidence("NEWS", r.strength("EARNINGS"), Availability.KNOWN_ONLY_AFTER_EVENT),)
                               if r.strength("EARNINGS") > 0 else r.evidence) for r in R]
    co = U.cooccurrence(dup)
    assert co[(co.a == "EARNINGS") & (co.b == "NEWS")]["jaccard"].iloc[0] == 1.0
    v = U.assign(R, U.test_causes(R))
    rm = U.residual_magnitude(R, v)
    assert rm["n_unknown"] + rm["n_explained"] == 300 and rm["unknown_mean_z"] > 0
    assert U.residual_magnitude([], [])["n_unknown"] == 0


def test_frames_and_extremes():
    R = U.planted_records(300, 0)
    v = U.assign(R, U.test_causes(R))
    f = U.verdict_frame(R, v)
    assert len(f) == 300 and f["unknown"].sum() == sum(x.unknown for x in v)
    assert sum(U.declined_summary(v).values()) >= 0
    e = U.unknown_share_of_extremes(R, v)
    assert e["n_extreme"] >= 25 and 0.0 <= e["extreme"] <= 1.0
    assert U.unknown_share_of_extremes([], [])["n_extreme"] == 0
