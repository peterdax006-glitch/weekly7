"""S01: KnowledgeObject (section 5), typed sub-records, versioning, store, adapters, and the decision contract (section 43).
Synthetic only; no real caches. Every mechanism has a planted defect it must catch plus the empty case."""
import dataclasses
import json
import math
import types

import pytest

from engine.learning import decision_contract as dc
from engine.learning import knowledge as kn
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle,
                                  Promotion, Provenance, Subsystem, TemporalClass, stable_hash)
from engine.learning.epistemic import (CONTEXT, CURRENT, HISTORICAL, EpistemicProfile, EvidenceSummary)

NOW = "2021-01-04"


def prov(learned="2020-01-01", through="2020-01-01", **kw):
    return Provenance(created_real="2026-09-29T00:00:00+00:00", learned_at=learned, code_hash="abc123", data_hash="d1",
                      config_hash="c1", experiment_id="E1", seed=7, outcomes_seen_through=through, **kw)


def base(**kw):
    d = dict(knowledge_id="K-test", created_at="2020-01-02", provenance=prov(),
             observation="stocks with q4 momentum earned more next week", hypothesis="momentum predicts",
             contexts=kn.context_from_text("volatility: vix >= 20"),
             effect=kn.Effect(1, 0.004, 0.001), confidence=Confidence(0.9, 0.7, 0.8, 0.6, 0.5, 0.2),
             evidence=kn.Evidence(200, 90.0, 0.8, 0.7, "2020-12-01"), decision_effect=(DecisionEffect.RANKING,),
             epistemic=Epistemic.SUPPORTED, lifecycle=Lifecycle.ACTIVE, promotion=Promotion.CHAMPION,
             mechanism_tags=("momentum",))
    d.update(kw)
    return kn.KnowledgeObject(**d)


# ------------------------------------------------------------------ record, validation, codec

def test_valid_object_and_contract_field_views():
    k = base().assert_valid()
    assert k.truth_confidence == 0.9 and k.current_reliability == 0.8 and k.failure_risk == 0.2
    assert k.effect_direction == 1 and k.effective_sample_size == 90.0 and k.experiment_id == "E1"
    assert k.updated_at == k.created_at and not core_conforms_missing(k)


def core_conforms_missing(k):
    from engine.learning.core import KnowledgeLike
    return KnowledgeLike.conforms(k)


@pytest.mark.parametrize("change,needle", [
    (dict(effect=kn.Effect(0, 0.01)), "direction 0"),
    (dict(effect=kn.Effect(1, 0.0)), "zero size"),
    (dict(evidence=kn.Evidence(10, 20.0)), "exceeds sample_size"),
    (dict(epistemic=Epistemic.UNKNOWN, promotion=Promotion.RESEARCH), "UNKNOWN must not carry truth"),
    (dict(decision_effect=(DecisionEffect.NONE, DecisionEffect.RANKING)), "NONE cannot be combined"),
    (dict(decision_effect=()), "decision_effect empty"),
    (dict(decision_effect=(DecisionEffect.NONE,)), "CHAMPION knowledge must change a decision"),
    (dict(epistemic=Epistemic.HYPOTHESIS), "CHAMPION knowledge must be SUPPORTED"),
    (dict(epistemic=Epistemic.RETIRED), "RETIRED must be set consistently"),
    (dict(relations=kn.Relations(supporting=("K-test",))), "references itself"),
    (dict(relations=kn.Relations(supporting=("a",), contradicting=("a",))), "both support and contradict"),
    (dict(confidence=Confidence(truth=1.4)), "outside [0,1]"),
    (dict(updated_at="2019-01-01"), "updated_at precedes"),
    (dict(version=2), "later version must have one"),
    (dict(mechanism_tags=("a", "a")), "unique"),
    (dict(contexts=kn.context_from_text("volatility: vix >= 30"), anti_contexts=kn.ContextSet(
        (kn.parse_condition("volatility: vix >= 20"),), any_of=True)), "swallow the whole context"),
    (dict(failure_explanations=(kn.FailureExplanation(FailureCause.REVERSAL, "2020-06-01"),)), "asserted without a note"),
    (dict(provenance=Provenance(created_real="", learned_at="", code_hash="")), "provenance.created_real missing"),
])
def test_validation_catches_planted_defects(change, needle):
    errs = base(**change).validate()
    assert any(needle in e for e in errs), errs
    with pytest.raises(kn.SchemaError):
        base(**change).assert_valid()


def test_validation_accepts_anti_inside_context():
    k = base(anti_contexts=kn.ContextSet((kn.parse_condition("volatility: vix >= 40"),), any_of=True))
    assert k.validate() == []


def test_json_round_trip_preserves_identity():
    k = base(relations=kn.Relations(supporting=("a", "b")), temporal_class=TemporalClass.REGIME_BOUND,
             failure_explanations=(kn.FailureExplanation(FailureCause.REGIME_CHANGE, "2020-08-01", Subsystem.SELECTION, "bear"),),
             anti_contexts=kn.ContextSet((kn.parse_condition("regime: trend in [bear]"),), any_of=True))
    back = kn.KnowledgeObject.from_json(k.to_json())
    assert back.record_hash() == k.record_hash() and back.content_hash() == k.content_hash()
    assert back.anti_contexts.any_of and back.relations.supporting == ("a", "b")
    assert back.temporal_class is TemporalClass.REGIME_BOUND and back.epistemic is Epistemic.SUPPORTED


def test_round_trip_with_scoped_epistemic_profile():
    ev = EvidenceSummary(n_events=200, n_eff=90, t_discovery=3, t_confirm=2.8, p_real=0.9, contradiction_rate=0.1,
                         recent_ratio=0.9, has_hypothesis=True, has_prediction=True)
    from engine.learning.epistemic import build_profile
    prof = build_profile(NOW, ev, None, ev)
    k = base(epistemic_profile=prof, epistemic=prof.headline())
    assert kn.KnowledgeObject.from_json(k.to_json()).epistemic_profile.summary() == prof.summary()


@pytest.mark.parametrize("mutate", [
    lambda d: d["record"].update(bogus=1),
    lambda d: d["record"]["effect"].update(size="big"),
    lambda d: d["record"].update(epistemic="MAYBE"),
    lambda d: d.update(schema=99),
    lambda d: d["record"].pop("knowledge_id"),
    lambda d: d["record"]["evidence"].update(sample_size=1.5),
])
def test_decoder_fails_closed(mutate):
    d = json.loads(base().to_json())
    mutate(d)
    with pytest.raises((kn.SchemaError, ValueError, TypeError)):
        kn.KnowledgeObject.from_dict(d)


def test_decoder_rejects_invalid_json_and_nonfinite_are_visible():
    with pytest.raises(kn.SchemaError):
        kn.KnowledgeObject.from_json("{not json")
    assert kn.encode(kn.Effect(1, float("nan")))["size"] == "nan"      # NaN stays visible, never silently 0


# ------------------------------------------------------------------ hashing and provenance

def test_hashes_are_deterministic_and_sensitive():
    a, b = base(), base()
    assert a.record_hash() == b.record_hash() and a.content_hash() == b.content_hash()
    c = base(effect=kn.Effect(1, 0.005, 0.001))
    assert c.content_hash() != a.content_hash()
    d = base(provenance=prov(learned="2020-01-01"))
    d2 = dataclasses.replace(d, provenance=dataclasses.replace(d.provenance, created_real="2030-01-01T00:00:00+00:00"))
    assert d.content_hash() == d2.content_hash() and d.record_hash() != d2.record_hash()


def test_derive_id_ignores_nothing_it_should_not():
    c = kn.context_from_text("regime: trend in [bull]")
    assert kn.derive_id("x", c) == kn.derive_id("x", c) != kn.derive_id("y", c)
    assert kn.derive_id("x", c) != kn.derive_id("x", c, source="other")


def test_make_provenance_hashes_code_data_config(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    p1 = kn.make_provenance("2020-05-05", data=str(f), config={"a": 1}, seed=3, code_hash="h")
    p2 = kn.make_provenance("2020-05-05", data=str(f), config={"a": 1}, seed=3, code_hash="h")
    assert (p1.data_hash, p1.config_hash) == (p2.data_hash, p2.config_hash) and p1.code_hash == "h"
    f.write_bytes(b"abd")
    assert kn.make_provenance("2020-05-05", data=str(f), code_hash="h").data_hash != p1.data_hash
    assert kn.make_provenance("2020-05-05", config={"a": 2}, code_hash="h").config_hash != p1.config_hash
    assert kn.make_provenance("2020-05-05").code_hash                  # falls back to the live engine code hash
    assert p1.check() == []


def test_future_knowledge_is_not_visible():
    k = base(provenance=prov(learned="2020-01-01", through="2021-02-01"))     # has seen outcomes after NOW
    assert not k.visible_at(NOW)
    with pytest.raises(FirewallBreach):
        k.assert_visible(NOW)
    assert base().visible_at(NOW) and not base().visible_at("2020-01-02")     # not visible on its own creation day


# ------------------------------------------------------------------ versioning and the store

def test_new_version_links_and_never_mutates_parent():
    v1 = base(promotion=Promotion.RESEARCH)
    h1 = v1.record_hash()
    v2 = v1.new_version("2020-06-01", "more evidence", evidence=kn.Evidence(300, 140.0, 0.9, 0.7, "2020-05-30"))
    assert (v2.version, v2.parent_hash, v2.knowledge_id) == (2, h1, v1.knowledge_id)
    assert v1.record_hash() == h1 and v1.evidence.sample_size == 200
    assert f"{v1.knowledge_id}@v1" in v2.provenance.parents
    assert [c.path for c in kn.diff(v1, v2)] == ["evidence.effective_sample_size", "evidence.last_evidence_at",
                                                 "evidence.recency", "evidence.sample_size", "provenance.parents"]
    assert "evidence" in kn.summarize_change(kn.diff(v1, v2))


@pytest.mark.parametrize("call,exc", [
    (lambda v: v.new_version("2020-06-01", "x"), kn.SchemaError),                                   # no-op
    (lambda v: v.new_version("2020-06-01", "x", version=9), kn.SchemaError),                         # managed field
    (lambda v: v.new_version("2020-06-01", "x", knowledge_id="other"), kn.SchemaError),
    (lambda v: v.new_version("2020-06-01", "x", nonsense=1), kn.SchemaError),
    (lambda v: v.new_version("2020-06-01", " ", effect=kn.Effect(1, 0.01)), kn.SchemaError),          # needs reason
    (lambda v: v.new_version("2019-01-01", "x", effect=kn.Effect(1, 0.01)), FirewallBreach),          # back in time
    (lambda v: v.new_version("2020-06-01", "x", learned_at="2019-01-01"), FirewallBreach),            # learned earlier
    (lambda v: v.new_version("2020-06-01", "x", effect=kn.Effect(0, 0.5)), kn.SchemaError),          # invalid result
])
def test_new_version_refusals(call, exc):
    with pytest.raises(exc):
        call(base(promotion=Promotion.RESEARCH))


def test_retire_is_a_version_not_a_deletion():
    store = kn.KnowledgeStore()
    v1 = store.add(base(promotion=Promotion.RESEARCH))
    v2 = store.add(v1.retire("2020-09-01", "reversed in 2020H2", FailureCause.REVERSAL))
    assert store.latest(v1.knowledge_id).epistemic is Epistemic.RETIRED and len(store.history(v1.knowledge_id)) == 2
    assert store.get(v1.knowledge_id, 1).epistemic is Epistemic.SUPPORTED          # history intact
    assert v2.failure_explanations[-1].cause is FailureCause.REVERSAL
    assert store.as_of(v1.knowledge_id, "2020-08-01").version == 1                  # before retirement it was live


def test_store_chain_rules_and_as_of():
    s = kn.KnowledgeStore()
    assert len(s) == 0 and s.visible(NOW) == [] and s.latest("x") is None and s.ids() == []
    v1 = s.add(base(promotion=Promotion.RESEARCH))
    with pytest.raises(kn.ChainError):
        s.add(v1)                                                                  # version 1 twice
    v2 = v1.new_version("2020-06-01", "r", effect=kn.Effect(1, 0.006, 0.001))
    v3 = v2.new_version("2020-09-01", "r", effect=kn.Effect(1, 0.002, 0.001))
    with pytest.raises(kn.ChainError):
        s.add(v3)                                                                  # skipped v2
    s.add(v2)
    forged = dataclasses.replace(v3, parent_hash="0" * 24)
    with pytest.raises(kn.ChainError):
        s.add(forged)
    s.add(v3)
    assert [k.version for k in s.history(v1.knowledge_id)] == [1, 2, 3] and s.verify() == []
    assert s.as_of(v1.knowledge_id, "2020-07-01").version == 2
    assert s.as_of(v1.knowledge_id, "2020-01-02") is None                           # did not exist yet
    assert s.as_of(v1.knowledge_id, "2020-09-01").version == 2                      # v3 is dated ON that day: not yet known
    assert len(s.visible(NOW)) == 1 and kn.epistemic_timeline(s, v1.knowledge_id)[0][0] == 1


def test_store_dump_load_and_tamper_detection(tmp_path):
    s = kn.KnowledgeStore()
    v1 = s.add(base(promotion=Promotion.RESEARCH))
    s.add(v1.new_version("2020-06-01", "r", effect=kn.Effect(1, 0.006, 0.001)))
    p = tmp_path / "k.jsonl"
    assert s.dump(p) == 2 and b"\r\n" not in p.read_bytes()
    assert kn.KnowledgeStore.load(p).latest(v1.knowledge_id).record_hash() == s.latest(v1.knowledge_id).record_hash()
    assert kn.verify_file(p) == []
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"size":0.004', '"size":0.009')                     # edit history after the fact
    p.write_bytes(("\n".join(lines) + "\n").encode())
    assert kn.verify_file(p)                                                        # child's parent_hash no longer matches
    p.write_bytes(b"")
    assert len(kn.KnowledgeStore.load(p)) == 0                                      # empty file is a valid empty store
    assert kn.verify_file(tmp_path / "missing.jsonl")


def test_store_queries_stats_lineage_and_future_audit():
    s = kn.KnowledgeStore()
    a = s.add(base(knowledge_id="K-a", promotion=Promotion.RESEARCH))
    s.add(base(knowledge_id="K-b", promotion=Promotion.RESEARCH, epistemic=Epistemic.HYPOTHESIS,
               decision_effect=(DecisionEffect.NONE,), mechanism_tags=("other",),
               provenance=prov(parents=("K-a@v1",))))
    s.add(base(knowledge_id="K-c", promotion=Promotion.RESEARCH,
               provenance=prov(learned="2020-01-01", through="2021-03-01")))
    assert [k.knowledge_id for k in kn.find(s, NOW, epistemic="SUPPORTED")] == ["K-a"]      # K-c is future
    assert [k.knowledge_id for k in kn.find(s, NOW, tag="other")] == ["K-b"]
    assert [k.knowledge_id for k in kn.find(s, NOW, effect=DecisionEffect.RANKING)] == ["K-a"]
    assert kn.lineage(s, "K-b") == ["K-a"] and kn.lineage(s, "K-a") == []
    stats = kn.store_stats(s, NOW)
    assert stats["items"] == 3 and stats["visible"] == 2 and stats["research_only"] == 1
    assert any("K-c" in x for x in kn.audit_future(s, NOW)) and not any("K-a" in x for x in kn.audit_future(s, NOW))


# ------------------------------------------------------------------ contexts (section 8 dimensions)

def test_conditions_and_tri_state_applicability():
    ctx = kn.context_from_text("volatility: vix >= 20", "regime: trend in [bull, sideways]")
    anti = kn.ContextSet((kn.parse_condition("liquidity: adv lt 1e6"), kn.parse_condition("stock: gap gt 0.2")), any_of=True)
    A = kn.Applicability
    assert kn.applicability(ctx, anti, {"vix": 25, "trend": "bull", "adv": 5e6, "gap": 0.0}) is A.APPLIES
    assert kn.applicability(ctx, anti, {"vix": 15, "trend": "bull", "adv": 5e6, "gap": 0.0}) is A.OUT_OF_CONTEXT
    assert kn.applicability(ctx, anti, {"vix": 25, "trend": "bull", "adv": 5e5, "gap": 0.0}) is A.EXCLUDED
    assert kn.applicability(ctx, anti, {"vix": 25, "trend": "bull", "adv": 5e6, "gap": 0.5}) is A.EXCLUDED   # either one excludes
    assert kn.applicability(ctx, anti, {"vix": 25, "trend": "bull", "adv": 5e6}) is A.UNKNOWN               # missing feature
    assert kn.applicability(ctx, anti, {"vix": float("nan"), "trend": "bull", "adv": 5e6, "gap": 0}) is A.UNKNOWN
    assert kn.applicability(kn.ContextSet(), kn.ContextSet(), {}) is A.APPLIES                               # no restriction
    assert set(ctx) == {"volatility", "regime"} and len(ctx) == 2 and ctx["regime"][0].labels == ("bull", "sideways")


@pytest.mark.parametrize("text", ["", "vol vix >= 3", "volatility: vix ~= 3", "volatility: vix >= abc def",
                                  "nonsense: vix >= 3", "regime: trend in [1, bull]", "volatility: vix between 5 1"])
def test_parse_condition_is_strict(text):
    with pytest.raises(kn.SchemaError):
        kn.parse_condition(text)


def test_condition_implication_and_overlap():
    p = kn.parse_condition
    assert p("volatility: vix between 25 30").implies(p("volatility: vix ge 20"))
    assert not p("volatility: vix ge 20").implies(p("volatility: vix ge 25"))
    assert p("regime: trend eq bull").implies(p("regime: trend in [bull, bear]"))
    assert p("regime: trend in [bull]").implies(p("regime: trend ne bear"))
    assert not p("regime: trend eq bull").implies(p("volatility: vix ge 20"))
    a, b = kn.context_from_text("volatility: vix >= 20"), kn.context_from_text("volatility: vix >= 30")
    grid = [{"vix": v} for v in range(10, 50)]
    assert kn.context_overlap(a, a, grid) == 1.0 and 0 < kn.context_overlap(a, b, grid) < 1
    assert kn.context_overlap(a, b, []) == 0.0


# ------------------------------------------------------------------ pooling and evidence arithmetic

def test_pool_effects_widens_uncertainty_under_heterogeneity():
    homog = [kn.Effect(1, 0.010, 0.002), kn.Effect(1, 0.011, 0.002), kn.Effect(1, 0.009, 0.002)]
    hetero = [kn.Effect(1, 0.030, 0.002), kn.Effect(-1, 0.020, 0.002), kn.Effect(1, 0.010, 0.002)]
    ph, pe = kn.pool_effects(homog), kn.pool_effects(hetero)
    assert ph.tau2 == 0.0 and ph.i2 == 0.0 and abs(ph.effect.size - 0.010) < 5e-4
    assert pe.tau2 > 0 and pe.i2 > 0.9 and pe.effect.uncertainty > 3 * ph.effect.uncertainty
    assert kn.pool_effects(hetero, random=False).effect.uncertainty < pe.effect.uncertainty
    assert kn.pool_effects([kn.Effect(-1, 0.01, 0.002)]).effect.direction == -1


@pytest.mark.parametrize("effs", [[], [kn.Effect(1, 0.01)], [kn.Effect(1, 0.01, 0.0)],
                                  [kn.Effect(1, 0.01, 0.1, "a"), kn.Effect(1, 0.01, 0.1, "b")]])
def test_pool_effects_refuses_unpoolable(effs):
    with pytest.raises(kn.SchemaError):
        kn.pool_effects(effs)


def test_evidence_pooling_and_scores():
    e = kn.pool_evidence([kn.Evidence(100, 50.0, 1.0, 0.0, "2019-01-01"), kn.Evidence(300, 100.0, 0.0, 1.0, "2020-01-01")])
    assert (e.sample_size, e.effective_sample_size, e.last_evidence_at) == (400, 150.0, "2020-01-01")
    assert abs(e.recency - 0.25) < 1e-12 and abs(e.modernity - 0.75) < 1e-12
    assert kn.pool_evidence([]) == kn.Evidence()
    assert abs(kn.recency_score("2020-01-01", "2021-01-01", 366) - 0.5) < 1e-3
    with pytest.raises(FirewallBreach):
        kn.recency_score("2021-01-01", "2021-01-01")
    assert kn.modernity_score(["2015-01-01", "2021-01-01"], "2020-01-01") == 0.5 and kn.modernity_score([], "2020-01-01") is None


def test_effect_interval_and_zero_exclusion():
    e = kn.Effect(-1, 0.004, 0.001)
    lo, hi = e.interval()
    assert e.signed == -0.004 and hi < 0 and e.excludes_zero() is True
    assert kn.Effect(1, 0.001, 0.002).excludes_zero() is False and kn.Effect(1, 0.001).excludes_zero() is None


# ------------------------------------------------------------------ adapters over existing stores

def test_adapter_from_pattern_row_maps_state_and_keeps_untested_dimensions_untested():
    row = {"key_named": "rsi q4 & vol q0", "effect": -0.012, "t_disc": 3.0, "t_conf": 2.4, "p_real": 0.83, "status": "no_gain"}
    k = kn.from_pattern_row(row, NOW, prov(), n=120, n_eff=60)
    assert k.epistemic is Epistemic.GATED and k.lifecycle is Lifecycle.DORMANT and k.promotion is Promotion.RESEARCH
    assert k.effect.direction == -1 and abs(k.effect.uncertainty - 0.012 / 2.4) < 1e-9
    assert k.confidence.truth == 0.83 and k.confidence.usefulness is None and k.confidence.current_reliability is None
    assert set(k.decision_effect) == {DecisionEffect.NONE}
    assert kn.from_pattern_row({**row, "status": "discarded"}, NOW, prov()).promotion is Promotion.RETIRED
    from engine.learning.epistemic import EpistemicError
    with pytest.raises(EpistemicError):
        kn.from_pattern_row({**row, "status": "mystery"}, NOW, prov())


def test_adapter_from_pattern_record_uses_its_identity_and_unless_becomes_anti_context():
    pi = pytest.importorskip("engine.pattern_identity")
    rec = pi.new_record("rsi q4 & vol q0 unless gap q4 unless beta q0", discovery=("2015-01-01", "2018-12-31"),
                        validation=("2019-01-01", "2020-12-31"),
                        stats={"effect": 0.01, "t_conf": 2.5, "p_real": 0.9, "n_eff": 80, "n_rows": 400}, state="active")
    k = kn.from_pattern_record(rec, NOW, prov(), (DecisionEffect.RANKING,))
    assert k.knowledge_id == "K-" + rec.id                                             # same identity as the pattern store
    assert k.epistemic is Epistemic.SUPPORTED and k.anti_contexts.any_of and len(k.anti_contexts.conditions) == 2
    assert k.evidence.sample_size == 400 and k.evidence.effective_sample_size == 80.0 and k.confidence.truth == 0.9
    sit = {"rsi.q": 4, "vol.q": 0, "gap.q": 1, "beta.q": 2}
    assert k.applicability(sit) is kn.Applicability.APPLIES
    assert k.applicability({**sit, "gap.q": 4}) is kn.Applicability.EXCLUDED
    assert k.applicability({**sit, "beta.q": 0}) is kn.Applicability.EXCLUDED
    dead = rec.transition("failed", "2020-06-01", "x").transition("cause_search", "2020-06-02").transition("discarded", "2020-06-03")
    kd = kn.from_pattern_record(dead, NOW, prov(), (DecisionEffect.RANKING,))
    assert kd.epistemic is Epistemic.RETIRED and set(kd.decision_effect) == {DecisionEffect.NONE}
    scoped = pi.new_record("rsi q4", discovery=("2015-01-01", "2018-12-31"), validation=("2019-01-01", "2020-12-31"),
                           state="rescoped", scope=pi.Scope("m_vix", "high", 10.0, 20.0), stats={"effect": 0.01})
    ks = kn.from_pattern_record(scoped, NOW, prov())
    assert ks.epistemic is Epistemic.CONDITIONAL and ks.applicability({"rsi.q": 4, "m_vix": 25}) is kn.Applicability.APPLIES
    assert ks.applicability({"rsi.q": 4, "m_vix": 15}) is kn.Applicability.OUT_OF_CONTEXT


def test_adapter_from_lesson_keeps_trust_as_reliability_only():
    ls = pytest.importorskip("engine.lessons")
    L = ls.Lesson(lid="L1", conds=[("m_vix", ">", 25.0), ("rsi", "<=", 0.3)], direction=-1, factor=0.5, category="regime_failure",
                  n=120, n_weeks=30, n_tickers=40, delta=-0.006, t=-2.5, p=0.01, val_delta=-0.004, a=8.0, b=2.0, born_tick=1,
                  ttl=400, support=["e1", "e2"], kind="oversized_loser", action="advisory:cap_size")
    k = kn.from_lesson(L, NOW, prov())
    assert k.confidence.current_reliability == 0.8 and k.confidence.truth is None and k.confidence.usefulness is None
    assert k.decision_effect == (DecisionEffect.POSITION_SIZE,) and k.effect.direction == -1
    assert k.failure_explanations[0].cause is FailureCause.REGIME_CHANGE and k.failure_explanations[0].subsystem is Subsystem.RISK
    assert k.applicability({"m_vix": 30, "rsi": 0.1}) is kn.Applicability.APPLIES
    assert k.applicability({"m_vix": 30, "rsi": 0.9}) is kn.Applicability.OUT_OF_CONTEXT
    L.status = "retired"
    assert kn.from_lesson(L, NOW, prov()).epistemic is Epistemic.RETIRED
    L.conds = [("x", "==", 1.0)]
    with pytest.raises(kn.SchemaError):
        kn.from_lesson(L, NOW, prov())


# ------------------------------------------------------------------ decision contract (section 43)

def champ(**kw):
    return base(**kw)


def test_research_only_is_refused_and_answers_the_question():
    r = base(promotion=Promotion.RESEARCH, decision_effect=(DecisionEffect.NONE,))
    v = dc.check(r, NOW)
    assert v.research_only and not v.allowed and "no decision effect" in v.reasons[0]
    with pytest.raises(dc.ResearchOnlyError):
        dc.require_production(r, NOW)
    assert "changes no decision" in dc.answer(r) and "candidate_rank_score" in dc.answer(champ())
    assert dc.research_only([r, champ()]) == ["K-test"] and dc.declared_effects(r) == ()


def test_champion_passes_and_each_gate_refuses_when_broken():
    assert dc.require_production(champ(), NOW).effects == (DecisionEffect.RANKING,)
    bad = {
        "promotion": champ(promotion=Promotion.CHALLENGER),
        "lifecycle": champ(lifecycle=Lifecycle.FAILURE, epistemic=Epistemic.SUPPORTED),
        "untested": champ(confidence=Confidence(truth=0.9, usefulness=None, current_reliability=0.8)),
        "usefulness": champ(confidence=Confidence(truth=0.9, usefulness=0.2, current_reliability=0.8)),
        "reliability": champ(confidence=Confidence(truth=0.9, usefulness=0.7, current_reliability=0.1)),
        "failure_risk": champ(confidence=Confidence(0.9, 0.7, 0.8, failure_risk=0.9)),
        "epistemic": champ(epistemic=Epistemic.GATED),
        "future": champ(provenance=prov(through="2021-06-01")),
    }
    for name, k in bad.items():
        v = dc.check(k, NOW)
        assert not v.allowed and v.reasons, name
        with pytest.raises(dc.ProductionRefused):
            dc.require_production(k, NOW)
    assert dc.check(champ(), "2020-01-02").allowed is False                     # not yet in existence at that day
    assert dc.check(bad["promotion"], NOW, dc.Mode.SHADOW).allowed and not dc.check(bad["promotion"], NOW).allowed
    assert dc.check(bad["untested"], NOW, dc.Mode.RESEARCH).allowed


def test_invalid_record_is_refused_even_if_fields_look_fine():
    k = base(effect=kn.Effect(0, 0.5))                                          # inconsistent effect
    v = dc.check(k, NOW)
    assert not v.allowed and any("invalid record" in r for r in v.reasons)


def test_position_size_needs_a_size_bound_and_research_priority_ignores_trading_gates():
    k = champ(decision_effect=(DecisionEffect.POSITION_SIZE,))
    assert not dc.check(k, NOW).allowed and "size_multiplier_max" in " ".join(dc.check(k, NOW).reasons)
    ok = dataclasses.replace(k, action_policy=kn.ActionPolicy(size_multiplier_max=1.5))
    assert dc.check(ok, NOW).allowed
    rp = base(promotion=Promotion.RESEARCH, epistemic=Epistemic.UNKNOWN, confidence=Confidence(),
              decision_effect=(DecisionEffect.RESEARCH_PRIORITY,), lifecycle=Lifecycle.BIRTH)
    assert dc.check(rp, NOW).allowed                                            # unknown items MAY steer research
    assert not dc.check(dataclasses.replace(rp, decision_effect=(DecisionEffect.RANKING,)), NOW).allowed
    retired = rp.retire("2020-06-01", "gone")
    assert not dc.check(retired, NOW).allowed                                   # retired cannot even steer research
    mixed = champ(decision_effect=(DecisionEffect.RANKING, DecisionEffect.POSITION_SIZE))
    v = dc.check(mixed, NOW)
    assert v.allowed and v.effects == (DecisionEffect.RANKING,) and any("POSITION_SIZE" in r for r in v.reasons)


def test_influence_weight_and_unknown_applicability_never_grant_weight():
    k = champ()
    inf = dc.influence(k, {"vix": 30}, NOW)
    assert 0 < inf.weight <= 0.8 and not inf.abstain                            # bounded by current_reliability 0.8
    unknown = dc.influence(k, {}, NOW)
    assert unknown.weight == 0.0 and unknown.abstain and "lacks a feature" in unknown.reason
    assert dc.influence(k, {"vix": 10}, NOW).weight == 0.0
    silent = dc.influence(champ(promotion=Promotion.RESEARCH), {"vix": 30}, NOW)
    assert silent.weight == 0.0 and silent.effects == ()
    capped = dataclasses.replace(k, action_policy=kn.ActionPolicy(weight_cap=0.3))
    assert dc.influence(capped, {"vix": 30}, NOW).weight == 0.3
    sized = dataclasses.replace(k, decision_effect=(DecisionEffect.POSITION_SIZE,), action_policy=kn.ActionPolicy(size_multiplier_max=1.5))
    i2 = dc.influence(sized, {"vix": 30}, NOW)
    assert 1.0 < dc.size_multiplier(i2, sized) <= 1.5 and dc.size_multiplier(unknown, sized) == 1.0


def test_epistemic_profile_veto_reaches_the_weight():
    ev = EvidenceSummary(n_events=200, n_eff=90, t_confirm=2.8, p_real=0.9, contradiction_rate=0.1, recent_ratio=0.9,
                         has_hypothesis=True, has_prediction=True, works_in=("vol:vix",), fails_in=("regime",))
    from engine.learning.epistemic import build_profile
    prof = build_profile(NOW, dataclasses.replace(ev, fails_in=()), None, dataclasses.replace(ev, fails_in=()))
    k = champ(epistemic_profile=prof, epistemic=prof.headline())
    assert dc.influence(k, {"vix": 30}, NOW).weight > 0


def test_apply_knobs_respect_their_limits():
    C = dc.Contribution
    R, S, P, X, A, F = (DecisionEffect.RANKING, DecisionEffect.SELECTION, DecisionEffect.POSITION_SIZE, DecisionEffect.STOP,
                        DecisionEffect.ABSTENTION, DecisionEffect.CONFIDENCE)
    base_scores = {"a": 1.0, "b": 0.0, "c": 0.5}
    huge = [C("k1", R, 1.0, 100.0, "b")]
    out = dc.apply_ranking(base_scores, huge)
    assert out["b"] == 0.5 and out["a"] == 1.0                                       # shift limited to half the spread
    assert dc.apply_ranking(base_scores, [C("k1", R, 1.0, 100.0, "zzz")]) == base_scores      # unknown target ignored
    assert dc.apply_ranking({}, huge) == {}
    kept, removed = dc.apply_selection(["a", "b"], [C("k", S, 1.0, -1.0, "a"), C("k2", S, 1.0, +5.0, "zzz"), C("k3", S, 0.0, -1.0, "b")])
    assert kept == ["b"] and removed == [("a", "k")]                                 # veto only; weight 0 does nothing
    assert dc.apply_position_size(0.1, [1.5, 0.5]) == pytest.approx(0.075)
    assert dc.apply_position_size(0.1, [10, 10]) == pytest.approx(0.2)              # capped at 2x
    with pytest.raises(dc.ContractError):
        dc.apply_position_size(0.1, [-1])
    assert dc.apply_stop(0.10, [C("k", X, 1.0, 0.05)]) == pytest.approx(0.05)
    assert dc.apply_stop(0.10, [C("k", X, 1.0, 0.30)]) == 0.10                       # cannot loosen
    assert dc.apply_stop(0.10, [C("k", X, 1.0, 0.30)], may_loosen=True) == pytest.approx(0.30)
    assert dc.apply_stop(0.10, []) == 0.10
    with pytest.raises(dc.ContractError):
        dc.apply_stop(0.0, [])
    assert dc.apply_abstention([C("k", A, 0.9)]) == (True, ["k"]) and dc.apply_abstention([C("k", A, 0.1)]) == (False, [])
    assert dc.apply_confidence(0.8, [C("k", F, 0.5, -1.0)]) == pytest.approx(0.4)
    assert dc.apply_confidence(0.8, [C("k", F, 0.5, +1.0)]) == 0.8                   # never inflated
    with pytest.raises(dc.ContractError):
        dc.apply_confidence(1.5, [])


def test_decision_log_chain_and_tamper_detection():
    log = dc.DecisionLog()
    assert len(log) == 0 and log.verify() == [] and log.usage_counts() == {}
    ks = [champ(knowledge_id="K-1"), champ(knowledge_id="K-2", promotion=Promotion.RESEARCH)]
    infs = log.record_all("D1", ks, {"vix": 30}, NOW)
    assert [i.weight > 0 for i in infs] == [True, False]
    log.record_all("D2", ks[:1], {}, "2021-01-05")
    assert log.verify() == [] and log.usage_counts() == {"K-1": 1} and log.silent_ids(["K-1", "K-2", "K-3"]) == ["K-2", "K-3"]
    assert len(log.for_decision("D1")) == 2
    with pytest.raises(FirewallBreach):
        log.record("D0", ks[0], infs[0], "2020-01-01")
    log._rows[0] = dataclasses.replace(log._rows[0], weight=0.99)
    assert log.verify()


def test_registry_audit_finds_orphans_conflicts_and_future():
    a, b = champ(knowledge_id="K-a", relations=kn.Relations(contradicting=("K-b",))), champ(knowledge_id="K-b")
    ghost = types.SimpleNamespace(knowledge_id="K-g", promotion=Promotion.CHAMPION, decision_effect=(DecisionEffect.NONE,))
    audit = dc.audit_registry([a, b, ghost], NOW)
    assert audit.orphan_champions == ("K-g",) and audit.contradictory_pairs == (("K-a", "K-b"),) and not audit.clean()
    assert audit.total == 3 and set(audit.allowed) == {"K-a", "K-b"} and "K-g" in audit.research_only
    assert dc.decision_map([a, b, ghost], NOW) == {"RANKING": ["K-a", "K-b"]}
    empty = dc.audit_registry([], NOW)
    assert empty.total == 0 and empty.clean() and dc.decision_map([], NOW) == {}
    dc.assert_no_future([a], NOW)
    with pytest.raises(FirewallBreach):
        dc.assert_no_future([a], "2020-01-02")


# ------------------------------------------------------------------ common edits are versions with guards

def test_confidence_failure_relation_edits_create_versions():
    v1 = base(promotion=Promotion.RESEARCH)
    v2 = kn.with_confidence(v1, "2020-06-01", "measured usefulness", usefulness=0.9)
    assert v2.confidence.usefulness == 0.9 and v2.confidence.truth == 0.9 and v1.confidence.usefulness == 0.7
    v3 = kn.with_confidence(v2, "2020-06-02", "re-mark untested", transfer=None)
    assert v3.confidence.transfer is None and v3.version == 3
    with pytest.raises(kn.SchemaError):
        kn.with_confidence(v1, "2020-06-01", "x", nonsense=0.5)
    with pytest.raises(kn.SchemaError):
        kn.with_confidence(v1, "2020-06-01", "x", truth=1.5)                          # invalid value rejected by validation
    f = kn.with_failure(v1, "2020-07-01", FailureCause.REGIME_CHANGE, Subsystem.SELECTION, "bear market", "ep-9")
    assert f.failure_explanations[0].cause is FailureCause.REGIME_CHANGE and f.version_reason.startswith("failure explained")
    with pytest.raises(kn.SchemaError):
        kn.with_failure(v1, "2020-07-01", FailureCause.REVERSAL)                      # a cause needs a note or evidence
    assert kn.with_failure(v1, "2020-07-01", FailureCause.UNKNOWN).failure_explanations[0].cause is FailureCause.UNKNOWN
    r = kn.with_relation(v1, "2020-08-01", "contradicting", "K-other")
    assert r.relations.contradicting == ("K-other",)
    with pytest.raises(kn.SchemaError):
        kn.with_relation(r, "2020-08-02", "contradicting", "K-other")
    with pytest.raises(kn.SchemaError):
        kn.with_relation(v1, "2020-08-01", "friend", "K-x")
    with pytest.raises(kn.SchemaError):
        kn.with_relation(kn.with_relation(v1, "2020-08-01", "supporting", "K-z"), "2020-08-02", "contradicting", "K-z")
    assert kn.with_temporal_class(v1, "2020-09-01", "SLOW_DECAY", "half-life fit").temporal_class is TemporalClass.SLOW_DECAY


def test_duplicate_lessons_are_reported_not_merged():
    s = kn.KnowledgeStore()
    s.add(base(knowledge_id="K-1", promotion=Promotion.RESEARCH))
    s.add(base(knowledge_id="K-2", promotion=Promotion.RESEARCH))
    s.add(base(knowledge_id="K-3", promotion=Promotion.RESEARCH, effect=kn.Effect(1, 0.02, 0.001)))
    assert kn.find_duplicates(s, NOW) == [("K-1", "K-2")] and len(s) == 3
    assert kn.same_content(s.latest("K-1"), s.latest("K-2")) and not kn.same_content(s.latest("K-1"), s.latest("K-3"))
    assert kn.find_duplicates(kn.KnowledgeStore(), NOW) == []


# ------------------------------------------------------------------ contract: policy, readiness, conflicts, log persistence

def test_policy_can_only_tighten_and_bad_policy_is_refused():
    k = champ(epistemic=Epistemic.DEGRADED, confidence=Confidence(0.9, 0.7, 0.8, failure_risk=0.4))
    assert dc.policy_check(k, NOW).allowed
    strict = dc.ContractPolicy(allow_degraded_in_production=False)
    v = dc.policy_check(k, NOW, policy=strict)
    assert not v.allowed and "excludes DEGRADED" in " ".join(v.reasons) and v.effects == ()
    assert not dc.policy_check(k, NOW, policy=dc.ContractPolicy(max_failure_risk=0.3)).allowed
    assert not dc.policy_check(k, NOW, policy=dc.ContractPolicy(min_truth=0.95)).allowed
    assert dc.policy_check(k, NOW, dc.Mode.RESEARCH, strict).allowed                    # research mode is not gated by trading policy
    with pytest.raises(dc.ContractError):
        dc.policy_check(k, NOW, policy=dc.ContractPolicy(min_usefulness=2.0))
    assert dc.DEFAULT_POLICY.as_dict()["min_usefulness"] == 0.5


def test_readiness_lists_exactly_what_is_missing():
    ok = dc.readiness(champ(promotion=Promotion.CHALLENGER), NOW)
    assert ok["ready"] and ok["missing"] == [] and ok["effects"] == ["RANKING"]
    weak = base(promotion=Promotion.RESEARCH, decision_effect=(DecisionEffect.NONE,), confidence=Confidence(truth=0.8),
                epistemic=Epistemic.HYPOTHESIS, lifecycle=Lifecycle.BIRTH)
    r = dc.readiness(weak, NOW)
    assert not r["ready"] and any("declare a decision effect" in m for m in r["missing"])
    assert set(r["untested_confidence"]) == {"usefulness", "current_reliability", "context", "transfer", "failure_risk"}
    changing = dataclasses.replace(weak, decision_effect=(DecisionEffect.RANKING,))
    m = dc.readiness(changing, NOW)["missing"]
    assert any("usefulness was never measured" in x for x in m) and any("not allowed for epistemic HYPOTHESIS" in x for x in m)


def test_contributions_conflicts_and_net_direction():
    R = DecisionEffect.RANKING
    a, b = champ(knowledge_id="K-a"), champ(knowledge_id="K-b", effect=kn.Effect(-1, 0.004, 0.001))
    ia, ib = dc.influence(a, {"vix": 30}, NOW), dc.influence(b, {"vix": 30}, NOW)
    ca, cb = dc.contribution_from(a, ia, R, "s1"), dc.contribution_from(b, ib, R, "s1")
    assert ca.signed == pytest.approx(0.004) and cb.signed == pytest.approx(-0.004)
    assert dc.contribution_from(a, ia, DecisionEffect.STOP, "s1") is None                       # effect not permitted
    assert dc.contribution_from(a, dc.influence(a, {}, NOW), R, "s1") is None                   # silent influence
    assert dc.conflicting_contributions([ca, cb]) == [("RANKING", "s1", "K-a", "K-b")]
    assert dc.conflicting_contributions([ca, dataclasses.replace(cb, target="s2")]) == []
    assert dc.conflicting_contributions([]) == []
    assert dc.net_direction([ca, cb], R, "s1") == pytest.approx(0.0, abs=1e-12)                 # equal weights, opposite signs
    assert dc.net_direction([ca], R, "s1") == pytest.approx(0.004) and dc.net_direction([], R, "s1") == 0.0
    stop_item = dataclasses.replace(a, decision_effect=(DecisionEffect.STOP,))
    stop = dc.contribution_from(stop_item, dc.influence(stop_item, {"vix": 30}, NOW), DecisionEffect.STOP, "", requested=0.05)
    assert stop.signed == 0.05


def test_decision_log_persistence_refuses_a_tampered_log():
    log = dc.DecisionLog()
    log.record_all("D1", [champ(knowledge_id="K-1")], {"vix": 30}, NOW)
    log.record_all("D2", [champ(knowledge_id="K-1")], {"vix": 30}, "2021-01-05")
    rows = dc.dump_log(log)
    assert dc.load_log(rows).verify() == [] and len(dc.load_log(rows)) == 2 and len(dc.load_log([])) == 0
    rows[0] = {**rows[0], "weight": 0.99}
    with pytest.raises(dc.ContractError):
        dc.load_log(rows)


def test_production_changes_track_what_started_and_stopped():
    early = champ(knowledge_id="K-early", provenance=prov(learned="2020-01-01", through="2020-01-01"))
    late = champ(knowledge_id="K-late", created_at="2020-11-01", provenance=prov(learned="2020-10-31", through="2020-10-31"))
    gone = champ(knowledge_id="K-gone", provenance=prov(learned="2020-01-01", through="2020-01-01"))
    items = [early, late, gone]
    ch = dc.production_changes(items, "2020-06-01", "2021-01-04")
    assert ch.added == ("K-late",) and ch.removed == () and ch.unchanged == ("K-early", "K-gone") and not ch.is_stable()
    retired = gone.retire("2020-09-01", "reversed")
    ch2 = dc.production_changes([early, gone, retired], "2020-06-01", "2021-01-04")
    assert ch2.removed == ("K-gone",) and "lifecycle RETIRED" in ch2.why_removed[0][1]
    assert dc.production_changes([], "2020-06-01", "2021-01-04").is_stable()
    with pytest.raises(FirewallBreach):
        dc.production_changes(items, "2021-01-04", "2020-06-01")
