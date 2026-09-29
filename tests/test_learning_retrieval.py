"""Tests for engine.learning.context and engine.learning.retrieval (contract C62 sections 8, 17; canon C56, C58-C61, C63).

Synthetic data only. Planted effects the code must find (a context-dependent pattern, an anti-context, a synergy, a boundary),
planted defects it must NOT be fooled by (pure noise, an effect that existed only early, redundant twins, a contradicted item,
a retrieval that has no skill), anti-leakage (future outcomes, future knowledge) and anti-memorization (ticker/date/year scrambling
leaves retrieval unchanged)."""
import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from engine.learning import context as CX
from engine.learning import retrieval as RT
from engine.learning import situation as ST
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance,
                                  TemporalClass, Unknown, UnknownAction)
from test_learning_situation import SIC, mk, panel

FAR = dt.date(2030, 1, 1)
REGS = ["bull_calm", "bull_volatile", "correction_calm", "bear_volatile", "stress"]
VOLREG = {"bull_calm": "low", "bull_volatile": "mid", "correction_calm": "low", "bear_volatile": "high", "stress": "crisis"}


def gen(pid, n, seed, effect, sd=0.02, start=dt.date(2016, 1, 4), extra=None):
    """n matured observations of pattern `pid`; effect(situation, i) is the true mean edge."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lab = REGS[int(rng.integers(len(REGS)))]
        sit = mk(regime__label=lab, regime__vol_regime=VOLREG[lab], volatility__vol_rank=float(rng.uniform(.05, .95)),
                 trend__state=str(rng.choice(["up", "down", "flat"])), liquidity__dv_rank=float(rng.uniform(.05, .95)),
                 patterns=(extra,) if extra and rng.random() < .5 else ())
        out.append(CX.Obs(pid, sit, float(effect(sit, i) + rng.normal(0, sd)), start + dt.timedelta(days=i)))
    return out


def cfg(**kw):
    return CX.ContextConfig(n_perm=200, **kw)


# ------------------------------------------------------------------------------------------------ condition language

def test_context_spec_three_valued_matching_and_mapping_roundtrip():
    spec = CX.ContextSpec((CX.Condition("regime.label", ("bull_calm",)), CX.Condition("volatility.vol_rank", ("b0", "b1"))))
    assert spec.matches(mk(regime__label="bull_calm", volatility__vol_rank=0.1)) is True
    assert spec.matches(mk(regime__label="stress", volatility__vol_rank=0.1)) is False
    assert spec.matches(ST.coarsen(mk(), keep=["regime"])) is None               # vol_rank unobserved: UNKNOWN, not False
    assert spec.matches(ST.coarsen(mk(regime__label="stress"), keep=["volatility"])) is False     # vol_rank b2 fails on its own
    assert spec.matches(ST.coarsen(mk(volatility__vol_rank=0.1), keep=["volatility"])) is None    # regime unobserved, nothing fails
    assert spec.matches_not(mk(regime__label="stress")) is True
    again = CX.ContextSpec.from_mapping(spec.to_mapping())
    assert again == spec and again.spec_id == spec.spec_id
    assert CX.ContextSpec().matches(mk()) is True and CX.ContextSpec().describe() == "always"
    assert CX.Condition("regime.label", ("stress",), negate=True).holds(mk(regime__label="bull_calm")) is True
    with pytest.raises(ValueError):
        CX.Condition("regime.label", ())
    with pytest.raises(KeyError):
        CX.Condition("nope.field", ("x",))
    with pytest.raises(ValueError):
        CX.ContextSpec((CX.Condition("regime.label", ("a",)), CX.Condition("regime.label", ("b",))))
    assert CX.Condition.from_mapping("regime.label", "stress").allowed == ("stress",)
    assert CX.Condition.from_mapping("regime.label", ["stress", "bull_calm"]).allowed == ("bull_calm", "stress")


def test_virtual_pattern_dimension():
    c = CX.Condition("pattern.has:p9", ("yes",))
    assert c.holds(mk(patterns=("p9",))) is True and c.holds(mk()) is False


# ------------------------------------------------------------------------------------------------ context discovery

def test_planted_context_dependent_pattern_is_discovered_with_both_sides_learned():
    obs = gen("P", 1500, 0, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0)
    m = CX.ContextModel(cfg())
    m.add_many(obs)
    rules = m.discover("P", FAR, seed=1)
    assert rules and rules[0].role == "CONTEXT" and rules[0].confirmed and rules[0].epistemic == Epistemic.CONDITIONAL
    top = rules[0]
    assert top.spec.conditions[0].path == "regime.label" and "bull_calm" in top.spec.conditions[0].allowed
    assert 0.02 < top.stats_in.mean < 0.04 and abs(top.stats_out.mean) < 0.006          # P(o|p,ctx) AND P(o|p,NOT ctx)
    assert top.p_adj <= 0.05 and top.t_holdout > 1.65 and top.validate() == []
    both = m.p_outcome("P", top.spec, FAR)
    assert both["diff"] > 0.02 and both["t"] > 5 and both["in"].reliable and both["out"].reliable
    k = top.to_knowledge_fields()
    assert k["anti_contexts"] == {} and "regime.label" in k["contexts"]
    inn = m.estimate("P", mk(regime__label="bull_calm"), FAR)
    out = m.estimate("P", mk(regime__label="stress"), FAR)
    assert inn.source == "rule" and out.source == "rule" and inn.matched_in and out.matched_out
    assert inn.expected > 0.02 > out.expected + 0.015
    assert "IN context" in inn.explanation and "OUTSIDE context" in out.explanation
    assert m.reports["P"].rules >= 1 and m.reports["P"].mde is not None


def test_pure_noise_pattern_yields_no_confirmed_context_and_says_why():
    m = CX.ContextModel(cfg())
    m.add_many(gen("N", 1500, 3, lambda s, i: 0.0))
    rules = m.discover("N", FAR, seed=1)
    assert not [r for r in rules if r.confirmed]
    rep = m.reports["N"]
    assert rep.candidates > 0 and rep.mde > 0 and (rep.rules == 0 and "cannot show" in rep.reason or rep.rules > 0)
    est = m.estimate("N", mk(), FAR)
    assert est.source == "ladder" and abs(est.expected) < 0.005


def test_anti_context_where_the_pattern_fails_is_found():
    m = CX.ContextModel(cfg())
    m.add_many(gen("A", 1500, 5, lambda s, i: -0.03 if s.get("regime.label") == "stress" else 0.02))
    rules = [r for r in m.discover("A", FAR, seed=2) if r.confirmed]
    anti = [r for r in rules if r.role == "ANTI_CONTEXT"]
    assert anti, "the stress failure must surface as an anti-context"
    r = anti[0]
    assert r.stats_in.mean < r.stats_out.mean - 0.03 and r.validate() == []
    assert set(r.spec.conditions[0].allowed) & {"stress", "crisis"}          # regime.label=stress and vol_regime=crisis are the same days
    assert r.to_knowledge_fields()["contexts"] == {} and r.to_knowledge_fields()["anti_contexts"]
    stress = mk(regime__label="stress", regime__vol_regime="crisis")
    assert m.estimate("A", stress, FAR).expected < m.estimate("A", mk(regime__label="bull_calm"), FAR).expected - 0.03


def test_an_effect_that_existed_only_early_is_not_confirmed_by_the_holdout():
    n = 1500
    m = CX.ContextModel(cfg())
    m.add_many(gen("D", n, 7, lambda s, i: 0.03 if (s.get("regime.label") == "bull_calm" and i < 0.6 * n) else 0.0))
    rules = m.discover("D", FAR, seed=1)
    assert not [r for r in rules if r.confirmed]                       # found in training, dead in the later 30%
    assert all(r.epistemic == Epistemic.HYPOTHESIS for r in rules)


def test_too_little_data_and_empty_model_are_unknown_not_guessed():
    m = CX.ContextModel(cfg())
    assert m.estimate("none", mk(), FAR).unknown == Unknown.UNTESTED and m.estimate("none", mk(), FAR).expected is None
    m.add_many(gen("S", 60, 1, lambda s, i: 0.02))
    assert m.discover("S", FAR) == ()
    assert m.reports["S"].unknown == Unknown.INSUFFICIENT_DATA and "insufficient" in m.reports["S"].reason
    e = m.estimate("S", mk(), FAR)
    assert e.expected is not None and e.pooled_to <= 1                 # only coarse pooling is allowed at n=60
    assert CX.min_detectable_effect(1, 1, 0.02) == float("inf")
    assert CX.min_detectable_effect(200, 200, .02) < CX.min_detectable_effect(50, 50, .02)
    assert m.p_outcome("zzz", CX.ContextSpec(), FAR)["diff"] is None


def test_observation_not_matured_before_now_fails_closed():
    m = CX.ContextModel(cfg())
    m.add(CX.Obs("F", mk(), 0.01, dt.date(2019, 1, 10)))
    with pytest.raises(FirewallBreach):
        m.estimate("F", mk(), dt.date(2019, 1, 10))                    # matures ON now: not yet known
    assert m.estimate("F", mk(), dt.date(2019, 1, 11)).n_support == 1
    with pytest.raises(ValueError):
        CX.Obs("F", mk(), float("nan"), dt.date(2019, 1, 1))
    with pytest.raises(ValueError):
        m.add(CX.Obs("F", ST.Situation(mk().blocks[:2]), 0.0, dt.date(2019, 1, 1)))


def test_hierarchical_pooling_stops_at_the_first_tiny_bucket():
    m = CX.ContextModel(cfg())
    common = gen("H", 900, 2, lambda s, i: 0.01)
    m.add_many(common)
    rare = mk(regime__label="stress", trend__state="flat", volatility__vol_rank=0.05, liquidity__dv_rank=0.05,
              sector__family="healthcare", breadth__state="washed_out")
    est = m.estimate("H", rare, FAR)
    assert est.pooled_to < len(ST.POOLING_LADDER) - 1 and "pooling stopped" in est.explanation
    audit = m.tiny_bucket_audit("H", FAR)
    assert audit[0]["tiny"] == 0 and audit[len(ST.POOLING_LADDER) - 1]["buckets"] >= 1
    deep = m.estimate("H", common[0].situation, FAR)
    assert all(s.n >= m.cfg.min_n for s in deep.trail)
    for s in est.trail[1:]:
        assert s.n >= m.cfg.min_n


def test_shrinkage_beats_raw_buckets_out_of_sample():
    m = CX.ContextModel(cfg())
    m.add_many(gen("C", 900, 4, lambda s, i: 0.02 if s.get("regime.label") == "bull_calm" else 0.0, sd=0.03))
    cv = m.pooling_cv("C", FAR, folds=4)
    assert cv["pooled_mse"] <= cv["raw_mse"] + 1e-12
    small = CX.ContextModel(cfg())
    assert np.isnan(small.pooling_cv("nothing", FAR)["pooled_mse"])


def test_additive_ridge_and_ladder_comparison_on_two_weak_effects():
    eff = lambda s, i: (0.015 if s.get("regime.label") == "bull_calm" else 0.0) + (0.015 if s.get("trend.state") == "up" else 0.0)
    m = CX.ContextModel(cfg())
    m.add_many(gen("R", 1200, 6, eff, sd=0.03))
    cmp = CX.compare_estimators(m, "R", FAR, folds=4)
    assert cmp["ridge"] < cmp["flat"] and cmp["ladder"] < cmp["flat"]
    rc = CX.RidgeContext(min_n=30).fit(m._obs["R"], FAR)
    top = {(p, l) for p, l, _ in rc.top_effects(4)}
    assert ("regime.label", "bull_calm") in top or ("trend.state", "up") in top
    assert rc.predict(mk(regime__label="bull_calm", trend__state="up")) > rc.predict(mk(regime__label="stress", trend__state="down"))
    assert np.isnan(CX.compare_estimators(CX.ContextModel(cfg()), "x", FAR)["flat"])
    with pytest.raises(FirewallBreach):
        CX.RidgeContext().fit([CX.Obs("x", mk(), 0.0, dt.date(2020, 1, 1))], dt.date(2019, 1, 1))


def test_context_skill_is_positive_with_structure_and_not_proven_without():
    good = CX.ContextModel(cfg())
    good.add_many(gen("G", 1000, 8, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else -0.01))
    assert CX.context_skill(good, "G", FAR)["skill"] > 0.05
    noise = CX.ContextModel(cfg())
    noise.add_many(gen("Z", 1000, 9, lambda s, i: 0.0))
    sk = CX.context_skill(noise, "Z", FAR)
    assert sk["status"] in ("FAILED", "UNPROVEN") and sk["skill"] < 0.05         # context adds nothing to noise
    assert CX.context_skill(CX.ContextModel(cfg()), "none", FAR)["status"] == "INSUFFICIENT_EVIDENCE"


def test_boundary_sweep_finds_the_planted_change_point():
    m = CX.ContextModel(cfg())
    m.add_many(gen("B", 1000, 10, lambda s, i: 0.03 if s.get("volatility.vol_rank") < 0.5 else -0.03))
    b = CX.sweep_boundary(m, "B", "volatility.vol_rank", FAR, seed=1)
    assert b.flips_sign and b.p_adj < 0.05 and b.mean_below > 0.02 and b.mean_above < -0.02
    assert set(b.below) <= {"b0", "b1", "b2"} and "b4" in b.above
    assert CX.sweep_boundary(m, "B", "regime.label", FAR) is None             # unordered dimension: no cut
    assert CX.sweep_boundary(m, "missing", "volatility.vol_rank", FAR) is None


def test_pattern_interaction_synergy_is_detected():
    m = CX.ContextModel(cfg())
    m.add_many(gen("X", 800, 11, lambda s, i: 0.03 if "Y" in s.pattern_ids else 0.0, extra="Y"))
    r = CX.interaction_effect(m, "X", "Y", FAR)
    assert r["verdict"] == "SYNERGY" and r["diff"] > 0.02
    rules = [r for r in m.discover("X", FAR) if r.confirmed]
    assert rules and rules[0].spec.conditions[0].path == "pattern.has:Y"
    assert CX.interaction_effect(m, "X", "never", FAR)["verdict"] == Unknown.INSUFFICIENT_DATA
    assert CX.interaction_effect(m, "nope", "Y", FAR)["verdict"] == Unknown.UNTESTED


def test_temporal_stability_separates_a_lasting_rule_from_an_era_artifact():
    n = 1600
    stable = CX.ContextModel(cfg())
    stable.add_many(gen("T", n, 12, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0))
    r = [x for x in stable.discover("T", FAR) if x.confirmed][0]
    assert CX.temporal_stability(stable, r, FAR)["verdict"] == "STABLE"
    era = CX.ContextModel(cfg())
    era.add_many(gen("E", n, 13, lambda s, i: 0.05 if (s.get("regime.label") == "bull_calm" and i < 0.55 * n) else
                     (-0.01 if s.get("regime.label") == "bull_calm" else 0.0)))
    rr = era.discover("E", FAR)
    pick = rr[0] if rr else r
    res = CX.temporal_stability(era, dataclasses.replace(pick, pattern_id="E"), FAR)
    assert res["verdict"] in ("UNSTABLE", Unknown.INSUFFICIENT_DATA) or (res["agree_share"] or 1) < 1.0
    assert CX.temporal_stability(CX.ContextModel(cfg()), r, FAR)["verdict"] == Unknown.UNTESTED


def test_rulebook_never_deletes_and_detects_conflicts():
    m = CX.ContextModel(cfg())
    m.add_many(gen("K", 1500, 14, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0))
    good = [x for x in m.discover("K", FAR) if x.confirmed][0]
    book = CX.RuleBook()
    assert book.add(good, FAR) and not book.add(good, FAR) and len(book) == 1
    book.retire(good.rule_id, FAR, "regime changed")
    assert book.is_retired(good.rule_id) and book.active() == [] and book.get(good.rule_id) == good
    book.reactivate(good.rule_id, FAR, "regime returned")
    assert [r.rule_id for r in book.active("K")] == [good.rule_id]
    assert [h["event"] for h in book.history()] == ["added", "retired", "reactivated"]
    with pytest.raises(ValueError):
        book.retire(good.rule_id, FAR, "")
    with pytest.raises(KeyError):
        book.reactivate("CR-none", FAR, "x")
    opposite = dataclasses.replace(good, role="ANTI_CONTEXT", diff=-good.diff)
    book.add(opposite, FAR)
    assert book.conflicts("K")
    match = book.matching("K", mk(regime__label="bull_calm"))
    assert good.rule_id in match["in"]
    bad = dataclasses.replace(good, role="ANTI_CONTEXT")
    with pytest.raises(ValueError):
        CX.RuleBook().add(bad, FAR)


def test_revalidation_retires_a_rule_that_later_data_contradict():
    m = CX.ContextModel(cfg())
    m.add_many(gen("V", 1500, 15, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0))
    good = [x for x in m.discover("V", FAR) if x.confirmed][0]
    book = CX.RuleBook()
    book.add(good, FAR)
    later_ok = gen("V", 300, 16, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0, start=dt.date(2021, 1, 4))
    later_bad = gen("V", 300, 17, lambda s, i: -0.03 if s.get("regime.label") == "bull_calm" else 0.0, start=dt.date(2021, 1, 4))
    assert m.drift_check(good, later_ok, FAR)["verdict"] == Epistemic.CONDITIONAL
    bad = m.drift_check(good, later_bad, FAR)
    assert bad["verdict"] == Epistemic.CONTRADICTED and bad["t"] < -1.65
    assert m.drift_check(good, later_ok[:5], FAR)["verdict"] == Unknown.INSUFFICIENT_DATA
    res = book.revalidate(m, later_bad, FAR)
    assert book.is_retired(good.rule_id) and res[good.rule_id]["verdict"] == Epistemic.CONTRADICTED
    with pytest.raises(FirewallBreach):
        m.drift_check(good, [CX.Obs("V", mk(), 0.0, dt.date(2031, 1, 1))], FAR)


def test_discovery_is_deterministic_and_state_roundtrips():
    obs = gen("Q", 900, 18, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0)
    a, b = CX.ContextModel(cfg()), CX.ContextModel(cfg())
    a.add_many(obs)
    b.add_many(obs)
    ra, rb = a.discover("Q", FAR, seed=3), b.discover("Q", FAR, seed=3)
    assert [r.rule_id for r in ra] == [r.rule_id for r in rb] and [r.p_adj for r in ra] == [r.p_adj for r in rb]
    c = CX.model_from_state(CX.model_state(a))
    assert c.fingerprint() == a.fingerprint()
    assert [r.rule_id for r in c.discover("Q", FAR, seed=3)] == [r.rule_id for r in ra]
    assert a.table(FAR)[0]["pattern"] == "Q"


def test_config_validation():
    assert CX.ContextConfig(min_n=3).validate() and CX.ContextConfig(holdout_frac=0.9).validate()
    with pytest.raises(ValueError):
        CX.ContextModel(CX.ContextConfig(n_perm=5))


# ------------------------------------------------------------------------------------------------ retrieval

PROV = Provenance(created_real="2019-01-01T00:00:00", learned_at="2018-06-01", code_hash="abc123", outcomes_seen_through="2018-06-01")
RNOW = dt.date(2019, 6, 3)


@dataclasses.dataclass(frozen=True)
class KItem:
    knowledge_id: str
    version: int = 1
    epistemic: Epistemic = Epistemic.SUPPORTED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    promotion: Promotion = Promotion.CHAMPION
    confidence: Confidence = Confidence(truth=.8, usefulness=.7, current_reliability=.7, context=.7, transfer=.6, failure_risk=.2)
    provenance: Provenance = PROV
    contexts: dict = dataclasses.field(default_factory=dict)
    anti_contexts: dict = dataclasses.field(default_factory=dict)
    decision_effect: tuple = (DecisionEffect.RANKING,)
    temporal_class: TemporalClass = TemporalClass.SLOW_DECAY
    n_support: int | None = None


def support(index, kid, n, seed, regime, edge=0.02, start=dt.date(2017, 1, 2), **over):
    rng = np.random.default_rng(seed)
    for i in range(n):
        index.add_support(kid, mk(regime__label=regime, regime__vol_regime=VOLREG[regime], volatility__vol_rank=float(rng.uniform(.3, .7)),
                                  market__vix=float(rng.uniform(13, 18)), **over),
                          start + dt.timedelta(days=i), float(edge + rng.normal(0, .01)), ref=f"{kid}-{i}")


def build_index():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("K1", contexts={"regime.label": {"in": ["bull_calm"]}}))
    ix.add_item(KItem("K2", contexts={"regime.label": {"in": ["stress"]}}))
    ix.add_item(KItem("K3"))
    support(ix, "K1", 40, 1, "bull_calm")
    support(ix, "K2", 40, 2, "stress")
    support(ix, "K3", 40, 3, "bull_calm")
    return ix


def test_planted_context_item_is_retrieved_only_in_its_context():
    ix = build_index()
    rt = RT.Retriever(ix)
    bull = rt.retrieve(mk(regime__label="bull_calm", volatility__vol_rank=0.5), RNOW)
    assert bull.ids()[0] == "K1" and "K2" not in bull.ids()
    assert any(r.knowledge_id == "K2" and "context does not hold" in r.reasons[0] for r in bull.rejected)
    stress = rt.retrieve(mk(regime__label="stress", regime__vol_regime="crisis", market__vix=40.0, volatility__vol_rank=0.5), RNOW)
    assert "K1" not in stress.ids() and any(r.knowledge_id == "K1" for r in stress.rejected)
    text = bull.items[0].explain()
    assert text.startswith("Retrieved K1 because:")
    for phrase in ("situation similarity =", "context match =", "current reliability =", "transfer evidence =", "failure risk ="):
        assert phrase in text
    assert bull.items[0].factor("context_match") == 1.0
    assert next(i for i in bull.items if i.knowledge_id == "K3").factor("context_match") == 0.6      # unconditional: applies, says little
    assert bull.influence and bull.skill_status == "UNMONITORED"


def test_rules_learned_by_the_context_model_drive_retrieval_end_to_end():
    m = CX.ContextModel(cfg())
    m.add_many(gen("P", 1500, 20, lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0))
    rule = [r for r in m.discover("P", FAR, seed=1) if r.confirmed][0]
    fields = rule.to_knowledge_fields()
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("KP", **fields))
    for o in m._obs["P"]:
        if rule.spec.matches(o.situation) is True:
            ix.add_support("KP", o.situation, o.matured, o.outcome)
    rt = RT.Retriever(ix)
    hits = misses = 0
    for lab in REGS:
        got = rt.retrieve(mk(regime__label=lab, regime__vol_regime=VOLREG[lab]), FAR)
        in_ctx = "bull_calm" in rule.spec.conditions[0].allowed and lab in rule.spec.conditions[0].allowed
        hits += int(in_ctx and "KP" in got.ids())
        misses += int(not in_ctx and "KP" not in got.ids())
    assert hits >= 1 and hits + misses == len(REGS)                    # retrieved in its context and ONLY there


def test_future_knowledge_and_future_support_fail_closed_or_are_excluded():
    ix = build_index()
    future = KItem("KF", provenance=dataclasses.replace(PROV, learned_at="2019-07-01", outcomes_seen_through="2019-07-01"))
    ix.add_item(future)
    with pytest.raises(FirewallBreach):
        RT.Retriever(ix).retrieve(mk(), RNOW)
    soft = RT.Retriever(ix, config=RT.RetrievalConfig(on_future="exclude")).retrieve(mk(), RNOW)
    assert "KF" not in soft.ids() and any(r.knowledge_id == "KF" and "future" in r.reasons[0] for r in soft.rejected)
    ix2 = build_index()
    ix2.add_support("K3", mk(), dt.date(2019, 6, 3), 0.1)              # outcome matures ON now
    with pytest.raises(FirewallBreach):
        RT.Retriever(ix2).retrieve(mk(), RNOW)
    assert "K3" not in RT.Retriever(ix2, config=RT.RetrievalConfig(on_future="exclude")).retrieve(mk(), RNOW).ids()


def test_unmeasured_factors_are_flagged_and_lower_the_score():
    ix = RT.KnowledgeIndex()
    known = KItem("A")
    unknown_conf = dataclasses.replace(known.confidence, current_reliability=None, transfer=None, failure_risk=None)
    ix.add_item(known)
    ix.add_item(KItem("B", confidence=unknown_conf))
    support(ix, "A", 40, 1, "bull_calm")
    support(ix, "B", 40, 1, "bull_calm", edge=0.0)
    r = RT.Retriever(ix).retrieve(mk(), RNOW)
    a, b = (next(i for i in r.items if i.knowledge_id == k) for k in "AB")
    assert {"current_reliability", "failure_safety"} <= set(b.untested) and "current_reliability" not in a.untested
    assert b.score < a.score and "UNTESTED" in b.explain()
    assert b.factor("current_reliability") is None


def test_gates_reject_retired_unproven_and_research_only_items():
    ix = RT.KnowledgeIndex()
    for kid, kw in {"R1": dict(epistemic=Epistemic.RETIRED), "R2": dict(promotion=Promotion.SHADOW),
                    "R3": dict(decision_effect=(DecisionEffect.NONE,)), "R4": dict(epistemic=Epistemic.CONTRADICTED), "OK": {}}.items():
        ix.add_item(KItem(kid, **kw))
        support(ix, kid, 30, 5, "bull_calm")
    r = RT.Retriever(ix).retrieve(mk(), RNOW)
    assert r.ids() == ("OK",)
    why = {x.knowledge_id: x.reasons[0] for x in r.rejected}
    assert "RETIRED" in why["R1"] and "SHADOW" in why["R2"] and "research knowledge only" in why["R3"] and "CONTRADICTED" in why["R4"]
    research = RT.Retriever(ix, config=RT.RetrievalConfig(allow_blocked=True, allowed_promotions=tuple(Promotion))).retrieve(mk(), RNOW, k=10)
    assert {"R1", "R2", "R4"} <= set(research.ids()) and any("BLOCKED" in f for i in research.items for f in i.flags)
    only = RT.Retriever(ix).retrieve(mk(), RNOW, decision_effect=DecisionEffect.STOP)
    assert only.ids() == () and only.unknown is not None


def test_older_version_never_replaces_a_newer_one():
    ix = RT.KnowledgeIndex()
    assert ix.add_item(KItem("V", version=2)) and not ix.add_item(KItem("V", version=1)) and ix.get("V").version == 2
    assert ix.add_item(KItem("V", version=3))
    with pytest.raises(TypeError):
        ix.add_item(object())
    with pytest.raises(KeyError):
        ix.add_support("nope", mk(), dt.date(2018, 1, 1))


def test_redundant_twins_are_discounted_and_contradicted_items_penalised():
    ix = RT.KnowledgeIndex()
    for kid in ("T1", "T2", "T3"):
        ix.add_item(KItem(kid))
        support(ix, kid, 40, 7, "bull_calm")
    ix.set_redundancy("T1", "T2", 0.95)
    r = RT.Retriever(ix).retrieve(mk(), RNOW)
    by = {i.knowledge_id: i for i in r.items}
    assert by["T2"].score < by["T3"].score                                              # the twin drops below an independent item
    assert by["T2"].factor("redundancy_penalty") > 0.9 and "redundant with T1" in by["T2"].explain()
    ix.set_contradiction("T3", "T1", 0.9)
    ix._items["T1"] = dataclasses.replace(ix.get("T1"), version=2, confidence=dataclasses.replace(ix.get("T1").confidence, current_reliability=0.95))
    r2 = RT.Retriever(ix).retrieve(mk(), RNOW)
    by2 = {i.knowledge_id: i for i in r2.items}
    assert by2["T3"].factor("contradiction_penalty") > 0.3 and "contradicted by T1" in by2["T3"].explain()
    with pytest.raises(ValueError):
        ix.set_redundancy("T1", "T2", 1.5)
    with pytest.raises(ValueError):
        ix.set_contradiction("T1", "T2", -0.1)


def test_retrieval_is_deterministic_and_scrambling_invariant():
    ix = build_index()
    rt = RT.Retriever(ix)
    q = mk(regime__label="bull_calm")
    assert rt.retrieve(q, RNOW).retrieval_id == rt.retrieve(q, RNOW).retrieval_id
    X = panel(seed=8)
    Y, mapping = ST.scramble_tickers(ST.shift_years(X, 6), seed=3)
    far = pd.Timestamp("2031-01-01")
    a = ST.SituationBuilder().build_panel(X, far, sic=SIC)
    b = ST.SituationBuilder().build_panel(Y.sort_index(), far, sic={mapping[t]: c for t, c in SIC.items()})
    inv = {v: k for k, v in mapping.items()}
    checked = 0
    for (d, t2), sb in list(b.items())[:40]:
        sa = a[(d - pd.DateOffset(years=6), inv[t2])]
        ra, rb = rt.retrieve(sa, RNOW), rt.retrieve(sb, RNOW)
        assert ra.ids() == rb.ids() and [i.score for i in ra.items] == [i.score for i in rb.items]
        checked += 1
    assert checked == 40


def test_ablation_stability_and_factor_report():
    ix = build_index()
    rt = RT.Retriever(ix)
    q = mk(regime__label="bull_calm")
    ab = rt.ablate(q, RNOW, "current_reliability")
    assert ab["factor"] == "current_reliability" and 0.0 <= ab["overlap"] <= 1.0
    with pytest.raises(ValueError):
        rt.ablate(q, RNOW, "contradiction_penalty")
    assert rt.stability(q, RNOW, seed=1, trials=6) >= 0.9
    rep = rt.factor_report([q, mk(regime__label="stress", regime__vol_regime="crisis", market__vix=40.0)], RNOW)
    assert set(rep) == set(RT.FACTORS) and rep["situation_similarity"]["n"] > 0
    assert RT.perturb(q, np.random.default_rng(0), 0.05).validate() == []


def test_why_not_explains_rejection_and_rank():
    ix = build_index()
    rt = RT.Retriever(ix)
    q = mk(regime__label="bull_calm")
    assert "rejected" in RT.why_not(rt, "K2", q, RNOW)
    assert "top item" in RT.why_not(rt, "K1", q, RNOW) or "ranks" in RT.why_not(rt, "K1", q, RNOW)
    assert "not in the index" in RT.why_not(rt, "K404", q, RNOW)


def test_retrieval_with_no_skill_reports_it_and_abstains():
    ix = build_index()
    q = mk(regime__label="bull_calm")
    bad = RT.SkillMonitor(min_n=40)
    good = RT.SkillMonitor(min_n=40)
    rng = np.random.default_rng(0)
    for i in range(80):
        e = float(rng.normal(0, .02))
        d0 = dt.date(2018, 1, 1) + dt.timedelta(days=i)
        bad.predict(f"r{i}", d0, e)
        bad.resolve(f"r{i}", d0 + dt.timedelta(days=5), -e + float(rng.normal(0, .002)))          # walk-forward skill < 0
        good.predict(f"r{i}", d0, e)
        good.resolve(f"r{i}", d0 + dt.timedelta(days=5), e + float(rng.normal(0, .005)))
    assert bad.status(RNOW)["status"] == "FAILED" and good.status(RNOW)["status"] == "PROVEN" and good.allows(RNOW)
    r_bad = RT.Retriever(ix, monitor=bad).retrieve(q, RNOW)
    assert r_bad.items and not r_bad.influence and r_bad.action == UnknownAction.ABSTAIN and r_bad.skill_status == "FAILED"
    assert r_bad.explain().startswith("WITHHELD")
    r_good = RT.Retriever(ix, monitor=good).retrieve(q, RNOW)
    assert r_good.influence and r_good.skill_status == "PROVEN"
    fresh = RT.Retriever(ix, monitor=RT.SkillMonitor()).retrieve(q, RNOW)
    assert not fresh.influence and fresh.skill_status == "INSUFFICIENT_EVIDENCE"
    early = bad.status(dt.date(2018, 1, 20))
    assert early["n"] == 14 and early["status"] == "INSUFFICIENT_EVIDENCE"      # outcomes not yet matured at that `now` are not read
    m = RT.SkillMonitor()
    m.predict("x", dt.date(2018, 1, 5), 0.01)
    with pytest.raises(FirewallBreach):
        m.resolve("x", dt.date(2018, 1, 5), 0.02)
    with pytest.raises(KeyError):
        m.resolve("unknown", dt.date(2018, 1, 9), 0.0)
    with pytest.raises(ValueError):
        RT.SkillMonitor(min_n=2)


def test_novel_situation_makes_retrieval_abstain():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("N1"))
    support(ix, "N1", 60, 4, "bull_calm")
    rt = RT.Retriever(ix, config=RT.RetrievalConfig(novelty_check=True, min_similarity=0.0))
    near = rt.retrieve(mk(regime__label="bull_calm", volatility__vol_rank=0.5, market__vix=15.0), RNOW)
    assert near.novel is False and near.influence
    far = mk(regime__label="stress", regime__vol_regime="crisis", market__vix=60.0, market__spy_ma200=-.3, volatility__vol_rank=.99,
             trend__r20=-.3, trend__state="down", breadth__breadth=.1, breadth__state="washed_out", liquidity__dv_rank=.05, liquidity__state="thin")
    out = rt.retrieve(far, RNOW)
    assert out.novel is True and not out.influence and out.action == UnknownAction.ABSTAIN and "novel situation" in out.withheld_reason


def test_transfer_evidence_separates_broad_from_narrow_knowledge():
    rng = np.random.default_rng(0)
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("BROAD", confidence=Confidence(truth=.8, current_reliability=.7, failure_risk=.2)))
    ix.add_item(KItem("NARROW", confidence=Confidence(truth=.8, current_reliability=.7, failure_risk=.2)))
    for i in range(60):
        peripheral = i % 3 == 0
        extra = dict(regime__label="bear_volatile", regime__vol_regime="high", market__vix=30.0, volatility__vol_rank=.9) if peripheral else {}
        s = mk(**extra)
        d = dt.date(2017, 1, 1) + dt.timedelta(days=i)
        ix.add_support("BROAD", s, d, 0.02 + rng.normal(0, .002))
        ix.add_support("NARROW", s, d, (0.0 if peripheral else 0.02) + rng.normal(0, .002))
    b, n = RT.transfer_evidence(ix, "BROAD", RNOW), RT.transfer_evidence(ix, "NARROW", RNOW)
    assert b["verdict"] == "TRANSFERS" and n["verdict"] in ("NARROW", "FAILS_TO_TRANSFER") and n["score"] < 0.5 < b["score"]
    assert RT.transfer_evidence(ix, "BROAD", RNOW, min_cases=1000)["verdict"] == "UNTESTED"
    r = RT.Retriever(ix, config=RT.RetrievalConfig(min_similarity=0.0)).retrieve(mk(), RNOW)
    tb = next(i for i in r.items if i.knowledge_id == "BROAD").factor("transfer_confidence")
    tn = next(i for i in r.items if i.knowledge_id == "NARROW").factor("transfer_confidence")
    assert tb > tn


def test_support_expectation_follows_the_similar_cases():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("E"))
    for i in range(30):
        ix.add_support("E", mk(volatility__vol_rank=0.15 + 0.001 * i), dt.date(2017, 1, 1) + dt.timedelta(days=i), 0.05)
        ix.add_support("E", mk(volatility__vol_rank=0.85 - 0.001 * i, regime__label="bear_volatile", regime__vol_regime="high"),
                       dt.date(2017, 3, 1) + dt.timedelta(days=i), -0.05)
    lo = RT.support_expectation(ix, "E", mk(volatility__vol_rank=0.15), RNOW)
    hi = RT.support_expectation(ix, "E", mk(volatility__vol_rank=0.85, regime__label="bear_volatile", regime__vol_regime="high"), RNOW)
    assert lo["mean"] > 0.02 > -0.02 > hi["mean"] and lo["n_eff"] > 1 and lo["q10"] <= lo["q90"]
    assert RT.support_expectation(ix, "E", mk(), RNOW)["n"] > 0
    empty = RT.KnowledgeIndex()
    empty.add_item(KItem("X"))
    assert RT.support_expectation(empty, "X", mk(), RNOW)["mean"] is None


def test_combine_discounts_duplicates_and_reports_disagreement():
    ix = RT.KnowledgeIndex()
    for kid, edge in (("C1", 0.03), ("C2", 0.03), ("C3", -0.02)):
        ix.add_item(KItem(kid))
        support(ix, kid, 40, 9, "bull_calm", edge=edge)
    ix.set_redundancy("C1", "C2", 0.9)
    r = RT.Retriever(ix).retrieve(mk(), RNOW)
    plain, disc = RT.combine(r), RT.combine(r, ix)
    assert disc["n_votes"] == 3 and plain["expected"] is not None
    assert disc["expected"] < plain["expected"]                         # the duplicate no longer votes twice for the positive side
    assert 0.5 <= disc["majority_share"] <= 1.0 and disc["disagreement"] > 0
    empty = RT.combine(RT.Retriever(RT.KnowledgeIndex()).retrieve(mk(), RNOW))
    assert empty["expected"] is None and empty["n_votes"] == 0


def test_retrieval_weight_fitting_finds_the_factor_that_predicts_outcomes():
    rng = np.random.default_rng(0)
    A = rng.uniform(0.1, 0.9, (400, 8))
    y = 0.05 * A[:, RT.POSITIVE_FACTORS.index("current_reliability")] + rng.normal(0, 0.003, 400)
    w, diag = RT.fit_retrieval_weights(A, y, ridge=1.0)
    d, prior = w.normalised(), RT.RetrievalWeights().normalised()
    assert d["current_reliability"] > prior["current_reliability"] and diag["r2_after"] >= diag["r2_before"] and diag["changed"]
    same, diag2 = RT.fit_retrieval_weights(A[:20], y[:20])
    assert same == RT.RetrievalWeights() and not diag2["changed"]
    with pytest.raises(ValueError):
        RT.fit_retrieval_weights(A[:, :3], y)
    assert RT.RetrievalWeights(values=(("x", 1.0),)).validate()


def test_retrieval_log_replay_detects_a_changed_retriever(tmp_path):
    ix = build_index()
    q = mk(regime__label="bull_calm")
    rt = RT.Retriever(ix)
    log = RT.RetrievalLog(tmp_path / "logs" / "r.jsonl")
    rec = log.append(rt.retrieve(q, RNOW), RNOW)
    log.append(rt.retrieve(mk(regime__label="stress", regime__vol_regime="crisis", market__vix=40.0), RNOW), RNOW)
    assert len(log) == 2 and log.read()[0]["retrieval_id"] == rec["retrieval_id"]
    assert log.replay(log.read()[0], rt, q)["match"]
    other = RT.Retriever(ix, weights=RT.RetrievalWeights().without("failure_safety"))
    res = log.replay(log.read()[0], other, q)
    assert not res["match"] and res["logged"] != res["now_id"]
    assert len(RT.RetrievalLog()) == 0


def test_score_and_rank_quality_and_helpers():
    ix = build_index()
    rt = RT.Retriever(ix)
    r = rt.retrieve(mk(regime__label="bull_calm"), RNOW)
    s = RT.score_retrieval(r, {r.items[0].knowledge_id: 0.02})
    assert s["n"] == 1 and s["hit_rate"] == 1.0
    assert RT.score_retrieval(r, {})["hit_rate"] is None
    assert np.isnan(RT.rank_quality([r], [{}])["spearman"])
    M = RT.overlap_matrix(RT.retrieve_many(rt, [mk(regime__label="bull_calm")] * 2 + [mk(regime__label="stress", regime__vol_regime="crisis", market__vix=40.0)], RNOW))
    assert M[0, 1] == 1.0 and M[0, 2] < 1.0


def test_index_support_export_import_roundtrip_and_empty_index():
    ix = build_index()
    ix.set_redundancy("K1", "K3", 0.5)
    st = RT.export_support(ix)
    fresh = RT.KnowledgeIndex()
    for it in ix.items():
        fresh.add_item(it)
    counts = RT.import_support(fresh, st)
    assert counts["cases"] == 120 and len(fresh.support("K1")) == 40 and fresh.redundancies("K1")["K3"] == 0.5
    q = mk(regime__label="bull_calm")
    assert RT.Retriever(fresh).retrieve(q, RNOW).retrieval_id == RT.Retriever(ix).retrieve(q, RNOW).retrieval_id
    partial = RT.KnowledgeIndex()
    partial.add_item(ix.get("K1"))
    assert RT.import_support(partial, st)["skipped_ids"] == 2
    none = RT.Retriever(RT.KnowledgeIndex()).retrieve(q, RNOW)
    assert none.items == () and none.unknown == Unknown.UNTESTED and none.action == UnknownAction.COLLECT_DATA
    with pytest.raises(ValueError):
        RT.Retriever(ix, config=RT.RetrievalConfig(top_k=0))


def test_temporal_relevance_by_class():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("RB", temporal_class=TemporalClass.REGIME_BOUND, contexts={"regime.label": {"in": ["bull_calm"]}}))
    ix.add_item(KItem("EV", temporal_class=TemporalClass.EVENT_BOUND))
    ix.add_item(KItem("FD", temporal_class=TemporalClass.FAST_DECAY))
    ix.add_item(KItem("PE", temporal_class=TemporalClass.PERSISTENT))
    for kid in ("RB", "EV", "FD", "PE"):
        support(ix, kid, 30, 6, "bull_calm")
    r = RT.Retriever(ix).retrieve(mk(shock__kind="none"), RNOW, k=4)
    tr = {i.knowledge_id: i.factor("temporal_relevance") for i in r.items}
    assert tr["RB"] == 1.0 and tr["EV"] == 0.2 and tr["FD"] < tr["PE"]
    ev = RT.Retriever(ix).retrieve(mk(shock__kind="earnings"), RNOW, k=4)
    assert next(i for i in ev.items if i.knowledge_id == "EV").factor("temporal_relevance") == 1.0


# ------------------------------------------------------------------------------------------------ context extras

def _fitted(pid="P", n=1500, seed=0, eff=None, **kw):
    eff = eff or (lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.0)
    m = CX.ContextModel(cfg(**kw))
    m.add_many(gen(pid, n, seed, eff))
    return m


def test_outcome_distribution_shows_a_tail_change_the_mean_alone_would_hide():
    rng = np.random.default_rng(0)
    m = CX.ContextModel(cfg())
    for o in gen("W", 900, 3, lambda s, i: 0.0, sd=0.01):
        stress = o.situation.get("regime.label") == "stress"
        y = float(rng.choice([-0.12, 0.12]) if (stress and rng.random() < 0.15) else o.outcome)
        m.add(CX.Obs("W", o.situation, y, o.matured))
    spec = CX.ContextSpec((CX.Condition("regime.label", ("stress",)),))
    d = CX.outcome_distribution(m, "W", spec, FAR)
    assert d["in"]["p_loss"] > 5 * max(d["out"]["p_loss"], 0.005) and d["in"]["shortfall"] < d["out"]["shortfall"]
    assert abs(d["in"]["mean"] - d["out"]["mean"]) < 0.02 and d["in"]["quantiles"][0.05] < d["out"]["quantiles"][0.05]
    assert CX.outcome_distribution(CX.ContextModel(cfg()), "none", spec, FAR)["in"]["n"] == 0


def test_dimension_importance_points_at_the_dimension_that_carries_the_effect():
    m = _fitted(n=1200, seed=21)
    imp = CX.dimension_importance(m, "P", FAR, folds=4, seed=1)
    assert imp["regime.label"] == max(imp.values()) and imp["regime.label"] > 0
    assert imp.get("liquidity.dv_rank", 0) < imp["regime.label"] / 3
    assert CX.dimension_importance(CX.ContextModel(cfg()), "none", FAR) == {}


def test_context_weight_is_neutral_when_the_context_model_has_no_skill():
    good = _fitted(seed=22, eff=lambda s, i: 0.03 if s.get("regime.label") == "bull_calm" else 0.005)
    w_in = CX.context_weight(good, "P", mk(regime__label="bull_calm"), FAR)
    w_out = CX.context_weight(good, "P", mk(regime__label="stress", regime__vol_regime="crisis"), FAR)
    assert w_in["used_context"] and w_in["multiplier"] > 1.0 > w_out["multiplier"] >= 0.0
    noise = _fitted(pid="Z", seed=23, eff=lambda s, i: 0.004)
    w = CX.context_weight(noise, "Z", mk(regime__label="bull_calm"), FAR)
    sk = CX.context_skill(noise, "Z", FAR)
    if sk["status"] in ("FAILED", "INSUFFICIENT_EVIDENCE"):
        assert not w["used_context"] and w["multiplier"] == 1.0 and "unconditional" in w["reason"]
    else:
        assert w["used_context"]
    empty = CX.context_weight(CX.ContextModel(cfg()), "none", mk(), FAR)
    assert empty["multiplier"] == 1.0 and not empty["used_context"]


def test_winsorising_stops_one_outlier_from_creating_a_context():
    base = gen("O", 900, 24, lambda s, i: 0.0, sd=0.01)
    obs = [CX.Obs("O", o.situation, (-2.0 if (i % 150 == 0 and o.situation.get("regime.label") == "stress") else o.outcome), o.matured)
           for i, o in enumerate(base)]
    plain, robust = CX.ContextModel(cfg()), CX.ContextModel(cfg(winsor=0.01))
    plain.add_many(obs)
    robust.add_many(obs)
    assert not [r for r in robust.discover("O", FAR, seed=1) if r.confirmed]
    stress = mk(regime__label="stress")
    assert robust.estimate("O", stress, FAR).se < plain.estimate("O", stress, FAR).se        # the -200% prints no longer dominate the spread
    with pytest.raises(ValueError):
        CX.ContextModel(CX.ContextConfig(winsor=0.5))


def test_bootstrap_interval_refinement_and_summary():
    m = _fitted(n=1600, seed=25, eff=lambda s, i: 0.02 * (s.get("regime.label") == "bull_calm") +
                0.03 * (s.get("regime.label") == "bull_calm" and s.get("trend.state") == "up"))
    rules = [r for r in m.discover("P", FAR, seed=1) if r.confirmed]
    assert rules
    base = rules[0]
    bs = CX.bootstrap_rule_effect(m, base, FAR, n_boot=200, seed=1)
    assert bs["lo"] < base.diff < bs["hi"] and bs["sign_share"] > 0.95
    ref = CX.refine_rule(m, base, FAR, seed=1)
    assert ref is None or (ref.spec.complexity == base.spec.complexity + 1 and abs(ref.diff) > abs(base.diff) and ref.confirmed)
    assert CX.bootstrap_rule_effect(CX.ContextModel(cfg()), base, FAR)["lo"] is None
    summ = CX.summarize_pattern(m, "P", FAR)
    assert summ["context_dependent"] and summ["contexts"] and summ["status"] == Epistemic.CONDITIONAL and summ["n"] == 1600
    assert CX.summarize_pattern(CX.ContextModel(cfg()), "none", FAR)["status"] == Unknown.UNTESTED
    text = CX.explain_pattern(m, "P", FAR)
    assert "matured observations" in text and "walk-forward context skill" in text and "CONFIRMED" in text
    assert "no observations" in CX.explain_pattern(CX.ContextModel(cfg()), "none", FAR)
    iv = CX.rule_hit_intervals(base)
    assert iv["in"][0] > iv["out"][0] and iv["in"][1] <= iv["in"][0] <= iv["in"][2]


def test_ladder_report_resolve_rules_estimate_many_and_rulebook_state():
    m = _fitted(seed=27)
    lr = CX.ladder_report(m, "P", FAR)
    assert len(lr) == len(ST.POOLING_LADDER) and lr[0]["speaking"] == 1 and lr[0]["buckets"] == 1
    assert CX.ladder_report(CX.ContextModel(cfg()), "none", FAR) == []
    rules = m.discover("P", FAR, seed=1)
    chosen = CX.resolve_rules(list(rules) + list(rules))
    dims = [c.path for r in chosen for c in r.spec.conditions]
    assert len(dims) == len(set(dims)) and chosen[0].rule_id == rules[0].rule_id
    est = CX.estimate_many(m, ["P", "P", "missing"], mk(regime__label="bull_calm"), FAR)
    assert list(est) == ["P", "missing"] and est["missing"].unknown == Unknown.UNTESTED
    book = CX.RuleBook()
    for r in rules:
        book.add(r, FAR)
    book.retire(rules[0].rule_id, FAR, "regime changed")
    back = CX.rulebook_from_state(CX.rulebook_state(book))
    assert [r.rule_id for r in back.active()] == [r.rule_id for r in book.active()] and back.is_retired(rules[0].rule_id)
    assert back.history() == book.history() and "RETIRED (regime changed)" in CX.rulebook_report(back)
    tampered = CX.rulebook_state(book)
    rid = next(iter(tampered["rules"]))
    tampered["rules"][rid]["role"] = "ANTI_CONTEXT" if tampered["rules"][rid]["role"] == "CONTEXT" else "CONTEXT"
    with pytest.raises(ValueError):
        CX.rulebook_from_state(tampered)


# ------------------------------------------------------------------------------------------------ retrieval extras

def test_decide_abstains_by_default_and_lists_every_reason():
    ix = build_index()
    rt = RT.Retriever(ix)
    q = mk(regime__label="bull_calm")
    r = rt.retrieve(q, RNOW)
    d = RT.decide(r, ix)
    assert d["action"] in ("USE", "ABSTAIN") and d["n_items"] == len(r.items)
    strict = RT.decide(r, ix, RT.RetrievalPolicy(min_top_score=0.999, min_expected_n=10_000))
    assert strict["action"] == "ABSTAIN" and strict["expected"] is None and len(strict["reasons"]) >= 2
    none = RT.decide(RT.Retriever(RT.KnowledgeIndex()).retrieve(q, RNOW))
    assert none["action"] == "ABSTAIN" and "nothing retrieved" in none["reasons"][0]
    withheld = RT.decide(RT.Retriever(ix, monitor=RT.SkillMonitor()).retrieve(q, RNOW), ix)
    assert withheld["action"] == "ABSTAIN" and any("may not influence" in x for x in withheld["reasons"])
    with pytest.raises(ValueError):
        RT.decide(r, ix, RT.RetrievalPolicy(min_expected_n=0))
    conflict = RT.KnowledgeIndex()
    for kid, e in (("U", .03), ("D", -.03)):
        conflict.add_item(KItem(kid))
        support(conflict, kid, 40, 3, "bull_calm", edge=e)
    c = RT.decide(RT.Retriever(conflict).retrieve(q, RNOW), conflict, RT.RetrievalPolicy(min_top_score=0.1))
    assert c["action"] == "ABSTAIN" and any("disagree" in x for x in c["reasons"])


def test_rank_change_counterfactual_and_support_audit():
    ix = build_index()
    rt = RT.Retriever(ix)
    q = mk(regime__label="bull_calm")
    before = rt.retrieve(q, RNOW)
    ix._items["K1"] = dataclasses.replace(ix.get("K1"), version=2, confidence=dataclasses.replace(ix.get("K1").confidence, current_reliability=0.1))
    after = rt.retrieve(q, RNOW)
    text = RT.explain_rank_change(before, after)
    assert "K1" in text and "current_reliability" in text
    assert RT.explain_rank_change(before, before) == "no change"
    cf = RT.counterfactual(RT.Retriever(build_index()), q, RNOW, without=["K1"])
    assert cf["changed"] and "K1" not in cf["without"] and "K1" in cf["real"]
    aud = RT.audit_support(ix, RNOW)
    assert aud["K1"]["n"] == 40 and aud["K1"]["distinct_share"] > 0.0
    ix2 = RT.KnowledgeIndex()
    ix2.add_item(KItem("D1"))
    for i in range(30):
        ix2.add_support("D1", mk(), dt.date(2017, 1, 1) + dt.timedelta(days=i), None if i < 3 else 0.01)
    ix2.add_support("D1", mk(), dt.date(2019, 7, 1), 0.0)
    flags = RT.audit_support(ix2, RNOW)["D1"]["flags"]
    assert any("low diversity" in f for f in flags) and any("one situation" in f for f in flags) and any("without an outcome" in f for f in flags)
    assert any("not matured" in f for f in flags)
    ix2.add_item(KItem("E1"))
    assert RT.audit_support(ix2, RNOW)["E1"]["flags"] == ["no support cases"]
    assert "thin support" in RT.index_health(ix2, RNOW) and RT.index_summary(ix)["items"] == 3


def test_calibration_predictiveness_and_concentration():
    ix = build_index()
    rt = RT.Retriever(ix, config=RT.RetrievalConfig(min_similarity=0.0))
    rng = np.random.default_rng(0)
    rets, outs = [], []
    for i in range(60):
        lab = REGS[i % 5]
        r = rt.retrieve(mk(regime__label=lab, regime__vol_regime=VOLREG[lab], market__vix=float(rng.uniform(12, 40))), RNOW)
        rets.append(r)
        outs.append({it.knowledge_id: (it.expected_edge or 0.0) + float(rng.normal(0, .002)) for it in r.items})
    cal = RT.calibration_of_expected(rets, outs)
    assert cal["status"] in ("CALIBRATED", "UNDERCONFIDENT", "OVERCONFIDENT") and cal["slope"] > 0.5
    rev = RT.calibration_of_expected(rets, [{k: -v for k, v in o.items()} for o in outs])
    assert rev["status"] == "MISLEADING"                                              # backwards evidence is named as such
    assert RT.calibration_of_expected(rets[:1], outs[:1])["status"] == "INSUFFICIENT_EVIDENCE"
    pred = RT.factor_predictiveness(rets, outs)
    assert all(-1 <= v <= 1 for v in pred.values())
    conc = RT.concentration_report(rets, ix)
    assert conc["retrievals"] == 60 and 0 < conc["herfindahl"] <= 1 and conc["top_share"] >= 1 / 3
    assert RT.concentration_report([], ix)["never_retrieved"] == list(ix.ids())


def test_retrieve_by_effect_keeps_decisions_separate():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("SEL", decision_effect=(DecisionEffect.SELECTION,)))
    ix.add_item(KItem("STP", decision_effect=(DecisionEffect.STOP,)))
    ix.add_item(KItem("BOTH", decision_effect=(DecisionEffect.SELECTION, DecisionEffect.STOP)))
    for kid in ("SEL", "STP", "BOTH"):
        support(ix, kid, 30, 4, "bull_calm")
    by = RT.retrieve_by_effect(RT.Retriever(ix), mk(), RNOW)
    assert set(by) == {"SELECTION", "STOP"} and set(by["SELECTION"].ids()) == {"SEL", "BOTH"} and set(by["STOP"].ids()) == {"STP", "BOTH"}


def test_prediction_feedback_closes_the_loop_and_stale_code_is_flagged(tmp_path):
    ix = build_index()
    mon = RT.SkillMonitor(min_n=10)
    rt = RT.Retriever(ix, monitor=mon)
    r = rt.retrieve(mk(regime__label="bull_calm"), RNOW)
    e = RT.register_prediction(r, mon, RNOW, ix)
    assert e is not None and mon.pending() == 1
    assert RT.resolve_outcome(r, mon, dt.date(2019, 6, 10), 0.01) and not RT.resolve_outcome(r, mon, dt.date(2019, 6, 10), 0.01)
    empty = RT.Retriever(RT.KnowledgeIndex()).retrieve(mk(), RNOW)
    assert RT.register_prediction(empty, mon, RNOW) is None and mon.pending() == 0
    log = RT.RetrievalLog(tmp_path / "l.jsonl")
    rec = log.append(r, RNOW)
    assert not RT.stale_code(rec) and RT.stale_code({**rec, "code_hash": "someone-elses-code"})
    assert log.replay(rec, RT.Retriever(ix), mk(regime__label="bull_calm"))["stale_code"] is False


def test_walk_forward_replay_withholds_retrieval_until_skill_is_proven():
    ix = build_index()
    rng = np.random.default_rng(1)
    events = []
    for i in range(140):
        d0 = dt.date(2019, 6, 3) + dt.timedelta(days=i)
        s = mk(regime__label="bull_calm", volatility__vol_rank=float(rng.uniform(.3, .7)), market__vix=float(rng.uniform(13, 18)))
        events.append({"situation": s, "made_on": d0, "matured": d0 + dt.timedelta(days=5), "edge": 0.02 + float(rng.normal(0, .003))})
    rep = RT.walk_forward_replay(RT.Retriever(ix, config=RT.RetrievalConfig(min_similarity=0.0)), events, RT.SkillMonitor(min_n=30))
    first, last = rep["rows"][0], rep["rows"][-1]
    assert first["influence"] is False and first["skill_status"] == "INSUFFICIENT_EVIDENCE" and first["action"] == "ABSTAIN"
    assert last["influence"] is True and last["skill_status"] == "PROVEN" and rep["final_skill"]["status"] == "PROVEN"
    assert 0.3 < rep["spoke_share"] < 1.0 and rep["mean_edge_when_used"] > 0
    backwards = [{**e, "edge": -e["edge"]} for e in events]
    bad = RT.walk_forward_replay(RT.Retriever(build_index(), config=RT.RetrievalConfig(min_similarity=0.0)), backwards, RT.SkillMonitor(min_n=30))
    assert bad["final_skill"]["status"] == "FAILED" and bad["spoke_share"] < 0.1              # retrieval with no skill stays silent


def test_stale_items_and_index_merge():
    ix = RT.KnowledgeIndex()
    old = dataclasses.replace(PROV, learned_at="2010-01-01", outcomes_seen_through="2010-01-01")
    ix.add_item(KItem("OLD", provenance=old, temporal_class=TemporalClass.FAST_DECAY))
    ix.add_item(KItem("NEW"))
    st = RT.find_stale(ix, RNOW)
    assert "OLD" in st and "NEW" not in st and any("years old" in w for w in st["OLD"])
    a, b = build_index(), RT.KnowledgeIndex()
    b.add_item(KItem("K4"))
    support(b, "K4", 10, 8, "bull_calm")
    b.add_item(KItem("K1", version=2, contexts={"regime.label": {"in": ["bull_calm"]}}))
    support(b, "K1", 5, 9, "bull_calm", start=dt.date(2017, 6, 1))
    b.set_contradiction("K1", "K3", 0.4)
    a.set_contradiction("K1", "K3", 0.7)
    merged = RT.merge_indexes(a, b)
    assert set(merged.ids()) == {"K1", "K2", "K3", "K4"} and merged.get("K1").version == 2
    assert len(merged.support("K1")) == 5                                       # the superseded version's cases are not carried over
    assert len(merged.support("K3")) == 40 and merged.contradictions("K1")["K3"] == 0.7


def test_false_context_rate_stays_near_alpha_and_catches_a_broken_procedure():
    noise = _fitted(pid="FN", n=900, seed=31, eff=lambda s, i: 0.0)
    rep = CX.false_context_rate(noise, "FN", FAR, n_shuffles=6, seed=1)
    assert rep["shuffles"] == 6 and rep["rate"] <= 0.34 and rep["status"] in ("OK", "HALLUCINATING")
    real = _fitted(pid="FR", n=900, seed=32)
    # a deliberately loose procedure (no family-wise control possible: alpha 0.9) must be flagged as hallucinating
    loose = CX.ContextModel(CX.ContextConfig(n_perm=100, alpha=0.9, holdout_frac=0.0))
    loose.add_many(gen("FL", 900, 33, lambda s, i: 0.0))
    broken = CX.false_context_rate(loose, "FL", FAR, n_shuffles=6, seed=2, count="any")
    assert broken["status"] == "HALLUCINATING" and broken["rate"] > 0.5 > rep["rate"]      # the control can fail
    assert CX.false_context_rate(noise, "FN", FAR, n_shuffles=6, seed=1, count="any")["rate"] <= broken["rate"]
    with pytest.raises(ValueError):
        CX.false_context_rate(noise, "FN", FAR, count="bogus")
    assert CX.false_context_rate(CX.ContextModel(cfg()), "none", FAR)["status"] == Unknown.INSUFFICIENT_DATA
    assert real.n_obs("FR") == 900


def test_rule_stability_and_predict_interval_and_regime_table_and_outcome_bins():
    m = _fitted(n=1200, seed=34)
    st = CX.rule_stability(m, "P", FAR, n_runs=4, seed=1)
    assert st["runs"] == 4 and st["stable"] and all(v >= 0.6 for v in (st["frequency"][k] for k in st["stable"]))
    noise = _fitted(pid="Z", n=900, seed=35, eff=lambda s, i: 0.0)
    assert not CX.rule_stability(noise, "Z", FAR, n_runs=4, seed=1)["stable"]
    assert CX.rule_stability(CX.ContextModel(cfg()), "none", FAR)["status"] == Unknown.INSUFFICIENT_DATA
    iv = CX.predict_interval(m, "P", mk(regime__label="bull_calm"), FAR)
    assert iv["lo"] < iv["expected"] < iv["hi"] and (iv["hi"] - iv["lo"]) > 0.03                     # one trade is noisy even if the mean is known
    assert CX.predict_interval(CX.ContextModel(cfg()), "none", mk(), FAR)["lo"] is None
    tab = CX.regime_table(m, FAR)
    by = {r["bucket"]: r for r in tab}
    assert by["bull_calm"]["mean"] > 0.02 > by["stress"]["mean"] and all(r["reliable"] for r in tab)
    spec = CX.ContextSpec((CX.Condition("regime.label", ("bull_calm",)),))
    bt = CX.outcome_bin_table(m, "P", spec, FAR)
    assert bt["tv"] > 0.3 and abs(sum(bt["in"].values()) - 1.0) < 1e-9 and bt["n_in"] + bt["n_out"] == 1200
    assert CX.outcome_bin_table(CX.ContextModel(cfg()), "none", spec, FAR)["tv"] is None


def test_retrieval_permutation_influence_leverage_and_config_records():
    ix = build_index()
    rt = RT.Retriever(ix, config=RT.RetrievalConfig(min_similarity=0.0))
    rng = np.random.default_rng(3)
    rets, outs = [], []
    for i in range(50):
        lab = REGS[i % 5]
        r = rt.retrieve(mk(regime__label=lab, regime__vol_regime=VOLREG[lab], market__vix=float(rng.uniform(12, 40))), RNOW)
        rets.append(r)
        outs.append({it.knowledge_id: it.score * 0.05 + float(rng.normal(0, .002)) for it in r.items})
    good = RT.rank_quality_p(rets, outs, n_perm=200, seed=1)
    shuffled = RT.rank_quality_p(rets, [dict(zip(o, rng.permutation(list(o.values())))) for o in outs], n_perm=200, seed=1)
    assert good["spearman"] > 0.5 and good["p"] < 0.02 and shuffled["p"] > good["p"]
    assert np.isnan(RT.rank_quality_p([], [])["p"])
    sh = RT.influence_shares(rets[0])
    assert abs(sum(sh.values()) - 1.0) < 1e-5 and "decision weight" in RT.explain_influence(rets[0])
    assert RT.influence_shares(RT.Retriever(RT.KnowledgeIndex()).retrieve(mk(), RNOW)) == {}
    single = RT.Retrieval("x", "y", (dataclasses.replace(rets[0].items[0], score=0.9), dataclasses.replace(rets[0].items[-1], score=0.05)), (), None, None, "w")
    assert "single-item bet" in RT.explain_influence(single)
    # leverage: one anecdote with a huge outcome dominates, a balanced item does not
    lev = RT.KnowledgeIndex()
    lev.add_item(KItem("ANEC"))
    lev.add_item(KItem("BAL"))
    for i in range(20):
        d = dt.date(2017, 1, 1) + dt.timedelta(days=i)
        lev.add_support("ANEC", mk(), d, 0.001, ref=f"a{i}")
        lev.add_support("BAL", mk(), d, 0.02, ref=f"b{i}")
    lev.add_support("ANEC", mk(), dt.date(2017, 3, 1), 5.0, ref="huge")
    assert RT.support_leverage(lev, "ANEC", mk(), RNOW)["verdict"] == "ANECDOTE" and RT.support_leverage(lev, "ANEC", mk(), RNOW)["top"][0][0] == "huge"
    assert RT.support_leverage(lev, "BAL", mk(), RNOW)["verdict"] == "ROBUST"
    thin = RT.KnowledgeIndex()
    thin.add_item(KItem("T"))
    assert RT.support_leverage(thin, "T", mk(), RNOW)["verdict"] == Unknown.INSUFFICIENT_DATA
    cfg_back = RT.config_from_record(RT.config_to_record(RT.RetrievalConfig(top_k=7, allowed_promotions=(Promotion.CHAMPION, Promotion.CHALLENGER))))
    assert cfg_back.top_k == 7 and Promotion.CHALLENGER in cfg_back.allowed_promotions
    bad = RT.config_to_record(RT.RetrievalConfig())
    bad["top_k"] = 0
    with pytest.raises(ValueError):
        RT.config_from_record(bad)


def test_tables_ablation_suite_and_drift_over_time():
    ix = build_index()
    rt = RT.Retriever(ix, config=RT.RetrievalConfig(min_similarity=0.0))
    q = mk(regime__label="bull_calm")
    r = rt.retrieve(q, RNOW)
    rows = RT.factor_table(r)
    assert rows[0]["rank"] == 1 and set(RT.FACTORS) <= set(rows[0])
    suite = RT.ablation_suite(rt, [q, mk(regime__label="stress", regime__vol_regime="crisis", market__vix=40.0)], RNOW)
    assert set(suite) == set(RT.POSITIVE_FACTORS) and all(0.0 <= v["overlap"] <= 1.0 for v in suite.values())
    assert RT.ablation_suite(RT.Retriever(RT.KnowledgeIndex()), [q], RNOW) == {}
    drift = RT.drift_over_time(rt, q, [dt.date(2019, 6, 3), dt.date(2022, 6, 3), dt.date(2030, 6, 3)])
    assert len(drift["ids"]) == 3 and len(drift["consecutive_overlap"]) == 2 and 0.0 <= drift["min_overlap"] <= 1.0
    part = RT.KnowledgeIndex()
    part.add_item(KItem("Q", provenance=dataclasses.replace(PROV, learned_at="2018-06-01")))
    support(part, "Q", 30, 2, "bull_calm")
    old = RT.drift_over_time(RT.Retriever(part), q, [dt.date(2018, 12, 1), dt.date(2019, 6, 1)])
    assert old["dates"] == ["2018-12-01", "2019-06-01"]


def test_index_grouping_and_retriever_description():
    ix = RT.KnowledgeIndex()
    ix.add_item(KItem("A", decision_effect=(DecisionEffect.STOP, DecisionEffect.EXIT)))
    ix.add_item(KItem("B", decision_effect=(DecisionEffect.STOP,)))
    assert RT.index_by_effect(ix) == {"EXIT": ["A"], "STOP": ["A", "B"]}
    d = RT.describe_retriever(RT.Retriever(ix, monitor=RT.SkillMonitor(min_n=25)))
    assert "skill monitor (min_n 25)" in d and "current_reliability 0.16" in d and "CHAMPION" in d
    assert "no skill monitor" in RT.describe_retriever(RT.Retriever(ix))
