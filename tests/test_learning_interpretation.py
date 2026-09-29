"""S01: observation != interpretation (section 7). Planted worlds prove the competing hypotheses H1-H6 are told apart, that
uncertainty is retained, and that causal wording is refused until the evidence ladder allows it."""
import dataclasses
import math

import numpy as np
import pytest

from engine.learning import interpretation as it
from engine.learning import knowledge as kn
from engine.learning.core import DecisionEffect, FirewallBreach
from engine.learning.interpretation import (ClaimError, HypKind, Level, MechStatus, Mechanism, Observation, StatRelation)

H1, H2, H3, H4, H5, H6 = it.CORE_KINDS
NOW = "2021-01-04"


def obs(**kw):
    d = dict(statement="stocks in the top momentum quintile earned more the following week", window_start="2016-01-01",
             window_end="2020-12-31", n=600, value=0.004)
    d.update(kw)
    return Observation(**d)


def rel(o, **kw):
    d = dict(observation_id=o.observation_id, estimate=0.004, se=0.001, n_eff=300.0, n_tests=1, out_of_sample=True,
             controls=("volatility",))
    d.update(kw)
    return StatRelation(**d)


# ------------------------------------------------------------------ separate records, each validated

@pytest.mark.parametrize("kw,needle", [
    (dict(statement="momentum causes the price to rise"), "worded causally"),
    (dict(statement="the move happened because of earnings"), "worded causally"),
    (dict(statement=" "), "no statement"),
    (dict(window_start="2021-01-01", window_end="2020-01-01"), "ends before it starts"),
    (dict(window_start="yesterday"), "ISO"),
    (dict(n=0), "n >= 1"),
    (dict(value=float("inf")), "not finite"),
])
def test_observation_rejects_causal_wording_and_bad_windows(kw, needle):
    assert any(needle in e for e in obs(**kw).check())


def test_relation_multiple_testing_and_consistency():
    o = obs()
    one = rel(o, estimate=0.0021, se=0.001, n_tests=1)                    # t = 2.1
    mined = rel(o, estimate=0.0021, se=0.001, n_tests=200)                # same t, found by searching 200 candidates
    assert one.significant() and not mined.significant()
    assert mined.adjusted_p() > 0.9 and one.adjusted_p() < 0.05
    assert rel(o, se=0.0).t is None and rel(o, se=0.0).adjusted_p() is None and not rel(o, se=0.0).significant()
    assert rel(o, p_value=0.5, n_tests=1).adjusted_p() == 0.5 and rel(o, p_value=1.0, n_tests=9).adjusted_p() == 1.0
    for bad in (dict(n_eff=0), dict(n_tests=0), dict(se=-1.0), dict(p_value=1.5), dict(method="X causes Y"),
                dict(estimate=float("nan"))):
        assert rel(o, **bad).check(), bad


def test_hypothesis_needs_testable_predictions():
    h = it.standard_hypotheses("pattern X", "2020-12-31")
    assert [x.kind for x in h] == list(it.CORE_KINDS) and all(x.check() == [] for x in h)
    assert len({x.hyp_id for x in h}) == 6
    assert dataclasses.replace(h[0], predictions=()).check()                                        # untestable
    assert dataclasses.replace(h[0], prior=1.0).check() and dataclasses.replace(h[0], stated_at="soon").check()
    assert it.Prediction("x", "").check()


def test_mechanism_needs_an_implication_and_evidence_when_supported():
    assert Mechanism("H1-x", "momentum", "investors under-react", "returns drift for weeks").check() == []
    assert Mechanism("H1-x", "momentum", "story", "").check()
    assert Mechanism("H1-x", "momentum", "story", "impl", MechStatus.SUPPORTED).check()
    assert it.DecisionUsefulness(DecisionEffect.RANKING, 0.001, 0.0004, 100).measured
    assert not it.DecisionUsefulness().measured and it.DecisionUsefulness().z is None
    assert it.DecisionUsefulness(DecisionEffect.NONE, 0.1, 0.1, 5).check() and it.DecisionUsefulness(DecisionEffect.RANKING, 0.1, None).check()


def test_causal_language_detector():
    assert it.causal_language("drives returns and leads to gains") == ["drives", "leads to"]
    assert it.causal_language("association +0.004 (se 0.001), controlling for volatility") == []
    assert it.causal_language("") == [] and it.causal_language(None) == []


# ------------------------------------------------------------------ competing hypotheses retain uncertainty

def test_uniform_start_and_bad_priors():
    s = it.HypothesisSet.uniform()
    assert all(abs(w - 1 / 6) < 1e-12 for _, w in s.weights) and s.check() == []
    assert abs(s.effective_number() - 6.0) < 1e-9 and not s.resolved() and s.margin() == 0.0
    for bad in ({}, {H1: -1.0, H2: 2.0}, {H1: 0.0}):
        with pytest.raises(kn.SchemaError):
            it.HypothesisSet.from_priors(bad)
    with pytest.raises(kn.SchemaError):
        it.HypothesisSet.from_priors({H1: 1.0, H2: 1.0}, floor=0.6)


def test_update_floor_keeps_every_rival_alive():
    s = it.HypothesisSet.uniform(floor=0.02)
    for i, name in enumerate(["survives_volatility_control", "survives_selection_control", "out_of_sample_confirmed",
                              "stable_across_regimes", "incremental_over_existing", "sign_consistent_rolling"]):
        s = s.apply_tests({name: True}, f"2020-0{i + 1}-01")
    assert s.leading()[0] is H1 and s.resolved()
    assert all(w >= 0.02 - 1e-12 for _, w in s.weights) and s.check() == []                         # nobody is eliminated
    assert s.effective_number() > 1.0
    assert len(s.log) == 6 and s.log[0].name == "survives_volatility_control"


def test_update_refusals_and_tempering():
    s = it.HypothesisSet.uniform()
    live = {k: 0.5 for k in it.CORE_KINDS}
    s1 = s.update("e1", {**live, H1: 0.9}, "2020-01-01")
    with pytest.raises(ClaimError):
        s1.update("e1", live, "2020-02-01")                                                         # same evidence twice
    with pytest.raises(ClaimError):
        s1.update("e2", {H1: 0.9}, "2020-02-01")                                                    # missing likelihoods
    with pytest.raises(ClaimError):
        s1.update("e2", {**live, H1: 1.5}, "2020-02-01")
    with pytest.raises(ClaimError):
        s1.update("e2", {k: 0.0 for k in it.CORE_KINDS}, "2020-02-01")
    with pytest.raises(ClaimError):
        s1.update("e2", live, "2019-01-01")                                                         # out of time order
    with pytest.raises(ClaimError):
        s1.update("e2", live, "2020-02-01", strength=0.0)
    with pytest.raises(FirewallBreach):
        s1.update("e2", live, "2020-02-01", evidence_through="2020-02-01")                          # evidence not before now
    strong = s.update("a", {**live, H1: 0.9}, "2020-01-01", strength=1.0).weight(H1)
    weak = s.update("a", {**live, H1: 0.9}, "2020-01-01", strength=0.3).weight(H1)
    assert 1 / 6 < weak < strong


def test_unknown_test_and_unrun_tests():
    s = it.HypothesisSet.uniform()
    assert s.apply_tests({"stable_across_regimes": None}, NOW).weights == s.weights                # not run => no update
    with pytest.raises(ClaimError):
        s.apply_tests({"astrology": True}, NOW)


# ------------------------------------------------------------------ planted worlds: the six explanations are separable

def world(kind, n=900, seed=0):
    rng = np.random.default_rng(seed)
    vol, z, e = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    regime = (rng.random(n) < 0.5).astype(int)
    if kind == "genuine":
        x = rng.normal(size=n)
        y = 0.6 * x + e
    elif kind == "vol_proxy":
        x = vol + 0.3 * rng.normal(size=n)
        y = 0.8 * vol + e
    elif kind == "regime":
        x = rng.normal(size=n)
        y = np.where(regime == 1, 0.9 * x, -0.1 * x) + e
    elif kind == "redundant":
        x = z + 0.02 * rng.normal(size=n)
        y = 0.6 * z + e
    elif kind == "unstable":
        x = rng.normal(size=n)
        sign = np.where((np.arange(n) // 90) % 2 == 0, 1.0, -1.0)
        y = 0.7 * sign * x + e
    else:
        raise ValueError(kind)
    return y, x, vol, regime, z


def run(kind, **kw):
    y, x, vol, regime, z = world(kind)
    outs = it.run_standard_tests(y, x, vol=vol, regime=regime, others=[z], **kw)
    return outs, it.HypothesisSet.uniform().apply_tests(it.outcomes_to_results(outs), NOW)


def test_genuine_effect_resolves_to_h1():
    outs, b = run("genuine", discovery_t=4.0, holdout_t=3.6)
    assert all(o.passed for o in outs), [(o.name, o.detail) for o in outs]
    assert b.leading()[0] is H1 and b.resolved()


def test_volatility_proxy_is_caught_as_h2():
    outs, b = run("vol_proxy", discovery_t=4.0, holdout_t=3.6)
    res = it.outcomes_to_results(outs)
    assert res["survives_volatility_control"] is False, [(o.name, o.detail) for o in outs]
    assert b.leading()[0] is H2 and b.weight(H2) > 2 * b.weight(H1)


def test_regime_specific_effect_is_caught_as_h3():
    outs, b = run("regime", discovery_t=4.0, holdout_t=3.6)
    assert it.outcomes_to_results(outs)["stable_across_regimes"] is False
    assert b.leading()[0] is H3 or b.weight(H3) > 2 * b.weight(H1)


def test_redundant_feature_is_caught_as_h4():
    outs, b = run("redundant", discovery_t=4.0, holdout_t=3.6)
    inc = [o for o in outs if o.name == "incremental_over_existing"][0]
    assert inc.passed is False or inc.passed is None
    assert b.leading()[0] is H4


def test_selection_bias_is_caught_as_h5():
    y, x, vol, regime, z = world("genuine")
    outs = it.run_standard_tests(y, x, vol=vol, regime=regime, others=[z], discovery_t=4.5, holdout_t=0.3)
    b = it.HypothesisSet.uniform().apply_tests(it.outcomes_to_results(outs), NOW)
    assert it.outcomes_to_results(outs)["survives_selection_control"] is False
    assert b.leading()[0] is H5


def test_unstable_effect_is_caught_as_h6():
    outs, b = run("unstable", discovery_t=4.0, holdout_t=3.6)
    assert it.outcomes_to_results(outs)["sign_consistent_rolling"] is False
    assert b.leading()[0] is H6 or b.weight(H6) > b.weight(H1)


def test_pure_noise_does_not_resolve_and_does_not_favour_h1():
    rng = np.random.default_rng(9)
    y, x, vol = rng.normal(size=900), rng.normal(size=900), rng.normal(size=900)
    outs = it.run_standard_tests(y, x, vol=vol, others=[rng.normal(size=900)], discovery_t=2.1, holdout_t=0.1)
    b = it.HypothesisSet.uniform().apply_tests(it.outcomes_to_results(outs), NOW)
    assert not b.resolved() or b.leading()[0] is not H1


def test_probes_report_none_when_they_cannot_run():
    rng = np.random.default_rng(1)
    y, x = rng.normal(size=30), rng.normal(size=30)
    assert it.probe_rolling_sign(y, x, window=60).passed is None
    assert it.probe_regime_stability(y, x, np.zeros(30)).passed is None                 # one regime only
    assert it.probe_selection_control(-1.0, 2.0).passed is None                         # discovery not in claimed direction
    assert it.probe_volatility_control(y[:5], x[:5], x[:5]).passed is None              # too few observations
    assert it.probe_volatility_control(y, x, x).passed is None                          # control identical to signal: rank deficient
    assert it.probe_incremental(y, x, [x]).passed is False                              # fully redundant => fails, not "unknown"
    with pytest.raises(ClaimError):
        it.probe_volatility_control(y, x[:10], x)
    assert it.probe_volatility_control(y, x, rng.normal(size=30)).passed is None            # noise: no raw effect to test
    y2 = 2.0 * x + 0.3 * rng.normal(size=30)
    y2[0] = np.nan
    assert it.probe_volatility_control(y2, x, rng.normal(size=30)).passed is True           # NaN row dropped, not propagated
    assert it.run_standard_tests(y, x)[0].name == "sign_consistent_rolling" and len(it.run_standard_tests(y, x)) == 1


# ------------------------------------------------------------------ the evidence ladder and claim control

def ladder_record():
    o = obs()
    rec = it.open_record(o, "2020-12-31", "momentum")
    return rec


def test_level_climbs_one_rung_at_a_time():
    rec = ladder_record()
    assert rec.level() is Level.OBSERVATION
    mined = rec.with_relation(rel(rec.observation, estimate=0.0021, se=0.001, n_tests=500, out_of_sample=True))
    assert mined.level() is Level.OBSERVATION                                        # significant only before the search penalty
    in_sample = rec.with_relation(rel(rec.observation, out_of_sample=False))
    assert in_sample.level() is Level.OBSERVATION
    assoc = rec.with_relation(rel(rec.observation))
    assert assoc.level() is Level.HYPOTHESIS                                         # hypotheses were opened by open_record
    y, x, vol, regime, z = world("genuine")
    tested = assoc.with_tests(it.outcomes_to_results(it.run_standard_tests(y, x, vol=vol, regime=regime, others=[z],
                                                                            discovery_t=4.0, holdout_t=3.6)), NOW)
    assert tested.level() is Level.HYPOTHESIS                                        # H1 leads, but no mechanism yet
    h1 = [h for h in tested.hypotheses if h.kind is H1][0]
    mech = Mechanism(h1.hyp_id, "underreaction", "slow diffusion of news", "drift should be larger after low-attention days",
                     MechStatus.SUPPORTED, tested.relations[0].relation_id)
    with_m = tested.with_mechanism(mech)
    assert with_m.level() is Level.MECHANISM
    useful = with_m.with_usefulness(it.DecisionUsefulness(DecisionEffect.RANKING, 0.0008, 0.0003, 200))
    assert useful.level() is Level.DECISION_VALUE
    weak = with_m.with_usefulness(it.DecisionUsefulness(DecisionEffect.RANKING, 0.0002, 0.0003, 200))
    assert weak.level() is Level.MECHANISM


def test_mechanism_on_an_unresolved_belief_does_not_reach_mechanism_level():
    rec = ladder_record().with_relation(rel(ladder_record().observation))
    h1 = [h for h in rec.hypotheses if h.kind is H1][0]
    m = Mechanism(h1.hyp_id, "t", "story", "implication", MechStatus.SUPPORTED, rec.relations[0].relation_id)
    assert rec.with_mechanism(m).level() is Level.HYPOTHESIS                         # a story cannot resolve the rivals


def test_claim_control_refuses_causal_wording_and_overclaims():
    rec = ladder_record()
    rec.assert_claim("association only", Level.OBSERVATION)
    with pytest.raises(ClaimError):
        rec.assert_claim("momentum drives returns", Level.OBSERVATION)              # causal wording at OBSERVATION level
    with pytest.raises(ClaimError):
        rec.assert_claim("clean wording", Level.ASSOCIATION)                        # level not earned
    bad = it.scan_causal_leaps([("X causes Y", rec), ("X is associated with Y", rec)])
    assert len(bad) == 1 and "causes" in bad[0] and it.scan_causal_leaps([]) == []


def test_bundle_validation_catches_planted_inconsistencies():
    rec = ladder_record()
    other = obs(statement="a different thing was seen")
    assert any("another observation" in e for e in dataclasses.replace(rec, relations=(rel(other),)).validate())
    assert any("unknown hypothesis" in e for e in dataclasses.replace(
        rec, mechanisms=(Mechanism("H9-none", "t", "s", "i"),)).validate())
    dup = dataclasses.replace(rec, hypotheses=rec.hypotheses + (rec.hypotheses[0],))
    assert any("duplicate hypothesis" in e for e in dup.validate())
    assert any("different hypotheses" in e for e in dataclasses.replace(rec, belief=it.HypothesisSet.uniform((H1, H2))).validate())
    with pytest.raises(kn.SchemaError):
        dataclasses.replace(rec, relations=(rel(other),)).assert_valid()
    with pytest.raises(ClaimError):
        dataclasses.replace(rec, belief=None).with_tests({"stable_across_regimes": True}, NOW)
    m = Mechanism(rec.hypotheses[0].hyp_id, "t", "s", "i", MechStatus.SUPPORTED, "REL-not-there")
    assert any("not in the record" in e for e in dataclasses.replace(rec, mechanisms=(m,)).validate())


# ------------------------------------------------------------------ bundle -> knowledge, serialisation

def test_record_round_trip_and_hash_stability():
    rec = ladder_record().with_relation(rel(ladder_record().observation))
    back = it.InterpretationRecord.from_dict(rec.to_dict())
    assert back.record_hash() == rec.record_hash() and back.level() == rec.level()
    assert dict(back.belief.weights) == pytest.approx(dict(rec.belief.weights), abs=1e-9) and back.hypotheses[0].hyp_id == rec.hypotheses[0].hyp_id
    changed = ladder_record().with_tests({"survives_volatility_control": False}, NOW)
    assert changed.record_hash() != rec.record_hash()
    with pytest.raises(kn.SchemaError):
        it.InterpretationRecord.from_dict({"schema": 5, "record": {}})


def test_report_and_wording_are_association_only():
    rec = ladder_record().with_relation(rel(ladder_record().observation, controls=("volatility", "beta")))
    text = rec.report()
    assert "level       : HYPOTHESIS" in text and "H1" in text and "controls=['volatility', 'beta']" in text
    w = rec.association_wording()
    assert "controlling for volatility, beta" in w and it.causal_language(w) == [] and "mechanism unproven" in w
    assert ladder_record().association_wording() == ""


def test_knowledge_fields_feed_a_valid_knowledge_object():
    rec = ladder_record().with_relation(rel(ladder_record().observation))
    f = rec.knowledge_fields()
    assert f["interpretation_ref"] == rec.record_hash() and f["mechanism_tags"] == ()
    assert it.causal_language(f["interpretation"]) == [] and "leading" in f["hypothesis"]
    prov = kn.make_provenance("2020-12-31", code_hash="h")
    k = kn.KnowledgeObject("K-i", "2021-01-01", prov, effect=kn.Effect(1, 0.004, 0.001), **f)
    assert k.validate() == [] and k.interpretation_ref == rec.record_hash()
    k2 = kn.KnowledgeObject.from_json(k.to_json())
    assert k2.interpretation == k.interpretation


def test_end_to_end_interpret_helper():
    y, x, vol, regime, z = world("vol_proxy")
    o = obs()
    outs = it.run_standard_tests(y, x, vol=vol, regime=regime, others=[z])
    rec = it.interpret(o, rel(o), outs, NOW, "2020-12-31", "the momentum pattern", evidence_through="2020-12-31")
    assert rec.belief.leading()[0] is H2 and rec.level() is Level.HYPOTHESIS and len(rec.belief.log) == len(outs)
    with pytest.raises(FirewallBreach):
        it.interpret(o, rel(o), outs, NOW, "2020-12-31", evidence_through=NOW)


# ------------------------------------------------------------------ choosing the next test and reading belief movement

def test_next_test_targets_the_ambiguity_and_information_gain_is_positive():
    b = it.HypothesisSet.uniform()
    gains = {t: it.expected_information_gain(b, t) for t in it.TESTS}
    assert all(g > 0 for g in gains.values())
    b2 = b.apply_tests({"survives_volatility_control": False}, NOW)                   # H2 now leads
    top = it.next_test(b2)
    assert top is not None and top[0] != "survives_volatility_control" and top[1] > 0
    assert it.unresolved_pairs(it.HypothesisSet.uniform(), within=0.01)[0][2] == pytest.approx(1 / 3)
    assert not it.unresolved_pairs(b2, within=0.01) or all(c < 0.9 for _, _, c in it.unresolved_pairs(b2, within=0.01))
    full = b
    for i, t in enumerate(sorted(it.TESTS)):
        full = full.apply_tests({t: True}, f"2020-0{i + 1}-01")
    assert it.next_test(full) is None
    with pytest.raises(ClaimError):
        it.expected_information_gain(b, "astrology")
    assert 0.0 < it.outcome_probability(b, "sign_consistent_rolling") < 1.0


def test_information_gain_is_largest_for_the_test_that_separates_the_leaders():
    b = it.HypothesisSet.from_priors({H1: 0.45, H2: 0.45, H3: 0.025, H4: 0.025, H5: 0.025, H6: 0.025})
    g = {t: it.expected_information_gain(b, t) for t in it.TESTS}
    assert max(g, key=g.get) == "survives_volatility_control"                         # only this test can split H1 from H2


def test_belief_shift_and_kl_and_evidence_ledger():
    a = it.HypothesisSet.uniform()
    assert it.kl_divergence(a, a) == 0.0
    b = a.apply_tests({"survives_volatility_control": False, "sign_consistent_rolling": True}, NOW, evidence_through="2020-12-30")
    sh = it.belief_shift(a, b)
    assert sh["leader_changed"] and sh["leader_after"] == "H2" and sh["kl"] > 0 and sh["alive_after"] < sh["alive_before"]
    rows = it.evidence_ledger(b)
    assert len(rows) == 2 and any("survives_volatility_control: favours H2" in r for r in rows)
    assert any("evidence through 2020-12-30" in r for r in rows) and it.evidence_ledger(a) == []
    with pytest.raises(ClaimError):
        it.kl_divergence(a, it.HypothesisSet.uniform((H1, H2)))
