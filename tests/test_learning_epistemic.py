"""S01: epistemic states (section 6) with enforced definitions/transitions/scopes, and unknown states (section 42).
Synthetic only. Each mechanism gets a planted case it must catch and the empty/degenerate case."""
import dataclasses

import numpy as np
import pytest

from engine.learning import epistemic as ep
from engine.learning import knowledge as kn
from engine.learning import unknowns as uk
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Health, Lifecycle, Promotion,
                                  Unknown, UnknownAction)
from engine.learning.epistemic import (CONTEXT, CURRENT, HISTORICAL, EpistemicError, EpistemicProfile, EvidenceSummary,
                                       IllegalTransition)

E = Epistemic
NOW = "2021-01-04"

STRONG = EvidenceSummary(n_events=300, n_eff=120, t_discovery=3.5, t_confirm=3.0, p_real=0.95, contradiction_rate=0.10,
                         recent_ratio=0.95, reversal_t=-1.0, has_hypothesis=True, has_prediction=True)


def ev(**kw):
    return dataclasses.replace(STRONG, **kw)


# ------------------------------------------------------------------ definitions are enforced

@pytest.mark.parametrize("state,good,bad", [
    (E.OBSERVED, EvidenceSummary(n_events=3), EvidenceSummary(n_events=0)),
    (E.OBSERVED, EvidenceSummary(n_events=3), EvidenceSummary(n_events=3, has_hypothesis=True)),
    (E.HYPOTHESIS, EvidenceSummary(n_events=3, has_hypothesis=True, has_prediction=True), EvidenceSummary(n_events=3, has_hypothesis=True)),
    (E.HYPOTHESIS, EvidenceSummary(has_hypothesis=True, has_prediction=True), ev()),           # confirmed evidence => not merely a hypothesis
    (E.SUPPORTED, STRONG, ev(n_eff=10)),
    (E.SUPPORTED, STRONG, ev(t_confirm=None)),
    (E.SUPPORTED, STRONG, ev(t_confirm=1.0)),
    (E.SUPPORTED, STRONG, ev(p_real=0.4)),
    (E.SUPPORTED, STRONG, ev(contradiction_rate=0.4)),
    (E.SUPPORTED, STRONG, ev(fails_in=("regime=bear",))),
    (E.CONDITIONAL, ev(works_in=("regime=bull",), fails_in=("regime=bear",)), ev()),
    (E.CONDITIONAL, ev(works_in=("regime=bull",), fails_in=("regime=bear",), t_confirm=0.3), ev(works_in=("a",), fails_in=("b",), n_eff=5)),
    (E.CONDITIONAL, ev(works_in=("regime=bull",), fails_in=("regime=bear",)), ev(works_in=("regime=bull",))),
    (E.DEGRADED, ev(prior_supported=True, recent_ratio=0.2), ev(recent_ratio=0.2)),
    (E.DEGRADED, ev(prior_supported=True, recent_ratio=0.2), ev(prior_supported=True, recent_ratio=0.9)),
    (E.DEGRADED, ev(prior_supported=True, recent_ratio=0.2), ev(prior_supported=True, recent_ratio=None)),
    (E.DEGRADED, ev(prior_supported=True, recent_ratio=0.2), ev(prior_supported=True, recent_ratio=-1.0, reversal_t=3.0)),
    (E.CONTRADICTED, ev(reversal_t=3.5), ev(reversal_t=1.0)),
    (E.CONTRADICTED, ev(contradiction_rate=0.7), ev(contradiction_rate=0.7, n_eff=5)),
    (E.GATED, ev(gate_reason="regime_gate"), ev()),
    (E.RETIRED, ev(retire_reason="reversed"), ev()),
    (E.UNKNOWN, EvidenceSummary(n_events=4, n_eff=4), STRONG),
])
def test_definition_holds_for_good_and_fails_for_planted_bad(state, good, bad):
    scope = CURRENT if state == E.DEGRADED else HISTORICAL
    assert ep.violations(state, good, scope) == [], state
    assert ep.violations(state, bad, scope), state


def test_scope_specific_definitions():
    assert ep.violations(E.DEGRADED, ev(prior_supported=True, recent_ratio=0.2), HISTORICAL)            # recent-window notion
    assert ep.violations(E.OBSERVED, EvidenceSummary(n_events=2), CURRENT)                             # only historical
    assert ep.violations(E.SUPPORTED, ev(recent_ratio=None), CURRENT)                                  # current needs recent evidence
    assert ep.violations(E.SUPPORTED, ev(recent_ratio=None), HISTORICAL) == []
    assert ep.violations(E.CONDITIONAL, ev(works_in=("bull",), fails_in=("low_vol",)), CONTEXT) == []
    assert ep.violations(E.CONDITIONAL, ev(works_in=("bull",), fails_in=("low_vol",), t_confirm=0.5), CONTEXT)   # needs confirmation there
    with pytest.raises(EpistemicError):
        ep.violations(E.SUPPORTED, STRONG, "SOMEWHERE")


@pytest.mark.parametrize("bad", [EvidenceSummary(n_events=5, n_eff=9), EvidenceSummary(p_real=1.5),
                                 EvidenceSummary(t_confirm=float("nan")), EvidenceSummary(n_events=-1),
                                 EvidenceSummary(works_in=("a",), fails_in=("a",))])
def test_malformed_evidence_is_rejected(bad):
    assert bad.check()
    with pytest.raises(EpistemicError):
        ep.derive_state(bad)


def test_thresholds_must_be_coherent():
    assert ep.Thresholds().check() == []
    assert ep.Thresholds(max_contradiction=0.6).check()                     # SUPPORTED and CONTRADICTED would overlap
    assert ep.Thresholds(min_n_eff=0).check() and ep.Thresholds(min_p_real=2).check()


def test_derived_state_always_satisfies_its_own_definition():
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(600):
        n = int(rng.integers(0, 400))
        e = EvidenceSummary(
            n_events=n, n_eff=float(rng.uniform(0, n)) if n else 0.0,
            t_confirm=None if rng.random() < .2 else float(rng.normal(1.5, 2)),
            p_real=None if rng.random() < .2 else float(rng.random()),
            contradiction_rate=None if rng.random() < .3 else float(rng.random()),
            recent_ratio=None if rng.random() < .3 else float(rng.normal(0.7, 0.8)),
            reversal_t=None if rng.random() < .5 else float(rng.normal(0, 2)),
            works_in=("a",) if rng.random() < .3 else (), fails_in=("b",) if rng.random() < .3 else (),
            has_hypothesis=bool(rng.random() < .8), has_prediction=bool(rng.random() < .8),
            gate_reason="g" if rng.random() < .05 else "", retire_reason="r" if rng.random() < .03 else "",
            prior_supported=bool(rng.random() < .5))
        for scope in (HISTORICAL, CONTEXT, CURRENT):
            s = ep.derive_state(e, scope)
            seen.add(s)
            problems = ep.violations(s, e, scope)
            assert not problems, (s, scope, e, problems)
    assert len(seen) >= 7                                                   # the sweep actually reaches most states


def test_diagnose_lists_every_state_and_justified_states():
    d = ep.diagnose(STRONG)
    assert set(d) == {s.value for s in ep.SCOPE_STATES[HISTORICAL]} and d["SUPPORTED"] == []
    assert E.SUPPORTED in ep.justified_states(STRONG) and E.RETIRED not in ep.justified_states(STRONG)
    assert ep.justified_states(EvidenceSummary()) == [E.UNKNOWN]


# ------------------------------------------------------------------ transitions

def test_transition_table_shape():
    g = ep.transition_graph()
    assert set(g["RETIRED"]) == {"HYPOTHESIS"} and "OBSERVED" not in g["SUPPORTED"]
    reach, todo = set(), ["START"]
    while todo:
        cur = todo.pop()
        for nxt in g.get(cur, []):
            if nxt not in reach:
                reach.add(nxt)
                todo.append(nxt)
    assert reach == {s.value for s in E}                                    # every state can be entered through legal moves
    assert not ep.legal(E.SUPPORTED, E.OBSERVED) and not ep.legal(E.RETIRED, E.SUPPORTED) and ep.legal(None, E.HYPOTHESIS)
    assert not ep.legal(None, E.SUPPORTED)                                  # belief forms through a hypothesis


def _hypothesis_profile():
    e = EvidenceSummary(n_events=5, has_hypothesis=True, has_prediction=True)
    return EpistemicProfile().apply(HISTORICAL, "", E.HYPOTHESIS, e, "2020-01-01"), e


def test_illegal_and_unjustified_moves_are_refused():
    p, _ = _hypothesis_profile()
    with pytest.raises(EpistemicError):                                     # legal edge, but the evidence does not justify it
        p.apply(HISTORICAL, "", E.SUPPORTED, ev(t_confirm=0.5), "2020-02-01")
    with pytest.raises(IllegalTransition):                                  # not a legal edge
        p.apply(HISTORICAL, "", E.OBSERVED, EvidenceSummary(n_events=3), "2020-02-01")
    with pytest.raises(IllegalTransition):                                  # same state
        p.apply(HISTORICAL, "", E.HYPOTHESIS, EvidenceSummary(n_events=5, has_hypothesis=True, has_prediction=True), "2020-02-01")
    with pytest.raises(EpistemicError):
        p.apply(CONTEXT, "", E.SUPPORTED, STRONG, "2020-02-01")             # context scope needs a key
    with pytest.raises(EpistemicError):
        p.apply(HISTORICAL, "k", E.SUPPORTED, STRONG, "2020-02-01")
    with pytest.raises(EpistemicError):
        p.apply(HISTORICAL, "", E.DEGRADED, ev(prior_supported=True, recent_ratio=0.1), "2020-02-01")   # DEGRADED not historical
    with pytest.raises(FirewallBreach):
        p.apply(HISTORICAL, "", E.SUPPORTED, STRONG, "2019-12-01")          # dated before the previous transition
    assert p.apply(HISTORICAL, "", E.SUPPORTED, STRONG, "2020-02-01").historical is E.SUPPORTED
    assert p.historical is E.HYPOTHESIS                                     # immutable: the original is unchanged


def test_retire_is_only_left_by_explicit_revival_with_new_evidence():
    p, _ = _hypothesis_profile()
    r = p.apply(HISTORICAL, "", E.RETIRED, ev(retire_reason="reversed"), "2020-03-01", "reversed")
    revive_ev = EvidenceSummary(n_events=8, has_hypothesis=True, has_prediction=True)
    with pytest.raises(IllegalTransition):
        r.apply(HISTORICAL, "", E.HYPOTHESIS, revive_ev, "2020-04-01")                   # no revive flag
    with pytest.raises(IllegalTransition):
        r.apply(HISTORICAL, "", E.HYPOTHESIS, EvidenceSummary(has_hypothesis=True, has_prediction=True), "2020-04-01", revive=True)
    with pytest.raises(IllegalTransition):
        r.apply(HISTORICAL, "", E.SUPPORTED, STRONG, "2020-04-01", revive=True)          # revival is only as hypothesis
    back = r.apply(HISTORICAL, "", E.HYPOTHESIS, revive_ev, "2020-04-01", revive=True)
    assert back.historical is E.HYPOTHESIS and len(back.history) == 3       # the retirement stays on the record


# ------------------------------------------------------------------ scoped states (the section-6 example)

def scoped_example():
    good = dataclasses.replace(STRONG, works_in=(), fails_in=())
    bull = ev(works_in=("regime=bull",), fails_in=("vix<15",))
    bear = ev(reversal_t=3.5, contradiction_rate=0.7)
    degraded_now = ev(prior_supported=True, recent_ratio=0.2, reversal_t=0.5, contradiction_rate=0.2)
    return ep.build_profile(NOW, good, {"regime=bull": bull, "regime=bear": bear}, degraded_now)


def test_supported_historically_conditional_by_regime_degraded_now_is_preserved():
    p = scoped_example()
    assert p.historical is E.SUPPORTED and p.current is E.DEGRADED
    assert p.contexts() == {"regime=bear": E.CONTRADICTED, "regime=bull": E.CONDITIONAL}
    assert p.headline() is E.DEGRADED
    s = p.summary()
    assert "SUPPORTED historically" in s and "CONDITIONAL at regime=bull" in s and "DEGRADED currently" in s
    assert p.coherence() == [] and p.verify_history() == []


def test_usage_reduces_but_does_not_drop_degraded_and_vetoes_bad_contexts():
    p = scoped_example()
    th = ep.DEFAULT_THRESHOLDS
    assert p.usage(("regime=bull",)).weight == th.degraded_weight
    assert p.usage(("regime=bear",)).weight == 0.0 and "CONTRADICTED" in p.usage(("regime=bear",)).reason
    assert EpistemicProfile().usage().weight == 0.0
    sup = ep.build_profile(NOW, dataclasses.replace(STRONG))
    assert sup.usage().weight == 1.0 and sup.headline() is E.SUPPORTED


def test_conditional_needs_a_working_context_to_carry_weight():
    hist = ev(works_in=("regime=bull",), fails_in=("regime=bear",))
    p = ep.build_profile(NOW, hist, {"regime=bull": ev(works_in=("regime=bull",)),
                                     "regime=bear": ev(reversal_t=4.0)}, None)
    assert p.historical is E.CONDITIONAL
    assert p.usage(("regime=bull",)).weight == 1.0 and p.usage(()).weight == 0.0 and p.usage(("regime=bear",)).weight == 0.0


def test_cross_scope_coherence_is_enforced():
    hyp, _ = _hypothesis_profile()
    with pytest.raises(EpistemicError):                                      # cannot be currently SUPPORTED if only a hypothesis before
        hyp.apply(CURRENT, "", E.SUPPORTED, STRONG, "2020-02-01")
    with pytest.raises(EpistemicError):
        ep.build_profile(NOW, ev(works_in=("a",), fails_in=("b",)), None, None)   # CONDITIONAL history with no working context scope
    assert EpistemicProfile().headline() is E.UNKNOWN and "no scoped evidence" in EpistemicProfile().summary()


def test_history_chain_replay_and_tamper_detection():
    p = scoped_example()
    assert ep.replay(p.history).scopes == p.scopes and len(p.history) == 5
    assert [w for _, w in ep.weight_over_time(p, ("regime=bull",))][-1] == ep.DEFAULT_THRESHOLDS.degraded_weight
    forged = dataclasses.replace(p.history[1], to=E.SUPPORTED)
    assert EpistemicProfile(p.scopes, p.history[:1] + (forged,) + p.history[2:]).verify_history()
    dropped = EpistemicProfile(p.scopes, p.history[1:])
    assert dropped.verify_history()
    assert "HISTORICAL" in ep.render_history(p) and ep.render_history(EpistemicProfile()) == "(no transitions)"


# ------------------------------------------------------------------ evidence from per-period results (planted)

def _series(rng, n, mu, sd=1.0):
    return rng.normal(mu, sd, n)


def test_planted_real_effect_is_supported_and_noise_is_not():
    rng = np.random.default_rng(3)
    real = ep.summarize_periods(_series(rng, 200, 0.4), p_real=0.95, confirm_from=100)
    noise = ep.summarize_periods(_series(rng, 200, 0.0), p_real=0.2, confirm_from=100)
    assert ep.derive_state(real, HISTORICAL) is E.SUPPORTED and real.t_confirm > 2
    assert ep.derive_state(noise, HISTORICAL) is not E.SUPPORTED


def test_decayed_effect_is_degraded_and_reversed_effect_is_contradicted():
    rng = np.random.default_rng(4)
    decayed = np.concatenate([_series(rng, 150, 0.5), _series(rng, 24, 0.05)])
    reversed_ = np.concatenate([_series(rng, 150, 0.5), _series(rng, 24, -0.9)])
    d = ep.summarize_periods(decayed, p_real=0.95, recent=24, confirm_from=75, prior_supported=True)
    r = ep.summarize_periods(reversed_, p_real=0.95, recent=24, confirm_from=75, prior_supported=True)
    assert d.recent_ratio < 0.5 and ep.derive_state(d, CURRENT) is E.DEGRADED
    assert r.reversal_t > 2 and ep.derive_state(r, CURRENT) is E.CONTRADICTED


def test_direction_sign_and_missing_out_of_sample_are_not_zero():
    rng = np.random.default_rng(5)
    neg = _series(rng, 200, -0.4)
    assert ep.summarize_periods(neg, direction=-1, confirm_from=100).t_discovery > 2
    assert ep.summarize_periods(neg, direction=1, confirm_from=100).t_discovery < -2
    no_split = ep.summarize_periods(_series(rng, 200, 0.4))
    assert no_split.t_confirm is None and ep.derive_state(no_split, HISTORICAL) is not E.SUPPORTED
    assert ep.summarize_periods([]).n_events == 0
    with pytest.raises(EpistemicError):
        ep.summarize_periods([1.0, float("nan")])
    with pytest.raises(EpistemicError):
        ep.summarize_periods([1.0], direction=0)


def test_effective_sample_size_shrinks_under_autocorrelation():
    rng = np.random.default_rng(6)
    iid = rng.normal(size=400)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    assert ep._eff_n(iid) > 300 and ep._eff_n(ar) < 60
    assert ep._eff_n(np.ones(50)) == 50.0 and ep._eff_n(np.array([1.0, 2.0])) == 2.0


def test_context_dependent_effect_becomes_conditional():
    rng = np.random.default_rng(7)
    labels = np.array(["bull"] * 150 + ["bear"] * 150)
    x = np.concatenate([_series(rng, 150, 0.6), _series(rng, 150, -0.6)])
    order = rng.permutation(300)
    whole, per = ep.summarize_by_context(x[order], labels[order], p_real=0.95)
    assert whole.works_in == ("bull",) and whole.fails_in == ("bear",) and set(per) == {"bear", "bull"}
    p = ep.build_profile(NOW, whole, {c: dataclasses.replace(e, works_in=(c,) if c in whole.works_in else ()) for c, e in per.items()})
    assert p.historical is E.CONDITIONAL and p.contexts()["bull"] in (E.CONDITIONAL, E.SUPPORTED)
    with pytest.raises(EpistemicError):
        ep.summarize_by_context([1.0], ["a", "b"])
    w2, p2 = ep.summarize_by_context([0.1, 0.2, 0.1], ["a", "a", "b"])
    assert w2.works_in == () and w2.fails_in == ()                            # too little data: neither works nor fails


# ------------------------------------------------------------------ existing state vocabularies

def test_every_existing_vocabulary_is_mapped_and_mapping_is_consistent():
    assert ep.mapping_problems() == []
    assert ep.unmapped_states() == {}, ep.unmapped_states()


def test_mapping_values_and_failure_on_unmapped():
    m = ep.map_state("pattern_lifecycle.STATES", "no_gain")
    assert m.epistemic is E.GATED and m.gate and "P(real)" in m.note
    assert ep.map_state("pattern_lifecycle.STATES", "active").epistemic is E.SUPPORTED
    assert ep.map_state("pattern_lifecycle.STATES", "rescoped").epistemic is E.CONDITIONAL
    assert ep.map_state("pattern_memory.mode", "disregarded").epistemic is E.UNKNOWN     # not significant != false
    assert ep.map_state("pattern_memory.mode", "gated").epistemic is E.GATED
    assert ep.map_state("pattern_reliability.STATUS_NAMES", "broken").health is Health.BROKEN
    assert ep.map_state("pattern_reliability.STATUS_NAMES", "unmonitored").health is Health.INSUFFICIENT_EVIDENCE
    assert ep.map_state("pattern_reliability.VERDICTS", "DISCARDED_UNPREDICTABLE").lifecycle is Lifecycle.RETIRED
    assert ep.map_state("pattern_bank.PRIOR_STATES", "watch").epistemic is E.DEGRADED
    with pytest.raises(EpistemicError):
        ep.map_state("pattern_lifecycle.STATES", "brand_new_state")
    with pytest.raises(EpistemicError):
        ep.map_state("no_such_module", "x")


def test_mapping_self_check_catches_a_planted_bad_entry(monkeypatch):
    bad = dict(ep.PATTERN_MEMORY_MODES)
    bad["oops"] = ep.StateMapping(E.GATED, Lifecycle.DORMANT, CURRENT)                      # GATED with no gate
    bad["oops2"] = ep.StateMapping(E.RETIRED, Lifecycle.ACTIVE, HISTORICAL, retire_reason="x")   # inconsistent lifecycle
    bad["oops3"] = ep.StateMapping(E.OBSERVED, None, CURRENT)                               # OBSERVED invalid at CURRENT
    monkeypatch.setitem(ep.VOCABULARIES, "pattern_memory.mode", bad)
    problems = ep.mapping_problems()
    assert len(problems) == 3 and any("without a gate" in p for p in problems)


# ------------------------------------------------------------------ unknown states (section 42)

def facts(**kw):
    d = dict(n_events=200, n_eff=90.0, ever_tested=True, tested_contexts=("bull",), support=0.8, against=0.1)
    d.update(kw)
    return uk.UnknownFacts(**d)


AV = uk.Availability(can_collect_data=True, can_run_experiment=True, general_rule="baseline_rank")


@pytest.mark.parametrize("kw,state,action", [
    (dict(ever_tested=False), Unknown.UNTESTED, UnknownAction.RUN_EXPERIMENT),
    (dict(asked_context="bear"), Unknown.UNTESTED, UnknownAction.RUN_EXPERIMENT),
    (dict(support=0.7, against=0.6), Unknown.CONFLICTED, UnknownAction.RUN_EXPERIMENT),
    (dict(n_events=5, n_eff=5.0), Unknown.INSUFFICIENT_DATA, UnknownAction.COLLECT_DATA),
    (dict(n_events=0, n_eff=0.0), Unknown.INSUFFICIENT_DATA, UnknownAction.COLLECT_DATA),
    (dict(n_events=400, n_eff=10.0), Unknown.INSUFFICIENT_DATA, UnknownAction.COLLECT_DATA),
    (dict(evidence_age_days=900, max_age_days=365), Unknown.INSUFFICIENT_DATA, UnknownAction.COLLECT_DATA),
    (dict(missing_features=("vix",)), Unknown.UNKNOWN, UnknownAction.USE_GENERAL_RULE),
])
def test_classification_and_action(kw, state, action):
    r = uk.classify("K1", facts(**kw), NOW, AV)
    assert r is not None and r.state is state and r.action is action and r.check() == []


def test_well_known_item_is_not_unknown_and_actions_degrade_to_abstain():
    assert uk.classify("K1", facts(), NOW, AV) is None
    assert uk.classify("K1", facts(asked_context="bull"), NOW, AV) is None
    bare = uk.Availability()
    for kw, state in [(dict(ever_tested=False), Unknown.UNTESTED), (dict(support=.7, against=.6), Unknown.CONFLICTED),
                      (dict(n_events=3, n_eff=3.0), Unknown.INSUFFICIENT_DATA), (dict(missing_features=("x",)), Unknown.UNKNOWN)]:
        r = uk.classify("K1", facts(**kw), NOW, bare)
        assert r.state is state and r.action is UnknownAction.ABSTAIN
    # a conflict is never settled by a fallback rule even when one exists
    only_rule = uk.Availability(general_rule="baseline_rank")
    assert uk.classify("K1", facts(support=.7, against=.6), NOW, only_rule).action is UnknownAction.ABSTAIN
    assert uk.classify("K1", facts(n_events=3, n_eff=3.0), NOW, only_rule).action is UnknownAction.USE_GENERAL_RULE
    assert uk.classify("K1", facts(ever_tested=False), NOW, uk.Availability(can_collect_data=True)).action is UnknownAction.ABSTAIN


def test_bad_facts_are_rejected_and_illegal_action_pairs_are_invalid():
    with pytest.raises(kn.SchemaError):
        uk.classify("K1", facts(n_events=5, n_eff=9.0), NOW)
    with pytest.raises(kn.SchemaError):
        uk.classify("K1", facts(support=1.5), NOW)
    bad = uk.UnknownRecord("K1", Unknown.CONFLICTED, (uk.UnknownReason.EVIDENCE_CONFLICT,), UnknownAction.USE_GENERAL_RULE, NOW)
    assert any("cannot be handled" in e for e in bad.check())
    assert uk.UnknownRecord("", Unknown.UNKNOWN, (), UnknownAction.ABSTAIN, "x").check()


def test_unknown_can_never_be_read_as_a_number():
    rec = uk.classify("K1", facts(ever_tested=False), NOW, AV)
    with pytest.raises(uk.UnknownAsConfidence):
        float(rec)
    s = uk.UnknownScore(unknown=rec)
    with pytest.raises(uk.UnknownAsConfidence):
        float(s)
    with pytest.raises(uk.UnknownAsConfidence):
        s.require()
    assert s.or_action() is UnknownAction.RUN_EXPERIMENT and not s.is_known
    assert uk.UnknownScore(value=0.7).or_action() == 0.7 and float(uk.UnknownScore(value=0.7)) == 0.7
    assert uk.as_confidence(rec) == Confidence() and uk.as_confidence(s) == Confidence() and uk.as_confidence(Unknown.UNKNOWN) == Confidence()
    assert uk.as_confidence(Confidence(truth=0.4)).truth == 0.4
    with pytest.raises(uk.UnknownAsConfidence):
        uk.as_confidence(uk.UnknownScore(value=0.5))
    with pytest.raises(TypeError):
        uk.as_confidence(0.5)
    for both in ({}, {"value": 0.5, "unknown": rec}, {"value": float("nan")}):
        with pytest.raises(kn.SchemaError):
            uk.UnknownScore(**both)


def test_combine_never_imputes_unknowns():
    rec = uk.classify("K1", facts(ever_tested=False), NOW, AV)
    U = uk.UnknownScore(unknown=rec)
    K = lambda v: uk.UnknownScore(value=v)
    got, cov = uk.combine([K(0.9), K(0.7), U], min_coverage=0.6)
    assert got.value == pytest.approx(0.8) and cov == pytest.approx(2 / 3)                # the unknown did not drag it to 0.5 or 0
    got, cov = uk.combine([K(0.9), U, U], min_coverage=0.6)
    assert not got.is_known and got.unknown.state is Unknown.UNTESTED and cov == pytest.approx(1 / 3)
    got, _ = uk.combine([K(0.9), U], [3.0, 1.0], min_coverage=0.6)
    assert got.value == 0.9                                                                # weighted coverage 0.75
    conflict = uk.classify("K2", facts(support=.7, against=.6), NOW, AV)
    got, _ = uk.combine([uk.UnknownScore(unknown=conflict)] * 2)
    assert got.unknown.state is Unknown.CONFLICTED
    assert uk.combine([])[0].unknown.state is Unknown.INSUFFICIENT_DATA and uk.combine([])[1] == 0.0
    with pytest.raises(kn.SchemaError):
        uk.combine([K(1.0)], [1.0, 2.0])


def test_ledger_lifecycle_time_order_and_resolution_evidence():
    led = uk.UnknownLedger()
    assert len(led) == 0 and led.open_rows() == [] and led.counts() == {} and led.plan()["ABSTAIN"] == []
    a = led.open(uk.classify("A", facts(ever_tested=False), "2020-01-05", AV))
    assert led.open(uk.classify("A", facts(ever_tested=False), "2020-03-01", AV)) is a           # same unknown: keep the first `since`
    led.open(uk.classify("B", facts(n_events=2, n_eff=2.0), "2020-02-01", uk.Availability()))
    assert led.counts() == {"INSUFFICIENT_DATA": 1, "UNTESTED": 1}
    plan = led.plan()
    assert plan["RUN_EXPERIMENT"] == ["A"] and plan["ABSTAIN"] == ["B"]
    assert [r.subject for r in led.stuck("2020-05-01", 30)] == ["A", "B"] and led.stuck("2020-01-06", 30) == []
    with pytest.raises(FirewallBreach):
        led.resolve("A", "2020-06-01", "2020-06-01", "tested")                                  # evidence not before now
    with pytest.raises(KeyError):
        led.resolve("Z", "2020-06-01", "2020-05-01", "tested")
    with pytest.raises(kn.SchemaError):
        led.resolve("A", "2020-06-01", "2020-05-01", " ")
    led.resolve("A", "2020-06-01", "2020-05-01", "experiment E12 confirmed in bear")
    assert [r.subject for r in led.open_rows()] == ["B"] and len(led) == 3
    with pytest.raises(kn.SchemaError):
        led.open(uk.classify("C", facts(ever_tested=False), "2019-01-01", AV))                  # goes back in time
    back = uk.UnknownLedger.from_dict(led.to_dict())
    assert back.rows() == led.rows()
    with pytest.raises(kn.SchemaError):
        uk.UnknownLedger.from_dict({"schema": 2, "rows": []})


def test_reassess_moves_or_closes_an_unknown():
    led = uk.UnknownLedger()
    led.open(uk.classify("A", facts(n_events=5, n_eff=5.0), "2020-01-05", AV))
    moved = led.reassess("A", facts(support=.7, against=.6), "2020-02-01", "2020-01-31", AV)
    assert moved.state is Unknown.CONFLICTED and [r.subject for r in led.open_rows()] == ["A"]
    assert led.reassess("A", facts(), "2020-03-01", "2020-02-28", AV) is None and led.open_rows() == []
    with pytest.raises(FirewallBreach):
        led.reassess("A", facts(), "2020-03-01", "2020-03-01", AV)


def test_confidence_laundering_audit_finds_planted_cases():
    K = kn.KnowledgeObject
    p = kn.make_provenance("2020-01-01", code_hash="h")
    clean = K("K-ok", "2020-01-02", p, evidence=kn.Evidence(50, 40.0), confidence=Confidence(truth=0.8), epistemic=Epistemic.HYPOTHESIS)
    nothing = K("K-none", "2020-01-02", p, confidence=Confidence(truth=0.7))                         # confidence with zero evidence
    coin = K("K-coin", "2020-01-02", p, confidence=Confidence(truth=0.5))                            # disguised unknown
    import types
    unk = types.SimpleNamespace(knowledge_id="K-unk", epistemic=Epistemic.UNKNOWN, confidence=Confidence(truth=0.6),
                                evidence=types.SimpleNamespace(effective_sample_size=80.0))
    findings = uk.audit_confidence_laundering([clean, nothing, coin, unk, types.SimpleNamespace(knowledge_id="x")])
    assert not any("K-ok" in f for f in findings)
    assert any("K-none" in f and "zero effective evidence" in f for f in findings)
    assert any("K-coin" in f and "0.5 placeholder" in f for f in findings)
    assert any("K-unk" in f and "UNKNOWN" in f for f in findings)
    assert uk.audit_confidence_laundering([]) == []


def test_assess_knowledge_uses_applicability_and_evidence():
    p = kn.make_provenance("2020-01-01", code_hash="h")
    k = kn.KnowledgeObject("K-a", "2020-01-02", p, contexts=kn.context_from_text("volatility: vix >= 20"),
                           evidence=kn.Evidence(200, 100.0), confidence=Confidence(truth=0.9), epistemic=Epistemic.HYPOTHESIS)
    assert uk.assess_knowledge(k, {"vix": 25}, NOW) is None
    miss = uk.assess_knowledge(k, {}, NOW, avail=AV)
    assert miss.state is Unknown.UNKNOWN and uk.UnknownReason.CONTEXT_FEATURE_MISSING in miss.reasons
    thin = dataclasses.replace(k, evidence=kn.Evidence(4, 4.0))
    assert uk.assess_knowledge(thin, {"vix": 25}, NOW, avail=AV).state is Unknown.INSUFFICIENT_DATA
    untested = dataclasses.replace(k, confidence=Confidence())
    assert uk.assess_knowledge(untested, {"vix": 25}, NOW, avail=AV).state is Unknown.UNTESTED
    rep = uk.coverage_report([k, thin, untested], NOW, {"vix": 25})
    assert rep["total"] == 3 and rep["counts"] == {"INSUFFICIENT_DATA": 1, "KNOWN": 1, "UNTESTED": 1}
    assert uk.coverage_report([], NOW, {})["known_share"] is None


def test_rank_unknowns_budget_and_explain():
    led = uk.UnknownLedger()
    led.open(uk.classify("cheap_big", facts(ever_tested=False), "2020-01-01", AV))
    led.open(uk.classify("costly", facts(ever_tested=False), "2020-01-02", AV))
    led.open(uk.classify("abstain", facts(ever_tested=False), "2020-01-03", uk.Availability()))
    led.open(uk.classify("nostake", facts(ever_tested=False), "2020-01-04", AV))
    stakes = {"cheap_big": uk.Stake("cheap_big", 10, 0.02, 1.0), "costly": uk.Stake("costly", 10, 0.02, 20.0),
              "abstain": uk.Stake("abstain", 100, 0.5, 1.0)}
    order = [s for s, _ in uk.rank_unknowns(led.open_rows(), stakes, "2020-02-01")]
    assert order[:2] == ["cheap_big", "costly"] and set(order[2:]) == {"abstain", "nostake"}   # ABSTAIN rows and unstaked rank last
    assert dict(uk.rank_unknowns(led.open_rows(), stakes, "2020-02-01"))["nostake"] == 0.0
    b = uk.AbstentionBudget(max_share=0.3, window=10)
    assert b.assess([])["ok"] and b.assess([False] * 40)["worst_window"] == 0.0
    burst = [False] * 30 + [True] * 6 + [False] * 4
    res = b.assess(burst)
    assert not res["ok"] and res["worst_window"] == 0.6 and "abstained on 60%" in res["note"]
    assert uk.AbstentionBudget(max_share=0).check()
    assert "UNTESTED because never tested" in uk.explain(led.open_rows()[0])
