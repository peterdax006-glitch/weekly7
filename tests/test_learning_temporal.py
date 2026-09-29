"""Tests for engine/learning/retirement.py, temporal.py, surprise.py (contract C62 sections 13, 14, 52; checklist B11, C07, J06, J10).
Synthetic data only. Every mechanism has a planted case it must catch, plus the empty case."""
import dataclasses
import datetime as dt
import math

import numpy as np
import pytest

from engine.learning import retirement as R
from engine.learning import surprise as S
from engine.learning import temporal as T
from engine.learning.core import FailureCause, FirewallBreach, Health, Lifecycle, TemporalClass as TC

NOW = dt.date(2027, 1, 1)


def day(i):
    return (dt.date(2020, 1, 1) + dt.timedelta(days=int(i))).isoformat()


# ------------------------------------------------------------------------------------------------- retirement

def ev(n=40, effect=0.02, se=0.004, start="2024-03-01", end="2024-09-30"):
    return R.Evidence(n, effect, se, start, end)


def ledger():
    led = R.RetirementLedger()
    led.register("k1", "2024-01-01")
    return led


def test_register_and_state_visibility_is_strictly_after():
    led = ledger()
    assert led.state("k1", "2024-01-01") is None        # decided on the 1st, known from the 2nd
    assert led.state("k1", "2024-01-02") is R.State.ACTIVE
    assert led.influence("k1", "2024-01-01") == 0.0     # unknown to the ledger => no influence
    assert led.influence("k1", "2024-06-01") == 1.0
    with pytest.raises(ValueError):
        led.register("k1", "2024-02-01")                # re-registration would rewrite history


def test_weak_evidence_degrades_then_dormant_then_zero_influence():
    led = ledger()
    v = led.evaluate("k1", ev(effect=0.001), "2024-10-01", apply=True)
    assert v.to_state is R.State.DEGRADED and led.state("k1", "2024-10-02") is R.State.DEGRADED
    assert led.influence("k1", "2024-10-02") == led.policy.degraded_influence
    assert led.influence("k1", "2024-10-02", reliability=0.5) == pytest.approx(led.policy.degraded_influence * 0.5)
    # still weak after the grace period -> DORMANT
    v2 = led.evaluate("k1", ev(effect=0.001, end="2025-01-30"), "2025-02-01", apply=True)
    assert v2.to_state is R.State.DORMANT
    assert led.influence("k1", "2025-02-05") == 0.0
    with pytest.raises(FirewallBreach):
        led.assert_clean({"k1": 0.3}, "2025-02-05")
    led.assert_clean({"k1": 0.0}, "2025-02-05")          # zero weight is fine


def test_thin_evidence_never_changes_state():
    led = ledger()
    v = led.evaluate("k1", ev(n=5, effect=-0.05), "2024-10-01", apply=True)
    assert not v.changes and "insufficient" in v.reason
    assert led.state("k1", "2024-10-02") is R.State.ACTIVE and len(led.history("k1")) == 1


def test_false_pattern_is_retired_outright_but_kept():
    led = ledger()
    v = led.evaluate("k1", ev(effect=-0.01), "2024-10-01", cause=FailureCause.FALSE_PATTERN, apply=True)
    assert v.to_state is R.State.RETIRED
    assert led.influence("k1", "2024-10-05") == 0.0
    assert len(led.history("k1")) == 2                   # nothing deleted
    assert led.evaluate("k1", ev(), "2024-11-01").reason.startswith("already retired")


def test_reversal_with_regime_cause_parks_with_recovery_condition():
    led = ledger()
    cond = R.RecoveryCondition("regime", "bull returns", equals="bull")
    v = led.evaluate("k1", ev(effect=-0.02), "2024-10-01", cause=FailureCause.REGIME_CHANGE, apply=True, conditions=[cond])
    assert v.to_state is R.State.DORMANT
    assert led.conditions("k1", "2024-10-05") == (cond,)


def test_illegal_moves_and_flapping_guard():
    led = ledger()
    with pytest.raises(ValueError):
        led.transition("k1", R.State.ACTIVE, "2024-03-01", "RECOVER_FULL", "nope")       # ACTIVE -> ACTIVE
    led.transition("k1", R.State.DEGRADED, "2024-03-01", "DEGRADE", "x")
    with pytest.raises(ValueError, match="flapping"):
        led.transition("k1", R.State.DORMANT, "2024-03-02", "DORMANT", "too soon")
    with pytest.raises(FirewallBreach):
        led.transition("k1", R.State.DORMANT, "2023-12-01", "DORMANT", "back-dated")
    with pytest.raises(KeyError):
        led.transition("zzz", R.State.DORMANT, "2024-03-10", "DORMANT", "x")


def park(led, kid="k1", at="2024-06-01", conds=()):
    led.transition(kid, R.State.DORMANT, at, "DORMANT", "parked", FailureCause.REGIME_CHANGE, conditions=conds)


def test_recovery_needs_new_evidence_not_the_old():
    led = ledger()
    park(led)
    old = ev(n=60, effect=0.03, start="2024-01-15", end="2024-05-20")               # before it went dormant
    v = led.attempt_recovery("k1", old, "2024-09-01")
    assert not v.changes and "old evidence" in v.reason
    new = ev(n=60, effect=0.03, start="2024-06-10", end="2024-08-30")
    assert led.attempt_recovery("k1", new, "2024-09-01").to_state is R.State.DEGRADED


def test_recovery_is_hysteretic_and_two_stage():
    led = ledger()
    park(led)
    weak = ev(n=60, effect=0.006, se=0.004, start="2024-06-10", end="2024-08-30")   # t=1.5: above degrade_t, below recover_t
    assert not led.attempt_recovery("k1", weak, "2024-09-01").changes
    good = ev(n=60, effect=0.03, start="2024-06-10", end="2024-08-30")
    v = led.attempt_recovery("k1", good, "2024-09-01", apply=True)
    assert v.kind == "RECOVER_PROBATION" and led.state("k1", "2024-09-02") is R.State.DEGRADED
    # a second, independent window is needed for ACTIVE; the first window is old evidence now
    again = led.attempt_recovery("k1", good, "2024-12-01")
    assert not again.changes
    second = ev(n=60, effect=0.03, start="2024-09-10", end="2024-11-25")
    full = led.attempt_recovery("k1", second, "2024-12-01", apply=True)
    assert full.to_state is R.State.ACTIVE and led.influence("k1", "2024-12-02") == 1.0


def test_recovery_outside_stated_condition_needs_more_t():
    led = ledger()
    cond = R.RecoveryCondition("regime", "bull", equals="bull")
    park(led, conds=[cond])
    borderline = ev(n=60, effect=0.0088, se=0.004, start="2024-06-10", end="2024-08-30")   # t=2.2: passes 2.0, fails 2.5
    assert not led.attempt_recovery("k1", borderline, "2024-09-01", context={"regime": "bear"}).changes
    assert led.attempt_recovery("k1", borderline, "2024-09-01", context={"regime": "bull"}).changes


def test_retired_revival_is_stricter_than_dormant_recovery():
    led = ledger()
    led.transition("k1", R.State.RETIRED, "2024-06-01", "RETIRE", "false", FailureCause.FALSE_PATTERN)
    ok_for_dormant = ev(n=40, effect=0.03, start="2024-06-10", end="2024-08-30")           # n < 30*2
    assert not led.attempt_recovery("k1", ok_for_dormant, "2024-09-01").changes
    strong = ev(n=70, effect=0.03, start="2024-06-10", end="2024-08-30")
    v = led.attempt_recovery("k1", strong, "2024-09-01")
    assert v.kind == "REVIVE" and v.to_state is R.State.DEGRADED


def test_probation_failure_reverts_to_dormant():
    led = ledger()
    park(led)
    led.attempt_recovery("k1", ev(n=60, effect=0.03, start="2024-06-10", end="2024-08-30"), "2024-09-01", apply=True)
    bad = ev(n=40, effect=0.0, start="2024-09-10", end="2024-11-25")
    v = led.probation_failed("k1", bad, "2024-12-01", apply=True)
    assert v.kind == "REVERT" and led.state("k1", "2024-12-02") is R.State.DORMANT


def test_evidence_after_now_fails_closed():
    led = ledger()
    with pytest.raises(FirewallBreach):
        led.evaluate("k1", ev(end="2024-10-05"), "2024-10-01")


def test_series_evidence_uses_only_the_open_window():
    dates = [day(i) for i in range(400)]
    vals = [0.01] * 400
    e = R.series_evidence(dates, vals, since=day(100), now=day(300))
    assert e.n == 199 and e.window_start == day(101) and e.window_end == day(299)     # both ends exclusive
    assert R.series_evidence([], [], "2024-01-01", "2024-06-01").n == 0


def test_chain_detects_tampering_and_roundtrip(tmp_path):
    led = ledger()
    led.evaluate("k1", ev(effect=-0.02), "2024-10-01", apply=True)
    assert led.verify_chain() == []
    path = tmp_path / "ledger.jsonl"
    led.dump(path)
    again = R.RetirementLedger.load(path)
    assert again.state("k1", "2025-01-01") == led.state("k1", "2025-01-01") and again.verify_chain() == []
    led._log[1] = dataclasses.replace(led._log[1], reason="edited after the fact")
    assert led.verify_chain()
    text = path.read_text(encoding="utf-8").replace("weak", "x").replace("degrade", "degradE")
    path.write_text(text.replace(led._log[0].id, "0" * 20), encoding="utf-8")
    with pytest.raises(FirewallBreach):
        R.RetirementLedger.load(path)


def test_recovery_index_wakes_only_matching_conditions():
    led = R.RetirementLedger()
    for k in ("a", "b", "c"):
        led.register(k, "2024-01-01")
    led.transition("a", R.State.DORMANT, "2024-03-01", "DORMANT", "x", conditions=[R.RecoveryCondition("regime", equals="bull")])
    led.transition("b", R.State.DORMANT, "2024-03-01", "DORMANT", "x", conditions=[R.RecoveryCondition("vix", lo=25, hi=80)])
    led.transition("c", R.State.DORMANT, "2024-03-01", "DORMANT", "x")
    idx = R.RecoveryIndex(led)
    assert [c.knowledge_id for c in idx.candidates({"regime": "bull", "vix": 12}, "2024-09-01")] == ["a"]
    assert [c.knowledge_id for c in idx.candidates({"vix": 40}, "2024-09-01")] == ["b"]
    assert idx.candidates({}, "2024-09-01") == []                                  # unknown context wakes nothing
    assert idx.unconditioned("2024-09-01") == ["c"]
    assert idx.by_condition_key("2024-09-01") == {"regime": ["a"], "vix": ["b"]}


def test_empty_ledger_and_helpers():
    led = R.RetirementLedger()
    assert led.health("2024-01-01") is Health.INSUFFICIENT_EVIDENCE and led.counts("2024-01-01")["ACTIVE"] == 0
    assert led.live_ids("2024-01-01") == () and led.verify_chain() == []
    assert R.pattern_states_covered() == []                                         # map is in sync with pattern_lifecycle.STATES
    assert R.adapt_pattern_state("watch") is R.State.DEGRADED and R.adapt_pattern_state("candidate") is None
    with pytest.raises(ValueError):
        R.adapt_pattern_state("bogus")
    assert R.State.DORMANT.lifecycle is Lifecycle.DORMANT
    assert R.RetirementPolicy(recover_t=0.5).validate()                             # hysteresis violated
    assert R.RecoveryCondition("x").validate() and not R.RecoveryCondition("x", equals="a").validate()


def test_dormant_cannot_influence_any_decision_effect():
    from engine.learning.core import DecisionEffect as DE
    assert R.decision_effect_allowed(R.State.ACTIVE, DE.POSITION_SIZE)
    assert not R.decision_effect_allowed(R.State.DEGRADED, DE.POSITION_SIZE)
    assert R.decision_effect_allowed(R.State.DORMANT, DE.RESEARCH_PRIORITY)
    assert not R.decision_effect_allowed(R.State.RETIRED, DE.SELECTION)


def test_batch_evaluate_is_deterministic_and_reports_every_item():
    led = R.RetirementLedger()
    for k in ("b", "a"):
        led.register(k, "2024-01-01")
    out = R.batch_evaluate(led, {"b": ev(effect=0.001), "a": ev()}, "2024-10-01", apply=True)
    assert [v.knowledge_id for v in out] == ["a", "b"] and R.states_summary(out) == {"NO_CHANGE": 1, "DEGRADED": 1}


# ------------------------------------------------------------------------------------------------- temporal

def series(effect_fn, n=60, period=30, se=0.004, seed=1, regimes=None, events=None):
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * period
    vals = [effect_fn(i, t) + rng.normal(0, se) for i, t in enumerate(ts)]
    return T.EffectSeries(tuple(day(t + 30) for t in ts), tuple(vals), tuple([se] * n), regimes, events)


def klass(s, now=NOW):
    return T.estimate("x", s, now)


def test_persistent_effect_is_persistent_with_a_lifetime_lower_bound():
    p = klass(series(lambda i, t: 0.02))
    assert p.klass == TC.PERSISTENT.value and p.lifetime_days is None
    assert p.lifetime_lo and p.lifetime_lo > 500                    # data cannot rule out decay any faster than this
    assert not p.validate()


def test_fast_decay_is_found_with_a_tau_near_the_truth():
    p = klass(series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48))
    assert p.klass == TC.FAST_DECAY.value
    assert 45 < p.tau_days < 180 and p.lifetime_days is not None and p.lifetime_lo <= p.lifetime_days <= p.lifetime_hi


def test_slow_decay_distinguished_from_fast_by_estimate_not_assumption():
    p = klass(series(lambda i, t: 0.03 * math.exp(-t / 700.0), n=60))
    assert p.klass == TC.SLOW_DECAY.value and p.tau_days > 300


def test_decay_to_a_floor_keeps_unbounded_lifetime():
    p = klass(series(lambda i, t: 0.02 + 0.03 * math.exp(-t / 150.0), n=60))
    assert p.klass in (TC.FAST_DECAY.value, TC.SLOW_DECAY.value)
    assert p.winner == "decay_floor" and p.lifetime_days is None


def test_regime_bound_effect_learns_the_regime_as_recovery_condition():
    regs = tuple("bull" if (i // 6) % 2 == 0 else "bear" for i in range(60))
    p = klass(series(lambda i, t: 0.03 if regs[i] == "bull" else 0.0, regimes=regs))
    assert p.klass == TC.REGIME_BOUND.value and p.active_groups == ("bull",)
    cond = p.recovery[0]
    assert cond.satisfied_by({"regime": "bull"}) and not cond.satisfied_by({"regime": "bear"}) and not cond.satisfied_by({})
    assert p.lifetime_days and 100 < p.lifetime_days < 260           # regimes last ~6 bins of 30 days


def test_event_bound_effect():
    ev_flags = tuple(bool(i % 5 == 0) for i in range(60))
    p = klass(series(lambda i, t: 0.04 if ev_flags[i] else 0.0, events=ev_flags))
    assert p.klass == TC.EVENT_BOUND.value and p.recovery[0].key == "event"


def test_seasonal_effect_with_months_recovery_window():
    def f(i, t):
        m = (dt.date(2020, 1, 31) + dt.timedelta(days=int(t))).month
        return 0.03 if m in (11, 12, 1) else 0.0
    p = klass(series(f, n=60))
    assert p.klass == TC.SEASONAL.value
    assert set(p.active_groups) == {"01", "11", "12"}
    assert any(c.satisfied_by({"month": 12}) for c in p.recovery) and not any(c.satisfied_by({"month": 6}) for c in p.recovery)


def test_episodic_bursts_over_a_quiet_baseline():
    on = {5, 6, 7, 8, 25, 26, 27, 28, 45, 46, 47, 48}
    p = klass(series(lambda i, t: 0.04 if i in on else 0.0, seed=3))
    assert p.klass == TC.EPISODIC.value and p.burst_days and p.gap_days and p.gap_days > p.burst_days


def test_noise_only_is_unknown_not_persistent():
    p = klass(series(lambda i, t: 0.0, seed=5))
    assert p.klass == TC.UNKNOWN.value and "no detectable effect" in p.reason
    assert not p.validate()


def test_too_little_evidence_is_unknown_and_says_so():
    p = klass(series(lambda i, t: 0.05, n=6))
    assert p.klass == TC.UNKNOWN.value and "insufficient" in p.reason
    empty = klass(T.EffectSeries((), (), ()))
    assert empty.klass == TC.UNKNOWN.value and empty.n_bins == 0


def test_negative_effects_are_oriented_not_lost():
    p = klass(series(lambda i, t: -0.02))
    assert p.klass == TC.PERSISTENT.value


def test_series_with_a_date_at_or_after_now_fails_closed():
    s = series(lambda i, t: 0.02)
    with pytest.raises(FirewallBreach):
        T.estimate("x", s, s.dates[-1])
    T.estimate("x", s, dt.date.fromisoformat(s.dates[-1]) + dt.timedelta(days=1))


def test_bad_series_rejected():
    with pytest.raises(ValueError):
        T.estimate("x", T.EffectSeries(("2020-02-01", "2020-01-01"), (0.1, 0.1)), NOW)
    with pytest.raises(ValueError):
        T.estimate("x", T.EffectSeries(("2020-01-01",), (0.1, 0.2)), NOW)


def test_estimate_is_deterministic():
    s = series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48)
    assert klass(s).profile_id == klass(s).profile_id


def test_from_outcomes_bins_and_drops_the_future():
    rng = np.random.default_rng(0)
    dates = [day(i) for i in range(0, 400)]
    vals = rng.normal(0.01, 0.02, 400)
    s = T.from_outcomes(dates, vals, now=day(300), period_days=30)
    assert s.dates and dt.date.fromisoformat(s.dates[-1]) < dt.date.fromisoformat(day(300))
    assert len(s.dates) == len(s.values) == len(s.ses) and 8 <= len(s.dates) <= 11
    assert T.from_outcomes([], [], NOW).dates == ()


def test_evidence_weights_follow_the_learned_class():
    dec = klass(series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48))
    obs = [dt.date(2026, 12, 1), dt.date(2026, 6, 1), dt.date(2025, 6, 1)]
    w = T.evidence_weights(dec, obs, NOW)
    assert w[0] > w[1] > w[2] and w[0] <= 1.0
    per = klass(series(lambda i, t: 0.02))
    assert np.allclose(T.evidence_weights(per, obs, NOW), 1.0)
    with pytest.raises(FirewallBreach):
        T.evidence_weights(per, [NOW], NOW)
    regs = tuple("bull" if (i // 6) % 2 == 0 else "bear" for i in range(60))
    rb = klass(series(lambda i, t: 0.03 if regs[i] == "bull" else 0.0, regimes=regs))
    w2 = T.evidence_weights(rb, obs, NOW, ["bull", "bear", "bull"], "bull")
    assert w2[0] == 1.0 and w2[1] < 0.1 and w2[2] == 1.0


def test_expected_influence_gates_conditional_classes():
    regs = tuple("bull" if (i // 6) % 2 == 0 else "bear" for i in range(60))
    rb = klass(series(lambda i, t: 0.03 if regs[i] == "bull" else 0.0, regimes=regs))
    assert T.expected_influence(rb, NOW, {"regime": "bull"}) == 1.0 and T.expected_influence(rb, NOW, {"regime": "bear"}) == 0.0
    fast = klass(series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48))
    later = NOW + dt.timedelta(days=3000)
    assert T.expected_influence(fast, later) < 0.05 < T.expected_influence(fast, NOW + dt.timedelta(days=1)) + 1
    unk = klass(series(lambda i, t: 0.0, seed=5))
    assert T.expected_influence(unk, NOW) == 0.5                     # unknown is neither zero nor one


def test_temporal_memory_versions_never_overwrite_and_reads_are_strictly_past():
    mem = T.TemporalMemory()
    p1 = T.estimate("x", series(lambda i, t: 0.02), NOW)
    mem.add(p1)
    later = NOW + dt.timedelta(days=200)
    dec = series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=60)
    p2 = T.estimate("x", dec, later)
    mem.add(p2)
    assert mem.get("x", NOW) is None                                 # profile dated NOW is not yet visible on NOW
    assert mem.get("x", NOW + dt.timedelta(days=1)).klass == TC.PERSISTENT.value
    assert mem.get("x", later + dt.timedelta(days=1)).klass != TC.PERSISTENT.value
    assert len(mem.history("x")) == 2 and mem.reclassified("x")[0][1] == TC.PERSISTENT.value
    with pytest.raises(FirewallBreach):
        mem.add(p1)                                                  # older than the newest version
    assert mem.class_counts(later + dt.timedelta(days=1))
    assert T.TemporalMemory().due(NOW) == [] and T.TemporalMemory().class_counts(NOW) == {}


def test_review_due_and_family_lifetimes():
    fast = klass(series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48))
    assert not T.needs_review(fast, NOW) and T.needs_review(fast, NOW + dt.timedelta(days=4000))
    mem = T.TemporalMemory()
    mem.add(dataclasses.replace(fast, knowledge_id="mom_a"))
    mem.add(dataclasses.replace(fast, knowledge_id="mom_b", lifetime_days=300.0))
    fam = mem.family_lifetimes(NOW + dt.timedelta(days=1), lambda k: k.split("_")[0])
    assert fam["mom"]["n"] == 2
    assert "IMPLEMENTED - NOT VALIDATED" in T.describe(fast)


def test_config_validation():
    assert T.TemporalConfig(min_bins=3).validate() and T.TemporalConfig(useful_fraction=1.5).validate()
    with pytest.raises(ValueError):
        T.estimate("x", series(lambda i, t: 0.02), NOW, T.TemporalConfig(min_bins=3))


# ------------------------------------------------------------------------------------------------- surprise

def tracker_with(rng, cells=8, per=25, bad_cell="cell3", bias=1.6):
    tr = S.SurpriseTracker()
    for c in range(cells):
        name = f"cell{c}"
        for i in range(per):
            d = dt.date(2024, 1, 1) + dt.timedelta(days=7 * i + c)
            shift = bias if name == bad_cell else 0.0
            actual = rng.normal(shift, 1.0)
            tr.observe(name, 0.0, actual, d, d + dt.timedelta(days=5), NOW, scale=1.0)
    return tr


def test_planted_persistent_surprise_cell_is_flagged_and_ranked_first():
    tr = tracker_with(np.random.default_rng(11))
    inv = tr.investigations(NOW)
    assert any(i.cell == "cell3" and "bias" in i.reason for i in inv)
    top = tr.research_priority(NOW)[0]
    assert top.cell == "cell3" and top.components["repetition"] > 0.5
    flagged_noise = {i.cell for i in inv if "bias" in i.reason} - {"cell3"}
    assert len(flagged_noise) <= 1                                   # FDR control: noise cells are not flagged by the dozen


def test_single_high_surprise_opens_investigation():
    tr = S.SurpriseTracker()
    r = tr.observe("c", 0.0, -6.0, "2024-01-01", "2024-01-08", NOW, scale=1.0)
    assert r.direction == -1 and r.magnitude == 6.0 and r.bits > 20
    inv = tr.investigations(NOW)
    assert inv and inv[0].record_ids == (r.record_id,) and "single surprise" in inv[0].reason
    assert tr.investigations(NOW, since="2024-01-08") == []          # already reviewed


def test_outcome_not_yet_matured_fails_closed():
    tr = S.SurpriseTracker()
    with pytest.raises(FirewallBreach):
        tr.observe("c", 0.0, 1.0, "2026-12-20", NOW, NOW, scale=1.0)
    tr.observe("c", 0.0, 1.0, "2026-12-20", "2026-12-31", NOW, scale=1.0)
    assert tr.records("2026-12-31") == []                            # matured that very day: unknown on that day
    assert len(tr.records(NOW)) == 1


def test_scale_uses_only_errors_matured_before_the_decision():
    tr = S.SurpriseTracker(S.SurpriseConfig(min_history=12))
    rng = np.random.default_rng(2)
    for i in range(30):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=i * 3)
        tr.observe("c", 0.0, rng.normal(0, 2.0), d, d + dt.timedelta(days=1), NOW, scale=2.0)
    sc, src = tr.scale_at("2024-03-01", "c")
    assert src == "history" and 1.0 < sc < 3.5
    assert tr.scale_at("2024-01-05", "c") == (1.0, "default")         # too little PAST history at that date
    huge = S.SurpriseTracker(S.SurpriseConfig(min_history=12))
    for i in range(30):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=i * 3)
        huge.observe("c", 0.0, 1000.0 if d >= dt.date(2024, 3, 1) else rng.normal(0, 1), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    assert huge.scale_at("2024-02-20", "c")[0] < 3.0                  # later blown-up errors cannot leak backwards


def test_duplicate_and_bad_inputs_rejected():
    tr = S.SurpriseTracker()
    tr.observe("c", 0.0, 1.0, "2024-01-01", "2024-01-08", NOW, scale=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        tr.observe("c", 0.0, 1.0, "2024-01-01", "2024-01-08", NOW, scale=1.0)
    with pytest.raises(ValueError):
        tr.observe("", 0.0, 1.0, "2024-01-01", "2024-01-08", NOW, scale=1.0)
    with pytest.raises(ValueError):
        tr.observe("c", 0.0, float("nan"), "2024-01-01", "2024-01-09", NOW, scale=1.0)
    with pytest.raises(ValueError):
        tr.observe("c", 0.0, 1.0, "2024-01-10", "2024-01-08", NOW, scale=1.0)   # decided after it matured
    with pytest.raises(ValueError):
        S.binary_z(0.5, 2)


def test_binary_surprise_confident_wrong_call_is_large():
    assert abs(S.binary_z(0.95, 0)) > 4 > abs(S.binary_z(0.55, 0))
    tr = S.SurpriseTracker()
    r = tr.observe_binary("c", 0.95, 0, "2024-01-01", "2024-01-08", NOW)
    assert r.kind == "binary" and r.direction == -1


def test_neighbour_cells_add_to_priority():
    tr = tracker_with(np.random.default_rng(11))
    base = {p.cell: p.score for p in tr.research_priority(NOW)}
    boosted = {p.cell: p.score for p in tr.research_priority(NOW, {"cell5": [("cell3", 1.0)]})}
    assert boosted["cell5"] >= base.get("cell5", 0.0) and boosted.get("cell5", 0) > 0
    assert boosted["cell3"] == base["cell3"]


def test_recency_lowers_priority_of_old_surprises():
    rng = np.random.default_rng(4)
    tr = S.SurpriseTracker()
    for name, start in (("old", dt.date(2022, 1, 1)), ("new", dt.date(2026, 6, 1))):
        for i in range(12):
            d = start + dt.timedelta(days=7 * i)
            tr.observe(name, 0.0, 3.0 + rng.normal(0, 0.1), d, d + dt.timedelta(days=2), NOW, scale=1.0)
    pr = {p.cell: p.score for p in tr.research_priority(NOW)}
    assert pr["new"] > 10 * pr["old"]


def test_monitor_flags_a_surprise_storm_and_stays_quiet_on_noise():
    rng = np.random.default_rng(6)
    calm, storm = S.SurpriseTracker(), S.SurpriseTracker()
    for i in range(300):
        d = NOW - dt.timedelta(days=300 - i + 1)
        calm.observe("c", 0.0, rng.normal(0, 1), d, d + dt.timedelta(days=1), NOW, scale=1.0)
        sd = 1.0 if (NOW - d).days > 90 else 3.0
        storm.observe("c", 0.0, rng.normal(0, sd), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    assert calm.monitor(NOW)["health"] == Health.HEALTHY.value
    m = storm.monitor(NOW)
    assert m["health"] in (Health.DEGRADING.value, Health.UNSTABLE.value) and m["big_rate"] > 0.3


def test_empty_tracker_and_report(tmp_path):
    tr = S.SurpriseTracker()
    assert tr.monitor(NOW)["health"] == Health.INSUFFICIENT_EVIDENCE.value
    assert tr.investigations(NOW) == [] and tr.research_priority(NOW) == [] and tr.cell_stats("x", NOW) is None
    assert S.benjamini_hochberg([], 0.1) == []
    tr2 = tracker_with(np.random.default_rng(11))
    assert "NOT VALIDATED" in tr2.report(NOW)
    path = tmp_path / "s.jsonl"
    tr2.dump(path)
    back = S.SurpriseTracker.load(path)
    assert len(back) == len(tr2) and back.research_priority(NOW)[0].cell == "cell3"


def test_benjamini_hochberg_known_answer():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert S.benjamini_hochberg(p, 0.05) == [True, True] + [False] * 8
    assert sum(S.benjamini_hochberg(p, 0.30)) >= 5


def test_config_validation_and_by_knowledge():
    assert S.SurpriseConfig(z_high=1.0, z_big=2.0).validate()
    with pytest.raises(ValueError):
        S.SurpriseTracker(S.SurpriseConfig(default_scale=0))
    tr = S.SurpriseTracker()
    tr.observe("c", 0.0, -3.0, "2024-01-01", "2024-01-08", NOW, scale=1.0, knowledge_ids=["k9"])
    assert tr.by_knowledge(NOW)["k9"]["mean_z"] == -3.0


# ------------------------------------------------------------------------------------------------- forecasting, recovery, lifetimes

def test_predicted_effect_follows_the_fitted_class():
    dec = klass(series(lambda i, t: 0.05 * math.exp(-t / 120.0), n=48))
    early, late = T.predicted_effect(dec, "2020-02-01"), T.predicted_effect(dec, "2030-01-01")
    assert early > 0.03 and abs(late) < 0.005 and early > late
    per = klass(series(lambda i, t: -0.02))
    assert T.predicted_effect(per, NOW) == pytest.approx(-0.02, abs=0.004)            # sign convention preserved
    regs = tuple("bull" if (i // 6) % 2 == 0 else "bear" for i in range(60))
    rb = klass(series(lambda i, t: 0.03 if regs[i] == "bull" else 0.0, regimes=regs))
    assert T.predicted_effect(rb, NOW, {"regime": "bull"}) > 0.02 > T.predicted_effect(rb, NOW, {"regime": "bear"})
    assert T.predicted_effect(klass(series(lambda i, t: 0.0, seed=5)), NOW) is None


def _decayed_then(after_fn, n_after=12, seed=3):
    base = series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48)
    prof = T.estimate("x", base, NOW)
    rng = np.random.default_rng(seed)
    dates, vals = [], []
    for j in range(n_after):
        d = dt.date.fromisoformat(prof.as_of) + dt.timedelta(days=30 * (j + 1))
        dates.append(d.isoformat())
        vals.append(after_fn(j) + rng.normal(0, 0.004))
    later = NOW + dt.timedelta(days=30 * (n_after + 2))
    return prof, T.EffectSeries(tuple(dates), tuple(vals), tuple([0.004] * n_after)), later


def test_recovery_test_separates_a_real_comeback_from_the_same_decay():
    prof, comeback, later = _decayed_then(lambda j: 0.03)
    assert T.recovery_test(prof, comeback, later).recovered
    prof, still_dead, later = _decayed_then(lambda j: 0.0)
    r = T.recovery_test(prof, still_dead, later)
    assert not r.recovered and "not significantly positive" in r.reason
    tiny = T.EffectSeries(comeback.dates[:2], comeback.values[:2], comeback.ses[:2])
    assert not T.recovery_test(prof, tiny, later).recovered


def test_recovery_test_ignores_evidence_before_the_profile_date():
    prof, comeback, later = _decayed_then(lambda j: 0.03)
    old = T.EffectSeries(tuple(day(30 * i) for i in range(1, 9)), tuple([0.04] * 8), tuple([0.004] * 8))
    assert not T.recovery_test(prof, old, later).recovered
    with pytest.raises(FirewallBreach):
        T.recovery_test(prof, comeback, comeback.dates[-1])


def test_to_evidence_feeds_the_retirement_gate():
    s = series(lambda i, t: 0.02, n=20)
    e = T.to_evidence(s, since=s.dates[2], now=NOW)
    assert e.n == 17 and e.effect == pytest.approx(0.02, abs=0.004) and e.se < 0.004 and not e.validate()
    assert T.to_evidence(T.EffectSeries((), (), ()), "2020-01-01", NOW).n == 0


def test_kaplan_meier_known_answer_with_censoring():
    km = T.kaplan_meier([2, 3, 3, 5, 8], [True, True, False, True, False])
    assert km.times == (2.0, 3.0, 5.0)
    assert km.survival == pytest.approx((0.8, 0.6, 0.3))
    assert km.at(1) == 1.0 and km.at(4) == pytest.approx(0.6) and km.median() == 5.0 and km.quantile(0.9) is None
    assert km.n == 5 and km.n_events == 3 and all(s >= 0 for s in km.se)
    allc = T.kaplan_meier([5, 6], [False, False])
    assert allc.times == () and allc.median() is None and allc.at(100) == 1.0        # nobody failed: no lifetime claimed
    with pytest.raises(ValueError):
        T.kaplan_meier([1, -1], [True, True])
    with pytest.raises(ValueError):
        T.kaplan_meier([1], [True, False])


def test_profile_lifetimes_censors_items_still_working():
    mem = T.TemporalMemory()
    fast = klass(series(lambda i, t: 0.05 * math.exp(-t / 60.0), n=48))
    per = klass(series(lambda i, t: 0.02))
    mem.add(dataclasses.replace(fast, knowledge_id="dead", lifetime_days=200.0))
    mem.add(dataclasses.replace(per, knowledge_id="alive"))
    born = {"dead": "2020-01-01", "alive": "2020-01-01"}
    dur, ended = T.profile_lifetimes(mem, NOW + dt.timedelta(days=1), born)
    assert ended == [False, True] and dur[1] == 200.0 and dur[0] > 2000                # sorted by id: alive, dead
    assert T.profile_lifetimes(T.TemporalMemory(), NOW, born) == ([], [])


def test_class_stability_uses_only_data_before_each_cut():
    s = series(lambda i, t: 0.02, n=60)
    st = T.class_stability("x", s, NOW)
    assert st["final"] == TC.PERSISTENT.value and st["stable"] and st["flips"] <= 1
    for cut_date, _ in st["sequence"]:
        assert cut_date <= (dt.date.fromisoformat(s.dates[-1]) + dt.timedelta(days=1)).isoformat()
    assert len({c for c, _ in st["sequence"]}) == len(st["sequence"])
    changing = series(lambda i, t: 0.05 * math.exp(-t / 90.0) if i < 30 else 0.0, n=60)
    seq = T.class_stability("y", changing, NOW)["sequence"]
    assert seq and seq[0][1] != TC.PERSISTENT.value                                    # early data alone already shows the fade
    assert T.class_stability("z", T.EffectSeries((), (), ()), NOW)["sequence"] == []


def test_profile_table_is_sorted_and_flat():
    a = klass(series(lambda i, t: 0.02))
    rows = T.profile_table([dataclasses.replace(a, knowledge_id="b"), dataclasses.replace(a, knowledge_id="a")])
    assert [r["id"] for r in rows] == ["a", "b"] and rows[0]["class"] == TC.PERSISTENT.value
    assert T.profile_table([]) == []


# ------------------------------------------------------------------------------------------------- surprise: similarity, charts, log

def test_cell_similarity_and_auto_neighbours():
    assert S.cell_similarity("a=1|b=2|c=3", "a=1|b=2|c=3") == 1.0
    assert S.cell_similarity("a=1|b=2", "a=1|b=9") == pytest.approx(1 / 3)
    assert S.cell_similarity("", "") == 0.0 and S.cell_similarity("a=1", "b=1") == 0.0
    tr = S.SurpriseTracker()
    for c in ("vol=hi|trend=up", "vol=hi|trend=dn", "vol=lo|trend=up|size=s"):
        tr.observe(c, 0.0, 1.0, "2024-01-01", "2024-01-05", NOW, scale=1.0)
    nb = S.auto_similar(tr, NOW, min_sim=0.2)
    assert [c for c, _ in nb["vol=hi|trend=up"]] == ["vol=hi|trend=dn", "vol=lo|trend=up|size=s"]
    assert "vol=hi|trend=up" not in S.auto_similar(tr, NOW, min_sim=0.9)


def test_planted_surprise_in_similar_cells_raises_neighbour_priority_automatically():
    rng = np.random.default_rng(31)
    tr = S.SurpriseTracker()
    cells = {"vol=hi|trend=up": 2.5, "vol=hi|trend=dn": 0.0, "vol=lo|trend=up": 0.0}
    for c, shift in cells.items():
        for i in range(20):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=7 * i)
            tr.observe(c, 0.0, rng.normal(shift, 1.0), d, d + dt.timedelta(days=2), NOW, scale=1.0)
    plain = {p.cell: p.score for p in tr.research_priority(NOW)}
    auto = {p.cell: p.score for p in tr.research_priority(NOW, S.auto_similar(tr, NOW, 0.3))}
    assert auto["vol=hi|trend=dn"] > plain.get("vol=hi|trend=dn", 0.0)
    assert max(auto, key=auto.get) == "vol=hi|trend=up"


def test_rollup_pools_cells_by_dropping_a_dimension():
    tr = S.SurpriseTracker()
    for size in ("s", "m", "l"):
        for i in range(8):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=7 * i)
            tr.observe(f"vol=hi|size={size}", 0.0, 1.2, d, d + dt.timedelta(days=2), NOW, scale=1.0, knowledge_ids=[f"k{size}{i}"])
    parents = S.rollup(tr, NOW, "size")
    assert list(parents) == ["vol=hi"] and parents["vol=hi"].n == 24 and parents["vol=hi"].bias_p < 1e-6
    assert S.rollup(S.SurpriseTracker(), NOW, "size") == {}


def test_ewma_chart_catches_a_slow_rise_in_surprise():
    rng = np.random.default_rng(32)
    tr = S.SurpriseTracker()
    for i in range(80):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=4 * i)
        sd = 1.0 if i < 50 else 2.2
        tr.observe("c", 0.0, rng.normal(0, sd), d, d + dt.timedelta(days=1), NOW, scale=1.0)
        tr.observe("calm", 0.0, rng.normal(0, 1.0), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    chart = S.ewma_chart(tr, "c", NOW)
    assert chart["alarm_index"] is not None and chart["alarm_index"] >= 45 and len(chart["ewma"]) == 80
    assert S.ewma_chart(tr, "calm", NOW)["alarm_index"] is None
    assert S.ewma_chart(tr, "nothing", NOW)["limit"] is None
    with pytest.raises(ValueError):
        S.ewma_chart(tr, "c", NOW, lam=0)


def test_expectation_skill_tells_informative_expectations_from_constant_ones():
    rng = np.random.default_rng(33)
    good, flat = S.SurpriseTracker(), S.SurpriseTracker()
    for i in range(60):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=3 * i)
        e = rng.normal(0, 1)
        good.observe("c", e, e + rng.normal(0, 0.5), d, d + dt.timedelta(days=1), NOW, scale=1.0)
        flat.observe("c", 0.1, e + rng.normal(0, 0.5), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    g = S.expectation_skill(good, NOW)
    assert 0.8 < g["slope"] < 1.2 and g["corr"] > 0.8
    assert math.isnan(S.expectation_skill(flat, NOW)["slope"])                      # constant expectation carries no skill
    assert S.expectation_skill(S.SurpriseTracker(), NOW)["n"] == 0


def test_direction_table_shows_tilt():
    tr = tracker_with(np.random.default_rng(11))
    rows = S.direction_table(tr, NOW)
    assert rows[0]["cell"] == "cell3" and rows[0]["tilt"] > 0.5 and len(rows) == 8


def test_persistence_profile_flags_a_long_same_direction_run():
    tr = S.SurpriseTracker()
    for i, z in enumerate([3, 3, -3, 3, 3, 3, 3, 3, 3, 3, 3, 0.1, 0.2]):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=7 * i)
        tr.observe("c", 0.0, float(z), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    pp = S.persistence_profile(tr, "c", NOW)
    assert pp["longest_run"] == 8 and pp["persistent"] and pp["direction"] == 1 and pp["big"] == 11
    assert not S.persistence_profile(tracker_with(np.random.default_rng(2), bias=0.0), "cell1", NOW)["persistent"]
    assert S.persistence_profile(S.SurpriseTracker(), "x", NOW)["longest_run"] == 0
    assert S.summary_table(S.SurpriseTracker(), NOW) == []


def test_discriminating_tokens_finds_the_context_that_carries_the_surprise():
    rng = np.random.default_rng(34)
    tr = S.SurpriseTracker()
    for vol in ("hi", "lo"):
        for trend in ("up", "dn", "flat"):
            for size in ("s", "l"):
                shift = 2.0 if vol == "hi" else 0.0
                for i in range(12):
                    d = dt.date(2025, 1, 1) + dt.timedelta(days=5 * i)
                    tr.observe(f"vol={vol}|trend={trend}|size={size}", 0.0, rng.normal(shift, 1.0) * rng.choice([-1, 1]), d,
                               d + dt.timedelta(days=1), NOW, scale=1.0)
    rows = S.discriminating_tokens(tr, NOW)
    assert rows[0]["token"] in ("vol=hi", "vol=lo") and rows[0]["p"] < 0.001
    q = S.research_questions(tr, NOW, top=3)
    assert q and all("cell" in r for r in q) and any(r["suggested_split"] == "vol=hi" for r in q)
    assert S.discriminating_tokens(S.SurpriseTracker(), NOW) == [] and S.research_questions(S.SurpriseTracker(), NOW) == []


def test_investigation_log_is_append_only_and_prevents_reflagging_noise():
    tr = tracker_with(np.random.default_rng(11))
    log = S.InvestigationLog()
    first = log.pending(tr, NOW)
    assert any(i.cell == "cell3" for i in first)
    log.record("cell3", "EXPLAINED", "2026-06-01", "regime shift in volatility")
    assert log.current("cell3", "2026-06-01") is None and log.current("cell3", "2026-06-02").state == "EXPLAINED"
    assert all(i.cell != "cell3" or "re-flagged" in i.reason for i in log.pending(tr, NOW))
    with pytest.raises(ValueError):
        log.record("cell3", "BOGUS", "2026-07-01")
    with pytest.raises(FirewallBreach):
        log.record("cell3", "OPEN", "2026-01-01")
    for d in ("2026-07-01", "2026-08-01"):
        log.record("cell3", "NO_CAUSE_FOUND", d)
    assert log.repeat_offenders(3) == ["cell3"] and len(log) == 3
    assert S.InvestigationLog().last_reviewed("x", NOW) is None


# ------------------------------------------------------------------------------------------------- S08 second pass (section 83)

def _monthly(effect_fn, years=5, seed=1, se=0.004):
    rng = np.random.default_rng(seed)
    ds, vs = [], []
    for k in range(years * 12):
        d = dt.date(2020 + k // 12, k % 12 + 1, 15)
        ds.append(d.isoformat())
        vs.append(effect_fn(d) + rng.normal(0, se))
    return T.EffectSeries(tuple(ds), tuple(vs), tuple([se] * len(ds)))


def test_seasonal_legitimacy_accepts_a_recurring_season():
    s = _monthly(lambda d: 0.03 if d.month in (11, 12) else 0.0)
    r = T.seasonal_legitimacy(s, [11, 12])
    assert r["legitimate"] and r["years_seen"] == 5 and r["years_positive"] == 5 and r["loo_min_t"] > 2


def test_seasonal_legitimacy_rejects_a_one_off_calendar_event():
    s = _monthly(lambda d: 0.06 if (d.year, d.month) in ((2022, 11), (2022, 12)) else 0.0)
    r = T.seasonal_legitimacy(s, [11, 12])
    assert not r["legitimate"] and ("single year" in r["reason"] or "same sign" in r["reason"])


def test_seasonal_legitimacy_artefact_removal_and_guards():
    s = _monthly(lambda d: 0.05 if d.month == 3 and d.year in (2021, 2022) else (0.05 if (d.year, d.month) == (2023, 3) else 0.0))
    assert not T.seasonal_legitimacy(s, [3], min_share=0.9, known_artefacts=[(2023, 3)])["legitimate"]
    assert not T.seasonal_legitimacy(s, list(range(1, 12)))["legitimate"]                    # ten+ months is not a season
    assert "distinct years" in T.seasonal_legitimacy(_monthly(lambda d: 0.03, years=2), [3])["reason"]


def test_bootstrap_lifetime_reproduces_a_clear_class_and_reports_spread():
    s = series(lambda i, t: 0.05 * math.exp(-t / 90.0), n=48)
    prof = T.estimate("x", s, NOW)
    b = T.bootstrap_lifetime(prof, s, NOW, np.random.default_rng(0), n_boot=40)
    assert b["p_same_class"] > 0.7 and b["n_decay"] >= 20 and b["q05"] <= b["q50"] <= b["q95"]
    assert T.bootstrap_lifetime(prof, s, NOW, np.random.default_rng(0), n_boot=40) == b        # seeded
    with pytest.raises(ValueError):
        T.bootstrap_lifetime(T.estimate("z", series(lambda i, t: 0.0, seed=5), NOW), s, NOW, np.random.default_rng(0))


def test_failure_context_separates_regime_from_event_failures():
    n = 60
    regs = tuple("bull" if (i // 10) % 2 == 0 else "bear" for i in range(n))
    slow = np.array([0.0 if r == "bull" else 1.0 for r in regs]) + np.random.default_rng(1).normal(0, 0.05, n)
    s_reg = series(lambda i, t: 0.03 if regs[i] == "bull" else 0.0, n=n, regimes=regs)
    r = T.failure_context_profile(s_reg, {"m_vix": slow, "junk": slow})
    assert r["verdict"] == TC.REGIME_BOUND.value and r["ctx_feature"] == "m_vix" and r["mean_failure_run"] >= 5
    ev = tuple(bool(i % 7 == 3) for i in range(n))
    s_ev = series(lambda i, t: 0.0 if ev[i] else 0.03, n=n, events=ev, seed=4)
    noise = np.random.default_rng(2).normal(0, 1, n)
    e = T.failure_context_profile(s_ev, {"m_noise": noise}, fail_quantile=0.15)
    assert e["verdict"] == TC.EVENT_BOUND.value and e["mean_failure_run"] < 2.5
    flat = T.failure_context_profile(series(lambda i, t: 0.03, n=n, seed=6), {"m_noise": noise})
    assert flat["verdict"] == TC.UNKNOWN.value


def _cluster_tracker(seed=51):
    rng = np.random.default_rng(seed)
    tr = S.SurpriseTracker()
    for c, persist in (("vol=hi|trend=up", True), ("vol=hi|trend=dn", True), ("size=xl|liq=lo", False)):
        level = 0.0
        for i in range(60):
            d = dt.date(2025, 1, 1) + dt.timedelta(days=5 * i)
            level = 0.9 * level + rng.normal(0, 1) if persist else rng.normal(0, 1)
            tr.observe(c, 0.0, -abs(level) * 0.0 + level - (1.0 if persist else 0.0), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    return tr


def test_cluster_cells_groups_similar_situations_only():
    tr = _cluster_tracker()
    cl = S.cluster_cells(tr, NOW, 0.3)
    assert ["vol=hi|trend=dn", "vol=hi|trend=up"] in cl and ["size=xl|liq=lo"] in cl
    assert S.cluster_cells(S.SurpriseTracker(), NOW) == []


def test_surprise_half_life_recovers_planted_persistence_and_refuses_noise():
    rng = np.random.default_rng(52)
    n, tau = 400, 30.0
    days = np.arange(n) * 5.0
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = math.exp(-5.0 / tau) * z[i - 1] + rng.normal(0, 1)
    hl = S.surprise_half_life(z, days)
    assert hl["half_life"] is not None and 10 < hl["half_life"] < 60          # true tau*ln2 = 20.8
    assert S.surprise_half_life(rng.normal(0, 1, n), days)["half_life"] is None
    assert S.surprise_half_life([1, 2, 3], [1, 2, 3])["half_life"] is None


def test_cluster_persistence_reports_pooled_half_life():
    tr = _cluster_tracker()
    cl = S.cluster_persistence(tr, ["vol=hi|trend=dn", "vol=hi|trend=up"], NOW)
    assert cl["n"] == 120 and cl["mean_z"] < -0.3 and cl["half_life"] is not None


def test_failure_handoff_maps_structure_to_a_cause_hypothesis():
    from engine.learning.core import FailureCause as FC
    tr = _cluster_tracker()
    hs = {tuple(h.cluster): h for h in S.failure_handoffs(tr, NOW, 0.3)}
    adverse = hs[("vol=hi|trend=dn", "vol=hi|trend=up")]
    assert adverse.cause in (FC.REGIME_CHANGE.value, FC.TEMPORARY_INACTIVITY.value) and adverse.direction == -1 and adverse.record_ids
    assert hs[("size=xl|liq=lo",)].cause == FC.UNKNOWN.value                   # noise: no cause invented
    thin = S.SurpriseTracker()
    thin.observe("a=1", 0.0, 3.0, "2025-01-01", "2025-01-05", NOW, scale=1.0)
    assert S.failure_handoffs(thin, NOW)[0].cause == FC.INSUFFICIENT_EVIDENCE.value
    two = S.SurpriseTracker()
    rng = np.random.default_rng(53)
    for i in range(40):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=4 * i)
        two.observe("a=1", 0.0, 3.0 * (1 if i % 2 else -1) + rng.normal(0, 0.3), d, d + dt.timedelta(days=1), NOW, scale=1.0)
    assert S.failure_handoffs(two, NOW)[0].cause == FC.WRONG_CONTEXT.value
    assert S.failure_handoffs(S.SurpriseTracker(), NOW) == []
