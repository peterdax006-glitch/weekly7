"""Tests for engine.learning belief updating, the three questions, knowledge competition and the complexity penalty
(contract C62 sections 32, 33, 39, 40).  Synthetic data only.  Every mechanism has a planted case it must catch and an empty case.
IMPLEMENTED - NOT VALIDATED: these prove the code does what it says on planted worlds, not that it learns on real data."""
import dataclasses as dc
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import belief as B
from engine.learning import competition as C
from engine.learning import complexity as X
from engine.learning import questions as Q
from engine.learning.boundary import ScopeStatus
from engine.learning.core import Epistemic, FirewallBreach, Health, Unknown, canonical_json

NOHL = B.BeliefConfig(half_life_days=None)


def ev(subject="s", when="2020-01-01", est=0.03, se=0.01, n=50, **kw):
    return B.Evidence(subject, when, est, se, n, **kw)


def ledger(cfg=NOHL):
    led = B.BeliefLedger(cfg)
    led.register("s", 0.0, cfg.prior_sd)
    return led


# ============================================================================================ belief: proportionality
def test_evidence_moves_belief_in_proportion_to_quality():
    r = B.proportionality_check()
    assert r["copy"] == 0.0                              # a pure copy (independence 0) carries no weight: planted, must not move it
    assert 0.0 < r["half"] < r["full"]                   # half quality moves it less than full quality
    assert r["full"] < 0.04                              # and never past the evidence itself (shrinkage toward the prior)


def test_in_sample_and_heavily_searched_evidence_is_discounted():
    a, b, c = ledger(), ledger(), ledger()
    a.update("s", ev(kind="OOS_TEST"), "2021-01-01")
    b.update("s", ev(kind="IN_SAMPLE_FIT"), "2021-01-01")
    c.update("s", ev(kind="IN_SAMPLE_FIT", n_trials=500), "2021-01-01")
    ma, mb, mc = (x.current("s").mean for x in (a, b, c))
    assert ma > mb > mc > 0


def test_more_precise_evidence_dominates_noisy_evidence():
    led = ledger()
    led.update("s", [ev(when="2020-01-01", est=0.10, se=0.20), ev(when="2020-02-01", est=0.02, se=0.005)], "2021-01-01")
    st = led.current("s")
    assert abs(st.mean - 0.02) < 0.005                   # the precise record wins, the noisy one barely registers


def test_posterior_is_independent_of_arrival_order():
    rng = np.random.default_rng(1)
    recs = [ev(when=f"2020-{m:02d}-01", est=float(rng.normal(0.02, 0.01)), se=0.01) for m in range(1, 9)]
    a, b = ledger(), ledger()
    a.update("s", recs, "2021-01-01")
    b.update("s", list(reversed(recs)), "2021-01-01")
    assert a.current("s").mean == pytest.approx(b.current("s").mean, abs=1e-12)
    assert a.current("s").sd == pytest.approx(b.current("s").sd, abs=1e-12)


def test_sequential_updates_equal_batch_update():
    recs = [ev(when=f"2020-{m:02d}-01", est=0.01 * m, se=0.02) for m in range(1, 6)]
    a, b = ledger(), ledger()
    for i, r in enumerate(recs):
        a.update("s", r, f"2020-{i + 2:02d}-15")
    b.update("s", recs, "2020-06-15")
    assert a.current("s").mean == pytest.approx(b.current("s").mean, abs=1e-12)


def test_posterior_credible_interval_is_calibrated():
    r = B.posterior_coverage(n_sims=800, seed=4)
    assert abs(r["coverage"] - r["target"]) < 0.04


# ============================================================================================ belief: never overwrite
def test_history_is_append_only_and_states_are_immutable():
    led = ledger()
    s1 = led.update("s", ev(when="2020-01-01"), "2020-02-01")
    h1 = s1.state_hash
    s2 = led.update("s", ev(when="2020-03-01", est=0.05), "2020-04-01")
    assert len(led.history("s")) == 2 and led.history("s")[0].state_hash == h1
    assert s2.parent == h1 and s2.version == 2
    with pytest.raises(dc.FrozenInstanceError):
        s1.mean = 9.9
    assert led.verify_integrity() == []


def test_duplicate_evidence_is_a_noop_not_a_double_count():
    led = ledger()
    led.update("s", ev(), "2021-01-01")
    n_before = len(led.history("s"))
    led.update("s", ev(), "2021-06-01")
    assert len(led.history("s")) == n_before and led.duplicates == 1
    assert led.current("s").n_evidence == 1


def test_tampered_state_is_detected():
    led = ledger()
    led.update("s", ev(), "2021-01-01")
    led.update("s", ev(when="2020-05-01", est=0.05), "2021-02-01")
    led._states["s"][0] = dc.replace(led._states["s"][0], mean=0.5)     # planted corruption
    errs = led.verify_integrity()
    assert errs and any("replay" in e or "parent" in e for e in errs)


def test_updates_cannot_be_back_dated():
    led = ledger()
    led.update("s", ev(when="2020-01-01"), "2021-01-01")
    with pytest.raises(B.BeliefError):
        led.update("s", ev(when="2020-02-01", est=0.04), "2020-06-01")


def test_prior_is_immutable_once_registered():
    led = ledger()
    with pytest.raises(B.BeliefError):
        led.register("s", 0.5, 0.05)
    led.register("s", 0.0, NOHL.prior_sd)                # identical re-registration is fine


# ============================================================================================ belief: firewall
def test_future_evidence_is_rejected_and_changes_nothing():
    led = ledger()
    led.update("s", ev(when="2020-01-01"), "2020-06-01")
    before = led.current("s").state_hash
    with pytest.raises(FirewallBreach):
        led.update("s", [ev(when="2020-03-01", est=0.02), ev(when="2020-06-01", est=0.9)], "2020-06-01")   # second matures ON now
    assert led.current("s").state_hash == before and len(led.evidence_for("s")) == 1


def test_view_at_past_date_uses_only_past_evidence():
    led = ledger()
    led.update("s", ev(when="2020-01-01", est=0.01), "2020-02-01")
    led.update("s", ev(when="2020-03-01", est=0.09), "2020-04-01")
    v = led.view("s", "2020-02-15")
    assert v.n_evidence == 1 and v.mean < 0.02


def test_evidence_from_future_dated_series_raises():
    idx = pd.date_range("2020-01-06", periods=30, freq="W-MON")
    s = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 30), index=idx)
    with pytest.raises(FirewallBreach):
        B.evidence_from_returns("s", s, idx[-1])         # last observation is dated ON now
    assert B.evidence_from_returns("s", s, idx[-1] + pd.Timedelta(days=1)) is not None


# ============================================================================================ belief: contradiction
def test_contradicting_evidence_is_counted_and_widens_uncertainty():
    led = ledger()
    for m in range(1, 6):
        led.update("s", ev(when=f"2020-{m:02d}-01", est=0.03, se=0.01), f"2020-{m:02d}-20")
    calm = led.current("s")
    led.update("s", ev(when="2020-08-01", est=-0.04, se=0.008), "2020-09-01")
    hit = led.current("s")
    assert calm.n_contra == 0 and hit.n_contra == 1
    assert hit.contradiction_mass > 0.05
    assert hit.tau2 > calm.tau2                           # heterogeneity is priced into the pool
    assert hit.mean < calm.mean
    # the same evidence WITHOUT contradiction would have left the belief tighter
    ctrl = ledger()
    for m in range(1, 6):
        ctrl.update("s", ev(when=f"2020-{m:02d}-01", est=0.03, se=0.01), f"2020-{m:02d}-20")
    ctrl.update("s", ev(when="2020-08-01", est=0.03, se=0.008), "2020-09-01")
    assert hit.sd > ctrl.current("s").sd


def test_epistemic_status_reflects_contradiction_and_data_volume():
    led = ledger()
    assert B.belief_status(led.view("s", "2020-01-01"))[1] == Unknown.UNTESTED
    for m in range(1, 6):
        led.update("s", ev(when=f"2020-{m:02d}-01", est=0.03, se=0.008), f"2020-{m:02d}-20")
    assert B.belief_status(led.current("s"))[0] == Epistemic.SUPPORTED
    led.update("s", [ev(when="2020-07-01", est=-0.05, se=0.008), ev(when="2020-08-01", est=-0.05, se=0.008),
                     ev(when="2020-09-01", est=-0.05, se=0.008)], "2020-10-01")
    epi, why = B.belief_status(led.current("s"))
    assert epi in (Epistemic.CONTRADICTED, Epistemic.CONDITIONAL) and why == Unknown.CONFLICTED


def test_update_records_log_surprise_and_evidence_strength():
    led = ledger()
    led.update("s", [ev(when=f"2020-0{m}-01", est=0.03, se=0.01) for m in range(1, 5)], "2020-06-01")
    led.update("s", ev(when="2020-07-01", est=-0.06, se=0.01), "2020-08-01")
    u = led.updates("s")[-1]
    assert u.contradiction and abs(u.predictive_z) > 2 and u.log_bayes_factor < 0
    rows = B.contradiction_report(led, "s")
    assert len(rows) == 1 and rows[0]["estimate"] == pytest.approx(-0.06)
    sup = led.updates("s")[0]
    assert not sup.contradiction


def test_heterogeneity_explained_by_a_context_tag():
    led = ledger()
    for m in range(1, 7):
        led.update("s", ev(when=f"2019-{m:02d}-01", est=0.04, se=0.01, context={"regime": "calm"}), f"2019-{m:02d}-20")
        led.update("s", ev(when=f"2019-{m:02d}-02", est=-0.03, se=0.01, context={"regime": "stress"}), f"2019-{m:02d}-21")
    rows = B.explain_heterogeneity(led, "s")
    assert rows and rows[0]["key"] == "regime" and rows[0]["explains"]
    led2 = ledger()                                      # control: same records, tag unrelated to sign
    for m in range(1, 7):
        led2.update("s", ev(when=f"2019-{m:02d}-01", est=0.04, se=0.01, context={"regime": "x"}), f"2019-{m:02d}-20")
        led2.update("s", ev(when=f"2019-{m:02d}-02", est=0.04, se=0.01, context={"regime": "y"}), f"2019-{m:02d}-21")
    r2 = B.explain_heterogeneity(led2, "s")
    assert not r2 or not r2[0]["explains"]


# ============================================================================================ belief: time and memory
def test_old_evidence_loses_weight_and_belief_gets_staler():
    led = B.BeliefLedger(B.BeliefConfig(half_life_days=180.0))
    led.register("s", 0.0, 0.05)
    led.update("s", [ev(when="2019-01-01", est=0.04, se=0.01)], "2019-02-01")
    now_sd = led.current("s").sd
    later = led.view("s", "2022-01-01")
    assert later.sd > now_sd and later.mean < led.current("s").mean     # decays back toward the prior (0)
    st = B.staleness(led, "s", "2022-01-01")
    assert st["stale"] and st["days_since"] > 1000


def test_persistence_roundtrip_preserves_every_state(tmp_path):
    led = ledger()
    for m in range(1, 5):
        led.update("s", ev(when=f"2020-{m:02d}-01", est=0.01 * m), f"2020-{m:02d}-20")
    p = tmp_path / "beliefs.jsonl"
    led.save_jsonl(p)
    led2 = B.BeliefLedger.load_jsonl(p)
    assert led2.verify_integrity() == []
    assert [s.state_hash for s in led2.history("s")] == [s.state_hash for s in led.history("s")]
    assert len(led2.updates("s")) == len(led.updates("s"))


def test_merge_of_two_workers_equals_one_ledger_with_all_evidence():
    a, b, both = ledger(), ledger(), ledger()
    ra = [ev(when=f"2020-0{m}-01", est=0.02 + 0.002 * m) for m in range(1, 4)]
    rb = [ev(when=f"2020-0{m}-01", est=0.02 + 0.002 * m, source="w2") for m in range(4, 7)]
    a.update("s", ra, "2020-12-01")
    b.update("s", rb, "2020-12-01")
    both.update("s", ra + rb, "2020-12-01")
    m = B.merge_ledgers(a, b, "2020-12-01")
    assert m.current("s").mean == pytest.approx(both.current("s").mean, abs=1e-12)


def test_influence_shows_a_single_dominant_record():
    led = ledger()
    led.update("s", [ev(when="2020-01-01", est=0.03, se=0.001), ev(when="2020-02-01", est=0.01, se=0.05),
                     ev(when="2020-03-01", est=0.02, se=0.05)], "2020-06-01")
    inf = B.evidence_influence(led, "s", "2020-06-01")
    assert inf[0]["weight"] > 0.8


def test_hierarchical_shrinkage_pulls_thin_contexts_toward_the_family():
    h = B.HierarchicalBeliefs("pat", NOHL)
    rng = np.random.default_rng(3)
    for c in ("a", "b", "c", "d"):
        for m in range(1, 6):
            h.add(c, ev(when=f"2020-{m:02d}-01", est=float(rng.normal(0.02, 0.003)), se=0.004), f"2020-{m:02d}-20")
    h.add("thin", ev(when="2020-03-01", est=0.12, se=0.06), "2020-07-01")     # one wild, imprecise record
    p = h.pooled("thin", "2020-08-01")
    assert p["weight_own"] < 0.5 and abs(p["mean"] - p["family_mean"]) < abs(p["own_mean"] - p["family_mean"])
    dep = h.departures("2020-08-01")
    assert all(d["context"] != "thin" for d in dep)      # the thin context is NOT called a departure on one noisy point


def test_beta_belief_counts_are_tempered_and_validated():
    b = B.BetaBelief().updated(60, 100, quality=1.0)
    b2 = B.BetaBelief().updated(60, 100, quality=0.5)
    assert b.mean > b2.mean > 0.5 and b.strength > b2.strength
    with pytest.raises(B.BeliefError):
        B.BetaBelief().updated(11, 10)
    assert B.BetaBelief().interval()[0] < 0.5 < B.BetaBelief().interval()[1]


def test_partial_pool_shrinks_noisy_groups_more():
    r = B.partial_pool([0.02, 0.021, 0.019, 0.20], [0.002, 0.002, 0.002, 0.08])
    assert r["weight_own"][3] < r["weight_own"][0]
    assert abs(r["shrunk"][3] - r["mu"]) < abs(0.20 - r["mu"])


def test_e_value_monitor_rejects_planted_effect_but_not_null():
    rng = np.random.default_rng(0)
    real, null = B.EValueMonitor(0.02), B.EValueMonitor(0.02)
    for _ in range(40):
        real.add(float(rng.normal(0.02, 0.02)), 0.02)
        null.add(float(rng.normal(0.0, 0.02)), 0.02)
    assert real.rejects and not null.rejects
    with pytest.raises(B.BeliefError):
        real.add(0.1, 0.0)


def test_predictive_calibration_flags_an_overconfident_belief():
    rng = np.random.default_rng(2)
    good = B.BeliefLedger(NOHL)
    good.register("s", 0.0, 0.05)
    over = B.BeliefLedger(B.BeliefConfig(half_life_days=None, prior_sd=0.001))
    over.register("s", 0.0, 0.001)
    for i in range(60):
        true = float(rng.normal(0, 0.05))
        d = (pd.Timestamp("2015-01-01") + pd.Timedelta(days=30 * i)).date().isoformat()
        e = B.Evidence("s", d, float(rng.normal(true, 0.01)), 0.01, 30)
        now = (pd.Timestamp(d) + pd.Timedelta(days=5)).date().isoformat()
        good.update("s", e, now)
        over.update("s", e, now)
    assert B.PredictiveCalibration(over.updates("s")).summary()["verdict"] != "CALIBRATED"
    assert B.PredictiveCalibration([]).summary()["verdict"] == "INSUFFICIENT_DATA"


def test_evidence_from_pattern_row_splits_discovery_from_confirmation():
    row = {"key_named": "A q1 & B q4", "effect": 0.02, "t_disc": 3.5, "t_conf": 2.1}
    ev2 = B.evidence_from_pattern_row(row, "2021-01-01", n_candidates=200)
    assert [str(e.kind) for e in ev2] == ["IN_SAMPLE_FIT", "OOS_TEST"]
    assert ev2[0].quality() < ev2[1].quality()
    assert B.evidence_from_pattern_row({"key_named": "x", "effect": 0.02, "t_disc": 3.0}, "2021-01-01")[0].kind == B.EvidenceKind.IN_SAMPLE_FIT
    assert B.evidence_from_pattern_row({"key_named": "x", "effect": 0.02}, "2021-01-01") == []


def test_empty_and_degenerate_belief_inputs():
    led = ledger()
    st = led.view("s", "2020-01-01")
    assert st.n_evidence == 0 and st.mean == 0.0 and B.influence_weight(st) == 0.0
    assert B.evidence_from_returns("s", pd.Series(dtype=float), "2021-01-01") is None
    assert B.evidence_from_returns("s", pd.Series([0.1, 0.2], index=pd.to_datetime(["2020-01-06", "2020-01-13"])), "2021-01-01") is None
    bad = B.Evidence("s", "2020-01-01", 0.01, -1.0, 5)
    assert bad.validate()
    led.update("s", bad, "2020-06-01")
    assert led.rejected and led.current("s").n_evidence == 0
    assert B.conservative_effect(led.view("s", "2020-01-01")) == 0.0
    with pytest.raises(B.BeliefError):
        B.BeliefLedger(B.BeliefConfig(prior_sd=-1))


# ============================================================================================ questions
def weekly(n, mean, sd=0.02, seed=0, start="2016-01-04"):
    idx = pd.date_range(start, periods=n, freq="W-MON")
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n), index=idx)


NOW = "2030-01-01"


def rel(effect, **kw):
    return Q.RelationEvidence("r", effect, **kw)


def test_planted_real_effect_is_real_and_planted_noise_is_not():
    eng = Q.QuestionEngine()
    assert eng.ask(rel(weekly(200, 0.01)), NOW).real.verdict == Q.Answer.YES
    yes_on_noise = sum(eng.ask(rel(weekly(200, 0.0, seed=s)), NOW).real.verdict == Q.Answer.YES for s in range(1, 31))
    assert yes_on_noise <= 3                              # a check that can fail: alpha 0.05, 30 noise series


def test_multiple_testing_burden_downgrades_a_marginal_finding():
    s = weekly(150, 0.0042, seed=11)
    e = Q.QuestionEngine()
    one = e.ask(rel(s, n_trials=1), NOW).real
    many = e.ask(rel(s, n_trials=400), NOW).real
    assert many.p_final > one.p_final and many.verdict != Q.Answer.YES


def test_an_effect_in_only_one_era_is_not_called_real():
    x = pd.concat([weekly(100, 0.02, seed=5), weekly(100, -0.001, seed=6, start="2018-01-01")])
    x.index = pd.date_range("2016-01-04", periods=200, freq="W-MON")
    ans = Q.QuestionEngine().ask(rel(x), NOW).real
    assert ans.verdict != Q.Answer.YES or ans.stability >= 0.6
    y = weekly(200, 0.0, seed=8)
    y.iloc[:50] += 0.05                                   # one lucky era, nothing after
    a2 = Q.QuestionEngine().ask(rel(y), NOW).real
    assert a2.verdict != Q.Answer.YES


def test_evidence_of_absence_needs_a_tight_interval_not_just_insignificance():
    tight = Q.QuestionEngine(Q.QuestionConfig(equiv_margin=0.01)).ask(rel(weekly(400, 0.0, sd=0.01, seed=3)), NOW).real
    assert tight.verdict == Q.Answer.NO
    wide = Q.QuestionEngine().ask(rel(weekly(25, 0.0, sd=0.05, seed=3)), NOW).real
    assert wide.verdict == Q.Answer.UNKNOWN


def test_real_but_decayed_relation_is_real_and_not_useful_now():
    x = pd.concat([weekly(180, 0.012, seed=1), weekly(20, -0.012, seed=2, start="2019-06-03")])
    x.index = pd.date_range("2016-01-04", periods=200, freq="W-MON")
    t = Q.QuestionEngine().ask(rel(x), NOW)
    assert t.real.verdict == Q.Answer.YES and t.useful_now.verdict == Q.Answer.NO
    assert Q.disposition(t).action == "RUN_EXPERIMENT"          # real, decision value untested, and not current
    w, wo = _decisions(200, 0.004)
    t2 = Q.QuestionEngine().ask(rel(x, with_decision=w, without_decision=wo, oos_start="2017-01-02"), NOW)
    assert t2.code == "YYN" and Q.disposition(t2).action == "DORMANT"


def test_regime_bound_relation_is_useful_now_only_inside_its_condition():
    x = weekly(200, 0.012, seed=4)
    ctx = {"vol": {"op": "<=", "value": 0.2}}
    inside = Q.QuestionEngine().ask(rel(x, contexts=ctx, context_now={"vol": 0.1}), NOW).useful_now
    outside = Q.QuestionEngine().ask(rel(x, contexts=ctx, context_now={"vol": 0.4}), NOW).useful_now
    missing = Q.QuestionEngine().ask(rel(x, contexts=ctx, context_now={}), NOW).useful_now
    anti = Q.QuestionEngine().ask(rel(x, anti_contexts={"vol": {"op": ">", "value": 0.3}}, context_now={"vol": 0.4}), NOW).useful_now
    assert inside.verdict == Q.Answer.YES and inside.scope == str(ScopeStatus.IN_SCOPE)
    assert outside.verdict == Q.Answer.NO and outside.scope == str(ScopeStatus.OUT_OF_SCOPE)
    assert missing.verdict == Q.Answer.UNKNOWN and missing.scope == str(ScopeStatus.UNKNOWN)   # never assumed in scope
    assert anti.verdict == Q.Answer.NO and anti.scope == str(ScopeStatus.ANTI_HIT)


def test_broken_health_overrides_recent_good_numbers():
    x = weekly(200, 0.012, seed=4)
    assert Q.QuestionEngine().ask(rel(x, health=Health.BROKEN), NOW).useful_now.verdict == Q.Answer.NO


def _decisions(n, gain, seed=0, tail=0.0):
    idx = pd.date_range("2016-01-04", periods=n, freq="W-MON")
    rng = np.random.default_rng(seed)
    base = pd.Series(rng.normal(0.004, 0.02, n), index=idx)
    with_ = base + gain + rng.normal(0, 0.003, n)
    if tail:
        with_.iloc[::10] -= tail
    return with_, base


def test_useful_requires_out_of_sample_gain_net_of_cost():
    w, wo = _decisions(150, 0.004)
    e = Q.QuestionEngine()
    yes = e.ask(rel(weekly(150, 0.01), with_decision=w, without_decision=wo, oos_start="2017-01-02"), NOW).useful
    assert yes.verdict == Q.Answer.YES and yes.gain > 0
    costly = e.ask(rel(weekly(150, 0.01), with_decision=w, without_decision=wo, oos_start="2017-01-02", cost_per_period=0.02), NOW).useful
    assert costly.verdict == Q.Answer.NO                  # the same gain, eaten by cost
    none = e.ask(rel(weekly(150, 0.01)), NOW).useful
    assert none.verdict == Q.Answer.UNKNOWN and none.unknown == Unknown.UNTESTED
    no_oos = e.ask(rel(weekly(150, 0.01), with_decision=w, without_decision=wo), NOW).useful
    assert no_oos.verdict == Q.Answer.UNKNOWN


def test_useful_is_vetoed_by_a_worse_tail_and_by_redundancy():
    w, wo = _decisions(200, 0.004, tail=0.09)
    e = Q.QuestionEngine()
    tail = e.ask(rel(weekly(200, 0.01), with_decision=w, without_decision=wo, oos_start="2016-06-01"), NOW).useful
    assert tail.verdict == Q.Answer.NO and tail.tail_change < 0
    w2, wo2 = _decisions(150, 0.004)
    red = e.ask(rel(weekly(150, 0.01), with_decision=w2, without_decision=wo2, oos_start="2017-01-02", redundancy=0.97), NOW).useful
    assert red.verdict == Q.Answer.NO and any("redundant" in r for r in red.reasons)


def test_useful_flags_misaligned_paired_series():
    w, wo = _decisions(100, 0.004)
    bad = wo.copy()
    bad.index = bad.index + pd.Timedelta(days=3)
    u = Q.QuestionEngine().ask(rel(weekly(100, 0.01), with_decision=w, without_decision=bad, oos_start="2016-06-01"), NOW).useful
    assert u.verdict == Q.Answer.UNKNOWN and u.unknown == Unknown.CONFLICTED


def test_three_answers_can_never_be_collapsed_into_one_number():
    t = Q.QuestionEngine().ask(rel(weekly(200, 0.01)), NOW)
    with pytest.raises(TypeError):
        float(t)
    with pytest.raises(TypeError):
        t < t
    with pytest.raises(AttributeError):
        t.real = None
    assert not hasattr(t, "score")
    c = t.confidence()
    assert c.truth is not None and c.usefulness is None   # useful was never tested: None, not 0


def test_different_combinations_lead_to_different_decisions():
    e = Q.QuestionEngine()
    w, wo = _decisions(150, 0.004)
    kw = dict(with_decision=w, without_decision=wo, oos_start="2017-01-02")
    use = Q.disposition(e.ask(rel(weekly(150, 0.01), **kw), NOW))
    real_not_useful = Q.disposition(e.ask(rel(weekly(150, 0.01), with_decision=wo + 0.0, without_decision=wo, oos_start="2017-01-02"), NOW))
    reject = Q.disposition(e.ask(rel(weekly(150, 0.0, seed=9)), NOW))
    dormant = Q.disposition(e.ask(rel(weekly(150, 0.01), health=Health.BROKEN, **kw), NOW))
    assert use.action == "USE" and use.size_multiplier == 1.0
    assert real_not_useful.action == "KEEP_AS_KNOWLEDGE" and real_not_useful.size_multiplier == 0.0
    assert reject.action in ("COLLECT_EVIDENCE", "REJECT")
    assert dormant.action == "DORMANT"
    assert len({use.action, real_not_useful.action, dormant.action}) == 3


def test_useful_without_real_is_quarantined():
    w, wo = _decisions(150, 0.005)
    t = Q.QuestionEngine(Q.QuestionConfig(equiv_margin=0.01)).ask(
        rel(weekly(400, 0.0, sd=0.01, seed=3), with_decision=w, without_decision=wo, oos_start="2017-01-02"), NOW)
    assert t.real.verdict == Q.Answer.NO and t.useful.verdict == Q.Answer.YES
    assert t.inconsistencies() and Q.disposition(t).action == "QUARANTINE"


def test_questions_refuse_future_data():
    s = weekly(60, 0.01)
    with pytest.raises(FirewallBreach):
        Q.QuestionEngine().ask(rel(s), s.index[-1])       # the last outcome matures ON now
    w, wo = _decisions(60, 0.003)
    with pytest.raises(FirewallBreach):
        Q.QuestionEngine().ask(rel(weekly(60, 0.01), with_decision=w, without_decision=wo, oos_start="2016-03-01"), w.index[-1])


def test_short_or_empty_history_is_unknown_not_a_guess():
    e = Q.QuestionEngine()
    t = e.ask(rel(weekly(8, 0.01)), NOW)
    assert t.real.verdict == Q.Answer.UNKNOWN and t.real.unknown == Unknown.INSUFFICIENT_DATA
    assert t.useful_now.verdict == Q.Answer.UNKNOWN
    empty = e.ask(rel(pd.Series(dtype=float)), NOW)
    assert empty.code == "UUU"
    assert Q.validate_relation(rel(pd.Series(dtype=float)))
    with pytest.raises(Q.QuestionError):
        Q.clean_series(pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-06", "2020-01-06"])), NOW, "dup")


def test_family_wise_control_tightens_the_real_verdict():
    rels = [Q.RelationEvidence(f"r{i}", weekly(120, 0.003, seed=20 + i)) for i in range(12)]
    solo = Q.QuestionEngine().ask(rels[0], NOW).real.p_final
    fam = Q.QuestionEngine().ask_many(rels, NOW)
    assert fam[0].real.p_final >= solo - 1e-12
    assert Q.QuestionEngine().ask_many([], NOW) == []


def test_answers_are_deterministic():
    a = Q.QuestionEngine().ask(rel(weekly(120, 0.006, seed=5)), NOW).to_dict()
    b = Q.QuestionEngine().ask(rel(weekly(120, 0.006, seed=5)), NOW).to_dict()
    assert canonical_json(a) == canonical_json(b)      # NaN-safe comparison


def test_walkforward_shows_when_a_planted_effect_became_detectable_and_never_before():
    x = pd.concat([weekly(120, 0.0, seed=3), weekly(200, 0.012, seed=4)])
    x.index = pd.date_range("2016-01-04", periods=320, freq="W-MON")
    onset = x.index[120]
    cuts = list(x.index[60::20])
    ans = Q.answers_walkforward(Q.RelationEvidence("r", x), cuts)
    lag = Q.detection_lag(ans, onset)
    assert lag is not None and lag >= 0                   # detected only after the onset, never before
    assert Q.time_to_verdict(ans[:3], "real") is None     # nothing to find in the null half


def test_answer_ledger_reports_decay_and_flip_rate():
    led = Q.AnswerLedger()
    e = Q.QuestionEngine()
    good = weekly(150, 0.012, seed=1)
    for cut in (good.index[80], good.index[120], good.index[149]):
        led.add(e.ask(Q.RelationEvidence("d", good[good.index < cut]), cut))
    bad = pd.concat([good[:130], weekly(20, -0.03, seed=2, start="2018-08-06")])
    bad.index = pd.date_range("2016-01-04", periods=150, freq="W-MON")
    led.add(e.ask(Q.RelationEvidence("d", bad), NOW))
    assert "d" in led.decayed() and any(t["question"] == "useful_now" for t in led.transitions("d"))
    with pytest.raises(Q.QuestionError):
        led.add(e.ask(Q.RelationEvidence("d", good[good.index < "2017-01-01"]), "2017-01-01"))
    assert 0.0 <= led.flip_rate("d") <= 1.0


def test_minimum_detectable_effect_grows_with_trials_and_noise():
    assert Q.minimum_detectable_effect(0.002, n_trials=100) > Q.minimum_detectable_effect(0.002, n_trials=1)
    assert Q.minimum_detectable_effect(0.004) == pytest.approx(2 * Q.minimum_detectable_effect(0.002))


def test_dissociation_table_and_collapse_risk():
    e = Q.QuestionEngine()
    answers = [e.ask(rel(weekly(150, m, seed=i)), NOW) for i, m in enumerate([0.01, 0.0, 0.012, 0.0, 0.009, 0.011])]
    tab = Q.dissociation_table(answers)
    assert tab["count"].sum() == len(answers)
    assert Q.dissociation_table([]).empty
    assert "n" in Q.collapse_risk(answers)
    assert isinstance(Q.format_answers(answers), str) and len(Q.answers_table(answers)) == len(answers)


# ============================================================================================ competition
def _arena(truth, n=400, seed=3, cfg=C.ArenaConfig(), extra=()):
    df = C.synthetic_world(truth, n=n, seed=seed)
    specs = C.standard_field("a", proxies=["vol"], gates=[("liq", ">", 0.0), ("trend", ">", 0.0)]) + list(extra)
    a = C.Arena("r", specs, cfg=cfg)
    C.run_arena(a, df, 20, "2030-01-01")
    return a, df, specs


@pytest.mark.parametrize("truth,winner", [("causal", "H_causal"), ("proxy", "H_proxy_vol"), ("liq", "H_cond_liq>0"),
                                          ("trend", "H_cond_trend>0")])
def test_the_planted_explanation_wins(truth, winner):
    a, _, _ = _arena(truth)
    assert a.winner() == winner and a.separated()
    assert a.weights()[winner] > 0.99


def test_no_winner_is_declared_when_nothing_is_there():
    wrong = 0
    for seed in range(6):
        a, _, _ = _arena("null", n=300, seed=40 + seed)
        w = a.winner()
        wrong += int(w not in (None, "H_null"))
    assert wrong <= 1                                     # a non-null story may rarely fluke through, never routinely


def test_hypotheses_are_kept_alive_until_evidence_separates_them():
    a, df, _ = _arena("causal", n=30)                     # too little data to separate
    assert not a.separated() and a.winner() is None
    assert all(s != C.HypStatus.ELIMINATED for s in a.status().values())
    assert a.undecided_reason()
    nd_min = a.cfg.min_disc
    for e in a.events:
        if "ELIMINATED" in e["event"]:
            n_rows = float(e["event"].split("over ")[1].split(" ")[0])
            assert n_rows >= nd_min - 1                   # no elimination on fewer discriminating rows than the bar


def test_identical_explanations_are_equivalent_not_eliminated():
    df = C.synthetic_world("causal", n=300, seed=8)
    twin = dc.replace(C.causal_spec("a"), hyp_id="H_causal_twin")
    a = C.Arena("r", [C.null_spec(), C.causal_spec("a"), twin])
    C.run_arena(a, df, 20, "2030-01-01")
    assert "H_causal_twin" in a.indistinguishable_from("H_causal") or a.leader() == "H_causal_twin"
    lead = a.leader()
    other = "H_causal_twin" if lead == "H_causal" else "H_causal"
    assert a.status()[other] == C.HypStatus.EQUIVALENT
    assert a.separated()                                  # separated from the null, equivalents grouped
    assert any(e[2].value == "REDUNDANT_WITH" for e in a.edges())


def test_predictions_are_made_before_outcomes_are_learned():
    a, df, _ = _arena("causal", n=200)
    assert C.leakage_probe(a, df.iloc[-20:]) == 0.0


def test_elimination_is_reversible_when_the_world_changes():
    part1 = C.synthetic_world("liq", n=260, seed=1)
    part2 = C.synthetic_world("causal", n=400, seed=2, start="2021-01-04")
    df = pd.concat([part1, part2])
    cfg = C.ArenaConfig(forget=0.985)
    a = C.Arena("r", C.standard_field("a", proxies=["vol"], gates=[("liq", ">", 0.0)]), cfg=cfg)
    C.run_arena(a, df, 20, "2030-01-01")
    text = " ".join(e["event"] for e in a.events)
    assert "ELIMINATED H_causal" in text and "REVIVED H_causal" in text
    assert a.leader() in ("H_causal", "H_cond_liq>0")


def test_late_entrant_is_scored_exactly_as_if_it_had_always_been_there():
    df = C.synthetic_world("liq", n=300, seed=5)
    specs = C.standard_field("a", proxies=["vol"], gates=[("liq", ">", 0.0)]) + [C.interaction_spec("a", "liq")]
    full = C.Arena("r", specs)
    C.run_arena(full, df, 20, "2030-01-01")
    late = C.Arena("r", specs[:-1])
    C.run_arena(late, df, 20, "2030-01-01")
    C.add_hypothesis(late, specs[-1])
    assert np.allclose(full._cum, late._cum) and np.allclose(full._llr, late._llr) and np.allclose(full._nd, late._nd)
    assert full.leader() == late.leader()
    with pytest.raises(C.CompetitionError):
        C.add_hypothesis(late, specs[-1])                 # duplicate id
    with pytest.raises(C.CompetitionError):
        C.add_hypothesis(late, C.causal_spec("unseen_col", "H_new"))


def test_interaction_story_wins_when_the_effect_scales_with_a_modifier():
    rng = np.random.default_rng(6)
    n = 500
    a_, liq, vol = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 0.02 * a_ * (1.0 + 0.9 * liq) + rng.normal(scale=0.02, size=n)
    df = pd.DataFrame({"a": a_, "liq": liq, "vol": vol, "y": y}, index=pd.date_range("2015-01-05", periods=n, freq="W-MON"))
    arena = C.Arena("r", [C.null_spec(), C.causal_spec("a"), C.proxy_spec("vol"), C.interaction_spec("a", "liq")])
    C.run_arena(arena, df, 25, "2030-01-01")
    assert arena.winner() == "H_int_axliq"


def test_time_rules_future_batches_and_backfills_are_refused():
    a, df, _ = _arena("causal", n=100)
    with pytest.raises(FirewallBreach):
        C.Arena("r", C.standard_field("a")).step(df.iloc[:20], df.index[19])
    with pytest.raises(C.CompetitionError):
        a.step(df.iloc[10:30], "2030-01-01")              # overlaps history already scored
    assert a.step(df.iloc[0:0], "2030-01-01").n_rows == 0


def test_field_validation_and_bad_specs():
    with pytest.raises(C.CompetitionError):
        C.Arena("r", [C.null_spec()])
    with pytest.raises(C.CompetitionError):
        C.Arena("r", [C.null_spec(), C.null_spec()])
    with pytest.raises(C.CompetitionError):
        C.Arena("r", [C.null_spec(), dc.replace(C.causal_spec("a"), gate=(("liq", "!!", 0.0),))])
    with pytest.raises(C.CompetitionError):
        C.synthetic_world("nonsense")
    with pytest.raises(C.CompetitionError):
        C.Arena("r", C.standard_field("a"), cfg=C.ArenaConfig(elim_weight=0.2, revive_weight=0.1))


def test_the_field_knows_where_to_look_next_and_how_long_it_will_take():
    a, _, _ = _arena("null", n=200, seed=41)
    sl = a.next_experiment()
    if sl is not None:
        assert sl.leader != sl.rival and sl.fraction_of_rows > 0 and sl.description
    tab = C.pairwise_table(a)
    assert len(tab) == len(a.ids) * (len(a.ids) - 1)
    assert C.expected_rows_to_separate(a) >= 0.0


def test_mixture_forecast_is_honest_and_carries_disagreement_as_uncertainty():
    a, df, _ = _arena("causal")
    assert C.mixture_calibration(a)["verdict"] in ("CALIBRATED", "UNDERCONFIDENT")
    early, dfe, _ = _arena("causal", n=60)
    mu, sd = early.predict(dfe.iloc[-10:])
    _, sd_leader = early.predict(dfe.iloc[-10:], "leader")
    assert (sd >= sd_leader - 1e-12).all()                # mixture is never more certain than its leader when they disagree
    assert C.mixture_calibration(C.Arena("r", C.standard_field("a")))["verdict"] == "INSUFFICIENT_DATA"


def test_graph_edges_and_replay_determinism():
    a, df, specs = _arena("liq")
    kinds = {e[2].value for e in a.edges()}
    assert {"CONTRADICTS", "SPECIALIZES"} <= kinds
    frames = [df.iloc[s:s + 20] for s in range(0, len(df), 20)]
    b = C.replay(specs, frames, "2030-01-01", "r")
    assert b.state_hash() == a.state_hash()
    a2, _, _ = _arena("liq", seed=4)
    assert a2.state_hash() != a.state_hash()               # a different world gives a different state: the hash is not constant


def test_competition_book_tracks_settled_and_unresolved_relations():
    book = C.CompetitionBook()
    book.open("easy", C.standard_field("a", proxies=["vol"]))
    book.open("thin", C.standard_field("a", proxies=["vol"]))
    easy, thin = C.synthetic_world("causal", n=400, seed=3), C.synthetic_world("causal", n=40, seed=4)
    for s in range(0, 40, 20):
        book.step({"thin": thin.iloc[s:s + 20]}, "2030-01-01")
    for s in range(0, 400, 20):
        book.step({"easy": easy.iloc[s:s + 20]}, "2030-01-01")
    assert book.settled() == {"easy": "H_causal"} and book.unresolved() == ["thin"]
    assert book.unknown_state("thin") == Unknown.CONFLICTED
    assert len(book.priorities()) == 1 and len(book.summary()) == 2
    with pytest.raises(C.CompetitionError):
        book.step({"ghost": easy}, "2030-01-01")
    with pytest.raises(C.CompetitionError):
        book.open("easy", C.standard_field("a"))


def test_conditional_story_can_be_built_from_a_learned_boundary():
    from engine.learning.boundary import Boundary, BoundaryKind
    b = Boundary("p", "liq", BoundaryKind.THRESHOLD, 0.0, ">", 100, 100, 0.02, 0.0, 0.002, 0.002, 6.0, 0.005, 0.01, 0.1,
                 "CONFIRMED", 3.0, 0.02, 0.0, "STOPS", True, (), "2020-01-01")
    s = C.conditional_from_boundary("a", b)
    assert s.gate == (("liq", ">", 0.0),) and s.family == C.HypFamily.CONDITIONAL


# ============================================================================================ complexity
def cand(rid, units_kw, oos, folds=None, ins=None):
    return X.Candidate(X.RuleSpec(rid, **units_kw), oos, folds, ins)


def oos_pair(n, gain, seed=0, noise=0.01, fold_pattern=None):
    idx = pd.date_range("2016-01-04", periods=n, freq="W-MON")
    rng = np.random.default_rng(seed)
    s = pd.Series(rng.normal(0.004, noise, n), index=idx)
    g = np.full(n, gain) if fold_pattern is None else np.repeat(fold_pattern, int(math.ceil(n / len(fold_pattern))))[:n]
    c = s + g + rng.normal(0, noise * 0.3, n)
    folds = pd.Series(idx.year, index=idx)
    return s, c, folds


def test_equal_performance_prefers_the_simple_rule():
    s, c, f = oos_pair(200, 0.0)
    v = X.compare(cand("simple", dict(n_features=1), s, f), cand("complex", dict(n_features=4, n_conditions=2, n_thresholds=2, n_interactions=2), c, f))
    assert v.verdict in (X.Verdict.TIE, X.Verdict.SIMPLE) and v.prefer == "simple"


def test_noise_clauses_almost_never_earn_their_place():
    earned = 0
    for seed in range(25):
        s, c, f = oos_pair(150, 0.0, seed=seed, noise=0.01)
        c = c + np.random.default_rng(100 + seed).normal(0, 0.004, len(c))       # extra clauses just add noise to the record
        v = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=3, n_conditions=2, n_thresholds=2), c, f))
        earned += int(v.verdict == X.Verdict.COMPLEX)
    assert earned <= 2


def test_a_genuine_transferable_gain_earns_the_complexity():
    s, c, f = oos_pair(260, 0.01, seed=3)
    v = X.compare(cand("s", dict(n_features=1), s, f, 0.005), cand("c", dict(n_features=4, n_conditions=2, n_thresholds=2), c, f, 0.011))
    assert v.verdict == X.Verdict.COMPLEX and v.prefer == "c" and v.transfer == 1.0 and not v.failed


def test_a_gain_that_does_not_transfer_across_folds_is_not_accepted():
    s, c, f = oos_pair(260, 0.0, seed=4)
    c = c.copy()
    c[f == f.min()] += 0.05                               # all the gain is in ONE year
    v = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=4, n_conditions=2, n_thresholds=2), c, f))
    assert v.verdict != X.Verdict.COMPLEX and any("transfer" in x for x in v.failed)


def test_a_gain_bought_with_a_worse_tail_is_not_failure_safe():
    s, c, f = oos_pair(260, 0.01, seed=5)
    c = c.copy()
    c.iloc[::13] -= 0.12
    v = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=3, n_conditions=1, n_thresholds=1), c, f))
    assert v.verdict != X.Verdict.COMPLEX and any("tail" in x for x in v.failed)


def test_hard_coded_exceptions_are_an_identity_smell_even_with_a_gain():
    s, c, f = oos_pair(260, 0.01, seed=6)
    v = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=2, n_conditions=1, n_thresholds=1, n_exceptions=3), c, f))
    assert v.verdict != X.Verdict.COMPLEX and any("exceptions" in x for x in v.failed)


def test_higher_complexity_demands_more_evidence():
    assert X.required_t(0) < X.required_t(2) < X.required_t(10)
    assert X.required_t(-3) == X.required_t(0)            # a simpler alternative is never penalised
    s, c, f = oos_pair(120, 0.0035, seed=7, noise=0.01)
    small = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=2), c, f))
    big = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=9, n_conditions=5, n_thresholds=5, n_interactions=4), c, f))
    assert small.t_required < big.t_required and small.gain_t == pytest.approx(big.gain_t)


def test_too_little_out_of_sample_data_is_need_data():
    s, c, f = oos_pair(10, 0.01)
    assert X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=3), c, f)).verdict == X.Verdict.NEED_DATA


def test_one_se_rule_and_pareto_front():
    cs = [("a", 1.0, 0.50, 0.05), ("b", 3.0, 0.53, 0.05), ("c", 8.0, 0.54, 0.05), ("d", 12.0, 0.40, 0.05)]
    assert X.one_se_rule(cs) == "a"                       # the simplest within one se of the best
    assert X.one_se_rule(cs, k=0.1) == "c"
    assert X.pareto_front([(i, u, s) for i, u, s, _ in cs]) == ["a", "b", "c"]
    with pytest.raises(X.ComplexityError):
        X.one_se_rule([])


def test_sequential_growth_stops_at_the_true_level_of_complexity():
    idx = pd.date_range("2016-01-04", periods=240, freq="W-MON")
    base = pd.Series(np.random.default_rng(1).normal(0.004, 0.01, 240), index=idx)
    folds = pd.Series(idx.year, index=idx)
    rng = np.random.default_rng(2)
    ladder = [cand("r1", dict(n_features=1), base, folds),
              cand("r2", dict(n_features=2), base + 0.008 + rng.normal(0, 0.002, 240), folds),           # real improvement
              cand("r3", dict(n_features=3, n_conditions=1, n_thresholds=1), base + 0.008 + rng.normal(0, 0.003, 240), folds),   # nothing more
              cand("r4", dict(n_features=5, n_conditions=3, n_thresholds=3), base + 0.008 + rng.normal(0, 0.004, 240), folds)]
    g = X.sequential_growth(ladder)
    assert g["chosen"] == "r2" and g["n_upgrades"] == 1
    assert len(X.pairwise_verdicts(ladder)) == 6 and len(X.rank_rules(ladder)) == 4
    rep = X.complexity_report(ladder)
    assert "EARN-YOUR-PLACE CHOICE: r2" in rep


def test_kappa_calibration_controls_false_promotion_of_noise_rules():
    r = X.calibrate_kappa(n_obs=160, sims=200, seed=1)
    assert r["met_target"]
    rates = r["false_earn_rates"][r["kappa"]]
    assert max(rates.values()) <= 0.05 + 1e-9
    assert X.calibrate_kappa(n_obs=160, sims=200, seed=1) == r      # deterministic


def test_optimism_grows_with_the_number_of_noise_parameters():
    o = X.null_optimism(120, [0, 3, 12], n_sims=60, seed=2)
    assert o[0] == pytest.approx(0.0, abs=1e-9) and o[3] > 0 and o[12] > o[3]


def test_structure_helpers_parse_patterns_and_generate_baselines():
    sp = X.spec_from_pattern_text("A q1 & B q4 unless C q0")
    assert sp.n_features == 3 and sp.n_conditions == 1 and sp.n_interactions == 1
    assert X.spec_from_pattern_text("A q1").units() < sp.units()
    vs = X.simpler_variants(X.RuleSpec("r", n_features=3, n_conditions=2, n_thresholds=2, n_interactions=1, n_exceptions=1, tree_depth=2))
    assert len(vs) == 4 and all(v.units() < 20 for v in vs)
    assert X.RuleSpec("bad", n_features=-1).validate()
    with pytest.raises(X.ComplexityError):
        X.RuleSpec("bad", n_features=-1).units()
    au = X.audit_specs([X.RuleSpec("a", n_features=1), X.RuleSpec("b", n_features=2, n_exceptions=2)])
    assert au["identity_smells"] == ["b"] and X.audit_specs([]) == {"n": 0}
    assert X.within_budget(X.RuleSpec("a", n_features=2), 100) and not X.within_budget(X.RuleSpec("a", n_features=40), 100)
    assert X.ridge_effective_df(np.random.default_rng(0).normal(size=(80, 6)), 50.0) < 6
    assert X.optimism_adjusted(1.0, 100, 5) > 1.0


def test_verdict_stability_under_resampling_and_across_eras():
    s, c, f = oos_pair(240, 0.01, seed=8)
    strong = X.bootstrap_earn_probability(cand("s", dict(n_features=1), s), cand("c", dict(n_features=3), c))
    s2, c2, _ = oos_pair(240, 0.0, seed=9)
    weak = X.bootstrap_earn_probability(cand("s", dict(n_features=1), s2), cand("c", dict(n_features=3), c2))
    assert strong["p_earn"] > 0.9 and weak["p_earn"] < 0.3
    eras = X.verdict_by_era(cand("s", dict(n_features=1), s), cand("c", dict(n_features=3), c))
    assert len(eras) == 3 and all(e["verdict"] == "COMPLEX" for e in eras)
    assert X.bootstrap_earn_probability(cand("s", dict(n_features=1), s[:5]), cand("c", dict(n_features=3), c[:5]))["p_earn"] != 1.0


def test_equivalence_test_and_comparison_ledger():
    s, c, f = oos_pair(300, 0.0, seed=10)
    p = X.equivalence_pvalue(cand("s", dict(n_features=1), s), cand("c", dict(n_features=3), c), margin=0.005)
    s2, c2, _ = oos_pair(300, 0.02, seed=11)
    p2 = X.equivalence_pvalue(cand("s", dict(n_features=1), s2), cand("c", dict(n_features=3), c2), margin=0.005)
    assert p < 0.05 < p2
    led = X.ComparisonLedger()
    v1 = X.compare(cand("s", dict(n_features=1), s, f), cand("c", dict(n_features=3), c, f))
    v2 = X.compare(cand("s", dict(n_features=1), s2, f), cand("c", dict(n_features=3), c2, f))
    led.record(v1, "2021-01-01")
    led.record(v2, "2022-01-01")
    assert led.reversals() and led.history("c")
    with pytest.raises(X.ComplexityError):
        led.record(v1, "2020-01-01")
    assert X.format_verdict(v1).startswith(str(v1.verdict))
