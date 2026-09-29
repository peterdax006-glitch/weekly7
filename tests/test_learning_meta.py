"""Tests for engine.learning.meta_learning (contract section 37; checklist G12, C16). Synthetic planted worlds only.
Meta-learning is itself evaluated out of sample, so the tests plant a world where the right answer is known, a null
world where the right answer is 'nothing', a label leak, and a future-label channel, and require each to be handled."""
import datetime as dt
import math

import numpy as np
import pytest

from engine.learning import meta_learning as ml
from engine.learning import research_policy as rp
from engine.learning.core import FailureCause, ValidationLabel


def store_of(recs):
    s = ml.MetaStore()
    s.extend(recs)
    return s


NOW = "2010-01-01"


# ------------------------------------------------------------------------------------------------ statistics

def test_wilson_and_group_rates_shrink_small_groups():
    lo, hi = ml.wilson(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = ml.wilson(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
    rates, prior = ml.group_rates([("big", True)] * 60 + [("big", False)] * 40 + [("tiny", True)] * 3 + [("mid", False)] * 20 + [("mid", True)] * 20)
    assert rates["tiny"].raw == 1.0 and rates["tiny"].shrunk < 0.95          # three wins are not certainty
    assert abs(rates["big"].shrunk - 0.6) < 0.03                            # lots of data barely moves
    assert ml.group_rates([])[0] == {}


def test_beta_prior_recovers_between_group_variance():
    rng = np.random.default_rng(0)
    true = rng.beta(8, 8, size=40)
    n = np.full(40, 60)
    s = rng.binomial(n, true)
    a, b = ml.fit_beta_prior(s, n)
    assert 0.4 < a / (a + b) < 0.6 and 6 < a + b < 80
    a2, b2 = ml.fit_beta_prior([10, 10, 10], [20, 20, 20])                    # identical groups: strong shrinkage to the pool
    assert a2 + b2 >= 100
    assert ml.fit_beta_prior([], []) == (1.0, 1.0)


def test_auc_brier_logloss_and_logistic_recovery():
    assert ml.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0 and ml.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert ml.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5 and ml.auc([1, 2], [1, 1]) is None
    assert ml.brier([1, 0], [1, 0]) == 0.0 and ml.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3)))).astype(float)
    w = ml.fit_logistic(X, y, l2=0.1)
    assert w[0] == pytest.approx(1.5, abs=0.25) and w[1] == pytest.approx(-1.0, abs=0.25) and w[2] == pytest.approx(0.3, abs=0.2)


def test_paired_bootstrap_is_deterministic_and_detects_difference():
    a, b = np.zeros(100), np.ones(100) * 0.1
    r1, r2 = ml.paired_bootstrap(a, b, 1), ml.paired_bootstrap(a, b, 1)
    assert r1 == r2 and r1["hi"] < 0
    assert ml.paired_bootstrap([1.0], [0.0], 1)["n"] == 1


# ------------------------------------------------------------------------------------------------ store & time discipline

def test_records_are_validated_on_entry():
    s = ml.MetaStore()
    with pytest.raises(ValueError):
        s.add(ml.DiscoveryRecord("d", "f", "2001-01-01", {"x": float("nan")}))
    with pytest.raises(ValueError):
        s.add(ml.DiscoveryRecord("d", "f", "2001-01-01", {"x": 1.0}, True, ""))                # label without resolution date
    with pytest.raises(ValueError):
        s.add(ml.DiscoveryRecord("d", "f", "2001-01-01", {"x": 1.0}, True, "2000-01-01"))     # resolved before discovered
    with pytest.raises(ValueError):
        s.add(ml.DecayRecord("i", "m", 10, 0.0, "2001-01-01"))
    with pytest.raises(TypeError):
        s.add(object())


def test_as_of_hides_labels_that_resolve_at_or_after_now():
    recs = ml.synthetic_discoveries(60, 1)
    s = store_of(recs)
    cut = recs[30].resolved_at
    seen = s.as_of(cut)
    assert all(ml.to_ts(r.resolved_at) < ml.to_ts(cut) for r in seen.discoveries) and len(seen.discoveries) == 30
    assert s.as_of("1990-01-01").counts()["discoveries"] == 0


# ------------------------------------------------------------------------------------------------ the OOS test of meta-learning

def test_planted_signal_is_found_out_of_sample():
    u = ml.MetaLearner(store_of(ml.synthetic_discoveries(400, 1))).update(NOW, seed=3)
    o = u.oos["survival"]
    assert o.beats_baseline and o.model_brier < o.baseline_brier - 0.03 and o.model_auc > 0.7
    assert o.label == ValidationLabel.NOT_VALIDATED.value                    # synthetic records can never validate
    assert u.oos["family_survival"].beats_baseline


def test_null_world_produces_no_false_claim():
    null = ml.synthetic_discoveries(400, 1, family_survival={"a": 0.4, "b": 0.4, "c": 0.4}, signal_feature=False)
    u = ml.MetaLearner(store_of(null)).update(NOW, seed=3)
    assert not u.oos["survival"].beats_baseline and not u.oos["family_survival"].beats_baseline
    assert u.oos["survival"].label != ValidationLabel.VALIDATED.value


def test_label_leak_is_flagged_not_celebrated():
    leaky = ml.synthetic_discoveries(300, 1, leak=True)
    assert ml.leak_audit(leaky)[0][0] == "post_hoc_flag"
    o = ml.MetaLearner(store_of(leaky)).update(NOW, seed=3).oos["survival"]
    assert o.label == ValidationLabel.FAILED_VALIDATION.value and o.leak_suspects
    assert ml.leak_audit(ml.synthetic_discoveries(300, 1)) == []


def test_future_labels_cannot_reach_a_decision_at_now():
    recs = ml.synthetic_discoveries(300, 2)
    base = ml.MetaLearner(store_of(recs)).update("2004-01-01", seed=1)
    later = [ml.replace(r, item_id="z" + r.item_id, discovered_at="2005-01-01", resolved_at="2006-01-01") for r in recs[:80]]
    with_future = ml.MetaLearner(store_of(recs + later)).update("2004-01-01", seed=1)
    assert base.counts == with_future.counts and base.advice.family_survival == with_future.advice.family_survival
    assert base.oos["survival"].model_brier == with_future.oos["survival"].model_brier


def test_labels_resolving_late_reduce_what_can_be_trained_on():
    fast = ml.walk_forward_survival(ml.synthetic_discoveries(300, 2, resolve_lag_days=30), NOW, 1)
    slow = ml.walk_forward_survival(ml.synthetic_discoveries(300, 2, resolve_lag_days=700), NOW, 1)
    assert slow.n_test < fast.n_test or slow.n_test == 0                     # the resolved_at rule is doing real work


def test_leaky_evaluation_is_more_flattering_than_the_honest_one():
    r = ml.leaky_evaluation(ml.synthetic_discoveries(500, 2), NOW, 1)
    assert r["leaky_skill"] > r["honest_skill"]


def test_walk_forward_is_deterministic_and_handles_tiny_input():
    recs = ml.synthetic_discoveries(300, 2)
    a, b = ml.walk_forward_survival(recs, NOW, 5), ml.walk_forward_survival(recs, NOW, 5)
    assert a == b
    tiny = ml.walk_forward_survival(recs[:10], NOW, 5)
    assert tiny.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value and tiny.n_test == 0
    assert ml.walk_forward_survival([], NOW, 5).n_test == 0
    assert ml.walk_forward_groups([], NOW, 1, "t").label == ValidationLabel.INSUFFICIENT_EVIDENCE.value


def test_certification_is_the_only_way_to_a_validated_label():
    recs = ml.synthetic_discoveries(600, 3)
    assert ml.walk_forward_survival(recs, NOW, 1, certified_real=False).label == ValidationLabel.NOT_VALIDATED.value
    assert ml.walk_forward_survival(recs, NOW, 1, certified_real=True).label == ValidationLabel.VALIDATED.value
    few = ml.walk_forward_survival(recs[:70], NOW, 1, min_test=200, certified_real=True)
    assert few.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value


def test_null_control_separates_skill_from_luck():
    real = ml.null_control(ml.synthetic_discoveries(400, 1), NOW, 1, n_perm=30)
    assert real["verdict"] == "SKILL" and real["p"] < 0.05 and real["real_gain"] > real["null_p95"]
    nullw = ml.null_control(ml.synthetic_discoveries(400, 1, family_survival={"a": .4, "b": .4, "c": .4}, signal_feature=False), NOW, 1, n_perm=30)
    assert nullw["verdict"] == "NO_SKILL_BEYOND_NULL"
    assert ml.null_control([], NOW, 1)["verdict"] == "INSUFFICIENT"


# ------------------------------------------------------------------------------------------------ the eight questions

def test_families_that_overfit_are_identified():
    u = ml.MetaLearner(store_of(ml.synthetic_discoveries(500, 4))).update(NOW, 1)
    ov = u.families.overfit
    assert ov["calendar_like"] > 0.7 and ov["tiny_sample"] > 0.7 and ov["momentum_like"] < 0.5
    assert u.advice.family_overfit["calendar_like"] > u.advice.family_overfit["momentum_like"]
    assert u.families.effect_ratio["momentum_like"] > u.families.effect_ratio["calendar_like"]


def test_survival_model_learns_the_planted_feature_direction():
    recs = ml.synthetic_discoveries(800, 5)
    m = ml.SurvivalModel().fit(recs)
    top = dict(m.importance())
    assert top["t_disc"] > 0.3
    assert abs(top["log_n"]) < abs(top["t_disc"])                            # noise feature ranks below the real one
    assert 0.3 < m.base < 0.5 and m.n == 800
    assert ml.SurvivalModel().fit([]).predict([]).shape == (0,)


def test_permutation_importance_agrees_and_ranks_noise_last():
    recs = ml.synthetic_discoveries(900, 6)
    imp = ml.permutation_importance(recs[:600], recs[600:], seed=1)
    names = [r["feature"] for r in imp]
    assert imp[0]["brier_increase"] > 0 and names.index("log_n") > names.index("t_disc")
    assert ml.permutation_importance(recs[:600], recs[:5], seed=1) == []


def test_context_transfer_rates_and_classification():
    recs = [ml.TransferRecord(f"i{i}", "same_sector", "f", i % 5 != 0, "2005-01-01") for i in range(40)] + \
           [ml.TransferRecord(f"j{i}", "other_regime", "f", i % 5 == 0, "2005-01-01") for i in range(40)]
    rep = ml.context_report(recs)
    assert rep["transfers"] == ["same_sector"] and rep["does_not_transfer"] == ["other_regime"]
    assert ml.context_report([])["n"] == 0


def test_decay_fit_recovers_planted_half_life_and_flags_rapid_types():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(80):
        age = float(rng.uniform(10, 900))
        recs.append(ml.DecayRecord(f"a{i}", "fast_memory", age, float(np.exp(-np.log(2) / 120 * age) * np.exp(rng.normal(0, 0.1))), "2005-01-01"))
        recs.append(ml.DecayRecord(f"b{i}", "durable_memory", age, float(np.exp(-np.log(2) / 5000 * age) * np.exp(rng.normal(0, 0.1))), "2005-01-01"))
    rep = ml.decay_report(recs, seed=1)
    fast, slow = rep["fits"]["fast_memory"], rep["fits"]["durable_memory"]
    assert fast.half_life_days == pytest.approx(120, rel=0.2) and slow.half_life_days > 1500
    assert rep["rapid_decay"] == ["fast_memory"]
    assert ml.decay_fit("x", [ml.DecayRecord("i", "x", 5, 0.9, "2005-01-01")] * 3) is None      # too few points
    assert ml.decay_report([])["fits"] == {}


def test_yield_and_bits_per_minute():
    recs = [ml.YieldRecord(f"y{i}", "diagnosis", i % 2 == 0, 0.3 if i % 2 == 0 else 0.0, 10, "2005-01-01") for i in range(20)] + \
           [ml.YieldRecord(f"z{i}", "tuning", False, 0.0, 30, "2005-01-01") for i in range(20)]
    rep = ml.yield_report(recs)
    assert rep["rates"]["diagnosis"].shrunk > 0.4 > rep["rates"]["tuning"].shrunk
    assert rep["bits_per_minute"]["diagnosis"] > 0 == rep["bits_per_minute"]["tuning"]


def test_fake_improvement_learners_are_named():
    recs = [ml.LearnerRecord("memory_bank", 0.002, False, "2005-01-01") for _ in range(6)] + \
           [ml.LearnerRecord("honest", 0.002, True, "2005-01-01", 0.0015) for _ in range(6)]
    rep = ml.learner_report(recs)
    assert rep["memory_bank"]["produces_fake_improvement"] and not rep["honest"]["produces_fake_improvement"]
    rank = ml.learner_ranking(recs, seed=1)
    assert rank[0]["learner"] == "honest" and rank[0]["demonstrated"] and not rank[1]["demonstrated"] and rank[1]["unverified_claim"] > 0
    assert ml.verify_learner_claim(0.01, 0.006) and not ml.verify_learner_claim(0.01, -0.006) and not ml.verify_learner_claim(0.01, None)


def test_explanation_precision_calibration_and_majority_baseline():
    causes = [FailureCause.WRONG_CONTEXT.value, FailureCause.REGIME_CHANGE.value, FailureCause.FALSE_PATTERN.value]
    recs = []
    rng = np.random.default_rng(0)
    for i in range(90):
        pred = causes[i % 3]
        good = (pred == causes[0] and rng.random() < 0.85) or (pred == causes[1] and rng.random() < 0.3) or (pred == causes[2] and rng.random() < 0.1)
        recs.append(ml.ExplanationRecord(f"e{i}", pred, 0.9, pred if good else FailureCause.UNKNOWN.value, "2005-01-01"))
    rep = ml.explanation_report(recs)
    assert rep["precision"][causes[0]].shrunk > rep["precision"][causes[2]].shrunk + 0.4
    assert rep["calibration"]["overconfident"]                               # says 0.9, right about 40 percent of the time
    assert ml.discount_explanation_confidence(recs, 0.9) < 0.7
    assert ml.explanation_calibration_curve([]) == []
    assert not ml.ExplanationRecord("x", "UNKNOWN", 0.5, "UNKNOWN", "2005-01-01").correct()   # an unconfirmable explanation earns no credit


def test_ablation_streaks_and_forecast():
    recs = [ml.AblationRecord("calendar_feats", True, f"r{i}", f"2005-01-{i + 1:02d}") for i in range(5)] + \
           [ml.AblationRecord("price_feats", i % 2 == 0, f"r{i}", f"2005-01-{i + 1:02d}") for i in range(5)] + \
           [ml.AblationRecord("volume_feats", False, "r0", "2005-01-01")]
    rep = ml.ablation_report(recs)
    assert rep["calendar_feats"]["repeatedly_fails"] and not rep["price_feats"]["repeatedly_fails"] and not rep["volume_feats"]["repeatedly_fails"]
    f = ml.ablation_forecast(recs)
    assert f["calendar_feats"]["candidate_for_removal"] and f["calendar_feats"]["p_fail_next"] > 0.8 > f["price_feats"]["p_fail_next"]


# ------------------------------------------------------------------------------------------------ calibration, drift, triage

def test_recalibration_fixes_overconfidence_on_heldout_data():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.8, 600)
    y = (rng.random(600) < p_true).astype(int)
    over = np.clip(0.5 + (p_true - 0.5) * 2.0, 0.01, 0.99)                    # a model twice as confident as it should be
    rc = ml.PlattRecalibrator().fit(over[:300], y[:300])
    fixed = rc.transform(over[300:])
    assert ml.expected_calibration_error(fixed, y[300:]) < ml.expected_calibration_error(over[300:], y[300:])
    assert rc.a < 1.0
    ident = ml.PlattRecalibrator().fit([0.5] * 5, [1, 0, 1, 0, 1])
    assert (ident.a, ident.b) == (1.0, 0.0)


def test_triage_value_and_precision_at_k():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(int)
    good_scores = y * 0.5 + rng.random(400) * 0.5
    t = ml.triage_value(good_scores, y, 0.5, cost_per_test=1.0, value_per_survivor=3.0)
    assert t["recall_kept"] > 0.85 and t["net_value"] > 0
    bad = ml.triage_value(rng.random(400), y, 0.5, cost_per_test=1.0, value_per_survivor=3.0)
    assert bad["net_value"] < t["net_value"]
    assert ml.precision_at_k(good_scores, y, 40) > 0.8 and ml.precision_at_k([0.1], [1], 5) is None
    assert ml.decile_lift(good_scores, y, 4)[0]["lift"] > 1.5 and ml.decile_lift([1, 2], [0, 1]) == []
    with pytest.raises(ValueError):
        ml.triage_value([], [], 0.5)


def test_survival_drift_detector_alarms_on_a_planted_rate_change():
    recs = ml.synthetic_discoveries(300, 1, family_survival={"a": 0.8}, signal_feature=False)
    flipped = [ml.replace(r, survived_oos=(i % 5 != 0) if i < 150 else (i % 5 == 0)) for i, r in enumerate(recs)]     # 80 percent, then 20
    assert any(a == "RATE_DOWN" for _, a in ml.drift_scan(flipped))
    steady = ml.synthetic_discoveries(300, 1, family_survival={"a": 0.5}, signal_feature=False)
    assert len(ml.drift_scan(steady, delta=0.05, lam=8.0)) <= 1


def test_era_confounding_is_detected():
    recs = []
    for i in range(240):
        era = i // 80
        fam = "late_family" if era == 2 and i % 2 == 0 else "steady"
        good = (era < 2)
        recs.append(ml.DiscoveryRecord(f"d{i}", fam, f"{2001 + era * 3}-01-{i % 27 + 1:02d}", {"x": 1.0}, good, f"{2002 + era * 3}-06-01"))
    rep = ml.era_confound_report(recs)
    assert rep["verdict"] == "OK" and rep["families"]["late_family"]["raw"] < rep["families"]["steady"]["raw"]
    assert rep["families"]["late_family"]["confounded"]                       # the family is bad only because its era was bad
    assert ml.era_confound_report(recs[:5])["verdict"] == "INSUFFICIENT"


def test_temporal_class_report():
    recs = [ml.DiscoveryRecord(f"p{i}", "f", "2001-01-01", {"x": 1.0}, i % 5 != 0, "2002-01-01", temporal_class="PERSISTENT") for i in range(30)] + \
           [ml.DiscoveryRecord(f"e{i}", "f", "2001-01-01", {"x": 1.0}, i % 5 == 0, "2002-01-01", temporal_class="EPISODIC") for i in range(30)]
    rep = ml.temporal_class_report(recs)
    assert rep["rates"]["PERSISTENT"].shrunk > rep["rates"]["EPISODIC"].shrunk + 0.3


# ------------------------------------------------------------------------------------------------ streaming, yield model, curve

def test_online_rates_equal_batch_rates():
    rows = [("a", True), ("a", False), ("b", True), ("b", True), ("c", False), ("a", True), ("c", False), ("b", False)] * 6
    online = ml.OnlineGroupRates()
    for k, o in rows:
        online.update(k, o, "2005-01-01")
    batch, bprior = ml.group_rates(rows)
    snap, sprior = online.snapshot()
    assert sprior == pytest.approx(bprior) and all(snap[k].shrunk == pytest.approx(batch[k].shrunk) for k in batch)
    assert online.stale("2010-01-01", 365) == ["a", "b", "c"] and online.stale("2005-02-01", 365) == []


def yield_examples(n, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        bits = float(rng.uniform(0, 1))
        useful = bool(rng.random() < (0.15 + 0.6 * bits if informative else 0.4))
        planned = dt.date(2001, 1, 1) + dt.timedelta(days=2 * i)
        out.append(ml.YieldExample(f"x{i}", "diagnosis" if i % 2 else "tuning", bits, float(rng.uniform(5, 50)), float(rng.uniform(0, 1)),
                                   float(rng.integers(0, 4)), useful, planned.isoformat(), (planned + dt.timedelta(days=40)).isoformat()))
    return sorted(out, key=lambda e: e.planned_at)


def test_yield_model_is_evaluated_out_of_sample_and_null_is_quiet():
    good = ml.walk_forward_yield(yield_examples(500, 1), "2010-01-01", 1)
    assert good.beats_baseline and good.model_auc > 0.6
    null = ml.walk_forward_yield(yield_examples(500, 1, informative=False), "2010-01-01", 1)
    assert not null.beats_baseline
    assert ml.walk_forward_yield([], "2010-01-01", 1).label == ValidationLabel.INSUFFICIENT_EVIDENCE.value


def test_meta_learning_curve_and_generalisation_across_eras():
    s = store_of(ml.synthetic_discoveries(600, 2))
    curve = ml.meta_learning_curve(s, ["2004-01-01", "2006-01-01", "2010-01-01"], seed=1)
    assert curve[0]["n_labels"] < curve[-1]["n_labels"] and curve[-1]["skill"] > 0.02
    learner = ml.MetaLearner(s)
    gen = ml.meta_generalisation(learner, [("2001-01-01", "2004-01-01"), ("2004-01-01", "2006-06-01"), ("2006-06-01", "2009-01-01")])
    assert gen["transfer_ratio"] is not None and gen["transfer_ratio"] >= 0.5
    assert ml.meta_generalisation(ml.MetaLearner(ml.MetaStore()), [("2001-01-01", "2002-01-01"), ("2002-01-01", "2003-01-01")])["transfer_ratio"] is None


# ------------------------------------------------------------------------------------------------ store health, IO, adapters, reports

def test_store_health_reports_problems_instead_of_hiding_them():
    assert ml.store_health(ml.MetaStore(), NOW)["usable"] is False
    same = [ml.DiscoveryRecord(f"d{i}", "f", "2001-01-01", {"x": 1.0}, True, "2001-06-01") for i in range(30)]
    h = ml.store_health(store_of(same), NOW)
    assert any("degenerate" in p for p in h["problems"]) and any("span" in p for p in h["problems"]) and any("single" in p for p in h["problems"])
    ok = ml.store_health(store_of(ml.synthetic_discoveries(300, 1)), NOW)
    assert ok["usable"] and ok["problems"] == []


def test_store_round_trip_reports_bad_lines(tmp_path):
    s = store_of(ml.synthetic_discoveries(30, 1))
    s.add(ml.TransferRecord("i", "sector", "f", True, "2005-01-01"))
    s.add(ml.ExplanationRecord("e", "WRONG_CONTEXT", 0.7, "WRONG_CONTEXT", "2005-01-01"))
    n = ml.save_store(s, tmp_path / "meta.jsonl")
    back, bad = ml.load_store(tmp_path / "meta.jsonl")
    assert n == 32 and bad == 0 and back.counts() == s.counts() and back.code_hash() == s.code_hash()
    with open(tmp_path / "meta.jsonl", "a", encoding="utf-8") as f:
        f.write("{not json\n" + '{"kind": "discoveries", "rec": {"item_id": "x"}}\n')
    _, bad2 = ml.load_store(tmp_path / "meta.jsonl")
    assert bad2 == 2
    assert ml.load_store(tmp_path / "missing.jsonl")[1] == 0


def test_pattern_row_adapter_never_uses_the_label_source_as_a_feature():
    rows = [{"key_named": "vol_q<0.3 & mom>0", "effect": 0.02, "t_disc": 4.0, "t_conf": 3.0, "p_real": 0.9, "n": 400},
            {"key_named": "cal_dow==4", "effect": 0.01, "t_disc": 3.0, "t_conf": -1.0, "p_real": 0.6, "n": 300},
            {"key_named": "pending", "effect": 0.01, "t_disc": 3.0, "t_conf": None, "p_real": 0.6, "n": 300}]
    recs = ml.discoveries_from_pattern_rows(rows, "2005-01-01", "2006-01-01")
    assert [r.survived_oos for r in recs] == [True, False, None] and recs[2].resolved_at == ""
    assert all("t_conf" not in r.features for r in recs) and recs[0].features["n_conditions"] == 2.0
    good = ml.explanations_from_postmortems([{"item_id": "a", "predicted_cause": "WRONG_CONTEXT", "verified_cause": "UNKNOWN"}], "2006-01-01")
    assert good[0].verified_cause == "UNKNOWN"
    with pytest.raises(ValueError):
        ml.explanations_from_postmortems([{"item_id": "a", "predicted_cause": "NOT_A_CAUSE", "verified_cause": "UNKNOWN"}], "2006-01-01")


def test_update_report_scorecard_and_diff(tmp_path):
    recs = ml.synthetic_discoveries(500, 2)
    s = store_of(recs)
    s.add(ml.LearnerRecord("memory_bank", 0.002, False, "2005-01-01"))
    early = ml.MetaLearner(s).update("2005-01-01", 1)
    late = ml.MetaLearner(s).update(NOW, 1)
    d = ml.diff_updates(early, late)
    assert d["n_records_new"] > d["n_records_old"]
    card = ml.scorecard_numbers(late)
    assert card["oos_skill"]["survival"] > 0 and card["families_overfit"] >= 2 and card["label"] == ValidationLabel.NOT_VALIDATED.value
    path = ml.write_report(late, tmp_path)
    assert path.exists() and "Meta-learning update" in path.read_text(encoding="utf-8") and (tmp_path / "meta_summary.json").exists()
    assert late.store_hash == ml.MetaLearner(s).update(NOW, 1).store_hash and late.health["usable"]
    learner = ml.MetaLearner(s)
    learner.update(NOW, 1)
    learner.save_history(tmp_path / "h.jsonl")
    assert (tmp_path / "h.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_empty_store_update_is_honest_and_advice_is_empty():
    u = ml.MetaLearner().update(NOW, 0)
    assert all(o.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value for o in u.oos.values())
    assert u.advice.family_overfit == {} and u.advice.trust() == 0.0 and not u.health["usable"]
    assert u.summary()["counts"]["discoveries"] == 0


def test_meta_advice_feeds_the_research_policy():
    u = ml.MetaLearner(store_of(ml.synthetic_discoveries(500, 4))).update(NOW, 1)
    assert isinstance(u.advice, rp.MetaAdvice) and u.advice.check() == []
    pol = rp.ResearchPolicy()
    mk = lambda fam: rp.Candidate("c_" + fam, "q", rp.ResearchTarget.WEAK_PATTERN, "2026-01-01", family=fam, info=rp.InfoModel("normal", {"n_now": 10, "n_new": 30}))
    ctx = rp.PolicyContext(now="2026-02-01", meta=u.advice)
    assert pol.priority.score(mk("momentum_like"), ctx).priority > pol.priority.score(mk("calendar_like"), ctx).priority * 1.3
    plain = rp.PolicyContext(now="2026-02-01")
    assert pol.priority.score(mk("momentum_like"), plain).priority == pytest.approx(pol.priority.score(mk("calendar_like"), plain).priority)


def test_meta_self_check_passes_and_is_deterministic():
    a, b = ml.self_check(2), ml.self_check(2)
    assert a == b and a["all_passed"], a


# ------------------------------------------------------------------------------------------------ part 2 mechanisms

def test_family_comparison_ranking_and_budgeting():
    rep = ml.family_report(ml.synthetic_discoveries(500, 4))
    order = ml.family_ranking(rep, seed=1)
    assert order[0]["family"] == "momentum_like" and order[-1]["family"] in {"calendar_like", "tiny_sample"}
    a, b = rep.rates["momentum_like"], rep.rates["calendar_like"]
    assert ml.prob_family_better(a, b, rep.prior, 1) > 0.99 > ml.prob_family_better(b, a, rep.prior, 1)
    assert ml.family_ranking(ml.family_report([])) == []
    budget = ml.discovery_budget(rep, want_survivors=10)
    assert budget[0]["family"] == "momentum_like" and budget[-1]["tests_needed"] > 3 * budget[0]["tests_needed"]
    assert all(r["tests_needed"] * r["rate"] >= 10 - 1e-9 for r in budget)


def test_leave_one_family_out_separates_feature_lessons_from_family_memorisation():
    good = ml.leave_one_family_out(ml.synthetic_discoveries(1200, 3))
    assert good["n_tested"] >= 3 and good["transfers_across_families"] and good["share_positive"] >= 0.75
    nosig = ml.leave_one_family_out(ml.synthetic_discoveries(1200, 3, signal_feature=False))
    assert not nosig["transfers_across_families"]                                     # only family levels differ: nothing transfers
    assert ml.leave_one_family_out([])["n_tested"] == 0


def test_advice_persistence_staleness_merge_and_stability():
    u = ml.MetaLearner(store_of(ml.synthetic_discoveries(500, 4))).update(NOW, 1)
    back = ml.advice_from_json(ml.advice_to_json(u.advice))
    assert back == u.advice
    with pytest.raises(ValueError):
        ml.advice_from_json('{"family_overfit": {"x": 2.0}}')
    assert ml.advice_age_days(u.advice, "2011-01-01") == pytest.approx(365, abs=1)
    old = ml.stale_discount(u.advice, "2011-01-01")
    assert old.n_observations < u.advice.n_observations * 0.6 and old.trust() < u.advice.trust()
    with pytest.raises(ml.FirewallBreach):
        ml.advice_age_days(u.advice, "2009-01-01")
    assert ml.stale_discount(rp.MetaAdvice(n_observations=100), "2011-01-01").n_observations == 0     # no fitted date: no trust
    merged = ml.combine_advice(u.advice, ml.MetaAdvice(family_survival={"momentum_like": 0.0}, n_observations=u.advice.n_observations, oos_label=ValidationLabel.FAILED_VALIDATION.value))
    assert merged.family_survival["momentum_like"] == pytest.approx(u.advice.family_survival["momentum_like"] / 2, abs=0.01)
    assert merged.oos_label == ValidationLabel.FAILED_VALIDATION.value and merged.n_observations == 2 * u.advice.n_observations
    early = ml.MetaLearner(store_of(ml.synthetic_discoveries(500, 4))).update("2006-01-01", 1)
    stab = ml.advice_stability(early.advice, u.advice)
    assert stab["stable"] and stab["spearman"] >= 0.7
    flipped = rp.MetaAdvice(family_survival={k: 1 - v for k, v in u.advice.family_survival.items()}, family_overfit=u.advice.family_overfit)
    assert not ml.advice_stability(u.advice, flipped)["stable"]
    assert ml.spearman([1, 2], [1, 2]) is None and ml.spearman([1, 1, 1], [1, 2, 3]) is None


def test_complexity_effect_on_survival_is_measured():
    recs = []
    rng = np.random.default_rng(0)
    for i in range(400):
        nc = float(rng.integers(1, 5))
        surv = bool(rng.random() < 0.8 - 0.18 * (nc - 1))
        recs.append(ml.DiscoveryRecord(f"c{i}", "f", "2001-01-01", {"n_conditions": nc}, surv, "2002-01-01"))
    rep = ml.survival_by_complexity(recs)
    assert rep["verdict"] == "OK" and rep["complexity_hurts"] and rep["by_value"][1.0]["rate"] > rep["by_value"][4.0]["rate"]
    flat = [ml.replace(r, survived_oos=bool(i % 2)) for i, r in enumerate(recs)]
    assert not ml.survival_by_complexity(flat)["complexity_hurts"]
    assert ml.survival_by_complexity(recs[:5])["verdict"] == "INSUFFICIENT"
    assert ml.survival_by_complexity([ml.replace(r, survived_oos=True) for r in recs])["verdict"] == "NO_VARIATION"


def test_markdown_report_lists_verdicts_and_health():
    s = store_of(ml.synthetic_discoveries(400, 2))
    for i in range(6):
        s.add(ml.LearnerRecord("memory_bank", 0.002, False, f"2005-0{i + 1}-01"))
    md = ml.to_markdown(ml.MetaLearner(s).update(NOW, 1))
    assert md.startswith("# Meta-learning as of") and "Out-of-sample evaluation" in md and "memory_bank" in md and "usable" in md and "none identified" not in md


def test_top_lessons_carry_counts_and_labels():
    s = store_of(ml.synthetic_discoveries(500, 4))
    for i in range(6):
        s.add(ml.LearnerRecord("memory_bank", 0.002, False, f"2005-0{i + 1}-01"))
    lessons = ml.top_lessons(ml.MetaLearner(s).update(NOW, 1), 10)
    assert any("overfits" in l for l in lessons) and any("memory_bank" in l for l in lessons) and all("NOT VALIDATED" in l for l in lessons)
    assert ml.top_lessons(ml.MetaLearner().update(NOW, 0)) == []
