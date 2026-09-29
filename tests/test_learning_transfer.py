"""Cross-context transfer evaluation and transfer scoring (contract C62 sections 26-27; modules engine/learning/transfer.py and
transfer_score.py). Every mechanism is checked against a planted case that it must catch, plus the empty case. Synthetic data only;
nothing reads the real archive. Status of what these tests prove: IMPLEMENTED - NOT VALIDATED."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning import transfer as T
from engine.learning import transfer_score as TS
from engine.learning.core import FirewallBreach, KnowledgeLike, Provenance, ValidationLabel

NOW = pd.Timestamp("2030-01-01")


def _world(seed=0, **kw):
    kw.setdefault("n_years", 6)
    kw.setdefault("n_tickers", 30)
    kw.setdefault("weeks_per_year", 20)
    return T.synthetic_transfer_units(seed=seed, **kw)


def _scored(u, fit, last_train_year, now=NOW):
    """Train `fit` on years <= last_train_year, score every unit with it, and return (units with learned, scope)."""
    d = T.prepare_units(u.assign(learned=u["base"]), now)
    tr = d[d["year"].astype(int) <= last_train_year]
    scope = T.TrainingScope.from_units(tr, dt.date(last_train_year + 1, 1, 1))
    pred = fit(tr)
    out = d.copy()
    out["learned"] = out["base"] + np.clip(pred(out), 0, 1) * (out["alt"] - out["base"])
    return out, scope


def _flat_units(n_clusters=24, per=12, gain_fn=None, seed=0, years=(2010, 2011, 2012, 2013)):
    """Hand-built unit table with an exactly controlled gain per unit (base 0, learned = gain)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        date = pd.Timestamp(year=years[c % len(years)], month=1 + (c // len(years)) % 12, day=5)
        for i in range(per):
            g = gain_fn(date, i, rng) if gain_fn else rng.normal(0.01, 0.01)
            rows.append((date + pd.Timedelta(days=i), date + pd.Timedelta(days=i + 7), f"T{c:03d}_{i:02d}", 0.0, float(g)))
    return pd.DataFrame(rows, columns=["date", "mature", "ticker", "base", "learned"])


# ======================================================================================== transfer_score: the ratio
def test_ratio_ok_and_negative_cross_gain_gives_negative_ratio():
    r = TS.transfer_ratio(0.008, 0.010)
    assert r.status == TS.RatioStatus.OK and r.value == pytest.approx(0.8)
    r = TS.transfer_ratio(-0.004, 0.010)
    assert r.status == TS.RatioStatus.OK and r.value == pytest.approx(-0.4)           # it hurt elsewhere: the sign survives


def test_ratio_degenerate_denominators_are_named_not_divided():
    assert TS.transfer_ratio(0.01, 0.0).status == TS.RatioStatus.TRANSFER_ONLY
    assert TS.transfer_ratio(0.01, -0.02).status == TS.RatioStatus.TRANSFER_ONLY
    assert TS.transfer_ratio(0.01, 0.0).value is None
    assert TS.transfer_ratio(-0.01, -0.02).status == TS.RatioStatus.HARMFUL            # negative/negative is NOT a positive ratio
    assert TS.transfer_ratio(0.0, 0.0).status == TS.RatioStatus.NO_GAIN
    assert TS.transfer_ratio(float("nan"), 0.01).status == TS.RatioStatus.INSUFFICIENT
    assert TS.transfer_ratio(0.01, 0.01, n_cross=2, n_same=100, min_n=5).status == TS.RatioStatus.INSUFFICIENT


def test_ratio_is_capped_when_the_denominator_is_tiny():
    r = TS.transfer_ratio(0.05, 0.0002)
    assert r.status == TS.RatioStatus.OK and r.value > 100 and r.capped == TS.RATIO_CAP


def test_ratio_ci_declares_itself_unbounded_when_the_denominator_reaches_zero():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(20), 5)
    noisy_same = rng.normal(0.0005, 0.02, 100)             # mean indistinguishable from zero
    cross = rng.normal(0.01, 0.02, 100)
    ci = TS.ratio_ci(cross, noisy_same, cl, cl, seed=3)
    assert not ci.bounded and ci.lo is None and ci.denominator_risk > 0.05
    solid_same = rng.normal(0.02, 0.005, 100)
    ok = TS.ratio_ci(cross, solid_same, cl, cl, seed=3)
    assert ok.bounded and ok.lo < ok.point.value < ok.hi


# ======================================================================================== transfer_score: specialisation
def test_over_specialised_learner_is_flagged_and_a_general_one_is_not():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(30), 10)
    same = TS.cluster_bootstrap_mean(rng.normal(0.012, 0.01, 300), cl, seed=1)
    cross_bad = TS.cluster_bootstrap_mean(rng.normal(0.0005, 0.01, 300), cl, seed=2)
    cross_ok = TS.cluster_bootstrap_mean(rng.normal(0.011, 0.01, 300), cl, seed=3)
    bad = TS.specialisation(same, cross_bad, TS.transfer_ratio(cross_bad.mean, same.mean))
    good = TS.specialisation(same, cross_ok, TS.transfer_ratio(cross_ok.mean, same.mean))
    assert bad.over_specialised and bad.severity > 0.8 and bad.flag == "OVER_SPECIALISED"
    assert not good.over_specialised and not good.suspected and good.severity < 0.3


def test_specialisation_needs_a_real_same_context_gain_first():
    zero = TS.cluster_bootstrap_mean(np.random.default_rng(0).normal(0, 0.01, 100), np.repeat(np.arange(10), 10))
    s = TS.specialisation(zero, zero, TS.transfer_ratio(zero.mean, zero.mean))
    assert not s.over_specialised and "no significant same-context gain" in s.reasons[0]


def test_untested_cross_context_is_suspected_not_transferred():
    same = TS.cluster_bootstrap_mean(np.random.default_rng(0).normal(0.02, 0.005, 100), np.repeat(np.arange(10), 10))
    none = TS.cluster_bootstrap_mean([], None)
    s = TS.specialisation(same, none, TS.transfer_ratio(float("nan"), same.mean))
    assert s.suspected and not s.over_specialised


# ======================================================================================== transfer_score: gaps
def test_memorization_gap_catches_a_lookup_table_and_passes_a_rule():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(20), 10)
    seen = rng.normal(0.02, 0.01, 200)
    fresh_rule = rng.normal(0.019, 0.01, 200)
    fresh_table = rng.normal(0.0, 0.01, 200)
    assert TS.memorization_gap(seen, fresh_table, cl, cl, seed=1).flagged
    assert not TS.memorization_gap(seen, fresh_rule, cl, cl, seed=1).flagged


def test_identity_gap_is_paired_and_detects_reliance_on_which_stock_it_was():
    rng = np.random.default_rng(3)
    base = rng.normal(0.01, 0.01, 300)
    cl = np.repeat(np.arange(30), 10)
    assert not TS.identity_gap(base, base + rng.normal(0, 1e-4, 300), cl, seed=1).flagged
    scrambled = np.zeros(300)                               # the benefit vanishes when identities are scrambled
    g = TS.identity_gap(base, scrambled, cl, seed=1)
    assert g.flagged and g.gap == pytest.approx(base.mean()) and g.relative == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TS.identity_gap(base, base[:-1])


def test_gain_concentration_separates_a_memoriser_from_a_rule():
    keys = np.repeat([f"K{i}" for i in range(40)], 5)
    spread = np.full(200, 0.01)
    lumpy = np.zeros(200)
    lumpy[:10] = 0.2                                        # 100% of the gain from two identities
    a, b = TS.gain_concentration(spread, keys), TS.gain_concentration(lumpy, keys)
    assert not a.concentrated and a.effective_n == pytest.approx(40)
    assert b.concentrated and b.top_share == pytest.approx(0.5) and b.effective_n == pytest.approx(2)
    assert np.isnan(TS.gain_concentration([], []).hhi)


# ======================================================================================== transfer_score: stability
def test_stability_distinguishes_steady_from_erratic_transfer():
    steady = TS.transfer_stability([0.010, 0.011, 0.009, 0.010, 0.012])
    erratic = TS.transfer_stability([0.03, -0.02, 0.04, -0.03, 0.01])
    negative = TS.transfer_stability([-0.01, -0.02, -0.01])
    assert steady.stable and steady.score > 0.8
    assert not erratic.stable and erratic.score < steady.score
    assert negative.score == 0.0
    e = TS.transfer_stability([])
    assert e.n_groups == 0 and e.score == 0.0 and not e.stable


def test_rolling_stability_finds_a_stretch_where_the_gain_died():
    series = [0.01] * 12 + [-0.02] * 10 + [0.01] * 12
    r = TS.rolling_stability(series, window=4)
    assert r.longest_negative_run >= 6 and r.worst_window_mean < 0 and r.sign_flips >= 2
    assert TS.rolling_stability([0.01] * 3, window=8).n_windows == 0
    with pytest.raises(ValueError):
        TS.rolling_stability([0.1], window=0)


# ======================================================================================== transfer_score: bootstrap, tests, verdicts
def test_cluster_bootstrap_is_wider_than_iid_for_correlated_units_and_honest_when_tiny():
    rng = np.random.default_rng(4)
    cluster_effect = np.repeat(rng.normal(0, 0.02, 20), 25)             # units in a cluster share one draw of luck
    v = cluster_effect + rng.normal(0, 0.002, 500)
    cl = np.repeat(np.arange(20), 25)
    clustered = TS.cluster_bootstrap_mean(v, cl, seed=1)
    iid = TS.cluster_bootstrap_mean(v, None, seed=1)
    assert (clustered.hi - clustered.lo) > 2 * (iid.hi - iid.lo)
    one = TS.cluster_bootstrap_mean([0.1, 0.2], ["a", "a"])
    assert one.lo == float("-inf") and one.hi == float("inf") and one.n_clusters == 1
    empty = TS.cluster_bootstrap_mean([], None)
    assert empty.n == 0 and np.isnan(empty.mean)
    with pytest.raises(ValueError):
        TS.cluster_bootstrap_mean([1, 2, 3], ["a", "b"])


def test_signflip_and_power_flag_a_test_that_could_not_have_seen_the_effect():
    rng = np.random.default_rng(5)
    cl = np.repeat(np.arange(24), 8)
    real = rng.normal(0.01, 0.005, 192)
    null = rng.normal(0.0, 0.02, 192)
    assert TS.cluster_signflip_p(real, cl, seed=1) < 0.01
    assert TS.cluster_signflip_p(null, cl, seed=1) > 0.05
    assert np.isnan(TS.cluster_signflip_p([], None))
    # 4 clusters of noisy data: an observed gain of 0.002 sits inside the blind zone
    few = rng.normal(0.002, 0.03, 40)
    pw = TS.power_analysis(few, np.repeat(np.arange(4), 10), observed=0.002)
    assert pw.underpowered and pw.mde_80 > 0.002 and "blind zone" in pw.statement()
    assert TS.power_analysis([0.1], None).underpowered


def test_adjust_many_keeps_nan_and_corrects_upwards():
    q = TS.adjust_many([0.01, 0.04, float("nan"), 0.03])
    assert np.isnan(q[2]) and (q[[0, 1, 3]] >= np.array([0.01, 0.04, 0.03])).all()
    assert TS.adjust_many([0.01, 0.04, 0.03], "holm").shape == (3,)


def test_classify_transfer_precedence_and_never_validated():
    rng = np.random.default_rng(6)
    cl = np.repeat(np.arange(30), 10)
    same = TS.cluster_bootstrap_mean(rng.normal(0.012, 0.01, 300), cl, seed=1)
    cross = TS.cluster_bootstrap_mean(rng.normal(0.011, 0.01, 300), cl, seed=2)
    ratio = TS.transfer_ratio(cross.mean, same.mean)
    spec = TS.specialisation(same, cross, ratio)
    ok = TS.classify_transfer(same, cross, ratio, spec)
    assert ok.label == TS.TransferVerdictLabel.GENERALISES and ok.validation == ValidationLabel.NOT_VALIDATED
    ident = TS.identity_gap(np.full(300, 0.01), np.zeros(300), cl, seed=1)
    assert TS.classify_transfer(same, cross, ratio, spec, ident=ident).label == TS.TransferVerdictLabel.IDENTITY_DEPENDENT
    thin = TS.cluster_bootstrap_mean([0.01, 0.02], ["a", "b"])
    assert TS.classify_transfer(thin, thin, TS.transfer_ratio(0.015, 0.015), spec).validation == ValidationLabel.INSUFFICIENT_EVIDENCE
    flat = TS.cluster_bootstrap_mean(rng.normal(0, 0.01, 300), cl, seed=4)
    assert TS.classify_transfer(flat, flat, TS.transfer_ratio(0.0, 0.0), TS.specialisation(flat, flat, TS.transfer_ratio(0.0, 0.0))).label == TS.TransferVerdictLabel.NO_LEARNING
    for v in (ok,):
        assert v.validation != ValidationLabel.VALIDATED


# ======================================================================================== transfer: unit table and firewall
def test_prepare_units_fails_closed_on_unfinished_outcomes_and_bad_rows():
    u = _flat_units()
    late = u.copy()
    late.loc[0, "mature"] = NOW
    with pytest.raises(FirewallBreach):
        T.prepare_units(late, NOW)
    kept, dropped = T.drop_immature(late, NOW)
    assert dropped == 1 and len(kept) == len(u) - 1
    dup = pd.concat([u, u.iloc[:1]])
    with pytest.raises(ValueError):
        T.prepare_units(dup, NOW)
    nan = u.copy()
    nan.loc[3, "learned"] = np.nan
    with pytest.raises(ValueError):
        T.prepare_units(nan, NOW)
    backwards = u.copy()
    backwards.loc[2, "mature"] = backwards.loc[2, "date"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        T.prepare_units(backwards, NOW)
    with pytest.raises(ValueError):
        T.prepare_units(u.drop(columns=["learned"]), NOW)
    assert len(T.prepare_units(pd.DataFrame(columns=T.REQUIRED), NOW)) == 0


def test_training_scope_refuses_outcomes_the_learner_could_not_have_used():
    d = T.prepare_units(_flat_units(), NOW)
    with pytest.raises(FirewallBreach):
        T.TrainingScope.from_units(d, dt.date(2010, 1, 1))              # scope claims to end before the training outcomes matured
    scope = T.TrainingScope.from_units(d, dt.date(2020, 1, 1))
    with pytest.raises(FirewallBreach):
        scope.validate(pd.Timestamp("2020-01-01"))                       # learned_through must be strictly before now
    scope.validate(NOW)


def test_replay_units_that_are_also_novel_or_late_break_the_scope():
    u, scope = _scored(_world(), T.rule_learner_fit, 2012)
    bad = dataclass_replace(scope, keys=frozenset(u.loc[u["year"] == "2014", "key"].iloc[:5]))   # 2014 units claimed as training replays
    with pytest.raises(FirewallBreach):
        T.evaluate_transfer(u, bad, NOW, n_boot=100)


def dataclass_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


# ======================================================================================== transfer: the planted evaluation
def test_rule_learner_generalises_and_identity_memoriser_is_over_specialised():
    u = _world(seed=1)
    good, scope_g = _scored(u, T.rule_learner_fit, 2012)
    bad, scope_b = _scored(u, T.identity_memoriser_fit, 2012)
    rg = T.evaluate_transfer(good, scope_g, NOW, axes=(T.Axis.YEAR,), n_boot=200)
    rb = T.evaluate_transfer(bad, scope_b, NOW, axes=(T.Axis.YEAR,), n_boot=200)
    assert rg.axes[T.Axis.YEAR].verdict.label == TS.TransferVerdictLabel.GENERALISES
    assert rg.cross_year_gain > 0.004
    # the memoriser earns on the situations it stored (replays) and nothing on the future
    yb = rb.axes[T.Axis.YEAR]
    assert yb.replay.mean > 0.008 and abs(yb.cross.mean) < 1e-9
    assert yb.verdict.label in (TS.TransferVerdictLabel.OVER_SPECIALISED, TS.TransferVerdictLabel.NO_LEARNING, TS.TransferVerdictLabel.HARMFUL, TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE)
    assert yb.verdict.label != TS.TransferVerdictLabel.GENERALISES


def test_novel_units_dated_before_learning_are_excluded_as_anachronistic():
    u = _world(seed=2)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    tr = d[d["year"].astype(int).isin([2011, 2012, 2013])]
    scope = T.TrainingScope.from_units(tr, dt.date(2014, 1, 1))
    out = d.copy()
    out["learned"] = out["base"] + T.rule_learner_fit(tr)(out) * (out["alt"] - out["base"])
    r = T.evaluate_transfer(out, scope, NOW, axes=(T.Axis.YEAR,), n_boot=100)
    ax = r.axes[T.Axis.YEAR]
    assert ax.n_anachronistic == int((d["year"] == "2010").sum())        # 2010 is novel but earlier than learned_through
    assert ax.cross.n == int((d["year"].astype(int) >= 2014).sum())     # only later years count toward the headline
    assert "anachronistic" in T.render_report(r)


def test_unknown_labels_are_untested_not_novel():
    u, scope = _scored(_world(seed=3), T.rule_learner_fit, 2012)
    u = u.copy()
    u["regime"] = "UNKNOWN"
    r = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.REGIME,), n_boot=100)
    ax = r.axes[T.Axis.REGIME]
    assert ax.n_unknown == len(u) and not ax.tested and T.Axis.REGIME in r.untested_axes
    assert np.isnan(r.cross_regime_gain)


def test_regime_bound_rule_fails_leave_one_regime_out_while_a_universal_rule_passes():
    flip = _world(seed=4, beta_flips_in="BEAR", n_years=6, weeks_per_year=30)
    plain = _world(seed=4, n_years=6, weeks_per_year=30)
    res_flip = T.cross_validate_transfer(flip, T.rule_learner_fit, NOW, axes=(T.Axis.REGIME,), n_boot=150)[T.Axis.REGIME]
    res_plain = T.cross_validate_transfer(plain, T.rule_learner_fit, NOW, axes=(T.Axis.REGIME,), n_boot=150)[T.Axis.REGIME]
    assert res_flip.cross.mean < 0.5 * res_plain.cross.mean            # a rule learned in one regime does not survive the other
    assert res_plain.verdict.label == TS.TransferVerdictLabel.GENERALISES
    assert res_flip.verdict.label != TS.TransferVerdictLabel.GENERALISES


def test_harness_selfcheck_can_tell_a_rule_from_a_lookup_table():
    chk = T.harness_selfcheck(seed=5, n_boot=150)
    assert chk["rule"]["YEAR"][0] == "GENERALISES" and chk["rule"]["STOCK"][0] == "GENERALISES"
    assert chk["identity_memoriser"]["YEAR"][0] == "OVER_SPECIALISED" and chk["identity_memoriser"]["STOCK"][0] == "OVER_SPECIALISED"
    # a per-ticker memoriser transfers across years (tickers persist) but not to new tickers: only the STOCK axis exposes it
    assert chk["ticker_memoriser"]["STOCK"][0] == "OVER_SPECIALISED"
    assert chk["ticker_memoriser"]["YEAR"][0] == "GENERALISES"


def test_make_folds_year_folds_are_forward_and_purged():
    d = T.prepare_units(_world(seed=6).assign(learned=lambda x: x["base"]), NOW)
    folds = T.make_folds(d, T.Axis.YEAR)
    assert folds and all(f.forward for f in folds)
    for f in folds:
        assert d.iloc[f.train_idx]["mature"].max() < d.iloc[f.test_idx]["date"].min()
        assert not set(f.train_idx) & set(f.test_idx)
    sf = T.make_folds(d, T.Axis.STOCK, seed=1, n_stock_folds=4)
    assert len(sf) == 4 and not any(f.forward for f in sf)
    for f in sf:
        assert not set(d.iloc[f.train_idx]["ticker"]) & set(d.iloc[f.test_idx]["ticker"])


def test_run_folds_rejects_a_learner_that_returns_bad_weights_and_a_broken_purge():
    u = _world(seed=7)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    folds = T.make_folds(d, T.Axis.YEAR)
    with pytest.raises(ValueError):
        T.run_folds(u, lambda tr: (lambda te: np.full(len(te), 1.5)), folds, NOW)
    with pytest.raises(ValueError):
        T.run_folds(u, lambda tr: (lambda te: np.zeros(len(te) - 1)), folds, NOW)
    f = folds[0]
    broken = T.Fold(f.axis, f.test_label, np.concatenate([f.train_idx, f.test_idx[:1]]), f.test_idx, True)
    with pytest.raises(FirewallBreach):
        T.run_folds(u, T.rule_learner_fit, [broken], NOW)


def test_identity_gap_appears_only_on_trained_situations_for_a_memoriser():
    u = _world(seed=8)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    f = T.make_folds(d, T.Axis.YEAR)[-1]
    rule = T.identity_probe_units(u, T.rule_learner_fit, f, NOW, on="train")
    mem = T.identity_probe_units(u, T.identity_memoriser_fit, f, NOW, on="train")
    g_rule = TS.identity_gap(rule["learned"] - rule["base"], rule["learned_disguised"] - rule["base"], rule["cluster"], seed=1)
    g_mem = TS.identity_gap(mem["learned"] - mem["base"], mem["learned_disguised"] - mem["base"], mem["cluster"], seed=1)
    assert not g_rule.flagged and g_mem.flagged and g_mem.relative > 0.9
    with pytest.raises(ValueError):
        T.identity_probe_units(u, T.rule_learner_fit, f, NOW, on="nowhere")


def test_evaluate_transfer_reports_identity_gap_and_classifies_it():
    u, scope = _scored(_world(seed=9), T.rule_learner_fit, 2012)
    u = u.copy()
    fwd = u["date"] > pd.Timestamp(scope.learned_through)
    u["learned_disguised"] = u["base"]                                     # the whole benefit vanishes when identities are scrambled
    u.loc[~fwd, "learned_disguised"] = u.loc[~fwd, "learned"]
    r = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=150)
    assert r.identity is not None and r.identity.flagged
    assert r.axes[T.Axis.YEAR].verdict.label == TS.TransferVerdictLabel.IDENTITY_DEPENDENT and r.overall == TS.TransferVerdictLabel.IDENTITY_DEPENDENT


def test_isolated_mode_compares_single_axis_novelty_only():
    u, scope = _scored(_world(seed=10), T.rule_learner_fit, 2012)
    marginal = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), mode="marginal", n_boot=100)
    isolated = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), mode="isolated", n_boot=100)
    assert isolated.axes[T.Axis.YEAR].mode == "isolated"
    assert 0 < isolated.axes[T.Axis.YEAR].cross.n <= marginal.axes[T.Axis.YEAR].cross.n
    with pytest.raises(ValueError):
        T.evaluate_transfer(u, scope, NOW, mode="nope")


def test_empty_units_give_insufficient_evidence_everywhere():
    scope = T.TrainingScope(dt.date(2020, 1, 1))
    r = T.evaluate_transfer(pd.DataFrame(columns=T.REQUIRED), scope, NOW, n_boot=100)
    assert r.n_units == 0 and r.overall == TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE and r.label == ValidationLabel.INSUFFICIENT_EVIDENCE
    assert np.isnan(r.cross_year_gain) and np.isnan(r.same_year_gain)
    assert T.transfer_confidence(r) is None
    assert T.axis_result_from_folds(T.Axis.YEAR, []).verdict.validation == ValidationLabel.INSUFFICIENT_EVIDENCE
    assert len(T.time_decay(pd.DataFrame(columns=T.REQUIRED), scope, NOW).frame) == 0
    assert T.walk_forward_transfer(pd.DataFrame(columns=T.REQUIRED + ("alt",)), T.rule_learner_fit, NOW).table.empty


# ======================================================================================== transfer: confidence, records, reports
def test_transfer_confidence_is_zero_for_a_memoriser_and_positive_for_a_rule():
    u = _world(seed=11)
    good, sg = _scored(u, T.rule_learner_fit, 2012)
    bad, sb = _scored(u, T.identity_memoriser_fit, 2012)
    cg = T.transfer_confidence(T.evaluate_transfer(good, sg, NOW, axes=(T.Axis.YEAR,), n_boot=150))
    cb = T.transfer_confidence(T.evaluate_transfer(bad, sb, NOW, axes=(T.Axis.YEAR,), n_boot=150))
    assert cg is not None and cg > 0.5
    assert cb == 0.0 or cb is None or cb < 0.2
    conf = T.knowledge_confidence(T.evaluate_transfer(good, sg, NOW, axes=(T.Axis.YEAR,), n_boot=150))
    assert conf.transfer == pytest.approx(cg) and conf.truth is None and not conf.check()


def test_report_record_is_content_addressed_and_render_never_says_improved():
    u, scope = _scored(_world(seed=12), T.rule_learner_fit, 2012)
    a = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=150, seed=1)
    b = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=150, seed=1)
    assert a.as_record()["report_id"] == b.as_record()["report_id"]
    assert "improved" not in T.render_report(a).lower() and "VALIDATED" not in T.render_report(a).replace("NOT VALIDATED", "")
    c = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=150, seed=2)
    assert c.as_record()["report_id"] != a.as_record()["report_id"]
    assert set(a.headline()) >= {"same_year_gain", "cross_year_gain", "cross_regime_gain", "cross_stock_gain", "cross_sector_gain"}


def test_save_report_is_atomic_content_addressed_and_refuses_a_different_report(tmp_path, monkeypatch):
    from engine import provenance
    monkeypatch.setattr(provenance, "stamp", lambda cfg=None, seed=None: {"code_hash": "test", "seed": seed})
    u, scope = _scored(_world(seed=13), T.rule_learner_fit, 2012)
    r1 = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=120, seed=1)
    r2 = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=120, seed=2)
    paths = T.save_report(r1, tmp_path, seed=1, name="rep")
    rec = json.loads(open(paths["json"], encoding="utf-8").read())
    assert rec["report_id"] == r1.as_record()["report_id"] and rec["provenance"]["seed"] == 1
    T.save_report(r1, tmp_path, seed=1, name="rep")                       # same content: idempotent
    with pytest.raises(FileExistsError):
        T.save_report(r2, tmp_path, seed=2, name="rep")
    assert not list(tmp_path.glob("*.tmp"))


def test_compare_reports_separates_better_and_worse_versions():
    u = _world(seed=14)
    good, sg = _scored(u, T.rule_learner_fit, 2012)
    bad, sb = _scored(u, T.identity_memoriser_fit, 2012)
    rg = T.evaluate_transfer(good, sg, NOW, axes=(T.Axis.YEAR,), n_boot=150)
    rb = T.evaluate_transfer(bad, sb, NOW, axes=(T.Axis.YEAR,), n_boot=150)
    assert T.compare_reports(rb, rg)["better"] == ["YEAR"]
    assert T.compare_reports(rg, rb)["worse"] == ["YEAR"]
    assert T.compare_reports(rg, rg)["same"] == ["YEAR"]


# ======================================================================================== transfer: context labellers
def test_regime_labels_use_only_the_past_and_refuse_a_series_that_reaches_now():
    idx = pd.date_range("2015-01-05", periods=200, freq="W-MON")
    r = pd.Series(np.random.default_rng(0).normal(0.002, 0.02, 200), index=idx)
    lab = T.label_regimes(r, NOW, lookback=20, vol_lookback=8)
    assert lab.iloc[:19].eq("UNKNOWN").all() and lab.iloc[40:].ne("UNKNOWN").all()
    r2 = r.copy()
    r2.iloc[150:] = -0.2                                              # a crash in the future must not touch earlier labels
    lab2 = T.label_regimes(r2, NOW, lookback=20, vol_lookback=8)
    assert lab.iloc[:150].equals(lab2.iloc[:150])
    with pytest.raises(FirewallBreach):
        T.label_regimes(r, idx[-1], lookback=20, vol_lookback=8)
    asof = T.regime_for_dates(lab, [idx[60] + pd.Timedelta(days=3), idx[0] - pd.Timedelta(days=30)])
    assert asof.iloc[0] == lab.iloc[60] and asof.iloc[1] == "UNKNOWN"


def test_volatility_buckets_and_stock_types_and_attach_context():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    sd = {"A": 0.005, "B": 0.01, "C": 0.03}
    rets = pd.DataFrame({t: np.random.default_rng(i).normal(0, s, 80) for i, (t, s) in enumerate(sd.items())}, index=idx)
    vb = T.label_volatility(rets, lookback=30, n_buckets=3)
    last = vb.xs(idx[-1], level="date")
    assert last["A"] == "V1" and last["C"] == "V3"
    assert idx[10] not in vb.index.get_level_values("date")             # no full window yet: absent, later UNKNOWN
    types = T.label_stock_type({"A": 0.5, "B": 1.5, "C": 1.0, "D": 0.7}, {"A": 1e6, "B": 5e6, "C": 9e6, "D": 2e6}, "2029-12-01", NOW)
    assert set(types) == {"A", "B", "C", "D"} and {v.split("_")[0] for v in types.values()} == {"HB", "LB"}
    with pytest.raises(FirewallBreach):
        T.label_stock_type({"A": 1.0}, {"A": 1.0}, "2030-01-01", NOW)
    units = pd.DataFrame({"date": [idx[50], idx[5]], "mature": [idx[55], idx[10]], "ticker": ["A", "C"], "base": 0.0, "learned": 0.0})
    a = T.attach_context(units, vol=vb, sectors={"A": "TECH"}, stock_types=types)
    assert a.loc[0, "vol_bucket"] == "V1" and a.loc[1, "vol_bucket"] == "UNKNOWN" and a.loc[1, "sector"] == "UNKNOWN"


def test_scope_from_knowledge_uses_item_contexts_and_refuses_items_from_the_future():
    class K:
        def __init__(self, kid, learned, ctx):
            self.knowledge_id, self.version = kid, 1
            self.epistemic = self.lifecycle = self.promotion = self.confidence = None
            self.provenance = Provenance(created_real="2029-01-01T00:00:00", learned_at=learned, code_hash="h", outcomes_seen_through=learned)
            self.contexts, self.anti_contexts, self.decision_effect = ctx, {}, ()
    train = T.prepare_units(_flat_units(), NOW)
    ok = K("k1", "2020-06-01", {"year": ["2010", "2011"], "sector": "S1"})
    sc = T.scope_from_knowledge([ok], train, NOW)
    assert sc.years == frozenset({"2010", "2011"}) and sc.sectors == frozenset({"S1"}) and sc.learned_through == dt.date(2020, 6, 1)
    with pytest.raises(FirewallBreach):
        T.scope_from_knowledge([K("k2", "2031-01-01", {})], train, NOW)
    with pytest.raises(ValueError):
        T.scope_from_knowledge([], train, NOW)
    assert KnowledgeLike.conforms(ok) == []


# ======================================================================================== transfer: decay, interaction, permutation, matrix
def test_time_decay_finds_a_half_life_only_when_the_gain_really_decays():
    lt = dt.date(2012, 12, 31)

    def decaying(date, i, rng):
        days = max((date - pd.Timestamp(lt)).days, 0)
        return 0.02 * 0.5 ** (days / 180.0) + rng.normal(0, 0.002)
    d = _flat_units(n_clusters=200, per=8, gain_fn=decaying, years=(2013, 2014, 2015, 2016, 2017))
    scope = T.TrainingScope(lt)
    dec = T.time_decay(d, scope, NOW, bin_days=91)
    assert dec.decays and 100 < dec.half_life < 400 and dec.spearman < 0
    const = _flat_units(n_clusters=200, per=8, gain_fn=lambda date, i, rng: 0.02 + rng.normal(0, 0.002), years=(2013, 2014, 2015, 2016, 2017))
    assert not T.time_decay(const, scope, NOW, bin_days=91).decays


def test_distance_decay_follows_a_gain_that_shrinks_with_unfamiliarity():
    rng = np.random.default_rng(15)
    n = 1500
    dates = pd.Timestamp("2014-01-06") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    x = rng.normal(0, 1.5, n)
    gain = 0.02 * np.exp(-np.abs(x) / 1.2) + rng.normal(0, 0.004, n)
    fwd = pd.DataFrame({"date": dates, "mature": dates + pd.Timedelta(days=7), "ticker": [f"F{i}" for i in range(n)], "base": 0.0, "learned": gain, "ctx": x})
    tdates = pd.Timestamp("2010-01-04") + pd.to_timedelta(rng.integers(0, 900, 400), unit="D")
    train = pd.DataFrame({"date": tdates, "mature": tdates + pd.Timedelta(days=7), "ticker": [f"R{i}" for i in range(400)], "base": 0.0, "learned": 0.0, "ctx": rng.normal(0, 0.3, 400)})
    scope = T.TrainingScope(dt.date(2013, 1, 1))
    dec = T.distance_decay(fwd, train, scope, NOW, ["ctx"], n_bins=4, n_boot=150)
    assert dec.decays and dec.spearman < -0.1 and dec.frame["gain"].iloc[0] > dec.frame["gain"].iloc[-1]
    flat = fwd.assign(learned=0.01 + rng.normal(0, 0.004, n))
    assert not T.distance_decay(flat, train, scope, NOW, ["ctx"], n_bins=4, n_boot=150).decays
    with pytest.raises(ValueError):
        T.distance_decay(fwd, train, scope, NOW, ["missing"])


def test_specialisation_permutation_test_separates_real_specialisation_from_noise():
    def cells(fam_gain, nov_gain):
        rows, rng = [], np.random.default_rng(16)
        for c in range(16):
            fam = c % 2 == 0
            date = pd.Timestamp(year=2014, month=1 + c % 12, day=3 + c // 12)
            for i in range(8):
                rows.append((date + pd.Timedelta(days=i), date + pd.Timedelta(days=i + 7), f"Q{c}_{i}", 0.0,
                             (fam_gain if fam else nov_gain) + rng.normal(0, 0.002), "2011" if fam else "2014"))
        df = pd.DataFrame(rows, columns=["date", "mature", "ticker", "base", "learned", "year"])
        return df
    scope = T.TrainingScope(dt.date(2013, 1, 1), years=frozenset({"2011"}))
    special = specialisation_test(cells(0.02, 0.0), scope)
    same = specialisation_test(cells(0.01, 0.01), scope)
    assert special["p"] < 0.02 and special["stat"] > 0.015
    assert same["p"] > 0.1
    assert np.isnan(T.specialisation_permutation_test(pd.DataFrame(columns=T.REQUIRED), scope, NOW)["p"])


def specialisation_test(df, scope):
    return T.specialisation_permutation_test(df, scope, NOW, T.Axis.YEAR, n_perm=400, seed=1)


def test_interaction_table_shows_where_the_lesson_stops():
    u, scope = _scored(_world(seed=17), T.rule_learner_fit, 2012)
    tab = T.interaction_table(u, scope, NOW, T.Axis.YEAR, T.Axis.STOCK, n_boot=100)
    assert set(tab["YEAR"]) == {"familiar", "novel"} and len(tab) == 4
    nov_fam = tab[(tab["YEAR"] == "novel") & (tab["STOCK"] == "familiar")].iloc[0]
    assert nov_fam["enough"] and nov_fam["gain"] > 0.004
    with pytest.raises(ValueError):
        T.interaction_table(u, scope, NOW, T.Axis.YEAR, T.Axis.YEAR)


def test_transfer_matrix_diagonal_and_offdiagonal_expose_a_memoriser():
    u = _world(seed=18)
    rule = T.transfer_matrix(u, T.rule_learner_fit, NOW).summary()
    mem = T.transfer_matrix(u, T.identity_memoriser_fit, NOW).summary()
    assert rule["diagonal_mean"] > 0.004 and rule["offdiagonal_mean"] > 0.004 and rule["share_offdiagonal_positive"] == 1.0
    assert mem["offdiagonal_mean"] == pytest.approx(0.0, abs=1e-9) and mem["diagonal_mean"] == pytest.approx(0.0, abs=1e-9)
    fwd = T.transfer_matrix(u, T.rule_learner_fit, NOW)
    assert fwd.forward.values.any() and not fwd.forward.values[np.tril_indices(len(fwd.forward), -1)].any()      # only later years are forward tests


def test_walk_forward_gives_a_track_record_and_flags_a_learner_that_reads_the_answer():
    u = _world(seed=19)
    wf = T.walk_forward_transfer(u, T.rule_learner_fit, NOW, min_train_units=300, n_boot=150)
    assert len(wf.table) >= 4 and wf.pooled.lo > 0 and wf.stability.stable and wf.p_signflip < 0.05
    dead = T.walk_forward_transfer(u, lambda tr: (lambda te: np.zeros(len(te))), NOW, min_train_units=300, n_boot=150)
    assert dead.pooled.mean == 0.0 and not dead.stability.stable
    with pytest.raises(ValueError):
        T.walk_forward_transfer(u.drop(columns=["alt"]).assign(learned=u["base"]), T.rule_learner_fit, NOW)


def test_evaluate_by_era_and_item_table_and_history():
    u, scope = _scored(_world(seed=20), T.rule_learner_fit, 2012)
    by_era = T.evaluate_by_era(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=100)
    assert set(by_era) == {"2001+"} and by_era["2001+"].n_units == len(u)
    good, sg = _scored(_world(seed=21), T.rule_learner_fit, 2012)
    rng = np.random.default_rng(1)
    null = good.copy()
    null["learned"] = null["base"] + rng.normal(0, 0.003, len(null))
    tab = T.item_transfer_table({"real": good, "null": null}, {"real": sg, "null": sg}, NOW, n_boot=120)
    row = tab.set_index("item")
    assert row.loc["real", "p"] < 0.05 and row.loc["null", "p"] > 0.05 and row.loc["real", "q"] >= row.loc["real", "p"]
    with pytest.raises(KeyError):
        T.item_transfer_table({"x": good}, {}, NOW)
    h = T.TransferHistory()
    bad, sb = _scored(_world(seed=21), T.identity_memoriser_fit, 2012)
    h.add("v1", T.evaluate_transfer(good, sg, NOW, axes=(T.Axis.YEAR,), n_boot=120))
    h.add("v2", T.evaluate_transfer(bad, sb, NOW, axes=(T.Axis.YEAR,), n_boot=120))
    assert h.regressions() == ["v2 transfers worse than v1 on YEAR"]
    with pytest.raises(ValueError):
        h.add("v1", T.evaluate_transfer(good, sg, NOW, axes=(T.Axis.YEAR,), n_boot=120))


def test_protocol_is_validated_hashed_and_equivalent_to_the_direct_call():
    p = T.TransferProtocol(axes=(T.Axis.YEAR,), n_boot=150, seed=3)
    assert p.fingerprint() == T.TransferProtocol(axes=(T.Axis.YEAR,), n_boot=150, seed=3).fingerprint()
    assert p.fingerprint() != T.TransferProtocol(axes=(T.Axis.YEAR,), n_boot=150, seed=4).fingerprint()
    assert T.TransferProtocol(n_boot=10, mode="x").validate()
    u, scope = _scored(_world(seed=22), T.rule_learner_fit, 2012)
    direct = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR,), n_boot=150, seed=3)
    assert p.run(u, scope, NOW).as_record()["report_id"] == direct.as_record()["report_id"]
    with pytest.raises(ValueError):
        T.TransferProtocol(n_boot=10).run(u, scope, NOW)


def test_reference_controls_behave_as_documented():
    u = _world(seed=23)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    tr, te = d[d["year"].astype(int) <= 2012], d[d["year"].astype(int) > 2012]
    w_mem = T.identity_memoriser_fit(tr)(te)
    assert w_mem.sum() == 0                                             # unseen keys: it knows nothing
    w_rep = T.identity_memoriser_fit(tr)(tr)
    assert w_rep.sum() > 0 and ((tr["alt"] - tr["base"]).to_numpy()[w_rep == 1] > 0).all()
    r1, r2 = T.random_learner_fit(1)(tr)(te), T.random_learner_fit(1)(tr)(te)
    assert (r1 == r2).all() and 0.3 < r1.mean() < 0.7                   # seeded and about a coin flip
    assert not (T.random_learner_fit(2)(tr)(te) == r1).all()
    sf = T.scrambled_frame(te, 5)
    assert sorted(sf["ticker"].unique()) == sorted(te["ticker"].unique()) and (sf["date"] != te["date"]).all()
    assert np.allclose(sf["f1"], te["f1"])


# ======================================================================================== report tables and planning
def test_clusters_needed_inverts_the_power_analysis():
    n = TS.clusters_needed(0.005, 0.01)
    assert n == 25                                           # ((1.645 + 0.842) * 0.01 / 0.005) ** 2 = 24.75, rounded up
    assert 2.487 * 0.01 / np.sqrt(n) <= 0.005 < 2.487 * 0.01 / np.sqrt(n - 1)       # n is the smallest count whose MDE reaches the effect
    with pytest.raises(ValueError):
        TS.clusters_needed(0.0, 0.01)
    with pytest.raises(ValueError):
        TS.clusters_needed(0.01, -1)
    assert TS.clusters_needed(0.01, 0.01) < TS.clusters_needed(0.005, 0.01)     # a bigger effect needs less data


def test_axis_and_group_tables_and_planning_note_keep_untested_axes_visible():
    u, scope = _scored(_world(seed=30), T.rule_learner_fit, 2012)
    r = T.evaluate_transfer(u, scope, NOW, axes=(T.Axis.YEAR, T.Axis.REGIME), n_boot=120)
    tab = T.axis_table(r)
    assert list(tab["axis"]) == ["YEAR", "REGIME"] and tab.loc[0, "label"] == "GENERALISES" and np.isnan(tab.loc[1, "cross"])
    g = T.group_table(r, T.Axis.YEAR)
    assert list(g.columns) == ["group", "n", "gain"] and g["gain"].is_monotonic_increasing and len(g) == 3
    assert T.group_table(r, T.Axis.SECTOR).empty
    note = T.planning_note(r, T.Axis.YEAR, 0.001)
    assert "clusters" in note and T.planning_note(r, T.Axis.REGIME) == ""
    assert TS.verdict_table([{"axis": "X"}]).loc[0, "label"] == ""


# ======================================================================================== section 26/27 completion
def test_over_specialisation_verdict_bootstraps_the_ratio_with_guards():
    rng = np.random.default_rng(40)
    cl = np.repeat(np.arange(30), 10)
    same = rng.normal(0.012, 0.006, 300)
    v_bad = TS.over_specialisation_verdict(rng.normal(0.0005, 0.006, 300), same, cl, cl, seed=1)
    v_ok = TS.over_specialisation_verdict(rng.normal(0.0115, 0.006, 300), same, cl, cl, seed=1)
    assert v_bad.label == "OVER_SPECIALISED" and v_bad.p_below_floor >= 0.95 and v_bad.ratio_hi < TS.RATIO_FLOOR + 0.2
    assert v_ok.label == "GENERAL" and v_ok.ratio_lo > TS.RATIO_FLOOR
    mid = TS.over_specialisation_verdict(rng.normal(0.0045, 0.02, 300), rng.normal(0.012, 0.02, 300), cl, cl, seed=1)
    assert mid.label in ("SUSPECTED", "UNDEFINED")
    zero_den = TS.over_specialisation_verdict(rng.normal(0.01, 0.006, 300), rng.normal(0.0, 0.02, 300), cl, cl, seed=1)
    assert zero_den.label == "UNDEFINED" and zero_den.ratio_lo is None and zero_den.denominator_risk > 0.05
    neg_den = TS.over_specialisation_verdict(rng.normal(-0.01, 0.006, 300), rng.normal(-0.02, 0.006, 300), cl, cl, seed=1)
    assert neg_den.label == "UNDEFINED"                                  # negative/negative never becomes a good ratio
    assert TS.over_specialisation_verdict([], [], None, None).label == "UNDEFINED"


def test_context_gain_table_gives_every_group_its_own_interval_and_keeps_thin_groups():
    u, scope = _scored(_world(seed=41, n_years=7), T.rule_learner_fit, 2012)
    for ax, expect in ((T.Axis.SECTOR, 4), (T.Axis.YEAR, 4), (T.Axis.VOLATILITY, 3), (T.Axis.STOCK_TYPE, 4)):
        tb = T.context_gain_table(u, scope, NOW, ax, n_boot=100)
        assert len(tb) >= 1 and list(tb.columns) == T.CONTEXT_GAIN_COLS
    sec = T.context_gain_table(u, scope, NOW, T.Axis.SECTOR, n_boot=100)
    assert len(sec) == 4 and sec["familiar"].all() and (sec["lo"] > 0).all() and (sec["p_signflip"] < 0.05).all()
    yr = T.context_gain_table(u, scope, NOW, T.Axis.YEAR, n_boot=100)
    assert not yr["familiar"].any()                                       # the forward years were never trained on
    thin = T.context_gain_table(u, scope, NOW, T.Axis.YEAR, n_boot=100, min_units=10 ** 6)
    assert thin["gain"].isna().all() and len(thin) == len(yr)
    assert T.context_gain_table(pd.DataFrame(columns=T.REQUIRED), scope, NOW, T.Axis.YEAR).empty


def test_market_condition_gains_use_training_cut_points_and_flag_unreached_bands():
    rng = np.random.default_rng(42)
    n = 1200
    dates = pd.Timestamp("2014-01-06") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    vix = rng.normal(0, 1, n)
    gain = np.where(vix > 0.5, -0.01, 0.01) + rng.normal(0, 0.003, n)            # the lesson breaks in high-stress conditions
    fwd = pd.DataFrame({"date": dates, "mature": dates + pd.Timedelta(days=7), "ticker": [f"M{i}" for i in range(n)], "base": 0.0, "learned": gain, "vix": vix})
    td = pd.Timestamp("2010-01-04") + pd.to_timedelta(rng.integers(0, 700, 500), unit="D")
    train = pd.DataFrame({"date": td, "mature": td + pd.Timedelta(days=7), "ticker": [f"R{i}" for i in range(500)], "base": 0.0, "learned": 0.0,
                          "vix": rng.uniform(-1, 0.5, 500)})
    scope = T.TrainingScope(dt.date(2013, 1, 1))
    tb = T.market_condition_gains(fwd, train, scope, NOW, ["vix"], n_bins=3, n_boot=100)
    top = tb[(tb["band"] == 2) & tb["outside_training"]]
    assert len(top) == 1 and top["gain"].iloc[0] < 0 and top["hi"].iloc[0] < 0      # beyond anything the learner saw, it hurts
    calm = tb[(tb["band"] == 0) & ~tb["outside_training"]]
    assert len(calm) == 1 and calm["lo"].iloc[0] > 0
    with pytest.raises(ValueError):
        T.market_condition_gains(fwd, train, scope, NOW, ["nope"])


def test_full_context_report_counts_contexts_that_gained_and_lost():
    u, scope = _scored(_world(seed=43, n_years=7), T.rule_learner_fit, 2012)
    d = T.prepare_units(u, NOW)
    rep = T.full_context_report(u, scope, NOW, axes=(T.Axis.YEAR, T.Axis.SECTOR, T.Axis.VOLATILITY), n_boot=100)
    cov = rep["coverage"]
    assert cov["SECTOR"]["gained"] == 4 and cov["SECTOR"]["lost"] == 0 and cov["SECTOR"]["p_more_gain_than_loss"] < 0.1
    assert cov["YEAR"]["contexts"] >= 1 and set(rep["tables"]) == {"YEAR", "SECTOR", "VOLATILITY"}
    bad, sb = _scored(_world(seed=43, n_years=7), T.identity_memoriser_fit, 2012)
    assert T.full_context_report(bad, sb, NOW, axes=(T.Axis.SECTOR,), n_boot=100)["coverage"]["SECTOR"]["gained"] == 0


def test_rule_ledger_derives_status_from_tests_and_orders_them_in_time():
    led = T.RuleTransferLedger()
    good = TS.BootMean(0.01, 0.005, 0.015, 500, 20)
    assert led.status("r1") == "UNTESTED" and led.outstanding("r1") == list(T.REQUIRED_TRANSFER_AXES)
    for ax in ("YEAR", "REGIME", "STOCK"):
        led.record("r1", ax, "2029-01-01", good, TS.TransferVerdictLabel.GENERALISES)
    assert led.status("r1") == "PARTIAL" and led.outstanding("r1") == ["SECTOR", "VOLATILITY", "MARKET_CONDITION"]
    for ax in ("SECTOR", "VOLATILITY", "MARKET_CONDITION"):
        led.record("r1", ax, "2029-06-01", good, TS.TransferVerdictLabel.GENERALISES)
    assert led.status("r1") == "COMPLETE" and led.overdue(NOW, 365) == []
    assert led.overdue(pd.Timestamp("2031-01-01"), 365) == ["r1"]           # stale tests come due again
    led.record("r2", "YEAR", "2029-01-01", good, TS.TransferVerdictLabel.OVER_SPECIALISED)
    assert led.status("r2") == "FAILED" and set(led.summary()["rule"]) == {"r1", "r2"}
    with pytest.raises(FirewallBreach):
        led.record("r1", "YEAR", "2028-01-01", good, TS.TransferVerdictLabel.GENERALISES)
    with pytest.raises(ValueError):
        led.record("r1", "MOON", "2030-01-01", good, TS.TransferVerdictLabel.GENERALISES)
    back = T.ledger_from_records(T.ledger_records(led))
    assert back.status("r1") == "COMPLETE" and back.status("r2") == "FAILED"
    bad = T.ledger_records(led)[::-1]
    with pytest.raises(FirewallBreach):
        T.ledger_from_records([r for r in bad if r["rule"] == "r1"])


def test_tracked_gains_from_folds_names_every_section_26_quantity_and_ledger_takes_them():
    u = _world(seed=44, n_years=6, weeks_per_year=30)
    tr = T.tracked_gains_from_folds(u, T.rule_learner_fit, NOW, n_boot=120)
    assert set(tr) >= {"same_year_gain", "cross_year_gain", "cross_regime_gain", "cross_stock_gain", "cross_sector_gain", "cross_volatility_gain"}
    for k in ("cross_year_gain", "cross_regime_gain", "cross_stock_gain", "cross_sector_gain", "cross_volatility_gain"):
        assert tr[k]["gain"].lo > 0 and tr[k]["verdict"] == "GENERALISES" and tr[k]["specialisation"].label in ("GENERAL", "UNDEFINED", "SUSPECTED")
    led = T.RuleTransferLedger()
    assert T.record_tracked(led, "rule", tr, "2029-01-01") == 5 and led.outstanding("rule") == ["MARKET_CONDITION"]
    mem = T.tracked_gains_from_folds(u, T.identity_memoriser_fit, NOW, n_boot=120)
    assert mem["cross_year_gain"]["specialisation"].label in ("OVER_SPECIALISED", "UNDEFINED")
    led2 = T.RuleTransferLedger()
    T.record_tracked(led2, "mem", mem, "2029-01-01")
    assert led2.status("mem") == "FAILED"
    one_label = u.assign(regime="BULL_CALM")
    assert T.tracked_gains_from_folds(one_label, T.rule_learner_fit, NOW, n_boot=100)["cross_regime_gain"] is None      # nothing to hold out: untested


def test_market_condition_result_reaches_the_ledger_only_with_two_measured_bands():
    led = T.RuleTransferLedger()
    ok = pd.DataFrame({"gain": [0.01, 0.012], "lo": [0.005, 0.006], "hi": [0.015, 0.018], "n": [200, 300]})
    bad = pd.DataFrame({"gain": [0.01, -0.02], "lo": [0.005, -0.03], "hi": [0.015, -0.01], "n": [200, 300]})
    assert T.record_market_conditions(led, "a", ok, "2029-01-01") and led.latest("a")["MARKET_CONDITION"]["verdict"] == "GENERALISES"
    assert T.record_market_conditions(led, "b", bad, "2029-01-01") and led.status("b") == "FAILED"
    assert not T.record_market_conditions(led, "c", ok.iloc[:1], "2029-01-01")
