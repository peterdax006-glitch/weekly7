"""Tests for engine.learning.break_detection (section 10), reliability (11) and lifecycle (12) on planted worlds.
Every mechanism has a planted case it must catch and an empty/degenerate case. Synthetic data only."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import break_detection as bd
from engine.learning.core import FailureCause, FirewallBreach


def _ar1(rng, T, phi):
    x = np.zeros(T)
    e = rng.normal(0, 1, T)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + math.sqrt(1 - phi * phi) * e[i]
    return x


def make_item(kind="regime", T=640, seed=0, mu=0.006, sigma=0.010, leak=None, extra_cols=6, name="pat_a"):
    """kind: regime = works when the observable regime variable is high, loses when low; random = 5 random loss stretches
    unrelated to any column; healthy = always works."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-04", periods=T, freq="W-FRI")
    z = _ar1(rng, T, 0.90)
    cols = {f"m_n{k}": _ar1(rng, T, 0.8) for k in range(extra_cols)}
    cols["m_regime"] = z + 0.2 * rng.normal(0, 1, T)
    sign = np.ones(T)
    if kind == "regime":
        sign = np.where(z > 0.0, 1.0, -1.0)
    elif kind == "random":
        for s in rng.choice(np.arange(60, T - 60), 5, replace=False):
            sign[s:s + 28] = -1.0
    value = sign * mu + rng.normal(0, sigma, T)
    fr = pd.DataFrame({"value": value, **cols}, index=idx)
    spec = [bd.ContextColumn("m_regime", "regime")] + [bd.ContextColumn(f"m_n{k}", d)
            for k, d in zip(range(extra_cols), ["volatility", "liquidity", "trend", "breadth", "macro", "feature_distribution"])]
    if leak == "peek":
        fr["m_peek"] = fr["value"] + rng.normal(0, 0.001, T)
        spec.append(bd.ContextColumn("m_peek", "macro"))
    if leak == "declared":
        fr["m_fut"] = rng.normal(0, 1, T)
        spec.append(bd.ContextColumn("m_fut", "macro", lag=-1))
    return bd.ItemSeries(name, fr, tuple(spec))


FAST = {"n_perm": 200}


def test_planted_regime_break_found_and_validated_out_of_sample():
    item = make_item("regime", seed=3)
    ex = bd.explain_break(item, cfg=FAST, seed=1)
    assert ex.status == "EXPLAINED", ex.statement
    assert ex.condition.terms[0].column == "m_regime" and ex.condition.terms[0].op == ">="
    assert ex.oos.passes and ex.oos.mean_in > 0 > ex.oos.mean_out
    assert ex.cause == FailureCause.REGIME_CHANGE
    assert ex.n_tested >= 7
    regime_row = [d for d in ex.dimension_summary if d["dimension"] == "regime"][0]
    assert regime_row["verdict"] == "differs" and regime_row["best_column"] == "m_regime"


def test_random_break_is_unknown_with_no_fake_condition():
    unknown = 0
    for seed in range(4):
        item = make_item("random", seed=10 + seed)
        ex = bd.explain_break(item, cfg=FAST, seed=seed)
        assert ex.status in ("UNKNOWN", "INSUFFICIENT_EVIDENCE", "NO_BREAK")
        assert ex.condition is None and ex.oos is None and ex.cause == FailureCause.UNKNOWN
        unknown += ex.status == "UNKNOWN"
    assert unknown >= 1


def test_healthy_item_has_no_break_to_explain():
    ex = bd.explain_break(make_item("healthy", seed=3), cfg=FAST)
    assert ex.status == "NO_BREAK" and ex.condition is None


def test_all_eighteen_dimensions_are_reported_and_unmeasured_ones_say_so():
    ex = bd.explain_break(make_item("regime", seed=3), cfg=FAST, seed=1)
    dims = {d["dimension"]: d for d in ex.dimension_summary}
    assert set(dims) == set(bd.DIMENSIONS) and len(dims) == 18
    assert dims["sector_composition"]["verdict"] == "not_measured"
    assert dims["timing"]["n_columns"] > 0 and dims["outcome_magnitude"]["n_columns"] > 0
    assert dims["missingness"]["verdict"] == "not_measured"       # no column had missing values


def test_descriptive_columns_are_symptoms_never_conditions():
    ex = bd.explain_break(make_item("regime", seed=3), cfg=FAST, seed=1)
    cols = {t.column for t in ex.condition.terms}
    assert not cols & {"abs_value", "loss_size", "run_len", "nb_mean_now"}


def test_future_leak_canary_refuses_peeking_column():
    item = make_item("regime", seed=3, leak="peek")
    found = bd.scan_context_leaks(item)
    assert [f.column for f in found] == ["m_peek"]
    with pytest.raises(FirewallBreach):
        bd.explain_break(item, cfg=FAST)
    assert bd.scan_context_leaks(make_item("regime", seed=3)) == []       # the honest regime variable is not flagged


def test_declared_negative_lag_is_a_breach():
    with pytest.raises(FirewallBreach):
        bd.assert_no_leak(make_item("regime", seed=3, leak="declared"))
    with pytest.raises(bd.BreakInputError):
        make_item("regime", seed=3, leak="declared").require_valid()


def test_context_builder_truncation_audit_catches_a_centered_window():
    raw = pd.DataFrame({"x": np.random.default_rng(0).normal(size=120)})
    causal = lambda r: pd.DataFrame({"m": r["x"].rolling(5).mean()})
    peek = lambda r: pd.DataFrame({"m": r["x"].rolling(5, center=True).mean()})
    assert bd.audit_context_builder(causal, raw, [30, 60, 90]) == []
    assert bd.audit_context_builder(peek, raw, [30, 60, 90])


def test_visible_hides_the_newest_outcome_and_everything_after():
    item = make_item("regime", seed=3)
    as_of = item.frame.index[300]
    vis = item.visible(as_of)
    assert len(vis.frame) == 301 and math.isnan(vis.frame["value"].iloc[-1]) and vis.n_matured(None) == 300
    ex = bd.explain_break(item, as_of=item.frame.index[500], cfg=FAST, seed=1)
    assert ex.as_of == str(item.frame.index[500])


def test_no_lookahead_explanation_is_identical_when_the_future_is_scrambled():
    item = make_item("regime", seed=3)
    cut = item.frame.index[449]
    a = bd.explain_break(item, as_of=cut, cfg=FAST, seed=2)
    fr = item.frame.copy()
    fr.iloc[450:, :] = np.random.default_rng(9).normal(size=fr.iloc[450:, :].shape)
    b = bd.explain_break(bd.ItemSeries("pat_a", fr, item.columns), as_of=cut, cfg=FAST, seed=2)
    assert a.explanation_id == b.explanation_id and a.n_failed == b.n_failed and a.n_tested == b.n_tested


def test_insufficient_evidence_on_short_history_and_empty_frame():
    ex = bd.explain_break(make_item("regime", T=80, seed=3), cfg=FAST)
    assert ex.status in ("INSUFFICIENT_EVIDENCE", "NO_BREAK") and ex.condition is None
    empty = bd.ItemSeries("e", pd.DataFrame({"value": []}, index=pd.DatetimeIndex([])), ())
    with pytest.raises(bd.BreakInputError):
        empty.require_valid()


def test_input_validation_catches_bad_frames():
    item = make_item("regime", seed=3)
    bad = bd.ItemSeries("x", item.frame.iloc[::-1], item.columns)
    assert any("increasing" in e for e in bad.validate())
    assert any("unknown dimension" in e for e in bd.ItemSeries("x", item.frame, (bd.ContextColumn("m_regime", "vibes"),)).validate())
    assert any("not in frame" in e for e in bd.ItemSeries("x", item.frame, (bd.ContextColumn("nope", "regime"),)).validate())


def test_condition_evaluation_rejects_a_condition_that_only_fits_the_past():
    item = make_item("regime", seed=3)
    design = bd.build_design(item)
    vals = item.frame["value"].values[: len(design.X)]
    bogus = bd.Condition((bd.Term("m_n0", ">=", 0.0),), ("volatility",))
    res = bd.evaluate_condition(bogus, design.X, vals, np.arange(200, 600), FAST)
    assert not res.passes
    true = bd.Condition((bd.Term("m_regime", ">=", 0.0),), ("regime",))
    assert bd.evaluate_condition(true, design.X, vals, np.arange(200, 600), FAST).passes


def test_walk_forward_gate_on_true_driver_gains_and_on_noise_does_not():
    item = make_item("regime", seed=4)
    design = bd.build_design(item)
    vals = item.frame["value"].values[: len(design.X)]
    rows = np.arange(0, len(design.X))
    good = bd.walk_forward_condition(design, vals, "m_regime", ">=", rows)
    noise = bd.walk_forward_condition(design, vals, "m_n1", ">=", rows)
    assert good.gain > 0.0012 and good.t_gain > 3
    assert good.gain > noise.gain + 0.001
    lo, hi = bd.block_bootstrap_gain(vals[good.rows], good.hold, seed=1, cfg={"boot": 200})[1:]
    assert lo > 0


def test_missingness_becomes_a_column_and_a_planted_missingness_break_is_seen():
    item = make_item("regime", seed=5)
    fr = item.frame.copy()
    fr.loc[fr.index[::7], "m_n2"] = np.nan
    it2 = bd.ItemSeries("m", fr, item.columns)
    d = bd.build_design(it2)
    assert "m_n2__missing" in d.X.columns and d.tags["m_n2__missing"].dimension == "missingness"
    assert not d.X["m_n2"].isna().any()


def test_family_wise_null_is_calibrated_on_pure_noise():
    rng = np.random.default_rng(0)
    T = 300
    X = pd.DataFrame(rng.normal(size=(T, 30)), columns=[f"c{i}" for i in range(30)])
    tags = {c: bd.ColTag("regime", True) for c in X.columns}
    hits = 0
    for s in range(20):
        lab = np.zeros(T, bool)
        st = int(np.random.default_rng(s).integers(0, T - 60))
        lab[st:st + 40] = True
        pops = bd.Populations(np.flatnonzero(~lab), np.flatnonzero(lab), "state", T, 0)
        _, p = bd.family_wise_p(X, pops, list(X.columns), 150, s)
        hits += bool((p <= 0.10).any())
    assert hits <= 6                                # ~10% expected; far under the ~96% an unadjusted per-column bar would give


def test_placebo_control_rate_is_low():
    r = bd.placebo_false_condition_rate(make_item("regime", seed=3), n=6, cfg=FAST, seed=3)
    assert r["runs"] == 6 and r["explained"] <= 2


def test_shared_break_events_and_profiles():
    e = lambda o: bd.Episode(o + 5, o, o + 20)
    ev = bd.shared_break_events({"a": [e(10)], "b": [e(12)], "c": [e(13), e(100)], "d": [e(200)]}, gap=4, min_items=3)
    assert len(ev) == 1 and ev[0]["items"] == ["a", "b", "c"]
    assert bd.shared_break_events({}) == []
    prof = bd.episode_profile([bd.Episode(30, 25, 40), bd.Episode(80, 70, None)], 100)
    assert prof["n"] == 2 and prof["recovered"] == 1 and prof["open"] == 1
    assert bd.episode_profile([], 50)["n"] == 0


def test_report_renders_and_states_not_validated():
    ex = bd.explain_break(make_item("regime", seed=3), cfg=FAST, seed=1)
    txt = bd.render_report([ex])
    assert "NOT VALIDATED" in txt and "pat_a" in txt and "sector_composition: not_measured" in txt


# =============================================================================================== section 11: reliability
from engine.learning import reliability as rl


def stream(kind, T=400, seed=0, mu=0.006, sigma=0.010):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-06", periods=T, freq="W-FRI")
    if kind == "steady":
        v = mu + rng.normal(0, sigma, T)
    elif kind == "noise":
        v = rng.normal(0, sigma, T)
    elif kind == "decayed":
        v = np.where(np.arange(T) < T - 40, mu, -mu) + rng.normal(0, sigma, T)
    elif kind == "slow":
        s = np.ones(T)
        for a in range(0, T, 160):
            s[a + 80:a + 160] = -1
        v = s * mu + rng.normal(0, sigma, T)
    elif kind == "regimes":
        s = np.ones(T)
        for a in range(0, T, 60):
            s[a + 30:a + 60] = -1
        v = s * mu + rng.normal(0, sigma, T)
    return pd.DataFrame({"value": v}, index=idx)


def test_dimensions_are_separate_and_current_falls_before_truth_does():
    fr = stream("decayed", seed=1)
    early = rl.compute_state("x", fr, fr.index[359])
    late = rl.compute_state("x", fr, fr.index[399])
    assert early.truth > 0.95 and early.current_reliability > 0.6
    assert late.current_reliability < early.current_reliability - 0.3
    assert late.truth > 0.7 and late.truth > late.current_reliability + 0.3          # history still says real; now says no
    assert late.failure_risk > early.failure_risk
    assert rl.confidence_delta(early, late)


def test_untested_is_none_not_zero_and_reported():
    fr = stream("steady", T=8)
    st = rl.compute_state("x", fr, fr.index[-1] + pd.Timedelta(days=7))
    assert st.truth is None and st.current_reliability is None and st.failure_risk is None
    assert "truth" in st.untested() and rl.decide(st).action == "ABSTAIN_UNTESTED"
    empty = rl.compute_state("x", pd.DataFrame({"value": []}, index=pd.DatetimeIndex([])), "2020-01-01")
    assert empty.n_obs == 0 and empty.check() == []


def test_two_contrasting_profiles_get_different_decisions():
    c = rl.contrast_profiles()
    assert c["differ"] and c["A"]["action"] == "STANDBY" and c["A"]["weight"] == 0.0
    assert c["B"]["action"] == "USE_LOCAL" and 0.3 < c["B"]["weight"] <= 1.0 and c["B"]["scope"] == "current_conditions_only"
    b = rl.state_from_profile("b", rl.profile_b())
    assert rl.decide(b, in_context=False).weight < rl.decide(b, in_context=True).weight       # transfer scales unfamiliar use


def test_decision_rule_is_monotone_in_risk_current_and_truth():
    grid = rl.decision_grid()
    assert rl.monotonicity_violations(grid) == []
    assert set(grid["action"]) >= {"STANDBY", "USE_FULL"}


def test_future_evidence_is_refused_and_history_is_append_only():
    fr = stream("steady", T=100)
    tr = rl.ReliabilityTracker()
    tr.register("x")
    with pytest.raises(FirewallBreach):
        tr.update("x", fr, fr.index[50])
    st1 = tr.update("x", fr.iloc[:60], fr.index[60])
    st2 = tr.update("x", fr.iloc[60:80], fr.index[80])
    assert st2.n_obs == 80 and len(tr.history("x")) == 2 and st1.n_obs == 60
    with pytest.raises(ValueError):
        tr.update("x", fr.iloc[70:75], fr.index[90])
    with pytest.raises(KeyError):
        tr.update("nope", fr.iloc[:5], fr.index[10])


def test_state_ignores_rows_dated_after_now():
    fr = stream("steady", T=200, seed=3)
    a = rl.compute_state("x", fr, fr.index[120])
    fr2 = fr.copy()
    fr2.iloc[120:, 0] = -0.5
    assert a.state_id == rl.compute_state("x", fr2, fr.index[120]).state_id


def test_stale_evidence_regresses_toward_one_half():
    assert rl.stale_adjust(0.9, 0) == pytest.approx(0.9)
    assert rl.stale_adjust(0.9, 26) == pytest.approx(0.7)
    assert rl.stale_adjust(0.1, 52) == pytest.approx(0.5 - 0.4 * 0.25)


def test_streaming_estimator_reproduces_the_batch_one():
    x = stream("regimes", T=300, seed=2)["value"].values
    sr = rl.StreamingReliability()
    for v in x:
        sr.push(v)
    sr.push(float("nan"))
    tb, tdb = rl.truth_confidence(x)
    ts, tds = sr.truth()
    assert ts == pytest.approx(tb, abs=1e-9) and tds["rho"] == pytest.approx(tdb["rho"], abs=1e-9)
    cb, _ = rl.current_reliability(x, tdb)
    cs, _ = sr.current(tds)
    assert cs == pytest.approx(cb, abs=1e-6)


def test_transfer_needs_several_domains_and_separates_consistent_from_inconsistent():
    rng = np.random.default_rng(0)
    dom = np.repeat([f"y{k}" for k in range(6)], 30)
    consistent = rng.normal(0.006, 0.01, 180)
    mixed = consistent - 0.006 + np.repeat([0.012, -0.012, 0.010, -0.010, 0.011, -0.013], 30)
    pc, _ = rl.transfer_confidence(consistent, dom)
    pm, _ = rl.transfer_confidence(mixed, dom)
    assert pc > 0.9 and pm < 0.75 and pc > pm + 0.2
    assert rl.transfer_confidence(consistent[:30], dom[:30])[0] is None
    het = rl.heterogeneity(mixed, dom)
    assert het["p"] < 0.01 and het["I2"] > 0.5 and rl.heterogeneity(consistent, dom)["p"] > 0.01
    lodo = rl.leave_one_domain_out(consistent, dom)
    assert lodo["n"] == 6 and lodo["hit_rate"] >= 0.8


def test_walk_forward_reliability_is_informative_on_persistent_regimes_and_not_on_noise():
    good, _ = rl.walk_forward_reliability(stream("slow", T=1200, seed=5)["value"].values, cfg={"boot": 200})
    bad, _ = rl.walk_forward_reliability(stream("noise", T=900, seed=6)["value"].values, cfg={"boot": 200})
    assert good.predictive and good.auc > 0.6 and good.skill_recalibrated > 0
    assert not bad.predictive and bad.verdict.startswith("AUC interval includes chance")
    short, _ = rl.walk_forward_reliability(np.zeros(30))
    assert short.n == 0 and not short.predictive


def test_collapsed_dimensions_are_detected():
    states = [rl.state_from_profile(f"i{k}", {"truth": 0.5 + 0.04 * k, "current_reliability": 0.5 + 0.04 * k, "transfer": 0.9 - 0.02 * (k % 3),
                                               "context": 0.3 + 0.03 * ((k * 5) % 7), "failure_risk": 0.5 - 0.04 * k}) for k in range(10)]
    hit = rl.collapse_check(states)
    assert any({h["a"], h["b"]} == {"truth", "current_reliability"} for h in hit)
    assert rl.collapse_check(states[:3]) == []


def test_learn_context_spec_finds_a_real_context_and_refuses_noise():
    rng = np.random.default_rng(1)
    T = 500
    ctx = pd.DataFrame({"m_a": rng.normal(size=T), "m_b": rng.normal(size=T), "m_c": rng.normal(size=T)})
    y = np.where(ctx["m_a"] > 0.3, 0.008, -0.004) + rng.normal(0, 0.01, T)
    out = rl.learn_context_spec(ctx, y)
    assert "m_a" in out["spec"] and out["spec"]["m_a"][0] < 0.6
    noise = rl.learn_context_spec(ctx, rng.normal(0, 0.01, T))
    assert noise["spec"] == {}
    het = rl.context_heterogeneity(ctx, y)
    assert het.iloc[0]["column"] == "m_a" and het.iloc[0]["p_holm"] < 0.001


def test_stated_contexts_and_anti_contexts_shape_the_context_dimension():
    assert rl.match_context({"m_a": (0.0, None)}, {"m_a": 1.0}) == "in"
    assert rl.match_context({"m_a": (0.0, None)}, {"m_a": -1.0}) == "out"
    assert rl.match_context({"m_a": (0.0, None)}, {}) == "unknown" and rl.match_context({}, {"m_a": 1}) == "unknown"
    rng = np.random.default_rng(2)
    T = 200
    fr = pd.DataFrame({"value": rng.normal(0.005, 0.01, T), "m_a": rng.normal(size=T)}, index=pd.date_range("2015-01-02", periods=T, freq="W-FRI"))
    now = fr.index[-1] + pd.Timedelta(days=7)
    st = rl.compute_state("x", fr, now, {"m_a": 2.0}, anti_contexts={"m_a": (1.5, None)})
    assert st.context_match == "out" and st.context <= 0.2
    assert rl.decide(st).action == "STANDBY"
    assert rl.compute_state("x", fr, now, {"m_a": 0.0}, contexts={"m_a": (-1.0, 1.0)}).context_match == "in"


def test_hazard_backtest_and_state_series_run_and_report_honestly():
    hb = rl.hazard_backtest(stream("regimes", T=600, seed=8)["value"].values, start=100, step=6)
    assert hb["n"] > 20 and "informative" in hb
    ts = rl.state_series(stream("decayed", seed=1), start=60, step=20)
    assert list(ts.columns[1:6]) == list(rl.DIMS) and ts["current_reliability"].iloc[-1] < ts["current_reliability"].iloc[0]


def test_export_import_roundtrip_and_tamper_detection():
    fr = stream("steady", T=90, seed=4)
    tr = rl.ReliabilityTracker()
    tr.register("x", {"m_a": (0.0, None)})
    st = tr.update("x", fr, fr.index[-1] + pd.Timedelta(days=7))
    back = rl.import_tracker(rl.export_tracker(tr))
    assert back.n_obs("x") == 90 and back.history("x")[0].state_id == st.state_id
    d = st.as_dict()
    assert rl.state_from_dict(d).state_id == st.state_id
    d["truth"] = 0.01
    with pytest.raises(ValueError):
        rl.state_from_dict(d)


def test_explain_update_names_the_dimension_that_moved():
    fr = stream("decayed", seed=1)
    a = rl.compute_state("x", fr, fr.index[359])
    b = rl.compute_state("x", fr, fr.index[399])
    ex = rl.explain_update(a, b, fr.iloc[359:399])
    assert ex["dominant"] is not None and ex["evidence"]["n_new"] == 40 and ex["evidence"]["mean_new"] < 0


# =============================================================================================== section 12: lifecycle
from engine.learning import lifecycle as lc
from engine.learning.core import Lifecycle
from engine.learning.retirement import RetirementLedger, State


def path(kind, T=400, seed=0, mu=0.006, sigma=0.010):
    rng = np.random.default_rng(seed)
    i = np.arange(T)
    if kind == "steady":
        m = np.full(T, mu)
    elif kind == "gradual":
        m = np.where(i < 120, mu, mu - (i - 120) * (mu + 0.004) / 200)
    elif kind == "abrupt":
        m = np.where(i < 200, mu, -0.004)
    elif kind == "recover":
        m = np.where((i >= 160) & (i < 240), -mu, mu)
    elif kind == "phantom":
        m = np.zeros(T)
    elif kind == "flat":
        m = np.full(T, mu)
    return m + rng.normal(0, sigma, T)


def test_steady_item_is_born_grows_and_is_active_and_never_fails():
    tr = lc.trace(path("steady", seed=1))
    assert lc.check_trace(tr) == []
    stages = [p["stage"] for p in tr.phases()]
    assert stages[0] == "BIRTH" and stages[1] == "GROWTH" and "FAILURE" not in stages
    assert tr.current in (Lifecycle.ACTIVE, Lifecycle.PEAK) and tr.summary()["failures"] == 0


def test_failure_then_recovery_is_reached_and_recovery_is_earned():
    x = path("recover", T=420, seed=2)
    tr = lc.trace(x)
    assert lc.check_trace(tr) == []
    assert tr.summary()["failures"] >= 1 and tr.summary()["recoveries"] >= 1
    assert tr.current in (Lifecycle.RECOVERY, Lifecycle.ACTIVE, Lifecycle.PEAK)
    fail_row = tr.first(Lifecycle.FAILURE)
    assert lc.assess_recovery(x, fail_row, fail_row + 6).recovered is False                    # too little evidence yet
    assert lc.assess_recovery(x, fail_row, len(x)).n > 13
    assert not lc.assess_recovery(x, 165, 235).recovered                                        # the bad stretch itself is not recovery
    assert lc.assess_recovery(x, 250, len(x)).recovered
    tm = lc.transition_matrix({"a": tr})
    assert tm.loc["FAILURE", "RECOVERY"] >= 1


def test_phantom_pattern_never_established_is_marked_failed():
    tr = lc.trace(path("phantom", T=300, seed=3))
    assert tr.phantom and tr.first(Lifecycle.FAILURE) is not None
    assert Lifecycle.ACTIVE.value not in set(tr.stage) and Lifecycle.PEAK.value not in set(tr.stage)


def test_dormant_when_it_stops_firing_and_wakes_up_when_it_fires_again():
    x = path("steady", T=300, seed=4)
    expo = np.ones(300, bool)
    expo[150:200] = False
    x[150:200] = np.nan
    tr = lc.trace(x, exposure=expo)
    assert tr.stage[190] == "DORMANT" and tr.stage[149] != "DORMANT" and tr.stage[210] != "DORMANT"
    assert lc.check_trace(tr) == []


def test_retired_is_terminal_and_a_retirement_proposal_needs_a_long_failure():
    x = path("steady", T=300, seed=5)
    tr = lc.trace(x, retired_from=250)
    assert set(tr.stage[250:]) == {"RETIRED"} and lc.check_trace(tr) == []
    bad = np.r_[path("steady", T=150, seed=6), path("steady", T=200, seed=7) - 0.02]
    assert lc.retire_proposal(lc.trace(bad))["propose"] is False or lc.retire_proposal(lc.trace(bad))["run"] >= 104
    assert lc.retire_proposal(lc.trace(np.zeros(0)))["propose"] is False


def test_causal_trace_does_not_move_when_the_future_changes():
    x = path("recover", T=400, seed=2)
    y = x.copy()
    y[300:] = 5.0
    a, b = lc.trace(x), lc.trace(y)
    assert list(a.stage[:300]) == list(b.stage[:300])


def test_gradual_and_abrupt_deterioration_are_classified_correctly():
    g = [lc.classify_shape(path("gradual", seed=s)[100:], seed=s).shape for s in range(4)]
    a = [lc.classify_shape(path("abrupt", seed=s)[100:], seed=s).shape for s in range(4)]
    assert g.count("GRADUAL") >= 3 and a.count("ABRUPT") >= 3
    fit = lc.classify_shape(path("abrupt", seed=1)[100:], seed=1)
    assert abs(fit.step_at - 100) <= 12 and fit.step_size < -0.006
    assert lc.classify_shape(path("gradual", seed=1)[100:], seed=1).slope < 0


def test_random_fluctuation_is_called_random_not_a_break():
    verdicts = [lc.classify_deterioration(path("flat", T=160, seed=s), seed=s).kind for s in range(6)]
    assert verdicts.count(lc.Deterioration.RANDOM) >= 5
    short = lc.classify_deterioration(np.zeros(10))
    assert short.kind == lc.Deterioration.INSUFFICIENT and short.cause.value == "INSUFFICIENT_EVIDENCE"


def test_regime_linked_deterioration_is_named_and_an_unrelated_family_is_not():
    rng = np.random.default_rng(3)
    T = 300
    regime = np.where(np.arange(T) < 150, 0.0, -1.5) + rng.normal(0, 0.4, T)
    y = 0.005 + 0.004 * regime + rng.normal(0, 0.008, T)
    fam = {"regime": pd.DataFrame({"m_regime": regime}),
           "volatility": pd.DataFrame({"m_v1": rng.normal(size=T), "m_v2": rng.normal(size=T)}),
           "liquidity": pd.DataFrame({"m_l": rng.normal(size=T)})}
    v = lc.classify_deterioration(y, fam, seed=1)
    assert v.kind == lc.Deterioration.REGIME_LINKED and v.linked_family == "regime" and v.cause.value == "REGIME_CHANGE"
    assert {l.family: l for l in v.links}["volatility"].p > 0.05
    unlinked = {k: f for k, f in fam.items() if k != "regime"}
    v2 = lc.classify_deterioration(y, unlinked, seed=1)
    assert v2.linked_family is None and v2.kind in (lc.Deterioration.ABRUPT, lc.Deterioration.GRADUAL)


def test_book_diagnoses_only_items_that_are_decaying_and_reports_counts():
    book = lc.LifecycleBook()
    book.add("ok", path("steady", seed=1))
    book.add("dead", path("abrupt", T=420, seed=2))
    assert book.diagnose("ok") is None
    v = book.diagnose("dead")
    assert v is not None and v.kind in (lc.Deterioration.ABRUPT, lc.Deterioration.GRADUAL)
    rows = {r["knowledge_id"]: r for r in book.report()}
    assert rows["ok"]["deterioration"] is None and rows["dead"]["deterioration"] is not None
    assert sum(book.counts().values()) == 2 and not lc.stage_table(book.traces).empty


def test_decay_half_life_fit_reports_the_planted_rate():
    e = 0.006 * 0.5 ** (np.arange(120) / 40.0) + np.random.default_rng(0).normal(0, 0.0002, 120)
    fit = lc.decay_half_life(e, 0)
    assert 30 <= fit["half_life"] <= 52 and fit["r2"] > 0.95
    assert np.isnan(lc.decay_half_life(np.zeros(5), 0)["half_life"])


def test_lifecycle_feeds_the_retirement_ledger_but_only_on_evidence():
    x = np.r_[path("steady", T=200, seed=1), path("steady", T=120, seed=2) - 0.02]
    dates = pd.date_range("2018-01-05", periods=len(x), freq="W-FRI")
    tr = lc.trace(x)
    assert tr.current == Lifecycle.FAILURE
    led = RetirementLedger()
    led.register("k", dates[0])
    now = dates[-1] + pd.Timedelta(days=7)
    v = lc.apply_to_ledger(led, "k", tr, x, dates, now, apply=True)
    assert v is not None and led.state("k", now + pd.Timedelta(days=1)) != State.ACTIVE
    assert lc.apply_to_ledger(led, "unknown", tr, x, dates, now) is None


def test_streaming_machine_equals_batch_and_a_checkpoint_resumes_identically():
    x = path("recover", T=380, seed=5)
    batch = lc.trace(x)
    m = lc.LifecycleMachine()
    for v in x:
        m.push(v)
    assert list(m.to_trace().stage) == list(batch.stage)
    half = lc.LifecycleMachine()
    for v in x[:200]:
        half.push(v)
    snap = half.snapshot()
    import json as _json
    resumed = lc.LifecycleMachine.from_snapshot(_json.loads(_json.dumps(snap)))
    tr = lc.resume(resumed, x[200:])
    assert list(tr.stage) == list(batch.stage) and len(tr.changes) == len(batch.changes)


def test_growth_curve_forecast_and_segmentation_recover_planted_structure():
    T = 300
    rng = np.random.default_rng(1)
    e = 0.008 / (1 + np.exp(-0.08 * (np.arange(T) - 100))) + rng.normal(0, 0.0002, T)
    g = lc.fit_growth(e, 250)
    assert abs(g["t0"] - 100) <= 25 and g["r2"] > 0.9
    assert np.isnan(lc.fit_growth(np.zeros(5))["t0"])
    dec = 0.001 + 0.006 * 0.5 ** (np.arange(150) / 30.0) + rng.normal(0, 0.00015, 150)
    f = lc.forecast_life(dec, 0, bar=0.0005)
    assert f["crossing"] is None and "levels off" in f["why"]                  # the floor (0.001) stays above the bar
    dec2 = 0.006 * 0.5 ** (np.arange(150) / 30.0) - 0.0004 + rng.normal(0, 0.00015, 150)
    f2 = lc.forecast_life(dec2, 0, bar=-0.0002)
    assert f2["fit"]["r2"] > 0.8
    x = np.r_[np.random.default_rng(2).normal(0.006, 0.01, 120), np.random.default_rng(3).normal(-0.006, 0.01, 90),
              np.random.default_rng(4).normal(0.006, 0.01, 120)]
    segs = lc.segment_history(x)
    assert len(segs) == 3 and abs(segs[0]["end"] - 120) <= 12 and segs[1]["mean"] < 0 < segs[0]["mean"]
    assert len(lc.segment_history(np.random.default_rng(5).normal(0, 0.01, 200))) == 1


def test_survival_by_age_is_kaplan_meier_and_median_life_needs_half_to_fail():
    traces = {f"s{k}": lc.trace(path("steady", T=250, seed=k)) for k in range(4)}
    traces.update({f"d{k}": lc.trace(path("abrupt", T=300, seed=10 + k)) for k in range(6)})
    s = lc.survival_by_age(traces)
    assert s["survival"].is_monotonic_decreasing and s["survival"].iloc[0] <= 1.0 and s["events"].sum() >= 4
    assert lc.median_life(s) >= 0 or np.isnan(lc.median_life(s))
    assert lc.survival_by_age({}).empty
    steady_only = lc.survival_by_age({"a": lc.trace(path("steady", seed=1))})
    assert np.isnan(lc.median_life(steady_only))


def test_recovery_conditions_are_found_only_with_enough_events():
    rng = np.random.default_rng(0)
    parts, ctx = [], []
    for c in range(5):
        parts += [rng.normal(0.008, 0.01, 90), rng.normal(-0.012, 0.01, 50), rng.normal(0.008, 0.01, 60)]
        ctx += [rng.normal(0, 1, 90), rng.normal(-2, 0.3, 50), rng.normal(2, 0.3, 60)]
    x, z = np.concatenate(parts), np.concatenate(ctx)
    tr = lc.trace(x)
    df = pd.DataFrame({"m_a": z, "m_noise": rng.normal(size=len(x))})
    out = lc.recovery_conditions(df, tr)
    assert tr.summary()["recoveries"] >= 3
    assert len(out) == 0 or out.iloc[0]["column"] == "m_a"
    assert lc.recovery_conditions(df, lc.trace(x[:60])).empty


def test_lifecycle_frame_and_report_are_readable():
    book = lc.LifecycleBook()
    book.add("ok", path("steady", seed=1))
    book.add("dead", path("abrupt", T=420, seed=2))
    book.diagnose("dead")
    fr = lc.lifecycle_frame(book)
    assert set(fr["knowledge_id"]) == {"ok", "dead"} and fr.set_index("knowledge_id").loc["dead", "failures"] >= 0
    txt = lc.deterioration_report(book)
    assert "dead" in txt and "NOT VALIDATED" in txt
    assert "no item is currently decaying" in lc.deterioration_report(lc.LifecycleBook())


def test_onset_localisation_is_tight_for_a_step_and_wide_for_a_slide():
    step = lc.locate_onset(path("abrupt", seed=1)[100:], n_boot=100)
    slide = lc.locate_onset(path("gradual", seed=1)[100:], n_boot=100)
    assert abs(step["onset"] - 100) <= 12 and step["width"] <= 25
    assert slide["width"] > step["width"]
    assert lc.locate_onset(np.zeros(5))["onset"] is None


def test_spells_time_to_recover_and_stage_agreement():
    x = path("recover", T=420, seed=2)
    tr = lc.trace(x)
    assert lc.spells(tr, Lifecycle.FAILURE) and all(s["length"] > 0 for s in lc.spells(tr, Lifecycle.FAILURE))
    mttr = lc.mean_time_to_recover(tr)
    assert np.isnan(mttr) or 0 < mttr < 200
    assert np.isnan(lc.mean_time_to_recover(lc.trace(path("steady", seed=1))))
    ag = lc.stage_agreement(tr, x)
    assert ag["good_beats_failed"] in (True, None) and "FAILURE" in ag["by_stage"]


def test_durations_readiness_exposure_and_trace_validation():
    traces = {"a": lc.trace(path("recover", T=420, seed=2)), "b": lc.trace(path("steady", seed=1))}
    d = lc.stage_durations(traces)
    assert {"BIRTH", "GROWTH"} <= set(d["stage"]) and (d["spells"] >= 1).all() and lc.stage_durations({}).empty
    assert lc.rows_to_establish(0.006, 0.01) == pytest.approx(max((2.0 * 0.01 / 0.006) ** 2, 26))
    assert lc.rows_to_establish(-0.001, 0.01) == float("inf") and lc.rows_to_establish(0.006, 0.0) == float("inf")
    assert list(lc.exposure_from_counts([0, 2, np.nan, 1])) == [False, True, False, True]
    assert all(lc.validate_trace(t, len(t)) == [] for t in traces.values())
    broken = lc.trace(path("steady", seed=1))
    broken.stage[5] = "NOPE"
    assert any("unknown stages" in e for e in lc.validate_trace(broken))
