"""Tests for engine.learning.boundary (contract C62 section 41; checklist C12 conditions, C13 anti-conditions).
Synthetic worlds with planted boundaries.  IMPLEMENTED - NOT VALIDATED: proves the mechanism on planted worlds, not on real data."""
import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

from engine.learning import boundary as B
from engine.learning.core import FirewallBreach, KnowledgeLike

NOW = "2030-01-01"
FAST = B.BoundaryConfig(n_perm=99, n_boot=20)


def world(n=420, thr=0.3, inside=0.02, outside=-0.006, seed=1, noise=0.02, flip_after=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    vol, liq, junk = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    edge = np.where(vol < thr, inside, outside) + rng.normal(0, noise, n)
    if flip_after is not None:                            # the relationship vanishes after this position
        edge[flip_after:] = rng.normal(0.005, noise, n - flip_after)
    return pd.DataFrame({"edge": edge, "vol": vol, "liq": liq, "junk": junk}, index=idx)


def learn(df, cols=("vol", "liq", "junk"), cfg=FAST, pid="p"):
    return B.BoundaryLearner(cfg).learn(pid, df, "edge", list(cols), NOW)


# ============================================================================================ discovery
def test_planted_threshold_is_recovered_and_confirmed_out_of_time():
    bs = learn(world())
    assert [b.feature for b in bs.boundaries] == ["vol"]
    b = bs.boundaries[0]
    assert b.works_when == "<=" and abs(b.threshold - 0.3) < 0.35
    assert b.edge_inside > 0.012 and b.edge_outside < 0.005 and b.oot_status == "CONFIRMED"
    assert b.outside == "REVERSES" and b.anti_context == {"op": ">", "value": b.threshold}
    assert bs.contexts["vol"] == {"op": "<=", "value": b.threshold}


def test_pure_noise_features_are_rejected_and_recorded():
    bs = learn(world())
    rej = {b.feature: b for b in bs.rejected}
    assert set(rej) == {"liq", "junk"} and not any(b.accepted for b in rej.values())
    assert all(b.reasons for b in rej.values())           # a rejection says why


def test_homogeneous_pattern_yields_no_boundary_most_of_the_time():
    false_acc = 0
    for seed in range(12):
        df = world(inside=0.01, outside=0.01, seed=100 + seed)
        false_acc += len(learn(df, cfg=B.BoundaryConfig(n_perm=79, n_boot=10)).boundaries)
    assert false_acc <= 2                                 # 36 feature tests at alpha 0.05 after Holm and out-of-time confirmation


def test_upper_side_boundary_and_stops_versus_reverses():
    rng = np.random.default_rng(3)
    n = 420
    idx = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    liq = rng.normal(size=n)
    edge = np.where(liq > 0.0, 0.02, 0.0) + rng.normal(0, 0.02, n)
    bs = B.BoundaryLearner(FAST).learn("p", pd.DataFrame({"edge": edge, "liq": liq}, index=idx), "edge", ["liq"], NOW)
    b = bs.boundaries[0]
    assert b.works_when == ">" and b.outside == "STOPS"
    assert b.anti_context == {"op": "<=", "value": b.threshold}


def test_boundary_where_pattern_merely_weakens_has_no_anti_condition():
    rng = np.random.default_rng(4)
    n = 600
    idx = pd.date_range("2012-01-02", periods=n, freq="W-MON")
    x = rng.normal(size=n)
    edge = np.where(x < 0.0, 0.03, 0.012) + rng.normal(0, 0.012, n)
    bs = B.BoundaryLearner(FAST).learn("p", pd.DataFrame({"edge": edge, "x": x}, index=idx), "edge", ["x"], NOW)
    b = bs.boundaries[0]
    assert b.outside == "WEAKENS" and b.anti_context is None and bs.anti_contexts == {}


def test_a_boundary_that_only_held_in_the_past_fails_out_of_time():
    df = world(n=420, flip_after=290)                     # relationship dies before the last 30% begins
    bs = learn(df)
    assert not any(b.feature == "vol" for b in bs.boundaries)
    vol = [b for b in bs.rejected if b.feature == "vol"]
    assert vol and (vol[0].oot_status in ("FAILED", "UNTESTED") or vol[0].p_adj >= 0.05)


def test_unstable_threshold_is_refused():
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2014-01-06", periods=n, freq="W-MON")
    x = rng.normal(size=n)
    edge = 0.0035 * np.tanh(x) + rng.normal(0, 0.02, n)   # a gentle slope, not a step: no crisp threshold
    bs = B.BoundaryLearner(B.BoundaryConfig(n_perm=79, n_boot=30)).learn("p", pd.DataFrame({"edge": edge, "x": x}, index=idx), "edge", ["x"], NOW)
    for b in bs.boundaries:
        assert b.threshold_sd <= 0.5


def test_learning_is_deterministic():
    df = world()
    a, b = learn(df), learn(df)
    assert a.set_id == b.set_id and B.set_to_dict(a) == B.set_to_dict(b)


def test_boundary_table_and_serialisation_roundtrip():
    bs = learn(world())
    tab = B.boundary_table(bs)
    assert len(tab) == 3 and tab["accepted"].sum() == 1
    back = B.set_from_dict(B.set_to_dict(bs))
    assert back.set_id == bs.set_id and back.contexts == bs.contexts


# ============================================================================================ time and degenerate inputs
def test_rows_dated_at_or_after_now_are_refused():
    df = world()
    with pytest.raises(FirewallBreach):
        B.BoundaryLearner(FAST).learn("p", df, "edge", ["vol"], df.index[-1])
    df2 = df.copy()
    df2["matured_at"] = df2.index + pd.Timedelta(days=400)
    with pytest.raises(FirewallBreach):
        B.BoundaryLearner(FAST).learn("p", df2, "edge", ["vol"], df2.index[-1] + pd.Timedelta(days=30))


def test_short_and_empty_frames_are_insufficient_not_guessed():
    short = learn(world(n=60))
    assert short.boundaries == () and short.skipped and "INSUFFICIENT" in short.skipped[0][1]
    empty = B.BoundaryLearner(FAST).learn("p", pd.DataFrame({"edge": [], "vol": []}, index=pd.DatetimeIndex([])), "edge", ["vol"], NOW)
    assert empty.boundaries == () and empty.n_periods == 0
    assert "no boundary" in empty.summary()


def test_bad_frames_raise_clear_errors():
    df = world()
    with pytest.raises(B.BoundaryError):
        B.BoundaryLearner(FAST).learn("p", df.drop(columns="edge"), "edge", ["vol"], NOW)
    with pytest.raises(B.BoundaryError):
        B.BoundaryLearner(FAST).learn("p", df, "edge", ["nope"], NOW)
    dup = pd.concat([df, df.iloc[:1]])
    with pytest.raises(B.BoundaryError):
        B.BoundaryLearner(FAST).learn("p", dup, "edge", ["vol"], NOW)


def test_missing_and_constant_features_are_skipped_with_a_reason():
    df = world()
    df["const"] = 1.0
    df["holes"] = np.where(np.arange(len(df)) % 3 == 0, df["vol"], np.nan)
    bs = learn(df, cols=("vol", "const", "holes"))
    sk = dict(bs.skipped)
    assert sk["const"] == "constant" and "missing" in sk["holes"]
    assert [b.feature for b in bs.boundaries] == ["vol"]


# ============================================================================================ scope semantics
def test_condition_holds_operators_and_malformed_specs():
    assert B.condition_holds({"op": "<=", "value": 1}, 1) is True
    assert B.condition_holds({"op": ">", "value": 1}, 1) is False
    assert B.condition_holds({"op": "between", "lo": 0, "hi": 2}, 1.5) is True
    assert B.condition_holds({"op": "in", "values": ["a", "b"]}, "c") is False
    assert B.condition_holds({"op": "<", "value": 1}, None) is None
    assert B.condition_holds({"op": "<", "value": 1}, float("nan")) is None
    with pytest.raises(B.BoundaryError):
        B.condition_holds({"value": 1}, 1)
    with pytest.raises(B.BoundaryError):
        B.condition_holds({"op": "~", "value": 1}, 1)


def test_scope_status_precedence_and_unknowns():
    ctx = {"vol": {"op": "<=", "value": 0.5}, "liq": {"op": ">", "value": 0.0}}
    anti = {"since_shock": {"op": "<=", "value": 2}}
    ok = {"vol": 0.1, "liq": 1.0, "since_shock": 10}
    assert B.scope_status(ctx, anti, ok) == B.ScopeStatus.IN_SCOPE
    assert B.scope_status(ctx, anti, {**ok, "since_shock": 1}) == B.ScopeStatus.ANTI_HIT
    assert B.scope_status(ctx, anti, {**ok, "vol": 0.9}) == B.ScopeStatus.OUT_OF_SCOPE
    assert B.scope_status(ctx, anti, {**ok, "since_shock": 1, "vol": 0.9}) == B.ScopeStatus.ANTI_HIT     # known-bad beats out-of-scope
    assert B.scope_status(ctx, anti, {"vol": 0.1}) == B.ScopeStatus.UNKNOWN
    assert B.scope_status(ctx, anti, {"vol": 0.9}) == B.ScopeStatus.OUT_OF_SCOPE       # a definite failure beats a missing feature
    assert B.scope_status({}, {}, {}) == B.ScopeStatus.IN_SCOPE
    assert B.scope_status({"vol": [{"op": ">", "value": 0}, {"op": "<", "value": 1}]}, {}, {"vol": 2}) == B.ScopeStatus.OUT_OF_SCOPE


def test_learned_boundary_applies_to_new_context():
    bs = learn(world())
    thr = bs.boundaries[0].threshold
    assert bs.scope({"vol": thr - 1}) == B.ScopeStatus.IN_SCOPE
    assert bs.scope({"vol": thr + 1}) == B.ScopeStatus.ANTI_HIT
    assert bs.scope({}) == B.ScopeStatus.UNKNOWN


# ============================================================================================ feature builders never look ahead
def test_trailing_features_use_only_the_past():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-05", periods=200, freq="W-MON")
    r = pd.Series(rng.normal(0, 0.02, 200), index=idx)
    r.iloc[120] = 0.30                                     # a shock
    full = B.shock_features(r)
    part = B.shock_features(r.iloc[:110])
    pd.testing.assert_frame_equal(full.iloc[:110], part)   # truncating the future changes nothing before it
    assert full["shock"].iloc[120] == 1.0 and full["since_shock"].iloc[123] == 3
    e = pd.Series(rng.normal(0.01, 0.02, 200), index=idx)
    cf, cp = B.confidence_deterioration(e), B.confidence_deterioration(e.iloc[:150])
    pd.testing.assert_frame_equal(cf.iloc[:150], cp)
    e2 = e.copy()
    e2.iloc[100] = 5.0
    assert B.confidence_deterioration(e2)["own_t"].iloc[100] == cf["own_t"].iloc[100]      # row t does not see its own outcome
    lab = pd.Series(["a"] * 60 + ["b"] * 60 + ["a"] * 80, index=idx)
    rt = B.regime_transition_features(lab)
    assert rt["since_regime_change"].iloc[60] == 0 and rt["since_regime_change"].iloc[65] == 5
    pd.testing.assert_frame_equal(rt.iloc[:90], B.regime_transition_features(lab.iloc[:90]))


def test_disagreement_and_concentration_features():
    sc = pd.DataFrame({"a": [1, 1, 0], "b": [1, -1, 0], "c": [1, 1, 0]}, index=pd.date_range("2020-01-06", periods=3, freq="W-MON"))
    d = B.disagreement_from_scores(sc)
    assert d.iloc[0] == 0.0 and d.iloc[1] == pytest.approx(1 - 1 / 3) and np.isnan(d.iloc[2])
    picks = pd.DataFrame({"date": ["2020-01-06"] * 4 + ["2020-01-13"] * 4, "sector": ["x", "x", "x", "x", "x", "y", "z", "w"]})
    h = B.concentration_hhi(picks)
    assert h["2020-01-06"] == 1.0 and h["2020-01-13"] == pytest.approx(0.25)
    assert B.concentration_hhi(pd.DataFrame(columns=["date", "sector"])).empty


# ============================================================================================ event-time boundaries
def test_shock_aftermath_learns_how_long_the_pattern_stays_off():
    rng = np.random.default_rng(9)
    n = 500
    idx = pd.date_range("2010-01-04", periods=n, freq="W-MON")
    mkt = pd.Series(rng.normal(0, 0.015, n), index=idx)
    shocks = [80, 160, 240, 320, 400, 460]
    for s in shocks:
        mkt.iloc[s] = 0.20
    edge = pd.Series(rng.normal(0.012, 0.01, n), index=idx)
    for s in shocks:
        edge.iloc[s + 1:s + 4] = rng.normal(-0.03, 0.01, 3)      # off for exactly 3 periods after each shock
    prof = B.shock_aftermath(edge, mkt, max_lag=8)
    assert B.off_duration(prof) >= 2
    assert prof.attrs["calm_mean"] > 0.008
    calm = pd.Series(rng.normal(0.012, 0.01, n), index=idx)      # control: no effect planted
    assert B.off_duration(B.shock_aftermath(calm, mkt, max_lag=8)) == 0


def test_regime_transition_window_is_learned():
    rng = np.random.default_rng(10)
    n = 480
    idx = pd.date_range("2010-01-04", periods=n, freq="W-MON")
    lab = pd.Series(np.repeat(["up", "down"] * 8, 30)[:n], index=idx)
    edge = pd.Series(rng.normal(0.012, 0.01, n), index=idx)
    since = B.regime_transition_features(lab)["since_regime_change"]
    edge[since.values <= 2] = rng.normal(-0.02, 0.01, int((since.values <= 2).sum()))
    prof = B.transition_effect(edge, lab, max_lag=6)
    assert B.off_duration(prof) >= 2


def test_frame_builder_covers_the_seven_families_and_finds_the_planted_one():
    rng = np.random.default_rng(11)
    n = 420
    idx = pd.date_range("2013-01-07", periods=n, freq="W-MON")
    vol = pd.Series(rng.normal(size=n), index=idx)
    edge = pd.Series(np.where(vol < 0.2, 0.02, -0.006) + rng.normal(0, 0.02, n), index=idx)
    scores = pd.DataFrame(rng.choice([-1, 0, 1], size=(n, 4)), index=idx)
    picks = pd.DataFrame({"date": np.repeat(idx, 5), "sector": rng.choice(list("abcde"), n * 5)})
    fr, kinds = B.build_boundary_frame(edge, volatility=vol, liquidity=pd.Series(rng.normal(size=n), index=idx),
                                       breadth=pd.Series(rng.normal(size=n), index=idx),
                                       market_ret=pd.Series(rng.normal(0, 0.02, n), index=idx),
                                       regime=pd.Series(rng.choice(["a", "b"], n), index=idx), pattern_scores=scores, picks=picks)
    assert {k.value for k in kinds.values()} == {"THRESHOLD", "SHOCK", "TRANSITION", "DISAGREEMENT", "CONCENTRATION", "CONFIDENCE"}
    cols = [c for c in fr.columns if c != "edge"]
    thin = B.BoundaryLearner(FAST).learn("p", fr, "edge", cols, NOW, kinds)
    assert any(f == "__config__" for f, _ in thin.skipped)           # too few permutations for 9 features: said out loud
    bs = B.BoundaryLearner(B.BoundaryConfig(n_perm=399, n_boot=20)).learn("p", fr, "edge", cols, NOW, kinds)
    assert "volatility" in [b.feature for b in bs.boundaries]
    bs2 = B.learn_pattern_boundaries("p", edge, NOW, B.BoundaryLearner(B.BoundaryConfig(n_perm=199, n_boot=20)), volatility=vol)
    assert bs2.boundaries and bs2.boundaries[0].feature == "volatility"


# ============================================================================================ using and maintaining boundaries
def test_gating_on_learned_boundaries_helps_on_newer_data():
    train = world(n=420, seed=1)
    bs = learn(train)
    later = world(n=300, seed=77)
    later.index = pd.date_range("2023-01-02", periods=300, freq="W-MON")
    g = B.gating_benefit(bs, later, "edge")
    assert g["lift"] > 0.004 and g["edge_kept"] > g["edge_skipped"] and 0.4 < g["coverage"] < 0.9
    assert g["t_kept_vs_skipped"] > 4
    assert B.gating_benefit(bs, later.iloc[:0], "edge")["n"] == 0


def test_rows_missing_a_boundary_feature_are_not_in_scope():
    bs = learn(world())
    df = world(n=50, seed=3)
    df.loc[df.index[:5], "vol"] = np.nan
    m = B.in_scope_mask(bs, df)
    assert not m[:5].any() and m[5:].any()


def test_boundary_health_detects_a_boundary_that_decays():
    bs = B.BoundaryLearner(FAST).learn("p", world(), "edge", ["vol", "liq", "junk"], "2022-06-01")
    b = bs.boundaries[0]
    later_ok = world(n=200, seed=21)
    later_ok.index = pd.date_range("2022-07-04", periods=200, freq="W-MON")
    rng = np.random.default_rng(5)
    later_bad = later_ok.copy()
    later_bad["edge"] = rng.normal(0.005, 0.02, 200)               # the region no longer matters
    assert B.boundary_health(b, later_ok, "edge", NOW)["status"] == "HOLDS"
    assert B.boundary_health(b, later_bad, "edge", NOW)["status"] in ("DEGRADED", "REVERSED")
    assert B.boundary_health(b, later_ok.iloc[:10], "edge", NOW)["status"] == "UNTESTED"


def test_registry_is_append_only_and_reports_boundary_drift():
    reg = B.BoundaryRegistry()
    a = learn(world(seed=1))
    later = dc.replace(learn(world(seed=2, thr=0.8)), learned_at="2031-01-01")
    assert reg.add(a) == 1 and reg.add(later) == 2
    with pytest.raises(B.BoundaryError):
        reg.add(dc.replace(a, learned_at="2020-01-01"))
    assert reg.latest("p", as_of="2030-06-01").learned_at == NOW and reg.latest("p").learned_at == "2031-01-01"
    d = reg.drift("p")
    assert len(d) == 1 and "vol" in d[0]["moved"] and reg.patterns() == ["p"] and reg.latest("nobody") is None


def test_boundary_is_storable_as_knowledge():
    bs = learn(world())
    k = B.to_knowledge(bs)
    assert KnowledgeLike.conforms(k) == []
    assert k.epistemic.value == "CONDITIONAL" and k.contexts == bs.contexts and k.anti_contexts == bs.anti_contexts
    assert k.provenance.check() == [] and k.confidence.check() == []
    none = B.to_knowledge(learn(world(inside=0.01, outside=0.01, seed=8), cols=("junk",)))
    assert none.epistemic.value == "UNKNOWN" and none.decision_effect[0].value == "NONE"


def test_conjunction_screen_and_cross_pattern_summary():
    rng = np.random.default_rng(12)
    n = 500
    idx = pd.date_range("2012-01-02", periods=n, freq="W-MON")
    v, l = rng.normal(size=n), rng.normal(size=n)
    edge = np.where((v < 0.0) & (l > 0.0), 0.04, -0.004) + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"edge": edge, "v": v, "l": l}, index=idx)
    bs = B.BoundaryLearner(FAST).learn("p", df, "edge", ["v", "l"], NOW)
    assert {b.feature for b in bs.boundaries} == {"v", "l"}
    r = B.refine_conjunction(df, "edge", *bs.boundaries)
    assert r["worthwhile"] and r["n_both"] > 50 and r["both"] > max(r["b1"], r["b2"])
    summ = B.anti_condition_summary([bs, learn(world(seed=5), pid="q"), learn(world(seed=6), pid="r")])
    assert summ.iloc[0]["feature"] == "vol" and summ.iloc[0]["n_patterns"] == 2
    assert B.anti_condition_summary([]).empty


# ============================================================================================ later additions
def test_step_is_told_from_slope_and_flat():
    rng = np.random.default_rng(13)
    n = 500
    idx = pd.date_range("2012-01-02", periods=n, freq="W-MON")
    x = rng.normal(size=n)
    step = pd.DataFrame({"edge": np.where(x < 0.2, 0.02, -0.01) + rng.normal(0, 0.02, n), "x": x}, index=idx)
    slope = pd.DataFrame({"edge": -0.01 * x + rng.normal(0, 0.02, n), "x": x}, index=idx)
    flat = pd.DataFrame({"edge": rng.normal(0, 0.02, n), "x": x}, index=idx)
    assert B.step_vs_slope(step, "edge", "x")["shape"] == "STEP"
    assert B.step_vs_slope(slope, "edge", "x")["shape"] == "SLOPE"
    assert B.step_vs_slope(flat, "edge", "x")["shape"] == "FLAT"
    assert "INSUFFICIENT" in B.step_vs_slope(step.iloc[:30], "edge", "x")["shape"]
    prof = B.feature_profile(step, "edge", "x")
    assert len(prof) == 8 and prof["mean"].iloc[0] > 0.01 > prof["mean"].iloc[-1] + 0.01
    assert B.feature_profile(step.iloc[:10], "edge", "x").empty


def test_whole_pipeline_has_power_and_a_controlled_false_acceptance_rate():
    r = B.detection_power(sims=12, seed=1)
    assert r["power"] >= 0.9 and r["false_acceptance"] <= 0.17
    weak = B.detection_power(n=120, inside=0.004, outside=0.0, sims=8, seed=2)
    assert weak["power"] < r["power"]                      # less data and a smaller step: the pipeline admits it cannot see it


def test_a_boundary_that_depends_on_the_tuning_constant_is_reported_unstable():
    df = world()
    st = B.stability_across_min_side("p", df, "edge", ["vol", "liq", "junk"], NOW)
    assert st["vol"]["stable"] and st["vol"]["threshold_range"] < 0.6 and "junk" not in st


def test_bad_configuration_is_refused():
    for bad in (B.BoundaryConfig(min_side=2), B.BoundaryConfig(n_perm=5), B.BoundaryConfig(holdout_frac=0.9)):
        with pytest.raises(B.BoundaryError):
            B.BoundaryLearner(bad)
    assert B.validate_config(B.BoundaryConfig()) == []
