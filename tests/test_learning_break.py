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
