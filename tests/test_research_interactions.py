"""Tests for engine.research.interactions (C66 section 26). Synthetic data only; planted effects must be found, noise must not."""
import dataclasses as dc
import math

import numpy as np
import pandas as pd
import pytest

import engine.research.interactions as I
from engine.learning.core import FailureCause, FirewallBreach
from engine.research.core import MaturedRecord, Namespace, Stage

CFG = I.SearchConfig()


@pytest.fixture(scope="module")
def planted():
    inp, now = I.synthetic_panel(seed=1)
    st = I.InteractionState(7)
    return inp, now, st, I.step(st, inp, now, CFG)


@pytest.fixture(scope="module")
def null():
    inp, now = I.synthetic_panel(seed=101, product=0.0, regime=0.0)
    st = I.InteractionState(7)
    return inp, now, st, I.step(st, inp, now, CFG)


@pytest.fixture(scope="module")
def prepared():
    inp, now = I.synthetic_panel(seed=5, n_dates=400)
    return I.prepare(inp, now, CFG), inp, now


# ------------------------------------------------------------------------- planted / null
def test_planted_product_is_validated(planted):
    rep = planted[3]
    ids = {(f.trial.a, f.trial.b, str(f.trial.form)) for f in rep.validated()}
    assert ("volume_ratio", "atr_pct", "PRODUCT") in ids
    assert rep.audit() == []
    assert all(f.epistemic.value == "SUPPORTED" for f in rep.validated())


def test_planted_regime_interaction_not_lost(planted):
    hits = [f for f in planted[3].findings if f.trial.a == "pat_a" and f.trial.b == "regime=high"]
    assert hits and hits[0].fate not in (I.Fate.NOISE, I.Fate.INTERESTING_ONLY)


def test_null_search_validates_nothing(null):
    rep = null[3]
    assert rep.validated() == []
    assert rep.audit() == []
    assert rep.lottery["verdict"] in ("LOTTERY", "AMBIGUOUS")


def test_null_raw_hits_are_explained_by_chance(null):
    rep = null[3]
    assert rep.lottery["raw_hits"] <= rep.lottery["expected_raw_hits"] + 3 * math.sqrt(rep.n_trials)
    assert all(f.fate in (I.Fate.NOISE, I.Fate.INTERESTING_ONLY) for f in rep.findings)


def test_null_across_many_seeds_has_no_false_validation():
    cfg = dc.replace(CFG, n_shuffles=20)
    for seed in range(200, 206):
        inp, now = I.synthetic_panel(seed=seed, n_dates=520, product=0.0, regime=0.0)
        rep = I.step(I.InteractionState(seed), inp, now, cfg)
        assert rep.validated() == [], seed


def test_self_test_passes():
    out = I.self_test()
    assert out["ok"] and out["planted_found"] and out["null_validated"] == 0


def test_pure_curvature_is_not_an_interaction():
    inp, now = I.synthetic_panel(seed=9, product=0.0, regime=0.0, curvature=0.05)
    rep = I.step(I.InteractionState(7), inp, now, CFG)
    bad = [f for f in rep.validated() if f.trial.a == "volume_ratio" and f.trial.b == "atr_pct"]
    assert bad == []


def test_single_year_effect_is_not_validated():
    inp, now = I.synthetic_panel(seed=4, product=0.12, regime=0.0, single_year=True, n_dates=780)
    rep = I.step(I.InteractionState(7), inp, now, CFG)
    hit = [f for f in rep.findings if f.trial.a == "volume_ratio" and f.trial.b == "atr_pct" and f.trial.form is I.Form.PRODUCT]
    assert hit and hit[0].fate is not I.Fate.VALIDATED


def test_label_leak_fails_closed():
    inp, now = I.synthetic_panel(seed=6, leak=True)
    assert "gap_leak" in inp.X.columns
    with pytest.raises(FirewallBreach):
        I.step(I.InteractionState(7), inp, now, CFG)


# ------------------------------------------------------------------------- multiple-testing counting
def test_ledger_counts_every_combination_and_accumulates():
    inp, now = I.synthetic_panel(seed=12, n_dates=420, product=0.0, regime=0.0)
    cfg = dc.replace(CFG, n_shuffles=12)
    st = I.InteractionState(7)
    r1 = I.step(st, inp, now, cfg)
    r2 = I.step(st, inp, now, cfg)
    assert r1.m_total == r1.n_trials
    assert r2.m_total == r1.m_total                   # same combinations: not double counted as new ones ...
    assert st.ledger.looks(r1.data_key) == 2 * r1.n_trials   # ... but every look is recorded
    assert st.ledger.repeats(r1.data_key) == r1.n_trials
    assert any("tried before" in n for n in r2.notes)


def test_corrected_q_pads_untested_trials():
    p = [0.001, 0.2, 0.5]
    small = I.corrected_q(p, 3)
    big = I.corrected_q(p, 3000)
    assert big[0] > small[0] * 100
    assert big[0] == pytest.approx(min(1.0, 0.001 * 3000))


def test_corrected_q_all_methods_ordered():
    p = np.array([0.0005, 0.004, 0.02, 0.3, 0.7])
    bh, by, bo = (I.corrected_q(p, 50, m) for m in ("bh", "by", "bonferroni"))
    assert np.all(bh <= by + 1e-12) and np.all(bh <= bo + 1e-12)
    with pytest.raises(I.InteractionError):
        I.corrected_q(p, 5, "sidak")


def test_detectable_t_grows_with_combinations():
    assert I.detectable_t(10) < I.detectable_t(1000) < I.detectable_t(1_000_000)
    assert I.detectable_t(1) == pytest.approx(1.96, abs=0.01)


def test_more_combinations_lower_the_verdict():
    """The same p-value that survives among 5 trials must not survive among 5,000 (the counting rule)."""
    assert I.corrected_q([0.004], 5)[0] < 0.05 < I.corrected_q([0.004], 5000)[0]


def test_shuffled_controls_measure_pipeline_false_positive_rate(prepared):
    P, _, _ = prepared
    win = I.split_windows(P.all_dates.size, CFG)
    Pd = P.window(*win.disc)
    trials, _ = I.enumerate_trials(P, CFG)
    ctl = I.run_shuffled_controls(Pd, trials, dc.replace(CFG, n_shuffles=15), I.SeedLedger(1), 0, len(trials))
    assert ctl.null_t.shape == (15, len(trials))
    assert 0.0 <= ctl.pipeline_fp_rate <= 0.34
    assert ctl.adj_p(0.0) > 0.9 and ctl.adj_p(float("nan")) == 1.0
    assert ctl.adj_p(50.0) == pytest.approx(1 / 16)


def test_lottery_diagnostic_verdicts():
    rng = np.random.default_rng(0)
    assert I.lottery_diagnostic(rng.uniform(size=200), 200, 0)["verdict"] == "LOTTERY"
    strong = np.concatenate([np.full(30, 1e-9), rng.uniform(size=170)])
    assert I.lottery_diagnostic(strong, 200, 5)["verdict"] == "SIGNAL"
    assert I.lottery_diagnostic([], 0, 0)["verdict"] == "EMPTY"


# ------------------------------------------------------------------------- windows, purge, time
def test_windows_are_purged_and_ordered():
    w = I.split_windows(600, CFG)
    assert w.check(CFG.horizon) == []
    assert w.valid[0] - w.disc[1] >= CFG.horizon and w.hold[0] - w.valid[1] >= CFG.horizon


def test_prepare_refuses_data_at_or_after_now():
    inp, now = I.synthetic_panel(seed=3, n_dates=200)
    with pytest.raises(FirewallBreach):
        I.prepare(inp, inp.X.index.get_level_values(0).max(), CFG)
    with pytest.raises(FirewallBreach):
        I.prepare(inp, pd.Timestamp("2015-03-01"), CFG)


def test_prepare_drops_unmatured_labels(prepared):
    P, inp, _ = prepared
    assert P.dropped_unmatured == CFG.horizon * inp.X.index.get_level_values(1).nunique()
    assert P.all_dates.size == inp.X.index.get_level_values(0).nunique() - CFG.horizon


def test_prepare_rejects_bad_inputs():
    inp, now = I.synthetic_panel(seed=3, n_dates=150)
    with pytest.raises(I.InteractionError):
        I.prepare(dc.replace(inp, X=inp.X.reset_index()), now, CFG)
    with pytest.raises(I.InteractionError):
        I.prepare(dc.replace(inp, horizon=3), now, CFG)
    dup = pd.concat([inp.X, inp.X.iloc[:5]])
    with pytest.raises(I.InteractionError):
        I.prepare(dc.replace(inp, X=dup), now, CFG)
    with pytest.raises(I.InteractionError):
        I.prepare(dc.replace(inp, roles={"volume": ("nope",)}), now, CFG)


def test_config_validation_lists_every_problem():
    bad = dc.replace(CFG, horizon=0, correction="x", n_shuffles=3, frac_discovery=0.9)
    errs = bad.validate()
    assert len(errs) >= 4
    with pytest.raises(I.InteractionError):
        I.prepare(I.synthetic_panel(n_dates=100)[0], pd.Timestamp("2030-01-01"), bad)


def test_empty_and_short_data_return_explained_empty_reports():
    inp, now = I.synthetic_panel(seed=3, n_dates=60)
    st = I.InteractionState(7)
    rep = I.step(st, inp, now, CFG)
    assert rep.n_trials == 0 and rep.validated() == [] and rep.notes and rep.lottery["verdict"] == "EMPTY"
    empty = dc.replace(inp, X=inp.X.iloc[0:0], y=inp.y.iloc[0:0])
    assert I.step(st, empty, now, CFG).notes
    assert st.ledger.runs == 2


def test_no_families_available_is_empty_report():
    inp, now = I.synthetic_panel(seed=3, n_dates=300)
    rep = I.step(I.InteractionState(7), inp, now, CFG, families=())
    assert rep.n_trials == 0 and "no testable" in rep.notes[0]


# ------------------------------------------------------------------------- seeds, vault
def test_seed_ledger_refuses_reuse_and_is_deterministic():
    a, b = I.SeedLedger(5), I.SeedLedger(5)
    assert a.take("fresh", 1) == b.take("fresh", 1)
    with pytest.raises(I.SeedReuse):
        a.take("fresh", 1)
    assert a.take("fresh", 2) != a.take("fresh", 3)
    assert a.rng("x", 1).random() == b.rng("x", 1).random()
    assert I.SeedLedger(6).take("fresh", 1) != I.SeedLedger(5).take("fresh", 1)


def test_fresh_seeds_never_equal_discovery_seeds():
    s = I.SeedLedger(3)
    disc = {s.take("shuffle", i) for i in range(200)}
    fresh = {s.take("fresh", i) for i in range(200)}
    assert not disc & fresh


def test_holdout_vault_is_one_shot():
    v = I.HoldoutVault()
    assert v.open("w", ["a", "b"]) == 2
    assert v.open("w", ["c"]) == 3
    with pytest.raises(I.HoldoutSpent):
        v.open("w", ["b", "d"])
    assert v.was_opened("w", "a") and not v.was_opened("w", "d") and v.count("w") == 3


def test_second_search_cannot_reuse_the_holdout(planted):
    inp, now, st, rep = planted
    assert rep.validated()
    rep2 = I.step(st, inp, now, CFG)
    assert rep2.validated() == []
    assert any("holdout" in r for f in rep2.findings if f.fate is I.Fate.NEEDS_MORE_EVIDENCE for r in f.reasons)


# ------------------------------------------------------------------------- enumeration and roles
def test_infer_roles_and_first_match_wins():
    r = I.infer_roles(["m_vix", "volume_ratio", "atr_pct", "gap_pct", "mom_20", "rev_5", "sector_rel", "rs_20", "zzz"])
    assert r["market_vol"] == ("m_vix",) and "zzz" not in sum(r.values(), ())
    assert "atr_pct" in r["volatility"] and "sector_rel" in r["sector_strength"] and "rs_20" in r["stock_strength"]


def test_enumeration_covers_families_and_skips_untestable(prepared):
    P, _, _ = prepared
    trials, skipped = I.enumerate_trials(P, CFG)
    fams = {t.family for t in trials}
    assert {"volume_x_volatility", "pattern_x_regime", "pattern_x_pattern", "event_x_pattern", "momentum_x_event"} <= fams
    assert len({t.trial_id for t in trials}) == len(trials)
    assert isinstance(skipped, list)
    forms = {t.form for t in trials}
    assert {I.Form.PRODUCT, I.Form.CORNER_HH, I.Form.CORNER_HL, I.Form.MODULATION, I.Form.CORNER_H} <= forms


def test_enumeration_truncation_is_seeded_and_counted(prepared):
    P, _, _ = prepared
    cfg = dc.replace(CFG, max_candidates=10)
    a, sk = I.enumerate_trials(P, cfg, I.SeedLedger(1), 0)
    b, _ = I.enumerate_trials(P, cfg, I.SeedLedger(1), 0)
    assert len(a) == 10 and [t.trial_id for t in a] == [t.trial_id for t in b]
    assert any("truncated" in s["reason"] for s in sk)


def test_collinear_atoms_are_skipped():
    inp, now = I.synthetic_panel(seed=2, n_dates=300)
    X = inp.X.copy()
    X["atr_pct"] = X["volume_ratio"] * 2.0 + 1e-6
    P = I.prepare(dc.replace(inp, X=X), now, CFG)
    _, skipped = I.enumerate_trials(P, CFG)
    assert any("collinear" in s["reason"] for s in skipped)


def test_trial_ids_change_with_every_choice():
    t = I.Trial("f", "a", "b", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    assert t.trial_id != dc.replace(t, q=0.33).trial_id
    assert t.trial_id != dc.replace(t, form=I.Form.CORNER_HH).trial_id
    assert t.trial_id != dc.replace(t, horizon=10).trial_id
    assert t.spec().units() > t.simple_spec().units()


# ------------------------------------------------------------------------- statistic engine
def test_series_stat_edge_cases():
    ts = I.TrialSeries(np.arange(5), np.ones(5))
    assert not I.series_stat(ts, 6).ok
    const = I.TrialSeries(np.arange(40), np.ones(40))
    assert not I.series_stat(const, 6).ok or I.series_stat(const, 6).t != 0
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 1, 300)
    assert I.series_stat(I.TrialSeries(np.arange(300), v), 6).t > 4
    z = rng.normal(size=300)
    mod = I.TrialSeries(np.arange(300), 0.8 * z + rng.normal(size=300), z)
    assert I.series_stat(mod, 6).t > 5
    flat_z = I.TrialSeries(np.arange(300), v, np.zeros(300))
    assert not I.series_stat(flat_z, 6).ok


def test_fm_series_recovers_planted_coefficient(prepared):
    P, _, _ = prepared
    tr = I.Trial("volume_x_volatility", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    st = I.trial_stat(P, tr, CFG)
    assert st.ok and st.t > 8 and st.beta > 0
    other = I.Trial("x", "mom_20", "rev_5", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    assert abs(I.trial_stat(P, other, CFG).t) < 4


def test_one_sided_p_sign_matters():
    assert I.one_sided_p(3.0, 1.0) < 0.01 < I.one_sided_p(-3.0, 1.0)
    assert I.one_sided_p(-3.0, -1.0) < 0.01
    assert I.one_sided_p(float("nan"), 1.0) == 1.0


def test_dispersion_control_removes_volatility_scaling_artifact():
    """A market-level covariate that only scales the dispersion of y must not look like a slope modulation."""
    inp, now = I.synthetic_panel(seed=8, n_dates=500, product=0.0, regime=0.0)
    vix = inp.X["m_vix"].to_numpy()
    scale = np.exp(0.4 * (vix - vix.mean()) / vix.std())
    y = inp.y * scale
    P = I.prepare(dc.replace(inp, y=y), now, CFG)
    tr = I.Trial("x", "mom_20", "m_vix", I.Form.MODULATION, 0.0, 5, "cont*cont")
    with_ctl = abs(I.trial_stat(P, tr, CFG).t)
    assert with_ctl < 3.5


# ------------------------------------------------------------------------- transfer checks
def _series_for(P, tr):
    return I.trial_series(P, tr, CFG)


def test_cross_year_check_passes_stable_and_fails_single_year():
    inp, now = I.synthetic_panel(seed=1, n_dates=800)
    P = I.prepare(inp, now, CFG)
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    ok = I.cross_year_check(_series_for(P, tr), P, CFG, 1.0)
    assert ok["pass"] and ok["n_years"] >= 3
    inp2, now2 = I.synthetic_panel(seed=1, n_dates=800, product=0.15, single_year=True)
    P2 = I.prepare(inp2, now2, CFG)
    bad = I.cross_year_check(_series_for(P2, tr), P2, CFG, 1.0)
    assert not bad["pass"]


def test_cross_year_unusable_with_too_few_years():
    inp, now = I.synthetic_panel(seed=1, n_dates=200)
    P = I.prepare(inp, now, CFG)
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    out = I.cross_year_check(_series_for(P, tr), P, CFG, 1.0)
    assert not out["usable"] and not out["pass"]


def test_walk_forward_check_on_real_and_sign_flipped(prepared):
    P, _, _ = prepared
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    ts = _series_for(P, tr)
    good = I.walk_forward_check(ts, P, 0, P.all_dates.size, CFG, 1.0)
    bad = I.walk_forward_check(ts, P, 0, P.all_dates.size, CFG, -1.0)
    assert good["pass"] and not bad["pass"]


def test_cross_stock_check_holds_for_planted_and_fails_for_noise(prepared):
    P, _, _ = prepared
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    cs = I.cross_stock_check(P, tr, dc.replace(CFG, min_names=5), 1.0)
    assert cs["usable"] and cs["agree"] >= 0.75
    noise = I.Trial("n", "mom_20", "rev_5", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    assert not I.cross_stock_check(P, noise, dc.replace(CFG, min_names=5), 1.0)["pass"]


def test_fresh_seed_check_is_deterministic_and_single_use(prepared):
    P, _, _ = prepared
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    a = I.fresh_seed_check(P, tr, CFG, 1.0, I.SeedLedger(2), 0)
    b = I.fresh_seed_check(P, tr, CFG, 1.0, I.SeedLedger(2), 0)
    assert a == b and a["pass"]
    s = I.SeedLedger(2)
    I.fresh_seed_check(P, tr, CFG, 1.0, s, 0)
    with pytest.raises(I.SeedReuse):
        I.fresh_seed_check(P, tr, CFG, 1.0, s, 0)


def test_oos_compare_prefers_interaction_only_when_real():
    inp, now = I.synthetic_panel(seed=1, n_dates=800)
    P = I.prepare(inp, now, CFG)
    win = I.split_windows(P.all_dates.size, CFG)
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    real = I.oos_compare(P.window(*win.disc), P.window(*win.valid), tr, CFG)
    assert real["pass"] and real["gain"] > 0
    fake = I.Trial("n", "mom_20", "rev_5", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    assert not I.oos_compare(P.window(*win.disc), P.window(*win.valid), fake, CFG)["pass"]


def test_curvature_check_kills_curvature_proxy_and_keeps_real():
    inp, now = I.synthetic_panel(seed=9, product=0.0, regime=0.0, curvature=0.08, n_dates=600)
    P = I.prepare(inp, now, dc.replace(CFG, max_atom_corr=0.99))
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    raw = I.trial_stat(P, tr, CFG)
    ctl = I.curvature_check(P, tr, CFG)
    assert abs(ctl["t"]) < abs(raw["t"] if isinstance(raw, dict) else raw.t) * 0.6
    real_inp, real_now = I.synthetic_panel(seed=1, n_dates=600)
    Pr = I.prepare(real_inp, real_now, CFG)
    assert I.curvature_check(Pr, tr, CFG)["t"] > 6


def test_transform_check_none_for_corners():
    inp, now = I.synthetic_panel(seed=1, n_dates=300)
    P = I.prepare(inp, now, CFG)
    corner = I.Trial("v", "volume_ratio", "atr_pct", I.Form.CORNER_HH, 0.33, 5, "cont*cont")
    assert I.transform_check(P, corner, CFG) is None
    prod = dc.replace(corner, form=I.Form.PRODUCT, q=0.0)
    assert I.transform_check(P, prod, CFG)["t"] > 5


def test_leak_tripwire_flags_only_the_leak():
    inp, now = I.synthetic_panel(seed=6, leak=True, n_dates=300)
    X = inp.X.copy()
    P = I.prepare(dc.replace(inp, X=X, roles={"gap": ("gap_leak", "gap_pct"), "volume": ("volume_ratio",)}), now, CFG)
    flagged = {d["atom"] for d in I.leak_tripwires(P)}
    assert flagged == {"gap_leak"}
    clean, cnow = I.synthetic_panel(seed=6, n_dates=300)
    assert I.leak_tripwires(I.prepare(clean, cnow, CFG)) == []


# ------------------------------------------------------------------------- understanding, clustering, shrinkage
def test_interaction_surface_shape_of_planted_product(prepared):
    P, _, _ = prepared
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    s = I.interaction_surface(P, tr, CFG)
    assert s["shape"] in ("multiplicative", "mixed", "corner") and s["contrast"] > 0
    assert "reinforces" in I.describe(tr, s)
    null = I.interaction_surface(P, dc.replace(tr, a="mom_20", b="rev_5"), CFG)
    assert abs(null["contrast"]) < abs(s["contrast"])


def test_describe_unresolved():
    tr = I.Trial("v", "a", "b", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    assert "no distinct" in I.describe(tr, {"shape": "unresolved", "contrast": float("nan")})


def test_redundancy_clusters_group_forms_of_one_effect(planted):
    rep = planted[3]
    multi = [c for c in rep.clusters if len(c) > 1]
    assert multi and 1.0 <= rep.n_effective < sum(len(c) for c in rep.clusters)
    assert I.redundancy_clusters({})["n_effective"] == 0.0
    rng = np.random.default_rng(0)
    pos = np.arange(200)
    indep = {k: I.TrialSeries(pos, rng.normal(size=200)) for k in "abc"}
    r = I.redundancy_clusters(indep)
    assert len(r["clusters"]) == 3 and r["n_effective"] > 2.5


def test_shrinkage_pulls_survivors_toward_zero(planted):
    rep = planted[3]
    assert rep.shrunk
    by_id = {f.trial.trial_id: f for f in rep.findings}
    for k, v in rep.shrunk.items():
        assert abs(v) <= abs(by_id[k].stats["disc_beta"]) + 1e-12


def test_family_table_columns_and_counts(planted):
    ft = I.family_table(planted[3].findings)
    assert ft["tried"].sum() == planted[3].n_trials
    assert ft["validated"].sum() == len(planted[3].validated())
    assert I.family_table([]).empty


def test_explain_failure_vocabulary(planted, null):
    causes = {I.explain_failure(f) for f in planted[3].findings + null[3].findings}
    assert FailureCause.UNKNOWN in causes or FailureCause.FALSE_PATTERN in causes
    v = planted[3].validated()[0]
    assert I.explain_failure(v) is None


# ------------------------------------------------------------------------- power, planning, monitoring
def test_detection_curve_is_monotone_and_mds_finite():
    inp, now = I.synthetic_panel(seed=1, n_dates=420, product=0.0, regime=0.0)
    P = I.prepare(inp, now, CFG)
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    cur = I.detection_curve(P, tr, [0.0, 0.05, 0.3, 1.0], CFG, 50, 4, I.SeedLedger(3))
    assert cur["power"].iloc[0] <= 0.25 and cur["power"].iloc[-1] >= 0.99
    assert cur["median_t"].is_monotonic_increasing
    assert 0.0 < I.minimum_detectable_size(cur) <= 1.0
    assert math.isnan(I.minimum_detectable_size(cur.iloc[:1]))


def test_plant_interaction_does_not_touch_original():
    inp, now = I.synthetic_panel(seed=1, n_dates=300)
    P = I.prepare(inp, now, CFG)
    y0 = P.y.copy()
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    Q = I.plant_interaction(P, tr, 0.5)
    assert np.array_equal(P.y, y0) and not np.array_equal(Q.y, y0)


def test_plan_search_budget_and_weights(prepared):
    P, _, _ = prepared
    trials, _ = I.enumerate_trials(P, CFG)
    full = I.plan_search(trials, CFG)
    assert not full.truncated and sum(full.planned.values()) == len(trials)
    small = I.plan_search(trials, CFG, budget=12)
    assert small.truncated and sum(small.planned.values()) <= 12 and small.bonferroni_t < full.bonferroni_t
    kept = I.restrict_trials(trials, small, I.SeedLedger(1), 0)
    assert len(kept) == sum(small.planned.values())
    w = I.family_weights([{"by_family": {"a": (100, 5), "b": (100, 0)}}], ["a", "b", "c"])
    assert w["a"] > w["b"] >= 0.05 / 1.2 and sum(w.values()) == pytest.approx(1.0)


def test_budgeted_step_counts_only_tested():
    inp, now = I.synthetic_panel(seed=1, n_dates=500)
    st = I.InteractionState(7)
    rep = I.step(st, inp, now, dc.replace(CFG, n_shuffles=12), budget=10)
    assert rep.n_trials <= 10 and rep.m_total == rep.n_trials and any("budget" in n for n in rep.notes)


def test_monitor_finding_statuses():
    inp, now = I.synthetic_panel(seed=1, n_dates=900)
    P = I.prepare(inp, now, CFG)
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    early = str(pd.Timestamp(P.all_dates[300]))[:10]
    assert I.monitor_finding(P, tr, 1.0, early, now, CFG)["status"] == "HOLDING"
    assert I.monitor_finding(P, tr, -1.0, early, now, CFG)["status"] == "CONTRADICTED"
    late = str(pd.Timestamp(P.all_dates[-5]))[:10]
    assert I.monitor_finding(P, tr, 1.0, late, now, CFG)["status"] == "INSUFFICIENT"
    noise = dc.replace(tr, a="mom_20", b="rev_5")
    assert I.monitor_finding(P, noise, 1.0, early, now, CFG)["status"] in ("UNCONFIRMED", "INSUFFICIENT")


def test_always_valid_p_controls_repeated_looks():
    rng = np.random.default_rng(0)
    fails = sum(I.always_valid_p(rng.normal(size=400))["stop"] is not None for _ in range(60))
    assert fails <= 6
    assert I.always_valid_p(rng.normal(0.4, 1, 300))["stop"] is not None
    assert I.always_valid_p([1.0, 2.0])["p"] == 1.0


def test_downside_profile_and_atom_health(prepared):
    P, _, _ = prepared
    d = I.downside_profile(pd.Series([0.01, -0.05, 0.02, 0.03, -0.01]))
    assert d["max_dd"] == pytest.approx(0.05) and d["worst5"] == pytest.approx(-0.05)
    assert math.isnan(I.downside_profile(pd.Series(dtype=float))["mean"])
    h = I.atom_health(P)
    assert h["healthy"].all() and set(h["level"]) == {"row", "date"}


def test_state_breakdown_reports_three_parts(prepared):
    P, _, _ = prepared
    tr = I.Trial("v", "volume_ratio", "atr_pct", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    sb = I.state_breakdown(P, tr, CFG)
    assert sb["atom"] == "m_vix" and len(sb["t_by_part"]) == 3


# ------------------------------------------------------------------------- firewall, records, persistence
def test_assert_identity_free_blocks_years_dates_tickers():
    I.assert_identity_free({"a": "volume x atr", "n": 3}, ["AAPL"])
    for bad in ({"a": "in 2008"}, {"a": "on 2020-03-16"}, {"a": "AAPL gapped"}, {"2019": 1}):
        with pytest.raises(FirewallBreach):
            I.assert_identity_free(bad, ["AAPL"])


def test_matured_records_gate_blocks_same_year_rerun(planted):
    rep = planted[3]
    recs = I.to_matured_records(rep, "2026-09-29T00:00:00", ["T001"], seed=7)
    assert recs and all(isinstance(r, MaturedRecord) and r.namespace is Namespace.MATURED_RESEARCH for r in recs)
    with pytest.raises(FirewallBreach):
        recs[0].gate(rep.evidence_end)                      # the same day
    with pytest.raises(FirewallBreach):
        recs[0].gate("2016-06-01")                          # a replay of an earlier period
    assert recs[0].gate("2099-01-01")["fate"] == "VALIDATED"
    I.assert_identity_free(recs[0].payload, ["T001"])


def test_no_records_for_empty_report():
    assert I.to_matured_records(I.empty_report(0, "", "x"), "2026-09-29") == []


def test_invalid_finding_is_refused():
    tr = I.Trial("f", "a", "b", I.Form.PRODUCT, 0.0, 5, "cont*cont")
    f = I.Finding(tr, I.Fate.VALIDATED, Stage.CHEAP_SCREEN, 5.0, {"disc_t": 3.0}, ("ok",))
    errs = f.validate()
    assert len(errs) >= 2
    assert I.Finding(tr, I.Fate.NOISE, Stage.CHEAP_SCREEN, 5.0, {}, ()).validate()


def test_state_roundtrip_and_integrity(planted, tmp_path):
    st = planted[2]
    p = tmp_path / "state.json"
    sha = I.save_state(st, p)
    back = I.load_state(p)
    assert back.ledger.to_dict() == st.ledger.to_dict() and back.seeds.used() == st.seeds.used()
    assert I.state_to_dict(back) == I.state_to_dict(st) and len(sha) == 32
    with pytest.raises(I.SeedReuse):
        back.seeds.take("shuffle", (0, 0))              # seeds spent before the save stay spent after the load
    raw = p.read_text(encoding="utf-8")
    assert '"base": 7' in raw
    p.write_text(raw.replace('"base": 7', '"base": 8'), encoding="utf-8")
    with pytest.raises(I.InteractionError):
        I.load_state(p)


def test_write_report_files(planted, tmp_path):
    out = I.write_report(planted[3], tmp_path / "r", seed=7)
    import json
    body = json.loads(open(out["json"], encoding="utf-8").read())
    assert body["m_total"] == planted[3].m_total and body["provenance"]
    assert "INTERACTION SEARCH" in open(out["text"], encoding="utf-8").read()


def test_render_text_states_the_chance_baseline(planted, null):
    txt = I.render_text(null[3])
    assert "expected from chance" in txt and "shuffled" in txt
    assert "VALIDATED" in I.render_text(planted[3])


def test_replication_across_base_seeds_is_stable():
    inp, now = I.synthetic_panel(seed=1, n_dates=800)
    out = I.replicate_search(inp, now, dc.replace(CFG, n_shuffles=12), [11, 12])
    assert out["n_runs"] == 2 and out["mean_jaccard"] >= 0.5
    assert all(v for v in out["validated"]) and out["stable"]


def test_determinism_same_inputs_same_report():
    inp, now = I.synthetic_panel(seed=1, n_dates=420)
    cfg = dc.replace(CFG, n_shuffles=12)
    a = I.step(I.InteractionState(7), inp, now, cfg)
    b = I.step(I.InteractionState(7), inp, now, cfg)
    assert I.report_to_dict(a) == I.report_to_dict(b)
