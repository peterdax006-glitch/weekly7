"""Tests for the same-year rerun harness and the five frozen controls (contract section 25, checklist E01-E07, L01-L05).
Synthetic planted worlds only; every planted case asserts the defect is caught, plus empty/degenerate cases."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning import controls as CT
from engine.learning import planted_world as PW
from engine.learning import same_year as SY
from engine.learning.core import FirewallBreach

ITEMS = (PW.Item("strong", PW.STRONG, (("f0", 4),), 0.02), PW.Item("neg", PW.NEGATIVE, (("f1", 4),), -0.015),
         PW.Item("weak", PW.WEAK, (("f2", 0),), 0.012), PW.Item("noise", PW.NOISE, (("f3", 4),)))


def small_world(seed=3, weeks=52, stocks=60, scale=1.0):
    items = tuple(dataclasses.replace(it, effect=it.effect * scale) for it in ITEMS)
    spec = PW.WorldSpec("sy_test", items, weeks=weeks, stocks=stocks, n_feat=8, discovery_end=weeks // 2, confirm_end=(weeks * 3) // 4).check()
    return PW.make_world(spec, seed)


@pytest.fixture(scope="module")
def world():
    return small_world()


@pytest.fixture(scope="module")
def harness(world):
    h = SY.SameYearHarness(world, cfg=SY.HarnessConfig(n_runs=6, seed=0))
    h.run_all()
    return h


# ------------------------------------------------------------------------------------------------ freezing (L01-L05)
def test_every_control_has_a_stable_code_and_config_hash():
    a, b = CT.standard_controls(), CT.standard_controls()
    for L in CT.LETTERS:
        assert a[L].record.code_hash == b[L].record.code_hash
        assert a[L].record.config_hash == b[L].record.config_hash
    assert len({a[L].record.code_hash for L in CT.LETTERS}) == 5


def test_patched_code_is_refused():
    class Probe(CT.NoLearning):
        letter, name = "A", "probe_a"
    fc = CT.FrozenControl(Probe, {})
    fc.build(0)
    Probe.decide = lambda self, X, moment: pd.Series(1.0, index=X.index)          # planted defect: behaviour changed after freezing
    with pytest.raises(CT.ControlChanged, match="code changed"):
        fc.build(0)


def test_config_is_detached_and_registry_rejects_a_changed_config(tmp_path):
    cfg = {"t_thr": 2.5}
    fc = CT.FrozenControl(CT.EvidenceLearner, cfg)
    cfg["t_thr"] = 0.1                                             # mutating the caller's dict must not reach the frozen one
    assert fc.config == {"t_thr": 2.5}
    reg = CT.ControlRegistry(tmp_path / "frozen.json")
    assert reg.register(fc) is True
    other = CT.FrozenControl(CT.EvidenceLearner, {"t_thr": 1.0})
    with pytest.raises(CT.ControlChanged, match="config_hash"):
        reg.verify(other)
    reg2 = CT.ControlRegistry(tmp_path / "frozen.json")             # persisted: a new process sees the same record
    assert reg2.latest("B")["config_hash"] == fc.record.config_hash


def test_supersede_appends_and_needs_a_reason(tmp_path):
    reg = CT.ControlRegistry(tmp_path / "f.json")
    fc = CT.FrozenControl(CT.NoLearning, {})
    reg.register(fc)
    with pytest.raises(CT.ControlContractError):
        reg.supersede(fc, "  ")
    v2 = reg.supersede(CT.FrozenControl(CT.NoLearning, {"x": 1}), "test change")
    assert v2.record.version == 2 and len(reg.history) == 2 and reg.latest("A")["version"] == 2


def test_pluggable_learner_is_part_of_the_freeze():
    class Mine(CT.EvidenceLearner):
        pass
    f1 = CT.FrozenControl(CT.LegitimateLearner, {}, factory=Mine)
    f2 = CT.FrozenControl(CT.LegitimateLearner, {})
    assert f1.record.code_hash != f2.record.code_hash
    assert isinstance(f1.build(0).inner, Mine)


def test_interface_check_rejects_an_incomplete_control():
    class Bad:
        def decide(self, X, m): return None
    with pytest.raises(CT.ControlContractError):
        CT.check_interface(Bad())
    with pytest.raises(CT.ControlContractError):
        CT.LegitimateLearner({}, 0, factory=lambda cfg, seed: Bad())


# ------------------------------------------------------------------------------------------------ time and evidence
def test_real_date_needs_a_gate_and_is_logged():
    m = CT.Moment(pd.Timestamp("2107-01-05"), 0, pd.Timestamp("2009-01-02"))
    with pytest.raises(FirewallBreach):
        m.real(object())
    assert m.real(CT.MemoryGate("B")) == pd.Timestamp("2009-01-02") and m.reads == ["B"]
    with pytest.raises(ValueError):
        CT.MemoryGate("")


def test_evidence_date_filter_hides_prior_runs_future():
    ev = CT.KeyEvidence(2)
    one = np.array([10.0, 10.0])
    s = np.array([1.0, -1.0])
    ev.begin_run()
    for d in ("2009-01-02", "2009-01-09", "2009-01-16"):
        ev.advance(pd.Timestamp(d) + pd.Timedelta(days=7))
        ev.add(pd.Timestamp(d), one, s, one * 0.01)
    ev.begin_run()                                                   # run 2 starts: nothing visible yet
    ev.advance(pd.Timestamp("2009-01-10"))                           # only week 1 (matures 01-09) may be visible
    assert ev.N[0] == 10.0
    ev.advance(pd.Timestamp("2009-01-16"))
    assert ev.N[0] == 20.0                                           # week 2 matured 01-16: <= now, week 3 (01-23) still hidden
    with pytest.raises(FirewallBreach, match="backwards"):
        ev.advance(pd.Timestamp("2009-01-01"))


def test_evidence_rejects_an_unmatured_record_and_scales_prior_runs():
    ev = CT.KeyEvidence(1, prior_weight=0.5)
    ev.begin_run()
    ev.advance(pd.Timestamp("2009-01-05"))
    with pytest.raises(FirewallBreach):
        ev.add(pd.Timestamp("2009-01-05"), np.ones(1), np.ones(1), np.ones(1))       # matures 01-12 > now
    ev.advance(pd.Timestamp("2009-01-12"))
    ev.add(pd.Timestamp("2009-01-05"), np.array([10.0]), np.array([1.0]), np.array([0.1]))
    ev.begin_run()
    ev.advance(pd.Timestamp("2009-02-01"))
    assert ev.N[0] == 5.0                                            # a rerun of a week counts for prior_weight, not for a new sample
    with pytest.raises(ValueError):
        CT.KeyEvidence(1, prior_weight=0.0)


def test_pattern_memory_veto_switches_off_a_key_that_broke(tmp_path):
    keys = ["f0 q4", "f1 q4"]
    g = CT.PatternMemoryGate(keys)
    recs, d = [], pd.Timestamp("2009-01-02")
    for w in range(48):
        eff = 0.02 if w < 34 else -0.03                                # key 0 works for 8 months then reverses; key 1 always works
        n = np.array([20.0, 20.0])
        mean = np.array([eff, 0.02])
        recs.append({"obs": (d + pd.Timedelta(days=7 * w)).to_datetime64(), "n": n, "s": n * mean, "ss": n * (0.05 ** 2 + mean ** 2)})
    assert g.record_run("r0", recs, min_n=30) > 0
    veto = g.vetoed(d + pd.Timedelta(days=7 * 48 + 30))
    assert "f0 q4" in veto and "f1 q4" not in veto
    assert g.vetoed(d + pd.Timedelta(days=7 * 48 + 30)) is g._cache[1]        # cached while nothing new matured
    assert g.vetoed(pd.Timestamp("2008-06-01")) == set()                       # before anything matured: no veto, no crash


# ------------------------------------------------------------------------------------------------ the controls
def week_data(world, i=10):
    p = SY.make_run_panel(world, 0, 0, None, "kept")
    return p, p.week(i)


def test_no_learning_ignores_experience(world):
    p, (X, y, m) = week_data(world)
    a = CT.standard_controls()["A"].build(5)
    s0 = a.decide(X, m)
    a.observe(X, y, m, p.week(11)[2])
    pd.testing.assert_series_equal(a.decide(X, m), s0)
    assert a.state_size() == 0 and CT.standard_controls()["A"].build(6).decide(X, m).equals(s0) is False


def test_learner_with_no_evidence_abstains_and_empty_week_is_safe(world):
    p, (X, y, m) = week_data(world)
    b = CT.standard_controls()["B"].build(0)
    b.begin_run(0)
    assert (b.decide(X, m) == 0).all()
    b.observe(X.iloc[0:0], y.iloc[0:0], m, m)                       # empty cross-section
    assert b.state_size() == 0


def test_canary_column_is_a_breach(world):
    X = world.X.xs(world.dates[3], level=0)
    with pytest.raises(FirewallBreach):
        CT.feature_columns(X)                                       # world.X still carries canary_future_ret
    with pytest.raises(FirewallBreach):
        SY.LeakGuard().check_frame(X)


def test_random_learner_matches_b_update_magnitude(world):
    p = SY.make_run_panel(world, 0, 0, None, "kept")
    b, d = CT.standard_controls()["B"].build(1), CT.standard_controls()["D"].build(1)
    b.begin_run(0); d.begin_run(0)
    total = 0.0
    for i in range(12):
        X, y, m = p.week(i)
        if i:
            b.observe(*prev, m); d.observe(*prev, m)
            X0, y0, _ = prev
            total += float(np.linalg.norm(b.inner.update_vector(X0, y0)))
        prev = (X, y, m)
    assert d.mass == pytest.approx(total, rel=1e-9)
    assert np.linalg.norm(d.w) > 0


def test_identity_memoriser_recalls_kept_identities_and_loses_them_under_reidentify(world):
    kept = SY.make_run_panel(world, 0, 0, None, "kept")
    c = CT.FrozenControl(CT.IdentityMemoriser, {"mode": "ticker_date"}).build(0)
    g = SY.LeakGuard()
    r1 = SY.play_run(c, kept, 0, g)
    r2 = SY.play_run(c, kept, 1, g)
    assert r2.gain > 0.05 and r2.gain > r1.gain + 0.05               # identical identities: it looks brilliant
    assert c.hit_rate() > 0.4
    fresh = SY.make_run_panel(world, 2, 0, None, "fresh_plain")
    h0, l0 = c.hits, c.lookups
    r3 = SY.play_run(c, fresh, 2, g)
    assert c.hits == h0 and c.lookups > l0                           # no code and no date matches after reidentification
    assert abs(r3.gain) < 0.01


def test_numeric_memoriser_needs_the_perturbations_to_be_stopped(world):
    c_fac = lambda: CT.FrozenControl(CT.IdentityMemoriser, {"mode": "row_hash"}).build(0)
    g = SY.LeakGuard()
    plain, pert = c_fac(), c_fac()
    SY.play_run(plain, SY.make_run_panel(world, 0, 0, None, "kept"), 0, g)
    SY.play_run(pert, SY.make_run_panel(world, 0, 0, None, "kept"), 0, g)
    gain_plain = SY.play_run(plain, SY.make_run_panel(world, 1, 0, None, "fresh_plain"), 1, g).gain
    gain_pert = SY.play_run(pert, SY.make_run_panel(world, 1, 0, SY.PerturbConfig.standard(), "fresh_perturbed"), 1, g).gain
    assert gain_plain > 0.05                                         # new codes and dates, same numbers: still recognised
    assert abs(gain_pert) < 0.01                                     # perturbed numbers: nothing to recall
    assert plain.hit_rate() > 0.3 and pert.hit_rate() < 0.01


def test_leaky_learner_is_flagged_with_provenance_and_ic(harness):
    fp = harness.results["fresh_perturbed"]
    assert all(r.guard.provenance_breaches and r.guard.ic_leak for r in fp.runs["E"])
    assert "not strictly before" in fp.runs["E"][0].guard.provenance_breaches[0]
    assert harness.skill_at_run_one("fresh_perturbed", "E") > 0.5 > harness.skill_at_run_one("fresh_perturbed", "B")
    for L in "ABD":
        assert not any(r.guard.flagged for r in fp.runs[L])


def test_a_leak_that_lies_about_its_provenance_is_caught_by_the_ic_test(world):
    class Liar(CT.LeakyLearner):
        def seen_through(self):
            return None
    panel = SY.make_run_panel(world, 0, 0, None, "kept")
    r = SY.play_run(Liar({}, 0), panel, 0, SY.LeakGuard())
    assert not r.guard.provenance_breaches and r.guard.ic_leak and r.guard.ic_mean > 0.5


def test_a_non_leaky_control_that_breaches_stops_the_run(world):
    class Sloppy(CT.NoLearning):
        def seen_through(self):
            return pd.Timestamp("2100-01-01")                        # claims to have seen a future outcome
    panel = SY.make_run_panel(world, 0, 0, None, "kept")
    with pytest.raises(FirewallBreach):
        SY.play_run(Sloppy({}, 0), panel, 0, SY.LeakGuard())


# ------------------------------------------------------------------------------------------------ the learning curves
def test_legitimate_learner_beats_no_learning_and_improves_over_runs(harness):
    fp = harness.results["fresh_perturbed"]
    beat = harness.beats_baseline()
    assert beat.positive and beat.mean > 0.003
    gb = fp.gains("B")
    assert gb[1:].mean() > gb[0] + 0.002                             # run 1 has nothing to learn from; later runs carry memory
    assert gb.mean() > fp.gains("A").mean() + 0.003


def test_legitimate_learner_does_not_resemble_the_memoriser(harness):
    gap_b, gap_c = harness.memorisation_gap("B"), harness.memorisation_gap("C")
    assert gap_c.positive and gap_c.mean > 0.05                      # the harness can see memorisation
    assert gap_b.includes_zero                                       # B gains the same with identities kept or disguised
    ident = harness.identity_gap("kept")
    assert ident.positive and ident.mean > 0.03
    fp = harness.results["fresh_perturbed"]
    assert abs(fp.gains("C").mean()) < 0.01                          # in the honest protocol C learns nothing


def test_random_learner_shows_no_curve(harness):
    assert not harness.curve_trend("fresh_perturbed", "D").positive
    assert abs(harness.results["fresh_perturbed"].gains("D").mean()) < 0.004
    g = SY.block_bootstrap_mean(SY.stacked(harness.results["fresh_perturbed"].runs["A"]).mean(0))
    assert g.includes_zero


def test_judgement_says_learning_and_records_the_label(harness):
    j = harness.judge()
    assert j.verdict == SY.Verdict.LEARNING, j.reasons
    assert j.label == "IMPLEMENTED — NOT VALIDATED"
    assert j.facts["E_flagged_share"] == 1.0 and j.facts["C_memorisation_gap"]["lo"] > 0
    assert j.as_record()["verdict"] == "LEARNING"


def test_render_and_curves_and_fingerprint(harness):
    txt = harness.render()
    assert "mode kept" in txt and "| E |" in txt
    cv = harness.curve("fresh_perturbed", "B")
    assert len(cv) == 6 and np.all(np.diff(cv.column("experience_count")) > 0)
    assert np.isfinite(harness.library_trend("fresh_perturbed", "B").n)
    assert harness.fingerprint() == harness.fingerprint()
    assert dataclasses.asdict(harness.perturbation_price("B"))["n"] > 0


def test_run_is_deterministic(world):
    a = SY.play_run(CT.standard_controls()["B"].build(4), SY.make_run_panel(world, 0, 7, SY.PerturbConfig.standard()), 0, SY.LeakGuard())
    b = SY.play_run(CT.standard_controls()["B"].build(4), SY.make_run_panel(world, 0, 7, SY.PerturbConfig.standard()), 0, SY.LeakGuard())
    np.testing.assert_array_equal(a.gains, b.gains)


def test_a_broken_harness_declares_itself_void(world):
    class FakeE(CT.NoLearning):                                     # planted defect: the 'leaky' control does not leak
        letter, name, expects_breach = "E", "fake_e", True
    frozen = CT.standard_controls()
    frozen["E"] = CT.FrozenControl(FakeE, {})
    h = SY.SameYearHarness(world, frozen, SY.HarnessConfig(n_runs=5, seed=1))
    h.run_all()
    j = h.judge()
    assert j.verdict == SY.Verdict.VOID_HARNESS
    assert any("leaky learner E was not caught" in r for r in j.reasons)


def test_a_learner_that_does_not_learn_is_reported_as_no_learning(world):
    frozen = CT.standard_controls(legit_factory=lambda cfg, seed: CT.NoLearning(cfg, seed))
    h = SY.SameYearHarness(world, frozen, SY.HarnessConfig(n_runs=5, seed=2))
    h.run_all()
    assert h.judge().verdict == SY.Verdict.NO_LEARNING


def test_too_few_runs_is_insufficient_not_a_verdict(world):
    h = SY.SameYearHarness(world, cfg=SY.HarnessConfig(n_runs=2, seed=0))
    h.run_all()
    assert h.judge().verdict == SY.Verdict.INSUFFICIENT_RUNS


def test_harness_refuses_a_changed_control(world):
    frozen = CT.standard_controls()
    reg = CT.ControlRegistry()
    h = SY.SameYearHarness(world, frozen, SY.HarnessConfig(n_runs=1), registry=reg)
    CT.NoLearning.decide, orig = (lambda self, X, moment: pd.Series(0.0, index=X.index)), CT.NoLearning.decide
    try:
        with pytest.raises(CT.ControlChanged):
            h.run_mode("kept")
    finally:
        CT.NoLearning.decide = orig
    with pytest.raises(ValueError):
        SY.SameYearHarness(world, {"A": frozen["A"]})


# ------------------------------------------------------------------------------------------------ perturbations
def test_disguise_changes_codes_and_dates_but_keeps_real_dates(world):
    p = SY.make_run_panel(world, 0, 0, None, "fresh_plain")
    kept = SY.make_run_panel(world, 0, 0, None, "kept")
    assert set(p.X.index.get_level_values(1)).isdisjoint(set(kept.X.index.get_level_values(1)))
    assert p.weeks[0][0] != p.weeks[0][1] and [r for _, r in p.weeks] == [r for _, r in kept.weeks]
    assert (p.X.index.get_level_values(0).min() - kept.X.index.get_level_values(0).min()).days % 364 == 0
    q = SY.make_run_panel(world, 1, 0, None, "fresh_plain")
    assert p.weeks[0][0] != q.weeks[0][0]                            # each run is freshly disguised


def test_each_perturbation_does_what_it_says(world):
    base = SY.make_run_panel(world, 0, 0, None, "fresh_plain")
    for name in SY.PERTURBATIONS:
        cfg = SY.PerturbConfig.standard().only(name)
        pan = SY.make_run_panel(world, 0, 0, cfg, "fresh_perturbed")
        if name == "sub_universe":
            assert pan.X.index.get_level_values(1).nunique() == round(60 * 0.7)
        if name == "week_jitter":
            assert len(pan) < len(base) and len(pan) >= len(base) - 6
        if name == "feature_noise":
            assert not np.allclose(pan.X["f0"].to_numpy(), base.X["f0"].to_numpy())
            np.testing.assert_allclose(pan.X["m_vix"].to_numpy(), base.X["m_vix"].to_numpy())
        if name == "market_noise":
            m = pan.X["m_vix"].groupby(level=0).nunique()
            assert (m == 1).all()                                    # one shock per week, shared by every name
            assert not np.allclose(pan.X["m_vix"].groupby(level=0).first().to_numpy(), base.X["m_vix"].groupby(level=0).first().to_numpy())
        if name == "common_mode":
            ex_b = base.y - base.y.groupby(level=0).transform("mean")
            ex_p = pan.y - pan.y.groupby(level=0).transform("mean")
            np.testing.assert_allclose(ex_p.to_numpy(), ex_b.to_numpy(), atol=1e-12)     # cancels in excess returns
            assert not np.allclose(pan.y.to_numpy(), base.y.to_numpy())
        if name == "vol_scaling":
            assert not np.allclose(pan.y.to_numpy(), base.y.to_numpy())
            wm_b, wm_p = base.y.groupby(level=0).mean().to_numpy(), pan.y.groupby(level=0).mean().to_numpy()
            np.testing.assert_allclose(wm_b, wm_p, atol=1e-12)         # the market's weekly return is untouched
            ex_b = base.y - base.y.groupby(level=0).transform("mean")
            ex_p = pan.y - pan.y.groupby(level=0).transform("mean")
            assert np.corrcoef(ex_b, ex_p)[0, 1] > 0.85                # names keep their order: patterns survive
    again = SY.make_run_panel(world, 0, 0, SY.PerturbConfig.standard(), "fresh_perturbed")
    pd.testing.assert_frame_equal(again.X, SY.make_run_panel(world, 0, 0, SY.PerturbConfig.standard(), "fresh_perturbed").X)
    assert not any(c.startswith("canary") for c in again.X.columns)


def test_perturbation_config_validation_and_degenerate_cases(world):
    assert SY.PerturbConfig().is_off and not SY.PerturbConfig.standard().is_off
    assert SY.PerturbConfig.standard().scaled(0).is_off
    with pytest.raises(ValueError):
        SY.make_run_panel(world, 0, 0, SY.PerturbConfig(sub_universe=0.05), "fresh_perturbed")
    with pytest.raises(ValueError):
        SY.make_run_panel(world, 0, 0, None, "bogus")
    with pytest.raises(KeyError):
        SY.PerturbConfig.standard().only("nope")
    with pytest.raises(ValueError):
        SY.PerturbConfig.standard().scaled(-1)
    assert len(SY.HarnessConfig(n_runs=0).validate()) == 1


# ------------------------------------------------------------------------------------------------ recognition probe and cost
@pytest.fixture(scope="module")
def probe_worlds(world):
    return world, [small_world(11), small_world(12)]


def probe(world, others, cfg, mode, n=6):
    same = SY.presentations(world, n, 0, cfg, mode)
    oth = [SY.presentations(o, 4, 50 + k, cfg, mode) for k, o in enumerate(others)]
    return SY.recognition_probe(same, oth, seed=0)


def test_probe_recognises_a_disguised_rerun_without_perturbation(probe_worlds):
    r = probe(*probe_worlds, None, "fresh_plain")
    assert r["tier1"].auc > 0.95 and r["tier1"].lo > 0.85 and not r["tier1"].consistent_with_chance
    assert r["tier2"].auc > 0.95


def test_perturbation_takes_the_numeric_fingerprint_to_near_chance(probe_worlds):
    plain = probe(*probe_worlds, None, "fresh_plain")
    pert = probe(*probe_worlds, SY.PerturbConfig.standard(), "fresh_perturbed")
    assert pert["tier1"].auc < plain["tier1"].auc - 0.25
    assert pert["tier1"].auc < 0.75 and pert["tier1"].single_best < 0.8
    # honest residual: a fuzzy row-matching adversary is NOT defeated at this strength; the probe must say so, not hide it
    assert pert["tier2"].auc > pert["tier1"].auc


def test_probe_needs_enough_material(probe_worlds):
    w, o = probe_worlds
    with pytest.raises(ValueError):
        SY.recognition_probe(SY.presentations(w, 2, 0, None, "fresh_plain"), [SY.presentations(o[0], 3, 1, None, "fresh_plain")])
    with pytest.raises(ValueError):
        SY.recognition_probe(SY.presentations(w, 4, 0, None, "fresh_plain"), [])


def test_statistics_helpers_and_degenerate_inputs():
    assert SY.auc_score(np.array([1, 1, 0, 0], bool), np.array([3, 2, 1, 0.0])) == 1.0
    assert SY.auc_score(np.array([1, 0], bool), np.array([1.0, 1.0])) == 0.5
    assert SY.auc_score(np.array([], bool), np.array([])) == 0.5
    assert np.isnan(SY.block_bootstrap_mean(np.array([1.0, 2.0])).mean)
    assert SY.paired_gap([], []).n == 0
    assert SY.slope_ci(np.zeros((2, 10))).lo == float("-inf")
    up = SY.slope_ci(np.arange(6)[:, None] * 0.01 + np.random.default_rng(0).normal(0, 1e-4, (6, 30)))
    assert up.positive
    assert SY.pick_top(pd.Series([], dtype=float), 3, (0,)).size == 0
    assert SY.rank_ic(np.array([1.0, 1, 1, 1, 1, 1]), np.arange(6.0)) == 0.0
    assert SY.max_lagged_corr(np.arange(20.0), np.arange(20.0)) == pytest.approx(1.0)
    assert SY.ks_distance(np.arange(5.0), np.arange(5.0)) == 0.0


def test_information_cost_shows_patterns_survive_and_what_each_perturbation_costs():
    rows = {c.name: c for c in SY.information_cost(small_world(scale=2.0), SY.PerturbConfig.standard(), seeds=(0, 1))}
    assert set(rows) == set(SY.PERTURBATIONS) | {"all"}
    assert rows["all"].detect_rate >= 0.5 and rows["all"].key_auc > 0.85          # planted patterns stay recoverable
    assert rows["feature_noise"].t_retention < 0.8 and rows["sub_universe"].t_retention < 0.9
    assert rows["market_noise"].gate_agreement < 0.9 and rows["market_noise"].t_retention == pytest.approx(1.0)
    assert rows["common_mode"].t_retention == pytest.approx(1.0, abs=1e-6)         # free perturbation: cancels in excess returns
    assert rows["all"].t_retention < min(rows["vol_scaling"].t_retention, rows["week_jitter"].t_retention)
    assert rows["sub_universe"].rows_ratio == pytest.approx(0.7, abs=0.02)


def test_information_cost_needs_a_planted_single_key():
    spec = PW.noise_only_spec(weeks=30, stocks=30)
    with pytest.raises(ValueError):
        SY.information_cost(PW.make_world(spec, 0), SY.PerturbConfig.standard())
    assert SY.planted_single_keys(small_world()) == {"f0 q4": 1, "f1 q4": -1, "f2 q0": 1}


def test_frontier_trades_recognisability_against_recoverability(probe_worlds):
    w, o = probe_worlds
    fr = SY.perturbation_frontier(w, o, levels=(0.0, 1.0, 1.5), n_runs=5, seed=0)
    assert fr["tier1_auc"].iloc[0] > 0.95 and fr["tier1_auc"].iloc[-1] < fr["tier1_auc"].iloc[0] - 0.2
    assert fr["t_retention"].is_monotonic_decreasing and fr["gate_agreement"].iloc[-1] < 0.9


# ------------------------------------------------------------------------------------------------ section 65, audit, record
def test_learning_delta_is_reported_per_dimension_and_never_as_one_number(harness):
    d = SY.harness_learning_delta(harness, "B")
    assert d["transfer_delta"].state.value == "UNTESTED" and d["calibration_delta"].state.value == "UNTESTED"
    assert not d.whole_system_claim_allowed
    assert "not proof" in d.statement()
    st = SY.delta_statements(harness)
    assert set(st) == set(CT.LETTERS)
    m = SY.run_metrics(harness.results["fresh_perturbed"].runs["B"][0])
    assert set(m) == {"mean_week", "in_band", "worst5", "max_dd", "pos_share", "dir_hit", "mover_hit"} and m["max_dd"] <= 0
    assert SY.run_metrics(SY.ControlRun("B", 0, np.zeros(0), np.zeros(0), np.zeros(0), 0, SY.GuardRecord("B"))) == {}


def test_disguise_audit_passes_fresh_runs_and_catches_a_reused_identity(world):
    good = SY.presentations(world, 3, 0, SY.PerturbConfig.standard(), "fresh_perturbed")
    assert SY.audit_disguises(good) == []
    kept = SY.presentations(world, 2, 0, None, "kept")
    bad = SY.audit_disguises(kept)
    assert any("share" in b for b in bad) and any("equal their real dates" in b for b in bad)
    assert SY.audit_disguises([]) == []


def test_record_is_json_serialisable_and_stamped(harness):
    import json
    rec = SY.record_of(harness)
    json.dumps(rec)
    assert rec["label"] == "IMPLEMENTED — NOT VALIDATED" and rec["judgement"]["verdict"] == "LEARNING"
    assert set(rec["modes"]) == set(SY.MODES) and len(rec["modes"]["kept"]["C"]["gains"]) == 6
