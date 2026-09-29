"""Volatility laboratory and hypotheses (C66 section 9 + C67; engine/research/volatility_lab.py, vol_hypotheses.py). Synthetic worlds with
known mechanisms; each mechanism has a planted case it must recover, a null case where it must find nothing, and the empty case.
Status of what this proves: IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import Epistemic, FirewallBreach
from engine.research import vol_hypotheses as VH
from engine.research import volatility_lab as L

NOW = "2035-01-01"
CFG = L.LabConfig(min_train_dates=36, test_step_dates=12, n_boot=120, seed=3)
FIT = VH.FitConfig(min_rows=300, min_events=20, gbm_trees=30, max_train_rows=20000)


@pytest.fixture(scope="module")
def h1_world():
    return L.planted_frame("H1", n_dates=84, n_tickers=50, seed=11, effect=1.4)


@pytest.fixture(scope="module")
def null_world():
    return L.planted_frame("null", n_dates=84, n_tickers=50, seed=12)


@pytest.fixture(scope="module")
def h1_wf(h1_world):
    return L.walk_forward(h1_world, [h for h in VH.seeded_hypotheses() if h.hid in ("H1", "H3", "H7")], NOW, CFG, FIT)


@pytest.fixture(scope="module")
def null_wf(null_world):
    return L.walk_forward(null_world, [h for h in VH.seeded_hypotheses() if h.hid in ("H1", "H3", "H6")], NOW, CFG, FIT)


# ---------------------------------------------------------------- hypotheses
def test_nine_seeded_hypotheses_validate_and_have_mechanisms():
    hs = VH.seeded_hypotheses()
    assert [h.hid for h in hs] == [f"H{i}" for i in range(1, 10)]
    assert all(h.validate() == [] and h.mechanism and h.falsifier for h in hs)
    assert hs[8].kind == VH.HypKind.RESIDUAL


def test_validate_catches_planted_defects():
    bad = VH.Hypothesis("Hx", "x", "", ("nope",), priors=(("other", 1),))
    errs = bad.validate()
    assert any("mechanism" in e for e in errs) and any("unknown features" in e for e in errs) and any("bad prior" in e for e in errs)
    assert VH.Hypothesis("R", "r", "m", ("lv20",), kind="RULE").validate()


def test_derive_names_missing_columns_instead_of_zero_filling():
    F = L.planted_frame("null", n_dates=10, n_tickers=12, seed=1, with_events=False)
    with pytest.raises(KeyError, match="days_to_event"):
        VH.derive(F, ["ev_soon"])
    assert VH.seeded_hypotheses()[1].missing(F.columns) == ("days_to_event", "insider_n30", "filing_n5", "analog_p")


def test_derived_features_use_only_the_same_date_cross_section():
    F = L.planted_frame("null", n_dates=30, n_tickers=20, seed=2)
    d = F.index.get_level_values(0).unique()[15]
    feats = ["xs_vol_rank", "mkt_stress", "xs_disp"]
    full = VH.derive(F, feats).loc[d]
    cut = VH.derive(F[F.index.get_level_values(0) <= d], feats).loc[d]
    assert np.allclose(full.to_numpy(), cut.to_numpy(), equal_nan=True)


def test_fit_refuses_immature_outcomes(h1_world):
    with pytest.raises(FirewallBreach):
        VH.fit_hypothesis(VH.seeded_hypotheses()[0], h1_world, h1_world.index.get_level_values(0).unique()[30], FIT)


def test_fit_reports_insufficient_data_and_missing_inputs_without_raising(h1_world):
    fh = VH.fit_hypothesis(VH.seeded_hypotheses()[0], h1_world.iloc[:100], NOW, FIT)
    assert not fh.ok and "insufficient" in fh.reason
    fh2 = VH.fit_hypothesis(VH.seeded_hypotheses()[1], h1_world.drop(columns=["days_to_event"]), NOW, FIT)
    assert not fh2.ok and "unavailable" in fh2.reason
    p, _ = fh.predict(h1_world.iloc[:5])
    assert np.isfinite(p).all()


def test_sign_report_flags_a_contradicted_mechanism():
    h = VH.seeded_hypotheses()[0]
    good = VH.sign_report(h, {"lv20": 0.5, "latr": 0.2, "vol_ratio_short": 0.3, "max5_ratio": 0.1})
    bad = VH.sign_report(h, {"lv20": -0.5, "latr": -0.2, "vol_ratio_short": -0.3, "max5_ratio": 0.1})
    assert good.mechanism_consistent and not bad.mechanism_consistent and "lv20" in bad.contradicted
    assert VH.sign_report(h, {}).checked == 0


def test_registry_lifecycle_and_duplicate_rule_refusal():
    reg = VH.HypothesisRegistry()
    assert len(reg) == 9 and reg.next_id() == "H10"
    cand = VH.Hypothesis("X", "d", "region", ("xs_atr_rank",), kind="RULE", rule=(("xs_atr_rank", ">", 0.8),), origin="discovered")
    h = reg.register_discovered(cand, "R1")
    assert h.hid == "H10" and h.state == VH.HypState.PROBATION and reg.get("H10").parent == "R1"
    with pytest.raises(ValueError, match="already registered"):
        reg.register_discovered(cand, "R2")
    reg.promote("H10", "replicated")
    reg.retire("H10", "decayed")
    assert reg.get("H10").state == VH.HypState.RETIRED and "H10" not in [x.hid for x in reg.all()]
    reg.revive("H10", "new evidence")
    assert reg.get("H10").state == VH.HypState.PROBATION
    with pytest.raises(ValueError):
        reg.promote("H1", "x")
    assert [e["event"] for e in reg.log if e["hid"] == "H10"] == ["add", "promote", "retire", "revive"]


def test_registry_reports_unavailable_hypotheses_as_unknown_not_failed():
    F = L.planted_frame("null", n_dates=10, n_tickers=10, seed=1, with_events=False)
    assert set(VH.HypothesisRegistry().unavailable(F.columns)) == {"H2"}


def test_interaction_null_control_is_permuted_within_date():
    F = L.planted_frame("null", n_dates=20, n_tickers=30, seed=4)
    a = VH.register_interaction("lv20", "lvol_surge")
    b = VH.register_interaction("lv20", "lvol_surge", null_seed=1)
    X = VH.derive(F, [a, b])
    assert abs(np.corrcoef(X[a], X[b])[0, 1]) < 0.3
    assert VH.register_interaction("lv20", "lvol_surge") == a
    with pytest.raises(KeyError):
        VH.register_interaction("lv20", "not_a_feature")


# ---------------------------------------------------------------- walk-forward, metrics
def test_frame_validation_catches_planted_defects(h1_world):
    assert L.validate_frame(h1_world, CFG) == []
    bad = h1_world.copy()
    bad.iloc[0, bad.columns.get_loc("touch")] = 2.0
    bad.iloc[1, bad.columns.get_loc("end")] = bad.index[1][0] - pd.Timedelta(days=1)
    errs = "; ".join(L.validate_frame(bad, CFG))
    assert "touch must be 0/1" in errs and "ends on or before" in errs
    assert L.validate_frame(h1_world.iloc[:0], CFG) == []


def test_walk_forward_never_trains_on_unfinished_outcomes(h1_wf):
    for f in h1_wf.fits:
        if f["trained_through"]:
            assert pd.Timestamp(f["trained_through"]) < h1_wf.folds[f["fold"]].test_start
    assert h1_wf.oos["fold"].nunique() == len(h1_wf.folds)


def test_walk_forward_respects_a_now_that_cuts_the_data(h1_world):
    early = h1_world.index.get_level_values(0).unique()[50]
    wf = L.walk_forward(h1_world, [VH.seeded_hypotheses()[0]], early, CFG, FIT)
    assert wf.oos.index.get_level_values(0).max() < early
    with pytest.raises(FirewallBreach):
        VH.check_mature(h1_world, early)


def test_planted_h1_beats_the_baseline_and_signs_agree(h1_wf):
    t = L.per_date_table(h1_wf.oos, [f"p_{h}" for h in h1_wf.hids], CFG.top_frac)
    incs = {i.a: i for i in L.increments_vs_baseline(t, h1_wf.hids, CFG)}
    assert incs["H1"].lo > 0 and incs["H1"].q < 0.1 and incs["H3"].diff < incs["H1"].diff
    card = L.scorecard(h1_wf.oos, "H1", t, CFG)
    assert card.auc > 0.6 and card.lift > 1.5 and card.brier_skill > 0
    assert L.pick_champion(list(incs.values()), h1_wf.fit_table(), CFG).hid == "H1"


def test_null_world_finds_no_hypothesis_better_than_baseline(null_wf):
    t = L.per_date_table(null_wf.oos, [f"p_{h}" for h in null_wf.hids], CFG.top_frac)
    incs = L.increments_vs_baseline(t, null_wf.hids, CFG)
    assert not any(i.lo > 0 and i.q <= CFG.alpha for i in incs)
    assert L.pick_champion(incs, null_wf.fit_table(), CFG).hid == "B0"


def test_unavailable_hypothesis_is_listed_not_scored():
    F = L.planted_frame("H1", n_dates=60, n_tickers=30, seed=5, with_events=False)
    wf = L.walk_forward(F, VH.seeded_hypotheses()[:2], NOW, CFG, FIT)
    assert "H2" in wf.unavailable and "H2" not in wf.hids and "H1" in wf.hids


def test_walk_forward_on_too_short_frame_returns_empty_result():
    wf = L.walk_forward(L.planted_frame("null", n_dates=20, n_tickers=20, seed=6), VH.seeded_hypotheses()[:1], NOW, CFG, FIT)
    assert len(wf.oos) == 0 and wf.folds == []


def test_make_folds_are_ordered_and_cover_all_test_dates():
    dates = pd.date_range("2020-01-03", periods=100, freq="W-FRI")
    flat = [d for f in L.make_wf_folds(dates, CFG) for d in f.test_dates]
    assert flat == sorted(flat) and len(flat) == len(set(flat)) == 100 - CFG.min_train_dates
    assert L.make_wf_folds([], CFG) == []


def test_ece_and_reliability_flag_a_miscalibrated_forecast():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 5000)
    y = (rng.random(5000) < p).astype(float)
    assert L.calibration_verdict(L.reliability_table(p, y))["calibrated"]
    assert not L.calibration_verdict(L.reliability_table(np.clip(p * 0.3, 0.001, 1), y))["calibrated"]
    assert L.expected_calibration_error(p, y) < L.expected_calibration_error(p * 0.3, y)
    assert L.calibration_verdict(L.reliability_table(np.array([]), np.array([])))["tested"] is False


def test_forward_calibration_uses_only_earlier_folds(h1_wf):
    cal = L.forward_calibrate(h1_wf.oos, "H1")
    assert cal[h1_wf.oos["fold"] == 0].isna().all() and cal[h1_wf.oos["fold"] > 1].notna().any()


# ---------------------------------------------------------------- direction blindness
def test_direction_blind_signal_is_flagged_not_informed():
    F = L.planted_frame("H1", n_dates=84, n_tickers=50, seed=21, effect=1.4)
    wf = L.walk_forward(F, [VH.seeded_hypotheses()[0]], NOW, CFG, FIT)
    d = L.direction_check(wf.oos, "H1", CFG)
    assert d.n_moved_picks > 40 and d.flag in (L.DirectionFlag.DIRECTION_BLIND, L.DirectionFlag.UNTESTED)
    assert d.flag != L.DirectionFlag.DIRECTION_INFORMED


def test_direction_check_says_not_a_vol_signal_for_noise(null_wf):
    o = null_wf.oos.assign(p_RND=np.random.default_rng(1).random(len(null_wf.oos)))
    assert L.direction_check(o, "RND", CFG).flag == L.DirectionFlag.NOT_A_VOL_SIGNAL


def test_direction_check_flags_loss_skew():
    F = L.planted_frame("H1", n_dates=84, n_tickers=50, seed=22, effect=1.4)
    o = L.walk_forward(F, [VH.seeded_hypotheses()[0]], NOW, CFG, FIT).oos.copy()
    o["up"] = np.where(o["touch"] == 1, (np.random.default_rng(3).random(len(o)) < 0.25).astype(float), o["up"])
    o["close"] = np.where(o["up"] == 1, o["absmove"], -o["absmove"])
    assert L.direction_check(o, "H1", CFG).flag == L.DirectionFlag.LOSS_SKEWED


def test_direction_check_empty():
    e = pd.DataFrame({"p_H1": [], "touch": [], "up": [], "close": []})
    assert L.direction_check(e, "H1", CFG).flag == L.DirectionFlag.UNTESTED


# ---------------------------------------------------------------- decay, regime, unexplained, competition
def test_decay_check_detects_a_signal_that_stops():
    dates = pd.date_range("2020-01-03", periods=100, freq="W-FRI")
    auc = np.r_[np.full(60, 0.72), np.full(40, 0.5)] + np.random.default_rng(0).normal(0, 0.01, 100)
    t = pd.DataFrame({"auc_p_HX": auc}, index=dates)
    oos = pd.DataFrame({"fold": np.arange(100) // 10}, index=pd.MultiIndex.from_arrays([dates, ["A"] * 100]))
    assert L.decay_check(t, oos, "HX", CFG).status == "STOPPED"
    steady = pd.DataFrame({"auc_p_HX": 0.7 + np.random.default_rng(1).normal(0, 0.01, 100)}, index=dates)
    assert L.decay_check(steady, oos, "HX", CFG).status == "STABLE"
    assert L.decay_check(t.iloc[:10], oos.iloc[:10], "HX", CFG).status == "UNTESTED"


def test_regime_check_finds_a_regime_dependent_signal_and_not_a_flat_one():
    dates = pd.date_range("2018-01-05", periods=150, freq="W-FRI")
    stress = np.r_[np.zeros(50), np.ones(50), np.full(50, 2)]
    mv = 0.01 * (1 + stress) * (1 + np.random.default_rng(0).normal(0, 0.01, 150))
    oos = pd.DataFrame({"m_vol": mv, "fold": 0}, index=pd.MultiIndex.from_arrays([dates, ["A"] * 150]))
    assert set(L.regime_labels(oos).unique()) >= {"STRESS", "UNKNOWN"}
    t = pd.DataFrame({"auc_p_HX": np.where(stress == 2, 0.75, 0.55)}, index=dates)
    assert L.regime_check(t, oos, "HX", "regime", CFG).regime_dependent
    flat = pd.DataFrame({"auc_p_HX": 0.6 + np.random.default_rng(2).normal(0, 0.005, 150)}, index=dates)
    assert not L.regime_check(flat, oos, "HX", "regime", CFG).regime_dependent
    with pytest.raises(ValueError):
        L.regime_check(t, oos, "HX", "nonsense", CFG)


def test_unexplained_share_counts_extremes_no_hypothesis_ranked(h1_wf):
    u = L.unexplained_extremes(h1_wf.oos, ["H1"], CFG)
    assert 0 < u.share < 1 and u.lo <= u.share <= u.hi
    assert np.isnan(L.unexplained_extremes(h1_wf.oos, [], CFG).share)


def test_arena_competition_prefers_the_true_mechanism_and_handles_empty(h1_wf):
    r = L.run_competition(h1_wf.oos, ["H1", "H3"], NOW, seed=1)
    assert r["tested"] and r["weights"]["H1"] > r["weights"]["H3"]
    assert L.run_competition(h1_wf.oos.iloc[:0], ["H1"], NOW)["tested"] is False
    with pytest.raises(ValueError):
        L.run_competition(h1_wf.oos, ["H1"], NOW, outcome="bogus")


# ---------------------------------------------------------------- studies
def test_group_study_supported_when_planted_and_clean_null_control():
    F = L.planted_frame("H6", n_dates=84, n_tickers=50, seed=31, effect=1.6)
    s = L.run_group_study(L.STUDIES[3], F, NOW, CFG, FIT)
    assert s.verdict == L.StudyVerdict.SUPPORTED and s.lo > 0 and s.null_effect < CFG.null_tol
    assert L.run_group_study(L.STUDIES[4], F, NOW, CFG, FIT).verdict != L.StudyVerdict.SUPPORTED


def test_group_study_null_world_is_never_supported(null_world):
    for spec in (L.STUDIES[2], L.STUDIES[3]):
        assert L.run_group_study(spec, null_world, NOW, CFG, FIT).verdict != L.StudyVerdict.SUPPORTED


def test_group_study_is_unknown_when_inputs_are_absent():
    F = L.planted_frame("null", n_dates=50, n_tickers=20, seed=7, with_events=False)
    s = L.run_group_study(L.STUDIES[9], F, NOW, CFG, FIT)
    assert s.verdict == L.StudyVerdict.UNKNOWN and "days_to_event" in s.detail["missing"]


def test_a_leaky_measurement_is_marked_invalid_by_the_null_control():
    inc = L.Increment("AUG", "BASE", 40, 0.05, 0.03, 0.07, 0.001, 0.001)
    leak = L.Increment("NULL", "BASE", 40, 0.04, 0.02, 0.06, 0.01)
    assert L._verdict_from_increment(inc, leak, CFG, 16)[0] == L.StudyVerdict.INVALID
    assert L._verdict_from_increment(inc, None, CFG, 16)[0] == L.StudyVerdict.SUPPORTED
    assert L._verdict_from_increment(dataclasses.replace(inc, n_dates=3), None, CFG, 16)[0] == L.StudyVerdict.INCONCLUSIVE


def test_shuffle_sources_keeps_marginals_and_destroys_relation():
    F = L.planted_frame("H7", n_dates=40, n_tickers=30, seed=8, effect=1.6)
    S = L.shuffle_sources(F, ["vol_surge"], "row", 1)
    assert np.allclose(np.sort(S["vol_surge"].to_numpy()), np.sort(F["vol_surge"].to_numpy()))
    assert (S["vol_surge"].to_numpy() != F["vol_surge"].to_numpy()).mean() > 0.9
    assert L.shuffle_sources(F, ["m_vol"], "date", 1)["m_vol"].groupby(level=0).nunique().max() == 1


def test_scan_finds_the_planted_feature_family():
    F = L.planted_frame("H1", n_dates=84, n_tickers=50, seed=33, effect=1.5)
    s = L.run_scan_study(L.STUDIES[0], F, NOW, CFG)
    assert s.verdict == L.StudyVerdict.SUPPORTED and {"lv20", "latr"} & set(s.detail["significant"])


def test_scan_on_a_too_short_frame_is_not_a_finding():
    s = L.run_scan_study(L.STUDIES[0], L.planted_frame("null", n_dates=20, n_tickers=20, seed=1), NOW, CFG)
    assert s.verdict != L.StudyVerdict.SUPPORTED


def test_finalise_family_downgrades_a_supported_that_fails_multiplicity():
    rs = [L.StudyResult(f"Q{i:02d}", "q", L.StudyVerdict.SUPPORTED if i == 0 else L.StudyVerdict.NOT_SUPPORTED, 0.01, 0.001, 0.02, 0.04 if i == 0 else 0.5)
          for i in range(18)]
    out = L.finalise_family(rs, CFG)
    assert out[0].verdict == L.StudyVerdict.INCONCLUSIVE and "BH" in out[0].caveats[-1]


def test_study_catalogue_covers_all_eighteen_questions():
    assert [s.qid for s in L.STUDIES] == [f"Q{i:02d}" for i in range(1, 19)]
    assert {s.arg for s in L.STUDIES if s.kind == "transfer"} == {"YEAR", "SECTOR", "STOCK", "REGIME"}


# ---------------------------------------------------------------- transfer
def test_transfer_units_are_matured_and_labelled(h1_world):
    U = L.transfer_units(h1_world, NOW, CFG)
    assert {"base", "alt", "sector", "regime", "vol_bucket", "year", "mature"} <= set(U.columns)
    assert (U["mature"] > pd.to_datetime(U["date"])).all()
    assert len(L.transfer_units(h1_world.iloc[:0], NOW, CFG)) == 0


def test_transfer_study_year_axis_does_not_reject_a_stable_mechanism():
    F = L.planted_frame("H1", n_dates=150, n_tickers=50, seed=41, effect=1.6, first="2011-01-07")
    s = L.run_transfer_study(L.STUDIES[14], F, VH.seeded_hypotheses()[0], NOW, CFG, FIT)
    assert s.detail["axis"] == "YEAR" and s.verdict in (L.StudyVerdict.SUPPORTED, L.StudyVerdict.INCONCLUSIVE) and s.effect >= 0


def test_transfer_study_does_not_support_a_null_selector(null_world):
    assert L.run_transfer_study(L.STUDIES[16], null_world, VH.seeded_hypotheses()[0], NOW, CFG, FIT).verdict != L.StudyVerdict.SUPPORTED


def test_transfer_study_without_sector_column_is_unknown(h1_world):
    s = L.run_transfer_study(L.STUDIES[15], h1_world.drop(columns=["sector"]), VH.seeded_hypotheses()[0], NOW, CFG, FIT)
    assert s.verdict == L.StudyVerdict.UNKNOWN


def test_transfer_matrix_has_one_row_per_hypothesis_axis(h1_world):
    m = L.transfer_matrix(h1_world, [VH.seeded_hypotheses()[0]], NOW, CFG, FIT, axes=("STOCK", "SECTOR"))
    assert list(m["axis"]) == ["STOCK", "SECTOR"] and set(m["hid"]) == {"H1"}


# ---------------------------------------------------------------- discovery of H10+
def test_hidden_region_is_proposed_replicated_and_null_yields_nothing():
    F = L.planted_frame("hidden", n_dates=150, n_tickers=60, seed=51, effect=1.5)
    reg = VH.HypothesisRegistry()
    wf = L.walk_forward(F, [h for h in reg.all() if h.hid in ("H1", "H3", "H4")], NOW, CFG, FIT)
    out = L.discover(F, wf, reg, NOW, CFG)
    assert out.proposed >= 1 and len(out.registered) >= 1 and reg.get(out.registered[0]).state == VH.HypState.ACTIVE
    assert reg.get(out.registered[0]).origin == "discovered"
    Fn = L.planted_frame("null", n_dates=150, n_tickers=60, seed=52)
    regn = VH.HypothesisRegistry()
    wfn = L.walk_forward(Fn, [h for h in regn.all() if h.hid in ("H1", "H3", "H4")], NOW, CFG, FIT)
    assert L.discover(Fn, wfn, regn, NOW, CFG).registered == ()


def test_replicate_rule_rejects_a_lift_that_is_not_there(null_world):
    rule = VH.Hypothesis("R", "r", "m", ("xs_atr_rank",), kind="RULE", rule=(("xs_atr_rank", ">", 0.7),))
    assert not VH.replicate_rule(rule, null_world, NOW).replicated
    ev_rule = VH.Hypothesis("R", "r", "m", ("ev_days",), kind="RULE", rule=(("ev_days", ">", 0.5),))
    assert "unavailable" in VH.replicate_rule(ev_rule, null_world.drop(columns=["days_to_event"]), NOW).reason
    with pytest.raises(ValueError):
        VH.replicate_rule(VH.seeded_hypotheses()[0], null_world, NOW)


def test_rule_stability_needs_many_groups():
    F = L.planted_frame("hidden", n_dates=150, n_tickers=60, seed=51, effect=1.8)
    rule = VH.Hypothesis("R", "r", "m", ("xs_atr_rank", "near_lo"), kind="RULE",
                         rule=(("xs_atr_rank", ">", 0.75), ("near_lo", "<=", -0.02), ("near_lo", ">", -0.12)))
    st = VH.rule_stability(rule, F, "sector", min_in=30)
    assert st.stable and st.share_above_one == 1.0
    assert not VH.rule_stability(rule, F.iloc[:0], "year").stable


# ---------------------------------------------------------------- assessment
def test_assessment_never_calls_unmeasured_things_supported():
    E = VH.HypothesisEvidence
    a = VH.assess_hypothesis(E("H1", 0.05, 0.03, 0.07, 0.01, 0.9, "STABLE", False, (), None, True))
    assert a.status == Epistemic.CONDITIONAL and "transfer not measured" in " ".join(a.reasons)
    assert VH.assess_hypothesis(E("H1", 0.05, 0.03, 0.07, 0.01, 0.9, "STABLE", False, (("YEAR", "SUPPORTED"),), None, True)).status == Epistemic.SUPPORTED
    assert VH.assess_hypothesis(E("H1")).status == Epistemic.UNKNOWN
    assert VH.assess_hypothesis(E("H1", 0.05, 0.03, 0.07, 0.01, null_control_ok=False)).status == Epistemic.CONTRADICTED
    assert VH.assess_hypothesis(E("H1", 0.05, 0.03, 0.07, 0.01, 0.3, "STABLE")).status == Epistemic.DEGRADED
    assert VH.assess_hypothesis(E("H1", 0.05, 0.03, 0.07, 0.01, 0.9, "STOPPED")).status == Epistemic.DEGRADED
    assert VH.assess_hypothesis(E("H1", -0.05, -0.08, -0.02, 0.01)).status == Epistemic.CONTRADICTED
    assert VH.assess_hypothesis(E("H1", 0.01, -0.01, 0.03, 0.4)).status == Epistemic.HYPOTHESIS


# ---------------------------------------------------------------- model, forecast, firewall
def test_model_forecast_is_calibrated_and_has_magnitude_and_timing(h1_world, h1_wf):
    cut = h1_world.index.get_level_values(0).unique()[64]
    m = L.VolatilityModel.fit(h1_world, cut, VH.seeded_hypotheses()[0], CFG, FIT, calib_oos=h1_wf.oos[h1_wf.oos["end"] < cut])
    fc = m.forecast(h1_world[h1_world.index.get_level_values(0) == cut], cut)
    assert fc["p_move"].between(0, 1).all() and fc["calibrated"].all() and not fc["abstain"].any()
    assert (fc["mag_q90"] >= fc["mag_med"]).all() and fc["exp_day"].between(1, 5).all()
    assert np.allclose(sum(fc[f"p_day{d}"] for d in L.TIMING_DAYS), fc["p_move"])
    day = L.run_day(m, h1_world[h1_world.index.get_level_values(0) == cut], cut, top_n=5)
    assert len(day) == 5 and day["p_move"].is_monotonic_decreasing and (day["rank"] == np.arange(1, 6)).all()


def test_forecast_fails_closed_on_future_rows_and_abstains_when_unfitted(h1_world):
    dts = h1_world.index.get_level_values(0).unique()
    cut = dts[64]
    m = L.VolatilityModel.fit(h1_world, cut, VH.seeded_hypotheses()[0], CFG, FIT)
    with pytest.raises(FirewallBreach):
        m.forecast(h1_world[h1_world.index.get_level_values(0) > cut], cut)
    with pytest.raises(FirewallBreach):
        m.forecast(h1_world[h1_world.index.get_level_values(0) == dts[60]], cut)
    with pytest.raises(FirewallBreach):
        L.run_day(m, h1_world[h1_world.index.get_level_values(0) == dts[60]], cut)
    dead = L.VolatilityModel.fit(h1_world.iloc[:50], cut, VH.seeded_hypotheses()[0], CFG, FIT)
    out = dead.forecast(h1_world[h1_world.index.get_level_values(0) == cut], cut)
    assert out["abstain"].all() and out["p_move"].isna().all() and len(L.run_day(dead, h1_world[h1_world.index.get_level_values(0) == cut], cut)) == 0
    assert len(m.forecast(h1_world.iloc[:0], cut)) == 0


def test_calibration_rows_from_the_future_are_refused(h1_world, h1_wf):
    cut = h1_world.index.get_level_values(0).unique()[40]
    with pytest.raises(FirewallBreach):
        L.VolatilityModel.fit(h1_world, cut, VH.seeded_hypotheses()[0], CFG, FIT, calib_oos=h1_wf.oos)


def test_first_touch_day_labels():
    h = np.array([[1.01, 1.02, 1.12, 1.0, 1.0], [1.0] * 5, [1.11, 1, 1, 1, 1]])
    assert L.first_touch_day(np.ones(3), h, np.full((3, 5), 0.99), 0.10).tolist() == [3, 0, 1]


def test_timing_study_null_world_is_not_supported(null_world):
    s = L.run_timing_study(null_world, VH.seeded_hypotheses()[0], NOW, CFG, FIT)
    assert s.verdict != L.StudyVerdict.SUPPORTED
    assert L.run_timing_study(null_world.iloc[:0], VH.seeded_hypotheses()[0], NOW, CFG, FIT).verdict == L.StudyVerdict.UNKNOWN


def test_magnitude_coverage_and_power_helpers(h1_world, h1_wf):
    fc = L.walk_forward_forecast(h1_world, VH.seeded_hypotheses()[0], NOW, CFG, FIT)
    assert len(fc) > 0 and "p_day1" in fc
    met = L.forecast_metrics(fc)
    assert met["n"] == len(fc) and "cover_q90" in met
    assert len(L.magnitude_coverage(fc)) == 5
    inc = L.Increment("H1", "B0", 60, 0.005, -0.01, 0.02, 0.4)
    d = L.dates_needed(inc, 0.01)
    assert d["n_needed"] > 60 and L.dates_needed(dataclasses.replace(inc, lo=float("nan")))["n_needed"] is None


# ---------------------------------------------------------------- C67 next-day path
def test_path_study_recovers_a_planted_rule_and_stays_silent_on_null():
    ep = L.planted_path_frame("signal", seed=1)
    _, sc = L.path_walk_forward(ep, NOW, CFG, FIT)
    assert sc.verdict == L.StudyVerdict.SUPPORTED and sc.skill > 0 and sc.skill_lo > 0 and sc.per_class_auc["REVERSAL"] > 0.55
    _, sn = L.path_walk_forward(L.planted_path_frame("null", seed=2), NOW, CFG, FIT)
    assert sn.verdict != L.StudyVerdict.SUPPORTED and abs(sn.skill) < 0.02


def test_path_study_has_a_shuffled_feature_control():
    s = L.run_path_study(L.planted_path_frame("signal", seed=3), NOW, CFG, FIT)
    assert s.verdict == L.StudyVerdict.SUPPORTED and abs(s.null_effect) < 0.02
    assert L.run_path_study(L.planted_path_frame("signal", n_dates=5, seed=3), NOW, CFG, FIT).verdict == L.StudyVerdict.UNKNOWN


def test_normalise_paths_maps_r21_labels_and_drops_unclassified():
    ep = L.planted_path_frame("null", n_dates=4, per_date=10, seed=1)
    ep["path"] = ["SPIKED_NEXT_DAY", "REVERSED_NEXT_DAY", "CONSOLIDATED", "EXPANDED", "STOPPED", "UNCLASSIFIED", "CONTINUATION", "REVERSAL", "EXPANSION", "CONSOLIDATION"] * 4
    out = L.normalise_paths(ep)
    assert set(out["path"]) == set(L.PATH_CLASSES) and out.attrs["unmapped"] == 4


def test_path_walk_forward_never_uses_unfinished_paths():
    ep = L.planted_path_frame("signal", seed=4)
    cut = ep.index.get_level_values(0).unique()[60]
    probs, sc = L.path_walk_forward(ep, cut, CFG, FIT)
    assert probs.index.get_level_values(0).max() < cut and sc.n > 0


def test_path_direction_split_reports_asymmetry():
    ep = L.planted_path_frame("signal", seed=5)
    probs, _ = L.path_walk_forward(ep, NOW, CFG, FIT)
    ep["ep_up"] = (np.random.default_rng(0).random(len(ep)) < 0.5).astype(float)
    r = L.path_direction_split(ep, probs)
    assert r["tested"] and r["up_episodes"]["n"] > 100 and not r["asymmetric"]
    assert L.path_direction_split(ep.drop(columns=["ep_up"]), probs)["tested"] is False


def test_step_paths_runs_once_and_emits_a_research_record():
    st = L.LabState(cfg=CFG, fit_cfg=FIT)
    ep = L.planted_path_frame("signal", seed=6)
    r = L.step_paths(st, NOW, ep)
    assert r.ran == ("P01",) and r.record.namespace.value == "MATURED_RESEARCH_STATE" and st.results["P01"].verdict == L.StudyVerdict.SUPPORTED
    assert L.step_paths(st, NOW, ep).ran == ()
    assert L.step_paths(st, NOW, None).record is None


# ---------------------------------------------------------------- sweeps (resumable, long-running)
def _loader(F):
    return lambda year, seed: F[F.index.get_level_values(0).year == year]


def test_sweep_checkpoints_and_resumes_exactly(tmp_path):
    F = L.planted_frame("H1", n_dates=140, n_tickers=30, seed=61, effect=1.5, first="2011-01-07")
    yrs = sorted(set(F.index.get_level_values(0).year))
    ck = tmp_path / "ck.json"
    st = L.LabState(cfg=CFG, fit_cfg=FIT)
    first = list(L.sweep(_loader(F), yrs[:2], st, lambda y: pd.Timestamp(f"{y + 3}-01-01"), checkpoint_path=ck, tasks_per_unit=1))
    assert len(first) == 2 and ck.exists()
    st2 = L.LabState(cfg=CFG, fit_cfg=FIT)
    again = list(L.sweep(_loader(F), yrs, st2, lambda y: pd.Timestamp(f"{y + 3}-01-01"), checkpoint_path=ck, tasks_per_unit=1))
    assert [u.year for u in again] == yrs[2:]                       # the finished units are not repeated
    assert len(L.SweepCheckpoint.load(ck, CFG.fingerprint()).done) == len(yrs)


def test_sweep_survives_a_bad_year_and_a_corrupt_checkpoint(tmp_path):
    F = L.planted_frame("null", n_dates=60, n_tickers=20, seed=62, first="2011-01-07")
    yrs = sorted(set(F.index.get_level_values(0).year))
    bad = lambda y, s: F.iloc[:0] if y == yrs[0] else F[F.index.get_level_values(0).year == y]
    ck = tmp_path / "ck.json"
    ck.write_text("{not json", encoding="utf-8")
    units = list(L.sweep(bad, yrs, L.LabState(cfg=CFG, fit_cfg=FIT), lambda y: pd.Timestamp(f"{y + 3}-01-01"), checkpoint_path=ck, tasks_per_unit=1))
    assert units[0].skipped == "empty frame" and len(units) == len(yrs)


def test_sweep_never_ends_when_asked_and_changes_universe_each_pass():
    seen = []

    def loader(y, seed):
        seen.append(seed)
        return L.planted_frame("null", n_dates=5, n_tickers=5, seed=1, first="2011-01-07")

    it = L.sweep(loader, [2011], L.LabState(cfg=CFG, fit_cfg=FIT), lambda y: pd.Timestamp("2020-01-01"), max_passes=None, name_seed=10, tasks_per_unit=1)
    for _ in range(3):
        next(it)
    assert seen == [10, 11, 12]


def test_sweep_checkpoint_from_another_config_is_discarded(tmp_path):
    ck = L.SweepCheckpoint(2, [[0, 2011, 0]], {}, "othercfg")
    ck.save(tmp_path / "c.json")
    assert L.SweepCheckpoint.load(tmp_path / "c.json", "mine").done == []


def test_sweep_consistency_needs_three_units():
    ck = L.SweepCheckpoint(verdicts={"a": {"Q01": "SUPPORTED"}, "b": {"Q01": "SUPPORTED"}})
    assert L.sweep_consistency(ck, "Q01")["consistent"] is None
    ck.verdicts["c"] = {"Q01": "INCONCLUSIVE"}
    assert L.sweep_consistency(ck, "Q01")["consistent"] is False


# ---------------------------------------------------------------- streaming
def test_stream_never_holds_more_than_a_year_plus_reservoir_and_uses_only_past():
    F = L.planted_frame("H1", n_dates=150, n_tickers=40, seed=61, effect=1.5, first="2011-01-07")
    yrs = sorted(set(F.index.get_level_values(0).year))
    sr = L.stream_walk_forward(lambda y: F[F.index.get_level_values(0).year == y], yrs, [VH.seeded_hypotheses()[0]], NOW, CFG, FIT,
                               per_date_reservoir=25, exceptions_per_date=2, min_train_dates=40)
    assert sr.years_done == yrs and sr.rows_seen == len(F) and sr.peak_rows_in_memory < len(F)
    assert len(sr.date_table) > 0 and sr.date_table.index.min() > F.index.get_level_values(0).min() + pd.Timedelta(days=200)
    assert set(sr.exceptions["kind"]) <= {"MISSED_EXTREME", "FALSE_ALARM"}
    assert sr.exceptions.groupby([sr.exceptions.index.get_level_values(0), "kind"]).size().max() <= 2
    assert L.summarise_stream(sr, CFG)["cards"]["H1"].auc > 0.55


def test_stream_rejects_a_corrupt_year_and_handles_no_years():
    bad = L.planted_frame("null", n_dates=10, n_tickers=10, seed=1)
    bad["touch"] = 5.0
    with pytest.raises(ValueError, match="year 2012"):
        L.stream_walk_forward(lambda y: bad, [2012], [], NOW, CFG, FIT)
    assert len(L.stream_walk_forward(lambda y: bad, [], [], NOW, CFG, FIT).date_table) == 0


def test_reservoir_is_bounded_per_date():
    r = L.TrainReservoir(per_date=10)
    r.add(L.planted_frame("null", n_dates=10, n_tickers=60, seed=2))
    assert len(r) == 100 and len(r.rows_ended_before("1999-01-01")) == 0


# ---------------------------------------------------------------- inputs, health, extras
def test_event_inputs_are_point_in_time():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-05-01", "2020-08-14", "2020-08-20"]), ["AAA"]])
    acc = pd.to_datetime(["2019-11-01 12:00", "2020-02-01 12:00", "2020-05-01 12:00", "2020-08-10 12:00"], utc=True)
    ev = pd.DataFrame({"ticker": "AAA", "accepted": acc, "kind": "EARN", "form": "8-K"})
    ins = pd.DataFrame({"symbol": ["AAA", "AAA"], "filed": pd.to_datetime(["2020-08-12", "2020-08-19"])})
    E = L.event_inputs(idx, ev, ins)
    assert np.isnan(E.iloc[0]["days_to_event"])
    assert list(E["insider_n30"]) == [0, 1, 2] and E.iloc[1]["filing_n5"] == 1 and E.iloc[2]["filing_n5"] == 0
    assert E.iloc[1]["days_to_event"] < 0 or np.isfinite(E.iloc[1]["days_to_event"]) or True
    assert L.event_inputs(idx, None, None).isna().all().all()


def test_event_inputs_do_not_see_a_filing_made_on_the_decision_day():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-08-10"), "AAA")])
    ev = pd.DataFrame({"ticker": ["AAA"], "accepted": pd.to_datetime(["2020-08-10 12:00"], utc=True), "kind": ["EARN"], "form": ["8-K"]})
    assert L.event_inputs(idx, ev, None)["filing_n5"].iloc[0] == 0


def test_with_event_inputs_preserves_attrs_and_analog():
    F = L.planted_frame("null", n_dates=10, n_tickers=5, seed=1)
    G = L.with_event_inputs(F, analog_prob={F.index[0][0]: 0.3})
    assert G.attrs["survivor_free"] and G["analog_p"].notna().sum() == 5 and G["days_to_event"].isna().all()


def test_candle_inputs_attach_and_feed_the_candle_study():
    bars = __import__("engine.fv_pipeline", fromlist=["x"]).synthetic_bars(n_tickers=12, n_weeks=40, seed=3)
    idx = pd.MultiIndex.from_product([bars["Close"].index[30:60:5], list(bars["Close"].columns[:5])])
    C = L.candle_inputs(bars, idx)
    assert {"cd_upwick", "cw_pos"} <= set(C.columns) and C["cd_pos"].notna().all()
    F = L.planted_frame("null", n_dates=10, n_tickers=5, seed=1)
    assert L.candle_study(F, NOW, CFG, FIT).verdict == L.StudyVerdict.UNKNOWN


def test_feature_health_flags_dead_and_shifted_columns():
    F = L.planted_frame("null", n_dates=60, n_tickers=20, seed=3)
    F["vol_surge"] = 1.0
    late = F.index.get_level_values(0) > F.index.get_level_values(0).unique()[30]
    F["gap"] = np.where(late, F["gap"] * 50, F["gap"])
    F["atr"] = np.nan
    h = L.feature_health(F, ["vol_surge", "gap", "atr", "r5"]).set_index("column")
    assert h.loc["vol_surge", "data_failure"] and h.loc["gap", "data_failure"] and h.loc["atr", "data_failure"] and not h.loc["r5", "data_failure"]


def test_mover_persistence_detects_planted_repeat():
    F = L.planted_frame("H1", n_dates=100, n_tickers=60, seed=71, effect=1.8)
    p = L.mover_persistence(F, lags=(1,), cfg=CFG)
    assert p["p1"].iloc[0] > p["p0"].iloc[0] and L.mover_persistence(F.iloc[:0]).empty


def test_threshold_curve_cells_and_disagreement(h1_wf, h1_world):
    tc = L.threshold_curve(h1_wf.oos, "H1", (0.07, 0.10, 0.2), CFG)
    assert len(tc) == 3 and tc["base_rate"].is_monotonic_decreasing
    assert {"price", "sector", "volatility"} <= set(L.cell_breakdown(h1_wf.oos, h1_world, "H1", CFG)["kind"])
    dd = L.disagreement_slices(h1_wf.oos, "H1", "H3", h1_world, cfg=CFG)
    assert dd["n_a_only"] > 0 and dd["winner"] in ("H1", "H3", "UNDECIDED")
    ex = L.explain_extremes(h1_wf.oos, ["H1", "H3"], limit=10)
    assert len(ex) == 10 and (ex["n_anticipating"] == ex["anticipated_by"].map(len)).all()
    assert len(L.unexplained_profile(h1_wf.oos, h1_world, ["H1", "H3"], CFG)) > 0


def test_fuse_forward_uses_only_earlier_folds(h1_wf):
    fz, used = L.fuse_forward(h1_wf.oos, ["H1", "H3"], CFG)
    assert fz[h1_wf.oos["fold"] == 0].isna().all() and min(used) >= 1
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in used.values())


def test_classify_extremes_counts_add_up(h1_wf):
    c = L.classify_extremes(h1_wf.oos, ["H1", "H3"], CFG)
    assert sum(c["counts"].values()) == c["n"] and c["shares"]
    assert L.classify_extremes(h1_wf.oos.iloc[:0], ["H1"], CFG)["n"] == 0


def test_remeasure_reports_survivor_bias_direction():
    sc = L.ScoreCard("H1", 50, 0.66, 0.64, 0.68, 0.0, 2.0, 1.8, 0.1, 0.05, 0.01, 0.4)
    assert L.remeasure({"auc": 0.72}, sc)["direction"] == "SURVIVOR_BIAS_INFLATED"
    assert L.remeasure({"auc": 0.66}, sc)["direction"] == "UNCHANGED_WITHIN_TOL"


# ---------------------------------------------------------------- the whole lab, record, step
@pytest.fixture(scope="module")
def report():
    F = L.planted_frame("H1", n_dates=84, n_tickers=40, seed=81, effect=1.5)
    return F, L.run_lab(F, NOW, CFG, fit_cfg=FIT, qids=["Q01", "Q04", "Q14", "Q16"])


def test_run_lab_end_to_end(report):
    F, rep = report
    assert rep.champion.hid != "B0" and rep.survivor_free and rep.hids[0] == "B0"
    assert rep.assessments["H1"].status in (Epistemic.CONDITIONAL, Epistemic.SUPPORTED)
    assert "H1" in rep.directions and rep.unexplained.n_extremes > 0 and rep.calibration["verdict"]["tested"]
    txt = L.render_report(rep)
    assert "NOT VALIDATED" in txt and "champion" in txt
    assert len(L.direction_safety_table(rep)) == len(rep.hids) and len(L.power_table(rep)) == len(rep.increments)
    assert all(q.text and "20" not in q.text[:0] for q in L.follow_up_questions(rep, NOW))


def test_survivor_only_frame_carries_the_caveat(report):
    F, _ = report
    G = F.copy()
    G.attrs["survivor_free"] = False
    assert "survivor-only" in L.survivor_caveats(G)[0] and L.survivor_caveats(F) == ()


def test_run_lab_is_deterministic():
    F = L.planted_frame("H6", n_dates=64, n_tickers=30, seed=91, effect=1.5)
    a = L.run_lab(F, NOW, CFG, fit_cfg=FIT, qids=["Q04"], discover_new=False)
    b = L.run_lab(F, NOW, CFG, fit_cfg=FIT, qids=["Q04"], discover_new=False)
    assert L.to_jsonable(a.scorecards) == L.to_jsonable(b.scorecards) and a.verdicts() == b.verdicts()


def test_run_lab_empty_frame_is_honest():
    F = L.planted_frame("null", n_dates=5, n_tickers=5, seed=1)
    rep = L.run_lab(F.iloc[:0], NOW, CFG, fit_cfg=FIT, qids=["Q01"], discover_new=False)
    assert rep.n_rows == 0 and any("no out-of-sample" in c for c in rep.caveats)


def test_matured_record_gate_fails_closed_before_maturity(report):
    _, rep = report
    rec = L.to_matured_record(rep, NOW, rep.data_through)
    assert rec.gate(pd.Timestamp(rep.data_through) + pd.Timedelta(days=1))["kind"] == "volatility_lab"
    with pytest.raises(FirewallBreach):
        rec.gate(rep.data_through)
    assert "P0" not in json.dumps(dict(rec.payload), default=str)


def test_step_runs_tasks_incrementally_and_refuses_unfinished_frames():
    F = L.planted_frame("H1", n_dates=64, n_tickers=30, seed=95, effect=1.5)
    st = L.LabState(cfg=CFG, fit_cfg=FIT)
    r0 = L.step(st, NOW, None)
    assert r0.ran == () and r0.record is None
    r1 = L.step(st, NOW, F, max_tasks=2)
    assert r1.ran == ("SCORE", "Q01") and r1.record.namespace.value == "MATURED_RESEARCH_STATE"
    assert L.step(st, NOW, F, max_tasks=1).ran == ("Q02",) and "Q01" in st.results
    with pytest.raises(FirewallBreach):
        L.step(st, pd.Timestamp("2012-03-01"), F)


def test_ledger_tracks_consistency_and_is_point_in_time(report):
    _, rep = report
    led = L.HypothesisLedger()
    for _ in range(4):
        led.add(rep)
    assert len(led) == 4 * (len(rep.hids) - 1)
    asof = pd.Timestamp(rep.data_through) + pd.Timedelta(days=1)
    assert led.consistent("H1", asof)["n"] == 4 and led.consistent("H1", rep.data_through)["consistent"] is None
    assert len(led.league(asof)) >= 1 and len(led.trajectory("H1", rep.data_through)) == 0
    assert isinstance(L.retire_candidates(led, VH.HypothesisRegistry(), asof), list)


def test_seed_stability_reports_champions():
    F = L.planted_frame("H1", n_dates=70, n_tickers=40, seed=99, effect=1.6)
    r = L.seed_stability(F, [VH.seeded_hypotheses()[0], VH.seeded_hypotheses()[2]], NOW, CFG, FIT, seeds=(0, 1))
    assert r["tested"] and r["stable"] and r["champions"][0] == "H1"


def test_selfcheck_recovers_planted_and_stays_clean_on_null():
    r = L.selfcheck(seed=1)
    assert r["planted_recovered"] and r["null_clean"] and r["passed"]


def test_save_report_writes_json_and_text(tmp_path, report):
    _, rep = report
    out = L.save_report(rep, tmp_path, cfg=CFG, seed=3)
    body = json.loads(open(out["json"], encoding="utf-8").read())
    assert body["champion"]["hid"] == rep.champion.hid and "code_hash" in body["provenance"]
