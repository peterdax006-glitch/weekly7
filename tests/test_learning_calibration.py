"""Tests for engine/learning/calibration.py (contract C62 section 54; checklist J08). Synthetic forecasts with a KNOWN true
probability, so every claim about over/under-confidence can be checked against the truth that generated the outcomes."""
import datetime as dt
import math

import numpy as np
import pytest

from engine.learning import calibration as C
from engine.learning.core import FirewallBreach, Health

NOW = dt.date(2027, 1, 1)


def forecasts(n, truth_fn, seed=0, spread=(0.5, 0.95)):
    """Stated probabilities p ~ U(spread), outcomes drawn from the TRUE probability truth_fn(p)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(*spread, n)
    q = truth_fn(p)
    y = (rng.random(n) < q).astype(float)
    return p, y


def fill(mon, p, y, ctx=None, start=dt.date(2024, 1, 1), kid=""):
    for i, (pi, yi) in enumerate(zip(p, y)):
        d = start + dt.timedelta(days=int(i * 1000 / len(p)))
        mon.add(float(pi), int(yi), d, d + dt.timedelta(days=3), NOW, context=(ctx[i] if ctx is not None else ""),
                knowledge_id=kid)


def test_calibrated_forecaster_keeps_full_influence():
    p, y = forecasts(1500, lambda p: p, seed=1)
    mon = C.CalibrationMonitor()
    fill(mon, p, y)
    a = mon.assess(NOW, seed=0)
    assert a.influence == 1.0 and not a.overconfident and a.health == Health.HEALTHY.value
    assert a.ece_p > 0.05 and abs(a.slope - 1.0) < 0.3


def test_planted_overconfidence_is_detected_and_influence_reduced():
    # says 0.5-0.95 but the truth is only half as extreme: 0.5 + 0.5*(p-0.5)
    p, y = forecasts(1500, lambda p: 0.5 + 0.5 * (p - 0.5), seed=2)
    mon = C.CalibrationMonitor()
    fill(mon, p, y)
    a = mon.assess(NOW, seed=0)
    assert a.overconfident and a.health == Health.DEGRADING.value
    assert a.ece_p < 0.01 and a.excess > 0.05
    assert 0.2 <= a.influence < 0.6                                  # edge ratio ~0.5
    assert a.slope_hi < 1.0 and a.slope < 0.75
    adj, f = mon.apply(0.9, NOW, seed=0)
    assert f == a.influence and 0.5 < adj < 0.9 and adj == pytest.approx(0.5 + f * 0.4)


def test_underconfidence_never_earns_extra_influence():
    p, y = forecasts(1500, lambda p: np.clip(0.5 + 1.6 * (p - 0.5), 0, 1), seed=3, spread=(0.5, 0.75))
    mon = C.CalibrationMonitor()
    fill(mon, p, y)
    a = mon.assess(NOW, seed=0)
    assert a.influence == 1.0 and not a.overconfident and a.slope > 1.2
    assert a.excess < 0


def test_influence_factor_rule_directly():
    pol = C.InfluencePolicy()
    assert C.influence_factor(0.10, 0.5, 0.001, 0.8, pol) == pytest.approx(0.5)
    assert C.influence_factor(0.10, 0.5, 0.40, 0.8, pol) == 1.0                     # not significant
    assert C.influence_factor(0.01, 0.5, 0.001, 0.8, pol) == 1.0                    # gap too small to matter
    assert C.influence_factor(-0.05, 1.0, 0.001, 1.2, pol) == 1.0                   # underconfident
    assert C.influence_factor(0.10, 0.5, 0.001, 0.3, pol) == pytest.approx(0.3)     # slope CI tightens it
    assert C.influence_factor(0.20, 0.0, 0.001, 0.0, pol) == pol.floor              # floor, never silence
    assert C.influence_factor(0.10, 0.5, 0.001, 0.8, pol, True) == pytest.approx(0.5 * pol.drift_penalty)
    assert C.influence_factor(float("nan"), 0.5, 0.0, 0.5, pol) == 1.0
    for e in np.linspace(0.03, 0.4, 8):                                             # never above 1, never below floor
        assert pol.floor <= C.influence_factor(e, 0.7, 0.001, 0.9, pol) <= 1.0


def test_reduction_is_monotone_in_overconfidence():
    facs = []
    for k in (0.8, 0.6, 0.4, 0.2):
        p, y = forecasts(1500, lambda p, k=k: 0.5 + k * (p - 0.5), seed=4)
        mon = C.CalibrationMonitor()
        fill(mon, p, y)
        facs.append(mon.assess(NOW, seed=0).influence)
    assert facs == sorted(facs, reverse=True) and facs[0] > facs[-1]


def test_insufficient_and_empty_cases():
    mon = C.CalibrationMonitor()
    a = mon.assess(NOW, seed=0)
    assert a.n == 0 and a.health == Health.INSUFFICIENT_EVIDENCE.value and a.influence == 1.0
    assert mon.apply(0.8, NOW, 0) == (0.8, 1.0)
    p, y = forecasts(30, lambda p: p)
    fill(mon, p, y)
    assert mon.assess(NOW, seed=0).health == Health.INSUFFICIENT_EVIDENCE.value
    assert C.reliability_diagram([], []) == [] and C.brier([], []) != C.brier([], [])   # NaN, not a fake 0
    assert "NOT VALIDATED" in C.report(a)


def test_outcome_not_matured_before_now_fails_closed():
    mon = C.CalibrationMonitor()
    with pytest.raises(FirewallBreach):
        mon.add(0.7, 1, "2026-12-30", NOW, NOW)
    mon.add(0.7, 1, "2026-12-20", "2026-12-31", NOW)
    assert len(mon.records("2026-12-31")) == 0 and len(mon.records(NOW)) == 1


def test_records_after_now_are_ignored_by_assess():
    p, y = forecasts(400, lambda p: p, seed=5)
    mon = C.CalibrationMonitor()
    fill(mon, p, y)
    early = dt.date(2024, 1, 20)
    assert mon.assess(early, seed=0).n < 30 and mon.assess(NOW, seed=0).n == 400


def test_input_validation():
    with pytest.raises(ValueError):
        C.reliability_diagram([1.2], [1])
    with pytest.raises(ValueError):
        C.reliability_diagram([0.5], [2])
    with pytest.raises(ValueError):
        C.reliability_diagram([0.5, 0.6], [1])
    with pytest.raises(ValueError):
        C.reliability_diagram([float("nan")], [1])
    with pytest.raises(ValueError):
        C.CalibrationMonitor().add(1.5, 1, "2024-01-01", "2024-01-05", NOW)
    with pytest.raises(ValueError):
        C.CalibrationMonitor().apply(1.5, NOW, 0)
    with pytest.raises(ValueError):
        C.InfluencePolicy(min_n=5).validate() and C.CalibrationMonitor(C.InfluencePolicy(min_n=5))


def test_reliability_diagram_known_answer():
    p = [0.1] * 10 + [0.9] * 10
    y = [0] * 9 + [1] + [1] * 8 + [0] * 2
    bins = C.reliability_diagram(p, y, 10)
    assert [b.n for b in bins] == [10, 10]
    assert bins[0].freq == pytest.approx(0.1) and bins[1].freq == pytest.approx(0.8)
    assert C.ece(bins) == pytest.approx((0 + 0.1 * 10) / 20)
    assert C.mce(bins, min_n=5) == pytest.approx(0.1)
    assert bins[1].ci_lo < 0.8 < bins[1].ci_hi and bins[1].gap == pytest.approx(0.1)


def test_quantile_bins_have_equal_counts_when_predictions_cluster():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.6, 0.02, 1000), 0, 1)
    y = (rng.random(1000) < p).astype(float)
    bins = C.reliability_diagram(p, y, 10, "quantile")
    assert max(b.n for b in bins) - min(b.n for b in bins) <= 5
    with pytest.raises(ValueError):
        C.reliability_diagram(p, y, 10, "bogus")


def test_wilson_interval_properties():
    lo, hi = C.wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    assert C.wilson(5, 0) == (0.0, 1.0)
    lo, hi = C.wilson(500, 1000)
    assert lo < 0.5 < hi and hi - lo < 0.07


def test_brier_decomposition_and_log_loss():
    p, y = forecasts(4000, lambda p: p, seed=6)
    d = C.brier_decomposition(p, y)
    assert d["reliability"] < 0.002 and d["resolution"] >= 0
    assert abs(d["residual"]) < 0.01
    assert C.log_loss(p, y) < C.log_loss(np.full_like(p, 0.5), y) + 0.05
    assert C.log_loss([1.0], [0.0]) > 10                     # clipped, finite, huge


def test_overconfidence_metrics_known_answer():
    o = C.overconfidence([0.9] * 10, [1] * 6 + [0] * 4)
    assert o["confidence"] == pytest.approx(0.9) and o["accuracy"] == pytest.approx(0.6)
    assert o["edge_ratio"] == pytest.approx(0.1 / 0.4)
    o2 = C.overconfidence([0.1] * 10, [0] * 9 + [1])              # a confident "no" is confident too
    assert o2["confidence"] == pytest.approx(0.9) and o2["accuracy"] == pytest.approx(0.9) and o2["excess"] == pytest.approx(0)
    assert math.isnan(C.overconfidence([], [])["excess"])


def test_platt_slope_recovers_planted_slope():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 6000)
    z = np.log(p / (1 - p))
    y = (rng.random(6000) < 1 / (1 + np.exp(-(0.2 + 0.5 * z)))).astype(float)
    r = C.platt_slope(p, y)
    assert abs(r["b"] - 0.5) < 0.08 and r["b_lo"] < 0.5 < r["b_hi"] and abs(r["a"] - 0.2) < 0.1
    assert math.isnan(C.platt_slope([0.5] * 5, [1, 0, 1, 0, 1])["b"])
    fixed = C.PlattCalibrator().fit(p, y).predict(p)
    assert C.ece(C.reliability_diagram(fixed, y)) < C.ece(C.reliability_diagram(p, y))


def test_isotonic_is_monotone_and_repairs_a_miscalibrated_map():
    p, y = forecasts(3000, lambda p: p ** 2, seed=8, spread=(0.1, 1.0))
    iso = C.IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.1, 1, 50)
    out = iso.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    p2, y2 = forecasts(3000, lambda p: p ** 2, seed=9, spread=(0.1, 1.0))
    assert C.ece(C.reliability_diagram(iso.predict(p2), y2)) < C.ece(C.reliability_diagram(p2, y2)) / 2
    with pytest.raises(RuntimeError):
        C.IsotonicCalibrator().predict([0.5])
    with pytest.raises(ValueError):
        C.IsotonicCalibrator().fit([], [])


def test_ece_null_distribution_separates_noise_from_signal():
    p, y = forecasts(200, lambda p: p, seed=10)
    e = C.ece(C.reliability_diagram(p, y, 10))
    assert C.ece_null_pvalue(p, 10, e, np.random.default_rng(0)) > 0.05          # small sample: raw ECE alone misleads
    py, yy = forecasts(2000, lambda p: 0.5 + 0.4 * (p - 0.5), seed=11)
    ee = C.ece(C.reliability_diagram(py, yy, 10))
    assert C.ece_null_pvalue(py, 10, ee, np.random.default_rng(0)) < 0.01
    a = C.ece_null_pvalue(p, 10, e, np.random.default_rng(3))
    assert a == C.ece_null_pvalue(p, 10, e, np.random.default_rng(3))              # seeded => reproducible


def test_hosmer_lemeshow_flags_miscalibration_only():
    p, y = forecasts(3000, lambda p: p, seed=12)
    _, ok_p, df = C.hosmer_lemeshow(C.reliability_diagram(p, y, 10))
    p2, y2 = forecasts(3000, lambda p: 0.5 + 0.5 * (p - 0.5), seed=12)
    _, bad_p, _ = C.hosmer_lemeshow(C.reliability_diagram(p2, y2, 10))
    assert ok_p > 0.01 and bad_p < 1e-6 and df >= 3
    assert math.isnan(C.hosmer_lemeshow([])[0])


def test_change_scan_finds_when_calibration_flipped():
    rng = np.random.default_rng(13)
    n = 800
    p = rng.uniform(0.5, 0.9, n)
    q = np.where(np.arange(n) < 400, p, 0.5 + 0.3 * (p - 0.5))           # honest for 400 forecasts, then overconfident
    y = (rng.random(n) < q).astype(float)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    d = C.confidence_drift(p, y, dates, window=100, step=50)
    assert d.shift_direction == "overconfident" and d.change_index is not None and 300 <= d.change_index <= 560       # localisation of a weak shift is approximate (se ~ 100 records)
    assert d.change_date == dates[d.change_index].isoformat()
    assert d.gap_slope > 0 and d.gap_slope_p < 0.05 and d.n_windows >= 10
    p0, y0 = forecasts(n, lambda p: p, seed=14)
    quiet = C.confidence_drift(p0, y0, dates, window=100, step=50)
    assert quiet.change_index is None
    assert C.confidence_drift([], [], [], 50, 25).n_windows == 0


def test_drift_uses_matured_order_not_input_order():
    rng = np.random.default_rng(15)
    n = 300
    p = rng.uniform(0.5, 0.9, n)
    y = (rng.random(n) < p).astype(float)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    perm = rng.permutation(n)
    a = C.confidence_drift(p, y, dates, 100, 50)
    b = C.confidence_drift(p[perm], y[perm], [dates[i] for i in perm], 100, 50)
    assert a.ece_by_window == b.ece_by_window


def test_drift_penalty_applies_when_recent_cusum_alarm_is_overconfident():
    rng = np.random.default_rng(16)
    n = 900
    p = rng.uniform(0.5, 0.9, n)
    q = np.where(np.arange(n) < 500, p, 0.5 + 0.2 * (p - 0.5))
    y = (rng.random(n) < q).astype(float)
    mon = C.CalibrationMonitor()
    fill(mon, p, y)
    a = mon.assess(NOW, seed=0)
    assert a.drift is not None and a.drift.shift_direction == "overconfident" and a.overconfident


def test_context_specific_overconfidence_is_localised():
    rng = np.random.default_rng(17)
    n = 3000
    ctx = np.array(["calm", "stress", "earnings"])[rng.integers(0, 3, n)]
    p = rng.uniform(0.55, 0.9, n)
    q = np.where(ctx == "stress", 0.5 + 0.2 * (p - 0.5), p)
    y = (rng.random(n) < q).astype(float)
    mon = C.CalibrationMonitor()
    fill(mon, p, y, ctx=ctx)
    a = mon.assess(NOW, seed=0)
    flagged = {c.context: c for c in a.contexts if c.flagged}
    assert set(flagged) == {"stress"} and flagged["stress"].direction == "overconfident"
    assert flagged["stress"].gap > 0.15 and abs(flagged["stress"].shrunk_gap) < abs(flagged["stress"].gap) + 1e-9
    _, f_stress = mon.apply(0.85, NOW, 0, context="stress")
    _, f_calm = mon.apply(0.85, NOW, 0, context="calm")
    assert f_stress < f_calm <= 1.0 and f_stress >= mon.policy.floor


def test_shrinkage_pulls_small_contexts_toward_global():
    mon = C.CalibrationMonitor(C.InfluencePolicy(min_context_n=30, shrink_k=100.0))
    rng = np.random.default_rng(18)
    p = rng.uniform(0.6, 0.9, 1000)
    y = (rng.random(1000) < p).astype(float)
    ctx = np.array(["a"] * 940 + ["tiny"] * 60)
    y[940:] = 0.0
    fill(mon, p, y, ctx=ctx)
    c = {x.context: x for x in mon.context_report(NOW)}
    assert c["tiny"].gap > 0.6 and c["tiny"].shrunk_gap < 0.5 * c["tiny"].gap


def test_per_knowledge_calibration_is_separate():
    rng = np.random.default_rng(19)
    mon = C.CalibrationMonitor()
    p = rng.uniform(0.55, 0.9, 1200)
    honest = (rng.random(1200) < p).astype(float)
    cocky = (rng.random(1200) < 0.5 + 0.3 * (p - 0.5)).astype(float)
    fill(mon, p, honest, kid="good")
    fill(mon, p, cocky, kid="cocky")
    res = mon.per_knowledge(NOW, seed=0)
    assert res["good"].influence == 1.0 and res["cocky"].influence < 0.7
    assert mon.per_knowledge(NOW, seed=0, min_n=5000) == {}


def test_determinism_and_persistence(tmp_path):
    p, y = forecasts(500, lambda p: 0.5 + 0.6 * (p - 0.5), seed=20)
    m1, m2 = C.CalibrationMonitor(), C.CalibrationMonitor()
    fill(m1, p, y)
    fill(m2, p, y)
    a1, a2 = m1.assess(NOW, seed=5), m2.assess(NOW, seed=5)
    assert (a1.ece, a1.ece_p, a1.influence) == (a2.ece, a2.ece_p, a2.influence)
    path = tmp_path / "cal.jsonl"
    assert m1.dump(path) == 500
    back = C.CalibrationMonitor.load(path)
    assert back.assess(NOW, seed=5).influence == pytest.approx(a1.influence, abs=1e-9)   # json keeps 12 decimals
    txt = C.report(a1)
    assert "reliability" not in txt.lower() or "predicted" in txt
    assert "influence factor" in txt and "NOT VALIDATED" in txt
    empty = tmp_path / "missing.jsonl"
    assert len(C.CalibrationMonitor.load(empty)) == 0


def test_change_scan_false_alarm_rate_is_controlled():
    """The reason the scan replaced a Page CUSUM: on honest forecasts it must almost never fire."""
    rng = np.random.default_rng(21)
    fires = 0
    for _ in range(200):
        p = rng.uniform(0.5, 0.9, 400)
        y = (rng.random(400) < p).astype(float)
        stat, idx = C.change_scan(C._confidence_residual(p, y))
        fires += idx is not None
    assert fires <= 8                                            # ~1% nominal; generous bound for 200 repeats
    assert C.change_scan(np.zeros(5)) == (0.0, None)


def test_wrappers_agree_with_the_existing_calibration_code():
    from engine import pattern_reliability as PR, pattern_stats as PS
    p, y = forecasts(600, lambda p: p, seed=22)
    assert C.brier(p, y) == PR.brier(p, y) and C.ece_equal_mass(p, y, 10) == PR.ece(p, y, 10)
    bins = C.reliability_diagram(p, y, 5)
    tab = PS.calibration_table(p, y.astype(bool), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001))
    tab = tab[tab.n > 0]
    assert [b.n for b in bins] == list(tab.n) and [b.freq for b in bins] == pytest.approx(list(tab.real_share))
    assert math.isnan(C.ece_null_pvalue([], 10, float("nan"), np.random.default_rng(0)))
    assert math.isnan(C.ece_equal_mass(p[:10], y[:10], 10))


# ------------------------------------------------------------------------------------------------- diagnostics and composition

def test_bootstrap_ece_interval_is_ordered_and_seeded():
    p, y = forecasts(800, lambda p: 0.5 + 0.5 * (p - 0.5), seed=41)
    pt, lo, hi = C.bootstrap_ece_ci(p, y, np.random.default_rng(1))
    assert lo <= hi and pt > 0.03
    assert C.bootstrap_ece_ci(p, y, np.random.default_rng(1)) == (pt, lo, hi)
    assert math.isnan(C.bootstrap_ece_ci(p[:10], y[:10], np.random.default_rng(1))[1])


def test_minimum_detectable_gap_shrinks_with_sample_size():
    g100, g1000 = C.minimum_detectable_gap(100), C.minimum_detectable_gap(1000)
    assert g100 > g1000 > 0 and g100 / g1000 == pytest.approx(math.sqrt(10), rel=1e-6)
    assert math.isinf(C.minimum_detectable_gap(1))
    rng = np.random.default_rng(42)                        # planted: a gap 1.3x the detectable size is caught most of the time
    hits = 0
    for _ in range(40):
        p = rng.uniform(0.6, 0.8, 300)
        y = (rng.random(300) < p - C.minimum_detectable_gap(300, 0.7) * 1.3).astype(float)
        hits += C.ece_null_pvalue(p, 10, C.ece_equal_mass(p, y, 10), rng, 100) < 0.05
    assert hits >= 25


def test_sharpness_and_discrimination():
    assert C.sharpness([0.5, 0.5])["mean_abs_dev"] == 0 and C.sharpness([0.5])["entropy_bits"] == pytest.approx(1.0)
    assert C.sharpness([0.99, 0.01])["entropy_bits"] < 0.1 and math.isnan(C.sharpness([])["mean_abs_dev"])
    p, y = forecasts(2000, lambda p: p, seed=43, spread=(0.05, 0.95))
    assert C.discrimination(p, y) > 0.7
    rng = np.random.default_rng(0)
    assert abs(C.discrimination(0.5 + rng.normal(0, 1e-3, 500), (rng.random(500) < 0.5).astype(float)) - 0.5) < 0.1


def test_confidence_bands_expose_the_overconfident_band():
    rng = np.random.default_rng(44)
    p = np.concatenate([np.full(400, 0.65), np.full(400, 0.92)])
    truth = np.concatenate([np.full(400, 0.65), np.full(400, 0.62)])
    y = (rng.random(800) < truth).astype(float)
    rows = C.confidence_bands(p, y)
    hi = [r for r in rows if r["lo"] == 0.9][0]
    lo = [r for r in rows if r["lo"] == 0.6][0]
    assert hi["stated"] == pytest.approx(0.92) and hi["hit_rate"] < 0.7 and hi["ci_hi"] < 0.72
    assert abs(lo["hit_rate"] - 0.65) < 0.08 and C.confidence_bands([], []) == []


def test_recalibrators_are_judged_out_of_sample():
    p1, y1 = forecasts(2500, lambda p: p ** 2, seed=45, spread=(0.1, 1.0))
    p2, y2 = forecasts(2500, lambda p: p ** 2, seed=46, spread=(0.1, 1.0))
    table, best = C.compare_recalibrators(p1, y1, p2, y2)
    assert best in ("platt", "isotonic") and table[best]["brier"] < table["identity"]["brier"]
    assert table["isotonic"]["ece"] < table["identity"]["ece"] / 2
    with pytest.raises(ValueError):
        C.compare_recalibrators(p1[:10], y1[:10], p2, y2)


def test_shrink_confidence_preserves_direction():
    out = C.shrink_confidence(np.array([0.9, 0.5, 0.1]), 0.5)
    assert out.tolist() == pytest.approx([0.7, 0.5, 0.3]) and (out > 0.5).tolist() == [True, False, False]
    assert C.shrink_confidence([0.8], 1.0)[0] == 0.8 and C.shrink_confidence([0.8], 0.0)[0] == 0.5
    with pytest.raises(ValueError):
        C.shrink_confidence([0.8], 1.2)


def test_gap_tracker_alarms_on_overconfidence_and_guards_time():
    rng = np.random.default_rng(47)
    honest, cocky = C.GapTracker(), C.GapTracker()
    for i in range(300):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=i)
        p = rng.uniform(0.6, 0.9)
        honest.update(p, int(rng.random() < p), d, NOW)
        cocky.update(p, int(rng.random() < 0.5 + 0.3 * (p - 0.5)), d, NOW)
    assert cocky.alarmed and not honest.alarmed
    with pytest.raises(FirewallBreach):
        cocky.update(0.7, 1, NOW, NOW)
    with pytest.raises(FirewallBreach):
        cocky.update(0.7, 1, "2025-01-01", NOW)                   # out of matured order
    with pytest.raises(ValueError):
        C.GapTracker(lam=0)
    assert not C.GapTracker().alarmed                              # warm-up: no alarm without evidence


def test_era_report_localises_a_bad_year():
    rng = np.random.default_rng(48)
    mon = C.CalibrationMonitor()
    for year, k in ((2023, 1.0), (2024, 1.0), (2025, 0.2)):
        for i in range(300):
            p = rng.uniform(0.55, 0.9)
            d = dt.date(year, 1, 1) + dt.timedelta(days=i)
            mon.add(p, int(rng.random() < 0.5 + k * (p - 0.5)), d, d + dt.timedelta(days=2), NOW)
    rows = C.era_report(mon, NOW)
    assert [r["year"] for r in rows] == [2023, 2024, 2025]
    assert rows[2]["excess"] > 0.15 > rows[0]["excess"] and C.era_report(C.CalibrationMonitor(), NOW) == []


def test_combined_influence_multiplies_and_names_the_limiting_reason():
    from engine.learning import retirement as R
    led = R.RetirementLedger()
    led.register("k", "2024-01-01")
    led.register("gone", "2024-01-01")
    led.transition("gone", R.State.DORMANT, "2024-06-01", "DORMANT", "x")
    mon = C.CalibrationMonitor()
    p, y = forecasts(600, lambda p: 0.5 + 0.5 * (p - 0.5), seed=49)
    fill(mon, p, y, kid="k")
    fill(mon, p, y, kid="gone")
    b = C.combined_influence("k", NOW, led, mon)
    assert b.lifecycle == 1.0 and b.calibration < 0.7 and b.total == pytest.approx(b.calibration) and b.limiting == "calibration"
    g = C.combined_influence("gone", NOW, led, mon)
    assert g.total == 0.0 and g.limiting == "lifecycle"                # dormant is 0 whatever calibration says
    led.transition("k", R.State.DEGRADED, "2025-01-01", "DEGRADE", "x")
    assert C.combined_influence("k", NOW, led, mon).total < b.total
    assert C.combined_influence("nobody", NOW, led, mon).total == 0.0  # unknown to the ledger => no influence
    assert C.combined_influence("k", NOW, led, None).calibration == 1.0


# ------------------------------------------------------------------------------------------------- S08 second pass (section 83)

def test_context_path_and_hierarchical_shrinkage():
    assert C.context_path("a=1/b=2") == ["a=1", "a=1/b=2"] and C.context_path("") == []
    rng = np.random.default_rng(61)
    n = 2000
    ctx = np.array(["vol=hi/trend=up"] * 20 + ["vol=hi/trend=dn"] * 980 + ["vol=lo/trend=up"] * 1000)
    p = rng.uniform(0.6, 0.9, n)
    q = np.where(ctx == "vol=hi/trend=up", p - 0.25, p)
    y = (rng.random(n) < q).astype(float)
    g = C.hierarchical_gaps(p, y, ctx, k=30)
    leaf = g["vol=hi/trend=up"]
    assert leaf["n"] == 20 and abs(leaf["shrunk"]) < abs(leaf["raw"]) and leaf["shrunk"] > 0
    assert abs(leaf["shrunk"] - leaf["parent"]) < abs(leaf["raw"] - leaf["parent"])          # pulled toward its parent
    assert g["vol=hi"]["n"] == 1000 and set(g) == {"vol=hi", "vol=hi/trend=up", "vol=hi/trend=dn", "vol=lo", "vol=lo/trend=up"}
    assert C.hierarchical_gaps([], [], []) == {}
    with pytest.raises(ValueError):
        C.hierarchical_gaps([0.5], [1], ["a", "b"])


def test_context_platt_shrinks_thin_contexts_to_the_global_curve():
    rng = np.random.default_rng(62)
    p = rng.uniform(0.55, 0.95, 3000)
    ctx = np.array(["big"] * 2900 + ["thin"] * 100)
    q = np.where(ctx == "thin", 0.5 + 0.1 * (p - 0.5), p)
    y = (rng.random(3000) < q).astype(float)
    maps = C.context_platt(p, y, ctx, k=60, min_n=40)
    g = C.platt_slope(p, y)
    thin_b, big_b = maps["thin"][1], maps["big"][1]
    assert thin_b < big_b and thin_b > C.platt_slope(p[ctx == "thin"], y[ctx == "thin"])["b"]     # shrunk, not raw
    assert abs(big_b - g["b"]) < 0.1
    below = C.context_platt(p, y, ["x"] * 2990 + ["tiny"] * 10, min_n=40)
    assert below["tiny"] == pytest.approx((C.platt_slope(p, y)["a"], C.platt_slope(p, y)["b"]))    # under min_n: global
    assert C.context_platt([0.5] * 5, [1, 0, 1, 0, 1], ["a"] * 5) == {}


def _records(p, y, start=dt.date(2024, 1, 1), lag=3):
    mon = C.CalibrationMonitor()
    for i, (pi, yi) in enumerate(zip(p, y)):
        d = start + dt.timedelta(days=i)
        mon.add(float(pi), int(yi), d, d + dt.timedelta(days=lag), NOW)
    return mon


def test_walk_forward_recalibration_uses_only_matured_records():
    p, y = forecasts(500, lambda p: 0.5 + 0.4 * (p - 0.5), seed=63)
    mon = _records(p, y)
    out, used = C.walk_forward_recalibrate(mon.records(NOW), "platt", min_fit=60)
    recs = sorted(mon.records(NOW), key=lambda r: (r.decided_at, r.record_id))
    for i in (0, 100, 300, 499):
        cut = recs[i].decided_at
        assert used[i] == sum(1 for r in recs if r.matured_at < cut)                  # never counts an unmatured outcome
    assert used[0] == 0 and out[0] == recs[0].predicted and used[-1] > 400
    raw = np.array([r.predicted for r in recs])
    yy = np.array([r.outcome for r in recs], float)
    assert C.brier(out[200:], yy[200:]) < C.brier(raw[200:], yy[200:])
    iso, _ = C.walk_forward_recalibrate(mon.records(NOW), "isotonic", min_fit=60)
    assert C.brier(iso[200:], yy[200:]) < C.brier(raw[200:], yy[200:])
    with pytest.raises(ValueError):
        C.walk_forward_recalibrate([], "bogus")
    assert C.walk_forward_recalibrate([], "platt")[0].size == 0


def test_walk_forward_recalibration_cannot_see_a_future_flip():
    """Planted look-ahead trap: outcomes AFTER a forecast are inverted. A past-only fit must be unaffected by them."""
    p, y = forecasts(400, lambda p: p, seed=64)
    y_flipped = y.copy()
    y_flipped[300:] = 1 - y_flipped[300:]
    a, _ = C.walk_forward_recalibrate(_records(p, y).records(NOW), "platt", 60)
    b, _ = C.walk_forward_recalibrate(_records(p, y_flipped).records(NOW), "platt", 60)
    assert np.allclose(a[:290], b[:290])                                              # forecasts before the flip are identical
    assert not np.allclose(a[380:], b[380:])


def test_context_influence_blends_own_and_global_factor():
    rng = np.random.default_rng(65)
    n = 3000
    ctx = np.array(["calm", "stress"])[rng.integers(0, 2, n)]
    p = rng.uniform(0.55, 0.9, n)
    q = np.where(ctx == "stress", 0.5 + 0.2 * (p - 0.5), p)
    y = (rng.random(n) < q).astype(float)
    mon = C.CalibrationMonitor()
    fill(mon, p, y, ctx=ctx)
    out = C.context_influence(mon, NOW, seed=0)
    assert out["stress"]["influence"] < 0.5 and out["calm"]["influence"] > out["stress"]["influence"] + 0.3
    assert out["stress"]["own"] <= out["stress"]["influence"] <= 1.0
    assert C.context_influence(C.CalibrationMonitor(), NOW, 0) == {}
