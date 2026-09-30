"""F26 (C75 Phase 3 3B/3C/3E, Phase 10 loop, Firewalls 5, 6, 10; canon C70-C74): the gate fixes that the F19 real-vs-noise benchmark
drove. Each mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case. Synthetic data only.

  1 leak-shaped features   construction_audit (truncation + future scramble of the candidate's own derivation) and
                           quality_gate.screen_future_dependence (does a value recorded at the decision know how the outcome realises?)
  2 calibration            the claim-shaped (within-date, fixed-margin) forecasts; the damped logistic fit; platt_slope's divergence pinned
  2 replication            power-designed run length, averaged matched control, measured discovery regime
  2 complexity / risk      the worst-fold tolerance in fold units; a magnitude claim bears no directional position risk
  5 retirement             SequentialPlan.retire: repeated FAILED looks retire, one never does
  6 multiplicity           FindingSpec.n_search = the screen's search size
  cost                     vectorised identity-harness scoring and per-date AUC equal the originals"""
from __future__ import annotations

import dataclasses
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning import calibration as CAL                                # noqa: E402
from engine.learning import complexity as CX                                  # noqa: E402
from engine.learning import identity_firewall as IDF                          # noqa: E402
from engine.pattern_movers import auc as rank_auc                             # noqa: E402
from engine.research import evidence as EV                                    # noqa: E402
from engine.research import loop as LP                                        # noqa: E402
from engine.research import pattern_benchmark as PB                           # noqa: E402
from engine.research import quality_gate as QG                                # noqa: E402
from engine.research import replication as RP                                 # noqa: E402
from engine.research import vol_hypotheses as VH                              # noqa: E402
from engine.research import volatility_lab as VL                              # noqa: E402

warnings.filterwarnings("ignore")


def panel(n_dates=40, n_names=30, seed=1, ties=True, nan_share=0.05):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"T{i:02d}" for i in range(n_names)]], names=["date", "ticker"])
    s = pd.Series(np.round(rng.normal(size=len(idx)), 1) if ties else rng.normal(size=len(idx)), index=idx)
    s[rng.random(len(idx)) < nan_share] = np.nan
    s.loc[d[3]] = 1.0                                                           # a date with constant scores
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    y[rng.random(len(idx)) < nan_share] = np.nan
    return s, y


@pytest.fixture(scope="module")
def h1():
    """A planted market where the volatility state (lv20) drives the touch AND the quiet magnitude (a persistent, legitimate predictor
    of both), plus leak columns = noise + g x z(realised magnitude)."""
    F = VL.planted_frame("H1", n_dates=110, n_tickers=48, seed=3)
    rng = np.random.default_rng(9)
    z = (F["absmove"] - F["absmove"].mean()) / F["absmove"].std()
    for g in (0.3, 1.0):
        F[f"bm_leak{int(g * 10)}"] = rng.normal(0, 1, len(F)) + g * z.to_numpy()
    F["bm_noise"] = rng.normal(0, 1, len(F))
    return F


# ============================================================================================================ cost: equivalences
def test_vectorised_identity_scoring_equals_the_harness_originals():
    for seed in (1, 2, 3):
        s, y = panel(seed=seed)
        a, b = IDF.per_date_ic(s, y), EV.fast_per_date_ic(s, y)
        assert a.index.equals(b.index) and np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)
        assert EV.fast_top_k_spread(s, y, 5) == pytest.approx(IDF.top_k_spread(s, y, 5), abs=1e-14)
        assert EV.fast_top_k_spread(s, y, 200) != EV.fast_top_k_spread(s, y, 200)      # no date with > k names: NaN, as the original
    e = pd.Series(dtype=float)
    assert EV.fast_per_date_ic(e, e).empty and IDF.per_date_ic(e, e).empty and math.isnan(EV.fast_top_k_spread(e, e))


def test_fast_multiset_key_keeps_the_equality_relation_and_fixes_signed_zero():
    s, y = panel(seed=4)
    X = pd.DataFrame({"f": s})
    same = (X.sample(frac=1, random_state=1), y.sample(frac=1, random_state=2))
    assert EV.fast_multiset_key(X, y) == EV.fast_multiset_key(*same)
    assert EV.fast_multiset_key(X, y) != EV.fast_multiset_key(X * 1.001, y)          # content changed: must differ
    assert EV.fast_multiset_key(X, y) != EV.fast_multiset_key(X, y.fillna(0.5))
    # values holding both -0.0 and 0.0 (rounded returns do): a pure shuffle must still be 'content preserved' (the original's JSON
    # key can differ here because the sort order of -0.0 and 0.0 is arbitrary - reported to identity_firewall's owner, not asserted)
    Xz = pd.DataFrame({"f": np.r_[np.full(50, -0.0), np.zeros(50), np.arange(20.0)]})
    for k in range(5):
        assert EV.fast_multiset_key(Xz, None) == EV.fast_multiset_key(Xz.sample(frac=1, random_state=k), None)


def test_harness_verdicts_identical_with_fast_scoring():
    s, y = panel(n_dates=60, seed=6, ties=False, nan_share=0.0)
    yy = y + 0.8 * s                                                            # skilled rule
    X = pd.DataFrame({"f": s})

    def learner(Xt, yt, Xe, seed=0):
        return Xe["f"].astype(float)
    tr = X.index.get_level_values(0) < X.index.get_level_values(0).unique()[20]
    run = lambda: IDF.IdentityHarness(learner, attacks=("ticker_permutation", "date_permutation", "stock_substitution"),   # noqa: E731
                                      seed=3, boot=50).run(X[tr], yy[tr], X[~tr], yy[~tr])
    slow = run()
    with EV.fast_identity_scoring():
        fast = run()
    assert IDF.per_date_ic is not EV.fast_per_date_ic                           # restored on exit
    assert [(v.kind, v.mode, v.status) for v in slow.verdicts] == [(v.kind, v.mode, v.status) for v in fast.verdicts]
    assert np.allclose([v.retention for v in slow.verdicts], [v.retention for v in fast.verdicts], equal_nan=True)


def test_per_date_effect_is_exactly_the_old_loop():
    s, y = panel(seed=7)
    lab = (y > 0.4).astype(float).where(y.notna())
    old = {}
    codes, uniq = pd.factorize(s.index.get_level_values(0), sort=True)
    for c in range(len(uniq)):
        m = (codes == c) & np.isfinite(s.to_numpy()) & np.isfinite(lab.to_numpy())
        if m.sum() < 8:
            continue
        L = lab.to_numpy()[m] >= 0.5
        if L.all() or not L.any():
            continue
        old[pd.Timestamp(uniq[c])] = rank_auc(s.to_numpy()[m], L) - 0.5
    new = EV.per_date_effect(s, lab, 8)
    assert new.index.equals(pd.Series(old).sort_index().index) and (new - pd.Series(old)).abs().max() == 0.0
    assert EV.per_date_effect(s.iloc[:0], lab.iloc[:0], 8).empty


# ============================================================================================================ 1 leaks
def test_future_dependence_catches_magnitude_leaks_and_spares_the_volatility_state(h1):
    F = h1.sort_index()
    y = F["touch"].astype(float)
    mag = F["absmove"].astype(float)
    past = EV.past_magnitude(F, mag)
    bar = EV.leak_bar(600, EV.EvidenceConfig())
    D = VH.derive(F, ("lv20", "vol_ratio_short"))
    res = {name: QG.screen_future_dependence(x, y, mag, past, F["vol20"], bar, 6, name)
           for name, x in (("leak3", F["bm_leak3"]), ("leak10", F["bm_leak10"]), ("noise", F["bm_noise"]), ("lv20", D["lv20"]),
                           ("vol_ratio_short", D["vol_ratio_short"]))}
    assert res["leak3"][0] and res["leak10"][0], {k: v[1] for k, v in res.items()}         # planted: caught
    assert res["leak10"][1]["z"] > res["leak3"][1]["z"] > bar
    for k in ("noise", "lv20", "vol_ratio_short"):                              # null and the genuine (persistent) predictors: nothing
        assert not res[k][0], (k, res[k][1])
    assert QG.screen_future_dependence(F["bm_noise"].iloc[:0], y.iloc[:0], mag.iloc[:0], past.iloc[:0])[0] == []


def test_past_magnitude_is_known_before_the_decision(h1):
    F = h1.sort_index()
    pm = EV.past_magnitude(F, F["absmove"])
    lag = F.groupby(level=1)["end"].shift(2)
    ok = pm.notna()
    assert ok.mean() > 0.9 and (pd.to_datetime(lag[ok]) < pd.to_datetime(F.index.get_level_values(0)[ok.to_numpy()])).all()
    assert EV.past_magnitude(F.drop(columns=["end"]), F["absmove"]).isna().all()


def test_leak_bar_is_set_by_the_search_size():
    ec = EV.EvidenceConfig()
    bars = [EV.leak_bar(n, ec) for n in (1, 10, 100, 1000, 100000)]
    assert bars[0] == ec.leak_z_floor and all(a <= b for a, b in zip(bars, bars[1:])) and bars[3] == pytest.approx(4.056, abs=0.01)
    assert EV.EvidenceConfig(leak_z_floor=2.0).validate() and EV.EvidenceConfig(leak_alpha=0.5).validate()


def test_construction_audit_refuses_a_derivation_that_reads_the_future(h1):
    F = h1.sort_index()
    added = {"f26_reads_outcome": (("vol20", "absmove"), lambda G: G["vol20"] * G["absmove"]),
             "f26_centred": (("vol20",), lambda G: G.groupby(level=1)["vol20"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())),
             "f26_clean": (("vol20",), lambda G: np.log(G["vol20"]))}
    VH.DERIVED.update(added)
    try:
        out = {k: EV.construction_audit(F, k, seed=1) for k in added}
    finally:
        for k in added:
            VH.DERIVED.pop(k, None)
    assert any("outcome column" in f for f in out["f26_reads_outcome"]) and any("scrambled" in f for f in out["f26_reads_outcome"])
    assert any("removed" in f for f in out["f26_centred"])                     # truncation: a centred window reads the next row
    assert out["f26_clean"] == [] and EV.construction_audit(F, "lv20") == []
    assert EV.construction_audit(F.iloc[:0], "lv20") == [] and EV.construction_audit(F, "not_a_feature") == []


def test_leak_gate_quarantines_the_planted_leak_only_with_the_fix(h1):
    F = h1
    now = pd.Timestamp(F.index.get_level_values(0).max()) + pd.Timedelta(days=9)
    with PB.registered(["bm_leak3", "bm_noise"]), PB.corpus_once():
        got = {}
        for f in ("bm_leak3", "bm_noise"):
            for fix in (True, False):
                cfg = EV.EvidenceConfig(f26_leak_screen=fix)
                spec = EV.FindingSpec("D" + f, f, 1.0, "VOLATILITY", n_tests_searched=5, seed=1, n_scanned=600)
                b = EV.assemble(F[PB.evidence_columns(f, F.columns)], spec, now, code_hash="c", data_hash="d",
                                created_real="2026-09-30T00:00:00+00:00", cfg=cfg, ledger=RP.ReplicationLedger(), look=1, plan=LP.REGATE_PLAN)
                d = QG.QualityGate(QG.QualityPolicy(code_hash="c")).evaluate(spec.subject_id, b.evidence, now)
                got[(f, fix)] = (d.outcome("leakage").state, b.parts.get("future_probe_caught"))
    assert got[("bm_leak3", True)][0] == QG.QUARANTINE and got[("bm_leak3", False)][0] != QG.QUARANTINE
    assert got[("bm_noise", True)] == (QG.PASS, True)                           # null: clean, and the screen proved it can see


# ============================================================================================================ 2 calibration
def test_recalibration_slope_converges_where_an_undamped_newton_oscillates():
    """A nearly flat log-odds in logit(p) (a weak ranking fed as a rank) is where engine.learning.calibration.platt_slope's full Newton
    steps oscillated to |b| ~ 1e7 (reported to its owner; not asserted here, so its fix cannot break this test). The damped fit and
    recalibration_slope must converge to the maximum-likelihood slope whatever platt_slope does."""
    rng = np.random.default_rng(0)
    r = rng.random(2000).clip(0.01, 0.99)
    y = (rng.random(2000) < 0.25 + 0.2 * (r - 0.5)).astype(float)
    f = QG.logistic_offset_fit(QG.logit(r), y)
    assert f["converged"] and abs(f["b"]) < 1.0
    s = QG.recalibration_slope(r, y)
    assert s["converged"] and s["b"] == pytest.approx(f["b"], abs=1e-6)
    q = QG.expit(f["a"] + f["b"] * QG.logit(r))                               # a stationary point of the likelihood
    assert abs(float(np.sum(q - y))) < 1e-4 * len(y) and abs(float(np.sum((q - y) * QG.logit(r)))) < 1e-4 * len(y)
    if math.isfinite(CAL.platt_slope(r, y)["b"]) and abs(CAL.platt_slope(r, y)["b"] - f["b"]) < 1e-6:
        assert s["b"] == pytest.approx(CAL.platt_slope(r, y)["b"], abs=1e-6)  # when platt_slope converges the two agree
    x = rng.normal(size=3000)
    yy = (rng.random(3000) < QG.expit(-1 + 0.8 * x)).astype(float)
    g = QG.logistic_offset_fit(x, yy)
    assert g["b_lo"] < 0.8 < g["b_hi"] and g["converged"]
    assert not QG.logistic_offset_fit(x[:5], yy[:5])["converged"] and not QG.logistic_offset_fit(x, np.zeros(3000))["converged"]


def test_fixed_margin_forecast_matches_each_dates_count():
    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(30), 40)
    x = rng.random(len(codes)) - 0.5
    k = rng.integers(0, 12, 30).astype(float)
    k[0], k[1] = 0.0, 40.0
    n = np.full(30, 40.0)
    p = EV.fixed_margin_forecast(x, 1.3, codes, k, n)
    ok = ~np.isin(codes, [0, 1])
    assert np.isnan(p[~ok]).all()
    sums = np.bincount(codes[ok], weights=p[ok], minlength=30)
    assert np.allclose(sums[2:], k[2:], atol=1e-8)
    p0 = EV.fixed_margin_forecast(x, 0.0, codes, k, n)
    assert np.allclose(p0[ok], (k / n)[codes[ok]])                             # no lift: every name at the date's rate


def era_frame(lift: float, lift_test: float | None = None, seed: int = 0, n_dates: int = 120, n_names: int = 48):
    """A ranking claim on a market whose base rate moves by era (8% then 30%): the claim is the within-date lift only."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2016-01-01", periods=n_dates, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"N{i}" for i in range(n_names)]])
    x = rng.normal(size=len(idx))
    t = np.repeat(np.arange(n_dates), n_names)
    base = np.where(t < n_dates * 0.4, -2.4, -0.85)
    b = np.where(t < n_dates * 0.3, lift, lift if lift_test is None else lift_test)
    y = (rng.random(len(idx)) < 1 / (1 + np.exp(-(base + b * x)))).astype(float)
    return pd.Series(x, index=idx), pd.Series(y, index=idx), t < n_dates * 0.3, t >= n_dates * 0.35


def test_claim_calibration_passes_a_true_ranking_and_fails_the_wrong_ones():
    pol = QG.QualityPolicy(code_hash="c")
    spec = EV.FindingSpec("D", "lv20", 1.0)
    out = {}
    for name, (lift, lt) in {"true": (0.6, None), "null": (0.0, None), "decayed": (0.9, 0.0)}.items():
        s, y, tr, te = era_frame(lift, lt, seed=4)
        G = pd.DataFrame(index=s.index)
        new, _ = EV.calibration(G, s, y, tr, te, spec, EV.EvidenceConfig())
        old, _ = EV.calibration(G, s, y, tr, te, spec, EV.EvidenceConfig(f26_claim_calibration=False))
        out[name] = (QG.gate_calibration(new, True, pol), QG.gate_calibration(old, True, pol))
    assert out["true"][0].state == QG.PASS, out["true"][0].detail                # the claim, judged as a claim
    assert out["true"][1].state == QG.FAIL                                      # the old base-rate forecast failed the same truth
    assert out["null"][0].state == QG.FAIL and "Brier skill" in out["null"][0].detail
    assert out["decayed"][0].state == QG.FAIL                                   # a lift fitted in train that is gone out of sample
    assert out["true"][0].measures["reference"] == "the reference forecast"


def test_calibration_gate_refuses_malformed_references():
    pol = QG.QualityPolicy(code_hash="c")
    ev = QG.CalibrationEvidence(tuple([0.2] * 300), tuple([0, 1] * 150), 0, tuple([0.2] * 299))
    assert QG.gate_calibration(ev, True, pol).state == QG.FAIL
    assert QG.gate_calibration(QG.CalibrationEvidence(), True, pol).state == QG.MISSING
    assert QG.verify_gate_can_fail()["wrong"] == {} and QG.verify_gate_can_fail()["clean_promoted"]


# ============================================================================================================ 2 replication
def disc(effect, sd=0.13, n=45, searched=600):
    return RP.Discovery("E", effect, sd, n, ("2016-01-01", "2016-12-30"), 5, frozenset({"A"}), (0,), frozenset({"calm"}), "c", "d",
                        "2017-01-10", searched)


def test_run_length_is_a_power_design_fixed_by_the_discovery():
    ec = EV.EvidenceConfig()
    strong, weak = disc(0.20, searched=10), disc(0.03)
    assert EV.run_length(strong, ec) == ec.repl_block                          # a strong effect: quarters are enough
    assert EV.run_length(weak, ec) == ec.repl_max_block                        # indistinguishable from search luck: the longest run
    mid = disc(0.08, searched=10)
    need = RP.required_n(RP.adjusted_effect(mid), mid.sd, RP.DEFAULT_POLICY.alpha, RP.DEFAULT_POLICY.power)
    assert EV.run_length(mid, ec) == min(ec.repl_max_block, max(ec.repl_block, math.ceil(need)))
    assert EV.run_length(mid, ec) == EV.run_length(mid, ec)                    # pure: the same at every look
    assert EV.EvidenceConfig(repl_max_block=5).validate()


def test_averaged_control_keeps_the_null_mean_and_cuts_its_noise():
    s, y = panel(n_dates=60, seed=8, ties=False, nan_share=0.0)
    lab = (y > 0.9).astype(float)
    one = EV.matched_control(s, lab, 8, 5, 1)
    many = EV.matched_control(s, lab, 8, 5, 20)
    assert one.index.equals(many.index) and abs(many.mean()) < 0.02 and many.std() < 0.5 * one.std()
    assert EV.matched_control(s.iloc[:0], lab.iloc[:0], 8, 5, 20).empty


def test_replication_of_a_true_effect_now_has_power(h1):
    """lv20 drives touches in H1; on the stock half the discovery never saw, the designed runs support it and beat the control."""
    F = h1
    now = pd.Timestamp(F.index.get_level_values(0).max()) + pd.Timedelta(days=9)
    spec = EV.FindingSpec("Dlv", "lv20", 1.0, n_tests_searched=5, n_scanned=600, seed=1)
    out = {}
    for fix in (True, False):
        ec = EV.EvidenceConfig(f26_repl_design=fix)
        G, s, y = EV._design(F, spec, ec)
        _, _, tr, te = EV.split_dates(G, ec)
        a, p = EV.replicate(G, s, y, tr, te, spec, ec, now, "c", "d", RP.ReplicationLedger(), LP.REGATE_PLAN.replication_policy(1))
        out[fix] = (a, p)
    a, p = out[True]
    assert a.n_supporting >= 1 and all(r.beats_control for r in a.run_results if r.outcome == RP.Outcome.SUPPORTS)
    assert p["repl_run_length"] >= EV.EvidenceConfig().repl_block
    assert a.n_supporting >= out[False][0].n_supporting
    disc_regimes = RP.ReplicationLedger()
    G, s, y = EV._design(F, spec, EV.EvidenceConfig())
    _, _, tr, te = EV.split_dates(G, EV.EvidenceConfig())
    EV.replicate(G, s, y, tr, te, spec, EV.EvidenceConfig(), now, "c", "d", disc_regimes)
    assert list(disc_regimes.discoveries().values())[0].regimes <= {"calm", "high_vol"}      # measured, never the literal 'train'


# ============================================================================================================ 2 complexity and risk
def test_worst_fold_tolerance_is_restated_in_fold_units():
    idx = pd.date_range("2018-01-05", periods=104, freq="W-FRI")
    folds = pd.Series([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in idx], index=idx)
    cand = CX.Candidate(CX.RuleSpec("c", n_features=1, n_free_params=1), pd.Series(0.01, index=idx), folds)
    cfg = QG.fold_scaled_config(CX.DEFAULT_CCFG, cand)
    sizes = folds.value_counts()
    assert cfg.worst_fold_tolerance == pytest.approx(CX.DEFAULT_CCFG.worst_fold_tolerance * math.sqrt(104 / sizes.median()))
    assert QG.fold_scaled_config(CX.DEFAULT_CCFG, dataclasses.replace(cand, folds=None)) == CX.DEFAULT_CCFG
    rng = np.random.default_rng(3)                                             # a genuine t ~ 4 effect: the scaled test passes it
    good = pd.Series(0.05 + rng.normal(0, 0.13, 104), index=idx)
    base = CX.Candidate(CX.RuleSpec("none", n_features=0, n_free_params=0), pd.Series(rng.normal(0, 0.13, 104), index=idx), folds)
    ev = QG.ComplexityEvidence(dataclasses.replace(cand, oos=good), base, 104.0)
    scaled = QG.gate_complexity(ev, QG.QualityPolicy(code_hash="c"))
    raw = QG.gate_complexity(ev, QG.QualityPolicy(code_hash="c", worst_fold_in_fold_se=False))
    assert scaled.state == QG.PASS and "worst fold" not in scaled.detail
    null = QG.ComplexityEvidence(dataclasses.replace(cand, oos=pd.Series(rng.normal(0, 0.13, 104), index=idx)), base, 104.0)
    assert QG.gate_complexity(null, QG.QualityPolicy(code_hash="c")).state != QG.PASS      # a null rule never earns its place
    assert raw.state in (QG.PASS, QG.FAIL, QG.MISSING)


def test_a_magnitude_claim_bears_no_directional_risk_and_a_direction_claim_does():
    ec = EV.EvidenceConfig()
    vol, dirn = EV.claim_risk(EV.FindingSpec("a", "lv20", problem="VOLATILITY"), ec), EV.claim_risk(EV.FindingSpec("b", "lv20", problem="DIRECTION"), ec)
    assert vol[0] is False and "magnitude" in vol[1] and dirn == (True, "")
    assert EV.claim_risk(EV.FindingSpec("a", "lv20"), EV.EvidenceConfig(f26_claim_risk=False)) == (True, "")
    na = QG.gate_risk(None, False, QG.QualityPolicy(code_hash="c"), vol[1])
    assert na.state == QG.NA and "magnitude" in na.detail
    assert QG.gate_risk(None, True, QG.QualityPolicy(code_hash="c")).state == QG.MISSING         # a directional claim still needs it


# ============================================================================================================ 5 retirement, 6 multiplicity
class _B:
    def __init__(self):
        self.evidence = QG.QualityEvidence()


def test_repeated_failed_looks_retire_and_one_never_does():
    plan = EV.SequentialPlan()
    assert plan.retire(_B(), ["FAILED"]) is None
    assert plan.retire(_B(), ["NEEDS_MORE_EVIDENCE", "FAILED"]) is None
    assert "2 consecutive" in plan.retire(_B(), ["NEEDS_MORE_EVIDENCE", "FAILED", "FAILED"])
    assert plan.retire(_B(), ["FAILED", "FAILED", "NEEDS_MORE_EVIDENCE"]) is None
    assert plan.retire(_B(), ["NEEDS_MORE_EVIDENCE"] * 50) is None             # evidence-limited never counts
    assert plan.retire(_B(), []) is None
    assert EV.SequentialPlan(retire_failed_looks=1).validate()
    assert EV.SequentialPlan(retire_failed_looks=3).retire(_B(), ["FAILED", "FAILED"]) is None


def test_multiplicity_is_the_search_size():
    s = EV.FindingSpec("a", "lv20", n_tests_searched=60, n_scanned=615)
    assert s.n_search == 615 and EV.FindingSpec("a", "lv20", n_tests_searched=60).n_search == 60
    assert EV.FindingSpec("a", "lv20", n_scanned=-1).validate()
    idx = pd.date_range("2019-01-04", periods=30, freq="W-FRI")
    eff = pd.Series(np.full(30, 0.02) + np.random.default_rng(1).normal(0, 0.05, 30), index=idx)
    st, _ = EV.statistical_and_oos(eff.iloc[:5], eff, s, idx[0] - pd.Timedelta(days=14), idx[-1] + pd.Timedelta(days=9))
    assert st.n_tests_searched == 615


def test_benchmark_passes_the_scan_size_and_refuses_unknown_ablations():
    assert PB.BenchConfig(evidence_ablation=("f26_nonsense",)).validate()
    assert PB.BenchConfig(evidence_ablation=("f26_leak_screen", "f26_claim_calibration")).validate() == []
    small = PB.BenchConfig(n_names=20, n_dates=90, frame_weeks=70, first_look=88, look_every=12, bands=(("obvious", 0.9),),
                           noise_counts=(("null", 4), ("leak", 2)), real_kinds=("linear",), oracle_draws=8, mt_pool=4,
                           null_world_share=0.0, tiers=(3,), screen_calls=1)
    w = PB.make_world(4, small)
    ans = PB.run_system(w.frame, "W4", 4, small)
    gated = [g for c in ans["candidates"].values() for g in c["gate"]]
    assert gated and all(g["n_search"] == g["n_scanned"] == len(PB.scan_universe(w.frame.columns)) for g in gated)
