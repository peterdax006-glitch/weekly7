"""Bible Phase 14 (canon C20): the missed-winner detector. Checks that its weight starts at zero, is earned only by
out-of-sample skill, is ramp-limited and capped; that the four required tests (detector alone, detector + base,
shuffled detector, future-scrambled) and the shuffled-winners control each behave on a planted signal AND on pure
noise (so a check that cannot fail is not what is passing); and the descriptive tools: why a winner was missed,
the missed-minus-picked profile, the ledger."""
import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A
from engine import missed_winners as W
from engine.missed_winners import DET_FEATS, MissedLedger, MissedWinnerDetector

N = 120


def week(seed, signal=0.0, n=N, leak=False):
    """One decision week: rank features, and forward returns. `signal` = how strongly 'e_max20' and 'vol20' drive the
    forward return (0 = pure noise). Returns (date, p0, fwd)."""
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"S{i:03d}" for i in range(n)], name="ticker")
    p = pd.DataFrame({c: rng.normal(size=n) for c in DET_FEATS}, index=idx)
    z = (p["e_max20"].rank(pct=True) - 0.5) + (p["vol20"].rank(pct=True) - 0.5)
    fwd = pd.Series(0.01 + signal * 0.12 * z + rng.normal(0, 0.05, n), index=idx)
    if leak:
        p["lk"] = fwd
    return pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=seed), p, fwd


def weeks(n, signal, seed0=0, **kw):
    return [week(seed0 + i, signal, **kw) for i in range(n)]


def train(det, ws):
    for _, p, f in ws:
        det.learn(p, f)


# ---------------------------------------------------------------- the model and its weight
def test_weight_starts_at_zero_and_stays_zero_until_enough_weeks_are_judged():
    det = MissedWinnerDetector({"det_min_weeks": 6})
    assert det.weight() == 0.0 and det.target_weight() == 0.0
    ws = weeks(6, 1.0)
    for i, (_, p, f) in enumerate(ws):
        det.learn(p, f)
        assert len(det.skill) <= 5 and det.weight() == 0.0           # first week is unjudged: it has nothing to be tested on


def test_planted_signal_is_learned_and_earns_a_positive_capped_weight():
    det = MissedWinnerDetector()
    train(det, weeks(40, 1.0))
    tab = det.coef_table()
    assert set(tab.index[:2]) == {"e_max20", "vol20"} and tab.iloc[0] > 0
    assert det.skill_mean() > 0.15 and 0.0 < det.weight() <= 0.5
    assert det.support()["supported"]


def test_pure_noise_earns_no_weight():
    for seed0 in (0, 1000, 2000):
        det = MissedWinnerDetector()
        train(det, weeks(40, 0.0, seed0=seed0))
        assert abs(det.skill_mean()) < 0.08 and det.weight() < 0.15   # coincidences are shrunk away


def test_hard_cap_holds_even_if_the_config_asks_for_more():
    det = MissedWinnerDetector({"det_max": 5.0, "det_z_span": 0.01})
    train(det, weeks(40, 2.0))
    assert det.target_weight() == pytest.approx(W.DET_HARD_CAP) and det.weight() == pytest.approx(W.DET_HARD_CAP)


def test_weight_rises_at_most_det_step_per_week_but_may_fall_at_once():
    det = MissedWinnerDetector({"det_step": 0.03, "det_z_span": 0.05})
    prev, ups = 0.0, []
    for _, p, f in weeks(40, 2.0):
        det.learn(p, f)
        assert det.weight() - prev <= 0.03 + 1e-12
        ups.append(det.weight() - prev); prev = det.weight()
    assert max(ups) > 0.02                                              # it did ramp
    # the signal disappears: skill turns to noise, the target falls, and the weight follows immediately
    det2 = MissedWinnerDetector({"det_step": 0.03, "det_z_span": 0.05, "det_skill_half_life": 5.0})
    train(det2, weeks(30, 2.0))
    hi = det2.weight()
    assert hi > 0.2
    low = hi
    for _, p, f in weeks(30, -2.0, seed0=500):                          # the relation reverses
        det2.learn(p, f)
        low = min(low, det2.weight())
    assert low < hi / 2                                                 # and the weight fell, in one move, when skill went


def test_skill_is_judged_before_the_week_is_learned():
    """Out-of-sample: the recorded skill equals the correlation of predictions made BEFORE learn() saw that week."""
    det = MissedWinnerDetector()
    train(det, weeks(5, 1.0))
    _, p, f = week(77, 1.0)
    before = det.predict(p)
    det.learn(p, f)
    assert det.skill[-1] == pytest.approx(float(before.rank().corr(f.rank())))
    assert not np.allclose(det.predict(p).values, before.values)         # and it did learn afterwards


def test_a_week_with_too_few_names_is_skipped():
    det = MissedWinnerDetector()
    _, p, f = week(1, 1.0, n=20)
    assert det.learn(p, f) is None and det.n == 0 and det.skill == []


def test_nan_forward_returns_are_dropped_not_learned():
    det = MissedWinnerDetector()
    _, p, f = week(1, 1.0)
    f2 = f.copy(); f2.iloc[:N // 2] = np.nan
    d = det.learn(p, f2)
    assert d["n"] == N - N // 2 and np.isfinite(det.coef).all()


def test_quiet_week_with_no_winner_learns_the_top_five_percent():
    det = MissedWinnerDetector({"det_winner": 10.0})                    # nothing can reach +1000%
    _, p, f = week(1, 1.0)
    d = det.learn(p, f)
    assert d["winners"] >= 1 and np.abs(det.coef).sum() > 0


def test_missing_feature_columns_are_tracked_and_do_not_crash():
    det = MissedWinnerDetector()
    _, p, f = week(1, 1.0)
    det.learn(p.drop(columns=["e_ear", "log_dv"]), f)
    assert det.missing == {"e_ear": 1, "log_dv": 1}


def test_point_in_time_rule_rejects_forward_looking_inputs():
    for bad in ("fwd_ret5", "future_gap", "next_day_open", "label", "target_hit"):
        with pytest.raises(ValueError):
            MissedWinnerDetector(extra_features=[bad])
    ok = MissedWinnerDetector(extra_features=["e_new_signal"])
    assert "e_new_signal" in ok.feats and len(ok.coef) == len(DET_FEATS) + 1


def test_detector_is_deterministic_and_fingerprint_moves_with_state():
    a, b = MissedWinnerDetector(), MissedWinnerDetector()
    ws = weeks(15, 1.0)
    train(a, ws); train(b, ws)
    assert a.fingerprint() == b.fingerprint()
    train(b, weeks(1, 1.0, seed0=99))
    assert a.fingerprint() != b.fingerprint()


def test_adapter_module_reexports_the_same_class():
    assert A.MissedWinnerDetector is MissedWinnerDetector and A.DET_FEATS is DET_FEATS


# ---------------------------------------------------------------- the four tests + the control
def test_walk_forward_honest_finds_the_planted_signal_and_scores_out_of_sample_only():
    r = W.walk_forward(weeks(40, 1.0), seed=0)
    assert len(r) == 39                                                  # the first week has no earlier model to test
    assert r["ic_det"].mean() > 0.15
    assert r["prec_det"].mean() > r["base_rate"].mean() * 1.2           # picks winners more often than chance
    assert (r["weight_used"].iloc[:5] == 0).all()                        # it had not earned anything yet


def test_shuffled_detector_collapses_to_zero_skill_while_honest_does_not():
    ws = weeks(50, 1.0)
    honest = W.walk_forward(ws, seed=1)
    shuf = W.walk_forward(ws, mode="pred_shuffled", seed=1)
    assert honest["ic_det"].mean() > 0.15 and abs(shuf["ic_det"].mean()) < 0.05


def test_label_shuffled_control_learns_nothing_from_the_same_features():
    ws = weeks(50, 1.0)
    real = W.walk_forward(ws, seed=1)
    ctl = W.walk_forward(ws, mode="label_shuffled", seed=1)
    assert real["ic_det"].mean() > 0.15 and abs(ctl["ic_det"].mean()) < 0.06
    assert ctl["weight_used"].mean() < real["weight_used"].mean()


def test_future_scramble_leaves_the_past_untouched_and_kills_later_skill():
    ws = weeks(50, 1.0)
    honest = W.walk_forward(ws, seed=1)
    scr = W.walk_forward(ws, mode="future_scrambled", seed=1, cut=25)
    cut_date = ws[25][0]
    a, b = honest[honest["date"] <= cut_date], scr[scr["date"] <= cut_date]
    assert len(a) == len(b) and np.allclose(a[["ic_det", "weight_used"]].values, b[["ic_det", "weight_used"]].values)
    assert scr[scr["date"] > cut_date]["ic_det"].mean() < honest[honest["date"] > cut_date]["ic_det"].mean() - 0.1


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        W.walk_forward(weeks(3, 1.0), mode="peek")


def test_evaluate_passes_a_real_signal_against_the_shuffled_winner_control():
    res = W.evaluate(weeks(45, 0.4), seed=3, n_controls=8)      # a believable edge: IC ~0.25, under the leak alarm
    assert res["weeks_scored"] == 44
    assert res["alone"]["ic"] > 0.15 and res["alone"]["prec_at_k"] > res["alone"]["base_rate"]
    assert res["shuffled_detector"]["collapsed"]
    assert res["future_scrambled"]["pre_cut_identical"]
    assert res["control"]["q95"] < res["with_base"]["ic_uplift"]
    assert res["verdict"] is True and not res["suspicious"]
    assert "VERDICT" in W.summary_text(res)


def test_evaluate_refuses_to_credit_pure_noise():
    res = W.evaluate(weeks(45, 0.0, seed0=300), seed=3, n_controls=8)
    assert res["verdict"] is False                                       # this is the check that must be able to fail


def test_evaluate_flags_a_leaking_input_as_suspicious_not_as_success():
    """A feature that IS the future return: its name passes the point-in-time rule, its skill is absurd - so the
    evaluation must call it a leak. Without the 'suspicious' guard this would be reported as a triumph."""
    ws = weeks(40, 0.0, leak=True)
    class LeakyDetector(MissedWinnerDetector):
        def __init__(self, meta=None, extra_features=()):
            super().__init__(meta, extra_features=("lk",))
    orig = W.MissedWinnerDetector
    W.MissedWinnerDetector = LeakyDetector
    try:
        res = W.evaluate(ws, None, seed=1, n_controls=4)
    finally:
        W.MissedWinnerDetector = orig
    assert res["alone"]["ic"] > W.SUSPICIOUS_IC and res["suspicious"] and res["verdict"] is False


def test_evaluate_on_no_usable_weeks_reports_it():
    res = W.evaluate([week(1, 1.0, n=10), week(2, 1.0, n=10)])
    assert res["weeks_scored"] == 0 and res["verdict"] is False
    assert "no weeks" in W.summary_text(res)


# ---------------------------------------------------------------- statistics helpers
def test_rank_ic_edges():
    assert W.rank_ic([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]) == 0.0
    assert W.rank_ic([1, 2, 3], [3, 2, 1]) == 0.0                        # fewer than 5 points is not evidence
    assert W.rank_ic(range(10), range(10)) == pytest.approx(1.0)
    assert W.rank_ic(range(10), list(range(10))[::-1]) == pytest.approx(-1.0)


def test_sign_flip_p_and_bootstrap():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.5, 1.0, 60)
    sym = rng.normal(0.0, 1.0, 60)
    assert W.sign_flip_p(pos, seed=1) < 0.01 and W.sign_flip_p(sym, seed=1) > 0.05
    lo, hi = W.boot_ci(pos, seed=1)
    assert lo < pos.mean() < hi and lo > 0
    assert W.sign_flip_p([1.0, 2.0], seed=0) == 1.0 and np.isnan(W.boot_ci([1.0])[0])
    assert W.boot_ci(pos, seed=5) == W.boot_ci(pos, seed=5)              # seeded


# ---------------------------------------------------------------- descriptive side
def test_why_missed_classifies_each_winner():
    idx = ["A", "B", "C", "D", "E"]
    p0 = pd.DataFrame({"log_dv": [20, 19, 18, 17, 16.0], "vol20": [.01, .02, .03, .04, .05]}, index=idx)
    fwd = pd.Series([0.10, 0.09, 0.08, 0.30, -0.02], index=idx)
    score = pd.Series([5, 4, 3, 2, 1.0], index=idx)
    elig = pd.Series([True, True, True, False, True], index=idx)
    r = W.why_missed(p0, fwd, picked=["A"], eligible=elig, score=score).set_index("ticker")
    assert r.loc["A", "why"] == "picked" and r.loc["B", "why"] == "below_cut" and r.loc["D", "why"] == "ineligible"
    assert "E" not in r.index                                            # not a winner
    assert W.why_missed(p0, fwd * 0, picked=[]).empty


def test_winner_type_is_from_information_available_at_the_decision():
    rng = np.random.default_rng(0)
    p0 = pd.DataFrame({"log_dv": rng.normal(18, 1, 50), "vol20": rng.uniform(0.01, 0.05, 50),
                       "e_dist_52wh": rng.uniform(0, 1, 50), "e_ear": 0.0}, index=[f"T{i}" for i in range(50)])
    p0.loc["T0", "log_dv"] = 5.0
    p0.loc["T1", "e_ear"] = 0.99
    p0.loc["T2", ["e_dist_52wh", "log_dv"]] = [1.0, 30.0]
    p0.loc["T3", ["vol20", "log_dv", "e_dist_52wh"]] = [0.5, 30.0, 0.0]
    assert [W.winner_type(p0, t) for t in ("T0", "T1", "T2", "T3", "nope")] == ["illiquid", "event", "momentum", "fast_mover", "other"]


def test_missed_profile_points_at_the_planted_blind_spot():
    n = 100
    idx = [f"T{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    p0 = pd.DataFrame({"vol20": rng.uniform(size=n), "max20": rng.uniform(size=n)}, index=idx)
    picked, missed = idx[:10], idx[10:20]
    p0.loc[missed, "vol20"] += 5                                          # winners we missed were wilder than what we bought
    prof = W.missed_profile(p0, picked, missed, feats=["vol20", "max20"])
    assert prof["vol20"] > 0.5 and abs(prof["max20"]) < 0.5
    assert W.missed_profile(p0, [], missed) == {} and W.missed_profile(p0, picked, []) == {}


def test_ledger_flags_a_systematic_blind_spot_and_not_noise():
    L = MissedLedger()
    rng = np.random.default_rng(1)
    for i in range(30):
        L.add(f"2020-{i:03d}", 10, 6, {"vol20": 0.3 + rng.normal(0, 0.05), "max20": rng.normal(0, 0.2)}, types={"fast_mover": 4, "other": 2},
              era="e1" if i < 15 else "e2")
    s = L.summary().set_index("feature")
    assert bool(s.loc["vol20", "systematic"]) and not bool(s.loc["max20", "systematic"])
    assert L.catch_rate() == pytest.approx(0.4)
    assert L.by_type().iloc[0] == 120 and set(L.by_era().index) == {"e1", "e2"}


def test_empty_ledger_is_safe():
    L = MissedLedger()
    assert L.summary().empty and L.catch_rate() == 0.0 and L.by_type().empty and L.by_era().empty


def test_ledger_needs_min_weeks_before_giving_a_t():
    L = MissedLedger()
    for i in range(3):
        L.add(str(i), 5, 3, {"vol20": 0.5})
    assert np.isnan(L.summary(min_weeks=4)["t"].iloc[0])


# ---------------------------------------------------------------- calibration and breakdowns
def test_calibration_beats_base_rate_only_for_informative_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)                      # p is the true probability
    good = W.calibration(p, y)
    flat = W.calibration(np.full(4000, y.mean()), y)
    assert good["brier_skill"] > 0.05 and abs(flat["brier_skill"]) < 1e-6
    t = good["table"]
    assert np.allclose(t["pred"], t["obs"], atol=0.05)
    assert W.calibration([0.5], [1.0])["table"].empty


def test_prior_correct_restores_the_base_rate():
    rng = np.random.default_rng(1)
    raw = np.clip(rng.beta(4, 4, 5000), 0.01, 0.99)                      # a model trained as if winners were 50% of names
    fixed = W.prior_correct(raw, 0.1)
    assert raw.mean() == pytest.approx(0.5, abs=0.02) and fixed.mean() < 0.2
    assert np.allclose(W.prior_correct(raw, 0.5), raw)                   # no shift when the training rate is the real rate


def test_detector_calibration_on_a_planted_signal_is_informative_after_prior_correction():
    r = W.detector_calibration(weeks(40, 1.0), seed=0)
    assert r["corrected"]["brier_skill"] > r["raw"]["brier_skill"]       # raw probabilities run high, correction fixes it
    assert r["corrected"]["brier_skill"] > 0.02
    assert np.isnan(W.detector_calibration([week(1, 1.0, n=10)])["base_rate"])


def test_topk_curve_precision_beats_base_rate_at_every_k_with_a_signal():
    t = W.topk_curve(weeks(30, 1.0), ks=(5, 20))
    assert list(t["k"]) == [5, 20] and (t["det"] > t["base_rate"] * 1.3).all()
    assert (t["ret_det"] > t["ret_base"]).all()


def test_evaluate_by_group_localises_an_edge_to_the_half_that_has_it():
    ws = weeks(30, 1.0) + weeks(30, 0.0, seed0=900)
    first_half = {w[0] for w in ws[:30]}
    g = W.evaluate_by_group(ws, lambda d: "signal" if d in first_half else "noise").set_index("group")
    assert g.loc["signal", "ic_det"] > 0.1 and abs(g.loc["noise", "ic_det"]) < 0.08
    assert W.evaluate_by_group([week(1, 1.0, n=10)], lambda d: "x").empty


def test_coef_stability_marks_the_planted_features_as_stable_and_noise_as_unstable():
    t = W.coef_stability(weeks(40, 1.0)).set_index("feature")
    assert t.loc["e_max20", "sign_share"] > 0.95 and t.loc["vol20", "sign_share"] > 0.95
    noise = W.coef_stability(weeks(40, 0.0, seed0=50)).set_index("feature")
    assert noise["sign_share"].drop(["e_max20", "vol20"]).mean() < 0.9
    assert W.coef_stability([week(1, 1.0, n=10)]).empty


def test_threshold_sweep_returns_one_row_per_definition_of_a_winner():
    t = W.threshold_sweep(weeks(25, 1.0), thresholds=(0.04, 0.10))
    assert list(t["threshold"]) == [0.04, 0.10] and t.loc[0, "base_rate"] > t.loc[1, "base_rate"]


def test_a_detector_that_only_rediscovers_volatility_is_shown_next_to_the_vol_only_ranking():
    """Winners come from wild stocks but wildness carries no direction: precision@k beats the base rate while rank IC
    stays ~0, and the evaluation must expose that the detector is no better than sorting by vol20."""
    def vol_week(seed, n=N):
        rng = np.random.default_rng(seed)
        idx = pd.Index([f"S{i:03d}" for i in range(n)], name="ticker")
        p = pd.DataFrame({c: rng.normal(size=n) for c in DET_FEATS}, index=idx)
        sd = 0.02 + 0.08 * p["vol20"].rank(pct=True)                    # wild names move more, in either direction
        return pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=seed), p, pd.Series(rng.normal(0, sd), index=idx)
    ws = [vol_week(i) for i in range(45)]
    r = W.walk_forward(ws, seed=0)
    assert r["prec_det"].mean() > r["base_rate"].mean() * 1.5 and abs(r["ic_det"].mean()) < 0.08
    assert r["prec_det"].mean() == pytest.approx(r["prec_vol"].mean(), abs=0.06)     # it learned volatility and nothing more
    res = W.evaluate(ws, None, seed=1, n_controls=4)
    assert res["alone"]["prec_vol_only"] > res["alone"]["base_rate"] and res["verdict"] is False
    assert "vol20-only" in W.summary_text(res)
