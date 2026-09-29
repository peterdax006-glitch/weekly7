"""Bible Phase 3.3 / 3.4 (canons C35, C37): the statistical validation library.

Three kinds of check, because a check that cannot fail is worthless:
  * equivalence  - engine.pattern_stats reproduces engine.patterns.PatternMiner on the same inputs;
  * planted truth - a known effect is found and calibrated, a known defect (overlap, look-ahead) is caught;
  * degenerate    - empty masks, too few clusters, zero weights, empty families.
Synthetic data only; the whole file runs in well under a minute."""
import math

import numpy as np
import pandas as pd
import pytest

from engine import pattern_stats as S
from engine.patterns import PatternMiner, _t_to_p, _wstats

FEATS = [f"f{i}" for i in range(6)]
CTX = ["m_vix", "m_vix_term", "m_spy_ma200", "m_breadth", "m_dispersion"]
MINER_PARAMS = {"max_pairs": 260, "max_unless": 40, "null_reps": 1, "min_n": 120, "max_rows": 10 ** 9, "p_method": "bonferroni",
                "hac_lags": 0}          # the pre-Phase-3 behaviour, so equivalence with the library is exact


def make_panel(n_dates=240, n_tick=50, seed=0, effect=0.012):
    """Rows (date, ticker). y = market shock of the date + planted effect for top-quintile f0 + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(n_tick)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), len(FEATS))), index=idx, columns=FEATS)
    ctx = pd.DataFrame(rng.normal(size=(n_dates, len(CTX))), index=dates, columns=CTX)
    for c in CTX:
        X[c] = ctx[c].reindex(dates).repeat(n_tick).values
    top = X["f0"].groupby(level=0).rank(pct=True).values >= 0.8
    shock = pd.Series(rng.normal(0, 0.02, n_dates), index=dates).reindex(idx.get_level_values(0)).values
    y = pd.Series(shock + effect * top + rng.normal(0, 0.03, len(idx)), index=idx)
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = make_panel()
    now = X.index.get_level_values(0).max()
    m = PatternMiner(MINER_PARAMS).fit(X, y, now)
    return X, y, now, m


def miner_inputs(X, y, now, m):
    """Re-derive exactly what fit() used: quantile matrix, relevance weights, row masks."""
    dates = X.index.get_level_values(0)
    Q = m._quintiles(X[m.feats])
    ctx = X[CTX].values
    ctx_now = X.xs(dates.max(), level=0)[CTX].iloc[0].values
    w = m._weights(dates, ctx, now, ctx_now)
    return dates, Q, ctx, ctx_now, w


# ------------------------------------------------------------------ basic moments
def test_weighted_mean_and_effective_n_known_values():
    assert S.weighted_mean([1, 1, 2], [0.0, 4.0, 4.0]) == pytest.approx(3.0)
    assert S.effective_n([1, 1, 1, 1]) == pytest.approx(4.0)
    assert S.effective_n([1, 1, 2]) == pytest.approx(16 / 6)
    assert S.effective_n([5, 0, 0]) == pytest.approx(1.0)               # one row carries everything


def test_weighted_stats_matches_patterns_wstats():
    rng = np.random.default_rng(1)
    w, y = rng.random(500), rng.normal(0.01, 0.05, 500)
    assert S.weighted_stats(w, y) == pytest.approx(_wstats(w, y), rel=1e-12)
    assert S.weighted_stats(np.zeros(5), np.ones(5)) == (0.0, 0.0, 0.0)


def test_negative_weights_and_mismatched_lengths_are_refused():
    with pytest.raises(S.StatsError):
        S.weighted_mean([1, -1], [0, 1])
    with pytest.raises(S.StatsError):
        S.weighted_mean([1, 1], [0, 1, 2])
    with pytest.raises(S.StatsError):
        S.effective_n([1, -2])


def test_t_to_p_agrees_with_patterns_and_is_exact_in_the_tail():
    for t in (0.0, 0.5, 1.96, 3.0, -4.2, 7.0):
        assert S.t_to_p(t) == pytest.approx(_t_to_p(t), abs=1e-12)
    assert S.t_to_p(1.959964) == pytest.approx(0.05, abs=1e-6)
    assert math.erf(12.0 / math.sqrt(2)) == 1.0     # the pre-Phase-3 formula 1 - erf(x) rounded to exactly zero here...
    assert 0.0 < S.t_to_p(12.0) < 1e-30            # ...the library keeps the tail (and patterns._t_to_p now uses it)
    assert S.norm_cdf(0.0) == pytest.approx(0.5)
    assert S.norm_cdf(np.array([-1.96, 1.96])).tolist() == pytest.approx([0.025, 0.975], abs=1e-4)


# ------------------------------------------------------------------ clustering
def test_cluster_test_reproduces_the_miner_on_every_reported_pattern(fitted):
    X, y, now, m = fitted
    dates, Q, ctx, ctx_now, w = miner_inputs(X, y, now, m)
    codes, n = S.date_codes(dates)
    split, recent = S.split_dates(dates)
    disc, conf, rec = dates < split, dates >= split, dates >= recent
    yv = y.values
    checked = 0
    for r in m.patterns.itertuples():
        mask = m._mask_from_key(r.key, Q, m.feats)
        d = S.cluster_test(codes, w, yv, mask & disc, n)
        c = S.cluster_test(codes, w, yv, mask & conf, n)
        a = S.cluster_test(codes, w, yv, mask, n)
        assert d.mean == pytest.approx(r.m_disc, rel=1e-9, abs=1e-12)
        assert d.t == pytest.approx(r.t_disc, rel=1e-9, abs=1e-9)
        assert c.t == pytest.approx(r.t_conf, rel=1e-9, abs=1e-9)
        assert a.mean == pytest.approx(r.m_all, rel=1e-9, abs=1e-12)
        assert a.n_eff == pytest.approx(r.n_eff, rel=1e-9)
        if (mask & rec).sum() >= 30:
            assert S.cluster_test(codes, w, yv, mask & rec, n).t == pytest.approx(r.t_recent, rel=1e-9, abs=1e-9)
        checked += 1
    assert checked >= 100                           # the equivalence covered a real population, not a handful


def test_split_dates_matches_the_miner_for_awkward_lengths():
    for n in (10, 37, 100, 241, 1000):
        dates = pd.bdate_range("2020-01-01", periods=n)
        split, recent = S.split_dates(dates)
        assert split == dates[int(n * 0.7)] and recent == dates[int(n * 0.8)]
    with pytest.raises(S.StatsError):
        S.split_dates(pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(S.StatsError):
        S.split_dates(pd.bdate_range("2020-01-01", periods=30), conf_frac=1.5)


def test_week_codes_group_by_iso_week_in_calendar_order():
    d = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-08", "2021-01-11", "2020-12-31", "2021-01-01"])
    codes, n = S.week_codes(d)
    assert n == 3
    assert codes[0] == codes[1] == codes[2]              # Mon-Fri of one ISO week
    assert codes[4] == codes[5] and codes[4] < codes[0] < codes[3]      # earlier week codes lower, year boundary ok


def test_empty_mask_and_too_few_clusters_are_not_evidence():
    codes, n = S.date_codes(pd.bdate_range("2020-01-01", periods=30).repeat(4))
    w, y = np.ones(120), np.random.default_rng(0).normal(0, 1, 120)
    e = S.cluster_test(codes, w, y, np.zeros(120, bool), n)
    assert (e.mean, e.t, e.n_eff, e.n_clusters) == (0.0, 0.0, 0.0, 0)
    few = np.zeros(120, bool)
    few[:16] = True                                       # four dates only
    f = S.cluster_test(codes, w, y, few, n)
    assert f.t == 0.0 and f.n_clusters == 4
    assert S.cluster_test(codes, np.zeros(120), y, np.ones(120, bool), n).n_clusters == 0     # zero weight everywhere


def test_hac_with_zero_lags_matches_the_iid_scale_and_lags_widen_the_error():
    rng = np.random.default_rng(2)
    n = 400
    codes = np.repeat(np.arange(n), 5)
    mk = np.convolve(rng.normal(size=n + 4), np.ones(5), "valid")           # overlapping 5-day sums: MA(4) series
    y = np.repeat(mk, 5) + rng.normal(0, 0.1, n * 5)
    w, mask = np.ones(n * 5), np.ones(n * 5, bool)
    iid = S.cluster_test(codes, w, y, mask, n)
    h0 = S.hac_cluster_test(codes, w, y, mask, n, lags=0)
    h4 = S.hac_cluster_test(codes, w, y, mask, n, lags=4)
    assert h0.se == pytest.approx(iid.se, rel=0.06)       # same order: different but consistent estimators
    hu = S.hac_cluster_test(codes, w, y, mask, n, lags=4, kernel="uniform")
    assert h4.se > 1.6 * iid.se                           # MA(4) noise: the true se is sqrt(5) = 2.2x larger
    assert hu.se == pytest.approx(2.24 * iid.se, rel=0.2)  # the uniform kernel is exact for that structure
    with pytest.raises(S.StatsError):
        S.hac_cluster_test(codes, w, y, mask, n, lags=4, kernel="parzen")
    with pytest.raises(S.StatsError):
        S.hac_cluster_test(codes, w, y, mask, n, lags=-1)
    assert S.hac_cluster_test(codes, w, y, np.zeros(n * 5, bool), n, lags=4).n_clusters == 0


def test_overlapping_returns_inflate_date_clustering_and_hac_repairs_it():
    """PLANTED DEFECT: outcomes are 5-day forward returns sampled daily, the mask carries no information. Per-date
    clustering (what patterns.py does) rejects far more than 5% of true nulls; the HAC statistic does not."""
    rng = np.random.default_rng(3)
    n, k, reps = 260, 8, 300
    codes = np.repeat(np.arange(n), k)
    w = np.ones(n * k)
    rej_iid = rej_hac = used = 0
    for _ in range(reps):
        daily = rng.normal(0, 0.01, n + 4)
        fwd = np.convolve(daily, np.ones(5), "valid")                       # each date's outcome overlaps the next 4
        y = np.repeat(fwd, k) + rng.normal(0, 0.005, n * k)
        state = np.repeat(rng.random(n // 26 + 1) < 0.5, 26)[:n]           # a persistent, information-free regime flag
        mask = np.repeat(state, k)
        if mask.sum() < 40:
            continue
        used += 1
        rej_iid += abs(S.cluster_test(codes, w, y, mask, n).t) > 1.96
        rej_hac += abs(S.hac_cluster_test(codes, w, y, mask, n, lags=4).t) > 1.96
    assert used > 250
    assert rej_iid / used > 0.20
    assert rej_hac / used < rej_iid / used - 0.08


def test_cluster_diagnostics_flag_serial_correlation_and_concentration():
    n = 200
    codes = np.repeat(np.arange(n), 3)
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(size=n))                                     # strongly autocorrelated cluster means
    d = S.cluster_diagnostics(codes, np.ones(n * 3), np.repeat(walk, 3), np.ones(n * 3, bool), n)
    assert d["lag1_autocorr"] > 0.8 and d["n_clusters"] == n
    w = np.ones(n * 3)
    w[:3] = 1e4
    assert S.cluster_diagnostics(codes, w, np.zeros(n * 3), np.ones(n * 3, bool), n)["max_weight_share"] > 0.9
    tiny = S.cluster_diagnostics(codes, np.ones(n * 3), np.zeros(n * 3), np.arange(n * 3) < 6, n)
    assert np.isnan(tiny["lag1_autocorr"])                                   # two clusters: no autocorrelation claimed


# ------------------------------------------------------------------ multiple testing
def test_bh_qvalues_textbook_example_and_agreement_with_the_miner_rule():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert S.bh_qvalues(p).tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert S.bh_reject(p, 0.03).tolist() == [True, False, False, True]
    rng = np.random.default_rng(5)
    for _ in range(50):
        pv = np.concatenate([rng.random(200), rng.random(30) * 1e-3])
        for q in (0.01, 0.05, 0.2):
            assert (S.bh_reject(pv, q) == (S.bh_qvalues(pv) <= q)).all()


def test_bh_rejection_reproduces_fdr_pass_of_the_miner(fitted):
    X, y, now, m = fitted
    pv = np.asarray(S.t_to_p(m.patterns["t_disc"].values))
    assert (S.bh_reject(pv, 0.05) == m.patterns["fdr_pass"].values).all()


def test_bh_controls_false_discovery_rate_on_planted_mixture():
    """900 true nulls + 100 real effects: the share of BH discoveries that are null stays near or under q."""
    rng = np.random.default_rng(6)
    fdp, naive = [], []
    truth_null = np.arange(1000) < 900
    for _ in range(40):
        t = np.concatenate([rng.normal(0, 1, 900), rng.normal(4.0, 1, 100)])
        p = S.t_to_p(t)
        rej = S.bh_reject(p, 0.05)
        fdp.append((rej & truth_null).sum() / max(rej.sum(), 1))
        naive.append(((p < 0.05) & truth_null).sum())
    assert np.mean(fdp) < 0.07
    assert np.mean(naive) > 30                          # without correction ~45 nulls slip through per run


def test_bh_edge_cases():
    assert S.bh_qvalues([]).size == 0 and S.bh_reject([], 0.05).size == 0
    assert S.bh_qvalues([0.5]).tolist() == [0.5]
    assert S.bh_qvalues([0.001] * 5).tolist() == pytest.approx([0.001] * 5)
    with pytest.raises(S.StatsError):
        S.bh_qvalues([0.1, float("nan")])
    assert S.bonferroni([0.01, 0.2, 0.9]).tolist() == pytest.approx([0.03, 0.6, 1.0])


# ------------------------------------------------------------------ permutation null and local fdr
def test_permutation_preserves_each_clusters_outcomes_and_kills_a_planted_link():
    rng = np.random.default_rng(7)
    n, k = 120, 30
    codes = np.repeat(np.arange(n), k)
    mask = rng.random(n * k) < 0.3
    y = rng.normal(0, 0.02, n * k) + 0.03 * mask + np.repeat(rng.normal(0, 0.03, n), k)
    yp = S.permute_within_clusters(y, codes, np.random.default_rng(8))
    for c in (0, 17, 99):
        assert np.sort(yp[codes == c]).tolist() == pytest.approx(np.sort(y[codes == c]).tolist())
    assert (yp != y).mean() > 0.9
    w = np.ones(n * k)
    real = S.cluster_test(codes, w, y, mask, n).t
    null = S.cluster_test(codes, w, yp, mask, n).t
    assert real > 6 and abs(null) < 3


def test_permutation_is_seeded_and_length_checked():
    codes = np.repeat(np.arange(10), 5)
    y = np.arange(50.0)
    a = S.permute_within_clusters(y, codes, np.random.default_rng(1))
    b = S.permute_within_clusters(y, codes, np.random.default_rng(1))
    c = S.permute_within_clusters(y, codes, np.random.default_rng(2))
    assert (a == b).all() and (a != c).any()
    with pytest.raises(S.StatsError):
        S.permute_within_clusters(y, codes[:-1], np.random.default_rng(1))


def test_local_fdr_equals_a_brute_force_count_and_is_capped():
    rng = np.random.default_rng(9)
    null_t = np.sort(np.abs(rng.normal(0, 1, 700)))
    real_t = np.sort(np.abs(np.concatenate([rng.normal(0, 1, 300), rng.normal(5, 1, 60)])))
    for t in (0.2, 1.0, 2.5, 4.0, 6.5, 12.0):
        fn = (null_t >= t).mean()
        fr = max((real_t >= t).mean(), 1e-9)
        assert S.local_fdr(t, null_t, real_t) == pytest.approx(min(1.0, fn / fr))
    assert S.local_fdr(0.0, null_t, real_t) == pytest.approx(1.0)
    assert S.local_fdr(50.0, null_t, real_t) == pytest.approx(0.0)
    assert S.local_fdr(1.0, np.array([]), real_t) <= 1.0            # no null patterns: the miner's [0.0] convention


def test_local_fdr_monotone_never_rewards_a_weaker_pattern():
    rng = np.random.default_rng(10)
    null_t = np.sort(np.abs(rng.normal(0, 1, 300)))
    real_t = np.sort(np.abs(np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 20)])))
    t = rng.uniform(0, 8, 60)
    f = S.local_fdr_monotone(t, null_t, real_t)
    order = np.argsort(t)
    assert (np.diff(f[order]) <= 1e-12).all()
    assert (f <= np.array([S.local_fdr(v, null_t, real_t) for v in t]) + 1e-12).all()


def test_null_distribution_search_on_pure_noise_is_calm_and_empty_safe():
    rng = np.random.default_rng(11)
    n, k = 150, 20
    codes = np.repeat(np.arange(n), k)
    y = rng.normal(0, 0.02, n * k)
    masks = [rng.random(n * k) < 0.25 for _ in range(60)]
    stat = lambda mk, yy: S.cluster_test(codes, np.ones(n * k), yy, mk, n).t
    null_t = S.null_t_distribution(masks, stat, y, codes, np.random.default_rng(0), reps=2)
    assert len(null_t) == 120 and (np.diff(null_t) >= 0).all()
    assert np.quantile(null_t, 0.95) < 3.0
    assert S.null_t_distribution([], stat, y, codes, np.random.default_rng(0)).tolist() == [0.0]
    thin = S.null_t_distribution(masks, lambda mk, yy: None, y, codes, np.random.default_rng(0))
    assert thin.tolist() == [0.0]                               # everything too thin to test: the miner's convention
    capped = S.null_t_distribution(masks, stat, y, codes, np.random.default_rng(0), reps=1, max_patterns=10)
    assert len(capped) == 10


# ------------------------------------------------------------------ P(real) and effect
def test_confirmation_factor_requires_the_same_sign():
    f = S.confirmation_factor([2.0, 2.0, 0.0], [0.01, -0.01, 0.01], [0.02, 0.02, 0.02])
    assert f[0] == pytest.approx(S.norm_cdf(2.0)) and f[1] == 0.0 and f[2] == pytest.approx(0.5)


def test_p_real_reproduces_the_miner_column_with_bonferroni(fitted):
    X, y, now, m = fitted
    R = m.patterns
    f = S.confirmation_factor(R["t_conf"].abs().values, R["m_conf"].values, R["m_disc"].values)
    pr = S.p_real(R["p_hallucinated"].values, R["p_coincidence"].values, f, method="bonferroni")
    assert pr == pytest.approx(R["p_real"].values, abs=1e-12)
    assert S.p_real(R["p_hallucinated"].values, R["p_coincidence"].values, f, method="bh").min() >= 0.0


def test_p_real_blueprint_bh_is_never_stricter_than_bonferroni_and_zero_on_sign_flip():
    rng = np.random.default_rng(12)
    p = np.sort(rng.random(400) ** 3)
    ph = np.full(400, 0.05)
    f = np.ones(400)
    bh = S.p_real(ph, p, f, "bh")
    bo = S.p_real(ph, p, f, "bonferroni")
    assert (bh >= bo - 1e-12).all() and (bh > bo + 0.05).any()
    assert (S.p_real(ph, p, np.zeros(400)) == 0).all()
    with pytest.raises(S.StatsError):
        S.p_real(ph, p[:10], f)
    with pytest.raises(S.StatsError):
        S.corrected_coincidence(p, "sidak")


def test_shrink_effect_matches_the_miner(fitted):
    X, y, now, m = fitted
    R = m.patterns
    # the k-rule column is exactly the helper; since 6a14d14 the REPORTED effect is the empirical-Bayes one by design
    assert S.shrink_effect(R["m_all"].values, R["n_eff"].values, 400) == pytest.approx(R["effect_k"].values, rel=1e-12)
    assert m.p["effect_method"] == "eb"
    assert R["effect"].values == pytest.approx(R["effect_eb"].values, rel=1e-12)
    assert not np.allclose(R["effect_eb"].values, R["effect_k"].values)   # the two rules really differ, so the check bites
    assert S.shrink_effect(0.02, 0.0, 400) == 0.0                     # no evidence, no effect


def test_empirical_bayes_recovers_planted_prior_variance_and_beats_raw():
    rng = np.random.default_rng(13)
    tau, k = 0.01, 3000
    true = rng.normal(0, tau, k)
    se = rng.uniform(0.005, 0.03, k)
    obs = true + rng.normal(0, 1, k) * se
    eb = S.eb_shrink(obs, se, center=0.0)
    assert eb["tau2"] == pytest.approx(tau ** 2, rel=0.15)
    assert ((eb["post_mean"] - true) ** 2).mean() < 0.7 * ((obs - true) ** 2).mean()
    assert (eb["B"] > 0).all() and (eb["B"] < 1).all()
    assert (np.diff(eb["B"][np.argsort(se)]) <= 1e-12).all()             # noisier -> shrunk more
    est = S.eb_shrink(obs + 0.004, se, center=None)                      # estimated centre finds the offset
    assert est["mu"] == pytest.approx(0.004, abs=0.0015)


def test_empirical_bayes_on_pure_noise_shrinks_everything_to_zero():
    rng = np.random.default_rng(14)
    se = rng.uniform(0.005, 0.02, 500)
    obs = rng.normal(0, 1, 500) * se
    eb = S.eb_shrink(obs, se, center=0.0)
    assert eb["tau2"] < 2e-6
    assert np.abs(eb["post_mean"]).max() < 0.1 * np.abs(obs).max() + 1e-4
    assert math.isinf(S.implied_k(0.0, se, np.ones(500)))


def test_implied_k_is_variance_over_tau_squared():
    se, n = np.array([0.01, 0.02]), np.array([100.0, 100.0])
    v = se ** 2 * n
    assert S.implied_k(2.5e-5, se, n) == pytest.approx(np.median(v) / 2.5e-5)


def test_eb_degenerate_inputs():
    e = S.eb_shrink([0.01], [0.005])
    assert e["n_used"] == 1 and e["post_mean"].tolist() == [0.01]      # one pattern: nothing to learn a prior from
    e = S.eb_shrink([0.01, np.nan, 0.02], [0.005, 0.005, 0.0])
    assert e["n_used"] == 1
    with pytest.raises(S.StatsError):
        S.eb_shrink([1, 2], [1])


# ------------------------------------------------------------------ the whole pipeline on planted truth
def planted_family(seed, n_null=60, effect=0.012, n_dates=260, n_tick=40):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    d = np.repeat(dates, n_tick)
    n = len(d)
    shock = pd.Series(rng.normal(0, 0.02, n_dates), index=dates).reindex(d).values
    real = rng.random(n) < 0.2
    y = shock + effect * real + rng.normal(0, 0.03, n)
    masks = [real] + [rng.random(n) < rng.uniform(0.1, 0.35) for _ in range(n_null)]
    names = ["planted"] + [f"null{i}" for i in range(n_null)]
    return y, masks, names, pd.DatetimeIndex(d)


def test_pipeline_finds_the_planted_effect_and_admits_almost_no_null():
    y, masks, names, dates = planted_family(15)
    w = S.relevance(dates, dates.max(), params={"half_life_years": 50.0})
    R = S.evaluate_masks(masks, y, w, dates, {"min_n": 200, "null_reps": 2}, seed=3, names=names)
    row = R[R["name"] == "planted"].iloc[0]
    assert row["p_real"] >= 0.8 and row["fdr_pass"] and row["m_disc"] > 0 and row["m_conf"] > 0
    nulls = R[R["name"] != "planted"]
    assert (nulls["p_real"] >= 0.8).sum() <= 1
    assert row["effect"] == pytest.approx(row["m_all"] * row["n_eff"] / (row["n_eff"] + 400))
    assert 0 < row["effect_eb"] < row["m_all"] + 1e-9
    assert R.attrs["n_tested"] == len(R) and R.attrs["null_n"] > 0


def test_pipeline_calibration_p_real_bin_matches_planted_truth():
    """Across several worlds, patterns scored P(real) >= 0.8 must be real at least that often (planted truth)."""
    ps, truth = [], []
    for seed in range(16, 22):
        y, masks, names, dates = planted_family(seed, n_null=40, effect=0.010)
        w = S.relevance(dates, dates.max(), params={"half_life_years": 50.0})
        R = S.evaluate_masks(masks, y, w, dates, {"min_n": 200, "null_reps": 1}, seed=seed, names=names)
        ps += R["p_real"].tolist()
        truth += (R["name"] == "planted").tolist()
    tab = S.calibration_table(np.array(ps), np.array(truth))
    top = tab.iloc[-1]
    assert top["n"] >= 3 and top["real_share"] >= 0.8          # every bin at/above 0.8 is at least 80% real
    assert tab.iloc[0]["real_share"] < 0.1                     # the lowest bin is almost all noise


def test_effect_that_flips_sign_after_discovery_gets_zero_p_real():
    """PLANTED DEFECT: an effect that exists only in the discovery window must not survive confirmation."""
    y, masks, names, dates = planted_family(23, n_null=10, effect=0.0)
    split, _ = S.split_dates(dates)
    y = y + np.where(masks[0] & (dates < split), 0.02, 0.0) - np.where(masks[0] & (dates >= split), 0.02, 0.0)
    R = S.evaluate_masks(masks, y, np.ones(len(y)), dates, {"min_n": 200}, names=names)
    row = R[R["name"] == "planted"].iloc[0]
    assert row["t_disc"] > 5 and row["t_conf"] < -5 and row["p_real"] == 0.0
    assert not S.admission_flags(R)[R["name"] == "planted"].iloc[0]


def test_pipeline_drops_thin_patterns_from_the_family_and_handles_empty_input():
    y, masks, names, dates = planted_family(24, n_null=4)
    tiny = np.zeros(len(y), bool)
    tiny[:30] = True
    R = S.evaluate_masks(masks + [tiny], y, np.ones(len(y)), dates, {"min_n": 200}, names=names + ["tiny"])
    assert "tiny" not in set(R["name"]) and len(R) == 5
    E = S.evaluate_masks([], y, np.ones(len(y)), dates)
    assert E.empty and "p_real" in E.columns
    assert S.evaluate_masks([tiny], y, np.ones(len(y)), dates, {"min_n": 200}).empty
    with pytest.raises(S.StatsError):
        S.evaluate_masks(masks, y[:-1], np.ones(len(y)), dates)
    y_nan = y.copy()
    y_nan[5] = np.nan
    with pytest.raises(S.StatsError):
        S.evaluate_masks(masks, y_nan, np.ones(len(y)), dates)


def test_pipeline_is_deterministic_for_a_seed():
    y, masks, names, dates = planted_family(25, n_null=20)
    kw = dict(params={"min_n": 200}, names=names)
    a = S.evaluate_masks(masks, y, np.ones(len(y)), dates, seed=5, **kw)
    b = S.evaluate_masks(masks, y, np.ones(len(y)), dates, seed=5, **kw)
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_hac_switch_shrinks_t_when_outcomes_overlap():
    y, masks, names, dates = planted_family(30, n_null=6, effect=0.0)
    n_d = len(dates.unique())
    fwd = np.convolve(np.random.default_rng(1).normal(0, 0.01, n_d + 4), np.ones(5), "valid")
    y = pd.Series(fwd).reindex(range(n_d)).repeat(40).values + np.random.default_rng(2).normal(0, 0.005, len(dates))
    iid = S.evaluate_masks(masks, y, np.ones(len(y)), dates, {"min_n": 200}, names=names)
    hac = S.evaluate_masks(masks, y, np.ones(len(y)), dates, {"min_n": 200}, names=names, hac_lags=4)
    assert hac["t_disc"].abs().mean() < iid["t_disc"].abs().mean()


def test_calibration_table_shapes_and_validation():
    t = S.calibration_table([0.05, 0.9, 0.95, 0.5], [False, True, True, False])
    assert t["n"].sum() == 4 and t.iloc[-1]["real_share"] == 1.0 and t.iloc[0]["real_share"] == 0.0
    with pytest.raises(S.StatsError):
        S.calibration_table([0.1], [True, False])
    assert S.admission_flags(pd.DataFrame()).empty


# ------------------------------------------------------------------ stability across blocks and eras
def test_block_breakdown_separates_a_persistent_edge_from_a_one_block_fluke():
    n, k = 300, 30
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2015-01-01", periods=n), k))
    rng = np.random.default_rng(26)
    mask = rng.random(n * k) < 0.3
    persistent = rng.normal(0, 0.03, n * k) + 0.012 * mask
    fluke = rng.normal(0, 0.03, n * k) + 0.06 * (mask & (np.arange(n * k) < n * k // 5))
    w = np.ones(n * k)
    bp = S.block_breakdown(mask, persistent, w, dates, 5)
    bf = S.block_breakdown(mask, fluke, w, dates, 5)
    assert S.sign_agreement(bp, 1.0)["same_sign"] >= 4
    assert bf["t"].iloc[0] > 5 and (bf["t"].iloc[1:].abs() < 3.5).all()
    assert (bp["n_rows"] > 0).all() and len(bp) == 5
    with pytest.raises(S.StatsError):
        S.block_edges(dates[:3], 10)
    assert np.isnan(S.sign_agreement(bp.iloc[0:0], 1.0)["share"])


def test_era_breakdown_and_bootstrap_ci_cover_the_truth():
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2000-01-03", "2021-12-31", freq="7D"), 20))
    rng = np.random.default_rng(27)
    mask = rng.random(len(dates)) < 0.4
    y = rng.normal(0, 0.03, len(dates)) + 0.02 * mask
    E = S.era_breakdown(mask, y, np.ones(len(y)), dates)
    assert list(E["era"]) == ["pre_decimal", "decimal", "electronic", "etf_dominant"] and (E["t"] > 3).all()
    codes, n = S.date_codes(dates)
    lo, hi = S.cluster_bootstrap_ci(codes, np.ones(len(y)), y, mask, n, np.random.default_rng(1), reps=300)
    assert lo < 0.02 < hi and hi - lo < 0.01
    empty = S.cluster_bootstrap_ci(codes, np.ones(len(y)), y, np.zeros(len(y), bool), n, np.random.default_rng(1))
    assert all(np.isnan(v) for v in empty)


# ------------------------------------------------------------------ redundancy and the validation-gain gate
def test_jaccard_and_containment():
    a = np.array([1, 1, 1, 0, 0], bool)
    b = np.array([1, 1, 0, 0, 0], bool)
    assert S.jaccard(a, b) == pytest.approx(2 / 3) and S.containment(b, a) == 1.0
    assert S.containment(a, b) == pytest.approx(2 / 3)
    assert S.jaccard(np.zeros(4, bool), np.zeros(4, bool)) == 0.0 and S.containment(np.zeros(4, bool), a[:4]) == 0.0


def test_redundancy_holds_on_real_miner_output(fitted):
    X, y, now, m = fitted
    dates, Q, *_ = miner_inputs(X, y, now, m)
    P = m.patterns
    masks = [m._mask_from_key(k, Q, m.feats) for k in P["key"]]
    live = [i for i, s in enumerate(P["status"]) if s in ("active", "rescoped", "no_gain")]
    for a in range(len(live)):
        for b in range(a):
            assert S.jaccard(masks[live[a]], masks[live[b]]) <= 0.8 + 1e-9
    names = P["key_named"].tolist()
    for i in [i for i, s in enumerate(P["status"]) if s == "duplicate"]:
        assert S.jaccard(masks[i], masks[names.index(P.at[i, "duplicate_of"])]) > 0.8


def test_prune_redundant_keeps_the_stronger_and_records_the_keeper():
    base = np.zeros(1000, bool)
    base[:400] = True
    child = base.copy()
    child[:20] = False                                    # 95% overlap with base
    other = np.zeros(1000, bool)
    other[600:900] = True
    kept, dup = S.prune_redundant([0, 1, 2], [base, child, other])
    assert kept == [0, 2] and dup == {1: 0}
    kept2, dup2 = S.prune_redundant([1, 0, 2], [base, child, other])       # order decides who survives
    assert kept2 == [1, 2] and dup2 == {0: 1}
    small = np.zeros(1000, bool)
    small[:100] = True
    assert S.prune_redundant([0, 3], [base, child, other, small])[1] == {}       # Jaccard 0.25 misses a nested pattern
    assert S.prune_redundant([0, 3], [base, child, other, small], use_containment=True)[1] == {3: 0}


def test_evidence_order_puts_simple_before_complex_then_strong_before_weak():
    names = ["a q1 & b q2", "a q1", "c q0", "a q1 & b q2 unless c q4"]
    assert S.evidence_order(names, [9.0, 2.0, 3.0, 20.0], [9.0, 2.0, 4.0, 20.0]) == [2, 1, 0, 3]


def test_validation_gain_gate_reproduces_the_miners_greedy_selection(fitted):
    X, y, now, m = fitted
    dates, Q, *_ = miner_inputs(X, y, now, m)
    split, _ = S.split_dates(dates)
    conf = np.asarray(dates >= split)
    P = m.patterns
    cand = [i for i, s in enumerate(P["status"]) if s in ("active", "rescoped", "no_gain")]
    masks = [m._mask_from_key(k, Q, m.feats) for k in P["key"]]
    kept, best, traj = S.validation_gain_gate(cand, masks, P["effect"].values, y.values[conf], conf, 0.0005)
    assert set(kept) == {i for i in cand if P.at[i, "status"] != "no_gain"}
    assert best == pytest.approx(m.gate_corr, abs=1e-12)
    assert len(traj) == len(cand)


def test_validation_gain_gate_rejects_noise_and_accepts_signal():
    rng = np.random.default_rng(28)
    n = 6000
    conf = np.ones(n, bool)
    sig = rng.random(n) < 0.3
    y = rng.normal(0, 0.03, n) + 0.02 * sig
    noise = [rng.random(n) < 0.3 for _ in range(10)]
    kept, best, _ = S.validation_gain_gate(list(range(11)), [sig] + noise, np.full(11, 0.02), y, conf, 0.0005)
    assert kept[0] == 0 and best > 0.15
    assert len(kept) <= 3                                           # noise patterns almost never add out-of-sample skill
    with pytest.raises(S.StatsError):
        S.validation_gain_gate([0], [sig], [0.1], y[:10], conf)
    k0, b0, t0 = S.validation_gain_gate([], [sig], [0.1], y, conf)
    assert (k0, b0, t0) == ([], 0.0, [])


# ------------------------------------------------------------------ relevance (3.3)
def test_recency_weight_halves_each_half_life_and_zeroes_the_future():
    now = pd.Timestamp("2020-01-01")
    d = pd.DatetimeIndex([now - pd.Timedelta(days=int(365.25 * 4)), now, now + pd.Timedelta(days=5)])
    w = S.recency_weight(d, now, 4.0)
    assert w[0] == pytest.approx(0.5, abs=1e-3) and w[1] == 1.0 and w[2] == 0.0
    with pytest.raises(S.LookAheadError):
        S.recency_weight(d, now, 4.0, on_future="raise")
    with pytest.raises(S.StatsError):
        S.recency_weight(d, now, 0.0)


def test_relevance_equals_the_miner_weights_when_era_is_neutral(fitted):
    X, y, now, m = fitted
    dates, Q, ctx, ctx_now, w_miner = miner_inputs(X, y, now, m)
    w = S.relevance(dates, now, ctx, ctx_now, features=["f0", "gap"])       # 'gap' is microstructure-sensitive: still neutral
    assert w == pytest.approx(w_miner, rel=1e-12)
    assert S.context_now(ctx, dates, now) == pytest.approx(ctx_now)


def test_relevance_uses_no_information_after_now():
    """PLANTED DEFECT: rows appended AFTER `now` (with a wild context that would inflate the scale) must change nothing."""
    rng = np.random.default_rng(29)
    dates = pd.bdate_range("2019-01-01", periods=200).repeat(3)
    ctx = rng.normal(size=(len(dates), 2))
    now = dates[300]
    past = dates <= now
    base = S.relevance(dates[past], now, ctx[past], ctx[past][-1])
    fut_dates = dates[past].append(pd.bdate_range(now + pd.Timedelta(days=1), periods=50).repeat(3))
    fut_ctx = np.vstack([ctx[past], rng.normal(0, 500, size=(150, 2))])
    ext = S.relevance(fut_dates, now, fut_ctx, ctx[past][-1])
    assert ext[: past.sum()] == pytest.approx(base, rel=1e-12)
    assert (ext[past.sum():] == 0).all()
    with pytest.raises(S.LookAheadError):
        S.relevance(fut_dates, now, fut_ctx, ctx[past][-1], params={"on_future": "raise"})
    with pytest.raises(S.LookAheadError):
        S.context_now(ctx, dates, pd.Timestamp("2000-01-01"))


def test_similar_context_outweighs_a_distant_one_and_missing_context_is_neutral():
    ctx = np.array([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [np.nan, 0.0]])
    sim = S.context_similarity(ctx, [0.0, 0.0], 1.5)
    assert sim[0] == 1.0 and sim[1] > sim[2] and sim[3] > 0.5
    with pytest.raises(S.StatsError):
        S.context_similarity(ctx, [0.0], 1.5)
    with pytest.raises(S.StatsError):
        S.context_similarity(ctx, [0, 0], 0.0)
    assert S.context_similarity(np.ones((4, 2)), [1, 1]).tolist() == [1.0] * 4     # zero-variance columns do not blow up


def test_era_weights_apply_only_to_microstructure_sensitive_patterns():
    d = pd.DatetimeIndex(["1999-06-01", "2004-06-01", "2010-06-01", "2020-06-01"])
    assert S.era_of(d).tolist() == ["pre_decimal", "decimal", "electronic", "etf_dominant"]
    table = {"pre_decimal": 0.2, "decimal": 0.5}
    assert S.era_weight(d, table, sensitive=True).tolist() == [0.2, 0.5, 1.0, 1.0]
    assert S.era_weight(d, table, sensitive=False).tolist() == [1.0] * 4
    assert S.microstructure_sensitive(["r20", "gap_today"]) and not S.microstructure_sensitive(["r20", "mom_12_1"])
    rel_sens = S.relevance(d, pd.Timestamp("2021-01-01"), features=["gap"], era_weights=table)
    rel_not = S.relevance(d, pd.Timestamp("2021-01-01"), features=["r20"], era_weights=table)
    assert rel_sens[0] == pytest.approx(rel_not[0] * 0.2)
    with pytest.raises(S.StatsError):
        S.era_weight(d, {"jurassic": 1.0})
    with pytest.raises(S.StatsError):
        S.era_weight(d, {"decimal": -1.0})


# ------------------------------------------------------------------ the miner's own integration of the library
def test_miner_detects_overlap_from_date_spacing_and_switches_the_standard_error():
    daily = PatternMiner({"horizon": 5})
    assert daily._overlap(np.sort(pd.bdate_range("2020-01-01", periods=100).values)) == {"spacing": 1.0, "hac_lags": 8}
    two = np.sort(pd.bdate_range("2020-01-01", periods=100)[::2].values)
    assert daily._overlap(two) == {"spacing": 2.0, "hac_lags": 4}
    weekly = np.sort(pd.bdate_range("2020-01-01", periods=300)[::5].values)
    assert daily._overlap(weekly) == {"spacing": 5.0, "hac_lags": 0}
    assert PatternMiner({"hac_lags": 3})._overlap(weekly)["hac_lags"] == 3
    assert PatternMiner({"hac_lags": 0})._overlap(np.sort(pd.bdate_range("2020-01-01", periods=50).values))["hac_lags"] == 0


def test_miner_on_overlapping_daily_rows_is_more_conservative_than_per_date_clustering():
    X, y = make_panel(n_dates=240, n_tick=40, seed=4)
    now = X.index.get_level_values(0).max()
    fwd = y.groupby(level=0).mean().rolling(5).sum().shift(-4).fillna(0.0)            # market-wide 5-day overlapping outcome
    y5 = y * 0.2 + fwd.reindex(X.index.get_level_values(0)).values
    base = {"max_pairs": 120, "max_unless": 20, "null_reps": 1, "min_n": 120, "p_method": "bonferroni"}
    naive = PatternMiner({**base, "hac_lags": 0}).fit(X, y5, now)
    robust = PatternMiner(base).fit(X, y5, now)
    assert robust.report["hac_lags"] == 8 and naive.report["hac_lags"] == 0
    assert robust.patterns["t_disc"].abs().median() < naive.patterns["t_disc"].abs().median()
    assert (robust.patterns["n_eff"] <= naive.patterns["n_eff"] + 1e-9).all()        # overlap shrinks the evidence


def test_rows_after_now_are_ignored_by_the_miner():
    """PLANTED DEFECT: a huge fake effect dated after `now` must not change anything the miner learns."""
    X, y = make_panel(n_dates=200, n_tick=40, seed=5)
    dates = X.index.get_level_values(0)
    now = dates.unique()[150]
    base = {"max_pairs": 100, "max_unless": 20, "null_reps": 1, "min_n": 120, "hac_lags": 0}
    a = PatternMiner(base).fit(X, y, now)
    y_leaky = y.copy()
    y_leaky[dates > now] += 5.0 * (X["f3"][dates > now] > 0)
    b = PatternMiner(base).fit(X, y_leaky, now)
    assert b.report["rows_after_now_dropped"] == int((dates > now).sum())
    pd.testing.assert_frame_equal(a.patterns.drop(columns=["scope"]), b.patterns.drop(columns=["scope"]))


def test_miner_bh_default_admits_at_least_what_bonferroni_admits():
    X, y = make_panel(n_dates=240, n_tick=50, seed=6)
    now = X.index.get_level_values(0).max()
    base = {**MINER_PARAMS, "hac_lags": 0}
    bo = PatternMiner({**base, "p_method": "bonferroni"}).fit(X, y, now)
    bh = PatternMiner({**base, "p_method": "bh"}).fit(X, y, now)
    assert (bh.patterns["p_real"].values >= bo.patterns["p_real"].values - 1e-12).all()
