"""Tests for the sector / stock analog engines and analog weighting (Bible PHASE 8). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from engine import analog_weighting as W
from engine import analogs_sector as S
from engine import analogs_stock as T


# ---- fixtures ----------------------------------------------------------------------------------------------------------
def ar1(rng, n, phi):
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return x


def planted(n=1800, n_noise=8, seed=1, signal=True, noise_phi=0.95):
    """Persistent latent state s drives the outcome; two fingerprint columns read s, the rest are persistent noise."""
    rng = np.random.default_rng(seed)
    s = ar1(rng, n, 0.97)
    idx = pd.bdate_range("2010-01-01", periods=n)
    F = pd.DataFrame({"sig_a": s + 0.3 * rng.normal(size=n), "sig_b": -s + 0.3 * rng.normal(size=n)}, index=idx)
    for j in range(n_noise):
        F[f"noise_{j}"] = ar1(rng, n, noise_phi)
    drive = s if signal else ar1(rng, n, 0.97)
    O = pd.DataFrame({"fwd_ret_21d": 0.03 * drive + 0.02 * rng.normal(size=n),
                      "fwd_vol_21d": 0.2 + 0.05 * np.abs(drive),
                      "fwd_maxdd_21d": -0.03 - 0.01 * np.abs(drive)}, index=idx)
    return F, O


def engine(F, O, **kw):
    return W.KNNEngine(F, O, "fwd_ret_21d", **kw)


# ---- primitives ----------------------------------------------------------------------------------------------------
def test_select_episodes_spacing_and_one_per_episode():
    order = np.array([10, 11, 12, 40, 41, 100, 30, 200])
    got = W.select_episodes(order, 5, 21)
    assert got == [10, 40, 100, 200]                       # 11,12 share 10's episode; 41 shares 40's; 30 is 20 from 10
    assert all(abs(a - b) >= 21 for i, a in enumerate(got) for b in got[:i])
    assert W.select_episodes(np.array([], int), 3, 21) == []


def test_distance_metrics_and_zero_weights():
    Z = np.array([[0.0, 0.0], [3.0, 4.0]])
    zt = np.zeros(2)
    assert W.weighted_distance(Z, zt, [1, 1], "euclid")[1] == pytest.approx(np.sqrt(12.5))
    assert W.weighted_distance(Z, zt, [1, 0], "euclid")[1] == pytest.approx(3.0)      # weight 0 removes a feature
    assert W.weighted_distance(Z, zt, [1, 1], "manhattan")[1] == pytest.approx(3.5)
    assert W.weighted_distance(np.array([[1.0, 1.0]]), np.array([2.0, 2.0]), [1, 1], "cosine")[0] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        W.weighted_distance(Z, zt, [0, 0])
    with pytest.raises(ValueError):
        W.weighted_distance(Z, zt, [1, 1], "nope")


def test_pit_moments_ignore_the_future():
    F, _ = planted(400)
    mu1, sd1, _ = W.pit_moments(F)
    G = F.copy()
    G.iloc[250:] += 1000.0
    mu2, sd2, _ = W.pit_moments(G)
    assert np.allclose(mu1[:250], mu2[:250], equal_nan=True) and np.allclose(sd1[:250], sd2[:250], equal_nan=True)
    assert not np.allclose(mu1[300], mu2[300])            # the comparison can fail: moments move once shifted rows are past


def test_publication_lag_shifts_by_column():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    out = W.publication_lag(df, {"a": 3})
    assert out["a"].iloc[3] == 0.0 and out["b"].iloc[1] == 0.0 and np.isnan(out["a"].iloc[2])


# ---- engine rules: gap, episodes, no look-ahead ----------------------------------------------------------------------
def test_engine_rules_gap_sep_and_output_fields():
    F, O = planted(900)
    eng = engine(F, O)
    for i in (400, 600, 899):
        r = eng.query(i, k=6)
        A = r["analogs"]
        pos = F.index.get_indexer(A["date"])
        assert (i - pos >= 63).all()                                             # analog end >= 63 sessions before the query
        assert all(abs(a - b) >= 21 for n_, a in enumerate(pos) for b in pos[:n_])
        assert (A["age_sessions"].values == i - pos).all() and (A["age_years"] > 0).all()
        assert np.isclose(A["weight"].sum(), 1.0)
        for key in ("forecast_return", "forecast_vol", "forecast_drawdown", "uniqueness", "n_analogs", "confidence"):
            assert key in r and np.isfinite(r[key])
        assert 0.0 <= r["confidence"] <= 1.0


def test_query_never_reads_after_the_query_row():
    F, O = planted(700)
    a = engine(F, O).query(500, k=5)
    F2, O2 = F.copy(), O.copy()
    F2.iloc[501:] = 9e3                                   # destroy the entire future, features and outcomes
    O2.iloc[501:] = -9e3
    b = engine(F2, O2).query(500, k=5)
    assert list(a["analogs"]["date"]) == list(b["analogs"]["date"])
    assert a["forecast_return"] == pytest.approx(b["forecast_return"])
    # the check can fail: a pool row inside the allowed past that is altered DOES change the answer
    F3 = F.copy()
    F3.iloc[int(F.index.get_indexer(a["analogs"]["date"])[0])] += 4.0
    c = engine(F3, O).query(500, k=5)
    assert list(c["analogs"]["date"]) != list(a["analogs"]["date"]) or c["forecast_return"] != a["forecast_return"]


def test_planted_exact_repeat_is_the_nearest_analog():
    F, O = planted(1000, signal=False)
    F.iloc[900] = F.iloc[300] + 1e-4
    O.iloc[300, 0] = 0.9
    r = engine(F, O).query(900, k=5)
    assert r["analogs"]["date"].iloc[0] == F.index[300]
    assert r["nearest_distance"] < 0.01 and r["forecast_return"] > 0.3
    assert r["uniqueness"] < 0.05                         # a day with an exact twin is not unique


def test_uniqueness_flags_a_state_with_no_precedent():
    F, O = planted(900)
    F.iloc[800] = 25.0                                    # nothing like it ever happened
    eng = engine(F, O)
    assert eng.query(800, k=5)["uniqueness"] > 3 * eng.query(700, k=5)["uniqueness"]


# ---- degenerate cases -------------------------------------------------------------------------------------------------
def test_degenerate_inputs():
    F, O = planted(900)
    eng = engine(F, O)
    assert eng.query(100) is None                         # not enough history
    assert eng.query(500, uniform=True)["n_analogs"] >= 1
    with pytest.raises(ValueError):
        W.KNNEngine(F, O.iloc[:-1], "fwd_ret_21d")
    with pytest.raises(KeyError):
        W.KNNEngine(F, O, "fwd_nope")
    with pytest.raises(ValueError):
        W.KNNEngine(F, O, "fwd_ret_21d", horizon=21, gap=10)
    Fc = F.copy()
    Fc["const"] = 1.0
    Fc["dead"] = np.nan
    r = engine(Fc, O).query(600)
    assert r is not None and r["n_features"] == F.shape[1]                       # constant and all-NaN columns are dropped
    E = W.KNNEngine(F.iloc[:0], O.iloc[:0], "fwd_ret_21d")
    assert W.predict_series(E, [], "weighted").empty
    assert W.ablation_table({"a": pd.DataFrame(columns=["pred", "actual"])}).shape[0] == 1
    with pytest.raises(ValueError):
        eng.query(500, mode="random")                     # a random control without a seed is refused
    assert W.eval_positions(engine(F.iloc[:200], O.iloc[:200])) == []


# ---- weighting: planted effect and null ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def signal_run():
    F, O = planted(1800, seed=3, n_noise=20, noise_phi=0.9)
    eng = engine(F, O)
    lo = eng.min_hist + eng.gap + eng.horizon + 8 * eng.sep
    sched = W.learn_weights_walk_forward(eng, list(range(lo, 1800, 250)), seed=5)
    pos = W.eval_positions(eng, F.index[lo + 250], None, 21)
    table, preds = W.compare_methods(eng, pos, k=5, seed=7, n_random=10)
    return eng, sched, table, preds


def test_analogs_beat_no_analog_and_random_when_an_effect_exists(signal_run):
    _, _, table, _ = signal_run
    assert table.loc["unweighted", "skill_ref"] > 0.2 and table.loc["unweighted", "p_ref"] < 0.05
    assert table.loc["unweighted", "skill_ctl"] > 0.2 and table.loc["unweighted", "p_ctl"] < 0.05
    for control in ("random", "shuffled"):
        assert table.loc[control, "mse"] > table.loc["none", "mse"] * 0.8      # controls carry no real information
    assert table.loc["weighted", "corr"] > 0.5


def test_learned_weights_find_the_informative_features_and_beat_uniform(signal_run):
    eng, sched, table, preds = signal_run
    assert sched.diagnostics["accepted"].sum() >= 1
    last = [w for w in sched.weights if w is not None][-1]
    assert last[["sig_a", "sig_b"]].mean() > 2 * last[[c for c in last.index if c.startswith("noise")]].mean()
    assert table.loc["weighted", "mse"] < table.loc["unweighted", "mse"]
    v = W.weighting_verdict(table, preds, seed=1)
    assert v["beats_random"] and v["beats_none"]
    wu = W.weighted_vs_unweighted(preds, seed=1)
    assert wu["skill"] > 0 and wu["n"] >= 40             # honest: ~49 spaced days cannot certify the small gain at p<0.05


def test_weights_never_use_data_after_their_refit_date(signal_run):
    eng, sched, _, _ = signal_run
    r = int(eng.F.index.get_loc(sched.dates[-1]))
    F2, O2 = eng.F.copy(), eng.O.copy()
    F2.iloc[r + 1:] = 7e3
    O2.iloc[r - eng.horizon + 1:] = 7e3                    # outcomes not yet public at r are destroyed too
    s2 = W.learn_weights_walk_forward(engine(F2, O2), [r], seed=5)
    a, b = sched.weights[-1], s2.weights[-1]
    assert (a is None) == (b is None)
    if a is not None:
        assert np.allclose(a.fillna(1).values, b.fillna(1).values)
    assert sched.at(eng.F.index[r]) is sched.weights[-1]
    assert sched.at(eng.F.index[0]) is None and W.WeightSchedule().at(eng.F.index[5]) is None


def test_no_effect_means_no_edge_and_no_verdict():
    F, O = planted(1800, seed=11, signal=False)
    eng = engine(F, O)
    lo = eng.min_hist + eng.gap + eng.horizon + 8 * eng.sep
    W.learn_weights_walk_forward(eng, list(range(lo, 1800, 250)), seed=2)
    pos = W.eval_positions(eng, F.index[lo + 250], None, 21)
    table, preds = W.compare_methods(eng, pos, k=5, seed=3, n_random=6)
    v = W.weighting_verdict(table, preds)
    assert not v["beats_unweighted"]
    assert not (table.loc["weighted", "skill_ref"] > 0.15 and table.loc["weighted", "p_ref"] < 0.01)


def test_predictions_are_deterministic_given_a_seed():
    F, O = planted(900, seed=4)
    eng = engine(F, O)
    pos = W.eval_positions(eng, step=42)
    for m in ("random", "shuffled", "nn_random"):
        a = W.predict_series(eng, pos, m, seed=9, n_random=3)["pred"]
        b = W.predict_series(eng, pos, m, seed=9, n_random=3)["pred"]
        c = W.predict_series(eng, pos, m, seed=10, n_random=3)["pred"]
        assert a.equals(b) and not a.equals(c)
    with pytest.raises(ValueError):
        W.predict_series(eng, pos, "bogus")


def test_hac_t_and_sign_flip_can_fail_and_pass():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 200)
    assert W.hac_t(d) > 4 and W.sign_flip_p(d, 1) < 0.01
    z = rng.normal(0, 1.0, 200)
    assert W.sign_flip_p(z, 1) > 0.05 and abs(W.hac_t(z)) < 3
    assert np.isnan(W.hac_t([1.0, 2.0])) and np.isnan(W.sign_flip_p([1.0]))


def test_price_state_and_outcomes_are_point_in_time():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=700)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700))), idx)
    m = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.008, 700))), idx)
    full = W.price_state(px, m)
    cut = W.price_state(px.iloc[:500], m.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], cut, check_exact=False)
    O = W.forward_outcomes(px, 5, m)
    assert O["fwd_ret_5d"].iloc[-5:].isna().all() and O["fwd_ret_5d"].iloc[-6] == pytest.approx(np.log(px.iloc[-1] / px.iloc[-6]))
    assert (O["fwd_maxdd_5d"].dropna() <= 0).all()


# ---- sector ------------------------------------------------------------------------------------------------------------
def make_prices(n_days=800, seed=0, n_per=6, sectors=("10", "20", "30")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    mkt = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n_days))), idx)
    cols, labels = {}, {}
    for sec in sectors:
        common = rng.normal(0, 0.006, n_days)
        for j in range(n_per):
            tk = f"{sec}_{j}"
            cols[tk] = 50 * np.exp(np.cumsum(common + rng.normal(0, 0.012, n_days)))
            labels[tk] = sec + "99"
    C = pd.DataFrame(cols, idx)
    V = pd.DataFrame(rng.integers(100_000, 500_000, C.shape).astype(float), idx, C.columns)
    return C, V, mkt, pd.Series(labels)


def test_sector_groups_and_index_handle_thin_sectors_and_entries():
    lab = pd.Series({"A": "1011", "B": "1012", "C": "1099", "D": "2011", "E": None, "F": ""})
    assert S.sector_groups(lab, min_members=2) == {"10": ["A", "B", "C"]}
    assert S.sector_groups(pd.Series(dtype=object)) == {}
    idx = pd.bdate_range("2020-01-01", periods=30)
    C = pd.DataFrame({"A": np.linspace(10, 12, 30), "B": np.linspace(20, 22, 30), "C": np.linspace(5, 6, 30)}, idx)
    C.loc[idx[:10], "C"] = np.nan                         # C lists late; before that only 2 names exist
    ix = S.sector_index(C, ["A", "B", "C"], min_names=3)
    assert ix.iloc[:11].isna().all() and ix.iloc[-1] > 0
    assert S.sector_index(C, ["ZZ"]).empty


def test_sector_fingerprints_are_point_in_time_and_complete():
    C, V, mkt, lab = make_prices(700)
    groups = S.sector_groups(lab, min_members=3)
    ctx = pd.DataFrame({"vix": 15 + np.arange(700) * 0.01}, C.index)
    fp = S.sector_fingerprints(C, V, groups, mkt, ctx, horizon=21)
    assert set(fp) == set(groups) and len(fp) == 3
    F, O = next(iter(fp.values()))
    for col in ("ret_3m", "drawdown", "breadth_50", "dispersion", "dv_share_chg", "top_heavy", "rel_1m", "m_vix"):
        assert col in F.columns, col
    assert {"fwd_ret_21d", "fwd_rel_21d", "fwd_vol_21d", "fwd_maxdd_21d"} <= set(O.columns)
    cut = 500
    fp2 = S.sector_fingerprints(C.iloc[:cut], V.iloc[:cut], groups, mkt.iloc[:cut], ctx.iloc[:cut], horizon=21)
    for sec in fp:
        pd.testing.assert_frame_equal(fp[sec][0].iloc[:cut], fp2[sec][0], check_exact=False, rtol=1e-9)


def test_sector_analogs_end_to_end_ranking_panel_and_ablation():
    C, V, mkt, lab = make_prices(1100, seed=5)
    groups = S.sector_groups(lab, min_members=3)
    sa = S.SectorAnalogs(S.sector_fingerprints(C, V, groups, mkt, horizon=21))
    assert sa.sectors() == sorted(groups)
    t = C.index[-30]
    r = sa.find("10", t)
    assert r["analogs"]["date"].max() <= t - pd.Timedelta(days=60)
    assert sa.find("nope", t) is None and sa.find("10", C.index[100]) is None
    rk = sa.rank(t)
    assert len(rk) == 3 and rk["score"].is_monotonic_decreasing
    panel = sa.panel_features([t], {"10_0": "10", "ghost": "zz"})
    assert np.isfinite(panel.loc[(t, "10_0"), "an_sec_fc"]) and np.isnan(panel.loc[(t, "ghost"), "an_sec_fc"])
    table, preds, verdict = sa.ablation("10", step=42, n_random=3)
    assert {"none", "unweighted", "weighted", "random", "shuffled", "nn_random"} == set(table.index)
    assert "beats_unweighted" in verdict


def test_sector_engine_with_planted_twin_episode():
    F, O = planted(1200, signal=False, seed=8)
    F.iloc[1000] = F.iloc[350] + 1e-4
    O.iloc[350, 0] = 0.8
    sa = S.SectorAnalogs({"tech": (F, O)}, target="fwd_ret_21d")
    r = sa.find("tech", F.index[1000])
    assert r["analogs"]["date"].iloc[0] == F.index[350] and r["forecast_return"] > 0.25
    assert sa.find("tech", F.index[1000] + pd.Timedelta(hours=5))["as_of"] == F.index[1000]     # never a later row


# ---- stock -----------------------------------------------------------------------------------------------------------
def test_stock_fingerprint_no_lookahead_and_sufficiency():
    C, V, mkt, _ = make_prices(700, seed=6)
    tk = C.columns[0]
    F, O = T.stock_fingerprint(C[tk], V[tk], mkt, horizon=5)
    F2, _ = T.stock_fingerprint(C[tk].iloc[:450], V[tk].iloc[:450], mkt.iloc[:450], horizon=5)
    pd.testing.assert_frame_equal(F.iloc[:450], F2, check_exact=False, rtol=1e-9)
    assert {"vol_surge", "skew_63", "up_share_21", "max_move_63", "rel_3m"} <= set(F.columns)
    assert T.data_sufficient(F, O, "fwd_ret_5d")
    assert not T.data_sufficient(F.iloc[:200], O.iloc[:200], "fwd_ret_5d")
    assert not T.data_sufficient(F.iloc[:0], O.iloc[:0], "fwd_ret_5d")
    assert not T.data_sufficient(F.iloc[:600], O.iloc[:600], "fwd_ret_5d", min_pool=900)


def test_stock_analogs_gate_short_history_and_answer_for_long():
    C, V, mkt, _ = make_prices(900, seed=7)
    fp = {tk: T.stock_fingerprint(C[tk], V[tk], mkt, 5) for tk in C.columns[:3]}
    short = C.columns[3]
    fp[short] = T.stock_fingerprint(C[short].iloc[-250:], V[short].iloc[-250:], mkt, 5)
    sa = T.StockAnalogs(fp, horizon=5)
    assert short in sa.insufficient and C.columns[0] in sa.engines
    t = C.index[-20]
    r = sa.find(C.columns[0], t, k=4)
    assert r is not None and len(r["analogs"]) <= 4 and r["analogs"]["date"].max() <= C.index[-20 - 63]
    assert sa.find(short, t) is None
    assert sa.find(C.columns[0], C.index[150]) is None
    table, _, verdict = sa.ablation(C.columns[0], step=10, n_random=3)
    assert "weighted" in table.index and "beats_none" in verdict


def pooled_fixture(n=700, n_t=10, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    fp = {}
    for j in range(n_t):
        F = pd.DataFrame(rng.normal(size=(n, 6)), idx, [f"f{c}" for c in range(6)])
        O = pd.DataFrame({"fwd_ret_5d": 0.01 * rng.normal(size=n), "fwd_vol_5d": np.full(n, 0.2),
                          "fwd_maxdd_5d": np.full(n, -0.02)}, idx)
        fp[f"S{j}"] = (F, O)
    return fp, idx


def test_pooled_analogs_find_a_planted_cross_stock_twin():
    fp, idx = pooled_fixture()
    v = np.array([3, -3, 3, -3, 3, -3.0])
    fp["S2"][0].iloc[200] = v
    fp["S2"][1].iloc[200, 0] = 0.4
    fp["S7"][0].iloc[600] = v
    r = T.PooledStockAnalogs(fp, horizon=5).query("S7", idx[600], k=5)
    a = r["analogs"]
    assert a["ticker"].iloc[0] == "S2" and a["date"].iloc[0] == idx[200]
    assert r["forecast_return"] > 0.1
    pos = idx.get_indexer(a["date"])
    assert (600 - pos >= 63).all()
    assert pd.Series(pos // 21).value_counts().max() <= 2                             # window cap holds
    for _, g in a.groupby("ticker"):
        p = idx.get_indexer(g["date"])
        assert all(abs(x - y) >= 21 for i, x in enumerate(p) for y in p[:i])


def test_pooled_query_ignores_the_future_and_degenerate_inputs():
    fp, idx = pooled_fixture(seed=5)
    a = T.PooledStockAnalogs(fp, horizon=5).query("S1", idx[500], k=5)
    for tk in fp:
        fp[tk][0].iloc[501:] = 555.0
        fp[tk][1].iloc[501:] = 555.0
    b = T.PooledStockAnalogs(fp, horizon=5).query("S1", idx[500], k=5)
    assert list(a["analogs"]["date"]) == list(b["analogs"]["date"])
    pa = T.PooledStockAnalogs(pooled_fixture(seed=5)[0], horizon=5)
    assert pa.query("ZZZ", idx[500]) is None and pa.query("S1", idx[100]) is None
    few, _ = pooled_fixture(n_t=4)
    assert T.PooledStockAnalogs(few, horizon=5, min_tickers=8).query("S1", idx[500]) is None     # cross-section too thin
    with pytest.raises(ValueError):
        T.PooledStockAnalogs(few, horizon=5, gap=3)
    X = pa.panel_features([idx[500]], ["S1", "S2"])
    assert list(X.columns) == ["an_stk_fc", "an_stk_conf", "an_stk_uniq"] and len(X) == 2


# ---- market fingerprint, audit, calibration, eras, features, blending, cross-sectional IC ------------------------------
def test_market_context_applies_publication_lags_and_is_point_in_time():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=800)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), idx)
    vix = pd.Series(15 + rng.normal(0, 1, 800), idx)
    macro = pd.DataFrame({"DGS10": np.full(800, 2.0), "BAMLH0A0HYM2": np.full(800, 4.0)}, idx)
    macro.iloc[500:, 0] = 3.0                                  # yield steps up on session 500
    F = W.market_context(spx, vix, vix * 1.1, macro, lags={"DGS10": 5})
    assert {"drawdown", "ret_12m", "vix", "vix_term", "rate_chg_1y", "credit_chg_3m"} <= set(F.columns)
    # the step is public from row 505; rate_chg_1y compares with 252 rows earlier, so it reads 1.0 from row 505 on
    assert F["rate_chg_1y"].iloc[504] == pytest.approx(0.0) and F["rate_chg_1y"].iloc[505] == pytest.approx(1.0)
    macro2 = macro.copy()
    macro2.iloc[600:, 0] = 99.0
    F2 = W.market_context(spx, vix, vix * 1.1, macro2, lags={"DGS10": 5})
    pd.testing.assert_frame_equal(F.iloc[:605], F2.iloc[:605])          # later releases cannot reach earlier rows
    assert "rate_chg_1y" not in W.market_context(spx).columns


def test_audit_flags_planted_violations_and_passes_clean_results():
    F, O = planted(900)
    eng = engine(F, O)
    assert W.audit_engine(eng, [400, 600, 800]) == {}
    r = eng.query(600, k=5)
    assert W.audit_result(r, 600, F.index) == []
    bad = {**r, "analogs": r["analogs"].copy()}
    bad["analogs"].loc[0, "date"] = F.index[590]                          # an analog ten sessions old
    assert "analog closer than gap" in W.audit_result(bad, 600, F.index)
    dup = {**r, "analogs": r["analogs"].copy()}
    dup["analogs"].loc[1, "date"] = dup["analogs"].loc[0, "date"]
    assert "duplicate analog date" in W.audit_result(dup, 600, F.index)
    close = {**r, "analogs": r["analogs"].copy()}
    close["analogs"].loc[1, "date"] = F.index[F.index.get_loc(close["analogs"].loc[0, "date"]) + 5]
    assert "analogs within one episode" in W.audit_result(close, 600, F.index)
    w = {**r, "analogs": r["analogs"].assign(weight=0.5)}
    assert "weights do not sum to 1" in W.audit_result(w, 600, F.index)
    assert W.audit_result(None, 5, F.index) == []


def test_confidence_calibration_detects_informative_and_uninformative_confidence():
    rng = np.random.default_rng(3)
    n = 600
    conf = rng.uniform(0.05, 1.0, n)
    actual = rng.normal(0, 1, n)
    good = pd.DataFrame({"pred": actual + rng.normal(0, 1, n) * (1.2 - conf), "actual": actual, "confidence": conf})
    bad = good.assign(confidence=rng.permutation(conf))
    tab, rho = W.confidence_calibration(good)
    assert rho > 0.3 and tab["mae"].iloc[0] > tab["mae"].iloc[-1] and tab["n"].sum() == n
    assert abs(W.confidence_calibration(bad)[1]) < 0.15
    tab0, rho0 = W.confidence_calibration(good.iloc[:5])
    assert tab0.empty and np.isnan(rho0)
    assert np.isnan(W.confidence_calibration(good.assign(confidence=0.5))[1])


def test_era_breakdown_localises_an_effect_to_its_era():
    rng = np.random.default_rng(4)
    d1 = pd.bdate_range("1995-01-02", periods=80, freq="21B")
    d2 = pd.bdate_range("2015-01-02", periods=80, freq="21B")
    idx = d1.append(d2)
    y = rng.normal(0, 1, len(idx))
    pred = np.where(idx.year < 2000, rng.normal(0, 1, len(idx)), 0.8 * y + rng.normal(0, 0.3, len(idx)))
    preds = {"none": pd.DataFrame({"pred": 0.0, "actual": y}, index=idx),
             "model": pd.DataFrame({"pred": pred, "actual": y}, index=idx)}
    e = W.era_breakdown(preds)
    m = e[e["mode"] == "model"].set_index("era")
    assert m.loc["1990s", "skill_ref"] < 0 and m.loc["2010s", "skill_ref"] > 0.5 and m.loc["2010s", "p_ref"] < 0.01
    thin = W.era_breakdown(preds, eras=[("tiny", "1995-01-01", "1995-03-01")])
    assert thin["skill_ref"].isna().all() and (thin["n"] < 8).all()
    assert W.era_breakdown({"a": pd.DataFrame(columns=["pred", "actual"])}).empty


def test_feature_report_ranks_leaned_on_features_and_stability():
    cols = ["a", "b", "c"]
    w = [pd.Series({"a": 2.0, "b": 0.5, "c": 0.5}), None, pd.Series({"a": 2.5, "b": 0.3, "c": 0.2}),
         pd.Series({"a": 2.2, "b": 0.6, "c": 0.2})]
    rep = W.feature_report(W.WeightSchedule(dates=list(range(4)), weights=w), cols)
    assert rep.index[0] == "a" and rep.loc["a", "share_above_uniform"] == 1.0 and rep.loc["c", "share_above_uniform"] == 0.0
    assert rep["stability"].iloc[0] > 0.8
    empty = W.feature_report(W.WeightSchedule(), cols)
    assert empty["mean_weight"].isna().all() and list(empty.index) == cols


def test_sweep_settings_covers_every_metric_and_k():
    F, O = planted(900, seed=6)
    eng = engine(F, O)
    tab = W.sweep_settings(eng, W.eval_positions(eng, step=42), ks=(3, 8))
    assert len(tab) == 6 and {"euclid_k3", "cosine_k8", "manhattan_k3"} <= set(tab.index)
    assert tab["n"].nunique() == 1 and tab["mse"].notna().all()


def test_blend_forecasts_weights_by_confidence_and_drops_missing_levels():
    idx = pd.bdate_range("2020-01-01", periods=3)
    a = pd.DataFrame({"pred": [0.0, 1.0, np.nan], "confidence": [1.0, 0.0, 0.5], "actual": [0.1, 0.2, 0.3]}, idx)
    b = pd.DataFrame({"pred": [4.0, 5.0, np.nan], "confidence": [3.0, 0.2, 0.9], "actual": [0.1, 0.2, 0.3]}, idx)
    out = W.blend_forecasts({"m": a, "s": b})
    assert out["pred"].iloc[0] == pytest.approx(3.0)                       # (0*1 + 4*3) / 4
    assert out["pred"].iloc[1] == pytest.approx(5.0) and out["n_levels"].iloc[1] == 1     # zero-confidence level ignored
    assert np.isnan(out["pred"].iloc[2]) and out["n_levels"].iloc[2] == 0
    assert W.blend_forecasts({"m": a, "s": b}, min_conf=2.0)["pred"].iloc[0] == pytest.approx(4.0)
    assert list(out["actual"]) == [0.1, 0.2, 0.3]


def test_cross_sectional_ic_detects_a_planted_ranking_and_a_null():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=60)
    Y = pd.DataFrame(rng.normal(size=(60, 12)), dates)
    good = Y + rng.normal(0, 1.0, Y.shape)
    junk = pd.DataFrame(rng.normal(size=Y.shape), dates)
    ic, s = W.cross_sectional_ic(good, Y)
    assert s["mean_ic"] > 0.4 and s["p"] < 0.01 and s["share_pos"] > 0.8 and len(ic) == 60
    assert W.cross_sectional_ic(junk, Y)[1]["p"] > 0.05
    assert W.cross_sectional_ic(good.iloc[:, :3], Y.iloc[:, :3])[1]["n_dates"] == 0            # too few names: no IC invented
    flat = good.copy()
    flat.iloc[:] = 1.0
    assert W.cross_sectional_ic(flat, Y)[0].empty
    assert W.render_report("x", {"t": good.head(2), "e": pd.DataFrame(), "s": "note"}, ["n"]).startswith("# x")


def test_sector_rank_ic_on_planted_sectors():
    fp = {}
    for j in range(5):
        fp[f"s{j}"] = planted(1500, seed=20 + j, n_noise=4)
    sa = S.SectorAnalogs(fp, target="fwd_ret_21d")
    dates = fp["s0"][0].index[600:1450:21]
    ic, s = sa.rank_ic(dates)
    assert s["mean_ic"] > 0.15 and s["n_dates"] > 20 and s["share_pos"] > 0.55
    P, Y = sa.forecast_matrix(dates[:3])
    assert P.shape == (3, 5) and Y.shape == (3, 5)


def test_pooled_rank_ic_planted_and_null():
    rng = np.random.default_rng(9)
    fp, idx = pooled_fixture(n=900, n_t=12, seed=9)
    for tk, (F, O) in fp.items():
        O["fwd_ret_5d"] = 0.02 * F["f0"] + 0.01 * rng.normal(size=len(F))               # planted: f0 drives the outcome
    _, s = T.PooledStockAnalogs(fp, horizon=5).rank_ic(idx[400:880:12], k=8)
    assert s["mean_ic"] > 0.2 and s["p"] < 0.05
    fp0, idx0 = pooled_fixture(n=900, n_t=12, seed=9)
    _, s0 = T.PooledStockAnalogs(fp0, horizon=5).rank_ic(idx0[400:880:12], k=8)
    assert s0["mean_ic"] < 0.1 or s0["p"] > 0.02                                        # null: no IC without an effect


# ---- quantiles, learners, coverage, causality, agreement, ledger, crowding ----------------------------------------------
def test_weighted_quantile_and_forecast_distribution_fields():
    assert W.weighted_quantile([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [0.5])[0] == pytest.approx(3.0)
    assert W.weighted_quantile([1, 100], [1, 0], [0.9])[0] == pytest.approx(1.0)          # zero weight cannot move a quantile
    assert np.isnan(W.weighted_quantile([np.nan], [1.0], [0.5])[0])
    F, O = planted(900)
    r = engine(F, O).query(600, k=7)
    q = r["forecast_quantiles"]
    assert q["p10"] <= q["p50"] <= q["p90"] and 0.0 <= r["prob_up"] <= 1.0
    O2 = O.copy()
    O2["fwd_ret_21d"] = O["fwd_ret_21d"].abs() + 0.001                        # every outcome positive: certainty of up
    up = engine(F, O2).query(700, k=7)
    assert up["prob_up"] == pytest.approx(1.0) and up["forecast_quantiles"]["p10"] > 0


def test_clone_is_independent_and_metric_can_change():
    F, O = planted(700)
    e = engine(F, O)
    e.schedule = W.WeightSchedule(dates=[F.index[10]], weights=[pd.Series(1.0, index=F.columns)])
    c = e.clone(metric="manhattan")
    assert c.metric == "manhattan" and c.schedule.dates == [] and c.F is e.F
    assert c.query(500) is not None


def learner_engine(n=1500, seed=13):
    F, O = planted(n, seed=seed, n_noise=12, noise_phi=0.9)
    e = engine(F, O)
    lo = e.min_hist + e.gap + e.horizon + 8 * e.sep
    return e, list(range(lo, n, 250))


def test_ic_learner_finds_signal_features_and_bad_method_is_refused():
    e, refits = learner_engine()
    s = W.learn_weights_walk_forward(e, refits, method="ic", seed=3)
    assert s.diagnostics["accepted"].sum() >= 1
    w = [x for x in s.weights if x is not None][-1]
    assert w[["sig_a", "sig_b"]].min() > w[[c for c in w.index if c.startswith("noise")]].mean()
    assert w.dropna().min() >= 0.25 * 0.9                                    # floor keeps every feature alive
    with pytest.raises(ValueError):
        W.learn_weights_walk_forward(e.clone(), refits, method="bogus")


def test_weight_learner_null_accepts_less_on_shuffled_outcomes():
    e, refits = learner_engine()
    r = W.weight_learner_null(e, refits, n_shuffles=4, seed=1, method="ic")
    assert r["accept_real"] >= 0.3 and r["accept_real"] > r["accept_null_mean"]
    assert r["accept_null_max"] < 1.0 and r["n_shuffles"] == 4
    a = W.block_shuffle(np.arange(200), 21, np.random.default_rng(0))
    assert sorted(a) == list(range(200)) and not (a == np.arange(200)).all()


def test_compare_learners_scores_every_arm_on_the_same_rows():
    e, refits = learner_engine(1300, seed=14)
    pos = W.eval_positions(e, e.F.index[refits[0] + 250], None, 21)
    tab, preds = W.compare_learners(e, refits, pos, methods=("ic",))
    assert list(tab.index) == ["none", "uniform", "learned_ic"] and tab["n"].nunique() == 1
    assert tab.loc["learned_ic", "skill_ref"] > 0 and set(preds) == set(tab.index)


def test_spec_coverage_lists_missing_items_and_matches_prefixes():
    idx = pd.bdate_range("2020-01-01", periods=50)
    F = pd.DataFrame({"drawdown": np.linspace(-0.1, 0, 50), "vix": np.nan, "mac_DGS10": 1.0}, idx)
    F.loc[idx[10]:, "vix"] = 20.0
    cov = W.spec_coverage(F)
    assert len(cov) == 18 and cov.loc["market drawdown", "present"] and cov.loc["FRED data", "columns"] == "mac_DGS10"
    assert not cov.loc["breadth", "present"] and cov.loc["breadth", "non_nan"] == 0.0
    assert cov.loc["VIX", "first"] == idx[10] and cov.loc["VIX", "non_nan"] == pytest.approx(40 / 50)
    rng = np.random.default_rng(1)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), pd.bdate_range("2015-01-01", periods=600))
    full = W.spec_coverage(W.market_context(spx, spx * 0 + 15, spx * 0 + 16))
    assert full["present"].sum() == 11 and not full.loc["credit-spread change", "present"]


def test_causality_audit_passes_a_causal_builder_and_names_a_leaky_one():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2016-01-01", periods=500)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), idx)
    slicer = lambda d, c: d.iloc[:c]
    assert W.causality_audit(lambda d: W.price_state(d), px, [200, 350], slicer) == []
    centered = lambda d: pd.DataFrame({"ok": d.rolling(20).mean(), "leak": d.rolling(21, center=True).mean()})
    bad = W.causality_audit(centered, px, [200, 350], slicer)
    assert {b[1] for b in bad} == {"leak"} and len(bad) >= 1
    global_norm = lambda d: pd.DataFrame({"z": (d - d.mean()) / d.std()})
    assert W.causality_audit(global_norm, px, [300], slicer)[0][1] == "z"
    shifted = lambda d: pd.DataFrame({"x": d.shift(-1)})                   # NaN pattern differs at the cut edge
    assert W.causality_audit(shifted, px, [300], slicer)


def test_level_agreement_agreement_days_are_more_accurate():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2000-01-03", periods=800, freq="21B")
    y = rng.normal(0, 1, 800)
    a = pd.DataFrame({"pred": y + rng.normal(0, 1.5, 800), "actual": y}, idx)
    b = pd.DataFrame({"pred": y + rng.normal(0, 1.5, 800), "actual": y}, idx)
    out = W.level_agreement({"market": a, "sector": b})
    assert out["hit_agree"] > out["hit_conflict"] + 0.05 and 0.0 < out["conflict_share"] < 0.6
    assert out["sign_agreement"].loc["market", "market"] == 1.0 and out["corr"].loc["market", "sector"] > 0.3
    assert W.level_agreement({"m": a.iloc[:0], "s": b.iloc[:0]})["n"] == 0


def test_ledger_flatten_and_idempotent_append(tmp_path):
    F, O = planted(900)
    eng = engine(F, O)
    res = [eng.query(600, k=4), None, eng.query(700, k=4)]
    fr = W.results_to_frame(res, "market", key=["a", "b", "c"])
    assert len(fr) == 8 and set(fr["key"]) == {"a", "c"} and "out_fwd_ret_21d" in fr and fr["rank"].max() == 3
    assert (fr["analog_date"] < fr["as_of"]).all() and W.results_to_frame([None], "x").empty
    path = tmp_path / "ledger" / "l.parquet"
    assert W.append_ledger(path, fr) == 8
    assert W.append_ledger(path, fr) == 8                                     # re-running replaces, never duplicates
    other = W.results_to_frame([eng.query(600, k=4)], "sector", key=["a"])
    assert W.append_ledger(path, other) == 12 and W.append_ledger(path, fr.iloc[:0]) == 0
    back = pd.read_parquet(path)
    assert set(back["tag"]) == {"market", "sector"}


def test_breadth_dispersion_crowding_and_stock_sector_features_are_causal():
    C, V, mkt, lab = make_prices(500, seed=12)
    up = C.copy()
    up.iloc[:] = np.arange(1, len(C) + 1)[:, None] * 1.0 + np.arange(C.shape[1])[None, :]
    b = S.breadth_dispersion_crowding(up, V, lab)
    assert b["breadth"].dropna().iloc[-1] == pytest.approx(1.0)               # every name above its own 50-day mean
    V2 = V.copy()
    V2.loc[V2.index[250]:, [c for c in V.columns if c.startswith("10_")]] *= 50.0     # sector 10 takes over the tape
    hhi = S.breadth_dispersion_crowding(C, V2, lab)["sector_crowding"]
    assert hhi.iloc[-1] > hhi.iloc[200] + 0.1 and S.breadth_dispersion_crowding(C, V, pd.Series(dtype=object)).columns.tolist() == ["breadth", "dispersion"]
    build = lambda d: S.breadth_dispersion_crowding(d[0], d[1], lab)
    assert W.causality_audit(build, (C, V), [250, 400], lambda d, c: (d[0].iloc[:c], d[1].iloc[:c])) == []
    tk = C.columns[0]
    sidx = S.sector_index(C, [c for c in C.columns if c.startswith("10_")])
    F, _ = T.stock_fingerprint(C[tk], V[tk], mkt, 5, sector_px=sidx)
    assert {"rel_sector_1m", "rel_sector_3m", "corr_sector"} <= set(F.columns)
    Fc, _ = T.stock_fingerprint(C[tk].iloc[:300], V[tk].iloc[:300], mkt.iloc[:300], 5, sector_px=sidx.iloc[:300])
    pd.testing.assert_frame_equal(F.iloc[:300], Fc, check_exact=False, rtol=1e-9)
