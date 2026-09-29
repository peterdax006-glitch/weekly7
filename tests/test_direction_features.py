"""Direction research (B25; Bible V2/V5, canons C23/C24/C56): labels, walk-forward protocol, frontier statistics, controls.
Synthetic data only. The planted world must be found at the coverage where the signal lives, the null world must stay
near 50% at every coverage, and a look-ahead feature must be caught by the audit."""
import numpy as np
import pandas as pd
import pytest

from engine import direction_features as D


# ------------------------------------------------------------------ synthetic worlds
def make_pool(seed=0, years=range(2014, 2022), per_week=60, signal=None, acc=0.9, sig_frac=0.10):
    """Weekly pool of picks. `signal`: None = null world; 'earn' = the feature states the direction with prob `acc` on
    the earnings segment only (~sig_frac of rows); 'all' = on every row."""
    rng = np.random.default_rng(seed)
    wk = pd.bdate_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="W-FRI")
    date = np.repeat(wk.values, per_week)
    n = len(date)
    seg = np.where(rng.uniform(size=n) < sig_frac, "earn", np.where(rng.uniform(size=n) < 0.3, "filing", "none"))
    up = (rng.uniform(size=n) < 0.52).astype(float)
    reg = np.where(rng.uniform(size=n) < 0.5, "bull", "bear")
    pool = pd.DataFrame({"date": pd.DatetimeIndex(date), "seg": seg, "reg": reg, "up": up})
    pool["year"] = pool["date"].dt.year
    pool["pick"] = True
    noise = rng.standard_normal((n, 3)).astype("float32")
    mask = (seg == "earn") if signal == "earn" else np.ones(n, bool) if signal == "all" else np.zeros(n, bool)
    sig = D.plant_signal(up, mask, acc=acc, seed=seed + 1)
    F = {"pead": np.column_stack([sig, noise[:, 0]]), "insider": noise[:, [1]], "reversal_mom": noise[:, [2]],
         "event_regime": np.column_stack([(seg == "earn"), (seg == "filing")]).astype("float32")}
    F["combined"] = np.column_stack([F[k] for k in ("pead", "insider", "reversal_mom", "event_regime")])
    return pool, F


def run(pool, F, years=range(2018, 2022), **kw):
    return D.walk_forward(pool, F, "up", years, seed=3, **kw)


# ------------------------------------------------------------------ labels
def bars(rows):
    """rows: list of (open, high, low, close) per session; one ticker."""
    a = np.array(rows, float)
    return tuple(a[:, [i]] for i in range(4))


def test_first_touch_up_before_down_and_entry_is_next_open():
    O, H, L, C = bars([(100, 101, 99, 100)] + [(100, 100, 99, 100)] + [(100, 111, 99, 105)] + [(105, 106, 89, 90)] + [(90, 91, 89, 90)] * 3)
    lab = D.weekly_labels(O, H, L, C, np.array([0]))
    assert lab["mover"][0, 0] and lab["up_first"][0, 0] == 1.0
    assert lab["entry"][0, 0] == 100.0                                   # open of session 1, not the close of session 0
    assert lab["fwd"][0, 0] == pytest.approx(90 / 100 - 1)


def test_down_first_and_same_session_double_touch_is_ambiguous():
    O, H, L, C = bars([(100, 100, 100, 100), (100, 100, 89, 95)] + [(95, 96, 94, 95)] * 5)
    assert D.weekly_labels(O, H, L, C, np.array([0]))["up_first"][0, 0] == 0.0
    O, H, L, C = bars([(100, 100, 100, 100), (100, 112, 88, 100)] + [(100, 101, 99, 100)] * 5)
    lab = D.weekly_labels(O, H, L, C, np.array([0]))
    assert lab["mover"][0, 0] and lab["amb"][0, 0] and np.isnan(lab["up_first"][0, 0])


def test_labels_ignore_bars_outside_the_window_and_a_gap_invalidates():
    rows = [(100, 100, 100, 100)] + [(100, 103, 98, 101)] * 5 + [(101, 150, 60, 100)]
    O, H, L, C = bars(rows)
    base = D.weekly_labels(O, H, L, C, np.array([0]))
    assert not base["mover"][0, 0]                                       # the huge bar at session 6 is outside sessions 1-5
    H2 = H.copy(); H2[0, 0] = 500.0                                      # the decision bar itself is not in the window either
    assert not D.weekly_labels(O, H2, L, C, np.array([0]))["mover"][0, 0]
    H3 = H.copy(); H3[3, 0] = np.nan
    assert not D.weekly_labels(O, H3, L, C, np.array([0]))["ok"][0, 0]


def test_week_end_positions_skips_the_incomplete_last_week():
    s = pd.bdate_range("2024-01-01", "2024-01-19")                       # three full weeks, Friday the 19th is last
    pos = D.week_end_positions(s)
    assert list(s[pos].strftime("%a")) == ["Fri", "Fri"] and s[-1] not in s[pos]


def test_labels_frame_entry_is_after_decision_and_audit_passes():
    rng = np.random.default_rng(0)
    s = pd.bdate_range("2020-01-01", periods=60)
    C = 100 * np.cumprod(1 + rng.normal(0, .02, (60, 4)), axis=0)
    lab = D.weekly_labels(C, C * 1.01, C * .99, C, D.week_end_positions(s))
    lf = D.labels_frame(s, list("ABCD"), lab)
    assert len(lf) > 20 and D.audit_point_in_time(pd.DataFrame(), lf)["ok"]
    bad = lf.copy(); bad["entry_date"] = bad.index.get_level_values(0)
    assert not D.audit_point_in_time(pd.DataFrame(), bad)["ok"]


# ------------------------------------------------------------------ segments / features
def test_event_type_precedence_and_regime():
    X = pd.DataFrame({"days_since_earn": [2, 300, 300, 300], "earn_in_week": [0, 1, 0, 0], "news5": [0, 0, 1, 0],
                      "ev_offering": [0, 0, 0, 0], "ev_shelf": [0, 0, 0, 0], "ev_activist": [0, 0, 0, 0],
                      "ev_activist_amend": [0, 0, 0, 0], "ev_agreement": [0, 0, 0, 0], "ev_red_flag": [0, 0, 0, 0],
                      "m_spy_ma200": [0.1, -0.1, np.nan, 0.0]})
    assert list(D.event_type(X)) == ["earn", "earn", "filing", "none"]
    assert list(D.regime_type(X)) == ["bull", "bear", "na", "bear"]


def test_ear_recent_zeroes_stale_reactions():
    X = pd.DataFrame({"ear": [0.05, 0.05, np.nan], "days_since_earn": [3, 40, 3], "ear_volsurge": [2.0, 2.0, 1.0],
                      "r5_nonews": [0.1, 0, 0], "r5_news": [0, 0, 0], "earn_in_week": [0, 0, 0], "news5": [0, 0, 0]})
    d = D.add_derived(X)
    assert list(d["ear_recent"]) == [0.05, 0.0, 0.0] and d["rev5_news_adj"].iloc[0] == pytest.approx(-0.1)


def test_xs_rank_is_per_date_and_market_columns_pass_through():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-03", "2020-01-10"]), list("abc")], names=["date", "ticker"])
    X = pd.DataFrame({"r5": [1, 2, 3, 10, 30, 20], "m_vix": [15, 15, 15, 30, 30, 30]}, index=idx, dtype=float)
    R = D.xs_rank(X, ["r5", "m_vix"])
    assert list(R["r5"].round(3)) == [-0.167, 0.167, 0.5, -0.167, 0.5, 0.167] or R["r5"].iloc[2] == R["r5"].iloc[4]
    assert list(R["m_vix"]) == [15, 15, 15, 30, 30, 30]


# ------------------------------------------------------------------ movers
def test_mover_walk_forward_scores_only_test_years_from_past_fits_and_finds_the_plant():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", "2019-12-31", freq="W-FRI")
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(80)]], names=["date", "ticker"])
    R = pd.DataFrame(rng.uniform(-.5, .5, (len(idx), 4)), index=idx, columns=list("abcd")).astype("float32")
    y = pd.Series((R["a"].to_numpy() + rng.normal(0, .3, len(idx)) > 0.35).astype(float), index=idx)
    s, log = D.mover_walk_forward(R, y, [2018, 2019], params=dict(n_estimators=40, n_jobs=2))
    yr = idx.get_level_values(0).year
    assert s[yr < 2018].isna().all() and s[yr >= 2018].notna().all()
    assert log["fitted"].all() and pd.Timestamp(log["train_end"].iloc[0]) < pd.Timestamp("2018-01-01") - pd.Timedelta(days=14)
    from scipy.stats import spearmanr
    assert spearmanr(s[yr >= 2018], R["a"][yr >= 2018])[0] > 0.6


def test_mover_score_cannot_see_the_test_year_labels():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2015-01-01", "2018-12-31", freq="W-FRI")
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(60)]], names=["date", "ticker"])
    R = pd.DataFrame(rng.uniform(-.5, .5, (len(idx), 3)), index=idx, columns=list("abc")).astype("float32")
    y = pd.Series((rng.uniform(size=len(idx)) < 0.15).astype(float), index=idx)
    p = dict(n_estimators=30, n_jobs=2)
    s1, _ = D.mover_walk_forward(R, y, [2018], params=p)
    y2 = y.copy(); y2[idx.get_level_values(0).year == 2018] = 1.0 - y2[idx.get_level_values(0).year == 2018]
    s2, _ = D.mover_walk_forward(R, y2, [2018], params=p)
    assert np.allclose(s1.dropna(), s2.dropna())


def test_select_picks_takes_top_per_date_and_ignores_nan():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-03"]), list("abcde")], names=["date", "ticker"])
    sc = pd.Series([0.1, 0.9, np.nan, 0.5, 0.7], index=idx)
    pk = D.select_picks(sc, n_pick=2, n_pool=3)
    assert list(pk["pick"]) == [False, True, False, False, True] and list(pk["pool"]) == [False, True, False, True, True]


# ------------------------------------------------------------------ walk-forward protocol
def test_blocks_are_point_in_time_and_audit_passes():
    pool, F = make_pool()
    pred, blocks = run(pool, F)
    assert blocks["fitted"].all() and D.audit_point_in_time(blocks)["ok"]
    for b in blocks.itertuples():
        assert pd.Timestamp(b.train_end) + pd.Timedelta(days=14) < pd.Timestamp(b.calib_start)
        assert pd.Timestamp(b.calib_end) < pd.Timestamp(year=b.year, month=1, day=1)
    assert (pred["year"] >= 2018).all() and pred["q"].between(0, 1).all()


def test_audit_catches_overlapping_blocks():
    bad = pd.DataFrame([dict(year=2019, fitted=True, train_end="2018-03-01", calib_start="2018-03-03",
                             calib_end="2018-12-21", test_start="2019-01-04")])
    r = D.audit_point_in_time(bad)
    assert not r["ok"] and any("embargo" in v or "still open" in v for v in r["violations"])


def test_planted_signal_in_one_event_type_is_found_at_that_coverage():
    pool, F = make_pool(signal="earn", acc=0.92, sig_frac=0.10)
    pred, _ = run(pool, F)
    ft = D.frontier(pred, "pead:linear", coverages=(0.05, 0.10, 1.0), B=300)
    at5 = ft[ft["cov_target"] == 0.05].iloc[0]
    assert at5["acc"] > 0.85 and at5["lo"] > 0.80                       # the planted rows sit at the top of the confidence order
    assert 0.5 < ft[ft["cov_target"] == 1.0].iloc[0]["acc"] < 0.62      # diluted to ~ (0.9*0.10 + 0.52*0.9) overall
    seg = D.frontier(pred, "pead:linear", (0.5, 1.0), segcol="seg", seg="earn", B=300).iloc[-1]
    assert seg["acc"] > 0.88
    other = D.frontier(pred, "pead:linear", (1.0,), segcol="seg", seg="none", B=300).iloc[0]
    assert other["acc"] < 0.60                                           # and it is absent where it was not planted
    q = D.eighty_question(D.frontier_table(pred, coverages=D.GRID, B=200))
    assert q["reached_lo"] > 0 and q["control_hits"] == 0


def test_null_world_stays_at_chance_at_every_coverage_and_controls_agree():
    pool, F = make_pool(signal=None, seed=5)
    pred, _ = run(pool, F)
    ft = D.frontier_table(pred, coverages=D.GRID, seg_cols=("seg",), B=300)
    ov = ft[ft["segcol"].isna() & (ft["n"] >= 200)]
    base = max(pool["up"].mean(), 1 - pool["up"].mean())
    assert (ov["acc"] < base + 0.06).all()
    assert (ov["lo"] < 0.60).all() and (ov["lo"] < 0.80).all()
    for m in ("ctl_random", "ctl_shuffled"):
        c = ov[(ov["model"] == m) & (ov["cov_target"] == 1.0)].iloc[0]
        assert 0.45 < c["acc"] < base + 0.05                              # a control is never better than the base rate
    e = D.eighty_question(ft)
    assert e["reached_lo"] == 0 and e["reached_lo_adj"] == 0


def test_lookahead_canary_is_caught_by_accuracy_and_by_control_of_shuffle():
    pool, F = make_pool(signal="all", acc=1.0)                           # a feature that IS the future label
    pred, _ = run(pool, F)
    ov = D.frontier(pred, "pead:gbm", (1.0,), B=100).iloc[0]
    assert ov["acc"] > 0.99                                              # a leak reads as near-perfect ...
    sh = D.frontier(pred, "ctl_shuffled", (1.0,), B=100).iloc[0]
    assert sh["acc"] < 0.6                                               # ... while the shuffled-label control on the same features does not


def test_calibration_table_shows_skill_only_where_there_is_signal():
    pool, F = make_pool(signal="all", acc=0.75, seed=4)
    pred, _ = run(pool, F)
    cal = D.calibration_table(pred, coverages=(1.0,)).set_index("model")
    assert cal.loc["pead:linear", "skill"] > 0.03
    null_pool, Fn = make_pool(signal=None, seed=4)
    pn, _ = run(null_pool, Fn)
    cn = D.calibration_table(pn, coverages=(1.0,)).set_index("model")
    assert cn.loc["combined:linear", "skill"] < 0.01 and abs(cn.loc["ctl_random", "skill"]) < 0.005


def test_segment_thresholds_use_the_calibration_block_of_the_same_segment():
    pool, F = make_pool(signal="earn", acc=0.92)
    pred, _ = run(pool, F)
    d = pred[(pred["model"] == "pead:linear") & (pred["seg"] == "earn")]
    assert 0.0 <= d["q_seg"].min() and d["q_seg"].max() <= 1.0
    top = d[d["q_seg"] <= 0.5]
    assert len(top) > 100 and ((top["p"] >= 0.5) == (top["up"] > 0.5)).mean() > 0.85


# ------------------------------------------------------------------ degenerate cases
def test_empty_pool_and_too_little_history_return_empty_results_not_errors():
    pool, F = make_pool(years=range(2014, 2016))
    pred, blocks = run(pool, F, years=[2019])
    assert pred.empty and not blocks["fitted"].any()
    ft = D.frontier(pred, "pead:linear", (0.05, 1.0))
    assert (ft["n"] == 0).all() and ft["acc"].isna().all()
    assert D.eighty_question(D.frontier_table(pred))["reached_lo"] == 0
    assert D.calibration_table(pred).empty and D.era_table(pred).empty


def test_constant_features_never_bet_below_full_coverage():
    pool, F = make_pool()
    F = {k: np.zeros_like(v) for k, v in F.items()}
    pred, _ = run(pool, F, models=("linear",), controls=False)
    ft = D.frontier(pred, "pead:linear", (0.01, 0.10, 1.0), B=100)
    assert list(ft["n"].iloc[:2]) == [0, 0] and ft["n"].iloc[2] == len(pred[pred["model"] == "pead:linear"])


def test_one_class_training_block_is_skipped():
    pool, F = make_pool()
    pool["up"] = 1.0
    pred, blocks = run(pool, F)
    assert pred.empty and not blocks["fitted"].any()


# ------------------------------------------------------------------ statistics and helpers
def test_bootstrap_interval_covers_a_known_accuracy_and_widens_with_clustering():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-05", periods=200, freq="W-FRI")
    n = 30
    d = pd.DataFrame({"model": "m", "date": np.repeat(dates, n), "p": 0.9, "q": 0.5})
    d["up"] = (rng.uniform(size=len(d)) < 0.7).astype(float)             # accuracy of always-up at 0.9 is ~0.7
    r = D.frontier(d, "m", (1.0,), B=800).iloc[0]
    assert r["lo"] < 0.70 < r["hi"] and abs(r["acc"] - 0.7) < 0.03
    week_shock = np.repeat(rng.uniform(size=200) < 0.5, n)               # whole weeks right or wrong together
    d["up"] = np.where(week_shock, 1.0, (rng.uniform(size=len(d)) < 0.3).astype(float))
    c = D.frontier(d, "m", (1.0,), B=800).iloc[0]
    iid_width = 2 * 1.96 * np.sqrt(c["acc"] * (1 - c["acc"]) / c["n"])
    assert (c["hi"] - c["lo"]) > 3 * iid_width                           # ignoring week clustering would be far too confident


def test_eighty_question_requires_lower_bound_and_min_bets_and_ignores_controls():
    ft = pd.DataFrame([
        dict(model="a", segcol=None, seg=None, cov_target=0.01, n=100, cov_real=.01, acc=.85, lo=.81, hi=.9, lo_adj=.7, wilson_lo=.8, up_rate=.5, weeks=50),
        dict(model="b", segcol=None, seg=None, cov_target=0.01, n=100, cov_real=.01, acc=.85, lo=.78, hi=.9, lo_adj=.7, wilson_lo=.8, up_rate=.5, weeks=50),
        dict(model="c", segcol=None, seg=None, cov_target=0.01, n=10, cov_real=.01, acc=.95, lo=.9, hi=1.0, lo_adj=.9, wilson_lo=.8, up_rate=.5, weeks=5),
        dict(model="ctl_random", segcol=None, seg=None, cov_target=0.01, n=100, cov_real=.01, acc=.9, lo=.85, hi=.95, lo_adj=.85, wilson_lo=.8, up_rate=.5, weeks=50)])
    q = D.eighty_question(ft)
    ft = pd.concat([ft, ft.iloc[[0]].assign(model="planted:linear"), ft.iloc[[0]].assign(model="leak:gbm")], ignore_index=True)
    q = D.eighty_question(ft)
    assert q["positive_controls_found"] and q["reached_lo"] == 1 and q["reached_lo_adj"] == 0 and q["control_hits"] == 1 and q["hits"][0]["model"] == "a"


def test_plant_signal_matches_requested_accuracy_and_mask():
    rng = np.random.default_rng(0)
    up = (rng.uniform(size=20000) < 0.5).astype(float)
    mask = rng.uniform(size=20000) < 0.1
    s = D.plant_signal(up, mask, acc=0.8, seed=1)
    assert (s[~mask] == 0).all()
    assert abs(((s[mask] > 0) == (up[mask] > 0.5)).mean() - 0.8) < 0.03


def test_miner_hook_scores_a_planted_direction_pattern_past_only():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2014-01-03", "2019-12-31", freq="W-FRI")
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(60)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), 5)).astype("float32"), index=idx, columns=[f"f{i}" for i in range(5)])
    ydir = np.where(X["f0"].to_numpy() + rng.normal(0, .8, len(idx)) > 0, 1.0, -1.0)
    date = idx.get_level_values(0).values
    fn = D.miner_hook(X, ydir, date, {"max_rows": 30000, "null_reps": 1, "horizon": 5})
    tr = np.flatnonzero(date < np.datetime64("2018-01-01"))
    ca = np.flatnonzero((date >= np.datetime64("2018-01-01")) & (date < np.datetime64("2019-01-01")))
    te = np.flatnonzero(date >= np.datetime64("2019-01-01"))[:1200]
    rc, rt, is_prob = fn(tr, ca[:1200], te)
    assert is_prob is False and len(rt) == len(te)
    assert np.corrcoef(rt, X["f0"].to_numpy()[te])[0, 1] > 0.3
