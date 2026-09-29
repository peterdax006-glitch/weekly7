"""Bible Phase 7: the heavy-test harness on synthetic panels with known effects, planted defects and degenerate input."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import heavy_tests as ht
from engine import pattern_movers as pm

WEEKS, STOCKS, NF = 300, 60, 6
CFG = {"min_train_dates": 80, "step_dates": 55, "gap_days": 10, "topk": 8, "min_era_dates": 20,
       "movement": True, "miner": {"max_pairs": 60, "max_unless": 10, "null_reps": 1, "min_n": 100, "half_life_years": 50}}


def make(seed=0, edge=0.012, noise_only=False, start="2003-01-06"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=WEEKS * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(STOCKS)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), NF)), index=idx, columns=[f"f{i}" for i in range(NF)])
    reg = pd.Series(rng.standard_normal(WEEKS), index=dates)
    X["m_vix"] = reg.reindex(idx.get_level_values(0)).values
    X["m_spy_ma200"] = pd.Series(np.sin(np.arange(WEEKS) / 25.0), index=dates).reindex(idx.get_level_values(0)).values
    q = X[[f"f{i}" for i in range(NF)]].groupby(level=0).rank(pct=True)
    y = pd.Series(rng.standard_normal(len(idx)) * 0.04, index=idx)
    y += np.repeat(rng.normal(0, 0.015, WEEKS), STOCKS)
    if not noise_only:
        y += edge * (q["f0"] >= 0.8) - 0.6 * edge * (q["f1"] <= 0.2)
    return X, y


@pytest.fixture(scope="module")
def planted():
    return make()


@pytest.fixture(scope="module")
def summary(planted, tmp_path_factory):
    X, y = planted
    return ht.run_heavy(X, y, CFG, seeds=(3, 5), null_seeds=(101,), n_hidden=8, hidden_len=30,
                        out_dir=tmp_path_factory.mktemp("rep"))


def test_planted_edge_is_found_out_of_sample(summary):
    o = summary["overall"]
    assert summary["ok"] and o["mean"] > 0.03 and o["t_nw"] > 3
    assert o["top_net"] > 0 and summary["false_discovery"]["oos_false_rate"] < 0.5
    assert (summary["by_era"]["rank_ic"] > 0).all()


def test_noise_control_is_flat_and_real_beats_it(summary):
    n = summary["null"]
    assert len(n) == 1 and abs(n["rank_ic"].iloc[0]) < 0.02 and abs(n["t_nw"].iloc[0]) < 2.5
    assert summary["verdict"]["gates"]["beats_noise_control"]["pass"]


def test_pure_noise_fails_the_verdict():
    X, y = make(seed=4, noise_only=True)
    s = ht.run_heavy(X, y, CFG, seeds=(3,), null_seeds=(101,), n_hidden=4, hidden_len=30)
    assert s["ok"] and not s["verdict"]["pass"]
    assert "mean_ic" in s["verdict"]["failed"] or "t_newey_west" in s["verdict"]["failed"]
    assert abs(s["overall"]["mean"]) < 0.02


def test_verdict_needs_every_gate_not_just_the_average():
    good = {"overall": {"mean": 0.05, "t_nw": 5.0, "top_net": 0.002},
            "by_era": pd.DataFrame({"rank_ic": [0.08, 0.07, -0.05], "enough": [True] * 3}, index=["a", "b", "c"]),
            "false_discovery": {"oos_false_rate": 0.2, "survival_next_mean": 0.6},
            "seed_table": pd.DataFrame({"rank_ic": [0.05, 0.04]}), "regime_gap": {"vol": 0.02},
            "null": pd.DataFrame({"rank_ic": [0.0], "t_nw": [0.4]})}
    v = ht.verdict(good)
    assert not v["pass"] and v["failed"] == ["worst_era_ic"]         # one dead era sinks a great average
    good["by_era"].loc["c", "rank_ic"] = 0.03
    assert ht.verdict(good)["pass"]
    good["regime_gap"] = {"vol": 0.20}
    assert ht.verdict(good)["failed"] == ["regime_sensitivity"]
    good["regime_gap"] = {"vol": 0.02}
    good["null"] = pd.DataFrame({"rank_ic": [0.05], "t_nw": [4.0]})
    assert ht.verdict(good)["failed"] == ["beats_noise_control"]
    good["null"] = pd.DataFrame()
    assert "beats_noise_control" in ht.verdict(good)["failed"]      # no control run = no pass


def test_no_lookahead_future_data_cannot_change_past_results(planted):
    X, y = planted
    ud = X.index.get_level_values(0).unique()
    d = X.index.get_level_values(0)
    Xs, ys = X[d <= ud[200]], y[d <= ud[200]]
    a = ht.walk_forward(X, y, CFG, seed=3)["frame"]
    b = ht.walk_forward(Xs, ys, CFG, seed=3)["frame"]
    a = a[a["date"] <= ud[150]].reset_index(drop=True)
    b = b[b["date"] <= ud[150]].reset_index(drop=True)
    assert len(b) > 30
    pd.testing.assert_frame_equal(a[["date", "rank_ic", "top_excess"]], b[["date", "rank_ic", "top_excess"]])


def test_leaked_future_is_caught_by_the_purge_check(planted):
    """A schedule whose gap is negative would let training touch the test block; the guard must stop it."""
    X, _ = planted
    with pytest.raises(AssertionError):
        ht.origin_schedule(X.index.get_level_values(0).unique(), {**ht.HEAVY_DEFAULT, **CFG, "gap_days": -30})


def test_schedule_purges_every_origin(planted):
    X, _ = planted
    s = ht.origin_schedule(X.index.get_level_values(0).unique(), {**ht.HEAVY_DEFAULT, **CFG})
    assert len(s) >= 3 and (s["purge_days"] >= 10).all() and (s["train_end"] < s["test_start"]).all()


def test_newey_west_is_more_cautious_than_naive_on_autocorrelated_series():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(400); x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.8 * x[i - 1] + e[i]
    x += 0.15
    st = ht.series_stats(x, lag=8)
    assert abs(st["t_nw"]) < abs(st["t"]) * 0.7
    assert np.isnan(ht.newey_west_t([1.0, 2.0]))


def test_decay_fit_recovers_planted_half_life():
    age = np.arange(0, 60, dtype=float)
    f = ht.decay_fit(age, 0.06 - 0.001 * age)
    assert abs(f["half_life_weeks"] - 30) < 0.5 and f["slope_per_week"] < 0
    assert np.isnan(ht.decay_fit(age, np.full(60, 0.03))["half_life_weeks"])


def test_planted_decay_shows_in_age_table():
    rng = np.random.default_rng(1)
    age = rng.uniform(0, 60, 900)
    f = pd.DataFrame({"age_weeks": age, "rank_ic": 0.08 - 0.0015 * age + rng.normal(0, 0.03, 900), "top_net": 0.0})
    tbl, fit = ht.by_age(f)
    assert tbl["rank_ic"].iloc[0] > tbl["rank_ic"].iloc[-1] and 20 < fit["half_life_weeks"] < 45


def test_regime_tags_use_only_the_past():
    X, _ = make()
    t1 = ht.regime_tags(X)
    X2 = X.copy()
    cut = X.index.get_level_values(0).unique()[200]
    X2.loc[X2.index.get_level_values(0) > cut, "m_vix"] = 99.0
    t2 = ht.regime_tags(X2)
    pd.testing.assert_frame_equal(t1[t1.index <= cut], t2[t2.index <= cut])
    assert set(t1["vol"]) <= {"high_vol", "low_vol", "unk"} and (t1["vol"].iloc[:20] == "unk").all()
    assert set(t1["trend"]) == {"bull", "bear"}


def test_regime_table_flags_a_planted_regime_dependent_edge():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2010-01-04", periods=400)
    tags = pd.DataFrame({"vol": np.where(np.arange(400) % 2 == 0, "high_vol", "low_vol")}, index=dates)
    ic = np.where(tags["vol"] == "high_vol", 0.10, -0.02) + rng.normal(0, 0.02, 400)
    fr = pd.DataFrame({"date": dates, "rank_ic": ic, "ic": ic, "top_excess": ic / 10, "top_net": ic / 10, "turnover": 0.3,
                       "cost": 0.0003, "dir_n": 50, "dir_hit": 26, "era": "modern"})
    tbl, gap = ht.by_regime(fr, tags)
    assert gap["vol"] > 0.1 and tbl.loc[("vol", "low_vol"), "rank_ic"] < 0


def test_era_labels_and_insufficient_era_is_marked():
    d = pd.to_datetime(["1999-05-03", "2001-04-09", "2001-04-06", "2010-01-04", "2020-01-06"])
    assert list(ht.era_of(d)) == ["pre_decimal", "post_decimal", "pre_decimal", "post_electronic", "modern"]
    f = pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=10), "rank_ic": 0.1, "ic": 0.1, "top_excess": 0.0,
                      "top_net": 0.0, "turnover": 0.1, "cost": 0.0, "dir_n": 0, "dir_hit": 0, "era": "modern"})
    assert not ht.by_era(f, {"min_era_dates": 20}).loc["modern", "enough"]


def test_validate_panel_rejects_broken_input(planted):
    X, y = planted
    ok = ht.validate_panel(X, y)
    assert ok["dates"] == WEEKS and ok["features"] == NF
    with pytest.raises(ValueError, match="duplicated"):
        ht.validate_panel(pd.concat([X.iloc[:5], X.iloc[:5]]), pd.concat([y.iloc[:5], y.iloc[:5]]))
    with pytest.raises(ValueError, match="sorted"):
        ht.validate_panel(X.iloc[::-1], y.iloc[::-1])
    Xi = X.copy(); Xi.iloc[3, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        ht.validate_panel(Xi, y)
    with pytest.raises(ValueError, match="share"):
        ht.validate_panel(X, y.iloc[1:])


def test_oos_check_marks_a_reversed_pattern_as_wrong_sign():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-05", periods=60)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(50)]], names=["date", "ticker"])
    X = pd.DataFrame({"f0": rng.standard_normal(len(idx))}, index=idx)
    q = X["f0"].groupby(level=0).rank(pct=True)
    y = pd.Series(rng.standard_normal(len(idx)) * 0.02 - 0.03 * (q >= 0.8), index=idx)
    bank = pm.Bank(["f0"], [], [("s", 0, 4)], ["f0 q4"], np.array([0.03]), [None])     # mined as positive, now reversed
    dc, _ = pm._date_codes(idx)
    yex = pm.demeaned(y).to_numpy()
    r = ht._oos_pattern_check(bank, X, yex, dc)
    assert r["wrong_sign"].all() and not r["held"].any()
    bank.effects[:] = -0.03
    assert ht._oos_pattern_check(bank, X, yex, dc)["held"].all()


def test_costs_and_turnover_are_accounted(summary):
    cs = summary["cost_sensitivity"]
    assert cs["net"].is_monotonic_decreasing and cs["cost_bps"].iloc[0] == 0
    f = summary["frame"]
    assert (f["turnover"].dropna().between(0, 1)).all() and (f["cost"].dropna() >= 0).all()
    assert f["top_net"].mean() <= f["top_excess"].mean() + 1e-12


def test_seed_table_hidden_windows_and_origins(summary):
    assert list(summary["seed_table"]["seed"]) == [3, 5] and summary["n_origins"] >= 3
    h = summary["hidden"]
    assert len(h) == 8 and h["dates"].between(25, 30).all()
    pd.testing.assert_frame_equal(h, ht.hidden_windows(summary["frame"], 8, 30, 3))
    o = summary["origins"]
    assert (o["n_live"] > 0).all() and o["oos_checked"].sum() > 0


def test_report_files_roundtrip_with_provenance(summary, tmp_path):
    jp, mp = ht.write_report(summary, tmp_path)
    j = json.loads(jp.read_text(encoding="utf-8"))
    rid = summary["provenance"]["run_id"]
    assert j["provenance"]["run_id"] == rid and j["provenance"]["seed"] == [3, 5] and "code_hash" in j["provenance"]
    assert j["verdict"]["gates"]["mean_ic"]["pass"] is True
    md = mp.read_text(encoding="utf-8")
    assert rid in md and "## Verdict" in md and "## By era" in md


def test_run_is_deterministic(planted):
    X, y = planted
    kw = dict(seeds=(3,), null_seeds=(), n_hidden=0)
    a = ht.run_heavy(X, y, {**CFG, "movement": False}, **kw)
    b = ht.run_heavy(X, y, {**CFG, "movement": False}, **kw)
    assert a["provenance"]["run_id"] == b["provenance"]["run_id"]
    assert a["overall"]["mean"] == b["overall"]["mean"]
    assert not a["verdict"]["gates"]["beats_noise_control"]["pass"]         # no null run supplied


def test_too_little_history_returns_not_ok():
    X, y = make()
    d = X.index.get_level_values(0)
    keep = d < d.unique()[60]
    s = ht.run_heavy(X[keep], y[keep], CFG, seeds=(3,))
    assert not s["ok"] and not s["verdict"]["pass"]


def test_ablation_across_eras_runs_only_where_history_exists(planted):
    X, y = planted
    ud = X.index.get_level_values(0).unique()
    eras = {"early": (str(ud[0].date()), str(ud[100].date())), "late": (str(ud[170].date()), str(ud[-1].date()))}
    cfg = {"n_controls": 1, "boot": 60, "min_test_dates": 20, "noise_mined": False,
           "miner": {"max_pairs": 60, "max_unless": 10, "null_reps": 1, "min_n": 100}}
    tbl, res = ht.ablation_across_eras(X, y, eras, cfg, min_train_dates=80)
    assert not tbl.loc["early", "ran"] and tbl.loc["late", "ran"]
    assert res["late"]["ok"] and tbl.loc["late", "gain"] == res["late"]["gain_vs_base"]["base+pattern"]


def test_stability_separates_a_steady_edge_from_a_lumpy_one():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2012-01-02", periods=300)
    steady = pd.DataFrame({"date": dates, "rank_ic": 0.04 + rng.normal(0, 0.03, 300), "top_net": 0.001})
    lumpy = steady.copy()
    lumpy["rank_ic"] = np.where(np.arange(300) < 150, 0.12, -0.04) + rng.normal(0, 0.03, 300)     # same-ish mean, two halves
    lumpy["top_net"] = np.where(np.arange(300) < 150, 0.004, -0.003)
    a, b = ht.stability(steady), ht.stability(lumpy)
    assert a["rolling_pos_share"] > 0.9 and b["rolling_pos_share"] < 0.7
    assert b["worst_window_ic"] < 0 < a["worst_window_ic"] and b["max_drawdown"] < a["max_drawdown"] - 0.1
    assert np.isnan(ht.stability(steady.iloc[:5])["icir"])


def test_pattern_lifetimes_distinguish_persistent_from_flickering_patterns():
    o = pd.to_datetime(["2015-01-05", "2015-07-06", "2016-01-04", "2016-07-04"])
    stable = pd.DataFrame({"name": "A", "origin": o, "seed": 1})
    flicker = pd.DataFrame({"name": ["B", "C", "D", "E"], "origin": o, "seed": 1})
    st, fl = ht.pattern_lifetimes(stable), ht.pattern_lifetimes(flicker)
    assert st["max_life"] == 4 and st["share_multi_origin"] == 1.0
    assert fl["max_life"] == 1 and fl["share_multi_origin"] == 0.0
    gap = pd.DataFrame({"name": "A", "origin": [o[0], o[1], o[3]], "seed": 1})
    assert ht.pattern_lifetimes(gap)["max_life"] == 2 or ht.pattern_lifetimes(pd.concat([gap, flicker]))["max_life"] >= 2
    assert ht.pattern_lifetimes(pd.DataFrame())["patterns"] == 0


def test_invented_ratio_and_era_intervals(planted, summary):
    assert ht.invented_ratio(pd.DataFrame({"n_live": [4, 6]}), 5.0) == 1.0
    assert np.isnan(ht.invented_ratio(pd.DataFrame({"n_live": []}), 3.0))
    ci = summary["era_ci"]
    assert (ci["lo5"] <= ci["rank_ic"]).all() and (ci["rank_ic"] <= ci["hi95"]).all()
    assert "invented_ratio" in summary["false_discovery"] and summary["false_discovery"]["invented_ratio"] < 0.5
    assert summary["stability"]["rolling_pos_share"] > 0.5 and summary["lifetimes"]["patterns"] > 0


def test_compare_reports_flags_regressions(summary):
    same = json.loads(json.dumps(ht._jsonable({k: v for k, v in summary.items() if k != "frame"})))
    assert ht.compare_reports(same, summary) == []
    worse = json.loads(json.dumps(same))
    worse["overall"]["mean"] = summary["overall"]["mean"] * 4
    worse["verdict"]["gates"]["mean_ic"]["pass"] = True
    worse["by_era"][0]["rank_ic"] = summary["by_era"]["rank_ic"].iloc[0] * 4
    cur = {**summary, "verdict": {"gates": {**summary["verdict"]["gates"], "mean_ic": {"pass": False, "value": 0, "need": ""}}}}
    flags = ht.compare_reports(worse, cur)
    assert any("mean_ic" in f for f in flags) and any("overall rank IC fell" in f for f in flags) and any("era" in f for f in flags)
