"""Bible Phases 6 and 34: pattern -> mover integration and the ablation framework, on synthetic panels with KNOWN
effects.  The planted movement pattern must survive every control; noise must never be deployed."""
import numpy as np
import pandas as pd
import pytest

from engine import pattern_movers as pm

WEEKS, STOCKS, NF = 190, 90, 8
HIDE = ["f4", "f5", "f6", "f7", "m_vix"]        # the mover model cannot see f0-f3; the miner can
CFG = {"n_controls": 2, "base_cols": HIDE, "boot": 150, "min_test_dates": 15,
       "miner": {"max_pairs": 120, "max_unless": 20, "null_reps": 1, "min_n": 120, "half_life_years": 50}}


def make(seed=0, move=1.6, direction=0.010, noise_only=False):
    """f0 top fifth -> much larger moves; f1 top fifth -> positive drift. Everything else is noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=WEEKS * 5)[::5]
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(STOCKS)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), NF)), index=idx, columns=[f"f{i}" for i in range(NF)])
    X["m_vix"] = pd.Series(rng.standard_normal(WEEKS), index=dates).reindex(idx.get_level_values(0)).values
    q = X[[f"f{i}" for i in range(NF)]].groupby(level=0).rank(pct=True)
    scale = 0.03 * (1 + (0 if noise_only else move) * (q["f0"] >= 0.8))
    y = pd.Series(rng.standard_normal(len(idx)) * scale.values, index=idx)
    if not noise_only:
        y += direction * (q["f1"] >= 0.8)
    return X, y - y.groupby(level=0).transform("mean")


@pytest.fixture(scope="module")
def planted():
    X, y = make()
    dates = X.index.get_level_values(0).unique()
    return X, y, dates[int(len(dates) * 0.62)]


@pytest.fixture(scope="module")
def planted_result(planted):
    X, y, split = planted
    return pm.run_ablation(X, y, split, CFG, seed=3)


def test_planted_movement_pattern_beats_every_control(planted_result):
    r = planted_result
    assert r["ok"] and r["deploy"], pm.format_ablation(r)
    g = r["gain_vs_base"]
    assert g["base+pattern"] > 0.002
    for names in r["families"].values():
        for n in names:
            assert g["base+pattern"] > g[n]
    assert r["arms"]["pattern_only"]["auc_daily"] > 0.55


def test_movement_bank_contains_planted_feature(planted_result):
    att = planted_result["attribution"]
    assert len(att) and att.iloc[0]["name"].startswith("f0 q4") or att["name"].str.contains("f0 q4").any()
    top = att[att["name"].str.contains("f0 q4")].iloc[0]
    assert top["mover_lift"] > 1.4 and top["loo_ic_drop"] >= 0


def test_pure_noise_is_not_deployed():
    X, y = make(seed=4, noise_only=True)
    d = X.index.get_level_values(0).unique()
    r = pm.run_ablation(X, y, d[int(len(d) * 0.62)], CFG, seed=5)
    assert r["ok"] and not r["deploy"]
    assert r["reasons"]


def test_no_incremental_value_when_mover_model_already_sees_the_signal(planted):
    """Real pattern, but the existing model has the same information: patterns add nothing and are not deployed."""
    X, y, split = planted
    r = pm.run_ablation(X, y, split, {**CFG, "base_cols": None}, seed=3)
    assert r["ok"] and not r["deploy"]
    assert r["gain_vs_base"]["base+pattern"] < 0.01


def test_controls_are_dead_on_planted_data(planted_result):
    g = planted_result["gain_vs_base"]
    for n in planted_result["families"]["shuffled"] + planted_result["families"]["scrambled"]:
        assert g[n] < 0.5 * g["base+pattern"]


def test_decide_refuses_when_a_control_matches_real():
    res = {"ok": True, "bank": {"direction": 3, "movement": 3},
           "gain_vs_base": {"base+pattern": 0.02, "base+random#0": 0.03, "base+shuffled#0": 0.0},
           "families": {"random": ["base+random#0"], "shuffled": ["base+shuffled#0"]},
           "real_vs_base": {"lo5": 0.01}, "arms": {"pattern_only": {"auc_daily": 0.6}}}
    d = pm.decide(res)
    assert not d["deploy"] and any("random control" in r for r in d["reasons"])
    res["gain_vs_base"]["base+random#0"] = 0.0
    assert pm.decide(res)["deploy"]


def test_empty_bank_and_short_history_degenerate_cases():
    b = pm.Bank(["f0"], [])
    X, y = make()
    s, n = pm.score_panel(b, X.iloc[:50])
    assert (s == 0).all() and (n == 0).all()
    assert len(pm.random_bank(b, np.random.default_rng(0))) == 0
    assert pm.attribute(b, X.iloc[:50], np.zeros(50, bool), np.zeros(50)).empty
    r = pm.run_ablation(X.iloc[:300], y.iloc[:300], X.index.get_level_values(0)[299], CFG)
    assert not r["ok"] and not r["deploy"]
    assert np.isnan(pm.auc([1, 2, 3], [True, True, True]))


def test_bank_scoring_matches_miner_score(planted):
    from engine.patterns import PatternMiner
    X, y, split = planted
    d = X.index.get_level_values(0)
    Xtr, ytr = X[d < split], y[d < split]
    M = PatternMiner({**CFG["miner"], "seed": 1}).fit(Xtr, pm.demeaned_abs(ytr), Xtr.index.get_level_values(0).max())
    bank = pm.bank_from_miner(M, pm.ctx_cols_of(Xtr))
    assert len(bank) > 0
    day = X[d == split]
    want = M.score(day)
    got, _ = pm.score_panel(bank, day)
    np.testing.assert_allclose(got.values, want.values, atol=1e-12)


def test_random_bank_keeps_shape_but_changes_conditions():
    keys = [("s", 0, 4), ("p", 1, 2, 3, 0), ("u", 0, 1, 2, 3, 4, 0)]
    b = pm.Bank([f"f{i}" for i in range(6)], [], keys, ["a", "b", "c"], np.array([.1, -.2, .3]), [None] * 3)
    r = pm.random_bank(b, np.random.default_rng(1))
    assert [k[0] for k in r.keys] == ["s", "p", "u"]
    assert r.keys != b.keys and sorted(r.effects) == sorted(b.effects)
    assert pm.random_bank(b, np.random.default_rng(1)).keys == r.keys      # seeded


def test_scramble_moves_every_date_and_shuffle_keeps_date_multiset():
    dc = np.repeat(np.arange(6), 10)
    v = np.arange(60, dtype=float)
    rng = np.random.default_rng(0)
    sc = pm.scramble_dates(v, dc, rng)
    for d in range(6):
        assert not np.array_equal(sc[dc == d], v[dc == d])
    sh = pm.shuffle_within_dates(v, dc, rng)
    for d in range(6):
        assert sorted(sh[dc == d]) == sorted(v[dc == d])


def test_calibrator_and_logit_recover_a_planted_probability():
    rng = np.random.default_rng(2)
    n = 6000
    pf = pd.DataFrame({"pat_mov": rng.standard_normal(n), "pat_dir": rng.standard_normal(n), "pat_fire": rng.integers(0, 5, n)})
    p = 1 / (1 + np.exp(-(-1.5 + 1.2 * pf["pat_mov"])))
    lab = rng.random(n) < p
    cal = pm.MovementCalibrator().fit(pf, lab)
    q = cal.predict(pf)
    assert np.corrcoef(q, p)[0, 1] > 0.97 and abs(q.mean() - lab.mean()) < 0.01


def test_interaction_table_flags_redundant_scores():
    rng = np.random.default_rng(3)
    dc = np.repeat(np.arange(40), 60)
    z = rng.standard_normal(len(dc))
    lab = rng.random(len(dc)) < 1 / (1 + np.exp(-(z - 1.5)))
    same = pm.interaction_table(z + 0.05 * rng.standard_normal(len(z)), z, lab, dc)
    indep = pm.interaction_table(z, rng.standard_normal(len(z)), lab, dc)
    assert same["rank_corr"] > 0.95 and abs(indep["rank_corr"]) < 0.1
    assert same["synergy_top"] < 1.0 <= indep["synergy_top"] * 1.6


def test_rolling_ablation_requires_majority_and_enough_origins(planted):
    X, y, _ = planted
    d = X.index.get_level_values(0).unique()
    few = pm.rolling_ablation(X, y, [d[120]], CFG, seed=1)
    assert not few["deploy"] and "usable origins" in few["reasons"][0]
    out = pm.rolling_ablation(X, y, [d[115], d[140], d[165]], CFG, seed=2, min_origins=3)
    assert out["n_ran"] == 3 and out["deploy"], out["reasons"]


def test_block_bootstrap_detects_positive_mean_and_null():
    rng = np.random.default_rng(0)
    pos = pm.block_bootstrap_mean(0.3 + rng.standard_normal(200), 5, 300, rng)
    nul = pm.block_bootstrap_mean(rng.standard_normal(200), 5, 300, rng)
    assert pos[1] > 0 and nul[1] < 0 < nul[2]


def test_calibration_table_flags_a_miscalibrated_probability():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 20000)
    good = rng.random(20000) < p
    bad = rng.random(20000) < np.clip(p * 0.4, 0, 1)                 # the model is far too confident
    _, g = pm.calibration_table(p, good)
    _, b = pm.calibration_table(p, bad)
    assert g["ece"] < 0.02 and b["ece"] > 0.1 and g["brier_skill"] > b["brier_skill"]
    assert np.isnan(pm.calibration_table(p[:10], good[:10])[1]["ece"])


def test_attribution_cuts_by_kind_and_feature():
    att = pd.DataFrame({"name": ["f0 q4", "f1 q0 & f2 q4", "f3 q4 & f4 q0 unless f0 q0"], "mass_share": [0.5, 0.3, 0.2],
                        "mover_lift": [1.8, 1.3, 1.1], "loo_ic_drop": [0.02, 0.005, -0.001]})
    k = pm.attribute_by_kind(att).set_index("kind")
    assert list(k.index) == ["exception", "pair", "single"] and abs(k["mass_share"].sum() - 1) < 1e-12
    f = pm.attribute_by_feature(att)
    assert f.iloc[0]["feature"] == "f0" and abs(f["mass"].sum() - 1) < 1e-12
    assert pm.attribute_by_kind(pd.DataFrame()).empty


def test_detectable_gain_shrinks_with_more_dates():
    rng = np.random.default_rng(0)
    small = pm.detectable_gain(rng.normal(0, 0.05, 40))
    big = pm.detectable_gain(rng.normal(0, 0.05, 400))
    assert big < small * 0.5 and np.isnan(pm.detectable_gain([0.1] * 5))


def test_ablation_sweep_reports_each_threshold(planted):
    X, y, split = planted
    t = pm.ablation_sweep(X, y, split, qs=(0.75, 0.9), cfg={**CFG, "n_controls": 1, "noise_mined": False, "boot": 50}, seed=3)
    assert list(t["mover_q"]) == [0.75, 0.9] and (t["gain"] > 0).all()


def test_model_deploys_planted_and_refuses_noise():
    X, y = make()
    d = X.index.get_level_values(0)
    ud = d.unique()
    tr = d < ud[150]
    cfg = {**CFG, "n_controls": 1, "noise_mined": False, "boot": 80}
    m = pm.PatternMoverModel(cfg, seed=3).fit(X[tr], y[tr], now=ud[150])
    assert m.decision["deploy"], m.decision
    day = X[d == ud[152]]
    f = m.features(day, as_of=ud[152])
    assert f["deployed"].all() and f["p_move"].between(0, 1).all()
    top = f["p_move"] >= f["p_move"].quantile(0.9)
    assert (day["f0"].groupby(level=0).rank(pct=True)[top] >= 0.8).mean() > 0.6     # p_move points at the planted stocks
    Xn, yn = make(seed=9, noise_only=True)
    mn = pm.PatternMoverModel(cfg, seed=3).fit(Xn[tr], yn[tr], now=ud[150])
    fn = mn.features(Xn[d == ud[152]], as_of=ud[152])
    assert not mn.decision["deploy"] and (fn["pat_dir"] == 0).all() and fn["p_move"].isna().all() and not fn["deployed"].any()


def test_model_refuses_lookahead_inputs():
    X, y = make()
    d = X.index.get_level_values(0); ud = d.unique()
    with pytest.raises(ValueError, match="strictly before"):
        pm.PatternMoverModel(CFG).fit(X, y, now=ud[100])
    m = pm.PatternMoverModel(CFG).fit(X[d < ud[40]], y[d < ud[40]], now=ud[40])       # too short: fitted but refuses
    assert not m.decision["deploy"] and "60" in m.decision["reasons"][0]
    with pytest.raises(ValueError, match="after as_of"):
        m.features(X[d == ud[50]], as_of=ud[45])
    with pytest.raises(ValueError, match="precedes"):
        m.features(X[d == ud[41]], as_of=ud[30])
    with pytest.raises(RuntimeError):
        pm.PatternMoverModel(CFG).features(X[d == ud[41]], as_of=ud[41])


def test_bank_overlap():
    a = pm.Bank(["f0"], [], [], ["x", "y"], np.zeros(2), [None] * 2)
    b = pm.Bank(["f0"], [], [], ["y", "z"], np.zeros(2), [None] * 2)
    assert abs(pm.bank_overlap(a, b) - 1 / 3) < 1e-12 and np.isnan(pm.bank_overlap(pm.Bank([], []), pm.Bank([], [])))
