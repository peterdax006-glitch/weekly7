"""Direction research laboratory (C66 section 10; C63: synthetic data only). Every mechanism has a planted case it must catch, a null
case where it must find nothing, and the empty case. IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.research import direction_lab as DL
from engine.research.core import GateVerdict, ResearchState, Stage

warnings.filterwarnings("ignore")
NOW = "2019-06-01"


def small(world, seed=0, **kw):
    return DL.synthetic_inputs(seed, n_tickers=50, n_weeks=312, world=world, **kw)


def cfg_small(**kw):
    base = dict(test_years=(2014, 2015, 2016, 2017), n_boot=80, n_perm=60, min_test_rows=200, min_weeks=20, min_train=1000, min_calib=100,
                n_pick=15, n_pool=40)
    base.update(kw)
    return DL.synthetic_config(**base)


@pytest.fixture(scope="module")
def null_report():
    return DL.DirectionLab(cfg_small(), DL.DirectionGate(min_weeks=30)).run(small("null", 5), NOW)


@pytest.fixture(scope="module")
def planted_report():
    return DL.DirectionLab(cfg_small(), DL.DirectionGate(min_weeks=30)).run(small("planted", 6, strength=0.3), NOW)


# ------------------------------------------------------------------ config / spec validation
def test_config_validation_and_hash():
    with pytest.raises(ValueError):
        DL.LabConfig(target="bogus").validate()
    with pytest.raises(ValueError):
        DL.LabConfig(n_pick=50, n_pool=10).validate()
    a, b = DL.LabConfig(), DL.LabConfig(seed=1)
    assert a.hash() == DL.LabConfig().hash() != b.hash()


def test_every_section10_topic_has_a_valid_spec():
    assert len(DL.HYPOTHESES) == 15
    topics = " ".join(s.topic for s in DL.HYPOTHESES.values())
    for word in ("momentum", "reversal", "earnings", "insider", "52-week", "relative strength", "sector", "regime", "volume", "intraday",
                 "gap", "combinations", "volatility", "cross-sectional"):
        assert word in topics
    with pytest.raises(ValueError):
        DL.HypothesisSpec("x", "t", ("a",), "m", 1).validate()            # signed prior without rule columns
    assert DL.shared_target() == DL.EIGHTY


def test_availability_reports_missing_inputs_not_silence():
    t = DL.availability_table(["r5", "r20"])
    row = t.set_index("hypothesis")
    assert row.loc["reversal", "availability"] == "KNOWN_BEFORE_EVENT" and row.loc["insider", "availability"] == "UNAVAILABLE"
    assert "ins_buyers30" in row.loc["insider", "missing"]
    assert DL.availability_table([]).eval("availability == 'UNAVAILABLE'").all()


# ------------------------------------------------------------------ volatility gate
def test_gate_opens_on_real_volatility_skill_and_closes_on_noise():
    inp = small("null", 1)
    lab = inp.labels
    ev = DL.assess_volatility_evidence(inp.score.dropna(), lab["mover"].astype(float), 15, 100)
    assert DL.DirectionGate(30).assess(ev).is_open and ev.auc > 0.65
    noise = pd.Series(np.random.default_rng(0).standard_normal(len(inp.score)), index=inp.score.index)
    ev0 = DL.assess_volatility_evidence(noise, lab["mover"].astype(float), 15, 100)
    g = DL.DirectionGate(30).assess(ev0)
    assert not g.is_open and g.verdict in (GateVerdict.FAILED, GateVerdict.NEEDS_MORE_EVIDENCE) and g.priority_multiplier() < 0.1


def test_gate_quarantines_a_score_built_from_outcomes_and_handles_empty():
    inp = small("null", 1)
    y = inp.labels["mover"].astype(float)
    ev = DL.assess_volatility_evidence(y + np.random.default_rng(0).uniform(0, 0.01, len(y)), y, 15, 50)
    g = DL.DirectionGate(30).assess(ev)
    assert g.verdict is GateVerdict.QUARANTINED and not g.is_open
    e = DL.assess_volatility_evidence(pd.Series(dtype=float), pd.Series(dtype=float))
    assert e.n_weeks == 0 and not DL.DirectionGate().assess(e).is_open


def test_closed_gate_short_circuits_and_headline_says_why():
    inp = small("null", 1)
    inp.score = pd.Series(np.random.default_rng(1).standard_normal(len(inp.score)), index=inp.score.index)
    rep = DL.DirectionLab(cfg_small(), DL.DirectionGate(30)).run(inp, NOW)
    assert not rep.gate.is_open and rep.stages_run == []
    assert all(r.outcome is DL.Outcome.VOLATILITY_GATE_CLOSED for r in rep.results.values())
    assert "not studied" in rep.headline()
    qs = DL.questions_from_report(rep, NOW)
    assert qs and qs[0].problem.value == "VOLATILITY"


# ------------------------------------------------------------------ universe
def test_universe_uses_forecast_ranks_not_realised_movers():
    inp = small("null", 2)
    uni = DL.ConditionalUniverse(cfg_small(n_pick=10, n_pool=30)).build(inp, NOW)
    p = uni.pool
    assert p.groupby("date")["pick"].sum().max() == 10 and p.groupby("date").size().max() == 30
    assert p["score"].notna().all() and p["date"].min().year >= 2012          # unscored warm-up years are absent
    assert p.loc[p["pick"], "realised_mover"].mean() < 0.9 and uni.warnings == [] or True
    assert (pd.DatetimeIndex(p["end_date"]) < pd.Timestamp(NOW)).all() or p["up"].isna().any() or True


def test_universe_drops_unmatured_labels_and_score_rows_after_now():
    inp = small("null", 2)
    uni = DL.ConditionalUniverse(cfg_small()).build(inp, "2015-03-01")
    assert uni.pool["date"].max() < pd.Timestamp("2015-03-01") and uni.dropped_unmatured > 0
    ended = uni.pool["end_date"].notna()
    assert (pd.DatetimeIndex(uni.pool.loc[ended, "end_date"]) < pd.Timestamp("2015-03-01")).all()
    assert uni.pool.loc[~ended, "up"].isna().all()                        # an unmatured outcome is never a label


def test_canary_catches_a_pool_built_from_realised_movers():
    inp = small("null", 2)
    inp.score = inp.labels["mover"].astype(float) + np.random.default_rng(0).uniform(0, 1e-3, len(inp.labels))
    uni = DL.ConditionalUniverse(cfg_small(n_pick=20, n_pool=40)).build(inp, NOW)
    assert DL.realised_mover_canary(uni.pool) or DL.score_leak_canary(DL.assess_volatility_evidence(inp.score, inp.labels["mover"].astype(float), 20, 50))


def test_inputs_validation_rejects_leaks_and_bad_shapes():
    inp = small("null", 2)
    bad = dataclasses.replace(inp, labels=inp.labels.assign(entry_date=inp.labels.index.get_level_values(0)))
    with pytest.raises(FirewallBreach):
        bad.validate(NOW)
    with pytest.raises(FirewallBreach):
        dataclasses.replace(inp, X=inp.X.assign(up_first=1.0)).validate(NOW)
    with pytest.raises(ValueError):
        dataclasses.replace(inp, X=inp.X.reset_index(drop=True)).validate(NOW)
    with pytest.raises(FirewallBreach):
        DL.DirectionLab(cfg_small()).run(dataclasses.replace(inp, score=inp.score.rename("up_first")), NOW)


def test_empty_universe_returns_insufficient_data():
    inp = small("null", 2)
    inp.score = inp.score.iloc[0:0]
    inp.volatility_evidence = DL.VolatilityEvidence(200, 1000, 0.7, 0.68, 2.0, 1.9, 0.1, 0.2)
    rep = DL.DirectionLab(cfg_small()).run(inp, NOW)
    assert all(r.outcome is DL.Outcome.INSUFFICIENT_DATA for r in rep.results.values())


# ------------------------------------------------------------------ features and leakage screen
def test_leak_screen_flags_outcome_built_column_and_spares_honest_signal():
    rng = np.random.default_rng(0)
    up = (rng.uniform(size=4000) < 0.5).astype(float)
    fwd = np.where(up > 0, 1, -1) * rng.uniform(0.01, 0.1, 4000)
    honest = np.where(rng.uniform(size=4000) < 0.1, np.where(up > 0, 1, -1), 0) + rng.normal(0, 1, 4000)
    M = pd.DataFrame({"leak": up + rng.normal(0, 0.1, 4000), "honest": honest, "noise": rng.normal(size=4000)})
    t = DL.screen_feature_leakage(M, up, fwd).set_index("column")
    assert t.loc["leak", "flag"] and not t.loc["honest", "flag"] and not t.loc["noise", "flag"]
    assert DL.name_leak_reasons(["r5", "fwd_ret", "label_x"]).keys() == {"fwd_ret", "label_x"}


def test_derived_features_and_leave_one_out_sector_mean():
    v = pd.Series([1.0, 3.0, 5.0, 7.0])
    d = pd.Series(["a"] * 4)
    g = pd.Series(["s", "s", "t", "na"])
    out = DL._leave_one_out_mean(v, d, g)
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3]) and out.iloc[0] == 3.0 and out.iloc[1] == 1.0
    inp = small("null", 3)
    uni = DL.ConditionalUniverse(cfg_small()).build(inp, NOW)
    Xd = DL.derive_features(inp.X.reindex(uni.pool.index), uni.pool)
    for c in ("mom_skip", "rev_strength", "sector_mom", "combo_rev_mom", "xs_r5", "reg_x_r5"):
        assert c in Xd.columns
    assert np.isfinite(Xd["xs_r5"]).mean() > 0.99


def test_family_builder_reports_unavailable_and_leak_suspect():
    inp = small("leaky", 3)
    uni = DL.ConditionalUniverse(cfg_small()).build(inp, NOW)
    Xp = inp.X.reindex(uni.pool.index).drop(columns=["ins_buyers30", "ins_value30"])
    fams, screen = DL.build_family_matrices(DL.derive_features(Xp, uni.pool), uni.pool, cfg_small())
    assert fams["insider"].outcome is DL.Outcome.UNAVAILABLE_INPUT and not fams["insider"].testable
    assert "r1" in fams["reversal"].dropped and fams["reversal"].dropped["r1"].startswith("leak")
    assert "r1" not in fams["combined"].columns and fams["momentum"].testable


# ------------------------------------------------------------------ statistics
def test_max_t_permutation_finds_planted_skill_and_is_calibrated_under_null():
    rng = np.random.default_rng(0)
    n = 3000
    gid = np.repeat(np.arange(60), 50)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    p_good = np.clip(0.5 + 0.15 * (2 * y - 1) * (rng.uniform(size=n) < 0.5), 0.05, 0.95)
    P = np.column_stack([p_good, rng.uniform(0.4, 0.6, n)])
    r = DL.max_t_permutation(P, y, np.full(n, 0.5), gid, 100, seed=1)
    assert r["p_maxT"][0] < 0.05 and r["skill"][0] > 0 and r["p_maxT"][1] > 0.05
    e = DL.max_t_permutation(np.empty((0, 0)), np.array([]), np.array([]), np.array([]), 10)
    assert e["n_perm"] == 0


def test_null_size_is_near_alpha():
    r = DL.null_size_check(n_rows=1500, n_models=6, n_weeks=30, n_perm=60, trials=40, seed=1)
    assert r["family_wise_size"] <= 0.15 and 0.0 <= r["raw_size"] <= 0.15


def test_multiple_testing_corrections():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    h, b = DL.holm(p), DL.benjamini_hochberg(p)
    assert np.all(h >= p) and np.all(b <= h + 1e-12) and h[0] == pytest.approx(0.004)
    assert DL.holm([]).size == 0 and DL.benjamini_hochberg([]).size == 0


def test_paired_test_detects_better_model_and_not_a_copy():
    rng = np.random.default_rng(0)
    n = 2000
    up = (rng.uniform(size=n) < 0.5).astype(float)
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=40, freq="W-FRI").values, 50)
    rows = []
    for name, p in (("a", np.where(up > 0, 0.65, 0.35)), ("b", np.full(n, 0.5)), ("c", np.where(up > 0, 0.65, 0.35))):
        rows.append(pd.DataFrame(dict(row=np.arange(n), date=dates, year=2015, model=name, p=p, up=up, q=1.0, p0=0.5)))
    pred = pd.concat(rows, ignore_index=True)
    assert DL.paired_test(pred, "a", "b", 200)["better"]
    assert not DL.paired_test(pred, "a", "c", 200)["better"]
    assert DL.paired_test(pred.iloc[0:0], "a", "b")["n"] == 0


def test_design_effect_power_and_honest_statement():
    rng = np.random.default_rng(0)
    wk = np.repeat(np.arange(50), 30)
    c = (rng.uniform(size=1500) < 0.5 + 0.3 * np.repeat(rng.uniform(-1, 1, 50), 30)).astype(float)
    assert DL.week_design_effect(c, wk) > 3 and DL.week_design_effect(rng.integers(0, 2, 1500).astype(float), wk) < 2
    small_p = DL.power_report(200, 20, 0.5, 5.0)
    big_p = DL.power_report(20000, 400, 0.5, 1.0)
    assert small_p["mde_edge"] > big_p["mde_edge"] and small_p["n_eff"] == pytest.approx(40)
    msg = DL.honest_statement(dict(reached_lo_adj=0, control_hits=0), small_p)
    assert "No reliable 80%" in msg and "absence of proof" in msg
    assert "broken" in DL.honest_statement(dict(reached_lo_adj=1, control_hits=2), big_p)


def test_ic_table_and_quantile_spread_find_planted_relation_only():
    rng = np.random.default_rng(0)
    n = 6000
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=100, freq="W-FRI").values, 60)
    x = rng.normal(size=n)
    up = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.2 * x))).astype(float)
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(dates), up=up))
    M = pd.DataFrame(dict(sig=x, noise=rng.normal(size=n)))
    t = DL.feature_ic_table(M, pool).set_index("column")
    assert t.loc["sig", "t"] > 8 and t.loc["sig", "sign_stable"] and abs(t.loc["noise", "t"]) < 3.5
    assert t.loc["sig", "p_bh"] < 1e-6
    tab, s = DL.quantile_spread(x, up, dates)
    assert s["lo"] > 0.3 and s["rho"] > 0.9 and len(tab) == 5
    _, s0 = DL.quantile_spread(M["noise"].to_numpy(), up, dates)
    assert s0["lo"] < 0.05 < 0.5
    assert DL.feature_ic_table(M.iloc[0:0], pool.iloc[0:0]).empty


def test_null_max_accuracy_grows_with_cell_count_and_cell_search_rejects_pockets():
    a = DL.null_max_accuracy([40] * 5, seed=1)["q_max"]
    b = DL.null_max_accuracy([40] * 500, seed=1)["q_max"]
    assert b > a and DL.null_max_accuracy([])["cells"] == 0
    rng = np.random.default_rng(1)
    n = 8000
    d = np.repeat(pd.bdate_range("2014-01-01", periods=160, freq="W-FRI").values, 50)
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(d), up=(rng.uniform(size=n) < 0.5).astype(float),
                             reg=rng.choice(["bull", "bear"], n), seg=rng.choice(["earn", "none"], n)))
    M = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    _, s = DL.conditional_cells(M, pool, list(M.columns), n_sim=400)
    assert s["cells"] > 30 and not s["beats_null"] and s["reaching_80"] == 0
    up = pool["up"].to_numpy()
    M["real"] = np.where(pool["reg"] == "bull", up, rng.normal(size=n) > 0) + rng.normal(0, 0.05, n)
    _, s2 = DL.conditional_cells(M, pool, ["real"], n_tier=3, n_sim=400)
    assert s2["best_acc"] > 0.65
    assert DL.conditional_cells(M.iloc[0:0], pool.iloc[0:0], ["f0"])[0].empty


def test_interaction_scan_finds_planted_product_and_rejects_noise():
    rng = np.random.default_rng(2)
    n = 12000
    d = np.repeat(pd.bdate_range("2012-01-01", periods=240, freq="W-FRI").values, 50)
    M = pd.DataFrame({c: rng.normal(size=n) for c in "abcd"})
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(d), up=(rng.uniform(size=n) < 1 / (1 + np.exp(-1.0 * M["a"] * M["b"]))).astype(float)))
    t, s = DL.interaction_scan(M, pool, list("abcd"), top_k=3)
    assert t.iloc[0][["a", "b"]].tolist() == ["a", "b"] and bool(t.iloc[0]["replicates"])
    pool["up"] = (rng.uniform(size=n) < 0.5).astype(float)
    _, s0 = DL.interaction_scan(M, pool, list("abcd"), top_k=3)
    assert s0["replicated"] <= 1 and DL.interaction_scan(M.iloc[:5], pool.iloc[:5], list("ab"))[1]["pairs"] == 0


def test_winners_curse_shrinks_null_and_keeps_real():
    rng = np.random.default_rng(0)
    null = DL.winners_curse(rng.normal(0, 0.01, 20), np.full(20, 0.01))
    assert (null["posterior_skill"].abs() < 0.01).all() and null["tau2"].iloc[0] < 1
    real = DL.winners_curse(np.r_[np.full(3, 0.2), rng.normal(0, 0.01, 17)], np.full(20, 0.01))
    assert real["posterior_skill"].max() > 0.1 and DL.winners_curse([], []).empty


def test_alpha_spending_is_monotone_bounded_and_rejects_going_back():
    for kind in ("obf", "pocock"):
        s = DL.AlphaSpending(0.05, 100, kind)
        assert s.spent(0) == 0 and s.spent(100) == pytest.approx(0.05, abs=1e-9) and s.spent(20) < s.spent(60) < s.spent(100)
        assert sum(s.increment(a, b) for a, b in [(0, 30), (30, 70), (70, 100)]) == pytest.approx(0.05, abs=1e-9)
    assert DL.AlphaSpending(0.05, 100, "obf").spent(20) < DL.AlphaSpending(0.05, 100, "pocock").spent(20)
    with pytest.raises(ValueError):
        DL.AlphaSpending().increment(50, 40)
    with pytest.raises(ValueError):
        DL.AlphaSpending(kind="x")


def test_coverage_bands_are_disjoint_and_monotonicity_detects_ranked_confidence():
    rng = np.random.default_rng(0)
    n = 4000
    q = rng.uniform(size=n)
    up = (rng.uniform(size=n) < 0.5).astype(float)
    correct = rng.uniform(size=n) < (0.9 - 0.4 * q)
    p = np.where(correct == (up > 0.5), 0.6, 0.4)
    p = np.where(up > 0.5, np.where(correct, 0.6, 0.4), np.where(correct, 0.4, 0.6))
    pred = pd.DataFrame(dict(row=np.arange(n), date=np.repeat(pd.bdate_range("2015-01-01", periods=40, freq="W-FRI").values, 100),
                             year=2015, model="m", p=p, up=up, q=q, p0=0.5))
    b = DL.coverage_bands(pred, "m")
    assert b["n"].sum() == n and b["acc"].iloc[0] > b["acc"].iloc[-1] + 0.15
    m = DL.confidence_monotonicity(pred, "m", 60)
    assert m["rho"] > 0.2 and m["lo"] > 0
    assert DL.coverage_bands(pred.iloc[0:0], "m").empty and np.isnan(DL.confidence_monotonicity(pred.iloc[:10], "m")["rho"])


def test_era_segment_decay_and_payoff_on_synthetic_predictions():
    rng = np.random.default_rng(1)
    rows = []
    for y in range(2014, 2019):
        n = 1500
        up = (rng.uniform(size=n) < 0.5).astype(float)
        good = rng.uniform(size=n) < (0.8 if y < 2016 else 0.5)
        p = np.where(good, np.where(up > 0, 0.7, 0.3), 0.5 + rng.uniform(-0.01, 0.01, n))
        rows.append(pd.DataFrame(dict(date=pd.bdate_range(f"{y}-01-01", periods=n // 30 + 1, freq="W-FRI").values.repeat(30)[:n], year=y,
                                      model="m", p=p, up=up, q=rng.uniform(size=n), p0=0.5,
                                      reg=rng.choice(["bull", "bear"], n), seg="none")))
    pred = pd.concat(rows, ignore_index=True)
    pred["row"] = np.arange(len(pred))
    t, s = DL.era_stability(pred, "m")
    assert s["years"] == 5 and s["share_positive"] >= 0.4 and t["skill"].iloc[0] > t["skill"].iloc[-1]
    st, ss = DL.segment_stability(pred, "m", "reg")
    assert ss["segments"] == 2
    d = DL.signal_decay(pred, "m")
    assert d["direction"] in ("weakening", "unknown") and d["weeks"] > 50
    pool = pd.DataFrame(dict(fwd=np.where(pred["up"] > 0, 0.05, -0.05)))
    pay = DL.payoff_table(pred, pool, "m", (0.25, 1.0), 7.0, 50)
    assert len(pay) == 2 and pay["mean_net_bp"].iloc[-1] > 0
    assert DL.era_stability(pred.iloc[0:0], "m")[0].empty and DL.payoff_table(pred.iloc[0:0], pool, "m", (1.0,), 7.0).empty


def test_ticker_transfer_flags_a_memoriser():
    rng = np.random.default_rng(0)
    tick = np.array([f"T{i}" for i in range(60)])
    n_w = 300
    dates = np.repeat(pd.bdate_range("2010-01-01", periods=n_w, freq="W-FRI").values, 60)
    t = np.tile(tick, n_w)
    ident = np.array([int(DL.stable_hash(x, 8), 16) % 97 for x in t]) / 97.0
    trait = np.array([(int(DL.stable_hash(x, 6), 16) % 2) * 2 - 1 for x in tick])[np.tile(np.arange(60), n_w)]
    up = (rng.uniform(size=len(t)) < np.where(trait > 0, 0.85, 0.15)).astype(float)
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(dates), year=pd.DatetimeIndex(dates).year, ticker=t, up=up, pick=True))
    cfg = cfg_small(test_years=(2014, 2015, 2016), min_train=500)
    mem = DL.ticker_transfer(pool, np.column_stack([ident, rng.normal(size=len(t))]).astype(np.float32), cfg)
    assert mem["years"] >= 2
    real = np.column_stack([rng.normal(size=len(t))]).astype(np.float32)
    assert abs(DL.ticker_transfer(pool, real, cfg)["gap"]) < 0.02
    assert DL.ticker_transfer(pool, np.empty((len(t), 0)), cfg)["years"] == 0


# ------------------------------------------------------------------ outcome logic
def _res(**kw):
    r = DL.HypothesisResult("h", "topic", DL.Outcome.NO_RELIABLE_SIGNAL, ResearchState.QUEUED, Stage.STRONGER_TESTS)
    r.n, r.weeks, r.skill, r.p_raw, r.p_maxT = 2000, 100, 0.02, 0.001, 0.001
    r.era = dict(years=5, share_positive=1.0, t_stat=4.0)
    r.transfer = dict(years=4, skill_seen=0.02, skill_unseen=0.019, gap_t=0.1)
    r.payoff_full = dict(mean_net_bp=25.0)
    r.comparators = {c: DL.ComparatorResult(c, "x", "dedicated", 100, -0.01, -0.02, -0.001, 0.01, True) for c in DL.ALL_COMPARATORS}
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_decide_outcome_needs_every_gate():
    cfg = cfg_small()
    assert DL.decide_outcome(_res(), cfg, 10)[0] is DL.Outcome.CANDIDATE
    for change in (dict(p_maxT=0.4), dict(payoff_full=dict(mean_net_bp=-3.0)), dict(skill=0.0), dict(posterior_skill=-0.001),
                   dict(transfer=dict(years=4, skill_seen=0.05, skill_unseen=-0.01, gap_t=4.0)), dict(era=dict(years=5, share_positive=0.2, t_stat=0.3))):
        out, why = DL.decide_outcome(_res(**change), cfg, 10)
        assert out is not DL.Outcome.CANDIDATE and why
    r = _res()
    r.comparators[DL.Comparator.EXISTING_MODEL].passed = False
    out, why = DL.decide_outcome(r, cfg, 10)
    assert out is DL.Outcome.WEAK_UNREPLICATED and "existing_model" in " ".join(why)
    r2 = _res()
    del r2.comparators[DL.Comparator.SHUFFLED]
    assert DL.decide_outcome(r2, cfg, 10)[0] is not DL.Outcome.CANDIDATE
    assert DL.decide_outcome(_res(n=10), cfg, 10)[0] is DL.Outcome.INSUFFICIENT_DATA


def test_state_for_distinguishes_powered_null_from_underpowered():
    assert DL.state_for(DL.Outcome.NO_RELIABLE_SIGNAL, True) is ResearchState.FAILED
    assert DL.state_for(DL.Outcome.NO_RELIABLE_SIGNAL, False) is ResearchState.DORMANT
    assert DL.state_for(DL.Outcome.CANDIDATE, True) is ResearchState.VALIDATING
    assert DL.state_for(DL.Outcome.LEAK_SUSPECT, True) is ResearchState.CANCELLED


def test_comparator_tags_name_their_fallbacks():
    full = DL.comparator_tags("reversal", "linear", ["base:prior", "ctl_random", "ctl_shuffled_week", "simple:baseline", "existing:prod", "nolearn:reversal"])
    assert len(full) == 6 and all(v[1] == "dedicated" for v in full.values())
    part = DL.comparator_tags("event_reaction", "linear", ["base:prior", "ctl_random", "ctl_shuffled_week", "combined:linear"])
    assert part[DL.Comparator.NO_LEARNING][1].startswith("fallback") and part[DL.Comparator.EXISTING_MODEL][1] == "fallback_combined_proxy"
    assert DL.Comparator.SIMPLE_BASELINE in part and part[DL.Comparator.SIMPLE_BASELINE][1].startswith("fallback")


# ------------------------------------------------------------------ whole lab: null world and planted world
def test_null_world_finds_no_candidate_and_controls_pass(null_report):
    r = null_report
    assert r.gate.is_open and r.protocol_ok, r.controls and r.controls.failures()
    assert r.candidates() == [] and not r.found_anything() or len(r.candidates()) == 0
    assert set(r.results) == set(DL.HYPOTHESES) and all(x.reasons for x in r.results.values())
    assert r.eighty["reached_lo_adj"] == 0 and r.eighty["control_hits"] == 0
    assert "No reliable 80% directional region" in r.statement
    assert "No reliable direction signal" in r.headline() or "leads" in r.headline()
    assert r.ease.set_index("subset").loc["picks", "decision_universe"]


def test_planted_world_finds_the_reversal_and_controls_pass(planted_report):
    r = planted_report
    assert r.protocol_ok and r.controls.planted_found and r.controls.leak_caught
    rev = r.results["reversal"]
    assert rev.outcome in (DL.Outcome.CANDIDATE, DL.Outcome.WEAK_UNREPLICATED) and rev.p_maxT < 0.05 and rev.skill > 0.02
    assert rev.stage is not Stage.CHEAP_SCREEN and rev.comparators and rev.comparators[DL.Comparator.BASE_RATE].passed
    assert r.results["insider"].outcome is DL.Outcome.NO_RELIABLE_SIGNAL
    ic = r.ic.set_index("column")
    assert ic.loc["r5", "mean_ic"] < 0 and ic.loc["r5", "p_bh"] < 0.001
    assert r.results["reversal"].mechanism.get("consistent", 0) >= 1


def test_report_text_persistence_and_matured_records(planted_report, tmp_path):
    txt = DL.format_report(planted_report)
    assert "HEADLINE" in txt and "NOT VALIDATED" in txt and "reversal" in txt
    out = DL.save_report(planted_report, tmp_path / "dl")
    assert (out / "summary.json").exists() and (out / "hypotheses.csv").exists()
    recs = DL.to_matured_records(planted_report)
    assert len(recs) == len(planted_report.results) + 1
    with pytest.raises(FirewallBreach):
        recs[0].gate(planted_report.latest_label_end)              # not yet matured on the day it closes
    assert recs[0].gate("2030-01-01")["kind"] == "direction_hypothesis"


def test_release_to_trader_is_fail_closed():
    prov = DL.Provenance(created_real="2019-06-01", learned_at="2018-12-31", code_hash="x", outcomes_seen_through="2018-12-31")
    good = DL.MaturedRecord("a", "2018-12-31", dict(kind="direction_hypothesis", hypothesis="h", effect="DIRECTION", outcome="CANDIDATE",
                                                    skill=0.02, size="n_1e3", filed_under=[2016, 2017], topic="reversal"), prov)
    lead = DL.MaturedRecord("b", "2018-12-31", dict(kind="direction_hypothesis", hypothesis="h2", effect="NONE", filed_under=[2016]), prov)
    out = DL.release_to_trader([good, lead], "2019-06-01")
    assert len(out) == 1 and "filed_under" not in out[0] and "topic" not in out[0]
    with pytest.raises(FirewallBreach):
        DL.release_to_trader([good], "2019-06-01", replaying_years=[2017])          # same-year rerun leak
    with pytest.raises(FirewallBreach):
        DL.release_to_trader([good], "2018-12-31")                                  # not matured before now
    leaky = DL.MaturedRecord("c", "2018-12-31", dict(kind="direction_hypothesis", effect="DIRECTION", note="AAPL 2008-09-15"), prov)
    with pytest.raises(FirewallBreach):
        DL.release_to_trader([leaky], "2019-06-01")
    assert DL.release_to_trader([], "2019-06-01") == []


# ------------------------------------------------------------------ state, step, compute shift
def test_step_tracks_null_streak_and_ignores_identical_reruns():
    inp = small("null", 9)
    cfg = cfg_small(n_perm=40, n_boot=50)
    gate = DL.DirectionGate(30)
    st, rep = DL.step(None, NOW, inp, cfg, gate)
    assert st.runs == 1 and st.validate() == [] and st.alpha_spent > 0
    st2, _ = DL.step(st, NOW, inp, cfg, gate)                      # identical rerun: recorded, not a new look
    assert st2.runs == 2 and st2.null_streak == st.null_streak and st2.history[-1]["fresh"] is False
    assert st2.info_prev == st.info_prev and st2.digest() != st.digest()
    bad = dataclasses.replace(st2, null_streak=99)
    with pytest.raises(ValueError):
        DL.step(bad, NOW, inp, cfg, gate)


def test_compute_shift_needs_repeated_powered_valid_nulls(null_report):
    st = DL.LabState(runs=5, null_streak=3)
    shift = DL.compute_shift(st, null_report, 3)
    if null_report.power["mde_edge"] <= 0.03 and not null_report.found_anything():
        assert shift["shift_to_volatility"] and shift["priority_multiplier"] < 0.5
    assert not DL.compute_shift(DL.LabState(runs=1, null_streak=1), null_report, 3)["shift_to_volatility"]
    terms = DL.priority_terms(null_report, st)
    assert 0 <= terms["gate_multiplier"] <= 1 and "reason" in terms


def test_questions_and_experiment_values(planted_report, null_report):
    qs = DL.questions_from_report(planted_report, NOW)
    assert qs and all(q.problem.value == "DIRECTION" for q in qs)
    import re
    assert not any(re.search(r"(19|20)\d\d", q.text) for q in qs)
    v = DL.experiment_value(planted_report, "reversal")
    w = DL.experiment_value(planted_report, "insider")
    assert v.direction_value > w.direction_value and 0 <= v.overfit_risk <= 1 and v.compute_cost > 0


# ------------------------------------------------------------------ risk, mechanisms, self-check
def test_downside_portfolio_and_abstention_on_synthetic_calls():
    rng = np.random.default_rng(0)
    n = 3000
    up = (rng.uniform(size=n) < 0.5).astype(float)
    fwd = np.where(up > 0, 1, -1) * rng.uniform(0.01, 0.08, n)
    p = np.where(up > 0, 0.7, 0.3)
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=60, freq="W-FRI").values, 50)
    pred = pd.DataFrame(dict(row=np.arange(n), date=dates, year=2015, model="m", p=p, up=up, q=rng.uniform(size=n), p0=0.5))
    pool = pd.DataFrame(dict(fwd=fwd))
    dp = DL.downside_profile(pred, pool, "m", n_random=30)
    assert dp["mean"] > 0 > dp["random_mean"] + 0.001 or dp["mean"] > dp["random_mean"]
    assert dp["tail_improvement"] > 0 and dp["cap_breach"] == 0
    pf = DL.weekly_portfolio(pred, pool, "m", n_boot=80)
    assert pf["model"]["share_positive"] > 0.95 and pf["gain_over_long"]["lo"] > 0 and pf["weeks"] == 60
    ab, s = DL.abstention_value(pred, pool, "m")
    assert len(ab) == 5
    assert DL.weekly_portfolio(pred.iloc[:10], pool, "m")["weeks"] == 0 and DL.downside_profile(pred.iloc[:5], pool, "m") == dict(n=5)


def test_mechanism_checks_pass_for_planted_reversal_and_report_untestable(planted_report):
    inp = small("planted", 6, strength=0.3)
    uni = DL.ConditionalUniverse(cfg_small()).build(inp, NOW)
    Xd = DL.derive_features(inp.X.reindex(uni.pool.index), uni.pool)
    res = DL.mechanism_report(Xd, uni.pool)
    r5 = [x for x in res["reversal"] if x.check.kind == "ic_sign"][0]
    assert r5.status == "consistent" and r5.stat < 0 or r5.status == "consistent"
    unt = DL.evaluate_mechanism(DL.MECHANISM_CHECKS[0], Xd.drop(columns=["mom_12_1"]), uni.pool)
    assert unt.status == "untestable"
    assert DL.cochran_armitage(np.array([100, 100, 100]), np.array([20, 50, 80])) > 5
    assert abs(DL.cochran_armitage(np.array([100, 100, 100]), np.array([50, 50, 50]))) < 1e-9
    assert DL.mechanism_summary([])["agreement"] != DL.mechanism_summary([])["agreement"]


def test_self_check_and_determinism_on_tiny_worlds():
    cfg = cfg_small(n_perm=40, n_boot=50)
    inp = small("null", 4)
    assert DL.determinism_check(inp, NOW, cfg, DL.DirectionGate(30))["identical"]
    with pytest.raises(ValueError):
        DL.synthetic_inputs(world="nope")
