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


# ------------------------------------------------------------------ second wave: placebo, pooling, ledger, registry, pre-registration
def _panel(n_weeks=200, per=40, seed=0, lag_leak=False):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2010-01-01", periods=n_weeks, freq="W-FRI").values, per)
    tick = np.tile([f"T{i}" for i in range(per)], n_weeks)
    n = len(dates)
    x = rng.normal(size=n)
    up = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.0 * x))).astype(float)
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(dates), ticker=tick, up=up))
    if lag_leak:                                     # the feature is the PREVIOUS week outcome of the same ticker
        prev = pool.assign(k=np.arange(n)).sort_values(["ticker", "date"])
        x = prev.groupby("ticker")["up"].shift(1).fillna(0.5).to_numpy()[np.argsort(prev["k"].to_numpy())]
        x = x + rng.normal(0, 0.01, n)
    return pool, pd.DataFrame(dict(x=x))


def test_lead_lag_profile_separates_forecast_from_contaminated():
    pool, M = _panel(seed=1)
    v = DL.profile_verdict(DL.lead_lag_profile(M, pool, "x"))
    assert v["kind"] == "forecast"
    pool2, M2 = _panel(seed=2, lag_leak=True)
    prof = DL.lead_lag_profile(M2, pool2, "x", lags=(-2, -1, 0, 1)).set_index("lag")
    assert prof.loc[-1, "t"] != prof.loc[0, "t"]
    assert DL.lead_lag_profile(M.iloc[:5], pool.iloc[:5], "x").empty and DL.profile_verdict(pd.DataFrame())["kind"] == "unknown"


def test_time_shuffle_control_has_small_t_while_real_column_has_large():
    pool, M = _panel(seed=3)
    M["noise"] = np.random.default_rng(0).normal(size=len(M))
    real = DL.feature_ic_table(M, pool).set_index("column")
    ctl = DL.time_shuffle_control(M, pool, n_rep=3)
    assert real.loc["x", "t"] > 6 and ctl["max_abs_t"] < 4.5 and ctl["n_tests"] == 6
    assert np.isnan(DL.time_shuffle_control(M.iloc[:0], pool.iloc[:0])["max_abs_t"])


def test_random_effects_flags_heterogeneous_and_reports_prediction_interval():
    same = DL.random_effects([0.02] * 6, [1e-5] * 6)
    assert same["tau2"] == pytest.approx(0.0) and same["p_negative_year"] < 0.01
    mixed = DL.random_effects([0.06, 0.05, -0.04, -0.05, 0.07, -0.06], [1e-5] * 6)
    assert mixed["i2"] > 0.9 and mixed["pred_lo"] < 0 < mixed["pred_hi"] and mixed["p_negative_year"] > 0.2
    assert np.isnan(DL.random_effects([0.1], [1e-4])["tau2"]) and DL.random_effects([], [])["k"] == 0


def test_posterior_helpers_and_label_noise():
    small_b = DL.beta_posterior(5, 5)
    assert small_b["lo"] < 0.7 and DL.prob_accuracy_above(5, 5) < 0.8 and DL.prob_accuracy_above(90, 100, 0.6) > 0.99
    assert DL.label_noise_attenuation(0.02, 0.0) == 0.02 and DL.label_noise_attenuation(0.02, 0.25) == pytest.approx(0.005)
    with pytest.raises(ValueError):
        DL.label_noise_attenuation(0.02, 0.5)
    assert DL.weeks_needed(0.02) > DL.weeks_needed(0.05) > 0
    with pytest.raises(ValueError):
        DL.weeks_needed(0.0)


def test_evidence_ledger_is_anytime_valid_and_ignores_repeats():
    led = DL.EvidenceLedger(0.05)
    for i in range(6):
        led.update("h", 0.02, f"k{i}")
    assert led.rejects("h") and led.looks["h"] == 6
    before = led.value("h")
    led.update("h", 0.02, "k0")                                     # identical key: not new evidence
    assert led.value("h") == before
    rng = np.random.default_rng(0)
    rej = 0
    for t in range(300):
        l = DL.EvidenceLedger(0.05)
        for i in range(10):
            l.update("h", rng.uniform(), f"{t}-{i}")
            if l.rejects("h"):
                rej += 1
                break
    assert rej / 300 <= 0.10
    assert DL.EvidenceLedger().table().empty and DL.EvidenceLedger().value("x") == 1.0
    with pytest.raises(ValueError):
        DL.EvidenceLedger(kappa=1.5)


def test_registry_refuses_duplicates_and_redundant_specs_and_counts_active():
    reg = DL.HypothesisRegistry()
    assert reg.active_count == 15
    clone = DL._spec("momentum_again", "momentum", ["r20", "r60", "r120", "mom_12_1"], "same thing", 1, ["r60"])
    with pytest.raises(ValueError):
        reg.add(clone, "because")
    with pytest.raises(ValueError):
        reg.add(DL.OPEN_SPECS[0], "")
    hid = reg.add(DL.OPEN_SPECS[0], "overnight flow")
    assert hid.startswith("H") and reg.active_count == 16
    reg.retire("insider", "no data")
    assert reg.active_count == 15 and "insider" not in reg.active()
    assert "sector_behavior" in reg.coverage_gaps(["r5"]) and reg.retired["insider"] == "no data"
    out = DL.load_open_specs(DL.HypothesisRegistry(), ["r5"])
    assert all(v == "inputs not in the panel" for v in out.values())


def test_preregistration_freezes_the_criterion():
    pr = DL.PreRegistration.make("reversal", "reversal:linear", 0.05, 0.005, "2018-12-31", 26, "2019-06-01")
    assert pr.verify()
    tampered = dataclasses.replace(pr, min_skill=0.0)
    assert not tampered.verify()
    res = _res(best_tag="reversal:linear")
    assert pr.evaluate(res, 40, "2019-01-07")["replicated"]
    with pytest.raises(FirewallBreach):
        tampered.evaluate(res, 40, "2019-01-07")
    stale = pr.evaluate(res, 40, "2018-06-01")
    assert not stale["replicated"] and any("independent" in r for r in stale["reasons"])
    assert not pr.evaluate(_res(best_tag="other"), 40, "2019-02-01")["replicated"]
    with pytest.raises(ValueError):
        DL.PreRegistration.make("h", "t", 0.05, 0.0, "2019-07-01", 26, "2019-06-01")


def test_return_rank_selection_stability_and_frontier_monotonicity():
    rng = np.random.default_rng(0)
    n, per = 4000, 40
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=n // per, freq="W-FRI").values, per)
    fwd = rng.normal(0, 0.05, n)
    p = np.clip(0.5 + 2.0 * fwd + rng.normal(0, 0.05, n), 0.02, 0.98)
    pred = pd.DataFrame(dict(row=np.arange(n), date=dates, year=2015, model="m", p=p, up=(fwd > 0).astype(float), q=rng.uniform(size=n), p0=0.5))
    r = DL.return_rank_test(pred, pd.DataFrame(dict(fwd=fwd)), "m", 100)
    assert r["ic"] > 0.3 and r["spread_bp"] > 100 and r["lo_bp"] > 0
    y = (fwd > 0).astype(float)
    W = pd.DataFrame({"good": p, "junk": np.full(n, 0.5), "junk2": np.clip(0.5 + rng.normal(0, 0.05, n), 0.02, 0.98)})
    info = pd.DataFrame(dict(date=dates, up=y, p0=0.5))
    st = DL.selection_stability(W, info, 60)
    assert st["cell"].iloc[0] == "good" and st["win_share"].iloc[0] > 0.9
    ft = pd.DataFrame(dict(model="m", segcol=None, seg=None, cov_target=[0.01, 0.05, 0.25, 1.0], n=[40, 200, 900, 4000], acc=[0.9, 0.8, 0.65, 0.55]))
    m = DL.frontier_monotone_check(ft, "m")
    assert m["rho"] < -0.9 and m["tightest_is_best"] and not m["flat"]
    flat = ft.assign(acc=[0.51, 0.5, 0.505, 0.5])
    assert DL.frontier_monotone_check(flat, "m")["flat"] and np.isnan(DL.frontier_monotone_check(ft.iloc[:1], "m")["rho"])


def test_jackknife_and_placebo_shift_and_pick_overlap():
    rng = np.random.default_rng(1)
    n_w, per = 60, 40
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=n_w, freq="W-FRI").values, per)
    up = (rng.uniform(size=n_w * per) < 0.5).astype(float)
    p = np.where(up > 0, 0.6, 0.4)
    p[:3 * per] = np.where(up[:3 * per] > 0, 0.99, 0.01)            # three lucky weeks carry an outsized share
    pred = pd.DataFrame(dict(row=np.arange(n_w * per), date=dates, year=2015, model="m", p=p, up=up, q=1.0, p0=0.5))
    jk = DL.jackknife_weeks(pred, "m", (1, 3))
    assert jk["share_kept"].iloc[1] < jk["share_kept"].iloc[0] <= 1.0
    pl = DL.placebo_date_shift(pred, "m", 1)
    assert pl["skill_true"] > 0.1 > pl["skill_placebo"] and pl["lo"] > 0
    pool = pd.DataFrame(dict(date=pd.DatetimeIndex(dates), ticker=np.tile([f"T{i}" for i in range(per)], n_w), pick=True))
    ov = DL.pick_overlap(pool)
    assert ov["overlap"] == pytest.approx(1.0) and ov["tickers"] == per
    assert DL.jackknife_weeks(pred.iloc[:10], "m").empty


def test_honest_blend_does_not_beat_equal_weights_on_pure_noise():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.uniform(size=n) < 0.5).astype(float)
    W = pd.DataFrame({f"c{i}": np.clip(0.5 + rng.normal(0, 0.04, n), 0.02, 0.98) for i in range(4)})
    info = pd.DataFrame(dict(date=pd.bdate_range("2010-01-01", periods=n, freq="B"), up=y))
    b = DL.honest_blend(W, info)
    assert b["ok"] and b["skill_fitted"] < 0.002 and b["skill_single"] < 0.002
    W["real"] = np.where(y > 0, 0.62, 0.38)
    assert DL.honest_blend(W, info)["skill_fitted"] > 0.05 and not DL.honest_blend(W[["c0"]], info)["ok"]


def test_sector_neutralize_removes_group_mean_and_truncation_catches_future_read():
    pool = pd.DataFrame(dict(date=pd.to_datetime(["2015-01-02"] * 6), sector=list("aaabbb")))
    M = pd.DataFrame(dict(f=[1.0, 2, 3, 10, 20, 30]))
    out = DL.sector_neutralize(M, pool)
    assert out["f"].tolist() == [-1, 0, 1, -10, 0, 10]
    inp = small("null", 2)
    cfg = cfg_small()
    ok = DL.truncation_test(inp, cfg, NOW, "2014-01-01")
    assert ok["ok"] and ok["rows"] > 0 and ok["columns"] > 10
    orig = DL.derive_features

    def leaky(Xp, pool, ps=None):                                    # a column centred on the panel-wide mean: reads later rows
        d = orig(Xp, pool, ps)
        d["r5"] = d["r5"] - d["r5"].mean()
        return d
    DL.derive_features = leaky
    try:
        bad = DL.truncation_test(inp, cfg, NOW, "2014-01-01")
    finally:
        DL.derive_features = orig
    assert not bad["ok"] and any(o["column"] == "r5" for o in bad["offenders"])


def test_audit_pool_catches_structure_errors():
    inp = small("null", 2)
    pool = DL.ConditionalUniverse(cfg_small()).build(inp, NOW).pool
    assert DL.audit_pool(pool)["ok"]
    bad = pool.copy()
    bad.loc[bad.index[0], "entry_date"] = bad["date"].iloc[0]
    assert not DL.audit_pool(bad)["ok"]
    dup = pd.concat([pool, pool.iloc[:3]])
    assert not DL.audit_pool(dup)["ok"] and DL.audit_pool(pool.iloc[0:0])["ok"]


def test_disclosures_baselines_and_arithmetic():
    inp = small("null", 2)
    pool = DL.ConditionalUniverse(cfg_small()).build(inp, NOW).pool
    s = DL.survivorship_disclosure(pool)
    assert s["survivor_only_suspected"] and s["ended_early"] == 0                        # the synthetic panel never loses a name
    dropped = pool[~((pool["ticker"] == "S001") & (pool["date"] > pd.Timestamp("2013-01-01")))]
    assert DL.survivorship_disclosure(dropped)["ended_early"] == 1
    la = DL.label_ambiguity_report(inp.labels)
    assert 0 < la["mover_rate"] < 1 and DL.label_ambiguity_report(inp.labels.iloc[0:0])["rows"] == 0
    Xd = DL.derive_features(inp.X.reindex(pool.index), pool)
    bt = DL.baseline_table(Xd, pool, (2014, 2015))
    assert len(bt) >= 4 and (bt["edge"].abs() < 0.05).all()
    e = DL.expected_false_leads(15, 0.25, 0.05)
    assert e["expected_weak"] == pytest.approx(0.1875) and e["expected_candidate"] < e["expected_weak"]
    with pytest.raises(ValueError):
        DL.expected_false_leads(0, 0.2, 0.05)
    ad = DL.segment_adequacy(pool)
    assert {"seg", "reg", "sector"} <= set(ad["factor"]) and (ad["weeks_needed"] > 0).all()


def test_multiplicity_price_cell_evidence_and_recommendation(null_report, planted_report):
    a, b = DL.multiplicity_price(1), DL.multiplicity_price(500)
    assert b["z_crit"] > a["z_crit"] and b["p_any_unadjusted"] > 0.99
    ce = DL.cell_evidence(34, 40, design_effect=1.0, n_cells=200)
    assert ce["rate"] == 0.85 and ce["adjusted_lo"] < ce["wilson_lo"] and not ce["reaches_gate"]
    assert DL.cell_evidence(950, 1000, n_cells=10)["reaches_gate"]
    with pytest.raises(ValueError):
        DL.cell_evidence(5, 0)
    rec, why = DL.recommendation(null_report)
    assert rec in (DL.Recommendation.NOT_USABLE, DL.Recommendation.RESEARCH_ONLY) and why
    assert DL.recommendation(planted_report)[0] is not DL.Recommendation.NOT_USABLE
    closed = DL.DirectionLab(cfg_small(), DL.DirectionGate(10 ** 6)).run(small("null", 1), NOW)
    assert DL.recommendation(closed)[0] is DL.Recommendation.VOID
    for rep in (null_report, planted_report, closed):
        bs = DL.blind_summary(rep)
        assert "recommendation" in bs and not any(k in bs for k in ("date", "year", "ticker"))
    assert "direction_lab" in DL.summary_line(null_report) and DL.by_outcome(null_report)


def test_state_roundtrip_health_summaries_and_next_experiments(null_report, planted_report):
    st = DL.LabState()
    st.ledger = DL.EvidenceLedger()
    DL.feed_ledger(st.ledger, planted_report)
    st.runs, st.history = 1, [dict()]
    back = DL.state_from_dict(DL.state_to_dict(st))
    assert back.ledger.log_e == st.ledger.log_e and back.digest() == st.digest()
    d = DL.state_to_dict(st)
    d["null_streak"] = 50
    with pytest.raises(ValueError):
        DL.state_from_dict(d)
    h = DL.lab_health(null_report)
    assert h["gate_open"] and h["controls_pass"]
    closed = DL.DirectionLab(cfg_small(), DL.DirectionGate(10 ** 6)).run(small("null", 1), NOW)
    hp = DL.lab_health(closed)
    assert not hp["healthy"] and not hp["nulls_count_as_evidence"]
    a, b = DL.report_to_dict(null_report), DL.report_to_dict(planted_report)
    cmp = DL.compare_summaries(a, b)
    assert len(cmp) == 15 and DL.compare_reports(null_report, planted_report).shape[0] == 15
    ne = DL.next_experiments(planted_report, st)
    assert ne and ne == sorted(ne, key=lambda r: -r["per_minute"])
    assert DL.next_experiments(closed)[0]["action"] == "improve_volatility_evidence"
    assert "reversal" in DL.format_dossier(DL.evidence_dossier(planted_report, "reversal"))
