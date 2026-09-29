"""Tests for engine/research/frontier.py (C66 section 11). Symmetry tests are in tests/test_research_symmetry.py.
Synthetic data only. Every mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach, Epistemic, Lifecycle, DecisionEffect
from engine.research import frontier as F

CFG = F.FrontierConfig(n_boot=120, n_sim=80, seed=1)
NOW = "2030-01-01"


@pytest.fixture(scope="module")
def planted():
    return F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="planted")


# ---------------------------------------------------------------- frontier: validation and basics
def test_config_validation():
    assert F.FrontierConfig().validate() == []
    assert F.FrontierConfig(coverages=(0.5, 0.5)).validate()
    assert F.FrontierConfig(gate=0.4).validate()
    with pytest.raises(ValueError):
        F.FrontierConfig(n_blocks=1).check()


def test_predictions_rejects_bad_input():
    good = F.synthetic_predictions(0, n_weeks=10, per_week=20).frame[["date", "ticker", "p", "up", "matured_at"]]
    bad = good.copy()
    bad["matured_at"] = bad["date"]
    with pytest.raises(ValueError):
        F.Predictions(bad)
    bad = good.copy()
    bad.loc[0, "p"] = 1.5
    with pytest.raises(ValueError):
        F.Predictions(bad)
    with pytest.raises(ValueError):
        F.Predictions(good.drop(columns=["p"]))
    with pytest.raises(ValueError):
        F.Predictions(pd.concat([good, good.iloc[:1]]))


def test_survival_and_thresholds():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    q = F.survival(conf)
    assert q == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    thr = F.fit_thresholds(conf, [1.0, 0.4])
    assert thr[1.0] == -math.inf and thr[0.4] == 0.8
    assert F.fit_thresholds(np.array([]), [0.5])[0.5] == math.inf      # fail closed: empty calibration takes nothing


def test_matured_before_hides_unmatured(planted):
    cut = planted.frame["matured_at"].sort_values().iloc[len(planted) // 2]
    vis = planted.matured_before(cut)
    assert 0 < len(vis) < len(planted)
    assert (vis.frame["matured_at"] < cut).all()


# ---------------------------------------------------------------- frontier: planted region found, fakes refused
def test_planted_region_is_found(planted):
    r = F.assess(planted, NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.CANDIDATE
    c = r.frontier.cell(0.10)
    assert c.accuracy > 0.85 and c.eff_lo > 0.80
    assert "validated" not in r.answer.headline.lower().replace("not validated", "")


def test_null_world_says_not_found():
    r = F.assess(F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="null"), NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.NOT_FOUND
    assert r.answer.headline == F.NOT_FOUND_SENTENCE


def test_tiny_sample_is_flagged_not_believed():
    P = F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="tiny")
    r = F.assess(P, NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.NOT_FOUND
    assert r.answer.tiny_sample_coverages                         # the 100%-accurate sliver was recognised as such
    tiny = {t.coverage: t for t in r.tiny}
    assert any(t.verdict == F.TinyVerdict.TINY_SAMPLE_ARTIFACT for t in tiny.values())


def test_lopsided_region_fails_transfer_or_stability():
    r = F.assess(F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="lopsided"), NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.NOT_FOUND


def test_overconfident_region_fails_calibration():
    r = F.assess(F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="overconf"), NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.NOT_FOUND


def test_selfcheck_is_ok():
    assert F.frontier_selfcheck(0, F.FrontierConfig(n_boot=80, n_sim=50, seed=0), n_weeks=45)["ok"]


def test_controls_are_sound(planted):
    res = F.instrument_controls(planted, NOW, CFG)
    assert res["random"] == "NOT_FOUND" and res["shuffled"] == "NOT_FOUND" and res["canary"] != "NOT_FOUND"
    assert res["sound"] == "True"


def test_unmatured_rows_are_dropped_and_counted(planted):
    cut = planted.frame["matured_at"].sort_values().iloc[len(planted) // 2]
    r = F.assess(planted, cut, CFG, code_hash="t")
    assert r.excluded_unmatured > 0 and r.frontier.n_rows + r.excluded_unmatured == len(planted)


# ---------------------------------------------------------------- frontier: statistics
def test_design_effect_grows_with_clustering():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(40), 25)
    indep = (rng.random(1000) < 0.5).astype(float)
    clustered = np.repeat((rng.random(40) < 0.5).astype(float), 25)
    assert F.design_effect(indep, codes) < 1.5
    assert F.design_effect(clustered, codes) > 10


def test_cluster_bootstrap_interval_contains_truth():
    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(60), 20)
    correct = (rng.random(len(codes)) < 0.7).astype(float)
    lo, hi, se = F.cluster_bootstrap(correct, codes, np.ones(len(codes), bool), np.random.default_rng(1), 300, 0.95)
    assert lo < 0.7 < hi and se > 0
    assert math.isnan(F.cluster_bootstrap(correct[:40], codes[:40], np.ones(40, bool), np.random.default_rng(1), 100, 0.95)[0])


def test_selection_null_makes_small_regions_look_lucky():
    P = F.synthetic_predictions(0, n_weeks=40, per_week=40, kind="null")
    null = F.selection_null(P, dataclasses.replace(CFG, n_sim=100))
    assert null.p_family[0.01] > 0.05                              # nothing survives the family correction in a null world


def test_null_max_accuracy_shrinks_with_n():
    rng = np.random.default_rng(0)
    assert F.null_max_accuracy(10, 0.5, rng) > F.null_max_accuracy(1000, 0.5, rng) + 0.2
    assert math.isnan(F.null_max_accuracy(0, 0.5, rng))


def test_walk_forward_thresholds_never_see_test_block(planted):
    o = F.walk_forward_frontier(planted, CFG)
    assert o.purged_rows > 0                                       # rows whose outcome was unknown at the boundary were excluded
    c = o.cell(0.10)
    assert c is not None and 0.05 < c.coverage_real < 0.2 and c.accuracy > 0.8


def test_evidence_planning():
    assert math.isinf(F.required_independent_obs(0.79))
    assert F.required_independent_obs(0.9) < F.required_independent_obs(0.83)
    assert F.detectable_accuracy(200) > F.detectable_accuracy(2000)
    assert math.isinf(F.detectable_accuracy(5))            # 5 independent calls could not certify even a perfect region


def test_compare_frontiers_detects_change_and_no_change(planted):
    a = F.compute_frontier(planted, CFG, with_null=False)
    b = F.compute_frontier(planted, CFG, with_null=False)
    assert not any(c.significant for c in F.compare_frontiers(a, b, CFG))
    worse = F.compute_frontier(F.random_control(planted), CFG, with_null=False)
    assert any(c.significant for c in F.compare_frontiers(a, worse, CFG))


def test_paired_comparison_prefers_informative_model(planted):
    rnd = F.random_control(planted, 2)
    r = F.paired_model_comparison(planted, rnd, ("real", "random"), 0.10, CFG)
    assert r.favours == "real" and r.diff > 0.2


def test_empty_frontier_and_answer():
    empty = F.Predictions(pd.DataFrame({c: pd.Series(dtype=float) for c in F.REQUIRED_COLUMNS}))
    r = F.assess(empty, NOW, CFG, code_hash="t")
    assert r.answer.status == F.Eighty.INSUFFICIENT_DATA
    assert r.answer.headline.startswith(F.NOT_FOUND_SENTENCE)
    assert all(c.n == 0 for c in r.frontier.cells)


def test_public_summary_is_identity_free(planted):
    r = F.assess(planted, NOW, CFG, code_hash="t")
    txt = str(F.public_summary(r))
    assert "S0" not in txt and "2015" not in txt
    rec = r.to_matured_record()
    with pytest.raises(FirewallBreach):
        rec.gate(r.provenance.learned_at)                          # not usable on the day it matured
    assert rec.gate("2031-01-01")["status"] == "CANDIDATE"


def test_dump_and_tamper_detection(tmp_path, planted):
    r = F.assess(planted, NOW, CFG, code_hash="t")
    p = tmp_path / "r.json"
    F.dump_report(r, p)
    assert F.load_report_summary(p)["summary"]["status"] == "CANDIDATE"
    p.write_text(p.read_text().replace("CANDIDATE", "NOT_FOUND", 1))
    with pytest.raises(FirewallBreach):
        F.load_report_summary(p)


# ---------------------------------------------------------------- ledger, coverage tracker, sweep, state
def test_ledger_cumulative_correction_and_no_reroll(tmp_path):
    L = F.TestLedger(tmp_path / "l.jsonl")
    L.log("a", "x", 0.001, "2020-01-01", "2020-02-01")
    assert L.survivors() == ["a"]
    L.log("a", "x", 0.9, "2020-01-01", "2020-02-01")                # re-logging keeps the FIRST p
    assert L.entries()[0].p == 0.001
    L.log_null("bulk", "x", 5000, "2020-01-01", "2020-02-01")       # a big search raises the price
    assert "a" not in L.survivors()
    assert len(L) == 5001


def test_ledger_refuses_future_evidence():
    L = F.TestLedger()
    with pytest.raises(FirewallBreach):
        L.log("a", "x", 0.01, "2020-02-01", "2020-02-01")
    with pytest.raises(ValueError):
        L.log("b", "x", 1.5, "2020-01-01", "2020-02-01")


def test_ledger_resumes_from_disk(tmp_path):
    L = F.TestLedger(tmp_path / "l.jsonl")
    for i in range(10):
        L.log(f"t{i}", "x", 0.5, "2020-01-01", "2020-02-01")
    L2 = F.TestLedger(tmp_path / "l.jsonl")
    assert len(L2) == 10 and L2.digest() == L.digest()


def test_ledger_detects_excess_small_p_in_aggregate():
    L = F.TestLedger()
    rng = np.random.default_rng(0)
    for i in range(300):
        L.log(f"n{i}", "x", float(rng.uniform()), "2020-01-01", "2020-02-01")
    assert L.excess_small_p()["p"] > 0.01
    L2 = F.TestLedger()
    for i in range(300):
        L2.log(f"s{i}", "x", float(rng.uniform() * (0.02 if i % 3 == 0 else 1)), "2020-01-01", "2020-02-01")
    assert L2.excess_small_p()["p"] < 1e-6


def test_coverage_tracker_takes_least_covered_and_resumes(tmp_path):
    T = F.CoverageTracker(tmp_path / "c.json")
    units = [("a", "2020", "all"), ("b", "2020", "all"), ("c", "2020", "all")]
    seen = []
    for _ in range(9):
        u = T.next_unit(units)
        seen.append(u)
        T.mark(*u, now="2030-01-01")
    assert {seen.count(u) for u in units} == {3}
    T.save()
    assert F.CoverageTracker(tmp_path / "c.json").counts == T.counts
    assert T.evenness(units) > 0.99 and T.next_unit([]) is None


def test_frontier_sweep_logs_tests_and_accepts_precursors(tmp_path):
    P = F.synthetic_predictions(0, n_weeks=60, per_week=30, kind="planted")
    sw = F.FrontierSweep(F.CoverageTracker(tmp_path / "c.json"), F.TestLedger(tmp_path / "l.jsonl"), F.FrontierConfig(n_boot=80, n_sim=60))
    sw.add_source("base", P)

    class Cand:
        precursor_id, source_family = "p1", "gap"
        frame = F.synthetic_predictions(1, n_weeks=40, per_week=40, kind="null").frame[["date", "ticker", "p", "up", "matured_at"]]
    assert sw.add_precursors([Cand()]) == 1
    with pytest.raises(ValueError):
        sw.add_precursors([object()])
    steps = sw.run("2031-01-01", 3)
    assert all(s.status == "assessed" for s in steps) and len(sw.ledger) > 0
    assert len({s.unit for s in steps}) == 3                        # four different units visited, none repeated


def test_state_step_and_trend():
    st = F.FrontierState(F.FrontierConfig(n_boot=80, n_sim=60))
    st.add(F.synthetic_predictions(0, n_weeks=40, per_week=40, kind="null").frame)
    r = F.step(st, NOW)
    assert r.answer.status == F.Eighty.NOT_FOUND and len(st.history) == 1
    assert st.trend()["ever_found"] is False


def test_transfer_fails_on_lopsided_world():
    P = F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="lopsided")
    t = F.leave_one_out_transfer(P, 0.10, "sector", CFG)
    assert t.verdict in (F.Stability.UNSTABLE, F.Stability.UNTESTED)
    ok = F.leave_one_out_transfer(F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="planted"), 0.10, "sector", CFG)
    assert ok.verdict == F.Stability.STABLE


