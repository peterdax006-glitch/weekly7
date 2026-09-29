"""Tests for engine/research/frontier.py and engine/research/symmetry.py (C66 sections 11 and 12).
Synthetic data only. Every mechanism has a planted case it must catch, a null case where it must find nothing, and the empty case."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach, Epistemic, Lifecycle, DecisionEffect
from engine.research import frontier as F
from engine.research import symmetry as S

CFG = F.FrontierConfig(n_boot=120, n_sim=80, seed=1)
NOW = "2030-01-01"


@pytest.fixture(scope="module")
def planted():
    return F.synthetic_predictions(0, n_weeks=45, per_week=40, kind="planted")


@pytest.fixture(scope="module")
def sym():
    return S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)


@pytest.fixture(scope="module")
def sym_null():
    return S.synthetic_symmetry_frame(0, "null", n_weeks=45, per_week=50)


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


# ================================================================ symmetry
def test_symframe_requires_unfired_universe():
    f = pd.DataFrame({"pattern_id": "p", "date": pd.to_datetime(["2020-01-06"] * 3), "ticker": list("abc"), "call": [1, 1, -1],
                      "fwd": [0.06, -0.06, 0.0]})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="only fired rows"):
        S.SymFrame(f)
    assert len(S.SymFrame(f, require_universe=False)) == 3


def test_symframe_rejects_bad_rows(sym):
    base = sym.frame.drop(columns=S.SymFrame._derived_cols()).iloc[:200].copy()
    b = base.copy()
    b["matured_at"] = b["date"]
    with pytest.raises(ValueError):
        S.SymFrame(b, require_universe=False)
    b = base.copy()
    b.loc[b.index[0], "call"] = 2
    with pytest.raises(ValueError):
        S.SymFrame(b, require_universe=False)


def test_confusion_arithmetic():
    c = S.confusion_of(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool), S.Space.WINNER)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3) and c.n == 5
    assert math.isnan(S.confusion_of(np.zeros(4, bool), np.ones(4, bool), S.Space.LOSER).precision)


def test_winner_only_pattern_is_not_trusted(sym):
    lib = S.analyse_library(sym)
    w = lib.get("winners")
    assert w.trust.verdict == S.Trust.NOT_TRUSTED
    assert w.loser.recall == 0.0 and w.winner.precision > 0.3       # finds winners, finds no losers
    kinds = {s.kind for s in w.failing_slices()}
    assert S.CaseKind.LOW_LIQUIDITY_FAILURE in kinds and S.CaseKind.LARGE_LOSS in kinds


def test_noise_pattern_is_never_trusted(sym, sym_null):
    for frame in (sym, sym_null):
        assert S.analyse_library(frame).get("noise").trust.verdict != S.Trust.TRUSTED


def test_good_pattern_gets_a_verdict_and_more_than_winners_only(sym):
    g = S.analyse_library(sym).get("good")
    assert g.loser.recall > 0.2 and g.winner.recall > 0.2 and g.hit_rate > 0.9
    assert g.errors.long_adverse_ratio < 1


def test_context_never_recorded_is_untested_not_ok(sym):
    f = sym.frame.drop(columns=S.SymFrame._derived_cols() + ["external_event", "regime_transition", "mover_outcome"])
    sf = S.SymFrame(f, sym.cfg, require_universe=False)
    p = S.analyse_pattern(sf, "good")
    assert p.slice_of(S.CaseKind.EXTERNAL_EVENT).status == S.SliceStatus.UNTESTED
    assert p.slice_of(S.CaseKind.EPISODE_SPIKED).status == S.SliceStatus.UNTESTED
    assert p.trust.verdict != S.Trust.TRUSTED or not p.trust.untested


def test_too_few_calls_is_unknown():
    sf = S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)
    small = sf.frame[sf.frame["pattern_id"] == "good"].head(300).drop(columns=S.SymFrame._derived_cols())
    p = S.analyse_pattern(S.SymFrame(small, sf.cfg, require_universe=False), "good")
    assert p.trust.verdict == S.Trust.UNKNOWN


def test_missed_movers_by_reason():
    f = pd.DataFrame({"pattern_id": "p", "date": pd.to_datetime(["2020-01-06"] * 6), "ticker": list("abcdef"),
                      "call": [1, 0, 0, -1, 0, 0], "lean": [1, 1, -1, -1, 1, 0], "margin": [0.5, -0.05, -0.5, 0.3, -0.5, 0.0],
                      "fwd": [0.08, 0.09, -0.08, 0.08, 0.10, 0.0]})
    f["matured_at"] = f["date"] + pd.Timedelta(days=7)
    sf = S.SymFrame(f, S.SymmetryConfig(), require_universe=False)
    m = S.miss_breakdown(sf.frame)
    assert m.winners_total == 4 and m.winner_caught == 1
    assert m.winner_near_miss == 1 and m.winner_wrong_way == 1 and m.winner_silent == 1
    assert m.losers_total == 1 and m.loser_silent == 1


def test_orphans_and_union_recall(sym):
    o = S.orphan_movers(sym)
    assert 0 < o["orphan_share"] < 1
    u = S.union_recall(sym)
    assert u["cum_winner_recall"].is_monotonic_increasing and u["cum_loser_recall"].is_monotonic_increasing


def test_bootstrap_confusion_interval(sym):
    f = sym.of("good")
    ci = S.bootstrap_confusion(f, np.random.default_rng(0), 150)
    assert ci["win_precision"].lo < ci["win_precision"].value < ci["win_precision"].hi
    assert math.isnan(S.bootstrap_confusion(f.iloc[:0], np.random.default_rng(0))["win_recall"].value)


def test_episode_outcomes_find_planted_kind():
    sf = S.synthetic_symmetry_frame(0, "planted", n_weeks=45, per_week=50)
    rows = S.episode_outcomes(sf, "good")
    assert len(rows) == 5 and not any(r.q < 0.01 for r in rows)      # episodes are random in the synthetic world: no false discovery
    f = sf.frame.drop(columns=S.SymFrame._derived_cols()).copy()
    f.loc[f["mover_outcome"] == "spiked", "fwd"] = 0.08
    rows = S.episode_outcomes(S.SymFrame(f, sf.cfg), "good")
    assert next(r for r in rows if r.episode == "spiked").q < 0.001
    assert S.episode_outcomes(S.SymFrame(f.iloc[:0], sf.cfg, require_universe=False)) == []


def test_dose_response_and_context_symmetry(sym):
    d = S.dose_response(sym.of("good"), "long")
    assert d.bins == 4 or d.bins == 0
    cs = S.context_symmetry(sym, "winners")
    assert any(c.dimension == "liquidity" and c.value == "low" and c.verdict == S.SliceStatus.FAILS for c in cs)
    assert not any(c.verdict == S.SliceStatus.FAILS for c in S.context_symmetry(S.synthetic_symmetry_frame(0, "null", n_weeks=45, per_week=50), "good"))


def test_pattern_overlap_and_blindness(sym):
    ov = S.pattern_overlap(sym)
    assert len(ov) == 3 and all(0 <= (o.winner_jaccard if math.isfinite(o.winner_jaccard) else 0) <= 1 for o in ov)
    assert len(S.mover_blindness(sym)) >= 1


# ---------------------------------------------------------------- loss-risk bank
@pytest.fixture(scope="module")
def mined(sym):
    return S.attach_mean_loss(S.mine_loss_risks(sym, NOW, S.MiningConfig(per_pattern=True)), sym, NOW)


def test_mining_finds_planted_loss_contexts(mined):
    assert any(("liquidity", "low") in i.context and i.oos_confirmed for i in mined)          # the planted low-liquidity weakness
    assert any(i.kind == S.LossKind.REGIME_SHIFT_RISK and i.oos_confirmed for i in mined)


def test_mining_finds_nothing_significant_in_null_world(sym_null):
    ledger = F.TestLedger()
    items = S.mine_loss_risks(sym_null, NOW, S.MiningConfig(q_max=0.01, per_pattern=False), ledger)
    assert [i for i in items if i.oos_confirmed and i.relative_risk > 2.0] == []


def test_mining_ignores_unmatured_rows(sym):
    early = sym.frame["matured_at"].sort_values().iloc[len(sym) // 10]
    assert S.mine_loss_risks(sym, early, S.MiningConfig()) == []


def test_bank_add_is_point_in_time_and_append_only(tmp_path, mined):
    bank = S.LossRiskBank(tmp_path / "b.jsonl")
    it = mined[0]
    with pytest.raises(FirewallBreach):
        bank.add(it, it.matured_at)                                # evidence not strictly older than now
    assert bank.add(it, NOW) == "added" and bank.add(it, NOW) == "stale"
    assert bank.items(it.matured_at) == []                         # invisible at its own maturity date
    assert len(bank.items(NOW)) == 1
    assert bank.verify_chain() and len(S.LossRiskBank(tmp_path / "b.jsonl")) == 1


def test_bank_file_tamper_fails_closed(tmp_path, mined):
    p = tmp_path / "b.jsonl"
    bank = S.LossRiskBank(p)
    bank.add(mined[0], NOW)
    p.write_text(p.read_text().replace('"n":', '"n" :', 1) if False else p.read_text().replace("0.", "9.", 1))
    with pytest.raises((FirewallBreach, ValueError, Exception)):
        S.LossRiskBank(p)


def test_item_validation_blocks_opportunity_effects(mined):
    it = mined[0]
    assert it.validate() == []
    bad = dataclasses.replace(it, effects=(DecisionEffect.RANKING,))
    assert any("not allowed" in e for e in bad.validate())
    assert dataclasses.replace(it, risk_id="OP123").validate()
    with pytest.raises(ValueError):
        S.LossRiskBank().add(bad, NOW)


def test_independence_audit_catches_shared_ids(mined):
    bank = S.LossRiskBank()
    bank.add(mined[0], NOW)
    assert S.independence_audit(bank, ["other"]) == []
    assert S.independence_audit(bank, [mined[0].risk_id])


def test_query_multiplier_only_shrinks_and_retire_stops_it(mined):
    bank = S.LossRiskBank()
    for it in mined:
        if it.oos_confirmed:
            bank.add(it, NOW)
    top = max((i for i in bank.items(NOW)), key=lambda i: i.weight())
    ctx = dict(top.context)
    m, ids = S.risk_multiplier(bank, NOW, ctx, top.pattern_id if top.pattern_id != "*" else None)
    assert 0.05 <= m < 1.0 and top.risk_id in ids
    assert S.risk_multiplier(bank, NOW, {"vol": "nonexistent"})[0] == 1.0
    for i in ids:
        bank.retire(i, "test", NOW)
    assert S.risk_multiplier(bank, NOW, ctx, top.pattern_id if top.pattern_id != "*" else None)[0] == 1.0
    assert bank.get(top.risk_id).lifecycle == Lifecycle.RETIRED      # retired, still readable


def test_trader_caution_is_identity_free(mined):
    bank = S.LossRiskBank()
    for it in mined[:10]:
        bank.add(it, NOW)
    out = S.trader_caution(bank, NOW, {"vol": "high"})
    assert set(out) == {"size_multiplier", "abstain", "risk_kinds", "items_matched"}
    with pytest.raises(Exception):
        S.trader_caution(bank, NOW, {"ticker": "AAPL 2019-03-04"})


def test_prune_and_priority_and_questions(mined):
    bank = S.LossRiskBank()
    for it in mined:
        bank.add(it, NOW)
    dead = S.prune_bank(bank, NOW)
    assert all(bank.get(d).lifecycle == Lifecycle.RETIRED for d in dead)
    qs = S.research_questions(bank, NOW, "2030-01-02")
    assert all(q.problem.value == "LOSS_AVOIDANCE" for q in qs)
    assert all(S.item_priority(i) >= 0 for i in mined)


def test_avoidance_tradeoff_and_coverage(sym, mined):
    it = next(i for i in mined if ("liquidity", "low") in i.context and i.measure == "large_loss" and i.pattern_id == "winners")
    a = S.avoidance_tradeoff(sym, it, NOW)
    assert a.losses_avoided > 0 and a.net_return_avoided > 0
    bank = S.LossRiskBank()
    for i in mined:
        bank.add(i, NOW)
    cov = S.loss_bank_coverage(bank, sym, NOW)
    assert cov["large_losses"] > 0 and cov["warned_share"] > 0.3


def test_revalidation_degrades_when_risk_disappears(sym, mined):
    it = next(i for i in mined if i.oos_confirmed and i.pattern_id == "*")
    bank = S.LossRiskBank()
    bank.add(dataclasses.replace(it, matured_at="2015-01-01"), NOW)
    calm = sym.frame.drop(columns=S.SymFrame._derived_cols()).copy()
    calm["fwd"] = np.where(calm["fwd"] < -0.15, -0.02, calm["fwd"])
    calm["fwd"] = np.where(calm["fwd"].abs() >= 0.05, calm["fwd"] * 0.5, calm["fwd"])
    new = S.revalidate_item(bank, it.risk_id, S.SymFrame(calm, sym.cfg), NOW)
    assert new.lifecycle in (Lifecycle.DEGRADED, Lifecycle.RETIRED) or new.oos_confirmed is not True


def test_walk_forward_bank_reduces_large_losses(sym):
    ev = S.evaluate_bank_walk_forward(sym, NOW, S.MiningConfig(per_pattern=False, min_ctx_n=20, min_losses=4, q_max=0.2))
    assert ev.items_used > 0 and ev.scaled_calls > 0
    assert ev.big_loss_sum_scaled < ev.big_loss_sum_unscaled and ev.helps is True
    assert S.evaluate_bank_walk_forward(sym, "2000-01-01").items_used == 0


# ---------------------------------------------------------------- state, entry point, sweep, output
def test_step_end_to_end_and_null(sym, sym_null):
    st = S.SymmetryState(mc=S.MiningConfig(per_pattern=False))
    st.add(sym)
    r = S.step(st, NOW, code_hash="t")
    assert r.independence_problems == () and r.ledger_size > 0
    assert r.library.get("winners").trust.verdict == S.Trust.NOT_TRUSTED
    rec = r.to_matured_record(st.bank)
    with pytest.raises(FirewallBreach):
        rec.gate(r.provenance.learned_at)
    assert str(rec.gate("2031-01-01")["patterns"][0]["id"]).startswith("P")
    st2 = S.SymmetryState(mc=S.MiningConfig(per_pattern=False, q_max=0.01))
    st2.add(sym_null)
    r2 = S.step(st2, NOW, code_hash="t")
    assert not [i for i in st2.bank.items(NOW) if i.relative_risk > 2.0 and i.oos_confirmed]


def test_step_on_empty_state():
    st = S.SymmetryState()
    r = S.step(st, NOW, code_hash="t")
    assert r.library.patterns == () and r.added == () and len(st.bank) == 0


def test_public_summary_hides_identity(sym):
    lib = S.analyse_library(sym)
    txt = str(S.public_summary(lib))
    assert "good" not in txt and "T0" not in txt and "2016" not in txt


def test_symmetry_sweep_visits_least_covered_and_resumes(tmp_path, sym):
    tr, led = F.CoverageTracker(tmp_path / "c.json"), F.TestLedger(tmp_path / "l.jsonl")
    sw = S.SymmetrySweep(tr, led, S.LossRiskBank(tmp_path / "b.jsonl"), mc=S.MiningConfig(per_pattern=False, min_ctx_n=30))
    sw.add_source("fam", sym)

    class Cand:
        precursor_id, source_family = "pc", "gapdown"
        frame = sym.frame[sym.frame["pattern_id"] == "good"].drop(columns=S.SymFrame._derived_cols()).head(4000)
    assert sw.add_precursors([Cand()]) == 1
    steps = sw.run("2031-01-01", 3)
    assert len({s.unit for s in steps}) == 3
    assert F.CoverageTracker(tmp_path / "c.json").counts == tr.counts
    with pytest.raises(ValueError):
        sw.add_precursors([object()])


def test_render_functions_run(sym):
    lib = S.analyse_library(sym)
    txt = S.render_library(lib)
    assert "NOT VALIDATED" in txt and "winners" in txt
    bank = S.LossRiskBank()
    assert "0 items" in S.render_bank(bank, NOW)
    assert "LOSS-FIRST" in S.loss_first_report(lib, bank, NOW)
    assert S.compare_libraries(lib, lib) == []
