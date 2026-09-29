"""S05 tests: missed-winner learning (contract section 22; D13 missed-winner analysis, D14 counterfactual distinction discovery).
Synthetic weeks with a PLANTED distinction (missed winners are high on f1 and low on f2) and a NULL version with none."""
import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from engine.learning import missed_winners as M
from engine.learning.core import DecisionEffect, Epistemic, FirewallBreach, Subsystem

RR = M.RejectionReason
FEATS = ["f0", "f1", "f2", "f3", "f4", "f5"]


def make_weeks(n=60, n_cand=60, k=8, seed=0, planted=True, fp_rule=False):
    """Selection ranks on f0 (blind to f1/f2). Winners are 5% baseline; with `planted`, 40% inside the region f1>0.7 & f2<0.35."""
    rng = np.random.default_rng(seed)
    start = dt.date(2019, 1, 7)
    weeks = []
    for w in range(n):
        F = {f: rng.random(n_cand) for f in FEATS}
        region = (F["f1"] > 0.7) & (F["f2"] < 0.35)
        pwin = np.where(region, 0.40, 0.05) if planted else np.full(n_cand, 0.12)
        win = rng.random(n_cand) < pwin
        fwd = np.where(win, rng.uniform(0.07, 0.15, n_cand), rng.normal(-0.01, 0.03, n_cand).clip(max=0.06))
        score = F["f0"] + 0.05 * rng.normal(size=n_cand)
        rank = (-score).argsort().argsort() + 1
        cands = tuple(M.Candidate(cid=f"w{w}c{i}", features={f: float(F[f][i]) for f in FEATS}, fwd=float(fwd[i]), picked=bool(rank[i] <= k),
                                  score=float(score[i]), rank=int(rank[i]), confidence=0.7, reliability=0.8) for i in range(n_cand))
        d0 = start + dt.timedelta(days=7 * w)
        weeks.append(M.Week(label=f"wk{w}", decided_at=str(d0), resolved_at=str(d0 + dt.timedelta(days=7)), candidates=cands, k=k,
                            era="early" if w < n // 2 else "late"))
    return weeks


PARAMS = M.MissedParams(n_perm=120, n_boot=60, boot=400, min_test_weeks=5)


@pytest.fixture(scope="module")
def planted_weeks():
    return make_weeks(planted=True)


@pytest.fixture(scope="module")
def null_weeks():
    return make_weeks(planted=False, seed=1)


# ------------------------------------------------------------------------------------------------ why was it rejected
def cand(**kw):
    base = dict(cid="c", features={"f0": 0.5}, fwd=0.1, picked=False, score=1.0, rank=12, eligible=True)
    base.update(kw)
    return M.Candidate(**base)


WEEK = M.Week("w", "2020-01-06", "2020-01-13", (cand(),), k=10)


def explain(**kw):
    return M.RejectionAnalyzer().explain(cand(**kw), WEEK, 60)


def test_each_section22_reason_is_reachable():
    assert explain(filters_hit=("risk_vol",), eligible=False, rank=4).primary == RR.OVER_AGGRESSIVE_RISK_FILTER
    assert explain(dir_side=-1, selection_side=1).primary == RR.DIRECTION_DISAGREEMENT
    assert explain(timing_blocked=True).primary == RR.TIMING
    assert explain(anti_context=True).primary == RR.WRONG_CONTEXT
    assert explain(reliability=0.05).primary == RR.BAD_RELIABILITY
    assert explain(confidence=0.05, rank=11).primary == RR.WRONG_CONFIDENCE
    assert explain(rank=12).primary == RR.WRONG_RANKING
    assert explain(rank=45).primary == RR.WRONG_RANKING and explain(rank=45).reasons[0].strength < explain(rank=12).reasons[0].strength
    assert explain(missing_frac=0.9, rank=None, score=None).primary == RR.INSUFFICIENT_EVIDENCE
    w = M.Week("w", "2020-01-06", "2020-01-13", (cand(),), k=10, interactions=(("f1", "f2"),))
    r = M.RejectionAnalyzer().explain(cand(features={"f1": 0.8, "f2": 0.8}, rank=None, score=None), w, 60)
    assert r.primary == RR.MISSING_INTERACTION


def test_unknown_is_returned_when_the_trail_supports_nothing():
    r = explain(rank=None, score=None)
    assert r.primary == RR.UNKNOWN and not r.named and r.reasons == ()
    assert explain(filters_hit=("sector_cap",), eligible=False, rank=None).primary == RR.UNKNOWN     # a filter that is not a risk filter


def test_mechanism_precedence_filter_before_ranking():
    r = explain(filters_hit=("risk_gap",), eligible=False, rank=3, dir_side=-1, selection_side=1, timing_blocked=True)
    assert r.primary == RR.OVER_AGGRESSIVE_RISK_FILTER
    assert [e.reason for e in r.reasons][:3] == [RR.OVER_AGGRESSIVE_RISK_FILTER, RR.DIRECTION_DISAGREEMENT, RR.TIMING]


def test_a_picked_candidate_was_not_rejected():
    with pytest.raises(ValueError):
        explain(picked=True)


def test_explain_missed_covers_exactly_the_missed_winners(planted_weeks):
    w = planted_weeks[0]
    rej = M.RejectionAnalyzer().explain_missed(w)
    assert {r.cid for r in rej} == {c.cid for c in w.missed()} and all(c.fwd >= w.thr and not c.picked for c in w.missed())


def test_filter_tradeoff_separates_costly_from_paying_filters():
    winners = [cand(cid=f"w{i}", filters_hit=("risk_vol",), fwd=0.10, eligible=False) for i in range(6)]
    losers = [cand(cid=f"l{i}", filters_hit=("risk_vol",), fwd=-0.08, eligible=False) for i in range(30)]
    w = M.Week("x", "2020-01-06", "2020-01-13", tuple(winners + losers), k=10)
    r = M.filter_tradeoff([w], "risk_vol")
    assert r["winners_foregone"] == 6 and r["losers_avoided"] == 30 and r["verdict"] == "PAYS"
    w2 = M.Week("y", "2020-01-06", "2020-01-13", tuple(winners + losers[:1]), k=10)
    assert M.filter_tradeoff([w2], "risk_vol")["verdict"] == "COSTLY"
    assert M.filter_tradeoff([w], "nope")["verdict"] == "NEVER_FIRED"
    with pytest.raises(FirewallBreach):
        M.filter_tradeoff([w], "risk_vol", now="2020-01-13")


def test_reason_win_rates_flag_a_costly_reason(planted_weeks):
    """Plant: everything rejected for a bad-reliability reason wins 45% (base ~8%): that reason is costly."""
    rng = np.random.default_rng(3)
    weeks = []
    for w in planted_weeks[:40]:
        cs = []
        for c in w.candidates:
            if not c.picked and rng.random() < 0.2:
                win = rng.random() < 0.45
                c = dataclasses.replace(c, reliability=0.05, fwd=0.1 if win else -0.02)
            cs.append(c)
        weeks.append(dataclasses.replace(w, candidates=tuple(cs)))
    rows = {r["reason"]: r for r in M.reason_win_rates(weeks)}
    assert rows["BAD_RELIABILITY"]["verdict"] == "COSTLY" and rows["BAD_RELIABILITY"]["q"] < 0.01
    assert M.reason_win_rates([]) == []


def test_gate_sweep_shows_where_a_gate_discriminates(planted_weeks):
    weeks = []
    rng = np.random.default_rng(4)
    for w in planted_weeks[:30]:
        cs = []
        for c in w.candidates:
            conf = float(rng.random())
            win = rng.random() < (0.5 if conf > 0.8 else 0.05)
            cs.append(dataclasses.replace(c, confidence=conf, fwd=0.1 if win else -0.01))
        weeks.append(dataclasses.replace(w, candidates=tuple(cs)))
    sweep = M.gate_sweep(weeks, "confidence", [0.4, 0.8])
    assert sweep[-1]["win_rate"] > 0.4 > sweep[0]["win_rate"] and len(sweep) == 3
    assert M.gate_sweep([], "confidence", [0.5]) == []


# ------------------------------------------------------------------------------------------------ distinction discovery
def test_cohorts_split_missed_winners_from_false_positives(planted_weeks):
    df = M.cohort_frame(planted_weeks, PARAMS)
    assert set(df["group"]) == {0, 1} and (df["group"] == 1).sum() > 60 and (df["group"] == 0).sum() > 60
    win_ids = {c.cid for w in planted_weeks for c in w.candidates if c.fwd >= w.thr and not c.picked}
    assert set(df.loc[df["group"] == 1, "cid"]) <= win_ids                       # band + matching can only drop, never add
    assert (df["group"] == 1).sum() == (df["group"] == 0).sum()                  # 1:1 rank-matched
    assert M.cohort_balance(df)["verdict"] == "BALANCED"
    unmatched = M.cohort_frame(planted_weeks, dataclasses.replace(PARAMS, match=False, band=2.0))
    assert M.cohort_balance(unmatched)["verdict"] == "SCORE_CONFOUNDED"          # picked losers vs un-picked winners: the trap
    with pytest.raises(FirewallBreach):
        M.cohort_frame(planted_weeks, PARAMS, now="2019-03-01")


def test_planted_distinction_is_found_and_significant(planted_weeks):
    found = M.find_distinctions(M.cohort_frame(planted_weeks, PARAMS), PARAMS, seed=1)
    assert found, "the planted rule must be found"
    top = found[0]
    feats = {c[0] for c in top.conds}
    assert feats & {"f1", "f2"} and top.p_fwer <= 0.05 and top.lift > 1.2 and top.stability > 0.8
    assert top.tpr > top.fpr and top.support_a >= 5
    used = [(c[0], c[1]) for f in found for c in f.conds]
    assert ("f1", ">") in used or ("f2", "<=") in used


def test_null_data_yields_no_significant_distinction(null_weeks):
    found = M.find_distinctions(M.cohort_frame(null_weeks, PARAMS), PARAMS, seed=1)
    assert all(d.p_fwer > 0.05 for d in found), [d.p_fwer for d in found]


def test_search_is_deterministic(planted_weeks):
    df = M.cohort_frame(planted_weeks, PARAMS)
    a = M.find_distinctions(df, PARAMS, seed=7)
    b = M.find_distinctions(df, PARAMS, seed=7)
    assert [(d.did, d.p_fwer, d.stability) for d in a] == [(d.did, d.p_fwer, d.stability) for d in b]


def test_small_cohorts_return_nothing_not_a_guess():
    assert M.find_distinctions(pd.DataFrame(), PARAMS) == []
    tiny = M.cohort_frame(make_weeks(n=1, n_cand=30), PARAMS)
    assert M.find_distinctions(tiny, PARAMS) == []
    assert M.cohort_frame([], PARAMS).empty and M.cohort_balance(pd.DataFrame())["verdict"] == "EMPTY"


def test_learnability_separates_signal_from_noise(planted_weeks, null_weeks):
    assert M.learnability(M.cohort_frame(planted_weeks, PARAMS), PARAMS)["verdict"] == "SEPARABLE"
    assert M.learnability(M.cohort_frame(null_weeks, PARAMS), PARAMS)["verdict"] == "NOT_SEPARABLE_FROM_FEATURES"
    assert M.learnability(pd.DataFrame(columns=["period", "group"]), PARAMS)["verdict"] == "INSUFFICIENT_DATA"


def test_interactions_are_found_only_when_planted():
    rng = np.random.default_rng(8)
    n = 800
    df = pd.DataFrame({"period": rng.integers(0, 40, n).astype(str), "cid": [f"c{i}" for i in range(n)],
                       "era": "e", "kind": "k", **{f: rng.random(n) for f in ("a", "b", "c", "d")}})
    both = (df["a"] > 0.7) & (df["b"] > 0.7)
    df["group"] = (rng.random(n) < np.where(both, 0.8, 0.25)).astype(int)
    found = M.discover_interactions(df, PARAMS, seed=2, top=2)
    assert found and set(found[0]["pair"]) == {"a", "b"} and found[0]["p_fwer"] <= 0.05
    df["group"] = (rng.random(n) < 0.3).astype(int)
    null = M.discover_interactions(df, PARAMS, seed=2, top=1)
    assert not null or null[0]["p_fwer"] > 0.05
    assert M.discover_interactions(pd.DataFrame(), PARAMS) == []


# ------------------------------------------------------------------------------------------------ out of sample
def test_planted_distinction_passes_out_of_sample(planted_weeks):
    train, test = planted_weeks[:40], planted_weeks[41:]
    d = M.find_distinctions(M.cohort_frame(train, PARAMS), PARAMS, seed=1)[0]
    r = M.validate_distinction(d, test, PARAMS, seed=1)
    assert r.verdict == "PASS" and r.lift >= 1.3 and r.mean_gain > 0 and r.excess_vs_random > 0 and r.p_value <= PARAMS.alpha
    assert r.ci[0] < r.mean_gain < r.ci[1]


def test_a_useless_rule_fails_out_of_sample(null_weeks):
    junk = M.Distinction("junk", (("f3", ">", 0.6),), 0.1, 0.5, 0.3, 1.2, 10, 10, 0.5, 0.5, 30, 30)
    r = M.validate_distinction(junk, null_weeks[30:], PARAMS, seed=1)
    assert r.verdict == "FAIL" and r.failed


def test_too_few_test_weeks_is_insufficient_not_pass(planted_weeks):
    d = M.find_distinctions(M.cohort_frame(planted_weeks[:40], PARAMS), PARAMS, seed=1)[0]
    r = M.validate_distinction(d, planted_weeks[41:43], PARAMS)
    assert r.verdict == "INSUFFICIENT" and "test weeks" in r.failed[0]
    assert M.validate_distinction(d, [], PARAMS).verdict == "INSUFFICIENT"


def test_walk_forward_discovers_and_confirms_the_plant(planted_weeks):
    rep = M.walk_forward_distinctions(planted_weeks, PARAMS, seed=1, train_min=30, fold_len=10)
    assert rep.folds == 2 and rep.discovered >= 3 and rep.passed >= 2 and rep.pass_rate > 0.5
    assert set(rep.by_era) <= {"early", "late"}
    txt = M.render_walk_forward(rep)
    assert "IMPLEMENTED - NOT VALIDATED" in txt and "PASS" in txt


def test_walk_forward_refuses_overlapping_horizons(planted_weeks):
    rep = M.walk_forward_distinctions(planted_weeks, dataclasses.replace(PARAMS, embargo=0), seed=1, train_min=30, fold_len=10)
    assert rep.folds == 0 and rep.discovered == 0                # the last training outcome matures on the first test decision


def test_external_validation_hook_is_used(planted_weeks):
    calls = []

    def hook(d, weeks):
        calls.append(len(weeks))
        return M.OOSResult(d.did, len(weeks), 0, 0.0, 0.0, 1.0, 0.0, (0.0, 0.0), 1.0, 0.0, "FAIL", ("external",))
    rep = M.walk_forward_distinctions(planted_weeks, PARAMS, seed=1, train_min=30, fold_len=10, hook=hook)
    assert calls and rep.passed == 0 and all(r.failed == ("external",) for _, r in rep.results)


def test_walk_forward_is_fail_closed_on_now(planted_weeks):
    with pytest.raises(FirewallBreach):
        M.walk_forward_distinctions(planted_weeks, PARAMS, now="2019-06-01")


def test_null_control_false_discovery_rate_is_low(planted_weeks):
    real = M.walk_forward_distinctions(planted_weeks, PARAMS, seed=1, train_min=30, fold_len=10)
    null = M.null_control(planted_weeks, PARAMS, reps=3, seed=5, train_min=30, fold_len=10)
    assert null["tested"] >= 0 and (null["false_discovery_rate"] != null["false_discovery_rate"] or null["false_discovery_rate"] < real.pass_rate)
    assert null["passed"] <= 1                                   # shuffled labels: essentially nothing may pass
    assert "false discovery rate" in M.render_walk_forward(real, null)


def test_shuffle_labels_keeps_everything_but_outcomes(planted_weeks):
    sh = M.shuffle_labels(planted_weeks[:3], np.random.default_rng(0))
    for a, b in zip(planted_weeks[:3], sh):
        assert [c.features for c in a.candidates] == [c.features for c in b.candidates]
        assert [c.picked for c in a.candidates] == [c.picked for c in b.candidates]
        assert sorted(c.fwd for c in a.candidates) == pytest.approx(sorted(c.fwd for c in b.candidates))
    assert any(x.fwd != y.fwd for x, y in zip(planted_weeks[0].candidates, sh[0].candidates))


def test_distinction_by_era_and_book(planted_weeks):
    d = M.find_distinctions(M.cohort_frame(planted_weeks[:40], PARAMS), PARAMS, seed=1)[0]
    by = M.distinction_by_era(d, planted_weeks)
    assert set(by) == {"early", "late"} and all(v["lift"] > 1.2 for v in by.values())
    book = M.DistinctionBook()
    for s in range(3):
        book.add_fold(M.find_distinctions(M.cohort_frame(planted_weeks[:40 + 5 * s], PARAMS), PARAMS, seed=s))
    st = book.stable(0.6)
    assert st and st[0]["found_in"] >= 2
    assert M.DistinctionBook().stable() == []


# ------------------------------------------------------------------------------------------------ hypotheses and ledger
def test_a_distinction_becomes_a_hypothesis_never_a_change(planted_weeks):
    d = M.find_distinctions(M.cohort_frame(planted_weeks[:40], PARAMS), PARAMS, seed=1)[0]
    h0 = M.distinction_to_hypothesis(d)
    h1 = M.distinction_to_hypothesis(d, M.validate_distinction(d, planted_weeks[41:], PARAMS, seed=1))
    for h in (h0, h1):
        assert h.epistemic == Epistemic.HYPOTHESIS and not h.production_effect and not h.validate()
        assert h.decision_effect == DecisionEffect.RANKING and h.subsystem == Subsystem.SELECTION
    assert h0.hid == h1.hid and "not yet tested" in h0.statement and "PASS" in h1.statement


def test_ledger_reports_reasons_by_type_and_era(planted_weeks):
    led = M.MissedLearningLedger()
    for w in planted_weeks[:20]:
        led.add_week(w)
    t = led.reason_table()
    assert t["n"].sum() == sum(len(w.missed()) for w in planted_weeks[:20])
    assert set(led.reason_table(by="era")["era"]) == {"early"} and 0.0 <= led.unknown_share() <= 1.0
    assert led.base.catch_rate() >= 0.0 and len(led.base.rows) == 20
    for w in planted_weeks[20:]:
        led.add_week(w)
    assert set(led.reason_stability()) <= {r.value for r in RR} and "IMPLEMENTED - NOT VALIDATED" in led.report()
    empty = M.MissedLearningLedger()
    assert empty.reason_table().empty and empty.unknown_share() != empty.unknown_share() and empty.reason_stability() == {}


def test_week_from_base_hides_names_and_uses_ranks():
    rng = np.random.default_rng(9)
    tick = [f"TK{i}" for i in range(40)]
    p0 = pd.DataFrame({"e_ear": rng.random(40), "vol20": rng.random(40) * 0.05, "log_dv": rng.random(40) * 5, "mu_raw": rng.normal(size=40)}, index=tick)
    fwd = pd.Series(rng.normal(0, 0.05, 40), index=tick)
    score = p0["mu_raw"]
    w = M.week_from_base("2020-01-06", p0, fwd, picked=tick[:5], score=score, eligible=pd.Series(True, index=tick), k=5, era="x")
    assert len(w.candidates) == 40 and not w.validate()
    assert not any("TK" in c.cid for c in w.candidates)
    assert all(0.0 <= v <= 1.0 for c in w.candidates for v in c.features.values())
    assert sum(c.picked for c in w.candidates) == 5 and w.resolved_at == "2020-01-13"
    w2 = M.week_from_base("2020-01-06", p0, fwd, picked=tick[:5], eligible=pd.Series([i % 2 == 0 for i in range(40)], index=tick))
    assert any(c.filters_hit == ("ineligible",) for c in w2.candidates)


def test_invalid_weeks_and_candidates_are_reported():
    bad = M.Week("w", "2020-01-13", "2020-01-06", (M.Candidate("a", {"f": 1.0}), M.Candidate("a", {"ticker": 1.0})), k=5)
    errs = bad.validate()
    assert any("resolved_at" in e for e in errs) and any("duplicate cid" in e for e in errs) and any("identity key" in e for e in errs)
    assert M.Week("e", "2020-01-06", "2020-01-13", ()).base_rate() != M.Week("e", "2020-01-06", "2020-01-13", ()).base_rate()


# ------------------------------------------------------------------------------------------------ later additions
def test_synthetic_weeks_are_deterministic_and_planted_only_when_asked():
    a, b = M.synthetic_weeks(n=10, seed=3), M.synthetic_weeks(n=10, seed=3)
    assert [c.fwd for w in a for c in w.candidates] == [c.fwd for w in b for c in w.candidates]
    def region_rate(ws):
        cs = [c for w in ws for c in w.candidates if c.features["f1"] > 0.7 and c.features["f2"] < 0.35]
        return float(np.mean([c.fwd >= WINNER for c in cs]))
    WINNER = M.WINNER
    assert region_rate(M.synthetic_weeks(n=40, seed=4, planted=True)) > 0.3 > region_rate(M.synthetic_weeks(n=40, seed=4, planted=False))


def test_distinction_selfcheck_finds_the_plant_and_nothing_on_null():
    r = M.distinction_selfcheck(seed=0)
    assert r["ok"] and r["planted"]["oos_pass"] >= 1 and r["null"]["significant"] == 0


def test_rule_confusion_and_backtest_and_combination(planted_weeks):
    d = M.find_distinctions(M.cohort_frame(planted_weeks[:40], PARAMS), PARAMS, seed=1)[0]
    rc = M.rule_confusion(d, planted_weeks[41:])
    assert rc["tp"] + rc["fp"] + rc["fn"] + rc["tn"] == sum(1 for w in planted_weeks[41:] for c in w.candidates if not c.picked)
    assert rc["lift"] > 1.2 and rc["precision"] > rc["base_rate"]
    bt = M.promote_backtest(d, planted_weeks[41:], seed=1)
    summ = M.backtest_summary(bt)
    assert summ["weeks"] == len(bt) > 5 and summ["mean_gain"] > summ["mean_random"] and summ["max_drawdown"] <= 0.0 and "late" in summ["by_era"]
    assert M.backtest_summary(M.promote_backtest(d, [], seed=1))["verdict"] == "NO_MATCHES"
    with pytest.raises(FirewallBreach):
        M.promote_backtest(d, planted_weeks[41:], now="2019-01-01")
    found = M.find_distinctions(M.cohort_frame(planted_weeks[:40], PARAMS), PARAMS, seed=1)
    inc = M.combine_rules(found, planted_weeks[41:])
    assert inc[0]["new_winners"] >= inc[-1]["new_winners"] and all(0 <= r["precision"] <= 1 for r in inc if r["precision"] == r["precision"])


def test_discovery_sensitivity_shows_the_finding_survives_settings(planted_weeks):
    rows = M.discovery_sensitivity(planted_weeks, PARAMS, "rank_caliper", [4, 6, 10], seed=1)
    assert len(rows) == 3 and all(r["found"] >= 1 for r in rows)
    assert all(set(r["top_features"]) & {"f1", "f2"} for r in rows)
    with pytest.raises(ValueError):
        M.discovery_sensitivity(planted_weeks, PARAMS, "nope", [1])


def test_reason_dynamics_and_tables(planted_weeks):
    tab = M.rejection_table(planted_weeks[:20])
    assert len(tab) == sum(len(w.missed()) for w in planted_weeks[:20]) and set(tab["reason"]) <= {r.value for r in RR}
    assert not M.reason_by_type(planted_weeks[:20]).empty and M.reason_by_type([]).empty
    curve = M.recovery_curve(planted_weeks[:20])
    assert curve[-1]["cumulative_share"] == pytest.approx(1.0) and curve[0]["return_missed"] >= curve[-1]["return_missed"]
    top = M.top_missed(planted_weeks[:20], n=3)
    assert len(top) == 3 and top[0]["fwd"] >= top[-1]["fwd"] and M.top_missed([]) == []
    rng = np.random.default_rng(2)
    early = [dataclasses.replace(w, candidates=tuple(dataclasses.replace(c, timing_blocked=False) for c in w.candidates)) for w in planted_weeks[:25]]
    late = [dataclasses.replace(w, candidates=tuple(dataclasses.replace(c, timing_blocked=(not c.picked and rng.random() < 0.7)) for c in w.candidates))
            for w in planted_weeks[25:]]
    shift = {r["reason"]: r for r in M.reason_shift(early, late, min_n=10)}
    assert shift["TIMING"]["shift"] > 0.3 and shift["TIMING"]["p"] < 0.001
    assert M.reason_shift([], []) == []


def test_explain_rejection_and_precedence_property():
    txt = M.explain_rejection(explain(timing_blocked=True))
    assert "TIMING" in txt and "entry-timing gate" in txt
    assert "does not support a reason" in M.explain_rejection(explain(rank=None, score=None))
    assert M.precedence_is_consistent() is True


def test_weeks_from_frame_roundtrip_and_cohort_description(planted_weeks):
    rows = []
    for w in planted_weeks[:12]:
        for c in w.candidates:
            rows.append({"period": w.label, "decided_at": w.decided_at, "resolved_at": w.resolved_at, "cid": c.cid, "fwd": c.fwd, "picked": c.picked,
                         "score": c.score, "rank": c.rank, "era": w.era, "kind": c.kind, "confidence": c.confidence,
                         "filters_hit": ",".join(c.filters_hit), **c.features})
    ws = M.weeks_from_frame(pd.DataFrame(rows), k=8)
    assert len(ws) == 12 and ws[0].candidates[0].features == planted_weeks[0].candidates[0].features and ws[0].era == "early"
    assert [w.label for w in ws] == [w.label for w in planted_weeks[:12]]
    with pytest.raises(ValueError):
        M.weeks_from_frame(pd.DataFrame({"period": [1]}))
    desc = M.describe_cohorts(M.cohort_frame(planted_weeks, PARAMS))
    assert desc["n_a"] == desc["n_b"] > 0 and set(desc["by_era"]) == {"early", "late"} and M.describe_cohorts(pd.DataFrame())["n_a"] == 0


def test_params_validation_and_summaries(planted_weeks):
    assert M.validate_params(PARAMS) == []
    assert len(M.validate_params(dataclasses.replace(PARAMS, alpha=0.9, n_perm=5, band=0.5, embargo=0))) == 4
    s = M.week_summary(planted_weeks[0])
    assert s["candidates"] == 60 and s["picked"] == 8 and s["missed"] + round(s["catch_rate"] * s["winners"]) == s["winners"]
    wk = M.winners_by_kind(planted_weeks)
    assert wk["other"]["winners"] > 0 and 0.0 <= wk["other"]["catch_rate"] <= 1.0 and M.winners_by_kind([]) == {}
    with pytest.raises(ValueError):
        M.RejectionAnalyzer().explain_missed  # attribute access is fine; explain on a picked candidate is not
        M.RejectionAnalyzer().explain(cand(picked=True), WEEK)
