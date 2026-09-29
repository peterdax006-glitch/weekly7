"""SELF_LEARNING_CONTRACT sections 20 and 21 (credit assignment, redundancy intelligence): every mechanism is checked against a
planted truth it must recover, a planted defect it must refuse to credit, and the empty case. Synthetic data only."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning import credit as C
from engine.learning import redundancy as R
from engine.learning.core import Edge, FailureCause, FirewallBreach, Subsystem

NOW = "2035-01-01"
CFG = C.CreditConfig(n_boot=120, n_perm=150, seed=3)
SUM4 = C.WeightedSumCombiner({c: 1.0 for c in C.SIGNAL_COMPONENTS})


def frame_of(ledger, now=NOW):
    ds, _ = ledger.mature(now)
    return C.DecisionFrame.build(ds, ledger.components(), CFG.neutral)


@pytest.fixture(scope="module")
def single_report():
    led = C.simulate_decisions(320, 1, lambda r, d: 0.4 * d.pattern, noise=0.5)
    return led, C.CreditEngine(SUM4, CFG).assess(led, NOW)


# ------------------------------------------------------------------------------------------ exactness of the attribution
def test_additive_decision_gives_exact_shapley_and_efficiency():
    led = C.simulate_decisions(120, 2, lambda r, d: 0.3 * d.pattern + 0.2 * d.analog, noise=0.3)
    w = {"pattern": 1.0, "analog": 0.5, "memory": 0.25, "direction": 2.0}
    fr = frame_of(led)
    co = C.Coalitions(fr, C.WeightedSumCombiner(w), C.Utility.PAYOFF, CFG.neutral)
    phi, D = C.shapley_values(co, CFG, np.random.default_rng(0))
    for k, c in enumerate(fr.scores.columns):
        expect = w.get(c, 0.0) * fr.scores[c].to_numpy() * fr.outcome         # additive => credit is exactly w*s*y
        assert np.allclose(phi[:, k], expect)
    assert np.allclose(phi.sum(axis=1), co.v(co.full_mask) - co.v(0))          # efficiency: nothing created or lost


def test_dividend_route_equals_textbook_shapley():
    led = C.simulate_decisions(80, 4, lambda r, d: 0.4 * d.pattern * d.timing * d.risk, noise=0.2)
    co = C.Coalitions(frame_of(led), C.StructuredCombiner({c: 1.0 for c in C.SIGNAL_COMPONENTS}), C.Utility.PAYOFF, CFG.neutral)
    via_div = C.shapley_from_dividends(C.harsanyi_dividends(co.all_values()), co.n_comp)
    assert np.allclose(via_div, C.shapley_direct(co))


def test_sampled_shapley_keeps_efficiency_exactly():
    led = C.simulate_decisions(60, 5, lambda r, d: 0.4 * d.pattern, noise=0.2)
    co = C.Coalitions(frame_of(led), C.StructuredCombiner({c: 1.0 for c in C.SIGNAL_COMPONENTS}), C.Utility.PAYOFF, CFG.neutral)
    phi = C.shapley_sampled(co, 30, np.random.default_rng(1))
    assert np.allclose(phi.sum(axis=1), co.v(co.full_mask) - co.v(0))


# ------------------------------------------------------------------------------------------ planted truth and refusals
def test_only_the_planted_component_earns_credit(single_report):
    _, rep = single_report
    assert rep.earners() == ["pattern"]
    assert rep.component("pattern").share == pytest.approx(1.0)
    assert all(rep.component(c).share == 0.0 for c in ("analog", "memory", "direction", "timing", "risk"))
    assert rep.component("pattern").ablate_verdict == "earns_keep"          # the engine.ablation cross-check agrees
    assert C.validate_report(rep) == []


def test_noise_only_world_grants_no_credit_at_all():
    led = C.simulate_decisions(320, 7, lambda r, d: 0.0 * d.pattern, noise=0.5)
    rep = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    assert rep.earners() == []
    assert rep.unattributed_fraction == 1.0


def test_calibration_harness_recovers_planted_scenarios():
    res = C.run_planted_calibration(seed=2, n=320, cfg=CFG)
    assert res["single"]["missed"] == [] and res["single"]["false_credit"] == []
    assert res["noise_only"]["false_credit"] == []
    assert res["two_independent"]["recall"] == 1.0
    assert all(v["errors"] == [] for v in res.values())


def test_duplicate_components_split_credit_and_loo_masks_them():
    # decision = pattern if it speaks, else direction: a duplicate covers for its twin, so leave-one-out says "free to drop" BOTH
    led = C.simulate_decisions(200, 9, lambda r, d: 0.4 * d.pattern, noise=0.2)
    fr = frame_of(led)
    fr.scores["direction"] = fr.scores["pattern"]
    comb = C.FunctionCombiner(lambda S: np.where(S["pattern"] != 0, S["pattern"], S["direction"]), "fallback")
    co = C.Coalitions(fr, comb, C.Utility.PAYOFF, CFG.neutral)
    phi = C.shapley_values(co, CFG, np.random.default_rng(0))[0]
    loo, solo = C.loo_and_solo(co)
    names = list(fr.scores.columns)
    ip, idr = names.index("pattern"), names.index("direction")
    assert abs(loo[:, ip].mean()) < 1e-12 and abs(loo[:, idr].mean()) < 1e-12
    assert np.allclose(phi[:, ip], phi[:, idr]) and phi[:, ip].mean() > 0.05      # Shapley shares what LOO hides
    assert set(C.redundancy_masking(loo, phi, solo)) == {ip, idr}


def test_timing_gate_creates_a_reliable_synergy_but_an_additive_model_does_not():
    led = C.simulate_decisions(400, 11, lambda r, d: 0.6 * d.pattern * (2 * d.timing - 1), noise=0.3)
    comb = C.StructuredCombiner({c: 1.0 for c in C.SIGNAL_COMPONENTS})
    rep = C.CreditEngine(comb, CFG).assess(led, NOW)
    pt = [i for i in rep.interactions if {i.a, i.b} == {"pattern", "timing"}][0]
    assert pt.kind == "SYNERGY" and pt.verdict == "RELIABLE" and pt.lo > 0
    assert ("pattern", Edge.COMPLEMENTS.value) in [(a, e) for a, e, b, w in C.credit_edges(rep)] or \
           ("timing", Edge.COMPLEMENTS.value) in [(a, e) for a, e, b, w in C.credit_edges(rep)]
    add = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    assert all(i.kind == "NONE" for i in add.interactions)


def test_a_component_present_but_useless_is_not_credited_despite_a_lucky_sign():
    led = C.simulate_decisions(300, 13, lambda r, d: 0.4 * d.pattern, noise=0.6)
    rep = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    for c in ("analog", "memory", "direction"):
        assert rep.component(c).verdict is not C.CreditVerdict.EARNS_CREDIT


def test_sign_flipping_component_is_marked_unstable_not_credited():
    n = 400
    led = C.simulate_decisions(n, 15, lambda r, d: np.where(d.index < 0.55 * n, 0.8, -0.4) * d.pattern, noise=0.3)
    rep = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    assert rep.component("pattern").verdict is not C.CreditVerdict.EARNS_CREDIT
    assert rep.component("pattern").share == 0.0


def test_blanket_credit_detector():
    assert C.blanket_credit([1.0, 1.05, 0.95, 0.0], 0.25)
    assert not C.blanket_credit([1.0, 0.2, 0.05, 0.0], 0.25)
    assert not C.blanket_credit([1.0, 1.0], 0.25)


def test_context_dependence_is_detected_and_credit_splits_by_context():
    n = 420
    led = C.simulate_decisions(n, 17, lambda r, d: 0.7 * d.pattern * (d.index < n / 2), noise=0.3, regime_flip_at=0.5)
    rep = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    assert rep.context_dependence["regime"]["pattern"] < 0.05
    cc = rep.context_credit["regime"]
    assert cc["early"]["pattern"] > 5 * abs(cc["late"]["pattern"])


# ------------------------------------------------------------------------------------------ firewall, empty, degenerate
def test_pending_outcomes_are_never_used_and_strict_mode_raises():
    led = C.simulate_decisions(120, 19, lambda r, d: 0.4 * d.pattern)
    cut = "2020-02-10"
    used, pending = led.mature(cut)
    assert pending > 0 and all(d.matured < cut for d in used)
    with pytest.raises(FirewallBreach):
        led.mature(cut, strict=True)
    rep = C.CreditEngine(SUM4, CFG).assess(led, cut)
    assert rep.n_pending == pending and rep.n_decisions == len(used)
    d0 = next(iter(led))
    with pytest.raises(FirewallBreach):
        C.CreditEngine(SUM4, CFG).explain(led, d0.decision_id, d0.asof)          # outcome not yet known at asof


def test_empty_ledger_and_too_few_decisions():
    rep = C.CreditEngine(SUM4, CFG).assess(C.DecisionLedger(), NOW)
    assert rep.n_decisions == 0 and rep.components == () and rep.unattributed_fraction == 1.0
    small = C.simulate_decisions(12, 21, lambda r, d: 0.9 * d.pattern, noise=0.01)
    rep = C.CreditEngine(SUM4, CFG).assess(small, NOW)
    assert rep.component("pattern").verdict is C.CreditVerdict.INSUFFICIENT and rep.earners() == []


def test_single_component_ledger_gets_all_the_effect():
    led = C.DecisionLedger()
    for i in range(60):
        d = pd.Timestamp("2021-01-04") + pd.offsets.BDay(i // 2)
        led.add(C.Decision(f"s{i}", str(d.date()), str((d + pd.offsets.BDay(3)).date()), {"pattern": 1.0}, 0.1 + 0.01 * (i % 3)))
    rep = C.CreditEngine(C.WeightedSumCombiner({"pattern": 1.0}), CFG).assess(led, NOW)
    assert rep.efficiency_error < 1e-12 and rep.component("pattern").mean_credit == pytest.approx(rep.full_value)


def test_invalid_records_and_config_are_rejected():
    with pytest.raises(ValueError):
        C.DecisionLedger([C.Decision("x", "2021-01-05", "2021-01-04", {"pattern": 1.0}, 0.1)])       # matured before asof
    with pytest.raises(ValueError):
        C.DecisionLedger([C.Decision("x", "2021-01-04", "2021-01-08", {"pattern": float("nan")}, 0.1)])
    led = C.DecisionLedger([C.Decision("a", "2021-01-04", "2021-01-08", {"pattern": 1.0}, 0.1)])
    with pytest.raises(ValueError):
        led.add(C.Decision("a", "2021-01-04", "2021-01-08", {"pattern": 1.0}, 0.1))                   # duplicate id
    with pytest.raises(ValueError):
        C.CreditEngine(SUM4, C.CreditConfig(n_perm=5))                                               # p-value floor >= alpha


def test_determinism_and_seed_sensitivity():
    led = C.simulate_decisions(220, 23, lambda r, d: 0.4 * d.pattern)
    a = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    b = C.CreditEngine(SUM4, CFG).assess(led, NOW)
    assert a.to_json() == b.to_json()
    c = C.CreditEngine(SUM4, dataclasses.replace(CFG, seed=CFG.seed + 1)).assess(led, NOW)
    assert c.component("pattern").lo != a.component("pattern").lo


# ------------------------------------------------------------------------------------------ postmortem and downstream views
def test_postmortem_blames_the_component_that_caused_the_loss():
    led = C.DecisionLedger()
    comb = C.WeightedSumCombiner({"pattern": 1.0, "direction": 1.0, "risk": 1.0})
    led.add(C.Decision("loss_pattern", "2021-03-01", "2021-03-08", {"pattern": 1.0, "direction": 0.05, "risk": 0.0}, -0.1))
    led.add(C.Decision("loss_risk", "2021-03-02", "2021-03-09", {"pattern": 0.02, "direction": 0.0, "risk": 2.0}, -0.1))
    led.add(C.Decision("win", "2021-03-03", "2021-03-10", {"pattern": 1.0, "direction": 0.0, "risk": 0.0}, 0.1))
    eng = C.CreditEngine(comb, CFG)
    p1, p2, w = (eng.explain(led, i, NOW) for i in ("loss_pattern", "loss_risk", "win"))
    assert (p1.dominant, p1.subsystem, p1.cause) == ("pattern", Subsystem.SELECTION, FailureCause.SELECTION_ERROR)
    assert (p2.dominant, p2.subsystem, p2.cause) == ("risk", Subsystem.RISK, FailureCause.RISK_ERROR)
    assert w.cause is None
    summary = C.failure_attribution_summary(eng.postmortems(led, NOW))
    assert set(summary["subsystem"]) == {"SELECTION", "RISK"} and summary["share"].sum() == pytest.approx(1.0)


def test_knowledge_credit_flows_to_the_ids_that_produced_the_signal():
    base = C.simulate_decisions(320, 29, lambda r, d: 0.5 * d.pattern, noise=0.4)
    led = C.DecisionLedger()
    for d in base:
        led.add(dataclasses.replace(d, knowledge={"pattern": {"k_good": 1.0}, "memory": {"k_noise": 1.0}}))
    out = {(k.knowledge_id, k.component): k for k in C.CreditEngine(SUM4, CFG).knowledge_credit(led, NOW)}
    assert out[("k_good", "pattern")].verdict is C.CreditVerdict.EARNS_CREDIT
    assert out[("k_noise", "memory")].verdict is not C.CreditVerdict.EARNS_CREDIT


def test_component_ablation_bridges_to_engine_ablation(single_report):
    led, _ = single_report
    res = C.component_ablation(frame_of(led), SUM4, CFG)
    assert res["keep"] == ["pattern"]


def test_baseline_and_utility_sensitivity_agree_on_a_real_signal(single_report):
    led, _ = single_report
    fr = frame_of(led)
    bs = C.baseline_sensitivity(fr, SUM4, CFG).set_index("component")
    assert bs.loc["pattern", "sign_agrees"] and bs.loc["pattern", "marginal"] > 0
    us = C.utility_sensitivity(fr, SUM4, CFG).set_index("component")
    assert us.loc["pattern", "PAYOFF_pos"] and us.loc["pattern", "HIT_pos"] and not us.loc["analog", "HIT_pos"]
    assert not us["robust_positive"].any() and us.loc["analog", "NEG_SQERR_neg"]     # scale-mismatched point forecast punishes noise


def test_credit_trend_flags_a_decaying_component_and_refuses_short_series():
    idx = pd.period_range("2020Q1", periods=8, freq="Q")
    ot = pd.DataFrame({"pattern": np.linspace(0.4, 0.0, 8), "analog": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
                       "n": 50}, index=idx)
    tr = C.credit_trend(ot)
    assert tr["pattern"]["tag"] == "DECAYING" and tr["analog"]["tag"] == "STEADY"
    assert C.credit_trend(ot.iloc[:3])["pattern"]["tag"] == "INSUFFICIENT"


def test_power_table_marks_underpowered_null_verdicts(single_report):
    led, rep = single_report
    fr = frame_of(led)
    phi = C.CreditEngine(SUM4, CFG).raw_credit(fr)["phi"]
    pt = C.power_table(rep, fr, phi).set_index("component")
    assert (pt["mde"] >= 0).all() and pt.loc["pattern", "mde"] > 0
    assert not bool(pt.loc["pattern", "underpowered"])


def test_ledger_hash_chain_detects_tampering(tmp_path, single_report):
    _, rep = single_report
    L = C.CreditLedger(tmp_path / "credit.jsonl")
    L.append(rep)
    L.append(dataclasses.replace(rep, now="2035-02-01"))
    assert L.verify() == []
    with pytest.raises(FirewallBreach):
        L.append(dataclasses.replace(rep, now="2034-01-01"))
    txt = (tmp_path / "credit.jsonl").read_text(encoding="utf-8").replace('"n_decisions":320', '"n_decisions":321', 1)
    (tmp_path / "credit.jsonl").write_text(txt, encoding="utf-8")
    assert any("edited" in e for e in L.verify())


def test_validate_report_catches_a_broken_attribution(single_report):
    _, rep = single_report
    assert C.validate_report(dataclasses.replace(rep, efficiency_error=0.5)) != []


# ============================================================================================================================
# redundancy (section 21)
# ============================================================================================================================
@pytest.fixture(scope="module")
def planted():
    return R.run_planted_calibration(0)


def test_all_planted_pair_classes_are_recovered(planted):
    assert planted["n_correct"] == planted["n_pairs"], planted["classes"]
    assert planted["errors"] == []


def test_same_prediction_different_protection_is_not_called_a_duplicate(planted):
    """The planted defect: a correlation-only rule would retire `guard`. It predicts like `base` (rho ~0.9) but never takes the
    crisis losses `base` takes. It must be kept."""
    p = planted["report"].pair("base", "guard")
    assert p.value(R.RedundancyKind.PREDICTIVE) > 0.8
    assert p.value(R.RedundancyKind.RISK) < 0.4
    assert p.klass is R.PairClass.DIFFERENT_PROTECTION and p.recommendation.startswith("KEEP BOTH")
    assert planted["report"].retire_candidates and all(d not in ("guard",) for d, _, _ in planted["report"].retire_candidates)


def test_edges_for_the_knowledge_graph(planted):
    rep = planted["report"]
    kinds = {(frozenset((s, d)), e) for s, e, d, w, _ in rep.edges()}
    assert (frozenset(("base", "dup")), Edge.REDUNDANT_WITH) in kinds
    assert (frozenset(("base", "hedge")), Edge.COMPLEMENTS) in kinds
    assert (frozenset(("base", "guard")), Edge.COMPLEMENTS) in kinds
    assert (frozenset(("base", "noise")), Edge.REDUNDANT_WITH) not in kinds


def test_duplicate_cluster_vif_and_forward_selection(planted):
    rep = planted["report"]
    assert rep.clusters == (("base", "dup"),)
    assert rep.vif["base"] > 50 and rep.vif["noise"] < 1.5
    panel, _ = R.planted_panel(0)
    chosen = [c for c, _, _ in R.forward_select(panel, ["base", "dup", "guard", "hedge", "noise"])]
    assert not ({"base", "dup"} <= set(chosen)) and "noise" not in chosen


def test_conditional_view_shows_the_regime_difference():
    panel, _ = R.planted_panel(0)
    t = R.conditional_predictive(panel, "base", "guard").set_index("context")
    assert t.loc["calm", "rho"] > 0.95 and abs(t.loc["crisis", "rho"]) < 0.2


def test_retirement_safety_is_asymmetric():
    panel, _ = R.planted_panel(0)
    assert R.retirement_safety(panel, keep="guard", drop="base")["safe"] is True
    assert R.retirement_safety(panel, keep="base", drop="guard")["safe"] is False


def test_untested_kinds_are_none_and_block_a_duplicate_verdict():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(300)
    panel = R.ObservationPanel.build(pd.DataFrame({"a": a, "b": a + 0.01 * rng.standard_normal(300)}), pd.Series(a * 0.1 + rng.standard_normal(300)),
                                     pd.bdate_range("2020-01-01", periods=300), None)             # one context only, nobody ever unavailable
    p = R.RedundancyAnalyzer().analyze_pair(panel, {}, "a", "b")
    assert p.value(R.RedundancyKind.CONTEXT) is None and p.value(R.RedundancyKind.MECHANISTIC) is None
    assert p.value(R.RedundancyKind.OPERATIONAL) is None
    assert p.klass is R.PairClass.UNRESOLVED and "context" in " ".join(p.flags).lower()
    tiny = R.ObservationPanel.build(pd.DataFrame({"a": a[:20], "b": a[:20]}), pd.Series(a[:20]), pd.bdate_range("2020-01-01", periods=20), None)
    assert R.RedundancyAnalyzer().analyze_pair(tiny, {}, "a", "b").klass is R.PairClass.UNRESOLVED


def test_mechanism_missing_is_unknown_not_different():
    panel, _ = R.planted_panel(1)
    an = R.RedundancyAnalyzer()
    p = an.analyze_pair(panel, {"base": R.ItemProfile("base"), "dup": R.ItemProfile("dup")}, "base", "dup")
    assert p.value(R.RedundancyKind.MECHANISTIC) is None and "UNKNOWN" in p.scores[R.RedundancyKind.MECHANISTIC].note
    q = an.analyze_pair(panel, {"base": R.ItemProfile("base", ("mom",)), "noise": R.ItemProfile("noise", ("mom",))}, "base", "noise")
    assert any("same mechanism but does not predict alike" in f for f in q.flags)


def test_era_instability_is_flagged():
    rng = np.random.default_rng(3)
    n = 600
    a = rng.standard_normal(n)
    b = a.copy()
    b[n // 2:] = rng.standard_normal(n - n // 2)                       # duplicate in the first half only
    panel = R.ObservationPanel.build(pd.DataFrame({"a": a, "b": b}), pd.Series(0.2 * a + rng.standard_normal(n)),
                                     pd.bdate_range("2019-01-01", periods=n), None)
    p = R.RedundancyAnalyzer().analyze_pair(panel, {}, "a", "b")
    assert p.era_stable is False and any("early and late" in f for f in p.flags)
    drift = R.redundancy_drift(R.redundancy_over_time(panel, "a", "b"))
    assert drift["unstable"] is True


def test_redundancy_firewall_and_degenerate_inputs():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=100)
    sig = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)})
    p = R.ObservationPanel.build(sig, pd.Series(rng.standard_normal(100)), dates, None, horizon_days=5, now="2020-03-02")
    assert p.dropped_future > 0 and (p.dates + pd.Timedelta(days=5) < pd.Timestamp("2020-03-02")).all()
    with pytest.raises(FirewallBreach):
        R.ObservationPanel.build(sig, pd.Series(rng.standard_normal(100)), dates, None, now="2020-03-02", strict=True)
    full = R.ObservationPanel.build(sig, pd.Series(rng.standard_normal(100)), dates, None)
    with pytest.raises(FirewallBreach):
        R.RedundancyAnalyzer().analyze(full, {}, "2020-02-01")          # newest outcome not yet matured at `now`
    one = R.ObservationPanel.build(sig[["a"]], pd.Series(rng.standard_normal(100)), dates, None)
    assert R.RedundancyAnalyzer().analyze(one, {}, "2030-01-01").pairs == ()
    assert R.RedundancyConfig(low=0.9, high=0.5).validate() != []
    with pytest.raises(ValueError):
        R.RedundancyAnalyzer(R.RedundancyConfig(low=0.9, high=0.5))
    with pytest.raises(ValueError):
        R.ObservationPanel.build(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1.0]), pd.bdate_range("2020-01-01", periods=2), None)


def test_redundancy_is_deterministic():
    a = R.run_planted_calibration(3)["report"]
    b = R.run_planted_calibration(3)["report"]
    assert [(p.a, p.b, p.klass, p.value(R.RedundancyKind.PREDICTIVE)) for p in a.pairs] == \
           [(p.a, p.b, p.klass, p.value(R.RedundancyKind.PREDICTIVE)) for p in b.pairs]


def test_coverage_preserving_reduction_refuses_to_archive_the_only_effective_item():
    panel, _ = R.planted_panel(0)
    rep = R.RedundancyAnalyzer().analyze(panel, {}, "2035-01-01", items=["base", "dup"])
    assert len(rep.retire_candidates) == 1 and not rep.retire_candidates[0][2].startswith("NOT")
    # planted defect: `weak` is a worse predictor than `base` overall (so it would be the one dropped) but it alone works in
    # context B. Declared a duplicate by hand, the reduction must refuse to archive it.
    rng = np.random.default_rng(8)
    n_a, n_b = 900, 300
    s_a, t_b = rng.standard_normal(n_a), rng.standard_normal(n_b)
    y = np.concatenate([0.5 * s_a, 0.5 * t_b]) + 0.5 * rng.standard_normal(n_a + n_b)
    base = np.concatenate([s_a, rng.standard_normal(n_b)])
    weak = np.concatenate([s_a + 2 * rng.standard_normal(n_a), t_b])
    ctx = ["A"] * n_a + ["B"] * n_b
    panel2 = R.ObservationPanel.build(pd.DataFrame({"base": base, "weak": weak}), pd.Series(y),
                                      pd.bdate_range("2019-01-01", periods=n_a + n_b), ctx)
    m = R.Measurer(panel2, {}, R.RedundancyConfig())
    assert m.effective_contexts("weak") - m.effective_contexts("base") == {"B"}
    fake = R.PairRedundancy("base", "weak", {k: R.KindScore(k, 0.9, 100) for k in R.RedundancyKind}, 100,
                            R.PairClass.TRUE_DUPLICATE, None, (), "", True)
    out = R.RedundancyAnalyzer().coverage_preserving_reduction(panel2, {}, [fake])
    assert out[0][0] == "weak" and out[0][2].startswith("NOT archived")


# ============================================================================================================================
# credit: ordering, single-decision audit, transfer, update proposals, hand-off
# ============================================================================================================================
def test_sequential_credit_is_order_dependent_but_sums_to_the_same_total():
    led = C.simulate_decisions(150, 31, lambda r, d: 0.5 * d.pattern * (2 * d.timing - 1), noise=0.2)
    fr = frame_of(led)
    comb = C.StructuredCombiner({c: 1.0 for c in C.SIGNAL_COMPONENTS})
    a = C.sequential_credit(fr, comb, CFG)
    b = C.sequential_credit(fr, comb, CFG, ["timing", "pattern", "analog", "memory", "direction", "risk"])
    co = C.Coalitions(fr, comb, C.Utility.PAYOFF, CFG.neutral)
    for x in (a, b):
        assert np.allclose(x.sum(axis=1), co.v(co.full_mask) - co.v(0))
    names = list(fr.scores.columns)
    assert not np.allclose(a[:, names.index("timing")], b[:, names.index("timing")])        # timing's credit depends on its position
    sens = C.order_sensitivity(fr, comb, CFG, n_orders=12).set_index("component")
    assert sens.loc["timing", "spread"] > 0.05 and sens.loc["timing", "spread"] > 3 * sens.loc["analog", "spread"]
    with pytest.raises(ValueError):
        C.sequential_credit(fr, comb, CFG, ["pattern"])


def test_counterfactual_table_is_the_audit_trail_of_one_decision():
    d = C.Decision("one", "2021-03-01", "2021-03-08", {"pattern": 1.0, "timing": 0.0, "risk": 2.0}, 0.05)
    t = C.counterfactual_table(d, C.StructuredCombiner({"pattern": 1.0}))
    assert len(t) == 8 and t.loc[t["n_active"] == 0, "utility"].iloc[0] == 0.0
    full = t[t["n_active"] == 3].iloc[0]
    assert full["decision"] == 0.0 and full["utility"] == 0.0                       # timing gate closed => no trade
    only_pattern = t[t["active"] == ("pattern",)].iloc[0]
    assert only_pattern["utility"] == pytest.approx(0.05)


def test_context_transfer_separates_portable_from_context_bound_credit():
    n = 480
    led = C.simulate_decisions(n, 33, lambda r, d: 0.6 * d.pattern + 0.6 * d.direction * (d.index < n / 2), noise=0.3, regime_flip_at=0.5)
    fr = frame_of(led)
    phi = C.CreditEngine(SUM4, CFG).raw_credit(fr)["phi"]
    t = C.context_transfer(fr, phi, "regime").set_index(["context", "component"])
    assert t.loc[("late", "pattern"), "transfers"] is True or t.loc[("late", "pattern"), "transfers"] == True   # noqa: E712
    assert t.loc[("late", "direction"), "inside"] < 0.5 * t.loc[("late", "direction"), "outside"]              # did not carry over
    with pytest.raises(KeyError):
        C.context_transfer(fr, phi, "nonexistent")


def test_update_proposals_are_bounded_and_never_move_belief_on_noise(single_report):
    led, rep = single_report
    props = {p.target: p for p in C.update_proposals(rep)}
    assert props["pattern"].action is C.UpdateAction.REINFORCE and 0 < props["pattern"].max_step <= 0.25
    for c in ("analog", "memory", "direction"):
        assert props[c].action is C.UpdateAction.HOLD and props[c].max_step == 0.0
    assert all(p.validate() == [] for p in props.values())
    bad = C.UpdateProposal("x", "component", C.UpdateAction.REINFORCE, 0.1, -0.1, 0.3, 50, max_step=0.2)
    assert bad.validate() != []                                                       # REINFORCE with an interval spanning 0
    assert C.UpdateProposal("x", "component", C.UpdateAction.HOLD, 0.0, -1, 1, 5, max_step=0.1).validate() != []


def test_knowledge_proposals_flag_a_context_where_credit_reverses():
    base = C.simulate_decisions(360, 35, lambda r, d: 0.5 * d.pattern * np.where(d.index < 180, 1.0, -0.5), noise=0.3, regime_flip_at=0.5)
    led = C.DecisionLedger(dataclasses.replace(d, knowledge={"pattern": {"k1": 1.0}}) for d in base)
    eng = C.CreditEngine(SUM4, CFG)
    rep = eng.assess(led, NOW)
    props = C.update_proposals(rep, eng.knowledge_credit(led, NOW))
    ctx = [p for p in props if p.level == "knowledge" and p.context != "all"]
    assert any(p.action is C.UpdateAction.INVESTIGATE and "anti-context" in p.rationale for p in ctx)


def test_component_confidence_uses_separate_dimensions_and_leaves_truth_untested(single_report):
    _, rep = single_report
    conf = C.component_confidence(rep.component("pattern"))
    assert conf.truth is None and "truth" in conf.untested()
    assert conf.usefulness == pytest.approx(1.0) and conf.current_reliability > 0.9 and conf.check() == []
    assert C.component_confidence(rep.component("analog")).usefulness == 0.0


def test_masked_pairs_hand_redundant_components_to_the_redundancy_module():
    led = C.simulate_decisions(200, 37, lambda r, d: 0.4 * d.pattern, noise=0.2)
    fr = frame_of(led)
    fr.scores["direction"] = fr.scores["pattern"] + 0.01 * np.random.default_rng(0).standard_normal(fr.n)
    comb = C.FunctionCombiner(lambda S: np.where(S["pattern"] != 0, S["pattern"], S["direction"]), "fallback")
    eng = C.CreditEngine(comb, CFG)
    pairs = C.masked_pairs(fr, eng.raw_credit(fr))
    assert [(p.a, p.b) for p in pairs] == [("pattern", "direction")] and pairs[0].score_corr > 0.99
    assert abs(pairs[0].loo_a) < 1e-3 < pairs[0].solo_a


def test_subsample_stability_keeps_the_true_component_on_top(single_report):
    led, _ = single_report
    s = C.subsample_stability(frame_of(led), SUM4, CFG, n_rep=12).set_index("component")
    assert s.loc["pattern", "same_sign"] == 1.0 and s.loc["pattern", "top_rank"] == 1.0


def test_reports_render_summarise_and_tabulate(single_report):
    _, rep = single_report
    s = C.summary_dict(rep)
    assert s["earners"] == ["pattern"] and s["problems"] == [] and s["label"] == "IMPLEMENTED — NOT VALIDATED"
    md = C.render_markdown(rep)
    assert md.startswith("# Credit assignment") and "IMPLEMENTED — NOT VALIDATED" in md and "| pattern |" in md
    assert "Why some components received no credit" in md
    tab = C.context_table(rep)
    assert {"dimension", "value", "component", "mean_credit", "dependence_p"} <= set(tab.columns)


def test_rolling_assessments_only_see_outcomes_matured_before_each_now():
    led = C.simulate_decisions(360, 41, lambda r, d: 0.5 * d.pattern, noise=0.4, start="2020-01-06")
    eng = C.CreditEngine(SUM4, dataclasses.replace(CFG, n_perm=150))
    reps = C.rolling_assessments(eng, led, ["2020-03-02", "2020-04-15", "2020-06-01"])
    assert [r.n_decisions for r in reps] == sorted(r.n_decisions for r in reps) and reps[0].n_decisions < reps[-1].n_decisions
    assert reps[0].n_pending > 0
    tab = C.earners_over_time(reps)
    assert list(tab.index) == ["2020-03-02", "2020-04-15", "2020-06-01"] and tab["pattern"].notna().iloc[-1]
    with pytest.raises(ValueError):
        C.rolling_assessments(eng, led, ["2020-06-01", "2020-03-02"])
    assert C.earners_over_time([]).empty


def test_blame_concentrates_in_the_context_where_the_component_fails():
    n = 600
    led = C.simulate_decisions(n, 43, lambda r, d: 0.5 * d.pattern * np.where(d.index < n / 2, 1.0, -1.0), noise=0.3, regime_flip_at=0.5)
    fr = frame_of(led)
    phi = C.CreditEngine(SUM4, CFG).raw_credit(fr)["phi"]
    tab = C.blame_by_context(fr, phi, "regime")
    late = tab[(tab.component == "pattern") & (tab.context == "late")].iloc[0]
    assert late["share_of_blame"] > 0.8 and late["lift"] > 1.5
    assert ("pattern", "late") == C.anti_context_candidates(tab)[0][:2]
    assert C.anti_context_candidates(pd.DataFrame()) == []
    with pytest.raises(KeyError):
        C.blame_by_context(fr, phi, "nope")


# ============================================================================================================================
# sampled Shapley for many components, and the automatic exact/sampled switch
# ============================================================================================================================
def _wide_ledger(k, n=160, seed=51, active=("c0", "c1", "c2")):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-06", periods=n // 4 + 1)
    led = C.DecisionLedger()
    names = [f"c{i}" for i in range(k)]
    for i in range(n):
        s = {c: float(rng.standard_normal()) for c in names}
        y = 0.4 * sum(s[c] for c in active) + 0.5 * rng.standard_normal()
        d = days[i // 4]
        led.add(C.Decision(f"w{i:04d}", str(d.date()), str((d + pd.offsets.BDay(5)).date()), s, float(y)))
    return led, C.WeightedSumCombiner({c: 1.0 for c in names})


def test_adaptive_sampling_agrees_with_exact_within_its_standard_error():
    led, comb = _wide_ledger(8)
    fr = frame_of(led)
    co = C.Coalitions(fr, comb, C.Utility.PAYOFF, CFG.neutral)
    exact = C.shapley_values(co, CFG, np.random.default_rng(0))[0].mean(axis=0)
    cfg = dataclasses.replace(CFG, shapley_method="sampled", sample_rel_se=0.02, max_orders=4000)
    phi, info = C.shapley_adaptive(co, cfg, np.random.default_rng(1))
    assert info.method == "sampled" and info.converged and info.n_orders <= cfg.max_orders
    assert np.allclose(phi.sum(axis=1), co.v(co.full_mask) - co.v(0))              # efficiency survives sampling
    assert np.all(np.abs(phi.mean(axis=0) - exact) <= 4 * info.max_se + 1e-12)        # within (4x) its own reported error
    assert np.array_equal(phi, C.shapley_adaptive(co, cfg, np.random.default_rng(1))[0])   # seeded => reproducible


def test_sampling_stops_at_the_budget_cap_and_says_it_did_not_converge():
    led, _ = _wide_ledger(8)
    comb = C.FunctionCombiner(lambda S: S["c0"] * S["c1"] + S["c2"] * S["c3"] * (S["c4"] > 0) + S["c5"], "nonadditive")
    co = C.Coalitions(frame_of(led), comb, C.Utility.PAYOFF, CFG.neutral)
    cfg = dataclasses.replace(CFG, sample_rel_se=0.0001, min_orders=8, max_orders=40)
    _, info = C.shapley_adaptive(co, cfg, np.random.default_rng(2))
    assert info.n_orders <= 40 and not info.converged and info.rel_se > 0.0001
    assert dataclasses.replace(CFG, max_orders=2, min_orders=8).validate() != []


def test_switch_is_automatic_and_recorded_in_the_report():
    led6 = C.simulate_decisions(200, 53, lambda r, d: 0.4 * d.pattern)
    rep6 = C.CreditEngine(SUM4, CFG).assess(led6, NOW)
    assert rep6.shapley_method == "exact" and rep6.shapley_orders == 0 and rep6.shapley_converged
    led, comb = _wide_ledger(12, n=120)
    co = C.Coalitions(frame_of(led), comb, C.Utility.PAYOFF, CFG.neutral)
    assert C.shapley_values_info(co, CFG, np.random.default_rng(0))[2].method == "sampled"
    forced = dataclasses.replace(CFG, shapley_method="exact")
    with pytest.raises(ValueError):
        big, bc = _wide_ledger(15, n=40)
        C.shapley_values_info(C.Coalitions(frame_of(big), bc, C.Utility.PAYOFF, CFG.neutral), forced, np.random.default_rng(0))
    assert "Shapley method: exact" in rep6.render_text()


def test_fourteen_components_run_end_to_end_and_find_the_planted_three():
    led, comb = _wide_ledger(14, n=160)
    cfg = dataclasses.replace(CFG, alpha=0.1, n_perm=150, n_boot=80, sample_rel_se=0.05, max_orders=400)
    import time
    t0 = time.time()
    rep = C.CreditEngine(comb, cfg).assess(led, NOW)
    assert time.time() - t0 < 60
    assert rep.shapley_method == "sampled" and rep.shapley_orders > 0 and rep.shapley_max_se > 0
    assert rep.interactions == ()                                  # dividends (hence pair indices) exist only for exact runs
    assert {"c0", "c1", "c2"} <= set(rep.earners())
    assert len(rep.earners()) <= 6
    assert rep.efficiency_error < 1e-9 and C.validate_report(rep) == []
