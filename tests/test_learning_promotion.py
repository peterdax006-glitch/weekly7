"""Tests for engine.learning.promotion (section 45) and engine.learning.champion (section 44).
Synthetic data only. Every gate has a planted defect that must be caught; the empty case is covered."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from engine.learning import champion as C
from engine.learning import promotion as PR
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle, Promotion,
                                  Provenance)

NOW = "2022-01-03"
HASH = "codehash0001"


@dataclasses.dataclass(frozen=True)
class FakeK:
    knowledge_id: str = "k1"
    version: int = 1
    epistemic: Epistemic = Epistemic.SUPPORTED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    promotion: Promotion = Promotion.CHALLENGER
    confidence: Confidence = Confidence(truth=0.9)
    provenance: Provenance = Provenance("2021-07-01T00:00:00", "2021-06-30", HASH, "d1", "c1", "e1", "r1", 7, "2021-06-30")
    contexts: dict = dataclasses.field(default_factory=dict)
    anti_contexts: dict = dataclasses.field(default_factory=dict)
    decision_effect: tuple = (DecisionEffect.RANKING,)


def good_evidence(seed=3) -> PR.PromotionEvidence:
    rng = np.random.default_rng(seed)
    oos_dates = tuple(str(dt.date(2021, 1, 4) + dt.timedelta(days=7 * i)) for i in range(30))
    return PR.PromotionEvidence(
        statistical=PR.StatisticalEvidence(0.010, 400, 5.0, 1e-7, 50),
        incremental=PR.IncrementalEvidence(tuple(rng.normal(0.004, 0.008, 60)), seed=1, decision_overlap=0.6),
        oos=PR.OOSEvidence("2020-12-31", oos_dates, tuple(rng.normal(0.006, 0.008, 30)), 0.010, ("w2021",), ("w2019", "w2020")),
        transfer=PR.TransferEvidence({f"c{i}": 0.006 + 0.001 * i for i in range(5)} | {"home": 0.01}, {f"c{i}": 30 for i in range(5)} | {"home": 99},
                                     ("home",), 0.010),
        risk=PR.RiskEvidence(40, -0.08, -0.15, -0.06, 0, -0.09, -0.16),
        memorization=PR.MemorizationEvidence(0.9, 0.85, ("m_vix", "rsi_14"), False, 0.1, 80),
        future=PR.FutureAudit(True, (), "2021-06-30", (), 0),
        repro=PR.ReproEvidence((PR.RerunRecord(0.0100, 1, HASH), PR.RerunRecord(0.0100, 1, HASH), PR.RerunRecord(0.0105, 2, HASH),
                                PR.RerunRecord(0.0098, 3, HASH))),
        stability=PR.StabilityEvidence((0.010, 0.012, 0.008, 0.011, 0.009, 0.013, 0.007, 0.010), (0.008, 0.009, 0.0095, 0.010), 0.010))


def gate(**kw):
    return PR.PromotionGate(PR.PromotionPolicy(**kw), code_hash=HASH)


def by_gate(dec):
    return {r.gate: r for r in dec.results}


def test_good_evidence_promotes():
    d = gate().evaluate(FakeK(), good_evidence(), NOW)
    assert d.promote, PR.render_rejection_report(d)
    assert all(r.status == PR.PASS for r in d.results)


def replace(ev, **kw):
    return dataclasses.replace(ev, **kw)


DEFECTS = {
    "statistical_validity": lambda e: replace(e, statistical=PR.StatisticalEvidence(0.010, 400, None, 0.02, 5000)),
    "incremental_value": lambda e: replace(e, incremental=PR.IncrementalEvidence(tuple(np.random.default_rng(0).normal(0.0, 0.01, 60)), 1)),
    "oos_confirmation": lambda e: replace(e, oos=dataclasses.replace(e.oos, oos_effects=tuple(-abs(x) for x in e.oos.oos_effects))),
    "cross_context_transfer": lambda e: replace(e, transfer=dataclasses.replace(e.transfer, context_effects={"c0": 0.01, "c1": -0.02, "c2": -0.01, "c3": 0.0, "home": 0.01})),
    "risk_acceptance": lambda e: replace(e, risk=dataclasses.replace(e.risk, worst_period=-0.73)),
    "anti_memorization": lambda e: replace(e, memorization=dataclasses.replace(e.memorization, identity_shuffle_retention=0.05)),
    "future_information_audit": lambda e: replace(e, future=PR.FutureAudit(True, ("label leaks into feature f7",), "2021-06-30", (), 0)),
    "reproducibility": lambda e: replace(e, repro=PR.ReproEvidence(tuple(dataclasses.replace(r, code_hash="oldcode") for r in e.repro.reruns))),
    "stability": lambda e: replace(e, stability=PR.StabilityEvidence((0.05, -0.01, -0.01, -0.02, -0.01, -0.01), (0.01,) * 4, 0.01)),
}


@pytest.mark.parametrize("name", sorted(DEFECTS))
def test_each_planted_defect_blocks_and_names_its_gate(name):
    d = gate().evaluate(FakeK(), DEFECTS[name](good_evidence()), NOW)
    assert not d.promote
    r = by_gate(d)[name]
    assert r.status == PR.FAIL, (r.status, r.detail)
    txt = PR.render_rejection_report(d)
    assert name in txt and "What evidence would change this" in txt


def test_provenance_defect_blocks():
    k = FakeK(provenance=Provenance("2021-07-01T00:00:00", "2021-06-30", HASH))        # no data/config/experiment/run/seed
    d = gate().evaluate(k, good_evidence(), NOW)
    assert by_gate(d)["provenance_completeness"].status == PR.FAIL
    assert "provenance_completeness" in d.critical_failures


@pytest.mark.parametrize("member", ["statistical", "incremental", "oos", "transfer", "risk", "memorization", "future", "repro", "stability"])
def test_missing_evidence_is_missing_not_pass(member):
    d = gate().evaluate(FakeK(), replace(good_evidence(), **{member: None}), NOW)
    assert not d.promote
    assert [r.status for r in d.results if r.status == PR.MISSING]


def test_empty_evidence_blocks_everything_measurable():
    d = gate().evaluate(FakeK(), PR.PromotionEvidence(), NOW)
    assert d.verdict == "BLOCK"
    st = by_gate(d)
    assert sum(r.status == PR.MISSING for r in d.results) == 9
    assert st["provenance_completeness"].status == PR.PASS       # the only gate that reads the knowledge object itself
    assert set(d.critical_failures) >= {"statistical_validity", "oos_confirmation", "reproducibility"}


def test_multiplicity_deflates_alpha():
    ev = replace(good_evidence(), statistical=PR.StatisticalEvidence(0.01, 400, None, 0.01, 1))
    assert by_gate(gate().evaluate(FakeK(), ev, NOW))["statistical_validity"].status == PR.PASS
    ev = replace(good_evidence(), statistical=PR.StatisticalEvidence(0.01, 400, None, 0.01, 5000))
    assert by_gate(gate().evaluate(FakeK(), ev, NOW))["statistical_validity"].status == PR.FAIL
    assert PR.adjusted_p(0.01, 1) == pytest.approx(0.01)
    assert PR.adjusted_p(0.01, 100, "bonferroni") == pytest.approx(1.0)
    assert PR.adjusted_p(0.0001, 100, "bonferroni") == pytest.approx(0.01)
    assert 0.0001 < PR.adjusted_p(0.0001, 100, "sidak") <= 0.0100001
    assert PR.one_sided_p(3.0) < 0.0015 < PR.one_sided_p(2.0) and PR.one_sided_p(-3.0) > 0.99


def test_effective_sample_can_fail_a_large_nominal_n():
    ev = replace(good_evidence(), statistical=PR.StatisticalEvidence(0.01, 4000, 6.0, 1e-9, 1, effective_n=12.0))
    r = by_gate(gate().evaluate(FakeK(), ev, NOW))["statistical_validity"]
    assert r.status == PR.FAIL and "effective sample" in r.detail


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    x = np.random.default_rng(1).normal(0.01, 0.02, 80)
    a, b = PR.block_bootstrap_ci(x, 5), PR.block_bootstrap_ci(x, 5)
    assert a == b
    assert a[0] < x.mean() < a[1]
    assert PR.block_bootstrap_ci(x, 6) != a
    assert np.isnan(PR.block_bootstrap_ci(np.array([1.0]), 1)[0])
    lo_, hi_ = PR.block_bootstrap_ci(x, 5, level=0.5)
    assert lo_ > a[0] and hi_ < a[1]


def test_incumbent_overlap_makes_incremental_value_unmeasurable():
    ev = replace(good_evidence(), incremental=dataclasses.replace(good_evidence().incremental, decision_overlap=0.995))
    assert by_gate(gate().evaluate(FakeK(), ev, NOW))["incremental_value"].status == PR.FAIL


def test_oos_dates_not_after_train_end_fail_and_future_dates_breach():
    e = good_evidence()
    early = dataclasses.replace(e.oos, oos_dates=("2020-06-01",) + e.oos.oos_dates[1:])
    r = by_gate(gate().evaluate(FakeK(), replace(e, oos=early), NOW))["oos_confirmation"]
    assert r.status == PR.FAIL and "not after train_end" in r.detail
    late = dataclasses.replace(e.oos, oos_dates=e.oos.oos_dates[:-1] + ("2022-01-03",))
    with pytest.raises(FirewallBreach):
        gate().evaluate(FakeK(), replace(e, oos=late), NOW)               # an outcome maturing ON now is not yet known


def test_oos_window_overlap_and_duplicates_fail():
    e = good_evidence()
    ov = dataclasses.replace(e.oos, train_windows=("w2021",))
    assert by_gate(gate().evaluate(FakeK(), replace(e, oos=ov), NOW))["oos_confirmation"].status == PR.FAIL
    dup = dataclasses.replace(e.oos, oos_dates=(e.oos.oos_dates[0],) + e.oos.oos_dates[:-1])
    assert "duplicate" in by_gate(gate().evaluate(FakeK(), replace(e, oos=dup), NOW))["oos_confirmation"].detail


def test_identity_bearing_feature_names_are_caught():
    m = dataclasses.replace(good_evidence().memorization, feature_names=("rsi_14", "ticker_rank", "trade_date"))
    r = by_gate(gate().evaluate(FakeK(), replace(good_evidence(), memorization=m), NOW))["anti_memorization"]
    assert r.status == PR.FAIL and "ticker_rank" in r.detail
    for ok in ("m_vix", "validate_flag", "dividend_yield"):
        assert not PR.IDENTITY_TOKENS.search(ok)


def test_answer_lookup_table_is_caught():
    m = dataclasses.replace(good_evidence().memorization, lookup_table_suspected=True, top_identity_share=0.8, n_distinct_identities=3)
    d = gate().evaluate(FakeK(), replace(good_evidence(), memorization=m), NOW)
    r = by_gate(d)["anti_memorization"]
    assert r.status == PR.FAIL and "lookup" in r.detail and "distinct" in r.detail


def test_reproducibility_needs_repeated_seed_and_deterministic_repeat():
    e = good_evidence()
    same = PR.ReproEvidence(tuple(PR.RerunRecord(0.01, 1, HASH) for _ in range(3)))
    r = by_gate(gate().evaluate(FakeK(), replace(e, repro=same), NOW))["reproducibility"]
    assert r.status == PR.FAIL and "one seed" in r.detail
    nondet = PR.ReproEvidence((PR.RerunRecord(0.0100, 1, HASH), PR.RerunRecord(0.0110, 1, HASH), PR.RerunRecord(0.0100, 2, HASH)))
    assert "not deterministic" in by_gate(gate().evaluate(FakeK(), replace(e, repro=nondet), NOW))["reproducibility"].detail
    flip = PR.ReproEvidence((PR.RerunRecord(0.01, 1, HASH), PR.RerunRecord(0.01, 1, HASH), PR.RerunRecord(-0.01, 2, HASH)))
    assert by_gate(gate().evaluate(FakeK(), replace(e, repro=flip), NOW))["reproducibility"].status == PR.FAIL


def test_stability_catches_one_lucky_period():
    st = PR.StabilityEvidence((0.10, 0.001, 0.001, -0.001, 0.001, 0.001, 0.001, 0.001), (0.01, 0.01, 0.01, 0.01), 0.01)
    r = by_gate(gate().evaluate(FakeK(), replace(good_evidence(), stability=st), NOW))["stability"]
    assert r.status == PR.FAIL and "one sub-period" in r.detail


def test_risk_gate_compares_with_incumbent_not_only_absolute_floors():
    r = dataclasses.replace(good_evidence().risk, worst_period=-0.12, incumbent_worst_period=-0.05)
    res = by_gate(gate().evaluate(FakeK(), replace(good_evidence(), risk=r), NOW))["risk_acceptance"]
    assert res.status == PR.FAIL and "incumbent" in res.detail


def test_non_finite_evidence_fails_the_gate_instead_of_crashing():
    inc = PR.IncrementalEvidence(tuple([0.01] * 40 + [float("nan")]), 1)
    d = gate().evaluate(FakeK(), replace(good_evidence(), incremental=inc), NOW)
    r = by_gate(d)["incremental_value"]
    assert r.status == PR.FAIL and "non-finite" in r.detail


def test_preconditions_block_wrong_state_and_no_decision_effect():
    d = gate().evaluate(FakeK(promotion=Promotion.SHADOW), good_evidence(), NOW)
    assert not d.promote and any("CHALLENGER" in p for p in d.preconditions)
    d = gate().evaluate(FakeK(epistemic=Epistemic.CONTRADICTED), good_evidence(), NOW)
    assert not d.promote and any("epistemic" in p for p in d.preconditions)
    d = gate().evaluate(FakeK(decision_effect=(DecisionEffect.NONE,)), good_evidence(), NOW)
    assert not d.promote and any("no decision" in p for p in d.preconditions)
    assert gate().evaluate(object(), good_evidence(), NOW).preconditions


def test_research_priority_knowledge_skips_risk_gate_only():
    k = FakeK(decision_effect=(DecisionEffect.RESEARCH_PRIORITY,))
    e = replace(good_evidence(), risk=None)
    d = gate().evaluate(k, e, NOW)
    assert by_gate(d)["risk_acceptance"].status == PR.NA and d.promote


def test_waivers_only_for_noncritical_gates_and_need_a_written_reason():
    assert PR.PromotionPolicy(waivers={"oos_confirmation": "x" * 40}).validate()
    assert PR.PromotionPolicy(waivers={"incremental_value": "x" * 40}).validate()
    assert PR.PromotionPolicy(waivers={"stability": "short"}).validate()
    with pytest.raises(ValueError):
        PR.PromotionGate(PR.PromotionPolicy(waivers={"future_information_audit": "we trust it completely"}))
    pol = PR.PromotionPolicy(waivers={"cross_context_transfer": "knowledge is scoped to one exchange by design"})
    d = PR.PromotionGate(pol, HASH).evaluate(FakeK(), replace(good_evidence(), transfer=None), NOW)
    assert by_gate(d)["cross_context_transfer"].status == PR.NA and d.promote
    assert pol.digest() != PR.PromotionPolicy().digest()


def test_decision_id_is_deterministic_and_sensitive():
    a = gate().evaluate(FakeK(), good_evidence(), NOW)
    b = gate().evaluate(FakeK(), good_evidence(), NOW)
    assert a.decision_id == b.decision_id
    c = gate().evaluate(FakeK(), DEFECTS["risk_acceptance"](good_evidence()), NOW)
    assert c.decision_id != a.decision_id
    assert gate(alpha=0.01).evaluate(FakeK(), good_evidence(), NOW).decision_id != a.decision_id


def test_report_is_written_once_and_lists_near_misses(tmp_path):
    d = gate().evaluate(FakeK(), DEFECTS["risk_acceptance"](good_evidence()), NOW)
    p = PR.write_rejection_report(tmp_path, d)
    assert p.exists() and p.with_suffix(".json").exists()
    assert "CRITICAL" in p.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        PR.write_rejection_report(tmp_path, d)
    e = good_evidence()
    near = replace(e, oos=dataclasses.replace(e.oos, in_sample_effect=0.0155))
    nd = gate().evaluate(FakeK(), near, NOW)
    assert "oos_confirmation" in [g for g, _ in PR.near_misses(nd, 0.5)] or by_gate(nd)["oos_confirmation"].ok


def test_failure_statistics_flags_always_blocked_gates():
    ds = [gate().evaluate(FakeK(), DEFECTS["reproducibility"](good_evidence(seed=s)), NOW) for s in range(4)]
    ds.append(gate().evaluate(FakeK(), good_evidence(), NOW))
    st = PR.failure_statistics(ds)
    assert st["n_decisions"] == 5 and st["promoted"] == 1
    assert st["by_gate"]["reproducibility"]["fail"] == 4
    assert "reproducibility" not in st["always_blocked"]
    assert PR.failure_statistics([])["never_failed"] == []


def test_policy_validation_rejects_nonsense():
    assert PR.PromotionPolicy(alpha=0.9).validate()
    assert PR.PromotionPolicy(multiplicity="none").validate()
    assert PR.PromotionPolicy(min_obs=1).validate()
    assert PR.PromotionPolicy(alpha=float("nan")).validate()
    assert not PR.PromotionPolicy().validate()


# ================================================================================================ champion board
def BK(kid="k1", **kw):
    kw.setdefault("promotion", Promotion.RESEARCH)
    return FakeK(kid, **kw)


def dates(n, start=dt.date(2021, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


def make_board(tmp_path, **pol):
    return C.KnowledgeBoard(tmp_path / "board.jsonl", gate(), C.BoardPolicy(**pol), tmp_path / "reports")


SLOT = C.Slot(DecisionEffect.RANKING, None, "global")


def shadow_to_challenger(b, k, start, cand=0.006, base=0.0, n=25, jitter=0.002, seed=0):
    rng = np.random.default_rng(seed)
    mid = b.register(k, SLOT, start)
    b.to_shadow(mid, start)
    for i, d in enumerate(dates(n, start + dt.timedelta(days=1))):
        b.record_shadow(mid, cand + rng.normal(0, jitter), base + rng.normal(0, jitter), d)
    b.open_challenge(mid, start + dt.timedelta(days=n + 2))
    return mid


T0 = dt.date(2021, 1, 1)
T_PROMO = dt.date(2022, 1, 3)


def test_first_champion_needs_gate_and_then_holds_the_slot(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    assert b.weight(mid) == 0.0                                       # a challenger never influences decisions
    out = b.attempt_promotion(k, good_evidence(), T_PROMO)
    assert out["promoted"], PR.render_rejection_report(out["decision"])
    assert b.champion(SLOT) == mid and b.weight(mid) == 1.0
    assert b.production_ids() == (mid,) and not b.invariants()


def test_failed_gate_leaves_slot_empty_and_writes_report(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    out = b.attempt_promotion(k, DEFECTS["anti_memorization"](good_evidence()), T_PROMO)
    assert not out["promoted"] and out["report"].exists()
    assert b.champion(SLOT) is None and b.view(mid).role == Promotion.CHALLENGER
    assert any(d["verdict"] == "BLOCK" for d in b.decisions)


def test_replacement_retires_incumbent_and_keeps_it(tmp_path):
    b = make_board(tmp_path)
    k1, k2 = BK("k1"), BK("k2")
    m1 = shadow_to_challenger(b, k1, T0)
    b.attempt_promotion(k1, good_evidence(), dt.date(2021, 9, 1))
    m2 = shadow_to_challenger(b, k2, dt.date(2021, 9, 5), cand=0.012, base=0.004)
    out = b.attempt_promotion(k2, good_evidence(), T_PROMO)
    assert out["promoted"] and b.champion(SLOT) == m2
    v = b.view(m1)
    assert v.role == Promotion.RETIRED and v.cause == "REPLACED" and len(v.path) >= 3      # retired, not deleted
    assert m1 in b.members and b.weight(m1) == 0.0


def test_challenger_not_better_than_incumbent_is_refused_even_if_gates_pass(tmp_path):
    b = make_board(tmp_path)
    k1, k2 = BK("k1"), BK("k2")
    shadow_to_challenger(b, k1, T0)
    b.attempt_promotion(k1, good_evidence(), dt.date(2021, 9, 1))
    m2 = shadow_to_challenger(b, k2, dt.date(2021, 9, 5), cand=0.0002, base=0.0002, jitter=0.004, seed=5)
    out = b.attempt_promotion(k2, good_evidence(), T_PROMO)
    assert out["decision"].promote and not out["head_to_head"]["ok"] and not out["promoted"]
    assert b.champion(SLOT) == "k1@v1" and b.view(m2).role == Promotion.CHALLENGER


def test_illegal_transitions_are_refused(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    mid = b.register(k, SLOT, T0)
    with pytest.raises(C.BoardError):
        b.open_challenge(mid, T0)                                      # RESEARCH -> CHALLENGER skips shadow
    with pytest.raises(C.BoardError):
        b.attempt_promotion(k, good_evidence(), T_PROMO)               # not a challenger
    with pytest.raises(C.BoardError):
        b.register(k, SLOT, T0)                                        # duplicate id: a change must be a new version
    b.to_shadow(mid, T0)
    with pytest.raises(C.BoardError):
        b.record_shadow(mid, float("nan"), 0.0, T0)
    with pytest.raises(C.BoardError):
        b.open_challenge(mid, T0)                                      # no shadow sessions yet


def test_research_only_knowledge_cannot_enter_a_slot(tmp_path):
    b = make_board(tmp_path)
    k = BK(decision_effect=(DecisionEffect.NONE,))
    with pytest.raises(C.BoardError):
        b.register(k, C.Slot(DecisionEffect.NONE), T0)
    with pytest.raises(C.BoardError):
        b.register(BK("k9"), C.Slot(DecisionEffect.EXIT), T0)        # effect not among the object's own effects


def test_shadow_that_harms_cannot_become_challenger(tmp_path):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, T0)
    b.to_shadow(mid, T0)
    rng = np.random.default_rng(0)
    for d in dates(25, T0 + dt.timedelta(days=1)):
        b.record_shadow(mid, -0.01 + rng.normal(0, 0.002), 0.0 + rng.normal(0, 0.002), d)
    with pytest.raises(C.BoardError, match="harm"):
        b.open_challenge(mid, dt.date(2021, 6, 1))


def test_max_challengers_per_slot(tmp_path):
    b = make_board(tmp_path, max_challengers_per_slot=1)
    shadow_to_challenger(b, BK("a"), T0)
    with pytest.raises(C.BoardError, match="challengers"):
        shadow_to_challenger(b, BK("b"), dt.date(2021, 3, 1))


def test_duplicate_shadow_session_is_refused(tmp_path):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, T0)
    b.to_shadow(mid, T0)
    b.record_shadow(mid, 0.01, 0.0, T0)
    with pytest.raises(C.BoardError, match="already exists"):
        b.record_shadow(mid, 0.02, 0.0, T0)


def test_watch_rolls_back_a_champion_that_underperforms_the_predecessor(tmp_path):
    b = make_board(tmp_path, watch_sessions=10)
    k1, k2 = BK("k1"), BK("k2")
    m1 = shadow_to_challenger(b, k1, T0)
    b.attempt_promotion(k1, good_evidence(), dt.date(2021, 9, 1))
    m2 = shadow_to_challenger(b, k2, dt.date(2021, 9, 5), cand=0.012, base=0.004)
    assert b.attempt_promotion(k2, good_evidence(), T_PROMO)["promoted"]
    assert b.review_watch(SLOT, T_PROMO)["status"] == "pending"
    rng = np.random.default_rng(1)
    for d in dates(12, T_PROMO + dt.timedelta(days=1)):
        b.record_watch(SLOT, -0.01 + rng.normal(0, 0.002), 0.004 + rng.normal(0, 0.002), d)
    res = b.review_watch(SLOT, T_PROMO + dt.timedelta(days=20))
    assert res["status"] == "rolled_back" and res["restored"] == m1
    assert b.champion(SLOT) == m1 and b.view(m2).role == Promotion.RETIRED and b.view(m2).cause == "ROLLED_BACK"
    assert not b.invariants()


def test_watch_confirms_a_good_champion(tmp_path):
    b = make_board(tmp_path, watch_sessions=10)
    k1, k2 = BK("k1"), BK("k2")
    shadow_to_challenger(b, k1, T0)
    b.attempt_promotion(k1, good_evidence(), dt.date(2021, 9, 1))
    m2 = shadow_to_challenger(b, k2, dt.date(2021, 9, 5), cand=0.012, base=0.004)
    b.attempt_promotion(k2, good_evidence(), T_PROMO)
    rng = np.random.default_rng(2)
    for d in dates(12, T_PROMO + dt.timedelta(days=1)):
        b.record_watch(SLOT, 0.01 + rng.normal(0, 0.002), 0.004 + rng.normal(0, 0.002), d)
    assert b.review_watch(SLOT, T_PROMO + dt.timedelta(days=20))["status"] == "confirmed"
    assert b.champion(SLOT) == m2


def test_retire_leaves_slot_empty_and_reinstate_goes_via_shadow(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    b.attempt_promotion(k, good_evidence(), dt.date(2021, 9, 1))
    with pytest.raises(C.BoardError):
        b.retire(mid, FailureCause.REGIME_CHANGE, "", dt.date(2021, 10, 1))
    b.retire(mid, FailureCause.REGIME_CHANGE, "edge reversed in the new regime", dt.date(2021, 10, 1))
    assert b.champion(SLOT) is None and b.production_ids() == () and b.weight(mid) == 0.0
    with pytest.raises(C.BoardError):
        b.retire(mid, FailureCause.UNKNOWN, "again", dt.date(2021, 10, 2))
    with pytest.raises(C.BoardError):
        b.reinstate(mid, dt.date(2021, 11, 1), "")
    b.reinstate(mid, dt.date(2021, 11, 1), "original regime returned")
    assert b.view(mid).role == Promotion.SHADOW
    with pytest.raises(C.BoardError):
        b.attempt_promotion(k, good_evidence(), T_PROMO)               # a reinstated item must re-earn CHALLENGER first


def test_replay_reproduces_state_and_detects_tampering(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    shadow_to_challenger(b, k, T0)
    b.attempt_promotion(k, good_evidence(), T_PROMO)
    b2 = make_board(tmp_path)
    assert b2.digest() == b.digest() and b2.champions == b.champions and b2.roles() == b.roles()
    assert b.ledger.verify()["ok"]
    lp = tmp_path / "board.jsonl"
    text = lp.read_text(encoding="utf-8")
    assert '"to_shadow"' in text
    lp.write_text(text.replace('"to_shadow"', '"reinstated"', 1), encoding="utf-8")       # edit history after the fact
    probs = make_board(tmp_path).invariants()
    assert probs and any(p.startswith("ledger") for p in probs)


def test_time_cannot_run_backwards_on_the_board(tmp_path):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, dt.date(2021, 6, 1))
    with pytest.raises(FirewallBreach):
        b.to_shadow(mid, dt.date(2021, 5, 1))


def test_stale_knowledge_object_cannot_override_board_role(tmp_path):
    b = make_board(tmp_path)
    k = FakeK(promotion=Promotion.CHAMPION)                             # object claims to be champion already
    with pytest.raises(C.BoardError):
        b.register(k, SLOT, T0)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    stale = dataclasses.replace(k, version=2)
    with pytest.raises(C.BoardError):
        b.attempt_promotion(stale, good_evidence(), T_PROMO)


def test_empty_board_is_well_defined(tmp_path):
    b = make_board(tmp_path)
    assert b.roles()["CHAMPION"] == [] and b.production_ids() == () and b.champion(SLOT) is None
    assert b.invariants() == [] and b.weight("nothing@v1") == 0.0
    assert "Knowledge board" in b.report()
    assert C.paired_stats([])["n"] == 0 and C.paired_stats([0.1])["t"] == 0.0
    assert C.BoardPolicy(min_shadow_sessions=1).validate()
    with pytest.raises(C.BoardError):
        C.KnowledgeBoard(tmp_path / "x.jsonl", policy=C.BoardPolicy(watch_sessions=1))


def test_slot_key_roundtrip():
    from engine.learning.core import Subsystem
    s = C.Slot(DecisionEffect.EXIT, Subsystem.RISK, "small_cap")
    assert C.Slot.from_key(s.key) == s
    assert C.Slot.from_key(C.Slot(DecisionEffect.RANKING).key) == C.Slot(DecisionEffect.RANKING)


# ================================================================================================ additions: evidence lint, policy, sensitivity
def test_validate_evidence_reports_malformed_input_without_raising():
    e = good_evidence()
    assert PR.validate_evidence(e, NOW) == []
    bad = replace(e, incremental=PR.IncrementalEvidence((0.1, float("nan")), 1, 1.7),
                  oos=dataclasses.replace(e.oos, oos_dates=("2021-03-01", "2021-01-01")),
                  transfer=dataclasses.replace(e.transfer, context_n={}),
                  risk=dataclasses.replace(e.risk, catastrophic_count=-1),
                  memorization=dataclasses.replace(e.memorization, identity_shuffle_retention=9.0),
                  future=PR.FutureAudit(True, (), "2022-01-03", (), 0))
    probs = " | ".join(PR.validate_evidence(bad, NOW))
    for frag in ("non-finite", "decision_overlap", "time order", "context_n missing", "negative", "implausible", "not before now"):
        assert frag in probs
    assert PR.validate_evidence(PR.PromotionEvidence(), NOW) == []


def test_evidence_completeness():
    assert PR.evidence_completeness(good_evidence()) == {"share": 1.0, "missing": []}
    c = PR.evidence_completeness(replace(good_evidence(), risk=None, oos=None))
    assert c["missing"] == ["oos", "risk"] and c["share"] == pytest.approx(7 / 9)


def test_compare_decisions_shows_what_new_evidence_fixed_or_broke():
    before = gate().evaluate(FakeK(), DEFECTS["reproducibility"](good_evidence()), NOW)
    after = gate().evaluate(FakeK(), good_evidence(), NOW)
    c = PR.compare_decisions(before, after)
    assert c["fixed"] == ["reproducibility"] and c["regressed"] == [] and c["same_policy"] and not c["same_evidence"]
    back = PR.compare_decisions(after, before)
    assert back["regressed"] == ["reproducibility"]


def test_sensitivity_flags_a_fragile_promotion_and_a_blocked_one_that_needs_loosening():
    fragile = PR.sensitivity(FakeK(), good_evidence(), NOW, code_hash=HASH)
    assert fragile["base"] == "PROMOTE" and not fragile["needs_loosening"]
    e = good_evidence()
    barely = replace(e, oos=dataclasses.replace(e.oos, in_sample_effect=0.0135))           # retention just under 0.35
    s = PR.sensitivity(FakeK(), barely, NOW, code_hash=HASH)
    if s["base"] == "BLOCK":
        assert s["needs_loosening"] and any(f["direction"] == "looser" and f["threshold"] == "min_oos_retention" for f in s["flips"])
    hopeless = PR.sensitivity(FakeK(), DEFECTS["risk_acceptance"](good_evidence()), NOW, code_hash=HASH)
    assert hopeless["base"] == "BLOCK"


def test_sensitivity_directions_are_labelled_correctly():
    tight = PR.sensitivity(FakeK(), good_evidence(), NOW, code_hash=HASH, factors=(1.25,))
    for f in tight["flips"]:
        assert f["direction"] in ("tighter", "looser")
        if f["threshold"] in PR.HIGHER_IS_STRICTER:
            assert (f["to"] > f["from"]) == (f["direction"] == "tighter") or f["from"] < 0
    assert PR.sensitivity(FakeK(), good_evidence(), NOW, code_hash=HASH, factors=())["robust"]


def test_diff_policies_labels_tighter_and_looser():
    old = PR.PromotionPolicy()
    new = dataclasses.replace(old, alpha=0.10, min_obs=100, worst_period_floor=-0.30, max_top_identity_share=0.2)
    d = {c["field"]: c["direction"] for c in PR.diff_policies(old, new)}
    assert d == {"alpha": "looser", "min_obs": "tighter", "worst_period_floor": "looser", "max_top_identity_share": "tighter"}
    assert PR.diff_policies(old, old) == []


def test_policy_log_requires_a_reason_to_loosen_and_remembers_history(tmp_path):
    log = PR.PolicyLog(tmp_path / "policies.jsonl")
    assert log.current() is None
    base = PR.PromotionPolicy()
    assert log.record(base, "2026-10-01", "initial policy")["recorded"]
    assert not log.record(base, "2026-10-02")["recorded"]                                # unchanged: nothing appended
    tighter = dataclasses.replace(base, min_obs=100)
    assert log.record(tighter, "2026-10-03")["loosened"] == []
    looser = dataclasses.replace(tighter, alpha=0.10)
    with pytest.raises(ValueError, match="written reason"):
        log.record(looser, "2026-10-04", "quick fix")
    r = log.record(looser, "2026-10-04", "alpha 0.05 was unreachable with the evidence available; owner approved")
    assert [c["field"] for c in r["loosened"]] == ["alpha"]
    assert log.current() == looser and len(log.history()) == 3
    assert log.loosened_since(base.digest()) == ["alpha"] and log.loosened_since(looser.digest()) == []
    assert log.ledger.verify()["ok"]
    with pytest.raises(ValueError):
        log.record(dataclasses.replace(looser, alpha=0.9), "2026-10-05", "x" * 30)


# ================================================================================================ additions: board analytics
def test_queue_orders_shadows_by_preregistered_gain_and_skips_unready(tmp_path):
    b = make_board(tmp_path, max_challengers_per_slot=2)
    mids = {}
    for name, gain in (("a", 0.001), ("b", 0.009), ("c", 0.005)):
        mids[name] = b.register(BK(name), SLOT, T0)
        b.to_shadow(mids[name], T0)
    rng = np.random.default_rng(5)
    for i, d in enumerate(dates(25, T0 + dt.timedelta(days=1))):       # sessions are recorded date by date, all members together
        for name in mids:
            if name != "c" or i < 3:                                    # c is observed for only 3 sessions
                b.record_shadow(mids[name], 0.006 + rng.normal(0, 0.002), rng.normal(0, 0.002), d)
    for name, gain in (("a", 0.001), ("b", 0.009), ("c", 0.005)):
        b.enqueue(mids[name], {"variant": name}, gain, dt.date(2021, 2, 1))
    with pytest.raises(Exception):
        b.enqueue(mids["a"], {"variant": "a"}, 0.001, dt.date(2021, 2, 1))          # already queued
    assert b.next_queued(SLOT, dt.date(2021, 6, 1)) == "b@v1"                        # highest expected gain that qualifies
    assert b.next_queued(SLOT, dt.date(2021, 6, 2)) == "a@v1"                        # c has too few sessions and is skipped
    assert b.next_queued(SLOT, dt.date(2021, 6, 3)) is None and [i["id"] for i in b.queue.items] == ["c@v1"]


def test_shadow_summary_and_sessions_needed(tmp_path):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, T0)
    b.to_shadow(mid, T0)
    assert C.shadow_summary(b, mid).n == 0
    rng = np.random.default_rng(0)
    for d in dates(40, T0 + dt.timedelta(days=1)):
        b.record_shadow(mid, 0.004 + rng.normal(0, 0.003), rng.normal(0, 0.003), d)
    s = C.shadow_summary(b, mid)
    assert s.n == 40 and s.mean > 0 and s.t > 2 and 0 < s.p_one_sided < 0.05 and 0.5 < s.hit_rate <= 1.0
    assert s.max_relative_drawdown <= 0 and s.first < s.last
    assert C.shadow_summary(b, mid, now=T0 + dt.timedelta(days=11)).n == 10                # only sessions strictly before now
    assert C.sessions_needed(0.004, 0.004) == pytest.approx(6.18 * 1.0, rel=0.02)
    assert C.sessions_needed(0.0, 0.01) == float("inf") and C.sessions_needed(-1.0, 0.01) == float("inf")
    assert C.sessions_needed(0.001, 0.01) > C.sessions_needed(0.002, 0.01)


def test_rank_challengers_corrects_for_picking_the_best_of_k(tmp_path):
    b = make_board(tmp_path, max_challengers_per_slot=5)
    rng = np.random.default_rng(7)
    mids = []
    for i in range(5):
        mids.append(b.register(BK(f"c{i}"), SLOT, T0))
        b.to_shadow(mids[-1], T0)
    for d in dates(30, T0 + dt.timedelta(days=1)):
        for i, mid in enumerate(mids):
            b.record_shadow(mid, (0.006 if i == 0 else 0.0) + rng.normal(0, 0.003), rng.normal(0, 0.003), d)
    for mid in mids:
        b.open_challenge(mid, dt.date(2021, 6, 1))
    rows = C.rank_challengers(b, SLOT, dt.date(2021, 6, 2))
    assert rows[0]["mid"] == "c0@v1" and rows[0]["eligible"]
    assert all(not r["eligible"] for r in rows[1:]) and rows[0]["p_adj"] >= rows[0]["p"]
    assert C.rank_challengers(b, C.Slot(DecisionEffect.EXIT), dt.date(2021, 6, 2)) == []


def test_scope_resolution_picks_the_most_specific_champion(tmp_path):
    b = make_board(tmp_path)
    general = C.Slot(DecisionEffect.RANKING, None, "global")
    small = C.Slot(DecisionEffect.RANKING, None, "small_cap")
    ks = {"g": (BK("g"), general), "s": (BK("s"), small)}
    mids = {n: b.register(k, sl, T0) for n, (k, sl) in ks.items()}
    for mid in mids.values():
        b.to_shadow(mid, T0)
    rng = np.random.default_rng(1)
    for d in dates(25, T0 + dt.timedelta(days=1)):
        for mid in mids.values():
            b.record_shadow(mid, 0.006 + rng.normal(0, 0.002), rng.normal(0, 0.002), d)
    for n in ks:
        b.open_challenge(mids[n], dt.date(2021, 6, 1))
    for n, (k, sl) in ks.items():
        assert b.attempt_promotion(k, good_evidence(), dt.date(2022, 1, 3))["promoted"]
    assert C.effective_champion(b, DecisionEffect.RANKING, None, ["small_cap", "global"]) == "s@v1"
    assert C.effective_champion(b, DecisionEffect.RANKING, None, ["large_cap", "global"]) == "g@v1"
    assert C.effective_champion(b, DecisionEffect.RANKING, None, ["large_cap"]) is None
    assert C.effective_champion(b, DecisionEffect.EXIT, None, ["global"]) is None
    assert C.scope_conflicts(b) == [("s@v1", "g@v1")]


def test_health_verdicts_map_to_board_actions(tmp_path):
    from engine.learning.core import Health
    b = make_board(tmp_path)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    b.attempt_promotion(k, good_evidence(), dt.date(2021, 9, 1))
    for h, action in ((Health.HEALTHY, "none"), (Health.DORMANT, "hold"), (Health.DEGRADING, "review"),
                      (Health.INSUFFICIENT_EVIDENCE, "abstain"), (Health.UNKNOWN, "abstain")):
        r = C.apply_health(b, mid, h, dt.date(2021, 9, 5))
        assert r["action"] == action and not r["applied"] and b.view(mid).role == Promotion.CHAMPION
    r = C.apply_health(b, mid, Health.CONTRADICTED, dt.date(2021, 9, 6), "edge reversed on 3 windows")
    assert r["applied"] and b.view(mid).role == Promotion.RETIRED and b.view(mid).cause == "REVERSAL"
    assert not C.apply_health(b, mid, Health.BROKEN, dt.date(2021, 9, 7))["applied"]        # already retired: no double retire


def test_broken_challenger_is_demoted_not_retired(tmp_path):
    from engine.learning.core import Health
    b = make_board(tmp_path)
    mid = shadow_to_challenger(b, BK(), T0)
    r = C.apply_health(b, mid, Health.BROKEN, dt.date(2021, 8, 1))
    assert r["applied"] and b.view(mid).role == Promotion.SHADOW


def test_review_due_slot_status_funnel_explain_and_snapshot(tmp_path):
    b = make_board(tmp_path)
    k = BK()
    mid = shadow_to_challenger(b, k, T0)
    b.attempt_promotion(k, DEFECTS["stability"](good_evidence()), dt.date(2021, 9, 1))
    b.attempt_promotion(k, good_evidence(), dt.date(2021, 9, 2))
    assert C.review_due(b, dt.date(2021, 10, 1)) == []
    due = C.review_due(b, dt.date(2022, 1, 15))
    assert due[0]["mid"] == mid and due[0]["age_days"] > 90
    rows = C.slot_status(b, dt.date(2022, 1, 15))
    assert rows[0]["champion"] == mid and rows[0]["tenure_days"] == due[0]["age_days"] and rows[0]["retired"] == 0
    f = C.promotion_funnel(b)
    assert f["attempts"] == 2 and f["events"]["promoted"] == 1 and f["refusal_reasons"] == {"stability": 1}
    story = C.explain_member(b, mid)
    assert any("promotion_refused" in ln and "stability" in ln for ln in story) and any("... 25 observations" in ln for ln in story)
    snap = C.to_snapshot(b)
    assert snap["champions"] == {SLOT.key: mid} and snap["as_of"] == "2021-09-02" and snap["digest"] == b.digest()
    from engine.learning import compute as CM
    st = CM.SnapshotStore(tmp_path / "snap")
    sid = st.create(snap, snap["as_of"])
    assert st.load(sid)["state"]["champions"] == snap["champions"]


def test_empty_board_analytics(tmp_path):
    b = make_board(tmp_path)
    assert C.slot_status(b, dt.date(2022, 1, 1)) == [] and C.review_due(b, dt.date(2022, 1, 1)) == []
    assert C.promotion_funnel(b) == {"events": {}, "refusal_reasons": {}, "attempts": 0}
    assert C.to_snapshot(b)["champions"] == {} and C.scope_conflicts(b) == []
    assert C.rank_challengers(b, SLOT, dt.date(2022, 1, 1)) == []
    with pytest.raises(C.BoardError):
        C.explain_member(b, "ghost@v1")
    with pytest.raises(C.BoardError):
        C.shadow_summary(b, "ghost@v1")


# ================================================================================================ additions: sequential, context, recovery, time travel
def test_obrien_fleming_boundary_demands_more_early():
    assert C.obrien_fleming_bound(1.0) == pytest.approx(1.645, abs=0.01)
    assert C.obrien_fleming_bound(0.25) == pytest.approx(3.29, abs=0.02)
    assert C.obrien_fleming_bound(0.25) > C.obrien_fleming_bound(0.5) > C.obrien_fleming_bound(1.0)
    with pytest.raises(ValueError):
        C.obrien_fleming_bound(0.0)


def shadow_board_with(tmp_path, cand_edge, n=30, seed=0, sd=0.002):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, T0)
    b.to_shadow(mid, T0)
    rng = np.random.default_rng(seed)
    for d in dates(n, T0 + dt.timedelta(days=1)):
        b.record_shadow(mid, cand_edge + rng.normal(0, sd), rng.normal(0, sd), d)
    return b, mid


def test_sequential_verdict_lets_a_real_edge_through_and_stops_a_harmful_one(tmp_path):
    b, mid = shadow_board_with(tmp_path, 0.006)
    v = C.sequential_verdict(b, mid, dt.date(2021, 6, 1), max_sessions=60)
    assert v.action == "READY" and v.z >= v.boundary and 0.4 < v.fraction < 0.6
    b2, mid2 = shadow_board_with(tmp_path / "harm", -0.006)
    assert C.sequential_verdict(b2, mid2, dt.date(2021, 6, 1)).action == "ABANDON"


def test_sequential_verdict_does_not_promote_noise_on_an_early_peek(tmp_path):
    b, mid = shadow_board_with(tmp_path, 0.0009, n=25, seed=3)                 # a small edge that looks fine at t>1.64 but not at OF
    v = C.sequential_verdict(b, mid, dt.date(2021, 6, 1), max_sessions=100)
    assert v.action in ("CONTINUE", "ABANDON") and v.action != "READY"
    b3, mid3 = shadow_board_with(tmp_path / "few", 0.01, n=8)
    assert C.sequential_verdict(b3, mid3, dt.date(2021, 6, 1)).action == "CONTINUE"
    assert C.sequential_verdict(b3, mid3, dt.date(2021, 6, 1)).boundary == float("inf")
    with pytest.raises(ValueError):
        C.sequential_verdict(b3, mid3, dt.date(2021, 6, 1), max_sessions=5)


def test_sequential_verdict_abandons_a_flat_record_at_half_budget(tmp_path):
    b, mid = shadow_board_with(tmp_path, -0.0002, n=32, seed=11)
    v = C.sequential_verdict(b, mid, dt.date(2021, 6, 1), max_sessions=60)
    assert v.action in ("ABANDON", "CONTINUE")
    if v.z < 0:
        assert v.action == "ABANDON"


def test_context_breakdown_flags_a_one_regime_edge(tmp_path):
    b = make_board(tmp_path)
    mid = b.register(BK(), SLOT, T0)
    b.to_shadow(mid, T0)
    labels = {}
    rng = np.random.default_rng(2)
    for i, d in enumerate(dates(40, T0 + dt.timedelta(days=1))):
        calm = i % 2 == 0
        labels[str(d)] = "calm" if calm else "stress"
        edge = 0.01 if calm else 0.0
        b.record_shadow(mid, edge + rng.normal(0, 0.002), rng.normal(0, 0.002), d)
    bd = C.shadow_by_context(b, mid, labels)
    assert set(bd) == {"calm", "stress"} and bd["calm"]["mean"] > 0.005 and abs(bd["stress"]["mean"]) < 0.003
    cc = C.context_consistency(bd)
    assert cc["concentrated"] and cc["top_context"] == "calm" and cc["contexts"] == 2
    assert C.shadow_by_context(b, mid, {})["unlabelled"]["n"] == 40                      # missing labels are kept, not dropped
    assert C.context_consistency({})["contexts"] == 0
    tiny = C.shadow_by_context(b, mid, labels, now=T0 + dt.timedelta(days=4))
    assert not any(v["reliable"] for v in tiny.values()) and C.context_consistency(tiny)["contexts"] == 0


def test_only_situational_retirements_are_recovery_candidates(tmp_path):
    b = make_board(tmp_path)
    ks = {n: BK(n) for n in ("dormant", "false", "reversed")}
    mids = {}
    for i, (n, k) in enumerate(ks.items()):
        mids[n] = shadow_to_challenger(b, k, dt.date(2021, 1, 1) + dt.timedelta(days=40 * i), seed=len(n))
    for n, cause in (("dormant", FailureCause.TEMPORARY_INACTIVITY), ("false", FailureCause.FALSE_PATTERN),
                     ("reversed", FailureCause.REVERSAL)):
        b.retire(mids[n], cause, f"retired as {cause.value}", dt.date(2021, 8, 1))
    ok = lambda m: True
    assert C.recovery_candidates(b, ok) == [mids["dormant"]]
    assert C.recovery_candidates(b, lambda m: False) == []
    assert C.reinstate_recovered(b, ok, dt.date(2021, 9, 1)) == [mids["dormant"]]
    assert b.view(mids["dormant"]).role == Promotion.SHADOW and b.view(mids["false"]).role == Promotion.RETIRED
    assert C.reinstate_recovered(b, ok, dt.date(2021, 9, 2)) == []


def test_board_as_of_shows_production_on_a_past_date_and_is_read_only(tmp_path):
    b = make_board(tmp_path, max_challengers_per_slot=3)
    k1, k2 = BK("k1"), BK("k2")
    m1 = shadow_to_challenger(b, k1, T0)
    b.attempt_promotion(k1, good_evidence(), dt.date(2021, 9, 1))
    m2 = shadow_to_challenger(b, k2, dt.date(2021, 9, 5), cand=0.012, base=0.004)
    b.attempt_promotion(k2, good_evidence(), T_PROMO)
    lp = tmp_path / "board.jsonl"
    assert C.production_at(lp, dt.date(2021, 8, 1)) == ()
    assert C.production_at(lp, dt.date(2021, 9, 2)) == (m1,)                 # events ON the date are not yet visible
    assert C.production_at(lp, dt.date(2021, 9, 3)) == (m1,)
    assert C.production_at(lp, dt.date(2022, 6, 1)) == (m2,)
    then, now_ = C.board_as_of(lp, dt.date(2021, 9, 3)), C.board_as_of(lp, dt.date(2022, 6, 1))
    d = C.diff_boards(then, now_)
    assert d["added"] == [m2] and d["removed"] == [m1] and d["replaced"] == [(m1, m2)] and m2 in d["new_members"]
    with pytest.raises(C.BoardError, match="read-only"):
        then.retire(m1, FailureCause.UNKNOWN, "tamper with history", dt.date(2022, 7, 1))
    assert C.diff_boards(then, then) == {"added": [], "removed": [], "replaced": [], "new_members": []}
    assert not then.invariants()
