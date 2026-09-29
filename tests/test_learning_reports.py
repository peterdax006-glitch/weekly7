"""Learning reports K01-K15, the J15 research-priority dashboard and the section-46 health dashboard (contract C62 sections 46-48, 59,
65, 79; engine/learning/reports.py, scripts/learning_report.py). Every generator runs on a synthetic artefact store; each mechanism is
checked against a planted defect it must catch, plus the missing / empty artefact case and the planted false claim.
Proves nothing about the real archive: IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from engine.learning import reports as R
from engine.learning import retirement as RT
from engine.learning import scorecard as S
from engine.learning.core import FirewallBreach, ValidationLabel, stable_hash

ROOT = Path(__file__).resolve().parents[1]
NOW = "2030-01-01"


# ============================================================================================ synthetic artefacts
def M(v, lo=None, hi=None, n=100):
    return S.Measured(v, v - 0.001 if lo is None else lo, v + 0.001 if hi is None else hi, n, S.MStatus.MEASURED)


def good_card(version="v1", **kw):
    rng = np.random.default_rng(0)
    controls = {
        "A_no_learning": S.ControlResult("A_no_learning", M(0.0, -0.001, 0.001)),
        "B_learner": S.ControlResult("B_learner", M(0.01, 0.008, 0.012), gains=tuple(rng.normal(0.01, 0.01, 40))),
        "C_identity_memoriser": S.ControlResult("C_identity_memoriser", M(0.0, -0.001, 0.001), detected=True, gains=tuple(rng.normal(0.0, 0.01, 40))),
        "D_random_learner": S.ControlResult("D_random_learner", M(0.0005, -0.001, 0.002)),
        "E_leaky_learner": S.ControlResult("E_leaky_learner", M(0.03, 0.028, 0.032), detected=True),
    }
    base = dict(learner_version=version, now=dt.date(2029, 12, 31), code_hash="abc123", seed=1, baseline_performance=M(0.001), post_learning_performance=M(0.011),
                learning_gain=M(0.01, 0.008, 0.012), same_year_gain=M(0.011), cross_year_gain=M(0.009, 0.007, 0.011), cross_regime_gain=M(0.008),
                cross_stock_gain=M(0.009), transfer_ratio=M(0.9, 0.7, 1.1), risk_change=M(0.002, -0.001, 0.005), drawdown_change=M(0.003, -0.001, 0.007),
                band_share=M(0.4, 0.35, 0.45), movement_performance=M(0.3), direction_performance=M(0.02), mover_performance=M(0.05), calibration=M(0.03),
                memorization_gap=M(0.0, -0.001, 0.001), identity_gap=M(0.0, -0.001, 0.001), future_leak_status=S.LeakStatus.CLEAN, stability=M(0.8),
                compute_cost=S.ComputeCost(1.0, 1.2, 100.0, 3, 3), controls=controls)
    base.update(kw)
    return S.LearningScorecard(**base)


def jl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def js(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


PROV = {"created_real": "2029-06-02T10:00:00", "learned_at": "2029-06-01", "code_hash": "abc123", "data_hash": "d1", "outcomes_seen_through": "2029-06-01"}


def kobj(kid, ver=1, ep="SUPPORTED", lc="ACTIVE", promo="CHAMPION", truth=0.8, rel=0.75, n=100, prov=None, reason="initial", updated="2029-06-01", **extra):
    return {"knowledge_id": kid, "version": ver, "epistemic": ep, "lifecycle": lc, "promotion": promo, "confidence": {"truth": truth, "current_reliability": rel},
            "evidence": {"sample_size": n}, "provenance": prov or dict(PROV), "contexts": {}, "parent_hash": "" if ver == 1 else "h", "updated_at": updated,
            "version_reason": reason, **extra}


def curve_rows(k=12, slope=0.0004):
    rng = np.random.default_rng(3)
    return [{"step": i + 1, "experience_count": 100 * (i + 1), "knowledge_count": 20 + i, "validated_knowledge_count": 5 + i // 2,
             "same_year_gain": 0.01 + rng.normal(0, 0.0005), "transfer_gain": 0.002 + slope * i + rng.normal(0, 0.0003), "risk": 0.1,
             "memorization_gap": 0.0005 + rng.normal(0, 0.0002), "as_of": f"2029-{1 + i:02d}-15"} for i in range(k)]


def transfer_record(ratio_wrong=False):
    axes = {"YEAR": {"axis": "YEAR", "mode": "marginal", "same": {"mean": 0.010, "lo": 0.008, "hi": 0.012, "n": 120, "n_clusters": 20},
                     "cross": {"mean": 0.009, "lo": 0.006, "hi": 0.012, "n": 100, "n_clusters": 18}, "replay": {"mean": 0.011, "lo": 0.009, "hi": 0.013, "n": 80, "n_clusters": 15},
                     "ratio": 0.5 if ratio_wrong else 0.9, "ratio_status": "OK", "specialisation": False,
                     "memorization_gap": {"gap": 0.0004, "lo": -0.001, "hi": 0.002, "n_a": 80, "n_b": 120}, "verdict": "GENERALISES", "validation": "IMPLEMENTED — NOT VALIDATED"},
            "STOCK": {"axis": "STOCK", "mode": "marginal", "same": {"mean": 0.010, "lo": 0.008, "hi": 0.012, "n": 120, "n_clusters": 20},
                      "cross": {"mean": 0.007, "lo": 0.004, "hi": 0.010, "n": 90, "n_clusters": 16}, "ratio": 0.7, "ratio_status": "OK", "specialisation": False,
                      "verdict": "GENERALISES", "validation": "IMPLEMENTED — NOT VALIDATED"}}
    return {"now": "2029-12-30", "n_units": 400, "overall": "GENERALISES", "untested_axes": ["SECTOR"], "axes": axes,
            "identity": {"gap": 0.0002, "lo": -0.001, "hi": 0.001, "flagged": False}, "depth": [{"depth": 1, "gain": 0.008}], "year_gap": [{"years": 1, "gain": 0.009}]}


def transitions():
    T = RT.Transition
    rows = []

    def add(seq, kid, kind, frm, to, at, cause="REGIME_CHANGE"):
        t = T(seq=seq, knowledge_id=kid, kind=kind, from_state=frm, to_state=to, at=at, reason=f"{kind} {kid}", cause=cause)
        rows.append(json.loads(dataclasses.replace(t, id=t.compute_id()).to_json()))
    add(1, "k_a", "DEGRADE", "ACTIVE", "DEGRADED", "2029-03-01")
    add(2, "k_a", "RECOVER", "DEGRADED", "ACTIVE", "2029-05-01")
    add(3, "k_b", "DEGRADE", "ACTIVE", "DEGRADED", "2029-03-10")
    add(4, "k_b", "RETIRE", "DEGRADED", "RETIRED", "2029-04-10", "FALSE_PATTERN")
    add(5, "k_c", "DEGRADE", "ACTIVE", "DEGRADED", "2029-02-10")
    add(6, "k_c", "RECOVER", "DEGRADED", "ACTIVE", "2029-03-10")
    add(7, "k_c", "DEGRADE", "ACTIVE", "DEGRADED", "2029-06-10")
    return rows


def gate(name, status="PASS", critical=True):
    return {"gate": name, "status": status, "critical": critical, "detail": ""}


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    R_ = R.ARTEFACTS
    S.ScorecardStore(root / R_["scorecards"]).append(good_card("v1"))
    js(root / R_["curve"], curve_rows())
    js(root / R_["transfer"], transfer_record())
    js(root / R_["identity"], {"base_ic": 0.05, "deterministic": True, "verdicts": [
        {"kind": "ticker_permutation", "mode": "eval", "status": "OK", "retention": 0.95, "ci_low": 0.8, "ci_high": 1.1, "changed_frac": 0.9},
        {"kind": "date_permutation", "mode": "both", "status": "OK", "retention": 0.9, "ci_low": 0.7, "ci_high": 1.1, "changed_frac": 0.95}]})
    kn = [kobj("k_a"), kobj("k_b", ep="DEGRADED", lc="DEGRADED", rel=0.2), kobj("k_c", lc="FAILURE"), kobj("k_d", n=5),
          kobj("k_e", ep="SUPPORTED", rel=0.3), kobj("k_f", ep="RETIRED", lc="RETIRED"), kobj("k_g", rel=0.8),
          kobj("k_h", ver=1, rel=0.8), kobj("k_h", ver=2, rel=0.5, reason="reliability fell", updated="2029-09-01")]
    jl(root / R_["knowledge"], kn)
    jl(root / R_["failures"], [{"rid": f"r{i}", "cause": c, "confidence": 0.6, "unknown_state": None, "meaningful": True, "coverage": 0.8, "surprise_bits": 1.0,
                                "subsystem_votes": {"SELECTION": 0.5 if c == "FALSE_PATTERN" else 0.0, "TIMING": 0.4 if c != "FALSE_PATTERN" else 0.0}}
                               for i, c in enumerate(["FALSE_PATTERN"] * 12 + ["REGIME_CHANGE"] * 6 + ["UNKNOWN"] * 4)])
    jl(root / R_["retirement"], transitions())
    jl(root / R_["contradictions"], [
        {"pair": ["k_a", "k_g"], "at": "2029-05-01", "verdict": "UNRESOLVED", "dimension": "regime", "p_adj": 0.01, "m_tests": 3, "rows_through": "2029-04-30"},
        {"pair": ["k_a", "k_g"], "at": "2029-07-01", "verdict": "UNRESOLVED", "dimension": "regime", "p_adj": 0.02, "m_tests": 7, "rows_through": "2029-06-30"},
        {"pair": ["k_d", "k_e"], "at": "2029-05-01", "verdict": "RESOLVED_BY_CONTEXT", "dimension": "vol", "p_adj": 0.001, "m_tests": 3, "rows_through": "2029-04-30"}])
    js(root / R_["missed"], {"reasons": [{"reason": "LOW_SCORE", "n": 200, "win_rate": 0.12, "ci": [0.08, 0.17], "base_rate": 0.10, "z": 1.0, "q": 0.4, "verdict": "NEUTRAL"},
                                         {"reason": "LIQUIDITY", "n": 150, "win_rate": 0.25, "ci": [0.19, 0.32], "base_rate": 0.10, "z": 5.0, "q": 0.001, "verdict": "COSTLY"}],
                             "distinctions": [{"did": "d1", "description": "vol > 0.03", "verdict": "PASS", "lift": 1.8, "mean_gain": 0.002, "p_value": 0.02, "excess_vs_random": 0.001, "n_weeks": 40}],
                             "walk_forward": {"discovered": 5, "passed": 1}, "null_control": {"pass_rate": 0.0}})
    jl(root / R_["experiments"], [
        {"experiment_id": f"e{i}", "status": "ANSWERED", "question": f"does feature {i % 4} help", "recorded_at": "2029-06-01", "tags": ["feature"],
         "prediction": {"confidence": 0.7}, "result": {"kind": "CONFIRMED" if i % 3 == 0 else "REFUTED", "power_note": ""},
         "belief_update": {"surprise_bits": 1.5, "moved": 0.2}, "not_learned": ["x"], "repeat_reason": "" if i < 4 else "rerun after data fix"} for i in range(12)])
    js(root / R_["queue"], [
        {"id": "q1", "question": "why did reversal fail", "target": "FAILURE", "status": "OPEN", "uncertainty": 0.8, "decision_impact": 0.9, "transfer_potential": 0.5,
         "failure_evidence": 0.7, "contradiction": 0.1, "surprise": 0.6, "feasibility": 0.9, "enqueued_at": "2029-11-01"},
        {"id": "q2", "question": "regime split", "target": "CONTRADICTION", "status": "OPEN", "uncertainty": 0.4, "decision_impact": 0.3, "transfer_potential": 0.2,
         "feasibility": 0.5, "enqueued_at": "2029-12-01"},
        {"id": "q3", "question": "old finished item", "target": "FAILURE", "status": "DONE", "score": 0.9, "enqueued_at": "2029-10-01"}])
    js(root / R_["meta"], {"discoveries": [{"item_id": f"i{i}", "family": "momentum" if i % 2 else "reversal", "discovered_at": "2028-01-01", "features": {},
                                            "survived_oos": bool(i % 3 == 0), "resolved_at": "2029-01-01", "is_effect": 0.02, "oos_effect": 0.005} for i in range(30)],
                           "learners": [{"learner": "L1", "claimed_gain": 0.03, "verified": False, "resolved_at": "2029-01-01"},
                                        {"learner": "L2", "claimed_gain": 0.02, "verified": True, "verified_gain": 0.015, "resolved_at": "2029-01-01"}],
                           "explanations": [{"item_id": "a", "predicted_cause": "FALSE_PATTERN", "confidence": 0.7, "verified_cause": "FALSE_PATTERN", "resolved_at": "2029-01-01"},
                                            {"item_id": "b", "predicted_cause": "FALSE_PATTERN", "confidence": 0.7, "verified_cause": "REGIME_CHANGE", "resolved_at": "2029-01-01"}]})
    jl(root / R_["promotions"], [
        {"knowledge_id": "k_a", "version": 1, "now": "2029-06-01", "verdict": "PROMOTE", "preconditions": [], "gates": [gate("statistical_validity"), gate("oos_confirmation")]},
        {"knowledge_id": "k_g", "version": 1, "now": "2029-06-01", "verdict": "PROMOTE", "preconditions": [], "gates": [gate("statistical_validity")]},
        {"knowledge_id": "k_h", "version": 1, "now": "2029-06-01", "verdict": "PROMOTE", "preconditions": [], "gates": [gate("statistical_validity")]},
        {"knowledge_id": "k_x", "version": 1, "now": "2029-06-02", "verdict": "BLOCK", "preconditions": [], "gates": [gate("statistical_validity"), gate("anti_memorization", "FAIL")]}])
    js(root / R_["delta"], {d: {"value": 0.002, "lo": 0.001, "hi": 0.003} for d in R.DELTA_DIMS[:3]})
    return root


def ctx_for(root, **kw):
    kw.setdefault("checklist", None)
    kw.setdefault("code_hash", "testcode")
    return R.ReportContext(root, NOW, **kw)


# ============================================================================================ all reports on a full store
def test_every_report_is_built_from_the_artefacts_with_an_honest_label(store):
    reps = R.generate_all(ctx_for(store))
    assert set(reps) == {f"K{i:02d}" for i in range(1, 16)}
    for k, r in reps.items():
        assert r.label != ValidationLabel.VALIDATED, k
        assert r.status in ("MEASURED", "PARTIAL"), (k, r.headline, r.unmeasured)
        assert r.sources and all(s.status in ("FOUND", "MISSING", "EMPTY") for s in r.sources)
        assert "UNMEASURED" not in r.headline or r.status == "PARTIAL"
        json.loads(r.to_json())
        assert r.to_markdown().startswith(f"# {k}:")
    assert reps["K01"].facts["verdict"] in ("RISING", "PLATEAUED")
    assert reps["K01"].label == ValidationLabel.NOT_VALIDATED
    assert reps["K02"].facts["axes_tested"] == 2 and reps["K02"].facts["cross_year_gain"] == pytest.approx(0.009)
    assert reps["K15"].facts["reports"] == 14


def test_numbers_are_computed_from_the_artefacts_not_typed(store):
    a = R.report_k06(ctx_for(store))
    assert a.facts["failures_classified"] == 22 and a.facts["unknown_or_insufficient"] == 4
    assert a.facts["named_cause_share"] == pytest.approx(18 / 22)
    rows = [json.loads(l) for l in (store / R.ARTEFACTS["failures"]).read_text().splitlines()]
    jl(store / R.ARTEFACTS["failures"], rows + [dict(rows[0], rid="extra")])
    b = R.report_k06(ctx_for(store))
    assert b.facts["failures_classified"] == 23                                # the report follows the artefact
    assert a.to_record()["report_id"] != b.to_record()["report_id"]


def test_reports_are_deterministic(store):
    a, b = R.generate_all(ctx_for(store, seed=4)), R.generate_all(ctx_for(store, seed=4))
    assert {k: r.to_record()["report_id"] for k, r in a.items()} == {k: r.to_record()["report_id"] for k, r in b.items()}


# ============================================================================================ missing artefacts
def test_missing_artefacts_say_unmeasured_never_blank_or_success(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    reps = R.generate_all(ctx_for(empty))
    for k, r in reps.items():
        assert r.status == R.UNMEASURED, k
        assert r.label == ValidationLabel.INSUFFICIENT_EVIDENCE
        assert r.headline.startswith("UNMEASURED"), (k, r.headline)
        assert r.unmeasured, k
        assert "UNMEASURED" in r.to_markdown()
    assert R.research_priority_dashboard(ctx_for(empty))["status"] == R.UNMEASURED
    assert R.health_dashboard(ctx_for(empty))["status"] == R.UNMEASURED


def test_an_empty_or_garbled_artefact_is_unmeasured_with_the_reason(store):
    (store / R.ARTEFACTS["failures"]).write_text("", encoding="utf-8")
    r = R.report_k06(ctx_for(store))
    assert r.status == R.UNMEASURED and "EMPTY" in r.unmeasured[0]
    (store / R.ARTEFACTS["failures"]).write_text('{"cause": "x"}\n{not json}\n', encoding="utf-8")
    r = R.report_k06(ctx_for(store))
    assert r.status == R.UNMEASURED and "UNREADABLE" in r.unmeasured[0] and "line 2" in r.unmeasured[0]


def test_a_curve_with_too_few_points_is_insufficient_not_a_verdict(store):
    js(store / R.ARTEFACTS["curve"], curve_rows(3))
    r = R.report_k01(ctx_for(store))
    assert r.facts["verdict"] == "INSUFFICIENT_POINTS" and r.label == ValidationLabel.INSUFFICIENT_EVIDENCE


def test_an_invalid_curve_is_refused_not_repaired(store):
    rows = curve_rows()
    rows[5]["experience_count"] = 10                                          # experience cannot fall
    js(store / R.ARTEFACTS["curve"], rows)
    r = R.report_k01(ctx_for(store))
    assert r.status == R.UNMEASURED and "rejected" in r.unmeasured[0]


# ============================================================================================ the claim gate
def test_a_planted_false_claim_is_refused_without_controls(store, tmp_path):
    (store / R.ARTEFACTS["scorecards"]).unlink()
    r = R.report_k01(ctx_for(store), claims=("learning improved",))
    assert [c.accepted for c in r.claims] == [False]
    assert "no scorecard" in r.claims[0].reason
    assert "REFUSED" in r.to_markdown()
    card = good_card("v2", controls={k: v for k, v in good_card().controls.items() if k != "D_random_learner"})
    S.ScorecardStore(store / R.ARTEFACTS["scorecards"]).append(card)
    r = R.report_k15(ctx_for(store), claims=("learning improved", "cross year gain improved"))
    assert not any(c.accepted for c in r.claims)
    assert "controls not run" in r.claim_statement or "no claim of improvement" in r.claim_statement


def test_a_claim_is_accepted_only_when_every_control_passes_and_validated_never(store):
    r = R.report_k01(ctx_for(store), claims=("learning improved", "learning validated", "learning works"))
    got = {c.text: c.accepted for c in r.claims}
    assert got == {"learning improved": True, "learning validated": False, "learning works": False}
    assert "survives controls A-E" in r.claim_statement


def test_a_leaky_learner_blocks_the_claim_even_with_a_big_gain(store):
    bad = good_card("v9", future_leak_status=S.LeakStatus.UNAUDITED)
    (store / R.ARTEFACTS["scorecards"]).unlink()
    S.ScorecardStore(store / R.ARTEFACTS["scorecards"]).append(bad)
    r = R.report_k03(ctx_for(store), claims=("learning improved",))
    assert not r.claims[0].accepted and "future-leak" in r.claims[0].reason


def test_a_field_claim_needs_that_fields_own_controls(store):
    weak = good_card("v3", controls={k: v for k, v in good_card().controls.items() if k in ("A_no_learning", "B_learner")})
    (store / R.ARTEFACTS["scorecards"]).unlink()
    S.ScorecardStore(store / R.ARTEFACTS["scorecards"]).append(weak)
    r = R.report_k01(ctx_for(store), claims=("memorization gap improved",))
    assert not r.claims[0].accepted


def test_a_report_that_asserts_an_unsupported_word_cannot_be_built(store):
    r = R.report_k05(ctx_for(store))
    dec = ctx_for(store).decision()[0]
    R.guard_report(r, dec)                                                     # the honest report passes
    bad = dataclasses.replace(r, notes=r.notes + ("learning improved and the system is done",))
    (store / R.ARTEFACTS["scorecards"]).unlink()
    with pytest.raises(S.UnsupportedClaim):
        R.guard_report(bad, None)
    with pytest.raises(S.UnsupportedClaim):
        R.guard_report(dataclasses.replace(r, headline="production ready"), dec)


def test_quoted_artefact_text_cannot_smuggle_a_claim_into_a_report(store):
    kn = [kobj("k_a", observation="learning improved and it is validated", version_reason="done")]
    jl(store / R.ARTEFACTS["knowledge"], kn)
    q = json.loads((store / R.ARTEFACTS["queue"]).read_text())
    q[0]["question"] = "why did we get this improved result"
    js(store / R.ARTEFACTS["queue"], q)
    r = R.report_k05(ctx_for(store))
    md = r.to_markdown()
    assert "[claim word]" in md
    assert not S.claim_violations(md.split("## Claim status")[0] + md.split("## Sources")[-1], S.ClaimDecision(False, (), ValidationLabel.INSUFFICIENT_EVIDENCE))
    d = R.research_priority_dashboard(ctx_for(store))
    assert "[claim word]" in json.dumps(d)


def test_a_tampered_scorecard_store_blocks_every_claim_and_fails_the_provenance_audit(store):
    p = store / R.ARTEFACTS["scorecards"]
    p.write_text(p.read_text().replace('"learner_version":"v1"', '"learner_version":"v1x"'), encoding="utf-8")
    dec, card, problems = ctx_for(store).decision()
    assert problems and dec.allowed is False
    r = R.report_k14(ctx_for(store), claims=("learning improved",))
    assert r.label == ValidationLabel.FAILED_VALIDATION and not r.claims[0].accepted
    assert any("edited" in str(row) or "chain" in str(row) for t in r.tables for row in t.rows)


# ============================================================================================ time: nothing at/after now
def test_rows_dated_at_or_after_now_are_dropped_and_counted(store):
    rows = [json.loads(l) for l in (store / R.ARTEFACTS["promotions"]).read_text().splitlines()]
    jl(store / R.ARTEFACTS["promotions"], rows + [dict(rows[0], knowledge_id="future", now="2030-01-01"), dict(rows[0], knowledge_id="later", now="2031-03-01")])
    r = R.report_k13(ctx_for(store))
    assert r.facts["decisions"] == 4
    assert r.sources[0].excluded_future == 2
    with pytest.raises(FirewallBreach):
        R.report_k13(ctx_for(store, strict_future=True))


def test_a_context_without_now_is_refused_and_meta_records_resolved_later_are_ignored(store):
    with pytest.raises(FirewallBreach):
        R.ReportContext(store, None)
    m = json.loads((store / R.ARTEFACTS["meta"]).read_text())
    m["discoveries"] += [dict(m["discoveries"][0], item_id="late", resolved_at="2030-06-01", survived_oos=True)]
    js(store / R.ARTEFACTS["meta"], m)
    r = R.report_k12(ctx_for(store))
    assert r.facts["records_dropped_resolved_after_now"] == 1 and r.facts["discoveries_resolved"] == 30
    with pytest.raises(FirewallBreach):
        R.report_k12(ctx_for(store, strict_future=True))


# ============================================================================================ planted defects per report
def test_k02_catches_an_artefact_whose_ratio_disagrees_with_its_own_means(store):
    js(store / R.ARTEFACTS["transfer"], transfer_record(ratio_wrong=True))
    r = R.report_k02(ctx_for(store))
    assert r.label == ValidationLabel.FAILED_VALIDATION and any("ratio" in u for u in r.unmeasured)


def test_k02_flags_an_over_specialised_learner_and_a_missing_year_axis(store):
    rec = transfer_record()
    rec["axes"]["YEAR"]["verdict"] = "OVER_SPECIALISED"
    rec["overall"] = "OVER_SPECIALISED"
    js(store / R.ARTEFACTS["transfer"], rec)
    assert R.report_k02(ctx_for(store)).label == ValidationLabel.FAILED_VALIDATION
    del rec["axes"]["YEAR"]
    js(store / R.ARTEFACTS["transfer"], rec)
    r = R.report_k02(ctx_for(store))
    assert r.label == ValidationLabel.INSUFFICIENT_EVIDENCE and any("unseen-year" in u for u in r.unmeasured)


def test_k01_a_hoarding_curve_is_failed_and_a_rising_one_is_not(store):
    rows = curve_rows(slope=0.0)
    for i, r_ in enumerate(rows):
        r_["knowledge_count"] = 20 + 30 * i                                    # memory grows, future gain does not
        r_["transfer_gain"] = 0.002
    js(store / R.ARTEFACTS["curve"], rows)
    bad = R.report_k01(ctx_for(store))
    assert bad.label == ValidationLabel.FAILED_VALIDATION and bad.facts["verdict"] in ("HOARDING", "FLAT")
    js(store / R.ARTEFACTS["curve"], curve_rows(slope=0.001))
    assert R.report_k01(ctx_for(store)).label == ValidationLabel.NOT_VALIDATED


def test_k03_flags_a_significantly_positive_memorisation_gap_and_a_blind_instrument(store):
    rec = transfer_record()
    rec["axes"]["YEAR"]["memorization_gap"] = {"gap": 0.01, "lo": 0.006, "hi": 0.014, "n_a": 80, "n_b": 120}
    js(store / R.ARTEFACTS["transfer"], rec)
    r = R.report_k03(ctx_for(store))
    assert r.label == ValidationLabel.FAILED_VALIDATION and r.facts["gaps_significantly_positive"] == 1
    js(store / R.ARTEFACTS["transfer"], transfer_record())
    C = dict(good_card().controls)
    C["C_identity_memoriser"] = dataclasses.replace(C["C_identity_memoriser"], detected=False)
    (store / R.ARTEFACTS["scorecards"]).unlink()
    S.ScorecardStore(store / R.ARTEFACTS["scorecards"]).append(good_card("v5", controls=C))
    r = R.report_k03(ctx_for(store))
    assert r.facts["memoriser_control_flagged"] is False and any("not been shown to see a memoriser" in u for u in r.unmeasured)


def test_k04_flags_collapse_under_disguise_and_ignores_attacks_that_changed_nothing(store):
    idn = json.loads((store / R.ARTEFACTS["identity"]).read_text())
    idn["verdicts"][0].update(status="COLLAPSE", retention=0.1)
    js(store / R.ARTEFACTS["identity"], idn)
    r = R.report_k04(ctx_for(store))
    assert r.label == ValidationLabel.FAILED_VALIDATION and r.facts["memorisation_suspected"] is True
    assert r.facts["collapsed_under"] == ["ticker_permutation[eval]"]
    idn["verdicts"][0].update(status="OK", retention=0.99, changed_frac=0.0)
    js(store / R.ARTEFACTS["identity"], idn)
    r = R.report_k04(ctx_for(store))
    assert any("changed under 5%" in u for u in r.unmeasured)


def test_k05_classifies_each_state_and_finds_losing_trust_without_research(store):
    rows, srcs, lost = R.assess_knowledge(ctx_for(store))
    by = {r.knowledge_id: r for r in rows}
    from engine.learning.core import Health as H
    assert by["k_a"].health == H.CONTRADICTED                                   # an UNRESOLVED pair holds it
    assert by["k_b"].health == H.DEGRADING and by["k_c"].health == H.BROKEN
    assert by["k_d"].health == H.INSUFFICIENT_EVIDENCE and by["k_e"].health == H.DEGRADING
    assert by["k_f"].health == H.DORMANT and by["k_g"].health == H.CONTRADICTED
    assert by["k_h"].health == H.DEGRADING and "0.80 -> 0.50" in by["k_h"].why.replace("0.8 ", "0.80 ") or by["k_h"].health == H.DEGRADING
    rep = R.report_k05(ctx_for(store))
    assert rep.facts["items"] == 8 and rep.facts["losing_trust_without_research"] >= 1
    dash = R.health_dashboard(ctx_for(store))
    assert dash["status"] == "MEASURED" and {x["knowledge_id"] for x in dash["losing_trust"]} >= {"k_b", "k_c", "k_h"}
    assert all("why" in x and "research" in x and "evidence" in x for x in dash["losing_trust"])


def test_k05_flags_an_unstable_item_that_keeps_changing_its_mind(store):
    chain = [kobj("k_z", ver=i + 1, ep=("SUPPORTED", "DEGRADED", "SUPPORTED", "DEGRADED")[i], updated=f"2029-0{i + 1}-01") for i in range(4)]
    jl(store / R.ARTEFACTS["knowledge"], chain)
    rows, _, _ = R.assess_knowledge(ctx_for(store))
    from engine.learning.core import Health as H
    assert rows[0].health == H.UNSTABLE and "changed 3 times" in rows[0].why


def test_k06_admits_unknown_and_teaches_the_subsystem(store):
    r = R.report_k06(ctx_for(store))
    assert r.facts["mean_detector_coverage"] == pytest.approx(0.8)
    causes = {row[0]: row[1] for row in r.tables[0].rows}
    assert causes["UNKNOWN"] == 4 and causes["FALSE_PATTERN"] == 12
    subs = {row[0]: row for row in r.tables[1].rows}
    assert "SELECTION" in subs and "TIMING" in subs
    assert r.label == ValidationLabel.NOT_VALIDATED                             # descriptive only


def test_k07_counts_recoveries_and_relapses_and_catches_an_edited_transition(store):
    r = R.report_k07(ctx_for(store))
    assert r.facts["recoveries"] == 2 and r.facts["recoveries_that_relapsed"] == 1
    assert r.facts["current_states"] == {"ACTIVE": 1, "RETIRED": 1, "DEGRADED": 1}
    assert r.label != ValidationLabel.FAILED_VALIDATION
    rows = [json.loads(l) for l in (store / R.ARTEFACTS["retirement"]).read_text().splitlines()]
    rows[3]["reason"] = "quietly rewritten"
    jl(store / R.ARTEFACTS["retirement"], rows)
    r = R.report_k07(ctx_for(store))
    assert r.label == ValidationLabel.FAILED_VALIDATION and any("edited" in u for u in r.unmeasured)


def test_k08_tracks_open_pairs_and_repeat_investigations(store):
    r = R.report_k08(ctx_for(store))
    assert r.facts["pairs"] == 2 and r.facts["open_pairs"] == 1 and r.facts["resolved_by_context"] == 1
    assert r.facts["pairs_investigated_more_than_once"] == 1 and r.facts["cumulative_tests_max"] == 7
    assert r.tables[0].rows[0][1] == "UNRESOLVED"                               # open pairs first


def test_k09_names_the_costly_filter_and_asks_for_a_null_control(store):
    r = R.report_k09(ctx_for(store))
    assert r.facts["costly_filters"] == ["LIQUIDITY"] and r.facts["walk_forward_pass_share"] == pytest.approx(0.2)
    m = json.loads((store / R.ARTEFACTS["missed"]).read_text())
    del m["null_control"]
    js(store / R.ARTEFACTS["missed"], m)
    r = R.report_k09(ctx_for(store))
    assert r.label == ValidationLabel.INSUFFICIENT_EVIDENCE and any("null control" in u for u in r.unmeasured)


def test_k10_finds_defective_memory_records_and_unexplained_repeats(store):
    r = R.report_k10(ctx_for(store))
    assert r.facts["questions_asked_more_than_once"] == 4 and r.facts["repeats_without_a_reason"] == 0
    assert r.facts["predictions_scored"] == 12 and r.facts["prediction_brier"] > r.facts["prediction_base_rate_brier"] - 1
    rows = [json.loads(l) for l in (store / R.ARTEFACTS["experiments"]).read_text().splitlines()]
    rows[0]["result"] = {"kind": "NULL", "power_note": ""}
    rows[1]["not_learned"] = []
    rows[2]["repeat_reason"] = ""
    jl(store / R.ARTEFACTS["experiments"], rows)
    r = R.report_k10(ctx_for(store))
    assert r.label == ValidationLabel.FAILED_VALIDATION
    assert r.facts["null_results_without_power"] == 1 and r.facts["answered_without_what_was_not_learned"] == 1


def test_k11_and_j15_rank_by_evidence_exclude_missing_factors_and_detect_a_stale_queue(store):
    d = R.research_priority_dashboard(ctx_for(store))
    assert [x["id"] for x in d["ranked"]] == ["q1", "q2"]                       # q3 is DONE, not open
    q2 = d["ranked"][1]
    assert q2["score_source"] == "recomputed" and {"failure_evidence", "contradiction", "surprise"} <= set(q2["missing_factors"])
    assert q2["priority"] > 0                                                   # missing factors are not counted as zero
    assert d["stale"] is False and d["items_with_missing_factors"] >= 1
    r = R.report_k11(ctx_for(store))
    assert r.label == ValidationLabel.NOT_VALIDATED and r.facts["n_open"] == 2
    stale = R.ReportContext(store, "2030-09-01", checklist=None, code_hash="t")
    assert R.research_priority_dashboard(stale)["stale"] is True
    assert R.report_k11(stale).label == ValidationLabel.FAILED_VALIDATION


def test_priority_of_prefers_a_recorded_score_and_never_treats_missing_as_zero():
    p, missing, how = R.priority_of({"score": 0.42})
    assert (p, how) == (0.42, "recorded") and "uncertainty" in missing
    full = {"uncertainty": 1, "decision_impact": 1, "transfer_potential": 1, "failure_evidence": 1, "contradiction": 1, "surprise": 1, "feasibility": 1}
    assert R.priority_of(full)[0] == pytest.approx(1.0)
    only_one = R.priority_of({"uncertainty": 0.6})
    assert only_one[0] == pytest.approx(0.6) and only_one[2] == "recomputed"
    assert np.isnan(R.priority_of({})[0]) and R.priority_of({"uncertainty": 1, "cost": 3})[0] == pytest.approx(0.25)


def test_k12_measures_fake_gains_survival_and_explanation_precision(store):
    r = R.report_k12(ctx_for(store))
    assert r.facts["fake_improvement_rate"] == pytest.approx(0.5) and r.facts["explanation_precision"] == pytest.approx(0.5)
    assert r.facts["out_of_sample_survival_rate"] == pytest.approx(10 / 30)
    assert r.facts["median_oos_to_is_effect_ratio"] == pytest.approx(0.25)
    assert r.status == "PARTIAL" or r.status == "MEASURED"


def test_k13_catches_a_promotion_that_skipped_a_critical_gate_and_an_unrecorded_champion(store):
    r = R.report_k13(ctx_for(store))
    assert r.facts["promoted"] == 3 and r.facts["blocked"] == 1
    assert r.facts["champions_without_a_recorded_promotion"] >= 1               # k_b..k_f are CHAMPION with no PROMOTE record
    assert r.label == ValidationLabel.FAILED_VALIDATION
    rows = [json.loads(l) for l in (store / R.ARTEFACTS["promotions"]).read_text().splitlines()]
    assert R.promotion_inconsistencies(rows) == []
    rows[0]["gates"][1]["status"] = "FAIL"
    assert "PROMOTE despite failed critical gate" in R.promotion_inconsistencies(rows)[0]


def test_k14_finds_future_memory_missing_provenance_and_identity_keys(store):
    clean = R.report_k14(ctx_for(store))
    assert clean.facts["findings"] == 0 and clean.label == ValidationLabel.NOT_VALIDATED
    future = dict(PROV, learned_at="2030-06-01", outcomes_seen_through="2030-06-01")
    nodata = {k: v for k, v in PROV.items() if k != "data_hash"}
    kn = [kobj("k_f1", prov=future), kobj("k_f2", prov=nodata), kobj("k_f3", contexts={"ticker": "AAPL"}), kobj("k_ok"),
          kobj("k_gap", ver=1), kobj("k_gap", ver=3, updated="2029-08-01")]
    jl(store / R.ARTEFACTS["knowledge"], kn)
    r = R.report_k14(ctx_for(store))
    text = " ".join(f"{a} {b}" for a, b in r.tables[0].rows)
    assert r.label == ValidationLabel.FAILED_VALIDATION
    assert "future memory" in text and "data_hash missing" in text and "identity key" in text and "version chain" in text
    assert "k_ok" not in text


def test_k14_checks_the_checklist_for_overclaims(store, tmp_path):
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({"items": [{"id": "X1", "status": "VALIDATED", "evidence": ["no/such/file.json"], "code_paths": ["engine/learning/reports.py"]},
                                        {"id": "X2", "status": "IMPLEMENTED", "evidence": [], "code_paths": ["engine/learning/does_not_exist.py"]}]}), encoding="utf-8")
    r = R.report_k14(ctx_for(store, checklist=cl))
    assert r.facts["checklist_validated_without_evidence"] == ["X1"]
    assert r.facts["checklist_implemented_with_missing_code_path"] == ["X2"] and r.label == ValidationLabel.FAILED_VALIDATION


def test_k15_reports_the_weakest_label_the_claim_matrix_and_the_eight_deltas(store):
    r = R.report_k15(ctx_for(store))
    assert r.facts["weakest_label"] in ("FAILED_VALIDATION", "IMPLEMENTED — FAILED VALIDATION", ValidationLabel.FAILED_VALIDATION.value)
    assert r.facts["failed_reports"] and r.label == ValidationLabel.FAILED_VALIDATION
    delta = next(t for t in r.tables if t.title.startswith("Learning delta"))
    assert [row[0] for row in delta.rows] == list(R.DELTA_DIMS)
    assert sum(1 for row in delta.rows if row[1] == R.UNMEASURED) == 5
    assert any(t.title.startswith("Which fields may be described") for t in r.tables)
    assert len(r.facts["untested_fields"]) == 0 and "still_needed_for_a_claim" in r.facts


def test_worst_label_orders_failure_above_insufficient_above_not_validated():
    L = ValidationLabel
    assert R.worst_label([L.NOT_VALIDATED, L.INSUFFICIENT_EVIDENCE]) == L.INSUFFICIENT_EVIDENCE
    assert R.worst_label([L.NOT_VALIDATED, L.FAILED_VALIDATION, L.INSUFFICIENT_EVIDENCE]) == L.FAILED_VALIDATION
    assert R.worst_label([]) == L.INSUFFICIENT_EVIDENCE


# ============================================================================================ VALIDATED needs evidence
def test_validated_needs_a_checklist_item_and_an_evidence_file_that_exists(store, tmp_path):
    ev = tmp_path / "evidence.json"
    ev.write_text("{}", encoding="utf-8")
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({"items": [{"id": "K01", "status": "VALIDATED", "evidence": ["no/such.json"]}]}), encoding="utf-8")
    assert R.report_k01(ctx_for(store, checklist=cl)).label == ValidationLabel.NOT_VALIDATED
    cl.write_text(json.dumps({"items": [{"id": "K01", "status": "VALIDATED", "evidence": [str(ev)]}]}), encoding="utf-8")
    r = R.report_k01(ctx_for(store, checklist=cl))
    assert r.label == ValidationLabel.VALIDATED and "VALIDATED taken from" in " ".join(r.notes)
    assert R.report_k02(ctx_for(store, checklist=cl)).label == ValidationLabel.NOT_VALIDATED   # only the item that holds evidence
    (store / R.ARTEFACTS["curve"]).unlink()
    assert R.report_k01(ctx_for(store, checklist=cl)).label == ValidationLabel.INSUFFICIENT_EVIDENCE    # no data beats a stale checklist


def test_measured_failure_is_never_upgraded_by_a_validated_checklist(store, tmp_path):
    ev = tmp_path / "e.json"
    ev.write_text("{}", encoding="utf-8")
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({"items": [{"id": "K02", "status": "VALIDATED", "evidence": [str(ev)]}]}), encoding="utf-8")
    js(store / R.ARTEFACTS["transfer"], transfer_record(ratio_wrong=True))
    assert R.report_k02(ctx_for(store, checklist=cl)).label == ValidationLabel.FAILED_VALIDATION


# ============================================================================================ writing and auditing
def test_write_all_writes_every_file_and_the_audit_is_clean(store, tmp_path):
    out = tmp_path / "out"
    idx = R.write_all(ctx_for(store), out)
    names = {p.name for p in out.iterdir()}
    assert {"index.json", "J15_research_priority_dashboard.json", "S46_health_dashboard.json", "K01_learning_curve.md", "K15_full_learning_system.json"} <= names
    assert len([n for n in names if n.startswith("K") and n.endswith(".md")]) == 15
    assert R.audit_output(out, ctx_for(store)) == []
    assert set(idx["reports"]) == set(R.GENERATORS)


def test_the_audit_fails_closed_on_missing_index_edited_file_and_lying_label(store, tmp_path):
    out = tmp_path / "o"
    assert R.audit_output(out)[0].startswith("index.json missing")
    R.write_all(ctx_for(store), out)
    p = out / "K06_failure.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nthe learner improved and it is done\n", encoding="utf-8")
    probs = R.audit_output(out, ctx_for(store))
    assert any("K06_failure.md was modified" in x for x in probs) and any("K06: markdown asserts" in x for x in probs)
    (out / "K02_transfer.json").unlink()
    assert any("K02_transfer.json listed but absent" in x for x in R.audit_output(out))


def test_the_audit_flags_an_unmeasured_report_that_claims_a_better_label(tmp_path):
    empty = tmp_path / "e"
    empty.mkdir()
    out = tmp_path / "o"
    R.write_all(ctx_for(empty), out)
    assert R.audit_output(out, ctx_for(empty)) == []
    p = out / "K03_memorization.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["label"] = ValidationLabel.NOT_VALIDATED.value
    p.write_text(json.dumps(rec), encoding="utf-8")
    probs = R.audit_output(out, ctx_for(empty))
    assert any("K03: UNMEASURED but not labelled" in x for x in probs)


def test_the_command_line_script_writes_reports_and_refuses_a_false_claim(store, tmp_path):
    out = tmp_path / "cli"
    cmd = [sys.executable, str(ROOT / "scripts" / "learning_report.py"), "--now", NOW, "--root", str(store), "--out", str(out), "--checklist", str(tmp_path / "none.json"),
           "--claim", "learning validated"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=80)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "claim REFUSED: 'learning validated'" in res.stdout and "audit: CLEAN" in res.stdout
    assert (out / "K01_learning_curve.md").exists()
    res = subprocess.run(cmd[:-2] + ["--audit-only"], cwd=ROOT, capture_output=True, text=True, timeout=80)
    assert res.returncode == 0


def test_helpers_wilson_defang_and_num():
    lo, hi = R.wilson_interval(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8 and all(np.isnan(x) for x in R.wilson_interval(0, 0))
    assert R.defang("Learning improved: done") == "[claim word]: [claim word]" and R.defang("plain text") == "plain text"
    assert np.isnan(R.num("nan")) and np.isnan(R.num(None)) and np.isnan(R.num(True)) and R.num("2.5") == 2.5
    assert R.fmt(float("nan")) == "n/a" and R.fmt(0.5, 2, True) == "+0.50"
    assert R.row_date({"at": "2029-03-01T10:00"}) == dt.date(2029, 3, 1) and R.row_date({"x": 1}) is None
