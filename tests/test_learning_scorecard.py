"""Learning scorecard and claim gate, predictive-vs-portfolio value, learning curve and multi-dimensional learning delta (contract
C62 sections 25, 34, 47, 48, 65; modules engine/learning/scorecard.py, portfolio_value.py, learning_curve.py). Each mechanism is
checked against a planted case it must catch, plus the empty case. Synthetic data only. Proves nothing about the real archive:
IMPLEMENTED - NOT VALIDATED."""
import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from engine import objective as O
from engine.learning import learning_curve as LC
from engine.learning import portfolio_value as PV
from engine.learning import scorecard as S
from engine.learning import transfer as T
from engine.learning import transfer_score as TS
from engine.learning.core import FirewallBreach, ValidationLabel

NOW = pd.Timestamp("2030-01-01")


# =============================================================================================== scorecard helpers
def M(v, lo=None, hi=None, n=100):
    return S.Measured(v, v - 0.001 if lo is None else lo, v + 0.001 if hi is None else hi, n, S.MStatus.MEASURED)


def good_card(**kw):
    rng = np.random.default_rng(0)
    controls = {
        "A_no_learning": S.ControlResult("A_no_learning", M(0.0, -0.001, 0.001)),
        "B_learner": S.ControlResult("B_learner", M(0.01, 0.008, 0.012), gains=tuple(rng.normal(0.01, 0.01, 40))),
        "C_identity_memoriser": S.ControlResult("C_identity_memoriser", M(0.0, -0.001, 0.001), detected=True, gains=tuple(rng.normal(0.0, 0.01, 40))),
        "D_random_learner": S.ControlResult("D_random_learner", M(0.0005, -0.001, 0.002)),
        "E_leaky_learner": S.ControlResult("E_leaky_learner", M(0.03, 0.028, 0.032), detected=True),
    }
    base = dict(learner_version="v1", now=dt.date(2029, 12, 31), code_hash="abc123", seed=1, baseline_performance=M(0.001), post_learning_performance=M(0.011),
                learning_gain=M(0.01, 0.008, 0.012), same_year_gain=M(0.011), cross_year_gain=M(0.009, 0.007, 0.011), cross_regime_gain=M(0.008),
                cross_stock_gain=M(0.009), transfer_ratio=M(0.9, 0.7, 1.1), risk_change=M(0.002, -0.001, 0.005), drawdown_change=M(0.003, -0.001, 0.007),
                band_share=M(0.4, 0.35, 0.45), movement_performance=M(0.3), direction_performance=M(0.02), mover_performance=M(0.05), calibration=M(0.03),
                memorization_gap=M(0.0, -0.001, 0.001), identity_gap=M(0.0, -0.001, 0.001), future_leak_status=S.LeakStatus.CLEAN, stability=M(0.8),
                compute_cost=S.ComputeCost(1.0, 1.2, 100.0, 3, 3), controls=controls)
    base.update(kw)
    return S.LearningScorecard(**base)


def mutate(card, **kw):
    return dataclasses.replace(card, **kw)


def with_control(card, name, **kw):
    c = dict(card.controls)
    c[name] = dataclasses.replace(c[name], **kw)
    return mutate(card, controls=c)


# =============================================================================================== Measured and helpers
def test_measured_untested_is_not_zero_and_significance_needs_an_interval():
    u = S.Measured.untested("never run")
    assert not u.measured and np.isnan(u.value) and not u.significantly_positive and not u.significantly_negative
    assert S.Measured.undefined("ratio undefined").status == S.MStatus.NOT_APPLICABLE
    assert M(0.01, 0.005, 0.015).significantly_positive and M(-0.01, -0.015, -0.005).significantly_negative
    assert not S.Measured.point(0.5, 10).significantly_positive                 # a point estimate with no interval proves nothing
    assert S.Measured.point(float("nan")).status == S.MStatus.UNTESTED
    assert S.Measured.from_boot(TS.cluster_bootstrap_mean([], None)).status == S.MStatus.UNTESTED
    assert S.Measured.from_gap(None).status == S.MStatus.UNTESTED


def test_leak_status_is_never_clean_by_default():
    F = type("F", (), {})
    fail, warn = F(), F()
    fail.severity, warn.severity = "fail", "warn"
    assert S.leak_status_from([warn]) == S.LeakStatus.CLEAN
    assert S.leak_status_from([warn, fail]) == S.LeakStatus.LEAK_DETECTED
    assert S.leak_status_from([{"severity": "fail"}]) == S.LeakStatus.LEAK_DETECTED and S.leak_status_from([{"status": "warn"}]) == S.LeakStatus.CLEAN
    assert S.leak_status_from(None) == S.LeakStatus.UNAUDITED
    assert S.leak_status_from([], audited=False) == S.LeakStatus.UNAUDITED
    assert S.leak_status_from([], breach=FirewallBreach("x")) == S.LeakStatus.LEAK_DETECTED


def test_compute_meter_measures_and_cost_rejects_negatives():
    with S.ComputeMeter() as m:
        sum(i * i for i in range(20000))
    c = m.cost(n_fits=2, n_evaluations=5)
    assert c.measured and c.cpu_seconds >= 0 and c.wall_seconds >= 0 and c.n_fits == 2 and not c.check()
    assert S.ComputeCost(cpu_seconds=-1.0).check()
    assert not S.ComputeCost().measured


def test_calibration_measure_sees_overconfidence_and_refuses_too_few_samples():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 600)
    calibrated = (rng.random(600) < p).astype(float)
    overconfident = (rng.random(600) < 0.5).astype(float)                  # says 5%..95%, is 50% right
    good, bad = S.calibration_measure(p, calibrated, n_boot=80), S.calibration_measure(p, overconfident, n_boot=80)
    assert good.measured and bad.measured and bad.value > good.value + 0.1
    assert S.calibration_measure(p[:10], calibrated[:10]).status == S.MStatus.UNTESTED
    with pytest.raises(ValueError):
        S.calibration_measure([1.5, 0.2], [1, 0])


# =============================================================================================== the claim gate
def test_a_card_with_every_control_and_a_real_transferring_gain_may_claim_improvement():
    card = good_card()
    d = S.gate_improvement_claim(card)
    assert d.allowed and not d.blockers and d.label == ValidationLabel.NOT_VALIDATED
    assert "NOT VALIDATED" in d.statement(card) and not card.check()
    assert len(card.card_id) == 16 and card.card_id == good_card().card_id


@pytest.mark.parametrize("mutation,check", [
    (lambda c: mutate(c, controls={k: v for k, v in c.controls.items() if k != "E_leaky_learner"}), "controls_present"),
    (lambda c: with_control(c, "E_leaky_learner", detected=False), "leak_seen"),
    (lambda c: with_control(c, "C_identity_memoriser", detected=False), "memoriser_seen"),
    (lambda c: with_control(c, "A_no_learning", gain=M(0.01, 0.008, 0.012)), "no_learning_flat"),
    (lambda c: with_control(c, "D_random_learner", gain=M(0.02, 0.015, 0.025)), "beats_luck_floor"),
    (lambda c: mutate(c, future_leak_status=S.LeakStatus.UNAUDITED), "learner_leak_clean"),
    (lambda c: mutate(c, future_leak_status=S.LeakStatus.LEAK_DETECTED), "learner_leak_clean"),
    (lambda c: mutate(c, learning_gain=M(0.002, -0.003, 0.007)), "learning_gain"),
    (lambda c: mutate(c, memorization_gap=M(0.01, 0.005, 0.015)), "not_memoriser"),
    (lambda c: mutate(c, identity_gap=M(0.01, 0.005, 0.015)), "not_memoriser"),
    (lambda c: mutate(c, identity_gap=S.Measured.untested()), "not_memoriser"),
    (lambda c: mutate(c, cross_year_gain=S.Measured.untested()), "cross_year_transfer"),
    (lambda c: mutate(c, cross_year_gain=M(0.001, -0.002, 0.004)), "cross_year_transfer"),
    (lambda c: mutate(c, transfer_ratio=M(0.1, 0.05, 0.15)), "cross_year_transfer"),
    (lambda c: mutate(c, transfer_ratio=S.Measured.undefined("TRANSFER_ONLY")), "cross_year_transfer"),
    (lambda c: mutate(c, cross_regime_gain=S.Measured.untested(), cross_stock_gain=S.Measured.untested()), "another_context"),
    (lambda c: mutate(c, cross_stock_gain=M(-0.01, -0.015, -0.005)), "another_context"),
    (lambda c: mutate(c, risk_change=M(-0.02, -0.03, -0.01)), "risk_not_worse"),
    (lambda c: mutate(c, drawdown_change=M(-0.05, -0.07, -0.03)), "risk_not_worse"),
    (lambda c: mutate(c, risk_change=S.Measured.untested()), "tier_measured"),
    (lambda c: mutate(c, band_share=S.Measured.untested()), "tier_measured"),
    (lambda c: mutate(c, stability=M(0.1)), "stability"),
    (lambda c: mutate(c, stability=S.Measured.untested()), "stability"),
    (lambda c: mutate(c, code_hash=""), "scorecard_valid"),
    (lambda c: mutate(c, seed=None), "scorecard_valid"),
])
def test_every_planted_defect_blocks_the_claim_through_its_own_check(mutation, check):
    d = S.gate_improvement_claim(mutation(good_card()))
    failed = {c.name for c in d.checks if c.blocking and not c.ok}
    assert not d.allowed and check in failed


def test_a_learner_whose_gains_mirror_the_identity_memoriser_is_blocked():
    card = good_card()
    mem = card.controls["C_identity_memoriser"]
    twin = dataclasses.replace(card.controls["B_learner"], gains=tuple(np.asarray(mem.gains) * 1.0 + np.random.default_rng(3).normal(0, 1e-4, len(mem.gains))))
    d = S.gate_improvement_claim(mutate(card, controls={**card.controls, "B_learner": twin}))
    assert not d.allowed and any("resemble the identity memoriser" in b for b in d.blockers)


def test_missing_measurements_are_insufficient_evidence_while_measured_damage_is_failed_validation():
    missing = S.gate_improvement_claim(mutate(good_card(), risk_change=S.Measured.untested()))
    damaged = S.gate_improvement_claim(mutate(good_card(), risk_change=M(-0.02, -0.03, -0.01)))
    assert missing.label == ValidationLabel.INSUFFICIENT_EVIDENCE and damaged.label == ValidationLabel.FAILED_VALIDATION
    assert S.gate_improvement_claim(good_card()).label != ValidationLabel.VALIDATED


def test_untested_fields_are_listed_and_a_card_can_never_be_labelled_validated():
    c = mutate(good_card(), calibration=S.Measured.untested(), future_leak_status=S.LeakStatus.UNAUDITED)
    assert set(c.untested_fields()) == {"calibration", "future_leak_status"}
    bad = mutate(good_card(), label=ValidationLabel.VALIDATED)
    assert any("VALIDATED" in e for e in bad.check())
    assert not S.gate_improvement_claim(bad).allowed
    assert set(S.SECTION_47_FIELDS) <= {f.name for f in dataclasses.fields(S.LearningScorecard)}


# =============================================================================================== text guard
def test_claim_text_is_rejected_unless_the_gate_passed_and_validated_is_never_allowed():
    refused = S.gate_improvement_claim(mutate(good_card(), future_leak_status=S.LeakStatus.UNAUDITED))
    allowed = S.gate_improvement_claim(good_card())
    with pytest.raises(S.UnsupportedClaim):
        S.assert_claim_ok("Learning improved on unseen years.", refused)
    S.assert_claim_ok("Learning improved on unseen years.", allowed)
    S.assert_claim_ok("There is no improvement; the result is not validated and the work is incomplete.", refused)
    for phrase in ("The learner is validated.", "This is production ready.", "The work is complete.", "Done.", "Learning works."):
        with pytest.raises(S.UnsupportedClaim):
            S.assert_claim_ok(phrase, allowed)                                # even a passing gate cannot validate
    assert S.claim_violations("It has NOT improved.", refused) == []
    assert S.claim_violations("It improved and is validated.", refused) == ["improved", "validated"]


def test_rendered_scorecard_and_markdown_pass_their_own_claim_check_and_show_untested_honestly():
    partial = mutate(good_card(), calibration=S.Measured.untested("no data"), risk_change=S.Measured.untested("no weeks"))
    assert "UNTESTED" not in S.render_scorecard(good_card()) and "UNTESTED: no weeks" in S.render_scorecard(partial)
    for card in (good_card(), partial):
        text, md = S.render_scorecard(card), S.scorecard_markdown(card)
        for f in S.SECTION_47_FIELDS:
            assert f in text and f in md
    assert "no claim of improvement" in S.render_scorecard(mutate(good_card(), calibration=S.Measured.untested(), risk_change=S.Measured.untested()))


# =============================================================================================== leak probe and the five controls
def test_outcome_dependence_probe_catches_a_predictor_that_reads_the_answer():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"f1": rng.normal(size=80), "alt": rng.normal(size=80), "base": rng.normal(size=80)})
    legit = lambda t: (t["f1"].to_numpy() > 0).astype(float)
    leaky = lambda t: (t["alt"].to_numpy() > t["base"].to_numpy()).astype(float)
    assert not S.outcome_dependence_probe(legit, frame).leaked
    p = S.outcome_dependence_probe(leaky, frame)
    assert p.leaked and p.changed_share > 0.2
    assert S.outcome_dependence_probe(leaky, frame, seed=1) == S.outcome_dependence_probe(leaky, frame, seed=1)        # deterministic
    tiny = S.outcome_dependence_probe(leaky, frame.iloc[:2])
    assert not tiny.leaked and np.isnan(tiny.changed_share)
    assert not S.outcome_dependence_probe(legit, frame[["f1"]]).leaked


def _world(seed=0):
    return T.synthetic_transfer_units(seed=seed, n_years=6, n_tickers=30, weeks_per_year=20)


def test_run_controls_sees_the_memoriser_and_the_leak_and_finds_the_rule_clean():
    suite = S.run_controls(_world(1), T.rule_learner_fit, NOW, seed=1, n_boot=150)
    by = {c.name: c for c in suite.controls}
    assert set(by) == set(S.CONTROL_NAMES)
    assert by["A_no_learning"].gain.value == 0.0
    assert by["C_identity_memoriser"].detected is True and by["E_leaky_learner"].detected is True
    assert by["E_leaky_learner"].gain.value > by["B_learner"].gain.value                         # the leak looks spectacular
    assert by["B_learner"].gain.significantly_positive and by["D_random_learner"].gain.hi < by["B_learner"].gain.value
    assert suite.leak == S.LeakStatus.CLEAN and not suite.identity_gap.significantly_positive
    assert suite.memorization_gap.measured and not suite.memorization_gap.significantly_positive


def test_scorecard_for_a_rule_is_blocked_only_by_the_untested_portfolio_tier_and_for_bad_learners_by_their_defects():
    rule = S.scorecard_for_learner(_world(2), T.rule_learner_fit, learner_version="rule", now=NOW, code_hash="h", seed=2, n_boot=150)
    dr = S.gate_improvement_claim(rule)
    assert [c.name for c in dr.checks if c.blocking and not c.ok] == ["tier_measured"] and dr.label == ValidationLabel.INSUFFICIENT_EVIDENCE
    mem = S.scorecard_for_learner(_world(2), T.identity_memoriser_fit, learner_version="mem", now=NOW, code_hash="h", seed=2, n_boot=150)
    dm = S.gate_improvement_claim(mem)
    assert not dm.allowed and dm.label == ValidationLabel.FAILED_VALIDATION
    assert {"learning_gain", "not_memoriser"} <= {c.name for c in dm.checks if c.blocking and not c.ok}
    leaky = S.scorecard_for_learner(_world(2), S.leaky_learner_fit, learner_version="leaky", now=NOW, code_hash="h", seed=2, n_boot=150)
    assert leaky.future_leak_status == S.LeakStatus.LEAK_DETECTED
    assert "learner_leak_clean" in {c.name for c in S.gate_improvement_claim(leaky).checks if c.blocking and not c.ok}
    assert rule.compute_cost.measured and rule.controls_hash == leaky.controls_hash and not rule.check()


def test_control_code_hashes_change_when_a_control_changes():
    h = S.control_code_hashes()
    assert set(h) >= {"A_no_learning", "C_identity_memoriser", "D_random_learner", "E_leaky_learner", "probe"}
    assert h == S.control_code_hashes()


# =============================================================================================== store and comparison
def test_store_is_append_only_hash_chained_and_detects_tampering(tmp_path):
    store = S.ScorecardStore(tmp_path / "cards.jsonl")
    assert store.verify() == [] and store.load() == [] and store.latest() is None
    id1 = store.append(good_card(learner_version="v1"))
    store.append(good_card(learner_version="v2"))
    assert store.verify() == [] and [r["learner_version"] for r in store.load()] == ["v1", "v2"] and store.latest()["learner_version"] == "v2"
    assert id1 == store.load()[0]["card_id"]
    with pytest.raises(FileExistsError):
        store.append(good_card(learner_version="v1"))
    with pytest.raises(ValueError):
        store.append(good_card(learner_version="v3", code_hash=""))
    raw = (tmp_path / "cards.jsonl").read_bytes()
    (tmp_path / "cards.jsonl").write_bytes(raw.replace(b'"seed":1', b'"seed":9', 1))
    assert any("edited" in p for p in store.verify())
    (tmp_path / "cards.jsonl").write_bytes(raw.split(b"\n")[1] + b"\n")            # first line removed
    assert any("line removed" in p or "chain" in p for p in store.verify())


def test_store_refuses_new_controls_silently_and_reports_trajectories_and_regressions(tmp_path):
    store = S.ScorecardStore(tmp_path / "c.jsonl")
    store.append(good_card(learner_version="v1", controls_hash="H1"))
    with pytest.raises(ValueError):
        store.append(good_card(learner_version="v2", controls_hash="H2"))
    store.append(good_card(learner_version="v2", controls_hash="H2"), allow_new_controls=True)
    store.append(good_card(learner_version="v3", controls_hash="H2", cross_year_gain=M(0.001, 0.0005, 0.0015)), allow_new_controls=True)
    tr = store.trajectory("cross_year_gain")
    assert [t[0] for t in tr] == ["v1", "v2", "v3"] and tr[0][1] == pytest.approx(0.009)
    assert store.regression_alerts() == ["v3 is significantly worse than v2 on cross_year_gain"]
    assert store.trajectory("no_such_field") == []


def test_compare_scorecards_needs_separated_intervals_and_a_passing_gate():
    old = good_card(learner_version="old")
    better = good_card(learner_version="new", cross_year_gain=M(0.02, 0.018, 0.022), learning_gain=M(0.02, 0.018, 0.022), stability=M(0.95, 0.94, 0.96))
    c = S.compare_scorecards(old, better)
    assert c.newer_is_better and "cross_year_gain" in c.gains and not c.regressions
    worse = good_card(learner_version="new", cross_year_gain=M(0.001, 0.0005, 0.0015))
    cw = S.compare_scorecards(old, worse)
    assert not cw.newer_is_better and cw.regressions == ("cross_year_gain",) and "NOT shown to be better" in cw.statement()
    same = S.compare_scorecards(old, good_card(learner_version="new"))
    assert not same.newer_is_better and not same.gains
    blocked = S.compare_scorecards(old, mutate(better, future_leak_status=S.LeakStatus.UNAUDITED))
    assert not blocked.newer_is_better                                       # better numbers but the gate refuses: still not "better"


def test_build_scorecard_fills_only_what_its_reports_measured():
    u = _world(3)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    tr = d[d["year"].astype(int) <= 2012]
    scope = T.TrainingScope.from_units(tr, dt.date(2013, 1, 1))
    out = d.copy()
    out["learned"] = out["base"] + T.rule_learner_fit(tr)(out) * (out["alt"] - out["base"])
    rep = T.evaluate_transfer(out, scope, NOW, axes=(T.Axis.YEAR,), n_boot=120)
    pairs = [{"before": {"worst5": -0.1, "max_dd": -0.3, "in_band": 0.2}, "after": {"worst5": -0.08, "max_dd": -0.25, "in_band": 0.3}} for _ in range(5)]
    delta = LC.compute_learning_delta(pairs, transfer_deltas=[0.01] * 5)
    card = S.build_scorecard(learner_version="b", now=NOW, code_hash="h", seed=1, baseline=M(0.0), post=M(0.01), learning_gain=M(0.01), transfer=rep, delta=delta)
    assert card.cross_year_gain.measured and card.same_year_gain.measured and card.transfer_ratio.status in (S.MStatus.MEASURED, S.MStatus.NOT_APPLICABLE)
    assert not card.cross_regime_gain.measured and not card.movement_performance.measured and not card.calibration.measured
    assert card.risk_change.measured and card.drawdown_change.measured and card.band_share.measured
    assert card.transfer_verdict == rep.overall.value and "cross_regime_gain" in card.untested_fields()
    assert not S.gate_improvement_claim(card).allowed


# =============================================================================================== portfolio value: panel checks
def test_panel_checks_fail_closed_and_reject_malformed_input():
    p = PV.synthetic_value_panel(seed=0, n_dates=30)
    spec = PV.ValueSpec()
    assert len(PV.check_panel(p, spec, NOW)) == len(p)
    with pytest.raises(FirewallBreach):
        PV.check_panel(p, spec, p.index.get_level_values(0).max())          # the last outcomes mature after this "now"
    with pytest.raises(ValueError):
        PV.check_panel(p.reset_index(), spec, NOW)
    with pytest.raises(ValueError):
        PV.check_panel(pd.concat([p, p.iloc[:1]]), spec, NOW)
    bad = p.copy()
    bad.iloc[3, bad.columns.get_loc("fwd")] = np.nan
    with pytest.raises(ValueError):
        PV.check_panel(bad, spec, NOW)
    dense = PV.synthetic_value_panel(seed=0, n_dates=30, spacing_days=1)
    with pytest.raises(ValueError):
        PV.decompose_value(dense, spec, NOW)                                  # overlapping holding periods
    assert len(PV.subsample_dates(dense, 7).index.get_level_values(0).unique()) == 5


def test_per_date_statistics_match_hand_computed_values():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-06", "2020-01-13"]), list("abcdefgh")], names=["date", "ticker"])
    x = pd.Series(np.tile(np.arange(8.0), 2), index=idx)
    assert (PV.per_date_rank_corr(x, x * 3 + 1).round(9) == 1.0).all() and (PV.per_date_rank_corr(x, -x).round(9) == -1.0).all()
    assert PV.per_date_rank_corr(x, pd.Series(1.0, index=idx)).isna().all()                       # a constant side has no rank correlation
    assert PV.per_date_rank_corr(x, x, min_names=9).isna().all()
    label = pd.Series(np.tile([0, 0, 0, 0, 0, 0, 1, 1], 2), index=idx)
    assert (PV.per_date_auc(x, label) == 1.0).all() and (PV.per_date_auc(-x, label) == 0.0).all()
    assert PV.per_date_auc(x, pd.Series(0, index=idx)).isna().all()
    rel = pd.Series(np.tile([0, 0, 0, 0, 0, 0, 1, 1], 2), index=idx).astype(float)
    rng = np.random.default_rng(0)
    assert (PV.per_date_ndcg(x, rel, 2, rng) == 1.0).all() and (PV.per_date_ndcg(-x, rel, 2, rng) < 0.5).all()


def test_top_k_breaks_ties_at_random_not_by_name_order():
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-06", periods=40, freq="W-MON"), [f"N{i:02d}" for i in range(20)]], names=["date", "ticker"])
    flat = pd.Series(1.0, index=idx)
    a = PV.top_k_picks(flat, 3, np.random.default_rng(1))
    b = PV.top_k_picks(flat, 3, np.random.default_rng(1))
    c = PV.top_k_picks(flat, 3, np.random.default_rng(2))
    assert a.equals(b) and not a.equals(c)
    picked = {t for (_, t), v in a.items() if v}
    assert len(picked) > 12 and a.groupby(level=0).sum().eq(3).all()          # spread over names; never the alphabetical first three
    thin = PV.top_k_picks(flat.iloc[:5], 3, np.random.default_rng(1))
    assert not thin.any()                                                     # fewer than min_names names: no pick


def test_incremental_ic_is_zero_for_a_relabelled_copy_and_positive_for_new_information():
    p = PV.synthetic_value_panel(seed=3, n_dates=60, score_skill=0.6)
    fwd = p["fwd"]
    copy = PV.incremental_ic(p["score_base"], p["score_base"] * 2 + 5, fwd)
    assert copy.isna().all() or abs(copy.mean()) < 1e-9
    new = PV.incremental_ic(p["score_base"], p["score_new"], fwd)
    assert new.mean() > 0.3


# =============================================================================================== portfolio value: the decomposition
def _dec(seed=1, n_dates=100, n_boot=120, spec=None, **skills):
    p = PV.synthetic_value_panel(seed=seed, n_dates=n_dates, **skills)
    return PV.decompose_value(p, spec or PV.ValueSpec(), NOW, n_boot=n_boot, seed=seed)


def test_a_blind_new_signal_shows_no_reliable_gain_on_any_prediction_component():
    d = _dec()
    for n in ("predictive_effect", "movement_prediction", "ranking_value", "selection_value"):
        assert not d[n].significant_gain, n
    assert d.label == ValidationLabel.NOT_VALIDATED and set(d.components) == set(PV.COMPONENTS)


def test_planted_movement_skill_is_found_in_the_movement_ranking_and_selection_components_only():
    d = _dec(move_skill=0.8)
    assert d["movement_prediction"].significant_gain and d["ranking_value"].significant_gain and d["selection_value"].significant_gain
    assert not d["predictive_effect"].significant_gain and not d["direction_value"].significant_gain
    assert d["movement_prediction"].tier == 1 and d["direction_value"].tier == 3 and d["risk_value"].tier == 2


def test_planted_score_and_direction_skill_land_in_their_own_components():
    d = _dec(score_skill=0.6, dir_skill=0.6)
    assert d["predictive_effect"].significant_gain and d["direction_value"].significant_gain
    assert not d["movement_prediction"].significant_gain


def test_timing_is_untested_without_exposure_and_positive_when_exposure_follows_the_move():
    assert _dec()["timing_value"].status == PV.STATUS_UNTESTED
    spec = PV.ValueSpec(exposure=("exp_base", "exp_new"))
    good = _dec(n_dates=120, spec=spec, timing_skill=1.0, move_skill=0.0)
    assert good["timing_value"].status == PV.STATUS_MEASURED and good["timing_value"].delta != 0.0
    blind = _dec(n_dates=120, spec=spec, timing_skill=0.0)
    assert abs(blind["timing_value"].delta) < 1e-12


def test_prediction_gain_the_objective_rejects_is_diagnosed_as_not_reaching_the_portfolio():
    mk = lambda name, gain: PV.ComponentValue(name, PV.TIER_OF[name], 0.0, gain, gain, gain - 0.01 if gain else -0.01, gain + 0.01 if gain else 0.01, 50)
    comps = {n: mk(n, 0.0) for n in PV.COMPONENTS}
    comps["movement_prediction"] = mk("movement_prediction", 0.2)
    assert PV.Diagnosis.PREDICTIVE_NOT_PORTFOLIO in PV.diagnose(comps, False, (), ())
    assert PV.Diagnosis.PORTFOLIO_WITHOUT_PREDICTIVE in PV.diagnose({n: mk(n, 0.0) for n in PV.COMPONENTS}, True, (), ())
    assert PV.diagnose({n: mk(n, 0.0) for n in PV.COMPONENTS}, None, (), ()) == (PV.Diagnosis.CLEAN,)
    dir_up = {n: mk(n, 0.0) for n in PV.COMPONENTS}
    dir_up["direction_value"] = mk("direction_value", 0.3)
    dir_up["risk_value"] = mk("risk_value", -0.05)
    assert PV.Diagnosis.DIRECTION_CANNOT_BUY_BACK in PV.diagnose(dir_up, False, (), ())
    assert PV.Diagnosis.RISK_BOUGHT_BY_HIDING in PV.diagnose({n: mk(n, 0.0) for n in PV.COMPONENTS}, False, ("cash_hiding",), ())
    assert PV.Diagnosis.RISK_BOUGHT_BY_HIDING not in PV.diagnose({n: mk(n, 0.0) for n in PV.COMPONENTS}, False, ("cash_hiding",), ("cash_hiding",))


def test_the_diagnosis_is_consistent_with_the_objective_verdict_on_real_simulations():
    for seed in (1, 2, 3):
        d = _dec(seed=seed, move_skill=0.8, n_dates=80, n_boot=80)
        pred = [n for n in ("predictive_effect", "movement_prediction", "ranking_value", "selection_value") if d[n].significant_gain]
        assert (PV.Diagnosis.PREDICTIVE_NOT_PORTFOLIO in d.diagnoses) == (bool(pred) and d.portfolio_accept is False)
        assert d.translation()["n_predictive_gains"] == len(pred)


def test_deltas_are_reported_separately_and_the_record_is_content_addressed():
    d = _dec(move_skill=0.8, n_boot=80)
    assert set(d.deltas()) == set(PV.COMPONENTS)
    r1, r2 = d.as_record(), _dec(move_skill=0.8, n_boot=80).as_record()
    assert r1["id"] == r2["id"] and "NOT VALIDATED" in d.render()
    assert PV.explain(d) and all(isinstance(x, str) for x in PV.explain(d))
    assert not any("learning improved" in x.lower() for x in PV.explain(d))


def test_portfolio_options_run_and_bad_options_are_refused():
    p = PV.synthetic_value_panel(seed=4, n_dates=50, move_skill=0.7, score_skill=0.4, dir_skill=0.5)
    for kw in (dict(pick="both"), dict(pick="score"), dict(weighting="rank"), dict(weighting="signal"), dict(cost_mode="turnover"),
               dict(direction_mode="filter"), dict(direction_mode="sign")):
        d = PV.decompose_value(p, PV.ValueSpec(**kw), NOW, n_boot=60)
        assert d.portfolio_accept is not None
    for kw in (dict(weighting="nope"), dict(cost_mode="nope")):
        with pytest.raises(ValueError):
            PV.decompose_value(p, PV.ValueSpec(**kw), NOW, n_boot=60)
    with pytest.raises(ValueError):
        PV.decompose_value(p.drop(columns=["dir_base", "dir_new"]), PV.ValueSpec(direction_mode="sign"), NOW, n_boot=60)
    turn = PV.run_arms(p, PV.ValueSpec(cost_mode="turnover"), NOW)["new"][1]
    full = PV.run_arms(p, PV.ValueSpec(cost_mode="round_trip"), NOW)["new"][1]
    assert turn.mean_week >= full.mean_week - 1e-12                            # only the changed names pay


def test_missing_columns_make_components_untested_and_an_empty_panel_is_insufficient():
    p = PV.synthetic_value_panel(seed=5, n_dates=40).drop(columns=["move_base", "move_new", "dir_base", "dir_new"])
    d = PV.decompose_value(p, PV.ValueSpec(pick="score"), NOW, n_boot=60)
    assert d["movement_prediction"].status == PV.STATUS_UNTESTED and d["direction_value"].status == PV.STATUS_UNTESTED
    assert d["predictive_effect"].status == PV.STATUS_MEASURED
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "ticker"])
    empty = pd.DataFrame({c: pd.Series([], dtype=float) for c in ("fwd", "score_base", "score_new", "move_base", "move_new", "dir_base", "dir_new")}, index=idx)
    empty["mature"] = pd.Series([], dtype="datetime64[ns]")
    e = PV.decompose_value(empty, PV.ValueSpec(), NOW)
    assert e.label == ValidationLabel.INSUFFICIENT_EVIDENCE and e.portfolio_accept is None and "NOT RUN" in e.render()
    assert all(c.status != PV.STATUS_MEASURED for c in e.components.values())


# =============================================================================================== portfolio value: sensitivity and attribution
def test_vol_matching_scales_with_the_past_only_and_names_the_verdict():
    w = pd.Series(np.random.default_rng(0).normal(0, 0.02, 40), index=pd.date_range("2020-01-06", periods=40, freq="W-MON"))
    s1 = PV.scale_to_target(w, lookback=8)
    w2 = w.copy()
    w2.iloc[30:] *= 10                                                        # a future shock must not change earlier scales
    s2 = PV.scale_to_target(w2, lookback=8)
    assert np.allclose(s1.iloc[:30], s2.iloc[:30]) and (s1.iloc[:8] == w.iloc[:8]).all()
    p = PV.synthetic_value_panel(seed=1, n_dates=80, move_skill=0.8)
    v = PV.vol_matched_comparison(p, PV.ValueSpec(), NOW)
    assert v["verdict"] in ("EDGE_SURVIVES_VOL_MATCHING", "ADVANTAGE_IS_LEVERAGE", "NO_ADVANTAGE") and v["matched"] is not None


def test_deciles_reliability_and_direction_conviction_read_the_planted_signals():
    p = PV.synthetic_value_panel(seed=2, n_dates=80, move_skill=0.8, dir_skill=0.8)
    spec = PV.ValueSpec()
    dc = PV.decile_compare(p, spec, NOW, "move", n_boot=80)
    assert dc["new"]["monotone"] and dc["new"]["spread"].lo > 0 and not dc["base"]["monotone"]
    mr = PV.mover_reliability(p, spec, NOW)
    assert mr["spearman"] > 0.8 and mr["top_lift"] > 0.2 and len(mr["table"]) == 10
    conv = PV.direction_by_conviction(p, spec, NOW, n_boot=80)
    assert len(conv) == 5 and (conv["accuracy"] > 0.5).all()
    blind = PV.direction_by_conviction(PV.synthetic_value_panel(seed=2, n_dates=80), spec, NOW, n_boot=80)
    assert abs(blind["accuracy"].mean() - 0.5) < 0.1
    assert PV.mover_reliability(p.iloc[:10], spec, NOW)["table"].empty
    with pytest.raises(ValueError):
        PV.decile_table(p, spec, NOW, "no_such_column")


def test_random_floor_separates_skill_from_luck():
    good = PV.random_floor(PV.synthetic_value_panel(seed=3, n_dates=60, move_skill=0.8), PV.ValueSpec(), NOW, n_sims=60)
    blind = PV.random_floor(PV.synthetic_value_panel(seed=3, n_dates=60), PV.ValueSpec(), NOW, n_sims=60)
    assert good.beats_luck and good.z > 3 and not blind.beats_luck


def test_signal_attribution_credits_the_signal_that_carried_the_change():
    p = PV.synthetic_value_panel(seed=4, n_dates=100, move_skill=0.9)
    a = PV.attribute_signals(p, PV.ValueSpec(), NOW).set_index("variant")
    assert set(a.index) >= {"bb", "nb", "bn", "nn", "marginal:movement", "marginal:score", "marginal:interaction"}
    assert a.loc["marginal:movement", "lift"] > a.loc["marginal:score", "lift"] + 0.05
    assert a.loc["bb", "reason"] == "baseline"
    with pytest.raises(ValueError):
        PV.attribute_signals(p.drop(columns=["score_base", "score_new"]), PV.ValueSpec(), NOW)


def test_sweeps_and_split_half_run_and_report_structure():
    p = PV.synthetic_value_panel(seed=5, n_dates=80, move_skill=0.8, score_skill=0.5)
    ks = PV.k_sweep(p, PV.ValueSpec(), NOW, ks=(3, 5), n_boot=60)
    assert list(ks["k"]) == [3, 5] and ks["selection_delta"].gt(0).all()
    tab, breakeven = PV.cost_sweep(p, PV.ValueSpec(), NOW, costs=(0, 10))
    assert list(tab["cost_bps"]) == [0.0, 10.0] and (breakeven is None or breakeven in (0.0, 10.0))
    sh = PV.split_half_consistency(p, PV.ValueSpec(), NOW, n_boot=60)
    assert "movement_prediction" in sh["agree"] and "ranking_value" in sh["agree"] and "movement_prediction" not in sh["disagree"]
    short = PV.split_half_consistency(p.iloc[:200], PV.ValueSpec(), NOW)
    assert short["first"] is None and short["unmeasured"] == list(PV.COMPONENTS)


def test_panel_from_predictions_refuses_an_answer_column_and_counts_unmatched_rows():
    p = PV.synthetic_value_panel(seed=6, n_dates=30)
    preds = p[["score_base", "score_new", "move_base", "move_new", "dir_base", "dir_new"]]
    outcomes = p[["fwd", "mature"]].iloc[:-40]
    joined, info = PV.panel_from_predictions(preds, outcomes, PV.ValueSpec(), NOW)
    assert info["predictions_without_outcome"] == 40 and len(joined) == len(p) - 40
    with pytest.raises(FirewallBreach):
        PV.panel_from_predictions(preds.assign(fwd=0.0), outcomes, PV.ValueSpec(), NOW)


def test_decompose_by_reports_thin_groups_as_none_not_dropped():
    p = PV.synthetic_value_panel(seed=7, n_dates=60, move_skill=0.8)
    dates = p.index.get_level_values(0).unique()
    labels = pd.Series(["early"] * 45 + ["tiny"] * 15, index=dates)
    out = PV.decompose_by(p, PV.ValueSpec(), labels, NOW, min_dates=20, n_boot=60)
    assert set(out) == {"early", "tiny"} and out["tiny"] is None and out["early"] is not None


# =============================================================================================== learning curve
def _curve(fn, n=14, name="c", start="2020-01-01", knowledge=lambda i: 10 * (i + 1)):
    c = LC.LearningCurve(name)
    d0 = pd.Timestamp(start)
    for i in range(n):
        tg, sy, mg = fn(i)
        c.add(LC.CurvePoint(i, 100 * (i + 1), knowledge(i), i // 2, same_year_gain=sy, transfer_gain=tg, risk=-0.05, memorization_gap=mg,
                            as_of=str((d0 + pd.Timedelta(days=7 * i)).date())))
    return c


def _noise(i, sd=0.0004, salt=0):
    return float(np.random.default_rng(1000 * salt + i).normal(0, sd))


def test_curve_point_and_curve_validate_their_inputs():
    assert LC.CurvePoint(1, 100, 10, 20).check()                              # validated knowledge cannot exceed knowledge
    assert LC.CurvePoint(-1, 100, 10, 2).check() and LC.CurvePoint(1, 100, 10, 2, as_of="not a date").check()
    assert LC.CurvePoint(1, 100, 10, 2, transfer_gain=float("inf")).check()
    c = LC.LearningCurve("x", [LC.CurvePoint(0, 100, 5, 1)])
    for bad in (LC.CurvePoint(0, 200, 5, 1), LC.CurvePoint(1, 50, 5, 1)):
        with pytest.raises(ValueError):
            c.add(bad)
    c.add(LC.CurvePoint(1, 100, 5, 1, as_of="2020-02-01"))
    with pytest.raises(ValueError):
        c.add(LC.CurvePoint(2, 120, 5, 1, as_of="2020-01-01"))                 # time moved backwards


def test_a_rising_transfer_curve_is_rising_and_reports_its_slope():
    a = LC.analyse_curve(_curve(lambda i: (0.0006 * i + _noise(i), 0.001 * i, 0.0)))
    assert a.verdict == LC.CurveVerdict.RISING and a.transfer.rising and a.transfer.slope == pytest.approx(0.0006, rel=0.3)
    assert a.label == ValidationLabel.NOT_VALIDATED and a.step is not None
    rec = a.as_record()
    assert rec["verdict"] == "RISING" and rec["id"] == LC.analyse_curve(_curve(lambda i: (0.0006 * i + _noise(i), 0.001 * i, 0.0))).as_record()["id"]


def test_flat_degrading_and_too_short_curves_are_named_correctly():
    assert LC.analyse_curve(_curve(lambda i: (0.003 + _noise(i, 0.0005), 0.0, 0.0), knowledge=lambda i: 10)).verdict == LC.CurveVerdict.FLAT
    deg = LC.analyse_curve(_curve(lambda i: (0.01 - 0.0006 * i + _noise(i), 0.0, 0.0)))
    assert deg.verdict == LC.CurveVerdict.DEGRADING
    short = LC.analyse_curve(_curve(lambda i: (0.001 * i, 0.0, 0.0), n=4))
    assert short.verdict == LC.CurveVerdict.INSUFFICIENT_POINTS
    empty = LC.analyse_curve(LC.LearningCurve("e"))
    assert empty.verdict == LC.CurveVerdict.INSUFFICIENT_POINTS and np.isnan(empty.validated_share)


def test_memory_growth_without_improvement_is_hoarding_and_same_year_gain_without_transfer_is_memorising():
    hoard = LC.analyse_curve(_curve(lambda i: (0.003 + _noise(i, 0.0004), 0.0, 0.0), knowledge=lambda i: 50 * (i + 1)))
    assert hoard.verdict == LC.CurveVerdict.HOARDING and hoard.memory.rising
    mem = LC.analyse_curve(_curve(lambda i: (0.003 + _noise(i, 0.0004), 0.002 * i, 0.0015 * i), knowledge=lambda i: 10))
    assert mem.verdict == LC.CurveVerdict.MEMORISING
    both = LC.analyse_curve(_curve(lambda i: (0.0007 * i + _noise(i), 0.002 * i, 0.0), knowledge=lambda i: 50 * (i + 1)))
    assert both.verdict == LC.CurveVerdict.RISING                            # knowledge growth WITH improvement is fine


def test_a_curve_that_rises_then_stops_is_plateaued_and_extrapolation_respects_the_asymptote():
    c = _curve(lambda i: (0.01 * (1 - np.exp(-i / 3)) + _noise(i, 0.0002), 0.0, 0.0), n=24)
    a = LC.analyse_curve(c)
    assert a.verdict == LC.CurveVerdict.PLATEAUED and a.saturation.preferred == "saturating" and a.saturation.asymptote == pytest.approx(0.01, rel=0.15)
    assert LC.experience_needed(c, 0.02)["reachable"] is False
    assert LC.experience_needed(c, 0.009)["method"] in ("observed", "saturating")
    lin = _curve(lambda i: (0.0005 * i + _noise(i, 0.0002), 0.0, 0.0), n=14)
    need = LC.experience_needed(lin, 0.01)
    assert need["reachable"] and need["experience"] > 1400
    flat = _curve(lambda i: (0.0 + _noise(i, 0.0003), 0.0, 0.0), n=14)
    assert LC.experience_needed(flat, 0.01)["reachable"] is False
    assert LC.experience_needed(LC.LearningCurve("e"), 0.01)["method"] is None


def test_forgetting_and_regressions_find_a_collapse():
    fell = _curve(lambda i: ((0.001 * i if i < 8 else 0.002 + _noise(i, 0.0002)) + _noise(i, 0.0001, 1), 0.0, 0.0), n=16)
    f = LC.forgetting_score(fell)
    assert f["score"] > 0.5 and f["at_experience"] >= 900
    x, y = fell.future_improvement()
    assert LC.regressions(x, y) and LC.regressions(x, y)[0]["drop"] > 0
    steady = _curve(lambda i: (0.001 * i + _noise(i, 0.00005), 0.0, 0.0))
    assert LC.forgetting_score(steady)["score"] < 0.2 and not LC.regressions(*steady.future_improvement())
    assert np.isnan(LC.forgetting_score(LC.LearningCurve("e"))["score"])


def test_change_point_and_trend_edge_cases():
    step = [0.0] * 8 + [0.01] * 8
    cp = LC.change_point(np.array(step) + np.random.default_rng(0).normal(0, 0.001, 16))
    assert cp is not None and cp[0] == 8 and cp[1] == pytest.approx(0.01, abs=0.002)
    assert LC.change_point(np.random.default_rng(1).normal(0, 0.01, 16)) is None
    assert LC.change_point([1.0, 2.0]) is None
    t = LC.trend([1, 2], [1, 2])
    assert t.lo == float("-inf") and not t.rising
    assert np.isnan(LC.trend([5, 5, 5], [1, 2, 3]).slope)
    assert LC.marginal_returns([1, 2, 3], [1, 2, 3]) == []


def test_compare_curves_needs_the_learner_to_climb_away_from_its_control():
    learner = _curve(lambda i: (0.0006 * i + _noise(i), 0.0, 0.0))
    drift = _curve(lambda i: (0.0006 * i + _noise(i, salt=1), 0.0, 0.0))            # the control drifts up just as fast: an easier era, not learning
    nolearn = _curve(lambda i: (_noise(i, salt=2), 0.0, 0.0))
    assert not LC.compare_curves(learner, drift).separates
    c = LC.compare_curves(learner, nolearn)
    assert c.separates and c.trend.slope > 0.0003 and len(c.grid) == 12
    assert not LC.compare_curves(learner, LC.LearningCurve("e")).separates


def test_curve_is_time_safe_and_round_trips_through_records():
    c = _curve(lambda i: (0.001 * i, 0.0, 0.0), n=10)
    with pytest.raises(FirewallBreach):
        LC.known_at(c, "2020-01-20")
    assert len(LC.known_at(c, "2020-01-20", drop=True)) == 3                  # 01-01, 01-08, 01-15 are strictly before 01-20
    undated = LC.LearningCurve("u", [LC.CurvePoint(0, 10, 1, 0)])
    with pytest.raises(FirewallBreach):
        LC.known_at(undated, NOW)
    back = LC.curve_from_records("c", LC.curve_to_records(c))
    assert back.fingerprint() == c.fingerprint()
    bad = LC.curve_to_records(c)[::-1]
    with pytest.raises(ValueError):
        LC.curve_from_records("c", bad)                                        # a reordered history is refused


def test_render_curve_leads_with_future_improvement_and_labels_memory_as_a_diagnostic():
    text = LC.render_curve(_curve(lambda i: (0.0006 * i + _noise(i), 0.001 * i, 0.0)))
    assert "FUTURE IMPROVEMENT" in text and "diagnostic: knowledge" in text and "NOT VALIDATED" in text.replace("—", " ").replace("—", " ")


def test_curve_point_from_transfer_uses_the_year_axis_and_leaves_untested_as_nan():
    u = _world(4)
    d = T.prepare_units(u.assign(learned=u["base"]), NOW)
    tr = d[d["year"].astype(int) <= 2012]
    scope = T.TrainingScope.from_units(tr, dt.date(2013, 1, 1))
    out = d.copy()
    out["learned"] = out["base"] + T.rule_learner_fit(tr)(out) * (out["alt"] - out["base"])
    rep = T.evaluate_transfer(out, scope, NOW, axes=(T.Axis.YEAR, T.Axis.REGIME), n_boot=100)
    p = LC.curve_point_from_transfer(3, 5000, rep, 40, 12, as_of="2012-12-31")
    assert p.transfer_gain > 0.003 and np.isfinite(p.same_year_gain) and not p.check()
    none = LC.curve_point_from_transfer(3, 5000, rep, 40, 12, forward_axes=("REGIME",))
    assert np.isnan(none.transfer_gain)                                       # the regime axis had no novel units: NaN, not zero


# =============================================================================================== multi-dimensional learning delta
def _pairs(n=8, **improve):
    """before/after metrics where `improve` maps a metric to its after-minus-before change."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        b = {"mover_hit": 0.4, "mean_week": 0.002, "dir_hit": 0.5, "worst5": -0.10, "max_dd": -0.30, "in_band": 0.20, "brier": 0.25}
        b = {k: v + rng.normal(0, 0.01) for k, v in b.items()}                   # window-to-window variation shared by before and after
        a = {k: v + (improve[k] + rng.normal(0, 0.002) if k in improve else 0.0) for k, v in b.items()}     # unimproved metrics do not move
        out.append({"before": b, "after": a})
    return out


def test_delta_orients_every_dimension_so_that_positive_is_better():
    d = LC.compute_learning_delta(_pairs(mover_hit=0.1, worst5=0.05, max_dd=0.05, brier=-0.05, in_band=0.1), transfer_deltas=[0.01] * 8)
    assert d["movement_delta"].state == LC.DeltaState.BETTER and d["movement_delta"].delta == pytest.approx(0.1, abs=0.01)
    assert d["risk_delta"].state == LC.DeltaState.BETTER                       # worst week less negative
    assert d["drawdown_delta"].state == LC.DeltaState.BETTER                   # drawdown shallower
    assert d["calibration_delta"].state == LC.DeltaState.BETTER and d["calibration_delta"].delta > 0     # lower Brier is better
    assert d["band_share_delta"].state == LC.DeltaState.BETTER and d["transfer_delta"].state == LC.DeltaState.BETTER
    assert d["direction_delta"].state == LC.DeltaState.NO_CHANGE and d["selection_delta"].state == LC.DeltaState.NO_CHANGE
    worse = LC.compute_learning_delta(_pairs(worst5=-0.05))
    assert worse["risk_delta"].state == LC.DeltaState.WORSE


def test_one_positive_dimension_is_never_proof_the_whole_system_works():
    d = LC.compute_learning_delta(_pairs(mover_hit=0.15))
    assert d["movement_delta"].state == LC.DeltaState.BETTER and not d.whole_system_claim_allowed
    st = d.statement()
    assert "1 of 8 dimensions better" in st and "not proof" in st and "improved" not in st.lower()
    assert d["transfer_delta"].state == LC.DeltaState.UNTESTED and d["transfer_delta"].delta != d["transfer_delta"].delta


def test_the_whole_system_claim_needs_every_dimension_measured_and_tier_one_and_transfer_better():
    every = dict(mover_hit=0.1, mean_week=0.01, dir_hit=0.1, worst5=0.05, max_dd=0.05, in_band=0.15, brier=-0.05)
    full = LC.compute_learning_delta(_pairs(**every), transfer_deltas=[0.01] * 8)
    assert full.whole_system_claim_allowed and "not validated" in full.statement().lower()
    assert not LC.compute_learning_delta(_pairs(**every)).whole_system_claim_allowed                     # transfer untested
    no_dir = {k: v for k, v in every.items() if k != "dir_hit"}
    assert LC.compute_learning_delta(_pairs(**no_dir), transfer_deltas=[0.01] * 8).whole_system_claim_allowed           # direction unchanged is fine...
    hurt = dict(every, worst5=-0.05)
    assert not LC.compute_learning_delta(_pairs(**hurt), transfer_deltas=[0.01] * 8).whole_system_claim_allowed         # ...but harm to risk is not


def test_a_higher_tier_gain_cannot_buy_back_lower_tier_damage():
    d = LC.compute_learning_delta(_pairs(dir_hit=0.15, in_band=-0.1, worst5=-0.05))
    assert d["direction_delta"].state == LC.DeltaState.BETTER and d["band_share_delta"].state == LC.DeltaState.WORSE
    conflicts = d.tier_conflicts()
    assert any("direction_delta improved but band_share_delta (tier 1) got worse" in c for c in conflicts)
    assert d.decided_by().startswith("tier 1: WORSE")
    assert LC.compute_learning_delta(_pairs()).decided_by() == "undecided"


def test_delta_handles_empty_small_and_missing_metric_inputs():
    e = LC.compute_learning_delta([])
    assert e.n_pairs == 0 and all(v.state == LC.DeltaState.UNTESTED for v in e.values.values()) and not e.whole_system_claim_allowed
    small = LC.compute_learning_delta(_pairs(n=2, mover_hit=0.2), min_pairs=3)
    assert small["movement_delta"].state == LC.DeltaState.INSUFFICIENT
    partial = _pairs(n=6, mover_hit=0.1)
    for p in partial[:4]:
        del p["after"]["mover_hit"]
    assert LC.compute_learning_delta(partial)["movement_delta"].n == 2
    assert LC.compute_learning_delta(_pairs())["movement_delta"].as_dict()["state"] == "NO_CHANGE"
    rec = LC.compute_learning_delta(_pairs(mover_hit=0.1)).as_record()
    assert rec["whole_system_claim_allowed"] is False and rec["id"]


def test_delta_from_harness_records_uses_only_past_only_transfer_pairs_and_groups_by_label():
    m = lambda v: {"mean_week": v, "in_band": 0.2, "worst5": -0.1, "max_dd": -0.3, "dir_hit": 0.5, "mover_hit": 0.4}
    recs = [{"run1": m(0.001), "run2": m(0.004), "transfer_s0": m(0.001), "transfer_s1": m(0.003), "transfer_anachronistic": False},
            {"run1": m(0.001), "run2": m(0.005), "transfer_s0": m(0.001), "transfer_s1": m(0.009), "transfer_anachronistic": True},
            {"run1": m(0.002), "run2": m(0.006)}]
    pairs, transfer = LC.pairs_from_delta_records(recs)
    assert len(pairs) == 3 and transfer == [pytest.approx(0.002)]              # the anachronistic transfer pair is dropped
    d = LC.delta_from_records(recs, min_pairs=1)
    assert d["selection_delta"].delta == pytest.approx(0.0037, abs=1e-3) and d["transfer_delta"].n == 1
    by = LC.delta_by_group(_pairs(6, mover_hit=0.1), ["a", "b", "a", "b", "a", "b"], min_pairs=2)
    assert list(by) == ["a", "b"] and by["a"].n_pairs == 3
    with pytest.raises(ValueError):
        LC.delta_by_group(_pairs(3), ["a"])


def test_same_year_records_make_a_curve_that_cannot_claim_transfer():
    recs = [{"tag": "main", "mean_week": 0.001 * (i + 1)} for i in range(12)] + [{"tag": "other", "mean_week": 0.5}]
    c = LC.curve_from_play_records(recs)
    assert len(c) == 12 and c.column("same_year_gain")[-1] == pytest.approx(0.011)
    assert LC.analyse_curve(c).verdict == LC.CurveVerdict.INSUFFICIENT_POINTS      # no transfer measurement: never the learning curve


# =============================================================================================== tables, comparisons, tail attribution
def test_component_table_comparison_and_tail_attribution():
    good = _dec(seed=1, move_skill=0.8, n_boot=80)
    blind = _dec(seed=1, n_boot=80)
    tab = PV.component_table(good)
    assert list(tab["component"]) == list(PV.COMPONENTS) and tab.loc[tab["component"] == "movement_prediction", "status"].iloc[0] == PV.STATUS_MEASURED
    cmp_ = PV.compare_decompositions(blind, good)
    assert "movement_prediction" in cmp_["better"] and "timing_value" in cmp_["unmeasured"]
    assert "movement_prediction" in PV.compare_decompositions(good, blind)["worse"]
    p = PV.synthetic_value_panel(seed=8, n_dates=80, move_skill=0.8)
    t = PV.tail_attribution(p, PV.ValueSpec(), NOW, tail=0.1)
    assert t["n_tail"] == 8 and 0.0 <= t["share_better"] <= 1.0 and np.isfinite(t["average_gain"])
    assert PV.tail_attribution(p.iloc[:40], PV.ValueSpec(), NOW)["n_tail"] == 0


def test_scorecard_diff_and_the_to_do_list_of_missing_evidence():
    old = good_card(learner_version="old")
    new = good_card(learner_version="new", cross_year_gain=M(0.02, 0.018, 0.022), stability=M(0.3, 0.29, 0.31))
    tab = S.scorecard_diff(old, new).set_index("field")
    assert tab.loc["cross_year_gain", "state"] == "better" and tab.loc["stability", "state"] == "worse" and tab.loc["calibration", "state"] == "same"
    partial = mutate(good_card(), risk_change=S.Measured.untested(), future_leak_status=S.LeakStatus.UNAUDITED)
    todo = S.missing_for_claim(partial)
    assert len(todo) == 2 and any("risk" in t for t in todo) and any("future-leak" in t for t in todo)
    assert S.missing_for_claim(good_card()) == []
    assert S.scorecard_diff(old, mutate(new, cross_year_gain=S.Measured.untested())).set_index("field").loc["cross_year_gain", "state"] == "untested"


# =============================================================================================== section 34 completion
def test_each_component_can_be_measured_alone_and_matches_the_full_decomposition():
    p = PV.synthetic_value_panel(seed=1, n_dates=80, move_skill=0.8, score_skill=0.5)
    full = PV.decompose_value(p, PV.ValueSpec(), NOW, n_boot=80, seed=3)
    for name in ("predictive_effect", "movement_prediction", "ranking_value", "selection_value", "risk_value", "portfolio_value"):
        one = PV.measure(name, p, PV.ValueSpec(), NOW, n_boot=80, seed=3)
        assert one.delta == pytest.approx(full[name].delta) and one.status == full[name].status
    with pytest.raises(ValueError):
        PV.measure("charisma", p, PV.ValueSpec(), NOW)


def test_waterfall_shows_where_prediction_stops_being_portfolio_value():
    ok = PV.value_waterfall(PV.synthetic_value_panel(seed=2, n_dates=100, move_skill=0.8), PV.ValueSpec(), NOW, n_boot=100)
    assert [s.stage for s in ok.stages][0] == "1 prediction" and len(ok.stages) == 6
    assert ok.stages[0].survives and ok.stages[1].survives and ok.stages[2].survives
    assert ok.first_loss is None or ok.first_loss in {s.stage for s in ok.stages}
    assert "survives" in ok.verdict or "lost between" in ok.verdict
    blind = PV.value_waterfall(PV.synthetic_value_panel(seed=2, n_dates=100), PV.ValueSpec(), NOW, n_boot=100)
    assert not any(s.survives for s in blind.stages[:3])
    assert "nothing was gained" in blind.verdict or "NO prediction" in blind.verdict         # never "the gain survives" for a blind signal
    assert "6 objective" in ok.render()
    # planted leak: strong prediction and selection, but at k=3 the tiered objective rejects the change (gaming guard: no edge in the weeks)
    p = PV.synthetic_value_panel(seed=5, n_dates=80, move_skill=0.8, score_skill=0.5)
    rej = PV.value_waterfall(p, PV.ValueSpec(), NOW, k=3, n_boot=100)
    st = {s.stage: s for s in rej.stages}
    assert st["3 selection"].survives and not st["6 objective"].survives
    assert rej.first_loss is not None and "lost between" in rej.verdict and rej.lost_between[1] == rej.first_loss


def test_conversion_summary_counts_prediction_gains_that_never_reached_the_portfolio():
    decs = [PV.decompose_value(PV.synthetic_value_panel(seed=s, n_dates=80, move_skill=0.8), PV.ValueSpec(), NOW, n_boot=60) for s in (1, 2, 3)]
    decs.append(PV.decompose_value(PV.synthetic_value_panel(seed=9, n_dates=80), PV.ValueSpec(), NOW, n_boot=60))
    cs = PV.conversion_summary(decs)
    assert cs["n"] == 4 and cs["with_prediction_gain"] == 3 and cs["converted"] + cs["lost"] == 3
    assert np.isnan(PV.conversion_rate(decs[-1])) and PV.conversion_summary([])["rate"] != PV.conversion_summary([])["rate"]
    sc = PV.significant_components(decs[0])
    assert sum(len(v) for v in sc.values()) == len(PV.COMPONENTS) and "movement_prediction" in sc["gained"]


def test_risk_breakdown_and_timing_split_separate_what_the_single_number_hides():
    p = PV.synthetic_value_panel(seed=4, n_dates=160, move_skill=0.8)
    rb = PV.risk_breakdown(p, PV.ValueSpec(), NOW, n_boot=100)
    assert list(rb["measure"]) == ["worst_5pct_week", "max_drawdown", "catastrophic_weeks", "overshoot_weeks"]
    assert np.isfinite(rb[["base", "new", "delta", "lo", "hi"]].to_numpy()).all() and (rb["n_windows"] == 6).all()
    assert PV.risk_breakdown(p.iloc[:200], PV.ValueSpec(), NOW).empty
    spec = PV.ValueSpec(exposure=("exp_base", "exp_new"))
    t = PV.timing_split(PV.synthetic_value_panel(seed=5, n_dates=120, timing_skill=1.0, move_skill=0.5), spec, NOW)
    assert t["available"] and t["total"] == pytest.approx(t["level"] + t["timing"])
    assert PV.timing_split(p, PV.ValueSpec(), NOW)["available"] is False


def test_picked_direction_value_uses_the_picks_and_needs_a_spread_not_only_accuracy():
    good = PV.picked_direction_value(PV.synthetic_value_panel(seed=6, n_dates=100, move_skill=0.8, dir_skill=0.8), PV.ValueSpec(), NOW, n_boot=100)
    blind = PV.picked_direction_value(PV.synthetic_value_panel(seed=6, n_dates=100, move_skill=0.8), PV.ValueSpec(), NOW, n_boot=100)
    assert good["available"] and good["new"]["accuracy"] > 0.9 and good["new"]["spread"] > 0.02 and good["accuracy_vs_coin"].lo > 0
    assert abs(blind["new"]["accuracy"] - 0.5) < 0.12 and blind["accuracy_vs_coin"].lo < 0.1
    assert PV.picked_direction_value(PV.synthetic_value_panel(seed=6, n_dates=30).drop(columns=["dir_new"]), PV.ValueSpec(), NOW) == {"available": False}


def test_value_by_group_selection_over_luck_and_signal_quality_curve():
    p = PV.synthetic_value_panel(seed=7, n_dates=90, move_skill=0.8)
    dates = p.index.get_level_values(0).unique()
    lab = pd.Series(["a"] * 45 + ["b"] * 30 + ["tiny"] * 15, index=dates)
    tab = PV.value_by_group(p, PV.ValueSpec(), lab, NOW, min_dates=20, n_boot=60)
    assert list(tab["group"]) == ["a", "b", "tiny"] and tab.set_index("group").loc["tiny"].isna().all()
    assert tab.set_index("group").loc["a", "movement_prediction"] > 0.3
    sol = PV.selection_over_luck(p, PV.ValueSpec(), NOW, n_sims=60)
    assert sol["new"].beats_luck and not sol["base"].beats_luck and sol["exceeds_luck_spread"] and sol["advantage"] > 0.2
    curve = PV.signal_quality_needed(p, PV.ValueSpec(), NOW, skills=(0.0, 0.5, 0.9), n_boot=60)
    assert list(curve["skill"]) == [0.0, 0.5, 0.9] and curve["movement_delta"].is_monotonic_increasing
    with pytest.raises(ValueError):
        PV.signal_quality_needed(p.drop(columns=["move_base"]), PV.ValueSpec(), NOW)


# =============================================================================================== section 47 completion
def test_every_one_of_the_19_fields_has_a_control_dependency_and_a_claim_verdict():
    assert set(S.FIELD_CONTROLS) == set(S.SECTION_47_FIELDS) and len(S.SECTION_47_FIELDS) == 19
    claims = S.field_claims(good_card())
    assert set(claims) == set(S.SECTION_47_FIELDS)
    assert claims["cross_year_gain"].may_claim_improved and claims["future_leak_status"].may_claim_improved
    assert not claims["compute_cost"].may_claim_improved                  # a cost is a fact
    assert not claims["identity_gap"].may_claim_improved or claims["identity_gap"].controls_missing == ()


@pytest.mark.parametrize("drop,affected", [
    ("D_random_learner", {"cross_year_gain", "same_year_gain", "band_share", "movement_performance"}),
    ("C_identity_memoriser", {"cross_year_gain", "transfer_ratio", "memorization_gap"}),
    ("E_leaky_learner", {"cross_year_gain", "future_leak_status"}),
    ("A_no_learning", {"baseline_performance", "risk_change", "stability"}),
])
def test_removing_a_control_removes_the_claim_from_exactly_the_fields_that_depend_on_it(drop, affected):
    card = good_card()
    card = mutate(card, controls={k: v for k, v in card.controls.items() if k != drop})
    claims = S.field_claims(card)
    for f in affected:
        assert not claims[f].may_claim_improved and drop in claims[f].controls_missing, f
    unaffected = [f for f, c in claims.items() if drop not in S.FIELD_CONTROLS[f] and f != "compute_cost"]
    assert any(claims[f].may_claim_improved for f in unaffected)


def test_a_misbehaving_control_also_withdraws_the_claim():
    noisy_a = with_control(good_card(), "A_no_learning", gain=M(0.01, 0.008, 0.012))          # the no-learning control moved
    assert not S.field_claims(noisy_a)["risk_change"].may_claim_improved
    blind_e = with_control(good_card(), "E_leaky_learner", detected=False)
    assert not S.field_claims(blind_e)["future_leak_status"].may_claim_improved
    unaudited = mutate(good_card(), future_leak_status=S.LeakStatus.UNAUDITED)
    assert not S.field_claims(unaudited)["future_leak_status"].may_claim_improved and not S.field_claims(unaudited)["future_leak_status"].measured


def test_text_claiming_improvement_in_a_field_without_its_control_is_refused():
    card = with_control(good_card(), "E_leaky_learner", detected=False)
    assert S.refuse_unsupported_field_claims(card, "The cross year gain improved on unseen years.") == ["cross_year_gain"]
    assert S.refuse_unsupported_field_claims(card, "The cross year gain did not improve, no improvement.") == []
    assert S.refuse_unsupported_field_claims(good_card(), "The cross year gain improved.") == []
    assert "compute_cost" in S.refuse_unsupported_field_claims(good_card(), "Compute cost is better now.")
    assert S.refuse_unsupported_field_claims(good_card(), "Nothing to report.") == []


def test_claim_matrix_and_claimable_fields_are_a_strict_subset_of_the_measured_ones():
    card = mutate(good_card(), risk_change=S.Measured.untested("no weeks"), calibration=M(0.03, 0.02, 0.04))
    tab = S.claim_matrix(card)
    assert len(tab) == 19 and not tab.set_index("field").loc["risk_change", "may_claim_improved"]
    cf = S.claimable_fields(card)
    assert "risk_change" not in cf and "compute_cost" not in cf and "cross_year_gain" in cf
    assert set(cf) <= set(tab.loc[tab["measured"], "field"])


def test_calibration_control_is_brier_skill_against_the_base_rate():
    rng = np.random.default_rng(50)
    p = rng.uniform(0.05, 0.95, 800)
    y = (rng.random(800) < p).astype(float)
    good = S.calibration_control(p, y)
    useless = S.calibration_control(np.full(800, y.mean()), y)
    assert good.measured and good.lo > 0 and "base-rate" in good.note
    assert useless.measured and abs(useless.value) < 1e-9
    assert not S.calibration_control(p[:10], y[:10]).measured and not S.calibration_control(p, np.ones(800)).measured


# =============================================================================================== section 48 completion
def test_curve_series_gives_every_section_48_quantity_against_experience_with_its_own_trend():
    c = _curve(lambda i: (0.0006 * i + _noise(i), 0.001 * i, 0.0002 * i), knowledge=lambda i: 10 * (i + 1))
    s = LC.curve_series(c)
    assert set(s) == set(LC.CURVE_SERIES)
    x, y, tr = s["transfer_gain"]
    assert len(x) == len(y) == 14 and tr.rising and s["knowledge_count"][2].rising and s["memorization_gap"][2].rising
    assert not s["risk"][2].rising and not s["risk"][2].falling
    nan_curve = LC.LearningCurve("n", [LC.CurvePoint(i, 100 * (i + 1), 5, 1) for i in range(5)])
    assert len(LC.curve_series(nan_curve)["transfer_gain"][0]) == 0
    share = LC.validated_share_curve(c)
    assert len(share) == 14 and share[0] == 0.0 and share[-1] == pytest.approx(6 / 140)
    assert np.isnan(LC.validated_share_curve(LC.LearningCurve("z", [LC.CurvePoint(0, 1, 0, 0)]))[0])


# =============================================================================================== section 65 completion
def test_every_delta_dimension_carries_a_confidence_interval_and_a_p_value():
    d = LC.compute_learning_delta(_pairs(mover_hit=0.1, worst5=0.05, in_band=0.1), transfer_deltas=[0.01, 0.012, 0.009, 0.011, 0.01])
    assert set(d.values) == set(LC.DIMENSIONS) and len(LC.DIMENSIONS) == 8
    for name in ("movement_delta", "risk_delta", "band_share_delta", "transfer_delta", "direction_delta"):
        v = d[name]
        assert v.lo <= v.delta <= v.hi and 0.0 <= v.p_signflip <= 1.0 and v.n >= 5
    assert d["movement_delta"].p_signflip < 0.05 and d["direction_delta"].p_signflip > 0.5
    assert d["calibration_delta"].n == 8
