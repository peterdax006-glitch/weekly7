"""SELF_LEARNING_CONTRACT section 53 (disagreement engine): who disagreed, why, who was right, in what context, and whether
disagreement itself predicts. Each mechanism is run against a planted truth and against a control that must find nothing."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning import disagreement as D
from engine.learning.core import FirewallBreach

NOW = "2035-01-01"
CFG = D.DisagreementConfig(n_boot=100, n_perm=100, seed=5)


@pytest.fixture(scope="module")
def trap():
    sf, truth = D.planted_stances(0, herd_trap=True)
    return sf, truth, D.DisagreementEngine(CFG).assess(sf, NOW)


@pytest.fixture(scope="module")
def clean():
    sf, _ = D.planted_stances(1, herd_trap=False, disagreement_predicts_move=True)
    return sf, D.DisagreementEngine(CFG).assess(sf, NOW)


def frame(stances, y=None, ctx=None, conf=None, **kw):
    df = pd.DataFrame(stances)
    n = len(df)
    return D.StanceFrame.build(df, pd.bdate_range("2021-01-04", periods=n), np.full(n, np.nan) if y is None else y,
                               ctx, conf, **kw)


# ------------------------------------------------------------------------------------------ the conflict measure
def test_conflict_score_is_graded_and_handles_abstention():
    S = np.array([[1, 1, 1], [1, -1, np.nan], [1, 1, -1], [0.9, -0.1, np.nan], [np.nan, np.nan, np.nan], [1, np.nan, np.nan]])
    c = D.conflict_score(S)
    assert c[0] == 0 and c[1] == pytest.approx(1.0) and c[2] == pytest.approx(1 - 1 / 3)
    assert c[3] < c[2] and c[4] == 0 and c[5] == 0                     # one voice or none is never a disagreement
    assert D.consensus(S).tolist() == [1, 0, 1, 1, 0, 1]


def test_features_frame_counts_sides():
    sf = frame({"a": [1, 1], "b": [-1, 1], "c": [np.nan, 0.05]})
    f = D.disagreement_features(sf, CFG)
    assert f.loc[0, ["n_pos", "n_neg", "n_abstain"]].tolist() == [1, 1, 1]
    assert f.loc[1, "conflict"] == 0 and f.loc[1, "n_neg"] == 0


def test_bh_adapter_ignores_nan():
    q = D.benjamini_hochberg([0.001, np.nan, 0.04, 0.5])
    assert np.isnan(q[1]) and q[0] < q[2] < q[3] and (q[~np.isnan(q)] <= 1).all()


# ------------------------------------------------------------------------------------------ who was right, in what context
def test_pair_table_finds_the_planted_crisis_winner_and_not_a_calm_winner(trap):
    _, _, rep = trap
    crisis = [p for p in rep.pairs if {p.a, p.b} == {"pattern", "analog"} and p.context == "regime=crisis"][0]
    calm = [p for p in rep.pairs if {p.a, p.b} == {"pattern", "analog"} and p.context == "regime=calm"][0]
    assert crisis.winner == "pattern" and crisis.rate > 0.8 and crisis.lo > 0.5
    assert calm.winner is None                                            # equally good in calm: no winner is declared


def test_source_records_show_the_herd_followers_lose_when_alone(trap):
    _, _, rep = trap
    rec = {s.source: s for s in rep.sources}
    assert rec["pattern"].acc_in_disagreement > rec["analog"].acc_in_disagreement + 0.1
    assert rec["pattern"].solo_dissent_wins > 0.55 > rec["direction"].solo_dissent_wins
    assert rec["movement"].n_spoke == 0                                    # stance 0 = no view, so it is never scored


def test_context_lift_marks_the_regime_that_breeds_disagreement(trap):
    _, _, rep = trap
    assert rep.context_lift["regime=crisis"] > 1.0 > rep.context_lift["regime=calm"]
    assert rep.reasons.get("CONTEXT_DRIVEN", 0) > 0
    assert rep.reasons.get("CONTEXT_DRIVEN", 0) + rep.reasons.get("UNEXPLAINED", 0) == len(rep.events)
    sf, _ = D.planted_stances(0, herd_trap=False)
    assert D.context_disagreement_rates(sf, CFG) == {}                    # no regime effect planted => none reported


# ------------------------------------------------------------------------------------------ learned rules and their controls
def test_trust_pattern_in_crisis_is_learned_and_beats_majority_vote_out_of_sample(trap):
    _, _, rep = trap
    assert any(r.kind == "TRUST" and r.trust == "pattern" and r.context == "regime=crisis" for r in rep.rules)
    ev = rep.rule_eval
    assert ev.gain > 0.05 and ev.lo > 0 and ev.placebo_rules == 0
    assert "improve" in ev.verdict


def test_no_trust_rule_when_there_is_no_herd_trap_or_when_outcomes_are_shuffled():
    for kw in (dict(herd_trap=False), dict(herd_trap=False, shuffle_outcomes=True)):
        sf, _ = D.planted_stances(2, **kw)
        rep = D.DisagreementEngine(CFG).assess(sf, NOW)
        assert [r for r in rep.rules if r.kind == "TRUST" and r.trust == "pattern" and r.context.startswith("regime")] == []
        if rep.rule_eval is not None:
            assert not (rep.rule_eval.lo > 0.02)                         # nothing that beats majority vote by a real margin


def test_rules_are_stable_under_week_resampling(trap):
    sf, _, _ = trap
    train, _ = sf.time_split(CFG.train_frac)
    st = D.rule_stability(train, CFG, NOW, n_rep=6).set_index(["kind", "trust", "against", "context"])
    assert st.loc[("TRUST", "pattern", "analog", "regime=crisis"), "reappear"] >= 0.8


def test_walk_forward_gain_is_consistent_across_folds(trap):
    sf, _, _ = trap
    wf = D.walk_forward_rules(sf, CFG, NOW, n_folds=3)
    assert len(wf) >= 2 and (wf["gain"] > 0).all() and (wf["placebo_rules"] == 0).all()


def test_arbitrate_follows_trust_rules_and_abstains_on_abstain_rules():
    S = pd.DataFrame({"a": [1.0, 1.0, 1.0, -1.0], "b": [-1.0, -1.0, -1.0, 1.0], "c": [-1.0, -1.0, -1.0, 1.0]})
    ctx = pd.DataFrame({"regime": ["x", "x", "y", "y"]})
    sf = D.StanceFrame.build(S, pd.bdate_range("2021-01-04", periods=4), np.zeros(4), ctx)
    trust = D.ArbitrationRule("t", "TRUST", "a", "b", "regime=x", 50, 0.8, 0.7, 0.9, 0.001, "2020-01-01", 100)
    absten = D.ArbitrationRule("z", "ABSTAIN", None, None, "regime=y", 50, 0.4, 0.3, 0.5, 0.001, "2020-01-01", 100)
    final, used = D.arbitrate(sf, [absten, trust], CFG)
    assert final.tolist() == [1.0, 1.0, 0.0, 0.0] and used.tolist()[:2] == [0, 0] and used[2] == 1
    assert D.arbitrate(sf, [], CFG)[0].tolist() == [-1.0, -1.0, -1.0, 1.0]        # no rules => plain majority vote


def test_rule_records_validate():
    bad = D.ArbitrationRule("t", "TRUST", "a", "a", "all", 10, 0.6, 0.5, 0.7, 0.01, "2020-01-01", 10)
    assert bad.validate() != []
    assert D.ArbitrationRule("t", "WAT", None, None, "all", 10, 0.6, 0.5, 0.7, 0.01, "x", 10).validate() != []


# ------------------------------------------------------------------------------------------ is disagreement predictive?
def test_conflict_predicts_move_size_and_consensus_errors_when_planted(clean):
    _, rep = clean
    assert rep.predictive_for("abs_move").verdict is D.Predictiveness.PREDICTIVE
    assert rep.predictive_for("consensus_wrong").verdict is D.Predictiveness.PREDICTIVE
    t = rep.predictive_for("abs_move")
    assert t.lo > 0 and t.auc_oos > 0.52 and t.rho_early > 0 and t.rho_late > 0


def test_no_predictiveness_claimed_when_the_link_is_absent_or_outcomes_are_shuffled():
    sf, _ = D.planted_stances(3, herd_trap=False, disagreement_predicts_move=False)
    t = D.DisagreementEngine(CFG).assess(sf, NOW).predictive_for("abs_move")
    assert t.verdict is D.Predictiveness.NOT_PREDICTIVE
    sf2, _ = D.planted_stances(3, herd_trap=False, shuffle_outcomes=True)
    rep = D.DisagreementEngine(CFG).assess(sf2, NOW)
    assert all(p.verdict is D.Predictiveness.NOT_PREDICTIVE for p in rep.predictive)


def test_a_regime_confound_is_labelled_context_artifact_not_predictive():
    """Planted defect: regime B is both more conflicted and more volatile; inside a regime conflict tells nothing."""
    rng = np.random.default_rng(4)
    n = 2400
    reg = np.where(rng.random(n) < 0.5, "A", "B")
    dates = pd.bdate_range("2015-01-05", periods=n // 4).repeat(4)[:n]
    u = rng.choice([-1.0, 1.0], n)
    flip = np.where(reg == "B", 0.45, 0.05)
    st = {c: np.where(rng.random(n) < flip, -u, u) * (0.6 + 0.4 * rng.random(n)) for c in ("a", "b", "c", "d")}
    y = np.where(reg == "B", 0.05, 0.01) * rng.standard_normal(n) + 0.002 * u
    sf = D.StanceFrame.build(pd.DataFrame(st), dates, y, pd.DataFrame({"regime": reg}))
    t = D.predictive_test(sf, CFG, "abs_move", np.random.default_rng(0))
    assert t.rho > 0.2 and abs(t.rho_within_context) < 0.5 * t.rho
    assert t.verdict is D.Predictiveness.CONTEXT_ARTIFACT


def test_consensus_accuracy_falls_with_conflict(clean):
    sf, _ = clean
    tab = D.consensus_accuracy_by_conflict(sf)
    assert tab["consensus_accuracy"].iloc[0] > tab["consensus_accuracy"].iloc[-1] + 0.05


def test_resolution_value_bounds_what_arbitration_can_add(trap):
    sf, _, _ = trap
    r = D.resolution_value(sf, CFG)
    assert r["oracle_ceiling"] >= r["best_single_hindsight"] >= 0 and r["oracle_ceiling"] >= r["majority_vote"]
    assert r["best_source"] == "pattern" and r["headroom"] > 0.2


# ------------------------------------------------------------------------------------------ why
def test_reasons_are_assigned_for_handmade_disagreements():
    stances = {"a": [0.9, 0.9, 0.9, 0.9, 0.9], "b": [-0.2, -0.9, -0.9, -0.9, -0.9],
               "c": [np.nan] * 5, "d": [np.nan, 0.9, 0.9, 0.9, 0.9], "e": [np.nan, np.nan, -0.9, -0.9, -0.9]}
    conf = pd.DataFrame({"a": [.9] * 5, "b": [.9, .9, .1, .9, .9], "c": [.5] * 5, "d": [.9, .9, .9, .9, .9], "e": [.9, .9, .1, .9, .9]})
    sf = frame(stances, conf=conf)
    mech = {"a": ("momentum",), "b": ("value",), "d": ("momentum",), "e": ("value",)}
    ev = D.find_events(sf, dataclasses.replace(CFG, conflict_min=0.2), mech)
    reasons = {e.index: e.reason for e in ev}
    assert reasons[0] is D.Reason.LOW_COVERAGE                      # 3 of 5 abstained
    assert reasons[1] is D.Reason.LOW_COVERAGE or reasons[1] is D.Reason.MECHANISM_CONFLICT
    assert reasons[2] is D.Reason.CONFIDENCE_GAP
    assert reasons[4] is D.Reason.MECHANISM_CONFLICT
    assert all(e.validate() == [] for e in ev)


def test_weak_stance_reason():
    sf = frame({"a": [0.9], "b": [-0.2], "c": [0.8], "d": [-0.15]})
    ev = D.find_events(sf, dataclasses.replace(CFG, conflict_min=0.1), None)
    assert ev and ev[0].reason is D.Reason.WEAK_STANCE


def test_driver_regression_finds_an_observable_driver_out_of_sample_and_reports_none_for_noise():
    rng = np.random.default_rng(6)
    n = 1800
    x = rng.standard_normal(n)
    u = rng.choice([-1.0, 1.0], n)
    p_flip = 1 / (1 + np.exp(-(1.8 * x - 0.5)))
    st = pd.DataFrame({c: np.where(rng.random(n) < p_flip * 0.6, -u, u) for c in ("a", "b", "c", "d")}) * 0.8
    dates = pd.bdate_range("2014-01-06", periods=n // 3).repeat(3)[:n]
    sf = D.StanceFrame.build(st, dates, u * 0.01, None, features=pd.DataFrame({"x": x, "junk": rng.standard_normal(n)}))
    d = D.disagreement_drivers(sf, CFG)
    assert d.r2_oos > 0.01 and abs(d.coefs["x"]) > 5 * abs(d.coefs["junk"]) and d.verdict == "drivers found"
    sf2 = dataclasses.replace(sf, features=pd.DataFrame({"j1": rng.standard_normal(n), "j2": rng.standard_normal(n)}))
    assert "no observable driver" in D.disagreement_drivers(sf2, CFG).verdict


# ------------------------------------------------------------------------------------------ firewall and degenerate inputs
def test_pending_outcomes_are_blanked_and_strict_mode_raises():
    n = 60
    dates = pd.bdate_range("2021-01-04", periods=n)
    st = pd.DataFrame({"a": np.ones(n), "b": -np.ones(n)})
    sf = D.StanceFrame.build(st, dates, np.full(n, 0.01), None, horizon_days=5, now="2021-02-15")
    assert sf.n_pending > 0 and np.isnan(sf.y[-1]) and np.isfinite(sf.y[0])
    assert (dates[np.isfinite(sf.y)] + pd.Timedelta(days=5) < pd.Timestamp("2021-02-15")).all()
    with pytest.raises(FirewallBreach):
        D.StanceFrame.build(st, dates, np.full(n, 0.01), None, now="2021-02-15", strict=True)


def test_training_window_reaching_now_is_refused(trap):
    sf, _, _ = trap
    with pytest.raises(FirewallBreach):
        D.learn_rules(sf, CFG, "2017-06-01")
    with pytest.raises(FirewallBreach):
        D.DisagreementEngine(CFG).assess(sf, "2017-06-01")


def test_empty_single_source_and_all_abstain_frames():
    empty = D.StanceFrame.build(pd.DataFrame({"a": [], "b": []}), pd.DatetimeIndex([]), np.array([]), None)
    rep = D.DisagreementEngine(CFG).assess(empty, NOW)
    assert rep.n_obs == 0 and rep.events == () and rep.rules == ()
    one = frame({"a": np.ones(50)}, y=np.full(50, 0.01))
    assert D.DisagreementEngine(CFG).assess(one, NOW).events == ()
    n = 200
    abst = frame({"a": np.full(n, np.nan), "b": np.full(n, np.nan)}, y=np.random.default_rng(0).standard_normal(n))
    rep = D.DisagreementEngine(CFG).assess(abst, NOW)
    assert rep.disagreement_rate == 0 and all(t.verdict is D.Predictiveness.INSUFFICIENT for t in rep.predictive)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        frame({"a": [2.0, 0.0], "b": [0.0, 0.0]})                                    # stance outside [-1, 1]
    with pytest.raises(ValueError):
        D.StanceFrame.build(pd.DataFrame({"a": [1.0]}), pd.bdate_range("2021-01-04", periods=2), np.zeros(1), None)
    with pytest.raises(ValueError):
        D.DisagreementEngine(D.DisagreementConfig(train_frac=0.99))


def test_determinism_and_report_validity(trap):
    sf, _, rep = trap
    again = D.DisagreementEngine(CFG).assess(sf, NOW)
    assert [r.rule_id for r in again.rules] == [r.rule_id for r in rep.rules]
    assert again.predictive_for("abs_move").rho == rep.predictive_for("abs_move").rho
    assert D.validate_report(rep) == []
    assert "IMPLEMENTED" in rep.render_text()


def test_calibration_harness_reports_the_planted_worlds():
    res = D.run_planted_calibration(0, CFG)
    assert res["trap"]["crisis_pattern_rule"] and res["trap"]["rule_gain"] > 0
    assert not res["no_trap"]["crisis_pattern_rule"] and not res["shuffled"]["crisis_pattern_rule"]
    assert res["shuffled"]["abs_move"] == "NOT_PREDICTIVE" and res["move"]["abs_move"] == "PREDICTIVE"
    assert all(v["errors"] == [] for v in res.values())
