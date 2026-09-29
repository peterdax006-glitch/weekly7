"""C68 checklists L, M, N, O, P, Q: selection constraint, anti-gaming, learned exits, exit independence, the +-1pp target measured
honestly, and the self-correcting network. Synthetic data only; each mechanism has a planted case it must catch, a null case where
it must find nothing, and the empty case."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine import exits as EX
from engine.learning.core import FirewallBreach
from engine.research import calibration_target as CT
from engine.research import exit_research as ER
from engine.research import quality_gate as QG
from engine.research import selection_constraint as SC
from engine.research import self_correct as SF
from engine.research.core import Availability, GateVerdict

NOW = "2021-06-01"


def make_paths(n_weeks=80, per_week=8, D=5, seed=0, start="2019-01-07", fade_share=0.5, drift_x=None):
    """Weekly positions. 'fade' names jump on day 1 then give it all back; 'trend' names drift up steadily. If drift_x is given,
    each name's weekly drift = drift_x * x where x is a returned entry feature."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    N = n_weeks * per_week
    kind = np.where(rng.random(N) < fade_share, "fade", "trend").astype(object)
    x = rng.uniform(0, 2, N)
    o, h, l, c = (np.zeros((N, D)) for _ in range(4))
    for i in range(N):
        px = 100.0
        for d in range(D):
            if drift_x is not None:
                mu = drift_x * x[i] / D
            elif kind[i] == "fade":
                mu = 0.04 if d == 0 else -0.02
            else:
                mu = 0.006
            op = px * (1 + rng.normal(0, 0.002))
            cl = op * (1 + mu + rng.normal(0, 0.004))
            o[i, d], c[i, d] = op, cl
            h[i, d], l[i, d] = max(op, cl) * (1 + abs(rng.normal(0, 0.002))), min(op, cl) * (1 - abs(rng.normal(0, 0.002)))
            px = cl
    wk = np.repeat(weeks.values.astype("datetime64[D]"), per_week)
    p = EX.Paths(o, h, l, c, o[:, 0] * 0.999, np.full(N, 0.01) + 0.002 * x, np.full(N, 0.015), wk, wk + np.timedelta64(D - 1, "D"),
                 kind, np.array([f"S{i % 50}" for i in range(N)], object))
    return p.check(), x


# ================================================================================================ L: selection constraint
def _gain_model(now="2021-01-01"):
    p, x = make_paths(n_weeks=60, drift_x=0.07, seed=1)
    tr = p.until(np.datetime64(now) - np.timedelta64(1, "D"))
    rule = EX.SpecRule(EX.ExitSpec(), "week_end", "week_end")
    F = np.stack([x[:len(tr)]], 1)
    m = SC.RealisableGainModel(rule, ("f_x",)).fit(tr, F, now)
    return m, rule


def test_selection_band_is_realisable_gain_under_intended_policy():
    m, rule = _gain_model()
    assert m.diagnostics["status"] == "fitted"
    fcs = m.forecast(["low", "mid", "high"], np.array([[0.2], [1.0], [2.0]]), ["trend"] * 3, "2021-01-04")
    rep = SC.select(fcs, "2021-01-04", SC.policy_id(rule))
    assert rep.eligible == ("mid",)                                   # ~+7% realisable
    assert rep.reason_of("low") == SC.Reason.BELOW_BAND               # rises (P(up) high) but only ~+1.4%
    assert rep.reason_of("high") == SC.Reason.ABOVE_BAND              # the most volatile / biggest mover is NOT chosen
    assert rep.funnel["selected"] == 1


def test_selection_refuses_stale_policy_wrong_basis_and_outcome_features():
    m, rule = _gain_model()
    fc = m.forecast(["mid"], np.array([[1.0]]), ["trend"], "2021-01-04")[0]
    other = EX.SpecRule(EX.ExitSpec(target=0.05), "target_5", "fixed_target")
    assert SC.policy_id(other) != SC.policy_id(rule)
    assert SC.eligibility(fc, SC.policy_id(other)).reason == SC.Reason.STALE_POLICY
    assert SC.eligibility(dataclasses.replace(fc, basis="p_up"), SC.policy_id(rule)).reason == SC.Reason.WRONG_QUANTITY
    assert SC.eligibility(dataclasses.replace(fc, quantity="p_up"), SC.policy_id(rule)).reason == SC.Reason.WRONG_QUANTITY
    with pytest.raises(FirewallBreach):
        SC.RealisableGainModel(rule, ("fwd_ret_5d",)).fit(*make_paths(n_weeks=10)[:1], np.zeros((80, 1)), "2021-01-01")
    with pytest.raises(FirewallBreach):                               # a forecast from the future
        SC.select([dataclasses.replace(fc, as_of="2021-02-01")], "2021-01-04", SC.policy_id(rule))


def test_selection_training_must_be_matured_and_band_is_fixed():
    p, x = make_paths(n_weeks=30, drift_x=0.07)
    rule = EX.SpecRule(EX.ExitSpec(), "week_end", "week_end")
    with pytest.raises(FirewallBreach):
        SC.RealisableGainModel(rule, ("f_x",)).fit(p, x[:, None], str(p.end.max()))    # last positions end ON now
    assert SC.SelectionConfig(lo=0.03).validate()                     # the band is the trading constraint, not a knob


def test_selection_empty_and_abstain():
    rep = SC.select([], NOW, "X")
    assert rep.eligible == () and rep.funnel["selected"] == 0
    p, x = make_paths(n_weeks=3, drift_x=0.07)
    rule = EX.SpecRule(EX.ExitSpec(), "week_end", "week_end")
    m = SC.RealisableGainModel(rule, ("f_x",)).fit(p, x[:, None], "2021-01-01")
    fc = m.forecast(["a"], np.array([[1.0]]), ["trend"], "2021-01-04")[0]
    assert SC.eligibility(fc, SC.policy_id(rule)).reason == SC.Reason.UNSUPPORTED
    kept, dropped = SC.apply_to_positions(pd.DataFrame({"ticker": ["a", "b"]}), rep)
    assert kept.empty and list(dropped.reason) == ["UNSUPPORTED", "UNSUPPORTED"]


def test_identification_curve_and_basis_audit():
    m, rule = _gain_model()
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 2, 400)
    fcs, real, at = [], {}, {}
    for i, xv in enumerate(xs):
        d = str((pd.Timestamp("2021-01-04") + pd.Timedelta(days=int(i // 4) * 7)).date())
        fcs.append(m.forecast([f"c{i}"], np.array([[xv]]), ["trend"], d)[0])
        real[f"c{i}"], at[f"c{i}"] = 0.07 * xv + rng.normal(0, 0.01), d
    now = "2023-01-01"
    curve = SC.identification_curve(fcs, real, at, now, SC.policy_id(rule))
    assert len(curve) >= 3 and (curve["precision"] > curve["base_rate"]).all()     # the constraint finds the band
    tr = SC.improvement_trend(curve)
    assert tr["n_periods"] == len(curve) and tr["slope"] is not None
    aud = SC.basis_audit(fcs, p_up={f"c{i}": float(x > 0.1) + x for i, x in enumerate(xs)}, intended_policy=SC.policy_id(rule))
    assert aud["p_up"]["top_quintile_eligible_share"] == 0.0         # highest P(up) names are above the band, not selected
    assert SC.identification_curve([], {}, {}, now, "X").empty


# ================================================================================================ N: learned exit
def test_learned_exit_beats_holding_on_planted_fade_and_never_on_trend():
    p, _ = make_paths(n_weeks=60, seed=2)
    rule = ER.LearnedExitRule().fit(p.until("2020-02-01"), now="2020-02-15")
    test = p.take(np.flatnonzero(p.week > np.datetime64("2020-02-15")))
    learned = rule.run(test)
    base = EX.run_exit(test, EX.ExitSpec())
    fade = test.kind == "fade"
    assert learned.net[fade].mean() > base.net[fade].mean() + 0.02    # sells the fade after its jump
    assert (learned.days[fade] < test.D).mean() > 0.8
    assert (learned.days[~fade] == test.D).mean() > 0.8              # holds the trend
    assert rule.threshold > -math.inf and rule.selection_table is not None


def test_exit_decision_at_close_uses_no_future_bars():
    p, _ = make_paths(n_weeks=40, seed=4)
    rule = ER.LearnedExitRule().fit(p.until("2019-08-01"), now="2019-08-05")
    test = p.take(np.flatnonzero(p.week > np.datetime64("2019-08-05")))
    for d in range(test.D - 1):
        a = ER.decide_at_close(rule, test, d)
        rng = np.random.default_rng(d)
        s = dataclasses.replace(test)
        for k in ("o", "h", "l", "c"):
            arr = getattr(test, k).copy()
            arr[:, d + 1:] = arr[:, d + 1:] * rng.uniform(0.5, 1.5, arr[:, d + 1:].shape)
            setattr(s, k, arr)
        s.h = np.maximum(s.h, np.maximum(s.o, s.c))
        s.l = np.minimum(s.l, np.minimum(s.o, s.c))
        b = ER.decide_at_close(rule, s, d)
        pd.testing.assert_frame_equal(a, b)


def test_policy_always_sells_below_threshold_and_fills_next_open():
    p, _ = make_paths(n_weeks=30, seed=5)
    m = ER.ExitValueModel().fit(p.until("2019-06-01"), now="2019-06-03")
    test = p.take(np.arange(20))
    res, tr = ER.simulate(test, m, math.inf)                         # EV is always below +inf: must sell at the first chance
    assert (res.days == 2).all() and (res.reason == ER.R_DECIDED).all()
    exp = test.o[:, 1] * (1 - 5e-4) / (test.o[:, 0] * (1 + 5e-4)) - 1 - 2 * 2e-4
    assert np.allclose(res.net, exp)
    res2, _ = ER.simulate(test, m, -math.inf)                        # never below: identical to the week-end exit
    assert np.allclose(res2.net, EX.run_exit(test, EX.ExitSpec()).net)


def test_exit_model_refuses_unmatured_and_anchor_inputs_and_handles_empty():
    p, _ = make_paths(n_weeks=10)
    with pytest.raises(FirewallBreach):
        ER.ExitValueModel().fit(p, now=str(p.end.max()))
    with pytest.raises(FirewallBreach):
        ER.state_features(p, 0, {"predicted_gain": np.zeros(len(p))})
    with pytest.raises(FirewallBreach):
        ER.state_features(p, 0, {"target_distance": np.zeros(len(p))})
    X, names = ER.state_features(p, 0, {"m_regime": np.ones(len(p))})
    assert "m_regime" in names and X.shape == (len(p), len(ER.STATE_FEATURES) + 1)
    empty = p.take(np.array([], int))
    res, _ = ER.simulate(empty, ER.ExitValueModel().fit(p.until("2019-02-20")), 0.0)
    assert len(res.net) == 0
    assert ER.LearnedExitRule().fit(empty).model is None               # abstains -> behaves as holding


def test_trajectory_expected_best_exit_known_at_entry():
    p, _ = make_paths(n_weeks=40, seed=6)
    tm = ER.TrajectoryModel().fit(p.until("2019-08-01"), now="2019-08-05")
    test = p.take(np.flatnonzero(p.week > np.datetime64("2019-08-05")))
    best = tm.expected_best_exit(test)
    fade = test.kind == "fade"
    assert (best["best_day"][fade] == 1).all() and (best["best_day"][~fade] == test.D).all()
    err = ER.best_exit_error(test, tm, "2021-01-01")
    assert err["mean_abs_days"] < 1.5
    reg = ER.exit_regret(test, EX.run_exit(test, EX.ExitSpec()), "2021-01-01")
    assert (reg["regret"] >= -1e-9).all()
    with pytest.raises(FirewallBreach):
        ER.exit_regret(test, EX.run_exit(test, EX.ExitSpec()), "2019-09-01")


def test_learned_exit_enters_existing_tournament():
    p, _ = make_paths(n_weeks=40, per_week=6, seed=7)
    sel_rule, sel = EX.select_rule(p, [EX.SpecRule(EX.ExitSpec()), ER.LearnedExitRule()], min_weeks=10, min_trades=40, boot=60)
    assert sel.table is not None and "learned_ev" in set(sel.table["rule"])
    assert not ER.feature_decay(ER.LearnedExitRule().fit(p)).empty


def test_path_model_feeds_checklist_a_expectations_into_the_ledger():
    from engine.research import expectations as EXP
    from engine.research.two_stage import DayDecision, Funnel
    assert ER.TRAJECTORY_FIELDS == EXP.TRAJECTORY_FIELDS
    p, _ = make_paths(n_weeks=40, seed=11)
    rule = ER.LearnedExitRule().fit(p.until("2019-08-01"), now="2019-08-05")
    pm = ER.PathModel(rule).fit(p.until("2019-08-01"), now="2019-08-05")
    with pytest.raises(FirewallBreach):                              # fitted later than the decision it would serve
        pm.bind({"AAA": {"vol": 0.012, "atr": 0.015, "kind": "fade"}}, "2019-08-01")
    with pytest.raises(FirewallBreach):
        pm.bind({"AAA": {"vol": 0.012, "atr": 0.015, "predicted_gain": 0.07}}, "2019-08-09")
    pm.bind({"AAA": {"vol": 0.012, "atr": 0.015, "kind": "fade"}, "CCC": {"vol": 0.012, "atr": 0.015, "kind": "trend"}}, "2019-08-09")
    idx = pd.MultiIndex.from_tuples([("2019-08-09", "AAA"), ("2019-08-09", "BBB"), ("2019-08-09", "CCC")])
    table = pd.DataFrame({"eligible": True, "p_move": [0.8, 0.7, 0.75], "mover": [True, True, True], "p_up": [0.7, 0.5, 0.66],
                          "p_down": [0.3, 0.5, 0.34], "side": [1, 0, 1], "reason": ["POSITION", "LOW_CONFIDENCE", "POSITION"]}, index=idx)
    f = Funnel()
    f.add("positions", 3, 2, "abstain on the rest")
    day = DayDecision("2019-08-09", table, f, {"open": True, "verdict": "OPEN"}, "dig")
    ctx = dict(entry_at="2019-08-12", timestamp="2019-08-09T16:00:00", market_regime="UP_CALM", sector_regime="UP",
               patterns=("mom",), pattern_strengths={"mom": 0.5}, interactions={})
    today = pd.DataFrame({"vol20": [0.012, 0.02, 0.012]}, index=idx)
    exps = EXP.expectations_from_day(day, pm, ctx, "2019-08-09", model_version="m1", information_set={"prices": "2019-08-09"}, today=today)
    led = EXP.ExpectationLedger(code_hash="t")
    ids = [led.record(e, "2019-08-09") for e in exps]
    assert len(ids) == 2 and led.verify()["ok"]
    a, c = (led.get(i) for i in ids)
    assert a.time_to_peak == 1 and c.time_to_peak == p.D             # the fade peaks on day 1, the trend at the end
    assert a.holding_period < c.holding_period                        # the learned exit is expected to leave the fade early
    assert a.exit_window[0] <= a.holding_period <= a.exit_window[1]
    book = CT.CommitmentBook()                                       # the frozen expectation is what the +-1pp grading commits to
    cm = book.commit(ids[0], a, rule.policy_id, "2019-08-01", "2019-08-16", "2019-08-09")
    assert cm.expectation_digest == led.meta(ids[0])["content_hash"] and cm.predicted == a.predicted_return
    before = rule.run(p.take(np.arange(10))).net.copy()
    CT.Target(tolerance=0.05)                                        # the path model and targets leave the exit policy untouched
    assert np.array_equal(before, rule.run(p.take(np.arange(10))).net)


# ================================================================================================ O: exit independence
def _pipeline(p, rule, target):
    """Full pipeline under one evaluation target: exits, then commitments and grading. Only the grading may depend on the target."""
    res = rule.run(p)
    book = CT.CommitmentBook()
    for i in range(len(p)):
        book.commit(f"p{i}", {"predicted_realisable": 0.07}, rule.policy_id, "2019-01-01", str(p.end[i]), str(p.week[i]), target)
    CT.evaluate(book, [CT.exit_record_from_policy(f"p{i}", str(p.end[i]), float(res.net[i]), rule.policy_id) for i in range(len(p))],
                "2022-01-01", target)
    return res


def test_changing_the_calibration_target_leaves_every_exit_identical():
    p, _ = make_paths(n_weeks=40, seed=8)
    rule = ER.LearnedExitRule().fit(p.until("2019-06-01"))
    test = p.take(np.flatnonzero(p.week > np.datetime64("2019-06-10"))[:40])
    targets = [CT.Target(), CT.Target(tolerance=0.05), CT.Target(tolerance=0.001, goal_share=0.5)]
    rep = ER.exit_independence_audit(lambda t: _pipeline(test, rule, t), targets)
    assert rep.identical and rep.n_targets == 3


def test_independence_audit_catches_a_target_aware_exit():
    p, _ = make_paths(n_weeks=10, seed=9)

    def gaming(t):                                                    # holds longer when the tolerance is tight: forbidden
        return EX.run_exit(p, EX.ExitSpec(hold_days=5 if t.tolerance < 0.02 else 3))
    rep = ER.exit_independence_audit(gaming, [CT.Target(), CT.Target(tolerance=0.05)])
    assert not rep.identical and "days" in rep.fields_differing


# ================================================================================================ P + M: honest +-1pp and anti-gaming
def _book(n=300, hit_share=0.95, seed=0, oos=True):
    rng = np.random.default_rng(seed)
    book = CT.CommitmentBook()
    book.register_target(CT.DEFAULT_TARGET, "2018-01-01")
    outs = []
    days = pd.bdate_range("2019-01-01", periods=n)
    for i in range(n):
        rec = str(days[i].date())
        c = book.commit(f"p{i}", {"predicted_realisable": 0.07}, "POL", "2018-12-31" if oos else rec, str((days[i] + pd.Timedelta(days=7)).date()), rec,
                        in_sample=not oos)
        err = rng.uniform(-0.009, 0.009) if rng.random() < hit_share else rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)
        outs.append(CT.exit_record_from_policy(c.pred_id, (days[i] + pd.Timedelta(days=7)).date(), 0.07 + err, "POL"))
    return book, outs


def test_target_measured_honestly_achieved_and_not_achieved():
    book, outs = _book(400, 0.97)
    r = CT.evaluate(book, outs, "2021-01-01")
    assert r.status == CT.Status.ACHIEVED and r.all_.n == 400 and r.oos.lo >= 0.8
    book2, outs2 = _book(400, 0.6, seed=1)
    r2 = CT.evaluate(book2, outs2, "2021-01-01")
    assert r2.status == CT.Status.NOT_ACHIEVED and 0.5 < r2.all_.share < 0.7
    assert "NOT achieved" in r2.headline and f"{r2.all_.share:.1%}" in r2.headline
    small, so = _book(50, 1.0)
    assert CT.evaluate(small, so, "2021-01-01").status == CT.Status.INSUFFICIENT_SAMPLE
    assert CT.evaluate(CT.CommitmentBook(), [], "2021-01-01").status == CT.Status.EMPTY
    assert CT.required_n(0.9, 0.8) is not None and CT.required_n(0.7, 0.8) is None


def test_suppressed_losers_count_as_misses():
    book, outs = _book(300, 0.97)
    losers = [o for o in outs if abs(o.realised - 0.07) > 0.01]
    kept = [o for o in outs if o not in losers]
    r = CT.evaluate(book, kept, "2021-01-01")
    assert r.status == CT.Status.COMPROMISED and r.n_unresolved == len(losers)
    assert {a.kind for a in r.abuses} == {CT.AbuseKind.SUPPRESSED}
    assert r.all_.share == pytest.approx(len(kept) / 300 * 1.0, abs=0.02) and r.resolved_only.share > r.all_.share


@pytest.mark.parametrize("abuse", ["held_longer", "delayed", "worse_exit", "not_policy", "redefined_value", "redefined_digest",
                                   "policy_mismatch", "uncommitted", "future_outcome"])
def test_each_outcome_abuse_is_detected(abuse):
    book, outs = _book(120, 0.9)
    o = outs[5]
    later = str((pd.Timestamp(o.exit_date) + pd.Timedelta(days=3)).date())
    bad = {"held_longer": dataclasses.replace(o, exit_date=later),
           "delayed": dataclasses.replace(o, fill_lag_sessions=3),
           "worse_exit": dataclasses.replace(o, realised=0.0705, policy_realised=0.03),
           "not_policy": dataclasses.replace(o, decided_by="manual"),
           "redefined_value": dataclasses.replace(o, predicted_as_reported=0.03),
           "redefined_digest": dataclasses.replace(o, expectation_digest_now="deadbeef"),
           "policy_mismatch": dataclasses.replace(o, exit_policy_id="OTHER"),
           "uncommitted": dataclasses.replace(o, pred_id="ghost"),
           "future_outcome": dataclasses.replace(o, exit_date="2030-01-01")}[abuse]
    r = CT.evaluate(book, outs[:5] + [bad] + outs[6:], "2021-01-01")
    assert r.status == CT.Status.COMPROMISED and "NOT a valid result" in r.headline
    if abuse == "worse_exit":
        assert any("TOWARD the prediction" in a.detail for a in r.abuses)
    clean = CT.evaluate(book, outs, "2021-01-01")
    assert not clean.abuses                                           # the null case: honest outcomes raise nothing


def test_cherry_picking_and_target_change_are_refused():
    book, outs = _book(120, 0.9)
    good = [o.pred_id for o in outs if abs(o.realised - 0.07) <= 0.01]
    r = CT.evaluate(book, outs, "2021-01-01", sample=good)
    assert CT.AbuseKind.CHERRY_PICKED in {a.kind for a in r.abuses}
    wide = CT.Target(tolerance=0.05)
    book.register_target(wide, "2020-12-01")                          # registered after seeing results
    r2 = CT.evaluate(book, outs, "2021-01-01", target=wide)
    assert CT.AbuseKind.TARGET_CHANGED in {a.kind for a in r2.abuses} and r2.status == CT.Status.COMPROMISED
    with pytest.raises(FileExistsError):
        book.commit("p1", {"predicted_realisable": 0.05}, "POL", "2018-01-01", "2019-01-10", "2019-01-03")
    with pytest.raises(FirewallBreach):
        book.commit("new", {"predicted_realisable": 0.05}, "POL", "2019-01-03", "2019-01-10", "2019-01-03")
    with pytest.raises(ValueError):                                   # a cohort must exist before its members
        book.commit("c0", {"predicted_realisable": 0.05}, "POL", "2019-01-01", "2019-01-10", "2019-01-03", cohort="calm")
    book.register_cohort("calm", "regime=calm", "2018-01-01")
    for i in range(3):
        book.commit(f"c{i}", {"predicted_realisable": 0.07}, "POL", "2018-12-31", "2019-01-10", "2019-01-03", cohort="calm")
    ok = CT.evaluate(book, [CT.exit_record_from_policy(f"c{i}", "2019-01-10", 0.07, "POL") for i in range(3)], "2021-01-01",
                     cohort="calm")
    assert CT.AbuseKind.CHERRY_PICKED not in {a.kind for a in ok.abuses} and ok.all_.n == 3


def test_oos_split_and_naive_comparator_and_trend():
    book, outs = _book(300, 0.9, oos=False)                           # model trained through the decision day itself: not OOS
    r = CT.evaluate(book, outs, "2021-01-01")
    assert r.oos.n == 0 and r.status == CT.Status.INSUFFICIENT_SAMPLE
    assert r.naive_share is not None
    t = CT.trend(book, outs, "2021-01-01")
    assert t["n"].sum() == 300
    prof = CT.tolerance_profile(book, outs, "2021-01-01")
    assert prof["share"].is_monotonic_increasing


# ================================================================================================ Q: self-correcting network
def error_frame(n=600, seed=0, plant=None, start="2014-01-06"):
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, periods=n, freq="B")[: n]
    f1, f2, u1, m1 = (rng.normal(0, 1, n) for _ in range(4))
    regime = np.where(rng.random(n) < 0.5, "calm", "stress")
    pred = 0.05 + 0.01 * f1
    real = pred + rng.normal(0, 0.01, n)
    recent = np.arange(n) >= int(0.7 * n)
    if plant == "missing":
        real = real + np.where(recent, 0.02 * u1, 0.0)
    if plant == "regime":
        real = real + np.where(recent & (regime == "stress"), -0.02, 0.0)
    if plant == "worse":
        real = real + np.where(recent, rng.normal(0, 0.03, n), 0.0)
    if plant == "u_everywhere":
        real = real + 0.015 * u1
    return pd.DataFrame({"date": d, "matured_at": d + pd.Timedelta(days=7), "predicted": pred, "realised": real, "f_1": f1, "f_2": f2,
                         "u_1": u1, "m_1": m1, "regime": regime})


def test_deterioration_detected_forward_only():
    now = "2016-06-01"
    f = SF.as_of(error_frame(plant="worse"), now)
    det = SF.detect_deterioration(f, now)
    assert det.detected and det.recent_mae > det.reference_mae
    null = SF.detect_deterioration(SF.as_of(error_frame(seed=1), now), now)
    assert not null.detected
    full = error_frame(plant="worse", n=900)
    scr = full.copy()
    late = pd.to_datetime(scr["matured_at"]).dt.date >= pd.Timestamp(now).date()
    scr.loc[late, "realised"] = np.random.default_rng(9).normal(0, 1, int(late.sum()))
    assert SF.detect_deterioration(SF.as_of(full, now), now) == SF.detect_deterioration(SF.as_of(scr, now), now)
    with pytest.raises(FirewallBreach):
        SF.detect_deterioration(full, now)


def test_diagnose_names_the_planted_component_and_nothing_on_null():
    now = "2017-01-01"
    miss = SF.diagnose(SF.as_of(error_frame(plant="missing"), now), now)
    assert SF.Component.MISSING_FEATURE in miss.implicated
    reg = SF.diagnose(SF.as_of(error_frame(plant="regime"), now), now)
    assert SF.Component.REGIME_RECOGNITION in reg.implicated
    null = SF.diagnose(SF.as_of(error_frame(seed=3), now), now)
    assert null.implicated == ()
    assert SF.Component.VOLATILITY_ESTIMATION in null.untested and SF.Component.TIMING in null.untested   # unknown stays unknown
    empty = SF.diagnose(pd.DataFrame(columns=list(SF.REQUIRED)), now)
    assert set(empty.untested) == set(SF.Component) and not empty.deterioration.detected


def _fix_u(name="use_u1", lag=0, avail=Availability.KNOWN_BEFORE_EVENT):
    def build(train, seed):
        X = np.stack([np.ones(len(train)), train["f_1"], train["u_1"]], 1)
        b, *_ = np.linalg.lstsq(X, train["realised"].to_numpy(float), rcond=None)
        return lambda fr: np.stack([np.ones(len(fr)), fr["f_1"], fr["u_1"]], 1) @ b
    return SF.CandidateFix(name, SF.Component.MISSING_FEATURE, (SF.FixInput("f_1"), SF.FixInput("u_1", lag, avail)), build)


def _fix_noise():
    def build(train, seed):
        return lambda fr: fr["predicted"].to_numpy(float) + np.random.default_rng(seed).normal(0, 0.005, len(fr))
    return SF.CandidateFix("noise", SF.Component.FEATURE_WEIGHTING, (SF.FixInput("f_1"),), build)


def test_fixes_are_tested_independently_and_promoted_only_through_the_gate():
    now = "2021-06-01"
    frame = SF.as_of(error_frame(n=1500, plant="u_everywhere", start="2014-01-06"), now)
    fixes = [_fix_u(), _fix_noise()]
    res = SF.test_fixes(fixes, frame, now)
    good, noise = res
    assert good.mean_effect > 0.003 and good.t > 5 and noise.mean_effect < 0
    assert good.reruns[0][1] == good.reruns[1][1]                    # same seed, same number
    base, pol = QG.reference_evidence(now)
    rep = SF.step(frame, fixes, now, base=base, policy=pol)
    assert "use_u1" in rep.promoted and "noise" not in rep.promoted
    assert rep.rejected["noise"].startswith("FAILED")
    bare = SF.step(frame, [_fix_u()], now)                           # no leak / identity / replication audits: never a pass
    assert bare.promoted == () and bare.rejected["use_u1"].split(" ")[0] in ("NEEDS_MORE_EVIDENCE", "QUARANTINED", "UNKNOWN")
    leaky = SF.step(frame, [_fix_u("leaky", lag=-3)], now, base=base, policy=pol)   # input known only after the decision
    assert leaky.promoted == () and leaky.rejected["leaky"].startswith("QUARANTINED")
    assert "fix use_u1" in rep.summary()


def test_fix_testing_empty_and_targeting():
    with pytest.raises(ValueError):
        SF.test_fix(_fix_u(), pd.DataFrame(columns=list(SF.REQUIRED)), NOW)
    with pytest.raises(ValueError):
        SF.test_fixes([_fix_u(), _fix_u()], error_frame(), "2030-01-01")
    diag = SF.diagnose(SF.as_of(error_frame(plant="missing"), "2017-01-01"), "2017-01-01")
    order = SF.targeted(diag, [_fix_noise(), _fix_u()])
    assert order[0].component == SF.Component.MISSING_FEATURE
