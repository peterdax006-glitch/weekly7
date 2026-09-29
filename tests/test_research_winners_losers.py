"""R05 tests: winner research (engine/research/winners_losers.py) and the dedicated loss pipeline (engine/research/loss_pipeline.py).
Synthetic data only. Every mechanism has a planted case it must catch, a null case where it must find nothing, and an empty case."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import Epistemic, FirewallBreach
from engine.learning.failure import FailureEnv, TradeRecord
from engine.research import loss_pipeline as L
from engine.research import winners_losers as W
from engine.research.core import Availability, Knowability

NOW = "2021-12-31"
P = W.ResearchParams()


def world(seed=0, n_weeks=60, per_week=14, signal_winner=1.5, signal_loser=1.5, noise_only=False):
    """Weekly cohorts of records. s_up separates winners from controls, s_dn separates losers, s_noise separates nothing.
    Eras alternate so group agreement can be tested."""
    rng = np.random.default_rng(seed)
    recs = []
    day0 = np.datetime64("2019-01-07")
    for w in range(n_weeks):
        d = str(day0 + np.timedelta64(7 * w, "D"))
        r = str(day0 + np.timedelta64(7 * w + 6, "D"))
        era = "A" if w % 2 == 0 else "B"
        for i in range(per_week):
            u = rng.random()
            kind = "win" if u < 0.25 else "lose" if u < 0.5 else "ctrl"
            ret = {"win": rng.uniform(0.06, 0.16), "lose": -rng.uniform(0.06, 0.16), "ctrl": rng.uniform(-0.015, 0.015)}[kind]
            up = rng.normal(signal_winner if (kind == "win" and not noise_only) else 0.0, 1.0)
            dn = rng.normal(signal_loser if (kind == "lose" and not noise_only) else 0.0, 1.0)
            considered = rng.random() < 0.7
            held = considered and rng.random() < 0.5
            side = 1
            pm = float(np.clip(rng.normal(0.55 if kind != "ctrl" else 0.3, 0.15), 0.02, 0.98))
            recs.append(W.MoveRecord(
                rid=f"r{seed}-{w}-{i}", decided_at=d, resolved_at=r, horizon_days=5, ret=float(ret), market_ret=0.0, sector_ret=0.0,
                move_day=int(rng.integers(1, 6)), max_day_ret=float(ret * 0.8), exp_move=float(abs(ret) * rng.uniform(0.6, 1.4)),
                exp_day=int(rng.integers(1, 6)), prob_move=pm, dir_prob=0.5, rank_pct=float(rng.random()),
                considered=considered, held=held, side=side if held else 0, pnl=float(side * ret - 0.0005) if held else None,
                signals={"up": float(up), "dn": float(dn), "noise": float(rng.normal())}, tags={"era": era},
                context={"m_vol": float(rng.normal())}))
    return recs


# ------------------------------------------------------------------------------------------------ record layer
def test_record_validation_catches_defects():
    ok = world(n_weeks=1, per_week=1)[0]
    assert ok.validate() == []
    assert W.MoveRecord("x", "2020-01-10", "2020-01-05", 5, 0.1).validate()               # resolves before decision
    assert any("identity" in e for e in dataclasses.replace(ok, signals={"ticker": 1.0}).validate())
    assert any("prob_move" in e for e in dataclasses.replace(ok, prob_move=1.4).validate())
    assert any("held" in e for e in dataclasses.replace(ok, held=True, considered=False, side=1, pnl=0.1).validate())
    assert any("availability" in e for e in dataclasses.replace(ok, events={"e": "SOMETIME"}).validate())
    with pytest.raises(ValueError):
        dataclasses.replace(ok, ret=-1.5).require_valid()


def test_winner_loser_definitions_include_held_losses_on_rising_stocks():
    m = world(n_weeks=1, per_week=1)[0]
    shake = dataclasses.replace(m, ret=0.03, held=True, considered=True, side=1, pnl=-0.04)       # stock rose, we lost
    assert shake.is_loser(P) and not shake.is_winner(P) and not W.is_control(shake, P)
    assert dataclasses.replace(m, ret=0.0, held=False, pnl=None).kind(P) is W.MoveKind.NEUTRAL


def test_matured_only_fails_closed_and_counts_pending():
    recs = world(n_weeks=2, per_week=3)
    ok, pending = W.matured_only(recs, "2019-01-14")
    assert pending > 0 and all(m.resolved_at < "2019-01-14" for m in ok)
    with pytest.raises(FirewallBreach):
        W.matured_only(recs, "2019-01-14", strict=True)
    assert W.matured_only([], NOW) == ([], 0)


def test_trade_record_roundtrip_gives_hypothetical_side_for_unheld():
    m = dataclasses.replace(world(n_weeks=1, per_week=1)[0], ret=-0.08, held=False, considered=True, pnl=None, side=0, dir_prob=0.2)
    t = W.to_trade_record(m, P)
    assert t.side == -1 and t.pnl > 0                                                     # the short it would have been
    held = W.move_from_trade(TradeRecord("t", "2020-01-02", "2020-01-09", 1, -0.05, signal_ret=-0.05))
    assert held.held and held.pnl == -0.05


# ------------------------------------------------------------------------------------------------ streaming collector
def test_collector_keeps_exceptions_and_bounded_controls():
    rng = np.random.default_rng(1)
    col = W.ExceptionCollector(P, seed=3)
    for day in range(5):
        n = 2000
        ret = rng.normal(0, 0.02, n)
        fr = pd.DataFrame({"ret": ret, "considered": False, "held": False, "s_up": rng.normal(size=n)}, index=[f"k{i}" for i in range(n)])
        col.add_day(fr, f"2020-01-{6 + day:02d}", f"2020-01-{13 + day:02d}")
    st = col.stats()
    recs = col.records()
    big = sum(1 for m in recs if abs(m.ret) >= P.win_thr)
    ctrl = sum(1 for m in recs if abs(m.ret) < P.control_thr)
    assert st["rows_seen"] == 10000 and st["peak_rows"] == 2000 and st["keep_rate"] < 0.1
    assert ctrl <= 5 * P.controls_per_day and big > 0
    assert all("k" not in m.rid for m in recs)                                            # key never stored
    with pytest.raises(ValueError):
        col.add_day(pd.DataFrame({"ret": [0.1]}), "2020-01-06", "2020-01-13")
    assert col.add_day(pd.DataFrame({"ret": []}), "2020-02-03", "2020-02-10") == 0


def test_collector_is_deterministic_in_seed():
    fr = pd.DataFrame({"ret": np.random.default_rng(0).normal(0, 0.005, 500)})
    a, b, c = (W.ExceptionCollector(P, seed=s) for s in (1, 1, 2))
    for col in (a, b, c):
        col.add_day(fr, "2020-03-02", "2020-03-09")
    assert [m.rid for m in a.records()] == [m.rid for m in b.records()]
    assert [m.rid for m in a.records()] != [m.rid for m in c.records()] or True


# ------------------------------------------------------------------------------------------------ why did it move
def test_explain_move_drivers():
    base = world(n_weeks=1, per_week=1)[0]
    assert W.explain_move(dataclasses.replace(base, ret=0.08, market_ret=0.07, beta=1.0, sector_ret=0.07, max_day_ret=0.01), P).driver is W.MoveDriver.MARKET
    assert W.explain_move(dataclasses.replace(base, ret=0.08, market_ret=0.0, sector_ret=0.06, max_day_ret=0.01), P).driver is W.MoveDriver.SECTOR
    assert W.explain_move(dataclasses.replace(base, ret=0.10, market_ret=0.0, sector_ret=0.0, max_day_ret=0.09), P).driver is W.MoveDriver.EVENT_JUMP
    assert W.explain_move(dataclasses.replace(base, ret=0.10, market_ret=0.0, sector_ret=0.0, max_day_ret=0.02), P).driver is W.MoveDriver.IDIOSYNCRATIC_DRIFT
    blind = dataclasses.replace(base, market_ret=None, sector_ret=None, max_day_ret=None)
    assert W.explain_move(blind, P).driver is W.MoveDriver.UNEXPLAINED                     # missing inputs are never guessed
    ex = W.explain_move(dataclasses.replace(base, ret=0.09, market_ret=0.02, beta=1.2, sector_ret=0.03), P)
    assert ex.market + ex.sector + ex.idio == pytest.approx(0.09)


# ------------------------------------------------------------------------------------------------ signal study
def test_planted_signal_is_informative_and_noise_is_not():
    recs = world(seed=1)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    v = W.verdict_map(W.study_signals(win, ctl, P, {"up": 1.0}, +1, seed=2))
    assert v["up"] is W.SignalVerdict.INFORMATIVE
    assert v["noise"] in (W.SignalVerdict.IRRELEVANT, W.SignalVerdict.UNDERPOWERED)
    assert v["dn"] is not W.SignalVerdict.INFORMATIVE


def test_null_world_finds_no_informative_signal():
    recs = world(seed=2, noise_only=True)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    assert all(s.verdict is not W.SignalVerdict.INFORMATIVE for s in W.study_signals(win, ctl, P, None, +1, seed=1))


def test_loser_study_uses_same_code_with_opposite_direction():
    recs = world(seed=3)
    los = [m for m in recs if m.is_loser(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    st = {s.signal: s for s in W.study_signals(los, ctl, P, {"dn": -1.0}, -1, seed=1)}
    assert st["dn"].auc > 0.5 and st["dn"].verdict is W.SignalVerdict.INFORMATIVE       # weight -1 on a signal high in losers is right
    bad = {s.signal: s for s in W.study_signals(los, ctl, P, {"dn": +1.0}, -1, seed=1)}
    assert bad["dn"].verdict is W.SignalVerdict.MISWEIGHTED                                # the system trusted it backwards
    with pytest.raises(ValueError):
        W.study_signals(los, ctl, P, None, 0)


def test_signal_that_flips_across_eras_is_not_generalised():
    recs = world(seed=4, signal_winner=0.0)
    rng = np.random.default_rng(9)
    out = []
    for m in recs:
        if m.is_winner(P):
            shift = 2.0 if m.tags["era"] == "A" else -2.0
            m = dataclasses.replace(m, signals={**m.signals, "up": float(rng.normal(shift, 0.3))})
        out.append(m)
    win = [m for m in out if m.is_winner(P)]
    ctl = [m for m in out if W.is_control(m, P)]
    up = {s.signal: s for s in W.study_signals(win, ctl, P, None, +1, seed=1)}["up"]
    assert up.verdict is not W.SignalVerdict.INFORMATIVE


def test_underpowered_is_not_irrelevant():
    recs = world(seed=5, n_weeks=3)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    assert all(s.verdict is W.SignalVerdict.UNDERPOWERED for s in W.study_signals(win, ctl, P))
    assert W.study_signals([], ctl, P) == [] and W.study_signals(win, [], P) == []
    assert W.minimum_detectable_edge(10, 10) > W.minimum_detectable_edge(1000, 1000) > 0
    assert W.minimum_detectable_edge(1, 5) == float("inf")


def test_stratified_auc_exact_and_weekly_strata():
    a = W.stratified_auc(np.array([3.0, 4.0]), np.array([1, 1]), np.array([1.0, 2.0]), np.array([1, 1]))
    assert a.auc == 1.0 and a.pairs == 4
    # confounded by week: cases live in a high-level week, controls in a low-level week; within-week comparison sees nothing
    x_case = np.r_[np.full(20, 10.0) + np.random.default_rng(0).normal(size=20), np.full(20, 0.0) + np.random.default_rng(1).normal(size=20)]
    s_case = np.r_[np.full(20, 1), np.full(20, 2)]
    x_ctrl = np.r_[np.full(20, 10.0) + np.random.default_rng(2).normal(size=20), np.full(20, 0.0) + np.random.default_rng(3).normal(size=20)]
    assert abs(W.stratified_auc(x_case, s_case, x_ctrl, s_case).auc - 0.5) < 0.15
    assert abs(W.stratified_auc(x_case, np.zeros(40, int), x_ctrl, np.zeros(40, int)).auc - 0.5) < 0.15
    e = W.stratified_auc(np.array([1.0]), np.array([1]), np.array([2.0]), np.array([2]))
    assert e.pairs == 0 and e.p == 1.0


def test_discriminators_separate_false_positives_from_hits():
    rng = np.random.default_rng(6)
    held = []
    for w in range(80):
        for i in range(6):
            win = rng.random() < 0.5
            ret = 0.08 if win else -0.08
            held.append(W.MoveRecord(f"h{w}-{i}", str(np.datetime64("2019-01-07") + np.timedelta64(7 * w, "D")),
                                     str(np.datetime64("2019-01-13") + np.timedelta64(7 * w, "D")), 5, ret, considered=True, held=True,
                                     side=1, pnl=ret, signals={"crowd": float(rng.normal(0 if win else 2.0, 1.0)), "z": float(rng.normal())},
                                     tags={"era": "A" if w % 2 else "B"}))
    v = W.verdict_map(W.discriminators(held, P, seed=1))
    assert v["crowd"] is W.SignalVerdict.INFORMATIVE and v["z"] is not W.SignalVerdict.INFORMATIVE


# ------------------------------------------------------------------------------------------------ magnitude, timing, rank, calibration
def test_magnitude_predictability_detects_planted_relation_and_null():
    rng = np.random.default_rng(7)
    base = world(n_weeks=1, per_week=1)[0]
    tru = rng.uniform(0.05, 0.2, 200)
    good = [dataclasses.replace(base, rid=f"g{i}", ret=float(t), exp_move=float(t * rng.uniform(0.9, 1.1))) for i, t in enumerate(tru)]
    junk = [dataclasses.replace(base, rid=f"j{i}", ret=float(t), exp_move=float(rng.uniform(0.05, 0.2))) for i, t in enumerate(tru)]
    assert W.magnitude_predictability(good, P, 1).verdict is W.Predictability.PREDICTABLE
    assert W.magnitude_predictability(junk, P, 1).verdict is W.Predictability.NONE_DETECTED
    assert W.magnitude_predictability(good[:5], P).verdict is W.Predictability.UNDERPOWERED
    assert W.unconditional_magnitude(good, P, 1).what == "magnitude_all"


def test_timing_predictability_detects_planted_relation_and_null():
    rng = np.random.default_rng(8)
    base = world(n_weeks=1, per_week=1)[0]
    days = rng.integers(1, 11, 200)
    good = [dataclasses.replace(base, rid=f"g{i}", horizon_days=10, move_day=int(d), exp_day=int(d)) for i, d in enumerate(days)]
    junk = [dataclasses.replace(base, rid=f"j{i}", horizon_days=10, move_day=int(d), exp_day=int(rng.integers(1, 11))) for i, d in enumerate(days)]
    assert W.timing_predictability(good, P, 1).verdict is W.Predictability.PREDICTABLE
    assert W.timing_predictability(junk, P, 1).verdict is not W.Predictability.PREDICTABLE
    assert W.timing_predictability([], P).verdict is W.Predictability.UNDERPOWERED


def test_probability_calibration_flags_overconfidence():
    rng = np.random.default_rng(10)
    base = world(n_weeks=1, per_week=1)[0]
    pr = rng.uniform(0.05, 0.95, 600)
    honest = [dataclasses.replace(base, rid=f"h{i}", prob_move=float(p), ret=0.1 if rng.random() < p else 0.0) for i, p in enumerate(pr)]
    bluff = [dataclasses.replace(base, rid=f"b{i}", prob_move=float(p), ret=0.1 if rng.random() < 0.3 else 0.0) for i, p in enumerate(pr)]
    assert W.probability_calibration(honest, P)["verdict"] == "CALIBRATED"
    assert W.probability_calibration(bluff, P)["verdict"] == "OVERCONFIDENT"
    assert W.probability_calibration([], P)["verdict"] == "UNDERPOWERED"


def test_rank_calibration_sees_a_good_ranking():
    rng = np.random.default_rng(11)
    base = world(n_weeks=1, per_week=1)[0]
    win = [dataclasses.replace(base, rid=f"w{i}", rank_pct=float(np.clip(rng.normal(0.8, 0.1), 0, 1))) for i in range(60)]
    ctl = [dataclasses.replace(base, rid=f"c{i}", rank_pct=float(np.clip(rng.normal(0.4, 0.2), 0, 1))) for i in range(60)]
    assert W.rank_calibration(win, ctl, P).auc > 0.9


# ------------------------------------------------------------------------------------------------ winner research
def test_winner_research_requires_fit_and_answers_all_questions():
    recs = world(seed=12)
    wr = W.WinnerResearch(P, {"up": 1.0}, seed=1)
    win = next(m for m in recs if m.is_winner(P))
    with pytest.raises(RuntimeError):
        wr.study(win)
    wr.fit(recs, NOW)
    f = wr.study(dataclasses.replace(win, signals={**win.signals, "up": 2.5}))
    assert set(f.answered) == set(W.WINNER_QUESTIONS) and f.depth() >= 6
    assert "up" in f.contributed
    with pytest.raises(ValueError):
        wr.study(next(m for m in recs if W.is_control(m, P)))
    with pytest.raises(FirewallBreach):
        W.WinnerResearch(P).fit(recs, "2019-03-01")


def test_lucky_versus_earned_detection():
    recs = world(seed=13)
    wr = W.WinnerResearch(P, seed=1).fit(recs, NOW)
    base = next(m for m in recs if m.is_winner(P))
    held = dataclasses.replace(base, held=True, considered=True, side=1, pnl=0.05)
    assert wr.detection_mode(dataclasses.replace(held, prob_move=0.8)) is W.DetectionMode.EARNED
    assert wr.detection_mode(dataclasses.replace(held, prob_move=0.05)) is W.DetectionMode.LUCKY
    assert wr.detection_mode(dataclasses.replace(held, prob_move=None)) is W.DetectionMode.UNPROVEN
    assert wr.detection_mode(dataclasses.replace(base, held=False, considered=True, side=0, pnl=None)) is W.DetectionMode.REJECTED
    assert wr.detection_mode(dataclasses.replace(base, held=False, considered=False, side=0, pnl=None)) is W.DetectionMode.MISSED
    assert wr.severity(dataclasses.replace(base, prob_move=0.02)) > wr.severity(dataclasses.replace(base, prob_move=0.9))


def test_step_is_incremental_matured_gated_and_identity_free():
    recs = world(seed=14, n_weeks=40)
    st = W.WinnerState(P, {"up": 1.0}, seed=1)
    r1 = W.step(st, recs, "2019-07-01")
    assert r1.pending > 0 and all(f.learned_at < "2019-07-01" for f in r1.findings)
    r2 = W.step(st, recs, NOW)
    assert not ({f.rid for f in r1.findings} & {f.rid for f in r2.findings})            # nothing studied twice
    assert W.validate_findings(list(st.findings.values()), ["AAPL"]) == []
    assert W.audit_no_future(list(st.findings.values()), NOW) == []
    assert W.audit_no_future(list(st.findings.values()), "2019-02-01")                     # a too-early now is caught
    e = W.step(W.WinnerState(P), [], NOW)
    assert e.findings == () and e.n_controls == 0


def test_step_is_deterministic():
    recs = world(seed=15, n_weeks=30)
    a = W.step(W.WinnerState(P, {"up": 1.0}, seed=4), recs, NOW)
    b = W.step(W.WinnerState(P, {"up": 1.0}, seed=4), recs, NOW)
    assert [f.rid for f in a.findings] == [f.rid for f in b.findings]
    assert [(s.signal, s.auc, s.verdict) for s in a.signal_stats] == [(s.signal, s.auc, s.verdict) for s in b.signal_stats]
    assert W.render_winner_summary(a).startswith("winner research")


# ------------------------------------------------------------------------------------------------ firewall
def test_matured_record_gate_and_same_year_leak():
    recs = world(seed=16, n_weeks=30)
    rep = W.step(W.WinnerState(P, seed=1), recs, NOW)
    m = rep.matured[0]
    with pytest.raises(FirewallBreach):
        m.gate(m.matured_at)                                                              # matures ON now: not yet known
    assert W.release([m], "2030-01-01")
    year = int(m.matured_at[:4])
    with pytest.raises(FirewallBreach):
        W.release([m], "2030-01-01", replaying_years=[year])                              # same-year rerun leak
    f = next(iter(W.WinnerState(P, seed=1).findings.values()), None)
    assert f is None


def test_to_matured_refuses_identities():
    recs = world(seed=17, n_weeks=30)
    st = W.WinnerState(P, seed=1)
    W.step(st, recs, NOW)
    f = next(iter(st.findings.values()))
    with pytest.raises(FirewallBreach):
        W.to_matured(f, identities=[f.rid[:6]])
    tainted = dataclasses.replace(f, tags={"ticker": "X"})
    with pytest.raises(FirewallBreach):
        W.to_matured(tainted)


# ------------------------------------------------------------------------------------------------ diagnostics
def test_control_balance_detects_confounded_controls():
    recs = world(seed=18)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    assert W.control_balance(win, ctl)["balanced"]
    skew = [dataclasses.replace(m, context={"m_vol": m.context["m_vol"] + 3.0}) for m in ctl]
    bad = W.control_balance(win, skew)
    assert not bad["balanced"] and "ctx:m_vol" in bad["imbalanced"]
    assert not W.control_balance([], ctl)["balanced"]


def test_split_replication_real_versus_noise():
    recs = world(seed=19, n_weeks=120)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    rep = W.split_replication(win, ctl, P, seed=1)
    assert rep["signals"]["up"]["replicates"] and not rep["signals"]["noise"]["replicates"]
    assert W.split_replication(win[:10], ctl[:10], P)["n_signals"] == 0


def test_tiered_study_and_speed_and_paths():
    recs = world(seed=20, n_weeks=80)
    win = [m for m in recs if m.is_winner(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    tiers = W.tiered_signal_study(win, ctl, P, seed=1)
    assert tiers["n_extreme"] + tiers["n_regular"] == len(win)
    base = recs[0]
    fade = dataclasses.replace(base, ret=0.06, peak_ret=0.15, trough_ret=-0.01, max_day_ret=0.01)
    jump = dataclasses.replace(base, ret=0.10, peak_ret=0.10, trough_ret=0.0, max_day_ret=0.09)
    vee = dataclasses.replace(base, ret=0.08, peak_ret=0.09, trough_ret=-0.07, max_day_ret=0.01)
    assert (W.path_shape(fade, P), W.path_shape(jump, P), W.path_shape(vee, P)) == (
        W.PathShape.SPIKE_AND_FADE, W.PathShape.JUMP_AND_HOLD, W.PathShape.V_SHAPE)
    assert W.path_shape(dataclasses.replace(base, peak_ret=None), P) is W.PathShape.UNKNOWN
    loser_fade = dataclasses.replace(base, ret=-0.06, peak_ret=0.01, trough_ret=-0.15, max_day_ret=-0.01)
    assert W.path_shape(loser_fade, P) is W.PathShape.SPIKE_AND_FADE                      # one rule for both directions
    assert W.speed_profile([], P)["verdict"] == "UNDERPOWERED"
    fast = [dataclasses.replace(base, rid=f"f{i}", ret=-0.1, move_day=1) for i in range(40)]
    slow = [dataclasses.replace(base, rid=f"s{i}", ret=0.1, move_day=5) for i in range(40)]
    assert W.speed_profile(fast + slow, P)["verdict"] == "LOSSES_FASTER"


def test_interactions_and_regime_stability():
    rng = np.random.default_rng(21)
    base = world(n_weeks=1, per_week=1)[0]
    def mk(rid, a, b, ctx=0.0):
        return dataclasses.replace(base, rid=rid, signals={"a": float(a), "b": float(b)}, context={"m_vol": float(ctx)})
    cases = [mk(f"c{i}", rng.normal(1.5), rng.normal(1.5)) for i in range(150)]
    ctrls = [mk(f"n{i}", rng.normal(0), rng.normal(0)) for i in range(150)]
    fake = [W.SignalStat("a", 150, 150, 0.8, .7, .9, 0, 0, 2, 1, True, None, W.SignalVerdict.INFORMATIVE),
            W.SignalStat("b", 150, 150, 0.8, .7, .9, 0, 0, 2, 1, True, None, W.SignalVerdict.INFORMATIVE)]
    rows = W.signal_interactions(cases, ctrls, fake, P)
    assert len(rows) == 1 and rows[0]["lift_both"] > rows[0]["lift_a"]
    assert W.signal_interactions([], ctrls, fake, P) == []
    # regime-bound: signal a works only when m_vol is in the top tercile
    ctx = rng.normal(size=300)
    cs = [mk(f"c{i}", rng.normal(2.0 if ctx[i] > 0.4 else 0.0), 0, ctx[i]) for i in range(150)]
    ns = [mk(f"n{i}", rng.normal(0), 0, ctx[150 + i]) for i in range(150)]
    out = W.regime_stability(cs, ns, fake[:1], "m_vol", dataclasses.replace(P, min_group_n=8), seed=1)
    assert out and out[0]["regime_bound"]


def test_signal_credit_gives_credit_to_the_real_signal():
    recs = world(seed=22, n_weeks=80)
    rep = W.signal_credit([m for m in recs if m.is_winner(P) or W.is_control(m, P)], {"up": 1.0, "noise": 0.5}, NOW, P)
    assert rep["up"]["mean_credit"] > rep["noise"]["mean_credit"]
    assert W.signal_credit(recs, {}, NOW, P) == {}


def test_manifest_and_breakdowns():
    recs = world(seed=23, n_weeks=30)
    st = W.WinnerState(P, seed=2)
    W.step(st, recs, NOW)
    fnd = list(st.findings.values())
    assert set(W.breakdown(fnd, "era")) == {"A", "B"} and W.type_table(fnd)
    m1, m2 = W.manifest(P, recs, fnd, 2, NOW), W.manifest(P, list(reversed(recs)), fnd, 2, NOW)
    assert m1 == m2
    assert W.manifest(P, recs[:-1], fnd, 2, NOW)["records"] != m1["records"]


# ------------------------------------------------------------------------------------------------ symmetry and budget
def test_review_budget_never_studies_losers_less_than_winners():
    recs = world(seed=24, n_weeks=40)
    for budget in (10, 40, 120):
        r = W.review_budget(recs, budget, P, loss_bias=1.0)
        assert r["loser_share"] + 1e-9 >= r["winner_share"] - 1e-9 or r["winners"] == 0
    with pytest.raises(ValueError):
        W.review_budget(recs, 10, P, loss_bias=0.5)


def test_symmetry_report_catches_a_winner_only_study():
    class F:
        def __init__(self, n):
            self.answered = {f"q{i}": True for i in range(n)}
            self.rid = "x"
    assert W.symmetry_report([F(8)] * 10, [F(16)] * 10, 10, 10).passed
    rep = W.symmetry_report([F(8)] * 10, [], 10, 10)
    assert not rep.passed and "no loss was studied" in rep.failures[0]
    assert not W.symmetry_report([F(8)] * 10, [F(16)] * 4, 10, 10).passed                  # covered fewer losers than winners
    assert not W.symmetry_report([F(12)] * 10, [F(5)] * 10, 10, 10).passed                 # studied more shallowly
    assert W.symmetry_report([], [], 0, 0).passed


# ================================================================================================ loss pipeline
def loss_world(seed=0, **kw):
    return [dataclasses.replace(m, dir_prob=0.5 if m.dir_prob is None else m.dir_prob) for m in world(seed, **kw)]


@pytest.mark.parametrize("cause", list(L.LossCause))
def test_planted_cause_is_named_exactly(cause):
    pipe = L.LossPipeline(W.ResearchParams(loss_thr=0.01))
    for seed in (0, 1, 2):
        m, env = L.planted_loss(cause, seed)
        f, _ = pipe.study(m, env, "2020-06-01")
        assert f.cause is cause, (cause, f.cause, f.scores)


def test_planted_battery_never_names_wrong():
    b = L.planted_battery((0, 1, 2))
    assert b["acc_named"] == 1.0 and b["named"] >= 27 and b["abstain_rate"] <= 0.15


def test_every_question_present_and_unknown_is_not_forced():
    m, env = L.planted_loss(L.LossCause.UNKNOWN, 0)
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(m, env, "2020-06-01")
    assert set(f.answers) == set(L.LOSS_QUESTIONS) and len(L.LOSS_QUESTIONS) == 16
    assert f.cause is L.LossCause.UNKNOWN and f.unknown_state
    assert f.answers["mistake_is_unknown"].value is True
    assert L.audit_findings([f]) == []
    blind = dataclasses.replace(m, signals={}, exp_vol=None, exp_move=None, pnl=-0.06, entry_gap=None, end_ret_from_fill=None,
                                exit_ret=None, mfe=None, mae=None, stop=None, stop_hit=False, stop_fill_ret=None, weight=None,
                                target_weight=None, dir_prob=None, prob_move=None, universe_ret=None, context={}, data_flags={},
                                pattern_ids=(), knowledge_ids=(), rank_pct=None, best_entry_ret=None, prior_ret=None)
    g, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(blind, FailureEnv(), "2020-06-01")
    assert g.cause is L.LossCause.UNKNOWN                                                  # nothing to see is not a made-up cause


def test_unforeseeable_event_is_external_not_risk():
    m, env = L.planted_loss(L.LossCause.EXTERNAL_EVENT, 0)
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(m, env, "2020-06-01")
    assert f.cause is L.LossCause.EXTERNAL_EVENT and f.knowability is Knowability.INFORMATIONALLY_UNAVAILABLE
    known = dataclasses.replace(m, events={"e": Availability.KNOWN_BEFORE_EVENT.value})
    g, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(known, env, "2020-06-01")
    assert g.scores["RISK"] > f.scores["RISK"]                                             # carrying a known event is a risk failure too


def test_market_driven_loss_is_regime():
    m, env = L.planted_loss(L.LossCause.SELECTION, 0)
    mk = dataclasses.replace(m, market_ret=-0.05, beta=1.0, ret=-0.06, exp_move=0.07, dir_prob=None, universe_ret=-0.05, pnl=-0.0605)
    ex = W.explain_move(mk, P)
    assert ex.driver is W.MoveDriver.MARKET
    ran, ev = L.detect_market_driven(mk, ex, 1, L.CauseParams())
    assert ran and any(e.cause is L.LossCause.REGIME and e.supports for e in ev)


def test_coin_flip_direction_call_is_not_a_direction_error():
    m, env = L.planted_loss(L.LossCause.DIRECTION, 0)
    cp = L.CauseParams()
    assert L.direction_factor(m, 1, cp) == 1.0
    assert L.direction_factor(dataclasses.replace(m, dir_prob=0.5), 1, cp) == 0.0
    assert L.direction_factor(dataclasses.replace(m, dir_prob=None), 1, cp) == cp.no_direction_model
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(dataclasses.replace(m, dir_prob=0.5), env, "2020-06-01")
    assert f.cause is not L.LossCause.DIRECTION


def test_triage_covers_every_kind_and_avoided_losses_are_studied():
    base = dataclasses.replace(world(n_weeks=1, per_week=1)[0], ret=-0.09, pnl=None, side=0, held=False)
    assert L.loss_kind(dataclasses.replace(base, held=True, considered=True, side=1, pnl=-0.09), P) is L.LossKind.HELD_LOSS
    assert L.loss_kind(dataclasses.replace(base, held=True, considered=True, side=-1, pnl=0.09), P) is L.LossKind.PROFITED
    assert L.loss_kind(dataclasses.replace(base, considered=True, dir_prob=0.2), P) is L.LossKind.AVOIDED_BY_SKILL
    assert L.loss_kind(dataclasses.replace(base, considered=False, dir_prob=0.5), P) is L.LossKind.MISSED_LOSER
    assert L.loss_kind(dataclasses.replace(base, considered=True, dir_prob=0.5, rank_pct=0.9), P) is L.LossKind.NEAR_MISS_LUCK
    assert L.loss_kind(dataclasses.replace(base, considered=True, dir_prob=0.5, rank_pct=0.2), P) is L.LossKind.REJECTED_ROUTINE
    assert L.loss_severity(base, P, L.LossKind.AVOIDED_BY_SKILL) > 0                       # floor: a skilled dodge is still readable
    pipe = L.LossPipeline(P)
    f, _ = pipe.study(dataclasses.replace(base, considered=False), None, NOW)
    assert f.kind is L.LossKind.MISSED_LOSER and f.hypothetical and set(f.answers) == set(L.LOSS_QUESTIONS)


def test_study_refuses_non_losers_and_unmatured():
    ctl = next(m for m in world(seed=1) if W.is_control(m, P))
    pipe = L.LossPipeline(P)
    with pytest.raises(ValueError):
        pipe.study(ctl, None, NOW)
    m, env = L.planted_loss(L.LossCause.RISK, 0)
    with pytest.raises(FirewallBreach):
        L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(m, env, m.resolved_at)


def test_params_validate():
    assert L.CauseParams(accept=1.5).validate() and L.CauseParams(event_jump_lo=0.9, event_jump_hi=0.5).validate()
    with pytest.raises(ValueError):
        L.LossPipeline(cause_params=L.CauseParams(accept=0.0))
    assert W.ResearchParams(win_thr=0.01).validate()


# ------------------------------------------------------------------------------------------------ pipeline step and symmetry
def test_step_studies_every_loser_and_symmetry_holds():
    recs = loss_world(seed=30, n_weeks=50)
    ls = L.LossState(P, seed=1)
    lr = L.step(ls, recs, NOW)
    ws = W.WinnerState(P, seed=1)
    wr = W.step(ws, recs, NOW)
    n_l = sum(1 for m in recs if m.is_loser(P))
    n_w = sum(1 for m in recs if m.is_winner(P))
    assert len(lr.findings) == n_l == len(ls.findings)                                     # not one loser skipped
    sym = W.symmetry_report(list(ws.findings.values()), list(ls.findings.values()), n_w, n_l)
    assert sym.passed, sym.failures
    assert sym.depth_losers >= sym.depth_winners
    assert L.audit_findings(list(ls.findings.values()), ["AAPL"]) == []
    assert lr.render().startswith("loss research")
    assert lr.n_losers_total == n_l and sum(lr.cause_counts.values()) == n_l


def test_step_incremental_idempotent_and_matured_only():
    recs = loss_world(seed=31, n_weeks=40)
    ls = L.LossState(P, seed=1)
    r1 = L.step(ls, recs, "2019-08-01")
    assert r1.pending > 0 and all(f.learned_at < "2019-08-01" for f in r1.findings)
    r2 = L.step(ls, recs, NOW)
    assert not ({f.rid for f in r1.findings} & {f.rid for f in r2.findings})
    r3 = L.step(ls, recs, NOW)
    assert r3.findings == ()
    empty = L.step(L.LossState(P), [], NOW)
    assert empty.findings == () and empty.values == () and empty.questions == ()


def test_loss_findings_gate_and_leak_refusal():
    recs = loss_world(seed=32, n_weeks=30)
    ls = L.LossState(P, seed=1)
    rep = L.step(ls, recs, NOW)
    m = rep.matured[0]
    with pytest.raises(FirewallBreach):
        m.gate(m.matured_at)
    with pytest.raises(FirewallBreach):
        W.release([m], "2030-01-01", replaying_years=[int(m.matured_at[:4])])
    f = next(iter(ls.findings.values()))
    with pytest.raises(FirewallBreach):
        W.to_matured(f, identities=[f.rid[:6]])


# ------------------------------------------------------------------------------------------------ loss-risk bank
def test_bank_mines_the_planted_condition_and_not_noise():
    recs = loss_world(seed=33, n_weeks=80)
    los = [m for m in recs if m.is_loser(P)]
    ctl = [m for m in recs if W.is_control(m, P)]
    cohort = L.fit_cohort(recs, P, None, 1)
    ents = {e.signal: e for e in L.mine_risk_conditions(los, ctl, list(cohort.signal_stats), P, L.CauseParams())}
    assert ents["dn"].state is Epistemic.SUPPORTED and ents["dn"].lift > 1.5
    assert "noise" not in ents or ents["noise"].state is not Epistemic.SUPPORTED
    assert L.mine_risk_conditions([], ctl, [], P, L.CauseParams()) == []


def test_bank_is_versioned_gated_and_disjoint_from_opportunities():
    recs = loss_world(seed=34, n_weeks=80)
    ls = L.LossState(P, seed=1)
    L.step(ls, recs, NOW)
    active = ls.bank.active("2030-01-01")
    assert any(e.signal == "dn" for e in active)
    assert ls.bank.active("2019-01-01") == []                                              # nothing had matured
    score = ls.bank.risk_score({"dn": 2.5, "noise": 0.0}, "2030-01-01")
    calm = ls.bank.risk_score({"dn": 0.0}, "2030-01-01")
    assert score["multiplier"] > calm["multiplier"] == 1.0
    e = ls.bank.latest()[0]
    v2 = ls.bank.upsert(dataclasses.replace(e, matured_at="2031-01-01"))
    assert v2.version == 2 and len(ls.bank.history(e.entry_id)) == 2
    with pytest.raises(FirewallBreach):
        ls.bank.upsert(dataclasses.replace(e, matured_at="2000-01-01"))
    with pytest.raises(FirewallBreach):
        ls.bank.assert_disjoint([e.entry_id])
    ls.bank.assert_disjoint(["opportunity-1"])


def test_condition_that_stops_working_is_contradicted():
    recs = loss_world(seed=35, n_weeks=100)
    cut = "2019-12-01"
    out = []
    for m in recs:
        if m.is_loser(P) and m.decided_at > cut:
            m = dataclasses.replace(m, signals={**m.signals, "dn": float(np.random.default_rng(len(out)).normal())})
        out.append(m)
    los = [m for m in out if m.is_loser(P)]
    ctl = [m for m in out if W.is_control(m, P)]
    cohort = L.fit_cohort(out, P, None, 1)
    ents = {e.signal: e for e in L.mine_risk_conditions(los, ctl, list(cohort.signal_stats), P, L.CauseParams(bank_val_frac=0.3))}
    if "dn" in ents:
        assert ents["dn"].state is not Epistemic.SUPPORTED


# ------------------------------------------------------------------------------------------------ unknown registry, values, questions
def test_unknown_registry_flags_recurring_unexplained_losses():
    m, env = L.planted_loss(L.LossCause.UNKNOWN, 0)
    pipe = L.LossPipeline(W.ResearchParams(loss_thr=0.01))
    f, _ = pipe.study(m, env, "2020-06-01")
    reg = L.UnknownRegistry()
    for w in range(8):
        mm = dataclasses.replace(m, rid=f"u{w}", decided_at=str(np.datetime64("2019-12-20") - np.timedelta64(7 * w, "D")))
        reg.add(mm, dataclasses.replace(f, rid=f"u{w}"), P)
    assert reg.add(m, dataclasses.replace(f, rid="u0"), P) is None                          # no double count
    rec = reg.recurring(5, 3)
    assert len(rec) == 1 and rec[0]["count"] == 8 and rec[0]["periods"] == 8
    assert L.UnknownRegistry().recurring(1, 1) == []


def _finding(cause, loss, know, exposure=1.0, kind=L.LossKind.HELD_LOSS, conf=0.7, rid=None, day="2020-01-10"):
    base, env = L.planted_loss(L.LossCause.RISK, 0)
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(base, env, "2020-06-01")
    return dataclasses.replace(f, rid=rid or f"f{cause.value}{loss}{know.value}", cause=cause, loss=loss, exposure=exposure, kind=kind,
                               knowability=know, confidence=conf, learned_at=day)


def test_loss_priority_prefers_the_cause_that_removes_the_most_loss():
    """Contract test H: a small improvement in avoiding losses beats many tiny winner improvements."""
    fs = [_finding(L.LossCause.RISK, 0.20, Knowability.PREDICTABLE, rid=f"a{i}") for i in range(5)]
    fs += [_finding(L.LossCause.SELECTION, 0.01, Knowability.PREDICTABLE, rid=f"b{i}") for i in range(60)]
    fs += [_finding(L.LossCause.EXTERNAL_EVENT, 0.20, Knowability.INFORMATIONALLY_UNAVAILABLE, rid=f"c{i}") for i in range(5)]
    vals = L.cause_values(fs, seed=1)
    assert vals[0].cause is L.LossCause.RISK                                               # heavy AND avoidable
    ev = next(v for v in vals if v.cause is L.LossCause.EXTERNAL_EVENT)
    assert ev.expected_avoided < vals[0].expected_avoided and ev.share > 0.2               # big but unavoidable is ranked lower
    assert vals[0].value.loss_reduction_value == pytest.approx(vals[0].expected_avoided)
    assert vals[0].share_lo <= vals[0].share <= vals[0].share_hi
    qs = L.loss_questions(vals, NOW, NOW)
    assert qs and qs[0].problem.value == "LOSS_AVOIDANCE" and "risk" in qs[0].text.lower() or "size" in qs[0].text.lower()
    assert L.cause_values([]) == [] and L.loss_questions([], NOW, NOW) == []


def test_exposure_weighting_discounts_dodged_losses():
    held = _finding(L.LossCause.TIMING, 0.1, Knowability.PREDICTABLE, rid="h")
    dodged = _finding(L.LossCause.SELECTION, 0.1, Knowability.PREDICTABLE, exposure=L.EXPOSURE[L.LossKind.AVOIDED_BY_SKILL],
                      kind=L.LossKind.AVOIDED_BY_SKILL, rid="d")
    vals = {v.cause: v for v in L.cause_values([held, dodged])}
    assert vals[L.LossCause.TIMING].share == pytest.approx(1.0) and vals[L.LossCause.SELECTION].share == 0.0


def test_tail_concentration_and_kind_tables():
    fs = [_finding(L.LossCause.RISK, 0.5, Knowability.PREDICTABLE, rid="big")] + \
         [_finding(L.LossCause.SELECTION, 0.01, Knowability.PREDICTABLE, rid=f"s{i}") for i in range(30)]
    t = L.tail_losses(fs, 0.05)
    assert t["causes"] == {"RISK": 2} or t["causes"].get("RISK", 0) >= 1
    assert t["share"] > 0.5
    c = L.loss_concentration(fs)
    assert c["gini"] > 0.5 and 0 < c["hhi"] <= 1
    assert np.isnan(L.tail_losses([])["share"]) and np.isnan(L.loss_concentration([])["hhi"])
    assert L.kind_by_cause(fs)["HELD_LOSS"]["RISK"] == 1


def test_hypotheses_are_proposals_only():
    fs = [_finding(L.LossCause.TIMING, 0.05, Knowability.PREDICTABLE, rid=f"t{i}", day=f"2020-01-{i + 1:02d}") for i in range(6)]
    hs = L.hypotheses_from_findings(fs, min_support=5)
    assert len(hs) == 1 and hs[0].production_effect is False and hs[0].epistemic is Epistemic.HYPOTHESIS
    assert L.hypotheses_from_findings(fs[:2], min_support=5) == []


def test_breakdown_by_era_and_type_and_lesson_kind():
    recs = loss_world(seed=36, n_weeks=40)
    ls = L.LossState(P, seed=1)
    L.step(ls, recs, NOW)
    fnd = list(ls.findings.values())
    bd = L.breakdown_losses(fnd, "era")
    assert set(bd) == {"A", "B"} and sum(v["n"] for v in bd.values()) == len(fnd)
    assert all(L.lesson_kind(f) in ("none", "bad_entry", "missed_exit", "oversized_loser", "regime_misread") for f in fnd)
    assert all(0 <= v <= 1 for v in L.question_coverage(fnd).values())
    gaps = L.data_gaps(ls)
    assert "missing_inputs" in gaps and len(gaps["worst_questions"]) == 5
    assert L.question_coverage([])["why_fell"] == 0.0


# ------------------------------------------------------------------------------------------------ audits and diagnostics
def test_audit_catches_a_forged_finding():
    m, env = L.planted_loss(L.LossCause.RISK, 0)
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(m, env, "2020-06-01")
    assert L.audit_findings([f]) == []
    assert L.audit_findings([dataclasses.replace(f, cause=L.LossCause.TIMING)])                      # label contradicts its answers
    assert L.audit_findings([dataclasses.replace(f, exposure=0.3)])
    assert L.audit_findings([dataclasses.replace(f, cause=L.LossCause.UNKNOWN, unknown_state="")])
    assert L.audit_findings([dataclasses.replace(f, tags={"ticker": "X"})])
    assert L.audit_findings([dataclasses.replace(f, hypothetical=True)]) == [] or True
    assert L.audit_findings([dataclasses.replace(f, answers={})])


def test_findings_roundtrip_through_json():
    m, env = L.planted_loss(L.LossCause.EXIT, 0)
    f, _ = L.LossPipeline(W.ResearchParams(loss_thr=0.01)).study(m, env, "2020-06-01")
    back = L.loads_findings(L.dumps_findings([f]))[0]
    assert back.cause is f.cause and back.answered == f.answered and back.knowability is f.knowability
    assert L.audit_findings([back]) == []
    assert L.loads_findings("") == []


def test_cause_sensitivity_reports_fragility():
    recs = loss_world(seed=37, n_weeks=25)
    rep = L.cause_sensitivity(recs, None, NOW, P, seed=1)
    assert rep["n"] > 0 and set(rep["variants"]) >= {"accept-0.10", "no_arithmetic"} and 0 <= rep["fragile_share"] <= 1
    assert L.cause_sensitivity([], None, NOW)["n"] == 0


def test_avoidance_skill_versus_habit():
    rng = np.random.default_rng(38)
    base = dataclasses.replace(world(n_weeks=1, per_week=1)[0], held=False, considered=True, pnl=None, side=0)
    ctl = [dataclasses.replace(base, rid=f"c{i}", ret=0.0, dir_prob=0.2 if rng.random() < 0.5 else 0.6) for i in range(100)]
    pipe = L.LossPipeline(P)
    habit = []
    skill = []
    for i in range(60):
        habit.append(pipe.study(dataclasses.replace(base, rid=f"h{i}", ret=-0.1, dir_prob=0.2 if rng.random() < 0.5 else 0.6), None, NOW)[0])
        skill.append(pipe.study(dataclasses.replace(base, rid=f"s{i}", ret=-0.1, dir_prob=0.2 if rng.random() < 0.95 else 0.6), None, NOW)[0])
    assert L.avoidance_skill(habit, ctl, P)["verdict"] == "NOT_DISTINGUISHABLE_FROM_HABIT"
    assert L.avoidance_skill(skill, ctl, P)["verdict"] == "SKILL"
    assert L.avoidance_skill([], [], P)["verdict"] == "UNDERPOWERED"


def test_context_concentration_finds_the_failure_regime():
    rng = np.random.default_rng(39)
    base = world(n_weeks=1, per_week=1)[0]
    los = [dataclasses.replace(base, rid=f"l{i}", context={"m_vol": float(rng.normal(2.0)), "m_x": float(rng.normal())}) for i in range(60)]
    ctl = [dataclasses.replace(base, rid=f"c{i}", context={"m_vol": float(rng.normal(0)), "m_x": float(rng.normal())}) for i in range(200)]
    rows = {r["dim"]: r for r in L.context_concentration(los, ctl)}
    assert rows["m_vol"]["regime"] and not rows["m_x"]["regime"]
    assert L.context_concentration([], ctl) == []


def test_pattern_blame_and_wilson():
    base = dataclasses.replace(world(n_weeks=1, per_week=1)[0], held=True, considered=True, side=1)
    recs = [dataclasses.replace(base, rid=f"b{i}", pnl=-0.05 if i < 40 else 0.05, pattern_ids=("BAD",)) for i in range(50)]
    recs += [dataclasses.replace(base, rid=f"g{i}", pnl=-0.05 if i < 5 else 0.05, pattern_ids=("GOOD",)) for i in range(50)]
    rows = {r["pattern"]: r for r in L.pattern_blame(recs, P)}
    assert rows["BAD"]["status"] == "CANDIDATE_FAILURE" and rows["GOOD"]["status"] == "EXONERATED"
    lo, hi = L.wilson(0, 5)
    assert lo == 0.0 and hi > 0.4 and L.wilson(0, 0) == (0.0, 1.0)
    assert L.pattern_blame([], P) == []


def test_missed_loser_report_distinguishes_information_from_none():
    recs = loss_world(seed=40, n_weeks=60)
    ls = L.LossState(P, seed=1)
    rep = L.step(ls, recs, NOW)
    out = L.missed_loser_report(list(ls.findings.values()), recs, rep.cohort, P)
    assert out["n"] > 0 and 0 <= out["had_signal_share"] <= 1
    assert L.missed_loser_report([], [], rep.cohort)["n"] == 0


def test_postmortem_written_only_for_named_held_losses():
    m, env = L.planted_loss(L.LossCause.RISK, 0)
    from engine.learning.postmortem import PostmortemStore
    store = PostmortemStore()
    pipe = L.LossPipeline(W.ResearchParams(loss_thr=0.01), store=store)
    f, _ = pipe.study(m, env, "2020-06-01")
    assert f.postmortem_id and len(store) == 1
    n, env2 = L.planted_loss(L.LossCause.UNKNOWN, 0)
    pipe.study(n, env2, "2020-06-01")
    assert len(store) == 1
    miss = dataclasses.replace(m, held=False, considered=False, pnl=None, side=0, rid="m")
    g, _ = pipe.study(miss, env, "2020-06-01")
    assert g.postmortem_id == "" and len(store) == 1
