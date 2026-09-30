"""P06: engine/research/error_loop.py - the C68 prediction-error pipeline wired into the research loop (checklists T, Y; canon C68, C69
sections 4 and 31; W-02) - and the four de-duplications the C69 audit asked for. Planted worlds only (C63). Every mechanism has a planted
case it must catch, a null / empty case, and the loop-level proof that data flows through every C68 stage.

The adversarial checklist-Z suite over a long planted run is tests/test_c68_adversarial.py."""
from __future__ import annotations

import dataclasses
import inspect
import json
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from engine import exits as EX
from engine.learning import break_detection as BD
from engine.research import break_research as BR
from engine.research import calibration_target as CT
from engine.research import error_loop as EL
from engine.research import expectations as XP
from engine.research import feeds as FD
from engine.research import knowability as KN
from engine.research import loop as LP
from engine.research import market_expectations as ME
from engine.research import pattern_change as PC
from engine.research import prediction_error as PE
from engine.research import regime_memory as RM
from engine.research import selection_constraint as SC
from engine.research import two_stage as TS
from engine.research import what_changed as WC
from engine.research.core import FirewallBreach, Knowability

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731 - fixed wall clock: reruns are reproducible
KEEP = ("observe.panel", "evaluate.two_stage", "questions.generate", "hypotheses.trees", "gain.priority")
TS_CFG = TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20)
FEED = FD.FeedConfig(warm_weeks=50)


def loop_cfg(**kw) -> LP.LoopConfig:
    base = dict(run_id="p06", free_gb=12.0, code_hash="p06-test", checkpoint="off", cadence={},
                disabled=tuple(n for n in LP.BUILTIN_STAGES if n not in KEEP and n != "report.cycle"), two_stage=TS_CFG)
    base.update(kw)
    return LP.LoopConfig(**base)


@pytest.fixture(scope="module", autouse=True)
def registered():
    EL.register()
    yield
    EL.unregister()


@pytest.fixture(scope="module")
def world():
    return EL.plant_world(EL.C68Plant())


@pytest.fixture(scope="module")
def short_run(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("c68short")
    feed = FD.WorldFeed(FD.InMemorySource(world), FEED)
    state, rt, _ = LP.open_loop(feed, root, loop_cfg(), clock=CLOCK)
    reps = [LP.step(state, rt) for _ in range(6)]
    return feed, state, rt, reps, root


# ============================================================================================================ registration
def test_register_is_idempotent_and_puts_every_step_in_loop_order():
    before = LP.STAGE_NAMES
    assert EL.register() == EL.STAGE_NAMES and LP.STAGE_NAMES == before          # a second registration adds nothing
    n = LP.STAGE_NAMES
    after = {name: prev for name, _, prev, _ in EL.STAGES}
    for name, prev in after.items():
        assert n.index(prev) < n.index(name), name
    assert n.index("c68.selection_policy") < n.index("evaluate.two_stage") < n.index("c68.expectations")      # gate before the decision
    assert n.index("c68.outcomes_errors") < n.index("c68.what_changed") < n.index("c68.error_research") < n.index("questions.generate")
    assert n.index("questions.generate") < n.index("c68.research_depth") < n.index("gain.priority")            # depth into the EXISTING queue
    assert EL.WORLD_KEY in FD.BUILDERS
    with pytest.raises(ValueError):
        LP.register_stage("c68.expectations", lambda c: (0, 0, ""), after="observe.panel")                   # a different fn: refused


def test_config_validation_refuses_planted_bad_settings():
    assert EL.C68Config().validate() == []
    assert EL.C68Config(min_train=10).validate()                                  # below the gain model's own support floor
    assert EL.C68Config(patterns=(EL.PatternRule("x", "fwd_ret", 0.8, True),)).validate()   # unknown (future) signal
    assert EL.C68Config(selection=SC.SelectionConfig(lo=0.02)).validate()        # the band is fixed by checklist L
    with pytest.raises(ValueError):
        EL.configure(EL.C68Config(horizon=1))


# ============================================================================================================ the pipeline ledger
def test_pipeline_ledger_links_every_step_and_refuses_disorder():
    pl = EL.PipelineLedger()
    assert pl.verify()["ok"] and pl.keys() == [] and pl.furthest() == {}                      # empty case
    h1 = pl.add("X1", "PREDICTION", "2020-01-03", {"p": 0.07})
    pl.add("X1", "EXPECTATION", "2020-01-03", {"hash": "a"})
    with pytest.raises(EL.PipelineBroken):
        pl.add("X2", "OUTCOME", "2020-01-10")                                     # an outcome cannot start a trail
    pl.add("X1", "OUTCOME", "2020-01-10")
    with pytest.raises(EL.PipelineBroken):
        pl.add("X1", "EXPECTATION", "2020-01-11")                                 # never an expectation after its outcome
    tr = pl.trail("X1")
    assert [e["step"] for e in tr] == ["PREDICTION", "EXPECTATION", "OUTCOME"] and tr[1]["prev_event"] == h1
    pl.add("FIX:a", "OOS_TEST", "2020-02-01", parents=["X1"])
    pl.add("FIX:a", "REJECTED", "2020-02-01")
    assert pl.furthest() == {"OUTCOME": 1, "REJECTED": 1} and pl.verify()["ok"]
    recs = pl.lane.chain._mem_recs                                                  # planted rewrite of history in the medium
    recs[1]["body"] = {**recs[1]["body"], "payload": {"hash": "b"}}
    rep = pl.verify()
    assert not rep["ok"] and rep["problems"]


def test_pipeline_ledger_persists_and_reloads(tmp_path):
    pl = EL.PipelineLedger(tmp_path)
    pl.add("X1", "PREDICTION", "2020-01-03")
    pl.add("X1", "EXPECTATION", "2020-01-03")
    again = EL.PipelineLedger(tmp_path)
    assert again.last_step("X1") == "EXPECTATION" and again.verify()["ok"] and len(again) == 2
    with pytest.raises(EL.PipelineBroken):
        again.add("X1", "PREDICTION", "2020-01-04")                               # the reloaded ledger keeps the order rule


# ============================================================================================================ paths and bars
def test_build_paths_keeps_only_positions_whose_every_bar_is_before_now(world):
    bv = EL.bar_view(world.before("2017-03-01"))
    d = pd.Timestamp("2017-02-10")
    rows = pd.DataFrame({"vol20": [0.02, 0.02, 0.02], "atr": [0.02] * 3, "sector": ["SEC0"] * 3},
                        index=pd.MultiIndex.from_tuples([(d, "W000"), (pd.Timestamp("2017-02-24"), "W001"), (d, "NOPE")]))
    P, src = EL.build_paths(bv, rows, 5, "2017-03-01")
    assert len(P) == 1 and list(src.index.get_level_values(-1)) == ["W000"]     # the 02-24 path ends after now; unknown name dropped
    pos = bv.pos_after(d)
    assert P.o[0, 0] == bv.O[pos, bv.col["W000"]] and P.prev_close[0] == bv.C[pos - 1, bv.col["W000"]]   # fill at the NEXT open
    assert P.end[0] < np.datetime64("2017-03-01") and P.check()
    P0, s0 = EL.build_paths(bv, rows.iloc[0:0], 5, "2017-03-01")
    assert len(P0) == 0 and len(s0) == 0


def test_signal_streams_at_t_ignore_everything_after_t(world):
    """pattern outcomes, market-day streams and single-name streams at session t are functions of bars up to t: scrambling every
    later bar leaves them bit-identical (no retrospective cheating in the streams the change detectors read)."""
    w = world.before("2017-05-01")
    bv = EL.bar_view(w)
    t = bv.T - 40
    cut = bv.sessions[t]
    bars2 = {}
    rng = np.random.default_rng(5)
    for f, df in w.bars.items():
        g = df.copy()
        late = g.index > cut
        g.loc[late] = g.loc[late].to_numpy()[rng.permutation(int(late.sum()))] * 1.7
        bars2[f] = g
    bv2 = EL.bar_view(FD.World(bars2, None, None, None, w.sectors, {}, FD.market_proxy(bars2), {}))
    cfg = EL.C68Config()
    f1, c1 = EL.pattern_frames(bv, cfg.patterns, cfg)
    f2, c2 = EL.pattern_frames(bv2, cfg.patterns, cfg)
    for p in f1:
        a, b = f1[p].loc[:cut, "effect"], f2[p].loc[:cut, "effect"]
        assert a.equals(b) and len(a) > 100
    for k in (t - 5, t):
        assert EL.unit_values(bv, k) == EL.unit_values(bv2, k)
        assert EL.sector_volatility(bv, k) == EL.sector_volatility(bv2, k)
        d1, d2 = EL.market_day(bv, k, ()), EL.market_day(bv2, k, ())
        assert d1.day_abs_move == d2.day_abs_move and np.array_equal(d1.returns_window, d2.returns_window)
    assert any(not f1[p]["effect"].equals(f2[p]["effect"]) for p in f1)          # the scramble did change the future


def test_plant_world_carries_the_named_mechanisms(world):
    assert world.validate() == [] and {"switch", "recover", "false_alarm", "shock", "pattern"} <= set(world.truth)
    bv = EL.bar_view(world)
    cfg = EL.C68Config()
    fr, _ = EL.pattern_frames(bv, cfg.patterns, cfg)
    e = fr["mom_r20_top"]["effect"]
    sw, rc = pd.Timestamp(world.truth["switch"]), pd.Timestamp(world.truth["recover"])
    pre, mid, post = e[e.index < sw], e[(e.index > sw) & (e.index < rc)], e[e.index > rc + pd.Timedelta(days=10)]
    assert pre.mean() > 0.008 and mid.mean() < pre.mean() / 3 and post.mean() > 2 * mid.mean()           # works, stops, returns
    name, day = world.truth["shock"]
    r = world.bars["Close"][name].pct_change()
    d = pd.Timestamp(day)
    assert r.loc[day] < -0.08 and r.loc[d:].iloc[1:20].std() > 1.4 * r.loc[:d].iloc[-60:-1].std()         # a gap, then its own turbulence
    with pytest.raises(ValueError):
        EL.plant_world(EL.C68Plant(n_names=5))


# ============================================================================================================ the band gate
def _gain_world(n=400, seed=0):
    """Synthetic matured positions whose realised gain rises with a feature `r20` (the planted, knowable driver)."""
    rng = np.random.default_rng(seed)
    r20 = rng.uniform(-0.1, 0.4, n)
    drift = 0.004 + 0.035 * np.clip(r20, 0, None)
    D = 5
    rets = drift[:, None] + rng.normal(0, 0.006, (n, D))
    c = 50 * np.cumprod(1 + rets, 1)
    o = np.c_[np.full(n, 50.0), c[:, :-1]]
    week = np.array(pd.bdate_range("2019-01-04", periods=n).values, "datetime64[D]")
    P = EX.Paths(o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c, np.full(n, 50.0), np.full(n, 0.02), np.full(n, 0.02),
                 week, week + 7, np.full(n, "all", object), np.array([f"T{i}" for i in range(n)], object))
    F = np.c_[np.full(n, 0.02), np.full(n, 0.02), r20, np.zeros(n)]
    return P, F


def test_band_gate_selects_the_band_and_influence_can_pull_a_name_out():
    P, F = _gain_world()
    rule = EX.SpecRule(EX.ExitSpec())
    gm = SC.RealisableGainModel(rule, ("vol20", "atr", "r20", "r5"), SC.SelectionConfig()).fit(P, F, "2021-01-01")
    today = pd.DataFrame({"vol20": 0.02, "atr": 0.02, "r20": [0.0, 0.3, 0.9], "r5": 0.0, "sector": "all"},
                         index=pd.MultiIndex.from_tuples([(pd.Timestamp("2021-01-04"), t) for t in ("LOW", "MID", "HIGH")]))
    fired = {"MID": ("mom_r20_top",), "HIGH": ("mom_r20_top",)}
    gate = EL.BandGate(gm, SC.policy_id(rule), SC.SelectionConfig(), ("vol20", "atr", "r20", "r5"), {"mom_r20_top": 1.0},
                       {"mom_r20_top": "r20"}, fired)
    out = gate(today, "2021-01-04")
    by = dict(zip(today.index.get_level_values(-1), out["reason"]))
    assert by == {"LOW": "BELOW_BAND", "MID": "ELIGIBLE", "HIGH": "ABOVE_BAND"}, by
    muted = dataclasses.replace(gate, influence={"mom_r20_top": 0.0})                 # a suspended pattern cannot carry MID into the band
    assert muted(today, "2021-01-04").loc[today.index[1], "reason"] == "BELOW_BAND"
    assert not EL.abstain_gate(today, "2021-01-04")["eligible"].any()                 # no fitted model: nobody is eligible
    with pytest.raises(FirewallBreach):
        gate(today, "2020-12-01")                                                     # a model fitted later cannot serve an earlier day


def test_apply_band_gate_drops_ungated_positions_and_is_idempotent():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2021-01-04"), t) for t in ("A", "B", "C")])
    t = pd.DataFrame({"eligible": True, "p_move": 0.3, "mag_q90": 0.1, "mover": True, "p_up": [0.7, 0.8, 0.5], "p_down": 0.3,
                      "calibrated": True, "side": [1, 1, 0], "weight": [0.5, 0.5, 0.0], "reason": ["POSITION", "POSITION", "LOW_CONFIDENCE"]},
                     index=idx)
    dec = TS.DayDecision("2021-01-04", t, TS.Funnel(), {"open": True}, "k")
    today = pd.DataFrame({"r20": [0.1, 0.2, 0.3]}, index=idx)
    only_b = lambda rows, now: pd.DataFrame({"eligible": [ix[-1] == "B" for ix in rows.index], "reason": "X", "point": 0.07,  # noqa: E731
                                             "p_band": 0.4}, index=rows.index)
    g = TS.apply_band_gate(dec, only_b, today, "2021-01-04")
    assert list(g.positions.index.get_level_values(-1)) == ["B"] and g.table.loc[idx[0], "reason"] == TS.Reason.OUT_OF_BAND
    assert g.table.loc[idx[1], "weight"] == 1.0 and TS.BAND_STAGE in g.funnel.as_dict()
    assert TS.apply_band_gate(g, EL.abstain_gate, today, "2021-01-04") is g                 # already gated: unchanged
    none = TS.apply_band_gate(dec, EL.abstain_gate, today, "2021-01-04")
    assert len(none.positions) == 0                                                    # no forecast, no position


# ============================================================================================================ the loop carries data
def test_short_run_carries_data_through_every_c68_stage(short_run):
    feed, state, rt, reps, root = short_run
    tab = EL.stage_table(reps)
    assert set(tab["stage"]) == set(EL.STAGE_NAMES)
    assert not tab["status"].isin(["FAILED", "REFUSED_LEAK", "SKIPPED_MISSING_MODULE"]).any(), tab[tab.status.isin(["FAILED", "REFUSED_LEAK"])]
    ok = tab[tab.status == "OK"].groupby("stage").size()
    for s in ("c68.selection_policy", "c68.expectations", "c68.market_regime", "c68.pattern_change", "c68.outcomes_errors",
              "c68.what_changed", "c68.error_research", "c68.research_depth", "c68.monitor_audit"):
        assert ok.get(s, 0) >= 1, s
    st = state.modules["c68"]
    led = rt.handles["c68.ledgers"]
    assert len(led.expectations) >= 5 and len(led.outcomes) >= 3 and len(led.errors) == len(led.outcomes)
    for pid in led.expectations.ids():
        e = led.expectations.get(pid)
        assert led.expectations.meta(pid)["recorded_at"] < e.entry_at                 # frozen BEFORE the next-open fill
        steps = [x["step"] for x in led.pipe.trail(pid)]
        assert steps[:2] == ["PREDICTION", "EXPECTATION"]
    assert led.pipe.verify()["ok"] and led.outcomes.verify()["ok"]
    rep = json.loads((root / "c68" / f"cycle_{state.cycle - 1:05d}.json").read_text())
    assert rep["ledgers"]["expectations"] == len(led.expectations) and rep["label"] == EL.LABEL
    assert any(q.subject in st.cells for q in state.questions.values())                 # C68 research reached the loop's own questions


def test_every_committed_prediction_is_the_number_that_selected_the_name(short_run):
    """One forecaster per quantity: the expectation's predicted return is the band gate's median for that name on that day, the
    commitment carries the same number, and every position sat inside the band when it was taken."""
    feed, state, rt, reps, root = short_run
    st, led = state.modules["c68"], rt.handles["c68.ledgers"]
    comm = {c.pred_id: c for c in led.book.commitments()}
    n = 0
    for pid in led.expectations.ids():
        e = led.expectations.get(pid)
        f = st.forecasts[e.decided_at][e.subject]
        assert e.predicted_return == pytest.approx(f["median"]) and comm[pid].predicted == pytest.approx(e.predicted_return)
        assert 0.05 <= e.predicted_return <= 0.10 and f["eligible"]
        n += 1
    assert n >= 5
    for dec in state.decisions:
        if len(dec.positions):
            assert TS.BAND_STAGE in dec.funnel.as_dict() and (dec.positions["band_reason"] == "ELIGIBLE").all()


def test_frame_feed_without_bars_is_reported_as_no_input(tmp_path):
    """The production CLI's default planted FrameFeed has no bars: every C68 stage says SKIPPED_NO_INPUT (never OK with nothing)."""
    from engine.research import volatility_lab as VL
    F = VL.planted_frame("H1", n_dates=90, n_tickers=40, seed=1)
    dates = sorted(pd.unique(F.index.get_level_values(0)))
    feed = LP.FrameFeed(F, [str(pd.Timestamp(d).date()) for d in dates[60:62]])
    state, reps = LP.run(feed, tmp_path, loop_cfg(), max_cycles=1, clock=CLOCK)
    tab = EL.stage_table(reps)
    assert set(tab["status"]) == {"SKIPPED_NO_INPUT"}, tab


# ============================================================================================================ resume
def test_a_loop_killed_mid_cycle_resumes_without_redoing_or_skipping_a_c68_stage(world, tmp_path):
    feed = FD.WorldFeed(FD.InMemorySource(world), FEED)
    cfg = loop_cfg(checkpoint="stage")
    with pytest.raises(KeyboardInterrupt):
        LP.run(feed, tmp_path, cfg, max_cycles=3, clock=CLOCK, kill_after="1|c68.selection_policy")      # the pipe (and gate) are lost
    state, reps = LP.run(FD.WorldFeed(FD.InMemorySource(world), FEED), tmp_path, cfg, max_cycles=2, clock=CLOCK)
    for c in range(3):
        for s in EL.STAGE_NAMES:
            assert state.exec_count.get(f"{c}|{s}", 0) == 1, (c, s)
    st = state.modules["c68"]
    led = EL.open_ledgers(tmp_path, st.tracker, st.cfg, "p06-test")
    assert led.outcomes.verify()["ok"] and len(led.expectations) == len(set(led.expectations.ids()))
    for dec in state.decisions:                                                   # the resumed decision was re-gated, not waved through
        if len(dec.positions):
            assert TS.BAND_STAGE in dec.funnel.as_dict()
    assert all(led.pipe.trail(k) for k in led.pipe.keys())


# ============================================================================================================ de-duplication (C69 audit)
def test_one_pm1pp_statistic_calibration_target_is_canonical(monkeypatch):
    """prediction_error.honest_tolerance is an adapter: its share, interval and verdict come from calibration_target.share_verdict,
    and its hit flags from calibration_target.within. Neither computation exists in prediction_error any more."""
    src = inspect.getsource(PE)
    assert "def wilson" not in src and "<= cfg.tol" not in src and "abs(err) <=" not in src
    seen = {}

    def fake(hits, n_unresolved=0, goal=0.8, min_n=200, level=0.95):
        seen["n"] = len(list(hits))
        return {"hits": 1, "n": 1, "share": 0.123, "lo": 0.0, "hi": 1.0, "n_unresolved": 0, "worst_case": 0.5, "goal": goal,
                "status": CT.Status.ACHIEVED, "clear_miss": False}
    monkeypatch.setattr(CT, "share_verdict", fake)
    out = PE.honest_tolerance([], 0)
    assert out["rate"] == 0.123 and out["verdict"] == "TARGET_MET" and seen == {"n": 0}
    monkeypatch.undo()
    assert CT.within(0.07, 0.06, 0.01) and not CT.within(0.0701, 0.06, 0.01) and not CT.within(float("nan"), 0.06, 0.01)
    sv = CT.share_verdict([True] * 170 + [False] * 30, n_unresolved=100)
    assert sv["status"] == CT.Status.NOT_ACHIEVED and sv["worst_case"] == pytest.approx(170 / 300) and not sv["clear_miss"]
    assert CT.share_verdict([])["status"] == CT.Status.EMPTY


def test_five_way_mapping_is_owned_by_knowability(monkeypatch):
    assert WC.FiveWay is KN.FiveWay and "class FiveWay" not in inspect.getsource(WC)
    for c in Knowability:
        v = KN.five_way(c)
        assert isinstance(v.klass, KN.FiveWay) and v.reasons
    assert KN.five_way(Knowability.EXTERNALLY_CAUSED, condition_replicated=True).klass == KN.FiveWay.GENUINELY_UNKNOWABLE
    assert KN.five_way(None).klass == KN.FiveWay.CURRENTLY_UNEXPLAINED                  # nothing is forced predictable
    marker = KN.FiveWayVerdict(KN.FiveWay.PARTIALLY_PREDICTABLE, ("planted",), False, "planted")
    monkeypatch.setattr(KN, "five_way", lambda *a, **k: marker)
    assert WC.classify_knowability(()).reasons == ("planted",)                         # what_changed asks knowability, decides nothing


def test_pattern_change_consumes_break_research_and_runs_no_detector_of_its_own(monkeypatch):
    src = inspect.getsource(PC)
    assert "LC.trace" not in src and "_failing_run" not in src and "BR.detect_events" in src
    r = np.random.default_rng(12)
    up = np.r_[r.normal(0.012, 0.02, 120), r.normal(-0.03, 0.02, 45), r.normal(0.02, 0.02, 60)]
    assert PC.recovery_rate(up, PC.PARAMS) > 0
    calls = []
    real = BR.detect_events

    def spy(item, as_of, cfg=None):
        calls.append(len(item.frame))
        return real(item, as_of, cfg)
    monkeypatch.setattr(BR, "detect_events", spy)
    idx = pd.bdate_range("2001-01-01", periods=261)
    f = pd.DataFrame({"effect": np.r_[r.normal(0.01, 0.02, 200), r.normal(-0.02, 0.02, 60)]}, index=idx[:260])
    v = PC.classify("p", f, idx[260])
    assert calls and v.change in PC.FAILING and v.lifecycle_stage == "FAILURE"
    monkeypatch.setattr(BR, "detect_events", lambda item, as_of, cfg=None: [])      # the detector sees no break: no failure spell
    assert PC.classify("p", f, idx[260]).failing_rows == 0 and math.isnan(PC.recovery_rate(up, PC.PARAMS))


def test_market_and_regime_memories_live_on_archive_chain_lanes(tmp_path):
    hist = [ME.MarketObs(str(d.date()), {"opportunity": 0.1 + 0.01 * np.sin(i), "volatility": 0.02, "breadth": 0.5, "n_movers": 20.0,
                                         "n_qualifying": 5.0}) for i, d in enumerate(pd.bdate_range("2020-01-06", periods=35, freq="W-FRI"))]
    eng = ME.MarketExpectationEngine(root=tmp_path)
    for o in hist:
        eng.step(o.date, o)
    led = eng.ledger
    assert isinstance(led.lane, XP.SealedLane) and led.lane.kind == ME.LANE_MARKET and len(led) >= 10 and led.verify() == []
    assert led.entries()[3].prev_hash == led.lane.lines()[3]["prev"]                  # each link IS the archive link
    again = ME.MarketExpectationLedger(tmp_path)
    assert [e.digest for e in again.entries()] == [e.digest for e in led.entries()] and again.verify() == []   # persistent
    mem_eng = ME.MarketExpectationEngine()
    for o in hist:
        mem_eng.step(o.date, o)
    recs = mem_eng.ledger.lane.chain._mem_recs
    recs[4]["body"] = {**recs[4]["body"], "made": "1999-01-01"}                         # planted rewrite inside the archive lane
    assert -1 in mem_eng.ledger.verify()
    assert "GENESIS" not in inspect.getsource(ME.MarketExpectationLedger) and "GENESIS" not in inspect.getsource(RM.RegimeMemory)
    mem = RM.RegimeMemory()
    assert mem.lane.kind == RM.LANE_REGIME and mem.verify() == [] and mem._head() == mem.lane.head
    mem.lane.chain._mem_recs.append({"seq": 99, "prev": "x", "kind": RM.LANE_REGIME, "body": {}, "hash": "y"})
    assert -1 in mem.verify()                                                           # a forged lane line breaks the chain
