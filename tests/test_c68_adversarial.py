"""C68 checklist Z - the twenty adversarial integration tests - through the REAL research loop on a planted world (canon C68, C69 sections
12-18 and 31; EXECUTION_LEDGER W-02 / W-09). C63: planted data only.

One loop run (engine.research.loop.step with the C68 stages registered by engine.research.error_loop) walks the planted C68 world
(error_loop.plant_world): a knowable trend pattern ('mom_r20_top') that pays inside the 5-10% band; a market-wide volatility burst while
the pattern still works (a FALSE regime alarm); one name that gaps down and turns turbulent with no information item (an unknowable
single-stock anomaly); a genuine market-wide regime switch that kills the pattern; its return. Every test below reads that run - its
state, its on-disk ledgers, its cycle reports - or re-drives one of the loop's own stages against a COPY of it with a planted defect.
Each checklist-Z bullet is one test, named after it."""
from __future__ import annotations

import copy
import dataclasses
import pickle
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import exits as EX
from engine.learning.core import stable_hash
from engine.research import calibration_target as CT
from engine.research import error_loop as EL
from engine.research import error_research as ER
from engine.research import exit_research as XR
from engine.research import expectations as XP
from engine.research import feeds as FD
from engine.research import loop as LP
from engine.research import pattern_change as PC
from engine.research import prediction_error as PE
from engine.research import regime_memory as RM
from engine.research import selection_constraint as SC
from engine.research import self_correct as SCX
from engine.research import two_stage as TS
from engine.research import what_changed as WC
from engine.research.core import FirewallBreach

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731
KEEP = ("observe.panel", "evaluate.two_stage", "questions.generate", "hypotheses.trees", "gain.priority")
FEED = FD.FeedConfig(warm_weeks=50)
N_CYCLES = 30                                                     # false alarm -> shock -> switch -> confirmation -> recovery
K_SCRAMBLE = 4


def loop_cfg(**kw) -> LP.LoopConfig:
    base = dict(run_id="p06z", free_gb=12.0, code_hash="p06-adv", checkpoint="off", cadence={},
                disabled=tuple(n for n in LP.BUILTIN_STAGES if n not in KEEP and n != "report.cycle"),
                two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    base.update(kw)
    return LP.LoopConfig(**base)


def snapshot(state: LP.LoopState, rt: LP.Runtime) -> dict:
    """What the C68 side knew after a cycle (compact: digests and the few series the tests read)."""
    st = state.modules["c68"]
    led = rt.handles["c68.ledgers"]
    esc = st.er.book.escalated(state.now)
    return {"now": state.now, "cycle": state.cycle - 1,
            "digest": stable_hash({"exp": sorted((p, led.expectations.meta(p)["content_hash"]) for p in led.expectations.ids()),
                                   "err": sorted((r.prediction_id, r.report_hash) for r in led.errors.reports()),
                                   "ews": st.ews.digest(), "market": st.market.content_hash(), "memory": st.memory.content_hash(),
                                   "influence": st.influence, "verdicts": st.verdicts}, 24),
            "ews": st.ews.digest(), "memory": st.memory.content_hash(), "market": st.market.content_hash(),
            "influence": dict(st.influence), "guard": dict(st.guard), "verdicts": dict(st.verdicts),
            "released": dict(st.gate.influence) if isinstance(st.gate, EL.BandGate) else {},
            "regimes": {r.record_id: st.memory.status(r.record_id).value for r in st.memory.records()},
            "esc": max([g.multiplier for g in esc], default=1.0), "esc_kinds": sorted({g.kind.value for g in esc})}


@dataclasses.dataclass
class Run:
    world: FD.World
    state: LP.LoopState
    rt: LP.Runtime
    reps: list
    snaps: list
    root: Path

    @property
    def st(self) -> EL.C68State:
        return self.state.modules["c68"]

    @property
    def led(self) -> EL.Ledgers:
        return self.rt.handles["c68.ledgers"]

    def between(self, a, b) -> list:
        return [s for s in self.snaps if pd.Timestamp(a) <= pd.Timestamp(s["now"]) < pd.Timestamp(b)]


@pytest.fixture(scope="module", autouse=True)
def registered():
    EL.register()
    yield
    EL.unregister()


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    world = EL.plant_world(EL.C68Plant())
    root = tmp_path_factory.mktemp("c68z")
    state, rt, _ = LP.open_loop(FD.WorldFeed(FD.InMemorySource(world), FEED), root, loop_cfg(), clock=CLOCK)
    reps, snaps = [], []
    for _ in range(N_CYCLES):
        reps.append(LP.step(state, rt))
        snaps.append(snapshot(state, rt))
    return Run(world, state, rt, reps, snaps, root)


def copy_of(run: Run, tmp_path: Path, feed=None) -> tuple[LP.LoopState, LP.Runtime]:
    """An independent copy of the finished run (state + on-disk ledgers) to plant a defect in without touching the fixture."""
    dst = tmp_path / "copy"
    shutil.copytree(run.root, dst)
    state = pickle.loads(pickle.dumps(run.state))
    rt = LP.Runtime(feed or FD.WorldFeed(FD.InMemorySource(run.world), FEED), dst, state.cfg, clock=CLOCK)
    return state, rt


def stage(name: str) -> LP.StageSpec:
    return next(s for s in LP.STAGES if s.name == name)


def scrambled_after(world: FD.World, day: str, seed: int = 9) -> FD.World:
    """The same world up to `day`; every later bar permuted in time and rescaled (a completely different future)."""
    rng = np.random.default_rng(seed)
    bars = {}
    for f, df in world.bars.items():
        g = df.copy()
        late = g.index > pd.Timestamp(day)
        g.loc[late] = g.loc[late].to_numpy()[rng.permutation(int(late.sum()))] * (1.9 if f != "Volume" else 0.3)
        bars[f] = g
    return FD.World(bars, None, None, None, world.sectors, {}, FD.market_proxy(bars), {"scrambled_after": day})


def test_the_run_carried_data_through_every_c68_stage(run):
    tab = EL.stage_table(run.reps)
    bad = tab[tab.status.isin(["FAILED", "REFUSED_LEAK", "SKIPPED_MISSING_MODULE"])]
    assert bad.empty, bad
    assert len(run.led.expectations) >= 30 and len(run.led.errors) >= 30 and len(run.st.memory.records()) >= 2


# ============================================================================================================ Z1
def test_z01_predictions_cannot_be_rewritten_after_outcomes(run, tmp_path):
    led = run.led
    pid = next(p for p in led.expectations.ids() if p in led.outcomes)
    exp = led.expectations.get(pid)
    later = run.state.now
    forged = dataclasses.replace(exp, predicted_return=exp.predicted_return + 0.01)
    with pytest.raises(XP.ExpectationRewrite):
        led.expectations.record(forged, exp.decided_at)                                 # a second version of the prediction
    with pytest.raises(FirewallBreach):
        led.expectations.record(forged, later)                                          # written after the fill: late, refused
    for fn in (led.expectations.update, led.expectations.delete, led.outcomes.update):
        with pytest.raises(FirewallBreach):
            fn(pid)
    with pytest.raises(FileExistsError):
        led.book.commit(pid, forged, "any", "2016-01-01", later, exp.decided_at)       # the calibration commitment is frozen too
    state, rt = copy_of(run, tmp_path)                                                  # rewrite the medium itself, then let the loop audit
    EL._ledgers(LP.Ctx(state, rt, state.now, state.cycle), state.modules["c68"])
    chain = rt.root / "c68" / "chain.jsonl"
    lines = chain.read_text(encoding="utf-8").splitlines()
    k = next(i for i, ln in enumerate(lines) if pid in ln and '"exp68"' in ln)
    forged_line = re.sub(r'("predicted_return":\s*)(-?[0-9.eE+-]+)', lambda m: m.group(1) + repr(float(m.group(2)) + 0.01), lines[k], count=1)
    assert forged_line != lines[k]
    lines[k] = forged_line
    chain.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec = LP.run_stage(stage("c68.monitor_audit"), LP.Ctx(state, rt, state.now, state.cycle))
    assert rec.status == LP.StageStatus.REFUSED_LEAK and "Tampered" in rec.reason


# ============================================================================================================ Z2
def test_z02_exits_cannot_be_manipulated_to_improve_prediction_statistics(run):
    st, led = run.st, run.led
    assert st.exit_records and all(r.decided_by == CT.LEARNED_POLICY and r.realised == r.policy_realised and r.fill_lag_sessions == 1
                                   for r in st.exit_records)
    assert all(not c["abuses"] for c in st.calibration)                                 # the loop's own statistic was never compromised
    comm = {c.pred_id: c for c in led.book.commitments()}
    r0 = next(r for r in st.exit_records if abs(r.realised - comm[r.pred_id].predicted) > 0.02)
    later = str((pd.Timestamp(r0.exit_date) + pd.Timedelta(days=3)).date())
    held = dataclasses.replace(r0, exit_date=later)                                     # held past the policy exit
    closer = dataclasses.replace(r0, realised=comm[r0.pred_id].predicted, decided_by="manual")   # a 'better' exit nearer the prediction
    for bad, kinds in ((held, {CT.AbuseKind.HELD_LONGER}), (closer, {CT.AbuseKind.WORSE_EXIT, CT.AbuseKind.NOT_POLICY_EXIT})):
        recs = [bad if r.pred_id == r0.pred_id else r for r in st.exit_records]
        rep = CT.evaluate(led.book, recs, run.state.now, st.cfg.target)
        assert rep.status == CT.Status.COMPROMISED and kinds <= {a.kind for a in rep.abuses}


# ============================================================================================================ Z3
def test_z03_the_pm1pp_evaluation_does_not_control_selling(run):
    st = run.st
    assert st.independence and all(x["identical"] for x in st.independence)           # audited on the loop's own exits every cycle
    assert len(st.independence) >= 10 and all(x["n_targets"] == 3 for x in st.independence)
    rep = EL.exit_independence(st)
    assert rep is None or rep.identical

    def target_aware(s, items):                                                        # a planted exit that 'waits for the prediction'
        out = EL.decide_exits(s, items)
        if s.cfg.target.tolerance < 0.01:
            out = [dataclasses.replace(r, days=np.minimum(r.days + 1, r.D)) for r in out]
        return out
    if st.exited:
        bad = EL.exit_independence(st, target_aware)
        assert not bad.identical and "pm02pp_95" in " ".join(bad.differing)


# ============================================================================================================ Z4
@pytest.fixture(scope="module")
def scrambled(run, tmp_path_factory):
    day = run.snaps[K_SCRAMBLE - 1]["now"]
    w2 = scrambled_after(run.world, day)
    state, rt, _ = LP.open_loop(FD.WorldFeed(FD.InMemorySource(w2), FEED), tmp_path_factory.mktemp("c68scr"), loop_cfg(), clock=CLOCK)
    snaps = []
    for _ in range(K_SCRAMBLE):
        LP.step(state, rt)
        snaps.append(snapshot(state, rt))
    return day, snaps


def test_z04_future_information_cannot_enter_error_analysis(run, scrambled, tmp_path):
    day, snaps = scrambled
    assert len(run.led.errors.reports(pd.Timestamp(day) + pd.Timedelta(days=1))) > 0     # errors existed by then
    for a, b in zip(run.snaps[:K_SCRAMBLE], snaps):
        assert a["now"] == b["now"] and a["digest"] == b["digest"], a["now"]          # expectations, errors, verdicts: bit-identical
    leak = FD.LeakPlant(EL.WORLD_KEY, run.snaps[0]["now"], days_ahead=3)             # a buggy adapter hands the C68 stages a future item
    state, reps = LP.run(FD.WorldFeed(FD.InMemorySource(run.world), FEED, leak=leak), tmp_path, loop_cfg(), max_cycles=1, clock=CLOCK)
    tab = EL.stage_table(reps).set_index("stage")
    for s in ("c68.selection_policy", "c68.market_regime", "c68.pattern_change"):
        assert tab.loc[s, "status"] == "REFUSED_LEAK", (s, tab.loc[s, "note"])
    assert EL.WORLD_KEY in tab.loc["c68.selection_policy", "note"]


# ============================================================================================================ Z5
def test_z05_future_information_cannot_enter_regime_detection(run, scrambled):
    day, snaps = scrambled
    for a, b in zip(run.snaps[:K_SCRAMBLE], snaps):
        assert (a["ews"], a["memory"], a["market"], a["regimes"]) == (b["ews"], b["memory"], b["market"], b["regimes"])
    for r in run.st.memory.records():
        assert r.information_through == r.detected_at and pd.Timestamp(r.change_date) <= pd.Timestamp(r.detected_at)
        assert all(pd.Timestamp(s.alarm_date) <= pd.Timestamp(r.detected_at) for s in r.earlier_signals)
    for d in run.st.ews.market.detections:
        assert d.information_through == d.alarm_date and d.change_index <= d.alarm_index


# ============================================================================================================ Z6
def test_z06_unknowable_events_are_allowed_to_remain_unknowable(run):
    wcs, led = run.st.wcs, run.led
    assert wcs.unknowable, "the planted world produced no unknowable outcome"
    for cid, concl in wcs.unknowable.items():
        inv = wcs.investigations[cid]
        assert inv.knowability.klass == WC.FiveWay.GENUINELY_UNKNOWABLE
        assert concl.model_change.effect.value == "NONE" and concl.test.kind == "NONE" and concl.new_knowledge.startswith("UNKNOWABLE_PRESERVED")
        assert not WC.reclassify_unknowable(wcs, cid, "no-such-claim") and cid in wcs.unknowable
        pid = next(p for p in led.expectations.ids() if EL.case_id(p) == cid)
        steps = [e["step"] for e in led.pipe.trail(pid)] if pid in led.pipe.keys() else []
        assert "HYPOTHESIS" not in steps                                               # nothing is invented to explain it
    for o, inv in run.st.er.investigations.items():                                   # confident-wrong cases judged unknowable ...
        if inv.by_number(14).status == ER.FindingStatus.UNKNOWABLE:
            assert inv.by_number(15).status == ER.FindingStatus.NOT_APPLICABLE             # ... are not mined for a precursor


# ============================================================================================================ Z7
def test_z07_tiny_errors_do_not_consume_excessive_compute(run):
    er = run.st.er
    tiny = [o for o in er.book.records(run.state.now) if abs(o.error) < er.cfg.material_error / 2]
    assert tiny, "no tiny error in the run"
    for o in tiny:
        it = er.intensities[o.obs_id]
        assert it.tier == ER.Tier.NONE and it.minutes == 0.0
        for t in ER.Tier:                                                             # no single-error job was ever created for it
            assert "ERR-" + stable_hash({"o": o.obs_id, "t": t.value}, 12) not in er.item_cells
    mins = {t: [er.intensities[o].minutes for o in er.intensities if er.intensities[o].tier == t] for t in ER.Tier}
    assert max(mins[ER.Tier.NONE]) == 0.0 and (not mins[ER.Tier.CHEAP] or max(mins[ER.Tier.CHEAP]) < min(mins[ER.Tier.DEEP] or [1e9]))


# ============================================================================================================ Z8
def test_z08_major_errors_trigger_deeper_investigation(run):
    er = run.st.er
    deep = [o for o, it in er.intensities.items() if it.tier == ER.Tier.DEEP]
    none = [o for o, it in er.intensities.items() if it.tier == ER.Tier.NONE]
    assert deep and none
    size = {o.obs_id: abs(o.z(er.cfg)) * o.confidence for o in er.book.records(run.state.now)}
    assert np.mean([size[o] for o in deep]) > 2 * np.mean([size[o] for o in none])
    assert er.investigations, "no confident-wrong prediction was investigated"
    for o, inv in er.investigations.items():
        assert er.intensities[o].tier in (ER.Tier.STANDARD, ER.Tier.DEEP) and er.intensities[o].confident_wrong
        assert [f.number for f in inv.findings] == list(range(1, 16)) and inv.answered_share > 0.3
    subjects = {q.subject: q for q in run.state.questions.values()}
    deep_q = [subjects[s] for s, c in run.st.cells.items() if s in subjects and any(o in run.st.subject_pids.get(s, []) for o in deep)]
    assert deep_q, "deep errors never reached the loop's own question queue"


# ============================================================================================================ Z9
def test_z09_repeated_errors_escalate_research_priority(run):
    sw, rc = run.world.truth["switch"], run.world.truth["recover"]
    before = [s["esc"] for s in run.between("2000-01-01", sw)]
    after = [s["esc"] for s in run.between(sw, rc)]
    assert max(after) > 1.0 and max(after) > max(before)
    kinds = {k for s in run.between(sw, rc) for k in s["esc_kinds"]}
    assert kinds & {"OVERPREDICTION", "SHRINKAGE"}, kinds                               # 'expected 8-10%, realised 3-6%': over-prediction
    pri = run.st.tracker.research_priority(run.state.now)
    assert pri and pri[0].components["repetition"] > 0


# ============================================================================================================ Z10
def test_z10_pattern_changes_can_be_detected(run):
    sw, rc = pd.Timestamp(run.world.truth["switch"]), pd.Timestamp(run.world.truth["recover"])
    during = run.between(sw + pd.Timedelta(days=21), rc)
    failing = [s["verdicts"]["mom_r20_top"] in {c.value for c in PC.FAILING} for s in during]
    assert np.mean(failing) >= 0.5, [s["verdicts"]["mom_r20_top"] for s in during]
    conf = [r for r in run.st.memory.records() if run.st.memory.status(r.record_id) == RM.RegimeStatus.CONFIRMED]
    assert conf and any(abs((pd.Timestamp(r.change_date) - sw).days) <= 14 for r in conf)
    assert any("mom_r20_top" in r.patterns_weakened for r in conf)


# ============================================================================================================ Z11
def test_z11_false_regime_changes_do_not_disable_successful_patterns(run):
    fa0 = pd.Timestamp(run.world.truth["false_alarm"][0])
    sw = pd.Timestamp(run.world.truth["switch"])
    recs = run.st.memory.records()
    false = [r for r in recs if run.st.memory.status(r.record_id) == RM.RegimeStatus.FALSE_ALARM and abs((pd.Timestamp(r.change_date) - fa0).days) <= 21]
    assert false, "the planted volatility burst raised no regime record"
    window = run.between(false[0].detected_at, sw)
    assert window
    for s in window:
        assert s["guard"]["mom_r20_top"] == RM.PatternAction.KEEP.value                # the alarm alone never switched it off
        assert s["influence"]["mom_r20_top"] > 0.0 and s["released"].get("mom_r20_top", 1.0) > 0.0


# ============================================================================================================ Z12
def test_z12_genuinely_degraded_patterns_can_reduce_influence(run):
    sw, rc = run.world.truth["switch"], run.world.truth["recover"]
    during = run.between(sw, rc)
    assert min(s["influence"]["mom_r20_top"] for s in during) <= 0.5
    reduced = [g for g in run.st.gate_log if pd.Timestamp(sw) <= pd.Timestamp(g["now"]) < pd.Timestamp(rc) and g["influence"]["mom_r20_top"] < 1]
    assert reduced, "the reduced influence never reached the decision path"
    fired = [g for g in reduced if g["n_fired"]["mom_r20_top"]]
    assert fired and np.mean([g["shift"]["mom_r20_top"] for g in fired]) < 0              # its names' forecasts were pulled down


# ============================================================================================================ Z13
def test_z13_recovered_patterns_can_regain_influence(run):
    rc = run.world.truth["recover"]
    assert min(s["influence"]["mom_r20_top"] for s in run.between(run.world.truth["switch"], rc)) < 1.0
    assert run.snaps[-1]["influence"]["mom_r20_top"] == 1.0, [s["influence"] for s in run.snaps[-6:]]
    assert run.st.gate_log[-1]["influence"]["mom_r20_top"] == 1.0                        # and the decision path got it back


# ============================================================================================================ Z14
def test_z14_market_wide_changes_are_distinguished_from_single_stock_anomalies(run):
    sw = pd.Timestamp(run.world.truth["switch"])
    conf = [r for r in run.st.memory.records() if abs((pd.Timestamp(r.change_date) - sw).days) <= 14]
    assert conf and all(r.scope == "MARKET_WIDE_REGIME_CHANGE" for r in conf)
    name, day = run.world.truth["shock"]
    uid = EL.unit_id(name)
    d0 = pd.Timestamp(day)
    hits = [d for d in run.st.ews.units.detections if d.unit == uid and d0 <= pd.Timestamp(d.alarm_date) <= d0 + pd.Timedelta(days=30)]
    assert hits, "the shocked name's own stream never alarmed"
    near = [w for w in run.st.warnings if d0 <= pd.Timestamp(w[0]) <= d0 + pd.Timedelta(days=30)]
    assert all(w[2] != "MARKET_WIDE_REGIME_CHANGE" for w in near if pd.Timestamp(w[0]) < sw - pd.Timedelta(days=5))
    assert not [r for r in run.st.memory.records() if abs((pd.Timestamp(r.change_date) - d0).days) <= 5]   # no regime from one stock


# ============================================================================================================ Z15
def test_z15_exit_timing_is_learned_independently(run):
    st = run.st
    assert st.policies and all(isinstance(p, XR.LearnedExitRule) and p.model is not None for p in st.policies.values())
    assert any(np.isfinite(p.threshold) for p in st.policies.values())               # a learned sell rule, not 'hold to the end'
    if st.exited:
        base = EL._concat(EL.decide_exits(st, st.exited))
        blind = copy.copy(st)
        blind.pending, blind.forecasts, blind.exit_records = {}, {}, []                 # take every prediction away from the exit
        again = EL._concat(EL.decide_exits(blind, st.exited))
        assert np.array_equal(base.days, again.days) and np.array_equal(base.net, again.net)
    with pytest.raises(FirewallBreach):
        XR.check_inputs(["predicted_return"])                                           # no exit input may name a prediction or target
    held = [r for r in st.exit_records if r.exit_date < next(c.matures_by for c in run.led.book.commitments() if c.pred_id == r.pred_id)]
    assert held, "the learned exit never sold before the planned end"


# ============================================================================================================ Z16
def test_z16_the_5_10_selection_constraint_is_enforced(run):
    led = run.led
    for pid in led.expectations.ids():
        e = led.expectations.get(pid)
        assert 0.05 <= e.predicted_return <= 0.10
    removed = 0
    for dec in run.state.decisions:
        if len(dec.positions):
            assert TS.BAND_STAGE in dec.funnel.as_dict()
            p = dec.positions
            assert (p["band_reason"] == "ELIGIBLE").all() and p["gain_pred"].between(0.05, 0.10).all() and (p["p_band"] >= 0.25).all()
        removed += int((dec.table["reason"] == TS.Reason.OUT_OF_BAND).sum())
    assert removed > 0                                                                  # it removed names the rest of the funnel wanted


# ============================================================================================================ Z17
def test_z17_stocks_outside_the_band_cannot_be_selected_to_improve_another_metric(run, tmp_path):
    fc = SC.GainForecast("A", "2020-01-02", "P", 0.30, {0.5: 0.30}, 0.9, 500, "d")
    for bad, why in ((dataclasses.replace(fc, basis="p_up"), SC.Reason.WRONG_QUANTITY), (dataclasses.replace(fc, basis="historical_statistic"),
                     SC.Reason.WRONG_QUANTITY), (dataclasses.replace(fc, mean=0.07, quantiles={0.5: 0.07}), SC.Reason.STALE_POLICY)):
        assert SC.select([bad], "2020-01-02", "Q" if why == SC.Reason.STALE_POLICY else "P").reason_of("A") == why
    state, rt = copy_of(run, tmp_path)                                                   # a second selector that waves everyone through

    def bypass(ctx):
        ctx.rt.pipe.band_gate = lambda rows, now: pd.DataFrame({"eligible": True, "reason": "ELIGIBLE", "point": 0.07, "p_band": 1.0},
                                                               index=rows.index)
        return 1, 1, "planted bypass selector"
    LP.register_stage("zz.bypass", bypass, after="c68.selection_policy")
    try:
        n0 = len(EL.open_ledgers(rt.root, None, state.modules["c68"].cfg, "x").expectations)
        rep = LP.step(state, rt)
    finally:
        LP.unregister_stage("zz.bypass")
    by = {s["stage"]: s for s in rep["stages"]}
    dec = state.decisions[-1]
    if len(dec.positions):
        assert by["c68.expectations"]["status"] == "REFUSED_LEAK" and "bypassed" in by["c68.expectations"]["reason"]
        assert len(EL.open_ledgers(rt.root, None, state.modules["c68"].cfg, "x").expectations) == n0
    else:
        assert by["c68.expectations"]["status"] in ("OK", "SKIPPED_NO_INPUT")


# ============================================================================================================ Z18
def test_z18_the_system_cannot_game_the_pm1pp_target(run):
    st, led, now = run.st, run.led, run.state.now
    honest = CT.evaluate(led.book, st.exit_records, now, st.cfg.target)
    assert honest.status != CT.Status.COMPROMISED and not honest.abuses and f"of {honest.all_.n} matured" in honest.headline
    losers = [r for r in st.exit_records if not CT.within(r.realised, next(c.predicted for c in led.book.commitments() if c.pred_id == r.pred_id), 0.01)]
    assert losers
    hidden = CT.evaluate(led.book, [r for r in st.exit_records if r not in losers], now, st.cfg.target)          # suppress the misses
    assert hidden.status == CT.Status.COMPROMISED and CT.AbuseKind.SUPPRESSED in {a.kind for a in hidden.abuses}
    assert hidden.all_.share == honest.all_.share                                       # ... and they still count as misses
    picked = CT.evaluate(led.book, st.exit_records, now, st.cfg.target, sample=[r.pred_id for r in st.exit_records if r not in losers])
    assert CT.AbuseKind.CHERRY_PICKED in {a.kind for a in picked.abuses}
    wider = CT.evaluate(led.book, st.exit_records, now, CT.Target(0.05, 0.8, name="pm5pp"))                    # move the goalposts
    assert CT.AbuseKind.TARGET_CHANGED in {a.kind for a in wider.abuses}
    reps = led.errors.reports(now)
    ad = PE.honest_tolerance(reps, 0, st.cfg.error_cfg)
    canon = CT.share_verdict([CT.within(r["return"].actual, r["return"].expected, st.cfg.error_cfg.tol) for r in reps])
    assert ad["hits"] == canon["hits"] and ad["n"] == canon["n"]                       # one statistic, whichever door it is read through


# ============================================================================================================ Z19
def test_z19_discoveries_must_pass_out_of_sample_validation(run):
    wcs = run.st.wcs
    assert wcs.claims and wcs.verdicts
    for cid, v in wcs.verdicts.items():
        assert v.status != WC.ClaimStatus.KNOWLEDGE and 5 in v.failed_steps()          # no fresh-period replication: never knowledge
    assert run.st.corrections
    for c in run.st.corrections:
        assert not c["promoted"]
        for fix, (eff, t, weeks) in c["effects"].items():
            assert weeks > 0                                                            # every candidate was scored on later data
    keys = [k for k in run.led.pipe.keys() if k.startswith("FIX:recency_refit@")]
    assert len(keys) == len(run.st.corrections)
    for k in keys:
        assert [e["step"] for e in run.led.pipe.trail(k)] == ["OOS_TEST", "MODEL_UPDATE", "VALIDATION", "REJECTED"]


# ============================================================================================================ Z20
def test_z20_the_learner_improves_only_when_evidence_justifies_improvement(run, tmp_path, monkeypatch):
    st = run.st
    assert st.production["name"] == "incumbent" and "placebo_promoted" not in st.counters
    state, rt = copy_of(run, tmp_path)
    ctx = LP.Ctx(state, rt, state.now, state.cycle)
    st2 = state.modules["c68"]
    real = SCX.step

    def gate_says(promote):
        def fake(frame, fixes, now, base=None, policy=None, cfg=SCX.SelfCorrectConfig(), only_if_deteriorated=False):
            rep = real(frame, fixes, now, base, policy, cfg)
            return dataclasses.replace(rep, promoted=tuple(promote), rejected={k: v for k, v in rep.rejected.items() if k not in promote})
        return fake
    monkeypatch.setattr(SCX, "step", gate_says(["placebo_noise"]))                     # even a (planted) gate pass of the placebo ...
    assert LP.run_stage(stage("c68.validate_promote"), ctx).status == LP.StageStatus.OK
    assert st2.production["name"] == "incumbent" and st2.counters.get("placebo_promoted") == 1   # ... is never applied
    monkeypatch.setattr(SCX, "step", gate_says(["recency_refit"]))                     # a promotion the gate granted IS applied
    LP.run_stage(stage("c68.validate_promote"), ctx)
    assert st2.production["name"] == "recency_refit" and st2.production["window_share"] == 0.4
    st2.production["since"] = str((pd.Timestamp(state.now) - pd.Timedelta(days=90)).date())
    worse = EL.correction_frame(st2, state.now)
    late = pd.to_datetime(worse["date"]) >= pd.Timestamp(st2.production["since"])
    worse.loc[late, "predicted"] = worse.loc[late, "realised"] + 0.2                     # the promoted learner then does much worse
    assert EL._monitor(st2, worse, state.now, EL.open_ledgers(rt.root, None, st2.cfg, "x")) == 1
    assert st2.production["name"] == "incumbent"                                         # monitoring rolled it back
