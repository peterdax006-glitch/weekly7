"""Tests for the autonomous research loop, the section-35 two-stage chain and the section-51 controller (C66 sections 3, 35, 47,
50, 51; canon C63: planted worlds only, every mechanism has a planted case it must catch, a null case and the empty case)."""
from __future__ import annotations

import dataclasses
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.learning import trader_view as TV
from engine.research import controller as CT
from engine.research import loop as LP
from engine.research import two_stage as TS
from engine.research import volatility_lab as VL
from engine.research.core import FirewallBreach, MaturedRecord, Provenance, Stage

CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731 - fixed wall clock: runs must be reproducible
HEAVY = ("evaluate.volatility_lab", "evaluate.direction_lab")


def world(truth: str = "H1", n_dates: int = 60, n_names: int = 40, seed: int = 5, direction: float = 0.6) -> pd.DataFrame:
    return VL.planted_frame(truth, n_dates=n_dates, n_tickers=n_names, seed=seed, effect=1.5, dir_signal=direction)


def cfg(**kw) -> LP.LoopConfig:
    base = dict(run_id="t", free_gb=12.0, code_hash="test-code", cadence={}, disabled=HEAVY, screen_top=4, max_new_questions=8,
                two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    base.update(kw)
    return LP.LoopConfig(**base)


def dates_of(F: pd.DataFrame) -> list:
    return sorted(pd.unique(F.index.get_level_values(0)))


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    F = world()
    feed = LP.FrameFeed(F, dates=dates_of(F)[30:36])
    root = tmp_path_factory.mktemp("loop")
    state, reps = LP.run(feed, root, cfg(), max_cycles=None, clock=CLOCK)
    return F, feed, root, state, reps


# ============================================================================================================ experiment science
def test_run_task_finds_planted_signal_not_noise_and_control_stays_null():
    ec = dataclasses.asdict(LP.ExperimentConfig())
    F = world()
    cutoff = str(pd.to_datetime(F["end"]).max().date())
    t = LP.ExperimentTask("k1", "B1", "VOLATILITY", "lv20", Stage.STRONGER_TESTS.value, cutoff, "", 3, ec)
    r = LP.run_task(t, F)
    assert r["ok"] and r["effect"] > 0.08 and r["effect"] / r["se"] > 4.0
    assert abs(r["control_effect"] / r["control_se"]) < 3.0
    assert r["n_units"] >= 3                                   # contexts (sectors) at rung 2
    N = world("null")
    rn = LP.run_task(dataclasses.replace(t, key="k2"), N)
    assert abs(rn["effect"] / rn["se"]) < 3.5                  # the null world must not produce a discovery


def test_run_task_never_reads_outcomes_after_its_cutoff():
    ec = dataclasses.asdict(LP.ExperimentConfig())
    F = world()
    d = dates_of(F)
    cutoff = str((pd.Timestamp(d[40]) + pd.Timedelta(days=8)).date())
    t = LP.ExperimentTask("k", "B", "VOLATILITY", "lv20", Stage.CHEAP_SCREEN.value, cutoff, "", 1, ec)
    clean = LP.run_task(t, F)
    G = F.copy()
    late = pd.to_datetime(G["end"]) > pd.Timestamp(cutoff)
    G.loc[late.to_numpy(), "touch"] = (G.loc[late.to_numpy(), "vol20"] > G["vol20"].median()).astype(float)   # planted future leak
    assert LP.run_task(t, G)["effect"] == pytest.approx(clean["effect"])
    assert clean["data_through"] <= cutoff


def test_resolve_feature_and_design_kind():
    cols = world().columns
    assert LP.resolve_feature("Does the newly discovered feature vol_ratio_short volatility generalize", "VOLATILITY", cols) == "vol_ratio_short"
    assert LP.resolve_feature("no feature named", "DIRECTION", cols) == "rel_r5"
    assert LP.resolve_feature("feature not_a_feature", "VOLATILITY", ["x"]) == ""          # nothing derivable: refuse, never guess
    assert LP.design_kind("DATA_QUALITY") == LP.design_kind("VOLATILITY") != LP.design_kind("DIRECTION")


# ============================================================================================================ the full cycle
def test_full_cycle_produces_section47_chain_without_future_information(planted_run):
    F, feed, root, state, reps = planted_run
    assert len(reps) == 6
    assert all(not r["refused"] for r in reps), [r["refused"] for r in reps]
    kinds = state.lineage.kinds()
    for k in ("QUESTION", "ITEM", "BRANCH", "JOB", "RESULT", "MEMORY", "EVENT", "DECISION"):
        assert kinds.get(k, 0) > 0, (k, kinds)
    chains = state.lineage.section47_chains()
    assert chains, "no question -> priority -> experiment -> result -> memory -> new question chain"
    q0, q1 = chains[0][0], chains[0][-1]
    assert state.lineage.nodes[q1]["cycle"] > state.lineage.nodes[q0]["cycle"]          # the new question came later
    assert all(v == 1 for v in state.exec_count.values())                                # every stage ran exactly once per cycle
    assert sum(r["validated"] for r in reps) >= 2
    assert any(r["ladder"] for r in reps)
    assert reps[-1]["statements"] and "allocate" in reps[-1]["statements"][0]
    for j in state.jobs.values():                                                       # every task used matured data only
        if j.result and j.result.get("data_through"):
            assert j.result["data_through"] < j.submitted_at
    assert (root / "reports" / "cycle_00005.json").exists()


def test_every_section3_phase_is_a_stage_in_order_and_every_named_module_is_called():
    phases = [s.phase for s in LP.STAGES]
    order = list(LP.LoopPhase)
    assert sorted(set(phases), key=order.index) == [p for p in order if p in set(phases)]
    assert set(order) == set(phases)
    named = {"observer", "autopsy", "missed", "winners_losers", "loss_pipeline", "knowability", "unknown_cause", "counterfactual",
             "break_research", "volatility_lab", "direction_lab", "frontier", "symmetry", "discovery", "interactions", "multiscale",
             "cross_section", "regimes", "precursors", "questions", "targets", "hypothesis_tree", "priority", "experiments",
             "failed_lab", "meta_research", "compute_manager", "value_accounting", "waste", "replication", "quality_gate", "scorecard",
             "research_graph", "decision_bridge", "science_memory", "brain_health", "diversity", "firewall", "controller", "two_stage"}
    src = Path(LP.__file__).read_text(encoding="utf-8")
    missing = [m for m in named if f"from engine.research import {m} as" not in src and m not in ("controller", "two_stage")]
    assert not missing, missing


def test_duplicate_designs_are_refused_and_their_questions_merged(planted_run):
    _, _, _, state, _ = planted_run
    refused = [j for j in state.jobs.values() if j.state == LP.JobState.REFUSED and "duplicate design" in j.error]
    if refused:
        led = state.modules["question_ledger"]
        assert any(led.fate(j.question_id) == "MERGED" for j in refused)
        assert state.counters.get("duplicate_designs", 0) == len(refused)
    keys = {}
    for j in state.jobs.values():
        if j.state != LP.JobState.REFUSED:
            k = (LP.design_kind(j.problem), j.feature, j.stage, j.task.get("repeat", 0), j.submitted_at)
            assert k not in keys, "the same design ran twice on the same data"
            keys[k] = j.key


# ============================================================================================================ leaks
class LeakyFeed(LP.FrameFeed):
    """A feed that slips one outcome that has not matured into the 'matured' panel."""

    def observe(self, now):
        obs = super().observe(now)
        row = self.frame[pd.to_datetime(self.frame.index.get_level_values(0)) == pd.Timestamp(now)].iloc[:1]
        obs.matured = pd.concat([obs.matured, row])
        return obs


def test_planted_future_row_is_refused_at_observe(tmp_path):
    F = world()
    d = dates_of(F)
    feed = LeakyFeed(F, dates=d[32:33])
    state, reps = LP.run(feed, tmp_path, cfg(checkpoint="off"), max_cycles=1, clock=CLOCK)
    r = reps[0]
    assert set(r["refused"]) == {"observe.panel"} and "on/after now" in r["refused"]["observe.panel"]
    by = {s["stage"]: s for s in r["stages"]}
    assert by["evaluate.two_stage"]["status"] == "SKIPPED_NO_INPUT"                 # nothing downstream saw the leaked panel
    assert r["launched"] == 0 and not state.jobs


def test_planted_future_event_is_refused_at_question_generation(tmp_path):
    from engine.research import questions as Q
    F = world()
    d = dates_of(F)
    feed = LP.FrameFeed(F, dates=d[32:33])
    state, rt, _ = LP.open_loop(feed, tmp_path, cfg(checkpoint="off"), clock=CLOCK)
    state.carry_events.append(Q.QuestionEvent("surprise", "feature lv20 surprise", str(pd.Timestamp(d[32]).date()), 0.9))
    rep = LP.step(state, rt)
    assert "questions.generate" in rep["refused"]
    assert "observe.panel" not in rep["refused"]


# ============================================================================================================ kill and resume
def test_killed_loop_resumes_without_redoing_or_skipping_work(planted_run, tmp_path):
    F, _, _, full, full_reps = planted_run
    feed = LP.FrameFeed(F, dates=dates_of(F)[30:36])
    with pytest.raises(KeyboardInterrupt):
        LP.run(feed, tmp_path, cfg(), max_cycles=None, clock=CLOCK, kill_after="1|learn.memory")
    assert (tmp_path / "checkpoints" / "INTERRUPTED.json").exists()
    state, reps = LP.run(feed, tmp_path, cfg(), max_cycles=2, clock=CLOCK)
    assert reps[0]["cycle"] == 1 and reps[0]["resume"]["action"] in ("CONTINUE", "RECOVER_EXPERIMENT", "RECONCILE_EXPERIMENT")
    for c in (0, 1, 2):                                                                # nothing redone, nothing skipped
        assert sorted(state.done_stages[c]) == sorted(LP.STAGE_NAMES)
        assert all(state.exec_count[f"{c}|{s}"] == 1 for s in LP.STAGE_NAMES)
    ref = {k for k, j in full.jobs.items() if j.cycle <= 2}
    got = {k for k, j in state.jobs.items() if j.cycle <= 2}
    assert got == ref
    ref_q = {q for q, v in full.lineage.nodes.items() if v["kind"] == "QUESTION" and v["cycle"] <= 2}
    got_q = {q for q, v in state.lineage.nodes.items() if v["kind"] == "QUESTION" and v["cycle"] <= 2}
    assert got_q == ref_q
    assert [r["validated"] for r in reps] == [r["validated"] for r in full_reps[1:3]]


def test_corrupted_newest_checkpoint_falls_back_to_the_previous_one(tmp_path):
    F = world()
    feed = LP.FrameFeed(F, dates=dates_of(F)[31:32])
    c = cfg(disabled=tuple(n for n in LP.STAGE_NAMES if n not in ("observe.panel", "report.cycle")))
    LP.run(feed, tmp_path, c, max_cycles=1, clock=CLOCK)
    blobs = sorted((tmp_path / "checkpoints" / "t").glob("state_*.pkl"))
    blobs[-1].write_bytes(b"torn write")
    ck = LP.Checkpointer(tmp_path, c)
    st, _ = ck.load("test-code")
    assert st is not None and isinstance(st, LP.LoopState)
    with pytest.raises(Exception):
        ck.load("other-code")                                                          # stale code refused (fail closed)


# ============================================================================================================ missing modules
def _only(*names: str) -> tuple:
    return tuple(n for n in LP.STAGE_NAMES if n not in names + ("observe.panel", "report.cycle"))


def test_missing_module_stage_is_reported_and_called_once_present(tmp_path, monkeypatch):
    import engine.research as pkg
    F = world()
    feed = LP.FrameFeed(F, dates=dates_of(F)[33:34])
    c = cfg(checkpoint="off", disabled=_only("surprises.regimes"))
    monkeypatch.setitem(sys.modules, "engine.research.regimes", None)
    monkeypatch.delattr(pkg, "regimes", raising=False)
    _, reps = LP.run(feed, tmp_path / "a", c, max_cycles=1, clock=CLOCK)
    by = {s["stage"]: s for s in reps[0]["stages"]}
    assert by["surprises.regimes"]["status"] == "SKIPPED_MISSING_MODULE"
    assert "surprises.regimes" in reps[0]["missing_modules"]
    calls = []
    fake = types.ModuleType("engine.research.regimes")
    fake.RegimeMonitor = lambda: "monitor"
    fake.PatternRegimeBook = lambda: "book"
    fake.regime_questions = lambda *a, **k: []

    def step(monitor, book, now, row, effects=()):
        calls.append((monitor, book, str(now), row["ret"]))
        return types.SimpleNamespace(changes=())
    fake.step = step
    monkeypatch.setitem(sys.modules, "engine.research.regimes", fake)
    monkeypatch.setattr(pkg, "regimes", fake, raising=False)
    _, reps2 = LP.run(feed, tmp_path / "b", c, max_cycles=1, clock=CLOCK)
    by2 = {s["stage"]: s for s in reps2[0]["stages"]}
    assert by2["surprises.regimes"]["status"] == "OK" and len(calls) == 1 and calls[0][:2] == ("monitor", "book")


def test_stage_without_inputs_is_skipped_not_passed(planted_run):
    _, _, _, _, reps = planted_run
    by = {s["stage"]: s for s in reps[0]["stages"]}
    for name in ("observe.observer", "missed.knowability", "breaks.break_research"):
        assert by[name]["status"] == "SKIPPED_NO_INPUT" and by[name]["reason"]


def test_empty_feed_runs_nothing(tmp_path):
    F = world()
    state, reps = LP.run(LP.FrameFeed(F, dates=[]), tmp_path, cfg(checkpoint="off"), max_cycles=3, clock=CLOCK)
    assert reps == [] and state.cycle == 0 and not state.jobs


def test_config_validation():
    with pytest.raises(ValueError):
        LP.new_state(cfg(mode="cluster"))
    with pytest.raises(ValueError):
        LP.new_state(cfg(disabled=("no.such.stage",)))


# ============================================================================================================ C67 sweeps
def test_sweeps_yield_to_higher_value_work_and_resume(tmp_path):
    made = []

    def factory(root):
        box = {"done": 0}
        made.append(box)

        def run(n, now):
            box["done"] += n
            return n
        return run
    spec = LP.SweepSpec("sweep.fake", "NEW_REPRESENTATION", factory)
    F = world()
    feed = LP.FrameFeed(F, dates=dates_of(F)[33:35])
    c = cfg(checkpoint="off", sweep_units=1, disabled=_only("sweeps.background"))
    state, rt, _ = LP.open_loop(feed, tmp_path, c, sweeps=[spec], clock=CLOCK)
    rt.executor.running = lambda: c.max_workers                                    # every worker busy with ranked experiments
    LP.step(state, rt)
    assert state.sweeps["sweep.fake"].yielded == 1 and state.sweeps["sweep.fake"].units == 0 and not made
    rt.executor.running = lambda: 0
    LP.step(state, rt)
    assert state.sweeps["sweep.fake"].units >= 1 and len(made) == 1
    state2, rt2, _ = LP.open_loop(LP.FrameFeed(F, dates=dates_of(F)[35:36]), tmp_path, c, sweeps=[spec], clock=CLOCK)
    state2.sweeps = state.sweeps
    LP.step(state2, rt2)
    assert state2.sweeps["sweep.fake"].units >= 2 and state2.sweeps["sweep.fake"].runs == 2


# ============================================================================================================ controller (section 51)
R, P = CT.CapabilityReading, CT.Phase


def _vol(v, day, se=0.005):
    return R(P.VOLATILITY, "vol_auc", v, day, se, 900)


def test_controller_moves_share_to_planted_problem_then_shifts_and_revisits_on_regression():
    st = CT.new_state()
    d1 = CT.step(st, "2020-01-10", [_vol(0.55, "2020-01-09")])
    assert d1.top == P.VOLATILITY and d1.share(P.VOLATILITY) >= 0.3
    assert d1.statements[0].kind == CT.StatementKind.ALLOCATE and "highest-value unresolved problem, so allocate" in d1.statements[0].text
    assert f"{round(100 * d1.share(P.VOLATILITY))}%" in d1.statements[0].text
    assert d1.share(P.DIRECTION) == pytest.approx(st.cfg.probe_share)               # gated: volatility first
    for day in ("2020-01-17", "2020-01-24"):
        d = CT.step(st, day, [_vol(0.74, str(pd.Timestamp(day) - pd.Timedelta(days=1))[:10]),
                              R(P.DIRECTION, "direction_acc", 0.51, str(pd.Timestamp(day) - pd.Timedelta(days=1))[:10], 0.01, 900)])
    assert st.satisfied.get(P.VOLATILITY.value)
    assert d.top == P.DIRECTION and d.share(P.DIRECTION) > d.share(P.VOLATILITY)
    assert any(s.kind == CT.StatementKind.SHIFT for s in d.statements) or st.history[-2]["top"] == P.DIRECTION.value
    for day in ("2020-01-31", "2020-02-07"):
        d = CT.step(st, day, [_vol(0.55, str(pd.Timestamp(day) - pd.Timedelta(days=1))[:10])])
    assert any(s.kind == CT.StatementKind.REGRESSION for s in d.statements)
    assert d.top == P.VOLATILITY and d.share(P.DIRECTION) == pytest.approx(st.cfg.probe_share)
    assert st.regressions and st.regressions[-1][1] == P.VOLATILITY.value


def test_controller_redirects_a_stalled_phase():
    def run(yields):
        st = CT.new_state()
        for day in ("2020-01-17", "2020-01-24"):
            CT.step(st, day, [_vol(0.74, str(pd.Timestamp(day) - pd.Timedelta(days=1))[:10]),
                              R(P.DIRECTION, "direction_acc", 0.52, str(pd.Timestamp(day) - pd.Timedelta(days=1))[:10], 0.01, 900)])
        return CT.step(st, "2020-02-01", [], yields)
    ys = [CT.YieldRecord(P.DIRECTION, 0.0, 200.0, f"2020-01-{d:02d}") for d in (25, 26, 27, 28, 29)]
    stalled, fresh = run(ys), run([])
    a = {x.phase: x for x in stalled.assessments}
    assert a[P.DIRECTION].status == CT.PhaseStatus.STALLED
    assert any(s.kind == CT.StatementKind.REDIRECT and s.phase == P.DIRECTION for s in stalled.statements)
    assert stalled.share(P.DIRECTION) < fresh.share(P.DIRECTION)


def test_controller_refuses_future_readings_and_states_no_eighty_honestly():
    st = CT.new_state()
    with pytest.raises(FirewallBreach):
        CT.step(st, "2020-01-10", [_vol(0.6, "2020-01-10")])                           # matured ON now: not yet a fact
    st = CT.new_state()
    d = CT.step(st, "2020-03-02", [_vol(0.74, "2020-03-01"), R(P.DIRECTION, "direction_acc", 0.62, "2020-03-01", 0.01, 900),
                                   R(P.DIRECTION_AT_COVERAGE, "acc_at_coverage", 0.64, "2020-03-01", 0.02, 400)])
    assert any(s.kind == CT.StatementKind.NO_EIGHTY and s.text == "No reliable 80% directional region has been found."
               for s in d.statements)


def test_controller_empty_and_feeds_priority_ladder_and_diversity():
    st = CT.new_state()
    d = CT.step(st, "2020-01-10")
    assert sum(d.shares.values()) == pytest.approx(1.0)
    assert all(a.status in (CT.PhaseStatus.UNMEASURED, CT.PhaseStatus.BLOCKED) for a in d.assessments)
    assert all(v <= st.cfg.cap + 1e-9 for v in d.shares.values())
    d = CT.step(st, "2020-01-17", [_vol(0.55, "2020-01-16")])
    w = CT.objective_weights(d)
    top_problem = max(d.problem_shares, key=d.problem_shares.get)
    assert CT.problem_weight(d)[top_problem.value] == pytest.approx(1.0)
    assert w.weight(top_problem) > 1.0
    assert CT.directives(d).area_multiplier["VOLATILITY"] > 1.0
    st2 = CT.state_from_json(CT.state_to_json(st))
    assert st2.shares == st.shares and st2.last_now == st.last_now


# ============================================================================================================ two-stage (section 35)
@pytest.fixture(scope="module")
def two_stage_run():
    F = world(n_dates=70, n_names=50)
    d = dates_of(F)
    pipe = TS.TwoStage(TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    decs = [TS.run_day(pipe, F, TS.point_in_time_state(F, now).frame, now) for now in d[50:66:3]]
    return F, d, pipe, decs


def test_two_stage_funnel_and_direction_only_on_predicted_movers(two_stage_run):
    F, d, pipe, decs = two_stage_run
    assert pipe.report.vol_ok and pipe.report.gate_open and pipe.report.dir_ok
    s = decs[-1].summary()
    assert list(s["funnel"]) == ["universe", "eligible", "p_volatility", "predicted_movers", "direction", "calibration", "health",
                                 "risk_filter", "positions"]
    t = decs[-1].table
    assert t.loc[~t["mover"], "p_up"].isna().all()                                    # no direction outside predicted movers
    assert (t.loc[t["side"] != 0, "mover"]).all()
    ev = TS.evaluate(decs, F, d[-1])
    G = F.copy()
    mv = TS.decisions_to_frame(decs)
    non = mv.index[~mv["mover"].to_numpy(bool)]
    G.loc[non, "up"] = 1.0 - G.loc[non, "up"]                                          # flip every NON-mover's direction
    ev2 = TS.evaluate(decs, G, d[-1])
    assert ev2.direction_acc == pytest.approx(ev.direction_acc) and ev2.n_direction == ev.n_direction
    assert ev.vol_auc > 0.6 and ev.n_direction > 0


def test_two_stage_refuses_leaky_frames_and_non_firewall_knowledge(two_stage_run):
    F, d, pipe, _ = two_stage_run
    now = d[60]
    with pytest.raises(FirewallBreach):
        pipe.decide(F[F.index.get_level_values(0) == now], now)                          # outcome columns on the decision day
    later = TS.point_in_time_state(F, d[62]).frame
    with pytest.raises(FirewallBreach):
        pipe.decide(later, now)                                                           # rows dated after now
    with pytest.raises(FirewallBreach):
        TS.run_day(pipe, F, TS.point_in_time_state(F, now).frame, now, release={"features": {"lv20": 1}})
    prov = Provenance("2024-01-01T00:00:00", str(pd.Timestamp(now).date()), "c")
    future = MaturedRecord("r1", str(pd.Timestamp(now).date()), {"features": {"gap_ratio": 1.0}, "lean": 0.0}, prov)
    with pytest.raises(FirewallBreach):
        TS.KnowledgeView.from_records([future], now)
    item = TV.TraderMemoryItem.make(TV.opaque_token({"f": "gap_ratio"}), "pattern", 1.0, {"gap_ratio": 1.0}, 0.0, 5)
    kv = TS.KnowledgeView.from_release(TV.TraderRelease(0, (item,)))
    assert kv.vol_features == ("gap_ratio",) and kv.digest != TS.KnowledgeView.empty().digest


def test_two_stage_null_world_keeps_direction_gate_closed_and_empty_is_explained():
    N = world("null", n_dates=50, n_names=40, direction=0.0)
    d = dates_of(N)
    pipe = TS.TwoStage(TS.TwoStageConfig(gate_min_weeks=6))
    dec = TS.run_day(pipe, N, TS.point_in_time_state(N, d[45]).frame, d[45])
    assert not dec.gate["open"] and (dec.table["side"] == 0).all()
    assert set(dec.table.loc[dec.table["mover"], "reason"]) <= {"DIRECTION_GATE_CLOSED", "NO_DIRECTION_MODEL"}
    empty = TS.TwoStage()
    rep = empty.fit(N.iloc[0:0], d[10])
    assert not rep.vol_ok and rep.notes
    dec0 = empty.decide(TS.point_in_time_state(N, d[10]).frame.iloc[0:0], d[10])
    assert len(dec0.table) == 0 and dec0.summary()["positions"] == 0
    ev = TS.evaluate([], N, d[-1])
    assert np.isnan(ev.vol_auc) and TS.capability_readings(ev, "2020-01-01") == []


def test_filed_knowledge_crosses_the_firewall_and_reaches_the_decision(tmp_path):
    """The promotion road after the quality gate: knowledge filed in MATURED_RESEARCH_STATE is released by the firewall at a later
    `now` and the two-stage chain uses its feature; the lineage records KNOWLEDGE -> RELEASE -> DECISION."""
    F = world()
    d = dates_of(F)
    feed = LP.FrameFeed(F, dates=d[34:35])
    c = cfg(checkpoint="off", disabled=_only("update.firewall_release", "evaluate.two_stage"))
    state, rt, _ = LP.open_loop(feed, tmp_path, c, clock=CLOCK)
    rt.obs = feed.observe(d[33])
    ctx = LP.Ctx(state, rt, str(pd.Timestamp(d[33]).date()), 0)
    through = str((pd.Timestamp(d[33]) - pd.Timedelta(days=3)).date())
    rec = LP.JobRecord("KEY", "BR1", "Q1", "r_Q1", "VOLATILITY", "gap_abs", Stage.INTEGRATION.value, through, through, ctx.now, 0, {},
                       result={"data_through": through, "sign": 1.0, "effect": 0.12})
    assert LP._file_knowledge(ctx, rec) == 1 and LP._file_knowledge(ctx, rec) == 0         # filed once, never twice
    rt.obs = None
    rep = LP.step(state, rt)
    by = {s["stage"]: s for s in rep["stages"]}
    assert by["update.firewall_release"]["status"] == "OK" and by["update.firewall_release"]["n_out"] == 1
    assert "gap_abs" in rt.pipe.report.vol_features
    assert state.lineage.knowledge_chains()


def test_a_fresh_start_moves_the_old_run_aside(tmp_path):
    """F17 (30 Sep): a fresh start inside an old run's folder reused its compute ledger, attempt folders and C68 chains (valid reruns
    rejected as stale_code; every C68 stage refused as LedgerTampered). The old run is moved aside intact and the new run starts empty."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "leftover.json").write_text("{}", encoding="utf-8")
    F = world()
    feed = LP.FrameFeed(F, dates=dates_of(F)[30:31])
    state, rt, info = LP.open_loop(feed, root, cfg(checkpoint="off"), fresh=True, clock=CLOCK)
    assert info["moved_aside"] and (Path(info["moved_aside"]) / "leftover.json").is_file()
    assert not (root / "leftover.json").exists()
    _, _, info2 = LP.open_loop(feed, tmp_path / "new", cfg(checkpoint="off"), fresh=True, clock=CLOCK)
    assert info2["moved_aside"] is None                              # nothing to move: an empty or missing folder is left alone
