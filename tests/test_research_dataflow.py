"""W02: the research loop's data flow (C66 sections 3, 35, 42, 47, 53; memory lesson 'wired means reachable AND data flows'; canon
C63: planted worlds only). Every mechanism has a planted case it must catch, a null case and the empty case:

  feeds      the planted world hides its mechanisms where the tests can score them; the world slice, the stage-input audit and every
             builder are point-in-time; a short loop run on the planted feed carries data into every feed-supplied stage
  evidence   the full quality-gate bundle PROMOTES the planted genuine pattern, never the planted coincidence, and an empty frame
             stays NEEDS_MORE_EVIDENCE; nothing passes by default
  fold       science_memory's graph fold is idempotent when a node's known_at changes (folding twice == once), and the loop's
             direct writes are never counted twice
  leaks      a planted future item is refused AT the stage that received it
  resume     a loop killed inside a cycle resumes without redoing or skipping any stage, and the feed bookkeeping survives
  end to end the section-47 proof on the rich planted world (question -> ... -> PROMOTE -> firewall -> decision change -> outcome ->
             new question), long: it runs when W7_LONG_TESTS=1 (python -m pytest tests/test_research_dataflow.py -m integration)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.research import evidence as EV
from engine.research import feeds as FD
from engine.research import loop as LP
from engine.research import research_graph as RG
from engine.research import science_memory as SM
from engine.research import two_stage as TS
from engine.research.core import FirewallBreach

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731 - fixed wall clock: reruns are reproducible
CODE = "test-code"
SMALL = FD.PlantConfig(n_names=30, n_days=640, vol_state_sd=0.07, seed=3)
SMALL_FEED = FD.FeedConfig(first_decision="2018-03-02", max_knowability_moves=6, max_counterfactual_events=2, frontier_boot=50)
HEAVY_CADENCE = {"questions.discovery": 4, "surprises.multiscale": 4, "questions.interactions": 4, "missed.counterfactual": 2,
                 "evaluate.symmetry": 2}


def loop_cfg(**kw) -> LP.LoopConfig:
    base = dict(run_id="w02", free_gb=12.0, code_hash=CODE, checkpoint="cycle", cadence=dict(HEAVY_CADENCE),
                disabled=("evaluate.volatility_lab", "evaluate.direction_lab"), screen_top=2, max_new_questions=4,
                two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    base.update(kw)
    return LP.LoopConfig(**base)


@pytest.fixture(scope="module")
def small_world():
    return FD.planted_world(SMALL)


@pytest.fixture(scope="module")
def short_run(small_world, tmp_path_factory):
    feed = FD.WorldFeed(FD.InMemorySource(small_world), SMALL_FEED)
    root = tmp_path_factory.mktemp("flow")
    state, rt, _ = LP.open_loop(feed, root, loop_cfg(), sweeps=feed.sweeps(), clock=CLOCK)
    reps = [LP.step(state, rt) for _ in range(3)]
    return feed, state, rt, reps


# ============================================================================================================ the planted world
def test_planted_world_hides_known_mechanisms(small_world):
    """Genuine: trailing volatility predicts next week's +-10% touch in every period. Coincidence: cheap names jump ONLY in the early
    part of the sample (the noise pattern), so price_low looks predictive early and not later. Null: insider filings carry nothing."""
    from engine.pattern_movers import auc
    from engine.research import vol_hypotheses as VH
    w = small_world
    assert w.validate() == [] and set(w.truth["noise"]) == {"price_low"} and "lv20" in w.truth["genuine"]
    F = FD.research_frame(w)
    until = pd.Timestamp(w.truth["coincidence_until"])
    d = pd.to_datetime(F.index.get_level_values(0))

    def a(feat, mask):
        s = VH.derive(F[mask], (feat,))[feat].to_numpy(float)
        y = F[mask]["touch"].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(y)
        return auc(s[ok], y[ok] >= 0.5)
    early, late = np.asarray(d <= until), np.asarray(d > until + pd.Timedelta(days=60))
    assert a("lv20", late) > 0.65 and a("lv20", early) > 0.6                 # genuine everywhere
    assert a("price_low", early) > 0.55 and abs(a("price_low", late) - 0.5) < 0.05   # the coincidence dies
    assert abs(a("insider_recent", late) - 0.5) < 0.06                         # null
    assert F.attrs["survivor_free"] is True and {"days_to_event", "insider_n30", "sector"} <= set(F.columns)


def test_world_before_is_strictly_point_in_time(small_world):
    now = "2017-06-14"
    b = small_world.before(now)
    n = pd.Timestamp(now)
    assert b.sessions.max() < n and all(v.index.max() < n for v in b.market.values())
    assert (pd.to_datetime(b.events["accepted"]).dt.tz_convert("UTC").dt.tz_localize(None) < n).all()
    assert (pd.to_datetime(b.insider["filed"]) + pd.Timedelta(days=1) < n).all() and b.macro.index.max() < n
    assert len(small_world.before("2000-01-01").sessions) == 0                  # empty past: nothing, not an error


def test_input_audit_catches_nested_future_dates_and_passes_clean_ones():
    from engine.research import knowability as KB
    now = "2018-05-01"
    clean = {"frame": pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2018-04-30"])),
             "items": [KB.InfoItem("a", KB.InfoKind.EVENT, "X", "2018-04-01", "2018-04-01", "s")]}
    assert FD.audit_stage_input("s", clean, now) >= 2                           # it LOOKED at dated values
    leak_item = dict(clean, items=[KB.InfoItem("b", KB.InfoKind.EVENT, "X", "2018-05-02", "2018-05-02", "s")])
    with pytest.raises(FirewallBreach, match="published_at"):
        FD.audit_stage_input("s", leak_item, now)
    leak_frame = dict(clean, frame=pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2018-05-01"])))
    with pytest.raises(FirewallBreach, match="reaches 2018-05-01"):              # AT now is already the future
        FD.audit_stage_input("s", leak_frame, now)
    col = pd.DataFrame({"matured_at": pd.to_datetime(["2018-04-01", "2018-06-01"])})
    with pytest.raises(FirewallBreach, match="matured_at"):
        FD.audit_stage_input("s", {"rows": col}, now)
    assert FD.audit_stage_input("s", {}, now) == 0 and FD.audit_stage_input("s", None, now) == 0


def test_feed_config_validation_and_empty_source():
    assert FD.FeedConfig(horizon=0).validate() and FD.FeedConfig(max_counterfactual_events=0).validate()
    assert FD.FeedConfig().validate() == []
    with pytest.raises(ValueError):
        FD.planted_world(FD.PlantConfig(n_names=4))
    w = FD.planted_world(FD.PlantConfig(n_names=8, n_days=300))
    empty = FD.World({f: v.iloc[0:0] for f, v in w.bars.items()}, sectors=w.sectors)
    assert FD.research_frame(empty).empty
    feed = FD.WorldFeed(FD.InMemorySource(w), FD.FeedConfig(warm_weeks=500))
    assert feed.dates() == []                                                   # no decision date survives the warm-up


# ============================================================================================================ data reaches the stages
def test_short_run_carries_data_into_every_feed_stage(short_run):
    feed, state, rt, reps = short_run
    table = FD.input_table(reps).set_index("stage")
    for stage in FD.FEED_STAGES:
        assert table.loc[stage, "ok"] >= 1, (stage, table.loc[stage, "last_skip"])
        assert table.loc[stage, "failed"] == 0, (stage, table.loc[stage, "last_skip"])
    # after the warm-up (a decision needs its horizon to resolve) no feed stage is starved
    late = [s for r in reps[2:] for s in r["stages"] if s["stage"] in FD.FEED_STAGES and s["status"] == "SKIPPED_NO_INPUT"]
    assert late == [], late
    assert all(feed.audit_counts.get(s, 0) > 0 for s in ("missed.knowability", "evaluate.symmetry", "surprises.cross_section"))
    assert not any(r["failed"] for r in reps), [r["failed"] for r in reps]
    assert state.modules["knowability"].ledger.rows() and "unknown_cause" in state.modules


def test_builders_feed_nothing_twice(short_run):
    feed, state, rt, reps = short_run
    kn = state.memo["feed:knowability"]["done"]
    assert len(kn) == len(set(kn)) and len(state.modules["knowability"].ledger.rows()) == len(kn)
    assert len(set(state.memo["feed:observer"]["done"])) == len(state.memo["feed:observer"]["done"]) >= 1


def test_c68_slot_builds_expectations_and_realised_paths(short_run):
    feed, state, rt, reps = short_run
    rt.obs = feed.observe(state.now)
    ctx = LP.Ctx(state, rt, state.now, state.cycle)
    out = FD.c68_inputs(feed, ctx)
    e, p = out["expectations"], out["realised_paths"]
    assert {"p_move", "p_up", "side", "decided_at"} <= set(e.columns) and len(e) > 0
    assert len(p) and (pd.to_datetime(p["resolved_at"]) < pd.Timestamp(state.now)).all()
    assert set(p["session"]) <= set(range(1, feed.cfg.horizon + 1))
    rt.obs = None


def test_builders_without_decisions_raise_no_input_not_errors(small_world, tmp_path):
    feed = FD.WorldFeed(FD.InMemorySource(small_world), SMALL_FEED)
    state, rt, _ = LP.open_loop(feed, tmp_path, loop_cfg(checkpoint="off"), clock=CLOCK)
    now = feed.dates()[0]
    rt.obs = feed.observe(now)
    ctx = LP.Ctx(state, rt, now, 0)
    for stage in ("observe.observer", "evaluate.frontier", "questions.targets", FD.C68_SLOT):
        with pytest.raises(LP.NoInput):
            ctx.extra(stage, "x")


# ============================================================================================================ leaks
def test_planted_future_leak_is_refused_at_the_stage_that_received_it(small_world, tmp_path):
    feed = FD.WorldFeed(FD.InMemorySource(small_world), SMALL_FEED, leak=FD.LeakPlant("missed.knowability", "2000-01-01"))
    quiet = ("evaluate.volatility_lab", "evaluate.direction_lab", "questions.discovery", "surprises.multiscale", "surprises.cross_section",
             "evaluate.symmetry", "questions.interactions", "breaks.break_research")
    state, reps = LP.run(feed, tmp_path, loop_cfg(checkpoint="off", cadence={}, disabled=quiet), max_cycles=1, clock=CLOCK)
    by = {s["stage"]: s for s in reps[0]["stages"]}
    assert by["missed.knowability"]["status"] == "REFUSED_LEAK" and "published_at" in by["missed.knowability"]["reason"]
    assert "knowability" not in state.modules or not state.modules["knowability"].ledger.rows()   # the module never saw it
    assert by["missed.counterfactual"]["status"] == "OK" and by["observe.panel"]["status"] == "OK"  # nothing else was blamed
    assert state.counters.get("leaks_refused") == 1


# ============================================================================================================ evidence
@pytest.fixture(scope="module")
def matured():
    feed = FD.planted_feed(FD.PlantConfig(vol_state_sd=0.07))                  # the rich world: 48 names, four years
    now = feed.dates()[-1]
    F = feed.store.upto(now)
    return F[pd.to_datetime(F["end"]) < pd.Timestamp(now)], now


def _bundle(M, now, feature, **kw):
    spec = EV.FindingSpec("S_" + feature, feature, 1.0, "VOLATILITY", n_tests_searched=20, has_falsifier=True)
    return EV.assemble(M, spec, now, code_hash=CODE, data_hash="dh", created_real="2026-09-29T00:00:00+00:00",
                       cfg=EV.EvidenceConfig(identity_boot=60), **kw)


def test_full_evidence_promotes_the_genuine_pattern_and_never_the_coincidence(matured):
    M, now = matured
    good, noise = _bundle(M, now, "lv20"), _bundle(M, now, "price_low")
    rep = EV.gate([good, noise], now, CODE)
    v = EV.verdicts(rep)
    assert v["S_lv20"] == "PROMOTE", EV.blocking(rep, "S_lv20")
    assert v["S_price_low"] != "PROMOTE" and "out_of_sample" in EV.blocking(rep, "S_price_low")
    assert good.missing == {} and good.parts["repl_status"] == "REPLICATED" and good.parts["own_probe_caught"] is True
    ev = good.evidence
    for field in ("pit", "leak", "identity", "oos", "replication", "calibration", "risk", "complexity", "transfer", "failure", "repro",
                  "provenance"):
        assert getattr(ev, field) is not None, field                             # every gate has its evidence
    assert rep.promoted == ("S_lv20",) or list(rep.promoted) == ["S_lv20"]


def test_evidence_never_passes_by_default(matured):
    M, now = matured
    empty = EV.assemble(M.iloc[0:0], EV.FindingSpec("S0", "lv20"), now, code_hash=CODE, data_hash="dh", created_real="x")
    assert EV.verdicts(EV.gate([empty], now, CODE))["S0"] == "NEEDS_MORE_EVIDENCE" and empty.missing
    no_sector = _bundle(M.drop(columns=["sector"]), now, "lv20")                 # transfer cannot be measured -> MISSING, blocks
    rep = EV.gate([no_sector], now, CODE)
    assert "transfer" in no_sector.missing and EV.verdicts(rep)["S_lv20"] != "PROMOTE" and "transfer" in EV.blocking(rep, "S_lv20")
    with pytest.raises(FirewallBreach):                                          # a row maturing at/after now is a breach, not a filter
        _bundle(M, str(pd.to_datetime(M["end"]).max().date()), "lv20")
    with pytest.raises(ValueError):
        EV.assemble(M, EV.FindingSpec("S", "not_a_feature"), now, code_hash=CODE, data_hash="d", created_real="x")


def test_failure_evidence_explains_sampling_noise_but_not_a_real_collapse():
    se = EV.hanley_mcneil_se(0.8, 6, 40)
    assert 0.05 < se < 0.2 and EV.hanley_mcneil_se(0.8, 60, 400) < se and EV.hanley_mcneil_se(0.8, 0, 10) == float("inf")


# ============================================================================================================ the science-memory fold
def _graph():
    g = RG.ResearchGraph()
    p = g.add_pattern("volatility:lv20", "2018-01-05", label="lv20", effect=0.10).node_id
    g.add_test("exp1", p, True, "2018-01-12", 0.10, "auc")
    return g, p


def test_fold_is_idempotent_when_a_node_known_at_changes():
    g, p = _graph()
    m = SM.ScienceMemory()
    assert SM.ingest_graph(m, g, "2018-02-01") == {"proposed": 1, "tested": 1}
    g.add_pattern("volatility:lv20", "2018-03-02", label="lv20", effect=0.25)    # a NEW VERSION: new known_at, revised effect
    g.add_research_node(RG.K.EXPERIMENT, "exp1", "2018-03-02")                  # a bare re-add of the experiment (answer_question does this)
    once = SM.ingest_graph(m, g, "2018-04-01")
    n1 = len(m)
    assert once == {"versions": 1}                                              # the version is history (a NOTE), not a re-proposal
    assert SM.ingest_graph(m, g, "2018-04-01") == {} and len(m) == n1           # folding twice == folding once
    h = m.history(p, "2018-05-01")
    assert [e.stage.value for e in h].count("PROPOSED") == 1 and [e.stage.value for e in h].count("TESTED") == 1
    assert next(e for e in h if e.stage.value == "PROPOSED").known_at == "2018-01-05"   # first version, never overwritten
    assert next(e for e in h if e.stage.value == "TESTED").payload["method"] == "auc"   # first attrs survive the bare re-add


def test_fold_does_not_double_count_what_the_loop_wrote_directly():
    g, p = _graph()
    m = SM.ScienceMemory()
    m.propose(p, "2018-01-05", "lv20 ranks volatility outcomes", "research_loop", SM.Falsifier("effect_below", 0.0, "falls to chance"))
    m.tested(p, "2018-01-12", "auc STAGE1", "ADVANCE", 30, 0.1, 0.02, evidence=("exp:exp1",))
    before = len(m)
    assert SM.ingest_graph(m, g, "2018-02-01") == {} and len(m) == before


def test_fold_of_an_empty_graph_writes_nothing():
    m = SM.ScienceMemory()
    assert SM.ingest_graph(m, RG.ResearchGraph(), "2018-02-01") == {} and len(m) == 0


# ============================================================================================================ kill and resume
def test_killed_loop_resumes_mid_cycle_with_the_feed_state_intact(small_world, short_run, tmp_path):
    feed0, state0, _, reps0 = short_run
    feed = FD.WorldFeed(FD.InMemorySource(small_world), SMALL_FEED)
    cfg = loop_cfg(checkpoint="stage")
    with pytest.raises(KeyboardInterrupt):
        LP.run(feed, tmp_path, cfg, max_cycles=2, sweeps=feed.sweeps(), clock=CLOCK, kill_after="1|missed.knowability")
    feed2 = FD.WorldFeed(FD.InMemorySource(small_world), SMALL_FEED)          # a fresh process: nothing transient survives
    state, reps = LP.run(feed2, tmp_path, cfg, max_cycles=1, sweeps=feed2.sweeps(), clock=CLOCK)
    assert [r["cycle"] for r in reps] == [1]
    assert all(v == 1 for v in state.exec_count.values())                      # no stage ran twice, none was skipped
    assert len(state.exec_count) == 2 * len(LP.STAGES)
    done0 = state0.memo["feed:knowability"]["done"]
    kn = state.memo["feed:knowability"]["done"]
    assert set(kn) <= set(done0) and len(kn) == len(set(kn)) and kn           # the same moves, none twice
    for a, b in zip(reps0[1:2], reps):
        sa = {s["stage"]: (s["status"], s["n_in"]) for s in a["stages"] if s["stage"] in FD.FEED_STAGES}
        sb = {s["stage"]: (s["status"], s["n_in"]) for s in b["stages"] if s["stage"] in FD.FEED_STAGES}
        assert sa == sb


# ============================================================================================================ the section-47 proof
LONG = os.environ.get("W7_LONG_TESTS") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not LONG, reason="long end-to-end run: set W7_LONG_TESTS=1 (about 4-6 minutes)")
def test_section47_end_to_end_on_the_rich_planted_world(tmp_path):
    """(a) data reaches every stage (none starved; after warm-up no feed stage skips), (b) the planted genuine pattern goes question ->
    experiment -> replication -> quality gate PROMOTE -> firewall -> live release -> two-stage decision change -> outcome measured ->
    new question, (c) the planted coincidence is never promoted, (d) nothing reached a stage from the future."""
    world = FD.planted_world(FD.PlantConfig(vol_state_sd=0.07))
    feed = FD.WorldFeed(FD.InMemorySource(world), FD.FeedConfig(first_decision="2018-06-01"))
    state, rt, _ = LP.open_loop(feed, tmp_path, loop_cfg(), sweeps=feed.sweeps(), clock=CLOCK)
    reps, promoted_at = [], None
    for _ in range(40):
        rep = LP.step(state, rt)
        if rep is None:
            break
        reps.append(rep)
        if promoted_at is None and state.knowledge:
            promoted_at = rep["cycle"]
        if promoted_at is not None and rep["cycle"] >= promoted_at + 4 and state.lineage.section47_chains():
            break
    table = FD.input_table(reps)
    assert FD.starved_stages(reps, exempt=("evaluate.volatility_lab", "evaluate.direction_lab")) == [], table.to_string()
    assert not any(r["failed"] for r in reps), [r["failed"] for r in reps if r["failed"]]
    assert not any(r["refused"] for r in reps)
    # (b) PROMOTE -> knowledge -> release -> decision
    assert promoted_at is not None, [r.get("ladder") for r in reps]
    gates = {k: v for k, v in state.lineage.nodes.items() if v["kind"] == "GATE"}
    assert any(v.get("verdict") == "PROMOTE" for v in gates.values())
    filed = {k["feature"] for k in state.knowledge.values()}
    assert filed and filed <= set(world.truth["genuine"]) | set(world.truth["genuine_event"])
    assert state.lineage.knowledge_chains(), "knowledge never reached a decision through the firewall"
    # decision change: the released knowledge changes P(volatility) against the same chain without it
    obs = feed.observe(state.decisions[-1].decided_at)
    base = TS.TwoStage(state.cfg.two_stage)
    no_k = TS.run_day(base, obs.matured, obs.today, state.decisions[-1].decided_at, None)
    with_k = state.decisions[-1]
    assert with_k.knowledge_digest != no_k.knowledge_digest
    assert not np.allclose(with_k.table["p_move"].fillna(0).to_numpy(), no_k.table["p_move"].fillna(0).to_numpy())
    # outcome measured after filing, and a new question raised from results
    assert any("post_effect" in k for k in state.knowledge.values())
    assert state.lineage.section47_chains()
    # (c) the coincidence never promoted
    assert not (filed & set(world.truth["noise"]))
