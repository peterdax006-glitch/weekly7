"""F30 (C75 Phase 2A/2C 'deterministic replay', 'checkpoints', 'crash recovery'; gap 15 of C75_PHASE0_MAPPING.md): whole-cycle replay.

The research loop is run on the W02 planted world (small config) from scratch twice with the same seed and everything it writes is
compared: the cycle reports, the filed knowledge, the lineage graph, the loop state inside every checkpoint, the checkpoint records, the
compute ledger and the experiment result envelopes. Then a run is killed mid-cycle, resumed from its checkpoint in a 'new process'
(fresh feed, fresh runtime) and compared with the uninterrupted run. Then the seed is changed and something must differ (the test can
fail). Modes: inline here, thread in test_whole_cycle_replay_threads.py.

What is removed before comparing, and why each is legitimately wall-clock / process-specific (nothing else is removed):
  seconds            StageRecord.seconds = time.perf_counter() around a stage (engine/research/loop.py run_stage)
  heartbeat, started, ended, history[*][0] after the submit row   compute.run_worker stamps its claim/finish with time.time() (the loop passes
                     no clock to run_worker; only the submit row uses the injected epoch)
  worker             f"w{os.getpid()}" (Executor._run_inline): the process id of whoever ran the job; also inside 'attempt 1 by w123'
  root path          every checkpoint record stores absolute artifact paths (and the sha256 of a blob that names the path) under the run folder
  resume             the per-run open_loop info (moved_aside carries a strftime stamp, 'action' differs by design between a fresh and a resumed run)
Everything else - including float values, every Provenance.created_real (F31: stamped from the injected clock) and every
ResearchStore.token (F31: derived from namespace, name and class, no longer a memory address) - must be equal."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import pickle
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.learning.checkpoints import StaleState
from engine.research import feeds as FD
from engine.research import loop as LP
from engine.research import two_stage as TS

warnings.filterwarnings("ignore")
CLOCK = lambda: 1_700_000_000.0                                   # noqa: E731 - injected wall clock: the loop's own reads are reproducible
CODE = "test-code"
SMALL = FD.PlantConfig(n_names=30, n_days=640, vol_state_sd=0.07, seed=3)
SMALL_FEED = FD.FeedConfig(first_decision="2018-03-02", max_knowability_moves=6, max_counterfactual_events=2, frontier_boot=50)
HEAVY_CADENCE = {"questions.discovery": 4, "surprises.multiscale": 4, "questions.interactions": 4, "missed.counterfactual": 2,
                 "evaluate.symmetry": 2}
CYCLES = 2
KILL_AT = "1|missed.knowability"                                  # mid cycle 1: later stages of the cycle are still to run
WALL_KEYS = {"seconds", "heartbeat", "started", "ended", "worker"}


def loop_cfg(**kw) -> LP.LoopConfig:
    base = dict(run_id="w02", free_gb=12.0, code_hash=CODE, checkpoint="stage", cadence=dict(HEAVY_CADENCE),
                disabled=("evaluate.volatility_lab", "evaluate.direction_lab"), screen_top=2, max_new_questions=4,
                two_stage=TS.TwoStageConfig(gate_min_weeks=6, min_direction_rows=60, min_calib_rows=20))
    base.update(kw)
    return LP.LoopConfig(**base)


@pytest.fixture(scope="module", autouse=True)
def pinned_engine_tree():
    """ENVIRONMENT, not loop nondeterminism: engine.provenance.engine_tree_hash() hashes every engine source file on disk, and other
    builders edit engine/ while this file runs, so two runs a minute apart got different 'code identities' wherever a module that has no
    injected code hash falls back to current_code_hash() (the loop's experiment records now carry LoopConfig.code_hash - F31). The tree
    hash is pinned for the replay comparison so only the loop's own behaviour is compared."""
    from engine import provenance
    mp = pytest.MonkeyPatch()
    mp.setattr(provenance, "engine_tree_hash", lambda *a, **k: "pinned-engine-tree")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def world():
    return FD.planted_world(SMALL)


def _feed(world):
    feed = FD.WorldFeed(FD.InMemorySource(world), SMALL_FEED)
    return feed, feed.sweeps()


def run_loop(world, root, cycles=CYCLES, kill_after=None, fresh=True, **kw):
    """One process's worth of loop: a NEW feed and runtime each call, like a restart."""
    feed, sweeps = _feed(world)
    return LP.run(feed, root, loop_cfg(**kw), max_cycles=cycles, sweeps=sweeps, clock=CLOCK, fresh=fresh, kill_after=kill_after)


# ============================================================================================================== canonical form
def canon(x, depth=0):
    """A deterministic, JSON-able image of any loop object. Sets are sorted, frames become csv text, numpy becomes python; an object
    whose repr would carry a memory address is reduced to its type and __dict__ so two equal runs compare equal."""
    if depth > 40:
        return "<deep>"
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return "nan" if x != x else x
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return canon(x.item(), depth)
    if isinstance(x, np.ndarray):
        return {"__nd__": str(x.dtype), "v": canon(x.tolist(), depth + 1)}
    if isinstance(x, pd.DataFrame):
        return {"__df__": x.to_csv()}
    if isinstance(x, pd.Series):
        return {"__sr__": x.to_csv()}
    if isinstance(x, pd.Timestamp):
        return str(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__dc__": type(x).__name__, **{f.name: canon(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}}
    if isinstance(x, dict) or hasattr(x, "items") and hasattr(x, "keys"):
        return {"__m__": [[canon(k, depth + 1), canon(v, depth + 1)] for k, v in sorted(((k, v) for k, v in x.items()), key=lambda kv: repr(kv[0]))]}
    if isinstance(x, (set, frozenset)):
        return {"__s__": sorted((canon(v, depth + 1) for v in x), key=lambda v: json.dumps(v, sort_keys=True, default=str))}
    if isinstance(x, (list, tuple)):
        return [canon(v, depth + 1) for v in x]
    if isinstance(x, (bytes, bytearray)):
        return hashlib.sha256(bytes(x)).hexdigest()
    if hasattr(x, "value") and type(x).__module__ != "builtins" and hasattr(type(x), "__members__"):
        return f"{type(x).__name__}.{x.name}"
    r = repr(x)
    if re.search(r"0x[0-9a-fA-F]{6,}", r):
        d = getattr(x, "__dict__", None)
        return {"__obj__": type(x).__name__, "d": canon(d, depth + 1) if d is not None else None}
    return r


def strip_wall(x):
    """Remove the legitimate wall-clock / process fields (module docstring) from a parsed JSON tree."""
    if isinstance(x, dict):
        return {k: strip_wall(v) for k, v in x.items() if k not in WALL_KEYS and k != "resume"}
    if isinstance(x, list):
        return [strip_wall(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"by w\d+", "by w<pid>", x)
    return x


def strip_ledger_history(ledger: dict) -> dict:
    """history rows are [time, state, note]; the submit row carries the injected epoch, later rows carry time.time()."""
    out = {}
    for k, e in ledger.items():
        e = dict(e)
        e["history"] = [[t if note == "submitted" else "<wall>", st, note] for t, st, note in e.get("history", [])]
        out[k] = e
    return out


def _norm_paths(text: str, root: Path) -> str:
    return text.replace(str(root), "<ROOT>").replace(str(root).replace("\\", "\\\\"), "<ROOT>")


def snapshot(root: Path, state) -> dict:
    """Everything the run left behind, normalised. Keys: reports, knowledge, lineage, state (live LoopState), checkpoint_states (every
    pickled state in the rolling store), checkpoint_records, ledger, results."""
    root = Path(root)
    snap: dict = {}
    snap["reports"] = {p.name: strip_wall(json.loads(p.read_text("utf-8"))) for p in sorted((root / "reports").glob("cycle_*.json"))}
    snap["knowledge"] = canon(state.knowledge)
    snap["lineage"] = canon(state.lineage)
    snap["state"] = canon(_scrub_state(state))
    cdir = root / "checkpoints" / state.cfg.run_id
    snap["checkpoint_states"] = {p.name: canon(_scrub_state(pickle.loads(p.read_bytes()))) for p in sorted(cdir.glob("state_*.pkl"))}
    recs = {}
    for p in sorted(cdir.glob("cp_*.json")):
        body = json.loads(_norm_paths(p.read_text("utf-8"), root))["body"]
        body["artifacts"] = sorted(body["artifacts"])                      # the digests name the path; the states are compared above
        recs[p.name] = body
    snap["checkpoint_records"] = recs
    lp = root / "compute" / "ledger.json"
    snap["ledger"] = strip_wall(strip_ledger_history(json.loads(lp.read_text("utf-8")))) if lp.exists() else {}
    snap["results"] = {str(p.relative_to(root)).replace("\\", "/"): strip_wall(json.loads(p.read_text("utf-8")))
                       for p in sorted((root / "compute" / "out").glob("*/attempt_*/result.json"))}
    return snap


def _scrub_state(state):
    """A copy of the LoopState with only the per-stage wall-clock timings zeroed (they are inside the bus records and the reports)."""
    st = dataclasses.replace(state)
    st.bus = _zero_seconds(state.bus)
    st.reports = _zero_seconds(state.reports)
    return st


def _zero_seconds(x):
    if isinstance(x, dict):
        return {k: (0.0 if k == "seconds" else _zero_seconds(v)) for k, v in x.items() if k != "resume"}
    if isinstance(x, list):
        return [_zero_seconds(v) for v in x]
    return x


def diff_paths(a, b, path="", limit=12) -> list[str]:
    """Where two canonical trees differ (first `limit` paths), so a failure names the nondeterministic field."""
    out: list[str] = []

    def rec(x, y, p):
        if len(out) >= limit:
            return
        if type(x) is not type(y):
            out.append(f"{p}: type {type(x).__name__} != {type(y).__name__}")
        elif isinstance(x, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y:
                    out.append(f"{p}/{k}: only in {'A' if k in x else 'B'}")
                else:
                    rec(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list):
            if len(x) != len(y):
                out.append(f"{p}: len {len(x)} != {len(y)}")
            for i, (u, v) in enumerate(zip(x, y)):
                rec(u, v, f"{p}[{i}]")
        elif x != y and not (x != x and y != y):                          # NaN == NaN here: both runs had no value
            out.append(f"{p}: {str(x)[:80]!r} != {str(y)[:80]!r}")
    rec(a, b, path)
    return out


def assert_same(a: dict, b: dict, what: str):
    bad = diff_paths(a, b)
    assert not bad, f"{what} differ:\n  " + "\n  ".join(bad)



# ============================================================================================================== fixtures
@pytest.fixture(scope="module")
def base(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("base")
    state, reps = run_loop(world, root / "r")
    assert len(reps) == CYCLES and state.cycle == CYCLES
    return root / "r", state, reps


@pytest.fixture(scope="module")
def repeat(world, tmp_path_factory):
    """The same configuration run again from scratch in a different folder."""
    root = tmp_path_factory.mktemp("repeat") / "r"
    state, reps = run_loop(world, root)
    return root, state, reps


@pytest.fixture(scope="module")
def killed_mid_cycle(world, tmp_path_factory):
    """A run killed inside cycle 1, its folder left exactly as the crash left it; tests copy it before resuming."""
    root = tmp_path_factory.mktemp("killed") / "r"
    with pytest.raises(KeyboardInterrupt):
        run_loop(world, root, kill_after=KILL_AT)
    return root


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# ============================================================================================================== replay
def test_same_seed_from_scratch_replays_byte_identically(base, repeat):
    root0, state0, _ = base
    root1, state1, _ = repeat
    s0, s1 = snapshot(root0, state0), snapshot(root1, state1)
    assert s0["knowledge"] is not None and s0["lineage"] and s0["reports"] and s0["checkpoint_states"]      # the comparison has content
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_states", "checkpoint_records", "ledger", "results"):
        assert_same(s0[part], s1[part], part)
    # the serialised cycle reports are byte-equal once seconds are gone: the stronger statement the section asks for
    a = json.dumps(s0["reports"], sort_keys=True).encode()
    b = json.dumps(s1["reports"], sort_keys=True).encode()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_the_loop_did_real_work_so_equality_is_not_vacuous(base):
    _, state, reps = base
    assert len(state.jobs) > 0 and len(state.questions) > 0 and len(state.lineage.nodes) > 10
    assert any(r["launched"] for r in reps) and sum(r["status_counts"].get("OK", 0) for r in reps) > 20


def test_a_different_seed_changes_something(base, world, tmp_path):
    """The equality above can fail: the loop seed feeds its experiments; a different one must change a result, the ledger or a report."""
    root0, state0, _ = base
    state1, _ = run_loop(world, tmp_path / "r", cycles=1, seed=7)
    s0, s1 = snapshot(root0, state0), snapshot(tmp_path / "r", state1)
    one = {"cycle_00000.json": s0["reports"]["cycle_00000.json"]}                      # seed 0's first cycle vs seed 7's only cycle
    changed = [p for p, a, b in (("reports", one, s1["reports"]), ("results", s0["results"], s1["results"]),
                                 ("ledger", s0["ledger"], s1["ledger"])) if diff_paths(a, b, limit=1)]
    assert changed, "seed 0 and seed 7 left identical artefacts: the seed does not reach the loop (or the comparison is blind)"


def test_a_planted_difference_is_caught_with_its_exact_path(base):
    """Planted defect: corrupt one number in a copy of the snapshot - the comparison must report the exact path."""
    root0, state0, _ = base
    s0 = snapshot(root0, state0)
    bad = json.loads(json.dumps(s0["reports"]))
    name = next(iter(bad))
    bad[name]["knowledge"] = bad[name]["knowledge"] + 1
    assert diff_paths(s0["reports"], bad) == [f"/{name}/knowledge: {str(s0['reports'][name]['knowledge'])!r} != {str(bad[name]['knowledge'])!r}"]


def test_the_normaliser_removes_only_the_listed_wall_clock_fields():
    """Null: it is the identity on a tree that carries none of them, and removes them where present."""
    tree = {"a": [1, {"seconds": 0.3, "b": 2}], "worker": "w9", "note": "attempt 1 by w123", "resume": {"x": 1}}
    assert strip_wall(tree) == {"a": [1, {"b": 2}], "note": "attempt 1 by w<pid>"}
    plain = {"a": [1, {"b": 2}], "c": "d"}
    assert strip_wall(plain) == plain


def test_empty_feed_replays_to_nothing(tmp_path):
    """Degenerate case: a feed with no decision dates runs zero cycles, identically, and leaves no reports."""
    empty = FD.planted_world(FD.PlantConfig(n_names=10, n_days=320, seed=1))
    feed = FD.WorldFeed(FD.InMemorySource(empty), FD.FeedConfig(first_decision="2031-01-01"))
    out = [LP.run(feed, tmp_path / f"e{i}", loop_cfg(), max_cycles=2, clock=CLOCK, fresh=True)[1] for i in range(2)]
    assert out == [[], []]


# ============================================================================================================== crash recovery
def _resume_and_compare(base, world, root, cycles_after):
    root0, state0, reps0 = base
    state1, reps1 = run_loop(world, root, cycles=cycles_after, fresh=False)      # new process (new feed, new runtime), same folder
    return state1, reps0, reps1, snapshot(root0, state0), snapshot(root, state1)


def test_killed_mid_cycle_and_resumed_run_equals_the_uninterrupted_run(world, base, killed_mid_cycle, tmp_path):
    """Kill inside cycle 1 (after missed.knowability), restart as a new process from the checkpoint, finish cycle 1: the final state,
    reports, knowledge, lineage and every checkpoint equal the run that was never interrupted, and no stage ran twice."""
    root = _copy(killed_mid_cycle, tmp_path / "r")
    assert (root / "checkpoints" / "INTERRUPTED.json").exists()                   # section 58: the interruption is recorded
    state1, _, reps1, s0, s1 = _resume_and_compare(base, world, root, 1)
    assert state1.cycle == CYCLES and [r["cycle"] for r in reps1] == [1]
    assert all(v == 1 for v in state1.exec_count.values()), {k: v for k, v in state1.exec_count.items() if v != 1}
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_states", "checkpoint_records", "ledger", "results"):
        assert_same(s0[part], s1[part], f"resumed vs uninterrupted {part}")


@pytest.mark.parametrize("kill", ["0|report.cycle", "0|close"])
def test_killed_at_the_cycle_boundary_loses_no_date_and_runs_no_stage_twice(world, base, tmp_path, kill):
    """F31 hard kill at the exact boundary that lost dates: after the report stage's checkpoint but before the cycle is closed
    ('0|report.cycle'), and right after the close's own checkpoint ('0|close'). Before the fix, the first left every stage of cycle 0
    marked done with the counter still at 0, so the resumed run took the NEXT date as a 'resumed' cycle, ran no stage and lost it.
    Now: the resumed run closes cycle 0 without running anything, researches every date exactly once, and ends equal to the
    uninterrupted run in every artefact."""
    root = tmp_path / "r"
    with pytest.raises(KeyboardInterrupt):
        run_loop(world, root, cycles=1, kill_after=kill)
    after_close = kill.endswith("close")
    state1, reps0, reps1, s0, s1 = _resume_and_compare(base, world, root, 1 if after_close else 2)
    dates0 = [r["now"] for r in reps0]
    done = sorted(json.loads(p.read_text("utf-8"))["now"] for p in (root / "reports").glob("cycle_*.json"))
    assert done == dates0, f"resumed run researched {done}, the uninterrupted run {dates0}"
    assert [r["cycle"] for r in reps1] == ([1] if after_close else [0, 1])            # cycle 0 is closed, never re-run
    assert state1.cycle == CYCLES
    assert all(v == 1 for v in state1.exec_count.values()), {k: v for k, v in state1.exec_count.items() if v != 1}
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_states", "checkpoint_records", "ledger", "results"):
        assert_same(s0[part], s1[part], f"resumed vs uninterrupted {part}")


def test_a_completed_but_unclosed_cycle_is_detected_and_an_open_one_is_not(base):
    """Null + planted for the detector alone: a state whose current cycle has every stage done is 'complete'; one stage short, or no
    stages at all (a fresh cycle), is not."""
    state = dataclasses.replace(base[1])
    state.done_stages = {state.cycle: list(LP.STAGE_NAMES)}
    assert LP._cycle_complete(state)
    state.done_stages = {state.cycle: list(LP.STAGE_NAMES[:-1])}
    assert not LP._cycle_complete(state)
    state.done_stages = {}
    assert not LP._cycle_complete(state)


def test_resume_from_the_wrong_code_is_refused_not_replayed(world, killed_mid_cycle, tmp_path):
    """Null/planted: a checkpoint written by other code must not be silently replayed (fail closed, C75 2C)."""
    root = _copy(killed_mid_cycle, tmp_path / "r")
    feed, sweeps = _feed(world)
    with pytest.raises(StaleState, match="other-code"):
        LP.open_loop(feed, root, loop_cfg(code_hash="other-code"), sweeps=sweeps, clock=CLOCK)


# ============================================================================================================== known nondeterminism
def test_research_store_token_is_reproducible(base, repeat):
    """F31: namespaces.ResearchStore.token hashed id(self) and was persisted in every checkpoint; now stable content only."""
    state0, state1 = base[1], repeat[1]
    t0 = {k: m.store.token for k, m in state0.modules.items() if hasattr(m, "store") and hasattr(m.store, "token")}
    t1 = {k: m.store.token for k, m in state1.modules.items() if hasattr(m, "store") and hasattr(m.store, "token")}
    assert t0 and t0 == t1
    from engine.research import namespaces as NS
    a, b = NS.ResearchStore("x:research"), NS.ResearchStore("x:research")
    assert a.token == b.token and a is not b                                   # two objects, one identity: no memory address
    pair = NS.NamespacePair("x")
    assert pair.research.token != pair.live.token                              # the separation check still has distinct tokens


def test_every_record_is_stamped_with_the_injected_clock(base):
    """F31: frontier, symmetry, counterfactual, volatility_lab and knowledge.make_provenance stamped datetime.now(); inside a loop stage
    they now read the loop's injected clock (knowledge.provenance_clock)."""
    c = json.dumps(canon(_scrub_state(base[1])), default=str)
    stamps = set(re.findall(r'"created_real": "([^"]+)"', c))
    injected = dt.datetime.fromtimestamp(CLOCK(), dt.timezone.utc).isoformat(timespec="seconds")
    # KNOWN, deterministic, out of F31's files: cross_section.py:784, multiscale.py:909, regimes.py:781/1978 and learning/postmortem.py:429
    # stamp created_real with the simulated decision date (str(as_date(now))). Reproducible, so allowed here; never the real time.
    # (The five fixed modules' records do not reach LoopState in this small configuration; the two tests below prove their routing.)
    sim_dates = {r["now"] for r in base[1].reports}
    assert stamps, "no created_real in the state: the check would be vacuous"
    assert stamps <= {injected} | sim_dates, sorted(stamps - {injected} - sim_dates)[:5]


def test_a_loop_stage_stamps_records_with_the_injected_clock(base):
    """Planted: a stage that files a provenance record through run_stage gets the loop's clock, not the real time."""
    import types as _types
    from engine.learning import knowledge as KN
    got = []
    spec = LP.StageSpec("observe.panel", LP.LoopPhase.OBSERVE, lambda ctx: (got.append(KN.make_provenance("2018-03-01").created_real) or (1, 1, "")))
    state = dataclasses.replace(base[1], exec_count={})
    rec = LP.run_stage(spec, LP.Ctx(state, _types.SimpleNamespace(clock=CLOCK), "2018-03-02", 0))
    assert rec.status == LP.StageStatus.OK, rec.reason
    assert got == [dt.datetime.fromtimestamp(CLOCK(), dt.timezone.utc).isoformat(timespec="seconds")]


def test_the_fixed_modules_take_created_real_from_the_injected_clock():
    """The five F31 sites no longer read datetime.now() for Provenance.created_real; each routes through knowledge.wall_stamp. A
    source check (each module's real entry needs a full fixture world), counted so a site that regresses is named."""
    root = Path(LP.__file__).resolve().parents[1]
    for rel in ("research/frontier.py", "research/symmetry.py", "research/counterfactual.py", "research/volatility_lab.py", "learning/knowledge.py"):
        src = (root / rel).read_text("utf-8")
        bad = [ln.strip()[:100] for ln in src.splitlines() if "Provenance(" in ln and "datetime.now(" in ln]
        assert not bad, f"{rel}: {bad}"
        assert "wall_stamp" in src, rel


def test_the_provenance_clock_is_scoped_and_the_default_is_real_time():
    """Planted + null: inside provenance_clock a stamp is the injected time; outside (no clock exists) it is the real time; nesting
    restores the outer clock."""
    from engine.learning import knowledge as KN
    with KN.provenance_clock(lambda: 0.0):
        assert KN.wall_stamp() == "1970-01-01T00:00:00+00:00"
        assert KN.make_provenance("2020-01-02").created_real == "1970-01-01T00:00:00+00:00"
        with KN.provenance_clock(CLOCK):
            assert KN.wall_stamp() == dt.datetime.fromtimestamp(CLOCK(), dt.timezone.utc).isoformat(timespec="seconds")
        assert KN.wall_stamp() == "1970-01-01T00:00:00+00:00"
    real = dt.datetime.fromisoformat(KN.wall_stamp())
    assert abs((real - dt.datetime.now(dt.timezone.utc)).total_seconds()) < 60


def test_experiment_records_carry_the_runs_configured_code_hash(base):
    """F31: the loop pins LoopConfig.code_hash; its experiment ledger records must carry it (they took the engine tree hash).
    new_record honours a configured hash and, with none, keeps the engine tree hash (the registry / pre-launch path)."""
    from engine.learning import experiment_memory as EM
    from engine.learning.core import current_code_hash
    state = base[1]
    c = json.dumps(canon(_scrub_state(state)), default=str)
    hashes = set(re.findall(r'"__dc__": "DesignSpec".*?"code_hash": "([^"]*)"', c))
    assert hashes, "the loop filed no experiment design: the check would be vacuous"
    assert hashes == {CODE}, hashes
    hyp = (EM.Hypothesis("h1", "x", 0.5), EM.Hypothesis("h0", "noise", 0.5))
    args = ("e1", "q?", "b", hyp, EM.Prediction("p"), EM.DesignSpec(config={"a": 1}), EM.uniform_expected(hyp, ("up", "flat"), {}), "2020-01-02")
    assert EM.new_record(*args, code_hash="run-pin").experiment.code_hash == "run-pin"
    assert EM.new_record(*args).experiment.code_hash == current_code_hash()
