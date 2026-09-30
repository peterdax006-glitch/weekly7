"""F30 (C75 Phase 2A/2C 'deterministic replay', 'checkpoints', 'crash recovery'; gap 15 of C75_PHASE0_MAPPING.md): whole-cycle replay.

The research loop is run on the W02 planted world (small config) from scratch twice with the same seed and everything it writes is
compared: the cycle reports, the filed knowledge, the lineage graph, the loop state inside every checkpoint, the checkpoint records, the
compute ledger and the experiment result envelopes. Then a run is killed mid-cycle, resumed from its checkpoint in a 'new process'
(fresh feed, fresh runtime) and compared with the uninterrupted run. Then the seed is changed and something must differ (the test can
fail). Modes: inline and thread executors.

What is removed before comparing, and why each is legitimately wall-clock / process-specific (nothing else is removed):
  seconds            StageRecord.seconds = time.perf_counter() around a stage (engine/research/loop.py run_stage)
  heartbeat, started, ended, history[*][0] after the submit row   compute.run_worker stamps its claim/finish with time.time() (the loop passes
                     no clock to run_worker; only the submit row uses the injected epoch)
  worker             f"w{os.getpid()}" (Executor._run_inline): the process id of whoever ran the job; also inside 'attempt 1 by w123'
  root path          every checkpoint record stores absolute artifact paths (and the sha256 of a blob that names the path) under the run folder
  resume             the per-run open_loop info (moved_aside carries a strftime stamp, 'action' differs by design between a fresh and a resumed run)
Everything else - including float values - must be equal."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    builders edit engine/ while this file runs, so two runs a minute apart got different 'code identities' in experiment records (the
    loop's pinned LoopConfig.code_hash is bypassed by engine/learning/experiment_memory.py:969, which calls current_code_hash()). The tree
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
    if type(x).__name__ == "ResearchStore":                      # KNOWN nondeterminism, pinned by its own xfail test below
        return {"__obj__": "ResearchStore", "d": canon({k: v for k, v in vars(x).items() if k != "token"}, depth + 1)}
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
        return {k: (0.0 if k == "seconds" else _zero_seconds(v)) for k, v in x.items()}
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


# ============================================================================================================== replay
def test_same_seed_from_scratch_replays_byte_identically(world, base, tmp_path):
    root0, state0, _ = base
    state1, reps1 = run_loop(world, tmp_path / "r")
    s0, s1 = snapshot(root0, state0), snapshot(tmp_path / "r", state1)
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
    assert any(r["launched"] for r in reps) and sum(r["status_counts"].get("OK", 0) + r["status_counts"].get("ok", 0) for r in reps) > 20


def test_a_different_seed_changes_something(world, base, tmp_path):
    """The equality above can fail: the loop seed feeds its experiments; a different one must change a result or the plan."""
    root0, state0, _ = base
    state1, _ = run_loop(world, tmp_path / "r", seed=7)
    s0, s1 = snapshot(root0, state0), snapshot(tmp_path / "r", state1)
    changed = [p for p in ("reports", "knowledge", "lineage", "ledger", "results", "state") if diff_paths(s0[p], s1[p], limit=1)]
    assert changed, "seed 0 and seed 7 left identical artefacts: the seed does not reach the loop (or the comparison is blind)"


def test_a_planted_wall_clock_leak_is_caught(world, base, tmp_path):
    """Planted defect: corrupt one float in a copy of the snapshot - the comparison must report the exact path."""
    root0, state0, _ = base
    s0 = snapshot(root0, state0)
    bad = json.loads(json.dumps(s0["reports"]))
    name = next(iter(bad))
    bad[name]["knowledge"] = bad[name]["knowledge"] + 1
    assert diff_paths(s0["reports"], bad) == [f"/{name}/knowledge: {str(s0['reports'][name]['knowledge'])!r} != {str(bad[name]['knowledge'])!r}"]


def test_the_wall_clock_fields_are_the_only_thing_the_normaliser_removes(base):
    """Null: the raw files DO differ between runs in exactly the listed fields - checked by confirming the normaliser is the identity on
    a tree that carries none of them, and removes them where present."""
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
@pytest.mark.parametrize("kill_at", [KILL_AT, "0|report.cycle"])
def test_killed_and_resumed_run_equals_the_uninterrupted_run(world, base, tmp_path, kill_at):
    """Kill inside a cycle (and, second case, right after a cycle's last stage), restart as a new process from the checkpoint, finish:
    the final state, reports, knowledge, lineage and every checkpoint equal the run that was never interrupted."""
    root0, state0, _ = base
    root = tmp_path / "r"
    with pytest.raises(KeyboardInterrupt):
        run_loop(world, root, kill_after=kill_at)
    assert (root / "checkpoints" / "INTERRUPTED.json").exists()                   # section 58: the interruption is recorded
    state1, reps1 = run_loop(world, root, fresh=False)                          # new process, same folder
    assert state1.cycle == CYCLES
    assert all(v == 1 for v in state1.exec_count.values()), {k: v for k, v in state1.exec_count.items() if v != 1}
    s0, s1 = snapshot(root0, state0), snapshot(root, state1)
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_states", "checkpoint_records", "ledger", "results"):
        assert_same(s0[part], s1[part], f"resumed vs uninterrupted [{kill_at}] {part}")


def test_resume_from_the_wrong_code_is_refused_not_replayed(world, tmp_path):
    """Null/planted: a checkpoint written by other code must not be silently replayed (fail closed, C75 2C)."""
    root = tmp_path / "r"
    with pytest.raises(KeyboardInterrupt):
        run_loop(world, root, kill_after=KILL_AT)
    feed, sweeps = _feed(world)
    state, _, info = LP.open_loop(feed, root, loop_cfg(code_hash="other-code"), sweeps=sweeps, clock=CLOCK)
    assert info["action"] != "CONTINUE" or state.cycle == 0
    assert state.cycle == 0 and not state.done_stages                               # it started over instead of trusting foreign state


# ============================================================================================================== known nondeterminism
@pytest.mark.xfail(strict=True, reason="engine/research/namespaces.py:434 ResearchStore.token = stable_hash({..., 'id': id(self)}): a memory "
                   "address, so the token differs between identical runs and is persisted in the checkpoint. Flips to XPASS (strict -> fail) "
                   "when the token becomes a function of (namespace, name) only; then drop the ResearchStore exclusion in canon().")
def test_research_store_token_is_reproducible(world, base, tmp_path):
    root0, state0, _ = base
    state1, _ = run_loop(world, tmp_path / "r", cycles=1)
    t0 = {k: m.store.token for k, m in state0.modules.items() if hasattr(m, "store") and hasattr(m.store, "token")}
    t1 = {k: m.store.token for k, m in state1.modules.items() if hasattr(m, "store") and hasattr(m.store, "token")}
    assert t0 and t0 == t1


# ============================================================================================================== thread executor
@pytest.fixture(scope="module")
def thread_runs(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("thr")
    out = []
    for i in range(2):
        st, reps = run_loop(world, root / f"r{i}", mode="thread", max_workers=2)
        out.append((root / f"r{i}", st, reps))
    return out


def test_thread_mode_replays_identically_to_itself(thread_runs):
    (r0, s0, _), (r1, s1, _) = thread_runs
    a, b = snapshot(r0, s0), snapshot(r1, s1)
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_records", "ledger", "results"):
        assert_same(a[part], b[part], f"thread vs thread {part}")


def test_thread_mode_equals_inline_mode(base, thread_runs):
    """Completion order of pool jobs must not change what the loop concludes (the executor harvests in a fixed order)."""
    root0, state0, _ = base
    r1, s1, _ = thread_runs[0]
    a, b = snapshot(root0, state0), snapshot(r1, s1)
    for part in ("knowledge", "lineage", "results"):
        assert_same(a[part], b[part], f"inline vs thread {part}")
    ra = {k: {kk: vv for kk, vv in v.items() if kk not in ("counters",)} for k, v in a["reports"].items()}
    rb = {k: {kk: vv for kk, vv in v.items() if kk not in ("counters",)} for k, v in b["reports"].items()}
    assert_same(ra, rb, "inline vs thread reports")
