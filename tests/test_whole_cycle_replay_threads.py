"""F30 (C75 2A/2C), thread-executor half of tests/test_whole_cycle_replay.py (split only to keep each file under 90 s). See that module for
the normaliser and the list of legitimately wall-clock fields.

FINDING: thread mode is NOT reproducible. st_harvest (engine/research/loop.py:1932) asks Executor.finished() (loop.py:668) which jobs have
returned, without waiting, so whether a pool job is harvested in cycle k or k+1 depends on thread scheduling: two identical thread runs
disagree on `harvested`, the AUDIT count, the lineage RESULT/MEMORY counts and the n_in/n_out of the harvest-dependent stages (observed:
cycle 1 harvested 2 vs 1). Inline mode is unaffected (a job has finished when submit returns)."""
from __future__ import annotations

import pytest

from tests.test_whole_cycle_replay import (CYCLES, assert_same, pinned_engine_tree, run_loop, snapshot, world)   # noqa: F401 (fixtures)


@pytest.fixture(scope="module")
def inline_run(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("inl") / "r"
    state, _ = run_loop(world, root)
    return root, state


@pytest.fixture(scope="module")
def thread_runs(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("thr")
    out = []
    for i in range(2):
        st, _ = run_loop(world, root / f"r{i}", mode="thread", max_workers=2)
        out.append((root / f"r{i}", st))
    return out


@pytest.mark.xfail(strict=False, reason="REAL NONDETERMINISM loop.py:1932 st_harvest / loop.py:668 Executor.finished(): harvest timing races the "
                   "thread pool (non-strict because a lucky schedule can pass). Fix: wait for the cycle's submitted jobs at a barrier before "
                   "harvest, or harvest strictly in submit order from a fixed cycle offset.")
def test_thread_mode_replays_identically_to_itself(thread_runs):
    (r0, s0), (r1, s1) = thread_runs
    a, b = snapshot(r0, s0), snapshot(r1, s1)
    for part in ("reports", "knowledge", "lineage", "state", "checkpoint_records", "ledger", "results"):
        assert_same(a[part], b[part], f"thread vs thread {part}")


def test_thread_mode_computes_the_same_experiments_as_inline_mode(inline_run, thread_runs):
    """Thread mode harvests jobs a stage later by design (the cycle continues while they run), so the lineage and plan may differ from
    inline. What must not differ: an experiment both modes ran (same key) has the same result envelope body."""
    a, b = snapshot(*inline_run)["results"], snapshot(*thread_runs[0])["results"]
    common = sorted(set(a) & set(b))
    assert common, "the two modes ran no experiment in common"
    for k in common:
        assert_same(a[k], b[k], f"inline vs thread result {k}")


def test_thread_mode_finishes_the_same_number_of_cycles(thread_runs):
    assert all(st.cycle == CYCLES for _, st in thread_runs)
