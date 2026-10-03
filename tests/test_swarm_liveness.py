"""A swarm round never waits forever (2 Oct 2026): a planned package QUEUED while RAM became too tight to start anything looped in
run_round with no way out. Now the round ends (RAM_TIGHT) and the queued plan is released as interrupted (not an attempt)."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from creator import kernel as K
from creator import planner as P2
from creator import swarm as W


def test_a_queued_plan_under_tight_ram_ends_the_round_and_is_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "state", scratch=tmp_path / "scratch")
    plans = iter([P2.Plan("g1", "K", "c1", "efficiency", None, "w1", "CPQ", "x", "e", 1)])
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: next(plans, None))
    monkeypatch.setattr(W.S, "head", lambda repo: "base")
    released: list[str] = []
    monkeypatch.setattr(W.P, "record_outcome", lambda led, plan, ok, reason: released.append(f"{plan.package_id}:{reason}"))
    monkeypatch.setattr(K, "execute", lambda *a, **k: pytest.fail("a plan must not start under tight RAM"))
    calls = {"n": 0}

    def free() -> float:                                       # room to PLAN, then too tight to START anything
        calls["n"] += 1
        return 10.0 if calls["n"] <= 3 else 0.1
    gov = W.Governor(ramp_s=3600.0, floor_min_gb=1.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=4, free=free,
                     total=lambda: 16.0, observe=lambda n: None, free_disk=lambda: 500.0)
    done = threading.Event()
    box: list[W.RoundReport] = []

    def go() -> None:
        box.append(W.run_round(cfg, lambda: None, gov, max_packages=1, poll_s=0.01, scheduled=False))
        done.set()
    threading.Thread(target=go, daemon=True).start()
    assert done.wait(60), "run_round never returned: a queued plan under tight RAM waits forever"
    assert box[0].outcome == "RAM_TIGHT"
    assert released and released[0].startswith("CPQ:interrupted")    # released, and 'interrupted' is not counted as an attempt


def test_fillers_run_while_prepare_runs_and_nothing_is_planned_before_it_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 Oct: K.prepare (minutes) ran before everything, so CPU tapered on start-up. Fillers start while it runs; planning waits for it."""
    import time as _t
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "state", scratch=tmp_path / "scratch")
    ev: list[str] = []
    prep_done = threading.Event()

    def prepare(cfg, led):                                     # type: ignore[no-untyped-def]
        _t.sleep(1.0)
        ev.append("prepare_end")
        prep_done.set()
        return object(), [], None

    def plan_one(*a, **k):                                     # type: ignore[no-untyped-def]
        ev.append("plan_after_prepare" if prep_done.is_set() else "PLAN_BEFORE_PREPARE")
        return None
    monkeypatch.setattr(K, "prepare", prepare)
    monkeypatch.setattr(K, "plan_one", plan_one)
    monkeypatch.setattr(W.S, "head", lambda repo: "base")

    def filler():                                              # type: ignore[no-untyped-def]
        def job() -> None:
            ev.append("filler_during_prepare" if not prep_done.is_set() else "filler_after")
        return job
    gov = W.Governor(ramp_s=0.05, floor_min_gb=0.0, floor_fraction=0.0, per_worker_gb=0.0, max_workers=4, free=lambda: 10.0,
                     total=lambda: 16.0, observe=lambda n: None)
    W.run_round(cfg, lambda: None, gov, max_packages=1, poll_s=0.02, filler=filler, filler_budget=6, scheduled=False)
    assert "filler_during_prepare" in ev and "PLAN_BEFORE_PREPARE" not in ev and "plan_after_prepare" in ev
