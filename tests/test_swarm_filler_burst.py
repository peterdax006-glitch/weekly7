"""Fillers may burst when RAM is plentiful (3 Oct 2026): the 15 s start spacing let only ONE short thinking-drill job run at a time
(CPU 29-65%, 19 GB free all night). A filler now starts inside the spacing window only while free RAM above the floor covers every
start of that window plus this one at a pessimistic size; when RAM is short the plain spacing holds."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from creator import kernel as K
from creator import swarm as W


def gov(free_gb: float, ramp_s: float = 3600.0) -> W.Governor:
    return W.Governor(ramp_s=ramp_s, floor_min_gb=1.0, floor_fraction=0.0, per_worker_gb=0.4, max_workers=64, free=lambda: free_gb, burst_gb=1.0,
                      total=lambda: 32.0, observe=lambda n: None, free_disk=lambda: 500.0)


def test_the_rule() -> None:
    g = gov(free_gb=20.0)
    assert g.filler_ramped(since_last=3601.0, recent_starts=99, running=0)       # spacing passed: always
    assert g.filler_ramped(since_last=0.0, recent_starts=0, running=0)           # 19 GB above the floor covers a burst ...
    assert g.filler_ramped(since_last=0.0, recent_starts=18, running=0)          # ... of 19 starts at 1 GB each
    assert not g.filler_ramped(since_last=0.0, recent_starts=19, running=0)      # the 20th waits for the spacing
    tight = gov(free_gb=1.5)
    assert not tight.filler_ramped(since_last=0.0, recent_starts=0, running=0)   # RAM short: the plain spacing holds
    measured = W.Governor(ramp_s=3600.0, floor_min_gb=1.0, floor_fraction=0.0, free=lambda: 20.0, total=lambda: 32.0, burst_gb=1.0,
                          observe=lambda n: 4.0, free_disk=lambda: 500.0)
    assert not measured.filler_ramped(since_last=0.0, recent_starts=3, running=2)  # measured 5 GB workers: 4 x 5 > 19


def _round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_gb: float, drain_after_s: float = 0.0) -> int:
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "state", scratch=tmp_path / "scratch")
    planned = threading.Event()         # the round planned (and found nothing) before the first filler starts: a filler that wins the race
                                        # against the instant prepare() moves the ramp window and the round then never plans (hung under load)

    def plan_one(*a, **k):
        planned.set()
        return None
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", plan_one)
    monkeypatch.setattr(W.S, "head", lambda repo: "base")
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def job() -> None:
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.4)
        with lock:
            live["now"] -= 1
    t0 = time.monotonic()
    drain = (lambda: time.monotonic() - t0 > drain_after_s) if drain_after_s else None
    W.run_round(cfg, lambda: None, gov(free_gb), max_packages=1, poll_s=0.01, scheduled=False, filler=lambda: job if planned.is_set() else None, filler_budget=6,
                drain=drain)
    return live["peak"]


def test_fillers_overlap_inside_one_spacing_window_when_ram_is_plentiful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _round(tmp_path, monkeypatch, free_gb=20.0) >= 4         # ramp_s is an hour: without the burst only one would ever start


def test_tight_ram_keeps_the_spacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _round(tmp_path, monkeypatch, free_gb=1.9, drain_after_s=1.5) == 1   # one start, then the hour-long spacing; the drain ends it


def test_the_burst_is_opt_in() -> None:
    g = W.Governor(ramp_s=3600.0, floor_min_gb=1.0, floor_fraction=0.0, free=lambda: 20.0, total=lambda: 32.0, free_disk=lambda: 500.0)
    assert g.burst_gb is None and not g.filler_ramped(since_last=0.0, recent_starts=0, running=0)   # default: the plain spacing


def test_the_live_swarm_bursts_only_in_thinking_focus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import json
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("creator_swarm_burst_t", root / "scripts" / "creator_swarm.py")
    cs = importlib.util.module_from_spec(spec)                       # type: ignore[arg-type]
    spec.loader.exec_module(cs)                                      # type: ignore[union-attr]
    from creator import focus as F
    monkeypatch.setattr(cs, "STATE", tmp_path)
    monkeypatch.setattr(cs.REG, "optional", lambda name: F if name == "focus" else None)
    assert cs.thinking_burst_gb() is None                            # no focus file
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    assert cs.thinking_burst_gb() == 3.0
