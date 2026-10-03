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
    return W.Governor(ramp_s=ramp_s, floor_min_gb=1.0, floor_fraction=0.0, per_worker_gb=0.4, max_workers=64, free=lambda: free_gb,
                      total=lambda: 32.0, observe=lambda n: None, free_disk=lambda: 500.0)


def test_the_rule() -> None:
    g = gov(free_gb=20.0)
    assert g.filler_ramped(since_last=3601.0, recent_starts=99, running=0)       # spacing passed: always
    assert g.filler_ramped(since_last=0.0, recent_starts=0, running=0)           # 19 GB above the floor covers a burst ...
    assert g.filler_ramped(since_last=0.0, recent_starts=18, running=0)          # ... of 19 starts at 1 GB each
    assert not g.filler_ramped(since_last=0.0, recent_starts=19, running=0)      # the 20th waits for the spacing
    tight = gov(free_gb=1.5)
    assert not tight.filler_ramped(since_last=0.0, recent_starts=0, running=0)   # RAM short: the plain spacing holds
    measured = W.Governor(ramp_s=3600.0, floor_min_gb=1.0, floor_fraction=0.0, free=lambda: 20.0, total=lambda: 32.0,
                          observe=lambda n: 4.0, free_disk=lambda: 500.0)
    assert not measured.filler_ramped(since_last=0.0, recent_starts=3, running=2)  # measured 5 GB workers: 4 x 5 > 19


def _round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_gb: float, drain_after_s: float = 0.0) -> int:
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "state", scratch=tmp_path / "scratch")
    monkeypatch.setattr(K, "prepare", lambda cfg, led: (object(), [], None))
    monkeypatch.setattr(K, "plan_one", lambda *a, **k: None)
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
    W.run_round(cfg, lambda: None, gov(free_gb), max_packages=1, poll_s=0.01, scheduled=False, filler=lambda: job, filler_budget=6,
                drain=drain)
    return live["peak"]


def test_fillers_overlap_inside_one_spacing_window_when_ram_is_plentiful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _round(tmp_path, monkeypatch, free_gb=20.0) >= 4         # ramp_s is an hour: without the burst only one would ever start


def test_tight_ram_keeps_the_spacing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _round(tmp_path, monkeypatch, free_gb=1.9, drain_after_s=1.5) == 1   # one start, then the hour-long spacing; the drain ends it
