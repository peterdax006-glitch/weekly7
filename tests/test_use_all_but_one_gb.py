"""Owner, 2 Oct 2026 (new PC): "I have a ton of free storage and memory make sure Nupen is set up to use all of it except a
single GB". Every limit that kept RAM or disk unused is tied to the one 1 GB floor."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from creator import device as D
from creator import swarm as W
from creator import testslots as TS

ROOT = Path(__file__).resolve().parents[1]


def _dev(ram: float = 33.78, logical: int = 12, physical: int = 12) -> D.Device:
    base = D.detect() if hasattr(D, "detect") else D.get()
    return D.Device(**{**base.__dict__, "ram_gb": ram, "cores_logical": logical, "cores_physical": physical,
                       "gpu_name": "", "vram_gb": 0.0})


def test_no_yielding_lets_every_cpu_thread_have_a_test_slot() -> None:
    never_yield = {"governor_floor_fraction": 0.0, "governor_floor_min_gb": 1.0, "user_aware": False}
    assert D.derive(_dev(), overrides=never_yield, lm_cuda=False)["test_slots"] == 12
    assert D.derive(_dev(), overrides={}, lm_cuda=False)["test_slots"] == 8          # the yielding default is unchanged
    pinned = {**never_yield, "test_slots": 5}
    assert D.derive(_dev(), overrides=pinned, lm_cuda=False)["test_slots"] == 5      # an owner pin still wins


def test_the_test_budget_keeps_the_same_reserve_as_the_governor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before: testslots kept max(0.8 GB, 7% of RAM) = 2.4 GB free on 33.8 GB, ignoring the owner's 1 GB floor."""
    monkeypatch.setattr(TS, "_device", lambda: type("Dv", (), {"settings": staticmethod(
        lambda: {"governor_floor_fraction": 0.0, "governor_floor_min_gb": 1.0})}))
    assert TS.reserve_gb(33.78) == 1.0
    assert TS.affordable(free_gb=1.2, total_gb=33.78, mb=150)                         # 1.2 - 1.0 GB still holds 150 MB
    assert not TS.affordable(free_gb=1.1, total_gb=33.78, mb=150)                     # the 1 GB floor is never crossed


def test_the_governor_never_starts_work_below_the_disk_floor() -> None:
    def gov(disk: float) -> W.Governor:
        return W.Governor(free=lambda: 30.0, total=lambda: 33.78, floor_fraction=0.0, floor_min_gb=1.0,
                          free_disk=lambda: disk, disk_floor_gb=1.0)
    assert gov(500.0).can_start(0)
    assert not gov(0.9).can_start(0)                                                  # less than 1 GB of disk left: no new work
    broken = W.Governor(free=lambda: 30.0, total=lambda: 33.78, free_disk=lambda: (_ for _ in ()).throw(OSError("x")))
    assert broken.can_start(0)                                                        # a failed measurement never blocks work


def test_training_and_practice_do_not_wait_for_the_owner_to_be_idle_when_yielding_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("nupen_service_t", ROOT / "scripts" / "nupen_service.py")
    svc = importlib.util.module_from_spec(spec)                                       # type: ignore[arg-type]
    spec.loader.exec_module(svc)                                                      # type: ignore[union-attr]
    monkeypatch.setattr(svc.DEV, "settings", lambda refresh=False: {"user_aware": False})
    assert svc.owner_idle_seconds(W)() == float("inf")
    monkeypatch.setattr(svc.DEV, "settings", lambda refresh=False: {"user_aware": True})
    assert svc.owner_idle_seconds(W) is W.user_idle_seconds


def test_a_29_gib_target_keeps_the_rest_of_this_machine_free() -> None:
    """Owner, 2 Oct 2026: "we want to use 29 GB 24/7 you as claude get to use what you need then Nupen uses the rest"."""
    s = D.derive(_dev(ram=33.78), overrides={"ram_use_target_gib": 29, "user_aware": False}, lm_cuda=False)
    assert s["governor_floor_fraction"] == 0.0
    assert s["governor_floor_min_gb"] == round(33.78 - 29 * 1.073741824, 2)          # 2.64 GB (decimal) = ~2.5 GiB free
    assert D.derive(_dev(ram=33.78), overrides={"ram_use_target_gib": 40}, lm_cuda=False)["governor_floor_min_gb"] == 0.5


def test_a_ledger_lock_left_by_a_killed_writer_is_taken_over_but_a_live_one_is_not(tmp_path: Path) -> None:
    """2 Oct: a pause killed the swarm mid-append; its lock file stayed and every later append failed after 30 s."""
    import subprocess
    import sys
    from creator import ledger as LG
    lock = tmp_path / "ledger.jsonl.lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(str(dead.pid), encoding="utf-8")
    assert LG._take_over_dead_lock(lock) and not lock.exists()                         # dead holder: taken over
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(str(live.pid), encoding="utf-8")
        assert not LG._take_over_dead_lock(lock) and lock.exists()                     # live holder: never stolen
    finally:
        live.kill()
    lock.write_text("not-a-pid", encoding="utf-8")
    assert not LG._take_over_dead_lock(lock)                                           # unreadable: treated as alive
