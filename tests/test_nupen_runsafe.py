"""Deploys must not corrupt the run: killed lock holders are reclaimed, safely, and the service drains instead of killing."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from creator import kernel as K
from creator import ledger as LG
from creator import testslots as TS

ROOT = Path(__file__).resolve().parents[1]


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_an_empty_lock_left_by_a_kill_between_create_and_write_is_taken_over(tmp_path: Path) -> None:
    lock = tmp_path / "ledger.jsonl.lock"
    lock.write_text("", encoding="utf-8")
    assert not LG._take_over_dead_lock(lock)                                  # just created: its writer may be about to write
    old = time.time() - 60
    os.utime(lock, (old, old))
    assert LG._take_over_dead_lock(lock) and not lock.exists()


def test_a_killed_ledger_writer_does_not_block_the_next_append(tmp_path: Path) -> None:
    led = LG.Ledger(tmp_path / "ledger.jsonl", evidence_root=tmp_path)
    led.lock_path.write_text(str(_dead_pid()), encoding="utf-8")
    with led._locked():
        assert led.lock_path.read_text(encoding="utf-8") == str(os.getpid())
    assert not led.lock_path.exists()


def test_the_takeover_never_deletes_a_lock_another_starter_just_took(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reviewer's race: starter B saw the dead holder, starter A replaced the lock, B must NOT unlink A's fresh lock."""
    lock = tmp_path / "kernel.lock"
    lock.write_text(str(_dead_pid()), encoding="utf-8")
    real = LG._holder_dead
    calls = {"n": 0}
    live = str(os.getppid() or 1)

    def racing(path: Path) -> object:
        calls["n"] += 1
        res = real(path)
        if calls["n"] == 1:                                                  # after B's first look, A takes over and re-creates it
            lock.write_text(live, encoding="utf-8")                          # a live process (the parent)
        return res
    monkeypatch.setattr(LG, "_holder_dead", racing)
    assert not LG._take_over_dead_lock(lock)
    assert lock.exists() and lock.read_text(encoding="utf-8") == live
    assert not (tmp_path / "kernel.lock.takeover").exists()                  # the guard is always released


def test_a_stale_takeover_guard_is_cleared_and_a_fresh_one_is_respected(tmp_path: Path) -> None:
    lock = tmp_path / "kernel.lock"
    lock.write_text(str(_dead_pid()), encoding="utf-8")
    guard = tmp_path / "kernel.lock.takeover"
    guard.write_text("", encoding="utf-8")
    assert not LG._take_over_dead_lock(lock) and lock.exists()               # another starter is mid-takeover
    old = time.time() - 60
    os.utime(guard, (old, old))
    assert not LG._take_over_dead_lock(lock)                                 # the leftover guard is removed ...
    assert not guard.exists()
    assert LG._take_over_dead_lock(lock) and not lock.exists()               # ... and the next try succeeds


def test_a_kernel_lock_of_a_dead_holder_is_reclaimed_and_a_live_one_refused(tmp_path: Path) -> None:
    (tmp_path / "kernel.lock").write_text(str(_dead_pid()), encoding="utf-8")
    with K._KernelLock(tmp_path):
        assert (tmp_path / "kernel.lock").read_text(encoding="utf-8") == str(os.getpid())
    assert not (tmp_path / "kernel.lock").exists()
    (tmp_path / "kernel.lock").write_text(str(os.getppid()), encoding="utf-8")
    with pytest.raises(K.KernelError):
        K._KernelLock(tmp_path).__enter__()


def test_two_kernel_starters_racing_on_a_dead_lock_never_both_hold_it(tmp_path: Path) -> None:
    (tmp_path / "kernel.lock").write_text(str(_dead_pid()), encoding="utf-8")
    won: list[int] = []
    barrier = threading.Barrier(6)

    def starter() -> None:
        barrier.wait()
        lk = K._KernelLock(tmp_path)
        try:
            lk.__enter__()
        except K.KernelError:
            return
        won.append(1)
        time.sleep(0.5)                                                      # hold it while the others try
    ts = [threading.Thread(target=starter) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(won) == 1


def test_a_test_slot_held_by_a_killed_process_is_free_again(tmp_path: Path) -> None:
    slot = tmp_path / "slot0.lock"
    code = ("import time\nfrom pathlib import Path\nfrom creator.testslots import _SlotFile\n"
            f"f = _SlotFile(Path(r'{slot}'))\nassert f.try_lock()\nprint('held', flush=True)\ntime.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "held"
        mine = TS._SlotFile(slot)
        assert not mine.try_lock()                                           # held while alive
        p.kill()
        p.wait()
        for _ in range(50):
            if mine.try_lock():
                break
            time.sleep(0.1)
        assert mine.fh is not None                                           # the OS released it: no stale slot file blocks
        mine.release()
    finally:
        p.kill()


def _svc():                                                                  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("nupen_service", ROOT / "scripts" / "nupen_service.py")
    mod = importlib.util.module_from_spec(spec)                              # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                             # type: ignore[union-attr]
    return mod


def test_the_service_drains_without_killing_and_does_not_restart_until_the_drain_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUPEN_LM", "0")
    monkeypatch.setenv("NUPEN_PRACTICE", "0")
    monkeypatch.setenv("NUPEN_WATCHDOG", "0")
    svc = _svc()
    for name, rel in (("STATE", ""), ("STOP", "NUPEN_STOP"), ("DRAIN", "NUPEN_DRAIN"), ("PIDFILE", "svc.pid"), ("LOG", "svc.log"),
                      ("HEARTBEAT", "svc.heartbeat"), ("SWARM_PIDFILE", "swarm.pid"), ("CRASHLOG", "crash.log"),
                      ("WATCHDOG_PIDFILE", "wd.pid")):
        monkeypatch.setattr(svc, name, tmp_path / rel if rel else tmp_path)
    monkeypatch.setattr(svc, "ensure_watchdog", lambda: None)
    done = tmp_path / "done.txt"
    # the 'swarm' sees the drain file itself, finishes its 'cycle' (1.5 s) and exits 0 - the supervisor must not have killed it
    started = tmp_path / "started.txt"
    code = (f"import time, pathlib\npathlib.Path(r'{started}').write_text('up')\nd = pathlib.Path(r'{tmp_path / 'NUPEN_DRAIN'}')\n"
            "while not d.exists(): time.sleep(0.05)\n"
            f"time.sleep(1.5)\npathlib.Path(r'{done}').write_text('finished')\n")
    monkeypatch.setattr(svc, "swarm_cmd", lambda py: [py, "-c", code])

    def pause_once_up() -> None:
        while not started.exists():
            time.sleep(0.05)
        (tmp_path / "NUPEN_DRAIN").touch()                                   # the deploy pause arrives while the swarm runs
        while not done.exists():
            time.sleep(0.05)
        time.sleep(2.0)                                                      # it exited by itself; the supervisor must idle, not restart
        (tmp_path / "NUPEN_STOP").touch()                                    # ends the test
    threading.Thread(target=pause_once_up, daemon=True).start()
    assert svc.run(sys.executable, poll_s=0.2) == 0
    assert done.read_text(encoding="utf-8") == "finished"                    # in-flight work completed, not killed
    log = (tmp_path / "svc.log").read_text(encoding="utf-8")
    assert log.count("swarm started") == 1                                   # and no second swarm while the drain file existed
