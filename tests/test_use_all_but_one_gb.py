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


def test_always_on_runs_several_local_model_servers_from_ram() -> None:
    """Owner, 2 Oct 2026: 'there is more RAM work that needs to be done' - one model server per machine queued every thinking
    student behind it while ~17 GiB sat free."""
    never_yield = {"governor_floor_fraction": 0.03, "governor_floor_min_gb": 0.5, "user_aware": False}
    assert D.derive(_dev(ram=33.78, physical=12), overrides=never_yield, lm_cuda=False)["llama_servers"] == 6     # logical 12 // 2 threads each
    assert D.derive(_dev(ram=16.76, physical=8), overrides={}, lm_cuda=False)["llama_servers"] == 1      # yielding: unchanged
    assert D.derive(_dev(), overrides={**never_yield, "llama_servers": 3}, lm_cuda=False)["llama_servers"] == 3


def test_each_local_model_takes_its_own_server_slot_and_the_next_waits(tmp_path: Path) -> None:
    from creator import generator as G
    def lm(n: int, wait: float = 60.0) -> "G.LocalModel":
        return G.LocalModel(model=tmp_path / "m.gguf", exe=tmp_path / "s", pidfile=tmp_path / "llama_server.pid",
                            servers=n, slot_wait_s=wait, threads=2, gpu_layers=0)
    a, b = lm(2), lm(2)
    a._acquire_slot()
    b._acquire_slot()
    try:
        assert a.lock.path.name == "llama_server.lock" and a.pidfile.name == "llama_server.pid"          # slot 0: old names
        assert b.lock.path.name == "llama_server.1.lock" and b.pidfile.name == "llama_server.1.pid"
        c = lm(2, wait=1.0)
        with pytest.raises(TimeoutError):                                               # both slots busy: the third waits
            c._acquire_slot()
    finally:
        a.lock.release()
        b.lock.release()
    d = lm(2, wait=5.0)
    d._acquire_slot()                                                                   # a freed slot is reused
    d.lock.release()


def test_the_swarm_status_line_names_what_stops_more_work(capsys: pytest.CaptureFixture[str]) -> None:
    """A STATUS line names the first limit, so machine under-use is read from the log, never guessed."""
    import json
    def line(free: float, load: int, planned: int, exhausted: bool) -> dict:
        g = W.Governor(free=lambda: free, total=lambda: 33.78, floor_fraction=0.03, floor_min_gb=0.5, free_disk=lambda: 500.0,
                       max_workers=64)
        __import__('creator.swarmops', fromlist=['x']).status_line(g, load, [1] * load, [1] * load, [], [], planned, 64, exhausted, False, 0, True)
        return json.loads(capsys.readouterr().out.split("STATUS ", 1)[1])
    assert line(20.0, 3, 3, True)["limit"].startswith("no more work planned")      # RAM free, nothing left to start
    assert line(0.9, 3, 3, False)["limit"].startswith("RAM:")                       # at the floor
    assert line(20.0, 64, 64, False)["limit"] == "worker cap 64"


def test_server_count_follows_ram_and_half_the_logical_cpus() -> None:
    never_yield = {"governor_floor_fraction": 0.03, "governor_floor_min_gb": 0.5, "user_aware": False}
    assert D.derive(_dev(ram=33.78, logical=14, physical=12), overrides=never_yield, lm_cuda=False)["llama_servers"] == 7    # this PC
    assert D.derive(_dev(ram=8.0, logical=16, physical=8), overrides=never_yield, lm_cuda=False)["llama_servers"] == 2       # RAM-bound
    assert D.derive(_dev(ram=64.0, logical=4, physical=4), overrides=never_yield, lm_cuda=False)["llama_servers"] == 2        # CPU-bound


def test_threads_per_server_are_capped_and_never_below_two(tmp_path: Path) -> None:
    from creator import generator as G
    def th(n: int) -> int:
        return G.LocalModel(model=tmp_path / "m", exe=tmp_path / "s", servers=n, pidfile=tmp_path / "p.pid", gpu_layers=0).threads
    cfg = D.settings()
    assert th(1) == int(cfg["llama_threads"])
    assert 2 <= th(2) <= D.SERVER_MAX_THREADS and th(64) == 2


def test_an_extra_server_slot_starts_only_while_free_ram_minus_a_server_stays_above_the_floor(tmp_path: Path) -> None:
    from creator import generator as G
    from creator import testslots as TS
    floor = TS.reserve_gb(D.get().ram_gb)
    def lm(free: float, wait: float = 1.0) -> "G.LocalModel":
        m = G.LocalModel(model=tmp_path / "m", exe=tmp_path / "s", pidfile=tmp_path / "p.pid", servers=3, slot_wait_s=wait,
                         threads=2, gpu_layers=0)
        m.free_gb = lambda: free
        return m
    tight = floor + D.SERVER_GB - 0.1
    a, b = lm(tight), lm(tight)
    a._acquire_slot()                                       # slot 0 is never RAM-gated
    try:
        assert a.lock.path.name == "llama_server.lock"
        with pytest.raises(TimeoutError):                   # slot 1 would eat into the floor: not started, waits instead
            b._acquire_slot()
        c = lm(floor + D.SERVER_GB + 0.1)
        c._acquire_slot()                                   # same machine with enough free RAM: the extra slot opens
        assert c.lock.path.name == "llama_server.1.lock"
        c.lock.release()
    finally:
        a.lock.release()


def test_startup_time_allowed_scales_with_servers_loading_together(tmp_path: Path) -> None:
    from creator import generator as G
    m = G.LocalModel(model=tmp_path / "m", exe=tmp_path / "s", pidfile=tmp_path / "p.pid", servers=3, startup_s=10.0, threads=2, gpu_layers=0)
    assert m._starting_count() == 1
    (tmp_path / "p.starting").write_text("x")
    (tmp_path / "p.1.starting").write_text("x")
    (tmp_path / "p.2.starting").write_text("x")
    assert m._starting_count() == 3
    import os, time as T
    old = T.time() - 1000
    os.utime(tmp_path / "p.2.starting", (old, old))          # a marker left by a crash goes stale
    assert m._starting_count() == 2
    m.pidfile = tmp_path / "p.1.pid"
    m._mark_starting()
    assert m.starting_marker == tmp_path / "p.1.starting" and m.starting_marker.is_file()
    m._unmark_starting()
    assert not (tmp_path / "p.1.starting").exists()


def _lm(tmp_path: Path, free: object, wait: float = 1.0, servers: int = 8):  # type: ignore[no-untyped-def]
    from creator import generator as G
    m = G.LocalModel(model=tmp_path / "m", exe=tmp_path / "s", pidfile=tmp_path / "p.pid", servers=servers, slot_wait_s=wait,
                     threads=2, gpu_layers=0, startup_s=10.0)
    m.free_gb = lambda: free  # type: ignore[assignment,return-value]
    return m


def test_simultaneous_starters_are_charged_for_servers_still_loading(tmp_path: Path) -> None:
    from creator import testslots as TS
    floor = TS.reserve_gb(D.get().ram_gb)
    free = floor + D.SERVER_GB + 0.1                        # room for exactly one more server above the floor
    m = _lm(tmp_path, free)
    assert m._ram_allows_extra_server()
    (tmp_path / "p.1.starting").write_text("x")             # one other server is loading and has not taken its RAM yet
    assert not m._ram_allows_extra_server()
    m.starting_marker = tmp_path / "p.1.starting"           # ... but our own marker is never charged to us
    assert m._ram_allows_extra_server()


def test_seven_concurrent_starters_get_a_deadline_for_seven_and_keep_their_markers_fresh(tmp_path: Path) -> None:
    import os
    import time as T
    ms = []
    for i in range(7):
        m = _lm(tmp_path, 99.0)
        m.pidfile = tmp_path / (f"p.{i}.pid" if i else "p.pid")
        m._mark_starting()
        ms.append(m)
    assert ms[0]._starting_count() == 7                      # 7 x startup_s of deadline, not capped at 5
    old = T.time() - 20                                      # a marker not refreshed for 20 s is still live; 600 s is a crash
    os.utime(ms[3].starting_marker, (old, old))
    assert ms[0]._starting_count() == 7
    ms[3]._touch_marker()
    assert T.time() - ms[3].starting_marker.stat().st_mtime < 5
    stale = T.time() - 600
    os.utime(ms[6].starting_marker, (stale, stale))
    assert ms[0]._starting_count() == 6


def test_the_starting_marker_is_removed_when_popen_or_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import generator as G
    (tmp_path / "m").write_bytes(b"x")
    (tmp_path / "s").write_bytes(b"x")
    m = _lm(tmp_path, 99.0, servers=1)

    def boom(*a: object, **k: object) -> None:
        raise OSError("cannot start")
    monkeypatch.setattr(G.subprocess, "Popen", boom)
    with pytest.raises(OSError):
        m.__enter__()
    assert not list(tmp_path.glob("*.starting"))

    class P:
        pid = 1
        _handle = 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            pass
    m2 = _lm(tmp_path, 99.0, servers=1)

    def timeout() -> None:
        raise TimeoutError("x")
    monkeypatch.setattr(m2, "_wait_healthy", timeout)
    monkeypatch.setattr(G.subprocess, "Popen", lambda *a, **k: P())
    with pytest.raises(TimeoutError):
        m2.__enter__()
    assert not list(tmp_path.glob("*.starting"))


def test_unknown_free_ram_denies_extra_slots_but_not_slot_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from creator import generator as G
    real = builtins.__import__

    def no_psutil(name: str, *a: object, **k: object):  # type: ignore[no-untyped-def]
        if name == "psutil":
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_psutil)
    assert G._free_ram_gb() is None
    monkeypatch.undo()
    a, b = _lm(tmp_path, None, wait=1.0, servers=3), _lm(tmp_path, None, wait=1.0, servers=3)
    a._acquire_slot()                                        # slot 0 unaffected
    try:
        assert a.lock.path.name == "llama_server.lock"
        assert not b._ram_allows_extra_server()
        with pytest.raises(TimeoutError):
            b._acquire_slot()
    finally:
        a.lock.release()
