"""Nupen runs whenever the computer is on, yields RAM to the owner while they work, and needs no teacher to keep going (owner,
1 Oct 2026)."""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import swarm as W

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):                                                       # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)                             # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                            # type: ignore[union-attr]
    return mod


def test_the_reserve_grows_while_the_owner_is_at_the_keyboard() -> None:
    def gov(idle: float) -> W.Governor:
        return W.Governor(floor_fraction=0.07, total=lambda: 16.0, user_active_floor_fraction=0.25, idle=lambda: idle)
    assert gov(10.0).floor() == pytest.approx(4.0)                          # active: 25% of 16 GB kept free
    assert gov(3600.0).floor() == pytest.approx(1.12)                       # idle: back to 7%
    plain = W.Governor(floor_fraction=0.07, total=lambda: 16.0, idle=lambda: 0.0)
    assert plain.floor() == pytest.approx(1.12)                             # off unless asked for


def test_idle_seconds_is_a_number() -> None:
    assert W.user_idle_seconds() >= 0.0


def test_an_absent_teacher_defers_instead_of_waiting(tmp_path: Path) -> None:
    sw = _load("creator_swarm")
    beat = tmp_path / "beat"
    s = sw.SerialSession(0.001, presence=beat, fresh_s=60)
    plan = SimpleNamespace(package_id="CPX")
    r = s(plan, None, tmp_path)                                             # no heartbeat at all
    assert not r.claimed_done and r.deferred and "absent" in r.notes
    beat.write_text("", encoding="utf-8")
    os.utime(beat, (time.time() - 3600, time.time() - 3600))                # stale heartbeat
    assert s(plan, None, tmp_path).deferred
    assert s.teacher_present() is False
    beat.touch()
    assert s.teacher_present() is True
    assert sw.SerialSession(0.001).teacher_present() is True                # without --teacher-presence: always hand off


def test_the_supervisor_stops_on_the_stop_file_and_never_runs_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    monkeypatch.setattr(svc, "STATE", tmp_path)
    monkeypatch.setattr(svc, "STOP", tmp_path / "NUPEN_STOP")
    monkeypatch.setattr(svc, "PIDFILE", tmp_path / "nupen_service.pid")
    monkeypatch.setattr(svc, "LOG", tmp_path / "nupen_service.log")
    monkeypatch.setattr(svc, "swarm_cmd", lambda py: [py, "-c", "import time; time.sleep(60)"])
    threading.Timer(2.0, (tmp_path / "NUPEN_STOP").touch).start()
    t0 = time.monotonic()
    assert svc.run(sys.executable, poll_s=0.2) == 0
    assert time.monotonic() - t0 < 30                                       # the stop file ended it, not the 60 s sleep
    log = (tmp_path / "nupen_service.log").read_text(encoding="utf-8")
    assert "NUPEN_STOP found" in log and "swarm exited" in log and not (tmp_path / "nupen_service.pid").exists()
    (tmp_path / "nupen_service.pid").write_text(str(os.getpid()), encoding="utf-8")     # a live supervisor holds it
    monkeypatch.setattr(svc.os, "getpid", lambda: 999999)
    (tmp_path / "NUPEN_STOP").unlink()
    assert svc.run(sys.executable, poll_s=0.2) == 0
    assert "another supervisor is alive" in (tmp_path / "nupen_service.log").read_text(encoding="utf-8")


def test_autostart_command_points_at_the_service() -> None:
    auto = _load("nupen_autostart")
    cmd = auto.command()
    assert "pythonw.exe" in cmd and "nupen_service.py" in cmd


def test_a_pidfile_from_before_the_boot_does_not_block_the_supervisor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a reboot the old pid may belong to any unrelated live process; the supervisor must still start (it is meant to start
    with the computer)."""
    svc = _load("nupen_service")
    pid = tmp_path / "nupen_service.pid"
    monkeypatch.setattr(svc, "PIDFILE", pid)
    monkeypatch.setattr(svc, "alive", lambda p: True)                       # the recycled pid is alive
    pid.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(svc, "boot_time", lambda: time.time() + 3600.0)     # booted after the file was written
    assert svc.claim_pidfile() is True and pid.read_text(encoding="utf-8") == str(os.getpid())
    pid.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(svc, "boot_time", lambda: time.time() - 3600.0)     # written after this boot: a real holder
    assert svc.claim_pidfile() is False
    assert svc.boot_time.__name__ == "<lambda>" and _load("nupen_service").boot_time() <= time.time()


def test_the_pidfile_is_claimed_exclusively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    monkeypatch.setattr(svc, "PIDFILE", tmp_path / "sub" / "nupen_service.pid")
    assert svc.claim_pidfile() is True and svc.PIDFILE.read_text(encoding="utf-8") == str(os.getpid())
    real_unlink = Path.unlink
    # a rival creates the file between our unlink and our exclusive create: we must lose, not overwrite it
    def rival(self, *a, **k):                                               # type: ignore[no-untyped-def]
        real_unlink(self, *a, **k)
        self.write_text("31337", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", rival)
    assert svc.claim_pidfile() is False and svc.PIDFILE.read_text(encoding="utf-8") == "31337"
