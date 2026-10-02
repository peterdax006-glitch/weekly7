"""The watchdog keeps Nupen running whenever it is supposed to (owner, 2 Oct 2026)."""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):                                                       # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)                             # type: ignore[arg-type]
    spec.loader.exec_module(mod)                                            # type: ignore[union-attr]
    return mod


@pytest.fixture()
def wd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):                    # type: ignore[no-untyped-def]
    w = _load("nupen_watchdog")
    for name, rel in (("STATE", ""), ("STOP", "NUPEN_STOP"), ("PIDFILE", "wd.pid"), ("SUPERVISOR_PIDFILE", "svc.pid"),
                      ("HEARTBEAT", "svc.heartbeat"), ("LOG", "wd.log")):
        monkeypatch.setattr(w, name, tmp_path / rel if rel else tmp_path)
    w.started = []
    w.killed = []
    monkeypatch.setattr(w, "start_supervisor", lambda: w.started.append(1))
    monkeypatch.setattr(w, "kill_tree", lambda pid: w.killed.append(pid))
    return w


def test_a_dead_supervisor_is_restarted(wd, tmp_path: Path) -> None:        # type: ignore[no-untyped-def]
    assert wd.check_once(now=time.time()) == "restarted:dead" and wd.started == [1]
    (tmp_path / "svc.pid").write_text("999999", encoding="utf-8")            # a pid that is not running
    assert wd.check_once(now=time.time()).startswith("restarted") and len(wd.started) == 2


def test_a_live_supervisor_with_a_fresh_beat_is_left_alone(wd, tmp_path: Path) -> None:   # type: ignore[no-untyped-def]
    (tmp_path / "svc.pid").write_text(str(os.getpid()), encoding="utf-8")    # this test process is alive
    (tmp_path / "svc.heartbeat").write_text(f"{os.getpid()} {time.time():.0f}", encoding="utf-8")
    assert wd.check_once(now=time.time()) == "ok" and wd.started == [] and wd.killed == []


def test_a_hung_supervisor_is_ended_and_restarted(wd, tmp_path: Path) -> None:     # type: ignore[no-untyped-def]
    (tmp_path / "svc.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "svc.heartbeat").write_text(f"{os.getpid()} {time.time() - 3600:.0f}", encoding="utf-8")
    assert wd.check_once(now=time.time()) == "restarted:hung"
    assert wd.killed == [os.getpid()] and wd.started == [1] and not (tmp_path / "svc.pid").exists()


def test_the_off_switch_is_always_respected(wd, tmp_path: Path) -> None:   # type: ignore[no-untyped-def]
    (tmp_path / "NUPEN_STOP").touch()
    assert wd.check_once(now=time.time()) == "stopped" and wd.started == []


def test_a_fresh_restart_gets_a_grace_period(wd) -> None:                  # type: ignore[no-untyped-def]
    now = time.time()
    assert wd.check_once(now=now, last_start=now - 10) == "grace" and wd.started == []


def test_one_watchdog_at_a_time(wd, tmp_path: Path) -> None:               # type: ignore[no-untyped-def]
    (tmp_path / "wd.pid").write_text(str(os.getpid() + 0), encoding="utf-8")
    assert wd.claim() is True                                                # its own pid: fine
    (tmp_path / "wd.pid").write_text(str(os.getppid()), encoding="utf-8")   # a live other process holds it
    assert wd.claim() is False


def test_the_supervisor_survives_an_error_and_keeps_the_watchdog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside supervision is logged in full and retried - never a silent death (2 Oct: the supervisor died on a
    ModuleNotFoundError right after 'up' and Nupen sat idle 34 minutes)."""
    import sys
    import threading
    svc = _load("nupen_service")
    for name, rel in (("STATE", ""), ("STOP", "NUPEN_STOP"), ("PIDFILE", "svc.pid"), ("LOG", "svc.log"),
                      ("HEARTBEAT", "svc.heartbeat"), ("CRASHLOG", "crash.log"), ("WATCHDOG_PIDFILE", "wd.pid")):
        monkeypatch.setattr(svc, name, tmp_path / rel if rel else tmp_path)
    monkeypatch.setenv("NUPEN_LM", "0")
    calls = {"n": 0, "wd": 0}

    def flaky(python, poll_s):                                              # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        (tmp_path / "NUPEN_STOP").touch()
    monkeypatch.setattr(svc, "_supervise", flaky)
    monkeypatch.setattr(svc, "ensure_watchdog", lambda: calls.__setitem__("wd", calls["wd"] + 1))
    t = threading.Timer(15.0, (tmp_path / "NUPEN_STOP").touch)
    t.start()
    try:
        assert svc.run(sys.executable, poll_s=0.05) == 0
    finally:
        t.cancel()
    text = (tmp_path / "svc.log").read_text(encoding="utf-8")
    assert calls["n"] == 2 and "supervisor error, retrying" in text and "RuntimeError: boom" in text
    assert (tmp_path / "svc.heartbeat").exists()
