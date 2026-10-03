"""Nupen runs whenever the computer is on, yields RAM to the owner while they work, and needs no teacher to keep going (owner,
1 Oct 2026)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import swarm as W

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_lm_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supervisor tests must never start the real language-model trainer, whatever the machine's idle state."""
    monkeypatch.setenv("NUPEN_LM", "0")
    monkeypatch.setenv("NUPEN_PRACTICE", "0")


@pytest.fixture(autouse=True)
def _no_real_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validator 6: the off-switch test ran the real supervisor loop, whose ensure_watchdog started a REAL detached watchdog from
    this checkout - one that outlived pytest and would restart a supervisor and swarm here."""
    monkeypatch.setenv("NUPEN_WATCHDOG", "0")


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


def test_the_owner_can_set_a_fixed_reserve_and_turn_off_yielding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner, 2 Oct 2026: 'building itself no matter what ... utilizing all but 1 GB' - via state/creator/device_overrides.json."""
    svc = _load("nupen_service")
    base = svc.DEV.settings()
    monkeypatch.setattr(svc.DEV, "settings", lambda refresh=False: {**base, "user_aware": False})
    assert "--user-aware" not in svc.swarm_cmd("py")
    monkeypatch.setattr(svc.DEV, "settings", lambda refresh=False: {**base, "user_aware": True})
    assert "--user-aware" in svc.swarm_cmd("py")
    fixed = W.Governor(floor_fraction=0.0, floor_min_gb=1.0, total=lambda: 33.8, idle=lambda: 0.0)
    assert fixed.floor() == pytest.approx(1.0)                              # all but 1 GB, owner at the keyboard or not


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
    for name, rel in (("HEARTBEAT", "svc.heartbeat"), ("CRASHLOG", "crash.log"), ("WATCHDOG_PIDFILE", "wd.pid")):
        monkeypatch.setattr(svc, name, tmp_path / rel)                       # never the real state directory
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


def test_the_off_switch_stops_the_swarm_and_everything_it_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 Oct: NUPEN_STOP terminated only the swarm; six pytest runs it had started kept running (with their model servers)."""
    import subprocess
    svc = _load("nupen_service")
    for name in ("STATE", "STOP", "PIDFILE", "LOG", "HEARTBEAT", "CRASHLOG", "WATCHDOG_PIDFILE"):       # nothing may touch the real state
        monkeypatch.setattr(svc, name, tmp_path / {"STATE": "", "STOP": "NUPEN_STOP", "PIDFILE": "svc.pid", "LOG": "svc.log",
                                                   "HEARTBEAT": "svc.heartbeat", "CRASHLOG": "crash.log", "WATCHDOG_PIDFILE": "wd.pid"}[name])
    marker = tmp_path / "grandchild.pid"
    parent_code = ("import subprocess, sys, time; "
                   "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
                   f"open(r'{marker}', 'w').write(str(c.pid)); time.sleep(120)")
    monkeypatch.setattr(svc, "swarm_cmd", lambda py: [py, "-c", parent_code])
    def stop_when_started() -> None:
        while not marker.exists():
            time.sleep(0.1)
        time.sleep(0.5)
        (tmp_path / "NUPEN_STOP").touch()
    threading.Thread(target=stop_when_started, daemon=True).start()
    assert svc.run(sys.executable, poll_s=0.2) == 0
    gpid = int(marker.read_text(encoding="utf-8"))
    def alive() -> bool:
        return str(gpid) in subprocess.run(["tasklist", "/FI", f"PID eq {gpid}", "/NH"], capture_output=True, text=True).stdout
    deadline = time.monotonic() + 30.0
    while alive() and time.monotonic() < deadline:                              # termination is asynchronous: wait, do not guess
        time.sleep(0.1)
    assert not alive()                                                  # the grandchild is gone too


# ---- LM trainer at idle priority (owner priority 3) ------------------------------------------------------------------

class _FakeProc:
    def __init__(self) -> None:
        self.pid, self.returncode = 4242, None

    def poll(self):                                                         # type: ignore[no-untyped-def]
        return self.returncode


def _trainer(tmp_path: Path, idle: list, free: list, python: Path | None = None):   # type: ignore[no-untyped-def]
    svc = _load("nupen_service")
    py = python or (tmp_path / "python.exe")
    if python is None:
        py.write_text("", encoding="utf-8")
    clock = [1000.0]
    spawned: list = []
    killed: list = []
    logs: list = []

    def spawn(cmd):                                                         # type: ignore[no-untyped-def]
        spawned.append(cmd)
        return _FakeProc()
    tr = svc.LMTrainer(python=py, idle=lambda: idle[0], free_gb=lambda: free[0], spawn=spawn, stop_file=tmp_path / "STOP",
                       stop=lambda p: (killed.append(p), setattr(p, "returncode", 1)), clock=lambda: clock[0], log=logs.append,
                       grace_s=90.0, restart_gap_s=60.0)
    return tr, spawned, killed, logs, clock


def test_lm_trainer_starts_only_when_idle_and_ram_allows(tmp_path: Path) -> None:
    idle, free = [30.0], [8.0]
    tr, spawned, _k, _l, _c = _trainer(tmp_path, idle, free)
    tr.tick()
    assert not spawned                                                      # owner at the keyboard
    idle[0], free[0] = 3600.0, 1.0
    tr.tick()
    assert not spawned                                                      # idle but too little RAM
    free[0] = 8.0
    tr.tick()
    assert len(spawned) == 1
    assert spawned[0][-9:] == ["train", "--minutes", "20", "--cycle-minutes", "60", "--mix", "dialogue", "--dialogue-share", "0.2"]
    assert spawned[0][0] == str(tr.python)


def test_lm_trainer_never_runs_two_at_once(tmp_path: Path) -> None:
    tr, spawned, _k, _l, clock = _trainer(tmp_path, [3600.0], [8.0])
    for _ in range(5):
        clock[0] += 100
        tr.tick()
    assert len(spawned) == 1
    tr.proc.returncode = 0                                                  # finished: the next may start (after the restart gap)
    clock[0] += 100
    tr.tick()
    assert len(spawned) == 2


def test_lm_trainer_stops_when_the_owner_returns(tmp_path: Path) -> None:
    idle = [3600.0]
    tr, spawned, killed, _l, clock = _trainer(tmp_path, idle, [8.0])
    tr.tick()
    idle[0] = 2.0
    tr.tick()
    assert (tmp_path / "STOP").exists() and not killed                      # asked politely first, through the STOP file
    clock[0] += 120                                                         # it ignored the file: the whole tree is ended
    tr.tick()
    assert len(killed) == 1 and tr.proc is None
    tr.tick()
    assert len(spawned) == 1                                                # owner still active: not restarted


def test_lm_trainer_ends_cleanly_when_it_obeys_the_stop_file(tmp_path: Path) -> None:
    idle = [3600.0]
    tr, _s, killed, _l, _c = _trainer(tmp_path, idle, [8.0])
    tr.tick()
    idle[0] = 1.0
    tr.tick()
    tr.proc.returncode = 0                                                  # exited by itself after the STOP file
    tr.tick()
    assert tr.proc is None and not killed


def test_nupen_stop_stops_the_lm_trainer_and_blocks_a_start(tmp_path: Path) -> None:
    tr, spawned, killed, _l, clock = _trainer(tmp_path, [3600.0], [8.0])
    tr.tick(halt=True)
    assert not spawned                                                      # NUPEN_STOP: never starts
    tr.tick()
    tr.tick(halt=True)
    assert (tmp_path / "STOP").exists()
    clock[0] += 120
    tr.tick(halt=True)
    assert len(killed) == 1 and len(spawned) == 1


def test_lm_trainer_without_its_environment_is_skipped_and_logged_once(tmp_path: Path) -> None:
    tr, spawned, _k, logs, clock = _trainer(tmp_path, [3600.0], [8.0], python=tmp_path / "missing" / "python.exe")
    for _ in range(3):
        clock[0] += 100
        tr.tick()
    assert not spawned and len(logs) == 1


def test_lm_trainer_uses_idle_priority_and_no_window() -> None:
    svc = _load("nupen_service")
    assert svc.IDLE_PRIORITY == 0x40 and svc.NO_WINDOW == 0x08000000
    assert str(svc.LM_PYTHON).replace("\\", "/").endswith("creator_runtime/lmenv/Scripts/python.exe") or sys.platform != "win32"


def test_the_supervisor_imports_creator_when_launched_from_anywhere(tmp_path: Path) -> None:
    """2 Oct: launched as a script (pythonw scripts/nupen_service.py) its import path starts at scripts/, so the LM trainer's
    `from creator import swarm` raised ModuleNotFoundError and the supervisor died silently right after 'supervisor up'."""
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts" / "nupen_service.py"
    code = ("import importlib.util, os, sys; sys.path[:] = [p for p in sys.path if p not in ('', os.getcwd())]; "
            f"spec = importlib.util.spec_from_file_location('svc', r'{script}'); m = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(m); m.LMTrainer(); print('ok')")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-800:]


def test_an_error_in_supervision_does_not_orphan_the_running_swarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validator 6: run() retries after a supervision error by starting a NEW swarm; the old one, still running, was left behind
    (two swarms on one ledger)."""
    import subprocess
    svc = _load("nupen_service")
    for name, rel in (("STATE", ""), ("STOP", "NUPEN_STOP"), ("PIDFILE", "svc.pid"), ("LOG", "svc.log"), ("HEARTBEAT", "svc.heartbeat")):
        monkeypatch.setattr(svc, name, tmp_path / rel if rel else tmp_path)
    marker = tmp_path / "swarm.pid"
    code = f"import os, time; open(r'{marker}', 'w').write(str(os.getpid())); time.sleep(120)"
    monkeypatch.setattr(svc, "swarm_cmd", lambda py: [py, "-c", code])
    monkeypatch.setattr(svc, "ensure_watchdog", lambda: None)
    calls = {"n": 0}

    def beat() -> None:
        calls["n"] += 1
        if calls["n"] >= 2:                                                  # the first beat is before the swarm starts
            while not marker.exists():
                time.sleep(0.05)
            raise RuntimeError("supervision broke")
    monkeypatch.setattr(svc, "beat", beat)
    with pytest.raises(RuntimeError):
        svc._supervise(sys.executable, 0.05)
    pid = int(marker.read_text(encoding="utf-8"))
    time.sleep(1.0)
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)       # tidy up if the assertion is about to fail
    assert str(pid) not in out


def test_a_supervisor_test_run_never_starts_a_real_watchdog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    for name, rel in (("STATE", ""), ("STOP", "NUPEN_STOP"), ("PIDFILE", "svc.pid"), ("LOG", "svc.log"), ("HEARTBEAT", "svc.heartbeat"),
                      ("WATCHDOG_PIDFILE", "wd.pid"), ("CRASHLOG", "crash.log")):
        monkeypatch.setattr(svc, name, tmp_path / rel if rel else tmp_path)
    watchdogs: list[object] = []
    real_popen = svc.subprocess.Popen

    def spy(cmd, *a, **k):                                                   # type: ignore[no-untyped-def]
        if any(str(c).endswith("nupen_watchdog.py") for c in cmd):
            watchdogs.append(cmd)
            return SimpleNamespace(pid=0)
        return real_popen(cmd, *a, **k)
    monkeypatch.setattr(svc.subprocess, "Popen", spy)
    monkeypatch.setattr(svc, "swarm_cmd", lambda py: [py, "-c", "import time; time.sleep(1)"])
    threading.Timer(2.0, (tmp_path / "NUPEN_STOP").touch).start()
    assert svc.run(sys.executable, poll_s=0.2) == 0
    assert watchdogs == []


def _sleeper() -> "subprocess.Popen[bytes]":
    import subprocess
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def test_a_swarm_orphaned_by_a_hard_killed_supervisor_is_ended_by_the_next_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    monkeypatch.setattr(svc, "SWARM_PIDFILE", tmp_path / "nupen_swarm.pid")
    monkeypatch.setattr(svc, "LOG", tmp_path / "svc.log")
    child = _sleeper()                                           # stands in for the orphaned swarm
    bystander = _sleeper()                                       # same image name, not recorded: must survive
    try:
        dead = _sleeper()
        dead.kill()
        dead.wait()
        (tmp_path / "nupen_swarm.pid").write_text(f"{child.pid} {time.time():.0f} {dead.pid}", encoding="utf-8")
        assert svc.reap_orphan_swarm() == child.pid
        child.wait(timeout=30)
        assert child.poll() is not None and bystander.poll() is None
        assert not (tmp_path / "nupen_swarm.pid").exists()
    finally:
        for p in (child, bystander):
            if p.poll() is None:
                p.kill()


def test_a_swarm_whose_supervisor_is_alive_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    monkeypatch.setattr(svc, "SWARM_PIDFILE", tmp_path / "nupen_swarm.pid")
    monkeypatch.setattr(svc, "LOG", tmp_path / "svc.log")
    child, sup = _sleeper(), _sleeper()
    try:
        (tmp_path / "nupen_swarm.pid").write_text(f"{child.pid} {time.time():.0f} {sup.pid}", encoding="utf-8")
        assert svc.reap_orphan_swarm() is None and child.poll() is None
    finally:
        for p in (child, sup):
            p.kill()


def test_supervisor_start_reaps_before_it_starts_a_swarm_and_records_the_new_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _load("nupen_service")
    for name in ("PIDFILE", "SWARM_PIDFILE", "LOG", "STOP", "HEARTBEAT", "CRASHLOG"):
        monkeypatch.setattr(svc, name, tmp_path / name.lower())
    monkeypatch.setattr(svc, "STATE", tmp_path)
    order: list[str] = []
    monkeypatch.setattr(svc, "reap_orphan_swarm", lambda: order.append("reap"))
    (tmp_path / "stop").write_text("", encoding="utf-8")         # the off switch: run() claims, reaps, then exits without a swarm
    assert svc.run(sys.executable, poll_s=0.01) == 0
    assert order == ["reap"]
    started: list[int] = []

    class FakeProc:
        pid = os.getpid() + 7
        returncode = 0
        def poll(self):                                          # type: ignore[no-untyped-def]
            return 0
    monkeypatch.setattr(svc.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(svc, "write_swarm_pid", lambda pid: started.append(pid))
    (tmp_path / "stop").unlink()
    monkeypatch.setattr(svc, "ensure_watchdog", lambda: None)
    monkeypatch.setattr(svc.time, "sleep", lambda s: (tmp_path / "stop").write_text("", encoding="utf-8"))
    svc.run(sys.executable, poll_s=0.01)
    assert started == [FakeProc.pid]


# ---- fair turns: LM training and practice alternate (2 Oct: the continuous trainer starved practice forever) ----------

def _pair(tmp_path: Path, idle: list, free: list):                          # type: ignore[no-untyped-def]
    svc = _load("nupen_service")
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    clock = [1000.0]
    spawned: list = []
    logs: list = []

    def spawn(cmd):                                                         # type: ignore[no-untyped-def]
        spawned.append(cmd)
        return _FakeProc()
    kill = lambda p: setattr(p, "returncode", 1)                            # noqa: E731
    lm = svc.LMTrainer(python=py, idle=lambda: idle[0], free_gb=lambda: free[0], spawn=spawn, stop_file=tmp_path / "STOP",
                       stop=kill, clock=lambda: clock[0], log=logs.append, grace_s=90.0, restart_gap_s=60.0, yield_max_s=300.0)
    pr = svc.PracticeRunner(python="py", idle=lambda: idle[0], free_gb=lambda: free[0], spawn=spawn, stop=kill,
                            clock=lambda: clock[0], log=logs.append, rows_path=tmp_path / "rows.jsonl", gap_s=300.0, empty_gap_s=1800.0)
    pr.lm_running, lm.blocked, lm.yield_to = lm.running, pr.running, pr.wants_turn
    return lm, pr, spawned, clock


def _step(lm, pr, clock, halt=False):                                       # type: ignore[no-untyped-def]
    clock[0] += 10
    lm.tick(halt)                                                           # the supervisor's order: trainer first, then practice
    pr.tick(halt)


def test_practice_gets_a_turn_between_lm_sessions(tmp_path: Path) -> None:
    lm, pr, spawned, clock = _pair(tmp_path, [float("inf")], [8.0])
    _step(lm, pr, clock)
    assert "train" in spawned[-1] and not pr.running()                      # trainer first
    lm.proc.returncode = 0                                                  # a session finished cleanly
    _step(lm, pr, clock)
    assert pr.running() and not lm.running()                                # practice takes the turn, trainer yields
    for _ in range(5):
        _step(lm, pr, clock)
    assert not lm.running()                                                 # still practising: one job at a time
    pr.proc.returncode = 0                                                  # practice done, no rows: long gap
    _step(lm, pr, clock)
    _step(lm, pr, clock)
    assert lm.running() and not pr.running()                                # the trainer resumes


def test_trainer_never_waits_for_a_practice_run_that_cannot_start(tmp_path: Path) -> None:
    free = [8.0]
    lm, pr, spawned, clock = _pair(tmp_path, [float("inf")], free)
    _step(lm, pr, clock)
    pr.next_start = clock[0] + 99999                                        # practice is in its gap: not ready
    lm.proc.returncode = 0
    clock[0] += 100
    _step(lm, pr, clock)
    assert lm.running()                                                     # no wasted time waiting for it
    lm.proc.returncode = 0
    pr.next_start = -1.0
    free[0] = 1.0                                                           # practice ready except RAM: not ready either
    assert not pr.wants_turn()
    clock[0] += 100
    lm.tick()
    assert not lm.running()                                                 # the trainer itself needs RAM too
    free[0] = 8.0
    pr.next_start = clock[0] + 99999
    clock[0] += 100
    lm.tick()
    assert lm.running()


def test_yield_is_bounded_and_nupen_stop_and_ram_rule_still_hold(tmp_path: Path) -> None:
    lm, pr, spawned, clock = _pair(tmp_path, [float("inf")], [8.0])
    lm.yield_to = lambda: True                                              # a practice that never starts must not starve the trainer
    pr.tick(True)                                                           # (halt) so practice does not start
    _step(lm, pr, clock)
    lm.proc.returncode = 0
    lm.tick()
    assert not lm.running()
    clock[0] += 400
    lm.tick()
    assert lm.running()                                                     # yield timed out
    lm.proc.returncode = 0
    lm.tick(True)                                                           # NUPEN_STOP: no yield bookkeeping, no start
    clock[0] += 400
    lm.tick(True)
    assert not lm.running()


def test_crashing_trainer_still_gives_practice_its_turn(tmp_path: Path) -> None:
    lm, pr, spawned, clock = _pair(tmp_path, [float("inf")], [8.0])
    _step(lm, pr, clock)
    lm.proc.returncode = 1                                                  # a crash, not a finished session
    clock[0] += 100
    _step(lm, pr, clock)
    assert pr.running() and not lm.running()
