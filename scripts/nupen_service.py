"""Nupen's supervisor: keeps the swarm running whenever the computer is on (owner, 1 Oct 2026: "the second I turn my computer on
it all starts running and then when i start doing things it adjusts how much RAM it can use based on how much RAM im using but i
shouldnt need to go to you to get it to start working as long as its capable of doing something more good than bad on its own").

- Starts the swarm (scripts/creator_swarm.py --rounds 0) at BELOW_NORMAL CPU priority, so the owner's programs always win the CPU.
- --user-aware: while the owner is at the keyboard (input in the last 5 min) 25% of RAM stays free for them; idle, Nupen may use
  all but 7%. The governor reads live free RAM, so anything the owner opens pulls Nupen's workers back.
- --teacher-presence: packages go to the teacher (Claude) only while its heartbeat is fresh; otherwise they are deferred and
  Nupen's students and own workers carry on alone.
- Safety is the kernel's: only a measured IMPROVEMENT with a clean sandbox evaluation is merged, failed post-merge checks roll
  back, the audit gates every cycle, nothing is ever pushed.
- Restarts the swarm if it exits (backoff up to 30 min). One supervisor at a time (pidfile, a live holder is never displaced).
- LM trainer (owner priority 3): while the owner has been idle for 10+ minutes and 3+ GB of RAM is free, Nupen's own language model
  trains (scripts/nupen_lm.py train --minutes 20) at IDLE priority, one run at a time. It is stopped through its STOP file (then as
  a process tree) when the owner returns or NUPEN_STOP appears. No lmenv: skipped silently. Log: state/creator/lm_service.log.
- Practice (same idle/RAM rule, never alongside the LM trainer): scripts/practice.py on the current HEAD at IDLE priority, then a
  chooser retrain on lessons + practice rows (only when new rows arrived). Log: state/creator/practice_service.log. NUPEN_PRACTICE=0 = off.
- OFF SWITCH: create state/creator/NUPEN_STOP (the swarm is stopped and the supervisor exits); or
  `python scripts/nupen_autostart.py uninstall` to stop it starting with the computer.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # launched as a script its path starts at scripts/: 2 Oct, the LM trainer's
    sys.path.insert(0, str(ROOT))                 # `from creator import swarm` killed the supervisor right after 'up'

from creator import device as DEV  # noqa: E402
STATE = ROOT / "state" / "creator"
STOP = STATE / "NUPEN_STOP"
PIDFILE = STATE / "nupen_service.pid"
LOG = STATE / "nupen_service.log"
HEARTBEAT = STATE / "nupen_service.heartbeat"          # touched every poll; the watchdog restarts a supervisor whose beat goes stale
CRASHLOG = STATE / "nupen_service.crash.log"           # faulthandler output: hard crashes leave a trace (pythonw has no console)
WATCHDOG_PIDFILE = STATE / "nupen_watchdog.pid"
SWARM_PIDFILE = STATE / "nupen_swarm.pid"              # "swarm_pid start_epoch supervisor_pid": lets a later supervisor end an orphaned swarm
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000
IDLE_PRIORITY = 0x00000040
LM_LOG = STATE / "lm_service.log"
LM_PYTHON = DEV.lm_python()
LM_IDLE_S = 600.0                    # the owner must have been away this long before training starts
LM_MIN_FREE_GB = float(DEV.settings()["lm_min_free_gb"])    # 3.0 on the 16 GB development machine; scales with RAM
LM_MINUTES = 20
LM_MIX = "dialogue"                 # training mix for the idle trainer: "dialogue" (stories + dialogue turns) or "none" (stories only)
LM_DIALOGUE_SHARE = 0.2
LM_GRACE_S = 90.0                    # how long a run gets to stop itself via its STOP file before the process tree is ended
LM_RESTART_GAP_S = 60.0              # never relaunch faster than this (a crashing trainer must not spin)


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32                                          # type: ignore[attr-defined]
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, pid)                               # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        k.CloseHandle(ctypes.c_void_p(h))
        return bool(ok) and code.value == 259                               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def boot_time() -> float:
    """Epoch seconds of the last boot, or 0.0 when unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32                                      # type: ignore[attr-defined]
            k.GetTickCount64.restype = ctypes.c_ulonglong
            return time.time() - k.GetTickCount64() / 1000.0
        return time.time() - float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError, AttributeError):
        return 0.0


def claim_pidfile() -> bool:
    raw: str | None = None
    try:
        raw = PIDFILE.read_text(encoding="utf-8")
        old = int(raw.strip() or 0)
        if PIDFILE.stat().st_mtime < boot_time() - 5.0:
            old = 0                                    # written before this boot: the pid now belongs to an unrelated process
    except (OSError, ValueError):
        old = 0
    if old and old != os.getpid() and alive(old):
        return False                                                        # never displace a live supervisor
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        if PIDFILE.read_text(encoding="utf-8") != raw:
            return False                                                    # a rival replaced it while we judged it stale
    except OSError:
        pass
    try:
        PIDFILE.unlink()                                                    # absent, ours, or judged stale above
    except OSError:
        pass
    try:
        fd = os.open(str(PIDFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)    # exclusive: of two supervisors starting together, one loses
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return True


def kill_pid_tree(pid: int) -> None:
    """End the process tree rooted at exactly this pid (never by image name)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def write_swarm_pid(pid: int) -> None:
    SWARM_PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    SWARM_PIDFILE.write_text(f"{pid} {time.time():.0f} {os.getpid()}", encoding="utf-8")


def clear_swarm_pid(pid: int) -> None:
    try:
        if SWARM_PIDFILE.read_text(encoding="utf-8").split()[0] == str(pid):
            SWARM_PIDFILE.unlink()
    except (OSError, IndexError):
        pass


def reap_orphan_swarm() -> Optional[int]:
    """2 Oct: a supervisor hard-killed without /T left its swarm running and the next supervisor started a SECOND one. A recorded
    swarm whose recording supervisor is dead is ended here (that pid's tree only), then the record is removed. A record written
    before this boot names a pid that now belongs to someone else: it is only deleted. Returns the pid ended, if any."""
    try:
        parts = SWARM_PIDFILE.read_text(encoding="utf-8").split()
        swarm, sup = int(parts[0]), int(parts[2])
        before_boot = SWARM_PIDFILE.stat().st_mtime < boot_time() - 5.0
    except (OSError, ValueError, IndexError):
        try:
            SWARM_PIDFILE.unlink()
        except OSError:
            pass
        return None
    ended: Optional[int] = None
    if not before_boot and sup != os.getpid() and not alive(sup) and swarm != os.getpid() and alive(swarm):
        log(f"recorded swarm pid={swarm} outlived its supervisor {sup}: ending its process tree")
        kill_pid_tree(swarm)
        ended = swarm
    elif not before_boot and alive(sup) and sup != os.getpid():
        return None                                                     # its supervisor is alive: not ours to touch
    try:
        SWARM_PIDFILE.unlink()
    except OSError:
        pass
    return ended


def stop_tree(proc: subprocess.Popen) -> None:
    """Stop the swarm AND everything it started (2 Oct: terminating only the swarm left six pytest runs - and their model
    servers - running in its sandboxes). On Windows `taskkill /T` ends the whole tree by PID; elsewhere the process group."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def _lm_log(msg: str) -> None:
    LM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LM_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


class LMTrainer:
    """Runs `nupen_lm.py train` only while the machine is idle, at most one at a time. `tick()` is called every poll."""

    def __init__(self, python: Path = LM_PYTHON, idle=None, free_gb=None, spawn=None, stop_file: Path | None = None,
                 stop=None, clock=time.monotonic, log=_lm_log, idle_s: float = LM_IDLE_S, min_free_gb: float = LM_MIN_FREE_GB,
                 grace_s: float = LM_GRACE_S, restart_gap_s: float = LM_RESTART_GAP_S, blocked=None) -> None:    # type: ignore[no-untyped-def]
        if idle is None or free_gb is None:
            from creator import swarm as W
            idle, free_gb = idle or W.user_idle_seconds, free_gb or W.free_ram_gb
        self.python, self.idle, self.free_gb, self.clock, self.log = python, idle, free_gb, clock, log
        self.spawn = spawn or self._spawn
        self.stop_tree = stop or stop_tree
        self.stop_file = stop_file if stop_file is not None else DEV.runtime_dir() / "lmckpt" / "STOP"
        self.idle_s, self.min_free_gb, self.grace_s, self.restart_gap_s = idle_s, min_free_gb, grace_s, restart_gap_s
        self.proc: Any = None
        self.stopping_since: float | None = None
        self.last_start = -1e18
        self.warned = False
        self.blocked = blocked or (lambda: False)          # another background job (practice) runs: one at a time

    def cmd(self) -> list[str]:
        c = [str(self.python), "-u", str(ROOT / "scripts" / "nupen_lm.py"), "train", "--minutes", str(LM_MINUTES)]
        if LM_MIX != "none":
            c += ["--mix", LM_MIX, "--dialogue-share", str(LM_DIALOGUE_SHARE)]
        return c

    def _spawn(self, cmd: list[str]) -> Any:
        flags = (IDLE_PRIORITY | NO_WINDOW) if sys.platform == "win32" else 0
        with (STATE / "lm_train.log").open("a", encoding="utf-8") as out:
            return subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                    start_new_session=sys.platform != "win32")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _request_stop(self, why: str) -> None:
        if self.stopping_since is None:
            self.stopping_since = self.clock()
            self.log(f"stopping the LM trainer: {why}")
            try:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.write_text("stop", encoding="utf-8")
            except OSError:
                pass

    def tick(self, halt: bool = False) -> None:
        """One supervision step. `halt` = NUPEN_STOP exists (the whole of Nupen is switched off)."""
        if self.proc is not None and self.proc.poll() is not None:
            self.log(f"LM trainer exited code={self.proc.returncode}")
            self.proc, self.stopping_since = None, None
        if self.proc is not None:
            if halt:
                self._request_stop("NUPEN_STOP")
            elif self.idle() < self.idle_s:
                self._request_stop("the owner is back")
            if self.stopping_since is not None and self.clock() - self.stopping_since >= self.grace_s:
                self.log("LM trainer did not stop by itself: ending its process tree")
                self.stop_tree(self.proc)
                self.proc, self.stopping_since = None, None
            return
        if halt or self.clock() - self.last_start < self.restart_gap_s or self.blocked():
            return
        if not Path(self.python).is_file():
            if not self.warned:
                self.warned = True
                self.log(f"no LM environment at {self.python}: the LM trainer is skipped")
            return
        if self.idle() < self.idle_s or self.free_gb() < self.min_free_gb:
            return
        self.last_start = self.clock()
        try:
            self.stop_file.unlink()                                         # a stale STOP would end the new run at once
        except OSError:
            pass
        self.proc = self.spawn(self.cmd())
        self.log(f"LM trainer started pid={getattr(self.proc, 'pid', '?')} (idle {self.idle():.0f}s, free {self.free_gb():.1f} GB)")

    def shutdown(self) -> None:
        if self.running():
            self._request_stop("supervisor exiting")
            end = self.clock() + self.grace_s
            while self.proc.poll() is None and self.clock() < end:       # a clean stop writes its checkpoint first
                time.sleep(1.0)
            if self.proc.poll() is None:
                self.stop_tree(self.proc)
        self.proc = None


PRACTICE_LOG = STATE / "practice_service.log"
PRACTICE_ROWS = STATE / "practice_rows.jsonl"
PRACTICE_MINUTES = 20
PRACTICE_GAP_S = 300.0               # between practice runs
PRACTICE_EMPTY_GAP_S = 1800.0        # a run that measured nothing new (every file at this revision is done): wait longer


def _practice_log(msg: str) -> None:
    PRACTICE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PRACTICE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


class PracticeRunner:
    """Offline practice rounds (scripts/practice.py) on the CURRENT main revision, then a chooser retrain, only while the owner has
    been idle 10+ min and RAM is free - modelled on LMTrainer, one background job at a time: it never starts while the LM trainer runs
    (and the trainer is blocked while this runs). No STOP file: practice rows are appended and flushed one by one, so the process
    tree is simply ended when the owner returns or NUPEN_STOP appears. Stages: 'practice', then 'retrain'
    (scripts/train_chooser.py --practice, a no-op unless new rows arrived)."""

    def __init__(self, python: str | None = None, idle=None, free_gb=None, spawn=None, stop=None, clock=time.monotonic,    # type: ignore[no-untyped-def]
                 log=_practice_log, lm_running=None, rows_path: Path = PRACTICE_ROWS, idle_s: float = LM_IDLE_S,
                 min_free_gb: float = LM_MIN_FREE_GB, gap_s: float = PRACTICE_GAP_S, empty_gap_s: float = PRACTICE_EMPTY_GAP_S,
                 minutes: int = PRACTICE_MINUTES) -> None:
        if idle is None or free_gb is None:
            from creator import swarm as W
            idle, free_gb = idle or W.user_idle_seconds, free_gb or W.free_ram_gb
        venv = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        self.python = python or str(venv if venv.exists() else sys.executable)
        self.idle, self.free_gb, self.clock, self.log = idle, free_gb, clock, log
        self.spawn = spawn or self._spawn
        self.stop_tree = stop or stop_tree
        self.lm_running = lm_running or (lambda: False)
        self.rows_path, self.idle_s, self.min_free_gb, self.minutes = rows_path, idle_s, min_free_gb, minutes
        self.gap_s, self.empty_gap_s = gap_s, empty_gap_s
        self.proc: Any = None
        self.stage = ""
        self.rows_before = 0
        self.next_start = -1e18

    def cmd(self, stage: str) -> list[str]:
        if stage == "practice":
            return [self.python, "-u", str(ROOT / "scripts" / "practice.py"), "--repo", str(ROOT), "--rev", "HEAD",
                    "--minutes", str(self.minutes)]
        return [self.python, "-u", str(ROOT / "scripts" / "train_chooser.py"), "--practice"]

    def _spawn(self, cmd: list[str]) -> Any:
        flags = (IDLE_PRIORITY | NO_WINDOW) if sys.platform == "win32" else 0
        with (STATE / "practice_run.log").open("a", encoding="utf-8") as out:
            return subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                    start_new_session=sys.platform != "win32")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _end(self, why: str) -> None:
        self.log(f"ending the practice job ({self.stage}): {why}")
        self.stop_tree(self.proc)
        self.proc, self.stage = None, ""
        self.next_start = self.clock() + self.gap_s

    def tick(self, halt: bool = False) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            self.log(f"practice job {self.stage} exited code={self.proc.returncode}")
            stage, self.proc, self.stage = self.stage, None, ""
            new = _count_lines(self.rows_path) - self.rows_before
            if stage == "practice" and new > 0 and not halt:
                self.stage = "retrain"
                self.proc = self.spawn(self.cmd("retrain"))
                self.log(f"{new} new practice rows: retraining the chooser")
                return
            self.next_start = self.clock() + (self.gap_s if new > 0 else self.empty_gap_s)
        if self.proc is not None:
            if halt:
                self._end("NUPEN_STOP")
            elif self.idle() < self.idle_s:
                self._end("the owner is back")
            return
        if halt or self.lm_running() or self.clock() < self.next_start:
            return
        if self.idle() < self.idle_s or self.free_gb() < self.min_free_gb:
            return
        self.rows_before = _count_lines(self.rows_path)
        self.stage = "practice"
        self.proc = self.spawn(self.cmd("practice"))
        self.log(f"practice started (idle {self.idle():.0f}s, free {self.free_gb():.1f} GB, {self.rows_before} rows so far)")

    def shutdown(self) -> None:
        if self.running():
            self.stop_tree(self.proc)
        self.proc = None


def swarm_cmd(python: str) -> list[str]:
    return [python, "-u", str(ROOT / "scripts" / "creator_swarm.py"), "--rounds", "0", "--packages", "24",
            "--teacher-presence"] + (["--user-aware"] if DEV.settings()["user_aware"] else [])   # device_overrides.json can turn it off


def beat() -> None:
    try:
        HEARTBEAT.write_text(f"{os.getpid()} {time.time():.0f}", encoding="utf-8")
    except OSError:
        pass


_WATCHDOG_SPAWNED = -1e9


def ensure_watchdog() -> None:
    """Keep the watchdog alive (2 Oct, owner: 'ensure whenever Nupen is supposed to be running it is running'): the supervisor
    and the watchdog each restart the other, so one dying - or being killed - never leaves Nupen down."""
    if os.environ.get("NUPEN_WATCHDOG", "1") == "0":
        return
    try:
        pid = int(WATCHDOG_PIDFILE.read_text(encoding="utf-8").strip() or 0)
        if WATCHDOG_PIDFILE.stat().st_mtime < boot_time() - 5.0:
            pid = 0                                                     # written before this boot: the pid is someone else's now
    except (OSError, ValueError):
        pid = 0
    if pid and alive(pid):
        return
    global _WATCHDOG_SPAWNED
    if time.monotonic() - _WATCHDOG_SPAWNED < 120.0:                    # a just-started watchdog has not claimed its pidfile yet
        return
    _WATCHDOG_SPAWNED = time.monotonic()
    pyw = DEV.venv_python(ROOT / ".venv", windowless=True)
    exe = str(pyw if pyw.exists() else sys.executable)
    flags = (0x00000008 | 0x00000200 | NO_WINDOW) if sys.platform == "win32" else 0     # detached, own group, no window
    try:
        subprocess.Popen([exe, str(ROOT / "scripts" / "nupen_watchdog.py")], cwd=ROOT, creationflags=flags, close_fds=True,
                         start_new_session=sys.platform != "win32")
        log("watchdog (re)started")
    except OSError as e:
        log(f"watchdog could not start: {e}")


def _safe(what: str, fn: Any, *a: Any) -> Any:
    """An optional part (the LM trainer, the watchdog) must never take the supervisor down with it."""
    try:
        return fn(*a)
    except Exception:                                                   # noqa: BLE001 - logged in full, never silent
        import traceback
        log(f"{what} failed (supervisor continues):\n{traceback.format_exc()}")
        return None


def run(python: str, poll_s: float = 10.0) -> int:
    if not claim_pidfile():
        log("another supervisor is alive; exiting")
        return 0
    log(f"supervisor up pid={os.getpid()}")
    _safe("orphan swarm reap", reap_orphan_swarm)
    DEV.write_snapshot(log=log)                                         # state/creator/device.json; a changed machine is logged
    try:
        import faulthandler
        faulthandler.enable(open(CRASHLOG, "a", encoding="utf-8"))     # noqa: SIM115 - must stay open for the process lifetime
    except OSError:
        pass
    try:
        while not STOP.exists():
            try:
                _supervise(python, poll_s)
                return 0
            except Exception:                                           # noqa: BLE001 - the supervisor never dies on an error
                import traceback
                log(f"supervisor error, retrying in 30 s:\n{traceback.format_exc()}")
                for _ in range(3):
                    if STOP.exists():
                        break
                    beat()
                    time.sleep(min(poll_s, 10.0))
    finally:
        try:
            if PIDFILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
        log("supervisor down")
    return 0


def _supervise(python: str, poll_s: float) -> None:
    backoff = 30.0
    pr: Any = _safe("practice start", PracticeRunner) if os.environ.get("NUPEN_PRACTICE", "1") != "0" else None
    lm: Any = _safe("LM trainer start", LMTrainer) if os.environ.get("NUPEN_LM", "1") != "0" else None
    if pr and lm:
        pr.lm_running = lm.running
        lm.blocked = pr.running
    beat()
    _safe("watchdog check", ensure_watchdog)
    try:
        while not STOP.exists():
            started = time.monotonic()
            with (STATE / "swarm_service.log").open("a", encoding="utf-8") as out:
                flags = (BELOW_NORMAL | NO_WINDOW) if sys.platform == "win32" else 0
                proc = subprocess.Popen(swarm_cmd(python), cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                        start_new_session=sys.platform != "win32")    # its own group: stop_tree ends it all
                log(f"swarm started pid={proc.pid}")
                write_swarm_pid(proc.pid)
                try:
                    while proc.poll() is None:
                        if STOP.exists():
                            log("NUPEN_STOP found: stopping the swarm")
                            stop_tree(proc)
                            break
                        if lm:
                            _safe("LM trainer tick", lm.tick, STOP.exists())
                        if pr:
                            _safe("practice tick", pr.tick, STOP.exists())
                        beat()
                        _safe("watchdog check", ensure_watchdog)
                        time.sleep(poll_s)
                except BaseException:                                   # an error here must not orphan the swarm: run() restarts us
                    if proc.poll() is None:                             # and would start a SECOND swarm beside the first
                        log("supervision failed with the swarm running: ending its process tree")
                        _safe("swarm stop", stop_tree, proc)
                    raise
            clear_swarm_pid(proc.pid)
            log(f"swarm exited code={proc.returncode}")
            if STOP.exists():
                break
            backoff = 30.0 if time.monotonic() - started > 1800 else min(backoff * 2, 1800.0)
            waited = 0.0
            while waited < backoff and not STOP.exists():               # the trainer keeps being supervised between swarm runs
                if lm:
                    _safe("LM trainer tick", lm.tick, False)
                if pr:
                    _safe("practice tick", pr.tick, False)
                beat()
                time.sleep(min(poll_s, backoff - waited))
                waited += poll_s
    finally:
        if lm:
            _safe("LM trainer shutdown", lm.shutdown)
        if pr:
            _safe("practice shutdown", pr.shutdown)


if __name__ == "__main__":
    venv = DEV.venv_python(ROOT / ".venv")
    raise SystemExit(run(str(venv if venv.exists() else sys.executable)))
