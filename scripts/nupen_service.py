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

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "creator"
STOP = STATE / "NUPEN_STOP"
PIDFILE = STATE / "nupen_service.pid"
LOG = STATE / "nupen_service.log"
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000


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
    try:
        old = int(PIDFILE.read_text(encoding="utf-8").strip() or 0)
        if PIDFILE.stat().st_mtime < boot_time() - 5.0:
            old = 0                                    # written before this boot: the pid now belongs to an unrelated process
    except (OSError, ValueError):
        old = 0
    if old and old != os.getpid() and alive(old):
        return False                                                        # never displace a live supervisor
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
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


def swarm_cmd(python: str) -> list[str]:
    return [python, "-u", str(ROOT / "scripts" / "creator_swarm.py"), "--rounds", "0", "--packages", "24",
            "--user-aware", "--teacher-presence"]


def run(python: str, poll_s: float = 10.0) -> int:
    if not claim_pidfile():
        log("another supervisor is alive; exiting")
        return 0
    log(f"supervisor up pid={os.getpid()}")
    backoff = 30.0
    try:
        while not STOP.exists():
            started = time.monotonic()
            with (STATE / "swarm_service.log").open("a", encoding="utf-8") as out:
                flags = (BELOW_NORMAL | NO_WINDOW) if sys.platform == "win32" else 0
                proc = subprocess.Popen(swarm_cmd(python), cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
                log(f"swarm started pid={proc.pid}")
                while proc.poll() is None:
                    if STOP.exists():
                        log("NUPEN_STOP found: stopping the swarm")
                        proc.terminate()
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    time.sleep(poll_s)
            log(f"swarm exited code={proc.returncode}")
            if STOP.exists():
                break
            backoff = 30.0 if time.monotonic() - started > 1800 else min(backoff * 2, 1800.0)
            time.sleep(backoff)
    finally:
        try:
            if PIDFILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
        log("supervisor down")
    return 0


if __name__ == "__main__":
    venv = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    raise SystemExit(run(str(venv if venv.exists() else sys.executable)))
