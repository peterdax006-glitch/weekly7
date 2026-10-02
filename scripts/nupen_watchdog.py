"""Nupen's watchdog (owner, 2 Oct 2026: "ensure whenever Nupen is supposed to be running it is running").

Every minute: unless state/creator/NUPEN_STOP exists, the supervisor (scripts/nupen_service.py) must be alive AND its heartbeat
(state/creator/nupen_service.heartbeat, written every poll) fresh. If it is dead, or alive but silent for longer than STALE_S
(hung), the watchdog ends the stale supervisor's process tree and starts a new one. The supervisor in turn restarts a dead
watchdog, so killing either one alone never leaves Nupen down. NUPEN_STOP is the only off switch and is always respected; the
watchdog keeps watching while stopped, so removing the file brings Nupen back. Every action is logged with its reason.
One watchdog at a time (pidfile; a live holder is never displaced)."""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATE = ROOT / "state" / "creator"
STOP = STATE / "NUPEN_STOP"
PIDFILE = STATE / "nupen_watchdog.pid"
SUPERVISOR_PIDFILE = STATE / "nupen_service.pid"
HEARTBEAT = STATE / "nupen_service.heartbeat"
LOG = STATE / "nupen_watchdog.log"
STALE_S = 300.0
GRACE_S = 120.0                                   # a freshly started supervisor gets this long to write its first beat
DETACHED = 0x00000008 | 0x00000200 | 0x08000000   # DETACHED_PROCESS | NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def log(msg: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def _alive(pid: int) -> bool:
    """The supervisor's own liveness check (one implementation; loading its module only defines functions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("nupen_service", ROOT / "scripts" / "nupen_service.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return bool(mod.alive(pid))


def _read_pid(p: Path) -> int:
    try:
        return int(p.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def heartbeat_age(now: Optional[float] = None) -> Optional[float]:
    try:
        stamp = float(HEARTBEAT.read_text(encoding="utf-8").split()[-1])
    except (OSError, ValueError, IndexError):
        return None
    return (now or time.time()) - stamp


def supervisor_state(now: Optional[float] = None) -> tuple[str, int]:
    """('ok' | 'dead' | 'hung', pid)."""
    pid = _read_pid(SUPERVISOR_PIDFILE)
    if not pid or not _alive(pid):
        return "dead", pid
    age = heartbeat_age(now)
    if age is not None and age > STALE_S:
        return "hung", pid
    return "ok", pid


def kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass


def start_supervisor() -> None:
    pyw = ROOT / ".venv" / ("Scripts/pythonw.exe" if sys.platform == "win32" else "bin/python")
    exe = str(pyw if pyw.exists() else sys.executable)
    subprocess.Popen([exe, str(ROOT / "scripts" / "nupen_service.py")], cwd=ROOT,
                     creationflags=DETACHED if sys.platform == "win32" else 0, close_fds=True,
                     start_new_session=sys.platform != "win32")


def check_once(now: Optional[float] = None, last_start: float = 0.0) -> str:
    """One watchdog decision; returns the action taken ('stopped', 'ok', 'restarted:<why>', 'grace')."""
    if STOP.exists():
        return "stopped"
    t = now or time.time()
    if t - last_start < GRACE_S:
        return "grace"
    state, pid = supervisor_state(t)
    if state == "ok":
        return "ok"
    if state == "hung":
        log(f"supervisor {pid} alive but heartbeat stale (> {STALE_S:.0f} s): ending its tree")
        kill_tree(pid)
        try:
            SUPERVISOR_PIDFILE.unlink()
        except OSError:
            pass
    else:
        log(f"supervisor {pid or '-'} not running")
    start_supervisor()
    log(f"supervisor restarted ({state})")
    return f"restarted:{state}"


def claim() -> bool:
    old = _read_pid(PIDFILE)
    if old and old != os.getpid() and _alive(old):
        return False
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def main(poll_s: float = 60.0) -> int:
    if not claim():
        return 0
    log(f"watchdog up pid={os.getpid()}")
    last_start = 0.0
    try:
        while True:
            try:
                act = check_once(last_start=last_start)
                if act.startswith("restarted"):
                    last_start = time.time()
            except Exception as e:                                      # noqa: BLE001 - the watchdog itself never dies
                log(f"watchdog error: {type(e).__name__}: {e}")
            time.sleep(poll_s)
    finally:
        if _read_pid(PIDFILE) == os.getpid():
            try:
                PIDFILE.unlink()
            except OSError:
                pass
        log("watchdog down")


if __name__ == "__main__":
    raise SystemExit(main())
