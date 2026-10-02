"""Make Nupen start when the owner logs on (owner's request, 1 Oct 2026) - or stop that. Works on every OS Nupen runs on.

    python scripts/nupen_autostart.py install     # Windows: per-user Run entry "Nupen" (HKCU\\...\\Run); Linux: systemd user unit;
                                                  # macOS: launchd agent. Starts at logon, no window
    python scripts/nupen_autostart.py write       # Linux/macOS: write the unit/plist file only, enable nothing (Windows: no-op)
    python scripts/nupen_autostart.py uninstall   # remove it (Nupen no longer starts with the computer)
    python scripts/nupen_autostart.py status
    python scripts/nupen_autostart.py start       # start the supervisor now (detached, no window)

The entry runs scripts/nupen_service.py with the venv's pythonw.exe (Windows, no console) or python (elsewhere), with the user's
normal rights - no administrator needed (a Task Scheduler ONLOGON task was refused with 'Access is denied' on 2 Oct). To stop a
running Nupen at once: create state/creator/NUPEN_STOP.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creator import device as DEV  # noqa: E402

NAME = "Nupen"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
DETACHED = 0x00000008 | 0x00000200 | 0x08000000                # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
SYSTEMD_UNIT = Path.home() / ".config" / "systemd" / "user" / "nupen.service"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.nupen.service.plist"


def interpreter(root: Path = ROOT) -> Path:
    py = DEV.venv_python(root / ".venv", windowless=True)
    return py if (py.exists() or DEV.is_windows()) else Path(sys.executable)   # the Run entry names the venv even before it exists


def command(root: Path = ROOT) -> str:
    return f'"{interpreter(root)}" "{root / "scripts" / "nupen_service.py"}"'


def systemd_unit(root: Path = ROOT) -> str:
    return (f"[Unit]\nDescription=Nupen (self-developing system)\n\n[Service]\nType=simple\nWorkingDirectory={root}\n"
            f"ExecStart={interpreter(root)} {root / 'scripts' / 'nupen_service.py'}\nRestart=on-failure\nRestartSec=60\n\n"
            "[Install]\nWantedBy=default.target\n")


def launchd_plist(root: Path = ROOT) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>\n'
            '<key>Label</key><string>com.nupen.service</string>\n'
            f'<key>ProgramArguments</key><array><string>{interpreter(root)}</string>'
            f'<string>{root / "scripts" / "nupen_service.py"}</string></array>\n'
            f'<key>WorkingDirectory</key><string>{root}</string>\n'
            '<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>\n</dict></plist>\n')


def write_unit(root: Path = ROOT) -> Path:
    """Write (never enable) the systemd user unit or launchd plist. Returns the file written."""
    path, text = (LAUNCHD_PLIST, launchd_plist(root)) if sys.platform == "darwin" else (SYSTEMD_UNIT, systemd_unit(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run(*cmd: str) -> int:
    return subprocess.run(list(cmd), capture_output=True).returncode


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "status"
    if what not in ("install", "write", "uninstall", "status", "start"):
        print(__doc__)
        return 2
    if sys.platform == "win32":
        return _windows(what)
    mac = sys.platform == "darwin"
    if what in ("install", "write"):
        path = write_unit()
        print(f"written: {path}")
        if what == "install":
            rc = _run("launchctl", "load", str(path)) if mac else (_run("systemctl", "--user", "daemon-reload")
                                                                    or _run("systemctl", "--user", "enable", "--now", "nupen.service"))
            print("enabled" if rc == 0 else f"enable failed (exit {rc}); enable it by hand")
    elif what == "uninstall":
        path = LAUNCHD_PLIST if mac else SYSTEMD_UNIT
        if mac:
            _run("launchctl", "unload", str(path))
        else:
            _run("systemctl", "--user", "disable", "--now", "nupen.service")
        path.unlink(missing_ok=True)
        print("removed")
    elif what == "status":
        path = LAUNCHD_PLIST if mac else SYSTEMD_UNIT
        print(f"installed: {path}" if path.exists() else "not installed")
    elif what == "start":
        p = subprocess.Popen([str(interpreter()), str(ROOT / "scripts" / "nupen_service.py")], cwd=ROOT, close_fds=True,
                             start_new_session=True)
        print(f"supervisor started pid={p.pid}")
    return 0


def _windows(what: str) -> int:
    import winreg
    if what == "install":
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, command())
        print(f"installed: {NAME} = {command()}")
    elif what == "write":
        print(f"nothing to write on Windows; the Run entry would be: {command()}")
    elif what == "uninstall":
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, NAME)
            print("removed")
        except FileNotFoundError:
            print("not installed")
    elif what == "status":
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                print(f"installed: {winreg.QueryValueEx(k, NAME)[0]}")
        except FileNotFoundError:
            print("not installed")
    elif what == "start":
        p = subprocess.Popen([str(interpreter()), str(ROOT / "scripts" / "nupen_service.py")], cwd=ROOT, creationflags=DETACHED,
                             close_fds=True)
        print(f"supervisor started pid={p.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
