"""Make Nupen start when the owner logs on (owner's request, 1 Oct 2026) - or stop that.

    python scripts/nupen_autostart.py install     # per-user Run entry "Nupen" (HKCU\\...\\Run): starts at logon, no window
    python scripts/nupen_autostart.py uninstall   # remove it (Nupen no longer starts with the computer)
    python scripts/nupen_autostart.py status
    python scripts/nupen_autostart.py start       # start the supervisor now (detached, no window)

The entry runs scripts/nupen_service.py with the venv's pythonw.exe (no console), with the user's normal rights - no
administrator needed (a Task Scheduler ONLOGON task was refused with 'Access is denied' on 2 Oct). To stop a running Nupen at
once: create state/creator/NUPEN_STOP.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Nupen"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
DETACHED = 0x00000008 | 0x00000200 | 0x08000000                # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW


def command() -> str:
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    return f'"{pyw}" "{ROOT / "scripts" / "nupen_service.py"}"'


def main(argv: list[str]) -> int:
    import winreg
    what = argv[0] if argv else "status"
    if what == "install":
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, command())
        print(f"installed: {NAME} = {command()}")
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
        pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        p = subprocess.Popen([str(pyw), str(ROOT / "scripts" / "nupen_service.py")], cwd=ROOT, creationflags=DETACHED,
                             close_fds=True)
        print(f"supervisor started pid={p.pid}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
