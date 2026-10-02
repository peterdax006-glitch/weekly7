"""Make Nupen start when the owner logs on (owner's request, 1 Oct 2026) - or stop that.

    python scripts/nupen_autostart.py install     # Windows Task Scheduler task "Nupen", at logon, as this user, no window
    python scripts/nupen_autostart.py uninstall   # remove it (Nupen no longer starts with the computer)
    python scripts/nupen_autostart.py status

The task runs scripts/nupen_service.py with the venv's pythonw.exe (no console). To stop a running Nupen at once: create
state/creator/NUPEN_STOP. Nothing here needs administrator rights; the task runs with the user's normal, limited rights.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "Nupen"


def command() -> str:
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    return f'"{pyw}" "{ROOT / "scripts" / "nupen_service.py"}"'


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "status"
    if what == "install":
        args = ["schtasks", "/Create", "/TN", TASK, "/SC", "ONLOGON", "/TR", command(), "/RL", "LIMITED", "/F"]
    elif what == "uninstall":
        args = ["schtasks", "/Delete", "/TN", TASK, "/F"]
    elif what == "status":
        args = ["schtasks", "/Query", "/TN", TASK, "/V", "/FO", "LIST"]
    else:
        print(__doc__)
        return 2
    r = subprocess.run(args, capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
