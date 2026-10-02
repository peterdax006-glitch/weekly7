"""Run a command at BELOW_NORMAL CPU priority (children inherit it), so Nupen's own work wins the CPU.

Owner, 1 Oct 2026: helpers may use the machine only where that is cheaper than giving the resources to Nupen. Measured that
evening: CPU at 100% on 8 cores with 4 GB RAM free - CPU, not RAM, is what helpers take from Nupen. Usage:
    python scripts/lowprio.py [--idle] <command> [args...]
--idle runs at IDLE priority instead (used for LM training and installs).
"""
from __future__ import annotations

import subprocess
import sys

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IDLE_PRIORITY_CLASS = 0x00000040


def lower_own_priority(cls: int = BELOW_NORMAL_PRIORITY_CLASS) -> None:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32                                # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p             # a HANDLE: the default int restype mangles it
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), cls):
            raise OSError(f"SetPriorityClass failed: {ctypes.GetLastError()}")
    else:
        import os
        os.nice(10)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    import os
    import shutil
    cls = BELOW_NORMAL_PRIORITY_CLASS
    if argv[0] == "--idle":
        cls = IDLE_PRIORITY_CLASS
        argv = argv[1:]
        if not argv:
            print(__doc__)
            return 2
    lower_own_priority(cls)
    argv = [os.path.abspath(shutil.which(argv[0]) or argv[0]), *argv[1:]]   # CreateProcess does not resolve 'a/b.exe' paths
    kw = {"creationflags": cls} if sys.platform == "win32" else {}
    return subprocess.call(argv, **kw)                                    # type: ignore[arg-type]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
