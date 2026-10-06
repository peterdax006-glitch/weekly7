"""Local Jelly compiler: Open-Jellycore (github.com/OpenJelly/Open-Jellycore, Swift) built inside WSL Ubuntu, called from Windows.

compile_code(code) -> {"available", "ok", "errors", "output"}; "available" is False (and nothing fails) when WSL or the built `jelly`
binary is missing, so callers can treat the compiler as an optional second opinion next to shortcutgen.validate's static rules.

Setup (once): WSL + Ubuntu, Swift via swiftly, then `swift build -c release` in the Open-Jellycore checkout (see
<runtime>/phone/PHONE_CONTROL.md). The binary path inside WSL is env NUPEN_JELLY_BIN, else JELLY_BIN below; the distro is
env NUPEN_JELLY_DISTRO, else Ubuntu.

The locally patched CLI prints one line per diagnostic ("level<TAB>line<TAB>description<TAB>recovery"), then either
"Successfully Compiled Shortcut" or "Found N errors" (exit 1). Warnings (e.g. an optional parameter left out) never fail a
compile. Open-Jellycore only knows part of the private Jellycore's language (no Objects), so a
compile error here is a strong hint, not proof that the Jellycuts iOS app would reject the code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

JELLY_BIN = "/root/src/Open-Jellycore/.build/release/jelly"
TIMEOUT_S = 120
OK_LINE = "Successfully Compiled Shortcut"
Runner = Callable[[list[str], int], tuple[int, str]]


def _distro() -> str:
    return os.environ.get("NUPEN_JELLY_DISTRO", "Ubuntu")


def _bin() -> str:
    return os.environ.get("NUPEN_JELLY_BIN", JELLY_BIN)


def wsl_path(p: Path) -> str:
    """C:\\Users\\x\\f.jelly -> /mnt/c/Users/x/f.jelly"""
    s = str(Path(p).resolve())
    if len(s) > 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    env = dict(os.environ, WSL_UTF8="1")
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return 124, f"{type(e).__name__}: {e}"
    out = (p.stdout or b"") + (b"\n" + p.stderr if p.stderr else b"")
    return p.returncode, out.decode("utf-8", "replace").replace("\x00", "")


def _wsl(args: list[str]) -> list[str]:
    return ["wsl.exe", "-d", _distro(), "-u", "root", "--", *args]


def available(run: Runner = _run) -> bool:
    if run is _run and not shutil.which("wsl.exe") and not shutil.which("wsl"):
        return False
    rc, _ = run(_wsl(["test", "-x", _bin()]), 60)
    return rc == 0


ERROR_LEVELS = ("syntax", "error", "fatal")


def parse(rc: int, out: str) -> dict[str, Any]:
    """Our local build prints 'level<TAB>line<TAB>description<TAB>recovery' per diagnostic (see <runtime>/jelly/patch_cli.py);
    an unpatched build prints 'description recovery' with no level, which then all count as errors when the compile failed."""
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    ok = rc == 0 and any(OK_LINE in ln for ln in lines)
    errors: list[str] = []
    warnings: list[str] = []
    for ln in lines:
        if OK_LINE in ln or ln.startswith("Found "):
            continue
        parts = ln.split("\t")
        if len(parts) >= 3 and parts[0] in ERROR_LEVELS + ("warning",):
            msg = (f"line {parts[1]}: " if parts[1] != "-" else "") + parts[2] + (f" ({parts[3]})" if len(parts) > 3 and parts[3] else "")
            (warnings if parts[0] == "warning" else errors).append(msg)
        elif not ok:
            errors.append(ln)
    if not ok and not errors:
        errors = [f"jelly exited {rc} with no diagnostics"]
    return {"available": True, "ok": ok, "errors": errors[:50], "warnings": warnings[:50], "output": out[-4000:]}


def compile_file(path: Path, export: Optional[Path] = None, run: Runner = _run) -> dict[str, Any]:
    args = [_bin(), wsl_path(path)]
    if export is not None:
        args += ["--export", "--out", wsl_path(export)]
    rc, out = run(_wsl(args), TIMEOUT_S)
    return parse(rc, out)


def compile_code(code: str, run: Runner = _run, check: Callable[[], bool] | None = None) -> dict[str, Any]:
    if not (check or (lambda: available(run)))():
        return {"available": False, "ok": None, "errors": [], "output": "jelly compiler not installed (WSL + Open-Jellycore build)"}
    d = Path(tempfile.mkdtemp(prefix="jellyc_"))
    try:
        f = d / "in.jelly"
        f.write_text(code, encoding="utf-8", newline="\n")
        return compile_file(f, run=run)
    finally:
        shutil.rmtree(d, ignore_errors=True)
