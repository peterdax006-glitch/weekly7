"""Static checks on touched Python files: compile (always, in-process), ruff (when installed), mypy (optional, opt-in).

Results are cached by CONTENT hash (creator/diskcache.py), keyed with the tool version, so an unchanged file is never re-checked and a
different file with identical bytes shares the entry. A missing tool is a graceful skip ("skipped"), never a failure. ruff and mypy run
once for all uncached files of a call, not once per file."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from creator import diskcache as DC

_SALT = DC.salt_of((), (b"staticcheck-v1",))


@dataclass(frozen=True)
class FileCheck:
    path: str
    compile_ok: bool
    ruff: str = "skipped"             # ok | issues | skipped
    mypy: str = "skipped"             # ok | issues | skipped
    messages: tuple[str, ...] = ()
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.compile_ok and self.ruff != "issues" and self.mypy != "issues"


@dataclass(frozen=True)
class CheckReport:
    files: tuple[FileCheck, ...]
    seconds: float = 0.0
    tools: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.files)

    def messages(self) -> list[str]:
        return [m for f in self.files for m in f.messages]


def _ruff_cmd() -> Optional[list[str]]:
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return None


def _mypy_ok() -> bool:
    return importlib.util.find_spec("mypy") is not None


def _version(cmd: list[str]) -> str:
    try:
        return subprocess.run([*cmd, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _compile(path: Path, raw: bytes) -> tuple[bool, str]:
    try:
        compile(raw, str(path), "exec", dont_inherit=True)
        return True, ""
    except (SyntaxError, ValueError) as e:
        line = getattr(e, "lineno", 0) or 0
        return False, f"{path.name}:{line}: compile: {getattr(e, 'msg', e)}"


def _split_by_file(out: str, root: Path, rels: dict[str, str]) -> dict[str, list[str]]:
    """Group tool output lines (`path:line:col: msg`) by the file they name."""
    got: dict[str, list[str]] = {r: [] for r in rels}
    for ln in out.splitlines():
        m = re.match(r"^(.+?):\d+(?::\d+)?[: ]", ln)
        if not m:
            continue
        p = m.group(1).replace("\\", "/")
        for rel in rels:
            if p == rel or p.endswith("/" + rel) or Path(p).as_posix() == (root / rel).as_posix():
                got[rel].append(ln.strip())
                break
    return got


def check_files(root: str | Path, rel_paths: Iterable[str], *, mypy: bool = False, timeout: float = 120.0) -> CheckReport:
    import time
    t0 = time.monotonic()
    rootp = Path(root)
    rels = [r.replace("\\", "/") for r in rel_paths if r.endswith(".py")]
    ruff_cmd = _ruff_cmd()
    use_mypy = mypy and _mypy_ok()
    tool_id = f"ruff={_version(ruff_cmd) if ruff_cmd else '-'};mypy={'1' if use_mypy else '-'}"
    results: dict[str, FileCheck] = {}
    todo: dict[str, tuple[str, bytes]] = {}
    for rel in rels:
        p = rootp / rel
        try:
            raw = p.read_bytes()
        except OSError:
            results[rel] = FileCheck(rel, False, messages=(f"{rel}: unreadable or missing",))
            continue
        key = DC.key_of(hashlib.sha256(raw).digest(), tool_id, rel if use_mypy else "")
        hit = DC.get("staticcheck", _SALT, key)
        if hit is not None:
            results[rel] = FileCheck(rel, hit[0], hit[1], hit[2], tuple(hit[3]), True)
        else:
            todo[rel] = (key, raw)
    comp: dict[str, tuple[bool, str]] = {rel: _compile(rootp / rel, raw) for rel, (_k, raw) in todo.items()}
    ruff_out: dict[str, list[str]] = {}
    if ruff_cmd and todo:
        good = {r: r for r in todo if comp[r][0]}
        if good:
            try:
                cp = subprocess.run([*ruff_cmd, "check", "--no-cache", "--output-format=concise", *good], cwd=str(rootp),
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
                ruff_out = _split_by_file(cp.stdout, rootp, good) if cp.returncode == 1 else {r: [] for r in good}
            except (OSError, subprocess.SubprocessError):
                ruff_out = {}
    mypy_out: dict[str, list[str]] = {}
    if use_mypy and todo:
        good = {r: r for r in todo if comp[r][0]}
        if good:
            try:
                cp = subprocess.run([sys.executable, "-m", "mypy", "--no-error-summary", "--show-column-numbers", "--follow-imports=silent",
                                     "--cache-dir", str(rootp / ".mypy_cache"), *good], cwd=str(rootp), capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout)
                mypy_out = _split_by_file(cp.stdout, rootp, good)
            except (OSError, subprocess.SubprocessError):
                mypy_out = {}
    for rel, (key, _raw) in todo.items():
        ok, msg = comp[rel]
        msgs = [msg] if msg else []
        rs = "skipped"
        if rel in ruff_out:
            rs = "issues" if ruff_out[rel] else "ok"
            msgs += ruff_out[rel]
        ms = "skipped"
        if rel in mypy_out:
            errs = [m for m in mypy_out[rel] if ": error:" in m]
            ms = "issues" if errs else "ok"
            msgs += errs
        fc = FileCheck(rel, ok, rs, ms, tuple(msgs))
        results[rel] = fc
        if rs != "skipped" or ms != "skipped" or not ruff_cmd:        # do not cache a result a missing optional tool could change
            DC.put("staticcheck", _SALT, key, (fc.compile_ok, fc.ruff, fc.mypy, list(fc.messages)))
    return CheckReport(tuple(results[r] for r in rels if r in results), time.monotonic() - t0,
                       {"ruff": "yes" if ruff_cmd else "skipped", "mypy": "yes" if use_mypy else "skipped"})


def report_dict(rep: CheckReport) -> dict[str, Any]:
    return {"ok": rep.ok, "seconds": round(rep.seconds, 3), "tools": rep.tools, "messages": rep.messages(),
            "files": {f.path: {"compile": f.compile_ok, "ruff": f.ruff, "mypy": f.mypy, "cached": f.cached} for f in rep.files}}
