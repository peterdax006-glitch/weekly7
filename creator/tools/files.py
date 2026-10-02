"""Nupen tools - file tools (Claude Code's Read / Write / Edit / Grep / Glob), confined by the same policy as the shell: every path is
resolved and gated (sandbox + own scratchpad only, protected and secret paths refused, per-project access applied). Paths starting
`@scratch/` name the package's scratchpad."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from creator.tools import policy as PO

MAX_READ_BYTES = 2_000_000


class FileTools:
    def __init__(self, policy: PO.Policy, cwd_of: Optional[Any] = None) -> None:
        self.policy = policy
        self.cwd_of = cwd_of or (lambda: str(policy.sandbox))

    def _abs(self, path: str) -> str:
        if path.startswith("@scratch/") or path == "@scratch":
            return str(Path(self.policy.scratch) / path[len("@scratch"):].lstrip("/\\"))
        return path

    def _gate(self, tool: str, path: str, write: bool, delete: bool = False) -> tuple[Optional[dict[str, Any]], str]:
        d = self.policy.check_path(self._abs(path), write, str(self.cwd_of()), "files", delete)
        if not d:
            self.policy.log(d, tool, path)
            return {"ok": False, "denied": d.reason, "rule": d.rule}, ""
        ab = self.policy.resolve(self._abs(path), str(self.cwd_of()))
        return None, ab

    def _done(self, tool: str, args: str, t0: float) -> None:
        self.policy.log(PO.Decision(True, "", f"profile:{self.policy.access.profile(self.policy.project).name}"), tool, args, time.monotonic() - t0)

    def read(self, path: str, offset: int = 1, limit: int = 2000) -> dict[str, Any]:
        t0 = time.monotonic()
        den, ab = self._gate("read", path, False)
        if den:
            return den
        p = Path(ab)
        if not p.is_file():
            return {"ok": False, "error": f"not a file: {path}"}
        if p.stat().st_size > MAX_READ_BYTES:
            return {"ok": False, "error": "file larger than 2 MB: use grep or a line range with a smaller file"}
        lines = p.read_bytes().decode("utf-8", errors="replace").splitlines()
        lo = max(1, int(offset))
        part = lines[lo - 1: lo - 1 + max(1, int(limit))]
        self._done("read", path, t0)
        return {"ok": True, "text": "\n".join(f"{lo + i}\t{ln}" for i, ln in enumerate(part)), "total_lines": len(lines)}

    def write(self, path: str, content: str) -> dict[str, Any]:
        t0 = time.monotonic()
        den, ab = self._gate("write", path, True)
        if den:
            return den
        p = Path(ab)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))                 # binary write: LF, never the Windows CRLF default
        self._done("write", path, t0)
        return {"ok": True, "bytes": p.stat().st_size}

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> dict[str, Any]:
        t0 = time.monotonic()
        den, ab = self._gate("edit", path, True)
        if den:
            return den
        p = Path(ab)
        if not p.is_file():
            return {"ok": False, "error": f"not a file: {path}"}
        text = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in text
        body = text.replace("\r\n", "\n")
        if not old:
            return {"ok": False, "error": "old string is empty"}
        n = body.count(old)
        if n == 0:
            return {"ok": False, "error": "old string not found"}
        if n > 1 and not replace_all:
            return {"ok": False, "error": f"old string is ambiguous ({n} matches): add context or set replace_all"}
        out = body.replace(old, new) if replace_all else body.replace(old, new, 1)
        p.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
        self._done("edit", path, t0)
        return {"ok": True, "replaced": n if replace_all else 1}

    def _walk(self, root: str):                                                       # type: ignore[no-untyped-def]
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "__pycache__", ".venv", "node_modules", ".pytest_cache")]
            for f in fns:
                yield os.path.join(dp, f)

    def grep(self, pattern: str, path: str = ".", glob: Optional[str] = None, ignore_case: bool = False, max_matches: int = 200) -> dict[str, Any]:
        t0 = time.monotonic()
        den, ab = self._gate("grep", path, False)
        if den:
            return den
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as e:
            return {"ok": False, "error": f"bad regex: {e}"}
        files = [ab] if os.path.isfile(ab) else self._walk(ab)
        out, skipped = [], 0
        for f in files:
            if glob and not PO.glob_match(os.path.relpath(f, ab).replace("\\", "/"), [glob if "/" in glob else "**/" + glob]):
                continue
            if not self.policy.check_abs(os.path.normcase(os.path.realpath(f)), False, "files"):
                skipped += 1
                continue
            try:
                if os.path.getsize(f) > MAX_READ_BYTES:
                    continue
                data = Path(f).read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:2048]:
                continue
            for i, ln in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(ln):
                    out.append(f"{os.path.relpath(f, ab) if os.path.isdir(ab) else f}:{i}:{ln[:300]}")
                    if len(out) >= max_matches:
                        self._done("grep", pattern, t0)
                        return {"ok": True, "matches": out, "truncated": True, "skipped_denied": skipped}
        self._done("grep", pattern, t0)
        return {"ok": True, "matches": out, "truncated": False, "skipped_denied": skipped}

    def glob(self, pattern: str, path: str = ".", max_results: int = 500) -> dict[str, Any]:
        t0 = time.monotonic()
        den, ab = self._gate("glob", path, False)
        if den:
            return den
        rx = PO._glob_re(pattern if "/" in pattern else "**/" + pattern)
        out = []
        for f in self._walk(ab):
            rel = os.path.relpath(f, ab).replace("\\", "/")
            if rx.match(rel) and self.policy.check_abs(os.path.normcase(os.path.realpath(f)), False, "files"):
                out.append(rel)
                if len(out) >= max_results:
                    break
        self._done("glob", pattern, t0)
        return {"ok": True, "files": sorted(out)}
