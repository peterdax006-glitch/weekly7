"""Nupen tools - patch applier. Accepts SEARCH/REPLACE blocks and unified diffs, tolerant to whitespace, ATOMIC (every hunk of every file
applies or nothing is written), REVERSIBLE (returns an undo record) and GATED (every path goes through the file policy of FileTools).

SEARCH/REPLACE block (path on the line above the marker; the path may be omitted when `default_path` is given):
    creator/x.py
    <<<<<<< SEARCH
    old lines
    =======
    new lines
    >>>>>>> REPLACE
Unified diff: `--- a/p` / `+++ b/p` headers and `@@ -l,c +l,c @@` hunks. Line numbers are only a HINT (nearest match wins), the counts
are ignored (model-written diffs get them wrong); `--- /dev/null` creates a file.

Matching ladder per hunk: exact, trailing whitespace ignored, all indentation ignored (the replacement is re-indented by the same
offset), inner whitespace runs collapsed. Zero or several matches (without a unambiguous nearest hint) fail the WHOLE patch.

Write protocol: new contents of every file are computed in memory, written to temp files next to the targets, an undo journal is
written (atomically), then the temps are os.replace()d one by one; any exception during that phase restores the files already
replaced. A hard kill in that phase leaves the journal; the next apply (or recover()) rolls it back first."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

_replace = os.replace                                      # indirection so tests can inject a fault between two file replacements
JOURNAL = "patch_journal.json"
_FENCE = re.compile(r"^\s*`{3,}\S*\s*$")


class PatchError(Exception):
    def __init__(self, msg: str, path: str = "", hunk: int = 0) -> None:
        super().__init__(msg)
        self.path, self.hunk = path, hunk


# ----------------------------------------------------------------------------------------------------------------- parsing
Hunk = tuple[str, list[str], list[str], Optional[int]]          # path, old lines, new lines, line hint (1-based) or None


def parse(text: str, default_path: str = "") -> list[Hunk]:
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if any(re.match(r"^<{5,9} SEARCH\s*$", ln) for ln in lines):
        return _parse_sr(lines, default_path)
    if any(ln.startswith("@@") for ln in lines):
        return _parse_unified(lines, default_path)
    raise PatchError("no SEARCH/REPLACE block and no unified-diff hunk found")


def _parse_sr(lines: list[str], default_path: str) -> list[Hunk]:
    out: list[Hunk] = []
    i, path = 0, default_path
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^<{5,9} SEARCH\s*$", ln):
            j = i + 1
            old: list[str] = []
            while j < len(lines) and not re.match(r"^={5,9}\s*$", lines[j]):
                old.append(lines[j])
                j += 1
            if j >= len(lines):
                raise PatchError("SEARCH block without ======= divider", path, len(out) + 1)
            k = j + 1
            new: list[str] = []
            while k < len(lines) and not re.match(r"^>{5,9} REPLACE\s*$", lines[k]):
                new.append(lines[k])
                k += 1
            if k >= len(lines):
                raise PatchError("REPLACE block without >>>>>>> REPLACE terminator", path, len(out) + 1)
            if not path:
                raise PatchError("block has no file path", "", len(out) + 1)
            out.append((path, old, new, None))
            i = k + 1
            continue
        s = ln.strip().strip("`").strip()
        if s and not _FENCE.match(ln) and not ln.startswith(("<<<", "===", ">>>")):
            path = s
        i += 1
    if not out:
        raise PatchError("no complete SEARCH/REPLACE block")
    return out


def _clean_path(p: str) -> str:
    p = p.split("\t")[0].strip().strip('"')
    if p == "/dev/null":
        return p
    return p[2:] if p.startswith(("a/", "b/")) else p


def _parse_unified(lines: list[str], default_path: str) -> list[Hunk]:
    out: list[Hunk] = []
    path, i = default_path, 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old_p, new_p = _clean_path(ln[4:]), _clean_path(lines[i + 1][4:])
            path = new_p if new_p != "/dev/null" else old_p
            if old_p == "/dev/null":
                path = new_p                                  # creation: the hunks have no context to match
            i += 2
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", ln)
        if not m:
            i += 1
            continue
        if not path:
            raise PatchError("hunk has no file path", "", len(out) + 1)
        old: list[str] = []
        new: list[str] = []
        i += 1
        while i < n:
            b = lines[i]
            if b.startswith("@@") or (b.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ ")):
                break
            if b.startswith("\\"):
                i += 1
                continue
            if b.startswith("+"):
                new.append(b[1:])
            elif b.startswith("-"):
                old.append(b[1:])
            elif b.startswith(" "):
                old.append(b[1:])
                new.append(b[1:])
            elif b == "":
                if i == n - 1:                                # the trailing newline of the patch text, not a context line
                    break
                old.append("")
                new.append("")
            else:
                break
            i += 1
        out.append((path, old, new, int(m.group(1))))
    if not out:
        raise PatchError("no usable unified-diff hunk")
    return out


# ---------------------------------------------------------------------------------------------------------------- matching
_NORMS = (
    lambda s: s,
    lambda s: s.rstrip(),
    lambda s: s.strip(),
    lambda s: " ".join(s.split()),
)


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _candidates(lines: list[str], old: list[str], norm: Any) -> list[int]:
    k = len(old)
    if k == 0 or k > len(lines):
        return []
    want = [norm(x) for x in old]
    have = [norm(x) for x in lines]
    return [i for i in range(len(lines) - k + 1) if have[i:i + k] == want]


def _reindent(old: list[str], new: list[str], matched: list[str]) -> list[str]:
    o = next((x for x in old if x.strip()), None)
    f = next((x for x in matched if x.strip()), None)
    if o is None or f is None:
        return new
    oi, fi = _indent(o), _indent(f)
    if oi == fi:
        return new
    if fi.startswith(oi):
        add = fi[len(oi):]
        return [add + x if x.strip() else x for x in new]
    if oi.startswith(fi):
        cut = oi[len(fi):]
        return [x[len(cut):] if x.startswith(cut) else x for x in new]
    return new


def apply_hunk(lines: list[str], hunk: Hunk, idx: int = 1) -> tuple[list[str], bool]:
    """Apply one hunk to a list of lines. Returns (new lines, fuzzy?). Raises PatchError."""
    path, old, new, hint = hunk
    if not any(x.strip() for x in old):
        if not any(x.strip() for x in lines):
            return (list(new) + [""]) if new else list(lines), False
        raise PatchError("empty SEARCH on a non-empty file: give context lines", path, idx)
    for level, norm in enumerate(_NORMS):
        cands = _candidates(lines, old, norm)
        if not cands:
            continue
        if len(cands) > 1:
            if hint is None:
                raise PatchError(f"SEARCH text is ambiguous ({len(cands)} matches): add context lines", path, idx)
            best = sorted(cands, key=lambda c: abs(c + 1 - hint))
            if abs(best[0] + 1 - hint) == abs(best[1] + 1 - hint):
                raise PatchError(f"hunk is ambiguous ({len(cands)} equally near matches)", path, idx)
            cands = best[:1]
        at = cands[0]
        repl = new if level < 2 else _reindent(old, new, lines[at:at + len(old)])
        return lines[:at] + list(repl) + lines[at + len(old):], level > 0
    raise PatchError("SEARCH text not found" + (f" near line {hint}" if hint else ""), path, idx)


# ------------------------------------------------------------------------------------------------------------------ writing
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_tmp(path: Path, data: bytes) -> Path:
    tmp = path.with_name(f".{path.name}.nupen{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return tmp


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_tmp(path, data)
    try:
        _replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _record(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": 1, "files": entries}


def undo(record: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Reverse an apply. Refuses a file that changed since the patch (after_sha) unless force. All-or-nothing check first."""
    ents = record.get("files", [])
    if not force:
        for e in ents:
            p = Path(e["path"])
            cur = _sha(p.read_bytes()) if p.is_file() else None
            if cur != e.get("after_sha"):
                return {"ok": False, "error": f"file changed since the patch: {e['path']}"}
    for e in ents:
        p = Path(e["path"])
        if e["before"] is None:
            if p.is_file():
                p.unlink()
        else:
            _atomic_write(p, base64.b64decode(e["before"]))
    return {"ok": True, "files": [e["path"] for e in ents]}


def recover(journal_dir: Path) -> bool:
    """Roll back a patch that was killed between file replacements. True when a journal was found and undone."""
    j = Path(journal_dir) / JOURNAL
    if not j.is_file():
        return False
    try:
        rec = json.loads(j.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        j.unlink()
        return False
    undo(rec, force=True)
    j.unlink()
    return True


def apply_to_files(hunks: list[Hunk], resolve: Any, journal_dir: Path) -> dict[str, Any]:
    """`resolve(path) -> (denial dict | None, absolute path)`. Pure of policy otherwise."""
    recover(journal_dir)
    by: dict[str, list[tuple[int, Hunk]]] = {}
    for n, h in enumerate(hunks, 1):
        by.setdefault(h[0], []).append((n, h))
    plan: dict[str, tuple[Path, Optional[bytes], bytes]] = {}
    fuzzy = 0
    try:
        for rel, hs in by.items():
            den, ab = resolve(rel)
            if den:
                return {**den, "file": rel}
            p = Path(ab)
            before: Optional[bytes] = p.read_bytes() if p.is_file() else None
            text = before.decode("utf-8") if before is not None else ""
            crlf = "\r\n" in text
            lines = text.replace("\r\n", "\n").split("\n") if before is not None else []
            for n, h in hs:
                lines, fz = apply_hunk(lines, h, n)
                fuzzy += fz
            body = "\n".join(lines)
            data = (body.replace("\n", "\r\n") if crlf else body).encode("utf-8")
            plan[rel] = (p, before, data)
    except PatchError as e:
        return {"ok": False, "error": str(e), "file": e.path, "hunk": e.hunk}
    except UnicodeDecodeError as e:
        return {"ok": False, "error": f"file is not valid UTF-8: {e}"}
    changed = {r: v for r, v in plan.items() if v[1] != v[2]}
    entries = [{"path": str(p), "before": None if b is None else base64.b64encode(b).decode("ascii"), "after_sha": _sha(d)}
               for p, b, d in changed.values()]
    rec = _record(entries)
    tmps: list[Path] = []
    done: list[tuple[Path, Optional[bytes]]] = []
    jpath = Path(journal_dir) / JOURNAL
    try:
        for p, _b, d in changed.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmps.append(_write_tmp(p, d))
        if entries:
            journal_dir = Path(journal_dir)
            journal_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(jpath, json.dumps(rec).encode("utf-8"))
        for (p, b, _d), t in zip(changed.values(), tmps):
            _replace(t, p)
            done.append((p, b))
    except BaseException:
        for p, b in reversed(done):
            try:
                if b is None:
                    p.unlink()
                else:
                    _atomic_write(p, b)
            except OSError:
                pass
        for t in tmps:
            try:
                if t.exists():
                    t.unlink()
            except OSError:
                pass
        try:
            jpath.unlink()
        except OSError:
            pass
        raise
    try:
        jpath.unlink()
    except OSError:
        pass
    return {"ok": True, "files": [str(p) for p, _b, _d in changed.values()], "hunks": len(hunks), "fuzzy": fuzzy, "undo": rec}
