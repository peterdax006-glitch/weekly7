"""Pending owner decisions: a file queue the engine / adopt cycle READS. Nothing here executes anything.

The phone server's POST /decision only calls record_answer(); an answer is data (approve | later) that the next engine cycle may act on
through its own gates. Files live in <runtime>/phone/decisions.jsonl (outside the repo), append-only: one 'pending' row per ask, one
'answer' row per owner tap (the first answer for an id wins).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

CHOICES = ("approve", "later")
FILE = "decisions.jsonl"
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_LOCK = threading.Lock()


def default_dir() -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir()) / "phone"


def valid_id(x: Any) -> bool:
    return isinstance(x, str) and bool(_ID.match(x))


def decision_key(token: str, decision_id: str) -> str:
    """Scoped key for one decision id: lets the phone approve THIS decision without ever holding the master token."""
    return hmac.new(token.encode(), b"decision:" + decision_id.encode(), hashlib.sha256).hexdigest()[:32]


def key_ok(token: str, decision_id: str, got: str) -> bool:
    return hmac.compare_digest(decision_key(token, decision_id).encode(), (got or "").encode())


def _rows(d: Path) -> list[dict[str, Any]]:
    f = d / FILE
    out: list[dict[str, Any]] = []
    try:
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict):
                out.append(r)
    except OSError:
        pass
    return out


def _append(d: Path, row: dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / FILE).open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, sort_keys=True) + "\n")


def ask(decision_id: str, title: str, d: Optional[Path] = None, now: Optional[float] = None) -> bool:
    """Register a decision as pending (idempotent). False if the id is invalid or already known."""
    d = d or default_dir()
    if not valid_id(decision_id):
        return False
    with _LOCK:
        if any(r.get("id") == decision_id for r in _rows(d)):
            return False
        _append(d, {"kind": "pending", "id": decision_id, "title": str(title)[:120], "at": time.time() if now is None else now})
    return True


def record_answer(decision_id: str, choice: str, d: Optional[Path] = None, now: Optional[float] = None) -> tuple[bool, str]:
    """Record the owner's tap. Returns (ok, reason). Only records; the first answer for an id wins; unknown ids are refused."""
    d = d or default_dir()
    if not valid_id(decision_id):
        return False, "bad id"
    if choice not in CHOICES:
        return False, "bad choice"
    with _LOCK:
        rows = _rows(d)
        if not any(r.get("kind") == "pending" and r.get("id") == decision_id for r in rows):
            return False, "unknown decision"
        if any(r.get("kind") == "answer" and r.get("id") == decision_id for r in rows):
            return False, "already answered"
        _append(d, {"kind": "answer", "id": decision_id, "choice": choice, "at": time.time() if now is None else now})
    return True, "recorded"


def pending(d: Optional[Path] = None) -> list[dict[str, Any]]:
    """Asked and not yet answered."""
    rows = _rows(d or default_dir())
    done = {r["id"] for r in rows if r.get("kind") == "answer"}
    return [r for r in rows if r.get("kind") == "pending" and r["id"] not in done]


def answers(d: Optional[Path] = None) -> dict[str, str]:
    """{decision id: 'approve' | 'later'} - what the engine / adopt cycle reads before it acts (through its own gates)."""
    return {r["id"]: r["choice"] for r in _rows(d or default_dir()) if r.get("kind") == "answer"}
