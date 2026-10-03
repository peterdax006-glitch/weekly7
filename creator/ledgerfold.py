"""Incremental ledger fold (h38 speed, 3 Oct 2026). Loaded on demand by creator.ledger.

Opening the ledger re-verified and re-folded every line (1.2-1.4 s for 3,560 records, linear in history; the swarm opens it several
times per round and every append by another process re-folded it all). Folding is a deterministic left fold over the lines: line n's
checks depend only on its own bytes, n, and the view folded from the lines before it. So when the first n lines are BYTE-IDENTICAL
(sha256 of their text) to lines this process already folded and fully verified with the same checking code, the view after them is the
same, and only the lines after n need folding - with every check Ledger._fold makes. Anything else (shorter file, a changed byte, a
different checking function) folds from line 0. Ledger.verify() never uses this: it re-reads and re-checks the whole chain."""
from __future__ import annotations

import hashlib
from typing import Any

from creator import ledger as L
from creator import model as M

MAX_PATHS = 8
_CACHE: dict[str, tuple[int, str, Any, L.View]] = {}       # path -> (lines folded, sha256 of their text, checker identity, view)


def _sha(lines: list[str], n: int) -> str:
    h = hashlib.sha256()
    for line in lines[:n]:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _checker() -> tuple[int, ...]:
    """Identity of the code a fold runs: a replaced checking function (a test's monkeypatch) never reuses an older fold."""
    return (id(L.Ledger._fold), id(L._line_hash), id(L.record_id), id(M.record_from_dict), id(L.View.apply))


def copy_view(v: L.View) -> L.View:
    """Independent containers (Ledger.append and View.apply mutate them); Entry objects are frozen and shared."""
    return L.View(entries=list(v.entries), by_id=dict(v.by_id), status=dict(v.status),
                  status_history={k: list(x) for k, x in v.status_history.items()}, unique=dict(v.unique),
                  children={k: list(x) for k, x in v.children.items()})


def fold(led: L.Ledger, lines: list[str]) -> L.View:
    key = str(led.path.resolve())
    hit = _CACHE.get(key)
    chk = _checker()
    start, view = 0, None
    if hit is not None and hit[2] == chk and 0 < hit[0] <= len(lines) and _sha(lines, hit[0]) == hit[1]:
        start, view = hit[0], copy_view(hit[3])
    out = led._fold(lines, view, start)                    # raises LedgerError on any bad new line: nothing is cached then
    _CACHE.pop(key, None)
    if len(_CACHE) >= MAX_PATHS:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (len(lines), _sha(lines, len(lines)), chk, copy_view(out))
    return out


def clear() -> None:
    _CACHE.clear()
