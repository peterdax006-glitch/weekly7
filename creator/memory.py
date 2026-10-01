"""K12 memory: development history by kind (failure, repair, research, architecture,
strategy, resource) and "seen this before?" retrieval. Memory records what happened;
it never stores or returns a hidden answer key.

Pure standard library.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

KINDS = ("failure", "repair", "research", "architecture", "strategy", "resource")

_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(str(text).lower()))


@dataclass(frozen=True)
class MemoryEntry:
    """One remembered fact: a kind from KINDS, a short subject, free-text detail and tags."""
    kind: str
    subject: str
    detail: str = ""
    tags: tuple = ()
    seq: int = 0

    def to_dict(self) -> dict:
        """Plain-dict (JSON-serialisable) form of the entry."""
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


@dataclass
class Memory:
    """An append-only collection of MemoryEntry records with similarity retrieval."""
    entries: list = field(default_factory=list)

    def remember(self, kind: str, subject: str, detail: str = "", tags=()) -> MemoryEntry:
        """Append an entry and return it (seq is its 1-based position).

        Raises ValueError if kind is not in KINDS or subject is blank.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind: {kind!r}")
        if not str(subject).strip():
            raise ValueError("subject must be non-empty")
        entry = MemoryEntry(kind, str(subject).strip(), str(detail),
                            tuple(str(t) for t in (tags or ())), len(self.entries) + 1)
        self.entries.append(entry)
        return entry

    def by_kind(self, kind: str) -> list[MemoryEntry]:
        """All entries of the given kind in insertion order ([] if none or kind unknown)."""
        return [e for e in self.entries if e.kind == kind]

    def seen_before(self, query: str, kind: str | None = None, limit: int = 5,
                    min_score: float = 0.2) -> list[tuple[float, MemoryEntry]]:
        """Return up to `limit` (score, entry) pairs, best first, whose token overlap
        (Jaccard over subject, detail and tags) with `query` is >= min_score.

        Optionally restricted to one kind. An empty query, empty memory or limit <= 0
        yields []. Ties are broken by most recent entry first.
        """
        q = _tokens(query)
        if not q or limit <= 0:
            return []
        scored = []
        for e in self.entries:
            if kind is not None and e.kind != kind:
                continue
            t = _tokens(" ".join([e.subject, e.detail, *e.tags]))
            if not t:
                continue
            score = len(q & t) / len(q | t)
            if score >= min_score:
                scored.append((score, e))
        scored.sort(key=lambda p: (-p[0], -p[1].seq))
        return scored[:limit]

    def save(self, path) -> None:
        """Write all entries to `path` as JSON lines, creating parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")


def load(path) -> Memory:
    """Load a Memory from a JSON-lines file written by Memory.save.

    A missing file gives an empty Memory; blank or corrupt lines and entries with an
    invalid kind are skipped rather than raising.
    """
    mem = Memory()
    p = Path(path)
    if not p.is_file():
        return mem
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            mem.remember(d["kind"], d["subject"], d.get("detail", ""), d.get("tags", ()))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return mem
