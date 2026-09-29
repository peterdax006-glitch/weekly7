"""Knowledge archive (contract C62 sections 49, 13; checklist B01-B15) - IMPLEMENTED, NOT VALIDATED.

Permanent, immutable, hash-chained raw-experience storage in ten layers (L0 raw ... L9 meta). Dynamic behaviour comes from
changing INFLUENCE (a new record), never from editing or deleting history: a record, once written, is fixed forever;
"changing" it means appending a `supersede` record, "retiring" it means appending an `influence` of 0. The chain storage
idea is engine/pattern_memory.py's (append-only jsonl, prev-hash links, incremental sync, torn-tail refusal, cross-process
lock from engine.pattern_bank.file_lock) generalised into `ChainFile`, which knowledge_graph.py / contradiction.py reuse.

Time (C56/C58): every record carries `occurred_at` (when it happened) and `matured_at` (when it became knowable - an
outcome that matures ON `now` is not yet known). A view at `now` exposes a record only if matured_at < now AND its
provenance could exist then AND every parent is itself visible (a pattern cannot be visible before its evidence). `audit`
recomputes visibility a second, independent way and proves chain, content, parent ordering and abstraction hygiene.

Abstract layers (L4+) must be identity-free: no ticker keys, no dates, no caller-listed identities (sections 28-29).
Indices: retrieval (token), context, temporal, reliability, failure, contradiction, recovery - all filter by `now`."""
from __future__ import annotations

import bisect
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import (FailureCause, FirewallBreach, KnowledgeLike, Layer, Provenance, as_date, canonical_json,
                   current_code_hash, stable_hash)

SCHEMA = 1
GENESIS = "0" * 64
MAX_PAYLOAD_BYTES = 256_000
MAX_DEPTH = 12
FORBIDDEN_KEYS = frozenset({"ticker", "tickers", "symbol", "symbols", "cusip", "isin", "permno", "name", "company",
                            "date", "dates", "year", "years", "timestamp"})
ABSTRACT_LAYERS = frozenset({Layer.L4_HYPOTHESIS, Layer.L5_PATTERN, Layer.L6_CONTEXT_RULE, Layer.L7_VALIDATED,
                             Layer.L8_POLICY})
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR = re.compile(r"(?<![A-Za-z0-9_.])(?:19|20)\d{2}(?![A-Za-z0-9_])")
_TOKEN = re.compile(r"[a-z0-9_]{2,}")
_ALL = frozenset(Layer)
LAYER_ORDER = {l: i for i, l in enumerate(Layer)}

# which layers may hold which record kind (section 49 layer semantics)
KIND_LAYERS: dict[str, frozenset] = {
    "observation": frozenset({Layer.L0_RAW}),
    "event": frozenset({Layer.L1_EVENT}),
    "episode": frozenset({Layer.L2_EPISODE}),
    "decision": frozenset({Layer.L2_EPISODE}),
    "outcome": frozenset({Layer.L2_EPISODE}),
    "situation": frozenset({Layer.L3_SITUATION}),
    "hypothesis": frozenset({Layer.L4_HYPOTHESIS}),
    "experiment": frozenset({Layer.L4_HYPOTHESIS}),
    "pattern": frozenset({Layer.L5_PATTERN}),
    "context_rule": frozenset({Layer.L6_CONTEXT_RULE}),
    "knowledge": frozenset({Layer.L7_VALIDATED}),
    "policy": frozenset({Layer.L8_POLICY}),
    "meta": frozenset({Layer.L9_META}),
    "failure": frozenset({Layer.L2_EPISODE, Layer.L4_HYPOTHESIS, Layer.L5_PATTERN, Layer.L6_CONTEXT_RULE,
                          Layer.L7_VALIDATED}),
    "reliability": frozenset({Layer.L5_PATTERN, Layer.L6_CONTEXT_RULE, Layer.L7_VALIDATED}),
    "contradiction": frozenset({Layer.L6_CONTEXT_RULE, Layer.L7_VALIDATED}),
    "recovery": frozenset({Layer.L2_EPISODE, Layer.L6_CONTEXT_RULE, Layer.L7_VALIDATED}),
    "influence": _ALL,
    "supersede": _ALL,
    "note": _ALL,
    "snapshot": frozenset({Layer.L9_META}),
}
KINDS_NEEDING_SUBJECT = frozenset({"failure", "reliability", "contradiction", "recovery", "influence", "knowledge"})
KINDS_NEEDING_PARENTS = frozenset({"episode", "situation", "supersede"})
STRUCTURAL_KINDS = frozenset({"influence", "supersede", "snapshot"})


class ArchiveError(RuntimeError):
    """A write was refused or the archive is inconsistent. Never caught-and-continued by callers."""


class ChainCorrupt(ArchiveError):
    """The hash chain failed verification: history was edited, truncated or mangled."""


class ImmutabilityViolation(ArchiveError):
    """Someone tried to change or remove a record. History is append-only (section 49)."""


# ------------------------------------------------------------------------------------------- chain storage

class ChainFile:
    """Append-only, hash-chained line store. Each line: {seq, prev, body, hash}; hash = sha256(prev + canonical(body)).
    `path=None` keeps the chain in memory (tests, sandboxes). Opening verifies every link and fails closed. A torn final
    line (a writer died mid-write) refuses further appends until repaired: we never guess at half a record."""

    def __init__(self, path: str | os.PathLike | None = None, lock_timeout: float = 30.0):
        self.path = os.fspath(path) if path is not None else None
        self.lock_timeout = lock_timeout
        self._lines: list[dict] = []
        self._offset = 0
        self._last = GENESIS
        self._consumed = 0
        self.torn_tail = False
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.sync()

    @staticmethod
    def link_hash(prev: str, body: Any) -> str:
        return hashlib.sha256((prev + canonical_json(body)).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def head(self) -> str:
        return self._last

    def sync(self) -> int:
        """Absorb lines other writers appended since our offset; returns how many were new."""
        if not self.path or not os.path.exists(self.path):
            return 0
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        if not data:
            return 0
        pieces = data.split(b"\n")
        tail = pieces.pop()
        self.torn_tail = bool(tail)
        added = 0
        for raw in pieces:
            if not raw.strip():
                self._offset += len(raw) + 1
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
                seq, prev, body, h = rec["seq"], rec["prev"], rec["body"], rec["hash"]
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                raise ChainCorrupt(f"unreadable line at seq {len(self._lines)}: {e}") from e
            if seq != len(self._lines) or prev != self._last or self.link_hash(prev, body) != h:
                raise ChainCorrupt(f"chain broken at seq {seq}")
            self._lines.append(rec)
            self._last = h
            self._offset += len(raw) + 1
            added += 1
        return added

    def append_many(self, bodies: Sequence[Mapping]) -> list[dict]:
        """Append bodies atomically (one lock, one write, one fsync). Returns the new lines."""
        if not bodies:
            return []
        if self.path:
            from engine.pattern_bank import file_lock
            with file_lock(self.path + ".lock", self.lock_timeout):
                self.sync()
                if self.torn_tail:
                    raise ChainCorrupt("torn tail: a writer died mid-line; repair the file before appending")
                new = self._extend(bodies)
                with open(self.path, "ab") as fh:
                    fh.write(("\n".join(canonical_json(x) for x in new) + "\n").encode("utf-8"))
                    fh.flush()
                    os.fsync(fh.fileno())
                self._offset = os.path.getsize(self.path)
                return new
        return self._extend(bodies)

    def _extend(self, bodies: Sequence[Mapping]) -> list[dict]:
        new = []
        for b in bodies:
            body = json.loads(canonical_json(b))            # round-trip so memory and disk hold the same numbers
            rec = {"seq": len(self._lines), "prev": self._last, "body": body}
            rec["hash"] = self.link_hash(self._last, body)
            self._lines.append(rec)
            self._last = rec["hash"]
            new.append(rec)
        return new

    def take_new(self) -> list[dict]:
        """Lines not yet handed to the owner (own appends plus other writers')."""
        out = self._lines[self._consumed:]
        self._consumed = len(self._lines)
        return out

    def lines(self) -> list[dict]:
        return list(self._lines)

    def verify(self) -> dict:
        """Re-verify from scratch (from disk when persistent) independent of the in-memory copy."""
        prev, n = GENESIS, 0
        if self.path:
            raw = open(self.path, "rb").read() if os.path.exists(self.path) else b""
            rows = []
            for line in raw.split(b"\n")[:-1]:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    return {"ok": False, "records": n, "first_bad_seq": n, "head": prev}
        else:
            rows = self._lines
        for rec in rows:
            try:
                ok = rec["seq"] == n and rec["prev"] == prev and self.link_hash(prev, rec["body"]) == rec["hash"]
            except KeyError:
                ok = False
            if not ok:
                return {"ok": False, "records": n, "first_bad_seq": n, "head": prev}
            prev, n = rec["hash"], n + 1
        return {"ok": True, "records": n, "first_bad_seq": None, "head": prev}


# ------------------------------------------------------------------------------------------- records

def _iso(x) -> str:
    return as_date(x).isoformat()


def _check_payload(obj: Any, path: str = "payload", depth: int = 0) -> list[str]:
    """JSON-able, finite, shallow, string-keyed. NaN/inf never enter memory (they would be silently coerced)."""
    errs: list[str] = []
    if depth > MAX_DEPTH:
        return [f"{path}: nesting deeper than {MAX_DEPTH}"]
    if obj is None or isinstance(obj, (str, bool, int)):
        return errs
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (list, tuple, dict)):
        obj = obj.item()
    if isinstance(obj, float):
        return errs if math.isfinite(obj) else [f"{path}: non-finite number {obj}"]
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                errs.append(f"{path}: non-string key {k!r}")
            else:
                errs += _check_payload(v, f"{path}.{k}", depth + 1)
        return errs
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            errs += _check_payload(v, f"{path}[{i}]", depth + 1)
        return errs
    return [f"{path}: unsupported type {type(obj).__name__}"]


def _identity_errors(obj: Any, forbidden: frozenset, path: str = "payload") -> list[str]:
    """L4+ knowledge is about conditions, never about a name or a date (sections 28-29)."""
    errs: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                errs.append(f"{path}.{k}: identity/date key in an abstract layer")
            errs += _identity_errors(v, forbidden, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            errs += _identity_errors(v, forbidden, f"{path}[{i}]")
    elif isinstance(obj, str):
        if _ISO.search(obj) or _YEAR.search(obj):
            errs.append(f"{path}: contains a date/year")
        toks = {t.upper() for t in re.findall(r"[A-Za-z0-9_.\-]+", obj)}
        hit = toks & forbidden
        if hit:
            errs.append(f"{path}: contains forbidden identity {sorted(hit)[0]}")
    return errs


@dataclasses.dataclass(frozen=True)
class ArchiveRecord:
    """One immutable fact. `payload_json` (canonical text) rather than a dict so the object is truly frozen."""
    seq: int
    rec_id: str
    layer: Layer
    kind: str
    subject: str
    occurred_at: str
    matured_at: str
    recorded_real: str
    contexts: tuple[tuple[str, str], ...]
    parents: tuple[str, ...]
    payload_json: str
    provenance: Provenance
    prev: str
    hash: str

    @property
    def payload(self) -> dict:
        return json.loads(self.payload_json)

    @property
    def context_dict(self) -> dict[str, str]:
        return dict(self.contexts)

    def timing_ok(self, now) -> bool:
        """Own-timing half of visibility: matured strictly before `now`, and the provenance could exist then."""
        n = as_date(now)
        return as_date(self.matured_at) < n and self.provenance.could_exist_at(n)

    def content(self) -> dict:
        """What rec_id hashes: everything except position, wall-clock and chain links."""
        return {"layer": self.layer.value, "kind": self.kind, "subject": self.subject, "occurred_at": self.occurred_at,
                "matured_at": self.matured_at, "contexts": [list(c) for c in self.contexts],
                "parents": list(self.parents), "payload": json.loads(self.payload_json),
                "provenance": dataclasses.asdict(self.provenance)}

    def body(self) -> dict:
        return {**self.content(), "rec_id": self.rec_id, "recorded_real": self.recorded_real, "schema": SCHEMA}

    @classmethod
    def from_line(cls, line: Mapping) -> "ArchiveRecord":
        b = line["body"]
        prov = dict(b["provenance"])
        for f in ("sealed_windows", "parents"):
            prov[f] = tuple(prov.get(f) or ())
        return cls(seq=int(line["seq"]), rec_id=b["rec_id"], layer=Layer(b["layer"]), kind=b["kind"],
                   subject=b["subject"], occurred_at=b["occurred_at"], matured_at=b["matured_at"],
                   recorded_real=b["recorded_real"], contexts=tuple((str(k), str(v)) for k, v in b["contexts"]),
                   parents=tuple(b["parents"]), payload_json=canonical_json(b["payload"]),
                   provenance=Provenance(**prov), prev=line["prev"], hash=line["hash"])

    def recompute_id(self) -> str:
        return stable_hash(self.content(), 24)


# ------------------------------------------------------------------------------------------- kind validators

def _need(payload: Mapping, name: str, kinds: tuple, errs: list, where: str):
    if name not in payload:
        errs.append(f"{where}: payload.{name} required")
    elif not isinstance(payload[name], kinds) or isinstance(payload[name], bool) and bool not in kinds:
        errs.append(f"{where}: payload.{name} must be {'/'.join(k.__name__ for k in kinds)}")


def _validate_kind(kind: str, payload: Mapping, subject: str, parents: Sequence[str]) -> list[str]:
    errs: list[str] = []
    w = kind
    if kind == "failure":
        _need(payload, "cause", (str,), errs, w)
        if isinstance(payload.get("cause"), str) and payload["cause"] not in {c.value for c in FailureCause}:
            errs.append(f"failure: unknown cause {payload['cause']!r}")
    elif kind == "reliability":
        _need(payload, "value", (int, float), errs, w)
        _need(payload, "n", (int,), errs, w)
        v = payload.get("value")
        if isinstance(v, (int, float)) and not (0.0 <= float(v) <= 1.0):
            errs.append("reliability: value outside [0,1]")
        if isinstance(payload.get("n"), int) and payload["n"] < 0:
            errs.append("reliability: negative n")
    elif kind == "contradiction":
        _need(payload, "other", (str,), errs, w)
        if payload.get("other") == subject:
            errs.append("contradiction: a subject cannot contradict itself")
    elif kind == "recovery":
        _need(payload, "condition", (str, dict), errs, w)
    elif kind == "influence":
        _need(payload, "target", (str,), errs, w)
        _need(payload, "weight", (int, float), errs, w)
        _need(payload, "reason", (str,), errs, w)
        wt = payload.get("weight")
        if isinstance(wt, (int, float)) and not (0.0 <= float(wt) <= 1.0):
            errs.append("influence: weight outside [0,1]")
        if isinstance(payload.get("reason"), str) and not payload["reason"].strip():
            errs.append("influence: a reason is required (history may only fade for a stated reason)")
    elif kind == "supersede":
        _need(payload, "reason", (str,), errs, w)
        if len(parents) != 1:
            errs.append("supersede: exactly one parent (the record being superseded)")
    elif kind == "situation":
        _need(payload, "features", (dict,), errs, w)
    elif kind == "knowledge":
        _need(payload, "knowledge_id", (str,), errs, w)
        _need(payload, "version", (int,), errs, w)
    elif kind == "hypothesis":
        _need(payload, "statement", (str,), errs, w)
    elif kind == "snapshot":
        _need(payload, "head_hash", (str,), errs, w)
    return errs


# ------------------------------------------------------------------------------------------- snapshots & audit

@dataclasses.dataclass(frozen=True)
class Snapshot:
    """A verifiable statement of 'this is what the archive contained and exposed at `now`' (B12)."""
    now: str
    head_seq: int
    head_hash: str
    digest: str                              # hash of the ids visible at `now` among the first head_seq+1 records
    n_visible: int
    layer_counts: tuple[tuple[str, int], ...]
    influence_digest: str
    label: str = ""

    @property
    def snapshot_id(self) -> str:
        return stable_hash(dataclasses.asdict(self), 16)


@dataclasses.dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: str                            # "error" fails the audit; "warn" is reported
    seq: int
    detail: str


@dataclasses.dataclass
class ArchiveAudit:
    now: str
    ok: bool
    records: int
    visible: int
    hidden_future: int
    findings: list[AuditFinding]
    head: str

    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "error"]

    def summary(self) -> str:
        c = Counter(f.code for f in self.findings)
        return (f"archive audit @ {self.now}: {'PASS' if self.ok else 'FAIL'} - {self.records} records, "
                f"{self.visible} visible, {self.hidden_future} hidden as future; findings={dict(c) or 'none'}")


# ------------------------------------------------------------------------------------------- indices

class ArchiveIndex:
    """Seven derived indices (B05-B11). They hold only seq numbers; every query is filtered by the archive's visible set
    for `now`, so an index can never surface a record a view would hide."""

    def __init__(self):
        self.tokens: dict[str, set[int]] = defaultdict(set)
        self.context: dict[tuple[str, str], set[int]] = defaultdict(set)
        self.temporal: list[tuple[int, int]] = []                 # (matured ordinal, seq) sorted
        self.reliability: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.failure: dict[str, list[int]] = defaultdict(list)    # subject -> seqs
        self.failure_cause: dict[str, list[int]] = defaultdict(list)
        self.contradiction: dict[str, list[int]] = defaultdict(list)
        self.recovery: dict[str, list[int]] = defaultdict(list)
        self.n = 0

    @staticmethod
    def tokenize(*texts: Any) -> set[str]:
        out: set[str] = set()
        for t in texts:
            if isinstance(t, str):
                out.update(_TOKEN.findall(t.lower()))
            elif isinstance(t, Mapping):
                for k, v in t.items():
                    out.update(_TOKEN.findall(str(k).lower()))
                    out |= ArchiveIndex.tokenize(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    out |= ArchiveIndex.tokenize(v)
        return out

    def add(self, r: ArchiveRecord):
        p = r.payload
        toks = self.tokenize(r.subject, r.kind, r.layer.value, dict(r.contexts), p.get("text"), p.get("tags"),
                             p.get("mechanism"), p.get("statement"), p.get("cause"))
        for t in toks:
            self.tokens[t].add(r.seq)
        for c in r.contexts:
            self.context[c].add(r.seq)
        bisect.insort(self.temporal, (as_date(r.matured_at).toordinal(), r.seq))
        if r.kind == "reliability":
            bisect.insort(self.reliability[r.subject], (as_date(r.matured_at).toordinal(), r.seq))
        elif r.kind == "failure":
            self.failure[r.subject].append(r.seq)
            self.failure_cause[p["cause"]].append(r.seq)
        elif r.kind == "contradiction":
            self.contradiction[r.subject].append(r.seq)
            self.contradiction[p["other"]].append(r.seq)
        elif r.kind == "recovery":
            self.recovery[r.subject].append(r.seq)
        self.n += 1


# ------------------------------------------------------------------------------------------- the archive

class Archive:
    """The permanent store. Open with a directory (persistent) or `root=None` (in memory)."""

    def __init__(self, root: str | os.PathLike | None = None, *, code_hash: str | None = None,
                 forbidden_identities: Iterable[str] = (), created_real: str | None = None):
        self.root = os.fspath(root) if root is not None else None
        path = os.path.join(self.root, "archive.jsonl") if self.root else None
        self._chain = ChainFile(path)
        self._code_hash = code_hash
        self.forbidden = frozenset(str(x).upper() for x in forbidden_identities)
        self._created_real = created_real
        self._recs: list[ArchiveRecord] = []
        self._by_id: dict[str, ArchiveRecord] = {}
        self.index = ArchiveIndex()
        self._vis_cache: dict[tuple, frozenset] = {}
        self._ingest()

    # ---- basics
    def _ingest(self):
        for line in self._chain.take_new():
            r = ArchiveRecord.from_line(line)
            if r.recompute_id() != r.rec_id:
                raise ChainCorrupt(f"content of record seq {r.seq} does not match its id")
            self._recs.append(r)
            self._by_id.setdefault(r.rec_id, r)
            self.index.add(r)
        self._vis_cache.clear()

    def refresh(self):
        """Pick up records other processes appended."""
        self._chain.sync()
        self._ingest()

    def __len__(self) -> int:
        return len(self._recs)

    def __contains__(self, rec_id: str) -> bool:
        return rec_id in self._by_id

    def get(self, rec_id: str) -> ArchiveRecord:
        try:
            return self._by_id[rec_id]
        except KeyError:
            raise KeyError(f"no archive record {rec_id}") from None

    def records(self) -> tuple[ArchiveRecord, ...]:
        return tuple(self._recs)

    @property
    def head(self) -> str:
        return self._chain.head

    def _default_provenance(self, matured_at: str) -> Provenance:
        if self._code_hash is None:
            self._code_hash = current_code_hash()
        created = self._created_real or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        return Provenance(created_real=created, learned_at=matured_at, code_hash=self._code_hash)

    # ---- writing
    def validate(self, *, layer, kind, payload, occurred_at, matured_at=None, subject="", contexts=None, parents=(),
                 provenance=None, now=None) -> list[str]:
        errs: list[str] = []
        try:
            layer = Layer.parse(layer)
        except ValueError:
            return [f"unknown layer {layer!r}"]
        if kind not in KIND_LAYERS:
            return [f"unknown kind {kind!r}"]
        if layer not in KIND_LAYERS[kind]:
            errs.append(f"kind {kind} is not allowed in {layer.value}")
        try:
            occ = as_date(occurred_at)
            mat = as_date(matured_at if matured_at is not None else occurred_at)
        except (ValueError, TypeError):
            return errs + ["occurred_at/matured_at are not dates"]
        if mat < occ:
            errs.append("matured_at precedes occurred_at (knowledge cannot precede the event)")
        if now is not None and mat >= as_date(now):
            errs.append(f"matured_at {mat} is not before the writer's now={as_date(now)}: not yet knowable")
        if kind in KINDS_NEEDING_SUBJECT and not subject:
            errs.append(f"kind {kind} needs a subject")
        if kind in KINDS_NEEDING_PARENTS and not parents:
            errs.append(f"kind {kind} needs at least one parent record")
        errs += _check_payload(payload)
        if not errs:
            if len(canonical_json(payload)) > MAX_PAYLOAD_BYTES:
                errs.append("payload too large")
            errs += _validate_kind(kind, payload, subject, parents)
        for k, v in dict(contexts or {}).items():
            if not isinstance(k, str) or not k:
                errs.append(f"context key {k!r} invalid")
            elif not isinstance(v, (str, int, float, bool)):
                errs.append(f"context {k}: value must be scalar")
        maxlayer = -1
        for pid in parents:
            if pid not in self._by_id:
                errs.append(f"parent {pid} does not exist")
                continue
            p = self._by_id[pid]
            maxlayer = max(maxlayer, LAYER_ORDER[p.layer])
            if as_date(p.matured_at) > mat and kind not in STRUCTURAL_KINDS:
                errs.append(f"parent {pid} matured after this record ({p.matured_at} > {mat}): time travel")
        if layer in ABSTRACT_LAYERS and kind not in STRUCTURAL_KINDS:
            errs += _identity_errors(payload, self.forbidden)
            for k, v in dict(contexts or {}).items():
                errs += _identity_errors({k: v}, self.forbidden, "contexts")
            if subject and (_ISO.search(subject) or _YEAR.search(subject)):
                errs.append("subject of an abstract record contains a date")
        if provenance is not None:
            errs += provenance.check()
            if as_date(provenance.learned_at) < mat and kind not in STRUCTURAL_KINDS:
                errs.append("provenance.learned_at precedes matured_at: evidence cannot be learned before it matures")
        return errs

    def append(self, *, layer, kind: str, payload: Mapping, occurred_at, matured_at=None, subject: str = "",
               contexts: Mapping | None = None, parents: Sequence[str] = (), provenance: Provenance | None = None,
               now=None) -> ArchiveRecord:
        """Write one record (idempotent: identical content returns the existing record; a rerun never inflates history)."""
        layer = Layer.parse(layer)
        mat = _iso(matured_at if matured_at is not None else occurred_at)
        errs = self.validate(layer=layer, kind=kind, payload=payload, occurred_at=occurred_at, matured_at=mat,
                             subject=subject, contexts=contexts, parents=tuple(parents), provenance=provenance, now=now)
        if errs:
            raise ArchiveError("; ".join(errs))
        prov = provenance or self._default_provenance(mat)
        ctx = tuple(sorted((str(k), str(v)) for k, v in dict(contexts or {}).items()))
        stub = ArchiveRecord(seq=-1, rec_id="", layer=layer, kind=kind, subject=subject, occurred_at=_iso(occurred_at),
                             matured_at=mat, recorded_real="", contexts=ctx, parents=tuple(parents),
                             payload_json=canonical_json(payload), provenance=prov, prev="", hash="")
        rid = stub.recompute_id()
        self.refresh()
        if rid in self._by_id:
            return self._by_id[rid]
        created = prov.created_real
        body = {**stub.content(), "rec_id": rid, "recorded_real": created, "schema": SCHEMA}
        self._chain.append_many([body])
        self._ingest()
        return self._by_id[rid]

    def append_many(self, specs: Sequence[Mapping]) -> list[ArchiveRecord]:
        """Write several records; each spec is the kwargs of `append`. Not atomic across specs (each is validated first)."""
        return [self.append(**s) for s in specs]

    def update(self, *_a, **_k):
        raise ImmutabilityViolation("records are immutable: append a `supersede` record instead")

    def delete(self, *_a, **_k):
        raise ImmutabilityViolation("history is never deleted: append an `influence` of 0 (retire, do not delete)")

    # ---- typed writers for the storage layers (B01-B04)
    def log_observation(self, subject: str, values: Mapping, occurred_at, matured_at=None, contexts=None, **kw):
        return self.append(layer=Layer.L0_RAW, kind="observation", payload=dict(values), occurred_at=occurred_at,
                           matured_at=matured_at, subject=subject, contexts=contexts, **kw)

    def log_event(self, subject: str, what: str, occurred_at, parents: Sequence[str] = (), matured_at=None,
                  contexts=None, **kw):
        return self.append(layer=Layer.L1_EVENT, kind="event", payload={"what": what}, occurred_at=occurred_at,
                           matured_at=matured_at, subject=subject, parents=parents, contexts=contexts, **kw)

    def log_episode(self, subject: str, events: Sequence[str], outcome: Mapping, occurred_at, matured_at=None,
                    contexts=None, **kw):
        return self.append(layer=Layer.L2_EPISODE, kind="episode", payload={"outcome": dict(outcome)},
                           occurred_at=occurred_at, matured_at=matured_at, subject=subject, parents=tuple(events),
                           contexts=contexts, **kw)

    def log_situation(self, subject: str, features: Mapping, episodes: Sequence[str], occurred_at, matured_at=None,
                      contexts=None, **kw):
        return self.append(layer=Layer.L3_SITUATION, kind="situation", payload={"features": dict(features)},
                           occurred_at=occurred_at, matured_at=matured_at, subject=subject, parents=tuple(episodes),
                           contexts=contexts, **kw)

    def log_failure(self, subject: str, cause: FailureCause | str, layer, occurred_at, matured_at=None, detail=None,
                    parents: Sequence[str] = (), contexts=None, **kw):
        return self.append(layer=layer, kind="failure", payload={"cause": FailureCause.parse(cause).value,
                                                                "detail": detail or {}},
                           occurred_at=occurred_at, matured_at=matured_at, subject=subject, parents=parents,
                           contexts=contexts, **kw)

    def log_reliability(self, subject: str, value: float, n: int, occurred_at, layer=Layer.L7_VALIDATED,
                        matured_at=None, contexts=None, **kw):
        return self.append(layer=layer, kind="reliability", payload={"value": float(value), "n": int(n)},
                           occurred_at=occurred_at, matured_at=matured_at, subject=subject, contexts=contexts, **kw)

    def log_contradiction(self, subject: str, other: str, occurred_at, matured_at=None, detail=None,
                          layer=Layer.L7_VALIDATED, contexts=None, **kw):
        return self.append(layer=layer, kind="contradiction", payload={"other": other, "detail": detail or {}},
                           occurred_at=occurred_at, matured_at=matured_at, subject=subject, contexts=contexts, **kw)

    def log_recovery(self, subject: str, condition, occurred_at, matured_at=None, layer=Layer.L7_VALIDATED,
                     contexts=None, **kw):
        return self.append(layer=layer, kind="recovery", payload={"condition": condition}, occurred_at=occurred_at,
                           matured_at=matured_at, subject=subject, contexts=contexts, **kw)

    def put_knowledge(self, k: KnowledgeLike, occurred_at, matured_at=None, parents: Sequence[str] = (), **kw):
        """Archive a versioned KnowledgeLike duck type as an L7 record (B04). The old version stays; the new one is
        another record, so the whole version history is auditable."""
        miss = KnowledgeLike.conforms(k)
        if miss:
            raise ArchiveError("not knowledge-like: " + ", ".join(miss))
        conf = k.confidence
        payload = {"knowledge_id": str(k.knowledge_id), "version": int(k.version), "epistemic": str(k.epistemic),
                   "lifecycle": str(k.lifecycle), "promotion": str(k.promotion),
                   "confidence": dataclasses.asdict(conf) if dataclasses.is_dataclass(conf) else dict(conf),
                   "contexts": {str(a): _plain(b) for a, b in dict(k.contexts).items()},
                   "anti_contexts": {str(a): _plain(b) for a, b in dict(k.anti_contexts).items()},
                   "decision_effect": [str(e) for e in k.decision_effect]}
        return self.append(layer=Layer.L7_VALIDATED, kind="knowledge", payload=payload, occurred_at=occurred_at,
                           matured_at=matured_at, subject=str(k.knowledge_id), parents=parents, provenance=k.provenance,
                           **kw)

    # ---- dynamic influence (B14)
    def set_influence(self, target: str, weight: float, reason: str, effective_at, **kw) -> ArchiveRecord:
        """Fade or restore something's say without touching history. `target` is a rec_id or a subject."""
        if target in self._by_id:
            tr = self._by_id[target]
            if as_date(effective_at) < as_date(tr.matured_at):
                raise ArchiveError("influence cannot pre-date the evidence it changes")
        return self.append(layer=Layer.L9_META, kind="influence", payload={"target": target, "weight": float(weight),
                                                                            "reason": reason},
                           occurred_at=effective_at, matured_at=effective_at, subject=target, **kw)

    def retire(self, target: str, reason: str, effective_at, **kw) -> ArchiveRecord:
        return self.set_influence(target, 0.0, reason, effective_at, **kw)

    def influence_at(self, target: str, now) -> float:
        """Weight in [0,1] at `now`: latest visible influence for the rec_id, else for its subject, else 1.0."""
        vis = self.visible_ids(now)
        subj = self._by_id[target].subject if target in self._by_id else None
        best: dict[str, tuple] = {}
        for r in self._recs:
            if r.kind != "influence" or r.rec_id not in vis:
                continue
            tgt = r.payload["target"]
            level = "rec" if tgt == target else "subject" if subj and tgt == subj else None
            if level is None and tgt == target:
                level = "rec"
            if level is None:
                continue
            key = (as_date(r.occurred_at).toordinal(), r.seq)
            if level not in best or key > best[level][0]:
                best[level] = (key, float(r.payload["weight"]))
        for level in ("rec", "subject"):
            if level in best:
                return best[level][1]
        return 1.0

    def influence_series(self, target: str, now) -> list[tuple[str, float, str]]:
        vis = self.visible_ids(now)
        rows = [(as_date(r.occurred_at), r.seq, r.payload) for r in self._recs
                if r.kind == "influence" and r.rec_id in vis and r.payload["target"] == target]
        rows.sort(key=lambda x: (x[0], x[1]))
        return [(d.isoformat(), float(p["weight"]), p["reason"]) for d, _, p in rows]

    def supersede(self, rec_id: str, replacement: str, reason: str, effective_at, **kw) -> ArchiveRecord:
        if replacement not in self._by_id:
            raise ArchiveError("replacement record does not exist")
        return self.append(layer=self.get(rec_id).layer, kind="supersede", payload={"replacement": replacement,
                                                                                   "reason": reason},
                           occurred_at=effective_at, matured_at=effective_at, parents=(rec_id,), subject=rec_id, **kw)

    def latest_version(self, rec_id: str, now) -> ArchiveRecord:
        """Follow visible supersede links to the newest record at `now`."""
        vis = self.visible_ids(now)
        nxt = {r.parents[0]: r.payload["replacement"] for r in self._recs if r.kind == "supersede" and r.rec_id in vis}
        seen, cur = {rec_id}, rec_id
        while cur in nxt and nxt[cur] not in seen:
            cur = nxt[cur]
            seen.add(cur)
        return self.get(cur)

    # ---- visibility & views
    def visible_ids(self, now, max_seq: int | None = None) -> frozenset:
        """Ids exposed at `now`: own timing ok AND all parents exposed (closure). `max_seq` = 'as known when the chain
        had only this many records' (used by snapshots)."""
        key = (as_date(now).toordinal(), max_seq, len(self._recs))
        hit = self._vis_cache.get(key)
        if hit is not None:
            return hit
        n = as_date(now)
        vis: set[str] = set()
        for r in self._recs:
            if max_seq is not None and r.seq > max_seq:
                break
            if r.timing_ok(n) and all(p in vis for p in r.parents):
                vis.add(r.rec_id)
        out = frozenset(vis)
        if len(self._vis_cache) > 16:
            self._vis_cache.clear()
        self._vis_cache[key] = out
        return out

    def view(self, now, *, layers: Iterable | None = None, kinds: Iterable[str] | None = None, subject: str | None = None,
             contexts: Mapping | None = None, max_seq: int | None = None, min_influence: float | None = None,
             ) -> tuple[ArchiveRecord, ...]:
        """The only sanctioned read. Everything returned could have existed and been known before `now`."""
        vis = self.visible_ids(now, max_seq)
        lset = {Layer.parse(x) for x in layers} if layers is not None else None
        kset = set(kinds) if kinds is not None else None
        out = []
        for r in self._recs:
            if r.rec_id not in vis or (lset and r.layer not in lset) or (kset and r.kind not in kset):
                continue
            if subject is not None and r.subject != subject:
                continue
            if contexts and not all(r.context_dict.get(k) == str(v) for k, v in contexts.items()):
                continue
            if min_influence is not None and r.kind not in STRUCTURAL_KINDS:
                if self.influence_at(r.rec_id, now) < min_influence:
                    continue
            out.append(r)
        return tuple(out)

    def active_view(self, now, **kw) -> tuple[ArchiveRecord, ...]:
        """Decision-facing read: visible records with non-zero influence that have not been superseded."""
        recs = self.view(now, min_influence=1e-12, **kw)
        vis = self.visible_ids(now)
        gone = {r.parents[0] for r in self._recs if r.kind == "supersede" and r.rec_id in vis}
        return tuple(r for r in recs if r.rec_id not in gone and r.kind not in STRUCTURAL_KINDS)

    def digest(self, now, max_seq: int | None = None) -> str:
        ids = sorted(i for i in self.visible_ids(now, max_seq) if self._by_id[i].kind != "snapshot")
        return stable_hash(ids, 24)

    def lineage(self, rec_id: str, now=None) -> list[ArchiveRecord]:
        """Ancestors (evidence chain), nearest first, optionally restricted to what is visible at `now`."""
        vis = self.visible_ids(now) if now is not None else None
        seen, out, frontier = {rec_id}, [], [rec_id]
        while frontier:
            nxt = []
            for cid in frontier:
                for pid in self.get(cid).parents:
                    if pid not in seen and (vis is None or pid in vis):
                        seen.add(pid)
                        out.append(self.get(pid))
                        nxt.append(pid)
            frontier = nxt
        return out

    def descendants(self, rec_id: str, now=None) -> list[ArchiveRecord]:
        vis = self.visible_ids(now) if now is not None else None
        kids: dict[str, list[str]] = defaultdict(list)
        for r in self._recs:
            for p in r.parents:
                kids[p].append(r.rec_id)
        seen, out, frontier = {rec_id}, [], [rec_id]
        while frontier:
            nxt = []
            for cid in frontier:
                for k in kids.get(cid, ()):
                    if k not in seen and (vis is None or k in vis):
                        seen.add(k)
                        out.append(self.get(k))
                        nxt.append(k)
            frontier = nxt
        return out

    def knowledge_versions(self, knowledge_id: str, now) -> list[ArchiveRecord]:
        return sorted(self.view(now, kinds=("knowledge",), subject=knowledge_id), key=lambda r: r.payload["version"])

    def latest_knowledge(self, knowledge_id: str, now) -> ArchiveRecord | None:
        v = self.knowledge_versions(knowledge_id, now)
        return v[-1] if v else None

    # ---- index queries (all filtered by `now`)
    def _seqs(self, seqs: Iterable[int], now) -> list[ArchiveRecord]:
        vis = self.visible_ids(now)
        return [r for r in (self._recs[s] for s in sorted(set(seqs))) if r.rec_id in vis]

    def search(self, text: str, now, limit: int = 20) -> list[tuple[ArchiveRecord, float]]:
        """Retrieval index (B05): tf-idf-ish token overlap, ranked; only visible records."""
        toks = ArchiveIndex.tokenize(text)
        vis = self.visible_ids(now)
        n = max(len(self._recs), 1)
        score: dict[int, float] = defaultdict(float)
        for t in toks:
            posting = self.index.tokens.get(t)
            if not posting:
                continue
            idf = math.log(1.0 + n / len(posting))
            for s in posting:
                score[s] += idf
        rows = [(self._recs[s], sc) for s, sc in score.items() if self._recs[s].rec_id in vis]
        rows.sort(key=lambda x: (-x[1], x[0].seq))
        return rows[:limit]

    def by_context(self, contexts: Mapping, now, mode: str = "all") -> list[ArchiveRecord]:
        """Context index (B06): records tagged with all (or any) of the given context values."""
        sets = [self.index.context.get((str(k), str(v)), set()) for k, v in contexts.items()]
        if not sets:
            return []
        seqs = set.intersection(*sets) if mode == "all" else set.union(*sets)
        return self._seqs(seqs, now)

    def context_values(self, dim: str, now) -> dict[str, int]:
        vis = self.visible_ids(now)
        out: Counter = Counter()
        for (d, v), seqs in self.index.context.items():
            if d == dim:
                out[v] = sum(1 for s in seqs if self._recs[s].rec_id in vis)
        return {k: c for k, c in sorted(out.items()) if c}

    def between(self, start, end, now) -> list[ArchiveRecord]:
        """Temporal index (B07): records that matured in [start, end) and are visible at `now`."""
        lo = bisect.bisect_left(self.index.temporal, (as_date(start).toordinal(), -1))
        hi = bisect.bisect_left(self.index.temporal, (as_date(end).toordinal(), -1))
        return self._seqs((s for _, s in self.index.temporal[lo:hi]), now)

    def reliability_trace(self, subject: str, now) -> list[tuple[str, float, int]]:
        """Reliability index (B08): the dated history of a subject's reliability, oldest first."""
        return [(r.matured_at, float(r.payload["value"]), int(r.payload["n"]))
                for r in self._seqs((s for _, s in self.index.reliability.get(subject, [])), now)]

    def reliability_at(self, subject: str, now) -> float | None:
        t = self.reliability_trace(subject, now)
        return t[-1][1] if t else None

    def reliability_trend(self, subject: str, now, k: int = 5) -> float | None:
        """Slope per record over the last k reliability points (negative = decaying); None if under 3 points."""
        t = self.reliability_trace(subject, now)[-k:]
        if len(t) < 3:
            return None
        return float(np.polyfit(np.arange(len(t)), [v for _, v, _ in t], 1)[0])

    def failures(self, now, subject: str | None = None, cause: FailureCause | str | None = None) -> list[ArchiveRecord]:
        """Failure index (B09)."""
        if subject is not None:
            seqs: Iterable[int] = self.index.failure.get(subject, [])
        elif cause is not None:
            seqs = self.index.failure_cause.get(FailureCause.parse(cause).value, [])
        else:
            seqs = [s for v in self.index.failure.values() for s in v]
        out = self._seqs(seqs, now)
        if cause is not None:
            c = FailureCause.parse(cause).value
            out = [r for r in out if r.payload["cause"] == c]
        return out

    def failure_counts(self, now) -> dict[str, int]:
        c = Counter(r.payload["cause"] for r in self.failures(now))
        return dict(sorted(c.items()))

    def contradictions_of(self, subject: str, now) -> list[ArchiveRecord]:
        """Contradiction index (B10)."""
        return self._seqs(self.index.contradiction.get(subject, []), now)

    def contradiction_pairs(self, now) -> list[tuple[str, str]]:
        vis = self.visible_ids(now)
        pairs = {tuple(sorted((r.subject, r.payload["other"]))) for r in self._recs
                 if r.kind == "contradiction" and r.rec_id in vis}
        return sorted(pairs)

    def recoveries_of(self, subject: str, now) -> list[ArchiveRecord]:
        """Recovery index (B11)."""
        return self._seqs(self.index.recovery.get(subject, []), now)

    # ---- snapshots (B12)
    def _influence_digest(self, now, max_seq: int | None) -> str:
        vis = self.visible_ids(now, max_seq)
        rows = sorted((r.payload["target"], as_date(r.occurred_at).toordinal(), r.payload["weight"])
                      for r in self._recs if r.kind == "influence" and r.rec_id in vis)
        return stable_hash(rows, 16)

    def snapshot(self, now, label: str = "", record: bool = True) -> Snapshot:
        """Freeze 'what the archive exposed at now'. The snapshot is itself written to L9 (unless record=False)."""
        head = len(self._recs) - 1
        vis = self.visible_ids(now)
        counts = Counter(self._by_id[i].layer.value for i in vis if self._by_id[i].kind != "snapshot")
        snap = Snapshot(now=_iso(now), head_seq=head, head_hash=self._recs[head].hash if head >= 0 else GENESIS,
                        digest=self.digest(now), n_visible=sum(counts.values()),
                        layer_counts=tuple(sorted(counts.items())), influence_digest=self._influence_digest(now, None),
                        label=label)
        if record:
            self.append(layer=Layer.L9_META, kind="snapshot", occurred_at=now, matured_at=now, subject=snap.snapshot_id,
                        payload={"head_hash": snap.head_hash, "head_seq": snap.head_seq, "digest": snap.digest,
                                 "n_visible": snap.n_visible, "layer_counts": [list(c) for c in snap.layer_counts],
                                 "influence_digest": snap.influence_digest, "label": label})
        return snap

    def verify_snapshot(self, snap: Snapshot) -> dict:
        """Is the snapshot still true? (1) the chain prefix still ends in the same hash, (2) recomputing the view over
        that prefix reproduces the digest. `late_arrivals` counts records added afterwards that claim to have matured
        before the snapshot's now - legitimate back-fill, but the view would differ if re-read today."""
        problems = []
        if snap.head_seq >= len(self._recs) or (snap.head_seq >= 0 and self._recs[snap.head_seq].hash != snap.head_hash):
            problems.append("chain prefix no longer ends in the snapshot's head hash (history was altered)")
        else:
            if self.digest(snap.now, max_seq=snap.head_seq) != snap.digest:
                problems.append("view digest recomputed over the prefix differs from the snapshot")
            if self._influence_digest(snap.now, snap.head_seq) != snap.influence_digest:
                problems.append("influence table differs from the snapshot")
        n = as_date(snap.now)
        late = sum(1 for r in self._recs[snap.head_seq + 1:] if r.kind != "snapshot" and as_date(r.matured_at) < n)
        return {"ok": not problems, "problems": problems, "late_arrivals": late}

    def snapshots(self) -> list[Snapshot]:
        out = []
        for r in self._recs:
            if r.kind == "snapshot":
                p = r.payload
                out.append(Snapshot(now=r.occurred_at, head_seq=p["head_seq"], head_hash=p["head_hash"],
                                    digest=p["digest"], n_visible=p["n_visible"],
                                    layer_counts=tuple((a, b) for a, b in p["layer_counts"]),
                                    influence_digest=p["influence_digest"], label=p.get("label", "")))
        return out

    # ---- audit (B15)
    def _reference_visible(self, now) -> set[str]:
        """Second, deliberately naive implementation of visibility (recursive, no cache) for the audit to compare with."""
        n = as_date(now)
        memo: dict[str, bool] = {}

        def ok(rid: str) -> bool:
            if rid in memo:
                return memo[rid]
            r = self._by_id[rid]
            good = (as_date(r.matured_at) < n and as_date(r.provenance.learned_at) < n
                    and as_date(r.provenance.outcomes_seen_through or r.provenance.learned_at) < n
                    and all(ok(p) for p in r.parents))
            memo[rid] = good
            return good

        return {rid for rid in self._by_id if ok(rid)}

    def audit(self, now) -> ArchiveAudit:
        """Verify the chain, every record's content id, parent ordering, abstraction hygiene, and that the view at `now`
        equals an independently computed set of records that could have existed then."""
        F: list[AuditFinding] = []
        v = self._chain.verify()
        if not v["ok"]:
            F.append(AuditFinding("CHAIN_BROKEN", "error", v["first_bad_seq"] or 0, "hash chain failed verification"))
        seen: set[str] = set()
        for r in self._recs:
            if r.recompute_id() != r.rec_id:
                F.append(AuditFinding("CONTENT_MISMATCH", "error", r.seq, "record content does not match its id"))
            for pid in r.parents:
                if pid not in seen:
                    F.append(AuditFinding("PARENT_ORDER", "error", r.seq, f"parent {pid} missing or written later"))
                elif as_date(self._by_id[pid].matured_at) > as_date(r.matured_at) and r.kind not in STRUCTURAL_KINDS:
                    F.append(AuditFinding("TIME_TRAVEL", "error", r.seq, "matured before its own parent"))
            for e in r.provenance.check():
                F.append(AuditFinding("PROVENANCE", "error", r.seq, e))
            if r.layer in ABSTRACT_LAYERS and r.kind not in STRUCTURAL_KINDS:
                for e in _identity_errors(r.payload, self.forbidden):
                    F.append(AuditFinding("IDENTITY_IN_ABSTRACT_LAYER", "error", r.seq, e))
            if r.kind == "influence" and r.payload["target"] not in self._by_id and not any(
                    x.subject == r.payload["target"] for x in self._recs):
                F.append(AuditFinding("DANGLING_INFLUENCE", "warn", r.seq, "influence targets an unknown id/subject"))
            if r.kind == "supersede" and r.payload["replacement"] not in self._by_id:
                F.append(AuditFinding("DANGLING_SUPERSEDE", "error", r.seq, "replacement record missing"))
            seen.add(r.rec_id)
        exposed = set(self.visible_ids(now))
        ref = self._reference_visible(now)
        for rid in sorted(exposed - ref):
            F.append(AuditFinding("VIEW_LEAK", "error", self._by_id[rid].seq, "exposed at now but could not have existed"))
        for rid in sorted(ref - exposed):
            F.append(AuditFinding("VIEW_HIDES", "warn", self._by_id[rid].seq, "could exist at now but is hidden"))
        for snap in self.snapshots():
            res = self.verify_snapshot(snap)
            for p in res["problems"]:
                F.append(AuditFinding("SNAPSHOT_BROKEN", "error", snap.head_seq, p))
            if res["late_arrivals"]:
                F.append(AuditFinding("SNAPSHOT_LATE_ARRIVALS", "warn", snap.head_seq,
                                      f"{res['late_arrivals']} records back-filled behind snapshot {snap.now}"))
        n = as_date(now)
        hidden = sum(1 for r in self._recs if as_date(r.matured_at) >= n)
        return ArchiveAudit(now=_iso(now), ok=not any(f.severity == "error" for f in F), records=len(self._recs),
                            visible=len(exposed), hidden_future=hidden, findings=F, head=self.head)

    def assert_clean(self, now):
        a = self.audit(now)
        if not a.ok:
            raise FirewallBreach(a.summary() + " :: " + "; ".join(f"{f.code}@{f.seq}" for f in a.errors()[:5]))
        return a

    # ---- stats & reports
    def stats(self, now) -> dict:
        vis = self.visible_ids(now)
        by_layer = Counter(self._by_id[i].layer.value for i in vis)
        by_kind = Counter(self._by_id[i].kind for i in vis)
        return {"now": _iso(now), "total": len(self._recs), "visible": len(vis), "by_layer": dict(sorted(by_layer.items())),
                "by_kind": dict(sorted(by_kind.items())), "failures": self.failure_counts(now),
                "contradiction_pairs": len(self.contradiction_pairs(now)), "head": self.head[:16]}

    def markdown(self, now, top: int = 10) -> str:
        s = self.stats(now)
        a = self.audit(now)
        lines = [f"# Knowledge archive as of {s['now']}", "", a.summary(), "",
                 "| layer | records |", "|---|---:|"]
        for layer in Layer:
            lines.append(f"| {layer.value} | {s['by_layer'].get(layer.value, 0)} |")
        lines += ["", f"failures by cause: {s['failures'] or 'none'}", f"open contradiction pairs: {s['contradiction_pairs']}", ""]
        rel = sorted({r.subject for r in self.view(now, kinds=('reliability',))})
        if rel:
            lines += ["| subject | reliability | trend |", "|---|---:|---:|"]
            for sub in rel[:top]:
                tr = self.reliability_trend(sub, now)
                lines.append(f"| {sub} | {self.reliability_at(sub, now):.3f} | {'n/a' if tr is None else f'{tr:+.3f}'} |")
        return "\n".join(lines)


def _plain(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v) for v in (sorted(x, key=str) if isinstance(x, (set, frozenset)) else x)]
    if hasattr(x, "value"):
        return x.value
    if hasattr(x, "item") and callable(x.item):
        return x.item()
    return x


def audit_prefix_invariance(arch: Archive, dates: Iterable) -> list[str]:
    """Rebuild an archive from ONLY the records visible at each date and check its view digest equals the full archive's:
    records that mature later must never change what was visible earlier. Returns a list of violations (empty = pass)."""
    bad = []
    for d in dates:
        keep = [arch.get(i) for i in sorted(arch.visible_ids(d), key=lambda i: arch.get(i).seq)]
        fresh = Archive(None, code_hash="prefix-audit", created_real="2000-01-01T00:00:00")
        for r in keep:
            fresh.append(layer=r.layer, kind=r.kind, payload=r.payload, occurred_at=r.occurred_at, matured_at=r.matured_at,
                         subject=r.subject, contexts=r.context_dict, parents=r.parents, provenance=r.provenance)
        if fresh.digest(d) != arch.digest(d):
            bad.append(f"view at {_iso(d)} changed when only earlier-known records were kept")
    return bad


def open_default(name: str = "learning_archive") -> Archive:
    """Persistent archive under state/learning/<name> (K.STATE)."""
    from engine import config as K
    return Archive(os.path.join(os.fspath(K.STATE), "learning", name))
