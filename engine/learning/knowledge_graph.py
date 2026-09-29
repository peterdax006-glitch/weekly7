"""Knowledge graph (contract C62 sections 50, 18; checklist F01-F11, F14, F15) - IMPLEMENTED, NOT VALIDATED.

Typed nodes (experience, situation, pattern, hypothesis, failure, condition, experiment, knowledge, decision, outcome) joined
by the ten section-18 edge types (SUPPORTS, CONTRADICTS, CONTAINS, SPECIALIZES, GENERALIZES, CAUSES_FAILURE_OF, RECOVERS_WITH,
REDUNDANT_WITH, COMPLEMENTS, DEPENDS_ON) plus the provenance links section 50's questions need (DERIVED_FROM, USED_IN,
RESULTED_IN, VALIDATED_BY, REFUTED_BY, TRANSFERRED_TO, APPLIES_IN, OBSERVED_IN).

Like the archive it never forgets: nodes and edges are versioned by `known_at`, an edge is "removed" by a retraction version,
and every query takes `now` and sees only what was known strictly before it (C56/C58). Persistence reuses archive.ChainFile, a lane on
engine.pattern_memory's hash chain, so a graph on disk is tamper-evident and can share one chain with the archive. The section-50 questions are first-class methods:
  what_caused_decision   failures_contradicting   situations_transferred   experiments_validated
plus knowledge_lineage / decision_lineage (F14/F15), traversal, impact analysis, PageRank and an integrity audit.
Contradictions are represented, never averaged: CONTRADICTS edges are symmetric and carry the context that explains them."""
from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter, defaultdict, deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .archive import ArchiveError, ChainFile, _identity_errors
from .core import Edge, FirewallBreach, _StrEnum, as_date, canonical_json, stable_hash


class NodeType(_StrEnum):
    EXPERIENCE = "EXPERIENCE"
    SITUATION = "SITUATION"
    PATTERN = "PATTERN"
    HYPOTHESIS = "HYPOTHESIS"
    FAILURE = "FAILURE"
    CONDITION = "CONDITION"
    EXPERIMENT = "EXPERIMENT"
    KNOWLEDGE = "KNOWLEDGE"
    DECISION = "DECISION"
    OUTCOME = "OUTCOME"


class Link(_StrEnum):
    """Provenance relations beyond the ten section-18 edges (needed to answer the four section-50 questions)."""
    DERIVED_FROM = "DERIVED_FROM"            # child -> what it was built from
    USED_IN = "USED_IN"                      # knowledge -> decision that used it (weight = contribution)
    RESULTED_IN = "RESULTED_IN"              # decision -> outcome
    VALIDATED_BY = "VALIDATED_BY"            # belief -> experiment that supported it
    REFUTED_BY = "REFUTED_BY"                # belief -> experiment that counted against it
    TRANSFERRED_TO = "TRANSFERRED_TO"        # knowledge -> situation it was applied in (attrs.success)
    APPLIES_IN = "APPLIES_IN"                # knowledge -> condition it holds in
    OBSERVED_IN = "OBSERVED_IN"              # experience -> situation


class GraphError(ArchiveError):
    pass


class GraphCycle(GraphError):
    pass


N = NodeType
_ABSTRACT = {N.PATTERN, N.HYPOTHESIS, N.KNOWLEDGE, N.CONDITION}
_BELIEF = {N.KNOWLEDGE, N.PATTERN, N.HYPOTHESIS}
_EVIDENCE = {N.EXPERIENCE, N.SITUATION, N.OUTCOME, N.EXPERIMENT, N.FAILURE}


@dataclasses.dataclass(frozen=True)
class RelSpec:
    src: frozenset
    dst: frozenset
    symmetric: bool = False
    acyclic: bool = False
    inverse: str | None = None
    doc: str = ""


def _s(*x):
    return frozenset(x)


RELATIONS: dict[str, RelSpec] = {
    Edge.SUPPORTS.value: RelSpec(frozenset(_BELIEF | _EVIDENCE), frozenset(_BELIEF), doc="evidence or belief backs a belief"),
    Edge.CONTRADICTS.value: RelSpec(frozenset(_BELIEF | _EVIDENCE), frozenset(_BELIEF), symmetric=True,
                                    doc="disagreement; never averaged, investigated"),
    Edge.CONTAINS.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN, N.SITUATION, N.CONDITION), _s(N.KNOWLEDGE, N.PATTERN, N.SITUATION,
                                                                                       N.CONDITION, N.HYPOTHESIS),
                                 acyclic=True, doc="container -> member"),
    Edge.SPECIALIZES.value: RelSpec(frozenset(_BELIEF | {N.CONDITION, N.SITUATION}),
                                    frozenset(_BELIEF | {N.CONDITION, N.SITUATION}), acyclic=True,
                                    inverse=Edge.GENERALIZES.value, doc="child (narrower) -> parent (broader)"),
    Edge.GENERALIZES.value: RelSpec(frozenset(_BELIEF | {N.CONDITION, N.SITUATION}),
                                    frozenset(_BELIEF | {N.CONDITION, N.SITUATION}), acyclic=True,
                                    inverse=Edge.SPECIALIZES.value, doc="parent (broader) -> child (narrower)"),
    Edge.CAUSES_FAILURE_OF.value: RelSpec(_s(N.CONDITION, N.FAILURE, N.EXPERIENCE, N.SITUATION, N.OUTCOME),
                                          _s(N.KNOWLEDGE, N.PATTERN, N.HYPOTHESIS, N.DECISION),
                                          doc="cause -> the thing that failed"),
    Edge.RECOVERS_WITH.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN), _s(N.CONDITION, N.EXPERIMENT, N.SITUATION),
                                      doc="belief comes back when this holds"),
    Edge.REDUNDANT_WITH.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN), _s(N.KNOWLEDGE, N.PATTERN), symmetric=True),
    Edge.COMPLEMENTS.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN), _s(N.KNOWLEDGE, N.PATTERN), symmetric=True),
    Edge.DEPENDS_ON.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN, N.DECISION, N.HYPOTHESIS),
                                   _s(N.KNOWLEDGE, N.PATTERN, N.CONDITION, N.HYPOTHESIS), acyclic=True),
    Link.DERIVED_FROM.value: RelSpec(frozenset(_BELIEF | {N.SITUATION, N.OUTCOME, N.DECISION, N.CONDITION, N.EXPERIMENT,
                                                          N.EXPERIENCE, N.FAILURE}),
                                     frozenset(N), acyclic=True, doc="built from"),
    Link.USED_IN.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN, N.CONDITION), _s(N.DECISION)),
    Link.RESULTED_IN.value: RelSpec(_s(N.DECISION), _s(N.OUTCOME)),
    Link.VALIDATED_BY.value: RelSpec(frozenset(_BELIEF), _s(N.EXPERIMENT)),
    Link.REFUTED_BY.value: RelSpec(frozenset(_BELIEF), _s(N.EXPERIMENT)),
    Link.TRANSFERRED_TO.value: RelSpec(_s(N.KNOWLEDGE, N.PATTERN), _s(N.SITUATION)),
    Link.APPLIES_IN.value: RelSpec(frozenset(_BELIEF), _s(N.CONDITION)),
    Link.OBSERVED_IN.value: RelSpec(_s(N.EXPERIENCE), _s(N.SITUATION)),
}
LINEAGE_RELS = (Link.DERIVED_FROM.value, Edge.SPECIALIZES.value)
DEPENDENCY_RELS = (Edge.DEPENDS_ON.value,)


@dataclasses.dataclass(frozen=True)
class Node:
    node_id: str
    ntype: NodeType
    label: str
    known_at: str
    attrs_json: str = "{}"

    @property
    def attrs(self) -> dict:
        return json.loads(self.attrs_json)

    def to_body(self) -> dict:
        return {"op": "node", "node_id": self.node_id, "ntype": self.ntype.value, "label": self.label,
                "known_at": self.known_at, "attrs": self.attrs}


@dataclasses.dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    rel: str
    weight: float
    known_at: str
    retracted: bool = False
    evidence: tuple[str, ...] = ()
    attrs_json: str = "{}"

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.rel)

    @property
    def attrs(self) -> dict:
        return json.loads(self.attrs_json)

    @property
    def edge_id(self) -> str:
        return stable_hash([self.src, self.dst, self.rel, self.known_at, self.retracted, self.weight], 16)

    def to_body(self) -> dict:
        return {"op": "edge", "src": self.src, "dst": self.dst, "rel": self.rel, "weight": self.weight,
                "known_at": self.known_at, "retracted": self.retracted, "evidence": list(self.evidence),
                "attrs": self.attrs}


@dataclasses.dataclass(frozen=True)
class EdgeProposal:
    """A relationship another module wants recorded (contradiction/hierarchy/redundancy builders return these so they do
    not have to import the graph store). `KnowledgeGraph.apply` turns proposals into edges after validation."""
    src: str
    dst: str
    rel: str
    weight: float = 1.0
    evidence: tuple[str, ...] = ()
    attrs: tuple[tuple[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class TraversalResult:
    start: str
    depth: dict[str, int]
    parent: dict[str, tuple[str, str]]              # node -> (came from, relation)

    def path_to(self, node: str) -> list[str]:
        if node not in self.depth:
            return []
        out = [node]
        while out[-1] != self.start:
            out.append(self.parent[out[-1]][0])
        return out[::-1]

    def nodes(self) -> list[str]:
        return sorted(self.depth, key=lambda n: (self.depth[n], n))


@dataclasses.dataclass(frozen=True)
class LineageStep:
    node_id: str
    ntype: str
    depth: int
    via: str                                        # relation taken from the previous step
    weight: float
    known_at: str


@dataclasses.dataclass(frozen=True)
class Lineage:
    root: str
    steps: tuple[LineageStep, ...]
    origins: tuple[str, ...]                        # leaf experiences/experiments the belief ultimately rests on

    @property
    def digest(self) -> str:
        return stable_hash([self.root, [dataclasses.astuple(s) for s in self.steps]], 16)

    def depth(self) -> int:
        return max((s.depth for s in self.steps), default=0)


@dataclasses.dataclass(frozen=True)
class DecisionCause:
    knowledge_id: str
    contribution: float                             # normalised share of the decision this knowledge drove
    raw_weight: float
    depends_on: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class DecisionLineage:
    decision: str
    causes: tuple[DecisionCause, ...]
    lineages: tuple[tuple[str, Lineage], ...]
    outcomes: tuple[str, ...]
    unsupported: bool                               # a decision with no knowledge behind it is a baseline decision

    def blame(self) -> dict[str, float]:
        return {c.knowledge_id: c.contribution for c in self.causes}


@dataclasses.dataclass(frozen=True)
class FailureLink:
    failure_id: str
    cause: str
    weight: float
    known_at: str
    via: str


@dataclasses.dataclass(frozen=True)
class TransferRecord:
    situation_id: str
    success: bool
    weight: float
    known_at: str


@dataclasses.dataclass(frozen=True)
class TransferSummary:
    knowledge_id: str
    successes: int
    failures: int
    rate: float | None
    lower_bound: float | None                       # Wilson lower bound: small samples cannot look certain
    situations: tuple[TransferRecord, ...]


@dataclasses.dataclass(frozen=True)
class ExperimentEvidence:
    experiment_id: str
    supports: bool
    weight: float
    known_at: str


@dataclasses.dataclass(frozen=True)
class GraphIssue:
    code: str
    severity: str
    where: str
    detail: str


def wilson_lower(k: int, n: int, z: float = 1.645) -> float | None:
    """One-sided 95% Wilson lower bound of a proportion; None with no trials (UNKNOWN, not 0)."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


def _canon_rel(rel) -> str:
    v = rel.value if hasattr(rel, "value") else str(rel)
    if v not in RELATIONS:
        raise GraphError(f"unknown relation {rel!r}")
    return v


class KnowledgeGraph:
    """Versioned, time-aware, tamper-evident graph. `path=None` keeps it in memory."""

    def __init__(self, path=None, forbidden_identities: Iterable[str] = ()):
        self._chain = ChainFile(path, "graph")
        self.forbidden = frozenset(str(x).upper() for x in forbidden_identities)
        self._nodes: dict[str, list[Node]] = {}
        self._edges: dict[tuple, list[GraphEdge]] = {}
        self._out: dict[str, set] = defaultdict(set)
        self._in: dict[str, set] = defaultdict(set)
        self._replay()

    # ------------------------------------------------------------------ persistence
    def _replay(self):
        for line in self._chain.take_new():
            b = line["body"]
            if b["op"] == "node":
                self._put_node(Node(b["node_id"], NodeType(b["ntype"]), b["label"], b["known_at"],
                                    canonical_json(b["attrs"])))
            elif b["op"] == "edge":
                self._put_edge(GraphEdge(b["src"], b["dst"], b["rel"], float(b["weight"]), b["known_at"],
                                         bool(b["retracted"]), tuple(b["evidence"]), canonical_json(b["attrs"])))
            else:
                raise GraphError(f"unknown op {b['op']!r} in graph chain")

    def refresh(self):
        self._chain.sync()
        self._replay()

    def verify(self) -> dict:
        return self._chain.verify()

    @property
    def head(self) -> str:
        return self._chain.head

    def _put_node(self, n: Node):
        self._nodes.setdefault(n.node_id, []).append(n)

    def _put_edge(self, e: GraphEdge):
        self._edges.setdefault(e.key, []).append(e)
        spec = RELATIONS[e.rel]
        self._out[e.src].add(e.key)
        self._in[e.dst].add(e.key)
        if spec.symmetric:
            self._out[e.dst].add(e.key)
            self._in[e.src].add(e.key)

    # ------------------------------------------------------------------ nodes
    def add_node(self, node_id: str, ntype: NodeType | str, known_at, label: str = "", attrs: Mapping | None = None) -> Node:
        """Create a node, or a NEW VERSION of it (type cannot change). Abstract node types must be identity-free."""
        nt = NodeType.parse(ntype)
        if not node_id or not isinstance(node_id, str):
            raise GraphError("node_id must be a non-empty string")
        a = dict(attrs or {})
        if nt in _ABSTRACT:
            errs = _identity_errors({"label": label, **a}, self.forbidden, "node")
            if errs:
                raise GraphError(f"{nt.value} node {node_id}: " + "; ".join(errs))
        cur = self._nodes.get(node_id)
        if cur and cur[0].ntype != nt:
            raise GraphError(f"node {node_id} is {cur[0].ntype.value}; cannot become {nt.value}")
        n = Node(node_id, nt, label, as_date(known_at).isoformat(), canonical_json(a))
        if cur and cur[-1] == n:
            return cur[-1]
        if cur and as_date(known_at) < as_date(cur[-1].known_at):
            raise GraphError(f"node {node_id}: version dated {n.known_at} is older than the latest {cur[-1].known_at}")
        self._chain.append_many([n.to_body()])
        self._replay()
        return n

    def node_at(self, node_id: str, now) -> Node | None:
        cur = self._nodes.get(node_id)
        if not cur:
            return None
        n = as_date(now)
        best = None
        for v in cur:
            if as_date(v.known_at) < n:
                best = v
        return best

    def has_node(self, node_id: str, now=None) -> bool:
        return (node_id in self._nodes) if now is None else self.node_at(node_id, now) is not None

    def nodes(self, now, ntype: NodeType | str | None = None) -> list[Node]:
        t = NodeType.parse(ntype) if ntype else None
        out = [self.node_at(i, now) for i in sorted(self._nodes)]
        return [n for n in out if n is not None and (t is None or n.ntype == t)]

    def first_known(self, node_id: str) -> str | None:
        cur = self._nodes.get(node_id)
        return cur[0].known_at if cur else None

    # ------------------------------------------------------------------ edges
    def _validate_edge(self, src: str, dst: str, rel: str, weight: float, known_at) -> list[str]:
        errs = []
        if src == dst:
            errs.append("self-loop")
        if not (isinstance(weight, (int, float)) and 0.0 <= float(weight) <= 1.0):
            errs.append(f"weight {weight!r} outside [0,1]")
        spec = RELATIONS[rel]
        ns, nd = self._nodes.get(src), self._nodes.get(dst)
        if not ns:
            errs.append(f"unknown source node {src}")
        if not nd:
            errs.append(f"unknown target node {dst}")
        if ns and nd:
            ts, td = ns[0].ntype, nd[0].ntype
            ok = ts in spec.src and td in spec.dst
            if spec.symmetric and not ok:
                ok = td in spec.src and ts in spec.dst
            if not ok:
                errs.append(f"{rel} not allowed from {ts.value} to {td.value}")
            k = as_date(known_at)
            for nid, first in ((src, ns[0].known_at), (dst, nd[0].known_at)):
                if k < as_date(first):
                    errs.append(f"edge known {k} before node {nid} existed ({first})")
        return errs

    def _would_cycle(self, src: str, dst: str, rel: str) -> bool:
        """Adding src->dst on an acyclic relation is a cycle iff dst already reaches src along live edges of that relation."""
        seen, stack = {dst}, [dst]
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            for key in self._out.get(cur, ()):
                if key[2] != rel or key[0] != cur:
                    continue
                if self._edges[key][-1].retracted:
                    continue
                if key[1] not in seen:
                    seen.add(key[1])
                    stack.append(key[1])
        return False

    def add_edge(self, src: str, dst: str, rel: Edge | Link | str, known_at, weight: float = 1.0,
                 evidence: Sequence[str] = (), attrs: Mapping | None = None) -> GraphEdge:
        """Add (or re-weight, as a new version) one relation. SPECIALIZES/GENERALIZES are stored as a mirrored pair."""
        r = _canon_rel(rel)
        spec = RELATIONS[r]
        if spec.symmetric and src > dst:
            src, dst = dst, src
        errs = self._validate_edge(src, dst, r, weight, known_at)
        if errs:
            raise GraphError(f"{r} {src}->{dst}: " + "; ".join(errs))
        cur = self._edges.get((src, dst, r))
        if spec.acyclic and (not cur or cur[-1].retracted) and self._would_cycle(src, dst, r):
            raise GraphCycle(f"{r} {src}->{dst} would create a cycle")
        e = GraphEdge(src, dst, r, float(weight), as_date(known_at).isoformat(), False, tuple(evidence),
                      canonical_json(dict(attrs or {})))
        if cur and cur[-1] == e:
            return cur[-1]
        if cur and as_date(known_at) < as_date(cur[-1].known_at):
            raise GraphError("edge version older than the latest version")
        bodies = [e.to_body()]
        if spec.inverse:
            inv = dataclasses.replace(e, src=dst, dst=src, rel=spec.inverse)
            icur = self._edges.get(inv.key)
            if not (icur and icur[-1] == inv):
                bodies.append(inv.to_body())
        self._chain.append_many(bodies)
        self._replay()
        return e

    def retract_edge(self, src: str, dst: str, rel: Edge | Link | str, known_at, reason: str) -> GraphEdge:
        """Withdraw a relation without erasing it: a retraction version is appended; earlier `now` queries still see it."""
        r = _canon_rel(rel)
        spec = RELATIONS[r]
        if spec.symmetric and src > dst:
            src, dst = dst, src
        cur = self._edges.get((src, dst, r))
        if not cur or cur[-1].retracted:
            raise GraphError(f"no live {r} edge {src}->{dst} to retract")
        if not reason.strip():
            raise GraphError("a retraction needs a reason")
        if as_date(known_at) < as_date(cur[-1].known_at):
            raise GraphError("retraction predates the edge version it retracts")
        base = cur[-1]
        e = dataclasses.replace(base, weight=0.0, known_at=as_date(known_at).isoformat(), retracted=True,
                                attrs_json=canonical_json({**base.attrs, "retracted_because": reason}))
        bodies = [e.to_body()]
        if spec.inverse:
            ic = self._edges.get((dst, src, spec.inverse))
            if ic and not ic[-1].retracted:
                bodies.append(dataclasses.replace(ic[-1], weight=0.0, known_at=e.known_at, retracted=True,
                                                  attrs_json=e.attrs_json).to_body())
        self._chain.append_many(bodies)
        self._replay()
        return e

    def edge_at(self, key: tuple, now) -> GraphEdge | None:
        """Latest version of an edge known strictly before `now`; None if unknown or retracted then."""
        n = as_date(now)
        best = None
        for v in self._edges.get(key, ()):
            if as_date(v.known_at) < n:
                best = v
        return None if best is None or best.retracted else best

    def edge_history(self, src: str, dst: str, rel) -> list[GraphEdge]:
        r = _canon_rel(rel)
        if RELATIONS[r].symmetric and src > dst:
            src, dst = dst, src
        return list(self._edges.get((src, dst, r), ()))

    def edges(self, now, rel=None) -> list[GraphEdge]:
        rels = {_canon_rel(x) for x in ([rel] if rel is not None and not isinstance(rel, (list, tuple, set)) else rel or ())}
        out = []
        for key in sorted(self._edges):
            if rels and key[2] not in rels:
                continue
            e = self.edge_at(key, now)
            if e is not None and self.node_at(key[0], now) and self.node_at(key[1], now):
                out.append(e)
        return out

    def apply(self, proposals: Iterable[EdgeProposal], known_at) -> list[GraphEdge]:
        """Record edge proposals from other modules; invalid ones raise (nothing is silently dropped)."""
        return [self.add_edge(p.src, p.dst, p.rel, known_at, p.weight, p.evidence, dict(p.attrs)) for p in proposals]

    # ------------------------------------------------------------------ traversal (F11)
    def neighbors(self, node: str, now, rels: Iterable | None = None, direction: str = "out",
                  min_weight: float = 0.0) -> list[tuple[str, GraphEdge]]:
        """(other node, edge) pairs live at `now`. Symmetric relations ignore direction."""
        if direction not in ("out", "in", "both"):
            raise GraphError("direction must be out/in/both")
        rs = {_canon_rel(r) for r in rels} if rels is not None else None
        keys = set()
        if direction in ("out", "both"):
            keys |= self._out.get(node, set())
        if direction in ("in", "both"):
            keys |= self._in.get(node, set())
        res = []
        for key in sorted(keys):
            src, dst, rel = key
            if rs is not None and rel not in rs:
                continue
            e = self.edge_at(key, now)
            if e is None or e.weight < min_weight:
                continue
            spec = RELATIONS[rel]
            if spec.symmetric:
                other = dst if node == src else src
            elif node == src and direction in ("out", "both") and key in self._out.get(node, ()):
                other = dst
            elif node == dst and direction in ("in", "both"):
                other = src
            else:
                continue
            if self.node_at(other, now) is not None:
                res.append((other, e))
        return res

    def traverse(self, start: str, now, rels: Iterable | None = None, direction: str = "out", max_depth: int = 6,
                 min_weight: float = 0.0) -> TraversalResult:
        if self.node_at(start, now) is None:
            raise GraphError(f"node {start} is not known at {as_date(now)}")
        depth, parent = {start: 0}, {}
        q = deque([start])
        while q:
            cur = q.popleft()
            if depth[cur] >= max_depth:
                continue
            for other, e in self.neighbors(cur, now, rels, direction, min_weight):
                if other not in depth:
                    depth[other] = depth[cur] + 1
                    parent[other] = (cur, e.rel)
                    q.append(other)
        return TraversalResult(start, depth, parent)

    def shortest_path(self, src: str, dst: str, now, rels: Iterable | None = None, direction: str = "both",
                      max_depth: int = 8) -> list[str]:
        t = self.traverse(src, now, rels, direction, max_depth)
        return t.path_to(dst)

    def paths(self, src: str, dst: str, now, rels: Iterable | None = None, direction: str = "out", max_depth: int = 5,
              limit: int = 20) -> list[list[str]]:
        """All simple paths up to max_depth (bounded by `limit`), shortest first."""
        out: list[list[str]] = []

        def dfs(cur: str, path: list[str]):
            if len(out) >= limit * 4 or len(path) > max_depth + 1:
                return
            if cur == dst and len(path) > 1:
                out.append(list(path))
                return
            for other, _ in self.neighbors(cur, now, rels, direction):
                if other not in path:
                    path.append(other)
                    dfs(other, path)
                    path.pop()

        dfs(src, [src])
        out.sort(key=lambda p: (len(p), p))
        return out[:limit]

    def ancestors(self, node: str, now, rels: Iterable = LINEAGE_RELS, max_depth: int = 12) -> TraversalResult:
        return self.traverse(node, now, rels, "out", max_depth)

    def descendants(self, node: str, now, rels: Iterable = LINEAGE_RELS, max_depth: int = 12) -> TraversalResult:
        return self.traverse(node, now, rels, "in", max_depth)

    def components(self, now, rels: Iterable | None = None) -> list[list[str]]:
        """Weakly connected components over live edges (union-find), largest first."""
        ids = [n.node_id for n in self.nodes(now)]
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        rs = {_canon_rel(r) for r in rels} if rels is not None else None
        for e in self.edges(now):
            if rs is None or e.rel in rs:
                a, b = find(e.src), find(e.dst)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        groups: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            groups[find(i)].append(i)
        return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))

    def topological_order(self, rel, now) -> list[str]:
        """Order nodes along an acyclic relation (parents before children by edge direction); raises on a cycle."""
        r = _canon_rel(rel)
        es = [e for e in self.edges(now, r)]
        nodes = sorted({x for e in es for x in (e.src, e.dst)})
        indeg = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = defaultdict(list)
        for e in es:
            adj[e.src].append(e.dst)
            indeg[e.dst] += 1
        ready = deque(sorted(n for n, d in indeg.items() if d == 0))
        order = []
        while ready:
            n = ready.popleft()
            order.append(n)
            for m in sorted(adj[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        if len(order) != len(nodes):
            raise GraphCycle(f"cycle in {r}")
        return order

    def find_cycles(self, now) -> list[str]:
        """Relations whose live edges contain a cycle (should be empty; add_edge prevents them)."""
        bad = []
        for r, spec in RELATIONS.items():
            if spec.acyclic:
                try:
                    self.topological_order(r, now)
                except GraphCycle:
                    bad.append(r)
        return bad

    # ------------------------------------------------------------------ metrics
    def degree(self, node: str, now, rels: Iterable | None = None) -> dict[str, int]:
        return {"out": len(self.neighbors(node, now, rels, "out")), "in": len(self.neighbors(node, now, rels, "in"))}

    def pagerank(self, now, rels: Iterable | None = None, damping: float = 0.85, iters: int = 100, tol: float = 1e-10
                 ) -> dict[str, float]:
        """Weighted PageRank over live edges (symmetric relations count both ways); deterministic."""
        ids = [n.node_id for n in self.nodes(now)]
        if not ids:
            return {}
        ix = {n: i for i, n in enumerate(ids)}
        M = np.zeros((len(ids), len(ids)))
        rs = {_canon_rel(r) for r in rels} if rels is not None else None
        for e in self.edges(now):
            if rs is not None and e.rel not in rs:
                continue
            M[ix[e.dst], ix[e.src]] += e.weight
            if RELATIONS[e.rel].symmetric:
                M[ix[e.src], ix[e.dst]] += e.weight
        colsum = M.sum(axis=0)
        dangling = colsum == 0
        M = np.divide(M, colsum, out=np.zeros_like(M), where=colsum > 0)
        r = np.full(len(ids), 1.0 / len(ids))
        for _ in range(iters):
            nr = damping * (M @ r + r[dangling].sum() / len(ids)) + (1 - damping) / len(ids)
            if np.abs(nr - r).sum() < tol:
                r = nr
                break
            r = nr
        return {n: float(r[i]) for n, i in ix.items()}

    def support_balance(self, node: str, now) -> dict[str, float]:
        """Weighted support vs contradiction pressure on a belief, without collapsing them into one number."""
        sup = sum(e.weight for o, e in self.neighbors(node, now, [Edge.SUPPORTS], "in"))
        sup += sum(e.weight for o, e in self.neighbors(node, now, [Link.VALIDATED_BY], "out"))
        con = sum(e.weight for o, e in self.neighbors(node, now, [Edge.CONTRADICTS], "both"))
        con += sum(e.weight for o, e in self.neighbors(node, now, [Link.REFUTED_BY], "out"))
        return {"support": float(sup), "contradiction": float(con), "contested": float(min(sup, con))}

    def redundancy_clusters(self, now) -> list[list[str]]:
        """Groups of beliefs connected by REDUNDANT_WITH (only clusters of 2+)."""
        comp = self.components(now, [Edge.REDUNDANT_WITH])
        red = {x for e in self.edges(now, Edge.REDUNDANT_WITH) for x in (e.src, e.dst)}
        return [g for g in comp if len(g) > 1 and set(g) <= red]

    def contradiction_pairs(self, now) -> list[tuple[str, str]]:
        return [(e.src, e.dst) for e in self.edges(now, Edge.CONTRADICTS)]

    def impact_analysis(self, node: str, now, max_depth: int = 8) -> dict[str, list[str]]:
        """What would be affected if this belief were retired: dependants, specialisations, and decisions that used it."""
        dep = self.traverse(node, now, DEPENDENCY_RELS, "in", max_depth).nodes()[1:]
        spec = self.traverse(node, now, [Edge.GENERALIZES], "out", max_depth).nodes()[1:]
        decisions = []
        for n in [node] + dep + spec:
            decisions += [o for o, _ in self.neighbors(n, now, [Link.USED_IN], "out")]
        return {"dependants": sorted(dep), "specialisations": sorted(spec), "decisions": sorted(set(decisions))}

    # ------------------------------------------------------------------ the four section-50 questions
    def what_caused_decision(self, decision_id: str, now, include_dependencies: bool = True) -> list[DecisionCause]:
        """Q1: which knowledge caused this decision? Contributions are normalised weights of live USED_IN edges."""
        n = self.node_at(decision_id, now)
        if n is None or n.ntype != NodeType.DECISION:
            raise GraphError(f"{decision_id} is not a decision known at {as_date(now)}")
        used = self.neighbors(decision_id, now, [Link.USED_IN], "in")
        tot = sum(e.weight for _, e in used)
        out = []
        for k, e in sorted(used, key=lambda t: (-t[1].weight, t[0])):
            deps: tuple[str, ...] = ()
            if include_dependencies:
                deps = tuple(sorted(self.traverse(k, now, DEPENDENCY_RELS, "out", 6).nodes()[1:]))
            out.append(DecisionCause(k, e.weight / tot if tot > 0 else 0.0, e.weight, deps))
        return out

    def failures_contradicting(self, pattern_id: str, now, include_beliefs: bool = False) -> list[FailureLink]:
        """Q2: what failures have contradicted this pattern? FAILURE nodes joined by CAUSES_FAILURE_OF or CONTRADICTS
        (and optionally contradicting beliefs)."""
        if self.node_at(pattern_id, now) is None:
            raise GraphError(f"{pattern_id} not known at {as_date(now)}")
        found: dict[str, FailureLink] = {}
        for rel, direction in ((Edge.CAUSES_FAILURE_OF, "in"), (Edge.CONTRADICTS, "both")):
            for other, e in self.neighbors(pattern_id, now, [rel], direction):
                nd = self.node_at(other, now)
                if nd.ntype == NodeType.FAILURE or (include_beliefs and nd.ntype in _BELIEF and rel == Edge.CONTRADICTS):
                    cause = nd.attrs.get("cause", e.attrs.get("cause", "UNKNOWN"))
                    prev = found.get(other)
                    if prev is None or e.weight > prev.weight:
                        found[other] = FailureLink(other, str(cause), e.weight, e.known_at, e.rel)
        return sorted(found.values(), key=lambda f: (-f.weight, f.failure_id))

    def situations_transferred(self, knowledge_id: str, now, successful: bool | None = True) -> list[TransferRecord]:
        """Q3: in which situations has this knowledge transferred? `successful` None returns successes and failures."""
        recs = []
        for sit, e in self.neighbors(knowledge_id, now, [Link.TRANSFERRED_TO], "out"):
            ok = bool(e.attrs.get("success", e.weight >= 0.5))
            if successful is None or ok == successful:
                recs.append(TransferRecord(sit, ok, e.weight, e.known_at))
        return sorted(recs, key=lambda r: (r.known_at, r.situation_id))

    def transfer_summary(self, knowledge_id: str, now) -> TransferSummary:
        recs = self.situations_transferred(knowledge_id, now, None)
        k = sum(r.success for r in recs)
        n = len(recs)
        return TransferSummary(knowledge_id, k, n - k, (k / n) if n else None, wilson_lower(k, n), tuple(recs))

    def experiments_validated(self, belief_id: str, now) -> list[ExperimentEvidence]:
        """Q4: which experiments validated this belief? Supportive results only; see experiment_record for both sides."""
        return [x for x in self.experiment_record(belief_id, now) if x.supports]

    def experiment_record(self, belief_id: str, now) -> list[ExperimentEvidence]:
        out = []
        for rel, sup in ((Link.VALIDATED_BY, True), (Link.REFUTED_BY, False)):
            for exp, e in self.neighbors(belief_id, now, [rel], "out"):
                out.append(ExperimentEvidence(exp, sup, e.weight, e.known_at))
        return sorted(out, key=lambda x: (x.known_at, x.experiment_id))

    # ------------------------------------------------------------------ lineage (F14, F15)
    def knowledge_lineage(self, knowledge_id: str, now, max_depth: int = 12) -> Lineage:
        """F14: everything a belief was derived from or specialises, nearest first, ending in raw experience/experiments."""
        if self.node_at(knowledge_id, now) is None:
            raise GraphError(f"{knowledge_id} not known at {as_date(now)}")
        t = self.traverse(knowledge_id, now, LINEAGE_RELS, "out", max_depth)
        steps = []
        for nid in t.nodes()[1:]:
            frm, rel = t.parent[nid]
            e = self.edge_at((frm, nid, rel), now)
            nd = self.node_at(nid, now)
            steps.append(LineageStep(nid, nd.ntype.value, t.depth[nid], rel, e.weight if e else 0.0, nd.known_at))
        leaves = tuple(sorted(s.node_id for s in steps if s.ntype in (NodeType.EXPERIENCE.value, NodeType.EXPERIMENT.value)
                              and not self.neighbors(s.node_id, now, LINEAGE_RELS, "out")))
        return Lineage(knowledge_id, tuple(steps), leaves)

    def decision_lineage(self, decision_id: str, now) -> DecisionLineage:
        """F15: decision <- knowledge <- (derivations) <- experience, and forward to what happened."""
        causes = self.what_caused_decision(decision_id, now)
        lin = tuple((c.knowledge_id, self.knowledge_lineage(c.knowledge_id, now)) for c in causes)
        outs = tuple(sorted(o for o, _ in self.neighbors(decision_id, now, [Link.RESULTED_IN], "out")))
        return DecisionLineage(decision_id, tuple(causes), lin, outs, unsupported=not causes)

    def outcome_attribution(self, decision_id: str, now, outcome_value: float) -> dict[str, float]:
        """Split an outcome across the knowledge that drove the decision, in proportion to contribution."""
        return {k: c * float(outcome_value) for k, c in self.decision_lineage(decision_id, now).blame().items()}

    # ------------------------------------------------------------------ ingestion helpers
    def add_knowledge(self, k, known_at, label: str = "") -> Node:
        """Register a KnowledgeLike duck type, its contexts as CONDITION nodes (APPLIES_IN) and anti-contexts likewise
        (CAUSES_FAILURE_OF: an anti-context is a condition that breaks it)."""
        node = self.add_node(str(k.knowledge_id), NodeType.KNOWLEDGE, known_at, label or str(k.knowledge_id),
                             {"version": int(k.version), "epistemic": str(k.epistemic), "lifecycle": str(k.lifecycle),
                              "promotion": str(k.promotion)})
        for kind, mapping in (("ctx", k.contexts), ("anti", k.anti_contexts)):
            for dim, cond in dict(mapping).items():
                cid = f"cond:{dim}={cond}"
                self.add_node(cid, NodeType.CONDITION, known_at, cid, {"dimension": str(dim), "value": str(cond)})
                if kind == "ctx":
                    self.add_edge(node.node_id, cid, Link.APPLIES_IN, known_at)
                else:
                    self.add_edge(cid, node.node_id, Edge.CAUSES_FAILURE_OF, known_at, attrs={"anti_context": True})
        return node

    def ingest_archive(self, archive, now) -> dict[str, int]:
        """Mirror the archive's visible records into the graph (nodes for evidence/hypotheses/patterns/knowledge, DERIVED_FROM
        for parents, CAUSES_FAILURE_OF for failure records, CONTRADICTS for contradiction records). Idempotent."""
        kmap = {"observation": NodeType.EXPERIENCE, "event": NodeType.EXPERIENCE, "episode": NodeType.EXPERIENCE,
                "situation": NodeType.SITUATION, "hypothesis": NodeType.HYPOTHESIS, "experiment": NodeType.EXPERIMENT,
                "pattern": NodeType.PATTERN, "context_rule": NodeType.CONDITION, "knowledge": NodeType.KNOWLEDGE,
                "decision": NodeType.DECISION, "policy": NodeType.DECISION, "outcome": NodeType.OUTCOME,
                "failure": NodeType.FAILURE}
        named = {"hypothesis", "pattern", "context_rule", "knowledge"}
        ids: dict[str, str] = {}
        counts: Counter = Counter()
        recs = sorted(archive.view(now), key=lambda r: (r.matured_at, r.seq))
        for r in recs:
            nt = kmap.get(r.kind)
            if nt is None:
                continue
            nid = r.subject if (r.kind in named and r.subject) else r.rec_id
            ids[r.rec_id] = nid
            attrs = {"layer": r.layer.value, "kind": r.kind}
            if r.kind == "failure":
                attrs["cause"] = r.payload["cause"]
            self.add_node(nid, nt, r.matured_at, r.subject or nid, attrs)
            counts["nodes"] += 1
        for r in recs:
            if r.rec_id not in ids:
                continue
            for p in r.parents:
                if p in ids and ids[p] != ids[r.rec_id]:
                    try:
                        self.add_edge(ids[r.rec_id], ids[p], Link.DERIVED_FROM, r.matured_at)
                        counts["derived"] += 1
                    except GraphCycle:
                        counts["cycle_skipped"] += 1
            if r.kind == "failure" and r.subject in self._nodes:
                self.add_edge(ids[r.rec_id], r.subject, Edge.CAUSES_FAILURE_OF, r.matured_at,
                              attrs={"cause": r.payload["cause"]})
                counts["failure_edges"] += 1
        for r in recs:
            if r.kind == "contradiction" and r.subject in self._nodes and r.payload["other"] in self._nodes:
                self.add_edge(r.subject, r.payload["other"], Edge.CONTRADICTS, r.matured_at,
                              attrs=r.payload.get("detail") or {})
                counts["contradicts"] += 1
        return dict(counts)

    # ------------------------------------------------------------------ audit & reports
    def audit(self, now) -> list[GraphIssue]:
        """Structural integrity at `now`: chain, dangling/ill-typed edges, cycles, mirror consistency, contradictions with
        no investigation context, beliefs with no lineage, decisions with no knowledge."""
        issues: list[GraphIssue] = []
        v = self.verify()
        if not v["ok"]:
            issues.append(GraphIssue("CHAIN_BROKEN", "error", str(v["first_bad_seq"]), "graph hash chain failed"))
        for key, versions in sorted(self._edges.items()):
            src, dst, rel = key
            for e in versions:
                for end in (src, dst):
                    first = self.first_known(end)
                    if first is None:
                        issues.append(GraphIssue("DANGLING_EDGE", "error", f"{rel}:{src}->{dst}", f"missing node {end}"))
                    elif as_date(e.known_at) < as_date(first):
                        issues.append(GraphIssue("EDGE_BEFORE_NODE", "error", f"{rel}:{src}->{dst}",
                                                 f"edge known {e.known_at} before node {end} ({first})"))
            spec = RELATIONS[rel]
            if spec.inverse:
                mirror = self.edge_at((dst, src, spec.inverse), now)
                mine = self.edge_at(key, now)
                if (mine is None) != (mirror is None):
                    issues.append(GraphIssue("MIRROR_MISSING", "error", f"{rel}:{src}->{dst}", "inverse edge out of step"))
        for r in self.find_cycles(now):
            issues.append(GraphIssue("CYCLE", "error", r, "acyclic relation contains a cycle"))
        for a, b in self.contradiction_pairs(now):
            e = self.edge_at((a, b, Edge.CONTRADICTS.value), now)
            if not e.attrs.get("context") and not e.attrs.get("investigated"):
                issues.append(GraphIssue("UNINVESTIGATED_CONTRADICTION", "warn", f"{a}<->{b}",
                                         "no explaining context recorded (never average; investigate)"))
        for n in self.nodes(now, NodeType.DECISION):
            if not self.neighbors(n.node_id, now, [Link.USED_IN], "in"):
                issues.append(GraphIssue("UNSUPPORTED_DECISION", "warn", n.node_id, "no knowledge behind this decision"))
        for n in self.nodes(now, NodeType.KNOWLEDGE):
            if not self.neighbors(n.node_id, now, LINEAGE_RELS, "out"):
                issues.append(GraphIssue("NO_LINEAGE", "warn", n.node_id, "knowledge with no recorded origin"))
        return issues

    def stats(self, now) -> dict:
        nodes = self.nodes(now)
        es = self.edges(now)
        return {"now": as_date(now).isoformat(), "nodes": len(nodes), "edges": len(es),
                "by_type": dict(sorted(Counter(n.ntype.value for n in nodes).items())),
                "by_rel": dict(sorted(Counter(e.rel for e in es).items())),
                "components": len(self.components(now)), "head": self.head[:16]}

    def markdown(self, now, top: int = 8) -> str:
        s = self.stats(now)
        iss = self.audit(now)
        pr = sorted(self.pagerank(now).items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        lines = [f"# Knowledge graph as of {s['now']}", "", f"{s['nodes']} nodes, {s['edges']} edges, {s['components']} components",
                 "", "| relation | edges |", "|---|---:|"]
        lines += [f"| {r} | {c} |" for r, c in s["by_rel"].items()]
        lines += ["", "| node | pagerank | support | contradiction |", "|---|---:|---:|---:|"]
        for n, p in pr:
            b = self.support_balance(n, now)
            lines.append(f"| {n} | {p:.4f} | {b['support']:.2f} | {b['contradiction']:.2f} |")
        errs = [i for i in iss if i.severity == "error"]
        lines += ["", f"integrity: {'PASS' if not errs else 'FAIL'} ({len(errs)} errors, {len(iss) - len(errs)} warnings)"]
        return "\n".join(lines)

    def to_dot(self, now) -> str:
        out = ["digraph knowledge {"]
        for n in self.nodes(now):
            out.append(f'  "{n.node_id}" [label="{n.ntype.value}\\n{n.label[:24]}"];')
        for e in self.edges(now):
            out.append(f'  "{e.src}" -> "{e.dst}" [label="{e.rel}"];')
        out.append("}")
        return "\n".join(out)

    def assert_clean(self, now):
        errs = [i for i in self.audit(now) if i.severity == "error"]
        if errs:
            raise FirewallBreach("graph integrity: " + "; ".join(f"{i.code}@{i.where}" for i in errs[:5]))


def record_redundancy(graph: KnowledgeGraph, names: Sequence[str], masks: Sequence[np.ndarray], order: Sequence[int],
                      known_at, max_overlap: float = 0.8, use_containment: bool = False) -> list[GraphEdge]:
    """Keep what engine.pattern_stats.prune_redundant throws away: for every dropped pattern add a REDUNDANT_WITH edge to
    the keeper that displaced it (weight = the overlap that condemned it). Patterns become PATTERN nodes when missing.
    The pruning decision itself stays with the miner; the graph only remembers the relationship."""
    from engine import pattern_stats as ps
    _, dup = ps.prune_redundant(order, masks, max_overlap, use_containment)
    out = []
    for i in list(dup) + list(dict.fromkeys(dup.values())):
        graph.add_node(str(names[i]), NodeType.PATTERN, known_at, str(names[i]))
    for dropped, keeper in sorted(dup.items()):
        ov = ps.containment(masks[dropped], masks[keeper]) if use_containment else ps.jaccard(masks[dropped], masks[keeper])
        out.append(graph.add_edge(str(names[dropped]), str(names[keeper]), Edge.REDUNDANT_WITH, known_at,
                                  weight=min(1.0, float(ov)), attrs={"overlap": round(float(ov), 6),
                                                                      "metric": "containment" if use_containment else "jaccard",
                                                                      "dropped": str(names[dropped])}))
    return out
