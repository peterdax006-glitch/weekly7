"""Research knowledge graph (contract C66 section 27) - IMPLEMENTED, NOT VALIDATED.

EXTENDS engine.learning.knowledge_graph.KnowledgeGraph (the one graph store; this is not a second graph). The base graph has ten
node types and a fixed relation table; section 27 wants patterns, features, stocks, sectors, regimes, events, outcomes, failures,
experiments, learners and research questions related to each other, and the graph must find the relationships nobody has asked
about yet. So:
  * `RKind` is a finer node vocabulary. Each kind is stored as one of the base node types with `attrs.rkind`, so the base
    audit, traversal, PageRank, snapshots and hash-chained persistence all keep working unchanged.
  * `Role` is a typed relationship (PREDICTS, WORKS_IN, FAILS_IN, EXPLAINED_BY, GATED_BY, ...). Each role is stored on one of the
    base relations, with the roles recorded on the edge, and `relate` enforces which kinds a role may join.
  * `discover` finds UNANSWERED relationships: untested transfers (link prediction with a degree-preserving null), context-blind
    patterns, unexplained and ungated failures, untested gates, unanswered questions, unexplained contradictions, one-sided
    evidence, unreplicated patterns, stale patterns and islands. Each gap becomes an identity-free ResearchQuestion.
Time (C56/C58): every query takes `now` and sees only what was known strictly before it. The graph is MATURED_RESEARCH_STATE;
it reaches the trader only through `handoff(...).gate(now)` (C66 section 31). Public entry: `step(graph, now, ...)`."""
from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.archive import _identity_errors
from engine.learning.core import (DecisionEffect, Edge, FailureCause, FirewallBreach, Provenance, _StrEnum, as_date,
                                  canonical_json, current_code_hash, stable_hash)
from engine.learning.knowledge_graph import (LINEAGE_RELS, RELATIONS, GraphEdge, GraphError, KnowledgeGraph, Link, Node,
                                             NodeType, wilson_lower)
from engine.research.core import ExperimentValue, MaturedRecord, Namespace, Problem, ResearchQuestion


class RKind(_StrEnum):
    """Section 27 node vocabulary. Stored as a base NodeType plus attrs.rkind."""
    PATTERN = "PATTERN"
    KNOWLEDGE = "KNOWLEDGE"
    HYPOTHESIS = "HYPOTHESIS"
    FEATURE = "FEATURE"
    STOCK_CLASS = "STOCK_CLASS"          # an identity-free cohort (e.g. small_cap_high_beta), never a ticker
    SECTOR = "SECTOR"
    REGIME = "REGIME"
    EVENT = "EVENT"                      # an event CLASS (earnings_window), never a dated event
    TARGET = "TARGET"                    # what a pattern predicts: volatility, direction, loss
    CONTEXT = "CONTEXT"                  # any other condition
    TEST_BED = "TEST_BED"                # a situation a pattern was tried in (era, cohort, regime slice)
    OUTCOME = "OUTCOME"
    FAILURE = "FAILURE"
    EXPERIMENT = "EXPERIMENT"
    LEARNER = "LEARNER"
    QUESTION = "QUESTION"
    DECISION = "DECISION"


K = RKind
KIND_BASE: dict[RKind, NodeType] = {
    K.PATTERN: NodeType.PATTERN, K.KNOWLEDGE: NodeType.KNOWLEDGE, K.HYPOTHESIS: NodeType.HYPOTHESIS,
    K.QUESTION: NodeType.HYPOTHESIS, K.FEATURE: NodeType.CONDITION, K.STOCK_CLASS: NodeType.CONDITION,
    K.SECTOR: NodeType.CONDITION, K.REGIME: NodeType.CONDITION, K.EVENT: NodeType.CONDITION,
    K.TARGET: NodeType.CONDITION, K.CONTEXT: NodeType.CONDITION, K.TEST_BED: NodeType.SITUATION,
    K.OUTCOME: NodeType.OUTCOME, K.FAILURE: NodeType.FAILURE, K.EXPERIMENT: NodeType.EXPERIMENT,
    K.LEARNER: NodeType.EXPERIMENT, K.DECISION: NodeType.DECISION}
KIND_PREFIX: dict[RKind, str] = {k: k.value.lower()[:4] for k in RKind}
KIND_PREFIX.update({K.STOCK_CLASS: "cls", K.PATTERN: "pat", K.KNOWLEDGE: "know", K.HYPOTHESIS: "hyp", K.FEATURE: "feat",
                    K.SECTOR: "sect", K.REGIME: "reg", K.EVENT: "evt", K.TARGET: "tgt", K.CONTEXT: "ctx", K.TEST_BED: "bed",
                    K.OUTCOME: "out", K.FAILURE: "fail", K.EXPERIMENT: "exp", K.LEARNER: "lrn", K.QUESTION: "qst",
                    K.DECISION: "dec"})
_DEFAULT_KIND = {NodeType.PATTERN: K.PATTERN, NodeType.KNOWLEDGE: K.KNOWLEDGE, NodeType.HYPOTHESIS: K.HYPOTHESIS,
                 NodeType.CONDITION: K.CONTEXT, NodeType.SITUATION: K.TEST_BED, NodeType.OUTCOME: K.OUTCOME,
                 NodeType.FAILURE: K.FAILURE, NodeType.EXPERIMENT: K.EXPERIMENT, NodeType.DECISION: K.DECISION,
                 NodeType.EXPERIENCE: K.OUTCOME}
CONTEXT_KINDS = frozenset({K.REGIME, K.SECTOR, K.EVENT, K.STOCK_CLASS, K.FEATURE, K.CONTEXT})
BELIEF_KINDS = frozenset({K.PATTERN, K.KNOWLEDGE})
_NAME_OK = re.compile(r"^[a-z0-9_.=<>&|:+\-]{1,80}$")


class Role(_StrEnum):
    PREDICTS = "PREDICTS"                    # pattern -> target
    USES_FEATURE = "USES_FEATURE"            # pattern -> feature
    WORKS_IN = "WORKS_IN"                    # pattern -> regime/sector/event/cohort/feature
    FAILS_IN = "FAILS_IN"                    # context or failure -> pattern
    EXPLAINED_BY = "EXPLAINED_BY"            # failure -> the context that explains it
    GATED_BY = "GATED_BY"                    # pattern -> the context that gates it
    RECOVERS_IN = "RECOVERS_IN"              # pattern -> context where it comes back
    VALIDATED_BY_EXP = "VALIDATED_BY_EXP"    # belief -> supporting experiment
    REFUTED_BY_EXP = "REFUTED_BY_EXP"        # belief -> refuting experiment
    PRODUCED_BY = "PRODUCED_BY"              # outcome -> the experiment that produced it
    IMPROVES = "IMPROVES"                    # outcome -> belief it supports
    DEGRADES = "DEGRADES"                    # outcome <-> belief it counts against (symmetric in the base)
    LEARNED_BY = "LEARNED_BY"                # pattern -> learner that found it
    ATTRIBUTED_TO = "ATTRIBUTED_TO"          # failure -> learner / experiment that produced it
    ASKS_ABOUT = "ASKS_ABOUT"                # question -> anything it is about
    ANSWERED_BY = "ANSWERED_BY"              # question -> experiment that answered it
    TRANSFERS_TO = "TRANSFERS_TO"            # belief -> test bed (attrs.success)
    COMPLEMENTS_WITH = "COMPLEMENTS_WITH"    # pattern <-> pattern
    REDUNDANT_WITH = "REDUNDANT_WITH"        # pattern <-> pattern
    CONTRADICTS_WITH = "CONTRADICTS_WITH"    # belief <-> belief
    SPECIALIZES_TO = "SPECIALIZES_TO"        # narrower belief -> broader belief
    CONTAINS_MEMBER = "CONTAINS_MEMBER"      # sector/regime -> narrower context
    USED_IN_DECISION = "USED_IN_DECISION"    # belief -> decision
    RESULTED_IN = "RESULTED_IN"              # decision -> outcome


_TARGETS = frozenset({K.TARGET})


@dataclasses.dataclass(frozen=True)
class RoleSpec:
    base: str
    src: frozenset
    dst: frozenset
    doc: str = ""


R = Role
_CTX_DST = frozenset(CONTEXT_KINDS)
ROLES: dict[Role, RoleSpec] = {
    R.PREDICTS: RoleSpec(Link.APPLIES_IN.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset(_TARGETS),
                         "what the pattern predicts"),
    R.USES_FEATURE: RoleSpec(Edge.DEPENDS_ON.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.FEATURE}),
                             "the inputs a pattern is built from"),
    R.WORKS_IN: RoleSpec(Link.APPLIES_IN.value, frozenset({K.PATTERN, K.KNOWLEDGE}), _CTX_DST, "where it works"),
    R.FAILS_IN: RoleSpec(Edge.CAUSES_FAILURE_OF.value, frozenset(_CTX_DST | {K.FAILURE}),
                         frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS, K.DECISION}), "where or through what it fails"),
    R.EXPLAINED_BY: RoleSpec(Link.DERIVED_FROM.value, frozenset({K.FAILURE}), _CTX_DST, "the context that explains a failure"),
    R.GATED_BY: RoleSpec(Edge.DEPENDS_ON.value, frozenset({K.PATTERN, K.KNOWLEDGE}), _CTX_DST, "a gating rule on a pattern"),
    R.RECOVERS_IN: RoleSpec(Edge.RECOVERS_WITH.value, frozenset({K.PATTERN, K.KNOWLEDGE}), _CTX_DST | {K.TEST_BED}),
    R.VALIDATED_BY_EXP: RoleSpec(Link.VALIDATED_BY.value, frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS}),
                                 frozenset({K.EXPERIMENT})),
    R.REFUTED_BY_EXP: RoleSpec(Link.REFUTED_BY.value, frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS}),
                               frozenset({K.EXPERIMENT})),
    R.PRODUCED_BY: RoleSpec(Link.DERIVED_FROM.value, frozenset({K.OUTCOME}), frozenset({K.EXPERIMENT})),
    R.IMPROVES: RoleSpec(Edge.SUPPORTS.value, frozenset({K.OUTCOME}), frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS})),
    R.DEGRADES: RoleSpec(Edge.CONTRADICTS.value, frozenset({K.OUTCOME}), frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS})),
    R.LEARNED_BY: RoleSpec(Link.DERIVED_FROM.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.LEARNER})),
    R.ATTRIBUTED_TO: RoleSpec(Link.DERIVED_FROM.value, frozenset({K.FAILURE}), frozenset({K.LEARNER, K.EXPERIMENT})),
    R.ASKS_ABOUT: RoleSpec(Link.DERIVED_FROM.value, frozenset({K.QUESTION}), frozenset(RKind) - {K.QUESTION}),
    R.ANSWERED_BY: RoleSpec(Link.VALIDATED_BY.value, frozenset({K.QUESTION}), frozenset({K.EXPERIMENT})),
    R.TRANSFERS_TO: RoleSpec(Link.TRANSFERRED_TO.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.TEST_BED})),
    R.COMPLEMENTS_WITH: RoleSpec(Edge.COMPLEMENTS.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.PATTERN, K.KNOWLEDGE})),
    R.REDUNDANT_WITH: RoleSpec(Edge.REDUNDANT_WITH.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.PATTERN, K.KNOWLEDGE})),
    R.CONTRADICTS_WITH: RoleSpec(Edge.CONTRADICTS.value, frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS}),
                                 frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS})),
    R.SPECIALIZES_TO: RoleSpec(Edge.SPECIALIZES.value, frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS}),
                               frozenset({K.PATTERN, K.KNOWLEDGE, K.HYPOTHESIS})),
    R.CONTAINS_MEMBER: RoleSpec(Edge.CONTAINS.value, frozenset({K.SECTOR, K.REGIME, K.STOCK_CLASS, K.CONTEXT}),
                                frozenset({K.STOCK_CLASS, K.EVENT, K.FEATURE, K.CONTEXT, K.REGIME})),
    R.USED_IN_DECISION: RoleSpec(Link.USED_IN.value, frozenset({K.PATTERN, K.KNOWLEDGE}), frozenset({K.DECISION})),
    R.RESULTED_IN: RoleSpec(Link.RESULTED_IN.value, frozenset({K.DECISION}), frozenset({K.OUTCOME}))}


def _base_of(role: Role) -> str:
    return ROLES[role].base


def clean_name(name: str) -> str:
    """Lower-case, restricted-alphabet node names so ids are stable and never carry spaces or quotes."""
    s = re.sub(r"\s+", "_", str(name).strip().lower())
    s = re.sub(r"[^a-z0-9_.=<>&|:+\-]", "", s)
    if not _NAME_OK.match(s):
        raise GraphError(f"node name {name!r} is empty or too long after cleaning")
    return s


def node_key(kind: RKind, name: str) -> str:
    """Canonical node id for a kind and a name: 'reg:high_vol'."""
    return f"{KIND_PREFIX[kind]}:{clean_name(name)}"


@dataclasses.dataclass(frozen=True)
class PatternStory:
    """The section-27 example chain for one pattern: predicts -> works -> fails -> explained -> gated -> gate tested -> OOS gain.
    Each step lists the node ids that carry it; `missing` names the steps the graph cannot yet support."""
    pattern: str
    predicts: tuple[str, ...]
    works_in: tuple[str, ...]
    fails_in: tuple[str, ...]
    failure_explanations: tuple[tuple[str, tuple[str, ...]], ...]
    gates: tuple[str, ...]
    gate_tests: tuple[tuple[str, str, bool], ...]          # (gate, experiment, supports)
    improvements: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def text(self) -> str:
        parts = [f"{self.pattern}: predicts {', '.join(self.predicts) or '?'}",
                 f"works in {', '.join(self.works_in) or 'no recorded context'}",
                 f"fails in {', '.join(self.fails_in) or 'no recorded context'}"]
        for f, ex in self.failure_explanations:
            parts.append(f"failure {f} explained by {', '.join(ex) if ex else 'nothing (cause unknown)'}")
        parts.append("gated by " + (", ".join(self.gates) or "nothing"))
        for g, e, ok in self.gate_tests:
            parts.append(f"gate {g} {'validated' if ok else 'refuted'} by {e}")
        parts.append("out-of-sample improvement: " + (", ".join(self.improvements) or "none recorded"))
        return "; ".join(parts) + (f" [missing: {', '.join(self.missing)}]" if self.missing else " [complete]")


class ResearchGraph(KnowledgeGraph):
    """KnowledgeGraph plus the section-27 vocabulary, typed roles, ingestion adapters and gap discovery."""

    namespace = Namespace.MATURED_RESEARCH

    def __init__(self, path=None, forbidden_identities: Iterable[str] = ()):
        super().__init__(path, forbidden_identities)
        self._kinds: dict[str, tuple[int, RKind]] = {}

    # ------------------------------------------------------------------ kinds
    def kind_of(self, node_id: str) -> RKind | None:
        """Research kind of a node (latest version); nodes written by the base graph get the default kind of their type."""
        vs = self._nodes.get(node_id)
        if not vs:
            return None
        hit = self._kinds.get(node_id)
        if hit is not None and hit[0] == len(vs):
            return hit[1]
        a = vs[-1].attrs
        kind = RKind(a["rkind"]) if "rkind" in a else _DEFAULT_KIND[vs[-1].ntype]
        self._kinds[node_id] = (len(vs), kind)
        return kind

    def nodes_of(self, kind: RKind | str, now) -> list[Node]:
        k = RKind.parse(kind)
        return [n for n in self.nodes(now) if self.kind_of(n.node_id) == k]

    def add_research_node(self, kind: RKind | str, name: str, known_at, label: str = "", attrs: Mapping | None = None) -> Node:
        """Add (or version) a node of a research kind. Every kind must be identity-free: no tickers, names, dates or years."""
        k = RKind.parse(kind)
        nid = name if (":" in name and name.split(":", 1)[0] == KIND_PREFIX[k]) else node_key(k, name)
        a = dict(attrs or {})
        a["rkind"] = k.value
        errs = _identity_errors({"id": nid, "label": label, **a}, self.forbidden, "node")
        if errs:
            raise GraphError(f"{k.value} node {nid}: " + "; ".join(errs))
        cur = self._nodes.get(nid)
        if cur and self.kind_of(nid) != k:
            raise GraphError(f"node {nid} is {self.kind_of(nid).value}; cannot become {k.value}")
        return self.add_node(nid, KIND_BASE[k], known_at, label or nid.split(":", 1)[1], a)

    def merge_attrs(self, node_id: str, known_at, **more) -> Node:
        """New version of a node with extra attributes (history keeps the old version)."""
        vs = self._nodes.get(node_id)
        if not vs:
            raise GraphError(f"unknown node {node_id}")
        cur = vs[-1]
        return self.add_research_node(self.kind_of(node_id), node_id, known_at, cur.label, {**cur.attrs, **more})

    # ------------------------------------------------------------------ roles
    def roles_of(self, edge: GraphEdge) -> frozenset[Role]:
        """Roles carried by an edge; edges written by the base graph get every role their endpoint kinds allow."""
        stored = edge.attrs.get("roles")
        if stored:
            return frozenset(Role(r) for r in stored)
        ks, kd = self.kind_of(edge.src), self.kind_of(edge.dst)
        out = {r for r, s in ROLES.items() if s.base == edge.rel and ks in s.src and kd in s.dst}
        if RELATIONS[edge.rel].symmetric:
            out |= {r for r, s in ROLES.items() if s.base == edge.rel and kd in s.src and ks in s.dst}
        return frozenset(out)

    def relate(self, src: str, dst: str, role: Role | str, known_at, weight: float = 1.0, evidence: Sequence[str] = (),
               attrs: Mapping | None = None) -> GraphEdge:
        """Record one typed relationship. The role decides which base relation carries it and which kinds it may join;
        a second role on the same pair merges into the edge (roles and per-role weights are kept, the edge weight is the max)."""
        rl = Role.parse(role)
        spec = ROLES[rl]
        ks, kd = self.kind_of(src), self.kind_of(dst)
        if ks is None or kd is None:
            raise GraphError(f"{rl.value}: unknown node {src if ks is None else dst}")
        sym = RELATIONS[spec.base].symmetric
        if not ((ks in spec.src and kd in spec.dst) or (sym and kd in spec.src and ks in spec.dst)):
            raise GraphError(f"{rl.value} cannot join {ks.value} to {kd.value}")
        a, b = (dst, src) if sym and src > dst else (src, dst)
        cur = self._edges.get((a, b, spec.base))
        live = cur[-1] if cur and not cur[-1].retracted else None
        old = live.attrs if live else {}
        roles = sorted(set(old.get("roles", [])) | {rl.value})
        role_w = {**old.get("role_weights", {}), rl.value: float(weight)}
        by_role = {**old.get("by_role", {}), rl.value: dict(attrs or {})}
        merged = {**{k: v for k, v in old.items() if k not in ("roles", "role_weights", "by_role")}, **dict(attrs or {}),
                  "roles": roles, "role_weights": role_w, "by_role": by_role}
        ev = tuple(dict.fromkeys(tuple(live.evidence if live else ()) + tuple(evidence)))
        return self.add_edge(src, dst, spec.base, known_at, max(role_w.values()), ev, merged)

    def by_role(self, node: str, role: Role | str, now, direction: str = "out") -> list[tuple[str, GraphEdge]]:
        """(other node, edge) pairs where the edge carries `role` and is live at `now`; 'out' = node is the role's source."""
        rl = Role.parse(role)
        return [(o, e) for o, e in self.neighbors(node, now, [ROLES[rl].base], direction) if rl in self.roles_of(e)]

    def role_weight(self, edge: GraphEdge, role: Role | str) -> float:
        return float(edge.attrs.get("role_weights", {}).get(Role.parse(role).value, edge.weight))

    def role_attrs(self, edge: GraphEdge, role: Role | str) -> dict:
        by = edge.attrs.get("by_role", {})
        return dict(by.get(Role.parse(role).value, {})) if by else {k: v for k, v in edge.attrs.items()
                                                                      if k not in ("roles", "role_weights", "by_role")}

    def targets(self, node: str, role: Role | str, now) -> list[str]:
        return sorted(o for o, _ in self.by_role(node, role, now, "out"))

    def sources(self, node: str, role: Role | str, now) -> list[str]:
        return sorted(o for o, _ in self.by_role(node, role, now, "in"))

    # ------------------------------------------------------------------ typed adders
    def add_pattern(self, name: str, known_at, label: str = "", effect: float | None = None, p_real: float | None = None,
                    status: str = "", n: int | None = None, target: str | None = None) -> Node:
        attrs: dict[str, Any] = {}
        for k, v in (("effect", effect), ("p_real", p_real), ("status", status or None), ("n", n)):
            if v is not None:
                attrs[k] = float(v) if isinstance(v, (float, np.floating)) else v
        node = self.add_research_node(K.PATTERN, name, known_at, label, attrs)
        if target:
            tid = self.add_research_node(K.TARGET, target, known_at).node_id
            self.relate(node.node_id, tid, R.PREDICTS, known_at)
        return node

    def add_context(self, kind: RKind | str, name: str, known_at, label: str = "", attrs: Mapping | None = None) -> Node:
        k = RKind.parse(kind)
        if k not in CONTEXT_KINDS:
            raise GraphError(f"{k.value} is not a context kind")
        return self.add_research_node(k, name, known_at, label, attrs)

    def add_failure(self, failure_id: str, pattern: str, cause: FailureCause | str, known_at, contexts: Sequence[str] = (),
                    weight: float = 1.0, learner: str | None = None, note: str = "", evidence: Sequence[str] = ()) -> Node:
        """A failure of a pattern. An asserted cause needs at least one explaining context or evidence (UNKNOWN needs neither and
        stays a visible gap: the honest answer is never forced into a label)."""
        c = FailureCause.parse(cause)
        if c != FailureCause.UNKNOWN and not (contexts or evidence or note):
            raise GraphError(f"failure {failure_id}: cause {c.value} asserted with no explaining context, note or evidence")
        node = self.add_research_node(K.FAILURE, failure_id, known_at, attrs={"cause": c.value, "note": note})
        self.relate(node.node_id, pattern, R.FAILS_IN, known_at, weight, evidence, {"cause": c.value})
        for ctx in contexts:
            self.relate(node.node_id, ctx, R.EXPLAINED_BY, known_at, weight, evidence)
        if learner:
            self.relate(node.node_id, learner, R.ATTRIBUTED_TO, known_at)
        return node

    def add_test(self, experiment_id: str, belief: str, supports: bool, known_at, effect: float | None = None, method: str = "",
                 weight: float = 1.0, gate: str | None = None, outcome: str | None = None) -> Node:
        """An experiment on a belief: VALIDATED_BY (supports) or REFUTED_BY (against). `gate` marks it as a test of a gating rule;
        `outcome` names an OUTCOME node it produced (which then IMPROVES or DEGRADES the belief)."""
        attrs: dict[str, Any] = {"method": method}
        if effect is not None:
            attrs["effect"] = float(effect)
        exp = self.add_research_node(K.EXPERIMENT, experiment_id, known_at, attrs=attrs)
        ea = {"gate": gate} if gate else {}
        self.relate(belief, exp.node_id, R.VALIDATED_BY_EXP if supports else R.REFUTED_BY_EXP, known_at, weight, attrs=ea)
        if outcome:
            oid = self.add_research_node(K.OUTCOME, outcome, known_at, attrs={"effect": effect} if effect is not None else {})
            self.relate(oid.node_id, exp.node_id, R.PRODUCED_BY, known_at)
            self.relate(oid.node_id, belief, R.IMPROVES if supports else R.DEGRADES, known_at, weight)
        return exp

    def add_transfer(self, belief: str, test_bed: str, success: bool, known_at, axis: str = "", weight: float = 1.0) -> GraphEdge:
        tb = self.add_research_node(K.TEST_BED, test_bed, known_at, attrs={"axis": axis} if axis else {})
        return self.relate(belief, tb.node_id, R.TRANSFERS_TO, known_at, weight, attrs={"success": bool(success), "axis": axis})

    def add_question(self, q: ResearchQuestion, about: Sequence[str], known_at, extra: Mapping | None = None) -> Node:
        """A research question as a node; identity-free by construction. `about` are the nodes it asks about."""
        node = self.add_research_node(K.QUESTION, q.question_id.lower(), known_at, q.text[:120],
                                      {"source": q.source, "problem": q.problem.value, "success": q.success_criterion,
                                       "failure": q.failure_criterion, **dict(extra or {})})
        for a in about:
            if self.kind_of(a) is not None and self.kind_of(a) != K.QUESTION:
                self.relate(node.node_id, a, R.ASKS_ABOUT, known_at)
        return node

    def answer_question(self, question_id: str, experiment_id: str, known_at, verdict: str = "answered") -> GraphEdge:
        exp = self.add_research_node(K.EXPERIMENT, experiment_id, known_at)
        return self.relate(question_id, exp.node_id, R.ANSWERED_BY, known_at, attrs={"verdict": verdict})

    def add_gate(self, pattern: str, context: str, known_at, weight: float = 1.0, rule: str = "") -> GraphEdge:
        return self.relate(pattern, context, R.GATED_BY, known_at, weight, attrs={"rule": rule} if rule else None)

    # ------------------------------------------------------------------ adapters from the rest of the system
    def resolve(self, subject: str, kind: RKind = K.PATTERN) -> str:
        """Node id for a subject named elsewhere (archive subject, knowledge id): reuse an existing node of that exact id,
        otherwise the canonical research id for `kind`."""
        if subject in self._nodes:
            return subject
        k = node_key(kind, subject.split(":", 1)[1] if subject.startswith(KIND_PREFIX[kind] + ":") else subject)
        return k

    def ingest_patterns(self, table, known_at, target: str | None = None, learner: str | None = None) -> dict[str, int]:
        """PatternMiner.patterns rows -> PATTERN nodes (+ FEATURE nodes and USES_FEATURE edges parsed from key_named, + PREDICTS
        to `target`, + LEARNED_BY to `learner`). Reuses engine.pattern_lifecycle's parser and id so a pattern has ONE identity."""
        from engine.pattern_lifecycle import parse_key_named, pattern_id
        counts: Counter = Counter()
        if table is None or len(table) == 0:
            return dict(counts)
        tid = self.add_research_node(K.TARGET, target, known_at).node_id if target else None
        lid = self.add_research_node(K.LEARNER, learner, known_at).node_id if learner else None
        for row in table.itertuples(index=False):
            key = str(getattr(row, "key_named"))
            try:
                names = parse_key_named(key)
            except ValueError:
                counts["unparseable"] += 1
                continue
            lits = list(zip(names[1::2], names[2::2]))
            attrs = {k: (float(getattr(row, k)) if k != "status" else str(getattr(row, k)))
                     for k in ("effect", "t_disc", "t_conf", "p_real", "status") if hasattr(row, k)
                     and not (isinstance(getattr(row, k), float) and math.isnan(getattr(row, k)))}
            attrs["shape"] = names[0]
            pnode = self.add_research_node(K.PATTERN, "pat:" + pattern_id(names[1:]), known_at, key, attrs)
            counts["patterns"] += 1
            for i, (feat, q) in enumerate(lits):
                fnode = self.add_research_node(K.FEATURE, feat, known_at)
                negated = names[0] == "u" and i == len(lits) - 1
                self.relate(pnode.node_id, fnode.node_id, R.USES_FEATURE, known_at, attrs={"quintile": int(q), "negated": negated})
                counts["features"] += 1
            if tid:
                self.relate(pnode.node_id, tid, R.PREDICTS, known_at)
            if lid:
                self.relate(pnode.node_id, lid, R.LEARNED_BY, known_at)
        return dict(counts)

    _DIM_KIND = {"regime": K.REGIME, "sector": K.SECTOR, "event": K.EVENT, "cohort": K.STOCK_CLASS, "stock_class": K.STOCK_CLASS,
                 "size": K.STOCK_CLASS, "feature": K.FEATURE}

    def context_node(self, dimension: str, value: Any, known_at) -> str:
        """Context node for a (dimension, value) pair, typed by the dimension name (regime -> REGIME, sector -> SECTOR, ...)."""
        kind = self._DIM_KIND.get(str(dimension).lower(), K.CONTEXT)
        return self.add_research_node(kind, f"{dimension}={value}", known_at).node_id

    def ingest_archive_failures(self, archive, now) -> dict[str, int]:
        """archive `failure` records -> FAILURE nodes joined to their subject. The record's contexts become the explaining
        contexts unless the cause is UNKNOWN (unknown stays unexplained: a gap, not a guess). Idempotent."""
        counts: Counter = Counter()
        for r in sorted(archive.failures(now), key=lambda r: (r.matured_at, r.seq)):
            subject = self.resolve(r.subject, K.PATTERN)
            if subject not in self._nodes:
                self.add_research_node(K.PATTERN, subject, r.matured_at)
            cause = FailureCause.parse(r.payload["cause"])
            ctxs = [] if cause == FailureCause.UNKNOWN else [self.context_node(d, v, r.matured_at) for d, v in r.contexts]
            self.add_failure(r.rec_id, subject, cause, r.matured_at, ctxs, note="" if ctxs or cause == FailureCause.UNKNOWN
                                   else "archived without context").node_id
            counts["failures"] += 1
            counts["explained"] += int(bool(ctxs))
        return dict(counts)

    def ingest_knowledge(self, k, known_at) -> Node:
        """A KnowledgeLike object: base add_knowledge (its contexts / anti-contexts become APPLIES_IN / CAUSES_FAILURE_OF), plus
        the decisions it declares as DECISION-effect notes so the bridge can find it."""
        node = self.add_knowledge(k, known_at)
        fx = [str(e) for e in getattr(k, "decision_effect", ()) if str(e) != DecisionEffect.NONE.value]
        if fx:
            return self.add_node(node.node_id, NodeType.KNOWLEDGE, known_at, node.label, {**node.attrs, "effects": sorted(fx)})
        return node

    def add_learner(self, name: str, known_at, family: str = "", attrs: Mapping | None = None) -> Node:
        return self.add_research_node(K.LEARNER, name, known_at, attrs={**dict(attrs or {}), **({"family": family} if family else {})})

    # ------------------------------------------------------------------ per-pattern views
    def contexts_of(self, pattern: str, now) -> dict[str, dict[str, int]]:
        """Every context a pattern has been observed in, with counts of where it worked and where it failed. Failure-explaining
        contexts and test beds count; a context with only a failure record is a failing context, never 'unknown'."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"works": 0, "fails": 0})
        for c in self.targets(pattern, R.WORKS_IN, now):
            out[c]["works"] += 1
        for src, e in self.by_role(pattern, R.FAILS_IN, now, "in"):
            if self.kind_of(src) in CONTEXT_KINDS:
                out[src]["fails"] += 1
            elif self.kind_of(src) == K.FAILURE:
                for c in self.targets(src, R.EXPLAINED_BY, now):
                    out[c]["fails"] += 1
        for bed, e in self.by_role(pattern, R.TRANSFERS_TO, now, "out"):
            ok = bool(self.role_attrs(e, R.TRANSFERS_TO).get("success", True))
            out[bed]["works" if ok else "fails"] += 1
        return {c: dict(v) for c, v in sorted(out.items())}

    def recorded_gap_ids(self) -> set[str]:
        """gap ids that already have a QUESTION node in any version and at any date (so a rerun never files a duplicate)."""
        return {json.loads(v.attrs_json).get("gap_id") for vs in self._nodes.values() for v in vs
                if v.ntype == NodeType.HYPOTHESIS and "gap_id" in v.attrs_json} - {None}

    def failures_of(self, pattern: str, now) -> list[str]:
        return sorted(s for s in self.sources(pattern, R.FAILS_IN, now) if self.kind_of(s) == K.FAILURE)

    def unexplained_failures(self, now, pattern: str | None = None) -> list[str]:
        """FAILURE nodes with no explaining context (cause UNKNOWN, or asserted from a note alone)."""
        pool = self.failures_of(pattern, now) if pattern else [n.node_id for n in self.nodes_of(K.FAILURE, now)]
        return sorted(f for f in pool if not self.targets(f, R.EXPLAINED_BY, now))

    def evidence_counts(self, pattern: str, now) -> dict[str, int]:
        return {"validating": len(self.by_role(pattern, R.VALIDATED_BY_EXP, now)),
                "refuting": len(self.by_role(pattern, R.REFUTED_BY_EXP, now)),
                "transfers_ok": sum(1 for _, e in self.by_role(pattern, R.TRANSFERS_TO, now)
                                    if self.role_attrs(e, R.TRANSFERS_TO).get("success", True)),
                "transfers_failed": sum(1 for _, e in self.by_role(pattern, R.TRANSFERS_TO, now)
                                        if not self.role_attrs(e, R.TRANSFERS_TO).get("success", True)),
                "improvements": len(self.sources(pattern, R.IMPROVES, now)),
                "degradations": len(self.by_role(pattern, R.DEGRADES, now, "both")),
                "failures": len(self.failures_of(pattern, now))}

    def gate_tests(self, pattern: str, now) -> list[tuple[str, str, bool]]:
        """(gate context, experiment, supports) for every experiment on the pattern that is marked as a test of a gate."""
        out = []
        for role, ok in ((R.VALIDATED_BY_EXP, True), (R.REFUTED_BY_EXP, False)):
            for exp, e in self.by_role(pattern, role, now, "out"):
                g = self.role_attrs(e, role).get("gate")
                if g:
                    out.append((str(g), exp, ok))
        return sorted(out)

    def pattern_story(self, pattern: str, now) -> PatternStory:
        """The section-27 chain (predicts, works, fails, explained, gated, gate tested, out-of-sample gain) for one pattern,
        with every missing link named. Steps that cannot exist yet (a gate test with no gate) are not listed as missing."""
        if self.kind_of(pattern) not in BELIEF_KINDS or self.node_at(pattern, now) is None:
            raise GraphError(f"{pattern} is not a pattern or knowledge node known at {as_date(now)}")
        ctx = self.contexts_of(pattern, now)
        predicts = tuple(self.targets(pattern, R.PREDICTS, now))
        works = tuple(c for c, v in ctx.items() if v["works"])
        fails = tuple(c for c, v in ctx.items() if v["fails"])
        expl = tuple((f, tuple(self.targets(f, R.EXPLAINED_BY, now))) for f in self.failures_of(pattern, now))
        gates = tuple(self.targets(pattern, R.GATED_BY, now))
        tests = tuple(self.gate_tests(pattern, now))
        gate_exps = {e for _, e, _ in tests}
        improving = self.sources(pattern, R.IMPROVES, now)
        imp = tuple(o for o in improving if not gate_exps or gate_exps & set(self.targets(o, R.PRODUCED_BY, now)))
        missing = [s for s, present in (("predicts", predicts), ("works_in", works), ("fails_in", fails)) if not present]
        if fails and not any(ex for _, ex in expl):
            missing.append("failure_explanation")
        if any(ex for _, ex in expl) and not gates:
            missing.append("gating_rule")
        if gates and not tests:
            missing.append("gate_test")
        if tests and not imp:
            missing.append("oos_improvement")
        return PatternStory(pattern, predicts, works, fails, expl, gates, tests, imp, tuple(missing))

    def coverage_matrix(self, now, kinds: Iterable[RKind] | None = None):
        """patterns x contexts: 1 = works, -1 = fails, 0 = both seen, NaN = never tested. The grid discovery reads."""
        import pandas as pd
        pats = [n.node_id for n in self.nodes(now) if self.kind_of(n.node_id) in BELIEF_KINDS]
        allow = set(kinds) if kinds is not None else CONTEXT_KINDS | {K.TEST_BED}
        rows = {p: {c: (1.0 if v["works"] and not v["fails"] else -1.0 if v["fails"] and not v["works"] else 0.0)
                    for c, v in self.contexts_of(p, now).items() if self.kind_of(c) in allow} for p in pats}
        cols = sorted({c for r in rows.values() for c in r})
        return pd.DataFrame([[rows[p].get(c, np.nan) for c in cols] for p in pats], index=pats, columns=cols, dtype=float)

    def importance(self, now) -> dict[str, float]:
        """0-1 importance of every belief node: half PageRank percentile, half |effect| relative to twice the median effect.
        Used only to order investigations; it is never evidence about truth."""
        pr = self.pagerank(now)
        beliefs = [n for n in self.nodes(now) if self.kind_of(n.node_id) in BELIEF_KINDS]
        if not beliefs:
            return {}
        vals = np.array([pr.get(n.node_id, 0.0) for n in beliefs])
        order = vals.argsort().argsort() / max(1, len(vals) - 1)
        effs = np.array([abs(float(n.attrs.get("effect", 0.0) or 0.0)) for n in beliefs])
        scale = 2.0 * float(np.median(effs[effs > 0])) if (effs > 0).any() else 1.0
        return {n.node_id: float(0.5 * order[i] + 0.5 * min(1.0, effs[i] / scale)) for i, n in enumerate(beliefs)}

    def handoff(self, node: str, now, created_real: str) -> MaturedRecord:
        """The ONLY way graph knowledge leaves the research namespace: an identity-free summary as a MaturedRecord whose
        `matured_at` is the newest evidence in it. The trader must call `.gate(its_now)`; a summary dated on or after the
        trader's now is refused there (C66 sections 29, 31)."""
        from engine.learning import trader_view
        nd = self.node_at(node, now)
        if nd is None:
            raise FirewallBreach(f"{node} is not known before {as_date(now)}")
        kind = self.kind_of(node)
        newest = [nd.known_at]
        payload: dict[str, Any] = {"kind": kind.value if kind else "?", "contexts": {}, "counts": {}}
        if kind in BELIEF_KINDS:
            payload["contexts"] = self.contexts_of(node, now)
            payload["counts"] = self.evidence_counts(node, now)
            payload["story_missing"] = list(self.pattern_story(node, now).missing)
            newest += [e.known_at for _, e in self.neighbors(node, now, None, "both")]
        trader_view.assert_trader_safe(payload, f"graph handoff {node}")
        matured = max(newest, key=as_date)
        prov = Provenance(created_real=created_real, learned_at=matured, code_hash=current_code_hash(), outcomes_seen_through=matured)
        return MaturedRecord(stable_hash([node, payload], 16), matured, payload, prov)


# ====================================================================================================== gap discovery

RETIRED_STATUSES = frozenset({"retired", "discarded", "rejected", "no_gain", "duplicate", "disregarded"})
_MATRIX_KINDS = frozenset({K.REGIME, K.SECTOR, K.EVENT, K.STOCK_CLASS, K.CONTEXT, K.TEST_BED})


class GapKind(_StrEnum):
    PREDICTED_TRANSFER = "PREDICTED_TRANSFER"          # similar patterns work in a context this one was never tried in
    PREDICTED_EXPOSURE = "PREDICTED_EXPOSURE"          # similar patterns fail in a context this one was never tried in
    CONTEXT_BLIND = "CONTEXT_BLIND"                    # predicts a target but no context is known either way
    UNEXPLAINED_FAILURE = "UNEXPLAINED_FAILURE"
    UNGATED_FAILURE = "UNGATED_FAILURE"                # a failure is explained by a context but nothing gates on it
    UNTESTED_GATE = "UNTESTED_GATE"
    UNANSWERED_QUESTION = "UNANSWERED_QUESTION"
    UNEXPLAINED_CONTRADICTION = "UNEXPLAINED_CONTRADICTION"
    ONE_SIDED_EVIDENCE = "ONE_SIDED_EVIDENCE"          # many wins, never a failure, never attacked
    UNREPLICATED = "UNREPLICATED"
    STALE = "STALE"
    ISLAND = "ISLAND"                                  # a cluster of beliefs never compared with the main body


@dataclasses.dataclass(frozen=True)
class GapConfig:
    alpha: float = 0.10                # BH q-value for predicted-transfer cells
    n_null: int = 200
    min_similarity: float = 0.25
    min_support: int = 2               # similar patterns tested in the context
    work_threshold: float = 0.70       # weighted share of similar patterns that must agree
    shrink: float = 1.0                # pseudo-mass that keeps thin evidence from scoring high
    stale_days: int = 180
    min_wins_one_sided: int = 3
    max_per_kind: int = 10
    seed: int = 0

    def check(self) -> list[str]:
        errs = []
        if not 0 < self.alpha < 1:
            errs.append("alpha must be in (0,1)")
        if self.n_null < 19:
            errs.append("n_null < 19 cannot resolve p <= 0.05")
        if not 0.5 < self.work_threshold < 1:
            errs.append("work_threshold must be in (0.5,1)")
        if self.min_support < 1 or self.max_per_kind < 1 or self.shrink <= 0:
            errs.append("min_support/max_per_kind >= 1 and shrink > 0 required")
        return errs


@dataclasses.dataclass(frozen=True)
class Gap:
    gap_id: str
    kind: GapKind
    subjects: tuple[str, ...]
    score: float                                  # 0-1 priority for investigation, never a probability of truth
    text: str
    problem: Problem
    expected: ExperimentValue
    known_through: str                            # newest known_at of the evidence the gap rests on
    p_value: float | None = None
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _gap(kind: GapKind, subjects: Sequence[str], score: float, text: str, problem: Problem, known_through: str, *,
         p: float | None = None, ev: Mapping | None = None, **value) -> Gap:
    subj = tuple(subjects)
    return Gap(stable_hash([kind.value, subj], 12), kind, subj, float(min(1.0, max(0.0, score))), text, problem,
               ExperimentValue(**value), known_through, p, dict(ev or {}))


_TARGET_PROBLEM = (("vol", Problem.VOLATILITY), ("dir", Problem.DIRECTION), ("loss", Problem.LOSS_AVOIDANCE),
                   ("draw", Problem.LOSS_AVOIDANCE), ("risk", Problem.LOSS_AVOIDANCE), ("cover", Problem.COVERAGE))


def problem_of(g: ResearchGraph, node: str, now) -> Problem:
    """Which of the two prediction problems a node serves, read from the targets its pattern predicts."""
    for t in g.targets(node, R.PREDICTS, now):
        for stem, prob in _TARGET_PROBLEM:
            if stem in t:
                return prob
    return Problem.RESEARCH_PROCESS


def _newest(g: ResearchGraph, nodes: Iterable[str], now) -> str:
    dates = []
    for n in nodes:
        nd = g.node_at(n, now)
        if nd is not None:
            dates.append(nd.known_at)
            dates += [e.known_at for _, e in g.neighbors(n, now, None, "both")]
    return max(dates, key=as_date) if dates else as_date(now).isoformat()


def _active_beliefs(g: ResearchGraph, now) -> list[str]:
    out = []
    for n in g.nodes(now):
        if g.kind_of(n.node_id) in BELIEF_KINDS and str(n.attrs.get("status", "")).lower() not in RETIRED_STATUSES \
                and str(n.attrs.get("lifecycle", "")).upper() != "RETIRED":
            out.append(n.node_id)
    return out


def _feature_profiles(g: ResearchGraph, pats: Sequence[str], now) -> dict[str, set[str]]:
    prof = {}
    for p in pats:
        prof[p] = set(g.targets(p, R.USES_FEATURE, now)) | set(g.targets(p, R.PREDICTS, now)) | set(g.targets(p, R.LEARNED_BY, now))
    return prof


def similarity_matrix(g: ResearchGraph, pats: Sequence[str], now, min_similarity: float = 0.0) -> np.ndarray:
    """Pairwise pattern similarity from what each is MADE of and FOR (features, predicted target, learner), IDF-weighted so a
    feature everybody uses says little. Contexts are deliberately not used: the question is whether construction predicts
    behaviour in a context, so the behaviour itself cannot also define similarity."""
    prof = _feature_profiles(g, pats, now)
    df = Counter(f for s in prof.values() for f in s)
    w = {f: 1.0 / math.log(2.0 + n) for f, n in df.items()}
    n = len(pats)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = prof[pats[i]], prof[pats[j]]
            if not a or not b:
                continue
            uni = sum(w[x] for x in a | b)
            S[i, j] = S[j, i] = (sum(w[x] for x in a & b) / uni) if uni > 0 else 0.0
    S[S < min_similarity] = 0.0
    return S


def _grid(g: ResearchGraph, pats: Sequence[str], now) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(context ids, T tested, W works) over the given patterns. A cell works when it has more successes than failures."""
    rows = [g.contexts_of(p, now) for p in pats]
    cols = sorted({c for r in rows for c in r if g.kind_of(c) in _MATRIX_KINDS})
    cx = {c: i for i, c in enumerate(cols)}
    T = np.zeros((len(pats), len(cols)))
    W = np.zeros_like(T)
    for i, r in enumerate(rows):
        for c, v in r.items():
            if c in cx:
                T[i, cx[c]] = 1.0
                W[i, cx[c]] = 1.0 if v["works"] > v["fails"] else 0.0
    return cols, T, W


def predicted_transfers(g: ResearchGraph, now, cfg: GapConfig = GapConfig(), imp: Mapping[str, float] | None = None) -> list[Gap]:
    """Link prediction with a null. For every untested (pattern, context) cell: the similarity-weighted share rho of tested
    similar patterns that work there. A cell is a gap only if rho is extreme AND beats a null in which the works/fails labels
    are permuted within each context (keeping every context's base rate, destroying 'similar patterns behave alike'), after
    Benjamini-Hochberg over all candidate cells. A graph with no real structure therefore yields nothing."""
    from engine import pattern_stats as ps
    pats = _active_beliefs(g, now)
    if len(pats) < cfg.min_support + 1:
        return []
    cols, T, W = _grid(g, pats, now)
    if not cols:
        return []
    imp = imp if imp is not None else g.importance(now)
    S = similarity_matrix(g, pats, now, cfg.min_similarity)
    den = S @ T
    support = (S > 0).astype(float) @ T
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, (S @ W) / np.where(den > 0, den, 1.0), np.nan)
    cand = (T == 0) & (support >= cfg.min_support) & (den > 0)
    hi, lo = cfg.work_threshold, 1.0 - cfg.work_threshold
    cells = [(i, j) for i, j in zip(*np.nonzero(cand)) if rho[i, j] >= hi or rho[i, j] <= lo]
    if not cells:
        return []
    rng = np.random.default_rng(cfg.seed)
    ge = np.zeros(len(cells))
    le = np.zeros(len(cells))
    ii = np.array([c[0] for c in cells])
    jj = np.array([c[1] for c in cells])
    tested_rows = [np.flatnonzero(T[:, j]) for j in range(len(cols))]
    for _ in range(cfg.n_null):
        Wn = W.copy()
        for j, rows in enumerate(tested_rows):
            if len(rows) > 1:
                Wn[rows, j] = rng.permutation(W[rows, j])
        rn = (S @ Wn)[ii, jj] / den[ii, jj]
        r0 = rho[ii, jj]
        ge += rn >= r0 - 1e-12
        le += rn <= r0 + 1e-12
    r0 = rho[ii, jj]
    pv = np.where(r0 >= hi, (1 + ge) / (cfg.n_null + 1), (1 + le) / (cfg.n_null + 1))
    keep = ps.bh_reject(pv, cfg.alpha)
    out = []
    for (i, j), p, k in zip(cells, pv, keep):
        if not k:
            continue
        pat, ctx = pats[i], cols[j]
        works = rho[i, j] >= hi
        mass = float(den[i, j])
        conf = mass / (mass + cfg.shrink) * abs(2 * rho[i, j] - 1)
        score = conf * (0.5 + 0.5 * imp.get(pat, 0.0))
        n_sup = int(support[i, j])
        if works:
            text = (f"Does {pat} also work in {ctx}? {n_sup} similar patterns were tested there and {rho[i, j]:.0%} of the "
                    f"similarity-weighted evidence says they work; {pat} has never been tried there.")
            kind = GapKind.PREDICTED_TRANSFER
        else:
            text = (f"Is {pat} exposed to the failure seen in {ctx}? {n_sup} similar patterns were tested there and only "
                    f"{rho[i, j]:.0%} of the weighted evidence says they work; {pat} has never been tried there.")
            kind = GapKind.PREDICTED_EXPOSURE
        out.append(_gap(kind, (pat, ctx), score, text, problem_of(g, pat, now), _newest(g, [pat, ctx], now), p=float(p),
                        ev={"rho": float(rho[i, j]), "mass": mass, "support": n_sup, "q_alpha": cfg.alpha},
                        information_gain=float(conf), transfer_potential=float(rho[i, j]) if works else None,
                        uncertainty_reduction=float(mass / (mass + cfg.shrink)), decision_value=float(imp.get(pat, 0.0)),
                        failure_reduction_value=None if works else float(1 - rho[i, j])))
    return out


def context_blind(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    """Patterns that claim to predict a target but have no context, transfer or failure record at all."""
    out = []
    for p in _active_beliefs(g, now):
        if g.targets(p, R.PREDICTS, now) and not g.contexts_of(p, now) and not g.failures_of(p, now):
            out.append(_gap(GapKind.CONTEXT_BLIND, (p,), 0.4 + 0.6 * imp.get(p, 0.0),
                            f"Where does {p} work and where does it fail? It predicts "
                            f"{', '.join(g.targets(p, R.PREDICTS, now))} but no context has ever been recorded for it.",
                            problem_of(g, p, now), _newest(g, [p], now), information_gain=0.6, decision_value=imp.get(p, 0.0)))
    return out


def candidate_explanations(g: ResearchGraph, failure: str, now, top: int = 3) -> list[tuple[str, float]]:
    """Hypotheses for an unexplained failure, from the graph only: contexts where SIMILAR patterns fail (similarity-weighted),
    plus contexts already implicated in this pattern's other failures. Hypotheses to test, never labels to assign."""
    pats = [p for p in g.targets(failure, R.FAILS_IN, now) if g.kind_of(p) in BELIEF_KINDS]
    if not pats:
        return []
    target = pats[0]
    others = [p for p in _active_beliefs(g, now) if p != target]
    votes: Counter = Counter()
    if others:
        allp = [target] + others
        S = similarity_matrix(g, allp, now)
        for q, s in zip(others, S[0, 1:]):
            if s > 0:
                for c, v in g.contexts_of(q, now).items():
                    if v["fails"] > v["works"] and g.kind_of(c) in _MATRIX_KINDS:
                        votes[c] += float(s)
    for f in g.failures_of(target, now):
        for c in g.targets(f, R.EXPLAINED_BY, now):
            votes[c] += 0.5
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


def unexplained_failures(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for f in g.unexplained_failures(now):
        pats = [p for p in g.targets(f, R.FAILS_IN, now)]
        if not pats:
            continue
        p = pats[0]
        e = g.by_role(f, R.FAILS_IN, now, "out")[0][1]
        n_same = len(g.unexplained_failures(now, p))
        recur = min(1.0, 0.5 + 0.25 * (n_same - 1))
        cands = candidate_explanations(g, f, now)
        hint = (" Candidate contexts: " + ", ".join(f"{c} ({v:.2f})" for c, v in cands) + ".") if cands else ""
        out.append(_gap(GapKind.UNEXPLAINED_FAILURE, (f, p), e.weight * recur * (0.4 + 0.6 * imp.get(p, 0.0)),
                        f"Why did {p} fail in {f}? No explaining context is recorded (cause "
                        f"{g.node_at(f, now).attrs.get('cause', 'UNKNOWN')}).{hint}",
                        Problem.LOSS_AVOIDANCE, _newest(g, [f, p], now), ev={"candidates": [c for c, _ in cands],
                                                                             "unexplained_same_pattern": n_same},
                        information_gain=0.7, failure_reduction_value=e.weight, loss_reduction_value=imp.get(p, 0.0)))
    return out


def ungated_failures(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for p in _active_beliefs(g, now):
        fails = g.failures_of(p, now)
        if not fails:
            continue
        gates = set(g.targets(p, R.GATED_BY, now))
        by_ctx: Counter = Counter()
        for f in fails:
            for c in g.targets(f, R.EXPLAINED_BY, now):
                by_ctx[c] += 1
        for c, k in sorted(by_ctx.items()):
            if c not in gates:
                share = k / len(fails)
                out.append(_gap(GapKind.UNGATED_FAILURE, (p, c), (0.3 + 0.7 * share) * (0.4 + 0.6 * imp.get(p, 0.0)),
                                f"{k} of {len(fails)} failures of {p} are explained by {c}, but {p} is not gated on it. "
                                f"Would abstaining or down-weighting in {c} improve out-of-sample results?",
                                Problem.LOSS_AVOIDANCE, _newest(g, [p, c], now), ev={"failures_in_context": k, "failures": len(fails)},
                                decision_value=imp.get(p, 0.0), loss_reduction_value=share, information_gain=0.5))
    return out


def untested_gates(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for p in _active_beliefs(g, now):
        tested = {t[0] for t in g.gate_tests(p, now)}
        for c in g.targets(p, R.GATED_BY, now):
            if c not in tested:
                out.append(_gap(GapKind.UNTESTED_GATE, (p, c), 0.5 + 0.5 * imp.get(p, 0.0),
                                f"{p} is gated on {c} but no experiment has tested that gate out of sample.",
                                Problem.LOSS_AVOIDANCE, _newest(g, [p, c], now), decision_value=imp.get(p, 0.0),
                                uncertainty_reduction=0.8, information_gain=0.6))
    return out


def unanswered_questions(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    from engine.research.core import OBJECTIVE_ORDER
    out = []
    n = as_date(now)
    for q in g.nodes_of(K.QUESTION, now):
        if g.targets(q.node_id, R.ANSWERED_BY, now):
            continue
        age = (n - as_date(q.known_at)).days
        prob = Problem(q.attrs.get("problem", Problem.RESEARCH_PROCESS.value))
        rank = OBJECTIVE_ORDER.index(prob) if prob in OBJECTIVE_ORDER else len(OBJECTIVE_ORDER)
        score = (1 - math.exp(-age / 60.0)) * (1.0 / (1.0 + 0.15 * rank))
        about = g.targets(q.node_id, R.ASKS_ABOUT, now)
        out.append(_gap(GapKind.UNANSWERED_QUESTION, (q.node_id,), score,
                        f"Question {q.node_id} has waited {age} days with no experiment: {q.label}", prob,
                        _newest(g, [q.node_id] + about, now), ev={"age_days": age}, information_gain=0.5))
    return out


def unexplained_contradictions(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for a, b in g.contradiction_pairs(now):
        e = g.edge_at((a, b, Edge.CONTRADICTS.value), now)
        if e is None or e.attrs.get("context") or e.attrs.get("investigated"):
            continue
        if g.kind_of(a) not in BELIEF_KINDS and g.kind_of(b) not in BELIEF_KINDS:
            continue
        s = 0.5 * (imp.get(a, 0.0) + imp.get(b, 0.0))
        out.append(_gap(GapKind.UNEXPLAINED_CONTRADICTION, (a, b), 0.4 + 0.6 * s,
                        f"{a} and {b} contradict each other and no context explaining the disagreement is recorded. "
                        f"Do they hold in different regimes, or is one wrong?", problem_of(g, a, now),
                        _newest(g, [a, b], now), decision_value=s, uncertainty_reduction=0.7, information_gain=0.7))
    return out


def one_sided_evidence(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for p in _active_beliefs(g, now):
        ev = g.evidence_counts(p, now)
        if ev["validating"] >= cfg.min_wins_one_sided and not (ev["refuting"] or ev["failures"] or ev["transfers_failed"]
                                                               or ev["degradations"]):
            v = ev["validating"]
            out.append(_gap(GapKind.ONE_SIDED_EVIDENCE, (p,), ((v - 2) / v) * (0.4 + 0.6 * imp.get(p, 0.0)),
                            f"{p} has {v} supporting experiments and not one failure, refutation or failed transfer. A real "
                            f"pattern fails somewhere: try to break it (adversarial context, holdout era, opposite regime).",
                            problem_of(g, p, now), _newest(g, [p], now), ev=ev, information_gain=0.6,
                            decision_value=imp.get(p, 0.0), overfit_risk=0.6))
    return out


def unreplicated(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    out = []
    for p in _active_beliefs(g, now):
        ev = g.evidence_counts(p, now)
        if ev["validating"] == 1 and ev["transfers_ok"] == 0 and not ev["refuting"]:
            out.append(_gap(GapKind.UNREPLICATED, (p,), 0.3 + 0.7 * imp.get(p, 0.0),
                            f"{p} rests on a single supporting experiment and has never been replicated or transferred.",
                            problem_of(g, p, now), _newest(g, [p], now), ev=ev, transfer_potential=0.5, uncertainty_reduction=0.6,
                            decision_value=imp.get(p, 0.0)))
    return out


def stale_beliefs(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    active = set(_active_beliefs(g, now))
    out = []
    for p in g.stale_beliefs(now, cfg.stale_days):
        if p in active and g.kind_of(p) in BELIEF_KINDS:
            out.append(_gap(GapKind.STALE, (p,), 0.3 + 0.7 * imp.get(p, 0.0),
                            f"{p} has no new supporting, refuting or transfer evidence for over {cfg.stale_days} days. Is it "
                            f"still true?", problem_of(g, p, now), _newest(g, [p], now), uncertainty_reduction=0.5,
                            decision_value=imp.get(p, 0.0)))
    return out


def islands(g: ResearchGraph, now, cfg: GapConfig, imp: Mapping[str, float]) -> list[Gap]:
    comps = [c for c in g.components(now) if any(g.kind_of(n) in BELIEF_KINDS for n in c)]
    if len(comps) < 2:
        return []
    main = comps[0]
    hub = max((n for n in main if g.kind_of(n) in BELIEF_KINDS), key=lambda n: (imp.get(n, 0.0), n))
    out = []
    total = sum(len(c) for c in comps)
    for c in comps[1:]:
        beliefs = sorted(n for n in c if g.kind_of(n) in BELIEF_KINDS)
        if len(beliefs) < 2:
            continue
        out.append(_gap(GapKind.ISLAND, tuple(beliefs[:3]) + (hub,), min(1.0, len(c) / total * 2 + 0.2),
                        f"{len(beliefs)} beliefs ({', '.join(beliefs[:3])}...) share no feature, context or target with the main "
                        f"body of knowledge around {hub}. Compare them: are they redundant, complementary or unrelated?",
                        Problem.RESEARCH_PROCESS, _newest(g, beliefs, now), ev={"size": len(c)}, information_gain=0.4))
    return out


_FINDERS = (context_blind, unexplained_failures, ungated_failures, untested_gates, unanswered_questions,
            unexplained_contradictions, one_sided_evidence, unreplicated, stale_beliefs, islands)


def discover(g: ResearchGraph, now, cfg: GapConfig = GapConfig()) -> list[Gap]:
    """Every unanswered relationship the graph can name at `now`, ranked. Deterministic; sees only what was known before `now`."""
    bad = cfg.check()
    if bad:
        raise GraphError("invalid GapConfig: " + "; ".join(bad))
    imp = g.importance(now)
    gaps = list(predicted_transfers(g, now, cfg, imp))
    for fn in _FINDERS:
        gaps += fn(g, now, cfg, imp)
    per: Counter = Counter()
    seen, out = set(), []
    for gp in sorted(gaps, key=lambda x: (-x.score, x.kind.value, x.gap_id)):
        if gp.gap_id in seen or per[gp.kind] >= cfg.max_per_kind:
            continue
        seen.add(gp.gap_id)
        per[gp.kind] += 1
        out.append(gp)
    return out


_CRITERIA = {GapKind.PREDICTED_TRANSFER: ("the pattern works in the context at a fresh holdout", "it fails there: record and gate"),
             GapKind.PREDICTED_EXPOSURE: ("the pattern is shown to fail there: gate it", "it works there: the similarity was misleading"),
             GapKind.CONTEXT_BLIND: ("one working and one failing context are identified", "no context separates behaviour"),
             GapKind.UNEXPLAINED_FAILURE: ("a context explains the failure and predicts the next one", "no context does: record UNKNOWN"),
             GapKind.UNGATED_FAILURE: ("a gate improves out-of-sample results", "the gate does not help: record why"),
             GapKind.UNTESTED_GATE: ("the gate improves an untouched holdout", "the gate does not help"),
             GapKind.UNANSWERED_QUESTION: ("an experiment answers it", "the question is retired as unanswerable"),
             GapKind.UNEXPLAINED_CONTRADICTION: ("a context separates the two", "no context does: one is retired"),
             GapKind.ONE_SIDED_EVIDENCE: ("an adversarial test finds where it fails", "it survives the adversarial test"),
             GapKind.UNREPLICATED: ("a second independent test agrees", "the replication fails"),
             GapKind.STALE: ("fresh evidence agrees", "fresh evidence disagrees: degrade or retire"),
             GapKind.ISLAND: ("the cluster is linked to the main body", "they are unrelated: record it")}


def gaps_to_questions(gaps: Iterable[Gap], created_real: str) -> list[ResearchQuestion]:
    """One identity-free ResearchQuestion per gap (its node ids are opaque, its text has no ticker, date or year)."""
    out = []
    for gp in gaps:
        ok, bad = _CRITERIA[gp.kind]
        out.append(ResearchQuestion.make(gp.text, "graph_gap:" + gp.kind.value.lower(), gp.problem, created_real,
                                         gp.known_through, ok, bad, expected=gp.expected))
    return out


# ====================================================================================================== audit, reports, entry

def research_audit(g: ResearchGraph, now) -> list:
    """Base structural audit (chain, dangling and ill-typed edges, cycles, mirrors) PLUS the section-27 checks: every edge role
    is legal for its endpoint kinds, no asserted failure cause without support (a forced label), questions ask about something,
    outcomes trace to an experiment, gates are contexts that were actually named, retired beliefs drive no decision, and no
    research node carries an identity."""
    from engine.learning.knowledge_graph import GraphIssue
    issues = list(g.audit(now))
    for e in g.edges(now):
        roles = g.roles_of(e)
        ks, kd = g.kind_of(e.src), g.kind_of(e.dst)
        if not roles:
            issues.append(GraphIssue("NO_LEGAL_ROLE", "warn", f"{e.rel}:{e.src}->{e.dst}", f"no research role fits {ks} -> {kd}"))
        for r in e.attrs.get("roles", []):
            spec = ROLES[Role(r)]
            sym = RELATIONS[e.rel].symmetric
            if spec.base != e.rel or not ((ks in spec.src and kd in spec.dst) or (sym and kd in spec.src and ks in spec.dst)):
                issues.append(GraphIssue("ILLEGAL_ROLE", "error", f"{e.rel}:{e.src}->{e.dst}", f"{r} does not fit {ks} -> {kd}"))
    for n in g.nodes(now):
        kind = g.kind_of(n.node_id)
        errs = _identity_errors({"id": n.node_id, "label": n.label, **{k: v for k, v in n.attrs.items()}}, g.forbidden, "node")
        if errs:
            issues.append(GraphIssue("IDENTITY_IN_NODE", "error", n.node_id, "; ".join(errs[:2])))
        if kind == K.FAILURE:
            cause = n.attrs.get("cause", "UNKNOWN")
            supported = g.targets(n.node_id, R.EXPLAINED_BY, now) or n.attrs.get("note") or \
                any(e.evidence for _, e in g.by_role(n.node_id, R.FAILS_IN, now, "out"))
            if cause != "UNKNOWN" and not supported:
                issues.append(GraphIssue("FORCED_FAILURE_LABEL", "error", n.node_id, f"cause {cause} with nothing behind it"))
            if not g.targets(n.node_id, R.FAILS_IN, now):
                issues.append(GraphIssue("FAILURE_OF_NOTHING", "warn", n.node_id, "failure joined to no pattern"))
        elif kind == K.QUESTION and not g.targets(n.node_id, R.ASKS_ABOUT, now):
            issues.append(GraphIssue("QUESTION_ABOUT_NOTHING", "warn", n.node_id, "question asks about no node"))
        elif kind == K.OUTCOME and not g.targets(n.node_id, R.PRODUCED_BY, now) and not g.sources(n.node_id, R.RESULTED_IN, now):
            issues.append(GraphIssue("OUTCOME_WITHOUT_ORIGIN", "warn", n.node_id, "outcome traces to no experiment or decision"))
        elif kind in BELIEF_KINDS:
            status = str(n.attrs.get("status", "")).lower()
            if status in RETIRED_STATUSES and g.targets(n.node_id, R.USED_IN_DECISION, now):
                issues.append(GraphIssue("RETIRED_BELIEF_IN_DECISION", "error", n.node_id, f"status {status} yet used in a decision"))
            declared = {t[0] for t in g.gate_tests(n.node_id, now)}
            stray = declared - set(g.targets(n.node_id, R.GATED_BY, now))
            for gate in sorted(stray):
                issues.append(GraphIssue("TEST_OF_UNNAMED_GATE", "warn", n.node_id, f"experiment tests gate {gate} that is not a GATED_BY edge"))
    return issues


def kind_stats(g: ResearchGraph, now) -> dict[str, Any]:
    kinds = Counter(g.kind_of(n.node_id).value for n in g.nodes(now))
    roles: Counter = Counter()
    for e in g.edges(now):
        roles.update(r.value for r in g.roles_of(e))
    return {"kinds": dict(sorted(kinds.items())), "roles": dict(sorted(roles.items())),
            "kinds_absent": sorted(k.value for k in RKind if k.value not in kinds),
            "roles_unused": sorted(r.value for r in Role if r.value not in roles)}


def story_completeness(g: ResearchGraph, now) -> dict[str, Any]:
    """How much of the section-27 chain the active patterns support, per step (a step counts only where its predecessors exist)."""
    pats = _active_beliefs(g, now)
    steps = ("predicts", "works_in", "fails_in", "failure_explanation", "gating_rule", "gate_test", "oos_improvement")
    miss: Counter = Counter()
    complete = 0
    for p in pats:
        s = g.pattern_story(p, now)
        complete += s.complete
        miss.update(s.missing)
    n = len(pats)
    return {"patterns": n, "complete": complete, "share_complete": (complete / n) if n else None,
            "missing_by_step": {s: miss.get(s, 0) for s in steps}}


@dataclasses.dataclass(frozen=True)
class GraphStepReport:
    now: str
    digest: str
    nodes: int
    edges: int
    issues: tuple[Any, ...]
    gaps: tuple[Gap, ...]
    new_questions: tuple[ResearchQuestion, ...]
    skipped_existing: int
    completeness: Mapping[str, Any]

    @property
    def errors(self) -> tuple[Any, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    def by_kind(self) -> dict[str, int]:
        return dict(sorted(Counter(gp.kind.value for gp in self.gaps).items()))


def record_questions(g: ResearchGraph, gaps: Sequence[Gap], created_real: str, now) -> tuple[list[ResearchQuestion], int]:
    """Turn gaps into QUESTION nodes (known at `now`, so visible from the next day: a question never influences the analysis
    that asked it). A gap that already has a question node is skipped, so re-running a day adds nothing."""
    have = g.recorded_gap_ids()
    fresh, skipped = [], 0
    for gp, q in zip(gaps, gaps_to_questions(gaps, created_real)):
        if gp.gap_id in have:
            skipped += 1
            continue
        g.add_question(q, [s for s in gp.subjects if g.kind_of(s) is not None], now, {"gap_id": gp.gap_id, "gap_kind": gp.kind.value})
        fresh.append(q)
    return fresh, skipped


def step(g: ResearchGraph, now, created_real: str, cfg: GapConfig = GapConfig(), record: bool = True) -> GraphStepReport:
    """PUBLIC ENTRY for the research loop: audit the graph, discover the unanswered relationships as of `now`, and (if `record`)
    file the new ones as questions. Reads only what was known before `now`."""
    issues = research_audit(g, now)
    gaps = discover(g, now, cfg)
    fresh, skipped = record_questions(g, gaps, created_real, now) if record else ([], 0)
    st = g.stats(now)
    return GraphStepReport(as_date(now).isoformat(), g.digest(now), st["nodes"], st["edges"], tuple(issues), tuple(gaps),
                           tuple(fresh), skipped, story_completeness(g, now))


def research_markdown(g: ResearchGraph, now, gaps: Sequence[Gap] | None = None, top: int = 10) -> str:
    """Page for the research desk: node kinds and roles in use, chain completeness, and the top unanswered relationships."""
    gaps = list(gaps) if gaps is not None else discover(g, now)
    ks = kind_stats(g, now)
    sc = story_completeness(g, now)
    lines = [f"# Research graph as of {as_date(now)}", "", "| kind | nodes |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in ks["kinds"].items()]
    lines += ["", "| role | edges |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in ks["roles"].items()]
    lines += ["", f"Kinds with no nodes: {', '.join(ks['kinds_absent']) or 'none'}. Roles never used: {', '.join(ks['roles_unused']) or 'none'}.",
              "", f"Section-27 chain complete for {sc['complete']} of {sc['patterns']} active patterns.", "",
              "| missing step | patterns |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sc["missing_by_step"].items()]
    lines += ["", "## Unanswered relationships", "", "| # | kind | score | subjects | question |", "|---:|---|---:|---|---|"]
    for i, gp in enumerate(gaps[:top], 1):
        lines.append(f"| {i} | {gp.kind.value} | {gp.score:.2f} | {', '.join(gp.subjects)} | {gp.text[:110]} |")
    return "\n".join(lines)


# ====================================================================================================== analysis over the graph

@dataclasses.dataclass(frozen=True)
class ContextRow:
    context: str
    kind: str
    tested: int
    works: int
    fails: int
    lower_work: float | None                 # Wilson lower bound of the work rate
    upper_fail: float | None                 # 1 - Wilson lower bound of the fail rate's complement (a cap on how safe it is)
    verdict: str                             # SAFE / DANGER / MIXED / THIN


def context_table(g: ResearchGraph, now, min_n: int = 3) -> list[ContextRow]:
    """Per context: how many active patterns were tested there and how many worked. SAFE needs a Wilson lower bound on the work
    rate of at least 0.6, DANGER an upper bound on it of at most 0.4; fewer than `min_n` patterns is THIN, never a verdict."""
    pats = _active_beliefs(g, now)
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in pats:
        for c, v in g.contexts_of(p, now).items():
            if g.kind_of(c) in _MATRIX_KINDS:
                tally[c][0 if v["works"] > v["fails"] else 1] += 1
    rows = []
    for c, (w, f) in sorted(tally.items()):
        n = w + f
        lw = wilson_lower(w, n)
        lf = wilson_lower(f, n)
        upper_work = None if lf is None else 1.0 - lf
        verdict = "THIN" if n < min_n else "SAFE" if lw is not None and lw >= 0.6 else \
            "DANGER" if upper_work is not None and upper_work <= 0.4 else "MIXED"
        rows.append(ContextRow(c, g.kind_of(c).value, n, w, f, lw, upper_work, verdict))
    return rows


def failure_hotspots(g: ResearchGraph, now, top: int = 5) -> dict[str, Any]:
    """Which contexts explain the most failures, how many distinct patterns each hurts, and how much failure is still unexplained."""
    fails = [n.node_id for n in g.nodes_of(K.FAILURE, now)]
    by_ctx: dict[str, set] = defaultdict(set)
    n_by_ctx: Counter = Counter()
    for f in fails:
        pats = g.targets(f, R.FAILS_IN, now)
        for c in g.targets(f, R.EXPLAINED_BY, now):
            n_by_ctx[c] += 1
            by_ctx[c].update(pats)
    unexplained = len(g.unexplained_failures(now))
    rows = [{"context": c, "failures": k, "patterns_hurt": len(by_ctx[c]), "share": k / max(1, len(fails))}
            for c, k in sorted(n_by_ctx.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]
    causes = Counter(g.node_at(f, now).attrs.get("cause", "UNKNOWN") for f in fails)
    return {"failures": len(fails), "unexplained": unexplained,
            "unexplained_share": (unexplained / len(fails)) if fails else None, "hotspots": rows,
            "causes": dict(sorted(causes.items()))}


def infer_relations(g: ResearchGraph, now, min_works: int = 2, redundant_j: float = 0.8, complement_j: float = 0.2) -> list:
    """Edge proposals read off the graph's own record of where patterns work: near-identical working sets on the same target
    with shared inputs are REDUNDANT_WITH; disjoint working sets on the same target are COMPLEMENTS; a working set strictly
    inside another's with shared inputs is SPECIALIZES (the narrower belief). Pairs that already have any relation are skipped.
    Nothing is written: `apply_relations` records them, so a reviewer can see what would change."""
    from engine.learning.knowledge_graph import EdgeProposal
    pats = _active_beliefs(g, now)
    works = {p: {c for c, v in g.contexts_of(p, now).items() if v["works"] > v["fails"]} for p in pats}
    tgt = {p: set(g.targets(p, R.PREDICTS, now)) for p in pats}
    feat = {p: set(g.targets(p, R.USES_FEATURE, now)) for p in pats}
    out = []
    for i, a in enumerate(pats):
        for b in pats[i + 1:]:
            if len(works[a]) < min_works or len(works[b]) < min_works or not (tgt[a] & tgt[b]):
                continue
            keys = [(a, b, r) for r in (Edge.REDUNDANT_WITH.value, Edge.COMPLEMENTS.value, Edge.SPECIALIZES.value, Edge.GENERALIZES.value)]
            keys += [(b, a, Edge.SPECIALIZES.value), (b, a, Edge.GENERALIZES.value)]
            if any(g.edge_at(k, now) for k in keys):
                continue
            inter, uni = len(works[a] & works[b]), len(works[a] | works[b])
            j = inter / uni
            shared = len(feat[a] & feat[b])
            if j >= redundant_j and shared >= 1:
                out.append(EdgeProposal(a, b, Edge.REDUNDANT_WITH.value, round(j, 6), (), (("jaccard", round(j, 6)), ("inferred", True))))
            elif j <= complement_j and inter == 0:
                out.append(EdgeProposal(a, b, Edge.COMPLEMENTS.value, round(1 - j, 6), (), (("jaccard", round(j, 6)), ("inferred", True))))
            elif shared >= 1 and (works[a] < works[b] or works[b] < works[a]):
                child, parent = (a, b) if works[a] < works[b] else (b, a)
                out.append(EdgeProposal(child, parent, Edge.SPECIALIZES.value, round(min(1.0, len(works[child]) / len(works[parent])), 6),
                                        (), (("inferred", True),)))
    return out


def apply_relations(g: ResearchGraph, proposals: Iterable, known_at) -> list[GraphEdge]:
    """Record inferred relations, but only as hypotheses: every one carries attrs.inferred so the audit and the reader can tell
    a relation somebody measured from one the graph guessed."""
    return g.apply(proposals, known_at)


@dataclasses.dataclass(frozen=True)
class ExplanationCheck:
    failure: str
    context: str
    n_tested: int                       # similar patterns (and the pattern) tested in that context
    n_failed: int
    base_rate: float | None             # failure rate of the same patterns over all their tested contexts
    lift: float | None
    lower: float | None                 # Wilson lower bound of the failure rate in the context
    verdict: str                        # SUPPORTED / WEAK / UNTESTED / CONTRADICTED


def check_explanation(g: ResearchGraph, failure: str, context: str, now, min_similarity: float = 0.25) -> ExplanationCheck:
    """Does the proposed explanation hold up beyond the one failure it was invented for? Among the failed pattern and the patterns
    built like it, compare the failure rate in `context` with their failure rate everywhere else. SUPPORTED needs a lift of at
    least 1.5 and a Wilson lower bound above the base rate; a context in which they mostly WORK is CONTRADICTED."""
    fp = [p for p in g.targets(failure, R.FAILS_IN, now) if g.kind_of(p) in BELIEF_KINDS]
    if not fp:
        raise GraphError(f"{failure} is not the failure of any belief")
    pats = _active_beliefs(g, now)
    target = fp[0]
    group = [target]
    if target in pats:
        S = similarity_matrix(g, [target] + [p for p in pats if p != target], now, min_similarity)
        group += [p for p, s in zip([p for p in pats if p != target], S[0, 1:]) if s > 0]
    inside, outside = [0, 0], [0, 0]
    for p in dict.fromkeys(group):
        for c, v in g.contexts_of(p, now).items():
            if g.kind_of(c) not in _MATRIX_KINDS:
                continue
            bad = v["fails"] > v["works"]
            tgt = inside if c == context else outside
            tgt[0] += 1
            tgt[1] += bad
    n, k = inside
    if n == 0:
        return ExplanationCheck(failure, context, 0, 0, None, None, None, "UNTESTED")
    base_n = inside[0] + outside[0]
    base = (inside[1] + outside[1]) / base_n if base_n else None
    lower = wilson_lower(k, n)
    lift = (k / n) / base if base else None
    if k / n < 0.4:
        verdict = "CONTRADICTED"
    elif lift is not None and lift >= 1.5 and lower is not None and base is not None and lower > base:
        verdict = "SUPPORTED"
    else:
        verdict = "WEAK"
    return ExplanationCheck(failure, context, n, k, base, lift, lower, verdict)


def explanation_audit(g: ResearchGraph, now) -> list[ExplanationCheck]:
    """Check every recorded failure explanation. WEAK or CONTRADICTED ones are the explanations that may be story-telling."""
    out = []
    for f in sorted(n.node_id for n in g.nodes_of(K.FAILURE, now)):
        for c in g.targets(f, R.EXPLAINED_BY, now):
            out.append(check_explanation(g, f, c, now))
    return out


def learner_yield(g: ResearchGraph, now) -> list[dict[str, Any]]:
    """What each learner's patterns turned into: how many were found, how many are still active, how many were validated,
    replicated, failed or refuted. Input to meta-learning about which search methods are worth their compute."""
    rows = []
    for lr in g.nodes_of(K.LEARNER, now):
        found = g.sources(lr.node_id, R.LEARNED_BY, now)
        ev = [g.evidence_counts(p, now) for p in found]
        active = set(_active_beliefs(g, now))
        n = len(found)
        rows.append({"learner": lr.node_id, "found": n, "active": sum(p in active for p in found),
                     "validated": sum(e["validating"] > 0 for e in ev), "replicated": sum(e["transfers_ok"] > 0 for e in ev),
                     "refuted": sum(e["refuting"] > 0 for e in ev), "failed": sum(e["failures"] > 0 for e in ev),
                     "survival": (sum(p in active for p in found) / n) if n else None,
                     "replication_rate": (sum(e["transfers_ok"] > 0 for e in ev) / n) if n else None})
    return sorted(rows, key=lambda r: (-(r["found"]), r["learner"]))


def feature_usage(g: ResearchGraph, now, toxic_min: int = 3) -> list[dict[str, Any]]:
    """Per feature: how many patterns use it and how those fared. A feature is 'toxic' when at least `toxic_min` patterns use
    it and every one of them has a failure or refutation and none has a validation."""
    active = set(_active_beliefs(g, now))
    rows = []
    for f in g.nodes_of(K.FEATURE, now):
        users = g.sources(f.node_id, R.USES_FEATURE, now)
        ev = {p: g.evidence_counts(p, now) for p in users}
        bad = [p for p, e in ev.items() if e["failures"] or e["refuting"]]
        good = [p for p, e in ev.items() if e["validating"] and not (e["failures"] or e["refuting"])]
        rows.append({"feature": f.node_id, "patterns": len(users), "active": sum(p in active for p in users),
                     "with_failures": len(bad), "clean_wins": len(good),
                     "toxic": len(users) >= toxic_min and len(bad) == len(users) and not any(e["validating"] for e in ev.values())})
    return sorted(rows, key=lambda r: (-r["patterns"], r["feature"]))


def context_reach(g: ResearchGraph, now, top: int = 5) -> list[tuple[str, int, float]]:
    """Contexts that touch the most active patterns: (context, patterns, share of active patterns). A context with a huge
    reach is where a regime error would hurt everything at once."""
    pats = _active_beliefs(g, now)
    reach: Counter = Counter()
    for p in pats:
        for c in g.contexts_of(p, now):
            if g.kind_of(c) in _MATRIX_KINDS:
                reach[c] += 1
    return [(c, k, k / len(pats)) for c, k in sorted(reach.items(), key=lambda kv: (-kv[1], kv[0]))[:top]] if pats else []


def query_patterns(g: ResearchGraph, now, **constraints: Sequence[str]) -> list[str]:
    """Patterns that satisfy every role constraint, e.g. query_patterns(g, now, WORKS_IN=['reg:high_vol'], FAILS_IN=['reg:shift']).
    Constraint names are Role names; FAILS_IN also matches through failure explanations (contexts_of)."""
    hits = None
    for name, targets in constraints.items():
        role = Role.parse(name)
        ok = set()
        for p in _active_beliefs(g, now):
            ctx = g.contexts_of(p, now)
            if role == R.WORKS_IN:
                have = {c for c, v in ctx.items() if v["works"]}
            elif role == R.FAILS_IN:
                have = {c for c, v in ctx.items() if v["fails"]}
            else:
                have = set(g.targets(p, role, now))
            if set(targets) <= have:
                ok.add(p)
        hits = ok if hits is None else hits & ok
    return sorted(hits if hits is not None else _active_beliefs(g, now))


def explain_link(g: ResearchGraph, a: str, b: str, now, max_depth: int = 6) -> str:
    """Why are two nodes connected? The shortest chain with the ROLE of every hop (not the base relation names)."""
    t = g.traverse(a, now, None, "both", max_depth)
    path = t.path_to(b)
    if not path:
        return f"no path from {a} to {b} within {max_depth} hops at {as_date(now)}"
    parts = [a]
    for x, y in zip(path, path[1:]):
        rel = t.parent[y][1]
        fwd = g.edge_at((x, y, rel), now)
        e = fwd or g.edge_at((y, x, rel), now)
        roles = sorted(r.value for r in g.roles_of(e)) if e else [rel]
        parts.append(f"--{'/'.join(roles)}--> {y}" if fwd or RELATIONS[rel].symmetric else f"<--{'/'.join(roles)}-- {y}")
    return " ".join(parts)


def question_board(g: ResearchGraph, now) -> list[dict[str, Any]]:
    """Open and answered research questions: problem, source, age, and the experiments that answered them."""
    n = as_date(now)
    rows = []
    for q in g.nodes_of(K.QUESTION, now):
        exps = g.targets(q.node_id, R.ANSWERED_BY, now)
        rows.append({"question": q.node_id, "problem": q.attrs.get("problem"), "source": q.attrs.get("source"),
                     "age_days": (n - as_date(q.known_at)).days, "answered": bool(exps), "experiments": exps,
                     "gap_kind": q.attrs.get("gap_kind")})
    return sorted(rows, key=lambda r: (r["answered"], -r["age_days"], r["question"]))


def reconcile_with_archive(g: ResearchGraph, archive, now) -> dict[str, Any]:
    """Compare the archive's failure records with the graph's FAILURE nodes. Anything the archive has that the graph does not
    is an ingestion gap; the counts per cause show whether the graph kept the archive's honesty (UNKNOWN stays UNKNOWN)."""
    recs = archive.failures(now)
    missing = sorted(r.rec_id for r in recs if node_key(K.FAILURE, r.rec_id) not in g._nodes or g.node_at(node_key(K.FAILURE, r.rec_id), now) is None)
    arch_causes = Counter(r.payload["cause"] for r in recs)
    graph_causes = Counter(n.attrs.get("cause", "UNKNOWN") for n in g.nodes_of(K.FAILURE, now))
    return {"archive_failures": len(recs), "missing_in_graph": missing, "archive_causes": dict(sorted(arch_causes.items())),
            "graph_causes": dict(sorted(graph_causes.items())),
            "unknown_preserved": graph_causes.get("UNKNOWN", 0) >= arch_causes.get("UNKNOWN", 0) - len(missing)}


@dataclasses.dataclass(frozen=True)
class GapFlow:
    t0: str
    t1: str
    opened: tuple[str, ...]
    closed: tuple[tuple[str, str], ...]        # (gap id, why)
    persisted: tuple[str, ...]


def gap_flow(g: ResearchGraph, t0, t1, cfg: GapConfig = GapConfig()) -> GapFlow:
    """Research progress as the graph sees it: which unanswered relationships opened and which closed between two dates, and for
    each closed one whether NEW EVIDENCE about its subjects arrived (answered) or its subjects merely stopped qualifying
    (retired, or the pattern left the active set)."""
    if as_date(t1) <= as_date(t0):
        raise FirewallBreach("gap_flow needs t1 after t0")
    a = {x.gap_id: x for x in discover(g, t0, cfg)}
    b = {x.gap_id: x for x in discover(g, t1, cfg)}
    closed = []
    for gid in sorted(set(a) - set(b)):
        new = 0
        for s in a[gid].subjects:
            new += sum(1 for _, e in g.neighbors(s, t1, None, "both") if as_date(t0) <= as_date(e.known_at) < as_date(t1))
        closed.append((gid, "new evidence about its subjects" if new else "subjects no longer qualify"))
    return GapFlow(as_date(t0).isoformat(), as_date(t1).isoformat(), tuple(sorted(set(b) - set(a))), tuple(closed),
                   tuple(sorted(set(a) & set(b))))


def persistent_gaps(g: ResearchGraph, dates: Sequence, cfg: GapConfig = GapConfig(), min_seen: int = 2) -> list[tuple[str, int]]:
    """Gaps present at at least `min_seen` of the given dates: the ones worth an experiment, since a gap that flickers with one
    day's noise is not. Returns (gap id, dates seen)."""
    seen: Counter = Counter()
    for d in dates:
        seen.update(x.gap_id for x in discover(g, d, cfg))
    return sorted(((k, v) for k, v in seen.items() if v >= min_seen), key=lambda kv: (-kv[1], kv[0]))


def evolution(g: ResearchGraph, dates: Sequence, cfg: GapConfig = GapConfig()) -> list[dict[str, Any]]:
    """The graph's growth and its open questions over time (one row per date): nodes, edges, open gaps by kind, and how much of
    the section-27 chain the active patterns support. Rising completeness with falling gaps is what learning looks like."""
    rows = []
    for d in dates:
        st = g.stats(d)
        sc = story_completeness(g, d)
        gaps = discover(g, d, cfg)
        rows.append({"date": st["now"], "nodes": st["nodes"], "edges": st["edges"], "gaps": len(gaps),
                     "gaps_by_kind": dict(sorted(Counter(x.kind.value for x in gaps).items())),
                     "share_complete": sc["share_complete"], "patterns": sc["patterns"]})
    return rows
