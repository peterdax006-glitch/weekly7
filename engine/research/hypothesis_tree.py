"""Research hypothesis tree (C66 section 41; also sections 33, 40; canon C66, C63).

Research is not a list of independent experiments. It is a tree:

    Question
    +-- Hypothesis A  -- Test A1, Test A2, Counterexample A3
    +-- Hypothesis B  -- Test B1, Test B2
    +-- Unknown explanation -- Search C1, Search C2

The tree keeps ONE belief over the live hypotheses plus an explicit unknown explanation (section 33: unknown is an answer and
can never be killed). Every recorded result moves that belief through a documented likelihood table (the existing
experiment_memory update with its model-misfit guard). A branch is marked FAILED only when the evidence is both strong (its
posterior is small AND the leader beats it by a Bayes factor) and sufficiently discriminating (it has been counted against by
several results, or a verified counterexample hit a universal claim). When a branch dies, its pending tests are CANCELLED and
research is redirected: the next action is always the pending test with the highest expected information gain per unit cost
over the SURVIVING explanations, and when every named hypothesis is dead the search of the unknown branch takes over.

Blind-trader rule (C64/C66 section 31): a tree is built from matured outcomes, so it lives in MATURED_RESEARCH_STATE. Results
are refused unless their evidence date is strictly before `now`; question and node text must be identity free.

Public entry: `step(forest, results, now)`. Builds on engine.learning.research_policy (entropy, mutual information),
engine.learning.experiment_memory (robust_update, normalise, question_key) and engine.learning.research_priority (hypothesis
sets and outcome tables). IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.experiment_memory import Hypothesis, normalise, question_key, robust_update, to_ts
from engine.learning.research_policy import entropy_bits, mutual_information
from engine.learning.research_priority import (SignalKind, expected_for, failure_hypotheses, generic_hypotheses, identity_leak)
from engine.research.core import (FirewallBreach, Namespace, ResearchState, _StrEnum, require_past, stable_hash)

LABEL = "IMPLEMENTED - NOT VALIDATED"
UNKNOWN_HID = "h_unknown"
EPS = 1e-12


class TreeError(ValueError):
    """A structural misuse of the tree: unknown node, second result for one test, a test under the wrong parent."""


class NodeKind(_StrEnum):
    QUESTION = "QUESTION"
    HYPOTHESIS = "HYPOTHESIS"
    TEST = "TEST"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    UNKNOWN = "UNKNOWN_EXPLANATION"
    SEARCH = "SEARCH"


class NodeStatus(_StrEnum):
    LIVE = "LIVE"                       # a hypothesis still in play (also the unknown branch, always)
    SUPPORTED = "SUPPORTED"             # a hypothesis the evidence now favours decisively
    FAILED = "FAILED"                   # a branch killed by evidence: never deleted, never revived silently
    PENDING = "PENDING"                 # a test / counterexample search / unknown search not yet run
    DONE = "DONE"
    CANCELLED = "CANCELLED"             # a pending item whose branch died or whose question was resolved


class TreeState(_StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"               # one named hypothesis is supported
    UNKNOWN_LEADS = "UNKNOWN_LEADS"     # every named hypothesis failed; the unknown explanation is what remains
    STALLED = "STALLED"                 # no pending item can move the belief


@dataclass(frozen=True)
class TreeConfig:
    """Every threshold in one place. Defaults are priors to be validated, not tuned values."""
    unknown_floor: float = 0.05         # the unknown explanation never falls below this share of belief
    kill_posterior: float = 0.04        # a hypothesis must be below this to be killed ...
    kill_bayes_factor: float = 12.0     # ... and the leader must beat it by this cumulative Bayes factor ...
    min_against: int = 3                # ... and at least this many results must have counted against it
    support_posterior: float = 0.80
    support_margin: float = 0.40        # lead over the runner-up (including unknown)
    min_results_to_support: int = 3
    sensitivity: float = 0.70           # P(counterexample search finds one | the claim is false)
    false_alarm: float = 0.05           # P(finds one | the claim is true)
    smoothing: float = 0.02
    misfit_below: float = 0.12
    misfit_leak: float = 0.30
    cost_exponent: float = 0.5          # value = eig / cost ** exponent: cost matters, but sub-linearly
    stall_days: float = 30.0
    max_hypotheses: int = 12
    against_ratio: float = 0.67         # a result counts against h when h's likelihood < ratio x the best likelihood

    def check(self) -> list:
        errs = []
        for name in ("unknown_floor", "kill_posterior", "support_posterior", "sensitivity", "false_alarm", "smoothing"):
            v = getattr(self, name)
            if not (0.0 <= v < 1.0) or math.isnan(v):
                errs.append(f"config.{name}={v!r} outside [0, 1)")
        if self.kill_bayes_factor <= 1.0:
            errs.append("kill_bayes_factor must exceed 1")
        if self.sensitivity <= self.false_alarm:
            errs.append("a counterexample search must be more likely to find one when the claim is false than when it is true")
        if self.support_posterior + self.unknown_floor > 1.0 + EPS:
            errs.append("support_posterior + unknown_floor exceeds 1: a hypothesis could never be supported")
        if self.min_against < 1 or self.min_results_to_support < 1:
            errs.append("minimum result counts must be at least 1")
        return errs


@dataclass(frozen=True)
class Node:
    """One vertex. Immutable: a status change writes a new Node (the tree keeps the old one only in its event log)."""
    nid: str
    kind: NodeKind
    parent: str
    text: str
    status: NodeStatus
    prior: float = 0.0                              # hypotheses only: belief at creation
    cost: float = 1.0                               # cpu-minutes for tests / searches
    table: tuple = ()                               # tests: ((hid, outcome, probability), ...)
    sensitivity: float = 0.0                        # counterexample: P(found | claim false)
    false_alarm: float = 0.0                        # counterexample: P(found | claim true)
    universal: bool = False                         # hypotheses: a universal claim dies on one verified counterexample
    signature: str = ""                             # tests: hash of the table, so an identical test is never counted twice
    result: str = ""
    result_at: str = ""
    evidence_through: str = ""
    note: str = ""

    def outcomes(self) -> tuple:
        return tuple(sorted({o for _, o, _ in self.table}))

    def likelihood_row(self, hid: str) -> dict:
        return {o: p for h, o, p in self.table if h == hid}


@dataclass(frozen=True)
class TreeEvent:
    seq: int
    at: str
    kind: str                                       # created / result / killed / supported / cancelled / redirect / orphan / duplicate / spawn / stall
    nid: str
    detail: str


@dataclass(frozen=True)
class Action:
    """What to run next and why."""
    nid: str
    tree_id: str
    kind: NodeKind
    text: str
    eig_bits: float
    cost: float
    value: float
    reason: str


@dataclass(frozen=True)
class UpdateReport:
    tree_id: str
    nid: str
    outcome: str
    before: Mapping
    after: Mapping
    killed: tuple
    supported: tuple
    cancelled: tuple
    misfit: bool
    counted_twice: bool
    state: TreeState
    next_action: Action | None


@dataclass(frozen=True)
class TestResult:
    """A finished experiment offered to the forest."""
    tree_id: str
    nid: str
    outcome: str
    evidence_through: str
    verified: bool = False                          # a counterexample was independently verified (kills a universal claim)


def _unit_check(name: str, v: float, errs: list) -> None:
    if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0) or math.isnan(v):
        errs.append(f"{name}={v!r} outside [0,1]")


def table_signature(table: Iterable[tuple]) -> str:
    return stable_hash(sorted((h, o, round(float(p), 6)) for h, o, p in table), 12)


class HypothesisTree:
    """One question's tree. Mutating methods return nothing; every change is an event. Reading methods are pure."""

    def __init__(self, tree_id: str, question_text: str, created_real: str, cfg: TreeConfig | None = None,
                 namespace: Namespace = Namespace.MATURED_RESEARCH, problem: str = ""):
        self.cfg = cfg or TreeConfig()
        errs = self.cfg.check()
        if errs:
            raise ValueError("; ".join(errs))
        leak = identity_leak(question_text)
        if leak:
            raise FirewallBreach(f"tree question {leak}: research questions are about situations, not calendar dates")
        if namespace != Namespace.MATURED_RESEARCH:
            raise FirewallBreach("a hypothesis tree holds matured outcomes and may only live in MATURED_RESEARCH_STATE")
        self.tree_id = tree_id
        self.namespace = namespace
        self.problem = problem
        self.created_real = created_real
        self.nodes: dict = {}
        self.events: list = []
        self.logev: dict = {}                       # hid -> cumulative log likelihood of everything seen
        self.against: dict = {}                     # hid -> number of results that counted against it
        self.seen_signatures: dict = {}             # signature -> nid that already contributed
        self.state = TreeState.OPEN
        self.root = "q_" + stable_hash([tree_id, question_text], 8)
        self.nodes[self.root] = Node(self.root, NodeKind.QUESTION, "", question_text, NodeStatus.LIVE)
        self.nodes[UNKNOWN_HID] = Node(UNKNOWN_HID, NodeKind.UNKNOWN, self.root, "none of the named explanations", NodeStatus.LIVE,
                                       prior=self.cfg.unknown_floor)
        self.logev[UNKNOWN_HID] = 0.0
        self.against[UNKNOWN_HID] = 0
        self._belief: dict = {}
        self._log(created_real, "created", self.root, question_text)

    # ---------------------------------------------------------------------------------------------------- bookkeeping
    def _log(self, at, kind: str, nid: str, detail: str) -> None:
        self.events.append(TreeEvent(len(self.events), str(at), kind, nid, detail))

    def _put(self, node: Node) -> None:
        self.nodes[node.nid] = node

    def get(self, nid: str) -> Node:
        if nid not in self.nodes:
            raise TreeError(f"unknown node {nid!r} in tree {self.tree_id}")
        return self.nodes[nid]

    def children(self, nid: str, kind: NodeKind | None = None) -> list:
        return sorted((n for n in self.nodes.values() if n.parent == nid and (kind is None or n.kind == kind)),
                      key=lambda n: (n.kind == NodeKind.UNKNOWN, n.nid))

    def hypotheses(self, live_only: bool = False) -> list:
        rows = [n for n in self.nodes.values() if n.kind in (NodeKind.HYPOTHESIS, NodeKind.UNKNOWN)]
        if live_only:
            rows = [n for n in rows if n.status in (NodeStatus.LIVE, NodeStatus.SUPPORTED)]
        return sorted(rows, key=lambda n: n.nid)

    def live_ids(self) -> list:
        return [n.nid for n in self.hypotheses(live_only=True)]

    # ---------------------------------------------------------------------------------------------------- building
    def add_hypothesis(self, hid: str, text: str, prior: float, universal: bool = False, at=None) -> str:
        if hid in self.nodes:
            raise TreeError(f"node {hid!r} already exists")
        leak = identity_leak(text)
        if leak:
            raise FirewallBreach(f"hypothesis {hid}: text {leak}")
        if not (0.0 < prior < 1.0):
            raise TreeError(f"hypothesis {hid}: prior {prior!r} must be in (0, 1)")
        if len([h for h in self.hypotheses() if h.kind == NodeKind.HYPOTHESIS]) >= self.cfg.max_hypotheses:
            raise TreeError(f"tree {self.tree_id} already holds {self.cfg.max_hypotheses} hypotheses")
        self._put(Node(hid, NodeKind.HYPOTHESIS, self.root, text, NodeStatus.LIVE, prior=float(prior), universal=universal))
        self.logev[hid] = 0.0
        self.against[hid] = 0
        self._belief = {}
        self._log(at or self.created_real, "created", hid, text)
        return hid

    def seal_priors(self, unknown_prior: float = 0.0) -> None:
        """Fix the initial belief: hypothesis priors are renormalised together with the unknown floor. Call once after the
        hypotheses are added; adding a hypothesis later goes through spawn_hypothesis, which takes mass from the unknown branch."""
        hyps = [h for h in self.hypotheses() if h.kind == NodeKind.HYPOTHESIS]
        if not hyps:
            raise TreeError("a tree needs at least one named hypothesis")
        u = min(max(self.cfg.unknown_floor, float(unknown_prior)), 0.9)
        scale = (1.0 - u) / sum(h.prior for h in hyps)
        for h in hyps:
            self._put(replace(h, prior=h.prior * scale))
        self._put(replace(self.get(UNKNOWN_HID), prior=u))
        self._belief = {h.nid: h.prior for h in self.hypotheses()}

    def add_test(self, tid: str, hid: str, text: str, table: Sequence[tuple], cost: float = 1.0, at=None) -> str:
        """A test under hypothesis `hid`. `table` is [(hypothesis id, outcome label, P(outcome | hypothesis))...] covering every
        hypothesis the test can tell apart; a hypothesis missing from it is treated as not distinguished (its row is the
        belief-weighted average of the rows given)."""
        parent = self.get(hid)
        if parent.kind not in (NodeKind.HYPOTHESIS, NodeKind.UNKNOWN):
            raise TreeError(f"test {tid} must hang under a hypothesis, not a {parent.kind}")
        if tid in self.nodes:
            raise TreeError(f"node {tid!r} already exists")
        table = tuple((str(h), str(o), float(p)) for h, o, p in table)
        self._validate_table(tid, table)
        self._put(Node(tid, NodeKind.TEST, hid, text, NodeStatus.PENDING, cost=max(float(cost), 0.01), table=table,
                       signature=table_signature(table)))
        self._log(at or self.created_real, "created", tid, text)
        return tid

    def add_counterexample(self, cid: str, hid: str, text: str, cost: float = 1.0, at=None) -> str:
        """A search for a case that contradicts hypothesis `hid`. Outcome labels are 'found' / 'not_found'."""
        parent = self.get(hid)
        if parent.kind != NodeKind.HYPOTHESIS:
            raise TreeError("a counterexample search hangs under a named hypothesis")
        if cid in self.nodes:
            raise TreeError(f"node {cid!r} already exists")
        self._put(Node(cid, NodeKind.COUNTEREXAMPLE, hid, text, NodeStatus.PENDING, cost=max(float(cost), 0.01),
                       sensitivity=self.cfg.sensitivity, false_alarm=self.cfg.false_alarm))
        self._log(at or self.created_real, "created", cid, text)
        return cid

    def add_search(self, sid: str, text: str, coverage: float = 0.3, cost: float = 1.0, at=None) -> str:
        """A search of the unknown branch: with probability `coverage` it produces a candidate explanation."""
        if sid in self.nodes:
            raise TreeError(f"node {sid!r} already exists")
        _errs: list = []
        _unit_check("coverage", coverage, _errs)
        if _errs:
            raise TreeError("; ".join(_errs))
        self._put(Node(sid, NodeKind.SEARCH, UNKNOWN_HID, text, NodeStatus.PENDING, cost=max(float(cost), 0.01),
                       prior=float(coverage)))
        self._log(at or self.created_real, "created", sid, text)
        return sid

    @staticmethod
    def _validate_table(tid: str, table: Sequence[tuple]) -> None:
        if not table:
            raise TreeError(f"test {tid} has an empty likelihood table")
        by_h: dict = {}
        for h, o, p in table:
            if not (0.0 <= p <= 1.0) or math.isnan(p):
                raise TreeError(f"test {tid}: P({o}|{h})={p!r} outside [0,1]")
            by_h[h] = by_h.get(h, 0.0) + p
        for h, tot in by_h.items():
            if abs(tot - 1.0) > 1e-6:
                raise TreeError(f"test {tid}: probabilities for hypothesis {h} sum to {tot:.6f}, not 1")

    def spawn_hypothesis(self, hid: str, text: str, mass: float, at, universal: bool = False) -> str:
        """The unknown search found a candidate explanation: give it `mass` of belief, taken first from the unknown branch (down
        to its floor) and then proportionally from the named hypotheses. It starts with no evidence against it (the tests
        already run did not test it), so it cannot be killed until it has faced new results."""
        if not (0.0 < mass <= 0.5):
            raise TreeError(f"a newly found explanation may start with at most half the belief, not {mass!r}")
        cur = self.belief()
        room = max(0.0, cur.get(UNKNOWN_HID, 0.0) - self.cfg.unknown_floor)
        from_unknown = min(room, mass)
        rest = mass - from_unknown
        others = sum(v for k, v in cur.items() if k != UNKNOWN_HID)
        self.add_hypothesis(hid, text, min(max(mass, 1e-6), 0.999), universal=universal, at=at)
        new = {k: (v - from_unknown if k == UNKNOWN_HID else v * (1 - rest / others) if others > 0 else v) for k, v in cur.items()}
        new[hid] = mass
        self._belief = self._floor_unknown(normalise(new))
        self.state = TreeState.OPEN
        self._log(at, "spawn", hid, f"{mass:.3f} of belief ({from_unknown:.3f} from the unknown branch)")
        return hid

    # ---------------------------------------------------------------------------------------------------- belief
    def belief(self) -> dict:
        """Current probability of each live hypothesis (the unknown branch included), summing to 1. Failed hypotheses are absent."""
        if not self._belief:
            if any(n.kind == NodeKind.HYPOTHESIS for n in self.nodes.values()):
                self.seal_priors()
        live = self.live_ids()
        b = {h: self._belief.get(h, 0.0) for h in live}
        tot = sum(b.values())
        return {h: v / tot for h, v in b.items()} if tot > 0 else {h: 1.0 / len(live) for h in live}

    def entropy(self) -> float:
        return entropy_bits(self.belief().values())

    def leader(self) -> tuple:
        b = self.belief()
        hid = max(b, key=lambda h: (b[h], h))
        return hid, b[hid]

    def margin(self) -> float:
        vals = sorted(self.belief().values(), reverse=True)
        return vals[0] - (vals[1] if len(vals) > 1 else 0.0)

    def _row_for(self, node: Node, hid: str, outcomes: Sequence[str], belief: Mapping) -> dict:
        """P(outcome | hid) for a test: its own row when the table has one; the unknown branch predicts nothing specific
        (uniform); an undistinguished hypothesis gets the belief-weighted average of the tabled rows."""
        row = node.likelihood_row(hid)
        if row:
            return {o: row.get(o, 0.0) for o in outcomes}
        if hid == UNKNOWN_HID:
            return {o: 1.0 / len(outcomes) for o in outcomes}
        tabled = sorted({h for h, _, _ in node.table})
        w = np.array([belief.get(h, 0.0) for h in tabled])
        if w.sum() <= 0:
            w = np.ones(len(tabled))
        w = w / w.sum()
        return {o: float(sum(wi * node.likelihood_row(h).get(o, 0.0) for wi, h in zip(w, tabled))) for o in outcomes}

    def test_table(self, nid: str) -> dict:
        """The full likelihood table of a pending or finished item over the CURRENT live hypotheses: {hid: {outcome: p}}."""
        node = self.get(nid)
        live = self.live_ids()
        belief = self.belief()
        if node.kind == NodeKind.COUNTEREXAMPLE:
            target = node.parent
            return {h: ({"found": node.false_alarm, "not_found": 1 - node.false_alarm} if h == target
                        else {"found": node.sensitivity, "not_found": 1 - node.sensitivity}) for h in live}
        if node.kind == NodeKind.SEARCH:
            cov = node.prior
            return {h: ({"candidate": cov, "nothing": 1 - cov} if h == UNKNOWN_HID else {"candidate": 0.0, "nothing": 1.0}) for h in live}
        if node.kind != NodeKind.TEST:
            raise TreeError(f"{nid} is a {node.kind}, not an experiment")
        outcomes = node.outcomes()
        return {h: self._row_for(node, h, outcomes, belief) for h in live}

    def eig(self, nid: str) -> float:
        """Expected information gain (bits) of running a pending item NOW, over the surviving explanations: the mutual
        information between which hypothesis is true and the outcome. A finished or cancelled item has none."""
        node = self.get(nid)
        if node.status != NodeStatus.PENDING:
            return 0.0
        if node.kind == NodeKind.SEARCH:
            return self._search_gain(node)
        if self._owner_dead(node):
            return 0.0
        b = self.belief()
        if len(b) < 2:
            return 0.0
        if node.signature in self.seen_signatures and node.kind == NodeKind.TEST:
            return 0.0
        try:
            return mutual_information(b, self.test_table(nid))
        except ValueError:
            return 0.0

    def _search_gain(self, node: Node) -> float:
        """A search of the unknown branch is worth its chance of producing a candidate times the unknown mass it could resolve."""
        u = self.belief().get(UNKNOWN_HID, 0.0)
        return float(node.prior * u * entropy_bits([max(u, EPS), max(1 - u, EPS)]) if 0 < u < 1 else 0.0)

    def _owner_dead(self, node: Node) -> bool:
        parent = self.nodes.get(node.parent)
        return parent is None or parent.status in (NodeStatus.FAILED, NodeStatus.CANCELLED)

    # ---------------------------------------------------------------------------------------------------- results
    def _smoothed(self, table: Mapping, outcome: str) -> dict:
        outs = sorted({o for row in table.values() for o in row})
        s = self.cfg.smoothing
        return {h: (1 - s) * row.get(outcome, 0.0) + s / max(1, len(outs)) for h, row in table.items()}

    def _floor_unknown(self, post: dict) -> dict:
        """The unknown explanation keeps at least `unknown_floor` of the belief: evidence can shrink it, never delete it."""
        u = post.get(UNKNOWN_HID, 0.0)
        if u >= self.cfg.unknown_floor - EPS:
            return post
        rest = sum(v for k, v in post.items() if k != UNKNOWN_HID)
        scale = (1.0 - self.cfg.unknown_floor) / rest if rest > 0 else 0.0
        out = {k: v * scale for k, v in post.items() if k != UNKNOWN_HID}
        out[UNKNOWN_HID] = self.cfg.unknown_floor
        return out

    def _bayes(self, before: Mapping, like: Mapping) -> tuple:
        """Bayes update with the misfit guard of experiment_memory.robust_update, judged on the NAMED hypotheses only (the unknown
        branch predicts everything a little, so it would always hide a misfit)."""
        post = normalise({k: before[k] * like.get(k, 0.0) for k in before})
        named = [like[h] for h in like if h != UNKNOWN_HID]
        if named and max(named) < self.cfg.misfit_below and UNKNOWN_HID in post:
            leak = self.cfg.misfit_leak
            moved = {k: v * (1 - leak) for k, v in post.items() if k != UNKNOWN_HID}
            moved[UNKNOWN_HID] = post[UNKNOWN_HID] * (1 - leak) + leak
            return normalise(moved), True
        return post, False

    def record_result(self, nid: str, outcome: str, now, evidence_through, verified: bool = False) -> UpdateReport:
        node = self.get(nid)
        if node.kind not in (NodeKind.TEST, NodeKind.COUNTEREXAMPLE):
            raise TreeError(f"{nid} is a {node.kind}; use record_search for searches of the unknown branch")
        if node.status == NodeStatus.DONE:
            raise TreeError(f"{nid} already has a result: evidence counts once")
        require_past(evidence_through, now, f"result for {nid}")
        before = self.belief()
        if node.status == NodeStatus.CANCELLED:
            self._put(replace(node, result=outcome, result_at=str(now), evidence_through=str(evidence_through), note="orphan: branch was closed"))
            self._log(now, "orphan", nid, f"result {outcome!r} arrived for a cancelled item and was not used")
            return self._report(nid, outcome, before, (), (), (), False, False)
        table = self.test_table(nid)
        allowed = sorted({o for row in table.values() for o in row})
        if outcome not in allowed:
            raise TreeError(f"{nid}: outcome {outcome!r} is not one of {allowed}")
        if node.kind == NodeKind.TEST and node.signature in self.seen_signatures and self.seen_signatures[node.signature] != nid:
            self._put(replace(node, status=NodeStatus.DONE, result=outcome, result_at=str(now), evidence_through=str(evidence_through),
                              note="identical test already counted"))
            self._log(now, "duplicate", nid, f"same table as {self.seen_signatures[node.signature]}; not counted twice")
            return self._report(nid, outcome, before, (), (), (), False, True)
        like = self._smoothed(table, outcome)
        post, misfit = self._bayes(before, like)
        post = self._floor_unknown(post)
        best = max(like.values())
        for h, p in like.items():
            self.logev[h] = self.logev.get(h, 0.0) + math.log(max(p, EPS))
            if p < self.cfg.against_ratio * best and h != UNKNOWN_HID:
                self.against[h] = self.against.get(h, 0) + 1
        self._belief = post
        self._put(replace(node, status=NodeStatus.DONE, result=outcome, result_at=str(now), evidence_through=str(evidence_through)))
        if node.kind == NodeKind.TEST:
            self.seen_signatures[node.signature] = nid
        self._log(now, "result", nid, f"{outcome} (misfit)" if misfit else outcome)
        killed = self._hard_kill(node, outcome, verified, now)
        return self._settle(nid, outcome, before, now, misfit, killed)

    def record_search(self, sid: str, found: bool, now, evidence_through, text: str = "", mass: float = 0.0) -> UpdateReport:
        """Result of a search of the unknown branch. Finding a candidate spawns a NEW hypothesis fed from the unknown mass."""
        node = self.get(sid)
        if node.kind != NodeKind.SEARCH:
            raise TreeError(f"{sid} is not a search")
        if node.status == NodeStatus.DONE:
            raise TreeError(f"{sid} already has a result")
        require_past(evidence_through, now, f"search result for {sid}")
        before = self.belief()
        self._put(replace(node, status=NodeStatus.DONE, result="candidate" if found else "nothing", result_at=str(now),
                          evidence_through=str(evidence_through)))
        self._log(now, "result", sid, "candidate" if found else "nothing")
        if found:
            if not text:
                raise TreeError("a search that found a candidate must say what it is")
            self.spawn_hypothesis("h_" + stable_hash([sid, text], 6), text, mass or 0.10, now)
        return self._settle(sid, "candidate" if found else "nothing", before, now, False, ())

    def _hard_kill(self, node: Node, outcome: str, verified: bool, now) -> tuple:
        """A universal claim ('always', 'never') dies on ONE verified counterexample, whatever the Bayes factor says."""
        if node.kind != NodeKind.COUNTEREXAMPLE or outcome != "found" or not verified:
            return ()
        target = self.nodes.get(node.parent)
        if target is None or not target.universal or target.status != NodeStatus.LIVE:
            return ()
        self._fail(target.nid, now, "verified counterexample to a universal claim")
        return (target.nid,)

    def _fail(self, hid: str, now, why: str) -> None:
        self._put(replace(self.get(hid), status=NodeStatus.FAILED, note=why))
        self._belief.pop(hid, None)
        self._log(now, "killed", hid, why)

    def _cancel_under(self, hid: str, now, why: str) -> list:
        out = []
        for c in self.children(hid):
            if c.status == NodeStatus.PENDING:
                self._put(replace(c, status=NodeStatus.CANCELLED, note=why))
                self._log(now, "cancelled", c.nid, why)
                out.append(c.nid)
        return out

    def _killable(self, hid: str, belief: Mapping, leader: str) -> bool:
        if hid == UNKNOWN_HID or hid == leader:
            return False
        gap = self.logev.get(leader, 0.0) - self.logev.get(hid, 0.0)
        return (belief.get(hid, 0.0) < self.cfg.kill_posterior and self.against.get(hid, 0) >= self.cfg.min_against
                and gap >= math.log(self.cfg.kill_bayes_factor))

    def _settle(self, nid: str, outcome: str, before: Mapping, now, misfit: bool, hard_killed: tuple) -> UpdateReport:
        """After any result: kill what the evidence has beaten, redirect, close the question if one explanation is supported."""
        killed = list(hard_killed)
        b = self.belief()
        leader = max(self.live_ids(), key=lambda h: (self.logev.get(h, 0.0), h))
        for h in [n.nid for n in self.hypotheses(live_only=True) if n.kind == NodeKind.HYPOTHESIS]:
            if h in killed:
                continue
            if self._killable(h, b, leader):
                self._fail(h, now, f"posterior {b[h]:.3f}, {self.against[h]} results against, Bayes factor "
                                   f"{math.exp(self.logev[leader] - self.logev[h]):.1f} to the leader")
                killed.append(h)
        cancelled: list = []
        for h in killed:
            cancelled += self._cancel_under(h, now, f"branch {h} failed")
        if killed:
            self._belief = self._floor_unknown(normalise({k: v for k, v in self._belief.items() if k in self.live_ids()}))
            b = self.belief()
            self._log(now, "redirect", self.root, f"research moves from {sorted(killed)} to {sorted(k for k in b if k != UNKNOWN_HID) or [UNKNOWN_HID]}")
        supported: list = []
        for n in self.hypotheses(live_only=True):
            if n.kind != NodeKind.HYPOTHESIS or n.status != NodeStatus.LIVE:
                continue
            if b[n.nid] >= self.cfg.support_posterior and self.margin() >= self.cfg.support_margin and self.n_results() >= self.cfg.min_results_to_support:
                self._put(replace(n, status=NodeStatus.SUPPORTED))
                supported.append(n.nid)
                self._log(now, "supported", n.nid, f"posterior {b[n.nid]:.3f}")
        if supported:
            for n in self.nodes.values():
                if n.status == NodeStatus.PENDING and n.kind != NodeKind.QUESTION:
                    self._put(replace(n, status=NodeStatus.CANCELLED, note="question resolved"))
                    cancelled.append(n.nid)
        self.state = self._compute_state(now)
        if self.state == TreeState.UNKNOWN_LEADS:
            self._ensure_searches(now)
        act = self.next_action(now)
        return self._report(nid, outcome, before, tuple(killed), tuple(supported), tuple(cancelled), misfit, False, act)

    def _report(self, nid, outcome, before, killed, supported, cancelled, misfit, twice, act=None) -> UpdateReport:
        return UpdateReport(self.tree_id, nid, outcome, dict(before), self.belief(), tuple(killed), tuple(supported), tuple(cancelled),
                            misfit, twice, self.state, act)

    def n_results(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == NodeStatus.DONE and n.kind in (NodeKind.TEST, NodeKind.COUNTEREXAMPLE))

    def _compute_state(self, now) -> TreeState:
        named = [n for n in self.nodes.values() if n.kind == NodeKind.HYPOTHESIS]
        if any(n.status == NodeStatus.SUPPORTED for n in named):
            return TreeState.RESOLVED
        if named and all(n.status == NodeStatus.FAILED for n in named):
            return TreeState.UNKNOWN_LEADS
        if not any(self.eig(n.nid) > 1e-9 for n in self.nodes.values() if n.status == NodeStatus.PENDING):
            return TreeState.STALLED
        return TreeState.OPEN

    def _ensure_searches(self, now) -> None:
        """Every named hypothesis died: make sure the unknown branch has something to do, or the tree just stops."""
        if any(n.status == NodeStatus.PENDING for n in self.children(UNKNOWN_HID)):
            return
        for i, txt in enumerate(("look for a shared feature among the cases the failed hypotheses could not explain",
                                 "compare the unexplained cases against matched cases the hypotheses did explain")):
            self.add_search(f"s_auto_{self.n_results()}_{i}", txt, coverage=0.25, cost=2.0, at=now)
        self.state = TreeState.UNKNOWN_LEADS

    # ---------------------------------------------------------------------------------------------------- choosing
    def pending(self) -> list:
        return [n for n in self.nodes.values() if n.status == NodeStatus.PENDING and n.kind in (NodeKind.TEST, NodeKind.COUNTEREXAMPLE, NodeKind.SEARCH)]

    def _value(self, node: Node) -> tuple:
        e = self.eig(node.nid)
        return e, e / (node.cost ** self.cfg.cost_exponent)

    def next_action(self, now=None) -> Action | None:
        """The pending item with the highest expected information gain per cost^exponent over the SURVIVING explanations. Items
        under dead branches never appear. None when nothing pending can move the belief (the tree is then STALLED or resolved)."""
        ranked = []
        for n in self.pending():
            e, v = self._value(n)
            if v > 1e-9:
                ranked.append((-v, n.nid, n, e))
        if not ranked:
            return None
        ranked.sort(key=lambda r: (r[0], r[1]))
        v, n, e = -ranked[0][0], ranked[0][2], ranked[0][3]
        alive = [k for k in self.belief() if k != UNKNOWN_HID]
        return Action(n.nid, self.tree_id, n.kind, n.text, e, n.cost, v,
                      f"highest information per cost among {len(self.pending())} pending; {len(alive)} named branches alive, "
                      f"{self.failed_count()} failed, leader {self.leader()[0]} at {self.leader()[1]:.2f}")

    def plan(self, k: int = 3, now=None) -> list:
        """The next k actions, best first, never two with the same likelihood table (they would count as one result)."""
        scored = []
        for n in self.pending():
            e, v = self._value(n)
            if v > 1e-9:
                scored.append((v, e, n))
        scored.sort(key=lambda t: (-t[0], t[2].nid))
        out, seen = [], set()
        for v, e, n in scored:
            if n.kind == NodeKind.TEST and n.signature in seen:
                continue
            seen.add(n.signature)
            out.append(Action(n.nid, self.tree_id, n.kind, n.text, e, n.cost, v, "planned"))
            if len(out) >= k:
                break
        return out

    def failed_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED)

    def failed_branches(self) -> list:
        return [(n.nid, n.note) for n in sorted(self.nodes.values(), key=lambda n: n.nid) if n.status == NodeStatus.FAILED]

    def answer(self) -> tuple | None:
        """(hypothesis id, posterior) when the question is resolved; None otherwise. UNKNOWN_HID when only the unknown remains."""
        for n in self.hypotheses():
            if n.status == NodeStatus.SUPPORTED:
                return n.nid, self.belief()[n.nid]
        if self.state == TreeState.UNKNOWN_LEADS:
            return UNKNOWN_HID, self.belief().get(UNKNOWN_HID, 1.0)
        return None

    def indistinguishable_pairs(self, min_tv: float = 0.15) -> list:
        """Pairs of live hypotheses that NO pending or finished test separates by at least `min_tv` (total variation between
        their outcome rows). Such a pair cannot be resolved by more of the same tests: the tree needs a new kind of test."""
        live = self.live_ids()
        best: dict = {}
        for n in self.nodes.values():
            if n.kind not in (NodeKind.TEST, NodeKind.COUNTEREXAMPLE) or n.status == NodeStatus.CANCELLED:
                continue
            try:
                tab = self.test_table(n.nid)
            except TreeError:
                continue
            outs = sorted({o for r in tab.values() for o in r})
            for i, a in enumerate(live):
                for b in live[i + 1:]:
                    tv = 0.5 * sum(abs(tab[a].get(o, 0.0) - tab[b].get(o, 0.0)) for o in outs)
                    best[(a, b)] = max(best.get((a, b), 0.0), tv)
        return sorted((a, b, best.get((a, b), 0.0)) for i, a in enumerate(live) for b in live[i + 1:] if best.get((a, b), 0.0) < min_tv)

    def stalled_for(self, now) -> float:
        """Days since the last result; a tree that has not moved for `stall_days` is reported so it can be re-planned or dropped."""
        last = [e for e in self.events if e.kind in ("result", "created", "spawn")]
        return (to_ts(now) - to_ts(last[-1].at)).total_seconds() / 86400.0 if last else 0.0

    # ---------------------------------------------------------------------------------------------------- integrity
    def validate(self) -> list:
        errs = []
        b = self.belief()
        if abs(sum(b.values()) - 1.0) > 1e-6:
            errs.append(f"belief sums to {sum(b.values()):.6f}")
        if b.get(UNKNOWN_HID, 0.0) < self.cfg.unknown_floor - 1e-6:
            errs.append("unknown explanation fell below its floor")
        if self.nodes[UNKNOWN_HID].status != NodeStatus.LIVE:
            errs.append("the unknown explanation must always stay LIVE")
        for n in self.nodes.values():
            if n.nid != self.root and n.parent not in self.nodes:
                errs.append(f"{n.nid}: parent {n.parent!r} missing")
            if n.kind in (NodeKind.TEST, NodeKind.COUNTEREXAMPLE) and n.status == NodeStatus.PENDING and self._owner_dead(n):
                errs.append(f"{n.nid}: pending under a failed branch (should be cancelled)")
            if n.status == NodeStatus.FAILED and n.nid in b:
                errs.append(f"{n.nid}: failed hypothesis still holds belief")
            if n.status == NodeStatus.DONE and not n.result:
                errs.append(f"{n.nid}: done without a result")
            leak = identity_leak(n.text)
            if leak:
                errs.append(f"{n.nid}: text {leak}")
        if sum(1 for n in self.nodes.values() if n.status == NodeStatus.SUPPORTED) > 1:
            errs.append("more than one hypothesis is SUPPORTED")
        return errs

    def state_hash(self) -> str:
        return stable_hash({"nodes": sorted((n.nid, n.status.value, n.result) for n in self.nodes.values()),
                            "belief": {k: round(v, 9) for k, v in sorted(self.belief().items())}}, 16)

    # ---------------------------------------------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        nodes = []
        for n in sorted(self.nodes.values(), key=lambda n: n.nid):
            d = {k: getattr(n, k) for k in n.__dataclass_fields__}
            d["kind"], d["status"], d["table"] = n.kind.value, n.status.value, [list(r) for r in n.table]
            nodes.append(d)
        return {"tree_id": self.tree_id, "created_real": self.created_real, "problem": self.problem, "cfg": dict(self.cfg.__dict__),
                "state": self.state.value, "nodes": nodes, "belief": dict(sorted(self._belief.items())), "logev": dict(self.logev),
                "against": dict(self.against), "signatures": dict(self.seen_signatures),
                "events": [[e.seq, e.at, e.kind, e.nid, e.detail] for e in self.events]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "HypothesisTree":
        t = cls.__new__(cls)
        t.cfg = TreeConfig(**d["cfg"])
        t.tree_id, t.created_real, t.problem = d["tree_id"], d["created_real"], d.get("problem", "")
        t.namespace = Namespace.MATURED_RESEARCH
        t.state = TreeState(d["state"])
        t.nodes = {}
        for nd in d["nodes"]:
            nd = dict(nd)
            nd["kind"], nd["status"] = NodeKind(nd["kind"]), NodeStatus(nd["status"])
            nd["table"] = tuple((h, o, float(p)) for h, o, p in nd["table"])
            t.nodes[nd["nid"]] = Node(**nd)
        t.root = next(n.nid for n in t.nodes.values() if n.kind == NodeKind.QUESTION)
        t._belief = {k: float(v) for k, v in d["belief"].items()}
        t.logev = {k: float(v) for k, v in d["logev"].items()}
        t.against = {k: int(v) for k, v in d["against"].items()}
        t.seen_signatures = dict(d["signatures"])
        t.events = [TreeEvent(*e) for e in d["events"]]
        return t

    def clone(self) -> "HypothesisTree":
        return HypothesisTree.from_dict(json.loads(json.dumps(self.to_dict())))

    def render(self) -> str:
        """The tree as text, in the shape section 41 draws: question, hypotheses with their posterior, tests and counterexamples."""
        b = self.belief()
        lines = [f"Question: {self.nodes[self.root].text}   [{self.state.value}]"]
        kids = self.children(self.root)
        for i, h in enumerate(kids):
            last = i == len(kids) - 1
            p = f" p={b[h.nid]:.2f}" if h.nid in b else ""
            lines.append(f"{'`--' if last else '+--'} {h.text} [{h.status.value}]{p}")
            sub = self.children(h.nid)
            for j, t in enumerate(sub):
                tail = "`--" if j == len(sub) - 1 else "+--"
                res = f" -> {t.result}" if t.result else ""
                lines.append(f"{'   ' if last else '|  '}{tail} {t.kind.value.lower()}: {t.text} [{t.status.value}]{res}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------ builders

def targeted_table(target: str, hids: Sequence[str], sensitivity: float, specificity: float, include_unknown: bool = True) -> tuple:
    """A test aimed at one hypothesis: outcome 'confirms' with probability `sensitivity` when `target` is true and with
    probability 1 - specificity when any rival is true. The unknown branch is handled by the tree (uniform)."""
    if not (0.5 < sensitivity < 1.0 and 0.5 < specificity < 1.0):
        raise ValueError("a targeted test needs sensitivity and specificity in (0.5, 1): otherwise it does not discriminate")
    rows = []
    for h in hids:
        p = sensitivity if h == target else 1.0 - specificity
        rows += [(h, "confirms", p), (h, "refutes", 1.0 - p)]
    return tuple(rows)


def build_tree(tree_id: str, question_text: str, created_real: str, hypotheses: Sequence[Hypothesis], expected: Sequence = (),
               cfg: TreeConfig | None = None, problem: str = "", targeted: Sequence[tuple] = ((0.80, 0.80), (0.70, 0.90)),
               searches: int = 2) -> HypothesisTree:
    """Question + hypothesis set (+ the shared discriminating experiment from `expected`) -> a complete tree:
    one shared test under the leading hypothesis, `targeted` (sensitivity, specificity) tests and one counterexample search
    under each hypothesis, and `searches` searches under the unknown branch. Noise/measurement hypotheses get no
    counterexample search (a claim of 'nothing here' has no counterexample to look for)."""
    t = HypothesisTree(tree_id, question_text, created_real, cfg, problem=problem)
    unknown_prior = sum(h.prior for h in hypotheses if h.hid == UNKNOWN_HID)
    hypotheses = [h for h in hypotheses if h.hid != UNKNOWN_HID]
    for h in hypotheses:
        t.add_hypothesis(h.hid, h.statement, max(h.prior, 1e-3), universal=False)
    t.seal_priors(unknown_prior)
    hids = [h.hid for h in hypotheses]
    lead = max(hypotheses, key=lambda h: (h.prior, h.hid))
    if expected:
        table = tuple((o.hid, o.outcome, o.probability) for o in expected if o.hid != UNKNOWN_HID)
        t.add_test(f"t_shared_{stable_hash(table, 4)}", lead.hid, "shared discriminating experiment across all explanations", table, cost=2.0)
    for h in hypotheses:
        for i, (sens, spec) in enumerate(targeted):
            t.add_test(f"t_{h.hid}_{i}", h.hid, f"targeted check {i + 1} of: {h.statement}", targeted_table(h.hid, hids, sens, spec), cost=1.0 + i)
        if h.kind in ("explanation", "mechanism"):
            t.add_counterexample(f"c_{h.hid}", h.hid, f"search for a case that contradicts: {h.statement}", cost=1.5)
    for i in range(searches):
        t.add_search(f"s_{i}", ("widen the feature set and look for a shared factor in the unexplained cases",
                                "match unexplained cases to explained ones and look for the difference")[i % 2], coverage=0.25, cost=2.0)
    return t


def tree_for_signal(tree_id: str, kind: SignalKind, subject: str, question_text: str, created_real: str, profile: Mapping | None = None,
                    cfg: TreeConfig | None = None, problem: str = "") -> HypothesisTree:
    """A tree from a signal kind, using the SAME hypothesis sets and outcome tables the learning-side priority engine uses, so a
    question and its tree never disagree about what the candidate explanations are."""
    if kind in (SignalKind.FAILURE, SignalKind.SURPRISE):
        hyps = failure_hypotheses(profile or {}) if kind == SignalKind.FAILURE else generic_hypotheses(kind, subject)
    else:
        hyps = generic_hypotheses(kind, subject)
    return build_tree(tree_id, question_text, created_real, hyps, expected_for(kind, hyps), cfg, problem)


# ------------------------------------------------------------------------------------------------------ the forest

@dataclass(frozen=True)
class TreeItem:
    """A pending experiment exported to the priority engine (engine.research.priority): the tree says how informative it is;
    the priority engine decides whether the information is worth its compute against everything else."""
    tree_id: str
    nid: str
    kind: str
    text: str
    eig_bits: float
    cost: float
    problem: str
    leader_share: float
    unknown_share: float


@dataclass(frozen=True)
class ForestStep:
    reports: tuple
    actions: tuple
    resolved: tuple                     # tree ids resolved by this step
    unknown_leads: tuple                # tree ids whose named hypotheses all failed
    killed: tuple                       # (tree id, hypothesis id) failed by this step
    stalled: tuple                      # tree ids with nothing left that can move the belief
    rejected: tuple                     # (tree id, node id, reason) results refused


class TreeForest:
    """All open questions' trees. One entry point (`step`) applies results and returns what to run next."""

    def __init__(self, cfg: TreeConfig | None = None):
        self.cfg = cfg or TreeConfig()
        self.trees: dict = {}

    def __len__(self) -> int:
        return len(self.trees)

    def add(self, tree: HypothesisTree) -> str:
        if tree.tree_id in self.trees:
            raise TreeError(f"tree {tree.tree_id!r} already exists")
        errs = tree.validate()
        if errs:
            raise TreeError("refusing an invalid tree: " + "; ".join(errs))
        self.trees[tree.tree_id] = tree
        return tree.tree_id

    def find_similar(self, question_text: str) -> str | None:
        """A tree that already asks this question (same question key): a duplicate question must join it, not fork it."""
        key = question_key(question_text)
        for t in self.trees.values():
            if question_key(t.nodes[t.root].text) == key:
                return t.tree_id
        return None

    def open_trees(self) -> list:
        return [t for t in self.trees.values() if t.state == TreeState.OPEN]

    def items(self, k_per_tree: int = 3) -> list:
        """Every open tree's best pending items, for the priority engine."""
        out = []
        for t in sorted(self.open_trees(), key=lambda t: t.tree_id):
            lead, share = t.leader()
            u = t.belief().get(UNKNOWN_HID, 0.0)
            for a in t.plan(k_per_tree):
                out.append(TreeItem(t.tree_id, a.nid, a.kind.value, a.text, a.eig_bits, a.cost, t.problem, share, u))
        return sorted(out, key=lambda i: (-i.eig_bits / max(i.cost, 0.01) ** 0.5, i.tree_id, i.nid))

    def actions(self, now, k: int = 5) -> list:
        acts = [a for t in self.open_trees() for a in [t.next_action(now)] if a is not None]
        return sorted(acts, key=lambda a: (-a.value, a.tree_id, a.nid))[:k]

    def redirect_log(self) -> list:
        return [(t.tree_id, e.at, e.detail) for t in self.trees.values() for e in t.events if e.kind == "redirect"]

    def stalled(self, now) -> list:
        return sorted(t.tree_id for t in self.trees.values() if t.state == TreeState.OPEN and t.stalled_for(now) > self.cfg.stall_days)

    def summary(self) -> dict:
        counts: dict = {}
        for t in self.trees.values():
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return {"trees": len(self.trees), "by_state": counts, "failed_branches": sum(t.failed_count() for t in self.trees.values()),
                "pending": sum(len(t.pending()) for t in self.trees.values())}

    def to_json(self) -> str:
        return json.dumps({"trees": [t.to_dict() for _, t in sorted(self.trees.items())]}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TreeForest":
        f = cls()
        for d in json.loads(text)["trees"]:
            t = HypothesisTree.from_dict(d)
            f.trees[t.tree_id] = t
        return f


def step(forest: TreeForest, results: Sequence[TestResult], now, k: int = 5) -> ForestStep:
    """Apply finished experiments, then say what to run next. A result for an unknown tree or node, an impossible outcome, or an
    evidence date at/after `now` is REJECTED (returned, not raised) so one bad row cannot block the rest; a FirewallBreach for
    a future-dated result is still raised, because that is a clock bug the caller must see."""
    reports, rejected = [], []
    killed, resolved, unk = [], [], []
    for r in sorted(results, key=lambda r: (r.evidence_through, r.tree_id, r.nid)):
        tree = forest.trees.get(r.tree_id)
        if tree is None:
            rejected.append((r.tree_id, r.nid, "unknown tree"))
            continue
        require_past(r.evidence_through, now, f"result {r.nid}")
        try:
            if tree.get(r.nid).kind == NodeKind.SEARCH:
                rep = tree.record_search(r.nid, r.outcome == "candidate", now, r.evidence_through)
            else:
                rep = tree.record_result(r.nid, r.outcome, now, r.evidence_through, verified=r.verified)
        except TreeError as e:
            rejected.append((r.tree_id, r.nid, str(e)))
            continue
        reports.append(rep)
        killed += [(r.tree_id, h) for h in rep.killed]
        if rep.state == TreeState.RESOLVED and rep.supported:
            resolved.append(r.tree_id)
        if rep.state == TreeState.UNKNOWN_LEADS:
            unk.append(r.tree_id)
    stalled = sorted(t.tree_id for t in forest.trees.values() if t.state == TreeState.STALLED)
    return ForestStep(tuple(reports), tuple(forest.actions(now, k)), tuple(sorted(set(resolved))), tuple(sorted(set(unk))),
                      tuple(killed), tuple(stalled), tuple(rejected))


# ------------------------------------------------------------------------------------------------------ simulation

def _draw(rng: np.random.Generator, probs: Mapping) -> str:
    keys = sorted(probs)
    p = np.array([max(probs[k], 0.0) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def true_outcome(tree: HypothesisTree, nid: str, truth: str | None, rng: np.random.Generator) -> str:
    """Draw the outcome an experiment would produce if `truth` (a hypothesis id, or None = none of the named ones) were the
    real explanation. Used only by tests and by resolution_forecast: the tree itself never sees the truth."""
    node = tree.get(nid)
    if node.kind == NodeKind.COUNTEREXAMPLE:
        p = node.false_alarm if truth == node.parent else node.sensitivity
        return "found" if rng.random() < p else "not_found"
    if node.kind == NodeKind.SEARCH:
        return "candidate" if truth is None and rng.random() < node.prior else "nothing"
    outs = node.outcomes()
    row = node.likelihood_row(truth) if truth is not None else {}
    return _draw(rng, row if row else {o: 1.0 / len(outs) for o in outs})


@dataclass(frozen=True)
class Trace:
    steps: int
    final_state: TreeState
    answer: tuple | None
    killed: tuple
    order: tuple


def simulate_investigation(tree: HypothesisTree, truth: str | None, rng: np.random.Generator, start: str = "2000-01-03",
                           max_steps: int = 30, verified_counterexamples: bool = True) -> Trace:
    """Run the tree against a known truth until it resolves, hits UNKNOWN_LEADS, stalls, or max_steps. Each step happens on a
    later synthetic day than the evidence it uses."""
    import datetime as _dt
    day = _dt.date.fromisoformat(start)
    order, killed = [], []
    for i in range(max_steps):
        act = tree.next_action(day)
        if act is None or tree.state in (TreeState.RESOLVED,):
            break
        ev, now = day + _dt.timedelta(days=1), day + _dt.timedelta(days=2)
        out = true_outcome(tree, act.nid, truth, rng)
        if act.kind == NodeKind.SEARCH:
            if out == "candidate":
                tree.record_search(act.nid, True, now.isoformat(), ev.isoformat(), text=f"candidate found by {act.nid}", mass=0.10)
            else:
                tree.record_search(act.nid, False, now.isoformat(), ev.isoformat())
            rep = None
        else:
            rep = tree.record_result(act.nid, out, now.isoformat(), ev.isoformat(), verified=verified_counterexamples and out == "found")
            killed += list(rep.killed)
        order.append(act.nid)
        day = now
    return Trace(len(order), tree.state, tree.answer(), tuple(killed), tuple(order))


def resolution_forecast(tree: HypothesisTree, seed: int, n_sim: int = 100, max_steps: int = 25) -> dict:
    """Before spending anything: if the truth is drawn from the CURRENT belief, how often does this tree end on the right
    answer, how many experiments does that take, and how often does it end wrong or unknown? Uses clones, never the tree."""
    rng = np.random.default_rng(seed)
    b = tree.belief()
    keys = sorted(b)
    p = np.array([b[k] for k in keys])
    right = wrong = unknown = unresolved = 0
    steps = []
    for _ in range(n_sim):
        truth = keys[int(rng.choice(len(keys), p=p / p.sum()))]
        tr = simulate_investigation(tree.clone(), None if truth == UNKNOWN_HID else truth, rng, max_steps=max_steps)
        steps.append(tr.steps)
        if tr.answer is None:
            unresolved += 1
        elif tr.answer[0] == truth:
            right += 1
        elif tr.answer[0] == UNKNOWN_HID:
            unknown += 1
        else:
            wrong += 1
    return {"n": n_sim, "p_right": right / n_sim, "p_wrong": wrong / n_sim, "p_unknown_answer": unknown / n_sim,
            "p_unresolved": unresolved / n_sim, "mean_steps": float(np.mean(steps)), "max_steps": max_steps}


def kill_calibration(make_tree, truths: Sequence[str | None], seeds: Sequence[int], max_steps: int = 25) -> dict:
    """How often does the kill rule wrongly kill the TRUE hypothesis (a false kill)? `make_tree()` builds a fresh tree.
    A rule that kills the truth even once in a while is too aggressive for the evidence it demands."""
    false_kills = total = correct_kills = 0
    for truth in truths:
        for s in seeds:
            tr = simulate_investigation(make_tree(), truth, np.random.default_rng(s), max_steps=max_steps)
            total += 1
            false_kills += int(truth in tr.killed)
            correct_kills += sum(1 for k in tr.killed if k != truth)
    return {"runs": total, "false_kill_rate": false_kills / max(total, 1), "wrong_hypotheses_killed": correct_kills}


def branch_report(tree: HypothesisTree) -> list:
    """One row per hypothesis: status, belief, evidence counted against it, and the reason it died - for the research report."""
    b = tree.belief()
    rows = []
    for h in tree.hypotheses():
        rows.append({"hid": h.nid, "text": h.text, "status": h.status.value, "belief": round(b.get(h.nid, 0.0), 4),
                     "against": tree.against.get(h.nid, 0), "log_evidence": round(tree.logev.get(h.nid, 0.0), 4), "note": h.note,
                     "pending": sum(1 for c in tree.children(h.nid) if c.status == NodeStatus.PENDING),
                     "done": sum(1 for c in tree.children(h.nid) if c.status == NodeStatus.DONE)})
    return rows


def research_state_of(tree: HypothesisTree) -> ResearchState:
    """Map a tree onto the section-18 research lifecycle so the compute manager and priority engine can treat it like any job."""
    if tree.state == TreeState.RESOLVED:
        return ResearchState.VALIDATING
    if tree.state == TreeState.UNKNOWN_LEADS:
        return ResearchState.DORMANT
    if tree.state == TreeState.STALLED:
        return ResearchState.DORMANT
    return ResearchState.EXPLORING if tree.n_results() else ResearchState.QUEUED


def belief_history(tree: HypothesisTree) -> list:
    """(event seq, node id, outcome) for every result in order: the audit trail from which the belief can be replayed."""
    return [(e.seq, e.nid, e.detail) for e in tree.events if e.kind == "result"]


# ------------------------------------------------------------------------------------------------------ redirecting research

def propose_discriminating_test(tree: HypothesisTree, cost: float = 2.0, min_separation: float = 0.6, at=None) -> str | None:
    """After a branch dies the two leading survivors may be poorly separated by every test left. Design the test that separates
    them: outcome 'favours_a' is likely (0.85) under a and unlikely (0.15) under b. Adds it under the leader and returns its
    id; None when the top two are already separated by an existing pending test (total variation >= min_separation)."""
    b = tree.belief()
    named = sorted((h for h in b if h != UNKNOWN_HID), key=lambda h: (-b[h], h))
    if len(named) < 2:
        return None
    a, c = named[0], named[1]
    for n in tree.pending():
        if n.kind != NodeKind.TEST:
            continue
        tab = tree.test_table(n.nid)
        outs = sorted({o for r in tab.values() for o in r})
        if 0.5 * sum(abs(tab[a].get(o, 0.0) - tab[c].get(o, 0.0)) for o in outs) >= min_separation:
            return None
    rows = []
    for h in named + [UNKNOWN_HID]:
        p = 0.85 if h == a else 0.15 if h == c else 0.5
        rows += [(h, "favours_a", p), (h, "favours_b", 1.0 - p)]
    tid = "t_disc_" + stable_hash([tree.tree_id, a, c, len(tree.nodes)], 6)
    tree.add_test(tid, a, f"discriminate {tree.get(a).text} from {tree.get(c).text}", tuple(rows), cost=cost, at=at or tree.events[-1].at)
    tree._log(at or tree.events[-1].at, "redirect", tid, f"new test designed to separate {a} from {c}")
    return tid


def absorb_weights(tree: HypothesisTree, weights: Mapping, now, evidence_through, strength: float = 0.5) -> dict:
    """Fold in the posterior of an EXTERNAL competition (engine.learning.competition.Arena.weights(), keyed by hypothesis id) as
    soft evidence: the tree's belief moves toward it by `strength` in log space, ONLY over hypotheses both sides know. It counts
    as one result against a hypothesis the external weights put below the tree's own by a factor of two or more."""
    require_past(evidence_through, now, "external hypothesis weights")
    if not (0.0 < strength <= 1.0):
        raise ValueError("strength must be in (0, 1]")
    cur = tree.belief()
    shared = [h for h in cur if h in weights and h != UNKNOWN_HID]
    if len(shared) < 2:
        return dict(cur)
    ext = normalise({h: max(float(weights[h]), 1e-6) for h in shared})
    mass = sum(cur[h] for h in shared)
    post = dict(cur)
    mix = normalise({h: math.exp((1 - strength) * math.log(max(cur[h] / mass, 1e-9)) + strength * math.log(ext[h])) for h in shared})
    for h in shared:
        post[h] = mix[h] * mass
        if ext[h] < 0.5 * cur[h] / mass:
            tree.against[h] = tree.against.get(h, 0) + 1
        tree.logev[h] = tree.logev.get(h, 0.0) + strength * math.log(max(ext[h], 1e-9))
    tree._belief = tree._floor_unknown(normalise(post))
    tree._log(now, "result", tree.root, f"external weights absorbed over {len(shared)} hypotheses")
    tree._settle(tree.root, "external", cur, now, False, ())
    return tree.belief()


def reopen(tree: HypothesisTree, now, reason: str, keep_supported: bool = False) -> None:
    """A resolved or unknown-leading tree can be reopened when later knowledge contradicts its answer. Supported hypotheses go
    back to LIVE (unless `keep_supported`), cancelled tests become pending again, failed branches STAY failed (they were killed
    by evidence, and new doubt about the answer is not new evidence for them). The reason is required and logged."""
    if not reason.strip():
        raise TreeError("reopening a tree needs a reason")
    n_back = 0
    for n in list(tree.nodes.values()):
        if n.status == NodeStatus.SUPPORTED and not keep_supported:
            tree._put(replace(n, status=NodeStatus.LIVE))
        if n.status == NodeStatus.CANCELLED and n.note == "question resolved":
            tree._put(replace(n, status=NodeStatus.PENDING, note=""))
            n_back += 1
    tree.state = TreeState.OPEN
    tree._log(now, "reopen", tree.root, f"{reason}; {n_back} tests pending again")
    tree.state = tree._compute_state(now)


def merge_trees(a: HypothesisTree, b: HypothesisTree, now) -> HypothesisTree:
    """Two trees that turned out to ask the same question: keep `a`, import the hypotheses `b` has that `a` lacks (with the
    smaller of their beliefs, taken from the unknown branch) and the results `b` already has as events, so nothing is lost.
    Results are NOT replayed: the tables differ, so combining them would double-count."""
    if question_key(a.nodes[a.root].text) != question_key(b.nodes[b.root].text):
        raise TreeError("only trees that ask the same question can merge")
    out = a.clone()
    for h in b.hypotheses():
        if h.kind != NodeKind.HYPOTHESIS or h.nid in out.nodes or h.status == NodeStatus.FAILED:
            continue
        take = min(b.belief().get(h.nid, 0.0), 0.25)
        if take > 1e-6:
            out.spawn_hypothesis(h.nid, h.text, take, now, universal=h.universal)
            for c in b.children(h.nid):
                if c.status == NodeStatus.PENDING and c.nid not in out.nodes:
                    out._put(replace(c, parent=h.nid))
    for e in b.events:
        if e.kind in ("killed", "supported"):
            out._log(now, "merged_" + e.kind, e.nid, f"from tree {b.tree_id}: {e.detail}")
    out.state = out._compute_state(now)
    return out


def kill_robustness(tree: HypothesisTree, hid: str, seed: int, n_draws: int = 200, concentration: float = 30.0) -> dict:
    """Would the kill of `hid` survive if the likelihood tables were slightly wrong? Replays every result with each table row
    redrawn from a Dirichlet centred on the stated row (`concentration` = trust in the table). Returns the share of draws in
    which `hid` still has a Bayes factor of at least kill_bayes_factor against it. A kill that only holds under the exact
    stated tables is fragile and the report says so."""
    rng = np.random.default_rng(seed)
    done = [n for n in tree.nodes.values() if n.status == NodeStatus.DONE and n.kind == NodeKind.TEST and n.result]
    if not done:
        return {"hid": hid, "n_results": 0, "robust_share": 0.0, "verdict": "NO_EVIDENCE"}
    holds = 0
    for _ in range(n_draws):
        gap = 0.0
        for n in done:
            outs = n.outcomes()
            rows = {}
            for h in {r[0] for r in n.table}:
                base = np.array([n.likelihood_row(h).get(o, 0.0) for o in outs]) + 1e-3
                rows[h] = rng.dirichlet(base / base.sum() * concentration)
            if hid in rows and rows:
                idx = outs.index(n.result)
                lead = max(rows, key=lambda h: rows[h][idx] if h != hid else -1.0)
                gap += math.log(max(rows[lead][idx], EPS)) - math.log(max(rows[hid][idx], EPS))
        holds += int(gap >= math.log(tree.cfg.kill_bayes_factor))
    share = holds / n_draws
    return {"hid": hid, "n_results": len(done), "robust_share": share, "verdict": "ROBUST" if share >= 0.8 else "FRAGILE"}


def explain_status(tree: HypothesisTree, hid: str) -> str:
    """Why a branch is in the state it is in, from the log: the audit answer to 'why did you stop researching that?'"""
    n = tree.get(hid)
    b = tree.belief()
    hist = [e for e in tree.events if e.nid == hid or (e.kind == "redirect" and hid in e.detail)]
    lines = [f"{hid}: {n.text} -- {n.status.value}"]
    if hid in b:
        lines.append(f"  belief {b[hid]:.3f}; {tree.against.get(hid, 0)} results counted against it; log evidence {tree.logev.get(hid, 0.0):+.2f}")
    if n.note:
        lines.append(f"  note: {n.note}")
    lines += [f"  [{e.seq}] {e.at} {e.kind}: {e.detail}" for e in hist]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------ forest health

def forest_health(forest: TreeForest, now) -> dict:
    """Are the trees doing their job? Fractions that ended in an answer, ended in UNKNOWN, stalled, or sit untouched; the mean
    number of results per resolved tree; and the mean belief in the unknown branch across open trees (a tree whose unknown
    mass keeps rising is telling the researcher the hypothesis set is wrong)."""
    trees = list(forest.trees.values())
    n = max(len(trees), 1)
    by = {s: sum(1 for t in trees if t.state == s) / n for s in TreeState}
    resolved = [t for t in trees if t.state == TreeState.RESOLVED]
    open_ = [t for t in trees if t.state == TreeState.OPEN]
    return {"trees": len(trees), "share_resolved": by[TreeState.RESOLVED], "share_unknown_leads": by[TreeState.UNKNOWN_LEADS],
            "share_stalled": by[TreeState.STALLED], "share_open": by[TreeState.OPEN],
            "mean_results_to_resolve": float(np.mean([t.n_results() for t in resolved])) if resolved else None,
            "mean_unknown_belief_open": float(np.mean([t.belief().get(UNKNOWN_HID, 0.0) for t in open_])) if open_ else None,
            "untouched": sorted(t.tree_id for t in open_ if t.n_results() == 0 and t.stalled_for(now) > forest.cfg.stall_days),
            "failed_branches": sum(t.failed_count() for t in trees)}


def killed_hypothesis_table(forest: TreeForest) -> list:
    """Every failed branch across the forest with its reason and the number of results it faced: the record that a hypothesis
    was tested and died, so the same idea is not proposed again without new evidence (feeds experiment memory)."""
    rows = []
    for t in sorted(forest.trees.values(), key=lambda t: t.tree_id):
        for hid, why in t.failed_branches():
            rows.append({"tree": t.tree_id, "hid": hid, "text": t.get(hid).text, "why": why, "against": t.against.get(hid, 0)})
    return rows


def prune_cancelled(tree: HypothesisTree) -> int:
    """Number of pending items still under a dead branch (should be zero after any result). A non-zero count is a bug signal."""
    return sum(1 for n in tree.pending() if tree._owner_dead(n))
