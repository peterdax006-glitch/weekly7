"""Creator CR082/CR083 - scheduling: Nupen decides its own work order (C77 secs 18-21).

Open gaps form a dependency graph (gaps.ranked: recorded + live requirement dependencies). Each node is costed from the ledger's own
history (median seconds a package of that ladder step took, and how often it succeeded; defaults when unknown). The CRITICAL PATH is
the longest cost-weighted chain of open gaps; a node's slack says how long it could wait without delaying the objective. `next_batch`
returns up to `slots` packages that depend on nothing open, touch no common file (component modules + tests) and none a running worker
holds, critical ones first - each with a recorded reason (plan_explanations.jsonl).

Wiring (swarm.run_round, instead of repeated K.plan_one): `plans = schedule.plan_batch(cfg, led, main, base_sha, slots, held,
held_files)` then start one worker per returned Plan. `plan_one` is untouched; plan_batch falls back to the shrink work."""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import planner as P
from creator import selfmodel as SM
from creator.ledger import Ledger

DEFAULT_COST_S = {"exists": 900.0, "tested": 600.0, "no_stubs": 900.0, "integrated": 900.0, "validated": 300.0}
FALLBACK_COST_S = 900.0
EXPLAIN_FILE = "plan_explanations.jsonl"
FAILED_STATES = (M.Status.FAILED, M.Status.ROLLED_BACK, M.Status.REJECTED)


@dataclasses.dataclass(frozen=True)
class Node:
    id: str
    component: str
    step: str
    value: float
    cost: float
    files: frozenset[str] = frozenset()
    deps: tuple[str, ...] = ()
    status: str = "ready"                      # ready | running | blocked


@dataclasses.dataclass(frozen=True)
class History:
    """Per ladder step: the durations (seconds) of finished packages and (successes, attempts)."""
    seconds: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)
    outcomes: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_samples(samples: Iterable[tuple[str, float, bool]]) -> "History":
        sec: dict[str, list[float]] = {}
        out: dict[str, tuple[int, int]] = {}
        for step, s, ok in samples:
            sec.setdefault(step, []).append(s)
            w, n = out.get(step, (0, 0))
            out[step] = (w + int(ok), n + 1)
        return History({k: tuple(v) for k, v in sec.items()}, out)

    def cost(self, step: str) -> float:
        d = self.seconds.get(step)
        return float(statistics.median(d)) if d else DEFAULT_COST_S.get(step, FALLBACK_COST_S)

    def success_rate(self, step: str) -> float:
        w, n = self.outcomes.get(step, (0, 0))
        return (w + 1) / (n + 2)                                        # Laplace: an unknown kind starts at 0.5


def _ts(e: Any) -> float:
    return datetime.fromisoformat(e.provenance.timestamp).timestamp()


def samples_from_ledger(ledger: Ledger) -> list[tuple[str, float, bool]]:
    """(step, seconds, succeeded) per finished work package: start = its IN_PROGRESS transition, end = the next transition."""
    started: dict[str, float] = {}
    out: list[tuple[str, float, bool]] = []
    for e in ledger.view.entries:
        if e.rtype != "Transition":
            continue
        r = e.record
        sid = getattr(r, "subject_id")
        if ledger.view.by_id[sid].rtype != "WorkPackage":
            continue
        to = getattr(r, "to_state")
        if to is M.Status.IN_PROGRESS:
            started[sid] = _ts(e)
        elif sid in started:
            gap = next((p for p in ledger.get(sid).parents if ledger.view.by_id[p].rtype == "Gap"), None)
            req = next((p for p in (ledger.get(gap).parents if gap else ()) if ledger.view.by_id[p].rtype == "Requirement"), None)
            t0 = started.pop(sid)
            if req is None:
                continue
            try:
                step = O.parse_check(getattr(ledger.get(req), "acceptance_test"))[0]
            except O.ObjectiveError:
                continue
            out.append((step, max(0.0, _ts(e) - t0), to not in FAILED_STATES))
    return out


def learn_history(ledger: Ledger) -> History:
    return History.from_samples(samples_from_ledger(ledger))


# ------------------------------------------------------------------------------------------------ graph and critical path

def build_nodes(ledger: Ledger, specs: Optional[Sequence[SM.CapabilitySpec]] = None, history: Optional[History] = None) -> list[Node]:
    spec_map = {s.id: s for s in (specs if specs is not None else SM.load_capabilities())}
    hist = history or learn_history(ledger)
    nodes: list[Node] = []
    for g in G.ranked(ledger):
        gap = ledger.get(g.gap_id)
        req = next((p for p in gap.parents if ledger.view.by_id[p].rtype == "Requirement"), None)
        if req is None:
            continue
        try:
            step, cid = O.parse_check(getattr(ledger.get(req), "acceptance_test"))
        except O.ObjectiveError:
            continue
        st = ledger.view.status.get(g.gap_id)
        spec = spec_map.get(cid)
        status = "running" if st is M.Status.IN_PROGRESS else "blocked" if st in (M.Status.BLOCKED, M.Status.IMPLEMENTED) else "ready"
        nodes.append(Node(g.gap_id, cid, step, g.importance * hist.success_rate(step), hist.cost(step),
                          frozenset(spec.modules + spec.tests) if spec else frozenset(), g.blocked_by, status))
    return nodes


@dataclasses.dataclass(frozen=True)
class Analysis:
    earliest: dict[str, float]          # when the node could start if everything before it ran back to back
    tail: dict[str, float]              # its own cost plus the longest chain after it
    length: float                       # the critical path length
    slack: dict[str, float]
    path: tuple[str, ...]               # one longest chain, first to last


def analyse(nodes: Sequence[Node]) -> Analysis:
    by = {n.id: n for n in nodes}
    deps = {n.id: [d for d in n.deps if d in by and d != n.id] for n in nodes}
    after: dict[str, list[str]] = {i: [] for i in by}
    for i, ds in deps.items():
        for d in ds:
            after[d].append(i)
    early: dict[str, float] = {}
    tail: dict[str, float] = {}
    busy: set[str] = set()

    def es(i: str) -> float:
        if i not in early:
            if i in busy:
                return 0.0                                              # a dependency cycle adds no length
            busy.add(i)
            early[i] = max((es(d) + by[d].cost for d in deps[i]), default=0.0)
            busy.discard(i)
        return early[i]

    def tl(i: str) -> float:
        if i not in tail:
            if i in busy:
                return 0.0
            busy.add(i)
            tail[i] = by[i].cost + max((tl(a) for a in after[i]), default=0.0)
            busy.discard(i)
        return tail[i]

    for i in sorted(by):
        es(i)
        tl(i)
    length = max((early[i] + tail[i] for i in by), default=0.0)
    slack = {i: round(length - early[i] - tail[i], 6) for i in by}
    path: list[str] = []
    cur = min((i for i in by if slack[i] == 0 and early[i] == 0), default=None)
    while cur is not None and cur not in path:
        path.append(cur)
        cur = min((a for a in after[cur] if slack[a] == 0 and abs(early[a] - early[cur] - by[cur].cost) < 1e-6), default=None)
    return Analysis(early, tail, length, slack, tuple(path))


# ------------------------------------------------------------------------------------------------ batches

@dataclasses.dataclass(frozen=True)
class Batch:
    picks: tuple[Node, ...]
    reasons: tuple[dict[str, Any], ...]            # one row per open node: what is done with it and WHY
    analysis: Analysis


def schedule(nodes: Sequence[Node], slots: int, held_components: Sequence[str] = (), held_files: Iterable[str] = (),
             steps: Optional[Sequence[str]] = None) -> Batch:
    """Up to `slots` ready nodes with no open dependency (and a step in `steps`), no component or file in common with each other or running work."""
    a = analyse(nodes)
    by = {n.id: n for n in nodes}
    busy_c, busy_f = set(held_components), set(held_files)
    for n in nodes:
        if n.status == "running":
            busy_c.add(n.component)
            busy_f |= n.files
    order = sorted((n for n in nodes if n.status == "ready"),
                   key=lambda n: (a.slack[n.id], -a.tail[n.id], -n.value / max(n.cost, 1.0), n.id))
    picks: list[Node] = []
    why: dict[str, str] = {}
    for n in order:
        if steps is not None and n.step not in steps:
            why[n.id] = f"not worker work: {n.step} goes to its own role"
            continue
        d = next((by[x] for x in n.deps if x in by), None)
        if d is not None:
            why[n.id] = (f"blocked by {d.id} ({d.component}.{d.step}): that gap is itself blocked" if d.status == "blocked"
                         else f"waiting on dependency {d.id} ({d.component}.{d.step}, {d.status})")
        elif n.component in busy_c or n.files & busy_f:
            why[n.id] = f"deferred: shares {sorted(n.files & busy_f)[:2] or [n.component]} with work already chosen or running"
        elif len(picks) >= slots:
            why[n.id] = "deferred: no free slot"
        else:
            picks.append(n)
            busy_c.add(n.component)
            busy_f |= n.files
            where = (f"on the critical path (length {a.length:.0f}s, tail {a.tail[n.id]:.0f}s)" if a.slack[n.id] == 0
                     else f"off the critical path (slack {a.slack[n.id]:.0f}s), free to run in parallel")
            why[n.id] = f"{where}; est {n.cost:.0f}s, value {n.value:.2f}"
    for n in nodes:
        if n.id not in why:
            d = next((by[x] for x in n.deps if x in by), None)
            why[n.id] = ("worker already on it" if n.status == "running"
                         else "blocked: the kernel gave up on this gap after repeated failures" if n.status == "blocked"
                         else f"waiting on dependency {d.id if d else '?'}")
    rows = tuple({"node": n.id, "component": n.component, "step": n.step, "chosen": n in picks, "slack_s": a.slack[n.id],
                  "cost_s": n.cost, "value": round(n.value, 4), "why": why[n.id]} for n in nodes)
    return Batch(tuple(picks), rows, a)


def next_batch(ledger: Ledger, slots: int, specs: Optional[Sequence[SM.CapabilitySpec]] = None, held_components: Sequence[str] = (),
               held_files: Iterable[str] = (), history: Optional[History] = None, explain_path: Optional[Path] = None,
               steps: Optional[Sequence[str]] = None) -> Batch:
    """The swarm's call: what to run together now, and why. Writes the reasons when `explain_path` is given."""
    b = schedule(build_nodes(ledger, specs, history), slots, held_components, held_files, steps)
    if explain_path is not None:
        record_explanations(explain_path, [r for r in b.reasons if r["chosen"] or r["slack_s"] == 0], b.analysis)
    return b


def explain_path_for(ledger_path: Any) -> Path:
    return Path(ledger_path).parent / EXPLAIN_FILE


def record_explanations(path: Path, rows: Sequence[dict[str, Any]], a: Analysis) -> None:
    """Nupen thinking out loud. No ledger record type fits a planning decision, so: one JSON line per decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"at": at, "critical_path_s": a.length, **r}, sort_keys=True) + "\n")


def plan_batch(cfg: Any, led: Ledger, main: Any, base_sha: str, slots: int, held: Sequence[str] = (),
               held_files: Iterable[str] = ()) -> list[P.Plan]:
    """Adapter for swarm.run_round: writes the package chain for each pick of next_batch; mode 'efficiency' or a shortfall falls
    back to the standing shrink work, as kernel.plan_one does."""
    plans: list[P.Plan] = []
    specs = cfg.specs()
    steps = tuple(s for s in cfg.steps if s in P.WORKER_STEPS)
    taken = list(held)
    if cfg.mode in ("auto", "gaps"):
        b = next_batch(led, slots, specs, held, held_files, explain_path=explain_path_for(cfg.ledger_path), steps=steps)
        for n in b.picks:
            p = P.plan_next(led, main.model, base_sha, specs, steps=steps, exclude_components=taken, prefer=(n.id,))
            if p is not None and p.gap_id == n.id:
                plans.append(p)
                taken.append(p.component)
    if len(plans) < slots and cfg.mode in ("auto", "efficiency"):
        p2 = P.plan_efficiency(led, cfg.repo, base_sha, avoid=tuple(held_files))
        if p2 is not None:
            plans.append(p2)
    return plans
