"""Creator K23 - oversight: the system watches its own direction (C77 checklist CR057 goal drift, CR059 knowledge gaps, CR083 critical
path) - IMPLEMENTED, NOT VALIDATED.

    goal_drift(ledger)      every ADOPTED change must trace (through its experiment, change proposal, package, gap and
                            requirement) to an objective the OWNER set; it also flags work concentrating in one kind while the
                            goal's own metrics do not move. Untraceable adoptions are drift.
    knowledge_gaps(ledger)  a failure the debugger could not classify (UNKNOWN / undetermined root cause) is something the system
                            does not know; each becomes a KNOWLEDGE gap with a ResearchQuestion under the self objective
                            (idempotent per failure) - the planner and research module pick it up like any other gap.
    critical_path(ledger)   open gaps ranked by how many other open gaps they block, directly or transitively - the work that
                            unblocks the most comes first (reported in STATUS; the planner's ranking already puts unblocked
                            gaps first).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator.ledger import Ledger


# ------------------------------------------------------------------------------------------------ goal drift

@dataclasses.dataclass(frozen=True)
class DriftReport:
    adopted: int
    traced_to_owner: int
    untraced: tuple[str, ...]
    by_kind: dict[str, int]
    concentrated: Optional[str]                     # a kind with > `concentration` of recent adoptions, if any
    drifting: bool
    why: str


def _kind(ledger: Ledger, experiment_id: str) -> str:
    for a in ledger.lineage(experiment_id):
        rec = ledger.view.by_id[a].record
        if isinstance(rec, M.Requirement):
            return rec.key.split(".")[-1] if rec.key.startswith("K") else rec.key
    return "unknown"


def goal_drift(ledger: Ledger, recent: int = 10, concentration: float = 0.8) -> DriftReport:
    owner_objectives = {e.id for e in ledger.of_type("Objective") if e.record.created_by is M.Role.OWNER}
    adopted = [e for e in ledger.of_type("Decision") if getattr(e.record, "verdict") is M.DecisionVerdict.ADOPT]
    untraced, kinds = [], []
    for e in adopted:
        subject = getattr(e.record, "subject_id")
        if subject not in ledger.view.by_id:
            untraced.append(e.id)
            continue
        if not owner_objectives & set(ledger.lineage(subject)):
            untraced.append(e.id)
        kinds.append(_kind(ledger, subject))
    by_kind: dict[str, int] = {}
    for k in kinds:
        by_kind[k] = by_kind.get(k, 0) + 1
    last = kinds[-recent:]
    top = max(set(last), key=last.count) if last else None
    concentrated = top if top and len(last) >= recent and last.count(top) / len(last) > concentration else None
    reasons = []
    if untraced:
        reasons.append(f"{len(untraced)} adoption(s) do not trace to an owner objective")
    if concentrated:
        reasons.append(f"{last.count(concentrated)}/{len(last)} recent adoptions are '{concentrated}' work")
    return DriftReport(len(adopted), len(adopted) - len(untraced), tuple(untraced), by_kind, concentrated, bool(untraced),
                       "; ".join(reasons) or "every adoption serves an owner objective")


# ------------------------------------------------------------------------------------------------ knowledge gaps

KNOWLEDGE_KEY = "GOAL.knowledge"


def knowledge_gaps(ledger: Ledger) -> list[str]:
    """Turn each unexplained failure into a KNOWLEDGE gap + ResearchQuestion (once per failure). Returns the new gap ids."""
    unexplained = [e for e in ledger.of_type("Diagnosis")
                   if getattr(e.record, "root_cause", "").startswith("undetermined")
                   or ledger.get(getattr(e.record, "failure_id")).classification == "UNKNOWN"]   # type: ignore[attr-defined]
    if not unexplained:
        return []
    objective = O.self_objective(ledger)
    req = ledger.view.unique.get(("Requirement", KNOWLEDGE_KEY)) or ledger.append(M.Requirement(
        created_by=M.Role.KERNEL, parents=(objective,), key=KNOWLEDGE_KEY, priority=M.Priority.HIGH,
        description="the system can explain its own failures (what it does not know becomes research)",
        acceptance_test="knowledge:explained (a standing objective)", measurement_method="creator.oversight.knowledge_gaps",
        failure_condition="a failure stays unexplained", validation_method="a later diagnosis of the same failure class",
        evidence_location="ledger Diagnosis / Finding records"))
    done = {getattr(ledger.get(g), "description", "") for g in ledger.view.children.get(req, [])}
    new = []
    for d in unexplained:
        fid = getattr(d.record, "failure_id")
        label = f"unexplained failure {fid}"
        if any(label in x for x in done):
            continue
        fail = ledger.get(fid)
        gid = ledger.append(M.Gap(created_by=M.Role.KERNEL, parents=(req,), kind=M.GapKind.KNOWLEDGE,
                                  description=f"{label}: {getattr(fail, 'symptom', '')[:200]}", importance=0.45))
        ledger.append(M.ResearchQuestion(created_by=M.Role.KERNEL, parents=(gid, d.id),
                                         question=f"Why does this fail: {getattr(fail, 'symptom', '')[:300]}? "
                                                  f"Hypotheses so far: {'; '.join(getattr(d.record, 'hypotheses', ())[:2])}",
                                         priority=M.Priority.HIGH))
        new.append(gid)
    return new


# ------------------------------------------------------------------------------------------------ critical path

@dataclasses.dataclass(frozen=True)
class PathItem:
    gap_id: str
    requirement_key: str
    blocks: int                                     # open gaps that wait on it, directly or transitively
    importance: float


def critical_path(ledger: Ledger) -> list[PathItem]:
    ranked = G.ranked(ledger)
    blocked_by = {g.gap_id: set(g.blocked_by) for g in ranked}
    waiting: dict[str, set[str]] = {g.gap_id: set() for g in ranked}
    for g, deps in blocked_by.items():
        for d in deps:
            waiting.setdefault(d, set()).add(g)

    def closure(gid: str) -> set[str]:
        seen: set[str] = set()
        todo = list(waiting.get(gid, ()))
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                todo.extend(waiting.get(x, ()))
        return seen
    items = [PathItem(g.gap_id, g.requirement_key, len(closure(g.gap_id)), g.importance) for g in ranked]
    return sorted(items, key=lambda i: (-i.blocks, -i.importance, i.requirement_key))


def summary(ledger: Ledger) -> dict[str, Any]:
    d = goal_drift(ledger)
    cp = critical_path(ledger)[:5]
    return {"goal_drift": dataclasses.asdict(d), "critical_path": [dataclasses.asdict(i) for i in cp]}
