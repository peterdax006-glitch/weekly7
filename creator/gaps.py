"""Creator K04 - gap analysis (C77 secs 14, 41, 42; package CR08) - IMPLEMENTED, NOT VALIDATED.

`assess()` recomputes every requirement's acceptance check (creator.objective.CHECKS) against the evidence-derived self-model.
`sync()` turns the assessment into ledger facts, every one of them computed:

    unmet, no open gap          -> Gap(kind from the ladder, importance, depends_on = the open gaps of its dependencies)
    met                         -> TestRun(command = the check, passed=1, evidence = the saved self-model snapshot) and the
                                   requirement (and its open gaps) advanced along the legal path to TESTED with that TestRun
                                   as justification
    previously TESTED, now unmet -> requirement -> FAILED (a regression is never silently absorbed) and a new gap is opened

`ranked()` orders open gaps for the planner: unblocked first (no open dependency gaps), then importance. Nothing here decides
VALIDATED - only an independent role can (ledger rule)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from creator import model as M
from creator import objective as O
from creator import selfmodel as SM
from creator.ledger import Ledger

SNAPSHOT_DIR = Path("state") / "creator" / "selfmodel"


@dataclasses.dataclass(frozen=True)
class Assessment:
    requirement_id: str
    key: str
    component: str
    step: str
    met: bool
    detail: str
    kind: M.GapKind
    importance: float
    status: M.Status
    stale: bool = False                     # unmet only because evidence is stale / not run: needs a re-test, not a failure


def assess(ledger: Ledger, model: SM.SelfModel) -> list[Assessment]:
    caps = {c.id: c for c in model.capabilities}
    cap_records = {getattr(e.record, "component", None): e.id for e in ledger.of_type("Capability")}
    out: list[Assessment] = []
    for e in ledger.of_type("Requirement"):
        rec = e.record
        try:
            step, comp = O.parse_check(getattr(rec, "acceptance_test"))
        except O.ObjectiveError:
            continue                                                    # owner-written requirement without a computed check
        c = caps.get(comp)
        if c is None:
            met, detail = False, f"component {comp} is not declared in the self-model"
        else:
            met, detail = O.CHECKS[step](model, c, ledger, cap_records.get(comp))
        kind = next(k for s, k, _w, _d in O.LADDER if s == step)
        stale = (not met and step == "tested" and c is not None and bool(c.last_results)
                 and not any(v in ("FAIL", "ERROR") for v in c.last_results.values())
                 and any(v in ("STALE", "NOT_RUN") for v in c.last_results.values()))
        out.append(Assessment(e.id, getattr(rec, "key"), comp, step, met, detail, kind,
                              O.importance(getattr(rec, "priority"), step, comp),
                              ledger.view.status.get(e.id, M.Status.NOT_STARTED), stale))
    return out


def open_gaps_for(ledger: Ledger, requirement_id: str) -> list[str]:
    return [g for g in ledger.view.children.get(requirement_id, [])
            if ledger.view.by_id[g].rtype == "Gap" and ledger.view.status.get(g) in M.OPEN_STATES]


def _advance_to_tested(ledger: Ledger, subject: str, testrun_id: str, reason: str) -> int:
    cur = ledger.view.status[subject]
    if cur in M.DONE_STATES:
        return 0
    path = M.reachable(cur, M.Status.TESTED)
    if not path:
        return 0
    steps = [s for s in path if s is not cur]
    for s in steps:
        ledger.transition(subject, s, reason, M.Role.KERNEL,
                          justification_ids=(testrun_id,) if s in M.DONE_STATES else ())
    return len(steps)


def save_snapshot(model: SM.SelfModel, root: Path) -> Path:
    d = root / SNAPSHOT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"selfmodel_{model.digest()}.json"
    if not path.exists():
        SM.save(model, path)
    return path


def dependency_order(ledger: Ledger, rows: Sequence[Assessment]) -> list[Assessment]:
    """Requirements before the requirements that depend on them (1 Oct: key order created K02.depth's gap before K02.exists's,
    so the depth gap recorded no blocker and was planned first). Ties by key, so the order is deterministic."""
    by_id = {r.requirement_id: r for r in rows}
    deps = {r.requirement_id: [d for d in getattr(ledger.get(r.requirement_id), "depends_on") if d in by_id] for r in rows}
    out: list[Assessment] = []
    done: set[str] = set()
    visiting: set[str] = set()

    def visit(rid: str) -> None:
        if rid in done:
            return
        if rid in visiting:
            raise ValueError(f"requirement dependency cycle through {by_id[rid].key}")
        visiting.add(rid)
        for d in sorted(deps[rid], key=lambda x: by_id[x].key):
            visit(d)
        visiting.discard(rid)
        done.add(rid)
        out.append(by_id[rid])

    for r in sorted(rows, key=lambda x: x.key):
        visit(r.requirement_id)
    return out


@dataclasses.dataclass(frozen=True)
class SyncReport:
    assessed: int
    met: int
    opened: tuple[str, ...]
    closed: tuple[str, ...]
    regressed: tuple[str, ...]
    snapshot: str
    stale: tuple[str, ...] = ()


def sync(ledger: Ledger, model: SM.SelfModel, now: Optional[float] = None) -> SyncReport:
    root = Path(ledger.evidence_root)
    snap = save_snapshot(model, root)
    ev = M.EvidenceRef.of(snap, root, "selfmodel")
    rows = assess(ledger, model)
    opened: list[str] = []
    closed: list[str] = []
    regressed: list[str] = []
    stale: list[str] = []
    gap_of: dict[str, list[str]] = {r.requirement_id: open_gaps_for(ledger, r.requirement_id) for r in rows}
    for r in dependency_order(ledger, rows):
        if r.met:
            if r.status in M.DONE_STATES and not gap_of[r.requirement_id]:
                continue
            tr = ledger.append(M.TestRun(created_by=M.Role.KERNEL, command=f"check:{r.step}:{r.component}", passed=1, failed=0,
                                         errors=0, skipped=0, duration_s=0.0, subject_ids=(r.requirement_id,),
                                         selection=(r.key,), evidence=(ev,)))
            _advance_to_tested(ledger, r.requirement_id, tr, f"computed check passed: {r.detail}")
            for g in gap_of[r.requirement_id]:
                _advance_to_tested(ledger, g, tr, f"requirement {r.key} now met: {r.detail}")
                closed.append(g)
            continue
        if r.status in M.DONE_STATES and r.stale:
            ledger.transition(r.requirement_id, M.Status.IN_PROGRESS, f"evidence stale, needs a re-test: {r.detail}",
                              M.Role.KERNEL)                            # unknown is not failed - but it is no longer TESTED
            stale.append(r.key)
        elif r.status in M.DONE_STATES:
            ledger.transition(r.requirement_id, M.Status.FAILED, f"regression: check:{r.step}:{r.component} now fails - {r.detail}",
                              M.Role.KERNEL)
            regressed.append(r.key)
        if not gap_of[r.requirement_id]:
            req = ledger.get(r.requirement_id)
            dep_gaps = tuple(g for d in getattr(req, "depends_on") for g in open_gaps_for(ledger, d))
            gid = ledger.append(M.Gap(created_by=M.Role.KERNEL, parents=(r.requirement_id,), kind=r.kind,
                                      description=f"{r.key} unmet: {r.detail}", importance=r.importance,
                                      depends_on=dep_gaps, evidence=(ev,)))
            gap_of[r.requirement_id] = [gid]
            opened.append(gid)
    return SyncReport(len(rows), sum(r.met for r in rows), tuple(opened), tuple(closed), tuple(regressed), ev.path, tuple(stale))


@dataclasses.dataclass(frozen=True)
class RankedGap:
    gap_id: str
    requirement_key: str
    kind: str
    importance: float
    blocked_by: tuple[str, ...]
    description: str


def ranked(ledger: Ledger) -> list[RankedGap]:
    """Open gaps, unblocked first, then by importance (desc), then requirement key (a deterministic, name-independent tiebreak
    would need content; the key is the declared component order, which IS the build order)."""
    out = []
    for e in ledger.of_type("Gap"):
        if ledger.view.status.get(e.id) not in M.OPEN_STATES:
            continue
        rec = e.record
        recorded = [d for d in getattr(rec, "depends_on") if ledger.view.status.get(d) in M.OPEN_STATES]
        req_ids = [p for p in rec.parents if ledger.view.by_id[p].rtype == "Requirement"]
        live = [g for r in req_ids for d in getattr(ledger.get(r), "depends_on") for g in open_gaps_for(ledger, d)]
        blockers = tuple(dict.fromkeys(recorded + live))      # live too: gaps recorded out of order (pre 1 Oct) missed blockers
        req_key = getattr(ledger.get(req_ids[0]), "key", "") if req_ids else ""
        out.append(RankedGap(e.id, req_key, getattr(rec, "kind").value, getattr(rec, "importance"), blockers,
                             getattr(rec, "description")))
    return sorted(out, key=lambda g: (bool(g.blocked_by), -g.importance, g.requirement_key))


def summary(ledger: Ledger) -> dict[str, Any]:
    rows = O.requirement_view(ledger)
    by: dict[str, int] = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    gaps = ranked(ledger)
    return {"requirements": len(rows), "by_status": by, "open_gaps": len(gaps),
            "unblocked": sum(1 for g in gaps if not g.blocked_by),
            "next": [dataclasses.asdict(g) for g in gaps[:5]], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
