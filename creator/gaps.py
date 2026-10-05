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
        if ledger.view.status.get(e.id) is M.Status.REJECTED:
            continue                                                    # superseded (e.g. the depth step, owner ruling 1 Oct)
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


# ------------------------------------------------------------------------------------------------ P1.4 stuck detector + fallback ladder + parking
# MASTER_BLUEPRINT 7.8: "never stuck". A goal with no measured progress for STUCK_CYCLES cycles (or RUNG_MAX_S seconds on one rung) climbs
# the ladder: smaller step -> different approach -> bigger rung -> PARK with its reason (re-tried when a dependency changes) -> a GPU/data
# job is queued when the gap is a skill/data gap -> an owner digest entry. A hard stall limit (STALL_MAX_S) parks the goal whatever rung it
# is on, so no goal stalls an hour. State is one JSON file (state/ladder.json) next to the parking lot; standard library only.
STUCK_CYCLES = 3
RUNG_MAX_S = 800.0                    # 4 phases (start + 3 rungs) of this stay under STALL_MAX_S
STALL_MAX_S = 3600.0
RUNGS = ("start", "smaller_step", "different_approach", "bigger_rung")
LADDER_FILE = "ladder.json"
JOBS_FILE = "jobs_queue.jsonl"        # GPU / data jobs wanted by parked goals (read by the GPU operator and the placement wishlist)
DIGEST_FILE = "digest_queue.jsonl"    # owner digest entries (scripts/nupen_digest.py may list them)
DATA_KINDS = ("skill", "data")        # a parked gap of these kinds also queues a job


def _lad_load(state: Path) -> dict[str, Any]:
    try:
        x = json.loads((Path(state) / LADDER_FILE).read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except (OSError, ValueError):
        return {}


def _lad_save(state: Path, d: Mapping[str, Any]) -> None:
    f = Path(state) / LADDER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(f)


def _lad_append(state: Path, name: str, row: Mapping[str, Any]) -> None:
    f = Path(state) / name
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def ladder_note(state: Path, goal: str, value: Optional[float], now: float, *, kind: str = "", deps: Optional[Mapping[str, str]] = None,
                eps: float = 1e-9) -> dict[str, Any]:
    """One finished cycle of `goal`: `value` is its measured progress metric (higher = better; None = no measurement), `deps` the
    fingerprints of what a parked goal waits on. Returns {'action': continue | smaller_step | different_approach | bigger_rung | park |
    parked, 'rung', 'reason'}. Progress = value above the best so far; it resets the rung counters (the ladder is for the stuck only)."""
    st = _lad_load(state)
    g = st.setdefault(goal, {"rung": 0, "rung_since": now, "no_progress": 0, "best": None, "progress_at": now, "status": "active"})
    if g["status"] == "parked":
        return {"action": "parked", "rung": g["rung"], "reason": g.get("park", {}).get("reason", "")}
    if value is not None and (g["best"] is None or value > g["best"] + eps):
        g.update(best=value, no_progress=0, rung=0, rung_since=now, progress_at=now)
        _lad_save(state, st)
        return {"action": "continue", "rung": 0, "reason": "progress"}
    g["no_progress"] += 1
    stalled = now - g["progress_at"] >= STALL_MAX_S
    stuck = g["no_progress"] >= STUCK_CYCLES or now - g["rung_since"] >= RUNG_MAX_S
    if not stuck and not stalled:
        _lad_save(state, st)
        return {"action": "continue", "rung": g["rung"], "reason": f"{g['no_progress']}/{STUCK_CYCLES} cycles without progress"}
    if g["rung"] < len(RUNGS) - 1 and not stalled:
        g.update(rung=g["rung"] + 1, rung_since=now, no_progress=0)
        _lad_save(state, st)
        return {"action": RUNGS[g["rung"]], "rung": g["rung"], "reason": f"no measured progress; climbing to {RUNGS[g['rung']]}"}
    why = (f"no measured progress for {(now - g['progress_at']) / 60:.0f} min and the ladder ({', '.join(RUNGS[1:])}) "
           f"{'was cut short by the stall limit' if stalled and g['rung'] < len(RUNGS) - 1 else 'is used up'}")
    g.update(status="parked", park={"reason": why, "at": now, "deps": dict(deps or {}), "kind": kind})
    _lad_save(state, st)
    jobs = None
    if kind in DATA_KINDS:
        jobs = {"at": now, "goal": goal, "job": "gpu_module" if kind == "skill" else "data_job", "why": why}
        _lad_append(state, JOBS_FILE, jobs)
    _lad_append(state, DIGEST_FILE, {"at": now, "goal": goal, "kind": "parked", "reason": why, "retry": "when a dependency changes",
                                     "job_queued": bool(jobs)})
    return {"action": "park", "rung": g["rung"], "reason": why, "job": jobs}


def ladder_due(state: Path, fingerprints: Mapping[str, str], now: float, retry_after_s: float = 86400.0) -> list[str]:
    """Parked goals to re-try now: a dependency fingerprint changed (or, with no dependency recorded, `retry_after_s` passed). Each is
    unparked with fresh counters (rung 0)."""
    st = _lad_load(state)
    due: list[str] = []
    for goal, g in st.items():
        if g.get("status") != "parked":
            continue
        p = g.get("park", {})
        deps = p.get("deps") or {}
        changed = [k for k, v in deps.items() if fingerprints.get(k, v) != v]
        if changed or (not deps and now - float(p.get("at", now)) >= retry_after_s):
            g.update(status="active", rung=0, rung_since=now, no_progress=0, progress_at=now, retried=changed or ["timer"])
            due.append(goal)
    if due:
        _lad_save(state, st)
        for goal in due:
            _lad_append(state, DIGEST_FILE, {"at": now, "goal": goal, "kind": "retried", "reason": f"dependency changed: {st[goal]['retried']}"})
    return due


def ladder_rung(state: Path, goal: str) -> str:
    return RUNGS[int(_lad_load(state).get(goal, {}).get("rung", 0))]
