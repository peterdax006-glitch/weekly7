"""Creator goal generation - the system PROPOSES new capabilities from evidence; only an approval makes them work.

Until now the Creator closed gaps in the capabilities somebody declared (creator/capabilities.json) and could never decide "I should
build something new". `propose()` computes, with no model call, candidate capabilities from five kinds of evidence:

    failure     a recurring Failure classification no declared capability addresses            (ledger Failure records)
    weakness    a development-process weakness whose recursion remedies are exhausted          (recursion.find_weaknesses)
    section     a master-prompt section whose checklist items are open and map to no code      (CREATOR_MASTER_CHECKLIST.json)
    student     a task kind every student fails (and only students are counted)                (curriculum.student_scores)
    knowledge   a recurring unexplained failure (UNKNOWN / undetermined root cause)            (same predicate as oversight)

Each Proposal carries the evidence ids/counts, a proposed capability spec (name, modules, tests, requirement ladder), a computed value
and cost, and is refused if it duplicates a declared/approved capability or a proposal already made (a rejected one is never made
again). Proposals live in state/creator/goal_proposals.jsonl (append-only events). A proposal is NOT work: `approve` (owner, or the
teacher = Role.BUILDER) records an Objective in the ledger, appends the capability to creator/capabilities_approved.json (which
selfmodel.load_capabilities also reads) and compiles its requirement ladder exactly like a declared capability, so the planner and
kernel need no change. `reject` is permanent."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from creator import curriculum as CUR
from creator import model as M
from creator import objective as O
from creator import recursion as R
from creator import selfmodel as SM
from creator.ledger import Ledger

PROPOSALS_FILE = "goal_proposals.jsonl"
RUN_FILE = "goal_proposals_run.json"
CHECKLIST_REL = "state/build/CREATOR_MASTER_CHECKLIST.json"
MIN_N = 3                                     # a recurring class needs at least this many records
PER_SOURCE = 3                                # at most this many NEW proposals per source per run
SOURCES = ("failure", "weakness", "section", "student", "knowledge")
OPEN_STATES = ("NOT_STARTED", "IN_PROGRESS")
STOP_WORDS = frozenset({"the", "and", "for", "of", "to", "a", "in", "is", "not", "do", "it", "that", "with", "every", "section",
                        "requirement", "this", "final", "rule"})


class GoalError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Proposal:
    id: str
    key: str                                  # source:subject - the duplicate key
    source: str
    title: str
    rationale: str
    evidence: tuple[str, ...]                 # ledger record ids, checklist ids or lesson task kinds
    spec: dict[str, Any]                      # name, modules, tests, floor, ladder
    value: float                              # 0..1 from the evidence share and count
    cost: dict[str, Any]                      # packages and lines estimate
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"event": "proposal", **dataclasses.asdict(self)}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP_WORDS}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "goal"


def _pid(key: str) -> str:
    return "GP-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _make(source: str, subject: str, title: str, rationale: str, evidence: Iterable[str], count: int, share: float,
          now: str) -> Proposal:
    slug = f"{source}_{_slug(subject)}"
    spec = {"name": title, "modules": [f"creator/goal_{slug}.py"], "tests": [f"tests/test_creator_goal_{slug}.py"], "floor": 0,
            "ladder": [step for step, *_ in O.LADDER]}
    value = round(min(1.0, share * min(1.0, count / 10.0) + (0.1 if source in ("student", "weakness") else 0.0)), 3)
    cost = {"packages": len(O.LADDER), "modules": 1, "tests": 1, "lines_estimate": 250}
    key = f"{source}:{subject}"
    return Proposal(_pid(key), key, source, title, rationale, tuple(evidence)[:25], spec, value, cost, now)


# ------------------------------------------------------------------------------------------------ the five sources

def _covers(specs: Sequence[SM.CapabilitySpec], words: set[str]) -> bool:
    """True when some declared capability already speaks of every significant word of the subject."""
    return bool(words) and any(words <= _words(" ".join((s.name, *s.modules))) for s in specs)


def from_failures(led: Ledger, specs: Sequence[SM.CapabilitySpec], now: str) -> list[Proposal]:
    by: dict[str, list[str]] = {}
    for e in led.of_type("Failure"):
        by.setdefault(getattr(e.record, "classification"), []).append(e.id)
    total = sum(len(v) for v in by.values())
    out = []
    for cls, ids in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(ids) < MIN_N or cls.upper() == "UNKNOWN" or _covers(specs, _words(cls.replace("_", " "))):
            continue
        out.append(_make("failure", cls, f"handle recurring failure class {cls}",
                         f"{len(ids)} of {total} Failure records are classified {cls}; no declared capability mentions it", ids,
                         len(ids), len(ids) / total, now))
    return out


def from_weaknesses(led: Ledger, specs: Sequence[SM.CapabilitySpec], now: str) -> list[Proposal]:
    out = []
    process, tried = R.current_process(led), R.tried_changes(led)
    for w in R.find_weaknesses(led, MIN_N):
        if R.design_change(w, process, tried) is not None:
            continue                                                    # a remedy is still open: recursion owns it
        out.append(_make("weakness", w.key, f"new remedy for {w.kind} weakness {w.key}",
                         f"{w.count} cases ({w.share:.0%}) of {w.kind} {w.key!r}; every process-parameter remedy is tried or at its bound",
                         w.evidence_ids, w.count, w.share, now))
    return out


def from_sections(checklist: Path, specs: Sequence[SM.CapabilitySpec], now: str) -> list[Proposal]:
    try:
        items = json.loads(checklist.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return []
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(str(it.get("group", "")), []).append(it)
    out = []
    for group, its in sorted(groups.items()):
        open_ = [i for i in its if i.get("status") in OPEN_STATES and not i.get("code_paths")]
        if not open_:
            continue
        subject = group.split("/", 1)[-1].strip()
        if _covers(specs, _words(subject)):
            continue
        out.append(_make("section", subject, f"cover master-prompt section: {subject[:60]}",
                         f"{len(open_)} of {len(its)} checklist items of {group!r} are open with no component mapped",
                         [str(i.get("id")) for i in open_], len(open_), len(open_) / len(its), now))
    out.sort(key=lambda p: (-p.value, p.key))
    return out


def from_students(lessons: Sequence[CUR.Lesson], now: str) -> list[Proposal]:
    cells = CUR.student_scores(lessons)["cells"].values()
    kinds: dict[str, list[int]] = {}
    for c in cells:
        if c["solver"] not in CUR.TEACHER:
            k = kinds.setdefault(c["task_kind"], [0, 0])
            k[0] += c["attempts"]
            k[1] += c["adopted"]
    out = []
    for kind, (att, ok) in sorted(kinds.items()):
        if att >= MIN_N and ok == 0:
            ids = [les.lesson_id for les in lessons if les.task_kind == kind and les.solver not in CUR.TEACHER][:25]
            out.append(_make("student", kind, f"student capability for {kind} tasks",
                             f"every student failed {kind} tasks: {att} attempts, 0 adopted", ids, att, 1.0, now))
    return out


def from_knowledge(led: Ledger, now: str) -> list[Proposal]:
    by: dict[str, list[str]] = {}
    for e in led.of_type("Diagnosis"):
        fail = led.get(getattr(e.record, "failure_id"))
        cls = getattr(fail, "classification", "UNKNOWN")
        if str(getattr(e.record, "root_cause", "")).startswith("undetermined") or cls == "UNKNOWN":
            by.setdefault(cls, []).append(e.id)
    out = []
    for cls, ids in sorted(by.items()):
        if len(ids) >= MIN_N:
            out.append(_make("knowledge", cls, f"root-cause knowledge for unexplained {cls} failures",
                             f"{len(ids)} diagnoses end undetermined for failure class {cls}", ids, len(ids), 1.0, now))
    return out


# ------------------------------------------------------------------------------------------------ storage and status

def _log(state: Path) -> Path:
    return Path(state) / PROPOSALS_FILE


def _events(state: Path) -> list[dict[str, Any]]:
    out = []
    try:
        lines = _log(state).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _append(state: Path, rec: dict[str, Any]) -> None:
    Path(state).mkdir(parents=True, exist_ok=True)
    with open(_log(state), "ab") as f:
        f.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))


def listing(state: Path) -> list[dict[str, Any]]:
    """Every proposal with its status (PENDING / APPROVED / REJECTED), latest decision wins except that REJECTED is final."""
    props: dict[str, dict[str, Any]] = {}
    for ev in _events(state):
        pid = str(ev.get("id", ""))
        if ev.get("event") == "proposal" and pid not in props:
            props[pid] = {**ev, "status": "PENDING"}
        elif ev.get("event") in ("approve", "reject") and pid in props and props[pid]["status"] == "PENDING":
            props[pid].update(status="APPROVED" if ev["event"] == "approve" else "REJECTED",
                              decided_by=ev.get("by", ""), decided_at=ev.get("at", ""), reason=ev.get("reason", ""),
                              capability_id=ev.get("capability_id", ""))
    return list(props.values())


def pending(state: Path) -> list[dict[str, Any]]:
    return [p for p in listing(state) if p["status"] == "PENDING"]


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def propose(led: Ledger, state: Path, repo: Path, specs: Optional[Sequence[SM.CapabilitySpec]] = None,
            now: Optional[str] = None) -> list[Proposal]:
    """Compute proposals from all five sources, drop duplicates, append the new ones to the log. Read-only on the ledger."""
    now = now or _now()
    specs = list(specs) if specs is not None else SM.load_capabilities()
    known = {p["key"] for p in listing(state)}
    modules = {m for s in specs for m in s.modules} | {m for p in listing(state) for m in p["spec"]["modules"]}
    lessons = CUR.LessonLog(Path(state) / "lessons.jsonl").lessons()
    by_source = {"failure": from_failures(led, specs, now), "weakness": from_weaknesses(led, specs, now),
                 "section": from_sections(Path(repo) / CHECKLIST_REL, specs, now), "student": from_students(lessons, now),
                 "knowledge": from_knowledge(led, now)}
    new: list[Proposal] = []
    for source in SOURCES:
        taken = 0
        for p in by_source[source]:
            if p.key in known or p.spec["modules"][0] in modules or taken >= PER_SOURCE:
                continue
            _append(state, p.to_dict())
            known.add(p.key)
            modules.add(p.spec["modules"][0])
            new.append(p)
            taken += 1
    return new


def maybe_propose(led: Ledger, state: Path, repo: Path, now: Optional[dt.datetime] = None, every_s: float = 86400.0) -> list[Proposal]:
    """At most once per `every_s` (the swarm calls this at every round start; it is cheap and never raises)."""
    now = now or dt.datetime.now()
    run = Path(state) / RUN_FILE
    try:
        last = dt.datetime.fromisoformat(json.loads(run.read_text(encoding="utf-8"))["at"])
        if (now - last).total_seconds() < every_s:
            return []
    except (OSError, ValueError, KeyError):
        pass
    try:
        new = propose(led, state, repo, now=now.isoformat(timespec="seconds"))
        Path(state).mkdir(parents=True, exist_ok=True)
        run.write_text(json.dumps({"at": now.isoformat(timespec="seconds"), "new": len(new)}), encoding="utf-8")
        return new
    except Exception:                                                   # noqa: BLE001 - goal generation never costs a round
        return []


# ------------------------------------------------------------------------------------------------ approval

def approved_path(repo: Path) -> Path:
    return Path(repo) / "creator" / SM.APPROVED_FILE


def _next_capability_id(repo: Path) -> str:
    ids = [int(s.id[1:]) for s in SM.load_capabilities(Path(repo) / "creator" / "capabilities.json")]
    ids += [int(s.id[1:]) for s in SM.load_capabilities(approved_path(repo))]
    nxt = max(ids or [0]) + 1
    if nxt > 99:
        raise GoalError("no capability id left (the check grammar is K00-K99)")
    return f"K{nxt:02d}"


def _find(state: Path, pid: str) -> dict[str, Any]:
    for p in listing(state):
        if p["id"] == pid:
            return p
    raise GoalError(f"no proposal {pid}")


def approve(state: Path, repo: Path, led: Ledger, pid: str, by: str) -> str:
    """Owner or teacher approves: an Objective record goes in the ledger, the capability into capabilities_approved.json and its
    requirement ladder is compiled. Returns the new capability id."""
    roles = {"owner": M.Role.OWNER, "teacher": M.Role.BUILDER}
    if by not in roles:
        raise GoalError("approval is by 'owner' or 'teacher' only")
    p = _find(state, pid)
    if p["status"] != "PENDING":
        raise GoalError(f"{pid} is already {p['status']}")
    cap_id = _next_capability_id(repo)
    spec = p["spec"]
    oid = led.append(M.Objective(
        created_by=roles[by], statement=f"APPROVED GOAL {pid}: {p['title']}",
        outcomes=(p["rationale"],), constraints=("the capability is built and validated like any declared one",),
        acceptance_criteria=(f"capability {cap_id} reaches VALIDATED: " + ", ".join(spec["ladder"]),),
        metrics=(f"evidence: {', '.join(p['evidence'][:5])}",)))
    ap = approved_path(repo)
    data: dict[str, Any] = json.loads(ap.read_text(encoding="utf-8")) if ap.is_file() else {"about": "Capabilities approved by the owner or teacher "
                                                                              "from creator/goals.py proposals.", "capabilities": []}
    data["capabilities"].append({"id": cap_id, "name": spec["name"], "modules": spec["modules"], "tests": spec["tests"],
                                 "floor": spec["floor"], "approved_from": pid})
    ap.write_text(json.dumps(data, indent=1), encoding="utf-8")
    O.compile_capabilities(led, oid, SM.load_capabilities(ap), created_by=M.Role.KERNEL)
    _append(state, {"event": "approve", "id": pid, "by": by, "at": _now(), "capability_id": cap_id, "objective_id": oid})
    return cap_id


def reject(state: Path, pid: str, reason: str) -> None:
    """Permanent: the key stays in the log, so propose() never makes this proposal again."""
    if not reason.strip():
        raise GoalError("a rejection needs a reason")
    p = _find(state, pid)
    if p["status"] != "PENDING":
        raise GoalError(f"{pid} is already {p['status']}")
    _append(state, {"event": "reject", "id": pid, "reason": reason.strip(), "at": _now()})
