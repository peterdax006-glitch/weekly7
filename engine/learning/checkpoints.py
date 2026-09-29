"""Continuous-execution checkpoints and the never-stop engine (contract C62 sections 58, 59, 84; checklist J07, J09).
IMPLEMENTED - NOT VALIDATED.

If work is interrupted the engine must SAVE STATE, WRITE CHECKPOINT, WRITE NEXT ACTION, WRITE CURRENT FAILURES, WRITE CURRENT
EXPERIMENT, WRITE CURRENT CODE HASH and CONTINUE FROM CHECKPOINT. This module makes that mechanical:
  * `ExecutionState`      - the seven items above as one validated, hashable record;
  * `CheckpointStore`     - rolling, atomic, self-verifying state files. A torn or corrupted newest file falls back to the
                            newest valid one (a crash mid-write must never lose the run); `seal_milestone` additionally writes
                            a write-once bundle through engine.checkpoint (Bible Phase 0.3);
  * `resume`              - decides what to do on restart: continue, redo the experiment that was in flight, or revalidate work
                            done under different code (stale-code detection, section 57);
  * `ExecutionLoop`       - runs dependent tasks, checkpoints after every attempt, retries, records failures, and keeps going
                            past a failed task (only a FirewallBreach stops it, because firewalls fail closed);
  * `ChecklistTracker`    - section 84 step 2 ("find the first incomplete critical item") and the rule that no item may be
                            marked VALIDATED without recorded evidence;
  * `evaluate_stop`       - the eight stop conditions of section 58; the engine may stop only when all hold;
  * `scan_false_completion` - section 59: a claim of done/validated/learning works without the honest label is a finding.
Times are explicit (`now`); nothing here reads a clock except through the injectable `clock`."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from engine import checkpoint as bundle

from .core import BuildStatus, FirewallBreach, ValidationLabel, as_date, canonical_json, current_code_hash, stable_hash


SCHEMA_VERSION = 1


class CheckpointCorrupt(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ the state record
@dataclass(frozen=True)
class FailureNote:
    at: str
    task: str
    kind: str                         # ERROR | OOM | CRASH | FIREWALL | STALE_CODE | TIMEOUT
    message: str
    attempt: int = 1
    resolved: bool = False
    escalated: bool = False           # a genuine scientific limitation the owner must see (section 58 condition 7)

    def open(self) -> bool:
        return not (self.resolved or self.escalated)


@dataclass(frozen=True)
class ExecutionState:
    run_id: str
    sequence: int
    saved_at: str
    phase: str
    next_action: str
    current_experiment: str           # experiment key, or "" when none is in flight
    code_hash: str
    failures: tuple[FailureNote, ...] = ()
    completed: Mapping[str, str] = field(default_factory=dict)      # task -> code hash it completed under
    pending: tuple[str, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)      # path -> sha256 at save time
    schema: int = SCHEMA_VERSION

    def validate(self) -> list[str]:
        errs = []
        if not self.run_id or Path(self.run_id).name != self.run_id:
            errs.append("run_id must be a plain identifier")
        if self.sequence < 0:
            errs.append("sequence must be non-negative")
        if not self.code_hash:
            errs.append("code_hash missing: a checkpoint without the code it ran under cannot be resumed safely")
        if not self.next_action:
            errs.append("next_action missing: a checkpoint must say what to do next")
        overlap = set(self.completed) & set(self.pending)
        if overlap:
            errs.append(f"tasks both completed and pending: {sorted(overlap)}")
        return errs

    @property
    def digest(self) -> str:
        return stable_hash(self, 20)

    def open_failures(self) -> tuple[FailureNote, ...]:
        return tuple(f for f in self.failures if f.open())

    def to_dict(self) -> dict:
        return json.loads(canonical_json(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExecutionState":
        """Tolerant reader: unknown keys (written by newer code) are ignored and missing ones (older code) take their defaults, so
        a run can resume across a code change. `schema` is kept so `audit_store` can report states from a newer schema."""
        names = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        fnames = {f.name for f in dataclasses.fields(FailureNote)}
        d["failures"] = tuple(FailureNote(**{k: v for k, v in f.items() if k in fnames}) for f in d.get("failures", ()))
        d["pending"] = tuple(d.get("pending", ()))
        return cls(**d)


# ------------------------------------------------------------------------------------------------ the store
def _atomic_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


class CheckpointStore:
    """root/<run_id>/cp_<seq>.json  + LATEST.json. Each file embeds the sha256 of its own body, so a truncated write, a hand
    edit or bit rot is detected on read. `save` never overwrites an earlier sequence number."""

    def __init__(self, root: str | Path, run_id: str, keep: int = 25):
        if not run_id or Path(run_id).name != run_id:
            raise ValueError(f"unusable run id {run_id!r}")
        if keep < 3:
            raise ValueError("keep at least 3 checkpoints so a torn newest file always has a valid predecessor")
        self.dir = Path(root) / run_id
        self.run_id, self.keep = run_id, keep

    def _path(self, seq: int) -> Path:
        return self.dir / f"cp_{seq:06d}.json"

    def sequences(self) -> list[int]:
        return sorted(int(p.stem[3:]) for p in self.dir.glob("cp_*.json")) if self.dir.is_dir() else []

    def save(self, now, phase: str, next_action: str, code_hash: str, current_experiment: str = "",
             failures: Iterable[FailureNote] = (), completed: Mapping[str, str] | None = None, pending: Sequence[str] = (),
             notes: Mapping[str, str] | None = None, artifacts: Iterable[str | Path] = ()) -> ExecutionState:
        seqs = self.sequences()
        seq = (seqs[-1] + 1) if seqs else 0
        arts = {}
        for a in artifacts:
            p = Path(a)
            arts[str(p)] = bundle.sha256_file(p) if p.is_file() else "MISSING"
        st = ExecutionState(self.run_id, seq, str(now), phase, next_action, current_experiment, code_hash, tuple(failures),
                            dict(completed or {}), tuple(pending), dict(notes or {}), arts)
        errs = st.validate()
        if errs:
            raise CheckpointCorrupt(f"refusing to write an invalid checkpoint: {errs}")
        body = json.loads(canonical_json(st))
        doc = json.dumps({"sha256": bundle._body_hash(body), "body": body}, sort_keys=True, allow_nan=False)
        if self._path(seq).exists():
            raise CheckpointCorrupt(f"checkpoint {seq} already exists")
        _atomic_text(self._path(seq), doc)
        _atomic_text(self.dir / "LATEST.json", json.dumps({"sequence": seq, "digest": st.digest}))
        self._prune()
        return st

    def read(self, seq: int) -> ExecutionState:
        p = self._path(seq)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            body = doc["body"]
        except (OSError, ValueError, KeyError) as e:
            raise CheckpointCorrupt(f"cp {seq} unreadable: {type(e).__name__}") from e
        if bundle._body_hash(body) != doc.get("sha256"):
            raise CheckpointCorrupt(f"cp {seq} content hash mismatch")
        return ExecutionState.from_dict(body)

    def latest_valid(self) -> tuple[ExecutionState | None, list[int]]:
        """The newest checkpoint that verifies, plus the newer sequence numbers that were skipped as corrupt."""
        skipped: list[int] = []
        for seq in reversed(self.sequences()):
            try:
                return self.read(seq), skipped
            except CheckpointCorrupt:
                skipped.append(seq)
        return None, skipped

    def _prune(self) -> None:
        seqs = self.sequences()
        for s in seqs[:-self.keep]:
            with_suppress = self._path(s)
            try:
                with_suppress.unlink()
            except OSError:
                pass

    def age_s(self, now_s: float) -> float | None:
        """Seconds since the newest checkpoint file was written (for stall detection); None when there is none."""
        seqs = self.sequences()
        return None if not seqs else max(0.0, now_s - self._path(seqs[-1]).stat().st_mtime)

    def seal_milestone(self, label: str, state: ExecutionState, metrics: Mapping[str, Any], seed: int, now,
                       root: str | Path | None = None) -> Path:
        """Write-once hashed bundle (engine.checkpoint) for a major result. Refuses to overwrite: a milestone is history."""
        r = Path(root) if root else self.dir / "milestones"
        return bundle.write_checkpoint(r, f"{self.run_id}_{label}", state.to_dict(), dict(metrics), {"seed": int(seed)},
                                       {"label": label, "sequence": state.sequence, "digest": state.digest}, str(now))


# ------------------------------------------------------------------------------------------------ resume
@dataclass(frozen=True)
class ResumePlan:
    action: str                       # START_FRESH | CONTINUE | REDO_EXPERIMENT | RECOVER_EXPERIMENT | RECONCILE_EXPERIMENT
    state: ExecutionState | None
    revalidate: tuple[str, ...]       # completed tasks whose completion code differs from the code now loaded
    skipped_corrupt: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def code_changed(self) -> bool:
        return bool(self.revalidate)


def resume(store: CheckpointStore, code_hash: str, experiment_state: Callable[[str], str | None] | None = None) -> ResumePlan:
    """Decide how to continue after an interruption. `experiment_state(key)` returns the compute-ledger state of the
    experiment that was in flight (PENDING/RUNNING/DONE/...), or None if unknown. Work is never assumed valid merely
    because a checkpoint says it was done: anything completed under other code is queued for revalidation."""
    st, skipped = store.latest_valid()
    reasons = []
    if skipped:
        reasons.append(f"newest checkpoint(s) {skipped} were corrupt; fell back")
    if st is None:
        return ResumePlan("START_FRESH", None, (), tuple(skipped), tuple(reasons + ["no valid checkpoint"]))
    stale = tuple(sorted(t for t, h in st.completed.items() if h != code_hash))
    if stale:
        reasons.append(f"{len(stale)} completed task(s) were done under other code and must be revalidated")
    action = "CONTINUE"
    if st.current_experiment:
        es = experiment_state(st.current_experiment) if experiment_state else None
        if st.code_hash != code_hash:
            action = "REDO_EXPERIMENT"
            reasons.append(f"experiment {st.current_experiment} started under code {st.code_hash}; engine is now {code_hash}")
        elif es == "DONE":
            action = "RECONCILE_EXPERIMENT"
            reasons.append(f"experiment {st.current_experiment} finished while we were down; reconcile before continuing")
        elif es != "DONE":
            action = "RECOVER_EXPERIMENT"
            reasons.append(f"experiment {st.current_experiment} was in flight (ledger says {es}); recover it")
    return ResumePlan(action, st, stale, tuple(skipped), tuple(reasons))


# ------------------------------------------------------------------------------------------------ the loop
@dataclass(frozen=True)
class Task:
    name: str
    fn: Callable[[], Any]
    depends: tuple[str, ...] = ()
    retries: int = 2
    experiment: str = ""              # compute-ledger key if the task launches an experiment

    def validate(self) -> list[str]:
        return [] if self.name and self.retries >= 0 else [f"task {self.name!r} invalid"]


@dataclass(frozen=True)
class LoopResult:
    done: tuple[str, ...]
    failed: tuple[str, ...]
    blocked: tuple[str, ...]          # never ran because a dependency failed
    skipped: tuple[str, ...]          # already complete under the current code
    state: ExecutionState | None


def topological_order(tasks: Sequence[Task]) -> list[Task]:
    """Deterministic dependency order (ties by name). Cycles and unknown dependencies are errors, not silent drops."""
    by = {t.name: t for t in tasks}
    if len(by) != len(tasks):
        raise ValueError("duplicate task names")
    for t in tasks:
        missing = [d for d in t.depends if d not in by]
        if missing:
            raise ValueError(f"task {t.name} depends on unknown {missing}")
    out: list[Task] = []
    state: dict[str, int] = {}

    def visit(n: str, stack: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ValueError(f"dependency cycle: {' -> '.join(stack + (n,))}")
        state[n] = 1
        for d in sorted(by[n].depends):
            visit(d, stack + (n,))
        state[n] = 2
        out.append(by[n])

    for n in sorted(by):
        visit(n, ())
    return out


class ExecutionLoop:
    """Checkpoint-driven task runner. It never stops because one task failed - it records the failure, blocks only the
    dependents, and moves on (section 58) - and it stops immediately, after saving state, for a FirewallBreach."""

    def __init__(self, store: CheckpointStore, tasks: Sequence[Task], code_hash: str | None = None,
                 clock: Callable[[], float] = time.time, phase: str = "run"):
        for t in tasks:
            errs = t.validate()
            if errs:
                raise ValueError(errs)
        self.store, self.order = store, topological_order(tasks)
        self.code_hash = code_hash if code_hash is not None else current_code_hash()
        self.clock, self.phase = clock, phase

    def _save(self, completed, failures, pending, next_action, current="", notes=None) -> ExecutionState:
        return self.store.save(self.clock(), self.phase, next_action, self.code_hash, current, failures, completed, pending, notes)

    def run(self, experiment_state: Callable[[str], str | None] | None = None) -> LoopResult:
        plan = resume(self.store, self.code_hash, experiment_state)
        prior = plan.state
        completed = {t: h for t, h in (prior.completed if prior else {}).items() if h == self.code_hash}
        failures = list(prior.failures) if prior else []
        done, failed, blocked, skipped = [], [], [], []
        failed_names: set[str] = set()
        names = [t.name for t in self.order]
        for task in self.order:
            if task.name in completed:
                skipped.append(task.name)
                continue
            if any(d in failed_names or d in blocked for d in task.depends):
                blocked.append(task.name)
                continue
            pending = [n for n in names if n not in completed and n != task.name]
            ok = False
            for attempt in range(1, task.retries + 2):
                self._save(completed, failures, pending, f"run task {task.name} (attempt {attempt})", task.experiment)
                try:
                    task.fn()
                    ok = True
                    break
                except FirewallBreach as e:
                    failures.append(FailureNote(str(self.clock()), task.name, "FIREWALL", str(e)[:300], attempt, escalated=True))
                    self._save(completed, failures, pending, f"INVESTIGATE firewall breach in {task.name}", task.experiment)
                    raise
                except MemoryError as e:
                    failures.append(FailureNote(str(self.clock()), task.name, "OOM", str(e)[:300], attempt))
                except Exception as e:
                    failures.append(FailureNote(str(self.clock()), task.name, "ERROR",
                                                f"{type(e).__name__}: {e}"[:300], attempt))
            if ok:
                completed[task.name] = self.code_hash
                done.append(task.name)
                failures = [dataclasses.replace(f, resolved=True) if f.task == task.name else f for f in failures]
            else:
                failed.append(task.name)
                failed_names.add(task.name)
        remaining = [n for n in names if n not in completed]
        nxt = f"next: {remaining[0]}" if remaining else "all tasks complete for this code hash; run validation"
        final = self._save(completed, failures, remaining, nxt)
        return LoopResult(tuple(done), tuple(failed), tuple(blocked), tuple(skipped), final)


# ------------------------------------------------------------------------------------------------ checklist tracking
@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    title: str
    critical: bool = True
    status: BuildStatus = BuildStatus.NOT_STARTED
    evidence: tuple[str, ...] = ()
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED


class ChecklistTracker:
    """Persisted master-checklist state (sections 69-80). The one hard rule: VALIDATED needs recorded evidence, and nothing
    may be marked VALIDATED by the same call that registered it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.items: dict[str, ChecklistItem] = {
            k: ChecklistItem(v["item_id"], v["title"], v["critical"], BuildStatus(v["status"]), tuple(v["evidence"]),
                             ValidationLabel(v["label"])) for k, v in raw.items()}

    def _save(self) -> None:
        _atomic_text(self.path, json.dumps({k: json.loads(canonical_json(v)) for k, v in sorted(self.items.items())},
                                           indent=1, sort_keys=True))

    def add(self, item_id: str, title: str, critical: bool = True) -> None:
        if item_id in self.items:
            raise ValueError(f"{item_id} already on the checklist")
        self.items[item_id] = ChecklistItem(item_id, title, critical)
        self._save()

    def mark(self, item_id: str, status: BuildStatus, evidence: Sequence[str] = (), label: ValidationLabel | None = None) -> ChecklistItem:
        it = self.items[item_id]
        status = BuildStatus.parse(status)
        if status == BuildStatus.VALIDATED:
            if not evidence:
                raise ValueError(f"{item_id}: VALIDATED needs evidence")
            label = ValidationLabel.VALIDATED
        elif status == BuildStatus.IMPLEMENTED:
            label = label or ValidationLabel.NOT_VALIDATED
            if label == ValidationLabel.VALIDATED:
                raise ValueError(f"{item_id}: IMPLEMENTED may not carry the VALIDATED label")
        elif status == BuildStatus.FAILED:
            label = ValidationLabel.FAILED_VALIDATION
        new = dataclasses.replace(it, status=status, evidence=tuple(it.evidence) + tuple(evidence), label=label or it.label)
        self.items[item_id] = new
        self._save()
        return new

    def next_incomplete(self, critical_only: bool = True) -> ChecklistItem | None:
        """Section 84 step 2: the first (by id) item that is not VALIDATED."""
        for k in sorted(self.items):
            it = self.items[k]
            if it.status != BuildStatus.VALIDATED and (it.critical or not critical_only):
                return it
        return None

    def falsely_complete(self) -> list[str]:
        """Items claiming VALIDATED with no evidence, or carrying a label that contradicts their status."""
        bad = []
        for k, it in sorted(self.items.items()):
            if it.status == BuildStatus.VALIDATED and (not it.evidence or it.label != ValidationLabel.VALIDATED):
                bad.append(k)
            if it.status != BuildStatus.VALIDATED and it.label == ValidationLabel.VALIDATED:
                bad.append(k)
        return bad

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.items.values():
            out[it.status.value] = out.get(it.status.value, 0) + 1
        return out


# ------------------------------------------------------------------------------------------------ stop conditions
@dataclass(frozen=True)
class CompletionEvidence:
    sections_complete: Mapping[str, bool] = field(default_factory=dict)
    line_depth: Mapping[str, tuple[int, int]] = field(default_factory=dict)     # section -> (meaningful lines, minimum)
    tests_exist: Mapping[str, bool] = field(default_factory=dict)
    firewalls_exist: Mapping[str, bool] = field(default_factory=dict)
    reports_exist: Mapping[str, bool] = field(default_factory=dict)
    checklist_complete: bool = False
    open_failures: int = 0
    falsely_marked: tuple[str, ...] = ()


@dataclass(frozen=True)
class StopVerdict:
    can_stop: bool
    unmet: tuple[str, ...]


def evaluate_stop(ev: CompletionEvidence) -> StopVerdict:
    """The eight conditions of section 58, all required. An empty evidence record is NOT complete: 'nothing listed' must
    never read as 'nothing missing'."""
    unmet = []
    if not ev.sections_complete or not all(ev.sections_complete.values()):
        unmet.append("1. every required implementation section complete: " +
                     (str(sorted(k for k, v in ev.sections_complete.items() if not v)) if ev.sections_complete else "none listed"))
    short = sorted(k for k, (have, need) in ev.line_depth.items() if have < need)
    if not ev.line_depth or short:
        unmet.append(f"2. line-depth minimums: {short if ev.line_depth else 'none listed'}")
    for n, label, m in ((3, "required tests", ev.tests_exist), (4, "required firewalls", ev.firewalls_exist),
                        (5, "required reports", ev.reports_exist)):
        missing = sorted(k for k, v in m.items() if not v)
        if not m or missing:
            unmet.append(f"{n}. {label}: {missing if m else 'none listed'}")
    if not ev.checklist_complete:
        unmet.append("6. master checklist not complete")
    if ev.open_failures:
        unmet.append(f"7. {ev.open_failures} unresolved failure(s) neither fixed nor escalated")
    if ev.falsely_marked:
        unmet.append(f"8. items falsely marked complete: {list(ev.falsely_marked)}")
    return StopVerdict(not unmet, tuple(unmet))


CLAIM = re.compile(r"\b(done|complete[d]?|finished|production[- ]ready|validated|learning works)\b", re.I)
NEGATED = re.compile(r"\b(not|no|never|un|in|isn't|is not|hasn't|cannot|without|until|unless|before|pending|awaiting)\b"
                     r"[^.\n]{0,25}\b(done|complete[d]?|finished|production[- ]ready|validated|learning works)\b|"
                     r"\b(incomplete|not validated|unvalidated|not yet)\b", re.I)
HONEST = tuple(v.value for v in ValidationLabel if v != ValidationLabel.VALIDATED)


def scan_false_completion(text: str, evidence_ids: Iterable[str] = ()) -> list[tuple[int, str]]:
    """(line number, line) for every line that claims completion/validation with neither an honest label on the line nor a
    cited evidence id. Negated statements ('not validated', 'incomplete') are fine. Section 59."""
    ev = tuple(evidence_ids)
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not CLAIM.search(line) or NEGATED.search(line) or any(h in line for h in HONEST):
            continue
        if ev and any(e in line for e in ev):
            continue
        out.append((i, line.strip()))
    return out


# ------------------------------------------------------------------------------------------------ handoff
def render_handoff(state: ExecutionState, plan: ResumePlan | None = None) -> str:
    """The section-58 interrupt record as plain text (for the Masterstock 'NEXT SESSION: START HERE' block)."""
    lines = ["SAVE STATE          run " + state.run_id + f", checkpoint #{state.sequence}, saved {state.saved_at}, digest {state.digest}",
             f"WRITE CHECKPOINT    phase '{state.phase}', {len(state.completed)} tasks complete, {len(state.pending)} pending",
             f"WRITE NEXT ACTION   {state.next_action}",
             "WRITE CURRENT FAILURES"]
    open_f = state.open_failures()
    lines += [f"    - [{f.kind}] {f.task} (attempt {f.attempt}): {f.message}" for f in open_f] or ["    none open"]
    lines += [f"WRITE CURRENT EXPERIMENT  {state.current_experiment or 'none in flight'}",
              f"WRITE CURRENT CODE HASH   {state.code_hash}"]
    if plan is not None:
        lines.append(f"CONTINUE FROM CHECKPOINT  action={plan.action}; " + "; ".join(plan.reasons))
    return "\n".join(lines) + "\n"


def write_handoff(path: str | Path, state: ExecutionState, plan: ResumePlan | None = None) -> Path:
    p = Path(path)
    _atomic_text(p, render_handoff(state, plan))
    return p


# ------------------------------------------------------------------------------------------------ failures over time
@dataclass(frozen=True)
class EscalationPolicy:
    """When a recurring failure stops being 'retry' and becomes 'a genuine limitation the owner must see' (section 58, cond. 7)."""
    max_same_failure: int = 3            # the same (task, kind) this many times -> escalate
    max_open_per_task: int = 5           # any mix of failures on one task
    always_escalate: tuple[str, ...] = ("FIREWALL",)

    def validate(self) -> list[str]:
        return [] if self.max_same_failure >= 2 and self.max_open_per_task >= 2 else ["thresholds must be >= 2"]


def summarise_failures(failures: Iterable[FailureNote], policy: EscalationPolicy | None = None) -> dict:
    """Group failures by (task, kind), find the recurring ones and say which must be escalated. Resolved failures are history
    (still counted in `recurrence`, because a failure that was 'fixed' three times is a pattern, not three accidents)."""
    pol = policy or EscalationPolicy()
    groups: dict[tuple[str, str], list[FailureNote]] = {}
    for f in failures:
        groups.setdefault((f.task, f.kind), []).append(f)
    per_task_open: dict[str, int] = {}
    for f in (f for g in groups.values() for f in g if f.open()):
        per_task_open[f.task] = per_task_open.get(f.task, 0) + 1
    escalate = []
    for (task, kind), g in sorted(groups.items()):
        reasons = []
        if kind in pol.always_escalate:
            reasons.append(f"{kind} always escalates")
        if len(g) >= pol.max_same_failure:
            reasons.append(f"{len(g)} occurrences of the same failure")
        if per_task_open.get(task, 0) >= pol.max_open_per_task:
            reasons.append(f"{per_task_open[task]} open failures on the task")
        if reasons:
            escalate.append({"task": task, "kind": kind, "count": len(g), "reasons": reasons, "last": g[-1].message[:120]})
    return {"n": sum(len(g) for g in groups.values()), "open": sum(1 for g in groups.values() for f in g if f.open()),
            "recurrence": {f"{t}/{k}": len(g) for (t, k), g in sorted(groups.items()) if len(g) > 1},
            "escalate": escalate}


def apply_escalations(failures: Sequence[FailureNote], policy: EscalationPolicy | None = None) -> tuple[FailureNote, ...]:
    """Mark the failures that meet the escalation policy as escalated (so they stop counting as 'unresolved and ignored' and
    start counting as 'reported to the owner'). Idempotent."""
    hot = {(e["task"], e["kind"]) for e in summarise_failures(failures, policy)["escalate"]}
    return tuple(dataclasses.replace(f, escalated=True) if (f.task, f.kind) in hot and f.open() else f for f in failures)


# ------------------------------------------------------------------------------------------------ progress and stalls
def detect_spinning(store: CheckpointStore, window: int = 4) -> dict:
    """A loop that keeps saving the same next action while completing nothing is busy but not progressing. True when the
    last `window` valid checkpoints share one next_action, one completed set and one pending set."""
    seqs = store.sequences()[-window:]
    states = []
    for q in seqs:
        try:
            states.append(store.read(q))
        except CheckpointCorrupt:
            continue
    if len(states) < window:
        return {"spinning": False, "checked": len(states)}
    sig = {(s.next_action, tuple(sorted(s.completed)), s.pending) for s in states}
    return {"spinning": len(sig) == 1, "checked": len(states), "next_action": states[-1].next_action}


def stall_report(store: CheckpointStore, now_s: float, max_age_s: float) -> dict:
    """Has anything been checkpointed recently? `now_s` is epoch seconds (explicit, never read from a clock here)."""
    age = store.age_s(now_s)
    return {"age_s": age, "stalled": age is None or age > max_age_s, "max_age_s": max_age_s,
            **{k: v for k, v in detect_spinning(store).items() if k != "checked"}}


def experiment_state_from_ledger(ledger) -> Callable[[str], str | None]:
    """Adapter so `resume` can ask the compute ledger (engine.learning.compute.ExperimentLedger) what became of the experiment
    that was in flight."""
    def look(key: str) -> str | None:
        e = ledger.get(key)
        return None if e is None else e["state"]
    return look


class ProgressLedger:
    """Append-only, hash-chained record of what the execution engine did (saves, resumes, failures, stops), on the existing
    engine.champion.Ledger. The checkpoint files say where we are; this says how we got here, and detects edits."""

    def __init__(self, path: str | Path):
        from engine.champion import Ledger
        self.ledger = Ledger(path)

    def note(self, event: str, subject: str, when, **detail: Any) -> None:
        self.ledger.append(event, subject, str(when), **{k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                                                         for k, v in detail.items()})

    def events(self, event: str | None = None) -> list[dict]:
        return [r for r in self.ledger.rows() if event is None or r["event"] == event]

    def verify(self) -> dict:
        return self.ledger.verify()

    def digest(self) -> dict:
        ev: dict[str, int] = {}
        for r in self.ledger.rows():
            ev[r["event"]] = ev.get(r["event"], 0) + 1
        return {"n": sum(ev.values()), "by_event": dict(sorted(ev.items())), "chain_ok": self.verify()["ok"]}


# ------------------------------------------------------------------------------------------------ line depth (section 60, 83)
def count_meaningful_lines(path: str | Path, include_docstrings: bool = True) -> int:
    """Non-blank, non-comment source lines. With include_docstrings=False, module/class/function docstrings are excluded
    too (the strict reading of 'meaningful'). Uses tokenize + ast so a '#' inside a string or a multi-line string is
    classified correctly."""
    import ast
    import io
    import tokenize
    src = Path(path).read_text(encoding="utf-8")
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in skip:
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    if not include_docstrings:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    lines.difference_update(range(first.lineno, cast(int, first.end_lineno) + 1))
    return len(lines)


def depth_report(sections: Mapping[str, tuple[Sequence[str | Path], int]], strict: bool = False) -> dict[str, tuple[int, int]]:
    """{section: (meaningful lines across its files, minimum)} for `CompletionEvidence.line_depth`. A missing file counts as
    zero lines rather than crashing: an absent module is a shortfall, and must show up as one."""
    out = {}
    for name, (paths, minimum) in sections.items():
        total = 0
        for p in paths:
            if Path(p).is_file():
                total += count_meaningful_lines(p, include_docstrings=not strict)
        out[name] = (total, int(minimum))
    return out


def derive_completion_evidence(tracker: "ChecklistTracker", depth: Mapping[str, tuple[int, int]], state: ExecutionState | None,
                               required_files: Mapping[str, Sequence[str | Path]], policy: EscalationPolicy | None = None) -> CompletionEvidence:
    """Build the stop-condition evidence from facts rather than assertions: sections are complete only if their checklist item
    is VALIDATED, tests/firewalls/reports exist only if the named files do, failures come from the last checkpoint."""
    def exist(kind: str) -> dict[str, bool]:
        return {n: bool(ps) and all(Path(p).is_file() for p in ps) for n, ps in required_files.items() if n.startswith(kind + ":")}

    failures = apply_escalations(state.failures, policy) if state else ()
    return CompletionEvidence(
        sections_complete={k: it.status == BuildStatus.VALIDATED for k, it in tracker.items.items()},
        line_depth=dict(depth), tests_exist=exist("test"), firewalls_exist=exist("firewall"), reports_exist=exist("report"),
        checklist_complete=bool(tracker.items) and tracker.next_incomplete(critical_only=False) is None,
        open_failures=sum(1 for f in failures if f.open()), falsely_marked=tuple(tracker.falsely_complete()))


def resume_report(plan: ResumePlan) -> str:
    """Plain-language account of what a restart is going to do and why (goes to the run log)."""
    lines = [f"resume action: {plan.action}"]
    if plan.state is not None:
        st = plan.state
        lines.append(f"from checkpoint #{st.sequence} saved {st.saved_at} (phase '{st.phase}', code {st.code_hash})")
        lines.append(f"{len(st.completed)} tasks complete, {len(st.pending)} pending, {len(st.open_failures())} failures open")
    lines += [f"- {r}" for r in plan.reasons]
    if plan.revalidate:
        lines.append("revalidate before trusting: " + ", ".join(plan.revalidate))
    if plan.skipped_corrupt:
        lines.append("corrupt checkpoint sequence numbers skipped: " + ", ".join(map(str, plan.skipped_corrupt)))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ store audit and repair
def audit_store(store: CheckpointStore) -> dict:
    """Health of the whole checkpoint directory, not just the newest file: which sequence numbers verify, which are corrupt,
    where numbering has gaps (a deleted file), whether LATEST.json points at the newest valid checkpoint, and whether any
    checkpoint was written under a newer schema than this code understands."""
    seqs = store.sequences()
    valid, corrupt, newer_schema = [], [], []
    for q in seqs:
        try:
            st = store.read(q)
            valid.append(q)
            if st.schema > SCHEMA_VERSION:
                newer_schema.append(q)
        except CheckpointCorrupt:
            corrupt.append(q)
    gaps = [q for q in range(seqs[0], seqs[-1] + 1) if q not in seqs] if seqs else []
    pointer = None
    lp = store.dir / "LATEST.json"
    if lp.is_file():
        try:
            pointer = json.loads(lp.read_text(encoding="utf-8")).get("sequence")
        except ValueError:
            pointer = None
    newest = valid[-1] if valid else None
    return {"sequences": len(seqs), "valid": valid, "corrupt": corrupt, "gaps": gaps, "newest_valid": newest,
            "latest_pointer": pointer, "latest_pointer_ok": pointer == newest, "newer_schema": newer_schema,
            "healthy": not corrupt and not gaps and pointer == newest and not newer_schema}


def repair_latest(store: CheckpointStore) -> int | None:
    """Point LATEST.json at the newest valid checkpoint (after a crash between writing a checkpoint and its pointer, or after
    the newest file was found corrupt). Never deletes or edits a checkpoint. Returns the sequence now pointed at."""
    st, _ = store.latest_valid()
    if st is None:
        return None
    _atomic_text(store.dir / "LATEST.json", json.dumps({"sequence": st.sequence, "digest": st.digest}))
    return st.sequence


def timeline(store: CheckpointStore) -> list[dict]:
    """One row per valid checkpoint: how the run progressed (tasks completed, open failures, what it planned next)."""
    rows: list[dict] = []
    for q in store.sequences():
        try:
            st = store.read(q)
        except CheckpointCorrupt:
            rows.append({"sequence": q, "corrupt": True})
            continue
        rows.append({"sequence": q, "saved_at": st.saved_at, "phase": st.phase, "completed": len(st.completed),
                     "pending": len(st.pending), "open_failures": len(st.open_failures()), "next_action": st.next_action,
                     "code_hash": st.code_hash, "experiment": st.current_experiment})
    return rows


def progress_made(store: CheckpointStore) -> dict:
    """Net movement between the first and last valid checkpoint: tasks completed, failures resolved, code changes."""
    rows = [r for r in timeline(store) if not r.get("corrupt")]
    if len(rows) < 2:
        return {"checkpoints": len(rows), "tasks_completed": 0, "code_changes": 0, "failures_delta": 0}
    first, last = rows[0], rows[-1]
    return {"checkpoints": len(rows), "tasks_completed": last["completed"] - first["completed"],
            "code_changes": sum(1 for a, b in zip(rows, rows[1:]) if a["code_hash"] != b["code_hash"]),
            "failures_delta": last["open_failures"] - first["open_failures"]}


def render_masterstock_block(state: ExecutionState, plan: ResumePlan | None = None, tracker: "ChecklistTracker | None" = None,
                             policy: EscalationPolicy | None = None) -> str:
    """The 'NEXT SESSION: START HERE' block for the Masterstock hand-off file, so any account can resume without asking:
    where we are, the very next action, what is broken and what must be escalated, and the exact labels the work may
    carry (never 'validated' without evidence - section 59)."""
    summ = summarise_failures(state.failures, policy)
    lines = ["NEXT SESSION: START HERE",
             f"run {state.run_id}, checkpoint #{state.sequence} ({state.saved_at}), phase '{state.phase}'",
             f"next action: {state.next_action}",
             f"code hash at save: {state.code_hash}   (tasks done under other code must be revalidated)"]
    if state.current_experiment:
        lines.append(f"experiment in flight: {state.current_experiment}")
    if plan is not None:
        lines.append(f"resume plan: {plan.action}" + ("; " + "; ".join(plan.reasons) if plan.reasons else ""))
    lines.append(f"open failures: {summ['open']}" + ("" if not summ["escalate"] else
                 " - ESCALATE: " + ", ".join(f"{e['task']}/{e['kind']} x{e['count']}" for e in summ["escalate"])))
    if tracker is not None:
        c = tracker.summary()
        lines.append("checklist: " + (", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "empty"))
        nxt = tracker.next_incomplete()
        lines.append("first incomplete critical item: " + (f"{nxt.item_id} {nxt.title} [{nxt.label.value}]" if nxt else "none"))
        bad = tracker.falsely_complete()
        if bad:
            lines.append("FALSELY MARKED COMPLETE (fix first): " + ", ".join(bad))
    lines.append("label to use for unproven work: " + ValidationLabel.NOT_VALIDATED.value)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ guarding a single risky step
@contextlib.contextmanager
def guarded_step(store: CheckpointStore, task: str, code_hash: str, clock: Callable[[], float] = time.time,
                 phase: str = "run", completed: dict | None = None, pending: Sequence[str] = (), experiment: str = "",
                 failures: Sequence[FailureNote] = (), next_action: str = "choose the next task"):
    """Wrap ONE step of a hand-driven script so it obeys the save/checkpoint/next-action protocol without a full
    ExecutionLoop: a checkpoint is written before the step ('running X'), and afterwards either the completed record
    (success), or a failure note plus an 'investigate' next action (any exception, which is then re-raised). A
    FirewallBreach is recorded as escalated. KeyboardInterrupt is recorded as INTERRUPTED so the resume plan knows the
    step did not finish. `completed` is updated in place on success."""
    done = completed if completed is not None else {}
    fails = list(failures)
    store.save(clock(), phase, f"running {task}", code_hash, experiment, fails, done, pending)
    try:
        yield
    except FirewallBreach as e:
        fails.append(FailureNote(str(clock()), task, "FIREWALL", str(e)[:300], escalated=True))
        store.save(clock(), phase, f"INVESTIGATE firewall breach in {task}", code_hash, experiment, fails, done, pending)
        raise
    except BaseException as e:
        kind = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "OOM" if isinstance(e, MemoryError) else "ERROR"
        fails.append(FailureNote(str(clock()), task, kind, f"{type(e).__name__}: {e}"[:300]))
        store.save(clock(), phase, f"retry or fix {task}", code_hash, experiment, fails, done, [task, *pending])
        raise
    else:
        done[task] = code_hash
        store.save(clock(), phase, next_action, code_hash, "", [dataclasses.replace(f, resolved=True) if f.task == task else f
                                                                for f in fails], done, pending)


def new_run_id(prefix: str, when, salt: str = "") -> str:
    """A run id that is unique per (prefix, date, salt) yet human-readable: 'prefix_YYYYMMDD_hhhh'. Deterministic - the same
    inputs always give the same id, so a restarted process rejoins its own run instead of starting a second one."""
    d = str(as_date(when)).replace("-", "")
    return f"{prefix}_{d}_{stable_hash([prefix, d, salt], 4)}"


def diff_states(old: ExecutionState, new: ExecutionState) -> dict:
    """What changed between two checkpoints of one run: tasks newly completed, failures added or resolved, next action, code."""
    return {"completed_added": sorted(set(new.completed) - set(old.completed)),
            "completed_lost": sorted(set(old.completed) - set(new.completed)),
            "failures_added": len(new.failures) - len(old.failures),
            "failures_resolved": sum(f.resolved for f in new.failures) - sum(f.resolved for f in old.failures),
            "next_action": None if old.next_action == new.next_action else (old.next_action, new.next_action),
            "code_changed": old.code_hash != new.code_hash, "experiment": (old.current_experiment, new.current_experiment)
            if old.current_experiment != new.current_experiment else None}


# ------------------------------------------------------------------------------------------------ resume that refuses stale state
class StaleState(RuntimeError):
    """The saved state was made by different code than the code now loaded, and the caller did not explicitly accept that."""


def resume_verified(store: CheckpointStore, code_hash: str, allow_code_change: bool = False,
                    experiment_state: Callable[[str], str | None] | None = None) -> ResumePlan:
    """`resume`, but fail-closed on stale state. If the newest valid checkpoint was saved under a different code hash - or
    records completed tasks done under one - the state is NOT resumed: StaleState names exactly what is stale. With
    allow_code_change=True the plan is returned instead and its `revalidate` list says what must be redone before any of it is
    trusted. A missing or fully corrupt store is a fresh start, never an error and never a silent guess."""
    plan = resume(store, code_hash, experiment_state)
    st = plan.state
    if st is None:
        return plan
    problems = []
    if st.code_hash != code_hash:
        problems.append(f"checkpoint #{st.sequence} was saved under code {st.code_hash}, engine is {code_hash}")
    if plan.revalidate:
        problems.append(f"tasks completed under other code: {list(plan.revalidate)}")
    if problems and not allow_code_change:
        raise StaleState("; ".join(problems))
    return plan


# ------------------------------------------------------------------------------------------------ the interruption record (section 58)
@dataclass(frozen=True)
class InterruptionRecord:
    """Exactly what section 58 says must be written when work is interrupted, as a typed, checksummed record: the next action,
    the current failures, the current experiment and the code hash (plus where it was saved, so a reader can find the full
    checkpoint)."""
    run_id: str
    checkpoint_sequence: int
    saved_at: str
    next_action: str
    current_experiment: str
    code_hash: str
    failures: tuple[FailureNote, ...] = ()
    reason: str = "interrupted"

    def validate(self) -> list[str]:
        errs = []
        if not self.next_action:
            errs.append("next_action missing")
        if not self.code_hash:
            errs.append("code_hash missing")
        if not self.run_id:
            errs.append("run_id missing")
        return errs

    @classmethod
    def from_state(cls, state: ExecutionState, reason: str = "interrupted") -> "InterruptionRecord":
        return cls(state.run_id, state.sequence, state.saved_at, state.next_action, state.current_experiment, state.code_hash,
                   state.open_failures(), reason)


def write_interruption(path: str | Path, rec: InterruptionRecord) -> Path:
    """Atomic JSON with an embedded body hash (engine.checkpoint._body_hash). An invalid record is refused, not written."""
    errs = rec.validate()
    if errs:
        raise CheckpointCorrupt(f"refusing to write an invalid interruption record: {errs}")
    body = json.loads(canonical_json(rec))
    p = Path(path)
    _atomic_text(p, json.dumps({"sha256": bundle._body_hash(body), "body": body}, sort_keys=True, indent=1))
    return p


def read_interruption(path: str | Path) -> InterruptionRecord:
    """Read and verify. A missing, unparseable, altered or invalid record raises CheckpointCorrupt - the caller then falls back
    to the checkpoint store instead of trusting a damaged hand-off."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        body = doc["body"]
    except (OSError, ValueError, KeyError) as e:
        raise CheckpointCorrupt(f"interruption record unreadable: {type(e).__name__}") from e
    if bundle._body_hash(body) != doc.get("sha256"):
        raise CheckpointCorrupt("interruption record content hash mismatch")
    fnames = {f.name for f in dataclasses.fields(FailureNote)}
    names = {f.name for f in dataclasses.fields(InterruptionRecord)}
    body = {k: v for k, v in body.items() if k in names}
    body["failures"] = tuple(FailureNote(**{k: v for k, v in f.items() if k in fnames}) for f in body.get("failures", ()))
    rec = InterruptionRecord(**body)
    errs = rec.validate()
    if errs:
        raise CheckpointCorrupt(f"interruption record invalid: {errs}")
    return rec


# ------------------------------------------------------------------------------------------------ pruning
@dataclass(frozen=True)
class PrunePolicy:
    keep_last: int = 10             # always keep this many newest checkpoints
    keep_every: int = 25            # and every Nth sequence number as a coarse history (0 disables)
    keep_sequences: tuple[int, ...] = ()     # and these (e.g. the sequence a milestone bundle was sealed from)

    def validate(self) -> list[str]:
        return [] if self.keep_last >= 3 and self.keep_every >= 0 else ["keep_last must be >= 3 and keep_every >= 0"]


def prune(store: CheckpointStore, policy: PrunePolicy | None = None, dry_run: bool = False) -> list[int]:
    """Delete old checkpoints that the policy does not protect. Never deletes the newest VALID checkpoint (it is what a resume
    would use), never deletes anything newer than it, and never deletes a corrupt file - a corrupt file is evidence and is left
    for `audit_store` to report. Returns the sequence numbers removed (or that would be removed, for a dry run)."""
    pol = policy or PrunePolicy()
    errs = pol.validate()
    if errs:
        raise ValueError(errs)
    seqs = store.sequences()
    newest, _ = store.latest_valid()
    if newest is None:
        return []
    protect = set(seqs[-pol.keep_last:]) | {newest.sequence} | set(pol.keep_sequences)
    if pol.keep_every:
        protect |= {q for q in seqs if q % pol.keep_every == 0}
    doomed = []
    for q in seqs:
        if q in protect or q > newest.sequence:
            continue
        try:
            store.read(q)
        except CheckpointCorrupt:
            continue
        doomed.append(q)
    if not dry_run:
        for q in doomed:
            store._path(q).unlink()
    return doomed
