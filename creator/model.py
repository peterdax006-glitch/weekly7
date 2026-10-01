"""Creator K01 data model (C77 secs 7, 13, 18, 22, 30, 31, 35, 37, 63, 71; package CR01) - IMPLEMENTED, NOT VALIDATED.

Typed, frozen, schema-validated development records. Every record is validated when it is constructed (fail closed): field types
are checked against the annotations, enums are the exact C77 vocabularies (sec 7 status states, sec 71 uncertainty states),
required text is non-empty, numbers are finite. Records carry their lineage (`parents`), the role that created them and evidence
references (path + sha256). Identity and provenance are added by the ledger (creator/ledger.py), never typed by a caller.

Verdicts are computed, not typed (C77 sec 30, 35): `improvement_verdict` is the one rule that turns measurements into
IMPROVEMENT / NO_EFFECT / REGRESSION / INSUFFICIENT_EVIDENCE, and the ledger re-runs it on every ImprovementClaim. TestRun has no
outcome field at all - its outcome is derived from its counts."""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import os
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEXN = re.compile(r"^[0-9a-f]{8,64}$")
_ID = re.compile(r"^[A-Z]{2,4}-[0-9a-f]{20}$")
_FUNC = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_PKG = re.compile(r"^[A-Z]{1,4}\d{1,4}[a-z]?$")


class ModelError(ValueError):
    """A record failed schema or semantic validation. Never caught-and-continued: the record does not exist."""


class Status(str, enum.Enum):
    """C77 sec 7 - exactly these states, never collapsed."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    IMPLEMENTED = "IMPLEMENTED"
    TESTED = "TESTED"
    INTENDED_BEHAVIOR_VERIFIED = "INTENDED_BEHAVIOR_VERIFIED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SCIENTIFICALLY_LIMITED = "SCIENTIFICALLY_LIMITED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class Uncertainty(str, enum.Enum):
    """C77 sec 71."""
    KNOWN = "KNOWN"
    LIKELY = "LIKELY"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"
    CONTRADICTED = "CONTRADICTED"
    UNTESTED = "UNTESTED"
    FAILED = "FAILED"
    VALIDATED = "VALIDATED"


class Role(str, enum.Enum):
    """Who wrote a record (ARCHITECTURE K05 worker roles, plus the kernel itself, the owner, and Claude as foundation builder)."""
    KERNEL = "kernel"
    RESEARCHER = "researcher"
    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    TESTER = "tester"
    DEBUGGER = "debugger"
    ADVERSARY = "adversary"
    VALIDATOR = "validator"
    AUDITOR = "auditor"
    OWNER = "owner"
    BUILDER = "builder"


INDEPENDENT_ROLES = frozenset({Role.VALIDATOR, Role.AUDITOR, Role.OWNER})       # who may move anything to VALIDATED (sec 48)


class Priority(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GapKind(str, enum.Enum):
    """ARCHITECTURE K04 classes."""
    CAPABILITY = "CAPABILITY"
    KNOWLEDGE = "KNOWLEDGE"
    ARCHITECTURE = "ARCHITECTURE"
    TESTING = "TESTING"
    INTEGRATION = "INTEGRATION"
    EVIDENCE = "EVIDENCE"


class Verdict(str, enum.Enum):
    """C77 sec 30: insufficient evidence is its own verdict, never PASS."""
    IMPROVEMENT = "IMPROVEMENT"
    NO_EFFECT = "NO_EFFECT"
    REGRESSION = "REGRESSION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DecisionVerdict(str, enum.Enum):
    ADOPT = "ADOPT"
    ROLLBACK = "ROLLBACK"
    REJECT = "REJECT"


class TestOutcome(str, enum.Enum):
    __test__ = False                                     # not a pytest class
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    EMPTY = "EMPTY"                                      # nothing ran: never a pass


class Split(str, enum.Enum):
    """Which part of the development-task suite a measurement came from (C77 sec 32; devbench dev/holdout)."""
    DEV = "dev"
    HOLDOUT = "holdout"
    NONE = "none"


# ------------------------------------------------------------------------------------------------ value objects

def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    """A file that supports a record: a path relative to the evidence root plus the sha256 it had when cited."""
    path: str
    sha256: str
    kind: str = "artifact"

    def __post_init__(self) -> None:
        errs = []
        if not isinstance(self.path, str) or not self.path.strip():
            errs.append("evidence.path empty")
        elif Path(self.path).is_absolute() or ".." in Path(self.path).parts or self.path.startswith(("/", "\\")):
            errs.append(f"evidence.path must be relative inside the evidence root: {self.path!r}")
        if not isinstance(self.sha256, str) or not _HEX64.match(self.sha256):
            errs.append(f"evidence.sha256 is not a sha256 hex digest: {self.sha256!r}")
        if not isinstance(self.kind, str) or not self.kind.strip():
            errs.append("evidence.kind empty")
        if errs:
            raise ModelError("; ".join(errs))

    @classmethod
    def of(cls, path: str | os.PathLike, root: str | os.PathLike, kind: str = "artifact") -> "EvidenceRef":
        """Cite a file as it is NOW (the hash is computed, never typed)."""
        p, r = Path(path), Path(root)
        full = p if p.is_absolute() else r / p
        rel = full.resolve().relative_to(r.resolve()).as_posix()
        return cls(path=rel, sha256=sha256_file(full), kind=kind)

    def problem(self, root: str | os.PathLike) -> Optional[str]:
        """None when the file exists under `root` with the cited hash; otherwise what is wrong."""
        full = Path(root) / self.path
        if not full.is_file():
            return f"evidence missing: {self.path}"
        got = sha256_file(full)
        return None if got == self.sha256 else f"evidence changed: {self.path} cited {self.sha256[:12]} now {got[:12]}"


@dataclass(frozen=True)
class ComputationRef:
    """The computation that produced a number or verdict: function (module:qualname), the hash of its code, digests of its
    inputs and output, and the ledger records it read. A verdict without one is refused (C77 sec 35)."""
    function: str
    code_hash: str
    inputs_sha256: str
    output_sha256: str
    record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_ids", tuple(self.record_ids))
        errs = []
        if not isinstance(self.function, str) or not _FUNC.match(self.function):
            errs.append(f"computation.function must be 'module:qualname': {self.function!r}")
        for name in ("code_hash", "inputs_sha256", "output_sha256"):
            v = getattr(self, name)
            if not isinstance(v, str) or not _HEXN.match(v):
                errs.append(f"computation.{name} is not a hex digest: {v!r}")
        for r in self.record_ids:
            if not isinstance(r, str) or not _ID.match(r):
                errs.append(f"computation.record_ids has a malformed id: {r!r}")
        if errs:
            raise ModelError("; ".join(errs))


@dataclass(frozen=True)
class Provenance:
    """C77 sec 35, filled by the ledger at append time from engine.provenance."""
    engine_tree_hash: str
    creator_tree_hash: str
    git_commit: str
    config_hash: Optional[str]
    seed: Optional[int]
    timestamp: str

    def __post_init__(self) -> None:
        errs = [f"provenance.{n} empty" for n in ("engine_tree_hash", "creator_tree_hash", "git_commit", "timestamp")
                if not isinstance(getattr(self, n), str) or not getattr(self, n).strip()]
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            errs.append("provenance.seed must be an int or None")
        if self.config_hash is not None and not isinstance(self.config_hash, str):
            errs.append("provenance.config_hash must be a str or None")
        if errs:
            raise ModelError("; ".join(errs))


# ------------------------------------------------------------------------------------------------ type checking

def _is_union(origin: Any) -> bool:
    return origin is typing.Union or origin is types.UnionType


def _coerce(value: Any, hint: Any) -> Any:
    """Normalise JSON-shaped input to the annotated type (list -> tuple, str -> enum, dict -> value object). Anything that does
    not fit is returned unchanged so that `_type_errors` reports it; coercion never invents a value."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if _is_union(origin):
        if value is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            c = _coerce(value, a)
            if not _type_errors(c, a, "x"):
                return c
        return value
    if origin is tuple and isinstance(value, (list, tuple)):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        if len(args) == len(value):
            return tuple(_coerce(v, a) for v, a in zip(value, args))
        return tuple(value)
    if isinstance(hint, type) and issubclass(hint, enum.Enum) and isinstance(value, str) and not isinstance(value, hint):
        try:
            return hint(value)
        except ValueError:
            return value
    if isinstance(hint, type) and dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
        hints = typing.get_type_hints(hint)
        try:
            return hint(**{k: _coerce(v, hints.get(k, Any)) for k, v in value.items()})
        except (TypeError, ModelError):
            return value
    return value


def _type_errors(value: Any, hint: Any, path: str) -> list[str]:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if hint is Any:
        return []
    if _is_union(origin):
        if any(not _type_errors(value, a, path) for a in args):
            return []
        return [f"{path}: {value!r} does not match {hint}"]
    if hint is type(None):
        return [] if value is None else [f"{path}: expected None"]
    if origin is tuple:
        if not isinstance(value, tuple):
            return [f"{path}: expected a tuple, got {type(value).__name__}"]
        if len(args) == 2 and args[1] is Ellipsis:
            return [e for i, v in enumerate(value) for e in _type_errors(v, args[0], f"{path}[{i}]")]
        if len(args) != len(value):
            return [f"{path}: expected {len(args)} items, got {len(value)}"]
        return [e for i, (v, a) in enumerate(zip(value, args)) for e in _type_errors(v, a, f"{path}[{i}]")]
    if hint is bool:
        return [] if isinstance(value, bool) else [f"{path}: expected bool"]
    if hint is int:
        return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{path}: expected int"]
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path}: expected a number"]
        return [] if math.isfinite(value) else [f"{path}: non-finite number {value!r}"]
    if hint is str:
        return [] if isinstance(value, str) else [f"{path}: expected str"]
    if isinstance(hint, type):
        return [] if isinstance(value, hint) else [f"{path}: expected {hint.__name__}, got {type(value).__name__}"]
    return [f"{path}: unsupported annotation {hint}"]


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    if cls not in _HINTS:
        _HINTS[cls] = {k: v for k, v in typing.get_type_hints(cls).items()
                       if k in {f.name for f in dataclasses.fields(cls)}}
    return _HINTS[cls]


def _plain(v: Any) -> Any:
    if isinstance(v, enum.Enum):
        return v.value
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return v


# ------------------------------------------------------------------------------------------------ records

@dataclass(frozen=True, kw_only=True)
class Record:
    """Base of every ledger record. Subclasses declare RTYPE/PREFIX, whether they carry a status (STATEFUL), the record types
    at least one parent must have (PARENTS; None = no lineage requirement, () = at least one parent of any type) and which
    fields are references to other records (REFS: field -> allowed types, () = any type)."""
    RTYPE: ClassVar[str] = ""
    PREFIX: ClassVar[str] = ""
    VERSION: ClassVar[int] = 1
    STATEFUL: ClassVar[bool] = False
    PARENTS: ClassVar[Optional[tuple[str, ...]]] = None
    REFS: ClassVar[Mapping[str, tuple[str, ...]]] = {}
    TEXT: ClassVar[tuple[str, ...]] = ()                    # str / tuple[str] fields that must be non-empty

    created_by: Role
    parents: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        for name, hint in _hints(type(self)).items():
            object.__setattr__(self, name, _coerce(getattr(self, name), hint))
        errs = self.validation_errors()
        if errs:
            raise ModelError(f"{self.RTYPE}: " + "; ".join(errs))

    def validation_errors(self) -> list[str]:
        # named so no record field can shadow it (TestRun has an `errors` count - 30 Sep, it did)
        errs: list[str] = []
        for name, hint in _hints(type(self)).items():
            errs += _type_errors(getattr(self, name), hint, name)
        if errs:
            return errs                                         # semantic checks assume the types are right
        for p in self.parents:
            if not _ID.match(p):
                errs.append(f"parents: malformed id {p!r}")
        if len(set(self.parents)) != len(self.parents):
            errs.append("parents: duplicate id")
        for name in self.TEXT:
            v = getattr(self, name)
            if isinstance(v, str) and not v.strip():
                errs.append(f"{name} is empty")
            elif isinstance(v, tuple) and (not v or any(not str(x).strip() for x in v)):
                errs.append(f"{name} must list at least one non-empty item")
        for name in self.REFS:
            for r in self.ref_values(name):
                if not _ID.match(r):
                    errs.append(f"{name}: malformed id {r!r}")
        errs += self._check()
        return errs

    def _check(self) -> list[str]:
        return []

    def ref_values(self, name: str) -> tuple[str, ...]:
        v = getattr(self, name)
        if v is None:
            return ()
        return tuple(v) if isinstance(v, tuple) else (v,)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def covers(self, subject_id: str) -> bool:
        """Does this record speak about `subject_id` (as a parent or through a subject field)?"""
        if subject_id in self.parents:
            return True
        for name in ("subject_id", "subject_ids"):
            if hasattr(self, name) and subject_id in self.ref_values(name):
                return True
        return False


def _uncertainty_check(rec: Any) -> list[str]:
    """KNOWN / VALIDATED are claims; claims need cited evidence, and VALIDATED needs the computation behind it (sec 35, 71)."""
    u = rec.uncertainty
    errs = []
    if u in (Uncertainty.KNOWN, Uncertainty.VALIDATED) and not rec.evidence:
        errs.append(f"uncertainty {u.value} without evidence")
    if u is Uncertainty.VALIDATED and getattr(rec, "computation", None) is None:
        errs.append("uncertainty VALIDATED without the computation that validated it")
    return errs


@dataclass(frozen=True, kw_only=True)
class Objective(Record):
    """C77 sec 12: an objective as a structured problem."""
    RTYPE, PREFIX, STATEFUL = "Objective", "OBJ", True
    TEXT = ("statement", "acceptance_criteria")
    statement: str
    outcomes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Requirement(Record):
    """C77 sec 13: every field the requirement compiler must store. The current state lives in the ledger's transitions."""
    RTYPE, PREFIX, STATEFUL, PARENTS = "Requirement", "REQ", True, ("Objective", "Requirement")
    REFS = {"depends_on": ("Requirement",)}
    TEXT = ("key", "description", "acceptance_test", "measurement_method", "failure_condition", "validation_method",
            "evidence_location")
    key: str
    description: str
    priority: Priority
    acceptance_test: str
    measurement_method: str
    failure_condition: str
    validation_method: str
    evidence_location: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Capability(Record):
    RTYPE, PREFIX, STATEFUL = "Capability", "CAP", True
    TEXT = ("name", "component", "description")
    name: str
    component: str
    description: str
    uncertainty: Uncertainty = Uncertainty.UNTESTED
    computation: Optional[ComputationRef] = None

    def _check(self) -> list[str]:
        return _uncertainty_check(self)


@dataclass(frozen=True, kw_only=True)
class Gap(Record):
    RTYPE, PREFIX, STATEFUL, PARENTS = "Gap", "GAP", True, ("Requirement", "Capability", "Objective")
    REFS = {"depends_on": ("Gap",)}
    TEXT = ("description",)
    kind: GapKind
    description: str
    importance: float
    depends_on: tuple[str, ...] = ()

    def _check(self) -> list[str]:
        return [] if 0.0 <= self.importance <= 1.0 else ["importance must lie in [0, 1]"]


@dataclass(frozen=True, kw_only=True)
class ResearchQuestion(Record):
    RTYPE, PREFIX, STATEFUL = "ResearchQuestion", "RQ", True
    PARENTS = ("Gap", "Objective", "Requirement", "Failure", "DesignOption", "ResearchQuestion", "Diagnosis")
    TEXT = ("question",)
    question: str
    priority: Priority
    uncertainty: Uncertainty = Uncertainty.UNKNOWN

    def _check(self) -> list[str]:
        return _uncertainty_check(self)


@dataclass(frozen=True, kw_only=True)
class Finding(Record):
    RTYPE, PREFIX, PARENTS = "Finding", "FND", ("ResearchQuestion", "Experiment", "Diagnosis", "Failure")
    TEXT = ("statement",)
    statement: str
    uncertainty: Uncertainty
    sources: tuple[str, ...] = ()
    computation: Optional[ComputationRef] = None

    def _check(self) -> list[str]:
        errs = _uncertainty_check(self)
        if self.uncertainty is Uncertainty.CONTRADICTED and not (self.sources or self.evidence):
            errs.append("CONTRADICTED needs the contradicting source")
        return errs


@dataclass(frozen=True, kw_only=True)
class DesignOption(Record):
    """K08: one of N alternatives; selection is a Decision, rejected options stay in the ledger."""
    RTYPE, PREFIX, STATEFUL = "DesignOption", "DES", True
    PARENTS = ("Gap", "Requirement", "ResearchQuestion", "Finding", "Objective", "WorkPackage")
    TEXT = ("decision_key", "summary")
    decision_key: str
    summary: str
    assumptions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    cost_estimate: float = 0.0
    risk_estimate: float = 0.5
    uncertainty: Uncertainty = Uncertainty.UNTESTED

    def _check(self) -> list[str]:
        errs = _uncertainty_check(self)
        if self.cost_estimate < 0:
            errs.append("cost_estimate < 0")
        if not 0.0 <= self.risk_estimate <= 1.0:
            errs.append("risk_estimate must lie in [0, 1]")
        return errs


@dataclass(frozen=True, kw_only=True)
class WorkPackage(Record):
    """C77 sec 63 (every field) plus the sec 18 extras (rationale = why_it_exists, outputs, failure conditions)."""
    RTYPE, PREFIX, STATEFUL = "WorkPackage", "WP", True
    PARENTS = ("Objective", "Requirement", "Gap", "WorkPackage", "DesignOption")
    TEXT = ("package_id", "objective", "why_it_exists", "implementation_requirements", "interfaces", "data_flow",
            "test_requirements", "validation_requirements", "expected_failure_modes", "evidence_requirements",
            "failure_conditions", "rollback_requirements", "completion_criteria", "anti_premature_completion")
    package_id: str
    objective: str
    why_it_exists: str
    prerequisites: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    implementation_requirements: tuple[str, ...]
    interfaces: tuple[str, ...]
    data_flow: str
    dependencies: tuple[str, ...]
    test_requirements: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    expected_failure_modes: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    rollback_requirements: tuple[str, ...]
    completion_criteria: tuple[str, ...]
    anti_premature_completion: tuple[str, ...]
    meaningful_code_depth: int

    def _check(self) -> list[str]:
        errs = []
        if not _PKG.match(self.package_id):
            errs.append(f"package_id {self.package_id!r} is not like CR01")
        if self.meaningful_code_depth < 0:
            errs.append("meaningful_code_depth < 0")
        if self.package_id in self.dependencies:
            errs.append("a package cannot depend on itself")
        return errs


@dataclass(frozen=True, kw_only=True)
class ChangeProposal(Record):
    """C77 sec 22: reason, parent objective, originating task, affected components, expected effect. The actual result is
    never typed here; it is the Decision / ImprovementClaim that later reference this proposal."""
    RTYPE, PREFIX, STATEFUL, PARENTS = "ChangeProposal", "CHG", True, ("WorkPackage", "Repair", "Objective")
    REFS = {"parent_objective": ("Objective",), "originating_task": ("WorkPackage",)}
    TEXT = ("reason", "affected_components", "expected_effect")
    reason: str
    parent_objective: str
    originating_task: str
    affected_components: tuple[str, ...]
    expected_effect: str


@dataclass(frozen=True, kw_only=True)
class Experiment(Record):
    RTYPE, PREFIX, STATEFUL = "Experiment", "EXP", True
    PARENTS = ("ChangeProposal", "ResearchQuestion", "DesignOption", "WorkPackage", "Gap")
    TEXT = ("hypothesis", "design", "metrics", "baseline_ref", "candidate_ref")
    hypothesis: str
    design: str
    metrics: tuple[str, ...]
    seed: int
    baseline_ref: str
    candidate_ref: str
    uses_holdout: bool = False


@dataclass(frozen=True, kw_only=True)
class TestRun(Record):
    """Raw counts and the log as evidence; the outcome is a derived property, never a stored claim."""
    __test__ = False
    RTYPE, PREFIX = "TestRun", "TR"
    REFS = {"subject_ids": ()}
    TEXT = ("command",)
    command: str
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_s: float
    subject_ids: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()

    def _check(self) -> list[str]:
        errs = [f"{n} < 0" for n in ("passed", "failed", "errors", "skipped") if getattr(self, n) < 0]
        if self.duration_s < 0:
            errs.append("duration_s < 0")
        if not self.evidence:
            errs.append("a test run must cite its raw output as evidence")
        return errs

    @property
    def outcome(self) -> TestOutcome:
        if self.errors:
            return TestOutcome.ERROR
        if self.failed:
            return TestOutcome.FAIL
        return TestOutcome.PASS if self.passed else TestOutcome.EMPTY


@dataclass(frozen=True, kw_only=True)
class Failure(Record):
    RTYPE, PREFIX = "Failure", "FL"
    REFS = {"subject_id": (), "test_run_id": ("TestRun",)}
    TEXT = ("subject_id", "symptom", "classification", "reproduction")
    subject_id: str
    symptom: str
    classification: str
    reproduction: str
    test_run_id: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class Diagnosis(Record):
    RTYPE, PREFIX, PARENTS = "Diagnosis", "DG", ("Failure", "Diagnosis")
    REFS = {"failure_id": ("Failure",)}
    TEXT = ("hypotheses", "root_cause")
    failure_id: str
    hypotheses: tuple[str, ...]
    root_cause: str
    uncertainty: Uncertainty = Uncertainty.UNCERTAIN

    def _check(self) -> list[str]:
        errs = _uncertainty_check(self)
        if self.failure_id not in self.parents:
            errs.append("failure_id must also be a parent (lineage)")
        return errs


@dataclass(frozen=True, kw_only=True)
class Repair(Record):
    RTYPE, PREFIX, PARENTS = "Repair", "RP", ("Diagnosis",)
    REFS = {"diagnosis_id": ("Diagnosis",), "change_proposal_id": ("ChangeProposal",)}
    TEXT = ("description",)
    diagnosis_id: str
    description: str
    change_proposal_id: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class Measurement(Record):
    """C77 sec 31: one measured number with its uncertainty, population and conditions, and the computation that made it."""
    RTYPE, PREFIX, PARENTS = "Measurement", "MS", ()
    TEXT = ("metric", "population", "conditions")
    metric: str
    value: float
    stderr: float
    n: int
    population: str
    conditions: str
    higher_is_better: bool
    split: Split
    computation: ComputationRef

    def _check(self) -> list[str]:
        errs = []
        if self.stderr < 0:
            errs.append("stderr < 0")
        if self.n < 1:
            errs.append("n < 1")
        return errs


@dataclass(frozen=True, kw_only=True)
class ImprovementClaim(Record):
    """C77 sec 30 as data: baseline, replicated candidate, regression pairs, holdout pair. The verdict must equal what
    `improvement_verdict` computes from the referenced measurements; the ledger recomputes it and refuses a mismatch."""
    RTYPE, PREFIX, PARENTS = "ImprovementClaim", "IC", ("ChangeProposal", "Experiment")
    REFS = {"subject_id": ("ChangeProposal", "Experiment"), "baseline_ids": ("Measurement",),
            "candidate_ids": ("Measurement",), "regression_baseline_ids": ("Measurement",),
            "regression_candidate_ids": ("Measurement",), "holdout_baseline_id": ("Measurement",),
            "holdout_candidate_id": ("Measurement",)}
    subject_id: str
    baseline_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    verdict: Verdict
    computation: ComputationRef
    regression_baseline_ids: tuple[str, ...] = ()
    regression_candidate_ids: tuple[str, ...] = ()
    holdout_baseline_id: Optional[str] = None
    holdout_candidate_id: Optional[str] = None
    min_effect: float = 0.0
    z: float = 2.0

    def _check(self) -> list[str]:
        errs = []
        if len(self.regression_baseline_ids) != len(self.regression_candidate_ids):
            errs.append("regression baseline/candidate ids must pair up")
        if (self.holdout_baseline_id is None) != (self.holdout_candidate_id is None):
            errs.append("holdout needs both a baseline and a candidate measurement")
        if self.min_effect < 0 or self.z <= 0:
            errs.append("min_effect must be >= 0 and z > 0")
        if self.computation.function != VERDICT_FUNCTION:
            errs.append(f"verdict must come from {VERDICT_FUNCTION}, not {self.computation.function}")
        return errs


@dataclass(frozen=True, kw_only=True)
class Decision(Record):
    RTYPE, PREFIX = "Decision", "DEC"
    REFS = {"subject_id": ("ChangeProposal", "Experiment", "DesignOption", "WorkPackage", "Repair"),
            "claim_id": ("ImprovementClaim",)}
    TEXT = ("subject_id", "reason")
    subject_id: str
    verdict: DecisionVerdict
    reason: str
    claim_id: Optional[str] = None

    def _check(self) -> list[str]:
        if self.verdict is DecisionVerdict.ADOPT and self.claim_id is None:
            return ["ADOPT needs the ImprovementClaim it rests on (C77 sec 30)"]
        return []


@dataclass(frozen=True, kw_only=True)
class StrategyOutcome(Record):
    """K13 input: how a development strategy fared on one problem. `success` is checked against the subject's ledger state."""
    RTYPE, PREFIX = "StrategyOutcome", "SO"
    REFS = {"subject_id": (), "measurement_ids": ("Measurement",)}
    TEXT = ("strategy_id", "problem_class")
    strategy_id: str
    problem_class: str
    subject_id: str
    success: bool
    cost: float
    duration_s: float
    measurement_ids: tuple[str, ...] = ()

    def _check(self) -> list[str]:
        return [f"{n} < 0" for n in ("cost", "duration_s") if getattr(self, n) < 0]


@dataclass(frozen=True, kw_only=True)
class CheckpointMarker(Record):
    """C77 sec 59-60: the chain head and the folded-state digest at a moment; the ledger checks both when it is appended."""
    RTYPE, PREFIX = "CheckpointMarker", "CKP"
    TEXT = ("label",)
    label: str
    chain_head: str
    chain_records: int
    view_digest: str
    source_rev: str = "unknown"
    note: str = ""

    def _check(self) -> list[str]:
        errs = []
        if not _HEX64.match(self.chain_head):
            errs.append("chain_head is not a sha256 digest")
        if self.chain_records < 0:
            errs.append("chain_records < 0")
        if not _HEXN.match(self.view_digest):
            errs.append("view_digest is not a hex digest")
        return errs


@dataclass(frozen=True, kw_only=True)
class Transition(Record):
    """A status change is itself a record (C77 sec 61): from, to, why, and which records justify it."""
    RTYPE, PREFIX = "Transition", "TRN"
    REFS = {"subject_id": (), "justification_ids": ()}
    TEXT = ("subject_id", "reason")
    subject_id: str
    from_state: Status
    to_state: Status
    reason: str
    justification_ids: tuple[str, ...] = ()

    def _check(self) -> list[str]:
        errs = []
        if self.to_state not in LEGAL[self.from_state]:
            errs.append(f"illegal transition {self.from_state.value} -> {self.to_state.value}")
        if self.to_state is Status.VALIDATED and self.created_by not in INDEPENDENT_ROLES:
            errs.append(f"{self.created_by.value} may not validate; only an independent validator/auditor/owner (sec 48)")
        return errs


RECORD_TYPES: dict[str, type[Record]] = {c.RTYPE: c for c in (
    Objective, Requirement, Capability, Gap, ResearchQuestion, Finding, DesignOption, WorkPackage, ChangeProposal, Experiment,
    TestRun, Failure, Diagnosis, Repair, Measurement, ImprovementClaim, Decision, StrategyOutcome, CheckpointMarker, Transition)}
STATEFUL_TYPES = frozenset(n for n, c in RECORD_TYPES.items() if c.STATEFUL)
UNIQUE_KEYS = {"Requirement": "key", "WorkPackage": "package_id"}

# (rtype, stored version) -> function upgrading the stored field dict by one version. Empty while every type is at version 1;
# a stored version newer than the code is refused rather than guessed at.
MIGRATIONS: dict[tuple[str, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def record_from_dict(rtype: str, version: int, data: Mapping[str, Any]) -> Record:
    """Rebuild a typed record from stored JSON, applying registered migrations; unknown types/fields/versions fail closed."""
    cls = RECORD_TYPES.get(rtype)
    if cls is None:
        raise ModelError(f"unknown record type {rtype!r}")
    if not isinstance(version, int) or version < 1 or version > cls.VERSION:
        raise ModelError(f"{rtype} schema version {version!r} is not readable by this code (current {cls.VERSION})")
    d = dict(data)
    while version < cls.VERSION:
        step = MIGRATIONS.get((rtype, version))
        if step is None:
            raise ModelError(f"no migration for {rtype} v{version} -> v{version + 1}")
        d, version = step(d), version + 1
    names = {f.name for f in dataclasses.fields(cls)}
    extra = sorted(set(d) - names)
    if extra:
        raise ModelError(f"{rtype}: unknown fields {extra}")
    try:
        return cls(**d)
    except TypeError as e:                                   # missing required field
        raise ModelError(f"{rtype}: {e}") from e


# ------------------------------------------------------------------------------------------------ status machine

S = Status
LEGAL: dict[Status, frozenset[Status]] = {
    S.NOT_STARTED: frozenset({S.IN_PROGRESS, S.BLOCKED, S.REJECTED}),
    S.IN_PROGRESS: frozenset({S.IMPLEMENTED, S.FAILED, S.BLOCKED, S.SCIENTIFICALLY_LIMITED, S.REJECTED}),
    S.IMPLEMENTED: frozenset({S.TESTED, S.FAILED, S.IN_PROGRESS, S.ROLLED_BACK, S.REJECTED}),
    S.TESTED: frozenset({S.INTENDED_BEHAVIOR_VERIFIED, S.FAILED, S.IN_PROGRESS, S.ROLLED_BACK, S.REJECTED}),
    S.INTENDED_BEHAVIOR_VERIFIED: frozenset({S.VALIDATED, S.FAILED, S.IN_PROGRESS, S.ROLLED_BACK, S.REJECTED}),
    S.VALIDATED: frozenset({S.FAILED, S.ROLLED_BACK}),
    S.FAILED: frozenset({S.IN_PROGRESS, S.REJECTED, S.SCIENTIFICALLY_LIMITED, S.ROLLED_BACK, S.BLOCKED}),
    S.BLOCKED: frozenset({S.IN_PROGRESS, S.NOT_STARTED, S.REJECTED}),
    S.SCIENTIFICALLY_LIMITED: frozenset({S.IN_PROGRESS, S.REJECTED}),
    S.ROLLED_BACK: frozenset({S.IN_PROGRESS, S.REJECTED}),
    S.REJECTED: frozenset(),
}
TERMINAL = frozenset(s for s, nxt in LEGAL.items() if not nxt)
DONE_STATES = frozenset({S.TESTED, S.INTENDED_BEHAVIOR_VERIFIED, S.VALIDATED})
OPEN_STATES = frozenset({S.NOT_STARTED, S.IN_PROGRESS, S.IMPLEMENTED, S.BLOCKED, S.FAILED})


def reachable(src: Status, dst: Status) -> list[Status]:
    """Shortest legal path src -> dst (empty when unreachable); used to explain a refusal."""
    frontier, seen = [[src]], {src}
    while frontier:
        nxt = []
        for path in frontier:
            for s in sorted(LEGAL[path[-1]], key=lambda x: x.value):
                if s == dst:
                    return path + [s]
                if s not in seen:
                    seen.add(s)
                    nxt.append(path + [s])
        frontier = nxt
    return []


# ------------------------------------------------------------------------------------------------ the improvement rule

VERDICT_FUNCTION = "creator.model:improvement_verdict"


def _pool(ms: Sequence[Measurement]) -> tuple[float, float]:
    """Mean of replicate values and a CONSERVATIVE standard error. Replicates in a controlled comparison share one population (the
    same tasks), so they are not independent samples of it: re-running the same tasks does not shrink the uncertainty about new
    ones. The SE is therefore the mean replicate SE (fully correlated), floored by the between-replicate spread - never the
    sqrt(k)-shrunk independent-replicate SE, which let 100 reruns of 10 tasks make any difference 'significant' (30 Sep)."""
    k = len(ms)
    mean = sum(m.value for m in ms) / k
    se = sum(m.stderr for m in ms) / k
    if k > 1:
        sd = math.sqrt(sum((m.value - mean) ** 2 for m in ms) / (k - 1))
        se = max(se, sd / math.sqrt(k))
    return mean, se


def _diff(base: Sequence[Measurement], cand: Sequence[Measurement]) -> tuple[float, float]:
    """Signed improvement (positive = better in the metric's own direction) and its standard error."""
    (mb, sb), (mc, sc) = _pool(base), _pool(cand)
    sign = 1.0 if base[0].higher_is_better else -1.0
    return sign * (mc - mb), math.sqrt(sb ** 2 + sc ** 2)


def _uncontrolled(ms: Sequence[Measurement], split: Split) -> Optional[str]:
    """A comparison is controlled only when every measurement is the same metric, direction, population and conditions."""
    first = ms[0]
    for m in ms:
        for attr in ("metric", "population", "conditions", "higher_is_better"):
            if getattr(m, attr) != getattr(first, attr):
                return f"{attr} differs ({getattr(first, attr)!r} vs {getattr(m, attr)!r})"
        if m.split is not split:
            return f"{m.metric} measured on split {m.split.value}, expected {split.value}"
    return None


def improvement_verdict(baseline: Sequence[Measurement], candidate: Sequence[Measurement],
                        regressions: Sequence[tuple[Measurement, Measurement]] = (),
                        holdout: Optional[tuple[Measurement, Measurement]] = None,
                        min_effect: float = 0.0, z: float = 2.0) -> tuple[Verdict, dict[str, Any]]:
    """C77 sec 30: CHANGE + BASELINE + CONTROLLED COMPARISON + MEASURABLE BENEFIT + REGRESSION CHECK + REPRODUCIBILITY +
    HOLDOUT = IMPROVEMENT. Any missing ingredient is INSUFFICIENT_EVIDENCE, never a pass. A primary or holdout effect whose
    whole interval lies below -min_effect, or any guard metric significantly worse, is REGRESSION."""
    detail: dict[str, Any] = {"min_effect": min_effect, "z": z}

    def out(v: Verdict, why: str) -> tuple[Verdict, dict[str, Any]]:
        detail["why"] = why
        return v, detail

    if not baseline or not candidate:
        return out(Verdict.INSUFFICIENT_EVIDENCE, "no baseline" if not baseline else "no candidate")
    bad = _uncontrolled(list(baseline) + list(candidate), Split.DEV)
    if bad:
        return out(Verdict.INSUFFICIENT_EVIDENCE, f"uncontrolled comparison: {bad}")
    d, se = _diff(baseline, candidate)
    lo, hi = d - z * se, d + z * se
    detail.update(diff=d, se=se, lo=lo, hi=hi)
    if hi < -min_effect:
        return out(Verdict.REGRESSION, "primary metric significantly worse")
    for i, (rb, rc) in enumerate(regressions):
        why = _uncontrolled([rb, rc], rb.split)
        if why:
            return out(Verdict.INSUFFICIENT_EVIDENCE, f"regression pair {i} uncontrolled: {why}")
        rd, rse = _diff([rb], [rc])
        detail.setdefault("regressions", []).append({"metric": rb.metric, "diff": rd, "se": rse})
        if rd + z * rse < 0:
            return out(Verdict.REGRESSION, f"guard metric {rb.metric} significantly worse")
    if holdout is not None:
        hb, hc = holdout
        why = _uncontrolled([hb, hc], Split.HOLDOUT)
        if why:
            return out(Verdict.INSUFFICIENT_EVIDENCE, f"holdout uncontrolled: {why}")
        hd, hse = _diff([hb], [hc])
        detail.update(holdout_diff=hd, holdout_se=hse)
        if hd + z * hse < -min_effect:
            return out(Verdict.REGRESSION, "holdout significantly worse")
    if lo <= min_effect:
        if hi <= min_effect and lo >= -min_effect:
            return out(Verdict.NO_EFFECT, "interval excludes a meaningful effect either way")
        return out(Verdict.INSUFFICIENT_EVIDENCE, "benefit not established (interval reaches min_effect)")
    if len(candidate) < 2:
        return out(Verdict.INSUFFICIENT_EVIDENCE, "benefit not reproduced: one candidate measurement")
    if any(_diff(baseline, [c])[0] <= 0 for c in candidate):
        return out(Verdict.INSUFFICIENT_EVIDENCE, "benefit not reproduced: a replicate shows no gain")
    if not regressions:
        return out(Verdict.INSUFFICIENT_EVIDENCE, "no regression check")
    if holdout is None:
        return out(Verdict.INSUFFICIENT_EVIDENCE, "no holdout")
    if detail["holdout_diff"] <= 0:
        return out(Verdict.INSUFFICIENT_EVIDENCE, "holdout does not confirm the gain")
    return out(Verdict.IMPROVEMENT, "all seven ingredients present")
