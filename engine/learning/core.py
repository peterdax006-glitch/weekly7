"""Shared vocabulary of the learning brain (contract C62 sections 5, 6, 9, 12-14, 42-46, 59, 68).

Every learning module imports its states, reasons and identifiers from here so that parallel builders speak one language.
Enums are str-valued so they serialise to JSON unchanged. Records are frozen dataclasses: history is immutable (section 49);
behaviour changes by writing a NEW record or a new version, never by mutating an old one."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


class _StrEnum(str, enum.Enum):
    def __str__(self):
        return self.value

    @classmethod
    def parse(cls, v):
        return v if isinstance(v, cls) else cls(str(v))


class Epistemic(_StrEnum):                 # section 6: what we believe about a learned item
    OBSERVED = "OBSERVED"
    HYPOTHESIS = "HYPOTHESIS"
    SUPPORTED = "SUPPORTED"
    CONDITIONAL = "CONDITIONAL"
    DEGRADED = "DEGRADED"
    CONTRADICTED = "CONTRADICTED"
    GATED = "GATED"
    RETIRED = "RETIRED"
    UNKNOWN = "UNKNOWN"


class Lifecycle(_StrEnum):                 # sections 12-13: where the item is in its life; RETIRED is never deletion
    BIRTH = "BIRTH"
    GROWTH = "GROWTH"
    PEAK = "PEAK"
    ACTIVE = "ACTIVE"
    DECAY = "DECAY"
    DEGRADED = "DEGRADED"
    FAILURE = "FAILURE"
    DORMANT = "DORMANT"
    RECOVERY = "RECOVERY"
    RETIRED = "RETIRED"


class Promotion(_StrEnum):                 # section 44: champion / challenger knowledge
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"
    RETIRED = "RETIRED"


class Health(_StrEnum):                    # section 46
    HEALTHY = "HEALTHY"
    DEGRADING = "DEGRADING"
    BROKEN = "BROKEN"
    CONTRADICTED = "CONTRADICTED"
    DORMANT = "DORMANT"
    RECOVERING = "RECOVERING"
    UNSTABLE = "UNSTABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


class Unknown(_StrEnum):                   # section 42: states that must never become false confidence
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CONFLICTED = "CONFLICTED"
    UNTESTED = "UNTESTED"


class UnknownAction(_StrEnum):
    ABSTAIN = "ABSTAIN"
    COLLECT_DATA = "COLLECT_DATA"
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    USE_GENERAL_RULE = "USE_GENERAL_RULE"


class FailureCause(_StrEnum):              # section 9; UNKNOWN is a correct answer, not a gap
    FALSE_PATTERN = "FALSE_PATTERN"
    TEMPORARY_INACTIVITY = "TEMPORARY_INACTIVITY"
    WRONG_CONTEXT = "WRONG_CONTEXT"
    REGIME_CHANGE = "REGIME_CHANGE"
    WEAKENING_EFFECT = "WEAKENING_EFFECT"
    REVERSAL = "REVERSAL"
    MEASUREMENT_ERROR = "MEASUREMENT_ERROR"
    REDUNDANCY = "REDUNDANCY"
    SELECTION_ERROR = "SELECTION_ERROR"
    TIMING_ERROR = "TIMING_ERROR"
    RISK_ERROR = "RISK_ERROR"
    INTERACTION_FAILURE = "INTERACTION_FAILURE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


class Subsystem(_StrEnum):                 # section 23: a failure must teach the subsystem that made it
    SELECTION = "SELECTION"
    TIMING = "TIMING"
    DIRECTION = "DIRECTION"
    RISK = "RISK"
    EXIT = "EXIT"


class TemporalClass(_StrEnum):             # section 14
    PERSISTENT = "PERSISTENT"
    SLOW_DECAY = "SLOW_DECAY"
    FAST_DECAY = "FAST_DECAY"
    EPISODIC = "EPISODIC"
    REGIME_BOUND = "REGIME_BOUND"
    EVENT_BOUND = "EVENT_BOUND"
    SEASONAL = "SEASONAL"
    UNKNOWN = "UNKNOWN"


class DecisionEffect(_StrEnum):            # section 43: what decision a knowledge item changes
    RANKING = "RANKING"
    SELECTION = "SELECTION"
    POSITION_SIZE = "POSITION_SIZE"
    DIRECTION = "DIRECTION"
    TIMING = "TIMING"
    EXIT = "EXIT"
    STOP = "STOP"
    ABSTENTION = "ABSTENTION"
    RESEARCH_PRIORITY = "RESEARCH_PRIORITY"
    PATTERN_WEIGHTING = "PATTERN_WEIGHTING"
    CONFIDENCE = "CONFIDENCE"
    NONE = "NONE"                          # research knowledge only; may not reach production


class Edge(_StrEnum):                      # section 18
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTAINS = "CONTAINS"
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    CAUSES_FAILURE_OF = "CAUSES_FAILURE_OF"
    RECOVERS_WITH = "RECOVERS_WITH"
    REDUNDANT_WITH = "REDUNDANT_WITH"
    COMPLEMENTS = "COMPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"


class Layer(_StrEnum):                     # section 49: archive layers
    L0_RAW = "L0_RAW"
    L1_EVENT = "L1_EVENT"
    L2_EPISODE = "L2_EPISODE"
    L3_SITUATION = "L3_SITUATION"
    L4_HYPOTHESIS = "L4_HYPOTHESIS"
    L5_PATTERN = "L5_PATTERN"
    L6_CONTEXT_RULE = "L6_CONTEXT_RULE"
    L7_VALIDATED = "L7_VALIDATED"
    L8_POLICY = "L8_POLICY"
    L9_META = "L9_META"


class BuildStatus(_StrEnum):               # section 68 (no DONE); section 59 honest labels via ValidationLabel
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    IMPLEMENTED = "IMPLEMENTED"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ValidationLabel(_StrEnum):
    NOT_VALIDATED = "IMPLEMENTED — NOT VALIDATED"
    FAILED_VALIDATION = "IMPLEMENTED — FAILED VALIDATION"
    INSUFFICIENT_EVIDENCE = "IMPLEMENTED — INSUFFICIENT EVIDENCE"
    VALIDATED = "VALIDATED"


class FirewallBreach(Exception):
    """Raised by any firewall (sections 28-30, 55). Callers must not catch-and-continue: fail closed."""


# ---------------------------------------------------------------- canonical hashing (A12)

def _canon(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canon(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_canon(v) for v in obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)                               # JSON has no NaN; keep it visible, never silently 0
        return round(float(obj), 12)                      # float() first: np.float64 repr is 'np.float64(..)' in numpy 2
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, "item") and callable(obj.item):       # numpy scalars
        return _canon(obj.item())
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, n: int = 16) -> str:
    """Deterministic content hash: same content -> same id on any machine and any run."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------- time (C56/C58: nothing after `now`)

def as_date(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    if hasattr(x, "date") and callable(x.date):          # pandas Timestamp
        return x.date()
    return dt.date.fromisoformat(str(x)[:10])


def require_past(when, now, what: str = "record") -> None:
    """Fail closed if `when` is not strictly before `now` (an outcome that matures ON `now` is not yet known)."""
    if when is None or now is None:
        raise FirewallBreach(f"{what}: missing timestamp (when={when}, now={now})")
    if as_date(when) >= as_date(now):
        raise FirewallBreach(f"{what}: dated {when} is not strictly before now={now}")


# ---------------------------------------------------------------- provenance (A05, A13, section 28)

@dataclasses.dataclass(frozen=True)
class Provenance:
    """Where a learned item came from. `learned_at` is the REAL (undisguised, trusted-side) date the evidence matured;
    `sealed_windows` lists evaluation windows that were sealed when it was created. A future-memory audit asks:
    could this item have existed at the decision timestamp?  -> learned_at < now and code/data were available."""
    created_real: str                      # wall-clock ISO time the record was written
    learned_at: str                        # real market date the newest evidence matured (trusted side only)
    code_hash: str
    data_hash: str = ""
    config_hash: str = ""
    experiment_id: str = ""
    run_id: str = ""
    seed: int | None = None
    outcomes_seen_through: str = ""        # real date of the newest outcome this item has ever seen
    sealed_windows: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()          # ids of records it was derived from

    def could_exist_at(self, now) -> bool:
        seen = self.outcomes_seen_through or self.learned_at
        return as_date(self.learned_at) < as_date(now) and as_date(seen) < as_date(now)

    def check(self) -> list[str]:
        errs = []
        for f in ("created_real", "learned_at", "code_hash"):
            if not getattr(self, f):
                errs.append(f"provenance.{f} missing")
        if self.outcomes_seen_through and self.learned_at and as_date(self.outcomes_seen_through) < as_date(self.learned_at):
            errs.append("provenance: outcomes_seen_through precedes learned_at")
        return errs


def current_code_hash() -> str:
    """The engine's code identity: a hash of every engine source file on disk (engine.provenance.engine_tree_hash). It does not
    depend on which modules happen to be loaded (C75 Phase 0: it used to, and identical results disagreed); falls back to hashing
    this package if provenance is unavailable."""
    try:
        from engine import provenance
        return provenance.engine_tree_hash()
    except Exception:
        import pathlib
        h = hashlib.sha256()
        for p in sorted(pathlib.Path(__file__).parent.glob("*.py")):
            h.update(p.read_bytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------- confidence dimensions (A08/A09, section 11, 33)

@dataclasses.dataclass(frozen=True)
class Confidence:
    """Separate dimensions that must never collapse into one number (sections 11 and 33). All in [0, 1];
    None = never measured (UNTESTED), which is different from 0."""
    truth: float | None = None             # IS IT REAL?
    usefulness: float | None = None        # IS IT USEFUL? (incremental decision value)
    current_reliability: float | None = None   # IS IT USEFUL NOW?
    context: float | None = None
    transfer: float | None = None
    failure_risk: float | None = None      # higher = worse

    def check(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is not None and not (isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0):
                errs.append(f"confidence.{f.name}={v!r} outside [0,1]")
        return errs

    def untested(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self) if getattr(self, f.name) is None)


class KnowledgeLike:
    """Duck-typed view every module may rely on, so modules built in parallel do not import each other's classes.
    engine.learning.knowledge.KnowledgeObject satisfies it. Anything with these attributes works."""
    knowledge_id: str
    version: int
    epistemic: Epistemic
    lifecycle: Lifecycle
    promotion: Promotion
    confidence: Confidence
    provenance: Provenance
    contexts: Mapping[str, Any]            # context dimension -> condition (see section 8)
    anti_contexts: Mapping[str, Any]
    decision_effect: tuple[DecisionEffect, ...]

    REQUIRED = ("knowledge_id", "version", "epistemic", "lifecycle", "promotion", "confidence", "provenance",
                "contexts", "anti_contexts", "decision_effect")

    @classmethod
    def conforms(cls, obj) -> list[str]:
        return [f"missing {a}" for a in cls.REQUIRED if not hasattr(obj, a)]


def clip01(x) -> float:
    x = float(x)
    if math.isnan(x):
        raise ValueError("NaN confidence")
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ids(items: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(getattr(i, "knowledge_id", i)) for i in items}))
