"""Contract C62 section 5 (A01, A05, A06, A10-A13): the durable, versioned KnowledgeObject.

Explicit frozen types, never one untyped dict: Condition/ContextSet (section 8 dimensions), Effect, Evidence, Dynamics,
TransferScores, Relations, FailureExplanation, ActionPolicy, plus the shared Confidence/Provenance/Epistemic vocabulary from
engine.learning.core. History is immutable (section 49): a change is a NEW version linked to its parent by hash and stored
append-only in `KnowledgeStore`; an old record is never edited and retiring never deletes.

Serialisation is a strict typed codec (`encode`/`decode`): unknown keys, wrong types and missing required fields fail closed.
Hashing is deterministic (core.stable_hash) so the same content has the same id on any machine.

Reuses: core (vocabulary, hashing, Provenance), epistemic (scoped states), and adapts PatternMiner rows
(`from_pattern_row`) rather than duplicating pattern statistics.
Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
import os
import types
import typing
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable

from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle,
                                  Promotion, Provenance, Subsystem, TemporalClass, as_date, canonical_json,
                                  current_code_hash, stable_hash)
from engine.learning.epistemic import EpistemicProfile

SCHEMA_VERSION = 1
DIMENSIONS = ("market", "sector", "stock", "stock_type", "volatility", "liquidity", "regime", "time")   # section 8
OPS = ("eq", "ne", "in", "not_in", "lt", "le", "gt", "ge", "between")


class SchemaError(ValueError):
    """A record failed schema validation or could not be decoded."""


class ChainError(ValueError):
    """The version chain of a knowledge item is broken or was written out of order."""


# ---------------------------------------------------------------- typed codec (A10)

def encode(obj: Any) -> Any:
    """Plain JSON-able structure with sorted keys; enums as their values, NaN kept visible as a string."""
    return json.loads(canonical_json(obj))


def decode(tp: Any, v: Any, path: str = "$") -> Any:
    """Inverse of `encode` driven by the dataclass type hints. Strict: extra keys and wrong shapes raise SchemaError."""
    if v is None:
        return None
    org = typing.get_origin(tp)
    try:
        if org in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            return decode(args[0], v, path)
        if org is tuple:
            args = typing.get_args(tp)
            if not isinstance(v, (list, tuple)):
                raise SchemaError(f"{path}: expected a list")
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(decode(args[0], x, f"{path}[{i}]") for i, x in enumerate(v))
            if len(args) != len(v):
                raise SchemaError(f"{path}: expected {len(args)} items, got {len(v)}")
            return tuple(decode(a, x, f"{path}[{i}]") for i, (a, x) in enumerate(zip(args, v)))
        if org is dict:
            kt, vt = typing.get_args(tp)
            return {str(k): decode(vt, x, f"{path}.{k}") for k, x in v.items()}
        if tp is bool:
            if not isinstance(v, bool):
                raise SchemaError(f"{path}: expected bool")
            return v
        if tp is float:
            return float(v)                      # accepts the "nan"/"inf" strings encode() emits
        if tp is int:
            if isinstance(v, bool) or int(v) != v:
                raise SchemaError(f"{path}: expected int")
            return int(v)
        if tp is str:
            if not isinstance(v, str):
                raise SchemaError(f"{path}: expected str")
            return v
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            return tp(v)
        if isinstance(tp, type) and dataclasses.is_dataclass(tp):
            if not isinstance(v, Mapping):
                raise SchemaError(f"{path}: expected an object for {tp.__name__}")
            hints = typing.get_type_hints(tp)
            names = {f.name for f in dataclasses.fields(tp) if f.init}
            extra = set(v) - names
            if extra:
                raise SchemaError(f"{path}: unknown keys {sorted(extra)} for {tp.__name__}")
            return tp(**{k: decode(hints[k], x, f"{path}.{k}") for k, x in v.items()})
    except SchemaError:
        raise
    except (TypeError, ValueError, KeyError) as e:
        raise SchemaError(f"{path}: {e}") from e
    return v


# ---------------------------------------------------------------- context (section 8 dimensions)

@dataclasses.dataclass(frozen=True)
class Condition:
    """One testable condition on one named feature of a situation, grouped under a section-8 context dimension."""
    dimension: str
    feature: str
    op: str
    nums: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()

    def check(self) -> list[str]:
        errs = []
        if self.dimension not in DIMENSIONS:
            errs.append(f"unknown context dimension {self.dimension!r}")
        if not self.feature:
            errs.append("condition without a feature")
        if self.op not in OPS:
            errs.append(f"unknown op {self.op!r}")
            return errs
        if any(math.isnan(x) or math.isinf(x) for x in self.nums):
            errs.append("non-finite bound")
        if self.op in ("eq", "ne", "in", "not_in"):
            if not (self.nums or self.labels):
                errs.append(f"{self.op} needs values")
            if self.nums and self.labels:
                errs.append("use numbers or labels, not both")
            if self.op in ("eq", "ne") and len(self.nums) + len(self.labels) != 1:
                errs.append(f"{self.op} takes exactly one value")
        elif self.op == "between":
            if len(self.nums) != 2 or self.labels:
                errs.append("between needs exactly two numbers")
            elif self.nums[0] > self.nums[1]:
                errs.append("between bounds reversed")
        elif len(self.nums) != 1 or self.labels:
            errs.append(f"{self.op} needs exactly one number")
        return errs

    def test(self, x: Any) -> bool | None:
        """True/False, or None when the situation does not carry the feature (an unknown is never a match)."""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if self.op in ("eq", "in", "ne", "not_in"):
            vals = self.labels if self.labels else self.nums
            hit = (str(x) in vals) if self.labels else (isinstance(x, (int, float)) and any(abs(float(x) - n) < 1e-12 for n in vals))
            return hit if self.op in ("eq", "in") else not hit
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if self.op == "between":
            return self.nums[0] <= v <= self.nums[1]
        n = self.nums[0]
        return {"lt": v < n, "le": v <= n, "gt": v > n, "ge": v >= n}[self.op]

    def extent(self) -> tuple[str, Any]:
        """('set', frozenset, negated) or ('range', lo, hi): the region of feature space this condition accepts."""
        if self.op in ("eq", "in"):
            return ("set", frozenset(self.labels or self.nums), False)
        if self.op in ("ne", "not_in"):
            return ("set", frozenset(self.labels or self.nums), True)
        if self.op == "between":
            return ("range", self.nums[0], self.nums[1])
        n = self.nums[0]
        inf = math.inf
        return {"lt": ("range", -inf, n), "le": ("range", -inf, n), "gt": ("range", n, inf), "ge": ("range", n, inf)}[self.op]

    def implies(self, other: "Condition") -> bool:
        """Every value accepted by self is accepted by other (same feature only)."""
        if self.feature != other.feature:
            return False
        a, b = self.extent(), other.extent()
        if a[0] == "range" and b[0] == "range":
            return a[1] >= b[1] and a[2] <= b[2]
        if a[0] == "set" and b[0] == "set":
            if not a[2] and not b[2]:
                return a[1] <= b[1]
            if not a[2] and b[2]:
                return not (a[1] & b[1])
            if a[2] and b[2]:
                return b[1] <= a[1]
            return False
        if a[0] == "set" and not a[2] and b[0] == "range":
            return all(isinstance(x, (int, float)) and b[1] <= x <= b[2] for x in a[1])
        return False

    def describe(self) -> str:
        vals = list(self.labels or self.nums)
        return f"{self.feature} {self.op} {vals[0] if len(vals) == 1 else vals}"


class Applicability(str, enum.Enum):
    APPLIES = "APPLIES"
    OUT_OF_CONTEXT = "OUT_OF_CONTEXT"        # a required condition is false
    EXCLUDED = "EXCLUDED"                    # an anti-context condition set is fully true
    UNKNOWN = "UNKNOWN"                      # the situation lacks a feature we need: never treated as APPLIES


@dataclasses.dataclass(frozen=True)
class ContextSet:
    """A conjunction of conditions. Behaves as a read-only Mapping dimension -> tuple[Condition] (KnowledgeLike)."""
    conditions: tuple[Condition, ...] = ()
    any_of: bool = False              # False: all conditions must hold (contexts). True: any one suffices (anti-contexts)

    def __getitem__(self, dim: str) -> tuple[Condition, ...]:
        got = tuple(c for c in self.conditions if c.dimension == dim)
        if not got:
            raise KeyError(dim)
        return got

    def __iter__(self):
        seen = []
        for c in self.conditions:
            if c.dimension not in seen:
                seen.append(c.dimension)
        return iter(seen)

    def __len__(self):
        return len({c.dimension for c in self.conditions})

    def __contains__(self, dim):
        return any(c.dimension == dim for c in self.conditions)

    def keys(self):
        return list(iter(self))

    def items(self):
        return [(d, self[d]) for d in self]

    def check(self) -> list[str]:
        errs = [f"{c.describe()}: {e}" for c in self.conditions for e in c.check()]
        seen = set()
        for c in self.conditions:
            if c in seen:
                errs.append(f"duplicate condition {c.describe()}")
            seen.add(c)
        return errs

    def evaluate(self, situation: Mapping[str, Any]) -> bool | None:
        """all_of: False if any is False; None if none is False but some are unknown; else True. An empty set is
        vacuously True (no restriction). any_of mirrors it: True if any is True; None if none is True but some unknown."""
        results = [c.test(situation.get(c.feature)) for c in self.conditions]
        if self.any_of:
            if not results:
                return False                     # an empty anti-context excludes nothing
            return True if any(r is True for r in results) else (None if any(r is None for r in results) else False)
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True

    def implies(self, other: "ContextSet") -> bool:
        """self is at least as narrow as other: every condition of other is implied by some condition of self."""
        return all(any(s.implies(o) for s in self.conditions) for o in other.conditions)

    def excluded_by(self, anti: "ContextSet") -> bool:
        """True if this (all_of) context lies entirely inside the region `anti` excludes."""
        if anti.any_of:
            return any(s.implies(a) for a in anti.conditions for s in self.conditions)
        return self.implies(anti)

    def features(self) -> tuple[str, ...]:
        return tuple(sorted({c.feature for c in self.conditions}))


Mapping.register(ContextSet)


def applicability(contexts: ContextSet, anti: ContextSet, situation: Mapping[str, Any]) -> Applicability:
    """Section 8: contexts say where it works, anti-contexts say where it must not be used. Unknown fails closed."""
    if anti.conditions:
        a = anti.evaluate(situation)
        if a is True:
            return Applicability.EXCLUDED
    c = contexts.evaluate(situation)
    if c is False:
        return Applicability.OUT_OF_CONTEXT
    if c is None or (anti.conditions and anti.evaluate(situation) is None):
        return Applicability.UNKNOWN
    return Applicability.APPLIES


# ---------------------------------------------------------------- typed sub-records

def _unit(name: str, v: float | None, errs: list[str]) -> None:
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or not 0.0 <= v <= 1.0):
        errs.append(f"{name}={v!r} outside [0,1]")


@dataclasses.dataclass(frozen=True)
class Effect:
    """What the item does to the outcome. size is a magnitude (>= 0) in `unit`; the sign lives only in `direction`."""
    direction: int = 0                 # +1 raises the outcome, -1 lowers it, 0 = no claimed effect
    size: float = 0.0
    uncertainty: float | None = None   # standard error of size; None = never estimated
    unit: str = "fwd_return"
    horizon_days: int = 5

    @property
    def signed(self) -> float:
        return self.direction * self.size

    def interval(self, z: float = 1.96) -> tuple[float, float] | None:
        if self.uncertainty is None:
            return None
        return (self.signed - z * self.uncertainty, self.signed + z * self.uncertainty)

    def excludes_zero(self, z: float = 1.96) -> bool | None:
        iv = self.interval(z)
        return None if iv is None else (iv[0] > 0 or iv[1] < 0)

    def check(self) -> list[str]:
        errs = []
        if self.direction not in (-1, 0, 1):
            errs.append("effect.direction must be -1, 0 or +1")
        if not isinstance(self.size, (int, float)) or math.isnan(self.size) or self.size < 0:
            errs.append("effect.size must be a finite magnitude >= 0")
        elif self.direction == 0 and self.size != 0:
            errs.append("effect.direction 0 (no effect) with size > 0")
        elif self.direction != 0 and self.size == 0:
            errs.append("effect has a direction but zero size")
        if self.uncertainty is not None and (math.isnan(self.uncertainty) or self.uncertainty < 0):
            errs.append("effect.uncertainty must be >= 0")
        if self.horizon_days < 1:
            errs.append("effect.horizon_days must be >= 1")
        return errs


@dataclasses.dataclass(frozen=True)
class Evidence:
    sample_size: int = 0
    effective_sample_size: float = 0.0
    recency: float | None = None        # 0 = stale evidence, 1 = evidence from the last window
    modernity: float | None = None      # share of evidence from the modern market structure
    last_evidence_at: str = ""

    def check(self) -> list[str]:
        errs = []
        if self.sample_size < 0 or self.effective_sample_size < 0 or math.isnan(self.effective_sample_size):
            errs.append("sample sizes must be >= 0")
        if self.effective_sample_size > self.sample_size + 1e-9:
            errs.append("effective_sample_size exceeds sample_size")
        _unit("recency", self.recency, errs)
        _unit("modernity", self.modernity, errs)
        if self.last_evidence_at:
            try:
                as_date(self.last_evidence_at)
            except ValueError:
                errs.append("last_evidence_at is not an ISO date")
        return errs


@dataclasses.dataclass(frozen=True)
class Dynamics:
    stability: float | None = None
    contradiction_rate: float | None = None
    failure_rate: float | None = None
    recovery_rate: float | None = None

    def check(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            _unit(f.name, getattr(self, f.name), errs)
        return errs


@dataclasses.dataclass(frozen=True)
class TransferScores:
    transfer_score: float | None = None
    cross_era: float | None = None
    cross_stock: float | None = None
    cross_sector: float | None = None
    cross_regime: float | None = None

    def check(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            _unit(f.name, getattr(self, f.name), errs)
        return errs

    def untested(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self) if getattr(self, f.name) is None)


@dataclasses.dataclass(frozen=True)
class Relations:
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    parent: tuple[str, ...] = ()
    child: tuple[str, ...] = ()
    complementary: tuple[str, ...] = ()
    redundant: tuple[str, ...] = ()

    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i for f in dataclasses.fields(self) for i in getattr(self, f.name)}))

    def check(self, own_id: str = "") -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            vals = getattr(self, f.name)
            if len(set(vals)) != len(vals):
                errs.append(f"relations.{f.name} has duplicates")
            if own_id and own_id in vals:
                errs.append(f"relations.{f.name} references itself")
        if set(self.supporting) & set(self.contradicting):
            errs.append("an item cannot both support and contradict the same knowledge")
        if set(self.parent) & set(self.child):
            errs.append("an item cannot be both parent and child of the same knowledge")
        if set(self.complementary) & set(self.redundant):
            errs.append("complementary and redundant are mutually exclusive")
        return errs


@dataclasses.dataclass(frozen=True)
class FailureExplanation:
    cause: FailureCause
    at: str
    subsystem: Subsystem | None = None
    note: str = ""
    evidence_id: str = ""

    def check(self) -> list[str]:
        errs = []
        try:
            as_date(self.at)
        except ValueError:
            errs.append("failure explanation needs an ISO date")
        if self.cause != FailureCause.UNKNOWN and not (self.note or self.evidence_id):
            errs.append(f"cause {self.cause.value} asserted without a note or evidence id (UNKNOWN is the honest alternative)")
        return errs


@dataclasses.dataclass(frozen=True)
class ActionPolicy:
    """How the item may touch a decision (section 43); enforced by decision_contract."""
    weight_cap: float = 1.0
    min_reliability: float = 0.5
    abstain_when_unknown: bool = True
    size_multiplier_max: float | None = None
    note: str = ""

    def check(self) -> list[str]:
        errs = []
        if not 0.0 < self.weight_cap <= 1.0:
            errs.append("action_policy.weight_cap must be in (0,1]")
        _unit("action_policy.min_reliability", self.min_reliability, errs)
        if self.size_multiplier_max is not None and not 0.0 < self.size_multiplier_max <= 2.0:
            errs.append("action_policy.size_multiplier_max must be in (0,2]")
        return errs


# ---------------------------------------------------------------- the object

@dataclasses.dataclass(frozen=True)
class KnowledgeObject:
    knowledge_id: str
    created_at: str
    provenance: Provenance
    version: int = 1
    updated_at: str = ""
    parent_hash: str = ""
    version_reason: str = "initial"
    source_experiences: tuple[str, ...] = ()
    source_experiments: tuple[str, ...] = ()
    source_runs: tuple[str, ...] = ()
    observation: str = ""
    interpretation: str = ""
    hypothesis: str = ""
    interpretation_ref: str = ""                 # record_hash of the full interpretation.InterpretationRecord
    contexts: ContextSet = ContextSet()
    anti_contexts: ContextSet = ContextSet()
    effect: Effect = Effect()
    confidence: Confidence = Confidence()
    evidence: Evidence = Evidence()
    dynamics: Dynamics = Dynamics()
    transfer: TransferScores = TransferScores()
    temporal_class: TemporalClass = TemporalClass.UNKNOWN
    mechanism_tags: tuple[str, ...] = ()
    relations: Relations = Relations()
    failure_explanations: tuple[FailureExplanation, ...] = ()
    decision_effect: tuple[DecisionEffect, ...] = (DecisionEffect.NONE,)
    action_policy: ActionPolicy = ActionPolicy()
    epistemic: Epistemic = Epistemic.HYPOTHESIS
    epistemic_profile: EpistemicProfile = EpistemicProfile()
    lifecycle: Lifecycle = Lifecycle.BIRTH
    promotion: Promotion = Promotion.RESEARCH

    def __post_init__(self):
        if not self.updated_at:
            object.__setattr__(self, "updated_at", self.created_at)

    # -- contract field names (section 5) as read-only views
    truth_confidence = property(lambda s: s.confidence.truth)
    current_reliability = property(lambda s: s.confidence.current_reliability)
    context_confidence = property(lambda s: s.confidence.context)
    transfer_confidence = property(lambda s: s.confidence.transfer)
    failure_risk = property(lambda s: s.confidence.failure_risk)
    sample_size = property(lambda s: s.evidence.sample_size)
    effective_sample_size = property(lambda s: s.evidence.effective_sample_size)
    experiment_id = property(lambda s: s.provenance.experiment_id)
    code_hash = property(lambda s: s.provenance.code_hash)
    data_hash = property(lambda s: s.provenance.data_hash)
    effect_direction = property(lambda s: s.effect.direction)
    effect_size = property(lambda s: s.effect.size)
    effect_uncertainty = property(lambda s: s.effect.uncertainty)
    promotion_state = property(lambda s: s.promotion)
    lifecycle_state = property(lambda s: s.lifecycle)

    # -- validation (A11)
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.knowledge_id or not self.knowledge_id.strip():
            errs.append("knowledge_id empty")
        if self.version < 1:
            errs.append("version must be >= 1")
        if (self.version == 1) != (self.parent_hash == ""):
            errs.append("version 1 has no parent_hash and every later version must have one")
        try:
            c, u = as_date(self.created_at), as_date(self.updated_at)
            if u < c:
                errs.append("updated_at precedes created_at")
        except ValueError:
            errs.append("created_at/updated_at must be ISO dates")
        errs += self.provenance.check()
        for part in (self.contexts, self.anti_contexts, self.effect, self.evidence, self.dynamics, self.transfer,
                     self.confidence, self.action_policy):
            errs += part.check()
        errs += self.relations.check(self.knowledge_id)
        for f in self.failure_explanations:
            errs += f.check()
        if self.contexts.conditions and self.anti_contexts.conditions and self.contexts.excluded_by(self.anti_contexts):
            errs.append("anti_contexts swallow the whole context: the item would exclude itself everywhere")
        if self.contexts.any_of:
            errs.append("contexts must be all_of (a conjunction); any_of is for anti_contexts")
        de = set(self.decision_effect)
        if not de:
            errs.append("decision_effect empty: use (NONE,) for research-only knowledge")
        if DecisionEffect.NONE in de and len(de) > 1:
            errs.append("decision_effect NONE cannot be combined with a real decision")
        if len(de) != len(self.decision_effect):
            errs.append("decision_effect has duplicates")
        if self.epistemic == Epistemic.UNKNOWN and self.confidence.truth is not None:
            errs.append("epistemic UNKNOWN must not carry truth confidence (unknown is not confidence)")
        if not self.epistemic_profile.is_empty():
            if self.epistemic_profile.headline() != self.epistemic:
                errs.append(f"epistemic {self.epistemic.value} disagrees with profile headline "
                            f"{self.epistemic_profile.headline().value}")
            errs += [f"profile: {e}" for e in self.epistemic_profile.coherence() + self.epistemic_profile.verify_history()]
        retired = (self.epistemic == Epistemic.RETIRED, self.lifecycle == Lifecycle.RETIRED, self.promotion == Promotion.RETIRED)
        if any(retired) and not all(retired):
            errs.append("RETIRED must be set consistently on epistemic, lifecycle and promotion")
        if self.promotion == Promotion.CHAMPION:
            if self.epistemic not in (Epistemic.SUPPORTED, Epistemic.CONDITIONAL, Epistemic.DEGRADED):
                errs.append("CHAMPION knowledge must be SUPPORTED, CONDITIONAL or DEGRADED (reduced weight until demoted)")
            if de <= {DecisionEffect.NONE}:
                errs.append("CHAMPION knowledge must change a decision")
        if self.effect.direction != 0 and self.epistemic == Epistemic.OBSERVED:
            errs.append("OBSERVED makes no claim about an effect; use HYPOTHESIS")
        if len(set(self.mechanism_tags)) != len(self.mechanism_tags) or any(not t for t in self.mechanism_tags):
            errs.append("mechanism_tags must be unique and non-empty")
        return errs

    def assert_valid(self) -> "KnowledgeObject":
        errs = self.validate()
        if errs:
            raise SchemaError(f"{self.knowledge_id} v{self.version}: " + "; ".join(errs))
        return self

    # -- serialisation (A10)
    def to_dict(self) -> dict:
        return {"schema": SCHEMA_VERSION, "record": encode(self)}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping) -> "KnowledgeObject":
        if not isinstance(d, Mapping) or d.get("schema") != SCHEMA_VERSION or "record" not in d:
            raise SchemaError(f"unsupported knowledge schema {d.get('schema') if isinstance(d, Mapping) else None!r}")
        return decode(cls, d["record"]).assert_valid()

    @classmethod
    def from_json(cls, s: str) -> "KnowledgeObject":
        try:
            return cls.from_dict(json.loads(s))
        except json.JSONDecodeError as e:
            raise SchemaError(f"invalid JSON: {e}") from e

    # -- hashing (A12)
    def record_hash(self) -> str:
        """Identity of this exact version, including its place in the chain and when it was written."""
        return stable_hash(self, 24)

    def content_hash(self) -> str:
        """Identity of the knowledge itself: ignores its id, version metadata and wall-clock, so the same lesson derived twice
        (even under two ids) gives the same hash (used to detect duplicate lessons and unchanged re-versions)."""
        d = encode(self)
        for k in ("knowledge_id", "created_at", "version", "parent_hash", "version_reason", "updated_at"):
            d.pop(k)
        d["provenance"].pop("created_real", None)
        d["provenance"].pop("parents", None)
        return stable_hash(d, 24)

    # -- time safety (sections 28-30)
    def visible_at(self, now) -> bool:
        return as_date(self.updated_at) < as_date(now) and self.provenance.could_exist_at(now)

    def assert_visible(self, now) -> None:
        if not self.visible_at(now):
            raise FirewallBreach(f"{self.knowledge_id} v{self.version} (updated {self.updated_at}, learned "
                                 f"{self.provenance.learned_at}) did not exist at now={now}")

    # -- versioning (A06)
    _FROZEN_FIELDS = ("knowledge_id", "created_at", "version", "parent_hash", "updated_at", "version_reason")

    def new_version(self, now, reason: str, learned_at: str | None = None, **changes) -> "KnowledgeObject":
        """A NEW immutable record linked to this one; this record is never touched."""
        bad = [k for k in changes if k in self._FROZEN_FIELDS]
        if bad:
            raise SchemaError(f"fields {bad} are managed by the versioning and cannot be changed")
        unknown = [k for k in changes if k not in {f.name for f in dataclasses.fields(self)}]
        if unknown:
            raise SchemaError(f"unknown fields {unknown}")
        if not reason.strip():
            raise SchemaError("a new version needs a reason")
        if as_date(now) < as_date(self.updated_at):
            raise FirewallBreach(f"new version dated {now} precedes its parent's update {self.updated_at}")
        prov = changes.pop("provenance", self.provenance)
        if learned_at is not None:
            if as_date(learned_at) < as_date(prov.learned_at):
                raise FirewallBreach("learned_at cannot move backwards")
            prov = dataclasses.replace(prov, learned_at=learned_at)
        me = f"{self.knowledge_id}@v{self.version}"
        prov = dataclasses.replace(prov, parents=tuple(dict.fromkeys(prov.parents + (me,))))
        nxt = dataclasses.replace(self, version=self.version + 1, updated_at=str(as_date(now)), parent_hash=self.record_hash(),
                                  version_reason=reason, provenance=prov, **changes)
        if nxt.content_hash() == self.content_hash():
            raise SchemaError("no-op version: nothing changed")
        return nxt.assert_valid()

    def retire(self, now, reason: str, cause: FailureCause = FailureCause.UNKNOWN) -> "KnowledgeObject":
        """Retire is a new version, never a deletion; the explanation stays on the record (section 13)."""
        note = FailureExplanation(cause, str(as_date(now)), None, reason)
        return self.new_version(now, f"retired: {reason}", epistemic=Epistemic.RETIRED, lifecycle=Lifecycle.RETIRED,
                                promotion=Promotion.RETIRED, failure_explanations=self.failure_explanations + (note,),
                                epistemic_profile=EpistemicProfile())

    def applicability(self, situation: Mapping[str, Any]) -> Applicability:
        return applicability(self.contexts, self.anti_contexts, situation)


# ---------------------------------------------------------------- provenance (A05, A13)

def file_hash(path: str | os.PathLike, n: int = 16) -> str:
    """Byte hash of a file: engine.repro.file_hash (the repo's one implementation), shortened."""
    from engine import repro
    return repro.file_hash(path)[:n]


def make_provenance(learned_at, *, data: Any = None, config: Any = None, experiment_id: str = "", run_id: str = "",
                    seed: int | None = None, outcomes_seen_through: str = "", sealed_windows: Iterable[str] = (),
                    parents: Iterable[str] = (), code_hash: str | None = None, created_real: str | None = None) -> Provenance:
    """Code, data and config provenance in one call. `data`/`config` may be any hashable-by-content structure (or a path
    string to an existing file, which is hashed by bytes). `created_real` is wall-clock and never enters content_hash."""
    def h(x):
        if x is None or x == "":
            return ""
        if isinstance(x, (str, os.PathLike)) and os.path.isfile(x):
            return file_hash(x)
        return stable_hash(x, 16)
    return Provenance(created_real=created_real or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      learned_at=str(as_date(learned_at)), code_hash=code_hash or current_code_hash() or "unknown",
                      data_hash=h(data), config_hash=h(config), experiment_id=experiment_id, run_id=run_id, seed=seed,
                      outcomes_seen_through=str(as_date(outcomes_seen_through)) if outcomes_seen_through else "",
                      sealed_windows=tuple(sealed_windows), parents=tuple(parents))


def derive_id(observation: str, contexts: ContextSet, anti: ContextSet = ContextSet(), source: str = "") -> str:
    """Deterministic id from what the item says, never from a ticker/date: same lesson -> same id."""
    return "K-" + stable_hash({"o": observation, "c": contexts, "a": anti, "s": source}, 12)


# ---------------------------------------------------------------- append-only store

class KnowledgeStore:
    """Append-only version chains. Nothing is overwritten or removed; `as_of(now)` returns what existed then."""

    def __init__(self):
        self._chains: dict[str, list[KnowledgeObject]] = {}

    def __len__(self):
        return len(self._chains)

    def ids(self) -> list[str]:
        return sorted(self._chains)

    def add(self, k: KnowledgeObject) -> KnowledgeObject:
        k.assert_valid()
        chain = self._chains.get(k.knowledge_id)
        if chain is None:
            if k.version != 1:
                raise ChainError(f"{k.knowledge_id}: first stored version must be 1, got {k.version}")
            self._chains[k.knowledge_id] = [k]
            return k
        last = chain[-1]
        if k.version != last.version + 1:
            raise ChainError(f"{k.knowledge_id}: expected version {last.version + 1}, got {k.version}")
        if k.parent_hash != last.record_hash():
            raise ChainError(f"{k.knowledge_id}: parent_hash does not match version {last.version}")
        if as_date(k.updated_at) < as_date(last.updated_at):
            raise ChainError(f"{k.knowledge_id}: version dated before its parent")
        chain.append(k)
        return k

    def history(self, kid: str) -> tuple[KnowledgeObject, ...]:
        return tuple(self._chains.get(kid, ()))

    def latest(self, kid: str) -> KnowledgeObject | None:
        c = self._chains.get(kid)
        return c[-1] if c else None

    def get(self, kid: str, version: int) -> KnowledgeObject:
        c = self._chains.get(kid, [])
        if not 1 <= version <= len(c):
            raise KeyError(f"{kid} v{version}")
        return c[version - 1]

    def as_of(self, kid: str, now) -> KnowledgeObject | None:
        """Newest version that already existed strictly before `now` (else None: it did not exist yet)."""
        for k in reversed(self._chains.get(kid, [])):
            if k.visible_at(now):
                return k
        return None

    def visible(self, now) -> list[KnowledgeObject]:
        return [k for k in (self.as_of(i, now) for i in self.ids()) if k is not None]

    def verify(self) -> list[str]:
        errs = []
        for kid, chain in self._chains.items():
            for i, k in enumerate(chain):
                if k.version != i + 1:
                    errs.append(f"{kid}: version gap at index {i}")
                if i and k.parent_hash != chain[i - 1].record_hash():
                    errs.append(f"{kid}: v{k.version} parent_hash mismatch (history altered)")
                errs += [f"{kid} v{k.version}: {e}" for e in k.validate()]
        return errs

    def dump(self, path: str | os.PathLike) -> int:
        """One JSON line per version, written atomically as bytes with LF endings."""
        lines = [k.to_json() for i in self.ids() for k in self._chains[i]]
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        os.replace(tmp, path)
        return len(lines)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "KnowledgeStore":
        s = cls()
        with open(path, "rb") as f:
            for n, raw in enumerate(f.read().decode("utf-8").splitlines(), 1):
                if raw.strip():
                    try:
                        s.add(KnowledgeObject.from_json(raw))
                    except (SchemaError, ChainError) as e:
                        raise type(e)(f"line {n}: {e}") from e
        return s


# ---------------------------------------------------------------- adapters from existing knowledge stores
# Existing stores (pattern_identity.PatternRecord, PatternMiner rows, lessons.Lesson) are PRODUCERS: they keep their own
# statistics and identity hashes. These adapters read them by duck type (no import of their modules) and never
# re-derive their numbers. State vocabularies are translated by epistemic.map_state, the single mapping table.

def _dimension_of(feature: str) -> str:
    """Section-8 dimension for a feature name: m_* are market context, everything else describes the stock."""
    if feature.startswith("m_"):
        return "market"
    for prefix, dim in (("vol", "volatility"), ("atr", "volatility"), ("liq", "liquidity"), ("adv", "liquidity"),
                        ("sector", "sector"), ("regime", "regime")):
        if feature.startswith(prefix):
            return dim
    return "stock"


def _mapped(vocabulary: str, state: str):
    from engine.learning.epistemic import map_state
    m = map_state(vocabulary, state)
    promotion = Promotion.RETIRED if m.epistemic == Epistemic.RETIRED else Promotion.RESEARCH
    return m, promotion


def from_pattern_row(row: Mapping[str, Any], now, provenance: Provenance, n: int = 0, n_eff: float | None = None,
                     decision_effect: tuple[DecisionEffect, ...] = (DecisionEffect.NONE,)) -> KnowledgeObject:
    """Wrap one PatternMiner.patterns row. p_real is stored as truth confidence (P(real)), NOT as usefulness or current
    reliability, which stay untested (section 33); the promotion stays RESEARCH - only the promotion gate may raise it."""
    status = str(row.get("status", "candidate"))
    m, promo = _mapped("pattern_lifecycle.STATES", status)
    eff = float(row.get("effect", 0.0) or 0.0)
    t = row.get("t_conf", row.get("t_disc"))
    se = abs(eff) / abs(float(t)) if t not in (None, 0) and not (isinstance(t, float) and math.isnan(t)) else None
    name = str(row.get("key_named", row.get("name", "")))
    p_real = row.get("p_real")
    truth = None if p_real is None or (isinstance(p_real, float) and math.isnan(p_real)) else min(1.0, max(0.0, float(p_real)))
    if m.epistemic == Epistemic.RETIRED:
        decision_effect = (DecisionEffect.NONE,)
    obs = f"pattern {name}: mean forward-return effect {eff:+.4f}"
    return KnowledgeObject(
        knowledge_id=derive_id(obs, ContextSet(), source="pattern"), created_at=str(as_date(now)), provenance=provenance,
        observation=obs, hypothesis=f"{name} predicts forward return",
        effect=Effect(direction=0 if eff == 0 else (1 if eff > 0 else -1), size=abs(eff), uncertainty=se),
        confidence=Confidence(truth=truth),
        evidence=Evidence(sample_size=int(n), effective_sample_size=float(n if n_eff is None else n_eff)),
        mechanism_tags=("mined_pattern",), decision_effect=decision_effect, epistemic=m.epistemic, lifecycle=m.lifecycle or
        Lifecycle.BIRTH, promotion=promo).assert_valid()


def from_pattern_record(rec: Any, now, provenance: Provenance, decision_effect: tuple[DecisionEffect, ...] = (DecisionEffect.NONE,),
                        n: int | None = None) -> KnowledgeObject:
    """Wrap a pattern_identity.PatternRecord. Identity is the record's own pattern_hash (`rec.id`), so the same pattern is
    the same knowledge in both systems. `unless` terms become anti_contexts (ANY of them excludes); a rescoped pattern's
    Scope becomes a required context. Quantile terms are conditions on the feature's quantile level ('<feature>.q')."""
    m, promo = _mapped("pattern_lifecycle.STATES", rec.state)
    expr = rec.expression
    conds = [Condition(_dimension_of(t.feature), f"{t.feature}.q", "eq", nums=(float(t.level),)) for t in expr.base]
    anti = [Condition(_dimension_of(t.feature), f"{t.feature}.q", "eq", nums=(float(t.level),)) for t in expr.unless]
    if rec.scope is not None:
        s = rec.scope
        conds.append(Condition("regime", s.context, "between", nums=(float(s.lo), float(s.hi)))
                     if s.label == "mid" else Condition("regime", s.context, "le" if s.label == "low" else "gt",
                                                        nums=(float(s.lo if s.label == "low" else s.hi),)))
    st = dict(rec.stats)
    eff = float(st.get("effect") if st.get("effect") is not None else (st.get("m_all") or 0.0))
    t = st.get("t_conf") if st.get("t_conf") is not None else st.get("t_all")
    n_eff = float(st.get("n_eff") or 0.0)
    n_rows = int(n if n is not None else (st.get("n_rows") or n_eff))
    truth = st.get("p_real")
    last = rec.history[-1][0] if rec.history else str(as_date(now))
    return KnowledgeObject(
        knowledge_id="K-" + rec.id, created_at=str(as_date(now)), provenance=provenance,
        observation=f"pattern {rec.text} on {rec.target} ({rec.transform})", hypothesis=f"{rec.text} predicts {rec.target}",
        contexts=ContextSet(tuple(conds)), anti_contexts=ContextSet(tuple(anti), any_of=True),
        effect=Effect(0 if eff == 0 else (1 if eff > 0 else -1), abs(eff),
                      abs(eff) / abs(t) if t not in (None, 0) and eff else None),
        confidence=Confidence(truth=None if truth is None else min(1.0, max(0.0, float(truth)))),
        evidence=Evidence(n_rows, min(float(n_rows), n_eff), None, None, str(as_date(last)) if rec.history else ""),
        mechanism_tags=("mined_pattern", f"family:{rec.family}"), decision_effect=decision_effect if m.epistemic != Epistemic.RETIRED
        else (DecisionEffect.NONE,), epistemic=m.epistemic, lifecycle=m.lifecycle or Lifecycle.BIRTH, promotion=promo).assert_valid()


_LESSON_EFFECT = {"bad_entry": DecisionEffect.RANKING, "regime_misread": DecisionEffect.RANKING,
                  "missed_winner": DecisionEffect.RANKING, "oversized_loser": DecisionEffect.POSITION_SIZE,
                  "missed_exit": DecisionEffect.EXIT}
_LESSON_SUBSYSTEM = {"bad_entry": Subsystem.TIMING, "regime_misread": Subsystem.SELECTION, "missed_winner": Subsystem.SELECTION,
                     "oversized_loser": Subsystem.RISK, "missed_exit": Subsystem.EXIT}
_LESSON_CAUSE = {"regime_failure": FailureCause.REGIME_CHANGE, "pattern_failure": FailureCause.WEAKENING_EFFECT,
                 "missed_winner": FailureCause.SELECTION_ERROR, "false_positive": FailureCause.SELECTION_ERROR,
                 "false_negative": FailureCause.SELECTION_ERROR}


def from_lesson(lesson: Any, now, provenance: Provenance) -> KnowledgeObject:
    """Wrap a lessons.Lesson. Its conditions become required contexts; its Beta trust becomes CURRENT RELIABILITY (how much the
    lesson has earned since birth) and nothing else: the lesson's p-value is not P(real), so truth stays untested. The
    lesson's declared kind fixes both the decision it may touch and the subsystem that made the mistake (section 23)."""
    ops = {">": "gt", "<=": "le"}
    conds = []
    for f, op, thr in lesson.conds:
        if op not in ops:
            raise SchemaError(f"lesson condition operator {op!r} unsupported")
        conds.append(Condition(_dimension_of(str(f)), str(f), ops[op], nums=(float(thr),)))
    retired = str(lesson.status) == "retired"
    delta = float(lesson.delta)
    t = float(lesson.t) if lesson.t else 0.0
    fail = () if lesson.category == "ok" else (FailureExplanation(
        _LESSON_CAUSE.get(lesson.category, FailureCause.UNKNOWN), str(as_date(now)), _LESSON_SUBSYSTEM.get(lesson.kind),
        f"lesson category {lesson.category}", str(lesson.lid)),)
    de = (DecisionEffect.NONE,) if retired else (_LESSON_EFFECT.get(lesson.kind, DecisionEffect.NONE),)
    trust = float(lesson.a) / (float(lesson.a) + float(lesson.b))
    return KnowledgeObject(
        knowledge_id="K-L" + stable_hash({"lid": lesson.lid, "conds": [list(c) for c in lesson.conds]}, 12),
        created_at=str(as_date(now)), provenance=provenance, source_experiences=tuple(str(s) for s in lesson.support[:50]),
        observation=lesson.describe(), hypothesis=f"decisions in this region earn {delta:+.4f} relative to elsewhere",
        contexts=ContextSet(tuple(conds)),
        effect=Effect(0 if delta == 0 else (1 if delta > 0 else -1), abs(delta), abs(delta) / abs(t) if t and delta else None,
                      "decision_pnl", 5),
        confidence=Confidence(current_reliability=min(1.0, max(0.0, trust))),
        evidence=Evidence(int(lesson.n), min(float(lesson.n), float(lesson.n_weeks)), None, None, ""),
        dynamics=Dynamics(stability=min(1.0, max(0.0, float(lesson.stability)))),
        mechanism_tags=("lesson", f"kind:{lesson.kind}"), failure_explanations=fail, decision_effect=de,
        epistemic=Epistemic.RETIRED if retired else Epistemic.HYPOTHESIS,
        lifecycle=Lifecycle.RETIRED if retired else Lifecycle.ACTIVE,
        promotion=Promotion.RETIRED if retired else Promotion.RESEARCH).assert_valid()


# ---------------------------------------------------------------- reading conditions from text

_COND_RE = None


def parse_condition(text: str) -> Condition:
    """'volatility: vix >= 25' | 'regime: trend in [bull, sideways]' | 'liquidity: adv between 1e6 1e8' -> Condition.
    The inverse of `Condition.describe` plus the dimension prefix; strict, so a typo cannot silently become a wider context."""
    global _COND_RE
    import re
    if _COND_RE is None:
        _COND_RE = re.compile(r"^\s*(\w+)\s*:\s*(\w+)\s+(eq|ne|in|not_in|lt|le|gt|ge|between|==|!=|<=|>=|<|>)\s+(.+?)\s*$")
    m = _COND_RE.match(text)
    if not m:
        raise SchemaError(f"cannot parse condition {text!r}")
    dim, feat, op, rest = m.groups()
    op = {"==": "eq", "!=": "ne", "<=": "le", ">=": "ge", "<": "lt", ">": "gt"}.get(op, op)
    rest = rest.strip().strip("[]()")
    toks = [t.strip().strip("'\"") for t in rest.replace(",", " ").split() if t.strip()]
    nums, labels = [], []
    for t in toks:
        try:
            nums.append(float(t))
        except ValueError:
            labels.append(t)
    if nums and labels:
        raise SchemaError(f"condition {text!r} mixes numbers and labels")
    c = Condition(dim, feat, op, tuple(nums), tuple(labels))
    errs = c.check()
    if errs:
        raise SchemaError(f"{text!r}: " + "; ".join(errs))
    return c


def context_from_text(*parts: str) -> ContextSet:
    return ContextSet(tuple(parse_condition(p) for p in parts))


def context_overlap(a: ContextSet, b: ContextSet, grid: Iterable[Mapping[str, Any]]) -> float:
    """Jaccard overlap of the situations two contexts accept, measured on a caller-supplied grid of situations (no data
    is read here). 1.0 means the two contexts are interchangeable on that grid; unknown evaluations count as not accepted."""
    both = either = 0
    for s in grid:
        ia, ib = a.evaluate(s) is True, b.evaluate(s) is True
        both += ia and ib
        either += ia or ib
    return both / either if either else 0.0


# ---------------------------------------------------------------- pooling evidence (real statistics, not averaging)

@dataclasses.dataclass(frozen=True)
class Pooled:
    effect: Effect
    q: float                    # Cochran Q
    i2: float                   # share of variation due to heterogeneity, [0,1]
    tau2: float                 # between-source variance (0 = fixed effect)
    k: int


def pool_effects(effects: Iterable[Effect], random: bool = True) -> Pooled:
    """Inverse-variance pooling of effects measured by independent sources (eras, stocks, runs). The DerSimonian-Laird
    random-effects step widens the uncertainty when sources disagree, so heterogeneous evidence cannot look precise.
    Effects with no estimated uncertainty cannot be weighted and are refused rather than given an arbitrary weight."""
    es = list(effects)
    if not es:
        raise SchemaError("nothing to pool")
    if any(e.uncertainty is None or e.uncertainty <= 0 for e in es):
        raise SchemaError("pooling needs a positive standard error on every effect")
    if len({(e.unit, e.horizon_days) for e in es}) != 1:
        raise SchemaError("cannot pool effects in different units or horizons")
    y = [e.signed for e in es]
    v = [e.uncertainty ** 2 for e in es]
    w = [1.0 / x for x in v]
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    q = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y))
    k = len(es)
    tau2 = 0.0
    if random and k > 1:
        c = sum(w) - sum(wi * wi for wi in w) / sum(w)
        tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
        w = [1.0 / (x + tau2) for x in v]
        mu = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    se = math.sqrt(1.0 / sum(w))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    eff = Effect(0 if abs(mu) < 1e-15 else (1 if mu > 0 else -1), abs(mu), se, es[0].unit, es[0].horizon_days)
    return Pooled(eff, q, i2, tau2, k)


def pool_evidence(items: Iterable[Evidence]) -> Evidence:
    """Sum of sample sizes; effective size adds too (sources are assumed independent - callers pooling overlapping windows
    must pass the already-deduplicated evidence); recency/modernity are size-weighted."""
    its = list(items)
    if not its:
        return Evidence()
    n = sum(e.sample_size for e in its)
    ne = sum(e.effective_sample_size for e in its)

    def wavg(name):
        pairs = [(getattr(e, name), e.sample_size) for e in its if getattr(e, name) is not None and e.sample_size > 0]
        tot = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / tot if tot else None

    last = max((e.last_evidence_at for e in its if e.last_evidence_at), default="")
    return Evidence(n, ne, wavg("recency"), wavg("modernity"), last)


def recency_score(last_evidence_at: str, now, half_life_days: float = 365.0) -> float:
    """Exponential decay of evidence age into [0,1]; evidence dated at/after `now` is a leak and raises."""
    age = (as_date(now) - as_date(last_evidence_at)).days
    if age <= 0:
        raise FirewallBreach(f"evidence dated {last_evidence_at} is not before now={now}")
    return 0.5 ** (age / half_life_days)


def modernity_score(evidence_dates: Iterable[str], modern_from: str) -> float | None:
    ds = [as_date(d) for d in evidence_dates]
    if not ds:
        return None
    cut = as_date(modern_from)
    return sum(d >= cut for d in ds) / len(ds)


# ---------------------------------------------------------------- comparing versions

@dataclasses.dataclass(frozen=True)
class FieldChange:
    path: str
    old: Any
    new: Any


def diff(a: KnowledgeObject, b: KnowledgeObject) -> list[FieldChange]:
    """Leaf-level differences between two versions (what a version actually changed), ignoring version bookkeeping."""
    skip = {"version", "parent_hash", "version_reason", "updated_at"}
    out: list[FieldChange] = []

    def walk(path, x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(f"{path}.{k}" if path else k, x.get(k), y.get(k))
        elif x != y:
            out.append(FieldChange(path, x, y))

    da, db = encode(a), encode(b)
    for s in skip:
        da.pop(s, None)
        db.pop(s, None)
    walk("", da, db)
    return out


def summarize_change(changes: list[FieldChange]) -> str:
    if not changes:
        return "no change"
    top = sorted({c.path.split(".")[0] for c in changes})
    return f"{len(changes)} field(s) changed in {', '.join(top)}"


# ---------------------------------------------------------------- store queries and audits

def _matches(k: KnowledgeObject, epistemic, lifecycle, promotion, tag, effect) -> bool:
    return ((epistemic is None or k.epistemic == Epistemic.parse(epistemic))
            and (lifecycle is None or k.lifecycle == Lifecycle.parse(lifecycle))
            and (promotion is None or k.promotion == Promotion.parse(promotion))
            and (tag is None or tag in k.mechanism_tags)
            and (effect is None or DecisionEffect.parse(effect) in k.decision_effect))


def find(store: "KnowledgeStore", now, *, epistemic=None, lifecycle=None, promotion=None, tag=None, effect=None
         ) -> list[KnowledgeObject]:
    """Latest versions visible at `now` matching every given filter. Never returns anything from the future."""
    return [k for k in store.visible(now) if _matches(k, epistemic, lifecycle, promotion, tag, effect)]


def audit_future(store: "KnowledgeStore", now) -> list[str]:
    """Anything in the store that would be a future-memory leak at `now`: knowledge learned or updated on/after it."""
    bad = []
    for kid in store.ids():
        for k in store.history(kid):
            if as_date(k.updated_at) >= as_date(now):
                continue                         # a later version does not matter: it is invisible to `as_of(now)`
            if not k.provenance.could_exist_at(now):
                bad.append(f"{kid} v{k.version}: learned {k.provenance.learned_at}, outcomes through "
                           f"{k.provenance.outcomes_seen_through or 'n/a'} - not knowable at {now}")
    return bad


def epistemic_timeline(store: "KnowledgeStore", kid: str) -> list[tuple[int, str, str, str]]:
    """(version, updated_at, epistemic, reason) per version: how belief in one item moved over time."""
    return [(k.version, k.updated_at, k.epistemic.value, k.version_reason) for k in store.history(kid)]


def lineage(store: "KnowledgeStore", kid: str) -> list[str]:
    """Ids this item was derived from, transitively (provenance.parents of every version; version tags stripped)."""
    seen, todo = [], [kid]
    while todo:
        cur = todo.pop()
        for k in store.history(cur):
            for p in k.provenance.parents + k.relations.parent:
                pid = p.split("@")[0]
                if pid != cur and pid not in seen:
                    seen.append(pid)
                    todo.append(pid)
    return sorted(seen)


def verify_file(path: str | os.PathLike) -> list[str]:
    """Integrity check of a dumped store: every line decodes, validates, and chains. Returns findings instead of raising."""
    errs: list[str] = []
    try:
        KnowledgeStore.load(path)
    except (SchemaError, ChainError) as e:
        errs.append(str(e))
    except OSError as e:
        errs.append(f"cannot read {path}: {e}")
    return errs


def store_stats(store: "KnowledgeStore", now) -> dict[str, Any]:
    """Counts a health page can show: by epistemic state, lifecycle, promotion, research-only share, version depth."""
    vis = store.visible(now)

    def by(f):
        return dict(sorted(Counter(f(k) for k in vis).items()))

    return {"items": len(store), "visible": len(vis),
            "epistemic": by(lambda k: k.epistemic.value), "lifecycle": by(lambda k: k.lifecycle.value),
            "promotion": by(lambda k: k.promotion.value),
            "research_only": sum(1 for k in vis if set(k.decision_effect) <= {DecisionEffect.NONE}),
            "max_versions": max((len(store.history(i)) for i in store.ids()), default=0)}


# ---------------------------------------------------------------- common edits, each one a new version

def with_confidence(k: KnowledgeObject, now, reason: str, **dims: float | None) -> KnowledgeObject:
    """New version with some confidence dimensions replaced. Dimensions are never derived from one another: passing
    truth does not touch usefulness. None re-marks a dimension untested."""
    bad = [d for d in dims if d not in {f.name for f in dataclasses.fields(Confidence)}]
    if bad:
        raise SchemaError(f"unknown confidence dimensions {bad}")
    return k.new_version(now, reason, confidence=dataclasses.replace(k.confidence, **dims))


def with_failure(k: KnowledgeObject, now, cause: FailureCause, subsystem: Subsystem | None = None, note: str = "",
                 evidence_id: str = "") -> KnowledgeObject:
    """Record why the item failed (section 9). UNKNOWN is an allowed cause; a specific cause needs a note or evidence id."""
    fe = FailureExplanation(cause, str(as_date(now)), subsystem, note, evidence_id)
    errs = fe.check()
    if errs:
        raise SchemaError("; ".join(errs))
    return k.new_version(now, f"failure explained: {cause.value}", failure_explanations=k.failure_explanations + (fe,))


_RELATION_FIELDS = tuple(f.name for f in dataclasses.fields(Relations))


def with_relation(k: KnowledgeObject, now, kind: str, other_id: str, reason: str = "") -> KnowledgeObject:
    """New version adding one edge (section 18). Adding an existing edge is refused as a no-op rather than duplicated."""
    if kind not in _RELATION_FIELDS:
        raise SchemaError(f"unknown relation kind {kind!r}; known: {list(_RELATION_FIELDS)}")
    cur = getattr(k.relations, kind)
    if other_id in cur:
        raise SchemaError(f"{k.knowledge_id} already has {other_id} as {kind}")
    return k.new_version(now, reason or f"{kind} += {other_id}",
                         relations=dataclasses.replace(k.relations, **{kind: cur + (other_id,)}))


def with_temporal_class(k: KnowledgeObject, now, cls: TemporalClass, reason: str) -> KnowledgeObject:
    return k.new_version(now, reason, temporal_class=TemporalClass.parse(cls))


def find_duplicates(store: "KnowledgeStore", now) -> list[tuple[str, str]]:
    """Pairs of visible items with identical content_hash (the same lesson learned twice under two ids). Reported, never
    merged automatically: merging is a redundancy decision (section 21) that needs evidence."""
    seen: dict[str, str] = {}
    out = []
    for k in store.visible(now):
        h = k.content_hash()
        if h in seen and seen[h] != k.knowledge_id:
            out.append((seen[h], k.knowledge_id))
        seen.setdefault(h, k.knowledge_id)
    return sorted(out)


def same_content(a: KnowledgeObject, b: KnowledgeObject) -> bool:
    return a.content_hash() == b.content_hash()
