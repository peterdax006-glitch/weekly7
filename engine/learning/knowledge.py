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
        """AND of all conditions: False if any is False; None if none is False but some are unknown; else True.
        An empty set is vacuously True (the item claims no context restriction)."""
        unknown = False
        for c in self.conditions:
            r = c.test(situation.get(c.feature))
            if r is False:
                return False
            unknown = unknown or r is None
        return None if unknown else True

    def implies(self, other: "ContextSet") -> bool:
        """self is at least as narrow as other: every condition of other is implied by some condition of self."""
        return all(any(s.implies(o) for s in self.conditions) for o in other.conditions)

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
        if self.contexts.conditions and self.anti_contexts.conditions and (
                self.contexts.implies(self.anti_contexts) or self.anti_contexts.implies(self.contexts)):
            errs.append("contexts and anti_contexts cover the same region: the item would exclude itself")
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
            if self.epistemic not in (Epistemic.SUPPORTED, Epistemic.CONDITIONAL):
                errs.append("CHAMPION knowledge must be SUPPORTED or CONDITIONAL")
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
        """Identity of the knowledge itself: ignores version metadata and wall-clock, so re-deriving the same lesson twice
        gives the same hash (used to detect duplicate lessons and unchanged re-versions)."""
        d = encode(self)
        for k in ("version", "parent_hash", "version_reason", "updated_at"):
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
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


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


# ---------------------------------------------------------------- adapter from the existing pattern miner

_PATTERN_STATUS = {          # PatternMiner/pattern_lifecycle status -> (epistemic, lifecycle, promotion)
    "active": (Epistemic.HYPOTHESIS, Lifecycle.ACTIVE, Promotion.RESEARCH),
    "rescoped": (Epistemic.CONDITIONAL, Lifecycle.RECOVERY, Promotion.RESEARCH),
    "watch": (Epistemic.DEGRADED, Lifecycle.DECAY, Promotion.RESEARCH),
    "no_gain": (Epistemic.GATED, Lifecycle.DORMANT, Promotion.RESEARCH),
    "duplicate": (Epistemic.GATED, Lifecycle.DORMANT, Promotion.RESEARCH),
    "candidate": (Epistemic.HYPOTHESIS, Lifecycle.BIRTH, Promotion.RESEARCH),
    "failed": (Epistemic.CONTRADICTED, Lifecycle.FAILURE, Promotion.RESEARCH),
    "cause_search": (Epistemic.CONTRADICTED, Lifecycle.FAILURE, Promotion.RESEARCH),
    "discarded": (Epistemic.RETIRED, Lifecycle.RETIRED, Promotion.RETIRED),
    "rejected": (Epistemic.RETIRED, Lifecycle.RETIRED, Promotion.RETIRED),
}


def from_pattern_row(row: Mapping[str, Any], now, provenance: Provenance, n: int = 0, n_eff: float | None = None,
                     decision_effect: tuple[DecisionEffect, ...] = (DecisionEffect.NONE,)) -> KnowledgeObject:
    """Wrap one PatternMiner.patterns row. Deliberately conservative: a mined pattern enters as a HYPOTHESIS at RESEARCH
    promotion; only the evidence machinery (epistemic/promotion modules) may raise it. p_real is stored as truth confidence
    (P(real)), NOT as usefulness or current reliability, which stay untested (section 33)."""
    status = str(row.get("status", "candidate"))
    if status not in _PATTERN_STATUS:
        raise SchemaError(f"unknown pattern status {status!r}")
    ep, lc, pr = _PATTERN_STATUS[status]
    eff = float(row.get("effect", 0.0) or 0.0)
    t = row.get("t_conf", row.get("t_disc"))
    se = abs(eff) / abs(float(t)) if t not in (None, 0) and not (isinstance(t, float) and math.isnan(t)) else None
    name = str(row.get("key_named", row.get("name", "")))
    p_real = row.get("p_real")
    truth = None if p_real is None or (isinstance(p_real, float) and math.isnan(p_real)) else min(1.0, max(0.0, float(p_real)))
    if ep == Epistemic.RETIRED:
        decision_effect = (DecisionEffect.NONE,)
    obs = f"pattern {name}: mean forward-return effect {eff:+.4f}"
    k = KnowledgeObject(
        knowledge_id=derive_id(obs, ContextSet(), source="pattern"), created_at=str(as_date(now)), provenance=provenance,
        observation=obs, hypothesis=f"{name} predicts forward return", effect=Effect(
            direction=0 if eff == 0 else (1 if eff > 0 else -1), size=abs(eff), uncertainty=se),
        confidence=Confidence(truth=truth), evidence=Evidence(sample_size=int(n), effective_sample_size=float(n if n_eff is None else n_eff)),
        mechanism_tags=("mined_pattern",), decision_effect=decision_effect, epistemic=ep, lifecycle=lc, promotion=pr)
    return k.assert_valid()
