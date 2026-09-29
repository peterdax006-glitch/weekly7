"""Contract C62 section 42 (A02/A08): UNKNOWN, INSUFFICIENT_DATA, CONFLICTED, UNTESTED as first-class answers.

These states exist so the learner can say "I do not know" without that answer being converted into a number. The rules the
module enforces:
  * an unknown state has an ACTION (ABSTAIN / COLLECT_DATA / RUN_EXPERIMENT / USE_GENERAL_RULE), never a confidence;
  * an `UnknownScore` cannot be cast to float, and combining scores never imputes a value for the unknown ones;
  * `as_confidence` on an unknown yields the all-None Confidence (untested), or raises; it can never yield 0.5;
  * `audit_confidence_laundering` finds items that carry confidence they have no evidence for.
Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Iterable, Mapping

from engine.learning.core import (Confidence, Unknown, UnknownAction, as_date, require_past, stable_hash)
from engine.learning.knowledge import SchemaError, decode, encode


class UnknownAsConfidence(TypeError):
    """An unknown state was about to be read as a number. That is the bug this module exists to prevent."""


class UnknownReason(str, enum.Enum):
    NO_DATA = "NO_DATA"
    TOO_FEW_EVENTS = "TOO_FEW_EVENTS"
    LOW_EFFECTIVE_N = "LOW_EFFECTIVE_N"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    NEVER_TESTED = "NEVER_TESTED"
    NEVER_TESTED_IN_CONTEXT = "NEVER_TESTED_IN_CONTEXT"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    CONTEXT_FEATURE_MISSING = "CONTEXT_FEATURE_MISSING"
    UNEXPLAINED = "UNEXPLAINED"


@dataclasses.dataclass(frozen=True)
class UnknownFacts:
    """What is known about how much is known. All measured by the caller on data that matured before `now`."""
    n_events: int = 0
    n_eff: float = 0.0
    min_n_eff: float = 30.0
    ever_tested: bool = False
    tested_contexts: tuple[str, ...] = ()
    asked_context: str = ""                 # the context the decision is being made in ("" = context-free question)
    support: float | None = None            # strength of evidence for the effect, in [0,1]
    against: float | None = None            # strength of evidence against, in [0,1]
    conflict_level: float = 0.35            # both sides above this = CONFLICTED
    evidence_age_days: int | None = None
    max_age_days: int | None = None
    missing_features: tuple[str, ...] = ()

    def check(self) -> list[str]:
        errs = []
        if self.n_events < 0 or self.n_eff < 0 or math.isnan(self.n_eff):
            errs.append("counts must be >= 0")
        if self.n_eff > self.n_events + 1e-9:
            errs.append("n_eff exceeds n_events")
        for n in ("support", "against"):
            v = getattr(self, n)
            if v is not None and not 0.0 <= v <= 1.0:
                errs.append(f"{n} outside [0,1]")
        if self.evidence_age_days is not None and self.evidence_age_days < 0:
            errs.append("evidence cannot be from the future (negative age)")
        return errs


@dataclasses.dataclass(frozen=True)
class Availability:
    """What the system could do about not knowing. The action is chosen from these, never from optimism."""
    can_collect_data: bool = False          # more of the same evidence will plausibly arrive in time
    can_run_experiment: bool = False        # a disambiguating test is affordable within the compute budget
    general_rule: str = ""                  # name of a safe fallback (e.g. "equal_weight", "baseline_rank")


@dataclasses.dataclass(frozen=True)
class UnknownRecord:
    subject: str
    state: Unknown
    reasons: tuple[UnknownReason, ...]
    action: UnknownAction
    since: str
    action_detail: str = ""
    resolved_at: str = ""
    resolution: str = ""

    @property
    def open(self) -> bool:
        return not self.resolved_at

    @property
    def record_id(self) -> str:
        return "UNK-" + stable_hash({"s": self.subject, "st": self.state, "since": self.since}, 10)

    def as_confidence(self):
        raise UnknownAsConfidence(f"{self.subject}: {self.state.value} is not a confidence; act with {self.action.value}")

    def __float__(self):
        return self.as_confidence()

    def check(self) -> list[str]:
        errs = []
        if not self.subject:
            errs.append("unknown record without a subject")
        if not self.reasons:
            errs.append("unknown record without a reason")
        legal = ACTIONS_FOR[self.state]
        if self.action not in legal:
            errs.append(f"{self.state.value} cannot be handled by {self.action.value}; allowed {sorted(a.value for a in legal)}")
        try:
            as_date(self.since)
            if self.resolved_at and as_date(self.resolved_at) < as_date(self.since):
                errs.append("resolved before it was opened")
        except ValueError:
            errs.append("since/resolved_at must be ISO dates")
        return errs


ACTIONS_FOR: dict[Unknown, frozenset[UnknownAction]] = {
    Unknown.INSUFFICIENT_DATA: frozenset({UnknownAction.COLLECT_DATA, UnknownAction.ABSTAIN, UnknownAction.USE_GENERAL_RULE}),
    Unknown.UNTESTED: frozenset({UnknownAction.RUN_EXPERIMENT, UnknownAction.ABSTAIN, UnknownAction.USE_GENERAL_RULE}),
    Unknown.CONFLICTED: frozenset({UnknownAction.RUN_EXPERIMENT, UnknownAction.ABSTAIN}),      # never a blind average
    Unknown.UNKNOWN: frozenset({UnknownAction.USE_GENERAL_RULE, UnknownAction.ABSTAIN, UnknownAction.COLLECT_DATA}),
}


def choose_action(state: Unknown, avail: Availability) -> tuple[UnknownAction, str]:
    """Deterministic policy. ABSTAIN is always the safe last resort and is never conditional on anything."""
    if state == Unknown.INSUFFICIENT_DATA:
        if avail.can_collect_data:
            return UnknownAction.COLLECT_DATA, "accumulate more evidence before relying on it"
        if avail.general_rule:
            return UnknownAction.USE_GENERAL_RULE, f"fall back to {avail.general_rule}"
    elif state == Unknown.UNTESTED:
        if avail.can_run_experiment:
            return UnknownAction.RUN_EXPERIMENT, "test it in the asked context before it may influence anything"
        if avail.general_rule:
            return UnknownAction.USE_GENERAL_RULE, f"fall back to {avail.general_rule}"
    elif state == Unknown.CONFLICTED:
        if avail.can_run_experiment:
            return UnknownAction.RUN_EXPERIMENT, "design a test that separates the two readings"
    elif state == Unknown.UNKNOWN:
        if avail.general_rule:
            return UnknownAction.USE_GENERAL_RULE, f"fall back to {avail.general_rule}"
        if avail.can_collect_data:
            return UnknownAction.COLLECT_DATA, "no basis to judge; observe first"
    return UnknownAction.ABSTAIN, "no safe way to act on this; do not act"


def classify(subject: str, facts: UnknownFacts, now, avail: Availability = Availability()) -> UnknownRecord | None:
    """Return the unknown state the facts amount to, or None when it is known well enough to be judged on its evidence.
    Order matters: never-tested is reported before thin data, and conflict before both, because it changes the action."""
    bad = facts.check()
    if bad:
        raise SchemaError("; ".join(bad))
    reasons: list[UnknownReason] = []
    state: Unknown | None = None
    if not facts.ever_tested:
        state, reasons = Unknown.UNTESTED, [UnknownReason.NEVER_TESTED]
    elif facts.asked_context and facts.asked_context not in facts.tested_contexts:
        state, reasons = Unknown.UNTESTED, [UnknownReason.NEVER_TESTED_IN_CONTEXT]
    elif (facts.support is not None and facts.against is not None
          and min(facts.support, facts.against) >= facts.conflict_level):
        state, reasons = Unknown.CONFLICTED, [UnknownReason.EVIDENCE_CONFLICT]
    else:
        if facts.n_events == 0:
            reasons.append(UnknownReason.NO_DATA)
        elif facts.n_events < facts.min_n_eff:
            reasons.append(UnknownReason.TOO_FEW_EVENTS)
        elif facts.n_eff < facts.min_n_eff:
            reasons.append(UnknownReason.LOW_EFFECTIVE_N)
        if (facts.evidence_age_days is not None and facts.max_age_days is not None
                and facts.evidence_age_days > facts.max_age_days):
            reasons.append(UnknownReason.STALE_EVIDENCE)
        if reasons:
            state = Unknown.INSUFFICIENT_DATA
        elif facts.missing_features:
            state, reasons = Unknown.UNKNOWN, [UnknownReason.CONTEXT_FEATURE_MISSING]
    if state is None:
        return None
    action, detail = choose_action(state, avail)
    rec = UnknownRecord(subject, state, tuple(reasons), action, str(as_date(now)), detail)
    errs = rec.check()
    if errs:
        raise SchemaError("; ".join(errs))
    return rec


def unknown_state(subject: str, now, reason: UnknownReason = UnknownReason.UNEXPLAINED,
                  avail: Availability = Availability()) -> UnknownRecord:
    """Explicit 'I cannot say' with no supporting facts (the UNKNOWN case)."""
    action, detail = choose_action(Unknown.UNKNOWN, avail)
    return UnknownRecord(subject, Unknown.UNKNOWN, (reason,), action, str(as_date(now)), detail)


# ---------------------------------------------------------------- values that can be unknown

@dataclasses.dataclass(frozen=True)
class UnknownScore:
    """A score that is either a real number or an unknown. Exactly one is set; the unknown cannot be cast to float."""
    value: float | None = None
    unknown: UnknownRecord | None = None

    def __post_init__(self):
        if (self.value is None) == (self.unknown is None):
            raise SchemaError("UnknownScore needs exactly one of value / unknown")
        if self.value is not None and (isinstance(self.value, bool) or math.isnan(self.value)):
            raise SchemaError("UnknownScore value must be a finite number; use unknown= for 'not known'")

    @property
    def is_known(self) -> bool:
        return self.unknown is None

    def __float__(self):
        if self.unknown is not None:
            raise UnknownAsConfidence(f"{self.unknown.subject}: {self.unknown.state.value} has no numeric value")
        return float(self.value)

    def require(self) -> float:
        return float(self)

    def or_action(self) -> float | UnknownAction:
        return self.value if self.unknown is None else self.unknown.action


def combine(scores: Iterable[UnknownScore], weights: Iterable[float] | None = None, min_coverage: float = 0.6,
            subject: str = "combined", now="1970-01-02") -> tuple[UnknownScore, float]:
    """Weighted mean over the KNOWN scores only. Unknowns are never imputed (no 0, no 0.5, no mean): they reduce coverage,
    and below `min_coverage` the whole answer becomes unknown. Returns (score, coverage of weight that was known)."""
    sc = list(scores)
    w = [1.0] * len(sc) if weights is None else [float(x) for x in weights]
    if len(w) != len(sc) or any(x < 0 or math.isnan(x) for x in w):
        raise SchemaError("weights must match scores and be non-negative")
    total = sum(w)
    if not sc or total <= 0:
        return UnknownScore(unknown=unknown_state(subject, now, UnknownReason.NO_DATA)), 0.0
    known = [(s.value, x) for s, x in zip(sc, w) if s.is_known]
    kw = sum(x for _, x in known)
    cov = kw / total
    if cov < min_coverage or kw <= 0:
        states = {s.unknown.state for s in sc if not s.is_known}
        if states == {Unknown.CONFLICTED}:
            st, why = Unknown.CONFLICTED, UnknownReason.EVIDENCE_CONFLICT
        elif states == {Unknown.UNTESTED}:
            st, why = Unknown.UNTESTED, UnknownReason.NEVER_TESTED
        else:
            st, why = Unknown.INSUFFICIENT_DATA, UnknownReason.LOW_EFFECTIVE_N
        act = choose_action(st, Availability())
        return UnknownScore(unknown=UnknownRecord(subject, st, (why,), act[0], str(as_date(now)), act[1])), cov
    return UnknownScore(value=sum(v * x for v, x in known) / kw), cov


def as_confidence(x: Any) -> Confidence:
    """Convert something to a Confidence without ever inventing a number from an unknown."""
    if isinstance(x, Confidence):
        return x
    if isinstance(x, UnknownRecord):
        return Confidence()                    # every dimension None = untested; never 0.5
    if isinstance(x, UnknownScore):
        if x.unknown is not None:
            return Confidence()
        raise UnknownAsConfidence("a known score is not a Confidence; assign it to a named dimension explicitly")
    if isinstance(x, (Unknown, str)) and str(x) in {u.value for u in Unknown}:
        return Confidence()
    raise TypeError(f"cannot build a Confidence from {type(x).__name__}")


# ---------------------------------------------------------------- ledger

class UnknownLedger:
    """Append-only book of what the learner does not know. Resolution is a new record, the old one is kept."""

    def __init__(self):
        self._rows: list[UnknownRecord] = []

    def __len__(self):
        return len(self._rows)

    def rows(self) -> tuple[UnknownRecord, ...]:
        return tuple(self._rows)

    def open_rows(self) -> list[UnknownRecord]:
        latest: dict[str, UnknownRecord] = {}
        for r in self._rows:
            latest[r.subject] = r
        return sorted((r for r in latest.values() if r.open), key=lambda r: (r.since, r.subject))

    def open(self, rec: UnknownRecord) -> UnknownRecord:
        errs = rec.check()
        if errs:
            raise SchemaError("; ".join(errs))
        cur = next((r for r in self.open_rows() if r.subject == rec.subject), None)
        if cur is not None and cur.state == rec.state and cur.action == rec.action:
            return cur                           # same unknown already tracked: keep the original `since`
        if self._rows and as_date(rec.since) < as_date(self._rows[-1].since):
            raise SchemaError("ledger entries must arrive in time order")
        self._rows.append(rec)
        return rec

    def resolve(self, subject: str, now, evidence_through, resolution: str) -> UnknownRecord:
        """Close an unknown. The evidence that resolves it must have matured strictly before `now`."""
        require_past(evidence_through, now, f"resolution of {subject}")
        cur = next((r for r in self.open_rows() if r.subject == subject), None)
        if cur is None:
            raise KeyError(f"{subject} is not an open unknown")
        if not resolution.strip():
            raise SchemaError("a resolution needs a description")
        done = dataclasses.replace(cur, resolved_at=str(as_date(now)), resolution=resolution)
        self._rows.append(done)
        return done

    def reassess(self, subject: str, facts: UnknownFacts, now, evidence_through,
                 avail: Availability = Availability()) -> UnknownRecord | None:
        """Re-classify with fresh facts: resolved (None) or moved to whatever unknown the new facts amount to."""
        require_past(evidence_through, now, f"reassessment of {subject}")
        new = classify(subject, facts, now, avail)
        if new is None:
            self.resolve(subject, now, evidence_through, "evidence now sufficient")
            return None
        cur = next((r for r in self.open_rows() if r.subject == subject), None)
        if cur is not None and (cur.state, cur.action) != (new.state, new.action):
            self.resolve(subject, now, evidence_through, f"became {new.state.value}")
        return self.open(new)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.open_rows():
            out[r.state.value] = out.get(r.state.value, 0) + 1
        return dict(sorted(out.items()))

    def plan(self) -> dict[str, list[str]]:
        """Work queue by action, oldest first: what to collect, what to test, where to abstain."""
        out: dict[str, list[str]] = {a.value: [] for a in UnknownAction}
        for r in self.open_rows():
            out[r.action.value].append(r.subject)
        return out

    def stuck(self, now, days: int = 30) -> list[UnknownRecord]:
        return [r for r in self.open_rows() if (as_date(now) - as_date(r.since)).days > days]

    def to_dict(self) -> dict:
        return {"schema": 1, "rows": [encode(r) for r in self._rows]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "UnknownLedger":
        if not isinstance(d, Mapping) or d.get("schema") != 1:
            raise SchemaError("unsupported ledger schema")
        led = cls()
        for row in d["rows"]:
            rec = decode(UnknownRecord, row)
            errs = rec.check()
            if errs:
                raise SchemaError("; ".join(errs))
            led._rows.append(rec)
        return led


# ---------------------------------------------------------------- audits on knowledge (duck-typed KnowledgeLike)

def assess_knowledge(k: Any, situation: Mapping[str, Any], now, min_n_eff: float = 30.0,
                     avail: Availability = Availability()) -> UnknownRecord | None:
    """Is this item known well enough to be used in THIS situation? Uses only the item's own fields and the situation."""
    ev = getattr(k, "evidence", None)
    n = int(getattr(ev, "sample_size", 0) or 0)
    n_eff = float(getattr(ev, "effective_sample_size", 0.0) or 0.0)
    missing: list[str] = []
    asked = ""
    tested: tuple[str, ...] = ()
    if hasattr(k, "applicability"):
        ap = k.applicability(situation)
        if str(getattr(ap, "value", ap)) == "UNKNOWN":
            need = set(getattr(k.contexts, "features", lambda: ())()) | set(getattr(k.anti_contexts, "features", lambda: ())())
            missing = sorted(f for f in need if situation.get(f) is None)
    ep = str(getattr(k, "epistemic", ""))
    facts = UnknownFacts(n_events=n, n_eff=n_eff, min_n_eff=min_n_eff,
                         ever_tested=getattr(k, "confidence", None) is not None and k.confidence.truth is not None,
                         tested_contexts=tested, asked_context=asked,
                         missing_features=tuple(missing))
    rec = classify(str(getattr(k, "knowledge_id", "?")), facts, now, avail)
    if rec is None and ep == "UNKNOWN":
        return unknown_state(str(k.knowledge_id), now, UnknownReason.UNEXPLAINED, avail)
    return rec


def audit_confidence_laundering(items: Iterable[Any]) -> list[str]:
    """Findings where an unknown became confidence: numbers with no evidence behind them, or an UNKNOWN item with numbers."""
    out = []
    for k in items:
        kid = str(getattr(k, "knowledge_id", "?"))
        conf = getattr(k, "confidence", None)
        ev = getattr(k, "evidence", None)
        n_eff = float(getattr(ev, "effective_sample_size", 0.0) or 0.0)
        if conf is None:
            continue
        given = {f.name: getattr(conf, f.name) for f in dataclasses.fields(conf) if getattr(conf, f.name) is not None}
        if str(getattr(k, "epistemic", "")) == "UNKNOWN" and given:
            out.append(f"{kid}: epistemic UNKNOWN but confidence set for {sorted(given)}")
        if n_eff == 0 and given:
            out.append(f"{kid}: confidence {sorted(given)} with zero effective evidence")
        if n_eff == 0 and any(abs(float(v) - 0.5) < 1e-12 for v in given.values()):
            out.append(f"{kid}: a 0.5 placeholder without evidence (unknown disguised as a coin flip)")
    return out
