"""The "could I have known?" test (contract C66 section 8; Bible research brain; canons C62 + C63 + C64 + C66).
IMPLEMENTED - NOT VALIDATED.

For a matured historical movement this module reconstructs the EXACT information state the system had at the decision
timestamp T (features, patterns, memory, macro, market state, events, price, volume, technical structure, cross-section),
then compares it with what became known later, and emits the five outputs section 8 names:

    knowledge_state_at_decision, future_information_used_by_auditor, information_that_would_have_been_available,
    information_that_was_unavailable, confidence_in_classification

plus a Knowability class (engine.research.core) and a per-item Availability class (known before / simultaneous / known only
after / uncertain / unavailable). Unknown stays unknown: a move with no identified pre-event evidence and no identified
post-event cause is UNKNOWN, never a forced label (RT20).

Two worlds, kept apart by construction (sections 29-31):
  * the KNOWLEDGE STATE is built only through an engine.pit Guard bound to T, so every read is fenced, logged and provably
    not after T (the audit log is re-checked, and `verify_future_invariance` rebuilds the state on a store whose future has
    been scrambled and demands an identical digest);
  * the AUDITOR reads a second Guard bound to a later date (the outcome has matured); everything it reads that was not
    available at T is recorded as `future_information_used_by_auditor` and can never flow back into the state.

The classification is hindsight. A report is a MATURED_RESEARCH_STATE object; it reaches any consumer only through
`ClassificationGate` (MaturedRecord.gate(now), the same-year rerun rule, engine.learning.trader_view.assert_trader_safe)
and `refuse_feedback` rejects it wherever it appears as a feature, label or column outside that gate.

Builds on: engine.pit (Guard, PITStore, Calendar, AuditLog, scrambled stores), engine.livesim.Feed's data contract
(`store_from_feed_data`), engine.learning.situation (SituationBuilder / CrossSection for the identity-free situation at T),
engine.learning.curator (era_similarity for memory relevance), engine.learning.trader_view (safety scan), engine.learning.core
(Provenance, Confidence, FirewallBreach) and engine.research.core (Knowability, Availability, MaturedRecord). Knowability
labels from R02 are duck-typed through `KnowabilityClassifier`: any callable(evidence bundle) -> (Knowability, rationale).

Public entries: `step(store, events, now, ...)` (list of reports), `run_batch(...)` (streaming, checkpointed), `ClassificationGate`.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from engine import pit
from engine.learning.core import (Confidence, FirewallBreach, Provenance, _StrEnum, as_date, canonical_json,
                                  current_code_hash, stable_hash)
from engine.learning.curator import era_similarity, robust_scale
from engine.learning.situation import CrossSection, MacroObs, SituationBuilder, SituationConfig, sic_family
from engine.learning.trader_view import find_violations
from engine.research.core import Availability, Knowability, MaturedRecord, MoveCategory, Namespace


# ------------------------------------------------------------------------------------------------ vocabulary

class Domain(_StrEnum):
    """The ten information domains section 8 says the reconstruction must cover."""
    FEATURE = "FEATURE"
    PATTERN = "PATTERN"
    MEMORY = "MEMORY"
    MACRO = "MACRO"
    MARKET_STATE = "MARKET_STATE"
    EVENT = "EVENT"
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    TECHNICAL = "TECHNICAL"
    CROSS_SECTION = "CROSS_SECTION"


ALL_DOMAINS = tuple(Domain)
CORE_DOMAINS = (Domain.PRICE, Domain.VOLUME)        # without these there is no reconstruction at all -> DATA_FAILURE
EXPLANATORY = (Availability.KNOWN_ONLY_AFTER_EVENT, Availability.SIMULTANEOUS, Availability.UNAVAILABLE)


class Channel(_StrEnum):
    """What a piece of pre-event evidence speaks to: the size of a move, its direction, or when it happens."""
    MAGNITUDE = "MAGNITUDE"
    DIRECTION = "DIRECTION"
    TIMING = "TIMING"


@dataclasses.dataclass(frozen=True)
class SourceMap:
    """Names of the sources inside the PITStore and of their fields. Defaults follow engine.livesim.Feed's data contract."""
    prices: str = "prices"
    market: str = "market"
    events: str = "events"
    macro: str | None = "macro"
    schedule: str | None = None
    close: str = "Close"
    open_: str = "Open"
    high: str = "High"
    low: str = "Low"
    volume: str = "Volume"
    spy: str = "SPY"
    vix: str = "^VIX"
    vix3m: str = "^VIX3M"
    macro_value: str = "value"

    def price_fields(self) -> tuple[str, ...]:
        return (self.close, self.open_, self.high, self.low, self.volume)


@dataclasses.dataclass(frozen=True)
class CounterfactualConfig:
    """Every threshold is a documented STARTING value, not a tuned one (C63: no tuning in the foundation wave)."""
    lookback: int = 300                    # sessions of price history read per decision day
    min_history: int = 60                  # fewer usable sessions than this -> the technical structure is UNKNOWN
    beta_window: int = 120                 # past-only sessions for the market beta used in the attribution
    event_lookback_days: int = 30          # calendar days of company events read before T
    settle_sessions: int = 1               # the auditor waits this many sessions after the event window ends
    boundary_sessions: int = 0             # sessions after the fill session still counted as "around the boundary"
    strong: float = 0.65                   # combined evidence needed for PREDICTABLE
    moderate: float = 0.40                 # ... POTENTIALLY_PREDICTABLE
    weak: float = 0.18                     # ... WEAKLY_PREDICTABLE
    min_direction_for_predictable: float = 0.25
    min_pointer: float = 0.25              # a class above UNKNOWN needs at least one pointer this strong (sub-threshold ones are noise)
    min_supporting_domains: int = 2
    magnitude_discount: float = 0.85       # magnitude evidence is worth less than direction evidence for a directional move
    external_share: float = 0.6            # market + peer share of the move at which it is called EXTERNALLY_CAUSED
    idio_share: float = 0.5                # idiosyncratic share needed to call a post-T company cause the reason
    min_cause_relevance: float = 0.3
    uncertain_share_unknown: float = 0.5   # this share of relevant items with unprovable timestamps forces UNKNOWN
    min_coverage: float = 0.5
    unknown_conf_cap: float = 0.6
    memory_bandwidth: float = 1.0
    memory_min_similarity: float = 0.5
    memory_min_reliability: float = 0.3
    memory_top: int = 5
    max_event_items: int = 20
    return_tolerance: float = 0.02         # tolerated gap between the spec's realised return and the audited one
    top_k_per_day: int = 25
    sample_rate: float = 0.1
    null_alpha: float = 0.05               # a price pointer must beat this tail of the same-day peer null to count (W-04)
    null_min_peers: int = 20               # fewer peers than this cannot calibrate a null -> price pointers count for nothing
    null_max_peers: int = 120              # seeded cap on peers scored per (day, direction); cost is shared by the day's events
    event_weights: tuple[tuple[str, float], ...] = (
        ("EARN", 1.0), ("RESTATEMENT", 0.95), ("BANKRUPTCY", 0.95), ("DELIST_NOTICE", 0.9), ("LATE_FILING", 0.7),
        ("AUDITOR_CHANGE", 0.6), ("ACQ_DONE", 0.9), ("OFFERING", 0.8), ("SHELF", 0.5), ("UNREG_SALE", 0.5),
        ("AGREEMENT", 0.8), ("ACTIVIST", 0.7), ("ACTIVIST_AMEND", 0.5), ("PERIODIC", 0.3))
    default_event_weight: float = 0.4
    version: str = "cf2"

    def weight_of(self, kind) -> float:
        return dict(self.event_weights).get(str(kind).upper(), self.default_event_weight)

    def check(self) -> list[str]:
        errs = []
        if not (0.0 < self.weak < self.moderate < self.strong <= 1.0):
            errs.append("thresholds must satisfy 0 < weak < moderate < strong <= 1")
        for name in ("lookback", "min_history", "beta_window", "event_lookback_days", "top_k_per_day"):
            if int(getattr(self, name)) < 1:
                errs.append(f"{name} must be >= 1")
        if self.settle_sessions < 0 or self.boundary_sessions < 0:
            errs.append("settle_sessions / boundary_sessions cannot be negative")
        if not 0.0 <= self.sample_rate <= 1.0:
            errs.append("sample_rate outside [0, 1]")
        for name in ("external_share", "idio_share", "min_coverage", "unknown_conf_cap", "min_cause_relevance"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                errs.append(f"{name} outside [0, 1]")
        if self.min_history > self.lookback:
            errs.append("min_history exceeds lookback")
        if not 0.0 < self.null_alpha < 1.0 or self.null_min_peers < 2 or self.null_max_peers < self.null_min_peers:
            errs.append("null_alpha must be in (0, 1) and 2 <= null_min_peers <= null_max_peers")
        return errs


def _clean(x) -> float | None:
    """Finite float or None. NaN is never carried as a value (it would read as a number downstream)."""
    if x is None or isinstance(x, (bool, str)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _iso(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    t = pd.Timestamp(x)
    if pd.isna(t):
        return ""
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return str(t.normalize().date())


def _ramp(x: float | None, lo: float, hi: float) -> float:
    """0 at or below lo, 1 at or above hi, linear between. None reads as 0 (absent evidence is not evidence)."""
    if x is None or hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def noisy_or(strengths: Iterable[float]) -> float:
    """Combination of independent pieces of evidence: 1 - prod(1 - s). Empty -> 0."""
    p = 1.0
    for s in strengths:
        p *= 1.0 - min(1.0, max(0.0, float(s)))
    return 1.0 - p


# ------------------------------------------------------------------------------------------------ availability


def classify_availability(effective, available, decision_ts, event_start, cal: pit.Calendar | None = None,
                          boundary_sessions: int = 0, provenance_ok: bool = True) -> tuple[Availability, int | None]:
    """When did one piece of information exist for the market, relative to the decision boundary?

    decision_ts is the close at which the decision was made; event_start is the first session whose open is the fill (the
    strict next session). Rules, in order (section 7):
      * timestamp missing, unparseable, published before it happened, or provenance not established -> UNCERTAIN;
      * available on or before the decision close -> KNOWN_BEFORE_EVENT;
      * the fact itself only came into being after the boundary -> UNAVAILABLE (nothing legitimate could have revealed it);
      * published no later than the boundary session (+ boundary_sessions) -> SIMULTANEOUS;
      * otherwise the fact existed but was published later -> KNOWN_ONLY_AFTER_EVENT.
    Returns (class, sessions between decision and availability; negative = before)."""
    try:
        eff = pd.Timestamp(effective) if effective not in (None, "") else pd.NaT
        av = pd.Timestamp(available) if available not in (None, "") else pd.NaT
        t0, t1 = pd.Timestamp(decision_ts).normalize(), pd.Timestamp(event_start).normalize()
    except (ValueError, TypeError):
        return Availability.UNCERTAIN, None
    if not provenance_ok or pd.isna(eff) or pd.isna(av):
        return Availability.UNCERTAIN, None
    eff, av = eff.normalize(), av.normalize()
    if av < eff:
        return Availability.UNCERTAIN, None
    lag = _sessions_between(t0, av, cal)
    if av <= t0:
        return Availability.KNOWN_BEFORE_EVENT, lag
    if eff > t1:
        return Availability.UNAVAILABLE, lag
    edge = t1 if not boundary_sessions else _shift(t1, boundary_sessions, cal)
    if av <= edge:
        return Availability.SIMULTANEOUS, lag
    return Availability.KNOWN_ONLY_AFTER_EVENT, lag


def _shift(d: pd.Timestamp, n: int, cal: pit.Calendar | None) -> pd.Timestamp:
    if cal is None:
        return d + pd.offsets.BDay(n)
    return pd.Timestamp(cal.shift([d], n)[0])


def _sessions_between(a: pd.Timestamp, b: pd.Timestamp, cal: pit.Calendar | None) -> int:
    """Signed session count from a to b (b - a). Weekday arithmetic when no calendar is given."""
    if a == b:
        return 0
    lo, hi = (a, b) if a < b else (b, a)
    hol = cal._hol if cal is not None else []
    n = int(np.busday_count(lo.date(), hi.date(), holidays=hol))
    return n if b >= a else -n


# ------------------------------------------------------------------------------------------------ records


@dataclasses.dataclass(frozen=True)
class EventSpec:
    """One matured historical movement to assess. `decision_ts` is the close the decision would have been made at;
    `event_start` is the strict next session (the fill session); `event_end` the last session of the movement window.
    `realized_return` is auditor-side hindsight (fill open of event_start to close of event_end) and may be None (measured).
    `model_score` is the signed score the system itself produced at T (positive = up); it is a decision-time value and
    `score_asof` must not be after decision_ts."""
    event_id: str
    ticker: str
    decision_ts: str
    event_start: str
    event_end: str
    direction: int
    category: str = ""
    realized_return: float | None = None
    model_score: float | None = None
    score_asof: str = ""
    peers: tuple[str, ...] = ()

    @staticmethod
    def make(ticker: str, decision_ts, event_end, direction: int, cal: pit.Calendar | None = None, category="",
             realized_return=None, model_score=None, score_asof="", peers: Sequence[str] = ()) -> "EventSpec":
        t0, end = pd.Timestamp(decision_ts).normalize(), pd.Timestamp(event_end).normalize()
        t1 = pd.Timestamp((cal or pit.Calendar()).strict_next([t0])[0])
        eid = "CF" + stable_hash({"t": ticker, "d": str(t0.date()), "e": str(end.date()), "s": int(direction)}, 12)
        return EventSpec(eid, str(ticker), str(t0.date()), str(t1.date()), str(end.date()), int(direction), str(category),
                         _clean(realized_return), _clean(model_score), str(score_asof), tuple(peers))

    def validate(self) -> list[str]:
        errs = []
        try:
            t0, t1, t2 = (pd.Timestamp(x) for x in (self.decision_ts, self.event_start, self.event_end))
        except (ValueError, TypeError):
            return ["event dates unparseable"]
        if not self.ticker:
            errs.append("empty ticker")
        if self.direction not in (-1, 1):
            errs.append(f"direction {self.direction!r} must be +1 or -1")
        if not t0 < t1:
            errs.append("event_start must be strictly after the decision close (decide at the close, fill at the next open)")
        if t2 < t1:
            errs.append("event_end precedes event_start")
        if self.realized_return is not None and self.direction * self.realized_return < 0:
            errs.append("realized_return contradicts the stated direction")
        if self.model_score is not None and self.score_asof:
            if pd.Timestamp(self.score_asof) > t0:
                errs.append("model_score is dated after the decision close: a future score is a leak")
        if self.model_score is not None and not self.score_asof:
            errs.append("model_score given without score_asof (timestamp ambiguity)")
        return errs

    @property
    def real_year(self) -> int:
        return pd.Timestamp(self.decision_ts).year


@dataclasses.dataclass(frozen=True)
class InfoItem:
    """One piece of information with its timestamps and its availability class relative to one decision boundary."""
    domain: Domain
    name: str
    value: float | str | None
    effective: str
    available: str
    source: str
    availability: Availability
    lag_sessions: int | None = None
    in_model: bool = False
    note: str = ""

    @staticmethod
    def make(domain: Domain, name: str, value, effective, available, source: str, decision_ts, event_start,
             cal: pit.Calendar | None = None, in_model: bool = False, note: str = "", boundary_sessions: int = 0,
             provenance_ok: bool = True) -> "InfoItem":
        if isinstance(value, str):
            v: float | str | None = value
        else:
            v = _clean(value)
        eff, av = _iso(effective), _iso(available)
        klass, lag = classify_availability(eff, av, decision_ts, event_start, cal, boundary_sessions, provenance_ok)
        return InfoItem(Domain.parse(domain), str(name), v, eff, av, str(source), klass, lag, bool(in_model), note)

    @property
    def key(self) -> str:
        return f"{self.domain.value}.{self.name}"

    def ident(self) -> str:
        return stable_hash([self.domain, self.name, self.value, self.effective, self.available, self.source], 10)


@dataclasses.dataclass(frozen=True)
class KnowledgeState:
    """What the system could see at the decision close. Built only through a Guard bound to that close."""
    event_id: str
    decision_ts: str
    items: tuple[InfoItem, ...]
    situation_id: str
    situation_coverage: float
    guard_max_available: str
    guard_denied: int
    guard_reads: int
    log_digest: str

    @property
    def digest(self) -> str:
        return stable_hash([[i.key, i.value, i.effective, i.available] for i in self.items] + [self.situation_id], 16)

    def by_domain(self, d: Domain) -> tuple[InfoItem, ...]:
        return tuple(i for i in self.items if i.domain == d)

    def value(self, domain: Domain, name: str, default=None):
        for i in self.items:
            if i.domain == domain and i.name == name:
                return default if i.value is None else i.value
        return default

    def coverage(self) -> dict[str, int]:
        c = Counter(i.domain.value for i in self.items)
        return {d.value: int(c.get(d.value, 0)) for d in ALL_DOMAINS}

    def missing_domains(self) -> tuple[str, ...]:
        return tuple(d for d, n in self.coverage().items() if n == 0)

    def validate(self) -> list[str]:
        """Fail-closed audit: a state may hold nothing that was not available at the decision close."""
        errs = []
        t0 = pd.Timestamp(self.decision_ts)
        for i in self.items:
            if i.availability != Availability.KNOWN_BEFORE_EVENT:
                errs.append(f"{i.key}: availability {i.availability.value} inside a knowledge state")
            if i.available and pd.Timestamp(i.available) > t0:
                errs.append(f"{i.key}: available {i.available} after the decision close {self.decision_ts}")
        if self.guard_max_available and pd.Timestamp(self.guard_max_available) > t0:
            errs.append(f"guard touched data available {self.guard_max_available} after {self.decision_ts}")
        if self.guard_denied:
            errs.append(f"{self.guard_denied} guard read(s) were denied while building the state")
        return errs


@dataclasses.dataclass(frozen=True)
class FutureUse:
    """A piece of post-decision information the AUDITOR used, and what for. Never allowed inside a KnowledgeState."""
    item: InfoItem
    used_for: str                        # outcome | attribution | cause | later_knowledge | timing
    days_after_decision: int
    relevance: float = 0.0               # 0..1 how plausibly it explains the move (0 for pure outcome measurement)


@dataclasses.dataclass(frozen=True)
class Evidence:
    """One pre-event piece of evidence that pointed toward the move that followed."""
    domain: Domain
    name: str
    strength: float
    channel: Channel
    agrees: bool | None                  # None = speaks to size/timing only
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class MoveAttribution:
    """Split of the realised move into market, peer and idiosyncratic parts (past-only beta; auditor-side return)."""
    realized: float | None
    beta: float | None
    beta_obs: int
    market_return: float | None
    market_component: float | None
    peer_return: float | None
    peer_component: float | None
    idiosyncratic: float | None
    market_share: float
    peer_share: float
    idio_share: float
    complete: bool

    def check(self) -> list[str]:
        errs = []
        for n in ("market_share", "peer_share", "idio_share"):
            v = getattr(self, n)
            if not 0.0 <= v <= 1.0 + 1e-9:
                errs.append(f"{n}={v} outside [0,1]")
        if self.complete and abs(self.market_share + self.peer_share + self.idio_share - 1.0) > 1e-6:
            errs.append("attribution shares do not sum to 1")
        return errs


@dataclasses.dataclass(frozen=True)
class ClassificationConfidence:
    """Confidence that the LABEL is right (not confidence in any prediction). Components stay separate (contract C62 s.11)."""
    overall: float
    provenance_certainty: float
    coverage: float
    margin: float
    attribution_completeness: float
    hindsight_dependence: float          # share of the decisive evidence that needed the outcome to be interpreted
    caps: tuple[str, ...] = ()

    def check(self) -> list[str]:
        return [f"{f.name}={getattr(self, f.name)} outside [0,1]" for f in dataclasses.fields(self)
                if f.name != "caps" and not 0.0 <= float(getattr(self, f.name)) <= 1.0]

    def as_confidence(self) -> Confidence:
        """The shared C62 dimensions: truth = overall; context = coverage; reliability = provenance. Usefulness and transfer
        are NOT measured here, so they stay None (untested), never 0."""
        return Confidence(truth=self.overall, context=self.coverage, current_reliability=self.provenance_certainty,
                          failure_risk=1.0 - self.overall)


@dataclasses.dataclass(frozen=True)
class CounterfactualReport:
    """The section-8 research report for one event. A MATURED_RESEARCH_STATE object: hindsight by construction."""
    event: EventSpec
    knowledge_state_at_decision: KnowledgeState
    future_information_used_by_auditor: tuple[FutureUse, ...]
    information_that_would_have_been_available: tuple[InfoItem, ...]
    information_that_was_unavailable: tuple[InfoItem, ...]
    evidence: tuple[Evidence, ...]
    attribution: MoveAttribution
    knowability: Knowability
    confidence_in_classification: ClassificationConfidence
    scores: Mapping[str, float]
    uncertain_items: tuple[InfoItem, ...]
    auditor_as_of: str
    matured_at: str
    notes: tuple[str, ...] = ()
    selection: str = "all"
    weight: float = 1.0
    code_hash: str = ""
    namespace: Namespace = Namespace.MATURED_RESEARCH

    @property
    def report_id(self) -> str:
        return "R" + stable_hash([self.event.event_id, self.knowability, self.knowledge_state_at_decision.digest,
                                  self.auditor_as_of, self.notes], 14)

    def provenance(self) -> Provenance:
        return Provenance(created_real=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), learned_at=self.matured_at,
                          code_hash=self.code_hash or current_code_hash(), outcomes_seen_through=self.auditor_as_of,
                          parents=(self.event.event_id,))

    def validate(self) -> list[str]:
        errs = self.event.validate() + self.knowledge_state_at_decision.validate() + self.attribution.check()
        errs += self.confidence_in_classification.check()
        t0 = pd.Timestamp(self.event.decision_ts)
        if pd.Timestamp(self.auditor_as_of) <= pd.Timestamp(self.event.event_end):
            errs.append("auditor_as_of is not after the event window: the outcome has not matured")
        for f in self.future_information_used_by_auditor:
            if f.item.availability == Availability.KNOWN_BEFORE_EVENT and f.used_for != "later_knowledge":
                errs.append(f"future-use item {f.item.key} is classed as known before the event")
        for i in self.information_that_would_have_been_available:
            if i.availability != Availability.KNOWN_BEFORE_EVENT:
                errs.append(f"would-have-been-available item {i.key} is {i.availability.value}")
        for i in self.information_that_was_unavailable:
            if i.availability == Availability.KNOWN_BEFORE_EVENT:
                errs.append(f"unavailable item {i.key} was in fact known before the event")
        if self.knowability == Knowability.PREDICTABLE and not self.information_that_would_have_been_available:
            errs.append("PREDICTABLE with no available evidence")
        if self.knowability == Knowability.UNKNOWN and self.confidence_in_classification.overall > 0.9:
            errs.append("UNKNOWN labelled with near-certainty")
        if pd.Timestamp(self.matured_at) <= t0:
            errs.append("matured_at is not after the decision close")
        return errs

    def to_dict(self) -> dict:
        d = {
            "report_id": self.report_id, "event": dataclasses.asdict(self.event),
            "knowledge_state_at_decision": {
                "decision_ts": self.knowledge_state_at_decision.decision_ts, "digest": self.knowledge_state_at_decision.digest,
                "situation_id": self.knowledge_state_at_decision.situation_id,
                "situation_coverage": self.knowledge_state_at_decision.situation_coverage,
                "coverage": self.knowledge_state_at_decision.coverage(),
                "missing_domains": list(self.knowledge_state_at_decision.missing_domains()),
                "guard_max_available": self.knowledge_state_at_decision.guard_max_available,
                "guard_reads": self.knowledge_state_at_decision.guard_reads,
                "guard_denied": self.knowledge_state_at_decision.guard_denied,
                "items": [dataclasses.asdict(i) for i in self.knowledge_state_at_decision.items]},
            "future_information_used_by_auditor": [{"item": dataclasses.asdict(f.item), "used_for": f.used_for,
                                                    "days_after_decision": f.days_after_decision, "relevance": f.relevance}
                                                   for f in self.future_information_used_by_auditor],
            "information_that_would_have_been_available": [dataclasses.asdict(i) for i in self.information_that_would_have_been_available],
            "information_that_was_unavailable": [dataclasses.asdict(i) for i in self.information_that_was_unavailable],
            "evidence": [dataclasses.asdict(e) for e in self.evidence], "attribution": dataclasses.asdict(self.attribution),
            "knowability": self.knowability.value, "confidence_in_classification": dataclasses.asdict(self.confidence_in_classification),
            "scores": dict(self.scores), "uncertain_items": [dataclasses.asdict(i) for i in self.uncertain_items],
            "auditor_as_of": self.auditor_as_of, "matured_at": self.matured_at, "notes": list(self.notes),
            "selection": self.selection, "weight": self.weight, "code_hash": self.code_hash, "namespace": self.namespace.value}
        return json.loads(canonical_json(d))

    def short(self) -> dict:
        """One flat row per report: the only shape the streaming runner keeps in memory."""
        ks = self.knowledge_state_at_decision
        return {"report_id": self.report_id, "event_id": self.event.event_id, "year": self.event.real_year,
                "category": self.event.category, "direction": self.event.direction,
                "knowability": self.knowability.value, "confidence": self.confidence_in_classification.overall,
                "n_available": len(self.information_that_would_have_been_available),
                "n_unavailable": len(self.information_that_was_unavailable), "n_future_used": len(self.future_information_used_by_auditor),
                "n_uncertain": len(self.uncertain_items), "coverage_domains": sum(1 for v in ks.coverage().values() if v),
                "market_share": self.attribution.market_share, "idio_share": self.attribution.idio_share,
                "combined": float(self.scores.get("combined", 0.0)), "selection": self.selection, "weight": self.weight,
                "matured_at": self.matured_at}

    def matured_record(self) -> MaturedRecord:
        """The research-world fact. It reaches a consumer only through record.gate(now)."""
        return MaturedRecord(self.report_id, self.matured_at, self._label_payload(), self.provenance())

    def _label_payload(self) -> dict:
        return {"knowability": self.knowability.value, "confidence": self.confidence_in_classification.overall,
                "category": self.event.category, "direction": self.event.direction,
                "n_available": len(self.information_that_would_have_been_available),
                "n_unavailable": len(self.information_that_was_unavailable)}


# ------------------------------------------------------------------------------------------------ store from the Feed contract


def store_from_feed_data(data: Sequence, srcs: SourceMap | None = None, macro: pd.DataFrame | None = None,
                         macro_lag: int | dict = 1, calendar: pit.Calendar | None = None) -> pit.PITStore:
    """A PITStore over exactly the tuple engine.livesim.Feed takes as `data=(stocks, market, events, insider, sic)`, so the
    research side reads the same shapes the Test path uses. `events` needs ticker, kind and a UTC `accepted` timestamp: a
    filing accepted before 16:00 New York time is public the same session, one accepted after the close the next session
    (RecordSource derives that from `timing`). Real (undisguised) tickers belong here: this is the trusted side."""
    srcs = srcs or SourceMap()
    stocks, market, events = data[0], data[1], data[2]
    if "Close" not in stocks:
        raise ValueError("stocks must map field -> DataFrame(date x ticker) and include Close")
    cal = calendar or pit.Calendar(stocks["Close"].index)
    store = pit.PITStore(cal)
    store.add_prices(srcs.prices, {k: v for k, v in stocks.items()})
    store.add_prices(srcs.market, {k: v for k, v in market.items()})
    ev = events.copy()
    if len(ev):
        acc = pd.to_datetime(ev["accepted"], utc=True).dt.tz_convert("America/New_York")
        ev["effective"] = acc.dt.tz_localize(None).dt.normalize()
        ev["timing"] = np.where(acc.dt.hour < 16, "bmo", "amc")
    else:
        ev = pd.DataFrame({"ticker": [], "kind": [], "effective": pd.to_datetime([]), "timing": []})
    store.add_events(srcs.events, ev[["ticker", "kind", "effective", "timing"]], effective="effective", timing="timing")
    if macro is not None and srcs.macro:
        store.add_macro(srcs.macro, macro, lag=macro_lag)
    return store


# ------------------------------------------------------------------------------------------------ providers


@dataclasses.dataclass(frozen=True)
class Providers:
    """The parts of the system whose state is reconstructed. All optional; a missing provider leaves its domain empty and
    the report says so (missing domains lower coverage and confidence, they are never invented)."""
    feature_fn: Callable[[pit.Guard, str, pd.Timestamp], Mapping[str, Any]] | None = None
    patterns: Sequence[Any] = ()             # duck: pattern_id, name, effect, p_real, provenance, fires(features) -> bool
    memory: Any = None                       # duck: memories(matured_before=None) -> items with mem_id/context/lean/...
    sic: Mapping[str, Any] | None = None
    classifier: Callable[[Mapping[str, Any]], tuple[Any, str]] | None = None      # R02 KnowabilityClassifier duck
    situation: SituationBuilder | None = None


def pattern_problems(p: Any) -> list[str]:
    """What is missing from a pattern-like object (so a bad pattern is refused loudly rather than silently ignored)."""
    errs = [f"pattern missing {a}" for a in ("pattern_id", "effect", "provenance", "fires") if not hasattr(p, a)]
    if not errs and not callable(p.fires):
        errs.append("pattern.fires is not callable")
    return errs


# ------------------------------------------------------------------------------------------------ the shared day snapshot


def _log_digest(log: pit.AuditLog) -> str:
    return log._digests[-1][:16] if getattr(log, "_digests", None) else ""


@dataclasses.dataclass
class DaySnapshot:
    """Everything read once per decision date and shared by every event of that date (mapping design risk 3: the real cost
    is the per-event reconstruction, so the market-wide work is done once). Built only through a Guard bound to the close."""
    decision_ts: pd.Timestamp
    event_start: pd.Timestamp
    guard: pit.Guard
    log: pit.AuditLog
    frames: dict[str, pd.DataFrame]
    market: dict[str, pd.DataFrame]
    xs: pd.DataFrame
    cs: CrossSection
    market_row: dict[str, float]
    macro_rows: pd.DataFrame
    macro_obs: list[MacroObs]
    events: dict[str, pd.DataFrame]
    schedule: pd.DataFrame
    members: list[str]
    price_lag: int
    cal: pit.Calendar
    srcs: SourceMap
    ev_effective: str = "effective"
    ev_key: str = "ticker"

    def bar_available(self, bar_date) -> pd.Timestamp:
        d = pd.Timestamp(bar_date).normalize()
        return d if not self.price_lag else pd.Timestamp(self.cal.shift([d], self.price_lag)[0])

    def series(self, ticker: str, field_: str) -> pd.Series:
        f = self.frames.get(field_)
        if f is None or ticker not in f.columns:
            return pd.Series(dtype=float)
        return f[ticker].dropna()


def _cross_section_table(frames: Mapping[str, pd.DataFrame], srcs: SourceMap) -> pd.DataFrame:
    C, V = frames.get(srcs.close), frames.get(srcs.volume)
    if C is None or not len(C):
        return pd.DataFrame()
    n = len(C)
    R = C.pct_change(fill_method=None)
    xs = pd.DataFrame(index=C.columns)
    xs["r1"] = R.iloc[-1]
    for k in (5, 20, 60):
        xs[f"r{k}"] = C.iloc[-1] / C.iloc[-1 - k] - 1.0 if n > k else np.nan
    xs["vol20"] = R.iloc[-20:].std() if n > 21 else np.nan
    xs["max20"] = R.iloc[-20:].max() if n > 21 else np.nan
    if V is not None and len(V):
        dv = (C * V.reindex_like(C)).iloc[-20:].mean()
        xs["log_dv"] = np.log(dv.where(dv > 0))
        xs["vol_surge5"] = (V.iloc[-5:].mean() / V.iloc[-65:-5].mean().where(lambda s: s > 0)) if n >= 30 else np.nan
    valid = C.iloc[-1].notna()
    return xs.loc[valid].replace([np.inf, -np.inf], np.nan)


def _market_row(mk: Mapping[str, pd.DataFrame], xs: pd.DataFrame, srcs: SourceMap) -> dict[str, float]:
    """The m_* market-context columns (panel convention), all from data visible at the close."""
    row: dict[str, float] = {}
    C = mk.get(srcs.close)
    if C is not None and srcs.spy in C.columns:
        s = C[srcs.spy].dropna()
        if len(s) > 200:
            row["m_spy_ma200"] = float(s.iloc[-1] / s.iloc[-200:].mean() - 1.0)
        if len(s) > 50:
            row["m_spy_ma50"] = float(s.iloc[-1] / s.iloc[-50:].mean() - 1.0)
        if len(s) > 5:
            row["m_spy_r5"] = float(s.iloc[-1] / s.iloc[-6] - 1.0)
    if C is not None and srcs.vix in C.columns:
        v = C[srcs.vix].dropna()
        if len(v):
            row["m_vix"] = float(v.iloc[-1])
        if len(v) > 5 and v.iloc[-6] > 0:
            row["m_vix_chg5"] = float(v.iloc[-1] / v.iloc[-6] - 1.0)
        if srcs.vix3m in C.columns:
            v3 = C[srcs.vix3m].dropna()
            if len(v3) and v3.iloc[-1] > 0 and len(v):
                row["m_vix_term"] = float(v.iloc[-1] / v3.iloc[-1])
    if len(xs) >= 20:
        if xs["r5"].notna().sum() >= 20:
            row["m_breadth"] = float((xs["r5"].dropna() > 0).mean())
        if xs["r1"].notna().sum() >= 20:
            row["m_dispersion"] = float(xs["r1"].dropna().std())
    return {k: v for k, v in row.items() if math.isfinite(v)}


def build_snapshot(store: pit.PITStore, decision_ts, srcs: SourceMap | None = None, cfg: CounterfactualConfig | None = None,
                   log: pit.AuditLog | None = None) -> DaySnapshot:
    """Read the market once through a Guard bound to `decision_ts`. Every read is fenced and logged; a request past the
    close raises pit.LookAheadError here, before any state exists."""
    srcs, cfg = srcs or SourceMap(), cfg or CounterfactualConfig()
    errs = cfg.check()
    if errs:
        raise ValueError("CounterfactualConfig invalid: " + "; ".join(errs))
    t0 = pd.Timestamp(decision_ts).normalize()
    log = log if log is not None else pit.AuditLog()
    g = store.view(t0, log=log)
    psrc = store.source(srcs.prices)
    frames = {f: g.wide(srcs.prices, f, lookback=cfg.lookback) for f in srcs.price_fields() if f in psrc.fields()}
    msrc = store.source(srcs.market)
    mk = {f: g.wide(srcs.market, f, lookback=cfg.lookback) for f in (srcs.close,) if f in msrc.fields()}
    xs = _cross_section_table(frames, srcs)
    cs = CrossSection(xs[[c for c in ("vol20", "log_dv", "max20", "vol_surge5", "r5", "r20") if c in xs.columns]] if len(xs) else None)
    macro_rows, obs = pd.DataFrame(), []
    if srcs.macro and srcs.macro in store.names():
        macro_rows = g.records(srcs.macro, latest=True)
        if len(macro_rows):
            src = store.source(srcs.macro)
            latest = macro_rows.sort_values(src.effective).groupby(src.key).tail(1)
            for _, r in latest.iterrows():
                v = _clean(r.get(srcs.macro_value))
                if v is not None:
                    obs.append(MacroObs(str(r[src.key]), v, pd.Timestamp(r[src.effective]).date(), pd.Timestamp(r["available"]).date()))
    esrc = store.source(srcs.events)
    ev = g.records(srcs.events, lookback_days=cfg.event_lookback_days)
    by_t = {t: d for t, d in ev.groupby(esrc.key)} if len(ev) else {}
    sched = pd.DataFrame()
    if srcs.schedule and srcs.schedule in store.names():
        sched = g.records(srcs.schedule)
    members = g.members() if store.listings is not None else [str(c) for c in xs.index]
    return DaySnapshot(t0, pd.Timestamp(store.cal.strict_next([t0])[0]), g, log, frames, mk, xs, cs, _market_row(mk, xs, srcs),
                       macro_rows, obs, by_t, sched, members, psrc.lag, store.cal, srcs, esrc.effective, esrc.key)


# ------------------------------------------------------------------------------------------------ technical structure


def technical_state(snap: DaySnapshot, ticker: str, cfg: CounterfactualConfig) -> tuple[dict[str, float], str]:
    """Price / volume / technical measures for one ticker from data visible at the close. Returns (values, last bar date).
    Anything that needs more history than exists is left out (UNKNOWN), never zero-filled."""
    s = snap.srcs
    c, o = snap.series(ticker, s.close), snap.series(ticker, s.open_)
    h, l, v = snap.series(ticker, s.high), snap.series(ticker, s.low), snap.series(ticker, s.volume)
    if len(c) < 2:
        return {}, ""
    last = str(pd.Timestamp(c.index[-1]).date())
    out: dict[str, float] = {"close": float(c.iloc[-1])}
    n = len(c)
    ret = c.pct_change(fill_method=None).dropna()
    for k in (1, 5, 20, 60):
        if n > k:
            out[f"r{k}"] = float(c.iloc[-1] / c.iloc[-1 - k] - 1.0)
    if n >= cfg.min_history:
        out["vol20"] = float(ret.iloc[-20:].std())
        out["vol60"] = float(ret.iloc[-60:].std())
        if out["vol60"] > 0:
            out["vol_ratio"] = out["vol20"] / out["vol60"]
        out["max20"], out["min20"] = float(ret.iloc[-20:].max()), float(ret.iloc[-20:].min())
        out["dist_ma50"] = float(c.iloc[-1] / c.iloc[-50:].mean() - 1.0)
        out["skew60"] = float(ret.iloc[-60:].skew())
        out["dist_52wh"] = float(c.iloc[-1] / c.iloc[-252:].max() - 1.0)
        up, dn = ret.clip(lower=0).iloc[-14:].mean(), (-ret.clip(upper=0)).iloc[-14:].mean()
        out["rsi14"] = float(100.0 - 100.0 / (1.0 + up / dn)) if dn > 0 else 100.0
        streak = 0
        for x in ret.iloc[::-1]:
            if (x > 0) == (ret.iloc[-1] > 0) and x != 0:
                streak += 1
            else:
                break
        out["streak"] = float(streak if ret.iloc[-1] > 0 else -streak)
    if n >= 200:
        out["dist_ma200"] = float(c.iloc[-1] / c.iloc[-200:].mean() - 1.0)
    if len(h) >= 20 and len(l) >= 20:
        rng = ((h - l) / c.reindex(h.index)).dropna()
        if len(rng) >= 20:
            out["atr_pct"] = float(rng.iloc[-14:].mean())
            out["range_compress"] = float(rng.iloc[-5:].mean() / rng.iloc[-20:].mean()) if rng.iloc[-20:].mean() > 0 else np.nan
            hist = rng.rolling(5).mean().dropna().iloc[:-1]                  # past windows only
            if len(hist) >= 30:
                out["squeeze_rank"] = float((hist >= rng.iloc[-5:].mean()).mean())   # 1 = tightest 5-day range in its history
    if len(o) and len(c) > 1 and o.index[-1] == c.index[-1]:
        out["gap_today"] = float(o.iloc[-1] / c.iloc[-2] - 1.0)
    if len(v) >= 25:
        base = v.iloc[-21:-1].mean()
        if base > 0:
            out["vol_surge1"] = float(v.iloc[-1] / base)
        b5 = v.iloc[-25:-5].mean()
        if b5 > 0:
            out["vol_surge5"] = float(v.iloc[-5:].mean() / b5)
        out["volume"] = float(v.iloc[-1])
        dv = float((c.reindex(v.index) * v).iloc[-20:].mean())
        if dv > 0:
            out["log_dv"] = math.log(dv)
    return {k: x for k, x in out.items() if math.isfinite(x)}, last


# ------------------------------------------------------------------------------------------------ reconstruction


def _memory_items(snap: DaySnapshot, memory: Any, cfg: CounterfactualConfig) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
    """(known-at-T, learned-later) memories with their era similarity to the market state at T. A memory is known only if
    its outcome matured strictly before the decision close (C58); one that matured later is later knowledge."""
    if memory is None or not snap.market_row:
        return [], []
    allm = list(memory.memories())
    if not allm:
        return [], []
    keys = sorted(snap.market_row)
    scales = {k: robust_scale([m.context.get(k, math.nan) for m in allm]) for k in keys}
    known: list[tuple[Any, float]] = []
    later: list[tuple[Any, float]] = []
    for m in allm:
        sim, ok = era_similarity(m.context, snap.market_row, scales, cfg.memory_bandwidth)
        if not ok:
            continue
        (known if as_date(m.matured_at) < snap.decision_ts.date() else later).append((m, float(sim)))
    key = lambda x: (-x[1], str(x[0].mem_id))
    return sorted(known, key=key), sorted(later, key=key)


def reconstruct_state(snap: DaySnapshot, event: EventSpec, providers: Providers | None = None,
                      cfg: CounterfactualConfig | None = None) -> tuple[KnowledgeState, dict[str, Any]]:
    """The information state of the system at the event's decision close, domain by domain. Returns the KnowledgeState and
    a side dict (`later_patterns`, `later_memories`, `features`, `tech`) for the auditor; the side dict is not part of the
    state and nothing in it is allowed to flow back into it."""
    cfg, providers = cfg or CounterfactualConfig(), providers or Providers()
    if pd.Timestamp(event.decision_ts).normalize() != snap.decision_ts:
        raise FirewallBreach("snapshot was built for a different decision close than the event's")
    t0, t1, cal = snap.decision_ts, snap.event_start, snap.cal
    items: list[InfoItem] = []

    def add(domain, name, value, eff, av, source, note="", in_model=False):
        items.append(InfoItem.make(domain, name, value, eff, av, source, t0, t1, cal, in_model, note, cfg.boundary_sessions))

    tech, last = technical_state(snap, event.ticker, cfg)
    side: dict[str, Any] = {"tech": tech, "later_patterns": [], "later_memories": [], "features": {}}
    if tech:
        bav = snap.bar_available(last)
        for k in ("close", "r1", "r5", "r20", "r60", "gap_today"):
            if k in tech:
                add(Domain.PRICE, k, tech[k], last, bav, snap.srcs.prices)
        for k in ("volume", "vol_surge1", "vol_surge5", "log_dv"):
            if k in tech:
                add(Domain.VOLUME, k, tech[k], last, bav, snap.srcs.prices)
        for k in ("vol20", "vol60", "vol_ratio", "max20", "min20", "dist_ma50", "dist_ma200", "dist_52wh", "skew60", "rsi14",
                  "streak", "atr_pct", "range_compress", "squeeze_rank"):
            if k in tech:
                add(Domain.TECHNICAL, k, tech[k], last, bav, snap.srcs.prices)
        if event.ticker in snap.xs.index:
            for col in ("vol20", "log_dv", "max20", "vol_surge5", "r5", "r20"):
                p = snap.cs.pct(col, snap.xs.at[event.ticker, col]) if snap.cs.has(col) else None
                if p is not None:
                    add(Domain.CROSS_SECTION, f"{col}_rank", p, last, bav, snap.srcs.prices)
            add(Domain.CROSS_SECTION, "universe_size", float(len(snap.xs)), last, bav, snap.srcs.prices)
    for k, v in snap.market_row.items():
        add(Domain.MARKET_STATE, k, v, snap.decision_ts, snap.decision_ts, snap.srcs.market)
    if snap.macro_obs:
        for o in snap.macro_obs:
            add(Domain.MACRO, o.name, o.value, o.period_end, o.published, snap.srcs.macro or "macro")
    evs = snap.events.get(event.ticker)
    if evs is not None and len(evs):
        for _, r in evs.sort_values("available").tail(cfg.max_event_items).iterrows():
            kind = str(r["kind"]) if "kind" in r else "EVENT"
            add(Domain.EVENT, f"{kind}", cfg.weight_of(kind), r[snap.ev_effective], r["available"], snap.srcs.events)
    if len(snap.schedule):
        m = snap.schedule
        key = "ticker" if "ticker" in m.columns else None
        rows = m if key is None else m[(m[key] == event.ticker) | m[key].isna()]
        for _, r in rows.iterrows():
            if "kind" in r:
                add(Domain.EVENT, f"scheduled:{r['kind']}", 1.0, r["announced"] if "announced" in r else r["available"],
                    r["available"], snap.srcs.schedule or "schedule", note=f"due {_iso(r.get('due'))}" if "due" in r else "")
    feats: dict[str, Any] = {}
    if providers.feature_fn is not None:
        feats = dict(providers.feature_fn(snap.guard, event.ticker, t0))
        for k, v in sorted(feats.items()):
            if _clean(v) is not None:
                add(Domain.FEATURE, str(k), v, last or snap.decision_ts, last or snap.decision_ts, "features", in_model=True)
    side["features"] = feats
    for p in providers.patterns:
        bad = pattern_problems(p)
        if bad:
            raise ValueError("; ".join(bad))
        learned = p.provenance.learned_at
        try:
            fires = bool(p.fires({**tech, **snap.market_row, **feats}))
        except (KeyError, TypeError, ValueError):
            fires = False
        nm = str(getattr(p, "name", p.pattern_id))
        if p.provenance.could_exist_at(t0):
            add(Domain.PATTERN, nm, (float(p.effect) if fires else 0.0), learned, learned, "patterns", note=("fires" if fires else "quiet"),
                in_model=True)
        elif fires:
            side["later_patterns"].append((p, learned))
    known, later = _memory_items(snap, providers.memory, cfg)
    for m, sim in known[:cfg.memory_top]:
        add(Domain.MEMORY, f"{m.kind}:{m.mem_id}", float(m.lean) * sim, m.filed_date, m.matured_at, "memory", note=f"sim={sim:.3f}",
            in_model=True)
    side["memory_known"] = known
    side["later_memories"] = [(m, s) for m, s in later if s >= cfg.memory_min_similarity][:cfg.memory_top]
    sit_id, sit_cov = "", 0.0
    builder = providers.situation or SituationBuilder(SituationConfig())
    if tech:
        try:
            sit = builder.build(dict(tech), snap.market_row, t0,
                                cross_section=snap.cs, sic=(providers.sic or {}).get(event.ticker), macro=snap.macro_obs,
                                data_asof=last)
            sit_id, sit_cov = sit.situation_id, float(sit.coverage())
        except ValueError:
            sit_id, sit_cov = "", 0.0
    state = KnowledgeState(event.event_id, str(t0.date()), tuple(items), sit_id, sit_cov,
                           _iso(snap.log.max_avail_touched(t0)), len(snap.log.denied()), len(snap.log), _log_digest(snap.log))
    return state, side


# ------------------------------------------------------------------------------------------------ the auditor (hindsight side)


@dataclasses.dataclass(frozen=True)
class AuditResult:
    """Everything the auditor learned after T. It is kept apart from the KnowledgeState and never merged into it."""
    measured_return: float | None
    future_uses: tuple[FutureUse, ...]
    unavailable: tuple[InfoItem, ...]
    attribution: MoveAttribution
    notes: tuple[str, ...]
    auditor_as_of: str
    max_available_used: str
    log_digest: str


def auditor_as_of_for(event: EventSpec, now, cal: pit.Calendar | None = None, settle_sessions: int = 1) -> pd.Timestamp | None:
    """The date the auditor may read up to: `settle_sessions` after the window ends. None when the outcome has not matured
    against the REAL clock `now` (rule 27: research on a matured outcome only; a move ending on `now` is not yet known)."""
    end = pd.Timestamp(event.event_end).normalize()
    a = _shift(end, settle_sessions, cal) if settle_sessions else end
    try:
        if as_date(a) >= as_date(now):
            return None
    except (ValueError, TypeError):
        return None
    return a


def market_beta(snap: DaySnapshot, ticker: str, window: int) -> tuple[float | None, int]:
    """Past-only beta of the ticker to SPY over the last `window` sessions visible at T (never the event window)."""
    s = snap.srcs
    c = snap.series(ticker, s.close)
    m = snap.market.get(s.close)
    if m is None or s.spy not in m.columns or len(c) < 30:
        return None, 0
    ri, rm = c.pct_change(fill_method=None), m[s.spy].pct_change(fill_method=None)
    both = pd.concat([ri, rm], axis=1, keys=["i", "m"]).dropna().iloc[-window:]
    if len(both) < 30 or both["m"].var() <= 0:
        return None, len(both)
    return float(both["i"].cov(both["m"]) / both["m"].var()), len(both)


def attribute_move(realized: float | None, beta: float | None, beta_obs: int, market_ret: float | None,
                   peer_ret: float | None) -> MoveAttribution:
    """Split a realised move into market (beta x market return), peer (excess of the peer mean over the market, sector beta
    fixed at 1: an estimated sector beta over a short past is noise) and idiosyncratic. Shares are taken in the direction of
    the move: a part that opposed the move explains none of it. Incomplete inputs give an incomplete attribution, which the
    classifier treats as unknown rather than as idiosyncratic."""
    if realized is None or beta is None or market_ret is None:
        return MoveAttribution(realized, beta, beta_obs, market_ret, None, peer_ret, None, None, 0.0, 0.0, 1.0, False)
    mc = beta * market_ret
    pc = (peer_ret - market_ret) if peer_ret is not None else 0.0
    idio = realized - mc - pc
    sign = 1.0 if realized >= 0 else -1.0
    parts = [max(0.0, sign * x) for x in (mc, pc, idio)]
    tot = sum(parts)
    if tot <= 0.0:
        return MoveAttribution(realized, beta, beta_obs, market_ret, mc, peer_ret, pc, idio, 0.0, 0.0, 1.0, False)
    ms, ps, isr = (p / tot for p in parts)
    return MoveAttribution(realized, beta, beta_obs, market_ret, mc, peer_ret, pc if peer_ret is not None else None, idio, ms, ps, isr, True)


def _window_frames(g: pit.Guard, srcs: SourceMap, tickers: Sequence[str], t0: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    for f in (srcs.close, srcs.open_, srcs.high, srcs.low, srcs.volume):
        try:
            out[f] = g.wide(srcs.prices, f, tickers=list(tickers), start=t0, end=end)
        except KeyError:
            continue
    return out


def _window_return(fr: Mapping[str, pd.DataFrame], ticker: str, t1: pd.Timestamp, srcs: SourceMap) -> tuple[float | None, str]:
    """Fill at the strict next session's open, mark at the window's last close (the simulator's own rule)."""
    C, O = fr.get(srcs.close), fr.get(srcs.open_)
    if C is None or ticker not in C.columns or not len(C):
        return None, "no close in window"
    c = C[ticker].dropna()
    if len(c) < 2:
        return None, "window has no bars after the decision close"
    if O is not None and ticker in O.columns and t1 in O.index and math.isfinite(O.at[t1, ticker]) and O.at[t1, ticker] > 0:
        return float(c.iloc[-1] / O.at[t1, ticker] - 1.0), ""
    base = c.iloc[0]
    return (float(c.iloc[-1] / base - 1.0), "fill-open missing: measured from the decision close") if base > 0 else (None, "bad base price")


def _outcome_item(domain: Domain, name: str, value, when, available, source: str, t0, t1, cal, note="") -> InfoItem:
    """Outcome facts ARE the event: nothing legitimate could have revealed them, whatever their nominal dates say."""
    it = InfoItem.make(domain, name, value, when, available, source, t0, t1, cal, False, note)
    return dataclasses.replace(it, availability=Availability.UNAVAILABLE)


def audit_event(store: pit.PITStore, snap: DaySnapshot, event: EventSpec, state: KnowledgeState, side: Mapping[str, Any],
                as_of, providers: Providers | None = None, cfg: CounterfactualConfig | None = None) -> AuditResult:
    """Read the matured outcome through a SECOND guard bound to `as_of` (after the window) and classify everything it needs
    that was not available at T. The auditor may only read what existed by `as_of`; it does not see beyond the outcome
    either, so the report cannot depend on knowledge later than the window it studies."""
    cfg, providers = cfg or CounterfactualConfig(), providers or Providers()
    srcs, cal = snap.srcs, snap.cal
    t0, t1, end = snap.decision_ts, snap.event_start, pd.Timestamp(event.event_end).normalize()
    as_of = pd.Timestamp(as_of).normalize()
    if as_of <= end:
        raise FirewallBreach("auditor_as_of must be after the event window")
    if as_of <= t0:
        raise FirewallBreach("auditor_as_of must be after the decision close")
    logA = pit.AuditLog()
    gA = store.view(as_of, log=logA)
    uses: list[FutureUse] = []
    unavailable: list[InfoItem] = []
    notes: list[str] = []
    peers = list(event.peers) or _peers_of(event.ticker, snap, providers)
    fr = _window_frames(gA, srcs, [event.ticker] + peers, t0, end)
    measured, why = _window_return(fr, event.ticker, t1, srcs)
    if why:
        notes.append(why)
    realized = measured
    if event.realized_return is not None:
        if measured is not None and abs(event.realized_return - measured) > cfg.return_tolerance:
            notes.append(f"spec realized_return {event.realized_return:.4f} differs from audited {measured:.4f}")
        realized = measured if measured is not None else event.realized_return
    if realized is None:
        notes.append("realised move could not be measured")
    days = lambda d: int(_sessions_between(t0, pd.Timestamp(d).normalize(), cal))
    C, V = fr.get(srcs.close), fr.get(srcs.volume)
    if realized is not None and C is not None and len(C):
        item = _outcome_item(Domain.PRICE, "window_return", realized, end, end, srcs.prices, t0, t1, cal)
        uses.append(FutureUse(item, "outcome", days(end), 0.0))
        c = C[event.ticker].dropna()
        if len(c) > 1:
            path = c.iloc[1:] / c.iloc[0] - 1.0
            exc = float(path.max() if event.direction > 0 else path.min())
            uses.append(FutureUse(_outcome_item(Domain.PRICE, "max_excursion", exc, path.idxmax() if event.direction > 0 else path.idxmin(),
                                                end, srcs.prices, t0, t1, cal), "outcome", days(end), 0.0))
    if V is not None and event.ticker in V.columns and len(V) > 1:
        base = snap.series(event.ticker, srcs.volume).iloc[-21:-1].mean()
        v = V[event.ticker].dropna().iloc[1:]
        if base and base > 0 and len(v):
            uses.append(FutureUse(_outcome_item(Domain.VOLUME, "window_volume_surge", float(v.mean() / base), end, end, srcs.prices, t0, t1, cal),
                                  "outcome", days(end), 0.0))
    mk_ret = _market_window_return(gA, srcs, t0, t1, end)
    peer_ret = _peer_window_return(fr, peers, t1, srcs)
    if mk_ret is not None:
        uses.append(FutureUse(_outcome_item(Domain.MARKET_STATE, "market_window_return", mk_ret, end, end, srcs.market, t0, t1, cal),
                              "attribution", days(end), 0.0))
    if peer_ret is not None:
        uses.append(FutureUse(_outcome_item(Domain.CROSS_SECTION, "peer_window_return", peer_ret, end, end, srcs.prices, t0, t1, cal,
                                            note=f"{len(peers)} peers"), "attribution", days(end), 0.0))
    elif not peers:
        notes.append("no peers available: sector share cannot be measured")
    beta, nobs = market_beta(snap, event.ticker, cfg.beta_window)
    att = attribute_move(realized, beta, nobs, mk_ret, peer_ret)
    esrc = store.source(srcs.events)
    try:
        post = gA.records(srcs.events, tickers=[event.ticker], start=t0 - pd.Timedelta(days=cfg.event_lookback_days), end=end)
    except pit.LookAheadError:
        post = pd.DataFrame()
    if len(post):
        post = post[post["available"] > t0]
    for _, r in post.iterrows():
        kind = str(r["kind"]) if "kind" in r else "EVENT"
        it = InfoItem.make(Domain.EVENT, kind, cfg.weight_of(kind), r[esrc.effective], r["available"], srcs.events, t0, t1, cal,
                           boundary_sessions=cfg.boundary_sessions)
        rel = cfg.weight_of(kind) * _timing_factor(it, t1, end)
        uses.append(FutureUse(it, "cause", days(it.available) if it.available else 0, rel))
        if it.availability in EXPLANATORY:
            unavailable.append(it)
    if srcs.macro and srcs.macro in store.names():
        for it, rel in _macro_releases(gA, store, srcs, t0, t1, end, cal, cfg):
            uses.append(FutureUse(it, "cause", days(it.available) if it.available else 0, rel))
            if it.availability in EXPLANATORY:
                unavailable.append(it)
    for p, learned in side.get("later_patterns", ()):
        it = InfoItem.make(Domain.PATTERN, str(getattr(p, "name", p.pattern_id)), float(p.effect), learned, learned, "patterns", t0, t1, cal,
                           note="learned after the decision; would have fired")
        pr = _clean(getattr(p, "p_real", None))
        uses.append(FutureUse(it, "later_knowledge", days(learned), 0.0 if pr is None else min(1.0, pr) * 0.5))
        unavailable.append(it)
    for m, sim in side.get("later_memories", ()):
        it = InfoItem.make(Domain.MEMORY, f"{m.kind}:{m.mem_id}", float(m.lean) * sim, m.filed_date, m.matured_at, "memory", t0, t1, cal,
                           note=f"matured after the decision; sim={sim:.3f}")
        uses.append(FutureUse(it, "later_knowledge", days(m.matured_at), 0.0))
        unavailable.append(it)
    used_av = [pd.Timestamp(u.item.available) for u in uses if u.item.available]
    return AuditResult(realized, tuple(uses), tuple(unavailable), att, tuple(notes), str(as_of.date()),
                       str(max(used_av).date()) if used_av else "", _log_digest(logA))


def _peers_of(ticker: str, snap: DaySnapshot, providers: Providers, limit: int = 15) -> list[str]:
    """Same coarse sector family (situation.sic_family), sorted for determinism; only names that trade at T."""
    sic = providers.sic or {}
    if ticker not in sic:
        return []
    fam = sic_family(sic[ticker])
    if fam == "unknown":
        return []
    live = set(snap.xs.index)
    same = sorted(t for t, code in sic.items() if t != ticker and t in live and sic_family(code) == fam)
    return same[:limit]


def _market_window_return(g: pit.Guard, srcs: SourceMap, t0, t1, end) -> float | None:
    try:
        C = g.wide(srcs.market, srcs.close, tickers=[srcs.spy], start=t0, end=end)
    except KeyError:
        return None
    if srcs.spy not in C.columns or len(C.dropna()) < 2:
        return None
    c = C[srcs.spy].dropna()
    try:
        O = g.wide(srcs.market, srcs.open_, tickers=[srcs.spy], start=t1, end=t1)
        base = float(O[srcs.spy].iloc[0]) if len(O) and math.isfinite(O[srcs.spy].iloc[0]) else float(c.iloc[0])
    except KeyError:
        base = float(c.iloc[0])
    return float(c.iloc[-1] / base - 1.0) if base > 0 else None


def _peer_window_return(fr: Mapping[str, pd.DataFrame], peers: Sequence[str], t1, srcs: SourceMap) -> float | None:
    rets = [r for r in (_window_return(fr, p, t1, srcs)[0] for p in peers) if r is not None]
    return float(np.mean(rets)) if len(rets) >= 3 else None


def _timing_factor(it: InfoItem, t1: pd.Timestamp, end: pd.Timestamp) -> float:
    """How plausibly a post-T record explains the move, from when it surfaced: at or just before the fill session it is the
    natural trigger; inside the window it may be cause or consequence; after the window it cannot be a cause."""
    if not it.effective:
        return 0.3
    e = pd.Timestamp(it.effective)
    if e > end:
        return 0.0
    if e <= t1:
        return 1.0
    span = max(1, int((end - t1).days))
    return float(0.4 + 0.3 * (1.0 - (e - t1).days / span))


def _macro_releases(g: pit.Guard, store: pit.PITStore, srcs: SourceMap, t0, t1, end, cal, cfg) -> list[tuple[InfoItem, float]]:
    """Macro observations released after T and inside the window, scored by how large the change from the previous known
    vintage was in robust units (an unremarkable release explains nothing)."""
    macro_name = srcs.macro
    if macro_name is None:
        return []
    src = store.source(macro_name)
    try:
        rows = g.records(macro_name, start=t0 - pd.Timedelta(days=400), end=end)
    except pit.LookAheadError:
        return []
    if not len(rows) or srcs.macro_value not in rows.columns:
        return []
    out = []
    for name, d in rows.sort_values(src.effective).groupby(src.key):
        vals = d[srcs.macro_value].astype(float).to_numpy()
        if len(vals) < 3:
            continue
        step = np.diff(vals)
        scale = robust_scale(list(step[:-1]))
        last = d.iloc[-1]
        if pd.Timestamp(last["available"]) <= t0:
            continue
        z = abs(step[-1]) / scale if scale > 0 else 0.0
        it = InfoItem.make(Domain.MACRO, str(name), float(vals[-1]), last[src.effective], last["available"], macro_name, t0, t1, cal,
                           note=f"change z={z:.2f}", boundary_sessions=cfg.boundary_sessions)
        out.append((it, _ramp(z, 1.0, 4.0) * 0.8))
    return out


# ------------------------------------------------------------------------------------------------ evidence (state + direction)


class StateReader(Protocol):
    """What collect_evidence reads of a state: a full KnowledgeState, or the price-only view a peer supplies."""

    def by_domain(self, d: Domain) -> tuple[InfoItem, ...]: ...

    def value(self, domain: Domain, name: str, default: Any = None) -> Any: ...


def collect_evidence(state: StateReader, event: EventSpec, cfg: CounterfactualConfig | None = None) -> list[Evidence]:
    """What in the knowledge state pointed toward this move. Reads ONLY the state and the event's direction/window (the
    direction is hindsight; it is used to ask 'did the evidence agree', never to add information to the state)."""
    cfg = cfg or CounterfactualConfig()
    d = float(event.direction)
    ev: list[Evidence] = []

    def add(dom, name, s, ch, agrees=None, detail=""):
        if s > 0.0:
            ev.append(Evidence(dom, name, float(min(1.0, s)), ch, agrees, detail))

    v = lambda dom, n: state.value(dom, n)
    T, V, P, X, M = Domain.TECHNICAL, Domain.VOLUME, Domain.PRICE, Domain.CROSS_SECTION, Domain.MARKET_STATE
    sq = v(T, "squeeze_rank")
    add(T, "range_squeeze", 0.6 * _ramp(sq, 0.7, 1.0), Channel.MAGNITUDE, None, f"squeeze_rank={sq}")
    vr = v(T, "vol_ratio")
    if vr is not None:
        add(T, "vol_contraction", 0.5 * _ramp(1.0 - vr, 0.1, 0.5), Channel.MAGNITUDE, None, f"vol_ratio={vr:.2f}")
        add(T, "vol_expansion", 0.4 * _ramp(vr - 1.0, 0.15, 0.8), Channel.MAGNITUDE, None, f"vol_ratio={vr:.2f}")
    d52, r20, dm50, rsi = v(T, "dist_52wh"), v(P, "r20"), v(T, "dist_ma50"), v(T, "rsi14")
    if d52 is not None and d > 0:
        add(T, "near_52w_high", 0.5 * _ramp(d52, -0.10, -0.005), Channel.DIRECTION, True, f"dist_52wh={d52:.3f}")
    if d52 is not None and d < 0 and d52 <= -0.30:
        add(T, "deep_below_high", 0.25 * _ramp(-d52, 0.30, 0.6), Channel.DIRECTION, True, f"dist_52wh={d52:.3f}")
    if r20 is not None and dm50 is not None:
        cont = d * r20
        add(T, "trend_continuation", 0.3 * _ramp(cont, 0.02, 0.15), Channel.DIRECTION, True, f"r20={r20:.3f}")
        add(T, "trend_opposes", 0.25 * _ramp(-cont, 0.08, 0.25), Channel.DIRECTION, False, f"r20={r20:.3f}")
    if rsi is not None:
        if d > 0 and rsi < 30:
            add(T, "oversold_reversal", 0.3 * _ramp(30 - rsi, 0, 20), Channel.DIRECTION, True, f"rsi14={rsi:.0f}")
        if d < 0 and rsi > 70:
            add(T, "overbought_reversal", 0.3 * _ramp(rsi - 70, 0, 20), Channel.DIRECTION, True, f"rsi14={rsi:.0f}")
    s5, s1 = v(V, "vol_surge5"), v(V, "vol_surge1")
    add(V, "volume_build", 0.5 * _ramp(s5, 1.2, 3.0), Channel.MAGNITUDE, None, f"vol_surge5={s5}")
    add(V, "volume_spike", 0.4 * _ramp(s1, 1.5, 4.0), Channel.MAGNITUDE, None, f"vol_surge1={s1}")
    r1, gap = v(P, "r1"), v(P, "gap_today")
    if r1 is not None and s1 is not None and d * r1 > 0:
        add(V, "volume_confirms_direction", 0.25 * _ramp(s1, 1.5, 3.0) * _ramp(abs(r1), 0.01, 0.05), Channel.DIRECTION, True, "")
    if gap is not None:
        add(P, "gap_into_close", 0.3 * _ramp(d * gap, 0.02, 0.08), Channel.DIRECTION, True, f"gap={gap:.3f}")
        add(P, "gap_against", 0.2 * _ramp(-d * gap, 0.03, 0.10), Channel.DIRECTION, False, f"gap={gap:.3f}")
    vrk, mrk = v(X, "vol20_rank"), v(X, "max20_rank")
    add(X, "high_vol_rank", 0.4 * _ramp(vrk, 0.8, 1.0), Channel.MAGNITUDE, None, f"vol20_rank={vrk}")
    add(X, "lottery_rank", 0.3 * _ramp(mrk, 0.85, 1.0), Channel.MAGNITUDE, None, f"max20_rank={mrk}")
    vix, chg = v(M, "m_vix"), v(M, "m_vix_chg5")
    add(M, "stress_regime", 0.3 * _ramp(vix, 24, 40), Channel.MAGNITUDE, None, f"vix={vix}")
    add(M, "vix_spike", 0.25 * _ramp(chg, 0.10, 0.40), Channel.MAGNITUDE, None, f"vix_chg5={chg}")
    for it in state.by_domain(Domain.EVENT):
        w = _clean(it.value) or 0.0
        age = max(0, -(it.lag_sessions or 0))
        decay = math.exp(-age / 10.0)
        if it.name.startswith("scheduled:"):
            add(Domain.EVENT, it.name, 0.5 * decay, Channel.TIMING, None, "a scheduled catalyst was known in advance")
        elif w >= 0.6:
            add(Domain.EVENT, it.name, 0.5 * w * decay, Channel.MAGNITUDE, None, f"known {it.name} filing {age} sessions before")
    for it in state.by_domain(Domain.PATTERN):
        eff = _clean(it.value) or 0.0
        if eff == 0.0:
            continue
        agree = eff * d > 0
        add(Domain.PATTERN, it.name, 0.6 * min(1.0, abs(eff) / 0.05) * 0.8, Channel.DIRECTION, agree, it.note)
    mem = [(it, _clean(it.value) or 0.0) for it in state.by_domain(Domain.MEMORY)]
    for it, sv in mem:
        if abs(sv) >= 0.05:
            add(Domain.MEMORY, it.name, 0.6 * min(1.0, abs(sv)), Channel.DIRECTION, sv * d > 0, it.note)
    if event.model_score is not None:
        sc = float(event.model_score)
        add(Domain.FEATURE, "model_score", 0.7 * min(1.0, abs(sc)), Channel.DIRECTION, sc * d > 0, f"model_score={sc:+.2f}")
    return ev


def score_evidence(ev: Sequence[Evidence], cfg: CounterfactualConfig | None = None) -> dict[str, float]:
    """Channel scores by noisy-or over the strongest piece per domain (correlated evidence inside one domain is not counted
    as independent), opposing direction evidence subtracted from agreeing, then combined."""
    cfg = cfg or CounterfactualConfig()

    def per_domain(items):
        best: dict[Domain, float] = {}
        for e in items:
            best[e.domain] = max(best.get(e.domain, 0.0), e.strength)
        return best

    mag = per_domain([e for e in ev if e.channel in (Channel.MAGNITUDE, Channel.TIMING) and e.agrees is not False])
    agree = per_domain([e for e in ev if e.channel == Channel.DIRECTION and e.agrees is True])
    oppose = per_domain([e for e in ev if e.channel == Channel.DIRECTION and e.agrees is False])
    m, a, o = noisy_or(mag.values()), noisy_or(agree.values()), noisy_or(oppose.values())
    dirn = max(0.0, a - o * (1.0 - a))
    combined = 1.0 - (1.0 - cfg.magnitude_discount * m) * (1.0 - dirn)
    domains = set(mag) | set(agree)
    strongest = max([e.strength for e in ev if e.agrees is not False] or [0.0])
    return {"magnitude": m, "direction": dirn, "opposing": o, "agreeing": a, "combined": combined,
            "n_supporting_domains": float(len(domains)), "strongest": float(strongest)}


# ------------------------------------------------------------------------------------------------ calibrated null (W-04)

PRICE_DOMAINS = (Domain.TECHNICAL, Domain.VOLUME, Domain.PRICE, Domain.CROSS_SECTION, Domain.MARKET_STATE)


class _PriceView:
    """The slice of a KnowledgeState that collect_evidence reads for a peer: price-derived domains only. A peer has no
    filings, patterns or memories attached, so the null it supplies is a null for PRICE pointers and nothing else."""

    def __init__(self, values: Mapping[tuple[Domain, str], float]):
        self._v = dict(values)

    def value(self, domain: Domain, name: str, default=None):
        return self._v.get((domain, name), default)

    def by_domain(self, domain: Domain) -> tuple:
        return ()


def _peer_view(snap: DaySnapshot, ticker: str, cfg: CounterfactualConfig) -> _PriceView | None:
    tech, _ = technical_state(snap, ticker, cfg)
    if not tech:
        return None
    vals: dict[tuple[Domain, str], float] = {}
    place = {Domain.PRICE: ("close", "r1", "r5", "r20", "r60", "gap_today"), Domain.VOLUME: ("volume", "vol_surge1", "vol_surge5", "log_dv"),
             Domain.TECHNICAL: ("vol20", "vol60", "vol_ratio", "max20", "min20", "dist_ma50", "dist_ma200", "dist_52wh", "skew60",
                                "rsi14", "streak", "atr_pct", "range_compress", "squeeze_rank")}
    for dom, names in place.items():
        for k in names:
            if k in tech:
                vals[(dom, k)] = tech[k]
    if ticker in snap.xs.index:
        for col in ("vol20", "log_dv", "max20", "vol_surge5", "r5", "r20"):
            p = snap.cs.pct(col, snap.xs.at[ticker, col]) if snap.cs.has(col) else None
            if p is not None:
                vals[(Domain.CROSS_SECTION, f"{col}_rank")] = p
    for k, v in snap.market_row.items():
        vals[(Domain.MARKET_STATE, k)] = v
    return _PriceView(vals)


def price_null_scores(snap: DaySnapshot, event: EventSpec, cfg: CounterfactualConfig | None = None) -> np.ndarray:
    """Combined price-pointer score of OTHER names on the same decision close, asked with the event's direction. This is what
    'a typical stock looked like that evening' scores, so a pointer only counts when it beats it. Deterministic: peers are a
    seeded sample keyed on the day. Cached on the snapshot per direction (the day's events share it). Reads the snapshot
    only, so it cannot see past the close."""
    cfg = cfg or CounterfactualConfig()
    cache = snap.__dict__.setdefault("_null_cache", {})
    key = (int(event.direction), cfg.null_max_peers)
    if key not in cache:
        peers = sorted(t for t in snap.frames.get(snap.srcs.close, pd.DataFrame()).columns if t != event.ticker)
        rng = np.random.default_rng(int(stable_hash(f"{snap.decision_ts.date()}|{key}", 8), 16) % (2 ** 32))
        if len(peers) > cfg.null_max_peers:
            peers = [peers[i] for i in sorted(rng.choice(len(peers), cfg.null_max_peers, replace=False))]
        probe = dataclasses.replace(event, model_score=None)
        out = []
        for t in peers:
            view = _peer_view(snap, t, cfg)
            if view is not None:
                out.append(score_evidence(collect_evidence(view, probe, cfg), cfg)["combined"])
        cache[key] = np.asarray(out, dtype=float)
    return cache[key]


def calibrate_pointers(snap: DaySnapshot, event: EventSpec, ev: Sequence[Evidence], scores: Mapping[str, float],
                       cfg: CounterfactualConfig | None = None) -> tuple[dict[str, float], list[str]]:
    """W-04. Price-derived pointers (technical / volume / price / cross-section / market state) point at a move only when their
    combined score beats the same-day peer null at `null_alpha` (one-sided permutation p, (1+#null>=obs)/(n+1)). Evidence that
    is not a price pattern (a filing, a learned pattern, a scheduled catalyst, a model score) keeps its own standing when it is
    at least min_pointer. When neither holds the pointers are treated as noise: combined and strongest are zeroed, exactly as
    the existing sub-threshold rule does, so the ladder falls to EXTERNAL / UNAVAILABLE / UNKNOWN. Too few peers to calibrate
    fails closed. Returns (scores with `null_p`/`combined_raw` added, notes)."""
    cfg = cfg or CounterfactualConfig()
    out = dict(scores)
    out["combined_raw"] = float(scores["combined"])
    price = [e for e in ev if e.domain in PRICE_DOMAINS]
    other = [e for e in ev if e.domain not in PRICE_DOMAINS and e.agrees is not False]
    other_strong = max((e.strength for e in other), default=0.0) >= cfg.min_pointer
    if scores["combined"] < cfg.weak or not ev:
        out["null_p"] = 1.0
        return out, []
    null = price_null_scores(snap, event, cfg)
    obs = score_evidence(price, cfg)["combined"]
    if len(null) < cfg.null_min_peers:
        p, note = 1.0, f"only {len(null)} peers to calibrate a null (< {cfg.null_min_peers}): price pointers cannot be shown to beat chance"
    else:
        p = (1.0 + float((null >= obs - 1e-12).sum())) / (len(null) + 1.0)
        note = f"price pointers combined {obs:.2f} vs {len(null)} same-day peers: p={p:.3f}"
    out["null_p"] = float(p)
    if p <= cfg.null_alpha or other_strong:
        return out, ([note] if p <= cfg.null_alpha else [])
    out["combined"], out["strongest"] = 0.0, 0.0
    return out, [note + f" > alpha {cfg.null_alpha}: treated as noise (a typical stock looks like this)"]


# ------------------------------------------------------------------------------------------------ classification


def _cause_items(audit: AuditResult, cfg: CounterfactualConfig) -> list[FutureUse]:
    return [u for u in audit.future_uses
            if u.used_for == "cause" and u.relevance >= cfg.min_cause_relevance and u.item.availability in EXPLANATORY]


def classify_knowability(state: KnowledgeState, scores: Mapping[str, float], audit: AuditResult, cfg: CounterfactualConfig,
                         relevant: Sequence[InfoItem] = ()) -> tuple[Knowability, list[str]]:
    """The ladder. Order matters and is the honesty contract: DATA_FAILURE before anything (a broken reconstruction must
    not be read as unknowability), evidence-based classes next, then external / unavailable only with a positive reason,
    and UNKNOWN whenever the cause is not identified. Returns (class, notes)."""
    notes: list[str] = []
    cov = state.coverage()
    covered = sum(1 for n in cov.values() if n)
    if any(cov[d.value] == 0 for d in CORE_DOMAINS):
        return Knowability.DATA_FAILURE, [f"core domain missing: {[d.value for d in CORE_DOMAINS if cov[d.value] == 0]}"]
    if covered / len(ALL_DOMAINS) < cfg.min_coverage - 1e-12:
        return Knowability.DATA_FAILURE, [f"only {covered}/{len(ALL_DOMAINS)} domains reconstructed"]
    if state.validate():
        return Knowability.DATA_FAILURE, ["knowledge state failed its own audit: " + state.validate()[0]]
    if audit.attribution.realized is None:
        return Knowability.DATA_FAILURE, ["the realised move could not be measured"]
    c, dirn, nd = scores["combined"], scores["direction"], scores["n_supporting_domains"]
    if scores["strongest"] < cfg.min_pointer:
        if c >= cfg.weak:
            notes.append(f"only sub-threshold pointers (strongest {scores['strongest']:.2f} < {cfg.min_pointer}): treated as noise")
        c = 0.0
    if c >= cfg.strong and dirn >= cfg.min_direction_for_predictable and nd >= cfg.min_supporting_domains:
        return Knowability.PREDICTABLE, notes
    if c >= cfg.strong:
        notes.append("strong evidence but not enough independent direction evidence for PREDICTABLE")
        return Knowability.POTENTIALLY_PREDICTABLE, notes
    if c >= cfg.moderate:
        return Knowability.POTENTIALLY_PREDICTABLE, notes
    if c >= cfg.weak:
        return Knowability.WEAKLY_PREDICTABLE, notes
    if scores["opposing"] >= cfg.moderate and scores["agreeing"] < cfg.weak:
        return Knowability.UNKNOWN, ["available evidence pointed the other way and nothing pointed toward the move"]
    n_rel = max(1, len(relevant))
    unc = sum(1 for i in relevant if i.availability == Availability.UNCERTAIN)
    if relevant and unc / n_rel >= cfg.uncertain_share_unknown:
        return Knowability.UNKNOWN, ["too many relevant items have unprovable timestamps"]
    att = audit.attribution
    if att.complete and att.market_share + att.peer_share >= cfg.external_share:
        notes.append(f"market {att.market_share:.2f} + peer {att.peer_share:.2f} share of the move")
        return Knowability.EXTERNALLY_CAUSED, notes
    causes = _cause_items(audit, cfg)
    if causes and att.complete and att.idio_share >= cfg.idio_share:
        top = max(causes, key=lambda u: u.relevance)
        notes.append(f"post-decision cause {top.item.key} ({top.item.availability.value}), relevance {top.relevance:.2f}")
        if top.item.availability == Availability.SIMULTANEOUS:
            notes.append("boundary case: the cause surfaced at the decision boundary")
        return Knowability.INFORMATIONALLY_UNAVAILABLE, notes
    if not att.complete:
        notes.append("attribution incomplete and no identified cause")
    else:
        notes.append("no pre-event evidence and no identified post-event cause")
    return Knowability.UNKNOWN, notes


def confidence_in_classification(kn: Knowability, state: KnowledgeState, scores: Mapping[str, float], audit: AuditResult,
                                 relevant: Sequence[InfoItem], cfg: CounterfactualConfig) -> ClassificationConfidence:
    """How much to trust the LABEL. Product of separately measured components (so one weak part cannot hide behind
    strong ones), then capped: an UNKNOWN is capped (it is a statement about the limit of the evidence), a label resting
    on an incomplete attribution is capped, and a DATA_FAILURE reads high only when the data really are missing."""
    cov = state.coverage()
    coverage = sum(1 for n in cov.values() if n) / len(ALL_DOMAINS)
    n_rel = len(relevant)
    prov = 1.0 - (sum(1 for i in relevant if i.availability == Availability.UNCERTAIN) / n_rel if n_rel else 0.0)
    att = audit.attribution
    att_c = 1.0 if att.complete and att.beta_obs >= 60 else 0.7 if att.complete else 0.4
    c = 0.0 if scores["strongest"] < cfg.min_pointer else scores["combined"]
    edges = (cfg.weak, cfg.moderate, cfg.strong)
    margin = min(1.0, min(abs(c - e) for e in edges) / 0.15) if kn in (Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE,
                                                                       Knowability.WEAKLY_PREDICTABLE) else \
        (0.5 if scores["opposing"] >= cfg.moderate else 1.0 - min(1.0, c / cfg.weak))
    if kn == Knowability.EXTERNALLY_CAUSED:
        margin = min(1.0, (att.market_share + att.peer_share - cfg.external_share) / 0.3 + 0.3)
    if kn == Knowability.INFORMATIONALLY_UNAVAILABLE:
        top = max((u.relevance for u in _cause_items(audit, cfg)), default=0.0)
        margin = min(1.0, top)
    hind = 0.0
    if kn in (Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE):
        hind = 1.0                       # the label is interpretable only with the outcome in hand
    elif scores["direction"] > 0:
        hind = min(1.0, scores["direction"])   # 'agreed with the direction' is a comparison against the realised sign
    caps: list[str] = []
    overall = (max(prov, 1e-9) * max(coverage, 1e-9) * max(att_c, 1e-9) * max(margin, 1e-9)) ** 0.25
    overall *= 1.0 - 0.25 * hind         # the more the label needed the outcome to be read, the less it can be trusted
    if kn == Knowability.DATA_FAILURE:
        overall, margin = min(1.0, 1.0 - coverage + 0.1), 1.0 - coverage
        caps.append("data failure: confidence is that the data are missing, not a statement about the market")
    if kn == Knowability.UNKNOWN and overall > cfg.unknown_conf_cap:
        overall = cfg.unknown_conf_cap
        caps.append("unknown is capped: absence of identified cause is not proof of no cause")
    if not att.complete and kn != Knowability.DATA_FAILURE and overall > 0.5:
        overall = 0.5
        caps.append("attribution incomplete")
    if any("boundary" in n for n in audit.notes) and overall > 0.6:
        overall = 0.6
        caps.append("boundary timing")
    return ClassificationConfidence(float(min(1.0, max(0.0, overall))), float(prov), float(coverage), float(max(0.0, margin)),
                                    float(att_c), float(hind), tuple(caps))


def available_evidence_items(state: KnowledgeState, ev: Sequence[Evidence], min_strength: float) -> tuple[InfoItem, ...]:
    """The state items that carried evidence: `information_that_would_have_been_available`."""
    names = {(e.domain, e.name) for e in ev if e.strength >= min_strength}
    keep = []
    for it in state.items:
        base = it.name.split(":")[0] if it.domain in (Domain.MEMORY,) else it.name
        if (it.domain, base) in names or (it.domain, it.name) in names or _evidence_reads(it, names):
            keep.append(it)
    return tuple(keep)


_EVIDENCE_SOURCES = {
    "range_squeeze": ("squeeze_rank",), "vol_contraction": ("vol_ratio",), "vol_expansion": ("vol_ratio",),
    "near_52w_high": ("dist_52wh",), "deep_below_high": ("dist_52wh",), "trend_continuation": ("r20", "dist_ma50"),
    "trend_opposes": ("r20", "dist_ma50"), "oversold_reversal": ("rsi14",), "overbought_reversal": ("rsi14",),
    "volume_build": ("vol_surge5",), "volume_spike": ("vol_surge1",), "volume_confirms_direction": ("vol_surge1", "r1"),
    "gap_into_close": ("gap_today",), "gap_against": ("gap_today",), "high_vol_rank": ("vol20_rank",),
    "lottery_rank": ("max20_rank",), "stress_regime": ("m_vix",), "vix_spike": ("m_vix_chg5",)}


def _evidence_reads(it: InfoItem, names: set) -> bool:
    for dom, nm in names:
        if it.name in _EVIDENCE_SOURCES.get(nm, ()):
            return True
    return False


# ------------------------------------------------------------------------------------------------ one event, end to end


def assess_event(store: pit.PITStore, snap: DaySnapshot, event: EventSpec, now, providers: Providers | None = None,
                 cfg: CounterfactualConfig | None = None, selection: str = "all", weight: float = 1.0) -> CounterfactualReport | None:
    """The whole test for one event on a shared day snapshot. Returns None when the outcome has not matured against the real
    clock `now` (the auditor is not allowed to look at a move that has not finished). Refuses an invalid event."""
    cfg, providers = cfg or CounterfactualConfig(), providers or Providers()
    errs = event.validate()
    if errs:
        raise ValueError(f"invalid event {event.event_id}: {'; '.join(errs)}")
    as_of = auditor_as_of_for(event, now, snap.cal, cfg.settle_sessions)
    if as_of is None:
        return None
    state, side = reconstruct_state(snap, event, providers, cfg)
    audit = audit_event(store, snap, event, state, side, as_of, providers, cfg)
    ev = collect_evidence(state, event, cfg)
    scores, null_notes = calibrate_pointers(snap, event, ev, score_evidence(ev, cfg), cfg)
    would = available_evidence_items(state, ev, cfg.weak / 2)
    relevant = tuple(would) + tuple(i for i in audit.unavailable)
    kn, notes = classify_knowability(state, scores, audit, cfg, relevant)
    notes = list(notes) + null_notes
    extra: list[str] = []
    if providers.classifier is not None:
        bundle = {"state": state, "scores": dict(scores), "evidence": ev, "attribution": audit.attribution, "baseline": kn}
        try:
            alt, why = providers.classifier(bundle)
            alt = Knowability.parse(alt)
            if alt != kn:
                extra.append(f"external classifier said {alt.value} ({why}); in-module ladder said {kn.value}")
                if kn == Knowability.DATA_FAILURE:
                    alt = kn
                kn = alt
        except (ValueError, TypeError, KeyError) as e:
            extra.append(f"external classifier unusable: {e}")
    conf = confidence_in_classification(kn, state, scores, audit, relevant, cfg)
    uncertain = tuple(i for i in relevant if i.availability == Availability.UNCERTAIN)
    uncertain += tuple(u.item for u in audit.future_uses if u.item.availability == Availability.UNCERTAIN)
    matured = audit.max_available_used or audit.auditor_as_of
    matured = str(max(pd.Timestamp(matured), pd.Timestamp(event.event_end)).date())
    rep = CounterfactualReport(event, state, audit.future_uses, would, audit.unavailable, tuple(ev), audit.attribution, kn, conf,
                               scores, uncertain, audit.auditor_as_of, matured, tuple(notes) + audit.notes + tuple(extra), selection,
                               float(weight), current_code_hash())
    bad = rep.validate()
    if bad:
        raise FirewallBreach(f"report for {event.event_id} failed its own audit: {bad[:3]}")
    return rep


# ------------------------------------------------------------------------------------------------ proof the state does not see the future


@dataclasses.dataclass(frozen=True)
class InvarianceResult:
    event_id: str
    base_digest: str
    scrambled_digests: tuple[str, ...]
    invariant: bool
    log_clean: bool
    detail: str = ""


def log_problems(log: pit.AuditLog, as_of) -> list[str]:
    """An independent re-check of one guard's audit trail: hash chain intact, no denied read, no allowed read that returned a
    row available after `as_of`."""
    errs = []
    if not log.verify():
        errs.append("audit hash chain broken")
    if log.denied():
        errs.append(f"{len(log.denied())} denied read(s)")
    if log.leaks():
        errs.append(f"{len(log.leaks())} read(s) returned data available after as_of")
    m = log.max_avail_touched()
    if m is not None and m > pd.Timestamp(as_of).normalize():
        errs.append(f"max availability touched {m.date()} is after {pd.Timestamp(as_of).date()}")
    return errs


def verify_future_invariance(store: pit.PITStore, event: EventSpec, providers: Providers | None = None,
                             cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None,
                             seeds: Sequence[int] = (0, 1)) -> InvarianceResult:
    """Rebuild the knowledge state on copies of the store whose future (everything not yet available at T) is scrambled and
    stuffed with fabricated rows. A state that depends on the future cannot reproduce its digest; one that does not, must."""
    cfg, srcs = cfg or CounterfactualConfig(), srcs or SourceMap()
    t0 = pd.Timestamp(event.decision_ts).normalize()

    def digest(s: pit.PITStore) -> tuple[str, list[str]]:
        snap = build_snapshot(s, t0, srcs, cfg)
        st, _ = reconstruct_state(snap, event, providers, cfg)
        return st.digest, log_problems(snap.log, t0) + st.validate()

    base, problems = digest(store)
    others, detail = [], list(problems)
    for sd in seeds:
        d, p = digest(store.scrambled(t0, seed=int(sd)))
        others.append(d)
        detail += p
    ok = all(d == base for d in others)
    if not ok:
        detail.append("state digest changed when only the future was scrambled: the reconstruction reads the future")
    return InvarianceResult(event.event_id, base, tuple(others), ok, not problems, "; ".join(detail))


# ------------------------------------------------------------------------------------------------ the gate: classification is hindsight

ALLOWED_EFFECTS = frozenset({"NONE", "RESEARCH_PRIORITY"})
HINDSIGHT_NAMES = frozenset({"knowability", "confidence_in_classification", "future_information_used_by_auditor",
                             "information_that_was_unavailable", "information_that_would_have_been_available",
                             "knowledge_state_at_decision"})
HINDSIGHT_PREFIX = "cf_"
_ORDER = {k: i for i, k in enumerate((Knowability.INFORMATIONALLY_UNAVAILABLE, Knowability.EXTERNALLY_CAUSED, Knowability.UNKNOWN,
                                      Knowability.DATA_FAILURE, Knowability.WEAKLY_PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE,
                                      Knowability.PREDICTABLE))}


def refuse_feedback(obj: Any, where: str = "object", _depth: int = 0) -> None:
    """Raise FirewallBreach if a hindsight classification, or any field named for one, appears anywhere inside `obj`: a dict
    key, a DataFrame column, a dataclass field, a list element. The classification may only leave the research world through
    ClassificationGate, so wherever else it turns up (a feature matrix, a model input, a memory item) it is a leak."""
    if _depth > 6:
        return
    if isinstance(obj, (CounterfactualReport, InfoItem, FutureUse, Evidence, ClassificationConfidence, MoveAttribution)):
        raise FirewallBreach(f"{where}: a {type(obj).__name__} is a hindsight object and may not leave the research world")
    if isinstance(obj, KnowledgeState) and obj.items:
        raise FirewallBreach(f"{where}: a KnowledgeState carries reconstruction internals; release labels through the gate")
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            if str(c) in HINDSIGHT_NAMES or str(c).startswith(HINDSIGHT_PREFIX):
                raise FirewallBreach(f"{where}: column {c!r} is a hindsight classification")
        return
    if isinstance(obj, Knowability):
        raise FirewallBreach(f"{where}: a Knowability label ({obj.value}) is hindsight")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) in HINDSIGHT_NAMES or str(k).startswith(HINDSIGHT_PREFIX):
                raise FirewallBreach(f"{where}: key {k!r} is a hindsight classification")
            refuse_feedback(v, f"{where}[{k!r}]", _depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            refuse_feedback(v, f"{where}[{i}]", _depth + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name in HINDSIGHT_NAMES:
                raise FirewallBreach(f"{where}: field {f.name!r} is a hindsight classification")
            refuse_feedback(getattr(obj, f.name), f"{where}.{f.name}", _depth + 1)


def _ordinal(kn: str) -> int:
    return _ORDER[Knowability.parse(kn)]


def disguised_feedback_columns(X: pd.DataFrame, reports: Sequence[CounterfactualReport], threshold: float = 0.9,
                               min_rows: int = 8) -> dict[str, float]:
    """Columns of X (indexed by event_id) that track the hindsight label or its scores under another name. A renamed copy of
    the classification is the obvious way to smuggle it in; a rank correlation with the label ordinal, the combined
    evidence score or the idiosyncratic share above `threshold` is flagged. Returns {column: |rho|}."""
    by_id = {r.event.event_id: r for r in reports}
    common = [i for i in X.index if i in by_id]
    if len(common) < min_rows:
        return {}
    targets = {
        "label": pd.Series([_ordinal(by_id[i].knowability) for i in common], index=common, dtype=float),
        "combined": pd.Series([by_id[i].scores.get("combined", 0.0) for i in common], index=common),
        "idio_share": pd.Series([by_id[i].attribution.idio_share for i in common], index=common),
        "confidence": pd.Series([by_id[i].confidence_in_classification.overall for i in common], index=common)}
    out: dict[str, float] = {}
    for c in X.columns:
        col = pd.to_numeric(X.loc[common, c], errors="coerce")
        if col.nunique() < 2:
            continue
        for tname, t in targets.items():
            if t.nunique() < 2:
                continue
            rho = abs(float(col.rank().corr(t.rank())))
            if math.isfinite(rho) and rho >= threshold:
                out[str(c)] = max(out.get(str(c), 0.0), rho)
    return out


def refuse_disguised_feedback(X: pd.DataFrame, reports: Sequence[CounterfactualReport], threshold: float = 0.9) -> None:
    refuse_feedback(X, "feature frame")
    bad = disguised_feedback_columns(X, reports, threshold)
    if bad:
        raise FirewallBreach(f"feature frame carries the hindsight classification under other names: {bad}")


class ClassificationGate:
    """The only road out of the research world for a classification (section 31; mapping risk (c)). Three rules:
      1. maturity: a record is visible only to a `now` strictly after its matured_at (MaturedRecord.gate, which also asks
         whether the record could have existed then);
      2. same-year rerun: research filed under real year Y is withheld while year Y is being replayed in disguise (C55/C64);
      3. purpose: a classification may drive RESEARCH_PRIORITY or nothing, never a live decision (ranking, selection, size,
         direction, timing, exit, stop), and what leaves carries no date, year, ticker or id the trader could key on."""

    def __init__(self):
        self._recs: dict[str, tuple[MaturedRecord, int, str]] = {}
        self.withheld: list[dict] = []

    def add(self, report: CounterfactualReport) -> str:
        rec = report.matured_record()
        self._recs[rec.record_id] = (rec, report.event.real_year, report.event.category)
        return rec.record_id

    def __len__(self) -> int:
        return len(self._recs)

    @staticmethod
    def _check_effect(effect) -> None:
        name = str(getattr(effect, "value", effect))
        if name not in ALLOWED_EFFECTS:
            raise FirewallBreach(f"a hindsight classification may not drive {name}; only {sorted(ALLOWED_EFFECTS)}")

    def release_one(self, report_id: str, now, replaying_years: Iterable[int] = (), effect="RESEARCH_PRIORITY") -> dict:
        """Strict form: raises on an immature record or a same-year rerun instead of withholding."""
        self._check_effect(effect)
        if report_id not in self._recs:
            raise KeyError(f"unknown report {report_id}")
        rec, year, cat = self._recs[report_id]
        if year in set(int(y) for y in replaying_years):
            raise FirewallBreach(f"report {report_id} is filed under a year that is being replayed: same-year rerun leak")
        payload = rec.gate(now)
        return self._safe_row(report_id, payload, cat)

    def release(self, now, replaying_years: Iterable[int] = (), effect="RESEARCH_PRIORITY") -> list[dict]:
        """Everything that has matured and is not same-year, as identity-free rows; the rest is withheld and counted."""
        self._check_effect(effect)
        replay = set(int(y) for y in replaying_years)
        rows = []
        for rid in sorted(self._recs):
            rec, year, cat = self._recs[rid]
            if year in replay:
                self.withheld.append({"report": rid, "why": "same_year_rerun"})
                continue
            if as_date(rec.matured_at) >= as_date(now):
                self.withheld.append({"report": rid, "why": "not_matured"})
                continue
            rows.append(self._safe_row(rid, rec.gate(now), cat))
        return rows

    def _safe_row(self, rid: str, payload: Mapping[str, Any], category: str) -> dict:
        row = {"token": "T" + stable_hash(["cf-label", rid], 12), "label": payload["knowability"],
               "confidence": round(float(payload["confidence"]), 3), "direction": int(payload["direction"]),
               "category": str(category).lower()}
        bad = find_violations(row)
        if bad:
            raise FirewallBreach(f"released row is not identity-free: {bad[0]}")
        return row

    def training_labels(self, now, replaying_years: Iterable[int] = ()) -> pd.DataFrame:
        """The explicit point-in-time training gate: labels dated at their maturity, only those strictly before `now`,
        columns marked with the cf_ prefix so refuse_feedback recognises them anywhere else they appear. Research-side."""
        replay = set(int(y) for y in replaying_years)
        rows = []
        for rid in sorted(self._recs):
            rec, year, _ = self._recs[rid]
            if year in replay or as_date(rec.matured_at) >= as_date(now):
                continue
            p = rec.gate(now)
            rows.append({"report_id": rid, "cf_knowability": p["knowability"], "cf_confidence": p["confidence"],
                         "cf_matured_at": rec.matured_at})
        return pd.DataFrame(rows, columns=["report_id", "cf_knowability", "cf_confidence", "cf_matured_at"])


def assert_label_consumer_safe(label_rows: pd.DataFrame, consumer_now) -> None:
    """A consumer that trains on labels at `consumer_now` may hold none matured at or after it."""
    if not len(label_rows):
        return
    late = label_rows[pd.to_datetime(label_rows["cf_matured_at"]) >= pd.Timestamp(consumer_now)]
    if len(late):
        raise FirewallBreach(f"{len(late)} label(s) matured at or after {pd.Timestamp(consumer_now).date()}: hindsight in training")


# ------------------------------------------------------------------------------------------------ selection, streaming, checkpoints


def events_from_frame(df: pd.DataFrame, cal: pit.Calendar | None = None) -> list[EventSpec]:
    """EventSpecs from a movers table with columns ticker, decision_ts, event_end, direction (+/-1) and optionally category,
    realized_return, model_score, score_asof. Invalid rows are refused, not repaired."""
    out = []
    for r in df.to_dict("records"):
        spec = EventSpec.make(r["ticker"], r["decision_ts"], r["event_end"], int(r["direction"]), cal, r.get("category", ""),
                              r.get("realized_return"), r.get("model_score"), r.get("score_asof", ""))
        errs = spec.validate()
        if errs:
            raise ValueError(f"movers row for {r['ticker']}: {'; '.join(errs)}")
        out.append(spec)
    return out


def select_events(events: Sequence[EventSpec], cfg: CounterfactualConfig, seed: int = 0) -> list[tuple[EventSpec, str, float]]:
    """Per decision day: the top-K by |realised move| in full, a seeded sample of the rest with inverse-probability weights
    (mapping design: cap the per-event reconstruction cost). The draw depends only on (seed, event_id), never on order."""
    by_day: dict[str, list[EventSpec]] = defaultdict(list)
    for e in events:
        by_day[e.decision_ts].append(e)
    out = []
    for day in sorted(by_day):
        ranked = sorted(by_day[day], key=lambda e: (-abs(e.realized_return or 0.0), e.event_id))
        for e in ranked[:cfg.top_k_per_day]:
            out.append((e, "top_k", 1.0))
        for e in ranked[cfg.top_k_per_day:]:
            if cfg.sample_rate <= 0:
                continue
            u = int(stable_hash(["cf-sample", seed, e.event_id], 8), 16) / 16 ** 8
            if u < cfg.sample_rate:
                out.append((e, "sampled", 1.0 / cfg.sample_rate))
    return out


@dataclasses.dataclass
class BatchResult:
    rows: pd.DataFrame
    reports: list[CounterfactualReport]
    skipped: list[dict]
    errors: list[dict]
    n_snapshots: int
    resumed: int = 0
    n_selected: int = 0

    def summary(self) -> dict:
        return summarize(self.rows)


def _done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["event"]["event_id"])
            except (ValueError, KeyError):
                continue                      # a torn last line from a kill is simply redone
    return done


def _append_line(path: Path, obj: Mapping) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_batch(store: pit.PITStore, events: Sequence[EventSpec], now, providers: Providers | None = None,
              cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None, out_dir: str | os.PathLike | None = None,
              seed: int = 0, keep_reports: bool = False, on_error: str = "record") -> BatchResult:
    """Stream the assessment day by day. One PIT snapshot per decision date is shared by that day's events, and the snapshot
    is dropped before the next date (the market-wide universe never sits in memory twice). With `out_dir`, each report is
    appended to cf_<year>.jsonl as it finishes and a rerun resumes past the ids already there (a kill costs one event).
    Only flat summary rows are kept in memory unless keep_reports. FirewallBreach / LookAheadError are never swallowed."""
    cfg, srcs, providers = cfg or CounterfactualConfig(), srcs or SourceMap(), providers or Providers()
    chosen = select_events(events, cfg, seed)
    root = Path(out_dir) if out_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        check_manifest(root, cfg, srcs, seed)
    done = {y: _done_ids(root / f"cf_{y}.jsonl") for y in sorted({e.real_year for e, _, _ in chosen})} if root else {}
    rows, reports, skipped, errors = [], [], [], []
    n_snap = resumed = 0
    by_day: dict[str, list] = defaultdict(list)
    for item in chosen:
        by_day[item[0].decision_ts].append(item)
    for day in sorted(by_day):
        todo = []
        for e, sel, w in by_day[day]:
            if root is not None and e.event_id in done[e.real_year]:
                resumed += 1
                continue
            if auditor_as_of_for(e, now, store.cal, cfg.settle_sessions) is None:
                skipped.append({"event": e.event_id, "why": "outcome not matured against the real clock"})
                continue
            todo.append((e, sel, w))
        if not todo:
            continue
        snap = build_snapshot(store, day, srcs, cfg)
        n_snap += 1
        for e, sel, w in todo:
            try:
                rep = assess_event(store, snap, e, now, providers, cfg, sel, w)
            except (FirewallBreach, pit.PITError):
                raise
            except (ValueError, KeyError, IndexError) as ex:
                if on_error == "raise":
                    raise
                errors.append({"event": e.event_id, "error": f"{type(ex).__name__}: {ex}"})
                continue
            if rep is None:
                skipped.append({"event": e.event_id, "why": "outcome not matured"})
                continue
            rows.append(rep.short())
            if root is not None:
                _append_line(root / f"cf_{e.real_year}.jsonl", rep.to_dict())
            if keep_reports:
                reports.append(rep)
        del snap
    frame = pd.DataFrame(rows, columns=list(_ROW_COLUMNS))
    return BatchResult(frame, reports, skipped, errors, n_snap, resumed, len(chosen))


_ROW_COLUMNS = ("report_id", "event_id", "year", "category", "direction", "knowability", "confidence", "n_available", "n_unavailable",
                "n_future_used", "n_uncertain", "coverage_domains", "market_share", "idio_share", "combined", "selection", "weight",
                "matured_at")


def step(store: pit.PITStore, events: Sequence[EventSpec], now, providers: Providers | None = None,
         cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None, seed: int = 0) -> list[CounterfactualReport]:
    """The single public entry the research loop calls: assess every matured event (after the per-day cap) and return the
    reports. `now` is the REAL trusted-side clock; an event whose window has not finished before it is skipped."""
    return run_batch(store, events, now, providers, cfg, srcs, None, seed, True).reports


# ------------------------------------------------------------------------------------------------ summaries and honesty checks


def summarize(rows: pd.DataFrame) -> dict:
    """Weighted shares (sampled events count 1/p), by class, category, year and direction, plus the honesty flags."""
    if not len(rows):
        return {"n": 0, "weighted_n": 0.0, "by_class": {}, "by_category": {}, "by_year": {}, "flags": [], "unknown_rate": None}
    w = rows["weight"].astype(float)
    tot = float(w.sum())
    by_class = (rows.assign(_w=w).groupby("knowability")["_w"].sum() / tot).round(4).to_dict()
    by_cat = {c: (g.assign(_w=g["weight"]).groupby("knowability")["_w"].sum() / g["weight"].sum()).round(4).to_dict()
              for c, g in rows.groupby(rows["category"].replace("", "uncategorised"))}
    by_year = {int(y): (g.assign(_w=g["weight"]).groupby("knowability")["_w"].sum() / g["weight"].sum()).round(4).to_dict()
               for y, g in rows.groupby("year")}
    conf = rows.groupby("knowability")["confidence"].mean().round(4).to_dict()
    return {"n": int(len(rows)), "weighted_n": tot, "by_class": by_class, "by_category": by_cat, "by_year": by_year,
            "mean_confidence": conf, "unknown_rate": float(by_class.get(Knowability.UNKNOWN.value, 0.0)),
            "external_rate": float(by_class.get(Knowability.EXTERNALLY_CAUSED.value, 0.0)),
            "hindsight_needed_rate": float(((rows["n_unavailable"] > 0) | (rows["n_future_used"] > 0)).mean()),
            "flags": honesty_flags(rows)}


def honesty_flags(rows: pd.DataFrame, min_n: int = 30) -> list[str]:
    """Patterns in the label distribution that mean the classifier is being forced or is leaking (RT20)."""
    flags = []
    if len(rows) < min_n:
        return ["too few reports for distribution checks"]
    shares = rows["knowability"].value_counts(normalize=True)
    if shares.get(Knowability.UNKNOWN.value, 0.0) == 0.0:
        flags.append("no UNKNOWN at all: unknown is a correct answer, a researcher that never says it is forcing labels")
    decided = rows[~rows["knowability"].isin([Knowability.UNKNOWN.value, Knowability.DATA_FAILURE.value])]
    if len(decided) and float((decided["confidence"] < 0.3).mean()) > 0.25:
        flags.append("many labels are assigned with confidence below 0.3: forced labelling")
    if shares.get(Knowability.PREDICTABLE.value, 0.0) > 0.5:
        flags.append("over half PREDICTABLE: check that hindsight direction is not doing the work")
    if shares.get(Knowability.DATA_FAILURE.value, 0.0) > 0.3:
        flags.append("over 30% DATA_FAILURE: the reconstruction is not covering the universe")
    if rows["n_uncertain"].gt(0).mean() > 0.5:
        flags.append("over half of the reports contain timestamps that cannot be established")
    return flags


def report_markdown(result: BatchResult | pd.DataFrame, title: str = "Could I have known? - counterfactual report", top: int = 10) -> str:
    """Trusted-side report (it carries real years). Never handed to the trader."""
    rows = result.rows if isinstance(result, BatchResult) else result
    s = summarize(rows)
    lines = [f"# {title}", "", "IMPLEMENTED - NOT VALIDATED (foundation wave; no real-data run).", "",
             f"Reports: {s['n']} (weighted {s['weighted_n']:.1f}).", ""]
    if not s["n"]:
        return "\n".join(lines + ["No reports: nothing matured or nothing selected."])
    lines += ["## Distribution of knowability", "", "| class | share | mean confidence |", "|---|---:|---:|"]
    for k, v in sorted(s["by_class"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:.3f} | {s['mean_confidence'].get(k, float('nan')):.3f} |")
    lines += ["", "## By category", ""]
    for c, d in sorted(s["by_category"].items()):
        lines.append(f"- {c}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]))
    lines += ["", "## By year", ""]
    for y, d in sorted(s["by_year"].items()):
        lines.append(f"- {y}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]))
    lines += ["", f"Unknown rate {s['unknown_rate']:.3f}; external {s['external_rate']:.3f}; "
              f"reports needing hindsight to interpret {s['hindsight_needed_rate']:.3f}.", "", "## Honesty flags", ""]
    lines += [f"- {f}" for f in s["flags"]] or ["- none raised"]
    if isinstance(result, BatchResult) and (result.skipped or result.errors):
        lines += ["", f"Skipped {len(result.skipped)}, errors {len(result.errors)}."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ how stable is the label


def reclassify(report: CounterfactualReport, cfg: CounterfactualConfig, drop_domain: Domain | None = None) -> Knowability:
    """Re-run the ladder on the report's own recorded state, evidence and attribution with a different config, optionally
    without one whole domain. Uses only what the report already holds, so it cannot pick up new information."""
    ev = [e for e in report.evidence if drop_domain is None or e.domain != drop_domain]
    scores = score_evidence(ev, cfg)
    state = report.knowledge_state_at_decision
    if drop_domain is not None:
        state = dataclasses.replace(state, items=tuple(i for i in state.items if i.domain != drop_domain))
    causes = tuple(FutureUse(u.item, u.used_for, u.days_after_decision, u.relevance) for u in report.future_information_used_by_auditor)
    audit = AuditResult(report.attribution.realized, causes, report.information_that_was_unavailable, report.attribution,
                        report.notes, report.auditor_as_of, "", "")
    kn, _ = classify_knowability(state, scores, audit, cfg, tuple(report.information_that_would_have_been_available)
                                 + tuple(report.information_that_was_unavailable))
    return kn


@dataclasses.dataclass(frozen=True)
class StabilityReport:
    event_id: str
    label: Knowability
    stable_share: float                     # share of threshold perturbations that keep the label
    flips_to: Mapping[str, int]
    decisive_domains: tuple[str, ...]       # domains whose removal changes the label
    fragile: bool


def perturbed_configs(cfg: CounterfactualConfig, scale: float = 0.2) -> list[CounterfactualConfig]:
    """Every threshold moved up and down by `scale` (keeping the ordering weak < moderate < strong): the label is only as firm
    as its invariance to where the starting thresholds were put."""
    out = []
    for f in (1.0 - scale, 1.0 + scale):
        weak, mod, strong = cfg.weak * f, cfg.moderate * f, min(cfg.strong * f, 1.0)
        if not weak < mod < strong:
            continue
        out.append(dataclasses.replace(cfg, weak=weak, moderate=mod, strong=strong, external_share=min(1.0, cfg.external_share * f),
                                       idio_share=min(1.0, cfg.idio_share * f)))
    for name in ("magnitude_discount", "min_direction_for_predictable", "min_cause_relevance"):
        for f in (1.0 - scale, 1.0 + scale):
            out.append(dataclasses.replace(cfg, **{name: min(1.0, getattr(cfg, name) * f)}))
    return out


def label_stability(report: CounterfactualReport, cfg: CounterfactualConfig | None = None, scale: float = 0.2) -> StabilityReport:
    """Threshold sensitivity plus leave-one-domain-out. `fragile` when fewer than 75% of perturbations keep the label, or when
    a single domain carries a non-UNKNOWN label (that is a finding: the label rests on one channel)."""
    cfg = cfg or CounterfactualConfig()
    base = reclassify(report, cfg)
    flips: Counter = Counter()
    cfgs = perturbed_configs(cfg, scale)
    for c in cfgs:
        k = reclassify(report, c)
        if k != base:
            flips[k.value] += 1
    stable = 1.0 - sum(flips.values()) / len(cfgs) if cfgs else 1.0
    decisive = []
    for d in ALL_DOMAINS:
        if any(e.domain == d for e in report.evidence) or d in (Domain.PRICE, Domain.VOLUME):
            if reclassify(report, cfg, d) != base:
                decisive.append(d.value)
    fragile = stable < 0.75 or (len(decisive) == 1 and base not in (Knowability.UNKNOWN, Knowability.DATA_FAILURE))
    return StabilityReport(report.event.event_id, base, float(stable), dict(flips), tuple(decisive), bool(fragile))


# ------------------------------------------------------------------------------------------------ lead time and missed information


@dataclasses.dataclass(frozen=True)
class LeadTime:
    domain: str
    name: str
    sessions_before: int          # how many sessions before the decision close the pointer first became public
    strength: float


def lead_times(report: CounterfactualReport) -> list[LeadTime]:
    """For each piece of evidence, how early was the underlying information public? A pointer that surfaced the same
    session is a different research target from one that was public for weeks (section 8: 'what became known later')."""
    by_key = {(i.domain, i.name): i for i in report.knowledge_state_at_decision.items}
    out = []
    for e in report.evidence:
        srcs = _EVIDENCE_SOURCES.get(e.name, (e.name,))
        lags = [-(by_key[(e.domain, s)].lag_sessions or 0) for s in srcs if (e.domain, s) in by_key]
        if lags:
            out.append(LeadTime(e.domain.value, e.name, int(max(lags)), e.strength))
    return sorted(out, key=lambda x: (-x.strength, x.name))


def earliest_warning(report: CounterfactualReport, min_strength: float = 0.2) -> int | None:
    """Sessions before the decision of the earliest sufficiently strong pointer; None if nothing pointed at all."""
    lts = [lt.sessions_before for lt in lead_times(report) if lt.strength >= min_strength]
    return max(lts) if lts else None


def missed_information(reports: Sequence[CounterfactualReport], min_strength: float = 0.3) -> pd.DataFrame:
    """Evidence that WAS public at T and pointed toward the move but was NOT an input the model used (item.in_model False):
    the actionable output of the test. One row per (domain, evidence name) with counts and mean strength, weighted by the
    sampling weights. Reports labelled UNKNOWN/DATA_FAILURE contribute nothing (no evidence to miss)."""
    acc: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    tot_w = sum(r.weight for r in reports) or 1.0
    for r in reports:
        if r.knowability in (Knowability.UNKNOWN, Knowability.DATA_FAILURE):
            continue
        model_keys = {(i.domain, i.name) for i in r.knowledge_state_at_decision.items if i.in_model}
        for e in r.evidence:
            if e.strength < min_strength or e.agrees is False:
                continue
            srcs = _EVIDENCE_SOURCES.get(e.name, (e.name,))
            if any((e.domain, s) in model_keys for s in srcs):
                continue
            acc[(e.domain.value, e.name)].append((e.strength, r.weight))
    rows = [{"domain": d, "evidence": n, "n": len(v), "weighted_share": sum(w for _, w in v) / tot_w,
             "mean_strength": float(np.average([s for s, _ in v], weights=[w for _, w in v]))} for (d, n), v in acc.items()]
    return pd.DataFrame(rows, columns=["domain", "evidence", "n", "weighted_share", "mean_strength"]).sort_values(
        ["weighted_share", "evidence"], ascending=[False, True]).reset_index(drop=True)


def domain_contribution(reports: Sequence[CounterfactualReport]) -> pd.DataFrame:
    """Which domains supplied the pointers, per label. Rows: domain; columns: label; values: mean strongest evidence."""
    rec: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in reports:
        best: dict[str, float] = {}
        for e in r.evidence:
            best[e.domain.value] = max(best.get(e.domain.value, 0.0), e.strength)
        for d in ALL_DOMAINS:
            rec[(d.value, r.knowability.value)].append(best.get(d.value, 0.0))
    if not rec:
        return pd.DataFrame()
    df = pd.Series({k: float(np.mean(v)) for k, v in rec.items()}).unstack(fill_value=0.0)
    return df.reindex([d.value for d in ALL_DOMAINS]).fillna(0.0)


# ------------------------------------------------------------------------------------------------ what became known later


def what_became_known(store: pit.PITStore, event: EventSpec, horizons: Sequence[int] = (1, 5, 20), providers: Providers | None = None,
                      cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None) -> pd.DataFrame:
    """The literal comparison section 8 asks for: the same reconstruction at T and at T + h sessions, diffed. For every
    horizon the rows are the state items that are new, changed or vanished, with their domain. Only the AUDITOR calls this
    (it builds guards after T) and the result is hindsight; the state at T is rebuilt from scratch through its own guard."""
    cfg, srcs = cfg or CounterfactualConfig(), srcs or SourceMap()
    t0 = pd.Timestamp(event.decision_ts).normalize()
    base_snap = build_snapshot(store, t0, srcs, cfg)
    base, _ = reconstruct_state(base_snap, event, providers, cfg)
    ref = {i.key: i for i in base.items}
    rows = []
    for h in horizons:
        later = _shift(t0, int(h), store.cal)
        ev = dataclasses.replace(event, decision_ts=str(later.date()), event_start=str(_shift(later, 1, store.cal).date()),
                                 event_end=str(_shift(later, 1, store.cal).date()), model_score=None, score_asof="")
        snap = build_snapshot(store, later, srcs, cfg)
        st, _ = reconstruct_state(snap, ev, providers, cfg)
        cur = {i.key: i for i in st.items}
        for k in sorted(set(ref) | set(cur)):
            a, b = ref.get(k), cur.get(k)
            if a is None and b is not None:
                rows.append({"horizon": h, "key": k, "domain": b.domain.value, "change": "new", "available": b.available, "before": None, "after": b.value})
            elif a is not None and b is None:
                rows.append({"horizon": h, "key": k, "domain": a.domain.value, "change": "gone", "available": a.available, "before": a.value, "after": None})
            elif a is not None and b is not None and a.value != b.value and a.domain in (Domain.EVENT, Domain.MACRO, Domain.PATTERN, Domain.MEMORY):
                rows.append({"horizon": h, "key": k, "domain": a.domain.value, "change": "revised", "available": b.available,
                             "before": a.value, "after": b.value})
    return pd.DataFrame(rows, columns=["horizon", "key", "domain", "change", "available", "before", "after"])


# ------------------------------------------------------------------------------------------------ persistence and comparison


def load_rows(out_dir: str | os.PathLike) -> pd.DataFrame:
    """Flat summary rows rebuilt from the cf_<year>.jsonl files a run left behind, so a resumed run summarises everything.
    Torn lines are skipped and counted in `frame.attrs['torn']`."""
    rows, torn = [], 0
    for f in sorted(Path(out_dir).glob("cf_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                ks, ev, conf = d["knowledge_state_at_decision"], d["event"], d["confidence_in_classification"]
                rows.append({"report_id": d["report_id"], "event_id": ev["event_id"], "year": int(ev["decision_ts"][:4]),
                             "category": ev["category"], "direction": ev["direction"], "knowability": d["knowability"],
                             "confidence": conf["overall"], "n_available": len(d["information_that_would_have_been_available"]),
                             "n_unavailable": len(d["information_that_was_unavailable"]),
                             "n_future_used": len(d["future_information_used_by_auditor"]), "n_uncertain": len(d["uncertain_items"]),
                             "coverage_domains": sum(1 for v in ks["coverage"].values() if v),
                             "market_share": d["attribution"]["market_share"], "idio_share": d["attribution"]["idio_share"],
                             "combined": d["scores"].get("combined", 0.0), "selection": d["selection"], "weight": d["weight"],
                             "matured_at": d["matured_at"]})
            except (ValueError, KeyError, TypeError):
                torn += 1
    frame = pd.DataFrame(rows, columns=list(_ROW_COLUMNS))
    frame.attrs["torn"] = torn
    return frame


def compare_reports(a: CounterfactualReport, b: CounterfactualReport) -> dict[str, Any]:
    """Field-level difference of two reports for the same event (a rerun, a changed config, a scrambled-future rebuild). A
    deterministic pipeline gives an empty diff apart from the wall-clock provenance stamp."""
    if a.event.event_id != b.event.event_id:
        raise ValueError("reports are for different events")
    da, db = a.to_dict(), b.to_dict()
    da.pop("report_id", None), db.pop("report_id", None)
    diff = {}
    for k in sorted(set(da) | set(db)):
        if canonical_json(da.get(k)) != canonical_json(db.get(k)):
            diff[k] = (da.get(k), db.get(k))
    return diff


def adapt_classifier(obj: Any) -> Callable[[Mapping[str, Any]], tuple[Knowability, str]]:
    """Wrap R02's knowability engine without importing it: accepts a callable(bundle) or an object with `.classify(bundle)`;
    the answer may be a Knowability, its name, or an object with `.knowability`. Anything else is refused at call time."""
    fn = obj if callable(obj) else getattr(obj, "classify", None)
    if fn is None:
        raise TypeError("classifier must be callable or have .classify(bundle)")

    def call(bundle: Mapping[str, Any]) -> tuple[Knowability, str]:
        ans = fn(bundle)
        why = ""
        if isinstance(ans, tuple):
            ans, why = ans[0], str(ans[1]) if len(ans) > 1 else ""
        ans = getattr(ans, "knowability", ans)
        return Knowability.parse(ans), why
    return call


# ------------------------------------------------------------------------------------------------ self-check (planted leaks)


def selfcheck(store: pit.PITStore, event: EventSpec, providers: Providers | None = None, cfg: CounterfactualConfig | None = None,
              srcs: SourceMap | None = None) -> dict[str, bool]:
    """Does the machinery actually refuse a leak? Each entry must be True for the pipeline to be trusted:
      future_invariant      the state is identical on a store whose future was scrambled;
      fence_refuses_future  a Guard bound to T refuses a request past T;
      state_rejects_future  a KnowledgeState holding a post-T item fails its own validation;
      gate_refuses_early    a report cannot be released at or before its maturity;
      feedback_refused      the classification is refused inside a feature frame."""
    cfg, srcs = cfg or CounterfactualConfig(), srcs or SourceMap()
    t0 = pd.Timestamp(event.decision_ts).normalize()
    out = {"future_invariant": verify_future_invariance(store, event, providers, cfg, srcs).invariant}
    g = store.view(t0)
    try:
        g.wide(srcs.prices, srcs.close, end=t0 + pd.Timedelta(days=5))
        out["fence_refuses_future"] = False
    except pit.LookAheadError:
        out["fence_refuses_future"] = True
    snap = build_snapshot(store, t0, srcs, cfg)
    st, _ = reconstruct_state(snap, event, providers, cfg)
    late = InfoItem.make(Domain.PRICE, "planted", 1.0, event.event_end, event.event_end, "planted", t0, event.event_start, snap.cal)
    out["state_rejects_future"] = bool(dataclasses.replace(st, items=st.items + (late,)).validate())
    rep = assess_event(store, snap, event, pd.Timestamp(event.event_end) + pd.Timedelta(days=30), providers, cfg)
    if rep is None:
        out["gate_refuses_early"] = out["feedback_refused"] = False
        return out
    gate = ClassificationGate()
    rid = gate.add(rep)
    try:
        gate.release_one(rid, rep.matured_at)
        out["gate_refuses_early"] = False
    except FirewallBreach:
        out["gate_refuses_early"] = True
    try:
        refuse_feedback(pd.DataFrame({"x": [1.0], "cf_knowability": ["UNKNOWN"]}), "selfcheck")
        out["feedback_refused"] = False
    except FirewallBreach:
        out["feedback_refused"] = True
    return out


# ------------------------------------------------------------------------------------------------ planted worlds (for tests and wave-2 self-tests)

PLANTED_KINDS = ("precursor", "external", "news", "unknown", "data_failure")


@dataclasses.dataclass(frozen=True)
class PlantedPattern:
    """A pattern-like object for tests: fires when `rule(features)` is true; `provenance.learned_at` decides when it existed."""
    pattern_id: str
    name: str
    effect: float
    p_real: float
    provenance: Provenance
    rule: Callable[[Mapping[str, float]], bool]

    def fires(self, features: Mapping[str, float]) -> bool:
        return bool(self.rule(features))


@dataclasses.dataclass(frozen=True)
class PlantedMemory:
    mem_id: str
    kind: str
    context: Mapping[str, float]
    lean: float
    filed_date: str
    matured_at: str
    reliability: float = 0.7


class PlantedMemoryStore:
    def __init__(self, items: Sequence[PlantedMemory]):
        self._items = list(items)

    def memories(self, matured_before=None):
        if matured_before is None:
            return list(self._items)
        return [m for m in self._items if as_date(m.matured_at) < as_date(matured_before)]


@dataclasses.dataclass(frozen=True)
class PlantedCase:
    kind: str
    store: pit.PITStore
    event: EventSpec
    providers: Providers
    truth: Knowability
    now: pd.Timestamp
    decision_index: int


def make_planted_case(kind: str, seed: int = 0, n_tickers: int = 40, n_days: int = 320) -> PlantedCase:
    """A synthetic market with ONE planted movement of a known cause. kinds: precursor (public build-up, then a jump in the
    direction it pointed), external (a market shock after T that drags every name), news (a company filing after T; the stock
    was quiet before), unknown (an idiosyncratic jump with no trace), data_failure (no volume data at all). The truth label
    each should receive is in `case.truth`. Everything is seeded; no real dates or tickers."""
    if kind not in PLANTED_KINDS:
        raise ValueError(f"unknown planted kind {kind!r}; choose from {PLANTED_KINDS}")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    tick = [f"T{i:02d}" for i in range(n_tickers)]
    D = n_days - 12
    beta = np.clip(rng.normal(1.0, 0.15, n_tickers), 0.6, 1.5)
    beta[0] = 1.2
    rm = rng.normal(0.0003, 0.007, n_days)
    idio = rng.normal(0.0, 0.011, (n_days, n_tickers))
    ret = beta[None, :] * rm[:, None] + idio
    vol_mult = np.ones((n_days, n_tickers))
    spread = np.full((n_days, n_tickers), 0.006)
    gap = np.zeros((n_days, n_tickers))
    events = []
    truth = Knowability.UNKNOWN
    direction = 1
    if kind == "precursor":
        ret[D - 40:D - 12, 0] += 0.006                    # a run-up into a 52-week high
        ret[D - 12:D + 1, 0] = rng.normal(0.0004, 0.0035, 13)   # then a tight coil
        spread[D - 12:D + 1, 0] = 0.0012
        vol_mult[D - 6:D + 1, 0] = 2.6
        ret[D + 1:D + 4, 0] += 0.045
        truth = Knowability.PREDICTABLE
    elif kind == "external":
        rm[D + 1:D + 4] = -0.022
        ret = beta[None, :] * rm[:, None] + idio
        direction, truth = -1, Knowability.EXTERNALLY_CAUSED
    elif kind == "news":
        ret[D + 2, 0] += 0.16
        gap[D + 2, 0] = 0.14
        events.append({"ticker": "T00", "kind": "EARN", "accepted": (dates[D + 2] + pd.Timedelta(hours=13)).tz_localize("UTC")})
        truth = Knowability.INFORMATIONALLY_UNAVAILABLE
    elif kind == "unknown":
        ret[D + 1:D + 4, 0] += 0.045
        truth = Knowability.UNKNOWN
    else:
        ret[D + 1:D + 4, 0] += 0.045
        truth = Knowability.DATA_FAILURE
    close = 100.0 * np.cumprod(1.0 + ret, axis=0)
    prev = np.vstack([close[:1], close[:-1]])
    opn = prev * (1.0 + gap + rng.normal(0.0, 0.0015, ret.shape))
    hi = np.maximum(opn, close) * (1.0 + spread * rng.uniform(0.6, 1.0, ret.shape))
    lo = np.minimum(opn, close) * (1.0 - spread * rng.uniform(0.6, 1.0, ret.shape))
    volume = 1.0e6 * np.exp(rng.normal(0.0, 0.25, ret.shape)) * vol_mult
    frames = {"Close": close, "Open": opn, "High": hi, "Low": lo, "Volume": volume}
    if kind == "data_failure":
        frames.pop("Volume")
    stocks = {f: pd.DataFrame(v, index=dates, columns=tick) for f, v in frames.items()}
    spy = 300.0 * np.cumprod(1.0 + rm)
    vix = 17.0 + np.cumsum(rng.normal(0, 0.15, n_days)) * 0.1 + np.where(np.arange(n_days) > D, 12.0 if kind == "external" else 0.0, 0.0)
    m_cols = ["SPY", "^VIX", "^VIX3M"]
    m_close = np.column_stack([spy, vix, np.full(n_days, 19.0)])
    market = {f: pd.DataFrame(m_close * (1.0 + (0.0 if f == "Close" else 0.0005)), index=dates, columns=m_cols)
              for f in ("Close", "Open", "High", "Low", "Volume")}
    market["Volume"] = market["Volume"] * 0.0 + 1.0
    old = [{"ticker": t, "kind": "PERIODIC", "accepted": (dates[D - 20] + pd.Timedelta(hours=21)).tz_localize("UTC")} for t in tick[:8]]
    ev = pd.DataFrame(old + events)
    mdates = pd.date_range("2019-11-01", dates[D], freq="MS")     # monthly through the decision: the scrambler's fabricated rows land after it
    macro = pd.DataFrame({"date": mdates, "series": "cpi", "value": 2.0 + 0.1 * np.sin(np.arange(len(mdates)))})
    store = store_from_feed_data((stocks, market, ev, None, None), macro=macro)
    peers = tuple(tick[1:9])
    cal = store.cal
    direction_hint = direction
    ev_spec = EventSpec.make("T00", dates[D], dates[D + 3], direction_hint, cal, category="mover", peers=peers)
    old_prov = Provenance("2020-01-01T00:00:00", "2019-12-01", "planted", outcomes_seen_through="2019-12-01")
    late_prov = Provenance("2021-01-01T00:00:00", str(dates[D + 8].date()), "planted", outcomes_seen_through=str(dates[D + 8].date()))
    pats = [PlantedPattern("P_old", "quiet_pattern", 0.05, 0.9, old_prov, lambda f: False),
            PlantedPattern("P_late", "learned_later_squeeze", 0.06, 0.8, late_prov, lambda f: True)]
    ctx = {"m_spy_ma200": 0.02, "m_spy_ma50": 0.01, "m_spy_r5": 0.003, "m_vix": 17.0, "m_vix_term": 0.9, "m_vix_chg5": 0.0}
    mem = PlantedMemoryStore([PlantedMemory("M_old", "pattern", ctx, 0.3, "2019-10-01", "2019-11-01"),
                              PlantedMemory("M_late", "pattern", ctx, 0.3, str(dates[D + 1].date()), str(dates[D + 9].date()))])
    prov = Providers(feature_fn=lambda g, t, asof: {"f_r5": float(g.wide("prices", "Close", tickers=[t], lookback=6).iloc[-1, 0]
                                                              / g.wide("prices", "Close", tickers=[t], lookback=6).iloc[0, 0] - 1.0)},
                     patterns=pats, memory=mem, sic={t: 3571 for t in tick})
    return PlantedCase(kind, store, ev_spec, prov, truth, dates[-1] + pd.Timedelta(days=1), D)


# ------------------------------------------------------------------------------------------------ audits of the report itself


def audit_report_timestamps(report: CounterfactualReport, cal: pit.Calendar | None = None, boundary_sessions: int = 0) -> list[str]:
    """Recompute every item's availability class and lag from its own dates and compare with what the report stored. A report
    whose stored class disagrees with its own timestamps (tampered, or built by a buggy path) is not trustworthy."""
    t0, t1 = report.event.decision_ts, report.event.event_start
    errs = []
    groups = (("state", report.knowledge_state_at_decision.items),
              ("future", tuple(u.item for u in report.future_information_used_by_auditor)),
              ("unavailable", report.information_that_was_unavailable), ("available", report.information_that_would_have_been_available))
    for label, items in groups:
        for it in items:
            if it.note.startswith("outcome") or it.availability == Availability.UNAVAILABLE and it.domain in (Domain.PRICE, Domain.VOLUME,
                                                                                                             Domain.MARKET_STATE, Domain.CROSS_SECTION):
                continue                      # outcome facts are forced UNAVAILABLE by construction
            klass, lag = classify_availability(it.effective, it.available, t0, t1, cal, boundary_sessions)
            if klass != it.availability:
                errs.append(f"{label}:{it.key}: stored {it.availability.value} but its dates say {klass.value}")
            elif lag != it.lag_sessions:
                errs.append(f"{label}:{it.key}: stored lag {it.lag_sessions} but recomputed {lag}")
    return errs


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a share k/n; the honest error bar on 'x% of moves were unpredictable'."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish effective sample size of the sampling weights: how many equally weighted reports the sample is worth."""
    w = np.asarray(list(weights), dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def share_table(rows: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Class shares within each group, weighted, with Wilson intervals on the effective sample. Rows: group x class."""
    if not len(rows):
        return pd.DataFrame(columns=[by, "knowability", "share", "lo", "hi", "n"])
    out = []
    for g, d in rows.groupby(rows[by].replace("", "uncategorised") if by == "category" else rows[by]):
        n_eff = effective_sample_size(d["weight"])
        tot = float(d["weight"].sum())
        for k, dk in d.groupby("knowability"):
            share = float(dk["weight"].sum() / tot)
            lo, hi = wilson_interval(share * n_eff, n_eff)
            out.append({by: g, "knowability": k, "share": share, "lo": lo, "hi": hi, "n": int(len(dk))})
    return pd.DataFrame(out)


def sampling_audit(rows: pd.DataFrame, chosen_from: int) -> dict[str, Any]:
    """Weights must reconstruct the population the sample was drawn from: their sum should match the eligible event count
    (within sampling noise) when every event was eligible. Returns the ratio and the effective sample size."""
    w = rows["weight"].astype(float) if len(rows) else pd.Series(dtype=float)
    total = float(w.sum())
    ratio = total / chosen_from if chosen_from else float("nan")
    return {"n": int(len(rows)), "weighted_total": total, "eligible": int(chosen_from), "ratio": ratio,
            "ess": effective_sample_size(w), "biased": bool(chosen_from and abs(ratio - 1.0) > 0.35)}


def explain(report: CounterfactualReport, top: int = 6) -> str:
    """A plain-language account of one event for the research log (trusted side; carries real dates, never given to a trader)."""
    e, ks = report.event, report.knowledge_state_at_decision
    lines = [f"{e.ticker} moved {report.attribution.realized:+.1%} after the close of {e.decision_ts}: {report.knowability.value} "
             f"(confidence {report.confidence_in_classification.overall:.2f})." if report.attribution.realized is not None
             else f"{e.ticker}: the move could not be measured: {report.knowability.value}."]
    lines.append(f"The state at the decision covered {sum(1 for v in ks.coverage().values() if v)}/{len(ALL_DOMAINS)} domains"
                 + (f"; missing {', '.join(ks.missing_domains())}." if ks.missing_domains() else "."))
    ev = sorted(report.evidence, key=lambda x: -x.strength)[:top]
    lines += [f"- pointer {x.domain.value}/{x.name}: strength {x.strength:.2f} ({x.channel.value.lower()}"
              + ("" if x.agrees is None else ", agreed" if x.agrees else ", opposed") + ")" for x in ev]
    if not ev:
        lines.append("- nothing in the available information pointed toward this move.")
    if report.information_that_was_unavailable:
        lines.append("Not knowable then: " + ", ".join(sorted({i.key for i in report.information_that_was_unavailable})[:top]) + ".")
    att = report.attribution
    if att.complete:
        lines.append(f"Attribution: market {att.market_share:.0%}, peers {att.peer_share:.0%}, idiosyncratic {att.idio_share:.0%}.")
    lines += [f"Note: {n}" for n in report.notes]
    lines += [f"Cap: {c}" for c in report.confidence_in_classification.caps]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ run manifests (a result needs its code)


def manifest_of(cfg: CounterfactualConfig, srcs: SourceMap, seed: int) -> dict:
    """What a run is: config, source map, seed and the hash of THIS module's code. Results from a different manifest must not
    be mixed into one output directory (a code edit mid-run once made two halves of a result incomparable)."""
    src = Path(__file__).read_bytes()
    return {"config": stable_hash(cfg, 16), "sources": stable_hash(srcs, 16), "seed": int(seed),
            "module": stable_hash(src.decode("utf-8", "replace"), 16), "version": cfg.version}


def check_manifest(out_dir: str | os.PathLike, cfg: CounterfactualConfig, srcs: SourceMap, seed: int, allow_change: bool = False) -> dict:
    """Write the manifest on first use; on resume refuse a directory written under a different one."""
    path = Path(out_dir) / "cf_manifest.json"
    cur = manifest_of(cfg, srcs, seed)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != cur and not allow_change:
            changed = sorted(k for k in cur if old.get(k) != cur[k])
            raise ValueError(f"{path} was written under a different manifest (changed: {changed}); results would not be comparable")
        return old
    path.write_text(canonical_json(cur), encoding="utf-8")
    return cur


# ------------------------------------------------------------------------------------------------ scheduled catalysts, replay, event discovery


def partial_knowledge(report: CounterfactualReport) -> list[dict]:
    """Catalysts whose TIMING was public at T but whose CONTENT was not: a scheduled filing/announcement in the state paired
    with the post-decision record of the same kind. The move was then predictable in when, not in what (section 7: known
    before vs known after are properties of pieces of information, not of events)."""
    sched = {i.name.split(":", 1)[1].upper(): i for i in report.knowledge_state_at_decision.items
             if i.domain == Domain.EVENT and i.name.startswith("scheduled:")}
    out = []
    for u in report.future_information_used_by_auditor:
        if u.used_for == "cause" and u.item.domain == Domain.EVENT and u.item.name.upper() in sched:
            s = sched[u.item.name.upper()]
            out.append({"kind": u.item.name, "timing_known_since": s.available, "content_available": u.item.available,
                        "content_class": u.item.availability.value, "relevance": u.relevance})
    return out


def replay_matches(store: pit.PITStore, event: EventSpec, now, providers: Providers | None = None,
                   cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None) -> bool:
    """Determinism: the same event assessed twice on fresh snapshots gives the same report (apart from the wall-clock stamp)."""
    cfg, srcs = cfg or CounterfactualConfig(), srcs or SourceMap()
    reps = []
    for _ in range(2):
        snap = build_snapshot(store, event.decision_ts, srcs, cfg)
        r = assess_event(store, snap, event, now, providers, cfg)
        if r is None:
            return False
        reps.append(r)
    return not compare_reports(reps[0], reps[1])


def discover_events(store: pit.PITStore, decision_ts, now, horizon_sessions: int = 5, tail: float = 0.01, min_abs: float = 0.08,
                    srcs: SourceMap | None = None, cfg: CounterfactualConfig | None = None) -> list[EventSpec]:
    """Market-wide movers for one decision date, found the way the audit world finds them: after the outcome has matured. Reads
    the window through a Guard bound to the window's last session, ranks every name by the return from the fill open to the
    window close, and keeps the extreme tails (top and bottom `tail` share, at least `min_abs` in size). Names without a
    complete window are skipped rather than filled. Empty when the window has not matured before the real clock `now`."""
    srcs, cfg = srcs or SourceMap(), cfg or CounterfactualConfig()
    t0 = pd.Timestamp(decision_ts).normalize()
    t1 = pd.Timestamp(store.cal.strict_next([t0])[0])
    end = _shift(t1, horizon_sessions - 1, store.cal)
    stub = EventSpec("stub", "X", str(t0.date()), str(t1.date()), str(end.date()), 1)
    as_of = auditor_as_of_for(stub, now, store.cal, cfg.settle_sessions)
    if as_of is None:
        return []
    g = store.view(as_of)
    C = g.wide(srcs.prices, srcs.close, start=t0, end=end)
    O = g.wide(srcs.prices, srcs.open_, start=t1, end=t1)
    if len(C) < horizon_sessions + 1 or not len(O) or pd.Timestamp(C.index[-1]) < end:
        return []
    ret = (C.iloc[-1] / O.iloc[0] - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret[(C.iloc[0].reindex(ret.index) > 0)]
    if len(ret) < 20:
        return []
    lo_q, hi_q = ret.quantile(tail), ret.quantile(1.0 - tail)
    picks = ret[((ret >= hi_q) | (ret <= lo_q)) & (ret.abs() >= min_abs)]
    out = []
    for t, r in picks.sort_values(key=lambda s: -s.abs()).items():
        cat = MoveCategory.EXTREME_UP if r > 0 else MoveCategory.EXTREME_DOWN
        out.append(EventSpec.make(str(t), t0, end, 1 if r > 0 else -1, store.cal, cat.value, float(r)))
    return out


def validate_providers(providers: Providers) -> list[str]:
    """Structural check before a long run: every pattern is pattern-like, the memory store answers memories(), and each
    memory has the fields the reconstruction reads. A provider that cannot be read would otherwise fail on the millionth event."""
    errs: list[str] = []
    for p in providers.patterns:
        errs += pattern_problems(p)
        prov = getattr(p, "provenance", None)
        if prov is not None:
            errs += [f"pattern {getattr(p, 'pattern_id', '?')}: {e}" for e in prov.check()]
    if providers.memory is not None:
        if not callable(getattr(providers.memory, "memories", None)):
            errs.append("memory store has no memories() method")
        else:
            need = ("mem_id", "kind", "context", "lean", "filed_date", "matured_at")
            for m in list(providers.memory.memories())[:50]:
                errs += [f"memory {getattr(m, 'mem_id', '?')} missing {a}" for a in need if not hasattr(m, a)]
                if hasattr(m, "matured_at") and hasattr(m, "filed_date") and as_date(m.matured_at) < as_date(m.filed_date):
                    errs.append(f"memory {m.mem_id}: matured before it was filed")
    if providers.feature_fn is not None and not callable(providers.feature_fn):
        errs.append("feature_fn is not callable")
    return errs


def run_year(store: pit.PITStore, year: int, now, out_dir: str | os.PathLike, providers: Providers | None = None,
             cfg: CounterfactualConfig | None = None, srcs: SourceMap | None = None, seed: int = 0, horizon_sessions: int = 5,
             stride: int = 5, tail: float = 0.01, min_abs: float = 0.08, max_days: int | None = None) -> BatchResult:
    """One year of the market-wide test, streamed: discover the extreme movers of every `stride`-th session (non-overlapping
    windows), assess them with the per-day cap, append to cf_<year>.jsonl and resume past what is already there. Only one
    day's snapshot is alive at a time. `now` is the real clock; windows not finished before it are skipped."""
    cfg, srcs, providers = cfg or CounterfactualConfig(), srcs or SourceMap(), providers or Providers()
    bad = validate_providers(providers)
    if bad:
        raise ValueError("providers failed validation: " + "; ".join(bad[:5]))
    sess = pd.DatetimeIndex(store.source(srcs.prices).frames[srcs.close].index).normalize()
    days = sess[sess.year == int(year)][::max(1, int(stride))]
    if max_days is not None:
        days = days[:int(max_days)]
    events: list[EventSpec] = []
    for d in days:
        events += discover_events(store, d, now, horizon_sessions, tail, min_abs, srcs, cfg)
    return run_batch(store, events, now, providers, cfg, srcs, out_dir, seed)


def coverage_report(reports: Sequence[CounterfactualReport]) -> pd.DataFrame:
    """Per domain: share of reports where the domain was reconstructed at all, mean item count, and share of those items that
    the model itself used. A domain that is empty for most events is a data or wiring gap, not a finding about the market."""
    if not reports:
        return pd.DataFrame(columns=["domain", "present_share", "mean_items", "in_model_share"])
    w = np.asarray([r.weight for r in reports], dtype=float)
    rows = []
    for d in ALL_DOMAINS:
        counts = np.asarray([sum(1 for i in r.knowledge_state_at_decision.items if i.domain == d) for r in reports], dtype=float)
        used = np.asarray([sum(1 for i in r.knowledge_state_at_decision.items if i.domain == d and i.in_model) for r in reports], dtype=float)
        rows.append({"domain": d.value, "present_share": float((w * (counts > 0)).sum() / w.sum()),
                     "mean_items": float((w * counts).sum() / w.sum()),
                     "in_model_share": float(used.sum() / counts.sum()) if counts.sum() else 0.0})
    return pd.DataFrame(rows)


def unavailable_breakdown(reports: Sequence[CounterfactualReport]) -> pd.DataFrame:
    """What kind of information was missing when the answer was 'unavailable': counts by domain and availability class over the
    reports, so 'we could not have known' can be traced to the kinds of facts that did not yet exist."""
    acc: Counter = Counter()
    for r in reports:
        for i in r.information_that_was_unavailable:
            acc[(i.domain.value, i.availability.value)] += 1
    rows = [{"domain": d, "availability": a, "n": n} for (d, a), n in sorted(acc.items())]
    return pd.DataFrame(rows, columns=["domain", "availability", "n"])


ENTRYPOINTS = {"step": step, "run_batch": run_batch, "run_year": run_year, "assess_event": assess_event,
               "release": ClassificationGate, "selfcheck": selfcheck}      # what the wave-2 research loop and reachability call
__all__ = sorted(n for n, o in dict(globals()).items() if not n.startswith("_") and getattr(o, "__module__", None) == __name__)
