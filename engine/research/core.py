"""Shared vocabulary of the research brain (contract C66 sections 1-8, 18, 29-33, 40-42).

Every engine/research module imports its states and records from here so parallel builders speak one language. It EXTENDS
engine.learning.core (C62): hashing, provenance, confidence, firewall breach and knowledge states come from there, never copied.
Two namespaces (C66 section 29): LIVE_POINT_IN_TIME_STATE is what the blind trader may see; MATURED_RESEARCH_STATE is what the
research/audit world studies. A matured record reaches the trader only through an explicit point-in-time gate (section 31)."""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from engine.learning.core import (_StrEnum, FirewallBreach, Provenance, as_date, require_past, stable_hash,  # noqa: F401
                                  Epistemic, Unknown, FailureCause, Subsystem, DecisionEffect)


class Namespace(_StrEnum):                  # section 29: no implicit shared state between the two worlds
    LIVE_POINT_IN_TIME = "LIVE_POINT_IN_TIME_STATE"
    MATURED_RESEARCH = "MATURED_RESEARCH_STATE"


class Knowability(_StrEnum):                # sections 7, 33: unknown stays unknown
    PREDICTABLE = "PREDICTABLE"
    POTENTIALLY_PREDICTABLE = "POTENTIALLY_PREDICTABLE"
    WEAKLY_PREDICTABLE = "WEAKLY_PREDICTABLE"
    UNKNOWN = "UNKNOWN"
    EXTERNALLY_CAUSED = "EXTERNALLY_CAUSED"
    INFORMATIONALLY_UNAVAILABLE = "INFORMATIONALLY_UNAVAILABLE"
    DATA_FAILURE = "DATA_FAILURE"


class Availability(_StrEnum):               # section 7: when a piece of information existed
    KNOWN_BEFORE_EVENT = "KNOWN_BEFORE_EVENT"
    KNOWN_ONLY_AFTER_EVENT = "KNOWN_ONLY_AFTER_EVENT"
    SIMULTANEOUS = "SIMULTANEOUS"
    UNCERTAIN = "UNCERTAIN_AVAILABILITY"
    UNAVAILABLE = "UNAVAILABLE"


class MoveCategory(_StrEnum):               # section 4: what the market-wide observer keeps every day
    CONSIDERED_HIGH = "CONSIDERED_HIGH"
    CONSIDERED_MEDIUM = "CONSIDERED_MEDIUM"
    REJECTED = "REJECTED"
    ABSTAINED = "ABSTAINED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    WINNER = "WINNER"
    LOSER = "LOSER"
    EXTREME_UP = "EXTREME_UP"
    EXTREME_DOWN = "EXTREME_DOWN"
    PREDICTABLE_MOVER = "PREDICTABLE_MOVER"
    UNPREDICTABLE_MOVER = "UNPREDICTABLE_MOVER"
    NEAR_MISS = "NEAR_MISS"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"


class ResearchState(_StrEnum):              # section 18 (+ lifecycle of research branches, section 20)
    QUEUED = "QUEUED"
    EXPLORING = "EXPLORING"
    PROMISING = "PROMISING"
    REPLICATING = "REPLICATING"
    ESCALATED = "ESCALATED"
    VALIDATING = "VALIDATING"
    FAILED = "FAILED"
    DORMANT = "DORMANT"
    RETIRED = "RETIRED"
    CANCELLED = "CANCELLED"


class Stage(_StrEnum):                      # section 18 escalation ladder
    CHEAP_SCREEN = "STAGE1_CHEAP_SCREEN"
    STRONGER_TESTS = "STAGE2_STRONGER_TESTS"
    CROSS_YEAR = "STAGE3_CROSS_YEAR"
    FRESH_HOLDOUT = "STAGE4_FRESH_HOLDOUT"
    INTEGRATION = "STAGE5_INTEGRATION"


class GateVerdict(_StrEnum):                # section 42: never PROMOTE on a critical failure
    PROMOTE = "PROMOTE"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"
    UNKNOWN = "UNKNOWN"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"


class Problem(_StrEnum):                    # section 0 / 34 objective hierarchy
    VOLATILITY = "VOLATILITY"
    LOSS_AVOIDANCE = "LOSS_AVOIDANCE"
    DIRECTION = "DIRECTION"
    COVERAGE = "COVERAGE"
    CONSISTENCY = "CONSISTENCY"
    EFFICIENCY = "EFFICIENCY"
    SECONDARY = "SECONDARY"
    RESEARCH_PROCESS = "RESEARCH_PROCESS"
    DATA_QUALITY = "DATA_QUALITY"


OBJECTIVE_ORDER = (Problem.VOLATILITY, Problem.LOSS_AVOIDANCE, Problem.DIRECTION, Problem.COVERAGE, Problem.CONSISTENCY,
                   Problem.EFFICIENCY, Problem.SECONDARY)          # section 34, approximately; learned weights may refine it


class Horizon(_StrEnum):                    # section 23: every pattern carries an explicit horizon
    MIN5 = "5min"
    MIN30 = "30min"
    HOUR1 = "1h"
    DAY1 = "1d"
    DAY3 = "3d"
    DAY5 = "5d"
    WEEK1 = "1w"
    WEEK2 = "2w"
    MONTH1 = "1m"


@dataclasses.dataclass(frozen=True)
class ExperimentValue:
    """Section 2/19 value vector. Estimates BEFORE a job (priority) and realised values AFTER it (accounting) use the same
    fields; None = not estimated (never read as 0)."""
    information_gain: float | None = None
    decision_value: float | None = None
    uncertainty_reduction: float | None = None
    transfer_potential: float | None = None
    failure_reduction_value: float | None = None
    loss_reduction_value: float | None = None          # section 34: expected future loss avoided
    volatility_value: float | None = None
    direction_value: float | None = None
    compute_cost: float | None = None                  # CPU-minutes
    overfit_risk: float | None = None
    redundancy: float | None = None

    def missing(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self) if getattr(self, f.name) is None)


@dataclasses.dataclass(frozen=True)
class ResearchQuestion:
    """Section 40: every question is a research object (hypotheses live in the hypothesis tree, section 41)."""
    question_id: str
    text: str                                          # identity-free: no tickers, no real dates, no years
    source: str                                        # surprise / contradiction / loss / missed_winner / break / regime / discovery / ...
    problem: Problem
    created_real: str                                  # trusted-side wall clock
    evidence_through: str                              # real date of the newest matured evidence it rests on
    success_criterion: str
    failure_criterion: str
    expected: ExperimentValue = ExperimentValue()
    parents: tuple[str, ...] = ()

    @staticmethod
    def make(text: str, source: str, problem: Problem, created_real: str, evidence_through: str,
             success: str, failure: str, **kw: Any) -> "ResearchQuestion":
        qid = "Q" + stable_hash({"t": text, "s": source, "p": str(problem), "e": evidence_through}, 12)
        return ResearchQuestion(qid, text, source, problem, created_real, evidence_through, success, failure, **kw)


@dataclasses.dataclass(frozen=True)
class MaturedRecord:
    """A research-world fact. It may enter LIVE state only via gate(now): its outcome must have matured strictly before now."""
    record_id: str
    matured_at: str
    payload: Mapping[str, Any]
    provenance: Provenance
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def gate(self, now) -> Mapping[str, Any]:
        require_past(self.matured_at, now, f"matured record {self.record_id}")
        if not self.provenance.could_exist_at(now):
            raise FirewallBreach(f"record {self.record_id} could not have existed at {now}")
        return dict(self.payload)
