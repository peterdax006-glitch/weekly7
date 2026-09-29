"""Dedicated loss research pipeline (C66 sections 5, 12, 33, 34; canons C62 section 9/23/24, C63, C64, C66).

"Study losers as aggressively as winners." For every meaningful loser - a stock that fell, or a position we lost money on even if the
stock rose - this pipeline asks every section-5 question and writes it down, with the full cause list (selection, direction, timing,
exit, risk, regime, data quality, external event, pattern failure, unknown):

    why did it fall / did we predict it / should we have / did the volatility model identify it / did the direction model
    call it positive / which evidence caused the mistake / was the mistake ...  x10 causes.

Stages (LossPipeline.study, one loss at a time; step() runs a batch)
  1. triage         LossKind: HELD_LOSS, PROFITED, AVOIDED_BY_SKILL, NEAR_MISS_LUCK, MISSED_LOSER, REJECTED_ROUTINE - an avoided loss is
                    a loss too, because dodging one by luck hides a hole;
  2. classify       engine.learning.failure.LossClassifier (eight detectors) on the real or hypothetical trade;
  3. blame          engine.learning.separation.attribute / route (the arithmetic decides who is taught);
  4. cause          the ten section-5 causes from the detectors' evidence routed to LossCause, plus two detectors the classifier lacks
                    (external event with availability, market-driven move) and the arithmetic blame; UNKNOWN is a correct answer;
  5. knowability    could the system have known (research.core.Knowability), from signal verdicts and event availability;
  6. postmortem     engine.learning.postmortem for held losses (section-24 record, hypothesis-only);
  7. cohort facts   loser signal study against non-movers (winners_losers.study_signals, cohort_dir -1), the volatility model's and the
                    direction model's ability to flag falls;
  8. loss-risk bank an opportunity-independent bank of conditions that raise loss risk (section 12), mined with Fisher tests, BH
                    control, distinct-week support and a later-slice validation; states use Epistemic;
  9. unknowns       UnknownRegistry clusters unexplained losses so recurrence becomes a research question (section 33);
 10. value          expected future loss avoided per cause (section 34) and the ResearchQuestions that follow from it.

Everything is identity-free (no ticker, no date in text). Findings are MATURED_RESEARCH_STATE and reach a decision only through
MaturedRecord.gate(now) (winners_losers.to_matured / release). Built on failure, separation, postmortem, winners_losers, research.core.
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from engine.learning.core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, Subsystem, Unknown, _StrEnum, as_date, clip01,
                                  current_code_hash, require_past, stable_hash)
from engine.learning.failure import (Classification, FailureEnv, FailureLedger, Hypothesis, LossClassifier, ramp, noisy_or,
                                     planted_case)
from engine.learning.postmortem import KnowledgeUse, PostmortemBuilder, PostmortemStore
from engine.learning.separation import Attribution, SeparationParams, attribute, route
from engine.learning.trader_view import find_violations
from engine.pattern_stats import bh_qvalues
from engine.research.core import Availability, ExperimentValue, Knowability, MaturedRecord, Problem, ResearchQuestion
from engine.research.winners_losers import (AucResult, MoveDriver, MoveExplanation, MoveRecord, ResearchParams, SignalStat,
                                            SignalVerdict, _fin, explain_move, is_control, matured_only, move_from_trade,
                                            move_surprise, stratified_auc, study_signals, to_matured, to_trade_record, verdict_map,
                                            week_of)

NL = chr(10)
FC = FailureCause


# ==================================================================================================================
# vocabulary
# ==================================================================================================================
class LossCause(_StrEnum):                    # the section-5 cause list, verbatim
    SELECTION = "SELECTION"
    DIRECTION = "DIRECTION"
    TIMING = "TIMING"
    EXIT = "EXIT"
    RISK = "RISK"
    REGIME = "REGIME"
    DATA_QUALITY = "DATA_QUALITY"
    EXTERNAL_EVENT = "EXTERNAL_EVENT"
    PATTERN_FAILURE = "PATTERN_FAILURE"
    UNKNOWN = "UNKNOWN"


NAMED_CAUSES = tuple(c for c in LossCause if c is not LossCause.UNKNOWN)


class LossKind(_StrEnum):
    HELD_LOSS = "HELD_LOSS"                   # we held it and lost money
    PROFITED = "PROFITED"                     # it fell and we made money (a short): the fall was seen
    AVOIDED_BY_SKILL = "AVOIDED_BY_SKILL"     # not held; the direction model leaned down
    NEAR_MISS_LUCK = "NEAR_MISS_LUCK"         # not held, would have been a long, ranked near the cut: dodged by chance
    MISSED_LOSER = "MISSED_LOSER"             # never considered: the volatility model did not see it
    REJECTED_ROUTINE = "REJECTED_ROUTINE"     # considered and rejected on an ordinary rank


class AnswerStatus(_StrEnum):
    ANSWERED = "ANSWERED"
    UNKNOWN = "UNKNOWN"                       # the question could not be answered from the inputs: never guessed


# how much of a kind's hypothetical loss is real exposure (section 34 accounting): a dodged loss still shows a hole
EXPOSURE = {LossKind.HELD_LOSS: 1.0, LossKind.NEAR_MISS_LUCK: 0.5, LossKind.MISSED_LOSER: 0.25, LossKind.REJECTED_ROUTINE: 0.25,
            LossKind.AVOIDED_BY_SKILL: 0.0, LossKind.PROFITED: 0.0}
# how much of a loss could be avoided if its cause were fixed, by how knowable it was (section 8)
AVOIDABLE = {Knowability.PREDICTABLE: 1.0, Knowability.POTENTIALLY_PREDICTABLE: 0.6, Knowability.WEAKLY_PREDICTABLE: 0.25,
             Knowability.UNKNOWN: 0.1, Knowability.EXTERNALLY_CAUSED: 0.05, Knowability.INFORMATIONALLY_UNAVAILABLE: 0.0,
             Knowability.DATA_FAILURE: 0.9}

CORE_QUESTIONS = ("why_fell", "predicted", "should_have_predicted", "volatility_model_identified", "direction_called_positive",
                  "evidence_caused_mistake")
CAUSE_QUESTIONS = tuple(f"mistake_is_{c.value.lower()}" for c in LossCause)
LOSS_QUESTIONS = CORE_QUESTIONS + CAUSE_QUESTIONS

# which detector (or own check) lets a cause be judged at all; if none ran, that cause's question is UNKNOWN, not "no"
DETECTOR_CAUSES: dict[str, tuple[LossCause, ...]] = {
    "selection": (LossCause.SELECTION,), "timing": (LossCause.TIMING, LossCause.EXIT), "direction": (LossCause.DIRECTION,),
    "risk": (LossCause.RISK,), "pattern": (LossCause.PATTERN_FAILURE,), "context": (LossCause.PATTERN_FAILURE,),
    "regime": (LossCause.REGIME,), "measurement": (LossCause.DATA_QUALITY,)}
FAILURE_TO_LOSS: dict[FailureCause, LossCause | None] = {
    FC.SELECTION_ERROR: LossCause.SELECTION, FC.RISK_ERROR: LossCause.RISK, FC.REGIME_CHANGE: LossCause.REGIME,
    FC.MEASUREMENT_ERROR: LossCause.DATA_QUALITY, FC.FALSE_PATTERN: LossCause.PATTERN_FAILURE,
    FC.TEMPORARY_INACTIVITY: LossCause.PATTERN_FAILURE, FC.WRONG_CONTEXT: LossCause.PATTERN_FAILURE,
    FC.WEAKENING_EFFECT: LossCause.PATTERN_FAILURE, FC.REDUNDANCY: LossCause.PATTERN_FAILURE,
    FC.INTERACTION_FAILURE: LossCause.PATTERN_FAILURE, FC.UNKNOWN: None, FC.INSUFFICIENT_EVIDENCE: None}
SUBSYSTEM_TO_LOSS = {Subsystem.SELECTION: LossCause.SELECTION, Subsystem.DIRECTION: LossCause.DIRECTION,
                     Subsystem.TIMING: LossCause.TIMING, Subsystem.EXIT: LossCause.EXIT, Subsystem.RISK: LossCause.RISK}


@dataclass(frozen=True)
class CauseParams:
    accept: float = 0.40                  # a cause needs this combined score to be named
    margin: float = 0.10                  # and must beat the runner-up by this, else CONFLICTED
    min_secondary: float = 0.30
    min_coverage: float = 0.34            # share of the classifier's detectors that must have run
    against_weight: float = 0.8
    arithmetic_weight: float = 0.6        # how far the decomposition's blame share counts as evidence
    protected_against: float = 0.25       # a subsystem the arithmetic shows did its job is argued against this hard
    dir_conf_full: float = 0.2            # a side taken with this much excess confidence (|p - 0.5|) makes a direction error real
    no_direction_model: float = 0.5       # weight of a direction error when no direction model spoke
    event_jump_lo: float = 0.3            # share of the adverse move that arrived in one day to start counting as an event
    event_jump_hi: float = 0.9
    market_share_lo: float = 0.4
    unattributed_jump: float = 0.3        # evidence given to a big one-day jump that has no attributed event
    # risk bank
    bank_min_weeks: int = 6
    bank_val_frac: float = 0.35
    bank_alpha: float = 0.10
    bank_min_lift: float = 1.3
    # unknown registry
    unknown_min_count: int = 5
    unknown_min_periods: int = 3
    # value
    min_question_value: float = 0.02

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.accept < 1 or not 0 <= self.margin < 1:
            errs.append("accept must be in (0,1) and margin in [0,1)")
        if not 0 <= self.min_coverage <= 1:
            errs.append("min_coverage must be in [0,1]")
        if self.event_jump_lo >= self.event_jump_hi:
            errs.append("event_jump_lo must be below event_jump_hi")
        if not 0 < self.bank_alpha < 0.5 or self.bank_min_lift <= 1.0:
            errs.append("bank_alpha must be in (0,0.5) and bank_min_lift above 1")
        return errs

    def hash(self) -> str:
        return stable_hash(self)


# ==================================================================================================================
# evidence, answers, findings
# ==================================================================================================================
@dataclass(frozen=True)
class LossEvidence:
    cause: LossCause
    strength: float
    supports: bool
    source: str                           # classifier | arithmetic | event | market
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "strength", clip01(self.strength))


@dataclass(frozen=True)
class Answer:
    question: str
    status: AnswerStatus
    value: Any = None
    evidence: tuple[str, ...] = ()
    strength: float | None = None

    @property
    def answered(self) -> bool:
        return self.status is AnswerStatus.ANSWERED


@dataclass(frozen=True)
class CauseAssessment:
    cause: LossCause
    score: float
    confidence: float
    unknown_state: Unknown | None
    coverage: float
    scores: Mapping[str, float]
    judged: Mapping[str, bool]            # cause -> whether any check that could see it ran
    secondary: tuple[tuple[str, float], ...]
    evidence: tuple[LossEvidence, ...]
    hypothetical: bool
    meaningful: bool
    note: str = ""

    @property
    def named(self) -> bool:
        return self.cause is not LossCause.UNKNOWN


@dataclass(frozen=True)
class LossFinding:
    rid: str
    kind: LossKind
    learned_at: str
    ret: float
    loss: float                           # magnitude of the (real or hypothetical) position loss, >= 0
    exposure: float
    severity: float
    driver: MoveDriver
    cause: LossCause
    cause_score: float
    confidence: float
    unknown_state: str
    scores: Mapping[str, float]
    secondary: tuple[tuple[str, float], ...]
    coverage: float
    knowability: Knowability
    failure_cause: str
    primary_subsystem: str
    teach: Mapping[str, float]
    postmortem_id: str
    hypothetical: bool
    answers: Mapping[str, Answer]
    answered: Mapping[str, bool]
    tags: Mapping[str, str] = field(default_factory=dict)

    def depth(self) -> int:
        return sum(1 for v in self.answered.values() if v)

    @property
    def weighted_loss(self) -> float:
        return self.loss * self.exposure


# ==================================================================================================================
# stage 1: triage
# ==================================================================================================================
def loss_kind(m: MoveRecord, p: ResearchParams) -> LossKind:
    """What sort of loss this is. Every meaningful loser gets a kind, and none is dropped: a dodge is a finding too."""
    if m.held:
        return LossKind.HELD_LOSS if (m.pnl or 0.0) < 0 else LossKind.PROFITED
    dp = _fin(m.dir_prob)
    if dp is not None and dp <= 0.5 - p.dir_margin:
        return LossKind.AVOIDED_BY_SKILL
    if not m.considered:
        return LossKind.MISSED_LOSER
    if m.rank_pct is not None and m.rank_pct >= p.rank_hi:
        return LossKind.NEAR_MISS_LUCK
    return LossKind.REJECTED_ROUTINE


def loss_magnitude(m: MoveRecord, p: ResearchParams) -> float:
    """Size of the position loss: real pnl for a held position, else what the side it would have taken would have lost."""
    if m.held and _fin(m.pnl) is not None:
        return max(0.0, -float(m.pnl))
    return max(0.0, -(m.hyp_side(p) * m.ret) + m.cost)


def loss_severity(m: MoveRecord, p: ResearchParams, kind: LossKind) -> float:
    """Priority for study: magnitude x model surprise. Kinds with no exposure keep a floor so a skilled dodge is still readable."""
    mag = loss_magnitude(m, p) if EXPOSURE[kind] > 0 else 0.3 * abs(m.ret)
    s = move_surprise(m)
    return float(mag * (1.0 + (1.0 if s is None else s)))


# ==================================================================================================================
# stage 4: cause evidence
# ==================================================================================================================
def direction_factor(m: MoveRecord, side: int, cp: CauseParams) -> float:
    """How much a 'the side was wrong' finding counts as a direction ERROR: zero for a coin-flip call (that is noise, and noise is
    UNKNOWN, not a mistake), full when the model backed the side it took with real confidence."""
    dp = _fin(m.dir_prob)
    if dp is None:
        return cp.no_direction_model
    conf = dp if side > 0 else 1.0 - dp
    return clip01((conf - 0.5) / cp.dir_conf_full)


def route_classification(cls: Classification, m: MoveRecord, side: int, cp: CauseParams) -> list[LossEvidence]:
    """The failure classifier's evidence, re-routed onto the ten section-5 causes. Timing evidence tagged EXIT (shake-outs,
    round trips) goes to EXIT; direction is read from the subsystem vote and scaled by how confident the call was."""
    out: list[LossEvidence] = []
    for cs in cls.scores:
        for group, sup in ((cs.support, True), (cs.against, False)):
            for e in group:
                if e.cause is FC.TIMING_ERROR:
                    target = LossCause.EXIT if e.subsystem is Subsystem.EXIT else LossCause.TIMING
                elif e.cause is FC.REVERSAL and e.subsystem is Subsystem.DIRECTION:
                    continue                                      # counted through the direction vote below
                else:
                    target = FAILURE_TO_LOSS.get(e.cause)
                if target is not None:
                    out.append(LossEvidence(target, e.strength, sup, "classifier", e.note))
    vote = float(cls.subsystem_votes.get(Subsystem.DIRECTION.value, 0.0))
    if vote > 0:
        out.append(LossEvidence(LossCause.DIRECTION, vote * direction_factor(m, side, cp), True, "classifier",
                                "the stock moved against the side taken"))
    elif vote < 0:
        out.append(LossEvidence(LossCause.DIRECTION, -vote, False, "classifier", "the side was right"))
    return out


def detect_external_event(m: MoveRecord, side: int, cp: CauseParams) -> tuple[bool, list[LossEvidence]]:
    """An external event is judged by when it was knowable (research.core.Availability) and how much of the adverse move it
    delivered in a single day. An event nobody could have known about explains the loss and ARGUES AGAINST blaming risk, selection
    and direction for it; an event that was known beforehand and traded through is a risk-policy failure as well."""
    adv = side * m.ret
    jump = _fin(m.max_day_ret)
    if adv >= 0 or (not m.events and jump is None):
        return False, []
    adv_jump = side * jump if jump is not None else None
    share = float(adv_jump / adv) if adv_jump is not None and adv_jump < 0 else 0.0
    base = ramp(share, cp.event_jump_lo, cp.event_jump_hi)
    avail = {Availability(v) for v in m.events.values()}
    ev: list[LossEvidence] = []
    if not avail:
        if share >= cp.event_jump_hi * 0.66:
            ev.append(LossEvidence(LossCause.EXTERNAL_EVENT, cp.unattributed_jump, True, "event", "large one-day jump, no event attributed"))
        return True, ev
    unforeseeable = avail & {Availability.KNOWN_ONLY_AFTER_EVENT, Availability.UNAVAILABLE, Availability.SIMULTANEOUS}
    if share < cp.event_jump_lo:
        ev.append(LossEvidence(LossCause.EXTERNAL_EVENT, 0.4, False, "event", "an event was flagged but the move was drift, not a jump"))
    elif unforeseeable:
        ev.append(LossEvidence(LossCause.EXTERNAL_EVENT, 0.95 * base, True, "event", "an unforeseeable event delivered the loss"))
        for c, w in ((LossCause.RISK, 0.5), (LossCause.SELECTION, 0.3), (LossCause.DIRECTION, 0.3)):
            ev.append(LossEvidence(c, w * base, False, "event", "an unforeseeable event is not this subsystem's error"))
    elif Availability.KNOWN_BEFORE_EVENT in avail:
        ev.append(LossEvidence(LossCause.EXTERNAL_EVENT, 0.5 * base, True, "event", "a known event delivered the loss"))
        ev.append(LossEvidence(LossCause.RISK, 0.6 * base, True, "event", "the position was carried through a known event"))
    else:
        ev.append(LossEvidence(LossCause.EXTERNAL_EVENT, 0.6 * base, True, "event", "an event of uncertain availability delivered the loss"))
    return True, ev


def detect_market_driven(m: MoveRecord, expl: MoveExplanation, side: int, cp: CauseParams) -> tuple[bool, list[LossEvidence]]:
    """A loss the whole market delivered is a regime story, not a stock-picking one."""
    if expl.market is None or expl.market_share is None or side * m.ret >= 0:
        return False, []
    adverse = side * expl.market < 0
    if not adverse or expl.market_share < cp.market_share_lo:
        return True, [LossEvidence(LossCause.REGIME, 0.3, False, "market", "the market did not deliver this loss")]
    s = min(0.85, expl.market_share * 0.9)
    return True, [LossEvidence(LossCause.REGIME, s, True, "market", "the market move explains most of the loss"),
                  LossEvidence(LossCause.SELECTION, 0.3 * expl.market_share, False, "market", "a market-wide fall is not a stock-specific miss")]


def arithmetic_evidence(att: Attribution | None, m: MoveRecord, side: int, cp: CauseParams) -> list[LossEvidence]:
    """The decomposition's blame shares, as evidence beside the detectors: a loss the arithmetic pins on timing supports TIMING even
    if no detector said so; a subsystem that demonstrably did its job is argued against."""
    if att is None or att.basis in ("insufficient", "no_loss"):
        return []
    out = []
    mag = ramp(att.magnitude, 0.005, 0.03)
    for s in Subsystem:
        share = att.share(s)
        cause = SUBSYSTEM_TO_LOSS[s]
        if share > 0:
            f = direction_factor(m, side, cp) if s is Subsystem.DIRECTION else 1.0
            out.append(LossEvidence(cause, share * mag * cp.arithmetic_weight * f, True, "arithmetic", f"blame share {share:.2f}"))
        elif s in att.protected:
            out.append(LossEvidence(cause, cp.protected_against, False, "arithmetic", "did its job"))
    return out


def combine_causes(evidence: Sequence[LossEvidence], against_weight: float) -> dict[LossCause, float]:
    sup: dict[LossCause, list[float]] = defaultdict(list)
    con: dict[LossCause, list[float]] = defaultdict(list)
    for e in evidence:
        (sup if e.supports else con)[e.cause].append(e.strength)
    return {c: clip01(noisy_or(sup[c]) * (1.0 - against_weight * max(con[c], default=0.0))) for c in NAMED_CAUSES}


def decide_cause(scores: Mapping[LossCause, float], coverage: float, cp: CauseParams
                 ) -> tuple[LossCause, float, float, Unknown | None, tuple[tuple[str, float], ...], str]:
    """(cause, score, confidence, unknown_state, secondary, note). UNKNOWN is returned - never forced - when coverage is too low,
    when two causes are indistinguishable, or when nothing reaches the acceptance level."""
    if coverage < cp.min_coverage:
        return LossCause.UNKNOWN, 0.0, 0.0, Unknown.INSUFFICIENT_DATA, (), f"only {coverage:.0%} of the checks could run"
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].value))
    secondary = tuple((c.value, s) for c, s in ranked if s >= cp.min_secondary)
    accepted = [(c, s) for c, s in ranked if s >= cp.accept]
    if not accepted:
        return LossCause.UNKNOWN, 0.0, 0.0, Unknown.UNKNOWN, secondary, "checks ran; no cause reached the acceptance level"
    top = accepted[0]
    if len(accepted) > 1 and top[1] - accepted[1][1] < cp.margin:
        return (LossCause.UNKNOWN, 0.0, 0.0, Unknown.CONFLICTED, secondary,
                f"{top[0].value} and {accepted[1][0].value} are indistinguishable")
    runner = accepted[1][1] if len(accepted) > 1 else 0.0
    conf = clip01(top[1] * (0.5 + 0.5 * coverage) * (0.6 + 0.4 * min(1.0, (top[1] - runner) / 0.5)))
    return top[0], top[1], conf, None, tuple((c, s) for c, s in secondary if c != top[0].value), ""


def assess_cause(m: MoveRecord, t, cls: Classification, att: Attribution | None, expl: MoveExplanation, cp: CauseParams,
                 p: ResearchParams) -> CauseAssessment:
    """Stage 4 for one loss. A loss the classifier calls not meaningful (variance, not a loss at all for the hypothetical side) is
    answered 'no mistake' for every cause, honestly, rather than being sent looking for a cause."""
    side = t.side
    hypothetical = not m.held
    if not cls.meaningful:
        zero = {c.value: 0.0 for c in NAMED_CAUSES}
        return CauseAssessment(LossCause.UNKNOWN, 0.0, 0.0, Unknown.UNKNOWN, 0.0, zero, {c.value: True for c in NAMED_CAUSES}, (), (),
                               hypothetical, False, cls.note)
    ev = route_classification(cls, m, side, cp)
    judged = {c.value: False for c in NAMED_CAUSES}
    for det in cls.ran:
        for c in DETECTOR_CAUSES.get(det, ()):
            judged[c.value] = True
    ran_ev, ev_events = detect_external_event(m, side, cp)
    ran_mk, ev_market = detect_market_driven(m, expl, side, cp)
    ev += ev_events + ev_market + arithmetic_evidence(att, m, side, cp)
    if ran_ev:
        judged[LossCause.EXTERNAL_EVENT.value] = True
    if ran_mk:
        judged[LossCause.REGIME.value] = True
    if att is not None and att.basis not in ("insufficient",):
        for s in Subsystem:
            judged[SUBSYSTEM_TO_LOSS[s].value] = True
    if cls.cause is FC.INSUFFICIENT_EVIDENCE and not ev:
        return CauseAssessment(LossCause.UNKNOWN, 0.0, 0.0, Unknown.INSUFFICIENT_DATA, cls.coverage,
                               {c.value: 0.0 for c in NAMED_CAUSES}, judged, (), (), hypothetical, True, cls.note)
    scores = combine_causes(ev, cp.against_weight)
    coverage = cls.coverage if cls.coverage else (1.0 if ev else 0.0)
    cause, score, conf, unk, secondary, note = decide_cause(scores, max(coverage, 0.0), cp)
    return CauseAssessment(cause, score, conf, unk, coverage, {c.value: float(s) for c, s in scores.items()}, judged, secondary,
                           tuple(ev), hypothetical, True, note)


# ==================================================================================================================
# stage 5: could the system have known?
# ==================================================================================================================
def could_have_known(m: MoveRecord, side: int, cause: LossCause, verdicts: Mapping[str, SignalVerdict],
                     p: ResearchParams) -> tuple[Knowability, str]:
    """Section 8 in code: events by availability first, data faults as data failures, then whether the system's own signals, or the
    cohort's informative loser signals, already pointed at the fall. Nothing here reads anything after the decision except the
    already-matured verdicts of earlier cohorts."""
    if cause is LossCause.DATA_QUALITY:
        return Knowability.DATA_FAILURE, "the loss rests on a data fault"
    avail = {Availability(v) for v in m.events.values()}
    if cause is LossCause.EXTERNAL_EVENT or (avail and side * m.ret < 0 and cause is LossCause.UNKNOWN):
        if Availability.KNOWN_BEFORE_EVENT in avail:
            return Knowability.POTENTIALLY_PREDICTABLE, "the event was known before the decision"
        if avail & {Availability.KNOWN_ONLY_AFTER_EVENT, Availability.UNAVAILABLE}:
            return Knowability.INFORMATIONALLY_UNAVAILABLE, "the event became known only after the decision"
        if Availability.SIMULTANEOUS in avail:
            return Knowability.EXTERNALLY_CAUSED, "the event arrived with the move"
        return Knowability.UNKNOWN, "event availability is uncertain"
    pm, dp = _fin(m.prob_move), _fin(m.dir_prob)
    if pm is not None and dp is not None and pm >= p.prob_hi and dp <= 0.5 - p.dir_margin:
        return Knowability.PREDICTABLE, "the volatility and direction models both flagged it"
    active = [k for k, v in m.signals.items() if _fin(v) is not None and abs(v) >= p.active_z]
    informative = [k for k in active if verdicts.get(k) is SignalVerdict.INFORMATIVE]
    if len(informative) >= 2 or (informative and pm is not None and pm >= p.prob_hi):
        return Knowability.POTENTIALLY_PREDICTABLE, f"{len(informative)} informative loser signals were active"
    if informative or (pm is not None and pm >= p.prob_hi):
        return Knowability.WEAKLY_PREDICTABLE, "a single informative signal or an elevated move probability"
    return Knowability.UNKNOWN, "nothing at the decision pointed at this fall"


# ==================================================================================================================
# stage 7: cohort facts for the losers (fit on matured records only)
# ==================================================================================================================
@dataclass(frozen=True)
class LossCohortFacts:
    n_losers: int
    n_controls: int
    signal_stats: tuple[SignalStat, ...]
    vol_auc: AucResult | None             # can the volatility model tell a name that will fall from one that will not move?
    direction_auc: AucResult | None       # does P(down) = 1 - dir_prob rank the fallers above the non-movers?
    considered_share: float               # share of the losers that were considered at all
    held_share: float

    def verdicts(self) -> dict[str, SignalVerdict]:
        return verdict_map(self.signal_stats)


def fit_cohort(records: Sequence[MoveRecord], p: ResearchParams, weights: Mapping[str, float] | None, seed: int) -> LossCohortFacts:
    losers = [m for m in records if m.is_loser(p)]
    controls = [m for m in records if is_control(m, p)]
    sig = study_signals(losers, controls, p, weights, cohort_dir=-1, seed=seed)

    def auc_of(getter) -> AucResult | None:
        a = [(getter(m), m.decided_at) for m in losers if getter(m) is not None]
        b = [(getter(m), m.decided_at) for m in controls if getter(m) is not None]
        if not a or not b:
            return None
        return stratified_auc(np.array([x for x, _ in a], dtype=float), week_of([d for _, d in a]),
                              np.array([x for x, _ in b], dtype=float), week_of([d for _, d in b]), p.n_boot, p.min_strata_boot, seed)

    return LossCohortFacts(
        len(losers), len(controls), tuple(sig), auc_of(lambda m: m.prob_move),
        auc_of(lambda m: None if m.dir_prob is None else 1.0 - m.dir_prob),
        float(np.mean([m.considered for m in losers])) if losers else float("nan"),
        float(np.mean([m.held for m in losers])) if losers else float("nan"))


# ==================================================================================================================
# stage 8: the loss-risk knowledge bank (section 12) - independent of any opportunity bank
# ==================================================================================================================
@dataclass(frozen=True)
class RiskEntry:
    entry_id: str
    kind: str                             # condition | cause
    signal: str
    sign: int                             # +1: risk when the signal is high, -1: when it is low
    cause: str
    n_loss: int
    n_ctrl: int
    rate_loss: float
    rate_ctrl: float
    lift: float
    p: float
    q: float
    n_weeks: int
    mean_loss: float
    val_lift: float | None
    state: Epistemic
    version: int
    matured_at: str
    code_hash: str

    def matches(self, signals: Mapping[str, float], active_z: float) -> bool:
        v = _fin(signals.get(self.signal))
        return v is not None and self.kind == "condition" and self.sign * v >= active_z


class LossRiskBank:
    """Append-only versions of loss-risk entries. Nothing is overwritten: a re-mined entry is a new version. Entries reach the
    trader side only through active(now), which gates on maturity (MaturedRecord.gate). Kept apart from opportunity knowledge:
    assert_disjoint() proves no id is shared, so a loss-risk entry can never be mistaken for, or averaged into, an opportunity."""

    def __init__(self):
        self._versions: dict[str, list[RiskEntry]] = {}

    def __len__(self):
        return len(self._versions)

    def upsert(self, e: RiskEntry) -> RiskEntry:
        hist = self._versions.setdefault(e.entry_id, [])
        if hist and as_date(e.matured_at) < as_date(hist[-1].matured_at):
            raise FirewallBreach(f"risk entry {e.entry_id}: a new version cannot mature before the one it replaces")
        e = RiskEntry(**{**e.__dict__, "version": len(hist) + 1})
        hist.append(e)
        return e

    def latest(self) -> list[RiskEntry]:
        return sorted((h[-1] for h in self._versions.values()), key=lambda e: e.entry_id)

    def history(self, entry_id: str) -> list[RiskEntry]:
        return list(self._versions.get(entry_id, []))

    def active(self, now, states: Iterable[Epistemic] = (Epistemic.SUPPORTED,)) -> list[RiskEntry]:
        """Entries the trader may use at `now`: the newest version that matured strictly before `now`, in an allowed state."""
        ok = set(states)
        out = []
        for hist in self._versions.values():
            usable = [e for e in hist if as_date(e.matured_at) < as_date(now)]
            if usable and usable[-1].state in ok:
                rec = MaturedRecord(usable[-1].entry_id, usable[-1].matured_at, {"lift": usable[-1].lift},
                                    _prov(usable[-1].matured_at, usable[-1].code_hash))
                rec.gate(now)
                out.append(usable[-1])
        return sorted(out, key=lambda e: e.entry_id)

    def risk_score(self, signals: Mapping[str, float], now, active_z: float = 1.0, cap: float = 3.0) -> dict[str, Any]:
        """Combined log-lift of every supported condition entry the current signals trigger (capped), and the entries responsible."""
        hits = [e for e in self.active(now) if e.matches(signals, active_z)]
        total = float(min(cap, sum(math.log(e.lift) for e in hits if e.lift > 1.0)))
        return {"score": total, "entries": tuple(e.entry_id for e in hits), "multiplier": float(math.exp(total))}

    def assert_disjoint(self, opportunity_ids: Iterable[str]) -> None:
        clash = sorted(set(self._versions) & {str(i) for i in opportunity_ids})
        if clash:
            raise FirewallBreach(f"loss-risk ids also appear in the opportunity bank: {clash}")


def _prov(learned_at: str, code_hash: str):
    from engine.learning.core import Provenance
    return Provenance(created_real=learned_at, learned_at=learned_at, code_hash=code_hash or "n/a", outcomes_seen_through=learned_at)


def mine_risk_conditions(losers: Sequence[MoveRecord], controls: Sequence[MoveRecord], sig_stats: Sequence[SignalStat],
                         p: ResearchParams, cp: CauseParams, code_hash: str = "") -> list[RiskEntry]:
    """For each signal the loser study found associated with falls, ask: how much likelier is a fall when that signal is extreme?
    Fisher exact test (one-sided) on active-vs-inactive counts, BH control across the signals tried, support from enough distinct
    weeks, and validation on the LATER slice of the losers: a condition whose lift vanishes there is CONTRADICTED, not SUPPORTED."""
    cands = [s for s in sig_stats if s.verdict in (SignalVerdict.INFORMATIVE, SignalVerdict.NOT_GENERALISED, SignalVerdict.MISWEIGHTED)]
    if not cands or not losers or not controls:
        return []
    rows = []
    t_l = np.array([as_date(m.decided_at).toordinal() for m in losers], dtype=float)
    cut = float(np.quantile(t_l, 1.0 - cp.bank_val_frac)) if len(t_l) else 0.0
    for s in cands:
        sign = 1 if s.auc > 0.5 else -1

        def active(recs):
            return np.array([(_fin(m.signals.get(s.signal)) is not None and sign * m.signals[s.signal] >= p.active_z) for m in recs])

        al, ac = active(losers), active(controls)
        a, b, c, d = int(al.sum()), int((~al).sum()), int(ac.sum()), int((~ac).sum())
        lift = ((a + 0.5) / (a + b + 1.0)) / ((c + 0.5) / (c + d + 1.0))
        pv = float(stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1])
        weeks = len(set(week_of([m.decided_at for m, x in zip(losers, al) if x]))) if a else 0
        late = t_l > cut
        vl = None
        if late.sum() >= 5:
            a2, b2 = int((al & late).sum()), int((~al & late).sum())
            vl = ((a2 + 0.5) / (a2 + b2 + 1.0)) / ((c + 0.5) / (c + d + 1.0))
        ml = float(np.mean([abs(m.ret) for m, x in zip(losers, al) if x])) if a else 0.0
        rows.append((s, sign, a, c, lift, pv, weeks, vl, ml, a / max(1, a + b), c / max(1, c + d)))
    q = bh_qvalues(np.array([r[5] for r in rows]))
    learned = max(m.resolved_at for m in losers)
    out = []
    for r, qq in zip(rows, q):
        s, sign, a, c, lift, pv, weeks, vl, ml, rl, rc = r
        if qq <= cp.bank_alpha and lift >= cp.bank_min_lift and weeks >= cp.bank_min_weeks:
            state = Epistemic.CONTRADICTED if (vl is not None and vl < 1.0) else \
                Epistemic.SUPPORTED if (vl is not None and vl >= 1.0 and s.verdict is SignalVerdict.INFORMATIVE) else Epistemic.HYPOTHESIS
        else:
            state = Epistemic.HYPOTHESIS
        out.append(RiskEntry(stable_hash({"c": "cond", "s": s.signal, "g": sign}, 12), "condition", s.signal, sign, "", a, c,
                             float(rl), float(rc), float(lift), float(pv), float(qq), weeks, ml,
                             None if vl is None else float(vl), state, 1, learned, code_hash))
    return out


def cause_entries(findings: Sequence[LossFinding], code_hash: str = "") -> list[RiskEntry]:
    """One entry per named cause: how much of the (exposure-weighted) loss it carries and how large its losses are."""
    tot = sum(f.weighted_loss for f in findings)
    out = []
    learned = max((f.learned_at for f in findings), default="")
    for c in NAMED_CAUSES:
        sub = [f for f in findings if f.cause is c]
        if not sub:
            continue
        w = sum(f.weighted_loss for f in sub)
        out.append(RiskEntry(stable_hash({"c": "cause", "k": c.value}, 12), "cause", "", 0, c.value, len(sub), 0,
                             len(sub) / len(findings), 0.0, 1.0, 1.0, 1.0, len({f.learned_at for f in sub}),
                             float(np.mean([f.loss for f in sub])), None, Epistemic.OBSERVED if w / tot < 0.05 else Epistemic.SUPPORTED
                             if tot > 0 else Epistemic.HYPOTHESIS, 1, learned, code_hash))
    return out


# ==================================================================================================================
# stage 9: unknown losses stay unknown, and recurring unknowns become questions (section 33)
# ==================================================================================================================
def unknown_signature(m: MoveRecord, f: LossFinding, p: ResearchParams) -> str:
    dp = _fin(m.dir_prob)
    return stable_hash({
        "kind": f.kind.value, "driver": f.driver.value, "size": "big" if abs(m.ret) >= p.extreme_thr else "mid",
        "call": "none" if dp is None else "up" if dp > 0.5 + p.dir_margin else "down" if dp < 0.5 - p.dir_margin else "flat",
        "rank": "none" if m.rank_pct is None else "hi" if m.rank_pct >= p.rank_hi else "lo",
        "event": bool(m.events), "state": f.unknown_state}, 10)


class UnknownRegistry:
    """Clusters unexplained losses by a coarse, identity-free signature. A cluster that recurs in several distinct weeks is no
    longer bad luck: it is a phenomenon without a name, and it goes to the research queue as such."""

    def __init__(self):
        self._n: dict[str, int] = defaultdict(int)
        self._weeks: dict[str, set[int]] = defaultdict(set)
        self._loss: dict[str, float] = defaultdict(float)
        self._desc: dict[str, dict[str, Any]] = {}
        self._seen: set[str] = set()
        self.total = 0

    def add(self, m: MoveRecord, f: LossFinding, p: ResearchParams) -> str | None:
        if f.cause is not LossCause.UNKNOWN or f.rid in self._seen:
            return None
        self._seen.add(f.rid)
        sig = unknown_signature(m, f, p)
        self.total += 1
        self._n[sig] += 1
        self._weeks[sig].add(int(week_of([m.decided_at])[0]))
        self._loss[sig] += f.weighted_loss
        self._desc.setdefault(sig, {"kind": f.kind.value, "driver": f.driver.value, "state": f.unknown_state,
                                    "size": "big" if abs(m.ret) >= p.extreme_thr else "mid"})
        return sig

    def recurring(self, min_count: int, min_periods: int) -> list[dict[str, Any]]:
        rows = [{"signature": s, "count": n, "periods": len(self._weeks[s]), "loss": self._loss[s], **self._desc[s]}
                for s, n in self._n.items() if n >= min_count and len(self._weeks[s]) >= min_periods]
        return sorted(rows, key=lambda r: (-r["loss"], r["signature"]))

    def share_of_total(self, signature: str) -> float:
        return self._n[signature] / self.total if self.total else 0.0


# ==================================================================================================================
# stage 10: what a loss is worth studying for (section 34)
# ==================================================================================================================
@dataclass(frozen=True)
class CauseValue:
    cause: LossCause
    n: int
    loss: float                           # exposure-weighted loss carried by the cause
    share: float
    share_lo: float
    share_hi: float
    avoidable: float                      # mean avoidability of these losses, by knowability
    expected_avoided: float               # share of ALL exposure-weighted loss that fixing this cause could avoid
    unknown_share: float                  # share of this cause's losses whose knowability is UNKNOWN
    mean_confidence: float
    value: ExperimentValue


def cause_values(findings: Sequence[LossFinding], seed: int = 0, n_boot: int = 400) -> list[CauseValue]:
    """Expected future loss avoided per cause: (exposure-weighted loss of the cause x how avoidable those losses were) over all loss.
    A bootstrap over the findings gives the share an interval, so a cause that owns 30% of the loss in 5 cases is not ranked above
    one that owns 25% in 200."""
    if not findings:
        return []
    w = np.array([f.weighted_loss for f in findings], dtype=float)
    av = np.array([AVOIDABLE[f.knowability] for f in findings], dtype=float)
    cause = np.array([f.cause.value for f in findings])
    total = float(w.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(findings), size=(n_boot, len(findings)))
    out = []
    for c in LossCause:
        mask = cause == c.value
        if not mask.any():
            continue
        share = float(w[mask].sum() / total) if total > 0 else 0.0
        bt = w[idx] * mask[idx]
        den = w[idx].sum(axis=1)
        shares = np.divide(bt.sum(axis=1), den, out=np.zeros(n_boot), where=den > 0)
        lo, hi = float(np.percentile(shares, 5)), float(np.percentile(shares, 95))
        avoid = float(np.average(av[mask], weights=w[mask])) if w[mask].sum() > 0 else float(av[mask].mean())
        expected = float((w[mask] * av[mask]).sum() / total) if total > 0 else 0.0
        ukn = float(np.mean([f.knowability is Knowability.UNKNOWN for f in findings if f.cause is c]))
        conf = float(np.mean([f.confidence for f in findings if f.cause is c]))
        val = ExperimentValue(loss_reduction_value=expected, failure_reduction_value=share,
                              information_gain=ukn if c is LossCause.UNKNOWN else None,
                              decision_value=expected * (1.0 - ukn), overfit_risk=float(1.0 - conf) if c is not LossCause.UNKNOWN else None)
        out.append(CauseValue(c, int(mask.sum()), float(w[mask].sum()), share, lo, hi, avoid, expected, ukn, conf, val))
    return sorted(out, key=lambda v: (-v.expected_avoided, v.cause.value))


def tail_losses(findings: Sequence[LossFinding], q: float = 0.05) -> dict[str, Any]:
    """The worst q of the losses: their share of all exposure-weighted loss and their cause mix (the contract's 'failure regime
    responsible for the largest 5% of historical losses')."""
    if not findings:
        return {"n": 0, "share": float("nan"), "causes": {}}
    ranked = sorted(findings, key=lambda f: (-f.weighted_loss, f.rid))
    k = max(1, int(math.ceil(q * len(ranked))))
    tail = ranked[:k]
    tot = sum(f.weighted_loss for f in findings)
    mix: dict[str, int] = defaultdict(int)
    for f in tail:
        mix[f.cause.value] += 1
    return {"n": k, "share": float(sum(f.weighted_loss for f in tail) / tot) if tot > 0 else float("nan"),
            "causes": dict(sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))}


_QUESTION_TEXT = {
    LossCause.SELECTION: ("Does a stricter movement filter remove the selections that stall, without removing the ones that move?",
                          Problem.LOSS_AVOIDANCE),
    LossCause.DIRECTION: ("Which conditions make a confident direction call unreliable, and can the call be withheld there?",
                          Problem.DIRECTION),
    LossCause.TIMING: ("Does waiting for the open, or skipping large overnight gaps, recover the losses booked on the fill?",
                       Problem.LOSS_AVOIDANCE),
    LossCause.EXIT: ("Do wider or later stops cut the round-trips and shake-outs without raising the tail loss?",
                     Problem.LOSS_AVOIDANCE),
    LossCause.RISK: ("What position size and event exclusion would cap the tail losses that stops did not?", Problem.LOSS_AVOIDANCE),
    LossCause.REGIME: ("Which market conditions turn otherwise reliable patterns into losers, and can they be detected in time?",
                       Problem.LOSS_AVOIDANCE),
    LossCause.DATA_QUALITY: ("Which data faults produced the losses, and would a validation gate before the decision catch them?",
                             Problem.DATA_QUALITY),
    LossCause.EXTERNAL_EVENT: ("Which scheduled events precede the losses, and does abstaining around them avoid loss at acceptable cost?",
                               Problem.LOSS_AVOIDANCE),
    LossCause.PATTERN_FAILURE: ("Which pattern failures repeat, and can their health be read before the loss instead of after?",
                                Problem.LOSS_AVOIDANCE),
    LossCause.UNKNOWN: ("What do the recurring unexplained losses have in common that the current checks cannot see?",
                        Problem.RESEARCH_PROCESS)}


def loss_questions(values: Sequence[CauseValue], created_real: str, evidence_through: str, top_k: int = 5,
                   min_value: float = 0.02) -> list[ResearchQuestion]:
    """Research questions ranked by expected loss avoided, not by how many winners they might find (section 34). The text is checked
    for dates, years and identities before a question is made."""
    out = []
    for v in values:
        if v.expected_avoided < min_value and v.cause is not LossCause.UNKNOWN:
            continue
        if v.cause is LossCause.UNKNOWN and (v.share < 0.15 or v.n < 5):
            continue
        text, problem = _QUESTION_TEXT[v.cause]
        bad = find_violations(text)
        if bad:
            raise FirewallBreach(f"loss question text is not identity-free: {bad[0]}")
        out.append(ResearchQuestion.make(
            text, "loss", problem, created_real, evidence_through,
            f"held-out loss carried by this cause falls by at least a quarter (now {v.share:.0%} of exposure-weighted loss)",
            "the held-out loss share is unchanged or the change does not survive a different period",
            expected=v.value))
        if len(out) >= top_k:
            break
    return out


def breakdown_losses(findings: Sequence[LossFinding], key: str) -> dict[str, dict[str, Any]]:
    """Per tag value (era / winner type): loss counts by cause and kind, unknown rate, mean severity."""
    out: dict[str, dict[str, Any]] = {}
    for tag in sorted({f.tags.get(key, "?") for f in findings}):
        sub = [f for f in findings if f.tags.get(key, "?") == tag]
        cs: dict[str, int] = defaultdict(int)
        ks: dict[str, int] = defaultdict(int)
        for f in sub:
            cs[f.cause.value] += 1
            ks[f.kind.value] += 1
        out[tag] = {"n": len(sub), "causes": dict(cs), "kinds": dict(ks),
                    "unknown_rate": cs["UNKNOWN"] / len(sub), "mean_severity": float(np.mean([f.severity for f in sub])),
                    "mean_depth": float(np.mean([f.depth() for f in sub]))}
    return out


# ==================================================================================================================
# answers: the section-5 loser questions, one record per loss
# ==================================================================================================================
def build_answers(m: MoveRecord, kind: LossKind, expl: MoveExplanation, ca: CauseAssessment, know: Knowability, know_note: str,
                  weights: Mapping[str, float], verdicts: Mapping[str, SignalVerdict], p: ResearchParams) -> dict[str, Answer]:
    A = AnswerStatus
    ans: dict[str, Answer] = {}
    ans["why_fell"] = (Answer("why_fell", A.ANSWERED, expl.driver.value, expl.notes,
                              abs(expl.market_share or 0.0) if expl.driver is MoveDriver.MARKET else None)
                       if expl.driver is not MoveDriver.UNEXPLAINED else Answer("why_fell", A.UNKNOWN, None, expl.notes))
    pm, dp = _fin(m.prob_move), _fin(m.dir_prob)
    if pm is None and dp is None:
        ans["predicted"] = Answer("predicted", A.UNKNOWN, None, ("neither a move probability nor a direction call was logged",))
    else:
        moved = pm is not None and pm >= p.prob_hi
        down = dp is not None and dp <= 0.5 - p.dir_margin
        level = "FULL" if moved and down else "MOVE_ONLY" if moved else "DIRECTION_ONLY" if down else "NO"
        ans["predicted"] = Answer("predicted", A.ANSWERED, level, (f"prob_move={pm}", f"dir_prob={dp}"))
    ans["should_have_predicted"] = Answer("should_have_predicted", A.ANSWERED if know is not Knowability.UNKNOWN else A.UNKNOWN,
                                          know.value, (know_note,))
    if pm is None:
        ans["volatility_model_identified"] = Answer("volatility_model_identified", A.UNKNOWN, None, ("no move probability logged",))
    else:
        ans["volatility_model_identified"] = Answer("volatility_model_identified", A.ANSWERED, bool(m.considered and pm >= p.prob_hi),
                                                    (f"considered={m.considered}", f"prob_move={pm:.3f}"), pm)
    if dp is not None:
        sb = -math.log2(min(1 - 1e-9, max(1e-9, dp)))
        ans["direction_called_positive"] = Answer("direction_called_positive", A.ANSWERED, bool(dp >= 0.5 + p.dir_margin),
                                                  (f"dir_prob={dp:.3f}", f"surprise_bits={sb:.2f}"), dp)
    elif m.held:
        ans["direction_called_positive"] = Answer("direction_called_positive", A.ANSWERED, bool(m.side > 0),
                                                  ("no direction model; the side was chosen without one",))
    else:
        ans["direction_called_positive"] = Answer("direction_called_positive", A.UNKNOWN, None, ("no direction call for a name not held",))
    if not ca.meaningful:
        ans["evidence_caused_mistake"] = Answer("evidence_caused_mistake", A.ANSWERED, (), ("no mistake: " + ca.note,))
    else:
        side = m.hyp_side(p)
        push = sorted(((k, float(weights[k]) * float(m.signals[k])) for k in weights
                       if k in m.signals and _fin(m.signals[k]) is not None and side * float(weights[k]) * float(m.signals[k]) > 0),
                      key=lambda kv: (-abs(kv[1]), kv[0]))[:3]
        mis = sorted(k for k, v in m.signals.items() if _fin(v) is not None and abs(v) >= p.active_z
                     and verdicts.get(k) is SignalVerdict.MISWEIGHTED)
        knowledge = tuple(m.knowledge_ids[:3])
        if push or mis or knowledge:
            ans["evidence_caused_mistake"] = Answer(
                "evidence_caused_mistake", A.ANSWERED, tuple(k for k, _ in push) + tuple(mis),
                tuple(f"{k} pushed {c:+.2f}" for k, c in push) + tuple(f"{k} is misweighted" for k in mis) +
                tuple(f"knowledge {k}" for k in knowledge))
        else:
            ans["evidence_caused_mistake"] = Answer("evidence_caused_mistake", A.UNKNOWN, None, ("no weights, signals or knowledge logged",))
    for c in LossCause:
        q = f"mistake_is_{c.value.lower()}"
        if c is LossCause.UNKNOWN:
            ans[q] = Answer(q, A.ANSWERED, ca.cause is LossCause.UNKNOWN,
                            (ca.unknown_state.value if ca.unknown_state else "named",) + ((ca.note,) if ca.note else ()),
                            1.0 - max(ca.scores.values(), default=0.0))
        elif not ca.judged.get(c.value, False):
            ans[q] = Answer(q, A.UNKNOWN, None, ("no check that could see this cause had its inputs",))
        else:
            ans[q] = Answer(q, A.ANSWERED, ca.cause is c, tuple(e.note for e in ca.evidence if e.cause is c and e.supports and e.note)[:3],
                            float(ca.scores.get(c.value, 0.0)))
    return ans


# ==================================================================================================================
# the pipeline
# ==================================================================================================================
class LossPipeline:
    """One dedicated loss pipeline. fit() learns the loser cohort facts from matured records; study() runs every stage on one loss;
    both are pure functions of their inputs and `seed`."""

    def __init__(self, params: ResearchParams | None = None, cause_params: CauseParams | None = None,
                 weights: Mapping[str, float] | None = None, seed: int = 0, classifier: LossClassifier | None = None,
                 store: PostmortemStore | None = None, sep: SeparationParams | None = None):
        self.p = params or ResearchParams()
        self.cp = cause_params or CauseParams()
        errs = self.p.validate() + self.cp.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.weights = dict(weights or {})
        self.seed = seed
        self.clf = classifier or LossClassifier()
        self.sep = sep or SeparationParams()
        self.builder = PostmortemBuilder(self.clf, self.sep)
        self.store = store
        self.cohort: LossCohortFacts | None = None
        self.code_hash = current_code_hash()

    def fit(self, records: Sequence[MoveRecord], now) -> LossCohortFacts:
        ok, _ = matured_only(records, now, strict=True)
        self.cohort = fit_cohort(ok, self.p, self.weights, self.seed)
        return self.cohort

    def study(self, m: MoveRecord, env: FailureEnv | None, now, uses: Sequence[KnowledgeUse] = ()) -> tuple[LossFinding, Classification]:
        m.require_valid()
        require_past(m.resolved_at, now, f"loss {m.rid} outcome")
        if not m.is_loser(self.p):
            raise ValueError(f"record {m.rid} is not a meaningful loser (ret={m.ret:.4f}, pnl={m.pnl})")
        p = self.p
        verdicts = self.cohort.verdicts() if self.cohort else {}
        kind = loss_kind(m, p)
        t = to_trade_record(m, p)
        cls = self.clf.classify(t, env, now)
        att = attribute(t, cls, self.sep) if cls.meaningful else None
        expl = explain_move(m, p)
        ca = assess_cause(m, t, cls, att, expl, self.cp, p)
        know, know_note = could_have_known(m, t.side, ca.cause, verdicts, p)
        answers = build_answers(m, kind, expl, ca, know, know_note, self.weights, verdicts, p)
        teach: dict[str, float] = {}
        if att is not None and cls.meaningful:
            teach = dict(route(t, cls, att).weights)
        pid = ""
        if kind is LossKind.HELD_LOSS and cls.meaningful and cls.named:
            pm = self.builder.build(t, env, now, uses)
            if pm is not None:
                pid = pm.pid
                if self.store is not None:
                    self.store.append(pm)
        finding = LossFinding(
            rid=m.rid, kind=kind, learned_at=m.resolved_at, ret=float(m.ret), loss=loss_magnitude(m, p), exposure=EXPOSURE[kind],
            severity=loss_severity(m, p, kind), driver=expl.driver, cause=ca.cause, cause_score=ca.score, confidence=ca.confidence,
            unknown_state=ca.unknown_state.value if ca.unknown_state else "", scores=dict(ca.scores), secondary=ca.secondary,
            coverage=ca.coverage, knowability=know, failure_cause=cls.cause.value,
            primary_subsystem=att.primary.value if att is not None and att.primary else "", teach=teach, postmortem_id=pid,
            hypothetical=ca.hypothetical, answers=answers, answered={q: a.answered for q, a in answers.items()}, tags=dict(m.tags))
        return finding, cls


# ==================================================================================================================
# state, report and the single public entry
# ==================================================================================================================
@dataclass
class LossState:
    """What the loss research accumulates across days. Records are exceptions and controls only (rule 27)."""
    params: ResearchParams = field(default_factory=ResearchParams)
    cause_params: CauseParams = field(default_factory=CauseParams)
    weights: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0
    records: dict[str, MoveRecord] = field(default_factory=dict)
    findings: dict[str, LossFinding] = field(default_factory=dict)
    bank: LossRiskBank = field(default_factory=LossRiskBank)
    unknowns: UnknownRegistry = field(default_factory=UnknownRegistry)
    ledger: FailureLedger = field(default_factory=FailureLedger)
    store: PostmortemStore | None = None

    def add(self, recs: Iterable[MoveRecord]) -> int:
        n = 0
        for m in recs:
            m.require_valid()
            if m.rid not in self.records:
                self.records[m.rid] = m
                n += 1
        return n


@dataclass(frozen=True)
class LossReport:
    now: str
    findings: tuple[LossFinding, ...]
    n_losers_total: int
    cohort: LossCohortFacts
    cause_counts: Mapping[str, int]
    kind_counts: Mapping[str, int]
    unknown_rate: float
    values: tuple[CauseValue, ...]
    questions: tuple[ResearchQuestion, ...]
    bank_entries: tuple[RiskEntry, ...]
    recurring_unknowns: tuple[Mapping[str, Any], ...]
    tail: Mapping[str, Any]
    matured: tuple[MaturedRecord, ...]
    pending: int
    params_hash: str

    def render(self) -> str:
        lines = [f"loss research: {len(self.findings)} new of {self.n_losers_total} losers, unknown rate {self.unknown_rate:.2f}, "
                 f"{self.pending} pending  [IMPLEMENTED - NOT VALIDATED]"]
        lines += [f"  cause {k:<16}{n:>5}" for k, n in self.cause_counts.items()]
        lines += [f"  kind  {k:<18}{n:>5}" for k, n in self.kind_counts.items()]
        for v in self.values[:6]:
            lines.append(f"  value {v.cause.value:<16} avoided {v.expected_avoided:.3f} share {v.share:.2f} [{v.share_lo:.2f},{v.share_hi:.2f}]")
        for q in self.questions:
            lines.append(f"  QUESTION [{q.problem.value}] {q.text}")
        return NL.join(lines)


def _env_for(envs: Any, rid: str) -> FailureEnv | None:
    if envs is None or isinstance(envs, FailureEnv):
        return envs
    if callable(envs):
        return envs(rid)
    return envs.get(rid)


def step(state: LossState, records: Iterable[MoveRecord], now, envs: Any = None, uses: Mapping[str, Sequence[KnowledgeUse]] | None = None,
         identities: Iterable[str] = (), created_real: str | None = None) -> LossReport:
    """One loss-research step. Absorb records, refit the loser cohort on everything matured before `now`, study every loser not yet
    studied (all of them - a budget is applied by winners_losers.review_budget upstream, never here), refresh the loss-risk bank,
    register unknowns, value the causes, and return the questions that follow. Records that have not matured are counted, not used."""
    ok, pending = matured_only(list(records), now)
    state.add(ok)
    known, _ = matured_only(state.records.values(), now, strict=True)
    pipe = LossPipeline(state.params, state.cause_params, state.weights, state.seed, store=state.store)
    cohort = pipe.fit(known, now)
    fresh: list[LossFinding] = []
    for m in sorted(known, key=lambda r: (r.resolved_at, r.rid)):
        if not m.is_loser(state.params) or m.rid in state.findings:
            continue
        f, cls = pipe.study(m, _env_for(envs, m.rid), now, (uses or {}).get(m.rid, ()))
        state.findings[m.rid] = f
        state.ledger.add(cls, m.tags)
        state.unknowns.add(m, f, state.params)
        fresh.append(f)
    allf = list(state.findings.values())
    losers = [m for m in known if m.is_loser(state.params)]
    controls = [m for m in known if is_control(m, state.params)]
    entries = mine_risk_conditions(losers, controls, cohort.signal_stats, state.params, state.cause_params, pipe.code_hash)
    entries += cause_entries(allf, pipe.code_hash)
    stored = []
    for e in entries:
        hist = state.bank.history(e.entry_id)
        if not hist or hist[-1].matured_at != e.matured_at or hist[-1].state != e.state:
            stored.append(state.bank.upsert(e))
    values = cause_values(allf, state.seed)
    through = max((f.learned_at for f in allf), default=str(as_date(now)))
    qs = loss_questions(values, created_real or str(as_date(now)), through, min_value=state.cause_params.min_question_value)
    counts: dict[str, int] = defaultdict(int)
    kinds: dict[str, int] = defaultdict(int)
    for f in allf:
        counts[f.cause.value] += 1
        kinds[f.kind.value] += 1
    ukn = counts["UNKNOWN"] / len(allf) if allf else float("nan")
    matured = tuple(to_matured(f, pipe.code_hash, seed=state.seed, identities=identities) for f in fresh)
    return LossReport(str(as_date(now)), tuple(fresh), len(losers), cohort, dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                      dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))), float(ukn), tuple(values), tuple(qs), tuple(stored),
                      tuple(state.unknowns.recurring(state.cause_params.unknown_min_count, state.cause_params.unknown_min_periods)),
                      tail_losses(allf), matured, pending, state.params.hash())


# ==================================================================================================================
# the pipeline's own control experiment: planted losses with a known cause
# ==================================================================================================================
def planted_loss(cause: LossCause, seed: int = 0) -> tuple[MoveRecord, FailureEnv]:
    """A synthetic held loss whose true cause is `cause`, every other check kept quiet. Built on failure.planted_case where the
    failure classifier already has the mechanism; the timing, exit, direction and external-event cases are built here."""
    if cause is LossCause.SELECTION:
        t, env = planted_case(FC.SELECTION_ERROR, seed)
    elif cause is LossCause.RISK:
        t, env = planted_case(FC.RISK_ERROR, seed)
    elif cause is LossCause.REGIME:
        t, env = planted_case(FC.REGIME_CHANGE, seed)
    elif cause is LossCause.DATA_QUALITY:
        t, env = planted_case(FC.MEASUREMENT_ERROR, seed)
    elif cause is LossCause.PATTERN_FAILURE:
        t, env = planted_case(FC.FALSE_PATTERN, seed)
    elif cause is LossCause.UNKNOWN:
        t, env = planted_case(FC.UNKNOWN, seed)
    else:
        base, env = planted_case(FC.UNKNOWN, seed)
        kw: dict[str, Any] = {}
        if cause is LossCause.DIRECTION:
            kw = dict(dir_prob=0.88, prior_ret=0.0)
        elif cause is LossCause.TIMING:
            kw = dict(signal_ret=0.0388, entry_gap=0.06, end_ret_from_fill=-0.02, exit_ret=-0.02, pnl=-0.0205, exp_vol=0.015,
                      mfe=0.0, mae=-0.03, dir_prob=0.6)
        elif cause is LossCause.EXIT:
            kw = dict(signal_ret=0.06, entry_gap=0.0, end_ret_from_fill=0.06, exit_ret=-0.03, stop_hit=True, stop=0.03,
                      stop_fill_ret=-0.03, pnl=-0.0305, exp_vol=0.02, mfe=0.002, mae=-0.035, dir_prob=0.6)
        elif cause is LossCause.EXTERNAL_EVENT:
            kw = dict(signal_ret=-0.15, end_ret_from_fill=-0.151, exit_ret=-0.151, pnl=-0.1515, exp_vol=0.08, dir_prob=0.5,
                      mae=-0.16, exp_move=0.10)
        else:
            raise ValueError(f"no planted loss for {cause}")
        import dataclasses as _dc
        t = _dc.replace(base, **kw)
    extra: dict[str, Any] = {}
    if cause is LossCause.EXTERNAL_EVENT:
        extra = dict(events={"scheduled_report": Availability.KNOWN_ONLY_AFTER_EVENT.value}, max_day_ret=-0.14, move_day=1,
                     prob_move=0.3)
    m = move_from_trade(t, **extra)
    return m, env


def planted_battery(seeds: Sequence[int] = (0, 1, 2), now: str = "2020-06-01") -> dict[str, Any]:
    """Run the pipeline's cause assessment over every planted cause and seed: confusion, accuracy among the ones it named, and
    the share it declined to name. The foundation check that the ten-way answer is not a label-copier."""
    pipe = LossPipeline(ResearchParams(loss_thr=0.01))
    mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    named = correct = total = 0
    for c in LossCause:
        for s in seeds:
            m, env = planted_loss(c, s)
            f, _ = pipe.study(m, env, now)
            mat[c.value][f.cause.value] += 1
            total += 1
            if f.cause is not LossCause.UNKNOWN:
                named += 1
                correct += int(f.cause is c)
    return {"matrix": {k: dict(v) for k, v in mat.items()}, "n": total, "named": named,
            "acc_named": correct / named if named else float("nan"), "abstain_rate": 1.0 - named / total if total else float("nan")}


# ==================================================================================================================
# is the answer robust, is it complete, is it honest
# ==================================================================================================================
def cause_sensitivity(records: Sequence[MoveRecord], envs: Any, now, params: ResearchParams | None = None,
                      cause_params: CauseParams | None = None, weights: Mapping[str, float] | None = None, seed: int = 0
                      ) -> dict[str, Any]:
    """How much of the cause list depends on the thresholds? Every loss is re-assessed under stricter and looser acceptance, wider
    and narrower margins, and with the arithmetic evidence switched off; the share of losses whose named cause changes (or whose
    UNKNOWN turns into a name, or the reverse) is the fragility of the answer. A cause that flips under a 0.05 change was never known."""
    p, cp = params or ResearchParams(), cause_params or CauseParams()
    losers = [m for m in records if m.is_loser(p)]
    if not losers:
        return {"n": 0, "variants": {}, "fragile_share": float("nan")}
    base = LossPipeline(p, cp, weights, seed)
    ref = {m.rid: base.study(m, _env_for(envs, m.rid), now)[0].cause for m in losers}
    variants = {"accept-0.10": dict(accept=max(0.05, cp.accept - 0.10)), "accept+0.10": dict(accept=min(0.95, cp.accept + 0.10)),
                "margin-0.05": dict(margin=max(0.0, cp.margin - 0.05)), "margin+0.05": dict(margin=cp.margin + 0.05),
                "no_arithmetic": dict(arithmetic_weight=0.0), "double_arithmetic": dict(arithmetic_weight=min(1.0, cp.arithmetic_weight * 2))}
    out: dict[str, Any] = {}
    changed_any: set[str] = set()
    for name, kw in variants.items():
        pipe = LossPipeline(p, dataclasses.replace(cp, **kw), weights, seed)
        flips = {m.rid: (ref[m.rid].value, f.cause.value) for m in losers
                 for f in [pipe.study(m, _env_for(envs, m.rid), now)[0]] if f.cause is not ref[m.rid]}
        changed_any |= set(flips)
        out[name] = {"changed": len(flips), "share": len(flips) / len(losers),
                     "became_unknown": sum(1 for a, b in flips.values() if b == "UNKNOWN"),
                     "from_unknown": sum(1 for a, b in flips.values() if a == "UNKNOWN")}
    return {"n": len(losers), "variants": out, "fragile_share": len(changed_any) / len(losers)}


def audit_findings(findings: Sequence[LossFinding], identities: Iterable[str] = (), cp: CauseParams | None = None) -> list[str]:
    """Self-audit of the loss findings. Checks: every section-5 question present; `answered` agrees with the answers; a named cause is
    also the cause its own question says yes to (and only that one); a named cause reached the acceptance level; UNKNOWN carries a
    reason; exposure matches the kind; a held loss is never hypothetical; no identity or forbidden key anywhere."""
    from engine.lessons import FORBIDDEN_KEYS
    cp = cp or CauseParams()
    ids = [str(i) for i in identities if str(i)]
    errs = []
    for f in findings:
        tag = f.rid
        if set(f.answers) != set(LOSS_QUESTIONS):
            errs.append(f"{tag}: question set is not the section-5 list ({sorted(set(LOSS_QUESTIONS) ^ set(f.answers))})")
            continue
        for q, a in f.answers.items():
            if f.answered.get(q) != a.answered:
                errs.append(f"{tag}: answered flag for {q} disagrees with its answer")
        yes = [c for c in LossCause if f.answers[f"mistake_is_{c.value.lower()}"].value is True and f.answers[f"mistake_is_{c.value.lower()}"].answered]
        if f.cause is not LossCause.UNKNOWN:
            if yes != [f.cause]:
                errs.append(f"{tag}: cause {f.cause.value} but the cause questions say {[c.value for c in yes]}")
            if f.cause_score + 1e-9 < cp.accept:
                errs.append(f"{tag}: cause {f.cause.value} named below the acceptance level ({f.cause_score:.2f})")
        elif LossCause.UNKNOWN not in yes:
            errs.append(f"{tag}: cause UNKNOWN but the unknown question does not say so")
        if f.cause is LossCause.UNKNOWN and not f.unknown_state:
            errs.append(f"{tag}: UNKNOWN without a reason state")
        if abs(f.exposure - EXPOSURE[f.kind]) > 1e-12:
            errs.append(f"{tag}: exposure {f.exposure} does not match kind {f.kind.value}")
        if f.kind is LossKind.HELD_LOSS and f.hypothetical:
            errs.append(f"{tag}: a held loss marked hypothetical")
        if f.loss < 0 or not math.isfinite(f.severity):
            errs.append(f"{tag}: negative loss or non-finite severity")
        text = stable_json(f)
        errs += [f"{tag}: identity {i!r} present" for i in ids if i in text]
        errs += [f"{tag}: identity key {k!r} in tags" for k in f.tags if str(k).lower() in FORBIDDEN_KEYS]
    return errs


def question_coverage(findings: Sequence[LossFinding]) -> dict[str, float]:
    """Share of losses for which each section-5 question could be answered, lowest first: the worst-covered questions name the
    inputs the system most needs to start logging."""
    if not findings:
        return {q: 0.0 for q in LOSS_QUESTIONS}
    cov = {q: float(np.mean([f.answered.get(q, False) for f in findings])) for q in LOSS_QUESTIONS}
    return dict(sorted(cov.items(), key=lambda kv: (kv[1], kv[0])))


def data_gaps(state: LossState) -> dict[str, Any]:
    """What to collect: the classifier's missing inputs, most frequent first, and the questions that were least often answerable."""
    gaps = state.ledger.coverage_gaps()
    cov = question_coverage(list(state.findings.values()))
    return {"missing_inputs": dict(list(gaps.items())[:10]), "worst_questions": dict(list(cov.items())[:5]),
            "classifier_unknown_rate": state.ledger.unknown_rate()}


def stable_json(obj: Any) -> str:
    from engine.learning.core import canonical_json
    return canonical_json(obj)


def avoidance_skill(findings: Sequence[LossFinding], records: Sequence[MoveRecord], p: ResearchParams | None = None) -> dict[str, Any]:
    """Are the losers we did not take avoided by skill or by luck? Among the losers not held, the share the direction model leaned
    down on (AVOIDED_BY_SKILL) is compared with the share of non-movers it also leaned down on. If the two are alike, the 'avoidance'
    is the direction model's habit, not evidence that it sees falls coming. One-sided Fisher exact test."""
    p = p or ResearchParams()
    by_kind: dict[str, int] = defaultdict(int)
    for f in findings:
        by_kind[f.kind.value] += 1
    not_held = [f for f in findings if f.kind is not LossKind.HELD_LOSS and f.kind is not LossKind.PROFITED]
    skill = sum(1 for f in not_held if f.kind is LossKind.AVOIDED_BY_SKILL)
    controls = [m for m in records if is_control(m, p) and m.dir_prob is not None]
    leaned_down = sum(1 for m in controls if m.dir_prob <= 0.5 - p.dir_margin)
    out: dict[str, Any] = {"kinds": dict(by_kind), "not_held": len(not_held), "skill": skill,
                           "skill_share": skill / len(not_held) if not_held else float("nan"),
                           "control_down_share": leaned_down / len(controls) if controls else float("nan")}
    if not_held and controls:
        table = [[skill, len(not_held) - skill], [leaned_down, len(controls) - leaned_down]]
        out["p_skill_above_chance"] = float(stats.fisher_exact(table, alternative="greater")[1])
        out["verdict"] = "SKILL" if out["p_skill_above_chance"] <= p.alpha and out["skill_share"] > out["control_down_share"] else "NOT_DISTINGUISHABLE_FROM_HABIT"
    else:
        out["p_skill_above_chance"] = float("nan")
        out["verdict"] = "UNDERPOWERED"
    return out


def context_concentration(losers: Sequence[MoveRecord], controls: Sequence[MoveRecord], tail: Sequence[str] = (),
                          p: ResearchParams | None = None, min_n: int = 8) -> list[dict[str, Any]]:
    """Which market conditions do the (tail) losses share? Per context dimension: mean value among the chosen losers against the
    controls, Welch t-test, BH q-value across dimensions. This is the contract's 'failure regime responsible for the largest losses':
    a regime is a condition the tail losses hold in common and the ordinary days do not."""
    p = p or ResearchParams()
    chosen = [m for m in losers if not tail or m.rid in set(tail)]
    dims = sorted({k for m in chosen for k in m.context} & {k for m in controls for k in m.context})
    rows = []
    for d in dims:
        a = np.array([m.context[d] for m in chosen if _fin(m.context.get(d)) is not None], dtype=float)
        b = np.array([m.context[d] for m in controls if _fin(m.context.get(d)) is not None], dtype=float)
        if len(a) < min_n or len(b) < min_n or (a.var() == 0 and b.var() == 0):
            continue
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        rows.append({"dim": d, "n_loss": len(a), "n_ctrl": len(b), "mean_loss": float(a.mean()), "mean_ctrl": float(b.mean()),
                     "smd": float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0, "p": float(pv)})
    if rows:
        q = bh_qvalues(np.array([r["p"] for r in rows]))
        for r, qq in zip(rows, q):
            r["q"] = float(qq)
            r["regime"] = bool(qq <= p.alpha and abs(r["smd"]) >= 0.3)
    return sorted(rows, key=lambda r: (-abs(r["smd"]), r["dim"]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion: honest at small n, where the normal interval says 0% +/- 0%."""
    if n <= 0:
        return 0.0, 1.0
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def pattern_blame(records: Sequence[MoveRecord], p: ResearchParams | None = None, min_uses: int = 10) -> list[dict[str, Any]]:
    """Per pattern id over the HELD positions: uses, loss rate with a Wilson interval, mean pnl, and lift over the overall loss rate.
    A pattern whose lower interval bound is above the overall rate is a pattern-failure candidate (the classifier's per-loss
    PATTERN_FAILURE calls should concentrate on these); one whose upper bound is below it is exonerated."""
    p = p or ResearchParams()
    held = [m for m in records if m.held and m.pnl is not None]
    if not held:
        return []
    overall = sum(1 for m in held if m.pnl < 0) / len(held)
    uses: dict[str, list[float]] = defaultdict(list)
    for m in held:
        for pid in m.pattern_ids:
            uses[pid].append(float(m.pnl))
    rows = []
    for pid, pnls in uses.items():
        if len(pnls) < min_uses:
            continue
        k = sum(1 for x in pnls if x < 0)
        lo, hi = wilson(k, len(pnls))
        rows.append({"pattern": pid, "uses": len(pnls), "loss_rate": k / len(pnls), "lo": lo, "hi": hi, "mean_pnl": float(np.mean(pnls)),
                     "lift": (k / len(pnls)) / overall if overall > 0 else float("nan"),
                     "status": "CANDIDATE_FAILURE" if lo > overall else "EXONERATED" if hi < overall else "UNDETERMINED"})
    return sorted(rows, key=lambda r: (-r["lift"] if math.isfinite(r["lift"]) else 0.0, r["pattern"]))


def missed_loser_report(findings: Sequence[LossFinding], records: Sequence[MoveRecord], cohort: LossCohortFacts,
                        p: ResearchParams | None = None, top: int = 10) -> dict[str, Any]:
    """The losers mirror of the missed-winner study: the losers we never considered (or dodged by luck), ranked by severity, and for
    each whether the signals that mark losers were ALREADY present (the system had the information and did not use it) or not (no
    information existed in the inputs)."""
    p = p or ResearchParams()
    by_rid = {m.rid: m for m in records}
    verdicts = cohort.verdicts()
    rows = []
    for f in findings:
        if f.kind not in (LossKind.MISSED_LOSER, LossKind.NEAR_MISS_LUCK):
            continue
        m = by_rid.get(f.rid)
        active = [] if m is None else [k for k, v in m.signals.items() if _fin(v) is not None and abs(v) >= p.active_z
                                       and verdicts.get(k) is SignalVerdict.INFORMATIVE]
        rows.append({"rid": f.rid, "kind": f.kind.value, "severity": f.severity, "driver": f.driver.value, "cause": f.cause.value,
                     "had_signal": bool(active), "informative_active": len(active)})
    rows.sort(key=lambda r: (-r["severity"], r["rid"]))
    n = len(rows)
    return {"n": n, "had_signal_share": float(np.mean([r["had_signal"] for r in rows])) if rows else float("nan"),
            "by_driver": dict(sorted(_count(r["driver"] for r in rows).items())), "top": rows[:top]}


def _count(it: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in it:
        out[x] += 1
    return dict(out)


# ---- bridge to the existing lesson memory (engine.lessons): the same five mistake kinds, one vocabulary
_LESSON_KIND = {LossCause.EXIT: "missed_exit", LossCause.RISK: "oversized_loser", LossCause.REGIME: "regime_misread"}


def lesson_kind(f: LossFinding) -> str:
    """The engine.lessons mistake kind a finding corresponds to, so the lesson book and this pipeline never disagree in words."""
    from engine.lessons import KINDS
    if f.kind is LossKind.MISSED_LOSER:
        return "none"
    kind = _LESSON_KIND.get(f.cause, "bad_entry")
    assert kind in KINDS
    return kind


# ---- persistence: findings round-trip through JSON so a long run can checkpoint (no file I/O here)
def finding_to_dict(f: LossFinding) -> dict[str, Any]:
    import json
    from engine.learning.core import canonical_json
    return json.loads(canonical_json(f))


def finding_from_dict(d: Mapping[str, Any]) -> LossFinding:
    d = dict(d)
    answers = {q: Answer(a["question"], AnswerStatus(a["status"]), _tuplify(a["value"]), tuple(a["evidence"]), a["strength"])
               for q, a in d["answers"].items()}
    return LossFinding(
        rid=d["rid"], kind=LossKind(d["kind"]), learned_at=d["learned_at"], ret=d["ret"], loss=d["loss"], exposure=d["exposure"],
        severity=d["severity"], driver=MoveDriver(d["driver"]), cause=LossCause(d["cause"]), cause_score=d["cause_score"],
        confidence=d["confidence"], unknown_state=d["unknown_state"], scores=dict(d["scores"]),
        secondary=tuple((c, s) for c, s in d["secondary"]), coverage=d["coverage"], knowability=Knowability(d["knowability"]),
        failure_cause=d["failure_cause"], primary_subsystem=d["primary_subsystem"], teach=dict(d["teach"]),
        postmortem_id=d["postmortem_id"], hypothetical=d["hypothetical"], answers=answers,
        answered={q: bool(v) for q, v in d["answered"].items()}, tags=dict(d["tags"]))


def _tuplify(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def dumps_findings(findings: Sequence[LossFinding]) -> str:
    """JSON lines, one finding per line, in a stable order."""
    import json
    return NL.join(json.dumps(finding_to_dict(f), sort_keys=True) for f in sorted(findings, key=lambda f: f.rid))


def loads_findings(text: str) -> list[LossFinding]:
    import json
    return [finding_from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


# ==================================================================================================================
# what the losses propose: hypotheses (never changes), concentration, and the kind x cause table
# ==================================================================================================================
_HYPOTHESIS = {
    LossCause.SELECTION: (Subsystem.SELECTION, DecisionEffect.RANKING, FC.SELECTION_ERROR,
                          "demote picks whose expected move is small relative to the realised movement of similar names"),
    LossCause.DIRECTION: (Subsystem.DIRECTION, DecisionEffect.DIRECTION, FC.UNKNOWN,
                          "withhold the side when the direction call is confident but the volatility model is not"),
    LossCause.TIMING: (Subsystem.TIMING, DecisionEffect.TIMING, FC.TIMING_ERROR,
                       "skip or delay entries when the overnight gap is adverse"),
    LossCause.EXIT: (Subsystem.EXIT, DecisionEffect.EXIT, FC.TIMING_ERROR,
                     "replace the fixed stop with one that survives the typical shake-out depth"),
    LossCause.RISK: (Subsystem.RISK, DecisionEffect.POSITION_SIZE, FC.RISK_ERROR,
                     "cap position size by expected volatility and by proximity to known events"),
    LossCause.REGIME: (Subsystem.SELECTION, DecisionEffect.CONFIDENCE, FC.REGIME_CHANGE,
                       "shrink positions when market context is far from its reference"),
    LossCause.DATA_QUALITY: (Subsystem.SELECTION, DecisionEffect.NONE, FC.MEASUREMENT_ERROR,
                             "quarantine suspect bars before they reach features"),
    LossCause.EXTERNAL_EVENT: (Subsystem.RISK, DecisionEffect.ABSTENTION, FC.UNKNOWN,
                               "abstain around scheduled events whose outcome is unknowable"),
    LossCause.PATTERN_FAILURE: (Subsystem.SELECTION, DecisionEffect.PATTERN_WEIGHTING, FC.FALSE_PATTERN,
                                "demote the patterns that recur in these losses pending re-test"),
}


def hypotheses_from_findings(findings: Sequence[LossFinding], min_support: int = 5) -> list[Hypothesis]:
    """One hypothesis per cause that has enough distinct losses behind it - a PROPOSED change with a test plan, never a change.
    Uses the existing failure.Hypothesis type (no new lesson type). UNKNOWN produces a data-collection hypothesis only."""
    out: dict[str, Hypothesis] = {}
    for c in NAMED_CAUSES:
        sub = [f for f in findings if f.cause is c]
        if len(sub) < min_support:
            continue
        s, eff, fcause, text = _HYPOTHESIS[c]
        weeks = len({f.learned_at for f in sub})
        h = Hypothesis(stable_hash({"src": "loss_pipeline", "cause": c.value}, 16), text, s, eff, fcause,
                       ("replay the rule on periods later than the losses that produced it, with an embargo",
                        "compare its avoided loss with the same rule fitted to shuffled outcomes",
                        "count what it costs in winners lost"),
                       min_support=len(sub), min_periods=weeks, source="loss_pipeline",
                       basis={"n": len(sub), "mean_loss": float(np.mean([f.loss for f in sub])), "mean_conf": float(np.mean([f.confidence for f in sub]))})
        out[h.hid] = h
    unk = [f for f in findings if f.cause is LossCause.UNKNOWN]
    if len(unk) >= min_support:
        h = Hypothesis(stable_hash({"src": "loss_pipeline", "cause": "UNKNOWN"}, 16),
                       "collect the inputs that would let recurring unexplained losses be classified", Subsystem.SELECTION,
                       DecisionEffect.RESEARCH_PRIORITY, FC.INSUFFICIENT_EVIDENCE,
                       ("count how many later unexplained losses become explainable with the new inputs",),
                       min_support=len(unk), min_periods=len({f.learned_at for f in unk}), source="loss_pipeline",
                       basis={"unknown_share": len(unk) / len(findings)})
        out[h.hid] = h
    for h in out.values():
        errs = h.validate()
        if errs:
            raise ValueError("; ".join(errs))
    return sorted(out.values(), key=lambda h: h.hid)


def loss_concentration(findings: Sequence[LossFinding]) -> dict[str, float]:
    """How concentrated is the loss? Herfindahl index of the cause shares (1 = one cause owns everything), and the Gini
    coefficient of the loss sizes (1 = one loss is everything). A concentrated loss is an opportunity: one fix removes most of it."""
    w = np.array([f.weighted_loss for f in findings], dtype=float)
    if len(w) == 0 or w.sum() <= 0:
        return {"hhi": float("nan"), "gini": float("nan"), "n": len(w)}
    by_cause: dict[str, float] = defaultdict(float)
    for f in findings:
        by_cause[f.cause.value] += f.weighted_loss
    shares = np.array(list(by_cause.values())) / w.sum()
    srt = np.sort(w)
    n = len(srt)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * srt) / (n * srt.sum())) - (n + 1) / n)
    return {"hhi": float(np.sum(shares ** 2)), "gini": gini, "n": n, "top_cause_share": float(shares.max())}


def kind_by_cause(findings: Sequence[LossFinding]) -> dict[str, dict[str, int]]:
    """Cross-tabulation of loss kind against named cause: e.g. whether missed losers are mostly SELECTION and held losses mostly
    EXIT, which is where the two different fixes (see more / act better) belong."""
    out: dict[str, dict[str, int]] = {k.value: {} for k in LossKind}
    for f in findings:
        row = out[f.kind.value]
        row[f.cause.value] = row.get(f.cause.value, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in out.items() if v}
