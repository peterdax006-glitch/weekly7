"""Research priority engine (contract C62 section 51; checklist G02, G03, G04, G06, C17; Bible canon C58-C62).

'What should I investigate next?' answered from experience, in four moving parts:

1. SIGNALS: surprises, contradictions, failures, missed winners, regime transitions, weakening patterns, unknown areas,
   interaction hints, representation gaps and data-quality issues, each dated, sized and free of identities.
2. QUESTIONS and COMPETING HYPOTHESES generated from the signals (G02, G03). A failure question carries a hypothesis set
   drawn from FailureCause, priors set by the evidence profile, and ALWAYS a coincidence hypothesis and an explicit UNKNOWN
   residual (section 9: 'unknown' is an allowed answer). EXPECTED OUTCOMES (G04) come from an outcome table per cause for a
   standard split experiment, so each hypothesis predicts a distribution over outcomes and the information gain is a real
   mutual information, not a guess.
3. An EVOLVING PRIORITY QUEUE (G06): candidates are re-scored with engine.learning.research_policy.PriorityFunction on every
   step, age so nothing starves, decay when their evidence goes stale, close when their signal is explained, and spawn
   follow-ups from results.
4. The C17 UPDATE: an executed experiment's realised information gain calibrates the policy and re-orders the queue.

No ticker or date appears in a question: signals reference knowledge ids and situation keys (firewall, section 29).
IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import enum
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .core import FailureCause, FirewallBreach, Subsystem, ValidationLabel, stable_hash
from .experiment_memory import (DesignSpec, ExpectedOutcome, ExperimentLedger, ExperimentRecord, Hypothesis, ResultKind,
                                normalise, question_key, to_ts)
from .research_policy import (Candidate, ComputeBudget, ComputeCost, Factors, InfoModel, MetaAdvice, PolicyContext,
                              ResearchPlan, ResearchPolicy, ResearchTarget, ScoredCandidate, entropy_bits, gain_from_record)

LABEL = ValidationLabel.NOT_VALIDATED.value
_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-4]\d)\b")


# ------------------------------------------------------------------------------------------------ signals

class SignalKind(str, enum.Enum):
    SURPRISE = "SURPRISE"
    CONTRADICTION = "CONTRADICTION"
    FAILURE = "FAILURE"
    MISSED_WINNER = "MISSED_WINNER"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    WEAK_PATTERN = "WEAK_PATTERN"
    UNKNOWN_AREA = "UNKNOWN_AREA"
    INTERACTION_HINT = "INTERACTION_HINT"
    REPRESENTATION_GAP = "REPRESENTATION_GAP"
    DATA_QUALITY = "DATA_QUALITY"
    RELIABLE_DRIFT = "RELIABLE_DRIFT"

    def __str__(self):
        return self.value


SIGNAL_TARGET = {
    SignalKind.SURPRISE: ResearchTarget.FAILURE, SignalKind.CONTRADICTION: ResearchTarget.CONTRADICTION,
    SignalKind.FAILURE: ResearchTarget.FAILURE, SignalKind.MISSED_WINNER: ResearchTarget.MISSED_WINNER,
    SignalKind.REGIME_TRANSITION: ResearchTarget.REGIME_TRANSITION, SignalKind.WEAK_PATTERN: ResearchTarget.WEAK_PATTERN,
    SignalKind.UNKNOWN_AREA: ResearchTarget.UNKNOWN_AREA, SignalKind.INTERACTION_HINT: ResearchTarget.INTERACTION,
    SignalKind.REPRESENTATION_GAP: ResearchTarget.NEW_REPRESENTATION, SignalKind.DATA_QUALITY: ResearchTarget.DATA_QUALITY,
    SignalKind.RELIABLE_DRIFT: ResearchTarget.KNOWN_RELIABLE}


def identity_leak(text: str) -> str:
    """Non-empty reason when text carries a calendar date or a bare year: questions must be about situations, not about
    'what happened in 2008' (section 29). Returns '' when clean."""
    if _DATE_RE.search(text):
        return "contains a calendar date"
    if _YEAR_RE.search(text):
        return "contains a year"
    return ""


@dataclass(frozen=True)
class Signal:
    sid: str
    kind: SignalKind
    when: str
    subject: str                                   # knowledge id / situation key / component name; never a ticker or date
    magnitude: float                               # >= 0. z-score for SURPRISE, |loss| share for FAILURE, strength for CONTRADICTION
    subsystem: str = ""
    cause: str = ""                                # FailureCause value if a cause is already claimed
    explained: bool = False                        # a resolved signal stops generating work
    counterpart: str = ""                          # second subject (CONTRADICTION, INTERACTION_HINT)
    contexts: Mapping = field(default_factory=dict)
    profile: Mapping = field(default_factory=dict) # evidence profile keys: see FAILURE_EVIDENCE
    stake: float = 0.5                             # 0..1 how much decision value rides on this
    n_obs: int = 0
    evidence: tuple = ()

    def check(self) -> list:
        errs = []
        if not self.sid or not self.subject:
            errs.append("signal needs sid and subject")
        if self.magnitude < 0 or math.isnan(self.magnitude):
            errs.append(f"{self.sid}: magnitude {self.magnitude!r} must be >= 0")
        if not (0.0 <= self.stake <= 1.0):
            errs.append(f"{self.sid}: stake outside [0,1]")
        leak = identity_leak(self.subject + " " + self.counterpart)
        if leak:
            errs.append(f"{self.sid}: subject {leak} (identity firewall)")
        if self.subsystem:
            try:
                Subsystem.parse(self.subsystem)
            except ValueError:
                errs.append(f"{self.sid}: unknown subsystem {self.subsystem!r}")
        if self.cause:
            try:
                FailureCause.parse(self.cause)
            except ValueError:
                errs.append(f"{self.sid}: unknown cause {self.cause!r}")
        return errs

    @property
    def target(self) -> ResearchTarget:
        return SIGNAL_TARGET[self.kind]


def make_signal(kind, when, subject, magnitude, **kw) -> Signal:
    """Signal with a content-derived id, so the same observation reported twice is one signal."""
    kind = SignalKind(kind)
    sid = stable_hash([kind.value, str(when)[:10], subject, kw.get("counterpart", ""), round(float(magnitude), 3)], 12)
    return Signal(sid=sid, kind=kind, when=str(when), subject=subject, magnitude=float(magnitude), **kw)


def signals_from_ledger(ledger: ExperimentLedger, now, surprise_bits: float = 2.0) -> list:
    """Signals the experiment memory itself implies: contradictory answers, highly surprising results, repeated 'not
    learned' items, and crashed experiments. This closes the loop: what experiments could not settle becomes the next question."""
    out = []
    view = ledger.view(now)
    for c in ledger.contradictory_answers(now):
        ra, rb = view[c["a"]], view[c["b"]]
        when = max(ra.result.observed_at, rb.result.observed_at)
        out.append(make_signal(SignalKind.CONTRADICTION, when, ra.experiment_id, 0.7, counterpart=rb.experiment_id,
                               stake=0.6, evidence=(ra.experiment_id, rb.experiment_id), n_obs=2))
    for r in view.values():
        if r.belief_update is not None and r.belief_update.surprise_bits >= surprise_bits and r.result is not None:
            z = min(6.0, r.belief_update.surprise_bits)
            out.append(make_signal(SignalKind.SURPRISE, r.result.observed_at, r.experiment_id, z,
                                   subsystem=r.experiment.subsystem, stake=0.5, evidence=(r.experiment_id,)))
    for key, e in ledger.not_learned_register(now).items():
        if e["count"] >= 2 and e["text"] != "NOT RECORDED":
            latest = max(view[i].recorded_at for i in e["experiments"])
            out.append(make_signal(SignalKind.UNKNOWN_AREA, latest, stable_hash(key, 8), min(1.0, 0.3 * e["count"]),
                                   contexts={"gap": e["text"]}, stake=0.4, evidence=tuple(e["experiments"])))
    return out


# ------------------------------------------------------------------------------------------------ hypotheses

# Prior over failure causes before looking at evidence. Deliberately flat-ish and conservative; the profile moves it.
CAUSE_BASE = {
    FailureCause.FALSE_PATTERN: 0.16, FailureCause.TEMPORARY_INACTIVITY: 0.08, FailureCause.WRONG_CONTEXT: 0.12,
    FailureCause.REGIME_CHANGE: 0.10, FailureCause.WEAKENING_EFFECT: 0.09, FailureCause.REVERSAL: 0.04,
    FailureCause.MEASUREMENT_ERROR: 0.05, FailureCause.REDUNDANCY: 0.04, FailureCause.SELECTION_ERROR: 0.06,
    FailureCause.TIMING_ERROR: 0.05, FailureCause.RISK_ERROR: 0.04, FailureCause.INTERACTION_FAILURE: 0.05,
    FailureCause.INSUFFICIENT_EVIDENCE: 0.07, FailureCause.UNKNOWN: 0.05}

# profile key -> {cause: multiplier}. Evidence, not opinion: every key is something a diagnostic can measure.
FAILURE_EVIDENCE = {
    "concentrated_in_context": {FailureCause.WRONG_CONTEXT: 3.0, FailureCause.INTERACTION_FAILURE: 1.5, FailureCause.FALSE_PATTERN: 0.6},
    "era_localised": {FailureCause.REGIME_CHANGE: 3.0, FailureCause.TEMPORARY_INACTIVITY: 1.5},
    "gradual_decline": {FailureCause.WEAKENING_EFFECT: 3.0, FailureCause.TEMPORARY_INACTIVITY: 0.5, FailureCause.REGIME_CHANGE: 0.7},
    "sudden_break": {FailureCause.REGIME_CHANGE: 2.0, FailureCause.MEASUREMENT_ERROR: 2.0, FailureCause.WEAKENING_EFFECT: 0.4},
    "sign_flipped": {FailureCause.REVERSAL: 4.0, FailureCause.REGIME_CHANGE: 1.5},
    "redundant_with_other": {FailureCause.REDUNDANCY: 4.0},
    "few_observations": {FailureCause.INSUFFICIENT_EVIDENCE: 4.0, FailureCause.FALSE_PATTERN: 2.0},
    "data_anomaly": {FailureCause.MEASUREMENT_ERROR: 5.0},
    "idle_then_returns": {FailureCause.TEMPORARY_INACTIVITY: 4.0},
    "worst_in_selection": {FailureCause.SELECTION_ERROR: 3.0},
    "timing_off": {FailureCause.TIMING_ERROR: 3.0},
    "tail_loss": {FailureCause.RISK_ERROR: 3.0},
    "combination_only": {FailureCause.INTERACTION_FAILURE: 3.5},
    "never_replicated": {FailureCause.FALSE_PATTERN: 3.0},
}

CAUSE_TAGS = {
    FailureCause.FALSE_PATTERN: ("coincidence", "multiple_testing"), FailureCause.WRONG_CONTEXT: ("context", "boundary"),
    FailureCause.REGIME_CHANGE: ("regime", "era"), FailureCause.WEAKENING_EFFECT: ("decay",),
    FailureCause.TEMPORARY_INACTIVITY: ("dormant",), FailureCause.REVERSAL: ("sign",),
    FailureCause.MEASUREMENT_ERROR: ("data", "artifact"), FailureCause.REDUNDANCY: ("overlap",),
    FailureCause.SELECTION_ERROR: ("selection",), FailureCause.TIMING_ERROR: ("timing",), FailureCause.RISK_ERROR: ("risk", "sizing"),
    FailureCause.INTERACTION_FAILURE: ("interaction",), FailureCause.INSUFFICIENT_EVIDENCE: ("power",), FailureCause.UNKNOWN: ("unknown",)}

CAUSE_TEXT = {
    FailureCause.FALSE_PATTERN: "it never was real; the earlier success was chance or selection",
    FailureCause.WRONG_CONTEXT: "it is real but only inside a context it was applied outside of",
    FailureCause.REGIME_CHANGE: "the market regime changed and the effect belonged to the old one",
    FailureCause.WEAKENING_EFFECT: "the effect is real and is decaying",
    FailureCause.TEMPORARY_INACTIVITY: "the effect is dormant and will return with its trigger",
    FailureCause.REVERSAL: "the effect has reversed sign",
    FailureCause.MEASUREMENT_ERROR: "the failure is a data or measurement artifact",
    FailureCause.REDUNDANCY: "its information is already carried by another item",
    FailureCause.SELECTION_ERROR: "the signal was fine but the wrong names were chosen with it",
    FailureCause.TIMING_ERROR: "the signal was fine but the entry or exit timing was off",
    FailureCause.RISK_ERROR: "the signal was fine but sizing or stops let one loss dominate",
    FailureCause.INTERACTION_FAILURE: "it works alone but fails in combination with another item",
    FailureCause.INSUFFICIENT_EVIDENCE: "there is too little data to say anything",
    FailureCause.UNKNOWN: "none of the above; the cause is not identifiable with current information"}


def failure_hypotheses(profile: Mapping, top_k: int = 5, unknown_floor: float = 0.05, claimed: str = "") -> tuple:
    """Competing explanations for a failure. Priors = base rate x product of evidence multipliers, normalised. Keeps the
    top_k causes, then FORCES FALSE_PATTERN (coincidence must always be a live alternative) and UNKNOWN (>= unknown_floor
    mass) back in, and renormalises. A claimed cause is included but gets no special prior: a claim is a hypothesis."""
    score = {c: p for c, p in CAUSE_BASE.items()}
    for key, mult in FAILURE_EVIDENCE.items():
        v = profile.get(key)
        strength = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (1.0 if v else 0.0)
        if strength <= 0:
            continue
        for cause, m in mult.items():
            score[cause] *= m ** min(1.0, strength)
    ranked = sorted(score, key=lambda c: -score[c])
    keep = [c for c in ranked if c not in (FailureCause.UNKNOWN,)][:top_k]
    for c in (FailureCause.FALSE_PATTERN,):
        if c not in keep:
            keep[-1] = c
    if claimed:
        cc = FailureCause.parse(claimed)
        if cc not in keep and cc != FailureCause.UNKNOWN:
            keep[-1] = cc
    keep = list(dict.fromkeys(keep)) + [FailureCause.UNKNOWN]
    pri = normalise({c: score[c] for c in keep})
    if pri[FailureCause.UNKNOWN] < unknown_floor:
        pri[FailureCause.UNKNOWN] = unknown_floor
        pri = normalise(pri)
    return tuple(Hypothesis(f"h_{c.value.lower()}", CAUSE_TEXT[c], pri[c], kind="noise" if c == FailureCause.FALSE_PATTERN else "explanation",
                            mechanism_tags=CAUSE_TAGS[c], cause=c.value) for c in keep)


def generic_hypotheses(kind: SignalKind, subject: str) -> tuple:
    """Hypothesis sets for the non-failure signal kinds. Each always contains a chance/noise alternative."""
    table = {
        SignalKind.SURPRISE: [("h_noise", "the deviation is chance", "noise", 0.34), ("h_context", "the situation was outside the contexts the belief covers", "explanation", 0.26),
                              ("h_regime", "the environment shifted", "explanation", 0.18), ("h_measure", "the outcome or the expectation was mismeasured", "measurement", 0.12),
                              ("h_model", "the expectation model is wrong", "explanation", 0.10)],
        SignalKind.CONTRADICTION: [("h_contexts", "both are true, in different contexts", "explanation", 0.34), ("h_one_false", "one of the two is not real", "noise", 0.28),
                                   ("h_interaction", "they interact: each holds only when the other is absent", "explanation", 0.14),
                                   ("h_measure", "they were measured differently", "measurement", 0.14), ("h_time", "one has decayed", "explanation", 0.10)],
        SignalKind.MISSED_WINNER: [("h_uncapturable", "the winner was unpredictable noise", "noise", 0.40), ("h_feature", "a feature we do not have would have found it", "explanation", 0.20),
                                   ("h_filtered", "a rule we apply vetoed it wrongly", "explanation", 0.20), ("h_timing", "we saw it but too late", "explanation", 0.12),
                                   ("h_sizing", "we held it but too small", "explanation", 0.08)],
        SignalKind.REGIME_TRANSITION: [("h_survives", "the item survives the transition", "explanation", 0.30), ("h_breaks", "it breaks in the new regime", "explanation", 0.30),
                                       ("h_recovers", "it breaks and then recovers", "explanation", 0.15), ("h_noise", "no transition effect distinguishable from noise", "noise", 0.25)],
        SignalKind.WEAK_PATTERN: [("h_decay", "it is permanently decaying", "explanation", 0.30), ("h_dormant", "it is temporarily inactive", "explanation", 0.25),
                                  ("h_context", "it narrowed to a sub-context", "explanation", 0.20), ("h_noise", "it was never distinguishable from noise", "noise", 0.25)],
        SignalKind.UNKNOWN_AREA: [("h_signal", "there is exploitable structure here", "explanation", 0.25), ("h_none", "there is nothing here beyond noise", "noise", 0.60),
                                  ("h_data", "the data cannot show it", "measurement", 0.15)],
        SignalKind.INTERACTION_HINT: [("h_interact", "the two items interact", "explanation", 0.30), ("h_additive", "their effects are simply additive", "explanation", 0.30),
                                      ("h_redundant", "one is redundant given the other", "explanation", 0.20), ("h_noise", "the apparent interaction is chance", "noise", 0.20)],
        SignalKind.REPRESENTATION_GAP: [("h_better", "a different representation separates the unexplained cases", "explanation", 0.30),
                                        ("h_noise", "the unexplained cases are irreducible noise", "noise", 0.50), ("h_data", "the missing information is not in the data", "measurement", 0.20)],
        SignalKind.DATA_QUALITY: [("h_bad", "the data is wrong", "measurement", 0.35), ("h_fine", "the data is fine and the anomaly is real", "explanation", 0.45),
                                  ("h_stale", "the data is stale or late", "measurement", 0.20)],
        SignalKind.RELIABLE_DRIFT: [("h_stable", "the item remains reliable", "explanation", 0.50), ("h_drifting", "reliability is drifting down", "explanation", 0.30),
                                    ("h_noise", "the drift is noise", "noise", 0.20)]}
    rows = table[kind]
    pri = normalise({h[0]: h[3] for h in rows})
    return tuple(Hypothesis(h[0], f"{h[1]} ({subject})", pri[h[0]], kind=h[2], mechanism_tags=(kind.value.lower(),)) for h in rows)


# ------------------------------------------------------------------------------------------------ expected outcomes

SPLIT_OUTCOMES = ("context_split_explains", "era_split_explains", "sign_flips_out_of_sample", "vanishes_everywhere",
                  "data_artifact_found", "inconclusive")

# P(outcome of the standard split experiment | cause). Each row sums to 1; rows differ so the experiment can discriminate.
SPLIT_TABLE = {
    FailureCause.FALSE_PATTERN: (0.06, 0.06, 0.07, 0.68, 0.03, 0.10), FailureCause.WRONG_CONTEXT: (0.68, 0.06, 0.04, 0.06, 0.02, 0.14),
    FailureCause.REGIME_CHANGE: (0.08, 0.66, 0.06, 0.06, 0.02, 0.12), FailureCause.WEAKENING_EFFECT: (0.05, 0.22, 0.05, 0.52, 0.02, 0.14),
    FailureCause.TEMPORARY_INACTIVITY: (0.10, 0.40, 0.03, 0.10, 0.02, 0.35), FailureCause.REVERSAL: (0.05, 0.15, 0.62, 0.08, 0.02, 0.08),
    FailureCause.MEASUREMENT_ERROR: (0.03, 0.05, 0.03, 0.09, 0.72, 0.08), FailureCause.REDUNDANCY: (0.14, 0.04, 0.03, 0.60, 0.02, 0.17),
    FailureCause.SELECTION_ERROR: (0.35, 0.08, 0.04, 0.15, 0.03, 0.35), FailureCause.TIMING_ERROR: (0.20, 0.15, 0.05, 0.10, 0.02, 0.48),
    FailureCause.RISK_ERROR: (0.12, 0.12, 0.05, 0.16, 0.03, 0.52), FailureCause.INTERACTION_FAILURE: (0.46, 0.06, 0.06, 0.20, 0.02, 0.20),
    FailureCause.INSUFFICIENT_EVIDENCE: (0.10, 0.10, 0.05, 0.20, 0.05, 0.50), FailureCause.UNKNOWN: (0.16, 0.16, 0.10, 0.16, 0.08, 0.34)}


def split_expected(hyps: Sequence[Hypothesis]) -> tuple:
    """Expected outcomes of the standard split experiment for a failure hypothesis set."""
    out = []
    for h in hyps:
        row = SPLIT_TABLE[FailureCause.parse(h.cause)]
        out += [ExpectedOutcome(h.hid, o, p) for o, p in zip(SPLIT_OUTCOMES, row)]
    return tuple(out)


GENERIC_OUTCOMES = ("supports_leading", "supports_noise", "supports_context_dependence", "supports_measurement", "inconclusive")


def generic_expected(hyps: Sequence[Hypothesis], strength: float = 0.6) -> tuple:
    """Expected outcomes for non-failure questions: each hypothesis predicts the outcome that matches its kind with
    probability `strength` and spreads the rest. A discriminating structure, but a coarse one, and it says so in the
    outcome names."""
    fav = {"noise": "supports_noise", "measurement": "supports_measurement"}
    out = []
    for i, h in enumerate(hyps):
        f = fav.get(h.kind) or ("supports_leading" if i == 0 else "supports_context_dependence")
        rest = [o for o in GENERIC_OUTCOMES if o != f]
        for o in GENERIC_OUTCOMES:
            out.append(ExpectedOutcome(h.hid, o, strength if o == f else (1 - strength) / len(rest)))
    return tuple(out)


def expected_for(kind: SignalKind, hyps: Sequence[Hypothesis]) -> tuple:
    return split_expected(hyps) if kind in (SignalKind.FAILURE, SignalKind.SURPRISE) and all(h.cause for h in hyps) else generic_expected(hyps)


# ------------------------------------------------------------------------------------------------ questions

@dataclass(frozen=True)
class ResearchQuestion:
    qid: str
    text: str
    target: ResearchTarget
    subsystem: str
    signals: tuple                                 # sids
    magnitude: float
    stake: float
    kind: SignalKind
    subjects: tuple
    profile: Mapping = field(default_factory=dict)
    contexts: Mapping = field(default_factory=dict)
    n_obs: int = 0
    first_seen: str = ""
    last_seen: str = ""

    def check(self) -> list:
        errs = []
        leak = identity_leak(self.text)
        if leak:
            errs.append(f"{self.qid}: question text {leak}")
        if not self.signals:
            errs.append(f"{self.qid}: question without a triggering signal")
        return errs


def question_text(s: Signal) -> str:
    ctx = ", ".join(f"{k}={v}" for k, v in sorted(s.contexts.items()) if k != "gap") or "its usual contexts"
    sub = s.subsystem.lower() or "system"
    if s.kind == SignalKind.SURPRISE:
        return f"Why did the outcome for {s.subject} deviate {s.magnitude:.1f} standard deviations from what was expected in {ctx}?"
    if s.kind == SignalKind.CONTRADICTION:
        return f"Which conditions separate {s.subject} from {s.counterpart}, which disagree with strength {s.magnitude:.2f}?"
    if s.kind == SignalKind.FAILURE:
        cause = f" (claimed cause {s.cause.lower()})" if s.cause else " (no cause established)"
        return f"What caused the {sub} failure of {s.subject}{cause}, and does it generalise?"
    if s.kind == SignalKind.MISSED_WINNER:
        return f"What distinguished the missed winners of situation {s.subject}, and was capturing them possible without look-ahead?"
    if s.kind == SignalKind.REGIME_TRANSITION:
        return f"Does {s.subject} survive the regime transition described by {ctx}?"
    if s.kind == SignalKind.WEAK_PATTERN:
        return f"Is the weakening of {s.subject} permanent decay, temporary inactivity or a narrowing to a sub-context?"
    if s.kind == SignalKind.UNKNOWN_AREA:
        gap = s.contexts.get("gap")
        return f"Can the unresolved gap '{gap}' be closed with available data?" if gap else f"What is known about {ctx}, and is there structure there?"
    if s.kind == SignalKind.INTERACTION_HINT:
        return f"Do {s.subject} and {s.counterpart} interact, or are their effects additive or redundant?"
    if s.kind == SignalKind.REPRESENTATION_GAP:
        return f"Is there a representation that separates the cases {s.subject} leaves unexplained?"
    if s.kind == SignalKind.DATA_QUALITY:
        return f"Is the {s.subject} data reliable, or is the anomaly an artifact?"
    return f"Is {s.subject} still as reliable as its record says?"


def saturating_sum(values: Iterable[float]) -> float:
    """1 - prod(1 - v_i) with v clipped to [0,1): independent evidence adds, but never past 1."""
    p = 1.0
    for v in values:
        p *= 1.0 - min(max(v, 0.0), 0.999)
    return 1.0 - p


def normalise_magnitude(kind: SignalKind, m: float) -> float:
    """Map a kind-specific magnitude onto [0, 1)."""
    if kind == SignalKind.SURPRISE:
        return 1.0 - math.exp(-m / 3.0)
    if kind == SignalKind.UNKNOWN_AREA or kind == SignalKind.CONTRADICTION:
        return min(m, 0.999)
    return min(m, 0.999) if m <= 1.0 else 1.0 - math.exp(-m)


class QuestionGenerator:
    """Signals -> questions. Same question key = one question: signals about the same thing merge, magnitudes combine by
    saturating sum, contexts/profile union. Explained signals generate nothing. Every signal must be strictly before `now`."""

    def generate(self, signals: Sequence[Signal], now, answered_keys: frozenset = frozenset()) -> list:
        cut = to_ts(now)
        groups: dict = {}
        for s in signals:
            errs = s.check()
            if errs:
                raise ValueError("; ".join(errs))
            if to_ts(s.when) >= cut:
                raise FirewallBreach(f"signal {s.sid} dated {s.when} is not before now={now}")
            if s.explained:
                continue
            text = question_text(s)
            key = question_key(text)
            if key in answered_keys:
                continue
            groups.setdefault(key, []).append((s, text))
        out = []
        for key, rows in groups.items():
            rows.sort(key=lambda r: (r[0].when, r[0].sid))
            first = rows[0][0]
            mag = saturating_sum(normalise_magnitude(s.kind, s.magnitude) for s, _ in rows)
            prof: dict = {}
            ctxs: dict = {}
            for s, _ in rows:
                for k, v in s.profile.items():
                    prof[k] = max(float(prof.get(k, 0.0)), float(v))
                for k, v in s.contexts.items():
                    ctxs.setdefault(k, v)
            out.append(ResearchQuestion(
                qid=key, text=rows[0][1], target=first.target, subsystem=first.subsystem, signals=tuple(s.sid for s, _ in rows),
                magnitude=mag, stake=max(s.stake for s, _ in rows), kind=first.kind, subjects=tuple(sorted({s.subject for s, _ in rows})),
                profile=prof, contexts=ctxs, n_obs=sum(s.n_obs for s, _ in rows), first_seen=rows[0][0].when, last_seen=rows[-1][0].when))
        return sorted(out, key=lambda q: (-q.magnitude, q.qid))


# ------------------------------------------------------------------------------------------------ candidates

@dataclass(frozen=True)
class ExperimentTemplate:
    name: str
    cost: ComputeCost
    data_needs: tuple
    families: str


TEMPLATES = {
    SignalKind.FAILURE: ExperimentTemplate("split_by_context_and_era", ComputeCost(15, 1.5, False, 10), ("pattern_ledger",), "diagnosis:failure"),
    SignalKind.SURPRISE: ExperimentTemplate("split_by_context_and_era", ComputeCost(15, 1.5, False, 10), ("pattern_ledger",), "diagnosis:surprise"),
    SignalKind.CONTRADICTION: ExperimentTemplate("conditional_replication", ComputeCost(25, 2.0, False, 15), ("pattern_ledger",), "diagnosis:contradiction"),
    SignalKind.MISSED_WINNER: ExperimentTemplate("missed_winner_capture", ComputeCost(40, 3.0, True, 25), ("price_panel",), "diagnosis:missed_winner"),
    SignalKind.REGIME_TRANSITION: ExperimentTemplate("regime_conditional_eval", ComputeCost(30, 2.5, True, 20), ("price_panel", "market_context"), "diagnosis:regime"),
    SignalKind.WEAK_PATTERN: ExperimentTemplate("decay_vs_dormancy_test", ComputeCost(20, 1.5, False, 12), ("pattern_ledger",), "diagnosis:decay"),
    SignalKind.UNKNOWN_AREA: ExperimentTemplate("coverage_probe", ComputeCost(60, 3.5, True, 40), ("price_panel",), "exploration:unknown"),
    SignalKind.INTERACTION_HINT: ExperimentTemplate("pairwise_interaction_scan", ComputeCost(45, 3.0, True, 30), ("price_panel",), "exploration:interaction"),
    SignalKind.REPRESENTATION_GAP: ExperimentTemplate("residual_representation_search", ComputeCost(90, 3.5, True, 60), ("price_panel",), "exploration:representation"),
    SignalKind.DATA_QUALITY: ExperimentTemplate("data_audit", ComputeCost(10, 1.0, False, 8), ("raw_cache_manifest",), "audit:data"),
    SignalKind.RELIABLE_DRIFT: ExperimentTemplate("reliability_recheck", ComputeCost(20, 1.5, False, 12), ("pattern_ledger",), "tuning:reliable")}

SUBSYSTEM_WEIGHT = {"SELECTION": 0.9, "RISK": 0.85, "DIRECTION": 0.7, "TIMING": 0.7, "EXIT": 0.65, "": 0.6}


def hypothesis_entropy_share(hyps: Sequence[Hypothesis]) -> float:
    """Normalised entropy of the prior in [0,1]: 1 = maximally undecided, 0 = already certain. This is the 'uncertainty' factor."""
    if len(hyps) < 2:
        return 0.0
    return entropy_bits(h.prior for h in hyps) / math.log2(len(hyps))


def transfer_potential(ctxs: Mapping, subjects: Sequence[str], n_signals: int) -> float:
    """More independent contexts / subjects / signals -> a finding is more likely to be general. Saturating in [0,1]."""
    breadth = len(ctxs) + len(subjects) + min(n_signals, 6) / 2.0
    return 1.0 - math.exp(-breadth / 4.0)


def feasibility_of(template: ExperimentTemplate, available: frozenset | None) -> float:
    if available is None:
        return 1.0
    missing = [d for d in template.data_needs if d not in available]
    return 0.0 if missing else 1.0


@dataclass(frozen=True)
class BuiltQuestion:
    question: ResearchQuestion
    hypotheses: tuple
    expected: tuple
    candidate: Candidate
    template: str


class CandidateBuilder:
    """Question -> hypotheses -> expected outcomes -> Candidate with an InfoModel that computes real mutual information."""

    def __init__(self, available_data: frozenset | None = None, config_base: Mapping | None = None):
        self.available = available_data
        self.config_base = dict(config_base or {})

    def build(self, q: ResearchQuestion, now, claimed_cause: str = "") -> BuiltQuestion:
        if q.kind in (SignalKind.FAILURE, SignalKind.SURPRISE):
            hyps = failure_hypotheses(q.profile, claimed=claimed_cause)
        else:
            hyps = generic_hypotheses(q.kind, q.subjects[0] if q.subjects else "?")
        expected = expected_for(q.kind, hyps)
        tpl = TEMPLATES[q.kind]
        info = InfoModel("discrete", {"hypotheses": [(h.hid, h.prior) for h in hyps],
                                      "expected": [(o.hid, o.outcome, o.probability) for o in expected]})
        unc = hypothesis_entropy_share(hyps)
        rel = min(1.0, 0.5 * SUBSYSTEM_WEIGHT.get(q.subsystem, 0.6) + 0.5 * q.magnitude)
        fac = Factors(uncertainty=unc, relevance=rel, transfer_potential=transfer_potential(q.contexts, q.subjects, len(q.signals)),
                      feasibility=feasibility_of(tpl, self.available), decision_value=q.stake)
        cid = stable_hash([q.qid, tpl.name], 12)
        cfg = {**self.config_base, "template": tpl.name, "target": q.target.value, "kind": q.kind.value,
               "subjects": list(q.subjects), "contexts": dict(q.contexts)}
        cand = Candidate(cid=cid, question=q.text, target=q.target, created_at=q.last_seen or str(now), factors=fac, info=info, cost=tpl.cost,
                         subsystem=q.subsystem, family=tpl.families, config=cfg, data_needs=tpl.data_needs,
                         overfit_hint=0.5 if q.kind in (SignalKind.UNKNOWN_AREA, SignalKind.REPRESENTATION_GAP, SignalKind.INTERACTION_HINT) else 0.1,
                         free_parameters=2 if q.kind in (SignalKind.REPRESENTATION_GAP, SignalKind.INTERACTION_HINT) else 0,
                         evidence=q.signals, magnitude=q.magnitude)
        return BuiltQuestion(q, hyps, expected, cand, tpl.name)

    def to_experiment_record(self, bq: BuiltQuestion, now, seed: int, belief: str = ""):
        """A ready-to-propose ExperimentRecord for a chosen candidate (imports lazily to keep this module's surface small)."""
        from .experiment_memory import Prediction, new_record
        lead = max(bq.hypotheses, key=lambda h: h.prior)
        return new_record(
            experiment_id=f"exp-{bq.candidate.cid}", question=bq.question.text,
            current_belief=belief or f"leading hypothesis ({lead.prior:.0%}): {lead.statement}", hypotheses=bq.hypotheses,
            prediction=Prediction(f"the data will favour: {lead.statement}", direction="none", confidence=min(0.9, max(0.1, lead.prior))),
            design=DesignSpec(config=dict(bq.candidate.config), seed=seed, controls=("shuffled_labels", "noise_pair"),
                              cost_minutes=bq.candidate.cost.cpu_minutes, subsystem=bq.question.subsystem, target=bq.question.target.value),
            expected=bq.expected, now=now, tags=(bq.template,))


# ------------------------------------------------------------------------------------------------ evolving queue

class ItemStatus(str, enum.Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    OBSOLETE = "OBSOLETE"
    BLOCKED = "BLOCKED"


@dataclass
class QueueItem:
    candidate: Candidate
    enqueued_at: str
    last_evidence_at: str
    status: ItemStatus = ItemStatus.OPEN
    score: ScoredCandidate | None = None
    adjusted: float = 0.0
    attempts: int = 0
    parent: str = ""
    note: str = ""


@dataclass(frozen=True)
class QueueConfig:
    age_rate_per_day: float = 0.02                 # priority bonus per day waiting (anti-starvation)
    age_cap_days: float = 30.0
    stale_half_life_days: float = 45.0             # priority halves every this many days without fresh evidence
    followup_uncertainty_decay: float = 0.8
    max_open: int = 500


class ResearchQueue:
    """The evolving queue. It never deletes: finished and obsolete items keep their history (section 49 spirit)."""

    def __init__(self, cfg: QueueConfig | None = None, log_path=None):
        self.cfg = cfg or QueueConfig()
        self.items: dict = {}
        self.log_path = Path(log_path) if log_path else None
        self.events = 0

    def _log(self, event: str, cid: str, now, **kw) -> None:
        self.events += 1
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": event, "cid": cid, "at": str(now), **kw}, default=str) + "\n")

    def enqueue(self, cand: Candidate, now, parent: str = "") -> QueueItem:
        """Add a candidate, or MERGE into the existing one with the same id (evidence accumulates, magnitude combines)."""
        if to_ts(cand.created_at) >= to_ts(now):
            raise FirewallBreach(f"candidate {cand.cid} created_at={cand.created_at} is not before now={now}")
        cur = self.items.get(cand.cid)
        if cur is not None:
            if cur.status in (ItemStatus.DONE, ItemStatus.OBSOLETE):
                cur.status = ItemStatus.OPEN                      # new evidence reopens a finished question
                cur.note = "reopened by new evidence"
            merged_ev = tuple(dict.fromkeys(cur.candidate.evidence + cand.evidence))
            mag = saturating_sum([cur.candidate.magnitude, cand.magnitude])
            cur.candidate = replace(cand, evidence=merged_ev, magnitude=mag, prior_tests_on_window=cur.candidate.prior_tests_on_window)
            cur.last_evidence_at = str(now)
            self._log("merge", cand.cid, now, evidence=len(merged_ev))
            return cur
        open_n = sum(1 for i in self.items.values() if i.status == ItemStatus.OPEN)
        if open_n >= self.cfg.max_open:
            weakest = min((i for i in self.items.values() if i.status == ItemStatus.OPEN), key=lambda i: i.adjusted)
            if weakest.adjusted >= (cand.magnitude or 0.0):
                raise ValueError("queue full and the new candidate is weaker than everything queued")
            weakest.status = ItemStatus.OBSOLETE
            weakest.note = "evicted by a stronger candidate"
        it = QueueItem(cand, str(now), str(now), parent=parent)
        self.items[cand.cid] = it
        self._log("enqueue", cand.cid, now, target=cand.target.value, parent=parent)
        return it

    def rescore(self, policy: ResearchPolicy, ctx: PolicyContext, budget: ComputeBudget | None = None) -> list:
        """Re-evaluate every OPEN item with the current policy. Adds the aging bonus, applies staleness decay, and marks items
        BLOCKED (with the reason) rather than hiding them."""
        now = to_ts(ctx.now)
        ctx = replace(ctx, history=tuple(policy.history) if not ctx.history else ctx.history)
        out = []
        for it in self.items.values():
            if it.status not in (ItemStatus.OPEN, ItemStatus.BLOCKED):
                continue
            age = max(0.0, (now - to_ts(it.enqueued_at)).total_seconds() / 86400.0)
            stale = max(0.0, (now - to_ts(it.last_evidence_at)).total_seconds() / 86400.0)
            it.score = policy.priority.score(it.candidate, ctx, budget)
            aging = 1.0 + self.cfg.age_rate_per_day * min(age, self.cfg.age_cap_days)
            decay = 0.5 ** (stale / self.cfg.stale_half_life_days)
            it.adjusted = it.score.priority * aging * decay
            it.status = ItemStatus.BLOCKED if it.score.blocked else ItemStatus.OPEN
            it.note = it.score.blocked or it.note
            out.append(it)
        return sorted(out, key=lambda i: -i.adjusted)

    def top(self, n: int = 5) -> list:
        return sorted((i for i in self.items.values() if i.status == ItemStatus.OPEN), key=lambda i: -i.adjusted)[:n]

    def start(self, cid: str, now) -> QueueItem:
        it = self.items[cid]
        if it.status != ItemStatus.OPEN:
            raise ValueError(f"{cid} is {it.status.value}, cannot start")
        it.status = ItemStatus.IN_PROGRESS
        it.attempts += 1
        self._log("start", cid, now)
        return it

    def complete(self, cid: str, now, follow_ups: Sequence[Candidate] = ()) -> list:
        it = self.items[cid]
        it.status = ItemStatus.DONE
        self._log("complete", cid, now, follow_ups=[c.cid for c in follow_ups])
        made = []
        for c in follow_ups:
            f = replace(c, factors=replace(c.factors, uncertainty=c.factors.uncertainty * self.cfg.followup_uncertainty_decay))
            made.append(self.enqueue(f, now, parent=cid))
        return made

    def close_explained(self, explained_sids: Iterable[str], now) -> list:
        """Items whose every triggering signal is now explained are OBSOLETE: the queue evolves as the world resolves."""
        done = set(explained_sids)
        closed = []
        for it in self.items.values():
            if it.status in (ItemStatus.OPEN, ItemStatus.BLOCKED) and it.candidate.evidence and set(it.candidate.evidence) <= done:
                it.status = ItemStatus.OBSOLETE
                it.note = "all triggering signals explained"
                self._log("obsolete", it.candidate.cid, now)
                closed.append(it.candidate.cid)
        return closed

    def starvation_report(self, now, days: float = 30.0) -> list:
        t = to_ts(now)
        return [{"cid": i.candidate.cid, "target": i.candidate.target.value, "waiting_days": (t - to_ts(i.enqueued_at)).total_seconds() / 86400.0,
                 "adjusted": i.adjusted} for i in self.items.values()
                if i.status == ItemStatus.OPEN and (t - to_ts(i.enqueued_at)).total_seconds() / 86400.0 > days]

    def summary(self) -> dict:
        by_s: dict = {}
        by_t: dict = {}
        for i in self.items.values():
            by_s[i.status.value] = by_s.get(i.status.value, 0) + 1
            if i.status == ItemStatus.OPEN:
                by_t[i.candidate.target.value] = by_t.get(i.candidate.target.value, 0) + 1
        return {"by_status": by_s, "open_by_target": by_t, "events": self.events}


# ------------------------------------------------------------------------------------------------ follow-ups from results

FOLLOW_UP_RULES = {
    "context_split_explains": ("context_boundary_map", ResearchTarget.WEAK_PATTERN, "Where exactly is the boundary of the context that explains {s}?"),
    "era_split_explains": ("regime_conditional_eval", ResearchTarget.REGIME_TRANSITION, "Which regime variable separates the eras for {s}, and does it predict the next transition?"),
    "sign_flips_out_of_sample": ("reversal_persistence", ResearchTarget.FAILURE, "Is the sign flip of {s} persistent or a one-window reversal?"),
    "data_artifact_found": ("data_audit", ResearchTarget.DATA_QUALITY, "Which other results relied on the artifact found around {s}?"),
    "inconclusive": ("power_extension", ResearchTarget.UNKNOWN_AREA, "What sample would make the {s} question decidable?")}


def follow_ups_from(record: ExperimentRecord, now) -> list:
    """Follow-up Candidates implied by a finished experiment's observed outcome (NOT by hope): only outcomes listed in
    FOLLOW_UP_RULES spawn work; 'vanishes_everywhere' correctly spawns nothing (the pattern is dead)."""
    if record.result is None:
        return []
    rule = FOLLOW_UP_RULES.get(record.result.outcome)
    if rule is None:
        return []
    name, target, text = rule
    subject = str(record.experiment.config.get("subjects", ["item"])[0]) if isinstance(record.experiment.config.get("subjects"), list) and record.experiment.config.get("subjects") else record.experiment_id
    prior = record.belief_update.posterior if record.belief_update else {}
    unc = entropy_bits(prior.values()) / math.log2(max(2, len(prior))) if prior else 0.5
    return [Candidate(cid=stable_hash([record.experiment_id, name], 12), question=text.format(s=subject), target=target,
                      created_at=record.result.observed_at,
                      factors=Factors(uncertainty=min(1.0, max(unc, 0.2)), relevance=0.6, transfer_potential=0.5, feasibility=1.0, decision_value=0.5),
                      info=InfoModel("proxy", {"resolvability": 0.6}), cost=TEMPLATES[SignalKind.FAILURE].cost,
                      subsystem=record.experiment.subsystem, family=f"followup:{name}", config={"template": name, "parent": record.experiment_id},
                      evidence=(record.experiment_id,), magnitude=0.4)]


# ------------------------------------------------------------------------------------------------ the engine (C17)

@dataclass(frozen=True)
class EngineStep:
    now: str
    questions: tuple
    built: tuple
    plan: ResearchPlan
    queue_summary: Mapping
    top: tuple
    label: str = LABEL


class ResearchPriorityEngine:
    """Signals in, ranked and budgeted research plan out; results in, queue and policy updated (C17)."""

    def __init__(self, policy: ResearchPolicy | None = None, queue: ResearchQueue | None = None,
                 builder: CandidateBuilder | None = None):
        self.policy = policy or ResearchPolicy()
        self.queue = queue or ResearchQueue()
        self.builder = builder or CandidateBuilder()
        self.generator = QuestionGenerator()
        self._built: dict = {}

    def _duplicate_check(self, ledger: ExperimentLedger, now):
        def check(c: Candidate):
            design = DesignSpec(config=dict(c.config), seed=0)
            return ledger.already_tested(c.question, design, now)
        return check

    def step(self, signals: Sequence[Signal], ledger: ExperimentLedger, budget: ComputeBudget, now, seed: int,
             meta: MetaAdvice | None = None, available_data: frozenset | None = None) -> EngineStep:
        """One planning cycle. Signals from the ledger itself (contradictions, surprises, repeated unknowns) are added, so
        what past experiments could not settle feeds the next question."""
        all_signals = list(signals) + signals_from_ledger(ledger, now)
        answered = frozenset(question_key(r.question) for r in ledger.answered(now)
                             if r.result and r.result.kind in ResultKind.DEFINITIVE)
        self.builder.available = available_data if available_data is not None else self.builder.available
        questions = self.generator.generate(all_signals, now, answered)
        built = []
        for q in questions:
            bq = self.builder.build(q, now)
            self._built[bq.candidate.cid] = bq
            self.queue.enqueue(bq.candidate, now)
            built.append(bq)
        self.queue.close_explained([s.sid for s in all_signals if s.explained], now)
        ctx = PolicyContext(now=str(now), duplicate_check=self._duplicate_check(ledger, now), available_data=available_data,
                            meta=meta or MetaAdvice.empty())
        ranked = self.queue.rescore(self.policy, ctx, budget)
        open_cands = [i.candidate for i in ranked if i.status == ItemStatus.OPEN or i.status == ItemStatus.BLOCKED]
        plan = self.policy.plan(open_cands, ctx, budget, seed) if open_cands else self.policy.plan([], ctx, budget, seed)
        return EngineStep(str(now), tuple(questions), tuple(built), plan, self.queue.summary(), tuple(i.candidate.cid for i in self.queue.top(5)))

    def built(self, cid: str) -> BuiltQuestion | None:
        return self._built.get(cid)

    def update_from_result(self, record: ExperimentRecord, cid: str, predicted_bits: float, now) -> dict:
        """C17: an executed experiment re-orders the queue. Its realised information gain is fed to the policy (calibrating
        EIG and the target bandit), the item is completed, and follow-ups implied by the observed outcome are enqueued."""
        item = self.queue.items.get(cid)
        target = item.candidate.target.value if item else record.experiment.target
        family = item.candidate.family if item else ""
        gain = gain_from_record(record, cid, predicted_bits, target, family)
        self.policy.observe(gain, now)
        ups = follow_ups_from(record, now)
        made = self.queue.complete(cid, now, ups) if item else []
        return {"realised_bits": gain.realised_bits, "useful": gain.useful, "follow_ups": [m.candidate.cid for m in made],
                "calibration_scale": self.policy.calibrator.scale(gain.target)}


def render_queue(engine: ResearchPriorityEngine, n: int = 10) -> str:
    lines = [f"Research queue   [{LABEL}]", f"summary: {engine.queue.summary()}"]
    for i in engine.queue.top(n):
        lines.append(f"  {i.adjusted:8.5f} [{i.candidate.target.value:<18}] {i.candidate.question}")
    blocked = [i for i in engine.queue.items.values() if i.status == ItemStatus.BLOCKED]
    if blocked:
        lines.append(f"blocked ({len(blocked)}):")
        lines += [f"  {i.candidate.cid}: {i.note}" for i in blocked[:n]]
    return "\n".join(lines)


# ==================================================================================================================
# Part 2: adapters from engine outputs, queue persistence and analytics, and a planted diagnosis world
# ==================================================================================================================

def signals_from_failure_rows(rows: Iterable[Mapping], now) -> list:
    """FAILURE signals from postmortem-shaped rows: {knowledge_id, when, loss_share, subsystem, cause?, explained?, profile?, n_obs?}.
    loss_share is the failure's share of the period's total loss (0..1): the size of the signal. Rows dated at/after `now` are
    refused (FirewallBreach) rather than skipped, because a future-dated failure means the caller's clock is wrong."""
    cut = to_ts(now)
    out = []
    for r in rows:
        if to_ts(r["when"]) >= cut:
            raise FirewallBreach(f"failure row {r.get('knowledge_id')} dated {r['when']} is not before now={now}")
        out.append(make_signal(SignalKind.FAILURE, r["when"], str(r["knowledge_id"]), float(r.get("loss_share", 0.1)),
                               subsystem=str(r.get("subsystem", "")), cause=str(r.get("cause", "")), explained=bool(r.get("explained", False)),
                               profile=dict(r.get("profile", {})), stake=float(r.get("stake", min(1.0, 2 * float(r.get("loss_share", 0.1))))),
                               n_obs=int(r.get("n_obs", 0)), contexts=dict(r.get("contexts", {}))))
    return out


def signals_from_missed_winners(rows: Iterable[Mapping], now) -> list:
    """MISSED_WINNER signals from rows {situation_key, when, gain_share, why_missed?}. The situation key is an identity-free
    description (from engine.learning.situation), never a ticker: a key that looks like a date is refused by Signal.check."""
    cut = to_ts(now)
    out = []
    for r in rows:
        if to_ts(r["when"]) >= cut:
            raise FirewallBreach(f"missed-winner row dated {r['when']} is not before now={now}")
        out.append(make_signal(SignalKind.MISSED_WINNER, r["when"], str(r["situation_key"]), float(r.get("gain_share", 0.1)),
                               subsystem=str(r.get("subsystem", "SELECTION")), contexts={"why_missed": r.get("why_missed", "unknown")},
                               stake=float(r.get("stake", 0.5)), n_obs=int(r.get("n_obs", 1))))
    return out


_HEALTH_KIND = {"BROKEN": (SignalKind.FAILURE, 0.9), "CONTRADICTED": (SignalKind.CONTRADICTION, 0.8), "DEGRADING": (SignalKind.WEAK_PATTERN, 0.5),
                "UNSTABLE": (SignalKind.WEAK_PATTERN, 0.4), "RECOVERING": (SignalKind.REGIME_TRANSITION, 0.3),
                "INSUFFICIENT_EVIDENCE": (SignalKind.UNKNOWN_AREA, 0.3), "DORMANT": (SignalKind.WEAK_PATTERN, 0.2)}


def signals_from_health(rows: Iterable[Mapping], now) -> list:
    """Signals from knowledge-health rows {knowledge_id, when, health, reliability?, counterpart?}. HEALTHY and UNKNOWN rows
    make no signal (nothing to investigate); a HEALTHY item whose reliability fell sharply makes a RELIABLE_DRIFT signal."""
    cut = to_ts(now)
    out = []
    for r in rows:
        if to_ts(r["when"]) >= cut:
            raise FirewallBreach(f"health row dated {r['when']} is not before now={now}")
        h = str(r.get("health", "UNKNOWN")).upper()
        if h in _HEALTH_KIND:
            kind, mag = _HEALTH_KIND[h]
            out.append(make_signal(kind, r["when"], str(r["knowledge_id"]), mag, counterpart=str(r.get("counterpart", "")),
                                   subsystem=str(r.get("subsystem", "")), stake=float(r.get("stake", 0.5))))
        elif h == "HEALTHY" and float(r.get("reliability_drop", 0.0)) >= 0.15:
            out.append(make_signal(SignalKind.RELIABLE_DRIFT, r["when"], str(r["knowledge_id"]), float(r["reliability_drop"]),
                                   stake=float(r.get("stake", 0.6))))
    return out


def signals_from_data_audit(channels: Mapping, when, now) -> list:
    """DATA_QUALITY signals from a leak-audit-style mapping {channel: status}. LEAK is a strong signal, QUARANTINED a medium one,
    anything else none. Channel names are component names, not identities."""
    if to_ts(when) >= to_ts(now):
        raise FirewallBreach(f"audit dated {when} is not before now={now}")
    weight = {"LEAK": 0.95, "QUARANTINED": 0.6, "UNVERIFIED": 0.4}
    return [make_signal(SignalKind.DATA_QUALITY, when, f"channel_{name}", weight[st], stake=0.9 if st == "LEAK" else 0.5)
            for name, st in sorted(channels.items()) if st in weight]


# ------------------------------------------------------------------------------------------------ queue persistence & analytics

def queue_snapshot(q: ResearchQueue) -> str:
    """Canonical JSON of the whole queue, including finished and obsolete items (nothing is dropped)."""
    from .research_policy import candidate_to_dict
    rows = [{"candidate": candidate_to_dict(i.candidate), "enqueued_at": i.enqueued_at, "last_evidence_at": i.last_evidence_at,
             "status": i.status.value, "attempts": i.attempts, "parent": i.parent, "note": i.note, "adjusted": i.adjusted}
            for _, i in sorted(q.items.items())]
    return json.dumps({"cfg": dict(q.cfg.__dict__), "items": rows, "events": q.events}, sort_keys=True)


def queue_restore(text: str, log_path=None) -> ResearchQueue:
    """Rebuild a queue from `queue_snapshot` output. Scores are NOT restored (they depend on the current policy and must be
    recomputed by rescore); statuses, attempts, parents and notes are."""
    from .research_policy import candidate_from_dict
    d = json.loads(text)
    q = ResearchQueue(QueueConfig(**d["cfg"]), log_path)
    for r in d["items"]:
        c = candidate_from_dict(r["candidate"])
        q.items[c.cid] = QueueItem(c, r["enqueued_at"], r["last_evidence_at"], ItemStatus(r["status"]), None, r.get("adjusted", 0.0),
                                   r["attempts"], r["parent"], r["note"])
    q.events = d.get("events", 0)
    return q


def replay_log(path) -> dict:
    """Summarise an append-only queue event log: counts per event type and the last event per candidate id. A torn line is counted."""
    counts: dict = {}
    last: dict = {}
    bad = 0
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                bad += 1
                continue
            counts[e["event"]] = counts.get(e["event"], 0) + 1
            last[e["cid"]] = e["event"]
    return {"counts": counts, "last_event": last, "unparseable": bad}


def target_balance(q: ResearchQueue) -> dict:
    """Share of OPEN queue items and of their total adjusted priority per research target, plus the exploit share."""
    open_items = [i for i in q.items.values() if i.status == ItemStatus.OPEN]
    tot = sum(i.adjusted for i in open_items) or 1.0
    by_n: dict = {}
    by_p: dict = {}
    for i in open_items:
        t = i.candidate.target.value
        by_n[t] = by_n.get(t, 0) + 1
        by_p[t] = by_p.get(t, 0.0) + i.adjusted / tot
    return {"n_open": len(open_items), "count_by_target": by_n, "priority_share_by_target": by_p,
            "exploit_priority_share": by_p.get(ResearchTarget.KNOWN_RELIABLE.value, 0.0)}


def why_not_top(engine: ResearchPriorityEngine, cid: str) -> str:
    """Plain reason an item is not at the head of the queue: blocked (and why), already done, or the largest log-term gap to the leader."""
    it = engine.queue.items.get(cid)
    if it is None:
        return f"{cid}: not in the queue"
    if it.status != ItemStatus.OPEN:
        return f"{cid}: status {it.status.value}" + (f" ({it.note})" if it.note else "")
    top = engine.queue.top(1)
    if not top or top[0].candidate.cid == cid:
        return f"{cid}: it is the top item"
    from .research_policy import why_ranked_above
    if it.score is None or top[0].score is None:
        return f"{cid}: not scored yet"
    return why_ranked_above(top[0].score, it.score)


def merge_similar_questions(questions: Sequence[ResearchQuestion], threshold: float = 0.8) -> list:
    """Collapse questions whose wording is nearly identical (question_similarity >= threshold) into one, keeping the larger
    magnitude and the union of signals. Different sentences about the same thing are one question, not two experiments."""
    from .experiment_memory import question_similarity
    merged: list = []
    for q in sorted(questions, key=lambda q: -q.magnitude):
        for i, m in enumerate(merged):
            if m.target == q.target and question_similarity(m.text, q.text) >= threshold:
                merged[i] = replace(m, signals=tuple(dict.fromkeys(m.signals + q.signals)), subjects=tuple(sorted(set(m.subjects) | set(q.subjects))),
                                    magnitude=saturating_sum([m.magnitude, q.magnitude]), stake=max(m.stake, q.stake))
                break
        else:
            merged.append(q)
    return merged


# ------------------------------------------------------------------------------------------------ planted diagnosis world

def simulate_diagnosis(true_cause: FailureCause, seed: int, max_experiments: int = 6, profile: Mapping | None = None) -> dict:
    """Closed-loop planted test of the whole question -> hypotheses -> experiment -> belief-update path. A failure with a
    KNOWN true cause is diagnosed by repeatedly running the standard split experiment, whose outcome is drawn from the
    true cause's row of SPLIT_TABLE. Returns the posterior trajectory: the true cause must gain mass and, given enough
    experiments, become the leading hypothesis. Also reports how often a WRONG leading hypothesis was declared (a diagnosis
    engine that is confidently wrong is worse than none)."""
    rng = np.random.default_rng(seed)
    hyps = failure_hypotheses(profile or {})
    expected = split_expected(hyps)
    ids = {h.cause: h.hid for h in hyps}
    in_set = true_cause.value in ids
    true_hid = ids.get(true_cause.value, ids[FailureCause.UNKNOWN.value])     # outside the set, UNKNOWN is the honest answer
    post = {h.hid: h.prior for h in hyps}
    row = SPLIT_TABLE[true_cause]
    traj = [post[true_hid]]
    from .experiment_memory import outcome_likelihoods, robust_update
    for _ in range(max_experiments):
        outcome = SPLIT_OUTCOMES[int(rng.choice(len(SPLIT_OUTCOMES), p=np.array(row) / sum(row)))]
        cur = tuple(replace(h, prior=post[h.hid]) for h in hyps)
        post, _ = robust_update(post, outcome_likelihoods(cur, expected, outcome), ids[FailureCause.UNKNOWN.value])
        traj.append(post[true_hid])
    lead = max(post, key=post.get)
    return {"true_cause": true_cause.value, "in_hypothesis_set": in_set, "prior": traj[0], "posterior": traj[-1], "trajectory": traj,
            "leading": lead, "diagnosed": lead == true_hid, "confident_wrong": lead != true_hid and post[lead] > 0.6}
