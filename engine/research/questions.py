"""Research question generator (C66 section 40; also 2, 34, 41; canon C66, C63).

Section 40: questions are generated automatically from surprises, contradictions, losses, missed winners, pattern breaks,
regime changes and new discoveries, and EVERY question becomes a research object with hypothesis, expected value, test plan,
success criterion, failure criterion and priority. This module turns typed matured-world events into such objects.

  Event  ->  QuestionObject(ResearchQuestion + hypotheses + test plan + expected value + priority)  ->  ResearchItem / tree

* Wording comes from a per-source template (identity free: refuses tickers, dates, years).
* The hypothesis set comes from the learning side (engine.learning.research_priority failure / generic hypothesis sets), so a
  question and its hypothesis tree never disagree about the candidate explanations. An explicit chance hypothesis and an explicit
  unknown explanation are always present (section 33).
* Expected value is a real section-2 vector: information from the mutual information of the test plan, loss reduction from the
  loss share (section 34), volatility / direction value from the objective the event touches, compute from the plan.
* Success and failure criteria are numeric and written BEFORE the test (a question that cannot fail is refused).
* Duplicates merge by question key; answered questions are not re-asked unless new evidence arrived after the answer.
* Question-quality health: easy-question bias and abandonment of hard questions are measured (section 37).

Blind-trader rule (C64/C66 sections 29-31): questions rest on matured outcomes and live in MATURED_RESEARCH_STATE; an event is
refused unless its evidence date is strictly before `now`. Public entry: `generate(events, now, ledger)`. Builds on
engine.learning.research_priority, engine.learning.experiment_memory, engine.research.priority, engine.research.hypothesis_tree.
IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.experiment_memory import Hypothesis, question_key, question_similarity, to_ts
from engine.learning.research_policy import mutual_information
from engine.learning.research_priority import SignalKind, expected_for, failure_hypotheses, generic_hypotheses, identity_leak
from engine.research import priority as PRI
from engine.research.core import (ExperimentValue, FirewallBreach, Namespace, Problem, ResearchQuestion, require_past, stable_hash)
from engine.research.hypothesis_tree import HypothesisTree, build_tree

LABEL = "IMPLEMENTED - NOT VALIDATED"
EPS = 1e-9

SOURCES = ("surprise", "contradiction", "loss", "missed_winner", "pattern_break", "regime_change", "new_discovery",
           "false_positive", "data_anomaly", "coverage_gap", "research_failure")


class QuestionError(ValueError):
    """A malformed event or question object."""


# ---------------------------------------------------------------------------------------------------------- events

@dataclass(frozen=True)
class QuestionEvent:
    """Something that happened in the matured world and deserves a question. `subject` is a knowledge id / situation key /
    component name, never a ticker or date. `magnitude` is normalised by the producer to [0, 1] (share of loss, strength of
    contradiction, |z|/6 for a surprise ...); `stake` is how much decision value rides on the answer."""
    source: str
    subject: str
    evidence_through: str                          # real date of the newest matured evidence
    magnitude: float
    stake: float = 0.5
    problem: Problem = Problem.VOLATILITY
    counterpart: str = ""
    contexts: Mapping = field(default_factory=dict)
    profile: Mapping = field(default_factory=dict)  # failure evidence profile (research_priority.FAILURE_EVIDENCE keys)
    n_obs: int = 0
    loss_share: float = 0.0                        # share of the loss budget in this event's regime (section 34)
    p_isolate: float = 0.5
    p_actionable: float = 0.5
    persistence: float = 0.6
    detail: str = ""

    def check(self) -> list:
        errs = []
        if self.source not in SOURCES:
            errs.append(f"unknown source {self.source!r}")
        if not self.subject.strip():
            errs.append("event without subject")
        for name in ("magnitude", "stake", "loss_share", "p_isolate", "p_actionable", "persistence"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0) or math.isnan(v):
                errs.append(f"{name}={v!r} outside [0,1]")
        for t in (self.subject, self.counterpart, self.detail, *[str(k) + str(v) for k, v in self.contexts.items()]):
            leak = identity_leak(str(t))
            if leak:
                errs.append(f"{leak} in {str(t)[:40]!r} (identity firewall)")
        return errs


@dataclass(frozen=True)
class TestPlan:
    """How the question will be answered: named steps, the sample it needs, the controls, and the compute it costs."""
    steps: tuple
    controls: tuple
    min_n: int
    cost_minutes: float
    real_data: bool
    data_needs: tuple = ()


@dataclass(frozen=True)
class QuestionObject:
    """Section 40 research object. `question` is the shared core record; the rest is what section 40 says it must carry."""
    question: ResearchQuestion
    source: str
    hypotheses: tuple                              # engine.learning.experiment_memory.Hypothesis
    expected: tuple                                # ExpectedOutcome rows of the plan's discriminating experiment
    plan: TestPlan
    value: ExperimentValue
    priority: float
    signals: int = 1
    subject: str = ""
    kind: SignalKind = SignalKind.SURPRISE

    @property
    def qid(self) -> str:
        return self.question.question_id

    def check(self) -> list:
        errs = []
        q = self.question
        for f in ("text", "success_criterion", "failure_criterion"):
            if not getattr(q, f).strip():
                errs.append(f"{q.question_id}: empty {f}")
        if q.success_criterion.strip() == q.failure_criterion.strip():
            errs.append(f"{q.question_id}: success and failure criteria are identical: the question cannot fail")
        for f in ("text", "success_criterion", "failure_criterion"):
            leak = identity_leak(getattr(q, f))
            if leak:
                errs.append(f"{q.question_id}: {f} {leak}")
        if len(self.hypotheses) < 2:
            errs.append(f"{q.question_id}: fewer than two hypotheses")
        if not any(h.kind == "noise" for h in self.hypotheses):
            errs.append(f"{q.question_id}: no chance hypothesis")
        if abs(sum(h.prior for h in self.hypotheses) - 1.0) > 1e-6:
            errs.append(f"{q.question_id}: hypothesis priors do not sum to 1")
        if not self.plan.steps:
            errs.append(f"{q.question_id}: empty test plan")
        if self.value.compute_cost is None:
            errs.append(f"{q.question_id}: no compute estimate")
        return errs

    def to_item(self, created: str, family: str = "") -> PRI.ResearchItem:
        """The priority engine's view of this question."""
        return PRI.ResearchItem(item_id="r_" + self.qid, text=self.question.text, problem=self.question.problem,
                                family=family or f"question:{self.source}", value=self.value, created=created, real_data=self.plan.real_data,
                                data_needs=self.plan.data_needs, question_id=self.qid, tags=(self.source,))

    def to_tree(self, created_real: str) -> HypothesisTree:
        return build_tree("t_" + self.qid, self.question.text, created_real, self.hypotheses, self.expected, problem=self.question.problem.value)

    def to_dict(self) -> dict:
        q = self.question
        return {"qid": q.question_id, "text": q.text, "source": self.source, "problem": q.problem.value, "created_real": q.created_real,
                "evidence_through": q.evidence_through, "success": q.success_criterion, "failure": q.failure_criterion,
                "hypotheses": [[h.hid, h.statement, round(h.prior, 6), h.kind] for h in self.hypotheses],
                "plan": {"steps": list(self.plan.steps), "controls": list(self.plan.controls), "min_n": self.plan.min_n,
                         "cost": self.plan.cost_minutes},
                "priority": round(self.priority, 8), "subject": self.subject}


# ---------------------------------------------------------------------------------------------------------- templates

_KIND = {"surprise": SignalKind.SURPRISE, "contradiction": SignalKind.CONTRADICTION, "loss": SignalKind.FAILURE,
         "missed_winner": SignalKind.MISSED_WINNER, "pattern_break": SignalKind.WEAK_PATTERN, "regime_change": SignalKind.REGIME_TRANSITION,
         "new_discovery": SignalKind.UNKNOWN_AREA, "false_positive": SignalKind.FAILURE, "data_anomaly": SignalKind.DATA_QUALITY,
         "coverage_gap": SignalKind.UNKNOWN_AREA, "research_failure": SignalKind.REPRESENTATION_GAP}


def _ctx(e: QuestionEvent) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(e.contexts.items())) or "its usual contexts"


def question_wording(e: QuestionEvent) -> str:
    """Section-40 wording per source. The texts are the examples in the contract, made concrete and identity free."""
    c = _ctx(e)
    s = e.subject
    if e.source == "surprise":
        return f"Why did {s} move despite weak volatility signals in {c}?"
    if e.source == "contradiction":
        return f"Why did two apparently identical setups, {s} and {e.counterpart or 'its twin'}, produce opposite outcomes?"
    if e.source == "loss":
        return f"Why did the highest-confidence prediction of {s} lose in {c}?"
    if e.source == "false_positive":
        return f"Why did {s} predict a move that did not happen in {c}, and what would have told us in advance?"
    if e.source == "missed_winner":
        return f"What information existed before the movement that {s} missed?"
    if e.source == "pattern_break":
        return f"What changed before {s} stopped working in {c}?"
    if e.source == "regime_change":
        return f"Which existing knowledge survived the regime transition described by {c}, and did {s} survive it?"
    if e.source == "new_discovery":
        return f"Does the newly discovered {s} generalize beyond the data it was found in?"
    if e.source == "data_anomaly":
        return f"Is the anomaly in {s} a data fault or a real change in {c}?"
    if e.source == "coverage_gap":
        return f"What is known about {c}, which {s} has never been studied in, and is there structure there?"
    return f"Why did the repeated research on {s} fail to improve anything, and what representation is missing?"


_CRITERIA = {
    "surprise": ("an identifiable observable, known before the move, separates the surprising moves from ordinary ones with a replicated out-of-sample lift over chance",
                 "no observable known before the move beats a shuffled-label control out of sample, so the move is recorded as unpredictable"),
    "contradiction": ("one condition, measurable in advance, separates the two setups and predicts the split in a held-out period",
                      "no measurable condition separates them out of sample: the setups are recorded as one pattern with irreducible variance"),
    "loss": ("the losses concentrate in an identifiable condition whose exclusion would have avoided at least 20 percent of the loss out of sample",
             "loss concentration in any single condition is no larger than under label shuffling, so the loss is recorded as unexplained"),
    "false_positive": ("false positives cluster in a condition whose exclusion raises precision out of sample without cutting hits by more than a fifth",
                       "excluding any condition costs as many hits as it removes false positives"),
    "missed_winner": ("a signal available strictly before the move separates missed winners from non-winners with out-of-sample lift",
                      "every candidate signal only exists at or after the move, or fails out of sample"),
    "pattern_break": ("a measurable change precedes the break in most past breaks and the pattern is reliable again once it is applied as a gate",
                      "no change precedes the breaks better than chance, so the break cause is recorded UNKNOWN"),
    "regime_change": ("the item's out-of-sample reliability inside the new regime is measured and is inside or outside its old band with a stated confidence",
                      "too few observations exist in the new regime to decide, and the item is marked unknown-in-regime"),
    "new_discovery": ("the effect replicates in at least two unseen periods and one unseen group of names with the same sign and at least half the size",
                      "the effect vanishes or reverses in an unseen period or group, and the discovery is recorded as not general"),
    "data_anomaly": ("an independent source confirms or refutes the reading within tolerance", "no independent source is available, and the reading is quarantined"),
    "coverage_gap": ("a probe finds a lift over chance that survives a shuffled-label control", "the probe finds nothing beyond shuffled-label controls; the area is marked studied-and-empty"),
    "research_failure": ("a new representation produces an out-of-sample improvement larger than the repeated effort achieved",
                         "no new representation improves out-of-sample results, and the line is retired"),
}


def criteria_for(e: QuestionEvent) -> tuple:
    return _CRITERIA[e.source]


def hypotheses_for(e: QuestionEvent) -> tuple:
    """Competing explanations from the learning side. Loss-type events use the failure-cause set with the event's evidence
    profile; everything else uses the generic set for its signal kind."""
    kind = _KIND[e.source]
    if kind == SignalKind.FAILURE:
        return failure_hypotheses(e.profile)
    hs = generic_hypotheses(kind, e.subject)
    if not any(h.kind == "noise" for h in hs):                   # every question keeps a live chance explanation (section 33)
        tot = 1.0 / 0.85
        hs = tuple(replace(h, prior=h.prior / tot) for h in hs) + (
            Hypothesis("h_chance", f"the observation is a one-off outlier with no cause ({e.subject})", 0.15 / tot * 0.85 / 0.85, kind="noise"),)
        s = sum(h.prior for h in hs)
        hs = tuple(replace(h, prior=h.prior / s) for h in hs)
    return hs


def plan_for(e: QuestionEvent) -> TestPlan:
    """The test plan: every plan splits by context and era, runs the standard controls, and states the sample it needs."""
    common_controls = ("shuffled_labels", "noise_pair", "placebo_window")
    table = {
        "surprise": (("collect matched ordinary moves", "search observables known before the move", "test lift out of sample"), 60, 25.0, True),
        "contradiction": (("align the two setups on all known features", "find separating conditions", "confirm on a held-out period"), 40, 30.0, False),
        "loss": (("locate the loss regime", "split by context and era", "measure loss avoided by a gate out of sample"), 50, 20.0, False),
        "false_positive": (("group false positives by condition", "test exclusion out of sample", "cost the hits lost"), 50, 20.0, False),
        "missed_winner": (("date every candidate signal", "keep only those available before the move", "test lift out of sample"), 80, 45.0, True),
        "pattern_break": (("find the change points of the pattern", "search what changed before them", "test a gate on past breaks"), 30, 25.0, False),
        "regime_change": (("mark the regime boundary", "re-measure the item inside the regime", "compare against its old band"), 30, 30.0, True),
        "new_discovery": (("re-test in unseen periods", "re-test in unseen groups", "compare effect sizes"), 100, 60.0, True),
        "data_anomaly": (("compare against an independent source", "check timestamps and revisions"), 10, 10.0, False),
        "coverage_gap": (("define the probe", "run it with controls", "record studied-and-empty if nothing"), 100, 60.0, True),
        "research_failure": (("list what the repeated effort varied", "propose representations it did not", "screen the best three"), 60, 90.0, True)}
    steps, n, cost, real = table[e.source]
    return TestPlan(steps, common_controls, n, cost * (1.0 + 0.5 * e.magnitude), real, ("price_panel",) if real else ("pattern_ledger",))


def outcome_table(e: QuestionEvent, hyps: Sequence) -> tuple:
    return expected_for(_KIND[e.source], hyps)


def information_bits(hyps: Sequence, expected: Sequence) -> float:
    prior = {h.hid: h.prior for h in hyps}
    like: dict = {h.hid: {} for h in hyps}
    for o in expected:
        if o.hid in like:
            like[o.hid][o.outcome] = like[o.hid].get(o.outcome, 0.0) + o.probability
    for h in like:
        tot = sum(like[h].values())
        if tot > 0:
            like[h] = {k: v / tot for k, v in like[h].items()}
    return float(mutual_information(prior, like))


def value_vector(e: QuestionEvent, bits: float, plan: TestPlan, n_prior_same: int = 0) -> ExperimentValue:
    """Section-2 vector for the question. Loss reduction uses section 34's expected-loss-avoided; volatility and direction value
    come from the objective the event belongs to; nothing is claimed for objectives the event does not touch (None, not 0)."""
    loss = PRI.expected_loss_avoided(e.loss_share, e.p_isolate, e.p_actionable, e.persistence) if e.loss_share > 0 else None
    vol = e.stake * e.magnitude * 0.5 if e.problem == Problem.VOLATILITY else None
    dirv = e.stake * e.magnitude * 0.3 if e.problem == Problem.DIRECTION else None
    dec = max(e.stake * e.magnitude * 0.5, loss or 0.0)
    return PRI.make_value(plan.cost_minutes, information_bits=bits, decision=dec, uncertainty=min(1.0, bits / 1.5 + 0.2), transfer=min(1.0, 0.25 + 0.15 * len(e.contexts)),
                          failure=(e.stake * e.magnitude if e.source in ("loss", "false_positive", "pattern_break") else None), loss=loss, volatility=vol,
                          direction=dirv, overfit=min(1.0, 0.1 + 0.08 * n_prior_same), redundancy=None)


# ---------------------------------------------------------------------------------------------------------- the ledger

@dataclass
class QuestionLedger:
    """Every question ever asked, with its fate, so duplicates merge, answered questions rest, and hard-question abandonment can
    be measured. Rows are append-only; a fate change is a new row."""
    rows: list = field(default_factory=list)       # dicts: qid, key, source, asked_at, evidence_through, fate, difficulty, at
    namespace: Namespace = Namespace.MATURED_RESEARCH
    difficulty_model: Any = None                   # DifficultyModel: learned per-source difficulty prior (see attach_difficulty_model)

    def keys(self) -> set:
        return {r["key"] for r in self.rows}

    def latest(self, qid: str) -> dict | None:
        mine = [r for r in self.rows if r["qid"] == qid]
        return mine[-1] if mine else None

    def ask(self, qo: QuestionObject, now, difficulty: float) -> None:
        self.rows.append({"qid": qo.qid, "key": question_key(qo.question.text), "source": qo.source, "asked_at": str(now),
                          "evidence_through": qo.question.evidence_through, "fate": "OPEN", "difficulty": float(difficulty), "at": str(now),
                          "text": qo.question.text})

    def set_fate(self, qid: str, fate: str, now) -> None:
        prev = self.latest(qid)
        if prev is None:
            raise QuestionError(f"unknown question {qid}")
        if fate not in ("OPEN", "ANSWERED", "ABANDONED", "UNKNOWN", "MERGED"):
            raise QuestionError(f"unknown fate {fate!r}")
        self.rows.append({**prev, "fate": fate, "at": str(now)})

    def fate(self, qid: str) -> str:
        r = self.latest(qid)
        return r["fate"] if r else "NONE"

    def answered_after(self, key: str) -> str | None:
        """Evidence date of the newest ANSWERED row for this question key (None if never answered)."""
        d = [r["evidence_through"] for r in self.rows if r["key"] == key and r["fate"] == "ANSWERED"]
        return max(d) if d else None

    def similar_open(self, text: str, threshold: float = 0.8) -> str | None:
        seen = {}
        for r in self.rows:
            seen[r["qid"]] = r
        for qid, r in sorted(seen.items()):
            if r["fate"] == "OPEN" and question_similarity(text, r["text"]) >= threshold:
                return qid
        return None

    def to_json(self) -> str:
        return json.dumps(self.rows, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "QuestionLedger":
        return cls(rows=json.loads(text))


def difficulty_of(e: QuestionEvent, plan: TestPlan, bits: float) -> float:
    """0 easy .. 1 hard: cost, need for real data, and how little information the plan is expected to give."""
    return float(min(1.0, 0.4 * min(plan.cost_minutes / 120.0, 1.0) + 0.2 * plan.real_data + 0.4 * (1.0 - min(bits / 1.5, 1.0))))


# ---------------------------------------------------------------------------------------------------------- generation

def build_question(e: QuestionEvent, now, ledger: QuestionLedger | None = None, priority_state: PRI.PriorityState | None = None) -> QuestionObject:
    errs = e.check()
    if errs:
        raise QuestionError("; ".join(errs))
    require_past(e.evidence_through, now, f"event {e.source}/{e.subject}")
    text = question_wording(e)
    succ, fail = criteria_for(e)
    hyps = all_hypotheses(e)
    exp = outcome_table(e, hyps)
    plan = plan_for(e)
    bits = information_bits(hyps, exp)
    same = sum(1 for r in (ledger.rows if ledger else []) if r["source"] == e.source and r["fate"] == "OPEN")
    val = value_vector(e, bits, plan, same)
    q = ResearchQuestion.make(text, e.source, e.problem, str(now), e.evidence_through, succ, fail, expected=val)
    item = PRI.ResearchItem("r_" + q.question_id, text, e.problem, f"question:{e.source}", val, str(e.evidence_through))
    pri = 0.0
    if priority_state is not None:
        pri = priority_state.model.rate(PRI.fill_missing(item, priority_state.history))
    else:
        pri = PRI.literal_priority(PRI.fill_missing(item, []), PRI.PriorityConfig())
    qo = QuestionObject(q, e.source, hyps, exp, plan, val, pri, 1, e.subject, _KIND[e.source])
    bad = qo.check()
    if bad:
        raise QuestionError("; ".join(bad))
    return qo


def merge_objects(a: QuestionObject, b: QuestionObject) -> QuestionObject:
    """Two events asked the same question: keep the one with the newer evidence, add the signal count, take the larger value."""
    keep, other = (a, b) if a.question.evidence_through >= b.question.evidence_through else (b, a)
    v = keep.value
    ov = other.value
    merged = replace(v, decision_value=max(v.decision_value or 0.0, ov.decision_value or 0.0),
                     loss_reduction_value=max(v.loss_reduction_value or 0.0, ov.loss_reduction_value or 0.0) or None,
                     transfer_potential=max(v.transfer_potential or 0.0, ov.transfer_potential or 0.0))
    return replace(keep, value=merged, signals=a.signals + b.signals, priority=max(a.priority, b.priority))


@dataclass(frozen=True)
class GenerationReport:
    questions: tuple
    merged: int
    skipped_answered: tuple
    refused: tuple                                  # (source, subject, reason)
    reopened: tuple


def generate(events: Sequence[QuestionEvent], now, ledger: QuestionLedger | None = None,
             priority_state: PRI.PriorityState | None = None, max_new: int = 25) -> GenerationReport:
    """THE public entry. Events -> merged, de-duplicated, ranked QuestionObjects. A question whose key was already ANSWERED is skipped
    unless the event carries evidence NEWER than the answer (then it is reopened). A malformed event is refused with its reason
    and does not stop the rest; an event whose evidence is not strictly before `now` raises FirewallBreach."""
    ledger = ledger if ledger is not None else QuestionLedger()
    groups: dict = {}
    refused, skipped, reopened = [], [], []
    merged = 0
    for e in sorted(events, key=lambda e: (e.evidence_through, e.source, e.subject)):
        require_past(e.evidence_through, now, f"event {e.source}/{e.subject}")
        try:
            qo = build_question(e, now, ledger, priority_state)
        except QuestionError as err:
            refused.append((e.source, e.subject, str(err)))
            continue
        key = question_key(qo.question.text)
        ans = ledger.answered_after(key)
        if ans is not None:
            if to_ts(e.evidence_through) <= to_ts(ans):
                skipped.append(qo.qid)
                continue
            reopened.append(qo.qid)
        if key in groups:
            groups[key] = merge_objects(groups[key], qo)
            merged += 1
        else:
            groups[key] = qo
    out = sorted(groups.values(), key=lambda q: (-q.priority, q.qid))[:max_new]
    for qo in out:
        dup = ledger.similar_open(qo.question.text)
        if dup is None or dup == qo.qid:
            if ledger.latest(qo.qid) is None:
                ledger.ask(qo, now, effective_difficulty(ledger, qo.source, difficulty_of(_event_stub(qo), qo.plan, PRI_bits(qo))))
        else:
            merged += 1
    return GenerationReport(tuple(out), merged, tuple(skipped), tuple(refused), tuple(reopened))


def _event_stub(qo: QuestionObject) -> QuestionEvent:
    return QuestionEvent(qo.source, qo.subject or "x", qo.question.evidence_through, 0.5)


def PRI_bits(qo: QuestionObject) -> float:
    return float(qo.value.information_gain or 0.0)


# ---------------------------------------------------------------------------------------------------------- health of the question stream

def easy_question_bias(ledger: QuestionLedger, min_n: int = 10) -> dict:
    """Section 37: does the researcher work only on easy questions? Compares the difficulty of questions that were ANSWERED with
    those ABANDONED or left OPEN. Answered mean well below unanswered mean is the bias; a hard question that is never answered
    is reported by count."""
    last: dict = {}
    for r in ledger.rows:
        last[r["qid"]] = r
    done = [r["difficulty"] for r in last.values() if r["fate"] == "ANSWERED"]
    rest = [r["difficulty"] for r in last.values() if r["fate"] in ("ABANDONED", "OPEN")]
    if len(done) < 3 or len(rest) < 3 or len(last) < min_n:
        return {"n": len(last), "verdict": "INSUFFICIENT"}
    gap = float(np.mean(rest) - np.mean(done))
    return {"n": len(last), "answered_mean": float(np.mean(done)), "unanswered_mean": float(np.mean(rest)), "gap": gap,
            "verdict": "EASY_BIAS" if gap > 0.15 else "OK"}


def source_mix(ledger: QuestionLedger) -> dict:
    """Share of questions per source. A brain that only asks about surprises (or only losses) is missing sections of the contract."""
    c: dict = {}
    for r in {r["qid"]: r for r in ledger.rows}.values():
        c[r["source"]] = c.get(r["source"], 0) + 1
    tot = sum(c.values()) or 1
    return {s: c.get(s, 0) / tot for s in SOURCES}


def stale_open(ledger: QuestionLedger, now, days: float = 60.0) -> list:
    last = {r["qid"]: r for r in ledger.rows}
    return sorted(q for q, r in last.items() if r["fate"] == "OPEN" and (to_ts(now) - to_ts(r["asked_at"])).total_seconds() / 86400.0 > days)


def abandon_stale(ledger: QuestionLedger, now, days: float = 60.0) -> list:
    out = stale_open(ledger, now, days)
    for q in out:
        ledger.set_fate(q, "ABANDONED", now)
    return out


def loss_priority_check(objs: Sequence[QuestionObject]) -> dict:
    """Section 34: is a loss-avoiding question ranked above questions that only add tiny winner gains? Returns the best loss-type
    rank and the best winner-type rank so a report can show the ordering the priority model produced."""
    ranked = sorted(objs, key=lambda q: (-q.priority, q.qid))
    def best(pred):
        for i, q in enumerate(ranked):
            if pred(q):
                return i + 1
        return None
    return {"best_loss_rank": best(lambda q: q.source in ("loss", "false_positive") or q.question.problem == Problem.LOSS_AVOIDANCE),
            "best_winner_rank": best(lambda q: q.source in ("missed_winner", "surprise") and q.question.problem == Problem.VOLATILITY)}


def from_signal_rows(rows: Iterable[Mapping], now) -> list:
    """Adapter: plain dict rows {source, subject, when, magnitude, ...} (what the daily autopsy and scorecard emit) -> events."""
    out = []
    for r in rows:
        require_past(r["when"], now, f"row {r.get('subject')}")
        out.append(QuestionEvent(source=r["source"], subject=str(r["subject"]), evidence_through=str(r["when"]), magnitude=float(r["magnitude"]),
                                 stake=float(r.get("stake", 0.5)), problem=Problem(r.get("problem", "VOLATILITY")), counterpart=str(r.get("counterpart", "")),
                                 contexts=dict(r.get("contexts", {})), profile=dict(r.get("profile", {})), n_obs=int(r.get("n_obs", 0)),
                                 loss_share=float(r.get("loss_share", 0.0))))
    return out


# ---------------------------------------------------------------------------------------------------------- numeric criteria per source

@dataclass(frozen=True)
class Criteria:
    """Numeric success/failure conditions written BEFORE the test. Every field is used by `judge`, so the criteria are executable,
    not decoration. `min_lift` is the out-of-sample improvement over the shuffled control that counts as success; `min_n` the
    sample below which the answer is UNDECIDED (neither success nor failure); `alpha` the multiplicity-adjusted level."""
    min_lift: float
    min_n: int
    alpha: float
    loss_avoided_min: float = 0.0
    replicate_periods: int = 1
    min_size_ratio: float = 0.5
    max_hit_cost: float = 1.0

    def text(self, source: str) -> tuple:
        succ, fail = _CRITERIA[source]
        extra = f" (lift >= {self.min_lift:.2f} over control, n >= {self.min_n}, alpha {self.alpha:.3f}"
        if self.loss_avoided_min:
            extra += f", loss avoided >= {self.loss_avoided_min:.0%}"
        if self.replicate_periods > 1:
            extra += f", replicated in {self.replicate_periods} periods at >= {self.min_size_ratio:.0%} of the size"
        return succ + extra + ")", fail + f" (or n < {self.min_n}: recorded UNDECIDED, not failed)"


def numeric_criteria(e: QuestionEvent) -> Criteria:
    """Per-source thresholds. The required sample grows as the expected effect shrinks (n ~ 1/effect^2, capped), and alpha is
    tightened for sources that search many candidates."""
    eff = max(0.05, 0.5 * e.magnitude)
    n = int(min(400, max(30, 8.0 / (eff * eff))))
    base = {
        "surprise": Criteria(0.05, n, 0.01), "contradiction": Criteria(0.05, n, 0.02), "loss": Criteria(0.05, max(n, 50), 0.02, loss_avoided_min=0.20),
        "false_positive": Criteria(0.05, max(n, 50), 0.02, loss_avoided_min=0.10, max_hit_cost=0.2), "missed_winner": Criteria(0.05, max(n, 80), 0.005),
        "pattern_break": Criteria(0.05, max(30, n // 2), 0.02), "regime_change": Criteria(0.03, 30, 0.05),
        "new_discovery": Criteria(0.03, max(n, 100), 0.01, replicate_periods=2, min_size_ratio=0.5), "data_anomaly": Criteria(0.0, 10, 0.05),
        "coverage_gap": Criteria(0.05, max(n, 100), 0.005), "research_failure": Criteria(0.05, max(n, 60), 0.01)}
    return base[e.source]


@dataclass(frozen=True)
class Outcome:
    """What a finished test measured, in the terms the criteria speak."""
    n: int
    lift: float
    p_value: float
    loss_avoided: float = 0.0
    periods_replicated: int = 0
    size_ratio: float = 1.0
    hit_cost: float = 0.0
    decision_changed: bool = False
    information_bits: float = 0.0


def judge(c: Criteria, o: Outcome) -> str:
    """SUCCESS / FAILURE / UNDECIDED against the criteria. UNDECIDED (too little data) is never converted to a failure or a success."""
    if o.n < c.min_n:
        return "UNDECIDED"
    ok = o.lift >= c.min_lift and o.p_value <= c.alpha
    if c.loss_avoided_min:
        ok = ok and o.loss_avoided >= c.loss_avoided_min
    if c.replicate_periods > 1:
        ok = ok and o.periods_replicated >= c.replicate_periods and o.size_ratio >= c.min_size_ratio
    ok = ok and o.hit_cost <= c.max_hit_cost
    return "SUCCESS" if ok else "FAILURE"


def with_numeric_criteria(qo: QuestionObject, e: QuestionEvent) -> QuestionObject:
    """Attach the executable criteria text to a built question (success and failure text now carry the numbers)."""
    c = numeric_criteria(e)
    succ, fail = c.text(e.source)
    q = replace(qo.question, success_criterion=succ, failure_criterion=fail)
    return replace(qo, question=q, plan=replace(qo.plan, min_n=c.min_n))


# ---------------------------------------------------------------------------------------------------------- per-source event builders

def event_from_surprise(subject: str, z: float, when: str, sd_floor: float = 2.5, **kw) -> QuestionEvent | None:
    """A move that the volatility signals rated weak: only |z| above the floor is a question (smaller is what noise does daily)."""
    if abs(z) < sd_floor:
        return None
    return QuestionEvent("surprise", subject, when, 1.0 - math.exp(-abs(z) / 3.0), stake=min(1.0, abs(z) / 6.0), **kw)


def event_from_contradiction(a: str, b: str, agree_a: int, n_a: int, agree_b: int, n_b: int, when: str, **kw) -> QuestionEvent | None:
    """Two setups with the same recorded features but different hit rates. Strength = the gap, confidence from a two-proportion z;
    a gap the sample cannot distinguish from noise makes no question."""
    if min(n_a, n_b) < 8:
        return None
    pa, pb = agree_a / n_a, agree_b / n_b
    pool = (agree_a + agree_b) / (n_a + n_b)
    se = math.sqrt(max(pool * (1 - pool) * (1 / n_a + 1 / n_b), 1e-12))
    z = abs(pa - pb) / se
    if z < 2.0:
        return None
    return QuestionEvent("contradiction", a, when, min(1.0, abs(pa - pb)), stake=min(1.0, z / 5.0), counterpart=b, n_obs=n_a + n_b, **kw)


def event_from_loss(subject: str, loss_share: float, when: str, confidence: float = 0.5, **kw) -> QuestionEvent:
    """A loss. Confidence (how sure the model was) raises the stake: a high-confidence loss says the model's self-knowledge is wrong."""
    return QuestionEvent("loss", subject, when, min(1.0, 2.0 * loss_share), stake=min(1.0, 0.4 + 0.6 * confidence), problem=Problem.LOSS_AVOIDANCE,
                         loss_share=min(1.0, loss_share), **kw)


def event_from_missed_winner(subject: str, gain_share: float, knowable_before: float, when: str, **kw) -> QuestionEvent:
    """A winner not taken. `knowable_before` in [0,1] is the availability estimate from the could-I-have-known test; a winner that
    could not have been known gets a small magnitude (it is a data question, not a model question)."""
    return QuestionEvent("missed_winner", subject, when, min(1.0, gain_share) * (0.2 + 0.8 * knowable_before), stake=0.3 + 0.5 * knowable_before, **kw)


def event_from_break(subject: str, reliability_before: float, reliability_after: float, n_after: int, when: str, **kw) -> QuestionEvent | None:
    """A pattern whose out-of-sample reliability dropped. Needs a real sample after the drop; a drop measured on five cases is noise."""
    drop = reliability_before - reliability_after
    if drop <= 0.1 or n_after < 15:
        return None
    return QuestionEvent("pattern_break", subject, when, min(1.0, drop / max(reliability_before, 0.1)), stake=min(1.0, 0.5 + drop), n_obs=n_after,
                         problem=Problem.LOSS_AVOIDANCE, **kw)


def event_from_regime(subject: str, shift_sd: float, when: str, n_items_exposed: int = 1, **kw) -> QuestionEvent | None:
    if shift_sd < 1.0:
        return None
    return QuestionEvent("regime_change", subject, when, min(1.0, shift_sd / 4.0), stake=min(1.0, 0.3 + 0.1 * n_items_exposed), n_obs=n_items_exposed, **kw)


def event_from_discovery(subject: str, t_stat: float, n_periods_seen: int, when: str, **kw) -> QuestionEvent | None:
    """A new pattern. Every discovery gets the generalisation question, sized by how strong and how narrowly-seen it is."""
    if t_stat < 2.0:
        return None
    return QuestionEvent("new_discovery", subject, when, min(1.0, t_stat / 6.0), stake=0.7 if n_periods_seen <= 2 else 0.5, n_obs=n_periods_seen, **kw)


# ---------------------------------------------------------------------------------------------------------- learning which questions pay

@dataclass
class QuestionOutcomeBook:
    """Question-quality learning. For every finished question: did it change a decision (or knowledge state), how many bits did it
    teach, what did it cost. Beta posteriors per source give the probability that a NEW question of that kind is decision-changing;
    the priority engine multiplies by it. Dated by the day the answer matured; invisible earlier."""
    rows: list = field(default_factory=list)
    prior_a: float = 1.0
    prior_b: float = 2.0                            # start pessimistic: most questions change nothing
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def record(self, qo: QuestionObject, difficulty: float, outcome: Outcome, verdict: str, at, now) -> None:
        require_past(at, now, f"answer of {qo.qid}")
        if verdict not in ("SUCCESS", "FAILURE", "UNDECIDED"):
            raise QuestionError(f"unknown verdict {verdict!r}")
        self.rows.append({"qid": qo.qid, "source": qo.source, "difficulty": float(difficulty), "decision_changed": bool(outcome.decision_changed),
                          "bits": float(outcome.information_bits), "cost": float(qo.plan.cost_minutes), "verdict": verdict, "at": str(at),
                          "predicted_bits": float(qo.value.information_gain or 0.0)})

    def _visible(self, now) -> list:
        return [r for r in self.rows if to_ts(r["at"]) < to_ts(now)]

    def p_decision_change(self, source: str, now, difficulty: float | None = None) -> tuple:
        """(posterior mean, n). UNDECIDED answers are excluded (they neither changed a decision nor showed none would), so a hard
        source is not punished for being slow; `undecided_share` reports them."""
        rows = [r for r in self._visible(now) if r["source"] == source and r["verdict"] != "UNDECIDED"]
        if difficulty is not None:
            rows = [r for r in rows if abs(r["difficulty"] - difficulty) <= 0.25]
        k = sum(1 for r in rows if r["decision_changed"])
        return (self.prior_a + k) / (self.prior_a + self.prior_b + len(rows)), len(rows)

    def undecided_share(self, source: str, now) -> float:
        rows = [r for r in self._visible(now) if r["source"] == source]
        return sum(1 for r in rows if r["verdict"] == "UNDECIDED") / len(rows) if rows else 0.0

    def bits_per_minute(self, source: str, now) -> float:
        rows = [r for r in self._visible(now) if r["source"] == source]
        cost = sum(r["cost"] for r in rows)
        return sum(r["bits"] for r in rows) / cost if cost > 0 else 0.0

    def overall(self, now) -> float:
        rows = [r for r in self._visible(now) if r["verdict"] != "UNDECIDED"]
        k = sum(1 for r in rows if r["decision_changed"])
        return (self.prior_a + k) / (self.prior_a + self.prior_b + len(rows))

    def multiplier(self, source: str, now, difficulty: float | None = None) -> float:
        """Priority multiplier: this source's decision-change probability relative to the all-source average, shrunk to 1 with few
        observations and bounded to [0.4, 2.5] so one lucky source cannot own the queue."""
        p, n = self.p_decision_change(source, now, difficulty)
        w = n / (n + 10.0)
        return float(min(2.5, max(0.4, (1 - w) + w * p / max(self.overall(now), 1e-6))))

    def table(self, now) -> dict:
        out = {}
        for s in SOURCES:
            p, n = self.p_decision_change(s, now)
            out[s] = {"p_decision_change": p, "n": n, "undecided_share": self.undecided_share(s, now), "bits_per_minute": self.bits_per_minute(s, now),
                      "multiplier": self.multiplier(s, now)}
        return out

    def calibration(self, source: str, now) -> dict:
        """Per-source calibration: does predicted information track realised information? Ratio of realised to predicted bits with a
        ridge toward 1 (5 pseudo-observations) and a rank correlation when there is data. 'overclaims' means this source's value
        estimates are inflated and its questions should be discounted."""
        rows = [r for r in self._visible(now) if r["source"] == source and r["verdict"] != "UNDECIDED"]
        if len(rows) < 3:
            return {"source": source, "n": len(rows), "verdict": "INSUFFICIENT"}
        pred = np.array([r["predicted_bits"] for r in rows])
        real = np.array([r["bits"] for r in rows])
        scale = (real.sum() + 5.0) / (pred.sum() + 5.0)
        rho = 0.0
        if len(rows) >= 8 and pred.std() > 0 and real.std() > 0:
            rho = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(real)))[0, 1])
        return {"source": source, "n": len(rows), "scale": float(scale), "rank_corr": rho,
                "verdict": "overclaims" if scale < 0.7 else "underclaims" if scale > 1.5 else "roughly calibrated"}

    def calibrated_bits(self, source: str, bits: float, now) -> float:
        return bits * self.calibration(source, now).get("scale", 1.0)

    def to_json(self) -> str:
        return json.dumps({"rows": self.rows, "a": self.prior_a, "b": self.prior_b}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "QuestionOutcomeBook":
        d = json.loads(text)
        return cls(rows=d["rows"], prior_a=d["a"], prior_b=d["b"])


def reprioritise(objs: Sequence[QuestionObject], book: QuestionOutcomeBook, now) -> list:
    """Multiply each question's priority by its source's learned decision-change multiplier and re-sort: sources that keep
    producing decision-changing answers rise; a source that only produces unused knowledge falls."""
    out = [replace(o, priority=o.priority * book.multiplier(o.source, now)) for o in objs]
    return sorted(out, key=lambda q: (-q.priority, q.qid))


def source_starvation(ledger: QuestionLedger, book: QuestionOutcomeBook, now, min_share: float = 0.03) -> list:
    """Sources with almost no questions although they are known to pay (multiplier >= 1): opportunity being missed."""
    mix = source_mix(ledger)
    return sorted(s for s in SOURCES if mix.get(s, 0.0) < min_share and book.multiplier(s, now) >= 1.0 and len(ledger.rows) >= 20)


# ---------------------------------------------------------------------------------------------------------- against experiment memory

@dataclass(frozen=True)
class MemoryCheck:
    qid: str
    status: str                                     # NOVEL / SEEN / BLOCKED
    message: str
    similar_open: str = ""


def check_against_memory(qo: QuestionObject, experiments, now, ledger: QuestionLedger | None = None, allow_repeat_reason: str = "") -> MemoryCheck:
    """De-duplicate against experiment memory (engine.learning.experiment_memory.ExperimentLedger, duck-typed by `already_tested`)
    and against our own open questions. A question whose experiment was already run is BLOCKED unless a repeat reason (new
    evidence) is given; a merely similar one is SEEN and lets the caller decide."""
    sim = ledger.similar_open(qo.question.text) if ledger is not None else None
    if sim and sim != qo.qid:
        return MemoryCheck(qo.qid, "BLOCKED", f"an equivalent question is already open ({sim})", sim)
    if experiments is None:
        return MemoryCheck(qo.qid, "NOVEL", "no experiment memory supplied")
    v = experiments.already_tested(qo.question.text, None, now, allow_repeat_reason)
    if getattr(v, "blocking", False):
        return MemoryCheck(qo.qid, "BLOCKED", getattr(v, "message", "already tested"))
    if getattr(v, "tested_before", False):
        return MemoryCheck(qo.qid, "SEEN", getattr(v, "message", "similar experiment exists"))
    return MemoryCheck(qo.qid, "NOVEL", "not seen in experiment memory")


def filter_novel(objs: Sequence[QuestionObject], experiments, now, ledger: QuestionLedger | None = None) -> tuple:
    """(kept, blocked): blocked questions are returned with their reason, never silently dropped."""
    keep, blocked = [], []
    for o in objs:
        c = check_against_memory(o, experiments, now, ledger)
        if c.status == "BLOCKED":
            blocked.append((o, c))
        else:
            keep.append(o)
    return keep, blocked


def generate_full(events: Sequence[QuestionEvent], now, ledger: QuestionLedger | None = None, experiments=None,
                  book: QuestionOutcomeBook | None = None, priority_state: PRI.PriorityState | None = None) -> dict:
    """Everything in order: build, merge and skip via the ledger, attach numeric criteria, de-duplicate against experiment memory,
    re-rank by learned source quality. Returns the report and the stages so a caller can audit each one."""
    ledger = ledger if ledger is not None else QuestionLedger()
    rep = generate(events, now, ledger, priority_state)
    by_event = {(e.source, e.subject): e for e in events}
    objs = [with_numeric_criteria(o, by_event[(o.source, o.subject)]) if (o.source, o.subject) in by_event else o for o in rep.questions]
    kept, blocked = filter_novel(objs, experiments, now, None)
    if book is not None:
        kept = reprioritise(kept, book, now)
    return {"report": rep, "questions": kept, "blocked": blocked}


# ---------------------------------------------------------------------------------------------------------- source-specific hypotheses

def _mk_h(rows: Sequence[tuple], subject: str) -> tuple:
    """(hid, statement, kind, weight) rows -> normalised Hypothesis tuple that always contains a chance hypothesis."""
    tot = sum(r[3] for r in rows)
    return tuple(Hypothesis(h, f"{txt} ({subject})", w / tot, kind=k, mechanism_tags=(h.replace("h_", ""),)) for h, txt, k, w in rows)


def specific_hypotheses(e: QuestionEvent) -> tuple | None:
    """Explanations written for the section-40 examples, sharper than the generic sets. Returns None to fall back to the shared
    sets. Every set has a chance hypothesis and an unknown-cause residual."""
    s = e.subject
    if e.source == "new_discovery":
        return _mk_h([("h_general", "the effect holds in unseen periods and groups", "explanation", 0.25),
                      ("h_period_bound", "the effect belongs to the period it was found in", "explanation", 0.20),
                      ("h_group_bound", "the effect belongs to the group of names it was found in", "explanation", 0.15),
                      ("h_selection", "the effect is a product of searching many candidates", "noise", 0.25),
                      ("h_proxy", "the effect is a proxy for volatility or liquidity", "explanation", 0.10),
                      ("h_unknown", "none of these", "explanation", 0.05)], s)
    if e.source == "pattern_break":
        return _mk_h([("h_regime", "a regime change preceded the break", "explanation", 0.22),
                      ("h_crowding", "the effect was arbitraged away", "explanation", 0.14), ("h_data", "an input feed changed meaning", "measurement", 0.10),
                      ("h_context", "the pattern only ever worked inside a narrower context", "explanation", 0.20),
                      ("h_chance", "the drop is within ordinary variation", "noise", 0.28), ("h_unknown", "none of these", "explanation", 0.06)], s)
    if e.source == "regime_change":
        return _mk_h([("h_survives", "the item is regime independent", "explanation", 0.25), ("h_inverts", "the item inverts in the new regime", "explanation", 0.15),
                      ("h_weakens", "the item weakens but keeps its sign", "explanation", 0.25), ("h_untestable", "there is not enough regime data to say", "measurement", 0.20),
                      ("h_chance", "any difference is sampling noise", "noise", 0.15)], s)
    if e.source == "missed_winner":
        return _mk_h([("h_unknowable", "nothing available beforehand distinguished it", "noise", 0.35), ("h_signal", "a feature we already compute flagged it", "explanation", 0.15),
                      ("h_filter", "one of our own filters removed it", "explanation", 0.20), ("h_late", "the information existed but arrived too late to trade", "explanation", 0.15),
                      ("h_missing", "a data source we do not use held the signal", "measurement", 0.10), ("h_unknown", "none of these", "explanation", 0.05)], s)
    return None


def all_hypotheses(e: QuestionEvent) -> tuple:
    """Specific set when one exists, else the shared learning-side set (chance hypothesis guaranteed)."""
    return specific_hypotheses(e) or hypotheses_for(e)


# ---------------------------------------------------------------------------------------------------------- executable tests

def permutation_lift(y: Sequence[float] | np.ndarray, flag: Sequence[bool] | np.ndarray, seed: int = 0, n_perm: int = 500) -> tuple:
    """Lift of mean(y | flag) over mean(y | not flag) with a one-sided permutation p-value against shuffled flags. This is the
    'lift over the shuffled control' the criteria speak of. Returns (lift, p, n)."""
    y = np.asarray(y, float)
    f = np.asarray(flag, bool)
    if y.size != f.size:
        raise QuestionError("y and flag differ in length")
    if f.sum() < 2 or (~f).sum() < 2:
        return 0.0, 1.0, int(y.size)
    obs = y[f].mean() - y[~f].mean()
    rng = np.random.default_rng(seed)
    k = int(f.sum())
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(y.size)
        cnt += int(y[idx[:k]].mean() - y[idx[k:]].mean() >= obs - 1e-12)
    return float(obs), (cnt + 1) / (n_perm + 1), int(y.size)


def outcome_from_split(y: Sequence[float], flag: Sequence[bool], seed: int = 0, decision_threshold: float = 0.05) -> Outcome:
    """Run the standard split test and express it as an Outcome. It counts as decision-changing when the lift is large enough that a
    gate on the flag would change what is traded (`decision_threshold`)."""
    lift, p, n = permutation_lift(y, flag, seed)
    bits = max(0.0, math.log2(1.0 / max(p, 1e-6))) if lift > 0 else 0.0
    return Outcome(n=n, lift=lift, p_value=p, decision_changed=lift >= decision_threshold and p <= 0.05, information_bits=min(bits, 8.0))


def outcome_from_avoidance(loss: Sequence[float] | np.ndarray, excluded: Sequence[bool] | np.ndarray, hits: Sequence[float] | None = None, seed: int = 0) -> Outcome:
    """Loss-type test: how much of the total loss would excluding the flagged cases have avoided, at what cost in gains (hits lost)?"""
    mag = np.abs(np.asarray(loss, dtype=float))
    ex = np.asarray(excluded, bool)
    tot = mag.sum()
    avoided = float(mag[ex].sum() / tot) if tot > 0 else 0.0
    lift, p, n = permutation_lift(mag, ex, seed)
    cost = 0.0
    if hits is not None:
        h = np.asarray(hits, float)
        cost = float(h[ex].sum() / h.sum()) if h.sum() > 0 else 0.0
    return Outcome(n=n, lift=max(lift, 0.0), p_value=p, loss_avoided=avoided, hit_cost=cost, decision_changed=avoided >= 0.2 and p <= 0.05,
                   information_bits=min(8.0, max(0.0, math.log2(1.0 / max(p, 1e-6)))))


def outcome_from_replication(effects: Sequence[float], ses: Sequence[float]) -> Outcome:
    """New-discovery test: effect estimates from independent periods/groups. Replicated periods = those with the same sign and a
    z above 1.64; size ratio = smallest replicated effect over the largest (1 = same size everywhere); p from the fixed-effect z."""
    e = np.asarray(effects, float)
    s = np.asarray(ses, float)
    if e.size == 0 or e.size != s.size or np.any(s <= 0):
        raise QuestionError("replication needs matching effects and positive standard errors")
    w = 1.0 / s ** 2
    pooled = float((w * e).sum() / w.sum())
    z = pooled * math.sqrt(w.sum())
    from math import erf
    p = 0.5 * (1.0 - erf(z / math.sqrt(2.0)))
    ok = (np.sign(e) == np.sign(pooled)) & (np.abs(e / s) > 1.64)
    ratio = float(np.abs(e[ok]).min() / np.abs(e[ok]).max()) if ok.sum() else 0.0
    return Outcome(n=int(e.size * 50), lift=abs(pooled), p_value=float(p), periods_replicated=int(ok.sum()), size_ratio=ratio,
                   decision_changed=bool(ok.sum() >= 2), information_bits=min(8.0, max(0.0, math.log2(1.0 / max(p, 1e-6)))))


# ---------------------------------------------------------------------------------------------------------- lineage: what an answer asks next

def follow_up_events(qo: QuestionObject, verdict: str, outcome: Outcome, when: str) -> list:
    """An answer is never the end of the tree (section 41). SUCCESS on anything but a discovery question asks whether the found
    structure generalises; SUCCESS on a discovery question asks whether it is a proxy; FAILURE on a loss question raises the chance
    the loss is an unknown cause (routed to the unknown-cause system, not dropped); UNDECIDED asks for more data on the same
    question with a larger sample. Returns events, all dated `when` (the day the answer matured)."""
    subj = qo.subject or "item"
    out = []
    if verdict == "SUCCESS":
        src = "new_discovery" if qo.source != "new_discovery" else "coverage_gap"
        out.append(QuestionEvent(src, f"{subj}_finding", when, min(1.0, 0.4 + outcome.lift), stake=0.6, problem=qo.question.problem, n_obs=outcome.n))
    elif verdict == "FAILURE" and qo.source in ("loss", "false_positive", "pattern_break"):
        out.append(QuestionEvent("research_failure", f"{subj}_unexplained", when, 0.5, stake=0.5, problem=Problem.RESEARCH_PROCESS, n_obs=outcome.n))
    elif verdict == "UNDECIDED":
        out.append(QuestionEvent(qo.source, subj, when, min(1.0, (qo.value.decision_value or 0.3) + 0.1), stake=0.5, problem=qo.question.problem,
                                 detail="more data needed", n_obs=outcome.n))
    return out


def answer(qo: QuestionObject, outcome: Outcome, event: QuestionEvent, book: QuestionOutcomeBook | None, difficulty: float, at, now,
           ledger: QuestionLedger | None = None) -> tuple:
    """Close a question: judge it against its numeric criteria, record it for quality learning, update the ledger fate, and return
    (verdict, follow-up events). UNDECIDED leaves the question OPEN; SUCCESS/FAILURE mark it ANSWERED."""
    verdict = judge(numeric_criteria(event), outcome)
    if book is not None:
        book.record(qo, difficulty, outcome, verdict, at, now)
    if ledger is not None:
        ledger.set_fate(qo.qid, "OPEN" if verdict == "UNDECIDED" else "ANSWERED", now)
    return verdict, follow_up_events(qo, verdict, outcome, str(at))


# ---------------------------------------------------------------------------------------------------------- portfolio of questions

def allocate_slots(objs: Sequence[QuestionObject], slots: int, min_per_source: int = 1, cap_share: float = 0.5) -> list:
    """Pick `slots` questions by priority but guarantee every source that has a question at least `min_per_source` and cap any one
    source at `cap_share` of the slots (a queue of only surprises, or only losses, is a monoculture)."""
    ranked = sorted(objs, key=lambda q: (-q.priority, q.qid))
    chosen: list = []
    per: dict = {}
    for s in sorted({q.source for q in ranked}):
        for q in [q for q in ranked if q.source == s][:min_per_source]:
            if len(chosen) < slots:
                chosen.append(q)
                per[s] = per.get(s, 0) + 1
    cap = max(1, int(cap_share * slots))
    for q in ranked:
        if len(chosen) >= slots:
            break
        if q in chosen or per.get(q.source, 0) >= cap:
            continue
        chosen.append(q)
        per[q.source] = per.get(q.source, 0) + 1
    return sorted(chosen, key=lambda q: (-q.priority, q.qid))


def themes(objs: Sequence[QuestionObject], threshold: float = 0.45) -> list:
    """Group questions whose wording overlaps into themes (single-link on question similarity). A theme with many questions and no
    answer is one research direction seen from many sides; it should get ONE tree, not many."""
    groups: list = []
    for q in sorted(objs, key=lambda q: q.qid):
        for g in groups:
            if any(question_similarity(q.question.text, m.question.text) >= threshold for m in g):
                g.append(q)
                break
        else:
            groups.append([q])
    return sorted((tuple(m.qid for m in g) for g in groups), key=lambda t: (-len(t), t))


def age_priority(qo: QuestionObject, now, half_life_days: float = 90.0, floor: float = 0.3) -> QuestionObject:
    """Evidence goes stale: priority halves every `half_life_days` since the newest evidence, bounded below so nothing vanishes."""
    age = max(0.0, (to_ts(now) - to_ts(qo.question.evidence_through)).total_seconds() / 86400.0)
    return replace(qo, priority=qo.priority * max(floor, 0.5 ** (age / half_life_days)))


def ledger_integrity(ledger: QuestionLedger) -> list:
    """Problems in a ledger: fate rows for unknown questions, answered questions with no ask row, asks before their evidence."""
    errs = []
    asked = {r["qid"] for r in ledger.rows if r["fate"] == "OPEN" and r["at"] == r["asked_at"]}
    for r in ledger.rows:
        if r["qid"] not in asked and r["fate"] != "OPEN":
            errs.append(f"{r['qid']}: fate {r['fate']} without an ask row")
        if to_ts(r["asked_at"]) <= to_ts(r["evidence_through"]):
            errs.append(f"{r['qid']}: asked at/before its evidence date")
    return errs


def coverage_of_sources(objs: Sequence[QuestionObject]) -> dict:
    """Which section-40 sources produced a question in this batch, and which were silent."""
    seen = {s: 0 for s in SOURCES}
    for o in objs:
        seen[o.source] = seen.get(o.source, 0) + 1
    return {"per_source": seen, "silent": sorted(s for s, n in seen.items() if n == 0)}


# ---------------------------------------------------------------------------------------------------------- remaining source builders

def event_from_false_positive(subject: str, n_wrong: int, n_predicted: int, base_precision: float, when: str, **kw) -> QuestionEvent | None:
    """A pattern that said 'mover' and was wrong more often than its own record. Significance is a binomial tail against the
    pattern's base precision; a wrong-count within chance makes no question."""
    from scipy.stats import binom
    if n_predicted < 6 or n_wrong < 2:
        return None
    p = float(binom.sf(n_wrong - 1, n_predicted, max(1.0 - base_precision, 0.02)))
    if p > 0.10:
        return None
    return QuestionEvent("false_positive", subject, when, min(1.0, n_wrong / n_predicted), stake=min(1.0, 0.4 + (1 - p) * 0.4),
                         problem=Problem.LOSS_AVOIDANCE, n_obs=n_predicted, loss_share=min(1.0, n_wrong / n_predicted * 0.5), **kw)


def event_from_anomaly(subject: str, severity: float, when: str, **kw) -> QuestionEvent | None:
    if severity < 0.3:
        return None
    return QuestionEvent("data_anomaly", subject, when, min(1.0, severity), stake=0.8, problem=Problem.DATA_QUALITY, **kw)


def event_from_coverage(subject: str, need: float, when: str, **kw) -> QuestionEvent | None:
    """An under-studied region (need from targets.CoverageModel). Low stake by design: coverage questions must not crowd out losses."""
    if need < 0.5:
        return None
    return QuestionEvent("coverage_gap", subject, when, min(1.0, need), stake=0.3, problem=Problem.COVERAGE, **kw)


def event_from_research_failure(subject: str, barren_streak: int, when: str, **kw) -> QuestionEvent | None:
    if barren_streak < 3:
        return None
    return QuestionEvent("research_failure", subject, when, min(1.0, 1.0 - math.exp(-barren_streak / 4.0)), stake=0.5, problem=Problem.RESEARCH_PROCESS, **kw)


BUILDERS = {"surprise": event_from_surprise, "contradiction": event_from_contradiction, "loss": event_from_loss, "missed_winner": event_from_missed_winner,
            "pattern_break": event_from_break, "regime_change": event_from_regime, "new_discovery": event_from_discovery,
            "false_positive": event_from_false_positive, "data_anomaly": event_from_anomaly, "coverage_gap": event_from_coverage,
            "research_failure": event_from_research_failure}


def builders_cover_sources() -> list:
    """Sources with no event builder (should be empty: every section-40 source has one)."""
    return sorted(s for s in SOURCES if s not in BUILDERS)


# ---------------------------------------------------------------------------------------------------------- cost and difficulty learning

@dataclass
class PlanCostModel:
    """Learn how wrong the test-plan cost estimates are, per source: log(actual/planned) with a ridge toward 0. Sources whose plans
    keep costing twice the estimate are re-costed before ranking, so the priority engine compares honest numbers."""
    k: float = 3.0
    sums: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)

    def observe(self, source: str, planned: float, actual: float) -> None:
        if planned <= 0 or actual <= 0:
            raise QuestionError("planned and actual minutes must be positive")
        self.sums[source] = self.sums.get(source, 0.0) + math.log(actual / planned)
        self.n[source] = self.n.get(source, 0) + 1

    def multiplier(self, source: str) -> float:
        return math.exp(self.sums.get(source, 0.0) / (self.n.get(source, 0) + self.k))

    def recost(self, qo: QuestionObject) -> QuestionObject:
        m = self.multiplier(qo.source)
        c = (qo.value.compute_cost or 1.0) * m
        return replace(qo, value=replace(qo.value, compute_cost=c), plan=replace(qo.plan, cost_minutes=qo.plan.cost_minutes * m))


class DifficultyModel:
    """Learn what makes a question hard from how it actually went: running mean of the ratio (actual minutes / planned) and the
    UNDECIDED share, per source, blended with the a-priori difficulty formula. Feeds `easy_question_bias` a truer difficulty."""

    def __init__(self):
        self.rows: dict = {}

    def observe(self, source: str, ratio: float, undecided: bool) -> None:
        self.rows.setdefault(source, []).append((float(ratio), bool(undecided)))

    def difficulty(self, source: str, prior: float) -> float:
        r = self.rows.get(source, [])
        if not r:
            return prior
        w = len(r) / (len(r) + 6.0)
        measured = min(1.0, 0.5 * min(np.mean([x[0] for x in r]) / 3.0, 1.0) + 0.5 * np.mean([x[1] for x in r]))
        return float((1 - w) * prior + w * measured)


# ---------------------------------------------------------------------------------------------------------- presenting and refining

def refine_question(qo: QuestionObject, outcome: Outcome, now) -> QuestionObject | None:
    """An UNDECIDED question is not re-asked verbatim: the refined version asks for the sample it was short of, at the cost that
    sample implies. Returns None if the outcome was decided or the required sample is unreachable (> 5x the plan)."""
    need = qo.plan.min_n
    if outcome.n >= need:
        return None
    scale = need / max(outcome.n, 1)
    if scale > 5.0:
        return None
    plan = replace(qo.plan, cost_minutes=qo.plan.cost_minutes * min(scale, 3.0), steps=qo.plan.steps + (f"collect {need - outcome.n} more observations",))
    return replace(qo, plan=plan, value=replace(qo.value, compute_cost=None if qo.value.compute_cost is None else qo.value.compute_cost * min(scale, 3.0)))


def explain_question(qo: QuestionObject) -> str:
    """Plain-text card with the six section-40 fields, for the research report."""
    q = qo.question
    lead = max(qo.hypotheses, key=lambda h: h.prior)
    lines = [f"[{qo.source}] {q.text}", f"  hypothesis: {lead.statement} ({lead.prior:.0%}); {len(qo.hypotheses)} explanations kept alive",
             f"  expected value: info {qo.value.information_gain or 0:.2f} bits, decision {qo.value.decision_value or 0:.2f}, "
             f"loss avoided {qo.value.loss_reduction_value if qo.value.loss_reduction_value is not None else 'n/a'}, cost {qo.plan.cost_minutes:.0f} min",
             "  test plan: " + "; ".join(qo.plan.steps), f"  success: {q.success_criterion}", f"  failure: {q.failure_criterion}", f"  priority: {qo.priority:.6f}"]
    return "\n".join(lines)


def agenda_markdown(objs: Sequence[QuestionObject], limit: int = 10) -> str:
    """Today's question agenda as a markdown table."""
    rows = ["| # | source | question | priority | cost (min) |", "|---|---|---|---|---|"]
    for i, o in enumerate(sorted(objs, key=lambda q: (-q.priority, q.qid))[:limit], 1):
        rows.append(f"| {i} | {o.source} | {o.question.text} | {o.priority:.5f} | {o.plan.cost_minutes:.0f} |")
    return "\n".join(rows)


def quality_report(ledger: QuestionLedger, book: QuestionOutcomeBook, now) -> dict:
    """One dictionary for the research-brain health system: the source mix, easy-question bias, stale open questions, per-source decision
    yield and calibration, sources missing from the mix that pay, and ledger integrity."""
    return {"n_questions": len({r["qid"] for r in ledger.rows}), "source_mix": source_mix(ledger), "easy_bias": easy_question_bias(ledger),
            "stale_open": stale_open(ledger, now), "book": book.table(now), "starved_sources": source_starvation(ledger, book, now),
            "calibration": {s: book.calibration(s, now)["verdict"] for s in SOURCES}, "integrity": ledger_integrity(ledger),
            "unbuilt_sources": builders_cover_sources()}


# ---------------------------------------------------------------------------------------------------------- themes by structure

SOURCE_GROUP = {"surprise": "unexpected", "missed_winner": "unexpected", "false_positive": "wrong", "loss": "wrong", "contradiction": "conflict",
                "pattern_break": "change", "regime_change": "change", "new_discovery": "new", "coverage_gap": "new", "data_anomaly": "data",
                "research_failure": "process"}


@dataclass(frozen=True)
class Structure:
    """What a question is ABOUT, independent of its words: the kind of source, the problem, the observable features it conditions on,
    and the family of the leading explanation. Two questions with the same structure are the same research direction."""
    qid: str
    group: str
    problem: str
    features: frozenset
    family: str


def hypothesis_family(qo: QuestionObject) -> str:
    """Mechanism tag of the leading non-chance hypothesis (regime, context, decay, ...), the family of the explanation being pursued."""
    live = [h for h in qo.hypotheses if h.kind != "noise" and h.hid != "h_unknown"]
    if not live:
        return "chance"
    lead = max(live, key=lambda h: (h.prior, h.hid))
    return (lead.mechanism_tags[0] if lead.mechanism_tags else lead.hid.replace("h_", "")).lower()


def structure_of(qo: QuestionObject, event: QuestionEvent | None = None) -> Structure:
    feats = frozenset(str(k).lower() for k in ((event.contexts if event else {}) or {})) | frozenset(str(k).lower() for k in ((event.profile if event else {}) or {}))
    return Structure(qo.qid, SOURCE_GROUP.get(qo.source, qo.source), qo.question.problem.value, feats, hypothesis_family(qo))


def structure_similarity(a: Structure, b: Structure) -> float:
    """0..1: same source group 0.25, same problem 0.25, feature Jaccard 0.30 (two questions with no features share that part only if
    both have none), same explanation family 0.20."""
    fj = 1.0 if not a.features and not b.features else len(a.features & b.features) / max(len(a.features | b.features), 1)
    return 0.25 * (a.group == b.group) + 0.25 * (a.problem == b.problem) + 0.30 * fj + 0.20 * (a.family == b.family)


def themes_by_structure(items: Sequence[Structure], threshold: float = 0.7) -> list:
    """Single-link clusters on structure similarity (union-find, deterministic order). Returns tuples of qids, largest first."""
    items = sorted(items, key=lambda s: s.qid)
    parent = {s.qid: s.qid for s in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if structure_similarity(a, b) >= threshold:
                ra, rb = find(a.qid), find(b.qid)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict = {}
    for s in items:
        groups.setdefault(find(s.qid), []).append(s.qid)
    return sorted((tuple(sorted(v)) for v in groups.values()), key=lambda t: (-len(t), t))


def theme_summary(items: Sequence[Structure], objs: Sequence[QuestionObject], threshold: float = 0.7) -> list:
    """Per theme: size, the shared structure (group / problem / family), the combined priority and the lead question - so a theme
    with many questions and one tree is reported as one research direction."""
    by = {o.qid: o for o in objs}
    st = {s.qid: s for s in items}
    out = []
    for th in themes_by_structure(items, threshold):
        members = [by[q] for q in th if q in by]
        if not members:
            continue
        lead = max(members, key=lambda o: (o.priority, o.qid))
        first = st[th[0]]
        out.append({"qids": th, "size": len(th), "group": first.group, "problem": first.problem, "family": first.family,
                    "features": sorted(set().union(*[st[q].features for q in th])), "priority": sum(o.priority for o in members), "lead": lead.qid})
    return sorted(out, key=lambda r: (-r["priority"], r["qids"]))


# ---------------------------------------------------------------------------------------------------------- learned difficulty wired into the ledger

def effective_difficulty(ledger: QuestionLedger, source: str, prior: float) -> float:
    """Difficulty of a NEW question of this source: the a-priori formula blended with what past questions of the source actually cost
    and how often they ended UNDECIDED (DifficultyModel attached to the ledger). Without a model it is the prior unchanged."""
    dm = getattr(ledger, "difficulty_model", None)
    return dm.difficulty(source, prior) if dm is not None else prior


def attach_difficulty_model(ledger: QuestionLedger, model: "DifficultyModel | None" = None) -> "DifficultyModel":
    """Wire a DifficultyModel into the ledger. `generate` then records each new question's difficulty through it, and `answer_learn`
    feeds actual outcomes back."""
    ledger.difficulty_model = model or DifficultyModel()
    return ledger.difficulty_model


def answer_learn(qo: QuestionObject, outcome: Outcome, event: QuestionEvent, book: QuestionOutcomeBook | None, ledger: QuestionLedger, at, now,
                 actual_minutes: float | None = None) -> tuple:
    """`answer`, plus the learning: the ledger's DifficultyModel sees (actual/planned minutes, undecided?) and the difficulty stored
    for the question is refreshed. Returns (verdict, follow-ups)."""
    row = ledger.latest(qo.qid)
    prior = row["difficulty"] if row else 0.5
    verdict, fups = answer(qo, outcome, event, book, prior, at, now, ledger)
    dm = getattr(ledger, "difficulty_model", None)
    if dm is not None:
        ratio = (actual_minutes / qo.plan.cost_minutes) if actual_minutes else 1.0
        dm.observe(qo.source, ratio, verdict == "UNDECIDED")
    return verdict, fups


# ---------------------------------------------------------------------------------------------------------- too hard / always undecided

@dataclass(frozen=True)
class HardnessFlag:
    scope: str                                      # "source" or "question"
    key: str
    undecided_share: float
    n: int
    action: str                                     # DEPRIORITISE (source is slow) / REDESIGN (this question never resolves)
    multiplier: float
    reason: str


def undecided_streak(book: QuestionOutcomeBook, qid: str, now) -> int:
    """Consecutive most-recent answers to this question that were UNDECIDED."""
    rows = [r for r in book._visible(now) if r["qid"] == qid]
    n = 0
    for r in reversed(sorted(rows, key=lambda r: r["at"])):
        if r["verdict"] != "UNDECIDED":
            break
        n += 1
    return n


def too_hard_flags(book: QuestionOutcomeBook, now, min_n: int = 4, source_share: float = 0.6, streak: int = 3) -> list:
    """Flags for the priority engine. A SOURCE is 'too hard' when most of its answers are UNDECIDED (its plans are underpowered or its
    questions unanswerable with the data): its priority is halved and the plans should be redesigned. A QUESTION asked `streak` times in
    a row and UNDECIDED every time is 'always undecided': REDESIGN, priority to 0.25 so compute stops flowing into a dead end. A source
    that is hard but decides when it does is left alone if it changes decisions more often than average."""
    out = []
    for s in SOURCES:
        rows = [r for r in book._visible(now) if r["source"] == s]
        if len(rows) < min_n:
            continue
        share = sum(1 for r in rows if r["verdict"] == "UNDECIDED") / len(rows)
        if share >= source_share:
            p, _ = book.p_decision_change(s, now)
            if p <= book.overall(now) * 1.2:
                out.append(HardnessFlag("source", s, share, len(rows), "DEPRIORITISE", 0.5, f"{share:.0%} of {len(rows)} answers were UNDECIDED"))
    for qid in sorted({r["qid"] for r in book._visible(now)}):
        k = undecided_streak(book, qid, now)
        if k >= streak:
            out.append(HardnessFlag("question", qid, 1.0, k, "REDESIGN", 0.25, f"UNDECIDED on the last {k} attempts"))
    return out


def hardness_multipliers(flags: Sequence[HardnessFlag], objs: Sequence[QuestionObject]) -> dict:
    """{priority item id: multiplier} for engine.research.priority (PriorityState.external_multipliers). Item ids are 'r_' + qid."""
    src = {f.key: f.multiplier for f in flags if f.scope == "source"}
    qs = {f.key: f.multiplier for f in flags if f.scope == "question"}
    out = {}
    for o in objs:
        m = min(src.get(o.source, 1.0), qs.get(o.qid, 1.0))
        if m < 1.0:
            out["r_" + o.qid] = m
    return out


def apply_hardness(state: PRI.PriorityState, flags: Sequence[HardnessFlag], objs: Sequence[QuestionObject]) -> dict:
    """Install the multipliers on the priority state so its ranking uses them. Returns what was installed."""
    mult = hardness_multipliers(flags, objs)
    state.external_multipliers.update(mult)
    return mult


# ---------------------------------------------------------------------------------------------------------- persistence of learned question state

class StateIntegrityError(RuntimeError):
    """A saved question-state file failed its integrity check. Raised, never repaired silently."""


def difficulty_to_dict(dm: "DifficultyModel") -> dict:
    return {s: [[float(r), bool(u)] for r, u in rows] for s, rows in sorted(dm.rows.items())}


def difficulty_from_dict(d: Mapping) -> "DifficultyModel":
    dm = DifficultyModel()
    for s, rows in d.items():
        if s not in SOURCES:
            raise StateIntegrityError(f"difficulty rows for unknown source {s!r}")
        dm.rows[s] = [(float(r), bool(u)) for r, u in rows]
    return dm


def flag_to_dict(f: HardnessFlag) -> dict:
    return {"scope": f.scope, "key": f.key, "undecided_share": f.undecided_share, "n": f.n, "action": f.action, "multiplier": f.multiplier, "reason": f.reason}


def flag_from_dict(d: Mapping) -> HardnessFlag:
    f = HardnessFlag(str(d["scope"]), str(d["key"]), float(d["undecided_share"]), int(d["n"]), str(d["action"]), float(d["multiplier"]), str(d["reason"]))
    errs = validate_flag(f)
    if errs:
        raise StateIntegrityError("invalid flag: " + "; ".join(errs))
    return f


def validate_flag(f: HardnessFlag) -> list:
    errs = []
    if f.scope not in ("source", "question"):
        errs.append(f"scope {f.scope!r}")
    if f.scope == "source" and f.key not in SOURCES:
        errs.append(f"unknown source {f.key!r}")
    if f.action not in ("DEPRIORITISE", "REDESIGN"):
        errs.append(f"action {f.action!r}")
    if not (0.0 < f.multiplier <= 1.0):
        errs.append(f"multiplier {f.multiplier!r} must be in (0, 1]: a flag may only lower priority")
    if not (0.0 <= f.undecided_share <= 1.0) or f.n < 0:
        errs.append("undecided_share/n out of range")
    return errs


def save_question_state(path, dm: "DifficultyModel", flags: Sequence[HardnessFlag], now) -> str:
    """Atomically write the difficulty model and the current too-hard flags with a checksum. Refuses invalid flags. Returns the checksum."""
    import os
    from pathlib import Path
    for f in flags:
        errs = validate_flag(f)
        if errs:
            raise StateIntegrityError("refusing to save an invalid flag: " + "; ".join(errs))
    body = {"difficulty": difficulty_to_dict(dm), "flags": [flag_to_dict(f) for f in flags], "saved_for": str(now)}
    checksum = stable_hash(body, 24)
    payload = {"version": 1, "checksum": checksum, "body": body}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return checksum


def load_question_state(path) -> tuple:
    """(DifficultyModel, [HardnessFlag], saved_for). FileNotFoundError when there is no file (the caller decides to start fresh);
    StateIntegrityError for unreadable JSON, a wrong version, a checksum mismatch or any invalid content."""
    from pathlib import Path
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        body = payload["body"]
        ok_version = payload.get("version") == 1
        checksum = payload["checksum"]
    except (KeyError, ValueError, TypeError) as e:
        raise StateIntegrityError(f"unreadable question state: {e}") from e
    if not ok_version:
        raise StateIntegrityError(f"unknown format version {payload.get('version')!r}")
    if checksum != stable_hash(body, 24):
        raise StateIntegrityError("checksum mismatch: the file was altered or partly written")
    try:
        return difficulty_from_dict(body["difficulty"]), [flag_from_dict(d) for d in body["flags"]], body["saved_for"]
    except (KeyError, ValueError, TypeError) as e:
        raise StateIntegrityError(f"invalid content: {e}") from e


def restore_state(ledger: QuestionLedger, path, priority_state: PRI.PriorityState | None = None, objs: Sequence[QuestionObject] = ()) -> dict:
    """Resume after a loop restart: reattach the saved difficulty model to the ledger and re-install the too-hard multipliers on the
    priority state. A missing file starts fresh (empty model, no flags); a damaged file raises."""
    try:
        dm, flags, saved = load_question_state(path)
    except FileNotFoundError:
        dm, flags, saved = DifficultyModel(), [], ""
    ledger.difficulty_model = dm
    installed = apply_hardness(priority_state, flags, objs) if priority_state is not None else {}
    return {"flags": len(flags), "installed": installed, "saved_for": saved, "sources_with_history": sorted(dm.rows)}


# ---------------------------------------------------------------------------------------------------------- learned merge of themes into trees

def signature(s: Structure) -> str:
    """A theme's identity independent of any single question: its structure minus the qid and the explanation family (the family is
    what the answers are compared on)."""
    return f"{s.group}|{s.problem}|{','.join(sorted(s.features))}"


@dataclass
class ThemeMergeLearner:
    """Learns which themes are really one direction. For each answered question we record (theme signature, the family of the
    answer that won). Two signatures whose answers repeatedly agree - the same dominant answer family, distributions within a small
    total variation - are merged into ONE tree from then on, so the research stops asking the same thing under two names. Until both
    have `min_n` answers nothing is merged, and merging stops the moment their answers disagree again (it is recomputed, not sticky)."""
    min_n: int = 3
    max_tv: float = 0.25
    min_dominance: float = 0.6
    rows: list = field(default_factory=list)         # (at, signature, answer_family)

    def observe(self, sig: str, answer_family: str, at, now) -> None:
        require_past(at, now, f"answer for theme {sig}")
        if not answer_family:
            raise QuestionError("an answer needs a family (use 'unknown' or 'chance' when that is what was found)")
        self.rows.append((str(at), sig, answer_family))

    def _dist(self, sig: str, now) -> dict:
        rows = [r for r in self.rows if r[1] == sig and to_ts(r[0]) < to_ts(now)]
        n = len(rows)
        d: dict = {}
        for _, _, fam in rows:
            d[fam] = d.get(fam, 0) + 1
        return {k: v / n for k, v in d.items()} if n else {}

    def n(self, sig: str, now) -> int:
        return sum(1 for r in self.rows if r[1] == sig and to_ts(r[0]) < to_ts(now))

    def agreement(self, a: str, b: str, now) -> dict:
        da, db = self._dist(a, now), self._dist(b, now)
        if not da or not db:
            return {"n_a": self.n(a, now), "n_b": self.n(b, now), "tv": 1.0, "same_dominant": False, "merge": False, "reason": "no answers yet"}
        tv = 0.5 * sum(abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in set(da) | set(db))
        ka, kb = max(da, key=lambda k: da[k]), max(db, key=lambda k: db[k])
        na, nb = self.n(a, now), self.n(b, now)
        same = ka == kb and da[ka] >= self.min_dominance and db[kb] >= self.min_dominance
        if na < self.min_n or nb < self.min_n:
            return {"n_a": na, "n_b": nb, "tv": tv, "same_dominant": same, "merge": False, "reason": f"needs {self.min_n} answers each"}
        if ka in ("unknown", "chance") and kb == ka:
            return {"n_a": na, "n_b": nb, "tv": tv, "same_dominant": same, "merge": False, "reason": "agreeing on 'nothing found' is not a shared explanation"}
        ok = same and tv <= self.max_tv
        return {"n_a": na, "n_b": nb, "tv": tv, "same_dominant": same, "merge": ok, "reason": "answers agree" if ok else "answers differ"}

    def merge_groups(self, sigs: Sequence[str], now) -> list:
        """Union-find over signatures whose answers agree. Returns tuples of signatures, largest first (singletons included)."""
        sigs = sorted(set(sigs))
        parent = {s: s for s in sigs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a in enumerate(sigs):
            for b in sigs[i + 1:]:
                if self.agreement(a, b, now)["merge"]:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        groups: dict = {}
        for s in sigs:
            groups.setdefault(find(s), []).append(s)
        return sorted((tuple(v) for v in groups.values()), key=lambda t: (-len(t), t))

    def to_json(self) -> str:
        body = {"min_n": self.min_n, "max_tv": self.max_tv, "min_dominance": self.min_dominance, "rows": self.rows}
        return json.dumps({"checksum": stable_hash(body, 16), "body": body}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ThemeMergeLearner":
        try:
            d = json.loads(text)
            body = d["body"]
        except (KeyError, ValueError, TypeError) as e:
            raise StateIntegrityError(f"unreadable theme memory: {e}") from e
        if d.get("checksum") != stable_hash(body, 16):
            raise StateIntegrityError("theme memory checksum mismatch")
        return cls(body["min_n"], body["max_tv"], body["min_dominance"], [tuple(r) for r in body["rows"]])


def union_hypotheses(objs: Sequence[QuestionObject]) -> tuple:
    """The union of the members' hypotheses by id, priors averaged over the members that held them and renormalised; a chance
    hypothesis is guaranteed. This is the hypothesis set of the merged tree."""
    acc: dict = {}
    for o in objs:
        for h in o.hypotheses:
            acc.setdefault(h.hid, []).append(h)
    merged = [replace(hs[0], prior=float(np.mean([h.prior for h in hs]))) for _, hs in sorted(acc.items())]
    if not any(h.kind == "noise" for h in merged):
        merged.append(Hypothesis("h_chance", "the observations are chance", 0.15, kind="noise"))
    tot = sum(h.prior for h in merged)
    return tuple(replace(h, prior=h.prior / tot) for h in merged)


def merge_themes_into_trees(structs: Sequence[Structure], objs: Sequence[QuestionObject], learner: ThemeMergeLearner, now, created_real: str,
                            build=None) -> dict:
    """One tree per merged group of themes and one per remaining question. Returns {"trees": {tree_id: HypothesisTree},
    "members": {tree_id: [qid, ...]}, "merged_groups": [...]}. A merged tree is built from the union of its members' hypotheses and the
    shared discriminating experiment of the highest-priority member, and is named after the group so the same group always maps to the
    same tree id. `build` is injectable for tests."""
    build = build or build_tree
    st = {s.qid: s for s in structs}
    by = {o.qid: o for o in objs}
    groups = learner.merge_groups([signature(s) for s in structs], now)
    trees: dict = {}
    members: dict = {}
    merged_groups = []
    for grp in groups:
        qids = sorted(q for q, s in st.items() if signature(s) in grp and q in by)
        if not qids:
            continue
        lead = max((by[q] for q in qids), key=lambda o: (o.priority, o.qid))
        if len(grp) > 1 and len(qids) > 1:
            tid = "tm_" + stable_hash(list(grp), 8)
            hyps = union_hypotheses([by[q] for q in qids])
            trees[tid] = build(tid, lead.question.text, created_real, hyps, [], None, lead.question.problem.value)
            members[tid] = qids
            merged_groups.append(tuple(grp))
        else:
            for q in qids:
                trees["t_" + q] = by[q].to_tree(created_real)
                members["t_" + q] = [q]
    return {"trees": trees, "members": members, "merged_groups": merged_groups}
