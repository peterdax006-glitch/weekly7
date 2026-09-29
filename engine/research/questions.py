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
    hyps = hypotheses_for(e)
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
                ledger.ask(qo, now, difficulty_of(_event_stub(qo), qo.plan, PRI_bits(qo)))
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
