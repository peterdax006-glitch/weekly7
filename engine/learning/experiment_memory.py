"""Experiment memory (contract C62 section 31; checklist G01, G07, G08; Bible phase 30 / canon C58-C62).

Every experiment answers eleven things: QUESTION, CURRENT BELIEF, COMPETING HYPOTHESES, PREDICTION, EXPERIMENT, EXPECTED
OUTCOMES, RESULT, BELIEF UPDATE, WHAT WAS LEARNED, WHAT WAS NOT LEARNED, NEXT ACTION. This module stores them as immutable
versioned records in an append-only jsonl ledger, and answers the question asked BEFORE launching anything: "we already
tested this" (exact repeat, same question answered, near-duplicate configuration, known failure nearby, already in flight).

Built on (not a copy of): engine.registry.fingerprint (canonical config hash), engine.experiment_memory.Space (normalised
configuration distance) and engine.experiment_memory.TriedIndex (search-space index; `sync_to_tried_index` feeds it). The
legacy state/experiments.jsonl log is imported honestly: fields it never recorded stay "NOT RECORDED", never invented.

Time discipline (C56): every read takes `now` and sees only versions recorded strictly before it and results observed
strictly before it. A result observed at/after `now` is invisible, so a decision at `now` cannot see its own future.
IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from engine import experiment_memory as legacy_space
from engine import registry as legacy_registry

from .core import (FailureCause, FirewallBreach, Subsystem, ValidationLabel, canonical_json, current_code_hash,
                   stable_hash)

LABEL = ValidationLabel.NOT_VALIDATED.value
NOT_RECORDED = "NOT RECORDED"


class ExperimentStatus(str):
    """Plain string constants (json-friendly): the lifecycle of one experiment."""
    PROPOSED = "PROPOSED"
    RUNNING = "RUNNING"
    ANSWERED = "ANSWERED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"            # the test itself was defective; says nothing about the question
    FAILED = "FAILED"              # crashed / never produced a result
    ABANDONED = "ABANDONED"
    ALL = ("PROPOSED", "RUNNING", "ANSWERED", "INCONCLUSIVE", "INVALID", "FAILED", "ABANDONED")
    TERMINAL = ("ANSWERED", "INCONCLUSIVE", "INVALID", "FAILED", "ABANDONED")


class ResultKind(str):
    CONFIRMED = "CONFIRMED"        # the prediction / leading hypothesis held
    REFUTED = "REFUTED"
    MIXED = "MIXED"
    NULL = "NULL"                  # nothing detectable; power stated in the result
    ALL = ("CONFIRMED", "REFUTED", "MIXED", "NULL")
    DEFINITIVE = ("CONFIRMED", "REFUTED", "NULL")


class DuplicateStatus(str):
    NOVEL = "NOVEL"
    IN_FLIGHT = "IN_FLIGHT"
    SAME_CONFIG_REPEAT = "SAME_CONFIG_REPEAT"
    SAME_QUESTION_ANSWERED = "SAME_QUESTION_ANSWERED"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    KNOWN_FAILURE_NEARBY = "KNOWN_FAILURE_NEARBY"
    RETEST_JUSTIFIED = "RETEST_JUSTIFIED"
    BLOCKING = ("IN_FLIGHT", "SAME_CONFIG_REPEAT", "SAME_QUESTION_ANSWERED", "NEAR_DUPLICATE", "KNOWN_FAILURE_NEARBY")


SECTION_31_FIELDS = ("question", "current_belief", "competing_hypotheses", "prediction", "experiment", "expected_outcomes",
                     "result", "belief_update", "learned", "not_learned", "next_action")


class DuplicateExperiment(Exception):
    """Raised by ExperimentLedger.propose when the experiment is a blocked repeat and no repeat reason was given."""

    def __init__(self, verdict: "DuplicateVerdict"):
        super().__init__(verdict.message)
        self.verdict = verdict


# ------------------------------------------------------------------------------------------------ time helpers

def to_ts(x) -> dt.datetime:
    """Any date/datetime/ISO string -> naive datetime. A bare date is midnight, so 'a date' orders before that day's events."""
    if isinstance(x, dt.datetime):
        return x.replace(tzinfo=None)
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day)
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime().replace(tzinfo=None)
    s = str(x)
    if not s or s in ("None", NOT_RECORDED):
        raise FirewallBreach(f"missing timestamp {x!r}")
    return dt.datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=None)


def _iso(x) -> str:
    return to_ts(x).isoformat()


# ------------------------------------------------------------------------------------------------ question language

_STOP = frozenset("a an the of to in on for and or is are was were be been do does did we it its this that these those with by "
                  "from as at than then not no how what which why when whether if can could should would will there their".split())
_SYN = {"improve": "gain", "improves": "gain", "improved": "gain", "better": "gain", "boost": "gain", "lift": "gain",
        "worse": "loss", "degrade": "loss", "degrades": "loss", "fail": "loss", "fails": "loss", "failure": "loss",
        "transfer": "generalize", "transfers": "generalize", "generalise": "generalize", "generalises": "generalize",
        "generalizes": "generalize", "predict": "forecast", "predicts": "forecast", "prediction": "forecast",
        "regimes": "regime", "years": "year", "weeks": "week", "patterns": "pattern", "stocks": "stock",
        "vol": "volatility", "volatile": "volatility", "oos": "outofsample", "unseen": "outofsample"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def question_tokens(text: str) -> frozenset:
    """Order- and wording-insensitive bag of content words. 'Does X improve OOS?' == 'Is there an unseen-year gain from X?'"""
    words = re.findall(r"[a-z0-9_]+", str(text).lower().replace("-", " ").replace("/", " "))
    out = set()
    for w in words:
        w = _SYN.get(w, w)
        if w in _STOP or len(w) < 2:
            continue
        out.add(_SYN.get(_stem(w), _stem(w)))
    return frozenset(out)


def question_similarity(a: str, b: str) -> float:
    """Jaccard over content tokens, with a containment bonus so a short question fully inside a longer one still matches."""
    ta, tb = question_tokens(a), question_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jac = inter / len(ta | tb)
    contain = inter / min(len(ta), len(tb))
    return float(min(1.0, 0.7 * jac + 0.3 * contain))


def question_key(text: str) -> str:
    return stable_hash(sorted(question_tokens(text)), 12)


# ------------------------------------------------------------------------------------------------ section-31 value types

@dataclass(frozen=True)
class Hypothesis:
    hid: str
    statement: str
    prior: float                                   # belief before the experiment; a set of hypotheses must sum to 1
    kind: str = "explanation"                      # explanation | null | noise | measurement | mechanism
    mechanism_tags: tuple = ()
    cause: str = ""                                # FailureCause value when the hypothesis is a failure explanation

    def check(self) -> list:
        errs = []
        if not self.hid:
            errs.append("hypothesis without id")
        if not self.statement.strip():
            errs.append(f"hypothesis {self.hid}: empty statement")
        if not (isinstance(self.prior, (int, float)) and 0.0 <= self.prior <= 1.0) or math.isnan(self.prior):
            errs.append(f"hypothesis {self.hid}: prior {self.prior!r} outside [0,1]")
        if self.cause:
            try:
                FailureCause.parse(self.cause)
            except ValueError:
                errs.append(f"hypothesis {self.hid}: unknown cause {self.cause!r}")
        return errs


@dataclass(frozen=True)
class ExpectedOutcome:
    """What the experiment would show IF hypothesis `hid` were true: P(outcome label | hid) plus an optional metric band."""
    hid: str
    outcome: str
    probability: float
    metric: str = ""
    lo: float | None = None
    hi: float | None = None

    def check(self) -> list:
        errs = []
        if not (0.0 <= self.probability <= 1.0) or math.isnan(self.probability):
            errs.append(f"expected outcome {self.hid}/{self.outcome}: probability {self.probability!r}")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            errs.append(f"expected outcome {self.hid}/{self.outcome}: lo > hi")
        return errs


@dataclass(frozen=True)
class Prediction:
    statement: str
    metric: str = ""
    direction: str = ""                            # up | down | none | within
    magnitude: float | None = None
    confidence: float = 0.5                        # stated BEFORE the run so calibration can be measured later

    def check(self) -> list:
        errs = []
        if not self.statement.strip():
            errs.append("prediction: empty statement")
        if self.direction not in ("", "up", "down", "none", "within"):
            errs.append(f"prediction: unknown direction {self.direction!r}")
        if not (0.0 <= self.confidence <= 1.0):
            errs.append("prediction: confidence outside [0,1]")
        return errs


@dataclass(frozen=True)
class DesignSpec:
    """The experiment itself: configuration, data windows, seed, controls, cost. `data_hash`/`code_hash` are what makes
    a repeat a genuine retest (the world changed) rather than a duplicate."""
    config: Mapping = field(default_factory=dict)
    windows: tuple = ()
    seed: int | None = None
    controls: tuple = ()
    cost_minutes: float = 0.0
    subsystem: str = ""
    target: str = ""                               # ResearchTarget value, kept as text to avoid a hard import cycle
    data_hash: str = ""
    code_hash: str = ""

    def check(self) -> list:
        errs = []
        if self.subsystem:
            try:
                Subsystem.parse(self.subsystem)
            except ValueError:
                errs.append(f"design: unknown subsystem {self.subsystem!r}")
        if self.cost_minutes < 0 or math.isnan(self.cost_minutes):
            errs.append("design: negative cost")
        if not self.config and not self.windows:
            errs.append("design: neither config nor windows given, nothing is being tested")
        if self.seed is None:
            errs.append("design: no seed (C56: every draw is seeded)")
        return errs

    def fingerprint(self) -> str:
        return legacy_registry.fingerprint({"config": dict(self.config), "windows": list(self.windows),
                                            "controls": list(self.controls)})


@dataclass(frozen=True)
class ExperimentResult:
    kind: str                                      # ResultKind, or INVALID marker via invalid_reason
    outcome: str = ""                              # label matching one of the ExpectedOutcome.outcome strings
    metrics: Mapping = field(default_factory=dict)
    n: int = 0
    ci: tuple = ()                                 # (lo, hi) of the headline metric, when there is one
    power_note: str = ""                           # NULL results must say what effect size they could have detected
    observed_at: str = ""
    summary: str = ""
    invalid_reason: str = ""

    def check(self) -> list:
        errs = []
        if self.kind not in ResultKind.ALL and not self.invalid_reason:
            errs.append(f"result: unknown kind {self.kind!r}")
        if not self.observed_at:
            errs.append("result: observed_at missing")
        if self.kind == ResultKind.NULL and not self.power_note:
            errs.append("result: a NULL result must state its power (a null with no power is 'unknown', not 'no')")
        if self.ci and (len(self.ci) != 2 or self.ci[0] > self.ci[1]):
            errs.append("result: ci must be (lo, hi)")
        return errs


@dataclass(frozen=True)
class BeliefUpdate:
    prior_belief: str
    posterior_belief: str
    prior: Mapping = field(default_factory=dict)       # hid -> prior probability
    posterior: Mapping = field(default_factory=dict)   # hid -> posterior probability
    moved: float = 0.0                                 # total-variation distance between prior and posterior
    surprise_bits: float = 0.0                         # -log2 P(observed outcome) under the prior predictive

    def check(self) -> list:
        errs = []
        if self.posterior and abs(sum(self.posterior.values()) - 1.0) > 1e-6:
            errs.append("belief update: posterior does not sum to 1")
        if not self.posterior_belief.strip():
            errs.append("belief update: posterior_belief empty")
        return errs


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    version: int
    status: str
    question: str
    current_belief: str
    competing_hypotheses: tuple                        # tuple[Hypothesis]
    prediction: Prediction
    experiment: DesignSpec
    expected_outcomes: tuple                           # tuple[ExpectedOutcome]
    created_at: str
    recorded_at: str
    result: ExperimentResult | None = None
    belief_update: BeliefUpdate | None = None
    learned: tuple = ()
    not_learned: tuple = ()
    next_action: str = ""
    tags: tuple = ()
    parent_ids: tuple = ()
    repeat_reason: str = ""
    legacy: bool = False
    legacy_missing: tuple = ()

    # -------------------------------------------------------------- validation
    def validate(self, stage: str = "proposal") -> list:
        """stage 'proposal' | 'answered'. An answered record without WHAT WAS NOT LEARNED is refused: 'nothing' is not an answer."""
        errs = []
        if not self.experiment_id:
            errs.append("no experiment_id")
        if self.status not in ExperimentStatus.ALL:
            errs.append(f"unknown status {self.status!r}")
        if not self.question.strip():
            errs.append("question empty")
        if not self.current_belief.strip():
            errs.append("current_belief empty")
        hyps = self.competing_hypotheses
        if len(hyps) < 2:
            errs.append("fewer than two competing hypotheses (a one-hypothesis experiment cannot be surprised)")
        for h in hyps:
            errs += h.check()
        if hyps and abs(sum(h.prior for h in hyps) - 1.0) > 1e-6:
            errs.append(f"hypothesis priors sum to {sum(h.prior for h in hyps):.6f}, not 1")
        if len({h.hid for h in hyps}) != len(hyps):
            errs.append("duplicate hypothesis ids")
        errs += self.prediction.check() + self.experiment.check()
        hids = {h.hid for h in hyps}
        covered = {o.hid for o in self.expected_outcomes}
        for o in self.expected_outcomes:
            errs += o.check()
            if o.hid not in hids:
                errs.append(f"expected outcome refers to unknown hypothesis {o.hid!r}")
        for hid in sorted(hids - covered):
            errs.append(f"hypothesis {hid!r} has no expected outcome")
        for hid in sorted(covered & hids):
            tot = sum(o.probability for o in self.expected_outcomes if o.hid == hid)
            if abs(tot - 1.0) > 1e-6:
                errs.append(f"expected outcomes of {hid!r} sum to {tot:.6f}, not 1")
        if stage == "answered":
            if self.result is None:
                errs.append("no result")
            else:
                errs += self.result.check()
            if self.belief_update is None:
                errs.append("no belief update")
            else:
                errs += self.belief_update.check()
            if not self.learned:
                errs.append("WHAT WAS LEARNED empty")
            if not self.not_learned:
                errs.append("WHAT WAS NOT LEARNED empty (state the limits of the test)")
            if not self.next_action.strip():
                errs.append("NEXT ACTION empty")
        return errs

    def completeness(self) -> float:
        """Share of the eleven section-31 fields genuinely filled (NOT RECORDED counts as empty)."""
        filled = 0
        for f in SECTION_31_FIELDS:
            v = getattr(self, f)
            if v is None or v == () or v == "" or v == NOT_RECORDED or f in self.legacy_missing:
                continue
            if isinstance(v, (list, tuple)) and all(x == NOT_RECORDED for x in v):
                continue
            filled += 1
        return filled / len(SECTION_31_FIELDS)

    # -------------------------------------------------------------- serialisation
    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "ExperimentRecord":
        def tup(x):
            return tuple(x or ())
        exp = d["experiment"]
        res = d.get("result")
        bu = d.get("belief_update")
        return cls(
            experiment_id=d["experiment_id"], version=int(d["version"]), status=d["status"], question=d["question"],
            current_belief=d["current_belief"],
            competing_hypotheses=tuple(Hypothesis(**{**h, "mechanism_tags": tup(h.get("mechanism_tags"))})
                                       for h in d["competing_hypotheses"]),
            prediction=Prediction(**d["prediction"]),
            experiment=DesignSpec(**{**exp, "windows": tup(exp.get("windows")), "controls": tup(exp.get("controls"))}),
            expected_outcomes=tuple(ExpectedOutcome(**o) for o in d["expected_outcomes"]),
            created_at=d["created_at"], recorded_at=d["recorded_at"],
            result=None if res is None else ExperimentResult(**{**res, "ci": tup(res.get("ci"))}),
            belief_update=None if bu is None else BeliefUpdate(**bu),
            learned=tup(d.get("learned")), not_learned=tup(d.get("not_learned")), next_action=d.get("next_action", ""),
            tags=tup(d.get("tags")), parent_ids=tup(d.get("parent_ids")), repeat_reason=d.get("repeat_reason", ""),
            legacy=bool(d.get("legacy", False)), legacy_missing=tup(d.get("legacy_missing")))

    def with_(self, **kw) -> "ExperimentRecord":
        return dataclasses.replace(self, **kw)


# ------------------------------------------------------------------------------------------------ Bayesian belief update

def normalise(p: Mapping[str, float], floor: float = 0.0) -> dict:
    """Probabilities that sum to exactly 1; a zero total is refused rather than turned into a uniform guess."""
    vals = {k: max(float(v), floor) for k, v in p.items()}
    tot = sum(vals.values())
    if not vals or tot <= 0 or math.isnan(tot):
        raise ValueError("cannot normalise an empty / zero / NaN distribution")
    return {k: v / tot for k, v in vals.items()}


def outcome_likelihoods(hyps: Sequence[Hypothesis], expected: Sequence[ExpectedOutcome], observed: str,
                        smoothing: float = 0.02) -> dict:
    """P(observed outcome | h) for every hypothesis. `smoothing` keeps a never-predicted outcome from zeroing a hypothesis
    for ever (the model of the experiment can itself be wrong); an outcome label no hypothesis mentions is a surprise the
    caller must handle (likelihoods then all equal `smoothing`, so nothing moves)."""
    out = {}
    for h in hyps:
        p = sum(o.probability for o in expected if o.hid == h.hid and o.outcome == observed)
        out[h.hid] = (1 - smoothing) * p + smoothing / max(1, len({o.outcome for o in expected}))
    return out


def bayes_update(prior: Mapping[str, float], likelihood: Mapping[str, float]) -> dict:
    return normalise({k: prior[k] * likelihood.get(k, 0.0) for k in prior})


def robust_update(prior: Mapping[str, float], likelihood: Mapping[str, float], unknown_hid: str = "", misfit_below: float = 0.12,
                  leak: float = 0.30) -> tuple:
    """Bayes update with a MODEL-MISFIT GUARD. When the observed outcome is improbable under EVERY named hypothesis (best
    likelihood < misfit_below) the hypothesis set is probably incomplete, and a plain Bayes update would crown whichever
    wrong hypothesis is least wrong. In that case a fraction `leak` of the posterior moves to the explicit UNKNOWN
    hypothesis (section 9: unknown is an answer). Returns (posterior, misfit_flag). Without an unknown hypothesis it is Bayes."""
    post = bayes_update(prior, likelihood)
    best = max(likelihood.values()) if likelihood else 1.0
    if unknown_hid and unknown_hid in post and best < misfit_below:
        moved = {k: v * (1 - leak) for k, v in post.items() if k != unknown_hid}
        moved[unknown_hid] = post[unknown_hid] * (1 - leak) + leak
        return normalise(moved), True
    return post, False


def predictive_probability(prior: Mapping[str, float], expected: Sequence[ExpectedOutcome], observed: str) -> float:
    return sum(prior.get(o.hid, 0.0) * o.probability for o in expected if o.outcome == observed)


def total_variation(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def make_belief_update(rec: ExperimentRecord, observed: str, posterior_belief: str = "", smoothing: float = 0.02) -> BeliefUpdate:
    """Belief update computed from the record's own hypotheses and expected outcomes, not typed in. The leading posterior
    hypothesis names the default posterior_belief when none is given."""
    prior = {h.hid: h.prior for h in rec.competing_hypotheses}
    like = outcome_likelihoods(rec.competing_hypotheses, rec.expected_outcomes, observed, smoothing)
    unknown = next((h.hid for h in rec.competing_hypotheses if h.cause == FailureCause.UNKNOWN.value or h.hid == "h_unknown"), "")
    post, _ = robust_update(prior, like, unknown)
    pred = predictive_probability(prior, rec.expected_outcomes, observed)
    bits = -math.log2(max(pred, 1e-9)) if pred > 0 else -math.log2(1e-9)
    lead = max(post, key=post.get)
    lead_stmt = next(h.statement for h in rec.competing_hypotheses if h.hid == lead)
    return BeliefUpdate(prior_belief=rec.current_belief,
                        posterior_belief=posterior_belief or f"leading hypothesis {lead}: {lead_stmt}",
                        prior=prior, posterior=post, moved=total_variation(prior, post), surprise_bits=bits)


# ------------------------------------------------------------------------------------------------ duplicate detection

@dataclass(frozen=True)
class DuplicateMatch:
    experiment_id: str
    version: int
    status: str
    question_similarity: float
    config_distance: float
    same_fingerprint: bool
    result_kind: str
    recorded_at: str
    learned: tuple = ()


@dataclass(frozen=True)
class DuplicateVerdict:
    status: str
    message: str
    matches: tuple = ()
    retest_reasons: tuple = ()

    @property
    def blocking(self) -> bool:
        return self.status in DuplicateStatus.BLOCKING

    @property
    def tested_before(self) -> bool:
        return self.status != DuplicateStatus.NOVEL


def infer_space(configs: Iterable[Mapping]) -> "legacy_space.Space":
    """A Space declared from observed configurations: numeric parameters are scaled by their largest magnitude (so distance
    means relative difference even when only two configs exist), everything else is categorical over the values seen. Lets the legacy distance be reused without hand-declaring spaces."""
    seen: dict = {}
    for c in configs:
        for k, v in c.items():
            seen.setdefault(k, []).append(v)
    spec: dict[str, Any] = {}
    for k, vs in seen.items():
        nums = [v for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        if nums and len(nums) == len(vs):
            m = max(abs(v) for v in nums)
            if m > 0:                                  # scale by magnitude, not by the (tiny) observed range: 0.80 vs 0.81 is near
                spec[k] = (-float(m), float(m), float(m) / 50.0)
                continue
        uniq = []
        for v in vs:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) >= 2 and all(isinstance(u, (str, int, float, bool)) for u in uniq):
            spec[k] = uniq
    return legacy_space.Space(spec)


def config_distance(a: Mapping, b: Mapping, space=None) -> float:
    """Mean normalised difference over the union of parameters (0 identical, 1 completely different)."""
    sp = space or infer_space([a, b])
    return sp.distance(sp.snap(dict(a)), sp.snap(dict(b)))


# ------------------------------------------------------------------------------------------------ legacy façade

class LegacyBridge:
    """ONE façade over the two older dedup mechanisms (CONTRACT_MAPPING: 'one façade in engine/learning/experiment_memory.py'):
    engine.experiment_memory.TriedIndex (search-space index) and engine.registry.ExperimentMemory (twelve-answer lessons,
    keyed by registry.fingerprint). The ledger consults both when asked 'did we test this?' and mirrors every answered
    experiment into both, so neither older store can silently disagree with the ledger and no third dedup path exists."""

    def __init__(self, tried_index=None, memory=None):
        self.tried = tried_index
        self.memory = memory

    def check(self, config: Mapping) -> list:
        """[(source, status, blocking, message)] from the older stores for this configuration."""
        out: list[tuple[str, str, bool, str]] = []
        if not config:
            return out
        if self.tried is not None:
            v = self.tried.check(dict(config))
            if v["verdict"] != "novel":
                st = {"repeat": DuplicateStatus.SAME_CONFIG_REPEAT, "near_duplicate": DuplicateStatus.NEAR_DUPLICATE,
                      "known_failure_nearby": DuplicateStatus.KNOWN_FAILURE_NEARBY}[v["verdict"]]
                out.append(("TriedIndex", st, bool(v["block"]),
                            f"TriedIndex {v['verdict']}: {v['exact'] or [i for _, i in v['close']]} reasons={v['reasons']}"))
        if self.memory is not None:
            h = self.memory.already_tried(dict(config))
            if h["tried"]:
                out.append(("registry.ExperimentMemory", DuplicateStatus.SAME_CONFIG_REPEAT if h["blocked"] else DuplicateStatus.NEAR_DUPLICATE,
                            bool(h["blocked"]), f"registry memory: tried {h['tried']}x {h['ids']} blocked={h['blocked']}"))
        return out

    @staticmethod
    def legacy_answers(rec: "ExperimentRecord") -> dict:
        """The twelve registry answers from a section-31 record. What the record does not hold is stated as such, never invented."""
        r = rec.result
        nm = "not measured in this record"
        metrics = dict(r.metrics) if r and r.metrics else {}
        return {"what_changed": rec.question, "why_changed": rec.current_belief,
                "data_used": ", ".join(rec.experiment.windows) or nm, "data_unseen": "; ".join(rec.experiment.controls) or nm,
                "baseline": rec.current_belief, "improved": json.dumps(metrics, sort_keys=True, default=str) if metrics else (r.kind if r else nm),
                "worsened": nm, "statistically_meaningful": (f"ci={list(r.ci)}" if r and r.ci else nm),
                "risk_changed": nm, "survived_another_window": nm,
                "adopted": bool(r and r.kind == ResultKind.CONFIRMED),
                "if_rejected_why": (rec.learned[0] if rec.learned else (r.summary if r else "")) or "refuted"}

    def mirror(self, rec: "ExperimentRecord", now) -> dict:
        """Write an answered experiment into both older stores (idempotent: an id already present is skipped)."""
        done = {"tried_index": False, "registry_memory": False}
        if rec.result is None or rec.status not in (ExperimentStatus.ANSWERED, ExperimentStatus.INCONCLUSIVE) or not rec.experiment.config:
            return done
        if self.tried is not None and not any(x["experiment_id"] == rec.experiment_id for x in self.tried.rows):
            outcome = {"CONFIRMED": "adopt", "REFUTED": "reject"}.get(rec.result.kind, "continue_testing")
            score = rec.result.metrics.get("score") if isinstance(rec.result.metrics, Mapping) else None
            reason = (rec.learned[0] if rec.learned else rec.result.summary) or "refuted"
            self.tried.add(rec.experiment_id, dict(rec.experiment.config), outcome, score if isinstance(score, (int, float)) else None,
                           reason if outcome == "reject" else None, now=now)
            done["tried_index"] = True
        if self.memory is not None and not any(e["experiment_id"] == rec.experiment_id for e in self.memory.entries):
            self.memory.record(rec.experiment_id, dict(rec.experiment.config), self.legacy_answers(rec), now)
            done["registry_memory"] = True
        return done


# ------------------------------------------------------------------------------------------------ ledger

class ExperimentLedger:
    """Append-only, versioned ledger. propose() writes version 1; each later state change writes a NEW version of the same id.
    Nothing is edited or deleted. `view(now)` is the only way to read: it is the ledger as it stood before `now`."""

    def __init__(self, path=None, near: float = 0.08, question_match: float = 0.72, bridge: LegacyBridge | None = None):
        self.bridge = bridge
        self.path = Path(path) if path else None
        self.near = near
        self.question_match = question_match
        self._rows: list[ExperimentRecord] = []
        self.unparseable = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._rows.append(ExperimentRecord.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    self.unparseable += 1                      # counted, never silently dropped

    # -------------------------------------------------------------- reading (always as of a time)
    def view(self, now) -> dict:
        """experiment_id -> latest version recorded strictly before `now`, with a result hidden unless it was observed
        strictly before `now` too (its status then falls back to RUNNING)."""
        cut = to_ts(now)
        latest: dict = {}
        for r in self._rows:
            if to_ts(r.recorded_at) >= cut:
                continue
            if r.result is not None and r.result.observed_at and to_ts(r.result.observed_at) >= cut:
                r = r.with_(result=None, belief_update=None, learned=(), not_learned=(), next_action="",
                            status=ExperimentStatus.RUNNING)
            if r.experiment_id not in latest or r.version >= latest[r.experiment_id].version:
                latest[r.experiment_id] = r
        return latest

    def get(self, experiment_id: str, now) -> ExperimentRecord | None:
        return self.view(now).get(experiment_id)

    def history(self, experiment_id: str) -> list:
        return sorted((r for r in self._rows if r.experiment_id == experiment_id), key=lambda r: r.version)

    def __len__(self):
        return len({r.experiment_id for r in self._rows})

    # -------------------------------------------------------------- the pre-launch question
    def already_tested(self, question: str, design: DesignSpec | None, now, allow_repeat_reason: str = "") -> DuplicateVerdict:
        """'We already tested this.' Checks, in order of severity: in flight, exact repeat (same fingerprint AND same code
        and data), near-duplicate configuration, same question already answered, known failure nearby. A genuine change of
        code or data hash on an otherwise identical design downgrades an exact repeat to RETEST_JUSTIFIED."""
        recs = list(self.view(now).values())
        if not recs and (self.bridge is None or design is None or not design.config):
            return DuplicateVerdict(DuplicateStatus.NOVEL, "nothing comparable has been run")
        cfgs = [dict(r.experiment.config) for r in recs] + ([dict(design.config)] if design else [])
        space = infer_space(cfgs) if cfgs else None
        fp = design.fingerprint() if design else None
        matches, reasons = [], []
        worst = DuplicateStatus.NOVEL
        rank = {DuplicateStatus.NOVEL: 0, DuplicateStatus.RETEST_JUSTIFIED: 1, DuplicateStatus.KNOWN_FAILURE_NEARBY: 2,
                DuplicateStatus.SAME_QUESTION_ANSWERED: 3, DuplicateStatus.NEAR_DUPLICATE: 4,
                DuplicateStatus.SAME_CONFIG_REPEAT: 5, DuplicateStatus.IN_FLIGHT: 6}
        for r in recs:
            qs = question_similarity(question, r.question)
            dist = config_distance(design.config, r.experiment.config, space) if design and design.config else 1.0
            same_fp = fp is not None and r.experiment.fingerprint() == fp
            if qs < self.question_match and not same_fp and dist > self.near:
                continue
            kind = r.result.kind if r.result else ""
            matches.append(DuplicateMatch(r.experiment_id, r.version, r.status, round(qs, 4), round(dist, 4), same_fp, kind,
                                          r.recorded_at, r.learned))
            st = DuplicateStatus.NOVEL
            if r.status in (ExperimentStatus.PROPOSED, ExperimentStatus.RUNNING) and (qs >= self.question_match or same_fp):
                st = DuplicateStatus.IN_FLIGHT
            elif r.status in (ExperimentStatus.INVALID, ExperimentStatus.FAILED, ExperimentStatus.ABANDONED):
                continue                                          # a defective / crashed attempt answered nothing
            elif same_fp and design:
                changed = []
                if design.code_hash and r.experiment.code_hash and design.code_hash != r.experiment.code_hash:
                    changed.append("code changed since the earlier run")
                if design.data_hash and r.experiment.data_hash and design.data_hash != r.experiment.data_hash:
                    changed.append("data changed since the earlier run")
                if r.status == ExperimentStatus.INCONCLUSIVE:
                    changed.append("earlier run was inconclusive")
                if changed:
                    st = DuplicateStatus.RETEST_JUSTIFIED
                    reasons += changed
                else:
                    st = DuplicateStatus.SAME_CONFIG_REPEAT
            elif design and dist <= self.near and qs >= self.question_match:
                st = DuplicateStatus.NEAR_DUPLICATE
            elif qs >= self.question_match and r.result and r.result.kind in ResultKind.DEFINITIVE:
                st = DuplicateStatus.SAME_QUESTION_ANSWERED
            elif design and dist <= self.near and r.result and r.result.kind == ResultKind.REFUTED:
                st = DuplicateStatus.KNOWN_FAILURE_NEARBY
            if rank[st] > rank[worst]:
                worst = st
        if self.bridge is not None and design is not None:
            for source, st, blocking, msg in self.bridge.check(design.config):
                reasons.append(msg)
                cand = st if blocking else DuplicateStatus.RETEST_JUSTIFIED if st == DuplicateStatus.NOVEL else worst
                if blocking and rank[cand] > rank[worst]:
                    worst = cand
                elif not blocking and worst == DuplicateStatus.NOVEL:
                    worst = DuplicateStatus.NOVEL
        if allow_repeat_reason.strip() and worst in DuplicateStatus.BLOCKING and worst != DuplicateStatus.IN_FLIGHT:
            reasons.append(f"repeat allowed by caller: {allow_repeat_reason.strip()}")
            worst = DuplicateStatus.RETEST_JUSTIFIED
        if worst == DuplicateStatus.NOVEL:
            return DuplicateVerdict(DuplicateStatus.NOVEL, "comparable experiments exist but none answers this question"
                                    if matches else "nothing comparable has been run", tuple(matches), tuple(reasons))
        top = sorted(matches, key=lambda m: (-m.question_similarity, m.config_distance))[0] if matches else None
        msg = f"we already tested this ({worst})"
        if top:
            msg += f": {top.experiment_id} v{top.version} [{top.status}/{top.result_kind or 'no result'}]"
            if top.learned:
                msg += " learned: " + "; ".join(top.learned[:2])
        return DuplicateVerdict(worst, msg, tuple(matches), tuple(reasons))

    # -------------------------------------------------------------- writing
    def _append(self, rec: ExperimentRecord) -> None:
        self._rows.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(rec.to_json() + "\n")

    def propose(self, rec: ExperimentRecord, now, allow_repeat_reason: str = "") -> ExperimentRecord:
        """Register an experiment before it runs. Refuses malformed records, ids already used, and blocked repeats."""
        errs = rec.validate("proposal")
        if errs:
            raise ValueError("experiment record invalid: " + "; ".join(errs))
        if rec.result is not None:
            raise ValueError("a proposal cannot carry a result")
        if any(r.experiment_id == rec.experiment_id for r in self._rows):
            raise ValueError(f"experiment {rec.experiment_id} already registered")
        verdict = self.already_tested(rec.question, rec.experiment, now, allow_repeat_reason)
        if verdict.blocking:
            raise DuplicateExperiment(verdict)
        stamped = rec.with_(version=1, status=ExperimentStatus.PROPOSED, recorded_at=_iso(now),
                            created_at=rec.created_at or _iso(now),
                            repeat_reason=allow_repeat_reason.strip() or ("; ".join(verdict.retest_reasons)
                                                                          if verdict.status == DuplicateStatus.RETEST_JUSTIFIED else ""))
        self._append(stamped)
        return stamped

    def mark_running(self, experiment_id: str, now) -> ExperimentRecord:
        cur = self._latest(experiment_id)
        if cur.status != ExperimentStatus.PROPOSED:
            raise ValueError(f"{experiment_id} is {cur.status}, cannot start")
        new = cur.with_(version=cur.version + 1, status=ExperimentStatus.RUNNING, recorded_at=_iso(now))
        self._append(new)
        return new

    def record_result(self, experiment_id: str, result: ExperimentResult, now, learned: Sequence[str],
                      not_learned: Sequence[str], next_action: str, posterior_belief: str = "") -> ExperimentRecord:
        """Close an experiment. The belief update is COMPUTED from the record's expected outcomes and the observed
        outcome label. A result dated at/after `now` is a FirewallBreach: an experiment cannot report the future."""
        cur = self._latest(experiment_id)
        if cur.status in ExperimentStatus.TERMINAL:
            raise ValueError(f"{experiment_id} already {cur.status}; write a follow-up experiment instead of editing history")
        if not result.observed_at or to_ts(result.observed_at) > to_ts(now):
            raise FirewallBreach(f"result of {experiment_id} observed_at={result.observed_at!r} is after now={now}")
        if to_ts(result.observed_at) < to_ts(cur.created_at):
            raise FirewallBreach(f"result of {experiment_id} observed before the experiment was created")
        status = (ExperimentStatus.INVALID if result.invalid_reason else
                  ExperimentStatus.INCONCLUSIVE if result.kind == ResultKind.MIXED else ExperimentStatus.ANSWERED)
        bu = None
        if not result.invalid_reason:
            if not result.outcome:
                raise ValueError("result.outcome (the label matching an expected outcome) is required for a valid result")
            bu = make_belief_update(cur, result.outcome, posterior_belief)
        else:
            bu = BeliefUpdate(cur.current_belief, posterior_belief or cur.current_belief,
                              {h.hid: h.prior for h in cur.competing_hypotheses},
                              {h.hid: h.prior for h in cur.competing_hypotheses}, 0.0, 0.0)
        new = cur.with_(version=cur.version + 1, status=status, result=result, belief_update=bu, learned=tuple(learned),
                        not_learned=tuple(not_learned), next_action=next_action, recorded_at=_iso(now))
        errs = new.validate("answered")
        if errs:
            raise ValueError("answer incomplete: " + "; ".join(errs))
        self._append(new)
        if self.bridge is not None:
            self.bridge.mirror(new, now)
        return new

    def fail(self, experiment_id: str, now, reason: str) -> ExperimentRecord:
        cur = self._latest(experiment_id)
        if cur.status in ExperimentStatus.TERMINAL:
            raise ValueError(f"{experiment_id} already {cur.status}")
        new = cur.with_(version=cur.version + 1, status=ExperimentStatus.FAILED, recorded_at=_iso(now),
                        not_learned=(f"nothing: the run failed ({reason})",), next_action="fix and re-propose with repeat_reason")
        self._append(new)
        return new

    def abandon(self, experiment_id: str, now, reason: str) -> ExperimentRecord:
        cur = self._latest(experiment_id)
        if cur.status in ExperimentStatus.TERMINAL:
            raise ValueError(f"{experiment_id} already {cur.status}")
        new = cur.with_(version=cur.version + 1, status=ExperimentStatus.ABANDONED, recorded_at=_iso(now),
                        not_learned=(f"abandoned before running: {reason}",), next_action="none until the reason is removed")
        self._append(new)
        return new

    def _latest(self, experiment_id: str) -> ExperimentRecord:
        h = self.history(experiment_id)
        if not h:
            raise KeyError(f"unknown experiment {experiment_id}")
        return h[-1]

    # -------------------------------------------------------------- knowledge queries (what the memory is FOR)
    def answered(self, now) -> list:
        return [r for r in self.view(now).values() if r.status == ExperimentStatus.ANSWERED]

    def open_questions(self, now) -> list:
        return [r for r in self.view(now).values() if r.status in (ExperimentStatus.PROPOSED, ExperimentStatus.RUNNING)]

    def not_learned_register(self, now) -> dict:
        """Everything experiments explicitly said they did NOT learn, with how many times each was left open. This is the
        raw material for the next question (the research-priority engine reads it)."""
        reg: dict = {}
        for r in self.view(now).values():
            for item in r.not_learned:
                e = reg.setdefault(item.strip().lower(), {"text": item.strip(), "count": 0, "experiments": []})
                e["count"] += 1
                e["experiments"].append(r.experiment_id)
        return dict(sorted(reg.items(), key=lambda kv: -kv[1]["count"]))

    def pending_next_actions(self, now) -> list:
        acts = []
        for r in self.view(now).values():
            if r.status in (ExperimentStatus.ANSWERED, ExperimentStatus.INCONCLUSIVE) and r.next_action.strip():
                followed = any(r.experiment_id in c.parent_ids for c in self.view(now).values())
                if not followed:
                    acts.append({"experiment_id": r.experiment_id, "next_action": r.next_action, "recorded_at": r.recorded_at})
        return sorted(acts, key=lambda a: a["recorded_at"])

    def belief_history(self, question: str, now) -> list:
        """Chronology of belief updates on (near-)identical questions: how the answer moved over time."""
        rows = [r for r in self.view(now).values() if r.belief_update is not None
                and question_similarity(question, r.question) >= self.question_match]
        rows.sort(key=lambda r: r.result.observed_at if r.result else r.recorded_at)
        return [{"experiment_id": r.experiment_id, "observed_at": r.result.observed_at if r.result else "",
                 "belief": r.belief_update.posterior_belief, "moved": r.belief_update.moved, "kind": r.result.kind if r.result else ""}
                for r in rows]

    def contradictory_answers(self, now) -> list:
        """Pairs of answered experiments on the same question with opposite definitive results (CONFIRMED vs REFUTED).
        These are contradictions the research-priority engine must look at, not average away."""
        recs = [r for r in self.answered(now) if r.result and r.result.kind in (ResultKind.CONFIRMED, ResultKind.REFUTED)]
        out = []
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if a.result.kind != b.result.kind and question_similarity(a.question, b.question) >= self.question_match:
                    out.append({"a": a.experiment_id, "b": b.experiment_id, "question": a.question,
                                "windows_a": a.experiment.windows, "windows_b": b.experiment.windows})
        return out

    def prediction_calibration(self, now, bins: int = 5) -> dict:
        """Were the stated prediction confidences honest? Compares stated confidence with whether the result CONFIRMED the
        prediction. n is reported; below 10 answered experiments the verdict is 'insufficient'."""
        pairs = [(r.prediction.confidence, 1.0 if r.result.kind == ResultKind.CONFIRMED else 0.0)
                 for r in self.answered(now) if r.result]
        if len(pairs) < 10:
            return {"verdict": "insufficient", "n": len(pairs), "label": ValidationLabel.INSUFFICIENT_EVIDENCE.value}
        conf = [p for p, _ in pairs]
        hit = [h for _, h in pairs]
        brier = sum((p - h) ** 2 for p, h in pairs) / len(pairs)
        base = sum(hit) / len(hit)
        base_brier = sum((base - h) ** 2 for h in hit) / len(hit)
        rows = []
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = [(p, h) for p, h in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
            if sel:
                rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "stated": sum(p for p, _ in sel) / len(sel),
                             "observed": sum(h for _, h in sel) / len(sel)})
        return {"verdict": "computed", "n": len(pairs), "brier": brier, "base_rate_brier": base_brier,
                "mean_stated": sum(conf) / len(conf), "observed_rate": base, "bins": rows,
                "overconfident": sum(conf) / len(conf) > base + 0.1}


# ------------------------------------------------------------------------------------------------ legacy log bridge

_LEGACY_OUTCOME_STATUS = {"adopt": ExperimentStatus.ANSWERED, "reject": ExperimentStatus.ANSWERED,
                          "continue_testing": ExperimentStatus.INCONCLUSIVE}


def legacy_to_record(row: Mapping, index: int) -> ExperimentRecord:
    """One row of state/experiments.jsonl as an ExperimentRecord. Honest: the legacy writer never captured hypotheses,
    predictions or expected outcomes, so those are placeholders explicitly marked NOT RECORDED and listed in
    `legacy_missing`; a placeholder hypothesis pair (the thing helped / it did not) keeps the record structurally valid
    without pretending anyone predicted anything."""
    eid = str(row.get("experiment_id") or f"legacy-{index:05d}-{stable_hash(row, 8)}")
    t = row.get("timestamp") or row.get("t") or "1970-01-01T00:00:00"
    try:
        stamp = _iso(t)
    except (ValueError, FirewallBreach):
        stamp = "1970-01-01T00:00:00"
    q = row.get("question") or row.get("hypothesis") or row.get("what") or row.get("event") or NOT_RECORDED
    cfg = row.get("config") or row.get("cfg") or {k: v for k, v in row.items()
                                                  if isinstance(v, (int, float, str, bool)) and k not in ("t", "timestamp", "event")}
    oc = str(row.get("outcome") or "").lower()
    status = _LEGACY_OUTCOME_STATUS.get(oc, ExperimentStatus.INCONCLUSIVE)
    kind = ResultKind.CONFIRMED if oc == "adopt" else ResultKind.REFUTED if oc == "reject" else ResultKind.MIXED
    metrics = cast(dict, row.get("metrics") if isinstance(row.get("metrics"), dict) else {})
    result = ExperimentResult(kind=kind, outcome=oc or "unknown",
                              metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)},
                              observed_at=stamp, summary=str(row.get("reason") or NOT_RECORDED))
    hyps = (Hypothesis("h_effect", NOT_RECORDED + ": the change helps", 0.5), Hypothesis("h_null", NOT_RECORDED + ": no effect", 0.5))
    exp = (ExpectedOutcome("h_effect", "adopt", 1.0), ExpectedOutcome("h_null", "reject", 1.0))
    missing = ["competing_hypotheses", "prediction", "expected_outcomes", "current_belief", "not_learned", "next_action", "belief_update"]
    if q == NOT_RECORDED:
        missing.append("question")
    if not row.get("reason"):
        missing.append("learned")
    return ExperimentRecord(
        experiment_id=eid, version=1, status=status, question=str(q), current_belief=NOT_RECORDED, competing_hypotheses=hyps,
        prediction=Prediction(NOT_RECORDED), experiment=DesignSpec(config=dict(cfg), seed=row.get("seed") if isinstance(row.get("seed"), int) else 0,
                                                                    data_hash=str(row.get("data_snapshot") or ""),
                                                                    code_hash=str(row.get("code_hash") or row.get("git_commit") or "")),
        expected_outcomes=exp, created_at=stamp, recorded_at=stamp, result=result,
        belief_update=None, learned=(str(row["reason"]),) if row.get("reason") else (), not_learned=(NOT_RECORDED,),
        next_action=NOT_RECORDED, tags=("legacy",), legacy=True, legacy_missing=tuple(missing))


def import_legacy(ledger: ExperimentLedger, rows: Iterable[Mapping]) -> dict:
    """Load legacy rows into a ledger (in memory and, if the ledger has a path, on disk). Rows with an id already present
    are skipped. Returns counts and the mean completeness, the honest measure of how much of the old log is usable memory."""
    have = {r.experiment_id for r in ledger._rows}
    added = skipped = 0
    comp = []
    for i, row in enumerate(rows):
        rec = legacy_to_record(row, i)
        if rec.experiment_id in have:
            skipped += 1
            continue
        have.add(rec.experiment_id)
        ledger._append(rec)
        comp.append(rec.completeness())
        added += 1
    return {"added": added, "skipped": skipped, "mean_completeness": (sum(comp) / len(comp)) if comp else 0.0,
            "label": LABEL}


def sync_to_tried_index(ledger: ExperimentLedger, tried_index, now) -> dict:
    """Push answered experiments with a config into engine.experiment_memory.TriedIndex so the old pre-launch check also
    knows them. A REFUTED result maps to 'reject' with its first learned line as the reason (the index refuses reasonless rejects)."""
    added = skipped = 0
    have = {r["experiment_id"] for r in tried_index.rows}
    for r in ledger.view(now).values():
        if r.experiment_id in have or not r.experiment.config or r.result is None or r.status not in (
                ExperimentStatus.ANSWERED, ExperimentStatus.INCONCLUSIVE):
            skipped += 1
            continue
        outcome = {"CONFIRMED": "adopt", "REFUTED": "reject"}.get(r.result.kind, "continue_testing")
        reason = (r.learned[0] if r.learned else r.result.summary) or "refuted"
        score = r.result.metrics.get("score") if isinstance(r.result.metrics, Mapping) else None
        tried_index.add(r.experiment_id, dict(r.experiment.config), outcome, score if isinstance(score, (int, float)) else None,
                        reason if outcome == "reject" else None, now=now)
        added += 1
    return {"added": added, "skipped": skipped}


# ------------------------------------------------------------------------------------------------ builders & reports

def new_record(experiment_id: str, question: str, current_belief: str, hypotheses: Sequence[Hypothesis],
               prediction: Prediction, design: DesignSpec, expected: Sequence[ExpectedOutcome], now,
               tags: Sequence[str] = (), parent_ids: Sequence[str] = ()) -> ExperimentRecord:
    """Convenience constructor: stamps the code hash into the design when the caller left it empty."""
    if not design.code_hash:
        design = dataclasses.replace(design, code_hash=current_code_hash())
    return ExperimentRecord(experiment_id=experiment_id, version=0, status=ExperimentStatus.PROPOSED, question=question,
                            current_belief=current_belief, competing_hypotheses=tuple(hypotheses), prediction=prediction,
                            experiment=design, expected_outcomes=tuple(expected), created_at=_iso(now), recorded_at=_iso(now),
                            tags=tuple(tags), parent_ids=tuple(parent_ids))


def uniform_expected(hyps: Sequence[Hypothesis], outcomes: Sequence[str], favoured: Mapping[str, str], strength: float = 0.8) -> tuple:
    """Expected outcomes where each hypothesis puts `strength` on its own favoured outcome and spreads the rest evenly.
    A lazy-but-valid default for callers who have a discriminating outcome per hypothesis and no finer model."""
    outs = []
    for h in hyps:
        fav = favoured.get(h.hid)
        others = [o for o in outcomes if o != fav]
        for o in outcomes:
            if fav is None:
                p = 1.0 / len(outcomes)
            elif o == fav:
                p = strength if others else 1.0
            else:
                p = (1 - strength) / len(others)
            outs.append(ExpectedOutcome(h.hid, o, p))
    return tuple(outs)


def coverage_report(records: Iterable[ExperimentRecord]) -> dict:
    """Per section-31 field: how many records genuinely carry it. The legacy log fails most of them, by design of the report."""
    recs = list(records)
    per = {f: 0 for f in SECTION_31_FIELDS}
    for r in recs:
        for f in SECTION_31_FIELDS:
            v = getattr(r, f)
            if v not in (None, (), "", NOT_RECORDED) and f not in r.legacy_missing:
                per[f] += 1
    n = len(recs)
    return {"n": n, "per_field": per, "share": {f: (c / n if n else 0.0) for f, c in per.items()},
            "legacy": sum(1 for r in recs if r.legacy), "mean_completeness": (sum(r.completeness() for r in recs) / n) if n else 0.0}


def render_report(ledger: ExperimentLedger, now, limit: int = 20) -> str:
    """Plain-text state of the memory at `now`: counts, open questions, not-learned register, contradictions, calibration."""
    v = list(ledger.view(now).values())
    by_status: dict = {}
    for r in v:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    lines = [f"Experiment memory as of {_iso(now)}   [{LABEL}]", f"experiments: {len(v)}  by status: {by_status}",
             f"unparseable ledger lines: {ledger.unparseable}"]
    cov = coverage_report(v)
    lines.append(f"mean section-31 completeness: {cov['mean_completeness']:.2f}  legacy rows: {cov['legacy']}")
    oq = ledger.open_questions(now)
    lines.append(f"open questions ({len(oq)}):")
    lines += [f"  - {r.experiment_id}: {r.question}" for r in oq[:limit]]
    nl = ledger.not_learned_register(now)
    lines.append(f"most repeated 'not learned' ({len(nl)} distinct):")
    lines += [f"  - x{e['count']}: {e['text']}" for e in list(nl.values())[:limit] if e["text"] != NOT_RECORDED]
    ca = ledger.contradictory_answers(now)
    lines.append(f"contradictory answers: {len(ca)}")
    lines += [f"  - {c['a']} vs {c['b']}: {c['question']}" for c in ca[:limit]]
    cal = ledger.prediction_calibration(now)
    lines.append(f"prediction calibration: {cal['verdict']} (n={cal['n']})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ lineage and staleness

def lineage(ledger: ExperimentLedger, experiment_id: str, now) -> dict:
    """Ancestors (via parent_ids) and descendants of an experiment, as visible at `now`. Follow-up chains are how a line of
    inquiry is read: what was asked, what it led to. A parent id that does not exist is reported, not ignored."""
    view = ledger.view(now)
    if experiment_id not in view:
        raise KeyError(f"{experiment_id} not visible at {now}")
    anc, missing, stack = [], [], list(view[experiment_id].parent_ids)
    while stack:
        pid = stack.pop()
        if pid in anc:
            continue
        if pid not in view:
            missing.append(pid)
            continue
        anc.append(pid)
        stack.extend(view[pid].parent_ids)
    desc, frontier = [], [experiment_id]
    while frontier:
        cur = frontier.pop()
        for r in view.values():
            if cur in r.parent_ids and r.experiment_id not in desc:
                desc.append(r.experiment_id)
                frontier.append(r.experiment_id)
    return {"ancestors": sorted(anc), "descendants": sorted(desc), "missing_parents": sorted(set(missing))}


def stale_answers(ledger: ExperimentLedger, now, current_code_hash_value: str, current_data_hash: str = "") -> list:
    """Answered experiments whose code hash (or data hash, when given) differs from the current one. Their answers describe
    a system that no longer exists: not wrong, but not evidence about the present. These are the legitimate RETEST candidates."""
    out = []
    for r in ledger.answered(now):
        why = []
        if r.experiment.code_hash and current_code_hash_value and r.experiment.code_hash != current_code_hash_value:
            why.append("code changed")
        if r.experiment.data_hash and current_data_hash and r.experiment.data_hash != current_data_hash:
            why.append("data changed")
        if why:
            out.append({"experiment_id": r.experiment_id, "question": r.question, "reasons": why, "answered_at": r.recorded_at})
    return out


def unanswered_hypotheses(ledger: ExperimentLedger, now, threshold: float = 0.25) -> list:
    """Hypotheses that still hold >= threshold posterior mass after an experiment that did not settle them: the open
    questions inside answered experiments (an ANSWERED record can still leave a live competitor)."""
    out = []
    for r in ledger.answered(now):
        if r.belief_update is None:
            continue
        for h in r.competing_hypotheses:
            p = r.belief_update.posterior.get(h.hid, 0.0)
            if p >= threshold and h.kind != "null":
                out.append({"experiment_id": r.experiment_id, "hypothesis": h.statement, "posterior": p})
    return sorted(out, key=lambda x: -x["posterior"])
