"""Research experiment memory (C66 section 15; extends C62 section 31; Bible phase 30; canon C58-C66).

C62's `engine.learning.experiment_memory` is the ONE ledger (question, belief, hypotheses, prediction, expected outcomes, result,
belief update, learned / not learned, next action, plus the legacy bridge to engine.registry and engine.experiment_memory). C66
section 15 asks for twenty-four fields and four pre-launch questions. This module ADDS what the ledger lacks, without a second
ledger: a sidecar of `DesignExtension` (dataset, universe, regime, time period, information cutoff, features, representation,
algorithm, hyperparameters, configuration hash, complexity, expected value and decision impact) written when an experiment is
proposed, and `OutcomeExtension` (failure mode, transfer result, memorisation result, realised value and impact, compute cost)
written when it closes. `ResearchExperimentMemory` composes ledger + sidecar and is the front door.

Before any launch, `launch_gate` answers section 15's questions in order: HAVE WE DONE THIS (question, fingerprint, design
overlap, both legacy stores), DID IT ANSWER (graded SOLID / WEAK / STALE / UNANSWERED, never assumed), IS A REPEAT JUSTIFIED
(each claimed reason is VERIFIED against the actual difference in the designs, so a false "fresh period" claim is rejected),
and WHAT DIFFERS (field-level diff with a materiality score). Section 32 lives in `replication_status`: one lucky experiment
cannot satisfy it. Everything time-aware takes `now`; an information cutoff at or after `now`, or a training period that ends
after its cutoff, is a FirewallBreach. Matured results leave this module only through `matured_record` (C66 section 29-31).

IMPLEMENTED - NOT VALIDATED. Public entry points: `ResearchExperimentMemory.launch_gate/register/close`, and `step` (the
research loop's maintenance sweep)."""
from __future__ import annotations

import dataclasses
import enum
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from engine import repro
from engine.learning import experiment_memory as em
from engine.learning.core import (DecisionEffect, FailureCause, FirewallBreach, Provenance, ValidationLabel, as_date,
                                  canonical_json, current_code_hash, stable_hash)
from engine.learning.experiment_memory import (DuplicateStatus, ExperimentLedger, ExperimentRecord, ExperimentResult,
                                               ExperimentStatus, ResultKind, question_similarity, to_ts)
from engine.research.core import ExperimentValue, GateVerdict, MaturedRecord

LABEL = ValidationLabel.NOT_VALIDATED.value
MISSING = "NOT RECORDED"

SECTION15_FIELDS = ("question", "hypothesis", "counter_hypotheses", "dataset", "time_period", "information_cutoff", "features",
                    "representation", "algorithm", "hyperparameters", "random_seed", "code_hash", "data_hash",
                    "configuration_hash", "controls", "result", "confidence_interval", "failure_mode", "transfer_result",
                    "memorization_result", "complexity", "compute_cost", "research_value", "decision_impact")
COUNTER_KINDS = ("null", "noise", "measurement", "mechanism")     # a hypothesis that is not simply 'the idea works'


class TestState(str, enum.Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_TESTED = "NOT_TESTED"            # unknown stays unknown: never read as PASSED
    NOT_APPLICABLE = "NOT_APPLICABLE"

    def __str__(self):
        return self.value


class ReplicationReason(str, enum.Enum):
    FRESH_PERIOD = "FRESH_PERIOD"
    FRESH_STOCKS = "FRESH_STOCKS"
    FRESH_REGIME = "FRESH_REGIME"
    FRESH_SEED = "FRESH_SEED"
    CONTROL_COMPARISON = "CONTROL_COMPARISON"
    INDEPENDENT_METHOD = "INDEPENDENT_METHOD"
    CODE_CHANGED = "CODE_CHANGED"
    DATA_CHANGED = "DATA_CHANGED"
    PRIOR_WEAK = "PRIOR_WEAK"
    CONTRADICTION = "CONTRADICTION"

    def __str__(self):
        return self.value


class LaunchDecision(str, enum.Enum):
    LAUNCH_NOVEL = "LAUNCH_NOVEL"
    LAUNCH_REPLICATION = "LAUNCH_REPLICATION"
    LAUNCH_AFTER_EXPLANATION = "LAUNCH_AFTER_EXPLANATION"
    REFUSE_ALREADY_ANSWERED = "REFUSE_ALREADY_ANSWERED"
    REFUSE_SAME_DESIGN = "REFUSE_SAME_DESIGN"
    WAIT_IN_FLIGHT = "WAIT_IN_FLIGHT"
    NEEDS_REASON = "NEEDS_REASON"
    NEEDS_EXPLANATION = "NEEDS_EXPLANATION"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"

    def __str__(self):
        return self.value

    @property
    def permits_launch(self) -> bool:
        return self in (LaunchDecision.LAUNCH_NOVEL, LaunchDecision.LAUNCH_REPLICATION, LaunchDecision.LAUNCH_AFTER_EXPLANATION)


class AnswerGrade(str, enum.Enum):
    SOLID = "SOLID"                      # a definitive result with an interval, enough data and the checks that matter
    WEAK = "WEAK"                        # definitive on paper but a check is missing (interval, transfer, memorisation, n)
    STALE = "STALE"                      # answered, but by code or data that no longer exist
    UNANSWERED = "UNANSWERED"            # inconclusive, invalid, failed or abandoned: it says nothing about the question

    def __str__(self):
        return self.value


class LaunchRefused(Exception):
    def __init__(self, assessment: "LaunchAssessment"):
        super().__init__(assessment.message)
        self.assessment = assessment


# ------------------------------------------------------------------------------------------------ value types

@dataclass(frozen=True)
class TimePeriod:
    """The data span an experiment learns from, as ISO dates (inclusive)."""
    start: str
    end: str

    def check(self) -> list:
        try:
            if as_date(self.start) > as_date(self.end):
                return [f"time_period: start {self.start} after end {self.end}"]
        except ValueError:
            return [f"time_period: unparseable {self.start!r}..{self.end!r}"]
        return []

    def overlaps(self, other: "TimePeriod") -> bool:
        return as_date(self.start) <= as_date(other.end) and as_date(other.start) <= as_date(self.end)

    def days(self) -> int:
        return (as_date(self.end) - as_date(self.start)).days + 1


@dataclass(frozen=True)
class Complexity:
    """How much freedom the experiment had to fit noise: parameters, features, variants tried on the way, observations."""
    n_parameters: int = 0
    n_features: int = 0
    n_free_choices: int = 1              # forking paths: how many variants were looked at before this one was reported
    n_observations: int = 0

    def check(self) -> list:
        errs = []
        for name in ("n_parameters", "n_features", "n_observations"):
            if getattr(self, name) < 0:
                errs.append(f"complexity.{name} negative")
        if self.n_free_choices < 1:
            errs.append("complexity.n_free_choices must be >= 1 (the reported variant is one choice)")
        return errs

    def observations_per_parameter(self) -> float | None:
        return None if self.n_parameters <= 0 or self.n_observations <= 0 else self.n_observations / self.n_parameters

    def overfit_pressure(self) -> float:
        """0..1 rule-of-thumb pressure: few observations per parameter and many free choices raise it. A heuristic to sort
        experiments by scrutiny, not a test statistic."""
        opp = self.observations_per_parameter()
        p_param = 0.5 if opp is None else min(1.0, 10.0 / max(opp, 1e-9))
        p_choice = min(1.0, math.log(self.n_free_choices) / math.log(200.0)) if self.n_free_choices > 1 else 0.0
        return float(min(1.0, 0.6 * p_param + 0.4 * p_choice))


@dataclass(frozen=True)
class DecisionImpact:
    effects: tuple = (DecisionEffect.NONE,)
    expected_size: float | None = None   # in the metric the effect is measured in; None = not estimated
    reaches_production: bool = False
    note: str = ""

    def check(self) -> list:
        errs = []
        if not self.effects:
            errs.append("decision_impact: no effects (use DecisionEffect.NONE for research-only knowledge)")
        if DecisionEffect.NONE in self.effects and len(self.effects) > 1:
            errs.append("decision_impact: NONE cannot be combined with other effects")
        if self.reaches_production and DecisionEffect.NONE in self.effects:
            errs.append("decision_impact: research-only knowledge (NONE) cannot reach production")
        if self.expected_size is not None and not math.isfinite(self.expected_size):
            errs.append("decision_impact: expected_size not finite")
        return errs

    @property
    def is_null(self) -> bool:
        return DecisionEffect.NONE in self.effects


@dataclass(frozen=True)
class TransferResult:
    state: TestState = TestState.NOT_TESTED
    effect: float | None = None
    ci: tuple = ()
    n_periods: int = 0
    note: str = ""

    def check(self) -> list:
        errs = []
        if self.state in (TestState.PASSED, TestState.FAILED):
            if not self.ci or len(self.ci) != 2 or self.ci[0] > self.ci[1]:
                errs.append("transfer_result: a tested transfer needs a (lo, hi) interval")
            if self.n_periods < 1:
                errs.append("transfer_result: a tested transfer needs n_periods >= 1")
        if self.state == TestState.PASSED and self.ci and self.ci[0] <= 0:
            errs.append("transfer_result: PASSED but the interval includes zero")
        return errs


@dataclass(frozen=True)
class MemorizationResult:
    state: TestState = TestState.NOT_TESTED     # FAILED = the result is explained by memorising the tested data
    same_period_effect: float | None = None
    fresh_period_effect: float | None = None
    control: str = ""                            # e.g. disguised rerun, label shuffle
    note: str = ""

    def check(self) -> list:
        errs = []
        if self.state in (TestState.PASSED, TestState.FAILED) and not self.control:
            errs.append("memorization_result: a tested memorisation check must name its control")
        if self.state == TestState.PASSED and None in (self.same_period_effect, self.fresh_period_effect):
            errs.append("memorization_result: PASSED needs both same-period and fresh-period effects")
        return errs

    def gap(self) -> float | None:
        if self.same_period_effect is None or self.fresh_period_effect is None:
            return None
        return self.same_period_effect - self.fresh_period_effect


@dataclass(frozen=True)
class DesignExtension:
    """Section-15 fields the C62 ledger record does not carry (question, hypotheses, seed, controls, code/data hash,
    windows and cost already live in ExperimentRecord/DesignSpec)."""
    dataset: str
    time_period: TimePeriod
    information_cutoff: str
    features: tuple
    representation: str
    algorithm: str
    hyperparameters: Mapping = field(default_factory=dict)
    universe: str = ""                   # label or hash of the stock set
    regime: str = ""
    complexity: Complexity = Complexity()
    expected_value: ExperimentValue = ExperimentValue()
    expected_impact: DecisionImpact = DecisionImpact()

    def check(self, now=None) -> list:
        errs = []
        for name in ("dataset", "representation", "algorithm"):
            if not str(getattr(self, name)).strip():
                errs.append(f"design.{name} empty")
        if not self.features:
            errs.append("design.features empty (state 'none' explicitly if the method uses no features)")
        errs += self.time_period.check() + self.complexity.check() + self.expected_impact.check()
        try:
            cutoff = as_date(self.information_cutoff)
            if as_date(self.time_period.end) > cutoff:
                errs.append(f"design: time_period ends {self.time_period.end}, after information_cutoff {self.information_cutoff}")
            if now is not None and cutoff >= as_date(now):
                errs.append(f"design: information_cutoff {self.information_cutoff} is not strictly before now={now}")
        except ValueError:
            errs.append(f"design: unparseable information_cutoff {self.information_cutoff!r}")
        for k, v in self.hyperparameters.items():
            if isinstance(v, float) and not math.isfinite(v):
                errs.append(f"design.hyperparameters[{k}] not finite")
        return errs

    @property
    def configuration_hash(self) -> str:
        """Hash of everything that defines WHAT was run (not when it was proposed): two runs with equal hashes are one design."""
        return repro.config_hash({"dataset": self.dataset, "universe": self.universe, "regime": self.regime,
                                  "features": sorted(self.features), "representation": self.representation,
                                  "algorithm": self.algorithm, "hyperparameters": dict(self.hyperparameters),
                                  "period": [self.time_period.start, self.time_period.end], "cutoff": self.information_cutoff})

    @classmethod
    def from_dict(cls, d: Mapping) -> "DesignExtension":
        ei = d.get("expected_impact") or {}
        return cls(dataset=d["dataset"], time_period=TimePeriod(**d["time_period"]), information_cutoff=d["information_cutoff"],
                   features=tuple(d["features"]), representation=d["representation"], algorithm=d["algorithm"],
                   hyperparameters=dict(d.get("hyperparameters") or {}), universe=d.get("universe", ""), regime=d.get("regime", ""),
                   complexity=Complexity(**(d.get("complexity") or {})),
                   expected_value=ExperimentValue(**(d.get("expected_value") or {})),
                   expected_impact=DecisionImpact(effects=tuple(DecisionEffect(e) for e in ei.get("effects", ["NONE"])),
                                                  expected_size=ei.get("expected_size"),
                                                  reaches_production=bool(ei.get("reaches_production", False)), note=ei.get("note", "")))


@dataclass(frozen=True)
class OutcomeExtension:
    failure_mode: FailureCause | None      # None = the experiment did not fail; a REFUTED result MUST name a cause (UNKNOWN allowed)
    transfer: TransferResult = TransferResult()
    memorization: MemorizationResult = MemorizationResult()
    realised_value: ExperimentValue = ExperimentValue()
    realised_impact: DecisionImpact = DecisionImpact()
    compute_cost_minutes: float = 0.0
    failure_note: str = ""

    def check(self, result: ExperimentResult | None = None) -> list:
        errs = self.transfer.check() + self.memorization.check() + self.realised_impact.check()
        if self.compute_cost_minutes < 0 or not math.isfinite(self.compute_cost_minutes):
            errs.append("outcome.compute_cost_minutes must be a finite non-negative number")
        if result is not None:
            if result.kind == ResultKind.REFUTED and self.failure_mode is None:
                errs.append("outcome: a REFUTED result must name a failure mode (FailureCause.UNKNOWN is a legitimate answer)")
            if result.kind == ResultKind.CONFIRMED and self.failure_mode not in (None,):
                errs.append("outcome: a CONFIRMED result cannot carry a failure mode")
            if result.kind in (ResultKind.CONFIRMED, ResultKind.REFUTED) and not result.ci:
                errs.append("outcome: a definitive result needs a confidence interval (result.ci)")
        return errs

    @classmethod
    def from_dict(cls, d: Mapping) -> "OutcomeExtension":
        ri = d.get("realised_impact") or {}
        tr, mz = d.get("transfer") or {}, d.get("memorization") or {}
        return cls(failure_mode=FailureCause(d["failure_mode"]) if d.get("failure_mode") else None,
                   transfer=TransferResult(TestState(tr.get("state", "NOT_TESTED")), tr.get("effect"), tuple(tr.get("ci") or ()),
                                           int(tr.get("n_periods", 0)), tr.get("note", "")),
                   memorization=MemorizationResult(TestState(mz.get("state", "NOT_TESTED")), mz.get("same_period_effect"),
                                                   mz.get("fresh_period_effect"), mz.get("control", ""), mz.get("note", "")),
                   realised_value=ExperimentValue(**(d.get("realised_value") or {})),
                   realised_impact=DecisionImpact(effects=tuple(DecisionEffect(e) for e in ri.get("effects", ["NONE"])),
                                                  expected_size=ri.get("expected_size"),
                                                  reaches_production=bool(ri.get("reaches_production", False)), note=ri.get("note", "")),
                   compute_cost_minutes=float(d.get("compute_cost_minutes", 0.0)), failure_note=d.get("failure_note", ""))


# ------------------------------------------------------------------------------------------------ sidecar store

class ExtensionStore:
    """Append-only jsonl of design and outcome extensions, keyed by experiment id. Like the ledger, reads are as of a time."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._rows: list = []
        self.unparseable = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row["kind"] not in ("design", "outcome"):
                        raise ValueError(row["kind"])
                    to_ts(row["recorded_at"])
                    self._rows.append(row)
                except (ValueError, KeyError, TypeError):
                    self.unparseable += 1

    def _append(self, experiment_id: str, kind: str, payload: Any, now) -> None:
        row = {"experiment_id": experiment_id, "kind": kind, "recorded_at": em._iso(now),
               "payload": json.loads(canonical_json(payload))}
        self._rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def put_design(self, experiment_id: str, ext: DesignExtension, now) -> None:
        if self._latest(experiment_id, "design", None) is not None:
            raise ValueError(f"design extension for {experiment_id} already written; history is immutable")
        self._append(experiment_id, "design", ext, now)

    def put_outcome(self, experiment_id: str, ext: OutcomeExtension, now) -> None:
        if self._latest(experiment_id, "outcome", None) is not None:
            raise ValueError(f"outcome extension for {experiment_id} already written; write a follow-up experiment")
        self._append(experiment_id, "outcome", ext, now)

    def _latest(self, experiment_id: str, kind: str, now) -> Mapping | None:
        cut = None if now is None else to_ts(now)
        best = None
        for r in self._rows:
            if r["experiment_id"] == experiment_id and r["kind"] == kind and (cut is None or to_ts(r["recorded_at"]) < cut):
                best = r
        return best

    def design(self, experiment_id: str, now) -> DesignExtension | None:
        r = self._latest(experiment_id, "design", now)
        return None if r is None else DesignExtension.from_dict(r["payload"])

    def outcome(self, experiment_id: str, now) -> OutcomeExtension | None:
        r = self._latest(experiment_id, "outcome", now)
        return None if r is None else OutcomeExtension.from_dict(r["payload"])

    def __len__(self):
        return len({r["experiment_id"] for r in self._rows})


# ------------------------------------------------------------------------------------------------ design differences

# How much each field's difference counts toward "this is a different experiment". Seed is nearly nothing: a new seed on
# the same design is a re-draw, not a new experiment.
MATERIALITY = {"dataset": 1.0, "algorithm": 1.0, "features": 1.0, "representation": 0.8, "period": 0.8, "universe": 0.6,
               "regime": 0.6, "hyperparameters": 0.5, "controls": 0.5, "cutoff": 0.3, "seed": 0.1}
MATERIAL_AT = 0.30                     # below this the two designs are the same experiment
METHOD_FIELDS = ("algorithm", "features", "representation")


@dataclass(frozen=True)
class DesignDiff:
    changed: Mapping                   # field -> (old, new)
    materiality: float                 # 0 identical .. 1 everything material differs
    unknown: tuple = ()                # fields that could not be compared because the prior had no extension

    @property
    def is_same_design(self) -> bool:
        return self.materiality < MATERIAL_AT

    def method_changed(self) -> bool:
        return sum(MATERIALITY[f] for f in METHOD_FIELDS if f in self.changed) >= 1.0

    def lines(self) -> list:
        out = [f"{k}: {_short(o)} -> {_short(n)}" for k, (o, n) in self.changed.items()]
        out += [f"{k}: prior did not record it (cannot compare)" for k in self.unknown]
        return out or ["no difference in any recorded design field"]


def _short(v, n: int = 60) -> str:
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def hyperparameter_distance(a: Mapping, b: Mapping) -> float:
    """0..1 relative difference over the union of parameters (reuses the ledger's inferred-space distance)."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return em.config_distance(dict(a), dict(b))


def design_diff(new_rec: ExperimentRecord, new_ext: DesignExtension, old_rec: ExperimentRecord, old_ext: DesignExtension | None) -> DesignDiff:
    changed: dict = {}
    unknown: list = []
    score = 0.0
    if new_rec.experiment.seed != old_rec.experiment.seed:
        changed["seed"] = (old_rec.experiment.seed, new_rec.experiment.seed)
        score += MATERIALITY["seed"]
    if set(new_rec.experiment.controls) != set(old_rec.experiment.controls):
        changed["controls"] = (sorted(old_rec.experiment.controls), sorted(new_rec.experiment.controls))
        if set(new_rec.experiment.controls) - set(old_rec.experiment.controls):
            score += MATERIALITY["controls"]
    if old_ext is None:
        unknown = ["dataset", "algorithm", "features", "representation", "period", "universe", "regime", "hyperparameters", "cutoff"]
        d = em.config_distance(new_rec.experiment.config, old_rec.experiment.config) if new_rec.experiment.config and old_rec.experiment.config else 1.0
        if d > 0:
            changed["hyperparameters"] = (dict(old_rec.experiment.config), dict(new_rec.experiment.config))
            score += MATERIALITY["hyperparameters"] * min(1.0, d * 4)
    else:
        for f in ("dataset", "algorithm", "representation", "universe", "regime"):
            a, b = getattr(old_ext, f), getattr(new_ext, f)
            if a != b:
                changed[f] = (a, b)
                score += MATERIALITY[f]
        if set(old_ext.features) != set(new_ext.features):
            changed["features"] = (sorted(old_ext.features), sorted(new_ext.features))
            j = len(set(old_ext.features) & set(new_ext.features)) / max(1, len(set(old_ext.features) | set(new_ext.features)))
            score += MATERIALITY["features"] * (1.0 - j)
        if old_ext.time_period != new_ext.time_period:
            changed["period"] = ((old_ext.time_period.start, old_ext.time_period.end), (new_ext.time_period.start, new_ext.time_period.end))
            score += MATERIALITY["period"] * (0.4 if new_ext.time_period.overlaps(old_ext.time_period) else 1.0)
        if old_ext.information_cutoff != new_ext.information_cutoff:
            changed["cutoff"] = (old_ext.information_cutoff, new_ext.information_cutoff)
            score += MATERIALITY["cutoff"]
        hd = hyperparameter_distance(old_ext.hyperparameters, new_ext.hyperparameters)
        if hd > 0.02:
            changed["hyperparameters"] = (dict(old_ext.hyperparameters), dict(new_ext.hyperparameters))
            score += MATERIALITY["hyperparameters"] * min(1.0, hd * 3)
    total = sum(MATERIALITY.values())
    return DesignDiff(changed, float(min(1.0, score / total * 2.2)), tuple(unknown))


# ------------------------------------------------------------------------------------------------ answer grading

@dataclass(frozen=True)
class GradePolicy:
    """Thresholds for 'did the earlier experiment answer the question'. Defaults are documented starting points; a caller with
    learned values passes its own (section 32: thresholds are learned, not arbitrarily optimised)."""
    min_n: int = 30
    require_ci: bool = True
    require_transfer_for_confirmed: bool = True
    require_memorization_check_for_confirmed: bool = True


@dataclass(frozen=True)
class GradedAnswer:
    grade: AnswerGrade
    reasons: tuple


def grade_answer(rec: ExperimentRecord, outcome: OutcomeExtension | None, policy: GradePolicy = GradePolicy(),
                 current_code: str = "", current_data: str = "") -> GradedAnswer:
    """Grade what an earlier experiment actually settled. NULL results need their stated power (the ledger already enforces
    it); a CONFIRMED result that never checked transfer or memorisation has not answered 'is it real'."""
    if rec.status != ExperimentStatus.ANSWERED or rec.result is None or rec.result.kind not in ResultKind.DEFINITIVE:
        return GradedAnswer(AnswerGrade.UNANSWERED, (f"status {rec.status}, result {rec.result.kind if rec.result else 'none'}",))
    weak = []
    r = rec.result
    if policy.require_ci and not r.ci:
        weak.append("no confidence interval")
    if r.n < policy.min_n:
        weak.append(f"n={r.n} below {policy.min_n}")
    if rec.legacy:
        weak.append("legacy record: hypotheses, prediction and expected outcomes were never recorded")
    if outcome is None:
        weak.append("outcome extension missing (transfer, memorisation, failure mode unknown)")
    elif r.kind == ResultKind.CONFIRMED:
        if policy.require_transfer_for_confirmed and outcome.transfer.state == TestState.NOT_TESTED:
            weak.append("confirmed but transfer never tested")
        if outcome.transfer.state == TestState.FAILED:
            weak.append("confirmed in-sample; transfer FAILED")
        if policy.require_memorization_check_for_confirmed and outcome.memorization.state == TestState.NOT_TESTED:
            weak.append("confirmed but memorisation never checked")
        if outcome.memorization.state == TestState.FAILED:
            weak.append("confirmed result is explained by memorisation")
    if weak:
        return GradedAnswer(AnswerGrade.WEAK, tuple(weak))
    stale = []
    if current_code and rec.experiment.code_hash and rec.experiment.code_hash != current_code:
        stale.append("code changed since the run")
    if current_data and rec.experiment.data_hash and rec.experiment.data_hash != current_data:
        stale.append("data changed since the run")
    if stale:
        return GradedAnswer(AnswerGrade.STALE, tuple(stale))
    return GradedAnswer(AnswerGrade.SOLID, ())


# ------------------------------------------------------------------------------------------------ the pre-launch gate

@dataclass(frozen=True)
class PriorMatch:
    experiment_id: str
    status: str
    result_kind: str
    question_similarity: float
    grade: AnswerGrade
    grade_reasons: tuple
    diff: DesignDiff
    learned: tuple = ()


@dataclass(frozen=True)
class LaunchAssessment:
    decision: LaunchDecision
    message: str
    have_we_done_this: bool
    priors: tuple = ()                         # PriorMatch, most similar first
    did_it_answer: bool = False                # some prior is SOLID
    verified_reasons: tuple = ()               # ReplicationReason claims that the designs actually support
    rejected_reasons: tuple = ()               # (reason, why not)
    available_reasons: tuple = ()              # reasons that hold but were not claimed
    differences: tuple = ()
    ledger_status: str = DuplicateStatus.NOVEL
    problems: tuple = ()

    def render(self) -> str:
        lines = [f"[{self.decision}] {self.message}", f"  have we done this: {'yes' if self.have_we_done_this else 'no'}; "
                 f"did it answer: {'yes' if self.did_it_answer else 'no'}"]
        for p in self.priors[:5]:
            lines.append(f"  prior {p.experiment_id} [{p.status}/{p.result_kind or 'no result'}] sim {p.question_similarity:.2f} "
                         f"grade {p.grade}" + (f" ({'; '.join(p.grade_reasons)})" if p.grade_reasons else ""))
        lines += [f"  differs: {d}" for d in self.differences[:8]]
        lines += [f"  verified reason: {r}" for r in self.verified_reasons]
        lines += [f"  rejected reason: {r} ({why})" for r, why in self.rejected_reasons]
        lines += [f"  problem: {p}" for p in self.problems]
        return "\n".join(lines)


def verify_reason(reason: ReplicationReason, rec: ExperimentRecord, ext: DesignExtension, prior: PriorMatch,
                  prior_rec: ExperimentRecord, prior_ext: DesignExtension | None, contradicted: set) -> tuple:
    """(holds, why). A claimed reason must be TRUE of the two designs: 'fresh period' with overlapping periods is rejected."""
    R = ReplicationReason
    if reason == R.FRESH_PERIOD:
        if prior_ext is None:
            return False, "the prior did not record its period, so freshness cannot be verified"
        return (not ext.time_period.overlaps(prior_ext.time_period)), "periods overlap" if ext.time_period.overlaps(prior_ext.time_period) else ""
    if reason == R.FRESH_STOCKS:
        ok = bool(ext.universe) and prior_ext is not None and bool(prior_ext.universe) and ext.universe != prior_ext.universe
        return ok, "" if ok else "universe identical, unrecorded or missing on the prior"
    if reason == R.FRESH_REGIME:
        ok = bool(ext.regime) and prior_ext is not None and bool(prior_ext.regime) and ext.regime != prior_ext.regime
        return ok, "" if ok else "regime identical, unrecorded or missing on the prior"
    if reason == R.FRESH_SEED:
        if rec.experiment.seed == prior_rec.experiment.seed:
            return False, "same seed"
        if prior.result_kind != ResultKind.CONFIRMED:
            return False, f"a new seed cannot overturn a {prior.result_kind} answer; only a CONFIRMED claim needs seed replication"
        return True, ""
    if reason == R.CONTROL_COMPARISON:
        extra = set(rec.experiment.controls) - set(prior_rec.experiment.controls)
        return bool(extra), "" if extra else "no control the earlier run lacked"
    if reason == R.INDEPENDENT_METHOD:
        ok = prior.diff.method_changed()
        return ok, "" if ok else "algorithm, features and representation are not materially different"
    if reason == R.CODE_CHANGED:
        ok = bool(rec.experiment.code_hash and prior_rec.experiment.code_hash and rec.experiment.code_hash != prior_rec.experiment.code_hash)
        return ok, "" if ok else "code hash equal or unrecorded"
    if reason == R.DATA_CHANGED:
        ok = bool(rec.experiment.data_hash and prior_rec.experiment.data_hash and rec.experiment.data_hash != prior_rec.experiment.data_hash)
        return ok, "" if ok else "data hash equal or unrecorded"
    if reason == R.PRIOR_WEAK:
        ok = prior.grade == AnswerGrade.WEAK
        return ok, "" if ok else f"the prior was graded {prior.grade}, not WEAK"
    if reason == R.CONTRADICTION:
        ok = prior.experiment_id in contradicted
        return ok, "" if ok else "no contradicting answer on file"
    return False, "unknown reason"


class ResearchExperimentMemory:
    """Ledger + section-15 sidecar. `ledger` is the C62 ExperimentLedger (with its LegacyBridge), so there is still one
    dedup path; this class only adds the extension fields and the richer gate in front of `ledger.propose`."""

    def __init__(self, ledger: ExperimentLedger, store: ExtensionStore | None = None, policy: GradePolicy = GradePolicy(),
                 question_match: float | None = None):
        self.ledger = ledger
        self.store = store or ExtensionStore()
        self.policy = policy
        self.question_match = question_match if question_match is not None else ledger.question_match

    # -------------------------------------------------------------- the four questions
    def launch_gate(self, rec: ExperimentRecord, ext: DesignExtension, now, claims: Sequence[ReplicationReason] = (),
                    difference_statement: str = "", current_code: str = "", current_data: str = "") -> LaunchAssessment:
        errs = rec.validate("proposal") + ext.check(now)
        if not [h for h in rec.competing_hypotheses if h.kind in COUNTER_KINDS]:
            errs.append(f"no counter-hypothesis (a hypothesis of kind {COUNTER_KINDS}); section 15 requires one")
        if rec.result is not None:
            errs.append("a proposal cannot carry a result")
        if errs:
            return LaunchAssessment(LaunchDecision.INVALID_PROPOSAL, "proposal is malformed: " + "; ".join(errs[:4]), False, problems=tuple(errs))
        ledger_verdict = self.ledger.already_tested(rec.question, rec.experiment, now)
        priors = self._related_priors(rec, ext, now, current_code, current_data)
        done = bool(priors) or ledger_verdict.tested_before
        if not done:
            return LaunchAssessment(LaunchDecision.LAUNCH_NOVEL, "nothing comparable has been run; this is new", False,
                                    ledger_status=ledger_verdict.status)
        in_flight = [p for p in priors if p.status in (ExperimentStatus.PROPOSED, ExperimentStatus.RUNNING)]
        if in_flight or ledger_verdict.status == DuplicateStatus.IN_FLIGHT:
            ids = [p.experiment_id for p in in_flight] or [m.experiment_id for m in ledger_verdict.matches]
            return LaunchAssessment(LaunchDecision.WAIT_IN_FLIGHT, f"an equivalent experiment is still running: {ids}", True,
                                    tuple(priors), differences=tuple(priors[0].diff.lines()) if priors else (),
                                    ledger_status=ledger_verdict.status)
        solid = [p for p in priors if p.grade == AnswerGrade.SOLID]
        stale = [p for p in priors if p.grade == AnswerGrade.STALE]
        diffs = tuple(priors[0].diff.lines()) if priors else ()
        contradicted = {x for c in self.ledger.contradictory_answers(now) for x in (c["a"], c["b"])}
        verified, rejected, available = self._verify_claims(rec, ext, priors, claims, contradicted, now)
        if solid or stale:
            return self._decide_answered(rec, priors, solid, stale, verified, rejected, available, claims, ledger_verdict, diffs)
        return self._decide_unanswered(priors, difference_statement, verified, rejected, available, ledger_verdict, diffs)

    def _related_priors(self, rec: ExperimentRecord, ext: DesignExtension, now, code: str, data: str) -> list:
        fp = rec.experiment.fingerprint()
        space = em.infer_space([dict(rec.experiment.config)] + [dict(r.experiment.config) for r in self.ledger.view(now).values()])
        out = []
        for r in self.ledger.view(now).values():
            qs = question_similarity(rec.question, r.question)
            same_fp = r.experiment.fingerprint() == fp
            pext = self.store.design(r.experiment_id, now)
            near = bool(rec.experiment.config) and bool(r.experiment.config) and \
                em.config_distance(rec.experiment.config, r.experiment.config, space) <= self.ledger.near
            same_cfg_hash = pext is not None and pext.configuration_hash == ext.configuration_hash
            if qs < self.question_match and not same_fp and not same_cfg_hash and not (near and qs >= 0.4):
                continue
            outcome = self.store.outcome(r.experiment_id, now)
            g = grade_answer(r, outcome, self.policy, code, data)
            out.append(PriorMatch(r.experiment_id, r.status, r.result.kind if r.result else "", round(qs, 4), g.grade, g.reasons,
                                  design_diff(rec, ext, r, pext), r.learned))
        return sorted(out, key=lambda p: (-p.question_similarity, p.diff.materiality))

    def _verify_claims(self, rec, ext, priors, claims, contradicted, now):
        relevant = [p for p in priors if p.grade in (AnswerGrade.SOLID, AnswerGrade.STALE, AnswerGrade.WEAK)]
        verified, rejected = [], []
        for c in dict.fromkeys(ReplicationReason(x) for x in claims):
            if not relevant:
                rejected.append((c, "no answered prior to replicate"))
                continue
            whys = []
            for p in relevant:
                ok, why = verify_reason(c, rec, ext, p, self.ledger.get(p.experiment_id, now), self.store.design(p.experiment_id, now), contradicted)
                if not ok:
                    whys.append(f"vs {p.experiment_id}: {why}")
            (rejected.append((c, "; ".join(whys))) if whys else verified.append(c))
        available = []
        for c in ReplicationReason:
            if c in verified or c in dict(rejected):
                continue
            if relevant and all(verify_reason(c, rec, ext, p, self.ledger.get(p.experiment_id, now), self.store.design(p.experiment_id, now),
                                              contradicted)[0] for p in relevant):
                available.append(c)
        return tuple(verified), tuple(rejected), tuple(available)

    def _decide_answered(self, rec, priors, solid, stale, verified, rejected, available, claims, lv, diffs) -> LaunchAssessment:
        top = (solid or stale)[0]
        base = dict(priors=tuple(priors), did_it_answer=bool(solid), verified_reasons=verified, rejected_reasons=rejected,
                    available_reasons=available, differences=diffs, ledger_status=lv.status, have_we_done_this=True)
        said = "; ".join(top.learned[:2]) or "see the record"
        if verified:
            return LaunchAssessment(LaunchDecision.LAUNCH_REPLICATION, f"justified replication of {top.experiment_id} "
                                    f"({', '.join(map(str, verified))}); earlier learned: {said}", **base)
        if available and not claims:
            return LaunchAssessment(LaunchDecision.NEEDS_REASON, f"{top.experiment_id} already answered this ({top.result_kind}); a repeat "
                                    f"needs a stated reason, and these hold: {', '.join(map(str, available))}", **base)
        if top.diff.is_same_design:
            return LaunchAssessment(LaunchDecision.REFUSE_SAME_DESIGN, f"same design as {top.experiment_id} "
                                    f"(materiality {top.diff.materiality:.2f}); repeating it teaches nothing", **base)
        return LaunchAssessment(LaunchDecision.REFUSE_ALREADY_ANSWERED, f"{top.experiment_id} already answered this question "
                                f"({top.result_kind}); do not repeat without a justified replication reason", **base)

    def _decide_unanswered(self, priors, statement, verified, rejected, available, lv, diffs) -> LaunchAssessment:
        top = priors[0] if priors else None
        base = dict(priors=tuple(priors), did_it_answer=False, verified_reasons=verified, rejected_reasons=rejected,
                    available_reasons=available, differences=diffs, ledger_status=lv.status, have_we_done_this=True)
        if top is None:                                     # only the legacy stores remembered it
            if len(statement.split()) >= 4:
                return LaunchAssessment(LaunchDecision.LAUNCH_AFTER_EXPLANATION, "a legacy store remembers a similar configuration; "
                                        f"launching on the stated difference: {statement.strip()}", **base)
            return LaunchAssessment(LaunchDecision.NEEDS_EXPLANATION, f"a legacy store remembers this ({lv.message}); explain what is different", **base)
        weak_note = f" (its weaknesses: {'; '.join(top.grade_reasons)})" if top.grade_reasons else ""
        words = len(statement.split())
        changed = top.diff.materiality >= MATERIAL_AT or top.status in (ExperimentStatus.INVALID, ExperimentStatus.FAILED, ExperimentStatus.ABANDONED)
        if words >= 4 and (changed or verified):
            return LaunchAssessment(LaunchDecision.LAUNCH_AFTER_EXPLANATION, f"{top.experiment_id} did not answer{weak_note}; "
                                    f"launching on the stated difference: {statement.strip()}", **base)
        if words >= 4:
            return LaunchAssessment(LaunchDecision.REFUSE_SAME_DESIGN, f"the statement claims a difference but the designs are the same "
                                    f"(materiality {top.diff.materiality:.2f}) as unanswered {top.experiment_id}", **base)
        return LaunchAssessment(LaunchDecision.NEEDS_EXPLANATION, f"{top.experiment_id} did not answer this{weak_note}; "
                                "state in a full sentence what is different this time", **base)

    # -------------------------------------------------------------- writing
    def register(self, rec: ExperimentRecord, ext: DesignExtension, now, claims: Sequence[ReplicationReason] = (),
                 difference_statement: str = "", current_code: str = "", current_data: str = "") -> tuple:
        """Gate, then write ledger + sidecar. A refused proposal writes nothing. Returns (stamped record, assessment)."""
        a = self.launch_gate(rec, ext, now, claims, difference_statement, current_code, current_data)
        if not a.decision.permits_launch:
            raise LaunchRefused(a)
        reason = ""
        if a.decision != LaunchDecision.LAUNCH_NOVEL:
            reason = "; ".join(map(str, a.verified_reasons)) or difference_statement.strip() or "explained difference"
        design = dataclasses.replace(rec.experiment, code_hash=rec.experiment.code_hash or current_code_hash())
        stamped = self.ledger.propose(rec.with_(experiment=design), now, allow_repeat_reason=reason)
        self.store.put_design(stamped.experiment_id, ext, now)
        return stamped, a

    def start(self, experiment_id: str, now) -> ExperimentRecord:
        return self.ledger.mark_running(experiment_id, now)

    def close(self, experiment_id: str, result: ExperimentResult, outcome: OutcomeExtension, now, learned: Sequence[str],
              not_learned: Sequence[str], next_action: str, posterior_belief: str = "") -> ExperimentRecord:
        """Close with the C62 answer AND the section-15 outcome. Refused before anything is written if the outcome is incomplete
        or the result claims information from after its own cutoff."""
        dext = self.store.design(experiment_id, now)
        errs = outcome.check(result)
        if errs:
            raise ValueError("outcome incomplete: " + "; ".join(errs))
        if dext is not None and result.observed_at and to_ts(result.observed_at) <= to_ts(dext.information_cutoff):
            raise FirewallBreach(f"{experiment_id}: result observed {result.observed_at} is not after its information cutoff {dext.information_cutoff}")
        new = self.ledger.record_result(experiment_id, result, now, learned, not_learned, next_action, posterior_belief)
        self.store.put_outcome(experiment_id, dataclasses.replace(
            outcome, compute_cost_minutes=outcome.compute_cost_minutes or new.experiment.cost_minutes), now)
        return new

    # -------------------------------------------------------------- the 24 fields
    def dossier(self, experiment_id: str, now) -> dict:
        """Every section-15 field for one experiment, as of `now`; anything not recorded is the explicit MISSING marker."""
        r = self.ledger.get(experiment_id, now)
        if r is None:
            raise KeyError(f"{experiment_id} not visible at {now}")
        d, o = self.store.design(experiment_id, now), self.store.outcome(experiment_id, now)
        res = r.result
        counter = [h.statement for h in r.competing_hypotheses if h.kind in COUNTER_KINDS]
        lead = max(r.competing_hypotheses, key=lambda h: h.prior).statement if r.competing_hypotheses else MISSING
        val = None if o is None else o.realised_value
        return {
            "question": r.question, "hypothesis": lead, "counter_hypotheses": counter or MISSING,
            "dataset": d.dataset if d else MISSING,
            "time_period": [d.time_period.start, d.time_period.end] if d else (list(r.experiment.windows) or MISSING),
            "information_cutoff": d.information_cutoff if d else MISSING, "features": list(d.features) if d else MISSING,
            "representation": d.representation if d else MISSING, "algorithm": d.algorithm if d else MISSING,
            "hyperparameters": dict(d.hyperparameters) if d else (dict(r.experiment.config) or MISSING),
            "random_seed": r.experiment.seed if r.experiment.seed is not None else MISSING,
            "code_hash": r.experiment.code_hash or MISSING, "data_hash": r.experiment.data_hash or MISSING,
            "configuration_hash": d.configuration_hash if d else r.experiment.fingerprint(),
            "controls": list(r.experiment.controls) or MISSING,
            "result": (res.kind if res else MISSING), "confidence_interval": list(res.ci) if res and res.ci else MISSING,
            "failure_mode": (str(o.failure_mode) if o and o.failure_mode else ("none" if o else MISSING)),
            "transfer_result": (o.transfer.state.value if o else MISSING), "memorization_result": (o.memorization.state.value if o else MISSING),
            "complexity": dataclasses.asdict(d.complexity) if d else MISSING,
            "compute_cost": (o.compute_cost_minutes if o else (r.experiment.cost_minutes or MISSING)),
            "research_value": (dataclasses.asdict(val) if val and len(val.missing()) < len(dataclasses.fields(val)) else MISSING),
            "decision_impact": ([str(e) for e in o.realised_impact.effects] if o else MISSING)}

    def field_coverage(self, now) -> dict:
        """Per section-15 field: how many visible experiments genuinely carry it (NOT RECORDED counts as absent)."""
        ids = list(self.ledger.view(now))
        per = {f: 0 for f in SECTION15_FIELDS}
        for eid in ids:
            dos = self.dossier(eid, now)
            for f in SECTION15_FIELDS:
                if dos[f] != MISSING:
                    per[f] += 1
        n = len(ids)
        return {"n": n, "per_field": per, "share": {f: (c / n if n else 0.0) for f, c in per.items()},
                "complete": sum(1 for eid in ids if all(v != MISSING for v in self.dossier(eid, now).values())),
                "missing_most": sorted(per, key=lambda f: per[f])[:5] if n else []}


# ------------------------------------------------------------------------------------------------ section 32: replication

@dataclass(frozen=True)
class ReplicationPolicy:
    """Section 32: 'some combination' of independent replication, fresh period / stocks / seed / regime, control comparison.
    These defaults are a documented starting point (one replicate, two independent axes, one of them a fresh period);
    the thresholds are meant to be replaced by learned ones, not tuned on P&L."""
    min_replicates: int = 1
    min_axes: int = 2
    require_fresh_period: bool = True
    block_on_discordant: bool = True


@dataclass(frozen=True)
class ReplicationStatus:
    gate: GateVerdict
    message: str
    original: str = ""
    concordant: tuple = ()
    discordant: tuple = ()
    axes_covered: tuple = ()
    axes_missing: tuple = ()


def replication_axes(orig_rec: ExperimentRecord, orig: DesignExtension | None, rep_rec: ExperimentRecord, rep: DesignExtension | None) -> tuple:
    """Which independence axes the replicate adds relative to the original."""
    axes = []
    if orig and rep and not rep.time_period.overlaps(orig.time_period):
        axes.append("fresh_period")
    if orig and rep and orig.universe and rep.universe and orig.universe != rep.universe:
        axes.append("fresh_stocks")
    if orig and rep and orig.regime and rep.regime and orig.regime != rep.regime:
        axes.append("fresh_regime")
    if orig_rec.experiment.seed != rep_rec.experiment.seed:
        axes.append("fresh_seed")
    if set(rep_rec.experiment.controls) - set(orig_rec.experiment.controls):
        axes.append("control_comparison")
    return tuple(axes)


def _ci_overlap(a: Sequence, b: Sequence) -> bool:
    return not a or not b or (a[0] <= b[1] and b[0] <= a[1])


def replication_status(mem: ResearchExperimentMemory, question: str, now, policy: ReplicationPolicy = ReplicationPolicy()) -> ReplicationStatus:
    """May the claim behind `question` count as replicated? One lucky experiment never can (section 32): a single CONFIRMED
    result is NEEDS_MORE_EVIDENCE; an unresolved discordant replicate is QUARANTINED; a memorised or non-transferring original is FAILED."""
    recs = [r for r in mem.ledger.answered(now) if question_similarity(question, r.question) >= mem.question_match and r.result]
    recs.sort(key=lambda r: (r.result.observed_at, r.experiment_id))
    if not recs:
        return ReplicationStatus(GateVerdict.UNKNOWN, "no answered experiment on this question")
    confirmed = [r for r in recs if r.result.kind == ResultKind.CONFIRMED]
    if not confirmed:
        return ReplicationStatus(GateVerdict.FAILED, f"{len(recs)} answered experiment(s), none confirmed the claim")
    orig = confirmed[0]
    oo = mem.store.outcome(orig.experiment_id, now)
    if oo is not None and (oo.memorization.state == TestState.FAILED or oo.transfer.state == TestState.FAILED):
        return ReplicationStatus(GateVerdict.FAILED, f"the original {orig.experiment_id} is explained by memorisation or does not transfer",
                                 orig.experiment_id)
    od = mem.store.design(orig.experiment_id, now)
    concordant, discordant, covered = [], [], []
    for r in recs:
        if r.experiment_id == orig.experiment_id:
            continue
        rd = mem.store.design(r.experiment_id, now)
        agree = r.result.kind == ResultKind.CONFIRMED and _ci_overlap(orig.result.ci, r.result.ci)
        if agree:
            concordant.append(r.experiment_id)
            covered += [a for a in replication_axes(orig, od, r, rd) if a not in covered]
        elif r.result.kind == ResultKind.REFUTED or not _ci_overlap(orig.result.ci, r.result.ci):
            discordant.append(r.experiment_id)
    needed = {"fresh_period"} if policy.require_fresh_period else set()
    missing = tuple(sorted(needed - set(covered)))
    if discordant and policy.block_on_discordant:
        return ReplicationStatus(GateVerdict.QUARANTINED, f"replicate(s) {discordant} disagree with the original; unresolved contradiction",
                                 orig.experiment_id, tuple(concordant), tuple(discordant), tuple(covered))
    if len(concordant) < policy.min_replicates:
        return ReplicationStatus(GateVerdict.NEEDS_MORE_EVIDENCE, f"only {len(concordant)} concordant replicate(s); {policy.min_replicates} needed "
                                 "(one lucky experiment must not change the system)", orig.experiment_id, tuple(concordant), (), tuple(covered),
                                 tuple(sorted(needed)))
    if missing or len(covered) < policy.min_axes:
        gap = tuple(sorted(set(missing) | ({"one more independent axis"} if len(covered) < policy.min_axes else set())))
        return ReplicationStatus(GateVerdict.NEEDS_MORE_EVIDENCE, f"replicated but independence is thin: covered {covered}, missing {list(gap)}",
                                 orig.experiment_id, tuple(concordant), (), tuple(covered), gap)
    return ReplicationStatus(GateVerdict.PROMOTE, f"replication requirement met: {len(concordant)} concordant replicate(s) over {covered}",
                             orig.experiment_id, tuple(concordant), (), tuple(covered))


# ------------------------------------------------------------------------------------------------ multiplicity

def family_trials(mem: ResearchExperimentMemory, question: str, now) -> int:
    """How many experiments (any outcome) probed this question family before `now`, counting the free choices each one made.
    Refuted variants COUNT: a discovery is only as strong as the number of things that were tried to find it."""
    total = 0
    for r in mem.ledger.view(now).values():
        if question_similarity(question, r.question) >= mem.question_match and r.status != ExperimentStatus.PROPOSED:
            d = mem.store.design(r.experiment_id, now)
            total += d.complexity.n_free_choices if d else 1
    return total


def deflated_threshold(alpha: float, n_trials: int) -> float:
    """Sidak per-test level keeping the family-wise error at alpha across n_trials looks."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return 1.0 - (1.0 - alpha) ** (1.0 / max(1, n_trials))


def deflated_p(p: float, n_trials: int) -> float:
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    return 1.0 - (1.0 - p) ** max(1, n_trials)


def expected_max_null_z(n_trials: int) -> float:
    """Expected best z-score among n_trials pure-noise looks (Blom approximation): the bar a single lucky look already clears."""
    n = max(1, int(n_trials))
    return statistics.NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


# ------------------------------------------------------------------------------------------------ cost and value

def value_accounting(mem: ResearchExperimentMemory, now) -> dict:
    """Compute cost against realised value for every closed experiment. Unmeasured value is counted separately, never as zero."""
    rows, unmeasured = [], 0
    for r in mem.ledger.view(now).values():
        o = mem.store.outcome(r.experiment_id, now)
        if o is None:
            continue
        gain = o.realised_value.information_gain
        if gain is None:
            unmeasured += 1
        rows.append({"experiment_id": r.experiment_id, "algorithm": (mem.store.design(r.experiment_id, now) or _NO_DESIGN).algorithm,
                     "cost": o.compute_cost_minutes, "gain": gain, "null_impact": o.realised_impact.is_null, "kind": r.result.kind if r.result else ""})
    cost = sum(x["cost"] for x in rows)
    null_cost = sum(x["cost"] for x in rows if x["null_impact"])
    by_alg: dict = {}
    for x in rows:
        e = by_alg.setdefault(x["algorithm"], {"n": 0, "cost": 0.0, "refuted": 0})
        e["n"] += 1
        e["cost"] += x["cost"]
        e["refuted"] += int(x["kind"] == ResultKind.REFUTED)
    return {"n": len(rows), "total_cost_minutes": cost, "cost_with_no_decision_impact": null_cost,
            "share_wasted": (null_cost / cost) if cost > 0 else 0.0, "value_unmeasured": unmeasured, "by_algorithm": by_alg,
            "gain_per_minute": {x["experiment_id"]: x["gain"] / x["cost"] for x in rows if x["gain"] is not None and x["cost"] > 0}}


_NO_DESIGN = DesignExtension("", TimePeriod("1970-01-01", "1970-01-01"), "1970-01-01", (), "", "")


# ------------------------------------------------------------------------------------------------ audits

def audit_cutoffs(mem: ResearchExperimentMemory, now) -> list:
    """Experiments whose recorded information cutoff cannot be true: cutoff not before proposal, training period past its
    cutoff, or a result observed at/before the cutoff of the information it used."""
    bad = []
    for r in mem.ledger.view(now).values():
        d = mem.store.design(r.experiment_id, now)
        if d is None:
            continue
        if as_date(d.information_cutoff) >= as_date(r.created_at):
            bad.append({"experiment_id": r.experiment_id, "problem": "cutoff not before the experiment was created"})
        if as_date(d.time_period.end) > as_date(d.information_cutoff):
            bad.append({"experiment_id": r.experiment_id, "problem": "training period ends after the information cutoff"})
        if r.result is not None and r.result.observed_at and to_ts(r.result.observed_at) <= to_ts(d.information_cutoff):
            bad.append({"experiment_id": r.experiment_id, "problem": "result observed before/at the information cutoff"})
    return bad


def legacy_drift(mem: ResearchExperimentMemory, now) -> dict:
    """Experiments the older stores hold that the ledger does not (and the reverse). The mapping wants ONE front; this
    reports where the three still disagree."""
    ids = set(mem.ledger.view(now))
    br = mem.ledger.bridge
    tried = {x["experiment_id"] for x in br.tried.rows} if br and br.tried is not None else set()
    memory = {e["experiment_id"] for e in br.memory.entries} if br and br.memory is not None else set()
    answered = {r.experiment_id for r in mem.ledger.answered(now) if r.experiment.config}
    return {"only_in_tried_index": sorted(tried - ids), "only_in_registry_memory": sorted(memory - ids),
            "answered_not_mirrored_to_tried_index": sorted(answered - tried) if br and br.tried is not None else [],
            "answered_not_mirrored_to_registry": sorted(answered - memory) if br and br.memory is not None else [],
            "consistent": not (tried - ids or memory - ids)}


# ------------------------------------------------------------------------------------------------ namespace bridge

def matured_record(mem: ResearchExperimentMemory, experiment_id: str, now) -> MaturedRecord:
    """An answered experiment as a research-world record. It reaches the trader only through record.gate(now), which refuses
    it unless its result matured strictly before `now` and could have existed then (C66 sections 29-31)."""
    r = mem.ledger.get(experiment_id, now)
    if r is None or r.result is None or r.status not in (ExperimentStatus.ANSWERED, ExperimentStatus.INCONCLUSIVE):
        raise ValueError(f"{experiment_id} has no answered result visible at {now}")
    d = mem.store.design(experiment_id, now)
    seen = d.time_period.end if d else r.result.observed_at
    prov = Provenance(created_real=r.recorded_at, learned_at=r.result.observed_at[:10], code_hash=r.experiment.code_hash or "unrecorded",
                      data_hash=r.experiment.data_hash, config_hash=r.experiment.fingerprint(), experiment_id=experiment_id,
                      seed=r.experiment.seed, outcomes_seen_through=str(seen)[:10] if as_date(seen) >= as_date(r.result.observed_at) else r.result.observed_at[:10])
    payload = {"question_key": em.question_key(r.question), "result": r.result.kind, "learned": list(r.learned),
               "grade": grade_answer(r, mem.store.outcome(experiment_id, now), mem.policy).grade.value}
    return MaturedRecord(record_id=f"exp:{experiment_id}", matured_at=r.result.observed_at, payload=payload, provenance=prov)


# ------------------------------------------------------------------------------------------------ the loop's sweep and reports

@dataclass(frozen=True)
class StepReport:
    now: str
    n_experiments: int
    stale: tuple
    unfollowed_actions: tuple
    unanswered_hypotheses: tuple
    contradictions: tuple
    cutoff_violations: tuple
    drift: Mapping
    coverage: Mapping
    needs_replication: tuple
    accounting: Mapping
    label: str = LABEL

    @property
    def healthy(self) -> bool:
        return not self.cutoff_violations and not self.contradictions and self.drift.get("consistent", True)


def step(mem: ResearchExperimentMemory, now, current_code: str = "", current_data: str = "",
         policy: ReplicationPolicy = ReplicationPolicy()) -> StepReport:
    """The research loop's daily call: what is stale, what follow-up nobody ran, which contradictions and cutoff breaches exist,
    which CONFIRMED claims still lack replication. Read-only."""
    needs = []
    seen_keys = set()
    for r in mem.ledger.answered(now):
        if r.result and r.result.kind == ResultKind.CONFIRMED:
            key = em.question_key(r.question)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rs = replication_status(mem, r.question, now, policy)
            if rs.gate != GateVerdict.PROMOTE:
                needs.append({"question": r.question, "gate": rs.gate.value, "message": rs.message, "missing": list(rs.axes_missing)})
    return StepReport(now=em._iso(now), n_experiments=len(mem.ledger.view(now)),
                      stale=tuple(em.stale_answers(mem.ledger, now, current_code, current_data)) if current_code else (),
                      unfollowed_actions=tuple(mem.ledger.pending_next_actions(now)),
                      unanswered_hypotheses=tuple(em.unanswered_hypotheses(mem.ledger, now)),
                      contradictions=tuple(mem.ledger.contradictory_answers(now)), cutoff_violations=tuple(audit_cutoffs(mem, now)),
                      drift=legacy_drift(mem, now), coverage=mem.field_coverage(now), needs_replication=tuple(needs),
                      accounting=value_accounting(mem, now))


def render_step(rep: StepReport, limit: int = 8) -> str:
    lines = [f"Research experiment memory as of {rep.now}   [{rep.label}]   healthy={rep.healthy}",
             f"experiments: {rep.n_experiments}; complete section-15 dossiers: {rep.coverage['complete']}",
             f"stale answers: {len(rep.stale)}; unfollowed next actions: {len(rep.unfollowed_actions)}; contradictions: {len(rep.contradictions)}; "
             f"cutoff violations: {len(rep.cutoff_violations)}"]
    lines += [f"  needs replication: {n['question']} -> {n['gate']} ({n['message']})" for n in rep.needs_replication[:limit]]
    lines += [f"  cutoff violation: {v['experiment_id']}: {v['problem']}" for v in rep.cutoff_violations[:limit]]
    acc = rep.accounting
    lines.append(f"compute: {acc['total_cost_minutes']:.1f} CPU-min, {acc['share_wasted']:.0%} on experiments with no decision impact; "
                 f"value unmeasured on {acc['value_unmeasured']}")
    lines.append("least-recorded fields: " + ", ".join(rep.coverage["missing_most"]))
    return "\n".join(lines)


def render_dossier(mem: ResearchExperimentMemory, experiment_id: str, now) -> str:
    d = mem.dossier(experiment_id, now)
    return "\n".join([f"Experiment {experiment_id} as of {em._iso(now)}"] + [f"  {f}: {_short(d[f], 100)}" for f in SECTION15_FIELDS])


# ------------------------------------------------------------------------------------------------ integrity and duplicate audit

def memory_integrity(mem: ResearchExperimentMemory, now) -> dict:
    """Do the ledger and the sidecar agree? Orphaned extension rows, closed experiments with no outcome extension, proposals
    with no design extension (all legacy rows), and outcome rows whose experiment never closed."""
    view = mem.ledger.view(now)
    cut = to_ts(now)
    design_ids = {r["experiment_id"] for r in mem.store._rows if r["kind"] == "design" and to_ts(r["recorded_at"]) < cut}
    outcome_ids = {r["experiment_id"] for r in mem.store._rows if r["kind"] == "outcome" and to_ts(r["recorded_at"]) < cut}
    closed = {i for i, r in view.items() if r.status in (ExperimentStatus.ANSWERED, ExperimentStatus.INCONCLUSIVE, ExperimentStatus.INVALID)}
    problems = []
    for i in sorted(design_ids - set(view)):
        problems.append(f"design extension for {i} has no ledger record")
    for i in sorted(outcome_ids - closed):
        problems.append(f"outcome extension for {i} but the experiment is not closed")
    no_outcome = sorted(i for i in closed - outcome_ids if not view[i].legacy)
    no_design = sorted(i for i in set(view) - design_ids if not view[i].legacy)
    return {"problems": problems, "closed_without_outcome": no_outcome, "proposed_without_design": no_design,
            "legacy": sum(1 for r in view.values() if r.legacy), "ok": not problems and not no_outcome and not no_design,
            "unparseable_sidecar_lines": mem.store.unparseable}


def duplicate_clusters(mem: ResearchExperimentMemory, now, min_similarity: float | None = None) -> list:
    """Groups of experiments that are the same question AND the same design: repeats that already happened. Each cluster lists
    whether a recorded reason justified the repeat, and the compute the unjustified copies cost."""
    thr = min_similarity if min_similarity is not None else mem.question_match
    recs = sorted(mem.ledger.view(now).values(), key=lambda r: (r.recorded_at, r.experiment_id))
    parent = {r.experiment_id: r.experiment_id for r in recs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if question_similarity(a.question, b.question) < thr:
                continue
            da, db = mem.store.design(a.experiment_id, now), mem.store.design(b.experiment_id, now)
            if da is not None and db is not None:
                same = da.configuration_hash == db.configuration_hash
            else:
                same = a.experiment.fingerprint() == b.experiment.fingerprint()
            if same:
                parent[find(b.experiment_id)] = find(a.experiment_id)
    groups: dict = {}
    for r in recs:
        groups.setdefault(find(r.experiment_id), []).append(r)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        unjustified = [m for m in members[1:] if not m.repeat_reason]
        cost = 0.0
        for m in unjustified:
            o = mem.store.outcome(m.experiment_id, now)
            cost += o.compute_cost_minutes if o else m.experiment.cost_minutes
        out.append({"experiments": [m.experiment_id for m in members], "question": members[0].question,
                    "justified": [m.experiment_id for m in members[1:] if m.repeat_reason],
                    "unjustified": [m.experiment_id for m in unjustified], "wasted_minutes": cost})
    return sorted(out, key=lambda g: -g["wasted_minutes"])


# ------------------------------------------------------------------------------------------------ what to ask next

@dataclass(frozen=True)
class NextQuestion:
    text: str
    origin: str                            # not_learned | live_hypothesis | follow_up | stale | replication | contradiction
    weight: float
    source_ids: tuple
    detail: str = ""


def suggest_next(mem: ResearchExperimentMemory, now, current_code: str = "", limit: int = 12,
                 policy: ReplicationPolicy = ReplicationPolicy()) -> list:
    """Questions the memory itself says are open, ranked by how many independent experiments point at them. Six sources: things
    experiments said they did not learn (more often = weightier), hypotheses still holding posterior mass after an answer, next
    actions nobody followed up, answers gone stale, confirmed claims lacking replication, and unresolved contradictions.
    Questions are returned, never launched: each still has to pass `launch_gate`."""
    out: list = []
    for e in list(mem.ledger.not_learned_register(now).values()):
        if e["text"] != MISSING and not e["text"].lower().startswith(("nothing", "abandoned")):
            out.append(NextQuestion(e["text"], "not_learned", 1.0 + 0.5 * (e["count"] - 1), tuple(e["experiments"])))
    for h in em.unanswered_hypotheses(mem.ledger, now):
        out.append(NextQuestion(f"Does '{h['hypothesis']}' hold?", "live_hypothesis", 1.0 + 2.0 * h["posterior"], (h["experiment_id"],),
                                f"posterior {h['posterior']:.2f} after the last experiment"))
    for a in mem.ledger.pending_next_actions(now):
        out.append(NextQuestion(a["next_action"], "follow_up", 1.2, (a["experiment_id"],), "recorded next action never followed"))
    if current_code:
        for s in em.stale_answers(mem.ledger, now, current_code):
            out.append(NextQuestion(f"Does the answer to '{s['question']}' still hold?", "stale", 1.1, (s["experiment_id"],), "; ".join(s["reasons"])))
    for c in mem.ledger.contradictory_answers(now):
        out.append(NextQuestion(f"Why do {c['a']} and {c['b']} disagree on '{c['question']}'?", "contradiction", 2.5, (c["a"], c["b"])))
    seen = set()
    for r in mem.ledger.answered(now):
        key = em.question_key(r.question)
        if r.result and r.result.kind == ResultKind.CONFIRMED and key not in seen:
            seen.add(key)
            rs = replication_status(mem, r.question, now, policy)
            if rs.gate == GateVerdict.NEEDS_MORE_EVIDENCE:
                out.append(NextQuestion(f"Replicate: {r.question}", "replication", 2.0, (r.experiment_id,), rs.message))
    merged: dict = {}
    for q in out:
        k = em.question_key(q.text)
        if k not in merged or q.weight > merged[k].weight:
            merged[k] = q
    return sorted(merged.values(), key=lambda q: (-q.weight, q.text))[:limit]


def evidence_summary(mem: ResearchExperimentMemory, question: str, now) -> dict:
    """Everything the memory holds on one question: outcomes by grade, the stance, trials made. The grade matters more than
    the count: five WEAK confirmations are not five confirmations."""
    rows = []
    for r in mem.ledger.view(now).values():
        if question_similarity(question, r.question) < mem.question_match:
            continue
        g = grade_answer(r, mem.store.outcome(r.experiment_id, now), mem.policy)
        rows.append({"experiment_id": r.experiment_id, "status": r.status, "kind": r.result.kind if r.result else "", "grade": g.grade.value,
                     "why": list(g.reasons)})
    by: dict = {}
    for x in rows:
        k = f"{x['kind'] or x['status']}/{x['grade']}"
        by[k] = by.get(k, 0) + 1
    solid = [x for x in rows if x["grade"] == AnswerGrade.SOLID.value]
    kinds = {x["kind"] for x in solid}
    if not rows:
        stance = "UNTESTED"
    elif not solid:
        stance = "UNSETTLED"
    elif kinds == {ResultKind.CONFIRMED}:
        stance = "SUPPORTED"
    elif kinds <= {ResultKind.REFUTED, ResultKind.NULL}:
        stance = "REFUTED"
    else:
        stance = "CONTRADICTED"
    return {"question": question, "n": len(rows), "by_outcome_and_grade": by, "stance": stance, "trials": family_trials(mem, question, now),
            "experiments": rows}


def render_launch(a: LaunchAssessment, question: str = "") -> str:
    head = f"Launch review{': ' + question if question else ''}"
    return head + "\n" + a.render()
