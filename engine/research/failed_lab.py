"""Failed-learner lab (C66 section 17; extends C62 section 38; Bible phase 30; canon C58-C66).

C62's `engine.learning.failed_learners` already stores each failed learner (hypothesis, implementation, failure mode, regimes,
numbers, reason, generalisation) and blocks a proposal that repeats one by name, tags or wording. Section 17 asks for more:
every serious failure becomes a scientific artifact that says WHAT IT ATTEMPTED, WHY IT LOOKED PROMISING, WHAT IT ACTUALLY DID,
WHERE IT FAILED, and whether it OVERFIT, MEMORISED, TRANSFERRED, INCREASED RISK and whether the FAILURE GENERALISED - and the lab
must answer, for a new idea, "we have already attempted this CLASS of learner 14 times: 11 failed because of overfitting, 2
because the signal did not transfer, 1 because of risk concentration; a new attempt requires a materially different hypothesis".

What this module adds on top of the registry (which it wraps, never copies):
  * `Autopsy` - the section-17 answers, each finding YES / NO / UNKNOWN / NOT_APPLICABLE with the number or note that supports it.
    UNKNOWN is a first-class answer: no finding is filled in without a basis, and nothing defaults to NO.
  * learner CLASS resolution - a declared mechanism taxonomy plus similarity to recorded attempts, so a renamed or re-tagged
    idea still lands in the class it belongs to. Every version of a learner is one ATTEMPT (a retest is another attempt).
  * `consult` - the class-level statement above, a material-difference test (new mechanism or a genuinely different hypothesis,
    plus the controls that would have caught the earlier failure modes), and a decision. A class is never closed for ever:
    a DEAD_CLASS verdict lists the conditions under which it reopens.
  * a consultation log (how many rediscoveries were redirected), pass records (so class statistics are not survivor-only),
    lab health, and the bridge from a closed experiment (engine.research.experiments) into a lab entry.

Failures are matured research state: `matured_record` is the only way one leaves this module toward a decision (C66 sections
29-31). IMPLEMENTED - NOT VALIDATED. Public entries: `FailedLearnerLab.consult`, `record_failure`, and `step`."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from engine.learning import experiment_memory as em
from engine.learning import failed_learners as fl_mod
from engine.learning.core import (FirewallBreach, Provenance, Subsystem, ValidationLabel, as_date, canonical_json,
                                  stable_hash)
from engine.learning.failed_learners import (REQUIRED_CONTROLS, DeadEndStatus, FailedLearner, FailedLearnerRegistry,
                                             FailureMode, Generalization, Justification, LearnerProposal,
                                             proposal_similarity, same_regime)
from engine.learning.experiment_memory import question_similarity, to_ts
from engine.research.core import MaturedRecord

LABEL = ValidationLabel.NOT_VALIDATED.value
NOT_RECORDED = "NOT RECORDED"


class Finding(str, enum.Enum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"                    # nobody measured it; NEVER read as NO
    NOT_APPLICABLE = "NOT_APPLICABLE"      # e.g. a learner that adopted nothing cannot have overfit

    def __str__(self):
        return self.value

    @property
    def known(self) -> bool:
        return self in (Finding.YES, Finding.NO)


class FailureStage(str, enum.Enum):
    DESIGN = "DESIGN"                      # the search space or gate could produce nothing
    IN_SAMPLE = "IN_SAMPLE"
    HOLDOUT = "HOLDOUT"
    TRANSFER = "TRANSFER"
    RISK_GUARD = "RISK_GUARD"
    EXECUTION = "EXECUTION"
    DATA = "DATA"
    VALIDATION = "VALIDATION"

    def __str__(self):
        return self.value


class ReasonBucket(str, enum.Enum):
    """The one primary reason a failure is counted under, so the class statement's counts add up to its attempts."""
    OVERFITTING = "overfitting"
    DID_NOT_TRANSFER = "the signal did not transfer"
    RISK_CONCENTRATION = "risk concentration"
    NO_SIGNAL = "no signal above the base rate"
    ADOPTED_NOTHING = "the learner adopted nothing"
    HARMFUL = "it made results worse"
    LEAKAGE = "a future-information leak"
    INFEASIBLE = "it could not run on the available data or compute"
    UNVERIFIED = "the claimed gain could not be reproduced"
    UNKNOWN = "an undetermined cause"

    def __str__(self):
        return self.value


BUCKET_CLAUSE = {ReasonBucket.OVERFITTING.value: "because of overfitting", ReasonBucket.DID_NOT_TRANSFER.value: "because the signal did not transfer",
                 ReasonBucket.RISK_CONCENTRATION.value: "because of risk concentration", ReasonBucket.NO_SIGNAL.value: "because there was no signal above the base rate",
                 ReasonBucket.ADOPTED_NOTHING.value: "because the learner adopted nothing", ReasonBucket.HARMFUL.value: "because it made results worse",
                 ReasonBucket.LEAKAGE.value: "because a future-information leak explained the gain",
                 ReasonBucket.INFEASIBLE.value: "because it could not run on the available data or compute",
                 ReasonBucket.UNVERIFIED.value: "because the claimed gain could not be reproduced", ReasonBucket.UNKNOWN.value: "for an undetermined cause"}


class Decision(str, enum.Enum):
    PROCEED = "PROCEED"                                        # nothing of this class has failed
    PROCEED_WITH_CONTROLS = "PROCEED_WITH_CONTROLS"            # class has failed, but this attempt is materially different
    NEEDS_DIFFERENT_HYPOTHESIS = "NEEDS_DIFFERENT_HYPOTHESIS"
    BLOCKED = "BLOCKED"                                        # a dead class and a restated idea

    def __str__(self):
        return self.value

    @property
    def permits(self) -> bool:
        return self in (Decision.PROCEED, Decision.PROCEED_WITH_CONTROLS)


class ClassVerdict(str, enum.Enum):
    DEAD_CLASS = "DEAD_CLASS"
    CONTESTED = "CONTESTED"                # failures of different kinds, or some passes recorded
    UNDER_EXPLORED = "UNDER_EXPLORED"      # too few attempts to call it
    OPEN = "OPEN"                          # a pass on record and no dominant failure

    def __str__(self):
        return self.value


class MateriallySameAttempt(Exception):
    def __init__(self, consultation: "Consultation"):
        super().__init__(consultation.statement)
        self.consultation = consultation


# ------------------------------------------------------------------------------------------------ the autopsy

@dataclass(frozen=True)
class FailureSite:
    stage: FailureStage
    detail: str = ""
    subsystem: Subsystem | None = None      # which part of the trading pipeline the failed learner would have served


@dataclass(frozen=True)
class Autopsy:
    """The nine section-17 items. `basis` maps a finding name to the number or note that supports a YES/NO; a YES/NO without a
    basis is refused, so no finding is opinion."""
    attempted: str
    why_promising: str
    what_it_did: str
    where_failed: FailureSite
    overfit: Finding = Finding.UNKNOWN
    memorized: Finding = Finding.UNKNOWN
    transferred: Finding = Finding.UNKNOWN
    increased_risk: Finding = Finding.UNKNOWN
    failure_generalized: Finding = Finding.UNKNOWN
    basis: Mapping = field(default_factory=dict)
    derived: bool = False                   # True = filled by derive_autopsy from the registry record, not written by a person

    FINDINGS = ("overfit", "memorized", "transferred", "increased_risk", "failure_generalized")

    def validate(self) -> list:
        errs = []
        for f in ("attempted", "what_it_did"):
            if not str(getattr(self, f)).strip():
                errs.append(f"autopsy.{f} empty")
        if not self.why_promising.strip():
            errs.append("autopsy.why_promising empty (write NOT RECORDED if it truly was not)")
        for name in self.FINDINGS:
            v = getattr(self, name)
            if v.known and not str(self.basis.get(name, "")).strip():
                errs.append(f"autopsy.{name}={v} has no basis (a finding without its number is opinion; use UNKNOWN)")
        if self.transferred == Finding.YES and self.memorized == Finding.YES:
            errs.append("autopsy: cannot both transfer and be memorised")
        if self.overfit == Finding.NOT_APPLICABLE and self.where_failed.stage in (FailureStage.HOLDOUT, FailureStage.IN_SAMPLE) \
                and self.memorized == Finding.YES:
            errs.append("autopsy: memorised learners are overfit by definition, overfit cannot be NOT_APPLICABLE")
        return errs

    def unknown_fields(self) -> tuple:
        return tuple(n for n in self.FINDINGS if getattr(self, n) == Finding.UNKNOWN)

    @classmethod
    def from_dict(cls, d: Mapping) -> "Autopsy":
        wf = d["where_failed"]
        return cls(attempted=d["attempted"], why_promising=d["why_promising"], what_it_did=d["what_it_did"],
                   where_failed=FailureSite(FailureStage(wf["stage"]), wf.get("detail", ""), Subsystem(wf["subsystem"]) if wf.get("subsystem") else None),
                   **{n: Finding(d.get(n, "UNKNOWN")) for n in cls.FINDINGS}, basis=dict(d.get("basis") or {}), derived=bool(d.get("derived", False)))


_STAGE_OF_MODE = {FailureMode.MEMORISATION: FailureStage.HOLDOUT, FailureMode.NO_TRANSFER: FailureStage.TRANSFER,
                  FailureMode.NO_SKILL: FailureStage.HOLDOUT, FailureMode.NO_EFFECT: FailureStage.DESIGN,
                  FailureMode.HARMFUL: FailureStage.TRANSFER, FailureMode.TRADEOFF_HARM: FailureStage.RISK_GUARD,
                  FailureMode.LEAKAGE: FailureStage.DATA, FailureMode.OVERFIT: FailureStage.HOLDOUT,
                  FailureMode.INFEASIBLE: FailureStage.EXECUTION, FailureMode.UNVALIDATED_CLAIM: FailureStage.VALIDATION}


def _num(vr: Mapping, *keys) -> float | None:
    for k in keys:
        v = vr.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return float(v)
    return None


def derive_autopsy(fl: FailedLearner) -> Autopsy:
    """Best-effort section-17 answers from what the registry already holds. Only findings the numbers actually settle are
    YES/NO; everything else stays UNKNOWN, and the record is flagged `derived` so the lab knows a person has not read it."""
    vr, mode = fl.validation_result, fl.failure_mode
    basis: dict = {}
    overfit = memorized = transferred = risk = Finding.UNKNOWN
    same_lo = _num(vr, "same_year_ci_lo", "ls1_ci_lo")
    tr_lo, tr_hi = _num(vr, "transfer_ci_lo", "gap_ci_lo"), _num(vr, "transfer_ci_hi", "ci90_hi")
    if mode in (FailureMode.MEMORISATION, FailureMode.OVERFIT):
        overfit = memorized = Finding.YES if mode == FailureMode.MEMORISATION else Finding.UNKNOWN
        if mode == FailureMode.OVERFIT:
            overfit = Finding.YES
        basis["overfit"] = basis["memorized"] = f"failure mode {mode}: gain on the tested data, none on unseen data"
    if same_lo is not None and same_lo > 0 and tr_lo is not None and tr_lo <= 0 and mode == FailureMode.NO_TRANSFER:
        memorized, overfit = Finding.YES, Finding.YES
        basis["memorized"] = basis["overfit"] = f"same-year interval lower bound {same_lo:.5f} > 0 while transfer interval includes zero"
    if mode == FailureMode.NO_EFFECT and _num(vr, "adopted") == 0:
        overfit, memorized, transferred = Finding.NOT_APPLICABLE, Finding.NOT_APPLICABLE, Finding.NOT_APPLICABLE
    elif mode in (FailureMode.NO_TRANSFER, FailureMode.MEMORISATION, FailureMode.HARMFUL, FailureMode.OVERFIT):
        transferred = Finding.NO
        basis["transferred"] = f"failure mode {mode}"
    if mode == FailureMode.TRADEOFF_HARM or vr.get("guard_worse") is True or any("tiers_worse" in k for k in vr):
        risk = Finding.YES
        basis["increased_risk"] = next((f"{k}={v}" for k, v in vr.items() if "tiers_worse" in k), "a guard metric worsened")
    elif mode == FailureMode.HARMFUL:
        risk = Finding.UNKNOWN
    gen = {Generalization.GENERALIZED: Finding.YES, Generalization.REGIME_SPECIFIC: Finding.NO,
           Generalization.SINGLE_REGIME: Finding.UNKNOWN, Generalization.UNKNOWN: Finding.UNKNOWN}[fl.generalization]
    if gen.known:
        basis["failure_generalized"] = f"{fl.generalization}: failed in {list(fl.regimes_failed)}, passed in {list(fl.regimes_passed)}"
    site = FailureSite(_STAGE_OF_MODE.get(mode, FailureStage.VALIDATION), fl.reason[:120])
    return Autopsy(attempted=fl.hypothesis, why_promising=f"{NOT_RECORDED} (the hypothesis was: {fl.hypothesis[:100]})",
                   what_it_did=fl.reason, where_failed=site, overfit=overfit, memorized=memorized, transferred=transferred,
                   increased_risk=risk, failure_generalized=gen, basis=basis, derived=True)


def reason_bucket(fl: FailedLearner, autopsy: Autopsy | None = None) -> ReasonBucket:
    """The single primary reason for a failure. Risk beats overfitting only when a guard metric is documented as worse."""
    a = autopsy or derive_autopsy(fl)
    m = fl.failure_mode
    if m == FailureMode.LEAKAGE:
        return ReasonBucket.LEAKAGE
    if m == FailureMode.TRADEOFF_HARM or (m == FailureMode.HARMFUL and a.increased_risk == Finding.YES):
        return ReasonBucket.RISK_CONCENTRATION
    if m in (FailureMode.MEMORISATION, FailureMode.OVERFIT) or a.overfit == Finding.YES:
        return ReasonBucket.OVERFITTING
    return {FailureMode.NO_TRANSFER: ReasonBucket.DID_NOT_TRANSFER, FailureMode.NO_SKILL: ReasonBucket.NO_SIGNAL,
            FailureMode.NO_EFFECT: ReasonBucket.ADOPTED_NOTHING, FailureMode.HARMFUL: ReasonBucket.HARMFUL,
            FailureMode.INFEASIBLE: ReasonBucket.INFEASIBLE, FailureMode.UNVALIDATED_CLAIM: ReasonBucket.UNVERIFIED}.get(m, ReasonBucket.UNKNOWN)


# ------------------------------------------------------------------------------------------------ learner classes

# Declared mechanism taxonomy: a class is a family of ideas, not a name. New tags are welcome; unknown ones fall back to
# similarity with recorded attempts.
CLASSES: dict = {
    "memory_recall": ("stored_numeric_memory", "memory_bank", "exact_path_recall", "episodic_retrieval", "analog_memory",
                      "similarity_lookup", "basis_expansion"),
    "cross_window_rules": ("pool_rank_rules", "regime_bucket_rules", "cfg_knob_search", "band_targeting", "movement_score",
                           "mover_selection", "learner_chain", "cross_year_learner", "past_only_transfer_gate"),
    "veto_lessons": ("veto_rules", "lessons_from_other_windows", "missed_winner_rules", "post_hoc_capture"),
    "direction_models": ("direction_classifier", "mover_features"),
    "deep_sequence": ("sequence_model",),
    "causal_structure": ("causal_discovery",),
}


def classes_of(tags: Iterable[str], taxonomy: Mapping = CLASSES) -> dict:
    """class -> share of the given tags that belong to it. A learner can straddle classes (a chain of veto rules)."""
    ts = set(tags)
    if not ts:
        return {}
    out = {c: len(ts & set(members)) / len(ts) for c, members in taxonomy.items() if ts & set(members)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


@dataclass(frozen=True)
class LabProposal(LearnerProposal):
    """A learner proposal in the lab's terms: what information it uses, which controls it will run, and which recorded failure
    reasons it claims to address. Extends (does not replace) the registry's LearnerProposal."""
    information_sources: tuple = ()
    controls_planned: tuple = ()
    addresses: tuple = ()                   # ReasonBucket values the design says it defeats, with the mechanism in `rationale`
    rationale: str = ""

    def validate(self) -> list:
        errs = []
        if not self.name.strip() or not self.hypothesis.strip():
            errs.append("proposal needs a name and a hypothesis")
        if not self.mechanism_tags:
            errs.append("proposal has no mechanism tags (use failed_learners.proposal_from_text to infer them)")
        for a in self.addresses:
            if a not in {b.value for b in ReasonBucket}:
                errs.append(f"addresses: {a!r} is not a ReasonBucket value")
        if self.addresses and len(self.rationale.split()) < 6:
            errs.append("addresses claims need a rationale of at least a sentence")
        return errs


def attempt_similarity(p: LearnerProposal, f: FailedLearner, taxonomy: Mapping = CLASSES) -> float:
    """0 unrelated .. 1 the same idea. Registry similarity (tags dominate, then wording, then family) lifted when the two share a
    declared class, so a proposal carrying only NEW tags of a known class is still recognised as that class."""
    base = proposal_similarity(p, f)
    shared = set(classes_of(p.mechanism_tags, taxonomy)) & set(classes_of(f.mechanism_tags, taxonomy))
    return float(min(1.0, base + (0.15 if shared else 0.0)))


IN_CLASS_AT = 0.30                 # below this an attempt is not the same class as the proposal


@dataclass(frozen=True)
class Attempt:
    learner: str
    version: int
    failure_mode: FailureMode
    bucket: ReasonBucket
    generalization: Generalization
    similarity: float
    recorded_at: str
    hypothesis: str
    tags: tuple
    regimes: tuple
    derived_autopsy: bool


@dataclass(frozen=True)
class PassRecord:
    learner: str
    hypothesis: str
    mechanism_tags: tuple
    regime: str
    recorded_at: str
    evidence: tuple = ()
    result: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class ClassHistory:
    proposal: str
    classes: Mapping
    attempts: tuple                        # Attempt, most similar first
    passes: tuple                          # PassRecord in the same class
    by_bucket: Mapping
    by_mode: Mapping
    regimes_tried: tuple
    distinct_hypotheses: int
    unknown_reason: int

    @property
    def n(self) -> int:
        return len(self.attempts)


def class_history(lab: "FailedLearnerLab", p: LearnerProposal, now) -> ClassHistory:
    """Every recorded attempt (every version) recorded strictly before `now` that belongs to the proposal's class."""
    cut = to_ts(now)
    attempts: list[Attempt] = []
    buckets: dict[str, int] = {}
    modes: dict[str, int] = {}
    regimes: list[str] = []
    hyps: list[str] = []
    for row in lab.registry._rows:
        if to_ts(row.recorded_at) >= cut:
            continue
        sim = attempt_similarity(p, row, lab.taxonomy)
        shared = set(classes_of(p.mechanism_tags, lab.taxonomy)) & set(classes_of(row.mechanism_tags, lab.taxonomy))
        if sim < IN_CLASS_AT and not shared:
            continue
        aut = lab.autopsy(row.learner, row.version, now)
        b = reason_bucket(row, aut)
        attempts.append(Attempt(row.learner, row.version, row.failure_mode, b, row.generalization, round(sim, 4), row.recorded_at,
                                row.hypothesis, tuple(row.mechanism_tags), tuple(row.data_regime), aut.derived))
        buckets[b.value] = buckets.get(b.value, 0) + 1
        modes[row.failure_mode.value] = modes.get(row.failure_mode.value, 0) + 1
        regimes += [r for r in row.data_regime if not any(same_regime(r, s) for s in regimes)]
        if not any(question_similarity(row.hypothesis, h) >= 0.8 for h in hyps):
            hyps.append(row.hypothesis)
    passes = tuple(x for x in lab.passes(now) if set(x.mechanism_tags) & set(p.mechanism_tags)
                   or set(classes_of(x.mechanism_tags, lab.taxonomy)) & set(classes_of(p.mechanism_tags, lab.taxonomy)))
    attempts.sort(key=lambda a: (-a.similarity, a.learner, a.version))
    return ClassHistory(p.name, classes_of(p.mechanism_tags, lab.taxonomy), tuple(attempts), passes,
                        dict(sorted(buckets.items(), key=lambda kv: -kv[1])), dict(sorted(modes.items(), key=lambda kv: -kv[1])),
                        tuple(regimes), len(hyps), buckets.get(ReasonBucket.UNKNOWN.value, 0))


def class_verdict(h: ClassHistory, min_attempts: int = 3, dominance: float = 0.6, min_regimes: int = 2) -> tuple:
    """(ClassVerdict, why). DEAD_CLASS needs enough attempts, one dominant reason, and failure in >= min_regimes regimes; anything
    thinner is UNDER_EXPLORED or CONTESTED. A recorded pass keeps the class OPEN/CONTESTED: a dead end is a finding, not a law."""
    if h.n == 0:
        return ClassVerdict.OPEN, "no recorded attempt in this class"
    if h.n < min_attempts:
        return ClassVerdict.UNDER_EXPLORED, f"only {h.n} attempt(s); {min_attempts} needed to call a class dead"
    top_bucket, top_n = next(iter(h.by_bucket.items()))
    if h.passes:
        return ClassVerdict.CONTESTED, f"{len(h.passes)} pass(es) on record against {h.n} failures"
    if top_n / h.n >= dominance and len(h.regimes_tried) >= min_regimes and h.distinct_hypotheses >= 2:
        return ClassVerdict.DEAD_CLASS, f"{top_n} of {h.n} attempts failed for one reason ({top_bucket}) across {len(h.regimes_tried)} regimes"
    return ClassVerdict.CONTESTED, f"{h.n} attempts fail for different reasons ({dict(h.by_bucket)}) or in too few regimes ({len(h.regimes_tried)})"


def class_statement(h: ClassHistory) -> str:
    """The section-17 sentence, computed from the record."""
    if h.n == 0:
        return "No learner of this class has failed on record."
    lines = [f"We have already attempted this class of learner {h.n} time{'s' if h.n != 1 else ''}."]
    for bucket, k in h.by_bucket.items():
        lines.append(f"{k} failed {BUCKET_CLAUSE[bucket]}.")
    if h.passes:
        lines.append(f"{len(h.passes)} attempt(s) passed and are on record.")
    lines.append("A new attempt requires a materially different hypothesis.")
    return " ".join(lines)


# ------------------------------------------------------------------------------------------------ material difference

@dataclass(frozen=True)
class MaterialDifference:
    material: bool
    novel_tags: tuple
    nearest_hypothesis_similarity: float
    hypothesis_restated: bool
    controls_missing: Mapping             # failure mode -> controls the earlier failures needed and this plan lacks
    unmet_addresses: tuple
    reasons: tuple

    def summary(self) -> str:
        return "materially different: " + "; ".join(self.reasons) if self.material else "NOT materially different: " + "; ".join(self.reasons)


def material_difference(p: LabProposal, h: ClassHistory, restate_at: float = 0.60) -> MaterialDifference:
    """A retry is materially different only if (a) its hypothesis is not a restatement of any earlier attempt's, (b) it brings a
    mechanism tag or information source the class has never used, and (c) it plans the controls that would have exposed each of
    the class's failure modes. (b) is what separates a new idea from the old one with a new name."""
    if h.n == 0:
        return MaterialDifference(True, tuple(p.mechanism_tags), 0.0, False, {}, (), ("nothing of this class has been attempted",))
    seen_tags = {t for a in h.attempts for t in a.tags}
    novel = tuple(sorted(set(p.mechanism_tags) - seen_tags))
    sims = [question_similarity(p.hypothesis, a.hypothesis) for a in h.attempts]
    nearest = max(sims) if sims else 0.0
    restated = nearest >= restate_at
    missing = {}
    for mode in h.by_mode:
        need = set(REQUIRED_CONTROLS.get(FailureMode(mode), ()))
        lack = sorted(need - set(p.controls_planned))
        if lack:
            missing[mode] = tuple(lack)
    by_bucket = {b.value for b in ReasonBucket}
    unmet = tuple(a for a in p.addresses if a in by_bucket and a not in h.by_bucket)
    reasons = []
    if restated:
        reasons.append(f"hypothesis restates an earlier attempt (similarity {nearest:.2f} >= {restate_at})")
    if not novel and not p.information_sources:
        reasons.append("no mechanism tag or information source the class has not already used")
    elif not novel:
        reasons.append("no new mechanism tag (information sources stated but tags are all previously tried)")
    if missing:
        reasons.append("controls missing for earlier failure modes: " + "; ".join(f"{m}: {list(c)}" for m, c in missing.items()))
    if unmet:
        reasons.append(f"claims to address reasons this class never failed for: {list(unmet)}")
    material = bool(novel or (p.information_sources and not restated)) and not restated and not missing
    if material:
        reasons = [f"new mechanism {list(novel)}" if novel else f"new information {list(p.information_sources)}",
                   f"hypothesis is distinct from every earlier one (nearest {nearest:.2f})", "plans the controls each earlier failure mode needed"]
    return MaterialDifference(material, novel, round(nearest, 4), restated, missing, unmet, tuple(reasons))


@dataclass(frozen=True)
class Consultation:
    proposal: str
    decision: Decision
    statement: str
    verdict: ClassVerdict
    verdict_why: str
    history: ClassHistory
    difference: MaterialDifference
    registry_status: DeadEndStatus
    required_controls: tuple
    reopen_conditions: tuple
    unknown_reasons: int = 0
    label: str = LABEL

    def render(self) -> str:
        lines = [f"[{self.decision}] {self.proposal}: {self.statement}", f"  class verdict: {self.verdict} ({self.verdict_why})",
                 f"  registry: {self.registry_status}", f"  {self.difference.summary()}"]
        if self.required_controls:
            lines.append(f"  required controls: {list(self.required_controls)}")
        lines += [f"  reopens if: {c}" for c in self.reopen_conditions]
        if self.unknown_reasons:
            lines.append(f"  note: {self.unknown_reasons} earlier failure(s) have an undetermined cause; the class statement understates what is known")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the lab

class FailedLearnerLab:
    """Registry + section-17 sidecar. The registry (C62) remains the single store of failed learners; the lab's own append-only
    jsonl holds autopsies, pass records and the consultation log, all read as of a time."""

    def __init__(self, registry: FailedLearnerRegistry | None = None, path=None, taxonomy: Mapping = CLASSES):
        self.registry = registry if registry is not None else FailedLearnerRegistry()   # NOT `or`: an empty registry is falsy
        self.taxonomy = dict(taxonomy)
        self.path = Path(path) if path else None
        self._rows: list = []
        self.unparseable = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row["kind"] not in ("autopsy", "pass", "consult"):
                        raise ValueError(row["kind"])
                    to_ts(row["recorded_at"])
                    self._rows.append(row)
                except (ValueError, KeyError, TypeError):
                    self.unparseable += 1

    def _append(self, kind: str, recorded_at, payload: Mapping) -> None:
        row = {"kind": kind, "recorded_at": em._iso(recorded_at), **json.loads(canonical_json(payload))}
        self._rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    # -------------------------------------------------------------- recording
    def record_failure(self, fl: FailedLearner, autopsy: Autopsy | None, now=None, retest: bool = False) -> FailedLearner:
        """Add a failed learner (or, with retest=True, a superseding version) together with its autopsy. Both are validated
        before either is written; an autopsy that contradicts the registry record is refused."""
        aut = autopsy or derive_autopsy(fl)
        errs = aut.validate() + self.consistency(fl, aut)
        if errs:
            raise ValueError(f"lab entry for {fl.learner!r} invalid: " + "; ".join(errs))
        stored = self.registry.retest(fl.learner, fl) if retest else self.registry.add(fl)
        self._append("autopsy", stored.recorded_at, {"learner": stored.learner, "version": stored.version,
                                                       "autopsy": json.loads(canonical_json(aut))})
        return stored

    @staticmethod
    def consistency(fl: FailedLearner, aut: Autopsy) -> list:
        """The autopsy may not contradict the failure mode the harness recorded."""
        errs = []
        if fl.failure_mode == FailureMode.MEMORISATION and aut.memorized in (Finding.NO, Finding.NOT_APPLICABLE):
            errs.append("failure mode MEMORISATION but autopsy says it did not memorise")
        if fl.failure_mode in (FailureMode.NO_TRANSFER, FailureMode.MEMORISATION, FailureMode.HARMFUL) and aut.transferred == Finding.YES:
            errs.append(f"failure mode {fl.failure_mode} but autopsy says it transferred")
        if fl.failure_mode == FailureMode.TRADEOFF_HARM and aut.increased_risk in (Finding.NO, Finding.NOT_APPLICABLE):
            errs.append("failure mode TRADEOFF_HARM but autopsy says risk did not increase")
        if fl.failure_mode == FailureMode.OVERFIT and aut.overfit in (Finding.NO, Finding.NOT_APPLICABLE):
            errs.append("failure mode OVERFIT but autopsy says it did not overfit")
        if fl.generalization == Generalization.GENERALIZED and aut.failure_generalized in (Finding.NO, Finding.NOT_APPLICABLE):
            errs.append("registry says GENERALIZED but autopsy says the failure did not generalise")
        return errs

    def record_pass(self, p: PassRecord) -> PassRecord:
        """A learner in this class that PASSED. Recorded so class statistics are not survivor-biased in the other direction."""
        if not p.mechanism_tags or not p.evidence:
            raise ValueError("a pass record needs mechanism tags and evidence (a claimed pass without evidence is a claim)")
        self._append("pass", p.recorded_at, {"pass": json.loads(canonical_json(p))})
        return p

    # -------------------------------------------------------------- reading
    def autopsy(self, learner: str, version: int, now) -> Autopsy:
        cut = to_ts(now)
        best = None
        for r in self._rows:
            if r["kind"] == "autopsy" and r["learner"] == learner and r["version"] == version and to_ts(r["recorded_at"]) < cut:
                best = r
        if best is not None:
            return Autopsy.from_dict(best["autopsy"])
        row = next((x for x in self.registry._rows if x.learner == learner and x.version == version), None)
        if row is None:
            raise KeyError(f"unknown learner/version {learner}/{version}")
        return derive_autopsy(row)

    def passes(self, now) -> list:
        cut = to_ts(now)
        out = []
        for r in self._rows:
            if r["kind"] == "pass" and to_ts(r["recorded_at"]) < cut:
                d = r["pass"]
                out.append(PassRecord(d["learner"], d["hypothesis"], tuple(d["mechanism_tags"]), d["regime"], d["recorded_at"],
                                      tuple(d.get("evidence", ())), dict(d.get("result", {}))))
        return out

    def consultations(self, now) -> list:
        cut = to_ts(now)
        return [r for r in self._rows if r["kind"] == "consult" and to_ts(r["recorded_at"]) < cut]

    # -------------------------------------------------------------- the consultation
    def consult(self, p: LabProposal, now, justification: Justification | None = None, log: bool = True) -> Consultation:
        """Ask the record of failures before writing a learner. Never raises for a bad idea (see `require`); a malformed
        proposal is refused with ValueError, because an unclassifiable idea cannot be checked."""
        errs = p.validate()
        if errs:
            raise ValueError("proposal invalid: " + "; ".join(errs))
        h = class_history(self, p, now)
        verdict, why = class_verdict(h)
        reg = self.registry.check_proposal(p, now, justification)
        diff = material_difference(p, h)
        modes = [FailureMode(m) for m in h.by_mode]
        needed = tuple(dict.fromkeys(c for m in modes for c in REQUIRED_CONTROLS.get(m, ())))
        if h.n == 0 and reg.status == DeadEndStatus.CLEAR:
            decision = Decision.PROCEED
        elif reg.status == DeadEndStatus.BLOCKED_DEAD_END and not diff.material:
            decision = Decision.BLOCKED
        elif diff.material:
            decision = Decision.PROCEED_WITH_CONTROLS
        elif verdict == ClassVerdict.DEAD_CLASS:
            decision = Decision.BLOCKED
        else:
            decision = Decision.NEEDS_DIFFERENT_HYPOTHESIS
        c = Consultation(p.name, decision, class_statement(h), verdict, why, h, diff, reg.status, needed,
                         self.reopen_conditions(h, p, verdict), h.unknown_reason)
        if log:
            self._append("consult", now, {"proposal": p.name, "proposal_hash": stable_hash([p.hypothesis, sorted(p.mechanism_tags)], 12),
                                          "decision": decision.value, "class_attempts": h.n, "verdict": verdict.value})
        return c

    def require(self, p: LabProposal, now, justification: Justification | None = None) -> Consultation:
        c = self.consult(p, now, justification)
        if not c.decision.permits:
            raise MateriallySameAttempt(c)
        return c

    def reopen_conditions(self, h: ClassHistory, p: LearnerProposal, verdict: ClassVerdict) -> tuple:
        """What would make a dead class worth another attempt. A class is never closed without saying how it reopens."""
        if h.n == 0 or verdict == ClassVerdict.OPEN:
            return ()
        out = []
        known_tags = {t for a in h.attempts for t in a.tags}
        out.append(f"a mechanism outside {sorted(known_tags)[:6]}")
        if h.unknown_reason:
            out.append(f"an explanation of the {h.unknown_reason} failure(s) whose cause is undetermined")
        if h.attempts and all(a.derived_autopsy for a in h.attempts):
            out.append("a human-written autopsy for at least one attempt (every current answer is derived from summary numbers)")
        out.append("a regime none of the attempts covered (tried: " + "; ".join(r[:40] for r in h.regimes_tried[:4]) + ")")
        for mode in h.by_mode:
            out.append(f"controls for {mode}: {list(REQUIRED_CONTROLS.get(FailureMode(mode), ()))}")
        return tuple(out)

    # -------------------------------------------------------------- knowledge extracted from the lab
    def class_table(self, now) -> list:
        """One row per declared class: attempts, reason mix, passes, verdict. The lab's overview."""
        rows = []
        for cname, members in self.taxonomy.items():
            probe = LabProposal(cname, cname, tuple(members))
            h = class_history(self, probe, now)
            v, why = class_verdict(h)
            rows.append({"class": cname, "attempts": h.n, "by_bucket": dict(h.by_bucket), "passes": len(h.passes),
                         "verdict": v.value, "why": why, "unknown_reasons": h.unknown_reason, "regimes": len(h.regimes_tried)})
        return rows

    def rediscovery_prevented(self, now) -> dict:
        """How many consultations ended in a redirect rather than a launch: the measurable value of the lab."""
        rows = self.consultations(now)
        redirected = [r for r in rows if r["decision"] in (Decision.BLOCKED.value, Decision.NEEDS_DIFFERENT_HYPOTHESIS.value)]
        return {"consultations": len(rows), "redirected": len(redirected), "proceeded": len(rows) - len(redirected),
                "share_redirected": (len(redirected) / len(rows)) if rows else 0.0}


# ------------------------------------------------------------------------------------------------ seeded autopsies

def _a(attempted, why, did, stage, detail, **kw):
    return Autopsy(attempted=attempted, why_promising=why, what_it_did=did, where_failed=FailureSite(stage, detail, kw.pop("sub", None)), **kw)


def seed_autopsies() -> dict:
    """Written (not derived) autopsies for the failed learners the repository already proved. Numbers come from each record's
    validation_result; where a report did not measure something the finding is UNKNOWN."""
    Y, N, U, NA = Finding.YES, Finding.NO, Finding.UNKNOWN, Finding.NOT_APPLICABLE
    return {
        "memory_bank": _a("Recall stored window-level numeric memories for a similar market", "Same-year effect was significantly positive: +0.23 percent per week, CI above zero",
                          "Recognised a disguised rerun of the same year and replayed its own numbers; no gain on unseen years", FailureStage.HOLDOUT,
                          "gain on the tested year, none on unseen years", sub=Subsystem.SELECTION, overfit=Y, memorized=Y, transferred=N, increased_risk=U,
                          failure_generalized=Y, basis={"overfit": "same_year_ci_lo > 0 with transfer_mean_week -0.03 percent", "memorized": "gap CI lower bound 0.000124 > 0: same-year exceeds transfer",
                                                        "transferred": "transfer_mean_week = -0.0003211", "failure_generalized": "failed in both pair sets ld1/ld2 and ls1/ls2"}),
        "basis_learner": _a("Learn a basis over window features so memory generalises", "Basis expansion is the standard cure for a lookup that only recalls",
                            "Same-year effect 0.093 percent per week, transfer -0.002 percent with a CI around zero: what it learned was still the tested years",
                            FailureStage.TRANSFER, "transfer CI (-0.034, 0.030) percent", sub=Subsystem.SELECTION, overfit=Y, memorized=Y, transferred=N,
                            basis={"overfit": "same_year_ci_lo 0.000205 > 0 while transfer CI includes zero", "memorized": "harness verdict MEMORISATION", "transferred": "transfer_ci includes zero, p_signflip 1.0"}),
        "episodic_memory_retrieval": _a("Retrieve similar past situations to predict the next decision", "Retrieved neighbours reproduce their own outcomes very well in-sample",
                                        "Walk-forward skill -0.16 (CI -0.20..-0.12): significantly worse than no memory; sign flip not explained",
                                        FailureStage.HOLDOUT, "walk-forward skill negative", sub=Subsystem.SELECTION, transferred=N, overfit=U, memorized=U,
                                        basis={"transferred": "walk_forward_skill CI entirely below zero"}),
        "missed_winner_detector": _a("Explain missed winners and turn the explanation into a capture rule", "Post-mortems of the biggest misses looked systematic",
                                     "Uplift over a control detector was zero (p 0.64): the misses were not predictable from what it inspects", FailureStage.DESIGN,
                                     "uplift 0 vs control", sub=Subsystem.SELECTION, overfit=U, memorized=U, transferred=U, failure_generalized=U),
        "lessons_from_other_windows": _a("Distil veto lessons from other windows to protect a new one", "Vetoing repeat losers ought to cut losses",
                                         "Cost 0.19 percent per week (CI -0.34..-0.02): vetoes removed names that would have done fine", FailureStage.TRANSFER,
                                         "weekly effect entirely below zero", sub=Subsystem.SELECTION, transferred=N, overfit=Y, memorized=U, increased_risk=U,
                                         basis={"transferred": "ci90_hi -0.00019 < 0", "overfit": "lessons fit the source windows and reverse elsewhere (negative transfer)"}),
        "band_cfg": _a("Choose configuration knobs across years to raise the in-band share", "In-band transfer was positive in both pair sets: +1.1 and +2.1 points",
                       "Raised the share of weeks in the 5-10 percent band but worsened worst-5-percent week and max drawdown in the replication", FailureStage.RISK_GUARD,
                       "tiers worse: worst5, max_dd", sub=Subsystem.RISK, overfit=N, memorized=N, transferred=Y, increased_risk=Y, failure_generalized=N,
                       basis={"overfit": "in-band transfer CI above zero on unseen years", "memorized": "gain persisted on unseen years", "transferred": "ls1 and ls3 CIs above zero",
                              "increased_risk": "ls3_tiers_worse = worst5,max_dd", "failure_generalized": "regimes_passed=[ls1]; failed only in ls3"}),
        "band_pool": _a("Keep the window pool where the portfolio lands in band", "Rank windows on volatility, max20, p_move looked separable", "The adoption gate never opened; nothing adopted",
                        FailureStage.DESIGN, "adopted = 0", sub=Subsystem.SELECTION, overfit=NA, memorized=NA, transferred=NA, failure_generalized=Y,
                        basis={"failure_generalized": "adopted nothing in both pair sets"}),
        "regime_map": _a("Learn a market-regime to pool-rule map with shrinkage", "Regimes plausibly select different pools", "No regime bucket rule cleared the cross-year gate; cause undetermined",
                         FailureStage.DESIGN, "adopted = 0", sub=Subsystem.SELECTION, overfit=NA, memorized=NA, transferred=NA, failure_generalized=Y,
                         basis={"failure_generalized": "adopted nothing in both pair sets"}),
        "lesson_learner": _a("Veto names in the top decile of a feature that lost in many years", "Repeated multi-year losses look like a lesson", "No lesson held in enough distinct years",
                             FailureStage.DESIGN, "adopted = 0", sub=Subsystem.SELECTION, overfit=NA, memorized=NA, transferred=NA, failure_generalized=Y,
                             basis={"failure_generalized": "adopted nothing in both pair sets"}),
        "mover_use": _a("Fit a movement score on realised absolute move across years", "Movement is the one thing with a measured edge", "Never beat the incumbent p_move under the sign-consistency gate",
                        FailureStage.DESIGN, "adopted = 0", sub=Subsystem.SELECTION, overfit=NA, memorized=NA, transferred=NA, failure_generalized=Y,
                        basis={"failure_generalized": "adopted nothing in both pair sets"}),
        "chain_learner": _a("Chain the five single learners so their transfer compounds", "Small independent gains would add", "Components that adopt nothing cannot compound",
                            FailureStage.DESIGN, "adopted = 0", sub=Subsystem.SELECTION, overfit=NA, memorized=NA, transferred=NA, failure_generalized=Y,
                            basis={"failure_generalized": "adopted nothing in both pair sets"}),
        "cross_year_learners_all": _a("Any of six cross-year learners passes past-only transfer", "Six different mechanisms gave six chances", "Zero of six pass in three independent pair sets",
                                      FailureStage.TRANSFER, "transfer_passing_learners = 0 of 6", sub=Subsystem.SELECTION, transferred=N, overfit=U, memorized=U, failure_generalized=Y,
                                      basis={"transferred": "transfer_passing_learners = 0", "failure_generalized": "independent_replications = 3, headline NO_TRANSFER every time"}),
        "direction_on_movers": _a("Predict the direction of a mover's next week", "Movers are large moves so any signal would be large", "Accuracy 51.8 percent; Brier equals the base rate; the 80 percent gate never opened",
                                  FailureStage.HOLDOUT, "accuracy 0.518 vs base rate", sub=Subsystem.DIRECTION, overfit=U, memorized=U, transferred=U, increased_risk=U),
    }


def seed_lab(lab: FailedLearnerLab, recorded_at: str = "2026-09-29T00:00:00") -> dict:
    """Load the historical failures and their written autopsies (registry seeds first, so nothing is duplicated)."""
    have = {r.learner for r in lab.registry._rows}
    autopsies = seed_autopsies()
    added = 0
    for f in fl_mod.seed_history(recorded_at):
        if f.learner in have:
            continue
        aut = autopsies.get(f.learner)
        lab.record_failure(f, aut)
        added += 1
    return {"added": added, "written_autopsies": sum(1 for f in lab.registry.as_of(recorded_at_plus(recorded_at)) if f.learner in autopsies),
            "label": LABEL}


def recorded_at_plus(ts: str) -> str:
    """The instant just after `ts`, so as_of(now) includes records written at `ts`."""
    return (to_ts(ts) + dt.timedelta(seconds=1)).isoformat()


# ------------------------------------------------------------------------------------------------ health and reports

@dataclass(frozen=True)
class LabHealth:
    n_learners: int
    n_attempts: int
    derived_autopsies: int
    unknown_findings: Mapping
    unknown_reasons: int
    evidence_missing: tuple
    classes_without_pass_records: tuple
    consult_redirect_rate: float
    ok: bool
    notes: tuple

    def render(self) -> str:
        return "\n".join([f"lab: {self.n_learners} learners, {self.n_attempts} attempts, ok={self.ok}", f"  derived (unread) autopsies: {self.derived_autopsies}",
                          f"  unknown findings: {dict(self.unknown_findings)}", f"  undetermined causes: {self.unknown_reasons}"] + [f"  note: {n}" for n in self.notes])


def lab_health(lab: FailedLearnerLab, now, root=None) -> LabHealth:
    """Honesty check on the lab itself: how much of the memory is inferred rather than examined, how many findings are UNKNOWN,
    whether evidence files exist, and whether any class is described by failures only."""
    cur = lab.registry.as_of(now)
    auts = {(f.learner, f.version): lab.autopsy(f.learner, f.version, now) for f in cur}
    unknown: dict = {}
    for a in auts.values():
        for n in a.unknown_fields():
            unknown[n] = unknown.get(n, 0) + 1
    ev = lab.registry.verify_evidence(root, now) if root is not None else {"unverified_learners": []}
    unk_reason = sum(1 for f in cur if f.reason.strip().upper().startswith("UNKNOWN"))
    table = lab.class_table(now)
    no_pass = tuple(r["class"] for r in table if r["attempts"] > 0 and r["passes"] == 0)
    rate = lab.rediscovery_prevented(now)["share_redirected"]
    notes = []
    derived = sum(1 for a in auts.values() if a.derived)
    if cur and derived / len(cur) > 0.5:
        notes.append(f"{derived} of {len(cur)} autopsies are derived from summary numbers, not written from the evidence")
    if unknown and cur and max(unknown.values()) / len(cur) > 0.5:
        worst = max(unknown, key=lambda n: unknown[n])
        notes.append(f"'{worst}' is UNKNOWN for {unknown[worst]} of {len(cur)} learners")
    if no_pass:
        notes.append(f"classes described by failures only (no pass record exists to compare): {list(no_pass)}")
    if ev["unverified_learners"]:
        notes.append(f"evidence files missing for: {ev['unverified_learners']}")
    return LabHealth(len(cur), len(lab.registry._rows), derived, unknown, unk_reason, tuple(ev["unverified_learners"]), no_pass, rate,
                     not ev["unverified_learners"] and lab.unparseable == 0, tuple(notes))


def render_lab(lab: FailedLearnerLab, now) -> str:
    lines = [f"Failed-learner lab as of {em._iso(now)}   [{LABEL}]"]
    for r in lab.class_table(now):
        mix = ", ".join(f"{k}: {v}" for k, v in r["by_bucket"].items()) or "none"
        lines.append(f"- {r['class']}: {r['attempts']} attempts ({mix}); {r['passes']} pass(es); {r['verdict']}")
    return "\n".join(lines)


def to_markdown(lab: FailedLearnerLab, now) -> str:
    """The lab as a document: one autopsy per failed learner, then the class table. Read before building a learner."""
    lines = [f"# Failed-learner lab as of {em._iso(now)}", ""]
    for f in lab.registry.as_of(now):
        a = lab.autopsy(f.learner, f.version, now)
        lines += [f"## {f.learner} (v{f.version}) - {f.failure_mode}", "", f"- attempted: {a.attempted}", f"- why promising: {a.why_promising}",
                  f"- what it did: {a.what_it_did}", f"- failed at: {a.where_failed.stage} ({a.where_failed.detail})",
                  f"- overfit {a.overfit}; memorised {a.memorized}; transferred {a.transferred}; increased risk {a.increased_risk}; failure generalised {a.failure_generalized}",
                  f"- evidence: {', '.join(f.evidence_paths) or NOT_RECORDED}", ""]
    lines += ["## Classes", ""] + [f"- {r['class']}: {r['attempts']} attempts, {r['verdict']}" for r in lab.class_table(now)]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ from experiments, and the namespace bridge

_MODE_OF_CAUSE = {"FALSE_PATTERN": FailureMode.NO_SKILL, "REGIME_CHANGE": FailureMode.NO_TRANSFER, "WRONG_CONTEXT": FailureMode.NO_TRANSFER,
                  "WEAKENING_EFFECT": FailureMode.NO_TRANSFER, "REVERSAL": FailureMode.HARMFUL, "MEASUREMENT_ERROR": FailureMode.UNVALIDATED_CLAIM,
                  "RISK_ERROR": FailureMode.TRADEOFF_HARM, "INSUFFICIENT_EVIDENCE": FailureMode.NO_EFFECT}


def lab_entry_from_experiment(mem, lab: FailedLearnerLab, experiment_id: str, learner: str, mechanism_tags: Sequence[str], regime: str,
                              implementation: str, now, family: str = "", evidence: Sequence[str] = ()) -> FailedLearner:
    """Turn a closed REFUTED experiment (engine.research.experiments) into a lab entry. The failure mode comes from the experiment's
    recorded outcome (memorisation FAILED -> MEMORISATION, transfer FAILED -> NO_TRANSFER, else the mapped failure cause), never
    from the caller's opinion. Anything the experiment did not measure stays UNKNOWN in the autopsy."""
    from engine.research.experiments import TestState
    r = mem.ledger.get(experiment_id, now)
    o = mem.store.outcome(experiment_id, now)
    if r is None or r.result is None or r.result.kind != em.ResultKind.REFUTED or o is None:
        raise ValueError(f"{experiment_id} is not a closed REFUTED experiment with an outcome extension at {now}")
    if o.memorization.state == TestState.FAILED:
        mode = FailureMode.MEMORISATION
    elif o.transfer.state == TestState.FAILED:
        mode = FailureMode.NO_TRANSFER
    else:
        mode = _MODE_OF_CAUSE.get(str(o.failure_mode), FailureMode.NO_SKILL) if o.failure_mode else FailureMode.NO_SKILL
    val = {"experiment": experiment_id, "result_kind": str(r.result.kind), "n": r.result.n}
    if r.result.ci:
        val["ci_lo"], val["ci_hi"] = r.result.ci
    if o.transfer.ci:
        val["transfer_ci_lo"], val["transfer_ci_hi"] = o.transfer.ci
    if o.memorization.same_period_effect is not None:
        val["same_year_effect"] = o.memorization.same_period_effect
    fl = FailedLearner(learner=learner, hypothesis=r.current_belief, implementation=implementation, failure_mode=mode, data_regime=(regime,),
                       validation_result=val, reason=(r.learned[0] if r.learned else (o.failure_note or "UNKNOWN")),
                       generalization=fl_mod.classify_generalization((regime,), (), 1), recorded_at=r.result.observed_at, mechanism_tags=tuple(mechanism_tags),
                       family=family, regimes_failed=(regime,), evidence_paths=tuple(evidence))
    aut = derive_autopsy(fl)
    aut = dataclasses.replace(aut, why_promising=r.prediction.statement, attempted=r.question,
                              transferred=Finding.NO if o.transfer.state == TestState.FAILED else aut.transferred,
                              memorized=Finding.YES if o.memorization.state == TestState.FAILED else aut.memorized,
                              basis={**aut.basis, **({"transferred": "experiment outcome: transfer FAILED"} if o.transfer.state == TestState.FAILED else {}),
                                     **({"memorized": "experiment outcome: memorisation check FAILED"} if o.memorization.state == TestState.FAILED else {})})
    return lab.record_failure(fl, aut, now)


def learner_launch_gate(mem, lab: FailedLearnerLab, rec, ext, proposal: LabProposal, now, claims=(), difference_statement: str = "",
                        justification: Justification | None = None) -> dict:
    """Both memories must agree before a learner experiment starts: the failed-learner lab (class attempted N times) and the
    experiment memory (have we run this exact test). Returns both assessments and a single `permitted` flag; writes nothing."""
    consult = lab.consult(proposal, now, justification, log=False)
    launch = mem.launch_gate(rec, ext, now, claims, difference_statement)
    return {"permitted": consult.decision.permits and launch.decision.permits_launch, "lab": consult, "experiment": launch,
            "reasons": tuple(x for x in (None if consult.decision.permits else consult.statement,
                                         None if launch.decision.permits_launch else launch.message) if x)}


def matured_record(lab: FailedLearnerLab, learner: str, now) -> MaturedRecord:
    """A failed learner as a research-world record: usable by a decision only through record.gate(now)."""
    f = lab.registry.get(learner, now)
    if f is None:
        raise KeyError(f"{learner} not visible at {now}")
    aut = lab.autopsy(f.learner, f.version, now)
    seen = f.recorded_at[:10]
    prov = Provenance(created_real=f.recorded_at, learned_at=seen, code_hash=f.code_hash or "unrecorded", experiment_id=f.learner,
                      outcomes_seen_through=seen)
    return MaturedRecord(f"failed:{f.learner}:v{f.version}", f.recorded_at, {"mode": f.failure_mode.value, "bucket": reason_bucket(f, aut).value,
                                                                                "generalized": f.generalization.value}, prov)


# ------------------------------------------------------------------------------------------------ the loop's sweep

@dataclass(frozen=True)
class LabStep:
    now: str
    health: LabHealth
    classes: tuple
    dead_classes: tuple
    reopenable: tuple
    audit: Mapping
    redirected: Mapping
    label: str = LABEL


def step(lab: FailedLearnerLab, now, root=None) -> LabStep:
    """The research loop's call: lab health, per-class verdicts, dead classes with their reopening conditions, and the registry's
    own integrity audit. Read-only."""
    table = lab.class_table(now)
    dead = tuple(r["class"] for r in table if r["verdict"] == ClassVerdict.DEAD_CLASS.value)
    reopen = []
    for r in table:
        if r["verdict"] in (ClassVerdict.DEAD_CLASS.value, ClassVerdict.CONTESTED.value):
            members = lab.taxonomy[r["class"]]
            probe = LabProposal(r["class"], r["class"], tuple(members))
            h = class_history(lab, probe, now)
            reopen.append({"class": r["class"], "conditions": list(lab.reopen_conditions(h, probe, ClassVerdict(r["verdict"])))})
    return LabStep(em._iso(now), lab_health(lab, now, root), tuple(table), dead, tuple(reopen), fl_mod.audit_registry(lab.registry, now, root),
                   lab.rediscovery_prevented(now))


# ------------------------------------------------------------------------------------------------ what the failures teach in aggregate

def stage_profile(lab: FailedLearnerLab, now) -> dict:
    """Where learners die: failures per stage and per pipeline subsystem. A pile-up at HOLDOUT means ideas are fit before they are
    tested; a pile-up at DESIGN means gates that cannot open. Subsystem is None when the autopsy did not say."""
    stages: dict = {}
    subs: dict = {}
    for f in lab.registry.as_of(now):
        a = lab.autopsy(f.learner, f.version, now)
        stages[a.where_failed.stage.value] = stages.get(a.where_failed.stage.value, 0) + 1
        k = a.where_failed.subsystem.value if a.where_failed.subsystem else "UNSPECIFIED"
        subs[k] = subs.get(k, 0) + 1
    n = sum(stages.values())
    top = max(stages, key=lambda s: stages[s]) if stages else ""
    return {"n": n, "by_stage": dict(sorted(stages.items(), key=lambda kv: -kv[1])), "by_subsystem": dict(sorted(subs.items(), key=lambda kv: -kv[1])),
            "dominant_stage": top, "dominant_share": (stages[top] / n) if n else 0.0}


def control_gaps(lab: FailedLearnerLab, now) -> dict:
    """Controls that earlier failure modes require, counted by how many recorded failures needed each: the checklist every new
    learner should meet before anything else."""
    need: dict = {}
    for f in lab.registry.as_of(now):
        for c in REQUIRED_CONTROLS.get(f.failure_mode, ()):
            need[c] = need.get(c, 0) + 1
    return dict(sorted(need.items(), key=lambda kv: (-kv[1], kv[0])))


def attempt_trend(lab: FailedLearnerLab, p: LearnerProposal, now) -> dict:
    """Are successive attempts in a class actually new? A REDISCOVERY is an attempt whose mechanism tags were all already tried and
    whose hypothesis restates an earlier one. The rate is the direct measure of infinite rediscovery of dead ends."""
    h = class_history(lab, p, now)
    ordered = sorted(h.attempts, key=lambda a: (a.recorded_at, a.learner, a.version))
    seen_tags: set = set()
    seen_hyp: list = []
    rediscoveries = []
    for a in ordered:
        if seen_hyp and set(a.tags) <= seen_tags and any(question_similarity(a.hypothesis, x) >= 0.6 for x in seen_hyp):
            rediscoveries.append(a.learner)
        seen_tags |= set(a.tags)
        seen_hyp.append(a.hypothesis)
    later = max(0, len(ordered) - 1)
    return {"attempts": len(ordered), "rediscoveries": rediscoveries, "rediscovery_rate": (len(rediscoveries) / later) if later else 0.0,
            "new_tags_per_attempt": (len(seen_tags) / len(ordered)) if ordered else 0.0}


@dataclass(frozen=True)
class RetryPlan:
    proposal: str
    add_controls: tuple
    regimes_to_use: tuple
    regimes_to_avoid_repeating: tuple
    hypothesis_guidance: tuple
    untried_tags: tuple


def plan_retry(lab: FailedLearnerLab, p: LabProposal, now, universe_regimes: Iterable[str] = (), candidate_tags: Iterable[str] = ()) -> RetryPlan:
    """What a serious retry would have to change, computed from the class history: the controls its plan lacks, regimes nobody has
    tried, and tags in the vocabulary the class has never used. A guide for the proposer, not a permission."""
    h = class_history(lab, p, now)
    diff = material_difference(p, h)
    add = tuple(dict.fromkeys(c for cs in diff.controls_missing.values() for c in cs))
    tried = list(h.regimes_tried)
    fresh = tuple(r for r in universe_regimes if not any(same_regime(r, t) for t in tried))
    seen_tags = {t for a in h.attempts for t in a.tags}
    guidance = []
    if diff.hypothesis_restated:
        guidance.append(f"restate the hypothesis as a different causal claim: it is {diff.nearest_hypothesis_similarity:.2f} similar to an earlier one")
    for bucket, k in list(h.by_bucket.items())[:3]:
        guidance.append(f"explain how the design defeats '{bucket}' ({k} of {h.n} earlier failures)")
    if h.unknown_reason:
        guidance.append("state which of the undetermined earlier failures this design would also have suffered")
    return RetryPlan(p.name, add, fresh, tuple(tried), tuple(guidance), tuple(sorted(set(candidate_tags) - seen_tags - set(p.mechanism_tags))))


_CAUSE_OF_BUCKET = {ReasonBucket.OVERFITTING: "FALSE_PATTERN", ReasonBucket.DID_NOT_TRANSFER: "WRONG_CONTEXT", ReasonBucket.RISK_CONCENTRATION: "RISK_ERROR",
                    ReasonBucket.NO_SIGNAL: "FALSE_PATTERN", ReasonBucket.ADOPTED_NOTHING: "INSUFFICIENT_EVIDENCE",
                    ReasonBucket.HARMFUL: "REVERSAL", ReasonBucket.LEAKAGE: "MEASUREMENT_ERROR", ReasonBucket.INFEASIBLE: "INSUFFICIENT_EVIDENCE",
                    ReasonBucket.UNVERIFIED: "MEASUREMENT_ERROR", ReasonBucket.UNKNOWN: "UNKNOWN"}


def failure_cause_of(bucket: ReasonBucket):
    """The research brain's FailureCause for a lab reason, so failed learners and failed patterns speak one vocabulary. A
    bucket the lab cannot resolve maps to UNKNOWN, which is an answer."""
    from engine.learning.core import FailureCause
    return FailureCause(_CAUSE_OF_BUCKET[bucket])


def proposal_from_description(name: str, description: str, controls_planned: Sequence[str] = (), information_sources: Sequence[str] = (),
                              addresses: Sequence[str] = (), rationale: str = "", regime: Sequence[str] = (), family: str = "") -> LabProposal:
    """A LabProposal whose mechanism tags are inferred from free text with the registry's vocabulary, so an idea cannot escape the
    class lookup by being described in new words. An untaggable description is refused (there is nothing to look up)."""
    base = fl_mod.proposal_from_text(name, description, family, regime)
    return LabProposal(base.name, base.hypothesis, base.mechanism_tags, base.family, base.data_regime, tuple(information_sources),
                       tuple(controls_planned), tuple(addresses), rationale)


# ------------------------------------------------------------------------------------------------ integrity of the lab itself

def audit_lab(lab: FailedLearnerLab, now) -> dict:
    """Do the registry and the lab sidecar agree, and does every autopsy still fit its record? Orphaned autopsies, versions with no
    written autopsy (they fall back to a derived one), and stored autopsies that contradict their registry record."""
    cut = to_ts(now)
    keys = {(r.learner, r.version): r for r in lab.registry._rows if to_ts(r.recorded_at) < cut}
    written = {}
    for row in lab._rows:
        if row["kind"] == "autopsy" and to_ts(row["recorded_at"]) < cut:
            written[(row["learner"], row["version"])] = row
    problems = []
    for k in sorted(set(written) - set(keys)):
        problems.append(f"autopsy for {k[0]} v{k[1]} has no registry record")
    for k, row in written.items():
        if k in keys:
            a = Autopsy.from_dict(row["autopsy"])
            problems += [f"{k[0]} v{k[1]}: {e}" for e in a.validate() + lab.consistency(keys[k], a)]
    unwritten = sorted(f"{k[0]} v{k[1]}" for k in set(keys) - set(written))
    return {"ok": not problems, "problems": problems, "derived_only": unwritten, "n_records": len(keys), "n_autopsies": len(written),
            "unparseable_lines": lab.unparseable}


# ------------------------------------------------------------------------------------------------ discovering classes from the data

def discover_classes(lab: FailedLearnerLab, now, link_at: float = 0.35) -> list:
    """Classes found from the failures themselves rather than the declared taxonomy: attempts are linked when their tags and
    hypotheses are similar enough (single linkage). Returns groups with their dominant tags and reasons. An attempt with no
    declared class is reported in `unclassified`, which is how the taxonomy learns where it has a gap."""
    rows = [r for r in lab.registry._rows if to_ts(r.recorded_at) < to_ts(now)]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            sim = attempt_similarity(LearnerProposal(a.learner, a.hypothesis, a.mechanism_tags, a.family, a.data_regime), b, lab.taxonomy)
            if sim >= link_at:
                parent[find(j)] = find(i)
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    out = []
    for members in groups.values():
        tags: dict = {}
        for m in members:
            for t in m.mechanism_tags:
                tags[t] = tags.get(t, 0) + 1
        buckets: dict = {}
        for m in members:
            b = reason_bucket(m, lab.autopsy(m.learner, m.version, now)).value
            buckets[b] = buckets.get(b, 0) + 1
        declared = {c for m in members for c in classes_of(m.mechanism_tags, lab.taxonomy)}
        out.append({"attempts": len(members), "learners": sorted({m.learner for m in members}),
                    "top_tags": [t for t, _ in sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))[:4]],
                    "by_bucket": dict(sorted(buckets.items(), key=lambda kv: -kv[1])), "declared_classes": sorted(declared),
                    "unclassified": not declared})
    return sorted(out, key=lambda g: (-g["attempts"], g["learners"]))


def screen_proposals(lab: FailedLearnerLab, proposals: Sequence[LabProposal], now) -> list:
    """Consult the lab for several proposals at once, without logging. Proposals in the same batch that are the same idea
    (registry similarity above the block threshold) are flagged, so one submission cannot smuggle a dead end in twice."""
    out = []
    for i, p in enumerate(proposals):
        c = lab.consult(p, now, log=False)
        twin = next((q.name for q in proposals[:i] if proposal_similarity(p, FailedLearner(
            q.name, q.hypothesis, "batch", FailureMode.NO_EFFECT, ("batch",), {"n": 1}, "batch", Generalization.UNKNOWN, "1970-01-01T00:00:00",
            tuple(q.mechanism_tags), q.family)) >= 0.5), "")
        out.append({"proposal": p.name, "decision": c.decision, "permitted": c.decision.permits and not twin, "duplicate_of_batch_member": twin,
                    "statement": c.statement})
    return out


def explain(c: Consultation) -> str:
    """A consultation as prose for the person who asked: the class statement, why the verdict, what to change."""
    lines = [c.statement, f"Verdict on the class: {c.verdict} - {c.verdict_why}.", c.difference.summary() + "."]
    if c.required_controls and not c.decision.permits:
        lines.append(f"Before any retry, include: {', '.join(c.required_controls)}.")
    if c.reopen_conditions:
        lines.append("This class would be worth reopening if: " + "; ".join(c.reopen_conditions[:3]) + ".")
    return "\n".join(lines)
