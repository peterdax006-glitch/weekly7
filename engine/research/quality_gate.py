"""Research quality gate (contract C66 section 42, with sections 32, 43 and 48; C62 section 45). IMPLEMENTED - NOT VALIDATED.

Before any discovery may influence the production learner the gate checks, in order:
    point_in_time  leakage  identity  out_of_sample  replication  calibration  risk  complexity  transfer  failure_behavior
plus two that the C62 gate already required and C66 keeps: reproducibility, and metric_alignment (section 43: more trades, more
patterns, more code or a higher backtest return are never a reason). If any critical gate does not pass the answer is NOT promote:
    QUARANTINED          integrity is in doubt (a leak, a future-dated input, an identity-memoriser): set aside, never used, never deleted
    FAILED               measured evidence is negative (does not survive out of sample, does not replicate, unsafe tail)
    UNKNOWN              the evidence exists but cannot settle it (uncertain availability, an audit blind to planted leaks, an unexplained
                         contradiction): waiting for more of the same will not help
    NEEDS_MORE_EVIDENCE  evidence is missing or too thin: more data WOULD settle it
Precedence when several fire: QUARANTINED > FAILED > UNKNOWN > NEEDS_MORE_EVIDENCE > PROMOTE. A gate that crashes is UNKNOWN,
a missing input is never a pass, and PROMOTE requires every applicable gate to pass.

EXTENDS engine.learning.promotion (its statistical, OOS, transfer, risk, memorisation, future-audit and reproducibility gate
functions are called, not copied), engine.learning.firewalls (a GateVerdict from LearningFirewallGate is accepted as leak
evidence and its planted corpus is reused), engine.learning.identity_firewall (IdentityReport), engine.learning.calibration,
engine.learning.complexity and engine.research.replication (the replication assessment). SAME-YEAR IMPROVEMENT = INTERESTING,
OOS TRANSFER = REQUIRED (section 48): the OOS gate needs periods in calendar years the discovery never trained on."""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd

from engine.learning import calibration as CAL
from engine.learning import complexity as CX
from engine.learning import firewalls as FW
from engine.learning import identity_firewall as IDF
from engine.learning import promotion as PR
from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, canonical_json, require_past, stable_hash)
from engine.research import replication as RP
from engine.research.core import Availability, GateVerdict, MaturedRecord, Namespace

SECTION = "C66 section 42"
PASS, FAIL, QUARANTINE, UNKNOWN, MISSING, NA = "PASS", "FAIL", "QUARANTINE", "UNKNOWN", "MISSING", "NOT_APPLICABLE"
GATES = ("point_in_time", "leakage", "identity", "out_of_sample", "replication", "calibration", "risk", "complexity", "transfer",
         "failure_behavior", "reproducibility", "metric_alignment")
DEFAULT_CRITICAL = frozenset({"point_in_time", "leakage", "identity", "out_of_sample", "replication", "calibration", "risk",
                              "transfer", "reproducibility", "metric_alignment"})
INTEGRITY_GATES = frozenset({"point_in_time", "leakage", "identity"})
FORBIDDEN_METRICS = frozenset({"n_trades", "n_predictions", "n_patterns", "n_experiments", "lines_of_code", "backtest_return",
                               "n_rules", "n_features"})
LEGIT_METRICS = frozenset({"oos_transfer", "calibration", "downside", "reliability", "replication", "unseen_year_gain", "brier_skill",
                           "precision_at_coverage", "tail_loss"})
PRECEDENCE = (GateVerdict.QUARANTINED, GateVerdict.FAILED, GateVerdict.UNKNOWN, GateVerdict.NEEDS_MORE_EVIDENCE)
STATE_TO_VERDICT = {QUARANTINE: GateVerdict.QUARANTINED, FAIL: GateVerdict.FAILED, UNKNOWN: GateVerdict.UNKNOWN,
                    MISSING: GateVerdict.NEEDS_MORE_EVIDENCE}


# ------------------------------------------------------------------------------------------------ policy
@dataclass(frozen=True)
class QualityPolicy:
    """Thresholds in one hashable place. The statistical ones live in `promotion` (one source of truth); this adds what section 42
    needs beyond C62. `critical` may be widened but never narrowed below the section-42 core (validate refuses)."""
    promotion: PR.PromotionPolicy = field(default_factory=PR.PromotionPolicy)
    complexity: CX.ComplexityConfig = CX.DEFAULT_CCFG
    critical: frozenset = DEFAULT_CRITICAL
    min_unseen_years: int = 2
    min_unseen_share: float = 0.5
    min_calibration_n: int = 200
    max_ece: float = 0.06
    ece_alpha: float = 0.01
    min_slope: float = 0.7
    max_overconfidence: float = 0.05
    min_brier_skill: float = 0.0
    calibration_bins: int = 10
    calibration_sims: int = 200
    min_failure_episodes: int = 5
    min_perturbation_retention: float = 0.5
    max_unknown_cause_share: float = 0.5
    obs_per_complexity_unit: float = 15.0
    code_hash: str = ""

    def validate(self) -> list[str]:
        errs = list(self.promotion.validate()) + CX.validate_config(self.complexity)
        missing_core = DEFAULT_CRITICAL - set(self.critical)
        if missing_core:
            errs.append(f"critical set may not drop section-42 core gates: {sorted(missing_core)}")
        unknown = set(self.critical) - set(GATES)
        if unknown:
            errs.append(f"unknown critical gates: {sorted(unknown)}")
        for name in ("min_unseen_share", "max_ece", "ece_alpha", "max_overconfidence", "min_perturbation_retention", "max_unknown_cause_share"):
            v = getattr(self, name)
            if not (isinstance(v, float) and 0.0 <= v <= 1.0):
                errs.append(f"{name} must be a float in [0,1]")
        if self.min_unseen_years < 1:
            errs.append("min_unseen_years must be >= 1: same-year evidence alone can never promote")
        if self.min_calibration_n < 30 or self.calibration_bins < 2:
            errs.append("calibration needs at least 30 forecasts and 2 bins")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


# ------------------------------------------------------------------------------------------------ evidence
@dataclass(frozen=True)
class FeatureUse:
    name: str
    available_at: str | None = None            # real date the input became knowable; None = not recorded
    availability: Availability = Availability.UNCERTAIN


@dataclass(frozen=True)
class PITEvidence:
    features: tuple[FeatureUse, ...] = ()
    decision_time: str | None = None           # earliest decision the discovery informs
    label_horizon_days: int | None = None
    train_end: str | None = None
    first_test_start: str | None = None
    fills_next_open: bool | None = None        # decisions at a close fill at the NEXT session's open (rule 3)
    newest_evidence: str | None = None


@dataclass(frozen=True)
class LeakEvidence:
    audit_ran: bool = False
    firewall: FW.GateVerdict | None = None
    planted_probe_caught: bool | None = None   # the audit was shown to catch a deliberately planted leak
    findings: tuple[str, ...] = ()
    sealed_windows_touched: tuple[str, ...] = ()
    outcomes_after_now: int | None = None


@dataclass(frozen=True)
class IdentityEvidence:
    report: IDF.IdentityReport | None = None
    memorization: PR.MemorizationEvidence | None = None


@dataclass(frozen=True)
class OOSBundle:
    statistical: PR.StatisticalEvidence | None = None
    oos: PR.OOSEvidence | None = None
    train_years: tuple[int, ...] = ()


@dataclass(frozen=True)
class CalibrationEvidence:
    p: tuple[float, ...] = ()
    y: tuple[int, ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class ComplexityEvidence:
    candidate: CX.Candidate | None = None
    baseline: CX.Candidate | None = None       # a simpler alternative with OOS results on the same dates
    n_eff: float | None = None


@dataclass(frozen=True)
class FailureEvidence:
    failure_conditions: tuple[str, ...] = ()   # identity-free descriptions of when it stops working
    n_failure_episodes: int | None = None
    has_retirement_trigger: bool | None = None
    abstains_out_of_scope: bool | None = None
    perturbation_retention: float | None = None    # effect kept under planted data degradation, as a share of the clean effect
    unknown_cause_share: float | None = None
    worst_case_loss: float | None = None


@dataclass(frozen=True)
class QualityEvidence:
    """Everything the gate may look at. None = not supplied (never a pass)."""
    pit: PITEvidence | None = None
    leak: LeakEvidence | None = None
    identity: IdentityEvidence | None = None
    oos: OOSBundle | None = None
    replication: RP.ReplicationAssessment | None = None
    outputs_probabilities: bool = False
    calibration: CalibrationEvidence | None = None
    changes_risk_decisions: bool = True
    risk: PR.RiskEvidence | None = None
    complexity: ComplexityEvidence | None = None
    transfer: PR.TransferEvidence | None = None
    failure: FailureEvidence | None = None
    repro: PR.ReproEvidence | None = None
    justification: tuple[str, ...] = ()        # names of the metrics the case for promotion rests on
    provenance: Provenance | None = None

    def digest(self) -> str:
        return stable_hash({f.name: _view(getattr(self, f.name)) for f in dataclasses.fields(self)})


def _view(v: Any) -> Any:
    """Hashable, JSON-safe view of an evidence member (Series, verdict objects and assessments carry their own digests)."""
    if isinstance(v, RP.ReplicationAssessment):
        return v.digest()
    if isinstance(v, LeakEvidence):
        return {**{f.name: getattr(v, f.name) for f in dataclasses.fields(v) if f.name != "firewall"}, "firewall": v.firewall.digest() if v.firewall else ""}
    if isinstance(v, IdentityEvidence):
        return {"report": v.report.digest() if v.report else "", "memorization": v.memorization}
    if isinstance(v, ComplexityEvidence):
        return {"n_eff": v.n_eff, **{k: None if c is None else [c.spec, [round(float(x), 12) for x in c.oos.to_numpy()]]
                                       for k, c in (("candidate", v.candidate), ("baseline", v.baseline))}}
    return v


# ------------------------------------------------------------------------------------------------ outcomes
@dataclass(frozen=True)
class GateOutcome:
    gate: str
    state: str
    critical: bool
    detail: str
    measures: Mapping[str, Any] = field(default_factory=dict)
    margin: float | None = None

    @property
    def ok(self) -> bool:
        return self.state in (PASS, NA)

    @property
    def verdict_hint(self) -> GateVerdict | None:
        """What this outcome alone would make the verdict. A failed NON-critical gate can only ask for more evidence."""
        v = STATE_TO_VERDICT.get(self.state)
        if v is None:
            return None
        if not self.critical and v in (GateVerdict.FAILED, GateVerdict.UNKNOWN):
            return GateVerdict.NEEDS_MORE_EVIDENCE
        return v


def _out(gate: str, state: str, detail: str, critical: bool, measures: Mapping[str, Any] | None = None, margin: float | None = None) -> GateOutcome:
    return GateOutcome(gate, state, critical, detail, dict(measures or {}), margin)


def _from_promotion(gate: str, r: PR.GateResult, critical: bool, integrity: bool = False) -> GateOutcome:
    """Translate a C62 GateResult: MISSING stays MISSING, PASS stays PASS, FAIL is FAIL - or QUARANTINE for integrity gates."""
    if r.status == PR.PASS:
        state = PASS
    elif r.status == PR.NA:
        state = NA
    elif r.status == PR.MISSING:
        state = MISSING
    else:
        state = QUARANTINE if integrity else FAIL
    return _out(gate, state, r.detail, critical, r.measures, r.margin)


@dataclass(frozen=True)
class QualityDecision:
    subject_id: str
    now: str
    verdict: GateVerdict
    gates: tuple[GateOutcome, ...]
    blocking: tuple[str, ...]
    reasons: tuple[str, ...]
    remediation: tuple[str, ...]
    policy_digest: str
    evidence_digest: str
    code_hash: str

    @property
    def promote(self) -> bool:
        return self.verdict == GateVerdict.PROMOTE

    @property
    def decision_id(self) -> str:
        return stable_hash([self.subject_id, self.now, self.verdict, self.policy_digest, self.evidence_digest, [g.gate + g.state for g in self.gates]])

    def outcome(self, gate: str) -> GateOutcome:
        return next(g for g in self.gates if g.gate == gate)

    def to_dict(self) -> dict:
        return {"decision_id": self.decision_id, "subject_id": self.subject_id, "now": self.now, "verdict": self.verdict.value,
                "blocking": list(self.blocking), "reasons": list(self.reasons), "policy": self.policy_digest, "evidence": self.evidence_digest,
                "code_hash": self.code_hash, "gates": [dict(gate=g.gate, state=g.state, critical=g.critical, detail=g.detail) for g in self.gates]}


REMEDIATION = {
    "point_in_time": "declare when every input became knowable; drop or lag any input available at or after the decision; respect the label-horizon embargo",
    "leakage": "run the leak audit with a planted-leak probe and clear every finding; a discovery that touched sealed or future data stays quarantined",
    "identity": "remove identity-bearing inputs and rerun the identity-shuffle and disguised-rerun controls until the effect survives them",
    "out_of_sample": "evaluate on periods in calendar years the discovery never trained on, with the search size stated",
    "replication": "replicate on a fresh period and fresh stocks or regime with a matched control (see engine.research.replication)",
    "calibration": "recalibrate on held-out forecasts or stop reporting probabilities; overconfidence lowers the discovery's influence",
    "risk": "cap sizing, add an exit, or exclude the events that produce the tail losses",
    "complexity": "compare against a simpler rule on the same dates; complexity must earn its place out of sample",
    "transfer": "test on more stocks, sectors, eras or regimes outside the home context, or scope the discovery narrowly",
    "failure_behavior": "document when it fails, study failure episodes, define a retirement trigger and test graceful degradation",
    "reproducibility": "rerun with current code on two or more seeds, repeating one seed to prove determinism",
    "metric_alignment": "justify promotion by out-of-sample transfer, calibration, downside or replication, never by counts or a backtest return",
}


# ------------------------------------------------------------------------------------------------ the individual gates
def gate_point_in_time(ev: PITEvidence | None, prov_problems: Sequence[str], now, pol: QualityPolicy) -> GateOutcome:
    g, crit = "point_in_time", True
    if ev is None or ev.decision_time is None or not ev.features:
        return _out(g, MISSING, "evidence missing: decision time and the availability of every input", crit)
    violations, uncertain = [], []
    dec = as_date(ev.decision_time)
    for f in ev.features:
        if f.available_at:
            if as_date(f.available_at) >= dec:
                violations.append(f"{f.name} became knowable {f.available_at}, not before the decision {ev.decision_time}")
        elif f.availability == Availability.KNOWN_BEFORE_EVENT:
            continue
        elif f.availability == Availability.UNCERTAIN:
            uncertain.append(f.name)
        else:
            violations.append(f"{f.name} is {f.availability.value}")
    if ev.fills_next_open is False:
        violations.append("decisions fill at the same close instead of the next session's open")
    if ev.label_horizon_days is not None and ev.train_end and ev.first_test_start:
        gap = (as_date(ev.first_test_start) - as_date(ev.train_end)).days
        if gap < ev.label_horizon_days:
            violations.append(f"test starts {gap}d after training ends but labels look {ev.label_horizon_days}d ahead: overlapping outcomes")
    elif ev.label_horizon_days is None or not ev.train_end or not ev.first_test_start:
        uncertain.append("train/test embargo not recorded")
    if ev.newest_evidence:
        try:
            require_past(ev.newest_evidence, now, "newest evidence")
        except FirewallBreach as e:
            violations.append(str(e))
    violations += [f"provenance: {p}" for p in prov_problems]
    if violations:
        return _out(g, QUARANTINE, "; ".join(violations), crit, {"violations": len(violations)})
    if uncertain:
        return _out(g, UNKNOWN, "availability cannot be established for: " + ", ".join(uncertain), crit, {"uncertain": len(uncertain)})
    return _out(g, PASS, f"{len(ev.features)} inputs all knowable strictly before the decision; embargo respected", crit, {"inputs": len(ev.features)})


def gate_leakage(ev: LeakEvidence | None, prov_ok: bool, now, pol: QualityPolicy) -> GateOutcome:
    g, crit = "leakage", True
    if ev is None or not ev.audit_ran:
        return _out(g, MISSING, "evidence missing: a completed leak audit", crit)
    problems = []
    if ev.firewall is not None:
        if not ev.firewall.relevant:
            problems.append("firewall gate checked no layer")
        elif not ev.firewall.passed:
            problems.append("firewall layers failed: " + ", ".join(k.value for k in ev.firewall.failed_layers))
    if ev.sealed_windows_touched:
        problems.append(f"sealed windows touched: {list(ev.sealed_windows_touched)}")
    problems += list(ev.findings)
    if ev.outcomes_after_now is None:
        return _out(g, MISSING, "outcomes_after_now was not measured", crit)
    if ev.outcomes_after_now > 0:
        problems.append(f"{ev.outcomes_after_now} outcomes dated at/after now")
    if not prov_ok:
        problems.append("provenance says the item could not have existed at `now`")
    if problems:
        return _out(g, QUARANTINE, "; ".join(problems), crit, {"findings": len(problems)})
    if ev.planted_probe_caught is None:
        return _out(g, MISSING, "the audit was never shown to catch a planted leak: a check that cannot fail proves nothing", crit)
    if ev.planted_probe_caught is False:
        return _out(g, UNKNOWN, "the audit missed a deliberately planted leak: its clean result is worthless", crit)
    return _out(g, PASS, "audit clean and proven able to see a planted leak", crit, {"layers": len(ev.firewall.relevant) if ev.firewall else 0})


def gate_identity(ev: IdentityEvidence | None, pol: QualityPolicy) -> GateOutcome:
    g, crit = "identity", True
    if ev is None or (ev.report is None and ev.memorization is None):
        return _out(g, MISSING, "evidence missing: identity-attack report or memorisation measures", crit)
    problems, quarantine = [], False
    if ev.report is not None:
        rep = ev.report
        if not rep.deterministic:
            problems.append("learner is not deterministic under identical input")
        if rep.memorization_suspected:
            problems.append("skill collapses when only the evaluated identities are disguised: memorisation (" + ", ".join(rep.collapsed) + ")")
            quarantine = True
        elif rep.logic_identity_dependent:
            problems.append("logic depends on identity (collapses when train and eval are relabelled together)")
            quarantine = True
        informative = IDF.informative_attacks(rep)
        if not informative:
            return _out(g, UNKNOWN, "no identity attack was informative (no baseline skill to lose): independence cannot be judged", crit)
    if ev.memorization is not None:
        r = PR.gate_anti_memorization(ev.memorization, pol.promotion)
        if r.status == PR.MISSING:
            return _out(g, MISSING, r.detail, crit)
        if r.status == PR.FAIL:
            problems.append(r.detail)
            quarantine = quarantine or bool(ev.memorization.lookup_table_suspected) or \
                any(m is not None and m < pol.promotion.min_identity_retention
                    for m in (ev.memorization.identity_shuffle_retention, ev.memorization.disguised_rerun_retention))
    if problems:
        return _out(g, QUARANTINE if quarantine else FAIL, "; ".join(problems), crit, {"problems": len(problems)})
    return _out(g, PASS, "survives identity scrambling and disguise", crit)


def gate_out_of_sample(ev: OOSBundle | None, pol: QualityPolicy, now) -> GateOutcome:
    g, crit = "out_of_sample", True
    if ev is None or ev.statistical is None or ev.oos is None:
        return _out(g, MISSING, "evidence missing: search-adjusted statistics and out-of-sample effects", crit)
    st = PR.gate_statistical_validity(ev.statistical, pol.promotion)
    oo = PR.gate_oos_confirmation(ev.oos, pol.promotion, now)
    for r in (st, oo):
        if r.status == PR.MISSING:
            return _from_promotion(g, r, crit)
    problems = [r.detail for r in (st, oo) if r.status == PR.FAIL]
    years = sorted({as_date(d).year for d in ev.oos.oos_dates})
    trained = set(int(y) for y in ev.train_years)
    unseen = [y for y in years if y not in trained]
    eff = np.asarray(ev.oos.oos_effects, dtype=float)
    unseen_share = float(np.mean([as_date(d).year not in trained for d in ev.oos.oos_dates])) if len(eff) else 0.0
    if not ev.train_years:
        return _out(g, MISSING, "training years not recorded: same-year improvement cannot be told from unseen-year transfer", crit)
    if len(unseen) < pol.min_unseen_years:
        problems.append(f"only {len(unseen)} unseen calendar year(s) in the OOS evidence (need {pol.min_unseen_years}): same-year improvement is interesting, not sufficient")
    elif unseen_share < pol.min_unseen_share:
        problems.append(f"{unseen_share:.0%} of OOS periods fall in unseen years (need {pol.min_unseen_share:.0%})")
    else:
        mask = np.array([as_date(d).year not in trained for d in ev.oos.oos_dates])
        um = float(eff[mask].mean())
        if um <= 0:
            problems.append(f"the effect on unseen years is {um:+.5f}: it lives only in years it trained on")
    if problems:
        return _out(g, FAIL, "; ".join(problems), crit, {"unseen_years": len(unseen), "unseen_share": unseen_share})
    return _out(g, PASS, f"{st.detail}; {oo.detail}; {len(unseen)} unseen years", crit, {**st.measures, **oo.measures, "unseen_years": len(unseen)}, oo.margin)


def gate_replication(a: RP.ReplicationAssessment | None, pol: QualityPolicy) -> GateOutcome:
    g, crit = "replication", True
    if a is None:
        return _out(g, MISSING, "evidence missing: a replication assessment", crit)
    m = {"status": a.status.value, "supporting": a.n_supporting, "refuting": a.n_refuting, "attempts": a.n_valid_groups}
    S = RP.Status
    if a.status == S.REPLICATED:
        return _out(g, PASS, f"replicated: {a.n_supporting} independent fresh runs of {a.n_valid_groups} attempts; axes {', '.join(a.axes_covered)}", crit, m)
    if a.status == S.FAILED:
        return _out(g, FAIL, f"failed to replicate on {a.n_refuting} powered independent run(s)", crit, m)
    if a.status == S.CONTRADICTED:
        return _out(g, UNKNOWN, f"{a.n_supporting} runs support it and {a.n_refuting} powered runs refute it, unexplained", crit, m)
    return _out(g, MISSING, f"{a.status.value}: {'; '.join(a.reasons[:2]) or 'requirements unmet'}", crit, m)


def gate_calibration(ev: CalibrationEvidence | None, outputs_probabilities: bool, pol: QualityPolicy) -> GateOutcome:
    g, crit = "calibration", True
    if not outputs_probabilities:
        return _out(g, NA, "the discovery states no probabilities, so there is nothing to calibrate", crit)
    if ev is None or not ev.p:
        return _out(g, MISSING, "evidence missing: forecast probabilities with their outcomes", crit)
    p, y = np.asarray(ev.p, dtype=float), np.asarray(ev.y, dtype=float)
    try:
        CAL._check_py(p, y)
    except ValueError as e:
        return _out(g, FAIL, f"malformed forecasts: {e}", crit)
    if len(p) < pol.min_calibration_n:
        return _out(g, MISSING, f"{len(p)} forecasts < {pol.min_calibration_n}", crit, {"n": int(len(p))}, PR._margin(len(p), pol.min_calibration_n))
    if float(y.min()) == float(y.max()):
        return _out(g, MISSING, "outcomes are constant: calibration cannot be assessed", crit)
    ece = CAL.ece_equal_mass(p, y, pol.calibration_bins)
    null_p = CAL.ece_null_pvalue(p, pol.calibration_bins, ece, np.random.default_rng(ev.seed), pol.calibration_sims)
    base = float(y.mean())
    brier = CAL.brier(p, y)
    skill = 1.0 - brier / max(base * (1 - base), 1e-12)
    slope = CAL.platt_slope(p, y)
    over = CAL.overconfidence(p, y)
    problems = []
    if ece > pol.max_ece and null_p < pol.ece_alpha:
        problems.append(f"ECE {ece:.3f} > {pol.max_ece} and beyond what perfect calibration would give (p={null_p:.3f})")
    if math.isfinite(slope["b_hi"]) and slope["b_hi"] < pol.min_slope:
        problems.append(f"calibration slope {slope['b']:.2f} (upper bound {slope['b_hi']:.2f}) < {pol.min_slope}: overconfident")
    if math.isfinite(over["excess"]) and over["excess"] > pol.max_overconfidence:
        problems.append(f"confidence exceeds accuracy by {over['excess']:.3f}")
    if skill <= pol.min_brier_skill:
        problems.append(f"Brier skill {skill:+.3f} over the base rate: forecasts add nothing")
    m = {"ece": ece, "ece_null_p": null_p, "brier": brier, "brier_skill": skill, "slope": slope["b"], "excess": over["excess"], "n": int(len(p))}
    return _out(g, FAIL if problems else PASS, "; ".join(problems) or f"ECE {ece:.3f}, slope {slope['b']:.2f}, Brier skill {skill:+.3f}", crit, m,
                PR._margin(pol.max_ece, ece))


def gate_risk(ev: PR.RiskEvidence | None, changes_risk: bool, pol: QualityPolicy) -> GateOutcome:
    g, crit = "risk", True
    if not changes_risk:
        return _out(g, NA, "the discovery only orders research; it never sizes or selects a holding", crit)
    return _from_promotion(g, PR.gate_risk_acceptance(ev, pol.promotion), crit)


def gate_complexity(ev: ComplexityEvidence | None, pol: QualityPolicy) -> GateOutcome:
    g, crit = "complexity", False
    if ev is None or ev.candidate is None:
        return _out(g, MISSING, "evidence missing: the discovery's structure and out-of-sample record", crit)
    cfg = pol.complexity
    spec = ev.candidate.spec
    if ev.n_eff is not None and not CX.within_budget(spec, ev.n_eff, cfg, pol.obs_per_complexity_unit):
        return _out(g, FAIL, f"{spec.units(cfg.weights):.1f} complexity units exceed what {ev.n_eff:.0f} effective observations support "
                             f"({CX.max_units(ev.n_eff, pol.obs_per_complexity_unit):.1f})", crit, {"units": spec.units(cfg.weights)})
    if spec.identity_smell():
        return _out(g, FAIL, f"{spec.n_exceptions} hard-coded exception(s): memorisation smell", crit, {"exceptions": spec.n_exceptions})
    if ev.baseline is None:
        return _out(g, MISSING, "no simpler alternative was compared on the same dates", crit)
    if ev.baseline.spec.units(cfg.weights) >= spec.units(cfg.weights):
        return _out(g, PASS, "no simpler rule than the baseline is being asked to be replaced", crit)
    v = CX.compare(ev.baseline, ev.candidate, cfg)
    if v.verdict == CX.Verdict.COMPLEX:
        return _out(g, PASS, f"complexity earned its place: gain t={v.gain_t:.2f} >= required {v.t_required:.2f}", crit, {"gain": v.gain, "t": v.gain_t})
    if v.verdict == CX.Verdict.NEED_DATA:
        return _out(g, MISSING, "; ".join(v.reasons), crit)
    return _out(g, FAIL, f"the simpler rule does as well ({v.verdict.value}): " + "; ".join(v.reasons[:2]), crit, {"gain": v.gain, "t": v.gain_t})


def gate_transfer(ev: PR.TransferEvidence | None, pol: QualityPolicy) -> GateOutcome:
    return _from_promotion("transfer", PR.gate_cross_context_transfer(ev, pol.promotion), True)


def gate_failure_behavior(ev: FailureEvidence | None, pol: QualityPolicy) -> GateOutcome:
    g, crit = "failure_behavior", False
    if ev is None:
        return _out(g, MISSING, "evidence missing: how and when the discovery fails", crit)
    problems, missing = [], []
    if ev.n_failure_episodes is None or ev.has_retirement_trigger is None or ev.abstains_out_of_scope is None or ev.perturbation_retention is None:
        return _out(g, MISSING, "failure episodes, retirement trigger, out-of-scope behaviour and perturbation retention must all be recorded", crit)
    if ev.n_failure_episodes < pol.min_failure_episodes:
        missing.append(f"only {ev.n_failure_episodes} failure episodes studied (< {pol.min_failure_episodes})")
    if not ev.failure_conditions and ev.n_failure_episodes >= pol.min_failure_episodes:
        problems.append(f"{ev.n_failure_episodes} failures studied yet no failure condition documented")
    if not ev.has_retirement_trigger:
        problems.append("no retirement trigger defined")
    if not ev.abstains_out_of_scope:
        problems.append("does not abstain outside its documented scope")
    if ev.perturbation_retention < pol.min_perturbation_retention:
        problems.append(f"keeps only {ev.perturbation_retention:.0%} of its effect under data degradation (< {pol.min_perturbation_retention:.0%})")
    if ev.unknown_cause_share is not None and ev.unknown_cause_share > pol.max_unknown_cause_share:
        return _out(g, UNKNOWN, f"{ev.unknown_cause_share:.0%} of its failures have no established cause", crit, {"unknown_share": ev.unknown_cause_share})
    if problems:
        return _out(g, FAIL, "; ".join(problems), crit, {"problems": len(problems)})
    if missing:
        return _out(g, MISSING, "; ".join(missing), crit)
    return _out(g, PASS, f"{len(ev.failure_conditions)} failure conditions documented, retirement trigger set, degrades gracefully", crit,
                {"conditions": len(ev.failure_conditions)})


def gate_reproducibility(ev: PR.ReproEvidence | None, pol: QualityPolicy) -> GateOutcome:
    return _from_promotion("reproducibility", PR.gate_reproducibility(ev, pol.promotion, pol.code_hash or PR.current_code_hash()), True)


def gate_metric_alignment(justification: Sequence[str], pol: QualityPolicy) -> GateOutcome:
    """Section 43: the case for promotion must rest on reliable, transferring, cheat-free, downside-controlled prediction."""
    g = "metric_alignment"
    if not justification:
        return _out(g, MISSING, "evidence missing: which metrics the case for promotion rests on", True)
    cited = set(justification)
    bad, good = sorted(cited & FORBIDDEN_METRICS), sorted(cited & LEGIT_METRICS)
    unknown = sorted(cited - FORBIDDEN_METRICS - LEGIT_METRICS)
    if bad and not good:
        return _out(g, FAIL, f"promotion rests on {bad}: counts and backtest returns are not evidence of reliable prediction", True, {"forbidden": bad})
    if not good:
        return _out(g, MISSING, f"none of the cited metrics ({unknown}) is a recognised measure of transferring prediction", True)
    return _out(g, PASS, f"rests on {good}" + (f"; also cites {bad} which carry no weight" if bad else ""), True, {"legit": good, "forbidden": bad})


# ------------------------------------------------------------------------------------------------ the gate
def combine(outcomes: Sequence[GateOutcome]) -> tuple[GateVerdict, tuple[str, ...], tuple[str, ...]]:
    """(verdict, blocking gate names, reasons). PROMOTE only if every outcome is ok; otherwise the highest-precedence hint."""
    hints = {o.gate: o.verdict_hint for o in outcomes if not o.ok}
    if not hints and outcomes:
        return GateVerdict.PROMOTE, (), ()
    verdict = next((v for v in PRECEDENCE if v in hints.values()), GateVerdict.NEEDS_MORE_EVIDENCE)
    blocking = tuple(sorted(g for g, v in hints.items() if v == verdict))
    reasons = tuple(f"{o.gate}: {o.detail}" for o in outcomes if o.gate in blocking)
    return verdict, blocking, reasons


class QualityGate:
    """Runs every gate and returns an immutable decision. Stateless apart from its policy: the same inputs always give the same
    decision id. A FirewallBreach from any gate (evidence dated at/after `now`) propagates: it is a bug in the caller, not a verdict."""

    def __init__(self, policy: QualityPolicy | None = None):
        self.policy = policy or QualityPolicy(code_hash=PR.current_code_hash())
        errs = self.policy.validate()
        if errs:
            raise ValueError(f"invalid quality policy: {errs}")

    def _run(self, gate: str, fn: Callable[[], GateOutcome]) -> GateOutcome:
        try:
            return fn()
        except FirewallBreach:
            raise
        except Exception as e:                                   # noqa: BLE001  a gate that crashes has not passed
            return _out(gate, UNKNOWN, f"gate raised {type(e).__name__}: {e}", gate in self.policy.critical)

    def evaluate(self, subject_id: str, ev: QualityEvidence, now) -> QualityDecision:
        pol = self.policy
        pv = PR.provenance_problems(_Holder(ev.provenance), now) if ev.provenance is not None else ["no provenance supplied"]
        prov_ok = ev.provenance is not None and ev.provenance.could_exist_at(now)
        table: dict[str, Callable[[], GateOutcome]] = {
            "point_in_time": lambda: gate_point_in_time(ev.pit, pv, now, pol),
            "leakage": lambda: gate_leakage(ev.leak, prov_ok, now, pol),
            "identity": lambda: gate_identity(ev.identity, pol),
            "out_of_sample": lambda: gate_out_of_sample(ev.oos, pol, now),
            "replication": lambda: gate_replication(ev.replication, pol),
            "calibration": lambda: gate_calibration(ev.calibration, ev.outputs_probabilities, pol),
            "risk": lambda: gate_risk(ev.risk, ev.changes_risk_decisions, pol),
            "complexity": lambda: gate_complexity(ev.complexity, pol),
            "transfer": lambda: gate_transfer(ev.transfer, pol),
            "failure_behavior": lambda: gate_failure_behavior(ev.failure, pol),
            "reproducibility": lambda: gate_reproducibility(ev.repro, pol),
            "metric_alignment": lambda: gate_metric_alignment(ev.justification, pol),
        }
        outs = []
        for name in GATES:
            o = self._run(name, table[name])
            outs.append(dataclasses.replace(o, critical=name in pol.critical))
        verdict, blocking, reasons = combine(outs)
        rem = tuple(REMEDIATION[b] for b in blocking)
        return QualityDecision(str(subject_id), str(as_date(now)), verdict, tuple(outs), blocking, reasons, rem, pol.digest(), ev.digest(), pol.code_hash)


class _Holder:
    """Adapter so promotion.provenance_problems (which reads .provenance) can audit a bare Provenance."""

    def __init__(self, provenance: Provenance | None):
        self.provenance = provenance


# ------------------------------------------------------------------------------------------------ reports and analysis
def render_decision(d: QualityDecision) -> str:
    lines = [f"# Quality gate: {d.subject_id} -> {d.verdict.value}", f"as of {d.now}; decision {d.decision_id}", "",
             "| gate | state | critical | detail |", "|---|---|---|---|"]
    lines += [f"| {g.gate} | {g.state} | {'yes' if g.critical else 'no'} | {g.detail} |" for g in d.gates]
    if d.blocking:
        lines += ["", "Blocking: " + ", ".join(d.blocking), "", "What would change this:"]
        lines += [f"- {r}" for r in d.remediation]
    lines += ["", "PROMOTE means the discovery may go on to the champion/challenger board; it does not by itself change any decision.",
              "IMPLEMENTED - NOT VALIDATED."]
    return "\n".join(lines)


def near_misses(d: QualityDecision, within: float = 0.25) -> list[tuple[str, float]]:
    """Passing gates whose margin is thin: the ones most likely to flip on new evidence."""
    return sorted(((g.gate, g.margin) for g in d.gates if g.state == PASS and g.margin is not None and math.isfinite(g.margin) and g.margin < within),
                  key=lambda t: t[1])


def funnel(decisions: Sequence[QualityDecision]) -> dict:
    """Verdict counts and, for every non-promoted decision, the gates that blocked it: which gate does the work, which never fires."""
    counts: dict[str, int] = {}
    by_gate: dict[str, int] = {g: 0 for g in GATES}
    for d in decisions:
        counts[d.verdict.value] = counts.get(d.verdict.value, 0) + 1
        for b in d.blocking:
            by_gate[b] += 1
    return {"n": len(decisions), "verdicts": counts, "blocked_by": by_gate, "never_fires": sorted(g for g, n in by_gate.items() if n == 0),
            "promote_rate": counts.get("PROMOTE", 0) / len(decisions) if decisions else float("nan")}


def sensitivity(gate: QualityGate, subject_id: str, ev: QualityEvidence, now, fixes: Mapping[str, Callable[[QualityEvidence], QualityEvidence]]) -> dict:
    """For each supplied one-gate fix, would applying it alone change the verdict? Shows which single missing piece is holding a
    discovery back (fixes are caller-supplied: the gate never invents evidence)."""
    base = gate.evaluate(subject_id, ev, now)
    out = {}
    for name, fix in fixes.items():
        after = gate.evaluate(subject_id, fix(ev), now)
        out[name] = {"before": base.verdict.value, "after": after.verdict.value, "changed": after.verdict != base.verdict,
                     "still_blocking": list(after.blocking)}
    return out


def as_matured_record(d: QualityDecision, matured_at: str, prov: Provenance) -> MaturedRecord:
    """Identity-free record of the decision for the curator: the verdict and blocking gate names, never evidence dates or tickers."""
    payload = {"verdict": d.verdict.value, "blocking": list(d.blocking), "promote": d.promote, "policy": d.policy_digest}
    return MaturedRecord("QG-" + d.decision_id, matured_at, payload, prov, Namespace.MATURED_RESEARCH)


# ------------------------------------------------------------------------------------------------ quarantine
class QuarantineStore(RP.ResearchLane):
    LANE = "qquar"

    """Append-only, hash-chained record of quarantines and releases. Quarantine is not deletion (history is immutable): the item stays,
    flagged, and is released only by a later decision that clears every integrity gate on evidence that differs from what
    quarantined it."""

    def quarantine(self, d: QualityDecision) -> None:
        if d.verdict != GateVerdict.QUARANTINED:
            raise ValueError(f"only a QUARANTINED decision can quarantine (is {d.verdict.value})")
        self.append("quarantine", {"subject_id": d.subject_id, "now": d.now, "evidence": d.evidence_digest, "reasons": list(d.reasons),
                                   "gates": list(d.blocking)})

    def active(self) -> dict[str, dict]:
        state: dict[str, dict] = {}
        for r in self.rows():
            b = r["body"]
            if r["kind"] == "quarantine":
                state[b["subject_id"]] = b
            elif r["kind"] == "release":
                state.pop(b["subject_id"], None)
        return state

    def is_quarantined(self, subject_id: str) -> bool:
        return subject_id in self.active()

    def release(self, new: QualityDecision) -> None:
        cur = self.active().get(new.subject_id)
        if cur is None:
            raise KeyError(f"{new.subject_id} is not quarantined")
        if new.evidence_digest == cur["evidence"]:
            raise ValueError("release needs new evidence: the evidence digest is unchanged")
        bad = [g.gate for g in new.gates if g.gate in INTEGRITY_GATES and not g.ok]
        if bad:
            raise ValueError(f"integrity gates still failing: {bad}")
        if new.verdict in (GateVerdict.QUARANTINED, GateVerdict.FAILED):
            raise ValueError(f"cannot release into {new.verdict.value}")
        self.append("release", {"subject_id": new.subject_id, "now": new.now, "decision": new.decision_id, "verdict": new.verdict.value})

    def history(self, subject_id: str) -> list[tuple[str, str]]:
        return [(r["kind"], r["body"]["now"]) for r in self.rows() if r["body"].get("subject_id") == subject_id]


# ------------------------------------------------------------------------------------------------ a clean reference and planted defects
def reference_evidence(now="2021-06-01", seed: int = 0) -> tuple[QualityEvidence, QualityPolicy]:
    """A synthetic, fully clean evidence bundle that the default-policy gate PROMOTEs, and the policy it is clean under. Nothing in it
    is real data; it exists so 'the gate can pass' and 'the gate can fail' are both demonstrable (a check that cannot fail is
    worthless). Every planted defect below is a mutation of this bundle."""
    rng = np.random.default_rng(seed)
    pol = QualityPolicy(code_hash="refcode")
    prov = Provenance("2026-09-29T00:00:00", "2019-12-31", "refcode", "dataA", "cfgA", "expA", "run1", 7, "2019-12-31")
    oos_dates = tuple(str(d.date()) for d in pd.bdate_range("2019-01-07", periods=140, freq="W-MON")[:70])
    oos_dates = tuple(d for d in oos_dates if as_date(d) > as_date("2017-12-31"))
    oos_eff = tuple(float(x) for x in rng.normal(0.004, 0.01, len(oos_dates)))
    d0 = RP.Discovery("D1", 0.0055, 0.011, 80, ("2016-01-04", "2017-12-29"), 10, frozenset(f"S{i}" for i in range(40)), (1,),
                      frozenset({"calm"}), "refcode", "dataA", "2018-01-15", n_tests_searched=8)
    runs = []
    for k, (win, reg, stocks, seed_k) in enumerate([(("2018-03-01", "2018-12-31"), "calm", range(100, 140), 11),
                                                     (("2019-03-01", "2019-12-31"), "stress", range(200, 240), 12)]):
        eff = rng.normal(0.0045, 0.011, 40)
        runs.append(RP.ReplicationRun(f"R{k}", "D1", win, frozenset(f"S{i}" for i in stocks), seed_k, reg, tuple(float(v) for v in eff),
                                      tuple(float(v) for v in rng.normal(0.0, 0.011, 40)), "refcode", "dataA", str(as_date(win[1]).replace(year=as_date(win[1]).year + 1)),
                                      implementation="alt" if k else "primary"))
    assess = RP.assess(d0, runs, now)
    p = np.clip(rng.beta(2.5, 4, 600), 0.02, 0.98)
    y = (rng.random(600) < p).astype(int)
    idx = pd.date_range("2018-01-05", periods=60, freq="W-FRI")
    good = pd.Series(rng.normal(0.004, 0.006, 60), index=idx)
    plain = pd.Series(good.values - 0.0012 + rng.normal(0, 0.001, 60), index=idx)
    folds = pd.Series(np.arange(60) % 4, index=idx)
    cand = CX.Candidate(CX.RuleSpec("cand", n_features=2, n_conditions=1, n_thresholds=1, tree_depth=2, n_free_params=1), good, folds, 0.005)
    base = CX.Candidate(CX.RuleSpec("base", n_features=1, n_free_params=1), plain, folds, 0.0045)
    mk = lambda kind, mode, status, ret: IDF.AttackVerdict(kind, mode, status, 0.05, 0.05 * ret, ret)      # noqa: E731
    report = IDF.IdentityReport(tuple(mk(k, m, "OK", 0.95) for k in ("ticker_permutation", "date_permutation", "stock_substitution") for m in ("eval", "both")),
                                0.05, 0.01, True, seed)
    ev = QualityEvidence(
        pit=PITEvidence(tuple(FeatureUse(n, "2017-12-01", Availability.KNOWN_BEFORE_EVENT) for n in ("f_range", "f_volume", "m_vol")), "2018-01-15", 10,
                        "2017-12-29", "2018-01-15", True, "2018-01-02"),
        leak=LeakEvidence(True, FW.LearningFirewallGate().evaluate(FW.reference_context("2020-06-01")), True, (), (), 0),
        identity=IdentityEvidence(report, PR.MemorizationEvidence(0.92, 0.9, ("f_range", "f_volume"), False, 0.05, 80)),
        oos=OOSBundle(PR.StatisticalEvidence(0.0055, 400, 6.0, None, 8, 300.0), PR.OOSEvidence("2017-12-29", oos_dates, oos_eff, 0.0055), (2016, 2017)),
        replication=assess, outputs_probabilities=True, calibration=CalibrationEvidence(tuple(float(v) for v in p), tuple(int(v) for v in y), seed),
        risk=PR.RiskEvidence(40, -0.06, -0.12, -0.05, 0, -0.08, -0.15),
        complexity=ComplexityEvidence(cand, base, 400.0),
        transfer=PR.TransferEvidence({"tech": 0.004, "energy": 0.003, "health": 0.0035, "fin": 0.0025, "home": 0.0055}, {k: 60 for k in ("tech", "energy", "health", "fin", "home")},
                                     ("home",), 0.0055),
        failure=FailureEvidence(("m_vol above 3x median", "earnings within 2 days"), 12, True, True, 0.7, 0.2, -0.06),
        repro=PR.ReproEvidence(tuple(PR.RerunRecord(0.0055 + 1e-9 * i, s, "refcode", "dataA") for i, s in enumerate((1, 1, 2, 3))), "dataA"),
        justification=("oos_transfer", "calibration", "downside"), provenance=prov)
    return ev, pol


def _mut(**kw) -> Callable[[QualityEvidence], QualityEvidence]:
    return lambda e: dataclasses.replace(e, **kw)


_T = TypeVar("_T")


def _clean(part: _T | None, name: str) -> _T:
    """A planted defect mutates a part of the CLEAN bundle; a clean bundle that lacks the part cannot be mutated."""
    if part is None:
        raise ValueError(f"the clean evidence bundle has no {name} to plant a defect in")
    return part


def planted_defects() -> dict[str, tuple[Callable[[QualityEvidence], QualityEvidence], str, GateVerdict]]:
    """name -> (mutation of the clean bundle, the gate that must catch it, the verdict it must produce). Twelve known defects, one
    per gate: `verify_gate_can_fail` runs them all and reports any the gate waved through."""
    def leak_corpus(name: str) -> LeakEvidence:
        ctx = FW.planted_corpus()[name](FW.reference_context("2020-06-01"))
        return LeakEvidence(True, FW.LearningFirewallGate().evaluate(ctx), True, (), (), 0)

    def bad_oos(e: QualityEvidence) -> QualityEvidence:
        o = _clean(e.oos, "oos")
        oos = _clean(o.oos, "oos.oos")
        eff = tuple(0.0 for _ in oos.oos_effects)
        return dataclasses.replace(e, oos=OOSBundle(o.statistical, dataclasses.replace(oos, oos_effects=tuple(x - 0.004 for x in eff)), o.train_years))

    def same_year(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, oos=dataclasses.replace(_clean(e.oos, "oos"), train_years=(2016, 2017, 2018, 2019)))

    def overconfident(e: QualityEvidence) -> QualityEvidence:
        cal = _clean(e.calibration, "calibration")
        p = np.asarray(cal.p)
        return dataclasses.replace(e, calibration=dataclasses.replace(cal, p=tuple(float(v) for v in np.clip((p - 0.5) * 2.2 + 0.5, 0.01, 0.99))))

    def collapse(e: QualityEvidence) -> QualityEvidence:
        ident = _clean(e.identity, "identity")
        rep = _clean(ident.report, "identity.report")
        vs = tuple(dataclasses.replace(v, status="COLLAPSE", retention=0.0) if v.mode == "eval" else v for v in rep.verdicts)
        return dataclasses.replace(e, identity=IdentityEvidence(dataclasses.replace(rep, verdicts=vs), ident.memorization))

    def future_feature(e: QualityEvidence) -> QualityEvidence:
        pit = _clean(e.pit, "pit")
        f = pit.features + (FeatureUse("fwd_ret_5d", "2018-01-20", Availability.KNOWN_ONLY_AFTER_EVENT),)
        return dataclasses.replace(e, pit=dataclasses.replace(pit, features=f))

    def unreplicated(e: QualityEvidence) -> QualityEvidence:
        d = _clean(e.replication, "replication")
        return dataclasses.replace(e, replication=dataclasses.replace(d, status=RP.Status.UNREPLICATED, n_supporting=0, run_results=(), reasons=("no runs",)))

    def failed_repl(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, replication=dataclasses.replace(_clean(e.replication, "replication"), status=RP.Status.FAILED, n_refuting=2, n_supporting=0))

    def fat_tail(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, risk=dataclasses.replace(_clean(e.risk, "risk"), worst_period=-0.6, catastrophic_count=2))

    def complex_no_gain(e: QualityEvidence) -> QualityEvidence:
        c = _clean(e.complexity, "complexity")
        noisy = _clean(c.baseline, "complexity.baseline").oos.copy()
        rng = np.random.default_rng(3)
        worse = dataclasses.replace(_clean(c.candidate, "complexity.candidate"), oos=noisy + rng.normal(0, 0.0005, len(noisy)))
        return dataclasses.replace(e, complexity=ComplexityEvidence(worse, c.baseline, c.n_eff))

    def no_transfer(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, transfer=dataclasses.replace(_clean(e.transfer, "transfer"), context_effects={"tech": -0.002, "energy": -0.003, "health": 0.0001, "fin": -0.001, "home": 0.0055}))

    def brittle(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, failure=dataclasses.replace(_clean(e.failure, "failure"), has_retirement_trigger=False, perturbation_retention=0.1))

    def nondeterministic(e: QualityEvidence) -> QualityEvidence:
        return dataclasses.replace(e, repro=PR.ReproEvidence(tuple(PR.RerunRecord(v, s, "refcode", "dataA") for v, s in ((0.0055, 1), (0.0031, 1), (0.0040, 2))), "dataA"))

    return {
        "future_input": (future_feature, "point_in_time", GateVerdict.QUARANTINED),
        "firewall_leak": (lambda e: dataclasses.replace(e, leak=leak_corpus("data_label_as_feature")), "leakage", GateVerdict.QUARANTINED),
        "blind_audit": (lambda e: dataclasses.replace(e, leak=dataclasses.replace(e.leak, planted_probe_caught=False)), "leakage", GateVerdict.UNKNOWN),
        "memoriser": (collapse, "identity", GateVerdict.QUARANTINED),
        "no_oos_effect": (bad_oos, "out_of_sample", GateVerdict.FAILED),
        "same_year_only": (same_year, "out_of_sample", GateVerdict.FAILED),
        "unreplicated": (unreplicated, "replication", GateVerdict.NEEDS_MORE_EVIDENCE),
        "failed_replication": (failed_repl, "replication", GateVerdict.FAILED),
        "overconfident": (overconfident, "calibration", GateVerdict.FAILED),
        "fat_tail": (fat_tail, "risk", GateVerdict.FAILED),
        "complexity_unearned": (complex_no_gain, "complexity", GateVerdict.NEEDS_MORE_EVIDENCE),
        "no_transfer": (no_transfer, "transfer", GateVerdict.FAILED),
        "brittle": (brittle, "failure_behavior", GateVerdict.NEEDS_MORE_EVIDENCE),
        "nondeterministic": (nondeterministic, "reproducibility", GateVerdict.FAILED),
        "wrong_metric": (_mut(justification=("n_trades", "backtest_return")), "metric_alignment", GateVerdict.FAILED),
        "missing_everything": (lambda e: QualityEvidence(), "point_in_time", GateVerdict.NEEDS_MORE_EVIDENCE),
    }


def verify_gate_can_fail(now="2021-06-01", seed: int = 0) -> dict:
    """Run the clean bundle and every planted defect. `clean_promoted` must be True and `wrong` empty; anything else means a gate
    was silently disabled or a mapping to a verdict drifted."""
    ev, pol = reference_evidence(now, seed)
    gate = QualityGate(pol)
    clean = gate.evaluate("reference", ev, now)
    wrong, seen = {}, {}
    for name, (mut, want_gate, want_verdict) in planted_defects().items():
        d = gate.evaluate(name, mut(ev), now)
        seen[name] = d.verdict.value
        if d.verdict != want_verdict or want_gate not in d.blocking:
            wrong[name] = {"got": d.verdict.value, "blocking": list(d.blocking), "want_gate": want_gate, "want_verdict": want_verdict.value}
    return {"clean_promoted": clean.promote, "clean_blocking": list(clean.blocking), "wrong": wrong, "verdicts": seen}


# ------------------------------------------------------------------------------------------------ public entry
@dataclass(frozen=True)
class Candidate:
    subject_id: str
    evidence: QualityEvidence


@dataclass(frozen=True)
class GateReport:
    now: str
    decisions: tuple[QualityDecision, ...]
    promoted: tuple[str, ...]
    newly_quarantined: tuple[str, ...]
    funnel: Mapping[str, Any]


def step(candidates: Sequence[Candidate], now, policy: QualityPolicy | None = None, store: QuarantineStore | None = None) -> GateReport:
    """The wave-2 research loop's entry: gate every candidate as of `now`, quarantine the tainted ones (skipping any already
    quarantined and refusing to re-evaluate them into PROMOTE without new evidence), and return the funnel."""
    gate = QualityGate(policy)
    decisions, promoted, quarantined = [], [], []
    for c in candidates:
        d = gate.evaluate(c.subject_id, c.evidence, now)
        if store is not None and store.is_quarantined(c.subject_id):
            cur = store.active()[c.subject_id]
            if cur["evidence"] == d.evidence_digest:
                d = dataclasses.replace(d, verdict=GateVerdict.QUARANTINED, blocking=tuple(cur["gates"]), reasons=("still quarantined: evidence unchanged",))
            elif d.verdict not in (GateVerdict.QUARANTINED, GateVerdict.FAILED) and not any(g.gate in INTEGRITY_GATES and not g.ok for g in d.gates):
                store.release(d)
        elif d.verdict == GateVerdict.QUARANTINED and store is not None:
            store.quarantine(d)
            quarantined.append(c.subject_id)
        if d.promote:
            promoted.append(c.subject_id)
        decisions.append(d)
    return GateReport(str(as_date(now)), tuple(decisions), tuple(promoted), tuple(quarantined), funnel(decisions))


# ------------------------------------------------------------------------------------------------ evidence hygiene before the gate runs
def validate_evidence(ev: QualityEvidence, now) -> list[str]:
    """Structural problems in a bundle that would make a gate misjudge it: mismatched lengths, non-finite numbers, dates that are not
    dates, a replication assessment made as of another day. These are caller bugs and are reported before (not instead of) the gate;
    the gate itself still fails closed on anything malformed."""
    errs = []
    if ev.calibration is not None:
        if len(ev.calibration.p) != len(ev.calibration.y):
            errs.append("calibration: p and y differ in length")
        elif ev.calibration.p and not np.isfinite(np.asarray(ev.calibration.p, dtype=float)).all():
            errs.append("calibration: non-finite forecast")
    if ev.oos is not None and ev.oos.oos is not None:
        o = ev.oos.oos
        if len(o.oos_dates) != len(o.oos_effects):
            errs.append("oos: dates and effects differ in length")
        for d in o.oos_dates:
            try:
                as_date(d)
            except (ValueError, TypeError):
                errs.append(f"oos: {d!r} is not a date")
                break
        if o.oos_effects and not np.isfinite(np.asarray(o.oos_effects, dtype=float)).all():
            errs.append("oos: non-finite effect")
    if ev.replication is not None and ev.replication.now != str(as_date(now)):
        errs.append(f"replication assessment was made as of {ev.replication.now}, not {as_date(now)}: it must be re-made as of the decision day")
    if ev.pit is not None:
        names = [f.name for f in ev.pit.features]
        if len(set(names)) != len(names):
            errs.append("pit: duplicate feature names")
        for f in ev.pit.features:
            if f.available_at is not None:
                try:
                    as_date(f.available_at)
                except (ValueError, TypeError):
                    errs.append(f"pit: {f.name} has unreadable available_at {f.available_at!r}")
    if ev.transfer is not None and any(not math.isfinite(float(v)) for v in ev.transfer.context_effects.values()):
        errs.append("transfer: non-finite context effect")
    if ev.justification and len(set(ev.justification)) != len(ev.justification):
        errs.append("justification: repeated metric names")
    return errs


def evidence_completeness(ev: QualityEvidence) -> dict[str, bool]:
    """Which gates have any evidence at all (calibration counts as present when probabilities are not stated, risk when it changes no
    risk-bearing decision). The first thing to read when a decision is NEEDS_MORE_EVIDENCE."""
    return {"point_in_time": ev.pit is not None, "leakage": ev.leak is not None, "identity": ev.identity is not None,
            "out_of_sample": ev.oos is not None, "replication": ev.replication is not None,
            "calibration": ev.calibration is not None or not ev.outputs_probabilities, "risk": ev.risk is not None or not ev.changes_risk_decisions,
            "complexity": ev.complexity is not None, "transfer": ev.transfer is not None, "failure_behavior": ev.failure is not None,
            "reproducibility": ev.repro is not None, "metric_alignment": bool(ev.justification)}


def evidence_plan(d: QualityDecision) -> list[tuple[str, str]]:
    """What to collect next, cheapest integrity checks first: (gate, action) for every gate that asked for evidence rather than
    failing on it. Integrity gates come first because nothing else matters if the discovery is tainted."""
    order = {g: i for i, g in enumerate(GATES)}
    asks = [g for g in d.gates if g.state in (MISSING, UNKNOWN)]
    asks.sort(key=lambda g: (g.gate not in INTEGRITY_GATES, order[g.gate]))
    return [(g.gate, REMEDIATION[g.gate]) for g in asks]


def stage_of(ev: QualityEvidence):
    """Which rung of the section-18 escalation ladder this bundle has reached: cheap screen (some statistics), stronger tests (OOS and
    integrity evidence), cross-year (unseen years and transfer), fresh holdout (replication), integration (everything supplied)."""
    from engine.research.core import Stage
    have = evidence_completeness(ev)
    if all(have.values()):
        return Stage.INTEGRATION
    if have["replication"] and have["out_of_sample"] and have["transfer"]:
        return Stage.FRESH_HOLDOUT
    if have["out_of_sample"] and have["transfer"]:
        return Stage.CROSS_YEAR
    if have["out_of_sample"] or (have["point_in_time"] and have["leakage"]):
        return Stage.STRONGER_TESTS
    return Stage.CHEAP_SCREEN


# ------------------------------------------------------------------------------------------------ building evidence from raw material
def pit_evidence(feature_dates: Mapping[str, str | None], decision_time: str, train_end: str, first_test_start: str, label_horizon_days: int,
                 newest_evidence: str, fills_next_open: bool = True, known_before: Sequence[str] = ()) -> PITEvidence:
    """PITEvidence from a name -> available_at mapping. A name in `known_before` with no date is declared KNOWN_BEFORE_EVENT by the
    caller; a name with no date and no declaration stays UNCERTAIN (and will make the gate answer UNKNOWN, never PASS)."""
    feats = tuple(FeatureUse(n, d, Availability.KNOWN_BEFORE_EVENT if (d or n in known_before) else Availability.UNCERTAIN)
                  for n, d in sorted(feature_dates.items()))
    return PITEvidence(feats, decision_time, label_horizon_days, train_end, first_test_start, fills_next_open, newest_evidence)


def screen_label_leak(X: pd.DataFrame, y: pd.Series, max_abs_corr: float = 0.9, min_rows: int = 50, max_rows: int = 200_000) -> list[str]:
    """Findings for features that reproduce the label: rank correlation with the forward return above `max_abs_corr` (nothing known
    beforehand predicts a weekly return that well), or a name that says it is the future. A screen, not a proof: it can only ADD
    findings to the leak audit."""
    if X.empty or len(X) < min_rows:
        return []
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=0)
    yy = y.reindex(X.index)
    out = []
    for c in X.columns:
        name = str(c).lower()
        if any(t in name for t in ("fwd", "future", "next_", "label", "target", "forward")):
            out.append(f"feature name {c!r} says it is the future")
        col = X[c]
        if col.dtype.kind not in "fiu" or col.nunique() < 3:
            continue
        ok = col.notna() & yy.notna()
        if ok.sum() < min_rows:
            continue
        rho = float(col[ok].rank().corr(yy[ok].rank()))
        if math.isfinite(rho) and abs(rho) >= max_abs_corr:
            out.append(f"feature {c!r} has rank correlation {rho:+.3f} with the forward return: it contains the answer")
    return out


def leak_evidence_from_panel(X: pd.DataFrame, y: pd.Series, firewall: FW.GateVerdict | None, planted_probe_caught: bool | None,
                             outcomes_after_now: int, sealed_touched: Sequence[str] = ()) -> LeakEvidence:
    """LeakEvidence whose findings include the panel screen, so the leak gate quarantines a label-in-features panel even when the
    caller forgot to run the firewall on it."""
    return LeakEvidence(True, firewall, planted_probe_caught, tuple(screen_label_leak(X, y)), tuple(sealed_touched), int(outcomes_after_now))


# ------------------------------------------------------------------------------------------------ comparing and logging decisions
def compare_decisions(a: QualityDecision, b: QualityDecision) -> dict:
    """What changed between two decisions on the same subject: verdict, gates that flipped, gates that newly block or newly clear."""
    ga, gb = {g.gate: g for g in a.gates}, {g.gate: g for g in b.gates}
    flipped = {k: (ga[k].state, gb[k].state) for k in GATES if k in ga and k in gb and ga[k].state != gb[k].state}
    return {"same_subject": a.subject_id == b.subject_id, "verdict": (a.verdict.value, b.verdict.value), "flipped": flipped,
            "newly_blocking": sorted(set(b.blocking) - set(a.blocking)), "newly_clear": sorted(set(a.blocking) - set(b.blocking)),
            "policy_changed": a.policy_digest != b.policy_digest, "evidence_changed": a.evidence_digest != b.evidence_digest}


class DecisionLog(RP.ResearchLane):
    LANE = "qdec"

    """Append-only, hash-chained log of gate decisions. Verdict flips of one subject over time are visible (a promotion that was once
    QUARANTINED needs an explanation), and a verdict may not be rewritten."""

    def add(self, d: QualityDecision) -> None:
        if any(r["body"]["decision_id"] == d.decision_id for r in self.rows()):
            raise FileExistsError(f"decision {d.decision_id} already logged")
        self.append("decision", d.to_dict())

    def history(self, subject_id: str) -> list[tuple[str, str]]:
        return [(r["body"]["now"], r["body"]["verdict"]) for r in self.rows() if r["body"]["subject_id"] == subject_id]

    def flips(self) -> dict[str, list[tuple[str, str]]]:
        """Subjects whose verdict changed between consecutive decisions, with the (from, to) pairs."""
        by: dict[str, list[str]] = {}
        for r in self.rows():
            by.setdefault(r["body"]["subject_id"], []).append(r["body"]["verdict"])
        return {s: [(a, b) for a, b in zip(v, v[1:]) if a != b] for s, v in by.items() if any(a != b for a, b in zip(v, v[1:]))}

    def promoted_after_quarantine(self) -> list[str]:
        """Subjects that were PROMOTE after having been QUARANTINED: legitimate only through QuarantineStore.release, so listed for audit."""
        out = []
        for s in {r["body"]["subject_id"] for r in self.rows()}:
            v = [x for _, x in self.history(s)]
            if "QUARANTINED" in v and "PROMOTE" in v[v.index("QUARANTINED"):]:
                out.append(s)
        return sorted(out)


# ------------------------------------------------------------------------------------------------ is the gate itself tested both ways?
def defect_coverage() -> dict:
    """Coverage in BOTH directions (a gate with no planted defect, or a defect no gate is named for, is untested): which gates have at
    least one planted defect targeting them, which do not, and which verdict kinds the defects exercise."""
    defects = planted_defects()
    targeted = {g for _, g, _ in defects.values()}
    verdicts = {v.value for _, _, v in defects.values()}
    return {"untargeted_gates": sorted(set(GATES) - targeted), "unknown_targets": sorted(targeted - set(GATES)),
            "verdicts_exercised": sorted(verdicts), "verdicts_missing": sorted({v.value for v in GateVerdict} - verdicts - {"PROMOTE"}),
            "n_defects": len(defects)}


def gate_ablation(now="2021-06-01", seed: int = 0) -> dict:
    """Remove one gate at a time from the critical set's reach (by dropping its evidence) and report what the verdict becomes: shows
    that no gate is redundant with another. A gate whose absence changes nothing would be dead weight - or hidden by another."""
    ev, pol = reference_evidence(now, seed)
    gate = QualityGate(pol)
    drop = {"point_in_time": _mut(pit=None), "leakage": _mut(leak=None), "identity": _mut(identity=None), "out_of_sample": _mut(oos=None),
            "replication": _mut(replication=None), "calibration": _mut(calibration=None), "risk": _mut(risk=None), "complexity": _mut(complexity=None),
            "transfer": _mut(transfer=None), "failure_behavior": _mut(failure=None), "reproducibility": _mut(repro=None),
            "metric_alignment": _mut(justification=())}
    out = {}
    for name, fn in drop.items():
        d = gate.evaluate(name, fn(ev), now)
        out[name] = {"verdict": d.verdict.value, "blocking": list(d.blocking), "only_this": d.blocking == (name,)}
    return out
