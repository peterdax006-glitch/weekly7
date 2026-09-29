"""Stop-wasting-compute controller (C66 section 20; canon C63/C66). IMPLEMENTED - NOT VALIDATED.

For every research branch the controller measures PROGRESS PER COMPUTE from the value accountant's records, and when repeated
experiments show the five waste symptoms of the contract - tiny effect, no transfer, no replication, no decision change, high
complexity - it deprioritises the branch and, if the evidence is strong and powered, moves it to DORMANT with a stated reason. It
never deletes anything:

  * dormancy is written to the manager's ladder (`compute_manager.park`) AND to the lifecycle ledger
    (`engine.learning.retirement.RetirementLedger`, ACTIVE -> DORMANT with RecoveryCondition objects), so a dormant idea is gated
    exactly like any other broken knowledge (contract section 20, last paragraph);
  * it returns only when something NEW arrives - new data, a new representation, a new regime, new evidence or a new hypothesis -
    and the new thing must not have been seen at dormancy time (`seen_triggers`), so the same trigger cannot revive it twice;
  * underpowered runs are never waste: a branch whose recent runs could not have seen the effect is not judged (it is either given
    a bigger test by the ladder or parked as INFEASIBLE, which is a resource statement, not a verdict on the idea);
  * a circuit breaker limits dormancy per tick, so a bug in the measurements cannot silently mothball the whole research portfolio;
  * a seeded sample of dormant branches is re-probed cheaply to estimate the controller's own false-dormancy rate.

Builds on: engine.research.compute_manager (states, park/revive), engine.research.value_accounting (ValueLedger, verdicts,
recent_value_rate), engine.learning.retirement (RetirementLedger, RecoveryCondition, Evidence), engine.learning.research_policy
(marginal_return_verdict, novelty_score). Public entry: `step(...)`. State lives in MATURED_RESEARCH_STATE."""
from __future__ import annotations

import dataclasses
import enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.core import FailureCause
from engine.learning.research_policy import marginal_return_verdict, novelty_score
from engine.learning.retirement import RecoveryCondition, RetirementLedger, State as LifeState
from engine.research import compute_manager as CM
from engine.research import value_accounting as VA
from engine.research.core import Namespace, ResearchState, as_date, stable_hash

NAMESPACE = Namespace.MATURED_RESEARCH


class Action(str, enum.Enum):
    CONTINUE = "CONTINUE"
    PROTECT = "PROTECT"              # recent progress: exempt from judgement
    UNRESOLVED = "UNRESOLVED"        # underpowered: not judged; the ladder should repeat bigger
    DEPRIORITISE = "DEPRIORITISE"
    DORMANT = "DORMANT"


class Reason(str, enum.Enum):
    NO_PROGRESS_PER_COMPUTE = "NO_PROGRESS_PER_COMPUTE"
    TINY_EFFECT_NO_TRANSFER = "TINY_EFFECT_NO_TRANSFER"
    UNREPLICATED_DRIFT = "UNREPLICATED_DRIFT"
    COMPLEXITY_WITHOUT_GAIN = "COMPLEXITY_WITHOUT_GAIN"
    FAMILY_UNRELIABLE = "FAMILY_UNRELIABLE"
    INFEASIBLE_UNDERPOWERED = "INFEASIBLE_UNDERPOWERED"
    DUPLICATED_EFFORT = "DUPLICATED_EFFORT"


# the lifecycle cause the retirement ledger records for each reason (retirement.py owns the vocabulary)
CAUSE_OF = {Reason.NO_PROGRESS_PER_COMPUTE: FailureCause.INSUFFICIENT_EVIDENCE, Reason.TINY_EFFECT_NO_TRANSFER: FailureCause.WEAKENING_EFFECT,
            Reason.UNREPLICATED_DRIFT: FailureCause.INSUFFICIENT_EVIDENCE, Reason.COMPLEXITY_WITHOUT_GAIN: FailureCause.REDUNDANCY,
            Reason.FAMILY_UNRELIABLE: FailureCause.WRONG_CONTEXT, Reason.INFEASIBLE_UNDERPOWERED: FailureCause.INSUFFICIENT_EVIDENCE,
            Reason.DUPLICATED_EFFORT: FailureCause.REDUNDANCY}


# ------------------------------------------------------------------------------------------------ policy
@dataclasses.dataclass(frozen=True)
class WastePolicy:
    window: int = 6                        # most recent experiments considered
    min_experiments: int = 4               # fewer than this: too early to judge
    min_spend_cpu_min: float = 8.0         # a branch that has cost less than this is never called wasteful
    rate_floor: float = 0.0004             # gross value units per CPU-minute below which progress is "none"
    relative_floor: float = 0.25           # ... or below this fraction of the family's own rate
    tiny_effect_ratio: float = 0.5         # median |effect| / min_effect below this is a TINY effect
    min_effect: float = 0.0025             # smallest useful effect (matches the ladder's)
    min_power: float = 0.6                 # runs below this power cannot count as evidence of no progress
    min_powered_runs: int = 3              # powered runs needed in the window before dormancy is allowed
    min_symptoms: int = 3                  # measured-true symptoms needed for DORMANT
    deprioritise_symptoms: int = 2
    complexity_units_limit: float = 12.0   # cumulative added parameters/features without proportional gain
    protect_runs: int = 3                  # a USEFUL/PREVENTIVE result within this many runs protects the branch
    deprioritise_factor: float = 0.5
    factor_floor: float = 0.1
    max_dormant_fraction: float = 0.15     # circuit breaker: at most this share of ACTIVE branches per tick (min 1)
    revival_cooldown_days: int = 30
    max_revivals: int = 3
    min_new_data_fraction: float = 0.10
    min_new_evidence: int = 5
    min_evidence_signal_t: float = 2.0
    probe_days: int = 90
    duplicate_novelty: float = 0.08        # config novelty below this = the same experiment again

    def validate(self) -> list[str]:
        errs = []
        if self.min_experiments < 2 or self.window < self.min_experiments:
            errs.append("window >= min_experiments >= 2 required")
        if not 0 < self.deprioritise_factor <= 1 or not 0 < self.factor_floor <= self.deprioritise_factor:
            errs.append("0 < factor_floor <= deprioritise_factor <= 1 required")
        if self.min_symptoms < self.deprioritise_symptoms or self.min_symptoms > 5:
            errs.append("deprioritise_symptoms <= min_symptoms <= 5 required")
        if not 0 < self.max_dormant_fraction <= 1 or self.min_effect <= 0 or self.rate_floor < 0:
            errs.append("max_dormant_fraction in (0,1], min_effect > 0, rate_floor >= 0 required")
        if self.min_powered_runs > self.window or not 0.5 <= self.min_power < 1:
            errs.append("min_powered_runs <= window and min_power in [0.5,1) required")
        if self.revival_cooldown_days < 0 or self.max_revivals < 0:
            errs.append("cooldown/max_revivals must be >= 0")
        return errs


# ------------------------------------------------------------------------------------------------ telemetry
@dataclasses.dataclass(frozen=True)
class Telemetry:
    """The recent experiments of one branch as the controller sees them: per-run cost, gross value, effect, power and the
    measured symptoms. Built from the manager's run log and the value accountant's records for the branch."""
    branch_id: str
    costs: tuple[float, ...]
    gross: tuple[float, ...]                  # value before paying for compute, per experiment
    effects: tuple[float, ...]
    powers: tuple[float, ...]
    verdicts: tuple[str, ...]
    transfer: tuple[float | None, ...]
    decision: tuple[float | None, ...]
    replicated: tuple[bool, ...]
    complexity: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.costs)


def collect_telemetry(state: CM.ManagerState, ledger: VA.ValueLedger, branch_id: str, now, window: int) -> Telemetry:
    """Gather the last `window` experiments of a branch. Records are joined by branch_id and only those finished strictly before
    `now` are visible; run-log entries supply effect and power. Missing measurements stay None (never zero)."""
    recs = [r for r in ledger.records(now) if r.branch_id == branch_id and r.verdict != VA.Verdict.CORRECTION.value][-window:]
    runs = [r for r in state.branches[branch_id].runs if "effect" in r and as_date(r["at"]) < as_date(now)][-window:]
    g = tuple(max(0.0, r.net_value + r.cost_value) for r in recs)
    return Telemetry(
        branch_id, tuple(r.cost_cpu_min for r in recs) or tuple(float(x.get("cost", 0.0)) for x in runs), g,
        tuple(abs(float(x["effect"])) for x in runs), tuple(float(x.get("power", 0.0)) for x in runs),
        tuple(r.verdict for r in recs), tuple(r.value.get("transfer_change") for r in recs),
        tuple(r.value.get("decision_change") for r in recs), tuple(bool(r.value.get("replicated")) for r in recs),
        tuple(float(r.value.get("complexity_units") or 0.0) for r in recs))


@dataclasses.dataclass(frozen=True)
class Symptoms:
    """Each symptom is True / False / None (unmeasured). Unmeasured is NEVER counted as present."""
    tiny_effect: bool | None
    no_transfer: bool | None
    no_replication: bool | None
    no_decision_change: bool | None
    high_complexity: bool | None

    def flags(self) -> dict[str, bool | None]:
        return dataclasses.asdict(self)

    @property
    def present(self) -> int:
        return sum(1 for v in self.flags().values() if v is True)

    @property
    def measured(self) -> int:
        return sum(1 for v in self.flags().values() if v is not None)


def evaluate_symptoms(t: Telemetry, pol: WastePolicy) -> Symptoms:
    tiny = None
    if t.effects:
        tiny = float(np.median(t.effects)) / pol.min_effect < pol.tiny_effect_ratio
    tr = [x for x in t.transfer if x is not None]
    no_tr = (max(tr) < 0.10) if tr else None
    no_rep = (not any(t.replicated)) if t.verdicts else None
    dec = [x for x in t.decision if x is not None]
    no_dec = (max(dec) < 0.02) if dec else None
    cx = None
    if t.verdicts:
        total_cx = sum(t.complexity)
        useful = sum(1 for v in t.verdicts if v in (VA.Verdict.USEFUL.value, VA.Verdict.PREVENTIVE.value))
        cx = total_cx >= pol.complexity_units_limit and useful == 0
    return Symptoms(tiny, no_tr, no_rep, no_dec, cx)


@dataclasses.dataclass(frozen=True)
class Progress:
    rate: float | None                 # gross value units per CPU-minute over the window
    family_rate: float | None
    relative: float | None             # rate / family_rate when the family has a positive rate
    marginal: str                      # research_policy.marginal_return_verdict on the per-experiment gross values
    total_cost: float
    powered_runs: int
    n: int


def progress_per_compute(t: Telemetry, family_rate: float | None, pol: WastePolicy) -> Progress:
    """Gross value per CPU-minute for the window, against the family's own rate, plus Wald-style 'has the trend stopped?' from
    research_policy.marginal_return_verdict. None where there is nothing to divide by."""
    cost = float(sum(t.costs))
    rate = (sum(t.gross) / cost) if cost > 0 and t.gross else None
    rel = (rate / family_rate) if (rate is not None and family_rate is not None and family_rate > 1e-12) else None
    per_exp = [g / c for g, c in zip(t.gross, t.costs) if c > 0]
    mv = marginal_return_verdict(per_exp, k=min(4, max(2, len(per_exp)) ), eps=max(pol.rate_floor, 1e-9))["verdict"] if len(per_exp) >= 2 else "INSUFFICIENT"
    return Progress(rate, family_rate, rel, mv, cost, sum(1 for p in t.powers if p >= pol.min_power), t.n)


@dataclasses.dataclass(frozen=True)
class WasteVerdict:
    branch_id: str
    action: Action
    reason: Reason | None
    multiplier: float                  # priority factor to apply (1.0 = none)
    symptoms: Symptoms | None
    progress: Progress | None
    notes: tuple[str, ...]


def assess_branch(state: CM.ManagerState, ledger: VA.ValueLedger, branch_id: str, now, pol: WastePolicy,
                  family_rate: float | None = None, unreliable_families: Iterable[str] = ()) -> WasteVerdict:
    """The controller's judgement of one branch. Ordered guards first (each returns a reasoned CONTINUE/PROTECT/UNRESOLVED), then
    the symptom-and-rate test. Nothing here mutates state."""
    b = state.branches[branch_id]
    if b.state not in CM.ACTIVE_STATES:
        return WasteVerdict(branch_id, Action.CONTINUE, None, b.waste_factor, None, None, (f"branch is {b.state.value}",))
    if b.family in set(unreliable_families):
        return WasteVerdict(branch_id, Action.DORMANT, Reason.FAMILY_UNRELIABLE, pol.factor_floor, None, None,
                            ("its feature family carries a standing 'unreliable' finding",))
    t = collect_telemetry(state, ledger, branch_id, now, pol.window)
    if t.n < pol.min_experiments:
        return WasteVerdict(branch_id, Action.CONTINUE, None, 1.0, None, None, (f"only {t.n} experiments: too early to judge",))
    prog = progress_per_compute(t, family_rate, pol)
    if prog.total_cost < pol.min_spend_cpu_min:
        return WasteVerdict(branch_id, Action.CONTINUE, None, 1.0, None, prog, ("too little compute spent to call it wasteful",))
    if any(v in (VA.Verdict.USEFUL.value, VA.Verdict.PREVENTIVE.value) for v in t.verdicts[-pol.protect_runs:]):
        return WasteVerdict(branch_id, Action.PROTECT, None, 1.0, None, prog, ("a recent result was useful: protected",))
    sym = evaluate_symptoms(t, pol)
    if prog.powered_runs < pol.min_powered_runs:
        return WasteVerdict(branch_id, Action.UNRESOLVED, None, 1.0, sym, prog,
                            (f"only {prog.powered_runs} of {t.n} runs had the power to see the effect: not judged, needs a bigger test",))
    low_rate = prog.rate is not None and (prog.rate < pol.rate_floor or (prog.relative is not None and prog.relative < pol.relative_floor))
    stopped = prog.marginal == "STOP"
    notes = [f"rate={prog.rate if prog.rate is None else round(prog.rate, 6)} floor={pol.rate_floor} relative={prog.relative if prog.relative is None else round(prog.relative, 2)}",
             f"marginal-return: {prog.marginal}", f"symptoms present {sym.present}/{sym.measured} measured"]
    if low_rate and sym.present >= pol.min_symptoms and (stopped or prog.marginal == "INSUFFICIENT"):
        reason = pick_reason(sym, prog)
        return WasteVerdict(branch_id, Action.DORMANT, reason, pol.factor_floor, sym, prog, tuple(notes))
    if low_rate and sym.present >= pol.deprioritise_symptoms or (stopped and sym.present >= pol.deprioritise_symptoms):
        mult = max(pol.factor_floor, pol.deprioritise_factor ** max(1, sym.present - pol.deprioritise_symptoms + 1))
        return WasteVerdict(branch_id, Action.DEPRIORITISE, None, mult, sym, prog, tuple(notes))
    return WasteVerdict(branch_id, Action.CONTINUE, None, 1.0, sym, prog, tuple(notes))


def pick_reason(sym: Symptoms, prog: Progress) -> Reason:
    """The most specific stated reason for dormancy, in order of how informative it is for later revival."""
    if sym.high_complexity:
        return Reason.COMPLEXITY_WITHOUT_GAIN
    if sym.tiny_effect and sym.no_transfer:
        return Reason.TINY_EFFECT_NO_TRANSFER
    if sym.no_replication and sym.no_decision_change:
        return Reason.UNREPLICATED_DRIFT
    return Reason.NO_PROGRESS_PER_COMPUTE


# @@APPEND@@
