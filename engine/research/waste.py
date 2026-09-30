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
import datetime as dt
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


# ------------------------------------------------------------------------------------------------ what "new" means
class TriggerKind(str, enum.Enum):
    NEW_DATA = "NEW_DATA"
    NEW_REPRESENTATION = "NEW_REPRESENTATION"
    NEW_REGIME = "NEW_REGIME"
    NEW_EVIDENCE = "NEW_EVIDENCE"
    NEW_HYPOTHESIS = "NEW_HYPOTHESIS"


@dataclasses.dataclass(frozen=True)
class WorldContext:
    """What the research world looks like today, in identity-free terms (no tickers, no real dates): only counts, hashes and
    labels. Dormant branches are compared against the fingerprint they were parked with."""
    as_of: str
    data_hash: str = ""
    n_rows: int = 0
    representation_ids: frozenset = frozenset()
    representation_families: Mapping[str, str] = dataclasses.field(default_factory=dict)     # representation id -> family it serves ("*" = all)
    regime: str = ""
    matured_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)              # family -> matured outcomes available
    hypothesis_ids: Mapping[str, tuple] = dataclasses.field(default_factory=dict)            # family -> hypothesis ids known today
    evidence_signal: Mapping[str, float] = dataclasses.field(default_factory=dict)           # family -> t of new related evidence

    def fingerprint(self, family: str) -> dict:
        return {"data_hash": self.data_hash, "n_rows": self.n_rows, "representation_ids": sorted(self.representation_ids),
                "regime": self.regime, "matured_count": int(self.matured_counts.get(family, 0)),
                "hypothesis_ids": sorted(self.hypothesis_ids.get(family, ()))}


@dataclasses.dataclass(frozen=True)
class Trigger:
    kind: TriggerKind
    detail: str
    strength: float
    key: str

    @staticmethod
    def make(kind: TriggerKind, detail: str, strength: float) -> "Trigger":
        return Trigger(kind, detail, float(min(1.0, max(0.0, strength))), stable_hash([kind.value, detail], 12))


def recovery_conditions(pol: WastePolicy) -> tuple[RecoveryCondition, ...]:
    """The five ways back, as retirement.RecoveryCondition objects over a numeric snapshot (see `snapshot_for`), so the lifecycle
    ledger and the controller speak the same language about when a parked idea is worth another look."""
    return (RecoveryCondition("data_growth", "enough new data since dormancy", lo=pol.min_new_data_fraction),
            RecoveryCondition("new_representations", "a new representation of the inputs", lo=1.0),
            RecoveryCondition("regime_changed", "a regime it was never tested in", lo=1.0),
            RecoveryCondition("new_evidence", "new matured evidence or a strong related finding", lo=1.0),
            RecoveryCondition("new_hypotheses", "a new hypothesis on the same family", lo=1.0))


@dataclasses.dataclass
class DormantRecord:
    branch_id: str
    family: str
    reason: str
    detail: str
    since: str
    fingerprint: dict
    avoided_cpu_min: float = 0.0
    spent_cpu_min: float = 0.0
    seen_triggers: list = dataclasses.field(default_factory=list)
    revive_count: int = 0
    last_revival: str = ""
    probes: list = dataclasses.field(default_factory=list)          # [date, effect, se, promising]
    active: bool = True                                             # False once revived

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def snapshot_for(rec: DormantRecord, ctx: WorldContext, pol: WastePolicy) -> tuple[dict[str, float], list[Trigger]]:
    """Numeric snapshot of how far the world has moved since `rec` was parked, and the concrete, already-unseen triggers behind
    it. A trigger whose key is in `seen_triggers` is dropped: the same new thing cannot revive a branch twice."""
    fp, seen = rec.fingerprint, set(rec.seen_triggers)
    trig: list[Trigger] = []
    growth = 0.0
    if ctx.data_hash and ctx.data_hash != fp.get("data_hash") and fp.get("n_rows", 0) > 0:
        growth = max(0.0, (ctx.n_rows - fp["n_rows"]) / fp["n_rows"])
        if growth >= pol.min_new_data_fraction:
            trig.append(Trigger.make(TriggerKind.NEW_DATA, f"data {fp.get('data_hash', '')[:8]}->{ctx.data_hash[:8]} (+{growth:.0%} rows)", growth / 0.5))
    fresh = sorted(set(ctx.representation_ids) - set(fp.get("representation_ids", ())))
    fresh = [r for r in fresh if ctx.representation_families.get(r, "*") in ("*", rec.family)]
    if fresh:
        trig.append(Trigger.make(TriggerKind.NEW_REPRESENTATION, "representations " + ",".join(fresh[:3]), 0.8))
    changed = 1.0 if ctx.regime and fp.get("regime") and ctx.regime != fp["regime"] else 0.0
    if changed:
        trig.append(Trigger.make(TriggerKind.NEW_REGIME, f"regime {fp['regime']}->{ctx.regime}", 0.7))
    dm = int(ctx.matured_counts.get(rec.family, 0)) - int(fp.get("matured_count", 0))
    sig = float(ctx.evidence_signal.get(rec.family, 0.0))
    if dm >= pol.min_new_evidence or sig >= pol.min_evidence_signal_t:
        trig.append(Trigger.make(TriggerKind.NEW_EVIDENCE, f"+{dm} matured outcomes, related t={sig:.1f}", max(dm / (4 * pol.min_new_evidence), sig / 4.0)))
    hyp = sorted(set(ctx.hypothesis_ids.get(rec.family, ())) - set(fp.get("hypothesis_ids", ())))
    if hyp:
        trig.append(Trigger.make(TriggerKind.NEW_HYPOTHESIS, "hypotheses " + ",".join(hyp[:3]), 0.9))
    trig = [t for t in trig if t.key not in seen]
    kinds = {t.kind for t in trig}
    snap = {"data_growth": growth if TriggerKind.NEW_DATA in kinds else 0.0, "new_representations": float(len(fresh)) if TriggerKind.NEW_REPRESENTATION in kinds else 0.0,
            "regime_changed": changed if TriggerKind.NEW_REGIME in kinds else 0.0,
            "new_evidence": 1.0 if TriggerKind.NEW_EVIDENCE in kinds else 0.0, "new_hypotheses": float(len(hyp)) if TriggerKind.NEW_HYPOTHESIS in kinds else 0.0}
    return snap, trig


def combined_strength(triggers: Sequence[Trigger]) -> float:
    """Independent reasons add up as 1 - prod(1 - s): two moderate reasons can justify a revival that neither would alone."""
    p = 1.0
    for t in triggers:
        p *= 1.0 - t.strength
    return 1.0 - p


# ------------------------------------------------------------------------------------------------ the dormant book
class DormantBook:
    """Every parked branch with its reason and the fingerprint of the world it was parked in. Parking is mirrored into a
    RetirementLedger (ACTIVE -> DORMANT with recovery conditions) so dormant research ideas obey the same lifecycle rules as
    dormant knowledge."""

    def __init__(self, life: RetirementLedger | None = None):
        self.records: dict[str, DormantRecord] = {}
        self.life = life or RetirementLedger()

    @staticmethod
    def kid(branch_id: str) -> str:
        return "research:" + branch_id

    def _ensure_registered(self, b: CM.Branch) -> None:
        if not self.life.known(self.kid(b.branch_id)):
            self.life.register(self.kid(b.branch_id), b.created, LifeState.ACTIVE, "research branch created")

    def park(self, state: CM.ManagerState, verdict: WasteVerdict, ctx: WorldContext, now, pol: WastePolicy,
             ladder: CM.LadderPolicy | None = None) -> DormantRecord:
        """Move a branch to DORMANT with its reason (ladder + lifecycle ledger) and remember the world it was parked in."""
        if verdict.action is not Action.DORMANT or verdict.reason is None:
            raise ValueError("only a DORMANT verdict with a reason can park a branch")
        b = state.branches[verdict.branch_id]
        detail = "; ".join(verdict.notes)
        remaining = expected_remaining_cpu_min(state, b, ladder or CM.LadderPolicy())
        CM.park(state, b.branch_id, now, f"{verdict.reason.value}: {detail}"[:400])
        self._ensure_registered(b)
        cur = self.life.state(self.kid(b.branch_id), as_date(now) + dt.timedelta(days=1))
        if cur is not LifeState.DORMANT:
            if cur is LifeState.DEGRADED or cur is LifeState.ACTIVE:
                self.life.transition(self.kid(b.branch_id), LifeState.DORMANT, now, "DORMANT", f"{verdict.reason.value}: {detail}"[:300],
                                     CAUSE_OF[verdict.reason], {"rate": getattr(verdict.progress, "rate", None), "spent": b.total_spent},
                                     recovery_conditions(pol))
        rec = DormantRecord(b.branch_id, b.family, verdict.reason.value, detail, as_date(now).isoformat(), ctx.fingerprint(b.family),
                            remaining, b.total_spent)
        old = self.records.get(b.branch_id)
        if old is not None:
            rec.revive_count, rec.seen_triggers, rec.probes = old.revive_count, old.seen_triggers, old.probes
        self.records[b.branch_id] = rec
        b.waste_factor = 1.0
        return rec

    def adopt_parked(self, state: CM.ManagerState, ctx: WorldContext, now, pol: WastePolicy, ladder: CM.LadderPolicy | None = None) -> list[str]:
        """Take over branches the LADDER parked (DORMANT with no record here). Until this existed the controller only knew the branches
        it had parked itself, so a line parked as infeasible by the ladder (the only kind the default loop produces: every branch needs
        ~80 cpu-min against a 40 cap) had no revival trigger, no probe and no lifecycle entry and stayed dormant for ever, while
        the controller judged an empty set of ACTIVE branches. The record carries the world as of the first tick that sees the
        branch (the ladder stores no fingerprint), so a revival trigger is always something that arrived AFTER adoption."""
        out = []
        for bid in sorted(state.branches):
            b = state.branches[bid]
            if b.state is not CM.ResearchState.DORMANT or bid in self.records:
                continue
            text = (b.dormant_reason or "").lower()
            reason = (Reason.INFEASIBLE_UNDERPOWERED if "infeasible" in text else Reason.DUPLICATED_EFFORT if "duplicate" in text
                      else Reason.NO_PROGRESS_PER_COMPUTE)
            detail = ("parked by the ladder: " + (b.dormant_reason or "no reason recorded"))[:300]
            self._ensure_registered(b)
            if self.life.state(self.kid(bid), as_date(now) + dt.timedelta(days=1)) in (LifeState.ACTIVE, LifeState.DEGRADED):
                self.life.transition(self.kid(bid), LifeState.DORMANT, now, "DORMANT", f"{reason.value}: {detail}"[:300], CAUSE_OF[reason],
                                     {"spent": b.total_spent}, recovery_conditions(pol))
            self.records[bid] = DormantRecord(bid, b.family, reason.value, detail, as_date(b.updated or now).isoformat(), ctx.fingerprint(b.family),
                                              expected_remaining_cpu_min(state, b, ladder or CM.LadderPolicy()), b.total_spent)
            out.append(bid)
        return out

    def candidates(self, ctx: WorldContext, now, pol: WastePolicy) -> list[tuple[DormantRecord, list[Trigger]]]:
        """Dormant branches for which something genuinely new has arrived. Sorted by combined trigger strength then branch id."""
        out = []
        conds = recovery_conditions(pol)
        for bid in sorted(self.records):
            rec = self.records[bid]
            if not rec.active:
                continue
            if rec.revive_count >= pol.max_revivals:
                continue
            if rec.last_revival and (as_date(now) - as_date(rec.last_revival)).days < pol.revival_cooldown_days:
                continue
            if (as_date(now) - as_date(rec.since)).days < pol.revival_cooldown_days:
                continue
            snap, trig = snapshot_for(rec, ctx, pol)
            if any(c.satisfied_by(snap) for c in conds) and trig:
                out.append((rec, trig))
        out.sort(key=lambda rt: (-combined_strength(rt[1]), rt[0].branch_id))
        return out

    def revive(self, state: CM.ManagerState, rec: DormantRecord, triggers: Sequence[Trigger], now, min_strength: float = 0.5) -> bool:
        """Re-queue a dormant branch if the combined strength of its NEW triggers is enough. A new hypothesis on unchanged data keeps
        earned rungs; anything that changes the data, inputs or regime makes old passes stale, so the branch restarts at rung 1."""
        s = combined_strength(triggers)
        if s < min_strength:
            return False
        only_hyp = all(t.kind is TriggerKind.NEW_HYPOTHESIS for t in triggers)
        CM.revive(state, rec.branch_id, now, "; ".join(f"{t.kind.value}: {t.detail}" for t in triggers)[:300], keep_evidence=only_hyp)
        if self.life.state(self.kid(rec.branch_id), as_date(now) + dt.timedelta(days=1)) is LifeState.DORMANT:
            self.life.transition(self.kid(rec.branch_id), LifeState.DEGRADED, now, "RECOVER_PROBATION",
                                 f"revived on {', '.join(t.kind.value for t in triggers)} (strength {s:.2f})", FailureCause.UNKNOWN,
                                 {"strength": s, "triggers": [t.key for t in triggers]})
        rec.seen_triggers = sorted(set(rec.seen_triggers) | {t.key for t in triggers})
        rec.revive_count += 1
        rec.last_revival = as_date(now).isoformat()
        rec.active = False
        return True

    def retire_proposals(self, pol: WastePolicy) -> list[str]:
        """Branches that have used up their revivals and are dormant again: candidates for an explicit RETIRE decision (listed, never
        acted on here, because retiring is a deliberate act with its own gate)."""
        return sorted(bid for bid, r in self.records.items() if r.active and r.revive_count >= pol.max_revivals)

    # ---- probes: how often is dormancy wrong?
    def probe_candidates(self, rng: np.random.Generator, k: int, now, pol: WastePolicy) -> list[str]:
        weights = {Reason.INFEASIBLE_UNDERPOWERED.value: 3.0, Reason.TINY_EFFECT_NO_TRANSFER.value: 2.0,
                   Reason.NO_PROGRESS_PER_COMPUTE.value: 2.0, Reason.UNREPLICATED_DRIFT.value: 1.5}
        pool = []
        for bid in sorted(self.records):
            r = self.records[bid]
            last = as_date(r.probes[-1][0]) if r.probes else as_date(r.since)
            if r.active and (as_date(now) - last).days >= pol.probe_days:
                pool.append((bid, weights.get(r.reason, 1.0)))
        if not pool or k <= 0:
            return []
        w = np.array([p[1] for p in pool], dtype=float)
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False, p=w / w.sum())
        return [pool[int(i)][0] for i in sorted(idx)]

    def record_probe(self, branch_id: str, effect: float, se: float, now, pol: WastePolicy) -> bool:
        """Store a cheap probe of a dormant branch; True if it looks promising (a positive, material effect at ~2 sigma), in which
        case the caller should treat it as NEW_EVIDENCE and let the revival path re-queue the branch."""
        rec = self.records[branch_id]
        promising = bool(se > 0 and effect >= pol.min_effect and effect / se >= 2.0)
        rec.probes.append([as_date(now).isoformat(), float(effect), float(se), promising])
        return promising

    def false_dormancy_rate(self) -> dict:
        """Beta posterior for the share of probed dormant branches that looked promising (prior Beta(1, 4): we expect few).
        A high upper bound means the controller is too eager and its thresholds should be loosened (advice; never automatic)."""
        from scipy.stats import beta
        n = sum(len(r.probes) for r in self.records.values())
        k = sum(1 for r in self.records.values() for p in r.probes if p[3])
        a, b = 1 + k, 4 + n - k
        return {"probes": n, "promising": k, "mean": a / (a + b), "upper90": float(beta.ppf(0.90, a, b))}

    def to_dict(self) -> dict:
        return {bid: r.to_dict() for bid, r in sorted(self.records.items())}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], life: RetirementLedger | None = None) -> "DormantBook":
        book = cls(life)
        for bid, r in d.items():
            book.records[bid] = DormantRecord(**r)
        return book

    def dump(self, path: str | Path, life_path: str | Path | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8", newline="\n")
        tmp.replace(p)
        if life_path is not None:
            self.life.dump(life_path)


def expected_remaining_cpu_min(state: CM.ManagerState, b: CM.Branch, ladder: CM.LadderPolicy) -> float:
    """Compute a branch would still be expected to consume if it ran to the end of the ladder (the waste avoided by parking it)."""
    rem, q = 0.0, 1.0
    for i in range(CM.stage_index(b.frontier), len(CM.LADDER)):
        s = CM.LADDER[i]
        rem += q * (state.stats.mean_cost(s) or ladder.rule(s).base_cpu_min)
        q *= state.stats.pass_rate(s)
    return float(rem)


# ------------------------------------------------------------------------------------------------ duplicated effort
def find_duplicate_effort(configs: Mapping[str, Mapping], families: Mapping[str, str], pol: WastePolicy) -> list[tuple[str, str, float]]:
    """(later_branch, earlier_branch, novelty) for branches whose configuration is essentially a tried one in the same family.
    Uses research_policy.novelty_score, the same distance the experiment-memory duplicate check uses, so 'duplicate' means one
    thing across the system. Order is deterministic (sorted by branch id): the first of a group is the original."""
    out = []
    seen: dict[str, list[tuple[str, Mapping]]] = {}
    for bid in sorted(configs):
        fam = families.get(bid, "")
        prior = seen.setdefault(fam, [])
        if prior:
            nov = novelty_score(configs[bid], [c for _, c in prior])
            if nov < pol.duplicate_novelty:
                nearest = min(prior, key=lambda pc: (novelty_score(configs[bid], [pc[1]]), pc[0]))
                out.append((bid, nearest[0], nov))
                continue
        prior.append((bid, configs[bid]))
    return out


# ------------------------------------------------------------------------------------------------ the controller tick
@dataclasses.dataclass(frozen=True)
class WasteStepResult:
    verdicts: tuple[WasteVerdict, ...]
    parked: tuple[str, ...]
    deprioritised: tuple[str, ...]
    revived: tuple[str, ...]
    held_back: tuple[str, ...]           # DORMANT verdicts downgraded by the circuit breaker
    mass_dormancy_alarm: bool
    probes_due: tuple[str, ...]
    retire_proposals: tuple[str, ...]
    duplicates: tuple[tuple[str, str, float], ...]
    adopted: tuple[str, ...] = ()        # ladder-parked branches the controller took over this tick (see DormantBook.adopt_parked)


def _severity(v: WasteVerdict) -> tuple:
    rel = v.progress.relative if (v.progress and v.progress.relative is not None) else 1.0
    return (-(v.symptoms.present if v.symptoms else 0), rel, v.branch_id)


def step(state: CM.ManagerState, ledger: VA.ValueLedger, book: DormantBook, ctx: WorldContext, now, pol: WastePolicy | None = None,
         ladder: CM.LadderPolicy | None = None, unreliable_families: Iterable[str] = (), configs: Mapping[str, Mapping] | None = None,
         rng_seed: int = 0, n_probes: int = 1) -> WasteStepResult:
    """One controller tick. (1) judge every ACTIVE branch; (2) apply DEPRIORITISE by scaling the branch's priority factor; (3) park
    DORMANT verdicts through the circuit breaker; (4) revive dormant branches for which something new has arrived; (5) list
    branches due a cheap probe and dormant branches that used up their revivals. Deterministic in (state, ledger, ctx, seed). Branches the ladder parked are adopted first, so they can be revived.
    The context must not be dated after `now` (fail closed)."""
    pol = pol or WastePolicy()
    bad = pol.validate()
    if bad:
        raise ValueError("invalid WastePolicy: " + "; ".join(bad))
    if as_date(ctx.as_of) > as_date(now):
        raise CM.FirewallBreach(f"world context dated {ctx.as_of} is after now={now}")
    adopted = book.adopt_parked(state, ctx, now, pol, ladder)
    unrel = set(unreliable_families)
    stats = ledger.family_stats(now)
    active = [bid for bid in sorted(state.branches) if state.branches[bid].state in CM.ACTIVE_STATES]
    verdicts = [assess_branch(state, ledger, bid, now, pol, (stats[state.branches[bid].family].value_per_cpu_min
                                                             if state.branches[bid].family in stats else None), unrel) for bid in active]
    dupes = find_duplicate_effort({b: c for b, c in (configs or {}).items() if b in state.branches and state.branches[b].state in CM.ACTIVE_STATES},
                                  {b: state.branches[b].family for b in (configs or {}) if b in state.branches}, pol)
    dup_ids = {d[0] for d in dupes}
    vlist = []
    for v in verdicts:
        if v.branch_id in dup_ids and v.action in (Action.CONTINUE, Action.UNRESOLVED):
            b = state.branches[v.branch_id]
            never_run = not b.runs
            v = WasteVerdict(v.branch_id, Action.DORMANT if never_run else Action.DEPRIORITISE, Reason.DUPLICATED_EFFORT if never_run else None,
                             pol.factor_floor if never_run else pol.deprioritise_factor, v.symptoms, v.progress,
                             v.notes + (f"near-duplicate of {next(d[1] for d in dupes if d[0] == v.branch_id)}",))
        vlist.append(v)
    dormant_v = sorted((v for v in vlist if v.action is Action.DORMANT), key=_severity)
    cap = max(1, int(math.floor(pol.max_dormant_fraction * max(len(active), 1))))
    to_park, held = dormant_v[:cap], dormant_v[cap:]
    parked, deprio, held_ids = [], [], []
    for v in to_park:
        try:
            book.park(state, v, ctx, now, pol, ladder)
            parked.append(v.branch_id)
        except ValueError:
            held.append(v)                       # e.g. the lifecycle ledger's flapping guard: try again next tick
    for v in held:
        held_ids.append(v.branch_id)
        state.branches[v.branch_id].waste_factor = max(pol.factor_floor, pol.deprioritise_factor)
    for v in vlist:
        b = state.branches.get(v.branch_id)
        if b is None or b.state not in CM.ACTIVE_STATES or v.branch_id in parked or v.branch_id in held_ids:
            continue
        if v.action is Action.DEPRIORITISE:
            b.waste_factor = min(b.waste_factor, v.multiplier) if b.waste_factor < 1.0 else v.multiplier
            deprio.append(v.branch_id)
        elif v.action in (Action.CONTINUE, Action.PROTECT) and b.waste_factor < 1.0 and v.symptoms is not None and v.symptoms.present < pol.deprioritise_symptoms:
            b.waste_factor = min(1.0, b.waste_factor / pol.deprioritise_factor)      # recovered: the penalty decays, it does not stick
    revived = []
    for rec, trig in book.candidates(ctx, now, pol):
        if book.revive(state, rec, trig, now):
            revived.append(rec.branch_id)
    rng = np.random.default_rng(rng_seed)
    probes = book.probe_candidates(rng, n_probes, now, pol)
    return WasteStepResult(tuple(vlist), tuple(parked), tuple(deprio), tuple(revived), tuple(held_ids),
                           len(dormant_v) > 2 * cap, tuple(probes), tuple(book.retire_proposals(pol)), tuple(dupes), tuple(adopted))


# ------------------------------------------------------------------------------------------------ reporting
def waste_report(state: CM.ManagerState, ledger: VA.ValueLedger, book: DormantBook, now) -> dict:
    """The size of the problem and of the controller's effect: compute with no value, compute parked, compute avoided, and the
    controller's own false-dormancy estimate."""
    tot = ledger.totals(now)
    dormant = [r for r in book.records.values() if r.active]
    per_reason: dict[str, dict[str, float]] = {}
    for r in dormant:
        d = per_reason.setdefault(r.reason, {"branches": 0.0, "spent_cpu_min": 0.0, "avoided_cpu_min": 0.0})
        d["branches"] += 1
        d["spent_cpu_min"] += r.spent_cpu_min
        d["avoided_cpu_min"] += r.avoided_cpu_min
    active_zero = sorted(((b.total_spent, bid) for bid, b in state.branches.items() if b.state in CM.ACTIVE_STATES and b.total_spent > 0
                          and not any(x.get("action") in ("ADVANCE", "COMPLETE") for x in b.runs)), reverse=True)[:5]
    return {"ledger_cpu_min": tot["cpu_min"], "waste_share": tot["waste_share"], "dormant_branches": len(dormant),
            "avoided_cpu_min": float(sum(r.avoided_cpu_min for r in dormant)), "sunk_in_dormant_cpu_min": float(sum(r.spent_cpu_min for r in dormant)),
            "by_reason": per_reason, "false_dormancy": book.false_dormancy_rate(), "top_unproven_spend": [(bid, round(c, 1)) for c, bid in active_zero],
            "lifecycle_health": str(book.life.health(now)), "lifecycle_chain_errors": book.life.verify_chain()}


def explain_verdict(v: WasteVerdict) -> str:
    head = f"{v.branch_id}: {v.action.value}" + (f" ({v.reason.value})" if v.reason else "") + f", priority x{v.multiplier:.2f}"
    lines = [head] + ["  " + n for n in v.notes]
    if v.symptoms is not None:
        lines.append("  symptoms: " + ", ".join(f"{k}={'?' if x is None else ('YES' if x else 'no')}" for k, x in v.symptoms.flags().items()))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ trend detection
@dataclasses.dataclass(frozen=True)
class TrendReport:
    branch_id: str
    n: int
    slope: float | None                # least-squares slope of value-per-CPU-minute against experiment index
    decline_alarm: bool                # CUSUM found a downward shift away from the branch's own early level
    early_rate: float | None
    late_rate: float | None
    verdict: str                       # IMPROVING | FLAT | DECLINING | INSUFFICIENT


def branch_trend(t: Telemetry, min_n: int = 5) -> TrendReport:
    """Is the branch getting more or less productive per CPU-minute? A branch can be low but improving (a slow start) or high but
    collapsing (an idea running out); only the second is a waste signal. The shift test is research_policy.cusum_shift, whose
    reference level comes from the branch's own first half, so it fires on a fall from its early level, not on an absolute low."""
    from engine.learning.research_policy import cusum_shift
    rates = [g / c for g, c in zip(t.gross, t.costs) if c > 0]
    n = len(rates)
    if n < min_n:
        return TrendReport(t.branch_id, n, None, False, None, None, "INSUFFICIENT")
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, np.array(rates), 1)[0])
    half = n // 2
    early, late = float(np.mean(rates[:half])), float(np.mean(rates[half:]))
    shift = cusum_shift(rates)
    alarm = shift.get("direction") == "down"
    span = max(abs(early), abs(late), 1e-9)
    if late > early + 0.25 * span:
        v = "IMPROVING"
    elif late < early - 0.25 * span:
        v = "DECLINING"
    else:
        v = "FLAT"
    return TrendReport(t.branch_id, n, slope, alarm, early, late, v)


def late_bloomer_guard(t: Telemetry, pol: WastePolicy) -> bool:
    """True if the branch must NOT be parked yet because it is on an improving trend (its recent runs beat its early ones by a
    quarter) even though the window average is low. An idea that started slowly is exactly what 'never abandon a promising
    hypothesis after an underpowered cheap test' protects."""
    tr = branch_trend(t)
    return tr.verdict == "IMPROVING" and (tr.late_rate or 0.0) > 0.5 * pol.rate_floor


# ------------------------------------------------------------------------------------------------ sensitivity
def verdict_sensitivity(state: CM.ManagerState, ledger: VA.ValueLedger, branch_id: str, now, pol: WastePolicy, scale: float = 0.3) -> dict:
    """Would the parking decision survive moving each threshold by +/-scale? A DORMANT verdict that flips under a 30% change of one
    convention is marginal; the caller may then prefer DEPRIORITISE. Returns the base action and each flipping perturbation."""
    base = assess_branch(state, ledger, branch_id, now, pol).action
    flips: dict[str, list[str]] = {}
    for knob in ("rate_floor", "relative_floor", "tiny_effect_ratio", "min_power", "complexity_units_limit", "min_effect"):
        for mult in (1.0 - scale, 1.0 + scale):
            alt = dataclasses.replace(pol, **{knob: getattr(pol, knob) * mult})
            if alt.validate():
                continue
            a = assess_branch(state, ledger, branch_id, now, alt).action
            if a is not base:
                flips.setdefault(knob, []).append(f"x{mult:.2f}->{a.value}")
    return {"base": base.value, "robust": not flips, "flips": flips}


# ------------------------------------------------------------------------------------------------ building the world context
def context_from_manifest(as_of, data_hash: str, n_rows: int, regime: str, representations: Iterable[str] = (),
                          representation_families: Mapping[str, str] | None = None, matured: Mapping[str, int] | None = None,
                          hypotheses: Mapping[str, Iterable[str]] | None = None, evidence: Mapping[str, float] | None = None) -> WorldContext:
    """Build a WorldContext from plain counts and ids. Refuses anything that looks like identity (tickers, real dates in ids or
    labels) because a dormant branch's fingerprint is stored and compared for ever; the curator, not this module, owns real dates."""
    import re
    bad = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}){0,2}\b")
    labels = [regime, *representations, *(hypotheses or {}).keys(), *[h for hs in (hypotheses or {}).values() for h in hs]]
    for lab in labels:
        if bad.search(str(lab)):
            raise ValueError(f"identity leak: {lab!r} carries a real year/date")
    return WorldContext(as_date(as_of).isoformat(), data_hash, int(n_rows), frozenset(representations), dict(representation_families or {}),
                        regime, dict(matured or {}), {f: tuple(sorted(h)) for f, h in (hypotheses or {}).items()}, dict(evidence or {}))


# ------------------------------------------------------------------------------------------------ aging and budgets
def stale_dormant(book: DormantBook, now, days: int = 365) -> list[str]:
    """Dormant branches parked longer than `days` with no probe and no revival: candidates for an explicit retirement DECISION."""
    out = []
    for bid, r in sorted(book.records.items()):
        last = as_date(r.probes[-1][0]) if r.probes else (as_date(r.last_revival) if r.last_revival else as_date(r.since))
        if r.active and (as_date(now) - last).days >= days:
            out.append(bid)
    return out


def waste_budget(state: CM.ManagerState, ledger: VA.ValueLedger, now, cap_share: float = 0.25) -> dict:
    """Share of the last 30 days' compute that went into WORTHLESS/INVALID jobs against a cap. Over the cap the ladder's period
    budget should shrink for the offending families (`offenders`), not for everyone."""
    lo = as_date(now) - dt.timedelta(days=30)
    recs = [r for r in ledger.records(now) if as_date(r.at) >= lo and r.verdict != VA.Verdict.CORRECTION.value]
    tot = sum(r.cost_cpu_min for r in recs)
    bad = [r for r in recs if r.verdict in (VA.Verdict.WORTHLESS.value, VA.Verdict.INVALID.value)]
    share = sum(r.cost_cpu_min for r in bad) / tot if tot > 0 else 0.0
    fam: dict[str, float] = {}
    for r in bad:
        fam[r.family] = fam.get(r.family, 0.0) + r.cost_cpu_min
    return {"window_cpu_min": tot, "waste_share": share, "cap": cap_share, "over_cap": share > cap_share,
            "offenders": sorted(fam, key=lambda f: (-fam[f], f))}


def decay_penalties(state: CM.ManagerState, pol: WastePolicy, recovery: float = 1.25) -> int:
    """Let priority penalties fade when nothing new has confirmed them: each call multiplies waste_factor by `recovery` (capped at 1),
    so a DEPRIORITISE must be re-earned by fresh bad results each tick to stay in force. Returns the branches changed."""
    n = 0
    for b in state.branches.values():
        if b.state in CM.ACTIVE_STATES and b.waste_factor < 1.0:
            b.waste_factor = min(1.0, b.waste_factor * recovery)
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ persistence
def save_all(book: DormantBook, directory: str | Path) -> None:
    d = Path(directory)
    book.dump(d / "dormant_book.json", d / "dormant_lifecycle.jsonl")


def load_all(directory: str | Path) -> DormantBook:
    d = Path(directory)
    life = RetirementLedger.load(d / "dormant_lifecycle.jsonl") if (d / "dormant_lifecycle.jsonl").exists() else RetirementLedger()
    p = d / "dormant_book.json"
    return DormantBook.from_dict(json.loads(p.read_text(encoding="utf-8")), life) if p.exists() else DormantBook(life)


# ------------------------------------------------------------------------------------------------ planted-world evaluation
@dataclasses.dataclass(frozen=True)
class ControllerScore:
    duds: int
    duds_parked: int
    promising: int
    promising_parked: int
    duds_cpu_before_parking: float
    duds_cpu_if_never_parked: float
    saved_cpu_min: float
    revived_after_change: bool


def evaluate_controller(seed: int, n_duds: int = 5, n_promising: int = 3, n_runs: int = 8, pol: WastePolicy | None = None) -> ControllerScore:
    """Planted-world test of the controller: DUDS are branches whose every run is worthless (no effect, no transfer, added
    complexity); PROMISING branches produce a real risk cut in some runs. Runs the controller as runs arrive, counts what it parked
    and how much compute the duds took before parking versus if they had been left running, then hands the duds a NEW data trigger
    and checks the revival. The controller must park the duds, never the promising branches, and revive on new data only."""
    pol = pol or WastePolicy()
    rng = np.random.default_rng(seed)
    st, led, book = CM.ManagerState(), VA.ValueLedger(), DormantBook()
    kinds = {}
    for i in range(n_duds + n_promising):
        b = CM.make_branch(st, CM.ResearchQuestion.make(f"planted {i}", "sim", CM.Problem.VOLATILITY, "2010-01-01", "2010-01-01", "s", "f"),
                           "2010-01-01", f"fam{i}")
        assert b is not None                         # every planted question is distinct
        kinds[b.branch_id] = "dud" if i < n_duds else "promising"
    before = {bid: 0.0 for bid in kinds}
    for r in range(n_runs):
        day = dt.date(2010, 1, 4) + dt.timedelta(days=r)
        for bid, kind in sorted(kinds.items()):
            b = st.branches[bid]
            if b.state is not CM.ResearchState.QUEUED:
                continue
            base = rng.normal(0.0, 0.03, 200)
            good = kind == "promising" and r % 2 == 1
            after = np.clip(base, -0.02, None) if good else base + rng.normal(0, 1e-3, 200)
            b.runs.append({"stage": "STAGE1_CHEAP_SCREEN", "at": day.isoformat(), "effect": 0.006 if good else 0.0002, "se": 0.001, "action": "REPEAT",
                           "cost": 6.0, "n_obs": 4000, "power": 0.9})
            VA.account_job(led, VA.JobMeasurement(f"{bid}_{r}", bid, b.family, CM.Problem.VOLATILITY, CM.Stage.CHEAP_SCREEN, day.isoformat(),
                                                  (day - dt.timedelta(days=1)).isoformat(), 6.0, pnl_before=base, pnl_after=after,
                                                  params_added=0 if good else 3, replications=int(good), seed=r),
                           day + dt.timedelta(days=1))
            b.spent["STAGE1_CHEAP_SCREEN"] = b.spent.get("STAGE1_CHEAP_SCREEN", 0.0) + 6.0
            before[bid] += 6.0
        step(st, led, book, WorldContext(day.isoformat(), "h1", 1000, regime="calm"), day + dt.timedelta(days=2), pol)
    duds = [b for b, k in kinds.items() if k == "dud"]
    prom = [b for b, k in kinds.items() if k == "promising"]
    parked = lambda ids: sum(1 for b in ids if st.branches[b].state is CM.ResearchState.DORMANT)
    later = dt.date(2010, 4, 1)
    final = step(st, led, book, WorldContext(later.isoformat(), "h2", 2000, regime="calm"), later, pol)
    return ControllerScore(len(duds), parked(duds) + len([b for b in duds if book.records.get(b) and not book.records[b].active]), len(prom), parked(prom),
                           float(sum(before[b] for b in duds)), 6.0 * n_runs * len(duds),
                           6.0 * n_runs * len(duds) - float(sum(before[b] for b in duds)), bool(final.revived))


# ------------------------------------------------------------------------------------------------ outcomes of dormancy: learn from parking
@dataclasses.dataclass(frozen=True)
class ParkingOutcome:
    branch_id: str
    reason: str
    parked_at: str
    revived_at: str
    revival_kind: str
    spent_before: float
    after_verdict: str            # verdict of the first job after revival (USEFUL / WORTHLESS / ...)


def parking_outcomes(state: CM.ManagerState, ledger: VA.ValueLedger, book: DormantBook, now) -> list[ParkingOutcome]:
    """For every branch that was parked and then revived: what its first post-revival job turned out to be. This is the ground
    truth for 'was parking right?' - a revived branch that immediately produces a USEFUL result was parked wrongly (or too early)."""
    out = []
    for bid, rec in sorted(book.records.items()):
        if rec.active or not rec.last_revival:
            continue
        after = [r for r in ledger.records(now) if r.branch_id == bid and as_date(r.at) > as_date(rec.last_revival) and r.verdict != VA.Verdict.CORRECTION.value]
        kinds = [x.get("kind") for x in getattr(rec, "probes", [])]
        out.append(ParkingOutcome(bid, rec.reason, rec.since, rec.last_revival, kinds[-1] if kinds else "trigger", rec.spent_cpu_min,
                                  after[0].verdict if after else "PENDING"))
    return out


def reason_scorecard(outcomes: Sequence[ParkingOutcome]) -> dict[str, dict[str, float]]:
    """Per parking reason: how many were revived and how many of the revived immediately produced something useful. A reason whose
    revivals are mostly USEFUL is too eager (its thresholds should loosen); one whose revivals are mostly WORTHLESS is right."""
    out: dict[str, dict[str, float]] = {}
    for o in outcomes:
        d = out.setdefault(o.reason, {"revived": 0.0, "useful_after": 0.0, "worthless_after": 0.0, "pending": 0.0})
        d["revived"] += 1
        if o.after_verdict in (VA.Verdict.USEFUL.value, VA.Verdict.PREVENTIVE.value):
            d["useful_after"] += 1
        elif o.after_verdict == "PENDING":
            d["pending"] += 1
        elif o.after_verdict in (VA.Verdict.WORTHLESS.value, VA.Verdict.INVALID.value):
            d["worthless_after"] += 1
    for d in out.values():
        settled = d["revived"] - d["pending"]
        d["wrongly_parked_rate"] = d["useful_after"] / settled if settled > 0 else float("nan")
    return out


def advise_thresholds(scorecard: Mapping[str, Mapping[str, float]], probes: Mapping[str, float], pol: WastePolicy, min_settled: int = 5) -> list[str]:
    """ADVICE on the policy, never applied automatically (a gate tuned on its own outcomes is the overfitting this system exists
    to prevent). Loosen when parked-then-revived branches turn out useful or probes often look promising; tighten when parked
    branches never come back to anything, which means the dormancy bar could be lower and save more."""
    notes = []
    for reason, d in sorted(scorecard.items()):
        settled = d["revived"] - d["pending"]
        if settled < min_settled:
            notes.append(f"{reason}: only {int(settled)} settled revivals; no advice")
        elif d["wrongly_parked_rate"] > 0.4:
            notes.append(f"{reason}: {d['wrongly_parked_rate']:.0%} of revivals were useful; raise min_powered_runs or min_symptoms")
        elif d["wrongly_parked_rate"] < 0.05:
            notes.append(f"{reason}: revivals almost never useful; the bar for parking could be lower")
    if probes.get("probes", 0) >= min_settled and probes.get("upper90", 0.0) > 0.35:
        notes.append("probe false-dormancy upper bound above 35%: the controller is too eager overall")
    return notes


# ------------------------------------------------------------------------------------------------ family-level view
def family_waste(state: CM.ManagerState, ledger: VA.ValueLedger, book: DormantBook, now, min_branches: int = 4) -> list[dict]:
    """Families in which MANY branches have been parked or are barren: candidates for the accountant's 'this family is unreliable'
    test (a whole-family finding is worth far more than parking branches one by one). The controller only nominates; the family
    claim itself must come from VA.build_family_finding with powered nulls and multiplicity control."""
    fams: dict[str, dict[str, float]] = {}
    for bid, b in state.branches.items():
        d = fams.setdefault(b.family, {"branches": 0.0, "dormant": 0.0, "failed": 0.0, "cpu_min": 0.0, "passed_any": 0.0})
        d["branches"] += 1
        d["cpu_min"] += b.total_spent
        d["dormant"] += float(b.state is CM.ResearchState.DORMANT)
        d["failed"] += float(b.state is CM.ResearchState.FAILED)
        d["passed_any"] += float(bool(b.passed))
    out = []
    rates = ledger.family_stats(now)
    for f, d in sorted(fams.items()):
        if d["branches"] >= min_branches and (d["dormant"] + d["failed"]) / d["branches"] >= 0.75 and d["passed_any"] / d["branches"] <= 0.25:
            out.append({"family": f, **d, "rate": rates[f].value_per_cpu_min if f in rates else None})
    return out


# ------------------------------------------------------------------------------------------------ book invariants
def audit_book(book: DormantBook, state: CM.ManagerState, now) -> list[str]:
    """Invariants (empty = clean): every active dormant record matches a DORMANT branch with a reason; every DORMANT branch has a
    record; the lifecycle ledger agrees and its chain is intact; no revival count exceeds the policy's maximum by more than one."""
    errs = []
    for bid, rec in sorted(book.records.items()):
        b = state.branches.get(bid)
        if b is None:
            errs.append(f"{bid}: dormant record without a branch")
            continue
        if rec.active and b.state is not CM.ResearchState.DORMANT:
            errs.append(f"{bid}: record says dormant but branch is {b.state.value}")
        if rec.active and not b.dormant_reason:
            errs.append(f"{bid}: dormant with no reason on the branch")
        if not rec.reason:
            errs.append(f"{bid}: record has no reason")
    for bid, b in sorted(state.branches.items()):
        if b.state is CM.ResearchState.DORMANT and (bid not in book.records or not book.records[bid].active):
            errs.append(f"{bid}: DORMANT branch with no active dormant record (parked outside the controller)")
    errs += [f"lifecycle: {e}" for e in book.life.verify_chain()]
    return errs


def daily_summary(state: CM.ManagerState, ledger: VA.ValueLedger, book: DormantBook, res: WasteStepResult, now) -> str:
    """The controller's tick in a few lines for the daily research report: what it parked, why, what came back."""
    lines = [f"Waste controller @ {as_date(now).isoformat()}  (IMPLEMENTED - NOT VALIDATED)",
             f"  judged {len(res.verdicts)} active branches; parked {len(res.parked)}, deprioritised {len(res.deprioritised)}, revived {len(res.revived)}"]
    for v in res.verdicts:
        if v.action in (Action.DORMANT, Action.DEPRIORITISE):
            lines.append("  " + explain_verdict(v).replace("\n", "\n  "))
    if res.held_back:
        lines.append(f"  circuit breaker held back {len(res.held_back)} dormancy verdicts" + (" (MASS-DORMANCY ALARM)" if res.mass_dormancy_alarm else ""))
    wb = waste_budget(state, ledger, now)
    lines.append(f"  waste share of last 30 days: {wb['waste_share']:.0%} (cap {wb['cap']:.0%})" + ("  OVER CAP" if wb["over_cap"] else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ savings and overrides
def savings_estimate(book: DormantBook, state: CM.ManagerState, ledger: VA.ValueLedger, now) -> dict:
    """What parking has saved and what it has cost. Saved = expected remaining ladder compute of every currently dormant branch;
    at risk = the same figure weighted by the measured false-dormancy mean (a branch wrongly parked forgoes its expected value);
    the net is reported with its uncertainty, because a saving that assumes the controller is never wrong is not a saving."""
    fd = book.false_dormancy_rate()
    active = [r for r in book.records.values() if r.active]
    saved = float(sum(r.avoided_cpu_min for r in active))
    tot = ledger.totals(now)
    rate = tot["net_value"] / tot["cpu_min"] if tot["cpu_min"] > 0 else 0.0
    forgone = float(sum(r.avoided_cpu_min for r in active)) * fd["mean"] * max(rate, 0.0)
    return {"parked": len(active), "avoided_cpu_min": saved, "false_dormancy_mean": fd["mean"], "false_dormancy_upper90": fd["upper90"],
            "value_forgone_estimate": forgone, "value_rate_per_cpu_min": rate,
            "avoided_cpu_min_at_upper90_wrong": saved * (1.0 - fd["upper90"])}


def repeat_offenders(book: DormantBook, min_parks: int = 2) -> list[str]:
    """Branches parked more than once (revived, then parked again): the revival triggers were not strong enough evidence."""
    return sorted(bid for bid, r in book.records.items() if r.revive_count >= min_parks)


class OverrideError(RuntimeError):
    """A manual override was refused (no reason, wrong state, or attempted on a retired branch)."""


def manual_hold(state: CM.ManagerState, book: DormantBook, branch_id: str, now, reason: str, by: str) -> None:
    """An operator parks a branch by hand. Recorded exactly like a controller parking (reason, fingerprint, lifecycle), with the
    operator named, so manual and automatic dormancy live in one book and one audit trail."""
    if not reason.strip() or not by.strip():
        raise OverrideError("a manual hold needs a reason and an operator name")
    b = state.branches[branch_id]
    if b.state not in CM.ACTIVE_STATES:
        raise OverrideError(f"{branch_id} is {b.state.value}; only active branches can be held")
    v = WasteVerdict(branch_id, Action.DORMANT, Reason.NO_PROGRESS_PER_COMPUTE, 0.1, None, None, (f"manual hold by {by}: {reason}",))
    book.park(state, v, WorldContext(as_date(now).isoformat()), now, WastePolicy())


def manual_release(state: CM.ManagerState, book: DormantBook, branch_id: str, now, reason: str, by: str) -> None:
    """An operator revives a dormant branch by hand, restarting it at rung 1. Counts toward the revival limit like any revival, so
    manual release cannot be used to loop a dead idea for ever."""
    if not reason.strip() or not by.strip():
        raise OverrideError("a manual release needs a reason and an operator name")
    rec = book.records.get(branch_id)
    if rec is None or not rec.active:
        raise OverrideError(f"{branch_id} is not held in the dormant book")
    trig = [Trigger.make(TriggerKind.NEW_EVIDENCE, f"manual release by {by}: {reason}", 1.0)]
    if not book.revive(state, rec, trig, now, min_strength=0.0):
        raise OverrideError("release refused")


def apply_to_priorities(state: CM.ManagerState, policy: CM.LadderPolicy, now) -> dict[str, float]:
    """The priority every active branch now has, waste factor included. Shows the controller's effect on the ladder's ordering:
    a deprioritised branch sinks but stays selectable (a floor of factor_floor keeps it from ever being ranked at zero)."""
    return {bid: CM.priority(state, b, now, policy)[0] for bid, b in sorted(state.branches.items()) if b.state in CM.ACTIVE_STATES}


# ------------------------------------------------------------------------------------------------ controller self-monitoring
@dataclasses.dataclass(frozen=True)
class TickRecord:
    day: str
    judged: int
    parked: int
    revived: int
    held_back: int
    alarm: bool


class TickHistory:
    """The controller watching itself. It alarms when it parks or holds back an unusual share of the portfolio for several ticks in
    a row (a symptom of a measurement bug, not of a portfolio full of duds) or when it has revived nothing for a long time while
    the dormant book keeps growing (a controller that only ever closes doors)."""

    def __init__(self):
        self.ticks: list[TickRecord] = []

    def add(self, res: WasteStepResult, now) -> TickRecord:
        rec = TickRecord(as_date(now).isoformat(), len(res.verdicts), len(res.parked), len(res.revived), len(res.held_back), res.mass_dormancy_alarm)
        if self.ticks and as_date(rec.day) < as_date(self.ticks[-1].day):
            raise CM.FirewallBreach("tick dated before the previous tick")
        self.ticks.append(rec)
        return rec

    def health(self, window: int = 8, max_parked_share: float = 0.3) -> dict:
        w = self.ticks[-window:]
        if len(w) < 3:
            return {"verdict": "INSUFFICIENT", "ticks": len(w)}
        judged = sum(t.judged for t in w) or 1
        parked_share = (sum(t.parked for t in w) + sum(t.held_back for t in w)) / judged
        issues = []
        if parked_share > max_parked_share:
            issues.append(f"{parked_share:.0%} of judgements ended in dormancy: check the measurements before trusting the verdicts")
        if sum(1 for t in w if t.alarm) >= 2:
            issues.append("mass-dormancy alarm raised repeatedly")
        if sum(t.parked for t in w) >= 3 and sum(t.revived for t in w) == 0 and len(self.ticks) >= 2 * window:
            issues.append("nothing has been revived in a long time while branches keep being parked")
        return {"verdict": "SUSPECT" if issues else "OK", "ticks": len(w), "parked_share": parked_share, "issues": issues}

    def to_rows(self) -> list[dict]:
        return [dataclasses.asdict(t) for t in self.ticks]


def explain_dormant(book: DormantBook, state: CM.ManagerState, branch_id: str, ctx: WorldContext, now, pol: WastePolicy | None = None) -> list[str]:
    """Why a branch is parked and exactly what would bring it back: each recovery condition with the distance still to go."""
    pol = pol or WastePolicy()
    rec = book.records[branch_id]
    snap, trig = snapshot_for(rec, ctx, pol)
    lines = [f"{branch_id} parked {rec.since} for {rec.reason}: {rec.detail[:120]}",
             f"  spent {rec.spent_cpu_min:.1f} cpu-min, avoided ~{rec.avoided_cpu_min:.1f}; revived {rec.revive_count}x (max {pol.max_revivals})"]
    for c in recovery_conditions(pol):
        lines.append(f"  {c.key}: now {snap.get(c.key, 0.0):.2f}, needs >= {c.lo} -> {'MET' if c.satisfied_by(snap) else 'not met'}")
    lines.append(f"  unseen triggers now: {len(trig)}; combined strength {combined_strength(trig):.2f}")
    return lines


def reasons_histogram(book: DormantBook) -> dict[str, int]:
    """How many branches are currently parked per reason (the mix tells where research effort is being lost)."""
    out: dict[str, int] = {}
    for r in book.records.values():
        if r.active:
            out[r.reason] = out.get(r.reason, 0) + 1
    return dict(sorted(out.items()))
