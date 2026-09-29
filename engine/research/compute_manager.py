"""Research compute manager (C66 section 18; Bible research-brain phases; canon C63/C66). IMPLEMENTED - NOT VALIDATED.

Compute is finite, so every research branch (one ResearchQuestion pursued) walks a five-stage escalation ladder and its
ResearchState (QUEUED, EXPLORING, PROMISING, REPLICATING, ESCALATED, VALIDATING, FAILED, DORMANT, RETIRED) is a function of how far
its evidence has earned it:

  Stage 1 cheap screen (many small tests) -> Stage 2 stronger tests -> Stage 3 cross-year -> Stage 4 fresh holdout -> Stage 5 integration.

Two failure modes are guarded in code, not in prose:
  * big compute on a weak hypothesis: `authorise` refuses to fund stage k unless stage k-1 was PASSED on recorded evidence, and
    a result that is implausibly good (t above `suspicion_t`) is sent to an AUDIT, never up the ladder;
  * abandoning a promising hypothesis after one underpowered cheap test: a null only counts when the test had the power to see the
    smallest economically meaningful effect AND the confidence interval excludes it; otherwise the branch REPEATs with a larger
    sample (or is parked as INFEASIBLE, which the waste controller later treats as DORMANT-with-reason, never as failure).

It builds on, and never copies: engine.learning.compute (ExperimentSpec/ExperimentLedger/admit for RAM admission, deterministic
seeds, duplicate prevention), engine.learning.research_policy (ComputeBudget, ComputeCost, CostModel, power_of, required_n) and
engine.resources (worker limits). Public entry for the wave-2 loop: `step(state, now, ...)`.
State lives in MATURED_RESEARCH_STATE: nothing here may reach the blind trader (section 29-31)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from engine import resources as R
from engine.learning import compute as C
from engine.learning.research_policy import ComputeBudget, ComputeCost, CostModel, power_of, required_n
from engine.research.core import (ExperimentValue, FirewallBreach, Namespace, OBJECTIVE_ORDER, Problem, ResearchQuestion,
                                  ResearchState, Stage, as_date, require_past, stable_hash)

NAMESPACE = Namespace.MATURED_RESEARCH
LADDER: tuple[Stage, ...] = tuple(Stage)
ACTIVE_STATES = frozenset({ResearchState.QUEUED, ResearchState.EXPLORING, ResearchState.PROMISING, ResearchState.REPLICATING,
                           ResearchState.ESCALATED, ResearchState.VALIDATING})
TERMINAL_STATES = frozenset({ResearchState.RETIRED, ResearchState.CANCELLED})
_NORM = NormalDist()

# who may move to whom. Progress moves one rung at a time; setbacks step back one rung; FAILED/DORMANT are reachable from any
# active state; DORMANT wakes only into QUEUED (revival is a re-queue, it never resumes mid-ladder without re-earning it).
ALLOWED: dict[ResearchState, frozenset[ResearchState]] = {
    ResearchState.QUEUED: frozenset({ResearchState.EXPLORING, ResearchState.DORMANT, ResearchState.CANCELLED}),
    ResearchState.EXPLORING: frozenset({ResearchState.PROMISING, ResearchState.FAILED, ResearchState.DORMANT}),
    ResearchState.PROMISING: frozenset({ResearchState.REPLICATING, ResearchState.EXPLORING, ResearchState.FAILED, ResearchState.DORMANT}),
    ResearchState.REPLICATING: frozenset({ResearchState.ESCALATED, ResearchState.PROMISING, ResearchState.FAILED, ResearchState.DORMANT}),
    ResearchState.ESCALATED: frozenset({ResearchState.VALIDATING, ResearchState.REPLICATING, ResearchState.FAILED, ResearchState.DORMANT}),
    ResearchState.VALIDATING: frozenset({ResearchState.RETIRED, ResearchState.ESCALATED, ResearchState.FAILED, ResearchState.DORMANT}),
    ResearchState.FAILED: frozenset({ResearchState.DORMANT, ResearchState.RETIRED}),
    ResearchState.DORMANT: frozenset({ResearchState.QUEUED, ResearchState.RETIRED, ResearchState.CANCELLED}),
    ResearchState.RETIRED: frozenset({ResearchState.DORMANT}),
    ResearchState.CANCELLED: frozenset(),
}
# state a branch is in while the job of a given stage runs, and the state it earns by PASSING that stage
LAUNCH_STATE = {Stage.CHEAP_SCREEN: ResearchState.EXPLORING, Stage.STRONGER_TESTS: ResearchState.REPLICATING,
                Stage.CROSS_YEAR: ResearchState.ESCALATED, Stage.FRESH_HOLDOUT: ResearchState.VALIDATING,
                Stage.INTEGRATION: ResearchState.VALIDATING}
PASS_STATE = {Stage.CHEAP_SCREEN: ResearchState.PROMISING, Stage.STRONGER_TESTS: ResearchState.ESCALATED,
              Stage.CROSS_YEAR: ResearchState.ESCALATED, Stage.FRESH_HOLDOUT: ResearchState.VALIDATING,
              Stage.INTEGRATION: ResearchState.RETIRED}


class LadderError(RuntimeError):
    """An illegal ladder move: skipping a stage, funding without evidence, or an illegal state change."""


class Act(str, enum.Enum):
    """What one stage result means for the branch."""
    ADVANCE = "ADVANCE"          # earned the next rung
    COMPLETE = "COMPLETE"        # passed the last rung: hand to the quality gate (section 42), research on it is finished
    REPEAT = "REPEAT"            # underpowered or ambiguous: run this rung again, bigger
    DEMOTE = "DEMOTE"            # later-rung evidence shrank without turning null: step back one rung and re-earn it
    FAIL = "FAIL"                # a test with the power to see the effect saw none
    PARK = "PARK"                # cannot be resolved with the compute allowed: hand to the waste controller as DORMANT
    AUDIT = "AUDIT"              # too good to be true: check for leakage before anything else is spent


def stage_index(stage: Stage | str) -> int:
    return LADDER.index(Stage(stage))


def multiplicity_threshold(base_t: float, n_looks: int) -> float:
    """t needed to pass when the family has looked n_looks times (two-sided Bonferroni normal quantile at the base test's
    alpha). One look keeps base_t; a hundred cheap looks raise it, so screening 100 ideas does not manufacture a winner."""
    if base_t < 0 or n_looks < 1:
        raise ValueError("base_t >= 0 and n_looks >= 1 required")
    alpha_base = 2.0 * (1.0 - _NORM.cdf(base_t))
    alpha = min(alpha_base, 1.0) / n_looks
    return _NORM.inv_cdf(1.0 - alpha / 2.0) if alpha > 0 else math.inf


# ------------------------------------------------------------------------------------------------ policy
@dataclasses.dataclass(frozen=True)
class StageRule:
    """What one rung costs and what evidence lets a branch leave it."""
    stage: Stage
    planned_tests: int             # looks that make up one run of this rung
    cpu_min_per_test: float
    max_cpu_min: float             # hard cap for one run of the rung (a repeat that would exceed it is PARKed instead)
    ram_gb: float
    real_data: bool                # needs the (single) real-data slot
    min_power: float               # power against `min_effect` before a null result may count as a negative
    pass_t: float                  # t to pass BEFORE the multiplicity correction
    min_consistency: float         # share of tests / contexts / years whose sign agrees with the pooled effect
    min_units: int                 # contexts (rung 2), years (rung 3); 1 elsewhere
    needs_fresh: bool = False      # rung 4: data the branch never touched
    needs_replication: bool = False
    max_repeats: int = 2

    def validate(self) -> list[str]:
        errs = []
        if self.planned_tests < 1 or self.cpu_min_per_test <= 0 or self.max_cpu_min < self.cpu_min_per_test:
            errs.append(f"{self.stage}: tests/cost must be positive and max_cpu_min >= one test")
        if not 0.5 <= self.min_power < 1.0:
            errs.append(f"{self.stage}: min_power outside [0.5,1)")
        if self.pass_t <= 0 or not 0.5 <= self.min_consistency <= 1.0 or self.min_units < 1 or self.max_repeats < 0:
            errs.append(f"{self.stage}: threshold out of range")
        if self.ram_gb <= 0:
            errs.append(f"{self.stage}: ram_gb must be positive")
        return errs

    @property
    def base_cpu_min(self) -> float:
        return self.planned_tests * self.cpu_min_per_test


def default_rules() -> tuple[StageRule, ...]:
    S = Stage
    return (StageRule(S.CHEAP_SCREEN, 100, 0.2, 40.0, 0.8, False, 0.60, 1.5, 0.55, 1),
            StageRule(S.STRONGER_TESTS, 10, 3.0, 90.0, 1.5, False, 0.80, 2.2, 0.70, 3),
            StageRule(S.CROSS_YEAR, 8, 12.0, 240.0, 2.5, True, 0.80, 2.0, 0.70, 3, needs_replication=True),
            StageRule(S.FRESH_HOLDOUT, 1, 60.0, 400.0, 3.0, True, 0.85, 2.0, 1.0, 1, needs_fresh=True, max_repeats=1),
            StageRule(S.INTEGRATION, 1, 90.0, 600.0, 3.5, True, 0.85, 1.65, 1.0, 1, max_repeats=1))


@dataclasses.dataclass(frozen=True)
class LadderPolicy:
    rules: tuple[StageRule, ...] = dataclasses.field(default_factory=default_rules)
    min_effect: float = 0.0025           # smallest effect (in the branch's own units) worth acting on
    alpha: float = 0.05
    growth: float = 2.0                  # sample multiplier per REPEAT
    suspicion_t: float = 6.0             # a cheap-rung t this high is a leak until proven otherwise
    shrink_floor: float = 0.5            # fresh-holdout effect must keep this share of the earlier pooled effect
    integration_floor: float = 0.0       # rung 5: integration delta must not be below this (in se units, see assess)
    explore_floor: float = 0.20          # share of a period's compute reserved for rung 1 (new ideas must keep entering)
    stage_share_cap: Mapping[str, float] = dataclasses.field(default_factory=lambda: {
        Stage.CHEAP_SCREEN.value: 1.0, Stage.STRONGER_TESTS.value: 0.5, Stage.CROSS_YEAR.value: 0.4,
        Stage.FRESH_HOLDOUT.value: 0.3, Stage.INTEGRATION.value: 0.25})
    age_boost: float = 0.05              # per log-day queued: nothing starves
    direction_gate_rung: int = 2         # section 0: direction branches may not go past this rung index until volatility has earned it
    volatility_rung_required: int = 2    # the volatility branch must have PASSED this rung index (0-based) somewhere first
    problem_weight: Mapping[str, float] = dataclasses.field(default_factory=lambda: {
        p.value: w for p, w in zip(OBJECTIVE_ORDER, (1.0, 0.9, 0.7, 0.6, 0.55, 0.5, 0.4))})

    def validate(self) -> list[str]:
        errs = [e for r in self.rules for e in r.validate()]
        if tuple(r.stage for r in self.rules) != LADDER:
            errs.append("rules must cover the five stages in ladder order")
        if self.min_effect <= 0 or not 0 < self.alpha < 0.5 or self.growth <= 1.0:
            errs.append("min_effect > 0, alpha in (0,0.5), growth > 1 required")
        if not 0.0 <= self.explore_floor < 1.0 or self.suspicion_t <= max((r.pass_t for r in self.rules), default=0):
            errs.append("explore_floor in [0,1) and suspicion_t above every pass_t required")
        if self.rules:
            caps = [r.max_cpu_min for r in self.rules]
            if caps != sorted(caps):
                errs.append("max_cpu_min must not shrink up the ladder (later rungs are the expensive ones)")
        return errs

    def rule(self, stage: Stage | str) -> StageRule:
        return self.rules[stage_index(stage)]


# ------------------------------------------------------------------------------------------------ evidence and assessment
@dataclasses.dataclass(frozen=True)
class StageEvidence:
    """What one run of a rung measured. effect is signed so that positive = the hypothesis worked; se its standard error."""
    stage: Stage
    n_obs: int
    effect: float
    se: float
    n_tests: int = 1
    n_positive: int = 0            # tests (rung 1) whose sign agreed with the hypothesis
    n_units: int = 1               # contexts (rung 2) or years (rung 3) examined
    units_positive: int = 0
    replications: int = 0          # independent replications that agreed
    fresh: bool = False            # the data were never used by this branch (rung 4)
    data_through: str = ""         # newest data date used; must be strictly before `now`
    cost_cpu_min: float = 0.0
    integration_delta: float | None = None     # rung 5: change of the live decision metric with the branch integrated
    integration_se: float | None = None
    complexity_added: float = 0.0

    def validate(self) -> list[str]:
        errs = []
        if self.n_obs < 0 or self.n_tests < 1 or self.n_units < 1:
            errs.append("counts must be non-negative (tests/units >= 1)")
        if not (math.isfinite(self.effect) and math.isfinite(self.se)) or self.se < 0:
            errs.append("effect/se must be finite and se >= 0")
        if self.n_positive > self.n_tests or self.units_positive > self.n_units:
            errs.append("positive counts exceed totals")
        if self.cost_cpu_min < 0 or self.complexity_added < 0:
            errs.append("cost/complexity negative")
        if self.integration_delta is not None and (self.integration_se is None or self.integration_se < 0):
            errs.append("integration_delta needs integration_se >= 0")
        return errs

    @property
    def t(self) -> float:
        if self.se > 0:
            return self.effect / self.se
        return 0.0 if self.n_obs < 2 or self.effect == 0 else math.copysign(math.inf, self.effect)

    @property
    def sd_obs(self) -> float:
        return self.se * math.sqrt(self.n_obs) if self.n_obs > 0 else 0.0


@dataclasses.dataclass(frozen=True)
class Assessment:
    stage: Stage
    passed: bool
    powered: bool
    null_established: bool
    suspicious: bool
    t: float
    t_required: float
    power: float
    needed_n: int | None
    consistency: float
    reasons: tuple[str, ...]


def _consistency(ev: StageEvidence) -> float:
    if ev.stage is Stage.CHEAP_SCREEN:
        return ev.n_positive / ev.n_tests
    if ev.stage in (Stage.STRONGER_TESTS, Stage.CROSS_YEAR):
        return ev.units_positive / ev.n_units
    return 1.0 if ev.effect > 0 else 0.0


def assess(ev: StageEvidence, policy: LadderPolicy, looks: int = 1, prior_effect: float | None = None,
           audit_cleared: bool = False) -> Assessment:
    """Decide what one run proves. `looks` = how many hypotheses the family has screened (multiplicity); `prior_effect` = the
    pooled effect earned at earlier rungs (rung 4 must keep `shrink_floor` of it). `audit_cleared` = a leakage audit already
    cleared this rung, so an extreme t is reproduced evidence rather than a suspect."""
    bad = ev.validate()
    if bad:
        raise ValueError("invalid StageEvidence: " + "; ".join(bad))
    rule = policy.rule(ev.stage)
    reasons: list[str] = []
    t, sd = ev.t, ev.sd_obs
    powered, power, needed = False, 0.0, None
    if ev.n_obs >= 2 and sd > 0:
        power = power_of(ev.n_obs, policy.min_effect, sd, policy.alpha)
        powered = power >= rule.min_power
        needed = required_n(policy.min_effect, sd, policy.alpha, rule.min_power)
    else:
        reasons.append("no variance estimate: the run cannot be judged")
    t_req = multiplicity_threshold(rule.pass_t, max(1, looks)) if ev.stage is Stage.CHEAP_SCREEN else rule.pass_t
    suspicious = ev.stage in (Stage.CHEAP_SCREEN, Stage.STRONGER_TESTS) and t > policy.suspicion_t and not audit_cleared
    if suspicious:
        reasons.append(f"t={t:.1f} exceeds suspicion_t={policy.suspicion_t}: audit for leakage before spending more")
    fails: list[str] = []
    if t < t_req:
        fails.append(f"t={t:.2f} below required {t_req:.2f}" + (f" (multiplicity over {looks} looks)" if looks > 1 else ""))
    cons = _consistency(ev)
    if cons < rule.min_consistency:
        fails.append(f"sign consistency {cons:.2f} below {rule.min_consistency:.2f}")
    if ev.stage in (Stage.STRONGER_TESTS, Stage.CROSS_YEAR) and ev.n_units < rule.min_units:
        fails.append(f"only {ev.n_units} {'contexts' if ev.stage is Stage.STRONGER_TESTS else 'years'} (< {rule.min_units})")
    if rule.needs_replication and ev.replications < 1:
        fails.append("no independent replication")
    if rule.needs_fresh:
        if not ev.fresh:
            fails.append("holdout data were not fresh")
        if prior_effect is not None and prior_effect > 0 and ev.effect < policy.shrink_floor * prior_effect:
            fails.append(f"effect {ev.effect:.4g} kept < {policy.shrink_floor:.0%} of the earlier {prior_effect:.4g}")
    if ev.stage is Stage.INTEGRATION:
        if ev.integration_delta is None:
            fails.append("no integration measurement")
        else:
            se_i = ev.integration_se or 0.0
            floor = policy.integration_floor - (1.645 * se_i)
            if ev.integration_delta < floor:
                fails.append(f"integration delta {ev.integration_delta:.4g} is below the floor {floor:.4g}")
    if not powered and ev.n_obs >= 2 and sd > 0:
        fails.append(f"power {power:.2f} below {rule.min_power:.2f} at the smallest useful effect")
    # a NEGATIVE is established only by a test that could have seen the effect and whose interval rules it out (or reversed it)
    upper = ev.effect + _NORM.inv_cdf(1.0 - policy.alpha / 2) * ev.se
    null = powered and (upper < policy.min_effect or t <= -2.0)
    passed = not fails and not suspicious
    return Assessment(ev.stage, passed, powered, null, suspicious, t, t_req, power, needed, cons, tuple(reasons + fails))


@dataclasses.dataclass(frozen=True)
class EscalationDecision:
    action: Act
    stage: Stage
    next_stage: Stage | None
    needed_n: int | None
    planned_cpu_min: float
    reasons: tuple[str, ...]


def repeat_cost(rule: StageRule, policy: LadderPolicy, repeats_done: int, needed_n: int | None = None, n_obs: int = 0) -> float:
    """CPU-minutes of the next repeat: the rung's base cost scaled by the sample growth the evidence asks for."""
    grow = policy.growth ** (repeats_done + 1)
    if needed_n and n_obs > 0:
        grow = max(grow, min(needed_n / n_obs, 16.0))
    return rule.base_cpu_min * grow


def decide(ev: StageEvidence, a: Assessment, policy: LadderPolicy, repeats_done: int = 0) -> EscalationDecision:
    """Turn an Assessment into a ladder action. The order matters: audit first, then a genuine pass, then a proven null, and
    only then the underpowered/ambiguous cases, which REPEAT (bigger) or PARK, and never FAIL."""
    rule = policy.rule(ev.stage)
    if a.suspicious:
        return EscalationDecision(Act.AUDIT, ev.stage, None, None, 0.0, a.reasons)
    if a.passed:
        nxt = LADDER[stage_index(ev.stage) + 1] if stage_index(ev.stage) + 1 < len(LADDER) else None
        return EscalationDecision(Act.ADVANCE if nxt else Act.COMPLETE, ev.stage, nxt, None,
                                  policy.rule(nxt).base_cpu_min if nxt else 0.0, ("passed " + ev.stage.value,))
    if a.null_established:
        return EscalationDecision(Act.FAIL, ev.stage, None, None, 0.0, ("a powered test excluded the smallest useful effect",) + a.reasons)
    if rule.needs_fresh and any("kept" in r for r in a.reasons) and a.powered and a.t > 0:
        prev = LADDER[stage_index(ev.stage) - 1]
        return EscalationDecision(Act.DEMOTE, ev.stage, prev, None, policy.rule(prev).base_cpu_min,
                                  ("fresh-data effect shrank; re-earn the earlier rung",) + a.reasons)
    cost = repeat_cost(rule, policy, repeats_done, a.needed_n, ev.n_obs)
    if repeats_done >= rule.max_repeats:
        return EscalationDecision(Act.PARK, ev.stage, None, a.needed_n, 0.0,
                                  (f"unresolved after {repeats_done} repeats: inconclusive, not failed",) + a.reasons)
    if cost > rule.max_cpu_min:
        return EscalationDecision(Act.PARK, ev.stage, None, a.needed_n, cost,
                                  (f"needs ~{cost:.0f} cpu-min > cap {rule.max_cpu_min:.0f}: INFEASIBLE at today's data/compute",) + a.reasons)
    return EscalationDecision(Act.REPEAT, ev.stage, ev.stage, a.needed_n, cost, ("underpowered or ambiguous: repeat larger",) + a.reasons)


# ------------------------------------------------------------------------------------------------ branch
@dataclasses.dataclass
class Branch:
    """One research branch. `frontier` is the rung its NEXT job would run at; `passed` maps rung -> the evidence that earned it."""
    branch_id: str
    question_id: str
    text: str
    problem: Problem
    family: str
    created: str
    state: ResearchState = ResearchState.QUEUED
    frontier: Stage = Stage.CHEAP_SCREEN
    passed: dict = dataclasses.field(default_factory=dict)
    repeats: dict = dataclasses.field(default_factory=dict)
    spent: dict = dataclasses.field(default_factory=dict)          # stage value -> cpu-minutes
    runs: list = dataclasses.field(default_factory=list)           # dicts: stage, at, effect, se, t, action, cost
    expected: ExperimentValue = ExperimentValue()
    in_flight: str = ""
    updated: str = ""
    last_progress: str = ""
    dormant_reason: str = ""
    audit_pending: bool = False
    audit_cleared: list = dataclasses.field(default_factory=list)      # rung values whose extreme result passed a leakage audit
    looks: int = 1
    n_obs_planned: int = 0
    waste_factor: float = 1.0                                          # (0,1]: set by the waste controller to deprioritise, never to delete
    parents: tuple[str, ...] = ()

    @property
    def total_spent(self) -> float:
        return float(sum(self.spent.values()))

    def highest_passed(self) -> int:
        return max((stage_index(s) for s in self.passed), default=-1)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["problem"], d["state"], d["frontier"] = str(self.problem.value), self.state.value, self.frontier.value
        d["parents"] = list(self.parents)
        return json.loads(json.dumps(d, default=lambda o: o.value if isinstance(o, enum.Enum) else str(o)))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Branch":
        kw = dict(d)
        kw["problem"], kw["state"], kw["frontier"] = Problem(kw["problem"]), ResearchState(kw["state"]), Stage(kw["frontier"])
        kw["expected"] = ExperimentValue(**kw.get("expected", {}))
        kw["parents"] = tuple(kw.get("parents", ()))
        return cls(**kw)


@dataclasses.dataclass(frozen=True)
class BranchTransition:
    """Immutable, hash-chained record of a state change. `at` is the decision date."""
    seq: int
    branch_id: str
    from_state: str
    to_state: str
    at: str
    reason: str
    prev_hash: str = ""
    id: str = ""

    def body(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("id")
        return d


class LadderStats:
    """Pass rate of every rung, learned from what actually happened (Laplace-smoothed). It feeds the priority function and the
    funnel report: if rung 1 passes 90% of ideas the screen is too lax; if rung 4 fails all of them, rung 2-3 are too lax."""

    def __init__(self):
        self.tries = {s.value: 0 for s in LADDER}
        self.passes = {s.value: 0 for s in LADDER}
        self.cost = {s.value: 0.0 for s in LADDER}

    def observe(self, stage: Stage, passed: bool, cost: float) -> None:
        self.tries[stage.value] += 1
        self.passes[stage.value] += int(passed)
        self.cost[stage.value] += cost

    def pass_rate(self, stage: Stage, prior: float = 0.3, strength: float = 4.0) -> float:
        n, k = self.tries[stage.value], self.passes[stage.value]
        return (k + prior * strength) / (n + strength)

    def mean_cost(self, stage: Stage) -> float | None:
        n = self.tries[stage.value]
        return self.cost[stage.value] / n if n else None

    def prob_reach_end(self, from_stage: Stage) -> float:
        p = 1.0
        for s in LADDER[stage_index(from_stage):]:
            p *= self.pass_rate(s)
        return p

    def to_dict(self) -> dict:
        return {"tries": dict(self.tries), "passes": dict(self.passes), "cost": dict(self.cost)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LadderStats":
        o = cls()
        o.tries.update(d.get("tries", {}))
        o.passes.update(d.get("passes", {}))
        o.cost.update(d.get("cost", {}))
        return o


@dataclasses.dataclass
class ManagerState:
    """Everything the manager knows; JSON-serialisable so a crash costs nothing (save/load are atomic)."""
    branches: dict = dataclasses.field(default_factory=dict)
    log: list = dataclasses.field(default_factory=list)
    stats: LadderStats = dataclasses.field(default_factory=LadderStats)
    cost_model: CostModel = dataclasses.field(default_factory=CostModel)
    period_start: str = ""
    period_spent: dict = dataclasses.field(default_factory=dict)      # stage value -> cpu-min in the current period
    questions_seen: set = dataclasses.field(default_factory=set)
    version: int = 1

    def to_dict(self) -> dict:
        cm = self.cost_model
        return {"version": self.version, "branches": {k: b.to_dict() for k, b in sorted(self.branches.items())},
                "log": [dataclasses.asdict(t) for t in self.log], "stats": self.stats.to_dict(),
                "cost_model": {"k": cm.k, "sum_log": cm.sum_log, "sum_sq": cm.sum_sq, "n": cm.n},
                "period_start": self.period_start, "period_spent": dict(self.period_spent),
                "questions_seen": sorted(self.questions_seen)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ManagerState":
        st = cls(version=int(d.get("version", 1)))
        st.branches = {k: Branch.from_dict(v) for k, v in d.get("branches", {}).items()}
        st.log = [BranchTransition(**t) for t in d.get("log", [])]
        st.stats = LadderStats.from_dict(d.get("stats", {}))
        c = d.get("cost_model", {})
        st.cost_model = CostModel(c.get("k", 3.0))
        st.cost_model.sum_log, st.cost_model.sum_sq, st.cost_model.n = dict(c.get("sum_log", {})), dict(c.get("sum_sq", {})), dict(c.get("n", {}))
        st.period_start, st.period_spent = d.get("period_start", ""), dict(d.get("period_spent", {}))
        st.questions_seen = set(d.get("questions_seen", []))
        return st


def save_state(state: ManagerState, path: str | Path) -> None:
    C.atomic_write_json(path, state.to_dict())


def load_state(path: str | Path) -> ManagerState:
    d = C.read_json(path, None)
    return ManagerState.from_dict(d) if d else ManagerState()


# ------------------------------------------------------------------------------------------------ transitions
def _record(state: ManagerState, b: Branch, to: ResearchState, now, reason: str) -> BranchTransition | None:
    if to == b.state:
        return None
    if to not in ALLOWED[b.state]:
        raise LadderError(f"{b.branch_id}: {b.state.value} -> {to.value} is not an allowed move")
    if state.log and as_date(now) < as_date(state.log[-1].at):
        raise FirewallBreach(f"transition dated {now} is before the previous record {state.log[-1].at}")
    prev = state.log[-1].id if state.log else ""
    t = BranchTransition(len(state.log), b.branch_id, b.state.value, to.value, as_date(now).isoformat(), reason, prev)
    t = dataclasses.replace(t, id=stable_hash(t.body(), 20))
    state.log.append(t)
    b.state, b.updated = to, t.at
    return t


def verify_log(state: ManagerState) -> list[str]:
    """Problems in the transition history (empty = intact): broken hash chain, edited records, unknown branches, illegal moves."""
    errs, prev, last_state = [], "", {}
    for i, t in enumerate(state.log):
        if t.seq != i:
            errs.append(f"record {i}: seq {t.seq}")
        if t.prev_hash != prev:
            errs.append(f"record {i}: prev_hash broken")
        if stable_hash(t.body(), 20) != t.id:
            errs.append(f"record {i}: content edited")
        if t.branch_id not in state.branches:
            errs.append(f"record {i}: unknown branch {t.branch_id}")
        cur = last_state.get(t.branch_id, ResearchState.QUEUED.value)
        if cur != t.from_state:
            errs.append(f"record {i}: {t.branch_id} was {cur}, not {t.from_state}")
        elif ResearchState(t.to_state) not in ALLOWED[ResearchState(t.from_state)]:
            errs.append(f"record {i}: illegal move {t.from_state}->{t.to_state}")
        last_state[t.branch_id] = t.to_state
        prev = t.id
    return errs


def make_branch(state: ManagerState, q: ResearchQuestion, now, family: str, looks: int = 1) -> Branch | None:
    """Register a branch for a question. The same question never creates two branches (returns None: it is already known)."""
    if q.question_id in state.questions_seen:
        return None
    if looks < 1:
        raise ValueError("looks >= 1")
    bid = "BR" + stable_hash([q.question_id, family], 12)
    if bid in state.branches:
        return None
    d = as_date(now).isoformat()
    b = Branch(bid, q.question_id, q.text, q.problem, family, d, expected=q.expected, updated=d, last_progress=d, looks=looks,
               parents=tuple(q.parents))
    state.branches[bid] = b
    state.questions_seen.add(q.question_id)
    return b


def evidence_entitlement(b: Branch, policy: LadderPolicy) -> float:
    """Total CPU-minutes the branch's EARNED evidence entitles it to have spent: the caps of every rung it has passed plus the
    rung it is trying, each with its allowed repeats. A branch stuck at rung 1 can therefore never accumulate rung-4 spend."""
    top = min(len(LADDER) - 1, b.highest_passed() + 1)
    return float(sum(policy.rules[i].max_cpu_min * (1 + policy.rules[i].max_repeats) for i in range(top + 1)))


def best_rung(state: ManagerState, problem: Problem) -> int:
    """Highest rung index any non-failed branch of the problem has PASSED (-1 = none). Failed and cancelled branches earned nothing."""
    return max((b.highest_passed() for b in state.branches.values()
                if b.problem is problem and b.state not in (ResearchState.FAILED, ResearchState.CANCELLED)), default=-1)


def authorise(state: ManagerState, b: Branch, stage: Stage, cpu_min: float, policy: LadderPolicy) -> tuple[bool, str]:
    """Gatekeeper for every job. Refuses (with the reason, never silently) unless the branch has earned this rung."""
    if b.state not in ACTIVE_STATES:
        return False, f"branch is {b.state.value}"
    if b.in_flight:
        return False, f"job {b.in_flight[:10]} already in flight"
    if b.audit_pending:
        return False, "held for a leakage audit"
    if stage != b.frontier:
        return False, f"frontier is {b.frontier.value}, not {stage.value}"
    idx = stage_index(stage)
    if idx > 0 and LADDER[idx - 1].value not in b.passed:
        return False, f"{LADDER[idx - 1].value} was never passed: no {stage.value} compute for an unproven hypothesis"
    if b.problem is Problem.DIRECTION and idx >= policy.direction_gate_rung and best_rung(state, Problem.VOLATILITY) < policy.volatility_rung_required:
        return False, "direction research past the early rungs waits for volatility to earn rung " + str(policy.volatility_rung_required + 1)
    rule = policy.rule(stage)
    if cpu_min > rule.max_cpu_min + 1e-9:
        return False, f"{cpu_min:.0f} cpu-min exceeds the {stage.value} cap {rule.max_cpu_min:.0f}"
    if b.total_spent + cpu_min > evidence_entitlement(b, policy) + 1e-9:
        return False, f"spend {b.total_spent + cpu_min:.0f} would exceed what its evidence entitles ({evidence_entitlement(b, policy):.0f})"
    return True, "ok"


def planned_cost(state: ManagerState, b: Branch, policy: LadderPolicy) -> float:
    """CPU-minutes of the branch's next job: the rung's base cost, grown for repeats, scaled by the learned overrun factor."""
    rule = policy.rule(b.frontier)
    done = b.repeats.get(b.frontier.value, 0)
    base = rule.base_cpu_min * (policy.growth ** done)
    return min(rule.max_cpu_min, base * state.cost_model.multiplier(b.frontier.value))


def priority(state: ManagerState, b: Branch, now, policy: LadderPolicy) -> tuple[float, dict]:
    """Value per compute-minute, discounted by the chance the branch survives the ladder. Missing estimates are NOT read as
    zero: they take a neutral prior and the shortfall is reported. Deterministic in its inputs."""
    ev = b.expected
    miss = ev.missing()
    val = np.mean([x for x in (ev.decision_value, ev.information_gain, ev.uncertainty_reduction, ev.transfer_potential,
                               ev.loss_reduction_value, ev.failure_reduction_value) if x is not None] or [0.3])
    pen = (1.0 - (ev.overfit_risk if ev.overfit_risk is not None else 0.3)) * (1.0 - 0.5 * (ev.redundancy or 0.0))
    p_end = state.stats.prob_reach_end(b.frontier)
    w_problem = policy.problem_weight.get(b.problem.value, 0.4)
    cost = max(planned_cost(state, b, policy), 1e-6)
    age = max(0.0, (as_date(now) - as_date(b.created)).days)
    momentum = 1.0 + 0.25 * (b.highest_passed() + 1)          # a branch that has earned rungs is worth finishing
    score = b.waste_factor * float(val) * pen * w_problem * (0.5 + p_end) * momentum / math.sqrt(cost) * (1.0 + policy.age_boost * math.log1p(age))
    return score, {"value": float(val), "penalty": pen, "problem_weight": w_problem, "p_reach_end": p_end,
                   "momentum": momentum, "cost": cost, "age_days": age, "missing_estimates": miss}


@dataclasses.dataclass(frozen=True)
class Allocation:
    branch_id: str
    stage: Stage
    cpu_min: float
    ram_gb: float
    real_data: bool
    score: float
    reason: str


@dataclasses.dataclass(frozen=True)
class Selection:
    allocations: tuple[Allocation, ...]
    deferred: tuple[tuple[str, str], ...]          # (branch_id, reason): every deferral is explained
    period_cpu_min: float
    used_cpu_min: float


def period_budget_used(state: ManagerState) -> float:
    return float(sum(state.period_spent.values()))


def roll_period(state: ManagerState, now, days: int = 7) -> bool:
    """Start a new spending period once `days` have passed. Returns True when it rolled."""
    if not state.period_start:
        state.period_start = as_date(now).isoformat()
        return True
    if (as_date(now) - as_date(state.period_start)).days >= days:
        state.period_start, state.period_spent = as_date(now).isoformat(), {}
        return True
    return False


def select(state: ManagerState, now, policy: LadderPolicy, budget: ComputeBudget, free_gb: float | None,
           period_cpu_min: float, real_data_running: int = 0) -> Selection:
    """Pick this round's jobs. Ranked by `priority`; the rung-1 explore floor is served FIRST so new ideas keep entering; each rung
    has a share cap of the period so the expensive rungs cannot eat everything; RAM admission is cumulative (five jobs cannot each
    see the same free GB) and fails closed when memory is unreadable; only one real-data job at a time unless the budget says more."""
    if period_cpu_min < 0:
        raise ValueError("period_cpu_min must be >= 0")
    ranked = []
    deferred: list[tuple[str, str]] = []
    for bid in sorted(state.branches):
        b = state.branches[bid]
        if b.state not in ACTIVE_STATES or b.in_flight:
            continue
        cost = planned_cost(state, b, policy)
        ok, why = authorise(state, b, b.frontier, cost, policy)
        if not ok:
            deferred.append((bid, why))
            continue
        score, _ = priority(state, b, now, policy)
        ranked.append((-score, bid, b, cost))
    ranked.sort(key=lambda t: (t[0], t[1]))
    left_cpu = min(budget.cpu_minutes, max(0.0, period_cpu_min - period_budget_used(state)))
    rd_slots = max(0, budget.real_data_slots - real_data_running)
    free = free_gb
    spent_stage = dict(state.period_spent)
    chosen: list[Allocation] = []

    def take(item, floor_pass: bool) -> bool:
        nonlocal left_cpu, rd_slots, free
        _, bid, b, cost = item
        rule = policy.rule(b.frontier)
        if cost > left_cpu + 1e-9:
            deferred.append((bid, f"needs {cost:.0f} cpu-min, {left_cpu:.0f} left this round"))
            return False
        cap = policy.stage_share_cap.get(b.frontier.value, 1.0) * period_cpu_min
        if spent_stage.get(b.frontier.value, 0.0) + cost > cap + 1e-9:
            deferred.append((bid, f"{b.frontier.value} would exceed its {policy.stage_share_cap.get(b.frontier.value, 1.0):.0%} share of the period"))
            return False
        ok, why = budget.admits(ComputeCost(cost, rule.ram_gb, rule.real_data))
        if not ok:
            deferred.append((bid, why))
            return False
        if rule.real_data and rd_slots < 1:
            deferred.append((bid, "the real-data slot is busy"))
            return False
        okm, whym = C.admit(rule.ram_gb, (lambda: free))
        if not okm:
            deferred.append((bid, whym))
            return False
        left_cpu -= cost
        spent_stage[b.frontier.value] = spent_stage.get(b.frontier.value, 0.0) + cost
        if rule.real_data:
            rd_slots -= 1
        free = (free or 0.0) - rule.ram_gb
        sc = -item[0]
        chosen.append(Allocation(bid, b.frontier, cost, rule.ram_gb, rule.real_data, sc,
                                 "explore floor" if floor_pass else "ranked by value per compute"))
        return True

    floor_cpu = policy.explore_floor * period_cpu_min
    used_floor = spent_stage.get(Stage.CHEAP_SCREEN.value, 0.0)
    taken: set[str] = set()
    for item in ranked:
        if item[2].frontier is Stage.CHEAP_SCREEN and used_floor < floor_cpu:
            if take(item, True):
                used_floor += item[3]
                taken.add(item[1])
    for item in ranked:
        if item[1] not in taken:
            take(item, False)
    used = sum(a.cpu_min for a in chosen)
    return Selection(tuple(chosen), tuple(sorted(set(deferred))), period_cpu_min, used)


def to_spec(state: ManagerState, alloc: Allocation, now, seed: int, snapshot_id: str = "", data_hash: str = "") -> C.ExperimentSpec:
    """The executable form of an allocation (engine.learning.compute.ExperimentSpec): identical inputs -> identical key, so the
    compute ledger's duplicate prevention covers ladder repeats too (a REPEAT differs in `repeat`, a rerun of one does not)."""
    b = state.branches[alloc.branch_id]
    params = {"branch": b.branch_id, "family": b.family, "stage": alloc.stage.value, "repeat": b.repeats.get(alloc.stage.value, 0),
              "cpu_min": round(alloc.cpu_min, 3), "question": b.question_id}
    return C.ExperimentSpec(f"{b.branch_id}_s{stage_index(alloc.stage) + 1}", params, int(seed), as_date(now).isoformat(), snapshot_id,
                            data_hash, est_gb=alloc.ram_gb, priority=stage_index(alloc.stage) + 1)


def _epoch(now) -> float:
    return dt.datetime.combine(as_date(now), dt.time(0, 0), tzinfo=dt.timezone.utc).timestamp()


def launch(state: ManagerState, alloc: Allocation, now, policy: LadderPolicy, seed: int = 0, ledger: "C.ExperimentLedger | None" = None,
           code_hash: str = "", snapshot_id: str = "", data_hash: str = "") -> C.ExperimentSpec:
    """Commit an allocation: re-checks authority (state may have changed since `select`), submits to the compute ledger when one is
    given (its DuplicateExperiment propagates: a duplicate is refused, not re-run), then moves the branch into the rung's state."""
    b = state.branches[alloc.branch_id]
    ok, why = authorise(state, b, alloc.stage, alloc.cpu_min, policy)
    if not ok:
        raise LadderError(f"{b.branch_id}: launch refused - {why}")
    spec = to_spec(state, alloc, now, seed, snapshot_id, data_hash)
    if ledger is not None:
        ledger.submit(spec, _epoch(now), code_hash)
    _record(state, b, LAUNCH_STATE[alloc.stage], now, f"launched {alloc.stage.value}")
    b.in_flight = spec.key
    return spec


def _park_state(state: ManagerState, b: Branch, now, reason: str) -> None:
    _record(state, b, ResearchState.DORMANT, now, reason)
    b.dormant_reason, b.in_flight = reason, ""


def park(state: ManagerState, branch_id: str, now, reason: str) -> BranchTransition | None:
    """Move a branch to DORMANT with a reason (never deletion). The waste controller adds the revival triggers."""
    if not reason.strip():
        raise ValueError("a parked branch must carry a reason")
    b = state.branches[branch_id]
    if b.state is ResearchState.DORMANT:
        return None
    _park_state(state, b, now, reason)
    return state.log[-1]


def revive(state: ManagerState, branch_id: str, now, reason: str, keep_evidence: bool = False) -> None:
    """DORMANT -> QUEUED. By default the branch must re-earn its rungs (the data/regime/representation changed, so old passes
    are stale); keep_evidence=True keeps passed rungs and resumes at the frontier (only for a NEW HYPOTHESIS on unchanged data)."""
    b = state.branches[branch_id]
    if b.state is not ResearchState.DORMANT:
        raise LadderError(f"{branch_id} is {b.state.value}, only DORMANT branches revive")
    _record(state, b, ResearchState.QUEUED, now, "revived: " + reason)
    b.dormant_reason = ""
    b.repeats = {}
    if not keep_evidence:
        b.passed, b.frontier = {}, Stage.CHEAP_SCREEN
    b.last_progress = as_date(now).isoformat()


def clear_audit(state: ManagerState, branch_id: str, now, leak_found: bool, note: str) -> None:
    """Resolve an AUDIT. A found leak FAILS the branch (its evidence was contaminated); a clean audit re-queues the same rung so
    the suspicious result is REPRODUCED rather than trusted."""
    b = state.branches[branch_id]
    if not b.audit_pending:
        raise LadderError(f"{branch_id} has no audit pending")
    b.audit_pending = False
    if leak_found:
        _record(state, b, ResearchState.FAILED, now, "audit found leakage: " + note)
    else:
        b.audit_cleared.append(b.frontier.value)
        b.runs.append({"stage": b.frontier.value, "at": as_date(now).isoformat(), "action": "AUDIT_CLEAN", "note": note})


def cancel(state: ManagerState, branch_id: str, now, reason: str) -> None:
    b = state.branches[branch_id]
    if b.state is ResearchState.QUEUED:
        _record(state, b, ResearchState.CANCELLED, now, reason)
    elif b.state is ResearchState.DORMANT:
        _record(state, b, ResearchState.CANCELLED, now, reason)
    else:
        raise LadderError("only QUEUED or DORMANT branches can be cancelled; running work is FAILED or PARKed by its evidence")


def record_result(state: ManagerState, branch_id: str, ev: StageEvidence, now, policy: LadderPolicy,
                  prior_effect: float | None = None) -> EscalationDecision:
    """Apply one finished job. Evidence must come from the branch's current rung, be complete, and use data strictly before `now`.
    Updates spend, the cost model, the pass-rate statistics and the branch state; returns the decision."""
    b = state.branches[branch_id]
    if ev.stage != b.frontier:
        raise LadderError(f"{branch_id}: evidence for {ev.stage.value} but the frontier is {b.frontier.value}")
    if b.state in TERMINAL_STATES or b.state in (ResearchState.FAILED, ResearchState.DORMANT):
        raise LadderError(f"{branch_id} is {b.state.value}; a result cannot be applied")
    if ev.data_through:
        require_past(ev.data_through, now, f"{branch_id} evidence data_through")
    if prior_effect is None and b.passed:
        prior_effect = max((float(p.get("effect", 0.0)) for p in b.passed.values()), default=None)
    a = assess(ev, policy, b.looks, prior_effect, ev.stage.value in b.audit_cleared)
    repeats_done = b.repeats.get(ev.stage.value, 0)
    dec = decide(ev, a, policy, repeats_done)
    key = ev.stage.value
    planned = planned_cost(state, b, policy)
    b.spent[key] = b.spent.get(key, 0.0) + ev.cost_cpu_min
    state.period_spent[key] = state.period_spent.get(key, 0.0) + ev.cost_cpu_min
    if ev.cost_cpu_min > 0 and planned > 0:
        state.cost_model.observe(key, planned, ev.cost_cpu_min)
    state.stats.observe(ev.stage, dec.action in (Act.ADVANCE, Act.COMPLETE), ev.cost_cpu_min)
    b.runs.append({"stage": key, "at": as_date(now).isoformat(), "effect": ev.effect, "se": ev.se, "t": None if not math.isfinite(a.t) else a.t,
                   "action": dec.action.value, "cost": ev.cost_cpu_min, "n_obs": ev.n_obs, "power": a.power,
                   "complexity": ev.complexity_added})
    b.in_flight = ""
    d = as_date(now).isoformat()
    if dec.action in (Act.ADVANCE, Act.COMPLETE):
        b.passed[key] = {"effect": ev.effect, "se": ev.se, "t": a.t if math.isfinite(a.t) else 99.0, "n_obs": ev.n_obs, "at": d}
        b.last_progress = d
        if dec.action is Act.ADVANCE:
            b.frontier = dec.next_stage
        _record(state, b, PASS_STATE[ev.stage], now, f"passed {key}: t={a.t:.2f}")
    elif dec.action is Act.REPEAT:
        b.repeats[key] = repeats_done + 1
        b.n_obs_planned = int(dec.needed_n or 0)
    elif dec.action is Act.DEMOTE:
        b.passed.pop(key, None)
        b.passed.pop(dec.next_stage.value, None)
        b.frontier = dec.next_stage
        b.repeats = {}
        _record(state, b, ResearchState.ESCALATED, now, dec.reasons[0])
    elif dec.action is Act.FAIL:
        _record(state, b, ResearchState.FAILED, now, dec.reasons[0])
    elif dec.action is Act.PARK:
        _park_state(state, b, now, "; ".join(dec.reasons[:2]))
    elif dec.action is Act.AUDIT:
        b.audit_pending = True
    return dec


def reconcile_ledger(state: ManagerState, ledger: "C.ExperimentLedger", now) -> list[str]:
    """Clear `in_flight` for jobs the compute ledger gave up on (crash/OOM cap reached), so a dead job cannot freeze a branch
    forever. The branch keeps its rung and is re-selected; a GAVE_UP job that repeats is PARKed by `record_stall`."""
    led = ledger.load()
    cleared = []
    for b in state.branches.values():
        if b.in_flight and led.get(b.in_flight, {}).get("state") in (C.GAVE_UP, C.SUPERSEDED):
            b.runs.append({"stage": b.frontier.value, "at": as_date(now).isoformat(), "action": "JOB_LOST", "job": b.in_flight})
            b.in_flight = ""
            cleared.append(b.branch_id)
    return cleared


def stalled_branches(state: ManagerState, now, max_lost: int = 2) -> list[str]:
    """Branches whose jobs were lost `max_lost` times in a row: an infrastructure problem, not a research verdict."""
    out = []
    for bid, b in sorted(state.branches.items()):
        tail = []
        for r in reversed(b.runs):
            if r.get("action") != "JOB_LOST":
                break
            tail.append(r)
        if len(tail) >= max_lost and b.state in ACTIVE_STATES:
            out.append(bid)
    return out


# ------------------------------------------------------------------------------------------------ diagnostics
def funnel(state: ManagerState) -> dict[str, dict]:
    """Where branches are, how much each rung has cost and what fraction passed: the shape of the ladder in one table."""
    out = {}
    for s in LADDER:
        here = [b for b in state.branches.values() if b.frontier is s and b.state in ACTIVE_STATES]
        out[s.value] = {"branches_at_frontier": len(here), "tries": state.stats.tries[s.value], "passes": state.stats.passes[s.value],
                        "pass_rate": state.stats.pass_rate(s), "cpu_min": state.stats.cost[s.value],
                        "mean_cost": state.stats.mean_cost(s)}
    return out


def funnel_warnings(state: ManagerState, min_tries: int = 8) -> list[str]:
    """Signs that the ladder is mis-set: a rung that passes nearly everything is not screening; a late rung that fails everything
    means the earlier rungs waste compute; compute concentrated at expensive rungs means weak ideas are being over-funded."""
    warn = []
    f = funnel(state)
    for s in LADDER:
        r = f[s.value]
        if r["tries"] >= min_tries and r["passes"] / r["tries"] > 0.9:
            warn.append(f"{s.value} passes {r['passes']}/{r['tries']}: it is not screening anything")
        if s.value != Stage.CHEAP_SCREEN.value and r["tries"] >= min_tries and r["passes"] == 0:
            warn.append(f"{s.value} has never passed in {r['tries']} tries: earlier rungs are advancing hypotheses that cannot survive")
    tot = sum(v["cpu_min"] for v in f.values())
    if tot > 0:
        late = sum(f[s.value]["cpu_min"] for s in LADDER[2:]) / tot
        if late > 0.85:
            warn.append(f"{late:.0%} of compute is at rungs 3-5: exploration has stopped")
    return warn


def audit_ladder(state: ManagerState, policy: LadderPolicy) -> list[str]:
    """Invariants that must always hold (empty = clean): the log is intact, no rung ran without the previous one passed, no branch
    spent beyond its evidence, no in-flight job on an inactive branch, dormant branches carry a reason."""
    errs = verify_log(state)
    for bid, b in sorted(state.branches.items()):
        ran = {r["stage"] for r in b.runs if "effect" in r}
        for s in ran:
            i = stage_index(s)
            if i > 0 and LADDER[i - 1].value not in ran:
                errs.append(f"{bid}: ran {s} without ever running {LADDER[i - 1].value}")
        if b.total_spent > evidence_entitlement(b, policy) * 1.5 + 1e-9 and b.highest_passed() < 0:
            errs.append(f"{bid}: spent {b.total_spent:.0f} cpu-min with no rung passed")
        if b.in_flight and b.state not in ACTIVE_STATES:
            errs.append(f"{bid}: job in flight while {b.state.value}")
        if b.state is ResearchState.DORMANT and not b.dormant_reason:
            errs.append(f"{bid}: DORMANT without a reason")
        if b.state is ResearchState.RETIRED and Stage.INTEGRATION.value not in b.passed and b.dormant_reason == "":
            if b.highest_passed() < len(LADDER) - 1 and not any(t.branch_id == bid and t.to_state == "RETIRED" and "dormant" in t.reason for t in state.log):
                errs.append(f"{bid}: RETIRED without completing the ladder or a stated retirement reason")
    return errs


def branch_table(state: ManagerState, now) -> list[dict]:
    rows = []
    for bid, b in sorted(state.branches.items()):
        rows.append({"branch": bid, "state": b.state.value, "frontier": b.frontier.value, "passed": sorted(b.passed),
                     "cpu_min": round(b.total_spent, 2), "runs": len(b.runs), "idle_days": (as_date(now) - as_date(b.last_progress or b.created)).days,
                     "family": b.family, "problem": b.problem.value})
    return rows


def report(state: ManagerState, now) -> str:
    lines = [f"Research compute manager @ {as_date(now).isoformat()}  (IMPLEMENTED - NOT VALIDATED)"]
    counts: dict[str, int] = {}
    for b in state.branches.values():
        counts[b.state.value] = counts.get(b.state.value, 0) + 1
    lines.append("states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "states: none")
    for s, r in funnel(state).items():
        lines.append(f"  {s:24s} at_frontier={r['branches_at_frontier']:3d} tries={r['tries']:4d} pass_rate={r['pass_rate']:.2f} cpu_min={r['cpu_min']:.0f}")
    lines += [f"WARNING: {w}" for w in funnel_warnings(state)]
    lines.append(f"period spend {period_budget_used(state):.0f} cpu-min since {state.period_start or 'n/a'}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ ladder simulator
@dataclasses.dataclass(frozen=True)
class LadderSimResult:
    n: int
    mean_cost_null: float
    mean_cost_real: float
    real_reach_end: float
    null_reach_end: float
    null_reach_rung3: float
    real_failed_early: float
    mean_repeats_real: float


def simulate_ladder(policy: LadderPolicy, true_effect: float, sd: float, n_branches: int, seed: int, obs_per_test: int = 60,
                    share_real: float = 0.5) -> LadderSimResult:
    """Run the policy against synthetic hypotheses of KNOWN truth (a share carries `true_effect`, the rest exactly zero) and report
    what it costs and how often it errs. This is the planted-world check of section 18: nulls must die cheap, real effects must
    survive an underpowered first look. Deterministic in `seed`; no data is read."""
    rng = np.random.default_rng(seed)
    cost = {True: [], False: []}
    reach = {True: 0, False: 0}
    rung3 = {True: 0, False: 0}
    early = 0
    reps_real = []
    cnt = {True: 0, False: 0}
    for i in range(n_branches):
        real = bool(rng.random() < share_real)
        cnt[real] += 1
        eff = true_effect if real else 0.0
        st = ManagerState()
        q = ResearchQuestion.make(f"sim {i}", "sim", Problem.VOLATILITY, "2000-01-01", "2000-01-01", "s", "f")
        b = make_branch(st, q, "2000-01-01", "sim", looks=max(1, n_branches // 10))
        day = dt.date(2000, 1, 2)
        total = 0.0
        reps = 0
        for _ in range(40):
            if b.state not in ACTIVE_STATES:
                break
            rule = policy.rule(b.frontier)
            grow = policy.growth ** b.repeats.get(b.frontier.value, 0)
            n_obs = int(obs_per_test * rule.planned_tests * grow * 4 ** stage_index(b.frontier))
            se = sd / math.sqrt(max(n_obs, 1))
            est = rng.normal(eff, se)
            units = max(1, rule.min_units + 1)
            tests = rule.planned_tests
            p_unit = _NORM.cdf(eff / (sd / math.sqrt(obs_per_test * max(1, n_obs // (units * obs_per_test)))))
            p_test = _NORM.cdf(eff / (sd / math.sqrt(obs_per_test)))
            pos, npos = int(rng.binomial(units, p_unit)), int(rng.binomial(tests, p_test))
            ev = StageEvidence(b.frontier, n_obs, float(est), float(se), tests, npos, units, pos, int(rng.random() < (0.8 if real else 0.2)),
                               True, (day - dt.timedelta(days=1)).isoformat(), rule.base_cpu_min * grow)
            b.in_flight = "sim"
            if b.state is ResearchState.QUEUED or LAUNCH_STATE[b.frontier] in ALLOWED[b.state]:
                _record(st, b, LAUNCH_STATE[b.frontier], day, "sim")
            dec = record_result(st, b.branch_id, dataclasses.replace(ev, integration_delta=(0.01 if real else -0.01), integration_se=0.005), day, policy)
            total += ev.cost_cpu_min
            if dec.action is Act.AUDIT:
                clear_audit(st, b.branch_id, day, False, "sim: no leak planted")
            if dec.action is Act.REPEAT:
                reps += 1
            day += dt.timedelta(days=1)
        cost[real].append(total)
        reach[real] += int(b.passed.get(Stage.INTEGRATION.value) is not None)
        rung3[real] += int(stage_index(b.frontier) >= 2 or b.highest_passed() >= 2)
        if real:
            early += int(b.state is ResearchState.FAILED and b.highest_passed() < 1)
            reps_real.append(reps)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return LadderSimResult(n_branches, mean(cost[False]), mean(cost[True]), reach[True] / max(cnt[True], 1), reach[False] / max(cnt[False], 1),
                           rung3[False] / max(cnt[False], 1), early / max(cnt[True], 1), mean(reps_real))


# ------------------------------------------------------------------------------------------------ public entry
@dataclasses.dataclass(frozen=True)
class StepResult:
    selection: Selection
    launched: tuple[str, ...]
    decisions: tuple[tuple[str, EscalationDecision], ...]
    lost: tuple[str, ...]
    rolled_period: bool
    problems: tuple[str, ...]


def step(state: ManagerState, now, policy: LadderPolicy | None = None, results: Iterable[tuple[str, StageEvidence]] = (),
         budget: ComputeBudget | None = None, free_gb: float | None = None, period_cpu_min: float = 600.0,
         ledger: "C.ExperimentLedger | None" = None, code_hash: str = "", seed: int = 0, dry_run: bool = False) -> StepResult:
    """One scheduling tick for the research loop: roll the spending period, apply finished results (in branch-id order, so the
    outcome does not depend on arrival order), reconcile lost jobs, select the next jobs and (unless dry_run) launch them. Never
    raises on a normal refusal: it reports it in `selection.deferred`. Fails closed when memory is unreadable (free_gb None)."""
    policy = policy or LadderPolicy()
    bad = policy.validate()
    if bad:
        raise ValueError("invalid LadderPolicy: " + "; ".join(bad))
    rolled = roll_period(state, now)
    decisions = []
    for bid, ev in sorted(results, key=lambda t: t[0]):
        decisions.append((bid, record_result(state, bid, ev, now, policy)))
    lost = tuple(reconcile_ledger(state, ledger, now)) if ledger is not None else ()
    budget = budget or ComputeBudget(period_cpu_min, free_gb or 0.0)
    sel = select(state, now, policy, budget, free_gb, period_cpu_min,
                 real_data_running=sum(1 for b in state.branches.values() if b.in_flight and policy.rule(b.frontier).real_data))
    launched = []
    if not dry_run:
        for a in sel.allocations:
            try:
                launch(state, a, now, policy, seed, ledger, code_hash)
                launched.append(a.branch_id)
            except (C.DuplicateExperiment, LadderError):
                continue
    return StepResult(sel, tuple(launched), tuple(decisions), lost, rolled, tuple(audit_ladder(state, policy)))


# ------------------------------------------------------------------------------------------------ run design
@dataclasses.dataclass(frozen=True)
class RunPlan:
    branch_id: str
    stage: Stage
    n_obs_needed: int | None
    n_obs_affordable: int
    power_if_affordable: float
    cpu_min: float
    feasible: bool
    reasons: tuple[str, ...]


def plan_run(state: ManagerState, branch_id: str, policy: LadderPolicy, sd_obs: float, obs_per_cpu_min: float) -> RunPlan:
    """Design the branch's next run BEFORE it is funded: how many observations it needs to see the smallest useful effect with the
    rung's required power, how many the rung's cap can afford, and the power actually bought. A run that cannot reach the power is
    reported infeasible so nobody launches an underpowered test whose null would be read as an answer."""
    b = state.branches[branch_id]
    rule = policy.rule(b.frontier)
    if sd_obs <= 0 or obs_per_cpu_min <= 0:
        raise ValueError("sd_obs and obs_per_cpu_min must be positive")
    need = required_n(policy.min_effect, sd_obs, policy.alpha, rule.min_power)
    cpu = min(planned_cost(state, b, policy), rule.max_cpu_min)
    afford = int(cpu * obs_per_cpu_min)
    pw = power_of(afford, policy.min_effect, sd_obs, policy.alpha) if afford > 0 else 0.0
    reasons = []
    if afford < need:
        cpu_needed = need / obs_per_cpu_min
        reasons.append(f"needs {need} observations (~{cpu_needed:.0f} cpu-min) but the rung affords {afford} (power {pw:.2f})")
        if cpu_needed > rule.max_cpu_min:
            reasons.append(f"beyond the {rule.max_cpu_min:.0f} cpu-min cap: PARK until more data or a cheaper representation exists")
    return RunPlan(branch_id, b.frontier, need, afford, pw, cpu, afford >= need, tuple(reasons))


@dataclasses.dataclass(frozen=True)
class ScreenResult:
    survivors: tuple[str, ...]
    scores: Mapping[str, float]
    total_cost: float
    looks: int
    t_required: float
    eliminated_round: Mapping[str, int]


def screen_variants(arms: Sequence[str], evaluate: Callable[[str, float], float], policy: LadderPolicy, total_cpu_min: float,
                    min_budget: float | None = None, eta: int = 3) -> ScreenResult:
    """Rung-1 screening of many variants of ONE question by successive halving (research_policy.successive_halving): every variant
    gets a cheap look, the best 1/eta get more. Returns the survivors together with the multiplicity-corrected t that a survivor's
    final score must still beat, because picking the best of N looks inflates its apparent effect."""
    from engine.learning.research_policy import successive_halving
    rule = policy.rule(Stage.CHEAP_SCREEN)
    floor = min_budget if min_budget is not None else rule.cpu_min_per_test
    res = successive_halving(list(arms), evaluate, floor, total_cpu_min, eta)
    scores = {a: float(evaluate(a, floor)) for a in res.survivors}
    return ScreenResult(tuple(res.survivors), scores, res.total_cost, len(arms), multiplicity_threshold(rule.pass_t, len(arms)),
                        dict(res.eliminated))


def early_stop_screen(outcomes: Sequence[int], null_rate: float = 0.5, alt_rate: float = 0.62, cost_per_test: float = 0.2,
                      planned_tests: int | None = None) -> dict:
    """Stop a cheap screen the moment a sequential test (Wald SPRT on 'did this test agree with the hypothesis') is decisive.
    Returns the decision and the compute saved against running all planned tests: a null screen that is already clearly null
    must not run to its planned length."""
    from engine.learning.research_policy import sprt_bernoulli
    st = sprt_bernoulli(list(outcomes), null_rate, alt_rate)
    planned = planned_tests if planned_tests is not None else len(outcomes)
    used = min(st.n, planned)
    return {"decision": st.decision, "tests_used": used, "tests_planned": planned, "cpu_min_used": used * cost_per_test,
            "cpu_min_saved": max(0, planned - used) * cost_per_test, "llr": st.llr}


def family_looks(state: ManagerState, family: str, now, window_days: int = 90) -> int:
    """How many branches of a family were started in the window: the multiplicity that a new branch's cheap screen inherits."""
    lo = as_date(now) - dt.timedelta(days=window_days)
    return 1 + sum(1 for b in state.branches.values() if b.family == family and as_date(b.created) >= lo)


# ------------------------------------------------------------------------------------------------ forecasting
@dataclasses.dataclass(frozen=True)
class FunnelForecast:
    expected_cpu_min: float
    by_stage: Mapping[str, float]
    expected_completions: float
    branches_considered: int
    cost_per_completion: float | None


def forecast_funnel(state: ManagerState, policy: LadderPolicy) -> FunnelForecast:
    """Expected compute to run every active branch to its end, from the learned pass rates: a branch at rung k pays rung k for
    certain, then rung k+1 with the pass probability of k, and so on. Compares with the period budget so a backlog that cannot
    be afforded is visible instead of silently starving the late rungs."""
    by = {s.value: 0.0 for s in LADDER}
    comp = 0.0
    n = 0
    for b in state.branches.values():
        if b.state not in ACTIVE_STATES:
            continue
        n += 1
        p = 1.0
        for i in range(stage_index(b.frontier), len(LADDER)):
            s = LADDER[i]
            mean_cost = state.stats.mean_cost(s) or policy.rule(s).base_cpu_min
            by[s.value] += p * mean_cost * state.cost_model.multiplier(s.value)
            p *= state.stats.pass_rate(s)
        comp += p
    tot = sum(by.values())
    return FunnelForecast(tot, by, comp, n, tot / comp if comp > 1e-9 else None)


def worth_finishing(state: ManagerState, branch_id: str, now, policy: LadderPolicy, value_of_completion: float,
                    cpu_min_value: float) -> tuple[bool, dict]:
    """Stop-loss test for one branch: expected value of finishing the ladder (probability of surviving every remaining rung x the
    value of a validated result) against the expected remaining compute priced at `cpu_min_value` per CPU-minute. The waste
    controller uses this alongside its own progress-per-compute record."""
    b = state.branches[branch_id]
    p = state.stats.prob_reach_end(b.frontier)
    remaining = 0.0
    q = 1.0
    for i in range(stage_index(b.frontier), len(LADDER)):
        s = LADDER[i]
        remaining += q * (state.stats.mean_cost(s) or policy.rule(s).base_cpu_min)
        q *= state.stats.pass_rate(s)
    ev = p * value_of_completion
    return ev > remaining * cpu_min_value, {"p_finish": p, "expected_value": ev, "expected_remaining_cpu_min": remaining,
                                            "price": remaining * cpu_min_value}


def starving(state: ManagerState, now, max_wait_days: int = 30) -> list[str]:
    """Active, idle branches that have waited longer than `max_wait_days` without a job: the priority function must never let a
    branch starve, so anything listed here means the budget is too small or an admission rule is too strict."""
    out = []
    for bid, b in sorted(state.branches.items()):
        if b.state in ACTIVE_STATES and not b.in_flight and not b.audit_pending:
            if (as_date(now) - as_date(b.last_progress or b.created)).days > max_wait_days:
                out.append(bid)
    return out


def jobs_overrunning(state: ManagerState, ledger: "C.ExperimentLedger", now_epoch: float, factor: float = 4.0,
                     seconds_per_cpu_min: float = 60.0) -> list[dict]:
    """Running jobs that have used more than `factor` times the wall time their plan allowed: candidates for the compute ledger's
    stale/crash handling. Read-only; killing a process stays with the process's owner (CONTEXT rule 11)."""
    led = ledger.load()
    out = []
    for b in state.branches.values():
        e = led.get(b.in_flight) if b.in_flight else None
        if not e or e.get("state") != C.RUNNING:
            continue
        started = next((h[0] for h in e["history"] if h[1] == C.RUNNING), None)
        planned = float(e["spec"]["params"].get("cpu_min", 1.0)) * seconds_per_cpu_min
        if started is not None and now_epoch - float(started) > factor * planned:
            out.append({"branch": b.branch_id, "job": b.in_flight, "running_s": now_epoch - float(started), "planned_s": planned})
    return out


# ------------------------------------------------------------------------------------------------ persistence with verification
def state_hash(state: ManagerState) -> str:
    return stable_hash(state.to_dict(), 20)


class Snapshotter:
    """Rotating, hash-verified snapshots of the manager state so a crash or a torn write costs at most one tick. `restore` walks
    back to the newest snapshot whose stored hash matches its content and reports what it skipped."""

    def __init__(self, directory: str | Path, keep: int = 5):
        if keep < 1:
            raise ValueError("keep >= 1")
        self.dir, self.keep = Path(directory), keep

    def _files(self) -> list[Path]:
        return sorted(self.dir.glob("manager_state.*.json"))

    def write(self, state: ManagerState, now) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        seq = 1 + max((int(p.stem.split(".")[-1]) for p in self._files()), default=0)
        body = state.to_dict()
        path = self.dir / f"manager_state.{seq:06d}.json"
        C.atomic_write_json(path, {"hash": stable_hash(body, 20), "saved": as_date(now).isoformat(), "state": body})
        for old in self._files()[:-self.keep]:
            old.unlink()
        return path

    def restore(self) -> tuple[ManagerState, dict]:
        skipped = []
        for p in reversed(self._files()):
            d = C.read_json(p, None)
            if isinstance(d, dict) and "state" in d and d.get("hash") == stable_hash(d["state"], 20):
                st = ManagerState.from_dict(d["state"])
                return st, {"file": p.name, "saved": d.get("saved"), "skipped": skipped}
            skipped.append(p.name)
        return ManagerState(), {"file": None, "saved": None, "skipped": skipped}


# ------------------------------------------------------------------------------------------------ explanation
def explain(state: ManagerState, branch_id: str, now, policy: LadderPolicy) -> list[str]:
    """Why the branch is where it is, in order: every transition with its reason, then what it would need next."""
    b = state.branches[branch_id]
    lines = [f"{branch_id} [{b.problem.value}/{b.family}] {b.text[:80]}"]
    lines += [f"  {t.at}  {t.from_state} -> {t.to_state}: {t.reason}" for t in state.log if t.branch_id == branch_id]
    score, parts = priority(state, b, now, policy)
    lines.append(f"  state={b.state.value} frontier={b.frontier.value} spent={b.total_spent:.1f} cpu-min "
                 f"entitled={evidence_entitlement(b, policy):.0f}")
    if b.state in ACTIVE_STATES:
        ok, why = authorise(state, b, b.frontier, planned_cost(state, b, policy), policy)
        lines.append(f"  next job: {'authorised' if ok else 'REFUSED - ' + why}; priority {score:.4f} (p_reach_end {parts['p_reach_end']:.2f}"
                     + (f"; missing estimates {parts['missing_estimates']}" if parts["missing_estimates"] else "") + ")")
    elif b.state is ResearchState.DORMANT:
        lines.append(f"  dormant because: {b.dormant_reason}")
    return lines


# ------------------------------------------------------------------------------------------------ hypothesis tree links
def children_of(state: ManagerState, branch_id: str) -> list[str]:
    q = state.branches[branch_id].question_id
    return sorted(bid for bid, b in state.branches.items() if q in b.parents)


def subtree_spend(state: ManagerState, branch_id: str) -> float:
    """CPU-minutes spent on a branch and everything descended from it (cycles cannot occur: a child names a parent question that
    existed when it was created, but the walk is guarded anyway)."""
    seen, stack, total = set(), [branch_id], 0.0
    while stack:
        bid = stack.pop()
        if bid in seen:
            continue
        seen.add(bid)
        total += state.branches[bid].total_spent
        stack.extend(children_of(state, bid))
    return total


def propagate_failure(state: ManagerState, branch_id: str, now, reason: str) -> list[str]:
    """When a branch FAILS on a powered null, queued (never-run) descendants that rest on it are parked: their premise was refuted,
    so they must not be funded. Descendants that already ran keep their own evidence and are only annotated. Returns the ids parked."""
    b = state.branches[branch_id]
    if b.state is not ResearchState.FAILED:
        raise LadderError("only a FAILED branch propagates")
    parked = []
    for cid in children_of(state, branch_id):
        c = state.branches[cid]
        if c.state is ResearchState.QUEUED and not c.runs:
            _park_state(state, c, now, f"premise refuted: parent {branch_id} failed ({reason})")
            parked.append(cid)
        elif c.state in ACTIVE_STATES:
            c.runs.append({"stage": c.frontier.value, "at": as_date(now).isoformat(), "action": "PARENT_FAILED", "parent": branch_id})
    return parked


# ------------------------------------------------------------------------------------------------ period planning
@dataclasses.dataclass(frozen=True)
class PeriodPlan:
    period_cpu_min: float
    by_stage: Mapping[str, float]
    by_problem: Mapping[str, float]
    explore_reserved: float
    backlog_cpu_min: float
    backlog_ratio: float                # >1: the backlog cannot be worked off this period
    notes: tuple[str, ...]


def plan_period(state: ManagerState, policy: LadderPolicy, period_cpu_min: float) -> PeriodPlan:
    """Split a period's compute across rungs and objectives. Rung shares follow the forecast (what the active branches will need)
    but are clipped to the policy caps, the explore floor is reserved first, and objectives get shares in proportion to their
    priority weights among the problems that actually have work (volatility first, section 0)."""
    fc = forecast_funnel(state, policy)
    notes = []
    floor = policy.explore_floor * period_cpu_min
    remaining = max(0.0, period_cpu_min - floor)
    want = {k: v for k, v in fc.by_stage.items()}
    tot_want = sum(want.values())
    by_stage = {}
    for s in LADDER:
        cap = policy.stage_share_cap.get(s.value, 1.0) * period_cpu_min
        share = (want[s.value] / tot_want * remaining) if tot_want > 0 else 0.0
        by_stage[s.value] = min(cap, share) + (floor if s is Stage.CHEAP_SCREEN else 0.0)
    active_problems = sorted({b.problem.value for b in state.branches.values() if b.state in ACTIVE_STATES})
    w = {p: policy.problem_weight.get(p, 0.4) for p in active_problems}
    wt = sum(w.values())
    by_problem = {p: period_cpu_min * v / wt for p, v in w.items()} if wt > 0 else {}
    if Problem.DIRECTION.value in by_problem and best_rung(state, Problem.VOLATILITY) < policy.volatility_rung_required:
        moved = by_problem[Problem.DIRECTION.value] * 0.5
        by_problem[Problem.DIRECTION.value] -= moved
        if Problem.VOLATILITY.value in by_problem:
            by_problem[Problem.VOLATILITY.value] += moved
        notes.append("volatility has not earned its rungs yet: half of the direction share moved to volatility")
    ratio = fc.expected_cpu_min / period_cpu_min if period_cpu_min > 0 else math.inf
    if ratio > 1.0:
        notes.append(f"backlog is {ratio:.1f}x the period: expect deferrals; the explore floor stays reserved")
    if not state.branches:
        notes.append("no branches yet: the whole period is unallocated exploration budget")
    return PeriodPlan(period_cpu_min, by_stage, by_problem, floor, fc.expected_cpu_min, ratio, tuple(notes))


# ------------------------------------------------------------------------------------------------ ladder calibration (advice only)
@dataclasses.dataclass(frozen=True)
class LadderAdvice:
    stage: Stage
    tries: int
    pass_rate: float
    next_stage_fail_rate: float | None
    advice: str


def calibrate_ladder(state: ManagerState, policy: LadderPolicy, min_tries: int = 10) -> list[LadderAdvice]:
    """Read the ladder's own record and say whether a rung looks mis-set. ADVICE only: thresholds are tuned in the validation wave,
    never automatically, because tuning a gate on its own outcomes is the overfitting this whole system exists to prevent.
      * a rung passing > 90% screens nothing -> raise pass_t;
      * a rung passing < 3% while later rungs never ran -> may be too strict (the rung-1 miss rate is unobservable without audits);
      * a rung whose passers mostly die at the next rung wastes that rung's compute -> tighten it."""
    out = []
    for i, s in enumerate(LADDER):
        tries, passes = state.stats.tries[s.value], state.stats.passes[s.value]
        if tries < min_tries:
            out.append(LadderAdvice(s, tries, passes / tries if tries else 0.0, None, "insufficient tries"))
            continue
        rate = passes / tries
        nxt_fail = None
        if i + 1 < len(LADDER):
            nt = state.stats.tries[LADDER[i + 1].value]
            if nt >= min_tries:
                nxt_fail = 1.0 - state.stats.passes[LADDER[i + 1].value] / nt
        if rate > 0.9:
            msg = f"raise pass_t (now {policy.rule(s).pass_t}): the rung passes {rate:.0%}"
        elif rate < 0.03:
            msg = "audit a sample of its rejects (sample_failed_for_recheck): a rung that rejects nearly everything may be discarding real effects"
        elif nxt_fail is not None and nxt_fail > 0.8:
            msg = f"tighten: {nxt_fail:.0%} of its passers die at the next rung, which is wasted compute"
        else:
            msg = "no change indicated"
        out.append(LadderAdvice(s, tries, rate, nxt_fail, msg))
    return out


def sample_failed_for_recheck(state: ManagerState, rng: np.random.Generator, k: int, min_age_days: int = 30, now=None) -> list[str]:
    """A seeded sample of FAILED branches to re-run once at a larger sample. The false-negative rate of the ladder is otherwise
    unobservable (rejected ideas are never looked at again), and 'never abandon a promising hypothesis' has to be checked, not
    assumed. Deterministic for a given rng state and branch set."""
    pool = sorted(bid for bid, b in state.branches.items() if b.state is ResearchState.FAILED
                  and (now is None or (as_date(now) - as_date(b.updated or b.created)).days >= min_age_days))
    if not pool or k <= 0:
        return []
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[int(i)] for i in sorted(idx)]


def reopen_failed(state: ManagerState, branch_id: str, now, reason: str) -> None:
    """FAILED -> DORMANT -> QUEUED for a re-check: the failure record stays in the log, the rung restarts with a doubled sample."""
    b = state.branches[branch_id]
    if b.state is not ResearchState.FAILED:
        raise LadderError(f"{branch_id} is {b.state.value}, not FAILED")
    _park_state(state, b, now, "re-check of a rejected branch: " + reason)
    revive(state, branch_id, now, "re-check: " + reason)
    b.repeats = {Stage.CHEAP_SCREEN.value: 1}


# ------------------------------------------------------------------------------------------------ cost accounting views
def cost_breakdown(state: ManagerState) -> dict[str, dict[str, float]]:
    """Where the compute went: by problem, by family, by final state, by rung. The raw material for the value accountant."""
    out: dict[str, dict[str, float]] = {"problem": {}, "family": {}, "state": {}, "stage": {}}
    for b in state.branches.values():
        c = b.total_spent
        for view, key in (("problem", b.problem.value), ("family", b.family), ("state", b.state.value)):
            out[view][key] = out[view].get(key, 0.0) + c
        for s, v in b.spent.items():
            out["stage"][s] = out["stage"].get(s, 0.0) + v
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def conservation_errors(state: ManagerState, tol: float = 1e-6) -> list[str]:
    """Compute is conserved: the sum of branch spend per rung equals the ladder statistics' cost for that rung."""
    errs = []
    per = cost_breakdown(state)["stage"]
    for s in LADDER:
        a, b = per.get(s.value, 0.0), state.stats.cost[s.value]
        if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            errs.append(f"{s.value}: branches spent {a:.3f} but statistics say {b:.3f}")
    return errs


def summary_rows(state: ManagerState, policy: LadderPolicy, now) -> list[dict]:
    """One row per branch with the priority and the authorisation verdict, for dashboards and the daily research report."""
    rows = []
    for r in branch_table(state, now):
        b = state.branches[r["branch"]]
        if b.state in ACTIVE_STATES:
            ok, why = authorise(state, b, b.frontier, planned_cost(state, b, policy), policy)
            r["priority"], r["authorised"], r["why"] = round(priority(state, b, now, policy)[0], 5), ok, why
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ closed-loop replay
def replay_schedule(policy: LadderPolicy, truth: Mapping[str, float], seed: int, ticks: int, period_cpu_min: float = 600.0,
                    sd: float = 0.05, per_tick_free_gb: float = 12.0, obs_per_test: int = 60) -> dict:
    """Drive the WHOLE manager (step -> results -> step ...) against synthetic hypotheses of known effect. `truth` maps
    'family:name' to its true effect. Each tick the selected jobs are 'executed' by drawing evidence from that truth; the function
    reports what the manager did with compute: spend per hypothesis, who reached the end, who failed, and whether any null was
    ever funded past rung 2. Deterministic in `seed`. This is the end-to-end planted-world check (no data is read)."""
    rng = np.random.default_rng(seed)
    st = ManagerState()
    day = dt.date(2001, 1, 1)
    name_of: dict[str, str] = {}
    for name in sorted(truth):
        q = ResearchQuestion.make(name, "sim", Problem.VOLATILITY, day.isoformat(), day.isoformat(), "s", "f")
        b = make_branch(st, q, day, name.split(":")[0], looks=max(1, len(truth) // 4))
        name_of[b.branch_id] = name
    pending: list[tuple[str, StageEvidence]] = []
    budget = ComputeBudget(period_cpu_min, per_tick_free_gb, safety_ram_gb=2.5, max_ram_per_job_gb=4.0, real_data_slots=2)
    for _ in range(ticks):
        day += dt.timedelta(days=1)
        res = step(st, day, policy, pending, budget, per_tick_free_gb, period_cpu_min, seed=int(rng.integers(1 << 30)))
        pending = []
        for a in res.selection.allocations:
            if a.branch_id not in res.launched:
                continue
            b = st.branches[a.branch_id]
            eff = truth[name_of[a.branch_id]]
            rule = policy.rule(a.stage)
            grow = policy.growth ** b.repeats.get(a.stage.value, 0)
            n_obs = int(obs_per_test * rule.planned_tests * grow * 4 ** stage_index(a.stage))
            se = sd / math.sqrt(n_obs)
            units = max(1, rule.min_units + 1)
            p_pos = _NORM.cdf(eff / (sd / math.sqrt(obs_per_test)))
            pending.append((a.branch_id, StageEvidence(
                a.stage, n_obs, float(rng.normal(eff, se)), se, rule.planned_tests, int(rng.binomial(rule.planned_tests, p_pos)), units,
                int(rng.binomial(units, min(0.999, _NORM.cdf(eff / se / math.sqrt(units))))), int(rng.random() < (0.8 if eff else 0.2)),
                True, (day - dt.timedelta(days=1)).isoformat(), a.cpu_min, eff if a.stage is Stage.INTEGRATION else None,
                0.002 if a.stage is Stage.INTEGRATION else None)))
        for bid, b in st.branches.items():
            if b.audit_pending:
                clear_audit(st, bid, day, False, "replay: no leak planted")
    out = {"spend": {}, "state": {}, "reached_end": [], "null_funded_past_rung2": []}
    for bid, b in sorted(st.branches.items()):
        n = name_of[bid]
        out["spend"][n], out["state"][n] = b.total_spent, b.state.value
        if Stage.INTEGRATION.value in b.passed:
            out["reached_end"].append(n)
        if truth[n] == 0.0 and b.highest_passed() >= 2:
            out["null_funded_past_rung2"].append(n)
    out["problems"] = audit_ladder(st, policy) + conservation_errors(st)
    return out


# ------------------------------------------------------------------------------------------------ evidence builders
def pooled_evidence(stage: Stage, per_unit: Mapping[str, tuple[float, float, int]], data_through: str, cost_cpu_min: float,
                    fresh: bool = False, replications: int = 0, n_tests: int = 1, n_positive: int | None = None,
                    complexity_added: float = 0.0) -> StageEvidence:
    """Build a StageEvidence from per-unit results {unit: (effect, se, n_obs)} (units = contexts at rung 2, years at rung 3).
    The pooled effect is the inverse-variance random-effects mean (DerSimonian-Laird), so a heterogeneous family of contexts widens
    the standard error instead of hiding disagreement; units_positive counts contexts agreeing in sign with the pooled effect."""
    items = [(float(e), float(s), int(n)) for e, s, n in per_unit.values() if math.isfinite(e) and math.isfinite(s) and s > 0]
    if not items:
        return StageEvidence(stage, 0, 0.0, 0.0, n_tests, n_positive or 0, 1, 0, replications, fresh, data_through, cost_cpu_min,
                             complexity_added=complexity_added)
    e = np.array([i[0] for i in items])
    w = 1.0 / np.array([i[1] ** 2 for i in items])
    fixed = float((w * e).sum() / w.sum())
    q = float((w * (e - fixed) ** 2).sum())
    k = len(items)
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    wr = 1.0 / (1.0 / w + tau2)
    pooled = float((wr * e).sum() / wr.sum())
    se = float(math.sqrt(1.0 / wr.sum()))
    agree = int(np.sum(np.sign(e) == (1.0 if pooled >= 0 else -1.0)))
    n_tot = sum(i[2] for i in items)
    return StageEvidence(stage, n_tot, pooled, se, n_tests, n_positive if n_positive is not None else agree, k, agree, replications,
                         fresh, data_through, cost_cpu_min, complexity_added=complexity_added)


def evidence_from_screen(stage_scores: Sequence[float], se_each: float, n_obs_each: int, data_through: str, cost_cpu_min: float) -> StageEvidence:
    """Rung-1 evidence from many cheap tests of the SAME hypothesis (different seeds/cuts): the mean effect over tests, its
    standard error from the tests' own spread (not from the nominal se, which the tests share), and the sign agreement count."""
    x = np.asarray(stage_scores, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return StageEvidence(Stage.CHEAP_SCREEN, 0, 0.0, 0.0, 1, 0, 1, 0, 0, False, data_through, cost_cpu_min)
    spread = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else se_each
    se = max(spread, se_each / math.sqrt(n))
    return StageEvidence(Stage.CHEAP_SCREEN, n * n_obs_each, float(x.mean()), se, n, int(np.sum(x > 0)), 1, 0, 0, False, data_through, cost_cpu_min)


# ------------------------------------------------------------------------------------------------ dispatch
def dispatch_jobs(state: ManagerState, sel: Selection, out_root: str | Path, spec_dir: str | Path, ledger_path: str | Path,
                  now, seed: int = 0, python: str | None = None) -> list:
    """resources.Job objects for a Selection, ready for engine.resources.JobQueue: each runs one ExperimentSpec in its own process
    (engine.learning.compute.job_for). Nothing is launched here; ordering is by rung then branch id, so cheap work is queued first."""
    jobs = []
    for a in sorted(sel.allocations, key=lambda a: (stage_index(a.stage), a.branch_id)):
        spec = to_spec(state, a, now, seed)
        jobs.append(C.job_for(spec, spec_dir, out_root, ledger_path, python))
    return jobs


def machine_budget(cpu_minutes: float, real_data_slots: int = 1, free_gb: float | None = None) -> ComputeBudget:
    """A round's ComputeBudget from the machine's actual free memory (CONTEXT rule 10: nothing starts under the RAM floor)."""
    free = free_gb if free_gb is not None else (R.memory_gb()[0] or 0.0)
    return ComputeBudget(cpu_minutes, free, safety_ram_gb=C.MIN_FREE_GB, real_data_slots=real_data_slots)


def family_fairness(state: ManagerState, max_share: float = 0.5) -> dict:
    """Share of all compute spent per family; a family above `max_share` is over-served and is listed, so one productive-looking
    line of work cannot crowd out the rest (research diversity, section 38)."""
    tot = sum(b.total_spent for b in state.branches.values())
    per: dict[str, float] = {}
    for b in state.branches.values():
        per[b.family] = per.get(b.family, 0.0) + b.total_spent
    shares = {f: (v / tot if tot > 0 else 0.0) for f, v in sorted(per.items())}
    return {"shares": shares, "over": {f: s for f, s in shares.items() if s > max_share}, "total_cpu_min": tot}


def diversify(alloc: Sequence[Allocation], state: ManagerState, max_share: float = 0.5) -> tuple[list[Allocation], list[tuple[str, str]]]:
    """Drop allocations of an over-served family until its share of THIS round falls to `max_share` (lowest-scored first).
    Returns (kept, dropped-with-reason); with a single family nothing is dropped (there is nothing to diversify toward)."""
    fams = {a.branch_id: state.branches[a.branch_id].family for a in alloc}
    if len(set(fams.values())) < 2:
        return list(alloc), []
    kept = sorted(alloc, key=lambda a: (-a.score, a.branch_id))
    dropped: list[tuple[str, str]] = []
    while True:
        tot = sum(a.cpu_min for a in kept)
        by: dict[str, float] = {}
        for a in kept:
            by[fams[a.branch_id]] = by.get(fams[a.branch_id], 0.0) + a.cpu_min
        worst = max(by, key=lambda f: (by[f], f))
        if tot <= 0 or by[worst] / tot <= max_share or len([a for a in kept if fams[a.branch_id] == worst]) <= 1:
            break
        victim = [a for a in kept if fams[a.branch_id] == worst][-1]
        kept.remove(victim)
        dropped.append((victim.branch_id, f"family {worst} would take {by[worst] / tot:.0%} of the round"))
    return kept, dropped


# ------------------------------------------------------------------------------------------------ history and what-if
def state_at(state: ManagerState, as_of) -> dict[str, str]:
    """Each branch's state as of a date (from the hash-chained log): only transitions strictly before `as_of` count."""
    out: dict[str, str] = {}
    cut = as_date(as_of)
    for bid in state.branches:
        if as_date(state.branches[bid].created) < cut:
            out[bid] = ResearchState.QUEUED.value
    for t in state.log:
        if as_date(t.at) < cut:
            out[t.branch_id] = t.to_state
    return out


def time_in_state(state: ManagerState, branch_id: str, now) -> dict[str, int]:
    """Days a branch spent in each state up to `now` (the last state runs to `now`)."""
    b = state.branches[branch_id]
    marks = [(as_date(b.created), ResearchState.QUEUED.value)] + [(as_date(t.at), t.to_state) for t in state.log if t.branch_id == branch_id]
    out: dict[str, int] = {}
    for (d0, s), (d1, _) in zip(marks, marks[1:] + [(as_date(now), "")]):
        out[s] = out.get(s, 0) + max(0, (d1 - d0).days)
    return out


def what_if(evidence: Sequence[StageEvidence], policies: Mapping[str, LadderPolicy], looks: int = 1) -> dict[str, list[str]]:
    """Run the same sequence of first-look evidence through different policies and show each policy's action per item. Answers
    'would a laxer/stricter rung have changed what was funded?' without spending anything. Deterministic; reads no state."""
    out: dict[str, list[str]] = {}
    for name, pol in sorted(policies.items()):
        acts = []
        for e in evidence:
            a = assess(e, pol, looks)
            acts.append(decide(e, a, pol, 0).action.value)
        out[name] = acts
    return out


def cost_overrun_report(state: ManagerState) -> dict[str, dict[str, float]]:
    """Learned overrun multiplier per rung (CostModel): >1 means jobs cost more than planned. A rung whose multiplier is far from 1
    is mis-planned and its budget shares in `plan_period` are wrong by the same factor."""
    cm = state.cost_model
    return {s.value: {"n": float(cm.n.get(s.value, 0)), "multiplier": cm.multiplier(s.value), "spread": cm.spread(s.value)} for s in LADDER}


def queue_age_report(state: ManagerState, now) -> dict:
    """Waiting-time statistics of QUEUED branches (days since creation): the median and the worst, per problem."""
    ages: dict[str, list[int]] = {}
    for b in state.branches.values():
        if b.state is ResearchState.QUEUED:
            ages.setdefault(b.problem.value, []).append((as_date(now) - as_date(b.created)).days)
    return {p: {"n": len(v), "median": float(np.median(v)), "max": max(v)} for p, v in sorted(ages.items())}


def rung_yield(state: ManagerState) -> dict[str, dict[str, float]]:
    """Per rung: CPU-minutes spent, branches that passed it, and CPU-minutes per pass. The price of one validated step, so a rung
    that is expensive per pass can be compared against the value the accountant credits at that rung."""
    out = {}
    for s in LADDER:
        passes = sum(1 for b in state.branches.values() if s.value in b.passed)
        cpu = state.stats.cost[s.value]
        out[s.value] = {"cpu_min": cpu, "passes": float(passes), "cpu_min_per_pass": (cpu / passes) if passes else math.inf}
    return out


def merge_states(a: ManagerState, b: ManagerState) -> ManagerState:
    """Combine two managers' states deterministically (for a crash where two ticks each wrote a partial file). A branch present in
    both keeps the one with more recorded runs (ties: the one updated later); logs are NOT merged, because a hash chain cannot be
    interleaved, so the merged log is the longer, intact one and any branch missing from it is reported by `audit_ladder`."""
    out = ManagerState()
    for bid in sorted(set(a.branches) | set(b.branches)):
        ba, bb = a.branches.get(bid), b.branches.get(bid)
        if ba is None or bb is None:
            out.branches[bid] = ba or bb
        else:
            out.branches[bid] = ba if (len(ba.runs), ba.updated) >= (len(bb.runs), bb.updated) else bb
    src = a if (len(a.log), a.period_start) >= (len(b.log), b.period_start) else b
    out.log = list(src.log)
    out.stats, out.cost_model = src.stats, src.cost_model
    out.period_start, out.period_spent = src.period_start, dict(src.period_spent)
    out.questions_seen = set(a.questions_seen) | set(b.questions_seen)
    return out


# ------------------------------------------------------------------------------------------------ intake and structural checks
@dataclasses.dataclass(frozen=True)
class IntakeResult:
    created: tuple[str, ...]
    duplicates: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]


def intake(state: ManagerState, questions: Iterable[tuple[ResearchQuestion, str]], now, policy: LadderPolicy | None = None,
           max_new: int | None = None) -> IntakeResult:
    """Register a batch of (question, family) pairs as branches. Each inherits the family's recent multiplicity (`family_looks`), so
    the hundredth idea of a family faces a higher cheap-screen bar than the first. Questions that name a ticker-like token or a
    real date are refused (section 29: questions handed onward must be identity-free), duplicates are reported, and `max_new`
    caps the intake so a burst of questions cannot outrun the compute that would test them. Order-independent: sorted by id."""
    import re
    created, dup, rej = [], [], []
    ident = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b[A-Z]{2,5}\b(?=\s+(?:stock|shares|ticker))")
    for q, fam in sorted(questions, key=lambda t: t[0].question_id):
        if ident.search(q.text):
            rej.append((q.question_id, "question text carries a real date or a ticker"))
            continue
        if max_new is not None and len(created) >= max_new:
            rej.append((q.question_id, "intake cap reached this tick"))
            continue
        b = make_branch(state, q, now, fam, family_looks(state, fam, now))
        (created if b is not None else dup).append(b.branch_id if b is not None else q.question_id)
    return IntakeResult(tuple(created), tuple(dup), tuple(rej))


def validate_state(state: ManagerState, policy: LadderPolicy) -> list[str]:
    """Structural invariants beyond the ladder audit (empty = clean): the frontier agrees with the rungs passed, rung records exist
    only for rungs before the frontier, repeats never exceed the allowance, and timestamps are ISO dates."""
    errs = []
    for bid, b in sorted(state.branches.items()):
        top = b.highest_passed()
        if b.state in ACTIVE_STATES and stage_index(b.frontier) > top + 1:
            errs.append(f"{bid}: frontier {b.frontier.value} is beyond the first unpassed rung")
        for s in b.passed:
            if stage_index(s) > stage_index(b.frontier) and b.state is not ResearchState.RETIRED:
                errs.append(f"{bid}: rung {s} recorded as passed beyond the frontier")
        for s, n in b.repeats.items():
            if n > policy.rule(s).max_repeats + 1:
                errs.append(f"{bid}: {n} repeats at {s} exceeds allowance {policy.rule(s).max_repeats}")
        for d in (b.created, b.updated, b.last_progress):
            if d:
                try:
                    as_date(d)
                except ValueError:
                    errs.append(f"{bid}: bad date {d!r}")
        if not 0.0 < b.waste_factor <= 1.0:
            errs.append(f"{bid}: waste_factor {b.waste_factor} outside (0,1]")
    return errs


# ------------------------------------------------------------------------------------------------ budget governor
@dataclasses.dataclass(frozen=True)
class BudgetAdvice:
    period_cpu_min: float
    reason: str
    backlog_ratio: float
    deferred_share: float


def budget_governor(state: ManagerState, policy: LadderPolicy, base_cpu_min: float, last_selection: Selection | None = None,
                    lo: float = 0.5, hi: float = 2.0) -> BudgetAdvice:
    """Suggest the next period's compute from what the ladder is asking for. ADVICE bounded to [lo, hi] x base: a growing backlog
    raises the ask a little (never past hi), an idle ladder lowers it so compute goes back to the shared machine. The operator, not
    this function, sets the real budget."""
    fc = forecast_funnel(state, policy)
    ratio = fc.expected_cpu_min / base_cpu_min if base_cpu_min > 0 else math.inf
    deferred = 0.0
    if last_selection is not None and (last_selection.allocations or last_selection.deferred):
        deferred = len(last_selection.deferred) / (len(last_selection.deferred) + len(last_selection.allocations))
    if fc.branches_considered == 0:
        return BudgetAdvice(base_cpu_min * lo, "no active branches: release compute", ratio, deferred)
    scale = min(hi, max(lo, 0.5 + 0.5 * min(ratio, 3.0) + 0.5 * deferred))
    why = f"backlog {ratio:.1f}x period, {deferred:.0%} of candidates deferred"
    return BudgetAdvice(base_cpu_min * scale, why, ratio, deferred)


def escalation_preview(state: ManagerState, policy: LadderPolicy) -> list[dict]:
    """For each active branch: the ladder ahead of it with the chance of getting to each rung and the expected cost, from the pass
    rates learned so far. It shows the price of hope: a weak branch with three rungs to go is visibly cheap only while it keeps
    failing early."""
    rows = []
    for bid, b in sorted(state.branches.items()):
        if b.state not in ACTIVE_STATES:
            continue
        p, ahead = 1.0, []
        for i in range(stage_index(b.frontier), len(LADDER)):
            s = LADDER[i]
            c = (state.stats.mean_cost(s) or policy.rule(s).base_cpu_min) * state.cost_model.multiplier(s.value)
            ahead.append({"stage": s.value, "p_reach": p, "expected_cost": p * c})
            p *= state.stats.pass_rate(s)
        rows.append({"branch": bid, "p_finish": p, "expected_total_cpu_min": sum(a["expected_cost"] for a in ahead), "ahead": ahead})
    return rows


def gate_table(policy: LadderPolicy) -> list[dict]:
    """The ladder's thresholds as plain rows (for the daily research report and for review of what each rung demands)."""
    return [{"stage": r.stage.value, "tests": r.planned_tests, "cpu_min": r.base_cpu_min, "cap_cpu_min": r.max_cpu_min, "ram_gb": r.ram_gb,
             "real_data": r.real_data, "pass_t": r.pass_t, "min_power": r.min_power, "consistency": r.min_consistency,
             "units": r.min_units, "fresh": r.needs_fresh, "replication": r.needs_replication, "max_repeats": r.max_repeats}
            for r in policy.rules]


def recover_orphans(state: ManagerState, ledger: "C.ExperimentLedger", now) -> list[str]:
    """Jobs the manager believes are in flight but the compute ledger has never heard of (a crash between launch and submit, or a
    ledger restored from an older backup). The branch is freed and the event is recorded so `stalled_branches` can see repeats."""
    led = ledger.load()
    freed = []
    for b in state.branches.values():
        if b.in_flight and b.in_flight not in led:
            b.runs.append({"stage": b.frontier.value, "at": as_date(now).isoformat(), "action": "JOB_LOST", "job": b.in_flight, "why": "orphan"})
            b.in_flight = ""
            freed.append(b.branch_id)
    return sorted(freed)


def settle_finished(state: ManagerState, ledger: "C.ExperimentLedger", evidence_by_job: Mapping[str, StageEvidence], now,
                    policy: LadderPolicy | None = None) -> list[tuple[str, EscalationDecision]]:
    """Apply results for jobs the compute ledger marks DONE. `evidence_by_job` maps job key -> evidence (produced by the job's own
    output, read by the caller). A DONE job with no evidence is left in flight and reported by `orphan` checks, never guessed."""
    policy = policy or LadderPolicy()
    led = ledger.load()
    out = []
    for bid, b in sorted(state.branches.items()):
        if not b.in_flight or led.get(b.in_flight, {}).get("state") != C.DONE:
            continue
        ev = evidence_by_job.get(b.in_flight)
        if ev is None:
            continue
        out.append((bid, record_result(state, bid, ev, now, policy)))
    return out


def audit_report(state: ManagerState, policy: LadderPolicy, now) -> dict:
    """Everything a reviewer needs in one dict: audits, warnings, starvation, stalls, conservation, fairness."""
    return {"ladder_audit": audit_ladder(state, policy), "structure": validate_state(state, policy), "conservation": conservation_errors(state),
            "funnel_warnings": funnel_warnings(state), "starving": starving(state, now), "stalled": stalled_branches(state, now),
            "fairness": family_fairness(state), "queue_age": queue_age_report(state, now)}


# ------------------------------------------------------------------------------------------------ rounds and backlog
@dataclasses.dataclass(frozen=True)
class RoundRecord:
    day: str
    launched: int
    deferred: int
    cpu_min_planned: float
    results_applied: int
    parked: int
    active: int


class RoundLog:
    """Append-only per-tick summary (one JSON line per round) so the research report can show throughput over time and a crash
    can be traced to the last completed round. `summary` gives the trend that matters: are rounds still launching work?"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, state: ManagerState, res: StepResult, now) -> RoundRecord:
        rec = RoundRecord(as_date(now).isoformat(), len(res.launched), len(res.selection.deferred), float(sum(a.cpu_min for a in res.selection.allocations)),
                          len(res.decisions), sum(1 for _, d in res.decisions if d.action is Act.PARK),
                          sum(1 for b in state.branches.values() if b.state in ACTIVE_STATES))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(dataclasses.asdict(rec), sort_keys=True) + "\n")
        return rec

    def read(self) -> list[RoundRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(RoundRecord(**json.loads(line)))
            except (ValueError, TypeError):
                continue                                   # a torn last line from a crash is skipped, never fatal
        return out

    def summary(self, last: int = 14) -> dict:
        rs = self.read()[-last:]
        if not rs:
            return {"rounds": 0}
        idle = sum(1 for r in rs if r.launched == 0 and r.active > 0)
        return {"rounds": len(rs), "launched": sum(r.launched for r in rs), "deferred": sum(r.deferred for r in rs),
                "idle_rounds_with_work_waiting": idle, "cpu_min_planned": sum(r.cpu_min_planned for r in rs),
                "stuck": idle >= max(3, len(rs) // 2)}


def hypothesis_tree_report(state: ManagerState) -> list[str]:
    """Indented tree of branches by parent question (section 41 link), each with state and spend. Roots are branches whose parents
    are not themselves branches here."""
    by_q = {b.question_id: bid for bid, b in state.branches.items()}
    kids: dict[str, list[str]] = {}
    roots = []
    for bid, b in sorted(state.branches.items()):
        parents = [by_q[p] for p in b.parents if p in by_q]
        if parents:
            for p in parents:
                kids.setdefault(p, []).append(bid)
        else:
            roots.append(bid)
    lines: list[str] = []

    def walk(bid: str, depth: int, seen: frozenset) -> None:
        b = state.branches[bid]
        lines.append(f"{'  ' * depth}{bid} {b.state.value} {b.frontier.value} spent={b.total_spent:.1f} {b.text[:40]}")
        for c in kids.get(bid, []):
            if c not in seen:
                walk(c, depth + 1, seen | {bid})
    for r in roots:
        walk(r, 0, frozenset())
    return lines


def simulate_backlog(policy: LadderPolicy, n_branches: int, period_cpu_min: float, periods: int, seed: int,
                     p_real: float = 0.2, effect: float = 0.006, sd: float = 0.05) -> dict:
    """How long does a backlog of ideas take to clear at a given weekly compute, and how many real effects does it find? Runs the
    full step/record loop on synthetic hypotheses (known truth) with a fixed period budget, so budget advice can be checked against
    behaviour. Deterministic in `seed`."""
    truth = {f"f{i % 4}:h{i}": (effect if (i * 7919 + seed) % 100 < p_real * 100 else 0.0) for i in range(n_branches)}
    r = replay_schedule(policy, truth, seed, periods * 7, period_cpu_min, sd)
    found = [n for n in r["reached_end"] if truth[n] > 0]
    n_real = sum(1 for v in truth.values() if v > 0)
    return {"n_real": n_real, "found": len(found), "recall": len(found) / n_real if n_real else None,
            "false_positives": [n for n in r["reached_end"] if truth[n] == 0], "total_cpu_min": float(sum(r["spend"].values())),
            "cpu_min_per_find": (sum(r["spend"].values()) / len(found)) if found else None, "problems": r["problems"]}


# ------------------------------------------------------------------------------------------------ hand-off to the quality gate
@dataclasses.dataclass(frozen=True)
class GateDossier:
    """What the section-42 quality gate receives when a branch completes the ladder: the evidence at every rung, the total price paid
    and the flags that make the result harder to believe. The manager never promotes anything itself."""
    branch_id: str
    question_id: str
    family: str
    problem: str
    rungs: Mapping[str, Mapping[str, float]]
    total_cpu_min: float
    repeats: int
    audit_cleared: tuple[str, ...]
    looks: int
    flags: tuple[str, ...]


def dossier(state: ManagerState, branch_id: str) -> GateDossier:
    """Assemble the gate dossier for a branch that passed the integration rung. Raises LadderError for any other branch, so an
    unfinished hypothesis cannot be handed to the gate by mistake."""
    b = state.branches[branch_id]
    if Stage.INTEGRATION.value not in b.passed:
        raise LadderError(f"{branch_id} has not passed {Stage.INTEGRATION.value}; nothing to hand to the quality gate")
    flags = []
    if b.audit_cleared:
        flags.append("an implausibly strong result was audited and cleared: reproduce it independently")
    reps = sum(b.repeats.values())
    if reps >= 2:
        flags.append(f"needed {reps} repeats: the effect is near the detection limit")
    if b.looks > 20:
        flags.append(f"screened among {b.looks} similar ideas: multiplicity applied at rung 1 only")
    effects = [float(p["effect"]) for p in b.passed.values()]
    if len(effects) >= 2 and effects[-1] < 0.5 * effects[0]:
        flags.append("effect shrank by more than half between the first and last rung")
    return GateDossier(b.branch_id, b.question_id, b.family, b.problem.value, {k: dict(v) for k, v in sorted(b.passed.items())},
                       b.total_spent, reps, tuple(b.audit_cleared), b.looks, tuple(flags))


def completed_branches(state: ManagerState) -> list[str]:
    """Branches that finished the whole ladder, in id order (the input of the quality gate)."""
    return sorted(bid for bid, b in state.branches.items() if Stage.INTEGRATION.value in b.passed)
