"""The research controller that knows when to change priorities (contract C66 sections 50 and 51; also 0, 34, 38, 48; canon C64,
C66, C67). IMPLEMENTED - NOT VALIDATED (C63: code and unit tests only).

Section 51: the highest-level controller continuously compares current capability with desired capability, the largest remaining
uncertainty, the largest remaining loss source, the largest remaining prediction gap and the highest-value research opportunity,
and moves research share between eight phases:

    P1 can we reliably identify volatile stocks?                     P5 does the whole system transfer across years/stocks/regimes?
    P2 predictable volatility versus unknowable external shocks      P6 can it avoid catastrophic losses?
    P3 direction among predicted movers                              P7 can it keep discovering new knowledge?
    P4 direction at high accuracy (80%) at meaningful coverage       P8 can it improve the process that discovers knowledge?

Mechanism (every step is data, never prose):
  * a CapabilityReading per phase (metric, value, standard error, sample, the real date its evidence matured) is turned into a
    normalised capability in [0, 1] against the phase target; an unmeasured phase is maximally uncertain, never assumed fine;
  * a phase is SATISFIED only after `confirm_n` consecutive readings whose lower bound clears the target, and REGRESSED when a
    satisfied phase falls back below it (section 51: "may temporarily revisit earlier phases when evidence shows a regression");
  * the progression is enforced as gates: direction (P3) waits for volatility (P1, section 0 "the volatility problem comes FIRST"),
    the 80% question (P4) waits for a direction signal; a gated phase keeps a small probe share so the gate can still open;
  * realised research yield per compute (value accounting) is shrunk towards a prior; a phase whose yield is negligible after
    enough spend is STALLED and loses share (section 50: "direction research is producing negligible improvement ... redirect");
  * expected value = objective weight x capability gap x (1 + uncertainty) x yield x gate x regression boost (+ loss share for P6);
  * the allocation is floored/capped (continuous discovery and the research process never go to zero) and smoothed, except that a
    regression moves immediately;
  * section-50 statements are emitted as records ("VOLATILITY is the highest-value unresolved problem, allocate 43%"), and so is
    the honest one: "No reliable 80% directional region has been found."

The allocation feeds the rest of the brain: `objective_weights` (engine.research.priority), `problem_weight`
(engine.research.compute_manager.LadderPolicy) and `directives` (engine.research.brain_health.Directives, honoured by
engine.research.diversity). Everything lives in MATURED_RESEARCH_STATE; a reading dated at/after `now` is a FirewallBreach.
Public entry: `step(state, now, readings, yields, loss_sources)`."""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.research.core import FirewallBreach, Namespace, Problem, _StrEnum, as_date, require_past, stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
NAMESPACE = Namespace.MATURED_RESEARCH
EPS = 1e-12


class Phase(_StrEnum):
    VOLATILITY = "P1_VOLATILITY"
    KNOWABILITY = "P2_PREDICTABLE_VS_UNKNOWABLE"
    DIRECTION = "P3_DIRECTION"
    DIRECTION_AT_COVERAGE = "P4_DIRECTION_AT_COVERAGE"
    TRANSFER = "P5_TRANSFER"
    CATASTROPHIC_LOSS = "P6_CATASTROPHIC_LOSS"
    DISCOVERY = "P7_CONTINUOUS_DISCOVERY"
    DISCOVERY_PROCESS = "P8_DISCOVERY_PROCESS"


PHASES: tuple[Phase, ...] = tuple(Phase)

PHASE_QUESTION = {
    Phase.VOLATILITY: "Can we reliably identify volatile stocks?",
    Phase.KNOWABILITY: "Can we distinguish predictable volatility from unknowable external shocks?",
    Phase.DIRECTION: "Can we determine direction among predicted movers?",
    Phase.DIRECTION_AT_COVERAGE: "Can direction reach high accuracy at meaningful coverage?",
    Phase.TRANSFER: "Can the entire system transfer across years/stocks/regimes?",
    Phase.CATASTROPHIC_LOSS: "Can it avoid catastrophic losses?",
    Phase.DISCOVERY: "Can it continuously discover new knowledge?",
    Phase.DISCOVERY_PROCESS: "Can it continuously improve the process by which it discovers knowledge?",
}

PHASE_NOUN = {
    Phase.VOLATILITY: "volatility prediction", Phase.KNOWABILITY: "separating predictable volatility from external shocks",
    Phase.DIRECTION: "directional discrimination among predicted movers", Phase.DIRECTION_AT_COVERAGE: "high-accuracy direction at coverage",
    Phase.TRANSFER: "transfer across years, stocks and regimes", Phase.CATASTROPHIC_LOSS: "loss-regime research",
    Phase.DISCOVERY: "continuous discovery", Phase.DISCOVERY_PROCESS: "improving the discovery process",
}

# how a phase's share is spent across the section-34 problems (a documented prior, not a tuned value)
PHASE_PROBLEMS: Mapping[Phase, Mapping[Problem, float]] = {
    Phase.VOLATILITY: {Problem.VOLATILITY: 1.0},
    Phase.KNOWABILITY: {Problem.VOLATILITY: 0.5, Problem.DATA_QUALITY: 0.5},
    Phase.DIRECTION: {Problem.DIRECTION: 1.0},
    Phase.DIRECTION_AT_COVERAGE: {Problem.DIRECTION: 0.5, Problem.COVERAGE: 0.5},
    Phase.TRANSFER: {Problem.CONSISTENCY: 1.0},
    Phase.CATASTROPHIC_LOSS: {Problem.LOSS_AVOIDANCE: 1.0},
    Phase.DISCOVERY: {Problem.VOLATILITY: 0.4, Problem.LOSS_AVOIDANCE: 0.3, Problem.DIRECTION: 0.3},
    Phase.DISCOVERY_PROCESS: {Problem.RESEARCH_PROCESS: 0.7, Problem.EFFICIENCY: 0.3},
}

# section-38 areas (engine.research.brain_health.Area values) each phase's share supports
PHASE_AREAS: Mapping[Phase, tuple[str, ...]] = {
    Phase.VOLATILITY: ("VOLATILITY",), Phase.KNOWABILITY: ("UNCERTAIN", "DATA_QUALITY"), Phase.DIRECTION: ("DIRECTION",),
    Phase.DIRECTION_AT_COVERAGE: ("DIRECTION",), Phase.TRANSFER: ("REGIME",), Phase.CATASTROPHIC_LOSS: ("RISK", "PATTERN_BREAK"),
    Phase.DISCOVERY: ("NEW_REPRESENTATION", "UNCERTAIN"), Phase.DISCOVERY_PROCESS: ("FAILED_NEW_HYPOTHESIS", "KNOWN_PROMISING"),
}


class PhaseStatus(_StrEnum):
    UNMEASURED = "UNMEASURED"        # no reading yet: maximal uncertainty
    OPEN = "OPEN"                    # measured, below target
    SATISFIED = "SATISFIED"          # target cleared on confirm_n consecutive readings (lower bound)
    REGRESSED = "REGRESSED"          # was satisfied, has fallen back: revisit now
    BLOCKED = "BLOCKED"              # a prerequisite phase has not earned it (probe share only)
    STALLED = "STALLED"              # enough spend, negligible realised yield: redirect


class StatementKind(_StrEnum):
    ALLOCATE = "ALLOCATE"
    SHIFT = "SHIFT"
    REDIRECT = "REDIRECT"
    REGRESSION = "REGRESSION"
    GATE = "GATE"
    NO_EIGHTY = "NO_EIGHTY"
    UNMEASURED = "UNMEASURED"


# ---------------------------------------------------------------------------------------------------------------- targets
@dataclasses.dataclass(frozen=True)
class PhaseTarget:
    """What 'capable' means for one phase: the metric, the desired value and the value that means 'no capability at all'. For a
    lower-is-better metric the floor is the WORST value (e.g. a 10% catastrophic-loss rate)."""
    phase: Phase
    metric: str
    desired: float
    floor: float
    higher_is_better: bool = True

    def validate(self) -> list[str]:
        errs = []
        if not self.metric:
            errs.append(f"{self.phase}: target without a metric")
        if not (math.isfinite(self.desired) and math.isfinite(self.floor)) or self.desired == self.floor:
            errs.append(f"{self.phase}: desired and floor must be finite and different")
        elif self.higher_is_better != (self.desired > self.floor):
            errs.append(f"{self.phase}: higher_is_better disagrees with desired vs floor")
        return errs

    def capability(self, value: float) -> float:
        """Normalised capability in [0, 1]: 0 = the floor (no capability), 1 = the desired level."""
        c = (value - self.floor) / (self.desired - self.floor)
        return float(min(1.0, max(0.0, c)))

    def scale(self) -> float:
        return abs(self.desired - self.floor)


def default_targets() -> dict[Phase, PhaseTarget]:
    """Section-0/51 targets. P4's 0.80 is the owner's long-term aim at coverage; it is a target to measure against, never a result to
    manufacture (section 0). P6's floor of 10% catastrophic losses is the level treated as 'no loss control at all'."""
    T = PhaseTarget
    return {
        Phase.VOLATILITY: T(Phase.VOLATILITY, "vol_auc", 0.70, 0.50),
        Phase.KNOWABILITY: T(Phase.KNOWABILITY, "explained_mover_share", 0.60, 0.0),
        Phase.DIRECTION: T(Phase.DIRECTION, "direction_acc", 0.60, 0.50),
        Phase.DIRECTION_AT_COVERAGE: T(Phase.DIRECTION_AT_COVERAGE, "acc_at_coverage", 0.80, 0.50),
        Phase.TRANSFER: T(Phase.TRANSFER, "transfer_retention", 0.80, 0.0),
        Phase.CATASTROPHIC_LOSS: T(Phase.CATASTROPHIC_LOSS, "catastrophic_rate", 0.01, 0.10, higher_is_better=False),
        Phase.DISCOVERY: T(Phase.DISCOVERY, "discovery_rate", 1.0, 0.0),
        Phase.DISCOVERY_PROCESS: T(Phase.DISCOVERY_PROCESS, "value_per_compute_trend", 0.5, 0.0),
    }


# ---------------------------------------------------------------------------------------------------------------- inputs
@dataclasses.dataclass(frozen=True)
class CapabilityReading:
    """One matured measurement of a phase's capability. `as_of` is the REAL date its newest evidence matured (trusted side); `se` is
    the standard error of `value` (None = unknown, treated as wide)."""
    phase: Phase
    metric: str
    value: float
    as_of: str
    se: float | None = None
    n: int = 0
    source: str = ""

    def validate(self) -> list[str]:
        errs = []
        if not math.isfinite(self.value):
            errs.append(f"{self.phase}/{self.metric}: value not finite")
        if self.se is not None and (not math.isfinite(self.se) or self.se < 0):
            errs.append(f"{self.phase}/{self.metric}: se must be finite and >= 0")
        if self.n < 0:
            errs.append(f"{self.phase}/{self.metric}: negative n")
        try:
            as_date(self.as_of)
        except (TypeError, ValueError):
            errs.append(f"{self.phase}/{self.metric}: unparseable as_of {self.as_of!r}")
        return errs


@dataclasses.dataclass(frozen=True)
class YieldRecord:
    """Realised research value of finished work in a phase (from value accounting): value in the phase's decision units, cost in
    CPU-minutes, the real date it was known."""
    phase: Phase
    value: float
    cost_minutes: float
    as_of: str
    transferable: bool | None = None       # None = not tested; False = did not survive out of sample (value then counts as 0)

    def validate(self) -> list[str]:
        errs = []
        if not math.isfinite(self.value):
            errs.append(f"yield {self.phase}: value not finite")
        if not (math.isfinite(self.cost_minutes) and self.cost_minutes > 0):
            errs.append(f"yield {self.phase}: cost must be positive")
        return errs

    @property
    def effective_value(self) -> float:
        """Section 43/48: a finding that did not transfer delivered nothing, whatever it looked like in sample."""
        return 0.0 if self.transferable is False else max(0.0, float(self.value))


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    order_weight: Mapping[Phase, float] = dataclasses.field(default_factory=lambda: {
        Phase.VOLATILITY: 1.0, Phase.KNOWABILITY: 0.6, Phase.DIRECTION: 0.7, Phase.DIRECTION_AT_COVERAGE: 0.5,
        Phase.TRANSFER: 0.6, Phase.CATASTROPHIC_LOSS: 0.9, Phase.DISCOVERY: 0.35, Phase.DISCOVERY_PROCESS: 0.25})
    floors: Mapping[Phase, float] = dataclasses.field(default_factory=lambda: {
        Phase.DISCOVERY: 0.05, Phase.DISCOVERY_PROCESS: 0.03, Phase.CATASTROPHIC_LOSS: 0.05})
    probe_share: float = 0.02              # what a BLOCKED phase keeps, so the check that opens its gate can still run
    cap: float = 0.60                      # no single phase may take more
    measure_cap: float = 0.10              # an UNMEASURED phase earns a measurement budget, not a research programme
    unmeasured_gap: float = 0.5            # prior gap of a phase never measured (unknown, not assumed to be the worst)
    uncertainty_weight: float = 0.5
    smoothing: float = 0.5                 # weight of the new target share each step (1 = no memory)
    confirm_n: int = 2                     # readings needed to call a phase SATISFIED (or REGRESSED)
    satisfied_level: float = 0.95          # capability lower bound that counts as meeting the target
    regress_margin: float = 0.15           # a satisfied phase whose capability falls below satisfied_level - margin has regressed
    regression_boost: float = 2.0
    gate_level: float = 0.5                # P1 capability that opens P3 (AUC 0.60 with default targets)
    direction_signal_level: float = 0.2    # P3 capability lower bound that opens P4
    yield_prior: float = 0.002             # prior realised value per CPU-minute
    yield_strength_min: float = 120.0      # pseudo-minutes of prior
    stall_min_minutes: float = 300.0       # spend before a phase can be called STALLED
    stall_ratio: float = 0.2               # realised yield below this share of the prior (upper bound) = negligible
    stalled_multiplier: float = 0.25
    loss_weight: float = 0.5               # how strongly the largest loss source's share lifts P6
    z: float = 1.645                       # one-sided 95% bound for 'lower bound clears the target'
    stale_days: int = 120                  # a reading older than this is treated as unmeasured again
    min_eighty_n: int = 200                # evidence needed before 'no 80% region' may be stated

    def validate(self) -> list[str]:
        errs = []
        for p in PHASES:
            if self.order_weight.get(p, 0.0) <= 0:
                errs.append(f"order_weight[{p}] must be positive")
        if not 0 < self.cap <= 1 or not 0 <= self.probe_share < self.cap or not self.probe_share <= self.measure_cap <= self.cap:
            errs.append("cap in (0,1], probe_share in [0,cap) and measure_cap in [probe_share, cap] required")
        if not 0 < self.unmeasured_gap <= 1:
            errs.append("unmeasured_gap in (0,1] required")
        if sum(self.floors.values()) + self.probe_share * len(PHASES) >= 1.0:
            errs.append("floors plus probe shares leave nothing to allocate")
        if not 0 < self.smoothing <= 1 or self.confirm_n < 1 or not 0 < self.satisfied_level <= 1:
            errs.append("smoothing in (0,1], confirm_n >= 1, satisfied_level in (0,1] required")
        if self.yield_prior <= 0 or self.yield_strength_min <= 0 or not 0 < self.stall_ratio < 1:
            errs.append("yield prior/strength positive and stall_ratio in (0,1) required")
        return errs


# ---------------------------------------------------------------------------------------------------------------- assessment
@dataclasses.dataclass(frozen=True)
class PhaseAssessment:
    phase: Phase
    status: PhaseStatus
    value: float | None
    capability: float | None
    capability_lo: float | None
    gap: float
    uncertainty: float
    yield_rate: float
    yield_hi: float
    spent_minutes: float
    gate_open: bool
    expected_value: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["phase"], d["status"] = self.phase.value, self.status.value
        return d


@dataclasses.dataclass(frozen=True)
class Statement:
    """A section-50 sentence as data. `share` is the phase's new share of research compute."""
    kind: StatementKind
    phase: Phase
    share: float
    text: str
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "phase": self.phase.value, "share": round(self.share, 6), "text": self.text,
                "evidence": dict(self.evidence)}


@dataclasses.dataclass(frozen=True)
class Situation:
    """The six section-51 comparisons, as of one step."""
    current: Mapping[str, float | None]
    desired: Mapping[str, float]
    largest_uncertainty: str
    largest_loss_source: str
    largest_loss_share: float
    largest_prediction_gap: str
    highest_value: str


@dataclasses.dataclass(frozen=True)
class ControllerDecision:
    now: str
    shares: Mapping[Phase, float]
    problem_shares: Mapping[Problem, float]
    assessments: tuple[PhaseAssessment, ...]
    situation: Situation
    statements: tuple[Statement, ...]
    top: Phase
    decision_id: str

    def share(self, p: Phase) -> float:
        return float(self.shares.get(p, 0.0))

    def to_dict(self) -> dict:
        return {"now": self.now, "decision_id": self.decision_id, "top": self.top.value,
                "shares": {p.value: round(v, 6) for p, v in self.shares.items()},
                "problem_shares": {p.value: round(v, 6) for p, v in self.problem_shares.items()},
                "assessments": [a.to_dict() for a in self.assessments], "statements": [s.to_dict() for s in self.statements],
                "situation": dataclasses.asdict(self.situation), "label": LABEL}


# ---------------------------------------------------------------------------------------------------------------- state
@dataclasses.dataclass
class ControllerState:
    cfg: ControllerConfig = dataclasses.field(default_factory=ControllerConfig)
    targets: dict = dataclasses.field(default_factory=default_targets)
    readings: dict = dataclasses.field(default_factory=dict)       # phase value -> list of CapabilityReading (arrival order)
    yields: list = dataclasses.field(default_factory=list)         # YieldRecord
    satisfied: dict = dataclasses.field(default_factory=dict)      # phase value -> date first satisfied
    regressions: list = dataclasses.field(default_factory=list)    # (date, phase value, capability before, capability now)
    shares: dict = dataclasses.field(default_factory=dict)         # phase value -> last share
    history: list = dataclasses.field(default_factory=list)        # decision dicts
    last_now: str = ""
    namespace: Namespace = NAMESPACE

    def validate(self) -> list[str]:
        errs = list(self.cfg.validate())
        for p in PHASES:
            t = self.targets.get(p)
            if t is None:
                errs.append(f"no target for {p}")
            else:
                errs += t.validate()
        return errs


def new_state(cfg: ControllerConfig | None = None, targets: Mapping[Phase, PhaseTarget] | None = None) -> ControllerState:
    st = ControllerState(cfg=cfg or ControllerConfig(), targets=dict(targets or default_targets()))
    bad = st.validate()
    if bad:
        raise ValueError("invalid controller configuration: " + "; ".join(bad))
    return st


def ingest(state: ControllerState, now, readings: Iterable[CapabilityReading] = (), yields: Iterable[YieldRecord] = ()) -> tuple[int, int]:
    """File new readings and yields. Every one must have matured strictly before `now` (FirewallBreach otherwise: a capability measured
    on outcomes that have not happened yet is a leak). Exact repeats (same phase, metric, as_of, value) are ignored so a replayed log is
    harmless. Returns (readings taken, yields taken)."""
    nr = ny = 0
    for r in sorted(readings, key=lambda r: (str(r.as_of), r.phase.value, r.metric)):
        bad = r.validate()
        if bad:
            raise ValueError("; ".join(bad))
        require_past(r.as_of, now, f"capability reading {r.phase.value}/{r.metric}")
        tgt = state.targets[r.phase]
        if r.metric != tgt.metric:
            raise ValueError(f"reading metric {r.metric!r} does not match the {r.phase.value} target metric {tgt.metric!r}")
        rows = state.readings.setdefault(r.phase.value, [])
        if any(x.as_of == r.as_of and x.value == r.value and x.metric == r.metric for x in rows):
            continue
        rows.append(r)
        nr += 1
    seen = {(y.phase, y.as_of, y.value, y.cost_minutes) for y in state.yields}
    for y in sorted(yields, key=lambda y: (str(y.as_of), y.phase.value)):
        bad = y.validate()
        if bad:
            raise ValueError("; ".join(bad))
        require_past(y.as_of, now, f"yield record {y.phase.value}")
        k = (y.phase, y.as_of, y.value, y.cost_minutes)
        if k in seen:
            continue
        seen.add(k)
        state.yields.append(y)
        ny += 1
    return nr, ny


def _fresh(rows: Sequence[CapabilityReading], now, stale_days: int) -> list[CapabilityReading]:
    d = as_date(now)
    return [r for r in rows if (d - as_date(r.as_of)).days <= stale_days]


def _cap_bounds(t: PhaseTarget, r: CapabilityReading, z: float) -> tuple[float, float, float]:
    """(capability, lower bound, upper bound) of one reading; the bound moves the raw value by z standard errors towards 'worse'."""
    se = r.se if r.se is not None else t.scale()
    worse = r.value - z * se if t.higher_is_better else r.value + z * se
    better = r.value + z * se if t.higher_is_better else r.value - z * se
    return t.capability(r.value), t.capability(worse), t.capability(better)


def phase_yield(state: ControllerState, phase: Phase, now) -> tuple[float, float, float]:
    """(shrunken realised value per CPU-minute, its rough upper bound, minutes spent). Records dated at/after now are not read."""
    cfg = state.cfg
    rows = [y for y in state.yields if y.phase == phase and as_date(y.as_of) < as_date(now)]
    spent = float(sum(y.cost_minutes for y in rows))
    val = float(sum(y.effective_value for y in rows))
    k = cfg.yield_strength_min
    rate = (val + k * cfg.yield_prior) / (spent + k)
    per = np.array([y.effective_value / y.cost_minutes for y in rows]) if rows else np.array([])
    sd = float(per.std(ddof=1)) if len(per) > 1 else cfg.yield_prior
    hi = rate + cfg.z * sd / math.sqrt(max(len(per), 1))
    return rate, hi, spent


def _gates(caps: Mapping[Phase, tuple[float | None, float | None]], cfg: ControllerConfig) -> dict[Phase, tuple[bool, str]]:
    """The section-51 progression as gates. caps: phase -> (capability, lower bound)."""
    def cap(p):
        return caps.get(p, (None, None))[0]

    def lo(p):
        return caps.get(p, (None, None))[1]
    out = {p: (True, "") for p in PHASES}
    c1 = cap(Phase.VOLATILITY)
    if c1 is None or c1 < cfg.gate_level:
        out[Phase.DIRECTION] = (False, "direction waits for volatility evidence (P1 capability "
                                       f"{'unmeasured' if c1 is None else f'{c1:.2f}'} < {cfg.gate_level})")
    l3 = lo(Phase.DIRECTION)
    if not out[Phase.DIRECTION][0]:
        out[Phase.DIRECTION_AT_COVERAGE] = (False, "the 80% question waits for the direction gate")
    elif l3 is None or l3 < cfg.direction_signal_level:
        out[Phase.DIRECTION_AT_COVERAGE] = (False, "no direction signal among predicted movers yet (P3 lower bound "
                                                   f"{'unmeasured' if l3 is None else f'{l3:.2f}'} < {cfg.direction_signal_level})")
    if c1 is None:
        out[Phase.TRANSFER] = (False, "nothing to transfer before volatility has been measured")
    return out


def assess(state: ControllerState, now, loss_share: float = 0.0) -> list[PhaseAssessment]:
    """Current vs desired capability for every phase, with status, gate, uncertainty, yield and expected value."""
    cfg = state.cfg
    caps: dict[Phase, tuple[float | None, float | None]] = {}
    per: dict[Phase, dict] = {}
    for p in PHASES:
        t = state.targets[p]
        rows = _fresh(state.readings.get(p.value, []), now, cfg.stale_days)
        if not rows:
            caps[p] = (None, None)
            per[p] = {"value": None, "cap": None, "lo": None, "unc": 1.0, "rows": []}
            continue
        last = rows[-1]
        c, lo, hi = _cap_bounds(t, last, cfg.z)
        caps[p] = (c, lo)
        per[p] = {"value": last.value, "cap": c, "lo": lo, "unc": float(min(1.0, max(0.0, hi - lo) / 2.0)), "rows": rows}
    gates = _gates(caps, cfg)
    out = []
    for p in PHASES:
        t = state.targets[p]
        d = per[p]
        reasons: list[str] = []
        rate, hi, spent = phase_yield(state, p, now)
        gate_open, gate_why = gates[p]
        if d["cap"] is None:
            status, gap = PhaseStatus.UNMEASURED, cfg.unmeasured_gap
            reasons.append("never measured (or the reading is stale): treated as maximally uncertain")
        else:
            gap = 1.0 - d["cap"]
            confirmed = [(_cap_bounds(t, r, cfg.z)[1]) for r in d["rows"][-cfg.confirm_n:]]
            meets = len(confirmed) >= cfg.confirm_n and all(x >= cfg.satisfied_level for x in confirmed)
            below = [(_cap_bounds(t, r, cfg.z)[0]) for r in d["rows"][-cfg.confirm_n:]]
            fell = len(below) >= cfg.confirm_n and all(x < cfg.satisfied_level - cfg.regress_margin for x in below)
            if p.value in state.satisfied and fell:
                status = PhaseStatus.REGRESSED
                reasons.append(f"was satisfied since {state.satisfied[p.value]}; capability now {d['cap']:.2f}")
            elif meets:
                status = PhaseStatus.SATISFIED
            else:
                status = PhaseStatus.OPEN
        if not gate_open:
            status = PhaseStatus.BLOCKED
            reasons.append(gate_why)
        stalled = spent >= cfg.stall_min_minutes and hi < cfg.stall_ratio * cfg.yield_prior
        if stalled and status in (PhaseStatus.OPEN, PhaseStatus.UNMEASURED):
            status = PhaseStatus.STALLED
            reasons.append(f"{spent:.0f} CPU-min spent, realised value per minute {rate:.2e} (upper {hi:.2e}) is negligible")
        w = cfg.order_weight[p]
        ev = w * gap * (1.0 + cfg.uncertainty_weight * d["unc"])
        ev *= (rate / cfg.yield_prior) ** 0.5 if spent > 0 else 1.0
        if status == PhaseStatus.STALLED:
            ev *= cfg.stalled_multiplier
        if status == PhaseStatus.REGRESSED:
            ev *= cfg.regression_boost
        if status == PhaseStatus.BLOCKED:
            ev = 0.0
        if status == PhaseStatus.SATISFIED:
            ev *= 0.25                                          # maintenance only: keep measuring so a regression is seen
        if p == Phase.CATASTROPHIC_LOSS:
            ev += cfg.loss_weight * w * float(min(1.0, max(0.0, loss_share)))
        out.append(PhaseAssessment(p, status, d["value"], d["cap"], d["lo"], float(gap), float(d["unc"]), float(rate), float(hi), float(spent),
                                   bool(gate_open), float(ev), tuple(reasons)))
    return out


# ---------------------------------------------------------------------------------------------------------------- allocation
def _bounded_normalise(raw: Mapping[Phase, float], lower: Mapping[Phase, float], upper: Mapping[Phase, float]) -> dict[Phase, float]:
    """Shares proportional to `raw`, summing to 1, each inside [lower, upper] (water-filling: clip, freeze, renormalise the rest)."""
    free = set(PHASES)
    out: dict[Phase, float] = {}
    for _ in range(3 * len(PHASES)):
        budget = 1.0 - sum(out.values())
        tot = sum(max(raw[p], 0.0) for p in free)
        prop = {p: (budget * max(raw[p], 0.0) / tot if tot > EPS else budget / len(free)) for p in free}
        viol = {p: (lower[p] if prop[p] < lower[p] else upper[p]) for p in free if prop[p] < lower[p] - 1e-12 or prop[p] > upper[p] + 1e-12}
        if not viol:
            out.update(prop)
            break
        for p, v in viol.items():
            out[p] = v
            free.discard(p)
        if not free:
            break
    s = sum(out.values())
    return {p: out.get(p, 0.0) / s for p in PHASES} if s > 0 else {p: 1.0 / len(PHASES) for p in PHASES}


def allocate(state: ControllerState, assessments: Sequence[PhaseAssessment]) -> dict[Phase, float]:
    cfg = state.cfg
    raw = {a.phase: a.expected_value for a in assessments}
    lower = {p: cfg.floors.get(p, 0.0) for p in PHASES}
    upper = {p: cfg.cap for p in PHASES}
    for a in assessments:
        if a.status == PhaseStatus.BLOCKED:
            lower[a.phase] = upper[a.phase] = cfg.probe_share
        elif a.status == PhaseStatus.UNMEASURED:
            lower[a.phase] = max(lower[a.phase], cfg.probe_share)
            upper[a.phase] = max(lower[a.phase], cfg.measure_cap)
        else:
            lower[a.phase] = max(lower[a.phase], cfg.probe_share)
    target = _bounded_normalise(raw, lower, upper)
    prev = {p: float(state.shares.get(p.value, 0.0)) for p in PHASES}
    if not any(prev.values()):
        return target
    status = {a.phase: a.status for a in assessments}
    mixed = {}
    for p in PHASES:
        a = 1.0 if status[p] in (PhaseStatus.REGRESSED, PhaseStatus.BLOCKED) else cfg.smoothing
        mixed[p] = (1 - a) * prev[p] + a * target[p]
    return _bounded_normalise(mixed, lower, upper)


def problem_shares(shares: Mapping[Phase, float]) -> dict[Problem, float]:
    out: dict[Problem, float] = {}
    for p, s in shares.items():
        for prob, f in PHASE_PROBLEMS[p].items():
            out[prob] = out.get(prob, 0.0) + s * f
    tot = sum(out.values()) or 1.0
    return {k: v / tot for k, v in sorted(out.items(), key=lambda kv: kv[0].value)}


# ---------------------------------------------------------------------------------------------------------------- statements
def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def statements(state: ControllerState, now, assessments: Sequence[PhaseAssessment], shares: Mapping[Phase, float],
               loss_source: str = "", loss_share: float = 0.0) -> list[Statement]:
    """The section-50 sentences, generated from the numbers (never typed by hand)."""
    by = {a.phase: a for a in assessments}
    open_ = [a for a in assessments if a.status not in (PhaseStatus.SATISFIED, PhaseStatus.BLOCKED)]
    top = max(PHASES, key=lambda p: (shares[p], -PHASES.index(p)))
    out: list[Statement] = []
    prev_top = state.history[-1]["top"] if state.history else ""
    a = by[top]
    ev = {"expected_value": round(a.expected_value, 6), "gap": round(a.gap, 4), "capability": a.capability, "status": a.status.value}
    unresolved = "unresolved" if a.status != PhaseStatus.SATISFIED else "maintained"
    out.append(Statement(StatementKind.ALLOCATE, top, shares[top],
                         f"{PHASE_NOUN[top].capitalize()} is currently the highest-value {unresolved} problem, so allocate "
                         f"{_pct(shares[top])} of research compute there.", ev))
    if prev_top and prev_top != top.value:
        old = Phase(prev_top)
        oa = by[old]
        why = {PhaseStatus.SATISFIED: "has improved substantially", PhaseStatus.STALLED: "is producing negligible transferable improvement",
               PhaseStatus.BLOCKED: "is gated", PhaseStatus.REGRESSED: "regressed"}.get(oa.status, "now has a smaller remaining gap")
        out.append(Statement(StatementKind.SHIFT, top, shares[top],
                             f"{PHASE_NOUN[old].capitalize()} {why}. The largest remaining expected decision value is now "
                             f"{PHASE_NOUN[top]}, so shift resources ({_pct(float(state.shares.get(old.value, 0.0)))} -> "
                             f"{_pct(shares[old])} for {old.value}).",
                             {"from": old.value, "from_status": oa.status.value, "from_capability": oa.capability}))
    for s in (x for x in assessments if x.status == PhaseStatus.STALLED):
        best = max((x for x in open_ if x.phase != s.phase and x.status != PhaseStatus.STALLED),
                   key=lambda x: x.expected_value, default=None)
        if best is not None:
            out.append(Statement(StatementKind.REDIRECT, s.phase, shares[s.phase],
                                 f"{PHASE_NOUN[s.phase].capitalize()} research is currently producing negligible transferable improvement, "
                                 f"while {PHASE_NOUN[best.phase]} has high expected value. Redirect compute.",
                                 {"spent_minutes": s.spent_minutes, "yield_per_minute": s.yield_rate, "to": best.phase.value}))
    for s in (x for x in assessments if x.status == PhaseStatus.REGRESSED):
        out.append(Statement(StatementKind.REGRESSION, s.phase, shares[s.phase],
                             f"{PHASE_NOUN[s.phase].capitalize()} regressed (capability {s.capability:.2f} after being satisfied); "
                             f"revisiting it with {_pct(shares[s.phase])} of compute.", {"capability": s.capability}))
    for s in (x for x in assessments if x.status == PhaseStatus.BLOCKED):
        out.append(Statement(StatementKind.GATE, s.phase, shares[s.phase], f"{PHASE_NOUN[s.phase].capitalize()} is gated: "
                             + "; ".join(s.reasons[-1:]), {}))
    p4 = state.readings.get(Phase.DIRECTION_AT_COVERAGE.value, [])
    if p4:
        r = p4[-1]
        t = state.targets[Phase.DIRECTION_AT_COVERAGE]
        _, _, hi = _cap_bounds(t, r, state.cfg.z)
        if r.n >= state.cfg.min_eighty_n and hi < 1.0:
            out.append(Statement(StatementKind.NO_EIGHTY, Phase.DIRECTION_AT_COVERAGE, shares[Phase.DIRECTION_AT_COVERAGE],
                                 "No reliable 80% directional region has been found.",
                                 {"best_accuracy": r.value, "se": r.se, "n": r.n, "as_of": r.as_of}))
    unm = [x.phase.value for x in assessments if x.status == PhaseStatus.UNMEASURED]
    if unm:
        out.append(Statement(StatementKind.UNMEASURED, top, shares[top], "Unmeasured (maximal uncertainty): " + ", ".join(unm), {}))
    if loss_source:
        out[0] = dataclasses.replace(out[0], evidence={**out[0].evidence, "largest_loss_source": loss_source,
                                                       "largest_loss_share": round(loss_share, 4)})
    return out


def situation(state: ControllerState, assessments: Sequence[PhaseAssessment], loss_sources: Mapping[str, float]) -> Situation:
    by = {a.phase: a for a in assessments}
    unc = max(PHASES, key=lambda p: (by[p].uncertainty, -PHASES.index(p)))
    pred = [Phase.VOLATILITY, Phase.DIRECTION, Phase.DIRECTION_AT_COVERAGE]
    gap = max(pred, key=lambda p: (by[p].gap if by[p].status != PhaseStatus.BLOCKED else -1.0, -PHASES.index(p)))
    hv = max(PHASES, key=lambda p: (by[p].expected_value, -PHASES.index(p)))
    ls = max(sorted(loss_sources.items()), key=lambda kv: kv[1], default=("", 0.0))
    return Situation({p.value: by[p].capability for p in PHASES}, {p.value: state.targets[p].desired for p in PHASES},
                     unc.value, ls[0], float(ls[1]), gap.value, hv.value)


# ---------------------------------------------------------------------------------------------------------------- public entry
def step(state: ControllerState, now, readings: Iterable[CapabilityReading] = (), yields: Iterable[YieldRecord] = (),
         loss_sources: Mapping[str, float] | None = None) -> ControllerDecision:
    """PUBLIC ENTRY. File the matured readings/yields, assess every phase against its target, allocate research share, record
    satisfied/regressed transitions and return the decision with its statements. `loss_sources` maps a loss source (a regime, a
    failure cause, a pattern family; identity free) to its share of realised losses."""
    if state.last_now and as_date(now) < as_date(state.last_now):
        raise FirewallBreach(f"controller clock went backwards: {now} < {state.last_now}")
    ingest(state, now, readings, yields)
    ls = dict(loss_sources or {})
    for k, v in ls.items():
        if not (isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"loss share for {k!r} must be in [0,1]")
    top_loss = max(sorted(ls.items()), key=lambda kv: kv[1], default=("", 0.0))
    ass = assess(state, now, top_loss[1])
    d = as_date(now).isoformat()
    for a in ass:
        if a.status == PhaseStatus.SATISFIED and a.phase.value not in state.satisfied:
            state.satisfied[a.phase.value] = d
        if a.status == PhaseStatus.REGRESSED:
            state.regressions.append((d, a.phase.value, 1.0, a.capability))
            state.satisfied.pop(a.phase.value, None)
    shares = allocate(state, ass)
    stm = statements(state, now, ass, shares, top_loss[0], top_loss[1])
    sit = situation(state, ass, ls)
    top = max(PHASES, key=lambda p: (shares[p], -PHASES.index(p)))
    did = "CD" + stable_hash({"now": d, "s": {p.value: round(v, 9) for p, v in shares.items()}}, 12)
    dec = ControllerDecision(d, shares, problem_shares(shares), tuple(ass), sit, tuple(stm), top, did)
    state.shares = {p.value: v for p, v in shares.items()}
    state.history.append({"now": d, "top": top.value, "shares": dict(state.shares), "id": did})
    state.last_now = d
    return dec


# ---------------------------------------------------------------------------------------------------------------- feeding the brain
def objective_weights(decision: ControllerDecision, base=None, strength: float = 1.0):
    """engine.research.priority.ObjectiveWeights moved towards the controller's problem shares: each weight is multiplied by
    (share / uniform share) ** strength, bounded to [1/3, 3] of the base so one decision cannot invert the hierarchy."""
    from engine.research.priority import ObjectiveWeights
    base = base or ObjectiveWeights()
    uni = 1.0 / max(len(decision.problem_shares), 1)
    new = dict(base.w)
    for prob in list(Problem):
        s = decision.problem_shares.get(prob, 0.0)
        mult = min(3.0, max(1.0 / 3.0, ((s + 0.01) / (uni + 0.01)) ** strength))
        new[prob] = float(base.weight(prob) * mult)
    return ObjectiveWeights(new)


def problem_weight(decision: ControllerDecision, floor: float = 0.2) -> dict[str, float]:
    """engine.research.compute_manager.LadderPolicy.problem_weight: the largest problem share maps to 1.0, none below `floor`."""
    mx = max(decision.problem_shares.values(), default=0.0) or 1.0
    return {p.value: float(floor + (1.0 - floor) * decision.problem_shares.get(p, 0.0) / mx) for p in Problem}


def area_multipliers(decision: ControllerDecision) -> dict[str, float]:
    """Per section-38 area: (share of the phases that use it) / (uniform share), bounded to [0.25, 4]."""
    acc: dict[str, float] = {}
    for p, s in decision.shares.items():
        areas = PHASE_AREAS[p]
        for a in areas:
            acc[a] = acc.get(a, 0.0) + s / len(areas)
    uni = 1.0 / max(len(acc), 1)
    return {a: float(min(4.0, max(0.25, v / uni))) for a, v in sorted(acc.items())}


def directives(decision: ControllerDecision):
    """engine.research.brain_health.Directives carrying the controller's area multipliers (diversity decides how to honour them)."""
    from engine.research.brain_health import Directives
    reasons = tuple(s.text for s in decision.statements if s.kind in (StatementKind.ALLOCATE, StatementKind.SHIFT, StatementKind.REDIRECT))
    return Directives(area_multiplier=area_multipliers(decision), reasons=reasons)


def merge_directives(ours, theirs):
    """Combine the controller's directives with brain health's own (multipliers multiply; the rest is kept from brain health)."""
    if theirs is None:
        return ours
    am = dict(theirs.area_multiplier)
    for a, m in ours.area_multiplier.items():
        am[a] = float(am.get(a, 1.0) * m)
    return dataclasses.replace(theirs, area_multiplier=am, reasons=tuple(theirs.reasons) + tuple(ours.reasons))


# ---------------------------------------------------------------------------------------------------------------- persistence + report
def state_to_json(state: ControllerState) -> str:
    return json.dumps({
        "readings": {k: [dataclasses.asdict(r) | {"phase": r.phase.value} for r in v] for k, v in sorted(state.readings.items())},
        "yields": [dataclasses.asdict(y) | {"phase": y.phase.value} for y in state.yields], "satisfied": state.satisfied,
        "regressions": state.regressions, "shares": state.shares, "history": state.history, "last_now": state.last_now}, sort_keys=True)


def state_from_json(text: str, cfg: ControllerConfig | None = None) -> ControllerState:
    d = json.loads(text)
    st = new_state(cfg)
    st.readings = {k: [CapabilityReading(**{**r, "phase": Phase(r["phase"])}) for r in v] for k, v in d["readings"].items()}
    st.yields = [YieldRecord(**{**y, "phase": Phase(y["phase"])}) for y in d["yields"]]
    st.satisfied, st.regressions, st.shares = dict(d["satisfied"]), [tuple(x) for x in d["regressions"]], dict(d["shares"])
    st.history, st.last_now = list(d["history"]), d["last_now"]
    return st


def render(decision: ControllerDecision) -> str:
    lines = [f"RESEARCH CONTROLLER {decision.now}  [{LABEL}]  top: {decision.top.value}"]
    for a in decision.assessments:
        cap = "   n/a" if a.capability is None else f"{a.capability:6.2f}"
        lines.append(f"  {a.phase.value:<30} {a.status.value:<10} share {_pct(decision.share(a.phase)):>4}  capability {cap}  "
                     f"gap {a.gap:.2f}  unc {a.uncertainty:.2f}  EV {a.expected_value:.3f}  spent {a.spent_minutes:.0f}m")
    s = decision.situation
    lines.append(f"  largest uncertainty {s.largest_uncertainty}; largest prediction gap {s.largest_prediction_gap}; "
                 f"highest value {s.highest_value}; largest loss source {s.largest_loss_source or 'none'} ({s.largest_loss_share:.0%})")
    lines += ["  > " + st.text for st in decision.statements]
    return "\n".join(lines)


def share_trajectory(state: ControllerState, phase: Phase) -> list[tuple[str, float]]:
    return [(h["now"], float(h["shares"].get(phase.value, 0.0))) for h in state.history]
