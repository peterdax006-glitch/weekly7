"""Knowledge-to-decision bridge (contract C66 section 28) - IMPLEMENTED, NOT VALIDATED.

"Every discovery must eventually answer: what does this change?"  This module forces the answer. A `Discovery` states, per claim,
WHICH of the nine section-28 outputs it moves (volatility ranking, direction ranking, position sizing, risk penalty, abstention,
pattern gating, confidence, research priority, data priority), on which target, under which machine-readable condition, by how
much, and how that was measured. `Bridge.submit` triages it into exactly one disposition:
  DECISION_CHANGING  a claim with measured, held-out evidence on the decision metric (ceiling: SHADOW; the promotion gate is the only
                     road to PRODUCTION and this module never grants it)
  SHADOW_ONLY        a protective claim (tighten, gate, abstain, penalise) with no adequate measurement yet
  PRIORITY           moves only the research or data queue
  INFORMATIONAL      changes no decision and no priority; recorded as such, credited zero, never dressed up as an improvement
  REFUSED            an unsafe or malformed claim (a loosening claim without strong evidence, an impossible magnitude, an identity)
It EXTENDS engine.learning.decision_contract (bindings, check/readiness/policy_check, Contribution, apply_*, conflicting_contributions)
and engine.learning.knowledge (a routed claim becomes a real KnowledgeObject whose declared decision_effect the contract then polices).
It also keeps the books C66 section 43 demands: credit comes only from REALISED, matured, measured change on the decision metric;
counts of discoveries, patterns or experiments are never value (`assert_value_metric`). Research-world material leaves only through
`release`, which returns a MaturedRecord (gate(now) at the trader) and refuses the same-year rerun leak (C66 section 27 rule).
Public entry: `step(bridge, discoveries, now)`."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from engine.learning import decision_contract as dc
from engine.learning import knowledge as kn
from engine.learning.archive import _identity_errors
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance, _StrEnum,
                                  as_date, canonical_json, current_code_hash, require_past, stable_hash)
from engine.learning.knowledge_graph import wilson_lower
from engine.research.core import ExperimentValue, MaturedRecord, Namespace, Problem, ResearchQuestion

E = DecisionEffect


class Output(_StrEnum):
    """The nine section-28 outputs."""
    VOLATILITY_RANKING = "VOLATILITY_RANKING"
    DIRECTION_RANKING = "DIRECTION_RANKING"
    POSITION_SIZING = "POSITION_SIZING"
    RISK_PENALTY = "RISK_PENALTY"
    ABSTENTION = "ABSTENTION"
    PATTERN_GATING = "PATTERN_GATING"
    CONFIDENCE = "CONFIDENCE"
    RESEARCH_PRIORITY = "RESEARCH_PRIORITY"
    DATA_PRIORITY = "DATA_PRIORITY"


O = Output
OUTPUT_EFFECT: dict[Output, DecisionEffect] = {
    O.VOLATILITY_RANKING: E.RANKING, O.DIRECTION_RANKING: E.RANKING, O.POSITION_SIZING: E.POSITION_SIZE,
    O.RISK_PENALTY: E.POSITION_SIZE, O.ABSTENTION: E.ABSTENTION, O.PATTERN_GATING: E.PATTERN_WEIGHTING,
    O.CONFIDENCE: E.CONFIDENCE, O.RESEARCH_PRIORITY: E.RESEARCH_PRIORITY, O.DATA_PRIORITY: E.RESEARCH_PRIORITY}
OUTPUT_PROBLEM: dict[Output, Problem | None] = {
    O.VOLATILITY_RANKING: Problem.VOLATILITY, O.DIRECTION_RANKING: Problem.DIRECTION, O.RISK_PENALTY: Problem.LOSS_AVOIDANCE,
    O.ABSTENTION: Problem.LOSS_AVOIDANCE, O.DATA_PRIORITY: Problem.DATA_QUALITY}
PRIORITY_OUTPUTS = frozenset({O.RESEARCH_PRIORITY, O.DATA_PRIORITY})


@dataclasses.dataclass(frozen=True)
class OutputRule:
    """What a claim on one output may say. Direction +1 means 'more of the named quantity' (more penalty, more gating, a larger
    size, higher confidence, a higher rank score, a higher priority). `loosens` is the direction that removes a protection."""
    directions: tuple[int, ...]
    max_magnitude: float
    needs_target: bool
    needs_condition: bool
    loosens: int | None = None


RULES: dict[Output, OutputRule] = {
    O.VOLATILITY_RANKING: OutputRule((-1, 1), dc.LIMITS["rank_max_share"], True, False),
    O.DIRECTION_RANKING: OutputRule((-1, 1), dc.LIMITS["rank_max_share"], True, False),
    O.POSITION_SIZING: OutputRule((-1, 1), 1.0, True, False, loosens=1),
    O.RISK_PENALTY: OutputRule((1,), 1.0, True, False),
    O.ABSTENTION: OutputRule((1,), 1.0, False, True),
    O.PATTERN_GATING: OutputRule((1,), 1.0, True, True),
    O.CONFIDENCE: OutputRule((-1, 1), 1.0, False, False, loosens=1),
    O.RESEARCH_PRIORITY: OutputRule((1,), 1.0, False, False),
    O.DATA_PRIORITY: OutputRule((1,), 1.0, False, False)}

# volume of activity is not value (C66 section 43): a metric with any of these words in it can never be credited
WRONG_METRICS = ("n_trades", "trades", "n_predictions", "predictions", "n_patterns", "patterns_found", "n_experiments",
                 "experiments_run", "lines_of_code", "code", "backtest_return", "raw_return", "count")


def assert_value_metric(name: str) -> None:
    """Fail closed if the metric someone wants to credit is a volume or in-sample-return metric."""
    low = str(name).lower()
    bad = [w for w in WRONG_METRICS if w in low]
    if bad:
        raise ValueError(f"metric {name!r} measures activity or in-sample return ({bad[0]}); it is not decision value (section 43)")


@dataclasses.dataclass(frozen=True)
class Measurement:
    """How a claimed improvement was MEASURED: the change in the decision metric, its one-sided lower bound, the sample, the number
    of independent windows, and whether it was measured on data the discovery was not found on."""
    metric: str
    delta: float
    lower: float | None
    n: int
    windows: int = 1
    holdout: bool = False

    def check(self) -> list[str]:
        errs = []
        for f in ("delta", "lower"):
            v = getattr(self, f)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v)):
                errs.append(f"measurement.{f} is not a finite number")
        if self.lower is not None and isinstance(self.delta, (int, float)) and self.lower > self.delta + 1e-12:
            errs.append("measurement.lower exceeds delta")
        if self.n < 0 or self.windows < 0:
            errs.append("measurement n/windows must be >= 0")
        try:
            assert_value_metric(self.metric)
        except ValueError as e:
            errs.append(str(e))
        return errs


@dataclasses.dataclass(frozen=True)
class Claim:
    """One statement of 'what this changes'."""
    output: Output
    target: str = ""                              # opaque id of the pattern / feature / candidate class it applies to
    direction: int = 1
    magnitude: float = 0.0
    conditions: tuple[str, ...] = ()              # machine-readable, e.g. 'regime: state eq high_vol' (knowledge.parse_condition)
    expected_delta: float | None = None           # claimed improvement in the binding's metric
    measurement: Measurement | None = None
    how_tested: str = ""
    question: str = ""                            # required for the two priority outputs

    @property
    def effect(self) -> DecisionEffect:
        return OUTPUT_EFFECT[self.output]

    @property
    def metric(self) -> str:
        return dc.BINDINGS[self.effect].metric

    @property
    def loosens(self) -> bool:
        return RULES[self.output].loosens == self.direction

    def check(self) -> list[str]:
        r = RULES[self.output]
        errs: list[str] = []
        if self.direction not in r.directions:
            errs.append(f"{self.output.value}: direction {self.direction} not allowed (allowed {r.directions})")
        if isinstance(self.magnitude, bool) or not isinstance(self.magnitude, (int, float)) or math.isnan(self.magnitude) \
                or not 0.0 <= self.magnitude <= r.max_magnitude:
            errs.append(f"{self.output.value}: magnitude {self.magnitude!r} outside [0, {r.max_magnitude}]")
        elif self.magnitude == 0.0:
            errs.append(f"{self.output.value}: a claim of zero size changes nothing")
        if r.needs_target and not self.target:
            errs.append(f"{self.output.value}: needs a target")
        if r.needs_condition and not self.conditions:
            errs.append(f"{self.output.value}: needs a condition (when does it apply?)")
        for c in self.conditions:
            try:
                kn.parse_condition(c)
            except kn.SchemaError as e:
                errs.append(f"condition not machine-readable: {e}")
        if self.output in PRIORITY_OUTPUTS and not self.question.strip():
            errs.append(f"{self.output.value}: needs the question it wants answered")
        if self.expected_delta is not None and (isinstance(self.expected_delta, bool) or math.isnan(float(self.expected_delta))):
            errs.append("expected_delta is not a number")
        if self.measurement is not None:
            errs += self.measurement.check()
            if self.measurement.metric != self.metric:
                errs.append(f"measured on {self.measurement.metric!r} but this output is judged on {self.metric!r}")
        errs += _identity_errors({"target": self.target, "conditions": list(self.conditions), "how_tested": self.how_tested,
                                  "question": self.question}, frozenset(), "claim")
        return errs


@dataclasses.dataclass(frozen=True)
class Discovery:
    """A finding from the research side, with its claims. Immutable; a revision is a new discovery that names its parent."""
    discovery_id: str
    statement: str                                # identity-free: no tickers, dates or years
    source: str                                   # which lab / module produced it
    matured_at: str                               # real date its newest evidence matured (trusted side only)
    provenance: Provenance
    claims: tuple[Claim, ...] = ()
    informational_note: str = ""                  # an explicit 'this changes nothing, and why'
    subjects: tuple[str, ...] = ()                # graph nodes / knowledge ids the discovery is about
    problem: Problem = Problem.RESEARCH_PROCESS
    p_real: float | None = None
    parent: str = ""

    @staticmethod
    def make(statement: str, source: str, matured_at, provenance: Provenance, claims: Sequence[Claim] = (),
             informational_note: str = "", subjects: Sequence[str] = (), problem: Problem = Problem.RESEARCH_PROCESS,
             p_real: float | None = None, parent: str = "") -> "Discovery":
        body = {"s": statement, "src": source, "m": str(as_date(matured_at)), "c": [dataclasses.asdict(c) for c in claims],
                "i": informational_note, "sub": list(subjects), "pa": parent}
        return Discovery("D" + stable_hash(body, 14), statement, source, str(as_date(matured_at)), provenance, tuple(claims),
                         informational_note, tuple(subjects), problem, p_real, parent)

    def check(self) -> list[str]:
        errs = list(self.provenance.check())
        if not self.statement.strip():
            errs.append("discovery has no statement")
        errs += _identity_errors({"statement": self.statement, "note": self.informational_note, "subjects": list(self.subjects)},
                                 frozenset(), "discovery")
        if self.claims and self.informational_note:
            errs.append("a discovery with claims cannot also declare itself informational")
        if self.p_real is not None and not 0.0 <= self.p_real <= 1.0:
            errs.append("p_real outside [0,1]")
        return errs

    def years(self) -> frozenset[int]:
        """Real calendar years the discovery is filed under (matured, learned, newest outcome seen)."""
        ds = {self.matured_at, self.provenance.learned_at, self.provenance.outcomes_seen_through} - {""}
        return frozenset(as_date(d).year for d in ds)


@dataclasses.dataclass(frozen=True)
class BridgeConfig:
    min_n_tighten: int = 10
    min_n_loosen: int = 30
    min_windows_loosen: int = 2
    require_holdout: bool = True
    weight_cap: float = 0.5                      # a routed knowledge object never starts with more than this influence
    shrink_k: float = 5.0                        # pseudo-observations pulling a claim source's hit rate toward 0.5
    inflation_ratio: float = 3.0                 # claimed / realised beyond this flags a source as inflating

    def check(self) -> list[str]:
        errs = []
        if self.min_n_loosen < self.min_n_tighten:
            errs.append("loosening a protection must need at least as much evidence as tightening one")
        if not 0 < self.weight_cap <= 1:
            errs.append("weight_cap must be in (0,1]")
        if self.min_windows_loosen < 1 or self.shrink_k <= 0 or self.inflation_ratio <= 1:
            errs.append("min_windows_loosen >= 1, shrink_k > 0, inflation_ratio > 1 required")
        return errs


class Disposition(_StrEnum):
    DECISION_CHANGING = "DECISION_CHANGING"
    SHADOW_ONLY = "SHADOW_ONLY"
    PRIORITY = "PRIORITY"
    INFORMATIONAL = "INFORMATIONAL"
    REFUSED = "REFUSED"


_STRENGTH = {Disposition.DECISION_CHANGING: 4, Disposition.SHADOW_ONLY: 3, Disposition.PRIORITY: 2,
             Disposition.INFORMATIONAL: 1, Disposition.REFUSED: 0}


@dataclasses.dataclass(frozen=True)
class ClaimVerdict:
    index: int
    output: Output
    status: Disposition
    reasons: tuple[str, ...]
    ceiling: dc.Mode | None                      # the highest mode the claim may reach through this bridge


def evidence_sufficient(m: Measurement | None, cfg: BridgeConfig, loosening: bool) -> tuple[bool, list[str]]:
    """Is the measurement good enough to let the claim influence decisions (in shadow)? Loosening a protection needs more."""
    if m is None:
        return False, ["no measurement on the decision metric"]
    why = []
    need = cfg.min_n_loosen if loosening else cfg.min_n_tighten
    if m.n < need:
        why.append(f"n={m.n} below {need}")
    if m.lower is None or m.lower <= 0:
        why.append("one-sided lower bound of the improvement is not above zero")
    if cfg.require_holdout and not m.holdout:
        why.append("not measured on held-out data")
    if loosening and m.windows < cfg.min_windows_loosen:
        why.append(f"only {m.windows} independent window(s); loosening needs {cfg.min_windows_loosen}")
    return not why, why


def triage_claim(index: int, claim: Claim, cfg: BridgeConfig) -> ClaimVerdict:
    """Decide what one claim is allowed to become. Malformed or unsafe claims are REFUSED with every reason listed."""
    errs = claim.check()
    if errs:
        return ClaimVerdict(index, claim.output, Disposition.REFUSED, tuple(errs), None)
    if claim.output in PRIORITY_OUTPUTS:
        return ClaimVerdict(index, claim.output, Disposition.PRIORITY, (), dc.Mode.RESEARCH)
    ok, why = evidence_sufficient(claim.measurement, cfg, claim.loosens)
    if ok:
        return ClaimVerdict(index, claim.output, Disposition.DECISION_CHANGING, (), dc.Mode.SHADOW)
    if claim.loosens:
        return ClaimVerdict(index, claim.output, Disposition.REFUSED,
                            tuple(["loosening a protection needs measured, held-out, multi-window evidence"] + why), None)
    return ClaimVerdict(index, claim.output, Disposition.SHADOW_ONLY, tuple(why), dc.Mode.SHADOW)


def usefulness_share(claim: Claim) -> float | None:
    """Share of the CLAIMED benefit still supported at the measurement's lower bound (None when nothing was measured or claimed):
    the honest reading of 'usefulness' as a fraction, not a score invented from p-values."""
    m = claim.measurement
    if m is None or m.lower is None or claim.expected_delta is None or claim.expected_delta <= 0:
        return None
    return float(min(1.0, max(0.0, m.lower / claim.expected_delta)))


# ====================================================================================================== routing

def claim_to_knowledge(d: Discovery, index: int, cfg: BridgeConfig = BridgeConfig()) -> kn.KnowledgeObject:
    """A claim as a real KnowledgeObject so decision_contract polices it. The bridge NEVER raises epistemic state or promotion: the
    object is a HYPOTHESIS in RESEARCH, its nudge is the knob change the claim asks for, and every confidence dimension the
    bridge cannot measure stays None (untested is not zero and not passing)."""
    c = d.claims[index]
    ctx = kn.context_from_text(*c.conditions) if c.conditions else kn.ContextSet()
    size_max = None
    if c.effect == E.POSITION_SIZE:
        size_max = 1.0 if c.output == O.RISK_PENALTY else min(2.0, 1.0 + c.magnitude)
    return kn.KnowledgeObject(
        knowledge_id="K-" + stable_hash([d.discovery_id, index], 12), created_at=d.matured_at, provenance=d.provenance,
        observation=d.statement, hypothesis=f"{c.output.value.lower()} on {c.target or 'the whole decision'}: "
                                            f"{'raise' if c.direction > 0 else 'lower'} by {c.magnitude:g}",
        contexts=ctx, effect=kn.Effect(direction=c.direction if c.magnitude else 0, size=c.magnitude, unit=c.output.value.lower()),
        confidence=Confidence(truth=d.p_real, usefulness=usefulness_share(c)),
        evidence=kn.Evidence(sample_size=c.measurement.n if c.measurement else 0,
                             effective_sample_size=float(c.measurement.n if c.measurement else 0),
                             last_evidence_at=d.matured_at),
        mechanism_tags=("bridge", c.output.value.lower()), decision_effect=(c.effect,),
        action_policy=kn.ActionPolicy(weight_cap=cfg.weight_cap, min_reliability=0.5, abstain_when_unknown=True,
                                      size_multiplier_max=size_max, note="set by the bridge; the promotion gate may raise it"),
        epistemic=Epistemic.HYPOTHESIS, lifecycle=Lifecycle.BIRTH, promotion=Promotion.RESEARCH)


@dataclasses.dataclass(frozen=True)
class BridgeEntry:
    """The one recorded answer to 'what does this change?' for a discovery. Chained: `hash` covers the previous entry's hash."""
    entry_id: str
    discovery_id: str
    source: str
    disposition: Disposition
    verdicts: tuple[ClaimVerdict, ...]
    outputs: tuple[str, ...]                     # outputs of surviving claims
    effects: tuple[str, ...]
    ceiling: str                                 # 'SHADOW' / 'RESEARCH' / '' (none)
    unstated: bool                               # informational only because nobody said what it changes
    knowledge_ids: tuple[str, ...]
    contract: tuple[tuple[str, bool, tuple[str, ...]], ...]      # (knowledge id, allowed in RESEARCH mode, reasons)
    readiness: tuple[tuple[str, tuple[str, ...]], ...]           # (knowledge id, what production still needs)
    answer: str
    recorded_at: str
    prev: str
    hash: str = ""

    def body(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "hash"}

    @property
    def changes_a_decision(self) -> bool:
        return self.disposition in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY)


def _seal(entry: BridgeEntry) -> BridgeEntry:
    return dataclasses.replace(entry, hash=stable_hash([entry.prev, entry.body()], 24))


def route(d: Discovery, now, cfg: BridgeConfig = BridgeConfig(), prev: str = "") -> tuple[BridgeEntry, list[kn.KnowledgeObject]]:
    """Triage one discovery into exactly one disposition and build the knowledge objects for its surviving claims."""
    reasons = d.check()
    verdicts: list[ClaimVerdict] = []
    kos: list[kn.KnowledgeObject] = []
    if reasons:
        verdicts = [ClaimVerdict(i, c.output, Disposition.REFUSED, tuple(reasons), None) for i, c in enumerate(d.claims)]
        disp = Disposition.REFUSED
    elif not d.claims:
        disp = Disposition.INFORMATIONAL
    else:
        for i, c in enumerate(d.claims):
            v = triage_claim(i, c, cfg)
            if v.status != Disposition.REFUSED:
                try:
                    ko = claim_to_knowledge(d, i, cfg)
                    bad = ko.validate()
                except (kn.SchemaError, ValueError) as e:
                    bad = [str(e)]
                if bad:
                    v = ClaimVerdict(i, c.output, Disposition.REFUSED, v.reasons + tuple(f"not valid knowledge: {b}" for b in bad), None)
                else:
                    kos.append(ko)
            verdicts.append(v)
        live = [v for v in verdicts if v.status != Disposition.REFUSED]
        disp = max((v.status for v in live), key=_STRENGTH.get) if live else Disposition.REFUSED
    ceilings = {v.ceiling for v in verdicts if v.ceiling is not None}
    ceiling = dc.Mode.SHADOW.value if dc.Mode.SHADOW in ceilings else dc.Mode.RESEARCH.value if ceilings else ""
    contract, ready = [], []
    for ko in kos:
        v = dc.check(ko, now, dc.Mode.RESEARCH)
        contract.append((ko.knowledge_id, v.allowed, v.reasons))
        rd = dc.readiness(ko, now)
        ready.append((ko.knowledge_id, tuple(rd["missing"])))
    live_v = [v for v in verdicts if v.status != Disposition.REFUSED]
    outputs = tuple(sorted({v.output.value for v in live_v}))
    effects = tuple(sorted({OUTPUT_EFFECT[v.output].value for v in live_v}))
    unstated = disp == Disposition.INFORMATIONAL and not d.informational_note
    if disp == Disposition.INFORMATIONAL:
        answer = (f"{d.discovery_id}: changes no decision and no research priority - informational only ("
                  f"{d.informational_note or 'nobody stated what it changes'}); credited zero value")
    elif disp == Disposition.REFUSED:
        answer = f"{d.discovery_id}: refused - " + "; ".join(dict.fromkeys(r for v in verdicts for r in v.reasons))[:400]
    else:
        answer = " | ".join(dc.answer(k) for k in kos)
    e = BridgeEntry("B" + stable_hash([d.discovery_id, str(as_date(now))], 12), d.discovery_id, d.source, disp, tuple(verdicts),
                    outputs, effects, ceiling, unstated, tuple(k.knowledge_id for k in kos), tuple(contract), tuple(ready), answer,
                    str(as_date(now)), prev)
    return _seal(e), kos


@dataclasses.dataclass(frozen=True)
class Realised:
    """What a released claim actually did, measured later on matured data. Only this creates credit."""
    entry_id: str
    claim_index: int
    metric: str
    delta: float
    n: int
    matured_at: str


@dataclasses.dataclass(frozen=True)
class PriorityRequest:
    """A research or data request the bridge hands to the research queue."""
    request_id: str
    question: ResearchQuestion
    data: bool
    urgency: float
    source: str


@dataclasses.dataclass(frozen=True)
class OutputCalibration:
    output: str
    realised: int
    hit_rate: float | None
    hit_lower: float | None
    claimed_over_realised: float | None          # mean expected / mean realised where both known (>1: claims run ahead of reality)
    shrunk_ratio: float                          # what to multiply the next claim's expected delta by


@dataclasses.dataclass(frozen=True)
class LedgerReport:
    now: str
    submitted: int
    by_disposition: Mapping[str, int]
    informational_unstated: int
    realised_by_metric: Mapping[str, float]      # net realised change per metric (harms subtract); never summed across metrics
    realised_entries: int
    note: str = "counts of discoveries are not value; only realised change on a decision metric is credited"


# ====================================================================================================== the register

class BridgeError(RuntimeError):
    pass


class Bridge:
    """Append-only register of routed discoveries. Every submission gets exactly one chained entry (no pending state); realised
    outcomes come later and are the only source of credit. Nothing here can promote knowledge: `release` hands a shadow-cleared
    claim to the trader side as a MaturedRecord and refuses anything the promotion gate has not cleared or that would leak."""

    def __init__(self, cfg: BridgeConfig = BridgeConfig()):
        bad = cfg.check()
        if bad:
            raise BridgeError("invalid BridgeConfig: " + "; ".join(bad))
        self.cfg = cfg
        self._entries: list[BridgeEntry] = []
        self._disc: dict[str, Discovery] = {}
        self._ko: dict[str, kn.KnowledgeObject] = {}
        self._realised: list[Realised] = []

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ submitting
    def submit(self, d: Discovery, now) -> BridgeEntry:
        """Route a discovery. Fails closed on a discovery that matured on or after `now` or whose provenance could not exist then.
        Idempotent for the same discovery; a different discovery under an existing id is refused (revise instead)."""
        require_past(d.matured_at, now, f"discovery {d.discovery_id}")
        if not d.provenance.could_exist_at(now):
            raise FirewallBreach(f"discovery {d.discovery_id} could not have existed at {as_date(now)}")
        old = self._disc.get(d.discovery_id)
        if old is not None:
            if old != d:
                raise BridgeError(f"discovery id {d.discovery_id} already holds different content; submit a revision")
            return self.entry_of(d.discovery_id)
        if d.parent and d.parent not in self._disc:
            raise BridgeError(f"discovery {d.discovery_id}: parent {d.parent} was never submitted")
        entry, kos = route(d, now, self.cfg, self._entries[-1].hash if self._entries else "")
        self._entries.append(entry)
        self._disc[d.discovery_id] = d
        self._ko.update({k.knowledge_id: k for k in kos})
        return entry

    def submit_all(self, ds: Iterable[Discovery], now) -> list[BridgeEntry]:
        return [self.submit(d, now) for d in ds]

    def revise(self, old_id: str, new: Discovery, now) -> BridgeEntry:
        """A revised discovery supersedes the old one for conflict checks and release; the old entry stays in the record."""
        if old_id not in self._disc:
            raise BridgeError(f"unknown discovery {old_id}")
        if new.parent != old_id:
            raise BridgeError("a revision must name the discovery it revises as parent")
        return self.submit(new, now)

    def superseded(self) -> frozenset[str]:
        return frozenset(d.parent for d in self._disc.values() if d.parent)

    def entries(self) -> tuple[BridgeEntry, ...]:
        return tuple(self._entries)

    def entry_of(self, discovery_id: str) -> BridgeEntry:
        for e in self._entries:
            if e.discovery_id == discovery_id:
                return e
        raise BridgeError(f"discovery {discovery_id} has no entry")

    def discovery(self, discovery_id: str) -> Discovery:
        return self._disc[discovery_id]

    def knowledge(self) -> list[kn.KnowledgeObject]:
        return [self._ko[k] for k in sorted(self._ko)]

    def live_entries(self) -> list[BridgeEntry]:
        gone = self.superseded()
        return [e for e in self._entries if e.discovery_id not in gone]

    # ------------------------------------------------------------------ realised outcomes and calibration
    def record_realised(self, entry_id: str, claim_index: int, delta: float, n: int, matured_at, now) -> Realised:
        """A measured change on the decision metric, from data that matured strictly before `now`. This is the only credit."""
        e = next((x for x in self._entries if x.entry_id == entry_id), None)
        if e is None:
            raise BridgeError(f"unknown entry {entry_id}")
        require_past(matured_at, now, f"realised outcome of {entry_id}")
        d = self._disc[e.discovery_id]
        if not 0 <= claim_index < len(d.claims):
            raise BridgeError(f"claim {claim_index} does not exist")
        if e.verdicts[claim_index].status in (Disposition.REFUSED, Disposition.INFORMATIONAL):
            raise BridgeError("a refused claim was never released; it cannot have a realised outcome")
        if math.isnan(delta) or math.isinf(delta) or n <= 0:
            raise BridgeError("realised outcome needs a finite delta and n > 0")
        if as_date(matured_at) < as_date(e.recorded_at):
            raise BridgeError("realised outcome predates the routing decision: it cannot be an outcome of it")
        r = Realised(entry_id, claim_index, d.claims[claim_index].metric, float(delta), int(n), str(as_date(matured_at)))
        self._realised.append(r)
        return r

    def realised(self, now=None) -> list[Realised]:
        return [r for r in self._realised if now is None or as_date(r.matured_at) < as_date(now)]

    def calibration(self, now) -> dict[str, OutputCalibration]:
        """Per output: how often released claims actually improved the metric, and how far claims ran ahead of reality. The
        shrunk ratio (claimed/realised, pulled toward 0.5 with `shrink_k` pseudo-observations) discounts the next claim."""
        by: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
        for r in self.realised(now):
            e = next(x for x in self._entries if x.entry_id == r.entry_id)
            c = self._disc[e.discovery_id].claims[r.claim_index]
            by[c.output.value].append((r.delta, c.expected_delta))
        out = {}
        for o in Output:
            rows = by.get(o.value, [])
            n = len(rows)
            hits = sum(1 for d, _ in rows if d > 0)
            both = [(d, x) for d, x in rows if x is not None and x > 0]
            ratio = (sum(x for _, x in both) / sum(max(d, 1e-12) for d, _ in both)) if both and sum(d for d, _ in both) > 0 else None
            shrunk = (sum(max(0.0, min(1.0, d / x)) for d, x in both) + 0.5 * self.cfg.shrink_k) / (len(both) + self.cfg.shrink_k)
            out[o.value] = OutputCalibration(o.value, n, (hits / n) if n else None, wilson_lower(hits, n), ratio, float(shrunk))
        return out

    def discounted_expected(self, claim: Claim, now) -> float | None:
        """The claim's expected improvement scaled by how well this output's past claims held up."""
        if claim.expected_delta is None:
            return None
        return float(claim.expected_delta * self.calibration(now)[claim.output.value].shrunk_ratio)

    def source_report(self, now) -> dict[str, dict[str, Any]]:
        """Per source lab: routed, decision-changing, realised, hit rate (shrunk toward 0.5), and whether its claims inflate."""
        rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"routed": 0, "decision_changing": 0, "realised": 0, "hits": 0,
                                                              "claimed": 0.0, "delivered": 0.0})
        for e in self._entries:
            r = rows[e.source]
            r["routed"] += 1
            r["decision_changing"] += e.disposition == Disposition.DECISION_CHANGING
        for x in self.realised(now):
            e = next(v for v in self._entries if v.entry_id == x.entry_id)
            c = self._disc[e.discovery_id].claims[x.claim_index]
            r = rows[e.source]
            r["realised"] += 1
            r["hits"] += x.delta > 0
            if c.expected_delta is not None and c.expected_delta > 0:
                r["claimed"] += c.expected_delta
                r["delivered"] += max(0.0, x.delta)
        for r in rows.values():
            r["trust"] = (r["hits"] + 0.5 * self.cfg.shrink_k) / (r["realised"] + self.cfg.shrink_k)
            r["inflating"] = bool(r["realised"] >= 3 and r["claimed"] > self.cfg.inflation_ratio * max(r["delivered"], 1e-12))
        return {k: dict(v) for k, v in sorted(rows.items())}

    def trust(self, source: str, now) -> float:
        return float(self.source_report(now).get(source, {"trust": 0.5})["trust"])

    # ------------------------------------------------------------------ research and data priorities
    def priority_requests(self, now) -> list[PriorityRequest]:
        """Every live priority claim as a ResearchQuestion request, ordered by urgency = magnitude x the source's trust. Duplicated
        questions merge to the strongest."""
        best: dict[str, PriorityRequest] = {}
        for e in self.live_entries():
            d = self._disc[e.discovery_id]
            for v in e.verdicts:
                if v.status != Disposition.PRIORITY:
                    continue
                c = d.claims[v.index]
                prob = OUTPUT_PROBLEM.get(c.output) or d.problem
                q = ResearchQuestion.make(c.question, "bridge:" + c.output.value.lower(), prob, e.recorded_at, d.matured_at,
                                          "the question is answered with measured evidence", "it cannot be answered from available data",
                                          expected=ExperimentValue(decision_value=c.expected_delta))
                urg = float(c.magnitude * self.trust(d.source, now))
                cur = best.get(q.question_id)
                if cur is None or urg > cur.urgency:
                    best[q.question_id] = PriorityRequest(q.question_id, q, c.output == O.DATA_PRIORITY, urg, d.source)
        return sorted(best.values(), key=lambda r: (-r.urgency, r.request_id))

    # ------------------------------------------------------------------ consistency
    def contributions(self, include_shadow_only: bool = True) -> list[dc.Contribution]:
        out = []
        for e in self.live_entries():
            if e.disposition not in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY if include_shadow_only else Disposition.DECISION_CHANGING):
                continue
            d = self._disc[e.discovery_id]
            for v in e.verdicts:
                if v.status in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY):
                    c = d.claims[v.index]
                    out.append(dc.Contribution(e.discovery_id, c.effect, 1.0, c.direction * c.magnitude, c.target))
        return out

    def conflicts(self) -> list[tuple[str, str, str, str]]:
        """Live claims that push the same knob on the same target in opposite directions (never averaged away: reported)."""
        return dc.conflicting_contributions(self.contributions())

    def duplicates(self) -> list[tuple[str, str]]:
        """Distinct live discoveries making the same claim (same output, target, direction and conditions)."""
        seen: dict[tuple, str] = {}
        dup = []
        for e in self.live_entries():
            d = self._disc[e.discovery_id]
            for v in e.verdicts:
                if v.status == Disposition.REFUSED:
                    continue
                c = d.claims[v.index]
                key = (c.output, c.target, c.direction, tuple(sorted(c.conditions)))
                if key in seen and seen[key] != d.discovery_id:
                    dup.append(tuple(sorted((seen[key], d.discovery_id))))
                seen.setdefault(key, d.discovery_id)
        return sorted(set(dup))

    def value_ledger(self, now) -> LedgerReport:
        """The honest books. Dispositions are counted (as bookkeeping, not as a score); credit is the net REALISED change per metric
        and INFORMATIONAL / refused entries can never contribute to it."""
        by = Counter(e.disposition.value for e in self._entries)
        net: dict[str, float] = defaultdict(float)
        seen_entries = set()
        for r in self.realised(now):
            assert_value_metric(r.metric)
            net[r.metric] += r.delta
            seen_entries.add(r.entry_id)
        return LedgerReport(str(as_date(now)), len(self._entries), dict(sorted(by.items())),
                            sum(1 for e in self._entries if e.unstated), dict(sorted(net.items())), len(seen_entries))

    def audit(self, now) -> list[str]:
        """Invariants: chain intact, every discovery routed exactly once, dispositions consistent with their claims and evidence,
        nothing dated after `now`, no loosening claim survived on thin evidence, no informational entry with an effect."""
        errs = []
        prev = ""
        for i, e in enumerate(self._entries):
            if e.prev != prev or e.hash != _seal(dataclasses.replace(e, hash="")).hash:
                errs.append(f"entry {i} {e.entry_id}: chain broken")
            prev = e.hash
            if as_date(e.recorded_at) > as_date(now):
                errs.append(f"entry {e.entry_id}: recorded after now")
            d = self._disc.get(e.discovery_id)
            if d is None:
                errs.append(f"entry {e.entry_id}: discovery missing")
                continue
            if e.disposition == Disposition.INFORMATIONAL and (e.effects or e.knowledge_ids):
                errs.append(f"entry {e.entry_id}: informational yet changes {e.effects}")
            if e.disposition == Disposition.DECISION_CHANGING:
                for v in e.verdicts:
                    if v.status == Disposition.DECISION_CHANGING:
                        c = d.claims[v.index]
                        ok, why = evidence_sufficient(c.measurement, self.cfg, c.loosens)
                        if not ok:
                            errs.append(f"entry {e.entry_id}: decision-changing claim {v.index} lacks evidence ({why[0]})")
            for v in e.verdicts:
                if v.status != Disposition.REFUSED and d.claims[v.index].loosens and v.status != Disposition.DECISION_CHANGING:
                    errs.append(f"entry {e.entry_id}: loosening claim {v.index} survived without evidence")
            if e.disposition == Disposition.SHADOW_ONLY and e.ceiling == "":
                errs.append(f"entry {e.entry_id}: shadow-only with no ceiling")
        ids = [e.discovery_id for e in self._entries]
        if len(ids) != len(set(ids)):
            errs.append("a discovery was routed more than once")
        for r in self._realised:
            if as_date(r.matured_at) >= as_date(now):
                errs.append(f"realised outcome of {r.entry_id} matured on/after now")
        return errs

    def verify(self) -> bool:
        return not [e for e in self.audit(dt.date.max) if "chain" in e]

    # ------------------------------------------------------------------ release to the trader side
    def release(self, entry_id: str, now, created_real: str, replaying: Iterable[int] = ()) -> MaturedRecord:
        """Hand a shadow-cleared entry to the trader side as a MaturedRecord (the trader must call .gate(its_now)). Refused for
        anything not DECISION_CHANGING, for a superseded discovery, for a non-empty conflict, and - the same-year rerun leak - when
        any real year the discovery is filed under is being replayed in disguise right now (`replaying`)."""
        from engine.learning import trader_view
        e = next((x for x in self._entries if x.entry_id == entry_id), None)
        if e is None:
            raise BridgeError(f"unknown entry {entry_id}")
        d = self._disc[e.discovery_id]
        if e.disposition != Disposition.DECISION_CHANGING:
            raise FirewallBreach(f"entry {entry_id} is {e.disposition.value}: only DECISION_CHANGING entries may be released")
        if e.discovery_id in self.superseded():
            raise FirewallBreach(f"entry {entry_id} was superseded by a revision")
        clash = d.years() & {int(y) for y in replaying}
        if clash:
            raise FirewallBreach(f"discovery {d.discovery_id} is filed under year(s) {sorted(clash)} that are being replayed: "
                                 f"releasing it would leak the answer into the same-year rerun")
        mine = {(c.effect.value, c.target) for c in (d.claims[v.index] for v in e.verdicts if v.status == Disposition.DECISION_CHANGING)}
        for eff, tgt, a, b in self.conflicts():
            if e.discovery_id in (a, b) and (eff, tgt) in mine:
                raise FirewallBreach(f"entry {entry_id} conflicts with another live claim on {eff}/{tgt}: resolve before release")
        claims = [{"output": c.output.value, "target": c.target, "direction": c.direction, "magnitude": c.magnitude,
                   "conditions": list(c.conditions), "ceiling": v.ceiling.value if v.ceiling else ""}
                  for v in e.verdicts if v.status == Disposition.DECISION_CHANGING for c in [d.claims[v.index]]]
        payload = {"claims": claims, "mode": "SHADOW"}
        trader_view.assert_trader_safe(payload, f"release of {entry_id}")
        prov = dataclasses.replace(d.provenance, created_real=created_real)
        return MaturedRecord(stable_hash([entry_id, payload], 16), d.matured_at, payload, prov)

    # ------------------------------------------------------------------ what a claim would do
    def preview(self, entry_id: str, base_scores: Mapping[str, float] | None = None, base_weight: float = 0.05,
                base_confidence: float = 0.5) -> dict[str, Any]:
        """Show the concrete change each surviving claim asks for on a toy decision, using decision_contract's own limiters:
        ranking scores, a position weight, an abstain flag, a confidence. It is the literal answer to 'what does this change?'."""
        e = next(x for x in self._entries if x.entry_id == entry_id)
        d = self._disc[e.discovery_id]
        out: dict[str, Any] = {}
        for v in e.verdicts:
            if v.status in (Disposition.REFUSED, Disposition.INFORMATIONAL):
                continue
            c = d.claims[v.index]
            contrib = dc.Contribution(e.discovery_id, c.effect, min(1.0, self.cfg.weight_cap), c.direction * c.magnitude, c.target)
            if c.effect == E.RANKING:
                base = dict(base_scores or {c.target: 1.0})
                out[c.output.value] = {"before": base, "after": dc.apply_ranking(base, [contrib])}
            elif c.effect == E.POSITION_SIZE:
                m = 1.0 + contrib.weight * contrib.signed
                m = min(m, 1.0) if c.output == O.RISK_PENALTY else m
                out[c.output.value] = {"before": base_weight, "after": dc.apply_position_size(base_weight, [max(0.0, m)])}
            elif c.effect == E.ABSTENTION:
                out[c.output.value] = {"before": False, "after": dc.apply_abstention([contrib])[0]}
            elif c.effect == E.CONFIDENCE:
                out[c.output.value] = {"before": base_confidence, "after": dc.apply_confidence(base_confidence, [contrib])}
            elif c.effect == E.PATTERN_WEIGHTING:
                out[c.output.value] = {"before": 1.0, "after": max(0.0, 1.0 - contrib.weight * contrib.signed)}
            else:
                out[c.output.value] = {"before": 0.0, "after": c.magnitude, "queue": "data" if c.output == O.DATA_PRIORITY else "research"}
        return out

    def promotion_brief(self, entry_id: str) -> dict[str, list[str]]:
        """Per knowledge object of an entry: what the promotion gate still needs before production could even be considered."""
        e = next(x for x in self._entries if x.entry_id == entry_id)
        return {k: list(m) for k, m in e.readiness}

    def revisit_informational(self, evidence_subjects: Iterable[str]) -> list[str]:
        """Informational entries whose subjects have since gained new evidence: an 'it changes nothing' verdict is only as good
        as the evidence it was given. Returns discovery ids worth re-submitting with a claim (or a firmer note)."""
        fresh = set(evidence_subjects)
        return sorted(e.discovery_id for e in self.live_entries() if e.disposition == Disposition.INFORMATIONAL
                      and set(self._disc[e.discovery_id].subjects) & fresh)




# ====================================================================================================== persistence

def discovery_to_dict(d: Discovery) -> dict:
    return json.loads(canonical_json(d))


def discovery_from_dict(x: Mapping) -> Discovery:
    claims = tuple(Claim(Output(c["output"]), c["target"], int(c["direction"]), float(c["magnitude"]), tuple(c["conditions"]),
                         c["expected_delta"], Measurement(**c["measurement"]) if c["measurement"] else None, c["how_tested"],
                         c["question"]) for c in x["claims"])
    pv = dict(x["provenance"])
    for f in ("sealed_windows", "parents"):
        pv[f] = tuple(pv.get(f) or ())
    return Discovery(x["discovery_id"], x["statement"], x["source"], x["matured_at"], Provenance(**pv), claims,
                     x["informational_note"], tuple(x["subjects"]), Problem(x["problem"]), x["p_real"], x["parent"])


def to_records(b: Bridge) -> dict:
    """Portable dump: the discoveries, the routing dates and the entry hashes. `from_records` re-routes every discovery on its
    recorded date and demands identical hashes, so a hand-edited disposition cannot survive a reload."""
    return {"cfg": dataclasses.asdict(b.cfg),
            "discoveries": [discovery_to_dict(b.discovery(e.discovery_id)) for e in b.entries()],
            "routed": [(e.discovery_id, e.recorded_at, e.hash) for e in b.entries()],
            "realised": [dataclasses.asdict(r) for r in b._realised]}


def from_records(rec: Mapping) -> Bridge:
    b = Bridge(BridgeConfig(**rec["cfg"]))
    by_id = {d["discovery_id"]: discovery_from_dict(d) for d in rec["discoveries"]}
    for did, at, h in rec["routed"]:
        e = b.submit(by_id[did], as_date(at) + dt.timedelta(days=0))
        if e.hash != h or e.recorded_at != at:
            raise BridgeError(f"entry for {did} does not reproduce its hash: the record was altered")
    for r in rec["realised"]:
        b._realised.append(Realised(**r))
    return b


# ====================================================================================================== adapters: graph -> discoveries

def condition_text(context_id: str) -> str:
    """A graph context node id as a machine-readable claim condition: 'reg:high_vol' -> 'regime: state eq high_vol'."""
    prefix, _, name = context_id.partition(":")
    dim = {"reg": "regime", "sect": "sector", "evt": "event", "cls": "cohort", "feat": "feature"}.get(prefix, "context")
    return f"{dim}: state eq {re.sub(r'[^a-z0-9_]', '_', name.lower())}"


def _prov(created_real: str, learned_at: str, code_hash: str | None) -> Provenance:
    return Provenance(created_real=created_real, learned_at=learned_at, code_hash=code_hash or current_code_hash(),
                      outcomes_seen_through=learned_at)


_DATA_GAPS = frozenset({"UNEXPLAINED_FAILURE", "CONTEXT_BLIND", "COVERAGE_HOLE"})


def discoveries_from_gaps(gaps: Iterable, now, created_real: str, code_hash: str | None = None) -> list[Discovery]:
    """Every graph gap answers 'what does this change?' as a research or data priority (or, for an ungated failure, a candidate
    pattern gate with no measurement yet). None is left unclassified."""
    out = []
    for gp in gaps:
        learned = gp.known_through
        prov = _prov(created_real, learned, code_hash)
        if gp.kind.value == "UNGATED_FAILURE":
            pat, ctx = gp.subjects
            share = float(gp.evidence.get("failures_in_context", 1)) / max(1.0, float(gp.evidence.get("failures", 1)))
            claim = Claim(O.PATTERN_GATING, pat, 1, max(0.05, min(1.0, share)), (condition_text(ctx),),
                          how_tested="graph co-occurrence only; no held-out test yet")
            out.append(Discovery.make(gp.text, "research_graph", learned, prov, [claim], subjects=gp.subjects, problem=gp.problem))
            continue
        output = O.DATA_PRIORITY if gp.kind.value in _DATA_GAPS else O.RESEARCH_PRIORITY
        claim = Claim(output, "", 1, max(0.01, min(1.0, gp.score)), question=gp.text, expected_delta=gp.expected.decision_value)
        out.append(Discovery.make(gp.text, "research_graph", learned, prov, [claim], subjects=gp.subjects, problem=gp.problem))
    return out


def discoveries_from_contexts(graph, now, created_real: str, code_hash: str | None = None, min_n: int = 3) -> list[Discovery]:
    """Contexts where the active patterns overwhelmingly fail (context_table verdict DANGER) become ABSTENTION candidates. The graph
    only counts patterns; it has no measurement on the decision metric, so these stay SHADOW_ONLY until the evidence lab measures them."""
    from engine.research import research_graph as rg
    out = []
    for row in rg.context_table(graph, now, min_n):
        if row.verdict != "DANGER":
            continue
        learned = rg._newest(graph, [row.context], now)
        claim = Claim(O.ABSTENTION, "", 1, min(1.0, row.fails / row.tested), (condition_text(row.context),),
                      how_tested="pattern-level tally in the research graph, not measured on the decision metric")
        out.append(Discovery.make(f"{row.fails} of {row.tested} active patterns fail in {row.context}; abstaining there is a candidate "
                                  f"protection", "research_graph", learned, _prov(created_real, learned, code_hash), [claim],
                                  subjects=(row.context,), problem=Problem.LOSS_AVOIDANCE))
    return out


def discoveries_from_stories(graph, now, created_real: str, metric: str = "portfolio_return_delta",
                             code_hash: str | None = None) -> list[Discovery]:
    """A complete section-27 chain that ends in an out-of-sample improvement is a candidate gate WITH a measurement: the outcome's
    recorded effect. Without an independent lower bound or a hold-out flag the bridge still routes it SHADOW_ONLY."""
    from engine.research import research_graph as rg
    out = []
    for p in rg._active_beliefs(graph, now):
        s = graph.pattern_story(p, now)
        if not s.complete or not s.gates or not s.improvements:
            continue
        eff = [graph.node_at(o, now).attrs.get("effect") for o in s.improvements]
        eff = [float(x) for x in eff if x is not None]
        if not eff:
            continue
        learned = rg._newest(graph, [p, *s.gates, *s.improvements], now)
        m = Measurement(metric, max(eff), None, 0, 1, False)
        claim = Claim(O.PATTERN_GATING, p, 1, 1.0, (condition_text(s.gates[0]),), expected_delta=max(eff), measurement=m,
                      how_tested="graph story: failure explained, gate tested, out-of-sample improvement recorded")
        out.append(Discovery.make(s.text(), "research_graph", learned, _prov(created_real, learned, code_hash), [claim],
                                  subjects=(p, *s.gates), problem=rg.problem_of(graph, p, now)))
    return out


def attach_to_graph(b: Bridge, graph, known_at) -> int:
    """Record, in the research graph, which beliefs feed which decision knob: a DECISION node per (effect, problem) and a
    USED_IN_DECISION edge from every belief that a live surviving claim names (weight = the claim's magnitude). Only claims that
    change a decision count; informational and priority-only entries add nothing. Returns edges written."""
    from engine.research import research_graph as rg
    n = 0
    for e in b.live_entries():
        if not e.changes_a_decision:
            continue
        d = b.discovery(e.discovery_id)
        for v in e.verdicts:
            if v.status not in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY):
                continue
            c = d.claims[v.index]
            dec = graph.add_research_node(rg.K.DECISION, f"{c.effect.value.lower()}-{c.output.value.lower()}", known_at)
            for s in {c.target, *d.subjects} - {""}:
                if graph.kind_of(s) in rg.BELIEF_KINDS:
                    graph.relate(s, dec.node_id, rg.R.USED_IN_DECISION, known_at, min(1.0, max(0.01, c.magnitude)))
                    n += 1
    return n


# ====================================================================================================== entry and reports

@dataclasses.dataclass(frozen=True)
class BridgeStepReport:
    now: str
    entries: tuple[BridgeEntry, ...]
    by_disposition: Mapping[str, int]
    priorities: tuple[PriorityRequest, ...]
    conflicts: tuple[tuple[str, str, str, str], ...]
    duplicates: tuple[tuple[str, str], ...]
    audit: tuple[str, ...]
    ledger: LedgerReport
    refused: tuple[tuple[str, str], ...]              # (discovery id, first reason)

    @property
    def clean(self) -> bool:
        return not self.audit and not self.conflicts


def step(b: Bridge, discoveries: Iterable[Discovery], now) -> BridgeStepReport:
    """PUBLIC ENTRY: route today's discoveries, then report priorities, conflicts, duplicates, audit findings and the ledger."""
    fresh = b.submit_all(list(discoveries), now)
    refused = tuple((e.discovery_id, next((r for v in e.verdicts for r in v.reasons), e.answer)) for e in fresh
                    if e.disposition == Disposition.REFUSED)
    return BridgeStepReport(str(as_date(now)), tuple(fresh), dict(sorted(Counter(e.disposition.value for e in fresh).items())),
                            tuple(b.priority_requests(now)), tuple(b.conflicts()), tuple(b.duplicates()), tuple(b.audit(now)),
                            b.value_ledger(now), refused)


def bridge_markdown(b: Bridge, now) -> str:
    led = b.value_ledger(now)
    lines = [f"# Knowledge-to-decision bridge as of {as_date(now)}", "", f"{led.submitted} discoveries routed. {led.note}.", "",
             "| disposition | discoveries |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in led.by_disposition.items()]
    lines += ["", f"{led.informational_unstated} informational because nobody said what they change.", "",
              "| metric | net realised change |", "|---|---:|"] + [f"| {k} | {v:+.4f} |" for k, v in led.realised_by_metric.items()]
    cal = [c for c in b.calibration(now).values() if c.realised]
    if cal:
        lines += ["", "| output | realised | hit rate | claimed/realised |", "|---|---:|---:|---:|"]
        lines += [f"| {c.output} | {c.realised} | {c.hit_rate:.2f} | {c.claimed_over_realised if c.claimed_over_realised is not None else float('nan'):.2f} |" for c in cal]
    lines += ["", "| discovery | disposition | answer |", "|---|---|---|"]
    lines += [f"| {e.discovery_id} | {e.disposition.value} | {e.answer[:110]} |" for e in b.entries()[:25]]
    return "\n".join(lines)


# ====================================================================================================== pre-flight and upgrade paths

@dataclasses.dataclass(frozen=True)
class Lint:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    predicted: Disposition


def lint(d: Discovery, cfg: BridgeConfig = BridgeConfig()) -> Lint:
    """Pre-flight for a discovery before submission: the exact refusals `route` would give, plus warnings that do not refuse but make a
    claim weak (no expected size, expected size with no measurement, wide claim with a narrow test)."""
    errs = list(d.check())
    warns: list[str] = []
    for i, c in enumerate(d.claims):
        errs += [f"claim {i}: {m}" for m in c.check()]
        if c.output not in PRIORITY_OUTPUTS:
            if c.expected_delta is None:
                warns.append(f"claim {i}: no expected improvement stated")
            if c.measurement is None:
                warns.append(f"claim {i}: never measured on {c.metric}")
            elif c.measurement.n < cfg.min_n_tighten:
                warns.append(f"claim {i}: measured on only {c.measurement.n} cases")
            if c.measurement is not None and c.expected_delta is not None and c.measurement.delta < 0.5 * c.expected_delta:
                warns.append(f"claim {i}: measured improvement is under half of what is claimed")
            if not c.how_tested.strip():
                warns.append(f"claim {i}: how_tested is empty")
    if not d.claims and not d.informational_note:
        warns.append("no claims and no informational note: it will be recorded as informational because nobody said what it changes")
    pred = route(d, dt.date.max - dt.timedelta(days=1))[0].disposition if not errs else Disposition.REFUSED
    return Lint(not errs, tuple(dict.fromkeys(errs)), tuple(warns), pred)


def upgrade_path(claim: Claim, cfg: BridgeConfig = BridgeConfig()) -> list[str]:
    """What a SHADOW_ONLY claim still needs to become DECISION_CHANGING (empty when it already is, or is a priority claim)."""
    if claim.output in PRIORITY_OUTPUTS:
        return []
    return evidence_sufficient(claim.measurement, cfg, claim.loosens)[1]


def upgrade_queue(b: Bridge, now) -> list[tuple[str, int, list[str], float]]:
    """(discovery id, claim index, what is missing, value if upgraded): shadow-only claims ranked by the size of the improvement they
    promise times their source's earned trust. This is how the bridge tells the research loop what measurement to run next."""
    out = []
    for e in b.live_entries():
        d = b.discovery(e.discovery_id)
        for v in e.verdicts:
            if v.status != Disposition.SHADOW_ONLY:
                continue
            c = d.claims[v.index]
            value = (c.expected_delta if c.expected_delta is not None else c.magnitude) * b.trust(d.source, now)
            out.append((e.discovery_id, v.index, upgrade_path(c, b.cfg), float(value)))
    return sorted(out, key=lambda t: (-t[3], t[0], t[1]))


def decision_coverage(b: Bridge) -> dict[str, Any]:
    """Which of the nine outputs any live discovery moves and which none does - an output nobody can move is a knob the research
    program has no way to turn."""
    live = Counter()
    for e in b.live_entries():
        for v in e.verdicts:
            if v.status not in (Disposition.REFUSED, Disposition.INFORMATIONAL):
                live[v.output.value] += 1
    return {"moved": dict(sorted(live.items())), "unmoved": sorted(o.value for o in Output if o.value not in live),
            "share_covered": len(live) / len(Output)}


def concentration(b: Bridge, max_share: float = 0.5) -> list[str]:
    """Outputs where one source supplies more than `max_share` of the live claims: a single lab steering a knob is a single point of
    failure for that decision."""
    by: dict[str, Counter] = defaultdict(Counter)
    for e in b.live_entries():
        for v in e.verdicts:
            if v.status not in (Disposition.REFUSED, Disposition.INFORMATIONAL):
                by[v.output.value][e.source] += 1
    out = []
    for o, c in sorted(by.items()):
        tot = sum(c.values())
        src, k = c.most_common(1)[0]
        if tot >= 3 and k / tot > max_share:
            out.append(f"{o}: {src} supplies {k} of {tot} live claims")
    return out


def stack_effect(b: Bridge, base_scores: Mapping[str, float], base_weight: float = 0.05, base_confidence: float = 0.5,
                 include_shadow_only: bool = False) -> dict[str, Any]:
    """The combined effect of ALL live decision-changing claims on one toy decision, through decision_contract's own limiters. Shows
    whether stacked lessons keep to the contract's bounds (rank shift capped, size within [0, cap], confidence never inflated)."""
    contribs = b.contributions(include_shadow_only)
    ranked = dc.apply_ranking(base_scores, contribs)
    factors = []
    for e_ in b.live_entries():
        d_ = b.discovery(e_.discovery_id)
        for v in e_.verdicts:
            c_ = d_.claims[v.index]
            if c_.effect == E.POSITION_SIZE and v.status in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY if include_shadow_only else Disposition.DECISION_CHANGING):
                f_ = 1.0 + b.cfg.weight_cap * c_.direction * c_.magnitude
                factors.append(max(0.0, min(f_, 1.0)) if c_.output == O.RISK_PENALTY else max(0.0, f_))
    size = dc.apply_position_size(base_weight, factors)
    abstain, voters = dc.apply_abstention(contribs)
    conf = dc.apply_confidence(base_confidence, contribs)
    spread = (max(base_scores.values()) - min(base_scores.values())) or 1.0 if base_scores else 1.0
    return {"ranking": ranked, "size": size, "abstain": abstain, "abstain_voters": voters, "confidence": conf,
            "max_rank_shift": max((abs(ranked[k] - base_scores[k]) for k in base_scores), default=0.0),
            "rank_limit": dc.LIMITS["rank_max_share"] * spread, "conflicts": b.conflicts()}


def attribute_realised(b: Bridge, now) -> dict[str, float]:
    """When several live claims moved the same decision knob, split each realised change among them in proportion to claimed
    magnitude (a claim never gets credit for a delta it did not share). Returns discovery id -> credited change."""
    credit: dict[str, float] = defaultdict(float)
    by_entry: dict[str, list[Realised]] = defaultdict(list)
    for r in b.realised(now):
        by_entry[r.entry_id].append(r)
    for eid, rs in by_entry.items():
        e = next(x for x in b.entries() if x.entry_id == eid)
        d = b.discovery(e.discovery_id)
        for r in rs:
            c = d.claims[r.claim_index]
            peers = [(x.discovery_id, b.discovery(x.discovery_id).claims[v.index]) for x in b.live_entries() for v in x.verdicts
                     if v.status in (Disposition.DECISION_CHANGING, Disposition.SHADOW_ONLY)
                     and b.discovery(x.discovery_id).claims[v.index].effect == c.effect
                     and b.discovery(x.discovery_id).claims[v.index].target == c.target]
            tot = sum(pc.magnitude for _, pc in peers) or c.magnitude
            credit[e.discovery_id] += r.delta * (c.magnitude / tot)
    return dict(sorted(credit.items()))


def stale_entries(b: Bridge, now, days: int = 180) -> list[str]:
    """Live shadow-only entries with no realised outcome after `days`: claims parked in shadow forever are neither tested nor retired."""
    out = []
    done = {r.entry_id for r in b.realised(now)}
    for e in b.live_entries():
        if e.disposition in (Disposition.SHADOW_ONLY, Disposition.DECISION_CHANGING) and e.entry_id not in done \
                and (as_date(now) - as_date(e.recorded_at)).days > days:
            out.append(e.discovery_id)
    return sorted(out)


def retire_failed(b: Bridge, now, min_realised: int = 3) -> list[str]:
    """Discoveries whose realised changes are, in the mean, non-positive after at least `min_realised` measurements: they promised a
    decision improvement and did not deliver one. Returned for the promotion gate; the bridge itself never demotes."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in b.realised(now):
        e = next(x for x in b.entries() if x.entry_id == r.entry_id)
        by[e.discovery_id].append(r.delta)
    return sorted(k for k, v in by.items() if len(v) >= min_realised and sum(v) / len(v) <= 0)


def from_pattern_rows(rows, now, created_real: str, output: Output = O.VOLATILITY_RANKING, metric: str | None = None,
                      code_hash: str | None = None) -> list[Discovery]:
    """PatternMiner.patterns rows (a DataFrame with key_named, effect, p_real, status) -> one discovery per ACTIVE pattern claiming a
    ranking nudge proportional to its effect. Rows without a usable effect, or in a retired status, produce an informational
    discovery (recorded, credited zero). No measurement is attached: the bridge will route them SHADOW_ONLY until the evidence lab
    measures them on the decision metric."""
    from engine.pattern_lifecycle import parse_key_named, pattern_id
    out = []
    if rows is None or len(rows) == 0:
        return out
    for r in rows.itertuples(index=False):
        key = str(r.key_named)
        try:
            pid = "pat:" + pattern_id(parse_key_named(key)[1:])
        except ValueError:
            continue
        eff = getattr(r, "effect", float("nan"))
        status = str(getattr(r, "status", "active")).lower()
        learned = str(as_date(now) - dt.timedelta(days=1))
        prov = _prov(created_real, learned, code_hash)
        if status in ("retired", "discarded", "rejected", "no_gain", "duplicate", "disregarded") or eff is None or math.isnan(float(eff)) or eff == 0:
            out.append(Discovery.make(f"pattern {pid} is {status} or has no usable effect", "pattern_miner", learned, prov,
                                      informational_note=f"status {status}: nothing to route", subjects=(pid,)))
            continue
        claim = Claim(output, pid, 1 if eff > 0 else -1, min(abs(float(eff)) * 10.0, RULES[output].max_magnitude),
                      how_tested="miner discovery-window statistic only")
        out.append(Discovery.make(f"pattern {pid} moves the forward return by {float(eff):+.4f}", "pattern_miner", learned, prov, [claim],
                                  subjects=(pid,), p_real=getattr(r, "p_real", None), problem=Problem.VOLATILITY))
    return out


def disposition_flow(b: Bridge, t0, t1) -> dict[str, dict[str, int]]:
    """Dispositions of entries recorded in [t0, t1): how the mix of what the research program produces is changing. A program that
    only produces informational findings is learning things that change nothing."""
    out: dict[str, Counter] = defaultdict(Counter)
    for e in b.entries():
        if as_date(t0) <= as_date(e.recorded_at) < as_date(t1):
            out[e.source][e.disposition.value] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}


def informational_rate(b: Bridge) -> float | None:
    """Share of routed discoveries that change nothing. Near 1.0 for a lab means it is producing knowledge without consequence."""
    n = len(b.entries())
    return (sum(e.disposition == Disposition.INFORMATIONAL for e in b.entries()) / n) if n else None
