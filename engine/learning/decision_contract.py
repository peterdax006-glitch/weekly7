"""Contract C62 section 43: KNOWLEDGE -> DECISION CONTRACT.

Every active knowledge object must answer "what decision does this change?". An item that cannot answer is research
knowledge: it may be kept, searched and learned from, but a production check REFUSES it. The check is the only door between
the knowledge base and a trading decision, and it fails closed: an untested confidence dimension, an unknown applicability,
a retired lifecycle, a future-dated record or a mismatch between the declared effect and the action policy each refuse.

Modes: PRODUCTION (CHAMPION only), SHADOW (logged, never traded: SHADOW/CHALLENGER/CHAMPION), RESEARCH (anything not retired).
Works on KnowledgeLike duck types (engine.learning.knowledge.KnowledgeObject satisfies it); imports nothing of other builders.
Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import enum
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from engine.learning.core import (DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, as_date, stable_hash)
from engine.learning.epistemic import DEFAULT_THRESHOLDS

E = DecisionEffect


class Mode(str, enum.Enum):
    PRODUCTION = "PRODUCTION"
    SHADOW = "SHADOW"
    RESEARCH = "RESEARCH"


class Operation(str, enum.Enum):
    ADD = "ADD"                    # adds to a score
    MULTIPLY = "MULTIPLY"          # scales a quantity inside [0, bound]
    VETO = "VETO"                  # can only remove or block
    BOUND = "BOUND"                # tightens a limit


@dataclasses.dataclass(frozen=True)
class Binding:
    """The concrete knob one decision effect turns, and how it may turn it."""
    effect: DecisionEffect
    knob: str
    operation: Operation
    metric: str                    # how the change is measured (section 34: portfolio value, not just prediction)
    trades_money: bool


BINDINGS: dict[DecisionEffect, Binding] = {
    E.RANKING: Binding(E.RANKING, "candidate_rank_score", Operation.ADD, "portfolio_return_delta", True),
    E.SELECTION: Binding(E.SELECTION, "selected_set", Operation.VETO, "portfolio_return_delta", True),
    E.POSITION_SIZE: Binding(E.POSITION_SIZE, "position_weight", Operation.MULTIPLY, "risk_adjusted_delta", True),
    E.DIRECTION: Binding(E.DIRECTION, "trade_side", Operation.VETO, "direction_accuracy_delta", True),
    E.TIMING: Binding(E.TIMING, "entry_day", Operation.BOUND, "timing_gain", True),
    E.EXIT: Binding(E.EXIT, "exit_rule", Operation.BOUND, "exit_gain", True),
    E.STOP: Binding(E.STOP, "stop_distance", Operation.BOUND, "tail_loss_delta", True),
    E.ABSTENTION: Binding(E.ABSTENTION, "abstain_flag", Operation.VETO, "loss_avoided", False),
    E.RESEARCH_PRIORITY: Binding(E.RESEARCH_PRIORITY, "research_queue_priority", Operation.ADD, "information_gain", False),
    E.PATTERN_WEIGHTING: Binding(E.PATTERN_WEIGHTING, "pattern_weight", Operation.MULTIPLY, "portfolio_return_delta", True),
    E.CONFIDENCE: Binding(E.CONFIDENCE, "decision_confidence", Operation.MULTIPLY, "calibration_delta", False),
}

_ACT = frozenset({Epistemic.SUPPORTED, Epistemic.CONDITIONAL, Epistemic.DEGRADED})
EPISTEMIC_ALLOWED: dict[DecisionEffect, frozenset[Epistemic]] = {e: _ACT for e in BINDINGS}
# Research priority is HOW unknowns get resolved, so it may use unproven and even contradicted knowledge - but not retired.
EPISTEMIC_ALLOWED[E.RESEARCH_PRIORITY] = frozenset(set(Epistemic) - {Epistemic.RETIRED})

LIFECYCLE_ACTIVE = frozenset({Lifecycle.GROWTH, Lifecycle.PEAK, Lifecycle.ACTIVE, Lifecycle.RECOVERY, Lifecycle.DECAY})
LIFECYCLE_DEAD = frozenset({Lifecycle.RETIRED, Lifecycle.FAILURE, Lifecycle.DORMANT})
PROMOTION_FOR_MODE: dict[Mode, frozenset[Promotion]] = {
    Mode.PRODUCTION: frozenset({Promotion.CHAMPION}),
    Mode.SHADOW: frozenset({Promotion.SHADOW, Promotion.CHALLENGER, Promotion.CHAMPION}),
    Mode.RESEARCH: frozenset({Promotion.RESEARCH, Promotion.SHADOW, Promotion.CHALLENGER, Promotion.CHAMPION}),
}


class ContractError(RuntimeError):
    pass


class ResearchOnlyError(ContractError):
    """The item changes no decision, so it is not production knowledge."""


class ProductionRefused(ContractError):
    pass


@dataclasses.dataclass(frozen=True)
class Verdict:
    knowledge_id: str
    mode: Mode
    allowed: bool
    effects: tuple[DecisionEffect, ...]        # the effects that PASSED (may be a subset of those declared)
    reasons: tuple[str, ...]                   # why anything was refused
    research_only: bool

    def explain(self) -> str:
        head = f"{self.knowledge_id} [{self.mode.value}] " + ("ALLOWED " + ",".join(e.value for e in self.effects)
                                                            if self.allowed else "REFUSED")
        return head + ("" if not self.reasons else ": " + "; ".join(self.reasons))


def declared_effects(k: Any) -> tuple[DecisionEffect, ...]:
    de = tuple(DecisionEffect.parse(x) for x in getattr(k, "decision_effect", ()) or ())
    return tuple(x for x in de if x != DecisionEffect.NONE)


def answer(k: Any) -> str:
    """'What decision does this change?' in words, or an explicit statement that it changes none."""
    eff = declared_effects(k)
    if not eff:
        return f"{getattr(k, 'knowledge_id', '?')}: changes no decision (research knowledge only)"
    parts = [f"{e.value} via {BINDINGS[e].knob} ({BINDINGS[e].operation.value}), measured by {BINDINGS[e].metric}" for e in eff]
    return f"{getattr(k, 'knowledge_id', '?')}: " + "; ".join(parts)


def _policy_errors(k: Any, effect: DecisionEffect) -> list[str]:
    pol = getattr(k, "action_policy", None)
    if pol is None:
        return ["no action_policy"]
    errs = []
    if pol.weight_cap <= 0:
        errs.append("action_policy.weight_cap is 0: it may not influence anything")
    if effect == E.POSITION_SIZE:
        m = getattr(pol, "size_multiplier_max", None)
        if m is None:
            errs.append("POSITION_SIZE declared but action_policy.size_multiplier_max is not set")
    return errs


def check(k: Any, now, mode: Mode = Mode.PRODUCTION, min_usefulness: float = 0.5) -> Verdict:
    """The production check. Every reason is reported, not just the first, so a refusal is diagnosable."""
    mode = Mode(mode)
    kid = str(getattr(k, "knowledge_id", "?"))
    eff = declared_effects(k)
    if not eff:
        return Verdict(kid, mode, False, (), ("research-only: no decision effect declared",), True)

    hard: list[str] = []                       # structural: nothing may pass while these exist
    validate = getattr(k, "validate", None)
    if callable(validate):
        hard += [f"invalid record: {e}" for e in validate()]
    try:
        upd = as_date(getattr(k, "updated_at"))
        if upd >= as_date(now):
            hard.append(f"version dated {upd} did not exist before now={now}")
    except (AttributeError, ValueError):
        hard.append("record has no usable updated_at")
    prov = getattr(k, "provenance", None)
    if prov is None or not prov.could_exist_at(now):
        hard.append("provenance says the item could not have existed at now (future knowledge)")

    life = getattr(k, "lifecycle", None)
    dead: list[str] = []                       # a retired/failed item may not even steer research priority
    if life in LIFECYCLE_DEAD or life is None:
        dead.append(f"lifecycle {getattr(life, 'value', life)} cannot influence decisions")

    gates: list[str] = []                      # the trading-side gates
    if mode != Mode.RESEARCH and not dead and life not in LIFECYCLE_ACTIVE:
        gates.append(f"lifecycle {life.value} has not reached an active stage")
    promo = getattr(k, "promotion", None)
    if promo not in PROMOTION_FOR_MODE[mode]:
        gates.append(f"promotion {getattr(promo, 'value', promo)} not allowed in {mode.value} mode")
    conf = getattr(k, "confidence", None)
    if mode != Mode.RESEARCH:
        if conf is None:
            gates.append("no confidence record")
        else:
            floor = getattr(getattr(k, "action_policy", None), "min_reliability", 0.5)
            for dim, lo in (("truth", 0.0), ("usefulness", min_usefulness), ("current_reliability", floor)):
                v = getattr(conf, dim, None)
                if v is None:
                    gates.append(f"confidence.{dim} was never measured (untested is not zero and not passing)")
                elif v < lo:
                    gates.append(f"confidence.{dim}={v:.2f} below {lo:.2f}")
            if getattr(conf, "failure_risk", None) is not None and conf.failure_risk > 0.5:
                gates.append(f"failure_risk {conf.failure_risk:.2f} above 0.5")

    ep = getattr(k, "epistemic", None)
    passed: list[DecisionEffect] = []
    reasons: list[str] = list(hard) + list(dead)
    for e in eff:
        own = [f"{e.value}: {m}" for m in _policy_errors(k, e)]
        if ep not in EPISTEMIC_ALLOWED[e]:
            own.append(f"{e.value} not allowed for epistemic {getattr(ep, 'value', ep)}")
        if e != E.RESEARCH_PRIORITY:                # research priority never moves money, so trading gates do not apply
            own += gates
        if hard or dead or own:
            reasons += own
        else:
            passed.append(e)
    reasons = list(dict.fromkeys(reasons))
    return Verdict(kid, mode, bool(passed), tuple(passed), tuple(reasons), False)


def require_production(k: Any, now, mode: Mode = Mode.PRODUCTION) -> Verdict:
    """Raise unless the item may influence a decision now. ResearchOnlyError is distinct: it is not a failure of the item."""
    v = check(k, now, mode)
    if v.research_only:
        raise ResearchOnlyError(v.explain())
    if not v.allowed:
        raise ProductionRefused(v.explain())
    return v


# ---------------------------------------------------------------- how much influence

@dataclasses.dataclass(frozen=True)
class Influence:
    knowledge_id: str
    effects: tuple[DecisionEffect, ...]
    weight: float                              # in [0, weight_cap]; 0 means present but silent
    abstain: bool                              # the situation is unknown to this item and its policy says to abstain
    reason: str


def _epistemic_weight(k: Any, active_contexts: tuple[str, ...]) -> tuple[float, str]:
    prof = getattr(k, "epistemic_profile", None)
    if prof is not None and not prof.is_empty():
        u = prof.usage(active_contexts)
        return u.weight, u.reason
    ep = getattr(k, "epistemic", None)
    if ep == Epistemic.SUPPORTED:
        return 1.0, "supported"
    if ep == Epistemic.CONDITIONAL:
        return (1.0, "conditional, context matched by applicability") if active_contexts else (0.0, "conditional with no matched context")
    if ep == Epistemic.DEGRADED:
        return DEFAULT_THRESHOLDS.degraded_weight, "degraded: reduced weight"
    return 0.0, f"{getattr(ep, 'value', ep)} carries no weight"


def influence(k: Any, situation: Mapping[str, Any], now, mode: Mode = Mode.PRODUCTION) -> Influence:
    """Effect and weight this item may exert in THIS situation. Applicability UNKNOWN never grants weight."""
    kid = str(getattr(k, "knowledge_id", "?"))
    v = check(k, now, mode)
    if not v.allowed:
        return Influence(kid, (), 0.0, False, "; ".join(v.reasons))
    app = k.applicability(situation) if hasattr(k, "applicability") else None
    name = str(getattr(app, "value", app))
    pol = k.action_policy
    if name == "UNKNOWN":
        return Influence(kid, v.effects, 0.0, bool(pol.abstain_when_unknown), "situation lacks a feature this item needs")
    if name in ("EXCLUDED", "OUT_OF_CONTEXT"):
        return Influence(kid, v.effects, 0.0, False, name.lower().replace("_", " "))
    ctx = tuple(f"{c.dimension}:{c.feature}" for c in getattr(k.contexts, "conditions", ()))
    w, why = _epistemic_weight(k, ctx)
    rel = getattr(k.confidence, "current_reliability", None)
    rel = 0.0 if rel is None else float(rel)
    return Influence(kid, v.effects, max(0.0, min(pol.weight_cap, w * rel)), False, why)


def size_multiplier(inf: Influence, k: Any) -> float:
    """Position-size factor in [1 - weight, 1 + weight*(max-1)] bounded by the policy; 1.0 (neutral) when silent."""
    if E.POSITION_SIZE not in inf.effects or inf.weight <= 0:
        return 1.0
    m = float(k.action_policy.size_multiplier_max)
    return min(m, max(0.0, 1.0 + inf.weight * (m - 1.0)))


# ---------------------------------------------------------------- registry-level views

@dataclasses.dataclass(frozen=True)
class RegistryAudit:
    total: int
    research_only: tuple[str, ...]
    allowed: tuple[str, ...]
    refused: tuple[str, ...]
    reason_counts: tuple[tuple[str, int], ...]
    contradictory_pairs: tuple[tuple[str, str], ...]
    orphan_champions: tuple[str, ...]

    def clean(self) -> bool:
        return not self.orphan_champions and not self.contradictory_pairs


def split(items: Iterable[Any], now, mode: Mode = Mode.PRODUCTION) -> tuple[list[tuple[Any, Verdict]], list[tuple[Any, Verdict]]]:
    allowed, refused = [], []
    for k in items:
        v = check(k, now, mode)
        (allowed if v.allowed else refused).append((k, v))
    return allowed, refused


def decision_map(items: Iterable[Any], now, mode: Mode = Mode.PRODUCTION) -> dict[str, list[str]]:
    """decision effect -> ids of items allowed to change it (sorted; the audit trail of what moves what)."""
    out: dict[str, list[str]] = {}
    for k, v in split(items, now, mode)[0]:
        for e in v.effects:
            out.setdefault(e.value, []).append(v.knowledge_id)
    return {e: sorted(ids) for e, ids in sorted(out.items())}


def research_only(items: Iterable[Any]) -> list[str]:
    return sorted(str(getattr(k, "knowledge_id", "?")) for k in items if not declared_effects(k))


def audit_registry(items: Iterable[Any], now, mode: Mode = Mode.PRODUCTION) -> RegistryAudit:
    items = list(items)
    allowed, refused = split(items, now, mode)
    counts: Counter = Counter()
    for _, v in refused:
        for r in v.reasons:
            counts[r.split("=")[0].split(":")[0][:60]] += 1
    ids = {str(k.knowledge_id): (k, v) for k, v in allowed}
    pairs = set()
    for kid, (k, v) in ids.items():
        rel = getattr(k, "relations", None)
        for other in getattr(rel, "contradicting", ()) if rel is not None else ():
            if other in ids and set(ids[other][1].effects) & set(v.effects):
                pairs.add(tuple(sorted((kid, other))))
    orphans = tuple(sorted(str(k.knowledge_id) for k in items
                           if getattr(k, "promotion", None) == Promotion.CHAMPION and not declared_effects(k)))
    return RegistryAudit(len(items), tuple(research_only(items)), tuple(sorted(v.knowledge_id for _, v in allowed)),
                         tuple(sorted(v.knowledge_id for _, v in refused)), tuple(sorted(counts.items())),
                         tuple(sorted(pairs)), orphans)


def assert_no_future(items: Iterable[Any], now) -> None:
    """Fail closed if the registry handed to a decision contains anything from `now` or later."""
    for k in items:
        if as_date(k.updated_at) >= as_date(now):
            raise FirewallBreach(f"{k.knowledge_id} v{getattr(k, 'version', '?')} updated {k.updated_at} >= now {now}")


# ---------------------------------------------------------------- how each effect turns its knob

@dataclasses.dataclass(frozen=True)
class Contribution:
    """One item's requested nudge to a knob, before any limit is applied."""
    knowledge_id: str
    effect: DecisionEffect
    weight: float                    # from Influence.weight, already in [0, weight_cap]
    signed: float = 0.0              # direction*size the item asserts (score units, multiplier delta, or bound)
    target: str = ""                 # candidate id / pattern id the nudge applies to; "" = whole decision


LIMITS = {"rank_max_share": 0.5,     # knowledge may move a rank score by at most half its base spread
          "size_floor": 0.0, "size_cap": 2.0,
          "stop_may_loosen": False}  # a learned item may only TIGHTEN a stop, never widen it (loss cap, section 43 safety)


def apply_ranking(base: Mapping[str, float], contribs: Iterable[Contribution], max_share: float = LIMITS["rank_max_share"]
                  ) -> dict[str, float]:
    """candidate_rank_score: base + sum(weight*signed), with the total shift per candidate limited to `max_share` of the base
    cross-section spread, so no accumulation of small lessons can override the base ranking."""
    vals = list(base.values())
    if not vals:
        return {}
    spread = (max(vals) - min(vals)) or 1.0
    shift = {k: 0.0 for k in base}
    for c in contribs:
        if c.effect not in (DecisionEffect.RANKING, DecisionEffect.PATTERN_WEIGHTING) or c.target not in base:
            continue
        shift[c.target] += c.weight * c.signed
    lim = max_share * spread
    return {k: base[k] + max(-lim, min(lim, shift[k])) for k in base}


def apply_selection(candidates: Iterable[str], contribs: Iterable[Contribution]) -> tuple[list[str], list[tuple[str, str]]]:
    """selected_set: VETO only. A SELECTION item can remove a candidate (signed < 0 with weight > 0), never add one that the
    base process did not propose. Returns (kept, [(removed, knowledge_id)])."""
    cand = list(candidates)
    removed: dict[str, str] = {}
    for c in contribs:
        if c.effect == DecisionEffect.SELECTION and c.weight > 0 and c.signed < 0 and c.target in cand:
            removed.setdefault(c.target, c.knowledge_id)
    return [x for x in cand if x not in removed], sorted(removed.items())


def apply_position_size(base_weight: float, factors: Iterable[float], floor: float = LIMITS["size_floor"],
                        cap: float = LIMITS["size_cap"]) -> float:
    """position_weight * product(factor), factors already bounded per item (size_multiplier). Result clipped to
    [floor, cap] * base: the portfolio layer keeps the final say on gross exposure."""
    f = 1.0
    for x in factors:
        if x < 0 or math.isnan(x):
            raise ContractError(f"invalid size factor {x}")
        f *= x
    return base_weight * max(floor, min(cap, f))


def apply_stop(base_distance: float, contribs: Iterable[Contribution], may_loosen: bool = LIMITS["stop_may_loosen"]) -> float:
    """stop_distance: contributions ask for a distance; the smallest requested distance wins (tighten). A request wider
    than the base is ignored unless loosening is explicitly permitted, because gap risk means a wider stop cannot cap
    a loss but a tighter one can only cost a whisker of upside."""
    if base_distance <= 0:
        raise ContractError("base stop distance must be positive")
    asks = [(c.weight, base_distance + c.weight * (c.signed - base_distance)) for c in contribs
            if c.effect == DecisionEffect.STOP and c.weight > 0 and c.signed > 0]
    if not asks:
        return base_distance
    if not may_loosen:
        return min([base_distance] + [a for _, a in asks])
    return max(asks)[1]                                  # loosening allowed: the most heavily weighted request decides


def apply_abstention(contribs: Iterable[Contribution], min_weight: float = 0.25) -> tuple[bool, list[str]]:
    """abstain_flag: any single sufficiently weighted ABSTENTION item can stop the decision (asymmetric on purpose: a
    reason to stay out beats several reasons to go). Returns (abstain, ids that voted for it)."""
    votes = sorted(c.knowledge_id for c in contribs if c.effect == DecisionEffect.ABSTENTION and c.weight >= min_weight)
    return bool(votes), votes


def apply_confidence(base: float, contribs: Iterable[Contribution]) -> float:
    """decision_confidence in [0,1]: multiplied down by CONFIDENCE items (signed < 0), never inflated above base."""
    if not 0.0 <= base <= 1.0:
        raise ContractError("base confidence must be in [0,1]")
    f = 1.0
    for c in contribs:
        if c.effect == DecisionEffect.CONFIDENCE and c.weight > 0:
            f *= max(0.0, 1.0 + c.weight * min(0.0, c.signed))
    return base * f


# ---------------------------------------------------------------- decision log (for credit assignment later)

@dataclasses.dataclass(frozen=True)
class LoggedInfluence:
    seq: int
    at: str
    decision_id: str
    knowledge_id: str
    version: int
    effects: tuple[str, ...]
    weight: float
    abstain: bool
    reason: str
    prev_hash: str
    hash: str = ""

    def body_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("hash")
        return stable_hash(d, 20)


class DecisionLog:
    """Append-only, hash-chained record of which knowledge influenced which decision, at what weight, and why not when
    it did not. Credit assignment (section 20) and the future-memory audit read this instead of guessing."""

    def __init__(self):
        self._rows: list[LoggedInfluence] = []

    def __len__(self):
        return len(self._rows)

    def rows(self) -> tuple[LoggedInfluence, ...]:
        return tuple(self._rows)

    def record(self, decision_id: str, k: Any, inf: Influence, now) -> LoggedInfluence:
        if self._rows and as_date(now) < as_date(self._rows[-1].at):
            raise FirewallBreach("decision log entries must arrive in time order")
        prev = self._rows[-1].hash if self._rows else ""
        row = LoggedInfluence(len(self._rows), str(as_date(now)), decision_id, inf.knowledge_id, int(getattr(k, "version", 0)),
                              tuple(e.value for e in inf.effects), float(inf.weight), bool(inf.abstain), inf.reason, prev)
        row = dataclasses.replace(row, hash=row.body_hash())
        self._rows.append(row)
        return row

    def record_all(self, decision_id: str, items: Iterable[Any], situation: Mapping[str, Any], now,
                   mode: Mode = Mode.PRODUCTION) -> list[Influence]:
        out = []
        for k in items:
            inf = influence(k, situation, now, mode)
            self.record(decision_id, k, inf, now)
            out.append(inf)
        return out

    def verify(self) -> list[str]:
        errs, prev = [], ""
        for i, r in enumerate(self._rows):
            if r.seq != i or r.prev_hash != prev:
                errs.append(f"row {i}: chain broken")
            if r.hash != r.body_hash():
                errs.append(f"row {i}: contents altered")
            prev = r.hash
        return errs

    def for_decision(self, decision_id: str) -> list[LoggedInfluence]:
        return [r for r in self._rows if r.decision_id == decision_id]

    def usage_counts(self) -> dict[str, int]:
        """knowledge_id -> number of decisions it actually influenced (weight > 0)."""
        out: dict[str, int] = {}
        for r in self._rows:
            if r.weight > 0:
                out[r.knowledge_id] = out.get(r.knowledge_id, 0) + 1
        return dict(sorted(out.items()))

    def silent_ids(self, ids: Iterable[str]) -> list[str]:
        """Allowed knowledge that never influenced anything: candidates for the research-only shelf (section 43)."""
        used = set(self.usage_counts())
        return sorted(set(ids) - used)


# ---------------------------------------------------------------- what is missing before an item may be promoted

def readiness(k: Any, now, min_usefulness: float = 0.5) -> dict:
    """Diagnostic for the promotion gate: for each requirement of PRODUCTION use, met or not, and the list still missing.
    Same rules as `check`, evaluated as if the item were a CHAMPION, so 'promotion' itself is never the blocker."""
    missing: list[str] = []
    if not declared_effects(k):
        missing.append("declare a decision effect (what decision does it change?)")
    proxy = dataclasses.replace(k, promotion=Promotion.CHAMPION) if dataclasses.is_dataclass(k) else k
    v = check(proxy, now, Mode.PRODUCTION, min_usefulness)
    missing += [r for r in v.reasons if not r.startswith("research-only")]
    conf = getattr(k, "confidence", None)
    untested = list(conf.untested()) if conf is not None and hasattr(conf, "untested") else []
    return {"ready": not missing, "missing": sorted(dict.fromkeys(missing)), "untested_confidence": untested,
            "effects": [e.value for e in declared_effects(k)]}


def contribution_from(k: Any, inf: Influence, effect: DecisionEffect, target: str = "", requested: float | None = None
                      ) -> Contribution | None:
    """Turn an allowed influence into the nudge it requests. `requested` is the distance/size an item asks for when the knob
    takes one (stop distance); otherwise the item's own signed effect is the nudge. None if the effect is not permitted."""
    if effect not in inf.effects or inf.weight <= 0:
        return None
    signed = requested if requested is not None else float(getattr(k.effect, "signed", 0.0))
    return Contribution(inf.knowledge_id, effect, inf.weight, signed, target)


def conflicting_contributions(contribs: Iterable[Contribution]) -> list[tuple[str, str, str, str]]:
    """Pairs of contributions on the same knob and target that push opposite ways: (effect, target, id_a, id_b). Opposite
    pushes are not averaged away silently; they are reported so the contradiction graph can learn from them."""
    by: dict[tuple[str, str], list[Contribution]] = {}
    for c in contribs:
        if c.weight > 0 and c.signed != 0:
            by.setdefault((c.effect.value, c.target), []).append(c)
    out = []
    for (eff, tgt), cs in sorted(by.items()):
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                if (a.signed > 0) != (b.signed > 0):
                    out.append((eff, tgt, *sorted((a.knowledge_id, b.knowledge_id))))
    return sorted(set(out))


def net_direction(contribs: Iterable[Contribution], effect: DecisionEffect, target: str = "") -> float:
    """Weight-weighted mean signed nudge on one knob/target, 0.0 when nothing pushes (never NaN)."""
    cs = [c for c in contribs if c.effect == effect and c.target == target and c.weight > 0]
    tot = sum(c.weight for c in cs)
    return sum(c.weight * c.signed for c in cs) / tot if tot > 0 else 0.0


def dump_log(log: "DecisionLog") -> list[dict]:
    """Plain rows for persistence; `load_log` re-verifies the chain and refuses a log that does not verify."""
    return [dataclasses.asdict(r) for r in log.rows()]


def load_log(rows: Iterable[Mapping[str, Any]]) -> "DecisionLog":
    log = DecisionLog()
    for r in rows:
        r = dict(r)
        r["effects"] = tuple(r["effects"])
        log._rows.append(LoggedInfluence(**r))
    problems = log.verify()
    if problems:
        raise ContractError("decision log failed verification: " + "; ".join(problems))
    return log


# ---------------------------------------------------------------- one policy object for the thresholds

@dataclasses.dataclass(frozen=True)
class ContractPolicy:
    """Every number the production check uses, in one validated place so a report can print exactly what was enforced."""
    min_truth: float = 0.0            # truth confidence is recorded and must be MEASURED; its floor is left to the epistemic state
    min_usefulness: float = 0.5
    max_failure_risk: float = 0.5
    allow_degraded_in_production: bool = True

    def check(self) -> list[str]:
        errs = []
        for n in ("min_truth", "min_usefulness", "max_failure_risk"):
            v = getattr(self, n)
            if not isinstance(v, (int, float)) or math.isnan(v) or not 0.0 <= v <= 1.0:
                errs.append(f"{n}={v!r} outside [0,1]")
        return errs

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_POLICY = ContractPolicy()


def policy_check(k: Any, now, mode: Mode = Mode.PRODUCTION, policy: ContractPolicy = DEFAULT_POLICY) -> Verdict:
    """`check` under an explicit policy. A DEGRADED item may be excluded from production by policy; the stricter of
    the policy and the item's own action_policy floors applies. Errors in the policy itself are raised, never ignored."""
    bad = policy.check()
    if bad:
        raise ContractError("invalid contract policy: " + "; ".join(bad))
    v = check(k, now, mode, policy.min_usefulness)
    reasons = list(v.reasons)
    conf = getattr(k, "confidence", None)
    if mode != Mode.RESEARCH and conf is not None:
        truth = getattr(conf, "truth", None)
        if truth is not None and truth < policy.min_truth:
            reasons.append(f"confidence.truth={truth:.2f} below policy floor {policy.min_truth:.2f}")
        fr = getattr(conf, "failure_risk", None)
        if fr is not None and fr > policy.max_failure_risk:
            reasons.append(f"failure_risk {fr:.2f} above policy cap {policy.max_failure_risk:.2f}")
    if (mode == Mode.PRODUCTION and not policy.allow_degraded_in_production
            and getattr(k, "epistemic", None) == Epistemic.DEGRADED):
        reasons.append("policy excludes DEGRADED knowledge from production")
    reasons = list(dict.fromkeys(reasons))
    tightened = len(reasons) > len(v.reasons)
    if tightened and v.allowed:
        return dataclasses.replace(v, allowed=False, effects=(), reasons=tuple(reasons))
    return dataclasses.replace(v, reasons=tuple(reasons))


# ---------------------------------------------------------------- what began and stopped influencing decisions

@dataclasses.dataclass(frozen=True)
class ProductionChange:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]
    why_removed: tuple[tuple[str, str], ...]

    def is_stable(self) -> bool:
        return not self.added and not self.removed


def _as_of(items: Iterable[Any], now) -> list[Any]:
    """Newest version of each id that already existed before `now` (items may include several versions of one id)."""
    best: dict[str, Any] = {}
    for k in items:
        try:
            if as_date(k.updated_at) >= as_date(now):
                continue
        except (AttributeError, ValueError):
            continue
        cur = best.get(str(k.knowledge_id))
        if cur is None or getattr(k, "version", 0) > getattr(cur, "version", 0):
            best[str(k.knowledge_id)] = k
    return [best[i] for i in sorted(best)]


def production_changes(items: Iterable[Any], t0, t1, mode: Mode = Mode.PRODUCTION) -> ProductionChange:
    """Which items were allowed at t0 versus t1 (pass every version; each date sees the newest one that existed before
    it). Knowledge that silently stopped being allowed
    (retired, aged out of visibility, degraded) or began to be allowed is listed with the first reason it was refused."""
    if as_date(t1) < as_date(t0):
        raise FirewallBreach("t1 precedes t0")
    items = list(items)
    a = {v.knowledge_id: v for _, v in split(_as_of(items, t0), t0, mode)[0]}
    later, refused = split(_as_of(items, t1), t1, mode)
    b = {v.knowledge_id: v for _, v in later}
    why = tuple(sorted((v.knowledge_id, v.reasons[0] if v.reasons else "") for _, v in refused if v.knowledge_id in a))
    return ProductionChange(tuple(sorted(set(b) - set(a))), tuple(sorted(set(a) - set(b))), tuple(sorted(set(a) & set(b))), why)
