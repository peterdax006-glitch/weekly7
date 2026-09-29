"""Knowledge retrieval (contract C62 section 17; uses sections 8, 14, 16, 42-46; canon C56, C58-C61, C63).

Given a Situation and `now`, rank the knowledge objects that should influence the decision, on ten explicit factors:
    situation_similarity, context_match, temporal_relevance, current_reliability, transfer_confidence, modernity,
    sample_quality, failure_safety                       (positive evidence, weighted)
    contradiction_penalty, redundancy_penalty            (multiplicative penalties)
and say why, in the section-17 form:
    Retrieved K123 because:
    - situation similarity = ...
    - context match = ...
    - current reliability = ...
    - transfer evidence = ...
    - failure risk = ...

Rules this module enforces:
* FAIL CLOSED on the future: an item whose provenance could not exist at `now`, or a support case whose outcome had not
  matured strictly before `now`, raises FirewallBreach (or, if `on_future='exclude'`, is dropped and logged) - never silently used;
* an unmeasured factor is NOT scored as good or bad: it takes a conservative prior, is listed as untested, and caps the
  score (section 42: unknown must not become false confidence);
* a context that does not hold (or an anti-context that does) removes the item; an unobservable context is flagged, not assumed;
* retired / contradicted / gated items and items whose promotion level may not influence decisions are rejected with a reason;
* redundant items are discounted against the better-ranked twin, so five copies of one idea do not fill the top five;
* every result carries a deterministic id, per-factor values and a text explanation, and there are diagnostics
  (`ablate`, `stability`, `factor_report`) to test that the ranking is not driven by one factor or by noise.

Works on KnowledgeLike duck types (engine.learning.core): knowledge_id, version, epistemic, lifecycle, promotion, confidence,
provenance, contexts, anti_contexts, decision_effect; optional temporal_class, pattern_id, n_support, n_effective.

Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .context import context_gate, contexts_from_knowledge
from .core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, KnowledgeLike, Lifecycle, Promotion, TemporalClass,
                   Unknown, UnknownAction, as_date, require_past, stable_hash)
from .similarity import DEFAULT as DEFAULT_SIMILARITY
from .similarity import SimilarityWeights, SituationMatrix
from .situation import Situation, BLOCK_SPECS

POSITIVE_FACTORS = ("situation_similarity", "context_match", "temporal_relevance", "current_reliability", "transfer_confidence",
                    "modernity", "sample_quality", "failure_safety")
PENALTIES = ("contradiction_penalty", "redundancy_penalty")
FACTORS = POSITIVE_FACTORS + PENALTIES

DEFAULT_FACTOR_WEIGHTS = {"situation_similarity": 0.22, "context_match": 0.14, "temporal_relevance": 0.08, "current_reliability": 0.16,
                          "transfer_confidence": 0.12, "modernity": 0.05, "sample_quality": 0.08, "failure_safety": 0.15}
BLOCKED_EPISTEMIC = (Epistemic.RETIRED, Epistemic.CONTRADICTED, Epistemic.GATED)
LIFECYCLE_SAFETY = {Lifecycle.FAILURE: 0.2, Lifecycle.DEGRADED: 0.5, Lifecycle.DECAY: 0.8, Lifecycle.DORMANT: 0.6,
                    Lifecycle.RECOVERY: 0.8, Lifecycle.RETIRED: 0.0}
EPISTEMIC_SAFETY = {Epistemic.DEGRADED: 0.6, Epistemic.HYPOTHESIS: 0.7, Epistemic.OBSERVED: 0.8, Epistemic.UNKNOWN: 0.5}


@dataclasses.dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    min_similarity: float = 0.25          # support-based similarity below this rejects an item as "not this kind of situation"
    support_top_m: int = 3                # support cases averaged for situation_similarity
    unknown_prior: float = 0.25           # value an unmeasured positive factor takes (conservative, and flagged)
    unknown_cap_per_factor: float = 0.06  # score is reduced by this per untested factor
    contradiction_strength: float = 0.7   # how much a full contradiction can reduce a score
    redundancy_strength: float = 0.8
    modernity_half_life_years: float = 5.0
    n_sample_half: float = 30.0           # support count giving sample_quality 0.5
    allowed_promotions: tuple[Promotion, ...] = (Promotion.CHAMPION,)
    on_future: str = "raise"              # raise | exclude
    require_active_pattern: bool = False
    allow_blocked: bool = False           # research mode: also rank retired / contradicted / gated items (flagged)
    novelty_check: bool = False           # abstain when the situation is unlike every stored support case

    def validate(self) -> list[str]:
        errs = []
        if self.top_k < 1:
            errs.append("top_k < 1")
        if not 0.0 <= self.min_similarity <= 1.0:
            errs.append("min_similarity outside [0, 1]")
        if self.on_future not in ("raise", "exclude"):
            errs.append("on_future must be raise or exclude")
        if not 0.0 <= self.unknown_prior <= 1.0 or self.modernity_half_life_years <= 0 or self.n_sample_half <= 0:
            errs.append("bad prior / half-life")
        return errs


@dataclasses.dataclass(frozen=True)
class RetrievalWeights:
    values: tuple[tuple[str, float], ...] = tuple(DEFAULT_FACTOR_WEIGHTS.items())

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def validate(self) -> list[str]:
        d = self.as_dict()
        errs = []
        if set(d) != set(POSITIVE_FACTORS):
            errs.append(f"weights must name exactly {POSITIVE_FACTORS}")
        if any((not math.isfinite(v)) or v < 0 for v in d.values()) or sum(d.values()) <= 0:
            errs.append("weights must be finite, non-negative and not all zero")
        return errs

    def normalised(self) -> dict[str, float]:
        d = self.as_dict()
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}

    def without(self, factor: str) -> "RetrievalWeights":
        return RetrievalWeights(tuple((k, 0.0 if k == factor else v) for k, v in self.values))


# ------------------------------------------------------------------------------------------------ index

@dataclasses.dataclass(frozen=True)
class SupportCase:
    """A situation in which a knowledge item was observed. `ref` is an opaque case number (no ticker, no date)."""
    situation: Situation
    matured: Any                          # trusted-side date the outcome became known
    outcome: float | None
    ref: str


class KnowledgeIndex:
    """Knowledge items + the situations they were observed in + contradiction / redundancy relations."""

    def __init__(self):
        self._items: dict[str, Any] = {}
        self._support: dict[str, list[SupportCase]] = {}
        self._matrix: dict[str, SituationMatrix] = {}
        self._contradicts: dict[str, dict[str, float]] = {}
        self._redundant: dict[str, dict[str, float]] = {}

    def add_item(self, item: Any) -> bool:
        """Insert or upgrade. An older version never replaces a newer one. Returns True if the index changed."""
        missing = KnowledgeLike.conforms(item)
        if missing:
            raise TypeError(f"not a KnowledgeLike: {missing}")
        kid = str(item.knowledge_id)
        cur = self._items.get(kid)
        if cur is not None and int(cur.version) >= int(item.version):
            return False
        self._items[kid] = item
        return True

    def add_support(self, kid: str, situation: Situation, matured, outcome: float | None = None, ref: str | None = None) -> None:
        if kid not in self._items:
            raise KeyError(f"unknown knowledge {kid!r}: add the item first")
        errs = situation.validate()
        if errs:
            raise ValueError("invalid support situation: " + "; ".join(errs[:3]))
        lst = self._support.setdefault(kid, [])
        lst.append(SupportCase(situation, matured, None if outcome is None else float(outcome), ref or f"{kid}#{len(lst)}"))
        self._matrix.pop(kid, None)

    def support(self, kid: str) -> tuple[SupportCase, ...]:
        return tuple(self._support.get(kid, ()))

    def matrix(self, kid: str, cases: Sequence[SupportCase]) -> SituationMatrix:
        key = kid
        m = self._matrix.get(key)
        if m is None or len(m) != len(cases):
            m = self._matrix[key] = SituationMatrix([c.situation for c in cases])
        return m

    def set_contradiction(self, a: str, b: str, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength outside [0, 1]")
        self._contradicts.setdefault(a, {})[b] = strength
        self._contradicts.setdefault(b, {})[a] = strength

    def set_redundancy(self, a: str, b: str, overlap: float) -> None:
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap outside [0, 1]")
        self._redundant.setdefault(a, {})[b] = overlap
        self._redundant.setdefault(b, {})[a] = overlap

    def contradictions(self, kid: str) -> Mapping[str, float]:
        return self._contradicts.get(kid, {})

    def redundancies(self, kid: str) -> Mapping[str, float]:
        return self._redundant.get(kid, {})

    def get(self, kid: str) -> Any:
        return self._items[kid]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def items(self) -> list[Any]:
        return [self._items[k] for k in self.ids()]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, kid: str) -> bool:
        return kid in self._items


# ------------------------------------------------------------------------------------------------ results

@dataclasses.dataclass(frozen=True)
class FactorValue:
    name: str
    value: float | None                   # None = untested (took the prior in the score)
    note: str = ""


@dataclasses.dataclass(frozen=True)
class RetrievedItem:
    knowledge_id: str
    version: int
    score: float
    base_score: float
    factors: tuple[FactorValue, ...]
    untested: tuple[str, ...]
    flags: tuple[str, ...]
    rank: int = 0
    expected_edge: float | None = None       # similarity-weighted outcome of this item's own support cases (None = no outcomes)
    expected_n: int = 0

    def factor(self, name: str) -> float | None:
        for f in self.factors:
            if f.name == name:
                return f.value
        raise KeyError(name)

    def explain(self) -> str:
        def fmt(name, label):
            v = self.factor(name)
            note = next(f.note for f in self.factors if f.name == name)
            return f"- {label} = {'UNTESTED' if v is None else format(v, '.2f')}" + (f" ({note})" if note else "")
        fr = self.factor("failure_safety")
        lines = [f"Retrieved {self.knowledge_id} because:",
                 fmt("situation_similarity", "situation similarity"), fmt("context_match", "context match"),
                 fmt("current_reliability", "current reliability"), fmt("transfer_confidence", "transfer evidence"),
                 f"- failure risk = {'UNTESTED' if fr is None else format(1 - fr, '.2f')}"]
        for n, label in (("temporal_relevance", "temporal relevance"), ("modernity", "modernity"), ("sample_quality", "sample quality")):
            lines.append(fmt(n, label))
        for n in PENALTIES:
            v = self.factor(n)
            if v:
                lines.append(f"- {n.replace('_', ' ')} = -{v:.2f}: " + next(f.note for f in self.factors if f.name == n))
        lines.append(f"= score {self.score:.3f}" + (f" (untested: {', '.join(self.untested)})" if self.untested else ""))
        if self.flags:
            lines.append("flags: " + ", ".join(self.flags))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"knowledge_id": self.knowledge_id, "version": self.version, "score": self.score, "rank": self.rank,
                "factors": {f.name: f.value for f in self.factors}, "untested": list(self.untested), "flags": list(self.flags)}


@dataclasses.dataclass(frozen=True)
class Rejection:
    knowledge_id: str
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Retrieval:
    retrieval_id: str
    situation_id: str
    items: tuple[RetrievedItem, ...]
    rejected: tuple[Rejection, ...]
    unknown: Unknown | None
    action: UnknownAction | None          # what to do when nothing was retrieved or retrieval may not influence
    weights_id: str
    influence: bool = True                # False = shown for research/audit but MUST NOT change a decision
    skill_status: str = "UNMONITORED"     # UNMONITORED | PROVEN | UNPROVEN | INSUFFICIENT_EVIDENCE | FAILED
    withheld_reason: str = ""
    novel: bool | None = None

    def ids(self) -> tuple[str, ...]:
        return tuple(i.knowledge_id for i in self.items)

    def explain(self) -> str:
        if not self.items:
            return f"nothing retrieved ({self.unknown}); action {self.action}; " + "; ".join(
                f"{r.knowledge_id}: {'/'.join(r.reasons)}" for r in self.rejected[:5])
        head = "" if self.influence else f"WITHHELD - these may not influence a decision: {self.withheld_reason}\n\n"
        return head + "\n\n".join(i.explain() for i in self.items)

    def record(self, now) -> dict:
        """Loggable record for later credit assignment (contains `now`, so it is trusted-side only)."""
        return {"retrieval_id": self.retrieval_id, "now": str(as_date(now)), "situation_id": self.situation_id,
                "items": [i.as_dict() for i in self.items], "rejected": [(r.knowledge_id, list(r.reasons)) for r in self.rejected]}


# ------------------------------------------------------------------------------------------------ factor functions

def _age_years(when, now) -> float:
    return max((as_date(now) - as_date(when)).days, 0) / 365.25


def modernity(item: Any, now, half_life_years: float) -> float:
    """How recent the newest evidence the item has seen is: 0.5 ** (age / half-life)."""
    seen = item.provenance.outcomes_seen_through or item.provenance.learned_at
    return float(0.5 ** (_age_years(seen, now) / half_life_years))


def sample_quality(item: Any, n_support: int, n_half: float) -> tuple[float | None, str]:
    n = getattr(item, "n_support", None)
    n = n_support if n is None else int(n)
    if n <= 0:
        return None, "no sample size recorded"
    neff = getattr(item, "n_effective", None)
    frac = 1.0 if neff is None else max(min(float(neff) / max(n, 1), 1.0), 0.05)
    return float(n / (n + n_half) * frac), f"n={n}" + ("" if neff is None else f", n_eff={float(neff):.0f}")


def failure_safety(item: Any) -> tuple[float | None, str]:
    fr = item.confidence.failure_risk
    if fr is None:
        return None, "failure risk never measured"
    s = 1.0 - float(fr)
    s *= LIFECYCLE_SAFETY.get(item.lifecycle, 1.0)
    s *= EPISTEMIC_SAFETY.get(item.epistemic, 1.0)
    return max(min(s, 1.0), 0.0), f"risk {fr:.2f}, lifecycle {item.lifecycle}, epistemic {item.epistemic}"


TEMPORAL_HALF_LIFE_DAYS = {TemporalClass.PERSISTENT: 3650.0, TemporalClass.SLOW_DECAY: 1095.0, TemporalClass.FAST_DECAY: 90.0,
                           TemporalClass.EPISODIC: 365.0}


def temporal_relevance(item: Any, sit: Situation, now, support: Sequence[SupportCase]) -> tuple[float | None, str]:
    """Is the item's kind of knowledge still expected to apply now? Depends on its TemporalClass (section 14)."""
    tc = TemporalClass.parse(getattr(item, "temporal_class", TemporalClass.UNKNOWN))
    seen = item.provenance.outcomes_seen_through or item.provenance.learned_at
    age = max((as_date(now) - as_date(seen)).days, 0)
    if tc in TEMPORAL_HALF_LIFE_DAYS:
        return float(0.5 ** (age / TEMPORAL_HALF_LIFE_DAYS[tc])), f"{tc}, evidence {age} days old"
    if tc == TemporalClass.REGIME_BOUND:
        reg = sit.get("regime.label")
        ctx, _ = contexts_from_knowledge(item)
        cond = next((c for c in ctx.conditions if c.path == "regime.label"), None)
        if cond is None:
            return 0.4, "regime-bound but no regime condition recorded"
        if reg is None or reg == "unknown":
            return None, "current regime unknown"
        return (1.0 if reg in cond.allowed else 0.0), f"regime-bound, now {reg}"
    if tc == TemporalClass.EVENT_BOUND:
        kind = sit.get("shock.kind")
        if kind is None:
            return None, "event state unobserved"
        return (0.2 if kind == "none" else 1.0), f"event-bound, shock {kind}"
    if tc == TemporalClass.SEASONAL:
        phase = sit.get("seasonality.month_of_quarter")
        if phase is None:
            return None, "seasonality not in this situation"
        seen_ph = [c.situation.get("seasonality.month_of_quarter") for c in support]
        seen_ph = [p for p in seen_ph if p is not None]
        if not seen_ph:
            return None, "no seasonal phase in support cases"
        return (float(np.mean([p == phase for p in seen_ph])), "seasonal phase match rate")
    return 0.4, "temporal class unknown"


def situation_similarity(index: KnowledgeIndex, kid: str, sit: Situation, now, cfg: RetrievalConfig,
                         sim_weights: SimilarityWeights) -> tuple[float | None, str, int]:
    """Mean of the top-m comparable support cases' similarity (scaled down when fewer than m are comparable)."""
    cases = index.support(kid)
    if not cases:
        return None, "no recorded support situations", 0
    for c in cases:
        require_past(c.matured, now, f"support case {c.ref} of {kid}")
    M = index.matrix(kid, cases)
    tot, _, ok = M.totals(sit, sim_weights)
    good = sorted((float(t) for t, o in zip(tot, ok) if o and not math.isnan(t)), reverse=True)
    if not good:
        return 0.0, f"none of {len(cases)} support cases comparable", 0
    top = good[: cfg.support_top_m]
    val = float(np.mean(top)) * math.sqrt(len(top) / cfg.support_top_m)
    return val, f"best {top[0]:.2f} of {len(good)} comparable support cases", len(good)


def context_match(item: Any, sit: Situation) -> tuple[float | None, str, bool | None]:
    ctx, anti = contexts_from_knowledge(item)
    ok, why = context_gate(sit, item)
    if ok is False:
        return 0.0, why, False
    if ok is None:
        return None, why, None
    if not ctx.conditions and not anti.conditions:
        return 0.6, "unconditional knowledge (applies everywhere, says nothing specific)", True
    return 1.0, why, True


# ------------------------------------------------------------------------------------------------ retriever

class Retriever:
    def __init__(self, index: KnowledgeIndex, weights: RetrievalWeights | None = None, config: RetrievalConfig | None = None,
                 sim_weights: SimilarityWeights = DEFAULT_SIMILARITY, monitor: "SkillMonitor | None" = None):
        self.monitor = monitor
        self.index = index
        self.weights = weights or RetrievalWeights()
        self.cfg = config or RetrievalConfig()
        self.sim_weights = sim_weights
        for errs in (self.weights.validate(), self.cfg.validate(), sim_weights.validate()):
            if errs:
                raise ValueError("; ".join(errs))

    def _weights_id(self) -> str:
        return stable_hash({"w": self.weights.values, "s": self.sim_weights.weights_id(), "c": dataclasses.asdict(self.cfg)})

    # ---- gates
    def _admissible(self, item: Any, sit: Situation, now, effect: DecisionEffect | None) -> list[str]:
        cfg, why = self.cfg, []
        if not item.provenance.could_exist_at(now):
            if cfg.on_future == "raise":
                raise FirewallBreach(f"knowledge {item.knowledge_id} could not have existed at now={as_date(now)} "
                                     f"(learned {item.provenance.learned_at}, seen through {item.provenance.outcomes_seen_through})")
            why.append("future: could not exist at now")
        if item.epistemic in BLOCKED_EPISTEMIC and not cfg.allow_blocked:
            why.append(f"epistemic={item.epistemic}")
        if item.promotion not in cfg.allowed_promotions and not cfg.allow_blocked:
            why.append(f"promotion={item.promotion} may not influence decisions")
        if effect is not None and effect not in item.decision_effect:
            why.append(f"does not change {effect}")
        if not item.decision_effect or all(e == DecisionEffect.NONE for e in item.decision_effect):
            if not cfg.allow_blocked:
                why.append("research knowledge only (decision_effect NONE)")
        pid = getattr(item, "pattern_id", None)
        if cfg.require_active_pattern and pid is not None and pid not in sit.pattern_ids:
            why.append("its pattern is not active in this situation")
        return why

    # ---- scoring
    def _factors(self, item: Any, sit: Situation, now) -> tuple[dict[str, FactorValue], list[str], bool | None]:
        cfg, kid = self.cfg, str(item.knowledge_id)
        support = self.index.support(kid)
        sim, sim_note, n_comp = situation_similarity(self.index, kid, sit, now, cfg, self.sim_weights)
        cm, cm_note, applies = context_match(item, sit)
        tr, tr_note = temporal_relevance(item, sit, now, support)
        sq, sq_note = sample_quality(item, len(support), cfg.n_sample_half)
        fs, fs_note = failure_safety(item)
        conf: Confidence = item.confidence
        f = {
            "situation_similarity": FactorValue("situation_similarity", sim, sim_note),
            "context_match": FactorValue("context_match", cm, cm_note),
            "temporal_relevance": FactorValue("temporal_relevance", tr, tr_note),
            "current_reliability": FactorValue("current_reliability", conf.current_reliability, "" if conf.current_reliability is not None else "never measured"),
            "transfer_confidence": self._transfer_factor(item, kid, now),
            "modernity": FactorValue("modernity", modernity(item, now, cfg.modernity_half_life_years), "recency of newest evidence"),
            "sample_quality": FactorValue("sample_quality", sq, sq_note),
            "failure_safety": FactorValue("failure_safety", fs, fs_note),
        }
        return f, [], applies

    def _transfer_factor(self, item: Any, kid: str, now) -> FactorValue:
        """The item's recorded transfer confidence, else one derived from its own support outcomes (flagged as derived)."""
        if item.confidence.transfer is not None:
            return FactorValue("transfer_confidence", item.confidence.transfer, "recorded")
        te = transfer_evidence(self.index, kid, now, self.sim_weights)
        if te["verdict"] in ("TRANSFERS", "NARROW", "FAILS_TO_TRANSFER"):
            return FactorValue("transfer_confidence", te["score"], f"derived from support cases: {te['verdict']}, far/near edge {te['ratio']:.2f}")
        return FactorValue("transfer_confidence", None, f"never measured ({te['verdict']})")

    def _contradiction(self, kid: str, reliability: dict[str, float], pool: set[str]) -> tuple[float, str]:
        best, who = 0.0, ""
        me = reliability.get(kid, 0.5)
        for other, s in self.index.contradictions(kid).items():
            if other not in pool:
                continue
            them = reliability.get(other, 0.5)
            p = s * them / (me + them + 1e-9)               # the more credible the opponent, the larger the penalty
            if p > best:
                best, who = p, other
        return best, (f"contradicted by {who}" if who else "")

    def _base_score(self, factors: Mapping[str, FactorValue]) -> tuple[float, list[str]]:
        w = self.weights.normalised()
        untested = [n for n in POSITIVE_FACTORS if factors[n].value is None and w[n] > 0]
        tot = sum(w[n] * (factors[n].value if factors[n].value is not None else self.cfg.unknown_prior) for n in POSITIVE_FACTORS)
        return tot - self.cfg.unknown_cap_per_factor * len(untested), untested

    def retrieve(self, sit: Situation, now, decision_effect: DecisionEffect | None = None, k: int | None = None) -> Retrieval:
        errs = sit.validate()
        if errs:
            raise ValueError("cannot retrieve for an invalid situation: " + "; ".join(errs[:3]))
        cfg = self.cfg
        k = cfg.top_k if k is None else k
        rejected: list[Rejection] = []
        cand: list[tuple[str, dict[str, FactorValue], float, list[str], list[str]]] = []
        reliability: dict[str, float] = {}
        for item in self.index.items():
            kid = str(item.knowledge_id)
            reliability[kid] = item.confidence.current_reliability if item.confidence.current_reliability is not None else 0.5
            why = self._admissible(item, sit, now, decision_effect)
            if why:
                rejected.append(Rejection(kid, tuple(why)))
                continue
            try:
                factors, _, applies = self._factors(item, sit, now)
            except FirewallBreach:
                if cfg.on_future == "raise":
                    raise
                rejected.append(Rejection(kid, ("future: a support case had not matured",)))
                continue
            if applies is False:
                rejected.append(Rejection(kid, (factors["context_match"].note,)))
                continue
            sim = factors["situation_similarity"].value
            if sim is not None and sim < cfg.min_similarity:
                rejected.append(Rejection(kid, (f"situation similarity {sim:.2f} < {cfg.min_similarity:.2f}: {factors['situation_similarity'].note}",)))
                continue
            base, untested = self._base_score(factors)
            flags = []
            if item.epistemic in BLOCKED_EPISTEMIC:
                flags.append(f"BLOCKED-{item.epistemic}")
            if applies is None:
                flags.append("context-unobservable")
            cand.append((kid, factors, base, untested, flags))
        pool = {c[0] for c in cand}
        scored = []
        for kid, factors, base, untested, flags in cand:
            pen, note = self._contradiction(kid, reliability, pool)
            factors = dict(factors)
            factors["contradiction_penalty"] = FactorValue("contradiction_penalty", round(pen, 6), note)
            scored.append([kid, factors, base * (1 - cfg.contradiction_strength * pen), base, untested, flags])
        scored.sort(key=lambda r: (-r[2], r[0]))
        chosen: list[list] = []
        for row in scored:                                   # greedy: discount against better-ranked redundant twins
            kid = row[0]
            red = max(((ov, o) for o, ov in self.index.redundancies(kid).items() if o in {c[0] for c in chosen}), default=(0.0, ""))
            row[1]["redundancy_penalty"] = FactorValue("redundancy_penalty", round(red[0], 6), f"redundant with {red[1]}" if red[1] else "")
            row[2] = row[2] * (1 - cfg.redundancy_strength * red[0])
            chosen.append(row)
        chosen.sort(key=lambda r: (-r[2], r[0]))
        built = []
        for i, r in enumerate(chosen[:k]):
            ex = support_expectation(self.index, r[0], sit, now, self.sim_weights)
            built.append(RetrievedItem(r[0], int(self.index.get(r[0]).version), round(float(r[2]), 6), round(float(r[3]), 6),
                                       tuple(r[1][n] for n in FACTORS), tuple(r[4]), tuple(r[5]), rank=i + 1,
                                       expected_edge=ex["mean"], expected_n=ex["n"]))
        items = tuple(built)
        unknown = None if items else (Unknown.UNTESTED if len(self.index) == 0 else Unknown.INSUFFICIENT_DATA)
        action = None if items else (UnknownAction.USE_GENERAL_RULE if len(self.index) else UnknownAction.COLLECT_DATA)
        influence, status, why, novel = True, "UNMONITORED", "", None
        if items:
            if self.monitor is not None:
                sk = self.monitor.status(now)
                status = sk["status"]
                if status != "PROVEN":
                    influence, why = False, f"walk-forward retrieval skill is {status} (n={sk['n']}, mean edge {sk['mean_edge']})"
            if cfg.novelty_check:
                nov = self._novelty(sit)
                novel = nov["novel"]
                if novel:
                    influence, why = False, (why + "; " if why else "") + f"novel situation: best support match {nov['best']:.2f} below the reference {nov['reference_p05']:.2f}"
            if not influence:
                action, unknown = UnknownAction.ABSTAIN, Unknown.UNKNOWN
        rid = stable_hash({"s": sit.exact_id, "n": str(as_date(now)), "w": self._weights_id(), "inf": influence,
                           "k": [(i.knowledge_id, i.version, i.score) for i in items]})
        return Retrieval(rid, sit.situation_id, items, tuple(sorted(rejected, key=lambda r: r.knowledge_id)), unknown, action,
                         self._weights_id(), influence, status, why, novel)

    def _novelty(self, sit: Situation) -> dict:
        from .similarity import novelty
        pooled, seen = [], set()
        for kid in self.index.ids():
            for c in self.index.support(kid):
                if c.situation.exact_id not in seen:
                    seen.add(c.situation.exact_id)
                    pooled.append(c.situation)
        return novelty(sit, pooled, self.sim_weights, n_ref=20)

    # ---- diagnostics
    def ablate(self, sit: Situation, now, factor: str, k: int | None = None) -> dict[str, Any]:
        """Re-rank with one positive factor's weight set to zero. Reports how much the top-k moves (rank correlation, overlap)."""
        if factor not in POSITIVE_FACTORS:
            raise ValueError(f"cannot ablate {factor!r}; choose from {POSITIVE_FACTORS}")
        full = self.retrieve(sit, now, k=k)
        alt = Retriever(self.index, self.weights.without(factor), self.cfg, self.sim_weights).retrieve(sit, now, k=k)
        a, b = full.ids(), alt.ids()
        common = [x for x in a if x in b]
        tau = None
        if len(common) >= 3:
            from scipy.stats import kendalltau
            tau = float(kendalltau([a.index(x) for x in common], [b.index(x) for x in common])[0])
        return {"factor": factor, "overlap": len(common) / max(len(a), 1), "kendall_tau": tau, "top1_same": bool(a and b and a[0] == b[0]),
                "full": a, "ablated": b}

    def factor_report(self, situations: Sequence[Situation], now) -> dict[str, dict[str, float]]:
        """Across many queries: per factor mean, spread and share untested among retrieved items, plus ablation sensitivity.
        A factor that is always identical carries no ranking information; one that is mostly untested is not yet earning its weight."""
        vals: dict[str, list[float | None]] = {f: [] for f in FACTORS}
        for s in situations:
            for it in self.retrieve(s, now).items:
                for f in FACTORS:
                    vals[f].append(it.factor(f))
        rep = {}
        for f, v in vals.items():
            got = [x for x in v if x is not None]
            rep[f] = {"n": len(v), "untested_share": 0.0 if not v else 1 - len(got) / len(v),
                      "mean": float(np.mean(got)) if got else float("nan"), "std": float(np.std(got)) if got else float("nan")}
        return rep

    def stability(self, sit: Situation, now, seed: int = 0, trials: int = 10, eps: float = 0.05, k: int | None = None) -> float:
        """Mean top-k overlap between the retrieval for `sit` and for slightly perturbed copies (numeric fields nudged by
        eps * scale). Near 1 means the ranking is not hostage to noise; a low value is a defect to investigate."""
        rng = np.random.default_rng(seed)
        base = self.retrieve(sit, now, k=k).ids()
        if not base:
            return 1.0
        ov = []
        for _ in range(trials):
            ov.append(len(set(base) & set(self.retrieve(perturb(sit, rng, eps), now, k=k).ids())) / len(base))
        return float(np.mean(ov))


def perturb(sit: Situation, rng: np.random.Generator, eps: float = 0.05) -> Situation:
    """A copy of `sit` with every observed numeric field shifted by N(0, eps * field scale), clipped to its sanity range."""
    from .situation import Block
    blocks = []
    for b in sit.blocks:
        vals = {}
        for spec, (name, v) in zip(BLOCK_SPECS[b.kind], b.values):
            if v is not None and spec.kind == "num":
                x = v + float(rng.normal(0.0, eps * spec.scale))
                if spec.lo is not None:
                    x = max(x, spec.lo)
                if spec.hi is not None:
                    x = min(x, spec.hi)
                v = x
            vals[name] = v
        blocks.append(Block.make(b.kind, **vals))
    return dataclasses.replace(sit, blocks=tuple(blocks))


# ------------------------------------------------------------------------------------------------ scoring retrievals

def score_retrieval(retrieval: Retrieval, outcomes: Mapping[str, float]) -> dict[str, Any]:
    """After outcomes are known: was the retrieved knowledge's expected sign right? `outcomes` maps knowledge_id -> realised edge
    of the decision that item drove. Returns hit rate and the score-weighted mean edge (for credit assignment, section 20)."""
    got = [(i, outcomes[i.knowledge_id]) for i in retrieval.items if i.knowledge_id in outcomes]
    if not got:
        return {"n": 0, "hit_rate": None, "weighted_edge": None}
    w = np.array([i.score for i, _ in got], dtype=float)
    y = np.array([o for _, o in got], dtype=float)
    return {"n": len(got), "hit_rate": float(np.mean(y > 0)),
            "weighted_edge": float((w * y).sum() / w.sum()) if w.sum() > 0 else float(y.mean())}


def rank_quality(retrievals: Sequence[Retrieval], outcomes: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Does a higher retrieval score go with a better realised edge? Spearman over all (score, edge) pairs pooled."""
    from scipy.stats import spearmanr
    xs, ys = [], []
    for r, o in zip(retrievals, outcomes):
        for i in r.items:
            if i.knowledge_id in o:
                xs.append(i.score)
                ys.append(o[i.knowledge_id])
    if len(xs) < 8 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return {"n": len(xs), "spearman": float("nan")}
    rho = spearmanr(xs, ys)[0]
    return {"n": len(xs), "spearman": float(rho)}


# ------------------------------------------------------------------------------------------------ evidence from support cases

def support_expectation(index: KnowledgeIndex, kid: str, sit: Situation, now,
                        sim_weights: SimilarityWeights = DEFAULT_SIMILARITY, bw: float = 0.5) -> dict[str, Any]:
    """What happened in this item's own support cases that resemble `sit`: a kernel-weighted mean outcome
    (engine.analog_weighting.kernel_weights, bandwidth = bw x the median distance), its effective sample size and the
    weighted 10th/90th percentiles. Only cases whose outcome matured strictly before `now` are read (FirewallBreach otherwise)."""
    from .. import analog_weighting as AW
    empty = {"mean": None, "n": 0, "n_eff": 0.0, "q10": None, "q90": None}
    cases = index.support(kid)
    if not cases:
        return empty
    for c in cases:
        require_past(c.matured, now, f"support case {c.ref} of {kid}")
    M = index.matrix(kid, cases)
    tot, _, ok = M.totals(sit, sim_weights)
    sel = [i for i, c in enumerate(cases) if c.outcome is not None and ok[i] and not math.isnan(tot[i])]
    if not sel:
        return empty
    d = 1.0 - tot[sel]
    y = np.array([cases[i].outcome for i in sel], dtype=float)
    ref = float(np.median(d)) if len(d) > 1 else float(d[0])
    w = AW.kernel_weights(d, ref if ref > 1e-9 else 1.0, bw)
    q10, q90 = AW.weighted_quantile(y, w, [0.1, 0.9])
    return {"mean": float((w * y).sum()), "n": len(sel), "n_eff": float(1.0 / (w ** 2).sum()), "q10": q10, "q90": q90}


def transfer_evidence(index: KnowledgeIndex, kid: str, now, sim_weights: SimilarityWeights = DEFAULT_SIMILARITY,
                      min_cases: int = 20) -> dict[str, Any]:
    """Does the item's edge survive away from its core? Support cases are split into the more typical half (highest mean
    similarity to the other cases) and the more peripheral half; the ratio far-edge / near-edge measures how well it
    transfers to situations unlike the typical one. Needs `min_cases` cases with outcomes, else UNTESTED (never a guess)."""
    cases = [c for c in index.support(kid) if c.outcome is not None]
    out = {"verdict": "UNTESTED", "score": None, "ratio": float("nan"), "near_mean": None, "far_mean": None, "n_near": 0, "n_far": 0}
    if len(cases) < min_cases:
        return out
    for c in cases:
        require_past(c.matured, now, f"support case {c.ref} of {kid}")
    M = SituationMatrix([c.situation for c in cases])
    typ = np.zeros(len(cases))
    for i, c in enumerate(cases):
        tot, _, ok = M.totals(c.situation, sim_weights)
        others = [tot[j] for j in range(len(cases)) if j != i and ok[j] and not math.isnan(tot[j])]
        typ[i] = float(np.mean(others)) if others else 0.0
    order = np.argsort(-typ, kind="stable")
    half = len(cases) // 2
    near, far = order[:half], order[len(cases) - half:]
    y = np.array([c.outcome for c in cases], dtype=float)
    nm, fm = float(y[near].mean()), float(y[far].mean())
    out.update(near_mean=nm, far_mean=fm, n_near=int(len(near)), n_far=int(len(far)))
    if nm <= 0:
        out["verdict"] = "NO_EDGE"
        return out
    ratio = fm / nm
    out["ratio"] = float(ratio)
    out["score"] = float(min(max(ratio, 0.0), 1.0))
    out["verdict"] = "TRANSFERS" if ratio >= 0.7 else "NARROW" if ratio >= 0.2 else "FAILS_TO_TRANSFER"
    return out


# ------------------------------------------------------------------------------------------------ walk-forward skill

class SkillMonitor:
    """Walk-forward record of whether retrieval's expected edge predicts the realised edge. Retrieval may FAIL: when its
    measured skill is not positive and proven, `Retriever.retrieve` marks the result `influence=False` and recommends ABSTAIN
    instead of returning neighbours as though they were knowledge. Predictions are recorded when made; an outcome counts only
    from the date it matured, and `status(now)` reads only outcomes matured strictly before `now`."""

    def __init__(self, min_n: int = 40, alpha: float = 0.05, seed: int = 0):
        if min_n < 10:
            raise ValueError("min_n < 10 cannot support a skill claim")
        self.min_n, self.alpha, self.seed = min_n, alpha, seed
        self._pred: dict[str, tuple[Any, float]] = {}
        self._done: list[tuple[Any, float, float]] = []                # (matured, expected, realised)

    def predict(self, retrieval_id: str, made_on, expected: float | None) -> None:
        if expected is None or not math.isfinite(float(expected)):
            return                                                    # nothing was predicted; nothing to score
        if retrieval_id in self._pred:
            raise ValueError(f"prediction {retrieval_id} already recorded")
        self._pred[retrieval_id] = (made_on, float(expected))

    def resolve(self, retrieval_id: str, matured, realised: float) -> None:
        if retrieval_id not in self._pred:
            raise KeyError(f"no prediction recorded for {retrieval_id}")
        made_on, exp = self._pred.pop(retrieval_id)
        if as_date(matured) <= as_date(made_on):
            raise FirewallBreach(f"prediction {retrieval_id}: outcome matured {as_date(matured)} is not after it was made {as_date(made_on)}")
        self._done.append((matured, exp, float(realised)))

    def pending(self) -> int:
        return len(self._pred)

    def status(self, now) -> dict[str, Any]:
        from scipy.stats import spearmanr
        from .. import analog_weighting as AW
        rows = [(e, r) for m, e, r in self._done if as_date(m) < as_date(now)]
        n = len(rows)
        base = {"n": n, "mean_edge": None, "hit_rate": None, "spearman": None, "hac_t": None, "p": None, "status": "INSUFFICIENT_EVIDENCE"}
        if n < self.min_n:
            return base
        e = np.array([x for x, _ in rows])
        r = np.array([y for _, y in rows])
        edge = np.sign(e) * r                                         # realised edge of acting on the prediction
        nz = r != 0
        rho = float(spearmanr(e, r)[0]) if e.std() > 1e-12 and r.std() > 1e-12 else float("nan")
        t = AW.hac_t(edge)
        p = AW.sign_flip_p(edge, seed=self.seed)
        mean_edge = float(edge.mean())
        base.update(mean_edge=round(mean_edge, 6), hit_rate=float(np.mean(np.sign(e[nz]) == np.sign(r[nz]))) if nz.any() else None,
                    spearman=rho, hac_t=float(t), p=float(p))
        if mean_edge <= 0 or (np.isfinite(rho) and rho <= 0):
            base["status"] = "FAILED"
        elif np.isfinite(p) and p < self.alpha:
            base["status"] = "PROVEN"
        else:
            base["status"] = "UNPROVEN"
        return base

    def allows(self, now) -> bool:
        return self.status(now)["status"] == "PROVEN"


# ------------------------------------------------------------------------------------------------ combining & learning

def combine(retrieval: Retrieval, index: KnowledgeIndex | None = None) -> dict[str, Any]:
    """One expected edge from several retrieved items. Weights are the scores discounted by redundancy with better-ranked
    items (so duplicates do not vote twice); disagreement is the weighted spread and the share of weight on the majority
    sign. Items without outcome evidence do not vote. The result carries `influence` from the retrieval it came from."""
    vote = [(i, i.expected_edge) for i in retrieval.items if i.expected_edge is not None]
    out = {"expected": None, "n_votes": len(vote), "disagreement": None, "majority_share": None, "influence": retrieval.influence}
    if not vote:
        return out
    w = []
    for pos, (i, _) in enumerate(vote):
        disc = 1.0
        if index is not None:
            for j, _ in vote[:pos]:
                disc *= 1.0 - index.redundancies(i.knowledge_id).get(j.knowledge_id, 0.0)
        w.append(max(i.score, 0.0) * disc)
    w = np.array(w)
    e = np.array([x for _, x in vote])
    if w.sum() <= 0:
        return out
    m = float((w * e).sum() / w.sum())
    out.update(expected=m, disagreement=float(np.sqrt((w * (e - m) ** 2).sum() / w.sum())),
               majority_share=float(max(w[e > 0].sum(), w[e <= 0].sum()) / w.sum()))
    return out


def fit_retrieval_weights(factor_rows: np.ndarray, outcomes: np.ndarray, prior: RetrievalWeights | None = None,
                          ridge: float = 20.0, min_rows: int = 100) -> tuple[RetrievalWeights, dict[str, float]]:
    """Re-fit the eight positive factor weights so that a higher retrieval score goes with a better realised edge:
    non-negative least squares of outcome on factor values, pulled toward `prior` by `ridge` pseudo-rows. Below `min_rows`
    the prior is returned unchanged. factor_rows: (R, 8) in POSITIVE_FACTORS order with NaN for untested (filled by the
    row's prior value); returns (weights, diagnostics with r2 before / after)."""
    from scipy.optimize import nnls
    prior = prior or RetrievalWeights()
    A, y = np.asarray(factor_rows, float), np.asarray(outcomes, float)
    if A.ndim != 2 or A.shape[1] != len(POSITIVE_FACTORS) or len(A) != len(y):
        raise ValueError("factor_rows must be (R, 8) and match outcomes")
    w0 = np.array([prior.normalised()[f] for f in POSITIVE_FACTORS])
    ok = np.isfinite(y)
    A, y = A[ok], y[ok]
    if len(y) < min_rows:
        return prior, {"n": len(y), "r2_before": float("nan"), "r2_after": float("nan"), "changed": False}
    A = np.where(np.isnan(A), 0.25, A)

    def r2(w):
        pred = A @ w
        if pred.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(pred, y)[0, 1] ** 2)
    lam = math.sqrt(ridge)
    ys = np.std(y) if np.std(y) > 0 else 1.0
    Aa = np.vstack([A, lam * np.eye(A.shape[1]) * 0.1])
    ya = np.concatenate([(y - y.mean()) / ys + 0.5, lam * 0.1 * w0])
    w, _ = nnls(Aa, ya)
    w = w0 if w.sum() <= 0 else w / w.sum()
    new = RetrievalWeights(tuple((f, float(v)) for f, v in zip(POSITIVE_FACTORS, w)))
    return new, {"n": len(y), "r2_before": r2(w0), "r2_after": r2(w), "changed": bool(np.abs(w - w0).max() > 1e-9)}


def factor_matrix(retrievals: Sequence[Retrieval], outcomes: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """(factor rows, realised edges) for every retrieved item whose decision outcome is known, ready for `fit_retrieval_weights`."""
    rows, ys = [], []
    for r, o in zip(retrievals, outcomes):
        for it in r.items:
            if it.knowledge_id in o:
                rows.append([np.nan if it.factor(f) is None else it.factor(f) for f in POSITIVE_FACTORS])
                ys.append(o[it.knowledge_id])
    return (np.array(rows, float).reshape(-1, len(POSITIVE_FACTORS)), np.array(ys, float))


# ------------------------------------------------------------------------------------------------ audit trail

class RetrievalLog:
    """Append-only JSONL log of retrievals (contains `now`: trusted side only). `replay` re-runs a logged retrieval against a
    retriever and checks it reproduces the same id - the determinism / stale-code check for retrieval itself."""

    def __init__(self, path=None):
        import pathlib
        self.path = None if path is None else pathlib.Path(path)
        self._mem: list[dict] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, retrieval: Retrieval, now) -> dict:
        import json
        rec = retrieval.record(now)
        rec["influence"], rec["skill_status"], rec["withheld_reason"] = retrieval.influence, retrieval.skill_status, retrieval.withheld_reason
        rec["code_hash"] = _code_hash()
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, sort_keys=True) + chr(10))
        self._mem.append(rec)
        return rec

    def read(self) -> list[dict]:
        import json
        if self.path is None or not self.path.exists():
            return list(self._mem)
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.read())

    def replay(self, record: Mapping[str, Any], retriever: "Retriever", sit: Situation) -> dict[str, Any]:
        """Re-run the logged retrieval on `sit` at the logged `now`. `match` False means the retriever, the index or the code
        no longer reproduces what was logged; the difference in ids is returned for the audit."""
        again = retriever.retrieve(sit, record["now"])
        return {"match": again.retrieval_id == record["retrieval_id"], "logged": record["retrieval_id"], "now_id": again.retrieval_id,
                "stale_code": stale_code(record),
                "logged_items": [i["knowledge_id"] for i in record["items"]], "now_items": list(again.ids())}


# ------------------------------------------------------------------------------------------------ batch & explanation helpers

def retrieve_many(retriever: Retriever, sits: Sequence[Situation], now, k: int | None = None) -> list[Retrieval]:
    return [retriever.retrieve(s, now, k=k) for s in sits]


def overlap_matrix(retrievals: Sequence[Retrieval]) -> np.ndarray:
    """Jaccard overlap of the retrieved id sets of every pair; rows of identical retrievals give 1. NaN for two empties."""
    n = len(retrievals)
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = set(retrievals[i].ids()), set(retrievals[j].ids())
            if a or b:
                out[i, j] = len(a & b) / len(a | b)
    return out


def why_not(retriever: Retriever, kid: str, sit: Situation, now) -> str:
    """Why is `kid` (not) among the retrieved items for `sit`? Rejection reasons if it was gated out, otherwise its rank and
    the factors on which it trails the top item."""
    full = retriever.retrieve(sit, now, k=max(len(retriever.index), 1))
    for r in full.rejected:
        if r.knowledge_id == kid:
            return f"{kid} was rejected: " + "; ".join(r.reasons)
    if kid not in full.ids():
        return f"{kid} is not in the index" if kid not in retriever.index else f"{kid} produced no candidate"
    me = next(i for i in full.items if i.knowledge_id == kid)
    top = full.items[0]
    if me.knowledge_id == top.knowledge_id:
        return f"{kid} is the top item (score {me.score:.3f})"
    gaps = sorted(((f.name, (top.factor(f.name) or 0.0) - (f.value or 0.0)) for f in me.factors if f.name in POSITIVE_FACTORS),
                  key=lambda kv: -kv[1])[:3]
    return (f"{kid} ranks {me.rank} of {len(full.items)} (score {me.score:.3f} vs top {top.score:.3f}); it trails {top.knowledge_id} most on "
           + ", ".join(f"{n} ({g:+.2f})" for n, g in gaps))


# ------------------------------------------------------------------------------------------------ index persistence

def export_support(index: KnowledgeIndex) -> dict[str, Any]:
    """JSON-safe support cases and relations of an index. Items themselves are re-attached by their owner (they are
    duck-typed objects owned by the knowledge module); ids that no longer exist are dropped on import, never invented."""
    return {"support": {kid: [{"situation": c.situation.to_dict(), "matured": str(as_date(c.matured)), "outcome": c.outcome, "ref": c.ref}
                              for c in index.support(kid)] for kid in index.ids()},
            "contradicts": {k: dict(sorted(v.items())) for k, v in sorted(index._contradicts.items())},
            "redundant": {k: dict(sorted(v.items())) for k, v in sorted(index._redundant.items())}}


def import_support(index: KnowledgeIndex, state: Mapping[str, Any]) -> dict[str, int]:
    counts = {"cases": 0, "skipped_ids": 0}
    for kid, cases in state.get("support", {}).items():
        if kid not in index:
            counts["skipped_ids"] += 1
            continue
        for c in cases:
            index.add_support(kid, Situation.from_dict(c["situation"]), c["matured"], c["outcome"], c["ref"])
            counts["cases"] += 1
    for a, row in state.get("contradicts", {}).items():
        for b, v in row.items():
            if a in index and b in index:
                index.set_contradiction(a, b, v)
    for a, row in state.get("redundant", {}).items():
        for b, v in row.items():
            if a in index and b in index:
                index.set_redundancy(a, b, v)
    return counts


# ------------------------------------------------------------------------------------------------ decision-ready output

@dataclasses.dataclass(frozen=True)
class RetrievalPolicy:
    """Thresholds that turn a Retrieval into use / abstain. Deliberately conservative: nothing retrieved is knowledge
    until it is similar enough, backed by enough outcome cases, in agreement, and allowed to influence."""
    min_top_score: float = 0.45
    min_expected_n: int = 8
    min_majority_share: float = 0.7
    max_disagreement: float = 0.05

    def validate(self) -> list[str]:
        errs = []
        if not 0.0 <= self.min_top_score <= 1.0 or not 0.5 <= self.min_majority_share <= 1.0:
            errs.append("score / majority thresholds out of range")
        if self.min_expected_n < 1 or self.max_disagreement <= 0:
            errs.append("min_expected_n < 1 or max_disagreement <= 0")
        return errs


def decide(retrieval: Retrieval, index: KnowledgeIndex | None = None, policy: RetrievalPolicy | None = None) -> dict[str, Any]:
    """{'action': 'USE' | 'ABSTAIN', 'reasons': [...], 'expected': float | None, ...}. ABSTAIN is the default answer: every
    condition that fails is listed, so a caller (and a later postmortem) can see exactly why retrieval declined to speak."""
    policy = policy or RetrievalPolicy()
    errs = policy.validate()
    if errs:
        raise ValueError("; ".join(errs))
    reasons: list[str] = []
    if not retrieval.items:
        reasons.append(f"nothing retrieved ({retrieval.unknown})")
    if not retrieval.influence:
        reasons.append(f"retrieval may not influence decisions: {retrieval.withheld_reason}")
    comb = combine(retrieval, index)
    if retrieval.items and retrieval.items[0].score < policy.min_top_score:
        reasons.append(f"best score {retrieval.items[0].score:.2f} < {policy.min_top_score:.2f}")
    if retrieval.items and comb["expected"] is None:
        reasons.append("no retrieved item has outcome evidence")
    elif comb["expected"] is not None:
        n_ev = sum(i.expected_n for i in retrieval.items if i.expected_edge is not None)
        if n_ev < policy.min_expected_n:
            reasons.append(f"only {n_ev} outcome cases behind the expectation (< {policy.min_expected_n})")
        if comb["majority_share"] < policy.min_majority_share:
            reasons.append(f"items disagree on direction (majority share {comb['majority_share']:.2f})")
        if comb["disagreement"] > policy.max_disagreement:
            reasons.append(f"items disagree on size (spread {comb['disagreement']:.3f})")
    return {"action": "ABSTAIN" if reasons else "USE", "reasons": reasons, "expected": None if reasons else comb["expected"],
            "candidate_expected": comb["expected"], "n_items": len(retrieval.items), "retrieval_id": retrieval.retrieval_id}


# ------------------------------------------------------------------------------------------------ comparing retrievals

def explain_rank_change(before: Retrieval, after: Retrieval) -> str:
    """After learning (new evidence, new weights), what moved and on which factors? The retrieval half of a learning delta:
    an item that rose because a factor became measured is a different story from one that rose because of noise."""
    b = {i.knowledge_id: i for i in before.items}
    a = {i.knowledge_id: i for i in after.items}
    lines = []
    for kid in sorted(set(b) | set(a)):
        if kid not in b:
            lines.append(f"+ {kid} entered at rank {a[kid].rank} (score {a[kid].score:.3f})")
        elif kid not in a:
            why = next((r.reasons[0] for r in after.rejected if r.knowledge_id == kid), "fell out of the top k")
            lines.append(f"- {kid} left (was rank {b[kid].rank}): {why}")
        elif a[kid].rank != b[kid].rank or abs(a[kid].score - b[kid].score) > 1e-9:
            moved = sorted(((f, (a[kid].factor(f) or 0.0) - (b[kid].factor(f) or 0.0)) for f in FACTORS
                            if (a[kid].factor(f) or 0.0) != (b[kid].factor(f) or 0.0)), key=lambda kv: -abs(kv[1]))[:3]
            lines.append(f"~ {kid} rank {b[kid].rank} -> {a[kid].rank}, score {b[kid].score:.3f} -> {a[kid].score:.3f}"
                         + ("; factors: " + ", ".join(f"{f} {d:+.2f}" for f, d in moved) if moved else ""))
    return "\n".join(lines) or "no change"


def counterfactual(retriever: Retriever, sit: Situation, now, without: Iterable[str], k: int | None = None) -> dict[str, Any]:
    """What would have been retrieved had the given knowledge not existed? Used by credit assignment: the difference
    between the real and the counterfactual retrieval is what those items contributed."""
    drop = set(without)
    ix = KnowledgeIndex()
    for it in retriever.index.items():
        if str(it.knowledge_id) not in drop:
            ix.add_item(it)
            for c in retriever.index.support(str(it.knowledge_id)):
                ix.add_support(str(it.knowledge_id), c.situation, c.matured, c.outcome, c.ref)
    for a, row in retriever.index._contradicts.items():
        for b, v in row.items():
            if a in ix and b in ix:
                ix.set_contradiction(a, b, v)
    for a, row in retriever.index._redundant.items():
        for b, v in row.items():
            if a in ix and b in ix:
                ix.set_redundancy(a, b, v)
    alt = Retriever(ix, retriever.weights, retriever.cfg, retriever.sim_weights, retriever.monitor).retrieve(sit, now, k=k)
    real = retriever.retrieve(sit, now, k=k)
    return {"real": real.ids(), "without": alt.ids(), "changed": real.ids() != alt.ids(),
            "expected_real": combine(real, retriever.index)["expected"], "expected_without": combine(alt, ix)["expected"]}


# ------------------------------------------------------------------------------------------------ evidence audits

def audit_support(index: KnowledgeIndex, now, min_distinct: float = 0.5, max_share: float = 0.5) -> dict[str, dict[str, Any]]:
    """Quality of each item's support cases, so an item resting on one repeated situation cannot look like broad evidence:
    distinct-situation share, share of the single most common situation, outcome coverage, and the newest case. Cases that
    matured at/after `now` are counted as `future` (and would make retrieval raise). Flags list every failed check."""
    out: dict[str, dict[str, Any]] = {}
    for kid in index.ids():
        cases = index.support(kid)
        n = len(cases)
        if n == 0:
            out[kid] = {"n": 0, "flags": ["no support cases"]}
            continue
        ids = [c.situation.situation_id for c in cases]
        counts: dict[str, int] = {}
        for i in ids:
            counts[i] = counts.get(i, 0) + 1
        distinct = len(counts) / n
        top_share = max(counts.values()) / n
        with_outcome = sum(c.outcome is not None for c in cases)
        future = sum(as_date(c.matured) >= as_date(now) for c in cases)
        flags = []
        if distinct < min_distinct:
            flags.append(f"low diversity: {len(counts)} distinct situations in {n} cases")
        if top_share > max_share:
            flags.append(f"{top_share:.0%} of the cases are one situation")
        if with_outcome < n:
            flags.append(f"{n - with_outcome} cases without an outcome")
        if future:
            flags.append(f"{future} cases not matured before now")
        out[kid] = {"n": n, "distinct_share": round(distinct, 4), "top_share": round(top_share, 4), "outcome_share": round(with_outcome / n, 4),
                    "future": future, "newest": str(max(as_date(c.matured) for c in cases)), "flags": flags}
    return out


def index_summary(index: KnowledgeIndex) -> dict[str, Any]:
    items = index.items()
    by_promo: dict[str, int] = {}
    by_epi: dict[str, int] = {}
    for it in items:
        by_promo[str(it.promotion)] = by_promo.get(str(it.promotion), 0) + 1
        by_epi[str(it.epistemic)] = by_epi.get(str(it.epistemic), 0) + 1
    return {"items": len(items), "support_cases": sum(len(index.support(str(i.knowledge_id))) for i in items),
            "promotion": by_promo, "epistemic": by_epi,
            "contradiction_edges": sum(len(v) for v in index._contradicts.values()) // 2,
            "redundancy_edges": sum(len(v) for v in index._redundant.values()) // 2}


def calibration_of_expected(retrievals: Sequence[Retrieval], outcomes: Sequence[Mapping[str, float]], bins: int = 4) -> dict[str, Any]:
    """Is the expected edge calibrated? Items binned by expected edge; per bin the mean expected and mean realised edge.
    slope = regression of realised on expected (1 = calibrated, 0 = uninformative, negative = misleading)."""
    xs, ys = [], []
    for r, o in zip(retrievals, outcomes):
        for i in r.items:
            if i.expected_edge is not None and i.knowledge_id in o:
                xs.append(i.expected_edge)
                ys.append(o[i.knowledge_id])
    if len(xs) < 4 * bins:
        return {"n": len(xs), "slope": float("nan"), "table": [], "status": "INSUFFICIENT_EVIDENCE"}
    x, y = np.array(xs), np.array(ys)
    order = np.argsort(x, kind="stable")
    table = [{"mean_expected": float(x[c].mean()), "mean_realised": float(y[c].mean()), "n": int(len(c))} for c in np.array_split(order, bins)]
    slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 1e-12 else float("nan")
    status = "MISLEADING" if slope <= 0 else "OVERCONFIDENT" if slope < 0.5 else "CALIBRATED" if slope <= 1.5 else "UNDERCONFIDENT"
    return {"n": len(xs), "slope": slope, "table": table, "status": status}


@functools.lru_cache(maxsize=1)
def _code_hash() -> str:
    from .core import current_code_hash
    return current_code_hash()


def stale_code(record: Mapping[str, Any]) -> bool:
    """True if a logged retrieval was made by different code than is loaded now (its result must not be trusted as a replay)."""
    return bool(record.get("code_hash")) and record["code_hash"] != _code_hash()

def register_prediction(retrieval: Retrieval, monitor: SkillMonitor, made_on, index: KnowledgeIndex | None = None) -> float | None:
    """Record what this retrieval expects, at the time it is made, so its later outcome can be scored walk-forward. Returns the
    expected edge recorded (None when the retrieval has no outcome evidence: nothing is predicted, nothing is scored)."""
    expected = combine(retrieval, index)["expected"]
    monitor.predict(retrieval.retrieval_id, made_on, expected)
    return expected


def resolve_outcome(retrieval: Retrieval, monitor: SkillMonitor, matured, realised: float) -> bool:
    """Feed the realised edge back once it is known. Returns False if this retrieval had recorded no prediction."""
    try:
        monitor.resolve(retrieval.retrieval_id, matured, realised)
    except KeyError:
        return False
    return True


def index_health(index: KnowledgeIndex, now, policy_min_cases: int = 20) -> str:
    """Readable summary of an index for a report: counts, items with thin or repetitive support, and unmatured cases."""
    aud = audit_support(index, now)
    summ = index_summary(index)
    thin = sorted(k for k, v in aud.items() if v["n"] < policy_min_cases)
    flagged = {k: v["flags"] for k, v in aud.items() if v["flags"] and v["n"] >= 1}
    lines = [f"{summ['items']} knowledge items, {summ['support_cases']} support cases, "
             f"{summ['contradiction_edges']} contradiction and {summ['redundancy_edges']} redundancy links"]
    if thin:
        lines.append(f"thin support (< {policy_min_cases} cases): {', '.join(thin)}")
    for k, f in sorted(flagged.items()):
        lines.append(f"{k}: " + "; ".join(f))
    return "\n".join(lines)

def retrieve_by_effect(retriever: Retriever, sit: Situation, now, effects: Iterable[DecisionEffect] | None = None,
                       k: int | None = None) -> dict[str, Retrieval]:
    """One retrieval per decision the knowledge can change (ranking, selection, direction, timing, exit, stop, size...).
    Knowledge that only changes one decision cannot leak into another; the caller sees per-decision evidence and per-decision
    abstentions instead of one blended list."""
    eff = list(effects) if effects is not None else sorted({e for it in retriever.index.items() for e in it.decision_effect
                                                            if e != DecisionEffect.NONE}, key=str)
    return {str(e): retriever.retrieve(sit, now, decision_effect=e, k=k) for e in eff}

def factor_predictiveness(retrievals: Sequence[Retrieval], outcomes: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Spearman correlation of each ranking factor with the realised edge, over every retrieved item whose outcome is known.
    A factor that does not predict is a candidate for a lower weight (`fit_retrieval_weights`); one that predicts backwards
    is a defect in how the factor is measured."""
    from scipy.stats import spearmanr
    cols: dict[str, list[float]] = {f: [] for f in FACTORS}
    ys: list[float] = []
    for r, o in zip(retrievals, outcomes):
        for it in r.items:
            if it.knowledge_id in o:
                ys.append(o[it.knowledge_id])
                for f in FACTORS:
                    v = it.factor(f)
                    cols[f].append(np.nan if v is None else v)
    out: dict[str, float] = {}
    y = np.array(ys, float)
    for f, v in cols.items():
        x = np.array(v, float)
        ok = ~np.isnan(x)
        if ok.sum() >= 15 and x[ok].std() > 1e-12 and y[ok].std() > 1e-12:
            out[f] = round(float(spearmanr(x[ok], y[ok])[0]), 4)
    return out


def walk_forward_replay(retriever: Retriever, events: Sequence[Mapping[str, Any]], monitor: SkillMonitor | None = None) -> dict[str, Any]:
    """Replay retrieval through time. Each event: {situation, made_on, matured, edge} (edge = realised edge of acting on it).
    Events are processed in `made_on` order; before each retrieval every earlier event whose outcome has matured strictly
    before `made_on` is fed back to the monitor, so the skill status a retrieval sees is exactly what was knowable then.
    Returns per-event influence/action, the final skill status and how often retrieval was allowed to speak."""
    if monitor is not None:
        retriever.monitor = monitor
    order = sorted(range(len(events)), key=lambda i: (as_date(events[i]["made_on"]), i))
    pending: list[tuple[Any, Retrieval, float]] = []
    rows = []
    for i in order:
        ev = events[i]
        made = as_date(ev["made_on"])
        due = [p for p in pending if as_date(p[0]) < made]
        pending = [p for p in pending if as_date(p[0]) >= made]
        for matured, r, edge in due:
            if retriever.monitor is not None:
                resolve_outcome(r, retriever.monitor, matured, edge)
        r = retriever.retrieve(ev["situation"], made)
        if retriever.monitor is not None and r.items:
            register_prediction(r, retriever.monitor, made, retriever.index)
        pending.append((ev["matured"], r, float(ev["edge"])))
        d = decide(r, retriever.index)
        rows.append({"event": i, "made_on": str(made), "influence": r.influence, "skill_status": r.skill_status, "action": d["action"],
                     "n_items": len(r.items), "expected": d["candidate_expected"], "edge": float(ev["edge"])})
    spoke = [r for r in rows if r["action"] == "USE"]
    final = None
    if retriever.monitor is not None and order:
        final = retriever.monitor.status(max(as_date(e["matured"]) for e in events) + dt.timedelta(days=1))
    return {"rows": rows, "n": len(rows), "spoke_share": len(spoke) / len(rows) if rows else float("nan"),
            "mean_edge_when_used": float(np.mean([r["edge"] * np.sign(r["expected"]) for r in spoke])) if spoke else None,
            "final_skill": final}


def find_stale(index: KnowledgeIndex, now, max_age_years: float = 3.0, min_temporal: float = 0.25) -> dict[str, list[str]]:
    """Items whose evidence is older than `max_age_years` or whose temporal-class relevance has decayed below `min_temporal`
    (blank situation: neutral). Stale items are not deleted; they are the queue for re-testing or retirement."""
    out: dict[str, list[str]] = {}
    for it in index.items():
        kid = str(it.knowledge_id)
        why = []
        age = _age_years(it.provenance.outcomes_seen_through or it.provenance.learned_at, now)
        if age > max_age_years:
            why.append(f"evidence is {age:.1f} years old")
        tr, _ = temporal_relevance(it, _neutral_situation(), now, index.support(kid))
        if tr is not None and tr < min_temporal:
            why.append(f"temporal relevance {tr:.2f} < {min_temporal:.2f}")
        if why:
            out[kid] = why
    return out


def _neutral_situation() -> Situation:
    from .situation import Block, BLOCK_ORDER
    return Situation(tuple(Block.make(k) for k in BLOCK_ORDER))


def merge_indexes(a: KnowledgeIndex, b: KnowledgeIndex) -> KnowledgeIndex:
    """Union of two indexes. For the same knowledge id the higher version wins (ties keep `a`); support cases are unioned by
    ref so a case present in both is not double counted; relations are unioned with the stronger value winning."""
    out = KnowledgeIndex()
    for src in (a, b):
        for it in src.items():
            out.add_item(it)
    for src in (a, b):
        for kid in src.ids():
            if int(src.get(kid).version) != int(out.get(kid).version):
                continue                                          # support of a superseded version is not carried over
            have = {c.ref for c in out.support(kid)}
            for c in src.support(kid):
                if c.ref not in have:
                    out.add_support(kid, c.situation, c.matured, c.outcome, c.ref)
                    have.add(c.ref)
    for rel, setter in (("_contradicts", out.set_contradiction), ("_redundant", out.set_redundancy)):
        best: dict[tuple, float] = {}
        for src in (a, b):
            for x, row in getattr(src, rel).items():
                for y, v in row.items():
                    if x in out and y in out:
                        key = tuple(sorted((x, y)))
                        best[key] = max(best.get(key, 0.0), v)
        for (x, y), v in best.items():
            setter(x, y, v)
    return out


def concentration_report(retrievals: Sequence[Retrieval], index: KnowledgeIndex) -> dict[str, Any]:
    """Over a run of retrievals: how often each item is retrieved, how concentrated that is (Herfindahl), and which items
    were never retrieved. Heavy reliance on one item is a fragility; items that never fire are dead weight or mis-scoped."""
    counts: dict[str, int] = {}
    total = 0
    for r in retrievals:
        for i in r.items:
            counts[i.knowledge_id] = counts.get(i.knowledge_id, 0) + 1
            total += 1
    hhi = float(sum((c / total) ** 2 for c in counts.values())) if total else float("nan")
    return {"retrievals": len(retrievals), "empty": sum(not r.items for r in retrievals), "counts": dict(sorted(counts.items())),
            "herfindahl": hhi, "never_retrieved": [k for k in index.ids() if k not in counts],
            "top_share": float(max(counts.values()) / total) if total else float("nan")}

def rank_quality_p(retrievals: Sequence[Retrieval], outcomes: Sequence[Mapping[str, float]], n_perm: int = 500, seed: int = 0) -> dict[str, float]:
    """Permutation p-value for `rank_quality`: is the score/edge rank correlation larger than when edges are shuffled among the
    retrieved items? Returns the observed Spearman, the null mean and a one-sided p (score predicts edge)."""
    from scipy.stats import rankdata
    xs, ys = [], []
    for r, o in zip(retrievals, outcomes):
        for i in r.items:
            if i.knowledge_id in o:
                xs.append(i.score)
                ys.append(o[i.knowledge_id])
    if len(xs) < 8 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return {"n": len(xs), "spearman": float("nan"), "null_mean": float("nan"), "p": float("nan")}
    rx, ry = rankdata(xs), rankdata(ys)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return {"n": len(xs), "spearman": obs, "null_mean": float(null.mean()), "p": float((1 + (null >= obs).sum()) / (n_perm + 1))}


def influence_shares(retrieval: Retrieval) -> dict[str, float]:
    """Share of the decision weight each retrieved item carries (its score over the total). A retrieval where one item holds
    nearly all of it is a single-item bet; the shares are what credit assignment (section 20) divides among items."""
    tot = sum(max(i.score, 0.0) for i in retrieval.items)
    if tot <= 0:
        return {}
    return {i.knowledge_id: round(max(i.score, 0.0) / tot, 6) for i in retrieval.items}


def support_leverage(index: KnowledgeIndex, kid: str, sit: Situation, now, top: int = 3,
                     sim_weights: SimilarityWeights = DEFAULT_SIMILARITY) -> dict[str, Any]:
    """How much does one support case move an item's expected edge here? Drops each contributing case in turn and reports the
    largest shifts. A single case with outsized leverage means the item's evidence in this situation is one anecdote."""
    cases = [c for c in index.support(kid) if c.outcome is not None]
    base = support_expectation(index, kid, sit, now, sim_weights)
    if base["mean"] is None or len(cases) < 3:
        return {"base": base["mean"], "max_shift": None, "top": [], "verdict": Unknown.INSUFFICIENT_DATA}
    shifts = []
    for c in cases:
        sub = KnowledgeIndex()
        sub.add_item(index.get(kid))
        for d in cases:
            if d.ref != c.ref:
                sub.add_support(kid, d.situation, d.matured, d.outcome, d.ref)
        alt = support_expectation(sub, kid, sit, now, sim_weights)["mean"]
        if alt is not None:
            shifts.append((c.ref, alt - base["mean"]))
    shifts.sort(key=lambda kv: -abs(kv[1]))
    worst = abs(shifts[0][1]) if shifts else 0.0
    scale = max(abs(base["mean"]), 1e-9)
    return {"base": base["mean"], "max_shift": worst, "top": shifts[:top], "verdict": "ANECDOTE" if worst > 0.5 * scale else "ROBUST"}

def factor_table(retrieval: Retrieval) -> list[dict[str, Any]]:
    """Report rows: one per retrieved item with every factor value (None = untested), the score and the flags."""
    return [{"rank": i.rank, "knowledge_id": i.knowledge_id, "score": i.score, "base": i.base_score, **{f.name: f.value for f in i.factors},
             "untested": ",".join(i.untested), "flags": ",".join(i.flags), "expected_edge": i.expected_edge, "expected_n": i.expected_n}
            for i in retrieval.items]


def ablation_suite(retriever: Retriever, sits: Sequence[Situation], now, k: int | None = None) -> dict[str, dict[str, float]]:
    """Ablate each positive factor across many queries: mean top-k overlap and the share of queries whose top-1 changes. A
    factor that never changes anything is inert (weight without effect); one that changes everything dominates."""
    out = {f: {"overlap": 0.0, "top1_change": 0.0} for f in POSITIVE_FACTORS}
    live = [s for s in sits if retriever.retrieve(s, now, k=k).items]
    if not live:
        return {}
    for f in POSITIVE_FACTORS:
        ov, ch = [], []
        for s in live:
            a = retriever.ablate(s, now, f, k=k)
            ov.append(a["overlap"])
            ch.append(not a["top1_same"])
        out[f] = {"overlap": float(np.mean(ov)), "top1_change": float(np.mean(ch))}
    return out

def config_to_record(cfg: RetrievalConfig) -> dict[str, Any]:
    d = dataclasses.asdict(cfg)
    d["allowed_promotions"] = [str(p) for p in cfg.allowed_promotions]
    return d


def config_from_record(d: Mapping[str, Any]) -> RetrievalConfig:
    d = dict(d)
    d["allowed_promotions"] = tuple(Promotion(p) for p in d["allowed_promotions"])
    cfg = RetrievalConfig(**d)
    errs = cfg.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return cfg


def drift_over_time(retriever: Retriever, sit: Situation, dates: Sequence[Any], k: int | None = None) -> dict[str, Any]:
    """The same situation retrieved at several `now`s. Reports the id set at each date and the Jaccard overlap between
    consecutive dates: how fast temporal relevance, modernity and newly matured evidence reshuffle the answer."""
    seq = sorted(dates, key=as_date)
    sets = [retriever.retrieve(sit, d, k=k).ids() for d in seq]
    ov = []
    for a, b in zip(sets, sets[1:]):
        ua, ub = set(a), set(b)
        ov.append(len(ua & ub) / len(ua | ub) if (ua or ub) else 1.0)
    return {"dates": [str(as_date(d)) for d in seq], "ids": sets, "consecutive_overlap": ov, "min_overlap": min(ov) if ov else 1.0}


def explain_influence(retrieval: Retrieval) -> str:
    shares = influence_shares(retrieval)
    if not shares:
        return "no retrieved item carries decision weight"
    parts = ", ".join(f"{k} {v:.0%}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1]))
    top = max(shares.values())
    tag = " (single-item bet)" if top > 0.8 and len(shares) > 1 else ""
    return f"decision weight: {parts}{tag}" + ("" if retrieval.influence else "; WITHHELD, no weight is applied")


def index_by_effect(index: KnowledgeIndex) -> dict[str, list[str]]:
    """Knowledge ids grouped by the decision they can change (contract section 43)."""
    out: dict[str, list[str]] = {}
    for it in index.items():
        for e in it.decision_effect:
            out.setdefault(str(e), []).append(str(it.knowledge_id))
    return {k: sorted(v) for k, v in sorted(out.items())}


def describe_retriever(retriever: Retriever) -> str:
    """One line naming the exact configuration behind a retrieval: factor weights, similarity weights id, gates, monitor."""
    w = retriever.weights.normalised()
    parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(w.items(), key=lambda kv: -kv[1]))
    gates = f"top_k {retriever.cfg.top_k}, min_similarity {retriever.cfg.min_similarity}, promotions {[str(p) for p in retriever.cfg.allowed_promotions]}"
    mon = "no skill monitor" if retriever.monitor is None else f"skill monitor (min_n {retriever.monitor.min_n})"
    return f"retriever [{parts}]; similarity {retriever.sim_weights.weights_id()}; {gates}; {mon}"
