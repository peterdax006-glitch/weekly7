"""Contract C62 section 7 (A01/A02 support): OBSERVATION != INTERPRETATION.

"Price rose after pattern X" must never silently become "X causes price to rise". Five things are stored as separate typed
records that reference each other but never merge: (1) Observation, (2) StatRelation (association only), (3) Hypothesis,
(4) Mechanism, (5) DecisionUsefulness. An `InterpretationRecord` bundles them and computes the highest claim level the
evidence actually supports (the evidence ladder); `assert_claim` refuses wording above that level.

Competing explanations H1-H6 are kept in a `HypothesisSet`: a probability vector updated by named tests through a likelihood
table, with a floor so uncertainty is retained (one test can never eliminate a rival), an evidence-once rule so the same
test cannot be counted twice, and a hash-free append-only evidence log. The likelihood table is a documented DESIGN PRIOR,
not a fitted quantity: calibrating it is future work.
Status: IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import enum
import math
import re
from typing import Mapping

from engine.learning.core import DecisionEffect, as_date, require_past, stable_hash
from engine.learning.knowledge import SchemaError, decode, encode


class Level(enum.IntEnum):
    """The evidence ladder. A claim may be worded at or below the level the record has earned, never above."""
    NONE = 0
    OBSERVATION = 1
    ASSOCIATION = 2
    HYPOTHESIS = 3
    MECHANISM = 4
    DECISION_VALUE = 5


class HypKind(str, enum.Enum):
    H1_GENUINE = "H1"          # the pattern genuinely predicts movement
    H2_VOLATILITY_PROXY = "H2"  # the pattern is proxying for volatility
    H3_REGIME_SPECIFIC = "H3"   # the pattern is regime-specific
    H4_REDUNDANT = "H4"         # the pattern is redundant with another feature
    H5_SELECTION_BIAS = "H5"    # the apparent effect is selection bias
    H6_UNSTABLE = "H6"          # the effect is unstable
    OTHER = "OTHER"

    def __str__(self):
        return self.value


CORE_KINDS = (HypKind.H1_GENUINE, HypKind.H2_VOLATILITY_PROXY, HypKind.H3_REGIME_SPECIFIC, HypKind.H4_REDUNDANT,
              HypKind.H5_SELECTION_BIAS, HypKind.H6_UNSTABLE)


class MechStatus(str, enum.Enum):
    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"


class ClaimError(ValueError):
    """Wording or a claim level that the recorded evidence does not support."""


_CAUSAL = re.compile(r"\b(cause[sd]?|causing|causal|drives?|driven by|drove|leads? to|led to|results? in|resulted in|"
                     r"because|due to|makes? (?:the )?(?:price|stock|return)|produces?|triggers?|explains?)\b", re.I)


def causal_language(text: str) -> list[str]:
    """Causal wording found in `text` (empty = pure association/observation wording)."""
    return sorted({m.group(0).lower() for m in _CAUSAL.finditer(text or "")})


def _unit(x, name, errs):
    if x is not None and (not isinstance(x, (int, float)) or math.isnan(x) or not 0.0 <= x <= 1.0):
        errs.append(f"{name}={x!r} outside [0,1]")


# ---------------------------------------------------------------- (1) observation

@dataclasses.dataclass(frozen=True)
class Observation:
    """What was seen, worded without any explanation. No ticker or calendar identity: a situation, a window and a number."""
    statement: str
    window_start: str
    window_end: str
    n: int
    measure: str = "mean_fwd_return"
    value: float | None = None

    @property
    def observation_id(self) -> str:
        return "OBS-" + stable_hash(self, 10)

    def check(self) -> list[str]:
        errs = []
        if not self.statement.strip():
            errs.append("observation has no statement")
        found = causal_language(self.statement)
        if found:
            errs.append(f"observation worded causally: {found}")
        try:
            if as_date(self.window_end) < as_date(self.window_start):
                errs.append("observation window ends before it starts")
        except ValueError:
            errs.append("observation window dates must be ISO")
        if self.n < 1:
            errs.append("an observation needs n >= 1")
        if self.value is not None and (math.isnan(self.value) or math.isinf(self.value)):
            errs.append("observation value not finite")
        return errs


# ---------------------------------------------------------------- (2) statistical relationship

@dataclasses.dataclass(frozen=True)
class StatRelation:
    """An association measured on an observation. It can be significant and still say nothing about cause."""
    observation_id: str
    estimate: float
    se: float
    n_eff: float
    n_tests: int = 1                      # how many comparable relations were searched to find this one
    out_of_sample: bool = False
    controls: tuple[str, ...] = ()        # what was held fixed (e.g. "volatility", "market_beta")
    method: str = "mean_difference"
    p_value: float | None = None

    @property
    def relation_id(self) -> str:
        return "REL-" + stable_hash(self, 10)

    @property
    def t(self) -> float | None:
        return None if self.se <= 0 else self.estimate / self.se

    def adjusted_p(self) -> float | None:
        """Sidak correction for the search that found it; the unadjusted p of a mined pattern is a hallucination factory."""
        if self.p_value is None:
            t = self.t
            if t is None:
                return None
            p = math.erfc(abs(t) / math.sqrt(2.0))
        else:
            p = self.p_value
        return 1.0 - (1.0 - p) ** self.n_tests if p < 1.0 else 1.0

    def significant(self, alpha: float = 0.05) -> bool:
        p = self.adjusted_p()
        return p is not None and p < alpha

    def check(self) -> list[str]:
        errs = []
        if not self.observation_id:
            errs.append("relation without an observation")
        if math.isnan(self.estimate) or math.isnan(self.se) or self.se < 0:
            errs.append("relation estimate/se invalid")
        if self.n_eff <= 0:
            errs.append("relation needs n_eff > 0")
        if self.n_tests < 1:
            errs.append("n_tests must be >= 1")
        _unit(self.p_value, "p_value", errs)
        found = causal_language(self.method)
        if found:
            errs.append(f"method worded causally: {found}")
        return errs


# ---------------------------------------------------------------- (3) hypotheses

@dataclasses.dataclass(frozen=True)
class Prediction:
    """A consequence that would be TRUE if the hypothesis were right and is checkable with a named test."""
    text: str
    test: str
    expected: bool = True

    def check(self) -> list[str]:
        return [] if self.text.strip() and self.test.strip() else ["prediction needs text and a named test"]


@dataclasses.dataclass(frozen=True)
class Hypothesis:
    kind: HypKind
    statement: str
    predictions: tuple[Prediction, ...]
    stated_at: str
    prior: float = 1.0 / 6.0

    @property
    def hyp_id(self) -> str:
        return f"{self.kind.value}-" + stable_hash({"k": self.kind, "s": self.statement}, 8)

    def check(self) -> list[str]:
        errs = []
        if not self.statement.strip():
            errs.append(f"{self.kind.value}: empty statement")
        if not self.predictions:
            errs.append(f"{self.kind.value}: no testable prediction (an untestable hypothesis is not a hypothesis)")
        for p in self.predictions:
            errs += p.check()
        if not 0.0 < self.prior < 1.0:
            errs.append(f"{self.kind.value}: prior must be in (0,1)")
        try:
            as_date(self.stated_at)
        except ValueError:
            errs.append(f"{self.kind.value}: stated_at must be an ISO date")
        return errs


def standard_hypotheses(subject: str, stated_at: str) -> tuple[Hypothesis, ...]:
    """The six competing explanations of section 7, each with the test that would embarrass it."""
    P = Prediction
    return (
        Hypothesis(HypKind.H1_GENUINE, f"{subject} genuinely predicts movement",
                   (P("effect persists out of sample", "out_of_sample_confirmed"),
                    P("effect survives a volatility control", "survives_volatility_control")), stated_at),
        Hypothesis(HypKind.H2_VOLATILITY_PROXY, f"{subject} is a proxy for volatility",
                   (P("effect vanishes once volatility is controlled", "survives_volatility_control", False),), stated_at),
        Hypothesis(HypKind.H3_REGIME_SPECIFIC, f"{subject} works only in some regimes",
                   (P("effect differs materially across regimes", "stable_across_regimes", False),), stated_at),
        Hypothesis(HypKind.H4_REDUNDANT, f"{subject} is redundant with another feature",
                   (P("no incremental value after existing features", "incremental_over_existing", False),), stated_at),
        Hypothesis(HypKind.H5_SELECTION_BIAS, f"{subject} is an artefact of how it was selected",
                   (P("effect disappears on data not used to select it", "survives_selection_control", False),), stated_at),
        Hypothesis(HypKind.H6_UNSTABLE, f"{subject} is unstable over time",
                   (P("sign flips across rolling windows", "sign_consistent_rolling", False),), stated_at),
    )


# Likelihood of seeing each test outcome UNDER each hypothesis. Design priors, not fitted values (UNPROVEN).
_H1, _H2, _H3, _H4, _H5, _H6 = CORE_KINDS
TESTS: dict[str, dict[bool, dict[HypKind, float]]] = {
    "survives_volatility_control": {
        True: {_H1: .85, _H2: .12, _H3: .60, _H4: .60, _H5: .30, _H6: .50},
        False: {_H1: .20, _H2: .90, _H3: .50, _H4: .50, _H5: .60, _H6: .50}},
    "stable_across_regimes": {
        True: {_H1: .80, _H2: .55, _H3: .10, _H4: .60, _H5: .40, _H6: .30},
        False: {_H1: .25, _H2: .45, _H3: .90, _H4: .50, _H5: .55, _H6: .70}},
    "incremental_over_existing": {
        True: {_H1: .80, _H2: .55, _H3: .60, _H4: .08, _H5: .45, _H6: .55},
        False: {_H1: .25, _H2: .50, _H3: .50, _H4: .92, _H5: .55, _H6: .50}},
    "survives_selection_control": {
        True: {_H1: .85, _H2: .60, _H3: .60, _H4: .60, _H5: .07, _H6: .50},
        False: {_H1: .15, _H2: .50, _H3: .45, _H4: .50, _H5: .93, _H6: .55}},
    "sign_consistent_rolling": {
        True: {_H1: .85, _H2: .60, _H3: .35, _H4: .60, _H5: .40, _H6: .10},
        False: {_H1: .35, _H2: .50, _H3: .60, _H4: .50, _H5: .60, _H6: .92}},
    "out_of_sample_confirmed": {
        True: {_H1: .80, _H2: .55, _H3: .55, _H4: .60, _H5: .15, _H6: .40},
        False: {_H1: .20, _H2: .50, _H3: .50, _H4: .50, _H5: .85, _H6: .60}},
}


@dataclasses.dataclass(frozen=True)
class EvidenceEntry:
    name: str
    at: str
    likelihoods: tuple[tuple[str, float], ...]
    evidence_through: str = ""


@dataclasses.dataclass(frozen=True)
class HypothesisSet:
    """Posterior over competing hypotheses. Immutable: `update` returns a new set and appends to the log."""
    weights: tuple[tuple[str, float], ...]
    log: tuple[EvidenceEntry, ...] = ()
    floor: float = 0.02

    @classmethod
    def from_priors(cls, priors: Mapping[HypKind, float], floor: float = 0.02) -> "HypothesisSet":
        if not priors:
            raise SchemaError("a HypothesisSet needs at least one hypothesis")
        tot = float(sum(priors.values()))
        if tot <= 0 or any(v < 0 for v in priors.values()):
            raise SchemaError("priors must be non-negative with positive total")
        if floor * len(priors) >= 1.0:
            raise SchemaError("floor too large for the number of hypotheses")
        return cls(tuple((HypKind(k).value, v / tot) for k, v in sorted(priors.items(), key=lambda kv: HypKind(kv[0]).value)), (), floor)

    @classmethod
    def uniform(cls, kinds: tuple[HypKind, ...] = CORE_KINDS, floor: float = 0.02) -> "HypothesisSet":
        return cls.from_priors({k: 1.0 for k in kinds}, floor)

    def as_dict(self) -> dict[str, float]:
        return dict(self.weights)

    def weight(self, kind: HypKind) -> float:
        return self.as_dict().get(HypKind(kind).value, 0.0)

    def entropy(self) -> float:
        return -sum(w * math.log(w) for _, w in self.weights if w > 0)

    def effective_number(self) -> float:
        """exp(entropy): how many hypotheses are still genuinely alive."""
        return math.exp(self.entropy())

    def leading(self) -> tuple[HypKind, float]:
        k, w = max(self.weights, key=lambda kv: (kv[1], kv[0]))
        return HypKind(k), w

    def margin(self) -> float:
        ws = sorted((w for _, w in self.weights), reverse=True)
        return ws[0] - (ws[1] if len(ws) > 1 else 0.0)

    def resolved(self, min_weight: float = 0.6, min_margin: float = 0.3) -> bool:
        """True only when one hypothesis clearly leads. Otherwise the honest answer is 'still ambiguous'."""
        _, w = self.leading()
        return w >= min_weight and self.margin() >= min_margin

    def check(self) -> list[str]:
        errs = []
        tot = sum(w for _, w in self.weights)
        if abs(tot - 1.0) > 1e-6:
            errs.append(f"weights sum to {tot:.6f}, not 1")
        if any(w < self.floor - 1e-9 for _, w in self.weights):
            errs.append("a weight fell below the uncertainty floor")
        if len({k for k, _ in self.weights}) != len(self.weights):
            errs.append("duplicate hypothesis kinds")
        return errs

    def update(self, name: str, likelihoods: Mapping[HypKind, float], now, evidence_through=None,
               strength: float = 1.0) -> "HypothesisSet":
        """Bayes step with tempering (`strength` < 1 discounts correlated evidence) and a floor that keeps every rival alive.
        The same named evidence may not be applied twice (double counting is the classic way to fake certainty)."""
        if any(e.name == name for e in self.log):
            raise ClaimError(f"evidence {name!r} already used; counting it again would fake certainty")
        if evidence_through is not None:
            require_past(evidence_through, now, f"evidence {name}")
        if self.log and as_date(now) < as_date(self.log[-1].at):
            raise ClaimError("evidence applied out of time order")
        if not 0.0 < strength <= 1.0:
            raise ClaimError("strength must be in (0,1]")
        cur = self.as_dict()
        missing = [k for k in cur if HypKind(k) not in likelihoods]
        if missing:
            raise ClaimError(f"likelihoods missing for {missing}: give an explicit value for every live hypothesis")
        post = {}
        for k, w in cur.items():
            L = float(likelihoods[HypKind(k)])
            if not 0.0 <= L <= 1.0 or math.isnan(L):
                raise ClaimError(f"likelihood {L} for {k} outside [0,1]")
            post[k] = w * (L ** strength)
        z = sum(post.values())
        if z <= 0:
            raise ClaimError("evidence has zero likelihood under every hypothesis")
        post = {k: v / z for k, v in post.items()}
        for _ in range(len(post)):                     # floor, then renormalise the free mass; converges in <= n passes
            low = {k for k, v in post.items() if v < self.floor}
            if not low:
                break
            free = 1.0 - self.floor * len(low)
            hi = sum(v for k, v in post.items() if k not in low)
            post = {k: (self.floor if k in low else v / hi * free) for k, v in post.items()}
        entry = EvidenceEntry(name, str(as_date(now)), tuple(sorted((HypKind(k).value, float(v)) for k, v in likelihoods.items())),
                              str(as_date(evidence_through)) if evidence_through is not None else "")
        return dataclasses.replace(self, weights=tuple(sorted(post.items())), log=self.log + (entry,))

    def apply_tests(self, results: Mapping[str, bool | None], now, evidence_through=None) -> "HypothesisSet":
        """Run named test outcomes through the likelihood table. None = test not run: no update, no pretending."""
        out = self
        for name in sorted(results):
            r = results[name]
            if r is None:
                continue
            if name not in TESTS:
                raise ClaimError(f"unknown test {name!r}; known: {sorted(TESTS)}")
            table = TESTS[name][bool(r)]
            live = {HypKind(k): table.get(HypKind(k), 0.5) for k in out.as_dict()}
            out = out.update(name, live, now, evidence_through)
        return out


# ---------------------------------------------------------------- (4) mechanism, (5) decision usefulness

@dataclasses.dataclass(frozen=True)
class Mechanism:
    """WHY it might work, stated so that it implies something else observable. A mechanism with no implication is a story."""
    hypothesis_id: str
    tag: str
    statement: str
    implication: str
    status: MechStatus = MechStatus.UNTESTED
    supporting_relation: str = ""

    def check(self) -> list[str]:
        errs = []
        if not (self.tag.strip() and self.statement.strip()):
            errs.append("mechanism needs a tag and a statement")
        if not self.implication.strip():
            errs.append(f"mechanism {self.tag}: no observable implication (untestable story)")
        if self.status == MechStatus.SUPPORTED and not self.supporting_relation:
            errs.append(f"mechanism {self.tag}: SUPPORTED without a supporting relation")
        return errs


@dataclasses.dataclass(frozen=True)
class DecisionUsefulness:
    """Incremental value of the item to a decision, measured against the decision without it. None = never measured."""
    decision_effect: DecisionEffect = DecisionEffect.NONE
    incremental_value: float | None = None
    se: float | None = None
    n_periods: int = 0

    @property
    def measured(self) -> bool:
        return self.incremental_value is not None and self.se is not None and self.n_periods > 0

    @property
    def z(self) -> float | None:
        return None if not self.measured or not self.se else self.incremental_value / self.se

    def check(self) -> list[str]:
        errs = []
        if (self.incremental_value is None) != (self.se is None):
            errs.append("usefulness value and se must be measured together")
        if self.se is not None and self.se < 0:
            errs.append("usefulness se < 0")
        if self.incremental_value is not None and self.decision_effect == DecisionEffect.NONE:
            errs.append("usefulness measured for no decision")
        return errs


# ---------------------------------------------------------------- the bundle

@dataclasses.dataclass(frozen=True)
class InterpretationRecord:
    observation: Observation
    relations: tuple[StatRelation, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    belief: HypothesisSet | None = None
    mechanisms: tuple[Mechanism, ...] = ()
    usefulness: DecisionUsefulness = DecisionUsefulness()

    def validate(self) -> list[str]:
        errs = list(self.observation.check())
        oid = self.observation.observation_id
        for r in self.relations:
            errs += r.check()
            if r.observation_id != oid:
                errs.append(f"relation {r.relation_id} points at another observation")
        kinds = [h.kind for h in self.hypotheses]
        if len(set(kinds)) != len(kinds):
            errs.append("duplicate hypothesis kinds")
        for h in self.hypotheses:
            errs += h.check()
        if self.belief is not None:
            errs += self.belief.check()
            hs = {h.kind.value for h in self.hypotheses}
            if hs and set(self.belief.as_dict()) != hs:
                errs.append("belief covers different hypotheses than the record declares")
        ids = {h.hyp_id for h in self.hypotheses}
        rel_ids = {r.relation_id for r in self.relations}
        for m in self.mechanisms:
            errs += m.check()
            if m.hypothesis_id not in ids:
                errs.append(f"mechanism {m.tag} refers to an unknown hypothesis")
            if m.supporting_relation and m.supporting_relation not in rel_ids:
                errs.append(f"mechanism {m.tag} cites a relation not in the record")
        errs += self.usefulness.check()
        return errs

    def assert_valid(self) -> "InterpretationRecord":
        errs = self.validate()
        if errs:
            raise SchemaError("; ".join(errs))
        return self

    def level(self, alpha: float = 0.05) -> Level:
        """Highest rung reached CONTIGUOUSLY: each rung needs the one below it."""
        lvl = Level.NONE
        if not self.observation.check():
            lvl = Level.OBSERVATION
        else:
            return lvl
        if any(r.out_of_sample and r.significant(alpha) for r in self.relations):
            lvl = Level.ASSOCIATION
        else:
            return lvl
        if self.hypotheses and self.belief is not None and not self.belief.check():
            lvl = Level.HYPOTHESIS
        else:
            return lvl
        if self.belief.resolved() and self.belief.leading()[0] == HypKind.H1_GENUINE and any(
                m.status == MechStatus.SUPPORTED for m in self.mechanisms):
            lvl = Level.MECHANISM
        else:
            return lvl
        z = self.usefulness.z
        if z is not None and z >= 1.645:
            lvl = Level.DECISION_VALUE
        return lvl

    def assert_claim(self, text: str, claimed: Level, alpha: float = 0.05) -> None:
        """Refuse a sentence whose level or wording exceeds the evidence."""
        have = self.level(alpha)
        if claimed > have:
            raise ClaimError(f"claim level {claimed.name} exceeds evidence level {have.name}")
        words = causal_language(text)
        if words and have < Level.MECHANISM:
            raise ClaimError(f"causal wording {words} needs a supported mechanism; evidence level is {have.name}")

    def association_wording(self) -> str:
        """The strongest sentence permitted without a mechanism; safe to store as KnowledgeObject.interpretation."""
        if not self.relations:
            return ""
        r = max(self.relations, key=lambda x: abs(x.t or 0.0))
        adj = r.adjusted_p()
        ctl = f" controlling for {', '.join(r.controls)}" if r.controls else ""
        oos = "out of sample" if r.out_of_sample else "in sample only"
        return (f"association {r.estimate:+.4g} (se {r.se:.3g}, adjusted p {adj:.3g} over {r.n_tests} comparisons, {oos}){ctl}; "
                "association only, mechanism unproven")

    def knowledge_fields(self) -> dict:
        """Ready-made values for KnowledgeObject(observation=, interpretation=, hypothesis=, interpretation_ref=, mechanism_tags=)."""
        lead = ""
        if self.belief is not None:
            k, w = self.belief.leading()
            lead = f"leading {k.value} at {w:.2f}; alive={self.belief.effective_number():.1f}"
        return {"observation": self.observation.statement, "interpretation": self.association_wording(),
                "hypothesis": lead, "interpretation_ref": self.record_hash(),
                "mechanism_tags": tuple(dict.fromkeys(m.tag for m in self.mechanisms if m.status == MechStatus.SUPPORTED))}

    def with_relation(self, rel: StatRelation) -> "InterpretationRecord":
        return dataclasses.replace(self, relations=self.relations + (rel,)).assert_valid()

    def with_tests(self, results: Mapping[str, bool | None], now, evidence_through=None) -> "InterpretationRecord":
        if self.belief is None:
            raise ClaimError("no hypotheses to test: call open_record first")
        return dataclasses.replace(self, belief=self.belief.apply_tests(results, now, evidence_through)).assert_valid()

    def with_mechanism(self, m: Mechanism) -> "InterpretationRecord":
        return dataclasses.replace(self, mechanisms=self.mechanisms + (m,)).assert_valid()

    def with_usefulness(self, u: DecisionUsefulness) -> "InterpretationRecord":
        return dataclasses.replace(self, usefulness=u).assert_valid()

    def record_hash(self) -> str:
        return stable_hash(self, 24)

    def to_dict(self) -> dict:
        return {"schema": 1, "record": encode(self)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "InterpretationRecord":
        if not isinstance(d, Mapping) or d.get("schema") != 1:
            raise SchemaError("unsupported interpretation schema")
        return decode(cls, d["record"]).assert_valid()

    def report(self) -> str:
        rows = [f"observation : {self.observation.statement}", f"level       : {self.level().name}"]
        for r in self.relations:
            rows.append(f"relation    : est {r.estimate:+.4g} t {r.t if r.t is None else round(r.t, 2)} adj-p {r.adjusted_p():.3g} "
                        f"oos={r.out_of_sample} controls={list(r.controls)}")
        if self.belief is not None:
            for k, w in sorted(self.belief.weights, key=lambda kv: -kv[1]):
                rows.append(f"  {k:<6}{w:6.3f}  {'#' * int(round(w * 40))}")
            rows.append(f"alive hypotheses: {self.belief.effective_number():.2f}; resolved={self.belief.resolved()}")
        for m in self.mechanisms:
            rows.append(f"mechanism   : {m.tag} [{m.status.value}]")
        return "\n".join(rows)


def open_record(observation: Observation, stated_at: str, subject: str = "the pattern", floor: float = 0.02) -> InterpretationRecord:
    """A fresh record: the observation plus the six standard competing hypotheses at equal weight (no favourite)."""
    hyps = standard_hypotheses(subject, stated_at)
    return InterpretationRecord(observation, (), hyps, HypothesisSet.uniform(CORE_KINDS, floor)).assert_valid()


def scan_causal_leaps(items: list[tuple[str, InterpretationRecord]]) -> list[str]:
    """Audit: (text, record) pairs whose wording outruns the evidence level. Empty list = clean."""
    bad = []
    for text, rec in items:
        words = causal_language(text)
        if words and rec.level() < Level.MECHANISM:
            bad.append(f"{rec.observation.observation_id}: {words} at level {rec.level().name}")
    return bad
