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
from typing import Any, Mapping, cast

import numpy as np

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
# Rule: a test discriminates ONLY the hypothesis that predicts its failure. Every other hypothesis (H1 included) is given the
# same likelihood, because a volatility proxy, a redundant feature or a regime-bound effect passes the other tests exactly as
# a genuine one does - letting passing tests favour H1 over them would count the same evidence twice.
_H1, _H2, _H3, _H4, _H5, _H6 = CORE_KINDS
_PASS, _FAIL, _TARGET_PASS, _TARGET_FAIL = 0.85, 0.15, 0.08, 0.92


def _table(target: HypKind) -> dict[bool, dict[HypKind, float]]:
    return {True: {k: (_TARGET_PASS if k is target else _PASS) for k in CORE_KINDS},
            False: {k: (_TARGET_FAIL if k is target else _FAIL) for k in CORE_KINDS}}


TESTS: dict[str, dict[bool, dict[HypKind, float]]] = {
    "survives_volatility_control": _table(_H2),
    "stable_across_regimes": _table(_H3),
    "incremental_over_existing": _table(_H4),
    "survives_selection_control": _table(_H5),
    "out_of_sample_confirmed": _table(_H5),
    "sign_consistent_rolling": _table(_H6),
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
        return None if not self.measured or not self.se else cast(float, self.incremental_value) / self.se

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


# ---------------------------------------------------------------- the tests that feed the hypothesis table

@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """A named test result with the numbers behind it. passed=None means the test could not be run (never a silent False)."""
    name: str
    passed: bool | None
    detail: str
    stat: float | None = None


def _ols(y: np.ndarray, cols: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and standard errors of y ~ 1 + cols (classical OLS). Rank-deficient designs raise: a control that
    is collinear with the signal is exactly the proxy case and must not be papered over."""
    X = np.column_stack([np.ones(len(y))] + cols)
    n, p = X.shape
    if n <= p + 2:
        raise ClaimError(f"too few observations ({n}) for {p} parameters")
    if np.linalg.matrix_rank(X) < p:
        raise ClaimError("design is rank deficient (control collinear with signal)")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - p)
    return beta, np.sqrt(np.diag(xtx_inv) * s2)


def _clean(*arrs) -> list[np.ndarray]:
    a = [np.asarray(x, dtype=float) for x in arrs]
    if len({len(x) for x in a}) != 1:
        raise ClaimError("arrays differ in length")
    ok = np.all([np.isfinite(x) for x in a], axis=0)
    return [x[ok] for x in a]


def probe_volatility_control(y, x, vol, min_t: float = 2.0, keep: float = 0.5) -> ProbeResult:
    """H2: does the signal's effect survive controlling for volatility? Passes if the coefficient keeps `keep` of its raw
    size AND stays significant. A pure volatility proxy loses both."""
    y, x, vol = _clean(y, x, vol)
    try:
        b0, s0 = _ols(y, [x])
        b1, s1 = _ols(y, [x, vol])
    except ClaimError as e:
        return ProbeResult("survives_volatility_control", None, str(e))
    if abs(b0[1] / s0[1]) < min_t:
        return ProbeResult("survives_volatility_control", None, "no significant raw effect to survive the control")
    t = float(b1[1] / s1[1]) if s1[1] > 0 else 0.0
    kept = float(b1[1] / b0[1]) if b0[1] != 0 else 0.0
    ok = abs(t) >= min_t and kept >= keep and np.sign(b1[1]) == np.sign(b0[1])
    return ProbeResult("survives_volatility_control", bool(ok), f"raw {b0[1]:+.4g}, controlled {b1[1]:+.4g}, kept {kept:.2f}, t {t:.2f}", t)


def probe_regime_stability(y, x, regime, min_group: int = 20, alpha: float = 0.01, min_rel_spread: float = 0.3) -> ProbeResult:
    """H3: is the slope the same in every regime? Cochran Q over per-regime slopes. Fails only if the heterogeneity is both
    statistically significant (alpha) AND material (slopes differ by more than `min_rel_spread` of the pooled size), or the
    sign flips - a significant but trivial difference is not regime dependence."""
    y, x, regime = _clean(y, x, regime)
    slopes, ses = [], []
    for r in sorted(set(regime.tolist())):
        m = regime == r
        if m.sum() < min_group:
            continue
        try:
            b, s = _ols(y[m], [x[m]])
        except ClaimError:
            continue
        slopes.append(float(b[1]))
        ses.append(float(s[1]))
    if len(slopes) < 2:
        return ProbeResult("stable_across_regimes", None, "fewer than two regimes with enough data")
    w = 1.0 / np.square(ses)
    mu = float(np.sum(w * slopes) / np.sum(w))
    q = float(np.sum(w * (np.array(slopes) - mu) ** 2))
    p = float(_chi2_sf(q, len(slopes) - 1))
    flips = len({np.sign(s) for s in slopes}) > 1
    spread = (max(slopes) - min(slopes)) / abs(mu) if mu != 0 else math.inf
    stable = not flips and (p >= alpha or spread <= min_rel_spread)
    return ProbeResult("stable_across_regimes", bool(stable),
                       f"Q={q:.2f} (p={p:.3f}) over {len(slopes)} regimes; spread {spread:.2f}; sign flip={flips}", p)


def _chi2_sf(q: float, k: int) -> float:
    from scipy import stats
    return float(stats.chi2.sf(q, k))


def probe_incremental(y, x, others: list, min_t: float = 2.0) -> ProbeResult:
    """H4: does the signal add anything once the features we already have are in the model?"""
    cols = _clean(y, x, *others)
    y, x, others = cols[0], cols[1], cols[2:]
    try:
        b0, s0 = _ols(y, [x])
        b, s = _ols(y, [x] + others)
    except ClaimError as e:
        return ProbeResult("incremental_over_existing", None if "rank" not in str(e) else False, str(e))
    if abs(b0[1] / s0[1]) < min_t:
        return ProbeResult("incremental_over_existing", None, "no significant marginal effect to add anything")
    t = float(b[1] / s[1]) if s[1] > 0 else 0.0
    return ProbeResult("incremental_over_existing", bool(abs(t) >= min_t), f"incremental t {t:.2f} after {len(others)} existing features", t)


def probe_selection_control(discovery_t: float, holdout_t: float, min_ratio: float = 0.4, min_holdout: float = 1.65) -> ProbeResult:
    """H5: an effect chosen because it looked good must keep looking good on data that played no part in choosing it.
    Winner's curse predicts a holdout t well below the discovery t; a real effect shrinks less."""
    if discovery_t <= 0:
        return ProbeResult("survives_selection_control", None, "discovery t must be positive (claimed direction)")
    ratio = holdout_t / discovery_t
    return ProbeResult("survives_selection_control", bool(holdout_t >= min_holdout and ratio >= min_ratio),
                       f"holdout t {holdout_t:.2f} vs discovery {discovery_t:.2f} (ratio {ratio:.2f})", ratio)


def probe_rolling_sign(y, x, window: int = 60, min_share: float = 0.7) -> ProbeResult:
    """H6: does the sign of the slope agree across rolling windows? A share below `min_share` means the effect wanders."""
    y, x = _clean(y, x)
    if len(y) < 2 * window:
        return ProbeResult("sign_consistent_rolling", None, f"need >= {2 * window} observations, have {len(y)}")
    signs: Any = []
    for i in range(0, len(y) - window + 1, max(1, window // 2)):
        yy, xx = y[i:i + window], x[i:i + window]
        if np.std(xx) == 0:
            continue
        signs.append(np.sign(np.cov(xx, yy)[0, 1]))
    if len(signs) < 3:
        return ProbeResult("sign_consistent_rolling", None, "too few usable windows")
    signs = np.array(signs)
    share = float(max(np.mean(signs > 0), np.mean(signs < 0)))
    return ProbeResult("sign_consistent_rolling", bool(share >= min_share), f"{share:.2f} of {len(signs)} windows share a sign", share)


def run_standard_tests(y, x, vol=None, regime=None, others=None, discovery_t: float | None = None,
                       holdout_t: float | None = None, window: int = 60) -> list[ProbeResult]:
    """Every test that the supplied inputs allow; the rest are simply absent (not run, not failed)."""
    out = [probe_rolling_sign(y, x, window)]
    if vol is not None:
        out.append(probe_volatility_control(y, x, vol))
    if regime is not None:
        out.append(probe_regime_stability(y, x, regime))
    if others:
        out.append(probe_incremental(y, x, list(others)))
    if discovery_t is not None and holdout_t is not None:
        out.append(probe_selection_control(discovery_t, holdout_t))
    return out


def outcomes_to_results(outcomes: list[ProbeResult]) -> dict[str, bool | None]:
    return {o.name: o.passed for o in outcomes}


def interpret(observation: Observation, relation: StatRelation, outcomes: list[ProbeResult], now, stated_at: str,
              subject: str = "the pattern", evidence_through=None) -> InterpretationRecord:
    """End-to-end: open the six hypotheses, attach the relation, run the outcomes through the likelihood table."""
    rec = open_record(observation, stated_at, subject).with_relation(relation)
    return rec.with_tests(outcomes_to_results(outcomes), now, evidence_through)


# ---------------------------------------------------------------- what to test next, and how belief moved

def outcome_probability(belief: HypothesisSet, test: str) -> float:
    """Predictive P(test passes) under the current belief: sum over hypotheses of weight * P(pass | hypothesis)."""
    table = TESTS[test][True]
    return sum(w * table[HypKind(k)] for k, w in belief.weights if HypKind(k) in table)


def expected_information_gain(belief: HypothesisSet, test: str) -> float:
    """Expected entropy reduction (nats) from running `test` next, computed with the same likelihood table and floor that
    `update` uses. This is the value of resolving the ambiguity, and feeds the research policy (section 36)."""
    if test not in TESTS:
        raise ClaimError(f"unknown test {test!r}")
    p_pass = outcome_probability(belief, test)
    h0 = belief.entropy()
    exp_h = 0.0
    for outcome, p in ((True, p_pass), (False, 1.0 - p_pass)):
        if p <= 1e-12:
            continue
        table = TESTS[test][outcome]
        live = {HypKind(k): table.get(HypKind(k), 0.5) for k, _ in belief.weights}
        post = belief.update(f"__preview_{test}_{outcome}", live, "2999-12-31")
        exp_h += p * post.entropy()
    return max(0.0, h0 - exp_h)


def next_test(belief: HypothesisSet) -> tuple[str, float] | None:
    """The not-yet-run test with the highest expected information gain, or None when every test has been used."""
    used = {e.name for e in belief.log}
    scored = [(name, expected_information_gain(belief, name)) for name in sorted(TESTS) if name not in used]
    return max(scored, key=lambda t: (t[1], t[0])) if scored else None


def kl_divergence(p: HypothesisSet, q: HypothesisSet) -> float:
    """KL(p || q) over the shared hypothesis kinds: how far belief moved (0 = unchanged)."""
    pd_, qd = p.as_dict(), q.as_dict()
    if set(pd_) != set(qd):
        raise ClaimError("beliefs cover different hypotheses")
    return sum(w * math.log(w / qd[k]) for k, w in pd_.items() if w > 0)


def belief_shift(before: HypothesisSet, after: HypothesisSet) -> dict:
    """Summary of a belief update for reports and audits: leader before/after, biggest mover, KL, alive hypotheses."""
    b, a = before.as_dict(), after.as_dict()
    move = {k: a[k] - b[k] for k in a}
    top = max(move, key=lambda k: abs(move[k]))
    lb, la = before.leading()[0], after.leading()[0]
    return {"leader_before": lb.value, "leader_after": la.value, "leader_changed": lb != la, "biggest_mover": top,
            "biggest_move": move[top], "kl": kl_divergence(after, before), "alive_before": before.effective_number(),
            "alive_after": after.effective_number()}


def evidence_ledger(belief: HypothesisSet) -> list[str]:
    """One readable line per piece of evidence used, in order, with the hypothesis it hurt or helped most."""
    rows = []
    for e in belief.log:
        lik = dict(e.likelihoods)
        hi, lo = max(lik, key=lik.__getitem__), min(lik, key=lik.__getitem__)
        rows.append(f"{e.at} {e.name}: favours {hi} ({lik[hi]:.2f}), hurts {lo} ({lik[lo]:.2f})"
                    + (f"; evidence through {e.evidence_through}" if e.evidence_through else ""))
    return rows


def unresolved_pairs(belief: HypothesisSet, within: float = 0.15) -> list[tuple[str, str, float]]:
    """Hypothesis pairs whose weights differ by less than `within`, both above the floor: the ambiguities a next test
    should target. Sorted by combined weight so the biggest ambiguity is first."""
    ws = sorted(belief.weights, key=lambda kv: (-kv[1], kv[0]))
    out = []
    for i, (a, wa) in enumerate(ws):
        for b, wb in ws[i + 1:]:
            if abs(wa - wb) < within and min(wa, wb) > belief.floor * 1.5:
                out.append((a, b, wa + wb))
    return sorted(out, key=lambda t: (-t[2], t[0], t[1]))
