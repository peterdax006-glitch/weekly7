"""Research priority engine (C66 sections 2, 19, 34, 43; also sections 10, 16, 20, 37; canon C66, C63).

Section 2: 'the system must understand the difference between doing a huge amount of work and accomplishing something
important'. Every candidate experiment carries the section-2 value vector (information gain, decision value, uncertainty
reduction, transfer potential, failure/loss reduction, volatility value, direction value, compute cost, overfit risk,
redundancy). The conceptual formula multiplies them; this module does NOT assume that formula is right. It

  1. keeps the literal formula only as a BASELINE (`literal_priority`),
  2. learns a priority model from what finished experiments actually delivered (`PriorityModel`: a ridge-anchored, discounted
     log-linear model over the value vector, per-objective intercepts, and a shrunken per-family effect), starting from the
     conceptual formula as its prior so it is sensible with no data,
  3. VALIDATES the learned model out of sample (`validate_model`): walk-forward, against the literal baseline and a permutation
     null, with a verdict that never says VALIDATED without the evidence; until then the learned model only gets a partial vote
     (`PriorityModel.trust`),
  4. values losses (section 34): `expected_loss_avoided`, `tail_loss_share`, an objective hierarchy whose weights can be refined by
     outcomes (`ObjectiveWeights`), and a direction lane opened only as far as volatility has evidence (`VolatilityEvidence`),
  5. stops wasting compute (`WasteTracker`): a family whose recent results are negligible loses priority (test I),
  6. never rewards volume: realised value is zeroed for any finding that did not survive out of sample (section 43).

Research-world rule (C64/C66 sections 29-31): everything here is built from matured outcomes, so a RealisedValue is refused
unless it matured strictly before `now`; item text must be identity free. Public entry: `step(state, now, items, results,
budget, seed)`. Builds on engine.learning.research_policy (PriorityFunction penalties, ComputeBudget, knapsack,
marginal_return_verdict, EVSI) and engine.learning.research_priority (identity_leak). IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from engine.learning.core import ValidationLabel
from engine.learning.experiment_memory import to_ts
from engine.learning.research_policy import (Candidate, ComputeBudget, ComputeCost, CostModel, Factors, InfoModel, PolicyConfig, PolicyContext,
                                             PriorityFunction, ResearchTarget, epsilon_schedule, knapsack, marginal_return_verdict,
                                             mutual_information)
from engine.learning.research_priority import identity_leak
from engine.research.core import (OBJECTIVE_ORDER, ExperimentValue, FirewallBreach, Namespace, Problem, require_past, stable_hash)

LABEL = ValidationLabel.NOT_VALIDATED.value
EPS = 1e-9
FEATURES = ("info", "decision", "uncertainty", "transfer", "loss", "failure", "volatility", "direction", "feasibility",
            "not_overfit", "not_redundant")
PROBLEMS = tuple(Problem)
DIM = len(FEATURES) + len(PROBLEMS)


class PriorityError(ValueError):
    """A malformed item, result or configuration. Raised, never patched over: a bad value vector must not become a ranking."""


# ------------------------------------------------------------------------------------------------------ configuration

@dataclass(frozen=True)
class PriorityConfig:
    """Every knob in one place. Defaults are priors, not tuned values (C63: no tuning in this wave)."""
    floor: float = 0.02                 # no log-feature goes below log(floor): a missing/zero term cannot zero a candidate
    info_scale_bits: float = 0.5        # this many bits of information gain maps to 1 - 1/e
    cost_exponent: float = 0.7          # value per cost^exponent: compute matters, but not linearly (one big job is not 10 small ones)
    prior_precision: float = 2.0        # strength of the anchor to the conceptual formula
    forget: float = 0.995               # per-observation discount of old data (the world drifts)
    noise_var_prior: float = 0.6        # prior variance of log realised value around the model
    noise_n_prior: float = 6.0
    family_shrink: float = 3.0          # family effect = sum(residual) / (n + shrink)
    explore_fraction: float = 0.10      # share of the compute plan drawn by posterior sampling instead of the mean
    min_records_validate: int = 40
    min_records_trust: int = 12
    validate_margin: float = 0.05       # learned top-k capture must beat the literal baseline by this much to be VALIDATED
    validate_alpha: float = 0.10
    w_info: float = 0.5                 # weight of a normalised bit of information in realised value
    w_decision: float = 1.0
    w_failure: float = 0.8
    waste_k: int = 4                    # results in the barren-streak window
    waste_eps: float = 0.03             # a result below this fraction of the family's first-quartile value is 'negligible'
    waste_decay: float = 0.35           # priority multiplier exp(-decay * streak) once the streak starts
    waste_floor: float = 0.05
    missing_penalty: float = 0.85       # per estimated-term that was never estimated (None is not zero, but it is not free)

    def check(self) -> list:
        errs = []
        if not (0 < self.floor < 0.5):
            errs.append("floor must be in (0, 0.5)")
        if not (0 < self.forget <= 1):
            errs.append("forget must be in (0, 1]")
        if self.prior_precision <= 0 or self.info_scale_bits <= 0:
            errs.append("prior_precision and info_scale_bits must be positive")
        if not (0 <= self.explore_fraction <= 0.5):
            errs.append("explore_fraction must be in [0, 0.5]")
        if self.cost_exponent < 0:
            errs.append("cost_exponent must be >= 0")
        return errs


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def norm_bits(bits: float | None, scale: float = 0.5) -> float | None:
    """Information in bits -> [0, 1): saturating, so a 10-bit claim cannot swamp every other term."""
    return None if bits is None else 1.0 - math.exp(-max(float(bits), 0.0) / scale)


# ------------------------------------------------------------------------------------------------------ records

@dataclass(frozen=True)
class ResearchItem:
    """One candidate experiment as the priority engine sees it. `value` holds ESTIMATES made before running; a term left None
    was not estimated (never read as 0). Dimensionless terms are shares in [0, 1]: decision_value is the fraction of the
    achievable decision improvement, loss_reduction_value the fraction of the historical loss budget expected to be avoided."""
    item_id: str
    text: str
    problem: Problem
    family: str
    value: ExperimentValue
    created: str
    target: ResearchTarget = ResearchTarget.UNKNOWN_AREA
    feasibility: float = 1.0
    data_needs: tuple = ()
    requires: tuple = ()
    real_data: bool = False
    ram_gb: float = 1.0
    subsystem: str = ""
    prior_tests_on_window: int = 0
    free_parameters: int = 0
    question_id: str = ""
    tags: tuple = ()

    def check(self) -> list:
        errs = []
        if not self.item_id:
            errs.append("item without id")
        leak = identity_leak(self.text)
        if leak:
            errs.append(f"{self.item_id}: text {leak} (identity firewall)")
        if not self.text.strip():
            errs.append(f"{self.item_id}: empty text")
        v = self.value
        for f in ("decision_value", "uncertainty_reduction", "transfer_potential", "failure_reduction_value", "loss_reduction_value",
                  "volatility_value", "direction_value", "overfit_risk", "redundancy"):
            x = getattr(v, f)
            if x is not None and not (0.0 <= x <= 1.0):
                errs.append(f"{self.item_id}: value.{f}={x!r} outside [0,1]")
        for f in ("information_gain", "compute_cost"):
            x = getattr(v, f)
            if x is not None and (x < 0 or math.isnan(x)):
                errs.append(f"{self.item_id}: value.{f}={x!r} negative")
        if v.compute_cost is None or v.compute_cost <= 0:
            errs.append(f"{self.item_id}: a candidate needs a positive compute_cost estimate")
        if not (0.0 <= self.feasibility <= 1.0):
            errs.append(f"{self.item_id}: feasibility outside [0,1]")
        return errs

    @property
    def cost(self) -> float:
        return float(self.value.compute_cost or 1.0)


@dataclass(frozen=True)
class RealisedValue:
    """What a finished experiment actually delivered, in the same units as its estimate. `survived_oos` False means the finding
    failed out-of-sample replication: its decision-type value is then ZERO whatever the in-sample number said (section 43)."""
    item: ResearchItem
    realised: ExperimentValue
    cost_minutes: float
    matured_at: str
    survived_oos: bool | None = None
    note: str = ""

    def check(self) -> list:
        errs = [f"realised {self.item.item_id}: {e}" for e in self.item.check()]
        if self.cost_minutes <= 0:
            errs.append(f"realised {self.item.item_id}: non-positive cost")
        if not self.matured_at:
            errs.append(f"realised {self.item.item_id}: missing matured_at")
        return errs


# ------------------------------------------------------------------------------------------------------ objectives

@dataclass(frozen=True)
class ObjectiveWeights:
    """The section-34 hierarchy as weights: volatility first, then avoiding losses, direction, coverage, consistency,
    efficiency, secondary metrics. They are a prior; `refined` moves them toward the value each objective actually delivered."""
    w: Mapping = field(default_factory=lambda: {
        Problem.VOLATILITY: 1.0, Problem.LOSS_AVOIDANCE: 0.9, Problem.DIRECTION: 0.5, Problem.COVERAGE: 0.25, Problem.CONSISTENCY: 0.2,
        Problem.EFFICIENCY: 0.1, Problem.SECONDARY: 0.03, Problem.RESEARCH_PROCESS: 0.3, Problem.DATA_QUALITY: 0.4})

    def weight(self, p: Problem) -> float:
        return float(self.w.get(Problem.parse(p), 0.1))

    def hierarchy_violations(self) -> list:
        """Adjacent objectives whose weights are out of the section-34 order (volatility >= loss >= direction >= coverage ...)."""
        out = []
        for a, b in zip(OBJECTIVE_ORDER, OBJECTIVE_ORDER[1:]):
            if self.weight(a) + 1e-9 < self.weight(b):
                out.append((a.value, b.value))
        return out

    def refined(self, rows: Sequence["RealisedValue"], scalar: Callable[["RealisedValue"], float], strength: float = 20.0,
                max_ratio: float = 3.0) -> "ObjectiveWeights":
        """Move each objective's weight toward its realised value per experiment, relative to the all-objective mean, shrunk with
        `strength` pseudo-observations and bounded to [1/max_ratio, max_ratio] of the prior so a lucky streak cannot invert the
        hierarchy in one update. Objectives with no data keep their prior."""
        by: dict = {}
        for r in rows:
            by.setdefault(r.item.problem, []).append(scalar(r))
        allv = [x for xs in by.values() for x in xs]
        if not allv or np.mean(allv) <= 0:
            return self
        mean_all = float(np.mean(allv))
        new = dict(self.w)
        for p, xs in by.items():
            n = len(xs)
            ratio = (np.sum(xs) + strength * mean_all) / ((n + strength) * mean_all)
            new[p] = float(self.weight(p) * min(max_ratio, max(1.0 / max_ratio, ratio)))
        return ObjectiveWeights(new)

    def to_dict(self) -> dict:
        return {p.value: float(v) for p, v in sorted(self.w.items(), key=lambda kv: kv[0].value)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ObjectiveWeights":
        return cls({Problem(k): float(v) for k, v in d.items()})


@dataclass(frozen=True)
class VolatilityEvidence:
    """Section 10: direction research opens only after volatility has evidence. This is the evidence the priority engine reads:
    the out-of-sample precision of the volatility selector against the base rate, how many cases, how many independent years."""
    precision: float = 0.0
    base_rate: float = 0.0
    n_eval: int = 0
    n_years: int = 0
    out_of_sample: bool = False

    def openness(self, min_years: int = 3) -> float:
        """0 = no evidence (direction lane nearly shut), 1 = strong evidence. Probability the precision exceeds the base rate
        (one-sided normal approximation), times the share of the required independent years, and 0 unless measured out of sample."""
        if not self.out_of_sample or self.n_eval <= 0 or not (0 < self.base_rate < 1):
            return 0.0
        se = math.sqrt(self.base_rate * (1.0 - self.base_rate) / self.n_eval)
        z = (self.precision - self.base_rate) / max(se, EPS)
        return float(norm.cdf(z - 1.645) * min(1.0, self.n_years / max(min_years, 1)))


# ------------------------------------------------------------------------------------------------------ value estimators

def tail_loss_share(losses: Sequence[float], top_frac: float = 0.05) -> float:
    """Share of total loss magnitude carried by the worst `top_frac` of loss events. Heavy tails are where a failure regime
    lives: 5 % of events carrying 40 % of the loss is a research target; 5 % carrying 6 % is not."""
    arr = np.sort(np.abs(np.asarray([x for x in losses if x is not None and not math.isnan(x)], float)))[::-1]
    if arr.size == 0 or arr.sum() <= 0:
        return 0.0
    k = max(1, int(math.ceil(top_frac * arr.size)))
    return float(arr[:k].sum() / arr.sum())


def share_by_tag(losses: Sequence[float], tags: Sequence[str]) -> dict:
    """Each tag's share of the total loss magnitude (tag = regime / pattern family / cause): which known failure carries the loss."""
    if len(losses) != len(tags):
        raise PriorityError("losses and tags differ in length")
    tot = float(np.sum(np.abs(losses)))
    out: dict = {}
    for x, t in zip(losses, tags):
        out[t] = out.get(t, 0.0) + abs(float(x))
    return {t: (v / tot if tot > 0 else 0.0) for t, v in sorted(out.items())}


def expected_loss_avoided(loss_share: float, p_isolate: float, p_actionable: float, persistence: float = 1.0,
                          capture: float = 0.6) -> float:
    """Section 34: expected FUTURE loss avoided, as a share of the loss budget. loss_share of the historical loss sits in the
    regime; p_isolate that research can isolate it; p_actionable that a decision rule can act on it; persistence that it recurs;
    capture the fraction of it a rule would avoid. A product of probabilities: honest about how much can go wrong."""
    for name, v in (("loss_share", loss_share), ("p_isolate", p_isolate), ("p_actionable", p_actionable), ("persistence", persistence),
                    ("capture", capture)):
        if not (0.0 <= v <= 1.0) or math.isnan(v):
            raise PriorityError(f"{name}={v!r} outside [0,1]")
    return float(loss_share * p_isolate * p_actionable * persistence * capture)


def loss_value_interval(loss_share: float, successes: int, trials: int, p_actionable: float, persistence: float, capture: float = 0.6,
                        n_draw: int = 400, seed: int = 0) -> tuple:
    """(mean, 10 %, 90 %) of the loss avoided when p_isolate is only known from `successes` of `trials` earlier isolation
    attempts of the same kind (Beta posterior with a uniform prior). Wide when the base rate is thin: the priority engine
    reads the spread as the value of learning it more precisely."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1 + successes, 1 + max(trials - successes, 0), size=n_draw)
    v = loss_share * p * p_actionable * persistence * capture
    return float(v.mean()), float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))


def winner_gain_value(hit_rate: float, uplift: float, achievable_gain: float) -> float:
    """Value of improving winner detection: the relative uplift in hit rate, weighted by how much of the achievable gain is left.
    A 0.5 % uplift on a 40 % hit rate against half the achievable gain is worth 0.005 * 0.5 -- tiny by construction."""
    if not (0 <= hit_rate <= 1 and 0 <= achievable_gain <= 1) or uplift < 0:
        raise PriorityError("winner_gain_value needs shares in [0,1] and a non-negative uplift")
    return _clip01(uplift * achievable_gain * (1.0 - hit_rate))


def volatility_value_estimate(precision_now: float, precision_new: float, coverage_gain: float, band_value: float = 1.0) -> float:
    """Section 0 objective 1: value of raising the precision (and coverage) of predicted movers. precision in [0,1]; the value is
    the relative precision uplift, boosted by extra coverage, scaled by how valuable being in the 5-10 % band is."""
    if not all(0 <= x <= 1 for x in (precision_now, precision_new, coverage_gain, band_value)):
        raise PriorityError("volatility_value_estimate takes shares in [0,1]")
    uplift = max(precision_new - precision_now, 0.0) / max(1.0 - precision_now, 0.05)
    return _clip01(band_value * (0.7 * uplift + 0.3 * coverage_gain))


def direction_value_estimate(accuracy_now: float, accuracy_new: float, mover_share: float, gate_openness: float) -> float:
    """Section 10: value of better direction among predicted movers, scaled by the volatility-evidence gate. At coin-flip accuracy
    with a closed gate the value is ~0 -- no direction research before volatility has evidence."""
    if not all(0 <= x <= 1 for x in (accuracy_now, accuracy_new, mover_share, gate_openness)):
        raise PriorityError("direction_value_estimate takes shares in [0,1]")
    edge_gain = max(accuracy_new - max(accuracy_now, 0.5), 0.0) / 0.5
    return _clip01(edge_gain * mover_share * gate_openness)


def information_from_model(prior: Mapping, likelihood: Mapping) -> float:
    """Bits the experiment is expected to teach: mutual information between the hypotheses and its outcome."""
    return float(mutual_information(prior, likelihood))


def uncertainty_reduction_estimate(sd_now: float, n_now: float, n_new: float) -> float:
    """Share by which the standard error of the quantity in question shrinks if `n_new` more independent observations arrive."""
    if sd_now < 0 or n_now < 0 or n_new < 0:
        raise PriorityError("negative sample size or sd")
    if n_now + n_new <= 0:
        return 0.0
    return _clip01(1.0 - math.sqrt(max(n_now, 1e-9) / (max(n_now, 1e-9) + n_new)))


def make_value(cost: float, information_bits: float | None = None, decision: float | None = None, uncertainty: float | None = None,
               transfer: float | None = None, failure: float | None = None, loss: float | None = None, volatility: float | None = None,
               direction: float | None = None, overfit: float | None = None, redundancy: float | None = None) -> ExperimentValue:
    """An ExperimentValue with every share clipped into [0,1]; NaN is refused (a NaN estimate is a bug, not a missing term)."""
    def c(x):
        if x is None:
            return None
        if math.isnan(x):
            raise PriorityError("NaN in a value estimate")
        return _clip01(x)
    if cost is None or cost <= 0 or math.isnan(cost):
        raise PriorityError("compute cost must be positive")
    return ExperimentValue(information_gain=None if information_bits is None else max(float(information_bits), 0.0),
                           decision_value=c(decision), uncertainty_reduction=c(uncertainty), transfer_potential=c(transfer),
                           failure_reduction_value=c(failure), loss_reduction_value=c(loss), volatility_value=c(volatility),
                           direction_value=c(direction), compute_cost=float(cost), overfit_risk=c(overfit), redundancy=c(redundancy))


# ------------------------------------------------------------------------------------------------------ features and the literal baseline

CORE_TERMS = ("information_gain", "decision_value", "uncertainty_reduction")


def missing_core(item: ResearchItem) -> int:
    """How many of the three terms every candidate should estimate were never estimated."""
    return sum(1 for f in CORE_TERMS if getattr(item.value, f) is None)


def _terms(item: ResearchItem, cfg: PriorityConfig, impute: float = 0.3) -> dict:
    v = item.value

    def g(x):
        return impute if x is None else float(x)
    return {"info": g(norm_bits(v.information_gain, cfg.info_scale_bits)), "decision": g(v.decision_value),
            "uncertainty": g(v.uncertainty_reduction), "transfer": g(v.transfer_potential), "loss": g(v.loss_reduction_value),
            "failure": g(v.failure_reduction_value), "volatility": g(v.volatility_value), "direction": g(v.direction_value),
            "feasibility": float(item.feasibility), "not_overfit": 1.0 - g(v.overfit_risk if v.overfit_risk is not None else 0.0),
            "not_redundant": 1.0 - g(v.redundancy if v.redundancy is not None else 0.0)}


def feature_vector(item: ResearchItem, cfg: PriorityConfig) -> np.ndarray:
    """Log of each value term (floored) followed by a one-hot of the objective the item serves. Fixed length DIM."""
    t = _terms(item, cfg)
    x = np.zeros(DIM)
    for i, name in enumerate(FEATURES):
        x[i] = math.log(max(t[name], cfg.floor))
    x[len(FEATURES) + PROBLEMS.index(Problem.parse(item.problem))] = 1.0
    return x


def literal_priority(item: ResearchItem, cfg: PriorityConfig) -> float:
    """The section-2 formula read literally: the product of the value terms over compute cost, redundancy and overfit risk. Two
    readings are forced: a missing numerator term is imputed neutral-low (0.3), and the risks divide as (1 + risk) so a zero
    risk does not divide by zero. Kept ONLY as the baseline the learned model must beat (`validate_model`)."""
    t = _terms(item, cfg)
    lossy = max(t["loss"], t["failure"])
    num = 1.0
    for v in (t["info"], t["decision"], t["uncertainty"], t["transfer"], lossy, t["volatility"], t["direction"], t["feasibility"]):
        num *= max(v, cfg.floor)
    v = item.value
    den = (item.cost ** cfg.cost_exponent) * (1.0 + (v.redundancy or 0.0)) * (1.0 + (v.overfit_risk or 0.0))
    return num / den


def realised_scalar(rv: RealisedValue, weights: ObjectiveWeights, cfg: PriorityConfig) -> float:
    """One number for what a finished experiment delivered, on the hierarchy of section 34. Information counts (a well-posed
    negative answer is knowledge) but weighs less than decisions; loss avoided, volatility and direction are weighed by the
    objective hierarchy. If the finding did NOT survive out of sample, every decision-type part is zero: an in-sample gain that
    does not replicate is not value (section 43)."""
    r = rv.realised
    keep = 0.0 if rv.survived_oos is False else 1.0
    parts = [cfg.w_info * (norm_bits(r.information_gain, cfg.info_scale_bits) or 0.0)]
    parts.append(keep * cfg.w_decision * (r.decision_value or 0.0))
    parts.append(keep * cfg.w_failure * (r.failure_reduction_value or 0.0))
    parts.append(keep * weights.weight(Problem.LOSS_AVOIDANCE) * (r.loss_reduction_value or 0.0))
    parts.append(keep * weights.weight(Problem.VOLATILITY) * (r.volatility_value or 0.0))
    parts.append(keep * weights.weight(Problem.DIRECTION) * (r.direction_value or 0.0))
    return float(sum(parts))


def realised_rate(rv: RealisedValue, weights: ObjectiveWeights, cfg: PriorityConfig) -> float:
    """Realised value per unit of compute (cost^exponent, the same denominator the ranking uses)."""
    return realised_scalar(rv, weights, cfg) / (max(rv.cost_minutes, 1.0) ** cfg.cost_exponent)


# ------------------------------------------------------------------------------------------------------ the learned model

class PriorityModel:
    """log(realised value) = x . theta + family_effect, fitted by discounted ridge toward the conceptual formula.

    theta's prior mean is 1 on every log value term (the section-2 product) and log(objective weight) on the objective
    intercepts, with precision `prior_precision`: with no data the model IS the conceptual formula plus the hierarchy; every
    finished experiment moves it. Old data is discounted by `forget` per observation. A per-family effect (shrunk mean residual)
    captures what the value vector cannot: 'this line of work reliably over- or under-delivers'. `trust` says how much of the
    fitted model to use: it grows with the effective sample and is halved until `validate_model` has said VALIDATED."""

    def __init__(self, cfg: PriorityConfig | None = None, weights: ObjectiveWeights | None = None):
        self.cfg = cfg or PriorityConfig()
        errs = self.cfg.check()
        if errs:
            raise PriorityError("; ".join(errs))
        self.weights = weights or ObjectiveWeights()
        self.m0 = np.ones(DIM)
        for i, p in enumerate(PROBLEMS):
            self.m0[len(FEATURES) + i] = math.log(max(self.weights.weight(p), 1e-3))
        self.P0 = np.diag([self.cfg.prior_precision] * len(FEATURES) + [0.5 * self.cfg.prior_precision] * len(PROBLEMS))
        self.obs: list = []                         # (x, y, family, matured_at, item_id)
        self.label = ValidationLabel.NOT_VALIDATED
        self._fit: tuple | None = None

    # -------- data
    def observe(self, rv: RealisedValue) -> float:
        """Add one finished experiment; returns the log target. The target is log(realised scalar + floor)."""
        errs = rv.check()
        if errs:
            raise PriorityError("; ".join(errs))
        y = math.log(realised_scalar(rv, self.weights, self.cfg) + self.cfg.floor)
        self.obs.append((feature_vector(rv.item, self.cfg), y, rv.item.family, str(rv.matured_at), rv.item.item_id))
        self._fit = None
        return y

    def n_effective(self) -> float:
        n = len(self.obs)
        g = self.cfg.forget
        return float((1 - g ** n) / (1 - g)) if g < 1 else float(n)

    # -------- fitting
    def _fitted(self) -> tuple:
        if self._fit is not None:
            return self._fit
        n = len(self.obs)
        A = self.P0.copy()
        b = self.P0 @ self.m0
        w = np.array([self.cfg.forget ** (n - 1 - i) for i in range(n)])
        if n:
            X = np.vstack([o[0] for o in self.obs])
            y = np.array([o[1] for o in self.obs])
            A = A + (X * w[:, None]).T @ X
            b = b + (X * w[:, None]).T @ y
        m = np.linalg.solve(A, b)
        fam: dict = {}
        sigma2 = self.cfg.noise_var_prior
        if n:
            res = y - X @ m
            sigma2 = (self.cfg.noise_n_prior * self.cfg.noise_var_prior + float((w * res ** 2).sum())) / (self.cfg.noise_n_prior + float(w.sum()))
            sums: dict = {}
            for wi, ri, o in zip(w, res, self.obs):
                s = sums.setdefault(o[2], [0.0, 0.0])
                s[0] += wi * ri
                s[1] += wi
            fam = {f: s[0] / (s[1] + self.cfg.family_shrink) for f, s in sums.items()}
        self._fit = (m, np.linalg.inv(A), float(sigma2), fam)
        return self._fit

    def trust(self) -> float:
        """0 until `min_records_trust` results; then n/(n+8), capped at 0.75 unless the last walk-forward validation said VALIDATED.
        (The ridge anchor to the conceptual formula already keeps a small-sample fit cautious; this is a second, coarser guard.)"""
        if len(self.obs) < self.cfg.min_records_trust:
            return 0.0
        n = self.n_effective()
        return float(n / (n + 8.0) * (1.0 if self.label == ValidationLabel.VALIDATED else 0.75))

    def theta(self) -> np.ndarray:
        """The coefficients actually used for ranking: fitted theta blended with the conceptual prior by trust."""
        m, _, _, _ = self._fitted()
        t = self.trust()
        return t * m + (1 - t) * self.m0

    def family_effect(self, family: str) -> float:
        return self.trust() * self._fitted()[3].get(family, 0.0)

    # -------- prediction
    def predict_log(self, item: ResearchItem, theta: np.ndarray | None = None) -> tuple:
        """(expected log value, predictive sd). The sd grows for items unlike anything seen (x A^-1 x) and is what
        posterior-sampling exploration and rank-stability read."""
        _, Ainv, s2, _ = self._fitted()
        x = feature_vector(item, self.cfg)
        th = self.theta() if theta is None else theta
        mean = float(x @ th) + self.family_effect(item.family)
        return mean, math.sqrt(s2 * (1.0 + float(x @ Ainv @ x)))

    def predict_value(self, item: ResearchItem, theta: np.ndarray | None = None) -> float:
        mean, _ = self.predict_log(item, theta)
        return math.exp(mean) * (self.cfg.missing_penalty ** missing_core(item))

    def rate(self, item: ResearchItem, theta: np.ndarray | None = None) -> float:
        """Predicted value per compute: the number the ranking sorts on."""
        return self.predict_value(item, theta) / (item.cost ** self.cfg.cost_exponent)

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        """One draw from the coefficient posterior (Gaussian, centred on the blended coefficients)."""
        _, Ainv, s2, _ = self._fitted()
        cov = s2 * Ainv
        cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(DIM)
        return rng.multivariate_normal(self.theta(), cov)

    # -------- introspection
    def coefficients(self) -> dict:
        m, _, s2, fam = self._fitted()
        out = {name: float(self.theta()[i]) for i, name in enumerate(FEATURES)}
        out.update({f"objective:{p.value}": float(self.theta()[len(FEATURES) + i]) for i, p in enumerate(PROBLEMS)})
        out["_noise_sd"] = math.sqrt(s2)
        out["_trust"] = self.trust()
        out["_n"] = len(self.obs)
        return out

    def drift_from_prior(self) -> dict:
        """How far each learned coefficient moved from the conceptual formula, largest first: what the data taught."""
        th = self.theta()
        rows = [(name, float(th[i] - self.m0[i])) for i, name in enumerate(FEATURES)]
        rows += [(f"objective:{p.value}", float(th[len(FEATURES) + i] - self.m0[len(FEATURES) + i])) for i, p in enumerate(PROBLEMS)]
        return dict(sorted(rows, key=lambda r: -abs(r[1])))

    def family_table(self) -> dict:
        """Family -> (effect, n): which lines of work over- or under-deliver relative to their estimates."""
        fam = self._fitted()[3]
        counts: dict = {}
        for o in self.obs:
            counts[o[2]] = counts.get(o[2], 0) + 1
        return {f: (float(v), counts.get(f, 0)) for f, v in sorted(fam.items())}

    # -------- persistence
    def to_dict(self) -> dict:
        return {"cfg": dataclasses.asdict(self.cfg), "weights": self.weights.to_dict(), "label": self.label.value,
                "obs": [[o[0].tolist(), o[1], o[2], o[3], o[4]] for o in self.obs]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "PriorityModel":
        m = cls(PriorityConfig(**d["cfg"]), ObjectiveWeights.from_dict(d["weights"]))
        m.label = ValidationLabel(d["label"])
        m.obs = [(np.array(o[0], float), float(o[1]), o[2], o[3], o[4]) for o in d["obs"]]
        return m


# ------------------------------------------------------------------------------------------------------ out-of-sample validation

@dataclass(frozen=True)
class ValidationReport:
    label: ValidationLabel
    n: int
    folds: int
    capture_learned: float
    capture_literal: float
    capture_random: float
    spearman_learned: float
    spearman_literal: float
    mean_gain: float                                # learned - literal capture, averaged over folds
    p_value: float                                  # one-sided sign-flip test that the learned model is no better than the literal
    fold_gains: tuple
    reason: str


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def capture(pred: Sequence[float], realised: Sequence[float], k: int) -> float:
    """Share of the best possible top-k realised value that the top-k by `pred` obtained (1 = perfect ranking)."""
    if len(pred) != len(realised) or k <= 0:
        raise PriorityError("capture needs equal-length inputs and k > 0")
    pred = np.asarray(pred, float)
    real = np.asarray(realised, float)
    k = min(k, len(pred))
    best = np.sort(real)[::-1][:k].sum()
    got = real[np.argsort(-pred, kind="stable")[:k]].sum()
    return float(got / best) if best > 0 else 0.0


def sign_flip_p(diffs: Sequence[float]) -> float:
    """One-sided exact sign-flip p-value for mean(diffs) > 0 (<= 12 folds exact, otherwise 4096 seeded flips)."""
    d = np.asarray(diffs, float)
    if d.size == 0:
        return 1.0
    obs = d.mean()
    if d.size <= 12:
        signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(d.size)] for i in range(2 ** d.size)])
    else:
        signs = np.random.default_rng(d.size).choice([-1, 1], size=(4096, d.size))
    means = (signs * d).mean(axis=1)
    return float((means >= obs - 1e-12).mean())


def validate_model(records: Sequence[RealisedValue], now, cfg: PriorityConfig | None = None, weights: ObjectiveWeights | None = None,
                   folds: int = 5, k: int = 3, seed: int = 0) -> ValidationReport:
    """Walk-forward test of the LEARNED ranking against the LITERAL formula and a random ranking. Records are ordered by the date
    their outcome matured (all must be strictly before `now`); each fold is predicted by a model trained only on earlier
    folds. VALIDATED requires: enough records, learned capture beating the literal by `validate_margin`, a sign-flip p-value
    under `validate_alpha`, and beating random. Worse than the literal by the margin is FAILED_VALIDATION. Anything else is
    reported as it is -- a middling result is never rounded up."""
    cfg = cfg or PriorityConfig()
    weights = weights or ObjectiveWeights()
    for r in records:
        require_past(r.matured_at, now, f"validation record {r.item.item_id}")
    rows = sorted(records, key=lambda r: (str(r.matured_at), r.item.item_id))
    n = len(rows)

    def empty(label, why):
        return ValidationReport(label, n, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, (), why)
    if n < cfg.min_records_validate:
        return empty(ValidationLabel.INSUFFICIENT_EVIDENCE, f"{n} finished experiments, need {cfg.min_records_validate}")
    chunks = np.array_split(np.arange(n), folds + 1)
    rng = np.random.default_rng(seed)
    cl, cf, cr, sl, sf, gains = [], [], [], [], [], []
    for j in range(1, folds + 1):
        train = [rows[i] for c in chunks[:j] for i in c]
        test = [rows[i] for i in chunks[j]]
        if len(test) < max(4, k + 1):
            continue
        model = PriorityModel(cfg, weights)
        model.label = ValidationLabel.VALIDATED          # score the model as it would rank once trusted: trust measures the fit itself
        for r in train:
            model.observe(r)
        real = [realised_rate(r, weights, cfg) for r in test]
        learned = [model.rate(r.item) for r in test]
        literal = [literal_priority(r.item, cfg) for r in test]
        kk = min(k, len(test) // 2)
        cl.append(capture(learned, real, kk))
        cf.append(capture(literal, real, kk))
        cr.append(float(np.mean([capture(rng.permutation(len(test)).astype(float), real, kk) for _ in range(100)])))
        sl.append(_spearman(learned, real))
        sf.append(_spearman(literal, real))
        gains.append(cl[-1] - cf[-1])
    if len(gains) < 3:
        return empty(ValidationLabel.INSUFFICIENT_EVIDENCE, "fewer than 3 usable test folds")
    mg = float(np.mean(gains))
    p = sign_flip_p(gains)
    args = (n, len(gains), float(np.mean(cl)), float(np.mean(cf)), float(np.mean(cr)), float(np.mean(sl)), float(np.mean(sf)), mg, p, tuple(gains))
    if mg >= cfg.validate_margin and p <= cfg.validate_alpha and np.mean(cl) > np.mean(cr):
        return ValidationReport(ValidationLabel.VALIDATED, *args, "learned ranking beat the literal formula out of sample")
    if mg <= -cfg.validate_margin:
        return ValidationReport(ValidationLabel.FAILED_VALIDATION, *args, "the literal formula ranked better than the learned model")
    return ValidationReport(ValidationLabel.NOT_VALIDATED, *args, "no reliable difference from the literal formula yet")


# ------------------------------------------------------------------------------------------------------ compute-waste control

class WasteTracker:
    """Section 20 / test I: a line of work that keeps producing negligible value loses priority. Per family, the realised value
    of each finished experiment in order; a family is barren after `waste_k` consecutive results below `waste_eps` with no rising
    trend (research_policy.marginal_return_verdict), and its multiplier falls with the length of the barren streak. One good
    result ends the streak: a family is never punished for its past, only for its present."""

    def __init__(self, cfg: PriorityConfig | None = None):
        self.cfg = cfg or PriorityConfig()
        self.hist: dict = {}

    def observe(self, family: str, value: float, when: str) -> None:
        if math.isnan(value) or value < 0:
            raise PriorityError("realised value must be a non-negative number")
        self.hist.setdefault(family, []).append((str(when), float(value)))
        self.hist[family].sort(key=lambda t: t[0])

    def values(self, family: str) -> list:
        return [v for _, v in self.hist.get(family, [])]

    def streak(self, family: str) -> int:
        n = 0
        for v in reversed(self.values(family)):
            if v >= self.cfg.waste_eps:
                break
            n += 1
        return n

    def verdict(self, family: str) -> dict:
        return marginal_return_verdict(self.values(family), k=self.cfg.waste_k, eps=self.cfg.waste_eps)

    def multiplier(self, family: str) -> float:
        """1.0 with no evidence or a healthy family; decays after the barren streak passes waste_k; never below waste_floor."""
        v = self.verdict(family)
        if v["verdict"] != "STOP":
            return 1.0
        extra = max(self.streak(family) - self.cfg.waste_k + 1, 1)
        return max(self.cfg.waste_floor, math.exp(-self.cfg.waste_decay * extra))

    def report(self) -> dict:
        return {f: {"n": len(h), "streak": self.streak(f), "verdict": self.verdict(f)["verdict"], "multiplier": self.multiplier(f)}
                for f, h in sorted(self.hist.items())}

    def to_dict(self) -> dict:
        return {f: [[w, v] for w, v in h] for f, h in self.hist.items()}

    @classmethod
    def from_dict(cls, d: Mapping, cfg: PriorityConfig | None = None) -> "WasteTracker":
        t = cls(cfg)
        t.hist = {f: [(w, float(v)) for w, v in h] for f, h in d.items()}
        return t


# ------------------------------------------------------------------------------------------------------ the engine state

@dataclass
class PriorityState:
    """Everything the priority engine remembers. Lives in MATURED_RESEARCH_STATE: it is built from finished experiments."""
    cfg: PriorityConfig
    model: PriorityModel
    waste: WasteTracker
    evidence: VolatilityEvidence = field(default_factory=VolatilityEvidence)
    history: list = field(default_factory=list)              # RealisedValue, in arrival order
    prequential: list = field(default_factory=list)          # (predicted rate, realised rate) recorded BEFORE learning each result
    done_ids: set = field(default_factory=set)
    last_validation: ValidationReport | None = None
    validated_at_n: int = 0
    namespace: Namespace = Namespace.MATURED_RESEARCH
    revalidate_every: int = 20
    refine_every: int = 25
    direction_floor: float = 0.10
    policy: PriorityFunction | None = None
    costs: CostModel = field(default_factory=CostModel)
    max_per_question: int = 2

    def weights(self) -> ObjectiveWeights:
        return self.model.weights


def new_state(cfg: PriorityConfig | None = None, weights: ObjectiveWeights | None = None, evidence: VolatilityEvidence | None = None,
              policy_cfg: PolicyConfig | None = None) -> PriorityState:
    cfg = cfg or PriorityConfig()
    return PriorityState(cfg=cfg, model=PriorityModel(cfg, weights), waste=WasteTracker(cfg), evidence=evidence or VolatilityEvidence(),
                         policy=PriorityFunction(policy_cfg))


def absorb(state: PriorityState, results: Iterable[RealisedValue], now) -> list:
    """Learn from finished experiments. Each must have matured strictly before `now` (FirewallBreach otherwise -- a result from the
    future is a clock bug and must not be silently skipped). Results already absorbed (same item id and maturity) are ignored, so
    replaying a log is safe. The model predicts each result BEFORE it learns it (prequential record for calibration)."""
    taken = []
    seen = {(r.item.item_id, str(r.matured_at)) for r in state.history}
    for rv in sorted(results, key=lambda r: (str(r.matured_at), r.item.item_id)):
        require_past(rv.matured_at, now, f"result {rv.item.item_id}")
        if (rv.item.item_id, str(rv.matured_at)) in seen:
            continue
        seen.add((rv.item.item_id, str(rv.matured_at)))
        real_rate = realised_rate(rv, state.weights(), state.cfg)
        state.prequential.append((state.model.rate(rv.item), real_rate, rv.item.family))
        state.model.observe(rv)
        state.costs.observe(rv.item.family, rv.item.cost, rv.cost_minutes)
        state.waste.observe(rv.item.family, realised_scalar(rv, state.weights(), state.cfg), rv.matured_at)
        state.history.append(rv)
        state.done_ids.add(rv.item.item_id)
        taken.append(rv.item.item_id)
    n = len(state.history)
    if taken and n >= state.cfg.min_records_validate and n - state.validated_at_n >= state.revalidate_every:
        rep = validate_model(state.history, now, state.cfg, state.weights())
        state.last_validation = rep
        state.validated_at_n = n
        state.model.label = rep.label
    return taken


def refine_weights(state: PriorityState) -> ObjectiveWeights:
    """Let realised value per objective move the hierarchy weights (bounded; see ObjectiveWeights.refined). The model is rebuilt
    on the same observations with the new prior so the two stay consistent."""
    w = state.weights().refined(state.history, lambda r: realised_scalar(r, state.weights(), state.cfg))
    old = state.model
    m = PriorityModel(state.cfg, w)
    m.label = old.label
    for rv in state.history:
        m.observe(rv)
    state.model = m
    return w


# ------------------------------------------------------------------------------------------------------ ranking

def to_candidate(item: ResearchItem, now) -> Candidate:
    """The learning-side Candidate for an item, so the EXISTING PriorityFunction penalties (duplicate, forking paths, free
    parameters, data availability, compute) apply unchanged. overfit_hint is 0 here: the item's overfit risk is already a model
    feature, and counting it twice would double-penalise."""
    v = item.value
    return Candidate(cid=item.item_id, question=item.text, target=item.target, created_at=str(item.created),
                     factors=Factors(0.5, 0.5, 0.5, item.feasibility, 0.5), info=InfoModel("proxy", {"resolvability": 0.5}),
                     cost=ComputeCost(cpu_minutes=item.cost, ram_gb=item.ram_gb, real_data=item.real_data),
                     subsystem=item.subsystem, family=item.family, data_needs=item.data_needs, overfit_hint=0.0,
                     free_parameters=item.free_parameters, prior_tests_on_window=item.prior_tests_on_window)


@dataclass(frozen=True)
class RankedItem:
    item: ResearchItem
    rate: float                         # final priority: predicted value per compute after every multiplier
    cost: float                         # the cost used: planned minutes times the learned overrun multiplier for the family
    value: float                        # predicted value before dividing by cost
    log_sd: float                       # predictive sd of log value
    literal: float                      # the conceptual formula, for comparison
    multipliers: Mapping                # waste / direction gate / duplicate / forking / compute
    blocked: str = ""
    rank: int = 0

    def explain(self) -> str:
        if self.blocked:
            return f"{self.item.item_id} [{self.item.problem}] BLOCKED: {self.blocked}"
        pens = ", ".join(f"{k}x{v:.2f}" for k, v in self.multipliers.items() if v < 0.999) or "none"
        return (f"#{self.rank} {self.item.item_id} [{self.item.problem}/{self.item.family}] rate={self.rate:.5f} value={self.value:.4f} "
                f"cost={self.cost:.0f}m sd={self.log_sd:.2f} literal={self.literal:.5f} penalties: {pens}")


def direction_lane(state: PriorityState, item: ResearchItem) -> float:
    """Section 10 gate: DIRECTION items are scaled by how much volatility evidence exists (floor keeps the lane from being
    permanently dead, so a first direction probe can still be run)."""
    if Problem.parse(item.problem) != Problem.DIRECTION:
        return 1.0
    return state.direction_floor + (1.0 - state.direction_floor) * state.evidence.openness()


def rank(state: PriorityState, items: Sequence[ResearchItem], now, budget: ComputeBudget | None = None, ctx: PolicyContext | None = None,
         theta: np.ndarray | None = None) -> list:
    """Rank candidate experiments by predicted value per compute, best first. An item created at/after `now`, or with a malformed
    value vector, raises. Blocked items (missing data, blocking duplicate, unfinished prerequisite, over budget) are returned LAST
    with their reason, never dropped."""
    ctx = ctx or PolicyContext(now=str(now))
    pf = state.policy or PriorityFunction()
    out = []
    for it in items:
        errs = it.check()
        if errs:
            raise PriorityError("; ".join(errs))
        require_past(it.created, now, f"item {it.item_id} creation")
        it = fill_missing(it, state.history)
        cost = it.cost * state.costs.multiplier(it.family)
        cand = to_candidate(replace(it, value=replace(it.value, compute_cost=cost)), now)
        mult = {"waste": state.waste.multiplier(it.family), "direction_gate": direction_lane(state, it)}
        dup, dup_msg = pf.duplicate_penalty(cand, ctx)
        mult["duplicate"] = dup
        mult["forking_paths"] = pf.overfit_penalty(cand, ctx)
        comp, comp_msg = pf.compute_penalty(cand, budget)
        mult["compute"] = comp
        block = pf.data_block(cand, ctx) or dup_msg or comp_msg
        waiting = [r for r in it.requires if r not in state.done_ids]
        if waiting and not block:
            block = "waiting for " + ", ".join(sorted(waiting))
        mean, sd = state.model.predict_log(it, theta)
        value = math.exp(mean) * (state.cfg.missing_penalty ** missing_core(it))
        pen = 1.0
        for v in mult.values():
            pen *= v
        rate = 0.0 if block or pen <= 0 else value / (cost ** state.cfg.cost_exponent) * pen
        out.append(RankedItem(it, rate, cost, value, sd, literal_priority(it, state.cfg), mult, blocked=block or ("a hard penalty is zero" if pen <= 0 else "")))
    out.sort(key=lambda r: (bool(r.blocked), -r.rate, r.item.item_id))
    return diversity_cap([replace(r, rank=i + 1) for i, r in enumerate(out)], state.max_per_question)


@dataclass(frozen=True)
class PriorityPlan:
    now: str
    ranked: tuple
    selected: tuple                     # item ids to run this round, in rank order
    explored: tuple                     # the subset chosen by posterior sampling rather than by the mean
    deferred: tuple                     # ((item id, reason), ...): never silently dropped
    minutes_used: float
    share_by_problem: Mapping
    share_by_family: Mapping
    model_trust: float
    validation: str
    notes: tuple = ()

    def top(self, n: int = 5) -> list:
        return [r for r in self.ranked if not r.blocked][:n]

    def explain(self, limit: int = 8) -> str:
        head = [f"round {self.now}: {len(self.selected)} selected ({self.minutes_used:.0f} cpu-min), model trust {self.model_trust:.2f}, "
                f"validation: {self.validation}"]
        head += [r.explain() for r in self.ranked[:limit]]
        head += [f"deferred {i}: {why}" for i, why in self.deferred[:limit]]
        head += [f"note: {n}" for n in self.notes]
        return "\n".join(head)


def _select(ranked: Sequence[RankedItem], budget: ComputeBudget, max_cells: int = 2_000_000) -> tuple:
    """Choose what fits the round: exact knapsack over rate x cost when small, greedy by rate otherwise. At most
    `real_data_slots` real-data jobs. Returns (chosen ranked items, deferred (id, reason))."""
    live = [r for r in ranked if not r.blocked and r.rate > 0]
    deferred = [(r.item.item_id, r.blocked or "zero priority") for r in ranked if r.blocked or r.rate <= 0]
    cap = int(budget.cpu_minutes)
    live = live[:60]
    weights = [max(1, int(math.ceil(r.cost))) for r in live]
    if live and len(live) * (cap + 1) <= max_cells:
        _, idx = knapsack([(r.rate * r.cost, w) for r, w in zip(live, weights)], cap)
        chosen = [live[i] for i in idx]
    else:
        chosen, used = [], 0.0
        for r in live:
            if used + r.cost <= budget.cpu_minutes:
                chosen.append(r)
                used += r.cost
    keep, real = [], 0
    for r in sorted(chosen, key=lambda r: r.rank):
        if r.item.real_data:
            if real >= budget.real_data_slots:
                deferred.append((r.item.item_id, "real-data slot limit"))
                continue
            real += 1
        keep.append(r)
    picked = {r.item.item_id for r in keep}
    deferred += [(r.item.item_id, "did not fit the round budget") for r in live if r.item.item_id not in picked and r.item.item_id not in {d[0] for d in deferred}]
    return keep, deferred


def plan_round(state: PriorityState, items: Sequence[ResearchItem], now, budget: ComputeBudget, seed: int, ctx: PolicyContext | None = None) -> PriorityPlan:
    """Rank, then fill the round: the exploit set by the mean model, plus a small explore set (posterior sampling) so a wrongly
    low-ranked family can prove itself. The explore share follows research_policy.epsilon_schedule and is capped by
    `explore_fraction`."""
    ranked = rank(state, items, now, budget, ctx)
    keep, deferred = _select(ranked, budget)
    rng = np.random.default_rng(seed)
    explored: list = []
    n_explore = int(round(min(state.cfg.explore_fraction, epsilon_schedule(len(state.history))) * max(len(keep), 1)))
    if n_explore and len(ranked) > len(keep):
        sampled = rank(state, items, now, budget, ctx, theta=state.model.sample_theta(rng))
        used = sum(r.cost for r in keep)
        taken = {r.item.item_id for r in keep}
        for r in sampled:
            if len(explored) >= n_explore:
                break
            if r.blocked or r.item.item_id in taken or used + r.cost > budget.cpu_minutes:
                continue
            if r.item.real_data and sum(1 for k in keep if k.item.real_data) >= budget.real_data_slots:
                continue
            explored.append(r.item.item_id)
            used += r.cost
            taken.add(r.item.item_id)
    by_id = {r.item.item_id: r for r in ranked}
    sel_ids = [r.item.item_id for r in keep] + explored
    chosen = [by_id[i] for i in sel_ids]
    total = sum(r.cost for r in chosen) or 1.0
    by_p: dict = {}
    by_f: dict = {}
    for r in chosen:
        by_p[r.item.problem.value] = by_p.get(r.item.problem.value, 0.0) + r.cost / total
        by_f[r.item.family] = by_f.get(r.item.family, 0.0) + r.cost / total
    notes = []
    if state.model.trust() == 0.0:
        notes.append("no trusted learned model yet: ranking follows the conceptual formula plus the objective hierarchy")
    v = state.last_validation
    return PriorityPlan(str(now), tuple(ranked), tuple(sel_ids), tuple(explored), tuple(deferred), float(sum(r.cost for r in chosen)),
                        by_p, by_f, state.model.trust(), v.label.value if v else "never validated", tuple(notes))


def step(state: PriorityState, now, items: Sequence[ResearchItem], results: Sequence[RealisedValue] = (), budget: ComputeBudget | None = None,
         seed: int = 0, ctx: PolicyContext | None = None) -> PriorityPlan:
    """THE public entry (rule 25). Learn from finished experiments (matured before `now`), refine the objective weights when due,
    revalidate when due, then rank and plan this round's compute. Deterministic given `seed`."""
    n0 = len(state.history)
    absorb(state, results, now)
    if len(state.history) // state.refine_every > n0 // state.refine_every:
        refine_weights(state)
    budget = budget or ComputeBudget(cpu_minutes=240.0, ram_gb_free=8.0)
    return plan_round(state, items, now, budget, seed, ctx)


# ------------------------------------------------------------------------------------------------------ diagnostics

def why_not_top(plan: PriorityPlan, item_id: str) -> str:
    """Why an item ranks below the leader: the ratio of value, cost and each multiplier, so 'why was that skipped?' has an answer."""
    by = {r.item.item_id: r for r in plan.ranked}
    top = next((r for r in plan.ranked if not r.blocked), None)
    r = by.get(item_id)
    if r is None or top is None:
        return f"{item_id}: not in the plan"
    if r.blocked:
        return f"{item_id}: blocked ({r.blocked})"
    if r.item.item_id == top.item.item_id:
        return f"{item_id} is the top item"
    parts = [f"predicted value {r.value:.4f} vs {top.value:.4f}", f"cost {r.cost:.0f}m vs {top.cost:.0f}m"]
    for k in r.multipliers:
        if abs(r.multipliers[k] - top.multipliers.get(k, 1.0)) > 0.01:
            parts.append(f"{k} x{r.multipliers[k]:.2f} vs x{top.multipliers.get(k, 1.0):.2f}")
    return f"{item_id} is #{r.rank} behind {top.item.item_id}: " + "; ".join(parts)


def rank_stability(state: PriorityState, items: Sequence[ResearchItem], now, seed: int, n_draws: int = 100, k: int = 3) -> dict:
    """How much would the ranking change if the model's coefficients were a plausible draw from their posterior? Reports the share
    of draws with the same #1 and the mean overlap of the top-k. A plan that flips on a coefficient wobble is not a decision."""
    base = [r for r in rank(state, items, now) if not r.blocked]
    if len(base) < 2:
        return {"n_items": len(base), "top1_same": 1.0, "topk_overlap": 1.0}
    rng = np.random.default_rng(seed)
    top1, overlap = 0, []
    bset = {r.item.item_id for r in base[:k]}
    for _ in range(n_draws):
        draw = [r for r in rank(state, items, now, theta=state.model.sample_theta(rng)) if not r.blocked]
        top1 += int(draw[0].item.item_id == base[0].item.item_id)
        overlap.append(len(bset & {r.item.item_id for r in draw[:k]}) / k)
    return {"n_items": len(base), "top1_same": top1 / n_draws, "topk_overlap": float(np.mean(overlap))}


def calibration_report(state: PriorityState, bins: int = 4) -> dict:
    """Progressive validation: every result was predicted BEFORE it was learned. Reports rank correlation of prediction and
    outcome, the log-ratio bias (does the engine over-promise?), and mean realised value by predicted quantile (a well-ordered
    engine has rising means)."""
    if len(state.prequential) < 8:
        return {"n": len(state.prequential), "verdict": "INSUFFICIENT"}
    pred = np.array([p for p, _, _ in state.prequential])
    real = np.array([r for _, r, _ in state.prequential])
    order = np.argsort(pred)
    groups = np.array_split(order, bins)
    means = [float(real[g].mean()) for g in groups]
    bias = float(np.mean(np.log((real + 1e-3) / (pred + 1e-3))))
    mono = all(b >= a - 1e-9 for a, b in zip(means, means[1:]))
    return {"n": len(pred), "spearman": _spearman(pred, real), "log_bias": bias, "mean_realised_by_quantile": means,
            "monotone": mono, "verdict": "ORDERED" if mono and _spearman(pred, real) > 0.2 else "WEAK"}


def objective_balance(plan: PriorityPlan, weights: ObjectiveWeights) -> dict:
    """Compare where the plan spends compute with the objective hierarchy. A plan that gives SECONDARY metrics more compute than
    LOSS_AVOIDANCE is flagged: that is the 'optimising the wrong metric' failure of section 43."""
    share = dict(plan.share_by_problem)
    flags = []
    for a, b in zip(OBJECTIVE_ORDER, OBJECTIVE_ORDER[1:]):
        if weights.weight(a) >= weights.weight(b) and share.get(b.value, 0.0) > share.get(a.value, 0.0) + 0.25:
            flags.append(f"{b.value} gets {share[b.value]:.0%} of compute but ranks below {a.value} ({share.get(a.value, 0.0):.0%})")
    return {"share": share, "flags": flags}


def wrong_metric_audit(state: PriorityState) -> dict:
    """Section 43: does the engine favour volume over substance? Reports, per family, how often its findings failed to survive
    out of sample and how much compute it took. A family that consumes compute and does not survive should be losing priority."""
    fam: dict = {}
    for rv in state.history:
        d = fam.setdefault(rv.item.family, {"n": 0, "failed_oos": 0, "minutes": 0.0, "value": 0.0})
        d["n"] += 1
        d["failed_oos"] += int(rv.survived_oos is False)
        d["minutes"] += rv.cost_minutes
        d["value"] += realised_scalar(rv, state.weights(), state.cfg)
    total_min = sum(d["minutes"] for d in fam.values()) or 1.0
    for f, d in fam.items():
        d["oos_failure_rate"] = d["failed_oos"] / d["n"]
        d["compute_share"] = d["minutes"] / total_min
        d["value_per_minute"] = d["value"] / max(d["minutes"], 1.0)
    bad = sorted(f for f, d in fam.items() if d["n"] >= 5 and d["oos_failure_rate"] > 0.6 and d["compute_share"] > 0.2)
    return {"families": fam, "wasteful": bad}


def state_to_json(state: PriorityState) -> str:
    """Canonical JSON of the learned state (model observations, waste history, evidence, validation label). RealisedValue
    records themselves are not stored -- only what the model learned from them -- so the file holds no experiment text."""
    return json.dumps({"model": state.model.to_dict(), "waste": state.waste.to_dict(), "evidence": dataclasses.asdict(state.evidence),
                       "done": sorted(state.done_ids), "validated_at_n": state.validated_at_n,
                       "prequential": [[p, r, f] for p, r, f in state.prequential]}, sort_keys=True)


def state_from_json(text: str) -> PriorityState:
    d = json.loads(text)
    model = PriorityModel.from_dict(d["model"])
    st = PriorityState(cfg=model.cfg, model=model, waste=WasteTracker.from_dict(d["waste"], model.cfg), evidence=VolatilityEvidence(**d["evidence"]),
                       policy=PriorityFunction())
    st.done_ids = set(d["done"])
    st.validated_at_n = int(d["validated_at_n"])
    st.prequential = [(float(p), float(r), f) for p, r, f in d["prequential"]]
    return st


# ------------------------------------------------------------------------------------------------------ planted research worlds

@dataclass(frozen=True)
class WorldFamily:
    """One line of work in a planted world. `est_*` is what the estimator CLAIMS before running (same for rival families on
    purpose: the estimator cannot tell them apart); `true_*` is what the experiment really delivers."""
    name: str
    problem: Problem
    share: float                                    # fraction of the candidate items that belong to this family
    est: Mapping                                    # ExperimentValue field -> (lo, hi) of the uniform claimed estimate
    true: Mapping                                   # ExperimentValue field -> (mean, sd) of the delivered value
    cost: tuple = (20.0, 40.0)
    survive_p: float = 0.9                          # probability the finding replicates out of sample
    decay: float = 1.0                              # true values multiply by decay ** (times the family has been run)


class PlantedWorld:
    """A synthetic research environment with a hidden truth (test types G, H, I of contract section 46). It generates candidate
    items whose ESTIMATES look alike across families and runs them against a truth the engine cannot see, so any preference the
    engine develops has to be learned from outcomes."""

    def __init__(self, families: Sequence[WorldFamily], seed: int):
        tot = sum(f.share for f in families)
        if not families or tot <= 0:
            raise PriorityError("a planted world needs at least one family with positive share")
        self.families = tuple(families)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.p = np.array([f.share / tot for f in families])
        self.runs: dict = {}
        self.counter = 0

    def by_name(self, name: str) -> WorldFamily:
        for f in self.families:
            if f.name == name:
                return f
        raise PriorityError(f"no family {name!r}")

    def make_items(self, n: int, created: str) -> list:
        out = []
        for _ in range(n):
            fam = self.families[int(self.rng.choice(len(self.families), p=self.p))]
            kw = {}
            for fld, (lo, hi) in fam.est.items():
                kw[fld] = float(self.rng.uniform(lo, hi))
            cost = float(self.rng.uniform(*fam.cost))
            self.counter += 1
            value = ExperimentValue(compute_cost=cost, **{k: (v if k == "information_gain" else _clip01(v)) for k, v in kw.items()})
            words = " ".join(f"w{int(x)}" for x in self.rng.integers(0, 900, size=4))
            out.append(ResearchItem(item_id=f"i{self.seed}_{self.counter}", text=f"planted experiment of kind {fam.name} probing {words}",
                                    problem=fam.problem, family=fam.name, value=value, created=created, question_id=f"q_{self.seed}_{self.counter}"))
        return out

    def run(self, item: ResearchItem, matured_at: str) -> RealisedValue:
        fam = self.by_name(item.family)
        k = self.runs.get(fam.name, 0)
        self.runs[fam.name] = k + 1
        scale = fam.decay ** k
        kw = {}
        for fld, (mu, sd) in fam.true.items():
            x = max(0.0, float(self.rng.normal(mu * scale, sd * scale)))
            kw[fld] = x if fld == "information_gain" else _clip01(x)
        survived = bool(self.rng.random() < fam.survive_p)
        return RealisedValue(item=item, realised=ExperimentValue(compute_cost=item.cost, **kw), cost_minutes=item.cost,
                             matured_at=matured_at, survived_oos=survived)


def world_information(seed: int = 0) -> PlantedWorld:
    """Test G: experiment kind A has huge information and decision value, kind B tiny gains -- and their estimates are identical.
    The engine can only find out by running them."""
    est = {"information_gain": (0.2, 0.6), "decision_value": (0.02, 0.06), "uncertainty_reduction": (0.3, 0.6), "transfer_potential": (0.3, 0.6),
           "volatility_value": (0.02, 0.06)}
    return PlantedWorld([
        WorldFamily("A_new_signal_search", Problem.VOLATILITY, 0.25, est, {"information_gain": (2.5, 0.5), "decision_value": (0.30, 0.05), "volatility_value": (0.30, 0.05)},
                    cost=(25.0, 45.0), survive_p=0.9),
        WorldFamily("B_parameter_tuning", Problem.VOLATILITY, 0.75, est, {"information_gain": (0.03, 0.01), "decision_value": (0.004, 0.002), "volatility_value": (0.004, 0.002)},
                    cost=(25.0, 45.0), survive_p=0.4)], seed)


def world_loss(seed: int = 0) -> PlantedWorld:
    """Test H: a small improvement in avoiding losses is worth more than a large number of tiny winner improvements. Estimates
    look the same (both claim ~0.03); the winner tweaks are filed under VOLATILITY, the objective the hierarchy ranks FIRST, so
    the prior hierarchy is actively wrong here and the engine has to unlearn it from outcomes."""
    est_w = {"information_gain": (0.1, 0.3), "decision_value": (0.02, 0.04), "volatility_value": (0.02, 0.04), "uncertainty_reduction": (0.3, 0.5),
             "transfer_potential": (0.3, 0.5)}
    est_l = {"information_gain": (0.1, 0.3), "decision_value": (0.02, 0.04), "loss_reduction_value": (0.02, 0.04), "uncertainty_reduction": (0.3, 0.5),
             "transfer_potential": (0.3, 0.5)}
    return PlantedWorld([
        WorldFamily("W_winner_tweaks", Problem.VOLATILITY, 0.80, est_w, {"decision_value": (0.006, 0.002), "volatility_value": (0.006, 0.002), "information_gain": (0.02, 0.01)},
                    cost=(15.0, 25.0), survive_p=0.6),
        WorldFamily("L_loss_regime", Problem.LOSS_AVOIDANCE, 0.20, est_l, {"loss_reduction_value": (0.20, 0.04), "decision_value": (0.05, 0.02), "information_gain": (0.4, 0.1)},
                    cost=(15.0, 25.0), survive_p=0.85)], seed)


def world_waste(seed: int = 0) -> PlantedWorld:
    """Test I: family W keeps producing negligible, shrinking improvements; family U produces a steady modest gain. Estimates match."""
    est = {"information_gain": (0.2, 0.4), "decision_value": (0.03, 0.05), "uncertainty_reduction": (0.3, 0.5), "transfer_potential": (0.3, 0.5),
           "volatility_value": (0.03, 0.05)}
    return PlantedWorld([
        WorldFamily("W_repeated_tuning", Problem.VOLATILITY, 0.5, est, {"decision_value": (0.010, 0.003), "volatility_value": (0.010, 0.003), "information_gain": (0.03, 0.01)},
                    cost=(20.0, 30.0), survive_p=0.5, decay=0.7),
        WorldFamily("U_steady_search", Problem.VOLATILITY, 0.5, est, {"decision_value": (0.06, 0.01), "volatility_value": (0.06, 0.01), "information_gain": (0.4, 0.05)},
                    cost=(20.0, 30.0), survive_p=0.9)], seed)


def world_null(seed: int = 0) -> PlantedWorld:
    """Null world: every family delivers the same thing. A learned priority model must find no reliable difference (and must not
    call itself validated)."""
    est = {"information_gain": (0.2, 0.6), "decision_value": (0.02, 0.06), "uncertainty_reduction": (0.3, 0.6), "transfer_potential": (0.3, 0.6),
           "volatility_value": (0.02, 0.06)}
    same = {"information_gain": (0.2, 0.1), "decision_value": (0.03, 0.02), "volatility_value": (0.03, 0.02)}
    return PlantedWorld([WorldFamily(f"N{i}", Problem.VOLATILITY, 1.0, est, same, cost=(20.0, 40.0), survive_p=0.7) for i in range(4)], seed)


@dataclass(frozen=True)
class SimResult:
    policy: str
    total_value: float
    per_round: tuple
    family_minutes: Mapping                 # family -> tuple of minutes per round
    last_third_value: float
    state: PriorityState | None
    offered: Mapping = field(default_factory=dict)       # family -> tuple of items offered per round
    chosen: Mapping = field(default_factory=dict)        # family -> tuple of items chosen per round

    def selection_rate(self, family: str, rounds: slice = slice(None)) -> float:
        """Of the items of `family` offered in these rounds, the fraction the policy chose: what preference looks like when
        availability differs between families."""
        off = sum(self.offered.get(family, ())[rounds])
        return sum(self.chosen.get(family, ())[rounds]) / off if off else 0.0

    def share(self, family: str, rounds: slice = slice(None)) -> float:
        tot = sum(sum(v[rounds]) for v in self.family_minutes.values()) or 1.0
        return sum(self.family_minutes.get(family, ())[rounds]) / tot


def simulate_policy(world: PlantedWorld, policy: str, rounds: int, seed: int, items_per_round: int = 12, minutes_per_round: float = 120.0,
                    cfg: PriorityConfig | None = None, weights: ObjectiveWeights | None = None) -> SimResult:
    """Run a research policy against a planted world for `rounds` rounds. policy: 'learned' (this module's engine), 'literal'
    (the conceptual formula, never learning), or 'random'. Results of round r mature after round r and are absorbed in round r+1
    -- the engine never sees an outcome on the day it is chosen."""
    import datetime as _dt
    cfg = cfg or PriorityConfig()
    weights = weights or ObjectiveWeights()
    state = new_state(cfg, weights)
    rng = np.random.default_rng(seed)
    day = _dt.date(2001, 1, 1)
    budget = ComputeBudget(cpu_minutes=minutes_per_round, ram_gb_free=16.0, real_data_slots=4)
    pending: list = []
    per_round, fam_min = [], {f.name: [] for f in world.families}
    fam_off = {f.name: [] for f in world.families}
    fam_ch = {f.name: [] for f in world.families}
    for r in range(rounds):
        now = (day + _dt.timedelta(days=r)).isoformat()
        created = (day + _dt.timedelta(days=r - 1)).isoformat()
        items = world.make_items(items_per_round, created)
        if policy == "learned":
            plan = step(state, now, items, pending, budget, seed=seed + r)
            chosen_ids = set(plan.selected)
        else:
            absorb_none = None
            if policy == "literal":
                order = sorted(items, key=lambda it: (-literal_priority(it, cfg), it.item_id))
            elif policy == "random":
                order = [items[i] for i in rng.permutation(len(items))]
            else:
                raise PriorityError(f"unknown policy {policy!r}")
            chosen_ids, used = set(), 0.0
            for it in order:
                if used + it.cost <= minutes_per_round:
                    chosen_ids.add(it.item_id)
                    used += it.cost
        matured = now
        pending = [world.run(it, matured) for it in items if it.item_id in chosen_ids]
        per_round.append(float(sum(realised_scalar(p, weights, cfg) for p in pending)))
        for f in world.families:
            fam_min[f.name].append(float(sum(p.cost_minutes for p in pending if p.item.family == f.name)))
            fam_off[f.name].append(sum(1 for it in items if it.family == f.name))
            fam_ch[f.name].append(sum(1 for p in pending if p.item.family == f.name))
    third = max(1, rounds // 3)
    return SimResult(policy, float(sum(per_round)), tuple(per_round), {k: tuple(v) for k, v in fam_min.items()}, float(sum(per_round[-third:])),
                     state if policy == "learned" else None, {k: tuple(v) for k, v in fam_off.items()}, {k: tuple(v) for k, v in fam_ch.items()})


def compare_policies(world_factory: Callable[[int], PlantedWorld], rounds: int, seeds: Sequence[int], **kw) -> dict:
    """Mean total and last-third realised value of each policy over several seeds, each policy in an identical fresh world."""
    out: dict = {}
    for pol in ("learned", "literal", "random"):
        tot, last = [], []
        for s in seeds:
            res = simulate_policy(world_factory(s), pol, rounds, s, **kw)
            tot.append(res.total_value)
            last.append(res.last_third_value)
        out[pol] = {"total": float(np.mean(tot)), "last_third": float(np.mean(last)), "sd_total": float(np.std(tot))}
    return out


# ------------------------------------------------------------------------------------------------------ the loss book (section 34)

class LossBook:
    """Historical losses with tags (regime, cause, pattern family), so 'which failure regime carries the largest 5 % of losses'
    is a computed number and not a guess. Every entry is dated by the day its outcome matured and is invisible to any query whose
    `now` is not later (fail closed)."""

    def __init__(self):
        self.rows: list = []                        # (matured_at, magnitude, tags)

    def add(self, magnitude: float, tags: Sequence[str], matured_at, now=None) -> None:
        if math.isnan(magnitude) or magnitude < 0:
            raise PriorityError("a loss magnitude is a non-negative number")
        if now is not None:
            require_past(matured_at, now, "loss entry")
        for t in tags:
            leak = identity_leak(t)
            if leak:
                raise FirewallBreach(f"loss tag {t!r} {leak}")
        self.rows.append((str(matured_at), float(magnitude), tuple(sorted(tags))))

    def _visible(self, now) -> list:
        return [r for r in self.rows if to_ts(r[0]) < to_ts(now)]

    def total(self, now) -> float:
        return float(sum(m for _, m, _ in self._visible(now)))

    def worst(self, now, top_frac: float = 0.05) -> list:
        vis = sorted(self._visible(now), key=lambda r: -r[1])
        k = max(1, int(math.ceil(top_frac * len(vis)))) if vis else 0
        return vis[:k]

    def tag_share(self, now) -> dict:
        tot = self.total(now)
        out: dict = {}
        for _, m, tags in self._visible(now):
            for t in tags:
                out[t] = out.get(t, 0.0) + m
        return {t: v / tot for t, v in sorted(out.items())} if tot > 0 else {}

    def tail_tag_share(self, now, top_frac: float = 0.05) -> dict:
        """Each tag's share of the WORST losses' magnitude: the regimes that carry the catastrophes, whatever their overall share."""
        tail = self.worst(now, top_frac)
        tot = sum(m for _, m, _ in tail)
        out: dict = {}
        for _, m, tags in tail:
            for t in tags:
                out[t] = out.get(t, 0.0) + m
        return {t: v / tot for t, v in sorted(out.items())} if tot > 0 else {}

    def persistence(self, tag: str, now, blocks: int = 4) -> float:
        """Share of equal-length time blocks in which the tag appears: a regime that shows up in every era will probably recur; one
        that appeared once, long ago, probably will not. 0 with fewer entries than blocks."""
        vis = sorted(self._visible(now), key=lambda r: r[0])
        if len(vis) < blocks:
            return 0.0
        parts = np.array_split(np.arange(len(vis)), blocks)
        return float(sum(1 for p in parts if any(tag in vis[i][2] for i in p)) / blocks)

    def regime_value(self, tag: str, now, p_isolate: float, p_actionable: float, capture: float = 0.6, tail_weight: float = 0.5) -> float:
        """Expected loss avoided by researching `tag`: its loss share (a blend of overall and worst-tail share -- catastrophes count
        more than their frequency) times the chance research isolates it, can act on it, and it persists."""
        overall = self.tag_share(now).get(tag, 0.0)
        tail = self.tail_tag_share(now).get(tag, 0.0)
        share = (1 - tail_weight) * overall + tail_weight * tail
        return expected_loss_avoided(min(share, 1.0), p_isolate, p_actionable, self.persistence(tag, now), capture)

    def ranked_regimes(self, now, p_isolate: float = 0.5, p_actionable: float = 0.5) -> list:
        return sorted(((t, self.regime_value(t, now, p_isolate, p_actionable)) for t in self.tag_share(now)), key=lambda kv: (-kv[1], kv[0]))


def evsi_decision_value(prior_mean: float, prior_sd: float, noise_sd: float, n_new: float, stake: float, threshold: float = 0.0) -> float:
    """Decision value of an experiment from the value of sample information (research_policy.evsi_normal), scaled onto [0, 1] by
    the stake at risk: how much better the adopt/reject decision is expected to be once the experiment has run."""
    from engine.learning.research_policy import decision_value_factor, evsi_normal
    return float(decision_value_factor(evsi_normal(prior_mean, prior_sd, noise_sd, n_new, threshold), stake))


def evidence_from_hits(hits: Sequence[int], base_rate: float, n_years: int, out_of_sample: bool) -> VolatilityEvidence:
    """Volatility evidence from 0/1 outcomes of the selector's picks (1 = the pick moved into the band)."""
    h = np.asarray(hits, float)
    return VolatilityEvidence(precision=float(h.mean()) if h.size else 0.0, base_rate=base_rate, n_eval=int(h.size), n_years=n_years,
                              out_of_sample=out_of_sample)


# ------------------------------------------------------------------------------------------------------ filling gaps in estimates

def estimate_redundancy(item: ResearchItem, history: Sequence[RealisedValue]) -> float:
    """How much of what this experiment would show has already been shown: the highest text similarity to an experiment already
    run in the same family. A repeat of something that FAILED out of sample is half as redundant (replication is legitimate)."""
    from engine.learning.experiment_memory import question_similarity
    best = 0.0
    for r in history:
        if r.item.family != item.family or r.item.item_id == item.item_id:
            continue
        s = question_similarity(item.text, r.item.text)
        if r.survived_oos is False:
            s *= 0.5
        best = max(best, s)
    return _clip01(best)


def estimate_overfit_risk(item: ResearchItem) -> float:
    """Multiple-testing risk from what the item itself says: each earlier test on the same window and each free parameter adds
    risk, saturating. (Family history enters through the learned family effect, not here.)"""
    return _clip01(1.0 - math.exp(-0.08 * item.prior_tests_on_window - 0.03 * item.free_parameters))


def fill_missing(item: ResearchItem, history: Sequence[RealisedValue]) -> ResearchItem:
    """Fill the two value-vector terms the questioner cannot know but the engine can: redundancy (from history) and overfit risk
    (from the forking-paths count). Every other None stays None -- it is unknown, not zero."""
    v = item.value
    red = v.redundancy if v.redundancy is not None else estimate_redundancy(item, history)
    ovf = v.overfit_risk if v.overfit_risk is not None else estimate_overfit_risk(item)
    return replace(item, value=replace(v, redundancy=red, overfit_risk=ovf))


def diversity_cap(ranked: Sequence[RankedItem], max_per_question: int = 2) -> list:
    """Never run more than `max_per_question` experiments for one question in a round: extra ones are BLOCKED with a reason (they
    return next round if the first ones do not settle it). Items without a question id are uncapped."""
    count: dict = {}
    out = []
    for r in ranked:
        q = r.item.question_id
        if q and not r.blocked:
            count[q] = count.get(q, 0) + 1
            if count[q] > max_per_question:
                out.append(replace(r, blocked=f"question {q} already has {max_per_question} experiments this round", rate=0.0))
                continue
        out.append(r)
    out.sort(key=lambda r: (bool(r.blocked), -r.rate, r.item.item_id))
    return [replace(r, rank=i + 1) for i, r in enumerate(out)]


# ------------------------------------------------------------------------------------------------------ explaining and stress-testing

def dominant_terms(state: PriorityState, item: ResearchItem, reference: Sequence[ResearchItem]) -> list:
    """Which value terms make this item rank where it does: each term's contribution to log value, relative to the mean item of a
    reference set, largest first. The answer to 'why is this item high?' and 'which estimate would change my mind?'"""
    th = state.model.theta()
    x = feature_vector(item, state.cfg)
    ref = np.mean([feature_vector(r, state.cfg) for r in reference], axis=0) if reference else np.zeros(DIM)
    rows = [(name, float((x[i] - ref[i]) * th[i])) for i, name in enumerate(FEATURES)]
    rows.append(("objective", float(sum((x[len(FEATURES) + j] - ref[len(FEATURES) + j]) * th[len(FEATURES) + j] for j in range(len(PROBLEMS))))))
    rows.append(("family_effect", state.model.family_effect(item.family)))
    return sorted(rows, key=lambda r: -abs(r[1]))


def marginal_value_of_compute(state: PriorityState, items: Sequence[ResearchItem], now, minutes: Sequence[float]) -> list:
    """Predicted total value of the best plan at each compute budget: [(minutes, value)]. The curve's slope is what one more
    CPU-minute is worth; a flat curve says the round is already saturated with good work (the compute manager, section 18, should
    give the next minute to something else)."""
    out = []
    for m in sorted(minutes):
        budget = ComputeBudget(cpu_minutes=m, ram_gb_free=16.0, real_data_slots=8)
        ranked = rank(state, items, now, budget)
        keep, _ = _select(ranked, budget)
        out.append((float(m), float(sum(r.value for r in keep))))
    return out


def futility_stop(interim_value: float, planned_value: float, fraction_done: float, sd_share: float = 0.5, min_fraction: float = 0.3,
                  p_floor: float = 0.10) -> dict:
    """Section 20: should a RUNNING experiment be stopped early? Projects the final value from what it has delivered so far
    (a random-walk projection with per-unit spread `sd_share` x planned) and stops when the probability of reaching even HALF of
    the planned value is under `p_floor`, but never before `min_fraction` of it has run (an early dip is not evidence)."""
    if not (0.0 < fraction_done <= 1.0) or planned_value <= 0 or interim_value < 0:
        raise PriorityError("futility_stop needs 0 < fraction_done <= 1, planned_value > 0, interim_value >= 0")
    if fraction_done < min_fraction:
        return {"stop": False, "reason": f"only {fraction_done:.0%} done; minimum {min_fraction:.0%}", "p_reach_half": None}
    remaining_mean = interim_value / fraction_done * (1.0 - fraction_done)
    remaining_sd = sd_share * planned_value * math.sqrt(1.0 - fraction_done)
    p = 1.0 - float(norm.cdf((0.5 * planned_value - interim_value - remaining_mean) / max(remaining_sd, EPS)))
    return {"stop": p < p_floor, "p_reach_half": p, "projected": interim_value + remaining_mean,
            "reason": f"probability of reaching half the planned value is {p:.3f}"}


def dormant_families(state: PriorityState, now, probe_days: float = 90.0) -> list:
    """Families the waste tracker has put to sleep, with the date a cheap re-probe is due. A barren line is not deleted (a regime
    change can revive it): it sleeps at a floor multiplier and is probed every `probe_days`."""
    out = []
    for fam, rows in sorted(state.waste.hist.items()):
        if state.waste.verdict(fam)["verdict"] != "STOP":
            continue
        last = rows[-1][0]
        age = (to_ts(now) - to_ts(last)).total_seconds() / 86400.0
        out.append({"family": fam, "barren_streak": state.waste.streak(fam), "last_result": last, "days_idle": age, "probe_due": age >= probe_days})
    return out


def lane_report(state: PriorityState, now) -> dict:
    """One-glance health of the priority engine for the research-brain health system (section 37): model trust and validation
    label, calibration verdict, dormant families, hierarchy violations, and the OOS-failure audit."""
    v = state.last_validation
    return {"n_results": len(state.history), "trust": state.model.trust(), "validation": v.label.value if v else "never validated",
            "validation_gain": v.mean_gain if v else None, "calibration": calibration_report(state)["verdict"],
            "dormant": [d["family"] for d in dormant_families(state, now)], "hierarchy_violations": state.weights().hierarchy_violations(),
            "wasteful_families": wrong_metric_audit(state)["wasteful"], "direction_lane_open": state.evidence.openness()}
