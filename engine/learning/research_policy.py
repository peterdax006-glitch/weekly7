"""Research policy (contract C62 sections 35 and 36; checklist G05, G10, G11, G12, C17; Bible canon C58-C62).

Two questions, one module. Section 36: given many possible next experiments, which has the highest expected value of
information? An explicit priority function

    priority = EIG x relevance x uncertainty x transfer_potential x feasibility x expected_decision_value  x  penalties

where EIG is a real expected information gain (mutual information between hypotheses and outcomes, a Beta-Binomial entropy
reduction, or a Gaussian sample-size gain) and the penalties are duplicate experiments, known-low-value families,
overfit-prone experiments, unavailable data and excessive compute. Section 35: how is research effort split across the ten
research targets? A Thompson-sampling + UCB allocator over per-target yield, with a floor for every open target and a cap
on 'known reliable' so compute is never spent only re-tuning what is already known. Compute-aware selection is an exact
knapsack under CPU minutes, RAM and the real-data slot limit. The policy learns: realised information gain calibrates
predicted EIG, and MetaAdvice from the meta-learner (sec. 37) moves the overfit penalty and the allocator priors.

Everything time-aware takes `now` and fails closed; every draw is seeded. IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import digamma, gammaln

from .core import FirewallBreach, Subsystem, ValidationLabel, stable_hash
from .experiment_memory import DuplicateStatus, to_ts

LABEL = ValidationLabel.NOT_VALIDATED.value
LN2 = math.log(2.0)
EPS = 1e-9


class ResearchTarget(str, enum.Enum):
    """Section 35: the places research effort can go."""
    KNOWN_RELIABLE = "KNOWN_RELIABLE"
    CONTRADICTION = "CONTRADICTION"
    FAILURE = "FAILURE"
    MISSED_WINNER = "MISSED_WINNER"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    WEAK_PATTERN = "WEAK_PATTERN"
    UNKNOWN_AREA = "UNKNOWN_AREA"
    INTERACTION = "INTERACTION"
    NEW_REPRESENTATION = "NEW_REPRESENTATION"
    DATA_QUALITY = "DATA_QUALITY"

    def __str__(self):
        return self.value

    @classmethod
    def parse(cls, v):
        return v if isinstance(v, cls) else cls(str(v))


TARGETS = tuple(ResearchTarget)
EXPLOIT_TARGETS = frozenset({ResearchTarget.KNOWN_RELIABLE})
ALWAYS_OPEN = frozenset({ResearchTarget.UNKNOWN_AREA, ResearchTarget.NEW_REPRESENTATION, ResearchTarget.INTERACTION})


# ------------------------------------------------------------------------------------------------ information theory

def entropy_bits(p: Iterable[float]) -> float:
    """Shannon entropy in bits of a probability vector (zeros contribute nothing; must sum to ~1)."""
    arr = np.asarray(list(p), float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < -1e-12) or abs(arr.sum() - 1.0) > 1e-6:
        raise ValueError(f"not a probability vector (sum={arr.sum():.6f})")
    arr = arr[arr > 0]
    return float(-(arr * np.log2(arr)).sum())


def mutual_information(prior: Mapping[str, float], likelihood: Mapping[str, Mapping[str, float]]) -> float:
    """I(H; O) in bits. prior[h] = P(h); likelihood[h][o] = P(o | h). This is the expected reduction in uncertainty about
    which hypothesis is true from observing the experiment's outcome."""
    hs = list(prior)
    outs = sorted({o for h in hs for o in likelihood.get(h, {})})
    if len(hs) < 2 or not outs:
        return 0.0
    for h in hs:
        tot = sum(likelihood.get(h, {}).values())
        if abs(tot - 1.0) > 1e-6:
            raise ValueError(f"likelihood of {h!r} sums to {tot:.6f}, not 1")
    h_prior = entropy_bits(prior.values())
    exp_post = 0.0
    for o in outs:
        p_o = sum(prior[h] * likelihood[h].get(o, 0.0) for h in hs)
        if p_o <= 0:
            continue
        post = [prior[h] * likelihood[h].get(o, 0.0) / p_o for h in hs]
        exp_post += p_o * entropy_bits(post)
    return max(0.0, h_prior - exp_post)


def eig_from_expected(hyps: Sequence, expected: Sequence) -> float:
    """EIG from duck-typed Hypothesis (.hid, .prior) and ExpectedOutcome (.hid, .outcome, .probability) lists."""
    prior = {h.hid: float(h.prior) for h in hyps}
    like: dict = {h.hid: {} for h in hyps}
    for o in expected:
        if o.hid in like:
            like[o.hid][o.outcome] = like[o.hid].get(o.outcome, 0.0) + float(o.probability)
    return mutual_information(prior, like)


def beta_entropy_bits(a: float, b: float) -> float:
    """Differential entropy of Beta(a, b) in bits."""
    if a <= 0 or b <= 0:
        raise ValueError("beta parameters must be positive")
    ln_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
    h = ln_beta - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)
    return float(h / LN2)


def eig_beta_binomial(a: float, b: float, m: int) -> float:
    """Expected entropy reduction (bits) about a success rate with a Beta(a, b) belief after m new Bernoulli trials.
    Exact: sums the beta-binomial predictive over every possible count k."""
    if m <= 0:
        return 0.0
    k = np.arange(m + 1)
    logp = (gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)
            + gammaln(a + k) + gammaln(b + m - k) - gammaln(a + b + m)
            - (gammaln(a) + gammaln(b) - gammaln(a + b)))
    p = np.exp(logp)
    p /= p.sum()
    post = np.array([beta_entropy_bits(a + kk, b + m - kk) for kk in k])
    return max(0.0, beta_entropy_bits(a, b) - float((p * post).sum()))


def eig_normal_mean(n_now: float, n_new: float) -> float:
    """Bits gained about a Gaussian mean when the effective sample grows from n_now to n_now + n_new."""
    if n_now <= 0 or n_new <= 0:
        return 0.0 if n_new <= 0 else 0.5 * math.log2(1.0 + n_new)
    return 0.5 * math.log2((n_now + n_new) / n_now)


@dataclass(frozen=True)
class InfoModel:
    """How a candidate's information gain is computed. kind:
    discrete  params: hypotheses [(hid, prior)], expected [(hid, outcome, probability)]
    beta      params: a, b, m          (a success rate: hit rate, survival rate)
    normal    params: n_now, n_new     (a mean effect: weekly return delta)
    proxy     params: resolvability    (no model: EIG guessed from uncertainty; haircut so it cannot outrank a modelled one)"""
    kind: str
    params: Mapping = field(default_factory=dict)

    def bits(self, uncertainty: float = 0.5, proxy_haircut: float = 0.5) -> tuple:
        """(bits, source). A malformed model degrades to the proxy and SAYS so in `source`."""
        try:
            if self.kind == "discrete":
                hyps = [type("H", (), {"hid": h, "prior": p}) for h, p in self.params["hypotheses"]]
                exp = [type("O", (), {"hid": h, "outcome": o, "probability": p}) for h, o, p in self.params["expected"]]
                return eig_from_expected(hyps, exp), "discrete"
            if self.kind == "beta":
                return eig_beta_binomial(float(self.params["a"]), float(self.params["b"]), int(self.params["m"])), "beta"
            if self.kind == "normal":
                return eig_normal_mean(float(self.params["n_now"]), float(self.params["n_new"])), "normal"
            if self.kind == "proxy":
                res = float(self.params.get("resolvability", 0.5))
                return proxy_haircut * uncertainty * max(0.0, min(1.0, res)), "proxy"
        except (KeyError, ValueError, TypeError):
            pass
        return proxy_haircut * uncertainty * 0.25, "proxy_fallback"


# ------------------------------------------------------------------------------------------------ candidates & context

@dataclass(frozen=True)
class ComputeCost:
    cpu_minutes: float = 10.0
    ram_gb: float = 1.0
    real_data: bool = False
    wall_minutes: float = 0.0

    def check(self) -> list:
        errs = []
        for f in ("cpu_minutes", "ram_gb", "wall_minutes"):
            v = getattr(self, f)
            if v < 0 or math.isnan(v):
                errs.append(f"cost.{f}={v!r} negative")
        return errs


@dataclass(frozen=True)
class ComputeBudget:
    """What one planning round may spend. Follows CONTEXT rules 10-11: shared RAM, one real-data job at a time by default."""
    cpu_minutes: float
    ram_gb_free: float
    safety_ram_gb: float = 2.5
    max_ram_per_job_gb: float = 4.0
    real_data_slots: int = 1

    def check(self) -> list:
        errs = []
        if self.cpu_minutes < 0 or self.ram_gb_free < 0 or self.real_data_slots < 0:
            errs.append("budget: negative resource")
        return errs

    def ram_ceiling(self) -> float:
        return max(0.0, min(self.max_ram_per_job_gb, self.ram_gb_free - self.safety_ram_gb))

    def admits(self, cost: ComputeCost) -> tuple:
        """(ok, reason). Reason is always explicit so a deferral is never silent."""
        if cost.cpu_minutes > self.cpu_minutes + EPS:
            return False, f"needs {cost.cpu_minutes:.0f} cpu-min, round has {self.cpu_minutes:.0f}"
        if cost.ram_gb > self.ram_ceiling() + EPS:
            return False, f"needs {cost.ram_gb:.1f} GB, ceiling {self.ram_ceiling():.1f} GB (free {self.ram_gb_free:.1f} - safety {self.safety_ram_gb})"
        return True, ""


@dataclass(frozen=True)
class Factors:
    """The multiplicative terms of the priority function, each in [0, 1]."""
    uncertainty: float = 0.5
    relevance: float = 0.5
    transfer_potential: float = 0.5
    feasibility: float = 1.0
    decision_value: float = 0.5

    def check(self) -> list:
        return [f"factor {k}={v!r} outside [0,1]" for k, v in self.__dict__.items()
                if not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0)]


@dataclass(frozen=True)
class Candidate:
    cid: str
    question: str
    target: ResearchTarget
    created_at: str
    factors: Factors = field(default_factory=Factors)
    info: InfoModel = field(default_factory=lambda: InfoModel("proxy", {"resolvability": 0.5}))
    cost: ComputeCost = field(default_factory=ComputeCost)
    subsystem: str = ""
    family: str = ""                                # e.g. 'pattern:volatility', 'learner:memory_bank'
    config: Mapping = field(default_factory=dict)
    data_needs: tuple = ()
    overfit_hint: float = 0.0                       # caller's own 0..1 belief that this search invites overfitting
    free_parameters: int = 0
    prior_tests_on_window: int = 0                  # how many earlier experiments already looked at the same window
    evidence: tuple = ()
    magnitude: float = 0.0                          # size of the signal that created it (z, loss, contradiction strength)

    def check(self) -> list:
        errs = []
        if not self.cid:
            errs.append("candidate without id")
        if not self.question.strip():
            errs.append(f"{self.cid}: empty question")
        errs += [f"{self.cid}: {e}" for e in self.factors.check() + self.cost.check()]
        if self.subsystem:
            try:
                Subsystem.parse(self.subsystem)
            except ValueError:
                errs.append(f"{self.cid}: unknown subsystem {self.subsystem!r}")
        if not (0.0 <= self.overfit_hint <= 1.0):
            errs.append(f"{self.cid}: overfit_hint outside [0,1]")
        return errs


@dataclass(frozen=True)
class RealisedGain:
    """What an executed experiment actually delivered, so predictions can be calibrated and low-value families identified."""
    cid: str
    target: ResearchTarget
    family: str
    predicted_bits: float
    realised_bits: float                            # entropy reduction of the hypothesis posterior (BeliefUpdate)
    useful: bool                                    # did it change a belief / decision / create knowledge?
    cost_minutes: float
    when: str


def realised_bits(prior: Mapping[str, float], posterior: Mapping[str, float]) -> float:
    """H(prior) - H(posterior), floored at 0: how much narrower the hypothesis set became. (A surprise can raise entropy;
    that is real, but it is 'gain' only when it eliminates something, so negatives clip to 0 and count as not useful.)"""
    if not prior or not posterior:
        return 0.0
    return max(0.0, entropy_bits(prior.values()) - entropy_bits(posterior.values()))


def gain_from_record(rec, cid: str | None = None, predicted_bits: float = 0.0, target: str = "", family: str = "",
                     useful_bits: float = 0.05) -> RealisedGain:
    """Build a RealisedGain from an experiment_memory.ExperimentRecord (duck-typed: needs belief_update, result, experiment)."""
    bu = rec.belief_update
    bits = realised_bits(bu.prior, bu.posterior) if bu is not None else 0.0
    tgt = ResearchTarget.parse(target or rec.experiment.target or ResearchTarget.UNKNOWN_AREA)
    return RealisedGain(cid or rec.experiment_id, tgt, family, predicted_bits, bits, bits >= useful_bits or bool(
        bu is not None and bu.moved >= 0.2), rec.experiment.cost_minutes, rec.result.observed_at if rec.result else rec.recorded_at)


@dataclass(frozen=True)
class MetaAdvice:
    """What the meta-learner (sec. 37) tells the research policy (G12). All values were fitted out of sample; `oos_label`
    carries whether meta-learning itself beat its baseline. With label != VALIDATED the policy trusts it only lightly."""
    family_overfit: Mapping = field(default_factory=dict)       # family -> P(overfit), 0..1
    family_survival: Mapping = field(default_factory=dict)      # family -> P(discovery survives OOS)
    target_yield: Mapping = field(default_factory=dict)         # ResearchTarget value -> P(experiment useful)
    context_transfer: Mapping = field(default_factory=dict)
    explanation_precision: Mapping = field(default_factory=dict)
    fitted_through: str = ""
    n_observations: int = 0
    oos_label: str = ValidationLabel.NOT_VALIDATED.value

    def trust(self) -> float:
        """How much weight the policy gives this advice: 0 with no data, growing with n, halved unless validated OOS."""
        base = self.n_observations / (self.n_observations + 30.0)
        return base * (1.0 if self.oos_label == ValidationLabel.VALIDATED.value else 0.5)

    def check(self) -> list:
        errs = []
        for name in ("family_overfit", "family_survival", "target_yield"):
            for k, v in getattr(self, name).items():
                if not (0.0 <= float(v) <= 1.0):
                    errs.append(f"meta advice {name}[{k}]={v!r} outside [0,1]")
        return errs

    @classmethod
    def empty(cls) -> "MetaAdvice":
        return cls()


DuplicateChecker = Callable[[Candidate], object]      # returns anything with .status and .blocking (DuplicateVerdict)


@dataclass
class PolicyContext:
    now: str
    duplicate_check: DuplicateChecker | None = None
    available_data: frozenset | None = None            # None = every dataset counts as available
    history: Sequence[RealisedGain] = ()
    meta: MetaAdvice = field(default_factory=MetaAdvice.empty)
    pressure: Mapping = field(default_factory=dict)    # ResearchTarget value -> nonnegative open-work pressure


@dataclass(frozen=True)
class PolicyConfig:
    """Every knob is declared here so nothing hides in a function body. Defaults are priors, not tuned values."""
    eig_scale_bits: float = 0.5                     # EIG of this many bits maps to 1 - 1/e
    factor_floor: float = 0.02
    proxy_haircut: float = 0.5
    low_value_gain_bits: float = 0.08               # families averaging less than this are 'known low value'
    low_value_min_trials: int = 3
    low_value_min_penalty: float = 0.25
    overfit_penalty_strength: float = 0.8
    forking_paths_rate: float = 0.10                # each earlier test on the same window costs this much priority
    free_param_rate: float = 0.03
    compute_curvature: float = 2.0                  # penalty exp(-c * (cost/budget)^2)
    duplicate_seen_penalty: float = 0.7
    retest_penalty: float = 0.85
    calibration_prior_strength: float = 5.0
    exploit_cap: float = 0.30                       # KNOWN_RELIABLE may never take more of a round than this
    explore_floor: float = 0.03                     # every open target keeps at least this
    ucb_weight: float = 0.5
    softmax_temperature: float = 0.35
    spend_window: int = 30                          # trailing experiments audited for exploit share

    def check(self) -> list:
        errs = []
        if not 0 < self.exploit_cap <= 1:
            errs.append("exploit_cap must be in (0, 1]")
        if self.explore_floor * len(TARGETS) > 1:
            errs.append("explore_floor x #targets exceeds 1: infeasible")
        if self.softmax_temperature <= 0 or self.eig_scale_bits <= 0:
            errs.append("temperature and eig_scale must be positive")
        return errs


# ------------------------------------------------------------------------------------------------ calibration of EIG

class EigCalibrator:
    """Predicted EIG is a claim; realised entropy reduction is the evidence. Per-target ratio realised/predicted with a
    ridge toward 1 (`prior_strength` pseudo-observations at ratio 1), so a few noisy runs cannot rescale the policy."""

    def __init__(self, prior_strength: float = 5.0):
        self.k = prior_strength
        self.pred: dict = {}
        self.real: dict = {}
        self.n: dict = {}

    def observe(self, gain: RealisedGain) -> None:
        t = ResearchTarget.parse(gain.target).value
        self.pred[t] = self.pred.get(t, 0.0) + max(gain.predicted_bits, 0.0)
        self.real[t] = self.real.get(t, 0.0) + max(gain.realised_bits, 0.0)
        self.n[t] = self.n.get(t, 0) + 1

    def scale(self, target) -> float:
        t = ResearchTarget.parse(target).value
        return (self.real.get(t, 0.0) + self.k * 1.0) / (self.pred.get(t, 0.0) + self.k * 1.0)

    def adjust(self, target, bits: float) -> float:
        return bits * self.scale(target)

    def report(self) -> dict:
        out = {}
        for t in self.n:
            out[t] = {"n": self.n[t], "predicted": self.pred[t], "realised": self.real[t], "scale": self.scale(t),
                      "verdict": "overclaims" if self.scale(t) < 0.7 else "underclaims" if self.scale(t) > 1.5 else "roughly calibrated"}
        return out


# ------------------------------------------------------------------------------------------------ scoring

@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    priority: float
    eig_bits: float
    eig_source: str
    log_terms: Mapping
    penalties: Mapping
    blocked: str = ""

    def explain(self) -> str:
        c = self.candidate
        if self.blocked:
            return f"{c.cid} [{c.target}] BLOCKED: {self.blocked}"
        pen = ", ".join(f"{k}x{v:.2f}" for k, v in self.penalties.items() if v < 0.999) or "none"
        return (f"{c.cid} [{c.target}] priority={self.priority:.5f} eig={self.eig_bits:.3f}b({self.eig_source}) "
                f"unc={c.factors.uncertainty:.2f} rel={c.factors.relevance:.2f} xfer={c.factors.transfer_potential:.2f} "
                f"feas={c.factors.feasibility:.2f} dv={c.factors.decision_value:.2f} penalties: {pen}")


def _floor(x: float, f: float) -> float:
    return max(float(x), f)


def family_mean_gain(history: Sequence[RealisedGain], family: str, target=None) -> tuple:
    """(mean realised bits, n) for a family, optionally within a target. Empty family -> (0.0, 0)."""
    rows = [g for g in history if g.family == family and (target is None or g.target == target)]
    return (sum(g.realised_bits for g in rows) / len(rows) if rows else 0.0), len(rows)


class PriorityFunction:
    """The section-36 priority function with its five penalties. Pure: same inputs, same score."""

    def __init__(self, cfg: PolicyConfig | None = None, calibrator: EigCalibrator | None = None):
        self.cfg = cfg or PolicyConfig()
        errs = self.cfg.check()
        if errs:
            raise ValueError("; ".join(errs))
        self.cal = calibrator or EigCalibrator(self.cfg.calibration_prior_strength)

    # -- penalties, each returns (multiplier in [0,1], hard_block_reason)
    def duplicate_penalty(self, c: Candidate, ctx: PolicyContext) -> tuple:
        if ctx.duplicate_check is None:
            return 1.0, ""
        v = ctx.duplicate_check(c)
        status = getattr(v, "status", DuplicateStatus.NOVEL)
        if getattr(v, "blocking", False):
            return 0.0, getattr(v, "message", "duplicate experiment")
        if status == DuplicateStatus.RETEST_JUSTIFIED:
            return self.cfg.retest_penalty, ""
        if status != DuplicateStatus.NOVEL:
            return self.cfg.duplicate_seen_penalty, ""
        return 1.0, ""

    def low_value_penalty(self, c: Candidate, ctx: PolicyContext) -> float:
        """Shrunk mean realised gain of the family vs the low-value line. Fewer than low_value_min_trials -> no penalty
        (no evidence is not evidence of low value)."""
        if not c.family:
            return 1.0
        mean, n = family_mean_gain(ctx.history, c.family)
        if n < self.cfg.low_value_min_trials:
            return 1.0
        allm = [g.realised_bits for g in ctx.history]
        prior = sum(allm) / len(allm) if allm else self.cfg.low_value_gain_bits
        shrunk = (mean * n + prior * 3.0) / (n + 3.0)
        frac = min(1.0, shrunk / max(self.cfg.low_value_gain_bits, EPS))
        return self.cfg.low_value_min_penalty + (1 - self.cfg.low_value_min_penalty) * frac

    def overfit_penalty(self, c: Candidate, ctx: PolicyContext) -> float:
        risk = c.overfit_hint
        fam = ctx.meta.family_overfit.get(c.family)
        if fam is not None:
            w = ctx.meta.trust()
            risk = max(risk, (1 - w) * risk + w * float(fam))
        pen = 1.0 - self.cfg.overfit_penalty_strength * risk
        pen /= 1.0 + self.cfg.forking_paths_rate * c.prior_tests_on_window
        pen /= 1.0 + self.cfg.free_param_rate * c.free_parameters
        return max(pen, 0.0)

    def data_block(self, c: Candidate, ctx: PolicyContext) -> str:
        if ctx.available_data is None:
            return ""
        missing = [d for d in c.data_needs if d not in ctx.available_data]
        return f"data not available: {', '.join(missing)}" if missing else ""

    def compute_penalty(self, c: Candidate, budget: ComputeBudget | None) -> tuple:
        if budget is None:
            return 1.0, ""
        ok, why = budget.admits(c.cost)
        if not ok:
            return 0.0, why
        ratio = c.cost.cpu_minutes / max(budget.cpu_minutes, EPS)
        return math.exp(-self.cfg.compute_curvature * ratio * ratio), ""

    def score(self, c: Candidate, ctx: PolicyContext, budget: ComputeBudget | None = None) -> ScoredCandidate:
        errs = c.check()
        if errs:
            raise ValueError("; ".join(errs))
        if to_ts(c.created_at) >= to_ts(ctx.now):
            raise FirewallBreach(f"candidate {c.cid} created_at={c.created_at} is not before now={ctx.now}")
        bits, src = c.info.bits(c.factors.uncertainty, self.cfg.proxy_haircut)
        bits = self.cal.adjust(c.target, bits)
        eig_norm = 1.0 - math.exp(-bits / self.cfg.eig_scale_bits)
        pens: dict = {}
        block = self.data_block(c, ctx)
        pens["unavailable_data"] = 0.0 if block else 1.0
        dup, dup_msg = self.duplicate_penalty(c, ctx)
        pens["duplicate"] = dup
        block = block or dup_msg
        pens["known_low_value"] = self.low_value_penalty(c, ctx)
        pens["overfit_prone"] = self.overfit_penalty(c, ctx)
        comp, comp_msg = self.compute_penalty(c, budget)
        pens["excess_compute"] = comp
        block = block or comp_msg
        f = c.factors
        fl = self.cfg.factor_floor
        terms = {"eig": math.log(_floor(eig_norm, fl)), "relevance": math.log(_floor(f.relevance, fl)),
                 "uncertainty": math.log(_floor(f.uncertainty, fl)), "transfer": math.log(_floor(f.transfer_potential, fl)),
                 "feasibility": math.log(_floor(f.feasibility, fl)), "decision_value": math.log(_floor(f.decision_value, fl))}
        pen_prod = 1.0
        for v in pens.values():
            pen_prod *= v
        if block or pen_prod <= 0:
            return ScoredCandidate(c, 0.0, bits, src, terms, pens, blocked=block or "a hard penalty is zero")
        pri = math.exp(sum(terms.values())) * pen_prod
        return ScoredCandidate(c, pri, bits, src, terms, pens)


# ------------------------------------------------------------------------------------------------ section 35: allocation

@dataclass
class TargetStats:
    trials: int = 0
    useful: int = 0
    gain_bits: float = 0.0
    minutes: float = 0.0


def project_box_simplex(raw: Mapping[str, float], lo: Mapping[str, float], hi: Mapping[str, float]) -> dict:
    """Shares s_i = clip(lambda * raw_i, lo_i, hi_i) with sum exactly 1 (bisection on lambda; monotone in lambda).
    Raises ValueError when the box cannot hold a distribution (sum lo > 1 or sum hi < 1)."""
    keys = list(raw)
    L = np.array([lo[k] for k in keys], float)
    H = np.array([hi[k] for k in keys], float)
    R = np.array([max(raw[k], 0.0) for k in keys], float)
    if L.sum() > 1 + 1e-9 or H.sum() < 1 - 1e-9 or np.any(L > H + 1e-12):
        raise ValueError("infeasible share bounds")
    if R.sum() <= 0:
        R = np.ones_like(R)
    lam_lo, lam_hi = 0.0, 1.0
    while np.clip(lam_hi * R, L, H).sum() < 1 and lam_hi < 1e12:
        lam_hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lam_lo + lam_hi)
        if np.clip(mid * R, L, H).sum() < 1:
            lam_lo = mid
        else:
            lam_hi = mid
    s = np.clip(0.5 * (lam_lo + lam_hi) * R, L, H)
    s = s / s.sum() if abs(s.sum() - 1) > 1e-12 and np.all(s > 0) else s
    return {k: float(v) for k, v in zip(keys, s)}


@dataclass(frozen=True)
class Allocation:
    shares: Mapping
    minutes: Mapping
    explore_share: float
    exploit_share: float
    open_targets: tuple
    rationale: tuple

    def check(self) -> list:
        errs = []
        if abs(sum(self.shares.values()) - 1.0) > 1e-6:
            errs.append("allocation shares do not sum to 1")
        return errs


class TargetAllocator:
    """Section 35 as a bandit. Per target: Beta belief that an experiment there is useful, a shrunk mean gain per useful
    result, and a UCB bonus for under-tried targets. Thompson draws make the split stochastic (seeded); floors and the
    KNOWN_RELIABLE cap make it safe. Open-work `pressure` gates targets that currently have nothing to do."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self.stats = {t: TargetStats() for t in TARGETS}

    def observe(self, target, useful: bool, gain_bits: float, minutes: float) -> None:
        s = self.stats[ResearchTarget.parse(target)]
        s.trials += 1
        s.useful += int(bool(useful))
        s.gain_bits += max(gain_bits, 0.0)
        s.minutes += max(minutes, 0.0)

    def total_trials(self) -> int:
        return sum(s.trials for s in self.stats.values())

    def yield_draw(self, t: ResearchTarget, rng: np.random.Generator, meta: MetaAdvice) -> float:
        s = self.stats[t]
        prior = meta.target_yield.get(t.value)
        w = meta.trust()
        a0, b0 = (1.0, 1.0) if prior is None else (1.0 + 2.0 * w * prior * 5, 1.0 + 2.0 * w * (1 - prior) * 5)
        p = rng.beta(a0 + s.useful, b0 + s.trials - s.useful)
        mean_gain = (s.gain_bits + 2.0 * 0.15) / (s.useful + 2.0)         # shrunk toward a 0.15-bit prior
        return float(p * mean_gain)

    def ucb_bonus(self, t: ResearchTarget) -> float:
        n = self.total_trials()
        return math.sqrt(2.0 * math.log(n + 2.0) / (self.stats[t].trials + 1.0))

    def allocate(self, budget_minutes: float, pressure: Mapping, seed: int, meta: MetaAdvice | None = None) -> Allocation:
        meta = meta or MetaAdvice.empty()
        rng = np.random.default_rng(seed)
        press = {t: float(pressure.get(t.value, pressure.get(t, 0.0))) for t in TARGETS}
        pmax = max(press.values()) or 1.0
        open_t = tuple(t for t in TARGETS if press[t] > 0 or t in ALWAYS_OPEN)
        raw, lo, hi, why = {}, {}, {}, []
        for t in TARGETS:
            if t not in open_t:
                raw[t.value], lo[t.value], hi[t.value] = 0.0, 0.0, 0.0
                continue
            y = self.yield_draw(t, rng, meta) + 1e-3
            bonus = 1.0 + self.cfg.ucb_weight * self.ucb_bonus(t)
            need = 0.25 + press[t] / pmax
            raw[t.value] = math.exp(math.log(y * bonus * need) / self.cfg.softmax_temperature)
            lo[t.value] = self.cfg.explore_floor
            hi[t.value] = self.cfg.exploit_cap if t in EXPLOIT_TARGETS else 1.0
        n_open = len(open_t)
        if n_open * self.cfg.explore_floor > 1:
            raise ValueError("more open targets than the floor allows")
        if n_open == 1 and open_t[0] in EXPLOIT_TARGETS:
            hi[open_t[0].value] = 1.0
            why.append("only known-reliable work is open: cap lifted (nothing else to explore)")
        shares = project_box_simplex({k: raw[k] for k in raw if k in {t.value for t in open_t}},
                                     {k: lo[k] for k in raw if k in {t.value for t in open_t}},
                                     {k: hi[k] for k in raw if k in {t.value for t in open_t}})
        for t in TARGETS:
            shares.setdefault(t.value, 0.0)
        exploit = sum(shares[t.value] for t in EXPLOIT_TARGETS)
        if exploit >= self.cfg.exploit_cap - 1e-9:
            why.append(f"known-reliable held at the {self.cfg.exploit_cap:.0%} cap")
        for t in open_t:
            if self.stats[t].trials == 0:
                why.append(f"{t.value}: never tried, UCB bonus {self.ucb_bonus(t):.2f}")
        mins = {k: v * budget_minutes for k, v in shares.items()}
        return Allocation(shares, mins, 1.0 - exploit, exploit, tuple(t.value for t in open_t), tuple(why))

    def report(self) -> dict:
        return {t.value: {"trials": s.trials, "useful": s.useful, "rate": (s.useful / s.trials) if s.trials else None,
                          "gain_bits": s.gain_bits, "minutes": s.minutes,
                          "bits_per_minute": (s.gain_bits / s.minutes) if s.minutes > 0 else None}
                for t, s in self.stats.items()}


def epsilon_schedule(n_done: int, base: float = 0.30, half_life: float = 200.0, floor: float = 0.10) -> float:
    """Exploration fraction that decays with experience but never below `floor`: the world is non-stationary, so the
    system must keep looking (never 'converge' to only exploiting)."""
    return max(floor, base * 0.5 ** (n_done / half_life))


def audit_spend(history: Sequence[RealisedGain], window: int = 30, cap: float = 0.30) -> dict:
    """Did recent compute go mostly to re-tuning known-reliable knowledge? Returns the trailing exploit share by minutes and
    by count and a violation flag. This is the section-35 tripwire ('do not spend all compute tuning known parameters')."""
    rows = sorted(history, key=lambda g: g.when)[-window:]
    if not rows:
        return {"n": 0, "violation": False, "exploit_share_minutes": 0.0, "exploit_share_count": 0.0}
    mins = sum(g.cost_minutes for g in rows)
    ex_m = sum(g.cost_minutes for g in rows if g.target in EXPLOIT_TARGETS)
    ex_c = sum(1 for g in rows if g.target in EXPLOIT_TARGETS)
    share_m = ex_m / mins if mins > 0 else ex_c / len(rows)
    return {"n": len(rows), "exploit_share_minutes": share_m, "exploit_share_count": ex_c / len(rows),
            "violation": share_m > cap + 0.10, "cap": cap}


# ------------------------------------------------------------------------------------------------ compute-aware selection

@dataclass(frozen=True)
class Selection:
    selected: tuple
    deferred: tuple                                 # ((ScoredCandidate, reason), ...)
    used_minutes: float
    by_target: Mapping

    def ids(self) -> tuple:
        return tuple(s.candidate.cid for s in self.selected)


def knapsack(items: Sequence[tuple], capacity: int) -> tuple:
    """0/1 knapsack, exact. items = [(value, integer_weight)]; returns (best value, chosen indices). O(n x capacity)."""
    n = len(items)
    cap = max(0, int(capacity))
    dp = [[0.0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v, w = items[i - 1]
        for c in range(cap + 1):
            best = dp[i - 1][c]
            if w <= c and dp[i - 1][c - w] + v > best:
                best = dp[i - 1][c - w] + v
            dp[i][c] = best
    chosen, c = [], cap
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= items[i - 1][1]
    return dp[n][cap], sorted(chosen)


def select_within_budget(scored: Sequence[ScoredCandidate], budget: ComputeBudget, allocation: Allocation | None,
                         rounds: int = 1, max_dp_cells: int = 2_000_000) -> Selection:
    """Choose experiments that maximise total priority within the round. Two passes: (1) each target's quota in minutes
    (exact knapsack, greedy by priority density when the table would be too large), (2) leftover minutes go to the best
    remaining candidates in ANY target except that KNOWN_RELIABLE stays under its cap. Real-data jobs are limited to
    `real_data_slots x rounds`; blocked candidates are deferred with their reason, never dropped silently."""
    deferred = [(s, s.blocked) for s in scored if s.blocked or s.priority <= 0]
    live = [s for s in scored if not s.blocked and s.priority > 0]
    chosen: list = []
    used = 0.0
    real_left = budget.real_data_slots * max(1, rounds)

    def take(s):
        nonlocal used, real_left
        chosen.append(s)
        used += s.candidate.cost.cpu_minutes
        if s.candidate.cost.real_data:
            real_left -= 1

    def pick(pool, minutes):
        nonlocal real_left
        pool = [s for s in pool if s not in chosen]
        if not pool or minutes <= 0:
            return
        weights = [max(1, int(math.ceil(s.candidate.cost.cpu_minutes))) for s in pool]
        if len(pool) * (int(minutes) + 1) <= max_dp_cells:
            _, idx = knapsack([(s.priority, w) for s, w in zip(pool, weights)], int(minutes))
            order = sorted(idx, key=lambda i: -pool[i].priority)
        else:
            order = sorted(range(len(pool)), key=lambda i: -pool[i].priority / weights[i])
            acc, keep = 0, []
            for i in order:
                if acc + weights[i] <= minutes:
                    keep.append(i)
                    acc += weights[i]
            order = keep
        for i in order:
            s = pool[i]
            if s.candidate.cost.real_data and real_left <= 0:
                deferred.append((s, "real-data slot limit reached this round"))
                continue
            take(s)

    if allocation is not None:
        for t in TARGETS:
            pick([s for s in live if s.candidate.target == t], allocation.minutes.get(t.value, 0.0))
    remaining = budget.cpu_minutes - used
    exploit_cap_min = (allocation.exploit_share if allocation else 1.0) * budget.cpu_minutes
    rest = [s for s in live if s not in chosen and s.candidate.cost.cpu_minutes <= remaining + EPS]
    for s in sorted(rest, key=lambda s: -s.priority / max(s.candidate.cost.cpu_minutes, 1.0)):
        if s.candidate.cost.cpu_minutes > budget.cpu_minutes - used + EPS:
            continue
        if s.candidate.target in EXPLOIT_TARGETS:
            spent = sum(c.candidate.cost.cpu_minutes for c in chosen if c.candidate.target in EXPLOIT_TARGETS)
            if allocation is not None and spent + s.candidate.cost.cpu_minutes > exploit_cap_min + EPS:
                deferred.append((s, "known-reliable exploit cap reached"))
                continue
        if s.candidate.cost.real_data and real_left <= 0:
            deferred.append((s, "real-data slot limit reached this round"))
            continue
        take(s)
    seen = {s.candidate.cid for s in chosen}
    for s in live:
        if s.candidate.cid not in seen and all(s.candidate.cid != d.candidate.cid for d, _ in deferred):
            deferred.append((s, "did not fit the round's minutes after higher-priority work"))
    by_t: dict = {}
    for s in chosen:
        by_t[s.candidate.target.value] = by_t.get(s.candidate.target.value, 0.0) + s.candidate.cost.cpu_minutes
    chosen.sort(key=lambda s: -s.priority)
    return Selection(tuple(chosen), tuple(deferred), used, by_t)


# ------------------------------------------------------------------------------------------------ diminishing returns

def marginal_return_verdict(gains: Sequence[float], k: int = 4, eps: float = 0.03) -> dict:
    """Should this line of experiments stop? STOP when the last k gains are all below eps AND the fitted trend is not rising.
    INSUFFICIENT under k observations. The policy uses this so tuning one family cannot go on for ever."""
    g = [float(x) for x in gains]
    if len(g) < k:
        return {"verdict": "INSUFFICIENT", "n": len(g)}
    tail = g[-k:]
    x = np.arange(len(g), dtype=float)
    slope = float(np.polyfit(x, g, 1)[0]) if len(g) >= 3 else 0.0
    stop = all(v < eps for v in tail) and slope <= 1e-4
    return {"verdict": "STOP" if stop else "CONTINUE", "n": len(g), "tail_mean": sum(tail) / k, "slope": slope}


# ------------------------------------------------------------------------------------------------ the policy object

@dataclass(frozen=True)
class ResearchPlan:
    now: str
    scored: tuple
    allocation: Allocation
    selection: Selection
    audit: Mapping
    label: str = LABEL

    def explain(self, limit: int = 10) -> str:
        lines = [f"Research plan at {self.now}   [{self.label}]",
                 "allocation: " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(self.allocation.shares.items(), key=lambda kv: -kv[1]) if v > 0),
                 f"explore {self.allocation.explore_share:.0%} / exploit {self.allocation.exploit_share:.0%}"]
        lines += [f"  note: {w}" for w in self.allocation.rationale[:5]]
        lines.append(f"selected {len(self.selection.selected)} uses {self.selection.used_minutes:.0f} min:")
        lines += ["  " + s.explain() for s in self.selection.selected[:limit]]
        lines.append(f"deferred {len(self.selection.deferred)}:")
        lines += [f"  {s.candidate.cid}: {why}" for s, why in self.selection.deferred[:limit]]
        if self.audit.get("violation"):
            lines.append(f"AUDIT: recent spend {self.audit['exploit_share_minutes']:.0%} on known-reliable tuning (cap {self.audit['cap']:.0%})")
        return "\n".join(lines)


class ResearchPolicy:
    """Stateful facade: scores candidates, splits the round across targets, selects within compute, and learns from results."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self.calibrator = EigCalibrator(self.cfg.calibration_prior_strength)
        self.priority = PriorityFunction(self.cfg, self.calibrator)
        self.allocator = TargetAllocator(self.cfg)
        self.history: list = []

    def observe(self, gain: RealisedGain, now=None) -> None:
        """Learn from an executed experiment. `now`, when given, must be after the gain (fail closed on the future)."""
        if now is not None and to_ts(gain.when) > to_ts(now):
            raise FirewallBreach(f"gain of {gain.cid} dated {gain.when} is after now={now}")
        self.history.append(gain)
        self.calibrator.observe(gain)
        self.allocator.observe(gain.target, gain.useful, gain.realised_bits, gain.cost_minutes)

    def plan(self, candidates: Sequence[Candidate], ctx: PolicyContext, budget: ComputeBudget, seed: int, rounds: int = 1) -> ResearchPlan:
        ids = [c.cid for c in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate ids in one planning round")
        for e in budget.check():
            raise ValueError(e)
        for g in self.history:
            if to_ts(g.when) >= to_ts(ctx.now):
                raise FirewallBreach(f"history entry {g.cid} dated {g.when} is not before now={ctx.now}")
        ctx = replace(ctx, history=tuple(self.history) if not ctx.history else ctx.history)
        scored = tuple(self.priority.score(c, ctx, budget) for c in candidates)
        pressure = dict(ctx.pressure)
        if not pressure:
            for s in scored:
                if s.priority > 0:
                    k = s.candidate.target.value
                    pressure[k] = pressure.get(k, 0.0) + s.candidate.factors.uncertainty
        alloc = self.allocator.allocate(budget.cpu_minutes, pressure, seed, ctx.meta)
        sel = select_within_budget(scored, budget, alloc, rounds)
        audit = audit_spend(self.history, self.cfg.spend_window, self.cfg.exploit_cap)
        return ResearchPlan(str(ctx.now), tuple(sorted(scored, key=lambda s: -s.priority)), alloc, sel, audit)

    def line_verdict(self, family: str) -> dict:
        return marginal_return_verdict([g.realised_bits for g in sorted(self.history, key=lambda g: g.when) if g.family == family])

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for g in self.history:
                f.write(json.dumps({**g.__dict__, "target": g.target.value}) + "\n")

    @classmethod
    def load(cls, path, cfg: PolicyConfig | None = None) -> "ResearchPolicy":
        pol = cls(cfg)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    pol.observe(RealisedGain(**{**d, "target": ResearchTarget.parse(d["target"])}))
        return pol


# ------------------------------------------------------------------------------------------------ planted-world simulator

def simulate_allocation(true_yield: Mapping[str, float], rounds: int, budget_minutes: float, seed: int,
                        cfg: PolicyConfig | None = None, policy: str = "bandit", gain_if_useful: float = 0.4) -> dict:
    """Run the allocator against a world where each target has a hidden probability that an experiment there is useful.
    policy: 'bandit' (TargetAllocator), 'exploit_only' (all effort on KNOWN_RELIABLE), 'uniform'. Reports cumulative useful
    bits and regret against an oracle that always funds the best target. A planted world so the allocator can be CAUGHT
    being worse than uniform or than the oracle; deterministic in `seed`."""
    cfg = cfg or PolicyConfig()
    rng = np.random.default_rng(seed)
    alloc = TargetAllocator(cfg)
    pressure = {t.value: 1.0 for t in TARGETS}
    per_exp_minutes = 10.0
    total = 0.0
    by_t = {t.value: 0 for t in TARGETS}
    best = max(true_yield.get(t.value, 0.0) for t in TARGETS)
    oracle = 0.0
    for r in range(rounds):
        n_exp = max(1, int(budget_minutes / per_exp_minutes))
        if policy == "bandit":
            a = alloc.allocate(budget_minutes, pressure, seed * 100003 + r)
            weights = np.array([a.shares[t.value] for t in TARGETS])
        elif policy == "exploit_only":
            weights = np.array([1.0 if t in EXPLOIT_TARGETS else 0.0 for t in TARGETS])
        else:
            weights = np.ones(len(TARGETS))
        weights = weights / weights.sum()
        picks = rng.choice(len(TARGETS), size=n_exp, p=weights)
        for i in picks:
            t = TARGETS[i]
            useful = bool(rng.random() < true_yield.get(t.value, 0.0))
            gain = gain_if_useful if useful else 0.0
            total += gain
            by_t[t.value] += 1
            alloc.observe(t, useful, gain, per_exp_minutes)
        oracle += n_exp * best * gain_if_useful
    return {"policy": policy, "total_bits": total, "oracle_bits": oracle, "regret": oracle - total,
            "efficiency": total / oracle if oracle > 0 else 0.0, "experiments_by_target": by_t}


def report(policy: ResearchPolicy) -> str:
    """Human-readable state of what the policy has learned."""
    lines = [f"Research policy   [{LABEL}]", f"history: {len(policy.history)} executed experiments"]
    lines.append("calibration (realised vs predicted bits):")
    for t, r in sorted(policy.calibrator.report().items()):
        lines.append(f"  {t}: n={r['n']} scale={r['scale']:.2f} {r['verdict']}")
    lines.append("targets:")
    for t, r in policy.allocator.report().items():
        if r["trials"]:
            lines.append(f"  {t}: {r['useful']}/{r['trials']} useful, {r['gain_bits']:.2f} bits, {r['minutes']:.0f} min")
    lines.append("stable id of config: " + stable_hash(policy.cfg))
    return "\n".join(lines)


# ==================================================================================================================
# Part 2: decision value, estimators, batch diversification, robustness, off-policy evaluation, non-stationarity
# ==================================================================================================================

def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def evsi_normal(prior_mean: float, prior_sd: float, noise_sd: float, n_new: float, threshold: float = 0.0,
                loss_scale: float = 1.0) -> float:
    """Expected value of sample information for an adopt/reject decision on an effect with a Normal(prior_mean, prior_sd)
    belief, adopted when its posterior mean exceeds `threshold`. `n_new` observations with per-observation noise `noise_sd`.
    Closed form via the pre-posterior: the posterior mean after sampling is Normal(prior_mean, s^2) with
    s^2 = prior_sd^2 - post_var, and EVSI = loss_scale * s * (phi(d/s) - |d|/s * (1 - Phi(|d|/s))) where d = prior_mean - threshold.
    This is the DECISION value of an experiment (the 'expected_decision_value' factor), as opposed to its information value."""
    if prior_sd <= 0 or n_new <= 0 or noise_sd <= 0:
        return 0.0
    post_var = 1.0 / (1.0 / prior_sd ** 2 + n_new / noise_sd ** 2)
    s2 = prior_sd ** 2 - post_var
    if s2 <= 0:
        return 0.0
    s = math.sqrt(s2)
    d = abs(prior_mean - threshold)
    u = d / s
    return float(loss_scale * s * (_norm_pdf(u) - u * (1.0 - _norm_cdf(u))))


def decision_value_factor(evsi: float, stake: float, saturation: float = 0.02) -> float:
    """Map an EVSI (in the units of the stake, e.g. weekly return) onto [0, 1] for the priority function."""
    return float(1.0 - math.exp(-max(evsi, 0.0) * max(stake, 0.0) / max(saturation, EPS)))


def uncertainty_from_posterior(mean: float, sd: float, scale: float) -> float:
    """Uncertainty factor in [0,1] from a belief's spread relative to the size of effect that matters (`scale`):
    sd >> scale means we cannot tell whether the effect matters (1); sd << scale means we can (0)."""
    if scale <= 0:
        return 0.0
    r = sd / scale
    return float(1.0 - math.exp(-r * r))


def uncertainty_from_counts(successes: int, trials: int, floor_n: int = 5) -> float:
    """Uncertainty of a rate as the width of its Wilson interval, rescaled to [0,1] (width 1 = complete ignorance)."""
    if trials <= 0:
        return 1.0
    z = 1.96
    p = successes / trials
    d = 1 + z * z / trials
    c = p + z * z / (2 * trials)
    r = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    width = min(1.0, (min(1.0, (c + r) / d) - max(0.0, (c - r) / d)))
    return float(width if trials >= floor_n else max(width, 0.75))


def transfer_potential_from_breadth(n_contexts: int, n_years: int, n_sectors: int = 0) -> float:
    """More independent contexts, years and sectors under a finding -> more chance it generalises. Saturating."""
    breadth = 0.5 * n_contexts + 0.35 * n_years + 0.15 * n_sectors
    return float(1.0 - math.exp(-breadth / 3.0))


def feasibility_from_coverage(coverage: Mapping, needs: Sequence[str], min_coverage: float = 0.6) -> tuple:
    """(feasibility in [0,1], reason). `coverage[dataset]` is the share of the needed history that exists. A dataset below
    `min_coverage` makes the experiment infeasible (0) rather than merely worse: partial data silently biases the answer."""
    worst = 1.0
    why = []
    for d in needs:
        c = float(coverage.get(d, 0.0))
        worst = min(worst, c)
        if c < min_coverage:
            why.append(f"{d} coverage {c:.0%} < {min_coverage:.0%}")
    return (0.0 if why else worst), "; ".join(why)


# ------------------------------------------------------------------------------------------------ batch diversification

def joint_information(prior: Mapping[str, float], likelihoods: Sequence[Mapping[str, Mapping[str, float]]]) -> float:
    """Mutual information between the hypothesis and the JOINT outcome of several conditionally independent experiments
    over the same hypothesis set. Enumerates the joint outcome space, so keep experiments x outcomes small (guarded)."""
    if not likelihoods:
        return 0.0
    space = 1
    for lk in likelihoods:
        space *= max(1, len({o for h in lk.values() for o in h}))
    if space > 200_000:
        raise ValueError(f"joint outcome space {space} too large to enumerate")
    hs = list(prior)
    outs = [sorted({o for h in lk.values() for o in h}) for lk in likelihoods]
    grid: list = [((), {h: prior[h] for h in hs})]
    for lk, os_ in zip(likelihoods, outs):
        nxt = []
        for combo, weights in grid:
            for o in os_:
                w = {h: weights[h] * lk.get(h, {}).get(o, 0.0) for h in hs}
                nxt.append((combo + (o,), w))
        grid = nxt
    h_prior = entropy_bits(prior.values())
    exp_post = 0.0
    for _, w in grid:
        tot = sum(w.values())
        if tot <= 0:
            continue
        exp_post += tot * entropy_bits([v / tot for v in w.values()])
    return max(0.0, h_prior - exp_post)


def candidate_likelihood(c: Candidate) -> tuple:
    """(prior, likelihood) of a discrete-model candidate, or None if it has no discrete model."""
    if c.info.kind != "discrete":
        return None
    prior = {h: float(p) for h, p in c.info.params["hypotheses"]}
    like: dict = {h: {} for h in prior}
    for h, o, p in c.info.params["expected"]:
        like[h][o] = like[h].get(o, 0.0) + float(p)
    return prior, like


def greedy_diverse_batch(scored: Sequence[ScoredCandidate], k: int, min_marginal_bits: float = 0.02) -> list:
    """Pick up to k candidates by MARGINAL information: two experiments that discriminate the same hypotheses in the same way
    add little to one another, so the second is discounted by the joint information actually gained. Candidates with
    different hypothesis sets are independent and keep their own bits. Greedy on priority x (marginal / standalone bits)."""
    live = [s for s in scored if not s.blocked and s.priority > 0]
    chosen: list = []
    by_hyp: dict = {}
    while live and len(chosen) < k:
        best, best_val, best_ratio = None, -1.0, 1.0
        for s in live:
            cl = candidate_likelihood(s.candidate)
            ratio = 1.0
            if cl is not None:
                key = tuple(sorted(cl[0]))
                group = by_hyp.get(key, [])
                if group:
                    alone = mutual_information(*cl)
                    joint_old = joint_information(cl[0], [g[1] for g in group])
                    joint_new = joint_information(cl[0], [g[1] for g in group] + [cl[1]])
                    marginal = joint_new - joint_old
                    ratio = marginal / alone if alone > EPS else 0.0
                    if marginal < min_marginal_bits:
                        ratio = 0.0
            val = s.priority * ratio
            if val > best_val:
                best, best_val, best_ratio = s, val, ratio
        if best is None or best_val <= 0:
            break
        chosen.append((best, best_ratio))
        cl = candidate_likelihood(best.candidate)
        if cl is not None:
            by_hyp.setdefault(tuple(sorted(cl[0])), []).append(cl)
        live = [s for s in live if s is not best]
    return chosen


# ------------------------------------------------------------------------------------------------ robustness of the ranking

def _kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            conc += s > 0
            disc += s < 0
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def rank_stability(policy: "ResearchPolicy", candidates: Sequence[Candidate], ctx: PolicyContext, seed: int, n_draws: int = 200,
                   noise: float = 0.25, top_k: int = 3, budget: ComputeBudget | None = None) -> dict:
    """How much does the ranking depend on the exact factor values? Every factor is jittered by lognormal noise (`noise` =
    sigma of the log), the candidates re-scored, and we report each candidate's probability of landing in the top_k plus
    the mean Kendall tau against the unperturbed ranking. A ranking that flips under 25 percent noise is not a decision."""
    rng = np.random.default_rng(seed)
    base = [policy.priority.score(c, ctx, budget).priority for c in candidates]
    if not candidates:
        return {"n": 0, "tau_mean": 1.0, "top_k_prob": {}}
    hits = {c.cid: 0 for c in candidates}
    taus = []
    for _ in range(n_draws):
        jitter = np.exp(rng.normal(0.0, noise, size=(len(candidates), 5)))
        sc = []
        for c, j in zip(candidates, jitter):
            f = c.factors
            fac = Factors(min(1.0, f.uncertainty * j[0]), min(1.0, f.relevance * j[1]), min(1.0, f.transfer_potential * j[2]),
                          min(1.0, f.feasibility * j[3]), min(1.0, f.decision_value * j[4]))
            sc.append(policy.priority.score(replace(c, factors=fac), ctx, budget).priority)
        order = np.argsort(-np.array(sc))[:top_k]
        for i in order:
            hits[candidates[i].cid] += 1
        taus.append(_kendall_tau(base, sc))
    return {"n": len(candidates), "draws": n_draws, "tau_mean": float(np.mean(taus)), "tau_min": float(np.min(taus)),
            "top_k_prob": {cid: h / n_draws for cid, h in hits.items()}, "noise": noise,
            "stable": bool(np.mean(taus) > 0.8)}


def factor_elasticity(policy: "ResearchPolicy", c: Candidate, ctx: PolicyContext, delta: float = 0.10,
                      budget: ComputeBudget | None = None) -> dict:
    """d log(priority) / d log(factor) by finite differences for each of the five multiplicative factors: which lever
    moves this candidate's priority most. Factors at their floor or ceiling report None (no room to move)."""
    base = policy.priority.score(c, ctx, budget).priority
    out = {}
    for name in ("uncertainty", "relevance", "transfer_potential", "feasibility", "decision_value"):
        v = getattr(c.factors, name)
        up = min(1.0, v * (1 + delta))
        if base <= 0 or up == v or v <= 0:
            out[name] = None
            continue
        p2 = policy.priority.score(replace(c, factors=replace(c.factors, **{name: up})), ctx, budget).priority
        out[name] = math.log(p2 / base) / math.log(up / v) if p2 > 0 else None
    return out


def why_ranked_above(a: ScoredCandidate, b: ScoredCandidate) -> str:
    """One-line reason a outranks b: the log-term or penalty with the biggest gap."""
    if a.blocked or b.blocked:
        return f"{'a' if b.blocked else 'b'} is blocked: {a.blocked or b.blocked}"
    gaps = {k: a.log_terms[k] - b.log_terms[k] for k in a.log_terms}
    for k in a.penalties:
        pa, pb = max(a.penalties[k], EPS), max(b.penalties[k], EPS)
        gaps["penalty:" + k] = math.log(pa) - math.log(pb)
    k = max(gaps, key=lambda x: abs(gaps[x]))
    return f"{a.candidate.cid} vs {b.candidate.cid}: largest difference is {k} ({gaps[k]:+.2f} in log priority)"


# ------------------------------------------------------------------------------------------------ off-policy evaluation

@dataclass(frozen=True)
class LoggedChoice:
    """One logged allocation decision: which target was funded, with what probability, and what it returned."""
    target: ResearchTarget
    propensity: float                              # probability the LOGGING policy funded this target
    reward: float                                  # realised bits per experiment (or per minute), whichever the caller uses


def ips_estimate(log: Sequence[LoggedChoice], new_shares: Mapping[str, float], clip: float = 10.0) -> dict:
    """Estimate the average reward a NEW allocation would have earned from logs of the old one (inverse propensity scoring,
    weights clipped at `clip`; self-normalised as well). Refuses zero propensities: a logging policy that never tried a
    target tells us nothing about it, and pretending otherwise is how off-policy estimates lie."""
    if not log:
        return {"n": 0, "ips": float("nan"), "snips": float("nan"), "ess": 0.0}
    w, r = [], []
    for c in log:
        if c.propensity <= 0:
            raise ValueError("zero-propensity choice in the log: off-policy estimate undefined")
        w.append(min(clip, new_shares.get(c.target.value, 0.0) / c.propensity))
        r.append(c.reward)
    w, r = np.array(w), np.array(r)
    ess = float(w.sum() ** 2 / max((w ** 2).sum(), EPS))
    return {"n": len(log), "ips": float((w * r).mean()), "snips": float((w * r).sum() / max(w.sum(), EPS)), "ess": ess,
            "clipped_share": float(np.mean(np.array([new_shares.get(c.target.value, 0.0) / c.propensity for c in log]) > clip))}


def compare_allocations(log: Sequence[LoggedChoice], a: Mapping[str, float], b: Mapping[str, float], seed: int, n_boot: int = 1000) -> dict:
    """Paired bootstrap of SNIPS(a) - SNIPS(b) over the logged choices."""
    if len(log) < 10:
        return {"verdict": "INSUFFICIENT", "n": len(log)}
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(log)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [log[i] for i in idx]
        diffs.append(ips_estimate(sample, a)["snips"] - ips_estimate(sample, b)["snips"])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    est = ips_estimate(log, a)["snips"] - ips_estimate(log, b)["snips"]
    return {"verdict": "A_BETTER" if lo > 0 else "B_BETTER" if hi < 0 else "NO_DIFFERENCE", "diff": float(est), "lo": float(lo),
            "hi": float(hi), "n": n}


# ------------------------------------------------------------------------------------------------ non-stationarity

class DiscountedAllocator(TargetAllocator):
    """Target allocator for a world that changes: every observation ages by `gamma` per new round, so old results count less
    (effective memory ~ 1/(1-gamma) rounds). With gamma = 1 it is the plain allocator."""

    def __init__(self, cfg: PolicyConfig | None = None, gamma: float = 0.97):
        super().__init__(cfg)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma
        self.f_trials = {t: 0.0 for t in TARGETS}
        self.f_useful = {t: 0.0 for t in TARGETS}
        self.f_gain = {t: 0.0 for t in TARGETS}

    def age(self) -> None:
        for t in TARGETS:
            self.f_trials[t] *= self.gamma
            self.f_useful[t] *= self.gamma
            self.f_gain[t] *= self.gamma

    def observe(self, target, useful: bool, gain_bits: float, minutes: float) -> None:
        super().observe(target, useful, gain_bits, minutes)
        t = ResearchTarget.parse(target)
        self.f_trials[t] += 1.0
        self.f_useful[t] += float(bool(useful))
        self.f_gain[t] += max(gain_bits, 0.0)

    def yield_draw(self, t, rng, meta) -> float:
        p = rng.beta(1.0 + self.f_useful[t], 1.0 + max(self.f_trials[t] - self.f_useful[t], 0.0))
        return float(p * (self.f_gain[t] + 0.3) / (self.f_useful[t] + 2.0))

    def ucb_bonus(self, t) -> float:
        n = sum(self.f_trials.values())
        return math.sqrt(2.0 * math.log(n + 2.0) / (self.f_trials[t] + 1.0))


def cusum_shift(series: Sequence[float], target_mean: float | None = None, k: float = 0.5, h: float = 4.0) -> dict:
    """Two-sided CUSUM on a standardised series: detects a shift in a target's yield. Returns the first alarm index (or None)
    and the direction. `k` is the allowance in sd units, `h` the decision threshold. Needs >= 8 points for a scale estimate."""
    x = np.asarray(series, float)
    if len(x) < 8:
        return {"alarm": None, "direction": None, "n": len(x), "verdict": "INSUFFICIENT"}
    base = x[: max(4, len(x) // 3)]
    mu = float(base.mean()) if target_mean is None else float(target_mean)
    sd = float(base.std(ddof=1)) or 1e-9
    hi = lo = 0.0
    for i, v in enumerate(x):
        z = (v - mu) / sd
        hi = max(0.0, hi + z - k)
        lo = max(0.0, lo - z - k)
        if hi > h:
            return {"alarm": i, "direction": "up", "n": len(x), "verdict": "SHIFT"}
        if lo > h:
            return {"alarm": i, "direction": "down", "n": len(x), "verdict": "SHIFT"}
    return {"alarm": None, "direction": None, "n": len(x), "verdict": "STABLE"}


# ------------------------------------------------------------------------------------------------ multi-round budget planning

@dataclass(frozen=True)
class HorizonPlan:
    rounds: tuple                                   # per round: (minutes, explore_reserve_minutes)
    total_minutes: float
    explore_reserve_total: float
    carryover: float


def plan_horizon(total_minutes: float, n_rounds: int, n_done: int = 0, front_load: float = 0.0, explore_floor: float = 0.10) -> HorizonPlan:
    """Split a multi-round compute budget. Each round reserves `epsilon_schedule` minutes for exploration that cannot be spent
    on exploitation even if exploitation looks better; `front_load` in [0,1) shifts minutes to early rounds (early results
    change what later rounds should be, so learning early is worth more). Weights are geometric and sum exactly to the total."""
    if n_rounds < 1 or total_minutes < 0:
        raise ValueError("need at least one round and a non-negative budget")
    if not 0.0 <= front_load < 1.0:
        raise ValueError("front_load must be in [0, 1)")
    ratio = 1.0 - front_load * 0.5
    w = np.array([ratio ** i for i in range(n_rounds)])
    w = w / w.sum()
    mins = w * total_minutes
    eps = [epsilon_schedule(n_done + i, floor=explore_floor) for i in range(n_rounds)]
    rounds = tuple((float(m), float(m * e)) for m, e in zip(mins, eps))
    return HorizonPlan(rounds, float(total_minutes), float(sum(r[1] for r in rounds)), 0.0)


# ------------------------------------------------------------------------------------------------ seeding from the real log

_TARGET_KEYWORDS = (
    (ResearchTarget.DATA_QUALITY, ("data", "quality", "audit", "leak", "provenance", "snapshot")),
    (ResearchTarget.CONTRADICTION, ("contradict", "conflict", "disagree")),
    (ResearchTarget.FAILURE, ("fail", "loss", "postmortem", "stop", "drawdown", "gap")),
    (ResearchTarget.MISSED_WINNER, ("missed", "winner", "capture")),
    (ResearchTarget.REGIME_TRANSITION, ("regime", "transition", "era", "bear", "bull")),
    (ResearchTarget.INTERACTION, ("interact", "combination", "pair", "chain")),
    (ResearchTarget.NEW_REPRESENTATION, ("representation", "basis", "feature", "embedding")),
    (ResearchTarget.WEAK_PATTERN, ("weak", "decay", "dormant", "degrad")),
    (ResearchTarget.KNOWN_RELIABLE, ("tune", "tuning", "grid", "sweep", "frontier", "threshold", "param")))


def infer_target(row: Mapping) -> ResearchTarget:
    """Best-effort research target for a legacy experiment row from its event / question / config text. UNKNOWN_AREA when
    no keyword matches. The mapping is keyword-only and says nothing the row does not; treat seeded yields as weak priors."""
    text = " ".join(str(row.get(k, "")) for k in ("event", "question", "what", "hypothesis", "name", "experiment_id")).lower()
    for target, words in _TARGET_KEYWORDS:
        if any(w in text for w in words):
            return target
    return ResearchTarget.UNKNOWN_AREA


def seed_from_log(policy: "ResearchPolicy", rows: Iterable[Mapping], now) -> dict:
    """Warm-start the policy from state/experiments.jsonl-shaped rows: an 'adopt' counts as useful, a 'reject' as not useful,
    'continue_testing' is skipped. Only rows strictly before `now` are used. Returns counts per target; weak evidence by design
    (realised bits are set to 0.15 for adopt and 0.02 for reject: a prior, not a measurement)."""
    cut = to_ts(now)
    counts: dict = {}
    for i, row in enumerate(rows):
        oc = str(row.get("outcome") or "").lower()
        if oc not in ("adopt", "reject"):
            continue
        stamp = row.get("timestamp") or row.get("t")
        if not stamp:
            continue
        try:
            if to_ts(stamp) >= cut:
                continue
        except Exception:
            continue
        tgt = infer_target(row)
        gain = RealisedGain(str(row.get("experiment_id") or f"log-{i}"), tgt, str(row.get("family") or ""), 0.1,
                            0.15 if oc == "adopt" else 0.02, oc == "adopt", float(row.get("cost_minutes") or 10.0), str(stamp))
        policy.observe(gain)
        counts[tgt.value] = counts.get(tgt.value, 0) + 1
    return counts


# ------------------------------------------------------------------------------------------------ invariants

def audit_plan(plan: ResearchPlan, cfg: PolicyConfig, budget: ComputeBudget) -> list:
    """Invariants every plan must satisfy; returns violations (empty = fine). This is what a plan test asserts."""
    v = []
    v += plan.allocation.check()
    for t, s in plan.allocation.shares.items():
        if t in plan.allocation.open_targets and s < cfg.explore_floor - 1e-9:
            v.append(f"open target {t} below floor: {s:.4f}")
    if plan.allocation.exploit_share > cfg.exploit_cap + 1e-9 and len(plan.allocation.open_targets) > 1:
        v.append(f"known-reliable share {plan.allocation.exploit_share:.3f} above cap {cfg.exploit_cap}")
    if plan.selection.used_minutes > budget.cpu_minutes + 1e-6:
        v.append(f"selection uses {plan.selection.used_minutes:.1f} > budget {budget.cpu_minutes:.1f}")
    real = sum(1 for s in plan.selection.selected if s.candidate.cost.real_data)
    if real > budget.real_data_slots:
        v.append(f"{real} real-data jobs selected, slots {budget.real_data_slots}")
    for s in plan.selection.selected:
        if s.blocked or s.priority <= 0:
            v.append(f"blocked candidate {s.candidate.cid} was selected")
        ok, why = budget.admits(s.candidate.cost)
        if not ok:
            v.append(f"{s.candidate.cid} selected but does not fit: {why}")
    return v


# ==================================================================================================================
# Part 3: sequential stopping, cost learning, dependencies, family caps, frontier, drift and a built-in self-check
# ==================================================================================================================

@dataclass(frozen=True)
class SprtState:
    llr: float
    n: int
    decision: str                                   # CONTINUE | ACCEPT_H1 | ACCEPT_H0
    lower: float
    upper: float


def sprt_bernoulli(outcomes: Sequence[int], p0: float, p1: float, alpha: float = 0.05, beta: float = 0.20) -> SprtState:
    """Wald's sequential probability ratio test on a stream of 0/1 outcomes: H0 rate p0 versus H1 rate p1 (p1 > p0). Stops an
    experiment the moment the evidence is decisive, which is how the policy saves compute: an experiment that has already
    answered its question must not keep running to its planned length. Bounds A = ln((1-beta)/alpha), B = ln(beta/(1-alpha))."""
    if not (0 < p0 < p1 < 1):
        raise ValueError("need 0 < p0 < p1 < 1")
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    llr = 0.0
    for i, x in enumerate(outcomes, 1):
        llr += math.log(p1 / p0) if x else math.log((1 - p1) / (1 - p0))
        if llr >= upper:
            return SprtState(llr, i, "ACCEPT_H1", lower, upper)
        if llr <= lower:
            return SprtState(llr, i, "ACCEPT_H0", lower, upper)
    return SprtState(llr, len(outcomes), "CONTINUE", lower, upper)


def expected_sample_saving(p0: float, p1: float, fixed_n: int, alpha: float = 0.05, beta: float = 0.20) -> dict:
    """Wald's approximate expected sample sizes under H0 and H1 versus a fixed-n design: what stopping early is worth."""
    a = math.log((1 - beta) / alpha)
    b = math.log(beta / (1 - alpha))
    d1 = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
    d0 = p0 * math.log(p1 / p0) + (1 - p0) * math.log((1 - p1) / (1 - p0))
    en1 = ((1 - beta) * a + beta * b) / d1
    en0 = ((alpha) * a + (1 - alpha) * b) / d0
    return {"expected_n_h1": en1, "expected_n_h0": en0, "fixed_n": fixed_n,
            "saving_h1": 1.0 - en1 / fixed_n, "saving_h0": 1.0 - en0 / fixed_n}


class CostModel:
    """Learns how wrong the planned compute costs are. Per template: log-ratio actual/planned is tracked with a ridge toward
    zero, so a template that keeps costing twice its estimate is planned at twice the estimate next time. Compute-aware
    allocation is only as good as its cost numbers."""

    def __init__(self, prior_strength: float = 3.0):
        self.k = prior_strength
        self.sum_log: dict = {}
        self.sum_sq: dict = {}
        self.n: dict = {}

    def observe(self, template: str, planned_minutes: float, actual_minutes: float) -> None:
        if planned_minutes <= 0 or actual_minutes <= 0:
            raise ValueError("planned and actual minutes must be positive")
        lr = math.log(actual_minutes / planned_minutes)
        self.sum_log[template] = self.sum_log.get(template, 0.0) + lr
        self.sum_sq[template] = self.sum_sq.get(template, 0.0) + lr * lr
        self.n[template] = self.n.get(template, 0) + 1

    def multiplier(self, template: str) -> float:
        n = self.n.get(template, 0)
        return math.exp(self.sum_log.get(template, 0.0) / (n + self.k))

    def spread(self, template: str) -> float:
        n = self.n.get(template, 0)
        if n < 2:
            return 0.5
        m = self.sum_log[template] / n
        return math.sqrt(max(0.0, (self.sum_sq[template] - n * m * m) / (n - 1)))

    def adjusted(self, template: str, cost: ComputeCost, risk_sd: float = 1.0) -> ComputeCost:
        """Planned cost scaled by the learned multiplier, plus `risk_sd` standard deviations of overrun on the CPU figure
        (a planning budget must cover the plausible bad case, not the mean)."""
        m = self.multiplier(template) * math.exp(risk_sd * self.spread(template) * 0.5)
        return replace(cost, cpu_minutes=cost.cpu_minutes * m, wall_minutes=cost.wall_minutes * m)


def schedule_with_dependencies(scored: Sequence[ScoredCandidate], requires: Mapping[str, Sequence[str]]) -> tuple:
    """Order selected candidates so that prerequisites come first (Kahn's algorithm, ties by priority). Candidates whose
    prerequisite is not in the set are DEFERRED with the missing id, not run blind. A cycle raises: two experiments that
    each need the other's result cannot both be planned."""
    by_id = {s.candidate.cid: s for s in scored}
    deferred, indeg, graph = [], {}, {}
    for cid, s in by_id.items():
        reqs = list(requires.get(cid, ()))
        missing = [r for r in reqs if r not in by_id]
        if missing:
            deferred.append((s, f"prerequisite not in this plan: {missing}"))
            continue
        indeg[cid] = len(reqs)
        for r in reqs:
            graph.setdefault(r, []).append(cid)
    drop = {s.candidate.cid for s, _ in deferred}
    changed = True
    while changed:                                   # a candidate depending on a deferred one is deferred as well
        changed = False
        for cid in list(indeg):
            if cid not in drop and any(r in drop for r in requires.get(cid, ())):
                drop.add(cid)
                deferred.append((by_id[cid], "depends on a deferred candidate"))
                changed = True
    ready = sorted((c for c, d in indeg.items() if d == 0 and c not in drop), key=lambda c: -by_id[c].priority)
    out = []
    indeg = {c: d for c, d in indeg.items() if c not in drop}
    while ready:
        cid = ready.pop(0)
        out.append(by_id[cid])
        for nxt in graph.get(cid, []):
            if nxt in indeg:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        ready.sort(key=lambda c: -by_id[c].priority)
    if len(out) + len(drop) != len(by_id):
        raise ValueError("dependency cycle among candidates: " + ", ".join(sorted(set(by_id) - {s.candidate.cid for s in out} - drop)))
    return tuple(out), tuple(deferred)


def apply_family_caps(scored: Sequence[ScoredCandidate], max_share: float = 0.5, max_items: int | None = None) -> tuple:
    """No single family may take more than `max_share` of the selected compute (monoculture guard: ten variants of one idea
    are one idea). Excess candidates are deferred with the reason. Input order is priority order and is preserved."""
    total = sum(s.candidate.cost.cpu_minutes for s in scored) or 1.0
    used: dict = {}
    count: dict = {}
    keep, deferred = [], []
    for s in scored:
        fam = s.candidate.family or "(none)"
        m = s.candidate.cost.cpu_minutes
        if fam != "(none)" and used.get(fam, 0.0) + m > max_share * total + EPS and used.get(fam, 0.0) > 0:
            deferred.append((s, f"family {fam} already holds {used[fam] / total:.0%} of this round"))
            continue
        if max_items is not None and count.get(fam, 0) >= max_items:
            deferred.append((s, f"family {fam} already has {max_items} items in this round"))
            continue
        used[fam] = used.get(fam, 0.0) + m
        count[fam] = count.get(fam, 0) + 1
        keep.append(s)
    return tuple(keep), tuple(deferred)


def pareto_frontier(scored: Sequence[ScoredCandidate]) -> list:
    """Candidates not dominated on (lower cost, higher priority): the only ones worth choosing between when compute is scarce."""
    live = [s for s in scored if not s.blocked and s.priority > 0]
    live.sort(key=lambda s: (s.candidate.cost.cpu_minutes, -s.priority))
    front, best = [], -1.0
    for s in live:
        if s.priority > best:
            front.append(s)
            best = s.priority
    return front


def allocation_entropy(history: Sequence[RealisedGain], window: int = 30) -> dict:
    """Entropy (bits) of where recent experiments went, against the maximum for the targets ever used. Collapse toward one
    target is a monoculture alarm even when no single choice looked wrong."""
    rows = sorted(history, key=lambda g: g.when)[-window:]
    if not rows:
        return {"n": 0, "entropy_bits": 0.0, "max_bits": 0.0, "collapsed": False}
    counts: dict = {}
    for g in rows:
        counts[g.target.value] = counts.get(g.target.value, 0) + 1
    p = [c / len(rows) for c in counts.values()]
    h = entropy_bits(p)
    ever = len({g.target.value for g in history})
    mx = math.log2(ever) if ever > 1 else 0.0
    return {"n": len(rows), "entropy_bits": h, "max_bits": mx, "targets_used_recently": len(counts), "targets_used_ever": ever,
            "collapsed": bool(mx > 0 and h < 0.4 * mx)}


def plan_rows(plan: ResearchPlan) -> list:
    """Flat dict rows of a plan for CSV / dashboard use: one per candidate with its terms and decision."""
    chosen = set(plan.selection.ids())
    why = {s.candidate.cid: w for s, w in plan.selection.deferred}
    rows = []
    for s in plan.scored:
        rows.append({"cid": s.candidate.cid, "target": s.candidate.target.value, "family": s.candidate.family,
                     "priority": s.priority, "eig_bits": s.eig_bits, "eig_source": s.eig_source, "selected": s.candidate.cid in chosen,
                     "reason_deferred": why.get(s.candidate.cid, ""), "cost_cpu_min": s.candidate.cost.cpu_minutes,
                     **{"pen_" + k: v for k, v in s.penalties.items()}, **{"log_" + k: v for k, v in s.log_terms.items()}})
    return rows


def self_check(seed: int = 0) -> dict:
    """Planted-defect self-check of the policy (C63): builds small worlds where the RIGHT answer is known and confirms the
    policy gives it. Each entry is a check that can fail:
      duplicate_blocked      a candidate the duplicate checker calls blocking gets priority 0 and a reason
      missing_data_blocked   a candidate needing an unavailable dataset is blocked
      exploit_cap_holds      known-reliable never exceeds its cap when other targets are open
      bandit_beats_exploit   the allocator earns more than exploit-only in a world where exploitation is barren
      informative_beats_flat an experiment that discriminates hypotheses outranks one that cannot
      budget_respected       nothing selected exceeds the round's minutes or the RAM ceiling"""
    results = {}
    pol = ResearchPolicy()
    now = "2026-01-10"
    base = Candidate("c_base", "q", ResearchTarget.FAILURE, "2026-01-01", info=InfoModel("normal", {"n_now": 10, "n_new": 30}))
    ctx = PolicyContext(now=now)
    dup = PolicyContext(now=now, duplicate_check=lambda c: type("V", (), {"status": DuplicateStatus.SAME_CONFIG_REPEAT, "blocking": True, "message": "we already tested this"})())
    s = pol.priority.score(base, dup)
    results["duplicate_blocked"] = bool(s.blocked and s.priority == 0.0)
    need = replace(base, cid="c_data", data_needs=("delisted_history",))
    s = pol.priority.score(need, PolicyContext(now=now, available_data=frozenset({"price_panel"})))
    results["missing_data_blocked"] = bool(s.blocked and "delisted_history" in s.blocked)
    alloc = TargetAllocator().allocate(600.0, {t.value: 1.0 for t in TARGETS}, seed)
    results["exploit_cap_holds"] = bool(alloc.exploit_share <= PolicyConfig().exploit_cap + 1e-9)
    yields = {"KNOWN_RELIABLE": 0.03, "FAILURE": 0.5, "UNKNOWN_AREA": 0.2}
    b = simulate_allocation(yields, 30, 100, seed, policy="bandit")
    e = simulate_allocation(yields, 30, 100, seed, policy="exploit_only")
    results["bandit_beats_exploit"] = bool(b["total_bits"] > e["total_bits"])
    flat = replace(base, cid="c_flat", info=InfoModel("discrete", {"hypotheses": [("a", 0.5), ("b", 0.5)],
                                                                 "expected": [("a", "x", 0.5), ("a", "y", 0.5), ("b", "x", 0.5), ("b", "y", 0.5)]}))
    sharp = replace(base, cid="c_sharp", info=InfoModel("discrete", {"hypotheses": [("a", 0.5), ("b", 0.5)],
                                                                   "expected": [("a", "x", 0.95), ("a", "y", 0.05), ("b", "x", 0.05), ("b", "y", 0.95)]}))
    results["informative_beats_flat"] = bool(pol.priority.score(sharp, ctx).priority > pol.priority.score(flat, ctx).priority)
    budget = ComputeBudget(cpu_minutes=60, ram_gb_free=8)
    cands = [replace(base, cid=f"c{i}", cost=ComputeCost(cpu_minutes=25 + 5 * i, ram_gb=1 + i), info=InfoModel("normal", {"n_now": 10, "n_new": 20 + i}))
             for i in range(6)]
    plan = pol.plan(cands, ctx, budget, seed)
    results["budget_respected"] = not audit_plan(plan, pol.cfg, budget)
    results["all_passed"] = all(results.values())
    results["label"] = LABEL
    return results


# ==================================================================================================================
# Part 4: power, design choice, novelty, diminishing information, serialisation, regret and meta-advice
# ==================================================================================================================

def required_n(effect: float, sd: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Observations needed to detect a mean shift `effect` with per-observation noise `sd` (two-sided z-test). An experiment
    planned below this cannot answer its own question; its NULL would mean 'unknown', not 'no'."""
    if effect == 0 or sd <= 0:
        raise ValueError("effect must be non-zero and sd positive")
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(math.ceil((z * sd / abs(effect)) ** 2))


def power_of(n: float, effect: float, sd: float, alpha: float = 0.05) -> float:
    """Power of a two-sided z-test at sample size n against a true shift `effect`."""
    from scipy.stats import norm
    if n <= 0 or sd <= 0:
        return 0.0
    zc = norm.ppf(1 - alpha / 2)
    shift = abs(effect) * math.sqrt(n) / sd
    return float(norm.cdf(shift - zc) + norm.cdf(-shift - zc))


@dataclass(frozen=True)
class DesignOption:
    """One way to run the same experiment: how many observations and what it costs."""
    name: str
    n_new: float
    cost: ComputeCost
    noise_sd: float = 1.0


def best_design(options: Sequence[DesignOption], prior_sd: float, budget: ComputeBudget | None = None, effect_of_interest: float | None = None,
                min_power: float = 0.5, n_now: float = 1.0) -> dict:
    """Choose among designs for ONE question by information per compute minute, subject to fitting the budget and (when
    `effect_of_interest` is given) having at least `min_power` to see it. Returns every option's numbers so the choice is auditable."""
    rows = []
    for o in options:
        ok, why = (True, "") if budget is None else budget.admits(o.cost)
        bits = eig_normal_mean(n_now, o.n_new)
        pw = power_of(o.n_new, effect_of_interest, o.noise_sd) if effect_of_interest else None
        underpowered = pw is not None and pw < min_power
        per_min = bits / max(o.cost.cpu_minutes, EPS)
        rows.append({"name": o.name, "bits": bits, "cost_min": o.cost.cpu_minutes, "bits_per_min": per_min, "power": pw,
                     "fits": ok, "underpowered": underpowered, "note": why or ("underpowered" if underpowered else "")})
    feasible = [r for r in rows if r["fits"] and not r["underpowered"]]
    pick = max(feasible, key=lambda r: r["bits_per_min"]) if feasible else None
    return {"chosen": pick["name"] if pick else None, "options": rows,
            "reason": "no design fits the budget with enough power" if pick is None else f"{pick['name']} has the best bits per minute among feasible designs"}


def novelty_score(config: Mapping, tried: Sequence[Mapping], scale: float = 0.25) -> float:
    """1 for a configuration far from everything tried, ~0 for one right next to a tried config. Uses the same normalised
    distance as the duplicate check (engine.learning.experiment_memory.config_distance), so 'novel' means the same thing in both."""
    from .experiment_memory import config_distance, infer_space
    if not tried:
        return 1.0
    space = infer_space(list(tried) + [config])
    d = min(config_distance(config, t, space) for t in tried)
    return float(1.0 - math.exp(-d / max(scale, EPS)))


def coverage_bonus(candidate: Candidate, tried_configs: Sequence[Mapping], weight: float = 0.5) -> float:
    """Multiplier >= 1 rewarding exploration of untried regions, applied only to exploration targets. Exploitation targets
    get no bonus for novelty (re-tuning a known parameter a little differently is not exploration)."""
    if candidate.target in EXPLOIT_TARGETS:
        return 1.0
    return 1.0 + weight * novelty_score(candidate.config, tried_configs)


def diminishing_information(bits: float, n_answered_same_question: int, rate: float = 0.5) -> float:
    """Each earlier answer to (nearly) the same question halves what another answer can teach: bits x rate^n. The prior over
    hypotheses should already have absorbed those answers; this guards against a prior that has not."""
    return bits * (rate ** max(0, n_answered_same_question))


def regret_report(history: Sequence[RealisedGain], window: int = 50) -> dict:
    """Regret of the allocation actually made against always funding the best-performing target in hindsight (an oracle that
    cannot exist in real time, so this is a diagnostic, not a target). Per target: experiments, mean realised bits."""
    rows = sorted(history, key=lambda g: g.when)[-window:]
    if not rows:
        return {"n": 0, "regret_bits": 0.0}
    by: dict = {}
    for g in rows:
        by.setdefault(g.target.value, []).append(g.realised_bits)
    means = {k: sum(v) / len(v) for k, v in by.items()}
    best = max(means, key=means.get)
    got = sum(g.realised_bits for g in rows)
    return {"n": len(rows), "best_target_in_hindsight": best, "best_mean_bits": means[best], "achieved_bits": got,
            "oracle_bits": means[best] * len(rows), "regret_bits": means[best] * len(rows) - got,
            "mean_by_target": means, "counts": {k: len(v) for k, v in by.items()}}


def allocation_sensitivity(pressure: Mapping, seed: int, budget_minutes: float = 600.0, caps: Sequence[float] = (0.15, 0.30, 0.50),
                           floors: Sequence[float] = (0.0, 0.03, 0.06)) -> list:
    """How the split moves with the two safety knobs (exploit cap, per-target floor) for the same beliefs and the same
    seed: shows the price of each guard in exploitation share and in number of funded targets."""
    out = []
    for cap in caps:
        for fl in floors:
            cfg = PolicyConfig(exploit_cap=cap, explore_floor=fl)
            try:
                a = TargetAllocator(cfg).allocate(budget_minutes, pressure, seed)
            except ValueError as e:
                out.append({"cap": cap, "floor": fl, "error": str(e)})
                continue
            out.append({"cap": cap, "floor": fl, "exploit_share": a.exploit_share, "funded_targets": sum(1 for v in a.shares.values() if v > 1e-9),
                        "max_share": max(a.shares.values())})
    return out


def meta_adjusted_factors(c: Candidate, meta: MetaAdvice) -> Candidate:
    """Fold MetaAdvice into a candidate's own factors (G12): a family with high learned survival gets a transfer-potential
    lift proportional to advice trust; a target with a low learned yield gets its decision value trimmed. The result is a NEW
    candidate; the original is untouched. With empty advice this is the identity."""
    w = meta.trust()
    if w <= 0:
        return c
    f = c.factors
    surv = meta.family_survival.get(c.family)
    y = meta.target_yield.get(c.target.value)
    tp = f.transfer_potential if surv is None else (1 - w) * f.transfer_potential + w * float(surv)
    dv = f.decision_value if y is None else (1 - 0.5 * w) * f.decision_value + 0.5 * w * f.decision_value * (0.5 + float(y))
    return replace(c, factors=replace(f, transfer_potential=min(1.0, max(0.0, tp)), decision_value=min(1.0, max(0.0, dv))))


# ------------------------------------------------------------------------------------------------ serialisation

def candidate_to_dict(c: Candidate) -> dict:
    return {"cid": c.cid, "question": c.question, "target": c.target.value, "created_at": c.created_at,
            "factors": dict(c.factors.__dict__), "info": {"kind": c.info.kind, "params": _plain(c.info.params)},
            "cost": dict(c.cost.__dict__), "subsystem": c.subsystem, "family": c.family, "config": _plain(c.config),
            "data_needs": list(c.data_needs), "overfit_hint": c.overfit_hint, "free_parameters": c.free_parameters,
            "prior_tests_on_window": c.prior_tests_on_window, "evidence": list(c.evidence), "magnitude": c.magnitude}


def candidate_from_dict(d: Mapping) -> Candidate:
    return Candidate(cid=d["cid"], question=d["question"], target=ResearchTarget.parse(d["target"]), created_at=d["created_at"],
                     factors=Factors(**d["factors"]), info=InfoModel(d["info"]["kind"], d["info"]["params"]), cost=ComputeCost(**d["cost"]),
                     subsystem=d.get("subsystem", ""), family=d.get("family", ""), config=d.get("config", {}),
                     data_needs=tuple(d.get("data_needs", ())), overfit_hint=d.get("overfit_hint", 0.0),
                     free_parameters=d.get("free_parameters", 0), prior_tests_on_window=d.get("prior_tests_on_window", 0),
                     evidence=tuple(d.get("evidence", ())), magnitude=d.get("magnitude", 0.0))


def _plain(x):
    """JSON-safe copy of nested params (tuples become lists, numpy scalars become python numbers)."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        return x.item()
    return x


def config_to_dict(cfg: PolicyConfig) -> dict:
    return dict(cfg.__dict__)


def config_from_dict(d: Mapping) -> PolicyConfig:
    known = {k: v for k, v in d.items() if k in PolicyConfig.__dataclass_fields__}
    unknown = sorted(set(d) - set(known))
    if unknown:
        raise ValueError(f"unknown policy config keys: {unknown}")
    return PolicyConfig(**known)


def allocation_table(a: Allocation) -> str:
    """Fixed-width table of an allocation for logs."""
    lines = ["target               share   minutes", "-" * 38]
    for t in TARGETS:
        s = a.shares.get(t.value, 0.0)
        if s > 0:
            lines.append(f"{t.value:<20} {s:6.1%} {a.minutes.get(t.value, 0.0):8.1f}")
    lines.append(f"explore {a.explore_share:.1%}  exploit {a.exploit_share:.1%}")
    return "\n".join(lines)
