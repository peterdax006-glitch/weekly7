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
