"""Experiment value accounting (C66 section 19; canon C63/C66). IMPLEMENTED - NOT VALIDATED.

After EVERY research job the accountant measures what the job was actually worth, from the job's own before/after data, in seven
quantities: compute_cost, information_gain, decision_change, risk_change, prediction_change, transfer_change, knowledge_redundancy
(+ complexity, replication). Nothing is taken from the job's own claim of success:

  * a component only COUNTS when its bootstrap interval excludes zero, so "+0.01% backtest improvement" is worth nothing unless it
    changed decisions, transfers, replicates and did not add complexity (`worthless_reasons` names exactly which of these failed);
  * a result that improves nothing right now but shows a whole FEATURE FAMILY is unreliable is HIGH value (verdict PREVENTIVE): its
    value is the compute it prevents, with the claim kept open to audit (`ClaimBook`) so a later contradiction reverses the credit;
  * an underpowered job is INCONCLUSIVE, never WORTHLESS: unknown stays unknown (section 33);
  * a job whose change would HURT (significant harm) is HARMFUL_IF_ADOPTED: it is a warning worth its information, not a win.

It builds on engine.learning.research_policy (`realised_bits`, `RealisedGain`, `ResearchTarget`, `EPS`) and the research-brain
vocabulary (`ExperimentValue`, `Problem`, `Stage`). Records are hash-chained, dated, and only visible to queries strictly after
their date (`as_of`). State lives in MATURED_RESEARCH_STATE. Public entry: `account_job(ledger, job, now, policy)`."""
from __future__ import annotations

import dataclasses
import enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.research_policy import RealisedGain, ResearchTarget, realised_bits
from engine.research.core import (ExperimentValue, FirewallBreach, Namespace, OBJECTIVE_ORDER, Problem, Stage, as_date,
                                  require_past, stable_hash)

NAMESPACE = Namespace.MATURED_RESEARCH
Z95 = 1.959963984540054


# ------------------------------------------------------------------------------------------------ intervals and resampling
@dataclasses.dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap 95% interval; `n` is the number of independent units it rests on."""
    est: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero_up(self) -> bool:
        return self.lo > 0.0

    @property
    def excludes_zero_down(self) -> bool:
        return self.hi < 0.0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict:
        return {"est": self.est, "lo": self.lo, "hi": self.hi, "n": self.n}


def _seed_for(job_id: str, seed: int) -> int:
    return (int(stable_hash(job_id, 8), 16) ^ (seed & 0xFFFFFFFF)) & 0x7FFFFFFF


def block_indices(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    """Moving-block bootstrap indices of length n (block=1 is the ordinary bootstrap). Weekly P&L is autocorrelated, so the risk
    interval must resample blocks or it is too narrow."""
    if n <= 0:
        return np.empty(0, dtype=int)
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_interval(fn, arrays: Sequence[np.ndarray], rng: np.random.Generator, n_boot: int = 400, block: int = 1) -> Interval:
    """Percentile interval for fn(*arrays) with the arrays resampled JOINTLY (paired data keep their pairing)."""
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        return Interval(0.0, 0.0, 0.0, 0)
    est = float(fn(*arrays))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        ix = block_indices(n, rng, block)
        v = fn(*[a[ix] for a in arrays])
        draws[i] = v if np.isfinite(v) else est
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Interval(est, float(lo), float(hi), n)


# ------------------------------------------------------------------------------------------------ risk
def cvar(returns: np.ndarray, q: float = 0.05) -> float:
    """Mean of the worst q share of outcomes (a negative number for a losing tail)."""
    r = np.sort(np.asarray(returns, dtype=float))
    if r.size == 0:
        return 0.0
    k = max(1, int(math.ceil(q * r.size)))
    return float(r[:k].mean())


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the compounded equity curve, as a positive fraction."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(r, -0.999999, None))
    peak = np.maximum.accumulate(eq)
    return float(np.max(1.0 - eq / peak))


def downside_deviation(returns: np.ndarray, target: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    d = np.minimum(r - target, 0.0)
    return float(math.sqrt(float(np.mean(d * d))))


def band_hit_rate(returns: np.ndarray, lo: float = 0.05, hi: float = 0.10) -> float:
    """Share of periods whose return lands in the 5-10% weekly band (Bible objective: volatility inside the band)."""
    r = np.asarray(returns, dtype=float)
    return float(np.mean((r >= lo) & (r <= hi))) if r.size else 0.0


def risk_profile(returns: np.ndarray) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    return {"cvar5": cvar(r, 0.05), "worst": float(r.min()) if r.size else 0.0, "max_drawdown": max_drawdown(r),
            "downside_dev": downside_deviation(r), "positive_share": float(np.mean(r > 0)) if r.size else 0.0,
            "band_hit": band_hit_rate(r), "mean": float(r.mean()) if r.size else 0.0}


def _tail_loss_reduction(before: np.ndarray, after: np.ndarray, q: float = 0.05) -> float:
    lb, la = -cvar(before, q), -cvar(after, q)
    return (lb - la) / max(lb, 1e-9) if lb > 1e-9 else (0.0 if la <= 1e-9 else -1.0)


@dataclasses.dataclass(frozen=True)
class RiskChange:
    reduction: Interval                 # relative CVaR(5%) reduction; positive = safer
    deltas: Mapping[str, float]         # after - before for every risk_profile metric
    n: int


def measure_risk(before: np.ndarray, after: np.ndarray, rng: np.random.Generator, block: int = 4, n_boot: int = 400) -> RiskChange:
    """Paired comparison of two P&L series over the SAME periods. The interval resamples periods jointly in blocks."""
    b, a = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if b.shape != a.shape:
        raise ValueError("risk series must cover the same periods (paired)")
    pb, pa = risk_profile(b), risk_profile(a)
    iv = bootstrap_interval(_tail_loss_reduction, [b, a], rng, n_boot, block)
    return RiskChange(iv, {k: pa[k] - pb[k] for k in pb}, len(b))


# ------------------------------------------------------------------------------------------------ prediction
@dataclasses.dataclass(frozen=True)
class PredictionChange:
    kind: str
    loss_before: float
    loss_after: float
    improvement: Interval               # relative loss reduction; positive = better
    bic_gain: float | None              # log-likelihood gain minus the BIC price of the added parameters (Gaussian error model)


def _loss(kind: str, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    if kind == "brier" or kind == "mse":
        return (p - y) ** 2
    if kind == "logloss":
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    if kind == "mae":
        return np.abs(p - y)
    raise ValueError(f"unknown loss kind {kind}")


def _rel_improvement(lb: np.ndarray, la: np.ndarray) -> float:
    mb = float(lb.mean())
    return (mb - float(la.mean())) / mb if mb > 1e-12 else 0.0


def bic_gain(mse_before: float, mse_after: float, n: int, k_added: int) -> float:
    """Gaussian log-likelihood gain (n/2) ln(mse_b/mse_a) minus the BIC price (k/2) ln n of k added parameters. Positive = the
    improvement pays for its own complexity; a tiny gain bought with many parameters comes out negative."""
    if n <= 1 or mse_before <= 0 or mse_after <= 0:
        return 0.0
    return 0.5 * n * math.log(mse_before / mse_after) - 0.5 * max(0, k_added) * math.log(n)


def measure_prediction(p_before: np.ndarray, p_after: np.ndarray, y: np.ndarray, rng: np.random.Generator, kind: str = "brier",
                       k_added: int = 0, n_boot: int = 400) -> PredictionChange:
    pb, pa, yy = (np.asarray(v, dtype=float) for v in (p_before, p_after, y))
    if not (pb.shape == pa.shape == yy.shape):
        raise ValueError("predictions and outcomes must align")
    lb, la = _loss(kind, pb, yy), _loss(kind, pa, yy)
    iv = bootstrap_interval(lambda x, z: _rel_improvement(x, z), [lb, la], rng, n_boot, 1)
    g = bic_gain(float(np.mean((pb - yy) ** 2)), float(np.mean((pa - yy) ** 2)), len(yy), k_added) if len(yy) else None
    return PredictionChange(kind, float(lb.mean()) if len(lb) else 0.0, float(la.mean()) if len(la) else 0.0, iv, g)


# ------------------------------------------------------------------------------------------------ decisions
@dataclasses.dataclass(frozen=True)
class DecisionChange:
    k: int
    fraction_changed: float             # share of the top-k selection replaced
    n_swapped: int
    rank_corr: float                    # Spearman between the two score vectors over the whole universe
    gain: Interval | None               # mean outcome of the newly chosen minus the dropped; None if nothing changed


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def measure_decisions(score_before: np.ndarray, score_after: np.ndarray, outcome: np.ndarray | None, k: int,
                      rng: np.random.Generator, n_boot: int = 400) -> DecisionChange:
    """Did the job change WHAT the system would have chosen? Ties are broken by index so the answer never depends on hidden state.
    If outcomes are supplied the swap is also judged: were the newly chosen items better than the ones they replaced?"""
    sb, sa = np.asarray(score_before, dtype=float), np.asarray(score_after, dtype=float)
    if sb.shape != sa.shape or sb.ndim != 1:
        raise ValueError("score vectors must be 1-D over the same universe")
    n = len(sb)
    if n == 0 or k < 1:
        return DecisionChange(max(k, 0), 0.0, 0, 0.0, None)
    k = min(k, n)
    top_b = set(np.lexsort((np.arange(n), -sb))[:k].tolist())
    top_a = set(np.lexsort((np.arange(n), -sa))[:k].tolist())
    new_only, dropped = sorted(top_a - top_b), sorted(top_b - top_a)
    rb, ra = _rank(sb), _rank(sa)
    rc = float(np.corrcoef(rb, ra)[0, 1]) if n > 2 and rb.std() > 0 and ra.std() > 0 else 1.0
    gain = None
    if outcome is not None and new_only:
        y = np.asarray(outcome, dtype=float)
        if len(y) != n:
            raise ValueError("outcome must align with the score vectors")
        yn, yd = y[new_only], y[dropped]
        est = float(yn.mean() - yd.mean())
        draws = np.empty(n_boot)
        for i in range(n_boot):
            draws[i] = yn[rng.integers(0, len(yn), len(yn))].mean() - yd[rng.integers(0, len(yd), len(yd))].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        gain = Interval(est, float(lo), float(hi), len(new_only))
    return DecisionChange(k, len(new_only) / k, len(new_only), rc, gain)


# ------------------------------------------------------------------------------------------------ transfer
@dataclasses.dataclass(frozen=True)
class TransferResult:
    n_contexts: int
    pooled: float
    pooled_se: float
    tau2: float                         # between-context variance (DerSimonian-Laird)
    i2: float                           # share of variation that is heterogeneity, 0..1
    same_sign: float                    # share of contexts whose effect has the pooled sign
    score: float                        # 0..1: high only if the effect appears in many contexts, with the same sign, consistently


def measure_transfer(effects: Mapping[str, tuple[float, float]], min_contexts: int = 3) -> TransferResult:
    """Random-effects summary of an effect measured in several contexts (years, sectors, regimes). The score is deliberately hard
    to earn: the pooled effect must be significant, most contexts must agree in sign, and heterogeneity must be low. Fewer than
    `min_contexts` contexts are scaled down (one context is not transfer at all)."""
    items = [(float(e), float(s)) for e, s in effects.values() if math.isfinite(e) and math.isfinite(s) and s > 0]
    k = len(items)
    if k == 0:
        return TransferResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    e = np.array([i[0] for i in items])
    v = np.array([i[1] ** 2 for i in items])
    w = 1.0 / v
    fixed = float((w * e).sum() / w.sum())
    q = float((w * (e - fixed) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    i2 = max(0.0, (q - (k - 1)) / q) if k > 1 and q > 0 else 0.0
    wr = 1.0 / (v + tau2)
    pooled = float((wr * e).sum() / wr.sum())
    se = float(math.sqrt(1.0 / wr.sum()))
    sign = 1.0 if pooled >= 0 else -1.0
    same = float(np.mean(np.sign(e) == sign))
    sig = 1.0 if abs(pooled) / se >= 2.0 else abs(pooled) / se / 2.0
    score = sig * same * (1.0 - min(1.0, i2)) * min(1.0, k / max(min_contexts, 1))
    if pooled <= 0:
        score = 0.0
    return TransferResult(k, pooled, se, tau2, i2, same, float(score))


# ------------------------------------------------------------------------------------------------ redundancy and information
@dataclasses.dataclass(frozen=True)
class RedundancyResult:
    r2_existing: float                  # share of the new signal explained by the existing ones
    max_abs_corr: float
    selection_overlap: float            # Jaccard overlap of what it selects with what existing knowledge selects
    redundancy: float                   # the largest of the three: redundant if ANY channel says it is a copy


def measure_redundancy(new_signal: np.ndarray | None, existing: np.ndarray | None, selected_new: Iterable | None = None,
                       selected_existing: Iterable | None = None) -> RedundancyResult:
    r2 = mc = ov = 0.0
    if new_signal is not None and existing is not None and np.size(existing):
        x = np.asarray(new_signal, dtype=float)
        E = np.asarray(existing, dtype=float)
        E = E.reshape(len(x), -1)
        if len(x) > 3 and x.std() > 0:
            xs = (x - x.mean()) / x.std()
            keep = E.std(axis=0) > 0
            if keep.any():
                Es = (E[:, keep] - E[:, keep].mean(axis=0)) / E[:, keep].std(axis=0)
                beta, *_ = np.linalg.lstsq(Es, xs, rcond=None)
                res = xs - Es @ beta
                r2 = float(max(0.0, 1.0 - (res @ res) / (xs @ xs)))
                mc = float(np.max(np.abs(Es.T @ xs) / len(xs)))
    if selected_new is not None and selected_existing is not None:
        a, b = set(selected_new), set(selected_existing)
        ov = len(a & b) / len(a | b) if (a | b) else 0.0
    return RedundancyResult(r2, mc, ov, max(r2, mc * mc, ov))


def information_gain_bits(prior: Mapping[str, float] | None, posterior: Mapping[str, float] | None,
                          prior_se: float | None = None, post_se: float | None = None) -> tuple[float, str]:
    """Bits learned. From hypothesis beliefs when the job updated them (entropy removed, via research_policy.realised_bits), else
    from the narrowing of an estimate's standard error (each halving = 1 bit), else 0 with the source 'unmeasured' so a job that
    cannot say what it learned is not credited with learning."""
    if prior and posterior:
        return float(realised_bits(prior, posterior)), "beliefs"
    if prior_se and post_se and prior_se > 0 and post_se > 0:
        return float(max(0.0, math.log2(prior_se / post_se))), "standard_error"
    return 0.0, "unmeasured"


# ------------------------------------------------------------------------------------------------ the job and the value vector
@dataclasses.dataclass(frozen=True)
class FamilyFinding:
    """What a job established about a whole FEATURE FAMILY (not one hypothesis): the raw counts behind 'this family is unreliable'.
    The accountant computes the strength itself; the job only supplies the facts."""
    family: str
    n_hypotheses: int                      # distinct hypotheses tested in the family
    n_powered_null: int                    # of those, tests with the power to see a useful effect that saw none
    n_significant: int                     # apparent winners after multiplicity
    n_significant_replicated: int          # of the winners, those that replicated / transferred
    pooled_effect: Interval | None         # family-wide effect estimate
    min_effect: float                      # smallest useful effect
    pending_cpu_min: float = 0.0           # queued experiments in this family that the finding would cancel
    future_cpu_min_per_week: float = 0.0   # the rate at which the research loop would otherwise keep generating them
    horizon_weeks: float = 26.0

    def validate(self) -> list[str]:
        errs = []
        if min(self.n_hypotheses, self.n_powered_null, self.n_significant, self.n_significant_replicated) < 0:
            errs.append("finding counts negative")
        if self.n_powered_null > self.n_hypotheses or self.n_significant > self.n_hypotheses:
            errs.append("finding counts exceed hypotheses tested")
        if self.n_significant_replicated > self.n_significant:
            errs.append("replicated winners exceed winners")
        if self.pending_cpu_min < 0 or self.future_cpu_min_per_week < 0 or self.horizon_weeks < 0 or self.min_effect <= 0:
            errs.append("finding costs/horizon/min_effect out of range")
        return errs


@dataclasses.dataclass(eq=False)
class JobMeasurement:
    """Everything the accountant needs about one finished job. All arrays are optional: a quantity that cannot be measured is
    reported as 'unmeasured' and is NEVER treated as zero improvement or as success."""
    job_id: str
    branch_id: str
    family: str
    problem: Problem
    stage: Stage
    at: str                                   # date the job finished
    data_through: str                         # newest data date it used
    cost_cpu_min: float
    wall_min: float = 0.0
    ram_gb: float = 0.0
    code_hash: str = ""
    data_hash: str = ""
    seed: int = 0
    target: str = ResearchTarget.UNKNOWN_AREA.value
    # decisions
    score_before: np.ndarray | None = None
    score_after: np.ndarray | None = None
    outcome: np.ndarray | None = None
    k: int = 0
    # risk
    pnl_before: np.ndarray | None = None
    pnl_after: np.ndarray | None = None
    # prediction
    pred_before: np.ndarray | None = None
    pred_after: np.ndarray | None = None
    target_values: np.ndarray | None = None
    loss_kind: str = "brier"
    # transfer
    context_effects: Mapping[str, tuple[float, float]] | None = None
    baseline_transfer: float = 0.0
    # redundancy
    new_signal: np.ndarray | None = None
    existing_signals: np.ndarray | None = None
    selected_new: frozenset | None = None
    selected_existing: frozenset | None = None
    # information
    prior_beliefs: Mapping[str, float] | None = None
    posterior_beliefs: Mapping[str, float] | None = None
    prior_se: float | None = None
    post_se: float | None = None
    powered_null: bool = False                # the job's own hypothesis got a powered null
    # replication and complexity
    replications: int = 0
    params_added: int = 0
    features_added: int = 0
    # family verdict and the pre-job estimate
    family_finding: FamilyFinding | None = None
    estimate: ExperimentValue | None = None


@dataclasses.dataclass(frozen=True)
class RealisedValue:
    """The section-19 vector, as measured. None = unmeasured (never read as 0)."""
    compute_cost: float
    information_gain: float | None
    information_source: str
    decision_change: float | None
    decision_gain: Interval | None
    risk_change: Interval | None
    risk_deltas: Mapping[str, float] | None
    prediction_change: Interval | None
    prediction_bic_gain: float | None
    transfer_change: float | None
    transfer: TransferResult | None
    knowledge_redundancy: float | None
    complexity_units: float
    replicated: bool
    family_unreliability: float
    prevented_cpu_min: float
    measured: tuple[str, ...]

    def to_dict(self) -> dict:
        def iv(x):
            return x.to_dict() if x is not None else None
        return {"compute_cost": self.compute_cost, "information_gain": self.information_gain, "information_source": self.information_source,
                "decision_change": self.decision_change, "decision_gain": iv(self.decision_gain), "risk_change": iv(self.risk_change),
                "risk_deltas": dict(self.risk_deltas) if self.risk_deltas is not None else None,
                "prediction_change": iv(self.prediction_change), "prediction_bic_gain": self.prediction_bic_gain,
                "transfer_change": self.transfer_change, "transfer_score": self.transfer.score if self.transfer else None,
                "knowledge_redundancy": self.knowledge_redundancy, "complexity_units": self.complexity_units,
                "replicated": self.replicated, "family_unreliability": self.family_unreliability,
                "prevented_cpu_min": self.prevented_cpu_min, "measured": list(self.measured)}


class Verdict(str, enum.Enum):
    USEFUL = "USEFUL"
    PREVENTIVE = "PREVENTIVE"                 # a family was shown unreliable: value = compute and error avoided
    INFORMATIVE_NULL = "INFORMATIVE_NULL"     # a powered null closed a question: small, real value
    WORTHLESS = "WORTHLESS"
    INCONCLUSIVE = "INCONCLUSIVE"             # too noisy to say either way; NOT worthless
    HARMFUL_IF_ADOPTED = "HARMFUL_IF_ADOPTED"  # significant deterioration: a warning that was worth its information
    INVALID = "INVALID"                       # the measurement itself is malformed; carries no value
    CORRECTION = "CORRECTION"                 # a later finding changed the credit of an earlier record (never an edit)


@dataclasses.dataclass(frozen=True)
class ValuePolicy:
    """Conventions, to be tuned in the validation wave. Weights follow the objective hierarchy: losses and volatility outrank
    direction (section 34)."""
    w_decision: float = 1.0
    w_risk: float = 1.2
    w_prediction: float = 0.8
    w_transfer: float = 0.8
    w_information_per_bit: float = 0.05
    information_bit_cap: float = 6.0
    price_per_cpu_min: float = 0.002          # value units a CPU-minute must return to break even
    complexity_price: float = 0.01            # value units per added parameter/feature
    unreplicated_discount: float = 0.5        # positive value from a job below the cross-year rung and not replicated
    preventive_multiplier: float = 2.0        # a prevented CPU-minute is worth more than a spent one: it also removes false leads
    min_hypotheses_for_family: int = 8
    preventive_strength: float = 0.7
    decision_scale: float = 0.01              # outcome gain of swapped picks that counts as "full" (tanh scale)
    risk_scale: float = 0.25
    prediction_scale: float = 0.05
    material_decision_fraction: float = 0.05
    material_risk: float = 0.02
    material_prediction: float = 0.005
    material_transfer: float = 0.10
    conclusive_width: Mapping[str, float] = dataclasses.field(default_factory=lambda: {"risk": 0.6, "prediction": 0.08, "decision": 0.08})
    n_boot: int = 400
    block: int = 4
    min_n_conclusive: int = 30                # fewer independent units than this can never make a component conclusive
    problem_weight: Mapping[str, float] = dataclasses.field(default_factory=lambda: {
        p.value: w for p, w in zip(OBJECTIVE_ORDER, (1.0, 1.1, 0.85, 0.7, 0.6, 0.5, 0.4))})

    def validate(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and (not math.isfinite(v) or v < 0):
                errs.append(f"{f.name} must be finite and >= 0")
        if not 0.0 < self.unreplicated_discount <= 1.0 or not 0.0 < self.preventive_strength <= 1.0:
            errs.append("discount/strength must be in (0, 1]")
        if self.price_per_cpu_min <= 0 or self.n_boot < 50 or self.min_hypotheses_for_family < 2:
            errs.append("price > 0, n_boot >= 50, min_hypotheses_for_family >= 2 required")
        return errs


def family_unreliability(f: FamilyFinding | None, policy: ValuePolicy) -> float:
    """How strongly a finding shows the family is unreliable, in [0, 1]. Requires (a) many hypotheses (a family is not judged from
    three), (b) most of them powered nulls, (c) a pooled interval that excludes a useful effect, and (d) the apparent winners
    failing to replicate. Each factor multiplies: any missing one collapses the strength, so the claim is hard to make cheaply."""
    if f is None or f.validate() or f.n_hypotheses < policy.min_hypotheses_for_family:
        return 0.0
    if f.pooled_effect is None or f.pooled_effect.hi >= f.min_effect:
        return 0.0
    p_null = f.n_powered_null / f.n_hypotheses
    survivors = f.n_significant_replicated / f.n_hypotheses
    sat = 1.0 - math.exp(-f.n_hypotheses / 12.0)
    fail_replication = 1.0 if f.n_significant == 0 else 1.0 - f.n_significant_replicated / f.n_significant
    return float(max(0.0, min(1.0, p_null * (1.0 - survivors) * sat * (0.5 + 0.5 * fail_replication))))


def prevented_cpu_min(f: FamilyFinding | None, strength: float) -> float:
    """Compute the finding removes from the future: queued experiments it cancels plus the expected stream of new ones over the
    horizon, discounted by the strength of the claim (a weak claim prevents little)."""
    if f is None or strength <= 0:
        return 0.0
    future = f.future_cpu_min_per_week * f.horizon_weeks * 0.5      # halved: the loop would have pruned some of it anyway
    return float(strength * (f.pending_cpu_min + future))


def measure_job(job: JobMeasurement, now, policy: ValuePolicy) -> tuple[RealisedValue, list[str]]:
    """Do the numeric work for one job. Returns the value vector and the list of problems with the measurement (non-empty = the
    caller must treat the job as INVALID). Data at/after `now` is a FirewallBreach, not a problem: it must never be silent."""
    require_past(job.data_through, now, f"job {job.job_id} data_through")
    problems: list[str] = []
    if not job.job_id or not job.family:
        problems.append("job_id and family are required")
    if not (math.isfinite(job.cost_cpu_min) and job.cost_cpu_min >= 0):
        problems.append("cost_cpu_min must be finite and >= 0")
    rng = np.random.default_rng(_seed_for(job.job_id, job.seed))
    measured: list[str] = []
    dec = risk = pred = tr = red = None
    if job.score_before is not None or job.score_after is not None:
        if job.score_before is None or job.score_after is None:
            problems.append("decision scores need both before and after")
        else:
            try:
                dec = measure_decisions(job.score_before, job.score_after, job.outcome, job.k, rng, policy.n_boot)
                measured.append("decision_change")
            except ValueError as e:
                problems.append(str(e))
    if job.pnl_before is not None or job.pnl_after is not None:
        if job.pnl_before is None or job.pnl_after is None:
            problems.append("risk series need both before and after")
        else:
            try:
                if not (np.all(np.isfinite(job.pnl_before)) and np.all(np.isfinite(job.pnl_after))):
                    raise ValueError("P&L series contain non-finite values")
                risk = measure_risk(job.pnl_before, job.pnl_after, rng, policy.block, policy.n_boot)
                measured.append("risk_change")
            except ValueError as e:
                problems.append(str(e))
    if job.pred_before is not None or job.pred_after is not None:
        if job.pred_before is None or job.pred_after is None or job.target_values is None:
            problems.append("prediction change needs before, after and the realised target")
        else:
            try:
                pred = measure_prediction(job.pred_before, job.pred_after, job.target_values, rng, job.loss_kind,
                                          job.params_added + job.features_added, policy.n_boot)
                measured.append("prediction_change")
            except ValueError as e:
                problems.append(str(e))
    if job.context_effects is not None:
        tr = measure_transfer(job.context_effects)
        measured.append("transfer_change")
    if job.new_signal is not None or job.selected_new is not None:
        red = measure_redundancy(job.new_signal, job.existing_signals, job.selected_new, job.selected_existing)
        measured.append("knowledge_redundancy")
    bits, src = information_gain_bits(job.prior_beliefs, job.posterior_beliefs, job.prior_se, job.post_se)
    if src != "unmeasured":
        measured.append("information_gain")
    ff = job.family_finding
    if ff is not None and ff.validate():
        problems.append("family finding malformed: " + "; ".join(ff.validate()))
        ff = None
    strength = family_unreliability(ff, policy)
    rv = RealisedValue(
        job.cost_cpu_min, bits if src != "unmeasured" else None, src, dec.fraction_changed if dec else None, dec.gain if dec else None,
        risk.reduction if risk else None, risk.deltas if risk else None, pred.improvement if pred else None,
        pred.bic_gain if pred else None, (tr.score - job.baseline_transfer) if tr else None, tr,
        red.redundancy if red else None, float(job.params_added + job.features_added), job.replications >= 1, strength,
        prevented_cpu_min(ff, strength), tuple(measured))
    return rv, problems


# ------------------------------------------------------------------------------------------------ judgement
@dataclasses.dataclass(frozen=True)
class Judgement:
    verdict: Verdict
    net_value: float                          # value units, after compute and complexity are paid for
    components: Mapping[str, float]           # signed contribution of each component, value units
    cost_value: float
    reasons: tuple[str, ...]
    worthless_reasons: tuple[str, ...]        # the contract's list, each stated only when it applies
    material: tuple[str, ...]                 # components that were significant AND large enough
    harmful: tuple[str, ...]
    inconclusive: tuple[str, ...]


def _gate(iv: Interval | None, material: float, width_limit: float, min_n: int = 0) -> tuple[int, bool | None]:
    """(sign, conclusive). +1/-1 only if the interval excludes zero on that side AND the estimate is material; otherwise 0.
    conclusive is False when the interval is so wide that 'no effect' cannot be claimed either; None when unmeasured."""
    if iv is None or iv.n == 0:
        return 0, None
    if iv.n < min_n:
        return 0, False                      # too few independent units for an interval to mean anything
    if iv.lo > 0 and iv.est >= material:
        return 1, True
    if iv.hi < 0 and iv.est <= -material:
        return -1, True
    return 0, iv.width <= width_limit


def judge(rv: RealisedValue, stage: Stage, problem: Problem, powered_null: bool, policy: ValuePolicy) -> Judgement:
    """Turn the measured vector into a verdict and a net value. The order encodes the contract: a component counts only if
    significant and material; redundancy and missing replication shrink value; a claim about a whole family is judged separately."""
    pw = policy.problem_weight.get(problem.value, 0.5)
    comps: dict[str, float] = {}
    material: list[str] = []
    harmful: list[str] = []
    inconc: list[str] = []
    wr: list[str] = []
    reasons: list[str] = []
    wl = policy.conclusive_width
    # decisions
    if rv.decision_change is not None:
        if rv.decision_change <= 0.0:
            wr.append("does not change decisions")
        elif rv.decision_gain is None:
            wr.append("changed decisions but with no outcome evidence that the change helped")
        else:
            sign, concl = _gate(rv.decision_gain, 0.0, wl["decision"], policy.min_n_conclusive // 6)
            frac = rv.decision_change
            if sign > 0 and frac >= policy.material_decision_fraction:
                comps["decision"] = pw * policy.w_decision * frac * math.tanh(rv.decision_gain.est / policy.decision_scale)
                material.append("decision")
            elif sign < 0 and frac >= policy.material_decision_fraction:
                comps["decision"] = -pw * policy.w_decision * frac * math.tanh(-rv.decision_gain.est / policy.decision_scale)
                harmful.append("decision")
            elif concl is False:
                inconc.append("decision")
            else:
                wr.append("decisions changed but not measurably for the better")
    # risk
    if rv.risk_change is not None:
        sign, concl = _gate(rv.risk_change, policy.material_risk, wl["risk"], policy.min_n_conclusive)
        if sign > 0:
            comps["risk"] = pw * policy.w_risk * math.tanh(rv.risk_change.est / policy.risk_scale)
            material.append("risk")
        elif sign < 0:
            comps["risk"] = -pw * policy.w_risk * math.tanh(-rv.risk_change.est / policy.risk_scale)
            harmful.append("risk")
        elif concl is False:
            inconc.append("risk")
        else:
            wr.append("does not reduce risk")
    # prediction
    if rv.prediction_change is not None:
        sign, concl = _gate(rv.prediction_change, policy.material_prediction, wl["prediction"], policy.min_n_conclusive)
        if sign > 0 and rv.prediction_bic_gain is not None and rv.prediction_bic_gain < 0:
            sign = 0
            concl = True
            wr.append("adds complexity: the improvement does not pay for its added parameters (BIC)")
        if sign > 0:
            comps["prediction"] = pw * policy.w_prediction * math.tanh(rv.prediction_change.est / policy.prediction_scale)
            material.append("prediction")
        elif sign < 0:
            comps["prediction"] = -pw * policy.w_prediction * math.tanh(-rv.prediction_change.est / policy.prediction_scale)
            harmful.append("prediction")
        elif concl is False:
            inconc.append("prediction")
        elif not any("BIC" in r for r in wr):
            wr.append("prediction improvement is below noise or below materiality")
    # transfer
    if rv.transfer_change is not None:
        if rv.transfer_change >= policy.material_transfer:
            comps["transfer"] = pw * policy.w_transfer * rv.transfer_change
            material.append("transfer")
        elif rv.transfer_change <= -policy.material_transfer:
            comps["transfer"] = pw * policy.w_transfer * rv.transfer_change
            harmful.append("transfer")
        else:
            wr.append("does not transfer")
    # information: always credited, never sufficient on its own
    bits = min(rv.information_gain or 0.0, policy.information_bit_cap)
    info_value = policy.w_information_per_bit * bits
    if rv.information_gain is not None:
        comps["information"] = info_value
    red = rv.knowledge_redundancy or 0.0
    if red >= 0.9:
        wr.append("redundant with existing knowledge")
    elif red > 0.5:
        wr.append(f"largely redundant ({red:.0%})")
    rep_ok = rv.replicated or stage_rank(stage) >= stage_rank(Stage.CROSS_YEAR)
    if not rep_ok:
        wr.append("does not replicate (no independent replication yet)")
    if rv.complexity_units > 0:
        wr.append(f"adds complexity ({rv.complexity_units:.0f} parameters/features)")
    positive = sum(v for k, v in comps.items() if k != "information" and v > 0)
    negative = sum(v for v in comps.values() if v < 0)
    positive *= (1.0 - min(red, 1.0))
    if positive > 0 and not rep_ok:
        positive *= policy.unreplicated_discount
    cx_value = policy.complexity_price * rv.complexity_units
    cost_value = policy.price_per_cpu_min * rv.compute_cost
    pv = 0.0
    if rv.family_unreliability >= policy.preventive_strength and rv.prevented_cpu_min > 0:
        pv = policy.price_per_cpu_min * rv.prevented_cpu_min * policy.preventive_multiplier
    comps["prevented_compute"] = pv
    comps["complexity"] = -cx_value
    comps["compute"] = -cost_value
    net = positive + info_value + pv - cx_value - cost_value
    if pv > 0:
        reasons.append(f"a feature family is unreliable (strength {rv.family_unreliability:.2f}): prevents ~{rv.prevented_cpu_min:.0f} cpu-min")
        return Judgement(Verdict.PREVENTIVE, net, comps, cost_value, tuple(reasons), tuple(wr), tuple(material), tuple(harmful), tuple(inconc))
    if harmful and not material:
        reasons.append("significantly worse than what it would replace: adopting it would hurt (" + ", ".join(harmful) + ")")
        return Judgement(Verdict.HARMFUL_IF_ADOPTED, info_value - cost_value, comps, cost_value, tuple(reasons), tuple(wr),
                         tuple(material), tuple(harmful), tuple(inconc))
    if material:
        transfer_ok = "transfer" in material
        if net > 0 and (rep_ok or transfer_ok):
            reasons.append("significant, material, " + ("replicated" if rep_ok else "transferring") + ": " + ", ".join(material))
            return Judgement(Verdict.USEFUL, net, comps, cost_value, tuple(reasons), (), tuple(material), tuple(harmful), tuple(inconc))
        if net > 0:
            reasons.append("promising lead (" + ", ".join(material) + ") but unreplicated and untransferred: not yet proven useful")
            return Judgement(Verdict.INCONCLUSIVE, net, comps, cost_value, tuple(reasons), tuple(wr), tuple(material), tuple(harmful), tuple(inconc))
        reasons.append("the gain does not repay its compute and complexity")
        return Judgement(Verdict.WORTHLESS, net, comps, cost_value, tuple(reasons), tuple(wr), tuple(material), tuple(harmful), tuple(inconc))
    if inconc:
        reasons.append("too noisy to say either way (" + ", ".join(inconc) + "): unknown, not worthless")
        return Judgement(Verdict.INCONCLUSIVE, info_value - cost_value, comps, cost_value, tuple(reasons), tuple(wr), (), tuple(harmful), tuple(inconc))
    if powered_null and bits > 0:
        reasons.append("a powered null closed a question")
        return Judgement(Verdict.INFORMATIVE_NULL, info_value - cost_value, comps, cost_value, tuple(reasons), tuple(wr), (), (), ())
    if not rv.measured:
        reasons.append("nothing was measured, so nothing can be credited")
        return Judgement(Verdict.INCONCLUSIVE, -cost_value, comps, cost_value, tuple(reasons), tuple(wr), (), (), ())
    reasons.append("measured, conclusive, and nothing material changed")
    return Judgement(Verdict.WORTHLESS, info_value - cost_value - cx_value, comps, cost_value, tuple(reasons), tuple(wr), (), (), ())


def stage_rank(stage: Stage) -> int:
    return list(Stage).index(stage)


# ------------------------------------------------------------------------------------------------ records and ledger
@dataclasses.dataclass(frozen=True)
class ValueRecord:
    seq: int
    job_id: str
    branch_id: str
    family: str
    problem: str
    stage: str
    target: str
    at: str                                   # date the job finished: visible to queries strictly after this
    accounted_at: str                         # date it was accounted (chain order)
    verdict: str
    net_value: float
    cost_cpu_min: float
    cost_value: float
    components: Mapping[str, float]
    value: Mapping[str, Any]
    reasons: tuple[str, ...]
    worthless_reasons: tuple[str, ...]
    estimate: Mapping[str, Any] | None
    code_hash: str
    data_hash: str
    seed: int
    prev_hash: str = ""
    id: str = ""

    def body(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("id")
        return json.loads(json.dumps(d, sort_keys=True, default=str))

    def compute_id(self) -> str:
        return stable_hash(self.body(), 20)


@dataclasses.dataclass(frozen=True)
class FamilyStats:
    family: str
    jobs: int
    cpu_min: float
    net_value: float
    counts: Mapping[str, int]
    value_per_cpu_min: float

    @property
    def worthless_share(self) -> float:
        return self.counts.get(Verdict.WORTHLESS.value, 0) / self.jobs if self.jobs else 0.0


class ValueLedger:
    """Append-only, hash-chained record of what every job was worth. Queries take `as_of` and see only jobs finished strictly
    before it. Corrections (a preventive claim later refuted) are new records, never edits."""

    def __init__(self):
        self._log: list[ValueRecord] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._log)

    def add(self, rec: ValueRecord) -> ValueRecord:
        if rec.job_id in self._ids:
            raise ValueError(f"job {rec.job_id} is already accounted; a job is valued once")
        if self._log and as_date(rec.accounted_at) < as_date(self._log[-1].accounted_at):
            raise FirewallBreach(f"accounting dated {rec.accounted_at} is before the previous record {self._log[-1].accounted_at}")
        rec = dataclasses.replace(rec, seq=len(self._log), prev_hash=self._log[-1].id if self._log else "")
        rec = dataclasses.replace(rec, id=rec.compute_id())
        self._log.append(rec)
        self._ids.add(rec.job_id)
        return rec

    def records(self, as_of=None) -> list[ValueRecord]:
        if as_of is None:
            return list(self._log)
        cut = as_date(as_of)
        return [r for r in self._log if as_date(r.at) < cut]

    def get(self, job_id: str) -> ValueRecord | None:
        return next((r for r in self._log if r.job_id == job_id), None)

    def verify_chain(self) -> list[str]:
        errs, prev = [], ""
        for i, r in enumerate(self._log):
            if r.seq != i or r.prev_hash != prev:
                errs.append(f"record {i}: chain broken")
            if r.compute_id() != r.id:
                errs.append(f"record {i}: content edited")
            prev = r.id
        return errs

    # ---- statistics
    def family_stats(self, as_of, prior_k: float = 5.0) -> dict[str, FamilyStats]:
        recs = self.records(as_of)
        jobs = [r for r in recs if r.verdict != Verdict.CORRECTION.value]
        tot_cpu = sum(r.cost_cpu_min for r in jobs)
        tot_val = sum(r.net_value for r in recs)
        glob = tot_val / tot_cpu if tot_cpu > 0 else 0.0
        out: dict[str, FamilyStats] = {}
        for fam in sorted({r.family for r in recs}):
            fr = [r for r in recs if r.family == fam]
            fj = [r for r in fr if r.verdict != Verdict.CORRECTION.value]
            cpu = sum(r.cost_cpu_min for r in fj)
            val = sum(r.net_value for r in fr)
            counts: dict[str, int] = {}
            for r in fj:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
            rate = (val + prior_k * glob) / (cpu + prior_k) if (cpu + prior_k) > 0 else 0.0
            out[fam] = FamilyStats(fam, len(fj), cpu, val, counts, rate)
        return out

    def drought(self, family: str, as_of) -> int:
        """Consecutive most-recent jobs of the family that were neither USEFUL, PREVENTIVE nor an INFORMATIVE_NULL."""
        n = 0
        for r in reversed([r for r in self.records(as_of) if r.family == family and r.verdict != Verdict.CORRECTION.value]):
            if r.verdict in (Verdict.USEFUL.value, Verdict.PREVENTIVE.value, Verdict.INFORMATIVE_NULL.value):
                break
            n += 1
        return n

    def totals(self, as_of) -> dict[str, float]:
        recs = self.records(as_of)
        jobs = [r for r in recs if r.verdict != Verdict.CORRECTION.value]
        cpu = sum(r.cost_cpu_min for r in jobs)
        wasted = sum(r.cost_cpu_min for r in jobs if r.verdict in (Verdict.WORTHLESS.value, Verdict.INVALID.value))
        by_verdict: dict[str, float] = {}
        for r in jobs:
            by_verdict[r.verdict] = by_verdict.get(r.verdict, 0.0) + r.cost_cpu_min
        return {"jobs": float(len(jobs)), "cpu_min": cpu, "net_value": sum(r.net_value for r in recs), "wasted_cpu_min": wasted,
                "waste_share": wasted / cpu if cpu > 0 else 0.0, **{f"cpu_{k}": v for k, v in sorted(by_verdict.items())}}

    def worthless_reason_counts(self, as_of) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records(as_of):
            if r.verdict == Verdict.WORTHLESS.value:
                for w in r.worthless_reasons:
                    key = w.split("(")[0].strip()
                    out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def gains(self, as_of) -> list[RealisedGain]:
        """The same jobs in research_policy's RealisedGain form, so the existing priority function's experience factor and
        barren-streak logic can learn from the accountant's verdicts instead of from the job's own claim."""
        out = []
        for r in self.records(as_of):
            if r.verdict == Verdict.CORRECTION.value:
                continue
            bits = float(r.value.get("information_gain") or 0.0)
            useful = r.verdict in (Verdict.USEFUL.value, Verdict.PREVENTIVE.value)
            try:
                tgt = ResearchTarget.parse(r.target)
            except ValueError:
                tgt = ResearchTarget.UNKNOWN_AREA
            out.append(RealisedGain(r.job_id, tgt, r.family, float((r.estimate or {}).get("information_gain") or 0.0), bits, useful,
                                    r.cost_cpu_min, r.at))
        return out

    # ---- persistence
    def dump(self, path: str | Path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(dataclasses.asdict(r), sort_keys=True, default=str) for r in self._log) + ("\n" if self._log else "")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(p)
        return len(self._log)

    @classmethod
    def load(cls, path: str | Path) -> "ValueLedger":
        led = cls()
        p = Path(path)
        if not p.exists():
            return led
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["reasons"], d["worthless_reasons"] = tuple(d["reasons"]), tuple(d["worthless_reasons"])
            r = ValueRecord(**d)
            led._log.append(r)
            led._ids.add(r.job_id)
        bad = led.verify_chain()
        if bad:
            raise ValueError("value ledger failed verification: " + "; ".join(bad[:3]))
        return led


# ------------------------------------------------------------------------------------------------ family findings from raw results
@dataclasses.dataclass(frozen=True)
class HypothesisResult:
    """One tested hypothesis of a family, as the research loop recorded it."""
    name: str
    effect: float
    se: float
    n_obs: int
    replicated: bool = False


def holm_rejections(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected while controlling the family-wise error at alpha."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (pvals[i], i))
    rej = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def build_family_finding(family: str, results: Sequence[HypothesisResult], min_effect: float, alpha: float = 0.05,
                         target_power: float = 0.80, pending_cpu_min: float = 0.0, future_cpu_min_per_week: float = 0.0,
                         horizon_weeks: float = 26.0) -> FamilyFinding:
    """Compute the facts behind 'this family is unreliable' from per-hypothesis results, so the claim rests on power calculations
    and multiplicity control done HERE rather than on a job's own summary:
      * a hypothesis is a POWERED NULL if its test had at least `target_power` to see `min_effect` and its interval excludes it;
      * winners are the Holm-corrected significant positive effects, and count as replicated only if the run said so;
      * the family effect is a random-effects pool across hypotheses (heterogeneity widens it)."""
    from scipy.stats import norm
    from engine.learning.research_policy import power_of
    valid = [r for r in results if math.isfinite(r.effect) and math.isfinite(r.se) and r.se > 0 and r.n_obs > 1]
    if not valid:
        return FamilyFinding(family, 0, 0, 0, 0, None, min_effect, pending_cpu_min, future_cpu_min_per_week, horizon_weeks)
    z_crit = float(norm.ppf(1 - alpha / 2))
    p = [float(2 * (1 - norm.cdf(abs(r.effect / r.se)))) for r in valid]
    rej = holm_rejections(p, alpha)
    powered_null = 0
    for r in valid:
        sd = r.se * math.sqrt(r.n_obs)
        if power_of(r.n_obs, min_effect, sd, alpha) >= target_power and r.effect + z_crit * r.se < min_effect:
            powered_null += 1
    winners = [r for r, rj in zip(valid, rej) if rj and r.effect > 0]
    pooled = measure_transfer({r.name: (r.effect, r.se) for r in valid}, min_contexts=1)
    half = Z95 * pooled.pooled_se
    return FamilyFinding(family, len(valid), powered_null, len(winners), sum(1 for r in winners if r.replicated),
                         Interval(pooled.pooled, pooled.pooled - half, pooled.pooled + half, len(valid)), min_effect,
                         pending_cpu_min, future_cpu_min_per_week, horizon_weeks)


# ------------------------------------------------------------------------------------------------ the public entry
def _make_record(job: JobMeasurement, rv: RealisedValue, j: Judgement, now) -> ValueRecord:
    return ValueRecord(0, job.job_id, job.branch_id, job.family, job.problem.value, job.stage.value, job.target,
                       as_date(job.at).isoformat(), as_date(now).isoformat(), j.verdict.value, float(j.net_value), float(job.cost_cpu_min),
                       float(j.cost_value), {k: float(v) for k, v in j.components.items()}, rv.to_dict(), j.reasons, j.worthless_reasons,
                       dataclasses.asdict(job.estimate) if job.estimate is not None else None, job.code_hash, job.data_hash, job.seed)


def account_job(ledger: ValueLedger, job: JobMeasurement, now, policy: ValuePolicy | None = None,
                claims: "ClaimBook | None" = None) -> ValueRecord:
    """Value one finished job and append it to the ledger. The job's compute is ALWAYS counted (even for an invalid measurement),
    because the CPU-minutes were spent whatever the data looked like. A malformed measurement is recorded as INVALID with zero
    credit; data at/after `now` raises FirewallBreach. A PREVENTIVE verdict opens an auditable claim in `claims`."""
    policy = policy or ValuePolicy()
    bad = policy.validate()
    if bad:
        raise ValueError("invalid ValuePolicy: " + "; ".join(bad))
    rv, problems = measure_job(job, now, policy)
    if problems:
        rv0 = dataclasses.replace(rv, decision_change=None, decision_gain=None, risk_change=None, prediction_change=None,
                                  transfer_change=None, knowledge_redundancy=None, family_unreliability=0.0, prevented_cpu_min=0.0,
                                  measured=())
        j = Judgement(Verdict.INVALID, -policy.price_per_cpu_min * job.cost_cpu_min, {}, policy.price_per_cpu_min * job.cost_cpu_min,
                      tuple(problems), (), (), (), ())
        return ledger.add(_make_record(job, rv0, j, now))
    j = judge(rv, job.stage, job.problem, job.powered_null, policy)
    rec = ledger.add(_make_record(job, rv, j, now))
    if claims is not None and j.verdict is Verdict.PREVENTIVE and job.family_finding is not None:
        claims.open_from(rec, job.family_finding.min_effect)
    return rec


def step(ledger: ValueLedger, jobs: Iterable[JobMeasurement], now, policy: ValuePolicy | None = None,
         claims: "ClaimBook | None" = None) -> dict:
    """Account a batch in (finish date, job_id) order, so the result does not depend on arrival order. Already-valued jobs are
    skipped and reported; a job whose data reaches `now` aborts the batch with FirewallBreach (fail closed)."""
    done, dup = [], []
    for job in sorted(jobs, key=lambda j: (as_date(j.at), j.job_id)):
        if ledger.get(job.job_id) is not None:
            dup.append(job.job_id)
            continue
        done.append(account_job(ledger, job, now, policy, claims))
    return {"accounted": [r.job_id for r in done], "duplicates": dup,
            "verdicts": {v.value: sum(1 for r in done if r.verdict == v.value) for v in Verdict if any(r.verdict == v.value for r in done)},
            "net_value": float(sum(r.net_value for r in done))}


# ------------------------------------------------------------------------------------------------ preventive claims stay auditable
class ClaimStatus(str, enum.Enum):
    OPEN = "OPEN"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"


@dataclasses.dataclass(frozen=True)
class Claim:
    claim_id: str
    job_id: str
    family: str
    made_at: str
    strength: float
    credited_value: float
    min_effect: float
    status: ClaimStatus = ClaimStatus.OPEN
    audits: tuple[tuple[float, float], ...] = ()


@dataclasses.dataclass(frozen=True)
class ClaimUpdate:
    claim: Claim
    correction: ValueRecord | None
    revive_family: bool
    note: str


class ClaimBook:
    """'This family is unreliable' is a claim about the future and can be wrong. It is credited at once (it prevents compute) but
    kept OPEN: later results from the same family (audit probes and revival tests run anyway) either CONFIRM it or REFUTE it, and a
    refutation appends a negative CORRECTION record and asks for the family to be revived."""

    def __init__(self, confirm_after: int = 6, alpha: float = 0.05):
        if confirm_after < 2 or not 0 < alpha < 0.5:
            raise ValueError("confirm_after >= 2 and alpha in (0, 0.5)")
        self.confirm_after, self.alpha = confirm_after, alpha
        self.claims: dict[str, Claim] = {}

    def open_from(self, rec: ValueRecord, min_effect: float) -> Claim:
        cid = "CL" + stable_hash([rec.job_id, rec.family], 10)
        c = Claim(cid, rec.job_id, rec.family, rec.accounted_at, float(rec.value.get("family_unreliability") or 0.0),
                  float(rec.components.get("prevented_compute", 0.0)), min_effect)
        self.claims[cid] = c
        return c

    def open_claims(self, family: str | None = None) -> list[Claim]:
        return sorted((c for c in self.claims.values() if c.status is ClaimStatus.OPEN and (family is None or c.family == family)),
                      key=lambda c: c.claim_id)

    def audit(self, claim_id: str, results: Sequence[tuple[float, float]], now, ledger: ValueLedger) -> ClaimUpdate:
        """Fold new (effect, se) results of the family into the claim. REFUTED by any effect that is both practically useful and
        significant after correcting for the number of audit looks; CONFIRMED only after `confirm_after` looks that all rule
        the useful effect out; otherwise still OPEN (more evidence needed either way)."""
        from scipy.stats import norm
        c = self.claims[claim_id]
        if c.status is not ClaimStatus.OPEN:
            return ClaimUpdate(c, None, False, f"claim already {c.status.value}")
        audits = c.audits + tuple((float(e), float(s)) for e, s in results if math.isfinite(e) and math.isfinite(s) and s > 0)
        looks = len(audits)
        zc = float(norm.ppf(1 - self.alpha / (2 * max(looks, 1))))
        refuting = [a for a in audits if a[0] >= c.min_effect and a[0] / a[1] >= zc]
        if refuting:
            corr = ValueRecord(0, c.job_id + "#correction", "", c.family, Problem.RESEARCH_PROCESS.value, Stage.INTEGRATION.value,
                               ResearchTarget.UNKNOWN_AREA.value, as_date(now).isoformat(), as_date(now).isoformat(),
                               Verdict.CORRECTION.value, -c.credited_value, 0.0, 0.0, {"prevented_compute": -c.credited_value},
                               {"refuted_by_effect": refuting[0][0], "refuted_by_se": refuting[0][1], "looks": looks},
                               (f"claim {c.claim_id} refuted: a later result in the family is real",), (), None, "", "", 0)
            rec = ledger.add(corr)
            self.claims[claim_id] = dataclasses.replace(c, status=ClaimStatus.REFUTED, audits=audits)
            return ClaimUpdate(self.claims[claim_id], rec, True, "refuted: revive the family")
        if looks >= self.confirm_after and all(e + 1.645 * s < c.min_effect for e, s in audits):
            self.claims[claim_id] = dataclasses.replace(c, status=ClaimStatus.CONFIRMED, audits=audits)
            return ClaimUpdate(self.claims[claim_id], None, False, f"confirmed by {looks} independent looks")
        self.claims[claim_id] = dataclasses.replace(c, audits=audits)
        return ClaimUpdate(self.claims[claim_id], None, False, f"still open after {looks} looks")

    def to_dict(self) -> dict:
        return {k: {**dataclasses.asdict(v), "status": v.status.value} for k, v in sorted(self.claims.items())}


# ------------------------------------------------------------------------------------------------ learning from the accounts
_FIELD_REALISED = ("information_gain", "decision_value", "transfer_potential", "loss_reduction_value")


def realised_fields(rec: ValueRecord, policy: ValuePolicy) -> dict[str, float]:
    """The realised counterpart of each pre-job estimate field, scaled to [0, 1] so estimates and outcomes are comparable."""
    v = rec.value
    bits = min(float(v.get("information_gain") or 0.0), policy.information_bit_cap) / policy.information_bit_cap
    dec = sum(max(0.0, rec.components.get(k, 0.0)) for k in ("decision", "prediction"))
    risk = v.get("risk_change") or {}
    return {"information_gain": min(1.0, bits), "decision_value": min(1.0, dec),
            "transfer_potential": min(1.0, max(0.0, float(v.get("transfer_change") or 0.0))),
            "loss_reduction_value": min(1.0, max(0.0, float(risk.get("est", 0.0)) / policy.risk_scale))}


class EstimateCalibrator:
    """Learns how wrong the pre-job value estimates were. For each estimate field the factor is (mean realised + s) / (mean
    estimated + s) (s stabilises small means), computed per family where there are enough jobs and globally otherwise, clipped to
    [0.25, 4]. A family whose estimates always exceed what it delivers gets its next estimates scaled down."""

    def __init__(self, ledger: ValueLedger, as_of, policy: ValuePolicy | None = None, min_n: int = 5, stab: float = 0.05):
        self.policy = policy or ValuePolicy()
        self.min_n, self.stab = min_n, stab
        self.rows: list[tuple[str, dict[str, float], dict[str, float]]] = []
        for r in ledger.records(as_of):
            if r.estimate is None or r.verdict in (Verdict.CORRECTION.value, Verdict.INVALID.value):
                continue
            est = {k: float(r.estimate[k]) for k in _FIELD_REALISED if r.estimate.get(k) is not None}
            if est:
                self.rows.append((r.family, est, realised_fields(r, self.policy)))

    def factor(self, family: str, field: str) -> float:
        for scope in (lambda f: f == family, lambda f: True):
            pairs = [(e[field], rl[field]) for f, e, rl in self.rows if scope(f) and field in e]
            if len(pairs) >= self.min_n:
                me, mr = (float(np.mean([p[i] for p in pairs])) for i in (0, 1))
                return float(min(4.0, max(0.25, (mr + self.stab) / (me + self.stab))))
        return 1.0

    def corrected(self, family: str, est: ExperimentValue) -> ExperimentValue:
        kw = {f: (getattr(est, f) * self.factor(family, f) if getattr(est, f) is not None else None) for f in _FIELD_REALISED}
        return dataclasses.replace(est, **{k: (min(1.0, v) if v is not None else None) for k, v in kw.items()})

    def bias_report(self) -> dict[str, dict[str, float]]:
        out = {}
        for field in _FIELD_REALISED:
            pairs = [(e[field], rl[field]) for _, e, rl in self.rows if field in e]
            if pairs:
                out[field] = {"n": float(len(pairs)), "mean_estimate": float(np.mean([p[0] for p in pairs])),
                              "mean_realised": float(np.mean([p[1] for p in pairs])),
                              "overestimate": float(np.mean([p[0] - p[1] for p in pairs]))}
        return out


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = _rank(np.asarray(a, dtype=float)), _rank(np.asarray(b, dtype=float))
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def priority_skill(ledger: ValueLedger, as_of, min_n: int = 8) -> dict:
    """Did the research priority ORDER predict which jobs turned out valuable? Spearman between the mean pre-job estimate and the
    realised net value per CPU-minute, plus the lift of the top half over the bottom half. Near zero means the priority function
    is not adding information and should be recalibrated (or that estimates are missing: coverage is reported)."""
    rows = []
    total = 0
    for r in ledger.records(as_of):
        if r.verdict == Verdict.CORRECTION.value:
            continue
        total += 1
        if r.estimate and r.cost_cpu_min > 0:
            vals = [float(r.estimate[k]) for k in _FIELD_REALISED if r.estimate.get(k) is not None]
            if vals:
                rows.append((float(np.mean(vals)), r.net_value / r.cost_cpu_min))
    if len(rows) < min_n:
        return {"n": len(rows), "coverage": len(rows) / total if total else 0.0, "spearman": None, "lift": None}
    rows.sort(key=lambda t: t[0])
    half = len(rows) // 2
    lo, hi = [r[1] for r in rows[:half]], [r[1] for r in rows[-half:]]
    return {"n": len(rows), "coverage": len(rows) / total, "spearman": _spearman([r[0] for r in rows], [r[1] for r in rows]),
            "lift": float(np.mean(hi) - np.mean(lo))}


def recent_value_rate(ledger: ValueLedger, family: str, as_of, window: int = 5) -> dict:
    """Net value per CPU-minute over the family's last `window` jobs versus its lifetime: falling returns show up here before a
    family has had a long barren run. The waste controller reads this next to its own progress measure."""
    rs = [r for r in ledger.records(as_of) if r.family == family and r.verdict != Verdict.CORRECTION.value]
    if not rs:
        return {"n": 0, "recent": None, "lifetime": None, "ratio": None}
    life = sum(r.net_value for r in rs) / max(sum(r.cost_cpu_min for r in rs), 1e-9)
    tail = rs[-window:]
    rec = sum(r.net_value for r in tail) / max(sum(r.cost_cpu_min for r in tail), 1e-9)
    return {"n": len(rs), "recent": rec, "lifetime": life, "ratio": (rec / life) if abs(life) > 1e-12 else None}


def compare_periods(ledger: ValueLedger, start_a, end_a, start_b, end_b) -> dict:
    """Value and waste in two windows (by finish date), for 'is the research getting more efficient?' (section 36)."""
    def window(lo, hi):
        rs = [r for r in ledger.records() if as_date(lo) <= as_date(r.at) < as_date(hi) and r.verdict != Verdict.CORRECTION.value]
        cpu = sum(r.cost_cpu_min for r in rs)
        return {"jobs": len(rs), "cpu_min": cpu, "net_value": sum(r.net_value for r in rs),
                "waste_share": (sum(r.cost_cpu_min for r in rs if r.verdict == Verdict.WORTHLESS.value) / cpu) if cpu > 0 else 0.0,
                "useful_share": (sum(1 for r in rs if r.verdict in (Verdict.USEFUL.value, Verdict.PREVENTIVE.value)) / len(rs)) if rs else 0.0}
    a, b = window(start_a, end_a), window(start_b, end_b)
    return {"a": a, "b": b, "delta_waste_share": b["waste_share"] - a["waste_share"], "delta_useful_share": b["useful_share"] - a["useful_share"]}


# ------------------------------------------------------------------------------------------------ reporting and self-check
def report(ledger: ValueLedger, as_of) -> str:
    tot = ledger.totals(as_of)
    lines = [f"Experiment value accounting as of {as_date(as_of).isoformat()}  (IMPLEMENTED - NOT VALIDATED)",
             f"jobs={int(tot['jobs'])} cpu_min={tot['cpu_min']:.1f} net_value={tot['net_value']:.4f} waste_share={tot['waste_share']:.1%}"]
    for fam, s in ledger.family_stats(as_of).items():
        cnt = ", ".join(f"{k}={v}" for k, v in sorted(s.counts.items()))
        lines.append(f"  {fam:24s} jobs={s.jobs:3d} cpu={s.cpu_min:8.1f} net={s.net_value:9.4f} rate={s.value_per_cpu_min:8.5f}/cpu-min  [{cnt}]")
    reasons = ledger.worthless_reason_counts(as_of)
    if reasons:
        lines.append("why jobs were worthless: " + "; ".join(f"{k} x{v}" for k, v in list(reasons.items())[:5]))
    return "\n".join(lines)


def self_check(seed: int = 0) -> dict[str, bool]:
    """Planted-world sanity checks of the judge itself (no data read): each case has a known correct verdict. If the accountant
    cannot pass these it must not be trusted to value real jobs."""
    rng = np.random.default_rng(seed)
    policy = ValuePolicy()
    n = 300
    base = rng.normal(0.0, 0.03, n)
    out: dict[str, bool] = {}

    def job(jid: str, **kw) -> JobMeasurement:
        return JobMeasurement(jid, "B", "fam", Problem.VOLATILITY, kw.pop("stage", Stage.CROSS_YEAR), "2010-01-05", "2010-01-04",
                              kw.pop("cost", 10.0), **kw)
    led = ValueLedger()
    r = account_job(led, job("safer", pnl_before=base, pnl_after=np.clip(base, -0.02, None), replications=1), "2010-02-01", policy)
    out["risk_cut_is_useful"] = r.verdict == Verdict.USEFUL.value
    r = account_job(led, job("noise", pnl_before=base, pnl_after=base + rng.normal(0, 1e-5, n), replications=1), "2010-02-01", policy)
    out["noise_is_worthless"] = r.verdict == Verdict.WORTHLESS.value
    y = (rng.random(n) < 0.5).astype(float)
    p0 = np.full(n, 0.5)
    r = account_job(led, job("tiny", pred_before=p0, pred_after=p0 - 0.0001 * (y - 0.5), target_values=y, params_added=20, replications=0,
                             stage=Stage.CHEAP_SCREEN), "2010-02-01", policy)
    out["tiny_gain_with_complexity_not_useful"] = r.verdict != Verdict.USEFUL.value
    r = account_job(led, job("short", pnl_before=base[:6], pnl_after=np.clip(base[:6], -0.02, None), replications=1), "2010-02-01", policy)
    out["tiny_sample_is_inconclusive"] = r.verdict in (Verdict.INCONCLUSIVE.value, Verdict.USEFUL.value) and r.verdict != Verdict.WORTHLESS.value
    res = [HypothesisResult(f"h{i}", float(rng.normal(0, 0.0004)), 0.0005, 40000) for i in range(20)]
    ff = build_family_finding("famX", res, 0.003, pending_cpu_min=300.0, future_cpu_min_per_week=20.0)
    r = account_job(led, job("family", family_finding=ff, prior_se=0.01, post_se=0.004), "2010-02-01", policy)
    out["unreliable_family_is_preventive"] = r.verdict == Verdict.PREVENTIVE.value and r.net_value > 0
    return out


# ------------------------------------------------------------------------------------------------ robustness of verdicts
def verdict_sensitivity(job: JobMeasurement, now, policy: ValuePolicy | None = None, scale: float = 0.3) -> dict:
    """Is the verdict a knife-edge? Re-judge the SAME measurement with every scale/materiality threshold moved up and down by
    `scale` (one at a time) and report which perturbations flip it. A verdict that flips when a convention moves 30% is not a
    finding, and USEFUL on such a basis must be treated as a lead. The measurement is done once; only the judgement is repeated."""
    policy = policy or ValuePolicy()
    rv, problems = measure_job(job, now, policy)
    if problems:
        return {"base": Verdict.INVALID.value, "stable": True, "flips": {}, "problems": problems}
    base = judge(rv, job.stage, job.problem, job.powered_null, policy).verdict
    flips: dict[str, list[str]] = {}
    knobs = ("decision_scale", "risk_scale", "prediction_scale", "material_decision_fraction", "material_risk", "material_prediction",
             "material_transfer", "price_per_cpu_min", "complexity_price", "unreplicated_discount", "preventive_strength")
    for knob in knobs:
        for mult in (1.0 - scale, 1.0 + scale):
            val = getattr(policy, knob) * mult
            if knob in ("unreplicated_discount", "preventive_strength"):
                val = min(1.0, val)
            v = judge(rv, job.stage, job.problem, job.powered_null, dataclasses.replace(policy, **{knob: val})).verdict
            if v is not base:
                flips.setdefault(knob, []).append(f"x{mult:.2f}->{v.value}")
    return {"base": base.value, "stable": not flips, "flips": flips, "problems": []}


# ------------------------------------------------------------------------------------------------ acting on preventive findings
@dataclasses.dataclass(frozen=True)
class PendingExperiment:
    exp_id: str
    family: str
    cpu_min: float
    started: bool = False


def cancellations(claims: ClaimBook, pending: Sequence[PendingExperiment], min_strength: float = 0.7) -> dict:
    """Turn open or confirmed PREVENTIVE claims into action: which not-yet-started experiments in an unreliable family should be
    cancelled, and the CPU-minutes that saves. Started experiments are never touched (CONTEXT rule 11: only their owner stops
    them). Refuted claims cancel nothing."""
    live = {c.family for c in claims.claims.values() if c.status is not ClaimStatus.REFUTED and c.strength >= min_strength}
    cancel = sorted((p for p in pending if p.family in live and not p.started), key=lambda p: p.exp_id)
    return {"cancel": [p.exp_id for p in cancel], "cpu_min_saved": float(sum(p.cpu_min for p in cancel)), "families": sorted(live),
            "kept": [p.exp_id for p in pending if p.family in live and p.started]}


def unreliable_families(claims: ClaimBook, min_strength: float = 0.7) -> list[str]:
    """Families the research loop should not open new work in (open or confirmed claims); refuted families are back in play."""
    return sorted({c.family for c in claims.claims.values() if c.status is not ClaimStatus.REFUTED and c.strength >= min_strength})


# ------------------------------------------------------------------------------------------------ ledger audit
def audit_ledger(ledger: ValueLedger, policy: ValuePolicy | None = None) -> list[str]:
    """Invariants a healthy ledger always satisfies (empty = clean): the chain is intact; compute is priced consistently; every
    verdict is supported by its own components; nothing is credited without measurement."""
    policy = policy or ValuePolicy()
    errs = ledger.verify_chain()
    for r in ledger.records():
        if r.verdict == Verdict.CORRECTION.value:
            if r.net_value > 0:
                errs.append(f"{r.job_id}: a correction cannot add value")
            continue
        if abs(r.cost_value - policy.price_per_cpu_min * r.cost_cpu_min) > 1e-9 * max(1.0, r.cost_cpu_min):
            errs.append(f"{r.job_id}: compute priced inconsistently")
        if r.verdict == Verdict.USEFUL.value:
            pos = [k for k, v in r.components.items() if k in ("decision", "risk", "prediction", "transfer") and v > 0]
            if not pos or r.net_value <= 0:
                errs.append(f"{r.job_id}: USEFUL without a positive material component and positive net value")
        if r.verdict == Verdict.PREVENTIVE.value and r.components.get("prevented_compute", 0.0) <= 0:
            errs.append(f"{r.job_id}: PREVENTIVE without prevented compute")
        if r.verdict == Verdict.INVALID.value and r.net_value > 0:
            errs.append(f"{r.job_id}: INVALID job credited with value")
        if r.verdict in (Verdict.WORTHLESS.value, Verdict.HARMFUL_IF_ADOPTED.value) and r.net_value > policy.w_information_per_bit * policy.information_bit_cap:
            errs.append(f"{r.job_id}: {r.verdict} with more value than information alone can supply")
        if not r.value.get("measured") and r.verdict in (Verdict.USEFUL.value, Verdict.WORTHLESS.value):
            errs.append(f"{r.job_id}: {r.verdict} on an unmeasured job")
    return errs


def cost_effectiveness(ledger: ValueLedger, as_of) -> dict[str, dict[str, float]]:
    """Net value per CPU-minute by ladder rung and by problem: are the expensive rungs earning what they cost?"""
    out: dict[str, dict[str, dict[str, float]]] = {"stage": {}, "problem": {}}
    for r in ledger.records(as_of):
        if r.verdict == Verdict.CORRECTION.value:
            continue
        for view, key in (("stage", r.stage), ("problem", r.problem)):
            d = out[view].setdefault(key, {"cpu_min": 0.0, "net_value": 0.0, "jobs": 0.0})
            d["cpu_min"] += r.cost_cpu_min
            d["net_value"] += r.net_value
            d["jobs"] += 1.0
    flat: dict[str, dict[str, float]] = {}
    for view, table in out.items():
        for key, d in sorted(table.items()):
            flat[f"{view}:{key}"] = {**d, "rate": d["net_value"] / d["cpu_min"] if d["cpu_min"] > 0 else 0.0}
    return flat


def explain_record(rec: ValueRecord) -> str:
    """One paragraph a person can read: the verdict, why, what each component contributed and what would have changed it."""
    parts = [f"{rec.job_id} [{rec.family}/{rec.stage}] -> {rec.verdict}, net {rec.net_value:+.4f} for {rec.cost_cpu_min:.1f} cpu-min"]
    parts += ["  because: " + r for r in rec.reasons]
    parts += ["  not counted: " + w for w in rec.worthless_reasons]
    comp = ", ".join(f"{k}={v:+.4f}" for k, v in sorted(rec.components.items()) if abs(v) > 1e-9)
    if comp:
        parts.append("  components: " + comp)
    unmeasured = [k for k in ("decision_change", "risk_change", "prediction_change", "transfer_change", "knowledge_redundancy",
                              "information_gain") if k not in (rec.value.get("measured") or [])]
    if unmeasured:
        parts.append("  unmeasured (never read as zero): " + ", ".join(unmeasured))
    return "\n".join(parts)


# ------------------------------------------------------------------------------------------------ reconciliation and portfolio views
def reconcile_with_manager(ledger: ValueLedger, branch_spend: Mapping[str, float], tol: float = 1e-6) -> dict:
    """Compute must be accounted exactly once. Compare the CPU-minutes the value ledger holds per branch with what the compute
    manager says each branch spent: `unaccounted` = spent but never valued (jobs that escaped the accountant), `overcounted` =
    valued for more than was spent (a double-counted or inflated job). Both should be empty."""
    per: dict[str, float] = {}
    for r in ledger.records():
        if r.verdict != Verdict.CORRECTION.value:
            per[r.branch_id] = per.get(r.branch_id, 0.0) + r.cost_cpu_min
    unacc = {b: c - per.get(b, 0.0) for b, c in sorted(branch_spend.items()) if c - per.get(b, 0.0) > tol * max(1.0, c)}
    over = {b: per[b] - branch_spend.get(b, 0.0) for b in sorted(per) if per[b] - branch_spend.get(b, 0.0) > tol * max(1.0, per[b])}
    return {"unaccounted": unacc, "overcounted": over, "ok": not unacc and not over}


def value_by_target(ledger: ValueLedger, as_of) -> dict[str, dict[str, float]]:
    """Net value and compute per research target (section 35 places effort can go), so the target allocator can be compared with
    what each target actually returned."""
    out: dict[str, dict[str, float]] = {}
    for r in ledger.records(as_of):
        if r.verdict == Verdict.CORRECTION.value:
            continue
        d = out.setdefault(r.target, {"jobs": 0.0, "cpu_min": 0.0, "net_value": 0.0, "useful": 0.0})
        d["jobs"] += 1
        d["cpu_min"] += r.cost_cpu_min
        d["net_value"] += r.net_value
        d["useful"] += float(r.verdict in (Verdict.USEFUL.value, Verdict.PREVENTIVE.value))
    for d in out.values():
        d["rate"] = d["net_value"] / d["cpu_min"] if d["cpu_min"] > 0 else 0.0
    return dict(sorted(out.items()))


def bits_per_cpu_min(ledger: ValueLedger, as_of, family: str | None = None) -> float | None:
    """Information gained per CPU-minute (None when no job measured information). Only measured jobs count in the numerator AND
    the denominator, so unmeasured jobs cannot dilute or inflate the rate."""
    bits = cpu = 0.0
    for r in ledger.records(as_of):
        if r.verdict == Verdict.CORRECTION.value or (family is not None and r.family != family):
            continue
        b = r.value.get("information_gain")
        if b is not None:
            bits += float(b)
            cpu += r.cost_cpu_min
    return bits / cpu if cpu > 0 else None


def stopping_advice(ledger: ValueLedger, family: str, as_of, k: int = 4, eps: float = 0.0005) -> dict:
    """Should the family's line of work stop? Uses research_policy.marginal_return_verdict on the per-job net value per CPU-minute
    of the family, and adds the drought length. The waste controller decides; this is the accountant's evidence for it."""
    from engine.learning.research_policy import marginal_return_verdict
    rs = [r for r in ledger.records(as_of) if r.family == family and r.verdict != Verdict.CORRECTION.value and r.cost_cpu_min > 0]
    rates = [max(0.0, r.net_value) / r.cost_cpu_min for r in rs]
    mv = marginal_return_verdict(rates, k=k, eps=eps)
    return {"family": family, "jobs": len(rs), "marginal": mv, "drought": ledger.drought(family, as_of),
            "recommend_stop": mv["verdict"] == "STOP" and ledger.drought(family, as_of) >= k}


def concentration(ledger: ValueLedger, as_of, top: int = 3) -> dict:
    """How concentrated is the value? If a few jobs carry all the net value, the vector is fragile (one refuted claim could erase
    it). Reports the share of positive value held by the top jobs and by the top family."""
    rs = [r for r in ledger.records(as_of) if r.net_value > 0]
    tot = sum(r.net_value for r in rs)
    if tot <= 0:
        return {"positive_value": 0.0, "top_jobs_share": 0.0, "top_family_share": 0.0}
    top_jobs = sum(sorted((r.net_value for r in rs), reverse=True)[:top])
    fam: dict[str, float] = {}
    for r in rs:
        fam[r.family] = fam.get(r.family, 0.0) + r.net_value
    return {"positive_value": tot, "top_jobs_share": top_jobs / tot, "top_family_share": max(fam.values()) / tot}


def job_from_run(run: Mapping[str, Any], branch_id: str, family: str, problem: Problem, stage: Stage, data_through: str,
                 job_id: str, **measurements) -> JobMeasurement:
    """Wrap a compute-manager run-log entry (stage, at, cost, ...) as a JobMeasurement, so every ladder run reaches the accountant
    through one door; measurement arrays are passed by keyword and anything not supplied stays unmeasured."""
    measurements.setdefault("params_added", int(run.get("complexity", 0) or 0))
    return JobMeasurement(job_id, branch_id, family, problem, stage, str(run["at"]), data_through, float(run.get("cost", 0.0)), **measurements)


# ------------------------------------------------------------------------------------------------ judge operating characteristics
@dataclasses.dataclass(frozen=True)
class JudgeCharacteristics:
    n: int
    false_useful_rate: float           # share of no-effect jobs judged USEFUL (should be near 0)
    false_harmful_rate: float          # share of no-effect jobs judged HARMFUL_IF_ADOPTED
    detection_rate: float              # share of real-effect jobs judged USEFUL
    inconclusive_rate_real: float


def judge_characteristics(seed: int, n: int = 40, periods: int = 200, tail_cap: float = 0.02, policy: ValuePolicy | None = None) -> JudgeCharacteristics:
    """Measure the accountant itself on planted jobs of known truth: `n` jobs whose 'improved' P&L is the baseline plus pure noise
    (no effect) and `n` whose improved P&L really clips the loss tail. A judge that calls noise USEFUL is broken however clean its
    real-data output looks; one that misses the real tail cut is too timid. Deterministic in `seed`."""
    policy = policy or ValuePolicy(n_boot=200)
    rng = np.random.default_rng(seed)
    false_u = false_h = hit = inc = 0
    for i in range(n):
        base = rng.normal(0.0, 0.03, periods)
        led = ValueLedger()
        null = account_job(led, JobMeasurement(f"n{i}", "B", "f", Problem.VOLATILITY, Stage.CROSS_YEAR, "2010-01-05", "2010-01-04", 5.0,
                                               pnl_before=base, pnl_after=base + rng.normal(0, 0.002, periods), replications=1), "2010-02-01", policy)
        real = account_job(led, JobMeasurement(f"r{i}", "B", "f", Problem.VOLATILITY, Stage.CROSS_YEAR, "2010-01-05", "2010-01-04", 5.0,
                                               pnl_before=base, pnl_after=np.clip(base, -tail_cap, None), replications=1), "2010-02-01", policy)
        false_u += null.verdict == Verdict.USEFUL.value
        false_h += null.verdict == Verdict.HARMFUL_IF_ADOPTED.value
        hit += real.verdict == Verdict.USEFUL.value
        inc += real.verdict == Verdict.INCONCLUSIVE.value
    return JudgeCharacteristics(n, false_u / n, false_h / n, hit / n, inc / n)


def leave_one_out_value(ledger: ValueLedger, as_of) -> dict[str, float]:
    """Total net value with each job removed: the drop is that job's marginal contribution. A total that collapses when one job
    is removed is fragile (see `concentration`)."""
    rs = [r for r in ledger.records(as_of)]
    tot = sum(r.net_value for r in rs)
    return {r.job_id: tot - r.net_value for r in rs}


def value_uncertainty(ledger: ValueLedger, as_of, seed: int = 0, n_boot: int = 500) -> Interval:
    """Bootstrap interval for the ledger's mean net value per CPU-minute (resampling jobs). A wide interval means the portfolio
    verdict about 'is research paying for itself' is itself unknown."""
    rs = [r for r in ledger.records(as_of) if r.verdict != Verdict.CORRECTION.value and r.cost_cpu_min > 0]
    if not rs:
        return Interval(0.0, 0.0, 0.0, 0)
    v = np.array([r.net_value for r in rs])
    c = np.array([r.cost_cpu_min for r in rs])
    return bootstrap_interval(lambda a, b: a.sum() / max(b.sum(), 1e-9), [v, c], np.random.default_rng(seed), n_boot, 1)


# ------------------------------------------------------------------------------------------------ tables
def verdict_matrix(ledger: ValueLedger, as_of) -> dict[str, dict[str, int]]:
    """Family x verdict counts (corrections excluded): where each kind of outcome comes from."""
    out: dict[str, dict[str, int]] = {}
    for r in ledger.records(as_of):
        if r.verdict != Verdict.CORRECTION.value:
            row = out.setdefault(r.family, {})
            row[r.verdict] = row.get(r.verdict, 0) + 1
    return {f: dict(sorted(v.items())) for f, v in sorted(out.items())}


def top_wasters(ledger: ValueLedger, as_of, n: int = 5) -> list[tuple[str, float, float]]:
    """(family, cpu_min spent on WORTHLESS/INVALID jobs, share of that family's compute): where waste is concentrated."""
    waste: dict[str, float] = {}
    total: dict[str, float] = {}
    for r in ledger.records(as_of):
        if r.verdict == Verdict.CORRECTION.value:
            continue
        total[r.family] = total.get(r.family, 0.0) + r.cost_cpu_min
        if r.verdict in (Verdict.WORTHLESS.value, Verdict.INVALID.value):
            waste[r.family] = waste.get(r.family, 0.0) + r.cost_cpu_min
    rows = [(f, w, w / total[f]) for f, w in waste.items() if total[f] > 0]
    return sorted(rows, key=lambda t: (-t[1], t[0]))[:n]


def ledger_diff(a: ValueLedger, b: ValueLedger) -> dict:
    """Jobs valued in one ledger but not the other, and jobs whose verdict differs (the same job judged under two policies or two
    code versions). Verdict drift under an unchanged policy means the accountant is not deterministic, which is a defect."""
    ra, rb = {r.job_id: r for r in a.records()}, {r.job_id: r for r in b.records()}
    changed = {j: (ra[j].verdict, rb[j].verdict) for j in sorted(set(ra) & set(rb)) if ra[j].verdict != rb[j].verdict}
    return {"only_a": sorted(set(ra) - set(rb)), "only_b": sorted(set(rb) - set(ra)), "changed": changed,
            "value_delta": sum(r.net_value for r in rb.values()) - sum(r.net_value for r in ra.values())}
