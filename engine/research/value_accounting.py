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


# @@APPEND@@
