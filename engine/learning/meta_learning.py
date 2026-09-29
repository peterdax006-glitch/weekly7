"""Meta-learning (contract C62 section 37; checklist G12, C16; Bible canon C58-C62).

WEEKLY7 learns about its own learning process. Eight questions, each answered from resolved records only:

  1. which discoveries survive out of sample?            (SurvivalModel: ridge logistic on discovery-time features)
  2. which pattern families overfit?                     (empirical-Bayes shrunk survival + IS->OOS effect ratio)
  3. which contexts transfer?                            (Wilson-bounded transfer rates per context dimension)
  4. which memory types decay rapidly?                   (log-linear decay fit and half-life with bootstrap interval)
  5. which experiments produce useful knowledge?         (yield rate and bits per compute-minute per experiment kind)
  6. which learners produce fake improvement?            (claimed vs independently verified improvement)
  7. which failure explanations tend to be correct?      (precision per predicted cause, confusion matrix, calibration)
  8. which feature families repeatedly fail ablation?    (failure counts and current streaks)

Meta-learning must itself be evaluated OUT OF SAMPLE (section 37): `walk_forward` trains on labels that had RESOLVED before a
cutoff, predicts records discovered after it, and compares against a base-rate baseline with a paired bootstrap. The result
label stays IMPLEMENTED - NOT VALIDATED unless the caller certifies the records as real (synthetic worlds never validate).
A single feature that perfectly predicts the label is treated as a leak, not a discovery.

Output for the rest of the brain: research_policy.MetaAdvice (overfit penalties, target yields, explanation precision).
Times: a label is only usable at `now` if resolved_at < now; features are frozen at discovered_at. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .core import (FailureCause, FirewallBreach, TemporalClass, ValidationLabel, canonical_json, current_code_hash,
                   stable_hash)
from .experiment_memory import to_ts
from .research_policy import MetaAdvice

LABEL = ValidationLabel.NOT_VALIDATED.value
Z95 = 1.959963984540054


# ------------------------------------------------------------------------------------------------ statistics

def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def wilson(successes: int, n: int, z: float = Z95) -> tuple:
    """Wilson score interval (lo, hi) for a proportion. n = 0 -> (0, 1): total ignorance, not a point estimate."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def fit_beta_prior(successes: Sequence[int], trials: Sequence[int], min_strength: float = 2.0, max_strength: float = 200.0) -> tuple:
    """Empirical-Bayes Beta(a, b) for group rates by method of moments on the between-group variance in excess of pure
    binomial sampling noise. Falls back to a weak prior at the pooled rate when groups are too few / too alike to say."""
    s = np.asarray(successes, float)
    n = np.asarray(trials, float)
    keep = n > 0
    s, n = s[keep], n[keep]
    if s.sum() + (n - s).sum() == 0:
        return 1.0, 1.0
    pooled = float(s.sum() / n.sum())
    pooled = min(max(pooled, 0.02), 0.98)
    if len(s) < 3:
        return pooled * min_strength * 2, (1 - pooled) * min_strength * 2
    rates = s / n
    w = n / n.sum()
    mean = float((w * rates).sum())
    var_between = float((w * (rates - mean) ** 2).sum())
    var_sampling = float((w * mean * (1 - mean) / n).sum())
    excess = var_between - var_sampling
    mean = min(max(mean, 0.02), 0.98)
    if excess <= 1e-6:
        strength = max_strength                                 # groups look identical: shrink hard to the pool
    else:
        strength = mean * (1 - mean) / excess - 1.0
    strength = min(max(strength, min_strength), max_strength)
    return mean * strength, (1 - mean) * strength


def shrunk_rate(successes: int, n: int, prior: tuple) -> float:
    return (successes + prior[0]) / (n + prior[0] + prior[1])


def auc(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float | None:
    """Mann-Whitney AUC with tie handling; None if one class is absent."""
    y = np.asarray(labels, int)
    s = np.asarray(scores, float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def brier(p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def log_loss(p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, eps: float = 1e-6) -> float:
    pc = np.clip(np.asarray(p, float), eps, 1 - eps)
    yf = np.asarray(y, float)
    return float(-np.mean(yf * np.log(pc) + (1 - yf) * np.log(1 - pc))) if len(yf) else float("nan")


def paired_bootstrap(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, seed: int, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Mean of (a - b) with a percentile interval, resampling pairs. Deterministic in `seed`."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 2:
        return {"n": n, "diff": float(d.mean()) if n else float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": n, "diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60, tol: float = 1e-8) -> np.ndarray:
    """Ridge logistic regression by Newton/IRLS. Returns weights with the intercept LAST. The intercept is not penalised."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    pen = np.full(d + 1, l2)
    pen[-1] = 0.0
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        W = p * (1 - p) + 1e-9
        g = Xb.T @ (y - p) - pen * w
        H = Xb.T @ (Xb * W[:, None]) + np.diag(pen + 1e-9)
        step = np.linalg.solve(H, g)
        w = w + step
        if float(np.linalg.norm(step)) < tol:
            break
    return w


# ------------------------------------------------------------------------------------------------ record types

@dataclass(frozen=True)
class DiscoveryRecord:
    item_id: str
    family: str
    discovered_at: str
    features: Mapping                              # frozen at discovery: t_disc, log_n, effect, n_conditions, p_real, n_tests, ...
    survived_oos: bool | None = None
    resolved_at: str = ""                          # when the OOS verdict became known; "" = unresolved
    is_effect: float | None = None
    oos_effect: float | None = None
    temporal_class: str = TemporalClass.UNKNOWN.value

    def check(self) -> list:
        errs = []
        if not self.item_id or not self.family:
            errs.append("discovery needs item_id and family")
        if self.survived_oos is not None and not self.resolved_at:
            errs.append(f"{self.item_id}: label without resolved_at")
        if self.resolved_at and to_ts(self.resolved_at) <= to_ts(self.discovered_at):
            errs.append(f"{self.item_id}: resolved_at not after discovered_at")
        for k, v in self.features.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                errs.append(f"{self.item_id}: feature {k}={v!r} not a finite number")
        return errs


@dataclass(frozen=True)
class TransferRecord:
    item_id: str
    context_dim: str
    family: str
    transferred: bool
    resolved_at: str


@dataclass(frozen=True)
class DecayRecord:
    item_id: str
    memory_type: str
    age_days: float
    reliability_ratio: float                       # current reliability / reliability at creation, > 0
    resolved_at: str


@dataclass(frozen=True)
class YieldRecord:
    experiment_id: str
    kind: str                                      # experiment kind or research target
    useful: bool
    gain_bits: float
    cost_minutes: float
    resolved_at: str


@dataclass(frozen=True)
class LearnerRecord:
    learner: str
    claimed_gain: float                            # what the learner's own evaluation reported
    verified: bool                                 # did independent, past-only, blind evaluation confirm it?
    resolved_at: str
    verified_gain: float | None = None


@dataclass(frozen=True)
class ExplanationRecord:
    item_id: str
    predicted_cause: str                           # FailureCause value the system asserted
    confidence: float
    verified_cause: str                            # FailureCause value later established; UNKNOWN allowed
    resolved_at: str

    def correct(self) -> bool:
        return self.predicted_cause == self.verified_cause and self.verified_cause != FailureCause.UNKNOWN.value


@dataclass(frozen=True)
class AblationRecord:
    feature_family: str
    failed: bool                                   # ablation showed the family adds nothing (removing it did not hurt)
    run_id: str
    resolved_at: str


@dataclass
class MetaStore:
    """Everything meta-learning learns from. Append-only in spirit: `as_of` returns a filtered COPY, never a mutation."""
    discoveries: list = field(default_factory=list)
    transfers: list = field(default_factory=list)
    decays: list = field(default_factory=list)
    yields: list = field(default_factory=list)
    learners: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    ablations: list = field(default_factory=list)

    KINDS = ("discoveries", "transfers", "decays", "yields", "learners", "explanations", "ablations")

    def add(self, rec) -> None:
        name = {DiscoveryRecord: "discoveries", TransferRecord: "transfers", DecayRecord: "decays", YieldRecord: "yields",
                LearnerRecord: "learners", ExplanationRecord: "explanations", AblationRecord: "ablations"}.get(type(rec))
        if name is None:
            raise TypeError(f"unknown meta record type {type(rec).__name__}")
        if isinstance(rec, DiscoveryRecord):
            errs = rec.check()
            if errs:
                raise ValueError("; ".join(errs))
        if isinstance(rec, DecayRecord) and (rec.reliability_ratio <= 0 or rec.age_days < 0):
            raise ValueError("decay record needs ratio > 0 and age >= 0")
        getattr(self, name).append(rec)

    def extend(self, recs: Iterable) -> None:
        for r in recs:
            self.add(r)

    @staticmethod
    def _time(rec) -> str:
        return rec.resolved_at

    def as_of(self, now) -> "MetaStore":
        """Records whose label was KNOWN strictly before `now`. Unresolved discoveries are kept out: they have no label."""
        cut = to_ts(now)
        out = MetaStore()
        for k in self.KINDS:
            keep = [r for r in getattr(self, k) if r.resolved_at and to_ts(r.resolved_at) < cut]
            setattr(out, k, keep)
        return out

    def counts(self) -> dict:
        return {k: len(getattr(self, k)) for k in self.KINDS}

    def code_hash(self) -> str:
        return stable_hash({k: [canonical_json(r) for r in getattr(self, k)] for k in self.KINDS}, 12)


# ------------------------------------------------------------------------------------------------ 1. survival model

class SurvivalModel:
    """P(a new discovery survives OOS | features known at discovery), ridge logistic on standardised features."""

    def __init__(self, l2: float = 2.0):
        self.l2 = l2
        self.names: tuple = ()
        self.mu = np.zeros(0)
        self.sd = np.ones(0)
        self.w = np.zeros(1)
        self.base = 0.5
        self.n = 0

    def fit(self, recs: Sequence[DiscoveryRecord]) -> "SurvivalModel":
        labelled = [r for r in recs if r.survived_oos is not None]
        self.n = len(labelled)
        y = np.array([1.0 if r.survived_oos else 0.0 for r in labelled])
        self.base = float(y.mean()) if len(y) else 0.5
        names = sorted({k for r in labelled for k in r.features}) + sorted({"fam=" + r.family for r in labelled})
        self.names = tuple(names)
        if len(labelled) < 8 or not names or y.min() == y.max():
            self.mu, self.sd, self.w = np.zeros(len(names)), np.ones(len(names)), np.zeros(len(names) + 1)
            self.w[-1] = math.log(max(self.base, 0.02) / max(1 - self.base, 0.02))
            return self
        X = self._matrix(labelled, fit=True)
        self.w = fit_logistic(X, y, self.l2)
        return self

    def _matrix(self, recs, fit: bool = False) -> np.ndarray:
        X = np.array([[(1.0 if k == "fam=" + r.family else 0.0) if k.startswith("fam=") else float(r.features.get(k, 0.0))
                       for k in self.names] for r in recs], float).reshape(len(recs), len(self.names))
        if fit:
            self.mu = X.mean(axis=0) if len(X) else np.zeros(len(self.names))
            sd = X.std(axis=0) if len(X) else np.ones(len(self.names))
            self.sd = np.where(sd < 1e-9, 1.0, sd)
        return (X - self.mu) / self.sd

    def predict(self, recs: Sequence[DiscoveryRecord]) -> np.ndarray:
        if not recs:
            return np.zeros(0)
        X = self._matrix(recs)
        return sigmoid(X @ self.w[:-1] + self.w[-1])

    def importance(self) -> list:
        """Standardised coefficients, largest magnitude first: what predicts survival (and its sign)."""
        return sorted(((n, float(w)) for n, w in zip(self.names, self.w[:-1])), key=lambda t: -abs(t[1]))


def leak_audit(recs: Sequence[DiscoveryRecord], threshold: float = 0.97) -> list:
    """Any single discovery-time feature that separates survivors from non-survivors this well is a label leak, not a
    finding (a feature computed AFTER the outcome, or the outcome itself). Returns [(feature, auc)] above threshold."""
    lab = [r for r in recs if r.survived_oos is not None]
    y = [1 if r.survived_oos else 0 for r in lab]
    out = []
    for k in sorted({k for r in lab for k in r.features}):
        a = auc([r.features.get(k, 0.0) for r in lab], y)
        if a is not None and max(a, 1 - a) >= threshold and len(lab) >= 10:
            out.append((k, round(a, 4)))
    return out


# ------------------------------------------------------------------------------------------------ 2-8. reports

@dataclass(frozen=True)
class GroupRate:
    key: str
    n: int
    successes: int
    raw: float
    shrunk: float
    lo: float
    hi: float


def group_rates(pairs: Iterable[tuple], min_n: int = 1) -> tuple:
    """pairs of (key, bool). Returns (dict key -> GroupRate, beta prior). Shrinkage is empirical-Bayes across the groups."""
    tally: dict = {}
    for k, ok in pairs:
        s, n = tally.get(k, (0, 0))
        tally[k] = (s + int(bool(ok)), n + 1)
    prior = fit_beta_prior([v[0] for v in tally.values()], [v[1] for v in tally.values()])
    out = {}
    for k, (s, n) in tally.items():
        if n < min_n:
            continue
        lo, hi = wilson(s, n)
        out[k] = GroupRate(k, n, s, s / n, shrunk_rate(s, n, prior), lo, hi)
    return out, prior


@dataclass(frozen=True)
class FamilyReport:
    rates: Mapping
    prior: tuple
    base_rate: float
    overfit: Mapping                               # family -> P(overfit) in [0,1], only for families with enough n
    effect_ratio: Mapping                          # family -> mean(oos_effect / is_effect) over usable records
    n: int


def family_report(recs: Sequence[DiscoveryRecord], min_n: int = 5, margin: float = 0.05) -> FamilyReport:
    lab = [r for r in recs if r.survived_oos is not None]
    rates, prior = group_rates(((r.family, r.survived_oos) for r in lab), 1)
    base = sum(1 for r in lab if r.survived_oos) / len(lab) if lab else 0.0
    overfit = {}
    for f, g in rates.items():
        if g.n < min_n:
            continue
        overfit[f] = min(1.0, max(0.0, 1.0 - g.shrunk)) if g.shrunk < base - margin else max(0.0, 1.0 - g.shrunk) * 0.5
    ratios: dict = {}
    for r in lab:
        if r.is_effect and r.oos_effect is not None and abs(r.is_effect) > 1e-12:
            ratios.setdefault(r.family, []).append(r.oos_effect / r.is_effect)
    eff = {f: float(np.mean(v)) for f, v in ratios.items() if len(v) >= 3}
    return FamilyReport(rates, prior, base, overfit, eff, len(lab))


def context_report(recs: Sequence[TransferRecord], min_n: int = 5) -> dict:
    rates, prior = group_rates(((r.context_dim, r.transferred) for r in recs), min_n)
    return {"rates": rates, "prior": prior, "n": len(recs),
            "transfers": sorted(k for k, g in rates.items() if g.lo > 0.5),
            "does_not_transfer": sorted(k for k, g in rates.items() if g.hi < 0.5)}


@dataclass(frozen=True)
class DecayFit:
    memory_type: str
    n: int
    rate_per_day: float                            # lambda in ratio = exp(-lambda * age)
    half_life_days: float
    lam_lo: float
    lam_hi: float
    r2: float


def decay_fit(memory_type: str, recs: Sequence[DecayRecord], seed: int = 0, n_boot: int = 500) -> DecayFit | None:
    """Fit ln(ratio) = -lambda * age through the origin (a memory starts at ratio 1). Bootstrapped interval on lambda."""
    if len(recs) < 4:
        return None
    age = np.array([r.age_days for r in recs], float)
    y = np.log(np.clip([r.reliability_ratio for r in recs], 1e-3, 1e3))
    denom = float((age * age).sum())
    if denom <= 0:
        return None
    lam = float(-(age * y).sum() / denom)
    ss_res = float(((y + lam * age) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(age), size=(n_boot, len(age)))
    lams = np.array([-(age[i] * y[i]).sum() / max((age[i] ** 2).sum(), 1e-12) for i in idx])
    lo, hi = np.quantile(lams, [0.025, 0.975])
    hl = math.log(2) / lam if lam > 1e-9 else float("inf")
    return DecayFit(memory_type, len(recs), lam, hl, float(lo), float(hi), r2)


def decay_report(recs: Sequence[DecayRecord], seed: int = 0) -> dict:
    by: dict = {}
    for r in recs:
        by.setdefault(r.memory_type, []).append(r)
    fits = {k: decay_fit(k, v, seed) for k, v in by.items()}
    fits_ok = {k: f for k, f in fits.items() if f is not None}
    rapid = sorted((k for k, f in fits_ok.items() if f.lam_lo > 0 and f.half_life_days < 365), key=lambda k: fits_ok[k].half_life_days)
    return {"fits": fits_ok, "rapid_decay": rapid, "n": len(recs)}


def yield_report(recs: Sequence[YieldRecord], min_n: int = 3) -> dict:
    rates, prior = group_rates(((r.kind, r.useful) for r in recs), min_n)
    per: dict = {}
    for r in recs:
        p = per.setdefault(r.kind, {"bits": 0.0, "minutes": 0.0})
        p["bits"] += max(r.gain_bits, 0.0)
        p["minutes"] += max(r.cost_minutes, 0.0)
    return {"rates": rates, "prior": prior, "bits_per_minute": {k: (v["bits"] / v["minutes"] if v["minutes"] > 0 else None) for k, v in per.items()}, "n": len(recs)}


def learner_report(recs: Sequence[LearnerRecord], min_n: int = 2) -> dict:
    by: dict = {}
    for r in recs:
        by.setdefault(r.learner, []).append(r)
    out = {}
    for name, rows in by.items():
        claimed = [r for r in rows if r.claimed_gain > 0]
        fake = [r for r in claimed if not r.verified]
        lo, hi = wilson(len(fake), len(claimed))
        out[name] = {"n": len(rows), "claims": len(claimed), "fake": len(fake), "fake_rate": (len(fake) / len(claimed)) if claimed else None,
                     "fake_lo": lo, "fake_hi": hi, "produces_fake_improvement": len(claimed) >= min_n and lo > 0.5}
    return out


def explanation_report(recs: Sequence[ExplanationRecord], min_n: int = 3) -> dict:
    """Per predicted cause: how often the later-verified cause agreed. UNKNOWN verdicts count as NOT correct (an explanation
    that cannot be confirmed is not credit). Also a confusion matrix and calibration of the stated confidences."""
    prec, prior = group_rates(((r.predicted_cause, r.correct()) for r in recs), min_n)
    conf: dict = {}
    for r in recs:
        row = conf.setdefault(r.predicted_cause, {})
        row[r.verified_cause] = row.get(r.verified_cause, 0) + 1
    acc = sum(1 for r in recs if r.correct()) / len(recs) if recs else 0.0
    counts: dict = {}
    for r in recs:
        counts[r.verified_cause] = counts.get(r.verified_cause, 0) + 1
    majority = max(counts.values()) / len(recs) if recs else 0.0
    cal = None
    if len(recs) >= 10:
        conf_arr = np.array([r.confidence for r in recs])
        hit = np.array([1.0 if r.correct() else 0.0 for r in recs])
        cal = {"mean_confidence": float(conf_arr.mean()), "accuracy": float(hit.mean()), "brier": brier(conf_arr, hit),
               "overconfident": bool(conf_arr.mean() > hit.mean() + 0.1)}
    return {"precision": prec, "prior": prior, "confusion": conf, "accuracy": acc, "majority_baseline": majority,
            "beats_majority": acc > majority + 0.02, "calibration": cal, "n": len(recs)}


def ablation_report(recs: Sequence[AblationRecord], streak: int = 3, min_n: int = 3) -> dict:
    by: dict = {}
    for r in sorted(recs, key=lambda r: r.resolved_at):
        by.setdefault(r.feature_family, []).append(r.failed)
    out = {}
    for fam, seq in by.items():
        cur = 0
        for f in reversed(seq):
            if not f:
                break
            cur += 1
        out[fam] = {"n": len(seq), "failures": sum(seq), "fail_rate": sum(seq) / len(seq), "current_streak": cur,
                    "repeatedly_fails": len(seq) >= min_n and cur >= streak}
    return out


# ------------------------------------------------------------------------------------------------ out-of-sample evaluation

@dataclass(frozen=True)
class OosResult:
    task: str
    n_train_min: int
    n_test: int
    folds: int
    model_brier: float
    baseline_brier: float
    model_auc: float | None
    diff: Mapping                                  # paired bootstrap of (model - baseline) Brier
    beats_baseline: bool
    leak_suspects: tuple
    label: str
    reason: str


def _label_for(beats: bool, n_test: int, min_test: int, certified_real: bool, leaks: Sequence) -> tuple:
    if leaks:
        return ValidationLabel.FAILED_VALIDATION.value, f"label leak suspected: {list(leaks)}"
    if n_test < min_test:
        return ValidationLabel.INSUFFICIENT_EVIDENCE.value, f"only {n_test} out-of-sample cases (< {min_test})"
    if not beats:
        return ValidationLabel.FAILED_VALIDATION.value, "does not beat the base-rate baseline out of sample"
    if not certified_real:
        return ValidationLabel.NOT_VALIDATED.value, "beats baseline on records not certified as real: mechanism only"
    return ValidationLabel.VALIDATED.value, "beats baseline out of sample on certified real records"


def walk_forward_survival(recs: Sequence[DiscoveryRecord], now, seed: int, n_folds: int = 4, min_train: int = 20,
                          min_test: int = 30, l2: float = 2.0, certified_real: bool = False) -> OosResult:
    """The meta-learning OOS test. Discoveries are ordered by discovery time and cut into chronological folds. For fold k
    the model trains ONLY on records whose label was resolved before the fold's first discovery, and predicts the fold. The
    baseline is the training base rate. A record is never in both train and test, and no label from the future reaches a
    prediction: the resolved_at filter is what enforces that (test it: shifting resolved_at later must lower the score)."""
    cut_now = to_ts(now)
    lab = sorted((r for r in recs if r.survived_oos is not None and r.resolved_at and to_ts(r.resolved_at) < cut_now),
                 key=lambda r: (to_ts(r.discovered_at), r.item_id))
    if len(lab) < min_train + n_folds:
        return OosResult("survival", min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         f"only {len(lab)} labelled discoveries")
    edges = np.linspace(len(lab) // 2, len(lab), n_folds + 1).astype(int)
    p_model, p_base, ys = [], [], []
    folds = 0
    for a, b in zip(edges[:-1], edges[1:]):
        test = lab[a:b]
        if not test:
            continue
        cutoff = to_ts(test[0].discovered_at)
        train = [r for r in lab[:a] if to_ts(r.resolved_at) < cutoff]
        if len(train) < min_train:
            continue
        m = SurvivalModel(l2).fit(train)
        p_model += list(m.predict(test))
        p_base += [m.base] * len(test)
        ys += [1 if r.survived_oos else 0 for r in test]
        folds += 1
    if not ys:
        return OosResult("survival", min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         "no fold had enough labels resolved before it started")
    se_m = (np.array(p_model) - np.array(ys)) ** 2
    se_b = (np.array(p_base) - np.array(ys)) ** 2
    diff = paired_bootstrap(se_m, se_b, seed)
    beats = bool(diff["hi"] < 0)
    leaks = tuple(leak_audit(lab))
    label, why = _label_for(beats, len(ys), min_test, certified_real, leaks)
    return OosResult("survival", min_train, len(ys), folds, float(se_m.mean()), float(se_b.mean()), auc(p_model, ys), diff, beats, leaks, label, why)


def walk_forward_groups(rows: Sequence[tuple], now, seed: int, task: str, min_train: int = 20, min_test: int = 30, n_folds: int = 4,
                        certified_real: bool = False) -> OosResult:
    """OOS test for any group-rate predictor (family survival, context transfer, experiment yield, explanation precision,
    learner honesty). rows = (occurred_at, resolved_at, group_key, outcome_bool). Training uses groups' shrunk rates from
    rows resolved before the fold begins; baseline is the pooled training rate. Unseen groups fall back to the pooled rate."""
    cut_now = to_ts(now)
    lab = sorted((r for r in rows if r[1] and to_ts(r[1]) < cut_now), key=lambda r: (to_ts(r[0]), str(r[2])))
    if len(lab) < min_train + n_folds:
        return OosResult(task, min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         f"only {len(lab)} resolved rows")
    edges = np.linspace(len(lab) // 2, len(lab), n_folds + 1).astype(int)
    pm, pb, ys = [], [], []
    folds = 0
    for a, b in zip(edges[:-1], edges[1:]):
        test = lab[a:b]
        if not test:
            continue
        cutoff = to_ts(test[0][0])
        train = [r for r in lab[:a] if to_ts(r[1]) < cutoff]
        if len(train) < min_train:
            continue
        rates, prior = group_rates(((r[2], r[3]) for r in train), 1)
        pooled = sum(1 for r in train if r[3]) / len(train)
        for r in test:
            pm.append(rates[r[2]].shrunk if r[2] in rates else pooled)
            pb.append(pooled)
            ys.append(1 if r[3] else 0)
        folds += 1
    if not ys:
        return OosResult(task, min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         "no fold had enough resolved training rows")
    se_m = (np.array(pm) - np.array(ys)) ** 2
    se_b = (np.array(pb) - np.array(ys)) ** 2
    diff = paired_bootstrap(se_m, se_b, seed)
    beats = bool(diff["hi"] < 0)
    label, why = _label_for(beats, len(ys), min_test, certified_real, ())
    return OosResult(task, min_train, len(ys), folds, float(se_m.mean()), float(se_b.mean()), auc(pm, ys), diff, beats, (), label, why)


# ------------------------------------------------------------------------------------------------ the meta-learner (C16)

@dataclass(frozen=True)
class MetaConfig:
    min_family_n: int = 5
    min_group_n: int = 3
    min_train: int = 20
    min_test: int = 30
    folds: int = 4
    overfit_margin: float = 0.05
    certified_real: bool = False                    # only the real-data validation wave may set this

    def check(self) -> list:
        return [] if self.min_family_n >= 1 and self.folds >= 2 and self.min_train >= 5 else ["bad MetaConfig"]


@dataclass(frozen=True)
class MetaUpdate:
    now: str
    counts: Mapping
    families: FamilyReport
    contexts: Mapping
    decay: Mapping
    yields: Mapping
    learners: Mapping
    explanations: Mapping
    ablations: Mapping
    oos: Mapping                                    # task -> OosResult
    survival_importance: tuple
    advice: MetaAdvice
    store_hash: str
    code_hash: str
    label: str
    health: Mapping = field(default_factory=dict)

    def summary(self) -> dict:
        return {"now": self.now, "counts": dict(self.counts), "label": self.label, "store_hash": self.store_hash,
                "oos": {k: {"label": v.label, "n_test": v.n_test, "beats": v.beats_baseline, "reason": v.reason} for k, v in self.oos.items()},
                "overfit_families": sorted(self.families.overfit, key=lambda f: -self.families.overfit[f])[:5],
                "rapid_decay": list(self.decay.get("rapid_decay", [])), "fake_learners": sorted(k for k, v in self.learners.items() if v["produces_fake_improvement"]),
                "failing_ablations": sorted(k for k, v in self.ablations.items() if v["repeatedly_fails"])}


class MetaLearner:
    """Fits all eight analyses on records resolved before `now`, evaluates them out of sample, and emits MetaAdvice.
    Groups with too little data are OMITTED from the advice (unknown is not neutral), and the advice carries the OOS label
    of the survival task so downstream trust is halved unless meta-learning has certified itself on real records."""

    def __init__(self, store: MetaStore | None = None, cfg: MetaConfig | None = None):
        self.store = store or MetaStore()
        self.cfg = cfg or MetaConfig()
        errs = self.cfg.check()
        if errs:
            raise ValueError("; ".join(errs))
        self.history: list = []

    def update(self, now, seed: int = 0) -> MetaUpdate:
        s = self.store.as_of(now)
        c = self.cfg
        fam = family_report(s.discoveries, c.min_family_n, c.overfit_margin)
        ctx = context_report(s.transfers, c.min_group_n)
        dec = decay_report(s.decays, seed)
        yl = yield_report(s.yields, c.min_group_n)
        lrn = learner_report(s.learners)
        exp = explanation_report(s.explanations, c.min_group_n)
        abl = ablation_report(s.ablations)
        model = SurvivalModel().fit(s.discoveries)
        oos = {
            "survival": walk_forward_survival(s.discoveries, now, seed, c.folds, c.min_train, c.min_test, certified_real=c.certified_real),
            "family_survival": walk_forward_groups([(r.discovered_at, r.resolved_at, r.family, bool(r.survived_oos)) for r in s.discoveries if r.survived_oos is not None],
                                                   now, seed, "family_survival", c.min_train, c.min_test, c.folds, c.certified_real),
            "context_transfer": walk_forward_groups([(r.resolved_at, r.resolved_at, r.context_dim, r.transferred) for r in s.transfers],
                                                    now, seed + 1, "context_transfer", c.min_train, c.min_test, c.folds, c.certified_real),
            "experiment_yield": walk_forward_groups([(r.resolved_at, r.resolved_at, r.kind, r.useful) for r in s.yields],
                                                    now, seed + 2, "experiment_yield", c.min_train, c.min_test, c.folds, c.certified_real),
            "explanation_precision": walk_forward_groups([(r.resolved_at, r.resolved_at, r.predicted_cause, r.correct()) for r in s.explanations],
                                                         now, seed + 3, "explanation_precision", c.min_train, c.min_test, c.folds, c.certified_real)}
        advice = MetaAdvice(
            family_overfit={f: round(p, 4) for f, p in fam.overfit.items()},
            family_survival={f: round(g.shrunk, 4) for f, g in fam.rates.items() if g.n >= c.min_family_n},
            target_yield={k: round(g.shrunk, 4) for k, g in yl["rates"].items()},
            context_transfer={k: round(g.shrunk, 4) for k, g in ctx["rates"].items()},
            explanation_precision={k: round(g.shrunk, 4) for k, g in exp["precision"].items()},
            fitted_through=str(now), n_observations=sum(s.counts().values()), oos_label=oos["survival"].label)
        errs = advice.check()
        if errs:
            raise ValueError("; ".join(errs))
        upd = MetaUpdate(str(now), s.counts(), fam, ctx, dec, yl, lrn, exp, abl, oos, tuple(model.importance()[:8]), advice,
                         s.code_hash(), current_code_hash(), LABEL, store_health(self.store, now))
        self.history.append(upd.summary())
        return upd

    def advice(self, now, seed: int = 0) -> MetaAdvice:
        return self.update(now, seed).advice

    def save_history(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json(self.history[-1]) + "\n")


def meta_generalisation(learner: MetaLearner, eras: Sequence[tuple], seed: int = 0) -> dict:
    """Does what meta-learning learned in one era carry to the next? eras = [(start, end)] chronological. For each era k>0
    the advice is fitted using ONLY records resolved before era k starts and scored on the survival outcomes of discoveries
    made in era k: mean Brier of the per-family predictor against the pooled rate. The 'meta transfer ratio'."""
    out = []
    for i in range(1, len(eras)):
        start, end = eras[i]
        train = learner.store.as_of(start)
        fam = family_report(train.discoveries, learner.cfg.min_family_n)
        test = [r for r in learner.store.discoveries if r.survived_oos is not None and to_ts(start) <= to_ts(r.discovered_at) < to_ts(end)
                and r.resolved_at and to_ts(r.resolved_at) < to_ts(end) + (to_ts(end) - to_ts(start))]
        if len(test) < 5 or fam.n < 10:
            out.append({"era": i, "n_test": len(test), "verdict": "INSUFFICIENT"})
            continue
        pooled = fam.base_rate
        pm = [fam.rates[r.family].shrunk if r.family in fam.rates else pooled for r in test]
        y = [1 if r.survived_oos else 0 for r in test]
        out.append({"era": i, "n_test": len(test), "brier_family": brier(pm, y), "brier_pooled": brier([pooled] * len(y), y),
                    "verdict": "TRANSFERS" if brier(pm, y) < brier([pooled] * len(y), y) else "NO_TRANSFER"})
    good = sum(1 for o in out if o["verdict"] == "TRANSFERS")
    scored = sum(1 for o in out if o["verdict"] != "INSUFFICIENT")
    return {"eras": out, "transfer_ratio": (good / scored) if scored else None, "label": LABEL}


# ------------------------------------------------------------------------------------------------ planted worlds (for C63 / tests)

def synthetic_discoveries(n: int, seed: int, family_survival: Mapping[str, float] | None = None, signal_feature: bool = True,
                          start: str = "2001-01-03", spacing_days: int = 5, resolve_lag_days: int = 90, leak: bool = False) -> list:
    """Planted meta-world. Each discovery has a family with a hidden true survival probability, and a continuous feature
    `t_disc` that raises survival when `signal_feature` (an honest, learnable relation). `leak=True` adds a feature that is
    the label itself (a defect the leak audit must catch). Dates advance by `spacing_days`; labels resolve `resolve_lag_days` later."""
    import datetime as dt
    fam = dict(family_survival or {"momentum_like": 0.65, "volatility_like": 0.55, "calendar_like": 0.12, "tiny_sample": 0.15})
    names = sorted(fam)
    rng = np.random.default_rng(seed)
    t0 = dt.date.fromisoformat(start)
    out = []
    for i in range(n):
        f = names[int(rng.integers(0, len(names)))]
        t = float(rng.normal(3.0, 1.0))
        logit = math.log(fam[f] / (1 - fam[f])) + (0.8 * (t - 3.0) if signal_feature else 0.0)
        surv = bool(rng.random() < 1 / (1 + math.exp(-logit)))
        d = t0 + dt.timedelta(days=i * spacing_days)
        feats = {"t_disc": t, "log_n": float(rng.normal(6.0, 1.0)), "n_conditions": float(rng.integers(1, 5))}
        if leak:
            feats["post_hoc_flag"] = 1.0 if surv else 0.0
        is_e = float(abs(rng.normal(0.02, 0.005)))
        out.append(DiscoveryRecord(f"d{i:05d}", f, d.isoformat(), feats, surv, (d + dt.timedelta(days=resolve_lag_days)).isoformat(),
                                   is_e, is_e * (0.8 if surv else float(rng.normal(0.1, 0.2)))))
    return out


def render_update(u: MetaUpdate) -> str:
    """Plain-text meta-learning report."""
    lines = [f"Meta-learning update as of {u.now}   [{u.label}]", f"records: {dict(u.counts)}"]
    lines.append(f"survival base rate {u.families.base_rate:.2f} over {u.families.n} resolved discoveries")
    for f, g in sorted(u.families.rates.items(), key=lambda kv: kv[1].shrunk)[:8]:
        flag = " OVERFITS" if f in u.families.overfit and u.families.overfit[f] > 0.6 else ""
        lines.append(f"  family {f}: {g.successes}/{g.n} raw {g.raw:.2f} shrunk {g.shrunk:.2f} [{g.lo:.2f},{g.hi:.2f}]{flag}")
    if u.decay.get("fits"):
        for k, f in u.decay["fits"].items():
            lines.append(f"  memory type {k}: half-life {f.half_life_days:.0f} d (lambda {f.rate_per_day:.5f} [{f.lam_lo:.5f},{f.lam_hi:.5f}], n={f.n})")
    for k, v in u.learners.items():
        lines.append(f"  learner {k}: {v['fake']}/{v['claims']} claimed improvements not verified" + (" -> FAKE" if v["produces_fake_improvement"] else ""))
    lines.append("out-of-sample evaluation of meta-learning itself:")
    for k, o in u.oos.items():
        lines.append(f"  {k}: {o.label} ({o.reason}); n_test={o.n_test}, model Brier {o.model_brier:.4f} vs baseline {o.baseline_brier:.4f}")
    return "\n".join(lines)


# ==================================================================================================================
# Part 2: calibration, drift, importance, triage value, null control, confounding, persistence, adapters
# ==================================================================================================================

class PlattRecalibrator:
    """Sigmoid recalibration p' = sigmoid(a * logit(p) + b), fitted on a HELD-OUT set (never on the model's own training
    rows, which are already fitted). A survival model that ranks well but is overconfident is fixed here, not re-trained."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.n = 0

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, l2: float = 0.5) -> "PlattRecalibrator":
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        self.n = len(y)
        if len(y) < 15 or y.min() == y.max():
            return self                                # too little to recalibrate: identity
        z = self._logit(p).reshape(-1, 1)
        w = fit_logistic(z, y, l2=l2)
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def transform(self, p: Sequence[float] | np.ndarray) -> np.ndarray:
        return sigmoid(self.a * self._logit(p) + self.b)


def reliability_bins(p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, bins: int = 5) -> list:
    """Predicted vs observed survival per probability bin, with the count and a Wilson interval on the observed rate."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.sum() == 0:
            continue
        lo, hi = wilson(int(y[m].sum()), int(m.sum()))
        out.append({"bin": f"{edges[i]:.2f}-{edges[i + 1]:.2f}", "n": int(m.sum()), "predicted": float(p[m].mean()),
                    "observed": float(y[m].mean()), "lo": lo, "hi": hi})
    return out


def expected_calibration_error(p: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, bins: int = 5) -> float:
    rows = reliability_bins(p, y, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / n) if n else float("nan")


def precision_at_k(scores: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, k: int) -> float | None:
    """Share of the k highest-scored discoveries that survived. The meta-learner's practical use is triage: which few to test first."""
    if k < 1 or len(scores) < k:
        return None
    order = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    return float(np.asarray(y, float)[order].mean())


def decile_lift(scores: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, n_groups: int = 5) -> list:
    """Survival rate by score quantile group, best group first, and lift versus the overall rate."""
    s = np.asarray(scores, float)
    yy = np.asarray(y, float)
    if len(s) < n_groups * 3:
        return []
    order = np.argsort(-s, kind="stable")
    parts = np.array_split(order, n_groups)
    base = float(yy.mean())
    return [{"group": i + 1, "n": len(ix), "rate": float(yy[ix].mean()), "lift": float(yy[ix].mean() / base) if base > 0 else None}
            for i, ix in enumerate(parts)]


def triage_value(scores: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray, skip_fraction: float, cost_per_test: float = 1.0, value_per_survivor: float = 10.0) -> dict:
    """Counterfactual: if the lowest-scored `skip_fraction` of discoveries had NOT been OOS-tested, how much compute is saved
    and how many real survivors are lost? Net value = saved compute - lost survivors x value. Positive net value is the only
    reason a survival model deserves to influence what gets tested."""
    s = np.asarray(scores, float)
    yy = np.asarray(y, int)
    n = len(s)
    if n == 0 or not 0.0 <= skip_fraction < 1.0:
        raise ValueError("need data and 0 <= skip_fraction < 1")
    skip = int(round(n * skip_fraction))
    order = np.argsort(s, kind="stable")
    skipped = order[:skip]
    lost = int(yy[skipped].sum())
    saved = skip * cost_per_test
    return {"n": n, "skipped": skip, "survivors_lost": lost, "survivors_total": int(yy.sum()), "compute_saved": saved,
            "net_value": saved - lost * value_per_survivor, "recall_kept": 1.0 - (lost / max(int(yy.sum()), 1))}


def permutation_importance(train: Sequence[DiscoveryRecord], test: Sequence[DiscoveryRecord], seed: int, n_rep: int = 20) -> list:
    """Out-of-sample permutation importance: the Brier increase on `test` when one feature is shuffled across test rows.
    Fitted on `train` only. A feature whose shuffling does not hurt is not being used, whatever its coefficient says."""
    m = SurvivalModel().fit(train)
    lab = [r for r in test if r.survived_oos is not None]
    if len(lab) < 10:
        return []
    y = np.array([1 if r.survived_oos else 0 for r in lab])
    base = brier(m.predict(lab), y)
    rng = np.random.default_rng(seed)
    out = []
    for name in m.names:
        incr = []
        for _ in range(n_rep):
            perm = rng.permutation(len(lab))
            shuffled = [replace(r, features={**dict(r.features), name: float(lab[j].features.get(name, 0.0))}) for r, j in zip(lab, perm)]
            incr.append(brier(m.predict(shuffled), y) - base)
        out.append({"feature": name, "brier_increase": float(np.mean(incr)), "sd": float(np.std(incr))})
    return sorted(out, key=lambda r: -r["brier_increase"])


def null_control(recs: Sequence[DiscoveryRecord], now, seed: int, n_perm: int = 50, **wf) -> dict:
    """Is the out-of-sample skill bigger than skill from RANDOM labels? Survival labels are permuted (feature/label pairing
    destroyed, dates and resolution times kept) and the whole walk-forward rerun n_perm times. The p-value is the share of
    permuted runs whose model-minus-baseline Brier is at least as good as the real one. A skill that null runs match is not skill."""
    real = walk_forward_survival(recs, now, seed, **wf)
    if real.n_test == 0:
        return {"verdict": "INSUFFICIENT", "reason": real.reason}
    rng = np.random.default_rng(seed + 7919)
    labelled = [r for r in recs if r.survived_oos is not None]
    flags = np.array([bool(r.survived_oos) for r in labelled])
    real_gain = real.baseline_brier - real.model_brier
    better = 0
    gains = []
    for _ in range(n_perm):
        perm = rng.permutation(len(flags))
        fake = [replace(r, survived_oos=bool(flags[j])) for r, j in zip(labelled, perm)]
        res = walk_forward_survival(fake, now, seed, **wf)
        g = res.baseline_brier - res.model_brier
        gains.append(g)
        better += int(g >= real_gain)
    p = (better + 1) / (n_perm + 1)
    return {"verdict": "SKILL" if p < 0.05 and real_gain > 0 else "NO_SKILL_BEYOND_NULL", "real_gain": float(real_gain),
            "null_mean": float(np.mean(gains)), "null_p95": float(np.quantile(gains, 0.95)), "p": float(p), "n_perm": n_perm,
            "n_test": real.n_test}


class SurvivalDrift:
    """Page-Hinkley detector on the stream of survival outcomes (in resolution order): has the base rate at which
    discoveries survive changed? If the world changes, everything meta-learning learned about families may be stale. Alarms
    are advisory and reset the detector; they never delete records."""

    def __init__(self, delta: float = 0.02, lam: float = 4.0):
        self.delta = delta
        self.lam = lam
        self.n = 0
        self.mean = 0.0
        self.cum_up = 0.0
        self.min_up = 0.0
        self.cum_dn = 0.0
        self.max_dn = 0.0

    def update(self, survived: bool) -> str | None:
        x = 1.0 if survived else 0.0
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum_up += x - self.mean - self.delta
        self.min_up = min(self.min_up, self.cum_up)
        self.cum_dn += x - self.mean + self.delta
        self.max_dn = max(self.max_dn, self.cum_dn)
        if self.n >= 10 and self.cum_up - self.min_up > self.lam:
            self.__init__(self.delta, self.lam)  # type: ignore[misc]  # in-place reset of the detector state
            return "RATE_UP"
        if self.n >= 10 and self.max_dn - self.cum_dn > self.lam:
            self.__init__(self.delta, self.lam)  # type: ignore[misc]  # in-place reset of the detector state
            return "RATE_DOWN"
        return None


def drift_scan(recs: Sequence[DiscoveryRecord], delta: float = 0.02, lam: float = 4.0) -> list:
    """Run SurvivalDrift over labelled discoveries in resolution order; returns [(resolved_at, alarm)]."""
    det = SurvivalDrift(delta, lam)
    out = []
    for r in sorted((r for r in recs if r.survived_oos is not None and r.resolved_at), key=lambda r: to_ts(r.resolved_at)):
        a = det.update(bool(r.survived_oos))
        if a:
            out.append((r.resolved_at, a))
    return out


def temporal_class_report(recs: Sequence[DiscoveryRecord], min_n: int = 3) -> dict:
    """Survival by declared temporal class (section 14): do EPISODIC or REGIME_BOUND discoveries survive OOS less than PERSISTENT ones?"""
    rates, prior = group_rates(((r.temporal_class, r.survived_oos) for r in recs if r.survived_oos is not None), min_n)
    return {"rates": rates, "prior": prior}


def era_confound_report(recs: Sequence[DiscoveryRecord], n_eras: int = 3, min_n: int = 4) -> dict:
    """Is a 'bad family' just a family that was discovered in a bad era? Survival is computed per family within era terciles
    of discovery date; a family is CONFOUNDED when its pooled shortfall vanishes within eras (Simpson's paradox guard).
    Families need >= min_n records in an era to be counted there."""
    lab = sorted((r for r in recs if r.survived_oos is not None), key=lambda r: (to_ts(r.discovered_at), r.item_id))
    if len(lab) < n_eras * min_n * 2:
        return {"verdict": "INSUFFICIENT", "n": len(lab)}
    parts = np.array_split(np.arange(len(lab)), n_eras)
    era_of = {}
    for e, ix in enumerate(parts):
        for i in ix:
            era_of[lab[i].item_id] = e
    era_rate = {e: float(np.mean([bool(lab[i].survived_oos) for i in ix])) for e, ix in enumerate(parts)}
    pooled = float(np.mean([bool(r.survived_oos) for r in lab]))
    fams = sorted({r.family for r in lab})
    out = {}
    for f in fams:
        rows = [r for r in lab if r.family == f]
        if len(rows) < min_n:
            continue
        raw = float(np.mean([bool(r.survived_oos) for r in rows]))
        exp = float(np.mean([era_rate[era_of[r.item_id]] for r in rows]))                # what the eras alone predict
        out[f] = {"n": len(rows), "raw": raw, "era_expected": exp, "excess_vs_era": raw - exp, "excess_vs_pooled": raw - pooled,
                  "confounded": bool(abs(raw - pooled) > 0.08 and abs(raw - exp) < 0.04)}
    return {"verdict": "OK", "era_rates": era_rate, "families": out}


# ------------------------------------------------------------------------------------------------ persistence

_KIND_TYPES = {"discoveries": DiscoveryRecord, "transfers": TransferRecord, "decays": DecayRecord, "yields": YieldRecord,
               "learners": LearnerRecord, "explanations": ExplanationRecord, "ablations": AblationRecord}


def save_store(store: MetaStore, path) -> int:
    """One JSON line per record: {'kind': ..., 'rec': {...}}. Overwrites the file (the store is the source of truth in memory);
    returns the number of records written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for kind in MetaStore.KINDS:
            for r in getattr(store, kind):
                f.write(json.dumps({"kind": kind, "rec": json.loads(canonical_json(r))}) + "\n")
                n += 1
    return n


def load_store(path) -> tuple:
    """(MetaStore, n_bad). Bad lines are counted, never silently dropped; records failing validation are counted as bad too."""
    store = MetaStore()
    bad = 0
    p = Path(path)
    if not p.exists():
        return store, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            cls = _KIND_TYPES[d["kind"]]
            rec = d["rec"]
            if cls is DiscoveryRecord:
                rec = {**rec, "features": dict(rec.get("features", {}))}
            store.add(cls(**rec))
        except (ValueError, KeyError, TypeError):
            bad += 1
    return store, bad


# ------------------------------------------------------------------------------------------------ adapters from the engine

def discoveries_from_pattern_rows(rows: Iterable[Mapping], discovered_at: str, resolved_at: str, family_fn: Callable[[Mapping], str] | None = None,
                                  survive_t: float = 2.0) -> list:
    """DiscoveryRecords from engine.patterns.PatternMiner-style rows (dicts with key_named, effect, t_disc, t_conf, p_real, n).
    Survival is DEFINED here as: the confirmation-window t statistic keeps the discovery sign and reaches `survive_t`. Features
    are those known at discovery only (t_disc, |effect|, condition count, p_real, log n); t_conf, the label source, is NOT a
    feature. `discovered_at`/`resolved_at` are the real dates of the discovery and confirmation windows (trusted side)."""
    out = []
    for i, r in enumerate(rows):
        key = str(r.get("key_named") or r.get("key") or f"row{i}")
        eff = float(r.get("effect") or 0.0)
        t_disc = float(r.get("t_disc") or 0.0)
        t_conf = r.get("t_conf")
        surv = None if t_conf is None or (isinstance(t_conf, float) and math.isnan(t_conf)) else bool(float(t_conf) * (1 if t_disc >= 0 else -1) >= survive_t)
        fam = family_fn(r) if family_fn else key.split(" ")[0].split("<")[0].split(">")[0][:24] or "unnamed"
        n = float(r.get("n") or 0.0)
        feats = {"t_disc": abs(t_disc), "abs_effect": abs(eff), "n_conditions": float(key.count("&") + key.count(" and ") + 1),
                 "p_real": float(r.get("p_real") or 0.0), "log_n": math.log1p(max(n, 0.0))}
        out.append(DiscoveryRecord(stable_hash([key, discovered_at], 10), fam, discovered_at, feats, surv,
                                   resolved_at if surv is not None else "", abs(eff) if eff else None, None))
    return out


def explanations_from_postmortems(rows: Iterable[Mapping], resolved_at: str) -> list:
    """ExplanationRecords from rows {item_id, predicted_cause, confidence, verified_cause}. Causes are validated against
    FailureCause; a row with an unknown label is rejected (loudly), not coerced to UNKNOWN."""
    out = []
    for r in rows:
        for k in ("predicted_cause", "verified_cause"):
            FailureCause.parse(r[k])
        out.append(ExplanationRecord(str(r["item_id"]), FailureCause.parse(r["predicted_cause"]).value, float(r.get("confidence", 0.5)),
                                     FailureCause.parse(r["verified_cause"]).value, resolved_at))
    return out


def explanation_trust(update: "MetaUpdate", cause: str, default: float | None = None) -> float | None:
    """How far to trust a failure explanation of this cause: its shrunk historical precision, or `default` (None = unknown)
    when the cause has too little history. Consumers must treat None as UNKNOWN, not as 0.5."""
    g = update.explanations["precision"].get(cause)
    return g.shrunk if g is not None else default


def verify_learner_claim(claimed_gain: float, verified_gain: float | None, tolerance: float = 0.5) -> bool:
    """A claim is verified when an INDEPENDENT evaluation reproduced at least `tolerance` of the claimed gain with the same sign."""
    if verified_gain is None or claimed_gain == 0:
        return False
    return (verified_gain * claimed_gain > 0) and abs(verified_gain) >= tolerance * abs(claimed_gain)


def write_report(update: "MetaUpdate", out_dir) -> Path:
    """Write the plain-text report and the JSON summary of an update with its provenance (store hash, code hash, label)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta_report.txt").write_text(render_update(update), encoding="utf-8", newline="\n")
    (d / "meta_summary.json").write_text(canonical_json(update.summary()), encoding="utf-8", newline="\n")
    return d / "meta_report.txt"


# ==================================================================================================================
# Part 3: streaming statistics, experiment-yield model, meta learning curve, and planted-world self-checks
# ==================================================================================================================

class OnlineGroupRates:
    """Streaming version of `group_rates`: one outcome at a time, same shrinkage as the batch code at any moment (the
    equivalence is tested). Lets the meta-learner run inside a long job without re-reading history."""

    def __init__(self):
        self.tally: dict = {}
        self.last_seen: dict = {}

    def update(self, key: str, outcome: bool, when: str = "") -> None:
        s, n = self.tally.get(key, (0, 0))
        self.tally[key] = (s + int(bool(outcome)), n + 1)
        if when:
            self.last_seen[key] = when

    def snapshot(self, min_n: int = 1) -> tuple:
        prior = fit_beta_prior([v[0] for v in self.tally.values()], [v[1] for v in self.tally.values()])
        out = {}
        for k, (s, n) in self.tally.items():
            if n >= min_n:
                lo, hi = wilson(s, n)
                out[k] = GroupRate(k, n, s, s / n, shrunk_rate(s, n, prior), lo, hi)
        return out, prior

    def stale(self, now, days: float = 365.0) -> list:
        """Keys with no update for `days`: their rate may describe a world that no longer exists."""
        cut = to_ts(now)
        return sorted(k for k, w in self.last_seen.items() if (cut - to_ts(w)).total_seconds() / 86400.0 > days)


@dataclass(frozen=True)
class YieldExample:
    """Features of an experiment known BEFORE it ran, and whether it produced useful knowledge."""
    experiment_id: str
    kind: str
    predicted_bits: float
    cost_minutes: float
    novelty: float
    prior_tests_on_window: float
    useful: bool
    planned_at: str
    resolved_at: str


class YieldModel:
    """Which experiments produce useful knowledge? Ridge logistic on pre-run features plus one-hot kind. Evaluated by the
    same chronological rule as everything else: trained on examples RESOLVED before the test example was PLANNED."""

    def __init__(self, l2: float = 2.0):
        self.l2 = l2
        self.kinds: tuple = ()
        self.mu = np.zeros(4)
        self.sd = np.ones(4)
        self.w = np.zeros(1)
        self.base = 0.5

    def _x(self, ex: Sequence[YieldExample]) -> np.ndarray:
        num = np.array([[e.predicted_bits, math.log1p(e.cost_minutes), e.novelty, e.prior_tests_on_window] for e in ex], float).reshape(len(ex), 4)
        oh = np.array([[1.0 if e.kind == k else 0.0 for k in self.kinds] for e in ex], float).reshape(len(ex), len(self.kinds))
        return np.hstack([(num - self.mu) / self.sd, oh])

    def fit(self, ex: Sequence[YieldExample]) -> "YieldModel":
        self.kinds = tuple(sorted({e.kind for e in ex}))
        y = np.array([1.0 if e.useful else 0.0 for e in ex])
        self.base = float(y.mean()) if len(y) else 0.5
        num = np.array([[e.predicted_bits, math.log1p(e.cost_minutes), e.novelty, e.prior_tests_on_window] for e in ex], float).reshape(len(ex), 4)
        self.mu = num.mean(axis=0) if len(ex) else np.zeros(4)
        sd = num.std(axis=0) if len(ex) else np.ones(4)
        self.sd = np.where(sd < 1e-9, 1.0, sd)
        if len(ex) < 12 or y.min() == y.max():
            self.w = np.zeros(4 + len(self.kinds) + 1)
            self.w[-1] = math.log(max(self.base, 0.02) / max(1 - self.base, 0.02))
            return self
        self.w = fit_logistic(self._x(ex), y, self.l2)
        return self

    def predict(self, ex: Sequence[YieldExample]) -> np.ndarray:
        if not ex:
            return np.zeros(0)
        return sigmoid(self._x(ex) @ self.w[:-1] + self.w[-1])


def walk_forward_yield(examples: Sequence[YieldExample], now, seed: int, n_folds: int = 4, min_train: int = 20, min_test: int = 30,
                       certified_real: bool = False) -> OosResult:
    """OOS evaluation of the yield model: a test example is predicted by a model trained only on examples resolved before it was planned."""
    cut = to_ts(now)
    ex = sorted((e for e in examples if e.resolved_at and to_ts(e.resolved_at) < cut), key=lambda e: (to_ts(e.planned_at), e.experiment_id))
    if len(ex) < min_train + n_folds:
        return OosResult("yield_model", min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         f"only {len(ex)} resolved experiments")
    edges = np.linspace(len(ex) // 2, len(ex), n_folds + 1).astype(int)
    pm, pb, ys = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        test = ex[a:b]
        if not test:
            continue
        start = to_ts(test[0].planned_at)
        train = [e for e in ex[:a] if to_ts(e.resolved_at) < start]
        if len(train) < min_train:
            continue
        m = YieldModel().fit(train)
        pm += list(m.predict(test))
        pb += [m.base] * len(test)
        ys += [1 if e.useful else 0 for e in test]
    if not ys:
        return OosResult("yield_model", min_train, 0, 0, float("nan"), float("nan"), None, {}, False, (), ValidationLabel.INSUFFICIENT_EVIDENCE.value,
                         "no fold had enough examples resolved before it was planned")
    se_m = (np.array(pm) - np.array(ys)) ** 2
    se_b = (np.array(pb) - np.array(ys)) ** 2
    diff = paired_bootstrap(se_m, se_b, seed)
    beats = bool(diff["hi"] < 0)
    label, why = _label_for(beats, len(ys), min_test, certified_real, ())
    return OosResult("yield_model", min_train, len(ys), 4, float(se_m.mean()), float(se_b.mean()), auc(pm, ys), diff, beats, (), label, why)


def meta_learning_curve(store: MetaStore, checkpoints: Sequence[str], seed: int = 0, cfg: MetaConfig | None = None) -> list:
    """Does meta-learning improve as evidence accrues? At each checkpoint date the whole survival evaluation is rerun using
    only records resolved before it. A rising skill (baseline Brier minus model Brier) with checkpoints is the evidence that
    the meta-learner is learning; a flat or falling curve says its lessons do not accumulate."""
    cfg = cfg or MetaConfig()
    rows: list[dict[str, Any]] = []
    for cp in checkpoints:
        s = store.as_of(cp)
        res = walk_forward_survival(s.discoveries, cp, seed, cfg.folds, cfg.min_train, cfg.min_test, certified_real=cfg.certified_real)
        skill = res.baseline_brier - res.model_brier if res.n_test else float("nan")
        rows.append({"checkpoint": cp, "n_labels": len([r for r in s.discoveries if r.survived_oos is not None]), "n_test": res.n_test,
                     "skill": skill, "auc": res.model_auc, "label": res.label})
    valid = [(i, r["skill"]) for i, r in enumerate(rows) if r["n_test"] and not math.isnan(r["skill"])]
    slope = float(np.polyfit([i for i, _ in valid], [s for _, s in valid], 1)[0]) if len(valid) >= 3 else None
    return [{**r, "trend_slope": slope} for r in rows]


def leaky_evaluation(recs: Sequence[DiscoveryRecord], now, seed: int) -> dict:
    """Deliberately WRONG evaluation that trains on labels resolved AFTER the fold starts (the classic future-label leak),
    used to show that the honest walk-forward is stricter. Returns both scores; the leaky one should look better on any
    world where labels carry information about the future (and is what a careless harness would report)."""
    honest = walk_forward_survival(recs, now, seed)
    lab = sorted((r for r in recs if r.survived_oos is not None), key=lambda r: (to_ts(r.discovered_at), r.item_id))
    if len(lab) < 40:
        return {"honest_skill": None, "leaky_skill": None}
    half = len(lab) // 2
    m = SurvivalModel().fit(lab[half:])                 # trains on the very rows it is scored on: the leak
    y = np.array([1 if r.survived_oos else 0 for r in lab[half:]])
    leaky = brier([m.base] * len(y), y) - brier(m.predict(lab[half:]), y)
    return {"honest_skill": honest.baseline_brier - honest.model_brier if honest.n_test else None, "leaky_skill": float(leaky)}


def self_check(seed: int = 0) -> dict:
    """Planted-world self-check of meta-learning (C63). Every entry can fail:
      finds_signal          family effects planted -> the OOS survival model beats the base rate
      null_no_false_claim   no structure -> it does NOT beat the base rate
      leak_flagged          a feature equal to the label -> FAILED_VALIDATION with the feature named
      future_labels_ignored records resolved after `now` change nothing
      overfit_family_found  the planted bad family is reported as overfitting, the good one is not
      synthetic_never_validated  certified_real=False can never return VALIDATED"""
    out: dict[str, Any] = {}
    sig = synthetic_discoveries(400, seed)
    st = MetaStore()
    st.extend(sig)
    ml = MetaLearner(st)
    u = ml.update("2010-01-01", seed)
    out["finds_signal"] = bool(u.oos["survival"].beats_baseline)
    out["overfit_family_found"] = bool(u.families.overfit.get("calendar_like", 0) > 0.6 and u.families.overfit.get("momentum_like", 1) < 0.6)
    out["synthetic_never_validated"] = all(o.label != ValidationLabel.VALIDATED.value for o in u.oos.values())
    null = synthetic_discoveries(400, seed, family_survival={"a": 0.4, "b": 0.4, "c": 0.4}, signal_feature=False)
    sn = MetaStore()
    sn.extend(null)
    out["null_no_false_claim"] = not MetaLearner(sn).update("2010-01-01", seed).oos["survival"].beats_baseline
    leak = MetaStore()
    leak.extend(synthetic_discoveries(300, seed, leak=True))
    ul = MetaLearner(leak).update("2010-01-01", seed)
    out["leak_flagged"] = bool(ul.oos["survival"].leak_suspects and ul.oos["survival"].label == ValidationLabel.FAILED_VALIDATION.value)
    early = MetaStore()
    early.extend(sig)
    cutoff = "2004-01-01"
    a = MetaLearner(early).update(cutoff, seed)
    extra = MetaStore()
    extra.extend(sig + [replace(r, item_id="late" + r.item_id, discovered_at="2005-06-01", resolved_at="2006-01-01") for r in sig[:50]])
    b = MetaLearner(extra).update(cutoff, seed)
    out["future_labels_ignored"] = bool(a.counts == b.counts and a.advice.family_survival == b.advice.family_survival)
    out["all_passed"] = all(out.values())
    out["label"] = LABEL
    return out


# ==================================================================================================================
# Part 4: store health, ablation forecasting, explanation calibration, learner ranking, update diffs
# ==================================================================================================================

def store_health(store: MetaStore, now) -> dict:
    """Is there anything to learn from? Class balance, time coverage, duplicate ids and the share of discoveries still
    unresolved at `now`. A store failing these gates is not meta-learned from (the update still runs, but reports the
    problems so nobody reads a number built on nothing)."""
    cut = to_ts(now)
    problems = []
    disc = store.discoveries
    resolved = [r for r in disc if r.survived_oos is not None and r.resolved_at and to_ts(r.resolved_at) < cut]
    unresolved = [r for r in disc if r.survived_oos is None or not r.resolved_at or to_ts(r.resolved_at) >= cut]
    ids = [r.item_id for r in disc]
    if len(ids) != len(set(ids)):
        problems.append("duplicate discovery ids")
    if resolved:
        rate = sum(1 for r in resolved if r.survived_oos) / len(resolved)
        if rate < 0.03 or rate > 0.97:
            problems.append(f"degenerate survival rate {rate:.2f}: no contrast to learn from")
        span = (max(to_ts(r.discovered_at) for r in resolved) - min(to_ts(r.discovered_at) for r in resolved)).days
        if span < 365:
            problems.append(f"discoveries span only {span} days: eras cannot be separated")
    else:
        rate, span = None, 0
    fams = {r.family for r in resolved}
    if resolved and len(fams) < 2:
        problems.append("a single pattern family: family effects cannot be estimated")
    future = [r for r in disc if r.resolved_at and to_ts(r.resolved_at) >= cut]
    return {"n_discoveries": len(disc), "resolved": len(resolved), "unresolved": len(unresolved), "unresolved_share": (len(unresolved) / len(disc)) if disc else 0.0,
            "in_future_of_now": len(future), "survival_rate": rate, "span_days": span, "families": len(fams), "problems": problems,
            "usable": not problems and len(resolved) >= 20}


def ablation_forecast(recs: Sequence[AblationRecord], prior_a: float = 1.0, prior_b: float = 1.0, recency: float = 0.85) -> dict:
    """P(feature family fails its NEXT ablation) with exponentially recency-weighted counts and a Beta(prior_a, prior_b)
    prior. Families whose forecast stays above 0.7 for their last several runs are candidates for removal from the feature
    set; the forecast is a triage aid for the ablation schedule, not a verdict (only a fresh ablation is a verdict)."""
    by: dict = {}
    for r in sorted(recs, key=lambda r: r.resolved_at):
        by.setdefault(r.feature_family, []).append(1.0 if r.failed else 0.0)
    out = {}
    for fam, seq in by.items():
        w = np.array([recency ** (len(seq) - 1 - i) for i in range(len(seq))])
        s = float((w * np.array(seq)).sum())
        n = float(w.sum())
        p = (s + prior_a) / (n + prior_a + prior_b)
        out[fam] = {"runs": len(seq), "p_fail_next": p, "effective_n": n, "candidate_for_removal": bool(len(seq) >= 4 and p > 0.7)}
    return out


def explanation_calibration_curve(recs: Sequence[ExplanationRecord], bins: int = 4) -> list:
    """Stated confidence against whether the explanation was later verified correct, by confidence bin. An explanation
    engine that says '90 percent sure' and is right half the time must have its confidences discounted before anyone uses them."""
    if not recs:
        return []
    return reliability_bins([r.confidence for r in recs], [1 if r.correct() else 0 for r in recs], bins)


def discount_explanation_confidence(recs: Sequence[ExplanationRecord], confidence: float, bins: int = 4, min_bin_n: int = 5) -> float:
    """The confidence to actually use: the observed correctness rate in the historical bin the stated confidence falls in,
    if that bin has >= min_bin_n cases; otherwise the stated value shrunk halfway toward the overall correctness rate."""
    curve = explanation_calibration_curve(recs, bins)
    overall = sum(1 for r in recs if r.correct()) / len(recs) if recs else 0.0
    for row in curve:
        lo, hi = (float(x) for x in row["bin"].split("-"))
        if lo <= confidence <= hi and row["n"] >= min_bin_n:
            return row["observed"]
    return 0.5 * confidence + 0.5 * overall


def learner_ranking(recs: Sequence[LearnerRecord], seed: int = 0, n_boot: int = 500) -> list:
    """Learners ranked by VERIFIED gain (mean over their records, missing verified gains counted as 0: unverified means no
    demonstrated gain) with a bootstrap interval, alongside how much of what they CLAIMED was never verified."""
    by: dict = {}
    for r in recs:
        by.setdefault(r.learner, []).append(r)
    rng = np.random.default_rng(seed)
    out = []
    for name, rows in by.items():
        ver = np.array([r.verified_gain if (r.verified and r.verified_gain is not None) else 0.0 for r in rows], float)
        cl = np.array([r.claimed_gain for r in rows], float)
        boots = ver[rng.integers(0, len(ver), size=(n_boot, len(ver)))].mean(axis=1) if len(ver) else np.zeros(1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out.append({"learner": name, "n": len(rows), "mean_verified": float(ver.mean()), "lo": float(lo), "hi": float(hi),
                    "mean_claimed": float(cl.mean()), "unverified_claim": float(max(cl.mean() - ver.mean(), 0.0)),
                    "demonstrated": bool(lo > 0)})
    return sorted(out, key=lambda r: -r["mean_verified"])


def diff_updates(old: "MetaUpdate", new: "MetaUpdate", min_move: float = 0.10) -> dict:
    """What changed between two meta-learning updates: advice entries that moved by more than `min_move`, entries that
    appeared or vanished, and the change in survival-model skill. Lets a reader tell a real lesson from jitter."""
    def moved(a: Mapping, b: Mapping) -> dict:
        return {k: (a[k], b[k]) for k in a.keys() & b.keys() if abs(a[k] - b[k]) >= min_move}
    ao, an = old.advice, new.advice
    so, sn = old.oos["survival"], new.oos["survival"]
    return {"family_overfit_moved": moved(ao.family_overfit, an.family_overfit), "family_survival_moved": moved(ao.family_survival, an.family_survival),
            "target_yield_moved": moved(ao.target_yield, an.target_yield),
            "appeared": sorted((set(an.family_survival) - set(ao.family_survival))), "vanished": sorted(set(ao.family_survival) - set(an.family_survival)),
            "skill_old": (so.baseline_brier - so.model_brier) if so.n_test else None, "skill_new": (sn.baseline_brier - sn.model_brier) if sn.n_test else None,
            "n_records_old": sum(old.counts.values()), "n_records_new": sum(new.counts.values())}


def scorecard_numbers(update: "MetaUpdate") -> dict:
    """The handful of numbers the learning scorecard (section 47) reads from meta-learning: OOS skill per task, how many
    families are known to overfit, how many learners produce fake improvement, and the explanation accuracy against its baseline."""
    skill = {k: (o.baseline_brier - o.model_brier) if o.n_test else None for k, o in update.oos.items()}
    ex = update.explanations
    return {"oos_skill": skill, "oos_labels": {k: o.label for k, o in update.oos.items()},
            "families_overfit": sum(1 for p in update.families.overfit.values() if p > 0.6),
            "fake_learners": sum(1 for v in update.learners.values() if v["produces_fake_improvement"]),
            "explanation_accuracy": ex["accuracy"], "explanation_beats_majority": ex["beats_majority"],
            "rapid_decay_types": len(update.decay.get("rapid_decay", [])), "records": sum(update.counts.values()), "label": update.label}


# ==================================================================================================================
# Part 5: pairwise family comparison, leave-one-family-out transfer, discovery budgeting
# ==================================================================================================================

def prob_family_better(a: GroupRate, b: GroupRate, prior: tuple, seed: int = 0, n: int = 20000) -> float:
    """P(true survival rate of family a > that of family b) from the two Beta posteriors (shared empirical-Bayes prior).
    A probability, not a verdict: 0.6 means 'lean towards a', and it says so instead of declaring a winner."""
    rng = np.random.default_rng(seed)
    pa = rng.beta(a.successes + prior[0], a.n - a.successes + prior[1], n)
    pb = rng.beta(b.successes + prior[0], b.n - b.successes + prior[1], n)
    return float((pa > pb).mean())


def family_ranking(rep: FamilyReport, seed: int = 0, min_n: int = 5) -> list:
    """Families ordered by shrunk survival with, for each, the probability it beats the median family. Ties in evidence show
    as probabilities near 0.5 rather than as an arbitrary order."""
    fams = [g for g in rep.rates.values() if g.n >= min_n]
    if len(fams) < 2:
        return []
    med = sorted(fams, key=lambda g: g.shrunk)[len(fams) // 2]
    return [{"family": g.key, "n": g.n, "shrunk": g.shrunk, "p_beats_median": prob_family_better(g, med, rep.prior, seed) if g is not med else 0.5}
            for g in sorted(fams, key=lambda g: -g.shrunk)]


def leave_one_family_out(recs: Sequence[DiscoveryRecord], seed: int = 0, l2: float = 2.0, min_family_n: int = 20) -> dict:
    """Do the FEATURE relationships transfer across pattern families? For each family F the survival model is fitted on all
    other families using features only (no family indicator possible: F is unseen) and scored on F against F's own base rate
    (the best a features-free predictor could know). Positive skill in most held-out families means the features carry a
    lesson that generalises; skill only inside seen families is family memorisation, the failure the contract warns about."""
    lab = [r for r in recs if r.survived_oos is not None]
    out: dict[str, Any] = {}
    for fam in sorted({r.family for r in lab}):
        test = [r for r in lab if r.family == fam]
        train = [r for r in lab if r.family != fam]
        if len(test) < min_family_n or len(train) < 40:
            continue
        names = sorted({k for r in train for k in r.features})
        X = np.array([[r.features.get(k, 0.0) for k in names] for r in train], float)
        mu, sd = X.mean(axis=0), np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        y = np.array([1.0 if r.survived_oos else 0.0 for r in train])
        w = fit_logistic((X - mu) / sd, y, l2)
        Xt = (np.array([[r.features.get(k, 0.0) for k in names] for r in test], float) - mu) / sd
        yt = np.array([1 if r.survived_oos else 0 for r in test])
        p = sigmoid(Xt @ w[:-1] + w[-1])
        # a features-only model has no family term, so its intercept reflects the OTHER families' base rate; re-centre on the
        # held-out family's own base rate so the comparison measures the feature signal, not the family's level
        logit = Xt @ w[:-1]
        shift = math.log(max(yt.mean(), 0.02) / max(1 - yt.mean(), 0.02))
        p_c = sigmoid(logit - float(np.mean(logit)) + shift)
        base = yt.mean()
        out[fam] = {"n": len(test), "auc": auc(p, yt), "brier_model": brier(p_c, yt), "brier_base": brier([base] * len(yt), yt),
                    "skill": brier([base] * len(yt), yt) - brier(p_c, yt)}
    skills = [v["skill"] for v in out.values()]
    return {"families": out, "n_tested": len(out), "share_positive": (sum(1 for s in skills if s > 0) / len(skills)) if skills else None,
            "transfers_across_families": bool(skills and sum(1 for s in skills if s > 0) / len(skills) >= 0.75), "label": LABEL}


def discovery_budget(rep: FamilyReport, want_survivors: int, cost_per_test: float = 1.0, min_n: int = 5, default_rate: float | None = None) -> list:
    """How many discoveries must be tested to EXPECT `want_survivors` survivors, by family: ceil(k / shrunk survival), and the
    compute that costs. Families with too little history use `default_rate` (the pooled rate) and are marked as guesses."""
    out = []
    pooled = default_rate if default_rate is not None else rep.base_rate
    for fam, g in sorted(rep.rates.items()):
        known = g.n >= min_n
        rate = g.shrunk if known else pooled
        if rate <= 0:
            continue
        n_tests = int(math.ceil(want_survivors / rate))
        out.append({"family": fam, "rate": rate, "tests_needed": n_tests, "compute": n_tests * cost_per_test, "guess": not known})
    return sorted(out, key=lambda r: r["tests_needed"])


# ==================================================================================================================
# Part 6: advice persistence, staleness and stability across updates
# ==================================================================================================================

def advice_to_json(a: MetaAdvice) -> str:
    return canonical_json(a)


def advice_from_json(text: str) -> MetaAdvice:
    d = json.loads(text)
    known = {k: v for k, v in d.items() if k in MetaAdvice.__dataclass_fields__}
    adv = MetaAdvice(**known)
    errs = adv.check()
    if errs:
        raise ValueError("stored advice invalid: " + "; ".join(errs))
    return adv


def advice_age_days(a: MetaAdvice, now) -> float:
    """Days between the advice's fitted_through date and `now`. Advice fitted at/after `now` is a firewall breach."""
    if not a.fitted_through:
        return float("inf")
    d = (to_ts(now) - to_ts(a.fitted_through)).total_seconds() / 86400.0
    if d < 0:
        raise FirewallBreach(f"advice fitted through {a.fitted_through} is from the future of now={now}")
    return d


def stale_discount(a: MetaAdvice, now, half_life_days: float = 365.0) -> MetaAdvice:
    """Advice loses trust as it ages: n_observations is scaled by 0.5 ** (age / half_life). A year-old lesson counts as half
    as many observations; advice with no fitted date counts as none. The entries stay (they are still information), but
    `trust()` falls, so the policy leans on them less."""
    age = advice_age_days(a, now)
    if math.isinf(age):
        return replace(a, n_observations=0)
    return replace(a, n_observations=int(a.n_observations * 0.5 ** (age / half_life_days)))


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def advice_stability(old: MetaAdvice, new: MetaAdvice) -> dict:
    """How stable are the lessons between two updates? Spearman correlation of family survival over the families both know, and
    the share of overfit flags (> 0.6) that agree. Unstable advice (low correlation between consecutive fits on overlapping
    data) is noise the policy should not be steered by."""
    fams = sorted(set(old.family_survival) & set(new.family_survival))
    rho = spearman([old.family_survival[f] for f in fams], [new.family_survival[f] for f in fams])
    flags = [(old.family_overfit.get(f, 0.0) > 0.6) == (new.family_overfit.get(f, 0.0) > 0.6) for f in fams]
    return {"n_families": len(fams), "spearman": rho, "overfit_flag_agreement": (sum(flags) / len(flags)) if flags else None,
            "stable": bool(rho is not None and rho >= 0.7 and (not flags or sum(flags) / len(flags) >= 0.8))}


def combine_advice(a: MetaAdvice, b: MetaAdvice) -> MetaAdvice:
    """Merge advice fitted on two disjoint record sets (e.g. two research streams) by observation-weighted average per key.
    Keys known to one side only are kept as they are; the merged trust follows the summed observations. The label is the
    weaker of the two (a merge is never MORE validated than its parts)."""
    wa, wb = max(a.n_observations, 0), max(b.n_observations, 0)

    def mix(x: Mapping, y: Mapping) -> dict:
        out = {}
        for k in set(x) | set(y):
            if k in x and k in y and wa + wb > 0:
                out[k] = (x[k] * wa + y[k] * wb) / (wa + wb)
            else:
                out[k] = x[k] if k in x else y[k]
        return out
    order = [ValidationLabel.VALIDATED.value, ValidationLabel.NOT_VALIDATED.value, ValidationLabel.INSUFFICIENT_EVIDENCE.value, ValidationLabel.FAILED_VALIDATION.value]
    weaker = max((a.oos_label, b.oos_label), key=lambda l: order.index(l) if l in order else len(order))
    return MetaAdvice(mix(a.family_overfit, b.family_overfit), mix(a.family_survival, b.family_survival), mix(a.target_yield, b.target_yield),
                      mix(a.context_transfer, b.context_transfer), mix(a.explanation_precision, b.explanation_precision),
                      max(a.fitted_through, b.fitted_through), wa + wb, weaker)


def to_markdown(u: "MetaUpdate") -> str:
    """The meta-learning update as a Markdown page: records, OOS verdicts with reasons, overfit families, fake learners."""
    lines = [f"# Meta-learning as of {u.now}", "", f"**{u.label}** - {sum(u.counts.values())} records, store `{u.store_hash}`, code `{u.code_hash}`", "",
             "## Out-of-sample evaluation of meta-learning itself", "", "| task | n test | model Brier | baseline | verdict |", "|---|---|---|---|---|"]
    for k, o in u.oos.items():
        lines.append(f"| {k} | {o.n_test} | {o.model_brier:.4f} | {o.baseline_brier:.4f} | {o.label}: {o.reason} |")
    lines += ["", "## Families", "", "| family | n | survival (shrunk) | overfit |", "|---|---|---|---|"]
    for f, g in sorted(u.families.rates.items(), key=lambda kv: kv[1].shrunk):
        lines.append(f"| {f} | {g.n} | {g.shrunk:.2f} [{g.lo:.2f}, {g.hi:.2f}] | {u.families.overfit.get(f, 0.0):.2f} |")
    fake = sorted(k for k, v in u.learners.items() if v["produces_fake_improvement"])
    lines += ["", f"Learners producing fake improvement: {', '.join(fake) if fake else 'none identified'}",
              f"Store health: {'usable' if u.health.get('usable') else 'NOT usable'} {u.health.get('problems', [])}"]
    return "\n".join(lines)


def survival_by_complexity(recs: Sequence[DiscoveryRecord], feature: str = "n_conditions", min_n: int = 8) -> dict:
    """Do more complex discoveries overfit more (section 40)? Survival rate per value of a complexity feature (e.g. the number
    of conditions in a pattern), plus the slope of the logistic relation on the standardised feature. A significantly negative
    slope is the empirical case for a complexity penalty."""
    lab = [r for r in recs if r.survived_oos is not None and feature in r.features]
    if len(lab) < 2 * min_n:
        return {"verdict": "INSUFFICIENT", "n": len(lab)}
    x = np.array([r.features[feature] for r in lab], float)
    y = np.array([1.0 if r.survived_oos else 0.0 for r in lab])
    by = {}
    for v in sorted(set(x.tolist())):
        m = x == v
        if m.sum() >= min_n:
            by[v] = {"n": int(m.sum()), "rate": float(y[m].mean())}
    sd = x.std()
    if sd < 1e-9 or y.min() == y.max():
        return {"verdict": "NO_VARIATION", "n": len(lab), "by_value": by}
    w = fit_logistic(((x - x.mean()) / sd).reshape(-1, 1), y, l2=1.0)
    return {"verdict": "OK", "n": len(lab), "by_value": by, "slope_per_sd": float(w[0]), "complexity_hurts": bool(w[0] < -0.2)}


def top_lessons(u: "MetaUpdate", n: int = 5) -> list:
    """The strongest things this update learned, as plain sentences ordered by evidence, for the owner. Every sentence carries
    its record count and its label, so a lesson from 12 records never reads like one from 12,000."""
    out = []
    for f, p in sorted(u.families.overfit.items(), key=lambda kv: -kv[1]):
        g = u.families.rates.get(f)
        if p > 0.6 and g is not None:
            out.append((g.n, f"family {f} overfits: only {g.successes} of {g.n} discoveries survived out of sample ({g.shrunk:.0%} shrunk)"))
    for name, v in u.learners.items():
        if v["produces_fake_improvement"]:
            out.append((v["claims"], f"learner {name} claimed improvement {v['claims']} times and {v['fake']} were not verified"))
    for k, f in u.decay.get("fits", {}).items():
        if k in u.decay.get("rapid_decay", []):
            out.append((f.n, f"memory type {k} decays fast: half-life about {f.half_life_days:.0f} days (n={f.n})"))
    for name, o in u.oos.items():
        if o.n_test and o.beats_baseline:
            out.append((o.n_test, f"{name} predictions beat the base rate out of sample (Brier {o.model_brier:.3f} vs {o.baseline_brier:.3f}, n={o.n_test})"))
    out.sort(key=lambda t: -t[0])
    return [f"{text} [{u.label}]" for _, text in out[:n]]
