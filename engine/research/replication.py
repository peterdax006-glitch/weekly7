"""Research replication requirements (contract C66 section 32; C62 sections 26 and 45; checklist: replication system).
STATUS: IMPLEMENTED - NOT VALIDATED (unit-tested on planted synthetic effects only; no real data was touched).

A discovery is not trustworthy because one experiment found it. This module decides whether a discovery has been REPLICATED
and, just as important, refuses to let anything less change the trading system.

A replication run earns some of six freshness axes against the original evidence:
    FRESH_PERIOD       its window is clear of the original window plus the label horizon (no shared outcomes)
    FRESH_STOCKS       at most a small share of its names were in the original evidence
    FRESH_SEED         a seed nobody used before (weak on its own: same data, different optimiser luck)
    FRESH_REGIME       a market regime the original never saw
    CONTROL_COMPARISON a matched control was run alongside and the discovery must beat it
    INDEPENDENT_CODE   a different implementation of the same idea, not a rerun of the same code
Each run is judged with a block bootstrap interval, a winner's-curse-adjusted retention target (a discovery picked as the best
of many searched tests is EXPECTED to shrink), a small-telescopes test (was the replication able to contradict the original at
all?) and a power check, so an under-powered miss is INCONCLUSIVE, never FAILED. Runs are grouped into independent
replications (two views of the same window/stocks/seed count once), pooled by DerSimonian-Laird random effects, and counted
against ALL attempts, so retrying until one attempt succeeds cannot pass. `authorize_system_change` is the single door: nothing
but a REPLICATED discovery, whose evidence year is not being replayed in disguise (the same-year rerun leak), may change the
system. Thresholds are learned from resolved discoveries by `learn_policy` and documented by `requirements_document`; with too
little history the documented prior stays and says so. Builds on engine.learning.promotion (block bootstrap, t, Sidak p) and
engine.research.core (MaturedRecord, Namespace); it is research-side (MATURED_RESEARCH_STATE) code and reaches the trader only
through MaturedRecord.gate(now)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engine.learning import promotion as PR
from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, canonical_json, require_past, stable_hash)
from engine.research.core import MaturedRecord, Namespace

SECTION = "C66 section 32"
SYSTEM_CHANGING = frozenset({"REPLICATED"})


class Axis(_StrEnum):
    PERIOD = "FRESH_PERIOD"
    STOCKS = "FRESH_STOCKS"
    SEED = "FRESH_SEED"
    REGIME = "FRESH_REGIME"
    CONTROL = "CONTROL_COMPARISON"
    CODE = "INDEPENDENT_CODE"


FRESH_DATA_AXES = (Axis.PERIOD, Axis.STOCKS, Axis.REGIME)        # axes that put genuinely new outcomes in front of the idea


class Status(_StrEnum):
    UNREPLICATED = "UNREPLICATED"          # no valid replication run exists
    INCONCLUSIVE = "INCONCLUSIVE"          # runs exist, none was decisive (usually under-powered)
    PARTIAL = "PARTIALLY_REPLICATED"       # supported, but the requirements are not all met yet
    REPLICATED = "REPLICATED"              # the only status that may change the system
    CONTRADICTED = "CONTRADICTED"          # independent runs both support and refute it: unexplained, not promotable
    FAILED = "FAILED_TO_REPLICATE"         # powered, fresh runs refuted it


class Outcome(_StrEnum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


class LuckyExperimentError(FirewallBreach):
    """A single, unreplicated (or same-year) result tried to change the system. Never caught-and-continued."""


# ------------------------------------------------------------------------------------------------ policy
@dataclass(frozen=True)
class ReplicationPolicy:
    """Every threshold in one validated, hashable place. `sources` records, per threshold, whether it is a documented PRIOR or was
    LEARNED from resolved discoveries (learn_policy), so a threshold nobody derived cannot pass as one somebody did."""
    min_independent: int = 2                 # independent fresh-data replications that must support the discovery
    min_periods: int = 12                    # periods a run needs before it can speak at all
    min_fresh_axes: int = 2                  # distinct fresh-data axes covered across the supporting runs (PERIOD is mandatory)
    require_control: bool = True
    min_retention_floor: float = 0.15        # replication effect / original effect can never be required below this
    retention_fraction: float = 0.5          # ... and is otherwise this share of the winner's-curse-adjusted expectation
    max_stock_overlap: float = 0.25          # share of a run's names allowed to be in the original evidence
    max_group_window_overlap: float = 0.5    # runs overlapping this much in time AND stocks are one independent replication
    max_group_stock_overlap: float = 0.5
    ci_level: float = 0.90                   # two-sided level whose lower bound must exceed zero (one-sided 5%)
    alpha: float = 0.05
    power: float = 0.80
    bootstrap_n: int = 400
    block: int = 3
    min_success_share: float = 0.6           # supporting / valid independent attempts: retrying until one works cannot pass
    contradiction_z: float = 2.58            # replication differs from the original by this many joint SE = "decayed"
    max_i2: float = 0.75                     # pooled heterogeneity above this: results disagree too much to be one effect
    fail_after: int = 1                      # powered independent refutations (with no support) that mark FAILED
    max_run_age_days: int = 3650
    sources: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not math.isfinite(v):
                errs.append(f"{f.name} is not finite")
        if self.min_independent < 2:
            errs.append("min_independent must be >= 2: one experiment can never be its own replication")
        if not 1 <= self.min_fresh_axes <= len(FRESH_DATA_AXES):
            errs.append("min_fresh_axes must be between 1 and 3")
        if self.min_periods < 4:
            errs.append("min_periods must be >= 4")
        if not 0 < self.alpha < 0.5 or not 0.5 < self.ci_level < 1 or not 0.5 <= self.power < 1:
            errs.append("alpha in (0,0.5), ci_level in (0.5,1), power in [0.5,1)")
        if not 0 <= self.min_retention_floor <= 1 or not 0 < self.retention_fraction <= 1:
            errs.append("retention thresholds must be in [0,1]")
        for name in ("max_stock_overlap", "max_group_window_overlap", "max_group_stock_overlap", "min_success_share"):
            if not 0 <= getattr(self, name) <= 1:
                errs.append(f"{name} must be in [0,1]")
        if self.fail_after < 1:
            errs.append("fail_after must be >= 1")
        return errs

    def digest(self) -> str:
        return stable_hash(self)

    def source_of(self, name: str) -> str:
        return self.sources.get(name, "PRIOR (documented default, never fitted to a result)")


DEFAULT_POLICY = ReplicationPolicy()


# ------------------------------------------------------------------------------------------------ records
def _window(w) -> tuple[dt.date, dt.date]:
    a, b = w
    lo, hi = as_date(a), as_date(b)
    if hi < lo:
        raise ValueError(f"window ends before it starts: {w!r}")
    return lo, hi


def window_overlap(a, b, pad_days: int = 0) -> float:
    """Shared share of the SHORTER window (a padded by pad_days each side). 0 = disjoint, 1 = one contains the other."""
    (s1, e1), (s2, e2) = _window(a), _window(b)
    s1, e1 = s1 - dt.timedelta(days=pad_days), e1 + dt.timedelta(days=pad_days)
    shared = (min(e1, e2) - max(s1, s2)).days + 1
    shorter = min((e1 - s1).days, (e2 - s2).days) + 1
    return max(0.0, min(1.0, shared / max(shorter, 1)))


def window_years(w) -> frozenset[int]:
    lo, hi = _window(w)
    return frozenset(range(lo.year, hi.year + 1))


def stock_overlap(run_stocks: frozenset, base: frozenset) -> float:
    """Share of the run's names that were already in `base` (1.0 = nothing fresh)."""
    return len(run_stocks & base) / len(run_stocks) if run_stocks else 1.0


@dataclass(frozen=True)
class Discovery:
    """The original evidence of a discovery: what it saw, so a replication can be judged against exactly that."""
    discovery_id: str
    effect: float                            # mean per-period effect on the original evidence (higher = better)
    sd: float                                # sd of the per-period effects
    n_periods: int
    window: tuple[str, str]
    horizon_days: int
    stocks: frozenset
    seeds: tuple[int, ...]
    regimes: frozenset
    code_hash: str
    data_hash: str
    matured_at: str
    n_tests_searched: int = 1
    implementation: str = "primary"

    def validate(self) -> list[str]:
        errs = []
        if not self.discovery_id:
            errs.append("discovery_id missing")
        if not (math.isfinite(self.effect) and math.isfinite(self.sd)) or self.sd <= 0:
            errs.append("effect must be finite and sd positive")
        if self.n_periods < 2:
            errs.append("n_periods < 2")
        if self.n_tests_searched < 1:
            errs.append("n_tests_searched must be >= 1 (the size of the search that produced it)")
        if self.horizon_days < 0:
            errs.append("horizon_days negative")
        if not self.stocks:
            errs.append("no stocks recorded: freshness of stocks cannot be judged")
        try:
            _window(self.window)
        except (ValueError, TypeError) as e:
            errs.append(str(e))
        return errs

    @property
    def se(self) -> float:
        return self.sd / math.sqrt(self.n_periods)

    @property
    def t(self) -> float:
        return self.effect / self.se

    @property
    def years(self) -> frozenset[int]:
        return window_years(self.window)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["stocks"], d["regimes"] = sorted(self.stocks), sorted(self.regimes)
        d["window"], d["seeds"] = list(self.window), list(self.seeds)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "Discovery":
        d = dict(d)
        d["stocks"], d["regimes"] = frozenset(d["stocks"]), frozenset(d["regimes"])
        d["window"], d["seeds"] = tuple(d["window"]), tuple(d["seeds"])
        return cls(**d)


@dataclass(frozen=True)
class ReplicationRun:
    """One attempt to reproduce a discovery on evidence it has not seen. `effects` are per-period, in time order; `control_effects`
    (optional) are the matched control's per-period effects on the SAME periods."""
    run_id: str
    discovery_id: str
    window: tuple[str, str]
    stocks: frozenset
    seed: int
    regime: str
    effects: tuple[float, ...]
    control_effects: tuple[float, ...] | None
    code_hash: str
    data_hash: str
    matured_at: str
    implementation: str = "primary"

    def validate(self, now=None) -> list[str]:
        errs = []
        if not self.run_id or not self.discovery_id:
            errs.append("run_id and discovery_id are required")
        e = np.asarray(self.effects, dtype=float)
        if e.size == 0:
            errs.append("no effects")
        elif not np.isfinite(e).all():
            errs.append("effects contain non-finite values: NaN is not a zero return")
        if self.control_effects is not None:
            c = np.asarray(self.control_effects, dtype=float)
            if c.size != e.size:
                errs.append("control_effects must cover exactly the same periods as effects")
            elif not np.isfinite(c).all():
                errs.append("control_effects contain non-finite values")
        if not self.stocks:
            errs.append("no stocks recorded")
        try:
            _window(self.window)
        except (ValueError, TypeError) as ex:
            errs.append(str(ex))
        if now is not None:
            try:
                require_past(self.matured_at, now, f"replication run {self.run_id}")
            except FirewallBreach as ex:
                errs.append(str(ex))
        return errs

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["stocks"] = sorted(self.stocks)
        d["window"], d["effects"] = list(self.window), list(self.effects)
        d["control_effects"] = None if self.control_effects is None else list(self.control_effects)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "ReplicationRun":
        d = dict(d)
        d["stocks"], d["window"], d["effects"] = frozenset(d["stocks"]), tuple(d["window"]), tuple(d["effects"])
        if d.get("control_effects") is not None:
            d["control_effects"] = tuple(d["control_effects"])
        return cls(**d)


@dataclass(frozen=True)
class AxisCheck:
    axis: Axis
    earned: bool
    detail: str
    measure: float | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outcome: Outcome
    axes: tuple[AxisCheck, ...]
    fresh_data: bool                         # earned at least one of PERIOD / STOCKS / REGIME
    n: int
    effect: float
    se: float
    t: float
    ci_low: float
    ci_high: float
    retention: float
    required_retention: float
    beats_control: bool | None
    control_gain: float | None
    powered: bool
    required_n: float
    decayed: bool                            # significantly smaller than the original (a warning, not a verdict)
    reasons: tuple[str, ...]

    def earned(self, axis: Axis) -> bool:
        return any(a.axis == axis and a.earned for a in self.axes)

    def earned_axes(self) -> tuple[Axis, ...]:
        return tuple(a.axis for a in self.axes if a.earned)


# ------------------------------------------------------------------------------------------------ statistics
def _z(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9); the repo has no scipy-free inverse."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_z(k: int) -> float:
    """Blom approximation of E[max of k standard normals]: the luck a best-of-k search banks before any real effect."""
    k = max(int(k), 1)
    return 0.0 if k == 1 else _z((k - 0.375) / (k + 0.25))


def adjusted_effect(d: Discovery) -> float:
    """Winner's-curse-adjusted original effect. Positive-part shrinkage on the t scale: if the observed t is no larger than the
    best a search of that size would produce from pure noise, nothing is left; a very large t barely shrinks."""
    t = d.t
    if t <= 0:
        return 0.0
    z = expected_max_z(d.n_tests_searched)
    return float(d.effect * max(0.0, 1.0 - (z / t) ** 2))


def lucky_probability(d: Discovery) -> float:
    """Probability that a search of n_tests_searched null hypotheses produces a best t at least as large as this one."""
    p = PR.one_sided_p(d.t)
    return PR.adjusted_p(p, d.n_tests_searched, "sidak")


def required_n(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Periods a one-sided test needs to detect `effect` with the given power. inf when the effect is not positive."""
    if not (effect > 0 and sd > 0):
        return math.inf
    return float(((_z(1 - alpha) + _z(power)) * sd / effect) ** 2)


def small_telescope_effect(d: Discovery, alpha: float = 0.05, power: float = 0.33) -> float:
    """The smallest effect the ORIGINAL evidence had only 33% power to detect (Simonsohn 2015). A replication that is confidently
    below it has shown the original could not have measured a real effect of that size: a decisive refutation."""
    return float(d.se * (_z(1 - alpha) + _z(power)))


def dersimonian_laird(effects: Sequence[float], ses: Sequence[float]) -> dict:
    """Random-effects pool. Returns pooled effect, its SE and 95% CI, tau2 and I2 (share of variation that is heterogeneity)."""
    y, s = np.asarray(effects, dtype=float), np.asarray(ses, dtype=float)
    k = y.size
    if k == 0:
        return {"k": 0, "effect": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "tau2": 0.0, "i2": 0.0}
    w = 1.0 / np.maximum(s, 1e-12) ** 2
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1.0 / (np.maximum(s, 1e-12) ** 2 + tau2)
    eff = float((ws * y).sum() / ws.sum())
    se = float(math.sqrt(1.0 / ws.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if k > 1 and q > 0 else 0.0
    return {"k": int(k), "effect": eff, "se": se, "ci_low": eff - 1.96 * se, "ci_high": eff + 1.96 * se, "tau2": float(tau2), "i2": float(i2)}


def stouffer(p_values: Sequence[float]) -> float:
    """Combined one-sided p of independent runs (Stouffer). 1.0 for no runs."""
    ps = [min(max(float(p), 1e-12), 1 - 1e-12) for p in p_values]
    if not ps:
        return 1.0
    z = sum(-_z(p) for p in ps) / math.sqrt(len(ps))
    return float(1.0 - 0.5 * math.erfc(-z / math.sqrt(2.0)))


# ------------------------------------------------------------------------------------------------ freshness and judgement
def classify_axes(d: Discovery, r: ReplicationRun, policy: ReplicationPolicy = DEFAULT_POLICY,
                  prior_runs: Sequence[ReplicationRun] = ()) -> tuple[AxisCheck, ...]:
    """Which of the six freshness axes this run earns against the original evidence (and, for the seed, earlier runs)."""
    ov = window_overlap(d.window, r.window, pad_days=d.horizon_days)
    so = stock_overlap(r.stocks, d.stocks)
    used_seeds = set(d.seeds) | {p.seed for p in prior_runs if p.run_id != r.run_id}
    ctl_ok = r.control_effects is not None and len(r.control_effects) == len(r.effects) and len(r.effects) >= policy.min_periods
    return (
        AxisCheck(Axis.PERIOD, ov == 0.0, f"window overlap with original (+{d.horizon_days}d label horizon) {ov:.0%}", ov),
        AxisCheck(Axis.STOCKS, so <= policy.max_stock_overlap, f"{so:.0%} of the run's names were in the original evidence", so),
        AxisCheck(Axis.SEED, r.seed not in used_seeds, "seed already used" if r.seed in used_seeds else "unused seed"),
        AxisCheck(Axis.REGIME, bool(r.regime) and r.regime not in d.regimes, f"regime {r.regime!r} " + ("already seen" if r.regime in d.regimes else "new")),
        AxisCheck(Axis.CONTROL, bool(ctl_ok), "matched control on the same periods" if ctl_ok else "no usable matched control"),
        AxisCheck(Axis.CODE, bool(r.implementation) and r.implementation != d.implementation and r.code_hash != d.code_hash,
                  f"implementation {r.implementation!r} vs original {d.implementation!r}"),
    )


def judge_run(d: Discovery, r: ReplicationRun, policy: ReplicationPolicy = DEFAULT_POLICY, now=None,
              prior_runs: Sequence[ReplicationRun] = ()) -> RunResult:
    """Judge one run. INVALID (malformed, unmatured, or a duplicate of the original), SUPPORTS, REFUTES (powered and confidently
    below the small-telescopes effect), or INCONCLUSIVE. Deterministic: the bootstrap is seeded by the run's own seed."""
    axes = classify_axes(d, r, policy, prior_runs)
    fresh = any(a.earned for a in axes if a.axis in FRESH_DATA_AXES)
    problems = r.validate(now)
    if r.discovery_id != d.discovery_id:
        problems.append("run belongs to a different discovery")
    if r.data_hash == d.data_hash and r.code_hash == d.code_hash and r.seed in d.seeds and not fresh:
        problems.append("byte-identical rerun of the original: reproducibility, not replication")
    if problems:
        nan = float("nan")
        return RunResult(r.run_id, Outcome.INVALID, axes, fresh, len(r.effects), nan, nan, nan, nan, nan, nan, nan, None, None,
                         False, math.inf, False, tuple(problems))
    x = np.asarray(r.effects, dtype=float)
    n = int(x.size)
    eff, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("inf")
    t = PR.t_stat(x)
    lo, hi = PR.block_bootstrap_ci(x, r.seed, policy.bootstrap_n, policy.block, policy.ci_level)
    adj = adjusted_effect(d)
    shrink = adj / d.effect if d.effect > 0 else 0.0
    need_ret = max(policy.min_retention_floor, policy.retention_fraction * shrink)
    retention = eff / d.effect if d.effect > 0 else float("nan")
    beats, gain = None, None
    if r.control_effects is not None and len(r.control_effects) == n:
        diff = x - np.asarray(r.control_effects, dtype=float)
        gain = float(diff.mean())
        clo, _ = PR.block_bootstrap_ci(diff, r.seed + 1, policy.bootstrap_n, policy.block, policy.ci_level)
        beats = bool(math.isfinite(clo) and clo > 0)
    n_need = required_n(adj, max(sd, d.sd, 1e-12), policy.alpha, policy.power)
    powered = n >= min(n_need, 10 ** 9) and n >= policy.min_periods
    telescope = small_telescope_effect(d, policy.alpha)
    joint = math.sqrt(d.se ** 2 + (se ** 2 if math.isfinite(se) else 0.0))
    decayed = bool(joint > 0 and (d.effect - eff) / joint >= policy.contradiction_z)
    reasons = []
    supports = (math.isfinite(lo) and lo > 0 and math.isfinite(retention) and retention >= need_ret)
    if not (math.isfinite(lo) and lo > 0):
        reasons.append(f"lower {policy.ci_level:.0%} bound {lo:+.5f} does not exceed zero")
    if not (math.isfinite(retention) and retention >= need_ret):
        reasons.append(f"retention {retention:.2f} below required {need_ret:.2f}")
    if beats is False and policy.require_control:
        supports = False
        reasons.append("does not beat the matched control")
    if beats is None and policy.require_control and supports:
        reasons.append("no matched control: supportive but not controlled")
    refutes = bool(not supports and powered and math.isfinite(hi) and hi < telescope) or bool(not supports and powered and eff <= 0 and math.isfinite(hi) and hi <= 0)
    if supports:
        outcome = Outcome.SUPPORTS
    elif refutes:
        outcome = Outcome.REFUTES
        reasons.append(f"powered ({n} >= {n_need:.0f} periods) and upper bound {hi:+.5f} is below the small-telescopes effect {telescope:+.5f}")
    else:
        outcome = Outcome.INCONCLUSIVE
        reasons.append(f"under-powered ({n} periods, {n_need:.0f} needed): a miss here proves nothing" if not powered else "powered but not decisive")
    if not fresh:
        reasons.append("earns no fresh-data axis: cannot count toward replication")
    return RunResult(r.run_id, outcome, axes, fresh, n, eff, se, t, lo, hi, retention, need_ret, beats, gain, powered, n_need, decayed, tuple(reasons))


# ------------------------------------------------------------------------------------------------ independence
def independence_groups(runs: Sequence[ReplicationRun], policy: ReplicationPolicy = DEFAULT_POLICY) -> list[list[str]]:
    """Runs that are the same experiment seen twice collapse into one group: (same window AND same seed) or (large window overlap AND
    large stock overlap). Union-find, deterministic order. Counting duplicates twice is how one lucky window fakes a replication."""
    ordered = sorted(runs, key=lambda r: r.run_id)
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            wo = window_overlap(a.window, b.window)
            so = max(stock_overlap(a.stocks, b.stocks), stock_overlap(b.stocks, a.stocks))
            same = (tuple(a.window) == tuple(b.window) and a.seed == b.seed)
            if same or (wo >= policy.max_group_window_overlap and so >= policy.max_group_stock_overlap):
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, r in enumerate(ordered):
        groups.setdefault(find(i), []).append(r.run_id)
    return sorted(groups.values())


# ------------------------------------------------------------------------------------------------ assessment
@dataclass(frozen=True)
class ReplicationAssessment:
    discovery_id: str
    status: Status
    now: str
    n_runs: int
    n_valid_groups: int                      # independent valid attempts (support + refute + inconclusive)
    n_supporting: int                        # independent fresh-data groups that support it
    n_refuting: int
    success_share: float
    axes_covered: tuple[str, ...]
    axes_missing: tuple[str, ...]
    requirements: Mapping[str, bool]
    pooled: Mapping[str, float]
    combined_p: float
    lucky_probability: float
    adjusted_effect: float
    run_results: tuple[RunResult, ...]
    reasons: tuple[str, ...]
    policy_digest: str
    evidence_digest: str

    @property
    def may_change_system(self) -> bool:
        return self.status.value in SYSTEM_CHANGING

    def digest(self) -> str:
        return stable_hash([self.discovery_id, self.status, self.now, self.n_supporting, self.n_refuting, self.axes_covered,
                            self.policy_digest, self.evidence_digest])

    def trader_safe_payload(self) -> dict:
        """Identity-free summary: no dates, tickers or years. Safe to hand to the curator for release through MaturedRecord.gate."""
        return {"status": self.status.value, "supporting": self.n_supporting, "refuting": self.n_refuting,
                "attempts": self.n_valid_groups, "axes": list(self.axes_covered), "pooled_effect": float(self.pooled.get("effect", float("nan"))),
                "pooled_ci_low": float(self.pooled.get("ci_low", float("nan"))), "i2": float(self.pooled.get("i2", 0.0)),
                "lucky_probability": self.lucky_probability}


def assess(d: Discovery, runs: Sequence[ReplicationRun], now, policy: ReplicationPolicy = DEFAULT_POLICY) -> ReplicationAssessment:
    """The status of one discovery as of `now`. Runs that had not matured by `now` are a FirewallBreach, never quietly skipped:
    a caller that hands the future to the assessor has a bug the assessment must not hide."""
    errs = policy.validate() + d.validate()
    if errs:
        raise ValueError(f"cannot assess: {errs}")
    require_past(d.matured_at, now, f"discovery {d.discovery_id}")
    mine = [r for r in runs if r.discovery_id == d.discovery_id]
    for r in mine:
        require_past(r.matured_at, now, f"replication run {r.run_id}")
    results: list[RunResult] = []
    for i, r in enumerate(sorted(mine, key=lambda r: (r.matured_at, r.run_id))):
        results.append(judge_run(d, r, policy, now, prior_runs=[p for p in sorted(mine, key=lambda p: (p.matured_at, p.run_id))[:i]]))
    by_id = {r.run_id: r for r in mine}
    valid = [x for x in results if x.outcome != Outcome.INVALID]
    groups = independence_groups([by_id[x.run_id] for x in valid], policy)
    res_by = {x.run_id: x for x in valid}
    supporting, refuting, attempts = [], [], 0
    for g in groups:
        members = [res_by[i] for i in g]
        attempts += 1
        best = next((m for m in members if m.outcome == Outcome.SUPPORTS and m.fresh_data), None)
        if best is not None:
            supporting.append(best)
        elif any(m.outcome == Outcome.REFUTES and m.fresh_data for m in members):
            refuting.append(next(m for m in members if m.outcome == Outcome.REFUTES and m.fresh_data))
    covered = sorted({a.value for s in supporting for a in s.earned_axes() if a in FRESH_DATA_AXES})
    missing = sorted(a.value for a in FRESH_DATA_AXES if a.value not in covered)
    any_control = any(s.beats_control for s in supporting)
    pooled = dersimonian_laird([s.effect for s in supporting], [s.se for s in supporting])
    p_comb = stouffer([PR.one_sided_p(s.t) for s in supporting]) if supporting else 1.0
    share = len(supporting) / attempts if attempts else 0.0
    req = {
        "enough_independent_replications": len(supporting) >= policy.min_independent,
        "period_is_fresh": Axis.PERIOD.value in covered,
        "enough_fresh_axes": len(covered) >= policy.min_fresh_axes,
        "control_beaten": any_control or not policy.require_control,
        "no_powered_refutation": not refuting,
        "success_share_ok": share >= policy.min_success_share,
        "pooled_positive": bool(math.isfinite(pooled["ci_low"]) and pooled["ci_low"] > 0),
        "heterogeneity_ok": pooled["i2"] <= policy.max_i2,
    }
    reasons = [f"{x.run_id}: invalid - {'; '.join(x.reasons)}" for x in results if x.outcome == Outcome.INVALID]
    if not mine:
        status = Status.UNREPLICATED
    elif not valid:
        status = Status.UNREPLICATED
        reasons.append("every submitted run was invalid")
    elif refuting and not supporting and len(refuting) >= policy.fail_after:
        status = Status.FAILED
    elif refuting and supporting:
        status = Status.CONTRADICTED
        reasons.append(f"{len(supporting)} independent runs support it and {len(refuting)} powered runs refute it: unexplained")
    elif all(req.values()):
        status = Status.REPLICATED
    elif supporting:
        status = Status.PARTIAL
        reasons.extend(f"requirement not met: {k}" for k, v in req.items() if not v)
    else:
        status = Status.INCONCLUSIVE
        reasons.append("no run was decisive")
    return ReplicationAssessment(d.discovery_id, status, str(as_date(now)), len(mine), attempts, len(supporting), len(refuting), share,
                                 tuple(covered), tuple(missing), req, pooled, p_comb, lucky_probability(d), adjusted_effect(d),
                                 tuple(results), tuple(reasons), policy.digest(),
                                 stable_hash([d.to_dict()] + [r.to_dict() for r in sorted(mine, key=lambda r: r.run_id)]))


# ------------------------------------------------------------------------------------------------ the single door
@dataclass(frozen=True)
class ChangeAuthorization:
    change_id: str
    discovery_ids: tuple[str, ...]
    now: str
    assessment_digests: tuple[str, ...]

    def digest(self) -> str:
        return stable_hash(self)


def same_year_conflict(discovery: Discovery, runs: Sequence[ReplicationRun], replayed_years: Sequence[int]) -> list[int]:
    """Real years of the evidence (original + replications) that are being replayed in disguise right now. Research filed under a
    real year must never be released while that year is being replayed: that is the same-year rerun leak (RESEARCH_MAPPING rule)."""
    yrs = set(discovery.years)
    for r in runs:
        if r.discovery_id == discovery.discovery_id:
            yrs |= window_years(r.window)
    return sorted(yrs & {int(y) for y in replayed_years})


def authorize_system_change(change_id: str, discoveries: Mapping[str, Discovery], runs: Sequence[ReplicationRun], now,
                            policy: ReplicationPolicy = DEFAULT_POLICY, replayed_years: Sequence[int] = ()) -> ChangeAuthorization:
    """Raise LuckyExperimentError unless EVERY discovery behind the change is REPLICATED as of `now` and none of its evidence years
    is being replayed in disguise. A change resting on no discovery is refused too: unexplained change is also a lucky change."""
    if not discoveries:
        raise LuckyExperimentError(f"change {change_id} names no discovery: nothing justifies it")
    digests = []
    for did in sorted(discoveries):
        d = discoveries[did]
        a = assess(d, runs, now, policy)
        if not a.may_change_system:
            raise LuckyExperimentError(f"change {change_id}: discovery {did} is {a.status.value}, not REPLICATED "
                                       f"({a.n_supporting} supporting of {a.n_valid_groups} attempts; {'; '.join(a.reasons[:2]) or 'requirements unmet'})")
        clash = same_year_conflict(d, runs, replayed_years)
        if clash:
            raise LuckyExperimentError(f"change {change_id}: discovery {did} rests on year(s) {clash} that are being replayed in disguise")
        digests.append(a.digest())
    return ChangeAuthorization(change_id, tuple(sorted(discoveries)), str(as_date(now)), tuple(digests))


def assessment_record(a: ReplicationAssessment, d: Discovery, created_real: str) -> MaturedRecord:
    """The assessment as a research-world fact that can reach the trader only through MaturedRecord.gate(now)."""
    prov = Provenance(created_real=created_real, learned_at=d.matured_at, code_hash=d.code_hash, data_hash=d.data_hash,
                      experiment_id=d.discovery_id, run_id=a.digest(), seed=d.seeds[0] if d.seeds else 0,
                      outcomes_seen_through=d.matured_at)
    return MaturedRecord("REPL-" + a.digest(), d.matured_at, a.trader_safe_payload(), prov, Namespace.MATURED_RESEARCH)


# ------------------------------------------------------------------------------------------------ planning the next replication
@dataclass(frozen=True)
class ReplicationRequest:
    request_id: str
    discovery_id: str
    window: tuple[str, str]
    stocks: frozenset
    seed: int
    regime: str
    want_control: bool
    axes_sought: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Pools:
    """What fresh evidence exists to replicate on. windows: (start, end, regime); stocks: opaque ids."""
    windows: tuple[tuple[str, str, str], ...]
    stocks: tuple[str, ...]
    stocks_per_run: int = 30


def plan_next(d: Discovery, runs: Sequence[ReplicationRun], pools: Pools, now, policy: ReplicationPolicy = DEFAULT_POLICY,
              seed: int = 0, max_requests: int = 3) -> list[ReplicationRequest]:
    """Requests that would cover the axes still missing. Deterministic given `seed`. Windows must have ended strictly before `now`
    (unmatured outcomes are never requested) and be clear of the original evidence and of earlier runs."""
    mine = [r for r in runs if r.discovery_id == d.discovery_id]
    a = assess(d, mine, now, policy)
    if a.status in (Status.REPLICATED, Status.FAILED):
        return []
    rng = np.random.default_rng([int(seed), int(stable_hash(d.discovery_id, 8), 16) % (2 ** 31)])
    used = {r.seed for r in mine} | set(d.seeds)
    fresh_stocks = sorted(set(pools.stocks) - set(d.stocks))
    seen_regimes = set(d.regimes) | {r.regime for r in mine}
    open_windows = []
    for s, e, reg in pools.windows:
        if as_date(e) >= as_date(now):
            continue
        if window_overlap(d.window, (s, e), pad_days=d.horizon_days) > 0:
            continue
        if any(window_overlap(r.window, (s, e), pad_days=d.horizon_days) > 0 for r in mine):
            continue
        open_windows.append((s, e, reg))
    out: list[ReplicationRequest] = []
    order = sorted(open_windows, key=lambda w: (w[2] in seen_regimes, w[0]))              # unseen regimes first
    for s, e, reg in order[:max_requests]:
        k = min(pools.stocks_per_run, len(fresh_stocks))
        if k == 0:
            break
        pick = frozenset(fresh_stocks[i] for i in rng.choice(len(fresh_stocks), size=k, replace=False))
        sd = int(rng.integers(1, 2 ** 31))
        while sd in used:
            sd = int(rng.integers(1, 2 ** 31))
        used.add(sd)
        sought = [Axis.PERIOD.value, Axis.STOCKS.value, Axis.SEED.value, Axis.CONTROL.value] + ([Axis.REGIME.value] if reg not in seen_regimes else [])
        out.append(ReplicationRequest(stable_hash([d.discovery_id, s, e, sd], 12), d.discovery_id, (s, e), pick, sd, reg, policy.require_control,
                                      tuple(sought), f"missing axes: {', '.join(a.axes_missing) or 'none'}; status {a.status.value}"))
        seen_regimes.add(reg)
    return out


# ------------------------------------------------------------------------------------------------ ledger
class ChainLog:
    """Append-only, hash-chained JSON-lines file (history is immutable). Rows are {prev, kind, body, chain}; verify() re-derives every
    link. Shared by the replication ledger, the quarantine store and the scorecard log so there is one chain implementation."""

    def __init__(self, path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_bytes().decode("utf-8").split("\n") if ln.strip()]

    def append(self, kind: str, body: Mapping) -> dict:
        rows = self.rows()
        prev = rows[-1]["chain"] if rows else "GENESIS"
        row = {"prev": prev, "kind": kind, "body": json.loads(canonical_json(dict(body)))}
        row["chain"] = stable_hash([prev, kind, row["body"]], 32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:                           # binary: text mode adds CRLF on Windows
            fh.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        return row

    def verify(self) -> list[str]:
        errs, prev = [], "GENESIS"
        for i, r in enumerate(self.rows()):
            if r["prev"] != prev:
                errs.append(f"row {i}: broken chain link")
            if r["chain"] != stable_hash([r["prev"], r["kind"], r["body"]], 32):
                errs.append(f"row {i}: chain hash mismatch (row edited)")
            prev = r["chain"]
        return errs


class ReplicationLedger(ChainLog):
    """Discoveries, runs and assessments on a ChainLog. Reads are as-of: `runs_for(id, now)` returns only runs matured strictly
    before `now`."""

    def add_discovery(self, d: Discovery) -> None:
        errs = d.validate()
        if errs:
            raise ValueError(f"invalid discovery: {errs}")
        if any(r["kind"] == "discovery" and r["body"]["discovery_id"] == d.discovery_id for r in self.rows()):
            raise FileExistsError(f"discovery {d.discovery_id} already recorded: history is never overwritten")
        self.append("discovery", d.to_dict())

    def add_run(self, r: ReplicationRun, now) -> None:
        errs = r.validate(now)
        if errs:
            raise ValueError(f"invalid run: {errs}")
        rows = self.rows()
        if not any(x["kind"] == "discovery" and x["body"]["discovery_id"] == r.discovery_id for x in rows):
            raise KeyError(f"run {r.run_id} names unknown discovery {r.discovery_id}")
        if any(x["kind"] == "run" and x["body"]["run_id"] == r.run_id for x in rows):
            raise FileExistsError(f"run {r.run_id} already recorded")
        self.append("run", r.to_dict())

    def record_assessment(self, a: ReplicationAssessment) -> None:
        self.append("assessment", {"discovery_id": a.discovery_id, "status": a.status.value, "now": a.now, "digest": a.digest(),
                                    "supporting": a.n_supporting, "refuting": a.n_refuting})

    def discoveries(self) -> dict[str, Discovery]:
        return {r["body"]["discovery_id"]: Discovery.from_dict(r["body"]) for r in self.rows() if r["kind"] == "discovery"}

    def runs_for(self, discovery_id: str, now) -> list[ReplicationRun]:
        out = [ReplicationRun.from_dict(r["body"]) for r in self.rows() if r["kind"] == "run" and r["body"]["discovery_id"] == discovery_id]
        return [r for r in out if as_date(r.matured_at) < as_date(now)]

    def status_history(self, discovery_id: str) -> list[tuple[str, str]]:
        return [(r["body"]["now"], r["body"]["status"]) for r in self.rows() if r["kind"] == "assessment" and r["body"]["discovery_id"] == discovery_id]


# ------------------------------------------------------------------------------------------------ learning and documenting the thresholds
@dataclass(frozen=True)
class ResolvedDiscovery:
    """A discovery whose truth became known (a planted world, or a long run of fresh evidence). `first_retention` is the
    retention of its FIRST replication attempt; `run_supports` says, for every attempt, whether it looked like support."""
    discovery_id: str
    truth: str                               # "held" | "failed"
    source: str                              # "planted_world" | "fresh_evidence" | ...
    first_retention: float
    run_supports: tuple[bool, ...]


@dataclass(frozen=True)
class PolicyEntry:
    name: str
    value: Any
    source: str
    evidence: str
    rationale: str


@dataclass(frozen=True)
class PolicyDocument:
    entries: tuple[PolicyEntry, ...]
    n_held: int
    n_failed: int
    learned: bool
    policy_digest: str

    def markdown(self) -> str:
        head = ["# Replication requirements (C66 section 32)",
                f"History used: {self.n_held} held, {self.n_failed} failed discoveries. Learned: {'yes' if self.learned else 'no - documented priors only'}.",
                "", "| threshold | value | source | why |", "|---|---|---|---|"]
        rows = [f"| {e.name} | {e.value} | {e.source} | {e.rationale} {e.evidence} |" for e in self.entries]
        return "\n".join(head + rows + ["", "IMPLEMENTED - NOT VALIDATED."])


def learn_policy(history: Sequence[ResolvedDiscovery], base: ReplicationPolicy = DEFAULT_POLICY, target_false_accept: float = 0.10,
                 min_history: int = 20, allow_loosen: bool = False) -> tuple[ReplicationPolicy, PolicyDocument]:
    """Derive thresholds from resolved discoveries, never from P&L. Two rules, both quantiles of the null:
      min_retention_floor  smallest retention that lets at most `target_false_accept` of eventually-FAILED discoveries pass their
                           first replication;
      min_independent      smallest k with (per-run false-support rate)^k <= target_false_accept.
    With fewer than `min_history` failed AND `min_history / 2` held discoveries nothing is changed, and the document says so.
    Learning tightens by default; loosening below the prior needs allow_loosen=True and is recorded as such."""
    bad = base.validate()
    if bad:
        raise ValueError(f"invalid base policy: {bad}")
    failed = [h for h in history if h.truth == "failed" and math.isfinite(h.first_retention)]
    held = [h for h in history if h.truth == "held"]
    if len(failed) < min_history or len(held) < min_history // 2:
        doc = _document(base, len(held), len(failed), False,
                        f"insufficient history (need {min_history} failed and {min_history // 2} held)")
        return base, doc
    rets = np.array(sorted(h.first_retention for h in failed))
    idx = min(len(rets) - 1, int(math.ceil((1.0 - target_false_accept) * len(rets))) - 1)
    floor = float(np.clip(rets[max(idx, 0)] + 1e-6, 0.0, 0.8))
    if not allow_loosen:
        floor = max(floor, base.min_retention_floor)
    runs = [s for h in failed for s in h.run_supports]
    rate = float(np.clip((sum(runs) + 1.0) / (len(runs) + 2.0), 0.02, 0.9)) if runs else 0.5      # Laplace-smoothed false-support rate
    k = int(np.clip(math.ceil(math.log(target_false_accept) / math.log(rate)), 2, 6))
    if not allow_loosen:
        k = max(k, base.min_independent)
    src = {"min_retention_floor": f"LEARNED from {len(failed)} failed discoveries", "min_independent": f"LEARNED from {len(runs)} refuted runs"}
    pol = dataclasses.replace(base, min_retention_floor=round(floor, 4), min_independent=k, sources={**dict(base.sources), **src})
    return pol, _document(pol, len(held), len(failed), True, f"false-support rate per run {rate:.2f}; target false accept {target_false_accept:.0%}")


def _document(pol: ReplicationPolicy, n_held: int, n_failed: int, learned: bool, note: str) -> PolicyDocument:
    why = {
        "min_independent": "independent replications needed; one experiment is never its own replication.",
        "min_periods": "a run shorter than this cannot speak; a miss is then INCONCLUSIVE.",
        "min_fresh_axes": "distinct fresh-data axes (period / stocks / regime) the supporting runs must cover; period is mandatory.",
        "require_control": "the discovery must beat a matched control, not just be positive.",
        "min_retention_floor": "replication effect over original effect can never be required below this.",
        "retention_fraction": "share of the winner's-curse-adjusted expectation a replication must retain.",
        "max_stock_overlap": "share of a run's names allowed to come from the original evidence.",
        "min_success_share": "supporting over all independent attempts: retrying until one works cannot pass.",
        "max_i2": "pooled heterogeneity above this means the runs disagree too much to be one effect.",
        "fail_after": "powered independent refutations that mark FAILED_TO_REPLICATE.",
    }
    entries = tuple(PolicyEntry(k, getattr(pol, k), pol.source_of(k), note if k in ("min_independent", "min_retention_floor") else "", v)
                    for k, v in why.items())
    return PolicyDocument(entries, n_held, n_failed, learned, pol.digest())


def requirements_document(policy: ReplicationPolicy = DEFAULT_POLICY, history: Sequence[ResolvedDiscovery] = ()) -> str:
    """The replication requirements as documentation: every threshold, where it came from and why. Learns from `history` when there
    is enough of it."""
    pol, doc = learn_policy(history, policy)
    return doc.markdown() + "\n\nA single lucky experiment can never change the system: see authorize_system_change."


# ------------------------------------------------------------------------------------------------ public entry
@dataclass(frozen=True)
class StepReport:
    now: str
    assessments: Mapping[str, ReplicationAssessment]
    requests: tuple[ReplicationRequest, ...]
    status_changes: tuple[tuple[str, str, str], ...]        # (discovery_id, old status, new status)
    authorized: tuple[str, ...]                              # discovery ids that may now change the system

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.assessments.values():
            out[a.status.value] = out.get(a.status.value, 0) + 1
        return out


def step(ledger: ReplicationLedger, now, pools: Pools | None = None, policy: ReplicationPolicy = DEFAULT_POLICY, seed: int = 0,
         replayed_years: Sequence[int] = ()) -> StepReport:
    """The wave-2 research loop's entry: assess every recorded discovery as of `now`, log status changes, plan the next runs for the
    unfinished ones, and list which discoveries may change the system (REPLICATED and clear of the same-year replay)."""
    assessments, requests, changes, ok = {}, [], [], []
    for did, d in sorted(ledger.discoveries().items()):
        if as_date(d.matured_at) >= as_date(now):
            continue
        runs = ledger.runs_for(did, now)
        a = assess(d, runs, now, policy)
        hist = ledger.status_history(did)
        if not hist or hist[-1][1] != a.status.value:
            changes.append((did, hist[-1][1] if hist else "NONE", a.status.value))
            ledger.record_assessment(a)
        assessments[did] = a
        if a.may_change_system and not same_year_conflict(d, runs, replayed_years):
            ok.append(did)
        if pools is not None:
            requests.extend(plan_next(d, runs, pools, now, policy, seed))
    return StepReport(str(as_date(now)), assessments, tuple(requests), tuple(changes), tuple(ok))


# ------------------------------------------------------------------------------------------------ auditing the runs themselves
@dataclass(frozen=True)
class RunAuditFinding:
    run_id: str
    check: str
    detail: str


def audit_runs(d: Discovery, runs: Sequence[ReplicationRun], tol: float = 1e-12) -> list[RunAuditFinding]:
    """Ways a 'replication' can be counterfeit: an effects series copied from another run or from the original, a control that IS the
    treatment, a constant series, a run dated before its own window ends, or a run claiming fresh stocks with a stock set identical to
    the original's. Every finding is a reason to treat the run as not independent evidence."""
    out: list[RunAuditFinding] = []
    seen: dict[tuple, str] = {}
    for r in sorted(runs, key=lambda r: r.run_id):
        x = np.asarray(r.effects, dtype=float)
        if x.size > 1 and float(x.std()) <= tol:
            out.append(RunAuditFinding(r.run_id, "constant_effects", "every period has the same effect: a placeholder, not a measurement"))
        key = tuple(np.round(x, 9).tolist())
        if len(key) > 2 and key in seen:
            out.append(RunAuditFinding(r.run_id, "copied_effects", f"effects are identical to run {seen[key]}"))
        seen.setdefault(key, r.run_id)
        if r.control_effects is not None:
            c = np.asarray(r.control_effects, dtype=float)
            if c.size == x.size and c.size > 1 and float(np.abs(c - x).max()) <= tol:
                out.append(RunAuditFinding(r.run_id, "control_is_treatment", "control effects equal the treatment effects"))
        if r.stocks == d.stocks:
            out.append(RunAuditFinding(r.run_id, "same_stocks_as_original", "identical stock set: not a fresh-stocks replication"))
        if as_date(r.matured_at) < _window(r.window)[1]:
            out.append(RunAuditFinding(r.run_id, "matured_before_window_end", f"matured {r.matured_at} before its window ends {r.window[1]}"))
        if abs(float(x.mean()) - d.effect) <= tol and x.size == d.n_periods:
            out.append(RunAuditFinding(r.run_id, "reproduces_original_exactly", "mean equals the original effect to machine precision"))
    return out


def clean_runs(d: Discovery, runs: Sequence[ReplicationRun]) -> tuple[list[ReplicationRun], list[RunAuditFinding]]:
    """Runs with an audit finding are separated, never deleted (they stay in the ledger); the rest may be assessed."""
    findings = audit_runs(d, runs)
    bad = {f.run_id for f in findings}
    return [r for r in runs if r.run_id not in bad], findings


# ------------------------------------------------------------------------------------------------ why did it fail to replicate?
@dataclass(frozen=True)
class FailureDiagnosis:
    discovery_id: str
    cause: str                               # a FailureCause value; UNKNOWN is an allowed, honest answer
    confidence: str                          # "supported" | "suggestive" | "unknown"
    evidence: tuple[str, ...]
    lesson: str


def diagnose_failure(d: Discovery, runs: Sequence[ReplicationRun], now, policy: ReplicationPolicy = DEFAULT_POLICY) -> FailureDiagnosis:
    """Learn WHY a discovery did not replicate, from which axes held and which broke (section 32/33: unknown stays unknown).
      supports on new periods but not new stocks      -> WRONG_CONTEXT (stock-specific)
      supports on old regimes, fails on a new regime  -> REGIME_CHANGE
      positive but does not beat the matched control  -> REDUNDANCY (the control explains it)
      retention ~0 and the search was large           -> FALSE_PATTERN (selection luck)
      significantly smaller than the original         -> WEAKENING_EFFECT
      negative on fresh data                          -> REVERSAL
      nothing decisive                                -> INSUFFICIENT_EVIDENCE
    Anything that fits none of these is UNKNOWN, never a guess."""
    a = assess(d, runs, now, policy)
    res = [x for x in a.run_results if x.outcome != Outcome.INVALID]
    if not res:
        return FailureDiagnosis(d.discovery_id, FailureCause.INSUFFICIENT_EVIDENCE.value, "unknown", ("no valid run",), "collect fresh runs")
    ev: list[str] = []
    fresh = [x for x in res if x.fresh_data]
    if not fresh:
        return FailureDiagnosis(d.discovery_id, FailureCause.INSUFFICIENT_EVIDENCE.value, "unknown", ("no run used fresh data",),
                                "replicate on a fresh period, stocks or regime")
    mean_eff = float(np.mean([x.effect for x in fresh]))
    if any(x.beats_control is False and math.isfinite(x.ci_low) and x.ci_low > 0 for x in fresh):
        ev.append("positive on fresh data but not above the matched control")
        return FailureDiagnosis(d.discovery_id, FailureCause.REDUNDANCY.value, "supported", tuple(ev),
                                "test against the control that explains it before trusting the pattern")
    period_ok = [x for x in fresh if x.earned(Axis.PERIOD) and not x.earned(Axis.STOCKS) and x.outcome == Outcome.SUPPORTS]
    stocks_bad = [x for x in fresh if x.earned(Axis.STOCKS) and x.outcome != Outcome.SUPPORTS]
    if period_ok and stocks_bad:
        ev.append(f"{len(period_ok)} same-stock fresh-period run(s) support it; {len(stocks_bad)} fresh-stock run(s) do not")
        return FailureDiagnosis(d.discovery_id, FailureCause.WRONG_CONTEXT.value, "supported", tuple(ev), "scope the discovery to the stocks it held on")
    reg_ok = [x for x in fresh if not x.earned(Axis.REGIME) and x.outcome == Outcome.SUPPORTS]
    reg_bad = [x for x in fresh if x.earned(Axis.REGIME) and x.outcome != Outcome.SUPPORTS]
    if reg_ok and reg_bad:
        ev.append(f"holds in {len(reg_ok)} known-regime run(s), fails in {len(reg_bad)} new-regime run(s)")
        return FailureDiagnosis(d.discovery_id, FailureCause.REGIME_CHANGE.value, "supported", tuple(ev), "condition the discovery on the regime")
    if mean_eff < 0 and all(x.ci_high < 0 for x in fresh if math.isfinite(x.ci_high)):
        ev.append(f"mean fresh effect {mean_eff:+.5f} with every upper bound below zero")
        return FailureDiagnosis(d.discovery_id, FailureCause.REVERSAL.value, "supported", tuple(ev), "the sign flipped: examine what changed in the market")
    if any(x.decayed for x in fresh) and mean_eff > 0:
        ev.append(f"positive ({mean_eff:+.5f}) but significantly below the original {d.effect:+.5f}")
        return FailureDiagnosis(d.discovery_id, FailureCause.WEAKENING_EFFECT.value, "suggestive", tuple(ev), "track the decay; consider a time-varying weight")
    near_zero = all(abs(x.effect) <= 2 * x.se for x in fresh if math.isfinite(x.se))
    if near_zero and d.n_tests_searched >= 20 and lucky_probability(d) > 0.05:
        ev.append(f"fresh effects indistinguishable from zero; a search of {d.n_tests_searched} tests makes this t={d.t:.2f} likely by luck "
                  f"(p={lucky_probability(d):.2f})")
        return FailureDiagnosis(d.discovery_id, FailureCause.FALSE_PATTERN.value, "supported", tuple(ev), "raise the discovery bar for searches this large")
    if a.status == Status.INCONCLUSIVE:
        return FailureDiagnosis(d.discovery_id, FailureCause.INSUFFICIENT_EVIDENCE.value, "suggestive",
                                (f"{len(res)} run(s), none decisive",), "run longer or more independent replications")
    ev.append(f"status {a.status.value}; no diagnostic pattern matched")
    return FailureDiagnosis(d.discovery_id, FailureCause.UNKNOWN.value, "unknown", tuple(ev), "cause not established; record it as unknown")


def scope_from_runs(d: Discovery, runs: Sequence[ReplicationRun], now, policy: ReplicationPolicy = DEFAULT_POLICY) -> dict:
    """Where a partly-replicating discovery DID hold: the regimes and windows of the supporting runs. A discovery that replicates only
    in some contexts is narrowed to them (never widened), and the narrowed claim has to earn its own replication."""
    a = assess(d, runs, now, policy)
    by = {r.run_id: r for r in runs}
    held = [by[x.run_id] for x in a.run_results if x.outcome == Outcome.SUPPORTS and x.run_id in by]
    failed = [by[x.run_id] for x in a.run_results if x.outcome == Outcome.REFUTES and x.run_id in by]
    held_reg = sorted({r.regime for r in held})
    failed_reg = sorted({r.regime for r in failed} - set(held_reg))
    return {"held_regimes": held_reg, "failed_regimes": failed_reg, "n_held": len(held), "n_failed": len(failed),
            "narrowed": bool(held and failed), "needs_own_replication": bool(held and failed),
            "claim": "hold only in the listed regimes" if held and failed else "unchanged"}


# ------------------------------------------------------------------------------------------------ across many discoveries
def decline_effect(pairs: Sequence[tuple[Discovery, ReplicationAssessment]], seed: int = 0, n_boot: int = 500) -> dict:
    """Do replications systematically come in below the originals? Ratio of pooled replication effect to original effect over
    discoveries that have one, with a bootstrap CI. A ratio well under 1 is the decline effect: the discovery process is still
    over-crediting luck and its bar should rise. NaN, not zero, when nothing has been replicated."""
    rows = [(d.effect, a.pooled["effect"]) for d, a in pairs if a.n_supporting + a.n_refuting > 0 and math.isfinite(a.pooled.get("effect", float("nan")))]
    if not rows:
        return {"n": 0, "ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "declining": None}
    o, r = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
    ratio = float(r.sum() / o.sum()) if o.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(rows), len(rows))
        boots.append(r[i].sum() / o[i].sum() if o[i].sum() > 0 else np.nan)
    lo, hi = np.nanpercentile(boots, [5, 95])
    return {"n": len(rows), "ratio": ratio, "lo": float(lo), "hi": float(hi), "declining": bool(hi < 1.0)}


def replication_rate(assessments: Sequence[ReplicationAssessment]) -> dict:
    """Status counts and the share of DECIDED discoveries (REPLICATED / FAILED / CONTRADICTED) that replicated. Discoveries still
    waiting are reported, not counted for or against."""
    counts: dict[str, int] = {}
    for a in assessments:
        counts[a.status.value] = counts.get(a.status.value, 0) + 1
    decided = counts.get("REPLICATED", 0) + counts.get("FAILED_TO_REPLICATE", 0) + counts.get("CONTRADICTED", 0)
    return {"counts": counts, "decided": decided, "waiting": len(assessments) - decided,
            "rate": counts.get("REPLICATED", 0) / decided if decided else float("nan")}


def axis_breakdown(assessments: Sequence[ReplicationAssessment]) -> dict[str, dict[str, float]]:
    """For every freshness axis: how many runs earned it and how many of those supported. Shows which kind of freshness kills
    discoveries (e.g. they die on fresh stocks but survive fresh periods)."""
    out: dict[str, dict[str, float]] = {ax.value: {"runs": 0, "supported": 0} for ax in Axis}
    for a in assessments:
        for x in a.run_results:
            if x.outcome == Outcome.INVALID:
                continue
            for ax in x.earned_axes():
                out[ax.value]["runs"] += 1
                out[ax.value]["supported"] += x.outcome == Outcome.SUPPORTS
    for v in out.values():
        v["support_rate"] = v["supported"] / v["runs"] if v["runs"] else float("nan")
    return out


def false_discovery_expectation(discoveries: Sequence[Discovery], alpha: float = 0.05) -> dict:
    """How many of these discoveries would exist from pure noise, given the size of the search that produced each? Sums the Sidak
    probability that a null search of that size yields a t this large: the expected number of lucky discoveries."""
    ps = [lucky_probability(d) for d in discoveries]
    return {"n": len(ps), "expected_lucky": float(sum(ps)), "share_plausibly_lucky": float(np.mean([p > alpha for p in ps])) if ps else float("nan"),
            "max_lucky_probability": float(max(ps)) if ps else float("nan")}


# ------------------------------------------------------------------------------------------------ deciding how much replication to buy
def sprt_replication(effects_by_run: Sequence[Sequence[float]], sd: float, adjusted: float, alpha: float = 0.05, beta: float = 0.2) -> dict:
    """Wald sequential test on the pooled fresh evidence: H1 = the effect is the winner's-curse-adjusted one, H0 = zero. Stops
    early when the evidence is decisive either way, so a clear failure is not replicated ten more times."""
    if sd <= 0 or adjusted <= 0:
        return {"decision": "H0", "llr": float("-inf"), "n": 0, "reason": "no positive adjusted effect to test"}
    x = np.concatenate([np.asarray(e, dtype=float) for e in effects_by_run]) if effects_by_run else np.array([])
    if x.size == 0:
        return {"decision": "CONTINUE", "llr": 0.0, "n": 0, "reason": "no evidence yet"}
    llr = float((adjusted / sd ** 2) * (x - adjusted / 2.0).sum())
    upper, lower = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    decision = "H1" if llr >= upper else "H0" if llr <= lower else "CONTINUE"
    return {"decision": decision, "llr": llr, "n": int(x.size), "upper": upper, "lower": lower,
            "reason": {"H1": "evidence favours the adjusted effect", "H0": "evidence favours no effect", "CONTINUE": "not decisive yet"}[decision]}


@dataclass(frozen=True)
class ReplicationPriority:
    discovery_id: str
    priority: float
    reasons: tuple[str, ...]


def replication_priority(d: Discovery, a: ReplicationAssessment, decision_value: float = 1.0, cost: float = 1.0) -> ReplicationPriority:
    """Value of a further replication per unit cost. High when the discovery would matter (decision_value), is uncertain (lucky
    probability near a coin flip is most informative; near 0 or 1 teaches little) and still short of the requirements. Zero for a
    decided discovery: REPLICATED and FAILED need no more compute (section 20 stop-wasting-compute)."""
    if a.status in (Status.REPLICATED, Status.FAILED):
        return ReplicationPriority(d.discovery_id, 0.0, (f"already {a.status.value}",))
    p = min(max(a.lucky_probability, 0.0), 1.0)
    uncertainty = 4.0 * p * (1.0 - p)
    unmet = sum(1 for v in a.requirements.values() if not v) / max(len(a.requirements), 1)
    val = max(decision_value, 0.0) * (0.25 + uncertainty) * (0.25 + unmet) / max(cost, 1e-9)
    return ReplicationPriority(d.discovery_id, float(val), (f"lucky probability {p:.2f}", f"{unmet:.0%} of requirements unmet", f"decision value {decision_value:g}"))


def pool_capacity(d: Discovery, runs: Sequence[ReplicationRun], pools: Pools, now, policy: ReplicationPolicy = DEFAULT_POLICY) -> dict:
    """How many more independent fresh replications the available evidence can even supply. If the answer is below what the policy
    needs, the discovery must wait for more history: it is reported as UNREPLICABLE_YET rather than replicated on reused data."""
    mine = [r for r in runs if r.discovery_id == d.discovery_id]
    free = []
    for s, e, reg in pools.windows:
        if as_date(e) >= as_date(now) or window_overlap(d.window, (s, e), d.horizon_days) > 0:
            continue
        if any(window_overlap(r.window, (s, e), d.horizon_days) > 0 for r in mine):
            continue
        free.append((s, e, reg))
    chosen: list[tuple[str, str, str]] = []
    for w in sorted(free, key=lambda w: (as_date(w[1]), w[0])):              # greedy: earliest-ending non-overlapping windows
        if all(window_overlap(c[:2], w[:2], d.horizon_days) == 0 for c in chosen):
            chosen.append(w)
    have = assess(d, mine, now, policy).n_supporting
    need = max(policy.min_independent - have, 0)
    return {"free_windows": len(free), "independent_slots": len(chosen), "fresh_stocks": len(set(pools.stocks) - set(d.stocks)),
            "needed": need, "sufficient": len(chosen) >= need and len(set(pools.stocks) - set(d.stocks)) >= pools.stocks_per_run,
            "label": "OK" if len(chosen) >= need else "UNREPLICABLE_YET"}


# ------------------------------------------------------------------------------------------------ as-of history and reporting
def status_timeline(d: Discovery, runs: Sequence[ReplicationRun], dates: Sequence, policy: ReplicationPolicy = DEFAULT_POLICY) -> list[tuple[str, str]]:
    """The status the world would have seen on each date: runs are filtered to those matured strictly before that date, so the
    timeline can be compared with what the loop actually decided (a look-ahead bug shows up as a difference)."""
    out = []
    for now in sorted(dates, key=as_date):
        if as_date(d.matured_at) >= as_date(now):
            out.append((str(as_date(now)), "NOT_YET_DISCOVERED"))
            continue
        vis = [r for r in runs if as_date(r.matured_at) < as_date(now)]
        out.append((str(as_date(now)), assess(d, vis, now, policy).status.value))
    return out


def render_assessment(a: ReplicationAssessment) -> str:
    """Plain-English report of one assessment, ending with what it may and may not do."""
    lines = [f"# Replication of {a.discovery_id}: {a.status.value}", f"as of {a.now}; policy {a.policy_digest}",
             f"{a.n_supporting} independent fresh run(s) support it, {a.n_refuting} refute it, of {a.n_valid_groups} independent attempt(s) "
             f"({a.success_share:.0%} success).",
             f"Winner's-curse-adjusted original effect {a.adjusted_effect:+.5f}; probability the original is search luck {a.lucky_probability:.2f}.",
             f"Pooled fresh effect {a.pooled.get('effect', float('nan')):+.5f} (95% [{a.pooled.get('ci_low', float('nan')):+.5f}, "
             f"{a.pooled.get('ci_high', float('nan')):+.5f}], I2 {a.pooled.get('i2', 0.0):.0%}).",
             "Freshness covered: " + (", ".join(a.axes_covered) or "none") + "; missing: " + (", ".join(a.axes_missing) or "none"), "", "Requirements:"]
    lines += [f"- [{'x' if v else ' '}] {k}" for k, v in a.requirements.items()]
    lines += ["", "Runs:"]
    lines += [f"- {x.run_id}: {x.outcome.value}, effect {x.effect:+.5f}, retention {x.retention:.2f} (need {x.required_retention:.2f}); " + "; ".join(x.reasons)
              for x in a.run_results]
    lines += ["", "May change the system: " + ("YES" if a.may_change_system else "NO"), "IMPLEMENTED - NOT VALIDATED."]
    return "\n".join(lines)
