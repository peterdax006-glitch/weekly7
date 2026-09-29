"""Knowledge promotion gate (contract C62 section 45; checklist J-phase). IMPLEMENTED - NOT VALIDATED.

Production promotion requires ALL applicable gates:
statistical validity, incremental value, OOS confirmation, cross-context transfer, risk acceptance, anti-memorization,
future-information audit, reproducibility, stability, provenance completeness. One critical failure blocks promotion and a
written rejection report says exactly which evidence was missing or too weak.

Fail-closed rules (section 59): missing evidence is a FAIL ("MISSING"), never a pass; an exception inside a gate is a FAIL;
a gate may be NOT_APPLICABLE only for a reason this module derives (the knowledge changes no risk-bearing decision) or that
the policy waives in writing, and critical gates can never be waived. Nothing here reads data: callers hand in evidence
records that were measured elsewhere, all dated strictly before `now`. Builds on engine.learning.core (vocabulary, hashing,
require_past) and mirrors, for knowledge, the model-level checks of engine.champion (independent windows, risk floors)."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeGuard, cast, overload

import numpy as np

from engine import experiment_memory as EM
from engine import pattern_stats as PS

from .core import (DecisionEffect, Epistemic, FirewallBreach, KnowledgeLike, Promotion, as_date, current_code_hash,
                   require_past, stable_hash)

GATES = ("statistical_validity", "incremental_value", "oos_confirmation", "cross_context_transfer", "risk_acceptance",
         "anti_memorization", "future_information_audit", "reproducibility", "stability", "provenance_completeness")
CRITICAL = frozenset({"statistical_validity", "oos_confirmation", "risk_acceptance", "anti_memorization",
                      "future_information_audit", "reproducibility", "provenance_completeness"})
WAIVABLE = frozenset(set(GATES) - CRITICAL - {"incremental_value"})       # transfer and stability, in writing only
PASS, FAIL, MISSING, NA = "PASS", "FAIL", "MISSING", "NOT_APPLICABLE"
RISK_FREE_EFFECTS = frozenset({DecisionEffect.RESEARCH_PRIORITY})          # changes what is studied, never what is held
PROMOTABLE_EPISTEMIC = (Epistemic.SUPPORTED, Epistemic.CONDITIONAL)
IDENTITY_TOKENS = re.compile(r"(^|_)(ticker|symbol|cusip|isin|permno|date|year|day|week|month|id|name|sedol)(_|$)", re.I)


# ------------------------------------------------------------------------------------------------ policy
@dataclass(frozen=True)
class PromotionPolicy:
    """Every threshold in one validated, hashable place (section 5: no giant untyped dict, and a threshold changed after
    seeing a result must show up as a different policy hash)."""
    alpha: float = 0.05
    multiplicity: str = "sidak"                 # sidak | bonferroni: n_tests_searched deflates alpha
    min_obs: int = 60
    min_effect: float = 0.0
    min_delta_periods: int = 30
    min_gain: float = 0.0
    min_gain_t: float = 2.0
    bootstrap_n: int = 600
    bootstrap_block: int = 4
    max_incumbent_overlap: float = 0.97
    min_oos_periods: int = 20
    min_oos_retention: float = 0.35
    min_oos_positive_share: float = 0.5
    min_oos_t: float = 1.64
    min_transfer_contexts: int = 3
    min_transfer_n: int = 15
    min_transfer_positive_share: float = 0.66
    min_transfer_ratio: float = 0.30
    worst_transfer_floor: float = -0.5           # worst context effect as a multiple of the home effect
    worst_period_floor: float = -0.20
    max_drawdown_floor: float = -0.35
    cvar05_floor: float = -0.15
    max_catastrophic: int = 0
    min_risk_periods: int = 20
    incumbent_risk_tolerance: float = 0.02
    min_identity_retention: float = 0.70
    max_top_identity_share: float = 0.35
    min_distinct_identities: int = 20
    determinism_tol: float = 1e-6
    seed_spread_tol: float = 0.30
    min_reruns: int = 3
    min_stability_periods: int = 6
    min_stability_positive_share: float = 0.6
    max_single_period_share: float = 0.5
    min_perturbation_retention: float = 0.5
    min_perturbations: int = 4
    waivers: Mapping[str, str] = field(default_factory=dict)      # gate -> written reason (>= 20 chars)

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.alpha < 0.5:
            errs.append("alpha must be in (0, 0.5)")
        if self.multiplicity not in ("sidak", "bonferroni"):
            errs.append("multiplicity must be sidak or bonferroni")
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not math.isfinite(v):
                errs.append(f"{f.name} is not finite")
        for name in ("min_obs", "min_delta_periods", "min_oos_periods", "min_reruns", "min_stability_periods"):
            if getattr(self, name) < 2:
                errs.append(f"{name} must be at least 2")
        if not 0 <= self.max_top_identity_share <= 1:
            errs.append("max_top_identity_share must be in [0, 1]")
        for g, why in self.waivers.items():
            if g not in GATES:
                errs.append(f"waiver for unknown gate {g}")
            elif g not in WAIVABLE:
                errs.append(f"gate {g} is critical or core and can never be waived")
            elif len(str(why).strip()) < 20:
                errs.append(f"waiver for {g} needs a written reason of at least 20 characters")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


# ------------------------------------------------------------------------------------------------ evidence records
@dataclass(frozen=True)
class StatisticalEvidence:
    effect: float | None = None
    n_obs: int | None = None
    t_stat: float | None = None
    p_value: float | None = None
    n_tests_searched: int = 1                    # how many hypotheses the discovery search looked at (multiplicity)
    effective_n: float | None = None             # after overlap / autocorrelation; may be far below n_obs


@dataclass(frozen=True)
class IncrementalEvidence:
    delta_series: tuple[float, ...] | None = None   # per-period decision value WITH minus WITHOUT the knowledge, in time order
    seed: int = 0
    decision_overlap: float | None = None           # share of decisions identical to the incumbent's (1.0 = adds nothing)


@dataclass(frozen=True)
class OOSEvidence:
    train_end: str | None = None
    oos_dates: tuple[str, ...] = ()
    oos_effects: tuple[float, ...] = ()
    in_sample_effect: float | None = None
    oos_windows: tuple[str, ...] = ()
    train_windows: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransferEvidence:
    context_effects: Mapping[str, float] = field(default_factory=dict)
    context_n: Mapping[str, int] = field(default_factory=dict)
    home_contexts: tuple[str, ...] = ()
    home_effect: float | None = None


@dataclass(frozen=True)
class RiskEvidence:
    n_periods: int = 0
    worst_period: float | None = None
    max_drawdown: float | None = None
    cvar05: float | None = None
    catastrophic_count: int | None = None
    incumbent_worst_period: float | None = None
    incumbent_max_drawdown: float | None = None


@dataclass(frozen=True)
class MemorizationEvidence:
    identity_shuffle_retention: float | None = None    # effect with tickers/dates scrambled, as a share of the real effect
    disguised_rerun_retention: float | None = None     # effect on a disguised rerun of a seen window, as a share of real
    feature_names: tuple[str, ...] = ()
    lookup_table_suspected: bool | None = None
    top_identity_share: float | None = None            # largest single ticker/date share of the effect
    n_distinct_identities: int | None = None


@dataclass(frozen=True)
class FutureAudit:
    audit_ran: bool = False
    findings: tuple[str, ...] = ()
    max_evidence_date: str | None = None
    sealed_windows_touched: tuple[str, ...] = ()
    outcomes_after_now: int | None = None


@dataclass(frozen=True)
class RerunRecord:
    value: float
    seed: int
    code_hash: str
    data_hash: str = ""


@dataclass(frozen=True)
class ReproEvidence:
    reruns: tuple[RerunRecord, ...] = ()
    expected_data_hash: str = ""


@dataclass(frozen=True)
class StabilityEvidence:
    period_effects: tuple[float, ...] = ()
    perturbation_effects: tuple[float, ...] = ()       # effect under +-20% parameter / definition perturbations
    reference_effect: float | None = None


@dataclass(frozen=True)
class PromotionEvidence:
    """Bundle handed to the gate. Any None member is MISSING evidence for its gate."""
    statistical: StatisticalEvidence | None = None
    incremental: IncrementalEvidence | None = None
    oos: OOSEvidence | None = None
    transfer: TransferEvidence | None = None
    risk: RiskEvidence | None = None
    memorization: MemorizationEvidence | None = None
    future: FutureAudit | None = None
    repro: ReproEvidence | None = None
    stability: StabilityEvidence | None = None

    def digest(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str                       # PASS | FAIL | MISSING | NOT_APPLICABLE
    critical: bool
    detail: str
    measures: Mapping[str, Any] = field(default_factory=dict)
    margin: float | None = None       # signed headroom to the threshold, in threshold units (>0 passes) - for near-miss lists

    @property
    def ok(self) -> bool:
        return self.status in (PASS, NA)


@dataclass(frozen=True)
class PromotionDecision:
    knowledge_id: str
    version: int
    now: str
    verdict: str                      # PROMOTE | BLOCK
    results: tuple[GateResult, ...]
    preconditions: tuple[str, ...]    # failed preconditions (not gates): state, decision effect, conformance
    policy_digest: str
    evidence_digest: str
    code_hash: str

    @property
    def decision_id(self) -> str:
        return stable_hash([self.knowledge_id, self.version, self.now, self.verdict, self.policy_digest,
                            self.evidence_digest, [r.gate + r.status for r in self.results]])

    @property
    def failed(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.ok)

    @property
    def critical_failures(self) -> tuple[str, ...]:
        return tuple(r.gate for r in self.results if not r.ok and r.critical)

    @property
    def promote(self) -> bool:
        return self.verdict == "PROMOTE"

    def to_dict(self) -> dict:
        return {"decision_id": self.decision_id, "knowledge_id": self.knowledge_id, "version": self.version, "now": self.now,
                "verdict": self.verdict, "preconditions": list(self.preconditions), "policy": self.policy_digest,
                "evidence": self.evidence_digest, "code_hash": self.code_hash,
                "gates": [dataclasses.asdict(r) for r in self.results]}


# ------------------------------------------------------------------------------------------------ statistics
@overload
def _clean(x: Sequence[float], what: str) -> np.ndarray: ...
@overload
def _clean(x: Sequence[float] | None, what: str) -> np.ndarray | None: ...
def _clean(x: Sequence[float] | None, what: str) -> np.ndarray | None:
    """Series as a float array, or None when absent. A non-finite value raises: NaN is not a zero return."""
    if x is None:
        return None
    a = np.asarray(list(x), dtype=float)
    if a.size and not np.isfinite(a).all():
        raise ValueError(f"{what} contains non-finite values")
    return a


def t_stat(x: np.ndarray) -> float:
    """One-sample t of the mean; 0 for an all-zero series and +-inf for a constant non-zero one (a constant is suspicious,
    the caller decides)."""
    if x.size < 2:
        return 0.0
    m, sd = float(x.mean()), float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0 if m == 0.0 else math.copysign(math.inf, m)
    return m / (sd / math.sqrt(x.size))


def one_sided_p(t: float) -> float:
    """One-sided normal p of a t-statistic (engine.pattern_stats.t_to_p is the repo's two-sided version; halved on the
    favourable side). The normal approximation is adequate at the >= 60 effective observations the gate demands."""
    p2 = float(PS.t_to_p(t))
    return p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0


def adjusted_p(p: float, n_tests: int, method: str = "sidak") -> float:
    """p after searching n_tests hypotheses. A pattern that is the best of 5,000 candidates must clear a far higher bar than
    one hypothesised in advance. Bonferroni is engine.pattern_stats.bonferroni; Sidak (exact for independent tests) is the
    only local formula."""
    n = max(1, int(n_tests))
    if method == "bonferroni":
        return float(PS.bonferroni([p], n)[0])
    return float(1.0 - (1.0 - p) ** n)


def block_bootstrap_ci(x: np.ndarray, seed: int, n_boot: int = 600, block: int = 4, level: float = 0.95) -> tuple[float, float]:
    """Circular-block bootstrap CI of the mean, delegated to engine.experiment_memory.bootstrap_diff (blocks keep the
    autocorrelation that overlapping weekly windows create). Deterministic given `seed`; NaN pair when n < 2."""
    r = EM.bootstrap_diff(x, None, n_boot=n_boot, seed=int(seed), block=block, alpha=1.0 - level)
    return (r["ci_low"], r["ci_high"]) if r["known"] else (float("nan"), float("nan"))


def positive_share(x: np.ndarray) -> float:
    return float((x > 0).mean()) if x.size else float("nan")


def rel_spread(v: np.ndarray) -> float:
    """(max - min) relative to the mean magnitude; inf when the values straddle zero with no scale."""
    scale = max(abs(float(v.mean())), 1e-12)
    return float((v.max() - v.min()) / scale)


def _fin(v) -> TypeGuard[float]:
    return v is not None and not isinstance(v, bool) and math.isfinite(float(v))


def _margin(value: float, threshold: float) -> float:
    """Headroom of `value` over `threshold` in units of the threshold (or 1.0 when the threshold is ~0)."""
    return float((value - threshold) / max(abs(threshold), 1e-9))


# ------------------------------------------------------------------------------------------------ individual gates
def _result(gate: str, ok: bool, detail: str, measures: Mapping[str, Any] | None = None, margin: float | None = None,
            missing: bool = False) -> GateResult:
    status = MISSING if missing else (PASS if ok else FAIL)
    return GateResult(gate, status, gate in CRITICAL, detail, dict(measures or {}), margin)


def _missing(gate: str, what: str) -> GateResult:
    return _result(gate, False, f"evidence missing: {what}", missing=True)


def gate_statistical_validity(ev: StatisticalEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "statistical_validity"
    if ev is None or not _fin(ev.effect) or ev.n_obs is None:
        return _missing(g, "effect and n_obs")
    n_eff = ev.effective_n if _fin(ev.effective_n) else float(ev.n_obs)
    if n_eff < pol.min_obs:
        return _result(g, False, f"effective sample {n_eff:.0f} < {pol.min_obs}", {"n_eff": n_eff}, _margin(n_eff, pol.min_obs))
    if ev.effect <= pol.min_effect:
        return _result(g, False, f"effect {ev.effect:+.4f} not above {pol.min_effect}", {"effect": ev.effect})
    p = ev.p_value
    if not _fin(p):
        if not _fin(ev.t_stat):
            return _missing(g, "p_value or t_stat")
        p = one_sided_p(float(ev.t_stat))
    p_adj = adjusted_p(float(p), ev.n_tests_searched, pol.multiplicity)
    ok = p_adj <= pol.alpha
    return _result(g, ok, f"p={float(p):.3g}, adjusted for {ev.n_tests_searched} searched = {p_adj:.3g} vs alpha={pol.alpha}",
                   {"p": float(p), "p_adj": p_adj, "n_eff": n_eff, "n_tests": ev.n_tests_searched},
                   float((pol.alpha - p_adj) / pol.alpha))


def gate_incremental_value(ev: IncrementalEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "incremental_value"
    if ev is None or ev.delta_series is None:
        return _missing(g, "with-vs-without delta series")
    d = _clean(ev.delta_series, "delta_series")
    if d.size < pol.min_delta_periods:
        return _result(g, False, f"{d.size} delta periods < {pol.min_delta_periods}", {"n": int(d.size)}, _margin(d.size, pol.min_delta_periods))
    if ev.decision_overlap is not None and ev.decision_overlap > pol.max_incumbent_overlap:
        return _result(g, False, f"decisions {ev.decision_overlap:.1%} identical to incumbent: nothing incremental to measure",
                       {"overlap": ev.decision_overlap})
    lo, hi = block_bootstrap_ci(d, ev.seed, pol.bootstrap_n, pol.bootstrap_block)
    t = t_stat(d)
    mean = float(d.mean())
    ok = mean > pol.min_gain and lo > 0.0 and t >= pol.min_gain_t
    return _result(g, ok, f"mean gain {mean:+.5f}, bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}], t={t:.2f} (need CI>0, t>={pol.min_gain_t})",
                   {"mean": mean, "ci_lo": lo, "ci_hi": hi, "t": t, "n": int(d.size)}, _margin(t, pol.min_gain_t))


def gate_oos_confirmation(ev: OOSEvidence | None, pol: PromotionPolicy, now) -> GateResult:
    g = "oos_confirmation"
    if ev is None or not ev.oos_effects or ev.train_end is None or not _fin(ev.in_sample_effect):
        return _missing(g, "train_end, in-sample effect and out-of-sample effects")
    if len(ev.oos_dates) != len(ev.oos_effects):
        return _result(g, False, "oos_dates and oos_effects differ in length")
    if len(set(ev.oos_dates)) != len(ev.oos_dates):
        return _result(g, False, "duplicate oos dates: the same outcome would count twice")
    early = [d for d in ev.oos_dates if as_date(d) <= as_date(ev.train_end)]
    if early:
        return _result(g, False, f"{len(early)} 'out-of-sample' dates are not after train_end {ev.train_end}", {"early": early[:5]})
    for d in ev.oos_dates:
        require_past(d, now, "oos outcome")                 # outcome not yet known at `now` is a FirewallBreach, not a fail
    shared = sorted(set(ev.oos_windows) & set(ev.train_windows))
    if shared:
        return _result(g, False, f"OOS windows overlap training windows: {shared}")
    e = _clean(ev.oos_effects, "oos_effects")
    if e.size < pol.min_oos_periods:
        return _result(g, False, f"{e.size} OOS periods < {pol.min_oos_periods}", {"n": int(e.size)}, _margin(e.size, pol.min_oos_periods))
    m, t, pos = float(e.mean()), t_stat(e), positive_share(e)
    retention = m / ev.in_sample_effect if ev.in_sample_effect > 0 else float("nan")
    ok = m > 0 and t >= pol.min_oos_t and pos >= pol.min_oos_positive_share and retention >= pol.min_oos_retention
    return _result(g, ok, f"OOS mean {m:+.5f} (t={t:.2f}, {pos:.0%} positive), retains {retention:.0%} of in-sample "
                          f"(need >={pol.min_oos_retention:.0%})",
                   {"oos_mean": m, "t": t, "positive_share": pos, "retention": retention, "n": int(e.size)},
                   _margin(retention, pol.min_oos_retention) if math.isfinite(retention) else -1.0)


def gate_cross_context_transfer(ev: TransferEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "cross_context_transfer"
    if ev is None or not ev.context_effects or not _fin(ev.home_effect):
        return _missing(g, "per-context effects and the home effect")
    home = set(ev.home_contexts)
    away = {c: float(e) for c, e in ev.context_effects.items() if c not in home and ev.context_n.get(c, 0) >= pol.min_transfer_n}
    if len(away) < pol.min_transfer_contexts:
        return _result(g, False, f"{len(away)} usable contexts outside home (need {pol.min_transfer_contexts}, each n>={pol.min_transfer_n})",
                       {"usable": sorted(away)}, _margin(len(away), pol.min_transfer_contexts))
    vals = np.array(list(away.values()))
    share = positive_share(vals)
    ratio = float(np.median(vals) / ev.home_effect) if ev.home_effect > 0 else float("nan")
    worst = float(vals.min() / ev.home_effect) if ev.home_effect > 0 else float("nan")
    ok = (share >= pol.min_transfer_positive_share and ratio >= pol.min_transfer_ratio and worst >= pol.worst_transfer_floor)
    return _result(g, ok, f"{share:.0%} of {len(away)} away contexts positive, median transfer ratio {ratio:.2f}, worst {worst:.2f}x home",
                   {"positive_share": share, "median_ratio": ratio, "worst_ratio": worst, "worst_context": min(away, key=lambda c: away[c])},
                   _margin(ratio, pol.min_transfer_ratio) if math.isfinite(ratio) else -1.0)


def gate_risk_acceptance(ev: RiskEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "risk_acceptance"
    if ev is None or not (_fin(ev.worst_period) and _fin(ev.max_drawdown) and _fin(ev.cvar05)) or ev.catastrophic_count is None:
        return _missing(g, "worst period, max drawdown, CVaR5 and catastrophic count")
    if ev.n_periods < pol.min_risk_periods:
        return _result(g, False, f"risk measured on {ev.n_periods} periods < {pol.min_risk_periods}", {}, _margin(ev.n_periods, pol.min_risk_periods))
    problems = []
    if ev.worst_period < pol.worst_period_floor:
        problems.append(f"worst period {ev.worst_period:.1%} below {pol.worst_period_floor:.0%}")
    if ev.max_drawdown < pol.max_drawdown_floor:
        problems.append(f"max drawdown {ev.max_drawdown:.1%} below {pol.max_drawdown_floor:.0%}")
    if ev.cvar05 < pol.cvar05_floor:
        problems.append(f"CVaR5 {ev.cvar05:.1%} below {pol.cvar05_floor:.0%}")
    if ev.catastrophic_count > pol.max_catastrophic:
        problems.append(f"{ev.catastrophic_count} catastrophic periods")
    tol = pol.incumbent_risk_tolerance
    if _fin(ev.incumbent_worst_period) and ev.worst_period < ev.incumbent_worst_period - tol:
        problems.append(f"worst period worse than incumbent by {ev.incumbent_worst_period - ev.worst_period:.1%}")
    if _fin(ev.incumbent_max_drawdown) and ev.max_drawdown < ev.incumbent_max_drawdown - tol:
        problems.append(f"drawdown worse than incumbent by {ev.incumbent_max_drawdown - ev.max_drawdown:.1%}")
    return _result(g, not problems, "; ".join(problems) or "within all absolute floors and not worse than the incumbent",
                   {"worst": ev.worst_period, "mdd": ev.max_drawdown, "cvar05": ev.cvar05, "cat": ev.catastrophic_count},
                   _margin(ev.worst_period, pol.worst_period_floor))


def gate_anti_memorization(ev: MemorizationEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "anti_memorization"
    if ev is None or not _fin(ev.identity_shuffle_retention) or not _fin(ev.disguised_rerun_retention) \
            or ev.lookup_table_suspected is None or not _fin(ev.top_identity_share) or ev.n_distinct_identities is None:
        return _missing(g, "identity-shuffle and disguised-rerun retention, lookup-table check, identity concentration")
    problems = []
    ident = [n for n in ev.feature_names if IDENTITY_TOKENS.search(n)]
    if ident:
        problems.append(f"identity-bearing inputs: {ident[:5]}")
    if ev.lookup_table_suspected:
        problems.append("knowledge behaves as an answer lookup table")
    if ev.identity_shuffle_retention < pol.min_identity_retention:
        problems.append(f"effect collapses to {ev.identity_shuffle_retention:.0%} when identities are scrambled")
    if ev.disguised_rerun_retention < pol.min_identity_retention:
        problems.append(f"effect collapses to {ev.disguised_rerun_retention:.0%} on a disguised rerun")
    if ev.top_identity_share > pol.max_top_identity_share:
        problems.append(f"one identity carries {ev.top_identity_share:.0%} of the effect")
    if ev.n_distinct_identities < pol.min_distinct_identities:
        problems.append(f"only {ev.n_distinct_identities} distinct identities support it")
    return _result(g, not problems, "; ".join(problems) or "survives identity scrambling and disguise",
                   {"shuffle": ev.identity_shuffle_retention, "disguise": ev.disguised_rerun_retention,
                    "top_share": ev.top_identity_share}, _margin(min(ev.identity_shuffle_retention, ev.disguised_rerun_retention),
                                                                  pol.min_identity_retention))


def gate_future_information(ev: FutureAudit | None, prov_ok: bool, pol: PromotionPolicy, now) -> GateResult:
    g = "future_information_audit"
    if ev is None or not ev.audit_ran:
        return _missing(g, "a completed future-information audit")
    problems = list(ev.findings)
    if ev.sealed_windows_touched:
        problems.append(f"sealed windows touched: {list(ev.sealed_windows_touched)}")
    if ev.outcomes_after_now is None:
        problems.append("outcomes_after_now not measured")
    elif ev.outcomes_after_now > 0:
        problems.append(f"{ev.outcomes_after_now} outcomes dated at/after now")
    if ev.max_evidence_date is None:
        problems.append("max_evidence_date not recorded")
    else:
        try:
            require_past(ev.max_evidence_date, now, "newest evidence")
        except FirewallBreach as e:
            problems.append(str(e))
    if not prov_ok:
        problems.append("provenance says the item could not have existed at `now`")
    return _result(g, not problems, "; ".join(problems) or "audit clean, newest evidence strictly before now", {"findings": len(problems)})


def gate_reproducibility(ev: ReproEvidence | None, pol: PromotionPolicy, code_hash: str) -> GateResult:
    g = "reproducibility"
    if ev is None or len(ev.reruns) < pol.min_reruns:
        return _missing(g, f"at least {pol.min_reruns} reruns (has {len(ev.reruns) if ev else 0})")
    problems = []
    vals = np.array([r.value for r in ev.reruns], float)
    if not np.isfinite(vals).all():
        return _result(g, False, "a rerun produced a non-finite value")
    stale = sorted({r.code_hash for r in ev.reruns if r.code_hash != code_hash})
    if stale:
        problems.append(f"reruns made with other code {stale[:3]} (current {code_hash})")
    if ev.expected_data_hash and any(r.data_hash and r.data_hash != ev.expected_data_hash for r in ev.reruns):
        problems.append("a rerun used different data")
    by_seed: dict[int, list[float]] = {}
    for r in ev.reruns:
        by_seed.setdefault(r.seed, []).append(r.value)
    if len(by_seed) < 2:
        problems.append("all reruns share one seed: seed sensitivity untested")
    if not any(len(v) >= 2 for v in by_seed.values()):
        problems.append("no seed was repeated: determinism untested")
    worst_det = 0.0
    for s, v in by_seed.items():
        if len(v) >= 2:
            sp = float(max(v) - min(v))
            worst_det = max(worst_det, sp)
            if sp > pol.determinism_tol:
                problems.append(f"seed {s} not deterministic (spread {sp:.3g})")
    means = np.array([np.mean(v) for v in by_seed.values()])
    spread = rel_spread(means) if means.size > 1 else 0.0
    if spread > pol.seed_spread_tol:
        problems.append(f"across-seed spread {spread:.0%} > {pol.seed_spread_tol:.0%}")
    if not (np.all(vals > 0) or np.all(vals < 0)):
        problems.append("sign flips across reruns")
    return _result(g, not problems, "; ".join(problems) or f"{len(vals)} reruns on {len(by_seed)} seeds agree",
                   {"n": len(vals), "seeds": len(by_seed), "seed_spread": spread, "det_spread": worst_det},
                   _margin(pol.seed_spread_tol, spread))


def gate_stability(ev: StabilityEvidence | None, pol: PromotionPolicy) -> GateResult:
    g = "stability"
    if ev is None or len(ev.period_effects) < pol.min_stability_periods:
        return _missing(g, f"at least {pol.min_stability_periods} sub-period effects")
    p = _clean(ev.period_effects, "period_effects")
    problems = []
    share = positive_share(p)
    if share < pol.min_stability_positive_share:
        problems.append(f"only {share:.0%} of sub-periods positive")
    pos_sum = float(p[p > 0].sum())
    top = float(p.max() / pos_sum) if pos_sum > 0 else 1.0
    if top > pol.max_single_period_share:
        problems.append(f"one sub-period supplies {top:.0%} of the gain")
    half = p.size // 2
    if p[:half].mean() <= 0 or p[half:].mean() <= 0:
        problems.append("one half of history shows no effect")
    ref = ev.reference_effect if _fin(ev.reference_effect) else float(p.mean())
    ret = float("nan")
    if len(ev.perturbation_effects) < pol.min_perturbations:
        problems.append(f"only {len(ev.perturbation_effects)} perturbation runs (need {pol.min_perturbations})")
    else:
        pe = _clean(ev.perturbation_effects, "perturbation_effects")
        ret = float(pe.min() / ref) if ref > 0 else float("nan")
        if not (pe > 0).all() or not ret >= pol.min_perturbation_retention:
            problems.append(f"under perturbation the effect falls to {ret:.0%} of reference or changes sign")
    return _result(g, not problems, "; ".join(problems) or "consistent across sub-periods, halves and perturbations",
                   {"positive_share": share, "top_period_share": top, "perturbation_retention": ret},
                   _margin(share, pol.min_stability_positive_share))


def provenance_problems(k: Any, now) -> list[str]:
    """Completeness of the provenance a promoted item must carry (section 28: could this item exist at `now`?)."""
    prov = getattr(k, "provenance", None)
    if prov is None:
        return ["no provenance"]
    errs = list(prov.check())
    for f in ("data_hash", "config_hash", "experiment_id", "run_id"):
        if not getattr(prov, f, ""):
            errs.append(f"provenance.{f} missing")
    if prov.seed is None:
        errs.append("provenance.seed missing")
    if not prov.could_exist_at(now):
        errs.append("provenance: item could not have existed at `now`")
    return errs


def gate_provenance(k: Any, pol: PromotionPolicy, now) -> GateResult:
    errs = provenance_problems(k, now)
    return _result("provenance_completeness", not errs, "; ".join(errs) or "complete", {"problems": len(errs)})


# ------------------------------------------------------------------------------------------------ the gate
class PromotionGate:
    """Runs the ten gates and returns an immutable decision. Stateless apart from its policy and the code hash it audits
    reruns against, so the same inputs always give the same decision id (section 56)."""

    def __init__(self, policy: PromotionPolicy | None = None, code_hash: str | None = None):
        self.policy = policy or PromotionPolicy()
        errs = self.policy.validate()
        if errs:
            raise ValueError(f"invalid promotion policy: {errs}")
        self.code_hash = code_hash if code_hash is not None else current_code_hash()

    def applicable(self, gate: str, effects: Sequence[DecisionEffect]) -> tuple[bool, str]:
        """(applies, reason). Only non-critical gates can be waived; risk does not apply to pure research-priority knowledge."""
        if gate == "risk_acceptance" and effects and all(e in RISK_FREE_EFFECTS for e in effects):
            return False, "knowledge only orders research; it never sizes or selects a holding"
        if gate in self.policy.waivers:
            return False, f"waived in writing: {self.policy.waivers[gate]}"
        return True, ""

    def preconditions(self, k: Any) -> list[str]:
        out = KnowledgeLike.conforms(k)
        if out:
            return out
        if Promotion.parse(k.promotion) != Promotion.CHALLENGER:
            out.append(f"only a CHALLENGER may be promoted (is {k.promotion})")
        if Epistemic.parse(k.epistemic) not in PROMOTABLE_EPISTEMIC:
            out.append(f"epistemic state {k.epistemic} cannot reach production")
        effects = tuple(DecisionEffect.parse(e) for e in k.decision_effect)
        if not effects or all(e == DecisionEffect.NONE for e in effects):
            out.append("knowledge changes no decision (DecisionEffect.NONE): research knowledge only")
        return out

    def _run(self, name: str, fn) -> GateResult:
        try:
            return fn()
        except FirewallBreach:
            raise
        except Exception as e:                               # a gate that crashes has not passed
            return _result(name, False, f"gate raised {type(e).__name__}: {e}")

    def evaluate(self, k: Any, ev: PromotionEvidence, now) -> PromotionDecision:
        pol = self.policy
        pre = self.preconditions(k)
        effects = tuple(DecisionEffect.parse(e) for e in getattr(k, "decision_effect", ()) or ())
        prov_ok = getattr(getattr(k, "provenance", None), "could_exist_at", lambda n: False)(now)
        table = {
            "statistical_validity": lambda: gate_statistical_validity(ev.statistical, pol),
            "incremental_value": lambda: gate_incremental_value(ev.incremental, pol),
            "oos_confirmation": lambda: gate_oos_confirmation(ev.oos, pol, now),
            "cross_context_transfer": lambda: gate_cross_context_transfer(ev.transfer, pol),
            "risk_acceptance": lambda: gate_risk_acceptance(ev.risk, pol),
            "anti_memorization": lambda: gate_anti_memorization(ev.memorization, pol),
            "future_information_audit": lambda: gate_future_information(ev.future, prov_ok, pol, now),
            "reproducibility": lambda: gate_reproducibility(ev.repro, pol, self.code_hash),
            "stability": lambda: gate_stability(ev.stability, pol),
            "provenance_completeness": lambda: gate_provenance(k, pol, now),
        }
        results = []
        for name in GATES:
            applies, why = self.applicable(name, effects)
            if not applies:
                results.append(GateResult(name, NA, name in CRITICAL, why))
            else:
                results.append(self._run(name, table[name]))
        blocked = bool(pre) or any(not r.ok for r in results)
        return PromotionDecision(str(getattr(k, "knowledge_id", "?")), int(getattr(k, "version", 0)), str(as_date(now)),
                                 "BLOCK" if blocked else "PROMOTE", tuple(results), tuple(pre), pol.digest(), ev.digest(), self.code_hash)


# ------------------------------------------------------------------------------------------------ rejection report
REMEDIATION = {
    "statistical_validity": "collect more independent observations or state the discovery search size honestly",
    "incremental_value": "measure the with-vs-without delta on more periods; check the knowledge is not a duplicate of the incumbent",
    "oos_confirmation": "wait for outcomes after train_end, or evaluate on windows never used in training",
    "cross_context_transfer": "test on more stocks, sectors, eras or regimes outside the home context, or scope the knowledge narrowly and waive in writing",
    "risk_acceptance": "cap sizing, add an exit condition, or exclude the events that produce the tail losses",
    "anti_memorization": "remove identity-bearing inputs; rerun the identity-shuffle and disguised-rerun controls",
    "future_information_audit": "run the leak audit; remove any evidence dated at or after the decision time",
    "reproducibility": "rerun with the current code on two or more seeds, repeating one seed to prove determinism",
    "stability": "run sub-period and parameter-perturbation checks; the effect must not hinge on one period",
    "provenance_completeness": "fill the missing provenance fields from the experiment record",
}


def near_misses(decision: PromotionDecision, within: float = 0.25) -> list[tuple[str, float]]:
    """Failed gates whose measured margin was within `within` of the threshold: the cheapest ones to fix with more evidence."""
    return sorted(((r.gate, r.margin) for r in decision.failed if r.margin is not None and -within <= r.margin < 0),
                  key=lambda t: -t[1])


def render_rejection_report(decision: PromotionDecision) -> str:
    """The written explanation that must accompany every BLOCK (and a short record for a PROMOTE)."""
    d = decision
    lines = [f"# Promotion decision {d.decision_id}: {d.verdict}", "",
             f"knowledge {d.knowledge_id} v{d.version}, decided as of {d.now}, code {d.code_hash}, policy {d.policy_digest}",
             "Status: IMPLEMENTED - NOT VALIDATED (gate code has not been validated on real data)", ""]
    if d.preconditions:
        lines += ["## Failed preconditions"] + [f"- {p}" for p in d.preconditions] + [""]
    lines.append("## Gates")
    for r in d.results:
        tag = "CRITICAL " if r.critical and not r.ok else ""
        lines.append(f"- [{r.status}] {tag}{r.gate}: {r.detail}")
    if d.verdict == "BLOCK":
        lines += ["", "## Why promotion is blocked"]
        lines.append(f"- critical failures: {', '.join(d.critical_failures) or 'none'}")
        lines.append(f"- other failures: {', '.join(r.gate for r in d.failed if not r.critical) or 'none'}")
        nm = near_misses(d)
        if nm:
            lines.append("- near misses: " + ", ".join(f"{g} ({m:+.0%})" for g, m in nm))
        lines += ["", "## What evidence would change this"]
        lines += [f"- {r.gate}: {REMEDIATION[r.gate]}" for r in d.failed]
        lines.append("A failed promotion leaves the champion untouched. The challenger stays a challenger; nothing is deleted.")
    return "\n".join(lines) + "\n"


def write_rejection_report(directory: str | os.PathLike, decision: PromotionDecision) -> Path:
    """Persist the report (.md) and the machine-readable decision (.json). Write-once: an existing decision is never edited."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{decision.knowledge_id}_v{decision.version}_{decision.decision_id}"
    md, js = d / f"{stem}.md", d / f"{stem}.json"
    if md.exists() or js.exists():
        raise FileExistsError(f"decision report {stem} already written; decisions are immutable")
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(render_rejection_report(decision), encoding="utf-8")
    os.replace(tmp, md)
    js.write_text(json.dumps(decision.to_dict(), indent=1, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
    return md


def failure_statistics(decisions: Sequence[PromotionDecision]) -> dict:
    """Which gates block most often across many decisions. A gate that never fails may be too lax; one that always fails
    may be unreachable with the evidence the system can produce (feeds the research-priority engine)."""
    n = len(decisions)
    out = {g: {"fail": 0, "missing": 0, "pass": 0, "na": 0} for g in GATES}
    for d in decisions:
        for r in d.results:
            key = {PASS: "pass", FAIL: "fail", MISSING: "missing", NA: "na"}[r.status]
            out[r.gate][key] += 1
    return {"n_decisions": n, "promoted": sum(d.promote for d in decisions),
            "by_gate": out, "never_failed": [g for g, c in out.items() if n and c["fail"] + c["missing"] == 0],
            "always_blocked": [g for g, c in out.items() if n and c["fail"] + c["missing"] == n]}


# ------------------------------------------------------------------------------------------------ evidence lint
def validate_evidence(ev: PromotionEvidence, now) -> list[str]:
    """Structural problems that would make a gate's answer meaningless, found BEFORE gating so the report can say
    'malformed' instead of 'weak'. Never raises for bad data; returns human-readable findings (empty = well formed)."""
    out = []

    def bad(name: str, seq) -> None:
        if seq is not None and len(seq) and not np.isfinite(np.asarray(list(seq), float)).all():
            out.append(f"{name} contains non-finite values")

    if ev.incremental is not None:
        bad("incremental.delta_series", ev.incremental.delta_series)
        if ev.incremental.decision_overlap is not None and not 0.0 <= ev.incremental.decision_overlap <= 1.0:
            out.append("incremental.decision_overlap outside [0, 1]")
    if ev.oos is not None:
        bad("oos.oos_effects", ev.oos.oos_effects)
        try:
            ds = [as_date(d) for d in ev.oos.oos_dates]
            if ds != sorted(ds):
                out.append("oos.oos_dates are not in time order")
        except ValueError:
            out.append("oos.oos_dates contains an unparseable date")
    if ev.transfer is not None:
        bad("transfer.context_effects", list(ev.transfer.context_effects.values()))
        missing_n = sorted(set(ev.transfer.context_effects) - set(ev.transfer.context_n))
        if missing_n:
            out.append(f"transfer.context_n missing for {missing_n[:5]}")
    if ev.risk is not None and ev.risk.catastrophic_count is not None and ev.risk.catastrophic_count < 0:
        out.append("risk.catastrophic_count is negative")
    if ev.memorization is not None:
        for f in ("identity_shuffle_retention", "disguised_rerun_retention", "top_identity_share"):
            v = getattr(ev.memorization, f)
            if v is not None and not 0.0 <= v <= 5.0:
                out.append(f"memorization.{f}={v} is implausible")
    if ev.stability is not None:
        bad("stability.period_effects", ev.stability.period_effects)
        bad("stability.perturbation_effects", ev.stability.perturbation_effects)
    if ev.repro is not None:
        bad("repro.reruns", [r.value for r in ev.repro.reruns])
    if ev.future is not None and ev.future.max_evidence_date is not None:
        try:
            if as_date(ev.future.max_evidence_date) >= as_date(now):
                out.append("future.max_evidence_date is not before now")
        except ValueError:
            out.append("future.max_evidence_date unparseable")
    return out


def evidence_completeness(ev: PromotionEvidence) -> dict:
    """Which evidence members were supplied. A promotion attempt with completeness < 1 cannot pass; the share tells the
    research planner how much measuring is left."""
    have = {f.name: getattr(ev, f.name) is not None for f in dataclasses.fields(ev)}
    return {"share": sum(have.values()) / len(have), "missing": sorted(k for k, v in have.items() if not v)}


# ------------------------------------------------------------------------------------------------ decision comparison
def compare_decisions(a: PromotionDecision, b: PromotionDecision) -> dict:
    """Gate-by-gate change between two decisions on the same knowledge (e.g. before/after collecting more evidence)."""
    sa, sb = {r.gate: r.status for r in a.results}, {r.gate: r.status for r in b.results}
    changed = {g: (sa[g], sb[g]) for g in GATES if sa.get(g) != sb.get(g)}
    return {"changed": changed, "fixed": sorted(g for g, (x, y) in changed.items() if x != PASS and y == PASS),
            "regressed": sorted(g for g, (x, y) in changed.items() if x == PASS and y != PASS),
            "same_policy": a.policy_digest == b.policy_digest, "same_evidence": a.evidence_digest == b.evidence_digest,
            "same_code": a.code_hash == b.code_hash}


def sensitivity(k: Any, ev: PromotionEvidence, now, policy: PromotionPolicy | None = None, code_hash: str | None = None,
                factors: Sequence[float] = (0.8, 1.25)) -> dict:
    """Does the verdict depend on knife-edge thresholds? Re-evaluate with every numeric threshold tightened and loosened by
    each factor (in the stricter/looser direction of THAT threshold). A PROMOTE that flips to BLOCK under a 25% tightening is
    'fragile' and is reported as such; a BLOCK that flips to PROMOTE only under loosening shows exactly how much a
    threshold change would have been needed - the temptation to resist (section 1.2: no manual threshold changes after
    seeing results)."""
    pol = policy or PromotionPolicy()
    base = PromotionGate(pol, code_hash).evaluate(k, ev, now)
    flips = []
    for name in sorted(set(HIGHER_IS_STRICTER) | set(LOWER_IS_STRICTER)):
        v = getattr(pol, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        for f in factors:
            stricter_up = name in HIGHER_IS_STRICTER
            nv = v * f if v >= 0 else v / f                          # negative floors move toward/away from zero
            if isinstance(v, int):
                nv = int(round(nv))
            if nv == v:
                continue
            direction = "tighter" if ((nv > v) == stricter_up) else "looser"
            try:
                d = PromotionGate(dataclasses.replace(pol, **{name: cast(Any, nv)}), code_hash).evaluate(k, ev, now)
            except ValueError:
                continue
            if d.verdict != base.verdict:
                flips.append({"threshold": name, "from": v, "to": nv, "direction": direction, "verdict": d.verdict})
    return {"base": base.verdict, "flips": flips,
            "fragile": base.promote and any(f["direction"] == "tighter" for f in flips),
            "needs_loosening": (not base.promote) and any(f["direction"] == "looser" for f in flips),
            "robust": not flips}


# ------------------------------------------------------------------------------------------------ policy accountability
HIGHER_IS_STRICTER = frozenset({f.name for f in dataclasses.fields(PromotionPolicy) if f.name.startswith("min_")} |
                               {"worst_period_floor", "max_drawdown_floor", "cvar05_floor", "worst_transfer_floor"})
LOWER_IS_STRICTER = frozenset({"alpha", "max_incumbent_overlap", "max_top_identity_share", "max_single_period_share", "max_catastrophic",
                               "incumbent_risk_tolerance", "determinism_tol", "seed_spread_tol"})


def diff_policies(old: PromotionPolicy, new: PromotionPolicy) -> list[dict]:
    """Every threshold that differs, labelled tighter or looser. Loosening a threshold is the classic way to 'learn' by
    moving the goalposts, so it is always listed explicitly."""
    out = []
    for f in dataclasses.fields(PromotionPolicy):
        a, b = getattr(old, f.name), getattr(new, f.name)
        if a == b:
            continue
        if f.name in HIGHER_IS_STRICTER:
            d = "tighter" if b > a else "looser"
        elif f.name in LOWER_IS_STRICTER:
            d = "tighter" if b < a else "looser"
        else:
            d = "changed"
        out.append({"field": f.name, "old": a, "new": b, "direction": d})
    return out


class PolicyLog:
    """Append-only, hash-chained log of the promotion policies in force (engine.champion.Ledger). Loosening a threshold needs a
    written reason AND is flagged; the log makes 'the gate was quietly relaxed after seeing results' visible in review."""

    def __init__(self, path: str | os.PathLike):
        from engine.champion import Ledger
        self.ledger = Ledger(path)

    def current(self) -> PromotionPolicy | None:
        rows = self.ledger.rows()
        if not rows:
            return None
        d = dict(rows[-1]["detail"]["policy"])
        d["waivers"] = dict(d.get("waivers") or {})
        return PromotionPolicy(**d)

    def record(self, policy: PromotionPolicy, now, reason: str = "") -> dict:
        errs = policy.validate()
        if errs:
            raise ValueError(f"invalid policy: {errs}")
        prev = self.current()
        changes = diff_policies(prev, policy) if prev is not None else []
        if not changes and prev is not None:
            return {"recorded": False, "changes": [], "loosened": []}
        loosened = [c for c in changes if c["direction"] == "looser"]
        if loosened and len(reason.strip()) < 20:
            raise ValueError(f"loosening {[c['field'] for c in loosened]} needs a written reason of at least 20 characters")
        self.ledger.append("policy", policy.digest(), str(as_date(now)), policy=json.loads(json.dumps(dataclasses.asdict(policy))),
                           reason=reason, changes=changes)
        return {"recorded": True, "changes": changes, "loosened": loosened}

    def history(self) -> list[dict]:
        return [{"t": r["t"], "digest": r["id"], "reason": r["detail"].get("reason", ""),
                 "loosened": [c["field"] for c in r["detail"].get("changes", []) if c["direction"] == "looser"]}
                for r in self.ledger.rows()]

    def loosened_since(self, digest: str) -> list[str]:
        """Fields loosened after the policy `digest` was in force: a promotion decided under `digest` may need re-deciding."""
        seen, out = False, []
        for h in self.history():
            if seen:
                out += h["loosened"]
            seen = seen or h["digest"] == digest
        return sorted(set(out))
