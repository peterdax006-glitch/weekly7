"""Research-brain health (contract C66 section 37; C66 section 43 'do not optimise the wrong metric'; checklist R-series).

The autonomous researcher itself can fail. This module watches the researcher, not the market: from a ledger of matured
experiment outcomes it measures research diversity, duplicate rate, success rate, false-discovery rate, replication rate,
transfer rate, compute efficiency, knowledge churn, overfitting rate, memorisation rate and research concentration, each with an
uncertainty interval and an honest UNKNOWN when there is too little evidence. On top of the metrics sit the pathology detectors
the contract names: 90% of compute on one tiny parameter family that yields negligible gain, abandoning hard questions, an
easy-area bias (research only where success is cheap), and the section-43 trap of counting experiments instead of verified
knowledge. Findings turn into Directives (area multipliers, family caps, questions to reopen) that engine.research.diversity
and engine.learning.research_policy consume; nothing here mutates the policy itself.

Built on: engine.learning.research_policy (audit_spend, allocation_entropy, epsilon_schedule, marginal_return_verdict,
cusum_shift, RealisedGain, ResearchTarget), engine.learning.meta_learning (wilson, group_rates) and engine.learning.core
(Health, require_past, stable_hash). Time-aware: every entry takes `now` and raises FirewallBreach for an outcome dated at or after
it. Public entry: step(outcomes, now, ...). IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from engine.learning import research_policy as RP
from engine.learning.core import (FirewallBreach, Health, ValidationLabel, as_date, current_code_hash, require_past,
                                  stable_hash, _StrEnum)
from engine.learning.meta_learning import auc, group_rates, wilson
from engine.research.core import Problem

LABEL = ValidationLabel.NOT_VALIDATED.value
EPS = 1e-9
_CODE_HASH: list = []


def code_hash() -> str:
    """Code hash stamped on reports; computed once per process because hashing the loaded modules costs ~0.2 s."""
    if not _CODE_HASH:
        _CODE_HASH.append(current_code_hash())
    return _CODE_HASH[0]


# ------------------------------------------------------------------------------------------------ shared vocabulary

class Area(_StrEnum):
    """Section 38: the ten places research effort is spread. Shared with engine.research.diversity."""
    KNOWN_PROMISING = "KNOWN_PROMISING"
    UNCERTAIN = "UNCERTAIN"
    NEW_REPRESENTATION = "NEW_REPRESENTATION"
    FAILED_NEW_HYPOTHESIS = "FAILED_NEW_HYPOTHESIS"
    RISK = "RISK"
    VOLATILITY = "VOLATILITY"
    DIRECTION = "DIRECTION"
    REGIME = "REGIME"
    DATA_QUALITY = "DATA_QUALITY"
    PATTERN_BREAK = "PATTERN_BREAK"


AREAS = tuple(Area)
EXPLOIT_AREAS = frozenset({Area.KNOWN_PROMISING})

# each area's home in the existing ten-target allocator (research_policy) and in the objective hierarchy (research core)
AREA_TO_TARGET = {
    Area.KNOWN_PROMISING: RP.ResearchTarget.KNOWN_RELIABLE,
    Area.UNCERTAIN: RP.ResearchTarget.UNKNOWN_AREA,
    Area.NEW_REPRESENTATION: RP.ResearchTarget.NEW_REPRESENTATION,
    Area.FAILED_NEW_HYPOTHESIS: RP.ResearchTarget.FAILURE,
    Area.RISK: RP.ResearchTarget.WEAK_PATTERN,
    Area.VOLATILITY: RP.ResearchTarget.MISSED_WINNER,
    Area.DIRECTION: RP.ResearchTarget.INTERACTION,
    Area.REGIME: RP.ResearchTarget.REGIME_TRANSITION,
    Area.DATA_QUALITY: RP.ResearchTarget.DATA_QUALITY,
    Area.PATTERN_BREAK: RP.ResearchTarget.CONTRADICTION,
}
AREA_TO_PROBLEM = {
    Area.RISK: Problem.LOSS_AVOIDANCE, Area.VOLATILITY: Problem.VOLATILITY, Area.DIRECTION: Problem.DIRECTION,
    Area.DATA_QUALITY: Problem.DATA_QUALITY, Area.KNOWN_PROMISING: Problem.CONSISTENCY,
}


class Level(_StrEnum):
    """Severity of a metric or finding. UNKNOWN is never OK: too little evidence stays visible (contract section 33)."""
    OK = "OK"
    WATCH = "WATCH"
    ALARM = "ALARM"
    UNKNOWN = "UNKNOWN"


_LEVEL_RANK = {Level.OK: 0, Level.UNKNOWN: 1, Level.WATCH: 2, Level.ALARM: 3}
_LEVEL_TO_HEALTH = {Level.OK: Health.HEALTHY, Level.WATCH: Health.DEGRADING, Level.ALARM: Health.BROKEN,
                    Level.UNKNOWN: Health.INSUFFICIENT_EVIDENCE}


def worst_level(levels: Iterable[Level]) -> Level:
    ls = list(levels)
    return max(ls, key=lambda l: _LEVEL_RANK[l]) if ls else Level.UNKNOWN


def to_learning_health(level: Level) -> Health:
    """Bridge to the C62 health vocabulary so the brain shows up in the same dashboards as learned knowledge."""
    return _LEVEL_TO_HEALTH[Level.parse(level)]


# ------------------------------------------------------------------------------------------------ records

@dataclass(frozen=True)
class Outcome:
    """One matured experiment as the researcher sees it. `difficulty` is the PRE-RUN prior (0 easy .. 1 hard) - it must never be
    derived from the result, or hard-question detection would be circular. None means 'not yet tested' and is never read as False."""
    exp_id: str
    when: str                                       # real date the outcome matured (trusted side)
    area: str
    family: str                                     # hypothesis / parameter family the job belongs to
    cost_minutes: float
    success: bool                                   # produced a verified useful result
    gain_bits: float = 0.0
    difficulty: float = 0.5
    question_id: str = ""
    claimed_discovery: bool = False
    p_value: float | None = None
    replicated: bool | None = None
    transferred: bool | None = None
    false_discovery: bool | None = None             # later verdict on a claim: True = it was not real
    duplicate_of: str = ""
    train_score: float | None = None
    holdout_score: float | None = None
    memorised: bool | None = None                   # True = the memorisation control succeeded, i.e. the result was memorised
    exploratory: bool = False                       # the job tried something genuinely new (feeds engine.research.diversity)
    config_hash: str = ""                           # hash of the exact configuration run; repeats reveal untagged duplicates

    def validate(self) -> list:
        errs = []
        if not self.exp_id:
            errs.append("exp_id empty")
        try:
            Area.parse(self.area)
        except ValueError:
            errs.append(f"unknown area {self.area!r}")
        if not math.isfinite(self.cost_minutes) or self.cost_minutes < 0:
            errs.append("cost_minutes must be finite and >= 0")
        if not math.isfinite(self.gain_bits) or self.gain_bits < 0:
            errs.append("gain_bits must be finite and >= 0")
        if not 0.0 <= self.difficulty <= 1.0:
            errs.append("difficulty outside [0,1]")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            errs.append("p_value outside [0,1]")
        if self.false_discovery and self.replicated:
            errs.append("a claim cannot be both false and replicated")
        if self.transferred and self.replicated is False:
            errs.append("transferred without replication")
        return errs

    @property
    def verified(self) -> bool:
        """The unit of real progress (section 43): a claim that replicated, transferred, was not memorised and was not false."""
        return bool(self.claimed_discovery and self.replicated and self.transferred and not self.memorised
                    and not self.false_discovery)


@dataclass(frozen=True)
class KnowledgeEvent:
    """Creation, retirement or revival of a piece of knowledge; drives the churn metric and the flip-flop detector."""
    kid: str
    when: str
    kind: str                                       # created | retired | revived

    def validate(self) -> list:
        return [] if self.kind in ("created", "retired", "revived") and self.kid else ["bad knowledge event"]


@dataclass(frozen=True)
class HealthConfig:
    """Every threshold in one place. Defaults are priors to be calibrated against planted worlds, not tuned values."""
    window: int = 60                                # outcomes per comparison window (recent vs the window before it)
    min_n: int = 15                                 # below this a metric is UNKNOWN
    n_areas: int = len(AREAS)
    diversity_warn: float = 0.60                    # normalised area entropy
    diversity_alarm: float = 0.40
    duplicate_warn: float = 0.15
    duplicate_alarm: float = 0.30
    success_floor: float = 0.02                     # a researcher that never succeeds is broken (or asking the impossible)
    success_ceiling: float = 0.80                   # ... and one that always succeeds is asking easy questions
    fdr_warn: float = 0.10
    fdr_alarm: float = 0.25
    replication_floor_warn: float = 0.50
    replication_floor_alarm: float = 0.30
    transfer_floor_warn: float = 0.40
    transfer_floor_alarm: float = 0.20
    efficiency_ratio_warn: float = 0.60             # recent bits/minute over the previous window
    efficiency_ratio_alarm: float = 0.35
    waste_warn: float = 0.25                        # share of minutes spent on repeats, memorised results or false claims
    waste_alarm: float = 0.50
    churn_warn: float = 0.50
    flipflop_warn: float = 0.15
    overfit_warn: float = 0.30
    overfit_alarm: float = 0.50
    overfit_gap: float = 0.5                        # holdout below (1 - gap) x train counts as overfit
    memorisation_warn: float = 0.05
    memorisation_alarm: float = 0.15
    top_family_warn: float = 0.50
    top_family_alarm: float = 0.75
    monopoly_share: float = 0.90                    # section 37: the named tripwire
    monopoly_soft_share: float = 0.60
    negligible_bits_per_min: float = 0.002
    family_cap: float = 0.40                        # the cap a redirect imposes
    hard_cut: float = 0.65
    hard_share_drop: float = 0.5                    # recent hard share below this fraction of the prior one = abandonment
    hard_share_floor: float = 0.10
    stale_days: int = 45
    min_hard_attempts: int = 3
    easy_bias_margin: float = 0.15
    easy_corr: float = 0.70
    bucket_days: int = 30
    explore_min_factor: float = 0.6                 # observed explore share below this x epsilon_schedule is under-exploring
    min_hold: int = 2                               # consecutive reports before a ledger calls an alarm persistent
    penalty_strength: float = 0.5
    seed: int = 0

    def validate(self) -> list:
        errs = []
        if self.window < 2 or self.min_n < 2 or self.min_n > 4 * self.window:
            errs.append("window/min_n inconsistent")
        for a, b in (("diversity_alarm", "diversity_warn"), ("duplicate_warn", "duplicate_alarm"), ("fdr_warn", "fdr_alarm"),
                     ("overfit_warn", "overfit_alarm"), ("memorisation_warn", "memorisation_alarm"),
                     ("top_family_warn", "top_family_alarm"), ("replication_floor_alarm", "replication_floor_warn"),
                     ("transfer_floor_alarm", "transfer_floor_warn"), ("efficiency_ratio_alarm", "efficiency_ratio_warn"),
                     ("monopoly_soft_share", "monopoly_share"), ("waste_warn", "waste_alarm")):
            if getattr(self, a) > getattr(self, b):
                errs.append(f"{a} must not exceed {b}")
        if not 0 < self.family_cap < self.monopoly_share:
            errs.append("family_cap must lie below monopoly_share")
        if not 0 < self.hard_cut < 1:
            errs.append("hard_cut must be in (0,1)")
        return errs


# ------------------------------------------------------------------------------------------------ small statistics

def visible(outcomes: Iterable[Outcome], now) -> list:
    """Chronological outcomes, each strictly before `now`. Anything dated at/after `now` is a firewall breach, not a filter:
    a caller that hands a future outcome to a health check has a bug that would silently corrupt every metric."""
    rows = []
    for o in outcomes:
        require_past(o.when, now, f"outcome {o.exp_id}")
        rows.append(o)
    rows.sort(key=lambda o: (as_date(o.when), o.exp_id))
    return rows


def recent_and_prior(rows: Sequence, window: int, prior_windows: int = 1) -> tuple:
    """Last `window` rows and the `prior_windows` x `window` rows before them (the baseline the recent window is judged by)."""
    cut = max(0, len(rows) - window)
    return list(rows[-window:]), list(rows[max(0, cut - prior_windows * window):cut])


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """z for p1 - p2 with a pooled variance; None when either side is empty or the pooled variance is zero."""
    if n1 <= 0 or n2 <= 0:
        return None
    p = (k1 + k2) / (n1 + n2)
    v = p * (1 - p) * (1 / n1 + 1 / n2)
    if v <= 0:
        return None
    return float((k1 / n1 - k2 / n2) / math.sqrt(v))


def shannon_bits(weights: Iterable[float]) -> float:
    w = np.array([x for x in weights if x > 0], float)
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(-(p * np.log2(p)).sum())


def normalised_entropy(weights: Iterable[float], k: int) -> float:
    """Entropy over the k possible categories, 1 = perfectly spread, 0 = everything in one. Zero-weight categories count."""
    if k <= 1:
        return 0.0
    return min(1.0, shannon_bits(weights) / math.log2(k))


def hhi(weights: Iterable[float]) -> float:
    w = np.array([max(x, 0.0) for x in weights], float)
    return float(((w / w.sum()) ** 2).sum()) if w.sum() > 0 else 0.0


def effective_number(weights: Iterable[float]) -> float:
    """Hill number of order 1 (exp of Shannon entropy): how many equally-used categories the spread is worth."""
    return 2.0 ** shannon_bits(weights)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() < EPS or ry.std() < EPS:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05) -> list:
    """Indices rejected at FDR level q by the step-up BH procedure. Used to compare what the researcher CLAIMED with what
    survives multiple-testing correction: claim inflation is a health signal in its own right."""
    p = np.asarray(pvals, float)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="stable")
    thresh = q * (np.arange(1, p.size + 1) / p.size)
    passed = np.nonzero(p[order] <= thresh)[0]
    if passed.size == 0:
        return []
    return sorted(int(i) for i in order[: passed.max() + 1])


def minutes_by(rows: Sequence[Outcome], key) -> dict:
    out: dict = defaultdict(float)
    for o in rows:
        out[key(o)] += max(o.cost_minutes, 0.0)
    return dict(out)


def _share_weights(rows: Sequence[Outcome], key) -> dict:
    """Compute-share of each category. Falls back to counts when no minutes were recorded so nothing divides by zero."""
    m = minutes_by(rows, key)
    if sum(m.values()) <= 0:
        m = dict(Counter(key(o) for o in rows))
    return m


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None
    n: int
    lo: float | None
    hi: float | None
    level: Level
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "n": self.n, "lo": self.lo, "hi": self.hi, "level": self.level.value,
                "note": self.note}


def _unknown(name: str, n: int, need: int, what: str = "") -> Metric:
    return Metric(name, None, n, None, None, Level.UNKNOWN, f"only {n} of {need} needed{(' ' + what) if what else ''}")


def rate_metric(name: str, k: int, n: int, cfg: HealthConfig, warn: float, alarm: float, bad: str = "high",
                min_n: int | None = None, note: str = "") -> Metric:
    """A proportion with a Wilson interval. `bad`='high': ALARM needs the point estimate past `alarm` AND the interval's lower
    end past `warn` (one unlucky window is not an alarm); `bad`='low' mirrors it. n below min_n is UNKNOWN, never OK."""
    need = cfg.min_n if min_n is None else min_n
    if n < need:
        return _unknown(name, n, need)
    lo, hi = wilson(k, n)
    v = k / n
    if bad == "high":
        level = Level.ALARM if (v >= alarm and lo >= warn) else Level.WATCH if v >= warn else Level.OK
    else:
        level = Level.ALARM if (v <= alarm and hi <= warn) else Level.WATCH if v <= warn else Level.OK
    return Metric(name, v, n, lo, hi, level, note)


# ------------------------------------------------------------------------------------------------ the eleven metrics

def metric_diversity(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Normalised entropy of compute over the section-38 areas (0 = one area, 1 = uniform)."""
    if len(rows) < cfg.min_n:
        return _unknown("diversity", len(rows), cfg.min_n)
    w = _share_weights(rows, lambda o: o.area)
    h = normalised_entropy(w.values(), cfg.n_areas)
    eff = effective_number(w.values())
    level = Level.ALARM if h < cfg.diversity_alarm else Level.WATCH if h < cfg.diversity_warn else Level.OK
    return Metric("diversity", h, len(rows), None, None, level, f"{eff:.1f} effective areas of {cfg.n_areas}; "
                  f"{len(w)} touched")


def metric_duplicate_rate(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    implied = implied_duplicates(rows)
    is_dup = [bool(o.duplicate_of) or o.exp_id in implied for o in rows]
    k = sum(is_dup)
    wasted = sum(o.cost_minutes for o, d in zip(rows, is_dup) if d)
    total = sum(o.cost_minutes for o in rows)
    note = (f"{wasted:.0f} of {total:.0f} minutes went to re-running known experiments; {len(implied)} untagged repeats"
            if total > 0 else "")
    return rate_metric("duplicate_rate", k, len(rows), cfg, cfg.duplicate_warn, cfg.duplicate_alarm, "high", note=note)


def metric_success_rate(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Success is judged on BOTH sides: a rate whose interval sits below the floor means the researcher finds nothing; one
    above the ceiling means it only asks questions it already knows the answer to."""
    n = len(rows)
    if n < cfg.min_n:
        return _unknown("success_rate", n, cfg.min_n)
    k = sum(1 for o in rows if o.success)
    lo, hi = wilson(k, n)
    v = k / n
    if hi < cfg.success_floor:
        level, note = Level.ALARM, "interval entirely below the floor: nothing is being found"
    elif lo > cfg.success_ceiling:
        level, note = Level.WATCH, "interval entirely above the ceiling: the questions may be too easy"
    else:
        level, note = Level.OK, ""
    return Metric("success_rate", v, n, lo, hi, level, note)


def metric_fdr(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Of the claimed discoveries that have since been judged, the share that were not real. Also compares the number claimed
    with what survives BH correction on the recorded p-values: a researcher that claims far more than BH allows inflates."""
    judged = [o for o in rows if o.claimed_discovery and o.false_discovery is not None]
    m = rate_metric("false_discovery_rate", sum(1 for o in judged if o.false_discovery), len(judged), cfg, cfg.fdr_warn,
                    cfg.fdr_alarm, "high", min_n=max(5, cfg.min_n // 3))
    ps = [(i, o.p_value) for i, o in enumerate([o for o in rows if o.claimed_discovery]) if o.p_value is not None]
    if len(ps) >= 5:
        kept = benjamini_hochberg([p for _, p in ps], 0.05)
        inflated = 1.0 - len(kept) / len(ps)
        extra = f"; {len(ps) - len(kept)} of {len(ps)} p-valued claims would not survive BH ({inflated:.0%})"
        level = m.level
        if inflated > 0.5 and level in (Level.OK, Level.UNKNOWN):
            level = Level.WATCH
        return dataclasses.replace(m, level=level, note=(m.note + extra).strip("; "))
    return m


def metric_replication(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    tested = [o for o in rows if o.claimed_discovery and o.replicated is not None]
    return rate_metric("replication_rate", sum(1 for o in tested if o.replicated), len(tested), cfg,
                       cfg.replication_floor_warn, cfg.replication_floor_alarm, "low", min_n=max(5, cfg.min_n // 3))


def metric_transfer(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Transfer is asked only of claims that replicated in-sample; a claim not yet tested for transfer is not a failure."""
    tested = [o for o in rows if o.claimed_discovery and o.replicated and o.transferred is not None]
    return rate_metric("transfer_rate", sum(1 for o in tested if o.transferred), len(tested), cfg, cfg.transfer_floor_warn,
                       cfg.transfer_floor_alarm, "low", min_n=max(5, cfg.min_n // 3))


def bits_per_minute(rows: Sequence[Outcome]) -> float | None:
    m = sum(o.cost_minutes for o in rows)
    return sum(o.gain_bits for o in rows) / m if m > 0 else None


def metric_efficiency(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Information per CPU-minute, recent window against the one before it, plus the share of minutes spent on repeats, memorised
    results or claims that proved false (honest negative results are not waste)."""
    recent, prior = recent_and_prior(rows, cfg.window, prior_windows=3)
    if len(recent) < cfg.min_n:
        return _unknown("compute_efficiency", len(recent), cfg.min_n)
    r = bits_per_minute(recent)
    p = bits_per_minute(prior) if len(prior) >= cfg.min_n else None
    mins = sum(o.cost_minutes for o in recent)
    implied = implied_duplicates(recent)
    bad = [o for o in recent if o.duplicate_of or o.exp_id in implied or o.memorised or (o.claimed_discovery and o.false_discovery)]
    waste = (sum(o.cost_minutes for o in bad) / mins) if mins > 0 else 0.0     # a clean null answer is NOT waste
    level, notes = Level.OK, [f"{waste:.0%} of minutes repeated, memorised or misled"]
    if waste >= cfg.waste_alarm:
        level = Level.ALARM
    elif waste >= cfg.waste_warn:
        level = Level.WATCH
    ratio = None
    if r is not None and p is not None and p > EPS:
        ratio = r / p
        notes.append(f"{ratio:.2f}x the previous windows")
        # gains are heavy-tailed, so a low ratio alone is noise: it must also be a significant drop in per-job gain (Welch z)
        g_r = np.array([o.gain_bits for o in recent], float)
        g_p = np.array([o.gain_bits for o in prior], float)
        se = math.sqrt(g_r.var(ddof=1) / len(g_r) + g_p.var(ddof=1) / len(g_p)) if len(g_r) > 1 and len(g_p) > 1 else 0.0
        z = (g_r.mean() - g_p.mean()) / se if se > 0 else 0.0
        if ratio < cfg.efficiency_ratio_alarm and z < -2.5:
            level = Level.ALARM
        elif ratio < cfg.efficiency_ratio_warn and z < -1.645 and level is Level.OK:
            level = Level.WATCH
    return Metric("compute_efficiency", r, len(recent), None, None, level, "; ".join(notes))


def metric_churn(events: Sequence[KnowledgeEvent], now, cfg: HealthConfig) -> Metric:
    """Knowledge churn: events per unit of standing knowledge over the window, and the flip-flop share - items that were
    created, retired and revived repeatedly are the researcher changing its mind without learning."""
    ev = []
    for e in events:
        require_past(e.when, now, f"knowledge event {e.kid}")
        ev.append(e)
    if len(ev) < cfg.min_n // 3:
        return _unknown("knowledge_churn", len(ev), cfg.min_n // 3, "events")
    ev.sort(key=lambda e: (as_date(e.when), e.kid))
    live: set[str] = set()
    for e in ev:
        (live.add if e.kind in ("created", "revived") else live.discard)(e.kid)
    by_kid: dict = defaultdict(list)
    for e in ev:
        by_kid[e.kid].append(e.kind)
    flip = sum(1 for ks in by_kid.values() if sum(1 for k in ks if k != "created") >= 2 and len(ks) >= 3)
    flip_share = flip / max(len(by_kid), 1)
    churn = len(ev) / max(len(live), 1)
    level = Level.WATCH if (churn >= cfg.churn_warn * 4 or flip_share >= cfg.flipflop_warn) else Level.OK
    if flip_share >= 2 * cfg.flipflop_warn:
        level = Level.ALARM
    return Metric("knowledge_churn", churn, len(ev), None, None, level, f"{flip_share:.0%} of items flip-flopped; {len(live)} standing")


def is_overfit(o: Outcome, gap: float) -> bool | None:
    """Holdout collapsed against train. None when either score is missing (never read as 'not overfit')."""
    if o.train_score is None or o.holdout_score is None:
        return None
    if o.train_score <= 0:
        return False
    return o.holdout_score < (1.0 - gap) * o.train_score


def metric_overfit(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    flags = [f for f in (is_overfit(o, cfg.overfit_gap) for o in rows) if f is not None]
    return rate_metric("overfitting_rate", sum(flags), len(flags), cfg, cfg.overfit_warn, cfg.overfit_alarm, "high",
                       min_n=max(5, cfg.min_n // 3))


def metric_memorisation(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    tested = [o for o in rows if o.memorised is not None]
    return rate_metric("memorisation_rate", sum(1 for o in tested if o.memorised), len(tested), cfg, cfg.memorisation_warn,
                       cfg.memorisation_alarm, "high", min_n=max(5, cfg.min_n // 3))


def metric_concentration(rows: Sequence[Outcome], cfg: HealthConfig) -> Metric:
    """Where the compute went by hypothesis family: top-family share, HHI and effective number of families."""
    if len(rows) < cfg.min_n:
        return _unknown("research_concentration", len(rows), cfg.min_n)
    w = _share_weights(rows, lambda o: o.family or "(none)")
    tot = sum(w.values())
    top_name, top = max(w.items(), key=lambda kv: kv[1])
    share = top / tot
    level = Level.ALARM if share >= cfg.top_family_alarm else Level.WATCH if share >= cfg.top_family_warn else Level.OK
    return Metric("research_concentration", share, len(rows), None, None, level,
                  f"top family {top_name!r}; HHI {hhi(w.values()):.2f}; {effective_number(w.values()):.1f} effective families")


# ------------------------------------------------------------------------------------------------ findings (the named pathologies)

@dataclass(frozen=True)
class Finding:
    """A diagnosed research pathology. `evidence` holds the numbers that justify it so a reader can re-derive the verdict."""
    kind: str
    level: Level
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "level": self.level.value, "message": self.message, "evidence": _plain(self.evidence)}


def _plain(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, _StrEnum):
        return x.value
    return x


def to_realised_gains(rows: Sequence[Outcome]) -> list:
    """Adapter into the existing policy layer, so the exploration/exploitation audit is research_policy's own, not a copy."""
    return [RP.RealisedGain(o.exp_id, AREA_TO_TARGET[Area.parse(o.area)], o.family, 0.0, o.gain_bits, o.success,
                            max(o.cost_minutes, 0.0), str(o.when)) for o in rows]


def detect_family_monopoly(rows: Sequence[Outcome], cfg: HealthConfig) -> Finding | None:
    """Section 37's named tripwire: one tiny parameter family swallowing (by default) 90% of compute while producing negligible
    gain. A dominant family that is still productive is a WATCH, not an alarm - focus can be right; stagnation is the fault."""
    recent = list(rows[-cfg.window:])
    if len(recent) < cfg.min_n:
        return None
    w = _share_weights(recent, lambda o: o.family or "(none)")
    top, mins = max(w.items(), key=lambda kv: kv[1])
    share = mins / sum(w.values())
    if share < cfg.monopoly_soft_share:
        return None
    fam = [o for o in recent if (o.family or "(none)") == top]
    rest = [o for o in recent if (o.family or "(none)") != top]
    bpm, rest_bpm = bits_per_minute(fam), bits_per_minute(rest)
    mean_cost = float(np.mean([o.cost_minutes for o in fam])) if fam else 0.0
    stop = RP.marginal_return_verdict([o.gain_bits for o in fam], k=4, eps=cfg.negligible_bits_per_min * max(mean_cost, EPS))
    negligible = (bpm is not None and bpm < cfg.negligible_bits_per_min) or stop["verdict"] == "STOP"
    beaten = bpm is not None and rest_bpm is not None and rest_bpm > 3 * max(bpm, EPS) and len(rest) >= 5
    ev = {"family": top, "share": share, "bits_per_minute": bpm, "rest_bits_per_minute": rest_bpm, "marginal": stop,
          "n_family": len(fam), "n_rest": len(rest)}
    if share >= cfg.monopoly_share and (negligible or beaten):
        return Finding("FAMILY_MONOPOLY", Level.ALARM, f"{share:.0%} of compute is on family {top!r} and it is not paying "
                       f"({bpm if bpm is not None else 0:.4f} bits/min): redirect", ev)
    if share >= cfg.monopoly_share or (negligible and share >= cfg.monopoly_soft_share):
        why = "still productive" if not negligible else "and yielding little"
        return Finding("FAMILY_MONOPOLY", Level.WATCH, f"family {top!r} holds {share:.0%} of compute ({why})", ev)
    return None


@dataclass(frozen=True)
class QuestionTrack:
    """What the ledger knows about one question: how hard it was rated, how often it was tried, when it was last touched."""
    question_id: str
    difficulty: float
    attempts: int
    last_touched: str
    resolved: bool                                  # some attempt succeeded
    last_success: bool = False                      # the most recent attempt succeeded (the question was not left failing)


def question_tracks(rows: Sequence[Outcome]) -> list:
    by_q: dict = defaultdict(list)
    for o in rows:
        if o.question_id:
            by_q[o.question_id].append(o)
    out = []
    for qid, os_ in by_q.items():
        last = max(os_, key=lambda o: (as_date(o.when), o.exp_id))
        out.append(QuestionTrack(qid, float(np.mean([o.difficulty for o in os_])), len(os_), last.when,
                                 any(o.success for o in os_), last.success))
    return sorted(out, key=lambda t: t.question_id)


def _days(a, b) -> int:
    return (as_date(a) - as_date(b)).days


def detect_hard_abandonment(rows: Sequence[Outcome], now, cfg: HealthConfig) -> Finding | None:
    """Two independent signs that the researcher is dodging difficulty: (A) the share of attempts spent on hard questions fell
    sharply against the previous window; (B) hard questions are left unresolved and untouched far more often than easy ones."""
    recent, prior = recent_and_prior(rows, cfg.window, prior_windows=4)
    if len(recent) < cfg.min_n or len(prior) < cfg.min_n:
        return None
    hard_r = sum(1 for o in recent if o.difficulty >= cfg.hard_cut)
    hard_p = sum(1 for o in prior if o.difficulty >= cfg.hard_cut)
    share_r, share_p = hard_r / len(recent), hard_p / len(prior)
    z = two_proportion_z(hard_r, len(recent), hard_p, len(prior))
    drop = share_p >= 0.15 and share_r < share_p * cfg.hard_share_drop and (z is not None and z < -1.64)
    floor_hit = drop and share_r < cfg.hard_share_floor
    tracks = question_tracks(rows)
    hard_q = [t for t in tracks if t.difficulty >= cfg.hard_cut]
    easy_q = [t for t in tracks if t.difficulty < cfg.hard_cut]

    def stale(t: QuestionTrack) -> bool:
        return (not t.last_success) and _days(now, t.last_touched) > cfg.stale_days

    hard_stale = [t for t in hard_q if stale(t)]
    hr = len(hard_stale) / len(hard_q) if len(hard_q) >= cfg.min_hard_attempts else None
    er = (sum(1 for t in easy_q if stale(t)) / len(easy_q)) if len(easy_q) >= cfg.min_hard_attempts else None
    quit_early = [t for t in hard_stale if t.attempts < cfg.min_hard_attempts]
    unequal = hr is not None and hr >= 0.5 and (er is None or hr >= 2 * er)
    if not (drop or unequal):
        return None
    reopen = tuple(t.question_id for t in sorted(hard_stale, key=lambda t: (-t.difficulty, t.attempts, t.question_id)))
    ev = {"hard_share_recent": share_r, "hard_share_prior": share_p, "z": z, "hard_stale_rate": hr, "easy_stale_rate": er,
          "reopen": list(reopen), "quit_early": [t.question_id for t in quit_early]}
    level = Level.ALARM if (floor_hit or (drop and unequal)) else Level.WATCH
    return Finding("HARD_ABANDONMENT", level, f"hard questions fell from {share_p:.0%} to {share_r:.0%} of attempts"
                   f"{'' if hr is None else f'; {hr:.0%} of hard questions are stale and unresolved'}", ev)


def area_profile(rows: Sequence[Outcome]) -> dict:
    """Per area: minutes, attempts, mean prior difficulty, success rate, bits per minute, mean bits per success."""
    by: dict = defaultdict(list)
    for o in rows:
        by[o.area].append(o)
    out = {}
    for a, os_ in by.items():
        succ = [o for o in os_ if o.success]
        out[a] = {"minutes": sum(o.cost_minutes for o in os_), "n": len(os_),
                  "difficulty": float(np.mean([o.difficulty for o in os_])), "success_rate": len(succ) / len(os_),
                  "bits_per_minute": bits_per_minute(os_) or 0.0,
                  "bits_per_success": (sum(o.gain_bits for o in succ) / len(succ)) if succ else 0.0}
    return out


def detect_easy_bias(rows: Sequence[Outcome], cfg: HealthConfig) -> Finding | None:
    """Section 37: 'if it starts only researching areas where success is easy, penalize the research policy'. Signs: the
    compute-weighted difficulty is well below the unweighted difficulty across areas, and success rate correlates with the
    share of compute an area gets. The penalties returned in the evidence are what the policy should multiply into those areas."""
    recent = list(rows[-cfg.window:])
    every = area_profile(rows)
    seen = area_profile(recent)
    prof = {a: p for a, p in seen.items() if p["n"] >= 3}
    if len(recent) < cfg.min_n or len(every) < 3:
        return None
    tot_all = sum(p["minutes"] for p in seen.values()) or 1.0
    weighted = sum(p["minutes"] * p["difficulty"] for p in seen.values()) / tot_all
    plain = float(np.mean([p["difficulty"] for p in every.values()]))   # the difficulty of what is on offer
    gap = plain - weighted
    tot = sum(p["minutes"] for p in prof.values()) or 1.0
    names = sorted(prof)
    corr = spearman([prof[a]["success_rate"] for a in names], [prof[a]["minutes"] / tot for a in names])
    gap_hit = gap >= cfg.easy_bias_margin
    corr_hit = corr is not None and corr >= cfg.easy_corr and len(prof) >= 4 and gap >= cfg.easy_bias_margin / 3
    if not (gap_hit or corr_hit):
        return None
    if not prof:
        prof = seen
        tot = tot_all
    med_s = float(np.median([p["success_rate"] for p in prof.values()]))
    med_d = float(np.median([p["difficulty"] for p in every.values()]))
    overall_bpm = bits_per_minute(recent) or 0.0
    penalties = {}
    for a, p in prof.items():
        if p["success_rate"] >= med_s and p["difficulty"] <= med_d and p["bits_per_minute"] <= overall_bpm:
            over = min(1.0, (p["minutes"] / tot) * len(prof) / 3.0)
            penalties[a] = max(0.3, 1.0 - cfg.penalty_strength * over)
    ev = {"weighted_difficulty": weighted, "plain_difficulty": plain, "gap": gap, "success_share_corr": corr,
          "penalties": penalties}
    level = Level.ALARM if (gap_hit and corr_hit) else Level.WATCH
    return Finding("EASY_BIAS", level, f"compute-weighted difficulty {weighted:.2f} vs {plain:.2f} across areas"
                   f"{'' if corr is None else f'; success/share correlation {corr:.2f}'}", ev)


def detect_wrong_metric(rows: Sequence[Outcome], now, cfg: HealthConfig) -> Finding | None:
    """Section 43: more experiments are not better. Flag a month with markedly more experiments than the recent baseline
    (mean of the three months before it) but no more verified knowledge and no more information."""
    def bucket(lo: int, hi: int) -> list:
        return [o for o in rows if lo < _days(now, o.when) <= hi]

    b0 = bucket(0, cfg.bucket_days)
    base = [bucket(k * cfg.bucket_days, (k + 1) * cfg.bucket_days) for k in (1, 2, 3)]
    base = [b for b in base if len(b) > 0]
    if len(base) < 2:
        return None
    n1 = float(np.mean([len(b) for b in base]))
    v1 = float(np.mean([sum(1 for o in b if o.verified) for b in base]))
    g1 = float(np.mean([sum(o.gain_bits for o in b) for b in base]))
    v0, g0 = sum(1 for o in b0 if o.verified), sum(o.gain_bits for o in b0)
    if n1 < max(4, cfg.min_n // 2) or len(b0) < 1.5 * n1 or v0 > v1 or g0 > 0.5 * g1 * len(b0) / n1:
        return None
    level = Level.ALARM if (len(b0) >= 2 * n1 and v1 > 0 and v0 <= 0.5 * v1) else Level.WATCH
    return Finding("VOLUME_NOT_KNOWLEDGE", level, f"experiments {n1:.0f} -> {len(b0)} per month but verified discoveries "
                   f"{v1:.1f} -> {v0} and information {g1:.2f} -> {g0:.2f} bits",
                   {"n": [n1, len(b0)], "verified": [v1, v0], "bits": [g1, g0]})


def detect_exploration_deficit(rows: Sequence[Outcome], now, cfg: HealthConfig) -> list:
    """Exploit/explore balance, judged by research_policy's own audit and entropy code: known-promising work taking more than
    its share, exploration below the decaying epsilon schedule, an allocation collapsed onto a few areas, and areas never run."""
    if len(rows) < cfg.min_n:
        return []
    hist = to_realised_gains(rows)
    audit = RP.audit_spend(hist, cfg.window, 0.30)
    ent = RP.allocation_entropy(hist, cfg.window)
    eps = RP.epsilon_schedule(len(rows))
    explore = 1.0 - audit["exploit_share_minutes"]
    out = []
    if explore < eps * cfg.explore_min_factor or audit["violation"]:
        lvl = Level.ALARM if (audit["violation"] and explore < eps * cfg.explore_min_factor) else Level.WATCH
        out.append(Finding("EXPLOIT_HEAVY", lvl, f"exploration is {explore:.0%} of recent compute against a {eps:.0%} schedule",
                           {"explore_share": explore, "epsilon": eps, "audit": audit}))
    if ent["collapsed"]:
        out.append(Finding("DIVERSITY_COLLAPSE", Level.WATCH, f"allocation entropy {ent['entropy_bits']:.2f} of "
                           f"{ent['max_bits']:.2f} bits", {"entropy": ent}))
    starved = [s for s in RP.starving_targets(hist, now, days=60.0) if not s["never_run"]]
    never = [t.value for t in RP.TARGETS if t.value not in {h.target.value for h in hist}]
    if starved or never:
        rev = {t.value: a for a, t in AREA_TO_TARGET.items()}
        names = sorted({rev[s["target"]].value for s in starved} | {rev[t].value for t in never})
        out.append(Finding("STARVED_AREAS", Level.WATCH, f"no recent work in: {', '.join(names)}", {"areas": names}))
    return out


# ------------------------------------------------------------------------------------------------ directives

@dataclass(frozen=True)
class Directives:
    """What the health system asks of the allocator. Advice only: engine.research.diversity decides how to honour it."""
    area_multiplier: Mapping[str, float] = field(default_factory=dict)
    family_caps: Mapping[str, float] = field(default_factory=dict)
    reopen_questions: tuple = ()
    min_explore_share: float = 0.0
    starved_areas: tuple = ()
    reasons: tuple = ()

    def validate(self) -> list:
        errs = []
        for a, m in self.area_multiplier.items():
            if m <= 0 or not math.isfinite(m):
                errs.append(f"multiplier for {a} must be positive")
            try:
                Area.parse(a)
            except ValueError:
                errs.append(f"unknown area {a}")
        for f, c in self.family_caps.items():
            if not 0 < c <= 1:
                errs.append(f"family cap for {f} outside (0,1]")
        if not 0 <= self.min_explore_share <= 1:
            errs.append("min_explore_share outside [0,1]")
        return errs

    def is_neutral(self) -> bool:
        return not (self.area_multiplier or self.family_caps or self.reopen_questions or self.min_explore_share or self.starved_areas)

    def to_dict(self) -> dict:
        return {"area_multiplier": dict(self.area_multiplier), "family_caps": dict(self.family_caps),
                "reopen_questions": list(self.reopen_questions), "min_explore_share": self.min_explore_share,
                "starved_areas": list(self.starved_areas), "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Directives":
        return cls(dict(d.get("area_multiplier", {})), dict(d.get("family_caps", {})), tuple(d.get("reopen_questions", ())),
                   float(d.get("min_explore_share", 0.0)), tuple(d.get("starved_areas", ())), tuple(d.get("reasons", ())))


def _hard_areas(rows: Sequence[Outcome], cfg: HealthConfig) -> list:
    prof = area_profile(list(rows[-cfg.window:]))
    return sorted((a for a, p in prof.items() if p["difficulty"] >= cfg.hard_cut - 0.1), key=lambda a: -prof[a]["difficulty"])


def derive_directives(findings: Sequence[Finding], rows: Sequence[Outcome], cfg: HealthConfig) -> Directives:
    mult: dict = defaultdict(lambda: 1.0)
    caps, reopen, reasons, starved = {}, [], [], []
    floor = 0.0
    for f in findings:
        if f.level not in (Level.WATCH, Level.ALARM):
            continue
        strong = f.level is Level.ALARM
        if f.kind == "FAMILY_MONOPOLY":
            caps[f.evidence["family"]] = cfg.family_cap if strong else min(cfg.monopoly_share, cfg.family_cap + 0.2)
            reasons.append(f"cap family {f.evidence['family']!r}")
        elif f.kind == "HARD_ABANDONMENT":
            reopen.extend(f.evidence.get("reopen", []))
            for a in _hard_areas(rows, cfg):
                mult[a] *= 1.0 + cfg.penalty_strength * (1.0 if strong else 0.5)
            reasons.append("reopen hard questions and lift hard areas")
        elif f.kind == "EASY_BIAS":
            for a, fac in f.evidence.get("penalties", {}).items():
                mult[a] *= fac if strong else 1.0 - (1.0 - fac) / 2
            reasons.append("penalise easy areas")
        elif f.kind == "EXPLOIT_HEAVY":
            floor = max(floor, min(0.9, f.evidence["epsilon"] + (0.2 if strong else 0.1)))
            reasons.append("raise the exploration floor")
        elif f.kind == "STARVED_AREAS":
            starved.extend(f.evidence["areas"])
            reasons.append("feed starved areas")
        elif f.kind == "DIVERSITY_COLLAPSE":
            floor = max(floor, 0.4)
            reasons.append("diversity collapsed: widen")
    return Directives(dict(mult), caps, tuple(dict.fromkeys(reopen)), floor, tuple(dict.fromkeys(starved)), tuple(reasons))


# ------------------------------------------------------------------------------------------------ report

def overall_level(metrics: Sequence[Metric], findings: Sequence[Finding]) -> Level:
    """ALARM/WATCH from anything; UNKNOWN when fewer than half of the metrics could be computed (a brain that cannot be
    measured is not healthy); otherwise OK."""
    levels = [m.level for m in metrics] + [f.level for f in findings]
    if Level.ALARM in levels:
        return Level.ALARM
    if Level.WATCH in levels:
        return Level.WATCH
    known = sum(1 for m in metrics if m.level is not Level.UNKNOWN)
    return Level.UNKNOWN if not metrics or known * 2 < len(metrics) else Level.OK


@dataclass(frozen=True)
class BrainHealthReport:
    now: str
    n_outcomes: int
    metrics: tuple
    findings: tuple
    level: Level
    directives: Directives
    code_hash: str
    config_hash: str
    label: str = LABEL

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def finding(self, kind: str) -> Finding | None:
        return next((f for f in self.findings if f.kind == kind), None)

    @property
    def kinds(self) -> tuple:
        return tuple(f.kind for f in self.findings)

    @property
    def report_id(self) -> str:
        return stable_hash({"now": str(self.now), "metrics": [m.to_dict() for m in self.metrics],
                            "findings": [f.to_dict() for f in self.findings], "config": self.config_hash}, 12)

    def to_dict(self) -> dict:
        return {"now": str(self.now), "n_outcomes": self.n_outcomes, "metrics": [m.to_dict() for m in self.metrics],
                "findings": [f.to_dict() for f in self.findings], "level": self.level.value,
                "directives": self.directives.to_dict(), "code_hash": self.code_hash, "config_hash": self.config_hash,
                "label": self.label, "report_id": self.report_id}

    @classmethod
    def from_dict(cls, d: Mapping) -> "BrainHealthReport":
        ms = tuple(Metric(m["name"], m["value"], m["n"], m["lo"], m["hi"], Level.parse(m["level"]), m.get("note", ""))
                   for m in d["metrics"])
        fs = tuple(Finding(f["kind"], Level.parse(f["level"]), f["message"], f.get("evidence", {})) for f in d["findings"])
        return cls(d["now"], d["n_outcomes"], ms, fs, Level.parse(d["level"]), Directives.from_dict(d["directives"]),
                   d.get("code_hash", ""), d.get("config_hash", ""), d.get("label", LABEL))

    def render(self) -> str:
        lines = [f"Research brain health as of {self.now}: {self.level.value} ({self.n_outcomes} outcomes)  [{self.label}]"]
        for m in self.metrics:
            v = "unknown" if m.value is None else f"{m.value:.3f}"
            ci = "" if m.lo is None else f" [{m.lo:.2f}, {m.hi:.2f}]"
            lines.append(f"  {m.level.value:7s} {m.name:24s} {v}{ci} n={m.n}  {m.note}".rstrip())
        for f in self.findings:
            lines.append(f"  ! {f.level.value:5s} {f.kind}: {f.message}")
        if self.directives.reasons:
            lines.append("  directives: " + "; ".join(self.directives.reasons))
        return "\n".join(lines)


def config_hash(cfg: HealthConfig) -> str:
    return stable_hash(dataclasses.asdict(cfg), 12)


def validate_inputs(outcomes: Sequence[Outcome], events: Sequence[KnowledgeEvent]) -> list:
    errs: list[str] = []
    seen: set[str] = set()
    for o in outcomes:
        errs.extend(f"{o.exp_id}: {e}" for e in o.validate())
        if o.exp_id in seen:
            errs.append(f"{o.exp_id}: duplicate experiment id")
        seen.add(o.exp_id)
    for e in events:
        errs.extend(f"{e.kid}: {x}" for x in e.validate())
    return errs


def compute_metrics(rows: Sequence[Outcome], events: Sequence[KnowledgeEvent], now, cfg: HealthConfig) -> tuple:
    """Behavioural metrics use the last `window` outcomes; claim-based metrics (FDR, replication, transfer) need the long view
    because verdicts on a claim arrive late, so they use the last 4 windows."""
    recent = list(rows[-cfg.window:])
    claims = list(rows[-4 * cfg.window:])
    return (metric_diversity(recent, cfg), metric_duplicate_rate(recent, cfg), metric_success_rate(recent, cfg),
            metric_fdr(claims, cfg), metric_replication(claims, cfg), metric_transfer(claims, cfg),
            metric_efficiency(rows, cfg), metric_churn(events, now, cfg), metric_overfit(recent, cfg),
            metric_memorisation(recent, cfg), metric_concentration(recent, cfg))


def run_detectors(rows: Sequence[Outcome], now, cfg: HealthConfig) -> tuple:
    found = [detect_family_monopoly(rows, cfg), detect_hard_abandonment(rows, now, cfg), detect_easy_bias(rows, cfg),
             detect_wrong_metric(rows, now, cfg), detect_unjudged_claims(rows, now, cfg)]
    found.extend(detect_exploration_deficit(rows, now, cfg))
    return tuple(f for f in found if f is not None)


def step(outcomes: Iterable[Outcome], now, cfg: HealthConfig | None = None, events: Sequence[KnowledgeEvent] = (),
         ledger: "HealthLedger | None" = None) -> BrainHealthReport:
    """Public entry (wave-2 research loop): measure the researcher as of `now`. Invalid configuration or malformed records
    raise ValueError; an outcome dated at/after `now` raises FirewallBreach; too little data yields UNKNOWN metrics, never OK.
    When a ledger is given the report is appended (ledger enforces strictly increasing `now`)."""
    cfg = cfg or HealthConfig()
    errs = cfg.validate()
    outcomes = list(outcomes)
    events = list(events)
    errs += validate_inputs(outcomes, events)
    if errs:
        raise ValueError("invalid brain-health input: " + "; ".join(errs[:8]))
    rows = visible(outcomes, now)
    metrics = compute_metrics(rows, events, now, cfg)
    findings = apply_difficulty_guard(run_detectors(rows, now, cfg), difficulty_validity(rows[-4 * cfg.window:]))
    for m in metrics:
        if m.name == "false_discovery_rate" and m.level in (Level.ALARM, Level.WATCH) and m.value is not None:
            findings = findings + (Finding("FALSE_DISCOVERIES", m.level, f"{m.value:.0%} of judged claims were not real",
                                           {"rate": m.value, "n": m.n}),)
    rep = BrainHealthReport(str(now), len(rows), metrics, findings, overall_level(metrics, findings),
                            derive_directives(findings, rows, cfg), code_hash(), config_hash(cfg))
    if ledger is not None:
        ledger.append(rep, cfg)
    return rep


# ------------------------------------------------------------------------------------------------ ledger over time

class HealthLedger:
    """Reports over time. History is append-only; `now` must strictly increase (a replayed or out-of-order report would let the
    past be rewritten). Persistence is a JSON list of report dicts. Alarms are debounced: a finding is PERSISTENT only after
    `min_hold` consecutive reports, so one noisy window does not redirect the whole research programme."""

    def __init__(self, reports: Iterable[BrainHealthReport] = ()):
        self._reports: list = []
        for r in reports:
            self.append(r)

    def __len__(self) -> int:
        return len(self._reports)

    @property
    def reports(self) -> tuple:
        return tuple(self._reports)

    def latest(self) -> BrainHealthReport | None:
        return self._reports[-1] if self._reports else None

    def append(self, rep: BrainHealthReport, cfg: HealthConfig | None = None) -> None:
        if self._reports and as_date(rep.now) <= as_date(self._reports[-1].now):
            raise FirewallBreach(f"health report for {rep.now} is not after the last one ({self._reports[-1].now})")
        self._reports.append(rep)

    def series(self, name: str) -> list:
        return [(r.now, m.value) for r in self._reports for m in [r.metric(name)] if m is not None and m.value is not None]

    def streak(self, kind: str, at_least: Level = Level.WATCH) -> int:
        """Consecutive most-recent reports containing a finding of `kind` at or above `at_least`."""
        n = 0
        for r in reversed(self._reports):
            f = r.finding(kind)
            if f is not None and _LEVEL_RANK[f.level] >= _LEVEL_RANK[at_least]:
                n += 1
            else:
                break
        return n

    def persistent(self, min_hold: int = 2) -> list:
        kinds = {k for r in self._reports[-min_hold:] for k in r.kinds}
        return sorted(k for k in kinds if self.streak(k) >= min_hold)

    def resolved(self) -> list:
        """Finding kinds present in the previous report but gone in the latest: the redirects that worked."""
        if len(self._reports) < 2:
            return []
        return sorted(set(self._reports[-2].kinds) - set(self._reports[-1].kinds))

    def flap_rate(self, kind: str) -> float:
        """Share of consecutive report pairs in which `kind` switched on or off. High = the detector or the fix is unstable."""
        if len(self._reports) < 3:
            return 0.0
        on = [kind in r.kinds for r in self._reports]
        return sum(1 for a, b in zip(on, on[1:]) if a != b) / (len(on) - 1)

    def trend(self, name: str, last: int = 6) -> dict:
        s = self.series(name)[-last:]
        if len(s) < 3:
            return {"n": len(s), "slope": None, "verdict": "INSUFFICIENT"}
        y = np.array([v for _, v in s], float)
        slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])
        scale = max(abs(float(y.mean())), 1e-6)
        rel = slope / scale
        return {"n": len(s), "slope": slope, "relative": rel, "verdict": "RISING" if rel > 0.05 else "FALLING" if rel < -0.05 else "FLAT"}

    def shift(self, name: str) -> dict:
        """CUSUM (research_policy's) over a metric series: has the researcher's behaviour changed regime?"""
        return RP.cusum_shift([v for _, v in self.series(name)])

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self._reports], sort_keys=True)

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path) -> "HealthLedger":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(BrainHealthReport.from_dict(d) for d in json.loads(p.read_text(encoding="utf-8")))


def explain_change(prev: BrainHealthReport | None, cur: BrainHealthReport, tol: float = 0.10) -> list:
    """Plain-language differences between two reports: level changes, findings that appeared or cleared, metrics that moved."""
    out = []
    if prev is None:
        return [f"first report: {cur.level.value}"]
    if prev.level is not cur.level:
        out.append(f"overall {prev.level.value} -> {cur.level.value}")
    for k in sorted(set(cur.kinds) - set(prev.kinds)):
        out.append(f"new finding {k}")
    for k in sorted(set(prev.kinds) - set(cur.kinds)):
        out.append(f"cleared finding {k}")
    for m in cur.metrics:
        p = prev.metric(m.name)
        if p is None or p.value is None or m.value is None:
            continue
        base = max(abs(p.value), 1e-6)
        if abs(m.value - p.value) / base > tol:
            out.append(f"{m.name} {p.value:.3f} -> {m.value:.3f}")
    return out


def worst_metrics(rep: BrainHealthReport, top: int = 3) -> list:
    """The metrics most in need of attention, worst level first, then narrowest evidence."""
    ranked = sorted((m for m in rep.metrics if m.level in (Level.ALARM, Level.WATCH)),
                    key=lambda m: (-_LEVEL_RANK[m.level], -m.n))
    return [m.name for m in ranked[:top]]


def researcher_scorecard(rows: Sequence[Outcome]) -> dict:
    """Section 43 in numbers: what the research produced versus how busy it was. Verified discoveries per 1,000 CPU-minutes is
    the headline; raw counts of experiments and claims are reported only so their ratio to it is visible."""
    mins = sum(o.cost_minutes for o in rows)
    verified = sum(1 for o in rows if o.verified)
    claims = sum(1 for o in rows if o.claimed_discovery)
    return {"experiments": len(rows), "claims": claims, "verified": verified, "minutes": mins,
            "verified_per_1000_min": (1000.0 * verified / mins) if mins > 0 else None,
            "claim_survival": (verified / claims) if claims else None}


def family_report(rows: Sequence[Outcome], min_n: int = 3) -> dict:
    """Success rate per hypothesis family with empirical-Bayes shrinkage (meta_learning.group_rates), so a family that is 2/2 is
    not reported as better than one that is 40/60."""
    rates, prior = group_rates(((o.family or "(none)", o.success) for o in rows), min_n=min_n)
    return {"prior": prior, "families": {k: {"n": v.n, "raw": v.raw, "shrunk": v.shrunk, "lo": v.lo, "hi": v.hi}
                                          for k, v in sorted(rates.items())}}


# ------------------------------------------------------------------------------------------------ guards on the health system itself

def difficulty_validity(rows: Sequence[Outcome], min_n: int = 40) -> dict:
    """Do the pre-run difficulty ratings actually predict failure? AUC of difficulty against 'did not succeed'. The hard-question
    and easy-area detectors are only as good as this rating; if it is uninformative (AUC near 0.5) they must not raise ALARMs."""
    if len(rows) < min_n:
        return {"verdict": "INSUFFICIENT", "n": len(rows), "auc": None}
    a = auc([o.difficulty for o in rows], [0 if o.success else 1 for o in rows])
    if a is None:
        return {"verdict": "INSUFFICIENT", "n": len(rows), "auc": None}
    return {"verdict": "INFORMATIVE" if a >= 0.55 else "UNINFORMATIVE", "n": len(rows), "auc": a}


def apply_difficulty_guard(findings: Sequence[Finding], validity: Mapping) -> tuple:
    """Downgrade difficulty-based ALARMs to WATCH when the difficulty rating is uninformative, recording why in the evidence,
    so a broken rating cannot redirect the whole research programme."""
    if validity.get("verdict") != "UNINFORMATIVE":
        return tuple(findings)
    out = []
    for f in findings:
        if f.kind in ("HARD_ABANDONMENT", "EASY_BIAS") and f.level is Level.ALARM:
            f = Finding(f.kind, Level.WATCH, f.message + " (downgraded: difficulty ratings are uninformative)",
                        {**dict(f.evidence), "difficulty_auc": validity.get("auc"), "downgraded": True})
        out.append(f)
    return tuple(out)


def stagnant_families(rows: Sequence[Outcome], cfg: HealthConfig, k: int = 4) -> list:
    """Every family whose last k gains are negligible and not trending up (research_policy.marginal_return_verdict), with the
    compute it has consumed. The 'stop wasting compute' input: families to retire or starve, worst waste first."""
    by: dict = defaultdict(list)
    for o in rows:
        by[o.family or "(none)"].append(o)
    out = []
    for fam, os_ in by.items():
        mean_cost = float(np.mean([o.cost_minutes for o in os_])) if os_ else 0.0
        v = RP.marginal_return_verdict([o.gain_bits for o in os_], k=k, eps=cfg.negligible_bits_per_min * max(mean_cost, EPS))
        if v["verdict"] == "STOP":
            out.append({"family": fam, "n": len(os_), "minutes": sum(o.cost_minutes for o in os_), "tail_mean": v["tail_mean"]})
    return sorted(out, key=lambda r: -r["minutes"])


def integrity_gate(report: BrainHealthReport) -> tuple:
    """May this researcher's new claims be trusted enough to promote? QUARANTINED when the false-discovery, memorisation or
    overfitting metric is in ALARM (the brain is producing claims that do not hold); NEEDS_MORE_EVIDENCE when those metrics
    are unknown; PROMOTE only when they are all measured and not in ALARM. Returns (GateVerdict, reasons)."""
    from engine.research.core import GateVerdict
    names = ("false_discovery_rate", "memorisation_rate", "overfitting_rate", "replication_rate")
    ms = [report.metric(n) for n in names]
    alarms = [m.name for m in ms if m is not None and m.level is Level.ALARM]
    unknown = [n for n, m in zip(names, ms) if m is None or m.level is Level.UNKNOWN]
    if alarms:
        return GateVerdict.QUARANTINED, [f"{a} in ALARM" for a in alarms]
    if unknown:
        return GateVerdict.NEEDS_MORE_EVIDENCE, [f"{u} not yet measurable" for u in unknown]
    return GateVerdict.PROMOTE, []


def write_report(rep: BrainHealthReport, out_dir, rows: Sequence[Outcome] = (), cfg: HealthConfig | None = None) -> Path:
    """Persist a report as JSON plus rendered text; the JSON carries the scorecard, per-area and per-family breakdowns and the
    stagnant families so a reader can drill from an alarm to the rows that caused it."""
    cfg = cfg or HealthConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"brain_health_{rep.now}"
    body = rep.to_dict()
    if rows:
        body["scorecard"] = researcher_scorecard(rows)
        body["by_area"] = breakdown(rows, lambda o: o.area)
        body["by_family"] = breakdown(rows, lambda o: o.family or "(none)")
        body["eras"] = era_breakdown(rows)
        body["stagnant_families"] = stagnant_families(rows, cfg)
        body["difficulty_validity"] = difficulty_validity(rows)
    (out / f"{stem}.txt").write_text(rep.render(), encoding="utf-8")
    path = out / f"{stem}.json"
    path.write_text(json.dumps(body, sort_keys=True, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------ unjudged claims, degradation, definitions

def unjudged_claims(rows: Sequence[Outcome], now, max_age_days: int = 60) -> dict:
    """Claims old enough to have been judged but still carrying no replication verdict. The false-discovery metric only sees
    JUDGED claims, so a growing pile of unjudged ones is survivor bias: the researcher may be shelving claims that would have
    failed. Returns counts and the oldest waiting claim."""
    claims = [o for o in rows if o.claimed_discovery]
    old = [o for o in claims if _days(now, o.when) > max_age_days]
    waiting = [o for o in old if o.replicated is None and o.false_discovery is None]
    return {"claims": len(claims), "old_claims": len(old), "unjudged": len(waiting),
            "share": (len(waiting) / len(old)) if old else None,
            "oldest_days": max((_days(now, o.when) for o in waiting), default=None)}


def detect_unjudged_claims(rows: Sequence[Outcome], now, cfg: HealthConfig) -> Finding | None:
    u = unjudged_claims(rows, now)
    if u["old_claims"] < max(5, cfg.min_n // 3) or u["share"] is None or u["share"] < 0.5:
        return None
    return Finding("CLAIMS_UNJUDGED", Level.WATCH, f"{u['unjudged']} of {u['old_claims']} claims older than 60 days were never "
                   f"replicated or refuted: the false-discovery rate is measured on survivors", u)


def degradation_report(rows: Sequence[Outcome], cfg: HealthConfig) -> list:
    """Which headline rates got significantly WORSE in the recent window against the four windows before it (two-proportion z,
    one-sided at ~5%)? Complements the absolute thresholds: a metric can be under its alarm line and still be sliding."""
    recent, prior = recent_and_prior(rows, cfg.window, prior_windows=4)
    if len(recent) < cfg.min_n or len(prior) < cfg.min_n:
        return []
    out = []
    tests = (("success_rate", lambda o: o.success, -1), ("duplicate_rate", lambda o: bool(o.duplicate_of), +1),
             ("memorisation_rate", lambda o: bool(o.memorised), +1))
    for name, pred, bad in tests:
        k1, k2 = sum(1 for o in recent if pred(o)), sum(1 for o in prior if pred(o))
        z = two_proportion_z(k1, len(recent), k2, len(prior))
        if z is not None and bad * z > 1.645:
            out.append({"metric": name, "recent": k1 / len(recent), "baseline": k2 / len(prior), "z": z})
    r, p = bits_per_minute(recent), bits_per_minute(prior)
    if r is not None and p is not None and p > EPS and r < 0.5 * p:
        out.append({"metric": "compute_efficiency", "recent": r, "baseline": p, "z": None})
    return out


def expected_false_watches(n_metrics: int, n_reports: int, per_check_rate: float = 0.05) -> float:
    """How many spurious WATCH levels to expect from chance alone across a run of reports: a reminder that a long-lived
    monitor will cry wolf, which is why persistence (min_hold) rather than a single report drives redirects."""
    return float(n_metrics * n_reports * per_check_rate)


def metric_definitions() -> dict:
    """The eleven section-37 quantities in plain terms: what is counted, which direction is bad, and where its evidence comes
    from. Kept as data so reports, docs and tests read one source."""
    return {
        "diversity": ("normalised entropy of compute over the ten areas", "low is bad", "recent window"),
        "duplicate_rate": ("share of experiments repeating an earlier one, tagged or implied by identical config", "high is bad", "recent window"),
        "success_rate": ("share of experiments with a verified useful result", "too low or too high is bad", "recent window"),
        "false_discovery_rate": ("share of judged claims later shown not real (+ BH cross-check)", "high is bad", "long window, judged claims"),
        "replication_rate": ("share of tested claims that replicated", "low is bad", "long window, tested claims"),
        "transfer_rate": ("share of replicated claims that transferred to unseen data", "low is bad", "long window, replicated claims"),
        "compute_efficiency": ("bits per CPU-minute vs the previous window; share of minutes spent on repeats or misleading work", "falling is bad", "recent and prior window"),
        "knowledge_churn": ("knowledge events per standing item and the flip-flop share", "high is bad", "knowledge events"),
        "overfitting_rate": ("share of scored experiments whose holdout collapsed against train", "high is bad", "recent window"),
        "memorisation_rate": ("share of memorisation-controlled experiments that were memorised", "high is bad", "recent window"),
        "research_concentration": ("top hypothesis family's share of compute (HHI in the note)", "high is bad", "recent window"),
    }


# ------------------------------------------------------------------------------------------------ per-area health and walk-forward series

def area_health(rows: Sequence[Outcome], cfg: HealthConfig) -> dict:
    """Health of each section-38 area on its own: success interval, whether the area has produced anything at all, whether its
    success series has shifted (CUSUM from research_policy), and how long since it last paid. A brain can look healthy overall
    while one area has silently died; this is where that shows."""
    by: dict = defaultdict(list)
    for o in rows:
        by[o.area].append(o)
    out = {}
    for a in AREAS:
        os_ = by.get(a.value, [])
        n, k = len(os_), sum(1 for o in os_ if o.success)
        lo, hi = wilson(k, n)
        shift = RP.cusum_shift([1.0 if o.success else 0.0 for o in os_])
        last_ok = next((o.when for o in reversed(os_) if o.success), None)
        if n < cfg.min_n // 2:
            level, why = Level.UNKNOWN, f"{n} trials"
        elif hi < cfg.success_floor * 5:
            level, why = Level.ALARM, "no useful result in this area"
        elif shift["verdict"] == "SHIFT" and shift["direction"] == "down":
            level, why = Level.WATCH, "success rate shifted down"
        else:
            level, why = Level.OK, ""
        out[a.value] = {"n": n, "successes": k, "lo": lo, "hi": hi, "level": level, "why": why, "last_success": last_ok,
                        "shift": shift["verdict"]}
    return out


def rolling_metric(rows: Sequence[Outcome], name: str, cfg: HealthConfig, checkpoints: Sequence, events: Sequence[KnowledgeEvent] = ()) -> list:
    """One metric evaluated walk-forward at each checkpoint on what was visible then: the series a trend or drift check needs.
    Unknown values are kept as None so a gap is visible rather than interpolated."""
    out = []
    for cp in sorted(checkpoints, key=as_date):
        seen = [o for o in rows if as_date(o.when) < as_date(cp)]
        ev = [e for e in events if as_date(e.when) < as_date(cp)]
        m = next((m for m in compute_metrics(visible(seen, cp), ev, cp, cfg) if m.name == name), None)
        if m is None:
            raise KeyError(f"unknown metric {name!r}")
        out.append((str(cp), m.value, m.level))
    return out


def compare_reports(a: BrainHealthReport, b: BrainHealthReport) -> dict:
    """Which metrics improved, worsened or stayed within a level between two reports (a = earlier). 'Better' is by level rank,
    never by raw value, because for some metrics higher is good and for others bad."""
    better: list[str] = []
    worse: list[str] = []
    same: list[str] = []
    for m in b.metrics:
        p = a.metric(m.name)
        if p is None or Level.UNKNOWN in (p.level, m.level):
            continue
        d = _LEVEL_RANK[m.level] - _LEVEL_RANK[p.level]
        (worse if d > 0 else better if d < 0 else same).append(m.name)
    return {"better": sorted(better), "worse": sorted(worse), "same": sorted(same),
            "new_findings": sorted(set(b.kinds) - set(a.kinds)), "cleared_findings": sorted(set(a.kinds) - set(b.kinds))}


def family_lifecycle(rows: Sequence[Outcome], cfg: HealthConfig) -> dict:
    """Where each hypothesis family is in its life: EMERGING (few trials), PRODUCTIVE (recent gain per minute at least half the
    family's own best window), STAGNANT (marginal return has stopped, research_policy verdict) or EXHAUSTED (stagnant AND a
    large sunk cost with nothing verified). The waste manager retires EXHAUSTED families; STAGNANT ones are starved first."""
    by: dict = defaultdict(list)
    for o in rows:
        by[o.family or "(none)"].append(o)
    out = {}
    for fam, os_ in by.items():
        minutes = sum(o.cost_minutes for o in os_)
        if len(os_) < cfg.min_hard_attempts + 2:
            out[fam] = {"state": "EMERGING", "n": len(os_), "minutes": minutes}
            continue
        mean_cost = float(np.mean([o.cost_minutes for o in os_]))
        verdict = RP.marginal_return_verdict([o.gain_bits for o in os_], k=4, eps=cfg.negligible_bits_per_min * max(mean_cost, EPS))
        half = len(os_) // 2
        best = max(bits_per_minute(os_[i:i + half]) or 0.0 for i in range(0, len(os_) - half + 1, max(1, half // 2)))
        recent = bits_per_minute(os_[-half:]) or 0.0
        verified = sum(1 for o in os_ if o.verified)
        if verdict["verdict"] == "STOP":
            state = "EXHAUSTED" if (verified == 0 and minutes >= 10 * cfg.window * 0.5) else "STAGNANT"
        else:
            state = "PRODUCTIVE" if recent >= 0.5 * best else "STAGNANT"
        out[fam] = {"state": state, "n": len(os_), "minutes": minutes, "recent_bpm": recent, "best_bpm": best, "verified": verified}
    return out


def claim_calibration(rows: Sequence[Outcome], bins: int = 4) -> list:
    """Are claimed p-values honest? Per p-value bin, the share of JUDGED claims that proved false. A researcher whose 'p < 0.01'
    claims fail as often as its 'p < 0.05' ones is not producing calibrated evidence, whatever the nominal p-values say."""
    judged = [o for o in rows if o.claimed_discovery and o.p_value is not None and o.false_discovery is not None]
    if len(judged) < bins * 3:
        return []
    ps = np.array([o.p_value for o in judged])
    edges = np.quantile(ps, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        sel = [o for o in judged if edges[i] <= o.p_value <= edges[i + 1] and (i == bins - 1 or o.p_value < edges[i + 1])]
        if sel:
            out.append({"p_lo": float(edges[i]), "p_hi": float(edges[i + 1]), "n": len(sel),
                        "false_rate": sum(1 for o in sel if o.false_discovery) / len(sel)})
    return out


# ------------------------------------------------------------------------------------------------ breakdowns

def breakdown(rows: Sequence[Outcome], key, min_n: int = 5) -> dict:
    """The rate metrics per group (area, family, era ...): where exactly is the researcher failing? Groups below min_n report
    counts only. Uses Wilson intervals throughout so a 1-of-2 group is not mistaken for a finding."""
    groups: dict = defaultdict(list)
    for o in rows:
        groups[key(o)].append(o)
    out = {}
    for g, os_ in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row: dict = {"n": len(os_), "minutes": sum(o.cost_minutes for o in os_)}
        if len(os_) >= min_n:
            k = sum(1 for o in os_ if o.success)
            row["success_rate"] = k / len(os_)
            row["success_ci"] = wilson(k, len(os_))
            row["bits_per_minute"] = bits_per_minute(os_)
            flags = [f for f in (is_overfit(o, 0.5) for o in os_) if f is not None]
            row["overfit_rate"] = (sum(flags) / len(flags)) if flags else None
            row["duplicate_rate"] = sum(1 for o in os_ if o.duplicate_of) / len(os_)
        out[str(g)] = row
    return out


def era_breakdown(rows: Sequence[Outcome], n_eras: int = 3) -> list:
    """Split history into equal-count eras and report each era's headline rates, so health that only looks good on average
    but degraded over time shows up as a trend across eras."""
    if not rows or n_eras < 1:
        return []
    edges = np.linspace(0, len(rows), n_eras + 1).astype(int)
    out = []
    for i in range(n_eras):
        seg = list(rows[edges[i]:edges[i + 1]])
        if not seg:
            continue
        claims = [o for o in seg if o.claimed_discovery]
        out.append({"era": i, "start": seg[0].when, "end": seg[-1].when, "n": len(seg),
                    "success_rate": sum(1 for o in seg if o.success) / len(seg), "bits_per_minute": bits_per_minute(seg),
                    "verified": sum(1 for o in seg if o.verified),
                    "claim_survival": (sum(1 for o in claims if o.verified) / len(claims)) if claims else None})
    return out


def implied_duplicates(rows: Sequence[Outcome]) -> set:
    """Experiments that repeat an earlier (family, config_hash) although the researcher did not tag them as duplicates. The
    tagged rate alone can be gamed by simply not tagging."""
    seen: dict = {}
    out = set()
    for o in rows:
        if not o.config_hash:
            continue
        key = (o.family, o.config_hash)
        if key in seen and not o.duplicate_of:
            out.add(o.exp_id)
        seen.setdefault(key, o.exp_id)
    return out


def data_sufficiency(rows: Sequence[Outcome], events: Sequence[KnowledgeEvent], cfg: HealthConfig) -> dict:
    """For every metric, how many more matured records it needs before it can leave UNKNOWN. Keeps 'we cannot tell yet' honest
    and actionable instead of silently reading as fine."""
    recent = list(rows[-cfg.window:])
    claims = list(rows[-4 * cfg.window:])
    small = max(5, cfg.min_n // 3)
    have = {
        "diversity": (len(recent), cfg.min_n), "duplicate_rate": (len(recent), cfg.min_n),
        "success_rate": (len(recent), cfg.min_n), "compute_efficiency": (len(recent), cfg.min_n),
        "research_concentration": (len(recent), cfg.min_n),
        "false_discovery_rate": (sum(1 for o in claims if o.claimed_discovery and o.false_discovery is not None), small),
        "replication_rate": (sum(1 for o in claims if o.claimed_discovery and o.replicated is not None), small),
        "transfer_rate": (sum(1 for o in claims if o.claimed_discovery and o.replicated and o.transferred is not None), small),
        "overfitting_rate": (sum(1 for o in recent if o.train_score is not None and o.holdout_score is not None), small),
        "memorisation_rate": (sum(1 for o in recent if o.memorised is not None), small),
        "knowledge_churn": (len(events), cfg.min_n // 3),
    }
    return {k: {"have": h, "need": n, "missing": max(0, n - h)} for k, (h, n) in have.items()}


def redirect_value(rows: Sequence[Outcome], finding: Finding, cfg: HealthConfig) -> dict:
    """What a monopoly redirect is worth, in the currency of the ledger. If the capped family's compute moved to the rest of the
    portfolio at the rest's own recent bits/minute, how many bits would that have bought? A FORECAST of a plan, never a result."""
    if finding.kind != "FAMILY_MONOPOLY":
        return {"applicable": False}
    fam = finding.evidence["family"]
    recent = list(rows[-cfg.window:])
    fam_rows = [o for o in recent if (o.family or "(none)") == fam]
    rest = [o for o in recent if (o.family or "(none)") != fam]
    total = sum(o.cost_minutes for o in recent)
    excess = max(0.0, sum(o.cost_minutes for o in fam_rows) - cfg.family_cap * total)
    rb, fb = bits_per_minute(rest), bits_per_minute(fam_rows)
    if rb is None or fb is None:
        return {"applicable": True, "movable_minutes": excess, "expected_bits": None}
    return {"applicable": True, "movable_minutes": excess, "family_bits_per_minute": fb, "rest_bits_per_minute": rb,
            "expected_bits": excess * (rb - fb), "is_forecast": True}


def success_collapse_sprt(rows: Sequence[Outcome], p_healthy: float, p_broken: float) -> dict:
    """Sequential test (research_policy.sprt_bernoulli) that the success rate has fallen from p_healthy to p_broken. Reaches a
    verdict with fewer outcomes than a fixed-window comparison, and says INCONCLUSIVE rather than guessing."""
    if not 0 < p_broken < p_healthy < 1:
        raise ValueError("need 0 < p_broken < p_healthy < 1")
    # run the test on FAILURES so that the 'broken' hypothesis is the higher-rate one research_policy's SPRT expects
    st = RP.sprt_bernoulli([0 if o.success else 1 for o in rows], 1.0 - p_healthy, 1.0 - p_broken)
    return {"decision": getattr(st, "decision", None), "n": len(rows), "llr": getattr(st, "llr", None)}


# ------------------------------------------------------------------------------------------------ replay over history

def replay(outcomes: Sequence[Outcome], checkpoints: Sequence, cfg: HealthConfig | None = None,
           events: Sequence[KnowledgeEvent] = ()) -> HealthLedger:
    """Walk-forward health: at every checkpoint only outcomes strictly before it are visible, exactly as the live system would
    have seen them. The resulting ledger is what detection-lag and false-alarm studies are computed from."""
    cfg = cfg or HealthConfig()
    ledger = HealthLedger()
    for cp in sorted(checkpoints, key=as_date):
        seen = [o for o in outcomes if as_date(o.when) < as_date(cp)]
        ev = [e for e in events if as_date(e.when) < as_date(cp)]
        step(seen, cp, cfg, ev, ledger)
    return ledger


def detection_lag(ledger: HealthLedger, onset, kind: str, at_least: Level = Level.WATCH) -> dict:
    """How long after `onset` (the date a defect was introduced) did the ledger first report `kind`? Reports before the onset that
    already carry the finding are FALSE ALARMS and are counted separately, not folded into a flattering lag."""
    false_alarms, first = 0, None
    for r in ledger.reports:
        f = r.finding(kind)
        hit = f is not None and _LEVEL_RANK[f.level] >= _LEVEL_RANK[at_least]
        if as_date(r.now) <= as_date(onset):
            false_alarms += int(hit)
        elif hit and first is None:
            first = r.now
    return {"kind": kind, "onset": str(onset), "first_detection": first, "lag_days": None if first is None else _days(first, onset),
            "false_alarms_before_onset": false_alarms, "detected": first is not None}


def stale_ledger(ledger: HealthLedger, now, max_age_days: int = 14) -> bool:
    """A health system nobody has run lately is itself unhealthy (a monitor that stopped monitoring reads as 'no alarms')."""
    last = ledger.latest()
    return last is None or _days(now, last.now) > max_age_days


# ------------------------------------------------------------------------------------------------ planted worlds

_AREA_DIFFICULTY = {Area.KNOWN_PROMISING: 0.15, Area.UNCERTAIN: 0.55, Area.NEW_REPRESENTATION: 0.70, Area.FAILED_NEW_HYPOTHESIS: 0.85,
                    Area.RISK: 0.60, Area.VOLATILITY: 0.65, Area.DIRECTION: 0.90, Area.REGIME: 0.75, Area.DATA_QUALITY: 0.30,
                    Area.PATTERN_BREAK: 0.70}
MODES = ("healthy", "monopoly", "easy_bias", "abandon_hard", "duplicates", "memorising", "volume")


def simulate_researcher(n: int, seed: int, mode: str = "healthy", start: str = "2020-01-01", per_day: int = 2) -> list:
    """A synthetic researcher with a known, planted health state, for tests and detector calibration. `mode` plants exactly one
    defect from the onset at 40% of the run: monopoly (92% of compute on a dead-end family), easy_bias (only cheap areas),
    abandon_hard (hard questions dropped), duplicates (re-running), memorising (results fail the memorisation control), volume
    (many more experiments, no more knowledge). Deterministic in `seed`."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    day0 = as_date(start).toordinal()
    onset = int(0.4 * n)
    out: list = []
    fams = {a: [f"{a.value[:3]}_{j}" for j in range(3)] for a in AREAS}
    easy_areas = [Area.KNOWN_PROMISING, Area.DATA_QUALITY]
    for i in range(n):
        planted = mode != "healthy" and i >= onset
        day = day0 + i // per_day
        if mode == "volume" and planted:
            day = day0 + onset // per_day + (i - onset) // (per_day * 4)
        a = AREAS[int(rng.integers(len(AREAS)))]
        fam = fams[a][int(rng.integers(3))]
        diff = float(np.clip(rng.normal(_AREA_DIFFICULTY[a], 0.08), 0.02, 0.98))
        cost = float(rng.uniform(4, 12))
        if planted and mode == "monopoly" and rng.random() < 0.92:
            a, fam, diff, cost = Area.KNOWN_PROMISING, "KNO_tune", 0.15, float(rng.uniform(4, 8))
        if planted and mode == "easy_bias":
            if rng.random() < 0.9:
                a = easy_areas[int(rng.integers(2))]
                fam, diff = fams[a][int(rng.integers(3))], float(np.clip(rng.normal(0.2, 0.05), 0.02, 0.5))
        if planted and mode == "abandon_hard" and diff >= 0.65:
            a = Area.DATA_QUALITY if rng.random() < 0.97 else a
            if a is Area.DATA_QUALITY:
                fam, diff = fams[a][int(rng.integers(3))], float(np.clip(rng.normal(0.3, 0.05), 0.02, 0.5))

        p_success = 0.05 + 0.5 * (1.0 - diff) ** 2
        if planted and mode == "volume":
            p_success *= 0.15
        success = bool(rng.random() < p_success)
        dead_end = planted and mode == "monopoly" and fam == "KNO_tune"
        if dead_end:
            success = bool(rng.random() < 0.05)
        gain = 0.0005 * cost if dead_end else float(rng.exponential(0.25 * (0.4 + diff))) if success else 0.0
        claimed = success and rng.random() < 0.6
        replicated = transferred = false_disc = memorised = None
        if claimed:
            good = rng.random() < (0.93 if diff > 0.3 else 0.97)
            replicated = bool(good and rng.random() < 0.85)
            transferred = bool(replicated and rng.random() < 0.7)
            false_disc = not good
        train = holdout = None
        if success or rng.random() < 0.3:
            train = float(rng.uniform(0.4, 1.0))
            holdout = float(train * rng.uniform(0.6, 1.1))
        if planted and mode == "memorising":
            memorised = bool(rng.random() < 0.35)
            if train is not None:
                holdout = float(train * rng.uniform(0.0, 0.4))
        elif claimed:
            memorised = bool(rng.random() < 0.01)
        dup = f"E{max(0, i - int(rng.integers(1, 20))):06d}" if (planted and mode == "duplicates" and rng.random() < 0.45) else ""
        out.append(Outcome(f"E{i:06d}", dt_iso(day), a.value, fam, cost, success, gain, diff, f"{a.value[:3]}{i % 4}",
                           claimed, float(rng.uniform(0.0, 0.05)) if claimed else None, replicated, transferred, false_disc, dup,
                           train, holdout, memorised, bool(a is not Area.KNOWN_PROMISING and rng.random() < 0.5),
                           config_hash=stable_hash([fam, i % 7 if dup else i], 6)))
    return out


def dt_iso(ordinal: int) -> str:
    import datetime as _dt
    return _dt.date.fromordinal(ordinal).isoformat()


def self_check(seed: int = 0, n: int = 480) -> dict:
    """Does each planted defect trip its detector, and does the healthy world stay quiet? Returns a table of detections and
    the false-alarm count on the healthy runs. This is the check that a health monitor can fail; it is not a validation."""
    expect = {"monopoly": "FAMILY_MONOPOLY", "easy_bias": "EASY_BIAS", "abandon_hard": "HARD_ABANDONMENT",
              "volume": "VOLUME_NOT_KNOWLEDGE"}
    cfg = HealthConfig()
    out: dict = {"label": LABEL, "modes": {}}
    for mode in MODES:
        rows = simulate_researcher(n, seed, mode)
        now = dt_iso(as_date(rows[-1].when).toordinal() + 1)
        rep = step(rows, now, cfg)
        entry: dict[str, Any] = {"kinds": list(rep.kinds), "level": rep.level.value}
        if mode in expect:
            entry["detected"] = expect[mode] in rep.kinds
        if mode == "duplicates":
            m = rep.metric("duplicate_rate")
            entry["detected"] = m is not None and m.level in (Level.WATCH, Level.ALARM)
        if mode == "memorising":
            m = rep.metric("memorisation_rate")
            entry["detected"] = m is not None and m.level in (Level.WATCH, Level.ALARM)
        out["modes"][mode] = entry
    healthy_flags = 0
    for s in range(seed, seed + 5):
        rows = simulate_researcher(n, s, "healthy")
        rep = step(rows, dt_iso(as_date(rows[-1].when).toordinal() + 1), cfg)
        healthy_flags += sum(1 for f in rep.findings if f.level is Level.ALARM)
    out["healthy_alarms_in_5_runs"] = healthy_flags
    return out
