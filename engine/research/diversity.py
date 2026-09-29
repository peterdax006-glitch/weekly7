"""Research diversity controller (contract C66 section 38; supports section 37 brain health and section 43).

Prevents the autonomous researcher from becoming trapped in one hypothesis family. Effort is spread over the ten section-38
areas (known promising, uncertain, new representations, failed areas with new hypotheses, risk, volatility, direction, regime,
data quality, pattern breaks) and the split is ADAPTIVE: a discounted Thompson-sampling / UCB allocator learns which areas
produce useful discoveries per CPU-minute - where 'useful' excludes duplicates, memorised results, false claims and overfit
ones, so the allocator cannot be paid for volume (section 43). A second learner measures, per area, whether genuinely
exploratory jobs beat routine ones and sets that area's exploratory fraction. Floors stop any area starving, caps stop any
area or family monopolising, floors relax for areas proven dead but never to zero, and a failed area is re-opened only by a
hypothesis that is actually new. Directives from engine.research.brain_health (family caps, easy-area penalties, hard-question
lift, exploration floor) are honoured through the same box-simplex projection, so the result is always a valid distribution.

Built on: engine.learning.research_policy (project_box_simplex, epsilon_schedule, cusum_shift, entropy_bits, ResearchTarget /
allocator vocabulary), engine.learning.meta_learning (fit_beta_prior, wilson) and engine.research.brain_health (Area, Outcome,
Directives). Every draw is seeded; time-aware calls take `now` and raise FirewallBreach for outcomes dated at/after it.
Public entry: step(controller, now, outcomes, budget_minutes, seed). IMPLEMENTED — NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning import research_policy as RP
from engine.learning.core import FirewallBreach, ValidationLabel, as_date, current_code_hash, require_past, stable_hash
from engine.learning.meta_learning import fit_beta_prior, wilson
from engine.research.brain_health import (AREA_TO_TARGET, AREAS, EXPLOIT_AREAS, Area, BrainHealthReport, Directives, Level,
                                          Outcome, effective_number, normalised_entropy, shannon_bits)

LABEL = ValidationLabel.NOT_VALIDATED.value
EPS = 1e-9
_CODE_HASH: list = []


def code_hash() -> str:
    """Code hash stamped on reports; computed once per process because hashing the loaded modules costs ~0.2 s."""
    if not _CODE_HASH:
        _CODE_HASH.append(current_code_hash())
    return _CODE_HASH[0]


# ------------------------------------------------------------------------------------------------ specification

@dataclass(frozen=True)
class AreaSpec:
    """One section-38 area: where it starts (prior share), the least it may ever get (floor) and the most (cap)."""
    area: Area
    prior_share: float
    floor: float
    cap: float
    note: str = ""


def default_specs() -> dict:
    """Priors, not tuned values. KNOWN_PROMISING is the only capped-low area: it is exploitation, and exploitation may not
    crowd out the rest (research_policy uses the same 30% ceiling)."""
    rows = [
        (Area.KNOWN_PROMISING, 0.20, 0.03, 0.35, "extend and maintain what already works"),
        (Area.UNCERTAIN, 0.12, 0.03, 0.45, "areas where we do not know whether anything is there"),
        (Area.NEW_REPRESENTATION, 0.10, 0.03, 0.45, "new ways to represent prices, volume, structure"),
        (Area.FAILED_NEW_HYPOTHESIS, 0.08, 0.02, 0.40, "areas that failed before, only with genuinely new hypotheses"),
        (Area.RISK, 0.12, 0.04, 0.45, "loss avoidance and downside control"),
        (Area.VOLATILITY, 0.12, 0.04, 0.45, "what makes a stock move ~7% in a week"),
        (Area.DIRECTION, 0.08, 0.03, 0.40, "direction among predicted movers"),
        (Area.REGIME, 0.06, 0.02, 0.40, "regime-aware research"),
        (Area.DATA_QUALITY, 0.06, 0.02, 0.30, "data quality and coverage"),
        (Area.PATTERN_BREAK, 0.06, 0.02, 0.40, "why patterns break"),
    ]
    return {a: AreaSpec(a, p, f, c, n) for a, p, f, c, n in rows}


@dataclass(frozen=True)
class DiversityConfig:
    """All knobs in one place. `half_life_days` is the memory of the allocator: evidence older than a few half-lives barely
    counts, because what pays changes (section 38: 'learn where exploration produces useful discoveries' is a moving target)."""
    half_life_days: float = 90.0
    prior_strength: float = 20.0                    # pseudo-observations of trust in the spec priors, shrinking with data
    softmax_temperature: float = 0.5
    ucb_weight: float = 0.4
    value_prior_bits: float = 0.15
    cost_prior_minutes: float = 8.0
    dead_min_trials: int = 30
    dead_rate: float = 0.12
    floor_min: float = 0.01                         # a proven-dead area still keeps this: never zero
    explore_default: float = 0.30
    explore_lo: float = 0.10
    explore_hi: float = 0.80
    explore_draws: int = 400
    family_cap: float = 0.60                        # default in-area cap for one hypothesis family
    max_share: float = 0.45
    starved_boost: float = 2.0                      # starved areas get this x their floor
    adherence_alarm: float = 0.30                   # total-variation gap between planned and realised shares
    drift_reset: float = 0.5                        # evidence retained when an area's yield is found to have shifted down
    stagnant_weight: float = 0.2                    # a family whose marginal return has stopped keeps this fraction of its weight
    eig_weight: float = 0.3                         # weight of the information value of exploring an uncertain area
    damping: float = 0.25                           # weight kept on the previous plan, so shares do not thrash round to round
    history_len: int = 200

    def validate(self) -> list:
        errs = []
        if self.half_life_days <= 0:
            errs.append("half_life_days must be positive")
        if not 0 < self.softmax_temperature:
            errs.append("temperature must be positive")
        if not 0 < self.floor_min <= 0.05:
            errs.append("floor_min must be in (0, 0.05]")
        if not 0 <= self.explore_lo <= self.explore_default <= self.explore_hi <= 1:
            errs.append("explore_lo <= explore_default <= explore_hi in [0,1] required")
        if not 0 < self.family_cap <= 1 or not 0 < self.max_share <= 1:
            errs.append("caps must be in (0,1]")
        if not 0 < self.drift_reset <= 1:
            errs.append("drift_reset must be in (0,1]")
        if not 0 <= self.damping < 1:
            errs.append("damping must be in [0,1)")
        return errs


def validate_specs(specs: Mapping[Area, AreaSpec], cfg: DiversityConfig | None = None) -> list:
    """The box must be feasible and cover every area, or the projection would raise in the middle of a research round."""
    errs = []
    if set(specs) != set(AREAS):
        errs.append("specs must cover exactly the ten section-38 areas")
        return errs
    if abs(sum(s.prior_share for s in specs.values()) - 1.0) > 1e-6:
        errs.append("prior shares must sum to 1")
    if sum(s.floor for s in specs.values()) > 1.0 - 1e-9:
        errs.append("floors sum to more than 1")
    if sum(s.cap for s in specs.values()) < 1.0 - 1e-9:
        errs.append("caps sum to less than 1: no distribution fits")
    for s in specs.values():
        if not 0 < s.floor <= s.prior_share <= s.cap <= 1:
            errs.append(f"{s.area.value}: need 0 < floor <= prior <= cap <= 1")
    if cfg is not None and sum(min(s.cap, cfg.max_share) for s in specs.values()) < 1.0 - 1e-9:
        errs.append("max_share leaves no feasible distribution")
    return errs


# ------------------------------------------------------------------------------------------------ what counts as useful

def is_useful(o: Outcome, overfit_gap: float = 0.5) -> bool:
    """A discovery the allocator may be paid for. A success that was a duplicate, was memorised, overfit its holdout, or was a
    claim that proved false (or failed to replicate) is NOT useful: paying for those would teach the allocator to farm them."""
    if not o.success or o.duplicate_of or o.memorised or o.false_discovery:
        return False
    if o.claimed_discovery and o.replicated is False:
        return False
    if o.train_score is not None and o.holdout_score is not None and o.train_score > 0 \
            and o.holdout_score < (1.0 - overfit_gap) * o.train_score:
        return False
    return True


def usefulness_weight(o: Outcome) -> float:
    """1 for a useful result, 2 when it also transferred (the strongest evidence a discovery generalises), 0 otherwise."""
    if not is_useful(o):
        return 0.0
    return 2.0 if o.verified else 1.0


# ------------------------------------------------------------------------------------------------ per-area evidence

@dataclass
class AreaStats:
    """Discounted evidence for one area (weights decay with age) plus raw counters for audit. Mutable on purpose: it is the
    controller's private state, saved and restored as a whole."""
    trials: float = 0.0
    useful: float = 0.0
    weighted: float = 0.0
    bits: float = 0.0
    minutes: float = 0.0
    exp_trials: float = 0.0
    exp_useful: float = 0.0
    routine_trials: float = 0.0
    routine_useful: float = 0.0
    raw_trials: int = 0
    raw_useful: int = 0
    stale_repeats: int = 0
    new_hypotheses: int = 0
    last_seen: str = ""
    last_useful: str = ""
    series: list = field(default_factory=list)          # 1/0 per outcome, newest last (for drift detection)

    def decay(self, factor: float) -> None:
        for name in ("trials", "useful", "weighted", "bits", "minutes", "exp_trials", "exp_useful", "routine_trials",
                     "routine_useful"):
            setattr(self, name, getattr(self, name) * factor)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "AreaStats":
        return cls(**{k: d[k] for k in d if k in {f.name for f in dataclasses.fields(cls)}})


@dataclass(frozen=True)
class Observation:
    """What the controller concluded about one outcome it ingested."""
    exp_id: str
    area: str
    useful: bool
    exploratory_credit: bool
    stale_repeat: bool
    ignored: bool = False
    reason: str = ""


# ------------------------------------------------------------------------------------------------ the plan

@dataclass(frozen=True)
class DiversityPlan:
    """The allocation for the next research round. `shares` always sums to 1, respects every floor and cap, and carries its
    own rationale so the split can be audited rather than trusted."""
    now: str
    budget_minutes: float
    seed: int
    shares: Mapping
    minutes: Mapping
    floors: Mapping
    caps: Mapping
    exploratory_fraction: Mapping
    explore_share: float
    exploit_share: float
    family_split: Mapping
    reopen_questions: tuple
    rationale: tuple
    adherence_previous: float | None = None
    n_observed: int = 0
    n_ignored: int = 0
    drift_events: tuple = ()
    code_hash: str = ""
    label: str = LABEL
    hard_quota: float = 0.25

    def validate(self) -> list:
        errs = []
        if abs(sum(self.shares.values()) - 1.0) > 1e-6:
            errs.append("shares do not sum to 1")
        for a, s in self.shares.items():
            if s < self.floors[a] - 1e-9:
                errs.append(f"{a} below its floor")
            if s > self.caps[a] + 1e-9:
                errs.append(f"{a} above its cap")
        if set(self.shares) != {a.value for a in AREAS}:
            errs.append("plan does not cover every area")
        if abs(sum(self.minutes.values()) - self.budget_minutes) > 1e-6 * max(1.0, self.budget_minutes):
            errs.append("minutes do not add up to the budget")
        for a, fam in self.family_split.items():
            if fam and abs(sum(fam.values()) - 1.0) > 1e-6:
                errs.append(f"family split for {a} does not sum to 1")
        return errs

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["family_split"] = {a: dict(f) for a, f in self.family_split.items()}
        return d

    def render(self) -> str:
        lines = [f"Diversity plan {self.now}: explore {self.explore_share:.0%} / exploit {self.exploit_share:.0%}  [{self.label}]"]
        for a in AREAS:
            v = a.value
            lines.append(f"  {v:22s} {self.shares[v]:6.1%}  ({self.floors[v]:.0%}-{self.caps[v]:.0%})  "
                         f"exploratory {self.exploratory_fraction[v]:.0%}  {self.minutes[v]:7.0f} min")
        lines.extend("  - " + r for r in self.rationale)
        return "\n".join(lines)


def target_pressure(plan: DiversityPlan) -> dict:
    """Bridge to research_policy.TargetAllocator.allocate(pressure=...): each area's share becomes the open-work pressure on
    its ResearchTarget, so the existing ten-target allocator is steered by this controller instead of duplicated."""
    return {AREA_TO_TARGET[Area.parse(a)].value: float(s) for a, s in plan.shares.items()}


# ------------------------------------------------------------------------------------------------ the controller

class DiversityController:
    """Holds the learned state; `step()` is the public way to advance it. Nothing here reads outside `now`."""

    def __init__(self, cfg: DiversityConfig | None = None, specs: Mapping[Area, AreaSpec] | None = None):
        self.cfg = cfg or DiversityConfig()
        self.specs = dict(specs or default_specs())
        errs = self.cfg.validate() + validate_specs(self.specs, self.cfg)
        if errs:
            raise ValueError("invalid diversity setup: " + "; ".join(errs))
        self.stats = {a: AreaStats() for a in AREAS}
        self.seen_ids: set = set()
        self.seen_hypotheses: set = set()
        self.family_stats: dict = {}
        self.clock: str | None = None
        self.plans: list = []
        self.spent_since_plan = {a.value: 0.0 for a in AREAS}
        self.reopen_marks = {a: (0, 0) for a in AREAS}       # (raw_trials, raw_useful) when the area was last re-opened
        self.total_experiments = 0
        self.rounds: list = []

    # -- time -------------------------------------------------------------------------------------------------
    def decay_factor(self, days: float) -> float:
        return 0.5 ** (max(days, 0.0) / self.cfg.half_life_days)

    def advance(self, now) -> None:
        """Age all evidence to `now`. Time may not run backwards: a rewound clock would let old evidence count as new."""
        if self.clock is None:
            self.clock = str(now)
            return
        dt = (as_date(now) - as_date(self.clock)).days
        if dt < 0:
            raise FirewallBreach(f"diversity clock cannot move back from {self.clock} to {now}")
        if dt > 0:
            f = self.decay_factor(dt)
            for st in self.stats.values():
                st.decay(f)
            for fs in self.family_stats.values():
                for k in ("trials", "useful", "bits", "minutes"):
                    fs[k] *= f
        self.clock = str(now)

    # -- evidence ---------------------------------------------------------------------------------------------
    def observe(self, o: Outcome, now) -> Observation:
        """Ingest one matured outcome, weighted by its age. Idempotent on exp_id. In FAILED_NEW_HYPOTHESIS a repeat of a
        hypothesis already run is a stale repeat: it earns nothing (a failed area is re-opened only by something new)."""
        require_past(o.when, now, f"outcome {o.exp_id}")
        errs = o.validate()
        if errs:
            raise ValueError(f"{o.exp_id}: " + "; ".join(errs))
        if o.exp_id in self.seen_ids:
            return Observation(o.exp_id, o.area, False, False, False, True, "already ingested")
        area = Area.parse(o.area)
        st = self.stats[area]
        key = (area.value, o.family, o.config_hash) if o.config_hash else None
        repeated = key is not None and key in self.seen_hypotheses
        stale = area is Area.FAILED_NEW_HYPOTHESIS and repeated
        credit = bool(o.exploratory and not repeated)
        useful = is_useful(o) and not stale
        w = self.decay_factor((as_date(now) - as_date(o.when)).days)
        st.trials += w
        st.useful += w * useful
        st.weighted += w * (usefulness_weight(o) if useful else 0.0)
        st.bits += w * (o.gain_bits if useful else 0.0)
        st.minutes += w * max(o.cost_minutes, 0.0)
        if credit:
            st.exp_trials += w
            st.exp_useful += w * useful
            st.new_hypotheses += 1
        else:
            st.routine_trials += w
            st.routine_useful += w * useful
        st.raw_trials += 1
        st.raw_useful += int(useful)
        st.stale_repeats += int(stale)
        st.last_seen = max(st.last_seen, str(o.when)) if st.last_seen else str(o.when)
        if useful:
            st.last_useful = max(st.last_useful, str(o.when)) if st.last_useful else str(o.when)
        st.series.append(1.0 if useful else 0.0)
        del st.series[:-self.cfg.history_len]
        fs = self.family_stats.setdefault((area.value, o.family or "(none)"), {"trials": 0.0, "useful": 0.0, "bits": 0.0, "minutes": 0.0, "gains": []})
        fs["gains"] = (fs["gains"] + [float(o.gain_bits)])[-20:]
        fs["trials"] += w
        fs["useful"] += w * useful
        fs["bits"] += w * (o.gain_bits if useful else 0.0)
        fs["minutes"] += w * max(o.cost_minutes, 0.0)
        self.spent_since_plan[area.value] += max(o.cost_minutes, 0.0)
        self.seen_ids.add(o.exp_id)
        if key is not None:
            self.seen_hypotheses.add(key)
        self.total_experiments += 1
        return Observation(o.exp_id, area.value, useful, credit, stale)

    def reopen(self, area: Area | str) -> None:
        """A genuinely new hypothesis for a failed area: forget its dead verdict so the floor and evidence restart."""
        a = Area.parse(area)
        st = self.stats[a]
        self.reopen_marks[a] = (st.raw_trials, st.raw_useful)

    # -- beliefs ----------------------------------------------------------------------------------------------
    def prior(self) -> tuple:
        """Empirical-Bayes Beta prior pooled across areas (meta_learning.fit_beta_prior): areas borrow strength from each
        other, so a two-trial area is not believed to be either hopeless or brilliant."""
        s = [int(round(self.stats[a].useful)) for a in AREAS]
        n = [int(round(self.stats[a].trials)) for a in AREAS]
        return fit_beta_prior(s, n)

    def is_dead(self, area: Area) -> bool:
        st = self.stats[area]
        t0, u0 = self.reopen_marks[area]
        n, k = st.raw_trials - t0, st.raw_useful - u0
        return n >= self.cfg.dead_min_trials and wilson(k, n)[1] < self.cfg.dead_rate

    def area_rate(self, area: Area) -> dict:
        st = self.stats[area]
        a0, b0 = self.prior()
        a, b = a0 + st.useful, b0 + max(st.trials - st.useful, 0.0)
        return {"mean": a / (a + b), "n_eff": st.trials, "raw_n": st.raw_trials, "dead": self.is_dead(area)}

    def exploration_lift(self, area: Area, seed: int) -> dict:
        """Does trying something new pay in this area? P(useful-rate of exploratory jobs > routine jobs), by seeded Monte Carlo
        over the two Beta posteriors. With fewer than 3 exploratory trials the answer is 'unknown' and the fraction stays default."""
        st = self.stats[area]
        rng = np.random.default_rng(seed)
        pe = rng.beta(1 + st.exp_useful, 1 + max(st.exp_trials - st.exp_useful, 0.0), self.cfg.explore_draws)
        pr = rng.beta(1 + st.routine_useful, 1 + max(st.routine_trials - st.routine_useful, 0.0), self.cfg.explore_draws)
        known = st.exp_trials >= 3
        p_better = float((pe > pr).mean())
        return {"p_explore_better": p_better if known else None, "lift": float((pe - pr).mean()) if known else None, "known": known}

    def exploratory_fraction(self, area: Area, seed: int) -> float:
        lift = self.exploration_lift(area, seed)
        if not lift["known"]:
            return self.cfg.explore_default
        f = self.cfg.explore_lo + (self.cfg.explore_hi - self.cfg.explore_lo) * lift["p_explore_better"]
        return float(min(self.cfg.explore_hi, max(self.cfg.explore_lo, f)))

    def drift(self) -> dict:
        return {a.value: RP.cusum_shift(self.stats[a].series) for a in AREAS}

    def handle_drift(self) -> tuple:
        """If an area's yield has shifted DOWN, keep only `drift_reset` of its evidence so the allocator re-learns quickly
        instead of clinging to a past that no longer holds. Shifts up are reported and left alone."""
        events = []
        for a, d in self.drift().items():
            if d["verdict"] != "SHIFT":
                continue
            events.append((a, d["direction"], d["alarm"]))
            if d["direction"] == "down":
                st = self.stats[Area.parse(a)]
                st.decay(self.cfg.drift_reset)
                st.series = st.series[d["alarm"]:]
        return tuple(events)

    # -- allocation -------------------------------------------------------------------------------------------
    def bounds(self, now, directives: Directives) -> tuple:
        """Per-area (floor, cap, notes). Floors relax (never below floor_min) for areas proven dead, and rise for starved ones;
        the exploit area's cap is pulled down so exploration keeps at least max(epsilon schedule, directive floor)."""
        lo, hi, why = {}, {}, []
        e_min = max(RP.epsilon_schedule(self.total_experiments), directives.min_explore_share)
        starved = set(directives.starved_areas)
        for a in AREAS:
            sp = self.specs[a]
            floor, cap = sp.floor, min(sp.cap, self.cfg.max_share)
            st = self.stats[a]
            if self.is_dead(a) and a.value not in starved:
                floor = self.cfg.floor_min
                why.append(f"{a.value}: no useful result in {st.raw_trials} trials, floor relaxed to {floor:.0%} (never zero)")
            elif a.value in starved or (st.last_seen and (as_date(now) - as_date(st.last_seen)).days > 60):
                floor = min(cap, self.cfg.starved_boost * sp.floor)
                why.append(f"{a.value}: starved, floor raised to {floor:.0%}")
            lo[a], hi[a] = floor, cap
        exploit_cap = min(sum(hi[a] for a in EXPLOIT_AREAS), 1.0 - e_min)
        for a in EXPLOIT_AREAS:
            if exploit_cap < hi[a]:
                hi[a] = max(lo[a], exploit_cap)
                why.append(f"exploitation held to {hi[a]:.0%} so exploration keeps {1 - hi[a]:.0%} (schedule {e_min:.0%})")
        return lo, hi, why

    def raw_scores(self, seed: int) -> dict:
        """Thompson draw of (useful rate) x (bits per useful result) / (minutes per job), inflated by a UCB bonus for
        under-tried areas. Higher = a better place for the next CPU-minute."""
        rng = np.random.default_rng(seed)
        a0, b0 = self.prior()
        n_total = sum(st.trials for st in self.stats.values())
        out = {}
        for a in AREAS:
            st = self.stats[a]
            p = float(rng.beta(a0 + st.useful, b0 + max(st.trials - st.useful, 0.0)))
            val = (st.bits + 2.0 * self.cfg.value_prior_bits) / (st.useful + 2.0)
            cost = (st.minutes + 2.0 * self.cfg.cost_prior_minutes) / (st.trials + 2.0)
            ucb = math.sqrt(2.0 * math.log(n_total + 2.0) / (st.trials + 1.0))
            eig = RP.eig_beta_binomial(a0 + st.useful, b0 + max(st.trials - st.useful, 0.0), 10)
            score = (p * val / cost + 1e-4) * (1.0 + self.cfg.ucb_weight * ucb + self.cfg.eig_weight * eig)
            out[a] = score ** (1.0 / self.cfg.softmax_temperature)
        return out

    def allocate(self, now, budget_minutes: float, seed: int, directives: Directives | None = None, commit: bool = True) -> DiversityPlan:
        if budget_minutes <= 0 or not math.isfinite(budget_minutes):
            raise ValueError("budget_minutes must be positive and finite")
        directives = directives or Directives()
        bad = directives.validate()
        if bad:
            raise ValueError("invalid directives: " + "; ".join(bad))
        if commit:
            self.advance(now)
        raw = self.raw_scores(seed)
        tot = sum(raw.values()) or 1.0
        n = sum(st.trials for st in self.stats.values())
        w_prior = self.cfg.prior_strength / (self.cfg.prior_strength + n)
        mix = {a: w_prior * self.specs[a].prior_share + (1.0 - w_prior) * raw[a] / tot for a in AREAS}
        rationale = [f"prior weight {w_prior:.0%} (evidence {n:.0f} effective trials)"]
        for name, m in directives.area_multiplier.items():
            mix[Area.parse(name)] *= m
            rationale.append(f"{name}: health multiplier x{m:.2f}")
        lo, hi, why = self.bounds(now, directives)
        rationale.extend(why)
        if sum(lo.values()) > 1.0 - 1e-9 or sum(hi.values()) < 1.0 - 1e-9:
            hi = {a: self.specs[a].cap for a in AREAS}
            rationale.append("max_share/exploration bound infeasible: caps fall back to the specifications")
        shares = RP.project_box_simplex({a.value: mix[a] for a in AREAS}, {a.value: lo[a] for a in AREAS},
                                        {a.value: hi[a] for a in AREAS})
        if self.plans and self.cfg.damping > 0:
            last = self.plans[-1][1]
            shares = RP.project_box_simplex({k: (1 - self.cfg.damping) * v + self.cfg.damping * last.get(k, v) for k, v in shares.items()},
                                            {a.value: lo[a] for a in AREAS}, {a.value: hi[a] for a in AREAS})
            rationale.append(f"damped {self.cfg.damping:.0%} toward the previous plan")
        explore = 1.0 - sum(shares[a.value] for a in EXPLOIT_AREAS)
        fam = {a.value: self.family_split(a, directives) for a in AREAS}
        prev = self.adherence()
        plan = DiversityPlan(
            str(now), float(budget_minutes), int(seed), shares, {a: s * budget_minutes for a, s in shares.items()},
            {a.value: lo[a] for a in AREAS}, {a.value: hi[a] for a in AREAS},
            {a.value: self.exploratory_fraction(a, seed + 1 + i) for i, a in enumerate(AREAS)}, explore, 1.0 - explore, fam,
            directives.reopen_questions, tuple(rationale), prev, 0, 0, (), code_hash(), LABEL, hard_quota(directives))
        if commit:
            self.spent_since_plan = {a.value: 0.0 for a in AREAS}
            self.plans.append((str(now), dict(shares)))
            del self.plans[:-self.cfg.history_len]
        return plan

    def family_split(self, area: Area, directives: Directives) -> dict:
        """Split an area's routine minutes over the hypothesis families seen in it, by shrunk useful-rate per minute, with a
        per-family cap (directive caps override the default). A family already at its cap cannot absorb more, so one
        parameter family can never take the whole area."""
        fams = {f: s for (a, f), s in self.family_stats.items() if a == area.value}
        if not fams:
            return {}
        names = sorted(fams)
        raw = {}
        for f in names:
            s = fams[f]
            rate = (s["useful"] + 1.0) / (s["trials"] + 2.0)
            per_min = (s["minutes"] + self.cfg.cost_prior_minutes) / (s["trials"] + 1.0)
            raw[f] = rate * (1.0 + s["bits"] / (s["trials"] + 1.0)) / per_min
        dead_ends = set(stagnant_families(self).get(area.value, ()))
        raw = {f: (r * self.cfg.stagnant_weight if f in dead_ends else r) for f, r in raw.items()}
        cap = {f: min(1.0, directives.family_caps.get(f, self.cfg.family_cap)) for f in names}
        cap_sum = sum(cap.values())
        if cap_sum < 1.0:
            cap = {f: c / cap_sum for f, c in cap.items()}                       # too few families to honour the caps
        floor = {f: min(0.02, 0.5 / len(names)) for f in names}
        return RP.project_box_simplex(raw, floor, cap)

    # -- self-audit -------------------------------------------------------------------------------------------
    def adherence(self) -> float | None:
        """Total-variation distance between the last plan and where minutes actually went since. A large gap means the
        allocator is planning one thing and the loop doing another - the plan is only advice until it is followed."""
        if not self.plans:
            return None
        spent = sum(self.spent_since_plan.values())
        if spent <= 0:
            return None
        last = self.plans[-1][1]
        return 0.5 * sum(abs(last[a] - self.spent_since_plan[a] / spent) for a in last)


def audit_plan(plan: DiversityPlan, ctrl: DiversityController) -> list:
    """Independent re-check of a plan against the specifications. Empty list = consistent."""
    errs = list(plan.validate())
    for a in AREAS:
        sp = ctrl.specs[a]
        if plan.shares[a.value] < ctrl.cfg.floor_min - 1e-9:
            errs.append(f"{a.value} starved below the absolute floor")
        if plan.shares[a.value] > sp.cap + 1e-9:
            errs.append(f"{a.value} exceeds its specified cap")
    if plan.exploit_share > 1.0 - ctrl.cfg.floor_min:
        errs.append("no exploration left")
    if any(not 0.0 <= f <= 1.0 for f in plan.exploratory_fraction.values()):
        errs.append("exploratory fraction outside [0,1]")
    return errs


def step(ctrl: DiversityController, now, outcomes: Iterable[Outcome] = (), budget_minutes: float = 600.0, seed: int = 0,
         report: BrainHealthReport | None = None, directives: Directives | None = None) -> DiversityPlan:
    """Public entry (wave-2 research loop): ingest the outcomes that matured before `now`, age the evidence, handle drift, take
    the brain-health directives, and return the next allocation. An outcome dated at/after `now` raises FirewallBreach."""
    ctrl.advance(now)
    rows = sorted(outcomes, key=lambda o: (as_date(o.when), o.exp_id))
    obs = [ctrl.observe(o, now) for o in rows]
    events = ctrl.handle_drift()
    d = directives if directives is not None else (report.directives if report is not None else Directives())
    for name in d.starved_areas:
        if any(o.area == name and o.exploratory_credit and not o.stale_repeat and not o.ignored for o in obs):
            ctrl.reopen(name)
    plan = ctrl.allocate(now, budget_minutes, seed, d)
    log_round(ctrl, now, plan, obs, rows)
    return dataclasses.replace(plan, n_observed=sum(1 for o in obs if not o.ignored), n_ignored=sum(1 for o in obs if o.ignored),
                               drift_events=events)


# ------------------------------------------------------------------------------------------------ balance and reporting

def balance_report(shares: Mapping[str, float]) -> dict:
    """How spread is a split? Entropy (normalised to the ten areas), effective number of areas, smallest and largest share,
    and the exploit/explore division."""
    vals = [float(shares.get(a.value, 0.0)) for a in AREAS]
    tot = sum(vals) or 1.0
    p = [v / tot for v in vals]
    return {"normalised_entropy": normalised_entropy(p, len(AREAS)), "effective_areas": effective_number(p), "min_share": min(p),
            "max_share": max(p), "exploit_share": sum(p[i] for i, a in enumerate(AREAS) if a in EXPLOIT_AREAS),
            "entropy_bits": shannon_bits(p)}


def realised_shares(outcomes: Sequence[Outcome]) -> dict:
    """Where the minutes actually went, per area, from matured outcomes (falls back to counts when no minutes recorded)."""
    m = {a.value: 0.0 for a in AREAS}
    for o in outcomes:
        m[o.area] += max(o.cost_minutes, 0.0)
    if sum(m.values()) <= 0:
        for o in outcomes:
            m[o.area] += 1.0
    tot = sum(m.values()) or 1.0
    return {a: v / tot for a, v in m.items()}


def total_variation(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def learned_exploration_table(ctrl: DiversityController, seed: int = 0) -> list:
    """Per area: routine vs exploratory usefulness and the posterior probability that exploring beats routine work - the
    section-38 question 'where does exploration produce useful discoveries?' answered from the ledger."""
    rows = []
    for i, a in enumerate(AREAS):
        st = ctrl.stats[a]
        lift = ctrl.exploration_lift(a, seed + i)
        rows.append({"area": a.value, "exploratory_trials": st.exp_trials, "exploratory_rate": (st.exp_useful / st.exp_trials) if st.exp_trials > 0 else None,
                     "routine_trials": st.routine_trials, "routine_rate": (st.routine_useful / st.routine_trials) if st.routine_trials > 0 else None,
                     "p_explore_better": lift["p_explore_better"], "fraction": ctrl.exploratory_fraction(a, seed + i)})
    return rows


def starving_areas(ctrl: DiversityController, now, days: int = 60) -> list:
    """Areas with no experiment for `days` (or never): the independent check that floors did their job."""
    out = []
    for a in AREAS:
        seen = ctrl.stats[a].last_seen
        if not seen:
            out.append({"area": a.value, "never_run": True, "days": None})
        elif (as_date(now) - as_date(seen)).days > days:
            out.append({"area": a.value, "never_run": False, "days": (as_date(now) - as_date(seen)).days})
    return out


def explain_shift(prev: Mapping[str, float] | None, plan: DiversityPlan, tol: float = 0.02) -> list:
    """Plain-language list of what moved between two plans and why (the rationale is attached), so a change of research
    direction is never silent."""
    if prev is None:
        return ["first plan: " + ", ".join(f"{a} {s:.0%}" for a, s in sorted(plan.shares.items(), key=lambda kv: -kv[1])[:3])]
    out = []
    for a in AREAS:
        d = plan.shares[a.value] - prev.get(a.value, 0.0)
        if abs(d) >= tol:
            out.append(f"{a.value} {'up' if d > 0 else 'down'} {abs(d):.1%} to {plan.shares[a.value]:.1%}")
    return out + [f"why: {r}" for r in plan.rationale[:3]] if out else ["no area moved by more than the tolerance"]


def preview_distribution(ctrl: DiversityController, now, budget_minutes: float, seed: int, n_draws: int = 50,
                        directives: Directives | None = None) -> dict:
    """How uncertain is the plan itself? Thompson sampling makes each plan a draw; this takes `n_draws` UNCOMMITTED plans (the
    controller is not changed) and reports the mean and spread of each area's share. A wide spread on an area means the
    evidence has not settled where compute should go, which is what exploration is for."""
    if n_draws < 2:
        raise ValueError("need at least 2 draws")
    rows = np.array([[ctrl.allocate(now, budget_minutes, seed * 7919 + i, directives, commit=False).shares[a.value] for a in AREAS]
                     for i in range(n_draws)])
    return {a.value: {"mean": float(rows[:, j].mean()), "sd": float(rows[:, j].std(ddof=1)),
                      "lo": float(np.percentile(rows[:, j], 5)), "hi": float(np.percentile(rows[:, j], 95))}
            for j, a in enumerate(AREAS)}


def compare_splits(ctrl: DiversityController, a: Mapping[str, float], b: Mapping[str, float], seed: int, n_draws: int = 2000) -> dict:
    """P(split a yields more useful results per minute than split b) under the controller's current posteriors. Each draw
    samples every area's useful-rate from its Beta belief and scores both splits with it; ties are split. A comparison of plans,
    not proof either is right."""
    rng = np.random.default_rng(seed)
    a0, b0 = ctrl.prior()
    draws = np.column_stack([rng.beta(a0 + ctrl.stats[x].useful, b0 + max(ctrl.stats[x].trials - ctrl.stats[x].useful, 0.0), n_draws)
                             for x in AREAS])
    wa = np.array([a.get(x.value, 0.0) for x in AREAS])
    wb = np.array([b.get(x.value, 0.0) for x in AREAS])
    diff = draws @ wa - draws @ wb
    return {"p_a_better": float((diff > 0).mean() + 0.5 * (diff == 0).mean()), "mean_diff": float(diff.mean()),
            "ci": (float(np.percentile(diff, 5)), float(np.percentile(diff, 95)))}


def coverage_matrix(ctrl: DiversityController) -> dict:
    """Area x family effective trial counts, and the cells that have never been tried. Empty cells are the map of what the
    researcher has not looked at yet."""
    fams = sorted({f for (_, f) in ctrl.family_stats})
    cells = {a.value: {f: round(ctrl.family_stats.get((a.value, f), {}).get("trials", 0.0), 3) for f in fams} for a in AREAS}
    empty = [(a, f) for a, row in cells.items() for f, n in row.items() if n == 0 and f.startswith(a[:3])]
    return {"families": fams, "cells": cells, "untried_within_area": empty}


# ------------------------------------------------------------------------------------------------ questions to areas

_SOURCE_AREA = {"loss": Area.RISK, "risk": Area.RISK, "break": Area.PATTERN_BREAK, "pattern_break": Area.PATTERN_BREAK,
                "regime": Area.REGIME, "missed_winner": Area.VOLATILITY, "surprise": Area.UNCERTAIN,
                "contradiction": Area.UNCERTAIN, "discovery": Area.NEW_REPRESENTATION, "data": Area.DATA_QUALITY,
                "failed": Area.FAILED_NEW_HYPOTHESIS, "known": Area.KNOWN_PROMISING}
_PROBLEM_AREA = {"VOLATILITY": Area.VOLATILITY, "LOSS_AVOIDANCE": Area.RISK, "DIRECTION": Area.DIRECTION,
                 "DATA_QUALITY": Area.DATA_QUALITY, "CONSISTENCY": Area.KNOWN_PROMISING}


def classify_question(source: str, problem=None) -> Area:
    """Which section-38 area a research question (section 40) belongs to: by its source first, then by the problem it serves,
    and UNCERTAIN when neither says. Unknown stays UNCERTAIN, never silently KNOWN_PROMISING."""
    src = str(source).lower()
    for key, area in _SOURCE_AREA.items():
        if key in src:
            return area
    if problem is not None:
        return _PROBLEM_AREA.get(str(getattr(problem, "value", problem)), Area.UNCERTAIN)
    return Area.UNCERTAIN


def select_questions(plan: DiversityPlan, questions: Sequence[Any], minutes_each: float = 30.0, reopen_first: bool = True) -> list:
    """Turn the plan into a work list. Each question (duck-typed: question_id, source, problem) is filed under its area; areas
    are served by deficit round-robin - every area gets questions in proportion to its share of the budget - and reopen-listed
    questions go first. Returns question ids in execution order, never exceeding the plan's minutes."""
    queues: dict = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.question_id):
        queues[classify_question(q.source, getattr(q, "problem", None)).value].append(q)
    credit = {a: 0.0 for a in queues}
    left = {a.value: plan.minutes[a.value] for a in AREAS}
    order: list = []
    if reopen_first:
        for a in sorted(queues):
            for q in [q for q in queues[a] if q.question_id in plan.reopen_questions]:
                if left[a] >= minutes_each:
                    order.append(q.question_id)
                    left[a] -= minutes_each
                    queues[a].remove(q)
    while True:
        ready = [a for a in queues if queues[a] and left[a] >= minutes_each]
        if not ready:
            return order
        for a in ready:
            credit[a] += plan.shares[a]
        pick = max(ready, key=lambda a: (credit[a], a))
        credit[pick] -= 1.0
        left[pick] -= minutes_each
        order.append(queues[pick].pop(0).question_id)


# ------------------------------------------------------------------------------------------------ learning accounting

def log_round(ctrl: DiversityController, now, plan: DiversityPlan, observations: Sequence[Observation],
              outcomes: Sequence[Outcome]) -> dict:
    """Append one line to the controller's round ledger: what the previous evidence looked like when this plan was made.
    Only ingested (non-ignored) outcomes count. The ledger is what learning_curve() and regret are computed from."""
    minutes = {a.value: 0.0 for a in AREAS}
    useful = {a.value: 0 for a in AREAS}
    by_id = {o.exp_id: o for o in outcomes}
    for ob in observations:
        if ob.ignored:
            continue
        minutes[ob.area] += max(by_id[ob.exp_id].cost_minutes, 0.0)
        useful[ob.area] += int(ob.useful)
    rec = {"now": str(now), "shares": dict(plan.shares), "minutes": minutes, "useful": useful}
    ctrl.rounds.append(rec)
    del ctrl.rounds[:-ctrl.cfg.history_len]
    return rec


def learning_curve(ctrl: DiversityController, seed: int = 0, n_boot: int = 500) -> dict:
    """Is the allocator getting better at finding useful results per CPU-minute? Compares the useful-per-1000-minutes of the
    earliest third of rounds with the latest third, with a seeded bootstrap over rounds. Needs >= 6 rounds with spend. A rising
    curve is necessary, not sufficient: the world may simply have got easier (compare against uniform in a planted world)."""
    rows = [r for r in ctrl.rounds if sum(r["minutes"].values()) > 0]
    if len(rows) < 6:
        return {"verdict": "INSUFFICIENT", "rounds": len(rows)}
    per = np.array([1000.0 * sum(r["useful"].values()) / sum(r["minutes"].values()) for r in rows])
    k = len(per) // 3
    early, late = per[:k], per[-k:]
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(late, k).mean() - rng.choice(early, k).mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))
    verdict = "IMPROVING" if lo > 0 else "WORSENING" if hi < 0 else "FLAT"
    return {"verdict": verdict, "rounds": len(rows), "early": float(early.mean()), "late": float(late.mean()), "diff_ci": (lo, hi)}


def realised_regret(ctrl: DiversityController) -> dict:
    """Ex-post regret of the plans that were followed: for each round, the useful-rate the spend achieved against the rate the
    best single area achieved that round. Positive regret is normal (exploration costs); what matters is that it shrinks."""
    out = []
    for r in ctrl.rounds:
        rates = {a: r["useful"][a] / r["minutes"][a] for a in r["minutes"] if r["minutes"][a] > 0}
        tot = sum(r["minutes"].values())
        if not rates or tot <= 0:
            continue
        got = sum(r["useful"].values()) / tot
        out.append({"now": r["now"], "achieved": got, "best_area": max(rates.values()), "regret": max(rates.values()) - got})
    if not out:
        return {"rounds": 0, "mean_regret": None}
    return {"rounds": len(out), "mean_regret": float(np.mean([o["regret"] for o in out])),
            "first_half": float(np.mean([o["regret"] for o in out[:len(out) // 2]])) if len(out) > 1 else None,
            "second_half": float(np.mean([o["regret"] for o in out[len(out) // 2:]])) if len(out) > 1 else None}


def thrash_index(ctrl: DiversityController, last: int = 10) -> float | None:
    """Mean plan-to-plan total-variation distance over the last plans. High = the split lurches; damping should keep it small."""
    seq = [s for _, s in ctrl.plans[-(last + 1):]]
    if len(seq) < 2:
        return None
    return float(np.mean([total_variation(a, b) for a, b in zip(seq, seq[1:])]))


# ------------------------------------------------------------------------------------------------ walk-forward and sensitivity

def replay_plans(outcomes: Sequence[Outcome], checkpoints: Sequence, budget_minutes: float = 600.0, seed: int = 0,
                 cfg: DiversityConfig | None = None, directives_by_checkpoint: Mapping | None = None) -> list:
    """Run the controller the way the live loop would: at each checkpoint only outcomes dated strictly before it (and not yet
    ingested) are visible. Returns the plan at each checkpoint. Look-ahead is impossible by construction: the outcomes handed
    to step() are filtered by date first, and step() raises FirewallBreach if that filter were ever removed."""
    ctrl = DiversityController(cfg)
    plans, done = [], set()
    for i, cp in enumerate(sorted(checkpoints, key=as_date)):
        fresh = [o for o in outcomes if as_date(o.when) < as_date(cp) and o.exp_id not in done]
        done.update(o.exp_id for o in fresh)
        d = (directives_by_checkpoint or {}).get(str(cp))
        plans.append(step(ctrl, cp, fresh, budget_minutes, seed + i, directives=d))
    return plans


def half_life_sensitivity(outcomes: Sequence[Outcome], checkpoint, half_lives: Sequence[float], seed: int = 0) -> dict:
    """How much does today's plan depend on the memory length? Rebuilds the controller under each half-life on the same
    outcomes and reports the plans and their largest pairwise distance. A plan that flips with the half-life is not settled."""
    shares = {}
    for hl in half_lives:
        ctrl = DiversityController(DiversityConfig(half_life_days=hl))
        shares[hl] = step(ctrl, checkpoint, [o for o in outcomes if as_date(o.when) < as_date(checkpoint)], 600.0, seed).shares
    keys = list(shares)
    worst = max((total_variation(shares[a], shares[b]) for i, a in enumerate(keys) for b in keys[i + 1:]), default=0.0)
    return {"shares": shares, "max_tv": worst, "stable": worst < 0.15}


# ------------------------------------------------------------------------------------------------ directive compliance

def directive_compliance(plan: DiversityPlan, directives: Directives, cfg: DiversityConfig | None = None) -> list:
    """Did the plan actually honour what brain health asked? Returns a list of violations (empty = complied). Checked from the
    plan alone so it can audit any controller, not only this one."""
    cfg = cfg or DiversityConfig()
    errs = []
    for fam, cap in directives.family_caps.items():
        for area, split in plan.family_split.items():
            if fam in split and split[fam] > max(cap, 1.0 / max(len(split), 1)) + 1e-6:
                errs.append(f"family {fam} in {area} has {split[fam]:.0%}, above its cap {cap:.0%}")
    if plan.explore_share + 1e-9 < min(directives.min_explore_share, 1.0 - cfg.floor_min):
        errs.append(f"explore share {plan.explore_share:.0%} below the requested floor {directives.min_explore_share:.0%}")
    for area in directives.starved_areas:
        if plan.shares[area] + 1e-9 < plan.floors[area]:
            errs.append(f"starved area {area} below its raised floor")
    for q in directives.reopen_questions:
        if q not in plan.reopen_questions:
            errs.append(f"reopen request {q} was dropped")
    return errs


def hard_quota(directives: Directives, base: float = 0.25) -> float:
    """Share of every area's minutes reserved for hard questions: the base rate, raised when health reports hard questions
    being abandoned. Inside-area protection against easy-area bias that the area split alone cannot give."""
    return float(min(0.6, base + (0.15 if directives.reopen_questions else 0.0)
                     + 0.1 * sum(1 for m in directives.area_multiplier.values() if m > 1.0) / max(len(AREAS), 1)))


# ------------------------------------------------------------------------------------------------ reports

def write_plan_report(plan: DiversityPlan, ctrl: DiversityController, out_dir) -> Path:
    """One JSON file per plan (audit-friendly: plan, learned exploration table, balance, audit errors, state hash) plus the
    rendered text beside it. Files are named by plan date and seed so a rerun overwrites itself, never a different plan."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"diversity_{plan.now}_{plan.seed}"
    body = {"plan": plan.to_dict(), "balance": balance_report(plan.shares), "exploration": learned_exploration_table(ctrl, plan.seed),
            "audit": audit_plan(plan, ctrl), "state_hash": state_hash(ctrl), "label": LABEL}
    (out / f"{stem}.txt").write_text(plan.render(), encoding="utf-8")
    path = out / f"{stem}.json"
    path.write_text(json.dumps(body, sort_keys=True, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------ stagnant families, information value

def stagnant_families(ctrl: DiversityController, k: int = 4, eps: float = 0.03) -> dict:
    """Families inside each area whose last k results all gained < eps bits with a non-rising trend, by research_policy's
    marginal_return_verdict. These are the 'tiny parameter families' section 37 warns about, found from the controller's own
    per-family gain series rather than from health's window, so the controller can starve them before health has to."""
    out: dict = {}
    for (area, fam), fs in sorted(ctrl.family_stats.items()):
        v = RP.marginal_return_verdict(fs.get("gains", []), k=k, eps=eps)
        if v["verdict"] == "STOP":
            out.setdefault(area, []).append(fam)
    return out


def information_value(ctrl: DiversityController, area: Area, trials: int = 10) -> float:
    """Expected entropy reduction (bits) about the area's useful-rate from `trials` more jobs (exact Beta-Binomial, from
    research_policy). An area we know little about is worth exploring even if its current estimate is modest."""
    a0, b0 = ctrl.prior()
    st = ctrl.stats[area]
    return RP.eig_beta_binomial(a0 + st.useful, b0 + max(st.trials - st.useful, 0.0), trials)


def explain_area(ctrl: DiversityController, area: Area) -> str:
    """Everything the controller believes about one area, in a sentence: evidence, rate with interval, whether it is dead,
    when it last paid, and whether exploring beats routine work there."""
    st = ctrl.stats[area]
    lo, hi = wilson(st.raw_useful, st.raw_trials)
    lift = ctrl.exploration_lift(area, 0)
    parts = [f"{area.value}: {st.raw_useful}/{st.raw_trials} useful ({lo:.0%}-{hi:.0%})"]
    parts.append("dead (floor relaxed)" if ctrl.is_dead(area) else "alive")
    parts.append(f"last useful {st.last_useful or 'never'}")
    if lift["known"]:
        parts.append(f"exploring beats routine with p={lift['p_explore_better']:.2f}")
    if st.stale_repeats:
        parts.append(f"{st.stale_repeats} stale repeats earned nothing")
    parts.append(f"{information_value(ctrl, area):.2f} bits to gain from 10 more jobs")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------ work orders

def largest_remainder(weights: Mapping[str, float], total: int) -> dict:
    """Apportion `total` integer slots by weight (Hamilton's method): each key gets floor(share x total) and the leftover slots
    go to the largest fractional remainders, ties broken by name. Deterministic, sums exactly to `total`."""
    if total < 0:
        raise ValueError("total must be >= 0")
    tot = sum(max(w, 0.0) for w in weights.values())
    if tot <= 0 or total == 0:
        return {k: 0 for k in weights}
    exact = {k: max(w, 0.0) / tot * total for k, w in weights.items()}
    base = {k: int(math.floor(v)) for k, v in exact.items()}
    left = total - sum(base.values())
    for k in sorted(weights, key=lambda k: (-(exact[k] - base[k]), k))[:left]:
        base[k] += 1
    return base


@dataclass(frozen=True)
class JobSlot:
    """One concrete unit of research the loop should run: which area, which family, exploratory or routine, hard or not."""
    area: str
    family: str
    exploratory: bool
    hard: bool
    minutes: float


def job_slots(plan: DiversityPlan, job_minutes: float = 10.0) -> list:
    """Turn a plan into work orders. Each area's minutes become whole jobs (largest remainder); within an area, jobs are split
    exploratory/routine by the learned fraction, routine jobs go to families by the family split, and the hard quota decides
    how many are aimed at hard questions. A plan that cannot be turned into whole jobs is rounded, never inflated."""
    if job_minutes <= 0:
        raise ValueError("job_minutes must be positive")
    per_area = largest_remainder(plan.shares, int(plan.budget_minutes // job_minutes))
    slots = []
    for a in AREAS:
        n = per_area[a.value]
        if n == 0:
            continue
        n_exp = int(round(n * plan.exploratory_fraction[a.value]))
        n_hard = int(round(n * plan.hard_quota))
        fam = largest_remainder(plan.family_split.get(a.value) or {"(new)": 1.0}, n - n_exp)
        rows = [(f, False) for f, c in sorted(fam.items()) for _ in range(c)] + [("(new)", True)] * n_exp
        for i, (f, ex) in enumerate(rows):
            slots.append(JobSlot(a.value, f, ex, i < n_hard, job_minutes))
    return slots


# ------------------------------------------------------------------------------------------------ one full research-brain cycle

def run_cycle(ctrl: DiversityController, ledger, outcomes: Sequence[Outcome], now, budget_minutes: float, seed: int,
              events: Sequence = (), health_cfg=None) -> tuple:
    """Health first, then allocation: measure the researcher from outcomes that matured before `now`, append the report to
    `ledger` (a brain_health.HealthLedger), and let the report's directives shape the next split. Returns (report, plan).
    This is the single call the wave-2 research loop makes each cycle."""
    from engine.research import brain_health as BH
    fresh = [o for o in outcomes if as_date(o.when) < as_date(now)]
    report = BH.step(fresh, now, health_cfg, events, ledger)
    plan = step(ctrl, now, [o for o in fresh if o.exp_id not in ctrl.seen_ids], budget_minutes, seed, report=report)
    return report, plan


# ------------------------------------------------------------------------------------------------ small budgets, revival, invariants

def job_slots_with_carry(plan: DiversityPlan, carry: Mapping[str, float] | None = None, job_minutes: float = 10.0) -> tuple:
    """When the budget is small a 2% floor is less than one job, so a per-round rounding would starve the floor areas for ever.
    Fractional jobs are therefore CARRIED: each round's exact entitlement plus last round's remainder is apportioned, and the
    new remainder is returned. Over many rounds every area receives its share. Returns (slots, new_carry)."""
    carry = dict(carry or {})
    exact = {a.value: plan.shares[a.value] * plan.budget_minutes / job_minutes + carry.get(a.value, 0.0) for a in AREAS}
    whole = largest_remainder(exact, int(sum(exact.values()) + 1e-9))
    new_carry = {a: exact[a] - whole[a] for a in exact}
    scaled = dataclasses.replace(plan, shares={a: whole[a] / max(sum(whole.values()), 1) for a in whole},
                                 budget_minutes=float(sum(whole.values()) * job_minutes))
    return job_slots(scaled, job_minutes), new_carry


def propose_hypothesis(ctrl: DiversityController, area: Area | str, family: str, config_hash: str) -> bool:
    """Offer a hypothesis for a (possibly dead) area. If this exact (area, family, config) was never run it is genuinely new:
    the area is re-opened and True is returned; otherwise nothing changes and False says 'you have already tried this'."""
    a = Area.parse(area)
    if not config_hash or (a.value, family, config_hash) in ctrl.seen_hypotheses:
        return False
    ctrl.reopen(a)
    return True


def revival_candidates(ctrl: DiversityController) -> list:
    """Areas currently written off as dead: the ones a new hypothesis could bring back. Their floor is at the minimum but not
    zero, so a proposal always has somewhere to run."""
    return [a.value for a in AREAS if ctrl.is_dead(a)]


def validate_state(ctrl: DiversityController) -> list:
    """Invariants the controller must never break. Used after loading persisted state and by the tests."""
    errs = []
    for a, st in ctrl.stats.items():
        for name in ("trials", "useful", "weighted", "bits", "minutes", "exp_trials", "exp_useful", "routine_trials", "routine_useful"):
            v = getattr(st, name)
            if v < -1e-9 or not math.isfinite(v):
                errs.append(f"{a.value}.{name} = {v}")
        if st.useful > st.trials + 1e-6:
            errs.append(f"{a.value}: more useful results than trials")
        if st.raw_useful > st.raw_trials:
            errs.append(f"{a.value}: raw useful exceeds raw trials")
        if any(v not in (0.0, 1.0) for v in st.series):
            errs.append(f"{a.value}: series holds non-binary values")
    if ctrl.total_experiments != sum(st.raw_trials for st in ctrl.stats.values()):
        errs.append("experiment count disagrees with the per-area counters")
    if len(ctrl.seen_ids) != ctrl.total_experiments:
        errs.append("seen ids disagree with the experiment count")
    for _, s in ctrl.plans:
        if abs(sum(s.values()) - 1.0) > 1e-6:
            errs.append("a stored plan does not sum to 1")
    return errs


def area_table(ctrl: DiversityController, now) -> list:
    """One structured row per area for dashboards: evidence, rate interval, dead flag, information value, days since it paid."""
    rows = []
    for a in AREAS:
        st = ctrl.stats[a]
        lo, hi = wilson(st.raw_useful, st.raw_trials)
        rows.append({"area": a.value, "trials": st.raw_trials, "useful": st.raw_useful, "rate_lo": lo, "rate_hi": hi,
                     "dead": ctrl.is_dead(a), "stale_repeats": st.stale_repeats, "new_hypotheses": st.new_hypotheses,
                     "info_bits": information_value(ctrl, a),
                     "days_since_useful": (as_date(now) - as_date(st.last_useful)).days if st.last_useful else None})
    return rows


# ------------------------------------------------------------------------------------------------ baselines and directive merging

def _baseline_shares(kind: str, ctrl: DiversityController, rng: np.random.Generator) -> dict:
    lo = {a.value: ctrl.specs[a].floor for a in AREAS}
    hi = {a.value: min(ctrl.specs[a].cap, ctrl.cfg.max_share) for a in AREAS}
    if kind == "uniform":
        raw = {a.value: 1.0 for a in AREAS}
    elif kind == "prior":
        raw = {a.value: ctrl.specs[a].prior_share for a in AREAS}
    elif kind == "greedy":                          # exploit the current best empirical rate, no exploration bonus
        raw = {a.value: math.exp(20 * (ctrl.stats[a].useful + 1) / (ctrl.stats[a].trials + 2)) for a in AREAS}
    elif kind == "random":
        raw = {a.value: float(rng.random()) + 0.05 for a in AREAS}
    else:
        raise ValueError(f"unknown baseline {kind!r}")
    return RP.project_box_simplex(raw, lo, hi)


def compare_with_baselines(true_yield: Mapping[Area, float], rounds: int, budget_minutes: float, seeds: Sequence[int],
                           cfg: DiversityConfig | None = None, job_minutes: float = 10.0) -> dict:
    """Does the adaptive controller earn more useful results than simple policies in a world with known yields? Each policy
    runs the same rounds on the same seeds; the controller's own evidence drives it, the baselines never learn (greedy learns
    but never explores). Reports mean expected useful rate over the last fifth of rounds, per policy. A planted-world check."""
    late = max(1, rounds // 5)
    res: dict = {"controller": [], "uniform": [], "prior": [], "greedy": [], "random": []}
    for sd in seeds:
        run = simulate_learning(true_yield, rounds, budget_minutes, sd, cfg, job_minutes=job_minutes)
        res["controller"].append(run["late_rate"])
        rng = np.random.default_rng(sd)
        for kind in ("uniform", "prior", "greedy", "random"):
            ctrl = DiversityController(cfg)
            vals = []
            for r in range(rounds):
                sh = _baseline_shares(kind, ctrl, rng)
                vals.append(sum(sh[a.value] * true_yield[a] for a in AREAS))
                day = _iso(as_date("2021-01-04").toordinal() + 7 * r)
                ctrl.advance(day)
                for a in AREAS:
                    jobs = int(round(sh[a.value] * budget_minutes / job_minutes))
                    for j in range(jobs):
                        ok = bool(rng.random() < true_yield[a])
                        ctrl.observe(_synthetic_outcome(r * 100000 + len(ctrl.seen_ids), _iso(as_date(day).toordinal() - 1), a,
                                                        job_minutes, ok, 0.3, False, f"{a.value[:3]}_{j % 3}"), day)
            res[kind].append(float(np.mean(vals[-late:])))
    means = {k: float(np.mean(v)) for k, v in res.items()}
    return {"mean_late_rate": means, "best_baseline": max((k for k in means if k != "controller"), key=means.get),
            "controller_beats_all": all(means["controller"] >= v for k, v in means.items() if k != "controller")}


def merge_directives(a: Directives, b: Directives) -> Directives:
    """Combine two directive sets conservatively: multipliers multiply, family caps take the tighter cap, the exploration floor
    takes the larger, question and starved-area lists are unioned in order. Lets brain health and another supervisor (for
    example the waste manager) both steer the allocator without either overriding the other."""
    mult = dict(a.area_multiplier)
    for k, v in b.area_multiplier.items():
        mult[k] = mult.get(k, 1.0) * v
    caps = dict(a.family_caps)
    for k, v in b.family_caps.items():
        caps[k] = min(caps.get(k, 1.0), v)
    return Directives(mult, caps, tuple(dict.fromkeys(a.reopen_questions + b.reopen_questions)),
                      max(a.min_explore_share, b.min_explore_share), tuple(dict.fromkeys(a.starved_areas + b.starved_areas)),
                      a.reasons + b.reasons)


def novelty_of_questions(ctrl: DiversityController, questions: Sequence[Any]) -> dict:
    """How much of the proposed work is in areas the controller has seen little of? Share of questions filed under areas whose
    effective trial count is below the median, plus the entropy of the area mix of the proposals. A question generator that
    only proposes more of the same shows up as low entropy and low novelty share."""
    if not questions:
        return {"n": 0, "novel_share": None, "area_entropy": 0.0}
    areas = [classify_question(q.source, getattr(q, "problem", None)) for q in questions]
    med = float(np.median([ctrl.stats[a].trials for a in AREAS]))
    counts = [sum(1 for x in areas if x is a) for a in AREAS]
    return {"n": len(areas), "novel_share": sum(1 for x in areas if ctrl.stats[x].trials <= med) / len(areas),
            "area_entropy": normalised_entropy(counts, len(AREAS)), "by_area": {a.value: c for a, c in zip(AREAS, counts) if c}}


def sensitivity_to_caps(ctrl: DiversityController, now, budget_minutes: float, seed: int, max_shares: Sequence[float] = (0.3, 0.45, 0.6)) -> dict:
    """How much does the plan depend on the per-area ceiling? Plans the same evidence under each `max_share` (uncommitted) and
    returns the resulting exploit share and largest area share. If the ceiling alone decides the split, the data have not."""
    out = {}
    for ms in max_shares:
        probe = controller_from_dict(controller_to_dict(ctrl))
        probe.cfg = dataclasses.replace(probe.cfg, max_share=ms)
        try:
            p = probe.allocate(now, budget_minutes, seed, commit=False)
            out[ms] = {"max_area_share": max(p.shares.values()), "exploit_share": p.exploit_share}
        except ValueError as e:
            out[ms] = {"error": str(e)}
    return out


def diversity_history(ctrl: DiversityController) -> list:
    """Normalised entropy and exploit share of every stored plan, oldest first: the plan-level diversity curve. A curve that
    only ever falls is the controller converging on a monoculture even though each single plan passed its floors."""
    return [{"now": n, **{k: v for k, v in balance_report(s).items() if k in ("normalised_entropy", "exploit_share", "max_share")}}
            for n, s in ctrl.plans]


def diversity_trend(ctrl: DiversityController, last: int = 8) -> dict:
    """FALLING when the entropy slope over the last plans is negative beyond noise (>0.01 per plan); INSUFFICIENT under 4 plans."""
    h = [r["normalised_entropy"] for r in diversity_history(ctrl)][-last:]
    if len(h) < 4:
        return {"verdict": "INSUFFICIENT", "n": len(h)}
    slope = float(np.polyfit(np.arange(len(h)), h, 1)[0])
    return {"verdict": "FALLING" if slope < -0.01 else "RISING" if slope > 0.01 else "STEADY", "slope": slope, "n": len(h)}


def plan_vs_realised(plan: DiversityPlan, outcomes: Sequence[Outcome]) -> dict:
    """Did the loop follow the plan? Per-area planned share against the share of minutes actually spent, and the total-variation
    gap. Above `adherence_alarm` the plan was advice the loop ignored; the areas furthest off are named."""
    got = realised_shares(outcomes)
    gap = total_variation(plan.shares, got)
    worst = sorted(AREAS, key=lambda a: -abs(plan.shares[a.value] - got[a.value]))[:3]
    return {"tv": gap, "ignored": gap > DiversityConfig().adherence_alarm,
            "worst": [(a.value, plan.shares[a.value], got[a.value]) for a in worst]}


def adapted_specs(ctrl: DiversityController, blend: float = 0.5) -> dict:
    """Section 38: 'the percentages should be adaptive'. Re-derive the prior shares from what has been learned: each area's
    posterior useful-rate per minute, blended with its current prior by `blend`, then projected into the floors and caps so the
    result is again a valid specification. Use it to re-seed a fresh controller after a long run; the live one adapts already."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    rates = {}
    for a in AREAS:
        st = ctrl.stats[a]
        a0, b0 = ctrl.prior()                       # unobserved areas fall back to the pooled rate, not to an optimistic 1/cost
        p_hat = (st.useful + 2.0 * a0 / (a0 + b0)) / (st.trials + 2.0)
        per_min = p_hat / ((st.minutes + 2.0 * ctrl.cfg.cost_prior_minutes) / (st.trials + 2.0))
        rates[a.value] = per_min
    tot = sum(rates.values())
    raw = {a.value: (1 - blend) * ctrl.specs[a].prior_share + blend * rates[a.value] / tot for a in AREAS}
    shares = RP.project_box_simplex(raw, {a.value: ctrl.specs[a].floor for a in AREAS}, {a.value: ctrl.specs[a].cap for a in AREAS})
    new = {a: dataclasses.replace(ctrl.specs[a], prior_share=shares[a.value]) for a in AREAS}
    errs = validate_specs(new, ctrl.cfg)
    if errs:
        raise ValueError("adapted specs invalid: " + "; ".join(errs))
    return new


def explore_exploit_audit(plan: DiversityPlan, outcomes: Sequence[Outcome]) -> dict:
    """Planned versus realised exploit share, and the exploratory fraction actually run in each area. Exploration that is
    planned but never executed (exploratory flags all False) is exploration in name only."""
    got = realised_shares(outcomes)
    realised_exploit = sum(got[a.value] for a in EXPLOIT_AREAS)
    per_area = {}
    for a in AREAS:
        os_ = [o for o in outcomes if o.area == a.value]
        per_area[a.value] = {"planned": plan.exploratory_fraction[a.value],
                             "realised": (sum(1 for o in os_ if o.exploratory) / len(os_)) if os_ else None}
    hollow = [a for a, v in per_area.items() if v["realised"] is not None and v["planned"] >= 0.2 and v["realised"] == 0.0
              and sum(1 for o in outcomes if o.area == a) >= 10]
    return {"planned_exploit": plan.exploit_share, "realised_exploit": realised_exploit,
            "exploit_overrun": realised_exploit - plan.exploit_share, "per_area": per_area, "hollow_exploration": hollow}


# ------------------------------------------------------------------------------------------------ persistence

def controller_to_dict(ctrl: DiversityController) -> dict:
    return {"cfg": dataclasses.asdict(ctrl.cfg),
            "specs": {a.value: {"prior_share": s.prior_share, "floor": s.floor, "cap": s.cap, "note": s.note}
                      for a, s in ctrl.specs.items()},
            "stats": {a.value: st.to_dict() for a, st in ctrl.stats.items()}, "seen_ids": sorted(ctrl.seen_ids),
            "seen_hypotheses": sorted(list(k) for k in ctrl.seen_hypotheses),
            "family_stats": [[a, f, v] for (a, f), v in sorted(ctrl.family_stats.items())], "clock": ctrl.clock,
            "plans": ctrl.plans, "spent_since_plan": ctrl.spent_since_plan,
            "reopen_marks": {a.value: list(m) for a, m in ctrl.reopen_marks.items()}, "total_experiments": ctrl.total_experiments,
            "rounds": ctrl.rounds}


def controller_from_dict(d: Mapping) -> DiversityController:
    specs = {Area.parse(a): AreaSpec(Area.parse(a), v["prior_share"], v["floor"], v["cap"], v.get("note", ""))
             for a, v in d["specs"].items()}
    ctrl = DiversityController(DiversityConfig(**d["cfg"]), specs)
    ctrl.stats = {Area.parse(a): AreaStats.from_dict(v) for a, v in d["stats"].items()}
    ctrl.seen_ids = set(d["seen_ids"])
    ctrl.seen_hypotheses = {tuple(k) for k in d["seen_hypotheses"]}
    ctrl.family_stats = {(a, f): dict(v) for a, f, v in d["family_stats"]}
    ctrl.clock = d["clock"]
    ctrl.plans = [(n, dict(s)) for n, s in d["plans"]]
    ctrl.spent_since_plan = dict(d["spent_since_plan"])
    ctrl.reopen_marks = {Area.parse(a): tuple(m) for a, m in d["reopen_marks"].items()}
    ctrl.total_experiments = d["total_experiments"]
    ctrl.rounds = list(d.get("rounds", []))
    return ctrl


def save_controller(ctrl: DiversityController, path) -> Path:
    """Atomic write (temp file then replace) so a crash mid-save cannot leave a half-written state."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(controller_to_dict(ctrl), sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def load_controller(path) -> DiversityController:
    p = Path(path)
    return controller_from_dict(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else DiversityController()


def state_hash(ctrl: DiversityController) -> str:
    return stable_hash(controller_to_dict(ctrl), 12)


# ------------------------------------------------------------------------------------------------ planted-world simulation

def _synthetic_outcome(idx: int, when: str, area: Area, minutes: float, success: bool, gain: float, exploratory: bool,
                       family: str) -> Outcome:
    return Outcome(f"S{idx:07d}", when, area.value, family, minutes, success, gain if success else 0.0, 0.5,
                   exploratory=exploratory, config_hash=stable_hash([area.value, family, idx], 6))


def simulate_learning(true_yield: Mapping[Area, float], rounds: int, budget_minutes: float, seed: int,
                      cfg: DiversityConfig | None = None, shift: tuple | None = None, start: str = "2021-01-04",
                      exploratory_yield: Mapping[Area, float] | None = None, job_minutes: float = 10.0) -> dict:
    """A world where each area's true useful-rate is known. Each round: plan, run the planned minutes as Bernoulli jobs,
    feed the outcomes back. `shift=(round, area, new_rate)` changes one area's rate mid-run (drift); `exploratory_yield`
    gives exploratory jobs a different rate from routine ones. Returns per-round shares and expected-value regret against
    the best feasible split and against uniform - the numbers a planted-world test asserts on."""
    rng = np.random.default_rng(seed)
    ctrl = DiversityController(cfg)
    truth = dict(true_yield)
    day0 = as_date(start).toordinal()
    trace, idx = [], 0
    pending: list = []
    for r in range(rounds):
        if shift is not None and r == shift[0]:
            truth[shift[1]] = shift[2]
        now = _iso(day0 + 7 * r)
        plan = step(ctrl, now, pending, budget_minutes, seed * 1009 + r)
        pending = []
        for a in AREAS:
            jobs = int(round(plan.minutes[a.value] / job_minutes))
            for j in range(jobs):
                explore = bool(rng.random() < plan.exploratory_fraction[a.value])
                rate = (exploratory_yield or {}).get(a, truth[a]) if explore else truth[a]
                ok = bool(rng.random() < rate)
                fam = f"{a.value[:3]}_{j % 3}"
                pending.append(_synthetic_outcome(idx, _iso(day0 + 7 * r + 1 + (j % 5)), a, job_minutes, ok,
                                                  float(rng.exponential(0.3)) if ok else 0.0, explore, fam))
                idx += 1
        ev = sum(plan.shares[a.value] * truth[a] for a in AREAS)
        trace.append({"round": r, "shares": dict(plan.shares), "expected_useful_rate": ev})
    hi = {a: min(ctrl.specs[a].cap, ctrl.cfg.max_share) for a in AREAS}
    lo = {a: ctrl.specs[a].floor for a in AREAS}
    oracle = RP.project_box_simplex({a.value: math.exp(30 * truth[a]) for a in AREAS}, {a.value: lo[a] for a in AREAS},
                                    {a.value: hi[a] for a in AREAS})
    best = sum(oracle[a.value] * truth[a] for a in AREAS)
    uniform = float(np.mean([truth[a] for a in AREAS]))
    last = trace[-max(1, rounds // 5):]
    got = float(np.mean([t["expected_useful_rate"] for t in last]))
    return {"trace": trace, "final_shares": trace[-1]["shares"], "oracle_rate": best, "uniform_rate": uniform, "late_rate": got,
            "regret_vs_oracle": best - got, "gain_over_uniform": got - uniform, "controller": ctrl}


def _iso(ordinal: int) -> str:
    import datetime as _dt
    return _dt.date.fromordinal(ordinal).isoformat()


def farming_probe(seed: int = 0, rounds: int = 12, budget_minutes: float = 600.0) -> dict:
    """Section 43 trap: can the allocator be paid for volume? One area reports 'successes' that are all duplicates or
    memorised. If the controller rewarded them, that area's share would climb above its honest twin's. Returns both shares."""
    ctrl = DiversityController()
    rng = np.random.default_rng(seed)
    day0 = as_date("2021-01-04").toordinal()
    idx = 0
    for r in range(rounds):
        now = _iso(day0 + 7 * r + 7)
        batch = []
        for a in AREAS:
            for j in range(10):
                honest = rng.random() < 0.2
                if a is Area.RISK:                                   # the farmer: high 'success', none of it useful
                    o = Outcome(f"F{idx:07d}", _iso(day0 + 7 * r + 1), a.value, "RIS_0", 10.0, bool(rng.random() < 0.9), 0.4, 0.5,
                                duplicate_of="E0" if rng.random() < 0.7 else "", memorised=bool(rng.random() < 0.7) or None)
                else:
                    o = Outcome(f"F{idx:07d}", _iso(day0 + 7 * r + 1), a.value, f"{a.value[:3]}_0", 10.0, honest, 0.4 if honest else 0.0, 0.5)
                batch.append(o)
                idx += 1
        plan = step(ctrl, now, batch, budget_minutes, seed + r)
    return {"farmer_share": plan.shares[Area.RISK.value], "farmer_prior": ctrl.specs[Area.RISK].prior_share,
            "farmer_rate": ctrl.area_rate(Area.RISK)["mean"], "honest_rate": ctrl.area_rate(Area.VOLATILITY)["mean"]}


def self_check(seed: int = 0) -> dict:
    """The planted-world checks in one call: a strong area gains share, floors hold, drift is followed, farming is not paid."""
    truth = {a: 0.05 for a in AREAS}
    truth[Area.VOLATILITY] = 0.5
    base = simulate_learning(truth, 40, 600.0, seed)
    drift = simulate_learning(truth, 60, 600.0, seed, shift=(30, Area.VOLATILITY, 0.02))
    farm = farming_probe(seed)
    ctrl = base["controller"]
    return {"label": LABEL, "best_area_share": base["final_shares"][Area.VOLATILITY.value],
            "min_share": min(base["final_shares"].values()), "regret_vs_oracle": base["regret_vs_oracle"],
            "gain_over_uniform": base["gain_over_uniform"],
            "share_after_drift": drift["final_shares"][Area.VOLATILITY.value], "farming": farm,
            "audit": audit_plan(step(ctrl, _iso(as_date("2022-01-03").toordinal()), (), 600.0, seed), ctrl)}
