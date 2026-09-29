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
    dead_min_trials: int = 15
    dead_rate: float = 0.03
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
        fs = self.family_stats.setdefault((area.value, o.family or "(none)"), {"trials": 0.0, "useful": 0.0, "bits": 0.0, "minutes": 0.0})
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
            score = (p * val / cost + 1e-4) * (1.0 + self.cfg.ucb_weight * ucb)
            out[a] = score ** (1.0 / self.cfg.softmax_temperature)
        return out

    def allocate(self, now, budget_minutes: float, seed: int, directives: Directives | None = None) -> DiversityPlan:
        if budget_minutes <= 0 or not math.isfinite(budget_minutes):
            raise ValueError("budget_minutes must be positive and finite")
        directives = directives or Directives()
        bad = directives.validate()
        if bad:
            raise ValueError("invalid directives: " + "; ".join(bad))
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
        explore = 1.0 - sum(shares[a.value] for a in EXPLOIT_AREAS)
        fam = {a.value: self.family_split(a, directives) for a in AREAS}
        prev = self.adherence()
        self.spent_since_plan = {a.value: 0.0 for a in AREAS}
        plan = DiversityPlan(
            str(now), float(budget_minutes), int(seed), shares, {a: s * budget_minutes for a, s in shares.items()},
            {a.value: lo[a] for a in AREAS}, {a.value: hi[a] for a in AREAS},
            {a.value: self.exploratory_fraction(a, seed + 1 + i) for i, a in enumerate(AREAS)}, explore, 1.0 - explore, fam,
            directives.reopen_questions, tuple(rationale), prev, 0, 0, (), current_code_hash())
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


# ------------------------------------------------------------------------------------------------ persistence

def controller_to_dict(ctrl: DiversityController) -> dict:
    return {"cfg": dataclasses.asdict(ctrl.cfg),
            "specs": {a.value: {"prior_share": s.prior_share, "floor": s.floor, "cap": s.cap, "note": s.note}
                      for a, s in ctrl.specs.items()},
            "stats": {a.value: st.to_dict() for a, st in ctrl.stats.items()}, "seen_ids": sorted(ctrl.seen_ids),
            "seen_hypotheses": sorted(list(k) for k in ctrl.seen_hypotheses),
            "family_stats": [[a, f, v] for (a, f), v in sorted(ctrl.family_stats.items())], "clock": ctrl.clock,
            "plans": ctrl.plans, "spent_since_plan": ctrl.spent_since_plan,
            "reopen_marks": {a.value: list(m) for a, m in ctrl.reopen_marks.items()}, "total_experiments": ctrl.total_experiments}


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
        now = _iso(day0 + 7 * r)
        batch = []
        for a in AREAS:
            for j in range(10):
                honest = rng.random() < 0.2
                if a is Area.RISK:                                   # the farmer: high 'success', none of it useful
                    o = Outcome(f"F{idx:07d}", _iso(day0 + 7 * r + 1), a.value, "RIS_0", 10.0, bool(rng.random() < 0.9), 0.4, 0.5,
                                duplicate_of="E0" if rng.random() < 0.5 else "", memorised=bool(rng.random() < 0.5) or None)
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
