"""Planted-truth world (contract C62 sections 62 tests 1-12 and 63; checklist I06-I15, I19, I20). STATUS: IMPLEMENTED — NOT VALIDATED.

A seeded synthetic market whose every pattern is KNOWN, plus an EXACT truth ledger answering, for every (item, week): is it
active, what is the true effect, and is its condition in state. The learner is never shown the ledger; scoring helpers turn
its claims into recall / precision / false-discovery by kind, degradation-detection delay and condition-recovery accuracy.

Planted kinds: strong, weak, negative, pair, regime (valid only while an observable m_ variable is in a state), decaying,
conditional (works, better when C), unless (works except when X), noise-only, hallucination (discovery split only, by
construction), reversal (sign flips at a known week), duplicate (redundant copy of a real pattern), hidden-in-evaluation
(present in training, absent in evaluation - using it must hurt).  Also: ticker / date / year re-identification (same world
under new identities), a year-swap generator (same situation classes, different era) and a future-leak canary column.

Builds on engine.planted (quintile convention, canonical keys, key parser) so a planted "f0 q4" is the miner's f0 q4.
Every draw comes from an explicit seed; nothing here reads a clock.  Real data stays the final authority (section 63): a world
is a calibration instrument, never a target to tune against."""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import planted as P
from engine.learning.core import FirewallBreach, current_code_hash, stable_hash

# ---------------------------------------------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------------------------------------------
STRONG, WEAK, NEGATIVE, PAIR, REGIME = "strong", "weak", "negative", "pair", "regime"
DECAYING, CONDITIONAL, UNLESS, NOISE = "decaying", "conditional", "unless", "noise"
HALLUCINATION, REVERSAL, DUPLICATE, HIDDEN_EVAL = "hallucination", "reversal", "duplicate", "hidden_eval"
KINDS = (STRONG, WEAK, NEGATIVE, PAIR, REGIME, DECAYING, CONDITIONAL, UNLESS, NOISE, HALLUCINATION, REVERSAL,
         DUPLICATE, HIDDEN_EVAL)
NEVER_HOLD = (NOISE, DUPLICATE)            # kinds a correct learner never holds as knowledge in their own right
CONDITIONED = (REGIME, CONDITIONAL, UNLESS)  # kinds whose truth includes a condition the learner must recover
CHANGING = (DECAYING, REVERSAL, HIDDEN_EVAL, HALLUCINATION)  # kinds with a known change point

PROFILES = ("const", "decay", "window", "flip")
M_COLS = ("m_vix", "m_breadth", "m_term")
CANARY_PREFIX = "canary_"
CANARY_COL = "canary_future_ret"
MIN_ACTIVE = 0.001                          # an effect under 0.1%/week is "no real effect" (same bar as engine.planted)
PHASES = ("discovery", "confirmation", "evaluation")


# ---------------------------------------------------------------------------------------------------------------
# specification records
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gate:
    """The observable state a conditioned item depends on.  kind 'm': lo < m_col <= hi on the raw market-context value.
    kind 'q': the cross-sectional quintile of feature `col` is in `qs`."""
    kind: str
    col: str
    lo: float = -np.inf
    hi: float = np.inf
    qs: tuple = ()

    def validate(self) -> list[str]:
        errs = []
        if self.kind == "m":
            if self.col not in M_COLS:
                errs.append(f"gate: unknown market column {self.col}")
            if not self.lo < self.hi:
                errs.append(f"gate: empty interval ({self.lo},{self.hi}]")
        elif self.kind == "q":
            if not self.qs or any(q not in range(5) for q in self.qs):
                errs.append(f"gate: quintile set {self.qs} not within 0..4")
        else:
            errs.append(f"gate: kind {self.kind!r} not m|q")
        return errs

    def mask(self, X: pd.DataFrame, Q: pd.DataFrame) -> np.ndarray:
        if self.kind == "m":
            v = X[self.col].to_numpy()
            return (v > self.lo) & (v <= self.hi)
        return Q[self.col].isin(self.qs).to_numpy()

    def spec(self) -> tuple:
        """The condition in the format condition_recovery() accepts from a learner."""
        return ("m", self.col, self.lo, self.hi) if self.kind == "m" else ("q", self.col, tuple(self.qs))


@dataclass(frozen=True)
class Item:
    """One planted item.  On the rows matching `conds` (all quintiles equal) and not matching `unless`, the planted weekly
    return is  multiplier(week) * (on_effect where the gate holds, off_effect elsewhere)."""
    item_id: str
    kind: str
    conds: tuple                                   # ((feature, quintile), ...)
    effect: float = 0.0                            # in-state effect per week
    off_effect: float = 0.0                        # effect on key rows where the gate does NOT hold
    unless: tuple | None = None                    # (feature, quintile) that cancels the effect
    gate: Gate | None = None
    profile: str = "const"
    params: tuple = ()                             # decay (start, end) | window (a, b) | flip (t0)  - in week indices
    redundant_with: str = ""                       # duplicate: item_id of the real pattern it copies
    note: str = ""

    @property
    def key(self):
        return P.canon_key(self.conds, self.unless)

    @property
    def key_named(self) -> str:
        s = " & ".join(f"{f} q{q}" for f, q in sorted(self.conds))
        return s + (f" unless {self.unless[0]} q{self.unless[1]}" if self.unless is not None else "")

    @property
    def features(self) -> tuple:
        fs = [f for f, _ in self.conds] + ([self.unless[0]] if self.unless else []) + ([self.gate.col] if self.gate and self.gate.kind == "q" else [])
        return tuple(dict.fromkeys(fs))

    def multipliers(self, n_weeks: int) -> np.ndarray:
        w = np.arange(n_weeks, dtype=float)
        if self.profile == "const":
            return np.ones(n_weeks)
        if self.profile == "decay":                # full until start, linear to zero at end, dead afterwards
            s, e = self.params
            return np.clip((e - w) / max(e - s, 1e-9), 0.0, 1.0)
        if self.profile == "window":               # exists only in [a, b)
            a, b = self.params
            return ((w >= a) & (w < b)).astype(float)
        if self.profile == "flip":                 # +1 before t0, -1 from t0
            return np.where(w < self.params[0], 1.0, -1.0)
        raise ValueError(f"profile {self.profile!r}")

    def change_week(self, n_weeks: int) -> int | None:
        """First week where the item is no longer what it was: multiplier at or below half of its starting value
        (decay midpoint, window close) or of opposite sign (flip).  None for an item that never changes."""
        m = self.multipliers(n_weeks)
        if abs(m[0]) < 1e-12:
            nz = np.nonzero(np.abs(m) > 1e-12)[0]
            return int(nz[0]) if len(nz) else None
        hit = np.nonzero(m * np.sign(m[0]) <= 0.5 * abs(m[0]) + 1e-12)[0]
        return int(hit[0]) if len(hit) else None

    def validate(self, n_weeks: int) -> list[str]:
        errs = []
        if self.kind not in KINDS:
            errs.append(f"{self.item_id}: unknown kind {self.kind!r}")
        if self.profile not in PROFILES:
            errs.append(f"{self.item_id}: unknown profile {self.profile!r}")
        elif self.profile in ("decay", "window") and (len(self.params) != 2 or not 0 <= self.params[0] < self.params[1]):
            errs.append(f"{self.item_id}: {self.profile} params {self.params} must be 0 <= a < b")
        elif self.profile == "flip" and (len(self.params) != 1 or not 0 < self.params[0] < n_weeks):
            errs.append(f"{self.item_id}: flip week {self.params} outside the sample")
        elif self.profile == "const" and self.params:
            errs.append(f"{self.item_id}: const takes no params")
        if not self.conds:
            errs.append(f"{self.item_id}: no conditions")
        for f, q in self.conds + ((self.unless,) if self.unless else ()):
            if q not in range(5):
                errs.append(f"{self.item_id}: quintile {q} for {f} not within 0..4")
        if self.gate is not None:
            errs += [f"{self.item_id}: {e}" for e in self.gate.validate()]
        if self.kind in (NOISE, DUPLICATE) and (self.effect or self.off_effect):
            errs.append(f"{self.item_id}: {self.kind} must inject no effect")
        if self.kind not in (NOISE, DUPLICATE) and abs(self.effect) < MIN_ACTIVE and abs(self.off_effect) < MIN_ACTIVE:
            errs.append(f"{self.item_id}: effect below the reporting floor {MIN_ACTIVE}")
        if self.kind == DUPLICATE and not self.redundant_with:
            errs.append(f"{self.item_id}: duplicate needs redundant_with")
        if self.kind in CONDITIONED and self.kind != UNLESS and self.gate is None:
            errs.append(f"{self.item_id}: {self.kind} needs a gate")
        if self.kind == UNLESS and self.unless is None:
            errs.append(f"{self.item_id}: unless item needs an unless condition")
        if self.kind == PAIR and len(self.conds) < 2:
            errs.append(f"{self.item_id}: pair needs two conditions")
        return errs


@dataclass(frozen=True)
class WorldSpec:
    name: str
    items: tuple
    weeks: int = 156
    stocks: int = 100
    n_feat: int = 18
    dup_features: tuple = ()                       # ((copy_feature, source_feature, correlation), ...)
    discovery_end: int = 78                        # weeks [0, discovery_end) are the discovery split
    confirm_end: int = 117                         # [discovery_end, confirm_end) confirmation; the rest evaluation
    start: str = "2009-01-02"                      # a Friday; weeks are Fridays
    tail_df: float = 4.0
    noise_sd: float = 0.05
    week_sd: float = 0.02
    sector_sd: float = 0.01
    n_sectors: int = 8
    persistence: float = 0.7
    regime_persistence: float = 0.9
    canary_noise: float = 0.1                      # canary = forward return + this much of its own sd of noise

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.discovery_end < self.confirm_end < self.weeks:
            errs.append(f"splits must satisfy 0 < {self.discovery_end} < {self.confirm_end} < {self.weeks}")
        if self.stocks < 25:
            errs.append("stocks < 25: quintiles need at least five names each")
        ids = [i.item_id for i in self.items]
        if len(set(ids)) != len(ids):
            errs.append("duplicate item ids")
        keys = [i.key for i in self.items]
        if len(set(keys)) != len(keys):
            errs.append("two items share a condition key: scoring would be ambiguous")
        feats = {f"f{i}" for i in range(self.n_feat)}
        for it in self.items:
            errs += it.validate(self.weeks)
            errs += [f"{it.item_id}: unknown feature {f}" for f in it.features if f not in feats]
            if it.redundant_with and it.redundant_with not in ids:
                errs.append(f"{it.item_id}: redundant_with {it.redundant_with!r} is not an item")
        for c, s, r in self.dup_features:
            if c not in feats or s not in feats or not 0 < r < 1:
                errs.append(f"dup feature ({c},{s},{r}) invalid")
        try:
            if pd.Timestamp(self.start).weekday() != 4:
                errs.append("start must be a Friday")
        except Exception:
            errs.append(f"start {self.start!r} unparsable")
        return errs

    def check(self) -> "WorldSpec":
        errs = self.validate()
        if errs:
            raise ValueError("invalid WorldSpec: " + "; ".join(errs))
        return self

    def item(self, item_id: str) -> Item:
        for it in self.items:
            if it.item_id == item_id:
                return it
        raise KeyError(item_id)

    def phase_bounds(self) -> dict:
        return {"discovery": (0, self.discovery_end), "confirmation": (self.discovery_end, self.confirm_end),
                "evaluation": (self.confirm_end, self.weeks)}

    def phase_of(self, week: int) -> str:
        for name, (a, b) in self.phase_bounds().items():
            if a <= week < b:
                return name
        raise IndexError(f"week {week} outside 0..{self.weeks - 1}")

    def situation_classes(self) -> tuple:
        """Era-independent description: what kinds of situation exist, not which dates they fall on."""
        return tuple(sorted((it.kind, it.key_named, None if it.gate is None else it.gate.spec(), it.profile) for it in self.items))


# ---------------------------------------------------------------------------------------------------------------
# spec builders
# ---------------------------------------------------------------------------------------------------------------
def standard_spec(weeks: int = 156, stocks: int = 100, scale: float = 1.0, name: str = "standard") -> WorldSpec:
    """Every planted kind at once.  `scale` multiplies every effect (tests use >1 for small samples).  Change points are laid
    out so each split has a job: hallucination dies at the end of discovery, the reversal lands inside confirmation, the
    decayer is half-dead by the end of discovery, the hidden item vanishes exactly when evaluation begins."""
    d_end, c_end = int(weeks * 0.5), int(weeks * 0.75)
    e = lambda x: x * scale
    items = (
        Item("strong", STRONG, (("f0", 4),), e(0.012)),
        Item("weak", WEAK, (("f1", 4),), e(0.004)),
        Item("negative", NEGATIVE, (("f2", 4),), e(-0.008)),
        Item("pair", PAIR, (("f3", 4), ("f4", 0)), e(0.020)),
        Item("regime", REGIME, (("f5", 4),), e(0.010), gate=Gate("m", "m_vix", 0.0, np.inf)),
        Item("decaying", DECAYING, (("f6", 4),), e(0.010), profile="decay", params=(int(weeks * 0.3), int(weeks * 0.7))),
        Item("conditional", CONDITIONAL, (("f7", 4),), e(0.014), off_effect=e(0.004), gate=Gate("q", "f8", qs=(3, 4))),
        Item("unless", UNLESS, (("f9", 4), ("f10", 4)), e(0.020), unless=("f11", 4)),
        Item("hallucination", HALLUCINATION, (("f12", 0),), e(0.012), profile="window", params=(0, d_end)),
        Item("reversal", REVERSAL, (("f13", 4),), e(0.010), profile="flip", params=(int(weeks * 0.6),)),
        Item("duplicate", DUPLICATE, (("f14", 4),), redundant_with="strong", note="f14 is a 0.97-correlated copy of f0"),
        Item("hidden", HIDDEN_EVAL, (("f15", 0),), e(0.010), profile="window", params=(0, c_end)),
        Item("noise_a", NOISE, (("f16", 4),)),
        Item("noise_b", NOISE, (("f17", 0),)),
    )
    return WorldSpec(name, items, weeks=weeks, stocks=stocks, n_feat=18, dup_features=(("f14", "f0", 0.97),),
                     discovery_end=d_end, confirm_end=c_end).check()


def noise_only_spec(weeks: int = 104, stocks: int = 80, n_feat: int = 8, name: str = "noise_only") -> WorldSpec:
    """No planted signal at all: the null world.  Two noise items exist only so the ledger has something to be silent about."""
    items = (Item("noise_a", NOISE, (("f0", 4),)), Item("noise_b", NOISE, (("f1", 0),)))
    return WorldSpec(name, items, weeks=weeks, stocks=stocks, n_feat=n_feat, discovery_end=weeks // 2,
                     confirm_end=(weeks * 3) // 4).check()


def multi_year_spec(years: int = 5, stocks: int = 40, noise: bool = False, weeks_per_year: int = 52, decay_years: tuple = (2, 3),
                    scale: float = 1.0, name: str | None = None) -> WorldSpec:
    """F10 (C69 W-07): the acceptance world long enough for the learning-claim evidence.  scorecard_for_learner's forward-YEAR
    folds need two earlier training years per held-out year and its stability check three held-out years, so a claim can only be
    judged from year 5 on; `years` < 4 is refused.  Truths that persist through every year: a strong long (f0 q4), a strong short
    (f1 q4) and a regime-gated long (f2 q4, only while m_vix > 0).  One truth that decays: f5 q4 at full strength until year
    decay_years[0], linearly to zero by year decay_years[1], dead after (a degrade there is CORRECT).  Two noise cells (f3 q4,
    f4 q0).  `noise=True` is the null world of the same length: the two noise cells only.  Six features, the miniature's layout."""
    if years < 4:
        raise ValueError("a multi-year acceptance world needs >= 4 years (forward-year folds need 2 training years per test year)")
    a, b = decay_years
    if not 0 <= a < b <= years:
        raise ValueError(f"decay_years {decay_years} must satisfy 0 <= start < end <= {years}")
    W = years * weeks_per_year
    e = lambda x: x * scale
    nz = (Item("noise_a", NOISE, (("f3", 4),)), Item("noise_b", NOISE, (("f4", 0),)))
    items = nz if noise else (
        Item("strong", STRONG, (("f0", 4),), e(0.02)), Item("negative", NEGATIVE, (("f1", 4),), e(-0.015)),
        Item("regime", REGIME, (("f2", 4),), e(0.02), gate=Gate("m", "m_vix", 0.0, np.inf)),
        Item("decaying", DECAYING, (("f5", 4),), e(0.02), profile="decay", params=(a * weeks_per_year, b * weeks_per_year),
             note=f"full until year {a}, dead from year {b}")) + nz
    return WorldSpec(name or ("multi_year_null" if noise else "multi_year"), items, weeks=W, stocks=stocks, n_feat=6,
                     discovery_end=W // 2, confirm_end=(W * 3) // 4).check()


def live_for_degrade(spec: WorldSpec, item_id: str, week: int) -> bool:
    """Was a DEGRADE of this item at `week` wrong?  True while the item is still what it was (before its change week: the decay
    midpoint, a window's close, a flip); False for noise and duplicates (nothing to lose) and from the change week on (the item really
    weakened, so the degrade was right).  A regime item out of state is still live: its in-state effect exists."""
    it = spec.item(item_id)
    if it.kind in NEVER_HOLD:
        return False
    cw = it.change_week(spec.weeks)
    return cw is None or int(week) < cw


def single_item_spec(item: Item, weeks: int = 104, stocks: int = 80, n_feat: int = 8, **kw) -> WorldSpec:
    """One planted item in an otherwise silent world (power curves, per-kind oracle tests)."""
    kw.setdefault("discovery_end", weeks // 2)
    kw.setdefault("confirm_end", (weeks * 3) // 4)
    return WorldSpec(f"single_{item.item_id}", (item,), weeks=weeks, stocks=stocks, n_feat=n_feat, **kw).check()


# ---------------------------------------------------------------------------------------------------------------
# truth ledger
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerEntry:
    item_id: str
    week: int
    date: str
    phase: str
    active: bool
    true_effect: float          # mean planted effect per affected row this week (before per-week demeaning)
    on_effect: float
    off_effect: float
    condition_state: bool | None   # None: the item has no condition; True: at least some affected rows are in state
    gate_frac: float            # share of the item's affected rows currently in state
    n_rows: int


class TruthLedger:
    """Exact answers, per (item, week).  Built from the realised panel, so gate fractions and row counts are what the
    learner actually faced, not what the spec hoped for."""

    def __init__(self, spec: WorldSpec, frame: pd.DataFrame, dates: Sequence):
        self.spec = spec
        self.frame = frame.reset_index(drop=True)
        self.dates = tuple(pd.Timestamp(d) for d in dates)
        self._idx = {(r.item_id, int(r.week)): i for i, r in enumerate(self.frame.itertuples(index=False))}

    # ------------------------------------------------------------ queries
    def entry(self, item_id: str, week: int) -> LedgerEntry:
        i = self._idx.get((item_id, int(week)))
        if i is None:
            raise KeyError(f"no ledger row for ({item_id}, week {week})")
        r = self.frame.iloc[i]
        cs = None if r["condition_state"] is None or (isinstance(r["condition_state"], float) and np.isnan(r["condition_state"])) else bool(r["condition_state"])
        return LedgerEntry(str(r["item_id"]), int(r["week"]), str(r["date"]), str(r["phase"]), bool(r["active"]),
                           float(r["true_effect"]), float(r["on_effect"]), float(r["off_effect"]), cs, float(r["gate_frac"]),
                           int(r["n_rows"]))

    def active(self, item_id: str, week: int) -> bool:
        return self.entry(item_id, week).active

    def effect(self, item_id: str, week: int) -> float:
        return self.entry(item_id, week).true_effect

    def condition_state(self, item_id: str, week: int):
        return self.entry(item_id, week).condition_state

    def phase(self, week: int) -> str:
        return self.spec.phase_of(week)

    def series(self, item_id: str) -> pd.DataFrame:
        return self.frame[self.frame["item_id"] == item_id].sort_values("week")

    def is_live(self, item_id: str, week: int, lookback: int = 13) -> bool:
        """Truly usable at `week`: the item's profile carried a non-trivial effect at some point in (week-lookback, week].
        A regime item that is momentarily out of state is still live knowledge (its in-state effect exists); a dead
        hallucination, a decayed pattern or a hidden item is not."""
        s = self.series(item_id)
        w = s[(s["week"] <= week) & (s["week"] > week - lookback)]
        return bool(len(w) and ((w["on_effect"].abs() >= MIN_ACTIVE) | (w["off_effect"].abs() >= MIN_ACTIVE)).any())

    def should_hold(self, week: int, lookback: int = 13) -> list[str]:
        """Items a perfect learner would hold as knowledge at `week` (using only what is true by then)."""
        return [it.item_id for it in self.spec.items if it.kind not in NEVER_HOLD and self.is_live(it.item_id, week, lookback)]

    def should_not_hold(self, week: int, lookback: int = 13) -> list[str]:
        held = set(self.should_hold(week, lookback))
        return [it.item_id for it in self.spec.items if it.item_id not in held]

    def sign_at(self, item_id: str, week: int) -> int:
        e = self.effect(item_id, week)
        return int(np.sign(e)) if abs(e) >= MIN_ACTIVE else 0

    def change_week(self, item_id: str) -> int | None:
        return self.spec.item(item_id).change_week(self.spec.weeks)

    def change_date(self, item_id: str):
        w = self.change_week(item_id)
        return None if w is None else self.dates[w]

    def traps_at(self, week: int, lookback: int = 13) -> dict:
        """item_id -> kind for every item a learner must NOT be holding at `week` (noise, redundant, dead, hidden...)."""
        return {i: self.spec.item(i).kind for i in self.should_not_hold(week, lookback)}

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()

    def content_hash(self) -> str:
        return stable_hash({"spec": self.spec.name, "rows": self.frame.round(9).to_dict("list")})

    def check(self) -> list[str]:
        """Internal consistency of the ledger against its own spec; an empty list means consistent."""
        errs = []
        exp = len(self.spec.items) * self.spec.weeks
        if len(self.frame) != exp:
            errs.append(f"ledger has {len(self.frame)} rows, expected {exp}")
        for it in self.spec.items:
            s = self.series(it.item_id)
            if s["week"].tolist() != list(range(self.spec.weeks)):
                errs.append(f"{it.item_id}: weeks not 0..{self.spec.weeks - 1}")
                continue
            if it.kind in NEVER_HOLD and s["active"].any():
                errs.append(f"{it.item_id}: {it.kind} is marked active")
            bad = s[(s["active"]) != (s["true_effect"].abs() >= MIN_ACTIVE)]
            if len(bad):
                errs.append(f"{it.item_id}: active flag disagrees with true_effect in {len(bad)} weeks")
        return errs


def _build_ledger(spec: WorldSpec, X: pd.DataFrame, Q: pd.DataFrame, week_idx: np.ndarray, dates) -> TruthLedger:
    W = spec.weeks
    rows = []
    for it in spec.items:
        base = np.ones(len(X), dtype=bool)
        for f, q in it.conds:
            base &= Q[f].to_numpy() == q
        if it.unless is not None:
            base &= Q[it.unless[0]].to_numpy() != it.unless[1]
        n_rows = np.bincount(week_idx[base], minlength=W)
        if it.gate is None:
            in_state = np.where(n_rows > 0, n_rows, 0)
        else:
            in_state = np.bincount(week_idx[base & it.gate.mask(X, Q)], minlength=W)
        frac = np.divide(in_state, n_rows, out=np.zeros(W), where=n_rows > 0)
        mult = it.multipliers(W)
        eff = mult * (frac * it.effect + (1 - frac) * it.off_effect)
        eff = np.where(n_rows > 0, eff, 0.0)
        for w in range(W):
            rows.append((it.item_id, it.kind, w, str(pd.Timestamp(dates[w]).date()), spec.phase_of(w),
                         bool(abs(eff[w]) >= MIN_ACTIVE), float(eff[w]), float(mult[w] * it.effect), float(mult[w] * it.off_effect),
                         None if it.gate is None else bool(frac[w] > 0), float(frac[w]), int(n_rows[w])))
    frame = pd.DataFrame(rows, columns=["item_id", "kind", "week", "date", "phase", "active", "true_effect", "on_effect",
                                        "off_effect", "condition_state", "gate_frac", "n_rows"])
    return TruthLedger(spec, frame, dates)


# ---------------------------------------------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------------------------------------------
def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(salt)])


@dataclass
class LearnerView:
    """What a learner standing at `now` is allowed to see: features through `now`, outcomes only for weeks that matured."""
    now: pd.Timestamp
    X: pd.DataFrame
    y: pd.Series
    dropped: tuple = ()


class PlantedWorld:
    """spec + realised panel + exact truth.  Arrays are row-aligned with X.index."""

    def __init__(self, spec: WorldSpec, seed: int, X: pd.DataFrame, y_raw: pd.Series, contrib: np.ndarray, week_idx: np.ndarray,
                 dates: Sequence, ledger: TruthLedger | None = None):
        self.spec, self.seed = spec, seed
        self.X, self.y_raw, self.contrib, self.week_idx = X, y_raw, contrib, week_idx
        self.dates = tuple(pd.Timestamp(d) for d in dates)
        self.y = y_raw - y_raw.groupby(level=0).transform("mean")           # excess return: what a ranker is judged on
        self._Q = P.quintiles(X[self.feature_columns()])
        self.ledger = ledger if ledger is not None else _build_ledger(spec, X, self._Q, week_idx, self.dates)
        self._tot_dm: np.ndarray | None = None

    # ------------------------------------------------------------ columns
    def feature_columns(self) -> list[str]:
        return [c for c in self.X.columns if c.startswith("f")]

    def market_columns(self) -> list[str]:
        return [c for c in self.X.columns if c.startswith("m_")]

    def canary_columns(self) -> list[str]:
        return [c for c in self.X.columns if c.startswith(CANARY_PREFIX)]

    def week_of(self, when) -> int:
        d = pd.Timestamp(when)
        for i, x in enumerate(self.dates):
            if x == d:
                return i
        raise KeyError(f"{when} is not a week of this world")

    def _row_mask(self, item: Item, weeks: tuple | None = None) -> np.ndarray:
        m = np.ones(len(self.X), dtype=bool)
        for f, q in item.conds:
            m &= self._Q[f].to_numpy() == q
        if item.unless is not None:
            m &= self._Q[item.unless[0]].to_numpy() != item.unless[1]
        if weeks is not None:
            m &= (self.week_idx >= weeks[0]) & (self.week_idx < weeks[1])
        return m

    def _key_mask(self, key, weeks: tuple | None = None) -> np.ndarray:
        conds, unless = key
        m = np.ones(len(self.X), dtype=bool)
        for c in conds:
            f, q = c.rsplit(" q", 1)
            if f not in self._Q:
                return np.zeros(len(self.X), dtype=bool)
            m &= self._Q[f].to_numpy() == int(q)
        if unless is not None:
            f, q = unless.rsplit(" q", 1)
            if f in self._Q:
                m &= self._Q[f].to_numpy() != int(q)
        if weeks is not None:
            m &= (self.week_idx >= weeks[0]) & (self.week_idx < weeks[1])
        return m

    # ------------------------------------------------------------ what a learner may see
    def learner_view(self, now, include_canary: bool = False) -> LearnerView:
        """Fail closed: features dated <= now, outcomes only for weeks whose NEXT week's date is <= now (a forward return
        matures at the following close).  `now` outside the sample raises rather than returning an empty or full panel."""
        now = pd.Timestamp(now)
        if now < self.dates[0]:
            raise FirewallBreach(f"now={now.date()} precedes the world's first week {self.dates[0].date()}")
        k = int(np.searchsorted(np.array(self.dates, dtype="datetime64[ns]"), np.datetime64(now), side="right"))  # weeks with date <= now
        cols = self.feature_columns() + self.market_columns() + (self.canary_columns() if include_canary else [])
        xm = self.week_idx < k
        ym = self.week_idx < k - 1
        dropped = tuple(self.canary_columns()) if not include_canary else ()
        return LearnerView(now, self.X.loc[xm, cols], self.y[ym], dropped)

    # ------------------------------------------------------------ exact effects
    def total_demeaned(self) -> np.ndarray:
        """Sum of every planted contribution, demeaned per week the way y is: the exact signal a ranker can earn."""
        if self._tot_dm is None:
            tot = pd.Series(self.contrib.sum(axis=1) if self.contrib.size else np.zeros(len(self.X)), index=self.X.index)
            self._tot_dm = (tot - tot.groupby(level=0).transform("mean")).to_numpy()
        assert self._tot_dm is not None
        return self._tot_dm

    def exact_key_effect(self, key, weeks: tuple | None = None) -> float:
        """Mean exact (demeaned) planted return over the rows a condition key selects, over weeks [a, b).  Counts the
        mechanical side effects of demeaning as real (if the top quintile gains, the rest truly loses a little)."""
        m = self._key_mask(key, weeks)
        return float(self.total_demeaned()[m].mean()) if m.any() else 0.0

    def exact_key_stats(self, key, weeks: tuple | None = None) -> tuple:
        """(mean exact effect, standard error from row-to-row spread of the planted signal, rows).  A narrow key overlaps the
        real patterns by chance; the standard error says how much of its exact mean is that overlap wobble."""
        m = self._key_mask(key, weeks)
        n = int(m.sum())
        if n == 0:
            return 0.0, 0.0, 0
        v = self.total_demeaned()[m]
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, n

    def exact_item_effect(self, item_id: str, weeks: tuple | None = None) -> float:
        """Mean of this item's own demeaned contribution over its own rows and weeks."""
        j = [i.item_id for i in self.spec.items].index(item_id)
        c = pd.Series(self.contrib[:, j], index=self.X.index)
        c = (c - c.groupby(level=0).transform("mean")).to_numpy()
        m = self._row_mask(self.spec.item(item_id), weeks)
        return float(c[m].mean()) if m.any() else 0.0

    def estimate_key_effect(self, key, weeks: tuple | None = None) -> dict:
        """What a statistician sees WITHOUT the ledger: per-week mean excess return of the key rows minus the other rows,
        averaged over weeks, with a t-statistic on the weekly differences (the honest unit: weeks, not rows)."""
        m = self._key_mask(key, weeks)
        y = self.y.to_numpy()
        n_w = self.spec.weeks
        cnt_in = np.bincount(self.week_idx[m], minlength=n_w)
        cnt_out = np.bincount(self.week_idx[~m], minlength=n_w)
        s_in = np.bincount(self.week_idx[m], weights=y[m], minlength=n_w)
        s_out = np.bincount(self.week_idx[~m], weights=y[~m], minlength=n_w)
        ok = (cnt_in > 0) & (cnt_out > 0)
        if weeks is not None:
            ok &= (np.arange(n_w) >= weeks[0]) & (np.arange(n_w) < weeks[1])
        if ok.sum() < 3:
            return {"effect": 0.0, "t": 0.0, "n_weeks": int(ok.sum())}
        d = s_in[ok] / cnt_in[ok] - s_out[ok] / cnt_out[ok]
        sd = d.std(ddof=1)
        return {"effect": float(d.mean()), "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else 0.0, "n_weeks": int(ok.sum())}

    def cell_weekly(self, item_id: str, weeks: tuple | None = None) -> np.ndarray:
        """F10: the weekly series a learner's retirement gate sees for this item - the mean excess return of the item's rows each
        week, signed by the planted direction (noise: +1).  Weeks without rows are skipped.  Trusted side: it calibrates the
        false-degrade study (retirement.degrade_study) on the same noise the learner faces."""
        it = self.spec.item(item_id)
        sign = -1.0 if (it.effect or it.off_effect) < 0 else 1.0
        m = self._row_mask(it, weeks)
        y = self.y.to_numpy()
        n = np.bincount(self.week_idx[m], minlength=self.spec.weeks)
        s = np.bincount(self.week_idx[m], weights=y[m], minlength=self.spec.weeks)
        ok = n > 0
        return sign * s[ok] / n[ok]

    # ------------------------------------------------------------ hidden-item economics (test 7)
    def use_pnl(self, item_id: str, weeks: tuple, cost: float = 0.01, exact: bool = False) -> float:
        """Mean weekly result of USING an item (long its key rows if its original effect is positive, short if negative)
        after a per-week cost of acting on it (spread, slippage, opportunity).  exact=True reads the noise-free planted
        contribution, so the answer is a property of the world, not of one noisy draw."""
        it = self.spec.item(item_id)
        sign = 1.0 if (it.effect or it.off_effect) >= 0 else -1.0
        m = self._row_mask(it, weeks)
        if not m.any():
            return -cost
        if exact:                                   # the item's own demeaned contribution, free of other items' side effects
            j = [i.item_id for i in self.spec.items].index(item_id)
            c = pd.Series(self.contrib[:, j], index=self.X.index)
            src = (c - c.groupby(level=0).transform("mean")).to_numpy()
        else:
            src = self.y.to_numpy()
        return float(sign * src[m].mean() - cost)

    # ------------------------------------------------------------ summaries / identity
    def summary(self) -> dict:
        return {"name": self.spec.name, "seed": self.seed, "rows": len(self.X), "weeks": self.spec.weeks,
                "stocks": self.spec.stocks, "items": {it.kind: sum(1 for j in self.spec.items if j.kind == it.kind) for it in self.spec.items},
                "phases": self.spec.phase_bounds(), "ledger_hash": self.ledger.content_hash()}

    def content_hash(self) -> str:
        """Determinism fingerprint of the realised panel (rounded so it is stable across BLAS/platform noise)."""
        return stable_hash({"spec": self.spec.name, "seed": self.seed, "n": len(self.X),
                            "X": [round(float(v), 6) for v in np.asarray(self.X.to_numpy()[:: max(len(self.X) // 97, 1)]).ravel()[:400]],
                            "y": [round(float(v), 8) for v in self.y_raw.to_numpy()[:: max(len(self.X) // 97, 1)]],
                            "ledger": self.ledger.content_hash()})

    def manifest(self) -> dict:
        """JSON-able record of exactly what was planted (for experiment memory / provenance)."""
        return {"spec": stable_hash(self.spec), "seed": self.seed, "hash": self.content_hash(),
                "items": [{"id": i.item_id, "kind": i.kind, "key": i.key_named, "effect": i.effect, "off": i.off_effect,
                           "profile": i.profile, "params": list(i.params), "change_week": i.change_week(self.spec.weeks)}
                          for i in self.spec.items]}


def make_world(spec: WorldSpec, seed: int = 0) -> PlantedWorld:
    """Generate the panel: persistent features, correlated copies, a persistent market regime plus two distractors, heavy-tailed
    idiosyncratic returns with week and sector shocks, the planted items, and the future-leak canary."""
    spec.check()
    W, S, F = spec.weeks, spec.stocks, spec.n_feat
    dates = pd.date_range(spec.start, periods=W, freq="W-FRI")
    tick = [f"T{i:03d}" for i in range(S)]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    g = _rng(seed, 1)
    Z = np.empty((W, S, F))
    Z[0] = g.standard_normal((S, F))
    a = spec.persistence
    for t in range(1, W):
        Z[t] = a * Z[t - 1] + np.sqrt(1 - a * a) * g.standard_normal((S, F))
    for c, s, r in spec.dup_features:
        ci, si = int(c[1:]), int(s[1:])
        Z[:, :, ci] = r * Z[:, :, si] + np.sqrt(1 - r * r) * Z[:, :, ci]
    X = pd.DataFrame(Z.reshape(W * S, F).astype(np.float64), index=idx, columns=[f"f{i}" for i in range(F)])
    gm = _rng(seed, 2)
    reg = np.empty(W)
    reg[0] = gm.standard_normal()
    ra = spec.regime_persistence
    for t in range(1, W):
        reg[t] = ra * reg[t - 1] + np.sqrt(1 - ra * ra) * gm.standard_normal()
    X["m_vix"] = np.repeat(reg, S)
    X["m_breadth"] = np.repeat(gm.standard_normal(W), S)
    X["m_term"] = np.repeat(gm.standard_normal(W), S)
    gr = _rng(seed, 3)
    df = spec.tail_df
    idio = gr.standard_t(df, W * S) * spec.noise_sd / np.sqrt(df / (df - 2))
    sector = gr.integers(0, spec.n_sectors, S)
    sec = gr.normal(0, spec.sector_sd, (W, spec.n_sectors))[:, sector].reshape(-1)
    y_raw = pd.Series(idio + np.repeat(gr.normal(0, spec.week_sd, W), S) + sec, index=idx)
    week_idx = np.repeat(np.arange(W), S)
    Q = P.quintiles(X[[f"f{i}" for i in range(F)]])
    contrib = np.zeros((W * S, len(spec.items)))
    for j, it in enumerate(spec.items):
        m = np.ones(W * S, dtype=bool)
        for f, q in it.conds:
            m &= Q[f].to_numpy() == q
        if it.unless is not None:
            m &= Q[it.unless[0]].to_numpy() != it.unless[1]
        gate = it.gate.mask(X, Q) if it.gate is not None else np.ones(W * S, dtype=bool)
        mult = it.multipliers(W)[week_idx]
        contrib[:, j] = np.where(m, mult * np.where(gate, it.effect, it.off_effect), 0.0)
    y_raw = y_raw + contrib.sum(axis=1)
    # canary: the realised forward return itself plus a little noise - information no learner may legitimately hold at decision time
    gc = _rng(seed, 4)
    X[CANARY_COL] = y_raw.to_numpy() + gc.standard_normal(W * S) * spec.canary_noise * float(y_raw.std())
    return PlantedWorld(spec, seed, X, y_raw, contrib, week_idx, dates)


# ---------------------------------------------------------------------------------------------------------------
# identity changes and era changes (tests 8 and 9)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reidentified:
    world: PlantedWorld
    ticker_map: Mapping[str, str]
    date_shift_days: int


def reidentify(world: PlantedWorld, seed: int, tickers: bool = True, shift_years: int = 0, shuffle_rows: bool = True) -> Reidentified:
    """The same world under new identities: every ticker renamed by a random bijection, every date moved by a whole number of
    52-week blocks (weekday structure preserved), rows shuffled inside each date so neither name order nor row order carries
    information (names leaked through ties three times before).  Values, truth and the ledger's effects are untouched, so a
    learner that genuinely learned patterns must score the same, and a name/date memoriser must not."""
    g = _rng(seed, 11)
    old = list(dict.fromkeys(world.X.index.get_level_values(1)))
    if tickers:
        codes = g.permutation(10 ** 5)[: len(old)]
        tmap = {o: f"Z{int(c):05d}" for o, c in zip(old, codes)}
    else:
        tmap = {o: o for o in old}
    shift = 364 * int(shift_years)
    n = len(world.X)
    order = np.arange(n)
    if shuffle_rows:
        for w in range(world.spec.weeks):
            lo = w * world.spec.stocks
            order[lo:lo + world.spec.stocks] = lo + g.permutation(world.spec.stocks)
    d = world.X.index.get_level_values(0).to_numpy()[order] + np.timedelta64(shift, "D")
    t = np.array([tmap[x] for x in world.X.index.get_level_values(1).to_numpy()[order]], dtype=object)
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(d), t], names=["date", "ticker"])
    X2 = world.X.iloc[order].copy(); X2.index = idx
    y2 = world.y_raw.iloc[order].copy(); y2.index = idx
    new_dates = [x + pd.Timedelta(days=shift) for x in world.dates]
    led = world.ledger.frame.copy()
    led["date"] = [str(new_dates[w].date()) for w in led["week"]]
    spec2 = dataclasses.replace(world.spec, start=str(new_dates[0].date()))
    w2 = PlantedWorld(spec2, world.seed, X2, y2, world.contrib[order], world.week_idx[order], new_dates,
                      ledger=TruthLedger(spec2, led, new_dates))
    return Reidentified(w2, tmap, shift)


def year_swap(world_or_spec, seed: int, years: int = 6, vol_scale: float = 1.3, regime_persistence: float = 0.8) -> PlantedWorld:
    """A different era with the same situation classes: identical items, conditions and effects, but new dates, a fresh
    noise draw, a different volatility level and a different market-regime rhythm.  Knowledge that is about the SITUATION
    must transfer; knowledge that is about the calendar must not."""
    spec = world_or_spec.spec if isinstance(world_or_spec, PlantedWorld) else world_or_spec
    if vol_scale <= 0:
        raise ValueError("vol_scale must be positive")
    start = pd.Timestamp(spec.start) + pd.Timedelta(days=364 * int(years))
    spec2 = dataclasses.replace(spec, name=f"{spec.name}@+{years}y", start=str(start.date()), noise_sd=spec.noise_sd * vol_scale,
                                week_sd=spec.week_sd * vol_scale, sector_sd=spec.sector_sd * vol_scale,
                                regime_persistence=regime_persistence)
    return make_world(spec2, seed)


def same_situation_classes(a: PlantedWorld, b: PlantedWorld) -> bool:
    return a.spec.situation_classes() == b.spec.situation_classes()


# ---------------------------------------------------------------------------------------------------------------
# future-leak canary (test 11)
# ---------------------------------------------------------------------------------------------------------------
def assert_no_canary(columns: Iterable[str]) -> None:
    """The firewall the canary exists to test: any learner input naming a canary column is a breach, fail closed."""
    bad = [c for c in columns if str(c).startswith(CANARY_PREFIX)]
    if bad:
        raise FirewallBreach(f"future-leak canary column(s) reached the learner: {bad}")


def leak_suspicion(pred: pd.Series, world: PlantedWorld, weeks: tuple | None = None, threshold: float = 0.3) -> dict:
    """Rank information coefficient of a learner's scores against realised excess returns.  Real signal in these worlds is a
    few percent of a standard deviation, so an IC above `threshold` is not skill - it is a leak (the canary scores ~0.9)."""
    df = pd.DataFrame({"p": pred, "y": world.y}).dropna()
    if weeks is not None:
        wi = pd.Series(world.week_idx, index=world.X.index).reindex(df.index)
        df = df[(wi >= weeks[0]) & (wi < weeks[1])]
    if len(df) < 30:
        return {"ic": 0.0, "n": int(len(df)), "leak": False}
    ic = df.groupby(level=0).apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 5 else np.nan).dropna()
    m = float(ic.mean()) if len(ic) else 0.0
    return {"ic": m, "n": int(len(df)), "leak": bool(abs(m) > threshold)}


# ---------------------------------------------------------------------------------------------------------------
# claims and scoring
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Claim:
    """One thing a learner claims to know.  Only `name` is required (the miner's key_named)."""
    name: str
    effect: float | None = None
    condition: Any = None                   # ("m", col, lo, hi) | ("q", feature, qs) | ("not", spec) | Gate | callable(X, Q)->mask


def as_claim(obj) -> Claim:
    if isinstance(obj, Claim):
        return obj
    if isinstance(obj, str):
        return Claim(obj)
    if isinstance(obj, Mapping):
        return Claim(str(obj.get("name") or obj.get("key_named")), obj.get("effect"), obj.get("condition"))
    name = getattr(obj, "key_named", None) or getattr(obj, "name", None)
    if not name:
        raise TypeError(f"cannot read a claim name from {type(obj).__name__}")
    return Claim(str(name), getattr(obj, "effect", None), getattr(obj, "condition", None))


def _safe_key(name: str):
    try:
        return P.parse_named(name)
    except Exception:
        return None


@dataclass
class ClaimScore:
    at_week: int
    n_claims: int = 0
    tp: int = 0
    fp: int = 0
    recall: float = 0.0
    precision: float = 0.0
    false_discovery_rate: float = 0.0
    recall_by_kind: dict = field(default_factory=dict)          # real_now kinds -> share claimed
    false_claim_by_kind: dict = field(default_factory=dict)     # trap kinds -> share claimed (lower is better)
    missed: list = field(default_factory=list)
    false_claims: list = field(default_factory=list)            # (claim name, reason)
    sign_errors: list = field(default_factory=list)
    unplanted: dict = field(default_factory=dict)               # claims matching no item: exact judgement

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def score_claims(world: PlantedWorld, claims: Iterable, at_week: int, lookback: int = 13, unplanted_min: float = 0.003,
                 unplanted_window: int = 26) -> ClaimScore:
    """Judge a learner's claimed knowledge set standing at `at_week` against the ledger.  A claim is a TRUE positive when it
    matches an item that is live at `at_week` (and, if the claim gives an effect, with the right sign); a FALSE discovery when
    it matches a trap (noise, duplicate, dead hallucination, hidden or decayed item, wrong sign) or is unparsable.  A claim
    matching no planted item is judged by the exact realised effect of its rows over the trailing `unplanted_window` weeks; it must
    clear `unplanted_min` and three standard errors (chance overlap with the real patterns is not a pattern) to count as real."""
    if not 0 <= at_week < world.spec.weeks:
        raise IndexError(f"at_week {at_week}")
    cl = [as_claim(c) for c in claims]
    seen, uniq = set(), []
    for c in cl:                                    # the same claim twice is one claim, not two hits
        k = _safe_key(c.name)
        kk = k if k is not None else c.name
        if kk in seen:
            continue
        seen.add(kk)
        uniq.append((c, k))
    by_key = {it.key: it for it in world.spec.items}
    holdable = set(world.ledger.should_hold(at_week, lookback))
    sc = ClaimScore(at_week, n_claims=len(uniq))
    claimed_ids = set()
    kind_claimed: dict = {}
    win = (max(0, at_week - unplanted_window + 1), at_week + 1)
    for c, k in uniq:
        if k is None:
            sc.fp += 1; sc.false_claims.append((c.name, "unparsable")); continue
        it = by_key.get(k)
        if it is None:
            eff, se, _ = world.exact_key_stats(k, win)
            real = abs(eff) >= unplanted_min and abs(eff) > 3 * se and (c.effect is None or np.sign(c.effect) == np.sign(eff))
            sc.unplanted[c.name] = {"exact_effect": eff, "se": se, "real": bool(real)}
            if real:
                sc.tp += 1
            else:
                sc.fp += 1; sc.false_claims.append((c.name, "unplanted_no_effect"))
            continue
        claimed_ids.add(it.item_id)
        kind_claimed.setdefault(it.kind, set()).add(it.item_id)
        if it.item_id not in holdable:
            sc.fp += 1; sc.false_claims.append((c.name, f"trap:{it.kind}")); continue
        true_sign = world.ledger.sign_at(it.item_id, at_week) or int(np.sign(it.effect))
        if c.effect is not None and np.sign(c.effect) != true_sign:
            sc.fp += 1; sc.sign_errors.append(it.item_id); sc.false_claims.append((c.name, f"wrong_sign:{it.kind}")); continue
        sc.tp += 1
    real_ids = sorted(holdable)
    sc.missed = [i for i in real_ids if i not in claimed_ids]
    hit_real = [i for i in real_ids if i in claimed_ids and i not in sc.sign_errors]
    sc.recall = len(hit_real) / len(real_ids) if real_ids else 1.0
    sc.precision = sc.tp / len(uniq) if uniq else 1.0
    sc.false_discovery_rate = sc.fp / len(uniq) if uniq else 0.0
    for kind in KINDS:
        ids = [i.item_id for i in world.spec.items if i.kind == kind]
        if not ids:
            continue
        live = [i for i in ids if i in holdable]
        dead = [i for i in ids if i not in holdable]
        if live:
            sc.recall_by_kind[kind] = sum(1 for i in live if i in hit_real) / len(live)
        if dead:
            sc.false_claim_by_kind[kind] = sum(1 for i in dead if i in claimed_ids) / len(dead)
    return sc


# ------------------------------------------------------------ degradation-detection delay
@dataclass
class DelayReport:
    per_item: dict = field(default_factory=dict)      # item_id -> {change_week, detected_week, delay}
    median_delay: float | None = None
    mean_delay: float | None = None
    missed: list = field(default_factory=list)
    premature: list = field(default_factory=list)     # flagged before the change actually happened
    false_alarms: list = field(default_factory=list)  # flagged items that never change
    detection_rate: float = 0.0


def degradation_delays(world: PlantedWorld, detections: Mapping, tolerance: int = 0) -> DelayReport:
    """`detections` maps item_id or key_named -> the week index or date at which the learner first flagged it as degraded.
    Delay = detected week - true change week (weeks).  Flagging more than `tolerance` weeks BEFORE the change is premature
    (a false alarm that happened to be near a change is not skill); flagging an item that never changes is a false alarm.
    Detections are only accepted if they lie inside the sample."""
    by_key = {it.key: it.item_id for it in world.spec.items}
    resolved: dict = {}
    for k, v in detections.items():
        iid = k if any(it.item_id == k for it in world.spec.items) else by_key.get(_safe_key(str(k)) or ())
        if iid is None:
            raise KeyError(f"detection for unknown item {k!r}")
        wk = world.week_of(v) if isinstance(v, (dt.date, pd.Timestamp, str)) else int(v)
        if not 0 <= wk < world.spec.weeks:
            raise IndexError(f"detection week {wk} outside the sample")
        resolved[iid] = min(wk, resolved.get(iid, wk))       # the FIRST flag counts
    rep = DelayReport()
    delays = []
    for it in world.spec.items:
        cw = it.change_week(world.spec.weeks)
        det = resolved.get(it.item_id)
        if cw is None or it.kind in NEVER_HOLD:
            if det is not None:
                rep.false_alarms.append(it.item_id)
            continue
        if det is None:
            rep.missed.append(it.item_id)
            rep.per_item[it.item_id] = {"change_week": cw, "detected_week": None, "delay": None}
            continue
        d = det - cw
        rep.per_item[it.item_id] = {"change_week": cw, "detected_week": det, "delay": d}
        if d < -tolerance:
            rep.premature.append(it.item_id)
        else:
            delays.append(d)
    if delays:
        rep.median_delay, rep.mean_delay = float(np.median(delays)), float(np.mean(delays))
    n_change = sum(1 for it in world.spec.items if it.change_week(world.spec.weeks) is not None and it.kind not in NEVER_HOLD)
    rep.detection_rate = len(delays) / n_change if n_change else 1.0
    return rep


# ------------------------------------------------------------ condition recovery
def _state_mask(world: PlantedWorld, spec_) -> np.ndarray:
    if isinstance(spec_, Gate):
        return spec_.mask(world.X, world._Q)
    if callable(spec_):
        return np.asarray(spec_(world.X, world._Q), dtype=bool)
    if not isinstance(spec_, (tuple, list)) or not spec_:
        raise TypeError(f"condition spec {spec_!r}")
    tag = spec_[0]
    if tag == "not":
        return ~_state_mask(world, spec_[1])
    if tag == "m":
        _, col, lo, hi = spec_
        v = world.X[col].to_numpy()
        return (v > lo) & (v <= hi)
    if tag == "q":
        _, feat, qs = spec_
        return world._Q[feat].isin(tuple(qs)).to_numpy()
    raise ValueError(f"condition tag {tag!r}")


@dataclass
class ConditionScore:
    item_id: str
    accuracy: float
    balanced_accuracy: float
    n_rows: int
    degenerate: bool = False           # only one true state present: balanced accuracy undefined, plain accuracy reported
    boundary_error: float | None = None


def _true_state(world: PlantedWorld, it: Item) -> np.ndarray:
    if it.kind == UNLESS:                                   # effect ON when the exception does not hold
        assert it.unless is not None
        return world._Q[it.unless[0]].to_numpy() != it.unless[1]
    assert it.gate is not None
    return it.gate.mask(world.X, world._Q)


def condition_recovery(world: PlantedWorld, claimed: Mapping) -> dict:
    """item_id -> ConditionScore.  `claimed` maps item_id (or key_named) to the condition the learner says the item needs.
    Scored on the item's own base rows while its profile is alive, as balanced accuracy of the learner's in-state call against
    the true in-state call - so 'always on' scores 0.5, not 0.9.  For an unless item the claimed condition is the EXCEPTION."""
    by_key = {it.key: it for it in world.spec.items}
    out = {}
    for k, cond in claimed.items():
        it = next((i for i in world.spec.items if i.item_id == k), None) or by_key.get(_safe_key(str(k)) or ())
        if it is None:
            raise KeyError(f"condition claimed for unknown item {k!r}")
        if it.kind not in CONDITIONED:
            raise ValueError(f"{it.item_id} ({it.kind}) has no condition to recover")
        truth = _true_state(world, it)
        claim = ~_state_mask(world, cond) if it.kind == UNLESS else _state_mask(world, cond)
        alive = world.ledger.series(it.item_id).set_index("week")
        live_weeks = set(alive.index[alive["on_effect"].abs() + alive["off_effect"].abs() > 0].tolist())
        rows = np.ones(len(world.X), dtype=bool)
        for f, q in it.conds:
            rows &= world._Q[f].to_numpy() == q
        rows &= np.isin(world.week_idx, list(live_weeks))
        if not rows.any():
            out[it.item_id] = ConditionScore(it.item_id, 0.0, 0.0, 0, True)
            continue
        t, c = truth[rows], claim[rows]
        acc = float((t == c).mean())
        pos, neg = t.sum(), (~t).sum()
        degenerate = pos == 0 or neg == 0
        if degenerate:
            bal = acc
        else:
            bal = 0.5 * float(((t & c).sum() / pos) + ((~t & ~c).sum() / neg))
        be = None
        if it.kind == REGIME and isinstance(cond, (tuple, list)) and cond and cond[0] == "m" and cond[1] == it.gate.col:
            be = float(abs(np.clip(cond[2], -50, 50) - np.clip(it.gate.lo, -50, 50)) + abs(np.clip(cond[3], -50, 50) - np.clip(it.gate.hi, -50, 50)))
        out[it.item_id] = ConditionScore(it.item_id, acc, bal, int(rows.sum()), bool(degenerate), be)
    return out


# ---------------------------------------------------------------------------------------------------------------
# reference learners: the oracle (reads the ledger) and null learners (know nothing)
# ---------------------------------------------------------------------------------------------------------------
def oracle_claims(world: PlantedWorld, at_week: int, lookback: int = 13) -> list[Claim]:
    """Exactly the knowledge a perfect learner holds at `at_week`: every live real item, its true sign and its condition."""
    out = []
    for iid in world.ledger.should_hold(at_week, lookback):
        it = world.spec.item(iid)
        s = world.ledger.series(iid)
        s = s[(s["week"] <= at_week) & (s["week"] > at_week - lookback) & s["active"]]
        eff = float(s["true_effect"].iloc[-1]) if len(s) else it.effect
        cond = None
        if it.kind == UNLESS:
            assert it.unless is not None
            cond = ("q", it.unless[0], (it.unless[1],))
        elif it.gate is not None:
            cond = it.gate.spec()
        out.append(Claim(it.key_named, eff, cond))
    return out


def oracle_detections(world: PlantedWorld, delay: int = 0) -> dict:
    return {it.item_id: min(it.change_week(world.spec.weeks) + delay, world.spec.weeks - 1)
            for it in world.spec.items if it.kind in CHANGING and it.change_week(world.spec.weeks) is not None}


def oracle_conditions(world: PlantedWorld) -> dict:
    out = {}
    for it in world.spec.items:
        if it.kind == UNLESS:
            out[it.item_id] = ("q", it.unless[0], (it.unless[1],))
        elif it.kind in CONDITIONED:
            out[it.item_id] = it.gate.spec()
    return out


def null_claims(world: PlantedWorld, at_week: int) -> list[Claim]:
    """A learner that has learned nothing claims nothing."""
    return []


def random_claims(world: PlantedWorld, n: int, seed: int) -> list[Claim]:
    """A learner guessing single-condition keys at random: expected precision is the planted base rate, not 1."""
    g = _rng(seed, 21)
    feats = world.feature_columns()
    keys = {(feats[int(g.integers(len(feats)))], int(g.integers(5))) for _ in range(n * 3)}
    return [Claim(f"{f} q{q}") for f, q in sorted(keys)[:n]]


def memoriser_claims(world: PlantedWorld) -> list[Claim]:
    """A memoriser 'knows' the past by identity: it claims every item that was ever active, whatever the date (test 10 foil)."""
    ever = world.ledger.frame.groupby("item_id")["active"].any()
    return [Claim(world.spec.item(i).key_named) for i, a in ever.items() if a]


# ---------------------------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------------------------
def render_ledger(world: PlantedWorld, at_week: int | None = None) -> str:
    """Plain-text table of what was planted and what is true at `at_week` (default: first evaluation week)."""
    at = world.spec.confirm_end if at_week is None else at_week
    hold = set(world.ledger.should_hold(at))
    lines = [f"world {world.spec.name} seed={world.seed} weeks={world.spec.weeks} stocks={world.spec.stocks}; truth at week {at} "
             f"({world.dates[at].date()}, {world.spec.phase_of(at)})",
             f"{'id':<14}{'kind':<14}{'key':<26}{'effect@wk':>10}{'change':>8}  hold"]
    for it in world.spec.items:
        cw = it.change_week(world.spec.weeks)
        lines.append(f"{it.item_id:<14}{it.kind:<14}{it.key_named:<26}{world.ledger.effect(it.item_id, at):>10.4f}"
                     f"{'-' if cw is None else cw:>8}  {'YES' if it.item_id in hold else 'no'}")
    return "\n".join(lines)


def render_score(sc: ClaimScore) -> str:
    rb = ", ".join(f"{k}={v:.2f}" for k, v in sc.recall_by_kind.items()) or "-"
    fb = ", ".join(f"{k}={v:.2f}" for k, v in sc.false_claim_by_kind.items()) or "-"
    return (f"week {sc.at_week}: claims={sc.n_claims} tp={sc.tp} fp={sc.fp} recall={sc.recall:.2f} precision={sc.precision:.2f} "
            f"FDR={sc.false_discovery_rate:.2f}\n  recall by kind: {rb}\n  false claims by trap kind: {fb}\n  missed: {sc.missed}")


# ---------------------------------------------------------------------------------------------------------------
# result stamps (test 12: a result computed by different code than is now loaded must not be accepted)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WorldStamp:
    world_hash: str
    spec_hash: str
    seed: int
    code_hash: str


def stamp_world(world: PlantedWorld, code_hash: str | None = None) -> WorldStamp:
    return WorldStamp(world.content_hash(), stable_hash(world.spec), world.seed, code_hash if code_hash is not None else current_code_hash())


def check_stamp(stamp: WorldStamp, world: PlantedWorld, code_hash: str | None = None) -> list[str]:
    """Reasons a stored result must be refused; empty means the stamp still describes this world under this code."""
    now_code = code_hash if code_hash is not None else current_code_hash()
    errs = []
    if stamp.code_hash != now_code:
        errs.append(f"stale: code changed ({stamp.code_hash} -> {now_code})")
    if stamp.spec_hash != stable_hash(world.spec) or stamp.seed != world.seed:
        errs.append("stale: world spec or seed differs")
    elif stamp.world_hash != world.content_hash():
        errs.append("stale: realised panel differs from the one scored")
    return errs
