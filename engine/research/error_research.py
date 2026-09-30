"""Error-driven research allocation (C68 checklists D, E, S, T, U; extends C66 sections 2, 19, 40; canon C66, C68, C63).

A prediction that was wrong is worth investigating in proportion to how wrong it was, how sure the system was, whether the same
error keeps coming back, what fixing it could be worth and how much of the market it touches. This module turns matured
prediction errors (P01's records, read by duck typing) into

  D  an intensity = magnitude x confidence x repeatability x potential value x market significance and a depth tier; tiny errors
     stay NONE/CHEAP (logged, never a big job), and the intensity becomes an ExperimentValue / ResearchItem for the EXISTING
     priority engine (engine.research.priority.step) - there is no second scheduler here;
  E  a confident-wrong detector that opens a structured 15-question investigation (which assumption / feature / pattern failed,
     did it change, was it overridden, interaction, volatility, correlation, sector, timing, exit, was there evidence before
     (engine.research.knowability), why was it missed, else which unknowable cause, can a precursor now be found
     (engine.research.precursors / discovery)); nothing is guessed - a question without evidence stays UNANSWERED;
  S  an error-PATTERN book: systematic under/over-prediction, shrinkage (expected 8-10%, realised 3-6%), compression, tested with a
     t statistic and a sign run over standardised errors, persistence via engine.learning.surprise.surprise_half_life; repeated
     similar errors raise the multiplier monotonically, isolated noise leaves it at 1.0;
  U  compute allocation: causes shown unknowable, and cells whose past investigations improved nothing, are deprioritised through
     the priority engine's own external_multipliers / WasteTracker hooks;
  T  the eleven self-research questions as ResearchQuestion objects whose value comes from measured evidence.

Research-world rule (C64/C66): everything here is built from matured outcomes (MATURED_RESEARCH_STATE). A record must have matured
strictly before `now` (FirewallBreach otherwise) and all text is identity free. Public entry: `step(state, now, records, ...)`.
Builds on engine.learning.surprise, engine.research.priority, engine.research.knowability, engine.research.core.
IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from engine.learning.research_priority import identity_leak
from engine.learning.research_policy import ComputeBudget, ResearchTarget
from engine.learning.surprise import benjamini_hochberg, continuous_z, surprise_bits, surprise_half_life
from engine.research import priority as P
from engine.research.core import (ExperimentValue, FirewallBreach, Knowability, Namespace, Problem, ResearchQuestion, require_past,
                                  stable_hash)

LABEL = "IMPLEMENTED - NOT VALIDATED"
DIMS = ("pattern", "sector", "regime", "stock_type")
UNKNOWABLE_CLASSES = (Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE, Knowability.UNKNOWN)
KNOWABLE_CLASSES = (Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE)


class ErrorResearchError(ValueError):
    pass


class Tier(str, enum.Enum):
    NONE = "NONE"
    CHEAP = "CHEAP"
    STANDARD = "STANDARD"
    DEEP = "DEEP"

    def __str__(self):
        return self.value


TIER_ORDER = (Tier.NONE, Tier.CHEAP, Tier.STANDARD, Tier.DEEP)


class BiasKind(str, enum.Enum):
    NONE = "NONE"
    UNDER = "UNDERPREDICTION"          # realised persistently larger than expected
    OVER = "OVERPREDICTION"
    SHRINK = "SHRINKAGE"               # over-prediction of a specific size: expected 8-10%, realised a fraction of it
    COMPRESSED = "COMPRESSED"          # realised barely moves with the expectation (slope well under 1)

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class ErrorConfig:
    """Priors, not tuned values (C63). Sizes are fractions (0.01 = 1 percentage point)."""
    default_scale: float = 0.02         # expected error sd when a record does not state one
    material_error: float = 0.01        # the +-1pp target: errors under half of it are noise for research purposes
    z_tiny: float = 1.0                 # |z| under this earns no magnitude at all
    z_sat: float = 3.0                  # z-excess at which the magnitude term is 1 - 1/e
    abs_sat: float = 1.5                # error in units of material_error at which the size term is 1 - 1/e
    conf_floor: float = 0.2             # confidence term = floor + (1 - floor) * confidence^2
    rep_base: float = 0.3               # repeatability of an isolated error
    potential_floor: float = 0.3
    market_floor: float = 0.2
    cw_conf: float = 0.7                # confident-wrong: confidence at least this ...
    cw_z: float = 2.5                   # ... and standardised error at least this ...
    cw_abs: float = 0.02                # ... and an absolute error this big (a tiny scale cannot make a trivial miss 'confident-wrong')
    cw_boost: float = 2.5
    tier_cut: tuple = (0.004, 0.02, 0.08)      # intensity cuts between NONE|CHEAP|STANDARD|DEEP
    tier_minutes: tuple = (0.0, 5.0, 30.0, 120.0)
    window: int = 60                    # newest matured errors per group used for pattern tests
    min_n: int = 6
    t0: float = 2.5                     # |t| a group must clear: many groups are tested at once, so 2.0 would flag noise routinely
    sign_p: float = 0.10                # one-sided binomial p of the sign run must be under this
    gain: float = 1.2                   # multiplier = 1 + gain * log1p(evidence) + run_gain * extra run
    run_gain: float = 0.05
    min_run: int = 3
    max_mult: float = 8.0
    shrink_ratio: float = 0.75
    compressed_slope: float = 0.6
    z_neutral: float = 0.5              # errors within this many sd are not counted as a direction for the sign run
    feature_shift_z: float = 2.0
    reliability_drop: float = 0.15
    vol_ratio: float = 1.5
    corr_shift: float = 0.25
    sector_share: float = 0.6
    timing_days: float = 1.0
    exit_regret_share: float = 0.25
    unknowable_mult: float = 0.1
    barren_decay: float = 0.5           # cell multiplier exp(-decay * consecutive investigations with no improvement)
    barren_eps: float = 0.03
    barren_floor: float = 0.05
    self_min_n: int = 12
    self_alpha: float = 0.10
    fdr_q: float = 0.10                 # Benjamini-Hochberg level across ALL groups tested at once (the error-pattern family)

    def check(self) -> list:
        errs = []
        if not (0 < self.tier_cut[0] < self.tier_cut[1] < self.tier_cut[2]):
            errs.append("tier cuts must be increasing and positive")
        if len(self.tier_minutes) != 4 or list(self.tier_minutes) != sorted(self.tier_minutes) or self.tier_minutes[0] != 0:
            errs.append("tier minutes must be 4 non-decreasing values starting at 0")
        if not (0 < self.cw_conf <= 1) or self.cw_z <= 0 or self.default_scale <= 0 or self.material_error <= 0:
            errs.append("confident-wrong or scale thresholds out of range")
        if self.min_n < 3 or self.window < self.min_n:
            errs.append("window must hold at least min_n >= 3 records")
        return errs


# ------------------------------------------------------------------------------------------------------ the error record

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class ErrorObs:
    """One matured prediction error in the units the research world needs. P01's record is adapted by `obs_from_record`."""
    obs_id: str
    decided_at: str
    matured_at: str
    expected: float
    realised: float
    confidence: float
    scale: float = 0.0                 # the error sd the prediction claimed; 0 = use the config default
    pattern: str = ""
    sector: str = ""
    regime: str = ""
    stock_type: str = ""
    market_weight: float = 0.5         # share of the book / market this situation touches, [0, 1]; 0.5 = not stated
    transfer: float = 0.5              # how far a lesson here would carry to other names / periods, [0, 1]
    exit_regret: float | None = None   # gain left on the table by the exit, as a fraction of the expected gain
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def check(self) -> list:
        errs = []
        if not self.obs_id:
            errs.append("error record without an id")
        for f in ("expected", "realised", "confidence"):
            if not math.isfinite(getattr(self, f)):
                errs.append(f"{self.obs_id}: {f} is not finite")
        if not 0.0 <= self.confidence <= 1.0:
            errs.append(f"{self.obs_id}: confidence outside [0,1]")
        if self.scale < 0 or not math.isfinite(self.scale):
            errs.append(f"{self.obs_id}: negative or non-finite scale")
        if not (0.0 <= self.market_weight <= 1.0 and 0.0 <= self.transfer <= 1.0):
            errs.append(f"{self.obs_id}: market_weight / transfer outside [0,1]")
        if str(self.matured_at) <= str(self.decided_at):
            errs.append(f"{self.obs_id}: outcome matured on or before the decision")
        for f in DIMS:
            if identity_leak(getattr(self, f)):
                errs.append(f"{self.obs_id}: {f} label carries a date or year (identity firewall)")
        return errs

    @property
    def error(self) -> float:
        return self.realised - self.expected

    def z(self, cfg: ErrorConfig) -> float:
        return continuous_z(self.expected, self.realised, self.scale if self.scale > 0 else cfg.default_scale)

    @property
    def direction_wrong(self) -> bool:
        return self.expected * self.realised < 0

    def cell(self, dims: Sequence[str] = DIMS) -> str:
        return "|".join(f"{d}={getattr(self, d)}" for d in dims if getattr(self, d))

    def groups(self) -> tuple:
        """Every group this error belongs to: the whole book, each single dimension, and pattern x regime."""
        out = ["all"] + [f"{d}={getattr(self, d)}" for d in DIMS if getattr(self, d)]
        if self.pattern and self.regime:
            out.append(f"pattern={self.pattern}|regime={self.regime}")
        return tuple(out)


_ALIASES = {"obs_id": ("record_id", "prediction_id", "id", "obs_id"), "decided_at": ("decided_at", "decision_date", "as_of"),
            "matured_at": ("matured_at", "outcome_date", "maturity"), "expected": ("expected", "expected_return", "expected_pct", "predicted"),
            "realised": ("realised", "realized", "realised_return", "actual"), "confidence": ("confidence", "conf"),
            "scale": ("scale", "expected_sd", "sigma"), "pattern": ("pattern", "pattern_id"), "sector": ("sector",),
            "regime": ("regime",), "stock_type": ("stock_type", "vol_bucket"), "market_weight": ("market_weight",),
            "transfer": ("transfer", "transfer_potential"), "exit_regret": ("exit_regret",)}


def obs_from_record(rec: Any) -> ErrorObs:
    """Adapt a prediction-error record (an object or a mapping) by field name; missing required fields raise, optional ones default."""
    def get(name, default=None):
        for a in _ALIASES[name]:
            v = rec.get(a) if isinstance(rec, Mapping) else getattr(rec, a, None)
            if v is not None:
                return v
        return default
    for req in ("obs_id", "decided_at", "matured_at", "expected", "realised", "confidence"):
        if get(req) is None:
            raise ErrorResearchError(f"prediction-error record lacks '{req}' (looked for {_ALIASES[req]})")
    o = ErrorObs(obs_id=str(get("obs_id")), decided_at=str(get("decided_at")), matured_at=str(get("matured_at")),
                 expected=float(get("expected")), realised=float(get("realised")), confidence=float(get("confidence")),
                 scale=float(get("scale", 0.0)), pattern=str(get("pattern", "")), sector=str(get("sector", "")),
                 regime=str(get("regime", "")), stock_type=str(get("stock_type", "")), market_weight=float(get("market_weight", 0.5)),
                 transfer=float(get("transfer", 0.5)), exit_regret=None if get("exit_regret") is None else float(get("exit_regret")))
    errs = o.check()
    if errs:
        raise ErrorResearchError("; ".join(errs))
    return o


# ------------------------------------------------------------------------------------------------------ D: intensity and depth

@dataclass(frozen=True)
class Intensity:
    obs_id: str
    z: float
    magnitude: float
    confidence: float
    repeatability: float
    potential: float
    market: float
    boost: float
    value: float
    tier: Tier
    confident_wrong: bool
    minutes: float

    def terms(self) -> dict:
        return {"magnitude": self.magnitude, "confidence": self.confidence, "repeatability": self.repeatability,
                "potential": self.potential, "market": self.market, "boost": self.boost}


def is_confident_wrong(o: ErrorObs, cfg: ErrorConfig) -> bool:
    """Checklist E trigger: substantially wrong AND sure. All three must hold; a big z on a trivially small miss does not count."""
    return o.confidence >= cfg.cw_conf and abs(o.z(cfg)) >= cfg.cw_z and abs(o.error) >= cfg.cw_abs


def magnitude_term(o: ErrorObs, cfg: ErrorConfig) -> float:
    """Size of the miss in [0, 1): the smaller of its statistical size (z beyond noise) and its economic size (against the +-1pp
    target). A big z on a 0.2pp miss and a 5pp miss inside a huge claimed sd are both small."""
    z_part = 1.0 - math.exp(-max(abs(o.z(cfg)) - cfg.z_tiny, 0.0) / cfg.z_sat)
    a = abs(o.error) / cfg.material_error
    abs_part = 0.0 if a <= 0.5 else 1.0 - math.exp(-(a - 0.5) / cfg.abs_sat)
    return min(z_part, abs_part)


def depth_tier(value: float, cfg: ErrorConfig) -> Tier:
    a, b, c = cfg.tier_cut
    return Tier.NONE if value < a else Tier.CHEAP if value < b else Tier.STANDARD if value < c else Tier.DEEP


def intensity(o: ErrorObs, cfg: ErrorConfig, escalation: float = 1.0) -> Intensity:
    """Research intensity = prediction error x confidence x repeatability x potential value x market significance (Checklist D),
    times the confident-wrong boost. `escalation` is the error-pattern multiplier (Checklist S) so repeated errors are worth more."""
    if escalation < 1.0 or not math.isfinite(escalation):
        raise ErrorResearchError("escalation multiplier must be a finite number >= 1")
    m = magnitude_term(o, cfg)
    conf = cfg.conf_floor + (1 - cfg.conf_floor) * o.confidence ** 2
    rep = cfg.rep_base + (1 - cfg.rep_base) * (1.0 - math.exp(-(escalation - 1.0)))
    pot = cfg.potential_floor + (1 - cfg.potential_floor) * _clip01(o.transfer)
    mkt = cfg.market_floor + (1 - cfg.market_floor) * _clip01(o.market_weight)
    cw = is_confident_wrong(o, cfg)
    boost = cfg.cw_boost if cw else 1.0
    val = m * conf * rep * pot * mkt * boost
    tier = depth_tier(val, cfg)
    return Intensity(o.obs_id, o.z(cfg), m, conf, rep, pot, mkt, boost, val, tier, cw, cfg.tier_minutes[TIER_ORDER.index(tier)])


def value_from_intensity(o: ErrorObs, it: Intensity, cfg: ErrorConfig, redundancy: float = 0.0) -> ExperimentValue | None:
    """The section-2 value vector for the priority engine. None when the tier is NONE: a tiny error gets no job at all."""
    if it.tier == Tier.NONE:
        return None
    bits = min(surprise_bits(it.z), 20.0)
    dec = _clip01(4.0 * it.value)
    return P.make_value(cost=it.minutes, information_bits=bits / 10.0, decision=dec, uncertainty=_clip01(it.magnitude * it.confidence),
                        transfer=_clip01(o.transfer), failure=_clip01(it.magnitude * it.confidence) if it.confident_wrong else 0.0,
                        loss=_clip01(it.magnitude * it.market) if o.error < 0 else 0.0, volatility=_clip01(it.magnitude * it.potential),
                        direction=_clip01(it.magnitude) if o.direction_wrong else None, overfit=_clip01(1.0 - it.repeatability),
                        redundancy=_clip01(redundancy))


def _safe_text(text: str, fallback: str) -> str:
    return fallback if identity_leak(text) else text


def item_for_error(o: ErrorObs, it: Intensity, cfg: ErrorConfig, now, redundancy: float = 0.0) -> P.ResearchItem | None:
    require_past(o.matured_at, now, f"item for {o.obs_id}")
    v = value_from_intensity(o, it, cfg, redundancy)
    if v is None:
        return None
    side = "over" if o.error < 0 else "under"
    what = "a confident prediction that failed" if it.confident_wrong else "a prediction error"
    text = _safe_text(f"Why did {what} ({side}-prediction, {it.tier.value.lower()} depth) happen"
                      + (f" in situation {o.cell()}" if o.cell() else "") + "?", f"Why did {what} happen?")
    family = "error_research/confident_wrong" if it.confident_wrong else "error_research/single_error"
    return P.ResearchItem(item_id="ERR-" + stable_hash({"o": o.obs_id, "t": it.tier.value}, 12), text=text,
                          problem=Problem.DIRECTION if (o.direction_wrong and it.confident_wrong) else Problem.VOLATILITY, family=family,
                          value=v, created=o.matured_at, target=ResearchTarget.FAILURE if it.confident_wrong else ResearchTarget.UNKNOWN_AREA,
                          real_data=it.tier == Tier.DEEP, ram_gb=2.0 if it.tier == Tier.DEEP else 1.0, subsystem="error_research",
                          tags=("error", o.cell() or "all", it.tier.value))


# ------------------------------------------------------------------------------------------------------ S: error patterns

@dataclass(frozen=True)
class GroupStats:
    group: str
    n: int
    mean_z: float
    t: float
    sign_k: int
    sign_p: float
    run: int
    evidence: float
    multiplier: float
    kind: BiasKind
    ratio: float | None            # sum(realised) / sum(expected) over positive expectations
    slope: float | None            # realised on expected
    mean_expected: float
    mean_realised: float
    half_life: float | None
    through: str
    members: str = ""              # hash of the window's record ids: groups with identical members are one finding
    aliases: tuple = ()            # the other lenses (groups) that hold exactly the same records: the same finding, other names

    @property
    def systematic(self) -> bool:
        return self.kind != BiasKind.NONE

    @property
    def lenses(self) -> tuple:
        """Every group name this finding is visible under (its label first)."""
        return (self.group,) + tuple(self.aliases)


# Which lens names a finding when several groups hold the SAME records (F11, 29 Sep): a compound group is the most specific; among
# single lenses the pattern is the most actionable (pattern influence is what research can change), then the stock type, the sector
# and the market regime; 'all' only when no narrower lens holds the same records. Before this, ties fell to the alphabet, so a
# stock-type label ('stock_type=...' > 'pattern=...' > 'all') named every shared finding and pattern / 'all' escalation never showed.
LENS_RANK = {"pattern": 4, "stock_type": 3, "sector": 2, "regime": 1, "all": 0}


def lens_rank(group: str) -> tuple:
    head = group.split("|")[0].split("=")[0]
    return (group.count("|"), LENS_RANK.get(head, 0))


def _neutral(n: int, group: str, through: str) -> GroupStats:
    return GroupStats(group, n, 0.0, 0.0, 0, 1.0, 0, 0.0, 1.0, BiasKind.NONE, None, None, 0.0, 0.0, None, through)


def group_stats(group: str, obs: Sequence[ErrorObs], cfg: ErrorConfig) -> GroupStats:
    """Is this group's error a PATTERN? Needs min_n errors, a t statistic on the standardised error beyond t0 and a same-sign
    run that a fair coin would rarely produce. Anything else has multiplier exactly 1.0 (one surprise may be noise)."""
    obs = sorted(obs, key=lambda o: (o.matured_at, o.obs_id))[-cfg.window:]
    n = len(obs)
    through = obs[-1].matured_at if obs else ""
    if n < cfg.min_n:
        return _neutral(n, group, through)
    zs = np.array([o.z(cfg) for o in obs])
    exp = np.array([o.expected for o in obs])
    real = np.array([o.realised for o in obs])
    mean_z = float(zs.mean())
    sd = max(float(zs.std(ddof=1)), 0.5)
    t = mean_z / (sd / math.sqrt(n))
    sgn = 1 if mean_z > 0 else -1
    dirs = np.where(np.abs(zs) >= cfg.z_neutral, np.sign(zs), 0)
    k = int((dirs == sgn).sum())
    m = int((dirs != 0).sum())
    p = float(sps.binomtest(k, m, 0.5, alternative="greater").pvalue) if m else 1.0
    run = 0
    for d in dirs[::-1]:
        if d == sgn:
            run += 1
        else:
            break
    pos = exp > 0
    ratio = float(real[pos].sum() / exp[pos].sum()) if pos.any() else None
    slope = float(np.polyfit(exp, real, 1)[0]) if n >= cfg.min_n and float(exp.std()) > 1e-9 else None
    material = abs(float((real - exp).mean())) >= 0.5 * cfg.material_error       # a consistent 0.2pp miss is not worth a job
    evidence = max(0.0, abs(t) - cfg.t0) if (p <= cfg.sign_p and abs(t) > cfg.t0 and material) else 0.0
    kind = BiasKind.NONE
    if evidence > 0:
        if mean_z > 0:
            kind = BiasKind.UNDER
        else:
            kind = BiasKind.SHRINK if (ratio is not None and ratio <= cfg.shrink_ratio) else BiasKind.OVER
    if evidence > 0 and slope is not None and slope < cfg.compressed_slope and kind == BiasKind.OVER:
        kind = BiasKind.COMPRESSED
    mult = 1.0
    if evidence > 0:
        mult = min(cfg.max_mult, 1.0 + cfg.gain * math.log1p(evidence) + cfg.run_gain * max(0, run - cfg.min_run + 1))
    days = [float(np.datetime64(o.matured_at, "D").astype(int)) for o in obs]
    hl = surprise_half_life(zs.tolist(), days)["half_life"] if n >= 8 else None
    return GroupStats(group, n, mean_z, float(t), k, p, run, evidence, float(mult), kind, ratio, slope, float(exp.mean()),
                      float(real.mean()), hl, through, stable_hash([o.obs_id for o in obs], 10))


class ErrorPatternBook:
    """The memory of matured errors, grouped by situation. Append-only, deduplicated by id, refuses anything not matured before
    `now` (the trader-side clock can never read an outcome from its own future)."""

    def __init__(self, cfg: ErrorConfig | None = None):
        self.cfg = cfg or ErrorConfig()
        self._obs: dict = {}

    def add(self, o: ErrorObs, now) -> bool:
        errs = o.check()
        if errs:
            raise ErrorResearchError("; ".join(errs))
        require_past(o.matured_at, now, f"error record {o.obs_id}")
        if o.obs_id in self._obs:
            if self._obs[o.obs_id] != o:
                raise ErrorResearchError(f"error record {o.obs_id} re-added with different content (records are immutable)")
            return False
        self._obs[o.obs_id] = o
        return True

    def records(self, now, group: str = "all") -> list:
        out = [o for o in self._obs.values() if str(o.matured_at) < str(now) and (group == "all" or group in o.groups())]
        return sorted(out, key=lambda o: (o.matured_at, o.obs_id))

    def __len__(self) -> int:
        return len(self._obs)

    def groups(self, now) -> list:
        return sorted({g for o in self.records(now) for g in o.groups()})

    def stats(self, now, group: str) -> GroupStats:
        return group_stats(group, self.records(now, group), self.cfg)

    def all_stats(self, now) -> dict:
        """Every group's pattern test, with the multiplicity across groups controlled (F11): every lens of every record is tested
        at once (patterns, sectors, regimes, stock types, pattern x regime, all - typically 20-30 groups), so a group's t beyond t0
        and a lucky sign run happened in 17 of 40 pure-noise books of 36 records. A group stays systematic only if its t-test p
        survives Benjamini-Hochberg over the whole family at cfg.fdr_q; otherwise it is neutral (multiplier exactly 1.0)."""
        key = (str(now), len(self._obs))
        if getattr(self, "_cache_key", None) == key:
            return dict(self._cache)
        recs = self.records(now)
        by: dict = {}
        for o in recs:
            for g in o.groups():
                by.setdefault(g, []).append(o)
        raw = {g: group_stats(g, v, self.cfg) for g, v in sorted(by.items())}
        tested = [g for g, st in raw.items() if st.n >= self.cfg.min_n]
        keep = benjamini_hochberg([float(2.0 * sps.t.sf(abs(raw[g].t), raw[g].n - 1)) for g in tested], self.cfg.fdr_q) if tested else []
        survive = {g for g, k in zip(tested, keep) if k}
        out = {g: (st if g in survive or not st.systematic else dataclasses.replace(st, evidence=0.0, multiplier=1.0, kind=BiasKind.NONE))
               for g, st in raw.items()}
        self._cache_key, self._cache = key, out
        return dict(out)

    def escalation(self, o: ErrorObs, now) -> tuple:
        """(largest multiplier over the groups this error belongs to, the GroupStats that produced it)."""
        fam = self.all_stats(now)
        best = None
        for g in o.groups():
            gs = fam.get(g) or self.stats(now, g)
            if best is None or gs.multiplier > best.multiplier:
                best = gs
        return (best.multiplier if best else 1.0), best

    def escalated(self, now) -> list:
        """Systematic error patterns, one per distinct set of records. The same records seen through several lenses are ONE finding,
        labelled by the most specific / most actionable lens (LENS_RANK) and carrying the other lenses as aliases, so a pattern-level
        or book-wide ('all') finding is never renamed to whichever label sorts last."""
        same: dict = {}
        for s in self.all_stats(now).values():
            if s.systematic:
                same.setdefault(s.members, []).append(s)
        out = []
        for grp in same.values():
            top = max(grp, key=lambda s: (s.multiplier, lens_rank(s.group), s.group))
            others = tuple(sorted((s.group for s in grp if s.group != top.group), key=lambda g: (tuple(-x for x in lens_rank(g)), g)))
            out.append(dataclasses.replace(top, aliases=others))
        return sorted(out, key=lambda s: (-s.multiplier, s.group))

    def escalated_lens(self, now, lens: str) -> list:
        """Findings visible under a lens kind ('pattern', 'stock_type', 'sector', 'regime' or 'all'), whatever label they carry."""
        def kind(g: str) -> str:
            return g.split("|")[0].split("=")[0]
        return [s for s in self.escalated(now) if any(kind(g) == lens for g in s.lenses)]


def hypothesis_for(gs: GroupStats) -> str:
    """The question the checklist tells us to ask, worded for the kind of pattern found."""
    where = f" ({gs.group})" if gs.group != "all" else ""
    if gs.kind == BiasKind.UNDER:
        return _safe_text(f"Is there a systematic missing factor causing underprediction of strong moves{where}?",
                          "Is there a systematic missing factor causing underprediction of strong moves?")
    if gs.kind == BiasKind.SHRINK:
        return _safe_text(f"What changed that is consistently suppressing the previously successful behavior{where}: expected moves "
                          f"realise only {gs.ratio:.0%} of their size?", "What changed that is consistently suppressing the previously "
                          "successful behavior?")
    if gs.kind == BiasKind.COMPRESSED:
        return _safe_text(f"Why does the realised move barely follow the expected move{where}?", "Why does the realised move barely follow "
                          "the expected move?")
    return _safe_text(f"Which assumption inflates the expectation and causes systematic overprediction{where}?",
                      "Which assumption inflates the expectation and causes systematic overprediction?")


def item_for_pattern(gs: GroupStats, cfg: ErrorConfig, now, potential: float = 0.6) -> P.ResearchItem | None:
    """A pattern job for the priority engine. Its information is the standardised evidence of the pattern; its cost is the STANDARD
    tier, or DEEP when the multiplier has passed 3 (many similar surprises: a deeper model failure is likely)."""
    if not gs.systematic:
        return None
    require_past(gs.through, now, f"pattern job for {gs.group}")
    deep = gs.multiplier >= 3.0
    minutes = cfg.tier_minutes[3 if deep else 2]
    sat = _clip01(1.0 - math.exp(-gs.evidence / 3.0))
    v = P.make_value(cost=minutes, information_bits=min(surprise_bits(gs.t), 20.0) / 10.0, decision=_clip01(sat * potential),
                     uncertainty=sat, transfer=_clip01(potential), failure=sat, loss=sat if gs.mean_z < 0 else 0.0,
                     volatility=sat, overfit=_clip01(1.0 - sat), redundancy=0.0)
    return P.ResearchItem(item_id="ERP-" + stable_hash({"g": gs.group, "k": gs.kind.value, "e": gs.through}, 12), text=hypothesis_for(gs),
                          problem=Problem.VOLATILITY, family="error_research/pattern", value=v, created=gs.through,
                          target=ResearchTarget.WEAK_PATTERN, real_data=deep, ram_gb=2.0 if deep else 1.0, subsystem="error_research",
                          tags=("pattern", gs.group, gs.kind.value))


# ------------------------------------------------------------------------------------------------------ E: the 15 questions

class FindingStatus(str, enum.Enum):
    ANSWERED = "ANSWERED"
    UNANSWERED = "UNANSWERED"            # no evidence supplied: never guessed
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWABLE = "UNKNOWABLE"            # classified as currently unknowable (question 14)
    OPEN = "OPEN"                        # handed to another lab as a research question (question 15)

    def __str__(self):
        return self.value


CHECKLIST_E = (
    "Which assumption failed?", "Which feature failed?", "Which pattern failed?", "Did the pattern itself change?",
    "Did another pattern override it?", "Did an interaction change?", "Did volatility change?", "Did market correlation change?",
    "Did sector behavior change?", "Did timing change?", "Did the optimal exit change?",
    "Was there evidence available before the prediction that could have revealed the failure?",
    "If yes, why did the system fail to recognize it?", "If no, classify the cause as currently unknowable.",
    "Can a precursor now be discovered for future detection?")


@dataclass(frozen=True)
class InvestigationContext:
    """What the research world knows about one failed prediction. Every field is optional: a missing one leaves its question
    UNANSWERED. `knowability` is a KnowabilityAssessment (duck typed: classification, information_that_would_have_been_available,
    explanations); `precursor_hits` are names a precursor search returned, None = the search was not run."""
    feature_shifts: Mapping[str, float] | None = None     # feature -> standardised shift between training and this prediction
    pattern_reliability_before: float | None = None
    pattern_reliability_after: float | None = None
    overriding_pattern: str | None = None
    interaction_change_z: float | None = None
    vol_ratio: float | None = None                         # realised / expected volatility
    corr_change: float | None = None                       # change in market correlation
    sector_error_share: float | None = None                # share of the sector that missed the same way
    timing_shift_days: float | None = None
    exit_regret: float | None = None
    model_inputs: Sequence[str] | None = None
    knowability: Any = None
    precursor_hits: Sequence[str] | None = None


@dataclass(frozen=True)
class Finding:
    number: int
    question: str
    status: FindingStatus
    answer: str = ""
    strength: float = 0.0              # how strongly the evidence supports the answer, [0, 1]


@dataclass(frozen=True)
class Investigation:
    event_id: str
    obs_id: str
    cell: str
    findings: tuple
    knowable: Knowability | None
    follow_ups: tuple                  # ResearchQuestion objects for what could not be answered
    causes: tuple                      # ((question number, answer, strength), ...) strongest first
    code_hash: str = ""

    def by_number(self, n: int) -> Finding:
        return self.findings[n - 1]

    @property
    def answered_share(self) -> float:
        applicable = [f for f in self.findings if f.status != FindingStatus.NOT_APPLICABLE]
        done = [f for f in applicable if f.status in (FindingStatus.ANSWERED, FindingStatus.UNKNOWABLE)]
        return len(done) / len(applicable) if applicable else 0.0

    def check(self) -> list:
        errs = []
        if [f.number for f in self.findings] != list(range(1, 16)):
            errs.append("an investigation must carry all 15 checklist-E questions in order")
        if self.by_number(12).status == FindingStatus.ANSWERED and self.by_number(12).answer.startswith("yes") \
                and self.by_number(14).status == FindingStatus.UNKNOWABLE:
            errs.append("cause classified unknowable although evidence existed before the prediction")
        return errs


def _class_of(kn: Any) -> Knowability | None:
    c = getattr(kn, "classification", None)
    if c is None:
        return None
    try:
        return Knowability(c)
    except ValueError:
        return None


def _f(n: int, status: FindingStatus, answer: str = "", strength: float = 0.0) -> Finding:
    return Finding(n, CHECKLIST_E[n - 1], status, answer, _clip01(strength))


def _threshold_finding(n: int, value: float | None, thr: float, yes: str, no: str, two_sided: bool = True) -> Finding:
    if value is None:
        return _f(n, FindingStatus.UNANSWERED)
    hit = abs(value) >= thr if two_sided else value >= thr
    strength = 1.0 - math.exp(-(abs(value) / thr - 1.0)) if hit else 0.0
    return _f(n, FindingStatus.ANSWERED, yes if hit else no, strength)


def investigate(o: ErrorObs, ctx: InvestigationContext, cfg: ErrorConfig, now, created_real: str = "", code_hash: str = "") -> Investigation:
    """Checklist E as data. Answers only from evidence in `ctx`; question 12 reads the knowability verdict (was the information
    there before the decision), 13 asks why it was not used, 14 classifies an unknowable cause without inventing one, 15 turns
    'can a precursor now be found' into a research question for the precursor / discovery lab unless a search has already run."""
    require_past(o.matured_at, now, f"investigation of {o.obs_id}")
    fs: dict = {}
    shifts = ctx.feature_shifts
    if shifts is None:
        fs[2] = _f(2, FindingStatus.UNANSWERED)
    elif not shifts:
        fs[2] = _f(2, FindingStatus.ANSWERED, "no feature shifted", 0.0)
    else:
        name, val = max(sorted(shifts.items()), key=lambda kv: abs(kv[1]))
        hit = abs(val) >= cfg.feature_shift_z
        fs[2] = _f(2, FindingStatus.ANSWERED, f"feature {name} shifted {val:+.1f} sd" if hit else "no feature shifted beyond noise",
                   1.0 - math.exp(-(abs(val) / cfg.feature_shift_z - 1.0)) if hit else 0.0)
    b, a = ctx.pattern_reliability_before, ctx.pattern_reliability_after
    if b is None or a is None:
        fs[3] = _f(3, FindingStatus.UNANSWERED)
        fs[4] = _f(4, FindingStatus.UNANSWERED)
    else:
        drop = b - a
        hit = drop >= cfg.reliability_drop
        fs[3] = _f(3, FindingStatus.ANSWERED, f"pattern {o.pattern or 'in use'} reliability fell {drop:.2f}" if hit else "pattern reliability held",
                   drop / max(b, 1e-9) if hit else 0.0)
        fs[4] = _f(4, FindingStatus.ANSWERED, "yes: reliability changed" if hit else "no: reliability within noise", drop / max(b, 1e-9) if hit else 0.0)
    fs[5] = (_f(5, FindingStatus.UNANSWERED) if ctx.overriding_pattern is None else
             _f(5, FindingStatus.ANSWERED, f"pattern {ctx.overriding_pattern} overrode it" if ctx.overriding_pattern else "no override found",
                0.6 if ctx.overriding_pattern else 0.0))
    fs[6] = _threshold_finding(6, ctx.interaction_change_z, cfg.feature_shift_z, "an interaction changed", "no interaction change")
    vr = None if ctx.vol_ratio is None else abs(math.log(max(ctx.vol_ratio, 1e-9)))
    fs[7] = _threshold_finding(7, vr, math.log(cfg.vol_ratio), f"volatility changed (ratio {ctx.vol_ratio})", "volatility as expected")
    fs[8] = _threshold_finding(8, ctx.corr_change, cfg.corr_shift, "market correlation changed", "market correlation unchanged")
    fs[9] = _threshold_finding(9, ctx.sector_error_share, cfg.sector_share, "the sector missed the same way", "sector not implicated",
                               two_sided=False)
    fs[10] = _threshold_finding(10, ctx.timing_shift_days, cfg.timing_days, "timing shifted", "timing as expected")
    fs[11] = _threshold_finding(11, ctx.exit_regret if ctx.exit_regret is not None else o.exit_regret, cfg.exit_regret_share,
                                "the optimal exit differed", "exit was near optimal", two_sided=False)
    kn = _class_of(ctx.knowability)
    avail = tuple(getattr(ctx.knowability, "information_that_would_have_been_available", ()) or ())
    if kn is None:
        fs[12] = _f(12, FindingStatus.UNANSWERED)
        fs[13] = _f(13, FindingStatus.UNANSWERED)
        fs[14] = _f(14, FindingStatus.UNANSWERED)
    elif kn in KNOWABLE_CLASSES or avail:
        fs[12] = _f(12, FindingStatus.ANSWERED, f"yes: {kn.value}; {len(avail)} item(s) were available", 0.7 if kn == Knowability.PREDICTABLE else 0.4)
        if ctx.model_inputs is None:
            fs[13] = _f(13, FindingStatus.UNANSWERED)
        else:
            unused = [i for i in avail if i not in set(ctx.model_inputs)]
            fs[13] = _f(13, FindingStatus.ANSWERED, f"not a model input: {', '.join(sorted(unused))}" if unused else
                        "was an input but its weight or threshold was too weak to act", 0.6 if unused else 0.3)
        fs[14] = _f(14, FindingStatus.NOT_APPLICABLE, "evidence existed before the prediction")
    else:
        fs[12] = _f(12, FindingStatus.ANSWERED, f"no: {kn.value}", 0.5)
        fs[13] = _f(13, FindingStatus.NOT_APPLICABLE, "nothing was available to recognise")
        fs[14] = _f(14, FindingStatus.UNKNOWABLE, f"currently unknowable: {kn.value}", 0.5)
    follow: list = []
    if not (kn in UNKNOWABLE_CLASSES and not avail):
        if ctx.precursor_hits is None:
            fs[15] = _f(15, FindingStatus.OPEN, "precursor search not run yet: queued for the precursor / discovery lab", 0.0)
            follow.append(ResearchQuestion.make(
                _safe_text(f"Can a leading indicator of the failure of {o.pattern or 'a prediction pattern'} be discovered from information "
                           "available before the decision?", "Can a leading indicator of a prediction failure be discovered?"),
                "confident_wrong", Problem.VOLATILITY, created_real or str(now), o.matured_at,
                "a precursor with an out-of-sample effect that survives multiple-testing control",
                "no precursor survives out-of-sample replication"))
        else:
            hits = tuple(sorted(ctx.precursor_hits))
            fs[15] = _f(15, FindingStatus.ANSWERED, f"precursor candidates: {', '.join(hits)}" if hits else "search found no precursor",
                        min(1.0, 0.3 * len(hits)))
    else:
        fs[15] = _f(15, FindingStatus.NOT_APPLICABLE, "cause classified unknowable; a precursor search would only mine noise")
    causes = tuple(sorted(((n, fs[n].answer, fs[n].strength) for n in range(2, 12)
                           if fs[n].status == FindingStatus.ANSWERED and fs[n].strength > 0), key=lambda c: (-c[2], c[0])))
    if causes:
        fs[1] = _f(1, FindingStatus.ANSWERED, f"strongest evidence: {causes[0][1]}", causes[0][2])
    elif all(fs[n].status == FindingStatus.UNANSWERED for n in range(2, 12)):
        fs[1] = _f(1, FindingStatus.UNANSWERED)
    else:
        fs[1] = _f(1, FindingStatus.ANSWERED, "no component shows a failure beyond noise: possibly an isolated miss", 0.0)
    for n in range(1, 16):
        if fs[n].status == FindingStatus.UNANSWERED and n not in (12, 13, 14):
            follow.append(ResearchQuestion.make(_safe_text(f"{CHECKLIST_E[n - 1]} (evidence still missing for {o.cell() or 'this situation'})",
                                                           CHECKLIST_E[n - 1]), "confident_wrong", Problem.VOLATILITY, created_real or str(now),
                                                o.matured_at, "the question is answered from point-in-time evidence",
                                                "no evidence can be assembled without future information"))
    inv = Investigation("CW-" + stable_hash({"o": o.obs_id}, 12), o.obs_id, o.cell(), tuple(fs[n] for n in range(1, 16)), kn,
                        tuple(follow), causes, code_hash)
    errs = inv.check()
    if errs:
        raise ErrorResearchError("; ".join(errs))
    return inv


# ------------------------------------------------------------------------------------------------------ U: compute allocation

class DepthLedger:
    """Where deep research has been spent and what it returned. A cell whose last investigations improved nothing loses priority;
    a cell whose cause is demonstrably unknowable is nearly closed; one good result reopens it (never punished for its past)."""

    def __init__(self, cfg: ErrorConfig | None = None):
        self.cfg = cfg or ErrorConfig()
        self.improvements: dict = {}
        self.unknowable: dict = {}

    def observe(self, cell: str, improvement: float) -> None:
        if not math.isfinite(improvement) or improvement < 0:
            raise ErrorResearchError("improvement must be a non-negative finite number")
        self.improvements.setdefault(cell, []).append(float(improvement))

    def mark(self, cell: str, inv: Investigation) -> None:
        if inv.knowable in UNKNOWABLE_CLASSES and inv.by_number(14).status == FindingStatus.UNKNOWABLE:
            self.unknowable[cell] = inv.knowable.value
        elif inv.knowable in KNOWABLE_CLASSES:
            self.unknowable.pop(cell, None)

    def barren_streak(self, cell: str) -> int:
        n = 0
        for v in reversed(self.improvements.get(cell, [])):
            if v >= self.cfg.barren_eps:
                break
            n += 1
        return n

    def multiplier(self, cell: str) -> float:
        m = self.cfg.unknowable_mult if cell in self.unknowable else 1.0
        return max(self.cfg.barren_floor, m * math.exp(-self.cfg.barren_decay * self.barren_streak(cell)))

    def redundancy(self, cell: str) -> float:
        return _clip01(1.0 - 1.0 / (1.0 + len(self.improvements.get(cell, []))))


# ------------------------------------------------------------------------------------------------------ T: self-research

@dataclass(frozen=True)
class SelfQuestion:
    number: int
    key: str
    question: ResearchQuestion
    item: P.ResearchItem
    finding: Mapping
    insufficient: bool


SELF_QUESTIONS = (
    ("errors_rising", "Which prediction errors are increasing?"), ("patterns_degrading", "Which patterns are degrading?"),
    ("systematic_bias", "Which predictions are systematically biased?"), ("confidence_miscalibrated", "Which confidence levels are miscalibrated?"),
    ("worst_regime", "Which regimes produce the worst errors?"), ("worst_stock_type", "Which stock types produce the worst errors?"),
    ("worst_sector", "Which sectors produce the worst errors?"), ("exit_regret", "Which exit decisions produce the largest missed gains?"),
    ("discoveries_oos", "Which research discoveries actually improve out-of-sample performance?"),
    ("barren_approaches", "Which research approaches repeatedly produce nothing?"), ("what_next", "What should be researched next?"))


def _mean_abs_z_by(recs: Sequence[ErrorObs], dim: str, cfg: ErrorConfig) -> dict:
    by: dict = {}
    for o in recs:
        v = getattr(o, dim)
        if v:
            by.setdefault(v, []).append(abs(o.z(cfg)))
    return {k: (len(v), float(np.mean(v))) for k, v in sorted(by.items()) if len(v) >= 3}


def _worst_dim(recs: Sequence[ErrorObs], dim: str, cfg: ErrorConfig) -> dict:
    tab = _mean_abs_z_by(recs, dim, cfg)
    if len(tab) < 2:
        return {"insufficient": True, "p": 1.0, "worst": None}
    worst = max(tab, key=lambda k: (tab[k][1], k))
    groups = [[abs(o.z(cfg)) for o in recs if getattr(o, dim) == k] for k in tab]
    p = float(sps.kruskal(*groups).pvalue) if all(len(g) >= 3 for g in groups) and len(set(np.concatenate(groups))) > 1 else 1.0
    return {"insufficient": False, "worst": worst, "mean_abs_z": tab[worst][1], "n": tab[worst][0], "table": tab, "p": p}


def self_findings(book: ErrorPatternBook, now, results: Sequence[Any] = (), priority_state: P.PriorityState | None = None) -> dict:
    """The measured answers behind the eleven self-research questions. Each is computed from matured errors only; a statistic
    that needs more data than exists is reported as insufficient, not as a finding."""
    cfg = book.cfg
    recs = book.records(now)
    out: dict = {}
    az = np.array([abs(o.z(cfg)) for o in recs])
    if len(recs) >= cfg.self_min_n:
        tau, p = sps.kendalltau(np.arange(len(az)), az)
        p = 1.0 if math.isnan(p) else float(p)
        out["errors_rising"] = {"insufficient": False, "tau": float(0 if math.isnan(tau) else tau), "p": p / 2 if tau > 0 else 1.0}
    else:
        out["errors_rising"] = {"insufficient": True, "p": 1.0}
    deg, checked = [], 0
    for pat in sorted({o.pattern for o in recs if o.pattern}):
        rs = [abs(o.z(cfg)) for o in recs if o.pattern == pat]
        if len(rs) >= 8:
            checked += 1
            h = len(rs) // 2
            p = float(sps.mannwhitneyu(rs[h:], rs[:h], alternative="greater").pvalue)
            if p <= cfg.self_alpha:
                deg.append((pat, p))
    out["patterns_degrading"] = {"insufficient": checked == 0, "degrading": sorted(deg, key=lambda t: (t[1], t[0])), "p": min([p for _, p in deg], default=1.0)}
    esc = book.escalated(now)
    out["systematic_bias"] = {"insufficient": len(recs) < cfg.min_n, "groups": [(s.group, s.kind.value, round(s.multiplier, 3)) for s in esc[:5]],
                              "p": float(min([s.sign_p for s in esc], default=1.0))}
    if len(recs) >= cfg.self_min_n:
        conf = np.array([o.confidence for o in recs])
        rho, p = sps.spearmanr(conf, az) if conf.std() > 0 and az.std() > 0 else (0.0, 1.0)
        out["confidence_miscalibrated"] = {"insufficient": False, "rho": float(rho), "p": float(p) / 2 if rho >= 0 else 1.0,
                                           "informative": bool(rho < 0 and p < cfg.self_alpha)}
    else:
        out["confidence_miscalibrated"] = {"insufficient": True, "p": 1.0}
    out["worst_regime"] = _worst_dim(recs, "regime", cfg)
    out["worst_stock_type"] = _worst_dim(recs, "stock_type", cfg)
    out["worst_sector"] = _worst_dim(recs, "sector", cfg)
    rg = {}
    for o in recs:
        if o.exit_regret is not None and o.pattern:
            rg.setdefault(o.pattern, []).append(o.exit_regret)
    rg = {k: (len(v), float(np.mean(v))) for k, v in sorted(rg.items()) if len(v) >= 3}
    out["exit_regret"] = ({"insufficient": True, "p": 1.0} if not rg else
                          {"insufficient": False, "worst": max(rg, key=lambda k: (rg[k][1], k)), "table": rg, "p": 1.0 / (1.0 + max(v[1] for v in rg.values()) * 10)})
    judged = [r for r in results if getattr(r, "survived_oos", None) is not None]
    surv = sum(1 for r in judged if r.survived_oos)
    out["discoveries_oos"] = ({"insufficient": True, "p": 1.0} if len(judged) < 3 else
                              {"insufficient": False, "n": len(judged), "survived": surv, "rate": surv / len(judged),
                               "p": float(sps.binomtest(surv, len(judged), 0.5, alternative="less").pvalue)})
    rep = priority_state.waste.report() if priority_state is not None else {}
    barren = sorted(f for f, r in rep.items() if r["verdict"] == "STOP")
    out["barren_approaches"] = {"insufficient": not rep, "barren": barren, "p": 0.05 if barren else 1.0}
    return out


def self_research_questions(book: ErrorPatternBook, now, created_real: str, results: Sequence[Any] = (),
                            priority_state: P.PriorityState | None = None, minutes: float = 10.0) -> list:
    """Checklist T: the eleven questions as ResearchQuestion + ResearchItem pairs. The information of each rides on how strongly the
    measured evidence points at a problem (-log2 p); an insufficient-data question stays queued but cheap and low value. The 11th
    picks the most promising of the first ten - it is answered by ranking, not by another test."""
    cfg = book.cfg
    fnd = self_findings(book, now, results, priority_state)
    recs = book.records(now)
    through = recs[-1].matured_at if recs else str(now)
    out: list = []
    scores: dict = {}
    for i, (key, text) in enumerate(SELF_QUESTIONS[:10], 1):
        f = fnd[key]
        insufficient = bool(f["insufficient"])
        bits = 0.0 if insufficient else min(-math.log2(max(f["p"], 1e-6)), 12.0)
        scores[key] = bits
        detail = "" if insufficient else " Evidence so far points at: " + _describe(key, f)
        out.append(_self_pair(i, key, text + detail, bits, insufficient, f, cfg, now, created_real, through, minutes))
    nxt = max(scores, key=lambda k: (scores[k], k)) if scores and max(scores.values()) > 0 else None
    f = {"insufficient": nxt is None, "next": nxt, "scores": {k: round(v, 3) for k, v in scores.items()}}
    text = SELF_QUESTIONS[10][1] + (f" The measured evidence is strongest for: {nxt}." if nxt else "")
    out.append(_self_pair(11, SELF_QUESTIONS[10][0], text, scores[nxt] if nxt else 0.0, nxt is None, f, cfg, now, created_real, through, minutes))
    return out


def _describe(key: str, f: Mapping) -> str:
    if key == "errors_rising":
        return f"errors trending up (tau {f['tau']:+.2f})"
    if key == "patterns_degrading":
        return f"{len(f['degrading'])} degrading pattern(s)"
    if key == "systematic_bias":
        return f"{len(f['groups'])} systematic group(s)"
    if key == "confidence_miscalibrated":
        return f"confidence vs error rank correlation {f['rho']:+.2f}"
    if key == "discoveries_oos":
        return f"only {f['survived']} of {f['n']} discoveries held out of sample"
    if key == "barren_approaches":
        return f"{len(f['barren'])} barren research family(ies)"
    if key == "exit_regret":
        return "exit regret concentrated in one pattern"
    return f"one {key.replace('worst_', '')} stands out"


def _self_pair(number: int, key: str, text: str, bits: float, insufficient: bool, finding: Mapping, cfg: ErrorConfig, now, created_real: str,
               through: str, minutes: float) -> SelfQuestion:
    text = _safe_text(text, SELF_QUESTIONS[number - 1][1])
    strength = _clip01(1.0 - math.exp(-bits / 4.0))
    v = P.make_value(cost=minutes if not insufficient else max(1.0, minutes / 4.0), information_bits=bits / 10.0,
                     decision=0.05 + 0.5 * strength, uncertainty=0.1 + 0.7 * strength, transfer=0.5, failure=0.5 * strength,
                     volatility=0.3 * strength, overfit=0.2, redundancy=0.0 if not insufficient else 0.3)
    q = ResearchQuestion.make(text, "self_research", Problem.RESEARCH_PROCESS, created_real or str(now), through,
                              "the measured statistic is significant after multiple-testing control and points to an actionable cause",
                              "the statistic is indistinguishable from chance on fresh data", expected=v)
    item = P.ResearchItem(item_id="ERT-" + stable_hash({"k": key, "e": through}, 12), text=text, problem=Problem.RESEARCH_PROCESS,
                          family="error_research/self", value=v, created=through, target=ResearchTarget.UNKNOWN_AREA,
                          subsystem="error_research", question_id=q.question_id, tags=("self", key))
    return SelfQuestion(number, key, q, item, dict(finding), insufficient)


# ------------------------------------------------------------------------------------------------------ the public entry

@dataclass
class ErrorResearchState:
    cfg: ErrorConfig
    book: ErrorPatternBook
    depth: DepthLedger
    investigations: dict = field(default_factory=dict)
    intensities: dict = field(default_factory=dict)
    item_cells: dict = field(default_factory=dict)       # item id -> cell, so finished results reach the depth ledger
    namespace: Namespace = Namespace.MATURED_RESEARCH
    label: str = LABEL


def new_state(cfg: ErrorConfig | None = None) -> ErrorResearchState:
    cfg = cfg or ErrorConfig()
    errs = cfg.check()
    if errs:
        raise ErrorResearchError("; ".join(errs))
    return ErrorResearchState(cfg, ErrorPatternBook(cfg), DepthLedger(cfg))


@dataclass(frozen=True)
class StepReport:
    now: str
    ingested: int
    tiers: Mapping
    tiny_ids: tuple
    confident_wrong: tuple
    investigations: tuple
    escalated: tuple
    items: tuple
    self_questions: tuple
    plan: Any
    minutes_requested: float

    def summary(self) -> str:
        return (f"{self.now}: {self.ingested} new errors, tiers {dict(self.tiers)}, {len(self.confident_wrong)} confident-wrong, "
                f"{len(self.escalated)} escalated groups, {len(self.items)} jobs ({self.minutes_requested:.0f} cpu-min requested)")


def step(state: ErrorResearchState, now, records: Iterable[Any], contexts: Mapping[str, InvestigationContext] | None = None,
         priority_state: P.PriorityState | None = None, results: Sequence[Any] = (), budget: ComputeBudget | None = None, seed: int = 0,
         created_real: str = "", with_self_questions: bool = True) -> StepReport:
    """THE public entry (rule 25). Ingest newly matured errors, score their intensity (with the pattern-book escalation), open
    the 15-question investigation for each confident-wrong one, add one job per systematic error pattern and the eleven
    self-research questions, apply the unknowable / barren multipliers, and hand everything to the EXISTING priority engine.
    Deterministic; refuses any record that has not matured strictly before `now`."""
    cfg = state.cfg
    contexts = contexts or {}
    new: list = []
    for r in sorted((obs_from_record(x) for x in records), key=lambda o: (o.matured_at, o.obs_id)):
        if state.book.add(r, now):
            new.append(r)
    for r in results:                       # finished jobs teach the depth ledger before this round is planned
        cell = state.item_cells.get(r.item.item_id)
        if cell is not None:
            state.depth.observe(cell, 0.0 if r.survived_oos is False else float(r.realised.decision_value or 0.0))
    tiers = {t.value: 0 for t in TIER_ORDER}
    items: list = []
    tiny: list = []
    cw: list = []
    invs: list = []
    for o in new:
        esc, gs = state.book.escalation(o, now)
        it = intensity(o, cfg, esc)
        state.intensities[o.obs_id] = it
        tiers[it.tier.value] += 1
        cell = o.cell() or "all"
        if it.tier == Tier.NONE:
            tiny.append(o.obs_id)
            continue
        if it.confident_wrong:
            cw.append(o.obs_id)
            inv = investigate(o, contexts.get(o.obs_id, InvestigationContext()), cfg, now, created_real)
            state.investigations[o.obs_id] = inv
            state.depth.mark(cell, inv)
            invs.append(inv)
        item = item_for_error(o, it, cfg, now, state.depth.redundancy(cell))
        if item is not None:
            items.append(item)
            state.item_cells[item.item_id] = cell
    esc_groups = state.book.escalated(now)
    for gs in esc_groups:
        pit = item_for_pattern(gs, cfg, now)
        if pit is not None:
            items.append(pit)
            state.item_cells[pit.item_id] = gs.group
    selfq: list = []
    if with_self_questions and len(state.book):
        selfq = self_research_questions(state.book, now, created_real, results, priority_state)
        items.extend(s.item for s in selfq)
    seen: set = set()
    items = [i for i in items if not (i.item_id in seen or seen.add(i.item_id))]
    plan = None
    if priority_state is not None and items:
        for i in items:
            cell = state.item_cells.get(i.item_id)
            m = state.depth.multiplier(cell) if cell is not None else 1.0
            if m < 1.0:
                priority_state.external_multipliers[i.item_id] = m
        plan = P.step(priority_state, now, items, results, budget, seed)
    return StepReport(str(now), len(new), tiers, tuple(tiny), tuple(cw), tuple(invs), tuple(esc_groups), tuple(items), tuple(selfq), plan,
                      float(sum(i.cost for i in items)))
