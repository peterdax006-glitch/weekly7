"""Situation representation (contract C62 section 15; supports sections 8, 16, 17; canon C56, C58, C63).

A Situation describes WHAT IS HAPPENING at a decision timestamp, never WHICH stock, WHICH date or WHICH year. It is a
tuple of typed blocks (market, sector, stock type, volatility, liquidity, trend, breadth, macro, regime, recent shock,
correlation structure, horizon, seasonality, pattern interaction, position/risk). Every block field has a FieldSpec
(kind, scale, sanity range, bin edges) that is shared with the similarity and context modules, so "how far apart" and
"which bucket" are defined in exactly one place.

Design rules enforced here:
* built only from data dated at/before `now`; anything later raises FirewallBreach (fail closed);
* macro observations are used only once PUBLISHED (period end + lag), never at their observation date;
* stock-level quantities are cross-sectional percentile ranks of the decision day, so a price level, a split or an era's
  volume scale cannot leak an identity; the rank mode is recorded because ranks and absolute fallbacks must not be mixed;
* a missing input stays None (UNKNOWN); it is never filled with 0 or a mean;
* seasonality is legitimate calendar structure only (month-of-quarter, turn-of-month), never a year or a date;
* ticker / date / year scrambling helpers exist so the identity tests (Tests 8, 9, 10) can prove invariance.

Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
import re
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import FirewallBreach, Layer, Unknown, as_date, stable_hash

SCHEMA_VERSION = "situation.v1"


# ------------------------------------------------------------------------------------------------ field specs

@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """kind: 'num' (continuous, distance = |a-b|/scale), 'ord' (ordered levels), 'cat' (unordered levels).
    bins: for 'num', the interior edges that define its pooling bucket (b0 .. bN)."""
    name: str
    kind: str
    scale: float = 1.0
    lo: float | None = None
    hi: float | None = None
    bins: tuple[float, ...] = ()
    levels: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in ("num", "ord", "cat"):
            raise ValueError(f"FieldSpec {self.name}: unknown kind {self.kind!r}")
        if self.kind == "num" and not self.scale > 0:
            raise ValueError(f"FieldSpec {self.name}: scale must be positive")
        if self.kind in ("ord", "cat") and len(self.levels) < 2:
            raise ValueError(f"FieldSpec {self.name}: needs at least two levels")
        if list(self.bins) != sorted(self.bins):
            raise ValueError(f"FieldSpec {self.name}: bin edges must ascend")

    def check(self, v) -> str | None:
        if v is None:
            return None
        if self.kind == "num":
            if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
                return f"{self.name}={v!r} not a finite number"
            if self.lo is not None and float(v) < self.lo - 1e-9:
                return f"{self.name}={v!r} below {self.lo}"
            if self.hi is not None and float(v) > self.hi + 1e-9:
                return f"{self.name}={v!r} above {self.hi}"
            return None
        if v not in self.levels:
            return f"{self.name}={v!r} not in {self.levels}"
        return None

    def bin_of(self, v) -> str:
        """Pooling bucket label; 'na' for missing. Never raises on valid input."""
        if v is None:
            return "na"
        if self.kind == "num":
            return f"b{int(np.searchsorted(self.bins, float(v), side='right'))}" if self.bins else "b0"
        return str(v)

    def distance(self, a, b) -> float | None:
        """Normalised distance in [0, inf); None when either side is missing (never a fabricated 0 or 1)."""
        if a is None or b is None:
            return None
        if self.kind == "num":
            return abs(float(a) - float(b)) / self.scale
        if self.kind == "ord":
            return abs(self.levels.index(a) - self.levels.index(b)) / max(len(self.levels) - 1, 1) * 2.0
        return 0.0 if a == b else 1.0


def N(name, scale, bins=(), lo=None, hi=None):
    return FieldSpec(name, "num", scale=scale, lo=lo, hi=hi, bins=tuple(bins))


def O(name, levels):
    return FieldSpec(name, "ord", levels=tuple(levels))


def C(name, levels):
    return FieldSpec(name, "cat", levels=tuple(levels))


REGIMES = ("bull_calm", "bull_volatile", "correction_calm", "bear_volatile", "stress", "unknown")
SECTOR_FAMILIES = ("resources", "manufacturing", "utilities_transport", "trade_consumer", "finance_realestate",
                   "technology", "healthcare", "services_other", "unknown")
STOCK_TYPES = ("lottery", "quiet_large", "momentum", "reversal_prone", "event_driven", "ordinary")
SHOCK_KINDS = ("none", "stock_gap", "earnings", "event", "market", "red_flag")

BLOCK_SPECS: dict[str, tuple[FieldSpec, ...]] = {
    "market": (
        N("spy_ma200", 0.06, (-0.08, -0.02, 0.02, 0.08), -1, 3),
        N("spy_ma50", 0.04, (-0.04, -0.01, 0.01, 0.04), -1, 3),
        N("spy_r5", 0.02, (-0.03, -0.01, 0.01, 0.03), -1, 1),
        N("vix", 6.0, (14, 18, 24, 32), 0, 200),
        N("vix_term", 0.1, (0.85, 0.95, 1.0, 1.1), 0, 5),
        N("vix_chg5", 0.15, (-0.15, -0.03, 0.03, 0.15), -3, 3),
    ),
    "sector": (
        C("family", SECTOR_FAMILIES),
        N("strength20", 0.25, (0.2, 0.4, 0.6, 0.8), 0, 1),
        N("strength60", 0.25, (0.2, 0.4, 0.6, 0.8), 0, 1),
        N("rel20", 0.05, (-0.05, -0.01, 0.01, 0.05), -3, 3),
        N("rel60", 0.08, (-0.08, -0.02, 0.02, 0.08), -3, 3),
    ),
    "stock_type": (
        C("kind", STOCK_TYPES),
        N("lottery_rank", 0.25, (0.2, 0.4, 0.6, 0.8), 0, 1),
        N("skew60", 0.8, (-0.8, -0.2, 0.2, 0.8), -20, 20),
        N("persistence", 0.15, (-0.1, -0.02, 0.02, 0.1), -1, 1),
        N("event_load", 1.0, (0.5, 1.5, 2.5), 0, 10),
    ),
    "volatility": (
        N("vol_rank", 0.25, (0.2, 0.4, 0.6, 0.8), 0, 1),
        N("atr_pct", 0.02, (0.01, 0.02, 0.035, 0.06), 0, 1),
        N("vol_ratio", 0.3, (0.8, 0.95, 1.1, 1.4), 0, 20),
        N("range_compress", 0.25, (0.5, 0.7, 0.9, 1.1), 0, 20),
        C("state", ("expanding", "contracting", "stable")),
    ),
    "liquidity": (
        N("dv_rank", 0.25, (0.2, 0.4, 0.6, 0.8), 0, 1),
        N("vol_surge1", 0.6, (0.7, 0.9, 1.3, 2.0), 0, 500),
        N("vol_surge5", 0.4, (0.8, 0.95, 1.2, 1.7), 0, 500),
        C("state", ("thin", "normal", "deep", "surging")),
    ),
    "trend": (
        N("r5", 0.03, (-0.05, -0.015, 0.015, 0.05), -3, 3),
        N("r20", 0.06, (-0.10, -0.03, 0.03, 0.10), -3, 3),
        N("r60", 0.12, (-0.20, -0.05, 0.05, 0.20), -3, 3),
        N("mom_12_1", 0.25, (-0.25, -0.05, 0.05, 0.25), -5, 5),
        N("dist_ma50", 0.05, (-0.08, -0.02, 0.02, 0.08), -1, 5),
        N("dist_ma200", 0.10, (-0.15, -0.04, 0.04, 0.15), -1, 10),
        N("dist_52wh", 0.10, (-0.30, -0.15, -0.05, -0.01), -1, 0.5),
        C("state", ("up", "down", "flat", "reversal_up", "reversal_down")),
    ),
    "breadth": (
        N("breadth", 0.15, (0.3, 0.45, 0.6, 0.75), 0, 1),
        N("dispersion", 0.004, (0.008, 0.011, 0.015, 0.02), 0, 1),
        C("state", ("narrow", "healthy", "washed_out", "euphoric")),
    ),
    "macro": (
        N("rates", 1.0, (1.0, 2.5, 4.0, 5.5), -2, 25),
        N("curve", 0.5, (-0.5, 0.0, 0.5, 1.2), -5, 5),
        N("credit", 1.0, (3.0, 4.0, 5.0, 7.0), 0, 30),
        N("fin_stress", 0.8, (-0.5, 0.0, 0.5, 1.5), -5, 20),
        O("recession", ("no", "yes")),
        N("staleness_days", 30.0, (30, 60, 120), 0, 5000),
        C("state", ("easy", "neutral", "tight", "stress", "unknown")),
    ),
    "regime": (
        C("label", REGIMES),
        O("vol_regime", ("low", "mid", "high", "crisis")),
        O("trend_regime", ("down", "flat", "up")),
    ),
    "shock": (
        C("kind", SHOCK_KINDS),
        N("gap_today", 0.02, (-0.04, -0.01, 0.01, 0.04), -1, 1),
        N("max20", 0.03, (0.02, 0.04, 0.07, 0.12), -1, 2),
        N("min20", 0.03, (-0.12, -0.07, -0.04, -0.02), -2, 1),
        N("days_since_earn", 20.0, (2, 5, 20, 60), 0, 5000),
        O("red_flag", ("no", "yes")),
        N("market_shock", 1.0, (0.5, 1.0, 2.0), 0, 50),
    ),
    "correlation": (
        N("avg_pair_corr", 0.15, (0.1, 0.25, 0.4, 0.6), -1, 1),
        N("top_eig_share", 0.15, (0.15, 0.3, 0.45, 0.65), 0, 1),
        N("corr_to_mkt", 0.3, (0.0, 0.3, 0.6, 0.8), -1, 1),
        N("beta", 0.5, (0.5, 0.9, 1.2, 1.7), -5, 10),
        C("state", ("dispersed", "normal", "herded")),
    ),
    "horizon": (
        N("days", 3.0, (2, 5, 10, 20), 1, 260),
        C("state", ("short", "week", "multi_week")),
    ),
    "seasonality": (
        O("month_of_quarter", ("1", "2", "3")),
        O("turn_of_month", ("no", "yes")),
        O("quarter_end", ("no", "yes")),
        O("january", ("no", "yes")),
    ),
    "pattern_interaction": (
        N("n_active", 2.0, (0.5, 1.5, 3.5, 6.5), 0, 500),
        N("agreement", 0.3, (-0.3, 0.3, 0.7), -1, 1),
        N("top_score", 1.0, (0.5, 1.0, 2.0), -50, 50),
        N("score_spread", 1.0, (0.3, 1.0, 2.0), 0, 100),
        C("mix", ("none", "single", "agree", "conflict")),
    ),
    "position_risk": (
        N("n_positions", 3.0, (0.5, 3.5, 7.5), 0, 500),
        N("gross", 0.3, (0.25, 0.6, 0.95), 0, 5),
        N("drawdown", 0.04, (-0.10, -0.05, -0.02), -1, 0),
        N("port_vol", 0.01, (0.01, 0.02, 0.035), 0, 1),
        N("position_weight", 0.05, (0.05, 0.12, 0.25), 0, 1),
        N("risk_used", 0.3, (0.3, 0.7, 1.0), 0, 20),
        C("state", ("flat", "normal", "stretched", "drawdown")),
    ),
}
BLOCK_ORDER = tuple(BLOCK_SPECS)
_SPEC_INDEX = {k: {s.name: s for s in v} for k, v in BLOCK_SPECS.items()}


def spec_of(path: str) -> FieldSpec:
    block, _, field = path.partition(".")
    try:
        return _SPEC_INDEX[block][field]
    except KeyError as e:
        raise KeyError(f"unknown situation dimension {path!r}") from e


def all_paths() -> tuple[str, ...]:
    return tuple(f"{b}.{s.name}" for b, specs in BLOCK_SPECS.items() for s in specs)


# ------------------------------------------------------------------------------------------------ block

@dataclasses.dataclass(frozen=True)
class Block:
    kind: str
    values: tuple[tuple[str, Any], ...]

    @classmethod
    def make(cls, kind: str, /, **vals) -> "Block":
        if kind not in BLOCK_SPECS:
            raise ValueError(f"unknown block {kind!r}")
        names = [s.name for s in BLOCK_SPECS[kind]]
        extra = sorted(set(vals) - set(names))
        if extra:
            raise ValueError(f"block {kind}: unknown fields {extra}")
        out = []
        for s in BLOCK_SPECS[kind]:
            v = vals.get(s.name)
            if v is not None and s.kind == "num":
                v = clean_number(v)
            out.append((s.name, v))
        return cls(kind, tuple(out))

    @functools.cached_property
    def _map(self) -> dict:
        return dict(self.values)

    def get(self, name: str, default=None):
        v = self._map.get(name)
        return default if v is None else v

    def as_dict(self) -> dict:
        return dict(self.values)

    def observed(self) -> int:
        return sum(v is not None for _, v in self.values)

    def coverage(self) -> float:
        return self.observed() / max(len(self.values), 1)

    def bins(self) -> dict[str, str]:
        return {f"{self.kind}.{s.name}": s.bin_of(self._map.get(s.name)) for s in BLOCK_SPECS[self.kind]}

    def validate(self) -> list[str]:
        errs = []
        specs = BLOCK_SPECS.get(self.kind)
        if specs is None:
            return [f"unknown block {self.kind!r}"]
        if [n for n, _ in self.values] != [s.name for s in specs]:
            errs.append(f"block {self.kind}: field order/names differ from the spec")
            return errs
        for s, (_, v) in zip(specs, self.values):
            e = s.check(v)
            if e:
                errs.append(f"{self.kind}.{e}")
        return errs


def clean_number(x) -> float | None:
    """Finite float rounded to 6 places, else None. Booleans are refused: a flag belongs in an 'ord' field."""
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, 6) if math.isfinite(f) else None


# ------------------------------------------------------------------------------------------------ situation

@dataclasses.dataclass(frozen=True)
class Situation:
    """Identity-free description of a decision moment. Equal situations are interchangeable by construction."""
    blocks: tuple[Block, ...]
    pattern_ids: tuple[str, ...] = ()          # content-hash ids of patterns active now (patterns are knowledge, not identity)
    rank_mode: str = "cross_section"           # cross_section | absolute | mixed
    schema: str = SCHEMA_VERSION
    LAYER: ClassVar[Layer] = Layer.L3_SITUATION

    @functools.cached_property
    def _by_kind(self) -> dict:
        return {b.kind: b for b in self.blocks}

    def block(self, kind: str) -> Block:
        try:
            return self._by_kind[kind]
        except KeyError as e:
            raise KeyError(f"situation has no block {kind!r}") from e

    def get(self, path: str, default=None):
        block, _, field = path.partition(".")
        b = self._by_kind.get(block)
        return default if b is None else b.get(field, default)

    @functools.cached_property
    def _bins(self) -> dict:
        out: dict[str, str] = {}
        for b in self.blocks:
            out.update(b.bins())
        return out

    def bins(self) -> dict[str, str]:
        """Bucket label of every dimension. Cached and shared: callers must treat the mapping as read-only."""
        return self._bins

    @functools.cached_property
    def situation_id(self) -> str:
        """Coarse id: identical for situations that fall in the same bucket on every dimension."""
        return stable_hash({"b": self.bins(), "p": self.pattern_ids, "r": self.rank_mode, "s": self.schema})

    @functools.cached_property
    def exact_id(self) -> str:
        return stable_hash({"v": {b.kind: b.as_dict() for b in self.blocks}, "p": self.pattern_ids, "r": self.rank_mode})

    def coverage(self) -> float:
        n = sum(len(b.values) for b in self.blocks)
        return sum(b.observed() for b in self.blocks) / max(n, 1)

    def coverage_by_block(self) -> dict[str, float]:
        return {b.kind: round(b.coverage(), 4) for b in self.blocks}

    def missing_paths(self) -> tuple[str, ...]:
        return tuple(f"{b.kind}.{n}" for b in self.blocks for n, v in b.values if v is None)

    def usable(self, min_coverage: float = 0.5) -> bool:
        return self.coverage() >= min_coverage

    def unknown_state(self, min_coverage: float = 0.5) -> Unknown | None:
        return None if self.usable(min_coverage) else Unknown.INSUFFICIENT_DATA

    def validate(self) -> list[str]:
        errs: list[str] = []
        kinds = [b.kind for b in self.blocks]
        if tuple(kinds) != BLOCK_ORDER:
            errs.append(f"blocks {kinds} differ from the canonical order")
        for b in self.blocks:
            errs.extend(b.validate())
        if self.rank_mode not in ("cross_section", "absolute", "mixed"):
            errs.append(f"rank_mode {self.rank_mode!r}")
        if list(self.pattern_ids) != sorted(set(self.pattern_ids)):
            errs.append("pattern_ids must be sorted and unique")
        return errs

    def to_dict(self) -> dict:
        return {"schema": self.schema, "rank_mode": self.rank_mode, "patterns": list(self.pattern_ids),
                "blocks": {b.kind: b.as_dict() for b in self.blocks}}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Situation":
        blocks = tuple(Block.make(k, **(d["blocks"].get(k) or {})) for k in BLOCK_ORDER)
        return cls(blocks, tuple(sorted(set(d.get("patterns", ())))), d.get("rank_mode", "cross_section"),
                   d.get("schema", SCHEMA_VERSION))

    def describe(self) -> str:
        """One human line per block, listing only observed fields. This is the 'what is happening' text."""
        lines = []
        for b in self.blocks:
            parts = [f"{n}={v:.4g}" if isinstance(v, float) else f"{n}={v}" for n, v in b.values if v is not None]
            lines.append(f"{b.kind}: " + (", ".join(parts) if parts else "unknown"))
        if self.pattern_ids:
            lines.append("patterns: " + ",".join(self.pattern_ids))
        return "\n".join(lines)

    def with_block(self, block: Block) -> "Situation":
        return dataclasses.replace(self, blocks=tuple(block if b.kind == block.kind else b for b in self.blocks))


# ------------------------------------------------------------------------------------------------ pooling ladder

# Hierarchical pooling ladder used by the context model: each rung is a set of dimensions; the rung above it is its
# parent. Rung 0 has no dimension (pattern-wide), so a bucket that is too small always has somewhere to collapse to.
POOLING_LADDER: tuple[tuple[str, ...], ...] = (
    (),
    ("regime.label",),
    ("regime.label", "volatility.vol_rank"),
    ("regime.label", "volatility.vol_rank", "trend.state"),
    ("regime.label", "volatility.vol_rank", "trend.state", "sector.family"),
    ("regime.label", "volatility.vol_rank", "trend.state", "sector.family", "liquidity.dv_rank", "breadth.state"),
)


def ladder_key(sit: Situation, rung: int) -> tuple[str, ...]:
    """The bucket a situation falls into at pooling rung `rung`. Missing dimensions become 'na' (own bucket, pooled up)."""
    if not 0 <= rung < len(POOLING_LADDER):
        raise IndexError(f"rung {rung} outside 0..{len(POOLING_LADDER) - 1}")
    b = sit.bins()
    return tuple(f"{p}={b.get(p, 'na')}" for p in POOLING_LADDER[rung])


# ------------------------------------------------------------------------------------------------ config & inputs

@dataclasses.dataclass(frozen=True)
class SituationConfig:
    min_cs_n: int = 20                 # smallest cross-section that may produce a percentile rank
    include_seasonality: bool = False  # legitimate calendar structure is opt-in; identity tests run with it off
    include_macro_levels: bool = True
    min_coverage: float = 0.5
    max_macro_staleness_days: int = 400
    lottery_rank: float = 0.85
    quiet_vol_rank: float = 0.3
    version: str = "s1"

    def validate(self) -> list[str]:
        errs = []
        if self.min_cs_n < 5:
            errs.append("min_cs_n < 5 gives meaningless ranks")
        if not 0.0 < self.min_coverage <= 1.0:
            errs.append("min_coverage outside (0, 1]")
        if self.max_macro_staleness_days < 30:
            errs.append("max_macro_staleness_days < 30 would discard monthly series")
        return errs


@dataclasses.dataclass(frozen=True)
class MacroObs:
    """One macro reading. `period_end` is what it measures, `published` is when the world could first see it."""
    name: str
    value: float
    period_end: dt.date
    published: dt.date

    def check(self) -> list[str]:
        errs = []
        if not math.isfinite(float(self.value)):
            errs.append(f"{self.name}: non-finite value")
        if as_date(self.published) < as_date(self.period_end):
            errs.append(f"{self.name}: published before the period it measures ends")
        return errs


@dataclasses.dataclass(frozen=True)
class ActivePattern:
    pattern_id: str
    score: float
    direction: int = 1                 # +1 long expectation, -1 short expectation

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if not math.isfinite(float(self.score)):
            raise ValueError("pattern score must be finite")


@dataclasses.dataclass(frozen=True)
class PortfolioState:
    n_positions: int = 0
    gross: float = 0.0
    drawdown: float = 0.0              # <= 0
    port_vol: float | None = None
    position_weight: float = 0.0       # weight the candidate position would take
    risk_used: float = 0.0             # share of the risk budget already used

    def check(self) -> list[str]:
        errs = []
        if self.n_positions < 0 or self.gross < 0:
            errs.append("negative positions or exposure")
        if self.drawdown > 1e-12:
            errs.append("drawdown must be <= 0")
        if not 0.0 <= self.position_weight <= 1.0:
            errs.append("position_weight outside [0, 1]")
        return errs


# publication lags in SESSIONS (mirrors engine.analogs.LAG, audited 2026-09-28); calendar days ~ sessions * 7/5.
try:
    from engine.analogs import LAG as _ANALOG_LAG
    MACRO_LAG_SESSIONS = dict(_ANALOG_LAG)
except Exception:                                      # analogs pulls in the data layer; the lag table is all we need
    MACRO_LAG_SESSIONS = {"CPIAUCSL": 35, "UNRATE": 30, "INDPRO": 35, "UMCSENT": 25, "USREC": 460, "NFCI": 7, "STLFSI4": 7}
MACRO_ALIASES = {"rates": ("DGS10",), "curve": ("T10Y2Y",), "credit": ("BAMLH0A0HYM2",), "fin_stress": ("STLFSI4", "NFCI"),
                 "recession": ("USREC",)}


def macro_obs_from_frame(macro: pd.DataFrame, now, lags: Mapping[str, int] | None = None) -> list[MacroObs]:
    """Turn a date-indexed macro frame into MacroObs whose `published` date includes the audited publication lag.
    Rows whose publication date is after `now` are dropped here, so a caller cannot forget to."""
    lags = MACRO_LAG_SESSIONS if lags is None else lags
    now_d = as_date(now)
    out = []
    for col in macro.columns:
        s = macro[col].dropna()
        lag_days = int(round(lags.get(col, 1) * 7 / 5))
        for ts, v in s.items():
            period = as_date(ts)
            pub = period + dt.timedelta(days=lag_days)
            if pub <= now_d:
                out.append(MacroObs(col, float(v), period, pub))
    return out


class CrossSection:
    """Percentile ranks of one decision day's universe. Ties get the mid-rank so tied inputs give equal ranks."""

    def __init__(self, frame: pd.DataFrame | None, min_n: int = 20):
        self.min_n = min_n
        self._sorted: dict[str, np.ndarray] = {}
        if frame is not None:
            for c in frame.columns:
                a = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
                a = np.sort(a[np.isfinite(a)])
                if len(a) >= min_n:
                    self._sorted[c] = a

    def has(self, col: str) -> bool:
        return col in self._sorted

    def pct(self, col: str, v) -> float | None:
        a = self._sorted.get(col)
        v = clean_number(v)
        if a is None or v is None:
            return None
        lo, hi = np.searchsorted(a, v, side="left"), np.searchsorted(a, v, side="right")
        return round(float((lo + hi) / 2.0 / len(a)), 6)


# centre / scale for the absolute fallback when no cross-section exists (logistic squashing to [0, 1])
ABSOLUTE_RANK = {"vol20": (0.02, 0.01), "log_dv": (16.5, 1.5), "max20": (0.04, 0.025), "ind_mom20": (0.0, 0.05),
                 "ind_mom60": (0.0, 0.10), "atr_pct": (0.025, 0.015)}


def sic_family(code) -> str:
    """Coarse sector family from a SIC code (first two digits). Families, not codes, so a scrambled code within a family
    changes nothing and no single industry is an identity."""
    try:
        d = int(str(code).strip()[:2])
    except (TypeError, ValueError):
        return "unknown"
    if d < 1 or d > 99:
        return "unknown"
    if d <= 14:
        return "resources"
    if d <= 39:
        return "technology" if d in (35, 36, 38) else "manufacturing"
    if d <= 49:
        return "utilities_transport"
    if d <= 59:
        return "trade_consumer"
    if d <= 67:
        return "finance_realestate"
    if d == 73:
        return "technology"
    if d in (80, 83):
        return "healthcare"
    if d <= 89:
        return "services_other"
    return "unknown"


def classify_regime(spy_ma200, vix, vix_term=None) -> tuple[str, str, str]:
    """(label, vol_regime, trend_regime) from the market state, or ('unknown', ...) when either driver is missing."""
    if spy_ma200 is None or vix is None:
        return "unknown", "unknown", "unknown"
    trend = "up" if spy_ma200 > 0.02 else "down" if spy_ma200 < -0.02 else "flat"
    backwardated = vix_term is not None and vix_term > 1.05
    if vix >= 32 or (backwardated and vix >= 26):
        return "stress", "crisis", trend
    vol = "low" if vix < 16 else "mid" if vix < 22 else "high"
    if spy_ma200 > 0:
        return ("bull_calm" if vix < 20 else "bull_volatile"), vol, trend
    return ("correction_calm" if vix < 24 else "bear_volatile"), vol, trend


def correlation_structure(returns: pd.DataFrame, column, now, market_col=None, window: int = 60,
                          min_obs: int = 30) -> dict[str, float | None]:
    """Correlation structure of the universe as of `now`: average pair correlation, share of variance on the first
    principal component, and the named column's correlation/beta to the market column. Rows after `now` raise.
    Uses at most the last `window` rows, needs `min_obs` complete rows; otherwise every value is None (UNKNOWN)."""
    empty = {"avg_pair_corr": None, "top_eig_share": None, "corr_to_mkt": None, "beta": None}
    if returns is None or returns.empty:
        return empty
    now_d = as_date(now)
    idx = pd.DatetimeIndex(returns.index)
    if len(idx) and as_date(idx.max()) > now_d:
        raise FirewallBreach(f"correlation_structure: returns dated {idx.max()} after now={now_d}")
    R = returns.iloc[-window:]
    R = R.loc[:, R.notna().mean() > 0.8].dropna()
    if len(R) < min_obs or R.shape[1] < 3:
        return empty
    sd = R.std()
    R = R.loc[:, sd > 1e-12]
    if R.shape[1] < 3:
        return empty
    corr = np.corrcoef(R.to_numpy(dtype=float), rowvar=False)
    n = corr.shape[0]
    avg = float((corr.sum() - n) / (n * (n - 1)))
    eig = np.linalg.eigvalsh(corr)
    out = dict(empty, avg_pair_corr=round(avg, 6), top_eig_share=round(float(eig[-1] / eig.sum()), 6))
    if column in R.columns and market_col is not None and market_col in returns.columns:
        m = returns[market_col].iloc[-window:].reindex(R.index)
        x = R[column]
        if m.std() > 1e-12 and x.std() > 1e-12 and m.notna().all():
            c = float(np.corrcoef(x, m)[0, 1])
            out["corr_to_mkt"] = round(c, 6)
            out["beta"] = round(c * float(x.std() / m.std()), 6)
    return out


def horizon_state(days) -> str | None:
    if days is None:
        return None
    return "short" if days <= 2 else "week" if days <= 6 else "multi_week"


# ------------------------------------------------------------------------------------------------ builder

class SituationBuilder:
    """Builds Situations from one day's inputs. Stateless apart from the config, so it is deterministic and thread-safe."""

    def __init__(self, config: SituationConfig | None = None):
        self.cfg = config or SituationConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("SituationConfig invalid: " + "; ".join(errs))
        self.dropped: dict[str, int] = {}      # dimension -> count of values outside their sanity range, made UNKNOWN

    def _sanitize(self, block: Block) -> Block:
        """A value outside its sanity range is a data error: it becomes UNKNOWN (None) and is counted, never clipped or kept."""
        specs = BLOCK_SPECS[block.kind]
        vals, changed = {}, False
        for spec, (name, v) in zip(specs, block.values):
            if v is not None and spec.check(v) is not None:
                self.dropped[f"{block.kind}.{name}"] = self.dropped.get(f"{block.kind}.{name}", 0) + 1
                v, changed = None, True
            vals[name] = v
        return Block.make(block.kind, **vals) if changed else block

    # ---- guards
    def _guard(self, now, asof, what: str) -> None:
        if now is None:
            raise FirewallBreach(f"{what}: no decision timestamp `now`")
        if asof is not None and as_date(asof) > as_date(now):
            raise FirewallBreach(f"{what}: data dated {as_date(asof)} is after now={as_date(now)}")

    # ---- ranks
    def _rank(self, cs: CrossSection | None, row: Mapping, col: str, modes: set) -> float | None:
        v = clean_number(row.get(col))
        if v is None:
            return None
        if cs is not None and cs.has(col):
            modes.add("cross_section")
            return cs.pct(col, v)
        if col in ABSOLUTE_RANK:
            c, s = ABSOLUTE_RANK[col]
            modes.add("absolute")
            return round(1.0 / (1.0 + math.exp(-(v - c) / s)), 6)
        return None

    # ---- blocks
    def _market(self, m: Mapping) -> Block:
        return Block.make("market", spy_ma200=m.get("m_spy_ma200"), spy_ma50=m.get("m_spy_ma50"), spy_r5=m.get("m_spy_r5"),
                          vix=m.get("m_vix"), vix_term=m.get("m_vix_term"), vix_chg5=m.get("m_vix_chg5"))

    def _sector(self, row, cs, sic, modes) -> Block:
        return Block.make("sector", family=sic_family(sic) if sic is not None else None,
                          strength20=self._rank(cs, row, "ind_mom20", modes), strength60=self._rank(cs, row, "ind_mom60", modes),
                          rel20=row.get("rel_ind20"), rel60=row.get("rel_ind60"))

    def _volatility(self, row, cs, modes) -> Block:
        vr = clean_number(row.get("vol_ratio"))
        state = None if vr is None else "expanding" if vr > 1.15 else "contracting" if vr < 0.85 else "stable"
        return Block.make("volatility", vol_rank=self._rank(cs, row, "vol20", modes), atr_pct=row.get("atr_pct"),
                          vol_ratio=vr, range_compress=row.get("range_compress"), state=state)

    def _liquidity(self, row, cs, modes) -> Block:
        dv = self._rank(cs, row, "log_dv", modes)
        s1, s5 = clean_number(row.get("vol_surge1")), clean_number(row.get("vol_surge5"))
        if dv is None:
            state = None
        elif s5 is not None and s5 > 1.7:
            state = "surging"
        else:
            state = "thin" if dv < 0.25 else "deep" if dv > 0.75 else "normal"
        return Block.make("liquidity", dv_rank=dv, vol_surge1=s1, vol_surge5=s5, state=state)

    def _trend(self, row) -> Block:
        r20, r60, d200 = (clean_number(row.get(k)) for k in ("r20", "r60", "dist_ma200"))
        state = None
        if r20 is not None and r60 is not None:
            if r60 > 0.03 and r20 > 0.0:
                state = "up"
            elif r60 < -0.03 and r20 < 0.0:
                state = "down"
            elif r60 > 0.03 and r20 < -0.03:
                state = "reversal_down"
            elif r60 < -0.03 and r20 > 0.03:
                state = "reversal_up"
            else:
                state = "flat"
        return Block.make("trend", r5=row.get("r5"), r20=r20, r60=r60, mom_12_1=row.get("mom_12_1"),
                          dist_ma50=row.get("dist_ma50"), dist_ma200=d200, dist_52wh=row.get("dist_52wh"), state=state)

    def _breadth(self, m: Mapping) -> Block:
        b, d = clean_number(m.get("m_breadth")), clean_number(m.get("m_dispersion"))
        state = None if b is None else "washed_out" if b < 0.25 else "narrow" if b < 0.45 else "euphoric" if b > 0.85 else "healthy"
        return Block.make("breadth", breadth=b, dispersion=d, state=state)

    def _macro(self, obs: Sequence[MacroObs], now) -> Block:
        now_d = as_date(now)
        latest: dict[str, MacroObs] = {}
        for o in obs:
            bad = o.check()
            if bad:
                raise ValueError("bad macro observation: " + "; ".join(bad))
            if as_date(o.published) > now_d:
                continue                                   # not yet public: not usable, and not an error
            cur = latest.get(o.name)
            if cur is None or as_date(o.period_end) > as_date(cur.period_end):
                latest[o.name] = o

        def pick(key):
            for nm in MACRO_ALIASES[key]:
                o = latest.get(nm)
                if o is not None and (now_d - as_date(o.published)).days <= self.cfg.max_macro_staleness_days + 400 * (key == "recession"):
                    return o
            return None
        picked = {k: pick(k) for k in MACRO_ALIASES}
        used = [o for o in picked.values() if o is not None]
        stale = max(((now_d - as_date(o.published)).days for o in used), default=None)
        rates = picked["rates"].value if picked["rates"] and self.cfg.include_macro_levels else None
        credit = picked["credit"].value if picked["credit"] else None
        stress = picked["fin_stress"].value if picked["fin_stress"] else None
        rec = None if picked["recession"] is None else ("yes" if picked["recession"].value >= 0.5 else "no")
        if not used:
            state = "unknown"
        elif rec == "yes" or (stress is not None and stress > 1.5) or (credit is not None and credit > 7.0):
            state = "stress"
        elif (stress is not None and stress > 0.3) or (credit is not None and credit > 5.0):
            state = "tight"
        elif (stress is not None and stress < -0.3) or (credit is not None and credit < 3.5):
            state = "easy"
        else:
            state = "neutral"
        return Block.make("macro", rates=rates, curve=picked["curve"].value if picked["curve"] else None, credit=credit,
                          fin_stress=stress, recession=rec, staleness_days=stale, state=state)

    def _regime(self, m: Mapping) -> Block:
        label, vol, trend = classify_regime(clean_number(m.get("m_spy_ma200")), clean_number(m.get("m_vix")),
                                            clean_number(m.get("m_vix_term")))
        return Block.make("regime", label=label, vol_regime=None if vol == "unknown" else vol,
                          trend_regime=None if trend == "unknown" else trend)

    def _shock(self, row, m) -> Block:
        gap, dse = clean_number(row.get("gap_today")), clean_number(row.get("days_since_earn"))
        red = clean_number(row.get("ev_red_flag"))
        news = clean_number(row.get("news5"))
        mshock = None
        sp, vc = clean_number(m.get("m_spy_r5")), clean_number(m.get("m_vix_chg5"))
        if sp is not None or vc is not None:
            mshock = max(abs(sp or 0.0) / 0.02, max(vc or 0.0, 0.0) / 0.15)
        if red is not None and red > 0:
            kind = "red_flag"
        elif mshock is not None and mshock >= 2.0:
            kind = "market"
        elif gap is not None and abs(gap) > 0.04:
            kind = "stock_gap"
        elif dse is not None and dse <= 2:
            kind = "earnings"
        elif news is not None and news > 0:
            kind = "event"
        elif gap is None and dse is None and red is None:
            kind = None
        else:
            kind = "none"
        return Block.make("shock", kind=kind, gap_today=gap, max20=row.get("max20"), min20=row.get("min20"),
                          days_since_earn=dse, red_flag=None if red is None else ("yes" if red > 0 else "no"),
                          market_shock=mshock)

    def _stock_type(self, row, cs, modes, vol_rank, dv_rank) -> Block:
        lot = self._rank(cs, row, "max20", modes)
        persist = clean_number(row.get("frog"))
        news, earn = clean_number(row.get("news5")), clean_number(row.get("earn_in_week"))
        load = None
        if news is not None or earn is not None:
            load = (news or 0.0) + (earn or 0.0) + sum(clean_number(row.get(k)) or 0.0 for k in ("ev_offering", "ev_shelf", "ev_activist", "ev_agreement"))
        kind = None
        if vol_rank is not None:
            r60 = clean_number(row.get("r60"))
            if lot is not None and lot > self.cfg.lottery_rank and vol_rank > 0.8:
                kind = "lottery"
            elif vol_rank < self.cfg.quiet_vol_rank and dv_rank is not None and dv_rank > 0.6:
                kind = "quiet_large"
            elif load is not None and load >= 1.0:
                kind = "event_driven"
            elif persist is not None and persist > 0.05 and r60 is not None and abs(r60) > 0.10:
                kind = "momentum"
            elif persist is not None and persist < -0.05:
                kind = "reversal_prone"
            else:
                kind = "ordinary"
        return Block.make("stock_type", kind=kind, lottery_rank=lot, skew60=row.get("skew60"), persistence=persist, event_load=load)

    def _correlation(self, corr: Mapping | None) -> Block:
        corr = corr or {}
        a = clean_number(corr.get("avg_pair_corr"))
        state = None if a is None else "herded" if a > 0.45 else "dispersed" if a < 0.15 else "normal"
        return Block.make("correlation", avg_pair_corr=a, top_eig_share=corr.get("top_eig_share"),
                          corr_to_mkt=corr.get("corr_to_mkt"), beta=corr.get("beta"), state=state)

    def _seasonality(self, now) -> Block:
        if not self.cfg.include_seasonality:
            return Block.make("seasonality")
        d = as_date(now)
        return Block.make("seasonality", month_of_quarter=str((d.month - 1) % 3 + 1),
                          turn_of_month="yes" if d.day >= 27 or d.day <= 3 else "no",
                          quarter_end="yes" if d.month in (3, 6, 9, 12) and d.day >= 20 else "no",
                          january="yes" if d.month == 1 else "no")

    def _patterns(self, active: Sequence[ActivePattern]) -> tuple[Block, tuple[str, ...]]:
        if not active:
            return Block.make("pattern_interaction", n_active=0, agreement=None, top_score=None, score_spread=None, mix="none"), ()
        ids = sorted({a.pattern_id for a in active})
        signed = np.array([a.direction * a.score for a in active], dtype=float)
        dirs = np.array([a.direction for a in active], dtype=float)
        agree = None if len(active) < 2 else float(abs(dirs.sum()) / len(dirs) * 2 - 1)
        if len(active) == 1:
            mix = "single"
        else:
            mix = "agree" if abs(dirs.sum()) == len(dirs) else "conflict"
        return Block.make("pattern_interaction", n_active=len(active), agreement=agree,
                          top_score=float(np.max(np.abs(signed))), score_spread=float(signed.std()), mix=mix), tuple(ids)

    def _position(self, ps: PortfolioState | None) -> Block:
        if ps is None:
            return Block.make("position_risk")
        errs = ps.check()
        if errs:
            raise ValueError("PortfolioState invalid: " + "; ".join(errs))
        state = ("flat" if ps.n_positions == 0 else "drawdown" if ps.drawdown < -0.05 else
                 "stretched" if ps.risk_used > 0.8 or ps.gross > 0.95 else "normal")
        return Block.make("position_risk", n_positions=ps.n_positions, gross=ps.gross, drawdown=ps.drawdown, port_vol=ps.port_vol,
                          position_weight=ps.position_weight, risk_used=ps.risk_used, state=state)

    # ---- public
    def build(self, stock_row: Mapping[str, Any], market_row: Mapping[str, Any] | None, now, *,
              cross_section: CrossSection | None = None, sic=None, macro: Sequence[MacroObs] = (),
              corr: Mapping[str, Any] | None = None, horizon_days: int | None = 5,
              patterns: Sequence[ActivePattern] = (), portfolio: PortfolioState | None = None,
              data_asof=None) -> Situation:
        """One Situation. `data_asof` is the date the feature row is for; it must not be after `now`. `market_row`
        defaults to the m_* columns of `stock_row` (the panel convention)."""
        self._guard(now, data_asof, "SituationBuilder.build")
        m = market_row if market_row is not None else {k: v for k, v in stock_row.items() if str(k).startswith("m_")}
        modes: set[str] = set()
        vol = self._volatility(stock_row, cross_section, modes)
        liq = self._liquidity(stock_row, cross_section, modes)
        pat_block, pat_ids = self._patterns(patterns)
        blocks = {
            "market": self._market(m), "sector": self._sector(stock_row, cross_section, sic, modes),
            "stock_type": self._stock_type(stock_row, cross_section, modes, vol.get("vol_rank"), liq.get("dv_rank")),
            "volatility": vol, "liquidity": liq, "trend": self._trend(stock_row), "breadth": self._breadth(m),
            "macro": self._macro(macro, now), "regime": self._regime(m), "shock": self._shock(stock_row, m),
            "correlation": self._correlation(corr),
            "horizon": Block.make("horizon", days=horizon_days, state=horizon_state(horizon_days)),
            "seasonality": self._seasonality(now), "pattern_interaction": pat_block, "position_risk": self._position(portfolio),
        }
        mode = "cross_section" if modes <= {"cross_section"} else "absolute" if modes == {"absolute"} else "mixed"
        sit = Situation(tuple(self._sanitize(blocks[k]) for k in BLOCK_ORDER), pat_ids, mode)
        errs = sit.validate()
        if errs:
            raise ValueError("built an invalid Situation: " + "; ".join(errs[:5]))
        return sit

    def build_panel(self, X: pd.DataFrame, now, *, sic: Mapping[str, Any] | pd.Series | None = None,
                    macro: Sequence[MacroObs] = (), corr_by_date: Mapping[Any, Mapping] | None = None,
                    horizon_days: int | None = 5, patterns_by_row: Mapping[tuple, Sequence[ActivePattern]] | None = None,
                    portfolio: PortfolioState | None = None) -> pd.Series:
        """Situations for every (date, ticker) row of a feature panel, each date ranked against its own cross-section.
        Any row dated after `now` raises before anything is built (no partial results from a leaky call)."""
        if X is None or len(X) == 0:
            return pd.Series([], dtype=object, index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
        dates = X.index.get_level_values(0)
        if as_date(dates.max()) > as_date(now):
            raise FirewallBreach(f"build_panel: rows dated {as_date(dates.max())} are after now={as_date(now)}")
        out = {}
        for d, day in X.groupby(level=0, sort=True):
            day = day.droplevel(0)
            cs = CrossSection(day, self.cfg.min_cs_n)
            corr = (corr_by_date or {}).get(d)
            for tk, row in day.iterrows():
                key = (d, tk)
                code = None if sic is None else (sic.get(tk) if hasattr(sic, "get") else None)
                out[key] = self.build(row.to_dict(), None, as_date(d), cross_section=cs, sic=code, macro=macro, corr=corr,
                                      horizon_days=horizon_days, patterns=(patterns_by_row or {}).get(key, ()),
                                      portfolio=portfolio, data_asof=d)
        return pd.Series(list(out.values()),index=pd.MultiIndex.from_tuples(list(out), names=["date", "ticker"]), dtype=object)


# ------------------------------------------------------------------------------------------------ identity audit & scrambles

_DATE_RE = re.compile(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])([-/](0[1-9]|[12]\d|3[01]))?\b")


def audit_identity_free(sit: Situation, tickers: Iterable[str] = (), years: Iterable[int] = (),
                        dates: Iterable[Any] = ()) -> list[str]:
    """Findings (empty = clean) if a Situation carries a known ticker, a calendar date, or a known year as text."""
    text = stable_hash_text(sit)
    found = []
    for t in tickers:
        if t and re.search(rf"(?<![A-Za-z0-9]){re.escape(str(t))}(?![A-Za-z0-9])", text):
            found.append(f"contains ticker {t}")
    for d in dates:
        if str(as_date(d)) in text:
            found.append(f"contains date {as_date(d)}")
    for y in years:
        if re.search(rf"(?<!\d){int(y)}(?!\d)", text):
            found.append(f"contains year {int(y)}")
    if _DATE_RE.search(text):
        found.append("contains a date-like string")
    return found


def stable_hash_text(sit: Situation) -> str:
    from .core import canonical_json
    return canonical_json(sit.to_dict())


def scramble_tickers(X: pd.DataFrame, seed: int, prefix: str = "Z") -> tuple[pd.DataFrame, dict]:
    """Relabel tickers with opaque names AND reorder rows within each date, so neither the name nor the row order (a
    name-order tie-break leaked twice before) can carry identity. Returns (frame, old->new mapping)."""
    rng = np.random.default_rng(seed)
    olds = sorted(set(X.index.get_level_values(1)))
    news = [f"{prefix}{i:05d}" for i in rng.permutation(len(olds))]
    mapping = dict(zip(olds, news))
    Y = X.copy()
    Y.index = pd.MultiIndex.from_arrays([X.index.get_level_values(0), X.index.get_level_values(1).map(mapping)],
                                        names=X.index.names)
    parts = []
    for _, day in Y.groupby(level=0, sort=True):
        parts.append(day.iloc[rng.permutation(len(day))])
    return pd.concat(parts), mapping


def shift_dates(X: pd.DataFrame, days: int) -> pd.DataFrame:
    Y = X.copy()
    Y.index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(X.index.get_level_values(0)) + pd.Timedelta(days=days),
                                         X.index.get_level_values(1)], names=X.index.names)
    return Y


def shift_years(X: pd.DataFrame, years: int) -> pd.DataFrame:
    """Move every date by whole calendar years (month/day preserved, so legitimate seasonality is unchanged)."""
    Y = X.copy()
    Y.index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(X.index.get_level_values(0)) + pd.DateOffset(years=years),
                                         X.index.get_level_values(1)], names=X.index.names)
    return Y


def scramble_sectors(sic: Mapping[str, Any], seed: int) -> dict:
    """Permute sector codes among tickers (keeps the multiset). Used for the 'where appropriate' sector test."""
    rng = np.random.default_rng(seed)
    keys = sorted(sic)
    vals = [sic[k] for k in keys]
    perm = rng.permutation(len(vals))
    return {k: vals[i] for k, i in zip(keys, perm)}


# ------------------------------------------------------------------------------------------------ library

class SituationLibrary:
    """Append-only store of situations seen, grouped by coarse id and pooling rung. Holds no ticker, no date: `ref` is
    an opaque caller key (e.g. a case number). Feeds equivalent-situation transfer checks and pooling statistics."""

    def __init__(self):
        self._items: dict[str, tuple[Situation, list[str]]] = {}
        self._n = 0

    def add(self, sit: Situation, ref: str) -> str:
        errs = sit.validate()
        if errs:
            raise ValueError("cannot store an invalid situation: " + "; ".join(errs[:3]))
        sid = sit.situation_id
        self._items.setdefault(sid, (sit, []))[1].append(str(ref))
        self._n += 1
        return sid

    def __len__(self) -> int:
        return self._n

    def distinct(self) -> int:
        return len(self._items)

    def refs(self, sid: str) -> tuple[str, ...]:
        return tuple(self._items[sid][1]) if sid in self._items else ()

    def get(self, sid: str) -> Situation | None:
        it = self._items.get(sid)
        return None if it is None else it[0]

    def equivalents(self, sit: Situation) -> tuple[str, ...]:
        """Refs of every stored situation in the same coarse bucket: the 'equivalent situation' set for transfer."""
        return self.refs(sit.situation_id)

    def rung_counts(self, rung: int) -> dict[tuple, int]:
        counts: dict[tuple, int] = {}
        for sit, refs in self._items.values():
            k = ladder_key(sit, rung)
            counts[k] = counts.get(k, 0) + len(refs)
        return counts

    def tiny_bucket_report(self, min_n: int = 30) -> dict[int, dict]:
        """Per pooling rung: how many buckets exist and how many hold fewer than `min_n` cases (the context-explosion check)."""
        rep = {}
        for r in range(len(POOLING_LADDER)):
            c = self.rung_counts(r)
            tiny = sum(1 for v in c.values() if v < min_n)
            rep[r] = {"buckets": len(c), "tiny": tiny, "cases_in_tiny": sum(v for v in c.values() if v < min_n),
                      "largest": max(c.values(), default=0)}
        return rep

    def deepest_supported_rung(self, sit: Situation, min_n: int = 30) -> int:
        """The most specific pooling rung whose bucket for `sit` already holds at least `min_n` cases."""
        best = 0
        for r in range(len(POOLING_LADDER)):
            if self.rung_counts(r).get(ladder_key(sit, r), 0) >= min_n:
                best = r
        return best


# ------------------------------------------------------------------------------------------------ input audit (reuse)

# every stock-level feature column of engine.features that the builder reads
INPUT_COLUMNS = ("r5", "r20", "r60", "mom_12_1", "dist_ma50", "dist_ma200", "dist_52wh", "vol20", "vol_ratio", "atr_pct",
                 "range_compress", "log_dv", "vol_surge1", "vol_surge5", "max20", "min20", "skew60", "frog", "gap_today",
                 "ind_mom20", "ind_mom60", "rel_ind20", "rel_ind60", "days_since_earn", "earn_in_week", "news5",
                 "ev_red_flag", "ev_offering", "ev_shelf", "ev_activist", "ev_agreement")


def market_columns() -> tuple[str, ...]:
    """The market-context columns memory.context_of reads (single source of truth: engine.memory.CTX)."""
    from engine.memory import CTX
    return tuple(CTX)


def market_columns_missing(row: Mapping[str, Any]) -> list[str]:
    return [c for c in market_columns() if clean_number(row.get(c)) is None]


def identity_proxy_audit(X: pd.DataFrame) -> dict[str, str]:
    """Columns of a feature panel that could stand in for a ticker or a date (engine.lessons.identity_proxy_columns:
    almost constant within a ticker, or a monotone function of the calendar). Only columns the builder reads are reported."""
    from engine.lessons import identity_proxy_columns
    used = [c for c in X.columns if c in INPUT_COLUMNS or str(c).startswith("m_")]
    if not used:
        return {}
    return identity_proxy_columns(X[used])


def panel_input_report(X: pd.DataFrame) -> dict[str, Any]:
    """What an input panel offers the builder: missing columns, the share of NaN per used column, identity proxies."""
    have = [c for c in INPUT_COLUMNS if c in X.columns]
    miss = [c for c in INPUT_COLUMNS if c not in X.columns]
    nan_share = {c: round(float(X[c].isna().mean()), 4) for c in have}
    mk = [c for c in market_columns() if c not in X.columns]
    return {"rows": int(len(X)), "columns_used": len(have), "columns_missing": miss, "market_columns_missing": mk,
            "nan_share": nan_share, "identity_proxies": identity_proxy_audit(X) if len(X) else {},
            "usable": len(have) >= 0.5 * len(INPUT_COLUMNS) and not mk}


# ------------------------------------------------------------------------------------------------ comparison utilities

@dataclasses.dataclass(frozen=True)
class SituationDelta:
    changed: tuple[tuple[str, str, str], ...]        # (path, bucket in a, bucket in b) where the buckets differ
    unknown: tuple[str, ...]                         # paths unobserved on at least one side
    pattern_added: tuple[str, ...]
    pattern_removed: tuple[str, ...]

    @property
    def n_changed(self) -> int:
        return len(self.changed)

    def describe(self) -> str:
        lines = [f"{p}: {x} -> {y}" for p, x, y in self.changed]
        lines += [f"+pattern {p}" for p in self.pattern_added] + [f"-pattern {p}" for p in self.pattern_removed]
        return "\n".join(lines) or "no bucket differs"


def diff(a: Situation, b: Situation) -> SituationDelta:
    ba, bb = a.bins(), b.bins()
    changed = tuple((p, ba[p], bb[p]) for p in all_paths() if ba[p] != bb[p] and ba[p] != "na" and bb[p] != "na")
    unknown = tuple(p for p in all_paths() if ba[p] == "na" or bb[p] == "na")
    return SituationDelta(changed, unknown, tuple(sorted(set(b.pattern_ids) - set(a.pattern_ids))),
                          tuple(sorted(set(a.pattern_ids) - set(b.pattern_ids))))


def coarsen(sit: Situation, keep: Iterable[str]) -> Situation:
    """A lower-resolution copy: every block not in `keep` becomes all-unknown. Used to ask whether knowledge transfers on
    fewer dimensions (equivalent-situation transfer) without fabricating values for the dropped ones."""
    keep = set(keep)
    bad = keep - set(BLOCK_ORDER)
    if bad:
        raise ValueError(f"unknown blocks {sorted(bad)}")
    blocks = tuple(b if b.kind in keep else Block.make(b.kind) for b in sit.blocks)
    return dataclasses.replace(sit, blocks=blocks, pattern_ids=sit.pattern_ids if "pattern_interaction" in keep else ())


def population_shift(a: Sequence[Situation], b: Sequence[Situation], min_share: float = 0.005) -> dict[str, float]:
    """Population-stability index per dimension between two sets of situations (e.g. two years). PSI > 0.25 is a large shift:
    knowledge learned on `a` is being applied to a different population on that dimension. Missing buckets are ignored."""
    out = {}
    for p in all_paths():
        ca, cb = {}, {}
        for s, c in ((a, ca), (b, cb)):
            for sit in s:
                lab = sit.bins()[p]
                if lab != "na":
                    c[lab] = c.get(lab, 0) + 1
        na, nb = sum(ca.values()), sum(cb.values())
        if na < 20 or nb < 20:
            continue
        psi = 0.0
        for lab in set(ca) | set(cb):
            pa, pb = max(ca.get(lab, 0) / na, min_share), max(cb.get(lab, 0) / nb, min_share)
            psi += (pa - pb) * math.log(pa / pb)
        out[p] = round(psi, 6)
    return out


def coverage_report(sits: Sequence[Situation]) -> dict[str, Any]:
    """Population diagnostics: coverage per block, missing share per dimension, and dimensions that carry no information
    (a single occupied bucket) - those cannot separate situations and should not be trusted to define context."""
    if not len(sits):
        return {"n": 0, "blocks": {}, "missing_share": {}, "degenerate": [], "entropy": {}}
    paths = all_paths()
    miss = {p: 0 for p in paths}
    occ: dict[str, dict[str, int]] = {p: {} for p in paths}
    for s in sits:
        for p, lab in s.bins().items():
            if lab == "na":
                miss[p] += 1
            else:
                occ[p][lab] = occ[p].get(lab, 0) + 1
    ent = {}
    for p, c in occ.items():
        n = sum(c.values())
        ent[p] = 0.0 if n == 0 else round(-sum(v / n * math.log(v / n) for v in c.values()), 4)
    return {"n": len(sits), "blocks": {k: round(float(np.mean([s.coverage_by_block()[k] for s in sits])), 4) for k in BLOCK_ORDER},
            "missing_share": {p: round(m / len(sits), 4) for p, m in miss.items()},
            "degenerate": [p for p in paths if len(occ[p]) == 1 and miss[p] < len(sits)], "entropy": ent}


def hash_label(label: str) -> int:
    """Stable integer for a bucket label (deterministic across processes, unlike the salted built-in hash)."""
    h = 1469598103934665603
    for ch in label.encode("utf-8"):
        h = ((h ^ ch) * 1099511628211) & 0x7FFFFFFFFFFFFFFF
    return h


def identity_recoverability(sits: Sequence[Situation], labels: Sequence[Any], k: int = 5, seed: int = 0,
                            n_perm: int = 200) -> dict[str, float]:
    """Identity memorization control (contract section 15): how well can a nearest-neighbour vote recover a label (ticker,
    year, date) from the situation alone? Leave-one-out accuracy is compared with chance (the majority-class share) and with
    a label-permutation null. Clean identity-free situations sit at chance (p ~ uniform); a leaked identity is far above."""
    n = len(sits)
    if n < 3 * k or len(labels) != n:
        return {"n": n, "accuracy": float("nan"), "chance": float("nan"), "lift": float("nan"), "p": float("nan")}
    paths = all_paths()
    codes = np.array([[hash_label(s.bins()[p]) for p in paths] for s in sits], dtype=np.int64)
    na = hash_label("na")
    y = np.asarray(pd.factorize(pd.Series(list(labels)))[0])
    both = (codes[:, None, :] != na) & (codes[None, :, :] != na)         # dimensions observed on both sides
    differ = (codes[:, None, :] != codes[None, :, :]) & both
    d = differ.sum(axis=2).astype(float) / np.maximum(both.sum(axis=2), 1)
    np.fill_diagonal(d, np.inf)
    rng = np.random.default_rng(seed)
    order = np.argsort(d + rng.random(d.shape) * 1e-9, axis=1)[:, :k]      # seeded tie-break: never row order

    def acc(lab):
        votes = lab[order]
        pred = np.array([np.bincount(v).argmax() for v in votes])
        return float(np.mean(pred == lab))
    a = acc(y)
    chance = float(np.bincount(y).max() / n)
    null = np.array([acc(rng.permutation(y)) for _ in range(n_perm)])
    return {"n": n, "accuracy": a, "chance": chance, "lift": a - chance, "p": float((1 + (null >= a).sum()) / (n_perm + 1))}


# ------------------------------------------------------------------------------------------------ history & serialisation

class SituationStream:
    """Rolling history of one decision stream's situations, for recent-history similarity. Time must strictly increase and
    can never pass `now`; the stream keeps timestamps only to enforce order and never exposes them."""

    def __init__(self, maxlen: int = 10):
        if maxlen < 1:
            raise ValueError("maxlen < 1")
        self.maxlen = maxlen
        self._buf: list[tuple[dt.date, Situation]] = []

    def push(self, when, sit: Situation, now) -> None:
        if as_date(when) > as_date(now):
            raise FirewallBreach(f"SituationStream: situation dated {as_date(when)} after now={as_date(now)}")
        if self._buf and as_date(when) <= self._buf[-1][0]:
            raise ValueError("SituationStream: timestamps must strictly increase")
        self._buf.append((as_date(when), sit))
        del self._buf[:-self.maxlen]

    def history(self, n: int | None = None) -> tuple[Situation, ...]:
        """Most recent first (index 0 = latest)."""
        items = [s for _, s in reversed(self._buf)]
        return tuple(items if n is None else items[:n])

    def __len__(self) -> int:
        return len(self._buf)


def to_json(sit: Situation) -> str:
    from .core import canonical_json
    return canonical_json(sit.to_dict())


def from_json(text: str) -> Situation:
    import json
    return Situation.from_dict(json.loads(text))
