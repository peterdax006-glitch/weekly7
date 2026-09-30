"""Market expectation versus market reality (PREDICTION_ERROR_ADDITION checklists F and G; canon C68, C66 sections 13/14/25, C63).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; no real-data run was made).

Before each period the engine records an IMMUTABLE, hash-chained expectation for the eleven checklist-F quantities (opportunity,
volatility, breadth, dispersion, correlations, sector leadership, momentum persistence, reversal probability, gap behaviour, number
of movers, number of qualifying 5-10% opportunities). When the period ends it resolves the expectation against reality with the
same standardised error the rest of the system uses (engine.learning.surprise.continuous_z / surprise_bits, so the records can be fed
straight into the shared SurpriseTracker). A large opportunity error (10% expected, 15% arrived; or 10% expected, 6% arrived) opens an
investigation that TESTS every checklist-F/G explanation instead of assuming one:
  * probe tests: has the explaining quantity shifted (recent vs reference window, autocorrelation-adjusted, BH-corrected across probes)
    AND is it linked to opportunity in the reference window AND does shift x link point the same way as the surprise?
  * combined small changes: probes that are individually insignificant are pooled (Stouffer, signed by their link) - "many small
    changes add up" is a hypothesis with its own test, not a leftover.
  * patterns, sectors and stocks: which strengthened, weakened, reversed, stopped responding, or woke up (BH across units).
  * 'did candidate selection fail?' is an explicit alternative: universe opportunity vs the pick hit rate (two-proportion test).
  * precursors: does a lagged probe predict later opportunity errors, replicated in both halves and beating a circular-shift null?
    That is the checklist-F objective - identifying the precursor BEFORE the next occurrence - and only replicated ones count.
Nothing is concluded from a missing quantity (UNTESTABLE, never 'not supported'), and 'the strategy stopped working' is refused
unless named patterns/units carry the evidence (may_claim_strategy_stopped).

Firewall: the engine is forward-only (an observation dated after `now` or not after the last one raises FirewallBreach); reports are
MATURED_RESEARCH_STATE and leave only through MaturedRecord (identity-free numbers, no tickers/dates).
Built on: engine.learning.surprise (continuous_z, surprise_bits), engine.learning.calibration (change_scan for step detection),
engine.research.multiscale (benjamini_hochberg, t_to_p), engine.research.observer (DayRecord -> observation), engine.research.
break_research (BreakCause vocabulary of the probes), engine.research.core, engine.learning.core."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from engine.learning.calibration import change_scan
from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, current_code_hash, stable_hash)
from engine.learning.surprise import continuous_z, surprise_bits
from engine.research.break_research import BreakCause
from engine.research.core import Knowability, MaturedRecord, Namespace, Problem, ResearchQuestion
from engine.research.expectations import SealedLane
from engine.research.multiscale import benjamini_hochberg, t_to_p

SCHEMA_VERSION = "market_expectations.v1"
QUANTITIES = ("opportunity", "volatility", "breadth", "dispersion", "correlation", "sector_leadership", "momentum_persistence",
              "reversal_probability", "gap_behavior", "n_movers", "n_qualifying")
EXPLANATORY = ("holding_period", "optimal_exit_days", "event_share", "microstructure", "pattern_overlap", "pick_hit_rate", "n_picked")
FORECAST = QUANTITIES + ("pick_hit_rate",)                       # the eleven checklist-F quantities plus the system's own hit rate
TARGETS = ("opportunity", "n_qualifying", "n_movers", "pick_hit_rate")   # quantities whose error can open an investigation
RANGES: dict[str, tuple[float, float]] = {
    "opportunity": (0.0, 1.0), "breadth": (0.0, 1.0), "reversal_probability": (0.0, 1.0), "correlation": (-1.0, 1.0),
    "sector_leadership": (0.0, 1.0), "event_share": (0.0, 1.0), "pick_hit_rate": (0.0, 1.0), "volatility": (0.0, math.inf),
    "dispersion": (0.0, math.inf), "gap_behavior": (0.0, math.inf), "n_movers": (0.0, math.inf), "n_qualifying": (0.0, math.inf),
    "n_picked": (0.0, math.inf), "holding_period": (0.0, math.inf), "optimal_exit_days": (0.0, math.inf),
    "microstructure": (-math.inf, math.inf), "pattern_overlap": (0.0, 1.0), "momentum_persistence": (-math.inf, math.inf)}


class Finding(_StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNTESTABLE = "UNTESTABLE"                    # a missing quantity is never read as 'not supported'


class Move(_StrEnum):
    EXPANSION = "EXPANSION"                      # checklist F: reality above expectation
    CONTRACTION = "CONTRACTION"                  # checklist G: reality below expectation
    NONE = "NONE"


class UnitStatus(_StrEnum):
    STRENGTHENED = "STRENGTHENED"
    WEAKENED = "WEAKENED"
    REVERSED = "REVERSED"
    STOPPED_RESPONDING = "STOPPED_RESPONDING"
    STABLE = "STABLE"
    INSUFFICIENT = "INSUFFICIENT"


class Attribution(_StrEnum):
    MARKET_CHANGE = "MARKET_CHANGE"
    SELECTION_FAILURE = "SELECTION_FAILURE"
    BOTH = "BOTH"
    NEITHER = "NEITHER"
    UNKNOWN = "UNKNOWN"


class Conclusion(_StrEnum):
    WITHIN_NOISE = "WITHIN_NOISE"
    EXPLAINED = "EXPLAINED"                      # supported probes/units account for most of the gap
    PARTLY_EXPLAINED = "PARTLY_EXPLAINED"
    SELECTION_FAILURE = "SELECTION_FAILURE"      # the market gave the opportunity, candidate selection missed it
    PATTERN_SPECIFIC = "PATTERN_SPECIFIC"        # named patterns weakened/reversed while the market did not change
    UNEXPLAINED = "UNEXPLAINED"                  # stays unexplained; never rewritten as 'the strategy stopped working'
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    name: str
    quantity: str
    cause: BreakCause                            # the C66 break vocabulary the probe speaks
    text: str


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec("volatility_change", "volatility", BreakCause.MARKET_STRUCTURE_CHANGE, "volatility expanded or compressed"),
    ProbeSpec("momentum_persistence_change", "momentum_persistence", BreakCause.REGIME_CHANGE, "momentum persisted more or decayed"),
    ProbeSpec("reversal_change", "reversal_probability", BreakCause.REGIME_CHANGE, "reversal behaviour increased or faded"),
    ProbeSpec("sector_concentration_change", "sector_leadership", BreakCause.SECTOR_SHIFT, "moves concentrated in fewer sectors or spread out"),
    ProbeSpec("dispersion_change", "dispersion", BreakCause.MARKET_STRUCTURE_CHANGE, "cross-sectional dispersion changed"),
    ProbeSpec("correlation_change", "correlation", BreakCause.MARKET_STRUCTURE_CHANGE, "correlations between names changed"),
    ProbeSpec("breadth_change", "breadth", BreakCause.REGIME_CHANGE, "market breadth changed"),
    ProbeSpec("gap_behavior_change", "gap_behavior", BreakCause.MARKET_STRUCTURE_CHANGE, "opening-gap behaviour changed"),
    ProbeSpec("holding_period_change", "holding_period", BreakCause.MARKET_STRUCTURE_CHANGE, "the time moves take to play out changed"),
    ProbeSpec("optimal_exit_change", "optimal_exit_days", BreakCause.FEATURE_RELATIONSHIP_CHANGED, "the best exit timing moved"),
    ProbeSpec("microstructure_change", "microstructure", BreakCause.LIQUIDITY_CHANGE, "trading microstructure (volume/spread) changed"),
    ProbeSpec("external_information", "event_share", BreakCause.EVENT_ENVIRONMENT, "identifiable event/news share changed"),
    ProbeSpec("pattern_interaction", "pattern_overlap", BreakCause.HIDDEN_INTERACTION, "patterns fired on the same names more or less"),
)


@dataclasses.dataclass(frozen=True)
class ExpectationConfig:
    min_history: int = 20                        # observations before a forecast may be made
    ewma_lambda: float = 0.15
    shrink: float = 0.6                          # weight on the EWMA against the long-run median
    long_window: int = 250
    scale_window: int = 250
    min_errors: int = 10                         # resolved errors before the error scale is learned instead of defaulted
    z_bar: float = 2.5                           # standardised error that opens an investigation
    rel_gap_bar: float = 0.2                     # ... and the relative gap (actual-expected)/expected must also reach this
    recent: int = 5                              # periods in the 'recent' window of an investigation
    reference: int = 60                          # periods in the reference window before it
    min_reference: int = 20
    q_fdr: float = 0.10
    beta_t: float = 2.0                          # |t| of the link between a probe quantity and opportunity
    t_bar: float = 2.0                           # 'established' effect for patterns/units
    stouffer_p: float = 0.01
    max_lag: int = 5
    n_perm: int = 200
    precursor_q: float = 0.10
    min_precursor_pairs: int = 40
    explained_share: float = 0.5

    def validate(self) -> list[str]:
        e = []
        if not 0 < self.ewma_lambda < 1 or not 0 <= self.shrink <= 1:
            e.append("ewma_lambda in (0,1) and shrink in [0,1] required")
        if self.recent < 2 or self.reference < self.min_reference or self.min_reference < 10:
            e.append("recent >= 2, min_reference >= 10 and reference >= min_reference required")
        if self.z_bar <= 0 or self.rel_gap_bar < 0 or not 0 < self.q_fdr < 1:
            e.append("z_bar > 0, rel_gap_bar >= 0 and 0 < q_fdr < 1 required")
        if self.min_history < 5 or self.max_lag < 1 or self.n_perm < 20:
            e.append("min_history >= 5, max_lag >= 1 and n_perm >= 20 required")
        return e


_CODE_HASH: list[str] = []


def _code_hash() -> str:
    """current_code_hash() reads and hashes the engine's sources; it cannot change within a run, so it is computed once."""
    if not _CODE_HASH:
        _CODE_HASH.append(current_code_hash())
    return _CODE_HASH[0]


def _clean(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclasses.dataclass(frozen=True)
class MarketObs:
    """One period's measured market quantities (identity-free numbers). Missing = None, never 0. sector_share: share of the
    period's qualifying movers per opaque sector code."""
    date: str
    values: Mapping[str, float | None]
    sector_share: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def get(self, q: str) -> float | None:
        return _clean(self.values.get(q))

    def validate(self) -> list[str]:
        errs = []
        try:
            as_date(self.date)
        except (ValueError, TypeError):
            return ["date unparseable"]
        for q, v in self.values.items():
            f = _clean(v)
            if v is not None and f is None:
                errs.append(f"{q} is non-finite")
            elif f is not None and q in RANGES and not RANGES[q][0] <= f <= RANGES[q][1]:
                errs.append(f"{q}={f} outside {RANGES[q]}")
        nm, nq = self.get("n_movers"), self.get("n_qualifying")
        if nm is not None and nq is not None and nq > nm + 1e-9:
            errs.append("qualifying 5-10% opportunities exceed all meaningful movers")
        if self.sector_share and (min(self.sector_share.values()) < 0 or sum(self.sector_share.values()) > 1.0 + 1e-6):
            errs.append("sector shares negative or sum above 1")
        return errs


def sector_concentration(sector_movers: Mapping[str, Sequence[int]], min_movers: int = 5) -> float | None:
    """Normalised Herfindahl of band movers over sectors, from observer.sector_table ([names, movers] per code): 0 = spread evenly,
    1 = all in one sector. None when there are too few movers or one sector (concentration is undefined, not zero)."""
    m = np.array([v[1] for v in sector_movers.values()], dtype="float64")
    if len(m) < 2 or m.sum() < min_movers:
        return None
    s = m / m.sum()
    return float((np.sum(s ** 2) - 1.0 / len(s)) / (1.0 - 1.0 / len(s)))


def observation_from_day_record(rec, extras: Mapping[str, float | None] | None = None) -> MarketObs:
    """engine.research.observer.DayRecord -> MarketObs. Opportunity is the share of the eligible universe that moved >= 5%; the
    quantities the daily record cannot see (correlation, momentum persistence, reversal probability, holding period, ...) come
    from `extras` (cross_section.market_structure, regimes indicators) and stay None when absent."""
    mk, md = rec.market, rec.model
    vals: dict[str, float | None] = {
        "opportunity": _clean(mk.get("mover_rate")), "volatility": _clean(0.5 * (mk.get("p95", float("nan")) - mk.get("p05", float("nan")))),
        "breadth": _clean(mk.get("breadth_up")), "dispersion": _clean(mk.get("dispersion")), "gap_behavior": _clean(mk.get("gap_median_abs")),
        "n_movers": _clean(mk.get("n_moved")), "n_qualifying": _clean(sum(v for k, v in rec.bands.items() if k.endswith("5_10"))),
        "sector_leadership": sector_concentration(mk.get("sector_movers", {})), "microstructure": _clean(mk.get("median_volume_ratio")),
        "pick_hit_rate": _clean(md.get("pick_hit_rate")), "n_picked": _clean(md.get("n_picked"))}
    vals.update({k: _clean(v) for k, v in (extras or {}).items()})
    tot = sum(v[1] for v in mk.get("sector_movers", {}).values())
    share = {k: v[1] / tot for k, v in mk.get("sector_movers", {}).items()} if tot > 0 else {}
    return MarketObs(rec.day, vals, share)


# ------------------------------------------------------------------------------------------------ expectation ledger

@dataclasses.dataclass(frozen=True)
class MarketExpectation:
    """Everything the engine believed about the coming period BEFORE it: immutable (tuples, frozen). `prev_hash` is the head of the
    archive lane it extends, so a rewrite of any earlier entry is detected by MarketExpectationLedger.verify()."""
    for_period: str
    made_at: str
    history_through: str
    quantities: tuple[tuple[str, float, float], ...]        # (name, expected, scale)
    n_history: int
    code_hash: str
    prev_hash: str
    digest: str = ""

    def body(self) -> dict[str, Any]:
        return {"for": self.for_period, "made": self.made_at, "through": self.history_through, "q": self.quantities,
                "n": self.n_history, "code": self.code_hash, "prev": self.prev_hash}

    def compute_digest(self) -> str:
        return stable_hash(self.body(), 24)

    def expected(self, q: str) -> float | None:
        return next((e for n, e, _ in self.quantities if n == q), None)

    def scale(self, q: str) -> float | None:
        return next((s for n, _, s in self.quantities if n == q), None)

    @classmethod
    def from_body(cls, b: Mapping[str, Any]) -> "MarketExpectation":
        return cls(b["for"], b["made"], b["through"], tuple((str(n), float(e), float(s)) for n, e, s in b["q"]), int(b["n"]), b["code"],
                   b["prev"], b["digest"])

    def validate(self) -> list[str]:
        errs = []
        if as_date(self.made_at) >= as_date(self.for_period):
            errs.append("made_at must be strictly before the period it forecasts")
        if as_date(self.history_through) > as_date(self.made_at):
            errs.append("history_through is after made_at")
        for n, e, s in self.quantities:
            if not (math.isfinite(e) and math.isfinite(s) and s > 0):
                errs.append(f"{n}: expected/scale not finite or scale <= 0")
        return errs


LANE_MARKET = "mex68"


class MarketExpectationLedger:
    """Append-only store on an ARCHIVE CHAIN LANE (engine.research.expectations.SealedLane over engine.learning.archive.ChainFile, kind
    'mex68' - the same hash-chained lanes as P01's exp68/out68/err68; P06 de-duplication: this module keeps no private hash chain). An
    expectation's `prev_hash` is the lane head it extends (so its link IS the archive link), there is no update or delete, and
    `verify` re-reads the chain and compares every cached entry with the body the chain holds. `root=None` keeps the lane in memory;
    with a root the ledger reloads from disk."""

    def __init__(self, root=None):
        self.lane = SealedLane(root, LANE_MARKET)
        lines = self.lane.lines()
        self._items: list[MarketExpectation] = [MarketExpectation.from_body(ln["body"]) for ln in lines]

    def __len__(self) -> int:
        return len(self._items)

    def head(self) -> str:
        return self.lane.head

    def append(self, exp: MarketExpectation) -> MarketExpectation:
        errs = exp.validate()
        if errs:
            raise ValueError("bad expectation: " + "; ".join(errs))
        if exp.prev_hash != self.head():
            raise FirewallBreach("expectation does not extend the chain head: history cannot be rewritten")
        if exp.digest != exp.compute_digest():
            raise FirewallBreach("expectation digest does not match its content")
        if self._items and as_date(exp.for_period) <= as_date(self._items[-1].for_period):
            raise FirewallBreach("expectation for a period that already has one (or an earlier one)")
        self.lane.append({**exp.body(), "digest": exp.digest})
        self._items.append(exp)
        return exp

    def get(self, period) -> MarketExpectation | None:
        p = as_date(period).isoformat()
        return next((e for e in self._items if e.for_period == p), None)

    def entries(self) -> tuple[MarketExpectation, ...]:
        return tuple(self._items)

    def verify(self) -> list[int]:
        """Indices whose content no longer matches its digest, differs from the body on the archive lane, or does not extend the lane
        position it was appended at; -1 when the lane itself is broken (edited, truncated, reordered). Empty = intact."""
        rep = self.lane.verify()
        lines = self.lane.lines()
        bad = [] if rep["ok"] else [-1]
        if len(lines) != len(self._items):
            bad.append(-1)
        for i, (e, ln) in enumerate(zip(self._items, lines)):
            if e.digest != e.compute_digest() or ln["body"].get("digest") != e.digest or e.prev_hash != ln["prev"]:
                bad.append(i)
        return sorted(set(bad))


# ------------------------------------------------------------------------------------------------ forecasting and error scale

def series_of(history: Sequence[MarketObs], q: str) -> np.ndarray:
    return np.array([np.nan if (v := o.get(q)) is None else v for o in history], dtype="float64")


def forecast_quantity(x: np.ndarray, cfg: ExpectationConfig) -> float | None:
    """Expectation for the next period from the past only: EWMA of the recent values shrunk toward the long-run median (a pure EWMA
    chases noise; a pure median ignores a level that really moved). None until min_history finite values exist."""
    v = x[np.isfinite(x)]
    if len(v) < cfg.min_history:
        return None
    w = (1 - cfg.ewma_lambda) ** np.arange(len(v))[::-1]
    return float(cfg.shrink * np.sum(w * v) / np.sum(w) + (1 - cfg.shrink) * np.median(v[-cfg.long_window:]))


def error_scale(errors: Sequence[float], series: np.ndarray, cfg: ExpectationConfig) -> tuple[float, str]:
    """Robust scale of the expectation error: from resolved errors once min_errors exist, else from the spread of the series itself
    (an honest upper bound before any error is known), floored so a constant series cannot produce an infinite z."""
    e = np.asarray(errors[-cfg.scale_window:], dtype="float64")
    if len(e) >= cfg.min_errors:
        s = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        src = "errors"
    else:
        v = series[np.isfinite(series)]
        s = 1.4826 * float(np.median(np.abs(v - np.median(v)))) if len(v) else 0.0
        src = "series"
    floor = 1e-3 * max(abs(float(np.nanmean(series))) if np.isfinite(series).any() else 0.0, 1e-6)
    return max(s, floor), src


@dataclasses.dataclass(frozen=True)
class MarketSurprise:
    quantity: str
    expected: float
    actual: float
    scale: float
    error: float
    z: float
    bits: float
    relative_gap: float | None                   # (actual - expected) / |expected|; None when expected is ~0

    @property
    def move(self) -> Move:
        return Move.NONE if self.z == 0 else (Move.EXPANSION if self.z > 0 else Move.CONTRACTION)


def resolve(exp: MarketExpectation, obs: MarketObs, now) -> dict[str, MarketSurprise]:
    """Compare the recorded expectation with what arrived. The observation must be for the forecast period and must not be dated
    after `now`; quantities missing from the observation are simply absent from the result."""
    if obs.date != exp.for_period:
        raise ValueError(f"observation {obs.date} is not the forecast period {exp.for_period}")
    if as_date(obs.date) > as_date(now):
        raise FirewallBreach(f"observation {obs.date} is after now={as_date(now)}")
    out = {}
    for q, e, s in exp.quantities:
        a = obs.get(q)
        if a is None:
            continue
        z = continuous_z(e, a, s)
        out[q] = MarketSurprise(q, e, a, s, a - e, z, surprise_bits(z), (a - e) / abs(e) if abs(e) > 1e-9 else None)
    return out


# ------------------------------------------------------------------------------------------------ statistics of a shift

def _rho(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    if len(v) < 6 or np.std(v) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(v[:-1], v[1:])[0, 1], 0.0, 0.9))


def _eff_n(x: np.ndarray) -> float:
    n = int(np.isfinite(x).sum())
    r = _rho(x)
    return max(1.0, n * (1 - r) / (1 + r))


def _robust_sd(v: np.ndarray) -> float:
    s = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    return s if s > 0 else float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


@dataclasses.dataclass(frozen=True)
class ShiftStat:
    mean_ref: float
    mean_recent: float
    sd_ref: float
    z: float
    p: float
    n_ref: int
    n_recent: int

    @property
    def shift(self) -> float:
        return self.mean_recent - self.mean_ref


def shift_test(x: np.ndarray, n_recent: int, min_reference: int = 20) -> ShiftStat | None:
    """Recent window (last n_recent) against the reference window before it, on a robust reference spread and autocorrelation-
    adjusted effective sample sizes (t distribution with the recent count as df: five periods do not get a normal test).
    None when either window is too short - the caller reports UNTESTABLE."""
    x = np.asarray(x, dtype="float64")
    rec, ref = x[-n_recent:], x[:-n_recent]
    rv, fv = rec[np.isfinite(rec)], ref[np.isfinite(ref)]
    if len(rv) < max(2, n_recent // 2) or len(fv) < min_reference:
        return None
    sd = _robust_sd(fv)
    if sd <= 1e-12:
        return None
    ne_rec, ne_ref = max(1.0, len(rv) * (1 - _rho(fv)) / (1 + _rho(fv))), _eff_n(ref)
    z = (float(rv.mean()) - float(fv.mean())) / (sd * math.sqrt(1.0 / ne_rec + 1.0 / ne_ref))
    p = float(2 * sps.t.sf(abs(z), df=max(len(rv) - 1.0, 2.0)))
    return ShiftStat(float(fv.mean()), float(rv.mean()), sd, float(z), p, len(fv), len(rv))


def slope_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """OLS slope of y on x (pairs where both exist) and its t statistic with the residual-autocorrelation-adjusted n. None below
    12 pairs or with a constant x."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 12 or np.std(x[ok]) < 1e-12:
        return None
    xs, ys = x[ok], y[ok]
    b = float(np.cov(xs, ys, ddof=1)[0, 1] / np.var(xs, ddof=1))
    res = ys - ys.mean() - b * (xs - xs.mean())
    ne = max(_eff_n(res), 3.0)
    se = math.sqrt(max(float(np.var(res, ddof=1)), 1e-18) / (np.var(xs, ddof=1) * ne))
    return b, b / se


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    name: str
    quantity: str
    cause: BreakCause
    finding: Finding
    shift_z: float | None
    p: float | None
    q: float | None
    beta: float | None
    beta_t: float | None
    implied: float | None                        # beta * shift: the part of the opportunity gap this quantity's move accounts for
    implied_share: float | None
    reason: str


def _probe(spec: ProbeSpec, opp: np.ndarray, x: np.ndarray, n_recent: int, cfg: ExpectationConfig) -> ProbeResult:
    st = shift_test(x, n_recent, cfg.min_reference)
    if st is None:
        return ProbeResult(spec.name, spec.quantity, spec.cause, Finding.UNTESTABLE, None, None, None, None, None, None, None,
                           f"{spec.quantity} missing or too short to compare windows")
    lk = slope_t(x[:-n_recent], opp[:-n_recent])
    if lk is None:
        return ProbeResult(spec.name, spec.quantity, spec.cause, Finding.UNTESTABLE, st.z, st.p, None, None, None, None, None,
                           f"no measurable link between {spec.quantity} and opportunity in the reference window")
    b, t = lk
    return ProbeResult(spec.name, spec.quantity, spec.cause, Finding.NOT_SUPPORTED, st.z, st.p, None, b, t, b * st.shift, None, "")


def _judge_probes(raw: list[ProbeResult], gap: float, cfg: ExpectationConfig) -> list[ProbeResult]:
    """BH across the probes that were testable, then the three-part rule: the shift is significant, the link to opportunity is
    significant, and shift x link pushes opportunity the way the surprise went."""
    q = benjamini_hochberg([r.p for r in raw])
    out = []
    for r, qi in zip(raw, q):
        if r.finding == Finding.UNTESTABLE:
            out.append(r)
            continue
        implied, beta_t = r.implied, r.beta_t
        if implied is None or beta_t is None:          # only UNTESTABLE probes lack a link, and they were passed through above
            out.append(r)
            continue
        share = implied / gap if abs(gap) > 1e-12 else None
        if qi is None or qi > cfg.q_fdr:
            why, f = f"{r.quantity} did not shift beyond chance (q={qi:.3f})" if qi is not None else "no p-value", Finding.NOT_SUPPORTED
        elif abs(beta_t) < cfg.beta_t:
            why, f = f"{r.quantity} shifted but is not linked to opportunity (|t|={abs(beta_t):.1f})", Finding.NOT_SUPPORTED
        elif implied * gap <= 0:
            why, f = f"{r.quantity} shifted but its link points against the surprise", Finding.NOT_SUPPORTED
        else:
            why, f = f"{r.quantity} shifted (z={r.shift_z:.1f}, q={qi:.3f}) and accounts for {share:.0%} of the gap", Finding.SUPPORTED
        out.append(dataclasses.replace(r, finding=f, q=qi, implied_share=share, reason=why))
    return out


def combined_small_changes(results: Sequence[ProbeResult], cfg: ExpectationConfig) -> ProbeResult:
    """'Multiple small changes combining into a large change' as its own test: the probes that are individually NOT supported but
    testable are pooled by Stouffer's method, each z signed by the direction its link says raises opportunity. A pooled z that is
    significant while no single one was is the signature; fewer than three usable probes means UNTESTABLE."""
    pool = [r for r in results if r.finding == Finding.NOT_SUPPORTED and r.shift_z is not None and r.beta is not None
            and (r.q is None or r.q > cfg.q_fdr)]
    if len(pool) < 3:
        return ProbeResult("combined_small_changes", "*", BreakCause.MARKET_STRUCTURE_CHANGE, Finding.UNTESTABLE, None, None, None, None, None, None, None,
                           "fewer than three testable sub-threshold probes")
    z = float(sum(math.copysign(1.0, float(r.beta or 0.0)) * float(r.shift_z or 0.0) for r in pool) / math.sqrt(len(pool)))
    p = float(2 * sps.norm.sf(abs(z)))
    ok = p < cfg.stouffer_p
    return ProbeResult("combined_small_changes", "*", BreakCause.MARKET_STRUCTURE_CHANGE, Finding.SUPPORTED if ok else Finding.NOT_SUPPORTED,
                       z, p, p, None, None, None, None,
                       f"pooled shift of {len(pool)} individually insignificant quantities: z={z:.2f}, p={p:.4f}")


def joint_attribution(history: Sequence[MarketObs], supported: Sequence[ProbeResult], n_recent: int, gap: float,
                      target: str = "opportunity") -> float | None:
    """Share of the opportunity shift the supported quantities explain TOGETHER (they overlap, so single shares do not add): a
    lightly ridged regression of the target on all of them in the reference window, applied to the recent shift. None when no
    supported probe or a degenerate window; may exceed 1 (over-explained) and is reported as is."""
    cols = [series_of(history, r.quantity) for r in supported if r.quantity != "*"]
    if not cols or abs(gap) < 1e-12:
        return None
    y = series_of(history, target)[:-n_recent]
    X = np.column_stack([c[:-n_recent] for c in cols])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 12 + len(cols):
        return None
    Xr, yr = X[ok], y[ok]
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (Xr - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + 1e-3 * len(Z) * np.eye(Z.shape[1]), Z.T @ (yr - yr.mean()))
    rec = np.column_stack([c[-n_recent:] for c in cols])
    rec = rec[np.isfinite(rec).all(axis=1)]
    if len(rec) == 0:
        return None
    return float(((rec.mean(axis=0) - mu) / sd) @ beta / gap)


# ------------------------------------------------------------------------------------------------ patterns, sectors, stocks

@dataclasses.dataclass(frozen=True)
class UnitShift:
    unit: str
    status: UnitStatus
    mean_ref: float | None
    mean_recent: float | None
    z: float | None
    q: float | None
    newly_active: bool = False                   # not established before, established now ('a previously weak pattern became strong')


def _tstat(v: np.ndarray, ne: float) -> float:
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return 0.0 if sd <= 1e-12 else float(v.mean() / (sd / math.sqrt(max(ne, 1.0))))


def unit_shifts(series_by_unit: Mapping[str, Sequence[float]], n_recent: int, cfg: ExpectationConfig) -> list[UnitShift]:
    """Checklist G 'which patterns / sectors / stocks changed': each unit's own series (effect per period, mover share, response)
    compared recent-vs-reference, BH across units. Status: REVERSED = the established sign flipped, STOPPED_RESPONDING = the effect
    fell below half of its reference size, WEAKENED / STRENGTHENED = same sign, smaller / larger. Units too short are INSUFFICIENT."""
    tests = {}
    for u, s in series_by_unit.items():
        x = np.asarray(s, dtype="float64")
        tests[u] = (x, shift_test(x, n_recent, cfg.min_reference))
    qv = dict(zip(tests, benjamini_hochberg([t[1].p if t[1] is not None else None for t in tests.values()])))
    out = []
    for u, (x, st) in tests.items():
        if st is None:
            out.append(UnitShift(u, UnitStatus.INSUFFICIENT, None, None, None, None))
            continue
        ref, rec = x[:-n_recent], x[-n_recent:]
        ref, rec = ref[np.isfinite(ref)], rec[np.isfinite(rec)]
        est_ref = abs(_tstat(ref, _eff_n(x[:-n_recent]))) >= cfg.t_bar
        q = qv[u]
        new = (not est_ref) and abs(_tstat(rec, max(len(rec) * 1.0, 1.0))) >= cfg.t_bar and q is not None and q <= cfg.q_fdr
        if q is None or q > cfg.q_fdr:
            status = UnitStatus.STABLE
        elif est_ref and st.mean_ref * st.mean_recent < 0:
            status = UnitStatus.REVERSED
        elif est_ref and abs(st.mean_recent) < 0.5 * abs(st.mean_ref):
            status = UnitStatus.STOPPED_RESPONDING
        elif abs(st.mean_recent) < abs(st.mean_ref):
            status = UnitStatus.WEAKENED
        else:
            status = UnitStatus.STRENGTHENED
        out.append(UnitShift(u, status, st.mean_ref, st.mean_recent, st.z, q, new))
    return out


# ------------------------------------------------------------------------------------------------ 'did candidate selection fail?'

@dataclasses.dataclass(frozen=True)
class SelectionCheck:
    attribution: Attribution
    universe_z: float | None
    hit_rate_ref: float | None
    hit_rate_recent: float | None
    hit_z: float | None
    reason: str


def selection_check(history: Sequence[MarketObs], move: Move, n_recent: int, cfg: ExpectationConfig) -> SelectionCheck:
    """Separate 'the market changed' from 'the candidate selection failed'. Universe side: did qualifying-opportunity supply move
    in the surprise direction? Selection side: did the hit rate of the picks fall (two-proportion z on pooled picks)? A
    contraction with an intact universe and a falling hit rate is a SELECTION failure; a contraction where supply fell and the
    hit rate held is a MARKET change; both moving is BOTH; neither is NEITHER. Missing pick data is UNKNOWN, never assumed fine."""
    uni = shift_test(series_of(history, "n_qualifying" if np.isfinite(series_of(history, "n_qualifying")).any() else "opportunity"),
                     n_recent, cfg.min_reference)
    hr, npk = series_of(history, "pick_hit_rate"), series_of(history, "n_picked")
    ok = np.isfinite(hr) & np.isfinite(npk) & (npk > 0)
    ref_ok, rec_ok = ok[:-n_recent], ok[-n_recent:]
    if uni is None or ref_ok.sum() < cfg.min_reference or rec_ok.sum() < 1:
        return SelectionCheck(Attribution.UNKNOWN, uni.z if uni else None, None, None, None, "pick or supply data missing")
    w_ref, w_rec = npk[:-n_recent][ref_ok], npk[-n_recent:][rec_ok]
    p_ref = float((hr[:-n_recent][ref_ok] * w_ref).sum() / w_ref.sum())
    p_rec = float((hr[-n_recent:][rec_ok] * w_rec).sum() / w_rec.sum())
    pool = (p_ref * w_ref.sum() + p_rec * w_rec.sum()) / (w_ref.sum() + w_rec.sum())
    se = math.sqrt(max(pool * (1 - pool), 1e-12) * (1 / w_ref.sum() + 1 / w_rec.sum()))
    hz = (p_rec - p_ref) / se
    want = 1.0 if move == Move.EXPANSION else -1.0
    market = uni.z * want >= 2.0 and uni.p <= 0.05
    selection = hz <= -2.0                                   # picks capture a smaller share of what moved, whatever the market did
    att = (Attribution.BOTH if market and selection else Attribution.MARKET_CHANGE if market else
           Attribution.SELECTION_FAILURE if selection else Attribution.NEITHER)
    return SelectionCheck(att, float(uni.z), p_ref, p_rec, float(hz),
                          f"supply z={uni.z:.1f}; pick hit rate {p_ref:.3f} -> {p_rec:.3f} (z={hz:.1f})")


# ------------------------------------------------------------------------------------------------ precursors

@dataclasses.dataclass(frozen=True)
class PrecursorResult:
    probe: str
    quantity: str
    lag: int
    r: float
    p: float                                     # autocorrelation-adjusted analytic p (Bartlett effective n)
    q: float | None                              # BH q over every probe x lag tested; None when above precursor_q
    replicated: bool
    n: int
    p_perm: float = 1.0                          # circular-shift null p: a second, distribution-free opinion

    @property
    def usable(self) -> bool:
        return self.replicated and self.q is not None and self.p_perm <= 0.05


def precursor_scan(history: Sequence[MarketObs], target_z: Sequence[float], cfg: ExpectationConfig, seed: int = 0,
                   probes: Sequence[ProbeSpec] = PROBES) -> list[PrecursorResult]:
    """Can the system see the cause BEFORE the next occurrence? For each probe quantity and lag 1..max_lag: the correlation of the
    quantity at t-lag with the target error z at t: an autocorrelation-adjusted analytic p (BH across every probe x lag tested),
    confirmed by a seeded circular-shift null (autocorrelation preserved), and required to keep its sign and size in BOTH halves. `target_z[i]` is the
    error for history[i] (NaN where unresolved). Only complete pairs inside the history are used; nothing after it exists."""
    rng = np.random.default_rng(seed)
    z = np.asarray(target_z, dtype="float64")
    n = len(z)
    rows = []
    for spec in probes:
        x = series_of(history, spec.quantity)
        for lag in range(1, cfg.max_lag + 1):
            xs, zs = x[: n - lag], z[lag:]
            ok = np.isfinite(xs) & np.isfinite(zs)
            if ok.sum() < cfg.min_precursor_pairs or np.std(xs[ok]) < 1e-12 or np.std(zs[ok]) < 1e-12:
                continue
            a, b = xs[ok], zs[ok]
            r = float(np.corrcoef(a, b)[0, 1])
            null = np.array([abs(np.corrcoef(np.roll(a, int(rng.integers(1, len(a)))), b)[0, 1]) for _ in range(cfg.n_perm)])
            p_perm = float((1 + np.sum(null >= abs(r))) / (1 + cfg.n_perm))
            rx, rz = _rho(a), _rho(b)
            n_eff = max(len(a) * (1 - rx * rz) / (1 + rx * rz), 5.0)
            p = float(2 * sps.t.sf(abs(r) * math.sqrt((n_eff - 2) / max(1 - r * r, 1e-12)), df=n_eff - 2))
            h = len(a) // 2
            r1, r2 = np.corrcoef(a[:h], b[:h])[0, 1], np.corrcoef(a[h:], b[h:])[0, 1]
            rep = bool(np.sign(r1) == np.sign(r2) == np.sign(r) and min(abs(r1), abs(r2)) >= 0.5 * abs(r) and min(abs(r1), abs(r2)) >= 0.1)
            rows.append((spec, lag, r, p, rep, int(ok.sum()), p_perm))
    qv = benjamini_hochberg([row[3] for row in rows])
    return [PrecursorResult(s.name, s.quantity, lag, r, p, q if (q is not None and q <= cfg.precursor_q) else None, rep, n_, pp)
            for (s, lag, r, p, rep, n_, pp), q in zip(rows, qv)]


# ------------------------------------------------------------------------------------------------ the investigation

@dataclasses.dataclass(frozen=True)
class InvestigationReport:
    period: str
    target: str
    move: Move
    expected: float
    actual: float
    z: float
    relative_gap: float | None
    probes: tuple[ProbeResult, ...]
    combined: ProbeResult
    joint_share: float | None
    patterns: tuple[UnitShift, ...]
    sectors: tuple[UnitShift, ...]
    stocks: tuple[UnitShift, ...]
    selection: SelectionCheck
    precursors: tuple[PrecursorResult, ...]
    conclusion: Conclusion
    regime_changed: bool | None                  # from the regime layer when it was supplied; None = not asked
    knowability: Knowability
    notes: tuple[str, ...] = ()

    def supported(self) -> tuple[ProbeResult, ...]:
        return tuple(p for p in self.probes if p.finding == Finding.SUPPORTED) + ((self.combined,) if self.combined.finding == Finding.SUPPORTED else ())

    def untestable(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.probes if p.finding == Finding.UNTESTABLE)

    def specifics(self) -> list[str]:
        """Named units carrying evidence of a decline: weakened / reversed / stopped patterns, sectors and stocks."""
        bad = (UnitStatus.WEAKENED, UnitStatus.REVERSED, UnitStatus.STOPPED_RESPONDING)
        return [f"{kind}:{u.unit}:{u.status.value}" for kind, us in (("pattern", self.patterns), ("sector", self.sectors), ("stock", self.stocks))
                for u in us if u.status in bad]

    def usable_precursors(self) -> tuple[PrecursorResult, ...]:
        return tuple(p for p in self.precursors if p.usable)

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 16)


def may_claim_strategy_stopped(rep: InvestigationReport) -> bool:
    """Checklist G refuses 'the strategy stopped working' until WHAT stopped is known: only when the report names at least one
    weakened/reversed/stopped pattern (evidence at pattern level), the contraction is real (not a selection failure alone or
    within noise) and no sub-threshold explanation was left untested for lack of data."""
    if rep.move != Move.CONTRACTION or rep.conclusion in (Conclusion.WITHIN_NOISE, Conclusion.INSUFFICIENT_DATA):
        return False
    pat_bad = [u for u in rep.patterns if u.status in (UnitStatus.WEAKENED, UnitStatus.REVERSED, UnitStatus.STOPPED_RESPONDING)]
    return bool(pat_bad) and rep.selection.attribution != Attribution.SELECTION_FAILURE


def investigate(history: Sequence[MarketObs], surprise: MarketSurprise, period: str, cfg: ExpectationConfig | None = None,
                target_z: Sequence[float] = (), pattern_effects: Mapping[str, Sequence[float]] | None = None,
                sector_series: Mapping[str, Sequence[float]] | None = None, stock_series: Mapping[str, Sequence[float]] | None = None,
                regime_changed: bool | None = None, seed: int = 0) -> InvestigationReport:
    """Test every checklist-F/G explanation for one target surprise. `history` ends with the surprising period (nothing later can
    exist). Windows: the last cfg.recent periods against the cfg.reference periods before them. The unit series must be aligned to
    the history (same length, oldest first)."""
    cfg = cfg or ExpectationConfig()
    n_rec = min(cfg.recent, max(2, len(history) // 4))
    target = surprise.quantity
    opp = series_of(history, target)
    st = shift_test(opp, n_rec, cfg.min_reference)
    gap = st.shift if st is not None else surprise.error
    raw = [_probe(s, opp, series_of(history, s.quantity), n_rec, cfg) for s in PROBES]
    judged = _judge_probes(raw, gap, cfg)
    comb = combined_small_changes(judged, cfg)
    supp = [p for p in judged if p.finding == Finding.SUPPORTED]
    joint = joint_attribution(history, supp, n_rec, gap, target) if st is not None else None
    pats = unit_shifts(pattern_effects or {}, n_rec, cfg)
    secs = unit_shifts(sector_series or {}, n_rec, cfg)
    stks = unit_shifts(stock_series or {}, n_rec, cfg)
    sel = selection_check(history, surprise.move, n_rec, cfg)
    prec = precursor_scan(history, target_z, cfg, seed) if len(target_z) == len(history) and len(history) else []
    changed = [u for u in pats if u.status != UnitStatus.STABLE and u.status != UnitStatus.INSUFFICIENT]
    if st is None:
        concl = Conclusion.INSUFFICIENT_DATA
    elif abs(surprise.z) < cfg.z_bar:
        concl = Conclusion.WITHIN_NOISE
    elif sel.attribution == Attribution.SELECTION_FAILURE:
        concl = Conclusion.SELECTION_FAILURE
    elif joint is not None and joint >= cfg.explained_share:
        concl = Conclusion.EXPLAINED
    elif supp or comb.finding == Finding.SUPPORTED or (joint is not None and joint > 0):
        concl = Conclusion.PARTLY_EXPLAINED
    elif changed:
        concl = Conclusion.PATTERN_SPECIFIC
    else:
        concl = Conclusion.UNEXPLAINED
    notes = []
    if regime_changed is None:
        notes.append("regime layer not consulted")
    if joint is not None and joint > 1.25:
        notes.append("supported quantities over-explain the gap (joint share > 1.25): treat single shares with caution")
    if [p.name for p in judged if p.finding == Finding.UNTESTABLE]:
        notes.append("untestable explanations remain: " + ", ".join(p.name for p in judged if p.finding == Finding.UNTESTABLE))
    know = (Knowability.PREDICTABLE if any(p.usable for p in prec) else Knowability.UNKNOWN if concl == Conclusion.UNEXPLAINED else
            Knowability.POTENTIALLY_PREDICTABLE if concl in (Conclusion.EXPLAINED, Conclusion.PARTLY_EXPLAINED) else Knowability.WEAKLY_PREDICTABLE)
    return InvestigationReport(period, target, surprise.move, surprise.expected, surprise.actual, surprise.z, surprise.relative_gap,
                               tuple(judged), comb, joint, tuple(pats), tuple(secs), tuple(stks), sel, tuple(prec), concl, regime_changed,
                               know, tuple(notes))


def detect_level_shift(x: Sequence[float], threshold: float = 3.5) -> tuple[float, int | None]:
    """Where did a quantity's level step (past-only scan of the history handed in)? Delegates to
    engine.learning.calibration.change_scan, whose false-alarm level is stated and tested; NaNs are dropped."""
    v = np.asarray(x, dtype="float64")
    v = v[np.isfinite(v)]
    return change_scan(v, threshold=threshold) if len(v) >= 20 else (0.0, None)


def investigation_questions(rep: InvestigationReport, created_real: str, evidence_through: str) -> list[ResearchQuestion]:
    """Turn what the investigation could not settle into research objects (C66 section 40): unexplained gaps, untestable
    explanations, and precursors worth replicating. Identity-free text."""
    qs = []
    kind = "above" if rep.move == Move.EXPANSION else "below"
    if rep.conclusion == Conclusion.UNEXPLAINED:
        qs.append(ResearchQuestion.make(f"What produced the market {rep.target} that arrived {kind} expectation when no tested explanation held?",
                                        "market_expectation", Problem.VOLATILITY, created_real, evidence_through,
                                        "a quantity or unit explains the gap and replicates in a later period",
                                        "no explanation replicates; the gap stays labelled unknown"))
    for name in rep.untestable():
        qs.append(ResearchQuestion.make(f"Can {name} be measured so the {kind}-expectation {rep.target} gap can be tested against it?",
                                        "market_expectation", Problem.DATA_QUALITY, created_real, evidence_through,
                                        "the quantity is recorded for the reference and recent windows", "the quantity cannot be measured"))
    for pr in rep.usable_precursors()[:3]:
        qs.append(ResearchQuestion.make(f"Does {pr.probe} at lag {pr.lag} keep predicting the {rep.target} error out of sample?",
                                        "market_expectation", Problem.VOLATILITY, created_real, evidence_through,
                                        "the sign and size hold on a period not used to find it", "it disappears on fresh periods"))
    return qs


# ------------------------------------------------------------------------------------------------ the engine

@dataclasses.dataclass(frozen=True)
class MarketStepResult:
    date: str
    resolved: Mapping[str, MarketSurprise]
    investigation: InvestigationReport | None
    next_expectation: MarketExpectation | None
    reason_no_expectation: str
    chain_intact: bool


class MarketExpectationEngine:
    """Forward-only market expectation machine. step(now, obs) resolves yesterday's expectation, investigates a large error, and
    records the expectation for the next period. Every decision uses observations dated <= now."""

    def __init__(self, cfg: ExpectationConfig | None = None, seed: int = 0, root=None):
        self.cfg = cfg or ExpectationConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("bad ExpectationConfig: " + "; ".join(errs))
        self.seed = seed
        self.history: list[MarketObs] = []
        self.ledger = MarketExpectationLedger(root)
        self.errors: dict[str, list[float]] = {q: [] for q in FORECAST}
        self.target_z: dict[str, list[float]] = {q: [] for q in TARGETS}
        self.reports: list[InvestigationReport] = []

    def step(self, now, obs: MarketObs, pattern_effects: Mapping[str, Sequence[float]] | None = None,
             sector_series: Mapping[str, Sequence[float]] | None = None, stock_series: Mapping[str, Sequence[float]] | None = None,
             regime_changed: bool | None = None, next_period=None) -> MarketStepResult:
        d = as_date(obs.date)
        if d > as_date(now):
            raise FirewallBreach(f"observation {obs.date} is after now={as_date(now)}")
        if self.history and d <= as_date(self.history[-1].date):
            raise FirewallBreach(f"observation {obs.date} is not after the last one {self.history[-1].date}")
        errs = obs.validate()
        if errs:
            raise ValueError("bad observation: " + "; ".join(errs))
        exp = self.ledger.get(obs.date)
        resolved = resolve(exp, obs, now) if exp is not None else {}
        for q, s in resolved.items():
            self.errors[q].append(s.error)
        for q in TARGETS:                                    # NaN where nothing was resolved keeps the z series aligned to history
            self.target_z[q].append(resolved[q].z if q in resolved else float("nan"))
        self.history.append(obs)
        report = None
        for q in TARGETS:
            hit = resolved.get(q)
            if hit is not None and abs(hit.z) >= self.cfg.z_bar and (hit.relative_gap is None or abs(hit.relative_gap) >= self.cfg.rel_gap_bar):
                report = investigate(self.history, hit, obs.date, self.cfg, self.target_z[q], pattern_effects, sector_series, stock_series,
                                     regime_changed, self.seed)
                self.reports.append(report)
                break
        nxt, why = self._forecast(d, next_period)
        return MarketStepResult(obs.date, resolved, report, nxt, why, not self.ledger.verify())

    def _forecast(self, d: dt.date, next_period) -> tuple[MarketExpectation | None, str]:
        target = as_date(next_period) if next_period is not None else d + dt.timedelta(days=1)
        if target <= d:
            raise ValueError("next_period must be after the observation")
        qs = []
        for q in FORECAST:
            x = series_of(self.history, q)
            f = forecast_quantity(x, self.cfg)
            if f is not None:
                qs.append((q, f, error_scale(self.errors[q], x, self.cfg)[0]))
        if not qs:
            return None, f"fewer than {self.cfg.min_history} observations of any quantity"
        exp = MarketExpectation(target.isoformat(), d.isoformat(), self.history[-1].date, tuple(qs), len(self.history), _code_hash(),
                                self.ledger.head())
        exp = dataclasses.replace(exp, digest=exp.compute_digest())
        return self.ledger.append(exp), ""

    def surprise_history(self, q: str) -> np.ndarray:
        return np.array(self.target_z.get(q, []), dtype="float64")

    def feed_tracker(self, tracker, result: MarketStepResult, now) -> int:
        """Hand the resolved errors to the shared engine.learning.surprise.SurpriseTracker (cell 'market|<quantity>'), so market
        surprises join the same ledger, FDR control and priorities as pattern surprises. Returns how many records were added."""
        exp = self.ledger.get(result.date)
        if exp is None:
            raise ValueError(f"no expectation was recorded for {result.date}")
        n = 0
        for q, s in result.resolved.items():
            tracker.observe(f"market|{q}", s.expected, s.actual, exp.made_at, result.date, now, scale=s.scale)
            n += 1
        return n

    def matured_report(self, rep: InvestigationReport, created_real: str) -> MaturedRecord:
        """Identity-free research-world record of an investigation; reaches the trader only through MaturedRecord.gate(now)."""
        prov = Provenance(created_real=created_real, learned_at=rep.period, code_hash=_code_hash(), outcomes_seen_through=rep.period)
        payload = {"target": rep.target, "move": rep.move.value, "z": rep.z, "conclusion": rep.conclusion.value,
                   "supported": [p.name for p in rep.supported()], "untestable": list(rep.untestable()),
                   "attribution": rep.selection.attribution.value, "joint_share": rep.joint_share,
                   "usable_precursors": [(p.probe, p.lag) for p in rep.usable_precursors()]}
        return MaturedRecord(stable_hash([rep.period, rep.digest()], 16), rep.period, payload, prov, Namespace.MATURED_RESEARCH)

    def content_hash(self) -> str:
        return stable_hash({"ledger": [e.digest for e in self.ledger.entries()], "errors": self.errors, "n": len(self.history)}, 16)


def step(engine: MarketExpectationEngine, now, obs: MarketObs, **kw) -> MarketStepResult:
    """The ONE public entry the research loop calls: resolve, investigate if warranted, expect the next period."""
    return engine.step(now, obs, **kw)


def run_history(observations: Iterable[MarketObs], cfg: ExpectationConfig | None = None, seed: int = 0, **kw) -> tuple[MarketExpectationEngine, list[MarketStepResult]]:
    """Stream observations through a fresh engine one at a time (now = the observation's own date)."""
    eng = MarketExpectationEngine(cfg, seed)
    return eng, [eng.step(o.date, o, **kw) for o in observations]


def expectation_skill(engine: MarketExpectationEngine, quantity: str, min_pairs: int = 20) -> dict[str, float | None]:
    """Is the expectation better than 'same as last time'? MAE of the recorded expectations against the naive previous-value forecast
    over the resolved periods. skill = 1 - MAE_model / MAE_naive (> 0 means the model earns its complexity); None below min_pairs."""
    h = engine.history
    vals = series_of(h, quantity)
    errs, naive = [], []
    for i in range(1, len(h)):
        e = engine.ledger.get(h[i].date)
        if e is None or e.expected(quantity) is None or not (np.isfinite(vals[i]) and np.isfinite(vals[i - 1])):
            continue
        errs.append(abs(vals[i] - e.expected(quantity)))
        naive.append(abs(vals[i] - vals[i - 1]))
    if len(errs) < min_pairs:
        return {"n": float(len(errs)), "mae_model": None, "mae_naive": None, "skill": None}
    mm, mn = float(np.mean(errs)), float(np.mean(naive))
    return {"n": float(len(errs)), "mae_model": mm, "mae_naive": mn, "skill": 1.0 - mm / mn if mn > 0 else None}


def render_report(rep: InvestigationReport) -> str:
    """Plain-text report of one investigation: the gap, every explanation with its verdict, units that changed, the selection
    alternative, usable precursors and what stays unknown."""
    lines = [f"MARKET {rep.move.value}: {rep.target} expected {rep.expected:.4g} actual {rep.actual:.4g} (z {rep.z:+.1f})  -> {rep.conclusion.value}"]
    for p in rep.probes:
        lines.append(f"  {p.finding.value:14s} {p.name:30s} {p.reason}")
    lines.append(f"  {rep.combined.finding.value:14s} {rep.combined.name:30s} {rep.combined.reason}")
    if rep.joint_share is not None:
        lines.append(f"  supported quantities together explain {rep.joint_share:.0%}; unexplained {1 - rep.joint_share:.0%}")
    lines.append(f"  selection: {rep.selection.attribution.value} ({rep.selection.reason})")
    for u in rep.patterns + rep.sectors + rep.stocks:
        if u.status not in (UnitStatus.STABLE, UnitStatus.INSUFFICIENT):
            lines.append(f"  unit {u.unit}: {u.status.value}{' (newly active)' if u.newly_active else ''}")
    for pr in rep.usable_precursors():
        lines.append(f"  precursor {pr.probe} lag {pr.lag} r={pr.r:+.2f}")
    lines += [f"  note: {n}" for n in rep.notes]
    return "\n".join(lines)
