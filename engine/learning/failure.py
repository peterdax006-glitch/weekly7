"""Failure learning: the loss classifier and its detectors (contract C62 section 9; checklist D01-D10).

A loss is first-class training data, never just "a bad trade". Given one resolved trade and what was known about the
knowledge that drove it, this module asks EIGHT independent detectors (selection, timing, direction, risk, pattern,
context, regime, measurement - D02..D09) for evidence, combines that evidence per FailureCause with an explicit
for/against calculus, and answers with one of the section-9 causes - or UNKNOWN / INSUFFICIENT_EVIDENCE when the
evidence does not support a cause (D10). A cause is never forced:

  * INSUFFICIENT_EVIDENCE  too few detectors could run (inputs missing) - the question was not answerable;
  * UNKNOWN + CONFLICTED   detectors ran but two different causes are indistinguishable;
  * UNKNOWN                detectors ran and nothing reached the acceptance level (or the loss is inside normal noise).

Evidence points at a CAUSE (why) and, independently, at a SUBSYSTEM (who made the mistake). The subsystem vote is kept
separate so that a right-idea/wrong-entry trade teaches TIMING and not SELECTION (section 23; see separation.py for
the additive return decomposition that settles it numerically).

Time discipline (C56/C58): classify() takes `now` and fails closed (FirewallBreach) unless the trade resolved strictly
before it and every pattern-history point used is dated strictly before it. Identity (ticker) is never a field:
`rid` is an opaque id the caller derives with a salted hash. Status of this module: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from engine.learning.core import (FailureCause, FirewallBreach, Subsystem, Unknown, as_date, clip01, require_past,
                                  stable_hash)

FC = FailureCause
CAUSES_ALL = tuple(c for c in FailureCause)
CAUSES_EXPLAINING = tuple(c for c in FailureCause if c not in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE))
NL = chr(10)


# ==================================================================================================================
# small numeric helpers
# ==================================================================================================================
def _f(x) -> float | None:
    """float or None (NaN and None both mean 'not measured')."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear between. Degenerate span -> step."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def noisy_or(vals: Sequence[float]) -> float:
    """Independent-evidence combination: probability at least one cause of the signal is real."""
    prod = 1.0
    for v in vals:
        prod *= 1.0 - clip01(v)
    return 1.0 - prod


def _mean_t(x: Sequence[float], sd_floor: float = 1e-9) -> tuple[float, float]:
    """(mean, t-statistic against zero) of a sample; t=0 when it cannot be formed."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else 0.0), 0.0
    sd = float(a.std(ddof=1))
    if sd < sd_floor:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.mean() / (sd / math.sqrt(len(a))))


def ols_trend(y: Sequence[float]) -> tuple[float, float]:
    """(slope per step, t-statistic of the slope) of y on 0..n-1; (0, 0) if fewer than 4 points or a perfect fit."""
    a = np.asarray(y, dtype=float)
    n = len(a)
    if n < 4:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    slope = float((xc * (a - a.mean())).sum() / sxx)
    resid = a - a.mean() - slope * xc
    dof = n - 2
    s2 = float((resid ** 2).sum() / dof)
    if s2 < 1e-18:
        return slope, (0.0 if abs(slope) < 1e-15 else math.copysign(99.0, slope))
    return slope, float(slope / math.sqrt(s2 / sxx))


# ==================================================================================================================
# parameters (frozen: a changed threshold is a new object with a new hash, never an in-place edit)
# ==================================================================================================================
@dataclass(frozen=True)
class FailureParams:
    min_loss: float = 0.01               # a loss smaller than this (fraction of position) is never analysed
    noise_z: float = 1.0                 # ...nor one inside noise_z * expected horizon vol
    accept: float = 0.45                 # a cause needs at least this combined score to be named
    margin: float = 0.12                 # top cause must beat the runner-up by this much, else CONFLICTED
    against_weight: float = 0.8          # how hard contradicting evidence pulls a cause down
    min_coverage: float = 0.375          # fraction of detectors that must have run, else INSUFFICIENT_EVIDENCE
    min_secondary: float = 0.3           # runner-up causes reported (not named) above this score
    # selection
    sel_ratio_hi: float = 0.6            # realised move / expected move below this is a shortfall
    sel_ratio_lo: float = 0.1
    sel_rel_z: float = 1.0               # relative-underperformance mode (only when no expected move is available)
    # direction
    dir_min_move: float = 0.01           # a move against the side smaller than this is not a direction failure
    trend_min: float = 0.02              # prior side-adjusted trend needed to call a turn a reversal
    # timing / exit
    timing_regret: float = 0.03
    gap_share_lo: float = 0.3
    giveback_mfe: float = 0.03
    giveback_frac: float = 0.5
    # risk
    risk_z: float = 2.5                  # loss beyond this many expected-vol sigmas is a tail the model should not have allowed
    oversize_tol: float = 0.25
    stop_slip_tol: float = 0.25
    # pattern
    recent_k: int = 6
    min_hist: int = 10
    false_p_real: float = 0.25
    rev_t: float = 2.0
    trend_t: float = 2.0
    weak_frac: float = 0.5
    inactive_t: float = 1.0
    redund_corr: float = 0.8
    # regime / context / measurement
    regime_z_lo: float = 1.5
    regime_z_hi: float = 3.5
    min_ctx_dims: int = 2
    gap_absurd: float = 0.5
    recon_tol: float = 0.01
    stale_days: float = 3.0
    imputed_frac: float = 0.25

    def hash(self) -> str:
        return stable_hash(self)


# ==================================================================================================================
# inputs
# ==================================================================================================================
@dataclass(frozen=True)
class TradeRecord:
    """One resolved decision, side-agnostic raw returns plus what the system expected. All returns are fractions.

    signal_ret        raw stock return from the DECISION close to the horizon end (what selection/direction judged)
    entry_gap         raw return from decision close to the next open (the fill: decisions at a close fill next open)
    end_ret_from_fill raw stock return from the fill to the horizon end
    exit_ret          raw stock return from the fill to the actual exit (stop / target / horizon)
    pnl               NET side-adjusted return of the position, fill to exit, after costs
    mfe / mae         side-adjusted best / worst excursion from the fill (mae <= 0)
    exp_move          expected absolute move that selection promised (e.g. 0.07)
    exp_vol           expected horizon volatility of the position (sigma of pnl)
    dir_prob          model P(up); None when no direction model was used
    prior_ret         side-adjusted return over the lookback the entry relied on (trend into the entry)
    """
    rid: str
    decided_at: str
    resolved_at: str
    side: int
    pnl: float
    decided_by: Subsystem = Subsystem.SELECTION
    weight: float | None = None
    target_weight: float | None = None
    cost: float = 0.0005
    signal_ret: float | None = None
    entry_gap: float | None = None
    end_ret_from_fill: float | None = None
    exit_ret: float | None = None
    mfe: float | None = None
    mae: float | None = None
    best_entry_ret: float | None = None
    exp_move: float | None = None
    exp_ret: float | None = None
    exp_vol: float | None = None
    dir_prob: float | None = None
    prior_ret: float | None = None
    score: float | None = None
    rank_pct: float | None = None
    universe_ret: float | None = None
    stop: float | None = None
    stop_hit: bool = False
    stop_fill_ret: float | None = None
    pattern_ids: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    context: Mapping[str, float] = field(default_factory=dict)
    data_flags: Mapping[str, float] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)      # era / type breakdown labels (never a ticker)

    def validate(self) -> list[str]:
        errs = []
        if not self.rid:
            errs.append("rid missing")
        if self.side not in (-1, 1):
            errs.append(f"side must be +1/-1, got {self.side!r}")
        if _f(self.pnl) is None:
            errs.append("pnl missing or not finite")
        try:
            if as_date(self.resolved_at) <= as_date(self.decided_at):
                errs.append("resolved_at must be after decided_at")
        except Exception as e:                              # unparsable date strings are errors, not crashes
            errs.append(f"dates unparsable: {e}")
        for name in ("signal_ret", "entry_gap", "end_ret_from_fill", "exit_ret", "mfe", "mae", "exp_move", "exp_vol",
                     "dir_prob", "rank_pct", "weight", "target_weight"):
            v = getattr(self, name)
            if v is not None and _f(v) is None:
                errs.append(f"{name} is not finite")
        if self.dir_prob is not None and _f(self.dir_prob) is not None and not 0.0 <= self.dir_prob <= 1.0:
            errs.append("dir_prob outside [0,1]")
        if self.rank_pct is not None and _f(self.rank_pct) is not None and not 0.0 <= self.rank_pct <= 1.0:
            errs.append("rank_pct outside [0,1]")
        if self.exp_vol is not None and _f(self.exp_vol) is not None and self.exp_vol <= 0:
            errs.append("exp_vol must be positive")
        for k in tuple(self.context) + tuple(self.data_flags) + tuple(self.tags):
            if str(k).lower() in ("ticker", "symbol", "permno", "cusip", "isin", "name"):
                errs.append(f"identity key {k!r} in trade record")
        return errs

    def require_valid(self):
        errs = self.validate()
        if errs:
            raise ValueError(f"TradeRecord {self.rid!r} invalid: " + "; ".join(errs))

    # side-adjusted views
    @property
    def loss(self) -> float:
        return -float(self.pnl)

    @property
    def side_signal(self) -> float | None:
        v = _f(self.signal_ret)
        return None if v is None else self.side * v


@dataclass(frozen=True)
class PatternHistory:
    """A pattern's recent track record, oldest to newest, as known at `as_of` (strictly before classify's now).
    effects are side-adjusted mean per-trade effects per period; triggers count firings per period."""
    pattern_id: str
    effects: tuple[float, ...]
    dates: tuple[str, ...] = ()
    triggers: tuple[int, ...] = ()
    p_real: float | None = None
    t_disc: float | None = None
    t_conf: float | None = None

    def validate(self) -> list[str]:
        errs = []
        if not self.pattern_id:
            errs.append("pattern_id missing")
        if self.dates and len(self.dates) != len(self.effects):
            errs.append("dates/effects length mismatch")
        if self.triggers and len(self.triggers) != len(self.effects):
            errs.append("triggers/effects length mismatch")
        if any(_f(e) is None for e in self.effects):
            errs.append("non-finite effect")
        if self.p_real is not None and not 0.0 <= self.p_real <= 1.0:
            errs.append("p_real outside [0,1]")
        return errs


@dataclass(frozen=True)
class ContextReference:
    """The market-context baseline the decision was made against (mean/sd per m_ feature, from data before decided_at)."""
    mean: Mapping[str, float]
    sd: Mapping[str, float]
    regime_then: str = ""
    regime_now: str = ""


@dataclass(frozen=True)
class FailureEnv:
    """Everything besides the trade itself that detectors may consult. Every field is optional: a detector whose
    inputs are absent reports that it could not run, which lowers coverage - it never guesses."""
    patterns: Mapping[str, PatternHistory] = field(default_factory=dict)
    context_ref: ContextReference | None = None
    knowledge: Mapping[str, Any] = field(default_factory=dict)          # id -> KnowledgeLike (contexts/anti_contexts)
    pattern_corr: Mapping[tuple[str, str], float] = field(default_factory=dict)     # trigger correlation of a pair
    pair_effects: Mapping[tuple[str, str], float] = field(default_factory=dict)     # mean effect when BOTH fire
    universe_sd: float | None = None


@dataclass(frozen=True)
class Evidence:
    """One finding of one detector. cause=None means it informs only the subsystem vote."""
    detector: str
    cause: FailureCause | None
    strength: float
    supports: bool = True
    subsystem: Subsystem | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "strength", clip01(self.strength))


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    ran: bool
    missing: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()


def _missing(t: TradeRecord, *names: str) -> tuple[str, ...]:
    return tuple(n for n in names if _f(getattr(t, n)) is None)


# ==================================================================================================================
# context conditions (contract section 8): one place that says whether a condition holds
# ==================================================================================================================
def condition_holds(cond: Any, value: Any) -> bool | None:
    """True/False, or None when the value is unknown (an unknown condition is never treated as satisfied)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if cond is None:
        return True
    if isinstance(cond, Mapping):
        lo, hi = cond.get("lo", cond.get("min")), cond.get("hi", cond.get("max"))
        if "eq" in cond:
            return value == cond["eq"]
        if "in" in cond:
            return value in cond["in"]
        return (lo is None or value >= lo) and (hi is None or value <= hi)
    if isinstance(cond, tuple) and len(cond) == 2 and all(c is None or isinstance(c, (int, float)) for c in cond):
        lo, hi = cond
        return (lo is None or value >= lo) and (hi is None or value <= hi)
    if isinstance(cond, (set, frozenset, list, tuple)):
        return value in cond
    return value == cond


@dataclass(frozen=True)
class ContextVerdict:
    holds_all: bool
    failed: tuple[str, ...]              # contexts dimensions whose condition was violated
    anti_hit: tuple[str, ...]            # anti-context dimensions that were active (knowledge said: do not use here)
    unknown: tuple[str, ...]             # dimensions the trade did not record: the missing-condition candidates
    n_checked: int


def evaluate_contexts(k: Any, ctx: Mapping[str, float]) -> ContextVerdict:
    failed, anti, unknown, n = [], [], [], 0
    for dim, cond in dict(getattr(k, "contexts", {}) or {}).items():
        n += 1
        r = condition_holds(cond, ctx.get(dim))
        if r is None:
            unknown.append(dim)
        elif not r:
            failed.append(dim)
    for dim, cond in dict(getattr(k, "anti_contexts", {}) or {}).items():
        n += 1
        r = condition_holds(cond, ctx.get(dim))
        if r is None:
            unknown.append(dim)
        elif r:
            anti.append(dim)
    return ContextVerdict(not failed and not anti and not unknown, tuple(failed), tuple(anti), tuple(unknown), n)


# ==================================================================================================================
# D02 selection failure detector
# ==================================================================================================================
def detect_selection(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    """Selection promised a stock that MOVES ~exp_move. If it did not move, selection erred - whichever way it went."""
    ev: list[Evidence] = []
    if _f(t.exp_move) is not None and _f(t.signal_ret) is not None:
        ratio = abs(t.signal_ret) / max(float(t.exp_move), 1e-9)
        if ratio < p.sel_ratio_hi:
            s = ramp(p.sel_ratio_hi - ratio, 0.0, p.sel_ratio_hi - p.sel_ratio_lo)
            if _f(t.rank_pct) is not None:                  # a top-ranked pick that stalled is a worse ranking error
                s *= 0.7 + 0.3 * float(t.rank_pct)
            ev.append(Evidence("selection", FC.SELECTION_ERROR, s, True, Subsystem.SELECTION,
                               {"move_ratio": ratio, "exp_move": t.exp_move, "realised_move": abs(t.signal_ret)},
                               "the stock moved far less than selection promised"))
        else:
            ev.append(Evidence("selection", FC.SELECTION_ERROR, ramp(ratio, 0.9, 1.5), False, Subsystem.SELECTION,
                               {"move_ratio": ratio}, "the stock moved at least as much as promised"))
        return DetectorResult("selection", True, (), tuple(ev))
    sd = _f(env.universe_sd)
    if _f(t.signal_ret) is not None and _f(t.universe_ret) is not None and sd and sd > 0:
        z = t.side * (t.signal_ret - t.universe_ret) / sd
        if z <= -p.sel_rel_z:
            ev.append(Evidence("selection", FC.SELECTION_ERROR, ramp(-z, p.sel_rel_z, p.sel_rel_z + 2.0), True,
                               Subsystem.SELECTION, {"rel_z": z}, "the pick lagged the universe"))
        elif z >= 0.5:
            ev.append(Evidence("selection", FC.SELECTION_ERROR, ramp(z, 0.5, 2.0), False, Subsystem.SELECTION,
                               {"rel_z": z}, "the pick beat the universe"))
        return DetectorResult("selection", True, (), tuple(ev))
    return DetectorResult("selection", False, ("exp_move|universe_ret", "signal_ret"))


# ==================================================================================================================
# D03 timing failure detector (entry timing and exit timing; exit problems carry subsystem EXIT)
# ==================================================================================================================
def detect_timing(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    miss = _missing(t, "signal_ret")
    if miss:
        return DetectorResult("timing", False, miss)
    ev: list[Evidence] = []
    ss = t.side_signal
    if ss <= p.dir_min_move:
        # the idea was not right over the horizon, so this is not a timing problem
        ev.append(Evidence("timing", FC.TIMING_ERROR, 0.8, False, Subsystem.TIMING,
                           {"side_signal": ss}, "the side was wrong over the horizon: nothing to time"))
        return DetectorResult("timing", True, (), tuple(ev))
    loss = max(t.loss, 1e-9)
    base = 0.35 * ramp(ss, p.dir_min_move, 0.05)
    ev.append(Evidence("timing", FC.TIMING_ERROR, base, True, Subsystem.TIMING,
                       {"side_signal": ss}, "right over the horizon yet a loss was booked"))
    gap = _f(t.entry_gap)
    if gap is not None:
        gap_cost = -t.side * gap                            # positive when the open was adverse to the position
        if gap_cost > 0:
            share = gap_cost / loss
            ev.append(Evidence("timing", FC.TIMING_ERROR, ramp(share, p.gap_share_lo, 1.0), True, Subsystem.TIMING,
                               {"gap_cost": gap_cost, "loss_share": share}, "the overnight gap ate the trade"))
    reg = _f(t.best_entry_ret)
    if reg is not None and _f(t.end_ret_from_fill) is not None:
        regret = reg - t.side * t.end_ret_from_fill
        if regret >= p.timing_regret:
            ev.append(Evidence("timing", FC.TIMING_ERROR, ramp(regret, p.timing_regret, 3 * p.timing_regret), True,
                               Subsystem.TIMING, {"entry_regret": regret}, "a much better entry existed in the window"))
    if t.stop_hit and _f(t.end_ret_from_fill) is not None and t.side * t.end_ret_from_fill > p.dir_min_move:
        ev.append(Evidence("timing", FC.TIMING_ERROR, ramp(t.side * t.end_ret_from_fill, p.dir_min_move, 0.06), True,
                           Subsystem.EXIT, {"recovered_after_stop": t.side * t.end_ret_from_fill},
                           "stopped out, then the position recovered (shake-out)"))
    mfe = _f(t.mfe)
    if mfe is not None and mfe >= p.giveback_mfe:
        given = (mfe - t.pnl) / mfe
        if given >= 1.0 + p.giveback_frac:
            ev.append(Evidence("timing", FC.TIMING_ERROR, ramp(given, 1.0 + p.giveback_frac, 3.0), True,
                               Subsystem.EXIT, {"mfe": mfe, "given_back_multiple": given},
                               "was well ahead, then round-tripped into a loss (exit failure)"))
    return DetectorResult("timing", True, (), tuple(ev))


# ==================================================================================================================
# D04 direction failure detector
# ==================================================================================================================
def surprise_bits(dir_prob: float | None, went_up: bool) -> float | None:
    """-log2 of the probability the model gave to what actually happened. A coin flip costs exactly 1 bit."""
    if dir_prob is None or _f(dir_prob) is None:
        return None
    pr = dir_prob if went_up else 1.0 - dir_prob
    return float(-math.log2(min(1 - 1e-9, max(1e-9, pr))))


def detect_direction(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    miss = _missing(t, "signal_ret")
    if miss:
        return DetectorResult("direction", False, miss)
    ss = t.side_signal
    ev: list[Evidence] = []
    if ss > -p.dir_min_move:
        ev.append(Evidence("direction", None, ramp(ss, 0.0, 0.05), False, Subsystem.DIRECTION,
                           {"side_signal": ss}, "the stock moved the way the side said (or barely moved)"))
        return DetectorResult("direction", True, (), tuple(ev))
    facts: dict[str, Any] = {"side_signal": ss}
    sb = surprise_bits(_f(t.dir_prob), t.signal_ret > 0)
    if sb is not None:
        facts["surprise_bits"] = sb
    ev.append(Evidence("direction", None, ramp(-ss, p.dir_min_move, 4 * p.dir_min_move), True, Subsystem.DIRECTION,
                       facts, "the stock moved against the side"))
    pr = _f(t.prior_ret)
    if pr is not None and pr >= p.trend_min:
        ev.append(Evidence("direction", FC.REVERSAL, ramp(-ss, p.dir_min_move, 4 * p.dir_min_move) * ramp(pr, p.trend_min, 4 * p.trend_min),
                           True, Subsystem.DIRECTION, {"prior_ret": pr, "side_signal": ss},
                           "entered with the prior trend, which then turned"))
    return DetectorResult("direction", True, (), tuple(ev))


# ==================================================================================================================
# D05 risk failure detector
# ==================================================================================================================
def detect_risk(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    have_vol = _f(t.exp_vol) is not None
    have_stop = _f(t.stop) is not None and t.stop_hit and _f(t.stop_fill_ret) is not None
    have_w = _f(t.weight) is not None and _f(t.target_weight) is not None and t.target_weight > 0
    if not (have_vol or have_stop or have_w):
        return DetectorResult("risk", False, ("exp_vol|stop|weight",))
    ev: list[Evidence] = []
    if have_vol:
        z = t.loss / float(t.exp_vol)
        if z >= p.risk_z:
            ev.append(Evidence("risk", FC.RISK_ERROR, ramp(z, p.risk_z, p.risk_z + 2.5), True, Subsystem.RISK,
                               {"loss_sigma": z}, "the loss was a tail event the risk model should have bounded"))
        elif z <= 1.5:
            ev.append(Evidence("risk", FC.RISK_ERROR, ramp(1.5 - z, 0.0, 1.0) * 0.8, False, Subsystem.RISK,
                               {"loss_sigma": z}, "the loss was within expected volatility"))
    if have_stop:
        slip = (-float(t.stop)) - float(t.stop_fill_ret)     # how much worse than the stop level the fill was
        if slip > p.stop_slip_tol * float(t.stop):
            ev.append(Evidence("risk", FC.RISK_ERROR, ramp(slip / float(t.stop), p.stop_slip_tol, 1.5), True,
                               Subsystem.RISK, {"stop": t.stop, "fill": t.stop_fill_ret, "slippage": slip},
                               "a gap jumped the stop: the stop did not cap the loss"))
    if have_w:
        over = t.weight / t.target_weight - 1.0
        if over > p.oversize_tol:
            ev.append(Evidence("risk", FC.RISK_ERROR, ramp(over, p.oversize_tol, 1.5), True, Subsystem.RISK,
                               {"oversize": over}, "position was larger than the plan allowed"))
        elif over <= 0.05 and not ev:
            ev.append(Evidence("risk", FC.RISK_ERROR, 0.3, False, Subsystem.RISK, {"oversize": over},
                               "position size was within plan"))
    return DetectorResult("risk", True, (), tuple(ev))


# ==================================================================================================================
# D06 pattern failure detector (false / weakening / reversal / inactivity / redundancy / interaction)
# ==================================================================================================================
def _dormant_recoveries(base: Sequence[float], k: int, base_mean: float) -> int:
    """How many times in the earlier history a window of length k went flat (<25% of the long-run effect) and the
    next window came back (>75%). Evidence that flatness is a habit of this pattern, not the end of it."""
    n, episodes, i = len(base), 0, 0
    while i + 2 * k <= n:
        a, b = np.mean(base[i:i + k]), np.mean(base[i + k:i + 2 * k])
        if base_mean > 0 and a < 0.25 * base_mean and b > 0.75 * base_mean:
            episodes += 1
            i += 2 * k
        else:
            i += 1
    return episodes


def pattern_verdicts(h: PatternHistory, p: FailureParams, now=None) -> tuple[list[Evidence], str]:
    """Evidence about ONE pattern from its own history. Returns (evidence, skip_reason)."""
    eff = [float(e) for e in h.effects]
    if now is not None:
        for d in h.dates:
            require_past(d, now, f"pattern {h.pattern_id} history")
    k = p.recent_k
    if len(eff) < max(p.min_hist, k + 4):
        return [], f"history too short ({len(eff)})"
    base, rec = eff[:-k], eff[-k:]
    base_mean, base_t = _mean_t(base)
    rec_mean, rec_t = _mean_t(rec, sd_floor=1e-9)
    base_sd = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
    if abs(rec_t) < 1e-12 and base_sd > 0:                 # flat recent sample: judge it against the long-run noise
        rec_t = rec_mean / (base_sd / math.sqrt(k))
    slope, trend_t = ols_trend(eff[-max(2 * k, p.min_hist):])
    out: list[Evidence] = []
    pid = h.pattern_id
    ok_before = base_mean > 0 and base_t >= 1.0
    eps = _dormant_recoveries(base, k, base_mean) if ok_before else 0
    if ok_before and rec_mean < 0 and rec_t <= -p.rev_t:
        out.append(Evidence("pattern", FC.REVERSAL, ramp(-rec_t, p.rev_t, p.rev_t + 2.0), True, Subsystem.SELECTION,
                            {"pattern": pid, "base_mean": base_mean, "recent_mean": rec_mean, "recent_t": rec_t},
                            "the effect flipped sign and the flip is significant"))
    if ok_before and 0 < rec_mean < base_mean * p.weak_frac and trend_t <= -p.trend_t:
        s = 0.5 * ramp(1 - rec_mean / base_mean, p.weak_frac, 1.0) + 0.5 * ramp(-trend_t, p.trend_t, p.trend_t + 2.0)
        if eps:                                            # it has gone flat and come back before: less likely decay
            s *= 0.5
        out.append(Evidence("pattern", FC.WEAKENING_EFFECT, s, True, Subsystem.SELECTION,
                            {"pattern": pid, "base_mean": base_mean, "recent_mean": rec_mean, "trend_t": trend_t},
                            "the effect is decaying with a significant downward trend"))
    if ok_before and abs(rec_t) < p.inactive_t and rec_mean < base_mean * p.weak_frac and (trend_t > -p.trend_t or eps):
        trig = list(h.triggers[-k:]) if h.triggers else []
        quiet = bool(trig) and sum(trig) <= max(1, 0.25 * np.mean(h.triggers[:-k])) * k if h.triggers else False
        s = 0.25 + 0.2 * min(eps, 3) + (0.15 if quiet else 0.0)
        out.append(Evidence("pattern", FC.TEMPORARY_INACTIVITY, s, True, Subsystem.SELECTION,
                            {"pattern": pid, "prior_recoveries": eps, "recent_t": rec_t, "quiet_triggers": quiet},
                            "flat recently, but not decaying; it has gone flat and recovered before"))
    if h.p_real is not None:
        if h.p_real < p.false_p_real:
            s = ramp(p.false_p_real - h.p_real, 0.0, p.false_p_real)
            if h.t_conf is not None and h.t_disc is not None and h.t_disc >= 3.0 and h.t_conf < 1.0:
                s = min(1.0, s + 0.25)                         # found strongly, never confirmed out of sample
            out.append(Evidence("pattern", FC.FALSE_PATTERN, max(s, 0.3), True, Subsystem.SELECTION,
                                {"pattern": pid, "p_real": h.p_real, "t_disc": h.t_disc, "t_conf": h.t_conf},
                                "low probability the pattern is real"))
        elif h.p_real >= 0.8 and (h.t_conf is None or h.t_conf >= 2.0):
            out.append(Evidence("pattern", FC.FALSE_PATTERN, 0.7, False, Subsystem.SELECTION,
                                {"pattern": pid, "p_real": h.p_real}, "the pattern was well confirmed"))
    return out, ""


def detect_pattern(t: TradeRecord, env: FailureEnv, p: FailureParams, now=None) -> DetectorResult:
    if not t.pattern_ids:
        return DetectorResult("pattern", False, ("pattern_ids",))
    ev: list[Evidence] = []
    ran, missing = False, []
    for pid in t.pattern_ids:
        h = env.patterns.get(pid)
        if h is None:
            missing.append(f"history:{pid}")
            continue
        if h.validate():
            missing.append(f"history:{pid}:invalid")
            continue
        e, why = pattern_verdicts(h, p, now)
        if why:
            missing.append(f"history:{pid}:{why}")
            continue
        ran = True
        ev.extend(e)
    ids = sorted(t.pattern_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            key = (ids[i], ids[j])
            c = env.pattern_corr.get(key, env.pattern_corr.get((key[1], key[0])))
            if c is not None and abs(c) >= p.redund_corr:
                ran = True
                ev.append(Evidence("pattern", FC.REDUNDANCY, ramp(abs(c), p.redund_corr, 1.0) * 0.9, True,
                                   Subsystem.SELECTION, {"pair": key, "corr": float(c)},
                                   "two near-identical patterns counted the same signal twice"))
            je = env.pair_effects.get(key, env.pair_effects.get((key[1], key[0])))
            if je is not None:
                ran = True
                ha, hb = env.patterns.get(key[0]), env.patterns.get(key[1])
                if ha is not None and hb is not None and ha.effects and hb.effects:
                    single = min(float(np.mean(ha.effects)), float(np.mean(hb.effects)))
                    if single > 0 and je < 0.25 * single:
                        ev.append(Evidence("pattern", FC.INTERACTION_FAILURE, ramp(single - je, 0.25 * single, 1.5 * single),
                                           True, Subsystem.SELECTION, {"pair": key, "joint": float(je), "single_min": single},
                                           "each pattern works alone; together they do not"))
    return DetectorResult("pattern", ran, tuple(missing) if not ran else (), tuple(ev))


# ==================================================================================================================
# D07 context failure detector
# ==================================================================================================================
def detect_context(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    ks = [(i, env.knowledge[i]) for i in t.knowledge_ids if i in env.knowledge]
    ks = [(i, k) for i, k in ks if getattr(k, "contexts", None) or getattr(k, "anti_contexts", None)]
    if not ks:
        return DetectorResult("context", False, ("knowledge.contexts",))
    ev: list[Evidence] = []
    all_unknown = True
    for kid, k in ks:
        v = evaluate_contexts(k, t.context)
        if v.n_checked and len(v.unknown) < v.n_checked:
            all_unknown = False
        if v.anti_hit:
            ev.append(Evidence("context", FC.WRONG_CONTEXT, 0.9, True, Subsystem.SELECTION,
                               {"knowledge": kid, "anti_hit": v.anti_hit}, "an anti-context of the knowledge was active"))
        elif v.failed:
            frac = len(v.failed) / max(1, v.n_checked)
            ev.append(Evidence("context", FC.WRONG_CONTEXT, 0.55 + 0.4 * frac, True, Subsystem.SELECTION,
                               {"knowledge": kid, "failed": v.failed}, "the trade sat outside the knowledge's contexts"))
        elif v.holds_all:
            ev.append(Evidence("context", FC.WRONG_CONTEXT, 0.7, False, Subsystem.SELECTION,
                               {"knowledge": kid}, "every context condition held"))
        elif v.unknown:
            ev.append(Evidence("context", FC.WRONG_CONTEXT, 0.15, True, Subsystem.SELECTION,
                               {"knowledge": kid, "unknown_dims": v.unknown},
                               "context dimensions were not recorded, so applicability is unknowable"))
    if all_unknown:
        return DetectorResult("context", False, ("context values",))
    return DetectorResult("context", True, (), tuple(ev))


# ==================================================================================================================
# D08 regime failure detector
# ==================================================================================================================
def regime_distance(ctx: Mapping[str, float], ref: ContextReference) -> tuple[float, int, dict[str, float]]:
    """RMS z-distance of the trade's market context from its reference over shared dimensions."""
    zs = {}
    for k, v in ctx.items():
        if k in ref.mean and k in ref.sd and _f(v) is not None:
            sd = float(ref.sd[k])
            if sd > 1e-12:
                zs[k] = (float(v) - float(ref.mean[k])) / sd
    if not zs:
        return 0.0, 0, {}
    return float(math.sqrt(np.mean(np.square(list(zs.values()))))), len(zs), zs


def detect_regime(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    ref = env.context_ref
    if ref is None:
        return DetectorResult("regime", False, ("context_ref",))
    d, n, zs = regime_distance(t.context, ref)
    if n < p.min_ctx_dims:
        return DetectorResult("regime", False, (f"context dims>={p.min_ctx_dims}",))
    ev: list[Evidence] = []
    label_shift = bool(ref.regime_then and ref.regime_now and ref.regime_then != ref.regime_now)
    s = ramp(d, p.regime_z_lo, p.regime_z_hi)
    if label_shift:
        s = min(1.0, s + 0.3)
    if s > 0:
        worst = sorted(zs, key=lambda k: -abs(zs[k]))[:3]
        ev.append(Evidence("regime", FC.REGIME_CHANGE, s, True, None,
                           {"rms_z": d, "worst_dims": tuple(worst), "label_shift": label_shift},
                           "market context is far from anything the decision was calibrated on"))
    elif d < 1.0 and not label_shift:
        ev.append(Evidence("regime", FC.REGIME_CHANGE, ramp(1.0 - d, 0.0, 1.0) * 0.7, False, None,
                           {"rms_z": d}, "market context was ordinary"))
    return DetectorResult("regime", True, (), tuple(ev))


# ==================================================================================================================
# D09 measurement failure detector
# ==================================================================================================================
def detect_measurement(t: TradeRecord, env: FailureEnv, p: FailureParams) -> DetectorResult:
    ev: list[Evidence] = []
    checked = 0
    fl = {k: _f(v) for k, v in t.data_flags.items()}
    if fl:
        checked += 1
        stale = fl.get("stale_price_days")
        if stale is not None and stale >= p.stale_days:
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, ramp(stale, p.stale_days, 3 * p.stale_days), True, None,
                               {"stale_price_days": stale}, "the price used was stale"))
        if fl.get("split_suspect"):
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, 0.85, True, None, {}, "a split/adjustment artefact is suspected"))
        if fl.get("fill_outside_bar"):
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, 0.8, True, None, {}, "the fill price lies outside that bar's range"))
        imp = fl.get("imputed_frac")
        if imp is not None and imp >= p.imputed_frac:
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, ramp(imp, p.imputed_frac, 0.8), True, None,
                               {"imputed_frac": imp}, "a large share of the inputs were imputed"))
        if fl.get("zero_volume"):
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, 0.5, True, None, {}, "the bar had zero volume"))
    gap = _f(t.entry_gap)
    if gap is not None:
        checked += 1
        if abs(gap) >= p.gap_absurd:
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, ramp(abs(gap), p.gap_absurd, 1.5 * p.gap_absurd), True, None,
                               {"entry_gap": gap}, "an overnight gap this large is more likely a data artefact"))
    if gap is not None and _f(t.signal_ret) is not None and _f(t.end_ret_from_fill) is not None:
        checked += 1
        implied = (1 + gap) * (1 + t.end_ret_from_fill) - 1
        err = abs(implied - t.signal_ret)
        if err > p.recon_tol:
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, ramp(err, p.recon_tol, 10 * p.recon_tol), True, None,
                               {"reconcile_error": err}, "gap x fill-to-end return does not reproduce the decision-to-end return"))
    if _f(t.exit_ret) is not None:
        checked += 1
        recon = t.side * t.exit_ret - float(t.cost)
        err = abs(recon - t.pnl)
        if err > p.recon_tol and not t.stop_hit:
            ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, ramp(err, p.recon_tol, 10 * p.recon_tol), True, None,
                               {"pnl_reconcile_error": err}, "pnl does not reconcile with the exit return and cost"))
    if not checked:
        return DetectorResult("measurement", False, ("data_flags|entry_gap|exit_ret",))
    if not ev:
        ev.append(Evidence("measurement", FC.MEASUREMENT_ERROR, 0.6, False, None, {"checks": checked},
                           "every data-integrity check passed"))
    return DetectorResult("measurement", True, (), tuple(ev))


DETECTORS: dict[str, Callable[..., DetectorResult]] = {
    "selection": detect_selection, "timing": detect_timing, "direction": detect_direction, "risk": detect_risk,
    "pattern": detect_pattern, "context": detect_context, "regime": detect_regime, "measurement": detect_measurement}
CHECKLIST_ID = {"selection": "D02", "timing": "D03", "direction": "D04", "risk": "D05", "pattern": "D06",
                "context": "D07", "regime": "D08", "measurement": "D09"}


# ==================================================================================================================
# D01 loss classifier + D10 unknown state
# ==================================================================================================================
@dataclass(frozen=True)
class CauseScore:
    cause: FailureCause
    score: float
    support: tuple[Evidence, ...] = ()
    against: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Classification:
    rid: str
    cause: FailureCause
    score: float                              # combined score of the named cause (0 for UNKNOWN / INSUFFICIENT)
    confidence: float                         # confidence in the explanation (section 24), 0 when no cause is named
    unknown_state: Unknown | None             # why no cause was named
    meaningful: bool
    coverage: float                           # share of detectors that could run
    ran: tuple[str, ...]
    missing: Mapping[str, tuple[str, ...]]
    secondary: tuple[tuple[FailureCause, float], ...]
    scores: tuple[CauseScore, ...]
    subsystem_votes: Mapping[str, float]      # Subsystem value -> net vote (support minus against)
    surprise_bits: float | None
    note: str = ""
    params_hash: str = ""

    @property
    def named(self) -> bool:
        return self.cause not in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE)

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        for s in self.scores:
            if s.cause == self.cause:
                return s.support
        return ()

    def top_subsystem(self) -> Subsystem | None:
        best = max(self.subsystem_votes.items(), key=lambda kv: kv[1], default=None)
        return Subsystem(best[0]) if best and best[1] > 0 else None

    def to_dict(self) -> dict:
        return {"rid": self.rid, "cause": self.cause.value, "score": round(self.score, 4), "confidence": round(self.confidence, 4),
                "unknown_state": self.unknown_state.value if self.unknown_state else None, "meaningful": self.meaningful,
                "coverage": round(self.coverage, 4), "ran": list(self.ran), "missing": {k: list(v) for k, v in self.missing.items()},
                "secondary": [(c.value, round(s, 4)) for c, s in self.secondary],
                "subsystem_votes": {k: round(v, 4) for k, v in self.subsystem_votes.items()},
                "surprise_bits": self.surprise_bits, "note": self.note, "params_hash": self.params_hash}


def combine_evidence(evidence: Sequence[Evidence], against_weight: float = 0.8) -> dict[FailureCause, CauseScore]:
    """Per cause: noisy-OR of the supporting strengths, pulled down by the strongest contradicting evidence."""
    sup: dict[FailureCause, list[Evidence]] = {}
    con: dict[FailureCause, list[Evidence]] = {}
    for e in evidence:
        if e.cause is None:
            continue
        (sup if e.supports else con).setdefault(e.cause, []).append(e)
    out = {}
    for c in set(sup) | set(con):
        s = noisy_or([e.strength for e in sup.get(c, [])])
        a = max((e.strength for e in con.get(c, [])), default=0.0)
        out[c] = CauseScore(c, clip01(s * (1.0 - against_weight * a)), tuple(sup.get(c, [])), tuple(con.get(c, [])))
    return out


def subsystem_votes(evidence: Sequence[Evidence]) -> dict[str, float]:
    """Who made the mistake. Supporting evidence votes for a subsystem, contradicting evidence votes against it
    (a subsystem that demonstrably did its job cannot be blamed). Net votes below zero are kept: they are the
    subsystems that should NOT be taught by this loss."""
    votes: dict[str, list[float]] = {}
    for e in evidence:
        if e.subsystem is None:
            continue
        votes.setdefault(e.subsystem.value, []).append(e.strength if e.supports else -e.strength)
    out = {}
    for k, v in votes.items():
        pos = noisy_or([x for x in v if x > 0])
        neg = noisy_or([-x for x in v if x < 0])
        out[k] = float(pos - neg)
    return out


class LossClassifier:
    """D01. Deterministic and stateless apart from its frozen parameters."""

    def __init__(self, params: FailureParams | None = None, detectors: Mapping[str, Callable] | None = None):
        self.params = params or FailureParams()
        self.detectors = dict(detectors or DETECTORS)
        if not self.detectors:
            raise ValueError("a classifier without detectors can only answer INSUFFICIENT_EVIDENCE")

    def is_meaningful(self, t: TradeRecord) -> tuple[bool, str]:
        p = self.params
        if t.pnl >= 0:
            return False, "not a loss"
        if t.loss < p.min_loss:
            return False, f"loss {t.loss:.4f} below the {p.min_loss} floor"
        if _f(t.exp_vol) is not None and t.loss < p.noise_z * float(t.exp_vol):
            return False, "loss inside expected volatility: variance, not a failure"
        return True, ""

    def run_detectors(self, t: TradeRecord, env: FailureEnv, now) -> list[DetectorResult]:
        res = []
        for name, fn in self.detectors.items():
            if name == "pattern":
                res.append(fn(t, env, self.params, now))
            else:
                res.append(fn(t, env, self.params))
        return res

    def classify(self, t: TradeRecord, env: FailureEnv | None, now) -> Classification:
        t.require_valid()
        require_past(t.resolved_at, now, f"trade {t.rid} resolution")
        env = env or FailureEnv()
        p = self.params
        ph = p.hash()
        meaningful, why = self.is_meaningful(t)
        if not meaningful:
            return Classification(t.rid, FC.UNKNOWN, 0.0, 0.0, Unknown.UNKNOWN, False, 0.0, (), {}, (), (), {}, None, why, ph)
        results = self.run_detectors(t, env, now)
        ran = tuple(r.detector for r in results if r.ran)
        missing = {r.detector: r.missing for r in results if not r.ran}
        coverage = len(ran) / len(results)
        evidence = [e for r in results if r.ran for e in r.evidence]
        votes = subsystem_votes(evidence)
        sb = None
        for e in evidence:
            if "surprise_bits" in e.facts:
                sb = float(e.facts["surprise_bits"])
        if coverage < p.min_coverage:
            return Classification(t.rid, FC.INSUFFICIENT_EVIDENCE, 0.0, 0.0, Unknown.INSUFFICIENT_DATA, True, coverage, ran,
                                  missing, (), (), votes, sb, f"only {len(ran)}/{len(results)} detectors could run", ph)
        scores = combine_evidence(evidence, p.against_weight)
        ranked = sorted(scores.values(), key=lambda s: (-s.score, s.cause.value))
        accepted = [s for s in ranked if s.score >= p.accept]
        secondary = tuple((s.cause, s.score) for s in ranked if p.min_secondary <= s.score)
        if not accepted:
            return Classification(t.rid, FC.UNKNOWN, 0.0, 0.0, Unknown.UNKNOWN, True, coverage, ran, missing, secondary,
                                  tuple(ranked), votes, sb, "detectors ran; no cause reached the acceptance level", ph)
        top = accepted[0]
        if len(accepted) > 1 and top.score - accepted[1].score < p.margin:
            return Classification(t.rid, FC.UNKNOWN, 0.0, 0.0, Unknown.CONFLICTED, True, coverage, ran, missing, secondary,
                                  tuple(ranked), votes, sb,
                                  f"{top.cause.value} and {accepted[1].cause.value} are indistinguishable", ph)
        runner = accepted[1].score if len(accepted) > 1 else 0.0
        conf = clip01(top.score * (0.5 + 0.5 * coverage) * (0.6 + 0.4 * min(1.0, (top.score - runner) / 0.5)))
        return Classification(t.rid, top.cause, top.score, conf, None, True, coverage, ran, missing,
                              tuple((s.cause, s.score) for s in ranked[1:] if s.score >= p.min_secondary),
                              tuple(ranked), votes, sb, "", ph)

    def classify_many(self, trades: Sequence[TradeRecord], env: FailureEnv | None, now) -> list[Classification]:
        return [self.classify(t, env, now) for t in trades]


# ==================================================================================================================
# ledger: what the classifier says across many losses, with honest UNKNOWN accounting
# ==================================================================================================================
class FailureLedger:
    """Append-only tally of classifications. The point is the distribution of causes, the UNKNOWN share (a classifier
    that never says UNKNOWN is forcing), and the per-tag breakdown (era / winner type) the contract asks for."""

    def __init__(self):
        self._rows: list[tuple[Classification, Mapping[str, str]]] = []
        self._ids: set[str] = set()

    def add(self, c: Classification, tags: Mapping[str, str] | None = None) -> None:
        if c.rid in self._ids:
            raise ValueError(f"duplicate classification for {c.rid}: history is append-only")
        self._ids.add(c.rid)
        self._rows.append((c, dict(tags or {})))

    def __len__(self):
        return len(self._rows)

    def cause_counts(self, tag: str | None = None, value: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for c, tg in self._rows:
            if not c.meaningful or (tag is not None and tg.get(tag) != value):
                continue
            out[c.cause.value] = out.get(c.cause.value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def unknown_rate(self) -> float:
        m = [c for c, _ in self._rows if c.meaningful]
        return float(sum(1 for c in m if not c.named) / len(m)) if m else float("nan")

    def by_tag(self, tag: str) -> dict[str, dict[str, int]]:
        vals = sorted({tg.get(tag, "") for _, tg in self._rows})
        return {v: self.cause_counts(tag, v) for v in vals}

    def coverage_gaps(self) -> dict[str, int]:
        """Which inputs were most often missing: what to collect to answer more losses."""
        out: dict[str, int] = {}
        for c, _ in self._rows:
            for det, miss in c.missing.items():
                for m in miss:
                    out[f"{det}:{m}"] = out.get(f"{det}:{m}", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def subsystem_totals(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c, _ in self._rows:
            if c.meaningful:
                for k, v in c.subsystem_votes.items():
                    out[k] = out.get(k, 0.0) + v
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def report(self) -> str:
        lines = [f"failure ledger: {len(self)} losses, unknown rate {self.unknown_rate():.2f}",
                 "IMPLEMENTED - NOT VALIDATED"]
        for k, n in self.cause_counts().items():
            lines.append(f"  {k:<22}{n:>5}")
        return NL.join(lines)


def confusion(pred: Sequence[FailureCause], truth: Sequence[FailureCause]) -> dict[str, Any]:
    """Confusion of classifier output against KNOWN (planted) causes. Reports the number that matters most for a
    classifier that must not force: accuracy among the ones it dared to name, and how often it named the wrong cause."""
    if len(pred) != len(truth):
        raise ValueError("pred/truth length mismatch")
    mat: dict[str, dict[str, int]] = {}
    for a, b in zip(pred, truth):
        mat.setdefault(b.value, {}).setdefault(a.value, 0)
        mat[b.value][a.value] += 1
    named = [(a, b) for a, b in zip(pred, truth) if a not in (FC.UNKNOWN, FC.INSUFFICIENT_EVIDENCE)]
    correct = sum(1 for a, b in named if a == b)
    return {"matrix": mat, "n": len(pred), "named": len(named), "acc_named": correct / len(named) if named else float("nan"),
            "wrong_named": len(named) - correct, "abstain_rate": 1 - len(named) / len(pred) if pred else float("nan")}


# ==================================================================================================================
# planted cases: known cause -> synthetic trade + environment (the classifier's own control experiment)
# ==================================================================================================================
def _healthy_history(rng: np.random.Generator, n: int = 30, level: float = 0.01, noise: float = 0.004) -> list[float]:
    return list(level + rng.normal(0, noise, n))


def _date(i: int) -> str:
    return (np.datetime64("2019-01-07") + np.timedelta64(7 * i, "D")).astype(str)


def _mk_history(pid: str, eff: Sequence[float], **kw) -> PatternHistory:
    return PatternHistory(pid, tuple(float(e) for e in eff), tuple(_date(i) for i in range(len(eff))), **kw)


def planted_case(cause: FailureCause, seed: int = 0, rid: str = "") -> tuple[TradeRecord, FailureEnv]:
    """A synthetic loss whose true cause is `cause` (UNKNOWN gives a loss that is plain bad luck). Every other detector
    is kept quiet so that the classifier is tested on one mechanism at a time. The trade resolves 2019-12-30, so any
    `now` after that is legitimate; pattern histories end before the decision."""
    rng = np.random.default_rng(seed)
    base = dict(rid=rid or f"planted-{cause.value}-{seed}", decided_at="2019-12-20", resolved_at="2019-12-30", side=1,
                pnl=-0.0515, weight=0.10, target_weight=0.10, signal_ret=-0.05, entry_gap=0.001,
                end_ret_from_fill=-0.0510, exit_ret=-0.0510, mfe=0.005, mae=-0.055, exp_move=0.07, exp_vol=0.03,
                dir_prob=0.52, rank_pct=0.9, universe_ret=0.0, pattern_ids=("P1",), knowledge_ids=("K1",),
                context={"m_vol": 0.0, "m_trend": 0.0}, data_flags={"stale_price_days": 0.0, "imputed_frac": 0.0})
    hist = {"P1": _mk_history("P1", _healthy_history(rng), p_real=0.95, t_disc=4.0, t_conf=3.0)}
    ref = ContextReference({"m_vol": 0.0, "m_trend": 0.0}, {"m_vol": 1.0, "m_trend": 1.0})
    know = {"K1": _Know({"m_vol": (-2.0, 2.0)}, {"m_trend": (3.0, None)})}
    env = dict(patterns=hist, context_ref=ref, knowledge=know, universe_sd=0.04)
    if cause == FC.UNKNOWN:
        base.update(exp_vol=0.03)                                        # plain wrong-way move, everything healthy
    elif cause == FC.SELECTION_ERROR:
        base.update(signal_ret=-0.004, end_ret_from_fill=-0.005, exit_ret=-0.005, pnl=-0.0105, mae=-0.02, mfe=0.004,
                    exp_vol=0.008, dir_prob=None)
    elif cause == FC.TIMING_ERROR:
        base.update(signal_ret=0.0404, entry_gap=0.0, end_ret_from_fill=0.0404, exit_ret=-0.03, stop_hit=True, stop=0.03,
                    stop_fill_ret=-0.03, pnl=-0.0305, mae=-0.035, mfe=0.002, best_entry_ret=0.075, exp_vol=0.02,
                    dir_prob=0.6)
    elif cause == FC.RISK_ERROR:
        base.update(signal_ret=-0.18, entry_gap=0.0, end_ret_from_fill=-0.18, exit_ret=-0.18, pnl=-0.1805, stop=0.05,
                    stop_hit=True, stop_fill_ret=-0.18, weight=0.22, exp_vol=0.03, mae=-0.19, mfe=0.0, dir_prob=0.5)
    elif cause == FC.MEASUREMENT_ERROR:
        base.update(signal_ret=-0.06, entry_gap=-0.70, end_ret_from_fill=3.0, exit_ret=-0.0485, pnl=-0.05,
                    data_flags={"stale_price_days": 8.0, "split_suspect": 1.0, "imputed_frac": 0.4}, exp_vol=0.03)
    elif cause == FC.WRONG_CONTEXT:
        base.update(context={"m_vol": 0.5, "m_trend": 3.3}, exp_vol=0.03)
    elif cause == FC.REGIME_CHANGE:
        base.update(context={"m_vol": 4.5, "m_trend": -4.0}, exp_vol=0.03)
        env["knowledge"] = {}
        env["context_ref"] = ContextReference({"m_vol": 0.0, "m_trend": 0.0}, {"m_vol": 1.0, "m_trend": 1.0}, "calm", "crisis")
    elif cause == FC.FALSE_PATTERN:
        env["patterns"] = {"P1": _mk_history("P1", list(rng.normal(0.0, 0.008, 30)), p_real=0.05, t_disc=3.6, t_conf=0.2)}
        base.update(exp_vol=0.03)
    elif cause == FC.WEAKENING_EFFECT:
        eff = list(0.012 + rng.normal(0, 0.003, 24)) + list(np.linspace(0.010, 0.003, 12) + rng.normal(0, 0.0015, 12))
        env["patterns"] = {"P1": _mk_history("P1", eff, p_real=0.9, t_disc=4.0, t_conf=3.0)}
        base.update(exp_vol=0.03)
    elif cause == FC.REVERSAL:
        eff = list(0.012 + rng.normal(0, 0.003, 24)) + list(-0.010 + rng.normal(0, 0.002, 6))
        env["patterns"] = {"P1": _mk_history("P1", eff, p_real=0.9, t_disc=4.0, t_conf=3.0)}
        base.update(exp_vol=0.03)
    elif cause == FC.TEMPORARY_INACTIVITY:
        eff = ([0.012] * 6 + [0.0] * 6 + [0.012] * 6 + [0.0] * 6 + [0.012] * 6)
        eff = list(np.array(eff) + rng.normal(0, 0.002, len(eff))) + [0.001, -0.001, 0.0005, -0.0005, 0.0, 0.0]
        env["patterns"] = {"P1": _mk_history("P1", eff, triggers=tuple([9] * 30 + [0] * 6), p_real=0.9, t_disc=4.0, t_conf=3.0)}
        base.update(exp_vol=0.03)
    elif cause == FC.REDUNDANCY:
        base.update(pattern_ids=("P1", "P2"), exp_vol=0.03)
        env["patterns"] = {**hist, "P2": _mk_history("P2", _healthy_history(rng), p_real=0.95, t_disc=4.0, t_conf=3.0)}
        env["pattern_corr"] = {("P1", "P2"): 0.97}
    elif cause == FC.INTERACTION_FAILURE:
        base.update(pattern_ids=("P1", "P2"), exp_vol=0.03)
        env["patterns"] = {**hist, "P2": _mk_history("P2", _healthy_history(rng), p_real=0.95, t_disc=4.0, t_conf=3.0)}
        env["pair_effects"] = {("P1", "P2"): -0.006}
        env["pattern_corr"] = {("P1", "P2"): 0.2}
    else:
        raise ValueError(f"no planted case for {cause}")
    return TradeRecord(**base), FailureEnv(**env)


@dataclass(frozen=True)
class _Know:
    """Minimal KnowledgeLike stand-in used by planted cases (contexts / anti_contexts only)."""
    contexts: Mapping[str, Any]
    anti_contexts: Mapping[str, Any]


def planted_battery(seeds: Sequence[int] = (0, 1, 2), params: FailureParams | None = None, now="2020-06-01"
                    ) -> dict[str, Any]:
    """Run the classifier over every planted cause and seed and return the confusion. Foundation self-check only."""
    clf = LossClassifier(params)
    pred, truth = [], []
    for c in list(CAUSES_EXPLAINING) + [FC.UNKNOWN]:
        for s in seeds:
            t, env = planted_case(c, s)
            pred.append(clf.classify(t, env, now).cause)
            truth.append(c)
    return confusion(pred, truth)
