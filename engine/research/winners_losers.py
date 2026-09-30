"""Winner research and the shared record layer for winner/loser research (C66 sections 5, 12, 29-31, 34; canons C62, C63, C64, C66).

Section 5 asks, for every meaningful WINNER: why did it move, why did we detect it, why did we rank it highly, which signals
contributed, which were irrelevant, which generalised, could we have predicted the magnitude, could we have predicted the timing.
This module answers those eight questions and, because a winner study that has no losers beside it is survivorship in disguise, it
owns the identity-free MoveRecord that BOTH studies read (engine/research/loss_pipeline.py builds the dedicated loss pipeline on it).

What it does
  * MoveRecord      one matured name-period outcome, with what the system believed and did at the decision, and no identity;
  * ExceptionCollector  streams a market year by year keeping only exception rows plus a seeded control reservoir per day (rule 27:
                    the whole market does not fit in memory);
  * explain_move    market / sector / idiosyncratic / one-day-jump decomposition of a move (why it moved);
  * study_signals   stratified (same-week) AUC of every signal, cases against controls, with a clustered bootstrap, BH control across
                    signals, per-group and per-time-half agreement: INFORMATIVE / NOT_GENERALISED / IRRELEVANT / UNDERPOWERED /
                    MISWEIGHTED. Works on winners (cohort_dir +1) and on losers (-1): the SAME machinery, which is what symmetry means;
  * magnitude_predictability, timing_predictability, rank_calibration  cohort tests against permutation nulls;
  * signal_credit   engine.learning.credit Shapley credit over the signals (own credit, blame on losses);
  * WinnerResearch  fit on matured records, then one WinnerFinding per winner; step() is the single public entry;
  * symmetry_report, review_budget  losers are studied at least as broadly and as deeply as winners, by construction and by audit;
  * to_matured / release  findings live in MATURED_RESEARCH_STATE and reach a decision only through MaturedRecord.gate(now); a
                    finding filed under a year that is being replayed in disguise is refused (same-year rerun leak).

Built on: engine.learning.failure (TradeRecord, surprise_bits), separation.decompose, credit (CreditEngine), lessons (identity keys),
pattern_stats (BH q-values, week clusters), research.core. Deterministic in `seed`; nothing reads a record that has not matured
strictly before `now`. Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, canonical_json, current_code_hash,
                                  require_past, stable_hash)
from engine.learning.credit import CreditConfig, CreditEngine, Decision, DecisionLedger, WeightedSumCombiner
from engine.learning.failure import TradeRecord
from engine.learning.separation import SeparationParams, decompose
from engine.lessons import FORBIDDEN_KEYS
from engine.pattern_stats import bh_qvalues, week_codes
from engine.research.core import Availability, MaturedRecord, Namespace

NL = chr(10)


@functools.lru_cache(maxsize=1)
def cached_code_hash() -> str:
    """engine.provenance walks every loaded module (~0.3 s); the code cannot change inside one run, so it is computed once."""
    return current_code_hash()


# ==================================================================================================================
# vocabulary
# ==================================================================================================================
class MoveKind(_StrEnum):
    WINNER = "WINNER"
    LOSER = "LOSER"
    NEUTRAL = "NEUTRAL"


class MoveDriver(_StrEnum):               # question 1: why did it move
    MARKET = "MARKET"
    SECTOR = "SECTOR"
    EVENT_JUMP = "EVENT_JUMP"
    IDIOSYNCRATIC_DRIFT = "IDIOSYNCRATIC_DRIFT"
    MIXED = "MIXED"
    UNEXPLAINED = "UNEXPLAINED"           # inputs missing: never guessed


class DetectionMode(_StrEnum):            # question 2: why did we detect it (or not)
    EARNED = "EARNED"                     # held, and the volatility model gave the move real probability beforehand
    LUCKY = "LUCKY"                       # held, but the model gave the move little probability: a win that confirms nothing
    UNPROVEN = "UNPROVEN"                 # held, no probability was logged
    REJECTED = "REJECTED"                 # considered and not taken
    MISSED = "MISSED"                     # never considered


class SignalVerdict(_StrEnum):            # questions 4-6: contributed / irrelevant / generalised
    INFORMATIVE = "INFORMATIVE"
    NOT_GENERALISED = "NOT_GENERALISED"
    IRRELEVANT = "IRRELEVANT"
    UNDERPOWERED = "UNDERPOWERED"
    MISWEIGHTED = "MISWEIGHTED"           # the effect is real but the system's weight has the opposite sign


class Predictability(_StrEnum):           # questions 7-8: magnitude and timing
    PREDICTABLE = "PREDICTABLE"
    WEAK = "WEAK"
    NONE_DETECTED = "NONE_DETECTED"       # well powered, no relation found
    UNDERPOWERED = "UNDERPOWERED"


WINNER_QUESTIONS = ("why_moved", "why_detected", "why_ranked", "signals_contributed", "signals_irrelevant",
                    "signals_generalised", "magnitude_predictable", "timing_predictable")


@dataclass(frozen=True)
class ResearchParams:
    win_thr: float = 0.05                 # |return| at or above this is a meaningful winner/loser
    loss_thr: float = 0.02                # a held position whose net pnl is this negative is a meaningful loss, whatever the stock did
    extreme_thr: float = 0.12
    control_thr: float = 0.02             # |return| below this can be a control (a name that did not move)
    prob_hi: float = 0.5                  # model probability of a large move counted as "the model saw it coming"
    rank_hi: float = 0.8
    dir_margin: float = 0.05              # |dir_prob - 0.5| below this is not a direction call
    min_n: int = 30                       # cases and controls needed before a signal can be called anything
    min_group_n: int = 12
    min_groups: int = 2
    agree_min: float = 0.75               # share of groups that must agree in sign with the pooled effect
    alpha: float = 0.05
    min_edge: float = 0.03                # |AUC - 0.5| that matters
    n_boot: int = 300
    min_strata_boot: int = 8              # fewer weekly strata than this and the CI is the null-variance normal one
    market_share: float = 0.6
    sector_share: float = 0.5
    event_share: float = 0.6
    active_z: float = 1.0                 # a signal is "active" in a record when |z| is at least this
    timing_tol: float = 0.25              # timing within this fraction of the horizon counts as predicted
    rho_predictable: float = 0.2
    default_side: int = 1                 # side a not-held name would have been given without a direction call
    cost: float = 0.0005
    controls_per_day: int = 20
    group_key: str = "era"
    max_credit_signals: int = 8

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.control_thr < self.win_thr < self.extreme_thr:
            errs.append("need 0 < control_thr < win_thr < extreme_thr")
        if not 0 < self.alpha < 0.5:
            errs.append("alpha must be in (0, 0.5)")
        if self.min_n < 5 or self.min_group_n < 3 or self.min_groups < 1:
            errs.append("sample minimums too small to test anything")
        if not 0.5 <= self.agree_min <= 1.0:
            errs.append("agree_min must be in [0.5, 1]")
        if self.default_side not in (-1, 1):
            errs.append("default_side must be +1 or -1")
        if not 0 < self.min_edge < 0.5:
            errs.append("min_edge must be in (0, 0.5)")
        return errs

    def hash(self) -> str:
        return stable_hash(self)


def _fin(x) -> float | None:
    """float(x), or None when x is missing / NaN / infinite. Missing is never read as zero."""
    if x is None or isinstance(x, (bool, str)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ==================================================================================================================
# the record both studies read
# ==================================================================================================================
@dataclass(frozen=True)
class MoveRecord:
    """One matured name-period, identity-free. `ret` is the raw return from the decision close to the horizon end. What the system
    knew and did is recorded beside what happened: considered / held / probabilities / rank / signals, so the same record can be
    asked 'why did it win' or 'why did it lose'. Keys of signals / context / events / tags must not be identities."""
    rid: str
    decided_at: str
    resolved_at: str
    horizon_days: int
    ret: float
    market_ret: float | None = None
    sector_ret: float | None = None
    beta: float = 1.0
    move_day: int | None = None           # trading day (1..horizon) of the largest single-day move
    max_day_ret: float | None = None
    peak_ret: float | None = None         # best raw excursion from the fill (>= 0)
    trough_ret: float | None = None       # worst raw excursion from the fill (<= 0)
    considered: bool = False
    held: bool = False
    side: int = 0
    exp_move: float | None = None
    prob_move: float | None = None        # the volatility model's probability of a large move
    exp_day: int | None = None
    dir_prob: float | None = None         # P(up); None when no direction model spoke
    rank_pct: float | None = None
    exp_vol: float | None = None
    weight: float | None = None
    target_weight: float | None = None
    cost: float = 0.0005
    pnl: float | None = None
    entry_gap: float | None = None
    end_ret_from_fill: float | None = None
    exit_ret: float | None = None
    mfe: float | None = None
    mae: float | None = None
    best_entry_ret: float | None = None
    prior_ret: float | None = None
    stop: float | None = None
    stop_hit: bool = False
    stop_fill_ret: float | None = None
    universe_ret: float | None = None
    signals: Mapping[str, float] = field(default_factory=dict)
    pattern_ids: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    context: Mapping[str, float] = field(default_factory=dict)
    events: Mapping[str, str] = field(default_factory=dict)          # event name -> Availability value at the decision
    data_flags: Mapping[str, float] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        if not self.rid:
            errs.append("rid missing")
        try:
            if as_date(self.resolved_at) <= as_date(self.decided_at):
                errs.append("resolved_at must be after decided_at")
        except Exception as e:
            errs.append(f"dates unparsable: {e}")
        if self.horizon_days < 1:
            errs.append("horizon_days must be >= 1")
        r = _fin(self.ret)
        if r is None or r <= -1.0:
            errs.append("ret must be finite and above -1")
        if self.move_day is not None and not 1 <= self.move_day <= max(1, self.horizon_days):
            errs.append("move_day outside 1..horizon_days")
        if self.exp_day is not None and not 1 <= self.exp_day <= max(1, self.horizon_days):
            errs.append("exp_day outside 1..horizon_days")
        for name in ("prob_move", "dir_prob", "rank_pct"):
            v = getattr(self, name)
            if v is not None and (_fin(v) is None or not 0.0 <= v <= 1.0):
                errs.append(f"{name} outside [0,1]")
        for name in ("exp_move", "exp_vol"):
            v = getattr(self, name)
            if v is not None and (_fin(v) is None or v <= 0):
                errs.append(f"{name} must be positive")
        if _fin(self.beta) is None:
            errs.append("beta not finite")
        if self.held and (self.side not in (-1, 1) or _fin(self.pnl) is None):
            errs.append("a held record needs side +1/-1 and a finite pnl")
        if self.held and not self.considered:
            errs.append("held implies considered")
        for group, d in (("signals", self.signals), ("context", self.context), ("events", self.events),
                         ("data_flags", self.data_flags), ("tags", self.tags)):
            for k in d:
                if str(k).lower() in FORBIDDEN_KEYS:
                    errs.append(f"identity key {k!r} in {group}")
        for k, v in self.signals.items():
            if isinstance(v, float) and math.isinf(v):
                errs.append(f"signal {k} is infinite")
        for k, v in self.events.items():
            if str(v) not in {a.value for a in Availability}:
                errs.append(f"event {k}: unknown availability {v!r}")
        return errs

    def require_valid(self) -> None:
        errs = self.validate()
        if errs:
            raise ValueError(f"MoveRecord {self.rid!r} invalid: " + "; ".join(errs))

    def kind(self, p: ResearchParams) -> MoveKind:
        if self.ret >= p.win_thr:
            return MoveKind.WINNER
        if self.ret <= -p.win_thr:
            return MoveKind.LOSER
        return MoveKind.NEUTRAL

    def is_winner(self, p: ResearchParams) -> bool:
        """A meaningful winner: the stock rose by win_thr, or a held position earned at least win_thr."""
        return self.ret >= p.win_thr or (self.held and self.pnl is not None and self.pnl >= p.win_thr)

    def is_loser(self, p: ResearchParams) -> bool:
        """A meaningful loser: the stock fell by win_thr, or a held position lost loss_thr net - even when the stock ended up
        (a shake-out, a gap, an oversized bet). Both populations are studied; a position can be lost without the stock falling."""
        return self.ret <= -p.win_thr or (self.held and self.pnl is not None and self.pnl <= -p.loss_thr)

    def hyp_side(self, p: ResearchParams) -> int:
        """The side this name would have been given: the real one when held, else the direction call, else the default."""
        if self.held:
            return self.side
        d = _fin(self.dir_prob)
        if d is not None and abs(d - 0.5) >= p.dir_margin:
            return 1 if d > 0.5 else -1
        return p.default_side

    def year_filed(self) -> int:
        return as_date(self.resolved_at).year


def to_trade_record(m: MoveRecord, p: ResearchParams | None = None) -> TradeRecord:
    """The existing failure-learning currency. A name that was not held is given the side it would have had, so 'what would this
    have cost us' can be asked of avoided and missed names with the same detectors."""
    p = p or ResearchParams()
    m.require_valid()
    side = m.hyp_side(p)
    held_pnl = _fin(m.pnl)
    pnl = held_pnl if m.held and held_pnl is not None else side * m.ret - m.cost
    mfe, mae = m.mfe, m.mae
    if not m.held and m.peak_ret is not None and m.trough_ret is not None:
        mfe, mae = (m.peak_ret, m.trough_ret) if side > 0 else (-m.trough_ret, -m.peak_ret)
    return TradeRecord(
        rid=m.rid, decided_at=m.decided_at, resolved_at=m.resolved_at, side=side, pnl=pnl, weight=m.weight,
        target_weight=m.target_weight, cost=m.cost, signal_ret=m.ret, entry_gap=m.entry_gap,
        end_ret_from_fill=m.end_ret_from_fill, exit_ret=m.exit_ret, mfe=mfe, mae=mae, best_entry_ret=m.best_entry_ret,
        exp_move=m.exp_move, exp_vol=m.exp_vol, dir_prob=m.dir_prob, prior_ret=m.prior_ret, rank_pct=m.rank_pct,
        universe_ret=m.market_ret if m.universe_ret is None else m.universe_ret, stop=m.stop, stop_hit=m.stop_hit,
        stop_fill_ret=m.stop_fill_ret, pattern_ids=m.pattern_ids, knowledge_ids=m.knowledge_ids,
        context=dict(m.context), data_flags=dict(m.data_flags), tags=dict(m.tags))


def move_from_trade(t: TradeRecord, horizon_days: int = 5, **extra: Any) -> MoveRecord:
    """The existing simulator/ledger currency (TradeRecord) as a held MoveRecord, so the trades the system already logs feed the
    same winner and loss studies. `extra` may add signals, events, prob_move and the rest of what a TradeRecord does not carry."""
    if t.signal_ret is None:
        raise ValueError(f"trade {t.rid}: signal_ret is required")
    kw: dict[str, Any] = dict(
        rid=t.rid, decided_at=t.decided_at, resolved_at=t.resolved_at, horizon_days=horizon_days, ret=float(t.signal_ret),
        market_ret=t.universe_ret, considered=True, held=True, side=t.side, exp_move=t.exp_move, dir_prob=t.dir_prob,
        rank_pct=t.rank_pct, exp_vol=t.exp_vol, weight=t.weight, target_weight=t.target_weight, cost=t.cost, pnl=t.pnl,
        entry_gap=t.entry_gap, end_ret_from_fill=t.end_ret_from_fill, exit_ret=t.exit_ret, mfe=t.mfe, mae=t.mae,
        best_entry_ret=t.best_entry_ret, prior_ret=t.prior_ret, stop=t.stop, stop_hit=t.stop_hit, stop_fill_ret=t.stop_fill_ret,
        universe_ret=t.universe_ret, pattern_ids=t.pattern_ids, knowledge_ids=t.knowledge_ids, context=dict(t.context),
        data_flags=dict(t.data_flags), tags=dict(t.tags))
    kw.update(extra)
    return MoveRecord(**kw)


def move_surprise(m: MoveRecord) -> float | None:
    """-log2 of the probability the volatility model gave to a large move, for a record in which one happened: a win or loss the
    model saw coming costs ~0 bits, one it thought a 5% chance costs ~4.3. None when no probability was logged."""
    if m.prob_move is None:
        return None
    return float(-math.log2(min(1 - 1e-9, max(1e-9, m.prob_move))))


def is_control(m: MoveRecord, p: ResearchParams) -> bool:
    """A name that did not move and that we did not win or lose on: the comparison group for every signal study."""
    return abs(m.ret) < p.control_thr and not m.is_winner(p) and not m.is_loser(p)


def matured_only(records: Iterable[MoveRecord], now, strict: bool = False) -> tuple[list[MoveRecord], int]:
    """Records whose outcome matured strictly before `now`, plus the count still pending. strict=True raises instead."""
    ok, pending = [], 0
    for m in records:
        try:
            require_past(m.resolved_at, now, f"record {m.rid} outcome")
        except FirewallBreach:
            if strict:
                raise
            pending += 1
            continue
        ok.append(m)
    return ok, pending


# ==================================================================================================================
# streaming market-wide collection (rule 27): exception rows plus a seeded control reservoir, never the whole market
# ==================================================================================================================
_SCALAR_COLUMNS = ("market_ret", "sector_ret", "beta", "move_day", "max_day_ret", "peak_ret", "trough_ret", "exp_move", "prob_move",
                   "exp_day", "dir_prob", "rank_pct", "exp_vol", "weight", "target_weight", "pnl", "entry_gap",
                   "end_ret_from_fill", "exit_ret", "mfe", "mae", "best_entry_ret", "prior_ret", "stop", "stop_fill_ret", "universe_ret")


class ExceptionCollector:
    """Consumes one point-in-time frame per day (rows = names, index = private key, never stored) and keeps the exceptions:
    meaningful winners/losers, held names, and highly ranked names that did nothing (false positives), plus `controls_per_day`
    names that did not move, drawn without replacement from a per-day child of the seed. Peak memory is one day's frame."""

    def __init__(self, params: ResearchParams | None = None, seed: int = 0, salt: str = "wl"):
        self.p = params or ResearchParams()
        self.seed = seed
        self.salt = salt
        self._rows: list[MoveRecord] = []
        self.days = 0
        self.rows_seen = 0
        self.peak_rows = 0
        self._seen_days: set[str] = set()

    def add_day(self, frame: pd.DataFrame, decided_at: str, resolved_at: str, horizon_days: int = 5,
                snapshot: Mapping[str, float] | None = None, tags: Mapping[str, str] | None = None) -> int:
        """Adds one day. Columns: ret (required); considered, held, side booleans/ints; any of _SCALAR_COLUMNS; s_* signal columns.
        `snapshot` is the day's market context (one per day). Returns how many records were kept."""
        if decided_at in self._seen_days:
            raise ValueError(f"day {decided_at} already collected: history is append-only")
        if "ret" not in frame.columns:
            raise ValueError("frame needs a 'ret' column")
        self._seen_days.add(decided_at)
        self.days += 1
        self.rows_seen += len(frame)
        self.peak_rows = max(self.peak_rows, len(frame))
        if frame.empty:
            return 0
        ret = frame["ret"].to_numpy(dtype=float)
        finite = np.isfinite(ret)
        big = finite & (np.abs(ret) >= self.p.win_thr)
        held = frame["held"].to_numpy(dtype=bool) if "held" in frame else np.zeros(len(frame), dtype=bool)
        rank = frame["rank_pct"].to_numpy(dtype=float) if "rank_pct" in frame else np.full(len(frame), np.nan)
        fp = finite & (np.nan_to_num(rank, nan=0.0) >= self.p.rank_hi)
        exc = big | (finite & held) | fp
        ctrl_pool = np.flatnonzero(finite & ~exc & (np.abs(ret) < self.p.control_thr))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(as_date(decided_at).toordinal())]))
        take = min(len(ctrl_pool), self.p.controls_per_day)
        ctrl = rng.choice(ctrl_pool, size=take, replace=False) if take else np.array([], dtype=int)
        keep = np.concatenate([np.flatnonzero(exc), np.sort(ctrl)]).astype(int)
        sig_cols = [c for c in frame.columns if c.startswith("s_")]
        n0 = len(self._rows)
        for pos in keep:
            row = frame.iloc[pos]
            key = frame.index[pos]
            kw: dict[str, Any] = {c: _fin(row[c]) for c in _SCALAR_COLUMNS if c in frame.columns}
            for c in ("move_day", "exp_day"):
                if kw.get(c) is not None:
                    kw[c] = int(kw[c])
            if kw.get("beta") is None:
                kw.pop("beta", None)
            sig = {c[2:]: float(row[c]) for c in sig_cols if _fin(row[c]) is not None}
            held_i = bool(row["held"]) if "held" in frame.columns else False
            side = int(row["side"]) if held_i and "side" in frame.columns else 0
            if held_i and kw.get("pnl") is None:
                kw["pnl"] = side * float(row["ret"]) - 0.0005
            self._rows.append(MoveRecord(
                rid=stable_hash({"s": self.salt, "d": decided_at, "k": str(key)}, 14), decided_at=decided_at,
                resolved_at=resolved_at, horizon_days=horizon_days, ret=float(row["ret"]),
                considered=bool(row["considered"]) if "considered" in frame.columns else held_i, held=held_i, side=side,
                signals=sig, context=dict(snapshot or {}), tags=dict(tags or {}), **kw))
        return len(self._rows) - n0

    def records(self) -> list[MoveRecord]:
        return list(self._rows)

    def stats(self) -> dict[str, float]:
        kept = len(self._rows)
        return {"days": self.days, "rows_seen": self.rows_seen, "kept": kept, "peak_rows": self.peak_rows,
                "keep_rate": kept / self.rows_seen if self.rows_seen else float("nan")}


# ==================================================================================================================
# question 1: why did it move
# ==================================================================================================================
@dataclass(frozen=True)
class MoveExplanation:
    ret: float
    market: float | None
    sector: float | None
    idio: float | None
    market_share: float | None
    sector_share: float | None
    idio_share: float | None
    event_share: float | None
    driver: MoveDriver
    notes: tuple[str, ...] = ()


def explain_move(m: MoveRecord, p: ResearchParams | None = None) -> MoveExplanation:
    """ret = beta*market + (sector - market) + idiosyncratic, by construction; the one-day jump share says whether the idiosyncratic
    part arrived as an event or as drift. Inputs that are missing leave their part None and the driver UNEXPLAINED, never guessed."""
    p = p or ResearchParams()
    r = float(m.ret)
    notes = []
    mk = _fin(m.market_ret)
    beta = _fin(m.beta)
    market = beta * mk if mk is not None and beta is not None else None
    sec_ret = _fin(m.sector_ret)
    sector = sec_ret - mk if sec_ret is not None and mk is not None else None
    if market is None:
        notes.append("market_ret missing: the market share of the move is unknown")
    idio = r - (market or 0.0) - (sector or 0.0)
    if abs(r) < 1e-9:
        return MoveExplanation(r, market, sector, idio, None, None, None, None, MoveDriver.UNEXPLAINED,
                               tuple(notes + ["return is zero: nothing to explain"]))
    share = lambda x: None if x is None else float(x / r)
    ms, ss, isr = share(market), share(sector), share(idio)
    jump = _fin(m.max_day_ret)
    ev = float(max(0.0, jump / r)) if jump is not None else None
    if market is None and ev is None:
        return MoveExplanation(r, market, sector, idio, ms, ss, isr, ev, MoveDriver.UNEXPLAINED, tuple(notes))
    if ms is not None and ms >= p.market_share:
        driver = MoveDriver.MARKET
    elif ss is not None and ss >= p.sector_share:
        driver = MoveDriver.SECTOR
    elif ev is not None and ev >= p.event_share and (isr is None or isr >= 0.3):
        driver = MoveDriver.EVENT_JUMP
    elif isr is not None and isr >= 0.5:
        driver = MoveDriver.IDIOSYNCRATIC_DRIFT
    else:
        driver = MoveDriver.MIXED
    if market is None:
        notes.append("driver rests on the one-day jump alone")
    return MoveExplanation(r, market, sector, idio, ms, ss, isr, ev, driver, tuple(notes))


# ==================================================================================================================
# signal study: stratified AUC of cases against controls
# ==================================================================================================================
@dataclass(frozen=True)
class AucResult:
    auc: float
    lo: float
    hi: float
    p: float
    n_case: int
    n_ctrl: int
    n_strata: int
    pairs: float


def _group_by_code(x: np.ndarray, codes: np.ndarray) -> dict[int, np.ndarray]:
    if len(x) == 0:
        return {}
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    cuts = np.flatnonzero(np.diff(cs)) + 1
    return {int(k): v for k, v in zip(np.unique(cs), np.split(x[order], cuts))}


def week_of(dates: Sequence[str]) -> np.ndarray:
    """ISO year-week code per date: names decided in one week share that week's market, so they are compared only with each other."""
    if len(dates) == 0:
        return np.array([], dtype=np.int64)
    ts = pd.DatetimeIndex([pd.Timestamp(as_date(d)) for d in dates])
    iso = ts.isocalendar()
    return (iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)).to_numpy()


def stratified_auc(x_case: np.ndarray, s_case: np.ndarray, x_ctrl: np.ndarray, s_ctrl: np.ndarray, n_boot: int = 300,
                   min_strata_boot: int = 8, seed: int = 0) -> AucResult:
    """P(case > control) within the same stratum, pooled over strata by pair count. p from the exact null moments of the Mann-Whitney
    U (with tie correction); the CI is a stratum-clustered bootstrap when there are enough strata, else the null-variance normal CI."""
    a, b = _group_by_code(x_case, s_case), _group_by_code(x_ctrl, s_ctrl)
    rows = []
    for c in sorted(set(a) & set(b)):
        u1, u0 = a[c], b[c]
        n1, n0 = len(u1), len(u0)
        allv = np.concatenate([u1, u0])
        r = stats.rankdata(allv)
        u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
        n = n1 + n0
        _, cnt = np.unique(allv, return_counts=True)
        ties = float((cnt ** 3 - cnt).sum())
        var = n1 * n0 / 12.0 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
        rows.append((u, n1 * n0, n1 * n0 / 2.0, var))
    if not rows:
        return AucResult(float("nan"), float("nan"), float("nan"), 1.0, len(x_case), len(x_ctrl), 0, 0.0)
    M = np.array(rows, dtype=float)
    pairs = float(M[:, 1].sum())
    auc = float(M[:, 0].sum() / pairs)
    var = float(M[:, 3].sum())
    z = (M[:, 0].sum() - M[:, 2].sum()) / math.sqrt(var) if var > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    k = len(rows)
    if k >= min_strata_boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, k, size=(n_boot, k))
        boots = M[idx, 0].sum(axis=1) / M[idx, 1].sum(axis=1)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        se = math.sqrt(var) / pairs if var > 0 else 0.5
        lo, hi = max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)
    return AucResult(auc, lo, hi, p, len(x_case), len(x_ctrl), k, pairs)


def signal_frame(records: Sequence[MoveRecord], group_key: str = "era") -> pd.DataFrame:
    """One row per record: its signals (NaN where absent), the week stratum, the time ordinal and the group label."""
    rows = [dict(m.signals) for m in records]
    df = pd.DataFrame(rows, index=range(len(records)), dtype=float)
    df["_week"] = week_of([m.decided_at for m in records])
    df["_t"] = [as_date(m.decided_at).toordinal() for m in records]
    df["_group"] = [str(m.tags.get(group_key, "?")) for m in records]
    return df


@dataclass(frozen=True)
class SignalStat:
    signal: str
    n_case: int
    n_ctrl: int
    auc: float
    lo: float
    hi: float
    p: float
    q: float
    n_groups: int
    agree: float | None                   # share of testable groups on the same side of 0.5 as the pooled effect
    half_agree: bool | None
    weight: float | None
    verdict: SignalVerdict
    groups: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    mde: float | None = None              # smallest |AUC - 0.5| this sample could have detected (80% power)

    @property
    def edge(self) -> float:
        return self.auc - 0.5


def minimum_detectable_edge(n_case: int, n_ctrl: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest |AUC - 0.5| a Mann-Whitney comparison of this size can detect: (z_alpha/2 + z_power) x the null standard error
    sqrt((n1 + n0 + 1) / (12 n1 n0)). It is what makes IRRELEVANT honest: 'no effect' is only a finding when the sample could have seen one."""
    if n_case < 2 or n_ctrl < 2:
        return float("inf")
    se = math.sqrt((n_case + n_ctrl + 1) / (12.0 * n_case * n_ctrl))
    return float((stats.norm.isf(alpha / 2.0) + stats.norm.ppf(power)) * se)


def _side(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def study_signals(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], p: ResearchParams | None = None,
                  weights: Mapping[str, float] | None = None, cohort_dir: int = 1, seed: int = 0) -> list[SignalStat]:
    """For each signal: is it different in the cases than in the same-week controls, by how much, is that stable across groups
    (eras / types) and across time, and does the system weight it in the direction the evidence supports?
    cohort_dir = +1 for a winner cohort, -1 for a loser cohort: a positively weighted signal should be HIGHER in winners and LOWER in
    losers, so the same code judges both. A signal that cannot be tested is UNDERPOWERED, never IRRELEVANT."""
    p = p or ResearchParams()
    if cohort_dir not in (-1, 1):
        raise ValueError("cohort_dir must be +1 or -1")
    if not cases or not controls:
        return []
    dc, dn = signal_frame(cases, p.group_key), signal_frame(controls, p.group_key)
    names = sorted((set(dc.columns) & set(dn.columns)) - {"_week", "_t", "_group"})
    tmed = float(np.median(dc["_t"].to_numpy()))
    raw: list[dict[str, Any]] = []
    for j, s in enumerate(names):
        xc, xn = dc[s].to_numpy(dtype=float), dn[s].to_numpy(dtype=float)
        mc, mn = np.isfinite(xc), np.isfinite(xn)
        res = stratified_auc(xc[mc], dc["_week"].to_numpy()[mc], xn[mn], dn["_week"].to_numpy()[mn], p.n_boot,
                             p.min_strata_boot, seed + j)
        groups: dict[str, float] = {}
        for g in sorted(set(dc["_group"]) & set(dn["_group"])):
            gc, gn = mc & (dc["_group"].to_numpy() == g), mn & (dn["_group"].to_numpy() == g)
            if gc.sum() >= p.min_group_n and gn.sum() >= p.min_group_n:
                gr = stratified_auc(xc[gc], dc["_week"].to_numpy()[gc], xn[gn], dn["_week"].to_numpy()[gn], 50, 10 ** 9, seed)
                if gr.pairs > 0:
                    groups[g] = gr.auc
        halves = []
        for early in (True, False):
            hc = mc & ((dc["_t"].to_numpy() <= tmed) == early)
            hn = mn & ((dn["_t"].to_numpy() <= tmed) == early)
            if hc.sum() >= p.min_group_n and hn.sum() >= p.min_group_n:
                hr = stratified_auc(xc[hc], dc["_week"].to_numpy()[hc], xn[hn], dn["_week"].to_numpy()[hn], 50, 10 ** 9, seed)
                halves.append(hr.auc if hr.pairs > 0 else float("nan"))
            else:
                halves.append(float("nan"))
        raw.append({"s": s, "res": res, "groups": groups, "halves": halves})
    testable = [i for i, r in enumerate(raw) if r["res"].pairs > 0]
    q = np.ones(len(raw))
    if testable:
        q[testable] = bh_qvalues(np.array([raw[i]["res"].p for i in testable]))
    out = []
    for i, r in enumerate(raw):
        res = r["res"]
        reasons: list[str] = []
        w = _fin(weights.get(r["s"])) if weights else None
        agree = None
        if res.pairs > 0 and r["groups"]:
            agree = float(np.mean([_side(v - 0.5) == _side(res.auc - 0.5) for v in r["groups"].values()]))
        half_ok: bool | None = None
        hv = [h for h in r["halves"] if math.isfinite(h)]
        if len(hv) == 2:
            half_ok = all(_side(h - 0.5) == _side(res.auc - 0.5) for h in hv)
        if res.pairs <= 0 or res.n_case < p.min_n or res.n_ctrl < p.min_n:
            verdict = SignalVerdict.UNDERPOWERED
            reasons.append(f"cases={res.n_case} controls={res.n_ctrl} (need {p.min_n} each and shared weeks)")
        else:
            sig = q[i] <= p.alpha and abs(res.auc - 0.5) >= p.min_edge
            if sig:
                if w is not None and w != 0 and _side(w) * cohort_dir != _side(res.auc - 0.5):
                    verdict = SignalVerdict.MISWEIGHTED
                    reasons.append(f"effect sign {_side(res.auc - 0.5):+d} contradicts weight {w:+.3g} for a cohort_dir {cohort_dir:+d} cohort")
                elif len(r["groups"]) < p.min_groups:
                    verdict = SignalVerdict.NOT_GENERALISED
                    reasons.append(f"only {len(r['groups'])} testable group(s): generalisation cannot be shown")
                elif agree is not None and agree < p.agree_min:
                    verdict = SignalVerdict.NOT_GENERALISED
                    reasons.append(f"only {agree:.0%} of groups agree with the pooled sign")
                elif half_ok is not True:
                    verdict = SignalVerdict.NOT_GENERALISED
                    reasons.append("time halves disagree or cannot be tested")
                else:
                    verdict = SignalVerdict.INFORMATIVE
            elif res.lo >= 0.5 - p.min_edge and res.hi <= 0.5 + p.min_edge and \
                    minimum_detectable_edge(res.n_case, res.n_ctrl, p.alpha) <= p.min_edge * 1.5:
                verdict = SignalVerdict.IRRELEVANT
                reasons.append("interval is inside the no-effect band and the sample could have detected an effect that size")
            else:
                verdict = SignalVerdict.UNDERPOWERED
                reasons.append("not significant, but the interval is too wide to call it irrelevant")
        out.append(SignalStat(r["s"], res.n_case, res.n_ctrl, res.auc, res.lo, res.hi, res.p, float(q[i]), len(r["groups"]),
                              agree, half_ok, w, verdict, dict(r["groups"]), tuple(reasons),
                              minimum_detectable_edge(res.n_case, res.n_ctrl, p.alpha)))
    return sorted(out, key=lambda s: (-abs(s.edge) if math.isfinite(s.edge) else 0.0, s.signal))


def verdict_map(stats_: Sequence[SignalStat]) -> dict[str, SignalVerdict]:
    return {s.signal: s.verdict for s in stats_}


def discriminators(selected: Sequence[MoveRecord], p: ResearchParams | None = None, weights: Mapping[str, float] | None = None,
                   seed: int = 0) -> list[SignalStat]:
    """Among the names the system chose, which signals separate the ones that lost from the ones that won (the false-positive
    signature)? Losers are the cases, winners the controls; the machinery is the one used against non-movers."""
    p = p or ResearchParams()
    good = [m for m in selected if m.hyp_side(p) * m.ret >= p.win_thr]
    bad = [m for m in selected if m.hyp_side(p) * m.ret <= -p.win_thr]
    return study_signals(bad, good, p, weights, cohort_dir=-1, seed=seed)


# ==================================================================================================================
# questions 7 and 8: magnitude and timing; and the ranking's calibration
# ==================================================================================================================
@dataclass(frozen=True)
class PredictabilityStat:
    what: str
    n: int
    rho: float | None
    p: float | None
    skill: float | None                   # 1 - error of the model / error of the trivial predictor
    verdict: Predictability
    detail: Mapping[str, float] = field(default_factory=dict)


def magnitude_predictability(movers: Sequence[MoveRecord], p: ResearchParams | None = None, seed: int = 0) -> PredictabilityStat:
    """Does the expected move size track the realised size among names that really moved? Spearman with a seeded permutation p-value,
    the calibration slope and an error skill score against predicting the mean realised size."""
    p = p or ResearchParams()
    rows = [(m.exp_move, abs(m.ret)) for m in movers if m.exp_move is not None]
    if len(rows) < p.min_n:
        return PredictabilityStat("magnitude", len(rows), None, None, None, Predictability.UNDERPOWERED,
                                  {"needed": float(p.min_n)})
    e, a = np.array([r[0] for r in rows], dtype=float), np.array([r[1] for r in rows], dtype=float)
    rho = float(stats.spearmanr(e, a)[0]) if np.ptp(e) > 0 and np.ptp(a) > 0 else 0.0
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(rng.permutation(e), a)[0] for _ in range(400)]) if np.ptp(e) > 0 else np.zeros(400)
    pv = float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null)))
    slope = float(np.polyfit(e, a, 1)[0]) if np.ptp(e) > 0 else 0.0
    mse_model, mse_base = float(np.mean((a - e) ** 2)), float(np.var(a))
    skill = 1.0 - mse_model / mse_base if mse_base > 0 else None
    verdict = (Predictability.PREDICTABLE if pv <= p.alpha and rho >= p.rho_predictable else
               Predictability.WEAK if pv <= p.alpha and rho > 0 else Predictability.NONE_DETECTED)
    return PredictabilityStat("magnitude", len(rows), rho, pv, skill, verdict,
                              {"slope": slope, "under_share": float(np.mean(a > e)), "mean_ratio": float(a.mean() / e.mean())})


def timing_predictability(movers: Sequence[MoveRecord], p: ResearchParams | None = None, seed: int = 0) -> PredictabilityStat:
    """Is the predicted day of the move closer to the realised day than a shuffled prediction would be? Median absolute error as a
    fraction of the horizon, against a permutation null; also reports where in the horizon moves land (early/late mass)."""
    p = p or ResearchParams()
    rows = [(m.exp_day, m.move_day, m.horizon_days) for m in movers if m.exp_day is not None and m.move_day is not None]
    days = [m.move_day / m.horizon_days for m in movers if m.move_day is not None]
    detail = {"early_mass": float(np.mean([d <= 1 / 3 for d in days])) if days else float("nan"),
              "late_mass": float(np.mean([d > 2 / 3 for d in days])) if days else float("nan")}
    if len(rows) < p.min_n:
        return PredictabilityStat("timing", len(rows), None, None, None, Predictability.UNDERPOWERED, detail)
    e = np.array([r[0] / r[2] for r in rows], dtype=float)
    d = np.array([r[1] / r[2] for r in rows], dtype=float)
    err = float(np.median(np.abs(e - d)))
    rng = np.random.default_rng(seed)
    null = np.array([np.median(np.abs(rng.permutation(e) - d)) for _ in range(400)])
    pv = float((1 + np.sum(null <= err)) / (1 + len(null)))
    skill = 1.0 - err / float(np.median(null)) if np.median(null) > 0 else None
    within = float(np.mean(np.abs(e - d) <= p.timing_tol))
    rho = float(stats.spearmanr(e, d)[0]) if np.ptp(e) > 0 and np.ptp(d) > 0 else 0.0
    verdict = (Predictability.PREDICTABLE if pv <= p.alpha and rho >= p.rho_predictable else
               Predictability.WEAK if pv <= p.alpha else Predictability.NONE_DETECTED)
    return PredictabilityStat("timing", len(rows), rho, pv, skill, verdict, {**detail, "within_tol": within, "median_err": err})


def rank_calibration(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], p: ResearchParams | None = None,
                     seed: int = 0) -> AucResult:
    """Did the ranking separate the movers from the non-movers? Same-week AUC of rank_pct, cases against controls."""
    p = p or ResearchParams()
    c = [m for m in cases if m.rank_pct is not None]
    n = [m for m in controls if m.rank_pct is not None]
    return stratified_auc(np.array([m.rank_pct for m in c], dtype=float), week_of([m.decided_at for m in c]),
                          np.array([m.rank_pct for m in n], dtype=float), week_of([m.decided_at for m in n]),
                          p.n_boot, p.min_strata_boot, seed)


# ==================================================================================================================
# Shapley credit over the signals (engine.learning.credit)
# ==================================================================================================================
def signal_credit(records: Sequence[MoveRecord], weights: Mapping[str, float], now, p: ResearchParams | None = None,
                  cfg: CreditConfig | None = None) -> dict[str, dict[str, Any]]:
    """Which of the system's weighted signals earned the outcome, and which cost it? Every record contributes its hypothetical
    side x return as the payoff; credit_on_wins / blame_on_losses show the two halves separately. Uses the top
    `max_credit_signals` signals by |weight| (exact Shapley is exponential in components)."""
    p = p or ResearchParams()
    top = sorted(((abs(w), k) for k, w in weights.items() if _fin(w)), reverse=True)[:p.max_credit_signals]
    keep = {k: float(weights[k]) for _, k in top}
    if not keep:
        return {}
    ledger = DecisionLedger()
    for m in records:
        sc = {k: float(m.signals[k]) for k in keep if k in m.signals and _fin(m.signals[k]) is not None}
        if not sc:
            continue
        ledger.add(Decision(m.rid, m.decided_at, m.resolved_at, sc, m.hyp_side(p) * m.ret, dict(m.tags)))
    base = cfg or CreditConfig()
    need = int(math.ceil(len(keep) / base.alpha)) + 1
    cfg = dataclasses.replace(base, n_perm=max(base.n_perm, need))
    rep = CreditEngine(WeightedSumCombiner(keep), cfg).assess(ledger, now)
    return {c.component: {"mean_credit": c.mean_credit, "lo": c.lo, "hi": c.hi, "verdict": c.verdict.value,
                          "credit_on_wins": c.credit_on_wins, "blame_on_losses": c.blame_on_losses, "share": c.share,
                          "n": c.n} for c in rep.components}


# ==================================================================================================================
# winners: one finding per meaningful winner
# ==================================================================================================================
@dataclass(frozen=True)
class WinnerFinding:
    rid: str
    kind: MoveKind
    learned_at: str
    ret: float
    severity: float
    driver: MoveDriver
    explanation: MoveExplanation
    detection: DetectionMode
    surprise_bits: float | None           # -log2 P(the model gave to a large move): a high value is a win that confirms nothing
    ranked_by: tuple[tuple[str, float], ...]
    contributed: tuple[str, ...]
    irrelevant: tuple[str, ...]
    not_generalised: tuple[str, ...]
    magnitude: Mapping[str, float]
    timing: Mapping[str, float]
    magnitude_class: Predictability
    timing_class: Predictability
    sep_selection: float | None           # separation.decompose: realised move minus the promised move
    tags: Mapping[str, str] = field(default_factory=dict)
    answered: Mapping[str, bool] = field(default_factory=dict)

    def depth(self) -> int:
        return sum(1 for v in self.answered.values() if v)


class WinnerResearch:
    """fit() learns cohort facts from MATURED records only (signal verdicts, magnitude and timing predictability, rank calibration);
    study() then writes one WinnerFinding per winner using those facts. fit() before study() is enforced."""

    def __init__(self, params: ResearchParams | None = None, weights: Mapping[str, float] | None = None, seed: int = 0):
        self.p = params or ResearchParams()
        errs = self.p.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.weights = dict(weights or {})
        self.seed = seed
        self.signal_stats: list[SignalStat] = []
        self.magnitude: PredictabilityStat | None = None
        self.timing: PredictabilityStat | None = None
        self.rank: AucResult | None = None
        self.fitted_through: str | None = None
        self.n_fit = (0, 0)

    def fit(self, records: Sequence[MoveRecord], now) -> "WinnerResearch":
        ok, _ = matured_only(records, now, strict=True)
        p = self.p
        winners = [m for m in ok if m.is_winner(p)]
        controls = [m for m in ok if is_control(m, p)]
        self.signal_stats = study_signals(winners, controls, p, self.weights, +1, self.seed)
        movers = [m for m in ok if m.is_winner(p) or m.is_loser(p)]
        self.magnitude = magnitude_predictability(movers, p, self.seed)
        self.timing = timing_predictability(movers, p, self.seed)
        self.rank = rank_calibration(winners, controls, p, self.seed)
        self.fitted_through = max((m.resolved_at for m in ok), default=None)
        self.n_fit = (len(winners), len(controls))
        return self

    def fitted(self) -> tuple[PredictabilityStat, PredictabilityStat, AucResult]:
        """The three cohort facts `fit` produced; asking before fitting is a programming error."""
        if self.magnitude is None or self.timing is None or self.rank is None:
            raise RuntimeError("WinnerResearch.study() before fit(): cohort facts are needed to say what generalised")
        return self.magnitude, self.timing, self.rank

    def detection_mode(self, m: MoveRecord) -> DetectionMode:
        if m.held:
            if m.prob_move is None:
                return DetectionMode.UNPROVEN
            return DetectionMode.EARNED if m.prob_move >= self.p.prob_hi else DetectionMode.LUCKY
        return DetectionMode.REJECTED if m.considered else DetectionMode.MISSED

    def study(self, m: MoveRecord) -> WinnerFinding:
        mag_stat, tim_stat, _ = self.fitted()
        p = self.p
        m.require_valid()
        if not m.is_winner(p):
            raise ValueError(f"record {m.rid} is not a meaningful winner (ret={m.ret:.4f})")
        vm = verdict_map(self.signal_stats)
        active = [k for k, v in m.signals.items() if _fin(v) is not None and abs(v) >= p.active_z]
        contributed = tuple(sorted(k for k in active if vm.get(k) is SignalVerdict.INFORMATIVE))
        irrelevant = tuple(sorted(k for k in active if vm.get(k) is SignalVerdict.IRRELEVANT))
        notgen = tuple(sorted(k for k in active if vm.get(k) in (SignalVerdict.NOT_GENERALISED, SignalVerdict.MISWEIGHTED)))
        ranked = tuple(sorted(((k, float(self.weights[k]) * float(m.signals[k])) for k in self.weights
                               if k in m.signals and _fin(m.signals[k]) is not None), key=lambda kv: (-abs(kv[1]), kv[0]))[:5])
        expl = explain_move(m, p)
        det = self.detection_mode(m)
        sb = move_surprise(m)
        mag: dict[str, float] = {"realised": abs(m.ret)}
        if m.exp_move is not None:
            mag.update(predicted=m.exp_move, ratio=abs(m.ret) / m.exp_move, error=abs(m.ret) - m.exp_move)
        tim: dict[str, float] = {}
        if m.move_day is not None:
            tim["move_frac"] = m.move_day / m.horizon_days
        if m.move_day is not None and m.exp_day is not None:
            tim["error_frac"] = abs(m.move_day - m.exp_day) / m.horizon_days
            tim["within_tol"] = float(tim["error_frac"] <= p.timing_tol)
        sel = None
        if m.exp_move is not None:
            sel = decompose(to_trade_record(m, p), SeparationParams()).selection
        answered = {
            "why_moved": expl.driver is not MoveDriver.UNEXPLAINED,
            "why_detected": det is not DetectionMode.UNPROVEN,
            "why_ranked": bool(ranked) or m.rank_pct is not None,
            "signals_contributed": bool(self.signal_stats) and any(s.verdict is not SignalVerdict.UNDERPOWERED for s in self.signal_stats),
            "signals_irrelevant": bool(self.signal_stats) and any(s.verdict is not SignalVerdict.UNDERPOWERED for s in self.signal_stats),
            "signals_generalised": bool(self.signal_stats) and any(s.n_groups >= p.min_groups for s in self.signal_stats),
            "magnitude_predictable": "predicted" in mag and mag_stat.verdict is not Predictability.UNDERPOWERED,
            "timing_predictable": "error_frac" in tim and tim_stat.verdict is not Predictability.UNDERPOWERED}
        return WinnerFinding(m.rid, MoveKind.WINNER, m.resolved_at, float(m.ret), self.severity(m), expl.driver, expl, det, sb,
                             ranked, contributed, irrelevant, notgen, mag, tim, mag_stat.verdict, tim_stat.verdict, sel,
                             dict(m.tags), answered)

    def severity(self, m: MoveRecord) -> float:
        """How much a record is worth studying: the size of the move times how surprised the model was."""
        s = move_surprise(m)
        return float(abs(m.ret) * (1.0 + (s if s is not None else 1.0)))

    def study_all(self, records: Sequence[MoveRecord]) -> list[WinnerFinding]:
        return [self.study(m) for m in records if m.is_winner(self.p)]


# ==================================================================================================================
# symmetry: losers are studied at least as broadly and as deeply as winners
# ==================================================================================================================
@dataclass(frozen=True)
class SymmetryReport:
    n_winners: int
    n_losers: int
    studied_winners: int
    studied_losers: int
    coverage_winners: float
    coverage_losers: float
    depth_winners: float
    depth_losers: float
    passed: bool
    failures: tuple[str, ...]


def symmetry_report(winner_findings: Sequence[Any], loser_findings: Sequence[Any], n_winners: int, n_losers: int,
                    min_depth_ratio: float = 1.0) -> SymmetryReport:
    """The contract's test: losses studied at least as hard as winners. Checked as coverage (share of all meaningful winners /
    losers that got a finding) and depth (questions actually answered per finding). Findings need an `answered` mapping."""
    cw = len(winner_findings) / n_winners if n_winners else float("nan")
    cl = len(loser_findings) / n_losers if n_losers else float("nan")
    dw = float(np.mean([sum(1 for v in f.answered.values() if v) for f in winner_findings])) if winner_findings else 0.0
    dl = float(np.mean([sum(1 for v in f.answered.values() if v) for f in loser_findings])) if loser_findings else 0.0
    fails = []
    if n_losers and not loser_findings:
        fails.append("no loss was studied")
    if math.isfinite(cw) and math.isfinite(cl) and cl + 1e-12 < cw:
        fails.append(f"losers covered {cl:.0%} < winners {cw:.0%}")
    if loser_findings and winner_findings and dl + 1e-12 < dw * min_depth_ratio:
        fails.append(f"loser depth {dl:.1f} < winner depth {dw:.1f} x {min_depth_ratio}")
    for f in loser_findings:
        if not getattr(f, "answered", None):
            fails.append(f"loss finding {getattr(f, 'rid', '?')} has no answered questions")
            break
    return SymmetryReport(n_winners, n_losers, len(winner_findings), len(loser_findings), cw, cl, dw, dl, not fails, tuple(fails))


def review_budget(records: Sequence[MoveRecord], budget: int, p: ResearchParams | None = None, loss_bias: float = 1.25,
                  ) -> dict[str, Any]:
    """Pick which meaningful records to study when budget is short. Score = |return| x model surprise; losers are multiplied by
    `loss_bias` (>= 1). After the greedy pick, winners are swapped out for unpicked losers until the share of losers studied is at
    least the share of winners studied, so a budget can never make the study asymmetric against losses."""
    p = p or ResearchParams()
    if loss_bias < 1.0:
        raise ValueError("loss_bias below 1 would study winners harder than losers")
    los = [m for m in records if m.is_loser(p)]
    lset = {m.rid for m in los}
    win = [m for m in records if m.is_winner(p) and m.rid not in lset]        # a record that is both is studied as a loss

    def score(m: MoveRecord) -> float:
        s = move_surprise(m)
        return abs(m.ret) * (1.0 + (1.0 if s is None else s)) * (loss_bias if m.rid in lset else 1.0)

    pool = sorted(win + los, key=lambda m: (-score(m), m.rid))
    picked = pool[:max(0, budget)]
    pw = [m for m in picked if m.rid not in lset]
    pl = [m for m in picked if m.rid in lset]
    rest_l = [m for m in los if m not in pl]
    swaps = 0
    while rest_l and pw and (len(pl) / len(los) if los else 1.0) + 1e-12 < (len(pw) / len(win) if win else 0.0):
        pw.sort(key=lambda m: (score(m), m.rid))
        pw.pop(0)
        pl.append(rest_l.pop(0))
        swaps += 1
    return {"picked": sorted(m.rid for m in pw + pl), "winners": len(pw), "losers": len(pl), "swaps": swaps,
            "winner_share": len(pw) / len(win) if win else float("nan"), "loser_share": len(pl) / len(los) if los else float("nan")}


# ==================================================================================================================
# breakdowns and rendering
# ==================================================================================================================
def breakdown(findings: Sequence[WinnerFinding], key: str) -> dict[str, dict[str, Any]]:
    """Per tag value (era / winner type): how many, mean return, how many were earned vs lucky, and the driver mix."""
    out: dict[str, dict[str, Any]] = {}
    for tag in sorted({f.tags.get(key, "?") for f in findings}):
        sub = [f for f in findings if f.tags.get(key, "?") == tag]
        det = {d.value: sum(1 for f in sub if f.detection is d) for d in DetectionMode}
        drv = {d.value: sum(1 for f in sub if f.driver is d) for d in MoveDriver}
        out[tag] = {"n": len(sub), "mean_ret": float(np.mean([f.ret for f in sub])), "detection": det, "drivers": drv,
                    "lucky_share": det["LUCKY"] / max(1, det["LUCKY"] + det["EARNED"]),
                    "mean_depth": float(np.mean([f.depth() for f in sub]))}
    return out


def validate_findings(findings: Sequence[WinnerFinding], identities: Iterable[str] = ()) -> list[str]:
    """Self-audit: every finding answered something, carries no identity, and the driver matches its explanation."""
    errs = []
    ids = [str(i) for i in identities if str(i)]
    for f in findings:
        if f.driver is not f.explanation.driver:
            errs.append(f"{f.rid}: driver differs from its explanation")
        if not f.answered:
            errs.append(f"{f.rid}: no questions recorded")
        elif set(f.answered) != set(WINNER_QUESTIONS):
            errs.append(f"{f.rid}: question set {sorted(f.answered)} is not the section-5 winner list")
        text = canonical_json(f)
        errs += [f"{f.rid}: identity {i!r} in finding" for i in ids if i in text]
        for k in list(f.tags):
            if str(k).lower() in FORBIDDEN_KEYS:
                errs.append(f"{f.rid}: identity key {k!r} in tags")
    return errs


def render_winner_summary(rep: "WinnerReport") -> str:
    lines = [f"winner research: {len(rep.findings)} winners, {rep.n_controls} controls, {rep.pending} pending  "
             "[IMPLEMENTED - NOT VALIDATED]"]
    for s in rep.signal_stats[:10]:
        lines.append(f"  {s.signal:<14}{s.verdict.value:<16} auc {s.auc:.3f} [{s.lo:.3f},{s.hi:.3f}] q {s.q:.3f} groups {s.n_groups}")
    lines.append(f"  magnitude: {rep.magnitude.verdict.value}  timing: {rep.timing.verdict.value}")
    det = {d.value: sum(1 for f in rep.findings if f.detection is d) for d in DetectionMode}
    lines.append("  detection: " + ", ".join(f"{k}={v}" for k, v in det.items() if v))
    return NL.join(lines)


# ==================================================================================================================
# MATURED_RESEARCH_STATE: findings reach a decision only through MaturedRecord.gate(now)
# ==================================================================================================================
def to_matured(finding: Any, code_hash: str | None = None, run_id: str = "", seed: int | None = None,
               identities: Iterable[str] = (), created_real: str = "") -> MaturedRecord:
    """Wrap a finding as a research-world fact whose maturity is the date its outcome resolved. Refuses a payload that contains
    any of the given identities (tickers, names): the research world may study them, the trader may never learn them."""
    payload = json_safe(finding)
    text = canonical_json(payload)
    hits = [str(i) for i in identities if str(i) and str(i) in text]
    if hits:
        raise FirewallBreach(f"finding {finding.rid} carries identities {hits}")
    for k in _walk_keys(payload):
        if str(k).lower() in FORBIDDEN_KEYS:
            raise FirewallBreach(f"finding {finding.rid} carries identity key {k!r}")
    prov = Provenance(created_real=created_real or finding.learned_at, learned_at=finding.learned_at,
                      code_hash=code_hash or cached_code_hash(), run_id=run_id, seed=seed,
                      outcomes_seen_through=finding.learned_at)
    return MaturedRecord(record_id=f"WL-{finding.rid}", matured_at=finding.learned_at, payload=payload, provenance=prov,
                         namespace=Namespace.MATURED_RESEARCH)


def json_safe(obj: Any) -> Any:
    import json
    return json.loads(canonical_json(obj))


def _walk_keys(obj: Any) -> Iterable[str]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_keys(v)


def release(records: Sequence[MaturedRecord], now, replaying_years: Iterable[int] = ()) -> list[Mapping[str, Any]]:
    """The only exit from the research world. A record is refused (FirewallBreach) if it matured on or after `now`, could not have
    existed at `now`, or was filed under a real year that is currently being replayed in disguise (the same-year rerun leak)."""
    banned = {int(y) for y in replaying_years}
    out = []
    for r in records:
        if as_date(r.matured_at).year in banned:
            raise FirewallBreach(f"record {r.record_id} was filed under a year that is being replayed: same-year rerun leak")
        out.append(r.gate(now))
    return out


# ==================================================================================================================
# the single public entry
# ==================================================================================================================
@dataclass
class WinnerState:
    """Everything the winner study accumulates across days. Records are exceptions and controls only."""
    params: ResearchParams = field(default_factory=ResearchParams)
    weights: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0
    records: dict[str, MoveRecord] = field(default_factory=dict)
    findings: dict[str, WinnerFinding] = field(default_factory=dict)

    def add(self, recs: Iterable[MoveRecord]) -> int:
        n = 0
        for m in recs:
            m.require_valid()
            if m.rid not in self.records:
                self.records[m.rid] = m
                n += 1
        return n


@dataclass(frozen=True)
class WinnerReport:
    now: str
    findings: tuple[WinnerFinding, ...]
    signal_stats: tuple[SignalStat, ...]
    magnitude: PredictabilityStat
    timing: PredictabilityStat
    rank: AucResult
    n_controls: int
    pending: int
    matured: tuple[MaturedRecord, ...]
    params_hash: str


def step(state: WinnerState, records: Iterable[MoveRecord], now, identities: Iterable[str] = ()) -> WinnerReport:
    """One research step: absorb new matured records, refit the cohort facts on everything matured before `now`, write a finding for
    every winner not yet studied, wrap each as a MaturedRecord. Records that have not matured are counted, not used."""
    incoming = list(records)
    ok, pending = matured_only(incoming, now)
    state.add(ok)
    known, _ = matured_only(state.records.values(), now, strict=True)
    wr = WinnerResearch(state.params, state.weights, state.seed).fit(known, now)
    new = [m for m in known if m.is_winner(state.params) and m.rid not in state.findings]
    for m in new:
        state.findings[m.rid] = wr.study(m)
    fresh = [state.findings[m.rid] for m in new]
    code = cached_code_hash() if fresh else ""
    matured = tuple(to_matured(f, code, seed=state.seed, identities=identities) for f in fresh)
    controls = sum(1 for m in known if is_control(m, state.params))
    mag_stat, tim_stat, rank_stat = wr.fitted()
    return WinnerReport(str(as_date(now)), tuple(fresh), tuple(wr.signal_stats), mag_stat, tim_stat, rank_stat, controls, pending,
                        matured, state.params.hash())


# ==================================================================================================================
# diagnostics that decide whether the studies above can be believed
# ==================================================================================================================
def control_balance(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], smd_max: float = 0.25) -> dict[str, Any]:
    """Are the controls a fair comparison? Standardised mean difference of every market-context value and of prior return / expected
    volatility between cases and controls, plus the share of cases whose decision week has any control. A signal AUC against controls
    drawn from other weeks or other conditions measures the condition, not the signal."""
    if not cases or not controls:
        return {"balanced": False, "reason": "no cases or no controls", "smd": {}, "week_overlap": 0.0}

    def col(recs, name):
        if name.startswith("ctx:"):
            k = name[4:]
            return np.array([np.nan if _fin(m.context.get(k)) is None else float(m.context[k]) for m in recs], dtype=float)
        return np.array([np.nan if getattr(m, name) is None else float(getattr(m, name)) for m in recs], dtype=float)

    names = sorted({"ctx:" + k for m in list(cases) + list(controls) for k in m.context}) + ["prior_ret", "exp_vol"]
    smd = {}
    for n in names:
        a, b = col(cases, n), col(controls, n)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            continue
        sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        smd[n] = float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0
    case_weeks = week_of([m.decided_at for m in cases])
    ctrl_weeks = set(week_of([m.decided_at for m in controls]).tolist())
    overlap = float(np.mean([int(w) in ctrl_weeks for w in case_weeks]))
    bad = sorted(k for k, v in smd.items() if abs(v) > smd_max)
    return {"balanced": not bad and overlap >= 0.9, "smd": smd, "imbalanced": bad, "week_overlap": overlap,
            "case_weeks": len(set(case_weeks.tolist())), "control_weeks": len(ctrl_weeks)}


def split_replication(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], p: ResearchParams | None = None,
                      weights: Mapping[str, float] | None = None, cohort_dir: int = 1, seed: int = 0) -> dict[str, Any]:
    """Do the signal effects replicate? Fit the study separately on the earlier and the later half of the cases (controls split at the
    same date). Reports each signal's edge in both halves and the share of early-significant signals that keep their sign and
    significance in the later half: the time-generalisation question, asked directly."""
    p = p or ResearchParams()
    if len(cases) < 2 * p.min_n or len(controls) < 2 * p.min_n:
        return {"n_signals": 0, "n_early_significant": 0, "replication_rate": float("nan"), "signals": {},
                "reason": "too few records to split"}
    cut = float(np.median([as_date(m.decided_at).toordinal() for m in cases]))

    def early(recs):
        return [m for m in recs if as_date(m.decided_at).toordinal() <= cut]

    def late(recs):
        return [m for m in recs if as_date(m.decided_at).toordinal() > cut]

    a = {s.signal: s for s in study_signals(early(cases), early(controls), p, weights, cohort_dir, seed)}
    b = {s.signal: s for s in study_signals(late(cases), late(controls), p, weights, cohort_dir, seed + 1)}
    out = {}
    for name in sorted(set(a) & set(b)):
        sa, sb = a[name], b[name]
        sig_a = sa.q <= p.alpha and abs(sa.edge) >= p.min_edge
        sig_b = sb.q <= p.alpha and abs(sb.edge) >= p.min_edge
        out[name] = {"edge_early": sa.edge, "edge_late": sb.edge, "sig_early": bool(sig_a), "sig_late": bool(sig_b),
                     "same_sign": _side(sa.edge) == _side(sb.edge),
                     "replicates": bool(sig_a and sig_b and _side(sa.edge) == _side(sb.edge))}
    first = [v for v in out.values() if v["sig_early"]]
    return {"n_signals": len(out), "n_early_significant": len(first),
            "replication_rate": float(np.mean([v["replicates"] for v in first])) if first else float("nan"), "signals": out}


def tiered_signal_study(winners: Sequence[MoveRecord], controls: Sequence[MoveRecord], p: ResearchParams | None = None,
                        weights: Mapping[str, float] | None = None, seed: int = 0) -> dict[str, Any]:
    """Regular winners against extreme winners (|return| >= extreme_thr), each against the same controls. A signal informative only
    for the extreme tier is a tail signal; one informative only for regular winners does not reach the tail. Extremes are few, so
    an UNDERPOWERED tier is reported as such, never merged into the regular one."""
    p = p or ResearchParams()
    ext = [m for m in winners if m.ret >= p.extreme_thr]
    reg = [m for m in winners if m.ret < p.extreme_thr]
    res = {"regular": study_signals(reg, controls, p, weights, +1, seed), "extreme": study_signals(ext, controls, p, weights, +1, seed + 7)}
    vr = {s.signal: s.verdict for s in res["regular"]}
    ve = {s.signal: s.verdict for s in res["extreme"]}
    tail_only = sorted(k for k in ve if ve[k] is SignalVerdict.INFORMATIVE and vr.get(k) is not SignalVerdict.INFORMATIVE)
    body_only = sorted(k for k in vr if vr[k] is SignalVerdict.INFORMATIVE and ve.get(k) is not SignalVerdict.INFORMATIVE)
    return {"n_regular": len(reg), "n_extreme": len(ext), "stats": res, "tail_only": tail_only, "body_only": body_only}


def unconditional_magnitude(records: Sequence[MoveRecord], p: ResearchParams | None = None, seed: int = 0) -> PredictabilityStat:
    """Magnitude predictability over EVERY record with an expected move - movers and non-movers alike. Studying only the names that
    moved restricts the range of the outcome and understates how well size is predicted; conditioning on the outcome is the winner's
    curse. This is the unconditioned companion of magnitude_predictability, and the two should be read together."""
    p = p or ResearchParams()
    st = magnitude_predictability(records, p, seed)
    return PredictabilityStat("magnitude_all", st.n, st.rho, st.p, st.skill, st.verdict, st.detail)


def winner_type(f: WinnerFinding) -> str:
    """One label per winner: how it was won x why it moved. EARNED_* is a win the volatility model saw; LUCKY_* one it did not."""
    return f"{f.detection.value}_{f.driver.value}"


def type_table(findings: Sequence[WinnerFinding]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for t in sorted({winner_type(f) for f in findings}):
        sub = [f for f in findings if winner_type(f) == t]
        out[t] = {"n": len(sub), "share": len(sub) / len(findings), "mean_ret": float(np.mean([f.ret for f in sub])),
                  "mean_severity": float(np.mean([f.severity for f in sub]))}
    return out


def audit_no_future(findings: Sequence[Any], now) -> list[str]:
    """Every finding must have matured strictly before `now`; returns the violations (empty when clean)."""
    bad = []
    for f in findings:
        try:
            require_past(f.learned_at, now, f"finding {f.rid}")
        except FirewallBreach as e:
            bad.append(str(e))
    return bad


def manifest(params: ResearchParams, records: Iterable[MoveRecord], findings: Iterable[Any], seed: int, now) -> dict[str, Any]:
    """Reproducibility stamp: hashes of parameters, the exact record set and the finding set, the code hash and the seed. Two runs
    with the same manifest read the same inputs and, being deterministic, must write the same findings."""
    recs = sorted(m.rid for m in records)
    fnd = sorted(f.rid for f in findings)
    return {"params": params.hash(), "records": stable_hash(recs), "findings": stable_hash(fnd), "n_records": len(recs),
            "n_findings": len(fnd), "code": cached_code_hash(), "seed": seed, "now": str(as_date(now))}


# ==================================================================================================================
# more of the winner questions, asked of losers with the very same code (symmetry by construction)
# ==================================================================================================================
class PathShape(_StrEnum):
    JUMP_AND_HOLD = "JUMP_AND_HOLD"       # most of the move arrived in one day and stayed
    SPIKE_AND_FADE = "SPIKE_AND_FADE"     # an excursion much larger than the end result: the exit question
    DRIFT = "DRIFT"                       # no single day carried it
    V_SHAPE = "V_SHAPE"                   # went the wrong way first, then finished in the direction of the move
    UNKNOWN = "UNKNOWN"                   # path columns missing


def path_shape(m: MoveRecord, p: ResearchParams | None = None) -> PathShape:
    """Shape of the path to the end result, from the excursions and the largest day. Defined on the direction of the MOVE (up for
    a winner, down for a loser) so that winners and losers are classified by one rule. The path decides whether an exit could have
    kept the move (a spike that faded could, a jump that held could not)."""
    p = p or ResearchParams()
    if m.peak_ret is None or m.trough_ret is None:
        return PathShape.UNKNOWN
    sgn = 1.0 if m.ret >= 0 else -1.0
    end = abs(m.ret)
    favourable = m.peak_ret if sgn > 0 else -m.trough_ret
    adverse = -m.trough_ret if sgn > 0 else m.peak_ret
    if favourable >= 1.6 * max(end, 1e-9) and favourable - end >= 0.5 * p.win_thr:
        return PathShape.SPIKE_AND_FADE
    if adverse >= 0.5 * end and adverse >= 0.5 * p.win_thr:
        return PathShape.V_SHAPE
    jump = _fin(m.max_day_ret)
    if jump is not None and sgn * jump >= p.event_share * end:
        return PathShape.JUMP_AND_HOLD
    return PathShape.DRIFT


def path_table(records: Sequence[MoveRecord], p: ResearchParams | None = None) -> dict[str, dict[str, float]]:
    """Path-shape mix for winners and for losers side by side: if losers spike-and-fade more often than winners, exits are the
    losers' problem; if winners jump-and-hold, no exit changed them. One function for both, by construction."""
    p = p or ResearchParams()
    out: dict[str, dict[str, float]] = {}
    for name, sel in (("winners", [m for m in records if m.is_winner(p)]), ("losers", [m for m in records if m.is_loser(p)])):
        shapes = [path_shape(m, p) for m in sel]
        n = len(shapes)
        out[name] = {"n": float(n), **{s.value: (sum(1 for x in shapes if x is s) / n if n else float("nan")) for s in PathShape}}
    return out


def speed_profile(records: Sequence[MoveRecord], p: ResearchParams | None = None) -> dict[str, Any]:
    """When in the horizon do the big days land for winners vs losers? Two-sample Kolmogorov-Smirnov on move_day / horizon. Losses
    that come faster than gains leave less time to react; the exit design should know."""
    p = p or ResearchParams()
    def frac(sel):
        return np.array([m.move_day / m.horizon_days for m in sel if m.move_day is not None], dtype=float)
    w = frac([m for m in records if m.is_winner(p) and not m.is_loser(p)])
    l = frac([m for m in records if m.is_loser(p)])
    if len(w) < p.min_group_n or len(l) < p.min_group_n:
        return {"n_winners": len(w), "n_losers": len(l), "verdict": Predictability.UNDERPOWERED.value}
    ks = stats.ks_2samp(w, l)
    return {"n_winners": len(w), "n_losers": len(l), "median_winners": float(np.median(w)), "median_losers": float(np.median(l)),
            "ks": float(ks.statistic), "p": float(ks.pvalue),
            "verdict": "LOSSES_FASTER" if ks.pvalue <= p.alpha and np.median(l) < np.median(w) else
            "GAINS_FASTER" if ks.pvalue <= p.alpha else "NO_DIFFERENCE"}


def signal_interactions(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], stats_: Sequence[SignalStat],
                        p: ResearchParams | None = None, top: int = 6) -> list[dict[str, Any]]:
    """Do two informative signals together do more than either alone? For the top informative pairs, the lift of the case rate when
    BOTH are active over the better single lift, Fisher-tested on the both-active cell against controls, BH-controlled across pairs.
    A pair that only ever fires together is reported as such (redundant), not as an interaction."""
    p = p or ResearchParams()
    inf = [s for s in stats_ if s.verdict is SignalVerdict.INFORMATIVE][:top]
    if len(inf) < 2 or not cases or not controls:
        return []
    def act(recs, s):
        sg = 1 if s.auc > 0.5 else -1
        return np.array([(_fin(m.signals.get(s.signal)) is not None and sg * m.signals[s.signal] >= p.active_z) for m in recs])
    rate = lambda a: (a.sum() + 0.5) / (len(a) + 1.0)
    rows: list[dict[str, Any]] = []
    for i in range(len(inf)):
        for j in range(i + 1, len(inf)):
            ac, bc, an, bn = act(cases, inf[i]), act(cases, inf[j]), act(controls, inf[i]), act(controls, inf[j])
            both_c, both_n = ac & bc, an & bn
            lift_both = rate(both_c) / rate(both_n)
            lift_a, lift_b = rate(ac) / rate(an), rate(bc) / rate(bn)
            table = [[int(both_c.sum()), int(len(both_c) - both_c.sum())], [int(both_n.sum()), int(len(both_n) - both_n.sum())]]
            pv = float(stats.fisher_exact(table, alternative="greater")[1])
            corr = float(np.corrcoef(np.concatenate([ac, an]).astype(float), np.concatenate([bc, bn]).astype(float))[0, 1]) \
                if ac.std() > 0 and bc.std() > 0 else float("nan")
            rows.append({"a": inf[i].signal, "b": inf[j].signal, "lift_both": float(lift_both), "lift_a": float(lift_a),
                         "lift_b": float(lift_b), "synergy": float(lift_both / max(lift_a, lift_b)), "p": pv, "activation_corr": corr,
                         "n_both": int(both_c.sum())})
    q = bh_qvalues(np.array([r["p"] for r in rows]))
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
        r["kind"] = ("REDUNDANT" if math.isfinite(r["activation_corr"]) and r["activation_corr"] > 0.8 else
                     "SYNERGY" if qq <= p.alpha and r["synergy"] >= 1.25 and r["n_both"] >= 5 else "NONE")
    return sorted(rows, key=lambda r: (-r["synergy"], r["a"], r["b"]))


def regime_stability(cases: Sequence[MoveRecord], controls: Sequence[MoveRecord], stats_: Sequence[SignalStat],
                     context: str, p: ResearchParams | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Does a signal's edge survive every market condition? Cases and controls are split by tercile of one context value (the
    tercile edges come from the controls only) and each informative signal's AUC is recomputed inside each tercile. A signal that
    flips sign, or vanishes, in a tercile is regime-bound and is reported as such (generalising across regimes, contract section 0)."""
    p = p or ResearchParams()
    cv = np.array([m.context[context] for m in controls if _fin(m.context.get(context)) is not None], dtype=float)
    if len(cv) < 3 * p.min_group_n:
        return []
    edges = np.quantile(cv, [1 / 3, 2 / 3])
    def bucket(m):
        v = _fin(m.context.get(context))
        return None if v is None else int(np.searchsorted(edges, v))
    rows = []
    for s in stats_:
        if s.verdict not in (SignalVerdict.INFORMATIVE, SignalVerdict.NOT_GENERALISED):
            continue
        per = {}
        for b in (0, 1, 2):
            c = [m for m in cases if bucket(m) == b and _fin(m.signals.get(s.signal)) is not None]
            n = [m for m in controls if bucket(m) == b and _fin(m.signals.get(s.signal)) is not None]
            if len(c) < p.min_group_n or len(n) < p.min_group_n:
                continue
            r = stratified_auc(np.array([m.signals[s.signal] for m in c], dtype=float), week_of([m.decided_at for m in c]),
                               np.array([m.signals[s.signal] for m in n], dtype=float), week_of([m.decided_at for m in n]),
                               50, 10 ** 9, seed)
            if r.pairs > 0:
                per[b] = r.auc
        held = [b for b, v in per.items() if _side(v - 0.5) == _side(s.auc - 0.5) and abs(v - 0.5) >= max(p.min_edge, 0.5 * abs(s.auc - 0.5))]
        rows.append({"signal": s.signal, "context": context, "auc_by_tercile": per, "n_terciles": len(per),
                     "regime_bound": bool(len(per) >= 2 and len(held) < len(per)),
                     "flips": bool(any(_side(v - 0.5) == -_side(s.auc - 0.5) for v in per.values()))})
    return rows


def probability_calibration(records: Sequence[MoveRecord], p: ResearchParams | None = None, bins: int = 5) -> dict[str, Any]:
    """Is prob_move honest? Records are binned by the volatility model's probability and the realised rate of a meaningful move
    (|ret| >= win_thr) is compared with the mean probability in the bin; also the Brier score against the base rate. This is the
    'why did we detect it' question's quality control: a detector that is confidently wrong explains nothing."""
    p = p or ResearchParams()
    rows = [(m.prob_move, float(abs(m.ret) >= p.win_thr)) for m in records if m.prob_move is not None]
    if len(rows) < bins * 5:
        return {"n": len(rows), "verdict": Predictability.UNDERPOWERED.value}
    pr, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    order = np.argsort(pr, kind="stable")
    table = []
    for chunk in np.array_split(order, bins):
        table.append({"mean_prob": float(pr[chunk].mean()), "realised": float(y[chunk].mean()), "n": int(len(chunk))})
    brier = float(np.mean((pr - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    gap = float(np.mean([abs(t["mean_prob"] - t["realised"]) for t in table]))
    slope = float(np.polyfit(pr, y, 1)[0]) if np.ptp(pr) > 0 else 0.0
    return {"n": len(rows), "table": table, "brier": brier, "brier_base": base, "brier_skill": 1.0 - brier / base if base > 0 else None,
            "mean_gap": gap, "slope": slope, "verdict": "CALIBRATED" if gap <= 0.05 and 0.6 <= slope <= 1.4 else
            "OVERCONFIDENT" if slope < 0.6 else "MISCALIBRATED"}
