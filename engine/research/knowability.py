"""Knowability engine (contract C66 sections 7 and 8; canons C62/C66). RESEARCH / AUDIT WORLD ONLY (MATURED_RESEARCH_STATE).

For each major historical move it answers "could the system have known?" by rebuilding the information state at the decision
boundary, tagging every information item KNOWN_BEFORE / ONLY_AFTER / SIMULTANEOUS / UNCERTAIN / UNAVAILABLE by timestamp and
provenance, and classifying the move PREDICTABLE / POTENTIALLY / WEAKLY / UNKNOWN / EXTERNALLY_CAUSED /
INFORMATIONALLY_UNAVAILABLE / DATA_FAILURE. Unknown stays unknown: a missing timestamp or provenance can only lower a
classification, never raise it, and a cause discovered after the fact is never turned into predictive knowledge.
The result is a hindsight label. It is a MaturedRecord: it may reach the blind trader only through gate(now) and the curator,
and `refuse_hindsight_columns` / `assert_not_trader_bound` fail closed if it is offered as a feature or handed to the trader.

Builds on: engine.pit.Calendar (sessions, publication lags), engine.edgar.classify (filing kinds), engine.learning.failure
(ramp, noisy_or), engine.research.core (Availability, Knowability, MaturedRecord, Namespace).
Public entry: step(state, now, inputs) -> (state, StepReport).
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import current_code_hash
from engine.learning.failure import noisy_or, ramp
from engine.pit import Calendar
from engine.research.core import (Availability, FirewallBreach, Knowability, MaturedRecord, Namespace, Provenance,
                                  _StrEnum, as_date, stable_hash)


class KnowabilityError(ValueError):
    """Malformed knowability input (bad timestamps, empty windows). Never swallowed: a broken audit is not an answer."""


class InfoKind(_StrEnum):
    FEATURE = "FEATURE"
    PATTERN = "PATTERN"
    MEMORY = "MEMORY"
    MACRO = "MACRO"
    MARKET_STATE = "MARKET_STATE"
    EVENT = "EVENT"
    FILING = "FILING"
    INSIDER = "INSIDER"
    PRICE = "PRICE"
    VOLUME = "VOLUME"
    TECHNICAL = "TECHNICAL"
    CROSS_SECTION = "CROSS_SECTION"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    NEWS_EXTERNAL = "NEWS_EXTERNAL"


CHANNELS = ("FEATURES", "PATTERNS", "MEMORY", "MACRO", "MARKET_STATE", "EVENT", "PRICE", "VOLUME", "TECHNICAL",
            "CROSS_SECTION")                                   # section 8: the ten things compared with what became known later
PREC_INTRADAY, PREC_DATE, PREC_UNKNOWN = "INTRADAY", "DATE", "UNKNOWN"
PROV_RECORDED, PROV_INFERRED, PROV_RECONSTRUCTED, PROV_UNKNOWN = "recorded", "inferred_lag", "reconstructed", "unknown"
PROVENANCES = (PROV_RECORDED, PROV_INFERRED, PROV_RECONSTRUCTED, PROV_UNKNOWN)
KIND_STRENGTH = {"EARN": 0.7, "BANKRUPTCY": 0.95, "ACQ_DONE": 0.8, "RESTATEMENT": 0.85, "DELIST_NOTICE": 0.8,
                 "AUDITOR_CHANGE": 0.5, "AGREEMENT": 0.55, "OFFERING": 0.5, "SHELF": 0.2, "ACTIVIST": 0.6,
                 "ACTIVIST_AMEND": 0.3, "LATE_FILING": 0.45, "PERIODIC": 0.15, "UNREG_SALE": 0.35}
FILING_KIND_TO_INFO = {k: InfoKind.FILING for k in KIND_STRENGTH}


def to_ny(x) -> tuple[pd.Timestamp | None, str]:
    """Any timestamp-like -> (naive New York wall time, precision). A date-only string or a naive midnight is DATE
    precision (cannot be ordered inside its own day); tz-aware values are converted to New York time."""
    if x is None:
        return None, PREC_UNKNOWN
    try:
        t = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None, PREC_UNKNOWN
    if pd.isna(t):
        return None, PREC_UNKNOWN
    if t.tzinfo is not None:
        return t.tz_convert("America/New_York").tz_localize(None), PREC_INTRADAY
    if (isinstance(x, str) and len(x.strip()) <= 10) or t == t.normalize():
        return t.normalize(), PREC_DATE
    return t, PREC_INTRADAY


def _sessions_between(a: pd.Timestamp, b: pd.Timestamp, calendar: Calendar | None) -> int:
    """Signed number of sessions from date a to date b (b after a -> positive)."""
    hol = calendar._hol if calendar is not None else np.array([], dtype="datetime64[D]")
    da, db = np.datetime64(a.normalize().date()), np.datetime64(b.normalize().date())
    if db >= da:
        return int(np.busday_count(da, db, holidays=hol))
    return -int(np.busday_count(db, da, holidays=hol))


# ==================================================================================================================
# information items and the decision boundary
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class InfoItem:
    """One piece of information with the two clocks that matter: when the thing happened (`effective_at`) and when the world
    could first see it (`published_at`). `provenance` says how the publication time is known. `scheduled` marks an
    announcement of a FUTURE event (an earnings date): it is legitimately published before it is effective."""
    item_id: str
    kind: InfoKind
    subject: str                                  # ticker, "MARKET", or a sector name (research side may hold real names)
    effective_at: str | None
    published_at: str | None
    source: str
    provenance: str = PROV_RECORDED
    exists: bool = True                           # False = the auditor established that nothing of this sort existed
    scheduled: bool = False
    strength: float = 0.5                         # how strongly this item would explain/anticipate a move, in [0,1]
    direction: int = 0                            # -1 / 0 / +1: sign of the move it points to (0 = magnitude only)
    detail: str = ""

    def check(self) -> list[str]:
        errs = []
        if not self.item_id:
            errs.append("item without an id")
        if self.provenance not in PROVENANCES:
            errs.append(f"{self.item_id}: unknown provenance {self.provenance!r}")
        if not 0.0 <= float(self.strength) <= 1.0:
            errs.append(f"{self.item_id}: strength {self.strength} outside [0,1]")
        if self.direction not in (-1, 0, 1):
            errs.append(f"{self.item_id}: direction must be -1, 0 or 1")
        if self.exists and self.published_at is None and self.provenance != PROV_INFERRED and self.provenance != PROV_UNKNOWN:
            errs.append(f"{self.item_id}: an item that exists needs a publication time or an honest 'unknown' provenance")
        return errs


@dataclasses.dataclass(frozen=True)
class DecisionBoundary:
    """The moment the system decides (the close of `decision_date`) and the moment the move can start (the next session's
    open). Information published inside [decision - tolerance, fill open] is SIMULTANEOUS: it arrived around the boundary
    and could not be used to decide, but it also is not hindsight."""
    decision_date: str
    fill_date: str
    decision_time: str = "16:00"
    fill_time: str = "09:30"
    tolerance_minutes: int = 30

    def check(self) -> list[str]:
        errs = []
        if as_date(self.fill_date) <= as_date(self.decision_date):
            errs.append("fill date must be strictly after the decision date (decisions at a close fill at the next open)")
        if self.tolerance_minutes < 0:
            errs.append("negative tolerance")
        return errs

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.decision_date} {self.decision_time}")

    @property
    def fill_open(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.fill_date} {self.fill_time}")

    @staticmethod
    def make(decision_date, calendar: Calendar | None = None, tolerance_minutes: int = 30) -> "DecisionBoundary":
        d = pd.Timestamp(as_date(decision_date))
        nxt = (calendar or Calendar()).strict_next([d])[0]
        return DecisionBoundary(str(d.date()), str(pd.Timestamp(nxt).date()), tolerance_minutes=tolerance_minutes)


@dataclasses.dataclass(frozen=True)
class LagPolicy:
    """Publication lags in sessions per source, for items that only record when they were EFFECTIVE (macro releases,
    fundamentals). An inferred publication is treated as UNCERTAIN whenever it lands within `margin_sessions` of the
    boundary, because a lag estimated on average says nothing about that one record."""
    lags: tuple[tuple[str, int], ...] = ()
    default: int = 1
    margin_sessions: int = 1

    def lag_for(self, source: str) -> int:
        return dict(self.lags).get(source, self.default)

    def check(self) -> list[str]:
        errs = [f"lag for {s} is negative" for s, v in self.lags if v < 0]
        if self.default < 0 or self.margin_sessions < 0:
            errs.append("default lag and margin must be >= 0")
        return errs


@dataclasses.dataclass(frozen=True)
class AvailabilityJudgement:
    item_id: str
    kind: InfoKind
    availability: Availability
    reason: str
    margin_minutes: float | None = None           # publication minus decision cutoff (negative = before)
    inferred: bool = False                        # publication time was derived from a lag, not recorded

    def usable_at_decision(self) -> bool:
        return self.availability == Availability.KNOWN_BEFORE_EVENT


def judge_item(item: InfoItem, b: DecisionBoundary, calendar: Calendar | None = None,
               lag: LagPolicy = LagPolicy()) -> AvailabilityJudgement:
    """Timestamp + provenance -> Availability. Every branch that cannot establish order returns UNCERTAIN, never KNOWN_BEFORE."""
    def out(av, why, margin=None, inferred=False):
        return AvailabilityJudgement(item.item_id, item.kind, av, why, margin, inferred)

    if not item.exists:
        return out(Availability.UNAVAILABLE, "auditor established that nothing of this kind existed")
    if item.provenance == PROV_UNKNOWN:
        return out(Availability.UNCERTAIN, "provenance unknown: publication time cannot be trusted")
    pub, prec = to_ny(item.published_at)
    inferred = False
    if pub is None:
        eff, _ = to_ny(item.effective_at)
        if item.provenance != PROV_INFERRED or eff is None:
            return out(Availability.UNCERTAIN, "no publication timestamp")
        cal = calendar or Calendar()
        pub = pd.Timestamp(cal.shift([eff], lag.lag_for(item.source))[0]).normalize()
        prec, inferred = PREC_DATE, True
    eff, _ = to_ny(item.effective_at)
    if eff is not None and pub < eff.normalize() - pd.Timedelta(days=1) and not item.scheduled:
        return out(Availability.UNCERTAIN, "published before it happened without being a scheduled announcement", None, inferred)
    if prec == PREC_DATE:
        d, dd, fd = pub.normalize(), pd.Timestamp(b.decision_date), pd.Timestamp(b.fill_date)
        margin = float((d - dd).days * 1440)
        if inferred:
            gap_before = _sessions_between(d, dd, calendar)
            gap_after = _sessions_between(fd, d, calendar)
            if gap_before < lag.margin_sessions and gap_after < lag.margin_sessions:
                return out(Availability.UNCERTAIN, "inferred publication lands within the lag margin of the boundary", margin, True)
        if d < dd:
            return out(Availability.KNOWN_BEFORE_EVENT, "published on an earlier date", margin, inferred)
        if d == dd:
            return out(Availability.SIMULTANEOUS, "date-only stamp on the decision day cannot be ordered against the close", margin, inferred)
        if d == fd:
            return out(Availability.UNCERTAIN, "date-only stamp on the fill day cannot be ordered against the open", margin, inferred)
        return out(Availability.KNOWN_ONLY_AFTER_EVENT, "published after the fill day", margin, inferred)
    margin = (pub - b.cutoff).total_seconds() / 60.0
    if margin <= -b.tolerance_minutes:
        return out(Availability.KNOWN_BEFORE_EVENT, "published before the decision cutoff", margin)
    if pub <= b.fill_open:
        return out(Availability.SIMULTANEOUS, "published around the boundary (after the tolerance, before the fill open)", margin)
    return out(Availability.KNOWN_ONLY_AFTER_EVENT, "published after the fill open", margin)


def judge_all(items: Iterable[InfoItem], b: DecisionBoundary, calendar: Calendar | None = None,
              lag: LagPolicy = LagPolicy()) -> dict[str, AvailabilityJudgement]:
    """Judge a batch. Bad items raise; duplicate ids raise (two clocks for one item would let the optimistic one win)."""
    berrs = b.check() + lag.check()
    if berrs:
        raise KnowabilityError("; ".join(berrs))
    res: dict[str, AvailabilityJudgement] = {}
    for it in items:
        errs = it.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        if it.item_id in res:
            raise KnowabilityError(f"duplicate information item {it.item_id}")
        res[it.item_id] = judge_item(it, b, calendar, lag)
    return res


def availability_counts(judgements: Iterable[AvailabilityJudgement]) -> dict[str, int]:
    out = {a.value: 0 for a in Availability}
    for j in judgements:
        out[j.availability.value] += 1
    return out


@dataclasses.dataclass(frozen=True)
class SourceProbe:
    """Result of asking one data source "did anything precursor-like exist before the event?". `covers` False means the source
    does not span the window at all, so its silence proves nothing."""
    source: str
    covers: bool
    precursors: tuple[str, ...] = ()              # item ids found before the boundary


def judge_precursors(cause: InfoItem, probes: Sequence[SourceProbe], b: DecisionBoundary, calendar: Calendar | None = None,
                     lag: LagPolicy = LagPolicy()) -> tuple[Availability, str]:
    """A cause found only afterwards is UNAVAILABLE (nothing legitimate existed) only when every source that could have
    revealed it covered the window and none held a precursor. One uncovered source turns silence into UNCERTAIN."""
    own = judge_item(cause, b, calendar, lag)
    if own.availability in (Availability.KNOWN_BEFORE_EVENT,):
        return own.availability, "the cause itself was public before the boundary"
    if not probes:
        return Availability.UNCERTAIN, "no source was probed for precursors"
    blind = [p.source for p in probes if not p.covers]
    if blind:
        return Availability.UNCERTAIN, f"sources not covering the window: {sorted(blind)}"
    found = sorted({i for p in probes for i in p.precursors})
    if found:
        return Availability.KNOWN_BEFORE_EVENT, f"precursors existed before the boundary: {found[:5]}"
    if own.availability == Availability.UNCERTAIN:
        return Availability.UNCERTAIN, "no precursors, but the cause's own timestamp is uncertain"
    return Availability.UNAVAILABLE, "every covering source was silent before the boundary"


# ==================================================================================================================
# adapters: repo data -> InfoItem (point-in-time timestamps preserved, never re-dated)
# ==================================================================================================================
def items_from_edgar_events(events: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """engine.edgar events.parquet rows (ticker, form, accepted UTC, kind) inside `window` (inclusive real dates). The
    accepted timestamp is the SEC acceptance moment, the earliest the world could see it: provenance is 'recorded'."""
    if events is None or len(events) == 0:
        return []
    need = {"ticker", "accepted", "kind"}
    if not need <= set(events.columns):
        raise KnowabilityError(f"events frame lacks {sorted(need - set(events.columns))}")
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1]) + pd.Timedelta(days=1)
    out = []
    sub = events[events["ticker"] == ticker]
    for row in sub.itertuples(index=False):
        t, prec = to_ny(getattr(row, "accepted"))
        if t is None or not (lo <= t.normalize() < hi):
            continue
        kind = str(getattr(row, "kind"))
        out.append(InfoItem(f"EDGAR-{ticker}-{kind}-{t.strftime('%Y%m%d%H%M%S')}", FILING_KIND_TO_INFO.get(kind, InfoKind.FILING),
                            ticker, str(t), str(t), "edgar", PROV_RECORDED, True, False, KIND_STRENGTH.get(kind, 0.3),
                            0, f"{getattr(row, 'form', '')} {kind}"))
    return out


def items_from_insider(ins: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """DERA insider purchases: FILING_DATE is a date (not a time), so publication is DATE precision; the transaction itself
    happened earlier (TRANS_DATE). Cluster buying is a bullish precursor, so direction is +1."""
    if ins is None or len(ins) == 0:
        return []
    cols = {c.upper(): c for c in ins.columns}
    for c in ("FILING_DATE", "ISSUERTRADINGSYMBOL"):
        if c not in cols:
            raise KnowabilityError(f"insider frame lacks {c}")
    sub = ins[ins[cols["ISSUERTRADINGSYMBOL"]] == ticker]
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    out = []
    for i, row in enumerate(sub.itertuples(index=False)):
        rec = dict(zip(sub.columns, row))
        fd, _ = to_ny(str(rec[cols["FILING_DATE"]])[:10])
        if fd is None or not (lo <= fd <= hi):
            continue
        td = rec.get(cols.get("TRANS_DATE", ""), None)
        out.append(InfoItem(f"INS-{ticker}-{fd.strftime('%Y%m%d')}-{i}", InfoKind.INSIDER, ticker,
                            None if td is None else str(td)[:10], str(fd.date()), "insider", PROV_RECORDED, True, False,
                            0.35, 1, "open-market purchase"))
    return out


def items_from_effective_series(values: pd.Series, kind: InfoKind, subject: str, source: str,
                                strength: float = 0.3) -> list[InfoItem]:
    """Series indexed by EFFECTIVE date (macro releases, fundamentals) -> items with provenance 'inferred_lag': their real
    publication time is not in the data, so judge_item derives it from a LagPolicy and refuses to trust it near the boundary."""
    out = []
    for ts, v in values.dropna().items():
        eff, _ = to_ny(ts)
        if eff is None:
            continue
        out.append(InfoItem(f"{source}-{subject}-{eff.strftime('%Y%m%d')}", kind, subject, str(eff.date()), None, source,
                            PROV_INFERRED, True, False, strength, int(np.sign(v)) if strength and v == v else 0,
                            f"value={float(v):.4g}"))
    return out


def scheduled_announcement(item_id: str, kind: InfoKind, subject: str, announced_at: str, event_at: str, source: str,
                           strength: float = 0.8, detail: str = "") -> InfoItem:
    """An earnings date / FOMC date announced before it happens. Published early on purpose; that is what makes it knowable."""
    return InfoItem(item_id, kind, subject, event_at, announced_at, source, PROV_RECORDED, True, True, strength, 0, detail)


# ==================================================================================================================
# configuration and the move under audit
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class KnowabilityConfig:
    """Every threshold in one hashed object, so a classification can always be traced to the exact rules that made it."""
    min_pre_bars: int = 60
    vol_long: int = 60
    vol_short: int = 5
    mom_lookback: int = 20
    vol_ratio: tuple[float, float] = (1.5, 3.0)           # short/long realised-vol ratio mapped to 0..1
    volume_z: tuple[float, float] = (1.0, 3.0)
    compression: tuple[float, float] = (0.3, 0.7)         # 1 - (10d range / 60d range)
    cs_rank: tuple[float, float] = (0.8, 0.98)
    mkt_ratio: tuple[float, float] = (1.5, 3.0)
    weights: tuple[tuple[str, float], ...] = (("FEATURES", 1.0), ("PATTERNS", 0.8), ("MEMORY", 0.6), ("MACRO", 0.5),
                                              ("MARKET_STATE", 0.5), ("EVENT", 0.9), ("PRICE", 0.7), ("VOLUME", 0.7),
                                              ("TECHNICAL", 0.5), ("CROSS_SECTION", 0.6))
    channel_present: float = 0.3                          # a channel counts as an independent known signal above this
    predictable_at: float = 0.65
    potential_at: float = 0.45
    weak_at: float = 0.2
    flag_pct: float = 0.9                                 # decision-time model percentile that counts as "it flagged this"
    ext_share: float = 0.6
    ext_market_z: float = 1.5
    explain_min: float = 0.5                              # post-hoc cause strength that counts as "a cause was found"
    failure_bar: float = 0.6
    split_tol: float = 0.03
    split_min_ret: float = 0.4
    badtick_ret: float = 0.3
    badtick_revert: float = 0.8
    stale_run: int = 4
    return_tol: float = 0.02
    uncertain_demote: float = 0.5
    min_fit: int = 30

    def check(self) -> list[str]:
        errs = []
        if not (0 < self.weak_at < self.potential_at < self.predictable_at <= 1):
            errs.append("thresholds must satisfy 0 < weak < potential < predictable <= 1")
        for name in ("vol_ratio", "volume_z", "compression", "cs_rank", "mkt_ratio"):
            lo, hi = getattr(self, name)
            if not lo < hi:
                errs.append(f"{name} ramp needs lo < hi")
        if {c for c, _ in self.weights} != set(CHANNELS):
            errs.append("weights must name exactly the ten channels")
        if any(not 0 < w <= 1 for _, w in self.weights):
            errs.append("channel weights must be in (0,1]")
        if self.min_pre_bars < self.vol_long:
            errs.append("min_pre_bars must cover the long volatility window")
        return errs

    def weight(self, channel: str) -> float:
        return dict(self.weights)[channel]

    def hash(self) -> str:
        return stable_hash(self, 12)

    def scaled(self, f: float) -> "KnowabilityConfig":
        """Every classification threshold moved by factor f (sensitivity analysis)."""
        return dataclasses.replace(self, predictable_at=min(1.0, self.predictable_at * f), potential_at=self.potential_at * f,
                                   weak_at=self.weak_at * f, ext_share=min(1.0, self.ext_share * f), explain_min=min(1.0, self.explain_min * f))


@dataclasses.dataclass(frozen=True)
class MoveEvent:
    """A major historical move to audit. `end_date` is when the outcome finished forming: the record matures then, not before."""
    move_id: str
    ticker: str
    decision_date: str
    fill_date: str
    end_date: str
    fwd_return: float
    horizon: int = 5
    regime: str = ""
    sector: str = ""
    model_pct: float | None = None                # decision-time model percentile for this name (None = no model output)
    model_confidence: float | None = None

    def check(self) -> list[str]:
        errs = DecisionBoundary(self.decision_date, self.fill_date).check()
        if as_date(self.end_date) < as_date(self.fill_date):
            errs.append("end date precedes the fill date")
        if not math.isfinite(self.fwd_return):
            errs.append("non-finite forward return")
        if self.horizon < 1:
            errs.append("horizon must be >= 1 session")
        for f in ("model_pct", "model_confidence"):
            v = getattr(self, f)
            if v is not None and not 0.0 <= v <= 1.0:
                errs.append(f"{f} outside [0,1]")
        return errs

    @property
    def boundary(self) -> DecisionBoundary:
        return DecisionBoundary(self.decision_date, self.fill_date)

    @property
    def direction(self) -> int:
        return int(np.sign(self.fwd_return))

    @property
    def matured_at(self) -> str:
        return self.end_date


@dataclasses.dataclass
class MoveInputs:
    """Everything the auditor is given about one move. Bars must include the pre-window, the move window and at least a few
    sessions after (used only to detect bad ticks; recorded as future information used by the auditor)."""
    move: MoveEvent
    bars: pd.DataFrame
    market: pd.Series | None = None
    sector: pd.Series | None = None
    items: Sequence[InfoItem] = ()
    probes: Mapping[str, Sequence[SourceProbe]] = dataclasses.field(default_factory=dict)
    cross: Mapping[str, float] = dataclasses.field(default_factory=dict)
    pattern_hits: Sequence[Mapping[str, Any]] = ()
    memory_hits: Sequence[Mapping[str, Any]] = ()
    base_rates: Mapping[str, float] = dataclasses.field(default_factory=dict)     # item_key -> chance a random window holds one

    def windows(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(pre, move window, after). `pre` ends at the decision date inclusive: the close is known at the cutoff."""
        m = self.move
        if not isinstance(self.bars.index, pd.DatetimeIndex):
            raise KnowabilityError("bars need a DatetimeIndex")
        b = self.bars.sort_index()
        dd, ed = pd.Timestamp(m.decision_date), pd.Timestamp(m.end_date)
        return b.loc[:dd], b.loc[(b.index > dd) & (b.index <= ed)], b.loc[b.index > ed]


# ==================================================================================================================
# data failure detection (a "move" that is really a data defect must never be studied as market behaviour)
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: float                                # 0..1
    detail: str
    uses_future: bool = False                      # needed sessions after the decision to see it


@dataclasses.dataclass(frozen=True)
class DataQuality:
    flags: tuple[QualityFlag, ...]
    failed: bool
    severity: float

    def codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.flags)


SPLIT_RATIOS = (0.5, 1 / 3, 0.25, 0.2, 0.1, 2.0, 3.0, 4.0, 5.0, 10.0)


def _flag_ohlc(df: pd.DataFrame) -> list[QualityFlag]:
    need = {"open", "high", "low", "close", "volume"}
    if not need <= set(df.columns):
        return [QualityFlag("MISSING_COLUMNS", 1.0, f"bars lack {sorted(need - set(df.columns))}")]
    out = []
    px = df[["open", "high", "low", "close"]]
    if px.isna().any().any() or (px <= 0).any().any():
        out.append(QualityFlag("BAD_PRICE", 1.0, "non-positive or missing price in the audited span"))
    bad = ((df["high"] < df["low"]) | (df["close"] > df["high"] * 1.0001) | (df["close"] < df["low"] * 0.9999)).sum()
    if bad:
        out.append(QualityFlag("OHLC_INCONSISTENT", 0.5 if bad < 3 else 0.8, f"{int(bad)} bars with close outside high/low"))
    if df.index.duplicated().any():
        out.append(QualityFlag("DUPLICATE_SESSIONS", 0.8, "repeated dates in the bar index"))
    return out


def _flag_gaps_and_stale(pre: pd.DataFrame, win: pd.DataFrame, cfg: KnowabilityConfig) -> list[QualityFlag]:
    out = []
    span = pd.concat([pre.tail(cfg.vol_long), win])
    if len(span) >= 2:
        gaps = np.diff(span.index.values.astype("datetime64[D]")).astype(int)
        if gaps.max(initial=0) > 5:
            out.append(QualityFlag("MISSING_SESSIONS", 0.4, f"gap of {int(gaps.max())} calendar days in the audited span"))
    for name, part in (("pre", pre.tail(cfg.vol_long)), ("window", win)):
        if len(part) < cfg.stale_run:
            continue
        same = (part["close"].diff().abs() < 1e-12) & (part["volume"] <= 0)
        run = best = 0
        for v in same.values:
            run = run + 1 if v else 0
            best = max(best, run)
        if best >= cfg.stale_run:
            out.append(QualityFlag("STALE_PRICES", 0.7, f"{best} consecutive unchanged closes with zero volume in the {name}"))
    return out


def _flag_split_and_ticks(pre: pd.DataFrame, win: pd.DataFrame, post: pd.DataFrame, items: Sequence[InfoItem],
                          cfg: KnowabilityConfig) -> list[QualityFlag]:
    out = []
    if len(win) == 0 or len(pre) < 2:
        return out
    full = pd.concat([pre, win, post])
    ret = full["close"].pct_change()
    vol_med = float(pre["volume"].tail(cfg.vol_long).median()) if len(pre) else 0.0
    has_action = any(i.kind == InfoKind.CORPORATE_ACTION for i in items)
    for ts in win.index:
        r = ret.loc[ts]
        if not math.isfinite(r) or abs(r) < cfg.split_min_ret:
            continue
        ratio = 1.0 + r
        near = [s for s in SPLIT_RATIOS if abs(ratio - s) / s < cfg.split_tol]
        vol_ratio = float(win.loc[ts, "volume"]) / vol_med if vol_med > 0 else float("nan")
        if near and not (vol_ratio == vol_ratio and vol_ratio > 4.0):
            out.append(QualityFlag("SPLIT_SIGNATURE", 0.9 if not has_action else 0.7,
                                   f"close ratio {ratio:.3f} matches a {near[0]:.3g}:1 split without a volume surge"
                                   + (" (recorded corporate action)" if has_action else "")))
        loc = full.index.get_loc(ts)
        tail = ret.iloc[loc + 1: loc + 3]
        if len(tail) and abs(r) >= cfg.badtick_ret:
            back = float(np.prod(1.0 + tail.values) - 1.0)
            if np.sign(back) == -np.sign(r) and abs(back) >= cfg.badtick_revert * abs(r) / (1 + abs(r)):
                out.append(QualityFlag("BAD_TICK", 0.75, f"{r:+.1%} day reversed {back:+.1%} within {len(tail)} sessions", True))
    return out


def _flag_label_mismatch(m: MoveEvent, win: pd.DataFrame, cfg: KnowabilityConfig) -> list[QualityFlag]:
    if len(win) == 0:
        return [QualityFlag("EMPTY_WINDOW", 1.0, "no bars inside the labelled move window")]
    try:
        real = float(win["close"].iloc[-1] / win["open"].iloc[0] - 1.0)
    except (KeyError, ZeroDivisionError):
        return [QualityFlag("EMPTY_WINDOW", 1.0, "cannot recompute the labelled move")]
    if not math.isfinite(real) or abs(real - m.fwd_return) > cfg.return_tol:
        return [QualityFlag("RETURN_MISMATCH", 0.8, f"label says {m.fwd_return:+.2%}, bars say {real:+.2%}")]
    return []


def assess_data_quality(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig()) -> DataQuality:
    """Run every defect check on the audited span. Failing any check at or above `failure_bar` makes the move DATA_FAILURE,
    the only class allowed to outrank everything else because a defective bar cannot support any market explanation."""
    pre, win, post = inp.windows()
    flags = _flag_ohlc(pd.concat([pre.tail(cfg.vol_long), win]))
    if not any(f.code == "MISSING_COLUMNS" for f in flags):
        flags += _flag_gaps_and_stale(pre, win, cfg)
        flags += _flag_label_mismatch(inp.move, win, cfg)
        flags += _flag_volume_and_truncation(inp.move, pre, win, cfg)
        if not any(f.code in ("BAD_PRICE", "EMPTY_WINDOW") for f in flags):
            flags += _flag_split_and_ticks(pre, win, post, inp.items, cfg)
            flags += split_adjustment_check(pd.concat([pre, win]), inp.items)
    if len(pre) < cfg.min_pre_bars:
        flags.append(QualityFlag("SHORT_HISTORY", 0.3, f"{len(pre)} pre-decision bars, need {cfg.min_pre_bars}"))
    sev = max((f.severity for f in flags), default=0.0)
    return DataQuality(tuple(flags), sev >= cfg.failure_bar, sev)


# ==================================================================================================================
# the ten information channels: what was knowable at the decision cutoff (reads only data at or before the cutoff)
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class ChannelEvidence:
    """anticipation is the strength of the case, from information available at the cutoff only, that a large move was coming
    (None = the channel had no data: that is NOT zero). `future_used` lists anything the auditor looked at beyond the cutoff."""
    channel: str
    anticipation: float | None
    direction: int = 0
    detail: str = ""
    n_items: int = 0
    uncertain_share: float = 0.0
    future_used: tuple[str, ...] = ()

    def has_signal(self, bar: float) -> bool:
        return self.anticipation is not None and self.anticipation >= bar

    def check(self) -> list[str]:
        if self.channel not in CHANNELS:
            return [f"unknown channel {self.channel}"]
        if self.anticipation is not None and not 0.0 <= self.anticipation <= 1.0:
            return [f"{self.channel}: anticipation {self.anticipation} outside [0,1]"]
        return []


def _none(ch: str, why: str) -> ChannelEvidence:
    return ChannelEvidence(ch, None, 0, why)


def _ret(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def price_channel(pre: pd.DataFrame, cfg: KnowabilityConfig) -> ChannelEvidence:
    """Volatility expansion into the cutoff (short vs long realised vol) plus prior momentum direction."""
    r = _ret(pre["close"]) if len(pre) else pd.Series(dtype=float)
    if len(r) < cfg.vol_long:
        return _none("PRICE", "too little price history")
    long_sd, short_sd = float(r.iloc[-cfg.vol_long:].std()), float(r.iloc[-cfg.vol_short:].std())
    if long_sd <= 0:
        return _none("PRICE", "zero long-run volatility")
    ratio = short_sd / long_sd
    mom = float(pre["close"].iloc[-1] / pre["close"].iloc[-1 - cfg.mom_lookback] - 1.0)
    z = mom / (long_sd * math.sqrt(cfg.mom_lookback))
    direction = int(np.sign(z)) if abs(z) > 1.0 else 0
    return ChannelEvidence("PRICE", ramp(ratio, *cfg.vol_ratio), direction, f"vol ratio {ratio:.2f}, momentum z {z:+.2f}")


def volume_channel(pre: pd.DataFrame, cfg: KnowabilityConfig) -> ChannelEvidence:
    """Recent volume against its own history: unusual participation before the move."""
    if len(pre) < cfg.vol_long or (pre["volume"] <= 0).all():
        return _none("VOLUME", "no usable volume history")
    v = pre["volume"].astype(float)
    base = v.iloc[-cfg.vol_long:-3]
    sd = float(base.std())
    if sd <= 0:
        return _none("VOLUME", "constant volume history")
    z = (float(v.iloc[-3:].mean()) - float(base.mean())) / sd
    day = pre["close"].pct_change().iloc[-3:]
    direction = int(np.sign(day.sum())) if z > cfg.volume_z[0] else 0
    return ChannelEvidence("VOLUME", ramp(z, *cfg.volume_z), direction, f"3-day volume z {z:+.2f}")


def technical_channel(pre: pd.DataFrame, cfg: KnowabilityConfig) -> ChannelEvidence:
    """Range compression (a coiled range) and proximity to a 52-week extreme, from the trailing bars only."""
    if len(pre) < cfg.vol_long:
        return _none("TECHNICAL", "too little history")
    rng = (pre["high"] - pre["low"]) / pre["close"]
    short, long = float(rng.iloc[-10:].mean()), float(rng.iloc[-cfg.vol_long:].mean())
    comp = ramp(1.0 - short / long, *cfg.compression) if long > 0 else 0.0
    brk, direction = 0.0, 0
    if len(pre) >= 252:
        hi, lo, last = float(pre["high"].iloc[-252:].max()), float(pre["low"].iloc[-252:].min()), float(pre["close"].iloc[-1])
        near_hi, near_lo = last / hi, lo / last
        brk = 0.6 * ramp(max(near_hi, near_lo), 0.98, 1.0)
        direction = 1 if near_hi >= near_lo else -1
    return ChannelEvidence("TECHNICAL", max(comp, brk), direction if brk > comp else 0, f"compression {comp:.2f}, extreme {brk:.2f}")


def cross_section_channel(cross: Mapping[str, float], cfg: KnowabilityConfig) -> ChannelEvidence:
    """The name's prior-volatility rank in the universe and its sector-relative strength, both computed at the cutoff."""
    if not cross or "vol_rank_pct" not in cross:
        return _none("CROSS_SECTION", "no cross-sectional snapshot supplied")
    pct = float(cross["vol_rank_pct"])
    if not 0.0 <= pct <= 1.0:
        raise KnowabilityError(f"vol_rank_pct {pct} outside [0,1]")
    rel = float(cross.get("sector_rel_z", 0.0))
    return ChannelEvidence("CROSS_SECTION", ramp(pct, *cfg.cs_rank), int(np.sign(rel)) if abs(rel) > 1.0 else 0,
                           f"volatility rank {pct:.2f}, sector-relative z {rel:+.2f}")


def market_state_channel(market: pd.Series | None, decision_date: str, cfg: KnowabilityConfig) -> ChannelEvidence:
    """Market-wide volatility regime at the cutoff: high market volatility makes any move more likely to be systematic."""
    if market is None or len(market) == 0:
        return _none("MARKET_STATE", "no market series")
    m = market.sort_index().loc[:pd.Timestamp(decision_date)].dropna()
    if len(m) < cfg.vol_long:
        return _none("MARKET_STATE", "too little market history")
    long_sd, short_sd = float(m.iloc[-cfg.vol_long:].std()), float(m.iloc[-cfg.vol_short:].std())
    if long_sd <= 0:
        return _none("MARKET_STATE", "zero market volatility")
    ratio = short_sd / long_sd
    return ChannelEvidence("MARKET_STATE", ramp(ratio, *cfg.mkt_ratio), 0, f"market vol ratio {ratio:.2f}")


def _usable(items: Sequence[InfoItem], kinds: set, judg: Mapping[str, AvailabilityJudgement]) -> tuple[list[InfoItem], int, int]:
    rel = [i for i in items if i.kind in kinds]
    known = [i for i in rel if judg[i.item_id].availability == Availability.KNOWN_BEFORE_EVENT]
    unc = sum(1 for i in rel if judg[i.item_id].availability == Availability.UNCERTAIN)
    return known, len(rel), unc


def event_channel(items: Sequence[InfoItem], judg: Mapping[str, AvailabilityJudgement], move: MoveEvent,
                  cfg: KnowabilityConfig) -> ChannelEvidence:
    """Company events (filings, scheduled earnings, insider buying) that were public before the cutoff and whose event time
    falls inside the move window or the days before it. Items known only afterwards are hindsight and are NOT counted here."""
    known, n_rel, unc = _usable(items, {InfoKind.EVENT, InfoKind.FILING, InfoKind.INSIDER}, judg)
    if n_rel == 0:
        return _none("EVENT", "no event items supplied")
    lo = pd.Timestamp(move.decision_date) - pd.Timedelta(days=5)
    hi = pd.Timestamp(move.end_date)
    live = []
    for i in known:
        eff, _ = to_ny(i.effective_at)
        if eff is not None and lo <= eff.normalize() <= hi:
            live.append(i)
    if not live:
        return ChannelEvidence("EVENT", 0.0, 0, "events existed but none timed into the move window", 0, unc / n_rel)
    ant = noisy_or([i.strength for i in live])
    d = int(np.sign(sum(i.direction for i in live)))
    return ChannelEvidence("EVENT", ant, d, f"{len(live)} event(s) known before the cutoff", len(live), unc / n_rel)


def macro_channel(items: Sequence[InfoItem], judg: Mapping[str, AvailabilityJudgement], move: MoveEvent,
                  cfg: KnowabilityConfig) -> ChannelEvidence:
    """Scheduled macro releases inside the move window that were announced before the cutoff (an FOMC date is knowable)."""
    known, n_rel, unc = _usable(items, {InfoKind.MACRO}, judg)
    if n_rel == 0:
        return _none("MACRO", "no macro items supplied")
    lo, hi = pd.Timestamp(move.fill_date), pd.Timestamp(move.end_date)
    live = [i for i in known if i.scheduled and (e := to_ny(i.effective_at)[0]) is not None and lo <= e.normalize() <= hi]
    ant = noisy_or([i.strength for i in live]) if live else 0.0
    return ChannelEvidence("MACRO", ant, 0, f"{len(live)} scheduled release(s) in the window", len(live), unc / n_rel)


def _hits_channel(name: str, hits: Sequence[Mapping[str, Any]], key: str, move: MoveEvent) -> ChannelEvidence:
    """Patterns / memory hits count only if they had matured before the decision date. Later ones are future information
    the auditor must not credit; they are reported, not used."""
    if not hits:
        return _none(name, "no hits supplied")
    dd = pd.Timestamp(move.decision_date)
    good, future = [], []
    for h in hits:
        learned = h.get("learned_at")
        if learned is None or pd.Timestamp(str(learned)[:10]) >= dd:
            future.append(str(h.get("id", "?")))
        else:
            good.append(h)
    if not good:
        return ChannelEvidence(name, 0.0, 0, "every hit was learned at or after the decision date", 0, 1.0, tuple(future))
    ant = noisy_or([min(1.0, max(0.0, float(h.get(key, 0.0)))) * min(1.0, max(0.0, float(h.get("strength", 1.0)))) for h in good])
    d = int(np.sign(sum(float(h.get("effect", 0.0)) for h in good)))
    return ChannelEvidence(name, ant, d, f"{len(good)} matured hit(s), {len(future)} excluded as later", len(good), 0.0, tuple(future))


def features_channel(move: MoveEvent, cfg: KnowabilityConfig, items: Sequence[InfoItem] = (),
                     judg: Mapping[str, AvailabilityJudgement] | None = None) -> ChannelEvidence:
    """What the decision-time model itself said about this name (its percentile) plus any decision-time feature items that
    were public before the cutoff (a feature stamped after the cutoff is a leak and contributes nothing)."""
    known, n_rel, unc = _usable(items, {InfoKind.FEATURE}, judg) if judg is not None else ([], 0, 0)
    feat = noisy_or([i.strength for i in known]) if known else None
    if move.model_pct is None and feat is None:
        return ChannelEvidence("FEATURES", None, 0, "no model output for this name" if not n_rel else "feature items exist but none were public before the cutoff",
                               0, unc / n_rel if n_rel else 0.0)
    ant = max(ramp(move.model_pct, 0.5, cfg.flag_pct) if move.model_pct is not None else 0.0, feat or 0.0)
    conf = "" if move.model_confidence is None else f", confidence {move.model_confidence:.2f}"
    pct = "" if move.model_pct is None else f"model percentile {move.model_pct:.2f}{conf}"
    return ChannelEvidence("FEATURES", ant, 0, "; ".join(x for x in (pct, f"{len(known)} feature item(s)" if known else "") if x), len(known),
                           unc / n_rel if n_rel else 0.0)


def build_channels(inp: MoveInputs, judg: Mapping[str, AvailabilityJudgement], cfg: KnowabilityConfig) -> dict[str, ChannelEvidence]:
    """The full information state at the decision cutoff, one entry per section-8 channel."""
    pre, _, _ = inp.windows()
    pre = pre.tail(max(cfg.vol_long * 5, 260))
    m = inp.move
    ch = {
        "FEATURES": features_channel(m, cfg, inp.items, judg),
        "PATTERNS": _hits_channel("PATTERNS", inp.pattern_hits, "p_real", m),
        "MEMORY": _hits_channel("MEMORY", inp.memory_hits, "similarity", m),
        "MACRO": macro_channel(inp.items, judg, m, cfg),
        "MARKET_STATE": market_state_channel(inp.market, m.decision_date, cfg),
        "EVENT": event_channel(inp.items, judg, m, cfg),
        "PRICE": price_channel(pre, cfg),
        "VOLUME": volume_channel(pre, cfg),
        "TECHNICAL": technical_channel(pre, cfg),
        "CROSS_SECTION": cross_section_channel(inp.cross, cfg),
    }
    bad = [e for c in ch.values() for e in c.check()]
    if bad:
        raise KnowabilityError("; ".join(bad))
    return ch


def combine_anticipation(channels: Mapping[str, ChannelEvidence], cfg: KnowabilityConfig) -> tuple[float, int, float]:
    """(total anticipation, number of independent present channels, coverage). Total is noisy-or over weighted channels, so
    many weak signals cannot masquerade as one strong one unless they are independent; channels with no data contribute
    nothing but lower `coverage`, the honest 'how much of the picture we could see'."""
    vals, present, seen = [], 0, 0.0
    tot_w = sum(w for _, w in cfg.weights)
    for name, c in channels.items():
        if c.anticipation is None:
            continue
        seen += cfg.weight(name)
        vals.append(cfg.weight(name) * c.anticipation)
        present += int(c.anticipation >= cfg.channel_present)
    return noisy_or(vals), present, seen / tot_w


# ==================================================================================================================
# external (systematic) attribution
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class ExternalDecomposition:
    beta_market: float
    beta_sector: float
    market_window_ret: float
    sector_window_ret: float
    explained: float
    idiosyncratic: float
    systematic_share: float                       # share of the move explained by market and sector, in [0,1]
    market_move_z: float
    sector_move_z: float
    n_fit: int


def _window_compound(s: pd.Series | None, lo: str, hi: str) -> float:
    if s is None:
        return 0.0
    w = s.sort_index().loc[(s.index > pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))].dropna()
    return float(np.prod(1.0 + w.values) - 1.0) if len(w) else 0.0


def decompose_external(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig()) -> ExternalDecomposition | None:
    """Betas are fitted on the PRE-decision window only, then applied to the market and sector returns over the move window.
    None when the fit is impossible: an unmeasurable external share must not be read as zero or as one."""
    pre, win, _ = inp.windows()
    m = inp.move
    if inp.market is None or len(win) == 0:
        return None
    y = pre["close"].pct_change().dropna().tail(cfg.vol_long * 2)
    mk = inp.market.sort_index().reindex(y.index)
    if int(mk.notna().sum()) < cfg.min_fit:
        return None
    sec = inp.sector.sort_index().reindex(y.index) if inp.sector is not None else None
    ok = mk.notna() & (sec.notna() if sec is not None else True)
    y, mk = y[ok], mk[ok]
    if len(y) < cfg.min_fit or float(mk.var()) <= 0:
        return None
    b_m = float(np.cov(y, mk)[0, 1] / mk.var())
    b_s, sec_orth_win = 0.0, 0.0
    sec_win = _window_compound(inp.sector, m.decision_date, m.end_date)
    mkt_win = _window_compound(inp.market, m.decision_date, m.end_date)
    if sec is not None:
        s = sec[ok]
        gam = float(np.cov(s, mk)[0, 1] / mk.var())
        s_orth = s - gam * mk
        if float(s_orth.var()) > 0:
            resid = y - b_m * mk
            b_s = float(np.cov(resid, s_orth)[0, 1] / s_orth.var())
            sec_orth_win = sec_win - gam * mkt_win
    b_m, b_s = 0.67 * b_m + 0.33, 0.67 * b_s                   # shrink the market beta toward 1 (Blume) and the sector loading toward 0
    explained = b_m * mkt_win + b_s * sec_orth_win
    share = 0.0
    if abs(m.fwd_return) > 1e-9 and np.sign(explained) == np.sign(m.fwd_return):
        share = float(min(1.0, abs(explained) / abs(m.fwd_return)))
    mk_sd = float(inp.market.sort_index().loc[:pd.Timestamp(m.decision_date)].tail(cfg.vol_long).std())
    z = abs(mkt_win) / (mk_sd * math.sqrt(m.horizon)) if mk_sd > 0 else 0.0
    sc_sd = float(inp.sector.sort_index().loc[:pd.Timestamp(m.decision_date)].tail(cfg.vol_long).std()) if inp.sector is not None else 0.0
    sz = abs(sec_orth_win) / (sc_sd * math.sqrt(m.horizon)) if sc_sd > 0 else 0.0
    return ExternalDecomposition(b_m, b_s, mkt_win, sec_win, explained, m.fwd_return - explained, share, z, sz, len(y))


# ==================================================================================================================
# hindsight explanations: causes found AFTER the fact, never converted into predictive knowledge
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class Explanation:
    item_id: str
    kind: InfoKind
    availability: Availability                 # of the cause itself
    strength: float
    precursor: Availability                    # could anything legitimate have revealed it beforehand?
    why: str

    def check(self) -> list[str]:
        return [] if 0.0 <= self.strength <= 1.0 else [f"{self.item_id}: strength outside [0,1]"]


def item_key(it: InfoItem) -> str:
    """Kind plus the last token of the detail (an EDGAR filing kind such as EARN), the unit base rates are counted in."""
    tail = it.detail.split()[-1] if it.detail.split() else ""
    return f"{it.kind.value}:{tail}" if it.kind == InfoKind.FILING and tail else it.kind.value


CAUSE_KINDS = {InfoKind.EVENT, InfoKind.FILING, InfoKind.INSIDER, InfoKind.MACRO, InfoKind.NEWS_EXTERNAL, InfoKind.CORPORATE_ACTION}


def find_explanations(inp: MoveInputs, judg: Mapping[str, AvailabilityJudgement], b: DecisionBoundary, cfg: KnowabilityConfig,
                      calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> list[Explanation]:
    """Items published at or after the boundary whose event time falls in the move window and whose direction does not
    contradict the move. Each is asked the knowability question: were there precursors in any covering source?"""
    m = inp.move
    lo, hi = pd.Timestamp(m.decision_date), pd.Timestamp(m.end_date)
    out = []
    for it in inp.items:
        j = judg[it.item_id]
        if it.kind not in CAUSE_KINDS or j.availability not in (Availability.KNOWN_ONLY_AFTER_EVENT, Availability.SIMULTANEOUS,
                                                                Availability.UNCERTAIN):
            continue
        eff, _ = to_ny(it.effective_at)
        adj = it.strength * (1.0 - min(1.0, max(0.0, float(inp.base_rates.get(item_key(it), 0.0)))))
        if eff is None or not (lo <= eff.normalize() <= hi) or adj < cfg.explain_min:
            continue
        if it.direction not in (0, m.direction):
            continue
        probes = inp.probes.get(it.item_id, ())
        if j.availability == Availability.UNCERTAIN:
            prec, why = Availability.UNCERTAIN, "the cause's own publication time is uncertain"
        else:
            prec, why = judge_precursors(it, probes, b, calendar, lag)
        out.append(Explanation(it.item_id, it.kind, j.availability, adj, prec, why))
    return sorted(out, key=lambda e: (-e.strength, e.item_id))


# ==================================================================================================================
# the assessment (section 8 report) and the classifier cascade
# ==================================================================================================================
VOL_EDGES = (0.015, 0.03)                       # prior daily volatility buckets: LOW < 1.5% <= MID < 3% <= HIGH
_DEMOTE = {Knowability.PREDICTABLE: Knowability.POTENTIALLY_PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE: Knowability.WEAKLY_PREDICTABLE}


def vol_bucket(daily_vol: float | None) -> str:
    if daily_vol is None or not math.isfinite(daily_vol):
        return "UNKNOWN"
    return "LOW" if daily_vol < VOL_EDGES[0] else "MID" if daily_vol < VOL_EDGES[1] else "HIGH"


@dataclasses.dataclass(frozen=True)
class KnowabilityAssessment:
    """The "could I have known?" report for one move (section 8). A hindsight artefact: namespace MATURED_RESEARCH."""
    move_id: str
    ticker: str
    decision_date: str
    matured_at: str
    classification: Knowability
    anticipation: float
    n_present_channels: int
    coverage: float
    systematic_share: float | None
    direction_knowable: bool
    knowledge_state_at_decision: Mapping[str, Any]
    future_information_used_by_auditor: tuple[str, ...]
    information_that_would_have_been_available: tuple[str, ...]
    information_that_was_unavailable: tuple[str, ...]
    uncertain_information: tuple[str, ...]
    simultaneous_information: tuple[str, ...]
    explanations: tuple[tuple[str, str, float, str], ...]      # (item id, availability, strength, precursor verdict)
    quality_flags: tuple[str, ...]
    confidence_in_classification: float
    trace: tuple[str, ...]
    regime: str = ""
    sector: str = ""
    vol_bucket: str = "UNKNOWN"
    model_pct: float | None = None
    model_confidence: float | None = None
    abs_move_z: float | None = None
    move_direction: int = 0
    fwd_return: float | None = None
    config_hash: str = ""
    code_hash: str = ""
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def check(self) -> list[str]:
        errs = []
        if self.namespace != Namespace.MATURED_RESEARCH:
            errs.append("a knowability assessment belongs to MATURED_RESEARCH_STATE only")
        if not 0.0 <= self.confidence_in_classification <= 1.0:
            errs.append("confidence outside [0,1]")
        if not self.trace:
            errs.append("classification without a decision trace")
        if self.classification == Knowability.DATA_FAILURE and not self.quality_flags:
            errs.append("DATA_FAILURE without a quality flag")
        if self.classification == Knowability.INFORMATIONALLY_UNAVAILABLE and not self.explanations:
            errs.append("INFORMATIONALLY_UNAVAILABLE needs a discovered cause")
        if as_date(self.matured_at) <= as_date(self.decision_date):
            errs.append("outcome cannot mature on or before the decision date")
        return errs

    def content_hash(self) -> str:
        return stable_hash(self._plain(), 16)

    def _plain(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["knowledge_state_at_decision"] = dict(self.knowledge_state_at_decision)
        return d

    def to_record(self, code_hash: str = "") -> MaturedRecord:
        """Wrap for the research world. Nothing here is trader-visible until MaturedRecord.gate(now) passes."""
        prov = Provenance(created_real=self.matured_at, learned_at=self.matured_at, code_hash=code_hash or self.code_hash or "none",
                          config_hash=self.config_hash, outcomes_seen_through=self.matured_at)
        return MaturedRecord("KNW-" + self.move_id, self.matured_at, self._plain(), prov)


def _channel_state(ch: Mapping[str, ChannelEvidence]) -> dict[str, dict]:
    return {k: {"anticipation": v.anticipation, "direction": v.direction, "detail": v.detail, "n_items": v.n_items,
                "uncertain_share": v.uncertain_share} for k, v in ch.items()}


def _prior_vol(pre: pd.DataFrame, n: int) -> float | None:
    r = _ret(pre["close"]) if len(pre) else pd.Series(dtype=float)
    return float(r.iloc[-n:].std()) if len(r) >= 20 else None


def _move_z(m: MoveEvent, pv: float | None) -> float | None:
    return abs(m.fwd_return) / (pv * math.sqrt(m.horizon)) if pv and pv > 0 else None


def _confidence(margin: float, coverage: float, unc: float) -> float:
    base = 0.5 + 0.5 * min(1.0, max(0.0, margin) / 0.2)
    return float(min(1.0, max(0.0, base * (0.6 + 0.4 * coverage) * (1.0 - 0.5 * unc))))


def classify_move(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), calendar: Calendar | None = None,
                  lag: LagPolicy = LagPolicy(), code_hash: str = "", drop_channels: Iterable[str] = ()) -> KnowabilityAssessment:
    """The cascade. Order matters and is part of the contract: a data defect outranks every market story; a mostly
    systematic move that no channel foresaw is external; then anticipation decides predictable / potential / weak; a move
    nobody could foresee is INFORMATIONALLY_UNAVAILABLE only when a cause was found AND no covering source held a precursor,
    otherwise it stays UNKNOWN. Missing or uncertain information only ever lowers a class."""
    m = inp.move
    errs = m.check() + cfg.check()
    if errs:
        raise KnowabilityError("; ".join(errs))
    b = m.boundary
    judg = judge_all(inp.items, b, calendar, lag)
    pre, _, post = inp.windows()
    pv = _prior_vol(pre, cfg.vol_long)
    ids = {a: sorted(i for i, j in judg.items() if j.availability == a) for a in Availability}
    future = [f"bars:{len(post)} sessions after the move (bad-tick check)"] if len(post) else []
    future += [f"item:{i} ({judg[i].availability.value})" for i in ids[Availability.KNOWN_ONLY_AFTER_EVENT]]
    common = dict(move_id=m.move_id, ticker=m.ticker, decision_date=m.decision_date, matured_at=m.matured_at, regime=m.regime,
                  sector=m.sector, vol_bucket=vol_bucket(pv), model_pct=m.model_pct, model_confidence=m.model_confidence,
                  abs_move_z=_move_z(m, pv), move_direction=m.direction, fwd_return=m.fwd_return, config_hash=cfg.hash(), code_hash=code_hash)
    dq = assess_data_quality(inp, cfg)
    if dq.failed:
        top = max(dq.flags, key=lambda f: f.severity)
        fut = tuple(future + [f"flag:{f.code}" for f in dq.flags if f.uses_future])
        a = KnowabilityAssessment(classification=Knowability.DATA_FAILURE, anticipation=0.0, n_present_channels=0, coverage=0.0,
                                  systematic_share=None, direction_knowable=False, knowledge_state_at_decision={},
                                  future_information_used_by_auditor=tuple(fut),
                                  information_that_would_have_been_available=tuple(ids[Availability.KNOWN_BEFORE_EVENT]),
                                  information_that_was_unavailable=tuple(ids[Availability.UNAVAILABLE]),
                                  uncertain_information=tuple(ids[Availability.UNCERTAIN]),
                                  simultaneous_information=tuple(ids[Availability.SIMULTANEOUS]), explanations=(),
                                  quality_flags=dq.codes(), confidence_in_classification=0.5 + 0.5 * dq.severity,
                                  trace=(f"data defect {top.code} severity {top.severity:.2f}: {top.detail}",), **common)
        return a
    ch = build_channels(inp, judg, cfg)
    for name in drop_channels:
        if name not in CHANNELS:
            raise KnowabilityError(f"cannot ablate unknown channel {name!r}")
        ch[name] = _none(name, "ablated")
    A, n_present, coverage = combine_anticipation(ch, cfg)
    ext = decompose_external(inp, cfg)
    expl = find_explanations(inp, judg, b, cfg, calendar, lag)
    flagged = m.model_pct is not None and m.model_pct >= cfg.flag_pct
    unc_shares = [c.uncertain_share for c in ch.values() if c.n_items or c.uncertain_share]
    unc = float(np.mean(unc_shares)) if unc_shares else 0.0
    trace = [f"anticipation {A:.2f} from {n_present} independent channel(s), coverage {coverage:.2f}, flagged={flagged}"]
    for e in expl[:1]:
        led = pre_publication_share(inp, next(i for i in inp.items if i.item_id == e.item_id))
        if led is not None and led >= 0.5:
            trace.append(f"PRICE_LED_NEWS: {led:.0%} of the move happened before {e.item_id} was public (leak, informed trading or a bad timestamp)")
    mkt_ant = ch["MARKET_STATE"].anticipation or 0.0
    sys_share = None if ext is None else ext.systematic_share
    ext_hit = ext is not None and ext.systematic_share >= cfg.ext_share and max(ext.market_move_z, ext.sector_move_z) >= cfg.ext_market_z
    margin = 0.0
    if ext_hit and A < cfg.predictable_at and mkt_ant < 0.5:
        cls = Knowability.EXTERNALLY_CAUSED
        margin = ext.systematic_share - cfg.ext_share
        trace.append(f"systematic share {ext.systematic_share:.2f} >= {cfg.ext_share}, market z {ext.market_move_z:.1f}, sector z "
                     f"{ext.sector_move_z:.1f}, no channel foresaw the market state")
    elif A >= cfg.predictable_at and n_present >= 2 and flagged:
        cls, margin = Knowability.PREDICTABLE, A - cfg.predictable_at
        trace.append(f"anticipation >= {cfg.predictable_at} with >=2 channels and the model flagged it")
    elif A >= cfg.predictable_at or (A >= cfg.potential_at and n_present >= 2):
        cls, margin = Knowability.POTENTIALLY_PREDICTABLE, min(abs(A - cfg.potential_at), abs(A - cfg.predictable_at))
        trace.append("information existed before the cutoff but " + ("the model did not flag it" if not flagged else "too few independent channels"))
    elif A >= cfg.weak_at:
        cls, margin = Knowability.WEAKLY_PREDICTABLE, min(A - cfg.weak_at, cfg.potential_at - A)
        trace.append(f"anticipation between {cfg.weak_at} and {cfg.potential_at}")
    else:
        best = next((e for e in expl), None)
        if best is None:
            cls, margin = Knowability.UNKNOWN, cfg.weak_at - A
            trace.append("no channel anticipated it and no cause was found: unknown stays unknown")
        elif best.precursor == Availability.UNAVAILABLE:
            cls, margin = Knowability.INFORMATIONALLY_UNAVAILABLE, best.strength - cfg.explain_min
            trace.append(f"cause {best.item_id} found after the fact and no covering source held a precursor")
        elif best.precursor == Availability.KNOWN_BEFORE_EVENT:
            cls, margin = Knowability.WEAKLY_PREDICTABLE, 0.0
            trace.append(f"cause {best.item_id} had precursors before the boundary that the channels did not capture: {best.why}")
        else:
            cls, margin = Knowability.UNKNOWN, cfg.weak_at - A
            trace.append(f"cause {best.item_id} found but its knowability cannot be established ({best.why}); left UNKNOWN")
    if cls in _DEMOTE and unc >= cfg.uncertain_demote:
        trace.append(f"{unc:.0%} of the timestamped evidence is UNCERTAIN: demoted from {cls.value}")
        cls, margin = _DEMOTE[cls], 0.0
    dirs = [c.direction for c in ch.values() if c.anticipation is not None and c.anticipation >= cfg.channel_present and c.direction]
    dir_ok = bool(dirs) and all(d == m.direction for d in dirs) and len(dirs) >= 2
    conf = _confidence(margin, coverage, unc) if cls != Knowability.UNKNOWN else float(min(1.0, 0.5 + 0.5 * coverage * (1.0 - unc)))
    fut = list(future) + [f"channel:{k}:{x}" for k, c in ch.items() for x in c.future_used]
    return KnowabilityAssessment(
        classification=cls, anticipation=A, n_present_channels=n_present, coverage=coverage, systematic_share=sys_share,
        direction_knowable=dir_ok, knowledge_state_at_decision=_channel_state(ch), future_information_used_by_auditor=tuple(fut),
        information_that_would_have_been_available=tuple(ids[Availability.KNOWN_BEFORE_EVENT]),
        information_that_was_unavailable=tuple(sorted(set(ids[Availability.UNAVAILABLE]) | set(ids[Availability.KNOWN_ONLY_AFTER_EVENT]))),
        uncertain_information=tuple(ids[Availability.UNCERTAIN]), simultaneous_information=tuple(ids[Availability.SIMULTANEOUS]),
        explanations=tuple((e.item_id, e.availability.value, e.strength, e.precursor.value) for e in expl),
        quality_flags=dq.codes(), confidence_in_classification=conf, trace=tuple(trace), **common)


# ==================================================================================================================
# robustness: is the class an accident of a threshold, or of the future?
# ==================================================================================================================
def class_stability(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), factors: Sequence[float] = (0.8, 0.9, 1.1, 1.25),
                    calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> dict[str, Any]:
    """Re-classify under every threshold scaled by `factors`. Stable classes survive; a class that flips under +-10% is a
    coin-flip near a boundary and should be reported with that fragility rather than trusted."""
    base = classify_move(inp, cfg, calendar, lag).classification
    flips = {}
    for f in factors:
        c = classify_move(inp, cfg.scaled(f), calendar, lag).classification
        if c != base:
            flips[f] = c.value
    return {"base": base.value, "stable_share": 1.0 - len(flips) / max(1, len(factors)), "flips": flips}


def _scramble_future(inp: MoveInputs, seed: int) -> MoveInputs:
    """Copy of the inputs with everything after the decision cutoff replaced by noise."""
    rng = np.random.default_rng(seed)
    m, dd = inp.move, pd.Timestamp(inp.move.decision_date)
    bars = inp.bars.sort_index().copy()
    fut = bars.index > dd
    for c in ("open", "high", "low", "close"):
        bars.loc[fut, c] = bars.loc[fut, c].to_numpy() * rng.uniform(0.5, 2.0, int(fut.sum()))
    bars.loc[fut, "volume"] = bars.loc[fut, "volume"].to_numpy() * rng.uniform(0.1, 10.0, int(fut.sum()))
    def scr(s):
        if s is None:
            return None
        s = s.sort_index().copy()
        f = s.index > dd
        s.loc[f] = rng.normal(0.0, 0.05, int(f.sum()))
        return s
    b = m.boundary
    items = []
    for it in inp.items:
        pub, _ = to_ny(it.published_at)
        if pub is not None and pub > b.cutoff:
            it = dataclasses.replace(it, strength=float(rng.uniform(0.0, 1.0)), direction=int(rng.choice([-1, 0, 1])))
        items.append(it)
    hits = lambda hs: [h if pd.Timestamp(str(h.get("learned_at", "2100-01-01"))[:10]) < dd else {**h, "p_real": float(rng.uniform()), "similarity": float(rng.uniform())} for h in hs]
    return dataclasses.replace(inp, bars=bars, market=scr(inp.market), sector=scr(inp.sector), items=items,
                               pattern_hits=hits(inp.pattern_hits), memory_hits=hits(inp.memory_hits))


def channels_future_invariant(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), seed: int = 0,
                              calendar: Calendar | None = None, lag: LagPolicy = LagPolicy(), atol: float = 1e-9) -> list[str]:
    """The no-peeking proof for the information state: scramble every future value and rebuild the ten channels. Any channel
    whose anticipation moves has read something it must not have. Returns the offending channel names (empty = clean)."""
    b = inp.move.boundary
    a = build_channels(inp, judge_all(inp.items, b, calendar, lag), cfg)
    alt = _scramble_future(inp, seed)
    c = build_channels(alt, judge_all(alt.items, b, calendar, lag), cfg)
    bad = []
    for k in CHANNELS:
        x, y = a[k].anticipation, c[k].anticipation
        if (x is None) != (y is None) or (x is not None and abs(x - y) > atol):
            bad.append(k)
    return bad


# ==================================================================================================================
# hindsight firewall: the label may never become a feature, a prompt or a trader input
# ==================================================================================================================
HINDSIGHT_MARKERS = tuple(k.value.lower() for k in Knowability) + (
    "knowability", "could_have_known", "hindsight", "informationally_unavailable", "unknown_cause", "explained_after",
    "externally_caused", "information_that_would_have_been_available")


def refuse_hindsight_columns(columns: Iterable[str]) -> None:
    """Fail closed if any feature/training column name looks like a knowability or hindsight label."""
    bad = sorted(str(c) for c in columns if any(mk in str(c).lower() for mk in HINDSIGHT_MARKERS))
    if bad:
        raise FirewallBreach(f"hindsight/knowability labels offered as features: {bad}")


def assert_not_trader_bound(obj: Any, where: str = "trader input") -> None:
    """A KnowabilityAssessment (or anything carrying MATURED_RESEARCH_STATE) reaching the trader is a breach, whatever its shape."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, (KnowabilityAssessment, Explanation, ChannelEvidence)) or getattr(o, "namespace", None) == Namespace.MATURED_RESEARCH:
            raise FirewallBreach(f"{type(o).__name__} is MATURED_RESEARCH_STATE and cannot enter {where}")
        if isinstance(o, Mapping):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)


def release_for_trader(a: KnowabilityAssessment, now, replaying_years: Iterable[int] = ()) -> dict[str, Any]:
    """The only road out: MaturedRecord.gate(now) (outcome matured strictly before real `now`), refusal while the record's own
    year is being replayed in disguise (the same-year rerun leak), and a payload with no ticker, date, year or id."""
    payload = a.to_record().gate(now)
    if as_date(a.decision_date).year in {int(y) for y in replaying_years}:
        raise FirewallBreach(f"knowability research filed under {as_date(a.decision_date).year} while that year is replayed")
    return {"knowability": payload["classification"].value if isinstance(payload["classification"], Knowability)
            else str(payload["classification"]), "anticipation_bucket": round(float(payload["anticipation"]) * 4) / 4,
            "confidence_bucket": round(float(payload["confidence_in_classification"]) * 4) / 4,
            "n_present_channels": int(payload["n_present_channels"])}


# ==================================================================================================================
# ledger and aggregate views (kept small: one assessment row per major move, never the bars)
# ==================================================================================================================
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a rate; (nan, nan) when n == 0 (an empty group has no rate, not a zero rate)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _era_labels(rows: Sequence[KnowabilityAssessment], n: int) -> list[str]:
    """Equal-count eras by decision date ("E1" earliest ... "En"), so era comparisons never depend on calendar-year lumpiness."""
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda i: (rows[i].decision_date, rows[i].move_id))
    lab = [""] * len(rows)
    for rank, i in enumerate(order):
        lab[i] = f"E{min(n, 1 + rank * n // len(rows))}"
    return lab


class KnowabilityLedger:
    """Append-only, one row per move id. Re-adding an identical assessment is a no-op; a different one for the same move is an
    error (history is immutable: reclassify under a new config by using a new ledger, not by overwriting)."""

    def __init__(self):
        self._rows: dict[str, KnowabilityAssessment] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def rows(self) -> list[KnowabilityAssessment]:
        return list(self._rows.values())

    def add(self, a: KnowabilityAssessment) -> bool:
        errs = a.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        cur = self._rows.get(a.move_id)
        if cur is not None:
            if cur.content_hash() != a.content_hash():
                raise KnowabilityError(f"{a.move_id} already classified differently; history is immutable")
            return False
        self._rows[a.move_id] = a
        return True

    def counts(self) -> dict[str, int]:
        out = {k.value: 0 for k in Knowability}
        for a in self._rows.values():
            out[a.classification.value] += 1
        return out

    def table(self, by: str = "regime") -> pd.DataFrame:
        """Class shares per group with Wilson intervals on each share. `by`: regime, sector, vol_bucket, year, confidence."""
        def key(a: KnowabilityAssessment) -> str:
            if by == "year":
                return str(as_date(a.decision_date).year)
            if by.startswith("era"):
                return eras[a.move_id]
            if by == "direction":
                return "UP" if a.move_direction > 0 else "DOWN" if a.move_direction < 0 else "FLAT"
            if by == "confidence":
                c = a.model_confidence
                return "NONE" if c is None else "LOW" if c < 0.34 else "MID" if c < 0.67 else "HIGH"
            v = getattr(a, by, None)
            if v is None:
                raise KnowabilityError(f"unknown grouping {by!r}")
            return str(v) or "UNSPECIFIED"
        rows = []
        eras = dict(zip((a.move_id for a in self._rows.values()), _era_labels(list(self._rows.values()), int(by[3:] or 3)))) if by.startswith("era") else {}
        groups: dict[str, list[KnowabilityAssessment]] = {}
        for a in self._rows.values():
            groups.setdefault(key(a), []).append(a)
        for g, items in sorted(groups.items()):
            n = len(items)
            for k in Knowability:
                c = sum(1 for a in items if a.classification == k)
                lo, hi = wilson(c, n)
                rows.append({"group": g, "class": k.value, "n": c, "of": n, "share": c / n, "lo": lo, "hi": hi})
        return pd.DataFrame(rows, columns=["group", "class", "n", "of", "share", "lo", "hi"])


def hindsight_dependence(ledger: KnowabilityLedger) -> dict[str, float]:
    """How much of what we can explain we could only explain afterwards. explained = moves with at least one discovered cause;
    after_only = of those, the share whose class says the cause was not knowable. A high value means the research world's
    understanding is mostly hindsight and must not be mistaken for a predictive edge."""
    rows = ledger.rows()
    expl = [a for a in rows if a.explanations]
    after = [a for a in expl if a.classification in (Knowability.INFORMATIONALLY_UNAVAILABLE, Knowability.UNKNOWN,
                                                     Knowability.EXTERNALLY_CAUSED)]
    return {"n_moves": len(rows), "n_explained": len(expl), "after_only_share": len(after) / len(expl) if expl else float("nan"),
            "predictable_share": (sum(1 for a in rows if a.classification == Knowability.PREDICTABLE) / len(rows)) if rows else float("nan")}


def model_gap(ledger: KnowabilityLedger, cfg: KnowabilityConfig = KnowabilityConfig()) -> dict[str, Any]:
    """Moves whose information existed beforehand (POTENTIALLY_PREDICTABLE with high anticipation) but the model did not flag:
    the size of the research opportunity that is about using knowable information better, not finding new data."""
    pot = [a for a in ledger.rows() if a.classification == Knowability.POTENTIALLY_PREDICTABLE]
    missed = [a for a in pot if a.model_pct is None or a.model_pct < cfg.flag_pct]
    by_channel: dict[str, int] = {}
    for a in missed:
        for ch, st in a.knowledge_state_at_decision.items():
            if st["anticipation"] is not None and st["anticipation"] >= cfg.channel_present and ch != "FEATURES":
                by_channel[ch] = by_channel.get(ch, 0) + 1
    return {"potential": len(pot), "unflagged": len(missed), "unflagged_share": len(missed) / len(pot) if pot else float("nan"),
            "channels_the_model_ignored": dict(sorted(by_channel.items(), key=lambda kv: -kv[1]))}


def render_report(ledger: KnowabilityLedger) -> str:
    """Plain-text summary for the research report (no tickers or dates: shares and counts only)."""
    n = len(ledger)
    if n == 0:
        return "knowability: no moves classified yet"
    lines = [f"knowability: {n} major moves classified"]
    for k, c in ledger.counts().items():
        lo, hi = wilson(c, n)
        lines.append(f"  {k:<28s} {c:>5d}  {c / n:6.1%}  [{lo:5.1%}, {hi:5.1%}]")
    h = hindsight_dependence(ledger)
    lines.append(f"  explained {h['n_explained']} moves; {h['after_only_share']:.0%} of those only in hindsight")
    g = model_gap(ledger)
    lines.append(f"  model ignored knowable information on {g['unflagged']} of {g['potential']} potentially predictable moves")
    return "\n".join(lines)


# ==================================================================================================================
# streaming selection: keep exception rows only, one snapshot per day (a market-wide scan cannot fit in memory)
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class DaySnapshot:
    date: str
    n_names: int
    market_return: float
    cross_dispersion: float
    median_abs_return: float
    frac_extreme: float


def select_major_moves(day: str, returns: pd.Series, prior_vol: pd.Series, z_min: float = 3.0,
                       max_n: int = 200) -> tuple[DaySnapshot, pd.DataFrame]:
    """One pass over a day's cross-section -> (its snapshot, the exception rows |return| >= z_min prior sigmas, capped at max_n
    by size). Names without a positive prior volatility are excluded by construction, not treated as huge movers."""
    r = returns.astype(float)
    pv = prior_vol.reindex(r.index).astype(float)
    ok = r.notna() & pv.notna() & (pv > 0)
    z = (r[ok] / pv[ok])
    ext = z[z.abs() >= z_min].sort_values(key=lambda s: -s.abs()).head(max_n)
    snap = DaySnapshot(str(as_date(day)), int(ok.sum()), float(r[ok].mean()) if ok.any() else float("nan"),
                       float(r[ok].std()) if ok.sum() > 1 else float("nan"), float(r[ok].abs().median()) if ok.any() else float("nan"),
                       float((z.abs() >= z_min).mean()) if ok.any() else float("nan"))
    return snap, pd.DataFrame({"z": ext, "ret": r.loc[ext.index], "prior_vol": pv.loc[ext.index]})


# ==================================================================================================================
# public entry
# ==================================================================================================================
@dataclasses.dataclass
class KnowabilityState:
    cfg: KnowabilityConfig = KnowabilityConfig()
    lag: LagPolicy = LagPolicy()
    ledger: KnowabilityLedger = dataclasses.field(default_factory=KnowabilityLedger)
    snapshots: dict[str, DaySnapshot] = dataclasses.field(default_factory=dict)
    immature: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class StepReport:
    now: str
    n_in: int
    n_classified: int
    n_immature: int
    n_duplicates: int
    counts: Mapping[str, int]
    mean_confidence: float | None
    low_confidence: tuple[str, ...]
    notes: tuple[str, ...]


def step(state: KnowabilityState, now, inputs: Sequence[MoveInputs], calendar: Calendar | None = None,
         code_hash: str | None = None, low_conf: float = 0.5) -> tuple[KnowabilityState, StepReport]:
    """Classify every input whose outcome matured strictly before real `now`; immature inputs are skipped and reported (their
    outcome is not yet a fact, so no hindsight class exists). Returns the (mutated) state and a report."""
    ch = code_hash if code_hash is not None else current_code_hash()
    n_new = n_dup = 0
    confs, low, notes = [], [], []
    for inp in inputs:
        m = inp.move
        if as_date(m.matured_at) >= as_date(now):
            state.immature.append(m.move_id)
            continue
        a = classify_move(inp, state.cfg, calendar, state.lag, ch)
        if state.ledger.add(a):
            n_new += 1
            confs.append(a.confidence_in_classification)
            if a.confidence_in_classification < low_conf:
                low.append(a.move_id)
        else:
            n_dup += 1
        if a.classification == Knowability.DATA_FAILURE:
            notes.append(f"{a.move_id}: data failure {','.join(a.quality_flags)}")
    n_imm = len(inputs) - n_new - n_dup
    return state, StepReport(str(as_date(now)), len(inputs), n_new, n_imm, n_dup, state.ledger.counts(),
                             float(np.mean(confs)) if confs else None, tuple(low), tuple(notes))


# ==================================================================================================================
# planted worlds: synthetic moves whose true knowability class is known by construction (used by tests and, later, by the
# wave-2 research loop to check that the classifier can still tell them apart)
# ==================================================================================================================
def _base_bars(rng: np.random.Generator, n_pre: int, horizon: int, n_post: int, mkt_beta: float = 1.0):
    """Pre-window returns = beta * market + idiosyncratic noise, so the external-attribution fit has something real to find."""
    n = n_pre + 1 + horizon + n_post
    dates = pd.bdate_range(end=pd.Timestamp("2019-06-28"), periods=n)
    mkt = pd.Series(rng.normal(0.0003, 0.008, n), index=dates)
    idio = rng.normal(0.0, 0.012, n)
    ret = pd.Series(mkt_beta * mkt.values + idio, index=dates)
    return dates, mkt, ret


def _finish_bars(dates, ret: pd.Series, rng: np.random.Generator, volume: np.ndarray) -> pd.DataFrame:
    close = 50.0 * (1.0 + ret).cumprod()
    opn = close.shift(1).fillna(close.iloc[0]) * (1.0 + rng.normal(0, 0.001, len(close)))
    hi = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0, 0.003, len(close))))
    lo = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0, 0.003, len(close))))
    return pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close, "volume": volume}, index=dates)


def planted_inputs(kind: str, seed: int = 0, n_pre: int = 300, horizon: int = 3, n_post: int = 4) -> MoveInputs:
    """A synthetic move of a known class. kind in: PREDICTABLE, POTENTIALLY_PREDICTABLE, WEAKLY_PREDICTABLE, EXTERNALLY_CAUSED,
    INFORMATIONALLY_UNAVAILABLE, UNKNOWN, DATA_FAILURE, UNCERTAIN_EVENT (an earnings item with no provenance: must not raise
    a class), HINDSIGHT_ONLY (a cause published after the move with a precursor-free but uncovered source)."""
    rng = np.random.default_rng(seed)
    dates, mkt, ret = _base_bars(rng, n_pre, horizon, n_post)
    vol = rng.lognormal(13.8, 0.3, len(dates))
    d_i = n_pre
    D = dates[d_i]
    fill, end = dates[d_i + 1], dates[d_i + horizon]
    items: list[InfoItem] = []
    probes: dict[str, list[SourceProbe]] = {}
    hits: list[dict] = []
    cross: dict[str, float] = {}
    pct: float | None = None
    target = 0.12 if rng.uniform() < 0.5 else -0.12
    sign = 1 if target > 0 else -1
    win = slice(d_i + 1, d_i + horizon + 1)
    per_day = (1.0 + target) ** (1.0 / horizon) - 1.0

    def set_window(total: float):
        ret.iloc[win] = (1.0 + total) ** (1.0 / horizon) - 1.0

    if kind in ("PREDICTABLE", "POTENTIALLY_PREDICTABLE"):
        ret.iloc[d_i - 4:d_i + 1] *= 3.0
        vol[d_i - 2:d_i + 1] *= 4.0
        set_window(target)
        items.append(scheduled_announcement("ER-1", InfoKind.EVENT, "T", str((D - pd.Timedelta(days=20)).date()),
                                            str((fill + pd.Timedelta(days=1)).date()), "calendar", 0.8, "earnings date"))
        hits.append({"id": "P1", "p_real": 0.9, "strength": 1.0, "effect": sign * 0.05, "learned_at": str((D - pd.Timedelta(days=90)).date())})
        cross = {"vol_rank_pct": 0.96, "sector_rel_z": 1.5 * sign}
        pct = 0.97 if kind == "PREDICTABLE" else 0.4
    elif kind == "WEAKLY_PREDICTABLE":
        base = vol[d_i - 62:d_i - 2]
        vol[d_i - 2:d_i + 1] = base.mean() + 1.7 * base.std()          # a modest, exactly 1.7-sigma volume bump and nothing else
        set_window(target)
    elif kind == "EXTERNALLY_CAUSED":
        mkt.iloc[win] = (1.0 - 0.07) ** (1.0 / horizon) - 1.0
        ret.iloc[win] = 1.3 * mkt.iloc[win].values + rng.normal(0, 0.002, horizon)
        target = float(np.prod(1.0 + ret.iloc[win].values) - 1.0)
    elif kind in ("INFORMATIONALLY_UNAVAILABLE", "HINDSIGHT_ONLY"):
        set_window(target)
        acc = str((dates[d_i + 2] + pd.Timedelta(hours=10)))
        items.append(InfoItem("8K-1", InfoKind.FILING, "T", acc, acc, "edgar", PROV_RECORDED, True, False, 0.85, sign, "acquisition closed"))
        probes["8K-1"] = [SourceProbe("edgar", True, ()), SourceProbe("insider", kind == "INFORMATIONALLY_UNAVAILABLE", ())]
    elif kind == "UNCERTAIN_EVENT":
        ret.iloc[d_i - 4:d_i + 1] *= 2.0
        vol[d_i - 2:d_i + 1] *= 3.0
        set_window(target)
        items.append(InfoItem("ER-U", InfoKind.EVENT, "T", str((fill + pd.Timedelta(days=1)).date()), None, "calendar",
                              PROV_UNKNOWN, True, True, 0.9, 0, "earnings date of unknown provenance"))
    elif kind == "BACKFILLED":
        set_window(target)
        late = str(dates[d_i + horizon + 2] + pd.Timedelta(hours=10))
        items.append(InfoItem("10KA-1", InfoKind.FILING, "T", str((D - pd.Timedelta(days=2)).date()), late, "edgar", PROV_RECORDED,
                              True, False, 0.9, sign, "10-K/A restated, filed late"))
    elif kind == "DATA_FAILURE":
        horizon = 1
        ret.iloc[d_i + 1] = 0.5
        ret.iloc[d_i + 2] = -0.34
        target = 0.5
    elif kind == "UNKNOWN":
        set_window(target)
    else:
        raise KnowabilityError(f"unknown planted kind {kind!r}")
    if kind == "DATA_FAILURE":
        end = dates[d_i + 1]
    bars = _finish_bars(dates, ret, rng, vol)
    fwd = float(bars["close"].loc[end] / bars["open"].loc[fill] - 1.0)
    mv = MoveEvent(f"M-{kind[:4]}-{seed}", "TKR", str(D.date()), str(fill.date()), str(end.date()), fwd, horizon,
                   regime="R1", sector="S1", model_pct=pct, model_confidence=None if pct is None else pct)
    sec = pd.Series(0.5 * mkt.values + rng.normal(0, 0.004, len(dates)), index=dates)
    if kind == "EXTERNALLY_CAUSED":
        sec.iloc[win] = 0.5 * mkt.iloc[win].values
    return MoveInputs(mv, bars, mkt, sec, items, probes, cross, hits, [])


# ==================================================================================================================
# extra data-failure checks (volume/price consistency, truncated windows, cross-source disagreement)
# ==================================================================================================================
def _flag_volume_and_truncation(m: MoveEvent, pre: pd.DataFrame, win: pd.DataFrame, cfg: KnowabilityConfig) -> list[QualityFlag]:
    """A window shorter than the labelled horizon (delisting, halt, feed gap) and large price moves that traded no shares."""
    out = []
    if len(win):
        expected = int(np.busday_count(np.datetime64(m.fill_date), np.datetime64(m.end_date))) + 1
        if len(win) < expected - 1:
            out.append(QualityFlag("TRUNCATED_WINDOW", 0.6, f"{len(win)} bars for a {expected}-session window"))
        r = win["close"].pct_change().fillna(win["close"] / pre["close"].iloc[-1] - 1.0 if len(pre) else 0.0)
        silent = (r.abs() > 0.10) & (win["volume"] <= 0)
        if silent.any():
            out.append(QualityFlag("MOVE_WITHOUT_VOLUME", 0.65, f"{int(silent.sum())} day(s) moved >10% on zero volume"))
    return out


def cross_source_check(bars_a: pd.DataFrame, bars_b: pd.DataFrame, tol: float = 0.02, min_overlap: int = 20) -> list[QualityFlag]:
    """Two vendors' closes for the same name: a disagreement beyond `tol` on any overlapping day means at least one is wrong,
    which makes a move that appears in only one of them a data question before it is a market question."""
    idx = bars_a.index.intersection(bars_b.index)
    if len(idx) < min_overlap:
        return [QualityFlag("SOURCES_NOT_COMPARABLE", 0.2, f"only {len(idx)} overlapping sessions")]
    diff = (bars_a.loc[idx, "close"] / bars_b.loc[idx, "close"] - 1.0).abs()
    bad = diff[diff > tol]
    if bad.empty:
        return []
    return [QualityFlag("SOURCE_DISAGREEMENT", min(1.0, 0.5 + float(bad.max())), f"{len(bad)} session(s) differ by more than {tol:.0%}; worst {float(bad.max()):.1%}")]


# ==================================================================================================================
# pre-publication drift: did the price move before the news was public?
# ==================================================================================================================
def pre_publication_share(inp: MoveInputs, item: InfoItem) -> float | None:
    """Share of the move already realised before `item` became public, from the daily bars of the move window. Intraday
    publication splits its own session evenly (a coarse but unbiased assumption); DATE-precision counts the whole day as after.
    A high share means the price led the news: leakage, informed trading, or a wrong timestamp. None when it cannot be measured."""
    pub, prec = to_ny(item.published_at)
    _, win, _ = inp.windows()
    if pub is None or len(win) == 0:
        return None
    total = float(np.log(win["close"].iloc[-1] / win["open"].iloc[0]))
    if abs(total) < 1e-9:
        return None
    day_log = np.log(win["close"] / win["open"]).to_numpy()
    before = 0.0
    for ts, lr in zip(win.index, day_log):
        if ts.normalize() < pub.normalize():
            before += float(lr)
        elif ts.normalize() == pub.normalize() and prec == PREC_INTRADAY:
            before += float(lr) * 0.5
    return float(min(1.0, max(0.0, before / total)))


# ==================================================================================================================
# sources: coverage registry, automated precursor probes, timestamp audit
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    first_date: str
    last_date: str
    granularity: str = PREC_INTRADAY
    typical_lag_sessions: int = 0

    def check(self) -> list[str]:
        errs = []
        if as_date(self.last_date) < as_date(self.first_date):
            errs.append(f"{self.name}: coverage ends before it starts")
        if self.granularity not in (PREC_INTRADAY, PREC_DATE):
            errs.append(f"{self.name}: granularity must be INTRADAY or DATE")
        return errs


class SourceRegistry:
    """What each data source can and cannot see. A source's silence is evidence only inside its own coverage window."""

    def __init__(self, specs: Iterable[SourceSpec] = ()):
        self._specs: dict[str, SourceSpec] = {}
        for s in specs:
            self.add(s)

    def add(self, spec: SourceSpec) -> None:
        errs = spec.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def covers(self, source: str, lo: str, hi: str) -> bool:
        s = self._specs.get(source)
        return s is not None and as_date(s.first_date) <= as_date(lo) and as_date(hi) <= as_date(s.last_date)

    def probe(self, source: str, cause: InfoItem, all_items: Sequence[InfoItem], b: DecisionBoundary, lookback_days: int = 30,
              calendar: Calendar | None = None, lag: LagPolicy = LagPolicy(), min_strength: float = 0.2) -> SourceProbe:
        """Ask one source for precursors of `cause`: same subject, published KNOWN_BEFORE the boundary within the lookback."""
        lo = str((pd.Timestamp(b.decision_date) - pd.Timedelta(days=lookback_days)).date())
        if not self.covers(source, lo, b.fill_date):
            return SourceProbe(source, False, ())
        found = []
        for it in all_items:
            if it.item_id == cause.item_id or it.source != source or it.subject != cause.subject or it.strength < min_strength:
                continue
            j = judge_item(it, b, calendar, lag)
            eff, _ = to_ny(it.effective_at)
            if j.availability == Availability.KNOWN_BEFORE_EVENT and eff is not None and eff >= pd.Timestamp(lo):
                found.append(it.item_id)
        return SourceProbe(source, True, tuple(sorted(found)))


def attach_probes(inp: MoveInputs, registry: SourceRegistry, sources: Sequence[str] | None = None, lookback_days: int = 30,
                  calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> MoveInputs:
    """Fill `inp.probes` for every cause candidate from the registry, so INFORMATIONALLY_UNAVAILABLE is decided by evidence
    about what each source held, never by the absence of a probe."""
    b = inp.move.boundary
    srcs = list(sources) if sources is not None else registry.names()
    probes = {c.item_id: [registry.probe(s, c, inp.items, b, lookback_days, calendar, lag) for s in srcs]
              for c in inp.items if c.kind in CAUSE_KINDS}
    return dataclasses.replace(inp, probes=probes)


@dataclasses.dataclass(frozen=True)
class TimestampFinding:
    code: str
    item_ids: tuple[str, ...]
    detail: str


def audit_items(items: Sequence[InfoItem], now, calendar: Calendar | None = None) -> list[TimestampFinding]:
    """Integrity checks on the timestamps themselves, before any knowability question is asked of them: publication before the
    event without a scheduled flag, publication in the future of `now`, duplicated items with different clocks, intraday
    stamps on non-sessions' weekends, and sources that never carry a time of day."""
    out: list[TimestampFinding] = []
    cutoff = pd.Timestamp(as_date(now))
    early, future, weekend, missing = [], [], [], []
    for it in items:
        pub, prec = to_ny(it.published_at)
        eff, _ = to_ny(it.effective_at)
        if pub is None:
            if it.exists and it.provenance != PROV_INFERRED:
                missing.append(it.item_id)
            continue
        if eff is not None and not it.scheduled and pub < eff.normalize() - pd.Timedelta(days=1):
            early.append(it.item_id)
        if pub >= cutoff:
            future.append(it.item_id)
        if prec == PREC_INTRADAY and pub.weekday() >= 5:
            weekend.append(it.item_id)
    for code, ids, detail in (("PUBLISHED_BEFORE_EVENT", early, "published before it happened and not scheduled"),
                              ("PUBLISHED_IN_FUTURE", future, f"publication at or after {cutoff.date()}"),
                              ("WEEKEND_TIMESTAMP", weekend, "intraday timestamp on a Saturday or Sunday"),
                              ("NO_PUBLICATION_TIME", missing, "exists but has no publication time and no lag rule")):
        if ids:
            out.append(TimestampFinding(code, tuple(sorted(ids)), detail))
    seen: dict[tuple, list[InfoItem]] = {}
    for it in items:
        seen.setdefault((it.subject, it.kind.value, str(it.effective_at)[:10], it.source), []).append(it)
    dup = [x.item_id for g in seen.values() if len(g) > 1 and len({y.published_at for y in g}) > 1 for x in g]
    if dup:
        out.append(TimestampFinding("CONFLICTING_CLOCKS", tuple(sorted(dup)), "same fact stored with different publication times"))
    by_src: dict[str, list[str]] = {}
    for it in items:
        by_src.setdefault(it.source, []).append(to_ny(it.published_at)[1])
    coarse = [s for s, v in by_src.items() if len(v) >= 5 and all(p == PREC_DATE for p in v)]
    if coarse:
        out.append(TimestampFinding("DATE_ONLY_SOURCE", tuple(sorted(coarse)), "source never carries a time of day: intraday order is unknowable"))
    return out


def merge_duplicate_items(items: Iterable[InfoItem]) -> list[InfoItem]:
    """Collapse the same fact reported by several sources to ONE item carrying the EARLIEST recorded publication (the moment the
    world first knew it); provenance stays that of the earliest record. Items with no publication time never win a merge."""
    groups: dict[tuple, list[InfoItem]] = {}
    for it in items:
        groups.setdefault((it.subject, it.kind.value, str(it.effective_at)[:10], it.direction), []).append(it)
    out = []
    for g in groups.values():
        timed = [(to_ny(x.published_at)[0], x) for x in g if to_ny(x.published_at)[0] is not None]
        best = min(timed, key=lambda t: (t[0], t[1].item_id))[1] if timed else sorted(g, key=lambda x: x.item_id)[0]
        out.append(dataclasses.replace(best, strength=max(x.strength for x in g)))
    return sorted(out, key=lambda x: x.item_id)


def estimate_lag_policy(effective, published, sources: Sequence[str], calendar: Calendar | None = None, quantile: float = 0.9,
                        margin_sessions: int = 1) -> LagPolicy:
    """Learn per-source publication lags from records that carry BOTH clocks (engine.pit.lag_profile). The lag used for
    inference is the given quantile, so a record with no publication time is assumed slow, not fast."""
    from engine.pit import lag_profile
    cal = calendar or Calendar()
    e = pd.to_datetime(pd.Series(effective)).reset_index(drop=True)
    p = pd.to_datetime(pd.Series(published)).reset_index(drop=True)
    src = np.asarray(list(sources))
    lags = []
    for s in sorted(set(src)):
        mask = (src == s) & e.notna().values & p.notna().values
        if int(mask.sum()) < 5:
            continue
        e_s, p_s = e[mask].to_numpy(), p[mask].to_numpy()
        prof = lag_profile(e_s, p_s, cal)
        if prof.empty or float(prof["negative_share"].iloc[0]) > 0.2:
            continue                                             # a source that publishes before it happens is not a lag source
        sessions = np.array([max(0, _sessions_between(pd.Timestamp(a), pd.Timestamp(c), cal)) for a, c in zip(e_s, p_s)])
        lags.append((s, int(math.ceil(float(np.quantile(sessions, quantile))))))
    return LagPolicy(tuple(lags), default=max((l for _, l in lags), default=1), margin_sessions=margin_sessions)


def availability_by_source(judgements: Iterable[AvailabilityJudgement], items: Sequence[InfoItem]) -> pd.DataFrame:
    """Share of each availability class per source. A source whose items are habitually SIMULTANEOUS or ONLY_AFTER is a poor
    input for decisions at a close, however rich its content."""
    src = {i.item_id: i.source for i in items}
    rows = [{"source": src[j.item_id], "availability": j.availability.value} for j in judgements if j.item_id in src]
    if not rows:
        return pd.DataFrame(columns=["source"] + [a.value for a in Availability] + ["n"])
    t = pd.crosstab(pd.DataFrame(rows)["source"], pd.DataFrame(rows)["availability"])
    for a in Availability:
        if a.value not in t.columns:
            t[a.value] = 0
    t = t[[a.value for a in Availability]]
    t["n"] = t.sum(axis=1)
    return t


# ==================================================================================================================
# more adapters: splits, earnings calendars, macro schedules
# ==================================================================================================================
def items_from_splits(splits: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """Recorded corporate actions (columns ticker, ex_date, ratio[, announced]). A split is announced weeks ahead, so when
    `announced` exists the item is a scheduled announcement; without it, publication is unknown (UNCERTAIN, never assumed)."""
    if splits is None or len(splits) == 0:
        return []
    out = []
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    for r in splits[splits["ticker"] == ticker].itertuples(index=False):
        ex = pd.Timestamp(str(r.ex_date)[:10])
        if not lo <= ex <= hi:
            continue
        ann = getattr(r, "announced", None)
        out.append(InfoItem(f"SPLIT-{ticker}-{ex:%Y%m%d}", InfoKind.CORPORATE_ACTION, ticker, str(ex.date()),
                            None if ann is None or pd.isna(ann) else str(ann)[:10], "splits",
                            PROV_RECORDED if ann is not None and not pd.isna(ann) else PROV_UNKNOWN, True, ann is not None,
                            0.9, 0, f"ratio {getattr(r, 'ratio', '?')}"))
    return out


def items_from_earnings_calendar(cal: pd.DataFrame, ticker: str, window: tuple[str, str], strength: float = 0.8) -> list[InfoItem]:
    """Scheduled earnings (columns ticker, event_date, announced). The announcement date is what makes the event knowable."""
    if cal is None or len(cal) == 0:
        return []
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    out = []
    for r in cal[cal["ticker"] == ticker].itertuples(index=False):
        ev = pd.Timestamp(str(r.event_date)[:10])
        if lo <= ev <= hi:
            out.append(scheduled_announcement(f"ER-{ticker}-{ev:%Y%m%d}", InfoKind.EVENT, ticker, str(pd.Timestamp(str(r.announced)[:10]).date()),
                                              str(ev.date()), "earnings_calendar", strength, "scheduled earnings"))
    return out


def macro_schedule(name: str, dates: Sequence[str], lead_days: int = 30, strength: float = 0.4) -> list[InfoItem]:
    """A macro calendar (FOMC, CPI) whose dates are published `lead_days` ahead; each is a scheduled MACRO item on subject MARKET."""
    out = []
    for d in dates:
        t = pd.Timestamp(d)
        out.append(scheduled_announcement(f"MACRO-{name}-{t:%Y%m%d}", InfoKind.MACRO, "MARKET",
                                          str((t - pd.Timedelta(days=lead_days)).date()), str(t.date()), name, strength, name))
    return out


# ==================================================================================================================
# are the channels informative at all? placebo decision dates on the same bars
# ==================================================================================================================
PRICE_CHANNELS = ("PRICE", "VOLUME", "TECHNICAL", "MARKET_STATE")


def price_only_anticipation(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig()) -> float:
    """Anticipation from price-derived channels only (no model, patterns or items), the only ones a placebo date can share."""
    pre, _, _ = inp.windows()
    pre = pre.tail(max(cfg.vol_long * 5, 260))
    ch = {"PRICE": price_channel(pre, cfg), "VOLUME": volume_channel(pre, cfg), "TECHNICAL": technical_channel(pre, cfg),
          "MARKET_STATE": market_state_channel(inp.market, inp.move.decision_date, cfg)}
    return noisy_or([cfg.weight(k) * v.anticipation for k, v in ch.items() if v.anticipation is not None])


def placebo_inputs(inp: MoveInputs, offset: int) -> MoveInputs | None:
    """The same name on an earlier ordinary date, `offset` sessions before the real decision, with its own realised window
    and no items, hits or model output. None when the bars cannot support it or the window would overlap the real move."""
    m = inp.move
    bars = inp.bars.sort_index()
    loc = bars.index.get_loc(pd.Timestamp(m.decision_date))
    j = loc - int(offset)
    if j < 61 or offset < m.horizon + 5:
        return None
    fill, end = bars.index[j + 1], bars.index[j + m.horizon]
    fwd = float(bars["close"].iloc[j + m.horizon] / bars["open"].iloc[j + 1] - 1.0)
    mv = MoveEvent(m.move_id + f"~p{offset}", m.ticker, str(bars.index[j].date()), str(fill.date()), str(end.date()), fwd, m.horizon,
                   m.regime, m.sector)
    return MoveInputs(mv, bars, inp.market, inp.sector)


def discrimination_auc(moves: Sequence[MoveInputs], seed: int = 0, per_move: int = 5, cfg: KnowabilityConfig = KnowabilityConfig(),
                       n_boot: int = 200) -> dict[str, float]:
    """AUC of price-only anticipation for real major moves against placebo dates on the same names (Mann-Whitney), with a
    bootstrap interval over moves. ~0.5 means the channels cannot tell a coming move from an ordinary day: every
    PREDICTABLE label built on them would be noise."""
    rng = np.random.default_rng(seed)
    real, plc, owner = [], [], []
    for k, inp in enumerate(moves):
        real.append(price_only_anticipation(inp, cfg))
        loc = inp.bars.sort_index().index.get_loc(pd.Timestamp(inp.move.decision_date))
        offs = rng.integers(inp.move.horizon + 5, max(inp.move.horizon + 6, loc - 61), per_move)
        for o in offs:
            p = placebo_inputs(inp, int(o))
            if p is not None:
                plc.append(price_only_anticipation(p, cfg))
                owner.append(k)
    if not real or not plc:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_real": len(real), "n_placebo": len(plc)}

    def auc(a, b):
        a, b = np.asarray(a), np.asarray(b)
        gt = (a[:, None] > b[None, :]).mean()
        eq = (a[:, None] == b[None, :]).mean()
        return float(gt + 0.5 * eq)
    base = auc(real, plc)
    owner_arr, plc_arr, real_arr = np.asarray(owner), np.asarray(plc), np.asarray(real)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(real), len(real))
        pl = np.concatenate([plc_arr[owner_arr == p] for p in pick]) if len(pick) else plc_arr
        if len(pl):
            boots.append(auc(real_arr[pick], pl))
    lo, hi = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if boots else (float("nan"), float("nan"))
    return {"auc": base, "lo": lo, "hi": hi, "n_real": len(real), "n_placebo": len(plc)}


# ==================================================================================================================
# which channel carries the class? (necessity by ablation) and classifier self-checks against planted truth
# ==================================================================================================================
def ablate_channels(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), calendar: Calendar | None = None,
                    lag: LagPolicy = LagPolicy()) -> dict[str, Any]:
    """Re-classify with each channel removed in turn. A channel whose removal changes the class is NECESSARY for it; a class
    that no single removal changes is redundantly supported (robust) or, if anticipation is ~0, simply unsupported."""
    base = classify_move(inp, cfg, calendar, lag)
    changed = {}
    for ch in CHANNELS:
        c = classify_move(inp, cfg, calendar, lag, drop_channels=(ch,))
        if c.classification != base.classification:
            changed[ch] = c.classification.value
    return {"base": base.classification.value, "necessary": sorted(changed), "becomes": changed,
            "redundant": not changed and base.anticipation >= cfg.weak_at}


PLANTED_TRUTH = {"PREDICTABLE": Knowability.PREDICTABLE, "POTENTIALLY_PREDICTABLE": Knowability.POTENTIALLY_PREDICTABLE,
                 "WEAKLY_PREDICTABLE": Knowability.WEAKLY_PREDICTABLE, "EXTERNALLY_CAUSED": Knowability.EXTERNALLY_CAUSED,
                 "INFORMATIONALLY_UNAVAILABLE": Knowability.INFORMATIONALLY_UNAVAILABLE, "UNKNOWN": Knowability.UNKNOWN,
                 "DATA_FAILURE": Knowability.DATA_FAILURE, "HINDSIGHT_ONLY": Knowability.UNKNOWN,
                 "UNCERTAIN_EVENT": Knowability.WEAKLY_PREDICTABLE, "BACKFILLED": Knowability.UNKNOWN}


def planted_battery(seeds: Sequence[int] = (0, 1, 2, 3, 4), cfg: KnowabilityConfig = KnowabilityConfig(),
                    kinds: Sequence[str] | None = None) -> dict[str, Any]:
    """Classifier canary: classify every planted kind over several seeds and report the confusion matrix and per-class recall.
    The wave-2 loop runs this after any change to thresholds or channels; a drop in recall is a regression."""
    kinds = list(kinds or PLANTED_TRUTH)
    conf: dict[str, dict[str, int]] = {k: {} for k in kinds}
    for k in kinds:
        for sd in seeds:
            got = classify_move(planted_inputs(k, sd), cfg).classification.value
            conf[k][got] = conf[k].get(got, 0) + 1
    recall = {k: conf[k].get(PLANTED_TRUTH[k].value, 0) / max(1, sum(conf[k].values())) for k in kinds}
    return {"confusion": conf, "recall": recall, "min_recall": min(recall.values()) if recall else float("nan")}


def confidence_calibration(pairs: Sequence[tuple[KnowabilityAssessment, Knowability]], bins: int = 5) -> dict[str, Any]:
    """Is stated confidence honest? Bin assessments by confidence and compare with accuracy against a known planted truth."""
    if not pairs:
        return {"n": 0, "ece": float("nan"), "bins": []}
    conf = np.array([a.confidence_in_classification for a, _ in pairs])
    ok = np.array([a.classification == t for a, t in pairs], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            gap = abs(float(conf[m].mean()) - float(ok[m].mean()))
            ece += gap * float(m.mean())
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(conf[m].mean()), "accuracy": float(ok[m].mean())})
    return {"n": len(pairs), "ece": float(ece), "bins": out, "brier": float(np.mean((conf - ok) ** 2))}


# ==================================================================================================================
# population views: common-cause days, heterogeneity, reclassification, intervals
# ==================================================================================================================
def common_cause_clusters(assessments: Iterable[KnowabilityAssessment], min_names: int = 3) -> list[dict[str, Any]]:
    """Moves that share a decision day and direction usually share a cause. Reports each cluster's class mix; a cluster whose
    members are classified differently (some EXTERNAL, some UNKNOWN) is inconsistent and should be reviewed."""
    groups: dict[tuple, list[KnowabilityAssessment]] = {}
    for a in assessments:
        if a.move_direction:
            groups.setdefault((a.decision_date, a.move_direction), []).append(a)
    out = []
    for (d, s), g in sorted(groups.items()):
        if len(g) < min_names:
            continue
        mix: dict[str, int] = {}
        for a in g:
            mix[a.classification.value] = mix.get(a.classification.value, 0) + 1
        top = max(mix.values())
        out.append({"date": d, "direction": s, "n": len(g), "mix": mix, "agreement": top / len(g), "inconsistent": top / len(g) < 0.6})
    return out


def heterogeneity_test(ledger: KnowabilityLedger, by: str = "regime", n_perm: int = 500, seed: int = 0, min_group: int = 5) -> dict[str, float]:
    """Do class shares differ across groups more than chance? Chi-square statistic with a permutation null (group labels
    shuffled), so small or lumpy groups do not inflate significance."""
    rows = [a for a in ledger.rows()]
    if by == "year":
        grp = [str(as_date(a.decision_date).year) for a in rows]
    elif by == "direction":
        grp = ["UP" if a.move_direction > 0 else "DOWN" if a.move_direction < 0 else "FLAT" for a in rows]
    elif by.startswith("era"):
        grp = _era_labels(rows, int(by[3:] or 3))
    else:
        grp = [str(getattr(a, by)) or "UNSPECIFIED" for a in rows]
    cls = [a.classification.value for a in rows]
    keep_g = {g for g in set(grp) if grp.count(g) >= min_group}
    idx = [i for i, g in enumerate(grp) if g in keep_g]
    if len(keep_g) < 2 or len(idx) < 2 * min_group:
        return {"chi2": float("nan"), "p": float("nan"), "n": len(idx), "groups": len(keep_g)}
    g_arr = np.array([grp[i] for i in idx])
    c_arr = np.array([cls[i] for i in idx])
    def chi2(gs):
        t = pd.crosstab(gs, c_arr).to_numpy().astype(float)
        exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nansum((t - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
    obs = chi2(g_arr)
    rng = np.random.default_rng(seed)
    null = [chi2(rng.permutation(g_arr)) for _ in range(n_perm)]
    return {"chi2": obs, "p": (1 + sum(x >= obs for x in null)) / (n_perm + 1), "n": len(idx), "groups": len(keep_g)}


def class_share_ci(ledger: KnowabilityLedger, cls: Knowability, n_boot: int = 500, seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """Bootstrap interval for one class's share (resampling moves)."""
    rows = ledger.rows()
    if not rows:
        return {"share": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    x = np.array([a.classification == cls for a in rows], dtype=float)
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    return {"share": float(x.mean()), "lo": float(np.quantile(b, a)), "hi": float(np.quantile(b, 1 - a)), "n": len(x)}


def reclassification_report(old: KnowabilityLedger, new: KnowabilityLedger) -> pd.DataFrame:
    """Transition matrix between two runs over the same moves (different config or code). Off-diagonal mass is what a rule
    change actually did; the diagonal is what it left alone. Only moves present in both are compared."""
    a, b = {r.move_id: r.classification.value for r in old.rows()}, {r.move_id: r.classification.value for r in new.rows()}
    common = sorted(set(a) & set(b))
    if not common:
        return pd.DataFrame()
    return pd.crosstab(pd.Series([a[i] for i in common], name="old"), pd.Series([b[i] for i in common], name="new"))


def explain_assessment(a: KnowabilityAssessment) -> str:
    """Human-readable 'could I have known' summary of one move for the research report (no dates or names)."""
    lines = [f"{a.classification.value} (confidence {a.confidence_in_classification:.2f}, anticipation {a.anticipation:.2f}, "
             f"{a.n_present_channels} independent channel(s), coverage {a.coverage:.2f})"]
    lines += [f"  - {t}" for t in a.trace]
    live = [k for k, v in a.knowledge_state_at_decision.items() if v["anticipation"] is not None and v["anticipation"] >= 0.3]
    blind = [k for k, v in a.knowledge_state_at_decision.items() if v["anticipation"] is None]
    if live:
        lines.append(f"  knew from: {', '.join(live)}")
    if blind:
        lines.append(f"  could not see: {', '.join(blind)}")
    if a.explanations:
        lines.append("  causes found in hindsight: " + "; ".join(f"{i} ({av}, precursor {pv})" for i, av, _, pv in a.explanations))
    if a.quality_flags:
        lines.append(f"  data flags: {', '.join(a.quality_flags)}")
    return "\n".join(lines)


# ==================================================================================================================
# persistence: append-only, hash-chained, verified on load (history is immutable)
# ==================================================================================================================
def assessment_to_plain(a: KnowabilityAssessment) -> dict[str, Any]:
    from engine.learning.core import canonical_json
    import json
    return json.loads(canonical_json(a._plain()))


def assessment_from_plain(d: Mapping[str, Any]) -> KnowabilityAssessment:
    tup = lambda x: tuple(x)
    kw = dict(d)
    kw["classification"] = Knowability.parse(kw["classification"])
    kw["namespace"] = Namespace(kw.get("namespace", Namespace.MATURED_RESEARCH.value))
    for f in ("future_information_used_by_auditor", "information_that_would_have_been_available", "information_that_was_unavailable",
              "uncertain_information", "simultaneous_information", "quality_flags", "trace"):
        kw[f] = tup(kw[f])
    kw["explanations"] = tuple(tuple(x) for x in kw["explanations"])
    known = {f.name for f in dataclasses.fields(KnowabilityAssessment)}
    return KnowabilityAssessment(**{k: v for k, v in kw.items() if k in known})


def save_ledger(ledger: KnowabilityLedger, path) -> str:
    """Write the ledger as JSON lines, each carrying the hash of the previous line. Atomic (temp file then replace).
    Returns the head hash."""
    import json, os
    prev = "GENESIS"
    lines = []
    for a in ledger.rows():
        body = assessment_to_plain(a)
        h = stable_hash({"prev": prev, "body": body}, 20)
        lines.append(json.dumps({"prev": prev, "hash": h, "body": body}, sort_keys=True))
        prev = h
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, path)
    return prev


def load_ledger(path) -> KnowabilityLedger:
    """Read and verify the chain. Any tampered, reordered or truncated-in-the-middle line raises."""
    import json, os
    led = KnowabilityLedger()
    if not os.path.exists(path):
        return led
    prev = "GENESIS"
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["prev"] != prev or stable_hash({"prev": prev, "body": rec["body"]}, 20) != rec["hash"]:
                raise KnowabilityError(f"ledger chain broken at line {n}")
            led.add(assessment_from_plain(rec["body"]))
            prev = rec["hash"]
    return led


# ==================================================================================================================
# streaming driver: one move's bars in memory at a time
# ==================================================================================================================
def classify_stream(loader: Callable[[MoveEvent], MoveInputs | None], moves: Iterable[MoveEvent], state: KnowabilityState, now,
                    batch: int = 25, calendar: Calendar | None = None, max_rows: int = 20000) -> list[StepReport]:
    """Classify a large set of major moves without ever holding more than `batch` of them. `loader` builds the inputs for one
    move (bars, items) and may return None (skipped and counted as a data failure of the loader, not silently dropped).
    Inputs bigger than `max_rows` bar rows are refused so a runaway loader cannot exhaust the shared machine."""
    reports, buf, skipped = [], [], []
    def flush():
        if buf:
            _, rep = step(state, now, list(buf), calendar)
            reports.append(rep)
            buf.clear()
    for mv in moves:
        inp = loader(mv)
        if inp is None:
            skipped.append(mv.move_id)
            continue
        if len(inp.bars) > max_rows:
            raise KnowabilityError(f"{mv.move_id}: {len(inp.bars)} bar rows exceeds max_rows={max_rows}")
        buf.append(inp)
        if len(buf) >= batch:
            flush()
    flush()
    if skipped:
        reports.append(StepReport(str(as_date(now)), len(skipped), 0, 0, 0, state.ledger.counts(), None, (), tuple(f"loader returned nothing for {s}" for s in skipped)))
    return reports


# ==================================================================================================================
# building moves from bars; the same move seen at other horizons and earlier decision times
# ==================================================================================================================
def make_move(bars: pd.DataFrame, ticker: str, decision_date, horizon: int = 5, **tags: Any) -> MoveEvent:
    """MoveEvent from bars alone: fill at the next session's open, end at the close `horizon` sessions after the decision."""
    b = bars.sort_index()
    d = pd.Timestamp(as_date(decision_date))
    if d not in b.index:
        raise KnowabilityError(f"{d.date()} is not a session in the bars")
    loc = b.index.get_loc(d)
    if loc + horizon >= len(b):
        raise KnowabilityError("bars end before the move window closes")
    fill, end = b.index[loc + 1], b.index[loc + horizon]
    fwd = float(b["close"].iloc[loc + horizon] / b["open"].iloc[loc + 1] - 1.0)
    mid = "M-" + stable_hash({"t": ticker, "d": str(d.date()), "h": horizon}, 10)
    return MoveEvent(mid, ticker, str(d.date()), str(fill.date()), str(end.date()), fwd, horizon, **tags)


def horizon_profile(inp: MoveInputs, horizons: Sequence[int], cfg: KnowabilityConfig = KnowabilityConfig(),
                    calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> pd.DataFrame:
    """The same decision judged at several horizons. A move can be knowable over a week and unknowable over a day (or the
    reverse); the horizon a pattern claims to serve must be the one its knowability was measured on."""
    bars = inp.bars.sort_index()
    rows = []
    loc = bars.index.get_loc(pd.Timestamp(inp.move.decision_date))
    for h in sorted(set(int(x) for x in horizons)):
        if h < 1 or loc + h >= len(bars):
            continue
        mv = dataclasses.replace(inp.move, end_date=str(bars.index[loc + h].date()), horizon=h,
                                 fwd_return=float(bars["close"].iloc[loc + h] / bars["open"].iloc[loc + 1] - 1.0),
                                 move_id=f"{inp.move.move_id}@h{h}")
        a = classify_move(dataclasses.replace(inp, move=mv), cfg, calendar, lag)
        rows.append({"horizon": h, "fwd_return": mv.fwd_return, "abs_z": a.abs_move_z, "class": a.classification.value,
                     "anticipation": a.anticipation, "confidence": a.confidence_in_classification})
    return pd.DataFrame(rows, columns=["horizon", "fwd_return", "abs_z", "class", "anticipation", "confidence"])


def redecide(inp: MoveInputs, back: int) -> MoveInputs | None:
    """Same move, decided `back` sessions earlier: the decision date and fill move back, the move end date stays, so the
    target is the same event. Model output and the cross-section snapshot belonged to the real decision time and are dropped."""
    bars = inp.bars.sort_index()
    loc = bars.index.get_loc(pd.Timestamp(inp.move.decision_date))
    j = loc - int(back)
    if back < 0 or j < 61:
        return None
    end_loc = bars.index.get_loc(pd.Timestamp(inp.move.end_date))
    fill = bars.index[j + 1]
    mv = dataclasses.replace(inp.move, decision_date=str(bars.index[j].date()), fill_date=str(fill.date()),
                             horizon=end_loc - j, fwd_return=float(bars["close"].iloc[end_loc] / bars["open"].iloc[j + 1] - 1.0),
                             model_pct=inp.move.model_pct if back == 0 else None,
                             model_confidence=inp.move.model_confidence if back == 0 else None,
                             move_id=f"{inp.move.move_id}@-{back}")
    return dataclasses.replace(inp, move=mv, cross=inp.cross if back == 0 else {})


def knowability_onset(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), max_back: int = 10,
                      calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> dict[str, Any]:
    """When did the move become knowable? Rebuild the information state at each earlier decision time and record the total
    anticipation. `lead_sessions` is how far back the anticipation stayed at or above the POTENTIAL bar without a gap; 0 means
    it was only knowable at the actual decision, None means it never was."""
    series = []
    for back in range(0, max_back + 1):
        p = redecide(inp, back)
        if p is None:
            break
        judg = judge_all(p.items, p.move.boundary, calendar, lag)
        A, n, cov = combine_anticipation(build_channels(p, judg, cfg), cfg)
        series.append({"back": back, "anticipation": A, "n_present": n, "coverage": cov})
    lead = None
    for row in series:
        if row["anticipation"] >= cfg.potential_at:
            lead = row["back"]
        else:
            break
    return {"series": pd.DataFrame(series), "lead_sessions": lead}


def lead_time_profile(inp: MoveInputs, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> list[dict[str, Any]]:
    """For each KNOWN_BEFORE item: how many sessions before the fill it was public. Items published the day before the move
    are barely actionable; weeks of lead time is a different kind of knowledge."""
    b = inp.move.boundary
    judg = judge_all(inp.items, b, calendar, lag)
    out = []
    for it in inp.items:
        j = judg[it.item_id]
        if j.availability != Availability.KNOWN_BEFORE_EVENT:
            continue
        pub, _ = to_ny(it.published_at)
        if pub is None:
            pub = pd.Timestamp((calendar or Calendar()).shift([to_ny(it.effective_at)[0]], lag.lag_for(it.source))[0])
        out.append({"item_id": it.item_id, "kind": it.kind.value, "source": it.source,
                    "sessions_before_fill": _sessions_between(pub, pd.Timestamp(b.fill_date), calendar), "inferred": j.inferred})
    return sorted(out, key=lambda r: (-r["sessions_before_fill"], r["item_id"]))


def information_timeline(inp: MoveInputs, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> pd.DataFrame:
    """Every item on one time axis against the decision boundary: the review artefact behind a classification."""
    b = inp.move.boundary
    judg = judge_all(inp.items, b, calendar, lag)
    rows = []
    for it in inp.items:
        j = judg[it.item_id]
        pub, _ = to_ny(it.published_at)
        rows.append({"item_id": it.item_id, "kind": it.kind.value, "source": it.source, "availability": j.availability.value,
                     "published": None if pub is None else str(pub), "minutes_from_cutoff": j.margin_minutes, "strength": it.strength,
                     "direction": it.direction, "reason": j.reason})
    df = pd.DataFrame(rows, columns=["item_id", "kind", "source", "availability", "published", "minutes_from_cutoff", "strength",
                                     "direction", "reason"])
    return df.sort_values(["minutes_from_cutoff", "item_id"], na_position="last").reset_index(drop=True)


# ==================================================================================================================
# explanation base rates and expected-reaction attribution (an event is a weak explanation if it is always there)
# ==================================================================================================================
def kind_base_rates(items: Sequence[InfoItem], span: tuple[str, str], window_days: int = 8, n_windows: int = 400,
                    seed: int = 0) -> dict[str, float]:
    """Probability that a random window of `window_days` calendar days inside `span` contains at least one item of each key
    (item_key). Counted on the name's own history, so a name that files an 8-K every fortnight gets weak explanations from one."""
    lo, hi = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    total = (hi - lo).days - window_days
    if total <= 0:
        raise KnowabilityError("span shorter than the window")
    starts = lo + pd.to_timedelta(np.random.default_rng(seed).integers(0, total, n_windows), unit="D")
    eff = {}
    for it in items:
        t, _ = to_ny(it.effective_at)
        if t is not None:
            eff.setdefault(item_key(it), []).append(t.normalize().value)
    out = {}
    for key, ts in eff.items():
        arr = np.sort(np.array(ts))
        left = np.searchsorted(arr, starts.values.astype("datetime64[ns]").astype("int64"), side="left")
        right = np.searchsorted(arr, (starts + pd.Timedelta(days=window_days)).values.astype("datetime64[ns]").astype("int64"), side="right")
        out[key] = float(np.mean(right > left))
    return out


@dataclasses.dataclass(frozen=True)
class MoveAttribution:
    """How the move splits into what markets explain, what the event history predicts, and what is left."""
    total: float
    systematic: float
    event_expected: float
    residual: float
    residual_z: float | None
    n_event_history: int


def event_reaction_history(inp: MoveInputs, item: InfoItem, window: int = 2) -> list[float]:
    """|return| over `window` sessions from the effective date of every EARLIER item with the same key, strictly before the
    decision date less the window (their outcomes were public at the decision)."""
    bars = inp.bars.sort_index()
    cut = pd.Timestamp(inp.move.decision_date) - pd.Timedelta(days=window + 2)
    out = []
    for other in inp.items:
        if other.item_id == item.item_id or item_key(other) != item_key(item):
            continue
        t, _ = to_ny(other.effective_at)
        if t is None or t + pd.Timedelta(days=window + 2) > cut:
            continue
        idx = bars.index.searchsorted(t.normalize())
        if idx < 1 or idx + window >= len(bars):
            continue
        out.append(abs(float(bars["close"].iloc[idx + window] / bars["close"].iloc[idx - 1] - 1.0)))
    return out


def attribute_move(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig()) -> MoveAttribution:
    """Split the move: market/sector part (betas from before the decision), the typical reaction to the strongest in-window
    event (median of this name's own earlier reactions to the same kind), and the residual in prior-volatility units. A large
    residual after both is the audit's honest 'still unexplained' number."""
    m = inp.move
    ext = decompose_external(inp, cfg)
    sysm = 0.0 if ext is None else ext.explained
    pre, _, _ = inp.windows()
    lo, hi = pd.Timestamp(m.decision_date) - pd.Timedelta(days=5), pd.Timestamp(m.end_date)
    best_expected, n_hist = 0.0, 0
    for it in inp.items:
        t, _ = to_ny(it.effective_at)
        if it.kind not in (InfoKind.EVENT, InfoKind.FILING) or t is None or not lo <= t.normalize() <= hi:
            continue
        hist = event_reaction_history(inp, it)
        if len(hist) >= 3 and float(np.median(hist)) > best_expected:
            best_expected, n_hist = float(np.median(hist)), len(hist)
    remaining = m.fwd_return - sysm
    event = float(np.sign(remaining) * min(best_expected, abs(remaining))) if remaining else 0.0
    resid = remaining - event
    pv = _prior_vol(pre, cfg.vol_long)
    z = resid / (pv * math.sqrt(m.horizon)) if pv else None
    return MoveAttribution(m.fwd_return, sysm, event, resid, z, n_hist)


# ==================================================================================================================
# timestamp-noise robustness
# ==================================================================================================================
def perturb_timestamps(items: Sequence[InfoItem], sigma_minutes: float, seed: int) -> list[InfoItem]:
    """Jitter the recorded intraday publication times (clock skew, vendor lag). DATE-precision times cannot be jittered
    honestly and are left alone."""
    rng = np.random.default_rng(seed)
    out = []
    for it in items:
        pub, prec = to_ny(it.published_at)
        if pub is not None and prec == PREC_INTRADAY and sigma_minutes > 0:
            pub = pub + pd.Timedelta(minutes=float(rng.normal(0.0, sigma_minutes)))
            it = dataclasses.replace(it, published_at=str(pub))
        out.append(it)
    return out


def timestamp_robustness(inp: MoveInputs, sigmas: Sequence[float] = (15.0, 60.0, 240.0), n: int = 12, seed: int = 0,
                         cfg: KnowabilityConfig = KnowabilityConfig(), calendar: Calendar | None = None) -> dict[float, float]:
    """Share of clock-noise draws that leave the class unchanged, per noise level. A class that does not survive a few
    minutes of skew is decided by the timestamps' precision, not by the world, and deserves a confidence penalty."""
    base = classify_move(inp, cfg, calendar).classification
    out = {}
    for si, sg in enumerate(sigmas):
        same = 0
        for k in range(n):
            alt = dataclasses.replace(inp, items=perturb_timestamps(inp.items, sg, seed * 1000 + si * 100 + k))
            same += int(classify_move(alt, cfg, calendar).classification == base)
        out[float(sg)] = same / n
    return out


# ==================================================================================================================
# direction knowability and population diagnostics
# ==================================================================================================================
class DirectionVerdict(_StrEnum):
    AGREES = "AGREES"
    CONTRADICTED = "CONTRADICTED"
    MIXED = "MIXED"
    NO_DIRECTION_INFO = "NO_DIRECTION_INFO"


def direction_verdict(a: KnowabilityAssessment, cfg: KnowabilityConfig = KnowabilityConfig()) -> dict[str, Any]:
    """Did the information available at the decision point the way the move went? An anticipation-weighted vote of the channels
    that carry a direction. This is a separate question from 'was a big move coming', and most moves fail it."""
    num = den = 0.0
    n = 0
    for ch, st in a.knowledge_state_at_decision.items():
        if st["anticipation"] is None or not st["direction"]:
            continue
        w = cfg.weight(ch) * st["anticipation"]
        num += w * st["direction"]
        den += w
        n += 1
    if n == 0 or den <= 0 or a.move_direction == 0:
        return {"verdict": DirectionVerdict.NO_DIRECTION_INFO.value, "vote": 0.0, "n_channels": n}
    vote = num / den
    agree = vote * a.move_direction
    v = DirectionVerdict.AGREES if agree >= 0.5 else DirectionVerdict.CONTRADICTED if agree <= -0.5 else DirectionVerdict.MIXED
    return {"verdict": v.value, "vote": float(vote), "n_channels": n}


def symmetry_report(ledger: KnowabilityLedger, n_perm: int = 300, seed: int = 0) -> dict[str, Any]:
    """Winners versus losers: do they differ in how knowable they were? Section 5 says losers are studied as hard as winners."""
    return {"table": ledger.table("direction"), "test": heterogeneity_test(ledger, "direction", n_perm, seed, min_group=3)}


def coverage_report(ledger: KnowabilityLedger) -> dict[str, float]:
    """Share of classified moves for which each channel had ANY data. A channel that is nearly always blind (no model output,
    no cross-section) makes every class in the ledger a statement about the remaining channels only."""
    rows = ledger.rows()
    if not rows:
        return {c: float("nan") for c in CHANNELS}
    return {c: sum(1 for a in rows if a.knowledge_state_at_decision.get(c, {}).get("anticipation") is not None) / len(rows) for c in CHANNELS}


def to_frame(ledger: KnowabilityLedger) -> pd.DataFrame:
    """Flat, identity-free-by-choice table of the ledger (research side: ticker and dates included)."""
    cols = ["move_id", "ticker", "decision_date", "classification", "anticipation", "n_present_channels", "coverage",
            "systematic_share", "direction_knowable", "confidence_in_classification", "regime", "sector", "vol_bucket",
            "move_direction", "fwd_return", "abs_move_z"]
    rows = [{c: (getattr(a, c).value if isinstance(getattr(a, c), Knowability) else getattr(a, c)) for c in cols} for a in ledger.rows()]
    return pd.DataFrame(rows, columns=cols)


# ==================================================================================================================
# semantic audit: does every stored class obey the rules that define it?
# ==================================================================================================================
def validate_assessment_semantics(a: KnowabilityAssessment, cfg: KnowabilityConfig = KnowabilityConfig()) -> list[str]:
    """Cross-field invariants of the cascade. A violation means the record was edited, produced under other thresholds, or the
    classifier changed without the ledger being rebuilt."""
    v = []
    demoted = any("demoted" in t for t in a.trace)
    c = a.classification
    if c == Knowability.PREDICTABLE:
        if a.n_present_channels < 2 or a.anticipation < cfg.predictable_at or (a.model_pct or 0.0) < cfg.flag_pct:
            v.append("PREDICTABLE without >=2 channels, anticipation above the bar and a flagging model")
    elif c == Knowability.POTENTIALLY_PREDICTABLE and not demoted:
        if a.anticipation < cfg.potential_at:
            v.append("POTENTIALLY_PREDICTABLE below the potential bar")
        if a.anticipation < cfg.predictable_at and a.n_present_channels < 2:
            v.append("POTENTIALLY_PREDICTABLE from a single channel below the predictable bar")
    elif c == Knowability.WEAKLY_PREDICTABLE and not demoted and a.anticipation < cfg.weak_at and not any("precursors" in t for t in a.trace):
        v.append("WEAKLY_PREDICTABLE below the weak bar")
    elif c == Knowability.EXTERNALLY_CAUSED and (a.systematic_share is None or a.systematic_share < cfg.ext_share):
        v.append("EXTERNALLY_CAUSED without a systematic share above the bar")
    elif c == Knowability.INFORMATIONALLY_UNAVAILABLE:
        if not any(e[3] == Availability.UNAVAILABLE.value for e in a.explanations):
            v.append("INFORMATIONALLY_UNAVAILABLE without an explanation whose precursors were verified absent")
        if a.anticipation >= cfg.weak_at:
            v.append("INFORMATIONALLY_UNAVAILABLE though channels anticipated the move")
    elif c == Knowability.UNKNOWN and a.anticipation >= cfg.potential_at and not demoted:
        v.append("UNKNOWN despite anticipation above the potential bar")
    elif c == Knowability.DATA_FAILURE and not a.quality_flags:
        v.append("DATA_FAILURE with no flag")
    if c in (Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE) and a.coverage <= 0:
        v.append("a predictability claim with zero channel coverage")
    if a.namespace != Namespace.MATURED_RESEARCH:
        v.append("namespace is not MATURED_RESEARCH")
    return v


def audit_ledger(ledger: KnowabilityLedger, cfg: KnowabilityConfig = KnowabilityConfig()) -> dict[str, list[str]]:
    """move_id -> violations, for every stored assessment that breaks its own rules (empty dict = consistent)."""
    return {a.move_id: e for a in ledger.rows() if (e := validate_assessment_semantics(a, cfg))}


# ==================================================================================================================
# checkpointing (a killed job costs minutes, not hours)
# ==================================================================================================================
def checkpoint_state(state: KnowabilityState, directory) -> str:
    """Persist ledger + snapshots + immature list + the config hash they were produced under."""
    import json, os
    os.makedirs(directory, exist_ok=True)
    head = save_ledger(state.ledger, os.path.join(directory, "ledger.jsonl"))
    meta = {"config_hash": state.cfg.hash(), "head": head, "immature": sorted(state.immature),
            "snapshots": {k: dataclasses.asdict(v) for k, v in sorted(state.snapshots.items())}}
    tmp = os.path.join(directory, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, sort_keys=True)
    os.replace(tmp, os.path.join(directory, "meta.json"))
    return head


def resume_state(directory, cfg: KnowabilityConfig = KnowabilityConfig(), lag: LagPolicy = LagPolicy()) -> KnowabilityState:
    """Reload a checkpoint. Resuming under different thresholds is refused: two rule sets in one ledger would make its class
    shares meaningless."""
    import json, os
    mp = os.path.join(directory, "meta.json")
    if not os.path.exists(mp):
        return KnowabilityState(cfg, lag)
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    if meta["config_hash"] != cfg.hash():
        raise KnowabilityError(f"checkpoint written under config {meta['config_hash']}, resuming under {cfg.hash()}")
    led = load_ledger(os.path.join(directory, "ledger.jsonl"))
    snaps = {k: DaySnapshot(**v) for k, v in meta["snapshots"].items()}
    return KnowabilityState(cfg, lag, led, snaps, list(meta["immature"]))


# ==================================================================================================================
# decision-time features: as-of stamps and the leak check on what the model saw
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class FeatureValue:
    """One model input at the decision: its value, the moment the underlying data became public (`asof`), and how much the model
    leans on it. `z` is the value in units of its own trailing spread, so a boring feature cannot anticipate anything."""
    name: str
    value: float
    asof: str
    source: str = "features"
    importance: float = 0.0
    z: float | None = None

    def check(self) -> list[str]:
        errs = []
        if not self.name:
            errs.append("feature without a name")
        if not math.isfinite(self.value):
            errs.append(f"{self.name}: non-finite value")
        if not 0.0 <= self.importance <= 1.0:
            errs.append(f"{self.name}: importance outside [0,1]")
        if to_ny(self.asof)[0] is None:
            errs.append(f"{self.name}: no usable as-of time")
        return errs


def feature_items(features: Iterable[FeatureValue], subject: str) -> list[InfoItem]:
    """Feature snapshot -> FEATURE items (published = as-of). Strength is importance times how unusual the value was, capped."""
    out = []
    for f in features:
        errs = f.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        strength = min(1.0, f.importance * min(1.0, abs(f.z) / 3.0)) if f.z is not None else 0.0
        out.append(InfoItem(f"FEAT-{subject}-{f.name}", InfoKind.FEATURE, subject, f.asof, f.asof, f.source, PROV_RECORDED,
                            True, False, strength, int(np.sign(f.z)) if f.z else 0, f"{f.name}={f.value:.4g}"))
    return out


def feature_leaks(features: Iterable[FeatureValue], b: DecisionBoundary, calendar: Calendar | None = None) -> list[str]:
    """Names of features whose as-of time is not strictly before the decision cutoff. Any entry means the trader's own inputs
    at this timestamp contained something it could not have had: a bug in the decision pipeline, not a knowability question."""
    bad = []
    for f in features:
        j = judge_item(feature_items([f], "X")[0], b, calendar)
        if j.availability != Availability.KNOWN_BEFORE_EVENT:
            bad.append(f"{f.name} ({j.availability.value})")
    return sorted(bad)


def assert_features_clean(features: Iterable[FeatureValue], b: DecisionBoundary, calendar: Calendar | None = None) -> None:
    bad = feature_leaks(features, b, calendar)
    if bad:
        raise FirewallBreach(f"decision-time features not public before the cutoff: {bad}")


def pattern_hits_from_frame(patterns: pd.DataFrame, matches: Callable[[Any], bool], decision_date: str) -> list[dict[str, Any]]:
    """Pattern-miner rows (pattern_id, p_real, effect, status, learned_at) -> hits the auditor may credit: ACTIVE, matching the
    decision-time situation, and learned strictly before the decision. Later-learned patterns are returned with learned_at
    intact so the channel excludes them and reports them as future information."""
    if patterns is None or len(patterns) == 0:
        return []
    need = {"pattern_id", "p_real", "effect", "status", "learned_at"}
    if not need <= set(patterns.columns):
        raise KnowabilityError(f"pattern frame lacks {sorted(need - set(patterns.columns))}")
    out = []
    for r in patterns.itertuples(index=False):
        if str(r.status) != "active" or not matches(r):
            continue
        out.append({"id": str(r.pattern_id), "p_real": float(r.p_real), "effect": float(r.effect), "strength": 1.0,
                    "learned_at": str(r.learned_at)[:10]})
    return out


# ==================================================================================================================
# more data-failure detail: split adjustment, overnight gap, look-ahead in the join
# ==================================================================================================================
def split_adjustment_check(bars: pd.DataFrame, items: Sequence[InfoItem], tol: float = 0.05) -> list[QualityFlag]:
    """A recorded split must be invisible in adjusted bars. If the close jumps by the split factor across the ex-date the bars
    are UNADJUSTED and every return spanning it is an artefact. Item detail carries 'ratio X' (X-for-1)."""
    out = []
    b = bars.sort_index()
    for it in items:
        if it.kind != InfoKind.CORPORATE_ACTION or "ratio" not in it.detail:
            continue
        try:
            ratio = float(it.detail.split("ratio")[-1].split()[0])
        except (ValueError, IndexError):
            continue
        ex, _ = to_ny(it.effective_at)
        if ex is None or ratio <= 0 or ex not in b.index:
            continue
        loc = b.index.get_loc(ex)
        if loc < 1:
            continue
        jump = float(b["close"].iloc[loc] / b["close"].iloc[loc - 1])
        if abs(jump * ratio - 1.0) < tol:
            out.append(QualityFlag("UNADJUSTED_SPLIT", 0.95, f"close changed by x{jump:.3f} across a recorded {ratio:g}-for-1 split"))
    return out


def overnight_gap(inp: MoveInputs, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy(), z_bar: float = 2.0) -> dict[str, Any]:
    """The close-to-open gap at the fill. It is outside the labelled return (which starts at the open) but it is where news
    that arrived after the cutoff lands. A gap beyond z_bar prior sigmas with SIMULTANEOUS items explains itself; a gap with no
    item at all is an unexplained overnight event."""
    pre, win, _ = inp.windows()
    if len(pre) < 2 or len(win) == 0:
        return {"gap": None, "gap_z": None, "news": (), "explained": None}
    gap = float(win["open"].iloc[0] / pre["close"].iloc[-1] - 1.0)
    pv = _prior_vol(pre, 60)
    z = gap / pv if pv else None
    judg = judge_all(inp.items, inp.move.boundary, calendar, lag)
    news = tuple(sorted(i for i, j in judg.items() if j.availability == Availability.SIMULTANEOUS))
    big = z is not None and abs(z) >= z_bar
    return {"gap": gap, "gap_z": z, "news": news, "explained": (bool(news) if big else None)}


def naive_items(items: Sequence[InfoItem]) -> list[InfoItem]:
    """What a join on the EFFECTIVE date sees: publication := effective. This is the look-ahead trap in miniature, kept only to
    measure how much it would have leaked (never used by the classifier)."""
    return [dataclasses.replace(it, published_at=it.effective_at, provenance=PROV_RECORDED) if it.effective_at is not None else it for it in items]


def naive_vs_pit_gap(inp: MoveInputs, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> dict[str, Any]:
    """Items a naive effective-date join would call known that were really public only later, plus engine.pit.visibility_gap
    on the two clocks. Non-empty `leaked` means a pipeline built that way would have looked ahead on this move."""
    from engine.pit import visibility_gap
    b = inp.move.boundary
    real = judge_all(inp.items, b, calendar, lag)
    naive = judge_all(naive_items(inp.items), b, calendar, lag)
    leaked = sorted(i for i in real if naive[i].availability == Availability.KNOWN_BEFORE_EVENT
                    and real[i].availability != Availability.KNOWN_BEFORE_EVENT)
    timed = [it for it in inp.items if it.effective_at and it.published_at]
    gap = pd.DataFrame()
    if timed:
        gap = visibility_gap([pd.Timestamp(str(i.effective_at)[:10]) for i in timed], [to_ny(i.published_at)[0] for i in timed],
                             by=[i.source for i in timed], calendar=calendar)
    return {"leaked": leaked, "gap": gap}


def classify_naive(inp: MoveInputs, cfg: KnowabilityConfig = KnowabilityConfig(), calendar: Calendar | None = None) -> Knowability:
    """The class a look-ahead join would have produced. Comparing it with the real class quantifies the leak per move."""
    return classify_move(dataclasses.replace(inp, items=naive_items(inp.items)), cfg, calendar).classification


# ==================================================================================================================
# population statistics that respect clustering (many big movers share a day) and eras
# ==================================================================================================================
def day_cluster_ci(ledger: KnowabilityLedger, cls: Knowability, n_boot: int = 500, seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """Bootstrap a class share resampling whole DECISION DAYS, because movers on one day share market conditions. Also returns
    the design effect against the iid interval: > 1 means the plain per-move interval is too optimistic by that factor."""
    rows = ledger.rows()
    if not rows:
        return {"share": float("nan"), "lo": float("nan"), "hi": float("nan"), "design_effect": float("nan"), "effective_n": 0.0}
    days: dict[str, list[int]] = {}
    for a in rows:
        days.setdefault(a.decision_date, []).append(int(a.classification == cls))
    keys = sorted(days)
    hits = np.array([sum(days[k]) for k in keys], dtype=float)
    size = np.array([len(days[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boot.append(hits[pick].sum() / size[pick].sum())
    boot_arr = np.array(boot)
    p = float(hits.sum() / size.sum())
    var_iid = p * (1 - p) / size.sum()
    deff = float(boot_arr.var() / var_iid) if var_iid > 0 else float("nan")
    a = (1.0 - level) / 2.0
    return {"share": p, "lo": float(np.quantile(boot_arr, a)), "hi": float(np.quantile(boot_arr, 1 - a)), "design_effect": deff,
            "effective_n": float(size.sum() / deff) if deff and deff > 0 else float("nan")}


def year_consistency(ledger: KnowabilityLedger, min_per_year: int = 10) -> dict[str, Any]:
    """Consistent = holds in every year (C58-C61). For each class: its share in every year with at least min_per_year moves,
    the min and max across years, and the range. A class share that lives in one year is not a property of markets."""
    by_year: dict[int, list[KnowabilityAssessment]] = {}
    for a in ledger.rows():
        by_year.setdefault(as_date(a.decision_date).year, []).append(a)
    years = {y: g for y, g in by_year.items() if len(g) >= min_per_year}
    out = {}
    for k in Knowability:
        shares = {y: sum(1 for a in g if a.classification == k) / len(g) for y, g in sorted(years.items())}
        out[k.value] = {"by_year": shares, "min": min(shares.values()) if shares else float("nan"),
                        "max": max(shares.values()) if shares else float("nan"),
                        "range": (max(shares.values()) - min(shares.values())) if shares else float("nan")}
    return {"years_used": sorted(years), "classes": out}


def knowability_ceiling(ledger: KnowabilityLedger) -> dict[str, Any]:
    """Upper bounds on what a decision-time system can capture. reachable = PREDICTABLE + POTENTIALLY_PREDICTABLE (information
    existed); weak adds the marginal class; structural = EXTERNAL + UNAVAILABLE + DATA_FAILURE (no decision-time information can
    help); unresolved = UNKNOWN. Wilson intervals on each."""
    n = len(ledger)
    c = ledger.counts()
    def share(*ks: Knowability) -> dict[str, float]:
        k = sum(c[x.value] for x in ks)
        lo, hi = wilson(k, n)
        return {"share": k / n if n else float("nan"), "lo": lo, "hi": hi, "n": k}
    return {"n_moves": n,
            "reachable": share(Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE),
            "reachable_incl_weak": share(Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE),
            "structural": share(Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE, Knowability.DATA_FAILURE),
            "unresolved": share(Knowability.UNKNOWN)}


# ==================================================================================================================
# outputs for the research loop (questions) and the trader (gated, aggregate, identity-free priors)
# ==================================================================================================================
def research_targets(ledger: KnowabilityLedger, created_real: str, cfg: KnowabilityConfig = KnowabilityConfig(),
                     min_moves: int = 30) -> list[Any]:
    """Turn knowability findings into ResearchQuestions (identity-free text, evidence dated by the newest matured outcome).
    Each carries a success and a failure criterion and a first estimate of value; the wave-2 loop prices them."""
    from engine.research.core import ExperimentValue, Problem, ResearchQuestion
    rows = ledger.rows()
    if len(rows) < min_moves:
        return []
    through = max(a.matured_at for a in rows)
    n = len(rows)
    c = ledger.counts()
    qs = []
    gap = model_gap(ledger, cfg)
    if gap["potential"] >= 5 and gap["unflagged_share"] >= 0.3:
        ig = list(gap["channels_the_model_ignored"])[:3]
        qs.append(ResearchQuestion.make(
            f"Why did the model not flag {gap['unflagged']} of {gap['potential']} moves whose information existed beforehand "
            f"(ignored channels: {', '.join(ig) or 'none identified'})?", "knowability_gap", Problem.VOLATILITY, created_real, through,
            "a candidate feature built from the ignored channels raises out-of-sample detection of these moves",
            "the added feature does not raise detection on unseen years", expected=ExperimentValue(
                information_gain=0.6, decision_value=0.7, transfer_potential=0.5, volatility_value=0.7, compute_cost=30.0)))
    if c[Knowability.DATA_FAILURE.value] / n >= 0.02:
        flags: dict[str, int] = {}
        for a in rows:
            for f in a.quality_flags:
                flags[f] = flags.get(f, 0) + 1
        top = max(flags, key=flags.get) if flags else "unspecified"
        qs.append(ResearchQuestion.make(
            f"{c[Knowability.DATA_FAILURE.value]} of {n} major moves are data failures (most common flag {top}); which source produces them?",
            "knowability_data", Problem.DATA_QUALITY, created_real, through, "the defect source is identified and the flag rate falls after the fix",
            "the flag rate is unchanged after the suspected source is corrected",
            expected=ExperimentValue(information_gain=0.5, decision_value=0.4, failure_reduction_value=0.6, compute_cost=10.0)))
    led = sum(1 for a in rows if any("PRICE_LED_NEWS" in t for t in a.trace))
    if led >= 3:
        qs.append(ResearchQuestion.make(
            f"In {led} moves the price moved before the explaining item was public; is this leakage, informed trading or bad timestamps?",
            "knowability_timestamps", Problem.DATA_QUALITY, created_real, through,
            "the offset is reproduced under an independent timestamp source", "the offset disappears when timestamps are re-derived",
            expected=ExperimentValue(information_gain=0.6, decision_value=0.3, failure_reduction_value=0.5, compute_cost=15.0)))
    if c[Knowability.INFORMATIONALLY_UNAVAILABLE.value] / n >= 0.15:
        qs.append(ResearchQuestion.make(
            "A large share of moves had causes that no available source foresaw; which additional data source would have revealed them?",
            "knowability_coverage", Problem.COVERAGE, created_real, through,
            "a candidate source shows a precursor before the boundary for a majority of these moves",
            "no candidate source shows precursors", expected=ExperimentValue(information_gain=0.5, decision_value=0.5, transfer_potential=0.3, compute_cost=60.0)))
    cov = coverage_report(ledger)
    blind = [k for k, v in cov.items() if v < 0.5]
    if c[Knowability.UNKNOWN.value] / n >= 0.4 and blind:
        qs.append(ResearchQuestion.make(
            f"Most unknown classes coincide with blind channels ({', '.join(blind)}); how much of UNKNOWN is just missing inputs?",
            "knowability_blind", Problem.DATA_QUALITY, created_real, through,
            "supplying the blind channels moves a large share of UNKNOWN into a known class",
            "the unknown share is unchanged when blind channels are filled", expected=ExperimentValue(information_gain=0.6, decision_value=0.4, compute_cost=20.0)))
    bad = [x for x in common_cause_clusters(ledger.rows()) if x["inconsistent"]]
    if bad:
        qs.append(ResearchQuestion.make(
            f"{len(bad)} same-day same-direction clusters are classified inconsistently; is the classifier unstable or are the causes different?",
            "knowability_consistency", Problem.CONSISTENCY, created_real, through,
            "cluster members converge on one class after the rule is fixed", "members still disagree with identical inputs",
            expected=ExperimentValue(information_gain=0.4, decision_value=0.3, failure_reduction_value=0.3, compute_cost=8.0)))
    return qs


def regime_knowability_prior(ledger: KnowabilityLedger, now, replaying_years: Iterable[int] = (), min_n: int = 20) -> dict[str, Any]:
    """Aggregate, identity-free knowledge for the curator to release: per regime tag, class shares (quarter-rounded) over
    assessments that matured strictly before real `now` and whose decision year is not being replayed in disguise. Groups
    below min_n are withheld (too few to be knowledge). The trader never sees this except through the curator."""
    replay = {int(y) for y in replaying_years}
    usable, immature, replayed = [], 0, 0
    for a in ledger.rows():
        if as_date(a.decision_date).year in replay:
            replayed += 1
            continue
        try:
            a.to_record().gate(now)
        except FirewallBreach:
            immature += 1
            continue
        usable.append(a)
    groups: dict[str, list[KnowabilityAssessment]] = {}
    for a in usable:
        groups.setdefault(a.regime or "UNSPECIFIED", []).append(a)
    prior = {}
    for g, items in sorted(groups.items()):
        if len(items) < min_n:
            continue
        prior[g] = {k.value: round(sum(1 for a in items if a.classification == k) / len(items) * 4) / 4 for k in Knowability}
        prior[g]["n_bucket"] = 10 ** int(math.log10(len(items)))
    return {"prior": prior, "withheld_groups": sum(1 for v in groups.values() if len(v) < min_n), "immature": immature,
            "replayed_year_excluded": replayed}


# ==================================================================================================================
# real-data availability adapters: one per source family in data/cache. Each maps the file's OWN columns to InfoItems and says,
# per row, which clock is trustworthy. Frames are passed in (never read here), so tests use small synthetic frames of the same shape.
# ==================================================================================================================
SOURCE_FAMILIES = ("edgar_events", "insider", "macro", "prices", "delisted")
CLOSE_TIME = "16:00:00"


def _win(window: tuple[str, str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if hi < lo:
        raise KnowabilityError("window ends before it starts")
    return lo, hi


def edgar_events_items(events: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """data/cache/events.parquet: columns ticker, form, accepted (UTC), kind. `accepted` is the SEC acceptance stamp, so
    publication is INTRADAY-precision and recorded. Acceptance after 17:30 ET is dated the next business day by the SEC, which
    only makes it later, never earlier, so using the stamp as published_at is conservative. Delegates to items_from_edgar_events
    (the single implementation) after checking the frame shape and the timezone."""
    if events is None or len(events) == 0:
        return []
    if "accepted" in events and getattr(events["accepted"].dtype, "tz", None) is None:
        raise KnowabilityError("events.accepted must be tz-aware UTC; a naive stamp cannot be ordered against the close")
    return items_from_edgar_events(events, ticker, window)


def insider_items(ins: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """data/cache/insider.parquet: columns acc, filed (date), tdate (transaction date), symbol, shares, price. `filed` carries no
    time of day, so publication is DATE precision and a same-day filing is SIMULTANEOUS at best. Strength grows with the dollar
    size of the purchase (log-scaled, capped) because a token purchase is not a signal. A row filed BEFORE its trade date is
    impossible and is emitted with no publication time and unknown provenance (UNCERTAIN) rather than trusted."""
    if ins is None or len(ins) == 0:
        return []
    need = {"filed", "tdate", "symbol"}
    if not need <= set(ins.columns):
        raise KnowabilityError(f"insider frame lacks {sorted(need - set(ins.columns))}")
    lo, hi = _win(window)
    out = []
    sub = ins[ins["symbol"] == ticker]
    for i, r in enumerate(sub.itertuples(index=False)):
        rec = r._asdict()
        f = pd.Timestamp(rec["filed"])
        t = pd.Timestamp(rec["tdate"]) if pd.notna(rec["tdate"]) else None
        if not lo <= f.normalize() <= hi:
            continue
        dollars = float(rec.get("shares") or 0.0) * float(rec.get("price") or 0.0)
        strength = float(min(0.6, max(0.05, (math.log10(dollars) - 3.0) / 5.0))) if dollars > 0 else 0.05
        impossible = t is not None and f.normalize() < t.normalize()
        acc = str(rec.get("acc", i))
        out.append(InfoItem(f"INS-{ticker}-{acc}-{i}", InfoKind.INSIDER, ticker, None if t is None else str(t.date()),
                            None if impossible else str(f.date()), "insider", PROV_UNKNOWN if impossible else PROV_RECORDED, True,
                            False, strength, 1, "open-market purchase" + (" (filed before trade date)" if impossible else "")))
    return out


def macro_items(macro: pd.DataFrame, window: tuple[str, str], lag_days: Mapping[str, int] | None = None, market_priced_lag: int = 1) -> list[InfoItem]:
    """data/cache/macro.parquet: wide frame, index = OBSERVATION date, one column per FRED series. There is no publication
    column, so publication is INFERRED: observation date plus the series' calendar-day publication lag (engine.leak_audit
    MACRO_LAG_DAYS), one day for market-priced series, 45 days for unknown ones. Revised series (MACRO_REVISED) get provenance
    'reconstructed': the stored VALUE is the current vintage, not what was first published, so timing is usable but the level is
    hindsight. Each item's detail carries the series and the change from the prior observation (a level is not an event)."""
    from engine import leak_audit as LA
    if macro is None or len(macro) == 0:
        return []
    lo, hi = _win(window)
    lags = dict(LA.MACRO_LAG_DAYS)
    lags.update(lag_days or {})
    out = []
    for col in macro.columns:
        s = macro[col].dropna()
        if s.empty:
            continue
        lag = lags.get(col, market_priced_lag if col in LA.MACRO_UNREVISED else 45)
        revised = col in LA.MACRO_REVISED or col not in LA.MACRO_UNREVISED
        chg = s.diff()
        sd = float(chg.std()) if len(chg) > 2 and float(chg.std()) > 0 else 0.0
        for obs, v in s.items():
            pub = (pd.Timestamp(obs) + pd.Timedelta(days=int(lag))).normalize()
            if not lo <= pub <= hi:
                continue
            z = float(chg.loc[obs] / sd) if sd and pd.notna(chg.loc[obs]) else 0.0
            out.append(InfoItem(f"MACRO-{col}-{pd.Timestamp(obs):%Y%m%d}", InfoKind.MACRO, "MARKET", str(pd.Timestamp(obs).date()),
                                str(pub.date()), col, PROV_RECONSTRUCTED if revised else PROV_INFERRED, True, False,
                                float(min(0.6, abs(z) / 6.0)), int(np.sign(z)), f"{col} change z={z:+.2f}"))
    return out


def price_bar_items(bars: pd.DataFrame, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """Daily OHLCV (stocks_*, market_*, index_hist_*, delisted_prices): a bar for date D is public at D 16:00 New York, not at
    D 00:00 (a naive join on the date would let the close be 'known' at the open). Bars with a non-positive close are skipped; a
    zero-volume bar gets unknown provenance because vendors fill missing prints with the last close. Strength is the bar's size in
    units of its trailing 20-day mean absolute return."""
    if bars is None or len(bars) == 0:
        return []
    need = {"close", "volume"}
    if not need <= set(bars.columns):
        raise KnowabilityError(f"bars lack {sorted(need - set(bars.columns))}")
    lo, hi = _win(window)
    b = bars.sort_index()
    ret = b["close"].pct_change()
    rng = ret.abs().rolling(20, min_periods=10).mean().shift(1)
    out = []
    for ts in b.index[(b.index >= lo) & (b.index <= hi)]:
        c = float(b["close"].loc[ts])
        if not c > 0:
            continue
        r, base = ret.loc[ts], rng.loc[ts]
        z = float(r / base) if pd.notna(r) and pd.notna(base) and base > 0 else 0.0
        stale = float(b["volume"].loc[ts]) <= 0
        out.append(InfoItem(f"PX-{ticker}-{ts:%Y%m%d}", InfoKind.PRICE, ticker, str(ts.date()), f"{ts:%Y-%m-%d} {CLOSE_TIME}", "prices",
                            PROV_UNKNOWN if stale else PROV_RECORDED, True, False, float(min(1.0, abs(z) / 8.0)), int(np.sign(z)),
                            f"bar z={z:+.2f}" + (" zero volume" if stale else "")))
    return out


def delisting_items(delist: pd.DataFrame, cik: int, ticker: str, window: tuple[str, str]) -> list[InfoItem]:
    """data/cache/delisted_events.parquet: cik, f25_date, f15_date, announced, delist_date, status, terminal. The earliest
    available announcement stamp (announced, Form 25, Form 15) is the publication time; `delist_date` is the last trading day and
    is EFFECTIVE, not a publication time. With no stamp at all the item has no publication time (UNCERTAIN). Terminal delistings
    (bankruptcy, liquidation) are strong and point down."""
    if delist is None or len(delist) == 0:
        return []
    lo, hi = _win(window)
    out = []
    for r in delist[delist["cik"] == cik].itertuples(index=False):
        rec = r._asdict()
        eff = rec.get("delist_date")
        stamps = [pd.Timestamp(rec[k]) for k in ("announced", "f25_date", "f15_date") if k in rec and pd.notna(rec[k])]
        pub = min(stamps) if stamps else None
        if eff is None or pd.isna(eff) or not lo <= pd.Timestamp(eff) <= hi:
            continue
        terminal = bool(rec.get("terminal", False))
        out.append(InfoItem(f"DELIST-{ticker}-{pd.Timestamp(eff):%Y%m%d}", InfoKind.CORPORATE_ACTION, ticker, str(pd.Timestamp(eff).date()),
                            None if pub is None else str(pub.date()), "delisted", PROV_RECORDED if pub is not None else PROV_UNKNOWN,
                            True, pub is not None, 0.95 if terminal else 0.6, -1 if terminal else 0,
                            f"delisting {rec.get('status', '')}".strip()))
    return out


ADAPTERS: dict[str, Callable[..., list[InfoItem]]] = {"edgar_events": edgar_events_items, "insider": insider_items, "macro": macro_items,
                                                     "prices": price_bar_items, "delisted": delisting_items}


def gather_items(ticker: str, window: tuple[str, str], *, events=None, insider=None, macro=None, bars=None, delisted=None, cik: int | None = None,
                 lag_days: Mapping[str, int] | None = None) -> list[InfoItem]:
    """All sources for one name and window, merged (the same fact from two sources collapses to the earliest publication) and sorted.
    Pass only the frames you have; a missing family contributes nothing."""
    items: list[InfoItem] = []
    items += edgar_events_items(events, ticker, window) if events is not None else []
    items += insider_items(insider, ticker, window) if insider is not None else []
    items += macro_items(macro, window, lag_days) if macro is not None else []
    items += price_bar_items(bars, ticker, window) if bars is not None else []
    if delisted is not None:
        if cik is None:
            raise KnowabilityError("delisting events are keyed by cik; pass cik=")
        items += delisting_items(delisted, cik, ticker, window)
    return merge_duplicate_items(items)


# ==================================================================================================================
# hindsight-side audits: what was UNCERTAIN and why; can any KNOWN_BEFORE tag be trusted; how risky is each source's revision
# ==================================================================================================================
FILING_DATE_SOURCES = {"insider", "delisted", "earnings_calendar", "splits"}          # sources whose date fields carry no time of day


@dataclasses.dataclass(frozen=True)
class UncertainEntry:
    item_id: str
    source: str
    kind: str
    reason: str
    minutes_from_cutoff: float | None


def uncertain_report(items: Sequence[InfoItem], b: DecisionBoundary, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> pd.DataFrame:
    """Every item whose availability was UNCERTAIN, with source and the judge's reason, sorted by source then reason. Summary rows
    at the end (item_id 'TOTAL:<source>') give each source's UNCERTAIN share of all its items."""
    cols = ["item_id", "source", "kind", "reason", "minutes_from_cutoff"]
    judg = judge_all(items, b, calendar, lag)
    rows = [dataclasses.asdict(UncertainEntry(i.item_id, i.source, i.kind.value, judg[i.item_id].reason, judg[i.item_id].margin_minutes))
            for i in items if judg[i.item_id].availability == Availability.UNCERTAIN]
    df = pd.DataFrame(rows, columns=cols).sort_values(["source", "reason", "item_id"]).reset_index(drop=True) if rows else pd.DataFrame(columns=cols)
    tot: dict[str, list[int]] = {}
    for i in items:
        t = tot.setdefault(i.source, [0, 0])
        t[1] += 1
        t[0] += int(judg[i.item_id].availability == Availability.UNCERTAIN)
    summ = pd.DataFrame([{"item_id": f"TOTAL:{s}", "source": s, "kind": "", "reason": f"{u}/{n} uncertain ({u / n:.0%})",
                          "minutes_from_cutoff": None} for s, (u, n) in sorted(tot.items())], columns=cols)
    return pd.concat([df, summ], ignore_index=True) if len(summ) else df


def uncertain_by_source(items: Sequence[InfoItem], b: DecisionBoundary, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> dict[str, dict[str, int]]:
    """source -> {reason: count} for UNCERTAIN items."""
    judg = judge_all(items, b, calendar, lag)
    out: dict[str, dict[str, int]] = {}
    for i in items:
        if judg[i.item_id].availability == Availability.UNCERTAIN:
            d = out.setdefault(i.source, {})
            d[judg[i.item_id].reason] = d.get(judg[i.item_id].reason, 0) + 1
    return out


@dataclasses.dataclass(frozen=True)
class GuaranteeFinding:
    item_id: str
    source: str
    problem: str


def audit_known_before(items: Sequence[InfoItem], b: DecisionBoundary, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy(),
                       guaranteed: Mapping[str, str] | None = None) -> list[GuaranteeFinding]:
    """No KNOWN_BEFORE tag may rest on a timestamp its source cannot guarantee. Findings:
    * KNOWN_BEFORE with unknown provenance;
    * KNOWN_BEFORE whose level is a current-vintage reconstruction (timing usable, value is hindsight);
    * KNOWN_BEFORE with an INFERRED publication (lag estimate, not a record) from a source that does not guarantee a stamp, or that
      lands within the lag margin of the cutoff;
    * KNOWN_BEFORE from a DATE-precision stamp on or after the decision date;
    * publication equal to the effective date for a date-only source without a guarantee (the two clocks were probably collapsed:
      the classic effective-date look-ahead).
    `guaranteed` maps source -> why its recorded stamp is trustworthy (default: edgar acceptance stamp, price session close)."""
    g = dict(guaranteed or {"edgar": "SEC acceptance timestamp", "prices": "session close"})
    judg = judge_all(items, b, calendar, lag)
    bad = []
    cut = pd.Timestamp(b.decision_date)
    for it in items:
        j = judg[it.item_id]
        if j.availability != Availability.KNOWN_BEFORE_EVENT:
            continue
        pub, prec = to_ny(it.published_at)
        eff, _ = to_ny(it.effective_at)
        if it.provenance == PROV_UNKNOWN:
            bad.append(GuaranteeFinding(it.item_id, it.source, "KNOWN_BEFORE with unknown provenance"))
        if it.provenance == PROV_RECONSTRUCTED:
            bad.append(GuaranteeFinding(it.item_id, it.source, "timing usable but value is a current-vintage reconstruction"))
        if j.inferred:
            if it.source not in g:
                bad.append(GuaranteeFinding(it.item_id, it.source, "publication inferred from an effective date; source does not guarantee a stamp"))
            else:
                near = eff is not None and abs(_sessions_between(cut, eff, calendar)) < lag.margin_sessions + lag.lag_for(it.source)
                if near:
                    bad.append(GuaranteeFinding(it.item_id, it.source, "inferred publication within the lag margin of the cutoff"))
        if prec == PREC_DATE and pub is not None and pub.normalize() >= cut:
            bad.append(GuaranteeFinding(it.item_id, it.source, "date-only stamp on or after the decision date tagged KNOWN_BEFORE"))
        if it.source in FILING_DATE_SOURCES and pub is not None and eff is not None and pub.normalize() == eff.normalize() and it.source not in g:
            bad.append(GuaranteeFinding(it.item_id, it.source, "publication equals effective date: the two clocks were probably collapsed"))
    return sorted(bad, key=lambda f: (f.source, f.item_id, f.problem))


def guarantee_summary(findings: Sequence[GuaranteeFinding]) -> dict[str, dict[str, int]]:
    """source -> {problem: count}."""
    out: dict[str, dict[str, int]] = {}
    for f in findings:
        d = out.setdefault(f.source, {})
        d[f.problem] = d.get(f.problem, 0) + 1
    return out


REVISION_RISK_DEFAULTS = {"edgar": (0.05, "filings are amended by later 8-K/A and 10-K/A; the original stamp stays"),
                          "insider": (0.15, "Form 4/A amendments restate size and date; our table keeps the latest"),
                          "prices": (0.10, "vendors restate bars for splits and corrections; adjusted history is rewritten"),
                          "delisted": (0.20, "delisting dates are reconstructed from Form 25 and successor records"),
                          "earnings_calendar": (0.30, "scheduled dates move; the announcement date is the only stable field"),
                          "splits": (0.05, "ex-dates are fixed once announced"),
                          "calendar": (0.25, "scheduled events are re-dated")}


def source_revision_risk(items: Sequence[InfoItem], macro_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Per source: revision-risk score in [0,1], reason, item count and how many items carry a reconstructed level. Macro series
    come from engine.leak_audit.macro_revision_risk (revised 0.8, market-priced 0.0, unknown 0.8). A source with no profile
    scores 0.5 and says so: unknown is not safe."""
    from engine.leak_audit import macro_revision_risk
    by_src: dict[str, list[InfoItem]] = {}
    for i in items:
        by_src.setdefault(i.source, []).append(i)
    cols = list(macro_columns)
    macro = macro_revision_risk(cols).set_index("series") if cols else None
    rows = []
    for s, its in sorted(by_src.items()):
        if macro is not None and s in macro.index:
            risk, why = (0.8 if bool(macro.loc[s, "revised"]) else 0.0), str(macro.loc[s, "reason"])
        elif s in REVISION_RISK_DEFAULTS:
            risk, why = REVISION_RISK_DEFAULTS[s]
        else:
            risk, why = 0.5, "no revision profile for this source; treated as risky"
        rows.append({"source": s, "n": len(its), "risk": float(risk), "reason": why,
                     "reconstructed": sum(1 for i in its if i.provenance == PROV_RECONSTRUCTED)})
    return pd.DataFrame(rows, columns=["source", "n", "risk", "reason", "reconstructed"])


def revision_adjusted_confidence(a: KnowabilityAssessment, risk: pd.DataFrame, items: Sequence[InfoItem]) -> float:
    """Confidence discounted by the revision risk of the sources behind the KNOWN_BEFORE information. Only ever lowers it."""
    if risk.empty or not a.information_that_would_have_been_available:
        return a.confidence_in_classification
    src = {i.item_id: i.source for i in items}
    r = risk.set_index("source")["risk"]
    used = [float(r.get(src[i], 0.5)) for i in a.information_that_would_have_been_available if i in src]
    if not used:
        return a.confidence_in_classification
    return float(a.confidence_in_classification * (1.0 - 0.5 * float(np.mean(used))))


# ==================================================================================================================
# calibration hooks: record stated confidence per class now, join it to later-confirmed outcomes, measure calibration in wave 3
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class CalibrationRecord:
    """One prediction the classifier made about itself. `confirmed_class` stays None until an independent later check (planted
    truth, replicated finding, manual review) confirms or corrects it. Records are never edited: a confirmation is a NEW record
    with the same move_id and a strictly later `confirmed_at`."""
    move_id: str
    predicted_class: str
    confidence: float
    config_hash: str
    matured_at: str
    regime: str = ""
    confirmed_class: str | None = None
    confirmed_at: str | None = None
    confirmed_by: str = ""

    def check(self) -> list[str]:
        errs = []
        classes = {k.value for k in Knowability}
        if not 0.0 <= self.confidence <= 1.0:
            errs.append("confidence outside [0,1]")
        if self.predicted_class not in classes:
            errs.append(f"unknown class {self.predicted_class!r}")
        if self.confirmed_class is not None:
            if self.confirmed_class not in classes:
                errs.append(f"unknown confirmed class {self.confirmed_class!r}")
            if not self.confirmed_at or as_date(self.confirmed_at) <= as_date(self.matured_at):
                errs.append("a confirmation must come strictly after the outcome matured")
            if not self.confirmed_by:
                errs.append("a confirmation needs a source")
        return errs


class CalibrationBook:
    """Append-only. `latest()` returns the newest record per move (a confirmation supersedes the unconfirmed prediction)."""

    def __init__(self):
        self._rows: list[CalibrationRecord] = []

    def __len__(self) -> int:
        return len(self._rows)

    def record(self, a: KnowabilityAssessment) -> CalibrationRecord:
        old = next((r for r in self._rows if r.move_id == a.move_id), None)
        if old is not None:
            return old
        rec = CalibrationRecord(a.move_id, a.classification.value, a.confidence_in_classification, a.config_hash, a.matured_at, a.regime)
        errs = rec.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        self._rows.append(rec)
        return rec

    def confirm(self, move_id: str, confirmed_class: str, confirmed_at, confirmed_by: str) -> CalibrationRecord:
        base = next((r for r in reversed(self._rows) if r.move_id == move_id), None)
        if base is None:
            raise KnowabilityError(f"{move_id} has no recorded prediction to confirm")
        new = dataclasses.replace(base, confirmed_class=confirmed_class, confirmed_at=str(as_date(confirmed_at)), confirmed_by=confirmed_by)
        errs = new.check()
        if errs:
            raise KnowabilityError("; ".join(errs))
        self._rows.append(new)
        return new

    def latest(self) -> list[CalibrationRecord]:
        cur: dict[str, CalibrationRecord] = {}
        for r in self._rows:
            cur[r.move_id] = r
        return list(cur.values())

    def confirmed(self) -> list[CalibrationRecord]:
        return [r for r in self.latest() if r.confirmed_class is not None]

    def rows(self) -> list[CalibrationRecord]:
        return list(self._rows)


def calibration_table(book: CalibrationBook, bins: int = 5, by_class: bool = True) -> pd.DataFrame:
    """Reliability table over CONFIRMED records: per (class, confidence bin) mean stated confidence, accuracy against the
    confirmed class, and n. Empty when nothing is confirmed (calibration is unmeasured, not perfect)."""
    cols = ["class", "lo", "hi", "n", "confidence", "accuracy", "gap"]
    rs = book.confirmed()
    if not rs:
        return pd.DataFrame(columns=cols)
    edges = np.linspace(0.0, 1.0, bins + 1)
    groups = sorted({r.predicted_class for r in rs}) if by_class else ["ALL"]
    rows = []
    for g in groups:
        sub = [r for r in rs if not by_class or r.predicted_class == g]
        conf = np.array([r.confidence for r in sub])
        ok = np.array([r.predicted_class == r.confirmed_class for r in sub], dtype=float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
            if m.any():
                rows.append({"class": g, "lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(conf[m].mean()),
                             "accuracy": float(ok[m].mean()), "gap": float(conf[m].mean() - ok[m].mean())})
    return pd.DataFrame(rows, columns=cols)


def calibration_summary(book: CalibrationBook, bins: int = 5) -> dict[str, Any]:
    """Expected calibration error, Brier score, overconfidence share and the predicted-vs-confirmed confusion."""
    rs = book.confirmed()
    if not rs:
        return {"n_confirmed": 0, "n_recorded": len(book.latest()), "ece": float("nan"), "brier": float("nan"), "overconfident": float("nan"), "confusion": {}}
    conf = np.array([r.confidence for r in rs])
    ok = np.array([r.predicted_class == r.confirmed_class for r in rs], dtype=float)
    t = calibration_table(book, bins, by_class=False)
    ece = float((t["n"] * t["gap"].abs()).sum() / t["n"].sum()) if len(t) else float("nan")
    conf_mat: dict[str, dict[str, int]] = {}
    for r in rs:
        d = conf_mat.setdefault(r.predicted_class, {})
        d[r.confirmed_class] = d.get(r.confirmed_class, 0) + 1
    return {"n_confirmed": len(rs), "n_recorded": len(book.latest()), "ece": ece, "brier": float(np.mean((conf - ok) ** 2)),
            "overconfident": float(np.mean(conf > ok)), "confusion": conf_mat}


def calibration_by_config(book: CalibrationBook) -> pd.DataFrame:
    """Accuracy of confirmed predictions per config hash, so a threshold change that hurts calibration shows up as a row."""
    rows: dict[str, list[float]] = {}
    for r in book.confirmed():
        rows.setdefault(r.config_hash, []).append(float(r.predicted_class == r.confirmed_class))
    return pd.DataFrame([{"config_hash": k, "n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(rows.items())],
                        columns=["config_hash", "n", "accuracy"])


def confirm_from_truth(book: CalibrationBook, truth: Mapping[str, Knowability], confirmed_at, source: str = "planted_truth") -> int:
    """Attach known truth (planted worlds, later replicated findings) to recorded, unconfirmed predictions; returns the count."""
    n = 0
    open_ids = {r.move_id for r in book.latest() if r.confirmed_class is None}
    for mid, cls_ in sorted(truth.items()):
        if mid in open_ids:
            book.confirm(mid, cls_.value, confirmed_at, source)
            n += 1
    return n


def record_ledger(book: CalibrationBook, ledger: KnowabilityLedger) -> int:
    """Register every assessment in a ledger; returns how many were new."""
    before = len(book)
    for a in ledger.rows():
        book.record(a)
    return len(book) - before


# ==================================================================================================================
# from cache-shaped frames to an audited move: coverage registry, precursor probes, classification, in one call
# ==================================================================================================================
def registry_from_frames(events=None, insider=None, macro=None, bars=None, delisted=None) -> SourceRegistry:
    """Coverage windows read off the frames themselves (first and last stamp), so a probe's silence outside that span is
    UNCERTAIN, not evidence. A family that was not supplied is not registered and therefore never 'covers' anything."""
    specs = []
    if events is not None and len(events):
        acc = pd.to_datetime(events["accepted"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        specs.append(SourceSpec("edgar", str(acc.min().date()), str(acc.max().date()), PREC_INTRADAY))
    if insider is not None and len(insider):
        specs.append(SourceSpec("insider", str(pd.Timestamp(insider["filed"].min()).date()), str(pd.Timestamp(insider["filed"].max()).date()), PREC_DATE))
    if macro is not None and len(macro):
        specs.append(SourceSpec("macro", str(pd.Timestamp(macro.index.min()).date()), str(pd.Timestamp(macro.index.max()).date()), PREC_DATE))
    if bars is not None and len(bars):
        specs.append(SourceSpec("prices", str(pd.Timestamp(bars.index.min()).date()), str(pd.Timestamp(bars.index.max()).date()), PREC_INTRADAY))
    if delisted is not None and len(delisted):
        specs.append(SourceSpec("delisted", str(pd.Timestamp(delisted["delist_date"].min()).date()),
                                str(pd.Timestamp(delisted["delist_date"].max()).date()), PREC_DATE))
    return SourceRegistry(specs)


def audit_move_from_frames(move: MoveEvent, bars: pd.DataFrame, *, events=None, insider=None, macro=None, delisted=None, cik: int | None = None,
                           market: pd.Series | None = None, sector: pd.Series | None = None, pattern_hits: Sequence[Mapping[str, Any]] = (),
                           memory_hits: Sequence[Mapping[str, Any]] = (), cross: Mapping[str, float] | None = None, lookback_days: int = 400,
                           cfg: KnowabilityConfig = KnowabilityConfig(), calendar: Calendar | None = None,
                           lag: LagPolicy = LagPolicy()) -> KnowabilityAssessment:
    """Assemble everything the auditor is allowed to know from cache-shaped frames and classify. Items are gathered over
    [decision - lookback, end + 10 days] (the tail past the move is hindsight and is tagged as such by the judge); precursor probes are
    attached from the coverage the frames themselves demonstrate. PRICE items are dropped from the cause candidates: the move's own
    bars are the outcome, not its explanation."""
    lo = str((pd.Timestamp(move.decision_date) - pd.Timedelta(days=lookback_days)).date())
    hi = str((pd.Timestamp(move.end_date) + pd.Timedelta(days=10)).date())
    items = [i for i in gather_items(move.ticker, (lo, hi), events=events, insider=insider, macro=macro, delisted=delisted, cik=cik)]
    inp = MoveInputs(move, bars, market, sector, items, {}, dict(cross or {}), list(pattern_hits), list(memory_hits))
    reg = registry_from_frames(events, insider, macro, bars, delisted)
    inp = attach_probes(inp, reg, calendar=calendar, lag=lag)
    return classify_move(inp, cfg, calendar, lag)


def frame_shape_errors(family: str, frame: pd.DataFrame) -> list[str]:
    """Does a frame have the columns and dtypes the adapter for `family` needs? Cheap guard for the wave-2 loader."""
    need = {"edgar_events": {"ticker": "any", "kind": "any", "form": "any", "accepted": "datetimetz"},
            "insider": {"symbol": "any", "filed": "datetime", "tdate": "datetime", "shares": "number", "price": "number"},
            "macro": {},
            "prices": {"close": "number", "volume": "number"},
            "delisted": {"cik": "number", "delist_date": "datetime"}}
    if family not in need:
        return [f"unknown family {family!r}; choose from {SOURCE_FAMILIES}"]
    errs = []
    for col, kind in need[family].items():
        if col not in frame.columns:
            errs.append(f"{family}: missing {col}")
        elif kind == "datetimetz" and getattr(frame[col].dtype, "tz", None) is None:
            errs.append(f"{family}.{col} must be tz-aware")
        elif kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(frame[col]):
            errs.append(f"{family}.{col} must be datetime")
        elif kind == "number" and not pd.api.types.is_numeric_dtype(frame[col]):
            errs.append(f"{family}.{col} must be numeric")
    if family == "macro" and not isinstance(frame.index, pd.DatetimeIndex):
        errs.append("macro must be indexed by observation date")
    if family == "prices" and not isinstance(frame.index, pd.DatetimeIndex):
        errs.append("prices must be indexed by session date")
    return errs


def source_staleness(items: Sequence[InfoItem], b: DecisionBoundary, calendar: Calendar | None = None, lag: LagPolicy = LagPolicy()) -> dict[str, int | None]:
    """Per source: sessions between its newest KNOWN_BEFORE item and the decision date (None = it had nothing before the cutoff).
    A source whose latest information was weeks old at the decision could not have anticipated a move, whatever it holds later."""
    judg = judge_all(items, b, calendar, lag)
    newest: dict[str, pd.Timestamp] = {}
    for it in items:
        pub, _ = to_ny(it.published_at)
        if pub is None:
            eff, _ = to_ny(it.effective_at)
            pub = None if eff is None else pd.Timestamp((calendar or Calendar()).shift([eff], lag.lag_for(it.source))[0])
        if pub is not None and judg[it.item_id].availability == Availability.KNOWN_BEFORE_EVENT:
            newest[it.source] = max(newest.get(it.source, pub), pub)
    out: dict[str, int | None] = {s: None for s in {i.source for i in items}}
    out.update({s: _sessions_between(t, pd.Timestamp(b.decision_date), calendar) for s, t in newest.items()})
    return dict(sorted(out.items()))


# ==================================================================================================================
# checklist K (C68): the five-way outcome classes. This module OWNS the mapping from its knowability classes (plus the few
# investigation facts that can decide without an assessment) to the five checklist-K classes; engine.research.what_changed
# calls `five_way` and re-exports FiveWay - there is no second five-way split anywhere (C69 duplication audit, P06).
# ==================================================================================================================
class FiveWay(_StrEnum):
    """Checklist K: exactly these five."""
    PREDICTABLE_BUT_MISSED = "PREDICTABLE_BUT_MISSED"
    PARTIALLY_PREDICTABLE = "PARTIALLY_PREDICTABLE"
    PREDICTABLE_UNDER_NEW_CONDITION = "PREDICTABLE_ONLY_UNDER_NEWLY_DISCOVERED_CONDITION"
    CURRENTLY_UNEXPLAINED = "CURRENTLY_UNEXPLAINED"
    GENUINELY_UNKNOWABLE = "GENUINELY_UNKNOWABLE_FROM_AVAILABLE_INFORMATION"


FIVE_WAY_OF_CLASS = {
    Knowability.PREDICTABLE: FiveWay.PREDICTABLE_BUT_MISSED,
    Knowability.POTENTIALLY_PREDICTABLE: FiveWay.PARTIALLY_PREDICTABLE,
    Knowability.WEAKLY_PREDICTABLE: FiveWay.PARTIALLY_PREDICTABLE,
    Knowability.UNKNOWN: FiveWay.CURRENTLY_UNEXPLAINED,
    Knowability.EXTERNALLY_CAUSED: FiveWay.GENUINELY_UNKNOWABLE,
    Knowability.INFORMATIONALLY_UNAVAILABLE: FiveWay.GENUINELY_UNKNOWABLE,
    Knowability.DATA_FAILURE: FiveWay.CURRENTLY_UNEXPLAINED,
}


@dataclasses.dataclass(frozen=True)
class FiveWayVerdict:
    klass: FiveWay
    reasons: tuple
    excluded_data_failure: bool        # a data defect is not a market event; it is excluded from market conclusions
    basis: str                         # which evidence decided: assessment / findings / condition


def five_way(classification: Knowability | None, *, condition_key: str = "", condition_replicated: bool = False,
             pattern_failing_lead: int | None = None, pre_outcome_cause: bool = False) -> FiveWayVerdict:
    """THE checklist-K mapping. Order is the contract: a data defect is excluded; unknowable evidence (externally caused,
    informationally unavailable) is never overridden by a story; PREDICTABLE is 'predictable but missed'; a potentially or weakly
    predictable move is 'partially predictable' unless a condition that has ALREADY been replicated makes it predictable (the
    new-condition class is never granted to a fresh claim). Without an assessment (`classification` None) only investigation facts
    decide: a pattern that was already failing `pattern_failing_lead` rows before the decision -> predictable but missed; a fired cause
    with pre-outcome evidence -> partially predictable; otherwise currently unexplained - nothing is forced predictable."""
    if classification is not None:
        c = Knowability(classification)
        if c == Knowability.DATA_FAILURE:
            return FiveWayVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("the move traces to a data defect, not market behaviour",), True, "assessment")
        if FIVE_WAY_OF_CLASS[c] == FiveWay.GENUINELY_UNKNOWABLE:
            return FiveWayVerdict(FiveWay.GENUINELY_UNKNOWABLE, (f"assessment {c.value}: nothing available beforehand carried it",), False, "assessment")
        if c == Knowability.PREDICTABLE:
            return FiveWayVerdict(FiveWay.PREDICTABLE_BUT_MISSED, ("the information was available and sufficient before the decision",), False, "assessment")
        if condition_replicated:
            why = "makes it predictable where it was only weakly so" if FIVE_WAY_OF_CLASS[c] == FiveWay.PARTIALLY_PREDICTABLE \
                else "explains what the assessment could not"
            return FiveWayVerdict(FiveWay.PREDICTABLE_UNDER_NEW_CONDITION, (f"a replicated condition ({condition_key}) {why}",), False, "condition")
        if FIVE_WAY_OF_CLASS[c] == FiveWay.PARTIALLY_PREDICTABLE:
            return FiveWayVerdict(FiveWay.PARTIALLY_PREDICTABLE, (f"assessment {c.value}: part of the move was foreseeable",), False, "assessment")
        return FiveWayVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("the assessment found no explanation; unknown stays unknown",), False, "assessment")
    if pattern_failing_lead is not None:
        return FiveWayVerdict(FiveWay.PREDICTABLE_BUT_MISSED, (f"the pattern was already failing {pattern_failing_lead} rows before the decision",),
                              False, "findings")
    if condition_replicated:
        return FiveWayVerdict(FiveWay.PREDICTABLE_UNDER_NEW_CONDITION, (f"replicated condition {condition_key}",), False, "condition")
    if pre_outcome_cause:
        return FiveWayVerdict(FiveWay.PARTIALLY_PREDICTABLE, ("a cause with pre-outcome evidence fired, but no knowability assessment confirms it",),
                              False, "findings")
    return FiveWayVerdict(FiveWay.CURRENTLY_UNEXPLAINED, ("no level explained the error with pre-outcome evidence; no forced explanation",), False, "findings")
