"""Complete outcome reconstruction (C68 checklist B): the whole realised path, never just entry and exit.

After a position has exited (and only then: `exit_at` must be strictly before `now`, every bar handed in must be strictly before
`now`) `reconstruct` rebuilds what happened against a frozen expectation (engine.research.expectations): the return at exit,
best/worst excursions (intraday) and best/worst close-to-close return before exit with WHEN they occurred, realised volatility
and direction, the holding period, every meaningful intraperiod movement (swing legs and outlier days), the market, sector and
peer moves over the same window, how each expected pattern and interaction actually behaved and where a pattern's strength
changed, external-event indicators (gaps, volume spikes, caller-supplied events) and the best exit that existed in hindsight
(inside the hold and, if the caller supplies post-exit bars that are still before `now`, after it).

Conventions: sessions are counted from the entry open, the entry session being 1 (so a position exited at the close of the entry
session held 1 session); returns are POSITION returns (a short gains when the price falls); market/sector/peer moves are raw
price moves. Missing data is reported (`data_gaps`) or raises `OutcomeUnavailable` - a NaN is never turned into a zero.

`OutcomeLedger` stores reconstructions on the same kind of hash-chained lane as the expectations and refuses an outcome whose
expectation was not frozen before the entry session, whose expectation hash differs, or which contradicts a stored outcome.
LABEL = IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.core import FirewallBreach, as_date, canonical_json, require_past, stable_hash
from engine.research.expectations import Expectation, ExpectationLedger, LedgerTampered, SealedLane

LABEL = "IMPLEMENTED - NOT VALIDATED"
LANE_OUTCOME = "out68"
DIRECTION_DEADZONE = 1e-4                 # |return| below this is "flat", neither right nor wrong about direction


class OutcomeUnavailable(ValueError):
    """The path cannot be reconstructed honestly (missing entry/exit, mismatched arrays, no data). Unknown stays unknown."""


class OutcomeRewrite(FirewallBreach):
    """A different outcome was offered for a prediction that already has one, or the outcome does not match its expectation."""


@dataclasses.dataclass(frozen=True)
class PathData:
    """Raw material for one position, all aligned on `dates`. `exit_at` is the session the position was actually closed;
    `exit_price` its actual fill (default: that session's close). Bars after exit_at may be included for the post-exit check."""
    dates: tuple
    open: tuple
    high: tuple
    low: tuple
    close: tuple
    exit_at: str
    exit_price: float | None = None
    volume: tuple | None = None
    market_close: tuple | None = None
    market_base: float | None = None            # market level just before the entry open; default = first close
    sector_close: tuple | None = None
    sector_base: float | None = None
    peers: Mapping = dataclasses.field(default_factory=dict)          # name -> close tuple (aligned)
    peer_base: Mapping = dataclasses.field(default_factory=dict)
    pattern_series: Mapping = dataclasses.field(default_factory=dict)   # pattern -> live strength per session
    pattern_realized: Mapping = dataclasses.field(default_factory=dict)      # pattern -> realised strength over the hold
    interaction_realized: Mapping = dataclasses.field(default_factory=dict)  # interaction -> realised strength
    events: tuple = ()                          # ({"date":.., "kind":.., "size":..}, ...) supplied indicators

    def n(self) -> int:
        return len(self.dates)

    def aligned_problems(self) -> list[str]:
        n, errs = self.n(), []
        for name in ("open", "high", "low", "close", "volume", "market_close", "sector_close"):
            v = getattr(self, name)
            if v is not None and len(v) != n:
                errs.append(f"{name} has {len(v)} values for {n} dates")
        for grp in (self.peers, self.pattern_series):
            for k, v in grp.items():
                if len(v) != n:
                    errs.append(f"series {k!r} has {len(v)} values for {n} dates")
        ds = [as_date(d) for d in self.dates]
        if any(b <= a for a, b in zip(ds, ds[1:])):
            errs.append("dates are not strictly increasing")
        return errs


@dataclasses.dataclass(frozen=True, kw_only=True)
class OutcomeReconstruction:
    prediction_id: str
    expectation_hash: str
    entry_at: str
    exit_at: str
    matured_at: str                             # first session on which this outcome is knowable (= exit_at)
    side: int
    entry_price: float
    exit_price: float
    exit_return: float                          # actual return at exit
    mfe: float                                  # best intraday excursion (>= 0)
    mae: float                                  # worst intraday excursion (<= 0)
    max_return: float                           # best close-to-close return before exit (exit fill included)
    min_return: float
    t_max: int                                  # sessions held at that best point
    t_min: int
    realized_vol: float                         # daily std of position returns (Parkinson range estimate if < 3 returns)
    vol_method: str
    actual_direction: int                       # +1 / -1 / 0 (flat)
    holding_days: int
    trajectory: tuple                           # close-based position return after each session held, exit fill last
    movements: tuple                            # swing legs and outlier days
    market_move: float | None
    market_vol: float | None
    sector_move: float | None
    sector_vol: float | None
    peer_moves: Mapping
    peer_mean_move: float | None
    excess_over_market: float | None            # raw price move of the stock minus the market move
    pattern_behavior: Mapping                   # per expected pattern: start/end/mean/min/max/change/flipped
    pattern_changes: tuple                      # patterns whose strength moved materially inside the hold
    pattern_realized: Mapping
    interaction_realized: Mapping
    external_events: tuple
    best_exit: Mapping                          # hindsight best exit inside the hold
    post_exit_best: Mapping | None              # best after the actual exit (only bars < now), else None
    regret: float                               # best close return in the hold minus the return actually taken
    data_gaps: tuple

    def __post_init__(self):
        for f in ("peer_moves", "pattern_behavior", "pattern_realized", "interaction_realized", "best_exit"):
            object.__setattr__(self, f, MappingProxyType(dict(sorted(getattr(self, f).items()))))
        if self.post_exit_best is not None:
            object.__setattr__(self, "post_exit_best", MappingProxyType(dict(self.post_exit_best)))
        for f in ("trajectory", "movements", "pattern_changes", "external_events", "data_gaps"):
            object.__setattr__(self, f, tuple(getattr(self, f)))

    @property
    def outcome_hash(self) -> str:
        return stable_hash(json.loads(canonical_json(self)), 32)

    def content(self) -> dict:
        return json.loads(canonical_json(self))

    @classmethod
    def from_record(cls, d: Mapping) -> "OutcomeReconstruction":
        return cls(**dict(d))


# ------------------------------------------------------------------------------------------------ path mathematics
def position_returns(price: Sequence[float], entry: float, side: int) -> np.ndarray:
    """Position return of holding the instrument from `entry` at each price: side * (p/entry - 1)."""
    return side * (np.asarray(price, float) / entry - 1.0)


def excursions(high: Sequence[float], low: Sequence[float], entry: float, side: int) -> tuple[float, float]:
    """(mfe, mae) of the position over the bars, from intraday extremes; mfe >= 0 >= mae because the entry itself is a point."""
    hi, lo = np.asarray(high, float), np.asarray(low, float)
    if side > 0:
        fav, adv = np.nanmax(hi) / entry - 1.0, np.nanmin(lo) / entry - 1.0
    else:
        fav, adv = -(np.nanmin(lo) / entry - 1.0), -(np.nanmax(hi) / entry - 1.0)
    return float(max(fav, 0.0)), float(min(adv, 0.0))


def parkinson_vol(high: Sequence[float], low: Sequence[float]) -> float:
    """Daily range-based volatility (Parkinson): sqrt(mean(ln(H/L)^2) / (4 ln 2))."""
    hl = np.log(np.asarray(high, float) / np.asarray(low, float))
    hl = hl[np.isfinite(hl)]
    return float(math.sqrt(np.mean(hl ** 2) / (4.0 * math.log(2.0)))) if len(hl) else float("nan")


def zigzag(values: Sequence[float], threshold: float) -> list[dict]:
    """Swing legs of a return path: a leg ends when the path reverses by more than `threshold` from its running extreme.
    Each leg is {"from_t","to_t","change"}; t is the index into `values` (0 = the entry). The last unfinished leg is included."""
    v = np.asarray(values, float)
    if len(v) < 2 or threshold <= 0:
        return []
    legs, pivot, ext, ext_i, direction = [], 0, v[0], 0, 0
    for i in range(1, len(v)):
        if not math.isfinite(v[i]):
            continue
        if direction == 0:                           # no leg established yet: wait for a move of `threshold` from the start
            if abs(v[i] - v[pivot]) >= threshold:
                direction, ext, ext_i = (1 if v[i] > v[pivot] else -1), v[i], i
            continue
        if direction * (v[i] - ext) > 0:             # the leg extends
            ext, ext_i = v[i], i
        elif direction * (ext - v[i]) >= threshold:  # reversed by threshold: the leg ends at its extreme
            legs.append({"from_t": pivot, "to_t": ext_i, "change": float(ext - v[pivot])})
            pivot, ext, ext_i, direction = ext_i, v[i], i, -direction
    if direction != 0 and ext_i > pivot:
        legs.append({"from_t": pivot, "to_t": ext_i, "change": float(ext - v[pivot])})
    return legs


def outlier_days(daily: Sequence[float], vol: float, k: float = 2.0) -> list[dict]:
    """Days whose position return exceeds k daily volatilities (a floor of 1% stops calm paths from flagging noise)."""
    cut = max(k * vol if math.isfinite(vol) else 0.0, 0.01)
    return [{"t": i + 1, "change": float(r)} for i, r in enumerate(daily) if math.isfinite(r) and abs(r) >= cut]


def _std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan")


def pattern_behaviour(series: Mapping, i_exit: int, change_frac: float = 0.5) -> tuple[dict, list[dict]]:
    """Per pattern: start, end, mean, min, max, change and whether the sign flipped; a pattern 'changed' when its strength
    moved by at least change_frac of max(|start|, 0.1) or flipped sign. Patterns without observations are absent, not zero."""
    beh, changes = {}, []
    for name, s in series.items():
        a = np.asarray(s, float)[: i_exit + 1]
        a = a[np.isfinite(a)]
        if not len(a):
            continue
        ch = float(a[-1] - a[0])
        flipped = bool(a[0] * a[-1] < 0)
        beh[name] = {"start": float(a[0]), "end": float(a[-1]), "mean": float(a.mean()), "min": float(a.min()),
                     "max": float(a.max()), "change": ch, "flipped": flipped}
        if flipped or abs(ch) >= change_frac * max(abs(a[0]), 0.1):
            changes.append({"pattern": name, "change": ch, "flipped": flipped})
    return beh, changes


def event_indicators(open_: np.ndarray, close: np.ndarray, volume: np.ndarray | None, dates: Sequence[str], vol: float,
                     supplied: Iterable[Mapping], gap_k: float = 3.0, vol_mult: float = 3.0) -> list[dict]:
    """External-event indicators inside the hold: overnight gaps beyond gap_k daily vols (>= 3%), volume spikes beyond
    vol_mult times the path median, and events the caller supplied (news/earnings flags), each with its session index."""
    out = []
    cut = max(gap_k * vol if math.isfinite(vol) else 0.0, 0.03)
    for i in range(1, len(open_)):
        g = open_[i] / close[i - 1] - 1.0
        if math.isfinite(g) and abs(g) >= cut:
            out.append({"t": i + 1, "date": dates[i], "kind": "gap", "size": float(g)})
    if volume is not None and len(volume) >= 5:
        med = np.nanmedian(volume)
        for i, v in enumerate(volume):
            if math.isfinite(v) and med > 0 and v >= vol_mult * med:
                out.append({"t": i + 1, "date": dates[i], "kind": "volume_spike", "size": float(v / med)})
    win = set(dates)
    for ev in supplied:
        if as_date(ev["date"]).isoformat() in win:
            out.append({"t": dates.index(as_date(ev["date"]).isoformat()) + 1, "date": as_date(ev["date"]).isoformat(),
                        "kind": str(ev["kind"]), "size": float(ev.get("size", 1.0))})
    return sorted(out, key=lambda e: (e["t"], e["kind"]))


# ------------------------------------------------------------------------------------------------ reconstruction
def reconstruct(exp: Expectation, exp_hash: str, path: PathData, now, *, swing: float | None = None,
                post_exit_sessions: int | None = None) -> OutcomeReconstruction:
    """Rebuild the realised path of `exp`. Fails closed: the outcome must have matured strictly before `now`, and no bar
    (in or after the hold) may be dated on or after `now`."""
    require_past(path.exit_at, now, "position exit (outcome maturity)")
    probs = path.aligned_problems()
    if probs:
        raise OutcomeUnavailable("; ".join(probs))
    if path.n() == 0:
        raise OutcomeUnavailable("no bars")
    ds = [as_date(d).isoformat() for d in path.dates]
    for d in ds:
        require_past(d, now, "price bar")
    try:
        i0, i1 = ds.index(exp.entry_at), ds.index(as_date(path.exit_at).isoformat())
    except ValueError as ex:
        raise OutcomeUnavailable(f"entry {exp.entry_at} or exit {path.exit_at} not among the bars") from ex
    if i1 < i0:
        raise OutcomeUnavailable("exit precedes entry")
    sl = slice(i0, i1 + 1)
    O, H, L, C = (np.asarray(getattr(path, k), float)[sl] for k in ("open", "high", "low", "close"))
    entry = float(O[0])
    if not (math.isfinite(entry) and entry > 0):
        raise OutcomeUnavailable("entry open missing or non-positive")
    exit_px = float(path.exit_price if path.exit_price is not None else C[-1])
    if not (math.isfinite(exit_px) and exit_px > 0):
        raise OutcomeUnavailable("exit price missing or non-positive")
    side = int(exp.direction)
    gaps = [f"{n} bar(s) missing in {name}" for name, a in (("open", O), ("high", H), ("low", L), ("close", C))
            if (n := int((~np.isfinite(a)).sum()))]
    if np.isfinite(H).sum() == 0 or np.isfinite(L).sum() == 0:
        raise OutcomeUnavailable("no usable highs/lows")
    exit_ret = float(side * (exit_px / entry - 1.0))
    closes_ret = position_returns(C[:-1], entry, side) if len(C) > 1 else np.array([])
    traj = np.concatenate([closes_ret, [exit_ret]])
    mfe, mae = excursions(H, L, entry, side)
    mfe, mae = max(mfe, exit_ret, 0.0), min(mae, exit_ret, 0.0)        # the fill itself is part of the path
    i_max, i_min = int(np.nanargmax(traj)), int(np.nanargmin(traj))
    lv = np.concatenate([[entry], [c for c in C[:-1]], [exit_px]])
    daily = side * (np.diff(lv) / lv[:-1])
    vol = _std(daily)
    method = "close_to_close"
    if not math.isfinite(vol) or len(daily) < 3:
        vol, method = parkinson_vol(H, L), "parkinson"
    hold = len(traj)
    a_dir = 0 if abs(exit_ret) < DIRECTION_DEADZONE else int(np.sign(exit_ret))
    thr = swing if swing is not None else max(0.02, 1.5 * (vol if math.isfinite(vol) else 0.0))
    moves = [{"kind": "swing", **s} for s in zigzag(np.concatenate([[0.0], traj]), thr)] + \
            [{"kind": "outlier_day", **d} for d in outlier_days(daily, vol)]
    mkt, mvol = _rebase(path.market_close, path.market_base, i0, i1)
    sec, svol = _rebase(path.sector_close, path.sector_base, i0, i1)
    peers = {k: _rebase(v, path.peer_base.get(k), i0, i1)[0] for k, v in path.peers.items()}
    peers = {k: v for k, v in peers.items() if v is not None}
    stock_raw = exit_px / entry - 1.0
    beh, changes = pattern_behaviour({k: np.asarray(v, float)[i0:] for k, v in path.pattern_series.items()}, i1 - i0)
    events = event_indicators(O, np.where(np.isfinite(C), C, O), None if path.volume is None else np.asarray(path.volume, float)[sl],
                              ds[sl], vol, path.events)
    best_i = int(np.nanargmax(traj))
    best = {"t": best_i + 1, "return": float(traj[best_i]), "intraday_upper_bound": float(mfe)}
    post = _post_exit(exp, path, i0, i1, entry, side, post_exit_sessions)
    return OutcomeReconstruction(
        prediction_id=exp.prediction_id, expectation_hash=exp_hash, entry_at=exp.entry_at, exit_at=ds[i1], matured_at=ds[i1],
        side=side, entry_price=entry, exit_price=exit_px, exit_return=exit_ret, mfe=mfe, mae=mae,
        max_return=float(traj[i_max]), min_return=float(traj[i_min]), t_max=i_max + 1, t_min=i_min + 1, realized_vol=float(vol),
        vol_method=method, actual_direction=a_dir, holding_days=hold, trajectory=tuple(float(x) for x in traj),
        movements=tuple(moves), market_move=mkt, market_vol=mvol, sector_move=sec, sector_vol=svol, peer_moves=peers,
        peer_mean_move=float(np.mean(list(peers.values()))) if peers else None,
        excess_over_market=None if mkt is None else float(stock_raw - mkt), pattern_behavior=beh,
        pattern_changes=tuple(changes), pattern_realized={k: float(v) for k, v in path.pattern_realized.items()},
        interaction_realized={k: float(v) for k, v in path.interaction_realized.items()}, external_events=tuple(events),
        best_exit=best, post_exit_best=post, regret=float(traj[best_i] - exit_ret), data_gaps=tuple(gaps))


def _rebase(series, base, i0: int, i1: int) -> tuple[float | None, float | None]:
    """Move and daily volatility of a context series over the SAME window as the position (entry session .. exit session)."""
    if series is None:
        return None, None
    s = np.asarray(series, float)
    seg = s[i0: i1 + 1]
    b = base if base is not None else (s[i0 - 1] if i0 > 0 else seg[0])
    if not (b and math.isfinite(b) and b > 0 and math.isfinite(seg[-1])):
        return None, None
    lv = np.concatenate([[b], seg])
    return float(seg[-1] / b - 1.0), _std(np.diff(lv) / lv[:-1])


def _post_exit(exp: Expectation, path: PathData, i0: int, i1: int, entry: float, side: int, sessions: int | None) -> dict | None:
    """Best close-to-close return in the sessions after the actual exit, up to the expectation's latest exit time (or `sessions`
    sessions after entry). Every bar handed in is already known to be before `now`. None when no such bars were supplied."""
    latest = int(math.ceil(exp.exit_window[1])) if sessions is None else sessions
    hi = min(path.n(), i0 + latest)
    if hi <= i1 + 1:
        return None
    seg = position_returns(np.asarray(path.close, float)[i1 + 1: hi], entry, side)
    if not np.isfinite(seg).any():
        return None
    k = int(np.nanargmax(seg))
    return {"t": i1 + 1 + k - i0 + 1, "return": float(seg[k]), "sessions_after_exit": len(seg)}


def reconstruct_many(pairs: Sequence[tuple[Expectation, str, PathData]], now, **kw) -> tuple[list[OutcomeReconstruction], dict]:
    """Reconstruct many; positions that are not yet matured or lack data come back as reasons, never as guesses."""
    done, skipped = [], {}
    for exp, h, path in pairs:
        try:
            done.append(reconstruct(exp, h, path, now, **kw))
        except (OutcomeUnavailable, FirewallBreach) as ex:
            skipped[exp.prediction_id] = f"{type(ex).__name__}: {ex}"
    return done, skipped


# ------------------------------------------------------------------------------------------------ ledger
class OutcomeLedger:
    """Immutable store of reconstructed outcomes, tied to an ExpectationLedger. Same guarantees as the expectations."""

    def __init__(self, expectations: ExpectationLedger, root=None):
        self.expectations = expectations
        self.lane = SealedLane(root, LANE_OUTCOME)
        self._by_id: dict[str, OutcomeReconstruction] = {}
        for ln in self.lane.lines():
            r = OutcomeReconstruction.from_record(ln["body"]["outcome"])
            self._by_id[r.prediction_id] = r

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, pid: str) -> bool:
        return pid in self._by_id

    @property
    def head(self) -> str:
        return self.lane.head

    def add(self, out: OutcomeReconstruction, now) -> str:
        """Store `out`. Requires: the expectation exists and hashes to `out.expectation_hash`; it was frozen no later than the
        entry session; the outcome matured strictly before `now`. A different outcome for the same prediction is a rewrite."""
        pid = out.prediction_id
        if pid not in self.expectations:
            raise OutcomeRewrite(f"outcome for {pid} has no recorded expectation")
        meta = self.expectations.meta(pid)
        if meta["content_hash"] != out.expectation_hash:
            raise OutcomeRewrite(f"outcome {pid} was reconstructed against different expectation content")
        if not self.expectations.frozen_before(pid, out.entry_at):
            raise OutcomeRewrite(f"expectation {pid} was not frozen before the position was entered")
        require_past(out.matured_at, now, f"outcome {pid} maturity")
        if pid in self._by_id:
            if self._by_id[pid].outcome_hash != out.outcome_hash:
                raise OutcomeRewrite(f"prediction {pid} already has a stored outcome with different content")
            return pid
        r = self.lane.append({"prediction_id": pid, "outcome_hash": out.outcome_hash, "recorded_at": as_date(now).isoformat(),
                              "outcome": out.content()})
        self._by_id[pid] = OutcomeReconstruction.from_record(r["body"]["outcome"])
        return pid

    def update(self, *_a, **_k):
        raise OutcomeRewrite("outcomes are immutable: no update")

    delete = update

    def get(self, pid: str, now=None) -> OutcomeReconstruction:
        out = self._by_id[pid]
        if now is not None:
            require_past(out.matured_at, now, f"outcome {pid} maturity")
        return out

    def matured(self, now) -> list[tuple[Expectation, OutcomeReconstruction]]:
        """(expectation, outcome) pairs whose outcome matured strictly before `now` - the only feed the error engine reads."""
        return [(self.expectations.get(p), o) for p, o in sorted(self._by_id.items())
                if as_date(o.matured_at) < as_date(now)]

    def pending(self, now) -> list[str]:
        """Frozen expectations with no stored outcome yet."""
        return [p for p in self.expectations.ids() if p not in self._by_id]

    def verify(self, anchors: Iterable[str] = ()) -> dict:
        def check(body: dict) -> str | None:
            try:
                out = OutcomeReconstruction.from_record(body["outcome"])
            except (KeyError, TypeError, ValueError) as ex:
                return f"unreadable outcome body: {ex}"
            return None if out.outcome_hash == body.get("outcome_hash") else f"outcome {body.get('prediction_id')} content does not match its hash"
        rep = self.lane.verify(anchors, check)
        exp_rep = self.expectations.verify()
        return {**rep, "ok": rep["ok"] and exp_rep["ok"], "problems": rep["problems"] + exp_rep["problems"]}

    def assert_intact(self, anchors: Iterable[str] = ()) -> dict:
        rep = self.verify(anchors)
        if not rep["ok"]:
            raise LedgerTampered("; ".join(rep["problems"]))
        return rep
