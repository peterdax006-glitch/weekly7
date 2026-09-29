"""Market-wide daily research observer (RESEARCH_BRAIN_CONTRACT C66 section 4; serves canon C66 + C62 + C63, Bible research loop).

Every simulated day the observer looks at the WHOLE eligible universe - not only the stocks the model picked - and keeps a
complete research record with every section-4 category (A-I): considered high / medium, rejected, abstentions, low confidence,
winners, losers, extreme up / down, predictable and apparently unpredictable movers, near misses, false positives and false
negatives. Classification is vectorised over the full cross-section (about 3,000 names cost a few milliseconds; measure_cost()
proves it), and only the EXCEPTION rows are persisted (tails, near misses, false positives / negatives, abstentions, the
extremes of each market metric), with exact counts kept for the rest and a seeded sample where a category overflows its cap.

Two worlds (section 29). The DecisionSnapshot is point-in-time state at the decision close; the DayOutcome is the matured
outcome. observe_day() refuses to run unless the outcome matured strictly before `now` and the snapshot was decided before the
outcome resolved. The resulting DayRecord lives in MATURED_RESEARCH_STATE: it reaches anything live only through
DayRecord.as_matured(...).gate(now), and a ledger filed under a real year refuses release while that year is replayed in disguise
(the same-year rerun leak, rule 27). "Predictable / unpredictable" here is a cheap precursor PROXY measured against its own lift;
the real could-I-have-known reconstruction lives in the knowability engine, and a mover the proxy cannot place is left unplaced,
never forced.

Built ON: engine.missed_winners (winner_types for the mover type, why_missed for the per-winner reason, WINNER) and
engine.learning.missed_winners (Candidate / Week / week_from_base vocabulary so the rejection analyser can read a day).
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from engine import missed_winners as base
from engine.learning import missed_winners as lmw
from engine.research.core import (FirewallBreach, MaturedRecord, MoveCategory, Provenance, as_date, require_past,
                                  stable_hash)

MC = MoveCategory
CATEGORIES: tuple[MoveCategory, ...] = tuple(MoveCategory)                 # bit i of a row's `flags` = CATEGORIES[i]
BIT: dict[MoveCategory, int] = {c: 1 << i for i, c in enumerate(CATEGORIES)}
MODEL_CATS = (MC.CONSIDERED_HIGH, MC.CONSIDERED_MEDIUM, MC.REJECTED, MC.ABSTAINED, MC.LOW_CONFIDENCE, MC.NEAR_MISS)
OUTCOME_CATS = (MC.WINNER, MC.LOSER, MC.EXTREME_UP, MC.EXTREME_DOWN, MC.PREDICTABLE_MOVER, MC.UNPREDICTABLE_MOVER,
                MC.FALSE_POSITIVE, MC.FALSE_NEGATIVE)
TOP_METRICS = ("gainers", "losers", "ranges", "volume", "gaps", "vol_expansion", "vol_contraction")
DEFAULT_PRECURSORS = (("vol20", 1.0), ("atr", 1.0), ("vol_surge", 0.75), ("range20", 0.75), ("r5", -0.25), ("log_dv", -0.25))


class ObserverError(ValueError):
    """Malformed observer input (misaligned, non-finite where it must be finite, duplicated names, wrong order in time)."""


# ==================================================================================================================
# parameters
# ==================================================================================================================
@dataclass(frozen=True)
class ObserverParams:
    horizon: int = 5                      # sessions the outcome covers (the fill is the next open)
    win_thr: float = base.WINNER          # a winner / loser is a total move of at least this from the decision close
    mover_thr: float = 0.07               # "moved substantially": best excursion from the fill reached this either way
    tail_q: float = 0.01                  # extreme movers: the top / bottom fraction of the cross-section ...
    extreme_floor: float = 0.05           # ... and at least this large (a quiet day has no extreme movers)
    k_pick: int = 10                      # positions the model can hold
    k_high: int = 20                      # ranks <= this were considered at high priority
    k_med: int = 60                       # ranks <= this were considered at medium priority
    near_frac: float = 0.5                # near miss: rank within k_pick * (1 + near_frac) ...
    near_score_gap: float = 0.10          # ... or a score within this fraction of the last pick's score
    conf_gate: float = 0.5
    pred_q: float = 0.80                  # predictable mover: precursor rank at/above this ...
    score_top_frac: float = 0.02          # ... or a model score in this top fraction ...
    unpred_q: float = 0.50                # unpredictable mover: precursor rank BELOW this and ...
    gap_share_thr: float = 0.60           # ... at least this much of the move arrived as an overnight gap, with no known event
    explain_limit: int = 10               # winners a day explained through engine.missed_winners.why_missed (0 = off)
    max_rows_per_cat: int = 60            # exception cap per category per day; the overflow is a seeded sample
    keep_top_share: float = 0.75          # of that cap, the largest |move| / highest score are kept outright
    n_top: int = 10                       # length of each market metric table
    min_universe: int = 1
    sample_seed: int = 0
    rank_features: tuple[str, ...] = ("vol20", "atr", "r5", "r20", "dist_hi", "log_dv", "vol_surge", "range20")
    precursors: tuple[tuple[str, float], ...] = DEFAULT_PRECURSORS
    salt: str = "obs"
    band_lo: float = 0.05                 # C67: the 5-10% band is [band_lo, band_hi), the >10% band is [band_hi, inf)
    band_hi: float = 0.10

    def validate(self) -> list[str]:
        e = []
        if self.horizon < 1:
            e.append("horizon < 1")
        for name in ("win_thr", "mover_thr", "extreme_floor"):
            if not getattr(self, name) > 0:
                e.append(f"{name} must be positive")
        if not 0 < self.tail_q < 0.5:
            e.append("tail_q outside (0, 0.5)")
        if not 1 <= self.k_pick <= self.k_high <= self.k_med:
            e.append("need 1 <= k_pick <= k_high <= k_med")
        if not 0 <= self.pred_q <= 1 or not 0 <= self.unpred_q <= 1 or self.unpred_q > self.pred_q:
            e.append("need 0 <= unpred_q <= pred_q <= 1")
        if not 0 < self.gap_share_thr <= 1:
            e.append("gap_share_thr outside (0, 1]")
        if self.max_rows_per_cat < 1 or not 0 < self.keep_top_share <= 1:
            e.append("row cap must be >= 1 and keep_top_share in (0, 1]")
        if self.n_top < 1:
            e.append("n_top < 1")
        if not 0 < self.band_lo < self.band_hi:
            e.append("need 0 < band_lo < band_hi")
        if not self.precursors:
            e.append("no precursor features")
        return e

    def require_valid(self) -> "ObserverParams":
        errs = self.validate()
        if errs:
            raise ValueError("invalid ObserverParams: " + "; ".join(errs))
        return self

    def hash(self) -> str:
        return stable_hash(self)


# ==================================================================================================================
# inputs: point-in-time snapshot at the decision close, and the matured outcome
# ==================================================================================================================
def _arr(x, n: int, dtype, fill) -> np.ndarray:
    if x is None:
        return np.full(n, fill, dtype=dtype)
    a = np.asarray(x)
    if a.shape != (n,):
        raise ObserverError(f"array of shape {a.shape} does not match {n} names")
    return a.astype(dtype, copy=False)


@dataclass(frozen=True, eq=False)
class DecisionSnapshot:
    """What the model knew and did at the decision close. `score` is NaN where the name was never scored; `confidence` and
    `dir_prob` are NaN where the model gave none. `filters` is a bitmask into `filter_names` (a hard risk / liquidity filter
    removed the name before ranking). Everything here is point-in-time: nothing may be dated after `decided_at`."""
    decided_at: str
    tickers: np.ndarray
    eligible: np.ndarray
    score: np.ndarray
    confidence: np.ndarray
    dir_prob: np.ndarray
    picked: np.ndarray
    abstained: np.ndarray
    filters: np.ndarray
    prev_close: np.ndarray
    event_known: np.ndarray
    sector: np.ndarray
    features: Mapping[str, np.ndarray]
    filter_names: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.tickers)

    @staticmethod
    def make(decided_at, tickers: Sequence, prev_close, *, eligible=None, score=None, confidence=None, dir_prob=None, picked=None,
             abstained=None, filters=None, event_known=None, sector=None, features: Mapping[str, Any] | None = None,
             filter_names: Sequence[str] = ()) -> "DecisionSnapshot":
        tk = np.asarray(list(tickers), dtype=object)
        n = len(tk)
        feats = {k: _arr(v, n, np.float32, np.nan) for k, v in (features or {}).items()}
        return DecisionSnapshot(str(as_date(decided_at)), tk, _arr(eligible, n, bool, True), _arr(score, n, np.float64, np.nan),
                                _arr(confidence, n, np.float64, np.nan), _arr(dir_prob, n, np.float64, np.nan),
                                _arr(picked, n, bool, False), _arr(abstained, n, bool, False), _arr(filters, n, np.int64, 0),
                                _arr(prev_close, n, np.float64, np.nan), _arr(event_known, n, bool, False),
                                _arr(sector, n, np.int64, -1), feats, tuple(filter_names))

    def validate(self) -> list[str]:
        e = []
        if len(set(self.tickers.tolist())) != self.n:
            e.append("duplicate tickers in the snapshot")
        if self.n and np.any(self.picked & ~self.eligible):
            e.append("a name was picked although it is ineligible")
        if self.n and np.any(self.picked & (self.filters != 0)):
            e.append("a name was picked although a hard filter removed it")
        if self.n and np.any(np.isinf(self.score)):
            e.append("infinite score")
        conf = np.nan_to_num(self.confidence, nan=0.5)
        if self.n and (np.any(conf < 0) or np.any(conf > 1)):
            e.append("confidence outside [0, 1]")
        if any(str(k).lower() in ("ticker", "symbol", "name", "date", "year") for k in self.features):
            e.append("identity key among the features")
        if self.filters.size and self.filters.max(initial=0) >= (1 << max(1, len(self.filter_names))) and self.filter_names:
            e.append("filter bit beyond the named filters")
        return e

    def digest(self) -> str:
        """Content hash of the point-in-time state, so one snapshot serves every event of the day and reruns are provably the same."""
        h = hashlib.sha256(self.decided_at.encode())
        for a in (self.score, self.picked, self.eligible, self.filters, self.prev_close):
            h.update(np.ascontiguousarray(a).tobytes())
        for k in sorted(self.features):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.features[k]).tobytes())
        h.update("|".join(map(str, self.tickers)).encode())
        return h.hexdigest()[:16]


@dataclass(frozen=True, eq=False)
class DayOutcome:
    """The matured result for the same names. Prices are over the holding horizon: `entry` is the next session's open (the
    fill), `hi` / `lo` the extreme prices reached, `close` the horizon close. NaN entry = no fill / no data (never read as 0)."""
    resolved_at: str
    tickers: np.ndarray
    entry: np.ndarray
    hi: np.ndarray
    lo: np.ndarray
    close: np.ndarray
    volume_ratio: np.ndarray
    delisted: np.ndarray
    day_hi: np.ndarray                    # the FIRST session's own high / low / close (the mover day, C67 bands)
    day_lo: np.ndarray
    day_close: np.ndarray

    @staticmethod
    def make(resolved_at, tickers: Sequence, entry, hi, lo, close, volume_ratio=None, delisted=None, day_hi=None, day_lo=None,
             day_close=None) -> "DayOutcome":
        """day_* default to the horizon values, which is exact when the horizon is one session."""
        tk = np.asarray(list(tickers), dtype=object)
        n = len(tk)
        f = lambda x: _arr(x, n, np.float64, np.nan)
        return DayOutcome(str(as_date(resolved_at)), tk, f(entry), f(hi), f(lo), f(close), f(volume_ratio), _arr(delisted, n, bool, False),
                          f(hi if day_hi is None else day_hi), f(lo if day_lo is None else day_lo),
                          f(close if day_close is None else day_close))

    def validate(self) -> list[str]:
        e = []
        if len(set(self.tickers.tolist())) != len(self.tickers):
            e.append("duplicate tickers in the outcome")
        ok = np.isfinite(self.entry) & np.isfinite(self.hi) & np.isfinite(self.lo) & np.isfinite(self.close)
        if np.any(ok & ((self.lo > self.hi * (1 + 1e-9)) | (self.entry <= 0) | (self.close <= 0))):
            e.append("impossible bar: low above high, or a non-positive price")
        if np.any(ok & ((self.close > self.hi * (1 + 1e-9)) | (self.close < self.lo * (1 - 1e-9)))):
            e.append("close outside the high-low range")
        if np.any(ok & ((self.entry > self.hi * (1 + 1e-9)) | (self.entry < self.lo * (1 - 1e-9)))):
            e.append("fill outside the high-low range")
        return e


def align_outcome(snap: DecisionSnapshot, out: DayOutcome) -> tuple[DayOutcome, int]:
    """Reindex the outcome onto the snapshot's names. Names the outcome lacks get NaN (counted, never dropped); names the
    outcome has that the snapshot lacks are reported as an error (an outcome for something the model never saw is a leak)."""
    if out.tickers.shape == snap.tickers.shape and bool(np.all(out.tickers == snap.tickers)):
        return out, 0
    pos = pd.Index(out.tickers).get_indexer(snap.tickers)
    extra = set(out.tickers.tolist()) - set(snap.tickers.tolist())
    if extra:
        raise ObserverError(f"outcome has {len(extra)} names absent from the decision snapshot")
    miss = pos < 0
    take = np.where(miss, 0, pos)

    def g(a, fill):
        r = a[take].copy() if len(a) else np.full(len(pos), fill, dtype=a.dtype)
        r[miss] = fill
        return r
    return (DayOutcome(out.resolved_at, snap.tickers, g(out.entry, np.nan), g(out.hi, np.nan), g(out.lo, np.nan),
                       g(out.close, np.nan), g(out.volume_ratio, np.nan), g(out.delisted, False), g(out.day_hi, np.nan),
                       g(out.day_lo, np.nan), g(out.day_close, np.nan)), int(miss.sum()))


# ==================================================================================================================
# vectorised primitives
# ==================================================================================================================
def pct_rank(x: np.ndarray) -> np.ndarray:
    """Ascending percentile rank in (0, 1]; NaN stays NaN; a constant column carries no ordering and becomes 0.5."""
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    v = x[m]
    out[m] = 0.5 if v.min() == v.max() else rankdata(v, method="average") / len(v)
    return out


def desc_rank(x: np.ndarray) -> np.ndarray:
    """1 = highest. Ties are broken by position (stable), so a rank is always a distinct integer; NaN gets no rank (0)."""
    out = np.zeros(len(x), dtype=np.int64)
    idx = np.flatnonzero(np.isfinite(x))
    order = idx[np.argsort(-x[idx], kind="stable")]
    out[order] = np.arange(1, len(order) + 1)
    return out


def precursor_score(features: Mapping[str, np.ndarray], spec: Sequence[tuple[str, float]], n: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Cross-sectional 0..1 score of how much the day-before picture looked like the eve of a big move: the weighted mean of
    the ranks of the available precursor features (a negative weight rewards a LOW rank). Names with none of them get NaN."""
    num, den, used = np.zeros(n), np.zeros(n), []
    for name, w in spec:
        f = features.get(name)
        if f is None or w == 0:
            continue
        r = pct_rank(np.asarray(f, dtype=np.float64))
        r = r if w > 0 else 1.0 - r + 1.0 / max(1, np.isfinite(r).sum())
        ok = np.isfinite(r)
        num[ok] += abs(w) * np.minimum(r[ok], 1.0)
        den[ok] += abs(w)
        used.append(name)
    out = np.full(n, np.nan)
    m = den > 0
    out[m] = num[m] / den[m]
    return out, tuple(used)


def gap_share(gap: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Share of the total path move that arrived as the overnight gap: |gap| / (|gap| + |fill-to-close move|)."""
    a, b = np.abs(gap), np.abs(ret)
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, a / den, np.nan)


def _cap_indices(idx: np.ndarray, key: np.ndarray, cap: int, keep_share: float, rng: np.random.Generator) -> np.ndarray:
    """At most `cap` of idx: the `keep_share` largest by key outright, the rest a seeded sample of the remainder."""
    if len(idx) <= cap:
        return idx
    n_top = max(1, int(round(cap * keep_share)))
    order = idx[np.argsort(-key[idx], kind="stable")]
    top, rest = order[:n_top], order[n_top:]
    pick = rng.choice(len(rest), size=cap - n_top, replace=False) if cap > n_top else np.array([], dtype=int)
    return np.concatenate([top, rest[np.sort(pick)]])


def _topk(key: np.ndarray, k: int, valid: np.ndarray, largest: bool = True) -> np.ndarray:
    idx = np.flatnonzero(valid & np.isfinite(key))
    if not len(idx):
        return idx
    order = idx[np.argsort(-key[idx] if largest else key[idx], kind="stable")]
    return order[:k]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC of `score` for a boolean `label` over the rows where both exist; NaN if either class is empty."""
    m = np.isfinite(score)
    y = label[m].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score[m], method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ==================================================================================================================
# the day's classification (every name, every category, no persistence yet)
# ==================================================================================================================
@dataclass(frozen=True, eq=False)
class DayClass:
    """Full-cross-section classification of one day. Arrays are aligned with the snapshot's tickers."""
    masks: Mapping[MoveCategory, np.ndarray]
    ret: np.ndarray                       # fill -> horizon close
    sig_ret: np.ndarray                   # decision close -> horizon close (what the market did after the information date)
    gap: np.ndarray                       # decision close -> fill
    hi_x: np.ndarray
    lo_x: np.ndarray
    exc: np.ndarray                       # best excursion either way from the fill (what a holder could have captured)
    exc_pc: np.ndarray                    # best excursion either way from the DECISION close (what the market did; includes the gap)
    rng: np.ndarray
    vol_x: np.ndarray                     # realised horizon range over the range the name's own ATR promised
    c2c: np.ndarray                       # mover-day close over the previous close (C67 primary band basis)
    o2c: np.ndarray                       # mover-day open (the fill) to close
    hl_range: np.ndarray                  # mover-day (high - low) over the previous close
    close_loc: np.ndarray                 # where the mover day closed inside its own range: 0 = at the low, 1 = at the high
    band: np.ndarray                      # C67 band of |c2c|: 0 none, +-1 the 5-10% band, +-2 the >10% band (sign = direction)
    band_o2c: np.ndarray                  # the same band read off open-to-close
    suspect: np.ndarray                   # 0 = clean; 1 split-like, 2 extreme, 3 no trade (SUSPECT_NAMES)
    rank: np.ndarray                      # 1 = best score; 0 = never scored
    precursor: np.ndarray
    gshare: np.ndarray
    moved: np.ndarray                     # the name moved substantially, measured from the decision close (market truth)
    moved_fill: np.ndarray                # ... measured from the fill (tradeable truth)
    valid: np.ndarray
    cutoff_score: float
    used_precursors: tuple[str, ...]

    def mask(self, cat: MoveCategory) -> np.ndarray:
        return self.masks[cat]


def outcome_arrays(snap: DecisionSnapshot, out: DayOutcome, horizon: int) -> dict[str, np.ndarray]:
    """Return / excursion arrays from prices. Anything that would divide by a missing or non-positive price is NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(out.entry > 0, out.entry, np.nan)
        pc = np.where(snap.prev_close > 0, snap.prev_close, np.nan)
        ret, hi_x, lo_x = out.close / e - 1.0, out.hi / e - 1.0, out.lo / e - 1.0
        gap, sig = e / pc - 1.0, out.close / pc - 1.0
        rng = (out.hi - out.lo) / e
        exc_pc = np.fmax(out.hi / pc - 1.0, 1.0 - out.lo / pc)
        atr = snap.features.get("atr")
        vol_x = rng / (np.asarray(atr, dtype=np.float64) * math.sqrt(horizon)) if atr is not None else np.full(len(e), np.nan)
    vol_x = np.where(np.isfinite(vol_x) & (vol_x >= 0) & (vol_x < 1e6), vol_x, np.nan)
    valid = np.isfinite(ret) & np.isfinite(hi_x) & np.isfinite(lo_x)
    with np.errstate(invalid="ignore", divide="ignore"):
        c2c = out.day_close / pc - 1.0
        o2c = out.day_close / e - 1.0
        hl = (out.day_hi - out.day_lo) / pc
        span = out.day_hi - out.day_lo
        loc = np.where(span > 0, (out.day_close - out.day_lo) / span, np.nan)
    return {"ret": ret, "sig_ret": np.where(np.isfinite(sig), sig, np.nan), "gap": gap, "hi_x": hi_x, "lo_x": lo_x, "rng": rng,
            "vol_x": vol_x, "valid": valid, "exc": np.fmax(hi_x, -lo_x), "exc_pc": exc_pc, "c2c": c2c, "o2c": o2c, "hl_range": hl,
            "close_loc": np.clip(loc, 0.0, 1.0)}


SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
SUSPECT_NAMES = {0: "", 1: "split_like", 2: "extreme_move", 3: "no_trade"}


def suspect_codes(snap: DecisionSnapshot, out: DayOutcome, c2c: np.ndarray, extreme: float = 0.6, split_tol: float = 0.03) -> np.ndarray:
    """A giant band mover is more often a data failure than a market event: 1 = the close/previous-close ratio sits on a split
    ratio (2, 3, 4, 5, 10, 20 or their inverses), 2 = a move beyond `extreme`, 3 = a bar with no range and no volume (a stale
    print). Suspect names are still counted and kept (never silently dropped); the autopsy reports them as DATA_FAILURE, not as
    movers to learn from."""
    code = np.zeros(snap.n, dtype=np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = 1.0 + c2c
        for k in SPLIT_RATIOS:
            near = (np.abs(ratio / k - 1.0) < split_tol) | (np.abs(ratio * k - 1.0) < split_tol)
            code[near & np.isfinite(ratio)] = 1
    code[(np.abs(c2c) > extreme) & (code == 0)] = 2
    stale = (out.day_hi == out.day_lo) & (np.nan_to_num(out.volume_ratio, nan=1.0) <= 0.0)
    code[stale & (code == 0)] = 3
    return code


def band_code(move: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """C67 band of a move: +-1 for lo <= |move| < hi, +-2 for |move| >= hi, 0 otherwise or when the move is missing."""
    a = np.abs(move)
    mag = np.where(np.isfinite(a), np.where(a >= hi, 2, np.where(a >= lo, 1, 0)), 0)
    return (np.sign(np.nan_to_num(move)) * mag).astype(np.int8)


BAND_NAMES = {-2: "down_gt10", -1: "down_5_10", 1: "up_5_10", 2: "up_gt10"}


def band_counts(band: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    b = band if mask is None else band[mask]
    return {name: int((b == code).sum()) for code, name in BAND_NAMES.items()}


def classify_day(snap: DecisionSnapshot, out: DayOutcome, p: ObserverParams) -> DayClass:
    """The section-4 categories A-I for every name at once. `out` must already be aligned to `snap`."""
    n = snap.n
    a = outcome_arrays(snap, out, p.horizon)
    ret, sig, valid, exc = a["ret"], a["sig_ret"], a["valid"], a["exc"]
    scored = np.isfinite(snap.score)
    rank_score = np.where(scored, snap.score, np.nan)
    rank = desc_rank(rank_score)
    considered = scored & snap.eligible & (snap.filters == 0)
    picked = snap.picked
    hard = snap.filters != 0
    m: dict[MoveCategory, np.ndarray] = {}
    m[MC.CONSIDERED_HIGH] = considered & (rank <= p.k_high) & (rank > 0)
    m[MC.CONSIDERED_MEDIUM] = considered & (rank > p.k_high) & (rank <= p.k_med)
    m[MC.REJECTED] = ~picked & ((scored & hard) | (considered & (rank <= p.k_med))) | (~picked & ~snap.eligible & scored)
    m[MC.ABSTAINED] = snap.abstained & ~picked
    m[MC.LOW_CONFIDENCE] = considered & (rank <= p.k_med) & np.isfinite(snap.confidence) & (snap.confidence < p.conf_gate)
    if picked.any():
        cutoff = float(np.nanmin(np.where(picked, snap.score, np.nan)))
    else:
        top = np.sort(snap.score[considered])[::-1]
        cutoff = float(top[min(p.k_pick, len(top)) - 1]) if len(top) else float("nan")
    near_rank = int(math.ceil(p.k_pick * (1 + p.near_frac)))
    near_score = np.isfinite(cutoff) & (snap.score >= cutoff - p.near_score_gap * abs(cutoff))
    m[MC.NEAR_MISS] = considered & ~picked & (((rank > 0) & (rank <= near_rank)) | near_score)
    m[MC.WINNER] = np.isfinite(sig) & (sig >= p.win_thr)
    m[MC.LOSER] = np.isfinite(sig) & (sig <= -p.win_thr)
    nv = int(np.isfinite(sig).sum())
    q = max(1, int(math.ceil(p.tail_q * nv))) if nv else 0
    if q:
        s = np.where(np.isfinite(sig), sig, np.nan)
        hi_cut, lo_cut = np.sort(s[np.isfinite(s)])[-q], np.sort(s[np.isfinite(s)])[q - 1]
        m[MC.EXTREME_UP] = np.isfinite(s) & (s >= hi_cut) & (s >= p.extreme_floor)
        m[MC.EXTREME_DOWN] = np.isfinite(s) & (s <= lo_cut) & (s <= -p.extreme_floor)
    else:
        m[MC.EXTREME_UP] = m[MC.EXTREME_DOWN] = np.zeros(n, dtype=bool)
    moved = valid & np.isfinite(a["exc_pc"]) & (a["exc_pc"] >= p.mover_thr)
    moved_fill = valid & (exc >= p.mover_thr)
    prec, used = precursor_score(snap.features, p.precursors, n)
    gs = gap_share(a["gap"], ret)
    score_pct = pct_rank(rank_score)
    top_score = np.isfinite(score_pct) & (score_pct >= 1.0 - p.score_top_frac)
    prec_hi = np.isfinite(prec) & (prec >= p.pred_q)
    pred = moved & (prec_hi | snap.event_known | top_score)
    unpred = moved & ~pred & np.isfinite(gs) & (gs >= p.gap_share_thr) & np.isfinite(prec) & (prec < p.unpred_q) & ~snap.event_known
    m[MC.PREDICTABLE_MOVER], m[MC.UNPREDICTABLE_MOVER] = pred, unpred
    predicted_to_move = picked | m[MC.CONSIDERED_HIGH]
    m[MC.FALSE_POSITIVE] = predicted_to_move & valid & ~moved
    m[MC.FALSE_NEGATIVE] = moved & ~picked
    for c in CATEGORIES:
        if c not in m:
            raise ObserverError(f"category {c} was not classified")
        if m[c].shape != (n,):
            raise ObserverError(f"category {c} mask has shape {m[c].shape}")
    band = band_code(a["c2c"], p.band_lo, p.band_hi)
    band_o = band_code(a["o2c"], p.band_lo, p.band_hi)
    return DayClass(m, ret, sig, a["gap"], a["hi_x"], a["lo_x"], exc, a["exc_pc"], a["rng"], a["vol_x"], a["c2c"],
                    a["o2c"], a["hl_range"], a["close_loc"], band, band_o, suspect_codes(snap, out, a["c2c"]), rank, prec, gs, moved, moved_fill, valid, cutoff, used)


def flag_bits(masks: Mapping[MoveCategory, np.ndarray], idx: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(idx), dtype=np.int64)
    for c, mk in masks.items():
        bits |= np.where(mk[idx], BIT[c], 0)
    return bits


def has(bits: Any, cat: MoveCategory) -> Any:
    """True where the category bit is set (works on a scalar, an array or a Series)."""
    return (np.asarray(bits) & BIT[cat]) != 0


def cats_of(bits: int) -> list[MoveCategory]:
    return [c for c in CATEGORIES if int(bits) & BIT[c]]


# ==================================================================================================================
# which rows are kept
# ==================================================================================================================
def market_tops(dc: DayClass, snap: DecisionSnapshot, out: DayOutcome, p: ObserverParams) -> dict[str, np.ndarray]:
    """Indices of the largest / smallest names for each market metric (autopsy MARKET). Always kept as rows."""
    v, k = dc.valid | np.isfinite(dc.sig_ret), p.n_top
    vol = out.volume_ratio
    return {"gainers": _topk(dc.sig_ret, k, v), "losers": _topk(dc.sig_ret, k, v, largest=False),
            "ranges": _topk(dc.rng, k, v), "volume": _topk(vol, k, np.ones(snap.n, dtype=bool)),
            "gaps": _topk(np.abs(dc.gap), k, np.isfinite(dc.gap)), "vol_expansion": _topk(dc.vol_x, k, np.isfinite(dc.vol_x)),
            "vol_contraction": _topk(dc.vol_x, k, np.isfinite(dc.vol_x), largest=False)}


def select_rows(dc: DayClass, snap: DecisionSnapshot, tops: Mapping[str, np.ndarray], p: ObserverParams
                ) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """The exception rows. Every pick, every metric-table name and every near miss / abstention within its cap is kept;
    outcome categories keep the largest moves and a seeded sample of the overflow. Returns (indices, kept per category, dropped)."""
    rng = np.random.default_rng(int(stable_hash([p.sample_seed, snap.decided_at], 8), 16))
    keep = np.zeros(snap.n, dtype=bool)
    keep[snap.picked] = True
    keep[dc.band != 0] = True                          # C67: EVERY 5-10% and >10% mover is kept, up and down, uncapped
    for ix in tops.values():
        keep[ix] = True
    kept, dropped = {}, {}
    move_key = np.nan_to_num(np.abs(dc.sig_ret), nan=0.0)
    score_key = np.nan_to_num(snap.score, nan=-1e9)
    for c in CATEGORIES:
        idx = np.flatnonzero(dc.masks[c])
        key = score_key if c in MODEL_CATS or c is MC.FALSE_POSITIVE else np.maximum(move_key, np.nan_to_num(dc.exc_pc, nan=0.0))
        if c is MC.CONSIDERED_MEDIUM:
            key = -dc.rank.astype(np.float64)
        sel = _cap_indices(idx, key, p.max_rows_per_cat, p.keep_top_share, rng)
        keep[sel] = True
        kept[c.value], dropped[c.value] = int(len(sel)), int(len(idx) - len(sel))
    return np.flatnonzero(keep), kept, dropped


# ==================================================================================================================
# the daily record
# ==================================================================================================================
@dataclass(frozen=True, eq=False)
class DayRecord:
    """One day's complete research record (MATURED_RESEARCH_STATE). Exact counts for every category over the whole universe;
    rows only for the exceptions; `kept`/`dropped` say how much of each category the rows represent."""
    day: str
    decided_at: str
    resolved_at: str
    params_hash: str
    snapshot_digest: str
    n_universe: int
    n_eligible: int
    n_scored: int
    n_no_outcome: int
    counts: Mapping[str, int]
    counts_eligible: Mapping[str, int]
    bands: Mapping[str, int]                          # C67: movers per band over the whole universe (close-to-close)
    bands_eligible: Mapping[str, int]
    bands_o2c: Mapping[str, int]                      # the same read off open-to-close
    kept: Mapping[str, int]
    dropped: Mapping[str, int]
    rows: pd.DataFrame
    tops: Mapping[str, tuple[str, ...]]
    market: Mapping[str, Any]
    model: Mapping[str, Any]
    filter_names: tuple[str, ...] = ()
    used_precursors: tuple[str, ...] = ()
    empty: bool = False
    name_ids: np.ndarray | None = None                # uint64 hash of every name in the universe (dropped by the ledger after churn)

    def category_rows(self, cat: MoveCategory) -> pd.DataFrame:
        if self.rows.empty:
            return self.rows
        return self.rows[has(self.rows["flags"].to_numpy(), cat)]

    def n(self, cat: MoveCategory) -> int:
        return int(self.counts.get(cat.value, 0))

    def coverage(self, cat: MoveCategory) -> float:
        """Share of the category the persisted rows represent (1.0 = all of it)."""
        tot = self.counts.get(cat.value, 0)
        return 1.0 if tot == 0 else self.kept.get(cat.value, 0) / tot

    def validate(self) -> list[str]:
        e = []
        if as_date(self.resolved_at) <= as_date(self.decided_at):
            e.append("resolved_at must be after decided_at")
        for c in CATEGORIES:
            if c.value not in self.counts:
                e.append(f"count missing for {c.value}")
            elif self.counts[c.value] < 0 or self.counts[c.value] > self.n_universe:
                e.append(f"count for {c.value} outside [0, universe]")
            elif self.kept.get(c.value, 0) + self.dropped.get(c.value, 0) != self.counts[c.value]:
                e.append(f"kept + dropped != count for {c.value}")
        if self.counts.get(MC.FALSE_NEGATIVE.value, 0) > self.n_universe:
            e.append("more false negatives than names")
        if sum(self.bands.values()) and (self.rows.empty or int((self.rows["band"] != 0).sum()) != sum(self.bands.values())):
            e.append("a band mover has no row: C67 requires every band mover to be kept")
        if not self.rows.empty:
            if self.rows["ticker"].duplicated().any():
                e.append("duplicate rows")
            if len(self.rows) > self.n_universe:
                e.append("more rows than names")
        return e

    def as_matured(self, prov: Provenance) -> MaturedRecord:
        """Wrap the identity-free summary for the research world. Rows stay behind: they carry tickers."""
        return MaturedRecord(stable_hash([self.day, self.snapshot_digest, self.params_hash], 16), self.resolved_at,
                             {"day": self.day, "counts": dict(self.counts), "market": dict(self.market), "model": dict(self.model),
                              "n_universe": self.n_universe}, prov)

    def summary(self) -> dict[str, Any]:
        return {"day": self.day, "n_universe": self.n_universe, "n_eligible": self.n_eligible, "counts": dict(self.counts),
                "market": dict(self.market), "model": dict(self.model)}


MOVE_EDGES = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.20, 0.30)


def move_histogram(c2c: np.ndarray) -> dict[str, list[int]]:
    """Counts of |close-to-close| moves in fine bins from 5% up (up and down separately), the shape behind the two C67 bands:
    a day whose 5-10% count is dominated by 5-6% names is a different day from one full of 9-10% names."""
    ok = np.isfinite(c2c)
    edges = np.array(MOVE_EDGES + (np.inf,))
    out = {}
    for side, m in (("up", ok & (c2c >= MOVE_EDGES[0])), ("down", ok & (c2c <= -MOVE_EDGES[0]))):
        out[side] = np.bincount(np.searchsorted(edges, np.abs(c2c[m]), side="right") - 1, minlength=len(MOVE_EDGES)).tolist()
    return out


def sector_table(sector: np.ndarray, band_mask: np.ndarray) -> dict[str, list[int]]:
    """[names, band movers] per sector code (-1 = unknown is left out)."""
    known = sector >= 0
    if not known.any():
        return {}
    codes, inv = np.unique(sector[known], return_inverse=True)
    n = np.bincount(inv, minlength=len(codes))
    m = np.bincount(inv, weights=band_mask[known].astype(float), minlength=len(codes))
    return {str(int(c)): [int(a), int(b)] for c, a, b in zip(codes, n, m)}


def feature_health(snap: DecisionSnapshot) -> dict[str, dict[str, float]]:
    """Per feature: share missing, share non-finite, and whether the cross-section is constant (a stale or broken feed shows up
    as a constant column or a sudden hole long before it shows up in a result)."""
    out = {}
    for k, v in snap.features.items():
        a = np.asarray(v, dtype=np.float64)
        fin = np.isfinite(a)
        out[k] = {"missing": float(1.0 - fin.mean()) if len(a) else 1.0, "constant": float(fin.any() and np.ptp(a[fin]) == 0.0)}
    return out


def decile_table(x: np.ndarray, moved: np.ndarray, valid: np.ndarray, bins: int = 10) -> list[list[int]]:
    """[names, movers] per equal-count bin of `x` (bin 0 = lowest). It shows in one row whether a score orders movers at all."""
    ok = np.isfinite(x) & valid
    if ok.sum() < bins:
        return []
    r = pct_rank(np.where(ok, x, np.nan))[ok]
    b = np.minimum(bins - 1, np.floor((r - 1e-12) * bins).astype(int))
    mv = moved[ok]
    return [[int((b == i).sum()), int(mv[b == i].sum())] for i in range(bins)]


def _summary_stats(dc: DayClass, snap: DecisionSnapshot, out: DayOutcome, p: ObserverParams) -> tuple[dict, dict]:
    """Market and model summaries for the day. Everything is a cross-sectional statistic; nothing names a stock."""
    sig = dc.sig_ret[np.isfinite(dc.sig_ret)]
    med = float(np.median(sig)) if len(sig) else float("nan")
    mad = float(1.4826 * np.median(np.abs(sig - med))) if len(sig) else float("nan")
    vx = dc.vol_x[np.isfinite(dc.vol_x)]
    vr = out.volume_ratio[np.isfinite(out.volume_ratio)]
    valid = dc.valid
    market = {"bands": band_counts(dc.band), "n_band_movers": int((dc.band != 0).sum()),
              "median_ret": med, "mean_ret": float(sig.mean()) if len(sig) else float("nan"), "dispersion": mad,
              "breadth_up": float((sig > 0).mean()) if len(sig) else float("nan"),
              "p05": float(np.quantile(sig, 0.05)) if len(sig) else float("nan"),
              "p95": float(np.quantile(sig, 0.95)) if len(sig) else float("nan"),
              "mover_rate": float(dc.moved[valid].mean()) if valid.any() else float("nan"),
              "median_vol_x": float(np.median(vx)) if len(vx) else float("nan"),
              "median_volume_ratio": float(np.median(vr)) if len(vr) else float("nan"),
              "gap_median_abs": float(np.nanmedian(np.abs(dc.gap))) if np.isfinite(dc.gap).any() else float("nan"),
              "delisted": int(out.delisted.sum())}
    market["move_hist"] = move_histogram(dc.c2c)
    market["sector_movers"] = sector_table(snap.sector, dc.band != 0)
    market["n_moved"] = int(dc.moved.sum())
    market["n_suspect"] = int((dc.suspect != 0).sum())
    market["n_suspect_band"] = int(((dc.suspect != 0) & (dc.band != 0)).sum())
    market["tone"] = ("up" if med > 0.005 else "down" if med < -0.005 else "flat") if len(sig) else "unknown"
    market["dispersion_level"] = "wide" if mad > 0.03 else "narrow" if mad < 0.01 else "normal"
    pk = snap.picked & valid
    hit = float(dc.moved_fill[pk].mean()) if pk.any() else float("nan")
    movers = int(dc.moved.sum())
    model = {"n_picked": int(snap.picked.sum()), "pick_mean_ret": float(dc.ret[pk].mean()) if pk.any() else float("nan"),
             "pick_hit_rate": hit, "universe_mean_ret": float(dc.ret[valid].mean()) if valid.any() else float("nan"),
             "recall_of_movers": float((dc.moved & snap.picked).sum() / movers) if movers else float("nan"),
             "score_auc": auc(np.where(np.isfinite(snap.score), snap.score, np.nan), dc.moved & valid),
             "precursor_auc": auc(dc.precursor, dc.moved & valid), "cutoff_score": dc.cutoff_score,
             "score_coverage": float(np.isfinite(snap.score).mean()) if snap.n else float("nan")}
    hi = np.isfinite(dc.precursor) & (dc.precursor >= p.pred_q) & valid
    base_rate = market["mover_rate"]
    model["precursor_lift"] = float(dc.moved[hi].mean() / base_rate) if hi.any() and base_rate and base_rate == base_rate else float("nan")
    model["score_deciles"] = decile_table(np.where(np.isfinite(snap.score), snap.score, np.nan), dc.moved, valid)
    model["precursor_deciles"] = decile_table(dc.precursor, dc.moved, valid)
    tp, fp = int((snap.picked & dc.moved_fill).sum()), int((snap.picked & valid & ~dc.moved_fill).sum())
    fn_, tn = int((~snap.picked & dc.moved_fill).sum()), int((~snap.picked & valid & ~dc.moved_fill).sum())
    den = math.sqrt(float(tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
    model["confusion_fill"] = {"tp": tp, "fp": fp, "fn": fn_, "tn": tn}
    model["mcc_fill"] = (tp * tn - fp * fn_) / den if den > 0 else float("nan")
    market["data_health"] = feature_health(snap)
    model["picks_in_movers_share"] = float(snap.picked[dc.moved].mean()) if movers else float("nan")
    return market, model


# ==================================================================================================================
# building the record
# ==================================================================================================================
ROW_SCHEMA: dict[str, str] = {
    "ticker": "research-side identity (never handed to the trader)", "cid": "salted hash id, safe to export",
    "flags": "bitmask over CATEGORIES (has(flags, MoveCategory.X))", "band": "C67 band of |close-to-close|: 0, +-1 (5-10%), +-2 (>10%)",
    "band_o2c": "the same band read off open-to-close", "c2c": "mover-day close / previous close - 1", "o2c": "mover-day close / fill(open) - 1",
    "hl_range": "(day high - day low) / previous close", "close_loc": "0 = closed at the day low, 1 = at the day high",
    "gap": "fill / previous close - 1", "ret": "horizon close / fill - 1", "sig_ret": "horizon close / previous close - 1",
    "hi_x": "best excursion up from the fill", "lo_x": "worst excursion from the fill", "exc": "largest excursion either way from the fill",
    "exc_pc": "largest excursion either way from the decision close (includes the gap)",
    "rng": "horizon (high - low) / fill", "vol_x": "realised range over the range the name's ATR promised",
    "volume_ratio": "session volume over its 20-day average", "score": "model score at the decision close (NaN = unscored)",
    "rank": "1 = best score, 0 = unscored", "confidence": "model confidence (NaN = none given)", "dir_prob": "model P(up) (NaN = none)",
    "picked": "the model held it", "eligible": "passed the universe filters", "abstained": "the model abstained",
    "filters": "bitmask of hard filters that removed it", "event_known": "a scheduled event was known beforehand",
    "sector": "opaque sector code (-1 unknown)", "precursor": "0..1 eve-of-move look score (proxy for predictability)",
    "gap_share": "share of the path move that arrived as the gap", "mover_type": "engine.missed_winners.winner_types label",
    "delisted": "the name stopped trading in the window", "suspect": "0 clean; 1 split-like, 2 extreme, 3 no trade (a data failure, not a mover)", "rk_*": "cross-sectional percentile rank of a PIT feature (identity-free)",
    "vol20": "raw 20-day volatility at the decision close", "atr": "raw ATR at the decision close"}
RECORD_SCHEMA: dict[str, str] = {
    "counts": "exact count of every MoveCategory over the whole universe", "counts_eligible": "the same over eligible names only",
    "bands": "C67: movers per band (up_5_10, up_gt10, down_5_10, down_gt10), close-to-close, whole universe",
    "bands_o2c": "the same read off open-to-close", "rows": "exception rows (ROW_SCHEMA); every band mover is a row",
    "kept/dropped": "rows kept vs sampled away per category (kept + dropped == count)", "tops": "ticker tuples per market metric",
    "market": "cross-sectional summary of the day", "model": "how the model did on the day"}


def _mover_types(snap: DecisionSnapshot) -> np.ndarray:
    """engine.missed_winners.winner_types over the full cross-section (vectorised); 'other' where its inputs are absent."""
    f = snap.features
    cols = {"vol20": f.get("vol20"), "e_dist_52wh": f.get("dist_hi"), "log_dv": f.get("log_dv"),
            "e_ear": np.where(snap.event_known, 1.0, np.nan).astype(np.float32)}
    p0 = pd.DataFrame({k: v for k, v in cols.items() if v is not None}, index=pd.Index(snap.tickers, name="ticker"))
    if len(p0) < 4:
        return np.full(snap.n, "other", dtype=object)
    return base.winner_types(p0).to_numpy(dtype=object)


def _rows(snap: DecisionSnapshot, out: DayOutcome, dc: DayClass, idx: np.ndarray, p: ObserverParams) -> pd.DataFrame:
    if not len(idx):
        return pd.DataFrame({"ticker": pd.Series(dtype=object), "flags": pd.Series(dtype=np.int64)})
    cols: dict[str, Any] = {"ticker": snap.tickers[idx],
                            "cid": [stable_hash({"s": p.salt, "d": snap.decided_at, "t": str(t)}, 12) for t in snap.tickers[idx]],
                            "flags": flag_bits(dc.masks, idx), "band": dc.band[idx], "band_o2c": dc.band_o2c[idx]}
    for name, arr in (("c2c", dc.c2c), ("o2c", dc.o2c), ("hl_range", dc.hl_range), ("close_loc", dc.close_loc), ("gap", dc.gap),
                      ("ret", dc.ret), ("sig_ret", dc.sig_ret), ("hi_x", dc.hi_x), ("lo_x", dc.lo_x), ("exc", dc.exc), ("exc_pc", dc.exc_pc),
                      ("rng", dc.rng), ("vol_x", dc.vol_x), ("volume_ratio", out.volume_ratio), ("score", snap.score),
                      ("rank", dc.rank), ("confidence", snap.confidence), ("dir_prob", snap.dir_prob), ("picked", snap.picked),
                      ("eligible", snap.eligible), ("abstained", snap.abstained), ("filters", snap.filters),
                      ("event_known", snap.event_known), ("sector", snap.sector), ("precursor", dc.precursor),
                      ("gap_share", dc.gshare), ("delisted", out.delisted), ("suspect", dc.suspect)):
        cols[name] = arr[idx]
    cols["mover_type"] = _mover_types(snap)[idx]
    for name in p.rank_features:
        f = snap.features.get(name)
        if f is not None:
            cols["rk_" + name] = pct_rank(np.asarray(f, dtype=np.float64))[idx].astype(np.float32)
    for raw in ("vol20", "atr"):
        if raw in snap.features:
            cols[raw] = snap.features[raw][idx]
    return pd.DataFrame(cols)


def winner_reasons(snap: DecisionSnapshot, dc: DayClass, p: ObserverParams) -> dict[str, int]:
    """Why the largest winners were or were not owned, via engine.missed_winners.why_missed (picked / ineligible / below_cut /
    unscored). Limited to the `explain_limit` largest tradeable winners a day: why_missed ranks the cross-section per winner."""
    if p.explain_limit <= 0 or snap.n < 4:
        return {}
    ret = pd.Series(dc.ret, index=snap.tickers)
    win = ret[(ret >= p.win_thr) & np.isfinite(ret)].nlargest(p.explain_limit)
    if win.empty:
        return {}
    cols = {k: snap.features[k] for k in ("log_dv", "vol20", "dist_hi") if k in snap.features}
    p0 = pd.DataFrame(cols, index=snap.tickers) if cols else pd.DataFrame(index=snap.tickers)
    sub = pd.concat([p0.loc[win.index], p0.sample(min(len(p0), 400), random_state=p.sample_seed)]).loc[lambda d: ~d.index.duplicated()]
    score = pd.Series(snap.score, index=snap.tickers)
    elig = pd.Series(snap.eligible & (snap.filters == 0), index=snap.tickers)
    df = base.why_missed(sub, ret.reindex(sub.index), snap.tickers[snap.picked], eligible=elig.reindex(sub.index),
                         score=score.reindex(sub.index), thr=p.win_thr, cut_rank=p.k_pick)
    return {str(k): int(v) for k, v in df["why"].value_counts().items()}


def observe_day(snap: DecisionSnapshot, out: DayOutcome, now, p: ObserverParams | None = None) -> DayRecord:
    """The whole day, all categories, exceptions persisted. Fails closed unless the outcome matured strictly before `now`.
    Alignment, validation and the empty universe are handled here so callers can stream days without their own guards."""
    p = (p or ObserverParams()).require_valid()
    errs = snap.validate() + out.validate()
    if errs:
        raise ObserverError("; ".join(errs))
    require_past(out.resolved_at, now, "day outcome resolved_at")
    if as_date(out.resolved_at) <= as_date(snap.decided_at):
        raise FirewallBreach(f"outcome resolved {out.resolved_at} is not after the decision {snap.decided_at}")
    if snap.n < p.min_universe:
        return empty_record(snap, out, p)
    out, _ = align_outcome(snap, out)
    dc = classify_day(snap, out, p)
    tops = market_tops(dc, snap, out, p)
    idx, kept, dropped = select_rows(dc, snap, tops, p)
    rows = _rows(snap, out, dc, idx, p)
    elig = snap.eligible
    counts = {c.value: int(dc.masks[c].sum()) for c in CATEGORIES}
    counts_e = {c.value: int((dc.masks[c] & elig).sum()) for c in CATEGORIES}
    market, model = _summary_stats(dc, snap, out, p)
    model["winner_why"] = winner_reasons(snap, dc, p)
    rec = DayRecord(day=snap.decided_at, decided_at=snap.decided_at, resolved_at=out.resolved_at, params_hash=p.hash(),
                    snapshot_digest=snap.digest(), n_universe=snap.n, n_eligible=int(elig.sum()), n_scored=int(np.isfinite(snap.score).sum()),
                    n_no_outcome=int((~dc.valid).sum()), counts=counts, counts_eligible=counts_e,
                    bands=band_counts(dc.band), bands_eligible=band_counts(dc.band, elig), bands_o2c=band_counts(dc.band_o2c),
                    kept=kept, dropped=dropped, rows=rows, tops={k: tuple(str(t) for t in snap.tickers[v]) for k, v in tops.items()},
                    market=market, model=model, filter_names=snap.filter_names, used_precursors=dc.used_precursors,
                    name_ids=name_ids(snap.tickers))
    bad = rec.validate()
    if bad:
        raise ObserverError("inconsistent day record: " + "; ".join(bad))
    return rec


def name_ids(tickers: np.ndarray) -> np.ndarray:
    """Stable uint64 id per ticker (pandas' siphash), for set arithmetic on universes without keeping the strings."""
    return np.sort(pd.util.hash_array(np.asarray(tickers, dtype=object)).astype(np.uint64)) if len(tickers) else np.zeros(0, np.uint64)


def universe_step(rec: DayRecord, prev: np.ndarray | None) -> dict[str, Any]:
    """Churn against the previous recorded day: names that entered, names that left, and the size change. A universe that only
    ever loses names is a survivor-only panel; a sudden collapse is a data outage. Neither is a market fact."""
    ids = rec.name_ids if rec.name_ids is not None else np.zeros(0, np.uint64)
    if prev is None:
        entered = left = 0
    else:
        entered, left = int(len(np.setdiff1d(ids, prev, assume_unique=True))), int(len(np.setdiff1d(prev, ids, assume_unique=True)))
    size_prev = None if prev is None else len(prev)
    change = float("nan") if not size_prev else (len(ids) - size_prev) / size_prev
    return {"day": rec.day, "n": int(len(ids)), "entered": entered, "left": left, "size_change": change,
            "delisted": int(rec.market.get("delisted", 0)), "outage": bool(change == change and change < -0.2)}


def empty_record(snap: DecisionSnapshot, out: DayOutcome, p: ObserverParams) -> DayRecord:
    """A day with no eligible universe (holiday gap, data outage): a real record of zeros, so the gap is visible downstream
    instead of being skipped. It is marked `empty` and carries no rows."""
    zero = {c.value: 0 for c in CATEGORIES}
    return DayRecord(snap.decided_at, snap.decided_at, out.resolved_at, p.hash(), snap.digest(), snap.n, 0, 0, snap.n, dict(zero),
                     dict(zero), band_counts(np.zeros(0, dtype=np.int8)), band_counts(np.zeros(0, dtype=np.int8)),
                     band_counts(np.zeros(0, dtype=np.int8)), dict(zero), dict(zero), pd.DataFrame({"ticker": [], "flags": []}), {},
                     {"bands": band_counts(np.zeros(0, dtype=np.int8)), "n_band_movers": 0, "tone": "unknown"}, {}, snap.filter_names, (), True)


# ==================================================================================================================
# the ledger (many days) and its checkpoints
# ==================================================================================================================
class ObserverLedger:
    """Append-only run of DayRecords in strictly increasing day order, all made with the same parameters. Filed under a real
    year on the trusted side: release() refuses while that year is being replayed in disguise (rule 27)."""

    def __init__(self, params: ObserverParams | None = None, filed_year: str | None = None):
        self.params = (params or ObserverParams()).require_valid()
        self.filed_year = filed_year
        self._recs: list[DayRecord] = []
        self._prev_ids: np.ndarray | None = None
        self.universe_log: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._recs)

    def __iter__(self) -> Iterator[DayRecord]:
        return iter(self._recs)

    def add(self, rec: DayRecord) -> None:
        if rec.params_hash != self.params.hash():
            raise ObserverError("record was made with different observer parameters")
        if self._recs and as_date(rec.day) <= as_date(self._recs[-1].day):
            raise ObserverError(f"day {rec.day} is not after the last recorded day {self._recs[-1].day}")
        bad = rec.validate()
        if bad:
            raise ObserverError("; ".join(bad))
        if rec.name_ids is not None:
            self.universe_log.append(universe_step(rec, self._prev_ids))
            self._prev_ids = rec.name_ids
            rec = dataclasses.replace(rec, name_ids=None)
        self._recs.append(rec)

    def known(self, now) -> list[DayRecord]:
        """Records whose outcome matured strictly before `now`."""
        cut = as_date(now)
        return [r for r in self._recs if as_date(r.resolved_at) < cut]

    def counts_frame(self, now=None) -> pd.DataFrame:
        recs = self._recs if now is None else self.known(now)
        rows = []
        for r in recs:
            d = {"day": r.day, "n_universe": r.n_universe, "n_eligible": r.n_eligible, "empty": r.empty}
            d.update(r.counts)
            d.update({"band_" + k: v for k, v in r.bands.items()})
            rows.append(d)
        return pd.DataFrame(rows).set_index("day") if rows else pd.DataFrame()

    def rows_frame(self, now=None) -> pd.DataFrame:
        recs = self._recs if now is None else self.known(now)
        parts = [r.rows.assign(day=r.day) for r in recs if not r.rows.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def category_rates(self, now=None) -> pd.Series:
        """Mean share of the universe in each category, over non-empty days."""
        cf = self.counts_frame(now)
        if cf.empty:
            return pd.Series(dtype=float)
        cf = cf[~cf["empty"].astype(bool)]
        if cf.empty:
            return pd.Series(dtype=float)
        return pd.Series({c.value: float((cf[c.value] / cf["n_universe"]).mean()) for c in CATEGORIES})

    def band_summary(self, now=None) -> pd.DataFrame:
        """Per band: total movers, mean per day, share of days with at least one (C67: hundreds a day are expected)."""
        cf = self.counts_frame(now)
        if cf.empty:
            return pd.DataFrame(columns=["total", "per_day", "days_with"])
        rows = {}
        for name in BAND_NAMES.values():
            s = cf["band_" + name]
            rows[name] = {"total": int(s.sum()), "per_day": float(s.mean()), "days_with": float((s > 0).mean())}
        return pd.DataFrame(rows).T

    def mover_types(self, now=None) -> pd.DataFrame:
        """Band movers cross-tabulated by mover_type (event / momentum / fast_mover / illiquid / other) and band."""
        rf = self.rows_frame(now)
        if rf.empty:
            return pd.DataFrame()
        rf = rf[rf["band"] != 0]
        return pd.crosstab(rf["mover_type"], rf["band"]) if len(rf) else pd.DataFrame()

    def predictability(self, now=None) -> dict[str, float]:
        """Across days: how many movers the precursor proxy called predictable / unpredictable / neither, and the mean AUCs."""
        cf = self.counts_frame(now)
        if cf.empty:
            return {}
        pr, un = float(cf[MC.PREDICTABLE_MOVER.value].sum()), float(cf[MC.UNPREDICTABLE_MOVER.value].sum())
        recs = self._recs if now is None else self.known(now)
        moved = float(sum(r.market.get("n_moved", 0) for r in recs))
        pa = [r.model.get("precursor_auc") for r in recs if r.model.get("precursor_auc") == r.model.get("precursor_auc")]
        sa = [r.model.get("score_auc") for r in recs if r.model.get("score_auc") == r.model.get("score_auc")]
        return {"predictable": pr, "unpredictable": un, "unplaced_share": 1.0 - (pr + un) / moved if moved else float("nan"),
                "mean_precursor_auc": float(np.mean(pa)) if pa else float("nan"),
                "mean_score_auc": float(np.mean(sa)) if sa else float("nan")}

    def coverage(self) -> dict[str, float]:
        """Minimum row coverage of each category across days: below 1.0 means rows are a sample and counts are the truth."""
        out = {}
        for c in CATEGORIES:
            cov = [r.coverage(c) for r in self._recs if not r.empty]
            out[c.value] = float(min(cov)) if cov else float("nan")
        return out

    def release(self, now, replaying_years: Iterable[str] = ()) -> list[DayRecord]:
        """Records that may leave the research world for `now`. Refuses outright while the filing year is being replayed."""
        if self.filed_year is not None and str(self.filed_year) in {str(y) for y in replaying_years}:
            raise FirewallBreach(f"research filed under {self.filed_year} cannot be released while that year is replayed in disguise")
        return self.known(now)

    def save(self, directory, tag: str) -> dict[str, str]:
        """Checkpoint: rows to parquet, everything else to json. Writes are atomic (temp file then rename)."""
        os.makedirs(directory, exist_ok=True)
        pq, js = os.path.join(directory, f"observer_{tag}.parquet"), os.path.join(directory, f"observer_{tag}.json")
        rf = self.rows_frame()
        tmp = pq + ".tmp"
        rf.to_parquet(tmp, index=False)
        os.replace(tmp, pq)
        meta = {"params": dataclasses.asdict(self.params), "params_hash": self.params.hash(), "filed_year": self.filed_year,
                "days": [{**{k: getattr(r, k) for k in ("day", "decided_at", "resolved_at", "params_hash", "snapshot_digest", "n_universe",
                                                        "n_eligible", "n_scored", "n_no_outcome", "empty")},
                          "counts": dict(r.counts), "counts_eligible": dict(r.counts_eligible), "bands": dict(r.bands),
                          "bands_eligible": dict(r.bands_eligible), "bands_o2c": dict(r.bands_o2c), "kept": dict(r.kept),
                          "dropped": dict(r.dropped), "tops": {k: list(v) for k, v in r.tops.items()},
                          "market": _jsonable(r.market), "model": _jsonable(r.model), "filter_names": list(r.filter_names),
                          "used_precursors": list(r.used_precursors)} for r in self._recs]}
        tmp = js + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp, js)
        return {"rows": pq, "meta": js}

    @classmethod
    def load(cls, directory, tag: str) -> "ObserverLedger":
        with open(os.path.join(directory, f"observer_{tag}.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        p = ObserverParams(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                              for k, v in meta["params"].items()})
        led = cls(p, meta.get("filed_year"))
        if led.params.hash() != meta["params_hash"]:
            raise ObserverError("checkpoint parameters do not reproduce their own hash")
        rf = pd.read_parquet(os.path.join(directory, f"observer_{tag}.parquet"))
        for d in meta["days"]:
            rows = rf[rf["day"] == d["day"]].drop(columns=["day"]).reset_index(drop=True) if len(rf) else pd.DataFrame({"ticker": [], "flags": []})
            led._recs.append(DayRecord(rows=rows, tops={k: tuple(v) for k, v in d["tops"].items()}, filter_names=tuple(d["filter_names"]),
                                       used_precursors=tuple(d["used_precursors"]),
                                       **{k: d[k] for k in ("day", "decided_at", "resolved_at", "params_hash", "snapshot_digest", "n_universe",
                                                            "n_eligible", "n_scored", "n_no_outcome", "empty", "counts", "counts_eligible",
                                                            "bands", "bands_eligible", "bands_o2c", "kept", "dropped", "market", "model")}))
        return led


def _jsonable(d: Mapping[str, Any]) -> dict[str, Any]:
    def f(v):
        if isinstance(v, Mapping):
            return {str(k): f(x) for k, x in v.items()}
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(float(v)) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    return {k: f(v) for k, v in d.items()}


# ==================================================================================================================
# streaming and the public entry
# ==================================================================================================================
@dataclass
class ObserverState:
    """Everything step() needs between days; holds only compact DayRecords, never the raw cross-sections."""
    params: ObserverParams = field(default_factory=ObserverParams)
    ledger: ObserverLedger | None = None

    def __post_init__(self):
        if self.ledger is None:
            self.ledger = ObserverLedger(self.params)


def step(state: ObserverState, snap: DecisionSnapshot, out: DayOutcome, now) -> DayRecord:
    """PUBLIC ENTRY. Observe one matured day and append it to the state's ledger. `now` is the trusted-side clock: the
    outcome must have matured strictly before it."""
    rec = observe_day(snap, out, now, state.params)
    state.ledger.add(rec)
    return rec


def stream(days: Iterable[tuple[DecisionSnapshot, DayOutcome]], state: ObserverState, now_of=None) -> Iterator[DayRecord]:
    """Consume days one at a time (a generator of (snapshot, outcome)); yield each DayRecord as soon as it exists so a consumer
    (the mover-episode research, the autopsy) can work per day. `now_of(outcome)` gives the clock for a day; by default the
    calendar day after the outcome resolves - the earliest moment a matured outcome can be studied."""
    import datetime as _dt
    for snap, out in days:
        now = now_of(out) if now_of else as_date(out.resolved_at) + _dt.timedelta(days=1)
        yield step(state, snap, out, now)


def run_year(year: str, days: Iterable[tuple[DecisionSnapshot, DayOutcome]], directory, params: ObserverParams | None = None,
             now_of=None, resume: bool = True) -> ObserverLedger:
    """Stream one year, then checkpoint it. With resume=True an existing checkpoint for the year is loaded and days at/before
    its last day are skipped, so a killed run costs at most one year of one worker, never the run (rule 11)."""
    tag = str(year)
    path = os.path.join(directory, f"observer_{tag}.json")
    led = ObserverLedger.load(directory, tag) if resume and os.path.exists(path) else ObserverLedger(params, tag)
    if resume and len(led) and params is not None and led.params.hash() != params.hash():
        raise ObserverError("checkpoint was made with different parameters; refusing to mix them")
    state = ObserverState(led.params, led)
    last = as_date(led._recs[-1].day) if len(led) else None
    todo = ((s, o) for s, o in days if last is None or as_date(s.decided_at) > last)
    for _ in stream(todo, state, now_of):
        pass
    led.save(directory, tag)
    return led


# ==================================================================================================================
# bridges to the existing learning vocabulary
# ==================================================================================================================
def filter_labels(bits: int, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(n for i, n in enumerate(names) if int(bits) & (1 << i)) or (("filtered",) if int(bits) else ())


def to_week(rec: DayRecord, p: ObserverParams) -> lmw.Week:
    """The day's exception rows as an engine.learning.missed_winners.Week, so RejectionAnalyzer can say why each missed
    winner was missed. Only exception rows exist, so base rates computed from this Week are NOT the universe's; use the
    record's counts for those. Features are the identity-free cross-sectional ranks."""
    rf = rec.rows
    cands = []
    for r in rf.itertuples(index=False):
        feats = {k[3:]: float(getattr(r, k)) for k in rf.columns if k.startswith("rk_") and getattr(r, k) == getattr(r, k)}
        conf = getattr(r, "confidence")
        cands.append(lmw.Candidate(
            cid=r.cid, features=feats, fwd=None if not np.isfinite(r.ret) else float(r.ret), picked=bool(r.picked),
            score=None if not np.isfinite(r.score) else float(r.score), rank=int(r.rank) if r.rank > 0 else None,
            eligible=bool(r.eligible), filters_hit=filter_labels(r.filters, rec.filter_names) if r.filters else (),
            confidence=None if not np.isfinite(conf) else float(conf), kind=str(r.mover_type)))
    return lmw.Week(label=stable_hash([rec.day, rec.snapshot_digest], 8), decided_at=rec.decided_at, resolved_at=rec.resolved_at,
                    candidates=tuple(cands), k=p.k_pick, thr=p.win_thr)


def missed_reasons(rec: DayRecord, p: ObserverParams, analyzer: lmw.RejectionAnalyzer | None = None) -> dict[str, int]:
    """Counts of the primary reason for each missed winner among the day's rows (UNKNOWN stays UNKNOWN)."""
    if rec.rows.empty:
        return {}
    wk = to_week(rec, p)
    if not wk.missed():
        return {}
    out: dict[str, int] = {}
    for r in (analyzer or lmw.RejectionAnalyzer()).explain_missed(wk):
        out[r.primary.value] = out.get(r.primary.value, 0) + 1
    return out


def as_records(rec: DayRecord, bands: bool = True) -> list[dict[str, Any]]:
    """The simple documented handoff for episode research (R21): one dict per persisted row, ROW_SCHEMA columns, with the day
    and mover-band attached. Only rows with a C67 band when bands=True."""
    rf = rec.rows if not bands else rec.rows[rec.rows["band"] != 0] if not rec.rows.empty else rec.rows
    return [dict(r, day=rec.day, decided_at=rec.decided_at, resolved_at=rec.resolved_at) for r in rf.to_dict("records")]


# ==================================================================================================================
# invariants and cross-day analytics
# ==================================================================================================================
def check_invariants(dc: DayClass, snap: DecisionSnapshot) -> list[str]:
    """Logical rules the categories must obey on any day. A failure is a bug in the classifier, never a market fact."""
    m, e = dc.masks, []
    pairs = ((MC.WINNER, MC.LOSER), (MC.EXTREME_UP, MC.EXTREME_DOWN), (MC.PREDICTABLE_MOVER, MC.UNPREDICTABLE_MOVER),
             (MC.FALSE_POSITIVE, MC.FALSE_NEGATIVE), (MC.CONSIDERED_HIGH, MC.CONSIDERED_MEDIUM))
    for a, b in pairs:
        if (m[a] & m[b]).any():
            e.append(f"{a.value} and {b.value} overlap")
    if (m[MC.PREDICTABLE_MOVER] | m[MC.UNPREDICTABLE_MOVER])[~dc.moved].any():
        e.append("a mover class was given to a name that did not move")
    if (m[MC.FALSE_NEGATIVE] & snap.picked).any():
        e.append("a picked name is a false negative")
    if (m[MC.FALSE_POSITIVE] & ~(snap.picked | m[MC.CONSIDERED_HIGH])).any():
        e.append("a false positive that was never predicted to move")
    if (m[MC.NEAR_MISS] & snap.picked).any():
        e.append("a picked name is a near miss")
    if (m[MC.EXTREME_UP] & ~(dc.sig_ret > 0)).any() or (m[MC.EXTREME_DOWN] & ~(dc.sig_ret < 0)).any():
        e.append("an extreme mover on the wrong side")
    unfilled = ~dc.valid
    for c in (MC.FALSE_POSITIVE, MC.FALSE_NEGATIVE, MC.PREDICTABLE_MOVER, MC.UNPREDICTABLE_MOVER):
        if (m[c] & unfilled).any():
            e.append(f"{c.value} assigned to a name with no outcome")
    return e


def repeat_movers(ledger: ObserverLedger, window: int = 5, now=None) -> dict[str, float]:
    """Do band movers cluster in time? For every band mover-day, whether the same name was already a band mover in the previous
    `window` recorded days. Compared with the rate expected if movers were scattered independently. Research side (tickers)."""
    recs = ledger._recs if now is None else ledger.known(now)
    hist: list[set] = []
    n_tot = n_rep = 0
    exp_num = exp_den = 0.0
    for r in recs:
        today = set(r.rows.loc[r.rows["band"] != 0, "ticker"]) if not r.rows.empty else set()
        prev = set().union(*hist[-window:]) if hist else set()
        n_tot += len(today)
        n_rep += len(today & prev)
        if r.n_universe:
            exp_num += len(today) * len(prev) / r.n_universe
            exp_den += len(today)
        hist.append(today)
    obs = n_rep / n_tot if n_tot else float("nan")
    exp = exp_num / exp_den if exp_den else float("nan")
    return {"observed_repeat_share": obs, "expected_if_independent": exp,
            "clustering_ratio": obs / exp if exp and exp == exp and obs == obs else float("nan"), "band_mover_days": float(n_tot)}


def by_era(ledger: ObserverLedger, era_of, now=None) -> pd.DataFrame:
    """Category rates per era label (era_of maps a day string to a label such as a year or a regime). Research side."""
    recs = ledger._recs if now is None else ledger.known(now)
    acc: dict[str, list] = {}
    for r in recs:
        if r.empty or not r.n_universe:
            continue
        acc.setdefault(str(era_of(r.day)), []).append({c.value: r.counts[c.value] / r.n_universe for c in CATEGORIES}
                                                      | {"band_movers": sum(r.bands.values()) / r.n_universe})
    return pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in acc.items()}).T if acc else pd.DataFrame()


def rate_stability(ledger: ObserverLedger, cat: MoveCategory, now=None) -> dict[str, float]:
    """First-half vs second-half rate of a category and a two-proportion z: is the market/model producing this category at a
    stable rate, or has something shifted (a regime, a model change, a data change)?"""
    cf = ledger.counts_frame(now)
    cf = cf[~cf["empty"].astype(bool)] if not cf.empty else cf
    if len(cf) < 4:
        return {"n_days": float(len(cf)), "z": float("nan")}
    h = len(cf) // 2
    a, b = cf.iloc[:h], cf.iloc[h:]
    ka, na = float(a[cat.value].sum()), float(a["n_universe"].sum())
    kb, nb = float(b[cat.value].sum()), float(b["n_universe"].sum())
    pa, pb, pool = ka / na, kb / nb, (ka + kb) / (na + nb)
    se = math.sqrt(pool * (1 - pool) * (1 / na + 1 / nb)) if 0 < pool < 1 else float("nan")
    return {"n_days": float(len(cf)), "first_half": pa, "second_half": pb, "z": (pb - pa) / se if se == se and se > 0 else float("nan")}


def render_ledger(ledger: ObserverLedger, now=None, top: int = 14) -> str:
    """Plain-text run report: universe size, category rates, bands, predictability, coverage. Deterministic."""
    recs = ledger._recs if now is None else ledger.known(now)
    if not recs:
        return "observer ledger: no days recorded"
    lines = [f"observer ledger: {len(recs)} days, {recs[0].day} .. {recs[-1].day}, params {ledger.params.hash()}",
             f"universe: median {int(np.median([r.n_universe for r in recs]))} names/day, "
             f"{sum(r.empty for r in recs)} empty days, {sum(r.n_no_outcome for r in recs)} name-days without an outcome"]
    rates = ledger.category_rates(now)
    for c, v in rates.head(top).items():
        lines.append(f"  {c:<22s} {100 * v:7.3f}% of the universe per day")
    lines.append("bands (movers per day):")
    for name, row in ledger.band_summary(now).iterrows():
        lines.append(f"  {name:<12s} total {int(row['total']):>7d}  per day {row['per_day']:8.1f}  days with >=1 {100 * row['days_with']:5.1f}%")
    pred = ledger.predictability(now)
    if pred:
        lines.append(f"predictability proxy: predictable {pred['predictable']:.0f}, unpredictable {pred['unpredictable']:.0f}, "
                     f"unplaced share of movers {pred['unplaced_share']:.2f}, precursor AUC {pred['mean_precursor_auc']:.3f}, "
                     f"score AUC {pred['mean_score_auc']:.3f}")
    cov = ledger.coverage()
    thin = ", ".join(f"{k} {v:.2f}" for k, v in cov.items() if v == v and v < 1.0)
    lines.append("row coverage (min over days): " + (thin or "all categories fully kept"))
    return "\n".join(lines)


# ==================================================================================================================
# a planted world (tests and the cost benchmark) with known ground truth
# ==================================================================================================================
@dataclass(frozen=True)
class Plant:
    """Counts of names to plant with a known category. The rest of the universe is quiet baseline noise that cannot, by
    construction, reach a band, a winner or a mover threshold."""
    n_hit: int = 4              # picked, move 9% (true positives)
    n_fp: int = 3               # picked, stay flat (false positives)
    n_pred: int = 6             # unpicked, loud precursors, 8% intraday move (predictable false negatives)
    n_unpred: int = 5           # unpicked, quiet precursors, +12% overnight gap then flat (unpredictable; not capturable)
    n_near: int = 3             # unpicked, score just under the pick cutoff, flat
    n_up_mid: int = 0           # exact close-to-close +6% (band 5-10), no other trait
    n_up_big: int = 0           # +14%
    n_down_mid: int = 0         # -7%
    n_down_big: int = 0         # -13%
    n_abstain: int = 2
    n_lowconf: int = 2

    def total(self) -> int:
        return sum(dataclasses.astuple(self))


def synthetic_day(n: int = 600, seed: int = 0, day: int = 0, p: ObserverParams | None = None, plant: Plant | None = Plant(),
                  start: str = "2020-01-06") -> tuple[DecisionSnapshot, DayOutcome, dict[str, np.ndarray]]:
    """One day of `n` names with planted groups; returns (snapshot, outcome, truth) where truth maps a group name to the index
    array of its members. Deterministic in (seed, day). Baseline names never move more than a few percent."""
    import datetime as _dt
    p = p or ObserverParams()
    plant = plant or Plant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if plant.total() + 5 > n:
        raise ValueError("universe too small for the planted groups")
    rng = np.random.default_rng([seed, day])
    d0 = as_date(start) + _dt.timedelta(days=int(day))
    tk = np.array([f"S{i:05d}" for i in range(n)], dtype=object)
    pc = np.exp(rng.uniform(math.log(8), math.log(150), n))
    vol20 = np.exp(rng.normal(math.log(0.02), 0.3, n))
    feats = {"vol20": vol20, "atr": vol20 * 1.1, "vol_surge": np.exp(rng.normal(0, 0.2, n)), "range20": vol20 * 4.0 * np.exp(rng.normal(0, 0.1, n)),
             "r5": rng.normal(0, 0.03, n), "r20": rng.normal(0, 0.06, n), "log_dv": rng.normal(16, 1, n), "dist_hi": -np.abs(rng.normal(0, 0.1, n))}
    score = rng.normal(0, 0.5, n)
    order = rng.permutation(n)
    cur = [0]

    def take(k):
        g = order[cur[0]:cur[0] + k]
        cur[0] += k
        return g
    truth = {name: take(getattr(plant, "n_" + name)) for name in ("hit", "fp", "pred", "unpred", "near", "up_mid", "up_big", "down_mid",
                                                                   "down_big", "abstain", "lowconf")}
    picks = np.concatenate([truth["hit"], truth["fp"]])
    picked = np.zeros(n, dtype=bool)
    picked[picks] = True
    score[picks] = 5.0 + np.arange(len(picks))[::-1] * 0.01
    cut = 5.0
    score[truth["near"]] = cut - 0.1
    score[truth["pred"]] = -3.0
    score[truth["unpred"]] = -3.0
    confidence = np.clip(rng.normal(0.7, 0.1, n), 0.55, 1.0)
    confidence[truth["lowconf"]] = 0.2
    score[truth["lowconf"]] = 4.0 + rng.uniform(0, 0.5, len(truth["lowconf"]))
    abstained = np.zeros(n, dtype=bool)
    abstained[truth["abstain"]] = True
    for k in ("pred",):
        g = truth[k]
        for name in ("vol20", "atr", "vol_surge", "range20"):
            feats[name][g] = feats[name].max() * (1.5 + rng.uniform(0, 0.5, len(g)))
    g = truth["unpred"]
    for name in ("vol20", "atr", "vol_surge", "range20"):
        feats[name][g] = feats[name].min() * 0.5
    entry, hi, lo, close = pc.copy(), pc.copy(), pc.copy(), pc.copy()
    gap = np.clip(rng.normal(0, 0.004, n), -0.01, 0.01)
    entry = pc * (1 + gap)
    body = np.clip(rng.normal(0, 0.01, n), -0.025, 0.025)
    close = entry * (1 + body)
    hi = np.maximum(entry, close) * (1 + rng.uniform(0, 0.008, n))
    lo = np.minimum(entry, close) * (1 - rng.uniform(0, 0.008, n))

    def set_path(idx, gap_v, body_v, up_extra, dn_extra):
        entry[idx] = pc[idx] * (1 + gap_v)
        close[idx] = entry[idx] * (1 + body_v)
        hi[idx] = np.maximum(entry[idx], close[idx]) * (1 + up_extra)
        lo[idx] = np.minimum(entry[idx], close[idx]) * (1 - dn_extra)
    set_path(truth["hit"], 0.0, 0.09, 0.011, 0.005)
    set_path(truth["fp"], 0.0, 0.002, 0.008, 0.008)
    set_path(truth["pred"], 0.0, 0.08, 0.005, 0.004)
    set_path(truth["unpred"], 0.12, 0.003, 0.004, 0.004)
    set_path(truth["near"], 0.0, 0.002, 0.006, 0.006)
    set_path(truth["up_mid"], 0.0, 0.06, 0.004, 0.004)
    set_path(truth["up_big"], 0.0, 0.14, 0.004, 0.004)
    set_path(truth["down_mid"], 0.0, -0.07, 0.004, 0.004)
    set_path(truth["down_big"], 0.0, -0.13, 0.004, 0.004)
    volume_ratio = np.exp(rng.normal(0, 0.3, n))
    snap = DecisionSnapshot.make(d0, tk, pc, score=score, confidence=confidence, picked=picked, abstained=abstained,
                                 features={k: v.astype(np.float32) for k, v in feats.items()}, event_known=np.zeros(n, dtype=bool))
    out = DayOutcome.make(d0 + _dt.timedelta(days=1), tk, entry, hi, lo, close, volume_ratio)
    return snap, out, truth


def synthetic_days(n_days: int, n: int = 600, seed: int = 0, plant: Plant | None = Plant(), start: str = "2020-01-06"
                   ) -> Iterator[tuple[DecisionSnapshot, DayOutcome]]:
    """A lazy stream of planted days (one per calendar day), for the streaming interface."""
    for d in range(n_days):
        s, o, _ = synthetic_day(n, seed, d, plant=plant, start=start)
        yield s, o


def measure_cost(n_names: int = 3000, n_days: int = 5, seed: int = 0, p: ObserverParams | None = None) -> dict[str, float]:
    """Time and peak Python-heap memory of observe_day on a synthetic universe of `n_names` (rule 10: measure, do not guess).
    Building the inputs is excluded from the timing; the record's persisted size is reported. Returns per-day figures."""
    p = p or ObserverParams()
    plant = Plant(n_up_mid=40, n_up_big=15, n_down_mid=40, n_down_big=15)
    days = [synthetic_day(n_names, seed, d, p, plant) for d in range(n_days)]
    later = lambda o: as_date(o.resolved_at) + __import__("datetime").timedelta(days=1)
    t0 = time.perf_counter()
    for s, o, _ in days:
        observe_day(s, o, later(o), p)
    dt_ = time.perf_counter() - t0
    tracemalloc.start()
    recs = [observe_day(s, o, later(o), p) for s, o, _ in days]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows = sum(len(r.rows) for r in recs)
    size = sum(int(r.rows.memory_usage(deep=True).sum()) for r in recs)
    return {"n_names": float(n_names), "ms_per_day": 1000.0 * dt_ / n_days, "peak_mb": peak / 1e6, "rows_per_day": rows / n_days,
            "row_kb_per_day": size / 1e3 / n_days, "input_mb_per_day": (days[0][0].n * 8 * 12) / 1e6}


# ==================================================================================================================
# model scorecard, sector breadth, counterfactual sampling, determinism
# ==================================================================================================================
def score_calibration(ledger: ObserverLedger, kind: str = "score", now=None) -> pd.DataFrame:
    """Mover rate by score (or precursor) decile pooled over days, with lift over the overall mover rate and a Spearman check
    that the deciles are ordered. `kind` is "score" or "precursor". A model whose top decile is not richer in movers than its
    bottom decile has not learned to find movers, whatever its picks earned."""
    key = kind + "_deciles"
    recs = [r for r in (ledger._recs if now is None else ledger.known(now)) if r.model.get(key)]
    if not recs:
        return pd.DataFrame(columns=["names", "movers", "mover_rate", "lift"])
    tab = np.sum([np.array(r.model[key]) for r in recs], axis=0)
    df = pd.DataFrame(tab, columns=["names", "movers"])
    total = df["movers"].sum() / max(1, df["names"].sum())
    df["mover_rate"] = df["movers"] / df["names"].clip(lower=1)
    df["lift"] = df["mover_rate"] / total if total > 0 else np.nan
    df.attrs["monotone_rho"] = float(pd.Series(df["mover_rate"]).corr(pd.Series(range(len(df))), method="spearman")) if len(df) > 2 else float("nan")
    return df


def sector_breadth(rec: DayRecord, min_movers: int = 8) -> pd.DataFrame:
    """Are the day's band movers concentrated in a few sectors? Per sector: names, band movers, the mover rate and a binomial z
    of the excess over the day's overall rate. Exact: it reads the per-sector counts over the whole universe, not the rows."""
    tab = rec.market.get("sector_movers") or {}
    cols = ["names", "movers", "rate", "z"]
    if not tab:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(tab, index=["names", "movers"]).T.astype(float)
    tot_n, tot_m = df["names"].sum(), df["movers"].sum()
    if tot_m < min_movers or tot_n <= 0:
        return pd.DataFrame(columns=cols)
    p0 = min(max(tot_m / tot_n, 1e-9), 1 - 1e-9)
    df["rate"] = df["movers"] / df["names"].clip(lower=1)
    df["z"] = (df["movers"] - df["names"] * p0) / np.sqrt(df["names"] * p0 * (1 - p0))
    return df.sort_values("z", ascending=False)[cols]


def counterfactual_sample(rec: DayRecord, k: int = 25, seed: int = 0, categories: Sequence[MoveCategory] = (
        MC.FALSE_NEGATIVE, MC.FALSE_POSITIVE, MC.NEAR_MISS, MC.UNPREDICTABLE_MOVER)) -> pd.DataFrame:
    """Which events get the expensive could-I-have-known reconstruction (rule 27: cap at the top K per day plus a seeded sample
    of the rest, one shared point-in-time snapshot). The top half of the budget is the largest moves; the rest is drawn with
    probability uniform over the remaining events, seeded by (seed, day), so a rerun picks the same ones and the sample is
    unbiased for the tail. `weight` = events represented per row, for re-weighting."""
    rf = rec.rows
    if rf.empty:
        return rf.assign(weight=pd.Series(dtype=float))
    mask = np.zeros(len(rf), dtype=bool)
    for c in categories:
        mask |= has(rf["flags"].to_numpy(), c)
    cand = rf[mask].copy()
    cand["_mag"] = cand["exc_pc"].abs().fillna(0.0)
    if len(cand) <= k:
        return cand.drop(columns="_mag").assign(weight=1.0)
    cand = cand.sort_values(["_mag", "cid"], ascending=[False, True])
    n_top = max(1, k // 2)
    top, rest = cand.iloc[:n_top], cand.iloc[n_top:]
    rng = np.random.default_rng(int(stable_hash([seed, rec.day, "cf"], 8), 16))
    pick = np.sort(rng.choice(len(rest), size=k - n_top, replace=False))
    samp = rest.iloc[pick]
    return pd.concat([top.assign(weight=1.0), samp.assign(weight=len(rest) / max(1, len(samp)))]).drop(columns="_mag")


def record_digest(rec: DayRecord) -> str:
    """Content hash of a record (counts, bands, market, model, ordered rows): the same snapshot and parameters must give the
    same digest on every run, which is what makes a rerun auditable."""
    rows = rec.rows.sort_values("cid").drop(columns=["ticker"]) if not rec.rows.empty else rec.rows
    body = rows.round(9).to_csv(index=False) if not rows.empty else ""
    return stable_hash([rec.day, rec.snapshot_digest, rec.params_hash, dict(rec.counts), dict(rec.bands), _jsonable(rec.market),
                        _jsonable(rec.model), hashlib.sha256(body.encode()).hexdigest()], 16)


# ==================================================================================================================
# from daily OHLCV bars to observer days (streams, per year, survivor-inclusive)
# ==================================================================================================================
def default_scorer(feats: Mapping[str, np.ndarray]) -> np.ndarray:
    """A neutral point-in-time mover score used when no model is attached: the mean cross-sectional rank of volatility,
    ATR and volume surge. It is only a baseline so the observer can run on bars alone; the real score arrives from the model."""
    parts = [pct_rank(np.asarray(feats[k], dtype=np.float64)) for k in ("vol20", "atr", "vol_surge") if k in feats]
    if not parts:
        return np.full(len(next(iter(feats.values()))), np.nan) if feats else np.zeros(0)
    return np.nanmean(np.vstack(parts), axis=0)


def bars_stream(bars: Mapping[str, pd.DataFrame], p: ObserverParams | None = None, scorer=None, start=None, end=None,
                min_price: float = 3.0, min_dollar_vol: float = 2e6, warmup: int = 80) -> Iterator[tuple[DecisionSnapshot, DayOutcome]]:
    """Yield (snapshot, outcome) for each session in [start, end] from wide OHLCV frames (date x ticker), one day at a time.
    Features come from engine.fv_pipeline.price_features (every value at row t uses bars up to t). The decision is at t's close;
    the fill is t+1's open; the outcome covers sessions t+1 .. t+horizon. The universe is every name with a close at t, whether
    or not it is tradable (`eligible` says), and a name that stops trading inside the window is carried at its last price and
    flagged `delisted` - a delisting exits at the last price, never earlier. Frames must include survivors AND the dead."""
    from engine.fv_pipeline import price_features
    p = (p or ObserverParams()).require_valid()
    C = bars["Close"]
    sessions = pd.DatetimeIndex(C.index)
    lo_i = 0 if start is None else max(0, int(sessions.searchsorted(pd.Timestamp(start))) - warmup)
    hi_i = len(sessions) if end is None else min(len(sessions), int(sessions.searchsorted(pd.Timestamp(end), side="right")) + p.horizon + 1)
    sl = slice(lo_i, hi_i)
    sub = {k: v.iloc[sl] for k, v in bars.items() if k in ("Open", "High", "Low", "Close", "Volume")}
    f = price_features(sub, min_price, min_dollar_vol)
    ses = pd.DatetimeIndex(sub["Close"].index)
    O, H, L, Cl = (sub[k].to_numpy(dtype=np.float64) for k in ("Open", "High", "Low", "Close"))
    surge = f["vol_surge"].to_numpy() if "vol_surge" in f else np.full(O.shape, np.nan, dtype=np.float32)
    tradable = f["tradable"].to_numpy()
    names = np.asarray(sub["Close"].columns, dtype=object)
    first = 0 if start is None else int(ses.searchsorted(pd.Timestamp(start)))
    last_t = len(ses) - p.horizon - 2 if end is None else min(len(ses) - p.horizon - 2, int(ses.searchsorted(pd.Timestamp(end), side="right")) - 1)
    for t in range(max(first, 1), last_t + 1):
        present = np.isfinite(Cl[t])
        if not present.any():
            continue
        idx = np.flatnonzero(present)
        feats = {k: f[k].to_numpy()[t, idx] for k in ("vol20", "atr", "r5", "r20", "dist_hi", "log_dv", "vol_surge", "range20") if k in f}
        core = np.ones(len(idx), dtype=bool)
        for k in ("vol20", "atr", "r5"):
            core &= np.isfinite(feats[k])
        elig = tradable[t, idx] & core
        score = (scorer or default_scorer)(feats)
        score = np.where(elig, score, np.nan)
        pick = np.zeros(len(idx), dtype=bool)
        order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")[:p.k_pick]
        pick[order[np.isfinite(score[order])]] = True
        w = slice(t + 1, t + 1 + p.horizon)
        Ow, Hw, Lw, Cw = O[w][:, idx], H[w][:, idx], L[w][:, idx], Cl[w][:, idx]
        entry = Ow[0]
        with np.errstate(invalid="ignore"):
            hi, lo = np.fmax.reduce(Hw, axis=0), np.fmin.reduce(Lw, axis=0)
        alive = np.isfinite(Cw)
        last_close = np.array([Cw[np.flatnonzero(alive[:, j])[-1], j] if alive[:, j].any() else np.nan for j in range(len(idx))])
        d0 = ses[t]
        snap = DecisionSnapshot.make(d0, names[idx], Cl[t, idx], eligible=elig, score=score, picked=pick, features=feats,
                                     filter_names=("untradable",), filters=(~tradable[t, idx]).astype(np.int64))
        out = DayOutcome.make(ses[t + 1], names[idx], entry, hi, lo, last_close, surge[t + 1, idx], delisted=~alive[-1],
                              day_hi=Hw[0], day_lo=Lw[0], day_close=Cw[0])
        yield snap, out


def year_slices(sessions: pd.DatetimeIndex) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """First and last session of each calendar year present (for streaming a long history one year at a time)."""
    s = pd.Series(sessions, index=sessions)
    g = s.groupby(s.index.year)
    return {int(y): (v.iloc[0], v.iloc[-1]) for y, v in g}


# ==================================================================================================================
# band profiles (C67), symmetry, universe health, definition sensitivity, checkpoints
# ==================================================================================================================
def band_profile(rec: DayRecord, p: ObserverParams | None = None) -> pd.DataFrame:
    """What the day's band movers looked like, per band: how many, how they moved (close-to-close, open-to-close, high-low
    range), where they closed inside their range, how much of the move was the gap, volume, whether the model held them, and
    how many were data failures. This is the hand-off the mover-episode research starts from; the next-day path is theirs."""
    cols = ["n", "mean_c2c", "mean_o2c", "mean_range", "mean_close_loc", "gap_driven_share", "mean_volume_ratio", "picked_share",
            "event_share", "suspect_share", "precursor_mean"]
    rf = rec.rows
    if rf.empty or (rf["band"] == 0).all():
        return pd.DataFrame(columns=cols)
    thr = (p or ObserverParams()).gap_share_thr
    rows = {}
    for code, name in BAND_NAMES.items():
        g = rf[rf["band"] == code]
        if g.empty:
            continue
        rows[name] = {"n": len(g), "mean_c2c": g["c2c"].mean(), "mean_o2c": g["o2c"].mean(), "mean_range": g["hl_range"].mean(),
                      "mean_close_loc": g["close_loc"].mean(), "gap_driven_share": float((g["gap_share"] >= thr).mean()),
                      "mean_volume_ratio": g["volume_ratio"].mean(), "picked_share": float(g["picked"].mean()),
                      "event_share": float(g["event_known"].mean()), "suspect_share": float((g["suspect"] != 0).mean()),
                      "precursor_mean": g["precursor"].mean()}
    return pd.DataFrame(rows).T.reindex(columns=cols)


def band_profile_ledger(ledger: ObserverLedger, p: ObserverParams | None = None, now=None) -> pd.DataFrame:
    """band_profile pooled over days, weighted by the number of movers each day contributed."""
    recs = ledger._recs if now is None else ledger.known(now)
    parts = []
    for r in recs:
        bp = band_profile(r, p or ledger.params)
        if not bp.empty:
            parts.append(bp.assign(_w=bp["n"].astype(float)).reset_index().rename(columns={"index": "band"}))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    num = [c for c in df.columns if c not in ("band", "n", "_w")]
    out = df.groupby("band").apply(lambda g: pd.Series({**{c: float(np.average(g[c].astype(float), weights=g["_w"])) for c in num},
                                                        "n": float(g["n"].sum())}), include_groups=False)
    return out[["n"] + num]


def symmetry(rec: DayRecord) -> dict[str, float]:
    """Winners vs losers, extreme up vs down, band up vs down, false positives vs false negatives on one day. Losers are studied
    as hard as winners (section 5), so the day's record carries the ratio that says which side the market was on and whether
    the model's misses were on the same side. A ratio of 1.0 is symmetric; NaN when a side is empty."""
    c = rec.counts
    r = lambda a, b: float(a / b) if b else (float("nan") if not a else float("inf"))
    up, dn = rec.bands["up_5_10"] + rec.bands["up_gt10"], rec.bands["down_5_10"] + rec.bands["down_gt10"]
    return {"winner_loser_ratio": r(c[MC.WINNER.value], c[MC.LOSER.value]), "extreme_up_down_ratio": r(c[MC.EXTREME_UP.value], c[MC.EXTREME_DOWN.value]),
            "band_up_down_ratio": r(up, dn), "fp_fn_ratio": r(c[MC.FALSE_POSITIVE.value], c[MC.FALSE_NEGATIVE.value]),
            "big_share_up": r(rec.bands["up_gt10"], up), "big_share_down": r(rec.bands["down_gt10"], dn), "band_movers": float(up + dn)}


def symmetry_ledger(ledger: ObserverLedger, now=None) -> pd.DataFrame:
    recs = ledger._recs if now is None else ledger.known(now)
    return pd.DataFrame([{"day": r.day, **symmetry(r)} for r in recs if not r.empty]).set_index("day") if recs else pd.DataFrame()


def universe_health(ledger: ObserverLedger, min_churn: float = 0.0005) -> dict[str, Any]:
    """Judge the universe series: size, churn, outages, and whether the universe only ever shrinks (a survivor-only panel).
    `survivor_only_suspect` is True when names leave but essentially none ever enter over a long run - real universes have both."""
    log = pd.DataFrame(ledger.universe_log)
    if log.empty:
        return {"days": 0, "survivor_only_suspect": False, "outages": 0}
    ent, left = float(log["entered"].iloc[1:].sum()), float(log["left"].iloc[1:].sum())
    mean_n = float(log["n"].mean())
    turnover = (ent + left) / max(1.0, 2 * mean_n * max(1, len(log) - 1))
    return {"days": int(len(log)), "mean_size": mean_n, "min_size": int(log["n"].min()), "max_size": int(log["n"].max()),
            "entered": ent, "left": left, "daily_turnover": turnover, "outages": int(log["outage"].sum()),
            "delisted_seen": int(log["delisted"].sum()),
            "survivor_only_suspect": bool(len(log) >= 60 and turnover < min_churn and int(log["delisted"].sum()) == 0)}


def threshold_sensitivity(snap: DecisionSnapshot, out: DayOutcome, now, p: ObserverParams | None = None,
                          grid: Sequence[float] = (0.05, 0.07, 0.10)) -> pd.DataFrame:
    """How much the category counts depend on the thresholds. The mover / winner definitions are conventions, not facts: if
    false negatives triple when mover_thr goes from 0.07 to 0.05, any conclusion drawn from them is a conclusion about the
    convention. Returns one row per threshold with the affected categories."""
    p = (p or ObserverParams()).require_valid()
    rows = []
    for thr in grid:
        q = dataclasses.replace(p, mover_thr=thr, win_thr=thr, band_lo=min(p.band_lo, thr * 0.9), band_hi=max(p.band_hi, thr * 1.1))
        rec = observe_day(snap, out, now, dataclasses.replace(q, explain_limit=0))
        rows.append({"threshold": thr, **{c.value: rec.counts[c.value] for c in (MC.WINNER, MC.LOSER, MC.FALSE_POSITIVE, MC.FALSE_NEGATIVE,
                                                                                  MC.PREDICTABLE_MOVER, MC.UNPREDICTABLE_MOVER)}})
    return pd.DataFrame(rows).set_index("threshold")


def confusion_ledger(ledger: ObserverLedger, now=None) -> dict[str, float]:
    """Picks vs fill-based movers pooled over days: precision, recall, base rate, lift and the Matthews correlation. This is the
    one number-set that says whether the picks concentrate movers at all, independent of what they earned."""
    recs = ledger._recs if now is None else ledger.known(now)
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in recs:
        for k, v in r.model.get("confusion_fill", {}).items():
            tot[k] += int(v)
    tp, fp, fn, tn = tot["tp"], tot["fp"], tot["fn"], tot["tn"]
    n = tp + fp + fn + tn
    prec, rec_, base_rate = (tp / (tp + fp) if tp + fp else float("nan")), (tp / (tp + fn) if tp + fn else float("nan")), ((tp + fn) / n if n else float("nan"))
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {**{k: float(v) for k, v in tot.items()}, "precision": prec, "recall": rec_, "base_rate": base_rate,
            "lift": prec / base_rate if base_rate and base_rate == base_rate and prec == prec else float("nan"),
            "mcc": (tp * tn - fp * fn) / den if den > 0 else float("nan")}


def verify_checkpoint(ledger: ObserverLedger, directory, tag: str) -> list[str]:
    """Reload a saved ledger and compare it with the live one, day by day, through record_digest. An empty list = identical."""
    live = ledger.save(directory, tag) and ObserverLedger.load(directory, tag)
    bad = []
    if len(live) != len(ledger):
        bad.append(f"day count {len(live)} != {len(ledger)}")
    for a, b in zip(ledger, live):
        if record_digest(a) != record_digest(b):
            bad.append(f"day {a.day} differs after reload")
    return bad


# ==================================================================================================================
# trailing context, cells, transitions, significance and the history runner
# ==================================================================================================================
MARKET_KEYS = ("mover_rate", "dispersion", "median_ret", "breadth_up", "median_vol_x", "median_volume_ratio", "gap_median_abs")


def market_zscores(rec: DayRecord, ledger: ObserverLedger, window: int = 60, min_days: int = 10) -> dict[str, float]:
    """Today's market statistics as robust z-scores against the trailing `window` recorded days that matured no later than
    today's decision (so a day is never standardised by itself or by its future). NaN until `min_days` of history exist."""
    cut = as_date(rec.decided_at)
    past = [r for r in ledger._recs if as_date(r.resolved_at) <= cut and not r.empty][-window:]
    out = {}
    for k in MARKET_KEYS:
        hist = np.array([r.market.get(k, np.nan) for r in past], dtype=np.float64)
        hist = hist[np.isfinite(hist)]
        x = rec.market.get(k, np.nan)
        if len(hist) < min_days or x is None or not np.isfinite(x):
            out[k] = float("nan")
            continue
        med = float(np.median(hist))
        sc = 1.4826 * float(np.median(np.abs(hist - med)))
        out[k] = (float(x) - med) / sc if sc > 1e-12 else float("nan")
    return out


def context_cell(rec: DayRecord, z: Mapping[str, float] | None = None) -> str:
    """An opaque, identity-free situation key for the day (tone, dispersion, mover activity, gap activity). SurpriseTracker cells
    are strings of tokens like this; it names no date, year or stock, so it cannot be used to recognise a replayed year."""
    m = rec.market
    z = z or {}
    act = z.get("mover_rate", float("nan"))
    activity = "hot" if act == act and act > 1.5 else "cold" if act == act and act < -1.0 else "usual"
    gaps = z.get("gap_median_abs", float("nan"))
    gap_tok = "gappy" if gaps == gaps and gaps > 1.5 else "orderly"
    return f"tone={m.get('tone', 'unknown')}|disp={m.get('dispersion_level', 'normal')}|movers={activity}|gaps={gap_tok}"


def category_transitions(ledger: ObserverLedger, cats: Sequence[MoveCategory] = (MC.NEAR_MISS, MC.CONSIDERED_HIGH, MC.FALSE_NEGATIVE,
                                                                                MC.FALSE_POSITIVE), now=None) -> pd.DataFrame:
    """P(category tomorrow | category today) over names present in both days' rows, from consecutive recorded days. Rows only, so
    it describes exception names: 'a false negative today - is it a false negative again tomorrow?' Persistence of misses is the
    first hint that a miss has a learnable cause rather than being noise. Research side."""
    recs = [r for r in (ledger._recs if now is None else ledger.known(now)) if not r.rows.empty]
    n_tab = np.zeros((len(cats), len(cats) + 1))
    for a, b in zip(recs[:-1], recs[1:]):
        m = a.rows[["ticker", "flags"]].merge(b.rows[["ticker", "flags"]], on="ticker", how="left", suffixes=("_a", "_b"))
        fb = m["flags_b"].fillna(0).to_numpy().astype(np.int64)
        fa = m["flags_a"].to_numpy().astype(np.int64)
        for i, ca in enumerate(cats):
            sel = (fa & BIT[ca]) != 0
            if not sel.any():
                continue
            for j, cb in enumerate(cats):
                n_tab[i, j] += int(((fb[sel] & BIT[cb]) != 0).sum())
            n_tab[i, -1] += int(sel.sum())
    df = pd.DataFrame(n_tab, index=[c.value for c in cats], columns=[c.value for c in cats] + ["n_today"])
    prob = df[[c.value for c in cats]].div(df["n_today"].replace(0, np.nan), axis=0)
    return prob.assign(n_today=df["n_today"])


def precursor_significance(ledger: ObserverLedger, kind: str = "precursor_auc", now=None) -> dict[str, float]:
    """Is the precursor proxy (or the model score) better than a coin at ordering movers? Day-level AUCs against 0.5 with a
    sign test and a t statistic (days, not names, are the independent units). A proxy that fails here labels nothing:
    'predictable' would then be an arbitrary tag, and the autopsy says so instead of using it."""
    recs = ledger._recs if now is None else ledger.known(now)
    a = np.array([r.model.get(kind, np.nan) for r in recs], dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return {"days": float(len(a)), "mean_auc": float("nan"), "t": float("nan"), "sign_p": float("nan"), "informative": False}
    from scipy.stats import binomtest
    d = a - 0.5
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("inf") if d.mean() > 0 else float("nan")
    p = float(binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue) if (d != 0).any() else 1.0
    return {"days": float(len(a)), "mean_auc": float(a.mean()), "t": t, "sign_p": p, "informative": bool(a.mean() > 0.52 and p < 0.05)}


def run_history(bars: Mapping[str, pd.DataFrame], directory, years: Iterable[int] | None = None, params: ObserverParams | None = None,
                scorer=None, resume: bool = True, log=None) -> dict[int, ObserverLedger]:
    """Stream a long history one calendar year at a time (rule 27: never the whole panel in memory as records; each year is a
    checkpoint, and a killed run resumes at the first unfinished year). `bars` are wide OHLCV frames including delisted names."""
    p = (params or ObserverParams()).require_valid()
    slices = year_slices(pd.DatetimeIndex(bars["Close"].index))
    out = {}
    for y in sorted(slices if years is None else [y for y in years if y in slices]):
        a, b = slices[y]
        t0 = time.perf_counter()
        days = bars_stream(bars, p, scorer, start=a, end=b)
        out[y] = run_year(str(y), days, directory, p, resume=resume)
        if log:
            log(f"observer year {y}: {len(out[y])} days in {time.perf_counter() - t0:.1f}s")
    return out


# ==================================================================================================================
# uncertainty, merging, the episode hand-off and a one-day narrative
# ==================================================================================================================
def universe_bootstrap(snap: DecisionSnapshot, out: DayOutcome, now, p: ObserverParams | None = None, reps: int = 30, seed: int = 0,
                       level: float = 0.90) -> pd.DataFrame:
    """How sure is a day's category share? Resample the universe's names with replacement `reps` times, recompute the counts,
    and report the share of each category with its percentile interval. A share whose interval spans zero on a busy day is a
    small-sample artefact of the day, not a feature of the market. Deterministic in `seed`."""
    p = dataclasses.replace((p or ObserverParams()).require_valid(), explain_limit=0)
    rng = np.random.default_rng(seed)
    base_rec = observe_day(snap, out, now, p)
    aligned, _ = align_outcome(snap, out)
    n = snap.n
    shares = np.zeros((reps, len(CATEGORIES)))
    for b in range(reps):
        ix = rng.integers(0, n, n)
        s2 = DecisionSnapshot(snap.decided_at, snap.tickers[ix] + "#" + np.arange(n).astype(str), snap.eligible[ix], snap.score[ix],
                              snap.confidence[ix], snap.dir_prob[ix], snap.picked[ix], snap.abstained[ix], snap.filters[ix],
                              snap.prev_close[ix], snap.event_known[ix], snap.sector[ix], {k: v[ix] for k, v in snap.features.items()},
                              snap.filter_names)
        o2 = DayOutcome(aligned.resolved_at, s2.tickers, aligned.entry[ix], aligned.hi[ix], aligned.lo[ix], aligned.close[ix],
                        aligned.volume_ratio[ix], aligned.delisted[ix], aligned.day_hi[ix], aligned.day_lo[ix], aligned.day_close[ix])
        dc = classify_day(s2, o2, p)
        shares[b] = [dc.masks[c].mean() for c in CATEGORIES]
    a = (1.0 - level) / 2.0
    return pd.DataFrame({"share": [base_rec.counts[c.value] / max(1, base_rec.n_universe) for c in CATEGORIES],
                         "lo": np.quantile(shares, a, axis=0), "hi": np.quantile(shares, 1 - a, axis=0)},
                        index=[c.value for c in CATEGORIES])


def merge_ledgers(ledgers: Sequence[ObserverLedger]) -> ObserverLedger:
    """Join year ledgers into one, in day order. Refuses ledgers made with different parameters or with overlapping days
    (two runs of the same year are a duplicate, not a longer history), and it never merges different filing years silently."""
    if not ledgers:
        raise ObserverError("nothing to merge")
    if len({l.params.hash() for l in ledgers}) != 1:
        raise ObserverError("ledgers were made with different observer parameters")
    out = ObserverLedger(ledgers[0].params, filed_year=None)
    for rec in sorted((r for l in ledgers for r in l), key=lambda r: r.day):
        out.add(rec)                                   # add() rejects a repeated or out-of-order day
    out.universe_log = [u for l in sorted(ledgers, key=lambda l: l._recs[0].day if l._recs else "") for u in l.universe_log]
    return out


def iter_band_events(ledger: ObserverLedger, now, since=None) -> Iterator[dict[str, Any]]:
    """The episode-research feed (C67): every band mover of every day that matured strictly before `now`, oldest first, one
    dict per mover with the ROW_SCHEMA columns plus the day. Streaming, so the consumer never needs the whole history."""
    lo = as_date(since) if since is not None else None
    for rec in ledger.known(now):
        if lo is not None and as_date(rec.day) < lo:
            continue
        yield from as_records(rec, bands=True)


def winner_loser_test(ledger: ObserverLedger, now=None) -> dict[str, float]:
    """Are there systematically more winners than losers (or the reverse) across days? A paired sign test on daily counts. The
    market drifts up, so a small imbalance is normal; a large one says which side of the market the study is really about."""
    from scipy.stats import binomtest
    cf = ledger.counts_frame(now)
    if cf.empty:
        return {"days": 0.0, "mean_diff": float("nan"), "sign_p": float("nan")}
    d = (cf[MC.WINNER.value] - cf[MC.LOSER.value]).to_numpy(dtype=float)
    nz = int((d != 0).sum())
    return {"days": float(len(d)), "mean_diff": float(d.mean()), "winner_days": float((d > 0).sum()), "loser_days": float((d < 0).sum()),
            "sign_p": float(binomtest(int((d > 0).sum()), nz, 0.5).pvalue) if nz else 1.0}


def explain_day(rec: DayRecord, p: ObserverParams | None = None) -> str:
    """A short deterministic paragraph for one day: the universe, the bands, the model's hit and miss counts, and the data flags.
    It names no ticker (counts and shares only), so it can go into a report the trader side may later read."""
    if rec.empty:
        return f"{rec.day}: no universe recorded"
    c, b = rec.counts, rec.bands
    lines = [f"{rec.day}: {rec.n_universe} names ({rec.n_eligible} eligible, {rec.n_scored} scored); market {rec.market.get('tone')}, "
             f"dispersion {rec.market.get('dispersion_level')}, breadth up {100 * rec.market.get('breadth_up', float('nan')):.0f}%.",
             f"bands: {b['up_5_10']} up 5-10%, {b['up_gt10']} up >10%, {b['down_5_10']} down 5-10%, {b['down_gt10']} down >10%"
             f" ({rec.market.get('n_suspect_band', 0)} suspect data).",
             f"model: {rec.model.get('n_picked', 0)} picked, {c[MC.FALSE_POSITIVE.value]} false positives, {c[MC.FALSE_NEGATIVE.value]} false negatives, "
             f"{c[MC.NEAR_MISS.value]} near misses, {c[MC.ABSTAINED.value]} abstentions, {c[MC.LOW_CONFIDENCE.value]} low-confidence.",
             f"movers: {c[MC.PREDICTABLE_MOVER.value]} placed predictable, {c[MC.UNPREDICTABLE_MOVER.value]} placed unpredictable, "
             f"{rec.market.get('n_moved', 0) - c[MC.PREDICTABLE_MOVER.value] - c[MC.UNPREDICTABLE_MOVER.value]} left unplaced."]
    dropped = {k: v for k, v in rec.dropped.items() if v}
    if dropped:
        lines.append("rows sampled (counts are exact): " + ", ".join(f"{k} -{v}" for k, v in sorted(dropped.items())))
    return "\n".join(lines)


# ==================================================================================================================
# C67 day-shape of the band movers: how they closed and what drove the move (input to the next-day path research)
# ==================================================================================================================
SHAPES = ("closed_at_high", "closed_near_high", "closed_mid", "closed_near_low", "closed_at_low")
DRIVERS = ("gap_driven", "intraday_driven", "mixed")


def shape_of(close_loc: np.ndarray) -> np.ndarray:
    """Where the day closed in its own range, as one of five shapes; NaN location (no range) is 'closed_mid'."""
    loc = np.where(np.isfinite(close_loc), close_loc, 0.5)
    return np.select([loc >= 0.9, loc >= 0.65, loc >= 0.35, loc >= 0.1], list(SHAPES[:4]), default=SHAPES[4]).astype(object)


def driver_of(gap: np.ndarray, o2c: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """gap_driven if the overnight gap is >= thr of the close-to-close path (in absolute terms), intraday_driven if it is <= 1-thr."""
    share = gap_share(gap, o2c)
    return np.where(~np.isfinite(share), "mixed", np.where(share >= thr, "gap_driven", np.where(share <= 1 - thr, "intraday_driven", "mixed")))


def band_shapes(rec: DayRecord, p: ObserverParams | None = None) -> pd.DataFrame:
    """Band movers cross-tabulated by band x closing shape and by band x driver. `closed_at_low` after a big gap up is a very
    different setup from `closed_at_high`: this is the table the episode research splits on before it looks at tomorrow."""
    rf = rec.rows
    if rf.empty:
        return pd.DataFrame()
    b = rf[rf["band"] != 0]
    if b.empty:
        return pd.DataFrame()
    thr = (p or ObserverParams()).gap_share_thr
    shape = shape_of(b["close_loc"].to_numpy(dtype=float))
    drive = driver_of(b["gap"].to_numpy(dtype=float), b["o2c"].to_numpy(dtype=float), thr)
    s = pd.crosstab(b["band"].map(BAND_NAMES), pd.Series(shape, index=b.index)).reindex(columns=list(SHAPES), fill_value=0)
    d = pd.crosstab(b["band"].map(BAND_NAMES), pd.Series(drive, index=b.index)).reindex(columns=list(DRIVERS), fill_value=0)
    return pd.concat([s, d], axis=1)


def band_shapes_ledger(ledger: ObserverLedger, p: ObserverParams | None = None, now=None) -> pd.DataFrame:
    """band_shapes summed over days."""
    recs = ledger._recs if now is None else ledger.known(now)
    tabs = [band_shapes(r, p or ledger.params) for r in recs]
    tabs = [t for t in tabs if not t.empty]
    return sum((t.reindex(index=list(BAND_NAMES.values()), fill_value=0).fillna(0) for t in tabs), start=0) if tabs else pd.DataFrame()


def volume_by_band(rec: DayRecord) -> dict[str, float]:
    """Median session-volume ratio of each band's movers against the universe median: do 5-10% moves come on volume?"""
    rf = rec.rows
    uni = rec.market.get("median_volume_ratio", float("nan"))
    if rf.empty:
        return {}
    out = {}
    for code, name in BAND_NAMES.items():
        v = rf.loc[rf["band"] == code, "volume_ratio"].astype(float)
        v = v[np.isfinite(v)]
        out[name] = float(np.median(v) / uni) if len(v) and uni == uni and uni > 0 else float("nan")
    return out


def stale_or_thin(rec: DayRecord, min_share: float = 0.5, min_scored: int = 20) -> list[str]:
    """Reasons a day's record should not be trusted for research: too few scored names, most features missing or constant, or a
    sudden universe collapse. The autopsy still runs, but its questions carry the warning."""
    out = []
    if rec.n_universe and rec.n_scored / rec.n_universe < min_share:
        out.append("fewer than half the names were scored")
    if rec.n_scored < min_scored and not rec.empty:
        out.append("too few scored names for stable ranks")
    health = rec.market.get("data_health", {})
    bad = [k for k, v in health.items() if v.get("missing", 0) > 0.5 or v.get("constant", 0)]
    if bad:
        out.append("features missing or constant: " + ", ".join(sorted(bad)))
    return out


def band_transitions(ledger: ObserverLedger, now=None) -> pd.DataFrame:
    """For every band mover of day t that is also present in day t+1's rows: which band it was in on t+1 (0 = not a band mover
    that day, or not persisted as a row). Rows: today's band; columns: next recorded day's band. A name absent from tomorrow's rows
    is counted as band 0 only when tomorrow's record kept every band mover (always true) and it was not otherwise an exception,
    so the 0 column is exact for band membership. Research side (uses tickers)."""
    recs = [r for r in (ledger._recs if now is None else ledger.known(now)) if not r.empty]
    codes = [-2, -1, 0, 1, 2]
    tab = pd.DataFrame(0, index=[c for c in codes if c], columns=codes)
    for a, b in zip(recs[:-1], recs[1:]):
        if as_date(b.decided_at) <= as_date(a.decided_at):
            continue
        ta = a.rows.loc[a.rows["band"] != 0, ["ticker", "band"]]
        nb = b.rows.loc[b.rows["band"] != 0].set_index("ticker")["band"]
        nxt = ta["ticker"].map(nb).fillna(0).astype(int)
        for code, nx in zip(ta["band"].astype(int), nxt):
            tab.at[code, nx] += 1
    return tab


def band_run_lengths(ledger: ObserverLedger, now=None, max_len: int = 10) -> dict[int, int]:
    """Distribution of consecutive-day band-mover runs per name (1 = a one-day mover). Long runs are momentum or a broken feed."""
    recs = [r for r in (ledger._recs if now is None else ledger.known(now)) if not r.empty]
    active: dict[str, int] = {}
    runs: dict[int, int] = {}
    for r in recs:
        today = set(r.rows.loc[r.rows["band"] != 0, "ticker"]) if not r.rows.empty else set()
        for t in list(active):
            if t not in today:
                n = min(active.pop(t), max_len)
                runs[n] = runs.get(n, 0) + 1
        for t in today:
            active[t] = active.get(t, 0) + 1
    for t, n in active.items():
        runs[min(n, max_len)] = runs.get(min(n, max_len), 0) + 1
    return dict(sorted(runs.items()))


def category_outcomes(ledger: ObserverLedger, now=None, mover_thr: float | None = None) -> pd.DataFrame:
    """What actually happened to the names in each category, from the exception rows: count of rows, mover share (best excursion
    from the decision close at or above mover_thr), mean and median absolute excursion. Abstentions, low-confidence names and
    near misses are only worth revisiting if they move more than the rest; this is the table that says so. Rows are capped per
    category, but the cap keeps the highest-score names outright and samples the rest, so read the shares as descriptive."""
    thr = ledger.params.mover_thr if mover_thr is None else mover_thr
    rf = ledger.rows_frame(now)
    cols = ["rows", "mover_share", "mean_abs_exc", "median_abs_exc"]
    if rf.empty:
        return pd.DataFrame(columns=cols)
    exc = rf["exc_pc"].astype(float).to_numpy()
    ok = np.isfinite(exc)
    out = {}
    for c in CATEGORIES:
        m = has(rf["flags"].to_numpy(), c) & ok
        if m.any():
            out[c.value] = {"rows": int(m.sum()), "mover_share": float((exc[m] >= thr).mean()), "mean_abs_exc": float(exc[m].mean()),
                            "median_abs_exc": float(np.median(exc[m]))}
    return pd.DataFrame(out).T.reindex(columns=cols)


def pick_vs_universe(ledger: ObserverLedger, now=None) -> dict[str, float]:
    """Mean fill-to-horizon return of the picks against the universe mean, per day and pooled, with a day-level sign count. The
    universe mean comes from each day's exact market summary (not the capped rows)."""
    recs = [r for r in (ledger._recs if now is None else ledger.known(now)) if not r.empty]
    d = [(r.model.get("pick_mean_ret"), r.model.get("universe_mean_ret")) for r in recs]
    d = [(a, b) for a, b in d if a is not None and b is not None and a == a and b == b]
    if not d:
        return {"days": 0.0}
    diff = np.array([a - b for a, b in d])
    return {"days": float(len(d)), "pick_mean": float(np.mean([a for a, _ in d])), "universe_mean": float(np.mean([b for _, b in d])),
            "mean_edge": float(diff.mean()), "days_ahead": float((diff > 0).mean())}
