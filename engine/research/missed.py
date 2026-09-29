"""Missed-opportunity discovery engine (RESEARCH_BRAIN_CONTRACT C66 section 6, with the section 8 "could I have known?" audit;
serves C62 section 22 and canon C66/C63). IMPLEMENTED - NOT VALIDATED.

At the end of every simulated day the engine looks at the LARGEST MOVES of the whole cross-section, winners AND losers, and asks
the eight section-6 questions: which did the system predict, partially predict, or completely miss; which were predictable
from pre-move information; which near-misses were informative; which apparent misses were actually unknowable.

It EXTENDS, never copies:
  * engine.learning.missed_winners  - RejectionAnalyzer / RejectionReason / Week / Candidate explain WHY a missed winner was
    rejected; _match_by_rank pairs missed movers with same-rank controls. Here both are reused unchanged and mirrored for LOSERS
    (LossReason / LossAnalyzer) because the base module only knows winners.
  * engine.missed_winners           - WINNER threshold, winner_type vocabulary, sign_flip_p for the symmetry test.
  * engine.learning.trader_view     - find_violations guards every research question against carrying a name, date or year.

Mechanisms added here:
  1. Capture ladder      - PREDICTED / PARTIAL (undersized, movement-only, near-band, protected) / MISSED / ANTI_PREDICTED, defined
                           separately for a winner (was it owned?) and a loser (was it avoided on purpose, or by luck?).
  2. Pre-move predictability - LiftStore keeps per-day sufficient statistics of (feature bin x mover) counts. The auditor reads
                           only days that MATURED STRICTLY BEFORE the decision date (view(as_of)), so classifying today's move can
                           never use today's outcome. Shrunk, significance-gated log-lifts combine with damping into a posterior
                           mover probability; unknown stays unknown.
  3. Knowability report  - section 8 fields (knowledge_state_at_decision, future_information_used_by_auditor, information that was /
                           was not available, confidence_in_classification). The report lives in MATURED_RESEARCH_STATE and reaches a
                           decision only through MaturedRecord.gate(now); a same-year release guard refuses records whose year is
                           being replayed in disguise.
  4. Near-miss analysis  - rank-matched controls; the separating feature is informative only if it repeats.
  5. Ranking             - nine criteria, bounded weights (no single criterion may dominate), knowability gate, loss boost, opaque
                           tie-breaks, and Pareto fronts as a cross-check against a single scalar being gamed (section 43).
  6. Permanent stream    - every signature of a missed opportunity becomes a ResearchStream entry with a state machine
                           (QUEUED..DORMANT/CANCELLED), recurrence statistics and identity-free ResearchQuestions.
  7. Streaming           - a market-wide day is reduced to a DaySnapshot plus exception rows; full cross-sections are dropped.

Public entry: submit(state, day) then step(state, now) -> list[DailyReport]. Deterministic; every draw takes a seed."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import missed_winners as base
from engine.learning import missed_winners as MW
from engine.learning import trader_view as TV
from engine.learning.core import _StrEnum, as_date, clip01, current_code_hash
from engine.research.core import (Availability, DecisionEffect, ExperimentValue, FirewallBreach, Knowability, MaturedRecord, Namespace, Problem,
                                  Provenance, ResearchQuestion, ResearchState, require_past, stable_hash)

NL = chr(10)
WINNER = base.WINNER
UP, DOWN = 1, -1
INFO_KINDS = ("price", "volume", "technical", "cross_section", "macro", "event", "filing", "memory", "pattern", "market_state")
EXTERNAL_KINDS = ("macro", "event", "filing")
UNKNOWABLE = (Knowability.UNKNOWN, Knowability.EXTERNALLY_CAUSED, Knowability.INFORMATIONALLY_UNAVAILABLE)
LEARNABLE = (Knowability.PREDICTABLE, Knowability.POTENTIALLY_PREDICTABLE, Knowability.WEAKLY_PREDICTABLE)
PATHS = ("continuation", "reversal", "consolidation", "expansion")          # C67: what a mover did on the NEXT day
PATH_ALIASES = {"continued": "continuation", "reversed": "reversal", "consolidated": "consolidation", "expanded": "expansion",
                "stopped": "consolidation", "spiked": "expansion", "extended": "continuation", "faded": "reversal"}
CRITERIA = ("magnitude", "predictability", "confidence", "information", "pattern_similarity", "novelty", "repeatability",
            "loss_reduction", "decision_relevance")


# ==================================================================================================================
# parameters
# ==================================================================================================================
@dataclass(frozen=True)
class Params:
    mover_lo: float = 0.05                 # C67: the 5-10% band of movers whose next-day path is studied (hundreds per day)
    path_min_n: int = 30
    move_thr: float = WINNER               # a move at/above this is a winner (Bible: the +7% week)
    loss_thr: float = WINNER               # a move at/below minus this is a loser
    top_frac: float = 0.005                # the largest moves also include this share of the cross-section per side
    top_cap: int = 60                      # ...but at most this many rows per side are kept as exceptions
    keep_near: int = 40                    # near-band rows kept per day so near misses survive the reduction
    k: int = 10
    near_band: float = 0.5                 # rank within (1 + near_band) * k is a near miss
    flag_gate: float = 0.5                 # model mover-probability at/above this counts as "the system flagged it"
    full_size: float = 0.5                 # position share (of an equal-weight slot) at/above which a hold is a full capture
    protect_frac: float = 0.5              # a stop that kept the realised loss under this share of the move protected the book
    n_bins: int = 5
    shrink: float = 20.0                   # pseudo-counts pulling a bin's rate toward the base rate
    z_min: float = 2.0                     # a bin contributes evidence only when it deviates from base by this many sigmas
    min_bin_n: int = 30
    damp: float = 0.5                      # correlated features: the i-th strongest lift is weighted damp**i
    top_evidence: int = 4
    p_cap: float = 0.95
    min_history_days: int = 5
    min_history_names: int = 300
    pred_post_min: float = 0.12            # posterior mover probability for PREDICTABLE
    pred_lift_min: float = 2.0
    weak_lift_min: float = 1.3
    data_failure_missing: float = 0.5
    near_margin: float = 0.7               # a model probability within this share of the gate is a near miss
    sep_min_d: float = 0.5                 # standardised paired difference for a separating feature
    sig_top: int = 3                       # a signature is built from the sig_top most extreme features
    sig_extreme: float = 0.25              # ...and a feature is extreme when its rank is within this of 0 or 1
    dormant_after: int = 20                # days without recurrence before a stream entry goes DORMANT
    cancel_unknowable: float = 0.9
    cancel_min_obs: int = 12
    promote_days: int = 3
    alpha: float = 0.05
    novelty_cap: int = 5000
    loss_boost: float = 1.25
    unknowable_factor: float = 0.15
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "magnitude": 0.14, "predictability": 0.17, "confidence": 0.06, "information": 0.10, "pattern_similarity": 0.08,
        "novelty": 0.10, "repeatability": 0.14, "loss_reduction": 0.13, "decision_relevance": 0.08})
    max_weight: float = 0.4

    def validate(self) -> list[str]:
        errs = []
        for name in ("move_thr", "loss_thr"):
            if not 0.0 < getattr(self, name) < 1.0:
                errs.append(f"{name} must lie in (0, 1)")
        for name in ("flag_gate", "full_size", "protect_frac", "near_margin", "p_cap", "alpha", "cancel_unknowable"):
            if not 0.0 < getattr(self, name) <= 1.0:
                errs.append(f"{name} must lie in (0, 1]")
        if self.n_bins < 2 or self.k < 1 or self.sig_top < 1:
            errs.append("n_bins >= 2, k >= 1 and sig_top >= 1 required")
        if set(self.weights) != set(CRITERIA):
            errs.append("weights must name exactly the nine section-6 criteria")
        else:
            if abs(sum(self.weights.values()) - 1.0) > 1e-9:
                errs.append(f"weights sum to {sum(self.weights.values()):.6f}, not 1")
            errs += [f"weight {c}={w} outside [0, {self.max_weight}]" for c, w in self.weights.items() if not 0.0 <= w <= self.max_weight]
        if self.pred_lift_min <= self.weak_lift_min:
            errs.append("pred_lift_min must exceed weak_lift_min")
        return errs


class Capture(_StrEnum):
    PREDICTED = "PREDICTED"
    PARTIAL_UNDERSIZED = "PARTIAL_UNDERSIZED"
    PARTIAL_MOVEMENT_ONLY = "PARTIAL_MOVEMENT_ONLY"      # the size of the move was flagged, its direction was not
    PARTIAL_NEAR_BAND = "PARTIAL_NEAR_BAND"
    PARTIAL_PROTECTED = "PARTIAL_PROTECTED"              # a loser we owned but a stop cut most of it
    LUCKY_AVOID = "LUCKY_AVOID"                          # a loser we did not own, with nothing in the trail that says why
    MISSED = "MISSED"
    ANTI_PREDICTED = "ANTI_PREDICTED"                    # the system bet against it
    NOT_A_MOVE = "NOT_A_MOVE"


FAMILY = {Capture.PREDICTED: "predicted", Capture.PARTIAL_UNDERSIZED: "partial", Capture.PARTIAL_MOVEMENT_ONLY: "partial",
          Capture.PARTIAL_NEAR_BAND: "partial", Capture.PARTIAL_PROTECTED: "partial", Capture.LUCKY_AVOID: "missed",
          Capture.MISSED: "missed", Capture.ANTI_PREDICTED: "missed", Capture.NOT_A_MOVE: "none"}


class LossReason(_StrEnum):
    """Mirror of learning.missed_winners.RejectionReason for the loser side: why did we OWN (or fail to flag) a big loser."""
    RISK_FLAG_OVERRIDDEN = "RISK_FLAG_OVERRIDDEN"
    OVERCONFIDENT_BET = "OVERCONFIDENT_BET"
    DIRECTION_WRONG = "DIRECTION_WRONG"
    CONTEXT_IGNORED = "CONTEXT_IGNORED"
    BAD_RELIABILITY_IGNORED = "BAD_RELIABILITY_IGNORED"
    GAP_UNAVOIDABLE = "GAP_UNAVOIDABLE"
    STOP_TOO_LOOSE = "STOP_TOO_LOOSE"
    NO_RISK_SIGNAL = "NO_RISK_SIGNAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


LOSS_PRECEDENCE = (LossReason.GAP_UNAVOIDABLE, LossReason.RISK_FLAG_OVERRIDDEN, LossReason.CONTEXT_IGNORED, LossReason.DIRECTION_WRONG,
                   LossReason.BAD_RELIABILITY_IGNORED, LossReason.OVERCONFIDENT_BET, LossReason.STOP_TOO_LOOSE,
                   LossReason.NO_RISK_SIGNAL, LossReason.INSUFFICIENT_EVIDENCE)
LOSS_EFFECT = {LossReason.RISK_FLAG_OVERRIDDEN: DecisionEffect.SELECTION, LossReason.OVERCONFIDENT_BET: DecisionEffect.CONFIDENCE,
               LossReason.DIRECTION_WRONG: DecisionEffect.DIRECTION, LossReason.CONTEXT_IGNORED: DecisionEffect.PATTERN_WEIGHTING,
               LossReason.BAD_RELIABILITY_IGNORED: DecisionEffect.PATTERN_WEIGHTING, LossReason.GAP_UNAVOIDABLE: DecisionEffect.POSITION_SIZE,
               LossReason.STOP_TOO_LOOSE: DecisionEffect.STOP, LossReason.NO_RISK_SIGNAL: DecisionEffect.SELECTION,
               LossReason.INSUFFICIENT_EVIDENCE: DecisionEffect.NONE, LossReason.UNKNOWN: DecisionEffect.NONE}
WIN_EFFECT = {MW.RR.OVER_AGGRESSIVE_RISK_FILTER: DecisionEffect.SELECTION, MW.RR.DIRECTION_DISAGREEMENT: DecisionEffect.DIRECTION,
              MW.RR.TIMING: DecisionEffect.TIMING, MW.RR.WRONG_CONTEXT: DecisionEffect.PATTERN_WEIGHTING,
              MW.RR.BAD_RELIABILITY: DecisionEffect.PATTERN_WEIGHTING, MW.RR.WRONG_CONFIDENCE: DecisionEffect.CONFIDENCE,
              MW.RR.MISSING_INTERACTION: DecisionEffect.RANKING, MW.RR.WRONG_RANKING: DecisionEffect.RANKING,
              MW.RR.INSUFFICIENT_EVIDENCE: DecisionEffect.NONE, MW.RR.UNKNOWN: DecisionEffect.NONE}
EFFECT_REACH = {DecisionEffect.SELECTION: 1.0, DecisionEffect.RANKING: 0.9, DecisionEffect.DIRECTION: 0.9, DecisionEffect.STOP: 0.8,
                DecisionEffect.POSITION_SIZE: 0.8, DecisionEffect.TIMING: 0.7, DecisionEffect.CONFIDENCE: 0.6,
                DecisionEffect.PATTERN_WEIGHTING: 0.6, DecisionEffect.EXIT: 0.7, DecisionEffect.ABSTENTION: 0.5,
                DecisionEffect.RESEARCH_PRIORITY: 0.3, DecisionEffect.NONE: 0.0}


def norm_path(x) -> str | None:
    """Map any next-day path label (R21's episode_paths taxonomy is duck-typed) onto the four classes; unknown -> None."""
    if x is None:
        return None
    s = str(getattr(x, "value", x)).lower().strip()
    s = PATH_ALIASES.get(s, s)
    return s if s in PATHS else None


# ==================================================================================================================
# data model
# ==================================================================================================================
@dataclass(frozen=True)
class InfoItem:
    """One piece of information the auditor knows about, with the moment it became available (section 8)."""
    kind: str
    name: str                              # opaque label ("earnings_release"), never a ticker
    available_at: str | None = None
    after_close: bool = False              # published on `available_at` but after the decision close
    expected: bool = True                  # such information should exist for this kind of move
    used_by_model: bool = False
    strength: float = 0.5                  # auditor's prior for how informative this kind of item is, 0..1

    def validate(self) -> list[str]:
        errs = []
        if self.kind not in INFO_KINDS:
            errs.append(f"info kind {self.kind!r} not one of {INFO_KINDS}")
        if not self.name or not 0.0 <= self.strength <= 1.0:
            errs.append("info item needs a name and strength in [0, 1]")
        return errs


def availability(item: InfoItem, decided_at, matured_at) -> Availability:
    """When did this item exist relative to the decision close? Missing timestamps are never read as 'known before'."""
    if item.available_at is None:
        return Availability.UNAVAILABLE if item.expected else Availability.UNCERTAIN
    d, dec = as_date(item.available_at), as_date(decided_at)
    if d < dec:
        return Availability.KNOWN_BEFORE_EVENT
    if d == dec:
        return Availability.SIMULTANEOUS if item.after_close else Availability.KNOWN_BEFORE_EVENT
    return Availability.KNOWN_ONLY_AFTER_EVENT


@dataclass(frozen=True)
class MoveObs:
    """One name on one decision day: the pre-move features and decision trail the system had, plus the outcome. `fwd` is read
    only by ex-post analyses; features are cross-sectional ranks in [0, 1] (NaN = missing), never levels that identify a name."""
    cid: str
    decided_at: str
    matured_at: str
    fwd: float
    features: Mapping[str, float]
    picked: bool = False
    weight: float | None = None            # share of the book; None with picked=True means one equal slot
    realised: float | None = None          # return actually booked after stops/exits, if owned
    gap: float | None = None               # the overnight-gap part of fwd
    score: float | None = None
    rank: int | None = None
    eligible: bool = True
    filters_hit: tuple[str, ...] = ()
    confidence: float | None = None
    pred_prob: float | None = None         # the system's probability that this name makes a large move
    pred_dir: int | None = None
    risk_flag: bool = False                # the system marked it "avoid"
    reliability: float | None = None
    anti_context: bool = False
    timing_blocked: bool = False
    missing_frac: float = 0.0
    kind: str = "other"
    info: tuple[InfoItem, ...] = ()
    next_path: str | None = None           # C67: what the mover did the next day; any R21 label is mapped by norm_path
    next_ret: float | None = None
    path_pred: str | None = None           # what the system predicted for that next day, if it predicted one

    def move_dir(self, p: "Params") -> int:
        if not math.isfinite(self.fwd):
            return 0
        return UP if self.fwd >= p.move_thr else DOWN if self.fwd <= -p.loss_thr else 0

    def validate(self) -> list[str]:
        errs = []
        if not self.cid:
            errs.append("cid missing")
        if any(str(k).lower() in ("ticker", "symbol", "name") for k in self.features):
            errs.append(f"{self.cid}: identity key in features")
        for k, v in self.features.items():
            if v is not None and math.isfinite(v) and not 0.0 <= v <= 1.0:
                errs.append(f"{self.cid}: feature {k}={v} is not a rank in [0, 1]")
        if as_date(self.matured_at) <= as_date(self.decided_at):
            errs.append(f"{self.cid}: matured_at must follow decided_at")
        for name in ("confidence", "pred_prob", "reliability"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                errs.append(f"{self.cid}: {name}={v} outside [0, 1]")
        if self.pred_dir not in (None, -1, 0, 1):
            errs.append(f"{self.cid}: pred_dir must be -1, 0, 1 or None")
        if self.weight is not None and self.weight < 0:
            errs.append(f"{self.cid}: negative weight")
        for nm in ("next_path", "path_pred"):
            v = getattr(self, nm)
            if v is not None and norm_path(v) is None:
                errs.append(f"{self.cid}: {nm}={v!r} is not a known path class")
        for it in self.info:
            errs += it.validate()
        return errs


@dataclass(frozen=True)
class DaySnapshot:
    """What survives of a full cross-section once the day is processed (rule 27: exception rows plus one snapshot)."""
    decided_at: str
    matured_at: str
    n_names: int
    n_up: int
    n_down: int
    market_ret: float
    dispersion: float
    era: str = ""
    context: Mapping[str, float] = field(default_factory=dict)

    @property
    def up_rate(self) -> float:
        return self.n_up / self.n_names if self.n_names else float("nan")

    @property
    def down_rate(self) -> float:
        return self.n_down / self.n_names if self.n_names else float("nan")


@dataclass(frozen=True)
class DayBook:
    """The whole cross-section of one closed decision day, transient: engine.step reduces it to a snapshot plus exceptions."""
    decided_at: str
    matured_at: str
    obs: tuple[MoveObs, ...]
    feature_names: tuple[str, ...]
    k: int = 10
    era: str = ""
    context: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        if as_date(self.matured_at) <= as_date(self.decided_at):
            errs.append("day: matured_at must follow decided_at")
        seen = set()
        for o in self.obs:
            errs += o.validate()
            if o.cid in seen:
                errs.append(f"duplicate cid {o.cid}")
            seen.add(o.cid)
            if o.decided_at != self.decided_at or o.matured_at != self.matured_at:
                errs.append(f"{o.cid}: dates differ from the day it is filed under")
        return errs

    def matrix(self) -> np.ndarray:
        X = np.full((len(self.obs), len(self.feature_names)), np.nan, dtype=np.float32)
        for i, o in enumerate(self.obs):
            for j, f in enumerate(self.feature_names):
                v = o.features.get(f)
                if v is not None:
                    X[i, j] = v
        return X

    def fwd(self) -> np.ndarray:
        return np.array([o.fwd if o.fwd is not None else np.nan for o in self.obs], dtype=np.float64)

    def snapshot(self, p: Params) -> DaySnapshot:
        f = self.fwd()
        ok = np.isfinite(f)
        n = int(ok.sum())
        return DaySnapshot(self.decided_at, self.matured_at, n, int((f[ok] >= p.move_thr).sum()), int((f[ok] <= -p.loss_thr).sum()),
                           float(f[ok].mean()) if n else float("nan"), float(f[ok].std()) if n > 1 else float("nan"), self.era,
                           dict(self.context))

    @staticmethod
    def from_frame(frame: pd.DataFrame, decided_at, matured_at, feature_cols: Sequence[str], fwd_col: str = "fwd", salt: str = "ms",
                   k: int = 10, era: str = "", context: Mapping[str, float] | None = None, info: Mapping[Any, Sequence[InfoItem]] | None = None
                   ) -> "DayBook":
        """Build from a ticker-indexed frame. Names become salted hashes and features cross-sectional ranks, so nothing that
        identifies a name or a year survives. Optional trail columns carry the MoveObs field names."""
        cols = [c for c in feature_cols if c in frame.columns]
        ranks = MW.cross_rank(frame, cols)
        trail = [f.name for f in dataclasses.fields(MoveObs) if f.name in frame.columns and f.name not in ("cid", "fwd", "features", "info")]
        obs = []
        for tkr in frame.index:
            row = frame.loc[tkr]
            kw = {}
            for name in trail:
                v = row[name]
                if isinstance(v, (float, np.floating)) and not math.isfinite(v):
                    continue
                kw[name] = v.item() if isinstance(v, np.generic) else v
            if "filters_hit" in kw and not isinstance(kw["filters_hit"], tuple):
                kw["filters_hit"] = tuple(kw["filters_hit"]) if isinstance(kw["filters_hit"], (list, set)) else ()
            f = float(row[fwd_col]) if fwd_col in row and pd.notna(row[fwd_col]) else float("nan")
            obs.append(MoveObs(cid=stable_hash({"s": salt, "d": str(decided_at), "t": str(tkr)}, 12), decided_at=str(as_date(decided_at)),
                               matured_at=str(as_date(matured_at)), fwd=f, features={c: float(ranks.at[tkr, c]) for c in cols},
                               info=tuple((info or {}).get(tkr, ())), **kw))
        return DayBook(str(as_date(decided_at)), str(as_date(matured_at)), tuple(obs), tuple(cols), k, era, dict(context or {}))


def select_exceptions(day: DayBook, p: Params) -> tuple[MoveObs, ...]:
    """The rows worth keeping once the day is processed: every move past a threshold or in the extreme tail of either side
    (capped), everything the system picked, and the near band under the cut. Order is by opaque cid, never by name."""
    rows = [o for o in day.obs if math.isfinite(o.fwd)]
    if not rows:
        return ()
    n_top = min(p.top_cap, max(1, int(math.ceil(p.top_frac * len(rows)))))
    by = sorted(rows, key=lambda o: (o.fwd, o.cid))
    keep = {o.cid: o for o in by[:n_top] + by[-n_top:]}
    movers = sorted([o for o in rows if o.move_dir(p) != 0], key=lambda o: (-abs(o.fwd), o.cid))[: 2 * p.top_cap]
    keep.update({o.cid: o for o in movers})
    keep.update({o.cid: o for o in rows if o.picked})
    band = sorted([o for o in rows if o.rank is not None and day.k < o.rank <= day.k * (1 + p.near_band) and not o.picked],
                  key=lambda o: (o.rank, o.cid))[: p.keep_near]
    keep.update({o.cid: o for o in band})
    return tuple(keep[c] for c in sorted(keep))


# ==================================================================================================================
# 1. the capture ladder: predicted / partially predicted / missed
# ==================================================================================================================
@dataclass(frozen=True)
class CaptureResult:
    capture: Capture
    coverage: float                        # 0..1: how much of the move the system captured (or of the loss it avoided)
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return FAMILY[self.capture]


def position_size(o: MoveObs, k: int) -> float:
    """Owned share expressed in equal-weight slots, capped at 1. Owned with no stated weight means one full slot."""
    if not o.picked:
        return 0.0
    if o.weight is None:
        return 1.0
    return float(min(1.0, o.weight * k))


def avoid_intent(o: MoveObs) -> bool:
    """A loser we did not own was avoided ON PURPOSE only if the trail says so: a risk flag, a bearish direction call or a hard
    risk-type filter. Everything else is luck and must not be counted as skill."""
    return bool(o.risk_flag or o.pred_dir == DOWN or (o.filters_hit and not o.eligible))


def classify_capture(o: MoveObs, p: Params, k: int | None = None) -> CaptureResult:
    """Place one move on the ladder. Winners are judged by whether we owned or flagged them; losers by whether we kept out of
    them intentionally. A non-move (inside both thresholds) is NOT_A_MOVE and never enters the stream."""
    k = k or p.k
    d = o.move_dir(p)
    size = position_size(o, k)
    flagged = o.pred_prob is not None and o.pred_prob >= p.flag_gate
    if d == 0:
        return CaptureResult(Capture.NOT_A_MOVE, 0.0)
    if d == UP:
        if o.picked:
            if size >= p.full_size:
                return CaptureResult(Capture.PREDICTED, size, {"size": size})
            return CaptureResult(Capture.PARTIAL_UNDERSIZED, size, {"size": size})
        if o.pred_dir == DOWN and flagged:
            return CaptureResult(Capture.ANTI_PREDICTED, 0.0, {"pred_prob": o.pred_prob})
        if flagged:
            return CaptureResult(Capture.PARTIAL_MOVEMENT_ONLY, 0.5, {"pred_prob": o.pred_prob, "dir_known": o.pred_dir is not None})
        if o.rank is not None and k < o.rank <= k * (1 + p.near_band):
            over = (o.rank - k) / max(1.0, k * p.near_band)
            return CaptureResult(Capture.PARTIAL_NEAR_BAND, round(0.35 * (1.0 - over) + 0.05, 6), {"rank": o.rank, "cut": k})
        return CaptureResult(Capture.MISSED, 0.0)
    if not o.picked:
        if avoid_intent(o):
            return CaptureResult(Capture.PREDICTED, 1.0 if (o.risk_flag or o.pred_dir == DOWN) else 0.8, {"intentional": True})
        return CaptureResult(Capture.LUCKY_AVOID, 0.25, {"intentional": False})
    if o.realised is not None and abs(o.fwd) > 0 and abs(min(o.realised, 0.0)) <= p.protect_frac * abs(o.fwd):
        return CaptureResult(Capture.PARTIAL_PROTECTED, round(1.0 - abs(min(o.realised, 0.0)) / abs(o.fwd), 6), {"realised": o.realised})
    if o.pred_dir == UP and (o.confidence or 0.0) >= 0.6:
        return CaptureResult(Capture.ANTI_PREDICTED, 0.0, {"confidence": o.confidence})
    if o.risk_flag and size <= p.full_size:
        return CaptureResult(Capture.PARTIAL_UNDERSIZED, round(1.0 - size, 6), {"size": size, "flagged_avoid": True})
    return CaptureResult(Capture.MISSED, 0.0, {"size": size})


def loss_incurred(o: MoveObs, k: int) -> float:
    """Book loss (as a fraction of an equal slot times the slot share) that this loser actually cost. Zero if not owned."""
    if not o.picked or o.fwd >= 0:
        return 0.0
    r = o.realised if o.realised is not None else o.fwd
    return float(-min(r, 0.0) * max(position_size(o, k), 1e-9) / k)


# ==================================================================================================================
# 2. signatures, novelty, known patterns
# ==================================================================================================================
def signature_codes(X: np.ndarray, p: Params) -> np.ndarray:
    """Integer code per row for its sig_top most extreme features and their side (low/high). 0 = nothing extreme. Vectorised so
    every name of the day, not just movers, can be given a signature and the enrichment of a signature among movers measured."""
    n, F = X.shape
    if n == 0 or F == 0:
        return np.zeros(n, dtype=np.int64)
    Xf = np.where(np.isfinite(X), X, 0.5)
    extreme = (Xf <= p.sig_extreme) | (Xf >= 1.0 - p.sig_extreme)
    dev = np.where(extreme, np.abs(Xf - 0.5), -1.0)
    top = min(p.sig_top, F)
    idx = np.argsort(-dev, axis=1, kind="stable")[:, :top]
    rows = np.arange(n)[:, None]
    valid = dev[rows, idx] >= 0
    tok = np.where(valid, idx * 2 + (Xf[rows, idx] >= 0.5).astype(np.int64) + 1, 0)
    tok = np.sort(tok, axis=1)
    B = 2 * F + 1
    code = np.zeros(n, dtype=np.int64)
    for i in range(top):
        code += tok[:, i].astype(np.int64) * (B ** i)
    return code


def decode_signature(code: int, names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Inverse of signature_codes: ((feature, 'high'|'low'), ...) sorted by feature name; () = undistinguished."""
    B = 2 * len(names) + 1
    out = []
    while code > 0:
        t = code % B
        code //= B
        if t > 0:
            out.append((names[(t - 1) // 2], "high" if (t - 1) % 2 else "low"))
    return tuple(sorted(out))


def signature_of(o: MoveObs, names: Sequence[str], p: Params) -> int:
    x = np.array([[o.features.get(f, np.nan) for f in names]], dtype=np.float32)
    return int(signature_codes(x, p)[0])


def feature_vector(o: MoveObs, names: Sequence[str]) -> np.ndarray:
    """Centered rank vector (rank - 0.5, missing = 0) scaled to unit length, for cosine similarity."""
    v = np.array([o.features.get(f, np.nan) for f in names], dtype=np.float64)
    v = np.where(np.isfinite(v), v - 0.5, 0.0)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-12 else v


class NoveltyIndex:
    """Ring buffer of the unit feature vectors of everything already put in the research stream. Novelty of a new opportunity
    is one minus its largest cosine to any stored vector; an empty index makes everything fully novel."""

    def __init__(self, dim: int, cap: int = 5000):
        self.dim, self.cap = dim, cap
        self.buf = np.zeros((0, dim), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.buf)

    def novelty(self, v: np.ndarray) -> float:
        if len(self.buf) == 0 or float(np.linalg.norm(v)) < 1e-12:
            return 1.0
        return float(clip01(1.0 - max(0.0, float((self.buf @ v).max()))))

    def add(self, v: np.ndarray) -> None:
        if float(np.linalg.norm(v)) < 1e-12:
            return
        self.buf = np.vstack([self.buf, v[None, :]])[-self.cap:]


@dataclass(frozen=True)
class PatternSig:
    """A known pattern reduced to rank intervals per feature, so a missed opportunity can be compared with what is known."""
    pattern_id: str
    direction: int
    conds: Mapping[str, tuple[float, float]]

    def match(self, features: Mapping[str, float]) -> float:
        """1.0 if every condition holds; falls linearly to 0 as a rank leaves its interval by 0.25. Missing feature = no match."""
        if not self.conds:
            return 0.0
        parts = []
        for f, (lo, hi) in self.conds.items():
            v = features.get(f)
            if v is None or not math.isfinite(v):
                parts.append(0.0)
                continue
            out = max(lo - v, v - hi, 0.0)
            parts.append(max(0.0, 1.0 - out / 0.25))
        return float(np.mean(parts))


def best_pattern_match(o: MoveObs, patterns: Sequence[PatternSig], direction: int) -> tuple[float, str | None]:
    """Highest match among patterns that point the same way as the move (a pattern for the opposite side is not 'known')."""
    best, pid = 0.0, None
    for pat in patterns:
        if pat.direction not in (0, direction):
            continue
        m = pat.match(o.features)
        if m > best:
            best, pid = m, pat.pattern_id
    return best, pid


# ==================================================================================================================
# 3. pre-move predictability from sufficient statistics that mature strictly before the decision
# ==================================================================================================================
COL_ALL, COL_UP, COL_DN, COL_BAND = 0, 1, 2, 3
COL_PATH0 = 4
N_COLS = COL_PATH0 + len(PATHS)


@dataclass(frozen=True)
class PreMoveEvidence:
    """What history known BEFORE the decision says about one name. p is None when history is too thin: that is INSUFFICIENT,
    never 'probability zero'."""
    target: str
    base: float | None
    p: float | None
    lift: float | None
    contributors: tuple[tuple[str, int, float, float, int], ...]      # (feature, bin, lift, z, n)
    n_days: int
    n_names: int
    enough: bool

    @property
    def n_contributors(self) -> int:
        return len(self.contributors)


class LiftView:
    """A frozen sum of the per-day counts of every day that matured strictly before `as_of`."""

    def __init__(self, names: Sequence[str], counts: np.ndarray, totals: np.ndarray, n_days: int, as_of: str, p: Params):
        self.names, self.counts, self.totals, self.n_days, self.as_of, self.p = tuple(names), counts, totals, n_days, as_of, p

    def _cols(self, target: str) -> tuple[list[int], list[int]]:
        if target == "up":
            return [COL_UP], [COL_ALL]
        if target == "down":
            return [COL_DN], [COL_ALL]
        if target == "move":
            return [COL_UP, COL_DN], [COL_ALL]
        if target == "band":
            return [COL_BAND], [COL_ALL]
        if target in PATHS:
            return [COL_PATH0 + PATHS.index(target)], list(range(COL_PATH0, N_COLS))
        raise ValueError(f"unknown target {target!r}")

    def base_rate(self, target: str) -> float | None:
        kc, nc = self._cols(target)
        n = float(self.totals[nc].sum())
        return float(self.totals[kc].sum() / n) if n > 0 else None

    def evidence(self, features: Mapping[str, float], target: str = "move") -> PreMoveEvidence:
        p = self.p
        kc, nc = self._cols(target)
        base_rate = self.base_rate(target)
        n_total = int(self.totals[nc].sum())
        need = p.path_min_n if target in PATHS else p.min_history_names
        enough = self.n_days >= p.min_history_days and n_total >= need and base_rate is not None and 0.0 < base_rate < 1.0
        if not enough:
            return PreMoveEvidence(target, base_rate, None, None, (), self.n_days, n_total, False)
        contrib = []
        for j, f in enumerate(self.names):
            v = features.get(f)
            if v is None or not math.isfinite(v):
                continue
            b = min(int(v * p.n_bins), p.n_bins - 1)
            cell = self.counts[j, b]
            n, kk = float(cell[nc].sum()), float(cell[kc].sum())
            if n < p.min_bin_n:
                continue
            rate = (kk + p.shrink * base_rate) / (n + p.shrink)
            lift = rate / base_rate
            z = (kk - n * base_rate) / math.sqrt(max(n * base_rate * (1.0 - base_rate), 1e-9))
            if abs(z) >= p.z_min:
                contrib.append((f, b, float(lift), float(z), int(n)))
        contrib.sort(key=lambda t: (-abs(math.log(max(t[2], 1e-9))), t[0]))
        top = tuple(contrib[: p.top_evidence])
        total = sum(math.log(max(c[2], 1e-9)) * (p.damp ** i) for i, c in enumerate(top))
        odds = base_rate / (1.0 - base_rate) * math.exp(total)
        post = min(p.p_cap, odds / (1.0 + odds))
        return PreMoveEvidence(target, base_rate, float(post), float(post / base_rate), top, self.n_days, n_total, True)


class LiftStore:
    """Per-day (feature bin x outcome) counts. Keeping one small tensor per day (not a running sum) is what makes the audit
    exact: view(as_of) can include precisely the days that matured before the decision being audited and no others."""

    def __init__(self, names: Sequence[str], p: Params | None = None):
        self.names, self.p = tuple(names), p or Params()
        self._days: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self._cache: dict[str, LiftView] = {}

    def __len__(self) -> int:
        return len(self._days)

    def add_day(self, day: DayBook) -> None:
        if tuple(day.feature_names) != self.names:
            raise ValueError("day's feature names differ from the store's")
        if day.decided_at in self._days:
            raise ValueError(f"day {day.decided_at} already added: history is append-only")
        p = self.p
        X, f = day.matrix(), day.fwd()
        ok = np.isfinite(f)
        band = ok & (np.abs(f) >= p.mover_lo)
        paths = np.array([PATHS.index(norm_path(o.next_path)) if norm_path(o.next_path) is not None else -1 for o in day.obs], dtype=np.int64)
        masks = [ok, ok & (f >= p.move_thr), ok & (f <= -p.loss_thr), band] + [band & (paths == i) for i in range(len(PATHS))]
        cnt = np.zeros((len(self.names), p.n_bins, N_COLS), dtype=np.float64)
        for j in range(len(self.names)):
            valid = np.isfinite(X[:, j])
            b = np.minimum((np.where(valid, X[:, j], 0.0) * p.n_bins).astype(np.int64), p.n_bins - 1)
            for c, m in enumerate(masks):
                sel = valid & m
                if sel.any():
                    cnt[j, :, c] = np.bincount(b[sel], minlength=p.n_bins)
        tot = np.array([m.sum() for m in masks], dtype=np.float64)
        self._days[day.decided_at] = (day.matured_at, cnt, tot)
        self._cache.clear()

    def view(self, as_of) -> LiftView:
        """Sum only days whose outcomes matured STRICTLY before as_of. A day never sees its own outcome or a later one."""
        key = str(as_date(as_of))
        if key in self._cache:
            return self._cache[key]
        cnt = np.zeros((len(self.names), self.p.n_bins, N_COLS), dtype=np.float64)
        tot = np.zeros(N_COLS, dtype=np.float64)
        n = 0
        for matured, c, t in self._days.values():
            if as_date(matured) < as_date(as_of):
                cnt += c
                tot += t
                n += 1
        v = LiftView(self.names, cnt, tot, n, key, self.p)
        self._cache[key] = v
        return v

    def through(self) -> str | None:
        return max((m for m, _, _ in self._days.values()), default=None)


class SigStats:
    """Signature frequencies over every name (not only movers), so a signature's enrichment among movers is a proper
    binomial test against how often the signature occurs at all."""

    def __init__(self, names: Sequence[str], p: Params):
        self.names, self.p = tuple(names), p
        self.rows: dict[int, list[int]] = {}      # code -> [all, up, down, band, days_with_a_mover, last_day_index, total_mover_occurrences]
        self.n_days = 0
        self.totals = np.zeros(4, dtype=np.float64)

    def add_day(self, day: DayBook) -> None:
        X, f = day.matrix(), day.fwd()
        ok = np.isfinite(f)
        codes = signature_codes(X, self.p)
        masks = [ok, ok & (f >= self.p.move_thr), ok & (f <= -self.p.loss_thr), ok & (np.abs(f) >= self.p.mover_lo)]
        self.totals += np.array([m.sum() for m in masks], dtype=np.float64)
        for c, m in enumerate(masks):
            if not m.any():
                continue
            u, cts = np.unique(codes[m], return_counts=True)
            for code, ct in zip(u.tolist(), cts.tolist()):
                r = self.rows.setdefault(int(code), [0, 0, 0, 0, 0, -1, 0])
                r[c] += int(ct)
                if c == 3:
                    if r[5] != self.n_days:
                        r[4] += 1
                        r[5] = self.n_days
                    r[6] += int(ct)
        self.n_days += 1

    def enrichment(self, code: int, direction: int) -> tuple[float, float, int]:
        """(rate ratio, one-sided binomial p, n occurrences) of `code` among movers of `direction` versus chance."""
        r = self.rows.get(int(code))
        n_all = self.totals[0]
        if r is None or n_all <= 0 or r[0] < self.p.min_bin_n:
            return 1.0, 1.0, 0 if r is None else r[0]
        col = COL_UP if direction == UP else COL_DN
        base_rate = self.totals[col] / n_all
        if base_rate <= 0 or base_rate >= 1:
            return 1.0, 1.0, r[0]
        hits = r[col]
        pv = float(sps.binom.sf(hits - 1, r[0], base_rate)) if hits > 0 else 1.0
        return float((hits / r[0]) / base_rate), pv, r[0]

    def spread(self, code: int) -> tuple[int, float]:
        """(distinct days a mover carried the signature, days observed expected if occurrences were scattered uniformly)."""
        r = self.rows.get(int(code))
        if r is None or self.n_days == 0:
            return 0, 0.0
        D = self.n_days
        return r[4], float(D * (1.0 - (1.0 - 1.0 / D) ** r[6]))


# ==================================================================================================================
# 4. "could I have known?" - the section-8 counterfactual audit
# ==================================================================================================================
@dataclass(frozen=True)
class KnowabilityReport:
    """Section 8 output. It is built from matured outcomes, so it belongs to MATURED_RESEARCH_STATE: the only way out is
    to_matured_record(...).gate(now). `future_information_used_by_auditor` is always non-empty (the outcome itself), which is
    exactly why the classification must never be handed to the trader."""
    cid: str
    decided_at: str
    direction: int
    knowability: Knowability
    confidence_in_classification: float
    knowledge_state_at_decision: Mapping[str, Any]
    future_information_used_by_auditor: tuple[str, ...]
    information_that_would_have_been_available: tuple[str, ...]
    information_that_was_unavailable: tuple[str, ...]
    p_mover: float | None
    lift: float | None
    reasons: tuple[str, ...] = ()
    namespace: Namespace = Namespace.MATURED_RESEARCH

    @property
    def learnable(self) -> bool:
        return self.knowability in LEARNABLE

    @property
    def unknowable(self) -> bool:
        return self.knowability in UNKNOWABLE + (Knowability.DATA_FAILURE,)


def _history_confidence(ev: PreMoveEvidence, p: Params, missing: float) -> float:
    depth = min(1.0, ev.n_days / max(1.0, 12.0 * p.min_history_days)) * min(1.0, ev.n_names / (10.0 * p.min_history_names))
    return clip01(depth * (1.0 - missing))


def assess(o: MoveObs, view: LiftView, p: Params) -> KnowabilityReport:
    """Reconstruct the information state at the decision date and classify. Uses only `view` (days matured before the decision)
    plus the availability stamps on the move's own info items. Never raises on thin history: that is UNKNOWN with confidence 0."""
    dirn = o.move_dir(p)
    target = "up" if dirn == UP else "down" if dirn == DOWN else "move"
    ev = view.evidence(o.features, target)
    avail = {it.name: availability(it, o.decided_at, o.matured_at) for it in o.info}
    before = [it for it in o.info if avail[it.name] in (Availability.KNOWN_BEFORE_EVENT,)]
    after = [it for it in o.info if avail[it.name] == Availability.KNOWN_ONLY_AFTER_EVENT]
    unavailable = [it for it in o.info if avail[it.name] == Availability.UNAVAILABLE]
    unused = [it for it in before if not it.used_by_model and it.strength >= 0.6]
    ext_after = [it for it in after if it.kind in EXTERNAL_KINDS and it.strength >= 0.5]
    state = {"history_days": ev.n_days, "history_names": ev.n_names, "base_rate": ev.base,
             "evidence": [(f, b, round(l, 4)) for f, b, l, _, _ in ev.contributors], "kinds_available": sorted({it.kind for it in before}),
             "kinds_used_by_model": sorted({it.kind for it in before if it.used_by_model}), "missing_frac": o.missing_frac}
    future = ("outcome:fwd",) + tuple(f"info_after_decision:{it.name}" for it in after)
    seen = tuple(it.name for it in before)
    lacking = tuple(it.name for it in unavailable)
    reasons: list[str] = []
    if o.missing_frac >= p.data_failure_missing:
        k, conf = Knowability.DATA_FAILURE, 0.7
        reasons.append(f"{o.missing_frac:.0%} of the inputs were missing at the decision")
    elif not ev.enough:
        k, conf = Knowability.UNKNOWN, 0.0
        reasons.append("history matured before the decision is too thin to say anything: unknown, not unpredictable")
    else:
        conf = _history_confidence(ev, p, o.missing_frac)
        strong = ev.p >= p.pred_post_min and ev.lift >= p.pred_lift_min and ev.n_contributors >= 2
        if strong:
            k = Knowability.PREDICTABLE
            reasons.append(f"posterior {ev.p:.2f} = {ev.lift:.1f}x base from {ev.n_contributors} independent-looking features")
        elif ev.lift >= p.pred_lift_min or (unused and ev.lift >= p.weak_lift_min) or (unused and not after):
            k = Knowability.POTENTIALLY_PREDICTABLE
            reasons.append("evidence was present but thin or unused" if ev.lift >= p.pred_lift_min else "informative data existed but the model did not use it")
            conf *= 0.8
        elif ev.lift >= p.weak_lift_min:
            k = Knowability.WEAKLY_PREDICTABLE
            reasons.append(f"a weak tilt of {ev.lift:.2f}x base")
            conf *= 0.6
        elif ext_after:
            k = Knowability.EXTERNALLY_CAUSED
            reasons.append("no pre-move signal, and an external item became known only after the decision")
            conf *= 0.7
        elif lacking:
            k = Knowability.INFORMATIONALLY_UNAVAILABLE
            reasons.append("no pre-move signal, and information that should exist was not in the data")
            conf *= 0.6
        else:
            k = Knowability.UNKNOWN
            reasons.append("no pre-move signal and no identified external cause: cause unknown")
            conf *= 0.5
    return KnowabilityReport(o.cid, o.decided_at, dirn, k, float(clip01(conf)), state, future, seen, lacking, ev.p, ev.lift, tuple(reasons))


def to_matured_record(rep: KnowabilityReport, prov: Provenance, matured_at: str) -> MaturedRecord:
    """The only exit from the auditor: a MaturedRecord that the trader side can read only through gate(now)."""
    payload = {"kind": "knowability", "cid": rep.cid, "direction": rep.direction, "knowability": str(rep.knowability),
               "confidence": rep.confidence_in_classification, "p_mover": rep.p_mover, "lift": rep.lift, "reasons": list(rep.reasons)}
    return MaturedRecord("KR" + stable_hash({"c": rep.cid, "d": rep.decided_at, "k": str(rep.knowability)}, 12), matured_at, payload, prov)


class ReleaseGate:
    """Same-year-rerun guard (mapping rule 27): research filed under a real year must never be released while that year is being
    replayed in disguise. The guard sits on the trusted side; the trader never sees which years are being replayed."""

    def __init__(self, replaying_years: Iterable[int] = ()):
        self.replaying = {int(y) for y in replaying_years}

    def set_replaying(self, years: Iterable[int]) -> None:
        self.replaying = {int(y) for y in years}

    def release(self, records: Sequence[MaturedRecord], now) -> list[Mapping[str, Any]]:
        out = []
        for r in records:
            if as_date(r.matured_at).year in self.replaying:
                raise FirewallBreach(f"record {r.record_id}: its year is being replayed in disguise; release refused")
            out.append(r.gate(now))
        return out


# ==================================================================================================================
# 5. near misses and the rejection reasons (extending learning.missed_winners for winners, mirrored for losers)
# ==================================================================================================================
@dataclass(frozen=True)
class NearMiss:
    cid: str
    direction: int
    source: str                            # rank | prob | partial
    gap: float                             # places under the cut, or probability under the gate
    informative: bool
    separating: tuple[tuple[str, float], ...]
    why: str


class NearMissLedger:
    """Accumulates, across days, the rank-matched paired difference (near-miss winner minus a same-rank false positive) per
    feature. A feature is a stable separator only if the mean pair difference is large, its t-statistic clears the bar, and its
    sign agrees on most days: one lucky day cannot make a near miss look informative."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        self.n = np.zeros(len(names))
        self.s = np.zeros(len(names))
        self.ss = np.zeros(len(names))
        self.pos_days = np.zeros(len(names))
        self.days = np.zeros(len(names))

    def add(self, diffs: np.ndarray) -> None:
        """diffs: (pairs, features) of paired rank differences for one day; NaN entries are skipped."""
        if diffs.size == 0:
            return
        ok = np.isfinite(diffs)
        d = np.where(ok, diffs, 0.0)
        cnt = ok.sum(axis=0)
        self.n += cnt
        self.s += d.sum(axis=0)
        self.ss += (d ** 2).sum(axis=0)
        has = cnt > 0
        self.days += has
        self.pos_days += has & (d.sum(axis=0) > 0)

    def stats(self) -> pd.DataFrame:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.n > 0, self.s / self.n, np.nan)
            var = np.where(self.n > 1, (self.ss - self.n * mean ** 2) / (self.n - 1), np.nan)
            se = np.sqrt(np.maximum(var, 0) / self.n)
            t = np.where(se > 0, mean / se, 0.0)
            agree = np.where(self.days > 0, np.maximum(self.pos_days, self.days - self.pos_days) / self.days, np.nan)
        return pd.DataFrame({"feature": self.names, "n_pairs": self.n.astype(int), "mean_diff": mean, "t": t, "day_agreement": agree,
                             "days": self.days.astype(int)})

    def stable(self, t_min: float = 2.5, min_pairs: int = 20, agree_min: float = 0.6, min_days: int = 3) -> list[str]:
        df = self.stats()
        ok = df[(df.n_pairs >= min_pairs) & (df.t.abs() >= t_min) & (df.day_agreement >= agree_min) & (df.days >= min_days)]
        return sorted(ok.feature.tolist())


def paired_differences(obs: Sequence[MoveObs], names: Sequence[str], k: int, p: Params, direction: int = UP) -> np.ndarray:
    """Rank-matched pairs via learning.missed_winners._match_by_rank. UP: each near-miss winner (unowned, just under the cut) is
    paired with the nearest-ranked false positive (an unmoved candidate in the ranked band that lost). DOWN: each owned big
    loser is paired with the nearest-ranked owned name that gained, so the difference is what warned of the loss.
    Returns pair differences (mover - control) with one column per feature."""
    band = k * (1 + p.near_band)
    rows = []
    for o in obs:
        if o.rank is None or o.rank > band * 2:
            continue
        d = o.move_dir(p)
        if direction == UP:
            g1 = d == UP and not o.picked and o.rank > k
            g0 = d == 0 and o.fwd <= 0
        else:
            g1 = d == DOWN and o.picked
            g0 = o.picked and o.fwd > 0
        if g1:
            rows.append({"period": o.decided_at, "cid": o.cid, "group": 1, "_rank": o.rank, "o": o})
        elif g0:
            rows.append({"period": o.decided_at, "cid": o.cid, "group": 0, "_rank": o.rank, "o": o})
    matched = MW._match_by_rank(rows, MW.MissedParams().rank_caliper)
    a = [r["o"] for r in matched if r["group"] == 1]
    b = [r["o"] for r in matched if r["group"] == 0]
    if not a:
        return np.zeros((0, len(names)))
    va = np.array([[o.features.get(f, np.nan) for f in names] for o in a])
    vb = np.array([[o.features.get(f, np.nan) for f in names] for o in b])
    return va - vb


def find_near_misses(obs: Sequence[MoveObs], k: int, p: Params, reports: Mapping[str, KnowabilityReport], stable: Sequence[str]
                     ) -> list[NearMiss]:
    """Moves that were not captured but came close, judged informative when the auditor found real evidence for them or the
    features that separate near misses from controls (across days) are the ones this move stands out on."""
    out = []
    for o in obs:
        d = o.move_dir(p)
        if d == 0:
            continue
        cap = classify_capture(o, p, k)
        if cap.family == "predicted" or cap.capture in (Capture.LUCKY_AVOID, Capture.ANTI_PREDICTED):
            continue
        if cap.capture == Capture.PARTIAL_NEAR_BAND and o.rank is not None:
            src, gap = "rank", float(o.rank - k)
        elif o.pred_prob is not None and p.flag_gate * p.near_margin <= o.pred_prob < p.flag_gate:
            src, gap = "prob", float(p.flag_gate - o.pred_prob)
        elif cap.family == "partial":
            src, gap = "partial", float(1.0 - cap.coverage)
        else:
            continue
        rep = reports.get(o.cid)
        extreme = sorted(((f, float(v - 0.5)) for f, v in o.features.items() if v is not None and math.isfinite(v) and f in stable),
                         key=lambda t: (-abs(t[1]), t[0]))[:3]
        has_ev = rep is not None and rep.learnable and (rep.p_mover or 0.0) >= p.pred_post_min * p.near_margin
        aligned = any(abs(v) >= 0.25 for _, v in extreme)
        why = ("the auditor found pre-move evidence" if has_ev else "") + (" and " if has_ev and aligned else "") + \
              ("it stands out on features that separate near misses from controls" if aligned else "")
        out.append(NearMiss(o.cid, d, src, gap, bool(has_ev or aligned), tuple(extreme), why or "nothing in the pre-move information singled it out"))
    return sorted(out, key=lambda n: (n.gap, n.cid))


def to_candidate(o: MoveObs) -> MW.Candidate:
    """MoveObs -> the base module's Candidate, so RejectionAnalyzer explains a missed winner without a second implementation."""
    feats = {f: float(v) for f, v in o.features.items() if v is not None and math.isfinite(v)}
    return MW.Candidate(cid=o.cid, features=feats, fwd=o.fwd if math.isfinite(o.fwd) else None, picked=o.picked, score=o.score, rank=o.rank,
                        eligible=o.eligible, filters_hit=tuple(o.filters_hit), confidence=o.confidence, dir_side=o.pred_dir,
                        selection_side=UP if o.pred_dir is not None else None, reliability=o.reliability, anti_context=o.anti_context,
                        timing_blocked=o.timing_blocked, missing_frac=o.missing_frac, kind=o.kind)


def explain_missed_winners(obs: Sequence[MoveObs], day: DayBook, p: Params, analyzer: MW.RejectionAnalyzer | None = None
                           ) -> dict[str, MW.Rejection]:
    """Base-module rejection reasons for every winner the system did not own. Missed winners with no supporting trail come back
    UNKNOWN: the trail, not a guess, decides."""
    analyzer = analyzer or MW.RejectionAnalyzer(MW.MissedParams())
    cands = tuple(to_candidate(o) for o in obs)
    week = MW.Week(day.decided_at, day.decided_at, day.matured_at, cands, k=day.k, thr=p.move_thr, era=day.era)
    n_ranked = sum(1 for o in day.obs if o.rank is not None) or None
    return {c.cid: analyzer.explain(c, week, n_ranked) for c in week.missed()}


@dataclass(frozen=True)
class LossExplanation:
    cid: str
    primary: LossReason
    reasons: tuple[tuple[LossReason, float], ...]
    effect: DecisionEffect
    avoidable_by_stop: bool


class LossAnalyzer:
    """Mirror of RejectionAnalyzer for a big loser we OWNED. Reasons are ordered by mechanism precedence exactly as the winner
    side does: a gap loss is unavoidable by any stop, so it is named first even when a risk flag also fired. Nothing is forced:
    no supported reason gives UNKNOWN."""

    def __init__(self, params: Params | None = None, accept: float = 0.4):
        self.p, self.accept = params or Params(), accept

    def evidence(self, o: MoveObs) -> list[tuple[LossReason, float]]:
        ev = []
        gap_share = abs(o.gap) / abs(o.fwd) if o.gap is not None and o.fwd else 0.0
        if o.fwd < 0 and o.gap is not None and o.gap < 0 and gap_share >= 0.6:
            ev.append((LossReason.GAP_UNAVOIDABLE, min(1.0, 0.5 + 0.5 * gap_share)))
        if o.risk_flag:
            ev.append((LossReason.RISK_FLAG_OVERRIDDEN, 0.9))
        if o.anti_context:
            ev.append((LossReason.CONTEXT_IGNORED, 0.85))
        if o.pred_dir == UP:
            ev.append((LossReason.DIRECTION_WRONG, 0.5 + 0.4 * (o.confidence if o.confidence is not None else 0.0)))
        if o.reliability is not None and o.reliability < 0.4:
            ev.append((LossReason.BAD_RELIABILITY_IGNORED, 0.4 + (0.4 - o.reliability)))
        if o.confidence is not None and o.confidence >= 0.7:
            ev.append((LossReason.OVERCONFIDENT_BET, 0.5 + 0.5 * (o.confidence - 0.7) / 0.3))
        if o.realised is not None and o.fwd < 0 and o.realised <= 0.85 * o.fwd and gap_share < 0.6:
            ev.append((LossReason.STOP_TOO_LOOSE, 0.6))
        if o.missing_frac >= 0.3:
            ev.append((LossReason.INSUFFICIENT_EVIDENCE, min(1.0, o.missing_frac + 0.2)))
        if not ev and not o.risk_flag:
            ev.append((LossReason.NO_RISK_SIGNAL, 0.5))
        return ev

    def explain(self, o: MoveObs) -> LossExplanation:
        if not o.picked:
            raise ValueError(f"{o.cid} was not owned: there is no loss to explain")
        acc = sorted([(r, s) for r, s in self.evidence(o) if s >= self.accept], key=lambda t: LOSS_PRECEDENCE.index(t[0]))
        if not acc:
            return LossExplanation(o.cid, LossReason.UNKNOWN, (), DecisionEffect.NONE, False)
        primary = acc[0][0]
        if primary == LossReason.INSUFFICIENT_EVIDENCE and len(acc) > 1:
            primary = acc[1][0]
        return LossExplanation(o.cid, primary, tuple(acc), LOSS_EFFECT[primary], primary != LossReason.GAP_UNAVOIDABLE)

    def explain_all(self, obs: Sequence[MoveObs]) -> dict[str, LossExplanation]:
        return {o.cid: self.explain(o) for o in obs if o.picked and o.move_dir(self.p) == DOWN}


# ==================================================================================================================
# 6. ranking missed opportunities (winners AND losers) by the nine section-6 criteria
# ==================================================================================================================
@dataclass(frozen=True)
class Opportunity:
    cid: str
    decided_at: str
    direction: int
    capture: Capture
    coverage: float
    fwd: float
    knowability: Knowability
    signature: int
    sig_id: str
    criteria: Mapping[str, float]
    score: float
    reason: str                            # RejectionReason / LossReason value, or UNKNOWN
    effect: DecisionEffect
    loss_cost: float
    pattern_id: str | None
    era: str = ""
    kind: str = "other"
    path: str | None = None                # next-day path class (C67), if known
    mult: float = 1.0                      # knowability gate x loss boost applied to the weighted criteria

    @property
    def is_loss(self) -> bool:
        return self.direction == DOWN


def information_score(o: MoveObs, p: Params) -> float:
    """Share of the (strength-weighted) information that existed before the decision. With no items catalogued the answer is a
    neutral half, scaled by completeness - absence of a catalogue is not evidence of absence of information."""
    complete = 1.0 - min(1.0, o.missing_frac)
    if not o.info:
        return 0.5 * complete
    tot = sum(max(it.strength, 1e-6) for it in o.info)
    got = sum(max(it.strength, 1e-6) for it in o.info if availability(it, o.decided_at, o.matured_at) == Availability.KNOWN_BEFORE_EVENT)
    return float(clip01(got / tot * complete))


def predictability_score(rep: KnowabilityReport, p: Params) -> float:
    if rep.p_mover is None or rep.lift is None:
        return 0.0
    return float(clip01((rep.lift - 1.0) / (2.0 * p.pred_lift_min - 1.0)) * rep.confidence_in_classification)


def repeatability_score(code: int, stats: SigStats, p: Params) -> float:
    """A signature that recurs on many separate days, is enriched among movers versus chance and is not one crash-day cluster."""
    if code == 0:
        return 0.0
    days, expected = stats.spread(code)
    r = stats.rows.get(int(code))
    if r is None or days == 0:
        return 0.0
    ratio, pv, _ = stats.enrichment(code, UP if r[1] >= r[2] else DOWN)
    persistence = min(1.0, days / 8.0)
    cluster = 0.5 if r[6] / max(days, 1) > 8.0 else 1.0
    signif = 1.0 if pv < p.alpha and ratio > 1.0 else 0.5
    return float(clip01(persistence * cluster * signif))


def loss_reduction_score(o: MoveObs, reason: str, info: float, p: Params) -> float:
    """Expected loss avoided if the cause were fixed: only owned losers count, a gap loss no stop can prevent is excluded,
    and the potential is discounted by how much information existed to act on."""
    if o.fwd >= 0 or not o.picked:
        return 0.0
    r = o.realised if o.realised is not None else o.fwd
    size = position_size(o, p.k)
    avoidable = 0.0 if reason == LossReason.GAP_UNAVOIDABLE.value else 1.0
    return float(clip01(-min(r, 0.0) / 0.30 * size * avoidable * (0.5 + 0.5 * info)))


def build_opportunities(obs: Sequence[MoveObs], day: DayBook, p: Params, reports: Mapping[str, KnowabilityReport],
                        rejections: Mapping[str, MW.Rejection], losses: Mapping[str, LossExplanation], patterns: Sequence[PatternSig],
                        novelty: NoveltyIndex, stats: SigStats) -> list[Opportunity]:
    """Score every uncaptured or partly captured move on all nine criteria. Nothing here reads the model's own future: only the
    decision trail, the auditor's report and the stream statistics up to and including today."""
    names = day.feature_names
    X = np.array([[o.features.get(f, np.nan) for f in names] for o in obs], dtype=np.float32) if obs else np.zeros((0, len(names)), np.float32)
    codes = signature_codes(X, p)
    out = []
    for o, code in zip(obs, codes.tolist()):
        cap = classify_capture(o, p, day.k)
        if cap.family in ("predicted", "none"):
            continue
        d = o.move_dir(p)
        rep = reports.get(o.cid)
        if rep is None:
            continue
        rej, lex = rejections.get(o.cid), losses.get(o.cid)
        reason = lex.primary.value if lex else (rej.primary.value if rej else "UNKNOWN")
        effect = lex.effect if lex else (WIN_EFFECT.get(rej.primary, DecisionEffect.NONE) if rej else DecisionEffect.RESEARCH_PRIORITY)
        info = information_score(o, p)
        pmatch, pid = best_pattern_match(o, patterns, d)
        vec = feature_vector(o, names)
        days_seen, _ = stats.spread(code)
        crit = {"magnitude": float(clip01(math.tanh(abs(o.fwd) / 0.25))), "predictability": predictability_score(rep, p),
                "confidence": float(abs((o.confidence if o.confidence is not None else 0.5) - 0.5) * 2.0), "information": info,
                "pattern_similarity": float(pmatch), "novelty": novelty.novelty(vec),
                "repeatability": repeatability_score(code, stats, p), "loss_reduction": loss_reduction_score(o, reason, info, p),
                "decision_relevance": float(EFFECT_REACH.get(effect, 0.0) * (0.5 + 0.5 * min(1.0, days_seen / 8.0)))}
        raw = sum(p.weights[c] * crit[c] for c in CRITERIA)
        gate = 0.0 if rep.knowability == Knowability.DATA_FAILURE else p.unknowable_factor if rep.knowability in UNKNOWABLE else \
            {Knowability.PREDICTABLE: 1.0, Knowability.POTENTIALLY_PREDICTABLE: 0.9, Knowability.WEAKLY_PREDICTABLE: 0.6}.get(rep.knowability, 0.5)
        mult = gate * (p.loss_boost if d == DOWN else 1.0)
        score = min(1.0, raw * mult)
        out.append(Opportunity(o.cid, o.decided_at, d, cap.capture, cap.coverage, o.fwd, rep.knowability, int(code),
                               stable_hash({"n": list(names), "c": int(code), "d": d}, 10), crit, float(score), reason, effect,
                               loss_incurred(o, day.k), pid, day.era, o.kind, norm_path(o.next_path), float(mult)))
    return rank_opportunities(out)


def rank_opportunities(opps: Sequence[Opportunity]) -> list[Opportunity]:
    """Highest score first. Ties break on an opaque hash of (cid, date): never on a name, never on input order."""
    return sorted(opps, key=lambda x: (-round(x.score, 12), stable_hash({"c": x.cid, "d": x.decided_at}, 12)))


def explain_rank(x: Opportunity, weights: Mapping[str, float], top: int = 3) -> str:
    parts = sorted(((weights[c] * x.criteria[c], c) for c in CRITERIA), reverse=True)[:top]
    return f"score {x.score:.3f} [{x.knowability}] " + ", ".join(f"{c}={v:.3f}" for v, c in parts)


def pareto_fronts(opps: Sequence[Opportunity]) -> list[int]:
    """Non-dominated front index (0 = best) of each opportunity across the nine criteria. If the scalar score and the Pareto
    front disagree a lot, the weights, not the data, are doing the ranking (section 43: do not optimise the wrong metric)."""
    if not opps:
        return []
    M = np.array([[x.criteria[c] for c in CRITERIA] for x in opps])
    front = np.full(len(opps), -1, dtype=int)
    remaining = np.arange(len(opps))
    level = 0
    while remaining.size:
        sub = M[remaining]
        dominated = np.array([np.any(np.all(sub >= sub[i], axis=1) & np.any(sub > sub[i], axis=1)) for i in range(len(remaining))])
        front[remaining[~dominated]] = level
        remaining = remaining[dominated]
        level += 1
    return front.tolist()


def rank_stability(opps: Sequence[Opportunity], p: Params, seed: int = 0, n: int = 50, top: int = 10, jitter: float = 0.3) -> dict[str, float]:
    """Share of random weight perturbations under which each opportunity stays in the top `top`. Perturbations respect the
    weight cap. A top item that survives only under the shipped weights is a property of the weights, not of the data."""
    if not opps:
        return {}
    rng = np.random.default_rng(seed)
    w0 = np.array([p.weights[c] for c in CRITERIA])
    M = np.array([[x.criteria[c] for c in CRITERIA] for x in opps])
    gates = np.array([x.mult for x in opps])
    hits = np.zeros(len(opps))
    for _ in range(n):
        w = np.clip(w0 * np.exp(rng.normal(0.0, jitter, size=w0.shape)), 1e-6, None)
        w = w / w.sum()
        w = np.minimum(w, p.max_weight)
        w = w / w.sum()
        s = np.minimum(1.0, (M @ w) * gates)
        hits[np.argsort(-s, kind="stable")[:top]] += 1
    return {x.cid: float(h / n) for x, h in zip(opps, hits)}


# ==================================================================================================================
# 7. the permanent research stream
# ==================================================================================================================
@dataclass(frozen=True)
class StreamEntry:
    """One recurring kind of missed opportunity. Immutable: advance() returns a new entry, so history is never rewritten."""
    sig_id: str
    code: int
    direction: int
    features: tuple[tuple[str, str], ...]
    state: ResearchState
    first_day: int
    last_day: int
    n_obs: int
    n_days: int
    sum_score: float
    best_score: float
    sum_abs_move: float
    loss_cost: float
    learnable: int
    unknowable: int
    reasons: tuple[tuple[str, int], ...]
    paths: tuple[tuple[str, int], ...] = ()
    history: tuple[tuple[int, str], ...] = ()
    eras: tuple[tuple[str, int], ...] = ()

    @property
    def mean_score(self) -> float:
        return self.sum_score / self.n_obs if self.n_obs else 0.0

    @property
    def unknowable_share(self) -> float:
        return self.unknowable / self.n_obs if self.n_obs else 0.0

    def top_reason(self) -> str:
        return max(self.reasons, key=lambda t: (t[1], t[0]))[0] if self.reasons else "UNKNOWN"

    def top_path(self) -> str | None:
        return max(self.paths, key=lambda t: (t[1], t[0]))[0] if self.paths else None


def _bump(pairs: tuple[tuple[str, int], ...], key: str) -> tuple[tuple[str, int], ...]:
    d = dict(pairs)
    d[key] = d.get(key, 0) + 1
    return tuple(sorted(d.items()))


class ResearchStream:
    """Append-only stream of candidate research topics. Entry states follow ResearchState: QUEUED on first sight; EXPLORING once
    it has recurred; PROMISING once it recurs on several days with real evidence; DORMANT when it stops recurring; CANCELLED when
    it is persistently unknowable (research on it would only waste compute). It never reaches FAILED/RETIRED here: only an
    experiment can fail a topic, and this module runs none."""

    def __init__(self, names: Sequence[str], p: Params | None = None):
        self.names, self.p = tuple(names), p or Params()
        self.entries: dict[str, StreamEntry] = {}
        self.day_index = 0

    def __len__(self) -> int:
        return len(self.entries)

    def update(self, opps: Sequence[Opportunity], stats: SigStats) -> list[StreamEntry]:
        """Fold one day's opportunities in and re-evaluate every entry's state. Returns the entries touched today."""
        self.day_index += 1
        touched = []
        seen_today: dict[str, list[Opportunity]] = {}
        for x in opps:
            if x.signature == 0 or x.knowability == Knowability.DATA_FAILURE:
                continue
            seen_today.setdefault(x.sig_id, []).append(x)
        for sid, group in seen_today.items():
            e = self.entries.get(sid)
            if e is None:
                x0 = group[0]
                e = StreamEntry(sid, x0.signature, x0.direction, decode_signature(x0.signature, self.names), ResearchState.QUEUED,
                                self.day_index, self.day_index, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, (), (), ((self.day_index, "QUEUED"),))
            reasons, paths, eras = e.reasons, e.paths, e.eras
            for x in group:
                eras = _bump(eras, x.era)
                reasons = _bump(reasons, x.reason)
                if x.path:
                    paths = _bump(paths, x.path)
            e = dataclasses.replace(
                e, last_day=self.day_index, n_obs=e.n_obs + len(group), n_days=e.n_days + 1, sum_score=e.sum_score + sum(x.score for x in group),
                best_score=max(e.best_score, max(x.score for x in group)), sum_abs_move=e.sum_abs_move + sum(abs(x.fwd) for x in group),
                loss_cost=e.loss_cost + sum(x.loss_cost for x in group), learnable=e.learnable + sum(x.knowability in LEARNABLE for x in group),
                unknowable=e.unknowable + sum(x.knowability in UNKNOWABLE for x in group), reasons=reasons, paths=paths, eras=eras)
            self.entries[sid] = self._transition(e, stats)
            touched.append(self.entries[sid])
        for sid, e in list(self.entries.items()):
            if sid not in seen_today:
                self.entries[sid] = self._transition(e, stats)
        return touched

    def _transition(self, e: StreamEntry, stats: SigStats) -> StreamEntry:
        p, st = self.p, e.state
        new = st
        if st in (ResearchState.QUEUED, ResearchState.EXPLORING, ResearchState.PROMISING, ResearchState.DORMANT):
            if e.n_obs >= p.cancel_min_obs and e.unknowable_share >= p.cancel_unknowable:
                new = ResearchState.CANCELLED
            elif self.day_index - e.last_day >= p.dormant_after:
                new = ResearchState.DORMANT if st != ResearchState.CANCELLED else st
            elif e.n_days >= 2 and st in (ResearchState.QUEUED, ResearchState.DORMANT):
                new = ResearchState.EXPLORING
            if new in (ResearchState.EXPLORING, ResearchState.PROMISING) and e.n_days >= p.promote_days and e.learnable / max(e.n_obs, 1) >= 0.5:
                ratio, pv, _ = stats.enrichment(e.code, e.direction)
                if pv < p.alpha and ratio > 1.0:
                    new = ResearchState.PROMISING
        if new != st:
            return dataclasses.replace(e, state=new, history=e.history + ((self.day_index, new.value),))
        return e

    def top(self, n: int = 10, states: Sequence[ResearchState] | None = None) -> list[StreamEntry]:
        """Best topics by mean score then by persistence; opaque sig_id breaks ties."""
        pool = [e for e in self.entries.values() if states is None or e.state in states]
        return sorted(pool, key=lambda e: (-round(e.mean_score * math.log1p(e.n_days), 9), e.sig_id))[:n]

    def questions(self, created_real: str, evidence_through: str, n: int = 10) -> list[ResearchQuestion]:
        """Identity-free research questions for the best live topics. Each one is checked against trader_view.find_violations
        before it leaves: a question that names a date, year or ticker-shaped token is dropped, never released."""
        out = []
        for e in self.top(n * 3, (ResearchState.QUEUED, ResearchState.EXPLORING, ResearchState.PROMISING)):
            side = "gain" if e.direction == UP else "lose"
            cond = " and ".join(f"{f} is {s}" for f, s in e.features) or "no single feature is extreme"
            path = e.top_path()
            text = f"Do names where {cond} precede a {side} of more than the mover threshold more often than chance, and can that be known before the close?"
            if path:
                text += f" After such a move, does the next day tend to show {path}?"
            if TV.find_violations({"text": text}):
                continue
            prob = Problem.LOSS_AVOIDANCE if e.direction == DOWN else Problem.VOLATILITY
            out.append(ResearchQuestion.make(text, "missed_loser" if e.direction == DOWN else "missed_winner", prob, created_real, evidence_through,
                                             f"enrichment among movers significant on unseen days (recurrence >= {self.p.promote_days} days)",
                                             "enrichment vanishes out of sample or the topic stays unknowable"))
            if len(out) >= n:
                break
        return out

    def to_json(self) -> dict[str, Any]:
        rows = []
        for e in self.entries.values():
            d = dataclasses.asdict(e)
            d["state"] = e.state.value
            rows.append(d)
        return {"day_index": self.day_index, "names": list(self.names), "entries": sorted(rows, key=lambda r: r["sig_id"])}

    @classmethod
    def from_json(cls, blob: Mapping[str, Any], p: Params | None = None) -> "ResearchStream":
        s = cls(blob["names"], p)
        s.day_index = int(blob["day_index"])
        for r in blob["entries"]:
            r = dict(r)
            r["state"] = ResearchState(r["state"])
            for key in ("features", "reasons", "paths", "history", "eras"):
                r[key] = tuple(tuple(x) for x in r[key])
            s.entries[r["sig_id"]] = StreamEntry(**r)
        return s


# ==================================================================================================================
# 8. streamed ledgers: filter cost, next-day paths (C67), tallies by era/type
# ==================================================================================================================
class FilterLedger:
    """Streaming version of learning.missed_winners.filter_tradeoff (which needs whole Weeks in memory): for every filter, the
    winners it removed against the losers it removed, accumulated day by day from the full cross-section before it is dropped."""

    def __init__(self):
        self.rows: dict[str, dict[str, float]] = {}

    def add_day(self, day: DayBook, p: Params) -> None:
        for o in day.obs:
            if not o.filters_hit or not math.isfinite(o.fwd):
                continue
            for f in o.filters_hit:
                r = self.rows.setdefault(f, {"removed": 0, "winners": 0, "win_sum": 0.0, "losers": 0, "loss_sum": 0.0, "days": 0, "last": ""})
                r["removed"] += 1
                if r["last"] != day.decided_at:
                    r["days"] += 1
                    r["last"] = day.decided_at
                if o.fwd >= p.move_thr:
                    r["winners"] += 1
                    r["win_sum"] += o.fwd
                elif o.fwd <= -p.loss_thr:
                    r["losers"] += 1
                    r["loss_sum"] += -o.fwd

    def review(self) -> pd.DataFrame:
        out = []
        for f, r in sorted(self.rows.items()):
            verdict = "COSTLY" if r["win_sum"] > r["loss_sum"] * 1.5 else "PAYS" if r["loss_sum"] > r["win_sum"] else "NEUTRAL"
            out.append({"filter": f, "removed": int(r["removed"]), "days": int(r["days"]), "winners_foregone": int(r["winners"]),
                        "gain_foregone": r["win_sum"], "losers_avoided": int(r["losers"]), "loss_avoided": r["loss_sum"], "verdict": verdict})
        return pd.DataFrame(out, columns=["filter", "removed", "days", "winners_foregone", "gain_foregone", "losers_avoided", "loss_avoided", "verdict"])


@dataclass(frozen=True)
class PathDay:
    """C67: what the 5-10% movers of one day did on the NEXT day, and how well their path could have been called beforehand."""
    n_band: int
    n_labelled: int
    distribution: Mapping[str, int]
    hits_view: int                          # pre-move history's most likely path equalled the realised path
    hits_majority: int                      # ...versus always guessing the historically commonest path
    hits_model: int                         # the system's own path_pred
    n_model: int
    mean_lift_realised: float | None        # mean posterior/base of the path that actually happened
    by_first_move: Mapping[int, Mapping[str, int]]


def analyse_paths(day: DayBook, view: LiftView, p: Params) -> PathDay:
    """Path lab for the whole band of movers (not only the exceptions). Every prediction reads the view of days that matured
    before this decision, so a hit is a genuine pre-move call."""
    band = [o for o in day.obs if math.isfinite(o.fwd) and abs(o.fwd) >= p.mover_lo]
    dist: Counter = Counter()
    by_dir: dict[int, Counter] = {UP: Counter(), DOWN: Counter()}
    hits_view = hits_major = hits_model = n_model = 0
    lifts = []
    bases = {c: view.base_rate(c) for c in PATHS}
    have_base = all(b is not None for b in bases.values())
    majority = max(PATHS, key=lambda c: (bases[c] or 0.0, c)) if have_base else None
    n_lab = 0
    for o in band:
        path = norm_path(o.next_path)
        if path is None:
            continue
        n_lab += 1
        dist[path] += 1
        by_dir[UP if o.fwd > 0 else DOWN][path] += 1
        if o.path_pred is not None and norm_path(o.path_pred) is not None:
            n_model += 1
            hits_model += int(norm_path(o.path_pred) == path)
        evs = {c: view.evidence(o.features, c) for c in PATHS}
        if all(e.enough for e in evs.values()):
            best = max(PATHS, key=lambda c: (evs[c].p, c))
            hits_view += int(best == path)
            hits_major += int(majority == path)
            lifts.append(evs[path].lift)
    return PathDay(len(band), n_lab, dict(dist), hits_view, hits_major, hits_model, n_model,
                   float(np.mean(lifts)) if lifts else None, {d: dict(c) for d, c in by_dir.items()})


class PathLedger:
    """Across days: does pre-move history call the next-day path better than always guessing the commonest one? Per-day rate
    differences feed a sign-flip test, so a lucky day cannot pass; the null (shuffled paths) must give edge ~ 0."""

    def __init__(self):
        self.days: list[tuple[int, int, int, int, int]] = []       # n_scored, hits_view, hits_majority, hits_model, n_model

    def add(self, pd_: PathDay) -> None:
        scored = pd_.hits_view + pd_.hits_majority
        if pd_.n_labelled and (scored > 0 or pd_.n_model):
            self.days.append((pd_.n_labelled, pd_.hits_view, pd_.hits_majority, pd_.hits_model, pd_.n_model))

    def edge(self, seed: int = 0) -> dict[str, Any]:
        if not self.days:
            return {"days": 0, "edge": float("nan"), "p": 1.0, "verdict": "NO_DATA"}
        arr = np.array(self.days, dtype=float)
        diff = (arr[:, 1] - arr[:, 2]) / np.maximum(arr[:, 0], 1)
        pv = base.sign_flip_p(diff, n=1000, seed=seed)
        lo, hi = base.boot_ci(diff, n=1000, seed=seed)
        model_rate = float(arr[:, 3].sum() / arr[:, 4].sum()) if arr[:, 4].sum() > 0 else float("nan")
        return {"days": len(arr), "edge": float(diff.mean()), "p": pv, "ci": (lo, hi), "view_rate": float(arr[:, 1].sum() / arr[:, 0].sum()),
                "majority_rate": float(arr[:, 2].sum() / arr[:, 0].sum()), "model_rate": model_rate,
                "verdict": "EDGE" if pv < 0.05 and diff.mean() > 0 else "NO_EDGE"}


class Tally:
    """Every classified move, counted by (era, kind, direction, capture, knowability) with total move size and book loss, so the
    per-era and per-type breakdowns cover the whole stream even though only exception rows are ever retained."""

    def __init__(self):
        self.cells: dict[tuple, list[float]] = {}

    def add(self, era: str, kind: str, direction: int, capture: Capture, know: Knowability, fwd: float, loss: float) -> None:
        c = self.cells.setdefault((era, kind, direction, capture.value, know.value), [0.0, 0.0, 0.0])
        c[0] += 1
        c[1] += abs(fwd)
        c[2] += loss

    def frame(self) -> pd.DataFrame:
        rows = [dict(zip(("era", "kind", "direction", "capture", "knowability"), k), n=v[0], sum_abs_move=v[1], loss_cost=v[2])
                for k, v in sorted(self.cells.items())]
        return pd.DataFrame(rows, columns=["era", "kind", "direction", "capture", "knowability", "n", "sum_abs_move", "loss_cost"])

    def breakdown(self, by: str = "era") -> pd.DataFrame:
        """Per `by` (era or kind) and direction: how many moves were predicted / partial / missed and what share of the misses
        were learnable versus unknowable."""
        df = self.frame()
        if df.empty:
            return pd.DataFrame(columns=[by, "direction", "moves", "predicted", "partial", "missed", "catch_rate", "learnable_miss", "unknowable_miss"])
        df["family"] = df.capture.map(lambda c: FAMILY[Capture(c)])
        out = []
        for (key, d), g in df.groupby([by, "direction"]):
            def n(fam, mask=None):
                s = g[g.family == fam]
                return float(s.n.sum()) if mask is None else float(s[s.knowability.isin(mask)].n.sum())
            moves = n("predicted") + n("partial") + n("missed")
            out.append({by: key, "direction": int(d), "moves": moves, "predicted": n("predicted"), "partial": n("partial"), "missed": n("missed"),
                        "catch_rate": (n("predicted") + 0.5 * n("partial")) / moves if moves else float("nan"),
                        "learnable_miss": n("missed", [k.value for k in LEARNABLE]), "unknowable_miss": n("missed", [k.value for k in UNKNOWABLE])})
        return pd.DataFrame(out)


# ==================================================================================================================
# 9. state, daily report and the public entry: submit() then step()
# ==================================================================================================================
@dataclass(frozen=True)
class DailyReport:
    decided_at: str
    snapshot: DaySnapshot
    captures: Mapping[int, Mapping[str, int]]          # direction -> capture value -> count
    knowability: Mapping[int, Mapping[str, int]]       # direction -> knowability of the uncaptured moves
    reasons: Mapping[str, int]
    n_moves: int
    n_learnable_misses: int
    n_unknowable_misses: int
    near_misses: tuple[NearMiss, ...]
    top: tuple[Opportunity, ...]
    audits: tuple[KnowabilityReport, ...]              # section-8 reports for the top opportunities (research side only)
    loss_cost: float
    loss_unavoidable: float
    stable_separators: tuple[str, ...]
    stream_size: int
    stream_states: Mapping[str, int]
    path: PathDay | None
    engagement: Mapping[int, tuple[int, int, int, int]] = field(default_factory=dict)   # direction -> (engaged movers, movers, engaged others, others)
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def engagement_lift(self, direction: int) -> float:
        """How much more often the system engaged (owned/flagged a winner, flagged-to-avoid a loser) with the movers of this side than
        with unmoved names. 1.0 = chance. Each cell gets one pseudo-observation at the pooled engagement rate, so an empty cell is
        shrunk toward 'no difference' (never toward a huge ratio) and nobody engaging at all is exactly 1.0."""
        e = self.engagement.get(direction)
        if e is None or e[1] == 0 or e[3] == 0:
            return float("nan")
        pooled = (e[0] + e[2]) / (e[1] + e[3])
        if pooled == 0.0:
            return 1.0
        return ((e[0] + pooled) / (e[1] + 1.0)) / ((e[2] + pooled) / (e[3] + 1.0))

    @property
    def unknowable_share(self) -> float:
        miss = sum(v for d in self.captures.values() for k, v in d.items() if FAMILY[Capture(k)] in ("missed", "partial"))
        return self.n_unknowable_misses / miss if miss else float("nan")

    def catch_rate(self, direction: int) -> float:
        c = self.captures.get(direction, {})
        pred = c.get(Capture.PREDICTED.value, 0)
        part = sum(v for k, v in c.items() if FAMILY[Capture(k)] == "partial")
        miss = sum(v for k, v in c.items() if FAMILY[Capture(k)] == "missed")
        n = pred + part + miss
        return (pred + 0.5 * part) / n if n else float("nan")


@dataclass
class MissedState:
    """Everything the engine carries between days. The checkpoint (save_checkpoint) preserves all of it except pending days,
    which the source re-submits on resume."""
    p: Params
    names: tuple[str, ...]
    lift: LiftStore
    sigs: SigStats
    stream: ResearchStream
    novelty: NoveltyIndex
    near_up: NearMissLedger
    near_down: NearMissLedger
    filters: FilterLedger
    paths: PathLedger
    tally: Tally
    patterns: list[PatternSig] = field(default_factory=list)
    pending: list[DayBook] = field(default_factory=list)
    done: dict[str, str] = field(default_factory=dict)             # decided_at -> matured_at of processed days
    reports: list[DailyReport] = field(default_factory=list)
    exceptions: dict[str, tuple[MoveObs, ...]] = field(default_factory=dict)
    snapshots: dict[str, DaySnapshot] = field(default_factory=dict)
    keep_reports: int = 400


def new_state(feature_names: Sequence[str], p: Params | None = None, patterns: Sequence[PatternSig] = ()) -> MissedState:
    p = p or Params()
    errs = p.validate()
    if errs:
        raise ValueError("invalid Params: " + "; ".join(errs))
    names = tuple(feature_names)
    if not names:
        raise ValueError("at least one feature name is required")
    return MissedState(p, names, LiftStore(names, p), SigStats(names, p), ResearchStream(names, p), NoveltyIndex(len(names), p.novelty_cap),
                       NearMissLedger(names), NearMissLedger(names), FilterLedger(), PathLedger(), Tally(), list(patterns))


def submit(state: MissedState, day: DayBook) -> None:
    """Queue a closed day. Refuses malformed days and repeats; the day is analysed only when step() is called with a `now`
    strictly after its outcomes matured."""
    errs = day.validate()
    if errs:
        raise ValueError("invalid day: " + "; ".join(errs[:5]))
    if tuple(day.feature_names) != state.names:
        raise ValueError("day's feature names differ from the engine's")
    if day.decided_at in state.done or any(d.decided_at == day.decided_at for d in state.pending):
        raise ValueError(f"day {day.decided_at} already submitted")
    state.pending.append(day)


def _process(state: MissedState, day: DayBook, now) -> DailyReport:
    p = state.p
    require_past(day.matured_at, now, f"day {day.decided_at}")
    snap = day.snapshot(p)
    view = state.lift.view(day.decided_at)            # history that matured before this decision: never the day itself
    exc = select_exceptions(day, p)
    movers = [o for o in exc if o.move_dir(p) != 0]
    audits = {o.cid: assess(o, view, p) for o in movers}
    rejections = explain_missed_winners([o for o in movers if o.move_dir(p) == UP], day, p)
    losses = LossAnalyzer(p).explain_all(movers)
    for ledger, direction in ((state.near_up, UP), (state.near_down, DOWN)):
        ledger.add(paired_differences(exc, state.names, day.k, p, direction))
    stable = tuple(state.near_up.stable())
    near = find_near_misses(movers, day.k, p, audits, stable)
    state.sigs.add_day(day)
    opps = build_opportunities(movers, day, p, audits, rejections, losses, state.patterns, state.novelty, state.sigs)
    for x in opps:
        o = next(o for o in movers if o.cid == x.cid)
        if x.score > 0:
            state.novelty.add(feature_vector(o, state.names))
    state.stream.update(opps, state.sigs)
    state.filters.add_day(day, p)
    path_day = analyse_paths(day, view, p)
    state.paths.add(path_day)
    caps: dict[int, Counter] = {UP: Counter(), DOWN: Counter()}
    know: dict[int, Counter] = {UP: Counter(), DOWN: Counter()}
    reasons: Counter = Counter()
    n_learn = n_unk = 0
    loss_total = loss_unavoid = 0.0
    for o in movers:
        cr = classify_capture(o, p, day.k)
        d = o.move_dir(p)
        caps[d][cr.capture.value] += 1
        rep = audits[o.cid]
        lc = loss_incurred(o, day.k)
        state.tally.add(day.era, o.kind, d, cr.capture, rep.knowability, o.fwd, lc)
        loss_total += lc
        if o.cid in losses and losses[o.cid].primary == LossReason.GAP_UNAVOIDABLE:
            loss_unavoid += lc
        if cr.family in ("missed", "partial"):
            know[d][rep.knowability.value] += 1
            if cr.family == "missed":
                n_learn += int(rep.learnable)
                n_unk += int(rep.knowability in UNKNOWABLE)
            reasons[losses[o.cid].primary.value if o.cid in losses else rejections[o.cid].primary.value if o.cid in rejections else "UNKNOWN"] += 1
    eng = {UP: [0, 0, 0, 0], DOWN: [0, 0, 0, 0]}
    for o in day.obs:
        if not math.isfinite(o.fwd):
            continue
        d = o.move_dir(p)
        engaged = {UP: bool(o.picked or (o.pred_prob is not None and o.pred_prob >= p.flag_gate and o.pred_dir != DOWN)), DOWN: avoid_intent(o)}
        for side in (UP, DOWN):
            if d == side:
                eng[side][0] += int(engaged[side])
                eng[side][1] += 1
            elif d == 0:
                eng[side][2] += int(engaged[side])
                eng[side][3] += 1
    top = tuple(opps[:20])
    report = DailyReport(day.decided_at, snap, {d: dict(c) for d, c in caps.items()}, {d: dict(c) for d, c in know.items()}, dict(reasons),
                         len(movers), n_learn, n_unk, tuple(near), top, tuple(audits[x.cid] for x in top), float(loss_total), float(loss_unavoid),
                         stable, len(state.stream), dict(Counter(e.state.value for e in state.stream.entries.values())), path_day,
                         {d: tuple(v) for d, v in eng.items()})
    state.lift.add_day(day)                            # only now may today's outcome become history for later days
    state.done[day.decided_at] = day.matured_at
    state.exceptions[day.decided_at] = exc
    state.snapshots[day.decided_at] = snap
    state.reports.append(report)
    if len(state.reports) > state.keep_reports:
        drop = state.reports.pop(0).decided_at
        state.exceptions.pop(drop, None)
    return report


def step(state: MissedState, now) -> list[DailyReport]:
    """Public entry. Process every queued day whose outcomes matured STRICTLY before `now`, oldest first; days that have not yet
    matured stay queued (their outcome is still the future). Returns one DailyReport per processed day."""
    ready = sorted([d for d in state.pending if as_date(d.matured_at) < as_date(now)], key=lambda d: d.decided_at)
    out = []
    for day in ready:
        out.append(_process(state, day, now))
        state.pending.remove(day)
    return out


def default_now(day: DayBook):
    """The first instant the day's outcomes count as history: the day after they matured."""
    return as_date(day.matured_at) + dt.timedelta(days=1)


def sweep(state: MissedState, source: Iterable[DayBook], now_fn=default_now, checkpoint: str | os.PathLike | None = None, every: int = 25):
    """Resumable long sweep (C67): a generator the wave-2 loop can schedule forever. Days already in state.done are skipped, so
    re-running after a kill continues where it stopped; a checkpoint is written every `every` processed days and at the end.
    Memory stays flat because each day is reduced to a snapshot plus exception rows before the next is read."""
    n = 0
    for day in source:
        if day.decided_at in state.done:
            continue
        submit(state, day)
        for rep in step(state, now_fn(day)):
            n += 1
            yield rep
        if checkpoint is not None and n and n % every == 0:
            save_checkpoint(state, checkpoint)
    if checkpoint is not None and n:
        save_checkpoint(state, checkpoint)


def replay(days: Sequence[DayBook], feature_names: Sequence[str], p: Params | None = None, patterns: Sequence[PatternSig] = ()
           ) -> tuple[MissedState, list[DailyReport]]:
    state = new_state(feature_names, p, patterns)
    return state, list(sweep(state, days))


# ==================================================================================================================
# 10. checkpoint / resume
# ==================================================================================================================
def _pattern_json(pat: PatternSig) -> dict[str, Any]:
    return {"pattern_id": pat.pattern_id, "direction": pat.direction, "conds": {f: list(iv) for f, iv in pat.conds.items()}}


def save_checkpoint(state: MissedState, path: str | os.PathLike) -> None:
    """Atomic write (temp file then os.replace) so a kill mid-save never leaves a half-written checkpoint. Pending days and the
    retained report objects are not saved: the source re-submits pending days, and reports are derived output."""
    keys = sorted(state.lift._days)
    F, nb = len(state.names), state.p.n_bins
    cnt = np.stack([state.lift._days[k][1] for k in keys]) if keys else np.zeros((0, F, nb, N_COLS))
    tot = np.stack([state.lift._days[k][2] for k in keys]) if keys else np.zeros((0, N_COLS))
    meta = {
        "version": 1, "params": dataclasses.asdict(state.p), "names": list(state.names), "done": state.done,
        "lift_keys": [[k, state.lift._days[k][0]] for k in keys],
        "sigs": {"rows": {str(c): r for c, r in state.sigs.rows.items()}, "n_days": state.sigs.n_days, "totals": state.sigs.totals.tolist()},
        "stream": state.stream.to_json(),
        "near": {nm: {a: getattr(led, a).tolist() for a in ("n", "s", "ss", "pos_days", "days")}
                 for nm, led in (("up", state.near_up), ("down", state.near_down))},
        "filters": state.filters.rows, "paths": state.paths.days,
        "tally": [[list(k), v] for k, v in sorted(state.tally.cells.items())],
        "patterns": [_pattern_json(x) for x in state.patterns],
        "snapshots": {k: dataclasses.asdict(v) for k, v in state.snapshots.items()},
    }
    target = os.fspath(path)
    tmp = target + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)), lift_cnt=cnt, lift_tot=tot, novelty=state.novelty.buf)
    os.replace(tmp, target)


def load_checkpoint(path: str | os.PathLike) -> MissedState:
    with np.load(os.fspath(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        cnt, tot, nov = z["lift_cnt"], z["lift_tot"], z["novelty"]
    if meta.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version {meta.get('version')}")
    prm = dict(meta["params"])
    prm["weights"] = dict(prm["weights"])
    p = Params(**prm)
    pats = [PatternSig(x["pattern_id"], x["direction"], {f: tuple(iv) for f, iv in x["conds"].items()}) for x in meta["patterns"]]
    st = new_state(meta["names"], p, pats)
    for i, (k, matured) in enumerate(meta["lift_keys"]):
        st.lift._days[k] = (matured, cnt[i].copy(), tot[i].copy())
    st.sigs.rows = {int(c): list(r) for c, r in meta["sigs"]["rows"].items()}
    st.sigs.n_days, st.sigs.totals = int(meta["sigs"]["n_days"]), np.array(meta["sigs"]["totals"], dtype=np.float64)
    st.stream = ResearchStream.from_json(meta["stream"], p)
    st.novelty.buf = nov.copy()
    for nm, led in (("up", st.near_up), ("down", st.near_down)):
        for a, v in meta["near"][nm].items():
            setattr(led, a, np.array(v, dtype=np.float64))
    st.filters.rows = {f: dict(r) for f, r in meta["filters"].items()}
    st.paths.days = [tuple(d) for d in meta["paths"]]
    st.tally.cells = {tuple(k): list(v) for k, v in meta["tally"]}
    st.done = dict(meta["done"])
    st.snapshots = {k: DaySnapshot(**{**v, "context": dict(v["context"])}) for k, v in meta["snapshots"].items()}
    return st


# ==================================================================================================================
# 11. controls: symmetry, shuffled-outcome null, predictable share
# ==================================================================================================================
def symmetry(reports: Sequence[DailyReport], seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Section 12 in miniature: is the system as good at engaging with big losers (flagging them to avoid) as with big winners
    (owning or flagging them)? The comparison is on engagement LIFT over unmoved names, not raw catch rate: a long-only book owns
    few names and avoids most by default, so raw rates are not comparable across sides. Paired per-day log-lift differences go
    through a sign-flip test in both directions, with a bootstrap interval."""
    ups, dns, cu, cd = [], [], [], []
    for r in reports:
        u, d = r.engagement_lift(UP), r.engagement_lift(DOWN)
        if math.isfinite(u) and math.isfinite(d):
            ups.append(math.log(u))
            dns.append(math.log(d))
            cu.append(r.catch_rate(UP))
            cd.append(r.catch_rate(DOWN))
    if len(ups) < 5:
        return {"days": len(ups), "verdict": "INSUFFICIENT", "diff": float("nan")}
    diff = np.array(ups) - np.array(dns)
    p_win = base.sign_flip_p(diff, n=2000, seed=seed)
    p_lose = base.sign_flip_p(-diff, n=2000, seed=seed + 1)
    lo, hi = base.boot_ci(diff, n=2000, seed=seed)
    verdict = "WINNER_BIASED" if p_win < alpha else "LOSER_BIASED" if p_lose < alpha else "SYMMETRIC"
    return {"days": len(ups), "winner_log_lift": float(np.mean(ups)), "loser_log_lift": float(np.mean(dns)), "diff": float(diff.mean()),
            "ci": (lo, hi), "p_winner_better": p_win, "p_loser_better": p_lose, "verdict": verdict,
            "winner_catch": float(np.nanmean(cu)), "loser_catch": float(np.nanmean(cd))}


def predictable_share(state: MissedState, direction: int | None = None) -> float:
    """Among moves that were not fully captured, the share the auditor called PREDICTABLE. The shuffled null must not raise it."""
    df = state.tally.frame()
    if df.empty:
        return float("nan")
    df = df[df.capture.map(lambda c: FAMILY[Capture(c)] in ("missed", "partial"))]
    if direction is not None:
        df = df[df.direction == direction]
    tot = float(df.n.sum())
    return float(df[df.knowability == Knowability.PREDICTABLE.value].n.sum() / tot) if tot else float("nan")


def shuffle_outcomes(day: DayBook, rng: np.random.Generator) -> DayBook:
    """Permute outcomes (return, gap, realised and next-day path together) across names within the day, keeping features and the
    decision trail with their names. Every pre-move relationship to the outcome is destroyed; the marginal distribution is not."""
    n = len(day.obs)
    perm = rng.permutation(n)
    src = day.obs
    obs = tuple(dataclasses.replace(o, fwd=src[j].fwd, gap=src[j].gap, realised=src[j].realised if o.picked else None,
                                    next_path=src[j].next_path, next_ret=src[j].next_ret)
                for o, j in zip(src, perm.tolist()))
    return dataclasses.replace(day, obs=obs)


def null_control(days: Sequence[DayBook], feature_names: Sequence[str], p: Params | None = None, seed: int = 0, reps: int = 1
                 ) -> dict[str, Any]:
    """Run the whole engine on the real days and on `reps` outcome-shuffled copies. The share of misses called PREDICTABLE on
    shuffled outcomes is the false-alarm rate of the auditor; it must sit well under the real share for any claim to stand."""
    real_state, real_reports = replay(days, feature_names, p)
    rng = np.random.default_rng(seed)
    nulls = []
    path_edges = []
    for _ in range(reps):
        st, _ = replay([shuffle_outcomes(d, rng) for d in days], feature_names, p)
        nulls.append(predictable_share(st))
        path_edges.append(st.paths.edge(seed)["edge"])
    real = predictable_share(real_state)
    null = float(np.nanmean(nulls)) if nulls else float("nan")
    ok = math.isfinite(real) and math.isfinite(null) and null <= max(0.05, 0.5 * real)
    return {"real_predictable_share": real, "null_predictable_share": null, "null_path_edge": float(np.nanmean(path_edges)) if path_edges else float("nan"),
            "real_path_edge": real_state.paths.edge(seed)["edge"], "clean": bool(ok) if math.isfinite(real) else None, "reps": reps,
            "days": len(real_reports)}


# ==================================================================================================================
# 12. release to the record, reports, summaries
# ==================================================================================================================
def make_provenance(created_real: str, matured_at: str, seed: int | None = None, run_id: str = "") -> Provenance:
    return Provenance(created_real=created_real, learned_at=matured_at, code_hash=current_code_hash(), outcomes_seen_through=matured_at,
                      experiment_id="R06_missed_opportunities", run_id=run_id, seed=seed)


def matured_records(report: DailyReport, created_real: str, seed: int | None = None) -> list[MaturedRecord]:
    """The day's audit reports as MATURED_RESEARCH records, dated at the day their outcomes matured. This is the ONLY form in
    which anything computed here may travel toward a decision: through record.gate(now) (or ReleaseGate.release)."""
    prov = make_provenance(created_real, report.snapshot.matured_at, seed)
    return [to_matured_record(a, prov, report.snapshot.matured_at) for a in report.audits]


def stream_questions(state: MissedState, created_real: str, n: int = 10) -> list[ResearchQuestion]:
    through = state.lift.through() or created_real
    return state.stream.questions(created_real, through, n)


def summary(state: MissedState) -> dict[str, Any]:
    """One dictionary for dashboards and the scorecard: coverage, unknowable share, stream size and states, path edge."""
    df = state.tally.frame()
    moves = float(df.n.sum()) if not df.empty else 0.0
    unc = df[df.capture.map(lambda c: FAMILY[Capture(c)] in ("missed", "partial"))] if not df.empty else df
    unk = float(unc[unc.knowability.isin([k.value for k in UNKNOWABLE])].n.sum()) if not unc.empty else 0.0
    return {"days": len(state.done), "moves": moves, "uncaptured": float(unc.n.sum()) if not unc.empty else 0.0,
            "unknowable_share": unk / float(unc.n.sum()) if not unc.empty and unc.n.sum() else float("nan"),
            "predictable_share": predictable_share(state), "stream": len(state.stream),
            "stream_states": dict(Counter(e.state.value for e in state.stream.entries.values())),
            "stable_separators_up": state.near_up.stable(), "stable_separators_down": state.near_down.stable(),
            "path_edge": state.paths.edge(), "loss_cost": float(df.loss_cost.sum()) if not df.empty else 0.0}


def render_report(rep: DailyReport, weights: Mapping[str, float] | None = None) -> str:
    """Plain-text daily report: the eight section-6 questions in order, then the ranked stream."""
    w = weights or Params().weights
    lines = [f"MISSED OPPORTUNITIES {rep.decided_at}  ({rep.n_moves} large moves; {rep.snapshot.n_up} up, {rep.snapshot.n_down} down of {rep.snapshot.n_names})"]
    for d, name in ((UP, "winners"), (DOWN, "losers")):
        c = rep.captures.get(d, {})
        lines.append(f"  {name}: " + (", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "none") + f"  catch_rate={rep.catch_rate(d):.2f}")
        k = rep.knowability.get(d, {})
        lines.append(f"    uncaptured by knowability: " + (", ".join(f"{a}={b}" for a, b in sorted(k.items())) or "none"))
    lines.append(f"  learnable misses={rep.n_learnable_misses}  unknowable misses={rep.n_unknowable_misses}  unknowable_share={rep.unknowable_share:.2f}")
    lines.append(f"  loss cost={rep.loss_cost:.4f} of which unavoidable by a stop={rep.loss_unavoidable:.4f}")
    lines.append("  reasons: " + (", ".join(f"{k}={v}" for k, v in sorted(rep.reasons.items())) or "none"))
    inf = [n for n in rep.near_misses if n.informative]
    lines.append(f"  near misses={len(rep.near_misses)} informative={len(inf)} stable separators={list(rep.stable_separators)}")
    for x in rep.top[:5]:
        lines.append(f"    #{x.cid[:6]} {'loss' if x.is_loss else 'gain'} {x.fwd:+.3f} {x.capture} reason={x.reason} " + explain_rank(x, w))
    lines.append(f"  stream: {rep.stream_size} topics " + ", ".join(f"{k}={v}" for k, v in sorted(rep.stream_states.items())))
    if rep.path is not None and rep.path.n_labelled:
        lines.append(f"  next-day paths of {rep.path.n_labelled} band movers: " + ", ".join(f"{k}={v}" for k, v in sorted(rep.path.distribution.items())))
    return NL.join(lines)


# ==================================================================================================================
# 13. input audit: a day that leaks its own outcome must never reach the engine
# ==================================================================================================================
@dataclass(frozen=True)
class DayAudit:
    n: int
    issues: tuple[str, ...]
    leaks: tuple[tuple[str, float], ...]        # (feature, rank correlation with the outcome) above the suspicion level
    ok: bool


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float("nan")
    rx, ry = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def audit_day(day: DayBook, p: Params, min_names: int = 200) -> DayAudit:
    """Structural and leak checks on one closed day. A feature whose cross-sectional rank correlation with the outcome exceeds
    engine.missed_winners.SUSPICIOUS_IC is treated as leakage (it could not have been known at the close), not as skill; the
    check needs a cross-section big enough that a correlation that large cannot be chance."""
    issues: list[str] = []
    leaks: list[tuple[str, float]] = []
    f = day.fwd()
    n = len(day.obs)
    if n == 0:
        return DayAudit(0, ("empty day",), (), False)
    miss = float(np.mean(~np.isfinite(f)))
    if miss > 0.2:
        issues.append(f"{miss:.0%} of outcomes are missing")
    if int(np.sum(np.abs(f[np.isfinite(f)]) > 1.0)):
        issues.append("outcomes beyond +-100% (a bad print or a split)")
    X = day.matrix()
    for j, name in enumerate(day.feature_names):
        col = X[:, j]
        if np.isfinite(col).sum() and np.nanstd(col) == 0:
            issues.append(f"feature {name} is constant")
        if n >= min_names:
            c = _rank_corr(col.astype(np.float64), f)
            if math.isfinite(c) and abs(c) >= base.SUSPICIOUS_IC:
                leaks.append((name, c))
    ranks = [o.rank for o in day.obs if o.rank is not None]
    if len(ranks) != len(set(ranks)):
        issues.append("duplicate ranks: ties would be broken by input order")
    if sum(o.picked for o in day.obs) > 3 * day.k:
        issues.append("far more picks than the cut allows")
    if leaks:
        issues.append("leak suspected: " + ", ".join(f"{a} (rho={c:+.2f})" for a, c in leaks))
    return DayAudit(n, tuple(issues), tuple(leaks), not leaks)


def submit_checked(state: MissedState, day: DayBook) -> DayAudit:
    """submit() behind the leak audit. A day with a suspected leak is refused (FirewallBreach), never analysed with a warning."""
    audit = audit_day(day, state.p)
    if audit.leaks:
        raise FirewallBreach(f"day {day.decided_at} refused: " + "; ".join(audit.issues))
    submit(state, day)
    return audit


# ==================================================================================================================
# 14. regimes, stream health and topic reports
# ==================================================================================================================
def regime_of(snap: DaySnapshot, dispersion_median: float | None) -> str:
    tone = "down" if snap.market_ret < -0.01 else "up" if snap.market_ret > 0.01 else "flat"
    if dispersion_median is None or not math.isfinite(snap.dispersion):
        return tone
    return tone + ("_wide" if snap.dispersion > dispersion_median else "_tight")


def regime_table(state: MissedState) -> pd.DataFrame:
    """Catch rate for winners and losers by day regime (market tone x dispersion versus its own running median, computed only
    from days up to the one being labelled). Misses concentrated in one regime are a finding; uniform misses are not."""
    disp: list[float] = []
    rows = []
    by_day = {r.decided_at: r for r in state.reports}
    for day in sorted(by_day):
        rep = by_day[day]
        med = float(np.median(disp)) if len(disp) >= 5 else None
        if math.isfinite(rep.snapshot.dispersion):
            disp.append(rep.snapshot.dispersion)
        rows.append({"regime": regime_of(rep.snapshot, med), "up_catch": rep.catch_rate(UP), "down_catch": rep.catch_rate(DOWN),
                     "moves": rep.n_moves, "loss_cost": rep.loss_cost})
    if not rows:
        return pd.DataFrame(columns=["regime", "days", "moves", "up_catch", "down_catch", "loss_cost"])
    df = pd.DataFrame(rows)
    g = df.groupby("regime")
    return pd.DataFrame({"days": g.size(), "moves": g.moves.sum(), "up_catch": g.up_catch.mean(), "down_catch": g.down_catch.mean(),
                         "loss_cost": g.loss_cost.sum()}).reset_index()


def stream_health(state: MissedState) -> dict[str, Any]:
    """Is the permanent stream still a useful research queue? Size and state mix, staleness, how much of it is cancelled as
    unknowable, and feature diversity (a stream that is 90% one feature is a monoculture, not a search)."""
    es = list(state.stream.entries.values())
    if not es:
        return {"topics": 0, "verdict": "EMPTY"}
    states = Counter(e.state.value for e in es)
    use = Counter(f for e in es for f, _ in e.features)
    tot = sum(use.values())
    probs = np.array([c / tot for c in use.values()]) if tot else np.array([1.0])
    entropy = float(-(probs * np.log(probs)).sum())
    top_share = float(probs.max())
    stale = sum(1 for e in es if state.stream.day_index - e.last_day >= state.p.dormant_after)
    live = sum(1 for e in es if e.state in (ResearchState.QUEUED, ResearchState.EXPLORING, ResearchState.PROMISING))
    cancelled = states.get(ResearchState.CANCELLED.value, 0)
    verdict = "MONOCULTURE" if top_share > 0.6 and len(use) > 3 else "STALE" if stale > 0.7 * len(es) else "HEALTHY"
    return {"topics": len(es), "states": dict(states), "live": live, "stale": stale, "cancelled_share": cancelled / len(es),
            "feature_entropy": entropy, "top_feature_share": top_share, "up_topics": sum(e.direction == UP for e in es),
            "down_topics": sum(e.direction == DOWN for e in es), "verdict": verdict}


def topic_report(state: MissedState, sig_id: str) -> dict[str, Any]:
    """Everything known about one stream topic: how often, how large, how learnable, its enrichment among movers against chance,
    how often it appeared with each rejection reason and next-day path."""
    e = state.stream.entries.get(sig_id)
    if e is None:
        raise KeyError(f"no stream topic {sig_id}")
    ratio, pv, n_occ = state.sigs.enrichment(e.code, e.direction)
    days, expected = state.sigs.spread(e.code)
    return {"sig_id": e.sig_id, "direction": e.direction, "features": e.features, "state": e.state.value, "observations": e.n_obs, "days": e.n_days,
            "mean_score": e.mean_score, "mean_abs_move": e.sum_abs_move / max(e.n_obs, 1), "loss_cost": e.loss_cost,
            "learnable_share": e.learnable / max(e.n_obs, 1), "unknowable_share": e.unknowable_share, "enrichment": ratio, "p_enrichment": pv,
            "occurrences_all_names": n_occ, "days_carried_by_a_mover": days, "days_expected_if_scattered": expected,
            "reasons": dict(e.reasons), "paths": dict(e.paths), "history": e.history}


def path_topics(state: MissedState, n: int = 10, min_obs: int = 20, alpha: float = 0.01) -> list[dict[str, Any]]:
    """C67: stream topics whose movers keep doing the same thing the next day (continue, reverse, consolidate, expand) far more
    often than movers in general. The reference rate is the store-wide path frequency; the test is a one-sided binomial."""
    view = state.lift.view(dt.date.max)
    out = []
    for e in state.stream.entries.values():
        tot = sum(c for _, c in e.paths)
        if tot < min_obs:
            continue
        path, cnt = max(e.paths, key=lambda t: (t[1], t[0]))
        b = view.base_rate(path)
        if b is None or not 0.0 < b < 1.0:
            continue
        pv = float(sps.binom.sf(cnt - 1, tot, b))
        if pv < alpha and cnt / tot > b:
            out.append({"sig_id": e.sig_id, "features": e.features, "direction": e.direction, "path": path, "share": cnt / tot, "base": b,
                        "lift": cnt / tot / b, "n": tot, "p": pv})
    return sorted(out, key=lambda r: (r["p"], r["sig_id"]))[:n]


def opportunity_frame(opps: Sequence[Opportunity]) -> pd.DataFrame:
    rows = [{"cid": x.cid, "decided_at": x.decided_at, "direction": x.direction, "capture": x.capture.value, "knowability": x.knowability.value,
             "fwd": x.fwd, "score": x.score, "reason": x.reason, "effect": x.effect.value, "loss_cost": x.loss_cost, "era": x.era, "kind": x.kind,
             "path": x.path, **{c: x.criteria[c] for c in CRITERIA}} for x in opps]
    return pd.DataFrame(rows, columns=["cid", "decided_at", "direction", "capture", "knowability", "fwd", "score", "reason", "effect", "loss_cost",
                                       "era", "kind", "path", *CRITERIA])


def criteria_redundancy(opps: Sequence[Opportunity], cutoff: float = 0.9) -> list[tuple[str, str, float]]:
    """Criteria pairs whose values are almost the same across the day's opportunities: two of the nine saying one thing means
    the ranking counts that thing twice."""
    if len(opps) < 8:
        return []
    M = np.array([[x.criteria[c] for c in CRITERIA] for x in opps])
    out = []
    for i in range(len(CRITERIA)):
        for j in range(i + 1, len(CRITERIA)):
            if M[:, i].std() < 1e-9 or M[:, j].std() < 1e-9:
                continue
            r = float(np.corrcoef(M[:, i], M[:, j])[0, 1])
            if abs(r) >= cutoff:
                out.append((CRITERIA[i], CRITERIA[j], r))
    return out


def priority_audit(reports: Sequence[DailyReport], top: int = 10) -> dict[str, Any]:
    """Section-5 loss priority: do losers get their share of the top of the ranking? Compares the loser share among the day's
    top-ranked opportunities with the loser share among all uncaptured moves. Far below parity means losses are being under-ranked."""
    top_loss = top_all = mv_loss = mv_all = 0
    for r in reports:
        t = r.top[:top]
        top_all += len(t)
        top_loss += sum(x.is_loss for x in t)
        for d, c in r.captures.items():
            n = sum(v for k, v in c.items() if FAMILY[Capture(k)] in ("missed", "partial"))
            mv_all += n
            mv_loss += n if d == DOWN else 0
    if top_all == 0 or mv_all == 0:
        return {"verdict": "NO_DATA"}
    ts, ms = top_loss / top_all, mv_loss / mv_all
    return {"top_loser_share": ts, "uncaptured_loser_share": ms, "ratio": ts / ms if ms else float("nan"),
            "verdict": "UNDER_RANKED" if ms > 0 and ts / ms < 0.8 else "OK"}


# ==================================================================================================================
# 15. bridge to the base module's distinction search (missed movers vs same-rank controls)
# ==================================================================================================================
def retained_weeks(state: MissedState) -> list[MW.Week]:
    """The exception rows kept in state, as base-module Weeks, oldest first. Only processed days exist here, so every week's
    outcomes matured before the engine ever saw it."""
    weeks = []
    for day in sorted(state.exceptions):
        cands = tuple(to_candidate(o) for o in state.exceptions[day])
        if not cands:
            continue
        weeks.append(MW.Week(day, day, state.done[day], cands, k=state.p.k, thr=state.p.move_thr))
    return weeks


def discover_distinctions(state: MissedState, now, seed: int = 0, params: MW.MissedParams | None = None) -> list[MW.Distinction]:
    """Search the retained days for rules separating missed winners from same-rank false positives, with the base module's
    family-wise permutation test. Returns [] when the cohorts are too small: no guess is made. `now` is enforced by cohort_frame."""
    prm = params or MW.MissedParams()
    df = MW.cohort_frame(retained_weeks(state), prm, now)
    return MW.find_distinctions(df, prm, seed)


# ==================================================================================================================
# 16. consistency across eras, research targets, and a narrative for one move
# ==================================================================================================================
def era_consistency(state: MissedState, min_obs: int = 5) -> list[dict[str, Any]]:
    """Which stream topics keep recurring in EVERY era seen so far (canon: consistent means every year, not one lucky stretch)?
    A topic carried by a single era is reported with share < 1 and is never a candidate for promotion on this evidence alone."""
    all_eras = sorted({era for e in state.stream.entries.values() for era, _ in e.eras if era})
    out = []
    for e in state.stream.entries.values():
        if e.n_obs < min_obs:
            continue
        per = {era: c for era, c in e.eras if era}
        share = len(per) / len(all_eras) if all_eras else float("nan")
        top = max(per.values()) / sum(per.values()) if per else float("nan")
        out.append({"sig_id": e.sig_id, "features": e.features, "direction": e.direction, "eras_seen": len(per), "eras_total": len(all_eras),
                    "era_share": share, "max_era_concentration": top, "state": e.state.value,
                    "consistent": bool(all_eras) and len(per) == len(all_eras) and top <= 0.8})
    return sorted(out, key=lambda r: (-(r["era_share"] if math.isfinite(r["era_share"]) else 0.0), r["max_era_concentration"], r["sig_id"]))


def cross_era_topics(state: MissedState, min_eras: int = 2) -> list[StreamEntry]:
    return [state.stream.entries[r["sig_id"]] for r in era_consistency(state) if r["eras_seen"] >= min_eras]


def research_targets(report: DailyReport, n: int = 10) -> list[dict[str, Any]]:
    """Section 22 hand-off: the day's best opportunities as research targets carrying an ExperimentValue the priority engine can
    rank against other work. Estimates only: fields the day cannot support stay None, never 0. The loss-reduction value is the
    book loss that fixing this cause could have avoided; unknowable causes carry no decision value."""
    out = []
    for x in report.top[:n]:
        learnable = x.knowability in LEARNABLE
        val = ExperimentValue(
            information_gain=x.criteria["novelty"] * x.criteria["predictability"] if learnable else 0.0,
            decision_value=x.criteria["decision_relevance"] if learnable else 0.0,
            loss_reduction_value=x.criteria["loss_reduction"] if x.is_loss else None,
            volatility_value=None if x.is_loss else x.criteria["magnitude"] * x.criteria["predictability"],
            transfer_potential=x.criteria["repeatability"], redundancy=1.0 - x.criteria["novelty"],
            failure_reduction_value=x.criteria["loss_reduction"] if x.is_loss else None,
            compute_cost=0.5 if x.capture == Capture.MISSED else 1.0)
        out.append({"cid": x.cid, "sig_id": x.sig_id, "direction": x.direction, "knowability": x.knowability.value, "reason": x.reason,
                    "effect": x.effect.value, "score": x.score, "value": val, "problem": Problem.LOSS_AVOIDANCE if x.is_loss else Problem.VOLATILITY,
                    "path": x.path})
    return out


def narrate(o: MoveObs, view: LiftView, p: Params, k: int | None = None) -> str:
    """One paragraph answering 'what happened and could it have been known?' for a single move: capture, why, knowability,
    the strongest pre-move evidence and the information that was or was not available. Research side only."""
    cap = classify_capture(o, p, k)
    rep = assess(o, view, p)
    side = "gain" if o.move_dir(p) == UP else "loss" if o.move_dir(p) == DOWN else "non-move"
    parts = [f"A {side} of {o.fwd:+.1%} was {cap.capture.value} (coverage {cap.coverage:.2f})."]
    if o.move_dir(p) == DOWN and o.picked:
        ex = LossAnalyzer(p).explain(o)
        parts.append(f"Owned through the loss; main cause {ex.primary.value}" + ("" if ex.avoidable_by_stop else " (no stop could have helped)") + ".")
    if rep.p_mover is not None:
        top = ", ".join(f"{f} in bin {b} ({lift:.1f}x)" for f, b, lift, _, _ in view.evidence(o.features, "up" if o.fwd > 0 else "down").contributors[:3])
        parts.append(f"Pre-move history gave {rep.p_mover:.2f} ({rep.lift:.1f}x base)" + (f" from {top}" if top else "") + ".")
    parts.append(f"Classified {rep.knowability.value} at confidence {rep.confidence_in_classification:.2f}: " + "; ".join(rep.reasons) + ".")
    if rep.information_that_would_have_been_available:
        parts.append("Available beforehand: " + ", ".join(rep.information_that_would_have_been_available) + ".")
    if rep.information_that_was_unavailable:
        parts.append("Not in the data: " + ", ".join(rep.information_that_was_unavailable) + ".")
    return " ".join(parts)


# ==================================================================================================================
# 17. does the missed-opportunity picture replicate? (same-window results are interesting, transfer is required)
# ==================================================================================================================
def coverage_curve(reports: Sequence[DailyReport], window: int = 10) -> pd.DataFrame:
    """Rolling engagement lift and unknowable share, so 'the system misses fewer learnable moves over time' is a curve to inspect
    rather than a claim. A rising winner or loser lift without a matching fall in learnable misses would be a warning."""
    rows = []
    for i in range(0, max(0, len(reports) - window + 1)):
        w = reports[i:i + window]
        lu = [math.log(r.engagement_lift(UP)) for r in w if math.isfinite(r.engagement_lift(UP))]
        ld = [math.log(r.engagement_lift(DOWN)) for r in w if math.isfinite(r.engagement_lift(DOWN))]
        moves = sum(r.n_moves for r in w)
        rows.append({"end": w[-1].decided_at, "winner_log_lift": float(np.mean(lu)) if lu else float("nan"),
                     "loser_log_lift": float(np.mean(ld)) if ld else float("nan"),
                     "learnable_miss_rate": sum(r.n_learnable_misses for r in w) / moves if moves else float("nan"),
                     "unknowable_miss_rate": sum(r.n_unknowable_misses for r in w) / moves if moves else float("nan"),
                     "loss_cost": float(sum(r.loss_cost for r in w))})
    return pd.DataFrame(rows, columns=["end", "winner_log_lift", "loser_log_lift", "learnable_miss_rate", "unknowable_miss_rate", "loss_cost"])


def half_split_replication(days: Sequence[DayBook], feature_names: Sequence[str], p: Params | None = None, top: int = 20) -> dict[str, Any]:
    """Section 32 in miniature. Run the whole engine independently on the early and the late half of the days and ask which
    stream topics are among the best in BOTH. A topic that only ranks in one half is same-window noise. Reports the overlap,
    the expected overlap under independence (hypergeometric) and its p-value, so 'they overlap' is not mistaken for 'they replicate'."""
    if len(days) < 12:
        return {"verdict": "INSUFFICIENT", "days": len(days)}
    mid = len(days) // 2
    ordered = sorted(days, key=lambda d: d.decided_at)
    sides = []
    for part in (ordered[:mid], ordered[mid:]):
        st, _ = replay(part, feature_names, p)
        sides.append(st)
    tops = [{e.sig_id for e in s.stream.top(top)} for s in sides]
    universe = set(sides[0].stream.entries) | set(sides[1].stream.entries)
    common = tops[0] & tops[1]
    n_u, n_a, n_b = len(universe), len(tops[0]), len(tops[1])
    pv = float(sps.hypergeom.sf(len(common) - 1, n_u, n_a, n_b)) if n_u and common else 1.0
    return {"days": len(days), "universe": n_u, "early_top": n_a, "late_top": n_b, "overlap": len(common), "expected_overlap": n_a * n_b / n_u if n_u else 0.0,
            "p_overlap": pv, "replicated": sorted(common), "verdict": "REPLICATES" if pv < 0.05 and common else "DOES_NOT_REPLICATE"}


# ==================================================================================================================
# 18. the trader-side hand-off and report export
# ==================================================================================================================
def trader_handoff(item: Any, now) -> Mapping[str, Any]:
    """The single door from this module toward a decision. Only a MaturedRecord may pass, only through gate(now), and only if its
    payload carries no date, year or name-shaped token (trader_view.assert_trader_safe). Reports, audits, opportunities and
    stream entries are research-side objects (they contain the outcome) and are refused outright rather than scrubbed."""
    if not isinstance(item, MaturedRecord):
        raise FirewallBreach(f"{type(item).__name__} is a research-side object and may not reach the trader; wrap it in a MaturedRecord")
    payload = item.gate(now)
    TV.assert_trader_safe(payload, f"record {item.record_id}")
    return payload


def report_to_dict(rep: DailyReport) -> dict[str, Any]:
    """JSON-safe form of a daily report for the results folder (research side: dates and counts are fine here)."""
    return {"decided_at": rep.decided_at, "n_moves": rep.n_moves, "captures": {str(d): dict(c) for d, c in rep.captures.items()},
            "knowability": {str(d): dict(c) for d, c in rep.knowability.items()}, "reasons": dict(rep.reasons),
            "learnable_misses": rep.n_learnable_misses, "unknowable_misses": rep.n_unknowable_misses, "loss_cost": rep.loss_cost,
            "loss_unavoidable": rep.loss_unavoidable, "stable_separators": list(rep.stable_separators), "stream_size": rep.stream_size,
            "stream_states": dict(rep.stream_states),
            "engagement_lift": {str(d): rep.engagement_lift(d) for d in (UP, DOWN)},
            "near_misses": [{"source": n.source, "gap": n.gap, "informative": n.informative, "direction": n.direction} for n in rep.near_misses],
            "top": [{"cid": x.cid, "direction": x.direction, "capture": x.capture.value, "knowability": x.knowability.value, "fwd": x.fwd,
                     "score": x.score, "reason": x.reason, "criteria": dict(x.criteria), "path": x.path} for x in rep.top],
            "path": None if rep.path is None else {"labelled": rep.path.n_labelled, "distribution": dict(rep.path.distribution)}}


def write_reports(reports: Sequence[DailyReport], path: str | os.PathLike) -> int:
    """Append-safe JSON-lines export written atomically (temp then replace). Returns the number of lines written."""
    target = os.fspath(path)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(report_to_dict(r), sort_keys=True, default=float) + NL)
    os.replace(tmp, target)
    return len(reports)
