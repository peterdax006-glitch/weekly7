"""Daily market autopsy (RESEARCH_BRAIN_CONTRACT C66 section 21, with the C67 band counts; canon C66 + C67 + C62 + C63).

After every simulated day the autopsy reads the observer's DayRecord (the whole-universe record of section 4) and writes five
sections - MARKET, MODEL, RESEARCH, RISK, LEARNING - and turns what it found into identity-free research questions that feed the
autonomous research queue (section 22 / 40). It does not decide anything and it changes no live state: it is MATURED_RESEARCH_STATE,
reachable by the trader only through MaturedRecord.gate(now) / the curator.

  MARKET    largest gainers / losers / ranges / volume anomalies / gaps / volatility expansions and contractions, and the C67 movers per
            band (5-10% and >10%, up and down, close-to-close and open-to-close) with how unusual today's counts are against the past.
  MODEL     best / worst predictions, largest missed winners (with the rejection analyser's reason) and missed losers, the highest-
            confidence mistakes and the lowest-confidence successes, and whether the score orders movers at all.
  RESEARCH  new and broken patterns (handed in by the pattern store), surprising observations (engine.learning.surprise), contradictions,
            unknown causes (a mover the proxy cannot place stays UNKNOWN; a suspect print is DATA_FAILURE), and the new questions.
  RISK      avoidable versus unavoidable losses (engine.learning.failure's classifier, never forced), concentration, correlated failures,
            gap risk, regime exposure.
  LEARNING  what changed in knowledge, in confidence, in research priority, and which experiments should run next.

Built ON: engine.research.observer (DayRecord, band profiles, rejection reasons, trailing z-scores), engine.learning.surprise
(SurpriseTracker cells and research priorities), engine.learning.failure (LossClassifier / TradeRecord), engine.learning.missed_winners
(RejectionAnalyzer via the observer) and engine.research.core (ResearchQuestion, ExperimentValue, Knowability, MaturedRecord).
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import failure as fl
from engine.learning import surprise as sp
from engine.research import observer as ob
from engine.research.core import (ExperimentValue, FailureCause, FirewallBreach, Knowability, MaturedRecord, MoveCategory, Problem,
                                  Provenance, ResearchQuestion, as_date, require_past, stable_hash)

MC = MoveCategory
FC = FailureCause
SECTIONS = ("market", "model", "research", "risk", "learning")
AVOIDABLE = frozenset({FC.SELECTION_ERROR, FC.TIMING_ERROR, FC.RISK_ERROR, FC.FALSE_PATTERN, FC.WRONG_CONTEXT, FC.WEAKENING_EFFECT,
                       FC.REDUNDANCY, FC.INTERACTION_FAILURE, FC.MEASUREMENT_ERROR})
UNAVOIDABLE = frozenset({FC.REGIME_CHANGE, FC.REVERSAL, FC.TEMPORARY_INACTIVITY})
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class AutopsyError(ValueError):
    """Malformed autopsy input or an output that would break an invariant (identity in a question, missing section)."""


# ==================================================================================================================
# parameters, inputs and small records
# ==================================================================================================================
@dataclass(frozen=True)
class AutopsyParams:
    top_n: int = 5                        # entries per ranked list
    loss_thr: float = 0.03                # a pick that lost this much (fraction, from the fill) is analysed
    cost: float = 0.0005                  # round-trip cost charged against a pick's return
    window: int = 60                      # trailing days for "unusual" (past days only)
    min_history: int = 10
    z_unusual: float = 2.0                # a market statistic this many robust sigmas from its trailing median is unusual
    hhi_high: float = 0.25                # pick concentration (Herfindahl over sectors / types) above this is flagged
    co_loss_z: float = 2.0                # losses clustered beyond this z (versus independence) are a correlated failure
    adverse_gap_atr: float = 2.0          # an entry gap beyond this many ATRs against the pick is a gap-risk event
    high_vol_rank: float = 0.9            # rank of vol20 at/above which a pick is a high-volatility exposure
    low_liq_rank: float = 0.1
    unplaced_flag: float = 0.5            # share of movers left unplaced above which the proxy is called weak
    max_questions: int = 12
    max_experiments: int = 5
    surprise_scale_floor: float = 1e-4

    def validate(self) -> list[str]:
        e = []
        if self.top_n < 1 or self.max_questions < 1 or self.max_experiments < 1:
            e.append("list lengths must be >= 1")
        if not 0 < self.loss_thr < 1:
            e.append("loss_thr outside (0, 1)")
        if self.cost < 0:
            e.append("cost < 0")
        if self.window < self.min_history or self.min_history < 3:
            e.append("need 3 <= min_history <= window")
        if not 0 < self.hhi_high <= 1 or not 0 <= self.low_liq_rank < self.high_vol_rank <= 1:
            e.append("concentration / rank thresholds out of range")
        if self.surprise_scale_floor <= 0:
            e.append("surprise_scale_floor must be positive")
        return e

    def require_valid(self) -> "AutopsyParams":
        errs = self.validate()
        if errs:
            raise ValueError("invalid AutopsyParams: " + "; ".join(errs))
        return self

    def hash(self) -> str:
        return stable_hash(self)


@dataclass(frozen=True)
class PatternEvent:
    """What the pattern store says happened to one pattern today. The autopsy reads it; it never mines or re-tests."""
    pattern_id: str
    kind: str                             # new | broken | revived | strengthened | weakened | held
    effect_before: float | None = None
    effect_after: float | None = None
    p_real: float | None = None
    n_fired: int = 0
    n_hit: int = 0
    expected_sign: int = 0                # +1 expects a rise, -1 a fall, 0 no directional claim
    cell: str = ""
    note: str = ""

    KINDS = ("new", "broken", "revived", "strengthened", "weakened", "held")

    def validate(self) -> list[str]:
        e = []
        if not self.pattern_id:
            e.append("pattern_id missing")
        if self.kind not in self.KINDS:
            e.append(f"pattern kind {self.kind!r} not in {self.KINDS}")
        if self.n_hit > self.n_fired or self.n_fired < 0 or self.n_hit < 0:
            e.append("n_hit must lie in [0, n_fired]")
        if self.p_real is not None and not 0.0 <= self.p_real <= 1.0:
            e.append("p_real outside [0, 1]")
        if self.expected_sign not in (-1, 0, 1):
            e.append("expected_sign must be -1, 0 or +1")
        return e


@dataclass(frozen=True)
class KnowledgeState:
    """A knowledge item's state at one moment, for the LEARNING diff. Any field may be None (never measured)."""
    kid: str
    confidence: float | None = None
    lifecycle: str = ""
    priority: float | None = None


@dataclass
class AutopsyContext:
    """Everything the autopsy needs besides the day's record. All optional: a section that lacks its inputs says so."""
    patterns: Sequence[PatternEvent] = ()
    pattern_hits: Mapping[str, Sequence[str]] = field(default_factory=dict)      # cid -> pattern ids that fired on it
    knowledge_before: Mapping[str, KnowledgeState] = field(default_factory=dict)
    knowledge_after: Mapping[str, KnowledgeState] = field(default_factory=dict)
    priorities_before: Mapping[str, float] = field(default_factory=dict)
    priorities_after: Mapping[str, float] = field(default_factory=dict)
    running: Sequence[str] = ()                                                    # question ids already being worked
    weights: Mapping[str, float] = field(default_factory=dict)                    # cid -> position weight (default equal)

    def validate(self) -> list[str]:
        e = []
        for pe in self.patterns:
            e += [f"pattern {pe.pattern_id}: {x}" for x in pe.validate()]
        for k, v in list(self.knowledge_before.items()) + list(self.knowledge_after.items()):
            if v.confidence is not None and not 0.0 <= v.confidence <= 1.0:
                e.append(f"knowledge {k}: confidence outside [0, 1]")
        for cid, w in self.weights.items():
            if not np.isfinite(w) or w < 0:
                e.append(f"weight of {cid} is not a non-negative number")
        return e


@dataclass(frozen=True)
class Entry:
    """One ranked item in a list. `ticker` is research-side only: identity_free() drops it and keeps the salted cid."""
    cid: str
    ticker: str
    value: float
    label: str = ""
    tags: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def identity_free(self) -> dict[str, Any]:
        return {"cid": self.cid, "value": self.value, "label": self.label, "tags": list(self.tags), "extra": dict(self.extra)}


def _nz(x: Any) -> float | None:
    """A finite float or None (NaN, inf and missing all become None; never 0)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _entries(df: pd.DataFrame, value: str, n: int, largest: bool = True, label: str = "", tags=lambda r: (), extra=lambda r: {}) -> list[Entry]:
    if df.empty or value not in df:
        return []
    d = df[np.isfinite(df[value].astype(float))]
    d = d.nlargest(n, value) if largest else d.nsmallest(n, value)
    out = []
    for r in d.itertuples(index=False):
        out.append(Entry(r.cid, str(r.ticker), float(getattr(r, value)), label, tuple(tags(r)), dict(extra(r))))
    return out


def _tags_of(r) -> tuple[str, ...]:
    return tuple(c.value for c in ob.cats_of(int(r.flags)))


# ==================================================================================================================
# MARKET
# ==================================================================================================================
@dataclass(frozen=True)
class MarketSection:
    gainers: tuple[Entry, ...]
    losers: tuple[Entry, ...]
    ranges: tuple[Entry, ...]
    volume: tuple[Entry, ...]
    gaps: tuple[Entry, ...]
    vol_expansion: tuple[Entry, ...]
    vol_contraction: tuple[Entry, ...]
    bands: Mapping[str, int]
    bands_o2c: Mapping[str, int]
    bands_eligible: Mapping[str, int]
    band_z: Mapping[str, float]
    band_profile: Mapping[str, Mapping[str, float]]
    move_hist: Mapping[str, Sequence[int]]
    zscores: Mapping[str, float]
    unusual: tuple[str, ...]
    cell: str
    tone: str
    n_suspect_band: int
    symmetry: Mapping[str, float]
    sector_hot: tuple[tuple[str, float], ...]

    def lists(self) -> dict[str, tuple[Entry, ...]]:
        return {k: getattr(self, k) for k in ob.TOP_METRICS}


def band_unusualness(rec: ob.DayRecord, ledger: ob.ObserverLedger, p: AutopsyParams) -> dict[str, float]:
    """Robust z of today's count in each C67 band against the trailing days' counts (past days only). NaN until enough history.
    A hundred 5-10% movers is ordinary in one regime and a storm in another; this says which."""
    cut = as_date(rec.decided_at)
    past = [r for r in ledger if as_date(r.resolved_at) <= cut and not r.empty][-p.window:]
    out = {}
    for name in ob.BAND_NAMES.values():
        hist = np.array([r.bands[name] / max(1, r.n_universe) for r in past], dtype=float)
        x = rec.bands[name] / max(1, rec.n_universe)
        if len(hist) < p.min_history:
            out[name] = float("nan")
            continue
        med = float(np.median(hist))
        sc = 1.4826 * float(np.median(np.abs(hist - med)))
        sc = max(sc, 0.5 * math.sqrt(max(med, 1.0 / max(1, rec.n_universe)) / max(1, rec.n_universe)))     # Poisson floor: a constant history is not certainty
        out[name] = (x - med) / sc
    return out


def market_section(rec: ob.DayRecord, ledger: ob.ObserverLedger, p: AutopsyParams, obs: ob.ObserverParams) -> MarketSection:
    rf = rec.rows.set_index("ticker", drop=False) if not rec.rows.empty else rec.rows
    lists = {}
    for metric in ob.TOP_METRICS:
        names = [t for t in rec.tops.get(metric, ()) if not rf.empty and t in rf.index][:p.top_n]
        sub = rf.loc[names] if names else rf.iloc[0:0]
        col = {"gainers": "sig_ret", "losers": "sig_ret", "ranges": "rng", "volume": "volume_ratio", "gaps": "gap",
               "vol_expansion": "vol_x", "vol_contraction": "vol_x"}[metric]
        lists[metric] = tuple(Entry(r.cid, str(r.ticker), float(getattr(r, col)), metric, _tags_of(r) + (("suspect",) if r.suspect else ()),
                                    {"band": int(r.band), "mover_type": str(r.mover_type), "picked": bool(r.picked)})
                              for r in sub.itertuples(index=False))
    z = ob.market_zscores(rec, ledger, p.window, p.min_history)
    bz = band_unusualness(rec, ledger, p)
    unusual = tuple(sorted([f"market:{k}" for k, v in z.items() if v == v and abs(v) >= p.z_unusual] +
                           [f"band:{k}" for k, v in bz.items() if v == v and abs(v) >= p.z_unusual]))
    prof = ob.band_profile(rec, obs)
    sect = ob.sector_breadth(rec)
    hot = tuple((str(i), float(r.z)) for i, r in sect.head(3).iterrows() if r.z == r.z and r.z >= 3.0) if not sect.empty else ()
    return MarketSection(**{m: lists[m] for m in ob.TOP_METRICS}, bands=dict(rec.bands), bands_o2c=dict(rec.bands_o2c),
                         bands_eligible=dict(rec.bands_eligible), band_z=bz,
                         band_profile={k: {c: float(v) for c, v in row.items()} for k, row in prof.iterrows()} if not prof.empty else {},
                         move_hist=dict(rec.market.get("move_hist", {})), zscores=z, unusual=unusual,
                         cell=ob.context_cell(rec, z), tone=str(rec.market.get("tone", "unknown")),
                         n_suspect_band=int(rec.market.get("n_suspect_band", 0)), symmetry=ob.symmetry(rec) if not rec.empty else {},
                         sector_hot=hot)


# ==================================================================================================================
# MODEL
# ==================================================================================================================
@dataclass(frozen=True)
class ModelSection:
    best: tuple[Entry, ...]
    worst: tuple[Entry, ...]
    missed_winners: tuple[Entry, ...]
    missed_losers: tuple[Entry, ...]
    dodged_losers: tuple[Entry, ...]
    confident_mistakes: tuple[Entry, ...]
    unconfident_successes: tuple[Entry, ...]
    reason_counts: Mapping[str, int]
    winner_why: Mapping[str, int]
    score_auc: float
    precursor_auc: float
    mcc: float
    recall: float
    precision: float
    decile_lift: float
    abstention: Mapping[str, float]
    low_confidence: Mapping[str, float]
    capturable_share: float
    n_picked: int


def _mask(rf: pd.DataFrame, cat: MoveCategory) -> np.ndarray:
    return np.asarray(ob.has(rf["flags"].to_numpy(), cat)) if not rf.empty else np.zeros(0, dtype=bool)


def _hit_rate(g: pd.DataFrame, thr: float) -> dict[str, float]:
    """Share of the rows that moved by `thr` either way from the decision close; the count is reported so a rate over three
    names is not mistaken for a rate."""
    if g.empty:
        return {"n": 0.0, "moved_share": float("nan"), "mean_abs_move": float("nan")}
    exc = g["exc_pc"].astype(float)
    ok = np.isfinite(exc)
    return {"n": float(len(g)), "moved_share": float((exc[ok] >= thr).mean()) if ok.any() else float("nan"),
            "mean_abs_move": float(exc[ok].mean()) if ok.any() else float("nan")}


def _decile_lift(tab: Sequence[Sequence[int]]) -> float:
    """Mover rate of the top score decile over the bottom decile (>1 means the score orders movers the right way round)."""
    if not tab or len(tab) < 4:
        return float("nan")
    lo, hi = tab[0], tab[-1]
    r_lo, r_hi = lo[1] / lo[0] if lo[0] else float("nan"), hi[1] / hi[0] if hi[0] else float("nan")
    if not (r_lo == r_lo and r_hi == r_hi):
        return float("nan")
    return float(r_hi / r_lo) if r_lo > 0 else (float("inf") if r_hi > 0 else float("nan"))


def model_section(rec: ob.DayRecord, p: AutopsyParams, obs: ob.ObserverParams) -> ModelSection:
    rf = rec.rows
    empty = ModelSection((), (), (), (), (), (), (), {}, {}, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), {}, {}, float("nan"), 0)
    if rf.empty:
        return empty
    picked = rf["picked"].to_numpy(bool)
    high = _mask(rf, MC.CONSIDERED_HIGH)
    win = _mask(rf, MC.WINNER)
    week = ob.to_week(rec, obs)
    rej = {r.cid: r for r in ob.lmw.RejectionAnalyzer().explain_missed(week)} if week.missed() else {}
    pk = rf[picked]
    fill_win = rf["ret"].astype(float) >= obs.win_thr
    best = _entries(pk, "ret", p.top_n, True, "best_prediction", _tags_of, lambda r: {"rank": int(r.rank), "score": _nz(r.score)})
    worst = _entries(pk, "ret", p.top_n, False, "worst_prediction", _tags_of, lambda r: {"rank": int(r.rank), "score": _nz(r.score)})
    mw = rf[win & ~picked].copy()
    mw["why_missed"] = [rej[c].primary.value if c in rej else ("GAP_BEFORE_FILL" if np.isfinite(g) and g >= obs.win_thr * 0.5 else "NOT_CAPTURABLE")
                     for c, g in zip(mw["cid"], mw["gap"].astype(float))]
    missed_w = tuple(Entry(r.cid, str(r.ticker), float(r.sig_ret), "missed_winner", _tags_of(r),
                           {"reason": r.why_missed, "capturable_ret": _nz(r.ret), "rank": int(r.rank), "mover_type": str(r.mover_type)})
                     for r in mw.nlargest(p.top_n, "sig_ret").itertuples(index=False)) if not mw.empty else ()
    loss = rf["ret"].astype(float) <= -p.loss_thr
    missed_l = _entries(rf[picked & loss], "ret", p.top_n, False, "missed_loser", _tags_of, lambda r: {"score": _nz(r.score), "confidence": _nz(r.confidence)})
    dodged = _entries(rf[high & ~picked & loss], "ret", p.top_n, False, "dodged_loser", _tags_of, lambda r: {"rank": int(r.rank)})
    conf_ok = np.isfinite(rf["confidence"].astype(float))
    cand = rf[(picked | high) & conf_ok]
    wrong = cand[(_mask(cand, MC.FALSE_POSITIVE)) | (cand["ret"].astype(float) <= -p.loss_thr)]
    ok = cand[(cand["ret"].astype(float) >= obs.win_thr)]
    cm = tuple(Entry(r.cid, str(r.ticker), float(r.confidence), "confident_mistake", _tags_of(r), {"ret": _nz(r.ret), "rank": int(r.rank)})
               for r in wrong.nlargest(p.top_n, "confidence").itertuples(index=False)) if not wrong.empty else ()
    us = tuple(Entry(r.cid, str(r.ticker), float(r.confidence), "unconfident_success", _tags_of(r), {"ret": _nz(r.ret), "rank": int(r.rank)})
               for r in ok.nsmallest(p.top_n, "confidence").itertuples(index=False)) if not ok.empty else ()
    reason_counts: dict[str, int] = {}
    for r in rej.values():
        reason_counts[r.primary.value] = reason_counts.get(r.primary.value, 0) + 1
    conf = rec.model.get("confusion_fill", {})
    tp, fp = conf.get("tp", 0), conf.get("fp", 0)
    winners_total = int(win.sum())
    cap = float((win & fill_win.to_numpy()).sum() / winners_total) if winners_total else float("nan")
    return ModelSection(tuple(best), tuple(worst), missed_w, tuple(missed_l), tuple(dodged), cm, us, reason_counts, dict(rec.model.get("winner_why", {})),
                        float(rec.model.get("score_auc", float("nan"))), float(rec.model.get("precursor_auc", float("nan"))),
                        float(rec.model.get("mcc_fill", float("nan"))), float(rec.model.get("recall_of_movers", float("nan"))),
                        float(tp / (tp + fp)) if tp + fp else float("nan"), _decile_lift(rec.model.get("score_deciles", [])),
                        _hit_rate(rf[_mask(rf, MC.ABSTAINED)], obs.mover_thr), _hit_rate(rf[_mask(rf, MC.LOW_CONFIDENCE)], obs.mover_thr),
                        cap, int(picked.sum()))


# ==================================================================================================================
# RESEARCH
# ==================================================================================================================
@dataclass(frozen=True)
class Surprise:
    metric: str
    cell: str
    expected: float
    actual: float
    z: float
    bits: float
    scale_source: str


@dataclass(frozen=True)
class Contradiction:
    kind: str
    cid: str
    detail: str
    magnitude: float


@dataclass(frozen=True)
class UnknownCause:
    """A mover whose cause the autopsy does NOT claim to know. `verdict` is a Knowability value, and PREDICTABLE is never given
    here: only the counterfactual (could-I-have-known) engine may prove that; the proxy at most says POTENTIALLY_PREDICTABLE."""
    verdict: Knowability
    count: int
    share_of_movers: float
    note: str


@dataclass(frozen=True)
class ResearchSection:
    new_patterns: tuple[PatternEvent, ...]
    broken_patterns: tuple[PatternEvent, ...]
    other_pattern_events: tuple[PatternEvent, ...]
    surprises: tuple[Surprise, ...]
    surprise_skipped: tuple[str, ...]
    contradictions: tuple[Contradiction, ...]
    unknown_causes: tuple[UnknownCause, ...]
    proxy_informative: bool
    questions: tuple[ResearchQuestion, ...]
    investigations: tuple[str, ...]


def trailing_series(ledger: ob.ObserverLedger, rec: ob.DayRecord, getter, window: int) -> np.ndarray:
    """A statistic from the past records only (resolved no later than today's decision), newest last, finite values."""
    cut = as_date(rec.decided_at)
    vals = [getter(r) for r in ledger if as_date(r.resolved_at) <= cut and not r.empty][-window:]
    a = np.array([np.nan if v is None else v for v in vals], dtype=float)
    return a[np.isfinite(a)]


SURPRISE_METRICS = (("mover_rate", lambda r: r.market.get("mover_rate"), lambda r: r.market.get("mover_rate")),
                    ("pick_hit_rate", lambda r: r.model.get("pick_hit_rate"), lambda r: r.model.get("pick_hit_rate")),
                    ("recall_of_movers", lambda r: r.model.get("recall_of_movers"), lambda r: r.model.get("recall_of_movers")),
                    ("gap_median_abs", lambda r: r.market.get("gap_median_abs"), lambda r: r.market.get("gap_median_abs")))


def observe_surprises(rec: ob.DayRecord, ledger: ob.ObserverLedger, tracker: sp.SurpriseTracker, cell: str, p: AutopsyParams,
                      now) -> tuple[list[Surprise], list[str]]:
    """Feed today's market and model statistics to the SurpriseTracker as expectation-versus-outcome. The expectation is the
    trailing median of the same statistic over past days and the scale its trailing robust spread, so 'surprising' is relative to
    what the system itself had seen. Returns (surprises at/above the investigation level, metrics skipped and why)."""
    require_past(rec.resolved_at, now, "autopsy day resolved_at")
    got, skipped = [], []
    for name, getter, _ in SURPRISE_METRICS:
        actual = _nz(getter(rec))
        hist = trailing_series(ledger, rec, getter, p.window)
        if actual is None:
            skipped.append(f"{name}: not measurable today")
            continue
        if len(hist) < p.min_history:
            skipped.append(f"{name}: only {len(hist)} past days")
            continue
        med = float(np.median(hist))
        scale = max(1.4826 * float(np.median(np.abs(hist - med))), p.surprise_scale_floor)
        try:
            r = tracker.observe(f"metric={name}|{cell}", med, actual, rec.decided_at, rec.resolved_at, now, scale=scale)
        except ValueError:                         # the same expectation was already recorded: a re-run of the same day
            skipped.append(f"{name}: already observed")
            continue
        if r.magnitude >= tracker.cfg.z_high:
            got.append(Surprise(name, r.cell, r.expected, r.actual, r.z, r.bits, r.scale_source))
    return got, skipped


def find_contradictions(rec: ob.DayRecord, ctx: AutopsyContext, obs: ob.ObserverParams, p: AutopsyParams) -> list[Contradiction]:
    """Places where the system disagreed with itself today. Each is a pair of things that should not both be true:
    a confident pick whose own eve-of-move proxy was in the bottom half; a loud proxy on a mover the score ranked last;
    a direction probability opposite to the side taken; a pattern that expected one sign on a name that went the other."""
    rf = rec.rows
    out: list[Contradiction] = []
    if rf.empty:
        return out
    prec = rf["precursor"].astype(float)
    picked = rf["picked"].to_numpy(bool)
    for r in rf[picked & (prec < 0.4) & (rf["confidence"].astype(float) >= 0.7) & (rf["ret"].astype(float) < obs.mover_thr * 0.5)].itertuples(index=False):
        out.append(Contradiction("confident_pick_quiet_precursor", r.cid, f"confidence {r.confidence:.2f} but precursor {r.precursor:.2f} "
                                 "and the name did not move", float(r.confidence - r.precursor)))
    fn = _mask(rf, MC.FALSE_NEGATIVE)
    for r in rf[fn & (prec >= 0.9) & (rf["rank"] > obs.k_med)].itertuples(index=False):
        out.append(Contradiction("loud_precursor_ranked_last", r.cid, f"precursor {r.precursor:.2f} yet score rank {int(r.rank)} and it moved",
                                 float(r.precursor)))
    dp = rf["dir_prob"].astype(float)
    for r in rf[picked & np.isfinite(dp) & (dp < 0.35)].itertuples(index=False):
        out.append(Contradiction("pick_against_direction_model", r.cid, f"picked long while P(up) = {r.dir_prob:.2f}", float(0.5 - r.dir_prob)))
    signs = {pe.pattern_id: pe.expected_sign for pe in ctx.patterns if pe.expected_sign}
    if signs and ctx.pattern_hits:
        by_cid = {r.cid: r for r in rf.itertuples(index=False)}
        for cid, pats in ctx.pattern_hits.items():
            row = by_cid.get(cid)
            if row is None or not np.isfinite(row.ret):
                continue
            for pid in pats:
                s = signs.get(pid)
                if s and s * row.ret <= -obs.mover_thr:
                    out.append(Contradiction("pattern_sign_reversed", cid, f"pattern {pid} expected {'a rise' if s > 0 else 'a fall'}, "
                                             f"the name moved {row.ret:+.2%}", float(abs(row.ret))))
    if rec.market.get("breadth_up") == rec.market.get("breadth_up") and rec.market.get("median_ret") == rec.market.get("median_ret"):
        if rec.market["breadth_up"] > 0.6 and rec.market["median_ret"] < -0.003:
            out.append(Contradiction("breadth_versus_median", "market", "most names rose yet the median return was negative", 1.0))
    out.sort(key=lambda c: (-c.magnitude, c.kind, c.cid))
    return out[:max(3 * p.top_n, 10)]


def unknown_causes(rec: ob.DayRecord, informative: bool) -> list[UnknownCause]:
    """Section 33: say what is not known. Movers split into: data failures (suspect prints), placed as potentially predictable
    (the proxy saw an eve-of-move signal, and only if the proxy is itself informative), placed as gap-driven with no known event
    (still UNKNOWN - no news channel exists to explain it), and unplaced. Nothing is forced into a cause."""
    moved = int(rec.market.get("n_moved", 0))
    if not moved:
        return []
    pred, unpred = rec.n(MC.PREDICTABLE_MOVER), rec.n(MC.UNPREDICTABLE_MOVER)
    rf = rec.rows
    sus = int(((rf["suspect"] != 0) & (rf["exc_pc"].astype(float) >= 0.07)).sum()) if not rf.empty and "suspect" in rf else 0
    out = []
    if sus:
        out.append(UnknownCause(Knowability.DATA_FAILURE, sus, sus / moved, "moves on split-like, extreme or no-trade prints; not market events"))
    if pred:
        v = Knowability.POTENTIALLY_PREDICTABLE if informative else Knowability.UNKNOWN
        out.append(UnknownCause(v, pred, pred / moved, "loud eve-of-move precursors, a known event or a top score" +
                                ("" if informative else "; the proxy has not shown it orders movers, so this is only a tag")))
    if unpred:
        out.append(UnknownCause(Knowability.UNKNOWN, unpred, unpred / moved, "an overnight gap with quiet precursors and no known event; "
                                "the cause is not observable here, so it is not called externally caused either"))
    left = moved - pred - unpred - sus
    if left > 0:
        out.append(UnknownCause(Knowability.UNKNOWN, left, left / moved, "movers the proxy could not place; left unplaced on purpose"))
    return out


# ==================================================================================================================
# RISK
# ==================================================================================================================
@dataclass(frozen=True)
class LossItem:
    cid: str
    ticker: str
    loss: float
    verdict: str                          # avoidable | unavoidable | undetermined
    cause: str                            # FailureCause value from the classifier, or "" when it named none
    reasons: tuple[str, ...]
    classifier_confidence: float


@dataclass(frozen=True)
class RiskSection:
    n_positions: int
    n_losses: int
    pnl_total: float
    loss_total: float
    avoidable: tuple[LossItem, ...]
    unavoidable: tuple[LossItem, ...]
    undetermined: tuple[LossItem, ...]
    avoidable_loss_share: float
    unknown_rate: float
    concentration: Mapping[str, float]
    correlated: Mapping[str, float]
    gap_risk: Mapping[str, float]
    regime_exposure: Mapping[str, float]
    flags: tuple[str, ...]


def trade_from_row(r, rec: ob.DayRecord, obs: ob.ObserverParams, weight: float, cost: float, universe_ret: float | None) -> fl.TradeRecord | None:
    """A pick's row as an engine.learning.failure TradeRecord. Fields the day's record cannot know (patterns, stops, targets) stay
    None, so the classifier's detectors for them report 'could not run' - lowering coverage - instead of guessing."""
    ret, atr = _nz(r.ret), _nz(getattr(r, "atr", None))
    if ret is None:
        return None
    exp_vol = atr * math.sqrt(obs.horizon) if atr and atr > 0 else None
    n_scored = max(1, rec.n_scored)
    rank_pct = min(1.0, max(0.0, int(r.rank) / n_scored)) if int(r.rank) > 0 else None
    dp = _nz(r.dir_prob)
    return fl.TradeRecord(
        rid=r.cid, decided_at=rec.decided_at, resolved_at=rec.resolved_at, side=1, pnl=ret - cost, weight=weight, cost=cost,
        signal_ret=_nz(r.sig_ret), entry_gap=_nz(r.gap), end_ret_from_fill=ret, exit_ret=ret, mfe=_nz(r.hi_x), mae=_nz(r.lo_x),
        exp_move=obs.mover_thr, exp_vol=exp_vol, dir_prob=dp if dp is not None and 0.0 <= dp <= 1.0 else None,
        score=_nz(r.score), rank_pct=rank_pct, universe_ret=universe_ret,
        data_flags={"split_suspect": float(r.suspect == 1), "zero_volume": float(r.suspect == 3)},
        tags={"mover_type": str(r.mover_type), "band": str(int(r.band))})


def preflight_flags(r, p: AutopsyParams, obs: ob.ObserverParams) -> list[str]:
    """Risk signs that were visible at the decision close (nothing after it): used only when the classifier names no cause."""
    out = []
    rv, rl = _nz(getattr(r, "rk_vol20", None)), _nz(getattr(r, "rk_log_dv", None))
    if rv is not None and rv >= p.high_vol_rank:
        out.append("volatility in the top decile")
    if rl is not None and rl <= p.low_liq_rank:
        out.append("liquidity in the bottom decile")
    if bool(r.event_known):
        out.append("a scheduled event fell inside the window")
    atr, gap = _nz(getattr(r, "atr", None)), _nz(r.gap)
    if atr and gap is not None and gap >= p.adverse_gap_atr * atr:
        out.append("the open gapped against the entry by more than the ATR limit")
    conf = _nz(r.confidence)
    if conf is not None and conf < 0.5:
        out.append("the pick carried low confidence")
    if int(r.suspect):
        out.append("the print was a suspect data point")
    return out


def judge_loss(cls: fl.Classification, flags: Sequence[str]) -> tuple[str, str]:
    """(verdict, cause). The classifier's named cause decides when it has one; a loss inside normal noise is unavoidable; an
    unnamed loss is avoidable only with two or more independent pre-trade risk signs, and is otherwise left undetermined."""
    if not cls.meaningful:
        return "unavoidable", ""
    if cls.named:
        c = cls.cause
        return ("avoidable" if c in AVOIDABLE else "unavoidable" if c in UNAVOIDABLE else "undetermined"), c.value
    return ("avoidable" if len(flags) >= 2 else "undetermined"), ""


def herfindahl(keys: Sequence[Any], weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if not len(w) or w.sum() <= 0:
        return float("nan")
    s = pd.Series(w).groupby(pd.Series(list(keys), dtype=object).astype(str).to_numpy()).sum()
    share = s / s.sum()
    return float((share ** 2).sum())


def risk_section(rec: ob.DayRecord, ctx: AutopsyContext, p: AutopsyParams, obs: ob.ObserverParams, classifier: fl.LossClassifier, now) -> RiskSection:
    rf = rec.rows
    picks = rf[rf["picked"].astype(bool)] if not rf.empty else rf
    if picks.empty:
        return RiskSection(0, 0, 0.0, 0.0, (), (), (), float("nan"), float("nan"), {}, {}, {}, {}, ())
    n = len(picks)
    w_raw = np.array([ctx.weights.get(c, 1.0 / n) for c in picks["cid"]], dtype=float)
    w = w_raw / w_raw.sum() if w_raw.sum() > 0 else np.full(n, 1.0 / n)
    ret = picks["ret"].astype(float).to_numpy()
    pnl = np.where(np.isfinite(ret), ret - p.cost, 0.0)
    lossmask = np.isfinite(ret) & (ret <= -p.loss_thr)
    uni = _nz(rec.market.get("median_ret"))
    items: list[LossItem] = []
    n_unknown = 0
    for r, wi, lm in zip(picks.itertuples(index=False), w, lossmask):
        if not lm:
            continue
        t = trade_from_row(r, rec, obs, float(wi), p.cost, uni)
        if t is None:
            continue
        cls = classifier.classify(t, None, now)
        flags = preflight_flags(r, p, obs)
        verdict, cause = judge_loss(cls, flags)
        n_unknown += int(not cls.named)
        reasons = tuple(f"{e.note}" for e in cls.evidence[:2]) + tuple(flags) if cls.named else tuple(flags) or (cls.note or "no supported cause",)
        items.append(LossItem(r.cid, str(r.ticker), float(-(r.ret - p.cost)), verdict, cause, reasons, float(cls.confidence)))
    wmap = dict(zip(picks["cid"], w))
    loss_total = float(sum(i.loss * wmap[i.cid] for i in items))
    by = {v: tuple(sorted([i for i in items if i.verdict == v], key=lambda i: -i.loss)[:p.top_n]) for v in ("avoidable", "unavoidable", "undetermined")}
    all_by = {v: [i for i in items if i.verdict == v] for v in by}
    av = sum(i.loss * wmap[i.cid] for i in all_by["avoidable"])
    sectors = picks["sector"].astype(int).to_numpy()
    conc = {"hhi_sector": herfindahl(sectors, list(w)), "hhi_type": herfindahl(picks["mover_type"].astype(str).to_numpy(), list(w)),
            "top_sector_share": float(pd.Series(w).groupby(sectors).sum().max()),
            "top_loss_share": float(max((i.loss * wmap[i.cid] for i in items), default=0.0) / loss_total) if loss_total > 0 else 0.0,
            "n_sectors_known": float(len(set(sectors[sectors >= 0].tolist())))}
    n_l = int((np.isfinite(ret) & (ret < 0)).sum())
    p_down = 1.0 - float(rec.market.get("breadth_up", 0.5)) if rec.market.get("breadth_up") == rec.market.get("breadth_up") else 0.5
    p_down = min(max(p_down, 1e-3), 1 - 1e-3)
    zc = (n_l - n * p_down) / math.sqrt(n * p_down * (1 - p_down))
    lsec = sectors[np.isfinite(ret) & (ret < 0) & (sectors >= 0)]
    top_loss_sector = float(pd.Series(lsec).value_counts(normalize=True).max()) if len(lsec) else float("nan")
    corr = {"n_picks": float(n), "n_losers": float(n_l), "expected_losers": float(n * p_down), "co_loss_z": float(zc),
            "loser_top_sector_share": top_loss_sector}
    atr = picks["atr"].astype(float).to_numpy() if "atr" in picks else np.full(n, np.nan)
    gap = picks["gap"].astype(float).to_numpy()
    adverse = np.isfinite(gap) & np.isfinite(atr) & (gap >= p.adverse_gap_atr * atr) & (atr > 0)
    losers_gs = picks["gap_share"].astype(float).to_numpy()[lossmask]
    gap_risk = {"adverse_entry_gaps": float(adverse.sum()), "adverse_gap_cost": float(np.nansum(np.where(adverse, gap, 0.0) * w)),
                "worst_mae": float(np.nanmin(picks["lo_x"].astype(float))) if picks["lo_x"].notna().any() else float("nan"),
                "gap_driven_loss_share": float(np.nanmean(losers_gs)) if len(losers_gs) and np.isfinite(losers_gs).any() else float("nan"),
                "universe_median_abs_gap": float(rec.market.get("gap_median_abs", float("nan")))}
    expo = {k[3:]: float(picks[k].astype(float).mean() - 0.5) for k in picks.columns if k.startswith("rk_") and picks[k].notna().any()}
    expo["high_vol_share"] = float((picks.get("rk_vol20", pd.Series(dtype=float)).astype(float) >= p.high_vol_rank).mean()) if "rk_vol20" in picks else float("nan")
    expo["low_liq_share"] = float((picks.get("rk_log_dv", pd.Series(dtype=float)).astype(float) <= p.low_liq_rank).mean()) if "rk_log_dv" in picks else float("nan")
    expo["event_share"] = float(picks["event_known"].astype(bool).mean())
    flags = []
    if conc["hhi_sector"] == conc["hhi_sector"] and conc["hhi_sector"] >= p.hhi_high:
        flags.append("concentrated in few sectors")
    if zc >= p.co_loss_z:
        flags.append("losses clustered beyond independence")
    if adverse.any():
        flags.append("adverse entry gaps")
    if expo["high_vol_share"] == expo["high_vol_share"] and expo["high_vol_share"] >= 0.5:
        flags.append("book is mostly top-decile volatility")
    return RiskSection(n, int(lossmask.sum()), float((pnl * w).sum()), loss_total, by["avoidable"], by["unavoidable"], by["undetermined"],
                       float(av / loss_total) if loss_total > 0 else float("nan"), float(n_unknown / len(items)) if items else float("nan"),
                       conc, corr, gap_risk, expo, tuple(flags))


# ==================================================================================================================
# research questions (section 22 / 40): identity-free, with a success and a failure criterion
# ==================================================================================================================
SOURCE_VALUE = {"loss": (0.70, 0.30), "missed_winner": (0.55, 0.25), "break": (0.50, 0.20), "surprise": (0.40, 0.30),
                "contradiction": (0.45, 0.30), "regime": (0.35, 0.25), "data": (0.30, 0.10), "discovery": (0.40, 0.35)}


def estimate_value(source: str, n: float, magnitude: float, problem: Problem, known: bool = False) -> ExperimentValue:
    """A rough, honest pre-experiment estimate of what answering a question is worth. Only the fields the evidence supports are
    filled; the rest stay None (not estimated, never read as zero). Evidence scales with the sample behind the question."""
    dec, unc = SOURCE_VALUE.get(source, (0.3, 0.3))
    info = float(min(1.0, math.log1p(max(n, 0.0)) / 5.0) * min(1.0, 0.4 + magnitude))
    return ExperimentValue(information_gain=info, decision_value=dec * min(1.0, 0.5 + magnitude / 2),
                           uncertainty_reduction=unc * info, compute_cost=5.0, overfit_risk=0.3 if n < 30 else 0.15,
                           redundancy=0.8 if known else 0.0,
                           loss_reduction_value=dec * min(1.0, magnitude) if problem == Problem.LOSS_AVOIDANCE else None,
                           volatility_value=dec * min(1.0, magnitude) if problem == Problem.VOLATILITY else None,
                           failure_reduction_value=info * 0.5 if source in ("loss", "break") else None)


def identity_violations(text: str, tickers: Sequence[str]) -> list[str]:
    """Reasons a piece of text is not identity-free: a year, an ISO date, or any ticker of the day as a word."""
    bad = []
    if _YEAR.search(text):
        bad.append("contains a year")
    if _DATE.search(text):
        bad.append("contains a date")
    words = set(re.findall(r"[A-Za-z0-9_#.-]+", text))
    hit = words & set(map(str, tickers))
    if hit:
        bad.append(f"names {len(hit)} ticker(s)")
    return bad


def _q(text: str, source: str, problem: Problem, rec: ob.DayRecord, created_real: str, success: str, failure: str, n: float,
       magnitude: float, known: set[str], tickers: Sequence[str]) -> ResearchQuestion | None:
    bad = identity_violations(text + " " + success + " " + failure, tickers)
    if bad:
        raise AutopsyError(f"question is not identity-free ({'; '.join(bad)}): {text!r}")
    q = ResearchQuestion.make(text, source, problem, created_real, str(as_date(rec.resolved_at)), success, failure)
    return dataclasses.replace(q, expected=estimate_value(source, n, magnitude, problem, q.question_id in known))


def make_questions(rec: ob.DayRecord, market: MarketSection, model: ModelSection, risk: RiskSection, pats: Sequence[PatternEvent],
                   surprises: Sequence[Surprise], contradictions: Sequence[Contradiction], causes: Sequence[UnknownCause],
                   tracker: sp.SurpriseTracker, p: AutopsyParams, obs: ob.ObserverParams, now, created_real: str,
                   known: set[str]) -> list[ResearchQuestion]:
    tickers = list(rec.rows["ticker"]) if not rec.rows.empty else []
    out: list[ResearchQuestion | None] = []
    def add(text: str, source: str, problem: Problem, *, success: str, failure: str, n: float, magnitude: float) -> None:
        out.append(_q(text, source, problem, rec=rec, created_real=created_real, success=success, failure=failure, n=n,
                      magnitude=magnitude, known=known, tickers=tickers))

    cell = market.cell
    moved = int(rec.market.get("n_moved", 0))
    if model.missed_winners:
        top = max(model.reason_counts.items(), key=lambda kv: kv[1], default=(None, 0))
        total = sum(model.reason_counts.values())
        if top[0] and total >= 2 and top[1] / total >= 0.4:
            add(f"Why does the model reject winners for the reason {top[0]} ({top[1]} of {total} missed winners today, context {cell})?",
                "missed_winner", Problem.COVERAGE, success=f"a separating rule for {top[0]} winners reaches out-of-sample lift >= {lmw_lift()}",
                failure="no rule survives the family-wise permutation test", n=float(total), magnitude=top[1] / total)
    unplaced = moved - rec.n(MC.PREDICTABLE_MOVER) - rec.n(MC.UNPREDICTABLE_MOVER)
    if moved >= 10 and unplaced / moved >= p.unplaced_flag:
        add(f"What separates the {unplaced} of {moved} movers the precursor proxy could not place from those it could (context {cell})?",
            "discovery", Problem.VOLATILITY, success="a feature set that places at least half of the unplaced movers out of sample",
            failure="unplaced movers stay indistinguishable from non-movers on every available feature", n=float(moved), magnitude=unplaced / moved)
    if rec.n(MC.UNPREDICTABLE_MOVER) >= 5:
        add(f"Is any of the {rec.n(MC.UNPREDICTABLE_MOVER)} overnight-gap band moves preceded by a measurable signal in the prior sessions (volume drift, range compression)?",
            "discovery", Problem.VOLATILITY, success="a prior-session signal with mover-rate lift >= 1.5 that replicates across days",
            failure="gap movers are indistinguishable from other names on every prior-session feature", n=float(rec.n(MC.UNPREDICTABLE_MOVER)),
            magnitude=rec.n(MC.UNPREDICTABLE_MOVER) / max(moved, 1))
    for kind, group in _by_kind(contradictions).items():
        if len(group) >= 2 and kind != "market":
            add(f"Why does the system contradict itself in the way '{kind}' ({len(group)} names today, context {cell})?", "contradiction",
                Problem.CONSISTENCY, success="the two disagreeing components are reconciled or one is shown to be wrong on held-out days",
                failure="the disagreement is noise: it does not predict the outcome", n=float(len(group)), magnitude=min(1.0, len(group) / 10))
    if risk.n_losses and risk.avoidable_loss_share == risk.avoidable_loss_share and risk.avoidable_loss_share >= 0.5:
        add(f"Which pre-trade risk signs (top-decile volatility, thin liquidity, a scheduled event, an adverse gap) precede the picks that lose {p.loss_thr:.0%} or more (context {cell})?",
            "loss", Problem.LOSS_AVOIDANCE, success="a risk sign whose presence at least doubles the loss rate out of sample",
            failure="losing picks carry the same signs as winning picks", n=float(risk.n_losses), magnitude=risk.avoidable_loss_share)
    if risk.n_losses >= 2 and risk.unknown_rate == risk.unknown_rate and risk.unknown_rate >= 0.7:
        add(f"Why can the loss classifier name no cause for {risk.unknown_rate:.0%} of today's losing picks?", "loss", Problem.RESEARCH_PROCESS,
            success="the missing inputs (patterns, stops, contexts) are identified and the unknown share falls under half",
            failure="the losses remain unexplained after every detector has its inputs", n=float(risk.n_losses), magnitude=risk.unknown_rate)
    if risk.correlated.get("co_loss_z", 0.0) >= p.co_loss_z:
        add(f"Do the simultaneous losses across the book share a driver (sector, volatility tercile, market tone {market.tone})?", "regime",
            Problem.LOSS_AVOIDANCE, success="one shared driver explains most of the clustering and can be capped",
            failure="the clustering is not repeatable across days", n=risk.correlated.get("n_losers", 0.0),
            magnitude=min(1.0, risk.correlated["co_loss_z"] / 4))
    if risk.gap_risk.get("adverse_entry_gaps", 0) >= 1 or (risk.gap_risk.get("gap_driven_loss_share", 0) or 0) >= 0.5:
        add("Can an entry-timing or gap filter avoid the adverse opening gaps that hit the book, without giving up the winners?", "loss",
            Problem.LOSS_AVOIDANCE, success="a filter that removes at least a third of gap losses while losing under a tenth of winners",
            failure="every gap filter removes winners in proportion", n=float(max(risk.n_losses, 1)),
            magnitude=float(risk.gap_risk.get("gap_driven_loss_share") or 0.3))
    for pe in sorted([e for e in pats if e.kind == "broken"], key=lambda e: -abs((e.effect_before or 0) - (e.effect_after or 0)))[:3]:
        add(f"Why did pattern {pe.pattern_id} stop working (effect {_fmt(pe.effect_before)} to {_fmt(pe.effect_after)}, {pe.n_hit} of {pe.n_fired} hits) in context {pe.cell or cell}?",
            "break", Problem.CONSISTENCY, success="a condition that separates the days it holds from the days it breaks, replicated on other days",
            failure="the break is indistinguishable from sampling noise", n=float(pe.n_fired), magnitude=min(1.0, abs((pe.effect_before or 0) - (pe.effect_after or 0)) * 20))
    for s in surprises:
        add(f"Why did {s.metric} land {s.z:+.1f} robust sigma from its trailing norm in context {cell}?", "surprise", Problem.VOLATILITY if "mover" in s.metric else Problem.CONSISTENCY,
            success="a context variable that predicts the deviation on later days", failure="the deviation does not repeat under the same context",
            n=float(rec.n_universe), magnitude=min(1.0, abs(s.z) / 6))
    for item in sp.research_questions(tracker, now, top=2):
        add(f"{item['question']} (surprise priority {item['priority']:.2f})", "surprise", Problem.CONSISTENCY,
            success="the repeated surprise is explained by a context split that holds out of sample", failure="the surprise cell stops being surprising without explanation",
            n=float(len(tracker)), magnitude=min(1.0, item["priority"]))
    hot = [(k, v) for k, v in market.band_z.items() if v == v and v >= p.z_unusual]
    if hot:
        k, v = max(hot, key=lambda kv: kv[1])
        add(f"Is today's unusually large {k.replace('_', ' ')} mover count ({rec.bands[k]} names, {v:+.1f} sigma above trend) a regime signal that persists over the following sessions?",
            "regime", Problem.VOLATILITY, success="the elevated count predicts elevated counts over the next five sessions better than the trailing median",
            failure="the count reverts at once", n=float(rec.bands[k]), magnitude=min(1.0, v / 6))
    if rec.market.get("n_suspect_band", 0) and rec.market["n_suspect_band"] / max(1, sum(rec.bands.values())) >= 0.1:
        add("Which data feed produced the split-like, extreme or no-trade prints among today's band movers?", "data", Problem.DATA_QUALITY,
            success="the offending names are traced to an adjustment or feed fault and excluded from mover studies", failure="the prints are genuine market events",
            n=float(rec.market["n_suspect_band"]), magnitude=rec.market["n_suspect_band"] / max(1, sum(rec.bands.values())))
    if moved >= 10 and model.score_auc == model.score_auc and model.score_auc <= 0.5:
        add("Does the model score order movers at all on the whole universe, or only inside the names it already ranks highly?", "discovery",
            Problem.VOLATILITY, success="a decile table over several days with top-decile mover rate above bottom-decile",
            failure="score deciles show no ordering", n=float(moved), magnitude=0.6)
    ab = model.abstention
    if ab.get("n", 0) >= 3 and ab.get("moved_share", 0) == ab.get("moved_share", 0) and ab["moved_share"] >= 0.3:
        add(f"Are abstentions leaving movers on the table ({ab['moved_share']:.0%} of {int(ab['n'])} abstained names moved)?", "missed_winner",
            Problem.COVERAGE, success="abstained names move at a rate above the base rate over several days", failure="abstentions move no more than the base rate",
            n=ab["n"], magnitude=ab["moved_share"])
    return [q for q in out if q is not None]


def _by_kind(cs: Sequence[Contradiction]) -> dict[str, list[Contradiction]]:
    out: dict[str, list[Contradiction]] = {}
    for c in cs:
        out.setdefault(c.kind, []).append(c)
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def lmw_lift() -> str:
    return f"{ob.lmw.MissedParams().min_lift:g}"


# ==================================================================================================================
# RESEARCH assembly
# ==================================================================================================================
def research_section(rec: ob.DayRecord, ledger: ob.ObserverLedger, ctx: AutopsyContext, market: MarketSection, model: ModelSection,
                     risk: RiskSection, tracker: sp.SurpriseTracker, p: AutopsyParams, obs: ob.ObserverParams, now, created_real: str,
                     known: set[str]) -> ResearchSection:
    pats = tuple(ctx.patterns)
    surprises, skipped = observe_surprises(rec, ledger, tracker, market.cell, p, now)
    contra = find_contradictions(rec, ctx, obs, p)
    sig = ob.precursor_significance(ledger, "precursor_auc", now)
    causes = unknown_causes(rec, bool(sig.get("informative")))
    qs = make_questions(rec, market, model, risk, pats, surprises, contra, causes, tracker, p, obs, now, created_real, known)
    seen, uniq = set(), []
    for q in qs:
        if q.question_id not in seen:
            seen.add(q.question_id)
            uniq.append(q)
    invs = tuple(i.reason for i in tracker.investigations(now)[:p.top_n])
    return ResearchSection(tuple(e for e in pats if e.kind in ("new", "revived", "strengthened")),
                           tuple(e for e in pats if e.kind in ("broken", "weakened")), tuple(e for e in pats if e.kind == "held"),
                           tuple(surprises), tuple(skipped), tuple(contra), tuple(causes), bool(sig.get("informative")),
                           tuple(uniq[:p.max_questions]), invs)


# ==================================================================================================================
# LEARNING
# ==================================================================================================================
@dataclass(frozen=True)
class LearningSection:
    knowledge_changes: tuple[dict[str, Any], ...]
    confidence_changes: tuple[dict[str, Any], ...]
    priority_changes: tuple[dict[str, Any], ...]
    n_new_questions: int
    n_duplicate_questions: int
    next_experiments: tuple[dict[str, Any], ...]
    nothing_changed: bool


def learning_section(ctx: AutopsyContext, questions: Sequence[ResearchQuestion], known: set[str], p: AutopsyParams) -> LearningSection:
    """What the day changed: knowledge items that appeared / vanished / changed lifecycle, confidence moves, priority moves, and
    the experiments worth running next (the new questions ranked by expected decision value per compute minute)."""
    kb, ka = ctx.knowledge_before, ctx.knowledge_after
    kn: list[dict[str, Any]] = []
    for kid in sorted(set(kb) | set(ka)):
        b, a = kb.get(kid), ka.get(kid)
        if b is None and a is not None:
            kn.append({"kid": kid, "change": "added", "lifecycle": a.lifecycle})
        elif a is None and b is not None:
            kn.append({"kid": kid, "change": "removed", "lifecycle": b.lifecycle})
        elif a is not None and b is not None and a.lifecycle != b.lifecycle:
            kn.append({"kid": kid, "change": "lifecycle", "from": b.lifecycle, "to": a.lifecycle})
    cc: list[dict[str, Any]] = []
    for kid in sorted(set(kb) & set(ka)):
        cb, ca = kb[kid].confidence, ka[kid].confidence
        if cb is not None and ca is not None and abs(ca - cb) >= 0.05:
            cc.append({"kid": kid, "before": cb, "after": ca, "delta": ca - cb})
    cc.sort(key=lambda d: -abs(d["delta"]))
    pc: list[dict[str, Any]] = []
    for k in sorted(set(ctx.priorities_before) | set(ctx.priorities_after)):
        pb, pa = ctx.priorities_before.get(k), ctx.priorities_after.get(k)
        if pb is None or pa is None or abs(pa - pb) >= 0.05:
            pc.append({"item": k, "before": pb, "after": pa})
    fresh = [q for q in questions if q.question_id not in known and q.question_id not in set(ctx.running)]
    ranked = sorted(fresh, key=lambda q: -((q.expected.decision_value or 0.0) * (q.expected.information_gain or 0.0)
                                            / max(q.expected.compute_cost or 1.0, 1e-9)))
    nxt = tuple({"question_id": q.question_id, "text": q.text, "problem": q.problem.value,
                 "value_per_minute": (q.expected.decision_value or 0.0) * (q.expected.information_gain or 0.0) / max(q.expected.compute_cost or 1.0, 1e-9)}
                for q in ranked[:p.max_experiments])
    return LearningSection(tuple(kn[:20]), tuple(cc[:p.top_n * 2]), tuple(pc[:p.top_n * 2]), len(fresh), len(questions) - len(fresh), nxt,
                           not (kn or cc or pc or fresh))


# ==================================================================================================================
# the report and the public entry
# ==================================================================================================================
@dataclass(frozen=True)
class Autopsy:
    day: str
    resolved_at: str
    params_hash: str
    empty: bool
    market: MarketSection | None
    model: ModelSection | None
    research: ResearchSection | None
    risk: RiskSection | None
    learning: LearningSection | None

    def sections(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in SECTIONS}

    @property
    def sec_market(self) -> MarketSection:
        return self.parts()[0]

    @property
    def sec_model(self) -> ModelSection:
        return self.parts()[1]

    @property
    def sec_research(self) -> ResearchSection:
        return self.parts()[2]

    @property
    def sec_risk(self) -> RiskSection:
        return self.parts()[3]

    def parts(self) -> tuple[MarketSection, ModelSection, ResearchSection, RiskSection, LearningSection]:
        """The five sections of a non-empty autopsy, narrowed for callers that render it."""
        if self.market is None or self.model is None or self.research is None or self.risk is None or self.learning is None:
            raise AutopsyError(f"autopsy {self.day} has a missing section")
        return self.market, self.model, self.research, self.risk, self.learning

    def questions(self) -> tuple[ResearchQuestion, ...]:
        return self.research.questions if self.research else ()

    def validate(self) -> list[str]:
        e = []
        if not self.empty:
            e += [f"section {k} missing" for k, v in self.sections().items() if v is None]
        for q in self.questions():
            bad = identity_violations(q.text, [])
            if bad:
                e.append(f"question {q.question_id}: {'; '.join(bad)}")
        return e

    def as_matured(self, prov: Provenance) -> MaturedRecord:
        """Identity-free digest for the research world: counts and question ids, never tickers."""
        m = self.market
        return MaturedRecord(stable_hash([self.day, self.params_hash, "autopsy"], 16), self.resolved_at,
                             {"day": self.day, "bands": dict(m.bands) if m else {}, "cell": m.cell if m else "",
                              "questions": [q.question_id for q in self.questions()],
                              "n_losses": self.risk.n_losses if self.risk else 0}, prov)


@dataclass
class AutopsyState:
    """Carried between days: the observer ledger (past days only), the surprise tracker and the question ids already raised."""
    params: AutopsyParams = field(default_factory=AutopsyParams)
    obs_params: ob.ObserverParams = field(default_factory=ob.ObserverParams)
    observer: ob.ObserverState | None = None
    tracker: sp.SurpriseTracker = field(default_factory=sp.SurpriseTracker)
    classifier: fl.LossClassifier = field(default_factory=fl.LossClassifier)
    known_questions: set = field(default_factory=set)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.params.require_valid()
        if self.observer is None:
            self.observer = ob.ObserverState(self.obs_params)


def autopsy_day(rec: ob.DayRecord, ledger: ob.ObserverLedger, state: AutopsyState, now, created_real: str,
                ctx: AutopsyContext | None = None) -> Autopsy:
    """Run the five sections on one DayRecord. `ledger` must hold only days BEFORE this one (the caller adds the day after)."""
    p, obs = state.params, state.obs_params
    require_past(rec.resolved_at, now, "autopsy day resolved_at")
    ctx = ctx or AutopsyContext()
    errs = ctx.validate()
    if errs:
        raise AutopsyError("; ".join(errs))
    if rec.empty:
        return Autopsy(rec.day, rec.resolved_at, p.hash(), True, None, None, None, None, None)
    mk = market_section(rec, ledger, p, obs)
    md = model_section(rec, p, obs)
    rk = risk_section(rec, ctx, p, obs, state.classifier, now)
    rs = research_section(rec, ledger, ctx, mk, md, rk, state.tracker, p, obs, now, created_real, state.known_questions)
    ln = learning_section(ctx, rs.questions, state.known_questions, p)
    state.known_questions |= {q.question_id for q in rs.questions}
    a = Autopsy(rec.day, rec.resolved_at, p.hash(), False, mk, md, rs, rk, ln)
    bad = a.validate()
    if bad:
        raise AutopsyError("; ".join(bad))
    return a


def step(state: AutopsyState, snap: ob.DecisionSnapshot, out: ob.DayOutcome, now, created_real: str, ctx: AutopsyContext | None = None) -> Autopsy:
    """PUBLIC ENTRY. Observe one matured day (engine.research.observer) and autopsy it; the day is appended to the observer ledger
    only after the autopsy, so a day is never compared with itself. Returns the Autopsy; its questions feed the research queue."""
    rec = ob.observe_day(snap, out, now, state.obs_params)
    assert state.observer is not None and state.observer.ledger is not None     # both set by the states' __post_init__
    a = autopsy_day(rec, state.observer.ledger, state, now, created_real, ctx)
    state.observer.ledger.add(rec)
    state.history.append(a)
    return a


def render(a: Autopsy, top: int = 3) -> str:
    """Plain-text autopsy. Research-side (shows tickers as cids only)."""
    if a.empty:
        return f"autopsy {a.day}: empty universe"
    m, md, rs, rk, ln = a.parts()
    L = [f"AUTOPSY {a.day}  cell {m.cell}"]
    L.append("MARKET  bands " + ", ".join(f"{k} {v}" for k, v in m.bands.items()) + f"; unusual: {', '.join(m.unusual) or 'none'}")
    for k, ents in m.lists().items():
        L.append(f"  {k}: " + ", ".join(f"{e.cid[:6]}({e.value:+.3f})" for e in ents[:top]))
    L.append(f"MODEL   picked {md.n_picked}, score AUC {md.score_auc:.3f}, precursor AUC {md.precursor_auc:.3f}, MCC {md.mcc:.3f}, "
             f"missed winners {len(md.missed_winners)} reasons {dict(md.reason_counts)}")
    L.append(f"RESEARCH new {len(rs.new_patterns)}, broken {len(rs.broken_patterns)}, surprises {len(rs.surprises)}, "
             f"contradictions {len(rs.contradictions)}, unknown causes " + ", ".join(f"{u.verdict.value}:{u.count}" for u in rs.unknown_causes))
    for q in rs.questions[:top]:
        L.append(f"  ? {q.text}")
    L.append(f"RISK    losses {rk.n_losses}/{rk.n_positions}, avoidable share {rk.avoidable_loss_share:.2f}, flags: {', '.join(rk.flags) or 'none'}")
    L.append(f"LEARNING new questions {ln.n_new_questions}, next experiments {len(ln.next_experiments)}" + (" (nothing changed)" if ln.nothing_changed else ""))
    return "\n".join(L)


# ==================================================================================================================
# across days: the autopsy ledger, question queue, trends, exports, audits
# ==================================================================================================================
def _entry_dict(e: Entry, keep_ticker: bool) -> dict[str, Any]:
    d = e.identity_free()
    if keep_ticker:
        d["ticker"] = e.ticker
    return d


def _plain(x: Any, keep_ticker: bool) -> Any:
    """JSON-safe copy of any autopsy structure. keep_ticker=False drops every ticker: the export the trader side may one day read."""
    if isinstance(x, Entry):
        return _entry_dict(x, keep_ticker)
    if isinstance(x, LossItem):
        d = {k: _plain(v, keep_ticker) for k, v in dataclasses.asdict(x).items()}
        if not keep_ticker:
            d.pop("ticker", None)
        return d
    if isinstance(x, ResearchQuestion):
        return {"question_id": x.question_id, "text": x.text, "source": x.source, "problem": x.problem.value,
                "success": x.success_criterion, "failure": x.failure_criterion,
                "expected": {k: v for k, v in dataclasses.asdict(x.expected).items() if v is not None}}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name), keep_ticker) for f in dataclasses.fields(x)}
    if hasattr(x, "value") and not isinstance(x, (int, float, str)):
        return x.value
    if isinstance(x, Mapping):
        return {str(k): _plain(v, keep_ticker) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_plain(v, keep_ticker) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def to_dict(a: Autopsy, keep_ticker: bool = False) -> dict[str, Any]:
    """The autopsy as plain JSON-able data. With keep_ticker=False it carries salted cids only, and identity_leaks() must be empty."""
    return {"day": a.day if keep_ticker else "", "params_hash": a.params_hash, "empty": a.empty,
            **{k: _plain(v, keep_ticker) for k, v in a.sections().items()}}


def identity_leaks(payload: Any, tickers: Sequence[str]) -> list[str]:
    """Search an exported structure for any ticker string or a year/date pattern. A non-empty list is a leak."""
    import json
    text = json.dumps(payload, default=str)
    bad = identity_violations(text, tickers)
    return bad


def to_markdown(a: Autopsy, top: int = 5) -> str:
    """The report a person reads: the five sections with tables. Research-side (cids, not tickers)."""
    if a.empty:
        return f"# Autopsy {a.day}\n\nEmpty universe; nothing was recorded.\n"
    m, md, rs, rk, ln = a.parts()
    o = [f"# Autopsy {a.day}", "", f"Context cell: `{m.cell}`", "", "## Market", "",
         "| band | close-to-close | open-to-close | eligible | z vs trend |", "|---|---:|---:|---:|---:|"]
    for k in O_BANDS:
        z = m.band_z.get(k, float("nan"))
        o.append(f"| {k} | {m.bands[k]} | {m.bands_o2c[k]} | {m.bands_eligible[k]} | {'n/a' if z != z else f'{z:+.1f}'} |")
    o.append("")
    for k, ents in m.lists().items():
        o.append(f"- **{k}**: " + (", ".join(f"`{e.cid[:8]}` {e.value:+.3f}" for e in ents[:top]) or "none"))
    o += ["", f"Unusual today: {', '.join(m.unusual) or 'nothing'}. Data-suspect band movers: {m.n_suspect_band}.", "", "## Model", "",
          f"Picked {md.n_picked}; score AUC {md.score_auc:.3f}, precursor AUC {md.precursor_auc:.3f}, MCC {md.mcc:.3f}, recall {md.recall:.3f}, "
          f"precision {md.precision:.3f}, decile lift {md.decile_lift:.2f}.", ""]
    for title, ents in (("Best", md.best), ("Worst", md.worst), ("Missed winners", md.missed_winners), ("Missed losers", md.missed_losers),
                        ("Confident mistakes", md.confident_mistakes), ("Unconfident successes", md.unconfident_successes)):
        o.append(f"- **{title}**: " + (", ".join(f"`{e.cid[:8]}` {e.value:+.3f}" + (f" ({e.extra['reason']})" if 'reason' in e.extra else "") for e in ents[:top]) or "none"))
    o += ["", "## Research", "", f"New patterns {len(rs.new_patterns)}, broken {len(rs.broken_patterns)}, surprises {len(rs.surprises)}, "
          f"contradictions {len(rs.contradictions)}.", ""]
    for u in rs.unknown_causes:
        o.append(f"- {u.verdict.value}: {u.count} movers ({u.share_of_movers:.0%}) - {u.note}")
    o += ["", "Questions:", ""] + [f"{i}. {q.text}" for i, q in enumerate(rs.questions, 1)]
    o += ["", "## Risk", "", f"{rk.n_losses} of {rk.n_positions} positions lost; avoidable share {rk.avoidable_loss_share:.2f}; "
          f"flags: {', '.join(rk.flags) or 'none'}.", "", f"Concentration {dict(rk.concentration)}", f"Gap risk {dict(rk.gap_risk)}", "",
          "## Learning", "", f"{ln.n_new_questions} new questions ({ln.n_duplicate_questions} already known)."]
    o += [f"- next: {e['text']} (value/min {e['value_per_minute']:.3f})" for e in ln.next_experiments]
    return "\n".join(o) + "\n"


O_BANDS = tuple(ob.BAND_NAMES.values())


@dataclass(frozen=True)
class QueuedQuestion:
    question: ResearchQuestion
    first_day: str
    last_day: str
    times_raised: int
    score: float


class QuestionQueue:
    """The autonomous research queue's intake from the autopsy. A question raised again on a later day is the SAME question
    (its id hashes text, source, problem and evidence date, so identical text on a new evidence date is new by design); repeats
    within the text-only key raise its score, because a question the market keeps asking is worth more than one it asked once."""

    def __init__(self):
        self._q: dict[str, QueuedQuestion] = {}
        self._text_key: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._q)

    @staticmethod
    def _key(q: ResearchQuestion) -> str:
        norm = re.sub(r"\d+(\.\d+)?%?", "#", q.text)
        norm = re.sub(r"context [^)]*", "context", norm)
        return stable_hash([norm, q.source, q.problem.value], 12)

    @staticmethod
    def _value(q: ResearchQuestion) -> float:
        e = q.expected
        return (e.decision_value or 0.0) * (0.5 + (e.information_gain or 0.0)) * (1.0 - (e.redundancy or 0.0)) / max(e.compute_cost or 1.0, 1.0)

    def add(self, q: ResearchQuestion, day: str) -> QueuedQuestion:
        key = self._key(q)
        old = self._q.get(self._text_key.get(key, ""))
        if old is None:
            item = QueuedQuestion(q, day, day, 1, self._value(q))
            self._q[q.question_id] = item
            self._text_key[key] = q.question_id
            return item
        n = old.times_raised + 1
        new = QueuedQuestion(old.question, old.first_day, day, n, self._value(old.question) * (1.0 + math.log(n)))
        self._q[old.question.question_id] = new
        return new

    def add_autopsy(self, a: Autopsy) -> list[QueuedQuestion]:
        return [self.add(q, a.day) for q in a.questions()]

    def top(self, k: int = 10, problem: Problem | None = None) -> list[QueuedQuestion]:
        items = [i for i in self._q.values() if problem is None or i.question.problem == problem]
        return sorted(items, key=lambda i: (-i.score, i.question.question_id))[:k]

    def by_problem(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self._q.values():
            out[i.question.problem.value] = out.get(i.question.problem.value, 0) + 1
        return out

    def stale(self, today: str, days: int = 30) -> list[QueuedQuestion]:
        cut = as_date(today)
        return [i for i in self._q.values() if (cut - as_date(i.last_day)).days > days]


class AutopsyLedger:
    """A run of autopsies in day order, for trends. Holds the compact numbers, not the entry lists."""

    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, a: Autopsy) -> None:
        if self.rows and as_date(a.day) <= as_date(self.rows[-1]["day"]):
            raise AutopsyError(f"autopsy for {a.day} is not after {self.rows[-1]['day']}")
        if a.empty:
            self.rows.append({"day": a.day, "empty": True})
            return
        m, md, rs, rk = a.sec_market, a.sec_model, a.sec_research, a.sec_risk
        self.rows.append({"day": a.day, "empty": False, **{"band_" + k: v for k, v in m.bands.items()},
                          "n_suspect_band": m.n_suspect_band, "n_unusual": len(m.unusual), "score_auc": md.score_auc,
                          "precursor_auc": md.precursor_auc, "mcc": md.mcc, "recall": md.recall, "n_missed_winners": len(md.missed_winners),
                          "n_contradictions": len(rs.contradictions), "n_surprises": len(rs.surprises), "n_questions": len(rs.questions),
                          "n_losses": rk.n_losses, "avoidable_share": rk.avoidable_loss_share, "unknown_rate": rk.unknown_rate,
                          "co_loss_z": rk.correlated.get("co_loss_z", float("nan")), "n_flags": len(rk.flags),
                          "top_reason": max(md.reason_counts.items(), key=lambda kv: kv[1])[0] if md.reason_counts else ""})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("day") if self.rows else pd.DataFrame()

    def reason_trend(self) -> pd.DataFrame:
        """Share of days each missed-winner reason was the top one, first half vs second half: a shifting reason is a shifting
        failure mode, and a reason that never changes is a standing hole."""
        f = self.frame()
        if f.empty or "top_reason" not in f:
            return pd.DataFrame()
        f = f[~f["empty"].astype(bool) & (f["top_reason"] != "")]
        if len(f) < 4:
            return pd.DataFrame()
        h = len(f) // 2
        a, b = f["top_reason"].iloc[:h].value_counts(normalize=True), f["top_reason"].iloc[h:].value_counts(normalize=True)
        return pd.DataFrame({"first_half": a, "second_half": b}).fillna(0.0).assign(shift=lambda d: d["second_half"] - d["first_half"])

    def score_skill(self) -> dict[str, float]:
        """Does the model score beat a coin, and beat the precursor proxy, over days? Day-level paired comparison."""
        f = self.frame()
        if f.empty or "score_auc" not in f:
            return {"days": 0.0}
        d = f[["score_auc", "precursor_auc"]].dropna()
        if len(d) < 5:
            return {"days": float(len(d))}
        diff = d["score_auc"] - d["precursor_auc"]
        sd = float(diff.std(ddof=1))
        return {"days": float(len(d)), "score_auc": float(d["score_auc"].mean()), "precursor_auc": float(d["precursor_auc"].mean()),
                "score_minus_precursor_t": float(diff.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("nan")}

    def band_regimes(self, z: float = 2.0) -> list[str]:
        """Days on which any band's count sat beyond `z` robust sigmas of the trailing days (computed here from the ledger alone)."""
        f = self.frame()
        cols = [c for c in f.columns if c.startswith("band_")]
        if f.empty or len(f) < 12 or not cols:
            return []
        tot = f[cols].sum(axis=1)
        med = tot.expanding(min_periods=10).median().shift(1)
        mad = (tot - tot.expanding(min_periods=10).median().shift(1)).abs().expanding(min_periods=10).median().shift(1) * 1.4826
        zs = (tot - med) / mad.clip(lower=1.0)
        return [str(d) for d in zs[zs.abs() >= z].index]


def coverage_audit(a: Autopsy, rec: ob.DayRecord) -> list[str]:
    """Does the autopsy account for the record it was built from? Every band mover must be reachable (as a row) and every
    section that has its inputs must have produced output. Returns a list of problems (empty = consistent)."""
    bad = []
    if a.empty:
        return [] if rec.empty else ["autopsy is empty but the record is not"]
    if a.sec_market.bands != dict(rec.bands):
        bad.append("autopsy band counts differ from the record")
    if rec.model.get("n_picked", 0) and a.sec_risk.n_positions != rec.model["n_picked"]:
        bad.append("risk section did not see every pick")
    if a.sec_risk.n_losses > a.sec_risk.n_positions:
        bad.append("more losses than positions")
    n_l = len(a.sec_risk.avoidable) + len(a.sec_risk.unavoidable) + len(a.sec_risk.undetermined)
    if n_l > a.sec_risk.n_losses:
        bad.append("more listed losses than counted losses")
    if any(u.count > rec.market.get("n_moved", 0) for u in a.sec_research.unknown_causes):
        bad.append("an unknown-cause bucket exceeds the number of movers")
    if sum(u.count for u in a.sec_research.unknown_causes) > rec.market.get("n_moved", 0) + 1e-9:
        bad.append("unknown-cause buckets overlap")
    for q in a.sec_research.questions:
        if not q.success_criterion or not q.failure_criterion:
            bad.append(f"question {q.question_id} lacks a success or failure criterion")
    return bad


def replay_digest(a: Autopsy) -> str:
    """Hash of an autopsy's identity-free content: the same day and history must always give the same digest."""
    return stable_hash(to_dict(a, keep_ticker=False), 16)


def run_days(days, state: AutopsyState, queue: QuestionQueue | None = None, alog: AutopsyLedger | None = None, now_of=None,
             created_real: str = "", ctx_of=None):
    """Stream (snapshot, outcome) days through step(), feeding the queue and the ledger. Yields each Autopsy as it is made."""
    import datetime as _dt
    for snap, out in days:
        now = now_of(out) if now_of else as_date(out.resolved_at) + _dt.timedelta(days=1)
        a = step(state, snap, out, now, created_real or str(now), ctx_of(snap) if ctx_of else None)
        if queue is not None:
            queue.add_autopsy(a)
        if alog is not None:
            alog.add(a)
        yield a


# ==================================================================================================================
# adapters from the existing stores, what-ifs, grading against planted truth
# ==================================================================================================================
def pattern_events(before: pd.DataFrame | None, after: pd.DataFrame | None, min_change: float = 0.005) -> list[PatternEvent]:
    """Diff two snapshots of engine.patterns.PatternMiner's `.patterns` table (columns: key_named, effect, p_real, status) taken
    before and after the day. A pattern that is new, that left the active set (broken), that came back (revived) or whose effect
    moved by at least `min_change` (strengthened / weakened) becomes a PatternEvent. Nothing is re-mined here."""
    def prep(df):
        if df is None or df.empty:
            return {}
        cols = {c: df[c] for c in ("effect", "p_real", "status") if c in df}
        idx = df["key_named"] if "key_named" in df else pd.Series(df.index.astype(str), index=df.index)
        return {str(k): {c: cols[c].iloc[i] for c in cols} for i, k in enumerate(idx)}
    b, a = prep(before), prep(after)
    out = []
    active = lambda d: str(d.get("status", "active")) in ("active", "rescoped")
    for k in sorted(set(a) | set(b)):
        x, y = b.get(k), a.get(k)
        eb = _nz(x.get("effect")) if x else None
        ea = _nz(y.get("effect")) if y else None
        pr = _nz(y.get("p_real")) if y else None
        if x is None and y is not None:
            kind = "new" if active(y) else "held"
        elif y is None:
            kind = "broken"
        elif active(x) and not active(y):
            kind = "broken"
        elif not active(x) and active(y):
            kind = "revived"
        elif eb is not None and ea is not None and ea - eb >= min_change:
            kind = "strengthened"
        elif eb is not None and ea is not None and eb - ea >= min_change:
            kind = "weakened"
        else:
            kind = "held"
        sign = 0 if ea is None or ea == 0 else int(np.sign(ea))
        out.append(PatternEvent(k, kind, eb, ea, pr, expected_sign=sign, note=str((y or x or {}).get("status", ""))))
    return out


def knowledge_states(items: Sequence[Any]) -> dict[str, KnowledgeState]:
    """KnowledgeLike objects (engine.learning.core duck type) as the autopsy's KnowledgeState. Confidence is the truth dimension
    when present, else current reliability; None stays None."""
    out = {}
    for k in items:
        c = getattr(k, "confidence", None)
        conf = None
        if c is not None:
            conf = _nz(getattr(c, "truth", None))
            if conf is None:
                conf = _nz(getattr(c, "current_reliability", None))
        out[str(k.knowledge_id)] = KnowledgeState(str(k.knowledge_id), conf, str(getattr(k, "lifecycle", "")), None)
    return out


def avoidable_loss_what_if(a: Autopsy, rec: ob.DayRecord, p: AutopsyParams) -> dict[str, float]:
    """If every pick carrying two or more visible risk signs had been skipped, what would today's book have earned, and how many
    winners would have gone with them? Descriptive, one day, hindsight-free in its rule (signs are decision-time) - and a single
    day proves nothing; it exists so the research queue can accumulate the counts across days."""
    rf = rec.rows
    picks = rf[rf["picked"].astype(bool)] if not rf.empty else rf
    if picks.empty:
        return {"n_picks": 0.0}
    skip = np.array([len(preflight_flags(r, p, ob.ObserverParams())) >= 2 for r in picks.itertuples(index=False)])
    ret = picks["ret"].astype(float).to_numpy() - p.cost
    ok = np.isfinite(ret)
    kept = ~skip & ok
    win = ret >= 0.07
    return {"n_picks": float(len(picks)), "n_skipped": float(skip.sum()), "mean_all": float(np.mean(ret[ok])) if ok.any() else float("nan"),
            "mean_kept": float(np.mean(ret[kept])) if kept.any() else float("nan"),
            "losses_avoided": float(((ret <= -p.loss_thr) & skip).sum()), "winners_lost": float((win & skip).sum())}


def grade(a: Autopsy, truth: Mapping[str, Sequence[str]], rec: ob.DayRecord) -> dict[str, float]:
    """Score an autopsy against a planted world's ground truth (group name -> tickers): did the missed-winner list contain the planted
    unpicked winners, did the false-positive group land in the model's worst list, did the loud-precursor movers avoid the
    'unpredictable' bucket? Used by the tests; also the self-check a wave-2 harness can run on every code change."""
    rows = rec.rows.set_index("ticker")
    by = lambda cids: {rows.index[rows["cid"] == c][0] for c in cids if (rows["cid"] == c).any()}
    mw = by(e.cid for e in a.sec_model.missed_winners)
    out = {}
    unp = set(truth.get("pred", ()))
    out["predictable_in_missed"] = float(len(unp & mw) / max(1, min(len(unp), len(mw)))) if unp else float("nan")
    fp = set(truth.get("fp", ()))
    worst = by(e.cid for e in a.sec_model.worst)
    out["fp_in_worst"] = float(len(fp & worst) / max(1, min(len(fp), len(worst)))) if fp else float("nan")
    unpred_flagged = int(sum(1 for t in truth.get("pred", ()) if t in rows.index and ob.has(rows.at[t, "flags"], MC.UNPREDICTABLE_MOVER)))
    out["predictable_called_unpredictable"] = float(unpred_flagged)
    return out


def mover_type_recall(ledger: ob.ObserverLedger, now=None) -> pd.DataFrame:
    """Recall of band movers by mover type: which kinds of mover does the model see and which does it never pick? Band rows are
    uncapped, so this is exact over the whole universe for every band mover."""
    rf = ledger.rows_frame(now)
    if rf.empty:
        return pd.DataFrame(columns=["movers", "picked", "recall"])
    b = rf[rf["band"] != 0]
    if b.empty:
        return pd.DataFrame(columns=["movers", "picked", "recall"])
    g = b.groupby("mover_type").agg(movers=("cid", "size"), picked=("picked", "sum"))
    g["recall"] = g["picked"] / g["movers"]
    return g.sort_values("movers", ascending=False)


def band_persistence(ledger: ob.ObserverLedger, lag: int = 1, now=None) -> dict[str, float]:
    """Lag-`lag` autocorrelation of the daily total band-mover share: do busy mover days cluster? A high value says a mover
    count today is informative about tomorrow's count (regime), zero says days are exchangeable."""
    cf = ledger.counts_frame(now)
    if cf.empty:
        return {"n_days": 0.0, "autocorr": float("nan")}
    cols = [c for c in cf.columns if c.startswith("band_")]
    x = (cf[cols].sum(axis=1) / cf["n_universe"].clip(lower=1)).to_numpy(dtype=float)
    if len(x) <= lag + 3 or np.ptp(x) == 0:
        return {"n_days": float(len(x)), "autocorr": float("nan")}
    return {"n_days": float(len(x)), "autocorr": float(np.corrcoef(x[:-lag], x[lag:])[0, 1]), "mean_share": float(x.mean())}


def regime_label(m: MarketSection) -> str:
    """One word for the day from its market section: storm (many unusual band/market stats), trend, calm or mixed."""
    if len(m.unusual) >= 3:
        return "storm"
    if m.tone in ("up", "down") and m.symmetry.get("band_up_down_ratio", 1.0) not in (float("nan"),):
        r = m.symmetry.get("band_up_down_ratio", 1.0)
        if r == r and (r > 2.0 or r < 0.5):
            return "trend"
    return "calm" if not m.unusual else "mixed"


def severity(a: Autopsy) -> float:
    """A 0..1 score of how much this day deserves attention, for ordering the review pile: unusual statistics, contradictions,
    surprises, avoidable losses and risk flags each add, saturating. It reorders reading, it never gates research."""
    if a.empty:
        return 0.0
    x = (min(len(a.sec_market.unusual), 4) / 4 * 0.25 + min(len(a.sec_research.contradictions), 6) / 6 * 0.2 + min(len(a.sec_research.surprises), 3) / 3 * 0.2
         + (a.sec_risk.avoidable_loss_share if a.sec_risk.avoidable_loss_share == a.sec_risk.avoidable_loss_share else 0.0) * 0.2
         + min(len(a.sec_risk.flags), 3) / 3 * 0.15)
    return float(min(1.0, x))


# ==================================================================================================================
# deeper model / risk / research diagnostics
# ==================================================================================================================
def confidence_calibration(rec: ob.DayRecord, obs: ob.ObserverParams, bins: Sequence[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> pd.DataFrame:
    """Among rows the model gave a confidence, how often did a >= mover_thr move (either way, from the fill) follow, by confidence
    bin? Confidence that does not rise with the hit rate is decoration. Only rows exist for exceptions, so this is the calibration of
    the model's *considered* names - which is the population its confidence was meant for."""
    rf = rec.rows
    cols = ["n", "hit_rate", "mean_conf"]
    if rf.empty:
        return pd.DataFrame(columns=cols)
    d = rf[np.isfinite(rf["confidence"].astype(float)) & np.isfinite(rf["exc"].astype(float))]
    if d.empty:
        return pd.DataFrame(columns=cols)
    cut = pd.cut(d["confidence"].astype(float), list(bins), right=False)
    g = d.assign(hit=(d["exc"].astype(float) >= obs.mover_thr)).groupby(cut, observed=True)
    return pd.DataFrame({"n": g.size(), "hit_rate": g["hit"].mean(), "mean_conf": g["confidence"].mean()})


def confidence_calibration_ledger(ledger: ob.ObserverLedger, obs: ob.ObserverParams, now=None) -> dict[str, float]:
    """Pooled over days: Spearman between confidence and outcome (mover or not) on considered names, and the Brier score of
    treating confidence as the probability of a mover, against the constant base-rate forecast. Brier skill <= 0 = no information."""
    rf = ledger.rows_frame(now)
    if rf.empty:
        return {"n": 0.0}
    d = rf[np.isfinite(rf["confidence"].astype(float)) & np.isfinite(rf["exc"].astype(float))]
    if len(d) < 20:
        return {"n": float(len(d))}
    y = (d["exc"].astype(float) >= obs.mover_thr).to_numpy(float)
    c = np.clip(d["confidence"].astype(float).to_numpy(), 0.0, 1.0)
    base = float(y.mean())
    bs, bs0 = float(np.mean((c - y) ** 2)), float(np.mean((base - y) ** 2))
    rho = float(pd.Series(c).corr(pd.Series(y), method="spearman")) if y.std() > 0 and c.std() > 0 else float("nan")
    return {"n": float(len(d)), "base_rate": base, "brier": bs, "brier_base": bs0, "brier_skill": 1.0 - bs / bs0 if bs0 > 0 else float("nan"), "spearman": rho}


def loss_clusters(rec: ob.DayRecord, p: AutopsyParams) -> list[dict[str, Any]]:
    """Which shared attributes do the day's losing picks have more often than the winning ones? Compares loser and non-loser
    picks on volatility rank, liquidity rank, mover type and sector; a group of at least three losers that all share one
    attribute with a non-loser rate at least 40 points lower is reported. Descriptive only: three names are not evidence."""
    rf = rec.rows
    picks = rf[rf["picked"].astype(bool)] if not rf.empty else rf
    if len(picks) < 5:
        return []
    lose = (picks["ret"].astype(float) <= -p.loss_thr).to_numpy()
    if lose.sum() < 3:
        return []
    out = []
    for name, series in (("mover_type", picks["mover_type"].astype(str)), ("sector", picks["sector"].astype(int).astype(str)),
                         ("vol_tercile", pd.cut(picks["rk_vol20"].astype(float), [0, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "high"]).astype(str)
                          if "rk_vol20" in picks else pd.Series("na", index=picks.index))):
        for val, idx in series.groupby(series).groups.items():
            m = series.index.isin(idx)
            n_l, n_w = int((lose & m).sum()), int((~lose & m).sum())
            share_l = n_l / max(1, lose.sum())
            rate_in = n_l / max(1, n_l + n_w)
            rate_out = int((lose & ~m).sum()) / max(1, int((~m).sum()))
            if n_l >= 3 and rate_in - rate_out >= 0.4 and val not in ("nan", "na", "-1"):
                out.append({"attribute": name, "value": val, "losers": n_l, "share_of_losers": share_l, "loss_rate_in": rate_in,
                            "loss_rate_out": rate_out})
    return sorted(out, key=lambda d: -d["loss_rate_in"])


def unknown_ledger(alog_rows: Sequence[Autopsy]) -> dict[str, float]:
    """Across autopsies: the shares of movers by knowability verdict. The UNKNOWN share should never be driven to zero by the
    labeller (section 33); a falling unknown share with a proxy that is not informative is a labelling bug."""
    tot: dict[str, float] = {}
    n = 0.0
    for a in alog_rows:
        if a.empty:
            continue
        for u in a.sec_research.unknown_causes:
            tot[u.verdict.value] = tot.get(u.verdict.value, 0.0) + u.count
            n += u.count
    return {k: v / n for k, v in sorted(tot.items())} if n else {}


def diff_questions(prev: Autopsy | None, cur: Autopsy) -> dict[str, list[str]]:
    """Questions that appeared, disappeared or persisted (by normalised text key) between two consecutive autopsies."""
    key = QuestionQueue._key
    a = {key(q): q.text for q in prev.questions()} if prev is not None and not prev.empty else {}
    b = {key(q): q.text for q in cur.questions()} if not cur.empty else {}
    return {"new": [b[k] for k in b if k not in a], "gone": [a[k] for k in a if k not in b], "persisting": [b[k] for k in b if k in a]}


def explain_surprises(ledger: ob.ObserverLedger, tracker: sp.SurpriseTracker, now, top: int = 5) -> list[dict[str, Any]]:
    """The tracker's repeated-surprise priorities, plus the contexts they occurred in; an input to the research queue's ordering."""
    out = []
    for pr in tracker.research_priority(now, sp.auto_similar(tracker, now))[:top]:
        out.append({"cell": pr.cell, "score": pr.score, "reason": pr.reason})
    return out


def selfcheck(seed: int = 0) -> dict[str, Any]:
    """Plant a known world, run the whole observer + autopsy, and score the autopsy against the truth. Returns booleans a harness
    can assert on after ANY code change: the planted false positives are found, predictable movers are not called unpredictable, the
    band counts equal the planted counts, and a null world raises no band, no missed winner and no unknown-cause bucket."""
    plant = O_PLANT(n_up_mid=20, n_up_big=8, n_down_mid=15, n_down_big=6)
    st = A_STATE()
    s, o, truth = ob.synthetic_day(800, seed, 0, st.obs_params, plant)
    a = step(st, s, o, "2035-01-01", "2026-09-29")
    assert st.observer is not None and st.observer.ledger is not None
    rec = st.observer.ledger._recs[-1]
    g = grade(a, {k: list(s.tickers[v]) for k, v in truth.items()}, rec)
    nul = A_STATE()
    s0, o0, _ = ob.synthetic_day(800, seed, 0, nul.obs_params, O_PLANT(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    a0 = step(nul, s0, o0, "2035-01-01", "2026-09-29")
    return {"bands_exact": a.sec_market.bands["down_gt10"] == 6 and a.sec_market.bands["down_5_10"] == 15, "fp_found": g["fp_in_worst"] >= 0.5,
            "no_misclassified_predictable": g["predictable_called_unpredictable"] == 0, "null_no_bands": sum(a0.sec_market.bands.values()) == 0,
            "null_no_missed_winners": not a0.sec_model.missed_winners, "null_no_causes": not a0.sec_research.unknown_causes,
            "coverage_ok": not coverage_audit(a, rec), "identity_free": not identity_leaks(to_dict(a), list(rec.rows["ticker"]))}


O_PLANT = ob.Plant


def A_STATE() -> AutopsyState:
    return AutopsyState()


# ==================================================================================================================
# persistence, priorities, roll-ups, day-over-day
# ==================================================================================================================
def save_queue(queue: QuestionQueue, path) -> None:
    """Write the queue as JSON (atomic). Questions keep their success / failure criteria and value vectors."""
    import json, os
    items = [{"first_day": i.first_day, "last_day": i.last_day, "times_raised": i.times_raised, "score": i.score,
              "question": {**_plain(i.question, False), "created_real": i.question.created_real,
                           "evidence_through": i.question.evidence_through, "parents": list(i.question.parents)}}
             for i in queue._q.values()]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(items, fh)
    os.replace(tmp, path)


def load_queue(path) -> QuestionQueue:
    import json
    q = QuestionQueue()
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    for it in items:
        d = it["question"]
        rq = ResearchQuestion(d["question_id"], d["text"], d["source"], Problem(d["problem"]), d["created_real"], d["evidence_through"],
                              d["success"], d["failure"], ExperimentValue(**d.get("expected", {})), tuple(d.get("parents", ())))
        item = QueuedQuestion(rq, it["first_day"], it["last_day"], int(it["times_raised"]), float(it["score"]))
        q._q[rq.question_id] = item
        q._text_key[QuestionQueue._key(rq)] = rq.question_id
    return q


def save_ledger(alog: AutopsyLedger, path) -> None:
    import json, os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump([_plain(r, False) for r in alog.rows], fh)
    os.replace(tmp, path)


def load_ledger(path) -> AutopsyLedger:
    import json
    al = AutopsyLedger()
    with open(path, encoding="utf-8") as fh:
        al.rows = [{k: (float("nan") if v is None else v) for k, v in r.items()} for r in json.load(fh)]
    return al


def queue_priorities(queue: QuestionQueue, k: int = 20) -> dict[str, float]:
    """The queue as a {question_id: score} map, ready to hand to AutopsyContext.priorities_after / the research policy."""
    return {i.question.question_id: i.score for i in queue.top(k)}


def day_over_day(prev: Autopsy | None, cur: Autopsy) -> dict[str, float]:
    """How the day differs from the previous autopsy: band counts, model quality, losses. NaN where either side is missing."""
    keys = ("score_auc", "precursor_auc", "mcc", "recall")
    out: dict[str, float] = {}
    if prev is None or prev.empty or cur.empty:
        return {"comparable": 0.0}
    out["comparable"] = 1.0
    for k in O_BANDS:
        out["d_band_" + k] = float(cur.sec_market.bands[k] - prev.sec_market.bands[k])
    for k in keys:
        a, b = getattr(prev.sec_model, k), getattr(cur.sec_model, k)
        out["d_" + k] = float(b - a) if a == a and b == b else float("nan")
    out["d_losses"] = float(cur.sec_risk.n_losses - prev.sec_risk.n_losses)
    out["d_severity"] = severity(cur) - severity(prev)
    return out


def rollup(autopsies: Sequence[Autopsy]) -> dict[str, Any]:
    """A period summary (a week, a year): mover-band totals, the most common regime label, the recurring missed-winner reason,
    the sum of losses and the avoidable share (loss-weighted), and how the unknown-cause shares split."""
    live = [a for a in autopsies if not a.empty]
    if not live:
        return {"days": 0}
    bands = {k: int(sum(a.sec_market.bands[k] for a in live)) for k in O_BANDS}
    reasons: dict[str, int] = {}
    for a in live:
        for k, v in a.sec_model.reason_counts.items():
            reasons[k] = reasons.get(k, 0) + v
    labels = pd.Series([regime_label(a.sec_market) for a in live]).value_counts()
    loss = np.array([a.sec_risk.loss_total for a in live])
    av = np.array([a.sec_risk.avoidable_loss_share if a.sec_risk.avoidable_loss_share == a.sec_risk.avoidable_loss_share else 0.0 for a in live])
    return {"days": len(live), "bands": bands, "bands_per_day": {k: v / len(live) for k, v in bands.items()}, "regimes": labels.to_dict(),
            "top_reason": max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "", "reasons": reasons,
            "losses": int(sum(a.sec_risk.n_losses for a in live)), "avoidable_share": float((av * loss).sum() / loss.sum()) if loss.sum() > 0 else float("nan"),
            "unknown_shares": unknown_ledger(live), "questions": int(sum(len(a.questions()) for a in live)),
            "mean_severity": float(np.mean([severity(a) for a in live]))}


def risk_history_z(cur: RiskSection, past: Sequence[RiskSection], min_days: int = 10) -> dict[str, float]:
    """Today's book concentration / co-loss / gap-cost against the trailing risk sections (past days the caller supplies)."""
    fields = {"hhi_sector": lambda r: r.concentration.get("hhi_sector"), "co_loss_z": lambda r: r.correlated.get("co_loss_z"),
              "adverse_gap_cost": lambda r: r.gap_risk.get("adverse_gap_cost"), "high_vol_share": lambda r: r.regime_exposure.get("high_vol_share")}
    out = {}
    for k, f in fields.items():
        h = np.array([_nz(f(r)) if _nz(f(r)) is not None else np.nan for r in past], dtype=float)
        h = h[np.isfinite(h)]
        x = _nz(f(cur))
        if x is None or len(h) < min_days:
            out[k] = float("nan")
            continue
        med = float(np.median(h))
        sc = max(1.4826 * float(np.median(np.abs(h - med))), 1e-9)
        out[k] = (x - med) / sc
    return out


def render_ledger_report(alog: AutopsyLedger, top: int = 5) -> str:
    """Plain-text trend report from the autopsy ledger."""
    f = alog.frame()
    if f.empty:
        return "autopsy ledger: no days"
    live = f[~f["empty"].astype(bool)]
    L = [f"autopsy ledger: {len(f)} days ({len(f) - len(live)} empty), {f.index[0]} .. {f.index[-1]}"]
    if live.empty:
        return "\n".join(L)
    for c in [c for c in live.columns if c.startswith("band_")]:
        L.append(f"  {c:<16s} mean {live[c].mean():8.1f}  max {int(live[c].max()):5d}")
    sk = alog.score_skill()
    L.append("score skill: " + ", ".join(f"{k} {v:.3f}" for k, v in sk.items()))
    rt = alog.reason_trend()
    if not rt.empty:
        L.append("missed-winner reason shifts: " + ", ".join(f"{k} {r['shift']:+.2f}" for k, r in rt.sort_values("shift", key=abs, ascending=False).head(top).iterrows()))
    reg = alog.band_regimes()
    L.append(f"band-storm days: {len(reg)}")
    return "\n".join(L)


def run_history(bars, directory, years=None, params: AutopsyParams | None = None, obs_params: ob.ObserverParams | None = None,
                created_real: str = "", log=None) -> tuple[QuestionQueue, AutopsyLedger]:
    """Autopsy a long OHLCV history one calendar year at a time (streamed, bounded memory), checkpointing the question queue and the
    autopsy ledger after each year. The observer/autopsy state is rebuilt per year from a fixed warm-up so a killed run resumes at the
    first unfinished year without a shared in-memory ledger. `bars` must include delisted names."""
    import os
    obs = obs_params or ob.ObserverParams()
    ap = params or AutopsyParams()
    os.makedirs(directory, exist_ok=True)
    slices = ob.year_slices(pd.DatetimeIndex(bars["Close"].index))
    qpath, lpath = os.path.join(directory, "autopsy_queue.json"), os.path.join(directory, "autopsy_ledger.json")
    queue = load_queue(qpath) if os.path.exists(qpath) else QuestionQueue()
    alog = load_ledger(lpath) if os.path.exists(lpath) else AutopsyLedger()
    done = as_date(alog.rows[-1]["day"]) if alog.rows else None
    for y in sorted(slices if years is None else [y for y in years if y in slices]):
        a, b = slices[y]
        if done is not None and as_date(b) <= done:
            continue
        st = AutopsyState(ap, obs)
        n = 0
        for aut in run_days(ob.bars_stream(bars, obs, None, start=a, end=b), st, queue, None, created_real=created_real):
            if done is None or as_date(aut.day) > done:
                alog.add(aut)
                n += 1
        save_queue(queue, qpath)
        save_ledger(alog, lpath)
        if log:
            log(f"autopsy year {y}: {n} days, queue {len(queue)}")
    return queue, alog


def question_health(queue: QuestionQueue, today: str) -> dict[str, float]:
    """Is the queue healthy? Size, share raised only once (noise candidates), share stale, and the concentration of the top
    scores. A queue of one-off questions that never repeat says the autopsy is generating noise, not a research agenda."""
    if not len(queue):
        return {"size": 0.0}
    items = list(queue._q.values())
    once = sum(1 for i in items if i.times_raised == 1) / len(items)
    sc = np.array(sorted((i.score for i in items), reverse=True))
    return {"size": float(len(items)), "raised_once_share": float(once), "stale_share": len(queue.stale(today)) / len(items),
            "top_score_share": float(sc[:3].sum() / sc.sum()) if sc.sum() > 0 else float("nan"),
            "problems": float(len(queue.by_problem()))}


def model_by_type(rec: ob.DayRecord) -> pd.DataFrame:
    """Per mover type on one day's rows: names, picks, band movers, and the share of band movers the model held."""
    rf = rec.rows
    cols = ["names", "picked", "band_movers", "band_recall"]
    if rf.empty:
        return pd.DataFrame(columns=cols)
    g = rf.groupby("mover_type")
    t = pd.DataFrame({"names": g.size(), "picked": g["picked"].sum(), "band_movers": g["band"].apply(lambda s: int((s != 0).sum()))})
    bm = rf[rf["band"] != 0].groupby("mover_type")["picked"].sum()
    t["band_recall"] = (bm.reindex(t.index).fillna(0) / t["band_movers"].replace(0, np.nan))
    return t[cols]


def risk_by_type(rec: ob.DayRecord, p: AutopsyParams) -> pd.DataFrame:
    """Per mover type: picks, losing picks, mean pick return. Which kinds of pick the book loses on today."""
    rf = rec.rows
    pk = rf[rf["picked"].astype(bool)] if not rf.empty else rf
    if pk.empty:
        return pd.DataFrame(columns=["picks", "losses", "mean_ret"])
    g = pk.assign(loss=pk["ret"].astype(float) <= -p.loss_thr).groupby("mover_type")
    return pd.DataFrame({"picks": g.size(), "losses": g["loss"].sum().astype(int), "mean_ret": g["ret"].mean()})


def question_sources(queue: QuestionQueue) -> dict[str, float]:
    """Share of queued questions by source (loss, missed_winner, break, surprise, ...). C66 section 34 says losses must carry
    weight; a queue with no loss questions after many days is a process fault worth surfacing."""
    if not len(queue):
        return {}
    c: dict[str, int] = {}
    for i in queue._q.values():
        c[i.question.source] = c.get(i.question.source, 0) + 1
    return {k: v / len(queue) for k, v in sorted(c.items())}


def narrative(a: Autopsy) -> str:
    """Three plain sentences for the day: market, model, risk. No tickers, no dates."""
    if a.empty:
        return "No universe was recorded."
    m, md, rk = a.sec_market, a.sec_model, a.sec_risk
    band = sum(m.bands.values())
    s1 = f"The market was {m.tone} with {band} band movers ({m.bands['up_5_10'] + m.bands['up_gt10']} up, {m.bands['down_5_10'] + m.bands['down_gt10']} down); regime {regime_label(m)}."
    s2 = (f"The model held {md.n_picked} names, captured {md.capturable_share:.0%} of the winners' tradeable move and missed {len(md.missed_winners)} listed winners, "
          f"mostly for {max(md.reason_counts.items(), key=lambda kv: kv[1])[0] if md.reason_counts else 'reasons it could not name'}.")
    s3 = (f"{rk.n_losses} of {rk.n_positions} positions lost at least {A_LOSS:.0%}; " +
          (f"{rk.avoidable_loss_share:.0%} of the loss was avoidable." if rk.avoidable_loss_share == rk.avoidable_loss_share else "no cause could be assigned."))
    return " ".join((s1, s2, s3))


A_LOSS = AutopsyParams().loss_thr


def loss_taxonomy(autopsies: Sequence[Autopsy]) -> pd.DataFrame:
    """Across days: how many listed losses per verdict and per named cause, and the loss-weighted mean size. UNKNOWN causes stay a
    row of their own ('' cause) - the classifier's honesty is part of the result."""
    rows = []
    for a in autopsies:
        if a.empty:
            continue
        for it in a.sec_risk.avoidable + a.sec_risk.unavoidable + a.sec_risk.undetermined:
            rows.append({"verdict": it.verdict, "cause": it.cause or "UNNAMED", "loss": it.loss})
    if not rows:
        return pd.DataFrame(columns=["n", "mean_loss"])
    g = pd.DataFrame(rows).groupby(["verdict", "cause"])["loss"]
    return pd.DataFrame({"n": g.size(), "mean_loss": g.mean()}).sort_values("n", ascending=False)


def severity_ranking(autopsies: Sequence[Autopsy], k: int = 10) -> list[tuple[str, float]]:
    """The days most worth a person's reading, by severity(); ties broken by day."""
    return sorted(((a.day, severity(a)) for a in autopsies), key=lambda t: (-t[1], t[0]))[:k]


def merge_queues(queues: Sequence[QuestionQueue]) -> QuestionQueue:
    """Combine queues from separate runs (e.g. per year). The same normalised question raised in several runs keeps the earliest
    first_day, the latest last_day and the summed times_raised, and its score is recomputed from the sum."""
    out = QuestionQueue()
    for q in queues:
        for item in q._q.values():
            key = QuestionQueue._key(item.question)
            cur = out._q.get(out._text_key.get(key, ""))
            if cur is None:
                out._q[item.question.question_id] = item
                out._text_key[key] = item.question.question_id
                continue
            n = cur.times_raised + item.times_raised
            out._q[cur.question.question_id] = QueuedQuestion(cur.question, min(cur.first_day, item.first_day), max(cur.last_day, item.last_day),
                                                              n, QuestionQueue._value(cur.question) * (1.0 + math.log(n)))
    return out


def gap_exposure_history(autopsies: Sequence[Autopsy]) -> dict[str, float]:
    """Across days: how often the book took an adverse entry gap and what it cost, so the gap-risk question has a base rate."""
    live = [a for a in autopsies if not a.empty and a.sec_risk.n_positions]
    if not live:
        return {"days": 0.0}
    n_adv = np.array([a.sec_risk.gap_risk.get("adverse_entry_gaps", 0.0) for a in live])
    cost = np.array([a.sec_risk.gap_risk.get("adverse_gap_cost", 0.0) for a in live])
    return {"days": float(len(live)), "days_with_adverse_gap": float((n_adv > 0).mean()), "mean_adverse_gaps": float(n_adv.mean()),
            "mean_cost": float(cost.mean()), "worst_cost": float(cost.max())}


def finding_stability(alog: AutopsyLedger, column: str, window: int = 20) -> dict[str, float]:
    """Is a per-day autopsy measure stable, drifting or noisy? Mean and spread over the last `window` days versus the earlier days,
    with a Welch t. A measure that swings wildly day to day cannot anchor a research question."""
    f = alog.frame()
    if f.empty or column not in f:
        return {"n": 0.0}
    x = f[column].astype(float).dropna().to_numpy()
    if len(x) < 2 * 5:
        return {"n": float(len(x))}
    late, early = x[-window:], x[:-window]
    if len(early) < 5 or len(late) < 5:
        late, early = x[len(x) // 2:], x[:len(x) // 2]
    se = math.sqrt(late.var(ddof=1) / len(late) + early.var(ddof=1) / len(early))
    return {"n": float(len(x)), "early_mean": float(early.mean()), "late_mean": float(late.mean()), "late_sd": float(late.std(ddof=1)),
            "welch_t": float((late.mean() - early.mean()) / se) if se > 0 else float("nan")}


def rerun_digests(days_factory, n_days: int, obs_params: ob.ObserverParams | None = None, ap: AutopsyParams | None = None) -> tuple[list[str], list[str]]:
    """Run the whole observer + autopsy twice from scratch on the same days and return both digest lists. Equal lists = the
    autopsy is deterministic (nothing depends on wall-clock, dict order, hidden state or a leaked previous run)."""
    runs = []
    for _ in range(2):
        st = AutopsyState(ap or AutopsyParams(), obs_params or ob.ObserverParams())
        runs.append([replay_digest(a) for a in run_days(days_factory(), st, None, None, created_real="fixed")][:n_days])
    return runs[0], runs[1]


def check_no_future(a: Autopsy, now) -> list[str]:
    """The autopsy and everything it evidences must be dated strictly before `now`. Returns the offending items (empty = clean)."""
    bad = []
    try:
        require_past(a.resolved_at, now, "autopsy")
    except FirewallBreach as e:
        bad.append(str(e))
    for q in a.questions():
        try:
            require_past(q.evidence_through, now, f"question {q.question_id} evidence")
        except FirewallBreach as e:
            bad.append(str(e))
    return bad


def pattern_break_summary(autopsies: Sequence[Autopsy]) -> pd.DataFrame:
    """Per pattern, across days: how often it was reported broken / new / revived and its last effect. A pattern that breaks on
    many days with a p_real that stays high is the highest-value break to study (C66 section 14)."""
    rows = []
    for a in autopsies:
        if a.empty:
            continue
        for e in a.sec_research.new_patterns + a.sec_research.broken_patterns:
            rows.append({"pattern": e.pattern_id, "kind": e.kind, "effect_after": e.effect_after, "p_real": e.p_real})
    if not rows:
        return pd.DataFrame(columns=["broken", "new", "revived", "weakened", "last_effect", "last_p_real"])
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="pattern", columns="kind", values="effect_after", aggfunc="size", fill_value=0)
    for c in ("broken", "new", "revived", "weakened"):
        if c not in piv:
            piv[c] = 0
    last = df.groupby("pattern").tail(1).set_index("pattern")
    piv["last_effect"], piv["last_p_real"] = last["effect_after"], last["p_real"]
    return piv[["broken", "new", "revived", "weakened", "last_effect", "last_p_real"]].sort_values("broken", ascending=False)


def context_warnings(rec: ob.DayRecord) -> list[str]:
    """Data-quality warnings the day carries into its questions (thin scoring, missing features, suspect band movers, an outage)."""
    w = ob.stale_or_thin(rec)
    nb = sum(rec.bands.values())
    if nb and rec.market.get("n_suspect_band", 0) / nb >= 0.25:
        w.append("a quarter or more of the band movers are suspect prints")
    if rec.n_no_outcome and rec.n_universe and rec.n_no_outcome / rec.n_universe >= 0.2:
        w.append("a fifth or more of the universe has no outcome")
    return w


def annotate(a: Autopsy, rec: ob.DayRecord) -> dict[str, Any]:
    """Wrap an Autopsy with its warnings, severity, regime label and narrative, ready for a report."""
    return {"day": a.day, "warnings": context_warnings(rec), "severity": severity(a), "regime": regime_label(a.sec_market) if not a.empty else "empty",
            "narrative": narrative(a), "coverage_problems": coverage_audit(a, rec), "count_problems": ob.verify_counts(rec)}


def question_texts(queue: QuestionQueue, k: int = 10) -> list[str]:
    """The top-k questions as text lines with their raise count, for a quick read."""
    return [f"[{i.times_raised}x, {i.question.problem.value}] {i.question.text}" for i in queue.top(k)]


def loss_first_order(queue: QuestionQueue, k: int = 10) -> list[QueuedQuestion]:
    """Section 34: loss-avoidance questions are ordered ahead of others at equal score, because a loss is worth studying before a
    missed gain. The order is (problem rank, -score); the objective order comes from engine.research.core.OBJECTIVE_ORDER."""
    from engine.research.core import OBJECTIVE_ORDER
    rank = {p: i for i, p in enumerate(OBJECTIVE_ORDER)}
    return sorted(queue._q.values(), key=lambda i: (rank.get(i.question.problem, len(rank)), -i.score, i.question.question_id))[:k]


def write_reports(autopsies: Sequence[Autopsy], directory, keep_ticker: bool = False) -> list[str]:
    """Write one markdown and one JSON file per autopsy (atomic). With keep_ticker=False the JSON carries no ticker and no date, and
    the file name is a content hash, so the directory can be handed to a reader that must not learn identities."""
    import json, os
    os.makedirs(directory, exist_ok=True)
    paths = []
    for a in autopsies:
        stem = f"autopsy_{a.day}" if keep_ticker else f"autopsy_{replay_digest(a)}"
        for ext, body in (("md", to_markdown(a)), ("json", json.dumps(to_dict(a, keep_ticker)))):
            path = os.path.join(directory, f"{stem}.{ext}")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, path)
            paths.append(path)
    return paths


def band_table(autopsies: Sequence[Autopsy]) -> pd.DataFrame:
    """One row per day, one column per C67 band (close-to-close), plus the open-to-close total: the series the episode research
    reads to see how many 5-10% and >10% movers each day offered."""
    rows = [{"day": a.day, **a.sec_market.bands, "o2c_total": sum(a.sec_market.bands_o2c.values()), "suspect": a.sec_market.n_suspect_band}
            for a in autopsies if not a.empty]
    return pd.DataFrame(rows).set_index("day") if rows else pd.DataFrame(columns=list(O_BANDS) + ["o2c_total", "suspect"])


def unresolved_questions(queue: QuestionQueue, answered: Sequence[str]) -> list[QueuedQuestion]:
    """Queued questions whose ids the research loop has not reported as answered, highest score first."""
    done = set(answered)
    return [i for i in queue.top(len(queue)) if i.question.question_id not in done]


def priorities_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """Change in each priority between two snapshots (added items count from 0, removed items to 0)."""
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(set(before) | set(after)) if abs(after.get(k, 0.0) - before.get(k, 0.0)) > 1e-12}


def top_reason(m: ModelSection) -> str:
    """The most frequent rejection reason among today's missed winners, or '' if none were analysed."""
    return max(m.reason_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if m.reason_counts else ""


def sources_of(a: Autopsy) -> dict[str, int]:
    """Number of today's questions by source."""
    out: dict[str, int] = {}
    for q in a.questions():
        out[q.source] = out.get(q.source, 0) + 1
    return out
