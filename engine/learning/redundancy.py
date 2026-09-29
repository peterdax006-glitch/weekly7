"""Redundancy intelligence (SELF_LEARNING_CONTRACT section 21; checklist F08 REDUNDANT_WITH and F09 COMPLEMENTS edges).
IMPLEMENTED - NOT VALIDATED.

The contract's rule: never assume "correlated = useless". Two patterns can make the same prediction and still protect
different regimes, and a learner that retires one of them on correlation alone throws away the protection. This module
measures FIVE separate kinds of redundancy between two learned items and only calls a pair a true duplicate when the
kinds that matter agree:

  predictive    do they make the same prediction?  rank correlation of their signals plus how much of each one's
                predictive value survives after the other is controlled for (partial rank IC, cluster-bootstrapped);
  context       are they effective in the SAME contexts?  overlap of the sets of contexts where each has real IC;
  mechanistic   do they claim the same cause?  overlap of declared mechanism tags / inputs / decision effects (this is
                DECLARED knowledge, never measured; empty tags -> untested, not "different");
  operational   can one stand in for the other at run time?  availability fallback and shared failure points (inputs);
  risk          do they lose together?  lower-tail co-loss lift and the hedge value of one when the other loses.

Each kind is a number in [0, 1] or None (untested - never a silent 0). The pair is then classified:
TRUE_DUPLICATE, DIFFERENT_PROTECTION (same prediction, different regime or risk behaviour: KEEP BOTH), PARTIAL_OVERLAP,
COMPLEMENT, INDEPENDENT or UNRESOLVED (evidence for a verdict is missing). Recommendations never delete - retirement
means archive (section 13) - and a member is only proposed for retirement when dropping it keeps context coverage.

Edges for the knowledge graph: REDUNDANT_WITH (F08) and COMPLEMENTS (F09) from core.Edge.
Set-level tools: variance-inflation (multi-item predictive collinearity), duplicate clusters, coverage-preserving
reduction, era stability of pairwise redundancy. Firewall: rows are usable only if date + horizon is strictly before `now`.
Deterministic in `seed`."""
from __future__ import annotations

import dataclasses
import enum
import itertools
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.stops import wilson                       # Wilson interval, reused (not re-implemented)
from engine.learning.core import (DecisionEffect, Edge, FirewallBreach, Unknown, as_date, canonical_json,
                                  current_code_hash, require_past, stable_hash)


class RedundancyKind(str, enum.Enum):
    PREDICTIVE = "PREDICTIVE"
    CONTEXT = "CONTEXT"
    MECHANISTIC = "MECHANISTIC"
    OPERATIONAL = "OPERATIONAL"
    RISK = "RISK"


class PairClass(str, enum.Enum):
    TRUE_DUPLICATE = "TRUE_DUPLICATE"
    DIFFERENT_PROTECTION = "DIFFERENT_PROTECTION"      # same prediction, different regime/risk behaviour: keep both
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
    COMPLEMENT = "COMPLEMENT"
    INDEPENDENT = "INDEPENDENT"
    UNRESOLVED = "UNRESOLVED"


@dataclasses.dataclass(frozen=True)
class RedundancyConfig:
    seed: int = 7
    high: float = 0.70                 # redundancy at or above this counts as "the same"
    low: float = 0.30                  # at or below this counts as "different"
    ctx_high: float = 0.60
    risk_high: float = 0.60
    min_overlap: int = 60              # rows where both items have an output
    min_context_n: int = 20
    ctx_ic_min: float = 0.03           # a context is "effective" when its IC exceeds this and is not noise
    ctx_t_min: float = 1.65
    tail_q: float = 0.10               # lower-tail definition for risk redundancy
    min_tail: int = 8
    active_eps: float = 1e-12
    n_boot: int = 200
    alpha: float = 0.05
    cover_min: float = 0.5             # each item must be >= this covered by the other to call a duplicate
    era_gap: float = 0.30              # redundancy that moves more than this between halves is unstable

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.low < self.high < 1:
            errs.append("need 0 < low < high < 1")
        if not 0 < self.tail_q < 0.5:
            errs.append("tail_q must be in (0, 0.5)")
        if self.min_overlap < 10 or self.min_context_n < 5 or self.min_tail < 3 or self.n_boot < 20:
            errs.append("sample thresholds too small to mean anything")
        return errs


@dataclasses.dataclass(frozen=True)
class ItemProfile:
    """Static description of a learned item: what it claims and what it consumes."""
    item_id: str
    mechanism_tags: tuple = ()
    inputs: tuple = ()
    decision_effect: tuple = (DecisionEffect.RANKING,)
    cost: float = 1.0

    def validate(self) -> list[str]:
        errs = []
        if not self.item_id:
            errs.append("item_id missing")
        if self.cost < 0:
            errs.append(f"{self.item_id}: negative cost")
        return errs


@dataclasses.dataclass
class ObservationPanel:
    """Wide table of item outputs on shared observations. NaN = the item produced no output (unavailable); 0 = available
    but no opinion. `y` is the realised forward return; `context` a label per observation (regime, vol bucket, ...)."""
    signals: pd.DataFrame
    y: pd.Series
    dates: pd.DatetimeIndex
    context: pd.Series
    horizon_days: int = 5
    dropped_future: int = 0

    @classmethod
    def build(cls, signals: pd.DataFrame, y: pd.Series, dates, context=None, horizon_days: int = 5, now=None,
              strict: bool = False) -> "ObservationPanel":
        if len(signals) != len(y) or len(signals) != len(dates):
            raise ValueError("signals, y and dates must have equal length")
        if signals.columns.duplicated().any():
            raise ValueError("duplicate item ids")
        dates = pd.DatetimeIndex(dates)
        ctx = pd.Series(["all"] * len(y) if context is None else list(context), dtype=object)
        keep = np.ones(len(y), dtype=bool)
        if now is not None:
            for i, d in enumerate(dates):
                try:
                    require_past(d + pd.Timedelta(days=horizon_days), now, "observation outcome")
                except FirewallBreach:
                    if strict:
                        raise
                    keep[i] = False
        sig = signals.reset_index(drop=True).iloc[np.flatnonzero(keep)].reset_index(drop=True)
        return cls(sig, pd.Series(np.asarray(y, dtype=float)[keep]), dates[keep],
                   ctx.iloc[np.flatnonzero(keep)].reset_index(drop=True), horizon_days, int((~keep).sum()))

    @property
    def n(self) -> int:
        return len(self.y)

    def weeks(self) -> np.ndarray:
        iso = self.dates.isocalendar()
        return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()

    def subset(self, mask: np.ndarray) -> "ObservationPanel":
        ix = np.flatnonzero(mask)
        return ObservationPanel(self.signals.iloc[ix].reset_index(drop=True), self.y.iloc[ix].reset_index(drop=True),
                                self.dates[ix], self.context.iloc[ix].reset_index(drop=True), self.horizon_days)


# ---------------------------------------------------------------------------------------------- statistics
def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(_rank(a), _rank(b))


def partial_corr(rx: np.ndarray, ry: np.ndarray, rz: np.ndarray) -> float:
    """Correlation of x and y after removing z (all inputs already rank-transformed)."""
    rxy, rxz, ryz = _corr(rx, ry), _corr(rx, rz), _corr(ry, rz)
    if not all(np.isfinite([rxy, rxz, ryz])):
        return float("nan")
    den = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-15))
    return float((rxy - rxz * ryz) / den)


def cluster_bootstrap(stat, n: int, codes: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Draws of an arbitrary statistic stat(row_index) under cluster (week) resampling; finite draws only. Needed because the
    statistics here (rank correlations) are not weighted means, which is all pattern_stats.cluster_bootstrap_ci handles."""
    uniq, inv = np.unique(codes, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        ix = np.concatenate([members[g] for g in pick])
        v = stat(ix)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def jaccard(a: Iterable, b: Iterable) -> float | None:
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------------------------- result records
@dataclasses.dataclass(frozen=True)
class KindScore:
    kind: RedundancyKind
    value: float | None                # in [0, 1]; None = untested
    n: int
    detail: Mapping[str, float] = dataclasses.field(default_factory=dict)
    note: str = ""

    def validate(self) -> list[str]:
        errs = []
        if self.value is not None and not (0.0 <= self.value <= 1.0):
            errs.append(f"{self.kind.value}: value {self.value} outside [0,1]")
        if self.value is None and not self.note:
            errs.append(f"{self.kind.value}: untested score must say why")
        return errs

    @property
    def state(self) -> Unknown | None:
        return None if self.value is not None else Unknown.UNTESTED


@dataclasses.dataclass(frozen=True)
class PairRedundancy:
    a: str
    b: str
    scores: Mapping[RedundancyKind, KindScore]
    n_overlap: int
    klass: PairClass
    subsumed_by: str | None            # the item that fully covers the other's predictive value, if any
    flags: tuple
    recommendation: str
    era_stable: bool | None

    def value(self, kind: RedundancyKind) -> float | None:
        return self.scores[kind].value

    def validate(self) -> list[str]:
        errs = []
        for s in self.scores.values():
            errs += s.validate()
        if self.klass is PairClass.TRUE_DUPLICATE and any(self.value(k) is None for k in
                                                          (RedundancyKind.PREDICTIVE, RedundancyKind.CONTEXT, RedundancyKind.RISK)):
            errs.append(f"{self.a}~{self.b}: TRUE_DUPLICATE declared with an untested core kind")
        return errs

    def edges(self) -> list[tuple[str, Edge, str, float, str]]:
        """(src, edge, dst, weight, why) for the knowledge graph. REDUNDANT_WITH only for pairs redundant on the core kinds;
        COMPLEMENTS for pairs that protect different regimes or hedge each other (F08, F09)."""
        if self.klass is PairClass.TRUE_DUPLICATE:
            w = float(np.mean([self.value(k) for k in (RedundancyKind.PREDICTIVE, RedundancyKind.CONTEXT,
                                                       RedundancyKind.RISK)]))
            return [(self.a, Edge.REDUNDANT_WITH, self.b, w, "duplicate on predictive+context+risk")]
        if self.klass in (PairClass.DIFFERENT_PROTECTION, PairClass.COMPLEMENT):
            p = self.value(RedundancyKind.PREDICTIVE)
            diffs = [v for v in (self.value(RedundancyKind.CONTEXT), self.value(RedundancyKind.RISK)) if v is not None]
            w = float(1.0 - min(diffs)) if diffs else 0.0
            why = "same prediction, different protection" if self.klass is PairClass.DIFFERENT_PROTECTION else "complementary"
            out = [(self.a, Edge.COMPLEMENTS, self.b, w, why)]
            if self.klass is PairClass.DIFFERENT_PROTECTION and p is not None:
                out.append((self.a, Edge.REDUNDANT_WITH, self.b, float(p), "predictive redundancy only"))
            return out
        return []


# ---------------------------------------------------------------------------------------------- the five measures
class Measurer:
    """Computes the five redundancy kinds for a pair on one panel. Holds no state except the config and a seed."""

    def __init__(self, panel: ObservationPanel, profiles: Mapping[str, ItemProfile], cfg: RedundancyConfig):
        self.panel, self.profiles, self.cfg = panel, profiles, cfg
        self.y = panel.y.to_numpy(dtype=float)
        self.weeks = panel.weeks()
        self.ctx = panel.context.to_numpy(dtype=object)

    def _sig(self, item: str) -> np.ndarray:
        return self.panel.signals[item].to_numpy(dtype=float)

    def _active(self, item: str) -> np.ndarray:
        s = self._sig(item)
        return np.isfinite(s) & (np.abs(np.nan_to_num(s)) > self.cfg.active_eps)

    # -- predictive
    def predictive(self, a: str, b: str, rng: np.random.Generator) -> KindScore:
        sa, sb = self._sig(a), self._sig(b)
        ov = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(self.y)
        n = int(ov.sum())
        if n < self.cfg.min_overlap:
            return KindScore(RedundancyKind.PREDICTIVE, None, n, note=f"only {n} overlapping outputs (need {self.cfg.min_overlap})")
        ra, rb, ry = _rank(sa[ov]), _rank(sb[ov]), _rank(self.y[ov])
        rho = _corr(ra, rb)
        if not np.isfinite(rho):
            return KindScore(RedundancyKind.PREDICTIVE, None, n, note="a signal is constant on the overlap")
        ic_a, ic_b = _corr(ra, ry), _corr(rb, ry)
        pa, pb = partial_corr(ra, ry, rb), partial_corr(rb, ry, ra)
        ca, cb = self._cover(ic_a, pa), self._cover(ic_b, pb)
        w = self.weeks[ov]

        def stat_rho(ix):
            return _corr(ra[ix], rb[ix])
        boots = cluster_bootstrap(stat_rho, n, w, rng, self.cfg.n_boot)
        lo, hi = (np.quantile(boots, [self.cfg.alpha / 2, 1 - self.cfg.alpha / 2]) if len(boots) > 10 else (float("nan"),) * 2)
        det = {"rho": rho, "rho_lo": float(lo), "rho_hi": float(hi), "ic_a": ic_a, "ic_b": ic_b, "partial_ic_a": pa,
               "partial_ic_b": pb, "cover_a": ca if ca is not None else float("nan"),
               "cover_b": cb if cb is not None else float("nan"),
               "sign_agreement": float((np.sign(sa[ov]) == np.sign(sb[ov])).mean())}
        return KindScore(RedundancyKind.PREDICTIVE, float(min(1.0, abs(rho))), n, det,
                         "" if rho >= 0 else "signals are negatively related: the same information with the sign flipped")

    @staticmethod
    def _cover(ic: float, partial: float) -> float | None:
        """Share of an item's predictive value that the OTHER item already explains: 1 - surviving/original. None when the
        item has no predictive value to be covered (then 'cover' is meaningless, not 1)."""
        if not np.isfinite(ic) or not np.isfinite(partial) or abs(ic) < 0.02:
            return None
        return float(min(1.0, max(0.0, 1.0 - max(0.0, partial * np.sign(ic)) / abs(ic))))

    # -- context
    def context_ic(self, item: str) -> dict:
        """context label -> (n, ic, t) for one item. A context is effective when ic > ctx_ic_min and t > ctx_t_min."""
        s = self._sig(item)
        out = {}
        for g in sorted(set(self.ctx)):
            m = (self.ctx == g) & np.isfinite(s) & np.isfinite(self.y)
            if m.sum() < self.cfg.min_context_n:
                continue
            r = spearman(s[m], self.y[m])
            if not np.isfinite(r):
                continue
            t = r * math.sqrt((m.sum() - 2) / max(1e-12, 1 - r * r))
            out[g] = (int(m.sum()), float(r), float(t))
        return out

    def effective_contexts(self, item: str) -> set:
        return {g for g, (n, ic, t) in self.context_ic(item).items() if ic > self.cfg.ctx_ic_min and t > self.cfg.ctx_t_min}

    def context(self, a: str, b: str) -> KindScore:
        ca, cb = self.context_ic(a), self.context_ic(b)
        shared = sorted(set(ca) & set(cb))
        if len(shared) < 2:
            return KindScore(RedundancyKind.CONTEXT, None, len(shared), note="fewer than two contexts with enough data for both")
        ea = {g for g in shared if ca[g][1] > self.cfg.ctx_ic_min and ca[g][2] > self.cfg.ctx_t_min}
        eb = {g for g in shared if cb[g][1] > self.cfg.ctx_ic_min and cb[g][2] > self.cfg.ctx_t_min}
        jac = jaccard(ea, eb)
        prof_a, prof_b = np.array([ca[g][1] for g in shared]), np.array([cb[g][1] for g in shared])
        pc = _corr(prof_a, prof_b) if len(shared) >= 3 else float("nan")
        det = {"only_a": float(len(ea - eb)), "only_b": float(len(eb - ea)), "both": float(len(ea & eb)),
               "profile_corr": float(pc) if np.isfinite(pc) else float("nan")}
        if jac is None:
            return KindScore(RedundancyKind.CONTEXT, None, len(shared), det,
                             "neither item is effective in any tested context - context redundancy is undefined")
        return KindScore(RedundancyKind.CONTEXT, float(jac), len(shared), det)

    # -- mechanistic (declared)
    def mechanistic(self, a: str, b: str) -> KindScore:
        pa, pb = self.profiles.get(a), self.profiles.get(b)
        if pa is None or pb is None:
            return KindScore(RedundancyKind.MECHANISTIC, None, 0, note="no profile for one of the items")
        tags = jaccard(pa.mechanism_tags, pb.mechanism_tags) if (pa.mechanism_tags and pb.mechanism_tags) else None
        inputs = jaccard(pa.inputs, pb.inputs) if (pa.inputs and pb.inputs) else None
        eff = jaccard([e.value for e in pa.decision_effect], [e.value for e in pb.decision_effect])
        parts = [(0.6, tags), (0.25, inputs), (0.15, eff)]
        got = [(w, v) for w, v in parts if v is not None]
        if tags is None:
            return KindScore(RedundancyKind.MECHANISTIC, None, 0,
                             {"input_overlap": inputs if inputs is not None else float("nan")},
                             "mechanism tags missing on at least one item: mechanism is UNKNOWN, not 'different'")
        val = sum(w * v for w, v in got) / sum(w for w, _ in got)
        return KindScore(RedundancyKind.MECHANISTIC, float(val), len(got),
                         {"tags": tags, "inputs": inputs if inputs is not None else float("nan"), "effect": eff or 0.0},
                         "declared, not measured")

    # -- operational
    def operational(self, a: str, b: str, predictive: float | None = None) -> KindScore:
        """Substitutability at run time = availability fallback x decision-effect overlap x predictive similarity: an item that
        is merely AVAILABLE when the other is not, but predicts something else, is not a backup."""
        av_a, av_b = np.isfinite(self._sig(a)), np.isfinite(self._sig(b))
        pa, pb = self.profiles.get(a), self.profiles.get(b)
        shared_inputs = sorted(set(pa.inputs) & set(pb.inputs)) if pa and pb else []
        out_a, out_b = ~av_a, ~av_b
        if out_a.sum() < self.cfg.min_tail and out_b.sum() < self.cfg.min_tail:
            return KindScore(RedundancyKind.OPERATIONAL, None, int(out_a.sum() + out_b.sum()),
                             {"shared_inputs": float(len(shared_inputs))},
                             "neither item was ever unavailable: fallback value untestable")
        fb_ab = float(av_b[out_a].mean()) if out_a.sum() >= self.cfg.min_tail else float("nan")
        fb_ba = float(av_a[out_b].mean()) if out_b.sum() >= self.cfg.min_tail else float("nan")
        fb = [v for v in (fb_ab, fb_ba) if np.isfinite(v)]
        eff = jaccard([e.value for e in pa.decision_effect], [e.value for e in pb.decision_effect]) if pa and pb else 0.0
        if predictive is None:
            return KindScore(RedundancyKind.OPERATIONAL, None, int(out_a.sum() + out_b.sum()),
                             {"shared_inputs": float(len(shared_inputs))}, "predictive similarity untested: cannot call either a backup")
        val = float(np.mean(fb)) * float(eff or 0.0) * float(predictive)
        return KindScore(RedundancyKind.OPERATIONAL, float(min(1.0, val)), int(out_a.sum() + out_b.sum()),
                         {"fallback_b_when_a_out": fb_ab, "fallback_a_when_b_out": fb_ba,
                          "shared_inputs": float(len(shared_inputs)), "effect_overlap": float(eff or 0.0)})

    # -- risk
    def pnl(self, item: str) -> np.ndarray:
        """Per-observation payoff of following the item: signal * outcome (0 where unavailable)."""
        return np.nan_to_num(self._sig(item)) * self.y

    def date_pnl(self, item: str) -> pd.Series:
        return pd.Series(self.pnl(item)).groupby(self.panel.dates.values).mean()

    def risk(self, a: str, b: str) -> KindScore:
        pa, pb = self.date_pnl(a), self.date_pnl(b)
        both = pa.index.intersection(pb.index)
        pa, pb = pa.loc[both], pb.loc[both]
        n = len(both)
        k = int(round(self.cfg.tail_q * n))
        if k < self.cfg.min_tail:
            return KindScore(RedundancyKind.RISK, None, n, note=f"{k} tail dates (need {self.cfg.min_tail})")
        thr_a, thr_b = np.sort(pa.values)[k - 1], np.sort(pb.values)[k - 1]
        la, lb = pa.values <= thr_a, pb.values <= thr_b
        co = float((la & lb).sum() / max(la.sum(), 1))               # P(B in tail | A in tail)
        q = float(lb.mean())
        lift = (co - q) / max(1 - q, 1e-12)
        lo, hi = wilson(int((la & lb).sum()), int(la.sum()))
        hedge = float(pb.values[la].mean() - pb.values.mean())       # what B earns, relative to normal, when A is in its tail
        hedge_sd = float(pb.values.std()) / math.sqrt(max(la.sum(), 1))
        det = {"p_co_loss": co, "p_co_loss_lo": lo, "p_co_loss_hi": hi, "base_tail": q, "hedge_gain": hedge,
               "hedge_z": hedge / max(hedge_sd, 1e-15), "corr": _corr(pa.values, pb.values)}
        return KindScore(RedundancyKind.RISK, float(min(1.0, max(0.0, lift))), n, det)

    # -- era stability
    def era_stability(self, a: str, b: str, rng: np.random.Generator) -> bool | None:
        order = np.argsort(self.panel.dates.values, kind="stable")
        half = len(order) // 2
        early = np.zeros(self.panel.n, dtype=bool)
        early[order[:half]] = True
        vals = []
        for m in (early, ~early):
            sub = Measurer(self.panel.subset(m), self.profiles, self.cfg)
            p = sub.predictive(a, b, rng)
            if p.value is None:
                return None
            vals.append(p.value)
        return abs(vals[0] - vals[1]) <= self.cfg.era_gap


# ---------------------------------------------------------------------------------------------- classification
def classify_pair(cfg: RedundancyConfig, scores: Mapping[RedundancyKind, KindScore]) -> tuple[PairClass, list[str]]:
    pred, ctx, risk = (scores[k].value for k in (RedundancyKind.PREDICTIVE, RedundancyKind.CONTEXT, RedundancyKind.RISK))
    flags: list[str] = []
    if pred is None:
        return PairClass.UNRESOLVED, ["predictive redundancy untested: " + scores[RedundancyKind.PREDICTIVE].note]
    det = scores[RedundancyKind.PREDICTIVE].detail
    if pred >= cfg.high:
        if ctx is None or risk is None:
            missing = [k.value for k, v in ((RedundancyKind.CONTEXT, ctx), (RedundancyKind.RISK, risk)) if v is None]
            return PairClass.UNRESOLVED, [f"same prediction but {'/'.join(missing)} untested: cannot rule out different protection"]
        differing = []
        if ctx < cfg.ctx_high:
            differing.append("context")
        if risk < cfg.risk_high:
            differing.append("risk")
        hz = scores[RedundancyKind.RISK].detail.get("hedge_z", 0.0)
        if hz > 2.0:
            differing.append("hedge")
        covers = [det.get("cover_a", float("nan")), det.get("cover_b", float("nan"))]
        finite = [c for c in covers if np.isfinite(c)]
        if not differing and finite and min(finite) < cfg.cover_min:
            differing.append("incremental_information")
        if differing:
            flags += [f"protects differently: {d}" for d in differing]
            return PairClass.DIFFERENT_PROTECTION, flags
        return PairClass.TRUE_DUPLICATE, flags
    if pred <= cfg.low:
        hz = scores[RedundancyKind.RISK].detail.get("hedge_z", 0.0) if scores[RedundancyKind.RISK].value is not None else 0.0
        det_c = scores[RedundancyKind.CONTEXT].detail
        complementary_ctx = ctx is not None and ctx <= cfg.low and det_c.get("only_a", 0) >= 1 and det_c.get("only_b", 0) >= 1
        if complementary_ctx or hz > 2.0:
            flags.append("complementary contexts" if complementary_ctx else "one hedges the other's tail losses")
            return PairClass.COMPLEMENT, flags
        return PairClass.INDEPENDENT, flags
    return PairClass.PARTIAL_OVERLAP, flags


# ---------------------------------------------------------------------------------------------- report
@dataclasses.dataclass(frozen=True)
class RedundancyReport:
    config: RedundancyConfig
    code_hash: str
    now: str
    n_obs: int
    dropped_future: int
    pairs: tuple
    vif: Mapping[str, float]
    clusters: tuple                    # tuples of item ids that are mutually TRUE_DUPLICATE
    retire_candidates: tuple           # (item, kept_instead, reason) - archive, never delete
    panel_hash: str

    def pair(self, a: str, b: str) -> PairRedundancy:
        for p in self.pairs:
            if {p.a, p.b} == {a, b}:
                return p
        raise KeyError((a, b))

    def edges(self) -> list:
        return [e for p in self.pairs for e in p.edges()]

    def matrix(self, kind: RedundancyKind) -> pd.DataFrame:
        items = sorted({p.a for p in self.pairs} | {p.b for p in self.pairs})
        M = pd.DataFrame(np.nan, index=items, columns=items)
        for p in self.pairs:
            v = p.value(kind)
            M.loc[p.a, p.b] = M.loc[p.b, p.a] = np.nan if v is None else v
        return M

    def label(self) -> str:
        return "IMPLEMENTED — NOT VALIDATED"

    def render_text(self) -> str:
        lines = [f"Redundancy @ {self.now}: {self.n_obs} observations ({self.dropped_future} dropped as not yet matured)  "
                 f"[{self.label()}]"]
        for p in sorted(self.pairs, key=lambda p: (p.klass.value, p.a, p.b)):
            cells = " ".join(f"{k.value[:4].lower()}={'  - ' if p.value(k) is None else format(p.value(k), '.2f')}" for k in RedundancyKind)
            lines.append(f"  {p.a} ~ {p.b}: {p.klass.value:<21}{cells}  n={p.n_overlap}")
            for f in p.flags:
                lines.append(f"      - {f}")
            lines.append(f"      => {p.recommendation}")
        for it, kept, why in self.retire_candidates:
            lines.append(f"  archive candidate: {it} (keep {kept}) - {why}")
        return "\n".join(lines)


class RedundancyAnalyzer:
    def __init__(self, cfg: RedundancyConfig | None = None):
        self.cfg = cfg or RedundancyConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))

    def analyze_pair(self, panel: ObservationPanel, profiles: Mapping[str, ItemProfile], a: str, b: str) -> PairRedundancy:
        for it in (a, b):
            if it not in panel.signals.columns:
                raise KeyError(f"item {it!r} not in panel")
        if a == b:
            raise ValueError("a pair needs two different items")
        seeds = np.random.SeedSequence([self.cfg.seed, _stable_int(a), _stable_int(b)]).spawn(2)
        m = Measurer(panel, profiles, self.cfg)
        scores = {RedundancyKind.PREDICTIVE: m.predictive(a, b, np.random.default_rng(seeds[0])),
                  RedundancyKind.CONTEXT: m.context(a, b), RedundancyKind.MECHANISTIC: m.mechanistic(a, b),
                  RedundancyKind.OPERATIONAL: None, RedundancyKind.RISK: m.risk(a, b)}
        scores[RedundancyKind.OPERATIONAL] = m.operational(a, b, scores[RedundancyKind.PREDICTIVE].value)
        klass, flags = classify_pair(self.cfg, scores)
        pd_ = scores[RedundancyKind.PREDICTIVE].detail
        subsumed = None
        ca, cb = pd_.get("cover_a", float("nan")), pd_.get("cover_b", float("nan"))
        if np.isfinite(ca) and np.isfinite(cb):
            if ca >= 0.8 and cb < self.cfg.cover_min:
                subsumed = b                       # b already explains a, but a does not explain b
            elif cb >= 0.8 and ca < self.cfg.cover_min:
                subsumed = a
        era = m.era_stability(a, b, np.random.default_rng(seeds[1])) if scores[RedundancyKind.PREDICTIVE].value is not None else None
        if era is False:
            flags.append("predictive redundancy differs between the early and late halves")
        if scores[RedundancyKind.MECHANISTIC].value is not None and scores[RedundancyKind.MECHANISTIC].value >= self.cfg.high \
                and (scores[RedundancyKind.PREDICTIVE].value or 0) < self.cfg.low:
            flags.append("claims the same mechanism but does not predict alike: one claim is wrong")
        op = scores[RedundancyKind.OPERATIONAL]
        if op.value is not None and op.value >= self.cfg.high:
            flags.append("operational backup: either can cover the other's outages")
        if op.detail.get("shared_inputs", 0) >= 1:
            flags.append(f"shared failure point: {int(op.detail['shared_inputs'])} common input(s)")
        rec = self._recommend(klass, a, b, subsumed, m)
        return PairRedundancy(a, b, scores, scores[RedundancyKind.PREDICTIVE].n, klass, subsumed, tuple(flags), rec, era)

    def _recommend(self, klass: PairClass, a: str, b: str, subsumed: str | None, m: Measurer) -> str:
        if klass is PairClass.TRUE_DUPLICATE:
            keep, drop = self._preference(a, b, m)
            return f"archive {drop}, keep {keep} (duplicate on predictive+context+risk; retirement is reversible)"
        if klass is PairClass.DIFFERENT_PROTECTION:
            return "KEEP BOTH: same prediction but different regime/risk protection - never retire on correlation alone"
        if klass is PairClass.COMPLEMENT:
            return "KEEP BOTH: they cover each other's weak contexts or tail losses"
        if klass is PairClass.UNRESOLVED:
            return "collect more evidence (context/risk data) before any retirement decision"
        if subsumed:
            return f"{subsumed} already explains the other's predictive value; retain the other only if its context differs"
        return "no action"

    def _preference(self, a: str, b: str, m: Measurer) -> tuple[str, str]:
        """Which duplicate to keep: higher standalone IC, then lower cost, then name (deterministic)."""
        def ic(x):
            s = m._sig(x)
            ok = np.isfinite(s) & np.isfinite(m.y)
            return spearman(s[ok], m.y[ok]) if ok.sum() > 5 else float("nan")
        ia, ib = ic(a), ic(b)
        ca = m.profiles[a].cost if a in m.profiles else 1.0
        cb = m.profiles[b].cost if b in m.profiles else 1.0
        key = lambda x, i, c: (-(i if np.isfinite(i) else -9), c, x)
        return (a, b) if key(a, ia, ca) <= key(b, ib, cb) else (b, a)

    def analyze(self, panel: ObservationPanel, profiles: Mapping[str, ItemProfile] | Sequence[ItemProfile], now,
                items: Sequence[str] | None = None) -> RedundancyReport:
        if not isinstance(profiles, Mapping):
            profiles = {p.item_id: p for p in profiles}
        for p in profiles.values():
            errs = p.validate()
            if errs:
                raise ValueError("; ".join(errs))
        items = list(items) if items is not None else list(panel.signals.columns)
        if panel.n == 0 or len(items) < 2:
            return RedundancyReport(self.cfg, current_code_hash(), str(as_date(now)), panel.n, panel.dropped_future, (), {}, (), (),
                                    stable_hash(items))
        latest = (panel.dates.max() + pd.Timedelta(days=panel.horizon_days))
        require_past(latest, now, "newest observation outcome")
        pairs = tuple(self.analyze_pair(panel, profiles, a, b) for a, b in itertools.combinations(items, 2))
        for p in pairs:
            errs = p.validate()
            if errs:
                raise ValueError("; ".join(errs))
        clusters = duplicate_clusters(pairs)
        retire = self.coverage_preserving_reduction(panel, profiles, pairs)
        return RedundancyReport(self.cfg, current_code_hash(), str(as_date(now)), panel.n, panel.dropped_future, pairs,
                                variance_inflation(panel, items), clusters, tuple(retire),
                                stable_hash({"n": panel.n, "items": items, "seed": self.cfg.seed}))

    def coverage_preserving_reduction(self, panel: ObservationPanel, profiles, pairs: Sequence[PairRedundancy]) -> list:
        """For each TRUE_DUPLICATE pair propose archiving the weaker member, but only if the remaining items still cover every
        context the pair covered (a duplicate that is the only effective item somewhere must stay)."""
        m = Measurer(panel, profiles, self.cfg)
        eff = {it: m.effective_contexts(it) for it in panel.signals.columns}
        removed: set = set()
        out = []
        for p in sorted(pairs, key=lambda p: (p.a, p.b)):
            if p.klass is not PairClass.TRUE_DUPLICATE or p.a in removed or p.b in removed:
                continue
            keep, drop = self._preference(p.a, p.b, m)
            others = set().union(*[eff[i] for i in eff if i != drop and i not in removed]) if len(eff) > 1 else set()
            lost = eff[drop] - others
            if lost:
                out.append((drop, keep, f"NOT archived: only effective in {sorted(lost)}"))
                continue
            removed.add(drop)
            out.append((drop, keep, "duplicate; every context it covered stays covered"))
        return [o for o in out if not o[2].startswith("NOT")] + [o for o in out if o[2].startswith("NOT")]


def _stable_int(s: str) -> int:
    return int(stable_hash(s, 8), 16) % (2 ** 31)


def duplicate_clusters(pairs: Sequence[PairRedundancy]) -> tuple:
    """Connected components of the TRUE_DUPLICATE graph, sorted; singletons omitted."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        if p.klass is PairClass.TRUE_DUPLICATE:
            parent[find(p.a)] = find(p.b)
    groups: dict = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values() if len(g) > 1))


def variance_inflation(panel: ObservationPanel, items: Sequence[str], ridge: float = 1e-6) -> dict:
    """VIF_i = 1/(1-R_i^2) from the rank-correlation matrix of the items on rows where every item has an output. A VIF
    above ~5 means an item is mostly a linear mix of the others (multi-way predictive redundancy that no pair reveals)."""
    sub = panel.signals[list(items)].dropna()
    if len(sub) < max(20, 3 * len(items)):
        return {}
    R = sub.rank().corr().to_numpy()
    R = R + ridge * np.eye(len(items))
    inv = np.linalg.pinv(R)
    return {it: float(max(1.0, inv[i, i])) for i, it in enumerate(items)}


def marginal_information(panel: ObservationPanel, items: Sequence[str]) -> pd.DataFrame:
    """R^2 of rank(y) on the ranks of `items`, and each item's drop-one loss of R^2: the multi-item version of 'what does
    this item add that the others do not'. Rows where any item is missing are dropped."""
    cols = list(items)
    d = panel.signals[cols].copy()
    d["_y"] = panel.y.to_numpy()
    d = d.dropna()
    if len(d) < max(30, 5 * len(cols)):
        return pd.DataFrame(columns=["item", "r2_full", "r2_without", "unique_r2"])
    Rk = d.rank()
    y = Rk["_y"].to_numpy() - Rk["_y"].mean()

    def r2(names):
        if not names:
            return 0.0
        X = Rk[names].to_numpy() - Rk[names].to_numpy().mean(axis=0)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum())
    full = r2(cols)
    return pd.DataFrame([{"item": c, "r2_full": full, "r2_without": r2([x for x in cols if x != c]),
                          "unique_r2": full - r2([x for x in cols if x != c])} for c in cols])


# ---------------------------------------------------------------------------------------------- planted synthetic panel
def planted_panel(seed: int = 0, n: int = 1600, horizon_days: int = 5) -> tuple[ObservationPanel, dict]:
    """A truth environment whose redundancy structure is known by construction (contract section 63 style). The market has
    'calm' days (base signal is right) and 'crisis' days (~22%; the base signal is WRONG). Items:
      base    real signal in calm, loses in crisis
      dup     base plus tiny noise: a TRUE DUPLICATE of base
      guard   equals base in calm, stands aside in crisis: SAME PREDICTION, DIFFERENT PROTECTION (no crisis losses)
      hedge   only speaks in crisis and is rewarded there: a COMPLEMENT of base
      noise   unrelated random output: INDEPENDENT
    Returns (panel, truth) mapping frozenset({a, b}) -> the PairClass the analyzer should recover."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n // 4)
    d = np.repeat(np.arange(len(dates)), 4)[:n]
    crisis_day = rng.random(len(dates)) < 0.22
    regime = np.where(crisis_day[d], "crisis", "calm")
    base = rng.standard_normal(n)
    hedge_sig = np.where(regime == "crisis", np.abs(rng.standard_normal(n)) + 0.3, rng.standard_normal(n) * 0.1)
    y = np.where(regime == "calm", 0.3 * base, -0.3 * base) + 0.9 * rng.standard_normal(n)
    y = y + np.where(regime == "crisis", 0.35 * hedge_sig, 0.0)
    dup = base + 0.05 * rng.standard_normal(n)
    guard = np.where(regime == "crisis", rng.standard_normal(n) * 0.05, base + 0.05 * rng.standard_normal(n))
    noise = rng.standard_normal(n)
    sig = pd.DataFrame({"base": base, "dup": dup, "guard": guard, "hedge": hedge_sig, "noise": noise})
    sig.loc[rng.random(n) < 0.05, "dup"] = np.nan                       # outages so operational redundancy is testable
    sig.loc[rng.random(n) < 0.05, "base"] = np.nan
    panel = ObservationPanel(sig, pd.Series(y), pd.DatetimeIndex(dates[d]), pd.Series(regime, dtype=object), horizon_days)
    T = PairClass
    truth = {frozenset(("base", "dup")): T.TRUE_DUPLICATE, frozenset(("base", "guard")): T.DIFFERENT_PROTECTION,
             frozenset(("base", "hedge")): T.COMPLEMENT, frozenset(("base", "noise")): T.INDEPENDENT}
    return panel, truth


# ---------------------------------------------------------------------------------------------- conditional and temporal views
def conditional_predictive(panel: ObservationPanel, a: str, b: str, cfg: RedundancyConfig | None = None) -> pd.DataFrame:
    """Predictive redundancy INSIDE each context: n, rank correlation of the two signals, each one's IC, and the partial ICs.
    Two items that are near-identical in calm markets and unrelated in a crisis are exactly the pair that pooled correlation
    calls redundant and a regime-aware learner must keep both of."""
    cfg = cfg or RedundancyConfig()
    m = Measurer(panel, {}, cfg)
    sa, sb = m._sig(a), m._sig(b)
    rows = []
    for g in sorted(set(m.ctx)):
        ov = (m.ctx == g) & np.isfinite(sa) & np.isfinite(sb) & np.isfinite(m.y)
        if ov.sum() < cfg.min_context_n:
            continue
        ra, rb, ry = _rank(sa[ov]), _rank(sb[ov]), _rank(m.y[ov])
        rows.append({"context": g, "n": int(ov.sum()), "rho": _corr(ra, rb), "ic_a": _corr(ra, ry), "ic_b": _corr(rb, ry),
                     "partial_a": partial_corr(ra, ry, rb), "partial_b": partial_corr(rb, ry, ra)})
    return pd.DataFrame(rows, columns=["context", "n", "rho", "ic_a", "ic_b", "partial_a", "partial_b"])


def redundancy_over_time(panel: ObservationPanel, a: str, b: str, freq: str = "Q", min_n: int = 40) -> pd.DataFrame:
    """Rank correlation of two items per calendar period. A duplicate that was only ever a duplicate in one era is not one."""
    sa, sb = panel.signals[a].to_numpy(dtype=float), panel.signals[b].to_numpy(dtype=float)
    ok = np.isfinite(sa) & np.isfinite(sb)
    per = panel.dates.to_period(freq)
    rows = []
    for p in sorted(set(per[ok])):
        m = ok & (np.asarray(per) == p)
        if m.sum() >= min_n:
            rows.append({"period": str(p), "n": int(m.sum()), "rho": spearman(sa[m], sb[m])})
    return pd.DataFrame(rows, columns=["period", "n", "rho"])


def redundancy_drift(over_time: pd.DataFrame, gap: float = 0.30) -> dict:
    """max-min of per-period correlation, and whether it exceeds `gap` (then a single pooled number misleads)."""
    r = over_time["rho"].dropna().to_numpy(dtype=float) if len(over_time) else np.array([])
    if len(r) < 3:
        return {"n_periods": int(len(r)), "range": float("nan"), "unstable": None}
    return {"n_periods": int(len(r)), "range": float(np.ptp(r)), "unstable": bool(np.ptp(r) > gap)}


# ---------------------------------------------------------------------------------------------- forward selection
def forward_select(panel: ObservationPanel, items: Sequence[str], min_gain: float = 0.002, max_items: int | None = None) -> list:
    """Greedy non-redundant subset: repeatedly add the item with the largest gain in R^2 of rank(y) on ranks of the chosen
    set, stop when the best gain is below `min_gain` (an in-sample penalty for complexity). Returns
    [(item, cumulative_r2, gain)]. Items that never get chosen are predictively redundant GIVEN the chosen ones - which is a
    statement about prediction only; use the pair report before acting on it."""
    d = panel.signals[list(items)].copy()
    d["_y"] = panel.y.to_numpy()
    d = d.dropna()
    if len(d) < max(30, 5 * len(items)):
        return []
    Rk = d.rank()
    y = Rk["_y"].to_numpy() - Rk["_y"].mean()

    def r2(cols):
        if not cols:
            return 0.0
        X = Rk[cols].to_numpy()
        X = X - X.mean(axis=0)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum())
    chosen, cur, out = [], 0.0, []
    remaining = list(items)
    while remaining and (max_items is None or len(chosen) < max_items):
        gains = {c: r2(chosen + [c]) - cur for c in remaining}
        best = max(gains, key=lambda c: (gains[c], c))
        if gains[best] < min_gain:
            break
        chosen.append(best)
        remaining.remove(best)
        cur += gains[best]
        out.append((best, cur, gains[best]))
    return out


# ---------------------------------------------------------------------------------------------- is retiring one safe?
def retirement_safety(panel: ObservationPanel, keep: str, drop: str, tail_q: float = 0.10, tolerance: float = 0.15) -> dict:
    """Simulate following {keep, drop} equally versus following `keep` alone, per date, and compare the mean, the lower-tail
    mean and the worst date. Retiring `drop` is 'safe' only if the tail does not deteriorate by more than `tolerance`
    (relative) - the empirical form of 'do not lose the protection'."""
    m = Measurer(panel, {}, RedundancyConfig())
    both = (m.date_pnl(keep) + m.date_pnl(drop)) / 2
    alone = m.date_pnl(keep)
    idx = both.index.intersection(alone.index)
    both, alone = both.loc[idx], alone.loc[idx]
    if len(idx) < 30:
        return {"safe": None, "reason": "fewer than 30 dates"}
    k = max(3, int(round(tail_q * len(idx))))
    tail = lambda s: float(np.sort(s.to_numpy())[:k].mean())
    tb, ta = tail(both), tail(alone)
    worse = (tb - ta) / max(abs(tb), 1e-12)                   # positive = alone has the worse tail
    safe = bool(worse <= tolerance)
    return {"safe": safe, "mean_both": float(both.mean()), "mean_alone": float(alone.mean()), "tail_both": tb, "tail_alone": ta,
            "worst_both": float(both.min()), "worst_alone": float(alone.min()), "tail_deterioration": float(worse),
            "reason": "tail protection preserved" if safe else "removing it would deepen the lower tail"}


# ---------------------------------------------------------------------------------------------- report checks
def validate_report(rep: RedundancyReport) -> list[str]:
    """Structural invariants: ranges, symmetry of the matrices, edge consistency, no duplicate declared on untested kinds."""
    errs = []
    seen = set()
    for p in rep.pairs:
        errs += p.validate()
        key = frozenset((p.a, p.b))
        if key in seen:
            errs.append(f"pair {p.a}~{p.b} reported twice")
        seen.add(key)
        for src, e, dst, w, _ in p.edges():
            if not (0.0 <= w <= 1.0):
                errs.append(f"edge {src}-{e.value}->{dst} weight {w} outside [0,1]")
            if e is Edge.REDUNDANT_WITH and p.klass is PairClass.COMPLEMENT:
                errs.append(f"{p.a}~{p.b}: COMPLEMENT pair emitted a REDUNDANT_WITH edge")
    for k in RedundancyKind:
        M = rep.matrix(k)
        if len(M) and not np.allclose(M.fillna(-1).to_numpy(), M.fillna(-1).to_numpy().T):
            errs.append(f"{k.value} matrix not symmetric")
    dup_pairs = {frozenset((p.a, p.b)) for p in rep.pairs if p.klass is PairClass.TRUE_DUPLICATE}
    for cl in rep.clusters:
        for x, y in itertools.combinations(cl, 2):
            if frozenset((x, y)) not in dup_pairs and len(cl) == 2:
                errs.append(f"cluster {cl} lacks its duplicate pair")
    return errs


def run_planted_calibration(seed: int = 0, cfg: RedundancyConfig | None = None) -> dict:
    """Run the analyzer on the planted panel and compare each pair's class with the truth. Calibration harness, not evidence."""
    panel, truth = planted_panel(seed)
    profs = [ItemProfile("base", ("momentum",), ("prices",)), ItemProfile("dup", ("momentum",), ("prices",)),
             ItemProfile("guard", ("momentum", "stop"), ("prices",)), ItemProfile("hedge", ("flight",), ("vix",)),
             ItemProfile("noise")]
    rep = RedundancyAnalyzer(cfg).analyze(panel, profs, "2035-01-01")
    got = {tuple(sorted(k)): rep.pair(*sorted(k)).klass for k in truth}
    ok = {tuple(sorted(k)): rep.pair(*sorted(k)).klass is v for k, v in truth.items()}
    return {"n_correct": sum(ok.values()), "n_pairs": len(ok), "classes": {"~".join(k): v.value for k, v in got.items()},
            "errors": validate_report(rep), "report": rep}
