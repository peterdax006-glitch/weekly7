"""Missed-winner learning (contract C62 section 22; checklist D13 missed-winner analysis, D14 counterfactual distinction
discovery). Built ON engine/missed_winners.py (why_missed, winner_type, MissedLedger, sign_flip_p, boot_ci, rank_ic); it
adds the two things that module does not do:

  1. WHY did the system reject the winner?  RejectionAnalyzer answers with one of the section-22 reasons
     (wrong ranking, wrong confidence, wrong context, bad reliability, missing interaction, over-aggressive risk filter,
     direction disagreement, timing, insufficient evidence, unknown) using the candidate's own decision trail. Reasons
     are ordered by MECHANISM precedence - a stock removed by a hard filter never reached the ranking, so the filter is
     the cause even if its rank would have been fine. Nothing is forced: no supported reason -> UNKNOWN.
  2. WHAT distinction would have separated the missed winners from the false positives?  find_distinctions() searches
     single and paired threshold conditions over cross-sectionally ranked features for the rule that best separates the
     two cohorts, and controls the search: the p-value of a distinction is a family-wise permutation p (labels shuffled
     WITHIN each period, the entire search re-run each time, the maximum statistic recorded), so a distinction that
     merely won a big search does not look significant. A complexity penalty prefers the simpler rule.
  3. Is the distinction REAL out of sample?  validate_distinction() applies a rule discovered on earlier periods to later
     periods it never saw (with an embargo), measuring lift, paired weekly gain versus the displaced picks, a random-swap
     control, sign-flip p and a bootstrap CI (statistics come from learning_delta and pattern_stats, not local copies). `walk_forward_distinctions` is the out-of-sample hook, and `null_control`
     runs the identical pipeline on shuffled winner labels: its pass rate is the false-discovery rate of the whole method.
     An external validator can be supplied as any callable with the OOSHook signature.

A distinction is only ever a HYPOTHESIS (distinction_to_hypothesis); it changes nothing in production. Candidates carry an
opaque id, never a ticker; features are cross-sectional ranks, so levels that identify names cannot be learned.
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import datetime as dt
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import learning_delta as LD
from engine import missed_winners as base
from engine import pattern_stats as PS
from engine.learning.core import DecisionEffect, Subsystem, as_date, clip01, require_past, stable_hash
from engine.learning.failure import evaluate_contexts, ramp
from engine.learning.postmortem import Hypothesis

NL = chr(10)
WINNER = base.WINNER


class RejectionReason(str, enum.Enum):
    WRONG_RANKING = "WRONG_RANKING"
    WRONG_CONFIDENCE = "WRONG_CONFIDENCE"
    WRONG_CONTEXT = "WRONG_CONTEXT"
    BAD_RELIABILITY = "BAD_RELIABILITY"
    MISSING_INTERACTION = "MISSING_INTERACTION"
    OVER_AGGRESSIVE_RISK_FILTER = "OVER_AGGRESSIVE_RISK_FILTER"
    DIRECTION_DISAGREEMENT = "DIRECTION_DISAGREEMENT"
    TIMING = "TIMING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value


RR = RejectionReason
# mechanism precedence: earlier reasons remove a candidate BEFORE later ones can matter
PRECEDENCE = (RR.OVER_AGGRESSIVE_RISK_FILTER, RR.DIRECTION_DISAGREEMENT, RR.TIMING, RR.WRONG_CONTEXT, RR.BAD_RELIABILITY,
              RR.WRONG_CONFIDENCE, RR.MISSING_INTERACTION, RR.WRONG_RANKING, RR.INSUFFICIENT_EVIDENCE)


@dataclass(frozen=True)
class MissedParams:
    accept: float = 0.4                   # a reason needs this much support to be named
    near_band: float = 0.5                # rank within (1 + near_band) * k counts as a near-miss
    conf_gate: float = 0.5
    rel_gate: float = 0.4
    missing_frac: float = 0.3
    inter_q: float = 0.7                  # both interacting features at/above this rank percentile
    inter_single: float = 0.9             # ...while neither alone reached this one
    fp_thr: float = 0.0                   # a false positive is a candidate in the ranked band that returned less than this
    band: float = 3.0                     # cohorts are drawn from ranks <= band * k (the region where the score is comparable)
    match: bool = True                    # pair missed winners with false positives of nearby rank within each period
    rank_caliper: int = 6
    min_a: int = 12                       # cohort sizes below which no distinction is searched
    min_b: int = 12
    min_cover: float = 0.08               # a rule must cover at least this share of the missed winners
    min_cover_abs: int = 5
    complexity_penalty: float = 0.03
    grid: tuple = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9)
    pair_grid: tuple = (0.35, 0.65, 0.85)
    n_perm: int = 200
    n_boot: int = 100
    top_n: int = 3
    max_overlap: float = 0.7
    alpha: float = 0.10
    embargo: int = 1                      # periods between the last training period and the first test period
    min_test_weeks: int = 6
    min_lift: float = 1.3
    swap_n: int = 3
    boot: int = 1000


# ==================================================================================================================
# data model
# ==================================================================================================================
@dataclass(frozen=True)
class Candidate:
    """One candidate of one decision period, with its decision trail. `fwd` is the outcome and is read ONLY by the
    ex-post analyses below; the rejection analysis never looks at it."""
    cid: str
    features: Mapping[str, float]
    fwd: float | None = None
    picked: bool = False
    score: float | None = None
    rank: int | None = None
    eligible: bool = True
    filters_hit: tuple[str, ...] = ()
    confidence: float | None = None
    dir_side: int | None = None
    selection_side: int | None = None
    reliability: float | None = None
    anti_context: bool = False
    timing_blocked: bool = False
    missing_frac: float = 0.0
    kind: str = "other"                   # winner_type label

    def validate(self) -> list[str]:
        errs = []
        if not self.cid:
            errs.append("cid missing")
        if any(str(k).lower() in ("ticker", "symbol", "name") for k in self.features):
            errs.append("identity key in features")
        for k, v in self.features.items():
            if v is not None and isinstance(v, float) and math.isinf(v):
                errs.append(f"feature {k} infinite")
        return errs


@dataclass(frozen=True)
class Week:
    """One decision period. `resolved_at` is when the outcomes matured; analyses take `now` and refuse weeks that
    resolve at or after it."""
    label: str
    decided_at: str
    resolved_at: str
    candidates: tuple[Candidate, ...]
    k: int = 10
    thr: float = WINNER
    era: str = ""
    interactions: tuple[tuple[str, str], ...] = ()      # feature pairs the knowledge base believes interact

    def validate(self) -> list[str]:
        errs = []
        if as_date(self.resolved_at) <= as_date(self.decided_at):
            errs.append(f"week {self.label}: resolved_at must be after decided_at")
        seen = set()
        for c in self.candidates:
            errs += c.validate()
            if c.cid in seen:
                errs.append(f"duplicate cid {c.cid}")
            seen.add(c.cid)
        return errs

    def winners(self) -> list[Candidate]:
        return [c for c in self.candidates if c.fwd is not None and c.fwd >= self.thr]

    def missed(self) -> list[Candidate]:
        return [c for c in self.winners() if not c.picked]

    def base_rate(self) -> float:
        known = [c for c in self.candidates if c.fwd is not None]
        return float(np.mean([c.fwd >= self.thr for c in known])) if known else float("nan")


def cross_rank(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Percentile ranks 0..1 within a cross-section; constant columns become 0.5 (they carry no ordering)."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[c].astype(float)
        out[c] = v.rank(pct=True) if v.nunique(dropna=True) > 1 else 0.5
    return out


def week_from_base(date, p0: pd.DataFrame, fwd: pd.Series, picked: Sequence, score: pd.Series | None = None,
                   eligible: pd.Series | None = None, k: int = 10, thr: float = WINNER, resolved_at: str | None = None,
                   era: str = "", salt: str = "mw", feats: Sequence[str] | None = None,
                   extra: Mapping[Any, Mapping[str, Any]] | None = None) -> Week:
    """Adapter from engine.missed_winners' (date, p0, fwd) weeks. Names are replaced by salted hashes, features by
    cross-sectional ranks; the base module's winner_type gives the coarse type and its DET_FEATS the default features."""
    feats = [c for c in (feats or base.DET_FEATS) if c in p0.columns]
    ranks = cross_rank(p0, feats)
    rk = score.rank(ascending=False, method="first") if score is not None else None
    pk = set(picked)
    extra = extra or {}
    cands = []
    for tkr in p0.index:
        ex = dict(extra.get(tkr, {}))
        elig = bool(eligible[tkr]) if eligible is not None and tkr in eligible.index else True
        hits = tuple(ex.pop("filters_hit", ()))
        if not elig and not hits:
            hits = ("ineligible",)
        f = float(fwd[tkr]) if fwd is not None and tkr in fwd.index and np.isfinite(fwd[tkr]) else None
        cands.append(Candidate(
            cid=stable_hash({"s": salt, "d": str(date), "t": str(tkr)}, 12),
            features={c: float(ranks.at[tkr, c]) for c in feats}, fwd=f, picked=tkr in pk,
            score=float(score[tkr]) if score is not None and tkr in score.index and np.isfinite(score[tkr]) else None,
            rank=int(rk[tkr]) if rk is not None and tkr in rk.index and np.isfinite(rk[tkr]) else None,
            eligible=elig, filters_hit=hits, kind=base.winner_type(p0, tkr), **ex))
    d0 = as_date(date)
    return Week(label=stable_hash({"s": salt, "d": str(date)}, 8), decided_at=str(d0),
                resolved_at=str(resolved_at or d0 + dt.timedelta(days=7)), candidates=tuple(cands), k=k, thr=thr, era=era)


# ==================================================================================================================
# 1. why was it rejected?  (D13)
# ==================================================================================================================
@dataclass(frozen=True)
class ReasonEvidence:
    reason: RejectionReason
    strength: float
    fact: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class Rejection:
    cid: str
    primary: RejectionReason
    reasons: tuple[ReasonEvidence, ...]           # every reason at/above accept, precedence order
    weak: tuple[ReasonEvidence, ...] = ()         # below accept: kept for audit, never counted
    note: str = ""

    @property
    def named(self) -> bool:
        return self.primary not in (RR.UNKNOWN, RR.INSUFFICIENT_EVIDENCE)


class RejectionAnalyzer:
    def __init__(self, params: MissedParams | None = None, risk_filters: Sequence[str] = ("risk", "vol", "gap", "liquidity", "size")):
        self.p = params or MissedParams()
        self.risk_filters = tuple(risk_filters)

    def evidence(self, c: Candidate, week: Week, n_ranked: int | None = None) -> list[ReasonEvidence]:
        p, ev = self.p, []
        hard = [f for f in c.filters_hit if any(f.startswith(r) or r in f for r in self.risk_filters)]
        if hard:
            inside = c.rank is not None and c.rank <= week.k
            ev.append(ReasonEvidence(RR.OVER_AGGRESSIVE_RISK_FILTER, 0.95 if inside else 0.7,
                                     {"filters": hard, "would_rank_inside_cut": inside},
                                     "removed by a risk filter before ranking" + (" although its rank cleared the cut" if inside else "")))
        elif c.filters_hit and not c.eligible:
            ev.append(ReasonEvidence(RR.UNKNOWN, 0.0, {"filters": list(c.filters_hit)}, "removed by a filter that is not a risk filter"))
        if c.dir_side is not None and c.selection_side is not None and c.dir_side != c.selection_side:
            ev.append(ReasonEvidence(RR.DIRECTION_DISAGREEMENT, 0.85, {"dir": c.dir_side, "sel": c.selection_side},
                                     "the direction model disagreed with the side selection wanted"))
        if c.timing_blocked:
            ev.append(ReasonEvidence(RR.TIMING, 0.8, {}, "the entry-timing gate held it back"))
        if c.anti_context:
            ev.append(ReasonEvidence(RR.WRONG_CONTEXT, 0.85, {}, "an anti-context of the selecting knowledge was active"))
        if c.reliability is not None and c.reliability < p.rel_gate:
            ev.append(ReasonEvidence(RR.BAD_RELIABILITY, ramp(p.rel_gate - c.reliability, 0.0, p.rel_gate), {"reliability": c.reliability},
                                     "the knowledge that favoured it was judged unreliable"))
        inside_rank = c.rank is not None and c.rank <= week.k * (1 + p.near_band)
        if c.confidence is not None and c.confidence < p.conf_gate and (inside_rank or c.rank is None):
            ev.append(ReasonEvidence(RR.WRONG_CONFIDENCE, ramp(p.conf_gate - c.confidence, 0.0, p.conf_gate) * 0.9 + 0.1,
                                     {"confidence": c.confidence}, "ranked well enough but confidence fell under the gate"))
        for a, b in week.interactions:
            fa, fb = c.features.get(a), c.features.get(b)
            if fa is not None and fb is not None and fa >= p.inter_q and fb >= p.inter_q and max(fa, fb) < p.inter_single:
                ev.append(ReasonEvidence(RR.MISSING_INTERACTION, 0.7, {"pair": (a, b), "ranks": (fa, fb)},
                                         "both interacting features were high but neither alone cleared the bar"))
        if c.rank is not None and c.rank > week.k and c.eligible and not c.filters_hit:
            n = n_ranked or max(1, len(week.candidates))
            over = (c.rank - week.k) / max(1.0, week.k * p.near_band)
            near = over <= 1.0
            ev.append(ReasonEvidence(RR.WRONG_RANKING, 0.8 - 0.3 * min(1.0, over - 1.0) if near else 0.45,
                                     {"rank": c.rank, "cut": week.k, "rank_pct": c.rank / n},
                                     "a near-miss under the cut" if near else "ranked well under the cut"))
        if c.missing_frac >= p.missing_frac:
            ev.append(ReasonEvidence(RR.INSUFFICIENT_EVIDENCE, ramp(c.missing_frac, p.missing_frac, 0.8), {"missing_frac": c.missing_frac},
                                     "too many of its inputs were missing to judge it"))
        return ev

    def explain(self, c: Candidate, week: Week, n_ranked: int | None = None) -> Rejection:
        if c.picked:
            raise ValueError(f"candidate {c.cid} was picked: nothing was rejected")
        ev = self.evidence(c, week, n_ranked)
        acc = [e for e in ev if e.strength >= self.p.accept and e.reason != RR.UNKNOWN]
        acc.sort(key=lambda e: PRECEDENCE.index(e.reason))
        weak = tuple(e for e in ev if e.strength < self.p.accept)
        if not acc:
            return Rejection(c.cid, RR.UNKNOWN, (), weak, "the decision trail supports no reason")
        primary = acc[0].reason
        if primary == RR.INSUFFICIENT_EVIDENCE and len(acc) > 1:
            primary = acc[1].reason
        return Rejection(c.cid, primary, tuple(acc), weak)

    def explain_missed(self, week: Week) -> list[Rejection]:
        n = len([c for c in week.candidates if c.rank is not None])
        return [self.explain(c, week, n) for c in week.missed()]


def filter_tradeoff(weeks: Sequence[Week], filter_name: str, now=None) -> dict[str, Any]:
    """What one filter cost and saved. Among candidates it removed: the winners foregone (count, mean return) against the
    losers avoided (count, mean loss). A filter 'over-aggressive' in one missed winner may be excellent overall."""
    win_r, lose_r, n_hit, n_wk = [], [], 0, 0
    for w in weeks:
        if now is not None:
            require_past(w.resolved_at, now, f"week {w.label}")
        hit = [c for c in w.candidates if filter_name in c.filters_hit and c.fwd is not None]
        if not hit:
            continue
        n_wk += 1
        n_hit += len(hit)
        for c in hit:
            (win_r if c.fwd >= w.thr else lose_r).append(c.fwd)
    if n_hit == 0:
        return {"filter": filter_name, "removed": 0, "verdict": "NEVER_FIRED"}
    foregone = float(np.sum(win_r)) if win_r else 0.0
    avoided = float(-np.sum([r for r in lose_r if r < 0])) if lose_r else 0.0
    return {"filter": filter_name, "removed": n_hit, "weeks": n_wk, "winners_foregone": len(win_r),
            "mean_winner_ret": float(np.mean(win_r)) if win_r else float("nan"),
            "losers_avoided": sum(1 for r in lose_r if r < 0), "mean_loser_ret": float(np.mean(lose_r)) if lose_r else float("nan"),
            "gain_foregone": foregone, "loss_avoided": avoided,
            "verdict": "COSTLY" if foregone > avoided * 1.5 else "PAYS" if avoided > foregone else "NEUTRAL"}


# ==================================================================================================================
# 2. cohorts and distinction discovery  (D14)
# ==================================================================================================================
META_COLS = ("period", "cid", "group", "era", "kind", "_rank")


def _match_by_rank(rows: list[dict], caliper: int) -> list[dict]:
    """Within each period, pair every missed winner with the not-yet-used false positive whose rank is nearest (1:1, greedy
    from the most extreme rank inward, within `caliper` places). Without this the comparison is confounded by construction:
    missed winners are un-picked (low score) and false positives are picked (high score), so ANY score-related feature
    'separates' them. Unmatched rows are dropped, which is why the cohort can shrink."""
    out: list[dict] = []
    by: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        by.setdefault(r["period"], {}).setdefault(r["group"], []).append(r)
    for per in sorted(by):
        A, B = by[per].get(1, []), list(by[per].get(0, []))
        for a_ in sorted(A, key=lambda r: r["_rank"]):
            if not B:
                break
            j = min(range(len(B)), key=lambda i: (abs(B[i]["_rank"] - a_["_rank"]), B[i]["cid"]))
            if abs(B[j]["_rank"] - a_["_rank"]) <= caliper:
                out += [a_, B.pop(j)]
    return out


def cohort_frame(weeks: Sequence[Week], p: MissedParams, now=None, feats: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows = missed winners (group 1: winners the system did not pick) and false positives (group 0: candidates that lost,
    whether picked or not). Scope is the ranked band (rank <= band * k) so both groups come from the same region of the
    score, and with p.match the groups are additionally paired on rank within each period (see _match_by_rank). Only weeks
    resolved before `now` are read. `_rank` is kept for diagnostics and is never a searchable feature."""
    rows = []
    for w in weeks:
        if now is not None:
            require_past(w.resolved_at, now, f"week {w.label}")
        for c in w.candidates:
            if c.fwd is None or c.rank is None or c.rank > p.band * w.k:
                continue
            if c.fwd >= w.thr and not c.picked:
                g = 1
            elif c.fwd < p.fp_thr:
                g = 0
            else:
                continue
            rows.append({"period": w.label, "cid": c.cid, "group": g, "era": w.era, "kind": c.kind, "_rank": int(c.rank),
                         **{k: v for k, v in c.features.items()}})
    if p.match:
        rows = _match_by_rank(rows, p.rank_caliper)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(META_COLS))
    cols = list(feats) if feats else [c for c in df.columns if c not in META_COLS]
    keep = [c for c in cols if c in df.columns and df[c].notna().mean() > 0.8 and base.check_feature_pit(c)]
    df[keep] = df[keep].astype(float).fillna(0.5)
    return df[list(META_COLS) + keep]


def cohort_balance(df: pd.DataFrame) -> dict[str, float]:
    """How comparable are the two cohorts on the selection score? A rank AUC far from 0.5 means a 'distinction' may only be
    the score in disguise."""
    from engine import pattern_reliability as PR
    if df.empty or "_rank" not in df or (df["group"] == 1).sum() == 0 or (df["group"] == 0).sum() == 0:
        return {"n_a": 0, "n_b": 0, "rank_auc": float("nan"), "verdict": "EMPTY"}
    auc = float(PR.auc(df["_rank"].to_numpy(dtype=float), df["group"].to_numpy(dtype=bool)))
    return {"n_a": int((df["group"] == 1).sum()), "n_b": int((df["group"] == 0).sum()), "rank_auc": auc,
            "mean_rank_gap": float(df.loc[df["group"] == 1, "_rank"].mean() - df.loc[df["group"] == 0, "_rank"].mean()),
            "verdict": "BALANCED" if abs(auc - 0.5) <= 0.1 else "SCORE_CONFOUNDED"}


Cond = tuple[str, str, float]


def _cond_mask(df: pd.DataFrame, conds: Sequence[Cond]) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for f, op, thr in conds:
        v = df[f].to_numpy(dtype=float)
        m &= (v > thr) if op == ">" else (v <= thr)
    return m


@dataclass(frozen=True)
class Distinction:
    did: str
    conds: tuple[Cond, ...]
    score: float                          # tpr - fpr - complexity penalty
    tpr: float
    fpr: float
    lift: float                           # P(missed winner | rule) / P(missed winner)
    support_a: int
    support_b: int
    p_fwer: float
    stability: float
    n_a: int
    n_b: int

    def describe(self) -> str:
        return " AND ".join(f"{f} {op} {t:.2f}" for f, op, t in self.conds)

    def apply(self, feats: Mapping[str, float]) -> bool:
        for f, op, thr in self.conds:
            v = feats.get(f)
            if v is None:
                return False
            if not (v > thr if op == ">" else v <= thr):
                return False
        return True


def _build_conditions(df: pd.DataFrame, feats: Sequence[str], p: MissedParams) -> tuple[np.ndarray, list[tuple[Cond, ...]]]:
    """Label-free matrix of candidate rules (thresholds come from pooled quantiles, never from the labels), so the
    permutation test can re-score the SAME family under shuffled labels."""
    cols, meta = [], []
    thr_of = {f: np.unique(np.quantile(df[f].to_numpy(dtype=float), p.grid)) for f in feats}
    for f in feats:
        for t in thr_of[f]:
            for op in (">", "<="):
                cols.append(_cond_mask(df, [(f, op, float(t))]))
                meta.append(((f, op, float(t)),))
    pair_thr = {f: np.unique(np.quantile(df[f].to_numpy(dtype=float), p.pair_grid)) for f in feats}
    for i, fa in enumerate(feats):
        for fb in feats[i + 1:]:
            for ta in pair_thr[fa]:
                for oa in (">", "<="):
                    ma = _cond_mask(df, [(fa, oa, float(ta))])
                    for tb in pair_thr[fb]:
                        for ob in (">", "<="):
                            cols.append(ma & _cond_mask(df, [(fb, ob, float(tb))]))
                            meta.append(((fa, oa, float(ta)), (fb, ob, float(tb))))
    return np.asarray(cols, dtype=np.float32).T if cols else np.zeros((len(df), 0), np.float32), meta


def _scores(C: np.ndarray, y: np.ndarray, penalty: np.ndarray, min_a: np.ndarray) -> np.ndarray:
    """Youden statistic tpr - fpr per rule, minus complexity, -inf where the rule covers too few missed winners."""
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    a = y.astype(np.float32) @ C
    b = (1.0 - y).astype(np.float32) @ C
    s = a / n1 - b / n0 - penalty
    return np.where(a >= min_a, s, -np.inf)


def _perm_within(codes: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels within each period (pattern_stats.permute_within_clusters): every period keeps its own share of
    missed winners, so period-level luck is preserved while the link to features is destroyed."""
    return PS.permute_within_clusters(y, codes, rng)


def find_distinctions(df: pd.DataFrame, p: MissedParams | None = None, seed: int = 0, feats: Sequence[str] | None = None
                      ) -> list[Distinction]:
    """Rules that best separate missed winners (group 1) from false positives (group 0), with family-wise significance.
    Returns [] (not a weak guess) when a cohort is too small."""
    p = p or MissedParams()
    if df.empty or "group" not in df:
        return []
    feats = [f for f in (feats or [c for c in df.columns if c not in META_COLS])
             if f in df.columns and df[f].nunique() > 1]
    n_a, n_b = int((df["group"] == 1).sum()), int((df["group"] == 0).sum())
    if n_a < p.min_a or n_b < p.min_b or len(feats) < 1:
        return []
    C, meta = _build_conditions(df, feats, p)
    if C.shape[1] == 0:
        return []
    y = df["group"].to_numpy(dtype=np.float32)
    period = df["period"].to_numpy()
    pcodes = pd.factorize(df["period"])[0]
    pen = np.array([p.complexity_penalty * (len(m) - 1) for m in meta], dtype=np.float32)
    min_a = np.full(C.shape[1], max(p.min_cover_abs, math.ceil(p.min_cover * n_a)), dtype=np.float32)
    obs = _scores(C, y, pen, min_a)
    rng = np.random.default_rng(seed)
    perm_max = np.empty(p.n_perm)
    for i in range(p.n_perm):
        perm_max[i] = _scores(C, _perm_within(pcodes, y, rng), pen, min_a).max()
    order = np.argsort(-obs, kind="stable")
    chosen: list[int] = []
    covers: list[np.ndarray] = []
    for j in order:
        if not np.isfinite(obs[j]) or obs[j] <= 0 or len(chosen) >= p.top_n:
            break
        cov = (C[:, j] > 0) & (y > 0)
        if any((cov & c).sum() / max(1, (cov | c).sum()) > p.max_overlap for c in covers):
            continue                                       # a near-copy of a rule already chosen
        chosen.append(int(j))
        covers.append(cov)
    out = []
    pers = np.unique(period)
    for j in chosen:
        m = C[:, j] > 0
        tpr, fpr = float(m[y > 0].mean()), float(m[y == 0].mean())
        prec = float(y[m].mean()) if m.any() else 0.0
        lift = prec / max(1e-9, float(y.mean()))
        pfw = float((1 + (perm_max >= obs[j]).sum()) / (1 + p.n_perm))
        pos = 0
        for _ in range(p.n_boot):
            samp = rng.choice(pers, size=len(pers), replace=True)
            idx = np.concatenate([np.flatnonzero(period == s) for s in samp])
            yy, mm = y[idx], m[idx]
            if yy.sum() > 0 and (1 - yy).sum() > 0 and mm[yy > 0].mean() - mm[yy == 0].mean() > 0:
                pos += 1
        conds = meta[j]
        out.append(Distinction(stable_hash({"conds": [list(c) for c in conds]}, 10), conds, float(obs[j]), tpr, fpr, lift,
                               int(m[y > 0].sum()), int(m[y == 0].sum()), pfw, pos / max(1, p.n_boot), n_a, n_b))
    return out


def distinctions_by_reason(weeks: Sequence[Week], analyzer: RejectionAnalyzer, p: MissedParams, seed: int = 0, now=None
                           ) -> dict[str, list[Distinction]]:
    """Run the search separately for missed winners with the same primary rejection reason: different mechanisms need
    different distinctions. Groups too small for a search are absent, not guessed."""
    df = cohort_frame(weeks, p, now)
    if df.empty:
        return {}
    reason_of: dict[str, str] = {}
    for w in weeks:
        for rej in analyzer.explain_missed(w):
            reason_of[rej.cid] = rej.primary.value
    df = df.assign(reason=df["cid"].map(reason_of).fillna("PICKED"))
    out = {}
    for r in sorted(set(reason_of.values())):
        sub = df[(df["reason"] == r) | (df["group"] == 0)]
        found = find_distinctions(sub.drop(columns=["reason"]), p, seed)
        if found:
            out[r] = found
    return out


# ==================================================================================================================
# 3. out-of-sample validation
# ==================================================================================================================
@dataclass(frozen=True)
class OOSResult:
    did: str
    n_weeks: int
    n_swaps: int
    precision: float                      # P(winner | rule, not picked) pooled over test weeks
    base_rate: float                      # P(winner | not picked)
    lift: float
    mean_gain: float                      # mean weekly gain of swapping rule-matches in for the lowest-scored picks
    ci: tuple[float, float]
    p_value: float                        # one-sided sign-flip p that the mean gain is > 0
    excess_vs_random: float               # gain minus the gain of swapping in random non-picked names of the same count
    verdict: str                          # PASS | FAIL | INSUFFICIENT
    failed: tuple[str, ...] = ()


OOSHook = Callable[[Distinction, Sequence[Week]], OOSResult]


def validate_distinction(d: Distinction, test_weeks: Sequence[Week], p: MissedParams | None = None, seed: int = 0) -> OOSResult:
    """Apply a rule found earlier to weeks it never saw. It passes only if it lifts the win rate among the names the
    system did NOT pick, and swapping rule-matches in for the weakest picks would have paid, beyond a random-swap control."""
    p = p or MissedParams()
    rng = np.random.default_rng(seed)
    gains, rgains, hits, sel, np_wins, np_n = [], [], 0, 0, 0, 0
    for w in test_weeks:
        cs = [c for c in w.candidates if c.fwd is not None]
        nonp = [c for c in cs if not c.picked]
        picks = [c for c in cs if c.picked]
        if not nonp or not picks:
            continue
        np_n += len(nonp)
        np_wins += sum(c.fwd >= w.thr for c in nonp)
        match = [c for c in nonp if d.apply(c.features)]
        sel += len(match)
        hits += sum(c.fwd >= w.thr for c in match)
        if not match:
            continue
        m = min(p.swap_n, len(match), len(picks))
        best = sorted(match, key=lambda c: -(c.score if c.score is not None else 0.0))[:m]
        worst = sorted(picks, key=lambda c: (c.score if c.score is not None else 0.0))[:m]
        gains.append(float(np.mean([c.fwd for c in best]) - np.mean([c.fwd for c in worst])))
        rnd = [nonp[i] for i in rng.choice(len(nonp), size=m, replace=False)]
        rgains.append(float(np.mean([c.fwd for c in rnd]) - np.mean([c.fwd for c in worst])))
    n_w = len(gains)
    base_rate = np_wins / np_n if np_n else float("nan")
    prec = hits / sel if sel else float("nan")
    lift = prec / base_rate if sel and base_rate and base_rate > 0 else float("nan")
    if n_w < p.min_test_weeks:
        return OOSResult(d.did, n_w, sel, prec, base_rate, lift, float(np.mean(gains)) if gains else float("nan"),
                         (float("nan"), float("nan")), 1.0, float("nan"), "INSUFFICIENT", (f"only {n_w} test weeks",))
    g = np.asarray(gains)
    diff = g - np.asarray(rgains)
    pv = LD.signflip_p(g, p.boot, seed) / 2.0 if g.mean() > 0 else 1.0     # two-sided test, one-sided claim
    _, lo, hi, _ = LD.boot_ci(g, p.boot, 0.95, seed)
    ci = (lo, hi)
    failed = []
    if not lift >= p.min_lift:
        failed.append(f"lift {lift:.2f} < {p.min_lift}")
    if not pv <= p.alpha:
        failed.append(f"p {pv:.3f} > {p.alpha}")
    if not g.mean() > 0:
        failed.append("mean swap gain not positive")
    if not diff.mean() > 0:
        failed.append("no better than a random swap")
    return OOSResult(d.did, n_w, sel, prec, base_rate, lift, float(g.mean()), ci, pv, float(diff.mean()),
                     "FAIL" if failed else "PASS", tuple(failed))


@dataclass(frozen=True)
class WalkForwardReport:
    folds: int
    discovered: int
    passed: int
    results: tuple[tuple[Distinction, OOSResult], ...]
    by_era: Mapping[str, tuple[int, int]]     # era -> (tested, passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.discovered if self.discovered else float("nan")


def walk_forward_distinctions(weeks: Sequence[Week], p: MissedParams | None = None, seed: int = 0, hook: OOSHook | None = None,
                              train_min: int = 20, fold_len: int = 10, now=None) -> WalkForwardReport:
    """Expanding-window discover-then-test. Weeks are sorted by decision date; the test block starts `embargo` periods after
    the training block ends, so a training outcome can never leak into the test rule. `hook` replaces the built-in
    validator (any callable with the OOSHook signature), which is how an external out-of-sample harness plugs in."""
    p = p or MissedParams()
    ws = sorted(weeks, key=lambda w: as_date(w.decided_at))
    if now is not None:
        for w in ws:
            require_past(w.resolved_at, now, f"week {w.label}")
    res: list[tuple[Distinction, OOSResult]] = []
    folds, era_stat = 0, {}
    start = train_min
    while start + p.embargo + fold_len <= len(ws):
        train, test = ws[:start], ws[start + p.embargo:start + p.embargo + fold_len]
        if as_date(train[-1].resolved_at) >= as_date(test[0].decided_at):
            start += fold_len                              # horizons overlap: refuse rather than leak
            continue
        folds += 1
        found = find_distinctions(cohort_frame(train, p), p, seed + folds)
        for d in found:
            r = (hook or (lambda dd, tw: validate_distinction(dd, tw, p, seed + folds)))(d, test)
            res.append((d, r))
            for w in test:
                t, ps = era_stat.get(w.era, (0, 0))
                era_stat[w.era] = (t + 1, ps + (1 if r.verdict == "PASS" else 0))
        start += fold_len
    passed = sum(1 for _, r in res if r.verdict == "PASS")
    return WalkForwardReport(folds, len(res), passed, tuple(res), era_stat)


def shuffle_labels(weeks: Sequence[Week], rng: np.random.Generator) -> list[Week]:
    """Permute the realised returns across candidates within each week (features, picks and ranks intact)."""
    out = []
    for w in weeks:
        cs = list(w.candidates)
        known = [i for i, c in enumerate(cs) if c.fwd is not None]
        vals = rng.permutation([cs[i].fwd for i in known])
        for i, v in zip(known, vals):
            cs[i] = Candidate(**{**c_dict(cs[i]), "fwd": float(v)})
        out.append(Week(w.label, w.decided_at, w.resolved_at, tuple(cs), w.k, w.thr, w.era, w.interactions))
    return out


def c_dict(c: Candidate) -> dict[str, Any]:
    return {f.name: getattr(c, f.name) for f in c.__dataclass_fields__.values()}


def null_control(weeks: Sequence[Week], p: MissedParams | None = None, reps: int = 5, seed: int = 0, **wf) -> dict[str, Any]:
    """The method's false-discovery rate: run the identical walk-forward on winner labels shuffled across names. If this
    passes distinctions at a rate near alpha or above, the pipeline is finding structure in noise and its real-data
    passes prove nothing. A rate near 0 with real passes is what learning looks like."""
    p = p or MissedParams()
    tested = passed = 0
    for r in range(reps):
        rep = walk_forward_distinctions(shuffle_labels(weeks, np.random.default_rng(seed + 1000 + r)), p, seed + r, **wf)
        tested += rep.discovered
        passed += rep.passed
    return {"reps": reps, "tested": tested, "passed": passed, "false_discovery_rate": passed / tested if tested else float("nan"),
            "alpha": p.alpha}


# ==================================================================================================================
# output as hypothesis, and the ledger
# ==================================================================================================================
def distinction_to_hypothesis(d: Distinction, oos: OOSResult | None = None) -> Hypothesis:
    status = "not yet tested out of sample" if oos is None else f"out-of-sample verdict {oos.verdict}" + (
        f" (lift {oos.lift:.2f}, p {oos.p_value:.3f})" if oos.verdict != "INSUFFICIENT" else "")
    return Hypothesis(
        hid=stable_hash({"dist": [list(c) for c in d.conds]}, 16),
        statement=f"promote candidates where {d.describe()} (separated missed winners from false positives; {status})",
        subsystem=Subsystem.SELECTION, decision_effect=DecisionEffect.RANKING, cause=__import__("engine.learning.core", fromlist=["FailureCause"]).FailureCause.SELECTION_ERROR,
        test_plan=("replay on periods after the training block with an embargo", "swap test against the weakest picks with random-swap control",
                   "shuffled-winner null control must not pass at the same rate"), target=d.describe())


class MissedLearningLedger:
    """Extends engine.missed_winners.MissedLedger with rejection reasons. The base ledger keeps the missed-minus-picked
    profile by feature, type and era; this one adds WHY, by reason, type and era, and how stable the reasons are."""

    def __init__(self, analyzer: RejectionAnalyzer | None = None):
        self.analyzer = analyzer or RejectionAnalyzer()
        self.base = base.MissedLedger()
        self.rows: list[dict[str, Any]] = []

    def add_week(self, w: Week) -> list[Rejection]:
        rejs = self.analyzer.explain_missed(w)
        winners = w.winners()
        picked = [c.cid for c in w.candidates if c.picked]
        prof = {}
        feats = sorted({f for c in w.candidates for f in c.features})
        for f in feats:
            mv = [c.features[f] for c in w.missed() if f in c.features]
            pv = [c.features[f] for c in w.candidates if c.picked and f in c.features]
            if mv and pv:
                prof[f] = float(np.mean(mv) - np.mean(pv))
        types: dict[str, int] = {}
        for c in w.missed():
            types[c.kind] = types.get(c.kind, 0) + 1
        self.base.add(w.decided_at, len(winners), len(w.missed()), prof, types=types, era=w.era or None)
        cmap = {c.cid: c for c in w.candidates}
        for r in rejs:
            self.rows.append({"period": w.label, "era": w.era, "reason": r.primary.value, "kind": cmap[r.cid].kind,
                              "named": r.named, "fwd": cmap[r.cid].fwd})
        return rejs

    def reason_table(self, by: str | None = None) -> pd.DataFrame:
        f = pd.DataFrame(self.rows)
        if f.empty:
            return pd.DataFrame(columns=["reason", "n", "share", "mean_fwd"])
        keys = ["reason"] if by is None else [by, "reason"]
        g = f.groupby(keys).agg(n=("reason", "size"), mean_fwd=("fwd", "mean")).reset_index()
        tot = g.groupby(by)["n"].transform("sum") if by else g["n"].sum()
        g["share"] = g["n"] / tot
        return g.sort_values(keys[:-1] + ["n"], ascending=[True] * (len(keys) - 1) + [False]).reset_index(drop=True)

    def unknown_share(self) -> float:
        f = pd.DataFrame(self.rows)
        return float((~f["named"]).mean()) if len(f) else float("nan")

    def reason_stability(self) -> dict[str, float]:
        """Share of each reason in the first half of periods minus the second half: large values mean the blind spot
        is moving, so a distinction learned early may not apply late."""
        f = pd.DataFrame(self.rows)
        if f.empty or f["period"].nunique() < 4:
            return {}
        pers = list(dict.fromkeys(f["period"]))
        h = len(pers) // 2
        a, b = f[f["period"].isin(pers[:h])], f[f["period"].isin(pers[h:])]
        out = {}
        for r in sorted(set(f["reason"])):
            out[r] = float((a["reason"] == r).mean() - (b["reason"] == r).mean()) if len(a) and len(b) else float("nan")
        return out

    def report(self) -> str:
        t = self.reason_table()
        lines = [f"missed-winner ledger: {len(self.rows)} rejections, unknown share {self.unknown_share():.2f}   IMPLEMENTED - NOT VALIDATED"]
        for _, r in t.iterrows():
            lines.append(f"  {r['reason']:<30}{int(r['n']):>5}  {r['share']:.2f}")
        return NL.join(lines)


# ==================================================================================================================
# are the rejection reasons themselves any good?  (a reason that rejects winners at the base rate is a fine reason)
# ==================================================================================================================
def reason_win_rates(weeks: Sequence[Week], analyzer: RejectionAnalyzer | None = None, min_n: int = 15, q: float = 0.10,
                     now=None) -> list[dict[str, Any]]:
    """For EVERY rejected candidate (winner or not), group by primary rejection reason and compare the reason's win rate with
    the win rate of all rejected candidates. A reason whose rejects win MORE often than average (BH-corrected) is a costly
    filter; one whose rejects win less is doing its job. Missed-winner counts alone cannot say this: they omit the
    rejected candidates that correctly lost."""
    from engine import pattern_stats as PS
    from engine.direction import wilson
    analyzer = analyzer or RejectionAnalyzer()
    by: dict[str, list[int]] = {}
    for w in weeks:
        if now is not None:
            require_past(w.resolved_at, now, f"week {w.label}")
        n_ranked = len([c for c in w.candidates if c.rank is not None])
        for c in w.candidates:
            if c.picked or c.fwd is None:
                continue
            r = analyzer.explain(c, w, n_ranked)
            by.setdefault(r.primary.value, []).append(int(c.fwd >= w.thr))
    allv = [v for vs in by.values() for v in vs]
    if not allv:
        return []
    p0 = float(np.mean(allv))
    names = [k for k, v in by.items() if len(v) >= min_n]
    if not names or p0 in (0.0, 1.0):
        return []
    z = np.array([(np.mean(by[k]) - p0) / math.sqrt(p0 * (1 - p0) / len(by[k])) for k in names])
    pv = np.array([PS.t_to_p(v) / 2.0 if v > 0 else 1.0 - PS.t_to_p(v) / 2.0 for v in z])
    qv = PS.bh_qvalues(pv)
    out = []
    for i, k in enumerate(names):
        lo, hi = wilson(int(np.sum(by[k])), len(by[k]))
        out.append({"reason": k, "n": len(by[k]), "win_rate": float(np.mean(by[k])), "ci": (lo, hi), "base_rate": p0, "z": float(z[i]),
                    "q": float(qv[i]), "verdict": "COSTLY" if qv[i] <= q and z[i] > 0 else "PAYS" if z[i] < 0 and (1 - qv[i]) >= 1 - q and pv[i] > 1 - q else "NEUTRAL"})
    return sorted(out, key=lambda r: (-r["z"], r["reason"]))


def gate_sweep(weeks: Sequence[Week], attr: str, grid: Sequence[float], now=None) -> list[dict[str, Any]]:
    """For a numeric gate on a candidate attribute (confidence, reliability): among NON-picked candidates, the win rate and
    mean return of those whose attribute falls in each band of `grid`. A win rate that stays flat across the gate says the
    gate does not discriminate; one that jumps just under the gate says it is set too tight."""
    vals, wins, rets = [], [], []
    for w in weeks:
        if now is not None:
            require_past(w.resolved_at, now, f"week {w.label}")
        for c in w.candidates:
            v = getattr(c, attr, None)
            if c.picked or c.fwd is None or v is None:
                continue
            vals.append(float(v))
            wins.append(c.fwd >= w.thr)
            rets.append(c.fwd)
    if not vals:
        return []
    v, wn, rt = np.array(vals), np.array(wins, dtype=float), np.array(rets)
    edges = list(grid)
    out = []
    for lo, hi in zip([-np.inf] + edges, edges + [np.inf]):
        m = (v > lo) & (v <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "win_rate": float(wn[m].mean()) if m.any() else float("nan"),
                    "mean_ret": float(rt[m].mean()) if m.any() else float("nan")})
    return out


def discover_interactions(df: pd.DataFrame, p: MissedParams | None = None, seed: int = 0, top: int = 5,
                          feats: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Feature pairs whose JOINT high region holds more missed winners (versus false positives) than the two single regions
    predict: lift(a and b) / (lift(a) * lift(b) / base), the multiplicative-independence benchmark. Significance is a
    within-period label permutation of the maximum interaction ratio over all pairs, so searching many pairs is priced in.
    Feeds Week.interactions for the MISSING_INTERACTION reason."""
    p = p or MissedParams()
    feats = [f for f in (feats or [c for c in df.columns if c not in META_COLS]) if f in df.columns]
    if df.empty or len(feats) < 2 or (df["group"] == 1).sum() < p.min_a or (df["group"] == 0).sum() < p.min_b:
        return []
    y = df["group"].to_numpy(dtype=float)
    hi = {f: (df[f].to_numpy(dtype=float) >= p.inter_q) for f in feats}
    pairs = [(a, b) for i, a in enumerate(feats) for b in feats[i + 1:]]
    pcodes = pd.factorize(df["period"])[0]

    def ratios(yy):
        base_rate = yy.mean()
        out = {}
        for a, b in pairs:
            ma, mb = hi[a], hi[b]
            mab = ma & mb
            if mab.sum() < max(p.min_cover_abs, 5) or ma.sum() == 0 or mb.sum() == 0:
                out[(a, b)] = 0.0
                continue
            la, lb, lab = yy[ma].mean() / base_rate, yy[mb].mean() / base_rate, yy[mab].mean() / base_rate
            out[(a, b)] = lab / max(1e-9, la * lb / 1.0) if la > 0 and lb > 0 else 0.0
        return out
    obs = ratios(y)
    rng = np.random.default_rng(seed)
    null_max = np.array([max(ratios(PS.permute_within_clusters(y, pcodes, rng)).values()) for _ in range(p.n_perm)])
    ranked = sorted(obs.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    out = []
    for (a, b), r in ranked:
        m = hi[a] & hi[b]
        out.append({"pair": (a, b), "ratio": float(r), "support": int(m.sum()), "joint_lift": float(y[m].mean() / y.mean()) if m.any() else 0.0,
                    "p_fwer": float((1 + (null_max >= r).sum()) / (1 + p.n_perm))})
    return out


def distinction_by_era(d: Distinction, weeks: Sequence[Week]) -> dict[str, dict[str, float]]:
    """Apply one rule inside each era: among non-picked candidates, the rule's win-rate lift over that era's non-picked base
    rate. A rule that only works in one era is regime-bound, and is reported as such rather than averaged away."""
    acc: dict[str, list[list[int]]] = {}
    for w in weeks:
        nonp = [c for c in w.candidates if not c.picked and c.fwd is not None]
        if not nonp:
            continue
        a = acc.setdefault(w.era or "-", [[0, 0, 0, 0]])[0]          # [rule_n, rule_wins, all_n, all_wins]
        for c in nonp:
            win = int(c.fwd >= w.thr)
            a[2] += 1
            a[3] += win
            if d.apply(c.features):
                a[0] += 1
                a[1] += win
    out = {}
    for era, [(rn, rw, an, aw)] in acc.items():
        base_rate = aw / an if an else float("nan")
        prec = rw / rn if rn else float("nan")
        out[era] = {"rule_n": rn, "precision": prec, "base_rate": base_rate, "lift": prec / base_rate if rn and base_rate else float("nan")}
    return out


def learnability(df: pd.DataFrame, p: MissedParams | None = None, seed: int = 0) -> dict[str, Any]:
    """How much of the miss is learnable AT ALL from these features? Best single-feature AUC of missed winners vs false
    positives, compared with the same statistic under within-period shuffled labels. If the real best AUC does not exceed the
    shuffled distribution, the honest conclusion is that the misses look like the false positives: UNKNOWN, not 'more
    features needed'."""
    from engine import pattern_reliability as PR
    p = p or MissedParams()
    feats = [c for c in df.columns if c not in META_COLS]
    if df.empty or len(feats) == 0 or (df["group"] == 1).sum() < p.min_a or (df["group"] == 0).sum() < p.min_b:
        return {"verdict": "INSUFFICIENT_DATA", "n_a": int((df["group"] == 1).sum()) if len(df) else 0}
    y = df["group"].to_numpy(dtype=bool)
    pc = pd.factorize(df["period"])[0]

    def best(yy):
        a = [max(abs(PR.auc(df[f].to_numpy(dtype=float), yy) - 0.5), 0.0) for f in feats]
        return max(a) + 0.5, feats[int(np.argmax(a))]
    obs, feat = best(y)
    rng = np.random.default_rng(seed)
    null = np.array([best(PS.permute_within_clusters(y, pc, rng))[0] for _ in range(max(50, p.n_perm // 2))])
    pv = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return {"best_auc": float(obs), "feature": feat, "null_p95": float(np.quantile(null, 0.95)), "p": pv,
            "verdict": "SEPARABLE" if pv <= p.alpha else "NOT_SEPARABLE_FROM_FEATURES"}


class DistinctionBook:
    """Tracks distinctions discovered fold after fold. A rule that is found again and again (same features, same direction)
    is a stable candidate; one found once is probably a search artefact even if it passed its own fold."""

    def __init__(self):
        self.seen: dict[str, dict[str, Any]] = {}
        self.folds = 0

    def add_fold(self, found: Sequence[Distinction], results: Mapping[str, OOSResult] | None = None) -> None:
        self.folds += 1
        for d in found:
            key = "|".join(sorted(f"{f}{op}" for f, op, _ in d.conds))            # thresholds may drift; features + directions identify
            r = self.seen.setdefault(key, {"describe": d.describe(), "folds": 0, "passes": 0, "lift": [], "p_fwer": []})
            r["folds"] += 1
            r["lift"].append(d.lift)
            r["p_fwer"].append(d.p_fwer)
            if results and results.get(d.did) and results[d.did].verdict == "PASS":
                r["passes"] += 1

    def stable(self, min_share: float = 0.5) -> list[dict[str, Any]]:
        if not self.folds:
            return []
        out = [{"key": k, "describe": v["describe"], "found_in": v["folds"], "share": v["folds"] / self.folds, "passes": v["passes"],
                "mean_lift": float(np.mean(v["lift"]))} for k, v in self.seen.items() if v["folds"] / self.folds >= min_share]
        return sorted(out, key=lambda r: (-r["share"], -r["passes"], r["key"]))


def render_walk_forward(rep: WalkForwardReport, null: Mapping[str, Any] | None = None) -> str:
    lines = [f"walk-forward distinctions: {rep.folds} folds, {rep.discovered} discovered, {rep.passed} passed out of sample "
             f"(pass rate {rep.pass_rate:.2f})   IMPLEMENTED - NOT VALIDATED"]
    for d, r in rep.results:
        lines.append(f"  [{r.verdict:<12}] {d.describe():<48} fold-lift {d.lift:.2f} p_fwer {d.p_fwer:.3f} | oos lift {r.lift:.2f} gain {r.mean_gain:+.4f} p {r.p_value:.3f}")
    for era, (t, ps) in sorted(rep.by_era.items()):
        lines.append(f"  era {era or '-':<10} tested {t:>4}  passed {ps:>4}")
    if null:
        lines.append(f"  null control: {null['passed']}/{null['tested']} shuffled-label distinctions passed "
                     f"(false discovery rate {null['false_discovery_rate']:.3f} vs alpha {null['alpha']})")
    return NL.join(lines)
