"""Disagreement engine (SELF_LEARNING_CONTRACT section 53; loop stage "MEASURE SURPRISE / ASSIGN CREDIT"). IMPLEMENTED - NOT
VALIDATED.

Disagreement between the pattern miner, analog engine, memory, direction model, movement model and risk model is
information. For every observation the engine records a STANCE per source in [-1, +1] (+1 = take it / expects up,
-1 = avoid / expects down, 0 = no view, NaN = the source abstained) and answers, with statistics rather than stories:

  who disagreed     which sources sat on opposite sides, and how strongly (a graded conflict score in [0, 1]);
  why               low coverage, a weakly-held side, a confidence gap, a mechanism conflict, or a context that produces
                    disagreement more often than chance (lift tests, BH-corrected) - else honestly UNEXPLAINED;
  who was right     per source and per opposing pair, with Wilson intervals and exact binomial tests, BH-corrected because
                    a grid of pair x context tests manufactures false winners;
  in what context   the same tables inside each context value;
  is disagreement itself predictive?  conflict vs |move| and vs 'the consensus was wrong', with a week-clustered
                    bootstrap, a within-context (Simpson-safe) version, and an out-of-sample time-split AUC;
  what to do        ARBITRATION RULES ("when A and B disagree in context C, follow A"; "when conflict is high in C,
                    abstain") learned on an earlier window, evaluated on a later one, and compared with a placebo learner
                    fed shuffled outcomes so the false-discovery rate of the rule learner is measured, not assumed.

Firewall: outcomes for observations whose date + horizon is not strictly before `now` are set to NaN (their stances stay
usable - a stance is a decision-time fact); rules record `learned_through` and `learn_rules` refuses a train window that
reaches `now`. Deterministic in `seed`. Reuses cluster_bootstrap / wilson / spearman from engine.learning.redundancy."""
from __future__ import annotations

import dataclasses
import enum
import itertools
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from engine.learning.core import FirewallBreach, Unknown, as_date, current_code_hash, require_past, stable_hash
from engine.learning.redundancy import cluster_bootstrap, spearman
from engine.pattern_reliability import auc as _auc
from engine.pattern_stats import bh_qvalues, cluster_bootstrap_ci, permute_within_clusters, week_codes
from engine.stops import wilson

SOURCES = ("pattern", "analog", "memory", "direction", "movement", "risk")


class Reason(str, enum.Enum):
    LOW_COVERAGE = "LOW_COVERAGE"              # half or more of the sources abstained
    WEAK_STANCE = "WEAK_STANCE"                # the minority side is barely held
    CONFIDENCE_GAP = "CONFIDENCE_GAP"          # one side is confident, the other is not
    MECHANISM_CONFLICT = "MECHANISM_CONFLICT"  # the two sides claim unrelated mechanisms
    CONTEXT_DRIVEN = "CONTEXT_DRIVEN"          # this context produces disagreement more often than chance
    UNEXPLAINED = "UNEXPLAINED"


class Predictiveness(str, enum.Enum):
    PREDICTIVE = "PREDICTIVE"                  # holds out of sample, in both eras and inside contexts
    NOT_PREDICTIVE = "NOT_PREDICTIVE"
    CONTEXT_ARTIFACT = "CONTEXT_ARTIFACT"      # only visible because contexts differ in both disagreement and outcome
    UNSTABLE = "UNSTABLE"                      # sign differs between eras
    INSUFFICIENT = "INSUFFICIENT"


@dataclasses.dataclass(frozen=True)
class DisagreementConfig:
    seed: int = 7
    deadband: float = 0.10             # |stance| at or below this is "no view" for the purposes of sides
    conflict_min: float = 0.40         # conflict score at or above this is a disagreement event
    min_pair_n: int = 25
    min_context_n: int = 30
    n_boot: int = 200
    n_perm: int = 200
    fdr: float = 0.10
    trust_min_rate: float = 0.55
    abstain_max_rate: float = 0.52     # consensus accuracy upper bound under which we abstain
    train_frac: float = 0.6
    conf_gap: float = 0.30
    weak_stance: float = 0.30
    lift_min: float = 1.15
    min_predictive_n: int = 120
    auc_min: float = 0.53
    alpha: float = 0.05

    def validate(self) -> list[str]:
        errs = []
        if not 0 <= self.deadband < 1:
            errs.append("deadband must be in [0, 1)")
        if not 0 < self.conflict_min <= 1:
            errs.append("conflict_min must be in (0, 1]")
        if not 0.3 <= self.train_frac <= 0.9:
            errs.append("train_frac must be in [0.3, 0.9]")
        if not 0 < self.fdr < 0.5 or not 0 < self.alpha < 0.5:
            errs.append("fdr and alpha must be in (0, 0.5)")
        if min(self.min_pair_n, self.min_context_n, self.n_boot, self.n_perm, self.min_predictive_n) < 5:
            errs.append("sample thresholds too small to mean anything")
        return errs


# ---------------------------------------------------------------------------------------------- data
@dataclasses.dataclass
class StanceFrame:
    """Stances of every source on shared observations plus outcomes and context."""
    stances: pd.DataFrame              # N x sources, values in [-1, 1] or NaN
    dates: pd.DatetimeIndex
    y: np.ndarray                      # realised forward return; NaN where not matured / unknown
    context: pd.DataFrame              # N x dims (object labels)
    conf: pd.DataFrame | None = None   # N x sources in [0, 1]
    features: pd.DataFrame | None = None
    horizon_days: int = 5
    n_pending: int = 0

    def validate(self) -> list[str]:
        errs = []
        n = len(self.stances)
        if not (len(self.dates) == len(self.y) == len(self.context) == n):
            errs.append("stances, dates, y and context must have equal length")
        v = self.stances.to_numpy(dtype=float)
        if np.isfinite(v).any() and (np.nanmax(np.abs(v)) > 1.0 + 1e-9):
            errs.append("stances must lie in [-1, 1]")
        if self.conf is not None:
            c = self.conf.to_numpy(dtype=float)
            if c.shape != v.shape:
                errs.append("conf must match stances in shape")
            elif np.isfinite(c).any() and (np.nanmin(c) < 0 or np.nanmax(c) > 1):
                errs.append("conf must lie in [0, 1]")
        if self.stances.columns.duplicated().any():
            errs.append("duplicate source names")
        return errs

    @classmethod
    def build(cls, stances: pd.DataFrame, dates, y, context=None, conf=None, features=None, horizon_days: int = 5,
              now=None, strict: bool = False) -> "StanceFrame":
        dates = pd.DatetimeIndex(dates)
        y = np.asarray(y, dtype=float).copy()
        ctx = pd.DataFrame({"all": ["all"] * len(y)}) if context is None else pd.DataFrame(context).reset_index(drop=True).astype(object)
        pending = 0
        if now is not None:
            for i, d in enumerate(dates):
                try:
                    require_past(d + pd.Timedelta(days=horizon_days), now, "observation outcome")
                except FirewallBreach:
                    if strict:
                        raise
                    y[i] = np.nan
                    pending += 1
        f = cls(stances.reset_index(drop=True), dates, y, ctx, None if conf is None else conf.reset_index(drop=True),
                None if features is None else features.reset_index(drop=True), horizon_days, pending)
        errs = f.validate()
        if errs:
            raise ValueError("; ".join(errs))
        return f

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def sources(self) -> list:
        return list(self.stances.columns)

    def weeks(self) -> np.ndarray:
        return week_codes(self.dates)[0]

    def take(self, ix: np.ndarray) -> "StanceFrame":
        ix = np.asarray(ix)
        return StanceFrame(self.stances.iloc[ix].reset_index(drop=True), self.dates[ix], self.y[ix],
                           self.context.iloc[ix].reset_index(drop=True),
                           None if self.conf is None else self.conf.iloc[ix].reset_index(drop=True),
                           None if self.features is None else self.features.iloc[ix].reset_index(drop=True),
                           self.horizon_days, 0)

    def time_split(self, frac: float) -> tuple["StanceFrame", "StanceFrame"]:
        order = np.argsort(self.dates.values, kind="stable")
        k = int(len(order) * frac)
        return self.take(np.sort(order[:k])), self.take(np.sort(order[k:]))


# ---------------------------------------------------------------------------------------------- core measures
def sides(S: np.ndarray, deadband: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean matrices (positive side, negative side); NaN and dead-band stances are on neither side."""
    with np.errstate(invalid="ignore"):
        return S > deadband, S < -deadband


def conflict_score(S: np.ndarray) -> np.ndarray:
    """1 - |sum s| / sum |s| over active sources: 0 when every view points the same way (or there is <2 views),
    1 when opposing views exactly cancel. Graded, so a 5-vs-1 split scores lower than 3-vs-3."""
    a = np.nan_to_num(S)
    tot = np.abs(a).sum(axis=1)
    net = np.abs(a.sum(axis=1))
    n_active = np.isfinite(S).sum(axis=1)
    out = np.where(tot > 1e-12, 1.0 - net / np.maximum(tot, 1e-12), 0.0)
    return np.where(n_active >= 2, out, 0.0)


def consensus(S: np.ndarray) -> np.ndarray:
    """Sign of the summed stance (0 when they cancel or nobody spoke)."""
    return np.sign(np.nansum(S, axis=1))


def disagreement_features(sf: StanceFrame, cfg: DisagreementConfig) -> pd.DataFrame:
    """The engine's own view of an observation, usable as model inputs by downstream code (all decision-time facts)."""
    S = sf.stances.to_numpy(dtype=float)
    pos, neg = sides(S, cfg.deadband)
    act = np.isfinite(S)
    p, q = pos.sum(axis=1), neg.sum(axis=1)
    share = np.where(p + q > 0, p / np.maximum(p + q, 1), 0.5)
    return pd.DataFrame({"n_active": act.sum(axis=1), "n_pos": p, "n_neg": q, "n_abstain": (~act).sum(axis=1),
                         "conflict": conflict_score(S), "polarization": 4 * share * (1 - share) * ((p > 0) & (q > 0)),
                         "spread": np.nanstd(np.where(act, S, np.nan), axis=1) if act.any() else np.zeros(len(S)),
                         "consensus": consensus(S), "mean_stance": np.nanmean(np.where(act, S, np.nan), axis=1)
                         if act.any() else np.zeros(len(S))}, index=sf.stances.index).fillna(0.0)


@dataclasses.dataclass(frozen=True)
class DisagreementEvent:
    """One observation on which sources took opposite sides."""
    index: int
    date: str
    conflict: float
    positive: tuple
    negative: tuple
    abstained: tuple
    reason: Reason
    detail: str
    context: Mapping[str, str]
    y: float | None
    consensus_right: bool | None
    winners: tuple                     # sources whose stance had the sign of y (empty until the outcome is known)

    def validate(self) -> list[str]:
        errs = []
        if not (self.positive and self.negative):
            errs.append(f"event {self.index}: a disagreement needs sources on both sides")
        if set(self.positive) & set(self.negative):
            errs.append(f"event {self.index}: a source cannot be on both sides")
        if not 0.0 <= self.conflict <= 1.0:
            errs.append(f"event {self.index}: conflict outside [0,1]")
        return errs


def find_events(sf: StanceFrame, cfg: DisagreementConfig, mechanisms: Mapping[str, Sequence[str]] | None = None,
                context_lift: Mapping[tuple, float] | None = None) -> list[DisagreementEvent]:
    """Every observation with sources on both sides and conflict >= conflict_min, each with a reason (first match wins):
    LOW_COVERAGE, WEAK_STANCE, CONFIDENCE_GAP, MECHANISM_CONFLICT, CONTEXT_DRIVEN, UNEXPLAINED."""
    S = sf.stances.to_numpy(dtype=float)
    pos, neg = sides(S, cfg.deadband)
    conf = conflict_score(S)
    cols = sf.sources
    C = None if sf.conf is None else sf.conf.to_numpy(dtype=float)
    cons = consensus(S)
    mech = mechanisms or {}
    events = []
    for i in np.flatnonzero(pos.any(axis=1) & neg.any(axis=1) & (conf >= cfg.conflict_min)):
        P = tuple(c for c, f in zip(cols, pos[i]) if f)
        N = tuple(c for c, f in zip(cols, neg[i]) if f)
        A = tuple(c for c, f in zip(cols, np.isnan(S[i])) if f)
        reason, detail = Reason.UNEXPLAINED, ""
        ctx = {d: str(v) for d, v in sf.context.iloc[i].items()}
        weak_side = min(np.abs(S[i][pos[i]]).max(), np.abs(S[i][neg[i]]).max())
        if len(A) * 2 >= len(cols):
            reason, detail = Reason.LOW_COVERAGE, f"{len(A)} of {len(cols)} sources abstained"
        elif weak_side < cfg.weak_stance:
            reason, detail = Reason.WEAK_STANCE, f"minority side peaks at {weak_side:.2f}"
        elif C is not None and abs(np.nanmean(C[i][pos[i]]) - np.nanmean(C[i][neg[i]])) >= cfg.conf_gap:
            reason, detail = Reason.CONFIDENCE_GAP, "confidence differs between the sides"
        elif mech and all(mech.get(s) for s in P + N) and not (set().union(*[set(mech[s]) for s in P]) &
                                                               set().union(*[set(mech[s]) for s in N])):
            reason, detail = Reason.MECHANISM_CONFLICT, "sides claim disjoint mechanisms"
        elif context_lift:
            hit = [(d, v) for d, v in ctx.items() if context_lift.get((d, v), 0.0) >= cfg.lift_min]
            if hit:
                reason, detail = Reason.CONTEXT_DRIVEN, f"{hit[0][0]}={hit[0][1]} disagrees {context_lift[hit[0]]:.2f}x more often"
        y = None if not np.isfinite(sf.y[i]) else float(sf.y[i])
        win = tuple(c for c, s in zip(cols, S[i]) if y is not None and np.isfinite(s) and s * y > 0)
        ev = DisagreementEvent(int(i), str(sf.dates[i].date()), float(conf[i]), P, N, A, reason, detail, ctx, y,
                               None if y is None or cons[i] == 0 else bool(cons[i] * y > 0), win)
        events.append(ev)
    return events


# ---------------------------------------------------------------------------------------------- who was right
def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """BH q-values via engine.pattern_stats.bh_qvalues; NaN p-values stay NaN and do not count toward the number of tests."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = bh_qvalues(p[ok])
    return q


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC (engine.pattern_reliability.auc); 0.5 - 'no information' - when a class is empty."""
    v = _auc(score, np.asarray(label, dtype=bool))
    return 0.5 if not np.isfinite(v) else float(v)


@dataclasses.dataclass(frozen=True)
class PairResult:
    a: str
    b: str
    context: str                       # 'all' or 'dim=value'
    n: int
    a_right: int
    rate: float                        # P(a right | a and b opposed)
    lo: float
    hi: float
    p: float
    q: float = float("nan")

    @property
    def winner(self) -> str | None:
        if not np.isfinite(self.q) or self.q > 0.10:
            return None
        return self.a if self.rate > 0.5 else self.b if self.rate < 0.5 else None


def pair_table(sf: StanceFrame, cfg: DisagreementConfig, by_context: bool = True) -> list[PairResult]:
    """For every unordered pair of sources, when they take opposite sides who is right, overall and per context value.
    Outcomes of exactly zero are dropped. q-values are BH over the whole grid."""
    S = sf.stances.to_numpy(dtype=float)
    cols = sf.sources
    y = sf.y
    out = []
    groups = [("all", np.ones(sf.n, dtype=bool))]
    if by_context:
        for d in sf.context.columns:
            lab = sf.context[d].to_numpy(dtype=object)
            for v in sorted(set(lab)):
                if d == "all" and v == "all":
                    continue
                groups.append((f"{d}={v}", lab == v))
    for (i, a), (j, b) in itertools.combinations(list(enumerate(cols)), 2):
        opp = (S[:, i] * S[:, j] < 0) & (np.abs(S[:, i]) > cfg.deadband) & (np.abs(S[:, j]) > cfg.deadband) & np.isfinite(y) & (y != 0)
        for name, g in groups:
            m = opp & g
            n = int(m.sum())
            need = cfg.min_pair_n if name == "all" else cfg.min_context_n // 2
            if n < need:
                continue
            k = int((S[m, i] * y[m] > 0).sum())
            lo, hi = wilson(k, n)
            out.append(PairResult(a, b, name, n, k, k / n, lo, hi, float(stats.binomtest(k, n, 0.5).pvalue)))
    q = benjamini_hochberg([r.p for r in out])
    return [dataclasses.replace(r, q=float(qq)) for r, qq in zip(out, q)]


@dataclasses.dataclass(frozen=True)
class SourceRecord:
    """How one source behaves when the sources disagree."""
    source: str
    n_spoke: int
    acc_overall: float
    acc_in_disagreement: float
    n_in_disagreement: int
    acc_as_minority: float
    n_minority: int
    acc_as_majority: float
    n_majority: int
    solo_dissent_wins: float           # when it alone opposed the rest, how often it was right
    n_solo: int


def source_records(sf: StanceFrame, cfg: DisagreementConfig) -> list[SourceRecord]:
    S = sf.stances.to_numpy(dtype=float)
    y = sf.y
    pos, neg = sides(S, cfg.deadband)
    conf = conflict_score(S)
    dis = pos.any(axis=1) & neg.any(axis=1) & (conf >= cfg.conflict_min)
    known = np.isfinite(y) & (y != 0)
    npos, nneg = pos.sum(axis=1), neg.sum(axis=1)
    out = []

    def acc(mask, j):
        m = mask & known
        return (float((S[m, j] * y[m] > 0).mean()) if m.any() else float("nan")), int(m.sum())
    for j, c in enumerate(sf.sources):
        spoke = (pos | neg)[:, j]
        mine_pos = pos[:, j]
        minority = dis & spoke & np.where(mine_pos, npos < nneg, nneg < npos)
        majority = dis & spoke & np.where(mine_pos, npos > nneg, nneg > npos)
        solo = dis & spoke & np.where(mine_pos, (npos == 1) & (nneg >= 2), (nneg == 1) & (npos >= 2))
        a_all, n_all = acc(spoke, j)
        a_dis, n_dis = acc(dis & spoke, j)
        a_min, n_min = acc(minority, j)
        a_maj, n_maj = acc(majority, j)
        a_solo, n_solo = acc(solo, j)
        out.append(SourceRecord(c, n_all, a_all, a_dis, n_dis, a_min, n_min, a_maj, n_maj, a_solo, n_solo))
    return out


def context_disagreement_rates(sf: StanceFrame, cfg: DisagreementConfig) -> dict:
    """(dim, value) -> lift of the disagreement rate over the overall rate, only where the lift is significant after BH
    (exact binomial against the overall rate). This is what upgrades an event's reason to CONTEXT_DRIVEN."""
    S = sf.stances.to_numpy(dtype=float)
    pos, neg = sides(S, cfg.deadband)
    dis = pos.any(axis=1) & neg.any(axis=1) & (conflict_score(S) >= cfg.conflict_min)
    base = dis.mean() if sf.n else 0.0
    keys, ps, lifts = [], [], []
    for d in sf.context.columns:
        lab = sf.context[d].to_numpy(dtype=object)
        for v in sorted(set(lab)):
            m = lab == v
            if m.sum() < cfg.min_context_n or base <= 0 or base >= 1:
                continue
            keys.append((d, str(v)))
            ps.append(float(stats.binomtest(int(dis[m].sum()), int(m.sum()), base).pvalue))
            lifts.append(float(dis[m].mean() / base))
    q = benjamini_hochberg(ps)
    return {k: l for k, l, qq in zip(keys, lifts, q) if qq <= cfg.fdr}


# ---------------------------------------------------------------------------------------------- what drives disagreement
@dataclasses.dataclass(frozen=True)
class DriverResult:
    features: tuple
    coefs: Mapping[str, float]         # standardised ridge coefficients on the training half
    r2_oos: float                      # R^2 on the later half (can be negative)
    verdict: str


def disagreement_drivers(sf: StanceFrame, cfg: DisagreementConfig, ridge: float = 5.0) -> DriverResult | None:
    """Ridge of the conflict score on observable features, fit on the early part, scored out of sample on the later part.
    Reported only as 'drivers found' if OOS R^2 > 0.01 - otherwise the honest answer is 'no observable driver'."""
    if sf.features is None or sf.features.shape[1] == 0 or sf.n < 60:
        return None
    X = sf.features.apply(pd.to_numeric, errors="coerce")
    t = conflict_score(sf.stances.to_numpy(dtype=float))
    ok = X.notna().all(axis=1).to_numpy()
    order = np.argsort(sf.dates.values, kind="stable")
    order = order[ok[order]]
    if len(order) < 60:
        return None
    k = int(len(order) * cfg.train_frac)
    tr, te = order[:k], order[k:]
    mu, sd = X.iloc[tr].mean(), X.iloc[tr].std().replace(0, 1)
    Z = ((X - mu) / sd).to_numpy()
    yt = t[tr] - t[tr].mean()
    beta = np.linalg.solve(Z[tr].T @ Z[tr] + ridge * np.eye(Z.shape[1]), Z[tr].T @ yt)
    pred = Z[te] @ beta + t[tr].mean()
    ss_res, ss_tot = ((t[te] - pred) ** 2).sum(), ((t[te] - t[te].mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    verdict = "drivers found" if np.isfinite(r2) and r2 > 0.01 else "no observable driver (out-of-sample R^2 <= 0.01)"
    return DriverResult(tuple(X.columns), {c: float(b) for c, b in zip(X.columns, beta)}, r2, verdict)


# ---------------------------------------------------------------------------------------------- is disagreement predictive?
@dataclasses.dataclass(frozen=True)
class PredictiveTest:
    target: str
    n: int
    rho: float                         # Spearman(conflict, target)
    lo: float
    hi: float
    rho_within_context: float          # after removing context means from both variables
    rho_early: float
    rho_late: float
    auc_oos: float                     # later half: does conflict rank the positive class above the negative class?
    perm_p: float
    verdict: Predictiveness
    reasons: tuple = ()


def _demean(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return x - pd.Series(x).groupby(labels).transform("mean").to_numpy()


def predictive_test(sf: StanceFrame, cfg: DisagreementConfig, target: str, rng: np.random.Generator) -> PredictiveTest:
    """target 'abs_move': conflict vs |y|.  target 'consensus_wrong': conflict vs 1[consensus disagrees with y]."""
    S = sf.stances.to_numpy(dtype=float)
    c = conflict_score(S)
    y = sf.y
    cons = consensus(S)
    if target == "abs_move":
        ok = np.isfinite(y)
        t = np.abs(y)
    elif target == "consensus_wrong":
        ok = np.isfinite(y) & (y != 0) & (cons != 0)
        t = (cons * y < 0).astype(float)
    else:
        raise ValueError(f"unknown target {target!r}")
    idx = np.flatnonzero(ok)
    if len(idx) < cfg.min_predictive_n or np.ptp(c[idx]) < 1e-9 or np.ptp(t[idx]) < 1e-12:
        return PredictiveTest(target, len(idx), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                              float("nan"), 0.5, float("nan"), Predictiveness.INSUFFICIENT,
                              (f"{len(idx)} usable observations or no variation in conflict/target",))
    cc, tt, w = c[idx], t[idx], sf.weeks()[idx]
    lab = sf.context.iloc[idx].astype(str).agg("|".join, axis=1).to_numpy(dtype=object)
    rho = spearman(cc, tt)
    boots = cluster_bootstrap(lambda ix: spearman(cc[ix], tt[ix]), len(idx), w, rng, cfg.n_boot)
    lo, hi = np.quantile(boots, [cfg.alpha / 2, 1 - cfg.alpha / 2]) if len(boots) > 10 else (float("nan"),) * 2
    rw = spearman(_demean(cc, lab), _demean(tt, lab)) if len(set(lab)) > 1 else rho
    order = np.argsort(sf.dates.values[idx], kind="stable")
    k = int(len(order) * cfg.train_frac)
    early, late = order[:k], order[k:]
    r_e, r_l = spearman(cc[early], tt[early]), spearman(cc[late], tt[late])
    pos_class = tt[late] > (np.median(tt[early]) if target == "abs_move" else 0.5)
    # out-of-sample AUC on conflict with the EARLY window's per-context mean removed: a regime that is both conflicted and
    # unpredictable must not masquerade as 'conflict predicts'
    shift = pd.Series(cc[early]).groupby(lab[early]).mean()
    adj = cc[late] - pd.Series(lab[late]).map(shift).fillna(cc[early].mean()).to_numpy(dtype=float)
    a = auc(adj, pos_class) if len(late) >= 30 else 0.5
    # permutation null: shuffle the target inside each week (keeps the calendar structure, breaks the link)
    perm_stats = []
    for _ in range(cfg.n_perm):
        pi = permute_within_clusters(np.arange(len(idx)), w, rng)
        perm_stats.append(spearman(cc, tt[pi]))
    perm_stats = np.array([v for v in perm_stats if np.isfinite(v)])
    pp = float((1 + np.sum(np.abs(perm_stats) >= abs(rho))) / (1 + len(perm_stats))) if len(perm_stats) else float("nan")
    reasons, verdict = [], Predictiveness.NOT_PREDICTIVE
    real = np.isfinite(lo) and (lo > 0 or hi < 0) and pp <= cfg.alpha
    if real:
        if np.isfinite(r_e) and np.isfinite(r_l) and np.sign(r_e) != np.sign(r_l):
            verdict = Predictiveness.UNSTABLE
            reasons.append(f"sign differs between eras ({r_e:+.3f} then {r_l:+.3f})")
        elif not np.isfinite(rw) or abs(rw) < 0.5 * abs(rho) or np.sign(rw) != np.sign(rho):
            verdict = Predictiveness.CONTEXT_ARTIFACT
            reasons.append(f"association shrinks from {rho:+.3f} to {rw:+.3f} once context is held fixed")
        elif abs(a - 0.5) >= cfg.auc_min - 0.5 and (a > 0.5) == (rho > 0):
            verdict = Predictiveness.PREDICTIVE
        else:
            reasons.append(f"in-sample link but out-of-sample AUC only {a:.3f}")
    else:
        reasons.append("no association distinguishable from a within-week shuffle")
    return PredictiveTest(target, len(idx), rho, float(lo), float(hi), float(rw), r_e, r_l, a, pp, verdict, tuple(reasons))


def consensus_accuracy_by_conflict(sf: StanceFrame, bins: Sequence[float] = (0.0, 1e-9, 0.4, 0.7, 1.01)) -> pd.DataFrame:
    """Accuracy of the majority vote in conflict buckets: does the consensus get worse as the sources disagree more?"""
    S = sf.stances.to_numpy(dtype=float)
    c, cons, y = conflict_score(S), consensus(S), sf.y
    ok = np.isfinite(y) & (y != 0) & (cons != 0)
    df = pd.DataFrame({"conflict": c[ok], "right": (cons[ok] * y[ok] > 0).astype(float)})
    if df.empty:
        return pd.DataFrame(columns=["bucket", "n", "consensus_accuracy", "lo", "hi"])
    df["bucket"] = pd.cut(df["conflict"], bins=list(bins), right=False, include_lowest=True)
    g = df.groupby("bucket", observed=True)["right"].agg(["count", "mean"]).reset_index()
    g[["lo", "hi"]] = [wilson(int(round(m * n)), int(n)) for n, m in zip(g["count"], g["mean"])]
    return g.rename(columns={"count": "n", "mean": "consensus_accuracy"})


# ---------------------------------------------------------------------------------------------- learned arbitration rules
@dataclasses.dataclass(frozen=True)
class ArbitrationRule:
    rule_id: str
    kind: str                          # TRUST or ABSTAIN
    trust: str | None
    against: str | None
    context: str                       # 'all' or 'dim=value'
    n: int
    rate: float                        # P(trusted side right) for TRUST; consensus accuracy for ABSTAIN
    lo: float
    hi: float
    q: float
    learned_through: str               # newest observation date the rule saw
    train_n: int

    def validate(self) -> list[str]:
        errs = []
        if self.kind not in ("TRUST", "ABSTAIN"):
            errs.append(f"unknown rule kind {self.kind}")
        if self.kind == "TRUST" and (not self.trust or not self.against or self.trust == self.against):
            errs.append("TRUST rule needs two different sources")
        if not (0 <= self.rate <= 1) or self.lo > self.hi:
            errs.append("rate/interval invalid")
        return errs

    def applies(self, sf: StanceFrame, cfg: DisagreementConfig) -> np.ndarray:
        S = sf.stances.to_numpy(dtype=float)
        cols = sf.sources
        if self.context == "all":
            ctx = np.ones(sf.n, dtype=bool)
        else:
            d, v = self.context.split("=", 1)
            ctx = (sf.context[d].astype(str).to_numpy() == v) if d in sf.context else np.zeros(sf.n, dtype=bool)
        if self.kind == "ABSTAIN":
            return ctx & (conflict_score(S) >= cfg.conflict_min)
        i, j = cols.index(self.trust), cols.index(self.against)
        return ctx & (S[:, i] * S[:, j] < 0) & (np.abs(S[:, i]) > cfg.deadband) & (np.abs(S[:, j]) > cfg.deadband)


def learn_rules(sf: StanceFrame, cfg: DisagreementConfig, now) -> list[ArbitrationRule]:
    """Rules from `sf` (a training window). Refuses a window whose outcomes reach `now`."""
    known = np.isfinite(sf.y)
    if known.any():
        newest = sf.dates[known].max() + pd.Timedelta(days=sf.horizon_days)
        require_past(newest, now, "newest training outcome")
    learned = str(sf.dates.max().date()) if sf.n else ""
    rules = []
    S0 = sf.stances.to_numpy(dtype=float)
    cons0 = consensus(S0)
    cols = sf.sources
    groups = [("all", np.ones(sf.n, dtype=bool))]
    for d in sf.context.columns:
        if d != "all":
            lab = sf.context[d].astype(str).to_numpy()
            groups += [(f"{d}={v}", lab == v) for v in sorted(set(lab))]
    cand = []                          # (trust, against, context, n, wins, p)
    for (i, a), (j, b) in itertools.combinations(list(enumerate(cols)), 2):
        opp = (S0[:, i] * S0[:, j] < 0) & (np.abs(S0[:, i]) > cfg.deadband) & (np.abs(S0[:, j]) > cfg.deadband)             & known & (sf.y != 0)
        for name, g in groups:
            for t_i, t_name, o_name in ((i, a, b), (j, b, a)):
                # the rule only changes the outcome where the trusted source differs from the majority vote, so that is
                # exactly the set on which it must win more than half of the time
                m = opp & g & (np.sign(S0[:, t_i]) != cons0)
                n = int(m.sum())
                if n < (cfg.min_pair_n if name == "all" else cfg.min_context_n // 2):
                    continue
                k = int((S0[m, t_i] * sf.y[m] > 0).sum())
                cand.append((t_name, o_name, name, n, k, float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)))
    q = benjamini_hochberg([c[5] for c in cand])
    for (t_name, o_name, name, n, k, _), qq in zip(cand, q):
        if qq > cfg.fdr or k / n < cfg.trust_min_rate:
            continue
        lo, hi = wilson(k, n)
        rid = stable_hash(["TRUST", t_name, o_name, name, learned])
        rules.append(ArbitrationRule(rid, "TRUST", t_name, o_name, name, n, k / n, lo, hi, float(qq), learned, sf.n))
    S = sf.stances.to_numpy(dtype=float)
    c, cons = conflict_score(S), consensus(S)
    for name, g in [("all", np.ones(sf.n, dtype=bool))] + [(f"{d}={v}", sf.context[d].astype(str).to_numpy() == v)
                                                            for d in sf.context.columns for v in sorted(set(sf.context[d].astype(str)))
                                                            if not (d == "all")]:
        m = g & (c >= cfg.conflict_min) & known & (cons != 0) & (sf.y != 0)
        n = int(m.sum())
        if n < cfg.min_context_n:
            continue
        k = int((cons[m] * sf.y[m] > 0).sum())
        lo, hi = wilson(k, n)
        if hi <= cfg.abstain_max_rate:
            rid = stable_hash(["ABSTAIN", name, learned])
            rules.append(ArbitrationRule(rid, "ABSTAIN", None, None, name, n, k / n, lo, hi,
                                         float(stats.binomtest(k, n, 0.5, alternative="less").pvalue), learned, sf.n))
    for r in rules:
        errs = r.validate()
        if errs:
            raise ValueError("; ".join(errs))
    return sorted(rules, key=lambda r: (r.kind, r.q, r.rule_id))


def arbitrate(sf: StanceFrame, rules: Sequence[ArbitrationRule], cfg: DisagreementConfig) -> tuple[np.ndarray, np.ndarray]:
    """(final stance sign per observation, rule index applied or -1). Default = sign of the summed stances. The first
    matching rule in order wins: TRUST rules before ABSTAIN (so a trusted source can still act inside a noisy context)."""
    S = sf.stances.to_numpy(dtype=float)
    cols = sf.sources
    final = consensus(S).astype(float)
    used = np.full(sf.n, -1, dtype=int)
    for k, r in enumerate(sorted(rules, key=lambda r: (r.kind != "TRUST", r.q))):
        m = r.applies(sf, cfg) & (used < 0)
        if r.kind == "TRUST":
            final[m] = np.sign(S[m, cols.index(r.trust)])
        else:
            final[m] = 0.0
        used[m] = k
    return final, used


@dataclasses.dataclass(frozen=True)
class RuleEvaluation:
    n_rules: int
    n_applied: int
    acc_default: float                 # majority vote on the rows where a rule applied
    acc_arbitrated: float              # arbitrated stance on the same rows, abstentions counted as 0.5
    gain: float
    lo: float
    hi: float
    placebo_rules: int                 # rules the same learner finds when outcomes are shuffled (false-discovery probe)
    verdict: str


def evaluate_rules(train: StanceFrame, test: StanceFrame, cfg: DisagreementConfig, now, rng: np.random.Generator) -> tuple[list[ArbitrationRule], RuleEvaluation]:
    rules = learn_rules(train, cfg, now)
    # placebo learner: outcomes shuffled inside each week, then the identical learner
    pi = permute_within_clusters(np.arange(train.n), train.weeks(), rng)
    placebo = dataclasses.replace(train, y=train.y[pi])
    placebo_n = sum(r.kind == "TRUST" for r in learn_rules(placebo, cfg, now))   # ABSTAIN on chance outcomes is correct, not a false find
    if not rules:
        return rules, RuleEvaluation(0, 0, float("nan"), float("nan"), 0.0, float("nan"), float("nan"), placebo_n,
                                     "no rules survived FDR control on the training window")
    final, used = arbitrate(test, rules, cfg)
    S = test.stances.to_numpy(dtype=float)
    cons = consensus(S)
    m = (used >= 0) & np.isfinite(test.y) & (test.y != 0)
    if m.sum() < cfg.min_pair_n:
        return rules, RuleEvaluation(len(rules), int(m.sum()), float("nan"), float("nan"), 0.0, float("nan"), float("nan"),
                                     placebo_n, "rules rarely fire in the test window: not enough to judge")
    right_def = np.where(cons[m] == 0, 0.5, (cons[m] * test.y[m] > 0).astype(float))
    right_new = np.where(final[m] == 0, 0.5, (final[m] * test.y[m] > 0).astype(float))
    diff = right_new - right_def
    w = test.weeks()[m]
    _, codes = np.unique(w, return_inverse=True)
    lo, hi = cluster_bootstrap_ci(codes.astype(np.int64), np.ones(len(diff)), diff, np.ones(len(diff), dtype=bool),
                                  int(codes.max()) + 1, rng, reps=cfg.n_boot, level=1 - cfg.alpha)
    verdict = ("rules improve on majority vote out of sample" if lo > 0 else
               "rules are worse than majority vote out of sample" if hi < 0 else
               "no out-of-sample difference from majority vote")
    if placebo_n >= max(1, len(rules)):
        verdict += "; WARNING: the learner finds as many rules on shuffled outcomes"
    return rules, RuleEvaluation(len(rules), int(m.sum()), float(right_def.mean()), float(right_new.mean()),
                                 float(diff.mean()), float(lo), float(hi), placebo_n, verdict)


# ---------------------------------------------------------------------------------------------- report
@dataclasses.dataclass(frozen=True)
class DisagreementReport:
    config: DisagreementConfig
    code_hash: str
    now: str
    n_obs: int
    n_pending: int
    disagreement_rate: float
    events: tuple
    reasons: Mapping[str, int]
    pairs: tuple
    sources: tuple
    context_lift: Mapping[str, float]
    drivers: DriverResult | None
    predictive: tuple
    consensus_by_conflict: tuple
    rules: tuple
    rule_eval: RuleEvaluation | None
    frame_hash: str

    def predictive_for(self, target: str) -> PredictiveTest:
        for t in self.predictive:
            if t.target == target:
                return t
        raise KeyError(target)

    def label(self) -> str:
        return "IMPLEMENTED — NOT VALIDATED"

    def render_text(self) -> str:
        L = [f"Disagreement @ {self.now}: {self.n_obs} observations ({self.n_pending} outcomes pending), "
             f"{self.disagreement_rate:.1%} in disagreement  [{self.label()}]",
             "  reasons: " + (", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items())) or "none")]
        for s in self.sources:
            L.append(f"  {s.source:<10} acc {s.acc_overall:.3f}  in disagreement {s.acc_in_disagreement:.3f} (n={s.n_in_disagreement})"
                     f"  minority {s.acc_as_minority:.3f} (n={s.n_minority})  solo dissent wins {s.solo_dissent_wins:.3f} (n={s.n_solo})")
        for p in self.pairs:
            if p.winner and p.context == "all":
                L.append(f"  {p.winner} beats the other in {p.a} vs {p.b}: {max(p.rate, 1 - p.rate):.3f} (n={p.n}, q={p.q:.3f})")
        for t in self.predictive:
            L.append(f"  conflict vs {t.target}: rho {t.rho:+.3f} [{t.lo:+.3f},{t.hi:+.3f}] within-context {t.rho_within_context:+.3f} "
                     f"auc_oos {t.auc_oos:.3f} -> {t.verdict.value}")
            for r in t.reasons:
                L.append(f"      - {r}")
        if self.drivers:
            L.append(f"  drivers: {self.drivers.verdict} (R2 oos {self.drivers.r2_oos:+.3f})")
        if self.rule_eval:
            L.append(f"  rules: {self.rule_eval.n_rules} learned, applied to {self.rule_eval.n_applied} test rows; "
                     f"gain {self.rule_eval.gain:+.3f} [{self.rule_eval.lo:+.3f},{self.rule_eval.hi:+.3f}]; "
                     f"placebo finds {self.rule_eval.placebo_rules}; {self.rule_eval.verdict}")
        return "\n".join(L)


class DisagreementEngine:
    def __init__(self, cfg: DisagreementConfig | None = None, mechanisms: Mapping[str, Sequence[str]] | None = None):
        self.cfg = cfg or DisagreementConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.mechanisms = dict(mechanisms or {})

    def assess(self, sf: StanceFrame, now) -> DisagreementReport:
        cfg = self.cfg
        errs = sf.validate()
        if errs:
            raise ValueError("; ".join(errs))
        known = np.isfinite(sf.y)
        if known.any():
            require_past(sf.dates[known].max() + pd.Timedelta(days=sf.horizon_days), now, "newest outcome used")
        if sf.n == 0 or sf.stances.shape[1] < 2:
            return DisagreementReport(cfg, current_code_hash(), str(as_date(now)), sf.n, sf.n_pending, 0.0, (), {}, (), (), {},
                                      None, (), (), (), None, stable_hash([]))
        seeds = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(5)]
        lift = context_disagreement_rates(sf, cfg)
        events = find_events(sf, cfg, self.mechanisms, lift)
        for e in events:
            errs = e.validate()
            if errs:
                raise ValueError("; ".join(errs))
        reasons: dict = {}
        for e in events:
            reasons[e.reason.value] = reasons.get(e.reason.value, 0) + 1
        preds = tuple(predictive_test(sf, cfg, t, seeds[0]) for t in ("abs_move", "consensus_wrong"))
        rules, ev = (), None
        if known.sum() >= 4 * cfg.min_predictive_n:
            train, test = sf.time_split(cfg.train_frac)
            cut = test.dates.min() if test.n else None
            if cut is not None:
                # the training window may only contain outcomes that had matured by the time the test window began
                mature_train = train.take(np.flatnonzero((train.dates + pd.Timedelta(days=train.horizon_days)) < cut))
                r, ev = evaluate_rules(mature_train, test, cfg, now, seeds[1])
                rules = tuple(r)
        return DisagreementReport(
            cfg, current_code_hash(), str(as_date(now)), sf.n, sf.n_pending, len(events) / sf.n, tuple(events), reasons,
            tuple(pair_table(sf, cfg)), tuple(source_records(sf, cfg)),
            {f"{d}={v}": l for (d, v), l in lift.items()}, disagreement_drivers(sf, cfg), preds,
            tuple(consensus_accuracy_by_conflict(sf).to_dict("records")) if known.any() else (), rules, ev,
            stable_hash({"n": sf.n, "sources": sf.sources, "seed": cfg.seed}))


# ---------------------------------------------------------------------------------------------- planted synthetic truth
def planted_stances(seed: int = 0, n: int = 3000, horizon_days: int = 5, disagreement_predicts_move: bool = True,
                    shuffle_outcomes: bool = False, herd_trap: bool = True) -> tuple[StanceFrame, dict]:
    """Truth environment with known structure. Latent direction u in {-1, +1} drives y; days are 'calm' or 'crisis' (30%).
      pattern    right 85% in every regime
      analog     right 85% in calm, but only 30% in crisis (it follows the herd)
      direction  right 75% in calm, 30% in crisis;  risk right 70% in calm, 35% in crisis
      memory     pure noise, abstains 70% of the time
      movement   a magnitude view: stance 0 (never opposes anyone), abstains 40% of the time
    So in crisis the majority vote is a trap and the learnable arbitration rule is 'trust pattern in crisis' (herd_trap=False
    removes the trap: every source keeps its calm accuracy in crisis).
    'Uncertain' days (25%) pull every source toward a coin flip and (if disagreement_predicts_move) multiply the size of
    the move by 4, so conflict predicts |y|; otherwise the size is independent of them. `shuffle_outcomes` destroys
    the link between stances and y entirely (the negative control). Returns (frame, truth)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=max(10, n // 6))
    d = np.sort(rng.integers(0, len(days), n))
    regime = np.where((rng.random(len(days)) < 0.3)[d], "crisis", "calm")
    crisis = regime == "crisis"
    uncertain = (rng.random(len(days)) < 0.25)[d]
    u = rng.choice([-1.0, 1.0], n)
    scale = 1.0 + (3.0 * uncertain if disagreement_predicts_move else 3.0 * (rng.random(n) < 0.25))
    y = (0.02 * u + 0.008 * rng.standard_normal(n)) * scale

    def stance(acc_calm, acc_crisis, mag=0.7):
        acc = np.where(crisis & herd_trap, acc_crisis, acc_calm)
        right = rng.random(n) < np.where(uncertain, 0.5 + (acc - 0.5) * 0.2, acc)
        return np.where(right, u, -u) * (mag + 0.3 * rng.random(n))
    df = pd.DataFrame({
        "pattern": stance(0.85, 0.85), "analog": stance(0.85, 0.30), "memory": np.where(
            rng.random(n) < 0.7, np.nan, rng.choice([-1.0, 1.0], n) * (0.3 + 0.7 * rng.random(n))),
        "direction": stance(0.75, 0.30), "movement": np.where(rng.random(n) < 0.4, np.nan, 0.0),
        "risk": stance(0.70, 0.35, 0.5)})
    y_used = np.random.default_rng(seed + 99).permutation(y) if shuffle_outcomes else y
    feats = pd.DataFrame({"vol_proxy": 1.0 + uncertain + 0.3 * rng.standard_normal(n), "noise_feat": rng.standard_normal(n)})
    sf = StanceFrame.build(df, days[d], y_used, pd.DataFrame({"regime": regime}), features=feats, horizon_days=horizon_days)
    truth = {"crisis_winner": "pattern", "crisis_losers": ("analog", "direction", "risk"), "noise_source": "memory",
             "disagreement_predicts_move": disagreement_predicts_move}
    return sf, truth


# ---------------------------------------------------------------------------------------------- what is resolving disagreement worth?
def resolution_value(sf: StanceFrame, cfg: DisagreementConfig) -> dict:
    """On disagreement events with known outcomes: accuracy of (a) the majority vote, (b) the best single source chosen with
    hindsight, (c) an ORACLE that always follows a source that was right. The oracle is an upper bound no learner can reach;
    the gap between (a) and (c) is the most that any arbitration could ever add, so a tiny gap means 'do not bother'."""
    S = sf.stances.to_numpy(dtype=float)
    pos, neg = sides(S, cfg.deadband)
    dis = pos.any(axis=1) & neg.any(axis=1) & (conflict_score(S) >= cfg.conflict_min)
    m = dis & np.isfinite(sf.y) & (sf.y != 0)
    n = int(m.sum())
    if n < cfg.min_pair_n:
        return {"n": n, "verdict": "INSUFFICIENT"}
    cons = consensus(S)[m]
    y = sf.y[m]
    maj = np.where(cons == 0, 0.5, (cons * y > 0).astype(float))
    per_src = {c: float((S[m, j] * y > 0)[np.abs(S[m, j]) > cfg.deadband].mean()) for j, c in enumerate(sf.sources)
               if (np.abs(S[m, j]) > cfg.deadband).any()}
    best = max(per_src, key=per_src.get)
    oracle = float(((S[m] * y[:, None] > 0) & (np.abs(S[m]) > cfg.deadband)).any(axis=1).mean())
    return {"n": n, "majority_vote": float(maj.mean()), "best_single_hindsight": per_src[best], "best_source": best,
            "oracle_ceiling": oracle, "headroom": oracle - float(maj.mean()), "verdict": "OK"}


# ---------------------------------------------------------------------------------------------- rule stability and walk-forward
def rule_stability(sf: StanceFrame, cfg: DisagreementConfig, now, n_rep: int = 12, seed: int = 0) -> pd.DataFrame:
    """Re-learn rules on week-resamples of the training window; per original rule, the share of resamples in which the same
    (kind, trusted, against, context) rule reappears. A rule that shows up in half the resamples is a coin flip, not knowledge."""
    base = learn_rules(sf, cfg, now)
    if not base:
        return pd.DataFrame(columns=["rule_id", "kind", "trust", "against", "context", "reappear"])
    rng = np.random.default_rng(seed)
    codes = sf.weeks()
    uniq = np.unique(codes)
    members = {u: np.flatnonzero(codes == u) for u in uniq}
    keys = lambda rules: {(r.kind, r.trust, r.against, r.context) for r in rules}
    hits = np.zeros(len(base))
    for _ in range(n_rep):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([members[u] for u in pick])
        got = keys(learn_rules(sf.take(ix), cfg, now))
        for k, r in enumerate(base):
            hits[k] += (r.kind, r.trust, r.against, r.context) in got
    return pd.DataFrame([{"rule_id": r.rule_id, "kind": r.kind, "trust": r.trust, "against": r.against, "context": r.context,
                          "reappear": float(h / n_rep)} for r, h in zip(base, hits)])


def walk_forward_rules(sf: StanceFrame, cfg: DisagreementConfig, now, n_folds: int = 4, seed: int = 0) -> pd.DataFrame:
    """Expanding-window evaluation: for each fold learn rules on all EARLIER matured data, evaluate on the fold. One row per
    fold with the rules' gain over majority vote. Consistent positive gain across folds is the evidence; one fold is not."""
    order = np.argsort(sf.dates.values, kind="stable")
    edges = np.linspace(0, len(order), n_folds + 2).astype(int)
    rows = []
    rng = np.random.default_rng(seed)
    for f in range(1, n_folds + 1):
        test_ix = np.sort(order[edges[f]:edges[f + 1]])
        if len(test_ix) == 0:
            continue
        cut = sf.dates[test_ix].min()
        tr_ix = np.sort(order[: edges[f]])
        tr_ix = tr_ix[(sf.dates[tr_ix] + pd.Timedelta(days=sf.horizon_days)) < cut]        # only matured outcomes
        if len(tr_ix) < 4 * cfg.min_pair_n:
            continue
        rules, ev = evaluate_rules(sf.take(tr_ix), sf.take(test_ix), cfg, now, rng)
        rows.append({"fold": f, "train_n": len(tr_ix), "test_n": len(test_ix), "n_rules": ev.n_rules, "n_applied": ev.n_applied,
                     "gain": ev.gain, "lo": ev.lo, "hi": ev.hi, "placebo_rules": ev.placebo_rules})
    return pd.DataFrame(rows, columns=["fold", "train_n", "test_n", "n_rules", "n_applied", "gain", "lo", "hi", "placebo_rules"])


# ---------------------------------------------------------------------------------------------- checks and calibration
def validate_report(rep: DisagreementReport) -> list[str]:
    """Structural invariants of a DisagreementReport."""
    errs = []
    if not 0.0 <= rep.disagreement_rate <= 1.0:
        errs.append("disagreement_rate outside [0,1]")
    if sum(rep.reasons.values()) != len(rep.events):
        errs.append("reason counts do not add up to the number of events")
    for e in rep.events:
        errs += e.validate()
    for r in rep.rules:
        errs += r.validate()
    for p in rep.pairs:
        if not (0.0 <= p.lo <= p.rate <= p.hi <= 1.0):
            errs.append(f"pair {p.a}/{p.b}/{p.context}: interval does not contain the rate")
    if rep.rule_eval and rep.rule_eval.n_rules != len(rep.rules):
        errs.append("rule_eval.n_rules disagrees with the rule list")
    return errs


def run_planted_calibration(seed: int = 0, cfg: DisagreementConfig | None = None) -> dict:
    """Run the four planted worlds and record what the engine concluded. Calibration harness, not evidence about real data:
      trap        herd trap in crisis -> expect a TRUST(pattern) crisis rule that beats majority vote
      no_trap     no herd trap        -> expect no TRUST rule gain
      shuffled    outcomes shuffled   -> expect no PREDICTIVE verdict and no positive rule gain
      move        sizes scale with uncertainty -> expect conflict PREDICTIVE for abs_move"""
    cfg = cfg or DisagreementConfig()
    worlds = {"trap": dict(herd_trap=True), "no_trap": dict(herd_trap=False, disagreement_predicts_move=False), "shuffled": dict(herd_trap=False, shuffle_outcomes=True),
              "move": dict(herd_trap=False, disagreement_predicts_move=True)}
    out = {}
    for name, kw in worlds.items():
        sf, _ = planted_stances(seed, **kw)
        rep = DisagreementEngine(cfg).assess(sf, "2035-01-01")
        trust = [r for r in rep.rules if r.kind == "TRUST"]
        out[name] = {"n_trust_rules": len(trust), "crisis_pattern_rule": any(r.trust == "pattern" and r.context == "regime=crisis" for r in trust),
                     "rule_gain": None if rep.rule_eval is None else rep.rule_eval.gain,
                     "abs_move": rep.predictive_for("abs_move").verdict.value,
                     "consensus_wrong": rep.predictive_for("consensus_wrong").verdict.value, "errors": validate_report(rep)}
    return out
