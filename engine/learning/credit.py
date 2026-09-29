"""Credit assignment over decision components (SELF_LEARNING_CONTRACT section 20; checklist C08; loop stage "ASSIGN CREDIT /
BLAME", section 4). IMPLEMENTED - NOT VALIDATED.

A decision depends on pattern, analog, memory, direction, timing and risk components. When it succeeds or fails the system
must say WHICH component deserves the credit or blame - never all of them. For every decision this module measures

    with component | without component | difference | interaction | stability

by counterfactual ablation: a component that is "removed" is replaced by its NEUTRAL value (signals 0, timing 1 = "go",
risk 1 = "full size"), the decision is recomputed with the same combiner, and the realised utility is compared.

Mechanisms
  * coalition values v(S) for every subset S of components (vectorised over decisions);
  * leave-one-out (LOO) and solo (add-one) contrasts - LOO alone hides redundant components, solo alone hides gated ones;
  * exact Shapley values (via Harsanyi dividends; sampled permutations when there are too many components), whose sum
    equals v(all) - v(none) exactly (efficiency), so no credit is created or lost;
  * Harsanyi dividends / pairwise Shapley interaction indices = the interaction terms (timing x signal, risk x signal ...);
  * a week-clustered bootstrap (decisions in one week share a market regime, so they are resampled together);
  * stability across resamples (sign consistency, rank stability) and across time halves;
  * a permutation null (the component's inputs are shuffled inside each date) so a component only EARNS credit if its
    Shapley value beats what a meaningless column of the same distribution earns;
  * a blanket-credit detector, an unattributed residual, credit on wins vs blame on losses, per-context breakdown with a
    context-dependence test, credit mapped down to knowledge ids, and postmortem attribution to Subsystem / FailureCause.

Built on engine.ablation (paired block-bootstrap CI and remove-one ablation, reused in `component_ablation`) and the
week-cluster bootstrap idea of engine.direction_ablate. Firewall: only decisions whose outcome matured STRICTLY before
`now` are used (core.require_past); anything else raises FirewallBreach or, in `mature()`, is counted as pending.
Deterministic in `seed`. No I/O except `CreditReport.to_json`."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import itertools
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import (FailureCause, FirewallBreach, Subsystem, as_date, canonical_json, current_code_hash,
                                  require_past, stable_hash)

COMPONENTS = ("pattern", "analog", "memory", "direction", "timing", "risk")
SIGNAL_COMPONENTS = ("pattern", "analog", "memory", "direction")
COMPONENT_SUBSYSTEM = {"pattern": Subsystem.SELECTION, "analog": Subsystem.SELECTION, "memory": Subsystem.SELECTION,
                       "direction": Subsystem.DIRECTION, "timing": Subsystem.TIMING, "risk": Subsystem.RISK}
COMPONENT_CAUSE = {"pattern": FailureCause.SELECTION_ERROR, "analog": FailureCause.SELECTION_ERROR,
                   "memory": FailureCause.SELECTION_ERROR, "direction": FailureCause.SELECTION_ERROR,
                   "timing": FailureCause.TIMING_ERROR, "risk": FailureCause.RISK_ERROR}
NEUTRAL = {"pattern": 0.0, "analog": 0.0, "memory": 0.0, "direction": 0.0, "timing": 1.0, "risk": 1.0}
EXACT_LIMIT = 10                       # 2^10 coalitions x N decisions is the most we enumerate exactly


class Utility(str, enum.Enum):
    PAYOFF = "PAYOFF"                  # decision * outcome : signed position times realised return
    NEG_SQERR = "NEG_SQERR"            # -(decision - outcome)^2 : the decision is a point forecast
    HIT = "HIT"                        # sign(decision) * sign(outcome) : direction only, 0 when abstaining


class CreditVerdict(str, enum.Enum):
    EARNS_CREDIT = "EARNS_CREDIT"                  # positive, distinguishable from a shuffled column, stable
    BLAMED = "BLAMED"                              # reliably harmful
    NO_DETECTABLE_EFFECT = "NO_DETECTABLE_EFFECT"  # interval spans zero: gets NO credit (no blanket credit)
    UNSTABLE = "UNSTABLE"                          # sign flips across resamples or time halves
    INSUFFICIENT = "INSUFFICIENT"                  # too few decisions/weeks to say anything


def utility_values(kind: Utility, decision: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Vectorised realised utility of `decision` against `outcome`."""
    kind = Utility(kind)
    if kind is Utility.PAYOFF:
        return decision * outcome
    if kind is Utility.NEG_SQERR:
        return -(decision - outcome) ** 2
    return np.sign(decision) * np.sign(outcome)


# ---------------------------------------------------------------------------------------------- configuration
@dataclasses.dataclass(frozen=True)
class CreditConfig:
    utility: Utility = Utility.PAYOFF
    seed: int = 7
    alpha: float = 0.05
    n_boot: int = 400
    n_perm: int = 60                   # permutation-null draws per component
    min_decisions: int = 30
    min_groups: int = 8                # distinct weeks
    stability_min: float = 0.80        # share of resamples that must agree in sign with the full sample
    half_agreement_min: bool = True    # the two time halves must not disagree in sign
    context_min_n: int = 15
    blanket_tol: float = 0.25          # relative deviation from equal split below which credit looks "blanket"
    interaction_min_share: float = 0.5 # |interaction| share of the total effect above which failure is INTERACTION
    n_perm_orders: int = 200           # permutations for sampled Shapley (only if components > EXACT_LIMIT)
    neutral: Mapping[str, float] = dataclasses.field(default_factory=lambda: dict(NEUTRAL))

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.alpha < 0.5:
            errs.append("alpha must be in (0, 0.5)")
        for f in ("n_boot", "n_perm", "min_decisions", "min_groups", "context_min_n", "n_perm_orders"):
            if getattr(self, f) < 1:
                errs.append(f"{f} must be >= 1")
        if self.n_perm >= 1 and 1.0 / (1 + self.n_perm) >= self.alpha:
            errs.append("n_perm too small: the smallest possible permutation p-value is not below alpha")
        if not 0.5 <= self.stability_min <= 1.0:
            errs.append("stability_min must be in [0.5, 1]")
        if not 0.0 <= self.blanket_tol <= 1.0:
            errs.append("blanket_tol must be in [0, 1]")
        try:
            Utility(self.utility)
        except ValueError:
            errs.append(f"unknown utility {self.utility!r}")
        return errs


# ---------------------------------------------------------------------------------------------- decision records
@dataclasses.dataclass(frozen=True)
class Decision:
    """One decision as logged at decision time plus its matured outcome. `scores` are what each component contributed
    (signal strength for pattern/analog/memory/direction, a go/no-go gate in [0,1] for timing, a size multiplier for
    risk). Missing components mean the component did not exist for this decision (treated as neutral, not as evidence).
    `knowledge` maps component -> {knowledge_id: weight}; it is what lets component credit flow down to knowledge items.
    Deliberately no ticker: credit is attributed to components and contexts, never to identities (section 29)."""
    decision_id: str
    asof: str                          # date the decision was made (close); fills at the NEXT session's open
    matured: str                       # date the outcome became known
    scores: Mapping[str, float]
    outcome: float
    context: Mapping[str, str] = dataclasses.field(default_factory=dict)
    knowledge: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        if not self.decision_id:
            errs.append("decision_id missing")
        try:
            if as_date(self.matured) <= as_date(self.asof):
                errs.append(f"{self.decision_id}: matured {self.matured} not after asof {self.asof}")
        except (ValueError, TypeError):
            errs.append(f"{self.decision_id}: unparseable dates")
        if not self.scores:
            errs.append(f"{self.decision_id}: no component scores")
        for k, v in self.scores.items():
            if not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
                errs.append(f"{self.decision_id}: score {k}={v!r} not finite")
        if not isinstance(self.outcome, (int, float, np.floating, np.integer)) or not math.isfinite(float(self.outcome)):
            errs.append(f"{self.decision_id}: outcome not finite")
        for comp, ks in self.knowledge.items():
            if comp not in self.scores:
                errs.append(f"{self.decision_id}: knowledge listed for component {comp} that has no score")
            if any(w < 0 for w in ks.values()):
                errs.append(f"{self.decision_id}: negative knowledge weight under {comp}")
        return errs


class DecisionLedger:
    """Append-only store of Decisions (history is never mutated, section 49). Duplicate ids are rejected."""

    def __init__(self, decisions: Iterable[Decision] = ()):
        self._d: dict[str, Decision] = {}
        for d in decisions:
            self.add(d)

    def add(self, d: Decision) -> None:
        errs = d.validate()
        if errs:
            raise ValueError("; ".join(errs))
        if d.decision_id in self._d:
            raise ValueError(f"duplicate decision id {d.decision_id}")
        self._d[d.decision_id] = d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d.values())

    def get(self, decision_id: str) -> Decision:
        return self._d[decision_id]

    def mature(self, now, strict: bool = False) -> tuple[list[Decision], int]:
        """Decisions whose outcome matured strictly before `now`, plus the count still pending. With strict=True a
        pending decision raises FirewallBreach (use it where pending outcomes must not even be present)."""
        ok, pending = [], 0
        for d in self._d.values():
            try:
                require_past(d.matured, now, f"decision {d.decision_id} outcome")
            except FirewallBreach:
                if strict:
                    raise
                pending += 1
                continue
            ok.append(d)
        ok.sort(key=lambda d: (as_date(d.asof), d.decision_id))
        return ok, pending

    def components(self) -> tuple[str, ...]:
        seen = []
        for d in self._d.values():
            for c in d.scores:
                if c not in seen:
                    seen.append(c)
        return tuple(sorted(seen, key=lambda c: (COMPONENTS.index(c) if c in COMPONENTS else 99, c)))


def week_code(asof: Sequence[str]) -> np.ndarray:
    """ISO year*100+week per decision - the cluster id for bootstraps (a week's decisions share a regime)."""
    iso = pd.DatetimeIndex([pd.Timestamp(as_date(a)) for a in asof]).isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


@dataclasses.dataclass
class DecisionFrame:
    """Arrays built once from matured decisions; every statistic below works on these."""
    ids: list
    scores: pd.DataFrame               # decisions x components (neutral-filled where a component was absent)
    outcome: np.ndarray
    weeks: np.ndarray
    dates: np.ndarray                  # datetime64[D] of asof
    context: pd.DataFrame              # decisions x context dimensions (object)
    knowledge: list
    present: pd.DataFrame              # bool: component actually present

    @property
    def n(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, decisions: Sequence[Decision], components: Sequence[str] | None = None,
              neutral: Mapping[str, float] | None = None) -> "DecisionFrame":
        neutral = dict(NEUTRAL if neutral is None else neutral)
        comps = list(components) if components is not None else list(DecisionLedger(decisions).components())
        rows, present = [], []
        for d in decisions:
            rows.append({c: float(d.scores.get(c, neutral.get(c, 0.0))) for c in comps})
            present.append({c: c in d.scores for c in comps})
        sc = pd.DataFrame(rows, columns=comps, index=range(len(decisions)))
        dims = sorted({k for d in decisions for k in d.context})
        ctx = pd.DataFrame([{k: d.context.get(k, "?") for k in dims} for d in decisions], columns=dims,
                           index=range(len(decisions)))
        return cls(ids=[d.decision_id for d in decisions], scores=sc,
                   outcome=np.array([d.outcome for d in decisions], dtype=float),
                   weeks=week_code([d.asof for d in decisions]) if decisions else np.array([], dtype=int),
                   dates=np.array([np.datetime64(as_date(d.asof)) for d in decisions], dtype="datetime64[D]"),
                   context=ctx, knowledge=[d.knowledge for d in decisions],
                   present=pd.DataFrame(present, columns=comps, index=range(len(decisions))))


# ---------------------------------------------------------------------------------------------- combiners
class Combiner:
    """Turns component scores into the decision value (a signed position / point forecast). `__call__` takes a
    DataFrame (decisions x components) and returns a vector. Missing columns are treated as the neutral value."""
    def __call__(self, S: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class WeightedSumCombiner(Combiner):
    """decision = sum_i w_i * score_i. Additive, so Shapley credit is exactly w_i*score_i*outcome under PAYOFF."""
    weights: Mapping[str, float]

    def __call__(self, S):
        out = np.zeros(len(S))
        for c, w in self.weights.items():
            if c in S:
                out += float(w) * S[c].to_numpy(dtype=float)
        return out

    def describe(self):
        return {"kind": "weighted_sum", "weights": dict(self.weights)}


@dataclasses.dataclass(frozen=True)
class StructuredCombiner(Combiner):
    """The production shape: signals are summed with weights, timing is a gate (below `gate` the trade is skipped, above
    it the signal passes), risk is a size multiplier. Because gate and size multiply the signal, timing and risk
    interact with the signals - the interaction terms of this module are what expose that."""
    weights: Mapping[str, float]
    gate: float = 0.5
    soft_gate: bool = False

    def __call__(self, S):
        sig = np.zeros(len(S))
        for c, w in self.weights.items():
            if c in S and c not in ("timing", "risk"):
                sig += float(w) * S[c].to_numpy(dtype=float)
        if "timing" in S:
            t = S["timing"].to_numpy(dtype=float)
            sig = sig * (np.clip(t, 0, 1) if self.soft_gate else (t >= self.gate).astype(float))
        if "risk" in S:
            sig = sig * S["risk"].to_numpy(dtype=float)
        return sig

    def describe(self):
        return {"kind": "structured", "weights": dict(self.weights), "gate": self.gate, "soft_gate": self.soft_gate}


@dataclasses.dataclass(frozen=True)
class FunctionCombiner(Combiner):
    """Wrap any callable(DataFrame)->array, e.g. a fitted model's predict. `name` identifies it in provenance."""
    fn: Callable
    name: str = "custom"

    def __call__(self, S):
        return np.asarray(self.fn(S), dtype=float)

    def describe(self):
        return {"kind": "function", "name": self.name}


# ---------------------------------------------------------------------------------------------- coalition values
class Coalitions:
    """v(S) per decision for every subset S of components. Component k is bit k of the mask. Components outside S are
    set to their neutral value. Results are cached, so LOO, solo, Shapley and dividends share the same evaluations."""

    def __init__(self, frame: DecisionFrame, combiner: Combiner, utility: Utility, neutral: Mapping[str, float]):
        self.comps = list(frame.scores.columns)
        self.n_comp = len(self.comps)
        self.frame, self.combiner, self.utility = frame, combiner, Utility(utility)
        self._S = frame.scores.to_numpy(dtype=float)
        self._neutral = np.array([neutral.get(c, 0.0) for c in self.comps], dtype=float)
        self._cache: dict[int, np.ndarray] = {}

    @property
    def full_mask(self) -> int:
        return (1 << self.n_comp) - 1

    def decision(self, mask: int) -> np.ndarray:
        cols = np.array([(mask >> k) & 1 for k in range(self.n_comp)], dtype=bool)
        M = np.where(cols[None, :], self._S, self._neutral[None, :])
        return self.combiner(pd.DataFrame(M, columns=self.comps))

    def v(self, mask: int) -> np.ndarray:
        if mask not in self._cache:
            self._cache[mask] = utility_values(self.utility, self.decision(mask), self.frame.outcome)
        return self._cache[mask]

    def all_values(self) -> np.ndarray:
        """(2^n, N) matrix of coalition values."""
        return np.vstack([self.v(m) for m in range(1 << self.n_comp)])


def harsanyi_dividends(V: np.ndarray) -> np.ndarray:
    """Moebius transform of coalition values: d(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T). d(S) for |S| >= 2 is the pure
    interaction of exactly the members of S; d({i}) is the standalone effect of i; sum_S d(S) = v(all) - v(empty) with
    v(empty) included as d(empty). Works on the leading axis, in O(n 2^n)."""
    D = V.copy()
    n = int(round(math.log2(D.shape[0])))
    for k in range(n):
        bit = 1 << k
        for m in range(D.shape[0]):
            if m & bit:
                D[m] -= D[m ^ bit]
    return D


def shapley_from_dividends(D: np.ndarray, n_comp: int) -> np.ndarray:
    """phi_i = sum_{S contains i} d(S)/|S|  ->  (N, n_comp)."""
    N = D.shape[1]
    phi = np.zeros((N, n_comp))
    for m in range(1, D.shape[0]):
        size = bin(m).count("1")
        for i in range(n_comp):
            if m >> i & 1:
                phi[:, i] += D[m] / size
    return phi


def shapley_direct(co: Coalitions) -> np.ndarray:
    """Textbook exact Shapley: weighted average of marginal contributions over all coalitions. Used to cross-check the
    dividend route (the two must agree to floating point)."""
    n = co.n_comp
    fact = [math.factorial(k) for k in range(n + 1)]
    phi = np.zeros((co.frame.n, n))
    for i in range(n):
        rest = [k for k in range(n) if k != i]
        for r in range(n):
            w = fact[r] * fact[n - r - 1] / fact[n]
            for sub in itertools.combinations(rest, r):
                m = sum(1 << k for k in sub)
                phi[:, i] += w * (co.v(m | (1 << i)) - co.v(m))
    return phi


def shapley_sampled(co: Coalitions, n_orders: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation-sampled Shapley for many components: the average marginal contribution along random orders. Each order
    telescopes to v(all)-v(none), so efficiency holds exactly for every sample."""
    n = co.n_comp
    phi = np.zeros((co.frame.n, n))
    for _ in range(n_orders):
        order = rng.permutation(n)
        mask, prev = 0, co.v(0)
        for i in order:
            mask |= 1 << int(i)
            cur = co.v(mask)
            phi[:, i] += cur - prev
            prev = cur
    return phi / n_orders


def shapley_values(co: Coalitions, cfg: CreditConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """(phi[N,n], dividends or None). Exact when n_comp <= EXACT_LIMIT."""
    if co.n_comp <= EXACT_LIMIT:
        D = harsanyi_dividends(co.all_values())
        return shapley_from_dividends(D, co.n_comp), D
    return shapley_sampled(co, cfg.n_perm_orders, rng), None


def loo_and_solo(co: Coalitions) -> tuple[np.ndarray, np.ndarray]:
    """LOO_i = v(all) - v(all \\ i) ; solo_i = v({i}) - v(empty). Both (N, n)."""
    n, full = co.n_comp, co.full_mask
    loo = np.column_stack([co.v(full) - co.v(full ^ (1 << i)) for i in range(n)])
    solo = np.column_stack([co.v(1 << i) - co.v(0) for i in range(n)])
    return loo, solo


def pair_interactions(D: np.ndarray, n_comp: int) -> dict[tuple[int, int], np.ndarray]:
    """Shapley interaction index I_ij = sum_{S contains i,j} d(S)/(|S|-1) per decision. Positive = the two components
    are worth more together than apart (synergy); negative = they substitute for each other (redundancy)."""
    out = {}
    for i, j in itertools.combinations(range(n_comp), 2):
        acc = np.zeros(D.shape[1])
        for m in range(1, D.shape[0]):
            if (m >> i & 1) and (m >> j & 1):
                acc += D[m] / (bin(m).count("1") - 1)
        out[(i, j)] = acc
    return out


# ---------------------------------------------------------------------------------------------- resampling statistics
def _group_index(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(codes, return_inverse=True)
    return uniq, inv


def cluster_bootstrap_mean(values: np.ndarray, codes: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap of the row-weighted mean, resampling whole clusters (weeks). values: (N,) or (N, k) -> (n_boot, k)."""
    v = values.reshape(len(values), -1)
    uniq, inv = _group_index(codes)
    g = len(uniq)
    sums = np.zeros((g, v.shape[1]))
    np.add.at(sums, inv, v)
    cnt = np.bincount(inv, minlength=g).astype(float)
    idx = rng.integers(0, g, size=(n_boot, g))
    return sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)[:, None]


def within_group_permutation(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index permutation that shuffles rows only inside their own group (date/week), so a shuffled component keeps its
    marginal distribution and its date-level regime but loses its link to this decision's outcome."""
    base = np.argsort(codes, kind="stable")
    shuffled = np.lexsort((rng.random(len(codes)), codes))
    perm = np.empty(len(codes), dtype=int)
    perm[base] = shuffled
    return perm


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def stability_across_resamples(phi: np.ndarray, codes: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """For each component: sign_stability = share of cluster-resamples whose mean credit has the sign of the full-sample
    mean; rank_stability = mean Spearman correlation between the resample's component ranking and the full ranking;
    top_frequency = share of resamples in which the component is the top earner."""
    boots = cluster_bootstrap_mean(phi, codes, rng, n_boot)
    full = phi.mean(axis=0)
    sign = np.sign(full)
    sign_stab = np.array([(np.sign(boots[:, k]) == sign[k]).mean() if sign[k] != 0 else 0.0 for k in range(phi.shape[1])])
    ranks = np.array([_spearman(full, b) for b in boots])
    top = np.bincount(boots.argmax(axis=1), minlength=phi.shape[1]) / n_boot
    return {"sign": sign_stab, "rank": float(np.nanmean(ranks)) if np.isfinite(ranks).any() else float("nan"),
            "top": top, "boots": boots}


def half_split_agreement(phi: np.ndarray, dates: np.ndarray) -> dict:
    """Split decisions at the median date; compare mean credit in the early and late halves. `agree[k]` is True when the
    two halves do not contradict in sign; `corr` is the correlation of the two credit vectors."""
    if len(dates) < 4:
        return {"agree": np.ones(phi.shape[1], dtype=bool), "early": np.full(phi.shape[1], np.nan),
                "late": np.full(phi.shape[1], np.nan), "corr": float("nan")}
    cut = np.sort(dates)[len(dates) // 2]
    early, late = phi[dates < cut], phi[dates >= cut]
    if len(early) == 0 or len(late) == 0:
        cut = np.median(dates.astype("int64"))
        early, late = phi[dates.astype("int64") <= cut], phi[dates.astype("int64") > cut]
    e, l = early.mean(axis=0), late.mean(axis=0)
    agree = (np.sign(e) * np.sign(l)) >= 0
    return {"agree": agree, "early": e, "late": l, "corr": _spearman(e, l)}


def permutation_null(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, comp_index: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Null distribution of component `comp_index`'s mean Shapley credit when its inputs are shuffled inside each date.
    A real component must beat this; a component that only 'wins' because it is present cannot."""
    comps = list(frame.scores.columns)
    out = np.empty(cfg.n_perm)
    for b in range(cfg.n_perm):
        perm = within_group_permutation(frame.dates.astype("int64"), rng)
        sc = frame.scores.copy()
        sc.iloc[:, comp_index] = sc.iloc[perm, comp_index].to_numpy()
        fr = dataclasses.replace(frame, scores=sc)
        co = Coalitions(fr, combiner, cfg.utility, cfg.neutral)
        D = harsanyi_dividends(co.all_values()) if len(comps) <= EXACT_LIMIT else None
        if D is not None:
            out[b] = shapley_from_dividends(D, len(comps))[:, comp_index].mean()
        else:
            out[b] = shapley_sampled(co, max(20, cfg.n_perm_orders // 10), rng)[:, comp_index].mean()
    return out


def null_p_value(observed: float, null: np.ndarray) -> float:
    """Two-sided permutation p with the +1 correction (never exactly 0)."""
    if len(null) == 0:
        return float("nan")
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))


# ---------------------------------------------------------------------------------------------- result records
@dataclasses.dataclass(frozen=True)
class ComponentCredit:
    component: str
    n: int
    n_groups: int
    mean_credit: float                 # mean Shapley value per decision, in utility units
    lo: float
    hi: float
    loo: float                         # v(all) - v(all without it)
    solo: float                        # v(only it) - v(nothing)
    share: float                       # share of EARNED positive credit (0 unless verdict is EARNS_CREDIT)
    null_p: float
    sign_stability: float
    half_agree: bool
    credit_on_wins: float              # mean Shapley on decisions with positive net utility
    blame_on_losses: float             # mean Shapley on decisions with negative net utility (negative = harmful)
    verdict: CreditVerdict
    reasons: tuple = ()

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 <= self.share <= 1.0):
            errs.append(f"{self.component}: share {self.share} outside [0,1]")
        if self.share > 0 and self.verdict is not CreditVerdict.EARNS_CREDIT:
            errs.append(f"{self.component}: credit share without EARNS_CREDIT")
        if self.lo > self.hi:
            errs.append(f"{self.component}: lo > hi")
        return errs


@dataclasses.dataclass(frozen=True)
class InteractionCredit:
    a: str
    b: str
    mean: float
    lo: float
    hi: float
    kind: str                          # SYNERGY / SUBSTITUTES / NONE
    verdict: str                       # RELIABLE / NOT_DISTINGUISHABLE


@dataclasses.dataclass(frozen=True)
class KnowledgeCredit:
    knowledge_id: str
    component: str
    n: int
    mean_credit: float
    lo: float
    hi: float
    verdict: CreditVerdict
    by_context: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DecisionCredit:
    """Postmortem of one decision: who gets credit / blame, and what the failure teaches (sections 9, 23, 24)."""
    decision_id: str
    net_utility: float
    baseline_utility: float
    credit: Mapping[str, float]
    loo: Mapping[str, float]
    interaction_share: float
    dominant: str | None
    subsystem: Subsystem | None
    cause: FailureCause | None
    note: str


@dataclasses.dataclass(frozen=True)
class CreditReport:
    config: CreditConfig
    code_hash: str
    now: str
    n_decisions: int
    n_pending: int
    base_value: float                  # mean v(empty): what "doing nothing" earns
    full_value: float                  # mean v(all)
    efficiency_error: float            # |sum of credits - (full - base)| : must be ~0
    components: tuple
    interactions: tuple
    context_credit: Mapping[str, Mapping[str, Mapping[str, float]]]
    context_dependence: Mapping[str, Mapping[str, float]]
    blanket: bool
    unattributed_fraction: float       # share of the total effect no component reliably owns
    decision_hash: str

    def component(self, name: str) -> ComponentCredit:
        for c in self.components:
            if c.component == name:
                return c
        raise KeyError(name)

    def earners(self) -> list[str]:
        return [c.component for c in self.components if c.verdict is CreditVerdict.EARNS_CREDIT]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(c) | {"verdict": c.verdict.value} for c in self.components])

    def to_json(self) -> str:
        return canonical_json(dataclasses.asdict(self))

    def label(self) -> str:
        return "IMPLEMENTED — NOT VALIDATED"

    def render_text(self) -> str:
        lines = [f"Credit assignment @ {self.now}: {self.n_decisions} matured decisions ({self.n_pending} pending, unused)  "
                 f"[{self.label()}]",
                 f"  do-nothing value {self.base_value:+.5f}   full value {self.full_value:+.5f}   "
                 f"efficiency error {self.efficiency_error:.2e}"]
        for c in sorted(self.components, key=lambda c: -c.mean_credit):
            lines.append(f"  {c.component:<10}{c.mean_credit:+.5f} [{c.lo:+.5f},{c.hi:+.5f}] loo {c.loo:+.5f} solo {c.solo:+.5f} "
                         f"share {c.share:5.1%} stab {c.sign_stability:4.2f} p {c.null_p:.3f}  {c.verdict.value}")
            for r in c.reasons:
                lines.append(f"      - {r}")
        for i in self.interactions:
            lines.append(f"  interaction {i.a} x {i.b}: {i.mean:+.5f} [{i.lo:+.5f},{i.hi:+.5f}] {i.kind} ({i.verdict})")
        if self.blanket:
            lines.append("  WARNING: credit is spread almost equally - looks like blanket credit, not attribution")
        lines.append(f"  unattributed: {self.unattributed_fraction:.1%} of the total effect has no reliable owner")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the engine
class CreditEngine:
    """Runs the full assessment on a ledger. Stateless between calls; everything is a function of (ledger, now, seed)."""

    def __init__(self, combiner: Combiner, cfg: CreditConfig | None = None):
        self.combiner = combiner
        self.cfg = cfg or CreditConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))

    # ---- core
    def _frame(self, ledger: DecisionLedger, now) -> tuple[DecisionFrame, int]:
        decisions, pending = ledger.mature(now)
        if not decisions:
            return DecisionFrame.build([], ledger.components(), self.cfg.neutral), pending
        return DecisionFrame.build(decisions, ledger.components(), self.cfg.neutral), pending

    def raw_credit(self, frame: DecisionFrame) -> dict:
        """Per-decision arrays: phi (Shapley), loo, solo, pair interaction indices, values, dividends."""
        rng = np.random.default_rng(np.random.SeedSequence(self.cfg.seed).spawn(1)[0])
        co = Coalitions(frame, self.combiner, self.cfg.utility, self.cfg.neutral)
        phi, D = shapley_values(co, self.cfg, rng)
        loo, solo = loo_and_solo(co)
        inter = pair_interactions(D, co.n_comp) if D is not None else {}
        return {"co": co, "phi": phi, "dividends": D, "loo": loo, "solo": solo, "pairs": inter,
                "v_empty": co.v(0), "v_full": co.v(co.full_mask)}

    def assess(self, ledger: DecisionLedger, now) -> CreditReport:
        cfg = self.cfg
        frame, pending = self._frame(ledger, now)
        code = current_code_hash()
        if frame.n == 0:
            return CreditReport(cfg, code, str(as_date(now)), 0, pending, 0.0, 0.0, 0.0, (), (), {}, {}, False, 1.0,
                                stable_hash([]))
        comps = list(frame.scores.columns)
        raw = self.raw_credit(frame)
        phi, loo, solo = raw["phi"], raw["loo"], raw["solo"]
        seeds = np.random.SeedSequence(cfg.seed).spawn(6)
        rng_boot, rng_stab, rng_null, rng_int, rng_ctx, _ = [np.random.default_rng(s) for s in seeds]
        boots = cluster_bootstrap_mean(phi, frame.weeks, rng_boot, cfg.n_boot)
        lo = np.quantile(boots, cfg.alpha / 2, axis=0)
        hi = np.quantile(boots, 1 - cfg.alpha / 2, axis=0)
        stab = stability_across_resamples(phi, frame.weeks, rng_stab, cfg.n_boot)
        half = half_split_agreement(phi, frame.dates)
        n_groups = len(np.unique(frame.weeks))
        net = raw["v_full"] - raw["v_empty"]
        wins, losses = net > 0, net < 0
        mean_phi = phi.mean(axis=0)
        enough = frame.n >= cfg.min_decisions and n_groups >= cfg.min_groups
        rows = []
        for k, c in enumerate(comps):
            null_p = null_p_value(mean_phi[k], permutation_null(frame, self.combiner, cfg, k, rng_null)) if enough else float("nan")
            reasons, verdict = [], CreditVerdict.NO_DETECTABLE_EFFECT
            if not enough:
                verdict = CreditVerdict.INSUFFICIENT
                reasons.append(f"only {frame.n} decisions in {n_groups} weeks (need {cfg.min_decisions}/{cfg.min_groups})")
            elif lo[k] > 0 and null_p <= cfg.alpha:
                verdict = CreditVerdict.EARNS_CREDIT
            elif hi[k] < 0 and null_p <= cfg.alpha:
                verdict = CreditVerdict.BLAMED
            elif lo[k] > 0 or hi[k] < 0:
                reasons.append(f"interval excludes 0 but is not distinguishable from a shuffled input (p={null_p:.3f})")
            else:
                reasons.append("interval spans zero: no credit granted")
            if verdict in (CreditVerdict.EARNS_CREDIT, CreditVerdict.BLAMED):
                if stab["sign"][k] < cfg.stability_min:
                    verdict = CreditVerdict.UNSTABLE
                    reasons.append(f"sign agrees in only {stab['sign'][k]:.0%} of resamples")
                elif cfg.half_agreement_min and not half["agree"][k]:
                    verdict = CreditVerdict.UNSTABLE
                    reasons.append("early and late halves disagree in sign")
            rows.append(dict(component=c, n=frame.n, n_groups=n_groups, mean_credit=float(mean_phi[k]), lo=float(lo[k]),
                             hi=float(hi[k]), loo=float(loo[:, k].mean()), solo=float(solo[:, k].mean()), null_p=null_p,
                             sign_stability=float(stab["sign"][k]), half_agree=bool(half["agree"][k]),
                             credit_on_wins=float(phi[wins, k].mean()) if wins.any() else float("nan"),
                             blame_on_losses=float(phi[losses, k].mean()) if losses.any() else float("nan"),
                             verdict=verdict, reasons=tuple(reasons)))
        earned = np.array([max(r["mean_credit"], 0.0) if r["verdict"] is CreditVerdict.EARNS_CREDIT else 0.0 for r in rows])
        shares = earned / earned.sum() if earned.sum() > 0 else earned
        components = tuple(ComponentCredit(share=float(s), **r) for s, r in zip(shares, rows))
        total = float(net.mean())
        unattributed = 1.0 - (float(earned.sum()) / total if total > 1e-15 and earned.sum() > 0 else 0.0)
        unattributed = float(min(1.0, max(0.0, unattributed))) if total > 1e-15 else 1.0
        interactions = self._interaction_records(raw, comps, frame, rng_int)
        ctx_credit, ctx_dep = self.context_breakdown(frame, phi, rng_ctx)
        eff_err = float(np.abs(phi.sum(axis=1) - net).max())
        return CreditReport(cfg, code, str(as_date(now)), frame.n, pending, float(raw["v_empty"].mean()),
                            float(raw["v_full"].mean()), eff_err, components, tuple(interactions), ctx_credit, ctx_dep,
                            blanket_credit(mean_phi, cfg.blanket_tol), unattributed,
                            stable_hash({"ids": frame.ids, "seed": cfg.seed}))

    # ---- interactions
    def _interaction_records(self, raw, comps, frame, rng) -> list[InteractionCredit]:
        out = []
        if not raw["pairs"]:
            return out
        keys = list(raw["pairs"])
        M = np.column_stack([raw["pairs"][k] for k in keys])
        boots = cluster_bootstrap_mean(M, frame.weeks, rng, self.cfg.n_boot)
        lo = np.quantile(boots, self.cfg.alpha / 2, axis=0)
        hi = np.quantile(boots, 1 - self.cfg.alpha / 2, axis=0)
        mean = M.mean(axis=0)
        for j, (a, b) in enumerate(keys):
            reliable = lo[j] > 0 or hi[j] < 0
            kind = "NONE" if not reliable else ("SYNERGY" if mean[j] > 0 else "SUBSTITUTES")
            out.append(InteractionCredit(comps[a], comps[b], float(mean[j]), float(lo[j]), float(hi[j]), kind,
                                         "RELIABLE" if reliable else "NOT_DISTINGUISHABLE"))
        out.sort(key=lambda i: -abs(i.mean))
        return out

    # ---- context
    def context_breakdown(self, frame: DecisionFrame, phi: np.ndarray, rng: np.random.Generator):
        """Mean Shapley credit per component inside each context value, and a permutation test of whether a component's
        credit depends on the context dimension at all (max-min spread of group means vs shuffled group labels)."""
        comps = list(frame.scores.columns)
        credit: dict = {}
        dep: dict = {}
        for dim in frame.context.columns:
            labels = frame.context[dim].to_numpy(dtype=object)
            groups = {g: np.flatnonzero(labels == g) for g in sorted(set(labels))}
            groups = {g: ix for g, ix in groups.items() if len(ix) >= self.cfg.context_min_n}
            credit[dim] = {g: {c: float(phi[ix, k].mean()) for k, c in enumerate(comps)} for g, ix in groups.items()}
            dep[dim] = {}
            if len(groups) < 2:
                continue
            keep = np.concatenate(list(groups.values()))
            lab = labels[keep]
            for k, c in enumerate(comps):
                vals = phi[keep, k]
                spread = _spread(vals, lab)
                null = np.array([_spread(vals, rng.permutation(lab)) for _ in range(max(50, self.cfg.n_perm))])
                dep[dim][c] = float((1 + np.sum(null >= spread)) / (1 + len(null)))
        return credit, dep

    # ---- knowledge
    def knowledge_credit(self, ledger: DecisionLedger, now, min_n: int = 5) -> list[KnowledgeCredit]:
        """Push component credit down to the knowledge items that produced the component's signal. A component's Shapley
        value on a decision is split among its knowledge ids in proportion to their weights; ids never listed get nothing."""
        decisions, _ = ledger.mature(now)
        if not decisions:
            return []
        frame = DecisionFrame.build(decisions, ledger.components(), self.cfg.neutral)
        phi = self.raw_credit(frame)["phi"]
        comps = list(frame.scores.columns)
        acc: dict = {}
        for r, kn in enumerate(frame.knowledge):
            for comp, ws in kn.items():
                tot = sum(ws.values())
                if comp not in comps or tot <= 0:
                    continue
                for kid, w in ws.items():
                    acc.setdefault((kid, comp), []).append((r, phi[r, comps.index(comp)] * w / tot))
        rng = np.random.default_rng(self.cfg.seed + 11)
        out = []
        for (kid, comp), items in sorted(acc.items()):
            rows = np.array([i for i, _ in items])
            vals = np.array([v for _, v in items])
            if len(vals) < min_n:
                out.append(KnowledgeCredit(kid, comp, len(vals), float(vals.mean()), float("nan"), float("nan"),
                                           CreditVerdict.INSUFFICIENT))
                continue
            b = cluster_bootstrap_mean(vals, frame.weeks[rows], rng, self.cfg.n_boot)[:, 0]
            lo, hi = np.quantile(b, [self.cfg.alpha / 2, 1 - self.cfg.alpha / 2])
            verdict = (CreditVerdict.EARNS_CREDIT if lo > 0 else CreditVerdict.BLAMED if hi < 0
                       else CreditVerdict.NO_DETECTABLE_EFFECT)
            by_ctx = {}
            for dim in frame.context.columns:
                lab = frame.context[dim].to_numpy(dtype=object)[rows]
                for g in sorted(set(lab)):
                    sel = lab == g
                    if sel.sum() >= max(3, min_n // 2):
                        by_ctx[f"{dim}={g}"] = float(vals[sel].mean())
            out.append(KnowledgeCredit(kid, comp, len(vals), float(vals.mean()), float(lo), float(hi), verdict, by_ctx))
        return out

    # ---- postmortem
    def explain(self, ledger: DecisionLedger, decision_id: str, now, report: CreditReport | None = None) -> DecisionCredit:
        """Who caused this outcome. The decision's own counterfactuals give credit/blame; the population `report` (if
        supplied) says whether that component is reliably a source of error, so a one-off blame is not over-read."""
        d = ledger.get(decision_id)
        require_past(d.matured, now, f"decision {decision_id} outcome")
        frame = DecisionFrame.build([d], ledger.components(), self.cfg.neutral)
        raw = self.raw_credit(frame)
        comps = list(frame.scores.columns)
        phi = raw["phi"][0]
        net = float(raw["v_full"][0] - raw["v_empty"][0])
        credit = {c: float(phi[k]) for k, c in enumerate(comps)}
        loo = {c: float(raw["loo"][0, k]) for k, c in enumerate(comps)}
        inter_mass = 0.0
        D = raw["dividends"]
        if D is not None and len(comps) >= 2:
            inter_mass = float(sum(abs(D[m, 0]) for m in range(1, D.shape[0]) if bin(m).count("1") >= 2))
            total_mass = inter_mass + float(sum(abs(D[1 << k, 0]) for k in range(len(comps))))
            inter_share = inter_mass / total_mass if total_mass > 1e-15 else 0.0
        else:
            inter_share = 0.0
        failed = net < 0
        if not failed:
            top = max(credit, key=credit.get) if any(v != 0 for v in credit.values()) else None
            return DecisionCredit(decision_id, float(raw["v_full"][0]), float(raw["v_empty"][0]), credit, loo, inter_share,
                                  top, COMPONENT_SUBSYSTEM.get(top) if top else None, None,
                                  "not a failure relative to doing nothing" if top else "no component moved the decision")
        worst = min(credit, key=credit.get)
        if abs(credit[worst]) < 1e-15:
            return DecisionCredit(decision_id, float(raw["v_full"][0]), float(raw["v_empty"][0]), credit, loo, inter_share,
                                  None, None, FailureCause.UNKNOWN, "loss with no component holding measurable blame")
        cause, note = COMPONENT_CAUSE.get(worst, FailureCause.UNKNOWN), f"largest blame on {worst}"
        if inter_share >= self.cfg.interaction_min_share:
            cause, note = FailureCause.INTERACTION_FAILURE, f"{inter_share:.0%} of the effect is interaction, led by {worst}"
        if report is not None:
            try:
                pop = report.component(worst)
                if pop.verdict is CreditVerdict.INSUFFICIENT:
                    cause, note = FailureCause.INSUFFICIENT_EVIDENCE, f"no population evidence about {worst} yet"
                elif pop.verdict in (CreditVerdict.NO_DETECTABLE_EFFECT, CreditVerdict.UNSTABLE) and cause != FailureCause.INTERACTION_FAILURE:
                    cause, note = FailureCause.UNKNOWN, f"{worst} blamed once but its population credit is {pop.verdict.value}"
            except KeyError:
                pass
        return DecisionCredit(decision_id, float(raw["v_full"][0]), float(raw["v_empty"][0]), credit, loo, inter_share,
                              worst, COMPONENT_SUBSYSTEM.get(worst), cause, note)

    def postmortems(self, ledger: DecisionLedger, now, report: CreditReport | None = None, worst_n: int = 20) -> list[DecisionCredit]:
        decisions, _ = ledger.mature(now)
        frame = DecisionFrame.build(decisions, ledger.components(), self.cfg.neutral) if decisions else None
        if frame is None:
            return []
        raw = self.raw_credit(frame)
        net = raw["v_full"] - raw["v_empty"]
        order = np.argsort(net)[:worst_n]
        return [self.explain(ledger, frame.ids[i], now, report) for i in order if net[i] < 0]


# ---------------------------------------------------------------------------------------------- diagnostics
def _spread(vals: np.ndarray, labels: np.ndarray) -> float:
    s = pd.Series(vals).groupby(labels).mean()
    return float(s.max() - s.min())


def blanket_credit(mean_credit: Sequence[float], tol: float) -> bool:
    """True if positive credit is (nearly) equally shared between three or more components. Real attribution is uneven;
    an equal split is what "everything gets credit" looks like."""
    v = np.asarray(mean_credit, dtype=float)
    pos = v[v > 0]
    if len(pos) < 3:
        return False
    return bool(np.all(np.abs(pos / pos.mean() - 1.0) <= tol))


def efficiency_gap(phi: np.ndarray, v_full: np.ndarray, v_empty: np.ndarray) -> float:
    return float(np.abs(phi.sum(axis=1) - (v_full - v_empty)).max())


def redundancy_masking(loo: np.ndarray, phi: np.ndarray, solo: np.ndarray, tol: float = 0.25) -> list[int]:
    """Components whose LOO is ~0 while their Shapley/solo credit is clearly positive: a duplicate covers for them, so
    dropping either one alone looks free while dropping both would cost. (Hand-off to redundancy.py.)"""
    l, p, s = loo.mean(axis=0), phi.mean(axis=0), solo.mean(axis=0)
    scale = max(np.abs(p).max(), 1e-15)
    return [k for k in range(len(p)) if abs(l[k]) < tol * scale and p[k] > tol * scale and s[k] > tol * scale]


# ---------------------------------------------------------------------------------------------- bridge to engine.ablation
def period_scores(frame: DecisionFrame, combiner: Combiner, utility: Utility, neutral: Mapping[str, float],
                  active: Iterable[str]) -> pd.Series:
    """Mean realised utility per decision date when only `active` components are on and the rest are neutral - the
    per-period out-of-sample score engine.ablation.ablate expects."""
    act = set(active)
    co = Coalitions(frame, combiner, utility, neutral)
    mask = sum(1 << k for k, c in enumerate(co.comps) if c in act)
    return pd.Series(co.v(mask), index=pd.DatetimeIndex(frame.dates)).groupby(level=0).mean()


def component_ablation(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig | None = None) -> dict:
    """Remove-one ablation through engine.ablation.ablate (paired moving-block bootstrap over dates). Cross-check for the
    LOO column of the credit report: the same contrast with the repo's own significance machinery."""
    from engine import ablation
    cfg = cfg or CreditConfig()
    comps = list(frame.scores.columns)
    return ablation.ablate(lambda active: period_scores(frame, combiner, cfg.utility, cfg.neutral, active), comps,
                           seed=cfg.seed, mode="remove", n_boot=cfg.n_boot)


def simulate_decisions(n: int, seed: int, truth: Callable[[np.random.Generator, pd.DataFrame], np.ndarray],
                       noise: float = 0.5, start: str = "2020-01-06", per_day: int = 4, horizon_days: int = 5,
                       regime_flip_at: float | None = None) -> DecisionLedger:
    """Synthetic decisions for planted-defect tests and calibration: each component score is N(0,1) (timing in {0,1}
    with p=.7, risk in [0.5,1.5]); the outcome is truth(scores) + noise. `truth` states exactly which components matter."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=int(math.ceil(n / per_day)))
    rows = []
    for i in range(n):
        d = days[i // per_day]
        rows.append({"pattern": rng.standard_normal(), "analog": rng.standard_normal(), "memory": rng.standard_normal(),
                     "direction": rng.standard_normal(), "timing": float(rng.random() < 0.7),
                     "risk": float(rng.uniform(0.5, 1.5)), "_d": d})
    df = pd.DataFrame(rows)
    y = truth(rng, df) + noise * rng.standard_normal(n)
    led = DecisionLedger()
    for i, r in df.iterrows():
        d = r["_d"]
        regime = "late" if (regime_flip_at is not None and i >= regime_flip_at * n) else "early"
        led.add(Decision(f"d{i:05d}", d.date().isoformat(), (d + pd.offsets.BDay(horizon_days)).date().isoformat(),
                         {c: float(r[c]) for c in COMPONENTS}, float(y[i]), {"regime": regime}))
    return led
