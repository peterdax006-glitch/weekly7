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
from scipy import stats

from engine.learning.core import (FailureCause, FirewallBreach, Subsystem, as_date, canonical_json, current_code_hash,
                                  require_past, stable_hash)
from engine.pattern_stats import bh_qvalues, cluster_bootstrap_ci, permute_within_clusters, week_codes

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
    n_perm: int = 200                  # permutation-null draws per component
    min_decisions: int = 30
    min_groups: int = 8                # distinct weeks
    stability_min: float = 0.80        # share of resamples that must agree in sign with the full sample
    half_agreement_min: bool = True    # the two time halves must not disagree in sign
    context_min_n: int = 15
    blanket_tol: float = 0.25          # relative deviation from equal split below which credit looks "blanket"
    interaction_min_rel: float = 0.02  # an interaction is material only above this share of the total effect
    interaction_min_share: float = 0.5 # |interaction| share of the total effect above which failure is INTERACTION
    n_perm_orders: int = 200           # orders per legacy sampled call (baseline sensitivity)
    shapley_method: str = "auto"       # auto | exact | sampled ; auto = exact up to EXACT_LIMIT components, sampled above
    sample_rel_se: float = 0.03        # stop sampling when max component SE <= this x the largest |mean credit|
    min_orders: int = 32               # orders drawn before the stopping rule may fire (antithetic pairs count 2)
    max_orders: int = 2000             # hard budget cap on sampled orders
    null_orders: int = 8               # orders per permutation-null draw when sampling
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
        if self.shapley_method not in ("auto", "exact", "sampled"):
            errs.append("shapley_method must be auto, exact or sampled")
        if not 0 < self.sample_rel_se < 1 or self.min_orders < 4 or self.max_orders < self.min_orders or self.null_orders < 2:
            errs.append("sampling controls invalid (need 0<sample_rel_se<1, 4<=min_orders<=max_orders, null_orders>=2)")
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
    """Cluster id per decision (ISO week, engine.pattern_stats.week_codes): a week's decisions share a regime."""
    return week_codes(pd.DatetimeIndex([pd.Timestamp(as_date(a)) for a in asof]))[0]


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

    def array(self, M: np.ndarray, cols: Sequence[str]) -> np.ndarray:
        """Same as __call__ on a plain matrix (columns named `cols`); subclasses override it to skip building a DataFrame,
        which dominates the cost of the permutation null."""
        return self(pd.DataFrame(M, columns=list(cols)))

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

    def array(self, M, cols):
        pos = {c: i for i, c in enumerate(cols)}
        out = np.zeros(len(M))
        for c, w in self.weights.items():
            if c in pos:
                out += float(w) * M[:, pos[c]]
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

    def array(self, M, cols):
        pos = {c: i for i, c in enumerate(cols)}
        sig = np.zeros(len(M))
        for c, w in self.weights.items():
            if c in pos and c not in ("timing", "risk"):
                sig += float(w) * M[:, pos[c]]
        if "timing" in pos:
            t = M[:, pos["timing"]]
            sig = sig * (np.clip(t, 0, 1) if self.soft_gate else (t >= self.gate).astype(float))
        if "risk" in pos:
            sig = sig * M[:, pos["risk"]]
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
        return self.combiner.array(M, self.comps)

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


@dataclasses.dataclass(frozen=True)
class ShapleyInfo:
    """Which Shapley method produced the numbers and how good they are. For 'exact' the error is zero by construction."""
    method: str                        # exact | sampled
    n_components: int
    n_orders: int                      # sampled orders actually drawn (0 for exact)
    max_se: float                      # largest standard error of a component's mean credit (0 for exact)
    rel_se: float                      # max_se / largest |mean credit|
    converged: bool                    # stopping rule met before the budget cap (always True for exact)
    budget: int


def shapley_adaptive(co: Coalitions, cfg: CreditConfig, rng: np.random.Generator) -> tuple[np.ndarray, ShapleyInfo]:
    """Permutation-sampled Shapley with antithetic pairs (an order and its reverse are averaged, which cancels much of the
    variance of order effects) and a standard-error stopping rule. Each sample yields a per-component mean-over-decisions
    vector; the SE of the running mean over samples is checked every 8 samples once `min_orders` orders are drawn, and
    sampling stops when max SE <= sample_rel_se x max |mean|, or at `max_orders`. Every sample telescopes to v(all)-v(none),
    so efficiency is exact regardless of when it stops. Seeded through `rng`."""
    n = co.n_comp
    total = np.zeros((co.frame.n, n))
    samples = []
    drawn, converged, rel = 0, False, float("inf")
    se = np.full(n, np.inf)
    while drawn + 2 <= cfg.max_orders:
        order = rng.permutation(n)
        acc = np.zeros((co.frame.n, n))
        for o in (order, order[::-1]):
            mask, prev = 0, co.v(0)
            for i in o:
                mask |= 1 << int(i)
                cur = co.v(mask)
                acc[:, i] += cur - prev
                prev = cur
        acc /= 2.0
        total += acc
        samples.append(acc.mean(axis=0))
        drawn += 2
        if drawn >= cfg.min_orders and len(samples) % 8 == 0:
            S = np.vstack(samples)
            se = S.std(axis=0, ddof=1) / math.sqrt(len(S))
            rel = float(se.max() / max(np.abs(S.mean(axis=0)).max(), 1e-12))
            if rel <= cfg.sample_rel_se:
                converged = True
                break
    S = np.vstack(samples)
    se = S.std(axis=0, ddof=1) / math.sqrt(len(S)) if len(S) > 1 else np.full(n, np.inf)
    rel = float(se.max() / max(np.abs(S.mean(axis=0)).max(), 1e-12))
    converged = converged or rel <= cfg.sample_rel_se
    return total / len(samples), ShapleyInfo("sampled", n, drawn, float(se.max()), rel, bool(converged), cfg.max_orders)


def shapley_values_info(co: Coalitions, cfg: CreditConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None, ShapleyInfo]:
    """Automatic switch: exact enumeration (with dividends, hence interaction terms) for n_comp <= EXACT_LIMIT, adaptive sampling
    above. shapley_method='exact' above 14 components is refused (2^n memory); 'sampled' forces sampling at any size."""
    n = co.n_comp
    method = cfg.shapley_method
    if method == "exact" and n > 14:
        raise ValueError(f"exact Shapley needs 2^{n} coalitions; use shapley_method='auto' or 'sampled'")
    if method == "exact" or (method == "auto" and n <= EXACT_LIMIT):
        D = harsanyi_dividends(co.all_values())
        return shapley_from_dividends(D, n), D, ShapleyInfo("exact", n, 0, 0.0, 0.0, True, 0)
    phi, info = shapley_adaptive(co, cfg, rng)
    return phi, None, info


def shapley_values(co: Coalitions, cfg: CreditConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """(phi[N,n], dividends or None); see shapley_values_info for the method record."""
    phi, D, _ = shapley_values_info(co, cfg, rng)
    return phi, D


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
    """JOINT bootstrap draws of the row-weighted mean of several columns, resampling whole clusters (weeks) once for all
    columns. Kept because stability (sign and rank agreement across components) needs the draws themselves, which
    pattern_stats.cluster_bootstrap_ci (a single interval) does not return. values: (N,) or (N, k) -> (n_boot, k)."""
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
    marginal distribution and its date-level regime but loses its link to this decision's outcome. Built on
    engine.pattern_stats.permute_within_clusters (dealing an index vector back inside each cluster)."""
    return permute_within_clusters(np.arange(len(codes)), np.asarray(codes), rng)


def cluster_ci(values: np.ndarray, codes: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> tuple[float, float]:
    """Week-clustered percentile CI of the mean of `values` via engine.pattern_stats.cluster_bootstrap_ci."""
    n_cl = int(codes.max()) + 1 if len(codes) else 0
    return cluster_bootstrap_ci(np.asarray(codes, dtype=np.int64), np.ones(len(values)), np.asarray(values, dtype=float),
                                np.ones(len(values), dtype=bool), n_cl, rng, reps=n_boot, level=1 - alpha)


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
                     rng: np.random.Generator, base: "Coalitions | None" = None) -> np.ndarray:
    """Null distribution of component `comp_index`'s mean Shapley credit when its inputs are shuffled inside each date.
    A real component must beat this; a component that only 'wins' because it is present cannot. Coalitions that do not
    contain the shuffled component are unaffected by the shuffle, so their values are copied from `base` (half the work)."""
    n = frame.scores.shape[1]
    base = base or Coalitions(frame, combiner, cfg.utility, cfg.neutral)
    keep = [m for m in range(1 << n) if not (m >> comp_index) & 1]
    out = np.empty(cfg.n_perm)
    for b in range(cfg.n_perm):
        perm = within_group_permutation(frame.dates.astype("int64"), rng)
        sc = frame.scores.copy()
        sc.iloc[:, comp_index] = sc.iloc[perm, comp_index].to_numpy()
        co = Coalitions(dataclasses.replace(frame, scores=sc), combiner, cfg.utility, cfg.neutral)
        for m in keep:
            co._cache[m] = base.v(m)
        if n <= EXACT_LIMIT:
            out[b] = shapley_from_dividends(harsanyi_dividends(co.all_values()), n)[:, comp_index].mean()
        else:
            out[b] = shapley_sampled(co, cfg.null_orders, rng)[:, comp_index].mean()
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
    ablate_verdict: str = ""           # engine.ablation.ablate remove-one verdict (earns_keep/no_effect/harmful/insufficient)
    ablate_delta: float = float("nan")

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
    verdict: str                       # RELIABLE / NOT_DISTINGUISHABLE / IMMATERIAL
    q: float = float("nan")            # BH q-value over all pairs (15 pairs at 5% would otherwise yield a false 'reliable')


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
    shapley_method: str = ""           # exact | sampled (empty for an empty report)
    shapley_orders: int = 0            # sampled orders drawn
    shapley_max_se: float = 0.0
    shapley_converged: bool = True

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
                 f"  Shapley method: {self.shapley_method}" + (f" ({self.shapley_orders} orders, max SE {self.shapley_max_se:.2e}, "
                                                               f"{'converged' if self.shapley_converged else 'BUDGET HIT, not converged'})"
                                                               if self.shapley_method == "sampled" else ""),
                 f"  do-nothing value {self.base_value:+.5f}   full value {self.full_value:+.5f}   "
                 f"efficiency error {self.efficiency_error:.2e}"]
        for c in sorted(self.components, key=lambda c: -c.mean_credit):
            lines.append(f"  {c.component:<10}{c.mean_credit:+.5f} [{c.lo:+.5f},{c.hi:+.5f}] loo {c.loo:+.5f} solo {c.solo:+.5f} "
                         f"share {c.share:5.1%} stab {c.sign_stability:4.2f} p {c.null_p:.3f} ablate {c.ablate_verdict}  {c.verdict.value}")
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
        phi, D, info = shapley_values_info(co, self.cfg, rng)
        loo, solo = loo_and_solo(co)
        inter = pair_interactions(D, co.n_comp) if D is not None else {}
        return {"co": co, "phi": phi, "dividends": D, "loo": loo, "solo": solo, "pairs": inter, "shapley_info": info,
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
        ci = [cluster_ci(phi[:, k], frame.weeks, rng_boot, cfg.n_boot, cfg.alpha) for k in range(phi.shape[1])]
        lo, hi = np.array([c[0] for c in ci]), np.array([c[1] for c in ci])
        stab = stability_across_resamples(phi, frame.weeks, rng_stab, cfg.n_boot)
        half = half_split_agreement(phi, frame.dates)
        n_groups = len(np.unique(frame.weeks))
        net = raw["v_full"] - raw["v_empty"]
        wins, losses = net > 0, net < 0
        mean_phi = phi.mean(axis=0)
        enough = frame.n >= cfg.min_decisions and n_groups >= cfg.min_groups
        abl = component_ablation(frame, self.combiner, cfg)["table"].set_index("component")
        if enough and (1.0 / (1 + cfg.n_perm)) * len(comps) > cfg.alpha:
            raise ValueError(f"n_perm={cfg.n_perm} cannot resolve BH significance across {len(comps)} components at alpha={cfg.alpha}")
        null_qs = np.full(len(comps), np.nan)
        if enough:
            null_ps = np.array([null_p_value(mean_phi[k], permutation_null(frame, self.combiner, cfg, k, rng_null, raw["co"]))
                                for k in range(len(comps))])
            null_qs = bh_qvalues(null_ps)
        rows = []
        for k, c in enumerate(comps):
            null_p = float(null_qs[k]) if enough else float("nan")          # BH-adjusted across components
            reasons, verdict = [], CreditVerdict.NO_DETECTABLE_EFFECT
            if not enough:
                verdict = CreditVerdict.INSUFFICIENT
                reasons.append(f"only {frame.n} decisions in {n_groups} weeks (need {cfg.min_decisions}/{cfg.min_groups})")
            elif lo[k] > 0 and null_p <= cfg.alpha:
                verdict = CreditVerdict.EARNS_CREDIT
            elif hi[k] < 0 and null_p <= cfg.alpha:
                verdict = CreditVerdict.BLAMED
            elif lo[k] > 0 or hi[k] < 0:
                reasons.append(f"interval excludes 0 but is not distinguishable from a shuffled input (q={null_p:.3f})")
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
                             verdict=verdict, reasons=tuple(reasons), ablate_verdict=str(abl.loc[c, "verdict"]),
                             ablate_delta=float(abl.loc[c, "delta"])))
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
                            stable_hash({"ids": frame.ids, "seed": cfg.seed}), shapley_method=raw["shapley_info"].method,
                            shapley_orders=raw["shapley_info"].n_orders, shapley_max_se=raw["shapley_info"].max_se,
                            shapley_converged=raw["shapley_info"].converged)

    # ---- interactions
    def _interaction_records(self, raw, comps, frame, rng) -> list[InteractionCredit]:
        """Pairwise Shapley interaction indices with week-clustered CIs and BH control over all pairs; an interaction is
        RELIABLE only if q <= alpha AND it is material (>= interaction_min_rel of the total effect)."""
        out: list[InteractionCredit] = []
        if not raw["pairs"]:
            return out
        keys = list(raw["pairs"])
        M = np.column_stack([raw["pairs"][k] for k in keys])
        boots = cluster_bootstrap_mean(M, frame.weeks, rng, self.cfg.n_boot)
        ci = [cluster_ci(M[:, j], frame.weeks, rng, self.cfg.n_boot, self.cfg.alpha) for j in range(M.shape[1])]
        mean = M.mean(axis=0)
        sd = boots.std(axis=0, ddof=1)
        # normal-approximation p from the clustered bootstrap SE: the raw bootstrap tail count has a floor of 1/(B+1), too
        # coarse to survive BH over 15 pairs
        z = np.where(sd > 1e-15, mean / np.where(sd > 1e-15, sd, 1.0), np.where(np.abs(mean) > 1e-12, np.inf, 0.0))
        p = 2 * stats.norm.sf(np.abs(z))
        q = bh_qvalues(p)
        total = abs(float((raw["v_full"] - raw["v_empty"]).mean()))
        floor = max(self.cfg.interaction_min_rel * total, 1e-9)
        for j, (a, b) in enumerate(keys):
            material = abs(mean[j]) >= floor
            reliable = q[j] <= self.cfg.alpha and material
            verdict = "RELIABLE" if reliable else ("IMMATERIAL" if q[j] <= self.cfg.alpha else "NOT_DISTINGUISHABLE")
            kind = "NONE" if not reliable else ("SYNERGY" if mean[j] > 0 else "SUBSTITUTES")
            out.append(InteractionCredit(comps[a], comps[b], float(mean[j]), float(ci[j][0]), float(ci[j][1]), kind, verdict, float(q[j])))
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
            _, inv = np.unique(frame.weeks[rows], return_inverse=True)
            lo, hi = cluster_ci(vals, inv, rng, self.cfg.n_boot, self.cfg.alpha)
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
            top = max(credit, key=lambda k: credit[k]) if any(v != 0 for v in credit.values()) else None
            return DecisionCredit(decision_id, float(raw["v_full"][0]), float(raw["v_empty"][0]), credit, loo, inter_share,
                                  top, COMPONENT_SUBSYSTEM.get(top) if top else None, None,
                                  "not a failure relative to doing nothing" if top else "no component moved the decision")
        worst = min(credit, key=lambda k: credit[k])
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


# ---------------------------------------------------------------------------------------------- baseline sensitivity
class _MarginalCoalitions(Coalitions):
    """Coalitions whose absent components take per-decision stand-in values instead of a constant."""

    def __init__(self, frame, combiner, utility, stand_in: np.ndarray):
        super().__init__(frame, combiner, utility, {})
        self._stand_in = stand_in

    def decision(self, mask: int) -> np.ndarray:
        cols = np.array([(mask >> k) & 1 for k in range(self.n_comp)], dtype=bool)
        M = np.where(cols[None, :], self._S, self._stand_in)
        return self.combiner.array(M, self.comps)


def marginal_baseline_credit(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, n_draws: int = 8,
                             seed: int = 0) -> np.ndarray:
    """Shapley credit when a 'removed' component is replaced by a draw from its own marginal distribution (another
    decision's value, shuffled inside the week) instead of the neutral constant. The neutral baseline can make a component
    look useful merely because 0 is a bad stand-in for it; if the two baselines disagree the credit is baseline-dependent.
    Returns the mean over draws of per-decision phi, shape (N, n)."""
    rng = np.random.default_rng(seed)
    n = frame.scores.shape[1]
    S = frame.scores.to_numpy(dtype=float)
    total = np.zeros((frame.n, n))
    for _ in range(n_draws):
        perm = np.column_stack([within_group_permutation(frame.weeks, rng) for _ in range(n)])
        sub = np.column_stack([S[perm[:, k], k] for k in range(n)])          # a marginal draw for every component
        co = _MarginalCoalitions(frame, combiner, cfg.utility, sub)
        if n <= EXACT_LIMIT:
            total += shapley_from_dividends(harsanyi_dividends(co.all_values()), n)
        else:
            total += shapley_sampled(co, max(20, cfg.n_perm_orders // 10), rng)
    return total / n_draws


def baseline_sensitivity(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, phi_neutral: np.ndarray | None = None) -> pd.DataFrame:
    """Per component: mean credit under the neutral baseline vs the marginal baseline, and whether they agree in sign.
    Disagreement = the credit is an artefact of what 'removed' means and must not be trusted without a human look."""
    comps = list(frame.scores.columns)
    if phi_neutral is None:
        co = Coalitions(frame, combiner, cfg.utility, cfg.neutral)
        phi_neutral = shapley_values(co, cfg, np.random.default_rng(cfg.seed))[0]
    phi_m = marginal_baseline_credit(frame, combiner, cfg, seed=cfg.seed + 5)
    rows = []
    for k, c in enumerate(comps):
        a, b = float(phi_neutral[:, k].mean()), float(phi_m[:, k].mean())
        rows.append({"component": c, "neutral": a, "marginal": b,
                     "sign_agrees": bool(np.sign(a) == np.sign(b) or min(abs(a), abs(b)) < 1e-12),
                     "ratio": (b / a) if abs(a) > 1e-12 else float("nan")})
    return pd.DataFrame(rows)


def utility_sensitivity(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig) -> pd.DataFrame:
    """Mean Shapley credit under each utility. A component that earns credit under PAYOFF but not under HIT is being paid for
    magnitude luck, not for being right. NEG_SQERR treats the decision as a point forecast, so it is only meaningful when the
    combiner's output is on the outcome's scale - with an arbitrary scale it punishes every noisy component, which is what
    `robust_*` (all three utilities) then reflects."""
    comps = list(frame.scores.columns)
    out = {}
    for u in Utility:
        co = Coalitions(frame, combiner, u, cfg.neutral)
        phi = shapley_values(co, cfg, np.random.default_rng(cfg.seed))[0]
        cis = [cluster_ci(phi[:, k], frame.weeks, np.random.default_rng(cfg.seed + k), cfg.n_boot, cfg.alpha)
               for k in range(len(comps))]
        out[u.value] = [(float(phi[:, k].mean()), cis[k][0] > 0, cis[k][1] < 0) for k in range(len(comps))]
    rows = []
    for k, c in enumerate(comps):
        r = {"component": c}
        for uname, v in out.items():
            r[f"{uname}_mean"], r[f"{uname}_pos"], r[f"{uname}_neg"] = v[k]
        r["robust_positive"] = all(out[u.value][k][1] for u in Utility)
        r["robust_negative"] = all(out[u.value][k][2] for u in Utility)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- credit over time
def credit_over_time(frame: DecisionFrame, phi: np.ndarray, freq: str = "Q", min_n: int = 20) -> pd.DataFrame:
    """Mean credit per component per calendar period. Credit that used to be earned and no longer is (a decaying pattern) is
    invisible in the pooled mean; this table and `credit_trend` expose it."""
    comps = list(frame.scores.columns)
    df = pd.DataFrame(phi, columns=comps)
    df["_p"] = pd.DatetimeIndex(frame.dates).to_period(freq)
    g = df.groupby("_p")
    out = g[comps].mean()
    out["n"] = g.size()
    return out[out["n"] >= min_n]


def credit_trend(over_time: pd.DataFrame, min_periods: int = 4) -> dict:
    """Per component: Spearman correlation of period index with period credit, and a DECAYING / IMPROVING / STEADY tag.
    A tag needs at least `min_periods` periods and |rho| >= 0.7 - a trend claim from three points is not evidence."""
    comps = [c for c in over_time.columns if c != "n"]
    res = {}
    t = np.arange(len(over_time), dtype=float)
    for c in comps:
        if len(over_time) < min_periods:
            res[c] = {"rho": float("nan"), "tag": "INSUFFICIENT"}
            continue
        r = _spearman(t, over_time[c].to_numpy(dtype=float))
        first = float(over_time[c].iloc[: len(over_time) // 2].mean())
        last = float(over_time[c].iloc[len(over_time) // 2:].mean())
        tag = "STEADY"
        if np.isfinite(r) and abs(r) >= 0.7:
            tag = "DECAYING" if last < first else "IMPROVING"
        res[c] = {"rho": float(r), "first_half": first, "second_half": last, "tag": tag}
    return res


# ---------------------------------------------------------------------------------------------- power
def credit_standard_errors(phi: np.ndarray, codes: np.ndarray, rng: np.random.Generator, n_boot: int = 300) -> np.ndarray:
    return cluster_bootstrap_mean(phi, codes, rng, n_boot).std(axis=0, ddof=1)


def minimum_detectable_credit(se: np.ndarray, alpha: float = 0.05, power: float = 0.8) -> np.ndarray:
    """Smallest true mean credit a two-sided test at `alpha` would detect with `power`, given the clustered standard error.
    A NO_DETECTABLE_EFFECT verdict is only informative for components whose MDE is small relative to what would matter."""
    from scipy.stats import norm
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.asarray(se, dtype=float)


def power_table(report: CreditReport, frame: DecisionFrame, phi: np.ndarray, seed: int = 0) -> pd.DataFrame:
    comps = list(frame.scores.columns)
    se = credit_standard_errors(phi, frame.weeks, np.random.default_rng(seed))
    mde = minimum_detectable_credit(se, report.config.alpha)
    rows = []
    for k, c in enumerate(comps):
        cc = report.component(c)
        rows.append({"component": c, "mean_credit": cc.mean_credit, "se": float(se[k]), "mde": float(mde[k]),
                     "underpowered": bool(cc.verdict is CreditVerdict.NO_DETECTABLE_EFFECT and abs(cc.mean_credit) < mde[k]),
                     "verdict": cc.verdict.value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- failure attribution summary
def failure_attribution_summary(postmortems: Sequence[DecisionCredit]) -> pd.DataFrame:
    """How the losing decisions distribute over (Subsystem, FailureCause): where the system's mistakes live. UNKNOWN is a
    legitimate row - the contract says an honest unknown beats a manufactured cause."""
    rows = [{"subsystem": (p.subsystem.value if p.subsystem else "NONE"), "cause": (p.cause.value if p.cause else "NONE"),
             "loss": -min(p.net_utility - p.baseline_utility, 0.0)} for p in postmortems if p.cause is not None]
    if not rows:
        return pd.DataFrame(columns=["subsystem", "cause", "n", "share", "total_loss"])
    df = pd.DataFrame(rows)
    g = df.groupby(["subsystem", "cause"]).agg(n=("loss", "size"), total_loss=("loss", "sum")).reset_index()
    g["share"] = g["n"] / g["n"].sum()
    return g.sort_values("total_loss", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- graph edges (F08, F09)
def credit_edges(report: CreditReport) -> list[tuple[str, str, str, float]]:
    """(component_a, EDGE, component_b, weight) from reliable interactions: SYNERGY -> COMPLEMENTS (worth more together than
    apart; F09), SUBSTITUTES -> REDUNDANT_WITH (each covers for the other; F08). Only RELIABLE interactions produce edges."""
    from engine.learning.core import Edge
    out = []
    for i in report.interactions:
        if i.verdict != "RELIABLE":
            continue
        e = Edge.COMPLEMENTS if i.kind == "SYNERGY" else Edge.REDUNDANT_WITH
        out.append((i.a, e.value, i.b, float(abs(i.mean))))
    return out


# ---------------------------------------------------------------------------------------------- report checks and history
def validate_report(rep: CreditReport, tol: float = 1e-9) -> list[str]:
    """Invariants any CreditReport must satisfy; a violation means the attribution machinery, not the market, is wrong."""
    errs = []
    if rep.efficiency_error > tol * max(1.0, abs(rep.full_value)):
        errs.append(f"efficiency violated: credits do not sum to full-minus-base (max gap {rep.efficiency_error:.3e})")
    if sum(c.share for c in rep.components) > 1 + 1e-9:
        errs.append("credit shares sum to more than 1")
    for c in rep.components:
        errs += c.validate()
    if any(c.share > 0 for c in rep.components) and rep.blanket:
        errs.append("blanket credit flagged while shares were granted")
    if rep.n_decisions == 0 and rep.components:
        errs.append("components reported for zero decisions")
    return errs


def credit_drift(prev: CreditReport, new: CreditReport) -> pd.DataFrame:
    """Component-by-component change between two reports: did who earns credit change, and by how much."""
    rows = []
    for b in new.components:
        try:
            a = prev.component(b.component)
        except KeyError:
            rows.append({"component": b.component, "prev": float("nan"), "new": b.mean_credit, "delta": float("nan"),
                         "verdict_changed": True})
            continue
        rows.append({"component": b.component, "prev": a.mean_credit, "new": b.mean_credit,
                     "delta": b.mean_credit - a.mean_credit, "verdict_changed": a.verdict is not b.verdict})
    return pd.DataFrame(rows)


class CreditLedger:
    """Append-only JSON-lines history of credit reports with a hash chain, so an edited past is detectable (section 49).
    Each line: {prev, hash, now, code_hash, report}."""

    def __init__(self, path):
        import pathlib
        self.path = pathlib.Path(path)

    def _lines(self) -> list[dict]:
        import json
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def append(self, rep: CreditReport) -> str:
        lines = self._lines()
        prev = lines[-1]["hash"] if lines else ""
        if lines and as_date(rep.now) < as_date(lines[-1]["now"]):
            raise FirewallBreach(f"ledger is chronological: {rep.now} precedes the newest entry")
        body = json_safe(dataclasses.asdict(rep))
        h = stable_hash({"prev": prev, "report": body}, 24)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json({"prev": prev, "hash": h, "now": rep.now, "code_hash": rep.code_hash, "report": body}) + "\n")
        return h

    def verify(self) -> list[str]:
        errs, prev = [], ""
        for i, line in enumerate(self._lines()):
            if line["prev"] != prev:
                errs.append(f"line {i}: broken chain")
            if stable_hash({"prev": line["prev"], "report": line["report"]}, 24) != line["hash"]:
                errs.append(f"line {i}: content does not match its hash (edited after writing)")
            prev = line["hash"]
        return errs


def json_safe(obj):
    """Round-trip through canonical JSON so what is hashed is exactly what is stored (enums, NaN, numpy scalars)."""
    import json
    return json.loads(canonical_json(obj))


# ---------------------------------------------------------------------------------------------- planted calibration (section 63)
def planted_scenarios() -> dict:
    """Truth functions with known answers: name -> (truth(rng, frame), set of components that truly matter)."""
    return {
        "single": (lambda r, d: 0.4 * d.pattern, {"pattern"}),
        "two_independent": (lambda r, d: 0.35 * d.pattern + 0.35 * d.direction, {"pattern", "direction"}),
        "gated": (lambda r, d: 0.5 * d.pattern * (2 * d.timing - 1), {"pattern", "timing"}),   # pattern is WRONG when the gate is closed
        "noise_only": (lambda r, d: 0.0 * d.pattern, set()),
    }


def recovery_score(report: CreditReport, truth: set, ignore: Iterable[str] = ()) -> dict:
    """Precision/recall of EARNS_CREDIT against the planted set. Components in `ignore` (e.g. a gate that only acts through an
    interaction) are excluded from both sides."""
    ign = set(ignore)
    got = set(report.earners()) - ign
    truth = set(truth) - ign
    tp = len(got & truth)
    return {"precision": tp / len(got) if got else 1.0, "recall": tp / len(truth) if truth else 1.0,
            "false_credit": sorted(got - truth), "missed": sorted(truth - got)}


def run_planted_calibration(seed: int = 0, n: int = 700, combiner: Combiner | None = None, cfg: CreditConfig | None = None) -> dict:
    """Run every planted scenario end to end and report recovery. This is the calibration harness later waves tune against -
    it is NOT evidence about real data."""
    combiner = combiner or WeightedSumCombiner({c: 1.0 for c in SIGNAL_COMPONENTS})
    cfg = cfg or CreditConfig(n_boot=150, n_perm=40)
    res = {}
    for name, (fn, truth) in planted_scenarios().items():
        led = simulate_decisions(n, seed, fn, noise=0.5)
        rep = CreditEngine(combiner, cfg).assess(led, "2035-01-01")
        res[name] = recovery_score(rep, truth, ignore=("timing", "risk")) | {"errors": validate_report(rep)}
    return res


# ---------------------------------------------------------------------------------------------- order sensitivity
def sequential_credit(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, order: Sequence[str] | None = None) -> np.ndarray:
    """Credit along the production pipeline order (default: signals, then timing, then risk): each component gets the marginal
    utility it adds ON TOP of everything upstream. Unlike Shapley this is order-dependent by design - it answers "given that
    selection already happened, what did timing add?" (N, n), columns in frame order. Sums to v(all)-v(none) exactly."""
    comps = list(frame.scores.columns)
    order = list(order) if order is not None else [c for c in COMPONENTS if c in comps] + [c for c in comps if c not in COMPONENTS]
    if sorted(order) != sorted(comps):
        raise ValueError("order must be a permutation of the components")
    co = Coalitions(frame, combiner, cfg.utility, cfg.neutral)
    out = np.zeros((frame.n, len(comps)))
    mask, prev = 0, co.v(0)
    for c in order:
        k = comps.index(c)
        mask |= 1 << k
        cur = co.v(mask)
        out[:, k] = cur - prev
        prev = cur
    return out


def order_sensitivity(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, n_orders: int = 24, seed: int = 0) -> pd.DataFrame:
    """Mean, min and max credit of each component over random pipeline orders. A wide [min, max] means the component's credit
    depends on where it sits in the pipeline (strong interactions); Shapley is the mean of this distribution."""
    comps = list(frame.scores.columns)
    rng = np.random.default_rng(seed)
    acc = np.zeros((n_orders, len(comps)))
    for i in range(n_orders):
        acc[i] = sequential_credit(frame, combiner, cfg, [comps[j] for j in rng.permutation(len(comps))]).mean(axis=0)
    return pd.DataFrame({"component": comps, "mean": acc.mean(axis=0), "min": acc.min(axis=0), "max": acc.max(axis=0),
                         "spread": acc.max(axis=0) - acc.min(axis=0)})


# ---------------------------------------------------------------------------------------------- one decision, every counterfactual
def counterfactual_table(decision: Decision, combiner: Combiner, cfg: CreditConfig | None = None) -> pd.DataFrame:
    """Every coalition of components for ONE decision: which were on, the decision value, and the realised utility. This is the
    audit trail behind a single credit number - a reviewer can read it without trusting the aggregate."""
    cfg = cfg or CreditConfig()
    fr = DecisionFrame.build([decision], list(decision.scores), cfg.neutral)
    co = Coalitions(fr, combiner, cfg.utility, cfg.neutral)
    rows = []
    for m in range(1 << co.n_comp):
        on = tuple(c for k, c in enumerate(co.comps) if m >> k & 1)
        rows.append({"active": on, "n_active": len(on), "decision": float(co.decision(m)[0]), "utility": float(co.v(m)[0])})
    return pd.DataFrame(rows).sort_values(["n_active", "utility"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- does credit transfer across contexts?
def context_transfer(frame: DecisionFrame, phi: np.ndarray, dim: str, min_n: int = 20) -> pd.DataFrame:
    """Leave-one-context-out: for each value g of context dimension `dim`, does the sign of a component's mean credit in the
    OTHER contexts predict its sign inside g? Credit that only holds where it was measured has not been shown to transfer
    (section 26-27); `transfers` is None when g has too few decisions."""
    if dim not in frame.context.columns:
        raise KeyError(dim)
    comps = list(frame.scores.columns)
    lab = frame.context[dim].to_numpy(dtype=object)
    rows = []
    for g in sorted(set(lab)):
        inside = lab == g
        if inside.sum() < min_n or (~inside).sum() < min_n:
            rows += [{"context": g, "component": c, "outside": np.nan, "inside": np.nan, "transfers": None} for c in comps]
            continue
        for k, c in enumerate(comps):
            o, i = float(phi[~inside, k].mean()), float(phi[inside, k].mean())
            rows.append({"context": g, "component": c, "outside": o, "inside": i,
                         "transfers": None if abs(o) < 1e-12 else bool(np.sign(o) == np.sign(i))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- credit -> belief updates
class UpdateAction(str, enum.Enum):
    REINFORCE = "REINFORCE"            # reliable positive credit: evidence for the knowledge behind the component
    WEAKEN = "WEAKEN"                  # reliable blame
    HOLD = "HOLD"                      # nothing distinguishable: leave belief alone (no drift on noise)
    INVESTIGATE = "INVESTIGATE"        # unstable, baseline-dependent or interaction-dominated: needs a human/experiment, not a nudge
    INSUFFICIENT = "INSUFFICIENT"


@dataclasses.dataclass(frozen=True)
class UpdateProposal:
    """What credit assignment asks the belief-update stage to do. It carries EVIDENCE (mean credit, interval, stability) and a
    bounded direction - never a raw win/loss - so an update cannot be driven by a recent P&L run alone (contract 1.2)."""
    target: str                        # component name or knowledge id
    level: str                         # 'component' or 'knowledge'
    action: UpdateAction
    evidence: float                    # mean credit
    lo: float
    hi: float
    n: int
    context: str = "all"
    rationale: str = ""
    max_step: float = 0.0              # the largest belief change this evidence can justify, in [0, 1]

    def validate(self) -> list[str]:
        errs = []
        if not 0.0 <= self.max_step <= 1.0:
            errs.append(f"{self.target}: max_step {self.max_step} outside [0,1]")
        if self.action in (UpdateAction.HOLD, UpdateAction.INSUFFICIENT, UpdateAction.INVESTIGATE) and self.max_step != 0.0:
            errs.append(f"{self.target}: {self.action.value} must not move belief")
        if self.action is UpdateAction.REINFORCE and not self.lo > 0:
            errs.append(f"{self.target}: REINFORCE without an interval above zero")
        if self.action is UpdateAction.WEAKEN and not self.hi < 0:
            errs.append(f"{self.target}: WEAKEN without an interval below zero")
        return errs


def update_proposals(report: CreditReport, knowledge: Sequence[KnowledgeCredit] = (), step_cap: float = 0.25) -> list[UpdateProposal]:
    """Translate a report into bounded belief-update proposals. The step is proportional to evidence strength (how far the
    interval sits from zero relative to its width) and capped, so one strong-looking sample cannot swing a belief."""
    out = []
    for c in report.components:
        scale = max(abs(c.lo), abs(c.hi), 1e-12)
        strength = float(min(1.0, min(abs(c.lo), abs(c.hi)) / scale)) if c.lo * c.hi > 0 else 0.0
        if c.verdict is CreditVerdict.EARNS_CREDIT:
            out.append(UpdateProposal(c.component, "component", UpdateAction.REINFORCE, c.mean_credit, c.lo, c.hi, c.n, "all",
                                      "positive credit, stable, beats shuffled input", round(step_cap * strength, 6)))
        elif c.verdict is CreditVerdict.BLAMED:
            out.append(UpdateProposal(c.component, "component", UpdateAction.WEAKEN, c.mean_credit, c.lo, c.hi, c.n, "all",
                                      "reliable negative credit", round(step_cap * strength, 6)))
        elif c.verdict is CreditVerdict.UNSTABLE:
            out.append(UpdateProposal(c.component, "component", UpdateAction.INVESTIGATE, c.mean_credit, c.lo, c.hi, c.n, "all",
                                      "; ".join(c.reasons) or "unstable"))
        elif c.verdict is CreditVerdict.INSUFFICIENT:
            out.append(UpdateProposal(c.component, "component", UpdateAction.INSUFFICIENT, c.mean_credit, c.lo, c.hi, c.n, "all",
                                      "; ".join(c.reasons)))
        else:
            out.append(UpdateProposal(c.component, "component", UpdateAction.HOLD, c.mean_credit, c.lo, c.hi, c.n, "all",
                                      "no distinguishable effect"))
    for k in knowledge:
        act = {CreditVerdict.EARNS_CREDIT: UpdateAction.REINFORCE, CreditVerdict.BLAMED: UpdateAction.WEAKEN,
               CreditVerdict.INSUFFICIENT: UpdateAction.INSUFFICIENT}.get(k.verdict, UpdateAction.HOLD)
        scale = max(abs(k.lo), abs(k.hi), 1e-12) if np.isfinite(k.lo) else 1.0
        step = round(step_cap * min(1.0, min(abs(k.lo), abs(k.hi)) / scale), 6) if act in (UpdateAction.REINFORCE, UpdateAction.WEAKEN) else 0.0
        out.append(UpdateProposal(k.knowledge_id, "knowledge", act, k.mean_credit, k.lo, k.hi, k.n, "all",
                                  f"credit through the {k.component} component", step))
        for ctx, v in sorted(k.by_context.items()):
            out.append(UpdateProposal(k.knowledge_id, "knowledge", UpdateAction.INVESTIGATE if (v * k.mean_credit < 0) else UpdateAction.HOLD,
                                      v, float("nan"), float("nan"), k.n, ctx,
                                      "credit sign differs in this context: candidate anti-context" if v * k.mean_credit < 0
                                      else "consistent with the pooled credit"))
    for u in out:
        errs = u.validate()
        if errs:
            raise ValueError("; ".join(errs))
    return out


def component_confidence(cc: ComponentCredit):
    """Map a component's credit record onto the contract's separate confidence dimensions (section 11/33), leaving what credit
    cannot speak to as None = UNTESTED rather than 0. usefulness = share of earned credit; current_reliability = sign stability;
    failure_risk = how often the component is blamed on losses."""
    from engine.learning.core import Confidence
    useful = cc.share if cc.verdict is CreditVerdict.EARNS_CREDIT else 0.0
    blamed = float(min(1.0, max(0.0, -cc.blame_on_losses / max(abs(cc.credit_on_wins), abs(cc.blame_on_losses), 1e-12)))) \
        if np.isfinite(cc.blame_on_losses) and np.isfinite(cc.credit_on_wins) and cc.blame_on_losses < 0 else 0.0
    conf = Confidence(truth=None, usefulness=float(useful), current_reliability=float(cc.sign_stability), failure_risk=blamed)
    errs = conf.check()
    if errs:
        raise ValueError("; ".join(errs))
    return conf


# ---------------------------------------------------------------------------------------------- hand-off to redundancy
@dataclasses.dataclass(frozen=True)
class MaskedPair:
    a: str
    b: str
    score_corr: float                  # rank correlation of the two components' scores across decisions
    solo_a: float
    solo_b: float
    loo_a: float
    loo_b: float


def masked_pairs(frame: DecisionFrame, raw: Mapping, min_corr: float = 0.7, tol: float = 0.25) -> list[MaskedPair]:
    """Component pairs that look free to drop one at a time (LOO ~ 0) yet each carries clear solo credit and their scores move
    together: the signature of redundancy hiding behind leave-one-out. These are the pairs redundancy.py should test with its
    five-kind analysis before anyone removes either."""
    comps = list(frame.scores.columns)
    hidden = redundancy_masking(raw["loo"], raw["phi"], raw["solo"], tol)
    out = []
    for i, j in itertools.combinations(hidden, 2):
        r = _spearman(frame.scores.iloc[:, i].to_numpy(), frame.scores.iloc[:, j].to_numpy())
        if np.isfinite(r) and abs(r) >= min_corr:
            out.append(MaskedPair(comps[i], comps[j], float(r), float(raw["solo"][:, i].mean()), float(raw["solo"][:, j].mean()),
                                  float(raw["loo"][:, i].mean()), float(raw["loo"][:, j].mean())))
    return out


def subsample_stability(frame: DecisionFrame, combiner: Combiner, cfg: CreditConfig, frac: float = 0.5, n_rep: int = 20,
                        seed: int = 0) -> pd.DataFrame:
    """Recompute mean Shapley credit on random half-samples of WEEKS (not rows): the share of subsamples in which each
    component keeps the sign and top-rank it has on the full sample. Complements the bootstrap, which resamples with
    replacement and so never drops a week."""
    comps = list(frame.scores.columns)
    co_full = Coalitions(frame, combiner, cfg.utility, cfg.neutral)
    phi = shapley_values(co_full, cfg, np.random.default_rng(seed))[0]
    full_mean = phi.mean(axis=0)
    weeks = np.unique(frame.weeks)
    rng = np.random.default_rng(seed + 1)
    same_sign = np.zeros(len(comps))
    top = np.zeros(len(comps))
    for _ in range(n_rep):
        pick = rng.choice(weeks, size=max(2, int(len(weeks) * frac)), replace=False)
        m = np.isin(frame.weeks, pick)
        mean = phi[m].mean(axis=0)
        same_sign += (np.sign(mean) == np.sign(full_mean)) & (full_mean != 0)
        top[int(np.argmax(mean))] += 1
    return pd.DataFrame({"component": comps, "full_mean": full_mean, "same_sign": same_sign / n_rep, "top_rank": top / n_rep})


# ---------------------------------------------------------------------------------------------- reports
def context_table(rep: CreditReport) -> pd.DataFrame:
    """Long table (dimension, value, component, mean_credit, dependence_p) from a report - what a dashboard or scorecard reads."""
    rows = []
    for dim, groups in rep.context_credit.items():
        for val, comps in groups.items():
            for comp, v in comps.items():
                rows.append({"dimension": dim, "value": val, "component": comp, "mean_credit": v,
                             "dependence_p": rep.context_dependence.get(dim, {}).get(comp, float("nan"))})
    return pd.DataFrame(rows, columns=["dimension", "value", "component", "mean_credit", "dependence_p"])


def summary_dict(rep: CreditReport) -> dict:
    """Flat JSON-friendly summary for the learning scorecard: who earns credit, how much of the effect is unowned, whether the
    machinery is sound. Every value is derived from the report; nothing is typed by hand (prose goes stale)."""
    return {"now": rep.now, "label": rep.label(), "n_decisions": rep.n_decisions, "n_pending": rep.n_pending,
            "earners": rep.earners(), "blamed": [c.component for c in rep.components if c.verdict is CreditVerdict.BLAMED],
            "unstable": [c.component for c in rep.components if c.verdict is CreditVerdict.UNSTABLE],
            "insufficient": [c.component for c in rep.components if c.verdict is CreditVerdict.INSUFFICIENT],
            "unattributed_fraction": rep.unattributed_fraction, "blanket": rep.blanket,
            "efficiency_error": rep.efficiency_error, "code_hash": rep.code_hash,
            "reliable_interactions": [(i.a, i.b, i.kind) for i in rep.interactions if i.verdict == "RELIABLE"],
            "problems": validate_report(rep)}


def render_markdown(rep: CreditReport) -> str:
    """Human-readable report with the honest label at the top, one table per view, and every non-credit reason spelled out."""
    L = [f"# Credit assignment @ {rep.now}", "", f"**{rep.label()}** - {rep.n_decisions} matured decisions, "
         f"{rep.n_pending} pending outcomes not used.", "",
         f"Do-nothing value {rep.base_value:+.5f}; full value {rep.full_value:+.5f}; unattributed "
         f"{rep.unattributed_fraction:.1%}; efficiency error {rep.efficiency_error:.1e}.", "",
         "| component | credit | 95% CI | LOO | solo | share | stability | q | ablation | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(rep.components, key=lambda c: -c.mean_credit):
        L.append(f"| {c.component} | {c.mean_credit:+.5f} | [{c.lo:+.5f}, {c.hi:+.5f}] | {c.loo:+.5f} | {c.solo:+.5f} | "
                 f"{c.share:.1%} | {c.sign_stability:.2f} | {c.null_p:.3f} | {c.ablate_verdict} | {c.verdict.value} |")
    why = [(c.component, r) for c in rep.components for r in c.reasons]
    if why:
        L += ["", "Why some components received no credit:"] + [f"- {n}: {r}" for n, r in why]
    rel = [i for i in rep.interactions if i.verdict == "RELIABLE"]
    if rel:
        L += ["", "Reliable interactions:"] + [f"- {i.a} x {i.b}: {i.kind.lower()} {i.mean:+.5f} (q={i.q:.3f})" for i in rel]
    if rep.blanket:
        L += ["", "> WARNING: credit is spread almost equally across components; treat it as blanket credit, not attribution."]
    probs = validate_report(rep)
    L += ["", "Integrity checks: " + ("all passed." if not probs else "; ".join(probs))]
    return "\n".join(L)


def rolling_assessments(engine: CreditEngine, ledger: DecisionLedger, nows: Sequence) -> list[CreditReport]:
    """Assess the same ledger at a sequence of `now` dates (strictly increasing). Each report sees only outcomes matured before
    ITS OWN now, so the sequence is exactly what the system would have known at each step - the input to credit trends."""
    dates = [as_date(n) for n in nows]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("nows must be strictly increasing")
    return [engine.assess(ledger, n) for n in nows]


def earners_over_time(reports: Sequence[CreditReport]) -> pd.DataFrame:
    """Which components held EARNS_CREDIT at each `now`: rows = now, columns = component, values = mean credit if it earned else NaN.
    A component that appears and disappears is the learning system's first hint that its knowledge is regime-bound."""
    rows = []
    for r in reports:
        row = {"now": r.now}
        row.update({c.component: (c.mean_credit if c.verdict is CreditVerdict.EARNS_CREDIT else float("nan")) for c in r.components})
        rows.append(row)
    return pd.DataFrame(rows).set_index("now") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------------------------- where does blame concentrate?
def blame_by_context(frame: DecisionFrame, phi: np.ndarray, dim: str, min_losses: int = 10) -> pd.DataFrame:
    """Failure learning input (section 24): among LOSING decisions only, each component's mean credit (negative = blame) inside
    each value of context dimension `dim`, and `lift` = that context's share of all blame on the component divided by its share
    of decisions. lift >> 1 means the component fails disproportionately there - a candidate anti-context."""
    if dim not in frame.context.columns:
        raise KeyError(dim)
    comps = list(frame.scores.columns)
    lab = frame.context[dim].to_numpy(dtype=object)
    net = phi.sum(axis=1)
    lose = net < 0
    rows = []
    for k, c in enumerate(comps):
        blame = np.where(lose, np.minimum(phi[:, k], 0.0), 0.0)
        total = blame.sum()
        for g in sorted(set(lab)):
            m = lab == g
            n_lose = int((m & lose).sum())
            if n_lose < min_losses:
                continue
            share_blame = float(blame[m].sum() / total) if total < -1e-15 else float("nan")
            rows.append({"component": c, "context": g, "n_losses": n_lose, "mean_blame": float(phi[m & lose, k].mean()),
                         "share_of_blame": share_blame, "share_of_decisions": float(m.mean()),
                         "lift": share_blame / float(m.mean()) if np.isfinite(share_blame) else float("nan")})
    return pd.DataFrame(rows, columns=["component", "context", "n_losses", "mean_blame", "share_of_blame", "share_of_decisions", "lift"])


def anti_context_candidates(blame: pd.DataFrame, min_lift: float = 1.5) -> list[tuple[str, str, float]]:
    """(component, context, lift) where blame is concentrated at least `min_lift` times beyond the context's size. Candidates
    only: the lift is descriptive until a held-out window confirms it."""
    if blame.empty:
        return []
    hit = blame[(blame["lift"] >= min_lift) & (blame["mean_blame"] < 0)].sort_values("lift", ascending=False)
    return [(r.component, r.context, float(r.lift)) for r in hit.itertuples()]
