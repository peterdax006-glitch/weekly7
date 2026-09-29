"""Context modelling (contract C62 section 8; canon C58-C61, C63).

Knowledge is never `pattern -> result`. It is `pattern + context + outcome`, and BOTH sides are learned:
    P(outcome | pattern, context)   and   P(outcome | pattern, NOT context).
Without the second, a pattern is applied outside the conditions where it works.

Mechanisms:
* a small condition language (`Condition`, `ContextSpec`) over the same dimensions and buckets as engine.learning.situation,
  with three-valued matching (True / False / None = a needed dimension is unobserved -> UNKNOWN, never guessed);
* hierarchical pooling (`ContextModel.estimate`): walk the POOLING_LADDER from the pattern-wide mean down to the most
  specific bucket, shrinking each level toward its parent with an empirical-Bayes strength k_r (method of moments), and STOP
  at the first bucket with fewer than `min_n` observations - no tiny buckets ever speak for themselves;
* context discovery (`discover`): every (dimension, bucket-set) split is scored by Welch t, family-wise error is controlled
  with a Westfall-Young max-|t| permutation test (seeded), rules found on the earlier part of the data must be CONFIRMED on the
  later part (same sign, own t), two-condition conjunctions are only proposed from the best singles and need a
  Bonferroni-corrected holdout confirmation;
* each rule records in-context AND out-of-context statistics, its role (CONTEXT = where the pattern works, ANTI_CONTEXT =
  where it fails) and maps onto the KnowledgeLike `contexts` / `anti_contexts` fields of contract section 5;
* `pooling_cv` measures out-of-sample whether pooling helped, `drift_check` re-tests a rule on later data.

Outcomes are supplied as signed edge (direction already applied). Every fit takes `now` and fails closed (FirewallBreach)
on an observation whose outcome had not matured strictly before it.

Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats as sps

from .core import Epistemic, Layer, Unknown, as_date, require_past, stable_hash
from .. import pattern_stats as PS          # reused: eb_shrink / implied_k / permute_within_clusters / t_to_p / bonferroni / week_codes
from .situation import POOLING_LADDER, Situation, ladder_key, spec_of

PATTERN_HAS = "pattern.has:"                 # virtual dimension: is another pattern co-active (pattern interaction)


# ------------------------------------------------------------------------------------------------ conditions

def _labels_of(sit: Situation) -> dict[str, str]:
    """Bucket labels of every dimension plus virtual pattern co-activity flags for the patterns active now."""
    b = dict(sit.bins())
    for pid in sit.pattern_ids:
        b[PATTERN_HAS + pid] = "yes"
    return b


def _label(sit: Situation, path: str) -> str:
    if path.startswith(PATTERN_HAS):
        return "yes" if path[len(PATTERN_HAS):] in sit.pattern_ids else "no"
    return sit.bins().get(path, "na")


def ordered_labels(path: str) -> tuple[str, ...] | None:
    """The ordered bucket labels of a dimension, or None for an unordered (categorical / virtual) one."""
    if path.startswith(PATTERN_HAS):
        return None
    s = spec_of(path)
    if s.kind == "num":
        return tuple(f"b{i}" for i in range(len(s.bins) + 1))
    if s.kind == "ord":
        return s.levels
    return None


@dataclasses.dataclass(frozen=True)
class Condition:
    """`path` falls in one of `allowed` buckets. negate=True is the complement (still False/True only when observed)."""
    path: str
    allowed: tuple[str, ...]
    negate: bool = False

    def __post_init__(self):
        if not self.allowed:
            raise ValueError("Condition needs at least one allowed bucket")
        if list(self.allowed) != sorted(set(self.allowed)):
            raise ValueError("allowed buckets must be sorted and unique")
        if not self.path.startswith(PATTERN_HAS):
            spec_of(self.path)                               # raises KeyError on an unknown dimension

    def holds(self, sit: Situation) -> bool | None:
        lab = _label(sit, self.path)
        if lab == "na":
            return None
        return (lab in self.allowed) != self.negate

    def flipped(self) -> "Condition":
        return dataclasses.replace(self, negate=not self.negate)

    def describe(self) -> str:
        body = self.allowed[0] if len(self.allowed) == 1 else "{" + ",".join(self.allowed) + "}"
        return f"{self.path} {'NOT in' if self.negate else 'in'} {body}"

    def to_mapping(self) -> dict:
        return {"in": list(self.allowed), **({"not": True} if self.negate else {})}

    @classmethod
    def from_mapping(cls, path: str, m: Any) -> "Condition":
        """Accepts {'in': [...]}, {'in': [...], 'not': True}, a bare list, or a single label."""
        if isinstance(m, Mapping):
            allowed = m.get("in")
            if allowed is None:
                raise ValueError(f"condition for {path!r} has no 'in' list: {m!r}")
            return cls(path, tuple(sorted({str(a) for a in allowed})), bool(m.get("not", False)))
        if isinstance(m, (list, tuple, set, frozenset)):
            return cls(path, tuple(sorted({str(a) for a in m})))
        return cls(path, (str(m),))


def _universe(path: str) -> tuple[str, ...]:
    if path.startswith(PATTERN_HAS):
        return ("no", "yes")
    lab = ordered_labels(path)
    if lab is None:
        return tuple(spec_of(path).levels)
    return lab


def _bucket_span(path: str, label: str) -> tuple[float, float]:
    """[lo, hi) of a numeric bucket label 'b<i>' (bucket i holds bins[i-1] <= v < bins[i])."""
    bins = spec_of(path).bins
    i = int(label[1:])
    return (bins[i - 1] if i > 0 else -math.inf), (bins[i] if i < len(bins) else math.inf)


def _accepted_buckets(c: Any) -> set[str]:
    """Buckets (on the feature's own axis) that a typed knowledge.Condition accepts."""
    path = c.feature
    uni = _universe(path)
    op = c.op
    if op in ("eq", "in", "ne", "not_in"):
        if c.labels:
            hit = {str(x) for x in c.labels}
        else:
            spec = spec_of(path)
            if spec.kind != "num":
                raise ValueError(f"{path}: numeric values on a non-numeric dimension")
            hit = {spec.bin_of(x) for x in c.nums}
        bad = hit - set(uni)
        if bad:
            raise ValueError(f"{path}: unknown buckets {sorted(bad)}; expected some of {list(uni)}")
        return set(uni) - hit if op in ("ne", "not_in") else hit
    if spec_of(path).kind != "num":
        raise ValueError(f"{path}: numeric bound {op} on a non-numeric dimension")
    lo_b, hi_b = {"lt": (-math.inf, c.nums[0] if c.nums else math.nan), "le": (-math.inf, c.nums[0] if c.nums else math.nan),
                  "gt": (c.nums[0] if c.nums else math.nan, math.inf), "ge": (c.nums[0] if c.nums else math.nan, math.inf),
                  "between": (c.nums[0] if c.nums else math.nan, c.nums[-1] if c.nums else math.nan)}[op]
    return {b for b in uni if lo_b <= _bucket_span(path, b)[0] and _bucket_span(path, b)[1] <= hi_b}


@dataclasses.dataclass(frozen=True)
class ContextSpec:
    """Conjunction of conditions on distinct dimensions. The empty spec matches everything."""
    conditions: tuple[Condition, ...] = ()

    def __post_init__(self):
        paths = [c.path for c in self.conditions]
        if len(set(paths)) != len(paths):
            raise ValueError("a ContextSpec may hold one condition per dimension")
        if paths != sorted(paths):
            object.__setattr__(self, "conditions", tuple(sorted(self.conditions, key=lambda c: c.path)))

    def matches(self, sit: Situation) -> bool | None:
        """True if every condition holds, False if any fails, None if none fails but some dimension is unobserved."""
        unknown = False
        for c in self.conditions:
            h = c.holds(sit)
            if h is False:
                return False
            unknown |= h is None
        return None if unknown else True

    def matches_not(self, sit: Situation) -> bool | None:
        m = self.matches(sit)
        return None if m is None else not m

    @property
    def spec_id(self) -> str:
        return stable_hash([c.to_mapping() | {"path": c.path} for c in self.conditions])

    @property
    def complexity(self) -> int:
        return len(self.conditions)

    def describe(self) -> str:
        return " AND ".join(c.describe() for c in self.conditions) or "always"

    def to_mapping(self) -> dict[str, dict]:
        return {c.path: c.to_mapping() for c in self.conditions}

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> "ContextSpec":
        return cls(tuple(Condition.from_mapping(p, v) for p, v in sorted((m or {}).items())))

    @classmethod
    def from_typed(cls, cs: Any) -> "ContextSpec":
        """Bucket spec of a typed knowledge.ContextSet (Condition objects with dimension / feature / op / nums / labels).
        Every typed condition is first expanded to the set of buckets it accepts on its feature's own axis, then the conditions on
        one feature are intersected (all_of) or united (any_of); a typed set that cannot be a conjunction of bucket conditions
        (any_of over several features) or that accepts nothing raises ValueError instead of being silently widened.
        Numeric bounds accept a bucket only when the WHOLE bucket lies inside the bound, so a typed context never matches more
        than it says."""
        conds = tuple(getattr(cs, "conditions", ()) or ())
        any_of = bool(getattr(cs, "any_of", False))
        by: dict[str, set[str]] = {}
        for c in conds:
            acc = _accepted_buckets(c)
            if c.feature in by:
                by[c.feature] = (by[c.feature] | acc) if any_of else (by[c.feature] & acc)
            else:
                by[c.feature] = acc
        if any_of and len(by) > 1:
            raise ValueError("an any_of context over several features is a disjunction: it has no conjunctive bucket form")
        out = []
        for path, acc in sorted(by.items()):
            if not acc:
                raise ValueError(f"typed context on {path!r} accepts no bucket")
            out.append(Condition(path, tuple(sorted(acc))))
        return cls(tuple(out))

    def with_condition(self, c: Condition) -> "ContextSpec":
        return ContextSpec(tuple(x for x in self.conditions if x.path != c.path) + (c,))


# ------------------------------------------------------------------------------------------------ statistics

@dataclasses.dataclass(frozen=True)
class CellStats:
    n: int
    n_eff: float
    mean: float | None
    var: float | None
    hit: float | None                       # share of outcomes above zero
    reliable: bool                          # n >= the model's min_n

    @classmethod
    def of(cls, y: np.ndarray, w: np.ndarray | None, min_n: int) -> "CellStats":
        n = int(len(y))
        if n == 0:
            return cls(0, 0.0, None, None, None, False)
        w = np.ones(n) if w is None else np.asarray(w, dtype=float)
        sw = float(w.sum())
        m = float((w * y).sum() / sw)
        neff = sw * sw / float((w * w).sum())
        var = float((w * (y - m) ** 2).sum() / sw) * (n / (n - 1)) if n > 1 else None
        return cls(n, round(neff, 4), m, var, float((w * (y > 0)).sum() / sw), n >= min_n)


def welch_t(m1, v1, n1, m0, v0, n0) -> float:
    """Welch t for the difference of two means, 0.0 where a variance or count makes it undefined (never NaN/inf)."""
    if n1 < 2 or n0 < 2 or v1 is None or v0 is None:
        return 0.0
    se = math.sqrt(max(v1, 0.0) / n1 + max(v0, 0.0) / n0)
    return 0.0 if se <= 1e-15 else float((m1 - m0) / se)


def beta_interval(hits: float, n: float, prior_mean: float = 0.5, prior_strength: float = 4.0, level: float = 0.9):
    """(posterior mean, lo, hi) of a hit rate with a Beta prior centred on `prior_mean`."""
    a0, b0 = prior_mean * prior_strength, (1 - prior_mean) * prior_strength
    a, b = a0 + hits, b0 + max(n - hits, 0.0)
    lo, hi = sps.beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a, b)
    return float(a / (a + b)), float(lo), float(hi)


@dataclasses.dataclass(frozen=True)
class ContextConfig:
    min_n: int = 30                     # smallest bucket allowed to speak (both in and out of a context)
    prior_strength: float = 20.0        # pseudo-observations pulling the pattern-wide mean toward zero edge
    k_min: float = 5.0                  # bounds on the per-rung shrinkage strength
    k_max: float = 400.0
    rule_shrink: float = 10.0           # shrinkage of a rule side toward the pattern-wide estimate
    alpha: float = 0.05                 # family-wise level for the max-T test
    n_perm: int = 300
    holdout_frac: float = 0.3           # latest share of observations kept to confirm rules
    min_effect: float = 0.0             # smallest |in - out| difference worth a rule (outcome units)
    max_rules: int = 6
    conj_top: int = 8                   # best singles considered for two-condition conjunctions
    half_life_days: float | None = None # recency weighting of the ladder estimate (None = equal weights)
    max_labels_per_dim: int = 40        # guard against a free-text dimension exploding the candidate set
    winsor: float | None = None         # clip outcomes to this two-sided quantile (returns are heavy-tailed; None = as given)

    def validate(self) -> list[str]:
        errs = []
        if self.min_n < 10:
            errs.append("min_n < 10 permits tiny buckets")
        if not 0.0 <= self.holdout_frac < 0.6:
            errs.append("holdout_frac outside [0, 0.6)")
        if not 0.0 < self.alpha < 1.0:
            errs.append("alpha outside (0, 1)")
        if self.k_min <= 0 or self.k_max < self.k_min:
            errs.append("bad k bounds")
        if self.n_perm < 50:
            errs.append("n_perm < 50 cannot resolve alpha")
        if self.half_life_days is not None and self.half_life_days <= 0:
            errs.append("half_life_days must be positive")
        if self.winsor is not None and not 0.0 < self.winsor < 0.25:
            errs.append("winsor must be in (0, 0.25)")
        return errs


def min_detectable_effect(n_in: float, n_out: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest |mean_in - mean_out| a two-sample test at level `alpha` detects with probability `power`. Reported next to
    every 'no context found' so an absence is read as 'not detectable at this effect size', never as 'no context'."""
    if n_in < 2 or n_out < 2 or sd <= 0:
        return float("inf")
    z = sps.norm.ppf(1 - min(max(alpha, 1e-12), 0.5) / 2) + sps.norm.ppf(power)
    return float(z * sd * math.sqrt(1.0 / n_in + 1.0 / n_out))


@dataclasses.dataclass(frozen=True)
class DiscoveryReport:
    """What one `discover` call could and could not see."""
    pattern_id: str
    n_obs: int
    n_train: int
    n_holdout: int
    candidates: int
    mde: float | None                   # minimum detectable effect at the median candidate size (Bonferroni-level alpha)
    rules: int
    reason: str
    unknown: Unknown | None


@dataclasses.dataclass(frozen=True)
class Obs:
    """One matured outcome of one pattern in one situation. `matured` is the (trusted-side) date the outcome became known."""
    pattern_id: str
    situation: Situation
    outcome: float                      # signed edge, direction applied
    matured: Any
    weight: float = 1.0

    def __post_init__(self):
        if not math.isfinite(float(self.outcome)):
            raise ValueError("outcome must be finite")
        if not float(self.weight) > 0:
            raise ValueError("weight must be positive")


@dataclasses.dataclass(frozen=True)
class ContextRule:
    """A learned condition under which a pattern behaves differently (layer L6). `stats_in` and `stats_out` are BOTH kept."""
    pattern_id: str
    spec: ContextSpec
    role: str                                   # CONTEXT (works here) | ANTI_CONTEXT (fails here)
    stats_in: CellStats
    stats_out: CellStats
    diff: float                                 # mean_in - mean_out
    t_train: float
    p_adj: float | None                         # max-T family-wise p on the training part (None for conjunctions)
    t_holdout: float | None
    confirmed: bool
    epistemic: Epistemic
    n_train: int
    n_holdout: int
    LAYER = Layer.L6_CONTEXT_RULE

    @property
    def rule_id(self) -> str:
        return "CR-" + stable_hash({"p": self.pattern_id, "s": self.spec.spec_id, "r": self.role}, 12)

    def validate(self) -> list[str]:
        errs = []
        if self.role not in ("CONTEXT", "ANTI_CONTEXT"):
            errs.append(f"role {self.role!r}")
        if self.stats_in.n == 0 or self.stats_out.n == 0:
            errs.append("a rule needs observations both inside and outside its context")
        if self.role == "CONTEXT" and self.diff < 0 or self.role == "ANTI_CONTEXT" and self.diff > 0:
            errs.append("role contradicts the sign of the in-out difference")
        if self.confirmed and self.t_holdout is None:
            errs.append("confirmed without a holdout statistic")
        return errs

    def to_knowledge_fields(self) -> dict[str, dict]:
        """The {contexts, anti_contexts} pair of a KnowledgeLike object (contract section 5)."""
        m = self.spec.to_mapping()
        return {"contexts": m, "anti_contexts": {}} if self.role == "CONTEXT" else {"contexts": {}, "anti_contexts": m}

    def describe(self) -> str:
        tag = "works when" if self.role == "CONTEXT" else "FAILS when"
        mi, mo = self.stats_in.mean, self.stats_out.mean
        return (f"{self.pattern_id} {tag} {self.spec.describe()}: in {mi:+.4f} (n={self.stats_in.n}) "
                f"vs out {mo:+.4f} (n={self.stats_out.n}); holdout t={self.t_holdout if self.t_holdout is None else round(self.t_holdout, 2)}"
                f" {'CONFIRMED' if self.confirmed else 'unconfirmed'}")


@dataclasses.dataclass(frozen=True)
class PoolStep:
    rung: int
    key: tuple[str, ...]
    n: int
    raw_mean: float | None
    shrunk_mean: float
    k: float


@dataclasses.dataclass(frozen=True)
class ContextEstimate:
    pattern_id: str
    expected: float | None                      # expected signed edge in this situation
    se: float | None
    p_hit: float | None
    source: str                                 # rule | ladder | prior | none
    unknown: Unknown | None
    trail: tuple[PoolStep, ...]
    pooled_to: int                              # deepest ladder rung that was allowed to speak
    matched_in: tuple[str, ...]                 # rule ids whose context holds now
    matched_out: tuple[str, ...]                # rule ids whose context does NOT hold now
    unknown_rules: tuple[str, ...]              # rules that cannot be evaluated (dimension unobserved)
    n_support: int
    explanation: str


# ------------------------------------------------------------------------------------------------ per-pattern index

class _Acc:
    __slots__ = ("n", "sw", "s1", "s2", "hits")

    def __init__(self):
        self.n, self.sw, self.s1, self.s2, self.hits = 0, 0.0, 0.0, 0.0, 0.0

    def add(self, y, w):
        self.n += 1
        self.sw += w
        self.s1 += w * y
        self.s2 += w * y * y
        self.hits += w * (y > 0)

    @property
    def mean(self):
        return self.s1 / self.sw if self.sw > 0 else None


class _Index:
    """Everything derived from one pattern's observations: arrays, ladder buckets and shrinkage strengths."""

    def __init__(self, obs: Sequence[Obs], now, cfg: ContextConfig):
        self.n = len(obs)
        order = sorted(range(self.n), key=lambda i: (as_date(obs[i].matured), i))
        self.obs = [obs[i] for i in order]
        self.y = np.array([o.outcome for o in self.obs], dtype=float)
        if cfg.winsor is not None and self.n >= 20:
            lo, hi = np.quantile(self.y, [cfg.winsor, 1 - cfg.winsor])
            self.y = np.clip(self.y, lo, hi)                 # one -70% gap must not decide a context
        self.day = np.array([as_date(o.matured).toordinal() for o in self.obs], dtype=float)
        self.codes, _ = PS.week_codes(pd.DatetimeIndex([pd.Timestamp(as_date(o.matured)) for o in self.obs]))
        now_o = as_date(now).toordinal()
        if cfg.half_life_days:
            self.w = np.array([o.weight * 0.5 ** ((now_o - d) / cfg.half_life_days) for o, d in zip(self.obs, self.day)])
        else:
            self.w = np.array([o.weight for o in self.obs], dtype=float)
        self.var = float(np.var(self.y, ddof=1)) if self.n > 1 else 0.0
        self.labels = [_labels_of(o.situation) for o in self.obs]
        self.rung_acc: list[dict[tuple, _Acc]] = []
        for r in range(len(POOLING_LADDER)):
            d: dict[tuple, _Acc] = {}
            for o, y, w in zip(self.obs, self.y, self.w):
                d.setdefault(ladder_key(o.situation, r), _Acc()).add(float(y), float(w))
            self.rung_acc.append(d)
        self.k = [self._rung_k(r, cfg) for r in range(len(POOLING_LADDER))]

    def _rung_k(self, r: int, cfg: ContextConfig) -> float:
        """Empirical-Bayes shrinkage strength of rung r: sigma^2 / tau^2 from the spread of bucket means."""
        if r == 0:
            return cfg.prior_strength
        cells = [a for a in self.rung_acc[r].values() if a.n >= cfg.min_n and a.sw > 0]
        if len(cells) < 3 or self.var <= 0:
            return cfg.k_max
        means = np.array([a.mean for a in cells])
        ns = np.array([a.n for a in cells], dtype=float)
        ses = np.sqrt(self.var / ns)
        eb = PS.eb_shrink(means, ses, center=None)          # DerSimonian-Laird tau^2 across the bucket means (shared code)
        k = PS.implied_k(eb["tau2"], ses, ns)  # type: ignore[arg-type]  # pattern_stats converts arrays with _as_float
        return cfg.k_max if not math.isfinite(k) else float(np.clip(k, cfg.k_min, cfg.k_max))


# ------------------------------------------------------------------------------------------------ the model

class ContextModel:
    """Learns, per pattern, how outcomes depend on situation context. Add matured observations, then `fit(now)`."""

    def __init__(self, config: ContextConfig | None = None):
        self.cfg = config or ContextConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("ContextConfig invalid: " + "; ".join(errs))
        self._obs: dict[str, list[Obs]] = {}
        self._idx: dict[str, tuple[tuple, _Index]] = {}
        self._rules: dict[str, tuple[ContextRule, ...]] = {}
        self.reports: dict[str, "DiscoveryReport"] = {}

    # ---- data
    def add(self, obs: Obs) -> None:
        errs = obs.situation.validate()
        if errs:
            raise ValueError("observation carries an invalid situation: " + "; ".join(errs[:3]))
        self._obs.setdefault(obs.pattern_id, []).append(obs)
        self._idx.pop(obs.pattern_id, None)
        self._rules.pop(obs.pattern_id, None)

    def add_many(self, obs: Iterable[Obs]) -> int:
        n = 0
        for o in obs:
            self.add(o)
            n += 1
        return n

    def patterns(self) -> tuple[str, ...]:
        return tuple(sorted(self._obs))

    def n_obs(self, pattern_id: str) -> int:
        return len(self._obs.get(pattern_id, ()))

    def _index(self, pattern_id: str, now) -> _Index | None:
        obs = self._obs.get(pattern_id)
        if not obs:
            return None
        for o in obs:
            require_past(o.matured, now, f"context observation for {pattern_id}")
        stamp = (as_date(now), len(obs))
        hit = self._idx.get(pattern_id)
        if hit is None or hit[0] != stamp:
            hit = self._idx[pattern_id] = (stamp, _Index(obs, now, self.cfg))
            self._rules.pop(pattern_id, None)                # rules were learned against another `now` / another data set
        return hit[1]

    # ---- hierarchical pooling
    def ladder_estimate(self, pattern_id: str, sit: Situation, now) -> tuple[float | None, float | None, float | None, tuple[PoolStep, ...], int, int]:
        """(expected, se, p_hit, trail, deepest rung used, n at that rung) by pooling down the ladder."""
        ix = self._index(pattern_id, now)
        if ix is None:
            return None, None, None, (), -1, 0
        cfg = self.cfg
        a0 = ix.rung_acc[0][()]
        k0 = ix.k[0]
        m = a0.s1 / (a0.sw + k0)                       # prior mean 0: an unproven pattern has no edge
        p = (a0.hits + 0.5 * k0) / (a0.sw + k0)
        trail = [PoolStep(0, (), a0.n, a0.mean, float(m), k0)]
        used, n_used, se2 = 0, a0.n, ix.var / (a0.sw + k0)
        for r in range(1, len(POOLING_LADDER)):
            acc = ix.rung_acc[r].get(ladder_key(sit, r))
            if acc is None or acc.n < cfg.min_n:
                break                                     # do not let a tiny bucket speak; stay at the parent
            k = ix.k[r]
            m = (acc.s1 + k * m) / (acc.sw + k)
            p = (acc.hits + k * p) / (acc.sw + k)
            se2 = ix.var / (acc.sw + k)
            trail.append(PoolStep(r, ladder_key(sit, r), acc.n, acc.mean, float(m), k))
            used, n_used = r, acc.n
        return float(m), float(math.sqrt(se2)), float(min(max(p, 0.0), 1.0)), tuple(trail), used, n_used

    # ---- discovery
    def _design(self, ix: _Index, rows: np.ndarray) -> tuple[list[tuple[str, tuple[str, ...]]], np.ndarray, np.ndarray]:
        """Candidate splits over `rows`: (path, bucket-set) list, in-mask matrix I and valid-mask matrix V (rows x C)."""
        cfg = self.cfg
        paths: dict[str, list[str]] = {}
        for i in rows:
            for p, lab in ix.labels[i].items():
                paths.setdefault(p, []).append(lab)
        for i in rows:                                  # virtual pattern flags are absent when the pattern is absent
            for p in paths:
                if p.startswith(PATTERN_HAS) and p not in ix.labels[i]:
                    ix.labels[i][p] = "no"
        cands: list[tuple[str, tuple[str, ...]]] = []
        cols_i, cols_v = [], []
        for p in sorted(paths):
            labs = np.array([ix.labels[i].get(p, "na") for i in rows], dtype=object)
            valid = labs != "na"
            uniq = sorted(set(labs[valid]))
            if len(uniq) < 2 or len(uniq) > cfg.max_labels_per_dim:
                continue
            order = ordered_labels(p)
            sets: list[tuple[str, ...]] = [(u,) for u in uniq]
            if order is not None:                        # ordered dimension: also every "<= k" prefix (its complement is ">")
                present = [o for o in order if o in uniq]
                sets += [tuple(sorted(present[: k + 1])) for k in range(1, len(present) - 1)]
            for s in dict.fromkeys(sets):
                inm = np.isin(labs, s) & valid
                nin, nout = int(inm.sum()), int((valid & ~inm).sum())
                if nin < cfg.min_n or nout < cfg.min_n:
                    continue                             # both sides must be big enough: no tiny buckets
                cands.append((p, s))
                cols_i.append(inm)
                cols_v.append(valid)
        if not cands:
            return [], np.zeros((len(rows), 0)), np.zeros((len(rows), 0))
        return cands, np.column_stack(cols_i).astype(float), np.column_stack(cols_v).astype(float)

    @staticmethod
    def _t_from_sums(I, V, Y):
        """Welch t for every candidate column and every outcome column of Y (rows x P). Returns (C x P)."""
        y2 = Y * Y
        n1 = I.sum(axis=0)[:, None]
        nv = V.sum(axis=0)[:, None]
        n0 = nv - n1
        s1, s2 = I.T @ Y, I.T @ y2
        t1, t2 = V.T @ Y, V.T @ y2
        with np.errstate(all="ignore"):
            m1 = s1 / n1
            m0 = (t1 - s1) / n0
            v1 = (s2 - n1 * m1 ** 2) / (n1 - 1)
            v0 = ((t2 - s2) - n0 * m0 ** 2) / (n0 - 1)
            se = np.sqrt(np.maximum(v1, 0) / n1 + np.maximum(v0, 0) / n0)
            t = (m1 - m0) / se
        return np.where(np.isfinite(t) & (se > 1e-15), t, 0.0), m1, m0

    def discover(self, pattern_id: str, now, seed: int = 0) -> tuple[ContextRule, ...]:
        """Learn the contexts of one pattern. Deterministic for a given seed. Returns [] when the data cannot support any."""
        cfg = self.cfg
        ix = self._index(pattern_id, now)
        if ix is None or ix.n < 4 * cfg.min_n:
            self._rules[pattern_id] = ()
            self.reports[pattern_id] = DiscoveryReport(pattern_id, 0 if ix is None else ix.n, 0, 0, 0, None, 0,
                                                       f"insufficient data: fewer than {4 * cfg.min_n} observations", Unknown.INSUFFICIENT_DATA)
            return ()
        n_hold = int(round(cfg.holdout_frac * ix.n)) if ix.n >= 2 * (2 * cfg.min_n) / max(1 - cfg.holdout_frac, 1e-9) else 0
        train = np.arange(ix.n - n_hold)
        hold = np.arange(ix.n - n_hold, ix.n)
        cands, I, V = self._design(ix, train)
        rules: list[ContextRule] = []
        mde = None
        if cands:
            n_in = float(np.median(I.sum(axis=0)))
            mde = min_detectable_effect(n_in, len(train) - n_in, float(np.std(ix.y[train])), cfg.alpha / max(len(cands), 1))
        if cands:
            y = ix.y[train]
            t_obs, _, _ = self._t_from_sums(I, V, y[:, None])
            t_obs = t_obs[:, 0]
            rng = np.random.default_rng(seed)
            codes = ix.codes[train]                      # a good week stays a good week: permute outcomes inside each week only
            Yp = np.column_stack([PS.permute_within_clusters(y, codes, rng) for _ in range(cfg.n_perm)])
            tp, _, _ = self._t_from_sums(I, V, Yp)
            maxt = np.abs(tp).max(axis=0)
            p_adj = (1 + (maxt[None, :] >= np.abs(t_obs)[:, None]).sum(axis=1)) / (1 + cfg.n_perm)
            rules += self._singles(pattern_id, ix, cands, t_obs, p_adj, train, hold)
            rules += self._conjunctions(pattern_id, ix, cands, t_obs, train, hold)
        rules.sort(key=lambda r: (not r.confirmed, r.p_adj if r.p_adj is not None else 1.0, -abs(r.diff), r.rule_id))
        rules = self._dedupe(rules)[: cfg.max_rules]
        self._rules[pattern_id] = tuple(rules)
        reason = "" if rules else ("no split has at least min_n observations on both sides" if not cands else
                                   "no split survives the family-wise test: the data cannot show a context (this is not proof there is none)")
        self.reports[pattern_id] = DiscoveryReport(pattern_id, ix.n, len(train), len(hold), len(cands), mde, len(rules), reason,
                                                   None if rules else Unknown.UNKNOWN)
        return self._rules[pattern_id]

    def _rule_from(self, pattern_id, ix, spec: ContextSpec, train, hold, t_train, p_adj) -> ContextRule | None:
        cfg = self.cfg
        inn = np.array([spec.matches(o.situation) is True for o in ix.obs])
        val = np.array([spec.matches(o.situation) is not None for o in ix.obs])
        out = val & ~inn
        if inn.sum() < cfg.min_n or out.sum() < cfg.min_n:
            return None
        s_in, s_out = CellStats.of(ix.y[inn], ix.w[inn], cfg.min_n), CellStats.of(ix.y[out], ix.w[out], cfg.min_n)
        diff = cast(float, s_in.mean) - cast(float, s_out.mean)
        if abs(diff) < cfg.min_effect:
            return None
        t_hold = None
        if len(hold):
            hi, ho = inn[hold], out[hold]
            if hi.sum() >= 5 and ho.sum() >= 5:
                yh = ix.y[hold]
                a, b = yh[hi], yh[ho]
                t_hold = welch_t(a.mean(), a.var(ddof=1), len(a), b.mean(), b.var(ddof=1), len(b))
        confirmed = t_hold is not None and np.sign(t_hold) == np.sign(diff) and abs(t_hold) >= 1.65
        role = "CONTEXT" if diff > 0 else "ANTI_CONTEXT"
        return ContextRule(pattern_id, spec, role, s_in, s_out, float(diff), float(t_train), p_adj, t_hold, bool(confirmed),
                           Epistemic.CONDITIONAL if confirmed else Epistemic.HYPOTHESIS, len(train), len(hold))

    def _singles(self, pattern_id, ix, cands, t_obs, p_adj, train, hold) -> list[ContextRule]:
        out = []
        for j in np.argsort(-np.abs(t_obs))[: 4 * self.cfg.max_rules]:
            if p_adj[j] > self.cfg.alpha:
                break
            path, labs = cands[j]
            r = self._rule_from(pattern_id, ix, ContextSpec((Condition(path, labs),)), train, hold, float(t_obs[j]), float(p_adj[j]))
            if r is not None:
                out.append(r)
        return out

    def _conjunctions(self, pattern_id, ix, cands, t_obs, train, hold) -> list[ContextRule]:
        cfg = self.cfg
        if len(hold) == 0:
            return []
        top = [j for j in np.argsort(-np.abs(t_obs)) if abs(t_obs[j]) >= 2.0][: cfg.conj_top]
        pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:] if cands[a][0] != cands[b][0]]
        out = []
        for a, b in pairs:
            spec = ContextSpec((Condition(*cands[a]), Condition(*cands[b])))
            r = self._rule_from(pattern_id, ix, spec, train, hold, float(max(abs(t_obs[a]), abs(t_obs[b]))), None)
            if r is None or r.t_holdout is None:
                continue
            p_h = float(PS.bonferroni([PS.t_to_p(r.t_holdout)], len(pairs))[0])
            best_single = max(abs(t_obs[a]), abs(t_obs[b]))
            if r.confirmed and p_h < cfg.alpha and abs(r.t_train) >= best_single:
                out.append(r)
        return out

    @staticmethod
    def _dedupe(rules: list[ContextRule]) -> list[ContextRule]:
        """Drop a rule when a simpler kept rule of the same role already covers the same dimension set at a similar effect."""
        kept: list[ContextRule] = []
        for r in rules:
            dup = False
            for k in kept:
                if k.role == r.role and {c.path for c in k.spec.conditions} <= {c.path for c in r.spec.conditions} \
                        and abs(r.diff) <= abs(k.diff) * 1.25:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        return kept

    def rules(self, pattern_id: str, now, seed: int = 0) -> tuple[ContextRule, ...]:
        if pattern_id not in self._rules:
            self.discover(pattern_id, now, seed)
        return self._rules[pattern_id]

    # ---- estimation
    def estimate(self, pattern_id: str, sit: Situation, now, seed: int = 0) -> ContextEstimate:
        """Expected signed edge of `pattern_id` in `sit`, using confirmed rules when one applies and ladder pooling otherwise."""
        ix = self._index(pattern_id, now)
        if ix is None:
            return ContextEstimate(pattern_id, None, None, None, "none", Unknown.UNTESTED, (), -1, (), (), (), 0,
                                   "no observations of this pattern: UNTESTED")
        exp_l, se_l, p_l, trail, used, n_used = self.ladder_estimate(pattern_id, sit, now)
        rules = [r for r in self.rules(pattern_id, now, seed) if r.confirmed]
        inn: list[ContextRule] = []
        outs: list[ContextRule] = []
        unk: list[ContextRule] = []
        for r in rules:
            m = r.spec.matches(sit)
            (unk if m is None else inn if m else outs).append(r)
        if exp_l is None:
            return ContextEstimate(pattern_id, None, None, None, "none", Unknown.INSUFFICIENT_DATA, trail, used, (), (), (), ix.n,
                                   "insufficient data")
        expected, se, p_hit, source = exp_l, se_l, p_l, "ladder"
        chosen = None
        if inn or outs:
            chosen = min(inn + outs, key=lambda r: (r.p_adj if r.p_adj is not None else 1.0, -abs(r.diff), r.rule_id))
            side = chosen.stats_in if chosen in inn else chosen.stats_out
            base = trail[0].shrunk_mean
            k = self.cfg.rule_shrink
            expected = (side.n * cast(float, side.mean) + k * base) / (side.n + k)
            se = math.sqrt((side.var if side.var is not None else ix.var) / (side.n + k))
            p_hit = (cast(float, side.hit) * side.n + 0.5 * k) / (side.n + k)
            source = "rule"
        lines = [f"pattern {pattern_id}: expected edge {expected:+.4f} +/- {se:.4f} via {source}"]
        for s in trail:
            lines.append(f"  rung {s.rung} {'/'.join(s.key) or 'pattern-wide'}: n={s.n}, raw={'na' if s.raw_mean is None else format(s.raw_mean, '+.4f')}, "
                         f"shrunk={s.shrunk_mean:+.4f} (k={s.k:.0f})")
        if used < len(POOLING_LADDER) - 1:
            lines.append(f"  pooling stopped at rung {used}: the next bucket has fewer than {self.cfg.min_n} observations")
        if chosen is not None:
            lines.append("  " + ("IN context: " if chosen in inn else "OUTSIDE context: ") + chosen.describe())
        for r in unk:
            lines.append(f"  rule {r.rule_id} cannot be evaluated: a needed dimension is unobserved")
        return ContextEstimate(pattern_id, float(cast(float, expected)), float(cast(float, se)), float(cast(float, p_hit)), source, None, trail, used,
                               tuple(r.rule_id for r in inn), tuple(r.rule_id for r in outs), tuple(r.rule_id for r in unk),
                               n_used, "\n".join(lines))

    def p_outcome(self, pattern_id: str, spec: ContextSpec, now) -> dict[str, Any]:
        """P(outcome | pattern, context) and P(outcome | pattern, NOT context) for an arbitrary spec (contract section 8)."""
        ix = self._index(pattern_id, now)
        if ix is None:
            return {"in": CellStats.of(np.array([]), None, self.cfg.min_n), "out": CellStats.of(np.array([]), None, self.cfg.min_n),
                    "unobserved": 0, "diff": None, "t": None}
        m = [spec.matches(o.situation) for o in ix.obs]
        inn = np.array([x is True for x in m])
        out = np.array([x is False for x in m])
        a, b = CellStats.of(ix.y[inn], ix.w[inn], self.cfg.min_n), CellStats.of(ix.y[out], ix.w[out], self.cfg.min_n)
        diff = None if a.mean is None or b.mean is None else a.mean - b.mean
        t = None if diff is None else welch_t(a.mean, a.var, a.n, b.mean, b.var, b.n)
        return {"in": a, "out": b, "unobserved": int(sum(x is None for x in m)), "diff": diff, "t": t}

    # ---- diagnostics
    def heterogeneity(self, pattern_id: str, now, path: str = "regime.label") -> dict[str, float | int | None]:
        """Cochran-style Q test: does the pattern's mean differ across the buckets of one dimension at all?"""
        ix = self._index(pattern_id, now)
        if ix is None:
            return {"Q": None, "df": 0, "p": None, "groups": 0}
        groups: dict[str, list[float]] = {}
        for lab, y in zip(ix.labels, ix.y):
            if lab.get(path, "na") != "na":
                groups.setdefault(lab[path], []).append(float(y))
        groups = {k: v for k, v in groups.items() if len(v) >= self.cfg.min_n}
        if len(groups) < 2:
            return {"Q": None, "df": 0, "p": None, "groups": len(groups)}
        F, p = sps.f_oneway(*[np.array(v) for v in groups.values()])
        return {"Q": float(F), "df": len(groups) - 1, "p": float(p), "groups": len(groups)}

    def tiny_bucket_audit(self, pattern_id: str, now) -> dict[int, dict[str, float]]:
        """Per ladder rung: buckets, tiny buckets (< min_n) and the share of observations sitting in tiny buckets."""
        ix = self._index(pattern_id, now)
        if ix is None:
            return {}
        rep = {}
        for r, d in enumerate(ix.rung_acc):
            tiny = [a.n for a in d.values() if a.n < self.cfg.min_n]
            rep[r] = {"buckets": len(d), "tiny": len(tiny), "obs_in_tiny": int(sum(tiny)), "k": round(ix.k[r], 3)}
        return rep

    def pooling_cv(self, pattern_id: str, now, folds: int = 5, seed: int = 0) -> dict[str, float]:
        """Out-of-sample squared error of (a) the pooled ladder estimate vs (b) the raw mean of the deepest rung with data.
        Cross-validated by folds over observations. pooled < raw means shrinkage earned its keep."""
        ix = self._index(pattern_id, now)
        if ix is None or ix.n < folds * self.cfg.min_n:
            return {"pooled_mse": float("nan"), "raw_mse": float("nan"), "n": 0 if ix is None else ix.n}
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, folds, size=ix.n)
        sub_cfg = self.cfg
        pooled, raw, ys = [], [], []
        for f in range(folds):
            tr = [o for o, g in zip(ix.obs, fold) if g != f]
            te = [o for o, g in zip(ix.obs, fold) if g == f]
            sub = _Index(tr, now, sub_cfg)
            for o in te:
                a0 = sub.rung_acc[0][()]
                m = a0.s1 / (a0.sw + sub.k[0])
                for r in range(1, len(POOLING_LADDER)):
                    acc = sub.rung_acc[r].get(ladder_key(o.situation, r))
                    if acc is None or acc.n < sub_cfg.min_n:
                        break
                    m = (acc.s1 + sub.k[r] * m) / (acc.sw + sub.k[r])
                # raw estimate: the deepest bucket with ANY data, however small (the no-pooling strawman)
                raw_est = a0.mean
                for r in range(1, len(POOLING_LADDER)):
                    acc = sub.rung_acc[r].get(ladder_key(o.situation, r))
                    if acc is None or acc.n < 1:
                        break
                    raw_est = acc.mean
                pooled.append(m)
                raw.append(raw_est)
                ys.append(o.outcome)
        ys_a = np.array(ys)
        return {"pooled_mse": float(np.mean((np.array(pooled) - ys_a) ** 2)), "raw_mse": float(np.mean((np.array(raw) - ys_a) ** 2)),
                "n": int(len(ys_a))}

    # ---- drift & export
    def drift_check(self, rule: ContextRule, later: Sequence[Obs], now) -> dict[str, Any]:
        """Does a rule still hold on observations that arrived after it was learned? Returns t of in-out and a verdict."""
        ys_in, ys_out = [], []
        for o in later:
            if o.pattern_id != rule.pattern_id:
                continue
            require_past(o.matured, now, "drift observation")
            m = rule.spec.matches(o.situation)
            if m is True:
                ys_in.append(o.outcome)
            elif m is False:
                ys_out.append(o.outcome)
        if len(ys_in) < 10 or len(ys_out) < 10:
            return {"verdict": Unknown.INSUFFICIENT_DATA, "t": None, "n_in": len(ys_in), "n_out": len(ys_out)}
        a, b = np.array(ys_in), np.array(ys_out)
        t = welch_t(a.mean(), a.var(ddof=1), len(a), b.mean(), b.var(ddof=1), len(b))
        same = np.sign(t) == np.sign(rule.diff)
        verdict = Epistemic.CONDITIONAL if same and abs(t) >= 1.65 else Epistemic.CONTRADICTED if not same and abs(t) >= 1.65 else Epistemic.DEGRADED
        return {"verdict": verdict, "t": float(t), "n_in": len(a), "n_out": len(b), "diff": float(a.mean() - b.mean())}

    def table(self, now, seed: int = 0) -> list[dict[str, Any]]:
        rows = []
        for pid in self.patterns():
            for r in self.rules(pid, now, seed):
                rows.append({"rule_id": r.rule_id, "pattern": pid, "role": r.role, "context": r.spec.describe(),
                             "n_in": r.stats_in.n, "n_out": r.stats_out.n, "mean_in": r.stats_in.mean, "mean_out": r.stats_out.mean,
                             "t_train": r.t_train, "p_adj": r.p_adj, "t_holdout": r.t_holdout, "confirmed": r.confirmed})
        return rows

    def fingerprint(self) -> str:
        return stable_hash({pid: [(o.situation.exact_id, round(o.outcome, 10), str(as_date(o.matured)), o.weight) for o in obs]
                            for pid, obs in sorted(self._obs.items())})


def contexts_from_knowledge(k: Any) -> tuple[ContextSpec, ContextSpec]:
    """(context, anti-context) specs of any KnowledgeLike object, from its `contexts` / `anti_contexts` mappings."""
    return _spec_of_field(getattr(k, "contexts", None)), _spec_of_field(getattr(k, "anti_contexts", None))


def _spec_of_field(field: Any) -> ContextSpec:
    """A `contexts` field is either the plain {path: {"in": [...]}} mapping or a typed knowledge.ContextSet; both give a spec."""
    if field is not None and hasattr(field, "conditions") and hasattr(field, "any_of"):
        return ContextSpec.from_typed(field)
    return ContextSpec.from_mapping(field)


def context_gate(sit: Situation, k: Any) -> tuple[bool | None, str]:
    """(applies?, reason). Applies when the context holds AND the anti-context does not. None = cannot tell (unobserved)."""
    ctx, anti = contexts_from_knowledge(k)
    c = ctx.matches(sit)
    a = anti.matches(sit) if anti.conditions else False
    if a is True:
        return False, f"anti-context holds: {anti.describe()}"
    if c is False:
        return False, f"context does not hold: {ctx.describe()}"
    if c is None or a is None:
        return None, "a dimension needed by the context is unobserved"
    return True, f"context holds: {ctx.describe()}"


# ------------------------------------------------------------------------------------------------ additive alternative

class RidgeContext:
    """Additive context model: outcome ~ intercept + sum of bucket effects, ridge-penalised (lambda by K-fold CV). It captures
    several weak context effects at once without needing any joint bucket to be large, so it is the counterpart to the
    interaction-friendly ladder: `compare_estimators` says which one predicts better out of sample for a pattern."""

    def __init__(self, paths: Sequence[str] | None = None, min_n: int = 30, lambdas: Sequence[float] = (3, 10, 30, 100, 300, 1000)):
        self.paths = tuple(paths) if paths is not None else tuple(dict.fromkeys(p for rung in POOLING_LADDER for p in rung))
        self.min_n = min_n
        self.lambdas = tuple(lambdas)
        self.columns: list[tuple[str, str]] = []
        self.intercept = 0.0
        self.coef = np.zeros(0)
        self.lam = float("nan")
        self.n = 0

    def _matrix(self, sits: Sequence[Situation]) -> np.ndarray:
        X = np.zeros((len(sits), len(self.columns)))
        idx = {c: j for j, c in enumerate(self.columns)}
        for i, s in enumerate(sits):
            b = s.bins()
            for p in self.paths:
                j = idx.get((p, b.get(p, "na")))
                if j is not None:
                    X[i, j] = 1.0
        return X

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
        mu = float(y.mean())
        A = X.T @ X + lam * np.eye(X.shape[1])
        return mu, np.linalg.solve(A, X.T @ (y - mu))

    def fit(self, obs: Sequence[Obs], now, seed: int = 0, folds: int = 5) -> "RidgeContext":
        for o in obs:
            require_past(o.matured, now, "ridge observation")
        self.n = len(obs)
        counts: dict[tuple[str, str], int] = {}
        for o in obs:
            b = o.situation.bins()
            for p in self.paths:
                lab = b.get(p, "na")
                if lab != "na":
                    counts[(p, lab)] = counts.get((p, lab), 0) + 1
        self.columns = sorted(c for c, n in counts.items() if n >= self.min_n)
        if not self.columns or self.n < 2 * self.min_n:
            self.coef, self.intercept, self.lam = np.zeros(len(self.columns)), float(np.mean([o.outcome for o in obs])) if obs else 0.0, float("nan")
            return self
        X = self._matrix([o.situation for o in obs])
        y = np.array([o.outcome for o in obs], dtype=float)
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, folds, size=len(y))
        best, best_mse = self.lambdas[-1], float("inf")
        for lam in self.lambdas:
            err = 0.0
            for f in range(folds):
                tr, te = fold != f, fold == f
                if tr.sum() < 5 or te.sum() == 0:
                    continue
                mu, c = self._solve(X[tr], y[tr], lam)
                err += float(((y[te] - mu - X[te] @ c) ** 2).sum())
            if err < best_mse:
                best, best_mse = lam, err
        self.lam = float(best)
        self.intercept, self.coef = self._solve(X, y, self.lam)
        return self

    def predict(self, sit: Situation) -> float:
        if not len(self.coef):
            return self.intercept
        return float(self.intercept + (self._matrix([sit]) @ self.coef)[0])

    def coefficients(self) -> dict[tuple[str, str], float]:
        return {c: float(v) for c, v in zip(self.columns, self.coef)}

    def top_effects(self, n: int = 5) -> list[tuple[str, str, float]]:
        order = np.argsort(-np.abs(self.coef))[:n]
        return [(self.columns[i][0], self.columns[i][1], float(self.coef[i])) for i in order]


def compare_estimators(model: ContextModel, pattern_id: str, now, folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Cross-validated squared error of four estimates of a pattern's edge in a situation: the pattern-wide shrunk mean,
    the pooled ladder, the additive ridge model, and the rule-free raw bucket mean. Lower is better; the winner tells you
    whether interaction (ladder) or additive (ridge) context structure is what the data supports."""
    obs = list(model._obs.get(pattern_id, ()))
    if len(obs) < folds * model.cfg.min_n:
        return {"n": len(obs), "flat": float("nan"), "ladder": float("nan"), "ridge": float("nan"), "raw": float("nan")}
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(obs))
    err: dict[str, list[float]] = {"flat": [], "ladder": [], "ridge": [], "raw": []}
    for f in range(folds):
        tr = [o for o, g in zip(obs, fold) if g != f]
        te = [o for o, g in zip(obs, fold) if g == f]
        sub = ContextModel(model.cfg)
        sub.add_many(tr)
        rc = RidgeContext(min_n=model.cfg.min_n).fit(tr, now, seed)
        ix = cast(_Index, sub._index(pattern_id, now))
        a0 = ix.rung_acc[0][()]
        flat = a0.s1 / (a0.sw + ix.k[0])
        for o in te:
            ladder, _, _, _, _, _ = sub.ladder_estimate(pattern_id, o.situation, now)
            raw = a0.mean
            for r in range(1, len(POOLING_LADDER)):
                acc = ix.rung_acc[r].get(ladder_key(o.situation, r))
                if acc is None:
                    break
                raw = acc.mean
            for name, val in (("flat", flat), ("ladder", ladder), ("ridge", rc.predict(o.situation)), ("raw", raw)):
                err[name].append((o.outcome - cast(float, val)) ** 2)
    return {"n": len(obs), **{k: float(np.mean(v)) for k, v in err.items()}}


def context_skill(model: ContextModel, pattern_id: str, now, blocks: int = 5) -> dict[str, Any]:
    """Walk-forward skill of context-aware estimates over the pattern-wide mean: split the observations in time into
    `blocks`; predict each block from a model fitted on the earlier blocks only. skill = 1 - MSE(ladder)/MSE(flat).
    Non-positive skill is reported as FAILED so a caller can fall back to the unconditional estimate instead of trusting
    a context model that has not earned it."""
    obs = sorted(model._obs.get(pattern_id, ()), key=lambda o: as_date(o.matured))
    n = len(obs)
    if n < blocks * model.cfg.min_n:
        return {"n": n, "skill": float("nan"), "status": "INSUFFICIENT_EVIDENCE", "blocks": 0}
    edges = np.linspace(0, n, blocks + 1).astype(int)
    e_ctx, e_flat = [], []
    used = 0
    for b in range(1, blocks):
        tr, te = obs[: edges[b]], obs[edges[b]: edges[b + 1]]
        if len(tr) < 2 * model.cfg.min_n or not te:
            continue
        cutoff = max(as_date(o.matured) for o in tr) + dt.timedelta(days=1)
        sub = ContextModel(model.cfg)
        sub.add_many(tr)
        ix = cast(_Index, sub._index(pattern_id, cutoff))
        a0 = ix.rung_acc[0][()]
        flat = a0.s1 / (a0.sw + ix.k[0])
        for o in te:
            ctx, _, _, _, _, _ = sub.ladder_estimate(pattern_id, o.situation, cutoff)
            e_ctx.append((o.outcome - cast(float, ctx)) ** 2)
            e_flat.append((o.outcome - flat) ** 2)
        used += 1
    if not e_ctx:
        return {"n": n, "skill": float("nan"), "status": "INSUFFICIENT_EVIDENCE", "blocks": 0}
    skill = 1.0 - float(np.mean(e_ctx)) / max(float(np.mean(e_flat)), 1e-18)
    gain = np.array(e_flat) - np.array(e_ctx)
    from .. import analog_weighting as AW
    t = AW.hac_t(gain)
    status = "FAILED" if skill <= 0 else "PROVEN" if np.isfinite(t) and t > 1.65 else "UNPROVEN"
    return {"n": n, "skill": skill, "hac_t": float(t), "status": status, "blocks": used}


# ------------------------------------------------------------------------------------------------ boundaries & interactions

@dataclasses.dataclass(frozen=True)
class Boundary:
    """A change point of a pattern's edge along one ordered dimension."""
    pattern_id: str
    path: str
    below: tuple[str, ...]              # buckets on the low side of the cut
    above: tuple[str, ...]
    mean_below: float
    mean_above: float
    n_below: int
    n_above: int
    t: float
    p_adj: float                        # max-T across all cuts of this dimension (within-week permutation)
    flips_sign: bool                    # the pattern changes sign across the cut


def sweep_boundary(model: ContextModel, pattern_id: str, path: str, now, seed: int = 0) -> Boundary | None:
    """Where along an ordered dimension does the pattern's edge change? Tries every cut, keeps the strongest, and
    corrects for having tried them all with a within-week permutation max-|t|. None when the dimension is unordered,
    unobserved, or no cut leaves `min_n` observations on both sides."""
    order = ordered_labels(path)
    ix = model._index(pattern_id, now)
    if order is None or ix is None:
        return None
    labs = np.array([lb.get(path, "na") for lb in ix.labels], dtype=object)
    valid = labs != "na"
    rank = {l: i for i, l in enumerate(order)}
    pos = np.array([rank.get(l, -1) for l in labs])
    cuts = []
    for c in range(len(order) - 1):
        lo, hi = valid & (pos <= c) & (pos >= 0), valid & (pos > c)
        if lo.sum() >= model.cfg.min_n and hi.sum() >= model.cfg.min_n:
            cuts.append((c, lo, hi))
    if not cuts:
        return None
    I = np.column_stack([lo for _, lo, _ in cuts]).astype(float)
    V = np.column_stack([lo | hi for _, lo, hi in cuts]).astype(float)
    t_obs = ContextModel._t_from_sums(I, V, ix.y[:, None])[0][:, 0]
    rng = np.random.default_rng(seed)
    Yp = np.column_stack([PS.permute_within_clusters(ix.y, ix.codes, rng) for _ in range(model.cfg.n_perm)])
    tp = ContextModel._t_from_sums(I, V, Yp)[0]
    maxt = np.abs(tp).max(axis=0)
    j = int(np.argmax(np.abs(t_obs)))
    c, lo, hi = cuts[j]
    p_adj = float((1 + (maxt >= abs(t_obs[j])).sum()) / (1 + model.cfg.n_perm))
    m_lo, m_hi = float(ix.y[lo].mean()), float(ix.y[hi].mean())
    return Boundary(pattern_id, path, tuple(order[: c + 1]), tuple(order[c + 1:]), m_lo, m_hi, int(lo.sum()), int(hi.sum()),
                    float(t_obs[j]), p_adj, bool(np.sign(m_lo) != np.sign(m_hi) and m_lo != 0 and m_hi != 0))


def interaction_effect(model: ContextModel, pattern_id: str, other_id: str, now) -> dict[str, Any]:
    """Pattern interaction: how does `pattern_id` behave when `other_id` is co-active versus not? A pattern that only works
    together with another (or that the other cancels) shows up as a significant difference with enough cases each side."""
    ix = model._index(pattern_id, now)
    if ix is None:
        return {"n_with": 0, "n_without": 0, "diff": None, "t": None, "verdict": Unknown.UNTESTED}
    with_ = np.array([other_id in o.situation.pattern_ids for o in ix.obs])
    a, b = ix.y[with_], ix.y[~with_]
    if len(a) < model.cfg.min_n or len(b) < model.cfg.min_n:
        return {"n_with": int(len(a)), "n_without": int(len(b)), "diff": None, "t": None, "verdict": Unknown.INSUFFICIENT_DATA}
    t = welch_t(a.mean(), a.var(ddof=1), len(a), b.mean(), b.var(ddof=1), len(b))
    verdict = "SYNERGY" if t > 2 else "INTERFERENCE" if t < -2 else "INDEPENDENT"
    return {"n_with": int(len(a)), "n_without": int(len(b)), "mean_with": float(a.mean()), "mean_without": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "t": float(t), "verdict": verdict}


def temporal_stability(model: ContextModel, rule: ContextRule, now, blocks: int = 4) -> dict[str, Any]:
    """Does a rule's in-minus-out difference keep its sign across time blocks? Reports per-block difference and t and the
    share of blocks agreeing with the full-sample sign. A context that only worked in one era is a regime artefact."""
    ix = model._index(rule.pattern_id, now)
    if ix is None:
        return {"blocks": [], "agree_share": None, "verdict": Unknown.UNTESTED}
    m = np.array([rule.spec.matches(o.situation) for o in ix.obs], dtype=object)
    inn, out = m == True, m == False                    # noqa: E712  (None must stay distinct from False)
    edges = np.linspace(0, ix.n, blocks + 1).astype(int)
    rows = []
    for b in range(blocks):
        sl = slice(edges[b], edges[b + 1])
        a, c = ix.y[sl][inn[sl]], ix.y[sl][out[sl]]
        if len(a) >= 8 and len(c) >= 8:
            rows.append({"block": b, "n_in": int(len(a)), "n_out": int(len(c)), "diff": float(a.mean() - c.mean()),
                         "t": welch_t(a.mean(), a.var(ddof=1), len(a), c.mean(), c.var(ddof=1), len(c))})
    if len(rows) < 3:
        return {"blocks": rows, "agree_share": None, "verdict": Unknown.INSUFFICIENT_DATA}
    share = float(np.mean([np.sign(r["diff"]) == np.sign(rule.diff) for r in rows]))
    return {"blocks": rows, "agree_share": share, "verdict": "STABLE" if share >= 0.75 else "UNSTABLE"}


# ------------------------------------------------------------------------------------------------ rule book

class RuleBook:
    """Append-only book of learned rules. Retiring a rule marks it; nothing is ever deleted (contract section 13), so a
    rule can be reactivated when its context returns and the reason it was retired stays on record."""

    def __init__(self):
        self._rules: dict[str, ContextRule] = {}
        self._log: list[dict[str, Any]] = []
        self._retired: dict[str, str] = {}

    def add(self, rule: ContextRule, now) -> bool:
        errs = rule.validate()
        if errs:
            raise ValueError("invalid rule: " + "; ".join(errs))
        rid = rule.rule_id
        new = rid not in self._rules
        if not new and self._rules[rid] == rule:
            return False
        self._rules[rid] = rule
        self._log.append({"event": "added" if new else "updated", "rule": rid, "now": str(as_date(now))})
        return True

    def retire(self, rule_id: str, now, reason: str) -> None:
        if rule_id not in self._rules:
            raise KeyError(rule_id)
        if not reason:
            raise ValueError("a retirement needs a reason")
        self._retired[rule_id] = reason
        self._log.append({"event": "retired", "rule": rule_id, "now": str(as_date(now)), "reason": reason})

    def reactivate(self, rule_id: str, now, evidence: str) -> None:
        if rule_id not in self._retired:
            raise KeyError(f"{rule_id} is not retired")
        del self._retired[rule_id]
        self._log.append({"event": "reactivated", "rule": rule_id, "now": str(as_date(now)), "evidence": evidence})

    def active(self, pattern_id: str | None = None) -> list[ContextRule]:
        return sorted((r for k, r in self._rules.items() if k not in self._retired and (pattern_id is None or r.pattern_id == pattern_id)),
                      key=lambda r: r.rule_id)

    def get(self, rule_id: str) -> ContextRule:
        return self._rules[rule_id]

    def is_retired(self, rule_id: str) -> bool:
        return rule_id in self._retired

    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._log)

    def __len__(self) -> int:
        return len(self._rules)

    def conflicts(self, pattern_id: str) -> list[tuple[str, str]]:
        """Pairs of active rules for one pattern that share a dimension yet assign opposite roles to overlapping buckets."""
        act = self.active(pattern_id)
        out = []
        for i, a in enumerate(act):
            for b in act[i + 1:]:
                if a.role == b.role:
                    continue
                shared = {c.path: c for c in a.spec.conditions} .keys() & {c.path: c for c in b.spec.conditions}.keys()
                for p in shared:
                    ca = next(c for c in a.spec.conditions if c.path == p)
                    cb = next(c for c in b.spec.conditions if c.path == p)
                    if set(ca.allowed) & set(cb.allowed) and ca.negate == cb.negate:
                        out.append((a.rule_id, b.rule_id))
                        break
        return out

    def matching(self, pattern_id: str, sit: Situation) -> dict[str, list[str]]:
        res: dict[str, list[str]] = {"in": [], "out": [], "unknown": []}
        for r in self.active(pattern_id):
            m = r.spec.matches(sit)
            res["unknown" if m is None else "in" if m else "out"].append(r.rule_id)
        return res

    def revalidate(self, model: ContextModel, later: Sequence[Obs], now) -> dict[str, dict[str, Any]]:
        """Run `drift_check` on every active rule; retire those that were CONTRADICTED by later data, mark DEGRADED ones."""
        out = {}
        for r in self.active():
            chk = model.drift_check(r, later, now)
            out[r.rule_id] = chk
            if chk["verdict"] == Epistemic.CONTRADICTED:
                self.retire(r.rule_id, now, f"contradicted by later data (t={chk['t']:.2f})")
        return out


# ------------------------------------------------------------------------------------------------ persistence

def model_state(model: ContextModel) -> dict[str, Any]:
    """JSON-safe state of a ContextModel (config + observations). Rules are re-derived, never trusted from disk."""
    return {"config": dataclasses.asdict(model.cfg),
            "obs": {pid: [{"situation": o.situation.to_dict(), "outcome": o.outcome, "matured": str(as_date(o.matured)),
                           "weight": o.weight} for o in obs] for pid, obs in sorted(model._obs.items())}}


def model_from_state(state: Mapping[str, Any]) -> ContextModel:
    cfg = ContextConfig(**state["config"])
    m = ContextModel(cfg)
    for pid, rows in state["obs"].items():
        for r in rows:
            m.add(Obs(pid, Situation.from_dict(r["situation"]), r["outcome"], r["matured"], r.get("weight", 1.0)))
    return m


# ------------------------------------------------------------------------------------------------ importance, weighting, reports

def dimension_importance(model: ContextModel, pattern_id: str, now, folds: int = 4, seed: int = 0) -> dict[str, float]:
    """Which situation dimensions carry the context effect? Drop-one importance on the additive model: the rise in
    cross-validated squared error when a dimension's buckets are removed. Positive = the dimension helps predict the edge;
    about zero or negative = it adds nothing (the cure for reading meaning into every dimension that merely exists)."""
    obs = list(model._obs.get(pattern_id, ()))
    paths = tuple(dict.fromkeys(p for rung in POOLING_LADDER for p in rung))
    if len(obs) < folds * model.cfg.min_n:
        return {}
    for o in obs:
        require_past(o.matured, now, "importance observation")
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(obs))

    def cv(use):
        err = 0.0
        for f in range(folds):
            tr = [o for o, g in zip(obs, fold) if g != f]
            te = [o for o, g in zip(obs, fold) if g == f]
            rc = RidgeContext(paths=use, min_n=model.cfg.min_n).fit(tr, now, seed)
            err += sum((o.outcome - rc.predict(o.situation)) ** 2 for o in te)
        return err / len(obs)
    full = cv(paths)
    return {p: float(cv(tuple(q for q in paths if q != p)) - full) for p in paths}


def rule_hit_intervals(rule: ContextRule, level: float = 0.9) -> dict[str, tuple[float, float, float]]:
    """Beta credible interval of the hit rate (outcome > 0) inside and outside a rule's context, prior centred on 0.5."""
    out = {}
    for side, st in (("in", rule.stats_in), ("out", rule.stats_out)):
        out[side] = beta_interval((st.hit or 0.0) * st.n, st.n, 0.5, 4.0, level)
    return out


def context_weight(model: ContextModel, pattern_id: str, sit: Situation, now, seed: int = 0, blocks: int = 5) -> dict[str, Any]:
    """A multiplier in [0, 1.5] for the pattern's weight in this situation: the context-aware expected edge relative to the
    unconditional one. If the context model has no proven walk-forward skill for this pattern (`context_skill` FAILED or
    INSUFFICIENT_EVIDENCE) the multiplier is 1.0 and the reason says so: an unearned context must not move a decision."""
    est = model.estimate(pattern_id, sit, now, seed)
    if est.expected is None:
        return {"multiplier": 1.0, "used_context": False, "reason": f"{est.unknown}: no estimate", "estimate": est}
    sk = context_skill(model, pattern_id, now, blocks)
    flat = model.ladder_estimate(pattern_id, sit, now)[3][0].shrunk_mean if est.trail else None
    if sk["status"] in ("FAILED", "INSUFFICIENT_EVIDENCE") or flat is None or abs(flat) < 1e-12:
        return {"multiplier": 1.0, "used_context": False, "estimate": est,
                "reason": f"context model skill is {sk['status']}: using the unconditional weight"}
    mult = float(np.clip(est.expected / flat, 0.0, 1.5))
    return {"multiplier": mult, "used_context": True, "estimate": est, "reason": f"skill {sk['skill']:+.3f} ({sk['status']}); conditional/unconditional edge"}


def explain_pattern(model: ContextModel, pattern_id: str, now, seed: int = 0) -> str:
    """One readable block: what is known about a pattern's context, what could not be seen, and how far to trust it."""
    ix = model._index(pattern_id, now)
    if ix is None:
        return f"{pattern_id}: no observations (UNTESTED)"
    rules = model.rules(pattern_id, now, seed)
    rep = model.reports.get(pattern_id)
    lines = [f"{pattern_id}: {ix.n} matured observations, mean edge {float(ix.y.mean()):+.4f}"]
    if rep is not None:
        lines.append(f"  searched {rep.candidates} splits (train {rep.n_train}, holdout {rep.n_holdout}); "
                     f"minimum detectable in-out difference {rep.mde if rep.mde is None else round(rep.mde, 4)}")
    if not rules:
        lines.append("  no context learned: " + (rep.reason if rep else "not fitted"))
    for r in rules:
        lines.append("  " + r.describe())
        iv = rule_hit_intervals(r)
        lines.append(f"    hit rate in {iv['in'][0]:.2f} [{iv['in'][1]:.2f},{iv['in'][2]:.2f}] vs out {iv['out'][0]:.2f} [{iv['out'][1]:.2f},{iv['out'][2]:.2f}]")
    sk = context_skill(model, pattern_id, now)
    lines.append(f"  walk-forward context skill: {sk['status']}" + ("" if not np.isfinite(sk["skill"]) else f" ({sk['skill']:+.3f})"))
    return "\n".join(lines)


def rulebook_state(book: RuleBook) -> dict[str, Any]:
    """JSON-safe snapshot of a RuleBook (rules, retirements, history). Retired rules stay in the snapshot with their reason."""
    def enc(r: ContextRule) -> dict:
        return {"pattern_id": r.pattern_id, "spec": r.spec.to_mapping(), "role": r.role, "diff": r.diff, "t_train": r.t_train,
                "p_adj": r.p_adj, "t_holdout": r.t_holdout, "confirmed": r.confirmed, "epistemic": str(r.epistemic),
                "n_train": r.n_train, "n_holdout": r.n_holdout,
                "in": dataclasses.asdict(r.stats_in), "out": dataclasses.asdict(r.stats_out)}
    return {"rules": {rid: enc(r) for rid, r in sorted(book._rules.items())}, "retired": dict(sorted(book._retired.items())),
            "history": list(book._log)}


def rulebook_from_state(state: Mapping[str, Any]) -> RuleBook:
    book = RuleBook()
    for rid, d in state["rules"].items():
        rule = ContextRule(d["pattern_id"], ContextSpec.from_mapping(d["spec"]), d["role"], CellStats(**d["in"]), CellStats(**d["out"]),
                           d["diff"], d["t_train"], d["p_adj"], d["t_holdout"], d["confirmed"], Epistemic.parse(d["epistemic"]),
                           d["n_train"], d["n_holdout"])
        if rule.rule_id != rid:
            raise ValueError(f"rule {rid} does not reproduce its own id: the snapshot was edited or the id scheme changed")
        book._rules[rid] = rule
    book._retired = dict(state.get("retired", {}))
    book._log = list(state.get("history", []))
    return book

def estimate_many(model: ContextModel, pattern_ids: Iterable[str], sit: Situation, now, seed: int = 0) -> dict[str, ContextEstimate]:
    """Context-aware estimates for several patterns in one situation (one pass, deterministic order)."""
    return {pid: model.estimate(pid, sit, now, seed) for pid in sorted(set(pattern_ids))}


def ladder_report(model: ContextModel, pattern_id: str, now) -> list[dict[str, Any]]:
    """Per pooling rung: bucket count, how many buckets are big enough to speak, the shrinkage strength and the spread of
    bucket means. A rung whose buckets barely differ (large k) is telling you that dimension set adds no context."""
    ix = model._index(pattern_id, now)
    if ix is None:
        return []
    rows = []
    for r, d in enumerate(ix.rung_acc):
        big = [a for a in d.values() if a.n >= model.cfg.min_n]
        means = [a.mean for a in big if a.mean is not None]
        rows.append({"rung": r, "dims": POOLING_LADDER[r], "buckets": len(d), "speaking": len(big), "k": round(ix.k[r], 3),
                     "mean_spread": float(np.std(means)) if len(means) > 1 else 0.0,
                     "obs_speaking": int(sum(a.n for a in big))})
    return rows


def resolve_rules(rules: Sequence[ContextRule]) -> list[ContextRule]:
    """From possibly overlapping rules pick a consistent set: confirmed first, then larger |effect| x sqrt(n); a rule that
    shares a dimension with an already-chosen rule of the OPPOSITE role is dropped (the two cannot both be the explanation),
    same-role rules on the same dimension keep only the stronger."""
    ranked = sorted(rules, key=lambda r: (not r.confirmed, -abs(r.diff) * math.sqrt(min(r.stats_in.n, r.stats_out.n)), r.rule_id))
    chosen: list[ContextRule] = []
    for r in ranked:
        dims = {c.path for c in r.spec.conditions}
        if any(dims & {c.path for c in k.spec.conditions} for k in chosen):
            continue
        chosen.append(r)
    return chosen

def outcome_distribution(model: ContextModel, pattern_id: str, spec: ContextSpec, now,
                         quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95), loss_at: float = -0.05) -> dict[str, Any]:
    """P(outcome | pattern, context) and P(outcome | pattern, NOT context) as distributions, not just means: weighted
    quantiles, the probability of a loss worse than `loss_at`, and the expected shortfall (mean of the worst 5%) on each
    side. A context can leave the mean alone and still change the tail, which is what a risk budget cares about."""
    from .. import analog_weighting as AW
    ix = model._index(pattern_id, now)
    empty = {"n": 0, "quantiles": {}, "p_loss": None, "shortfall": None, "mean": None}
    if ix is None:
        return {"in": dict(empty), "out": dict(empty), "unobserved": 0}
    m = [spec.matches(o.situation) for o in ix.obs]
    res: dict[str, Any] = {}
    for name, sel in (("in", [x is True for x in m]), ("out", [x is False for x in m])):
        sel_mask = np.array(sel)
        y, w = ix.y[sel_mask], ix.w[sel_mask]
        if len(y) == 0:
            res[name] = dict(empty)
            continue
        qs = AW.weighted_quantile(y, w, list(quantiles))
        k = max(int(math.ceil(0.05 * len(y))), 1)
        res[name] = {"n": int(len(y)), "quantiles": {q: v for q, v in zip(quantiles, qs)},
                     "p_loss": float((w * (y < loss_at)).sum() / w.sum()), "shortfall": float(np.sort(y)[:k].mean()),
                     "mean": float((w * y).sum() / w.sum())}
    res["unobserved"] = int(sum(x is None for x in m))
    return res

def bootstrap_rule_effect(model: ContextModel, rule: ContextRule, now, n_boot: int = 400, seed: int = 0,
                          level: float = 0.9) -> dict[str, Any]:
    """Uncertainty of a rule's in-minus-out difference with a WEEK-cluster bootstrap (weeks resampled whole, so the shared
    market move inside a week is not counted as independent evidence). Returns the interval and the share of resamples
    that keep the rule's sign; a confirmed rule whose interval straddles zero is a warning, not a result."""
    ix = model._index(rule.pattern_id, now)
    if ix is None:
        return {"lo": None, "hi": None, "sign_share": None, "n_boot": 0}
    m = [rule.spec.matches(o.situation) for o in ix.obs]
    inn = np.array([x is True for x in m])
    out = np.array([x is False for x in m])
    weeks = np.unique(ix.codes)
    by_week = {w: np.where(ix.codes == w)[0] for w in weeks}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        idx = np.concatenate([by_week[w] for w in pick])
        a, b = ix.y[idx][inn[idx]], ix.y[idx][out[idx]]
        if len(a) >= 5 and len(b) >= 5:
            diffs.append(a.mean() - b.mean())
    if len(diffs) < n_boot // 2:
        return {"lo": None, "hi": None, "sign_share": None, "n_boot": len(diffs)}
    d = np.array(diffs)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"lo": float(lo), "hi": float(hi), "sign_share": float(np.mean(np.sign(d) == np.sign(rule.diff))), "n_boot": len(diffs)}


def refine_rule(model: ContextModel, rule: ContextRule, now, seed: int = 0) -> ContextRule | None:
    """Does a second condition sharpen a rule INSIDE its context? Searches single conditions on other dimensions restricted to
    the rule's in-set (train part), keeps the strongest one only if it is confirmed on the holdout with the same sign, and
    returns the conjunction as a new rule. None when nothing survives - most rules should not refine."""
    cfg = model.cfg
    ix = model._index(rule.pattern_id, now)
    if ix is None or ix.n < 4 * cfg.min_n:
        return None
    inn = np.array([rule.spec.matches(o.situation) is True for o in ix.obs])
    n_hold = int(round(cfg.holdout_frac * ix.n))
    train = np.arange(ix.n - n_hold)
    hold = np.arange(ix.n - n_hold, ix.n)
    rows_tr = train[inn[train]]
    if len(rows_tr) < 2 * cfg.min_n:
        return None
    cands, I, V = model._design(ix, rows_tr)
    used = {c.path for c in rule.spec.conditions}
    keep = [j for j, (p, _) in enumerate(cands) if p not in used]
    if not keep:
        return None
    cands, I, V = [cands[j] for j in keep], I[:, keep], V[:, keep]
    t, _, _ = ContextModel._t_from_sums(I, V, ix.y[rows_tr][:, None])
    rng = np.random.default_rng(seed)
    Yp = np.column_stack([PS.permute_within_clusters(ix.y[rows_tr], ix.codes[rows_tr], rng) for _ in range(cfg.n_perm)])
    tp, _, _ = ContextModel._t_from_sums(I, V, Yp)
    j = int(np.argmax(np.abs(t[:, 0])))
    p_adj = float((1 + (np.abs(tp).max(axis=0) >= abs(t[j, 0])).sum()) / (1 + cfg.n_perm))
    if p_adj > cfg.alpha:
        return None
    spec = rule.spec.with_condition(Condition(*cands[j]))
    new = model._rule_from(rule.pattern_id, ix, spec, train, hold, float(t[j, 0]), p_adj)
    if new is None or not new.confirmed or np.sign(new.diff) != np.sign(rule.diff) or abs(new.diff) <= abs(rule.diff):
        return None
    return new


def summarize_pattern(model: ContextModel, pattern_id: str, now, seed: int = 0) -> dict[str, Any]:
    """Everything downstream needs to know about a pattern's context in one record: overall edge, whether context matters at
    all (heterogeneity across regimes), the confirmed contexts and anti-contexts as knowledge fields, walk-forward context
    skill, and what the data could not have shown (minimum detectable effect)."""
    ix = model._index(pattern_id, now)
    if ix is None:
        return {"pattern_id": pattern_id, "n": 0, "status": Unknown.UNTESTED}
    rules = [r for r in model.rules(pattern_id, now, seed) if r.confirmed]
    ctx: dict[str, Any] = {}
    anti: dict[str, Any] = {}
    for r in rules:
        (ctx if r.role == "CONTEXT" else anti)[r.rule_id] = r.spec.to_mapping()
    het = model.heterogeneity(pattern_id, now)
    sk = context_skill(model, pattern_id, now)
    rep = model.reports.get(pattern_id)
    return {"pattern_id": pattern_id, "n": ix.n, "mean_edge": float(ix.y.mean()), "hit_rate": float((ix.y > 0).mean()),
            "context_dependent": bool(het["p"] is not None and het["p"] < 0.01 or rules), "regime_heterogeneity_p": het["p"],
            "contexts": ctx, "anti_contexts": anti, "context_skill": sk["status"], "mde": None if rep is None else rep.mde,
            "status": Epistemic.CONDITIONAL if rules else Epistemic.SUPPORTED if ix.y.mean() > 0 else Epistemic.HYPOTHESIS}


def rulebook_report(book: RuleBook) -> str:
    lines = [f"{len(book)} rules, {len(book.active())} active"]
    for r in sorted(book._rules.values(), key=lambda r: (r.pattern_id, r.rule_id)):
        tag = f"RETIRED ({book._retired[r.rule_id]})" if book.is_retired(r.rule_id) else "active"
        lines.append(f"  [{tag}] {r.describe()}")
    return "\n".join(lines)

def false_context_rate(model: ContextModel, pattern_id: str, now, n_shuffles: int = 8, seed: int = 0,
                       count: str = "confirmed") -> dict[str, Any]:
    """Empirical false-discovery control. Outcomes are shuffled inside each week (destroying any link to context while keeping
    the weekly market move) and `discover` is run on the shuffled data; the share of shuffles that still yield a CONFIRMED rule
    estimates how often this procedure hallucinates a context. It should sit near or below alpha. count="any" also counts
    unconfirmed (HYPOTHESIS) rules, which measures the search stage alone."""
    if count not in ("confirmed", "any"):
        raise ValueError("count must be confirmed or any")
    ix = model._index(pattern_id, now)
    if ix is None or ix.n < 4 * model.cfg.min_n:
        return {"shuffles": 0, "false_confirmed": 0, "rate": float("nan"), "status": Unknown.INSUFFICIENT_DATA}
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_shuffles):
        y = PS.permute_within_clusters(np.array([o.outcome for o in ix.obs]), ix.codes, rng)
        sub = ContextModel(model.cfg)
        sub.add_many(Obs(pattern_id, o.situation, float(v), o.matured, o.weight) for o, v in zip(ix.obs, y))
        found = sub.discover(pattern_id, now, int(rng.integers(1 << 30)))
        bad += any(r.confirmed for r in found) if count == "confirmed" else bool(found)
    rate = bad / n_shuffles
    return {"shuffles": n_shuffles, "false_confirmed": bad, "rate": rate, "status": "OK" if rate <= 0.2 else "HALLUCINATING"}


def predict_interval(model: ContextModel, pattern_id: str, sit: Situation, now, level: float = 0.9, seed: int = 0) -> dict[str, Any]:
    """Predictive interval for ONE new outcome of the pattern in this situation: the context-aware expected edge plus/minus
    the normal quantile of sqrt(residual variance + estimate variance). The residual variance is the pattern's own outcome
    variance, so the interval is honest about how noisy a single trade is even when the mean is well known."""
    est = model.estimate(pattern_id, sit, now, seed)
    ix = model._index(pattern_id, now)
    if est.expected is None or ix is None or ix.n < 5:
        return {"lo": None, "hi": None, "expected": None, "unknown": est.unknown or Unknown.INSUFFICIENT_DATA}
    z = float(sps.norm.ppf(0.5 + level / 2))
    sd = math.sqrt(max(ix.var, 0.0) + cast(float, est.se) ** 2)
    return {"lo": est.expected - z * sd, "hi": est.expected + z * sd, "expected": est.expected, "sd": sd, "unknown": None}

def rule_stability(model: ContextModel, pattern_id: str, now, n_runs: int = 6, frac: float = 0.8, seed: int = 0) -> dict[str, Any]:
    """Stability selection. Discovery is re-run on random `frac` subsamples (whole weeks kept together); the share of runs in
    which each rule reappears is its selection frequency. A rule found in fewer than about 60% of subsamples depends on a few
    observations and should not be relied on however small its p-value."""
    ix = model._index(pattern_id, now)
    if ix is None or ix.n < 4 * model.cfg.min_n:
        return {"runs": 0, "frequency": {}, "stable": [], "status": Unknown.INSUFFICIENT_DATA}
    rng = np.random.default_rng(seed)
    weeks = np.unique(ix.codes)
    seen: dict[str, int] = {}
    for _ in range(n_runs):
        keep = set(rng.choice(weeks, size=max(int(frac * len(weeks)), 1), replace=False).tolist())
        sub = ContextModel(model.cfg)
        sub.add_many(o for o, w in zip(ix.obs, ix.codes) if int(w) in keep)
        for r in sub.discover(pattern_id, now, int(rng.integers(1 << 30))):
            if r.confirmed:
                seen[r.rule_id] = seen.get(r.rule_id, 0) + 1
    freq = {k: v / n_runs for k, v in sorted(seen.items())}
    return {"runs": n_runs, "frequency": freq, "stable": [k for k, v in freq.items() if v >= 0.6], "status": "OK"}


def regime_table(model: ContextModel, now, path: str = "regime.label") -> list[dict[str, Any]]:
    """Report table: for every pattern and every bucket of one dimension, n, raw mean, hit rate and the reliable flag. The
    quick look that shows whether a pattern behaves the same across regimes before any rule is learned."""
    rows = []
    for pid in model.patterns():
        ix = cast(_Index, model._index(pid, now))
        by: dict[str, list[float]] = {}
        for lab, y in zip(ix.labels, ix.y):
            by.setdefault(lab.get(path, "na"), []).append(float(y))
        for bucket, ys in sorted(by.items()):
            a = np.array(ys)
            rows.append({"pattern": pid, "bucket": bucket, "n": len(a), "mean": float(a.mean()), "hit": float((a > 0).mean()),
                         "reliable": len(a) >= model.cfg.min_n})
    return rows

def outcome_bin_table(model: ContextModel, pattern_id: str, spec: ContextSpec, now) -> dict[str, Any]:
    """P(outcome bin | pattern, context) and P(outcome bin | pattern, NOT context) on the coarse signed bins engine.memory
    already uses for lessons (`outcome_bin`): lessons keep direction and rough size, never the exact realised value. Returns the
    two distributions over the union of bins seen and the total-variation distance between them."""
    from engine.memory import outcome_bin
    ix = model._index(pattern_id, now)
    if ix is None:
        return {"in": {}, "out": {}, "tv": None, "n_in": 0, "n_out": 0}
    counts: dict[str, dict[Any, float]] = {"in": {}, "out": {}}
    for o, y in zip(ix.obs, ix.y):
        m = spec.matches(o.situation)
        if m is None:
            continue
        side = "in" if m else "out"
        b = outcome_bin(float(y))
        counts[side][b] = counts[side].get(b, 0) + 1
    n_in, n_out = sum(counts["in"].values()), sum(counts["out"].values())
    keys = sorted(set(counts["in"]) | set(counts["out"]))
    pin = {k: counts["in"].get(k, 0) / n_in if n_in else 0.0 for k in keys}
    pout = {k: counts["out"].get(k, 0) / n_out if n_out else 0.0 for k in keys}
    tv = None if not (n_in and n_out) else 0.5 * float(sum(abs(pin[k] - pout[k]) for k in keys))
    return {"in": pin, "out": pout, "tv": tv, "n_in": n_in, "n_out": n_out}
