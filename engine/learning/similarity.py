"""Situation similarity (contract C62 section 16; feeds section 17 retrieval; canon C56, C63).

Similarity is NEVER a single opaque number. Two situations are compared on ten named components
(structural, market, volatility, liquidity, pattern, regime, sector, stock-type, recent-history, failure-risk); every
component score, its coverage, its strongest agreements and its strongest disagreements are stored in a
SimilarityResult, and `explain()` says in words why the two were (or were not) considered alike.

Mechanisms:
* field similarity is defined by the FieldSpec that also defines the bucket (engine.learning.situation): numeric
  exp(-|a-b|/scale), ordinal linear, categorical exact; a missing side is UNKNOWN and drops out (never scored as 0 or 1);
* components with too little observed data are None and the total is renormalised over what remains; too little
  overall -> Unknown.INSUFFICIENT_DATA, so missing inputs cannot masquerade as similarity;
* hard vetoes (e.g. a regime mismatch) mark a pair non-comparable no matter how high the total is;
* a vectorised matrix path (`SituationMatrix`) ranks thousands of candidates and is unit-tested against the scalar path;
* component weights can be re-fitted from observed outcome agreement (`fit_weights`), with a ridge pull to the prior
  so a small sample cannot rewrite the notion of similarity.

Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import FirewallBreach, Unknown, as_date, stable_hash
from .situation import BLOCK_SPECS, FieldSpec, Situation, all_paths, spec_of

COMPONENTS = ("structural", "market", "volatility", "liquidity", "pattern", "regime", "sector", "stock_type",
              "recent_history", "failure_risk")

# component -> dimensions it looks at ('block.*' takes every field of a block). structural is computed separately.
COMPONENT_FIELDS: dict[str, tuple[str, ...]] = {
    "market": ("market.*", "breadth.*", "macro.*"),
    "volatility": ("volatility.*",),
    "liquidity": ("liquidity.*",),
    "pattern": ("pattern_interaction.*",),
    "regime": ("regime.*",),
    "sector": ("sector.*",),
    "stock_type": ("stock_type.kind", "stock_type.persistence", "stock_type.event_load", "correlation.beta",
                   "correlation.corr_to_mkt", "correlation.state"),
    "recent_history": ("trend.*", "shock.gap_today", "shock.max20", "shock.min20"),
    "failure_risk": ("shock.kind", "shock.red_flag", "shock.market_shock", "shock.days_since_earn", "position_risk.*",
                     "stock_type.lottery_rank", "stock_type.skew60", "volatility.vol_ratio", "correlation.avg_pair_corr"),
}
DEFAULT_WEIGHTS = {"structural": 0.10, "market": 0.14, "volatility": 0.12, "liquidity": 0.06, "pattern": 0.14, "regime": 0.14,
                   "sector": 0.05, "stock_type": 0.08, "recent_history": 0.10, "failure_risk": 0.07}
DEFAULT_VETOES = {"regime": 0.4}          # component -> floor below which the pair is not comparable


def expand_fields(patterns: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for p in patterns:
        if p.endswith(".*"):
            out.extend(f"{p[:-2]}.{s.name}" for s in BLOCK_SPECS[p[:-2]])
        else:
            spec_of(p)                                                  # raises on a typo
            out.append(p)
    return tuple(dict.fromkeys(out))


COMPONENT_PATHS = {c: expand_fields(v) for c, v in COMPONENT_FIELDS.items()}
PATHS = all_paths()
PATH_INDEX = {p: i for i, p in enumerate(PATHS)}
SPECS = tuple(spec_of(p) for p in PATHS)


def field_similarity(spec: FieldSpec, a, b) -> float | None:
    """Similarity in [0, 1] of two field values; None when either is missing."""
    if a is None or b is None:
        return None
    if spec.kind == "num":
        return math.exp(-abs(float(a) - float(b)) / spec.scale)
    if spec.kind == "ord":
        return 1.0 - spec.distance(a, b) / 2.0
    return 1.0 if a == b else 0.0


def jaccard(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Overlap of two pattern-id sets; None when both are empty (no evidence either way)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb)


# ------------------------------------------------------------------------------------------------ weights

@dataclasses.dataclass(frozen=True)
class SimilarityWeights:
    values: tuple[tuple[str, float], ...] = tuple(DEFAULT_WEIGHTS.items())
    vetoes: tuple[tuple[str, float], ...] = tuple(DEFAULT_VETOES.items())
    field_weights: tuple[tuple[str, float], ...] = ()      # per-dimension importance inside a component (default 1.0 each)
    min_component_coverage: float = 0.4     # share of a component's fields that must be observed on both sides
    min_total_coverage: float = 0.5         # share of total weight that must be available for a total to exist

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def validate(self) -> list[str]:
        errs = []
        d = self.as_dict()
        if set(d) != set(COMPONENTS):
            errs.append(f"weights must name exactly {COMPONENTS}")
        if any((not math.isfinite(v)) or v < 0 for v in d.values()):
            errs.append("weights must be finite and non-negative")
        if sum(d.values()) <= 0:
            errs.append("weights sum to zero")
        for c, f in self.vetoes:
            if c not in COMPONENTS or not 0.0 <= f <= 1.0:
                errs.append(f"bad veto {c}:{f}")
        for p, w in self.field_weights:
            if p not in PATH_INDEX or not math.isfinite(w) or w < 0:
                errs.append(f"bad field weight {p}:{w}")
        if not 0.0 < self.min_component_coverage <= 1.0 or not 0.0 < self.min_total_coverage <= 1.0:
            errs.append("coverage thresholds outside (0, 1]")
        return errs

    def normalised(self) -> dict[str, float]:
        d = self.as_dict()
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}

    def weights_id(self) -> str:
        return stable_hash({"w": self.values, "v": self.vetoes, "f": self.field_weights,
                            "c": (self.min_component_coverage, self.min_total_coverage)})

    def field_weight(self, path: str) -> float:
        return dict(self.field_weights).get(path, 1.0)

    def field_vector(self) -> np.ndarray:
        fw = dict(self.field_weights)
        return np.array([fw.get(p, 1.0) for p in PATHS], dtype=float)

    @classmethod
    def from_dict(cls, w: Mapping[str, float], **kw) -> "SimilarityWeights":
        out = cls(values=tuple((c, float(w[c])) for c in COMPONENTS), **kw)
        errs = out.validate()
        if errs:
            raise ValueError("; ".join(errs))
        return out


DEFAULT = SimilarityWeights()


# ------------------------------------------------------------------------------------------------ result

@dataclasses.dataclass(frozen=True)
class ComponentScore:
    name: str
    score: float | None                      # None = not enough observed data to say
    coverage: float                          # observed share of the component's fields (both sides)
    n_fields: int
    agree: tuple[tuple[str, float], ...] = ()      # strongest agreeing fields (path, field similarity)
    differ: tuple[tuple[str, float], ...] = ()     # strongest disagreeing fields


@dataclasses.dataclass(frozen=True)
class SimilarityResult:
    components: tuple[ComponentScore, ...]
    total: float | None
    coverage: float                          # share of total weight that was available
    vetoes: tuple[str, ...]                  # components below their veto floor
    unknown: Unknown | None
    weights_id: str

    def score(self, name: str) -> float | None:
        for c in self.components:
            if c.name == name:
                return c.score
        raise KeyError(name)

    @property
    def comparable(self) -> bool:
        return not self.vetoes and self.unknown is None

    def as_dict(self) -> dict:
        return {"total": self.total, "coverage": self.coverage, "vetoes": list(self.vetoes),
                "unknown": None if self.unknown is None else str(self.unknown),
                "components": {c.name: {"score": c.score, "coverage": c.coverage} for c in self.components}}

    def explain(self, top: int = 2) -> str:
        """Plain-language reason. States what matched, what did not, what was unknown, and any veto."""
        if self.unknown is not None:
            return f"not comparable: {self.unknown} (only {self.coverage:.0%} of the similarity weight is observable)"
        lines = [f"similarity {self.total:.3f} (coverage {self.coverage:.0%})"]
        for c in sorted((c for c in self.components if c.score is not None), key=lambda c: -c.score):
            lines.append(f"- {c.name} {c.score:.2f}")
            if c.agree:
                lines.append("    alike: " + ", ".join(f"{p} ({s:.2f})" for p, s in c.agree[:top]))
            if c.differ:
                lines.append("    unlike: " + ", ".join(f"{p} ({s:.2f})" for p, s in c.differ[:top]))
        unk = [c.name for c in self.components if c.score is None]
        if unk:
            lines.append("- unobserved components: " + ", ".join(unk))
        if self.vetoes:
            lines.append("- VETO: " + ", ".join(self.vetoes) + " too different to treat as the same situation")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ scalar comparison

def field_sims(a: Situation, b: Situation) -> dict[str, float | None]:
    """Field similarity for every dimension (None where either side is missing)."""
    return {p: field_similarity(s, a.get(p), b.get(p)) for p, s in zip(PATHS, SPECS)}


def _component(name: str, fs: Mapping[str, float | None], paths: Sequence[str], cfg: SimilarityWeights,
               extra: float | None = None) -> ComponentScore:
    vals = [(p, fs[p]) for p in paths if fs.get(p) is not None]
    n = len(paths) + (1 if extra is not None or name == "pattern" else 0)
    scores = [v for _, v in vals] + ([extra] if extra is not None else [])
    cov = len(scores) / max(n, 1)
    if not scores or cov < cfg.min_component_coverage:
        return ComponentScore(name, None, round(cov, 4), n)
    w = [cfg.field_weight(p) for p, _ in vals] + ([1.0] if extra is not None else [])
    if sum(w) <= 0:
        return ComponentScore(name, None, round(cov, 4), n)
    ordered = sorted(vals, key=lambda kv: kv[1])
    agree = tuple((p, round(v, 3)) for p, v in reversed(ordered) if v >= 0.8)[:3]
    differ = tuple((p, round(v, 3)) for p, v in ordered if v <= 0.4)[:3]
    return ComponentScore(name, round(float(np.average(scores, weights=w)), 6), round(cov, 4), n, agree, differ)


def bin_agreement(a: Situation, b: Situation) -> float | None:
    """Share of dimensions, observed on both sides, that fall in the same bucket."""
    ba, bb = a.bins(), b.bins()
    both = [p for p in PATHS if ba[p] != "na" and bb[p] != "na"]
    return None if not both else sum(ba[p] == bb[p] for p in both) / len(both)


def structural_score(a: Situation, b: Situation, fs: Mapping[str, float | None]) -> float | None:
    """Overall shape: mean of per-block similarity (so a block with many fields does not dominate) blended with bucket agreement."""
    per_block = []
    for kind, specs in BLOCK_SPECS.items():
        v = [fs[f"{kind}.{s.name}"] for s in specs if fs[f"{kind}.{s.name}"] is not None]
        if v:
            per_block.append(float(np.mean(v)))
    if not per_block:
        return None
    ba = bin_agreement(a, b)
    return 0.5 * float(np.mean(per_block)) + 0.5 * (ba if ba is not None else float(np.mean(per_block)))


def compare(a: Situation, b: Situation, weights: SimilarityWeights = DEFAULT) -> SimilarityResult:
    """Full multi-factor comparison with component scores and explanation. Symmetric in (a, b)."""
    errs = weights.validate()
    if errs:
        raise ValueError("invalid SimilarityWeights: " + "; ".join(errs))
    fs = field_sims(a, b)
    comps = []
    st = structural_score(a, b, fs)
    comps.append(ComponentScore("structural", None if st is None else round(st, 6), 1.0 if st is not None else 0.0, len(PATHS)))
    for name in COMPONENTS[1:]:
        extra = jaccard(a.pattern_ids, b.pattern_ids) if name == "pattern" else None
        comps.append(_component(name, fs, COMPONENT_PATHS[name], weights, extra))
    w = weights.normalised()
    avail = [(c, w[c.name]) for c in comps if c.score is not None]
    cov = sum(x for _, x in avail)
    unknown = None
    total = None
    if cov < weights.min_total_coverage:
        unknown = Unknown.INSUFFICIENT_DATA
    else:
        total = round(sum(c.score * x for c, x in avail) / cov, 6)
    floors = dict(weights.vetoes)
    vetoes = tuple(c.name for c in comps if c.score is not None and c.name in floors and c.score < floors[c.name])
    return SimilarityResult(tuple(comps), total, round(cov, 4), vetoes, unknown, weights.weights_id())


# ------------------------------------------------------------------------------------------------ vectorised path

class SituationMatrix:
    """Situations encoded as an (N, F) float matrix (NaN = missing) plus their pattern-id sets, for fast ranking."""

    def __init__(self, situations: Sequence[Situation]):
        self.situations = tuple(situations)
        n = len(self.situations)
        self.X = np.full((n, len(PATHS)), np.nan)
        for i, s in enumerate(self.situations):
            for j, (p, spec) in enumerate(zip(PATHS, SPECS)):
                v = s.get(p)
                if v is None:
                    continue
                self.X[i, j] = float(v) if spec.kind == "num" else float(spec.levels.index(v))
        self.patterns = tuple(frozenset(s.pattern_ids) for s in self.situations)
        self.bin_codes = np.array([[hash_bin(s.bins()[p]) for p in PATHS] for s in self.situations], dtype=np.int64) \
            if n else np.zeros((0, len(PATHS)), dtype=np.int64)

    def __len__(self):
        return len(self.situations)

    def field_sims(self, q: Situation) -> np.ndarray:
        """(N, F) similarities of every stored row to query `q`; NaN where either side is missing."""
        qv = SituationMatrix([q])
        out = np.full(self.X.shape, np.nan)
        for j, spec in enumerate(SPECS):
            a, b = self.X[:, j], qv.X[0, j]
            if math.isnan(b):
                continue
            if spec.kind == "num":
                out[:, j] = np.exp(-np.abs(a - b) / spec.scale)
            elif spec.kind == "ord":
                out[:, j] = 1.0 - np.abs(a - b) / max(len(spec.levels) - 1, 1)
            else:
                out[:, j] = (a == b).astype(float)
            out[np.isnan(a), j] = np.nan
        return out

    def component_matrix(self, q: Situation, weights: SimilarityWeights = DEFAULT) -> np.ndarray:
        """(N, 10) component scores in COMPONENTS order; NaN = component unavailable."""
        n = len(self)
        F = self.field_sims(q)
        out = np.full((n, len(COMPONENTS)), np.nan)
        if n == 0:
            return out
        qbins = SituationMatrix([q]).bin_codes[0]
        na = hash_bin("na")
        both = (self.bin_codes != na) & (qbins != na)[None, :]
        agree = ((self.bin_codes == qbins[None, :]) & both).sum(axis=1)
        nboth = both.sum(axis=1)
        block_means = []
        for kind, specs in BLOCK_SPECS.items():
            cols = [PATH_INDEX[f"{kind}.{s.name}"] for s in specs]
            sub = F[:, cols]
            cnt = (~np.isnan(sub)).sum(axis=1)
            block_means.append(np.where(cnt > 0, np.nansum(sub, axis=1) / np.maximum(cnt, 1), np.nan))
        stack = np.vstack(block_means)
        bcnt = (~np.isnan(stack)).sum(axis=0)
        bm = np.where(bcnt > 0, np.nansum(stack, axis=0) / np.maximum(bcnt, 1), np.nan)
        binfrac = np.where(nboth > 0, agree / np.maximum(nboth, 1), np.nan)
        out[:, 0] = np.where(np.isnan(bm), np.nan, 0.5 * bm + 0.5 * np.where(np.isnan(binfrac), bm, binfrac))
        fw = weights.field_vector()
        for k, name in enumerate(COMPONENTS[1:], start=1):
            cols = [PATH_INDEX[p] for p in COMPONENT_PATHS[name]]
            sub = F[:, cols]
            present = ~np.isnan(sub)
            cnt = present.sum(axis=1).astype(float)
            wcol = fw[cols][None, :]
            total_fields = float(len(cols))
            with np.errstate(all="ignore"):
                s = np.nansum(sub * wcol, axis=1)
            wsum = (present * wcol).sum(axis=1)
            if name == "pattern":
                jac = np.array([jaccard(q.pattern_ids, p) if (q.pattern_ids or p) else np.nan for p in self.patterns], dtype=float)
                has = ~np.isnan(jac)
                s = s + np.where(has, jac, 0.0)
                wsum = wsum + has
                cnt = cnt + has
                total_fields += 1.0
            cov = cnt / total_fields
            out[:, k] = np.where((cnt > 0) & (cov >= weights.min_component_coverage) & (wsum > 0), s / np.where(wsum > 0, wsum, 1), np.nan)
        return out

    def totals(self, q: Situation, weights: SimilarityWeights = DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(total, coverage, comparable) per row; total NaN where the pair is Unknown."""
        M = self.component_matrix(q, weights)
        w = weights.normalised()
        wv = np.array([w[c] for c in COMPONENTS])
        avail = ~np.isnan(M)
        cov = (avail * wv[None, :]).sum(axis=1)
        with np.errstate(all="ignore"):
            tot = np.where(cov >= weights.min_total_coverage, np.nansum(M * wv[None, :], axis=1) / np.where(cov > 0, cov, 1), np.nan)
        ok = ~np.isnan(tot)
        for c, floor in weights.vetoes:
            col = M[:, COMPONENTS.index(c)]
            ok &= ~(~np.isnan(col) & (col < floor))
        return tot, cov, ok


def hash_bin(label: str) -> int:
    """Stable small integer for a bucket label (deterministic across processes, unlike Python's salted hash())."""
    h = 1469598103934665603
    for ch in label.encode("utf-8"):
        h = ((h ^ ch) * 1099511628211) & 0x7FFFFFFFFFFFFFFF
    return h


# ------------------------------------------------------------------------------------------------ neighbours

@dataclasses.dataclass(frozen=True)
class Neighbour:
    index: int
    ref: str
    result: SimilarityResult


def nearest(query: Situation, cases: Sequence[Situation], refs: Sequence[str] | None = None, k: int = 5,
            weights: SimilarityWeights = DEFAULT, min_total: float = 0.0, require_comparable: bool = True,
            diversity: float = 0.0) -> list[Neighbour]:
    """The k most similar cases with full explanations. `diversity` in [0,1) applies an MMR-style penalty so near-duplicate
    cases (same bucket) do not fill all k places. Ties break on ref, never on input order alone."""
    if k <= 0 or len(cases) == 0:
        return []
    if not 0.0 <= diversity < 1.0:
        raise ValueError("diversity must be in [0, 1)")
    refs = [str(i) for i in range(len(cases))] if refs is None else [str(r) for r in refs]
    if len(refs) != len(cases):
        raise ValueError("refs and cases differ in length")
    M = SituationMatrix(cases)
    tot, _, ok = M.totals(query, weights)
    cand = [i for i in range(len(cases)) if not math.isnan(tot[i]) and tot[i] >= min_total and (ok[i] or not require_comparable)]
    cand.sort(key=lambda i: (-tot[i], refs[i]))
    chosen: list[int] = []
    if diversity == 0.0:
        chosen = cand[:k]
    else:
        pool = list(cand)
        while pool and len(chosen) < k:
            def mmr(i):
                pen = max((1.0 if cases[i].situation_id == cases[j].situation_id else 0.0 for j in chosen), default=0.0)
                return (1 - diversity) * tot[i] - diversity * pen
            best = None
            for i in sorted(pool, key=lambda i: refs[i]):          # ties resolve to the smaller ref
                if best is None or mmr(i) > mmr(best) + 1e-12:
                    best = i
            chosen.append(best)
            pool.remove(best)
    return [Neighbour(i, refs[i], compare(query, cases[i], weights)) for i in chosen]


# ------------------------------------------------------------------------------------------------ learning weights

@dataclasses.dataclass(frozen=True)
class WeightFit:
    weights: SimilarityWeights
    n_pairs: int
    r2_before: float
    r2_after: float
    shifts: tuple[tuple[str, float], ...]      # new - old normalised weight per component


def fit_weights(component_rows: np.ndarray, agreement: np.ndarray, prior: SimilarityWeights = DEFAULT, ridge: float = 5.0,
                min_pairs: int = 60) -> WeightFit:
    """Re-fit component weights so that similarity predicts how much two situations' outcomes agreed.
    component_rows: (P, 10) component scores of P pairs (NaN allowed, filled with the row mean of the rest);
    agreement: (P,) in [0,1] (1 = same outcome). Non-negative least squares with a ridge pull to `prior` scaled by `ridge`
    pseudo-pairs, so a small sample barely moves anything. Fewer than `min_pairs` returns the prior unchanged."""
    from scipy.optimize import nnls
    A = np.asarray(component_rows, dtype=float)
    y = np.asarray(agreement, dtype=float)
    if A.ndim != 2 or A.shape[1] != len(COMPONENTS) or len(y) != len(A):
        raise ValueError("component_rows must be (P, 10) and match agreement")
    w0 = np.array([prior.normalised()[c] for c in COMPONENTS])
    keep = np.isfinite(y)
    A, y = A[keep], y[keep]
    if len(y) < min_pairs:
        return WeightFit(prior, len(y), float("nan"), float("nan"), tuple((c, 0.0) for c in COMPONENTS))
    rowmean = np.nanmean(np.where(np.isnan(A), np.nan, A), axis=1)
    A = np.where(np.isnan(A), rowmean[:, None], A)
    A = np.where(np.isnan(A), 0.5, A)

    def r2(w):
        pred = A @ w
        pred = pred / max(w.sum(), 1e-12)
        ss = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - (pred - pred.mean() + y.mean())) ** 2).sum() / ss) if ss > 0 else float("nan")
    lam = math.sqrt(ridge) * math.sqrt(len(y))
    Aa = np.vstack([A, lam * np.eye(len(COMPONENTS)) / max(len(COMPONENTS), 1)])
    ya = np.concatenate([y, lam * w0 / max(len(COMPONENTS), 1)])
    w, _ = nnls(Aa, ya)
    if w.sum() <= 0:
        w = w0.copy()
    w = w / w.sum()
    fitted = SimilarityWeights.from_dict(dict(zip(COMPONENTS, w)), vetoes=prior.vetoes, field_weights=prior.field_weights,
                                         min_component_coverage=prior.min_component_coverage,
                                         min_total_coverage=prior.min_total_coverage)
    return WeightFit(fitted, len(y), r2(w0), r2(w), tuple((c, round(float(w[i] - w0[i]), 6)) for i, c in enumerate(COMPONENTS)))


# ------------------------------------------------------------------------------------------------ diagnostics

def component_correlation(cases: Sequence[Situation], seed: int = 0, n_pairs: int = 400,
                          weights: SimilarityWeights = DEFAULT) -> dict[tuple[str, str], float]:
    """Correlation between component scores over random pairs. Components correlated above 0.9 double-count one signal."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    if n < 3:
        return {}
    rows = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        r = compare(cases[i], cases[j], weights)
        rows.append([np.nan if c.score is None else c.score for c in r.components])
    A = np.array(rows, dtype=float)
    out = {}
    for a in range(len(COMPONENTS)):
        for b in range(a + 1, len(COMPONENTS)):
            ok = ~np.isnan(A[:, a]) & ~np.isnan(A[:, b])
            if ok.sum() >= 10 and A[ok, a].std() > 1e-9 and A[ok, b].std() > 1e-9:
                out[(COMPONENTS[a], COMPONENTS[b])] = round(float(np.corrcoef(A[ok, a], A[ok, b])[0, 1]), 4)
    return out


def consistency_report(cases: Sequence[Situation], weights: SimilarityWeights = DEFAULT, seed: int = 0, n_pairs: int = 100) -> dict:
    """Properties a similarity must have: self-similarity 1, symmetry, range [0,1], scalar/vector agreement."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    rep = {"self_min": 1.0, "asym_max": 0.0, "range_ok": True, "vector_gap_max": 0.0, "pairs": 0}
    if n == 0:
        return rep
    for c in cases[: min(n, 20)]:
        r = compare(c, c, weights)
        if r.total is not None:
            rep["self_min"] = min(rep["self_min"], r.total)
    M = SituationMatrix(cases)
    for _ in range(min(n_pairs, n * n)):
        i, j = (int(x) for x in rng.integers(0, n, size=2))
        a, b = compare(cases[i], cases[j], weights), compare(cases[j], cases[i], weights)
        if a.total is not None and b.total is not None:
            rep["asym_max"] = max(rep["asym_max"], abs(a.total - b.total))
            rep["range_ok"] &= 0.0 <= a.total <= 1.0
            tot, _, _ = M.totals(cases[i], weights)
            if not math.isnan(tot[j]):
                rep["vector_gap_max"] = max(rep["vector_gap_max"], abs(tot[j] - a.total))
        rep["pairs"] += 1
    return rep


def discrimination(cases: Sequence[Situation], outcomes: Sequence[float], seed: int = 0, n_pairs: int = 600,
                   weights: SimilarityWeights = DEFAULT) -> dict[str, float]:
    """Per component: correlation between pair similarity and closeness of pair outcomes (negative |diff|). A component
    with no relation to outcomes carries no transfer information; the caller can lower its weight via fit_weights."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    y = np.asarray(outcomes, dtype=float)
    if n < 4 or len(y) != n:
        return {}
    rows, gaps = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        r = compare(cases[i], cases[j], weights)
        rows.append([np.nan if c.score is None else c.score for c in r.components])
        gaps.append(-abs(y[i] - y[j]))
    A, g = np.array(rows), np.array(gaps)
    out = {}
    for k, c in enumerate(COMPONENTS):
        ok = ~np.isnan(A[:, k])
        if ok.sum() >= 20 and A[ok, k].std() > 1e-9 and g[ok].std() > 1e-9:
            out[c] = round(float(np.corrcoef(A[ok, k], g[ok])[0, 1]), 4)
    return out


# ------------------------------------------------------------------------------------------------ recent-history similarity

def compare_with_history(a: Situation, b: Situation, hist_a: Sequence[Situation], hist_b: Sequence[Situation],
                         weights: SimilarityWeights = DEFAULT, decay: float = 0.6, blend: float = 0.5) -> SimilarityResult:
    """Like `compare`, but the recent-history component also looks at the last few sessions of each stream (most recent
    first): the snapshot's recent-history score is blended with an exponentially decayed mean of the aligned earlier
    snapshots' recent-history scores. Two situations that look alike today but arrived by different paths score lower."""
    base = compare(a, b, weights)
    if not 0.0 <= blend <= 1.0 or not 0.0 < decay <= 1.0:
        raise ValueError("blend must be in [0, 1] and decay in (0, 1]")
    n = min(len(hist_a), len(hist_b))
    past = []
    for i in range(n):
        r = compare(hist_a[i], hist_b[i], weights)
        sc = r.score("recent_history")
        if sc is not None:
            past.append((decay ** (i + 1), sc))
    now_sc = base.score("recent_history")
    if not past:
        return base
    path_sc = sum(w * s for w, s in past) / sum(w for w, _ in past)
    merged = path_sc if now_sc is None else (1 - blend) * now_sc + blend * path_sc
    comps = tuple(dataclasses.replace(c, score=round(merged, 6), coverage=max(c.coverage, 0.5)) if c.name == "recent_history" else c
                  for c in base.components)
    w = weights.normalised()
    avail = [(c, w[c.name]) for c in comps if c.score is not None]
    cov = sum(x for _, x in avail)
    unknown = Unknown.INSUFFICIENT_DATA if cov < weights.min_total_coverage else None
    total = None if unknown else round(sum(c.score * x for c, x in avail) / cov, 6)
    floors = dict(weights.vetoes)
    vetoes = tuple(c.name for c in comps if c.score is not None and c.name in floors and c.score < floors[c.name])
    return SimilarityResult(comps, total, round(cov, 4), vetoes, unknown, weights.weights_id())


# ------------------------------------------------------------------------------------------------ contrast in words

def contrast(a: Situation, b: Situation, weights: SimilarityWeights = DEFAULT) -> str:
    """Why two situations are or are not alike, in words: the per-component verdict plus the bucket-level differences."""
    from .situation import diff
    r = compare(a, b, weights)
    d = diff(a, b)
    head = r.explain()
    if r.unknown is not None:
        return head
    lines = [head, f"{d.n_changed} of {len(PATHS)} dimensions fall in different buckets; {len(d.unknown)} unobserved on at least one side."]
    if d.changed:
        lines.append("differences: " + "; ".join(f"{p} {x}->{y}" for p, x, y in d.changed[:8]) + (" ..." if d.n_changed > 8 else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ clustering & prototypes

@dataclasses.dataclass(frozen=True)
class Cluster:
    members: tuple[int, ...]
    medoid: int
    cohesion: float                      # mean pairwise total similarity inside the cluster


def cluster_situations(cases: Sequence[Situation], threshold: float = 0.8, weights: SimilarityWeights = DEFAULT) -> list[Cluster]:
    """Leader clustering into equivalence classes: a case joins the first cluster whose leader it resembles at `threshold`
    (comparable and not vetoed), else starts its own. Deterministic in input order; the result is sorted by size then leader.
    The medoid is the member with the highest mean similarity to the rest, the natural prototype of the class."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    n = len(cases)
    if n == 0:
        return []
    M = SituationMatrix(cases)
    leaders: list[int] = []
    groups: list[list[int]] = []
    for i in range(n):
        placed = False
        if leaders:
            tot, _, ok = M.totals(cases[i], weights)
            for gi, ld in enumerate(leaders):
                if ok[ld] and tot[ld] >= threshold:
                    groups[gi].append(i)
                    placed = True
                    break
        if not placed:
            leaders.append(i)
            groups.append([i])
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(Cluster((g[0],), g[0], 1.0))
            continue
        sub = SituationMatrix([cases[i] for i in g])
        mat = np.full((len(g), len(g)), np.nan)
        for a, i in enumerate(g):
            tot, _, _ = sub.totals(cases[i], weights)
            mat[a] = tot
        np.fill_diagonal(mat, np.nan)
        with np.errstate(all="ignore"):
            means = np.nanmean(mat, axis=1)
        means = np.where(np.isnan(means), -1.0, means)
        out.append(Cluster(tuple(g), g[int(np.argmax(means))], float(np.nanmean(mat)) if np.isfinite(mat).any() else 1.0))
    out.sort(key=lambda c: (-len(c.members), c.members[0]))
    return out


# ------------------------------------------------------------------------------------------------ metric sanity

def triangle_violation_rate(cases: Sequence[Situation], weights: SimilarityWeights = DEFAULT, n_triples: int = 300,
                            seed: int = 0, slack: float = 0.0) -> dict[str, float]:
    """With distance = 1 - total, how often is d(a,c) > d(a,b) + d(b,c) + slack? A bounded-similarity built from means of
    per-field kernels is not a metric; this measures how far from one it is, so nobody relies on transitivity blindly."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    if n < 3:
        return {"triples": 0, "rate": float("nan"), "worst": float("nan")}
    viol, worst, m = 0, 0.0, 0
    for _ in range(n_triples):
        i, j, k = (int(x) for x in rng.choice(n, size=3, replace=False))
        t = [compare(cases[x], cases[y], weights).total for x, y in ((i, j), (j, k), (i, k))]
        if any(v is None for v in t):
            continue
        dab, dbc, dac = (1 - v for v in t)
        m += 1
        over = dac - (dab + dbc) - slack
        if over > 1e-12:
            viol += 1
            worst = max(worst, over)
    return {"triples": m, "rate": viol / m if m else float("nan"), "worst": worst}


# ------------------------------------------------------------------------------------------------ calibration of similarity

class SimilarityCalibrator:
    """Monotone (pool-adjacent-violators) map from total similarity to P(outcomes agree). Fit on pairs whose outcomes are
    known; `predict` then turns a raw similarity into a probability that two such situations behaved alike, so retrieval can
    say 'transfer evidence' in units of probability instead of an arbitrary index."""

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.n = 0
        self.base_rate = float("nan")

    def fit(self, totals: Sequence[float], agree: Sequence[float], min_n: int = 40) -> "SimilarityCalibrator":
        t, a = np.asarray(totals, float), np.asarray(agree, float)
        ok = np.isfinite(t) & np.isfinite(a)
        t, a = t[ok], a[ok]
        self.n = len(t)
        self.base_rate = float(a.mean()) if len(a) else float("nan")
        if len(t) < min_n:
            self.x = self.y = np.array([])
            return self
        o = np.argsort(t, kind="stable")
        t, a = t[o], a[o]
        vals, wts, xs = [], [], []
        for xv, yv in zip(t, a):                                       # PAV with block merging
            vals.append(float(yv))
            wts.append(1.0)
            xs.append(float(xv))
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                x = (xs[-2] * wts[-2] + xs[-1] * wts[-1]) / w
                vals[-2:], wts[-2:], xs[-2:] = [v], [w], [x]
        self.x, self.y = np.array(xs), np.array(vals)
        return self

    @property
    def fitted(self) -> bool:
        return len(self.x) > 0

    def predict(self, total) -> float | None:
        """None when unfitted (never a made-up probability)."""
        if not self.fitted or total is None or not math.isfinite(float(total)):
            return None
        return float(np.interp(float(total), self.x, self.y))

    def lift_over_base(self, total) -> float | None:
        p = self.predict(total)
        return None if p is None else p - self.base_rate


# ------------------------------------------------------------------------------------------------ novelty & skill

def novelty(query: Situation, cases: Sequence[Situation], weights: SimilarityWeights = DEFAULT, seed: int = 0,
            n_ref: int = 40) -> dict[str, float | bool | None]:
    """How unlike anything stored is `query`? Compares its best match with the distribution of leave-one-out best matches
    among the stored cases themselves. A query whose best match is below the 5th percentile of that reference is NOVEL:
    neighbours exist but none is really similar, and retrieval should abstain rather than hand back the least-bad ones."""
    n = len(cases)
    if n < 10:
        return {"best": None, "reference_p05": None, "novel": None, "n": n}
    M = SituationMatrix(cases)
    tot, _, ok = M.totals(query, weights)
    best = float(np.nanmax(np.where(ok, tot, np.nan))) if ok.any() else 0.0
    rng = np.random.default_rng(seed)
    ref = []
    for i in rng.choice(n, size=min(n_ref, n), replace=False):
        t, _, o = M.totals(cases[int(i)], weights)
        t = np.where(o, t, np.nan)
        t[int(i)] = np.nan
        if np.isfinite(t).any():
            ref.append(float(np.nanmax(t)))
    if len(ref) < 5:
        return {"best": best, "reference_p05": None, "novel": None, "n": n}
    p05 = float(np.quantile(ref, 0.05))
    return {"best": best, "reference_p05": p05, "novel": bool(best < p05), "n": n}


def walk_forward_skill(cases: Sequence[Situation], outcomes: Sequence[float], times: Sequence[Any], k: int = 10,
                       weights: SimilarityWeights = DEFAULT, min_history: int = 50, min_total: float = 0.0,
                       seed: int = 0) -> dict[str, Any]:
    """Does 'the k most similar PAST cases predict this case's outcome'? Cases are replayed in time order; each is predicted
    only from cases matured strictly before it (kernel-weighted by similarity, engine.analog_weighting.kernel_weights).
    Reports rank skill (Spearman), MSE skill against the trailing-mean forecast (1 - MSE/MSE0), HAC t of the squared-error
    improvement and a sign-flip p. Negative skill is reported as negative: retrieval must be allowed to FAIL and abstain."""
    from scipy.stats import spearmanr
    from .. import analog_weighting as AW
    n = len(cases)
    if n != len(outcomes) or n != len(times):
        raise ValueError("cases, outcomes and times differ in length")
    y = np.asarray(outcomes, float)
    day = np.array([as_date(t).toordinal() for t in times])
    order = np.argsort(day, kind="stable")
    M = SituationMatrix([cases[i] for i in order])
    ys, ds = y[order], day[order]
    pred, actual, base, gain = [], [], [], []
    for pos in range(n):
        past = np.where(ds < ds[pos])[0]
        if len(past) < min_history:
            continue
        tot, _, ok = M.totals(cases[order[pos]], weights)
        cand = [j for j in past if ok[j] and not math.isnan(tot[j]) and tot[j] >= min_total]
        if len(cand) < max(3, k // 2):
            continue
        cand.sort(key=lambda j: (-tot[j], j))
        top = np.array(cand[:k])
        wk = np.exp(-(1.0 - tot[top]) * 5.0)
        wk = wk / wk.sum()
        p = float((wk * ys[top]).sum())
        b = float(ys[past].mean())
        pred.append(p)
        actual.append(float(ys[pos]))
        base.append(b)
        gain.append((ys[pos] - b) ** 2 - (ys[pos] - p) ** 2)
    m = len(pred)
    if m < 20:
        return {"n": m, "spearman": float("nan"), "mse_skill": float("nan"), "hac_t": float("nan"), "p_sign_flip": float("nan"),
                "status": "INSUFFICIENT_EVIDENCE"}
    pred, actual, base = np.array(pred), np.array(actual), np.array(base)
    rho = float(spearmanr(pred, actual)[0]) if pred.std() > 1e-12 and actual.std() > 1e-12 else float("nan")
    mse0 = float(np.mean((actual - base) ** 2))
    mse1 = float(np.mean((actual - pred) ** 2))
    gain = np.array(gain)
    t = AW.hac_t(gain)
    p = AW.sign_flip_p(gain, seed=seed)
    skill = 1 - mse1 / mse0 if mse0 > 0 else float("nan")
    status = "INSUFFICIENT_EVIDENCE" if m < 40 else "FAILED" if (skill <= 0 or (np.isfinite(rho) and rho <= 0)) else \
        "PROVEN" if np.isfinite(p) and p < 0.05 else "UNPROVEN"
    return {"n": m, "spearman": rho, "mse_skill": float(skill), "hac_t": float(t), "p_sign_flip": float(p), "status": status}


# ------------------------------------------------------------------------------------------------ time-safe case index

class TemporalCaseIndex:
    """Cases with the date their outcome matured. Queries take `now`; a case whose outcome is not strictly before `now` is
    either dropped or raises (fail closed), so an index can never feed a decision an outcome from its own future."""

    def __init__(self, weights: SimilarityWeights = DEFAULT, on_future: str = "raise"):
        if on_future not in ("raise", "drop"):
            raise ValueError("on_future must be raise or drop")
        self.weights = weights
        self.on_future = on_future
        self._cases: list[tuple[Situation, Any, float | None, str, str]] = []      # (situation, matured, outcome, ref, episode)
        self._M: SituationMatrix | None = None

    def add(self, sit: Situation, matured, outcome: float | None = None, ref: str | None = None, episode: str = "") -> None:
        errs = sit.validate()
        if errs:
            raise ValueError("invalid situation: " + "; ".join(errs[:3]))
        self._cases.append((sit, matured, outcome, ref or str(len(self._cases)), episode))
        self._M = None

    def __len__(self) -> int:
        return len(self._cases)

    def _visible(self, now) -> list[int]:
        vis = []
        for i, (_, matured, _, ref, _) in enumerate(self._cases):
            if matured is None or as_date(matured) >= as_date(now):
                if self.on_future == "raise":
                    raise FirewallBreach(f"case {ref}: outcome matured {matured} is not before now={as_date(now)}")
                continue
            vis.append(i)
        return vis

    def query(self, sit: Situation, now, k: int = 5, min_total: float = 0.0, max_per_episode: int = 0) -> list[Neighbour]:
        """k most similar visible cases (comparable, at least `min_total`), with at most `max_per_episode` per episode
        label when that is positive (a week of near-identical days is one episode, not five independent cases)."""
        vis = self._visible(now)
        if not vis or k <= 0:
            return []
        if self._M is None or len(self._M) != len(self._cases):
            self._M = SituationMatrix([c[0] for c in self._cases])
        tot, _, ok = self._M.totals(sit, self.weights)
        cand = [i for i in vis if ok[i] and not math.isnan(tot[i]) and tot[i] >= min_total]
        cand.sort(key=lambda i: (-tot[i], self._cases[i][3]))
        out, per_ep = [], {}
        for i in cand:
            ep = self._cases[i][4]
            if max_per_episode > 0 and ep:
                if per_ep.get(ep, 0) >= max_per_episode:
                    continue
                per_ep[ep] = per_ep.get(ep, 0) + 1
            out.append(Neighbour(i, self._cases[i][3], compare(sit, self._cases[i][0], self.weights)))
            if len(out) == k:
                break
        return out

    def expected_outcome(self, sit: Situation, now, k: int = 10, min_total: float = 0.3) -> dict[str, float | int | None]:
        """Similarity-weighted mean outcome of the visible neighbours that have one, with the effective sample size."""
        nb = [n for n in self.query(sit, now, k=k, min_total=min_total) if self._cases[n.index][2] is not None]
        if not nb:
            return {"mean": None, "n": 0, "n_eff": 0.0}
        w = np.array([n.result.total for n in nb], float)
        y = np.array([self._cases[n.index][2] for n in nb], float)
        return {"mean": float((w * y).sum() / w.sum()), "n": len(nb), "n_eff": float(w.sum() ** 2 / (w ** 2).sum())}


def pair_dataset(cases: Sequence[Situation], outcomes: Sequence[float], n_pairs: int = 500, seed: int = 0,
                 weights: SimilarityWeights = DEFAULT, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(component rows, agreement) for `fit_weights` / the calibrator: agreement = exp(-|y_i - y_j| / scale), scale defaulting
    to the median absolute outcome difference so 'agree' means closer than a typical pair."""
    rng = np.random.default_rng(seed)
    n = len(cases)
    y = np.asarray(outcomes, float)
    if n < 3 or len(y) != n:
        return np.zeros((0, len(COMPONENTS))), np.zeros(0)
    idx = [tuple(int(v) for v in rng.choice(n, size=2, replace=False)) for _ in range(n_pairs)]
    diffs = np.array([abs(y[i] - y[j]) for i, j in idx])
    sc = scale if scale is not None else max(float(np.median(diffs)), 1e-12)
    rows = np.array([[np.nan if c.score is None else c.score for c in compare(cases[i], cases[j], weights).components] for i, j in idx])
    return rows, np.exp(-diffs / sc)
