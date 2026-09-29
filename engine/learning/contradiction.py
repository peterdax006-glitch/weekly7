"""Contradiction investigation (contract C62 section 18; checklist F02, F12) - IMPLEMENTED, NOT VALIDATED.

Two pieces of knowledge disagree. The one thing this module refuses to do is average them: an average of "up in bull markets"
and "down in bear markets" is a claim about nothing. Instead it asks the section-18 question - WHICH CONTEXT EXPLAINS THE
DISAGREEMENT? - and answers it with a search that cannot fool itself:

  detect        classify the disagreement (sign conflict / magnitude conflict / disjoint scopes / none) with cluster-robust
                statistics (observations on the same date are one unit of evidence, not many) and shared-date covariance.
  investigate   for each context dimension, split both claims' evidence by level and test whether the A-B disagreement
                CHANGES across levels (Cochran Q). The search runs on the EARLY half of time only, pays a multiple-testing
                bar (Bonferroni over every split tried now AND in earlier investigations of the pair - the cumulative ledger),
                then the single chosen split must replicate on the LATE half with the same sign pattern. A split that fits
                the past but not the later half is reported SPURIOUS_SPLIT, never kept.
  verdicts      RESOLVED_BY_CONTEXT, PARTLY_RESOLVED, UNRESOLVED (keep both, mark CONTRADICTED), SPURIOUS_SPLIT,
                INSUFFICIENT_DATA, NO_DISAGREEMENT. `Investigation.pooled()` raises ContradictionError for any conflict.
  ledger        hash-chained history of investigations (archive.ChainFile), status per pair, cumulative test counts,
                re-open when new evidence has matured, and the epistemic state each knowledge id should carry.
  plan          when data is too thin: which contexts to stratify and how many independent dates each level needs.

Everything is time-aware: rows dated at/after `now` raise FirewallBreach (fail closed, rule 17)."""
from __future__ import annotations

import dataclasses
import itertools
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_stats as _ps
from .archive import ArchiveError, ChainFile
from .core import (Edge, Epistemic, FailureCause, FirewallBreach, Unknown, _StrEnum, as_date, canonical_json,
                   stable_hash)
from .knowledge_graph import EdgeProposal

RESERVED = ("source", "effect", "when", "cluster", "weight")


class ContradictionError(ArchiveError):
    pass


class DisagreementKind(_StrEnum):
    NONE = "NONE"
    SIGN_CONFLICT = "SIGN_CONFLICT"
    MAGNITUDE_CONFLICT = "MAGNITUDE_CONFLICT"
    SCOPE_ONLY = "SCOPE_ONLY"                 # the claims speak about different contexts: not a real contradiction
    INSUFFICIENT = "INSUFFICIENT"


class Verdict(_StrEnum):
    NO_DISAGREEMENT = "NO_DISAGREEMENT"
    RESOLVED_BY_CONTEXT = "RESOLVED_BY_CONTEXT"
    PARTLY_RESOLVED = "PARTLY_RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    SPURIOUS_SPLIT = "SPURIOUS_SPLIT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class Status(_StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    PARTLY_RESOLVED = "PARTLY_RESOLVED"
    UNRESOLVED_KEEP_BOTH = "UNRESOLVED_KEEP_BOTH"
    NEEDS_DATA = "NEEDS_DATA"
    CLOSED = "CLOSED"


PARAMS = {
    "alpha": 0.05,               # bar after multiple-testing correction
    "z_crit": 1.96,              # a level "still disagrees" at this |z|
    "min_clusters": 12,          # independent dates per source before anything is claimed
    "min_level_clusters": 6,     # independent dates per (source, level) for a level to enter a split
    "n_bins": 3,                 # quantile bins for numeric contexts
    "numeric_unique": 7,         # numeric column with more unique values than this is binned
    "concord_share": 0.75,       # share of informative levels whose sign must replicate out of sample
    "interaction_top_k": 3,      # dims combined pairwise if no single split passes
    "confirm_alpha": 0.05,       # single pre-specified test on the late half
}


# ------------------------------------------------------------------------------------------- statistics

@dataclasses.dataclass(frozen=True)
class ClusterStat:
    mean: float
    se: float
    clusters: int
    n: int
    sw: float

    def z(self) -> float:
        return self.mean / self.se if self.se and math.isfinite(self.se) and self.se > 0 else 0.0


def _cluster_resid(y: np.ndarray, w: np.ndarray, c: np.ndarray) -> tuple[float, float, dict]:
    sw = float(w.sum())
    m = float((w * y).sum() / sw)
    r: dict[Any, float] = defaultdict(float)
    for yi, wi, ci in zip(y, w, c):
        r[ci] += wi * (yi - m)
    return m, sw, r


def cluster_mean(y, w=None, cluster=None) -> ClusterStat:
    """Weighted mean with a cluster-robust standard error (one cluster = one date). se = inf with fewer than 2 clusters:
    a single date is not evidence of a variance."""
    y = np.asarray(y, float)
    if y.size == 0:
        return ClusterStat(float("nan"), float("inf"), 0, 0, 0.0)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    c = np.arange(y.size) if cluster is None else np.asarray(cluster)
    if np.any(w < 0) or w.sum() <= 0:
        raise ContradictionError("weights must be non-negative with a positive sum")
    m, sw, r = _cluster_resid(y, w, c)
    G = len(r)
    if G < 2:
        return ClusterStat(m, float("inf"), G, y.size, sw)
    var = sum(v * v for v in r.values()) / (sw * sw) * G / (G - 1)
    return ClusterStat(m, math.sqrt(max(var, 0.0)), G, y.size, sw)


def cluster_diff(ya, wa, ca, yb, wb, cb) -> tuple[float, float, int]:
    """(mean_a - mean_b, se, clusters in union). When both samples touch the same dates their errors covary; the
    cluster-summed residuals of the two means are differenced per cluster, so shared-date noise cancels correctly."""
    ya, yb = np.asarray(ya, float), np.asarray(yb, float)
    if ya.size == 0 or yb.size == 0:
        return float("nan"), float("inf"), 0
    ma, swa, ra = _cluster_resid(ya, np.asarray(wa, float), np.asarray(ca))
    mb, swb, rb = _cluster_resid(yb, np.asarray(wb, float), np.asarray(cb))
    keys = set(ra) | set(rb)
    G = len(keys)
    if G < 2:
        return ma - mb, float("inf"), G
    var = sum((ra.get(k, 0.0) / swa - rb.get(k, 0.0) / swb) ** 2 for k in keys) * G / (G - 1)
    return ma - mb, math.sqrt(max(var, 0.0)), G


def heterogeneity(diffs: Sequence[float], ses: Sequence[float]) -> tuple[float, int, float]:
    """Cochran's Q across levels: do the A-B differences differ between levels? -> (Q, df, p)."""
    d = np.asarray(diffs, float)
    s = np.asarray(ses, float)
    ok = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s = d[ok], s[ok]
    if d.size < 2:
        return 0.0, 0, 1.0
    w = 1.0 / s ** 2
    dbar = float((w * d).sum() / w.sum())
    q = float((w * (d - dbar) ** 2).sum())
    return q, int(d.size - 1), float(sps.chi2.sf(q, d.size - 1))


def bonferroni(p: float, m: int) -> float:
    """Bonferroni over `m` tried splits, delegating to engine.pattern_stats (one implementation of multiple testing)."""
    return float(_ps.bonferroni([float(p)], max(int(m), 1))[0])


def power_needed(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Independent dates per arm to detect a mean difference `delta` with per-date sd `sigma` (two-sample, two-sided)."""
    if delta <= 0 or sigma <= 0:
        raise ContradictionError("sigma and delta must be positive")
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    return int(math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / delta ** 2))


def bin_numeric(values: pd.Series, edges: np.ndarray | None = None, k: int = 3) -> tuple[pd.Series, np.ndarray]:
    """Quantile bins; the edges come from a reference sample so later data is binned by the SAME thresholds."""
    v = pd.to_numeric(values, errors="coerce")
    if edges is None:
        qs = np.linspace(0, 1, k + 1)[1:-1]
        edges = np.unique(np.nanquantile(v.to_numpy(float), qs))
    idx = np.digitize(v.to_numpy(float), edges)
    labels = pd.Series([f"q{int(i) + 1}" for i in idx], index=values.index)
    labels[v.isna()] = "na"
    return labels, np.asarray(edges, float)


# ------------------------------------------------------------------------------------------- claims

@dataclasses.dataclass(frozen=True)
class Claim:
    """One side of a disagreement, summarised: signed effect, its standard error, sample, and the scope it claims."""
    knowledge_id: str
    effect: float
    se: float
    n: int
    contexts: tuple[tuple[str, str], ...] = ()

    def check(self) -> list[str]:
        errs = []
        if not math.isfinite(self.effect):
            errs.append("effect not finite")
        if not (math.isfinite(self.se) and self.se > 0):
            errs.append("se must be positive and finite")
        if self.n < 0:
            errs.append("negative n")
        return errs


@dataclasses.dataclass(frozen=True)
class Disagreement:
    a: str
    b: str
    kind: DisagreementKind
    diff: float
    se: float
    z: float
    p: float
    scope_overlap: bool
    shared_scope: tuple[tuple[str, str], ...]
    note: str = ""


def scopes_overlap(a: Mapping, b: Mapping) -> tuple[bool, tuple]:
    """Scopes overlap unless they name the same dimension with different values. Returns (overlap, shared conditions)."""
    shared = []
    for k in set(a) & set(b):
        if str(a[k]) != str(b[k]):
            return False, ()
        shared.append((k, str(a[k])))
    return True, tuple(sorted(shared))


def detect(a: Claim, b: Claim, alpha: float = 0.05, min_n: int = 10) -> Disagreement:
    """Do two claims actually disagree, or do they just talk about different places? Summary-level test."""
    for c in (a, b):
        e = c.check()
        if e:
            raise ContradictionError(f"claim {c.knowledge_id}: " + "; ".join(e))
    overlap, shared = scopes_overlap(dict(a.contexts), dict(b.contexts))
    diff = a.effect - b.effect
    se = math.sqrt(a.se ** 2 + b.se ** 2)
    z = diff / se
    p = float(2 * sps.norm.sf(abs(z)))
    if not overlap:
        return Disagreement(a.knowledge_id, b.knowledge_id, DisagreementKind.SCOPE_ONLY, diff, se, z, p, False, (),
                            "different contexts on a shared dimension: not a contradiction, two scoped claims")
    if a.n < min_n or b.n < min_n:
        return Disagreement(a.knowledge_id, b.knowledge_id, DisagreementKind.INSUFFICIENT, diff, se, z, p, True, shared,
                            f"n below {min_n}")
    if p >= alpha:
        return Disagreement(a.knowledge_id, b.knowledge_id, DisagreementKind.NONE, diff, se, z, p, True, shared)
    za, zb = a.effect / a.se, b.effect / b.se
    opposite = a.effect * b.effect < 0 and abs(za) >= 1.0 and abs(zb) >= 1.0
    kind = DisagreementKind.SIGN_CONFLICT if opposite else DisagreementKind.MAGNITUDE_CONFLICT
    return Disagreement(a.knowledge_id, b.knowledge_id, kind, diff, se, z, p, True, shared)


def claims_from_evidence(ev: EvidenceSet, sources: Sequence[str] | None = None) -> list[Claim]:
    """Summarise each source's evidence as a Claim (cluster-robust mean and se, n = independent dates), scoped by the
    context columns that are constant within the source (a source collected entirely in one regime says so)."""
    out = []
    for src in sources or ev.sources():
        g = ev.frame[ev.frame["source"].astype(str) == src]
        st = cluster_mean(g["effect"], g["weight"], g["cluster"])
        scope = tuple(sorted((c, str(g[c].iloc[0])) for c in ev.context_cols if c in g.columns and g[c].astype(str).nunique() == 1))
        out.append(Claim(str(src), st.mean, st.se if math.isfinite(st.se) and st.se > 0 else float("nan"), st.clusters, scope))
    return out


def disagreement_map(claims: Sequence[Claim]) -> pd.DataFrame:
    """Pairwise z of the A-B difference for a set of claims (symmetric matrix, 0 on the diagonal, NaN where the scopes
    cannot be compared), for a glance at where a knowledge base disagrees with itself."""
    ids = [c.knowledge_id for c in claims]
    m = pd.DataFrame(0.0, index=ids, columns=ids)
    for a, b in itertools.combinations(claims, 2):
        d = detect(a, b)
        v = float("nan") if d.kind == DisagreementKind.SCOPE_ONLY else d.z
        m.loc[a.knowledge_id, b.knowledge_id] = v
        m.loc[b.knowledge_id, a.knowledge_id] = -v if math.isfinite(v) else v
    return m


def triage(claims: Sequence[Claim], alpha: float = 0.05) -> list[Disagreement]:
    """All genuine disagreements among claims, worst first (sign conflicts before magnitude, then |z|)."""
    rank = {DisagreementKind.SIGN_CONFLICT: 0, DisagreementKind.MAGNITUDE_CONFLICT: 1}
    out = [d for a, b in itertools.combinations(claims, 2) for d in [detect(a, b, alpha)] if d.kind in rank]
    return sorted(out, key=lambda d: (rank[d.kind], -abs(d.z), d.a, d.b))


# ------------------------------------------------------------------------------------------- evidence

class EvidenceSet:
    """Rows of realised effects tagged by source (which claim's evidence), date, and context columns. `cluster` defaults to
    the date so that many names on one day count as one independent observation."""

    def __init__(self, frame: pd.DataFrame, context_cols: Sequence[str] | None = None):
        f = frame.copy()
        miss = [c for c in ("source", "effect", "when") if c not in f.columns]
        if miss:
            raise ContradictionError(f"evidence missing columns {miss}")
        f["when"] = pd.to_datetime(f["when"])
        if "cluster" not in f.columns:
            f["cluster"] = f["when"].dt.strftime("%Y-%m-%d")
        if "weight" not in f.columns:
            f["weight"] = 1.0
        f["effect"] = pd.to_numeric(f["effect"], errors="coerce")
        bad = ~np.isfinite(f["effect"].to_numpy(float))
        if bad.any():
            raise ContradictionError(f"{int(bad.sum())} non-finite effects in evidence")
        if (f["weight"] < 0).any():
            raise ContradictionError("negative weights")
        self.frame = f.reset_index(drop=True)
        self.context_cols = tuple(context_cols) if context_cols is not None else tuple(
            c for c in f.columns if c not in RESERVED)

    def __len__(self) -> int:
        return len(self.frame)

    def sources(self) -> list[str]:
        return sorted(self.frame["source"].astype(str).unique())

    def through(self) -> str | None:
        return None if not len(self) else self.frame["when"].max().date().isoformat()

    def coverage_matrix(self, col: str, a: str, b: str) -> pd.DataFrame:
        """Independent dates per (level of `col`) x (source): where each claim's evidence actually sits. A composition
        confound shows up here as a lopsided table before any statistics are run."""
        f = self.pair(a, b)
        if col not in f.columns:
            raise ContradictionError(f"no context column {col!r}")
        return f.groupby([f[col].astype(str), f["source"].astype(str)])["cluster"].nunique().unstack(fill_value=0)

    def assert_before(self, now):
        n = pd.Timestamp(as_date(now))
        late = self.frame["when"] >= n
        if late.any():
            raise FirewallBreach(f"{int(late.sum())} evidence rows dated at/after now={as_date(now)}")

    def pair(self, a: str, b: str) -> pd.DataFrame:
        return self.frame[self.frame["source"].astype(str).isin([a, b])]

    def time_split(self, a: str, b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Early/late halves by DATE (never splitting one date across halves)."""
        f = self.pair(a, b)
        days = np.sort(f["when"].unique())
        if len(days) < 2:
            return f, f.iloc[0:0]
        cut = days[len(days) // 2]
        return f[f["when"] < cut], f[f["when"] >= cut]


def _stat(f: pd.DataFrame, source: str) -> ClusterStat:
    g = f[f["source"].astype(str) == source]
    return cluster_mean(g["effect"], g["weight"], g["cluster"])


def _diff(f: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    ga, gb = f[f["source"].astype(str) == a], f[f["source"].astype(str) == b]
    return cluster_diff(ga["effect"], ga["weight"], ga["cluster"], gb["effect"], gb["weight"], gb["cluster"])


# ------------------------------------------------------------------------------------------- investigation results

@dataclasses.dataclass(frozen=True)
class LevelResult:
    level: str
    effect_a: float
    effect_b: float
    diff: float
    se: float
    z: float
    clusters_a: int
    clusters_b: int

    def disagrees(self, z_crit: float) -> bool:
        return math.isfinite(self.se) and abs(self.z) >= z_crit


@dataclasses.dataclass(frozen=True)
class ScopedClaim:
    knowledge_id: str
    contexts: tuple[tuple[str, str], ...]
    effect: float
    se: float
    clusters: int


@dataclasses.dataclass(frozen=True)
class Investigation:
    a: str
    b: str
    now: str
    verdict: Verdict
    kind: DisagreementKind
    overall_a: float
    overall_b: float
    overall_diff: float
    overall_z: float
    dimension: tuple[str, ...] = ()
    levels: tuple[LevelResult, ...] = ()
    p_raw: float = 1.0
    p_adj: float = 1.0
    m_tests: int = 0
    p_confirm: float | None = None
    concordance: float | None = None
    residual_levels: tuple[str, ...] = ()
    scoped: tuple[ScopedClaim, ...] = ()
    cause_hint: str = FailureCause.UNKNOWN.value
    temporal_note: str = ""
    rows_through: str = ""
    tried: tuple[tuple[str, float], ...] = ()       # every split tried on the search half: (dimension, raw p)
    mechanism: str = ""                             # EFFECT_MODIFICATION (diff varies by level) or COMPOSITION (mix differs)
    z_adjusted: float | None = None                 # stratified A-B difference after removing the context mix
    p_composition: float | None = None

    @property
    def pair(self) -> tuple[str, str]:
        return cast(tuple[str, str], tuple(sorted((self.a, self.b))))

    @property
    def conflicted(self) -> bool:
        return self.verdict not in (Verdict.NO_DISAGREEMENT, Verdict.RESOLVED_BY_CONTEXT)

    def pooled(self) -> float:
        """The average of two claims that disagree. Deliberately unavailable."""
        if self.verdict == Verdict.NO_DISAGREEMENT:
            return 0.5 * (self.overall_a + self.overall_b)
        raise ContradictionError(f"{self.a} vs {self.b} disagree ({self.verdict.value}): refusing to average; use the "
                                 "context-scoped claims")

    def disagreement_contexts(self, z_crit: float = PARAMS["z_crit"]) -> dict[str, list[str]]:
        dim = "x".join(self.dimension)
        return {dim: [l.level for l in self.levels if l.disagrees(z_crit)]} if self.dimension else {}

    def agreement_contexts(self, z_crit: float = PARAMS["z_crit"]) -> dict[str, list[str]]:
        dim = "x".join(self.dimension)
        return {dim: [l.level for l in self.levels if not l.disagrees(z_crit)]} if self.dimension else {}

    def estimate_in_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """The sanctioned replacement for an average: what each claim says INSIDE the named context. Only levels the
        investigation actually measured are answered; anything else is UNKNOWN, never an interpolation. `contested` says
        whether the two claims still disagree there."""
        if not self.dimension or not self.levels:
            return {"state": Unknown.UNKNOWN.value, "reason": "no context split was established for this pair"}
        try:
            level = "|".join(str(context[d]) for d in self.dimension)
        except KeyError as e:
            return {"state": Unknown.INSUFFICIENT_DATA.value, "reason": f"context lacks {e.args[0]!r}"}
        hit = next((l for l in self.levels if l.level == level), None)
        if hit is None:
            return {"state": Unknown.INSUFFICIENT_DATA.value, "reason": f"level {level!r} was never measured"}
        return {"state": None, self.a: hit.effect_a, self.b: hit.effect_b, "diff": hit.diff, "z": hit.z,
                "contested": hit.disagrees(PARAMS["z_crit"]), "level": hit.level}

    def epistemic(self) -> Epistemic:
        """The state the two claims should carry after this investigation."""
        return {Verdict.NO_DISAGREEMENT: Epistemic.SUPPORTED, Verdict.RESOLVED_BY_CONTEXT: Epistemic.CONDITIONAL,
                Verdict.PARTLY_RESOLVED: Epistemic.CONTRADICTED, Verdict.UNRESOLVED: Epistemic.CONTRADICTED,
                Verdict.SPURIOUS_SPLIT: Epistemic.CONTRADICTED, Verdict.INSUFFICIENT_DATA: Epistemic.UNKNOWN}[self.verdict]

    def next_step(self) -> str:
        """What the loop should do about this pair next (the research hand-off), as one plain sentence."""
        steps = {
            Verdict.NO_DISAGREEMENT: "nothing: the claims agree; pooling them is allowed",
            Verdict.RESOLVED_BY_CONTEXT: f"scope both claims to {'x'.join(self.dimension) or 'the found context'} and re-test out of sample",
            Verdict.PARTLY_RESOLVED: f"investigate inside {', '.join(self.residual_levels) or 'the leftover levels'} (residual_scan) "
                                     "or gate both claims out of the contested context",
            Verdict.UNRESOLVED: "keep both claims, mark them CONTRADICTED, and collect data stratified by the candidate contexts",
            Verdict.SPURIOUS_SPLIT: "discard the split; it did not replicate later in time; gather more dates before searching again",
            Verdict.INSUFFICIENT_DATA: "collect more independent dates for both claims (see Investigator.plan)",
        }
        return steps[self.verdict]

    def summary_line(self) -> str:
        where = f" via {'x'.join(self.dimension)} ({self.mechanism.lower()}, p_adj {self.p_adj:.2g})" if self.dimension else ""
        return f"{self.a} vs {self.b}: {self.verdict.value}{where}; overall z {self.overall_z:+.1f}"

    def average_hides(self) -> dict[str, float]:
        """How wrong the forbidden average would be: distance from the pooled value to each level's two claims."""
        if not self.levels:
            return {}
        pooled = 0.5 * (self.overall_a + self.overall_b)
        gaps = [max(abs(l.effect_a - pooled), abs(l.effect_b - pooled)) for l in self.levels]
        flip = any(l.effect_a * l.effect_b < 0 for l in self.levels)
        return {"pooled": float(pooled), "max_gap_to_level_claims": float(max(gaps)), "sign_flip_levels": float(flip)}

    def edge_proposals(self) -> list[EdgeProposal]:
        """CONTRADICTS (carrying the explaining context), plus COMPLEMENTS when the split resolves it cleanly."""
        attrs = (("investigated", True), ("verdict", self.verdict.value), ("p_adj", round(self.p_adj, 6)),
                 ("context", canonical_json(self.disagreement_contexts())), ("dimension", "x".join(self.dimension)))
        out = []
        if self.verdict != Verdict.NO_DISAGREEMENT:
            out.append(EdgeProposal(self.a, self.b, Edge.CONTRADICTS.value, 1.0 - min(self.p_adj, 1.0), (), attrs))
        if self.verdict == Verdict.RESOLVED_BY_CONTEXT:
            out.append(EdgeProposal(self.a, self.b, Edge.COMPLEMENTS.value, 1.0 - min(self.p_adj, 1.0), (), attrs))
        return out

    def to_dict(self) -> dict:
        """JSON-safe dossier (NaN kept as text so nothing silently becomes 0). `from_dict` restores it exactly."""
        def num(x):
            return x if not isinstance(x, float) or math.isfinite(x) else str(x)
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["verdict"], d["kind"] = self.verdict.value, self.kind.value
        d["levels"] = [{k: num(v) for k, v in dataclasses.asdict(l).items()} for l in self.levels]
        d["scoped"] = [{**{k: num(v) for k, v in dataclasses.asdict(c).items()}, "contexts": [list(x) for x in c.contexts]}
                       for c in self.scoped]
        d["dimension"], d["residual_levels"] = list(self.dimension), list(self.residual_levels)
        d["tried"] = [list(t) for t in self.tried]
        return {k: num(v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Investigation":
        def f(x):
            return float(x) if isinstance(x, str) and x in ("nan", "inf", "-inf") else x
        levels = tuple(LevelResult(**{k: f(v) for k, v in l.items()}) for l in d["levels"])
        scoped = tuple(ScopedClaim(**{**{k: f(v) for k, v in c.items() if k != "contexts"},
                                      "contexts": tuple(tuple(x) for x in c["contexts"])}) for c in d["scoped"])
        kw = {k: f(v) for k, v in d.items() if k not in ("levels", "scoped")}
        kw.update(verdict=Verdict(d["verdict"]), kind=DisagreementKind(d["kind"]), levels=levels, scoped=scoped,
                  dimension=tuple(d["dimension"]), residual_levels=tuple(d["residual_levels"]),
                  tried=tuple((a, b) for a, b in d["tried"]))
        return cls(**kw)

    def knowledge_updates(self) -> dict[str, dict[str, dict[str, str]]]:
        """What each side's knowledge object should record so the disagreement stops being averaged away: contexts where
        the two claims AGREE are shared contexts; contexts where they still disagree go to BOTH as contested
        anti-contexts until an experiment settles them. Empty when nothing was localised."""
        if not self.dimension or not self.levels:
            return {}
        key = "|".join(self.dimension)
        agree = {key: sorted(l.level for l in self.levels if not l.disagrees(PARAMS["z_crit"]))}
        clash = {key: sorted(l.level for l in self.levels if l.disagrees(PARAMS["z_crit"]))}
        out = {}
        for kid in (self.a, self.b):
            out[kid] = {"contexts": {k: "/".join(v) for k, v in agree.items() if v},
                        "anti_contexts": {k: "/".join(v) for k, v in clash.items() if v}}
        return out

    def markdown(self) -> str:
        head = [f"### {self.a} vs {self.b} @ {self.now}: {self.verdict.value}",
                f"overall {self.overall_a:+.4f} vs {self.overall_b:+.4f} (diff {self.overall_diff:+.4f}, z {self.overall_z:+.2f})"]
        if self.dimension:
            head.append(f"explaining context: {' x '.join(self.dimension)}; p_raw {self.p_raw:.2e}, p_adj {self.p_adj:.3f} "
                        f"over {self.m_tests} splits; late-half p {self.p_confirm}; concordance {self.concordance}")
            head += ["", "| level | A | B | diff | z |", "|---|---:|---:|---:|---:|"]
            head += [f"| {l.level} | {l.effect_a:+.4f} | {l.effect_b:+.4f} | {l.diff:+.4f} | {l.z:+.2f} |" for l in self.levels]
        if self.temporal_note:
            head.append(f"note: {self.temporal_note}")
        return "\n".join(head)


@dataclasses.dataclass(frozen=True)
class DimScore:
    """One candidate split scored on the search half."""
    dims: tuple[str, ...]
    table: list
    p_het: float
    p_comp: float
    p: float                     # min of the two explanations' p-values (each already a single test)
    mechanism: str
    z_overall: float
    z_adj: float


# ------------------------------------------------------------------------------------------- the investigator

@dataclasses.dataclass(frozen=True)
class PermutationNull:
    """Family-wise calibration of the context search. T = the smallest p-value over every dimension tried (the statistic a
    'best of many' explanation is judged by). The null world keeps every date's noise and the A-B gap but re-deals whole
    dates' context vectors, destroying any context/effect association."""
    dims: tuple[str, ...]
    reps: int
    observed_p: float
    p_family: float                   # (1 + #null T <= observed) / (reps + 1): valid whatever the dependence between dims
    threshold: float                  # the alpha-quantile of the null T: a split must beat this raw p to count
    fwer_at_bonferroni: float         # how often the Bonferroni bar would have (wrongly) fired in the null worlds
    null_quantiles: tuple[float, ...] # 1%, 5%, 25%, 50% of null T

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_family <= alpha


class Investigator:
    def __init__(self, params: Mapping | None = None):
        self.p = {**PARAMS, **(params or {})}

    # ---- level assignment
    def _levels(self, f: pd.DataFrame, dims: tuple[str, ...], edges: dict) -> pd.Series:
        parts = []
        for d in dims:
            col = f[d]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > self.p["numeric_unique"]:
                lab, _ = bin_numeric(col, edges.get(d), self.p["n_bins"])
            else:
                lab = col.astype(str)
            parts.append(lab.astype(str))
        out = parts[0]
        for extra in parts[1:]:
            out = out + "|" + extra
        return out

    def _edges(self, f: pd.DataFrame, cols: Sequence[str]) -> dict:
        edges = {}
        for d in cols:
            col = f[d]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > self.p["numeric_unique"]:
                _, edges[d] = bin_numeric(col, None, self.p["n_bins"])
        return edges

    def _table(self, f: pd.DataFrame, a: str, b: str, dims: tuple[str, ...], edges: dict, min_clusters: int
               ) -> list[LevelResult]:
        lv = self._levels(f, dims, edges)
        out = []
        for level in sorted(lv.unique()):
            g = f[lv == level]
            ga, gb = g[g["source"].astype(str) == a], g[g["source"].astype(str) == b]
            if ga["cluster"].nunique() < min_clusters or gb["cluster"].nunique() < min_clusters:
                continue
            d, se, _ = _diff(g, a, b)
            z = d / se if math.isfinite(se) and se > 0 else 0.0
            out.append(LevelResult(str(level), float(ga["effect"].mean()), float(gb["effect"].mean()), d, se, z,
                                   int(ga["cluster"].nunique()), int(gb["cluster"].nunique())))
        return out

    def _split_p(self, table: list[LevelResult]) -> tuple[float, float]:
        q, df, p = heterogeneity([l.diff for l in table], [l.se for l in table])
        return q, p

    @staticmethod
    def _stratified(table: list[LevelResult]) -> tuple[float, float]:
        """Inverse-variance pooled within-level A-B difference (the disagreement left once context mix is held fixed)."""
        ok = [l for l in table if math.isfinite(l.se) and l.se > 0]
        if not ok:
            return 0.0, float("inf")
        w = np.array([1.0 / l.se ** 2 for l in ok])
        d = float((w * np.array([l.diff for l in ok])).sum() / w.sum())
        return d, float(math.sqrt(1.0 / w.sum()))

    def _score(self, f: pd.DataFrame, a: str, b: str, dims: tuple[str, ...], edges: dict, min_lv: int) -> DimScore | None:
        """Two separate ways a context can explain a disagreement, each with its own p-value:
        EFFECT_MODIFICATION - the A-B difference itself changes across levels (Cochran Q);
        COMPOSITION         - the overall difference is significant, but inside each level it vanishes AND the two claims'
                              evidence sits in different mixes of levels (Simpson): chi-square on cluster counts."""
        tab = self._table(f, a, b, dims, edges, min_lv)
        if len(tab) < 2:
            return None
        _, p_het = self._split_p(tab)
        lv = self._levels(f, dims, edges)
        keep = set(l.level for l in tab)
        sub = f[lv.isin(keep)]
        d_all, se_all, _ = _diff(sub, a, b)
        z_all = d_all / se_all if math.isfinite(se_all) and se_all > 0 else 0.0
        d_adj, se_adj = self._stratified(tab)
        z_adj = d_adj / se_adj if math.isfinite(se_adj) and se_adj > 0 else 0.0
        counts = np.array([[l.clusters_a for l in tab], [l.clusters_b for l in tab]], float)
        p_comp = float(sps.chi2_contingency(counts)[1]) if counts.shape[1] >= 2 and (counts.sum(axis=0) > 0).all() else 1.0
        explains = abs(z_all) >= self.p["z_crit"] and abs(z_adj) < self.p["z_crit"] and abs(z_adj) < 0.5 * abs(z_all)
        p_c_eff = p_comp if explains else 1.0
        mech = "COMPOSITION" if p_c_eff < p_het else "EFFECT_MODIFICATION"
        return DimScore(dims, tab, p_het, p_c_eff, min(p_het, p_c_eff), mech, z_all, z_adj)

    # ---- main entry
    def investigate(self, a: str, b: str, ev: EvidenceSet, now, prior_tries: int = 0,
                    context_cols: Sequence[str] | None = None) -> Investigation:
        ev.assert_before(now)
        P = self.p
        f = ev.pair(a, b)
        n_iso = as_date(now).isoformat()
        sa, sb = _stat(f, a), _stat(f, b)
        base: dict[str, Any] = dict(a=a, b=b, now=n_iso, overall_a=sa.mean if sa.clusters else float("nan"),
                    overall_b=sb.mean if sb.clusters else float("nan"), rows_through=ev.through() or "")
        if sa.clusters < P["min_clusters"] or sb.clusters < P["min_clusters"]:
            return Investigation(verdict=Verdict.INSUFFICIENT_DATA, kind=DisagreementKind.INSUFFICIENT,
                                 overall_diff=float("nan"), overall_z=0.0, **base)
        d, se, _ = _diff(f, a, b)
        z = d / se if math.isfinite(se) and se > 0 else 0.0
        opposite = sa.mean * sb.mean < 0 and abs(sa.z()) >= 1 and abs(sb.z()) >= 1
        kind = (DisagreementKind.NONE if abs(z) < P["z_crit"] else
                DisagreementKind.SIGN_CONFLICT if opposite else DisagreementKind.MAGNITUDE_CONFLICT)
        base.update(overall_diff=d, overall_z=z, kind=kind)
        note = self._temporal_note(ev, a, b)
        if kind == DisagreementKind.NONE:
            return Investigation(verdict=Verdict.NO_DISAGREEMENT, temporal_note=note, **base)
        cols = tuple(context_cols if context_cols is not None else ev.context_cols)
        cols = tuple(c for c in cols if c in f.columns and c not in RESERVED)
        early, late = ev.time_split(a, b)
        min_lv = P["min_level_clusters"]
        edges = self._edges(early, cols)
        cands: list[DimScore] = []
        for dim in cols:
            sc = self._score(early, a, b, (dim,), edges, min_lv)
            if sc is not None:
                cands.append(sc)
        m = 2 * len(cands)                                     # two tests (modification, composition) per dimension
        best = min(cands, key=lambda c: (c.p, c.dims)) if cands else None
        if best is None or bonferroni(best.p, m + prior_tries) >= P["alpha"]:
            top = [c.dims[0] for c in sorted(cands, key=lambda c: (c.p, c.dims))[:P["interaction_top_k"]]]
            pairs = []
            for d1, d2 in itertools.combinations(top, 2):
                sc = self._score(early, a, b, (d1, d2), edges, min_lv)
                if sc is not None and len(sc.table) >= 3:
                    pairs.append(sc)
            m += 2 * len(pairs)
            cands += pairs
            best = min(cands, key=lambda c: (c.p, c.dims)) if cands else None
        tried = tuple(("x".join(c.dims), float(c.p)) for c in cands)
        if best is None:
            return Investigation(verdict=Verdict.UNRESOLVED, m_tests=m + prior_tries, temporal_note=note, tried=tried, **base)
        p_adj = bonferroni(best.p, m + prior_tries)
        common: dict[str, Any] = dict(dimension=best.dims, p_raw=best.p, p_adj=p_adj, m_tests=m + prior_tries, temporal_note=note,
                      tried=tried, mechanism=best.mechanism, z_adjusted=best.z_adj, p_composition=best.p_comp, **base)
        if p_adj >= P["alpha"]:
            return Investigation(verdict=Verdict.UNRESOLVED, cause_hint=self._hint(note, None), **common)
        # out-of-sample: the ONE chosen split, tested on the late half
        p_conf, conc = self._confirm(best, late, a, b, edges, min_lv)
        if p_conf is None or p_conf >= P["confirm_alpha"] or conc is None or conc < P["concord_share"]:
            return Investigation(verdict=Verdict.SPURIOUS_SPLIT, p_confirm=p_conf, concordance=conc,
                                 cause_hint=FailureCause.FALSE_PATTERN.value if p_conf is not None else
                                 FailureCause.INSUFFICIENT_EVIDENCE.value, **common)
        # report the split refit on all rows
        final = self._table(f, a, b, best.dims, self._edges(f, cols), min_lv) or best.table
        resid = tuple(l.level for l in final if l.disagrees(P["z_crit"]))
        verdict = Verdict.RESOLVED_BY_CONTEXT if not resid else Verdict.PARTLY_RESOLVED
        scoped = []
        for l in final:
            ctx = tuple(zip(best.dims, l.level.split("|")))
            for kid, eff, cl in ((a, l.effect_a, l.clusters_a), (b, l.effect_b, l.clusters_b)):
                scoped.append(ScopedClaim(kid, ctx, eff, float("nan"), cl))
        flips = any(l.effect_a * l.effect_b < 0 for l in final)
        hint = (FailureCause.WRONG_CONTEXT.value if flips or not resid else FailureCause.INTERACTION_FAILURE.value)
        return Investigation(verdict=verdict, levels=tuple(final), p_confirm=p_conf, concordance=conc,
                             residual_levels=resid, scoped=tuple(scoped), cause_hint=hint, **common)

    def _confirm(self, sc: DimScore, late: pd.DataFrame, a: str, b: str, edges: dict, min_lv: int
                 ) -> tuple[float | None, float | None]:
        """Late-half test of the ONE chosen split. Modification: Q p-value plus sign concordance of the informative levels.
        Composition: the stratified difference must stay ~0 while the level mix must still differ between the claims;
        concordance is then the share of levels whose stratified difference is small relative to the overall one."""
        if late.empty:
            return None, None
        tab = self._table(late, a, b, sc.dims, edges, max(3, min_lv // 2))
        common = {l.level for l in tab} & {l.level for l in sc.table}
        if len(common) < 2:
            return None, None
        lt = [l for l in tab if l.level in common]
        by = {l.level: l for l in lt}
        if sc.mechanism == "COMPOSITION":
            counts = np.array([[l.clusters_a for l in lt], [l.clusters_b for l in lt]], float)
            p_comp = float(sps.chi2_contingency(counts)[1]) if (counts.sum(axis=0) > 0).all() else 1.0
            d_adj, se_adj = self._stratified(lt)
            z_adj = d_adj / se_adj if math.isfinite(se_adj) and se_adj > 0 else 0.0
            small = sum(1 for l in lt if abs(l.z) < self.p["z_crit"])
            return (p_comp if abs(z_adj) < self.p["z_crit"] else 1.0), small / len(lt)
        _, p = self._split_p(lt)
        informative = [l for l in sc.table if l.level in common and abs(l.z) >= 1.0]
        if not informative:
            return p, 0.0
        same = sum(1 for l in informative if np.sign(l.diff) == np.sign(by[l.level].diff))
        return p, same / len(informative)

    def _temporal_note(self, ev: EvidenceSet, a: str, b: str) -> str:
        early, late = ev.time_split(a, b)
        if early.empty or late.empty:
            return ""
        de, se_e, _ = _diff(early, a, b)
        dl, se_l, _ = _diff(late, a, b)
        ze = de / se_e if se_e and math.isfinite(se_e) else 0.0
        zl = dl / se_l if se_l and math.isfinite(se_l) else 0.0
        c = self.p["z_crit"]
        if abs(ze) >= c and abs(zl) < 1.0:
            return "disagreement present early, absent late: suspect a fading effect or regime change, not a context"
        if abs(zl) >= c and abs(ze) < 1.0:
            return "disagreement appears only late: suspect a new regime or a drifting effect"
        if np.sign(de) != np.sign(dl) and abs(ze) >= 1.0 and abs(zl) >= 1.0:
            return "disagreement reverses sign between halves: suspect a reversal"
        return ""

    @staticmethod
    def _hint(note: str, _: Any) -> str:
        if "regime" in note:
            return FailureCause.REGIME_CHANGE.value
        if "fading" in note:
            return FailureCause.WEAKENING_EFFECT.value
        if "reversal" in note:
            return FailureCause.REVERSAL.value
        return FailureCause.UNKNOWN.value

    def selection_stability(self, a: str, b: str, ev: EvidenceSet, now, reps: int = 40, seed: int = 0,
                            context_cols: Sequence[str] | None = None) -> dict[str, float]:
        """How often would the search pick each dimension if the DATES were resampled? A real explanation is chosen
        almost every time; noise wanders. Cluster bootstrap over the search half, seeded, whole dates kept together."""
        ev.assert_before(now)
        rng = np.random.default_rng(seed)
        early, _ = ev.time_split(a, b)
        cols = tuple(c for c in (context_cols if context_cols is not None else ev.context_cols) if c in early.columns)
        edges = self._edges(early, cols)
        groups = {c: g for c, g in early.groupby("cluster")}
        days = sorted(groups)
        if len(days) < self.p["min_clusters"]:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for _ in range(reps):
            pick = rng.integers(0, len(days), len(days))
            boot = []
            for j, i in enumerate(pick):
                g = groups[days[i]].copy()
                g["cluster"] = f"{days[i]}#{j}"
                boot.append(g)
            frame = pd.concat(boot, ignore_index=True)
            scored = [sc for sc in (self._score(frame, a, b, (d,), edges, self.p["min_level_clusters"]) for d in cols)
                      if sc is not None]
            counts[min(scored, key=lambda x: (x.p, x.dims)).dims[0] if scored else "<none>"] += 1
        return {k: v / reps for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}

    def dimension_table(self, a: str, b: str, ev: EvidenceSet, now, context_cols: Sequence[str] | None = None) -> pd.DataFrame:
        """Every single-dimension split scored on the search half - both explanations' p-values, the mechanism that won,
        and the Bonferroni-adjusted p. The table behind an investigation's headline, for people who want to see the losers."""
        ev.assert_before(now)
        early, _ = ev.time_split(a, b)
        cols = tuple(c for c in (context_cols if context_cols is not None else ev.context_cols) if c in early.columns)
        edges = self._edges(early, cols)
        rows = []
        for d in cols:
            sc = self._score(early, a, b, (d,), edges, self.p["min_level_clusters"])
            if sc is not None:
                rows.append({"dimension": d, "levels": len(sc.table), "p_modification": sc.p_het, "p_composition": sc.p_comp,
                             "mechanism": sc.mechanism, "z_overall": sc.z_overall, "z_stratified": sc.z_adj, "p_best": sc.p})
        df = pd.DataFrame(rows, columns=["dimension", "levels", "p_modification", "p_composition", "mechanism", "z_overall",
                                         "z_stratified", "p_best"])
        if len(df):
            df["p_adjusted"] = [bonferroni(p, 2 * len(df)) for p in df["p_best"]]
            df = df.sort_values(["p_best", "dimension"]).reset_index(drop=True)
        return df

    def residual_scan(self, inv: Investigation, a: str, b: str, ev: EvidenceSet, now) -> dict[str, Investigation]:
        """After a PARTLY_RESOLVED verdict, ask the same question INSIDE each level that still disagrees: is there a second
        context that explains what the first left over? The multiple-testing count carries forward, so the bar keeps rising."""
        ev.assert_before(now)
        if inv.verdict != Verdict.PARTLY_RESOLVED or not inv.dimension:
            return {}
        f = ev.pair(a, b)
        lv = self._levels(f, inv.dimension, self._edges(f, inv.dimension))
        out = {}
        for level in inv.residual_levels:
            sub = f[lv == level]
            inner_cols = [c for c in ev.context_cols if c not in inv.dimension]
            out[level] = self.investigate(a, b, EvidenceSet(sub, inner_cols), now, prior_tries=inv.m_tests,
                                          context_cols=inner_cols)
        return out

    def permutation_null(self, a: str, b: str, ev: EvidenceSet, now, reps: int = 200, seed: int = 0,
                         context_cols: Sequence[str] | None = None) -> PermutationNull:
        """Calibrated family-wise bar from a permutation null (see PermutationNull). Cheap enough for hundreds of shuffles
        because only the search statistic is recomputed, not the whole investigation. Seeded and deterministic."""
        ev.assert_before(now)
        rng = np.random.default_rng(seed)
        early, _ = ev.time_split(a, b)
        cols = tuple(c for c in (context_cols if context_cols is not None else ev.context_cols) if c in early.columns)
        edges = self._edges(early, cols)
        min_lv = self.p["min_level_clusters"]

        def stat(frame: pd.DataFrame) -> float:
            ps = [sc.p for sc in (self._score(frame, a, b, (d,), edges, min_lv) for d in cols) if sc is not None]
            return min(ps) if ps else 1.0

        obs = stat(early)
        per_date = early.groupby("cluster")[list(cols)].first()
        null = np.empty(reps)
        for i in range(reps):
            shuffled = per_date.iloc[rng.permutation(len(per_date))].set_axis(per_date.index)
            g = early.copy()
            for c in cols:
                g[c] = g["cluster"].map(shuffled[c])
            null[i] = stat(g)
        m = 2 * max(len(cols), 1)
        return PermutationNull(cols, reps, float(obs), float((1 + np.sum(null <= obs)) / (reps + 1)),
                               float(np.quantile(null, self.p["alpha"])), float(np.mean(null * m < self.p["alpha"])),
                               tuple(float(np.quantile(null, q)) for q in (0.01, 0.05, 0.25, 0.5)))

    def null_calibration(self, a: str, b: str, ev: EvidenceSet, now, reps: int = 20, seed: int = 0,
                         context_cols: Sequence[str] | None = None) -> dict[str, float]:
        """False-positive rate of the WHOLE procedure. Source labels are shuffled within each date, which keeps the dates,
        contexts and noise but destroys any real A-B difference; every rep runs the full investigation and counts how often
        it still claims a context explains something. Should sit at or below `alpha`; if it does not, the bar is too low."""
        rng = np.random.default_rng(seed)
        f = ev.pair(a, b).copy()
        hits = 0
        ran = 0
        for _ in range(reps):
            g = f.copy()
            for _, idx in g.groupby("cluster").groups.items():
                g.loc[idx, "source"] = rng.permutation(g.loc[idx, "source"].to_numpy())
            inv = self.investigate(a, b, EvidenceSet(g, context_cols if context_cols is not None else ev.context_cols), now)
            if inv.verdict == Verdict.NO_DISAGREEMENT:
                continue
            ran += 1
            hits += inv.verdict in (Verdict.RESOLVED_BY_CONTEXT, Verdict.PARTLY_RESOLVED)
        return {"reps": float(reps), "investigated": float(ran), "false_explanations": float(hits),
                "false_positive_rate": float(hits / reps)}

    # ---- planning when the data cannot decide
    def plan(self, a: Claim, b: Claim, candidate_contexts: Mapping[str, Sequence[str]], sigma: float,
             delta: float | None = None) -> dict:
        """What to collect so the disagreement can be investigated: contexts to stratify and dates needed per level."""
        d = detect(a, b)
        delta = abs(delta if delta is not None else d.diff) or 1e-9
        need = power_needed(sigma, delta, self.p["alpha"])
        rows = []
        for dim, levels in sorted(candidate_contexts.items()):
            k = len(levels)
            bonf_alpha = self.p["alpha"] / max(len(candidate_contexts), 1)
            rows.append({"dimension": dim, "levels": k, "dates_per_level_per_arm": power_needed(sigma, delta, bonf_alpha),
                         "total_dates": 2 * k * power_needed(sigma, delta, bonf_alpha)})
        return {"pair": (a.knowledge_id, b.knowledge_id), "disagreement": d.kind.value, "z": d.z,
                "dates_per_arm_unstratified": need, "stratified": sorted(rows, key=lambda r: r["total_dates"])}


# ------------------------------------------------------------------------------------------- ledger

@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    pair: tuple[str, str]
    at: str
    verdict: str
    dimension: str
    p_adj: float
    m_tests: int
    rows_through: str


class ContradictionLedger:
    """Hash-chained record of every investigation. Pair status, cumulative multiple-testing count, re-open triggers and the
    epistemic state a knowledge id should carry are all read from here, never recomputed from memory of what we hoped."""

    def __init__(self, path=None):
        self._chain = ChainFile(path, "contra")
        self._entries: list[LedgerEntry] = []
        self._dossiers: list[dict | None] = []
        self._load()

    def _load(self):
        for line in self._chain.take_new():
            b = line["body"]
            self._entries.append(LedgerEntry(tuple(b["pair"]), b["at"], b["verdict"], b["dimension"], float(b["p_adj"]),
                                             int(b["m_tests"]), b["rows_through"]))
            self._dossiers.append(b.get("dossier"))

    def record(self, inv: Investigation) -> LedgerEntry:
        if inv.verdict == Verdict.INSUFFICIENT_DATA:
            verdict = Status.NEEDS_DATA.value
        else:
            verdict = inv.verdict.value
        body = {"pair": list(inv.pair), "at": inv.now, "verdict": verdict, "dimension": "x".join(inv.dimension),
                "p_adj": float(inv.p_adj), "m_tests": int(inv.m_tests), "rows_through": inv.rows_through,
                "dossier": inv.to_dict()}
        self._chain.append_many([body])
        self._load()
        return self._entries[-1]

    def dossier(self, pair: Sequence[str], now) -> "Investigation | None":
        """The full Investigation behind the latest ledger entry for a pair (what was found, not just the verdict)."""
        key, n = tuple(sorted(pair)), as_date(now)
        for e, d in zip(reversed(self._entries), reversed(self._dossiers)):
            if e.pair == key and as_date(e.at) < n and d is not None:
                return Investigation.from_dict(d)
        return None

    def markdown(self, now) -> str:
        pairs = sorted({e.pair for e in self.entries(None, now)})
        lines = [f"# Contradiction ledger @ {as_date(now)}", "", self.summary(now), "",
                 "| pair | status | splits tried | last dimension | p_adj | investigated |", "|---|---|---:|---|---:|---|"]
        for p in pairs:
            last = self.entries(p, now)[-1]
            lines.append(f"| {p[0]} vs {p[1]} | {self.status(p, now).value} | {self.tries(p, now)} | {last.dimension or '-'} | "
                         f"{last.p_adj:.3g} | {last.at} |")
        return "\n".join(lines)

    def entries(self, pair: Sequence[str] | None, now) -> list[LedgerEntry]:
        n = as_date(now)
        key = tuple(sorted(pair)) if pair is not None else None
        return [e for e in self._entries if (key is None or e.pair == key) and as_date(e.at) < n]

    def status(self, pair: Sequence[str], now) -> Status:
        es = self.entries(pair, now)
        if not es:
            return Status.OPEN
        v = es[-1].verdict
        return {Verdict.RESOLVED_BY_CONTEXT.value: Status.RESOLVED, Verdict.PARTLY_RESOLVED.value: Status.PARTLY_RESOLVED,
                Verdict.NO_DISAGREEMENT.value: Status.CLOSED, Status.NEEDS_DATA.value: Status.NEEDS_DATA,
                }.get(v, Status.UNRESOLVED_KEEP_BOTH)

    def tries(self, pair: Sequence[str], now) -> int:
        """Cumulative splits ever tried for this pair (the multiple-testing count feeds the next investigation)."""
        es = self.entries(pair, now)
        return max((e.m_tests for e in es), default=0)

    def needs_reinvestigation(self, pair: Sequence[str], now, evidence_through, growth_days: int = 60) -> bool:
        """Re-open when materially newer evidence exists than the last investigation saw."""
        es = self.entries(pair, now)
        if not es:
            return True
        last = es[-1]
        if not last.rows_through:
            return True
        gap = (as_date(evidence_through) - as_date(last.rows_through)).days
        return gap >= growth_days

    def pairs_of(self, knowledge_id: str, now) -> list[tuple[str, str]]:
        return sorted({e.pair for e in self.entries(None, now) if knowledge_id in e.pair})

    def contradiction_rate(self, knowledge_id: str, now) -> float | None:
        """Share of this knowledge's investigated pairs that remain conflicted; None if it was never investigated."""
        pairs = self.pairs_of(knowledge_id, now)
        if not pairs:
            return None
        bad = sum(1 for p in pairs if self.status(p, now) in (Status.UNRESOLVED_KEEP_BOTH, Status.PARTLY_RESOLVED))
        return bad / len(pairs)

    def epistemic_for(self, knowledge_id: str, now) -> Epistemic | None:
        """CONTRADICTED while any pair stays unresolved; CONDITIONAL once every conflict is explained by context; None if
        never investigated (the ledger says nothing)."""
        sts = [self.status(p, now) for p in self.pairs_of(knowledge_id, now)]
        if not sts:
            return None
        if any(s == Status.UNRESOLVED_KEEP_BOTH for s in sts):
            return Epistemic.CONTRADICTED
        if any(s in (Status.PARTLY_RESOLVED, Status.NEEDS_DATA) for s in sts):
            return Epistemic.CONTRADICTED if any(s == Status.PARTLY_RESOLVED for s in sts) else None
        if any(s == Status.RESOLVED for s in sts):
            return Epistemic.CONDITIONAL
        return None

    def unknown_state(self, knowledge_id: str, now) -> Unknown | None:
        e = self.epistemic_for(knowledge_id, now)
        return Unknown.CONFLICTED if e == Epistemic.CONTRADICTED else None

    def verify(self) -> dict:
        return self._chain.verify()

    def summary(self, now) -> str:
        pairs = sorted({e.pair for e in self.entries(None, now)})
        cnt: dict[str, int] = defaultdict(int)
        for p in pairs:
            cnt[self.status(p, now).value] += 1
        return f"{len(pairs)} investigated pairs @ {as_date(now)}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items()))


def next_to_investigate(claims: Sequence[Claim], ledger: ContradictionLedger, now, evidence_through=None,
                        growth_days: int = 60, alpha: float = 0.05) -> list[dict]:
    """Research queue over disagreements between claims: genuine conflicts only, sign conflicts first, weighted by how loud
    the disagreement is and how much evidence stands behind it; pairs already settled (or investigated with no new
    evidence since) are left alone."""
    queue = []
    for d in triage(claims, alpha):
        pair = (d.a, d.b)
        st = ledger.status(pair, now)
        through = evidence_through if evidence_through is not None else now
        stale = ledger.needs_reinvestigation(pair, now, through, growth_days)
        if st in (Status.RESOLVED, Status.CLOSED) and not stale:
            continue
        if st in (Status.UNRESOLVED_KEEP_BOTH, Status.PARTLY_RESOLVED, Status.NEEDS_DATA) and not stale:
            continue
        n = min(c.n for c in claims if c.knowledge_id in pair)
        prio = abs(d.z) * math.sqrt(max(n, 1)) * (2.0 if d.kind == DisagreementKind.SIGN_CONFLICT else 1.0)
        queue.append({"a": d.a, "b": d.b, "kind": d.kind.value, "z": d.z, "status": st.value, "priority": prio,
                      "why": "never investigated" if st == Status.OPEN else "new evidence since last look"})
    return sorted(queue, key=lambda r: (-r["priority"], r["a"], r["b"]))


def calibration_suite(inv: Investigator, worlds: Sequence[tuple[str, str, EvidenceSet]], now, reps: int = 100, seed: int = 0,
                      context_cols: Sequence[str] | None = None, alpha: float = 0.05) -> dict[str, float]:
    """Run the permutation null over many worlds that are KNOWN to contain no context effect and report how often the
    procedure claims one. A calibrated bar rejects about `alpha` of them; the Bonferroni figure shows how conservative the
    per-investigation rule is by comparison. The harness the next (validation) wave feeds real and planted worlds through."""
    if not worlds:
        return {"worlds": 0.0, "rejection_rate": float("nan"), "mean_p_family": float("nan"), "mean_fwer_bonferroni": float("nan")}
    res = [inv.permutation_null(a, b, ev, now, reps=reps, seed=seed + i, context_cols=context_cols)
           for i, (a, b, ev) in enumerate(worlds)]
    ps = np.array([r.p_family for r in res])
    lo = float(np.mean(ps <= alpha))
    # binomial upper limit on the rejection rate, so "holds alpha" is a statement with its uncertainty attached
    upper = float(sps.beta.ppf(0.975, lo * len(ps) + 1, len(ps) - lo * len(ps) + 1)) if len(ps) else float("nan")
    return {"worlds": float(len(ps)), "rejection_rate": lo, "rejection_rate_upper95": upper, "mean_p_family": float(ps.mean()),
            "mean_fwer_bonferroni": float(np.mean([r.fwer_at_bonferroni for r in res])), "alpha": alpha,
            "holds_alpha": bool(upper >= alpha and lo <= max(alpha * 3, 0.2))}


def investigate_calibrated(inv: Investigator, a: str, b: str, ev: EvidenceSet, now, prior_tries: int = 0, reps: int = 100,
                           seed: int = 0, context_cols: Sequence[str] | None = None) -> tuple[Investigation, PermutationNull | None]:
    """Investigation plus a second, distribution-free opinion: a context explanation is kept only if BOTH the Bonferroni bar
    and the permutation-null family-wise p agree. If the permutation null disagrees the verdict is downgraded to
    UNRESOLVED (the pair stays contradicted) and the reason is written into the temporal note. No search, no null."""
    res = inv.investigate(a, b, ev, now, prior_tries=prior_tries, context_cols=context_cols)
    if res.verdict not in (Verdict.RESOLVED_BY_CONTEXT, Verdict.PARTLY_RESOLVED):
        return res, None
    null = inv.permutation_null(a, b, ev, now, reps=reps, seed=seed, context_cols=context_cols)
    if null.significant(inv.p["alpha"]):
        return res, null
    note = f"permutation null disagrees (family-wise p {null.p_family:.3f}): explanation not trusted"
    return dataclasses.replace(res, verdict=Verdict.UNRESOLVED, residual_levels=(), scoped=(),
                               temporal_note=(res.temporal_note + "; " if res.temporal_note else "") + note), null


def investigate_and_record(inv: Investigator, ledger: ContradictionLedger, a: str, b: str, ev: EvidenceSet, now,
                           context_cols: Sequence[str] | None = None) -> Investigation:
    """One call for the loop: pay the pair's cumulative test count, investigate, write the ledger."""
    res = inv.investigate(a, b, ev, now, prior_tries=ledger.tries((a, b), now), context_cols=context_cols)
    ledger.record(res)
    return res
