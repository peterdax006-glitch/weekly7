"""Knowledge hierarchy (contract C62 section 19; checklist F13) - IMPLEMENTED, NOT VALIDATED.

    general rule -> market condition -> sector condition -> stock-type condition -> volatility condition -> specific interaction

Specific knowledge overrides general knowledge ONLY when it has the evidence; otherwise it shrinks toward its parent. That is
what stops "tiny sample -> extreme confidence". Mechanisms:

  evidence      each node keeps per-cluster sufficient statistics (cluster = date by default): a hundred names on one day are
                one observation, not a hundred. Variance of a node mean is cluster-robust and floored by the general rule's
                cross-cluster noise divided by the node's cluster count, so 2 lucky dates cannot claim a tiny variance.
  shrinkage     normal-normal empirical Bayes per parent: weight on the child's own mean w = tau^2/(tau^2 + v_child), tau^2
                estimated across siblings by DerSimonian-Laird (floored). Small v_child (lots of evidence) -> own mean,
                large v_child (little) -> the parent's posterior. Posterior variance carries the parent's uncertainty down.
  override gate a node's estimate is USED instead of its parent's only if it has enough independent dates, enough own weight,
                AND differs from the parent's used estimate by more than a Sidak-adjusted z for the number of siblings tested.
                Failing the gate means the parent's estimate is used (parsimony: a distinction that changes nothing is not made).
  confidence    p(sign is right) is additionally capped by sample size, so a tiny cell can never report near-certainty.
  novelty       a context value never seen falls back to the deepest known ancestor, with a note. No data at all is
                Unknown.INSUFFICIENT_DATA, never a fabricated 0.
  time          rows dated at/after `now` raise FirewallBreach; `update` is incremental and order-independent.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_stats as _ps
from .archive import ArchiveError
from .core import Edge, FirewallBreach, Unknown, as_date, canonical_json, stable_hash
from .knowledge_graph import EdgeProposal


class HierarchyError(ArchiveError):
    pass


class Level(enum.IntEnum):
    GENERAL = 0
    MARKET = 1
    SECTOR = 2
    STOCK_TYPE = 3
    VOLATILITY = 4
    INTERACTION = 5


LEVEL_DIMS: dict[Level, str] = {Level.MARKET: "market", Level.SECTOR: "sector", Level.STOCK_TYPE: "stock_type",
                                Level.VOLATILITY: "volatility", Level.INTERACTION: "interaction"}
DIM_ORDER = tuple(LEVEL_DIMS[l] for l in sorted(LEVEL_DIMS))
Path = tuple  # tuple[tuple[str, str], ...]: the (dimension, value) pairs from the root down

PARAMS = {
    "min_root_clusters": 5,        # the general rule itself needs this many independent dates
    "min_clusters_override": 12,   # independent dates before any node may replace its parent's estimate
    "min_n_eff": 8.0,              # Kish effective number of clusters
    "w_min": 0.35,                 # own-evidence weight required to override
    "alpha_override": 0.10,        # family-wise level across a parent's children (Sidak)
    "tau_min_frac": 0.10,          # tau floor as a fraction of the general rule's cross-cluster sd
    "tau_single_frac": 0.50,       # tau when a parent has a single child (cannot be estimated)
    "n_full": 40.0,                # n_eff at which the sample-size cap on sign confidence disappears
    "ci_z": 1.96,
}


def node_id(path: Path) -> str:
    """Stable, identity-free name for the rule at a path (also a valid knowledge-graph node id)."""
    return "rule:" + ("/".join(f"{d}={v}" for d, v in path) if path else "general")


# ------------------------------------------------------------------------------------------- sufficient statistics

class NodeStats:
    """Per-cluster sums for one node: cluster -> [sum w, sum w*y, count]. Order-independent and additive."""
    __slots__ = ("cl",)

    def __init__(self):
        self.cl: dict[str, list] = {}

    def add(self, y: float, w: float, c: str):
        s = self.cl.get(c)
        if s is None:
            self.cl[c] = [w, w * y, 1]
        else:
            s[0] += w
            s[1] += w * y
            s[2] += 1

    @property
    def clusters(self) -> int:
        return len(self.cl)

    @property
    def n(self) -> int:
        return sum(s[2] for s in self.cl.values())

    @property
    def sw(self) -> float:
        return sum(s[0] for s in self.cl.values())

    def mean(self) -> float:
        sw = self.sw
        return sum(s[1] for s in self.cl.values()) / sw if sw > 0 else float("nan")

    def robust_var(self) -> float:
        """Cluster-robust variance of the mean (small-sample corrected); inf with under 2 clusters."""
        G = self.clusters
        if G < 2:
            return float("inf")
        m, sw = self.mean(), self.sw
        return sum((s[1] - m * s[0]) ** 2 for s in self.cl.values()) / (sw * sw) * G / (G - 1)

    def n_eff(self) -> float:
        """Kish effective number of clusters: (sum w)^2 / sum of squared cluster weights."""
        sq = sum(s[0] ** 2 for s in self.cl.values())
        return self.sw ** 2 / sq if sq > 0 else 0.0

    def cluster_means(self) -> np.ndarray:
        return np.array([s[1] / s[0] for s in self.cl.values() if s[0] > 0])

    def to_json(self) -> dict:
        return {c: list(s) for c, s in sorted(self.cl.items())}

    @classmethod
    def from_json(cls, d: Mapping) -> "NodeStats":
        n = cls()
        n.cl = {str(c): [float(s[0]), float(s[1]), int(s[2])] for c, s in d.items()}
        return n


@dataclasses.dataclass(frozen=True)
class Posterior:
    """Everything known about one rule after shrinkage and gating."""
    path: Path
    level: Level
    clusters: int
    n: int
    n_eff: float
    raw_mean: float
    raw_var: float
    tau2: float
    weight_own: float            # w: share of the posterior mean that comes from this node's own data
    post_mean: float
    post_var: float
    z_vs_parent: float
    z_needed: float
    qualifies: bool
    used_path: Path              # the node whose estimate is actually used at this path
    used_mean: float
    used_var: float
    reason: str

    @property
    def id(self) -> str:
        return node_id(self.path)


@dataclasses.dataclass(frozen=True)
class HierarchyEstimate:
    """The answer for a context: the estimate to use, how sure, and the whole chain that produced it."""
    context: tuple[tuple[str, str], ...]
    path: Path                   # deepest node found
    used_path: Path              # node whose estimate is used (<= path depth)
    level: Level
    mean: float | None
    se: float | None
    ci: tuple[float, float] | None
    p_sign: float | None         # P(true effect has the sign of `mean`)
    p_sign_capped: float | None  # ... limited by sample size
    n_eff: float
    clusters: int
    weight_own: float | None
    chain: tuple[Posterior, ...]
    notes: tuple[str, ...]
    state: Unknown | None        # INSUFFICIENT_DATA when there is nothing to say

    def is_known(self) -> bool:
        return self.state is None and self.mean is not None


# ------------------------------------------------------------------------------------------- the hierarchy

class KnowledgeHierarchy:
    def __init__(self, params: Mapping | None = None, dims: Sequence[str] = DIM_ORDER):
        self.p = {**PARAMS, **(params or {})}
        bad = validate_params(self.p)
        if bad:
            raise HierarchyError("invalid hierarchy parameters: " + "; ".join(bad))
        self.dims = tuple(dims)
        self._stats: dict[Path, NodeStats] = {(): NodeStats()}
        self._children: dict[Path, set] = defaultdict(set)
        self._post: dict[Path, Posterior] = {}
        self._dirty = True
        self.through: str | None = None
        self.rows_seen = 0

    # ---- data in
    def _validate(self, rows: pd.DataFrame, now) -> pd.DataFrame:
        if rows.empty:
            return rows
        for c in ("effect", "when"):
            if c not in rows.columns:
                raise HierarchyError(f"rows need column {c!r}")
        f = rows.copy()
        f["when"] = pd.to_datetime(f["when"])
        n = pd.Timestamp(as_date(now))
        late = f["when"] >= n
        if late.any():
            raise FirewallBreach(f"{int(late.sum())} hierarchy rows dated at/after now={as_date(now)}")
        eff = pd.to_numeric(f["effect"], errors="coerce").to_numpy(float)
        if not np.isfinite(eff).all():
            raise HierarchyError("non-finite effects")
        f["effect"] = eff
        if "weight" not in f.columns:
            f["weight"] = 1.0
        if (f["weight"] < 0).any():
            raise HierarchyError("negative weights")
        if "cluster" not in f.columns:
            f["cluster"] = f["when"].dt.strftime("%Y-%m-%d")
        f["cluster"] = f["cluster"].astype(str)
        return f

    def update(self, rows: pd.DataFrame, now) -> int:
        """Add evidence dated strictly before `now`. Each row feeds every node on its path (a row missing a level's value
        stops at the level above). Returns rows absorbed."""
        f = self._validate(rows, now)
        if f.empty:
            return 0
        cols = [d for d in self.dims if d in f.columns]
        y, w, c = f["effect"].to_numpy(float), f["weight"].to_numpy(float), f["cluster"].to_numpy(str)
        vals = {d: f[d].astype(object).where(f[d].notna(), None).to_numpy() for d in cols}
        for i in range(len(f)):
            path: Path = ()
            self._stats[()].add(y[i], w[i], c[i])
            for d in self.dims:
                v = vals[d][i] if d in vals else None
                if v is None:
                    break
                nxt = path + ((d, str(v)),)
                st = self._stats.get(nxt)
                if st is None:
                    st = self._stats[nxt] = NodeStats()
                    self._children[path].add(nxt)
                st.add(y[i], w[i], c[i])
                path = nxt
        self.rows_seen += len(f)
        last = f["when"].max().date().isoformat()
        self.through = last if self.through is None or last > self.through else self.through
        self._dirty = True
        return len(f)

    def fit(self, rows: pd.DataFrame, now) -> "KnowledgeHierarchy":
        self.__init__(self.p, self.dims)
        self.update(rows, now)
        return self

    # ---- statistics
    def _root_scale(self) -> float:
        cm = self._stats[()].cluster_means()
        return float(np.var(cm, ddof=1)) if cm.size >= 3 else float("nan")

    def _node_var(self, st: NodeStats, s2: float) -> float:
        """Robust variance floored by the cross-cluster noise / clusters: two lucky dates cannot claim a tiny variance."""
        floor = s2 / max(st.clusters, 1) if math.isfinite(s2) else 0.0
        rv = st.robust_var()
        return max(rv, floor) if math.isfinite(rv) else max(s2, 1e-12) if math.isfinite(s2) else float("inf")

    def _dl_tau2(self, ms: np.ndarray, vs: np.ndarray, sd_root: float) -> float:
        """Between-sibling variance by DerSimonian-Laird (engine.pattern_stats.eb_shrink, centre estimated), floored."""
        P = self.p
        if len(ms) < 2:
            return (P["tau_single_frac"] * sd_root) ** 2
        dl = _ps.eb_shrink(ms, np.sqrt(vs), center=None)["tau2"]
        return max(float(dl), (P["tau_min_frac"] * sd_root) ** 2)

    def _compute(self):
        if not self._dirty:
            return
        P = self.p
        self._post = {}
        root = self._stats[()]
        s2 = self._root_scale()
        if root.clusters < P["min_root_clusters"] or not math.isfinite(s2):
            self._dirty = False
            return
        sd_root = math.sqrt(s2)
        rv = self._node_var(root, s2)
        rm = root.mean()
        self._post[()] = Posterior((), Level.GENERAL, root.clusters, root.n, root.n_eff(), rm, rv, float("nan"), 1.0, rm, rv,
                                   0.0, 0.0, True, (), rm, rv, "the general rule")
        frontier: list[Path] = [()]
        while frontier:
            nxt: list[Path] = []
            for parent in frontier:
                kids = sorted(self._children.get(parent, ()))
                if not kids:
                    continue
                pp = self._post[parent]
                ms = np.array([self._stats[k].mean() for k in kids])
                vs = np.array([self._node_var(self._stats[k], s2) for k in kids])
                tau2 = self._dl_tau2(ms, vs, sd_root)
                zc = float(sps.norm.isf((1.0 - (1.0 - P["alpha_override"]) ** (1.0 / len(kids))) / 2.0))     # Sidak over siblings
                for k, m, v in zip(kids, ms, vs):
                    st = self._stats[k]
                    w = tau2 / (tau2 + v) if math.isfinite(v) else 0.0
                    pm = w * m + (1 - w) * pp.post_mean
                    pv = w * v + (1 - w) ** 2 * pp.post_var if math.isfinite(v) else pp.post_var
                    z = float((m - pp.used_mean) / math.sqrt(v)) if math.isfinite(v) and v > 0 else 0.0
                    enough = st.clusters >= P["min_clusters_override"] and st.n_eff() >= P["min_n_eff"]
                    strong = w >= P["w_min"]
                    differs = abs(z) >= zc
                    ok = enough and strong and differs
                    why = ("overrides parent" if ok else
                           "too few independent dates" if not enough else
                           "own evidence too weak against sibling noise" if not strong else
                           "not distinguishable from the parent")
                    post = Posterior(k, Level(len(k)), st.clusters, st.n, st.n_eff(), float(m), float(v), tau2, float(w),
                                     float(pm), float(pv), z, zc, ok, k if ok else pp.used_path,
                                     pm if ok else pp.used_mean, pv if ok else pp.used_var, why)
                    self._post[k] = post
                    nxt.append(k)
            frontier = nxt
        self._dirty = False

    # ---- queries
    def posterior(self, path: Path) -> Posterior | None:
        self._compute()
        return self._post.get(tuple(path))

    def paths(self) -> list[Path]:
        self._compute()
        return sorted(self._post, key=lambda p: (len(p), p))

    def estimate(self, context: Mapping[str, Any]) -> HierarchyEstimate:
        """Best available estimate for a context, walking down until the evidence (or the context) runs out."""
        self._compute()
        ctx = tuple((d, str(context[d])) for d in self.dims if context.get(d) is not None)
        root = self._post.get(())
        if root is None:
            return HierarchyEstimate(ctx, (), (), Level.GENERAL, None, None, None, None, None, 0.0, self._stats[()].clusters,
                                     None, (), ("not enough independent dates in the general rule",), Unknown.INSUFFICIENT_DATA)
        chain = [root]
        notes: list[str] = []
        path: Path = ()
        for d in self.dims:
            v = context.get(d)
            if v is None:
                break
            nxt = path + ((d, str(v)),)
            po = self._post.get(nxt)
            if po is None:
                notes.append(f"{d}={v} has no usable history: falling back to {node_id(path)}")
                break
            chain.append(po)
            path = nxt
        end = chain[-1]
        se = math.sqrt(end.used_var) if end.used_var > 0 else 0.0
        used = self._post[end.used_path]
        z = self.p["ci_z"]
        p_sign = float(sps.norm.cdf(abs(end.used_mean) / se)) if se > 0 else 1.0
        cap = min(1.0, used.n_eff / self.p["n_full"])
        capped = 0.5 + (p_sign - 0.5) * cap
        if end.path != end.used_path:
            notes.append(f"{node_id(end.path)} does not override its parent: {end.reason}")
        return HierarchyEstimate(ctx, end.path, end.used_path, Level(len(end.used_path)), end.used_mean, se,
                                 (end.used_mean - z * se, end.used_mean + z * se), p_sign, capped, used.n_eff,
                                 used.clusters, used.weight_own, tuple(chain), tuple(notes), None)

    def explain(self, context: Mapping[str, Any]) -> str:
        e = self.estimate(context)
        if not e.is_known():
            return "no estimate: " + "; ".join(e.notes)
        lines = [f"context {dict(e.context)} -> use {node_id(e.used_path)} ({e.level.name})",
                 f"  estimate {e.mean:+.5f} +/- {e.se:.5f}; P(sign right) {e.p_sign:.3f}, sample-capped {e.p_sign_capped:.3f}"]
        for po in e.chain:
            lines.append(f"  {po.level.name:<11} {po.id:<40} raw {po.raw_mean:+.5f} (dates {po.clusters}) w_own {po.weight_own:.2f} "
                         f"-> {'USED' if po.used_path == po.path else 'parent used'}: {po.reason}")
        return "\n".join(lines + [f"  note: {n}" for n in e.notes])

    # ---- diagnostics
    def table(self) -> pd.DataFrame:
        self._compute()
        rows = []
        for p in self.paths():
            po = self._post[p]
            rows.append({"path": po.id, "level": po.level.name, "clusters": po.clusters, "n": po.n, "n_eff": po.n_eff,
                         "raw_mean": po.raw_mean, "raw_se": math.sqrt(po.raw_var) if math.isfinite(po.raw_var) else np.nan,
                         "w_own": po.weight_own, "post_mean": po.post_mean, "post_se": math.sqrt(max(po.post_var, 0.0)),
                         "z_vs_parent": po.z_vs_parent, "qualifies": po.qualifies, "used": node_id(po.used_path),
                         "reason": po.reason})
        return pd.DataFrame(rows)

    def shrinkage_by_level(self) -> dict[str, dict[str, float]]:
        """Where the heterogeneity lives: mean own-weight, mean |raw - posterior| and qualification rate per level."""
        self._compute()
        out: dict[str, dict[str, float]] = {}
        for lv in Level:
            pos = [p for p in self._post.values() if p.level == lv and lv != Level.GENERAL]
            if not pos:
                continue
            out[lv.name] = {"nodes": float(len(pos)), "mean_w_own": float(np.mean([p.weight_own for p in pos])),
                            "mean_shift": float(np.mean([abs(p.raw_mean - p.post_mean) for p in pos])),
                            "qualified_share": float(np.mean([p.qualifies for p in pos])),
                            "mean_tau2": float(np.nanmean([p.tau2 for p in pos]))}
        return out

    def evidence_needed(self, path: Path) -> dict[str, float]:
        """How many more independent dates this node would need to override its parent if its current mean held."""
        self._compute()
        po = self._post.get(tuple(path))
        if po is None or not path:
            raise HierarchyError("no such non-root node")
        cur = po.clusters
        if abs(po.z_vs_parent) < 1e-9:
            return {"clusters_now": float(cur), "clusters_for_override": float("inf"), "extra": float("inf")}
        need_z = max(po.z_needed, 0.0)
        need_g = cur * (need_z / abs(po.z_vs_parent)) ** 2
        need_g = max(need_g, float(self.p["min_clusters_override"]))
        return {"clusters_now": float(cur), "clusters_for_override": float(math.ceil(need_g)),
                "extra": float(max(0, math.ceil(need_g) - cur))}

    def detectable_difference(self, path: Path) -> float:
        """Smallest gap to the parent this node's data could show as significant right now (minimum detectable effect)."""
        po = self.posterior(path)
        if po is None:
            raise HierarchyError("no such node")
        return float(po.z_needed * math.sqrt(po.raw_var)) if math.isfinite(po.raw_var) else float("inf")

    def qualified(self) -> list[Path]:
        self._compute()
        return sorted((p for p, po in self._post.items() if p and po.qualifies), key=lambda p: (len(p), p))

    def edge_proposals(self) -> list[EdgeProposal]:
        """SPECIALIZES (child -> parent) for every node that earns its distinction, weighted by its own-evidence share."""
        self._compute()
        out = []
        for p in self.qualified():
            po = self._post[p]
            out.append(EdgeProposal(node_id(p), node_id(p[:-1]), Edge.SPECIALIZES.value, float(min(max(po.weight_own, 0.0), 1.0)),
                                    (), (("z_vs_parent", round(po.z_vs_parent, 4)), ("clusters", po.clusters))))
        return out

    def reversals(self) -> list[tuple[Path, Path]]:
        """(child, parent) for every qualified rule whose effect has the OPPOSITE sign to the rule it specialises: the parent
        is wrong in that context, so the pair is a contradiction to investigate, not an average to take."""
        self._compute()
        out = []
        for p in self.qualified():
            po, parent = self._post[p], self._post.get(p[:-1])
            if parent is not None and po.post_mean * parent.used_mean < 0:
                out.append((p, p[:-1]))
        return out

    def fingerprint(self) -> str:
        self._compute()
        return stable_hash([(node_id(p), round(po.post_mean, 10), round(po.post_var, 12), po.qualifies)
                            for p, po in sorted(self._post.items())], 20)

    # ---- persistence
    def to_dict(self) -> dict:
        return {"params": self.p, "dims": list(self.dims), "through": self.through, "rows_seen": self.rows_seen,
                "nodes": {json.dumps(p): st.to_json() for p, st in sorted(self._stats.items())}}

    @classmethod
    def from_dict(cls, d: Mapping) -> "KnowledgeHierarchy":
        h = cls(d["params"], d["dims"])
        h._stats = {}
        for k, v in d["nodes"].items():
            path = tuple((a, b) for a, b in json.loads(k))
            h._stats[path] = NodeStats.from_json(v)
            if path:
                h._children[path[:-1]].add(path)
        h._stats.setdefault((), NodeStats())
        h.through, h.rows_seen = d.get("through"), int(d.get("rows_seen", 0))
        return h

    def save(self, path) -> str:
        blob = canonical_json(self.to_dict())
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob)
        return stable_hash(blob, 20)

    @classmethod
    def load(cls, path) -> "KnowledgeHierarchy":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def markdown(self, top: int = 15) -> str:
        t = self.table()
        if t.empty:
            return "# Knowledge hierarchy\n\nnot enough evidence for a general rule (INSUFFICIENT_DATA)"
        lines = [f"# Knowledge hierarchy (through {self.through}, {self.rows_seen} rows)", "",
                 "| rule | level | dates | raw | own weight | used |", "|---|---|---:|---:|---:|---|"]
        for _, r in t.head(top).iterrows():
            w_own = "-" if pd.isna(r["w_own"]) else format(r["w_own"], ".2f")
            lines.append(f"| {r['path']} | {r['level']} | {int(r['clusters'])} | {r['raw_mean']:+.4f} | {w_own} | {r['used']} |")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------- out-of-sample checks

def _row_paths(rows: pd.DataFrame, dims: Sequence[str]) -> list[dict]:
    out = []
    for rec in rows[[d for d in dims if d in rows.columns]].to_dict("records"):
        out.append({d: v for d, v in rec.items() if v is not None and not (isinstance(v, float) and math.isnan(v))})
    return out


def compare_predictors(train: pd.DataFrame, test: pd.DataFrame, now_train, params: Mapping | None = None) -> dict[str, float]:
    """Held-out mean squared error of four predictors of a row's effect: the general mean, the raw mean of the deepest cell
    seen, the shrunk posterior of that cell (no gate), and the gated USED estimate. Shrinkage should beat raw cells when
    cells are small - this is the check that the hierarchy earns its complexity."""
    h = KnowledgeHierarchy(params).fit(train, now_train)
    h._compute()
    if () not in h._post:
        raise HierarchyError("training data too thin for a general rule")
    ctxs = _row_paths(test, h.dims)
    y = pd.to_numeric(test["effect"]).to_numpy(float)
    preds = {"general": [], "raw_cell": [], "shrunk_cell": [], "used": []}
    for c in ctxs:
        e = h.estimate(c)
        deepest = h._post[e.path]
        preds["general"].append(h._post[()].raw_mean)
        preds["raw_cell"].append(deepest.raw_mean)
        preds["shrunk_cell"].append(deepest.post_mean)
        preds["used"].append(e.mean)
    return {k: float(np.mean((np.array(v) - y) ** 2)) for k, v in preds.items()} | {"n_test": float(len(y))}


def override_stability(rows: pd.DataFrame, now, params: Mapping | None = None) -> dict[str, Any]:
    """Do override decisions replicate? Fit on the early and late halves (by date) separately and compare which nodes qualify.
    A hierarchy whose distinctions do not survive a time split is fitting noise."""
    f = rows.copy()
    f["when"] = pd.to_datetime(f["when"])
    days = np.sort(f["when"].unique())
    if len(days) < 4:
        return {"jaccard": None, "early": [], "late": [], "note": "too few dates"}
    cut = days[len(days) // 2]
    a = KnowledgeHierarchy(params).fit(f[f["when"] < cut], now)
    b = KnowledgeHierarchy(params).fit(f[f["when"] >= cut], now)
    qa, qb = set(a.qualified()), set(b.qualified())
    union = qa | qb
    return {"jaccard": (len(qa & qb) / len(union)) if union else None, "early": sorted(node_id(p) for p in qa),
            "late": sorted(node_id(p) for p in qb), "note": ""}


def early_late_sign_agreement(rows: pd.DataFrame, now, params: Mapping | None = None) -> float | None:
    """Among nodes that qualify in the early half, the share whose posterior mean has the same sign in the late half."""
    f = rows.copy()
    f["when"] = pd.to_datetime(f["when"])
    days = np.sort(f["when"].unique())
    if len(days) < 4:
        return None
    cut = days[len(days) // 2]
    a = KnowledgeHierarchy(params).fit(f[f["when"] < cut], now)
    b = KnowledgeHierarchy(params).fit(f[f["when"] >= cut], now)
    qual = a.qualified()
    same = [np.sign(a.posterior(p).post_mean) == np.sign(b.posterior(p).post_mean) for p in qual if b.posterior(p) is not None]
    return float(np.mean(same)) if same else None


# ------------------------------------------------------------------------------------------- synthetic worlds

def simulate_panel(seed: int, cells: Sequence[Mapping], n_days: int = 150, rows_per_day: int = 3, noise: float = 0.02,
                   start: str = "2020-01-01", date_shock: float = 0.0) -> pd.DataFrame:
    """Synthetic evidence with KNOWN truth. Each cell is {"ctx": {dim: value, ...}, "effect": true mean, "p": share of dates
    on which the cell is observed}. `date_shock` adds a shared per-date shock (what makes clustering matter). Deterministic
    in `seed`; the truth is attached as df.attrs["cells"]."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    shocks = rng.normal(0, date_shock, n_days) if date_shock > 0 else np.zeros(n_days)
    rows = []
    for d in range(n_days):
        for cell in cells:
            if rng.random() > float(cell.get("p", 1.0)):
                continue
            for _ in range(rows_per_day):
                rows.append({"when": t0 + pd.Timedelta(days=d), "effect": float(cell["effect"]) + shocks[d] + rng.normal(0, noise),
                             **dict(cell["ctx"])})
    df = pd.DataFrame(rows)
    df.attrs["cells"] = [dict(c) for c in cells]
    return df


# ------------------------------------------------------------------------------------------- parameter validation

def validate_params(p: Mapping) -> list[str]:
    """Range checks so a mistyped threshold cannot silently disable the safeguards (e.g. w_min=0 would let anything override)."""
    errs = []
    rules = {"min_root_clusters": (2, 10 ** 6), "min_clusters_override": (3, 10 ** 6), "min_n_eff": (1.0, 10 ** 6),
             "w_min": (0.05, 1.0), "alpha_override": (1e-6, 0.5), "tau_min_frac": (0.0, 5.0),
             "tau_single_frac": (0.01, 5.0), "n_full": (1.0, 10 ** 6), "ci_z": (0.5, 6.0)}
    for k, (lo, hi) in rules.items():
        v = p.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
            errs.append(f"{k}={v!r} outside [{lo}, {hi}]")
    return errs


# ------------------------------------------------------------------------------------------- graph integration

def register_in_graph(h: "KnowledgeHierarchy", graph, known_at) -> dict[str, int]:
    """Write the fitted hierarchy into the knowledge graph: one KNOWLEDGE node per fitted rule, SPECIALIZES edges for every
    rule that earns its distinction, and a CONTRADICTS edge (marked uninvestigated) for every reversal - a qualified
    child whose effect has the opposite sign to the rule it specialises. Returns counts."""
    from .knowledge_graph import NodeType
    h._compute()
    counts = {"nodes": 0, "specializes": 0, "reversals": 0}
    for p in h.paths():
        po = h._post[p]
        graph.add_node(po.id, NodeType.KNOWLEDGE, known_at, po.id,
                       {"level": po.level.name, "clusters": po.clusters, "post_mean": round(po.post_mean, 8),
                        "qualifies": po.qualifies})
        counts["nodes"] += 1
    for prop in h.edge_proposals():
        graph.add_edge(prop.src, prop.dst, prop.rel, known_at, prop.weight, prop.evidence, dict(prop.attrs))
        counts["specializes"] += 1
    for child, parent in h.reversals():
        graph.add_edge(node_id(child), node_id(parent), Edge.CONTRADICTS, known_at, weight=0.5,
                       attrs={"source": "hierarchy", "investigated": False, "dimension": child[-1][0]})
        counts["reversals"] += 1
    return counts


# ------------------------------------------------------------------------------------------- walk-forward timeline

@dataclasses.dataclass(frozen=True)
class QualEvent:
    """A node's qualification state changed (or was first seen) at a walk-forward checkpoint."""
    node: str
    at: str
    qualified: bool
    post_mean: float
    z_vs_parent: float
    clusters: int


class HierarchyTimeline:
    """Refit the hierarchy at successive dates using only rows before each date, and remember when each rule started or
    stopped earning its override. A rule that flips on and off is fitting noise or a moving regime; the ledger (a lane on the
    hash chain, resumable) is the evidence. Nothing is recomputed for a checkpoint already recorded."""

    def __init__(self, params: Mapping | None = None, path=None):
        from .archive import ChainFile
        self.params = dict(params or {})
        self._chain = ChainFile(path, "hier")
        self.events: list[QualEvent] = []
        self.checkpoints: dict[str, str] = {}
        self._state: dict[str, bool] = {}
        self._load()

    def _load(self):
        for line in self._chain.take_new():
            b = line["body"]
            if b["op"] == "event":
                ev = QualEvent(b["node"], b["at"], bool(b["qualified"]), float(b["post_mean"]), float(b["z_vs_parent"]),
                               int(b["clusters"]))
                self.events.append(ev)
                self._state[ev.node] = ev.qualified
            else:
                self.checkpoints[b["at"]] = b["fingerprint"]

    def run(self, rows: pd.DataFrame, checkpoints: Iterable) -> list[QualEvent]:
        f = rows.copy()
        f["when"] = pd.to_datetime(f["when"])
        new: list[QualEvent] = []
        for c in sorted({as_date(x) for x in checkpoints}):
            iso = c.isoformat()
            if iso in self.checkpoints:
                continue
            h = KnowledgeHierarchy(self.params).fit(f[f["when"] < pd.Timestamp(c)], c)
            h._compute()
            bodies = []
            for p in h.paths():
                if not p:
                    continue
                po = h._post[p]
                if self._state.get(po.id) != po.qualifies:
                    ev = QualEvent(po.id, iso, po.qualifies, po.post_mean, po.z_vs_parent, po.clusters)
                    bodies.append({"op": "event", **dataclasses.asdict(ev)})
                    self._state[po.id] = po.qualifies
                    new.append(ev)
            bodies.append({"op": "checkpoint", "at": iso, "fingerprint": h.fingerprint(), "nodes": len(h._post)})
            self._chain.append_many(bodies)
            self._chain.take_new()
            self.checkpoints[iso] = h.fingerprint()
            self.events.extend(e for e in new if e.at == iso)
        return new

    def history(self, node: str, now=None) -> list[QualEvent]:
        n = as_date(now) if now is not None else None
        return [e for e in self.events if e.node == node and (n is None or as_date(e.at) < n)]

    def qualified_at(self, now) -> list[str]:
        n = as_date(now)
        state: dict[str, bool] = {}
        for e in self.events:
            if as_date(e.at) < n:
                state[e.node] = e.qualified
        return sorted(k for k, v in state.items() if v)

    def flips(self, node: str, now=None) -> int:
        h = self.history(node, now)
        return sum(1 for a, b in zip(h, h[1:]) if a.qualified != b.qualified)

    def first_qualified(self, node: str) -> str | None:
        return next((e.at for e in self.events if e.node == node and e.qualified), None)

    def flaky(self, min_flips: int = 2, now=None) -> list[str]:
        nodes = sorted({e.node for e in self.events})
        return [n for n in nodes if self.flips(n, now) >= min_flips]

    def stability(self, now=None) -> float | None:
        """Share of ever-qualified rules that never flipped after first qualifying (None if none ever qualified)."""
        ever = sorted({e.node for e in self.events if e.qualified and (now is None or as_date(e.at) < as_date(now))})
        if not ever:
            return None
        return sum(1 for n in ever if self.flips(n, now) == 0) / len(ever)


# ------------------------------------------------------------------------------------------- calibration & sensitivity

def coverage_check(train: pd.DataFrame, test: pd.DataFrame, now_train, params: Mapping | None = None,
                   min_test_clusters: int = 8) -> dict[str, float]:
    """Are the stated uncertainties honest? For every rule with enough history in BOTH samples compare the later sample's
    mean with (a) the shrunk posterior and (b) the raw early mean. Standardised gaps use both sides' variances; a calibrated
    95% interval covers about 95%. Shrinkage should cover at least as often as raw cells, most of all for small ones."""
    a = KnowledgeHierarchy(params).fit(train, now_train)
    a._compute()
    b = KnowledgeHierarchy(params).fit(test, pd.Timestamp(test["when"].max()) + pd.Timedelta(days=1))
    b._compute()
    z_post, z_raw, weights = [], [], []
    for p, po in a._post.items():
        if not p:
            continue
        tb = b._post.get(p)
        if tb is None or tb.clusters < min_test_clusters or not math.isfinite(tb.raw_var):
            continue
        z_post.append((tb.raw_mean - po.post_mean) / math.sqrt(po.post_var + tb.raw_var))
        if math.isfinite(po.raw_var):
            z_raw.append((tb.raw_mean - po.raw_mean) / math.sqrt(po.raw_var + tb.raw_var))
        weights.append(po.weight_own)
    if not z_post:
        return {"nodes": 0.0, "coverage_post": float("nan"), "coverage_raw": float("nan")}
    zc = 1.96
    return {"nodes": float(len(z_post)), "coverage_post": float(np.mean(np.abs(z_post) < zc)),
            "coverage_raw": float(np.mean(np.abs(z_raw) < zc)) if z_raw else float("nan"),
            "rms_post": float(np.sqrt(np.mean(np.square(z_post)))),
            "rms_raw": float(np.sqrt(np.mean(np.square(z_raw)))) if z_raw else float("nan"),
            "mean_w_own": float(np.mean(weights))}


def parameter_sweep(rows: pd.DataFrame, now, grid: Mapping[str, Sequence], base: Mapping | None = None) -> pd.DataFrame:
    """How many rules qualify under each setting of the safeguards? One row per grid point (cartesian product), with the
    fitted hierarchy's fingerprint so any two settings that produce the same knowledge are visibly identical."""
    import itertools
    keys = sorted(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**(base or {}), **dict(zip(keys, combo))}
        h = KnowledgeHierarchy(params).fit(rows, now)
        h._compute()
        qual = h.qualified()
        out.append({**dict(zip(keys, combo)), "nodes": len(h._post), "qualified": len(qual),
                    "qualified_ids": ",".join(node_id(p) for p in qual), "fingerprint": h.fingerprint()})
    return pd.DataFrame(out)


def robust_rules(sweep: pd.DataFrame, min_share: float = 0.75) -> list[str]:
    """Rules qualifying in at least `min_share` of the swept settings: distinctions that do not depend on a lucky threshold."""
    if sweep.empty:
        return []
    counts: dict[str, int] = defaultdict(int)
    for ids in sweep["qualified_ids"]:
        for i in filter(None, ids.split(",")):
            counts[i] += 1
    need = min_share * len(sweep)
    return sorted(k for k, c in counts.items() if c >= need)


# ------------------------------------------------------------------------------------------- one hierarchy per knowledge item

class HierarchyBank:
    """Many knowledge items, each with its own hierarchy. Answers 'what do all my beliefs say about THIS context?', abstains
    (Unknown) where a belief has no evidence, and converts qualified rules into the context / anti-context maps a knowledge
    object stores (section 8): where the effect is positive and earned, and where it is negative and earned."""

    def __init__(self, params: Mapping | None = None):
        self.params = dict(params or {})
        self._h: dict[str, KnowledgeHierarchy] = {}

    def __len__(self) -> int:
        return len(self._h)

    def fit(self, knowledge_id: str, rows: pd.DataFrame, now) -> KnowledgeHierarchy:
        h = KnowledgeHierarchy(self.params).fit(rows, now)
        self._h[str(knowledge_id)] = h
        return h

    def update(self, knowledge_id: str, rows: pd.DataFrame, now) -> int:
        h = self._h.setdefault(str(knowledge_id), KnowledgeHierarchy(self.params))
        return h.update(rows, now)

    def get(self, knowledge_id: str) -> KnowledgeHierarchy:
        try:
            return self._h[str(knowledge_id)]
        except KeyError:
            raise HierarchyError(f"no hierarchy for {knowledge_id}") from None

    def estimates(self, context: Mapping[str, Any]) -> pd.DataFrame:
        rows = []
        for kid, h in sorted(self._h.items()):
            e = h.estimate(context)
            rows.append({"knowledge_id": kid, "known": e.is_known(), "mean": e.mean, "se": e.se, "level": e.level.name,
                         "used": node_id(e.used_path) if e.is_known() else None, "p_sign_capped": e.p_sign_capped,
                         "state": None if e.state is None else e.state.value})
        return pd.DataFrame(rows, columns=["knowledge_id", "known", "mean", "se", "level", "used", "p_sign_capped", "state"])

    def rank(self, context: Mapping[str, Any], min_p_sign: float = 0.0) -> list[str]:
        """Known beliefs ordered by their sample-capped confidence in the sign of the effect, then by |mean|; unknowns and
        low-confidence beliefs are left out rather than ranked on a guess."""
        df = self.estimates(context)
        df = df[df["known"].astype(bool)]
        df = df[df["p_sign_capped"].astype(float) >= min_p_sign]
        df = df.assign(_a=df["mean"].astype(float).abs()).sort_values(["p_sign_capped", "_a", "knowledge_id"],
                                                                      ascending=[False, False, True])
        return list(df["knowledge_id"])

    def context_rules(self, knowledge_id: str) -> list[dict]:
        """Every earned rule as {'when': {dim: value}, 'effect', 'se', 'level', 'clusters'}, general rule first."""
        h = self.get(knowledge_id)
        h._compute()
        out = []
        for p in h.paths():
            po = h._post[p]
            if p and not po.qualifies:
                continue
            out.append({"when": dict(p), "effect": po.post_mean, "se": math.sqrt(max(po.post_var, 0.0)),
                        "level": po.level.name, "clusters": po.clusters})
        return out

    def to_contexts(self, knowledge_id: str) -> tuple[dict[str, str], dict[str, str]]:
        """(contexts, anti_contexts): earned rules with a positive / negative effect, each keyed 'dim|dim' -> 'val|val'.
        The general rule contributes nothing - it is the baseline, not a condition."""
        pos: dict[str, str] = {}
        neg: dict[str, str] = {}
        for r in self.context_rules(knowledge_id):
            if not r["when"]:
                continue
            key, val = "|".join(r["when"]), "|".join(r["when"].values())
            (pos if r["effect"] > 0 else neg)[key] = val
        return pos, neg


def compare_bank(bank: HierarchyBank, contexts: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Long table of every belief's estimate for a list of contexts (for inspection pages and tests)."""
    frames = []
    for c in contexts:
        df = bank.estimates(c)
        df.insert(0, "context", json.dumps(dict(c), sort_keys=True, default=str))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
