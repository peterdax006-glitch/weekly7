"""Bible Phase 3.3 + 3.4 - statistical validation library and relevance weighting (canons C35, C37).

The blueprint's "Is it real?" framework (MASTER_BLUEPRINT section 30) as small, separately testable functions:

  week-clustered observations -> weighted mean -> effective sample size -> t -> p -> Benjamini-Hochberg over every
  candidate tried (P(coincidence)); outcomes shuffled inside each cluster -> null |t| distribution -> local false-discovery
  rate (P(hallucinated)); the later 30% of dates as confirmation (sign must survive) ->
      P(real) = (1 - max(P(hallucinated), corrected P(coincidence))) x Phi(t_confirm)
  effect = mean x n_eff / (n_eff + k), plus an empirical-Bayes alternative that estimates the prior from the candidates.

Every function here has an equivalent in engine/patterns.py; tests/test_pattern_stats.py runs both on the same inputs.
Where the library deliberately differs it is stated at the function: (1) `corrected P(coincidence)` defaults to the
blueprint's BH-adjusted q-value (patterns.py multiplies p by the candidate count, i.e. Bonferroni; method="bonferroni"
reproduces that exactly); (2) `hac_cluster_test` corrects overlapping forward returns, which per-date clustering
(patterns.py) does not.  Relevance (3.3) is recency x context similarity x era weight, computed point-in-time: rows after
`now` never receive weight and the context scale uses only rows up to `now`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

STAT_DEFAULT = {"fdr_q": 0.05, "shrink_k": 400.0, "min_n": 300, "conf_frac": 0.3, "recent_frac": 0.2,
                "null_reps": 2, "null_max_patterns": 1500, "p_real_min": 0.8, "min_clusters": 5,
                "min_recent_rows": 30, "p_method": "bh"}
VAR_FLOOR = 1e-12


class StatsError(ValueError):
    """Inputs that cannot be tested honestly (mismatched lengths, NaNs, look-ahead)."""


# ---------------------------------------------------------------- normal distribution helpers
def norm_cdf(x):
    """Phi(x), vectorised, exact via erfc (no cancellation in the tails)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(math.erfc, otypes=[float])(-x / math.sqrt(2.0)) if x.ndim else \
        0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def t_to_p(t):
    """Two-sided normal p-value of a t-statistic. Same quantity as patterns._t_to_p; erfc keeps it accurate where
    1 - erf(x) rounds to exactly 0 (|t| above ~8.3)."""
    a = np.abs(np.asarray(t, dtype=float))
    if a.ndim:
        return np.vectorize(lambda v: math.erfc(v / math.sqrt(2.0)), otypes=[float])(a)
    return math.erfc(float(a) / math.sqrt(2.0))


# ---------------------------------------------------------------- weighted moments
def _as_float(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise StatsError(f"{name} must be one-dimensional")
    return a


def weighted_mean(w, y) -> float:
    w, y = _as_float(w, "w"), _as_float(y, "y")
    if len(w) != len(y):
        raise StatsError("w and y differ in length")
    if (w < 0).any():
        raise StatsError("weights must be non-negative")
    sw = w.sum()
    return float((w * y).sum() / sw) if sw > 0 else 0.0


def effective_n(w) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2: n for equal weights, shrinking as relevance concentrates."""
    w = _as_float(w, "w")
    if (w < 0).any():
        raise StatsError("weights must be non-negative")
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def weighted_stats(w, y):
    """Row-level (mean, t, n_eff). Mirrors patterns._wstats. Rows of one week are NOT independent, so this t is the
    over-confident stock-level statistic the blueprint replaced; it exists for comparison and the cause search."""
    w, y = _as_float(w, "w"), _as_float(y, "y")
    sw = w.sum()
    if sw <= 0:
        return 0.0, 0.0, 0.0
    m = float((w * y).sum() / sw)
    v = float((w * (y - m) ** 2).sum() / sw)
    n_eff = float(sw ** 2 / (w ** 2).sum())
    return m, m / math.sqrt(max(v, VAR_FLOOR) / max(n_eff, 1.0)), n_eff


# ---------------------------------------------------------------- clustering
def date_codes(dates) -> tuple:
    """(codes, n_clusters): one cluster per distinct date, in order of first appearance (patterns.py convention)."""
    codes, uniq = pd.factorize(pd.DatetimeIndex(dates))
    return codes.astype(np.int64), int(len(uniq))


def week_codes(dates) -> tuple:
    """(codes, n_clusters): one cluster per ISO year-week, coded in calendar order. Use this when observations are
    daily but the outcome spans a week - all stocks on all days of one week share that week's market move."""
    d = pd.DatetimeIndex(dates)
    iso = d.isocalendar()
    key = (iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)).values
    uniq, codes = np.unique(key, return_inverse=True)
    return codes.astype(np.int64), int(len(uniq))


@dataclass(frozen=True)
class ClusterStat:
    mean: float
    t: float
    n_eff: float
    n_clusters: int
    se: float = float("nan")

    def as_tuple(self):
        return self.mean, self.t, self.n_eff


def cluster_means(codes, w, y, mask, n_clusters):
    """Per-cluster weighted mean and weight over the rows in `mask`; empty clusters are dropped."""
    wm = w[mask]
    sw = np.bincount(codes[mask], weights=wm, minlength=n_clusters)
    sy = np.bincount(codes[mask], weights=wm * y[mask], minlength=n_clusters)
    ok = sw > 0
    return sy[ok] / sw[ok], sw[ok], np.flatnonzero(ok)


def cluster_test(codes, w, y, mask, n_clusters, min_clusters: int = 5) -> ClusterStat:
    """Week-clustered test: one observation per cluster, weighted by that cluster's relevance weight.

    Equivalent to the `ctest` closure in PatternMiner.fit: mean = weighted mean of cluster means, n_eff = Kish n of the
    cluster weights, t = mean / sqrt(variance / n_eff). Fewer than `min_clusters` populated clusters gives t = 0
    (nothing can be concluded from four weeks)."""
    if mask.sum() == 0:
        return ClusterStat(0.0, 0.0, 0.0, 0)
    wk_mean, ww, _ = cluster_means(codes, w, y, mask, n_clusters)
    if len(wk_mean) < min_clusters:
        return ClusterStat(0.0, 0.0, float(len(wk_mean)), int(len(wk_mean)))
    m = float((ww * wk_mean).sum() / ww.sum())
    v = float((ww * (wk_mean - m) ** 2).sum() / ww.sum())
    n_eff = float(ww.sum() ** 2 / (ww ** 2).sum())
    se = math.sqrt(max(v, VAR_FLOOR) / max(n_eff, 1.0))
    return ClusterStat(m, m / se, n_eff, int(len(wk_mean)), se)


def hac_cluster_test(codes, w, y, mask, n_clusters, lags: int, min_clusters: int = 5,
                     kernel: str = "bartlett") -> ClusterStat:
    """Cluster test whose standard error survives OVERLAPPING outcomes (Newey-West / Bartlett kernel on the cluster
    series). A 5-day forward return measured every day makes consecutive clusters share four days of price path, so
    per-date clustering (what patterns.py does) counts each independent week about five times. Pass lags = horizon - 1.
    Clusters missing from the mask contribute zero to the score series, which keeps their calendar gap honest.
    kernel='bartlett' (Newey-West, always positive, but with lags = h-1 it under-corrects an exact MA(h-1) overlap by
    roughly 20% in se); kernel='uniform' (Hansen-Hodrick) is exact for that structure and can be noisier."""
    if lags < 0:
        raise StatsError("lags must be >= 0")
    if kernel not in ("bartlett", "uniform"):
        raise StatsError(f"unknown kernel {kernel!r}")
    if mask.sum() == 0:
        return ClusterStat(0.0, 0.0, 0.0, 0)
    wk_mean, ww, where = cluster_means(codes, w, y, mask, n_clusters)
    if len(wk_mean) < min_clusters:
        return ClusterStat(0.0, 0.0, float(len(wk_mean)), int(len(wk_mean)))
    m = float((ww * wk_mean).sum() / ww.sum())
    e = np.zeros(n_clusters)
    e[where] = ww * (wk_mean - m) / ww.sum()               # influence of each cluster on the mean
    var = float((e ** 2).sum())
    for lag in range(1, min(lags, n_clusters - 1) + 1):
        wt = 1.0 - lag / (lags + 1.0) if kernel == "bartlett" else 1.0
        var += 2.0 * wt * float(np.dot(e[:-lag], e[lag:]))
    se = math.sqrt(max(var, VAR_FLOOR))
    return ClusterStat(m, m / se, float(ww.sum() ** 2 / (ww ** 2).sum()), int(len(wk_mean)), se)


def cluster_diagnostics(codes, w, y, mask, n_clusters) -> dict:
    """Why a cluster t may be untrustworthy: too few clusters, one cluster carrying the weight, serial correlation."""
    wk_mean, ww, where = cluster_means(codes, w, y, mask, n_clusters)
    out = {"n_clusters": int(len(wk_mean)), "max_weight_share": float(ww.max() / ww.sum()) if len(ww) else float("nan"),
           "lag1_autocorr": float("nan")}
    if len(wk_mean) >= 8:
        x = wk_mean - wk_mean.mean()
        denom = float((x ** 2).sum())
        adj = np.diff(where) == 1                         # only truly adjacent clusters count as neighbours
        if denom > 0 and adj.any():
            out["lag1_autocorr"] = float((x[:-1] * x[1:])[adj].sum() / denom * len(adj) / max(adj.sum(), 1))
    return out


# ---------------------------------------------------------------- multiple testing
def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values): reject at level q iff q_value <= q."""
    p = _as_float(p, "p")
    if np.isnan(p).any():
        raise StatsError("p-values contain NaN")
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def bh_reject(p, q: float = 0.05) -> np.ndarray:
    """The BH step-up rejection set, written as patterns.py writes it: the largest k with p_(k) <= q k / m, reject the
    k smallest. Equal to `bh_qvalues(p) <= q` (tested), kept literal so the equivalence to the miner is auditable."""
    p = _as_float(p, "p")
    m = len(p)
    rej = np.zeros(m, bool)
    if m == 0:
        return rej
    order = np.argsort(p, kind="mergesort")
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    if passed.any():
        rej[order[: int(np.max(np.flatnonzero(passed))) + 1]] = True
    return rej


def bonferroni(p, m: Optional[int] = None) -> np.ndarray:
    p = _as_float(p, "p")
    return np.minimum(1.0, p * (len(p) if m is None else m))


def by_qvalues(p) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values: BH valid under ARBITRARY dependence, q_BH x sum_{i<=m} 1/i (capped at 1).
    Candidates here are heavily dependent (pairs and exceptions are built from the strongest singles), which is the
    regime where plain BH's independence / positive-dependence assumption fails."""
    p = _as_float(p, "p")
    m = len(p)
    if m == 0:
        return p.copy()
    return np.minimum(1.0, bh_qvalues(p) * float((1.0 / np.arange(1, m + 1)).sum()))


def corrected_coincidence(p, method: str = "bh") -> np.ndarray:
    """P(coincidence) after correcting for every candidate tried. 'bh' = blueprint section 30 (BH adjusted, q = 0.05
    is applied later as the admission threshold); 'bonferroni' = engine/patterns.py today (min(1, p x m))."""
    if method == "bh":
        return bh_qvalues(p)
    if method == "by":
        return by_qvalues(p)
    if method == "bonferroni":
        return bonferroni(p)
    raise StatsError(f"unknown correction {method!r}")


# ---------------------------------------------------------------- permutation null and local fdr
def permute_within_clusters(y, codes, rng: np.random.Generator) -> np.ndarray:
    """Shuffle outcomes among the rows of each cluster. Keeps every cluster's own mean and spread (so a good week
    stays a good week) while destroying any link between a row's features and its outcome. Vectorised: sort rows by
    (cluster, random key) and deal the cluster's outcomes back in that order."""
    y = np.asarray(y)
    codes = np.asarray(codes)
    if len(y) != len(codes):
        raise StatsError("y and codes differ in length")
    home = np.argsort(codes, kind="stable")                   # rows grouped by cluster, original order inside
    shuffled = np.lexsort((rng.random(len(y)), codes))        # same groups, random order inside
    out = np.empty_like(y)
    out[home] = y[shuffled]
    return out


def null_t_distribution(masks: Sequence[np.ndarray], stat_fn: Callable, y, codes, rng: np.random.Generator,
                        reps: int = 2, max_patterns: int = 1500) -> np.ndarray:
    """|t| of the same search re-run on shuffled outcomes. `stat_fn(mask, y_perm)` returns a t-statistic or None when
    the pattern is too thin to test (exactly as the real run treats it). Sorted ascending, ready for local_fdr.
    Returns array([0.0]) if nothing was testable, the miner's convention."""
    out = []
    n = len(masks)
    for _ in range(reps):
        yperm = permute_within_clusters(y, codes, rng)
        idx = np.arange(n) if n <= max_patterns else rng.choice(n, max_patterns, replace=False)
        for i in idx:
            t = stat_fn(masks[int(i)], yperm)
            if t is not None:
                out.append(abs(float(t)))
    return np.sort(np.array(out)) if out else np.array([0.0])


def local_fdr(t, null_t, real_t) -> float:
    """(share of null patterns at least this strong) / (share of real patterns at least this strong), capped at 1.
    `null_t` and `real_t` are ascending |t| arrays. Identical to the closure in PatternMiner.fit."""
    null_t, real_t = np.asarray(null_t), np.asarray(real_t)
    if len(null_t) == 0:
        null_t = np.array([0.0])
    frac_null = 1 - np.searchsorted(null_t, t) / len(null_t)
    frac_real = max(1 - np.searchsorted(real_t, t) / len(real_t), 1e-9)
    return float(min(1.0, frac_null / frac_real))


def local_fdr_monotone(t, null_t, real_t) -> np.ndarray:
    """local_fdr for many |t| with the natural constraint that a stronger t can never look MORE hallucinated than a
    weaker one (the raw ratio of two empirical tail shares is jumpy). Running minimum from the weakest upward."""
    t = np.asarray(t, dtype=float)
    raw = np.array([local_fdr(v, null_t, real_t) for v in t])
    order = np.argsort(t, kind="mergesort")
    fixed = np.minimum.accumulate(raw[order])
    out = np.empty_like(raw)
    out[order] = fixed
    return out


# ---------------------------------------------------------------- confirmation, P(real), effect
def split_dates(dates, conf_frac: float = 0.3, recent_frac: float = 0.2):
    """(split, recent_start): first date of the confirmation block and of the recent stretch, computed from the
    sorted unique dates exactly as fit() does (discovery = earlier 70%)."""
    if not 0 < conf_frac < 1 or not 0 < recent_frac < 1:
        raise StatsError("fractions must lie in (0, 1)")
    ud = np.sort(pd.DatetimeIndex(dates).unique())
    if len(ud) < 4:
        raise StatsError("need at least 4 distinct dates to split discovery from confirmation")
    split = ud[min(int(len(ud) * (1 - conf_frac)), len(ud) - 1)]
    recent = ud[min(int(len(ud) * (1 - recent_frac)), len(ud) - 1)]
    return split, recent


def confirmation_factor(t_conf, m_conf, m_disc):
    """Phi(|t_conf|) if the confirmation block kept the discovery sign, else 0. Vectorised."""
    t_conf, m_conf, m_disc = (np.asarray(a, dtype=float) for a in (t_conf, m_conf, m_disc))
    return np.where(np.sign(m_conf) == np.sign(m_disc), norm_cdf(np.abs(t_conf)), 0.0)


def p_real(p_halluc, p_coincidence, factor, method: str = "bh", n_tests: Optional[int] = None):
    """P(real) = (1 - max(P(hallucinated), corrected P(coincidence))) x confirmation factor.

    `p_coincidence` is the RAW p-value vector over every candidate tried (correction needs the whole family).
    method='bh' -> Benjamini-Hochberg adjusted (blueprint); 'bonferroni' -> min(1, p x m), which is what
    patterns.py computes today. A pattern that fails confirmation scores exactly 0 either way."""
    ph = np.asarray(p_halluc, dtype=float)
    pc = np.asarray(p_coincidence, dtype=float)
    f = np.asarray(factor, dtype=float)
    if not (len(ph) == len(pc) == len(f)):
        raise StatsError("p_halluc, p_coincidence and factor must align")
    corr = np.minimum(1.0, pc * (len(pc) if n_tests is None else n_tests)) if method == "bonferroni" \
        else corrected_coincidence(pc, method)
    return (1.0 - np.maximum(ph, corr)) * f


def shrink_effect(mean, n_eff, k: float = 400.0):
    """The blueprint's effect size: mean x n_eff / (n_eff + k)."""
    mean, n_eff = np.asarray(mean, dtype=float), np.asarray(n_eff, dtype=float)
    return mean * n_eff / (n_eff + k)


def eb_shrink(means, ses, center: Optional[float] = 0.0) -> dict:
    """Empirical-Bayes shrinkage of many effects toward a common prior estimated from the candidates themselves.

    Model: true effect ~ N(mu, tau^2), observed ~ N(true, se_i^2). tau^2 by DerSimonian-Laird moments across the
    candidates (so it absorbs the winner's curse: a search that returns mostly noise learns a small tau and shrinks
    hard). `center=0.0` shrinks toward zero (a pattern with no evidence should predict nothing); center=None estimates mu.
    Returns posterior means, posterior sds, per-pattern shrink factor B = tau^2/(tau^2 + se^2), tau2 and mu;
    `implied_k` converts tau2 into the k of the fixed n_eff/(n_eff + k) rule for comparison."""
    m, s = _as_float(means, "means"), _as_float(ses, "ses")
    if len(m) != len(s):
        raise StatsError("means and ses differ in length")
    good = np.isfinite(m) & np.isfinite(s) & (s > 0)
    if good.sum() < 2:
        return {"post_mean": m.copy(), "post_sd": s.copy(), "B": np.zeros(len(m)), "tau2": 0.0,
                "mu": 0.0 if center is None else float(center), "n_used": int(good.sum())}
    mg, sg = m[good], s[good]
    wi = 1.0 / sg ** 2
    mu_fe = float((wi * mg).sum() / wi.sum())
    Q = float((wi * (mg - mu_fe) ** 2).sum())
    C = float(wi.sum() - (wi ** 2).sum() / wi.sum())
    tau2 = max(0.0, (Q - (good.sum() - 1)) / C) if C > 0 else 0.0
    if center is None:
        wr = 1.0 / (sg ** 2 + tau2)
        mu = float((wr * mg).sum() / wr.sum())
    else:
        # fixed centre: E[(m - mu)^2] = tau^2 + se^2, so the second moment about mu gives tau^2 directly
        mu = float(center)
        tau2 = max(0.0, float(np.mean((mg - mu) ** 2 - sg ** 2)))
    B = np.zeros(len(m))
    B[good] = tau2 / (tau2 + sg ** 2)
    post = m.copy()
    post[good] = mu + B[good] * (m[good] - mu)
    post_sd = s.copy()
    post_sd[good] = np.sqrt(B[good]) * sg
    return {"post_mean": post, "post_sd": post_sd, "B": B, "tau2": float(tau2), "mu": float(mu),
            "n_used": int(good.sum())}


def implied_k(tau2: float, se: Sequence[float], n_eff: Sequence[float]) -> float:
    """The k of `n_eff / (n_eff + k)` that reproduces EB shrinkage: B = tau^2/(tau^2 + se^2) and se^2 = v/n_eff give
    k = v/tau^2 with v = se^2 x n_eff. Returned as the median over patterns; inf when tau^2 = 0 (shrink everything)."""
    se, n = _as_float(se, "se"), _as_float(n_eff, "n_eff")
    if tau2 <= 0:
        return float("inf")
    v = se ** 2 * n
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v) / tau2) if len(v) else float("nan")


# ---------------------------------------------------------------- the whole validation, in one pass
def evaluate_masks(masks: Sequence[np.ndarray], y, w, dates, params: Optional[dict] = None,
                   seed: int = 7, names: Optional[Sequence[str]] = None, codes=None, n_clusters=None,
                   hac_lags: Optional[int] = None) -> pd.DataFrame:
    """Blueprint section 30 over a list of row masks.

    y: forward excess return per row; w: relevance weight per row (see `relevance`); dates: per-row dates.
    Discovery = earlier (1 - conf_frac) of dates, confirmation = later conf_frac. Patterns too thin to test are
    dropped, as the miner drops them, and never enter the multiple-testing count. `hac_lags` switches the tests to
    the overlap-robust standard error. Returns one row per tested pattern with columns m_disc, t_disc, m_conf,
    t_conf, m_all, t_all, n_eff, m_recent, t_recent, n_recent, p_coincidence, q_value, fdr_pass, p_hallucinated,
    conf_factor, p_real, effect, effect_eb, `idx` (position in `masks`)."""
    P = {**STAT_DEFAULT, **(params or {})}
    y, w = _as_float(y, "y"), _as_float(w, "w")
    if len(y) != len(w) or len(y) != len(dates):
        raise StatsError("y, w and dates must have the same length")
    if np.isnan(y).any() or np.isnan(w).any():
        raise StatsError("y and w must not contain NaN (drop unlabeled rows first)")
    dates = pd.DatetimeIndex(dates)
    if codes is None:
        codes, n_clusters = date_codes(dates)
    elif n_clusters is None:
        n_clusters = int(np.max(codes)) + 1
    split, recent = split_dates(dates, P["conf_frac"], P["recent_frac"])
    disc, conf, rec = dates < split, dates >= split, dates >= recent
    rng = np.random.default_rng(seed)

    def ctest(mask, yy):
        if hac_lags is None:
            return cluster_test(codes, w, yy, mask, n_clusters, P["min_clusters"])
        return hac_cluster_test(codes, w, yy, mask, n_clusters, hac_lags, P["min_clusters"])

    def t_disc_only(mask, yy):
        md = mask & disc
        if md.sum() < P["min_n"] or (mask & conf).sum() < P["min_n"] // 3:
            return None
        return ctest(md, yy).t

    rows, kept = [], []
    for i, mask in enumerate(masks):
        md, mc = mask & disc, mask & conf
        if md.sum() < P["min_n"] or mc.sum() < P["min_n"] // 3:
            continue
        d, c, a = ctest(md, y), ctest(mc, y), ctest(mask, y)
        mr = mask & rec
        r = ctest(mr, y) if mr.sum() >= P["min_recent_rows"] else ClusterStat(float("nan"), 0.0, 0.0, 0)
        rows.append((d.mean, d.t, c.mean, c.t, a.mean, a.t, a.n_eff, r.mean, r.t, r.n_eff, a.se))
        kept.append(i)
    cols = ["m_disc", "t_disc", "m_conf", "t_conf", "m_all", "t_all", "n_eff", "m_recent", "t_recent", "n_recent",
            "se_all"]
    R = pd.DataFrame(rows, columns=cols)
    R["idx"] = kept
    R["name"] = [names[i] for i in kept] if names is not None else [f"pattern_{i}" for i in kept]
    if R.empty:
        for c in ("p_coincidence", "q_value", "fdr_pass", "p_hallucinated", "conf_factor", "p_real", "effect",
                  "effect_eb"):
            R[c] = pd.Series(dtype=float)
        return R
    pv = np.asarray(t_to_p(R["t_disc"].values))
    R["p_coincidence"] = pv
    R["q_value"] = bh_qvalues(pv)
    R["fdr_pass"] = bh_reject(pv, P["fdr_q"])
    null_t = null_t_distribution([masks[i] for i in kept], t_disc_only, y, codes, rng, P["null_reps"],
                                 P["null_max_patterns"])
    real_t = np.sort(R["t_disc"].abs().values)
    R["p_hallucinated"] = [local_fdr(abs(t), null_t, real_t) for t in R["t_disc"]]
    R["conf_factor"] = confirmation_factor(R["t_conf"].abs().values, R["m_conf"].values, R["m_disc"].values)
    R["p_real"] = p_real(R["p_hallucinated"].values, pv, R["conf_factor"].values, P["p_method"])
    R["effect"] = shrink_effect(R["m_all"].values, R["n_eff"].values, P["shrink_k"])
    eb = eb_shrink(R["m_all"].values, R["se_all"].values, center=0.0)
    R["effect_eb"] = eb["post_mean"]
    R.attrs.update({"null_n": int(len(null_t)), "null_t_95": float(np.quantile(null_t, 0.95)),
                    "real_t_95": float(np.quantile(real_t, 0.95)), "tau2": eb["tau2"], "n_tested": int(len(R)),
                    "split": str(pd.Timestamp(split).date()), "recent": str(pd.Timestamp(recent).date())})
    return R


def calibration_table(p, truth, edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)) -> pd.DataFrame:
    """Does P(real) mean what it says? Bin patterns by P(real) and compare with how many were truly real (planted).
    A well-calibrated bin at 0.8-1.0 has at least ~80% real. Only meaningful with planted truth (tests, scripts)."""
    p, truth = _as_float(p, "p"), np.asarray(truth, dtype=bool)
    if len(p) != len(truth):
        raise StatsError("p and truth differ in length")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi)
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(sel.sum()),
                     "real_share": float(truth[sel].mean()) if sel.any() else float("nan"),
                     "mean_p": float(p[sel].mean()) if sel.any() else float("nan")})
    return pd.DataFrame(rows)


def admission_flags(R: pd.DataFrame, p_real_min: float = 0.8) -> pd.Series:
    """Statistical part of admission (3.5): P(real) at threshold and sign preserved by confirmation. Redundancy and the
    validation-gain gate are separate gates and are not reproduced here."""
    if R.empty:
        return pd.Series(dtype=bool)
    return (R["p_real"] >= p_real_min) & (np.sign(R["m_conf"]) == np.sign(R["m_disc"]))


# ---------------------------------------------------------------- relevance (3.3)
class LookAheadError(StatsError):
    """A relevance weight was requested for a moment earlier than data it would have to read."""


RELEVANCE_DEFAULT = {"half_life_years": 4.0, "ctx_bandwidth": 1.5, "on_future": "zero", "min_sd": 1e-9}
# Microstructure regimes (blueprint section 29, 'Modernity' [DESIGNED]): weights are LEARNED later; the default table
# is neutral (1.0 everywhere) so that switching the feature on changes nothing until a learned table is supplied.
ERA_BOUNDS = (("pre_decimal", None, "2001-04-09"), ("decimal", "2001-04-09", "2007-10-01"),
              ("electronic", "2007-10-01", "2015-01-01"), ("etf_dominant", "2015-01-01", None))
NEUTRAL_ERA_WEIGHTS = {"pre_decimal": 1.0, "decimal": 1.0, "electronic": 1.0, "etf_dominant": 1.0}
MICROSTRUCTURE_FEATURES = frozenset({
    "gap_today", "gap", "gap_filled", "overnight20", "intraday20", "close_loc", "vol_surge1", "vol_surge5",
    "range_compress", "atr_pct", "reversal_1d", "reversal_vs_week", "inside_day", "outside_day", "engulf", "streak",
    "cd_body", "cd_upwick", "cd_lowwick", "cd_pos", "cd_range", "day_vs_week_body"})


def microstructure_sensitive(features: Iterable[str]) -> bool:
    """True if the pattern reads a feature whose meaning changed with tick size, electronic trading or ETF flows."""
    return any(f in MICROSTRUCTURE_FEATURES for f in features)


def _check_point_in_time(dates, now, on_future):
    d = pd.DatetimeIndex(dates)
    fut = d > pd.Timestamp(now)
    if fut.any() and on_future == "raise":
        raise LookAheadError(f"{int(fut.sum())} row(s) are dated after now={pd.Timestamp(now).date()} "
                             f"(latest {d.max().date()})")
    return fut


def recency_weight(dates, now, half_life_years: float = 4.0, on_future: str = "zero") -> np.ndarray:
    """0.5 ** (age / half-life). Rows dated after `now` get weight 0 (patterns.py gives them weight 1, which is silent
    look-ahead if the caller ever slips); on_future='raise' turns that into an error."""
    if half_life_years <= 0:
        raise StatsError("half_life_years must be positive")
    fut = _check_point_in_time(dates, now, on_future)
    age = (pd.Timestamp(now) - pd.DatetimeIndex(dates)).days.values / 365.25
    w = 0.5 ** (np.maximum(age, 0.0) / half_life_years)
    return np.where(fut, 0.0, w)


def context_similarity(ctx, ctx_now, bandwidth: float = 1.5, dates=None, now=None, min_sd: float = 1e-9) -> np.ndarray:
    """exp(-d^2 / (2 bw^2)) with d the standardised distance between each row's market context and today's.

    Point-in-time: the standardising sd is computed from rows dated <= now only when `dates` and `now` are given
    (patterns.py uses every row it is handed). Missing context values count as 0 distance in that column - unknown is
    not evidence of dissimilarity - and rows dated after `now` get similarity 0."""
    C = np.asarray(ctx, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    cn = np.asarray(ctx_now, dtype=float).reshape(-1)
    if C.shape[1] != len(cn):
        raise StatsError("context matrix and today's context differ in width")
    if bandwidth <= 0:
        raise StatsError("bandwidth must be positive")
    past = np.ones(len(C), bool)
    if dates is not None and now is not None:
        past = pd.DatetimeIndex(dates) <= pd.Timestamp(now)
    ref = C[past] if past.any() else C
    sd = np.nanstd(ref, axis=0)
    sd = np.where(~np.isfinite(sd) | (sd < min_sd), 1.0, sd)
    diff = (C - cn) / sd
    diff = np.nan_to_num(diff, nan=0.0)
    sim = np.exp(-(diff ** 2).sum(axis=1) / (2 * bandwidth ** 2))
    return np.where(past, sim, 0.0)


def era_of(dates, bounds=ERA_BOUNDS) -> np.ndarray:
    """Era label per date from half-open [start, end) bounds; None bounds are open."""
    d = pd.DatetimeIndex(dates)
    lab = np.empty(len(d), dtype=object)
    for name, lo, hi in bounds:
        sel = np.ones(len(d), bool)
        if lo is not None:
            sel &= d >= pd.Timestamp(lo)
        if hi is not None:
            sel &= d < pd.Timestamp(hi)
        lab[sel] = name
    return lab


def era_weight(dates, weights: Optional[dict] = None, sensitive: bool = True, bounds=ERA_BOUNDS) -> np.ndarray:
    """Per-row era weight. Patterns that do not read microstructure-sensitive features get 1.0 in every era."""
    if not sensitive:
        return np.ones(len(pd.DatetimeIndex(dates)))
    table = {**NEUTRAL_ERA_WEIGHTS, **(weights or {})}
    unknown = set(table) - {b[0] for b in bounds}
    if unknown:
        raise StatsError(f"era weights name unknown eras: {sorted(unknown)}")
    if any(v < 0 for v in table.values()):
        raise StatsError("era weights must be non-negative")
    lab = era_of(dates, bounds)
    return np.array([table[l] for l in lab], dtype=float)


def relevance(dates, now, ctx=None, ctx_now=None, params: Optional[dict] = None, features: Iterable[str] = (),
              era_weights: Optional[dict] = None) -> np.ndarray:
    """Bible 3.3: relevance = recency_weight x context_similarity_weight x era_weight, point-in-time as of `now`.
    Without a context matrix the similarity factor is 1. `features` decides whether the era factor applies."""
    P = {**RELEVANCE_DEFAULT, **(params or {})}
    w = recency_weight(dates, now, P["half_life_years"], P["on_future"])
    if ctx is not None and ctx_now is not None:
        w = w * context_similarity(ctx, ctx_now, P["ctx_bandwidth"], dates, now, P["min_sd"])
    return w * era_weight(dates, era_weights, microstructure_sensitive(features))


def context_now(ctx, dates, now) -> np.ndarray:
    """Today's context as known at `now`: the mean over rows of the last date <= now (the market columns are constant
    within a date, so this equals patterns.py's first-row pick while staying defined if rows disagree)."""
    d = pd.DatetimeIndex(dates)
    ok = d <= pd.Timestamp(now)
    if not ok.any():
        raise LookAheadError("no context is dated on or before now")
    last = d[ok].max()
    return np.nanmean(np.asarray(ctx, dtype=float)[d == last], axis=0)


# ---------------------------------------------------------------- stability of evidence across blocks and eras
def block_edges(dates, n_blocks: int) -> list:
    """Split the sorted unique dates into n_blocks contiguous groups of (almost) equal date count: [(start, end), ...]."""
    if n_blocks < 1:
        raise StatsError("n_blocks must be >= 1")
    ud = np.sort(pd.DatetimeIndex(dates).unique())
    if len(ud) < n_blocks:
        raise StatsError(f"{len(ud)} distinct dates cannot fill {n_blocks} blocks")
    parts = np.array_split(np.arange(len(ud)), n_blocks)
    return [(ud[p[0]], ud[p[-1]]) for p in parts]


def block_breakdown(mask, y, w, dates, n_blocks: int = 5, codes=None, n_clusters=None, min_clusters: int = 5) -> pd.DataFrame:
    """The pattern's cluster-t inside each of n_blocks consecutive date blocks. A real edge repeats; a fluke lives in
    one block. Blocks with too little data report t = 0 and are counted as 'thin', never as agreement."""
    dates = pd.DatetimeIndex(dates)
    if codes is None:
        codes, n_clusters = date_codes(dates)
    rows = []
    for lo, hi in block_edges(dates, n_blocks):
        inb = (dates >= lo) & (dates <= hi)
        st = cluster_test(codes, w, y, mask & inb, n_clusters, min_clusters)
        rows.append({"start": lo, "end": hi, "n_rows": int((mask & inb).sum()), "mean": st.mean, "t": st.t,
                     "n_eff": st.n_eff, "thin": st.n_clusters < min_clusters})
    return pd.DataFrame(rows)


def sign_agreement(breakdown: pd.DataFrame, overall_sign: float) -> dict:
    """How many non-thin blocks share the overall sign, and the share. `later confirmation` generalised from one
    30% block to every block - the lifecycle's confirm-on-every-window rule (blueprint section 36)."""
    live = breakdown[~breakdown["thin"]]
    if not len(live) or overall_sign == 0:
        return {"blocks": int(len(live)), "same_sign": 0, "share": float("nan")}
    same = int((np.sign(live["mean"]) == np.sign(overall_sign)).sum())
    return {"blocks": int(len(live)), "same_sign": same, "share": same / len(live)}


def era_breakdown(mask, y, w, dates, bounds=ERA_BOUNDS, min_clusters: int = 5) -> pd.DataFrame:
    """Cluster-t per microstructure era (relevance not applied inside an era beyond the weights passed in)."""
    dates = pd.DatetimeIndex(dates)
    codes, n = date_codes(dates)
    lab = era_of(dates, bounds)
    rows = []
    for name, _, _ in bounds:
        inera = lab == name
        st = cluster_test(codes, w, y, mask & inera, n, min_clusters)
        rows.append({"era": name, "n_rows": int((mask & inera).sum()), "mean": st.mean, "t": st.t,
                     "thin": st.n_clusters < min_clusters})
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(codes, w, y, mask, n_clusters, rng: np.random.Generator, reps: int = 400,
                         level: float = 0.95) -> tuple:
    """Percentile CI of the clustered weighted mean, resampling whole clusters with replacement. Independent of the
    normal approximation behind the t-statistic, so a large gap between the two is itself a warning."""
    wk_mean, ww, _ = cluster_means(codes, w, y, mask, n_clusters)
    if len(wk_mean) < 5:
        return float("nan"), float("nan")
    n = len(wk_mean)
    idx = rng.integers(0, n, size=(reps, n))
    num = (ww[idx] * wk_mean[idx]).sum(axis=1)
    den = ww[idx].sum(axis=1)
    est = num / den
    a = (1 - level) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


# ---------------------------------------------------------------- redundancy (3.6) and the validation-gain gate (3.7)
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|A and B| / |A or B| for two boolean row masks (0 when both are empty)."""
    uni = int((a | b).sum())
    return float((a & b).sum() / uni) if uni else 0.0


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Share of A's rows that are also in B. Jaccard misses a small pattern sitting entirely inside a big one;
    containment catches it."""
    na = int(a.sum())
    return float((a & b).sum() / na) if na else 0.0


def prune_redundant(order: Sequence[int], masks: Sequence[np.ndarray], max_overlap: float = 0.8,
                    use_containment: bool = False) -> tuple:
    """Walk patterns strongest-first; a later one that overlaps an already-kept one above `max_overlap` (Jaccard, or
    containment of the newcomer in the keeper) is a duplicate of it and never adds score on its own.
    `order` lists mask positions strongest first (the miner orders simpler-then-stronger). Returns
    (kept_positions, {dropped_position: keeper_position})."""
    kept, dup = [], {}
    for i in order:
        hit = None
        for j in kept:
            ov = containment(masks[i], masks[j]) if use_containment else jaccard(masks[i], masks[j])
            if ov > max_overlap:
                hit = j
                break
        if hit is None:
            kept.append(i)
        else:
            dup[i] = hit
    return kept, dup


def evidence_order(names: Sequence[str], t_disc, t_conf) -> list:
    """The miner's ordering for redundancy pruning: simpler expressions first (single < pair < unless), then by
    combined |t| of discovery and confirmation - NOT by raw effect, which a small noisy child wins by chance."""
    t_disc, t_conf = _as_float(t_disc, "t_disc"), _as_float(t_conf, "t_conf")
    k = np.array([n.count(" & ") + 2 * (" unless " in n) for n in names])
    strength = -(np.nan_to_num(np.abs(t_disc)) + np.nan_to_num(np.abs(t_conf)))
    return [int(i) for i in np.lexsort((np.arange(len(names)), strength, k))]


def validation_gain_gate(order: Sequence[int], masks: Sequence[np.ndarray], effects, y_conf, conf_mask,
                         min_gain: float = 0.0005) -> tuple:
    """Bible 3.7. Add patterns greedily; keep one only if the correlation of the running score with the outcome on the
    UNSEEN confirmation rows rises by more than `min_gain`. Returns (kept_positions, final_corr, trajectory)."""
    effects = _as_float(effects, "effects")
    yc = _as_float(y_conf, "y_conf")
    conf_mask = np.asarray(conf_mask, dtype=bool)
    if conf_mask.sum() != len(yc):
        raise StatsError("y_conf must have one value per confirmation row")
    score = np.zeros(len(conf_mask))
    best, kept, traj = 0.0, [], []
    for i in order:
        trial = score + effects[i] * masks[i]
        sc = trial[conf_mask]
        corr = float(np.corrcoef(sc, yc)[0, 1]) if sc.std() > 0 and yc.std() > 0 else 0.0
        gain = corr - best
        traj.append((int(i), gain))
        if gain > min_gain:
            score, best = trial, corr
            kept.append(int(i))
    return kept, best, traj
