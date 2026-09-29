"""Complexity penalty: a more complicated explanation must earn its complexity (contract C62 section 40).
IMPLEMENTED - NOT VALIDATED.

    simple rule = same OOS performance, complex rule = same OOS performance   ->  keep the simple one
    complex rule better on five measured things                               ->  complexity has earned its place

The five things measured (contract wording): incremental gain, transfer, stability, failure safety, complexity.  Complexity is
priced in *evidence*, not in return units: a rule with `d` extra complexity units must beat the simpler rule by a paired t of
z0 + kappa*d, so the bar is scale-free and rises with every clause added.  `calibrate_kappa` picks kappa so that a rule made of
pure noise clauses earns its place no more often than alpha (a planted-null calibration, not a hand-set constant).

Nothing here rejects complexity for being complicated: a rule whose gain is large, transfers across folds and does not worsen
the tail passes at any size.  Exceptions (hard-coded tickers/dates) are priced heavily: they are how memorisation hides.

Builds on engine.pattern_reliability.nw_t.  Time: this module receives already-out-of-sample series and never reads dated data
itself, so it has no `now`; callers are responsible for the OOS construction (see engine.learning.questions)."""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.learning.core import _StrEnum, stable_hash


class ComplexityError(ValueError):
    pass


@dc.dataclass(frozen=True)
class ComplexityWeights:
    feature: float = 1.0
    condition: float = 1.5                 # a gate / context clause
    threshold: float = 1.0                 # a free numeric cut point
    interaction: float = 2.0
    depth: float = 0.5                     # per level of rule nesting beyond the first
    free_param: float = 1.0                # fitted coefficient not counted above
    exception: float = 5.0                 # hard-coded exclusion: memorisation smell


DEFAULT_W = ComplexityWeights()


@dc.dataclass(frozen=True)
class RuleSpec:
    """Structural description of a rule or explanation.  Deliberately about STRUCTURE, not about fitted values."""
    rule_id: str
    n_features: int = 0
    n_conditions: int = 0
    n_thresholds: int = 0
    n_interactions: int = 0
    tree_depth: int = 1
    n_free_params: int = 0
    n_exceptions: int = 0

    def validate(self) -> list[str]:
        errs = []
        for f in dc.fields(self):
            if f.name != "rule_id" and getattr(self, f.name) < 0:
                errs.append(f"{f.name} < 0")
        if self.tree_depth < 1:
            errs.append("tree_depth < 1")
        return errs

    def units(self, w: ComplexityWeights = DEFAULT_W) -> float:
        e = self.validate()
        if e:
            raise ComplexityError("; ".join(e))
        return (w.feature * self.n_features + w.condition * self.n_conditions + w.threshold * self.n_thresholds
                + w.interaction * self.n_interactions + w.depth * (self.tree_depth - 1) + w.free_param * self.n_free_params
                + w.exception * self.n_exceptions)

    def n_params(self) -> int:
        """Rough count of numbers that must be estimated (for description length)."""
        return self.n_features + self.n_thresholds + self.n_free_params + self.n_interactions + 1

    def mdl_bits(self, n_obs: int) -> float:
        """Two-part-code cost of the parameters: 0.5*log2(n) bits each (Rissanen), plus a bit per structural clause."""
        return 0.5 * self.n_params() * math.log2(max(n_obs, 2)) + self.n_conditions + self.n_interactions + 8.0 * self.n_exceptions

    def identity_smell(self) -> bool:
        return self.n_exceptions > 0


def spec_from_hypothesis(rule_id: str, features: Sequence[str], gate: Sequence[tuple], interactions: int = 0) -> RuleSpec:
    """Structural spec of a linear-with-gates hypothesis (competition.HypothesisSpec) without importing it."""
    return RuleSpec(rule_id, n_features=len(features), n_conditions=len(gate), n_thresholds=len(gate),
                    n_interactions=interactions, tree_depth=1 + (1 if gate else 0), n_free_params=1)


def spec_from_knowledge(rule_id: str, contexts: Mapping, anti_contexts: Mapping, n_features: int = 1) -> RuleSpec:
    """Structural spec of a knowledge item's scope: each condition clause is a condition with a threshold."""
    def clauses(m):
        return sum(len(v) if isinstance(v, (list, tuple)) else 1 for v in (m or {}).values())
    c = clauses(contexts) + clauses(anti_contexts)
    return RuleSpec(rule_id, n_features=n_features, n_conditions=c, n_thresholds=c, tree_depth=1 + (1 if c else 0))


def occam_log_prior(units: float, lam: float = 0.5) -> float:
    """Log prior mass of a hypothesis of given complexity units (used by knowledge competition): -lam*units."""
    return -float(lam) * float(units)


# ------------------------------------------------------------------------------------------------ inputs
@dc.dataclass(eq=False)
class Candidate:
    """A rule and its out-of-sample record.  `oos` is per-period OOS performance (higher = better), dated; `folds` labels each
    period with the fold/year/sector-group it belongs to (transfer is judged across folds); `in_sample` is the mean in-sample
    performance if known (for the optimism gap)."""
    spec: RuleSpec
    oos: pd.Series
    folds: pd.Series | None = None
    in_sample: float | None = None


@dc.dataclass(frozen=True)
class ComplexityConfig:
    z0: float = 1.28                       # base one-sided bar for any improvement
    kappa: float = 0.35                    # extra t required per complexity unit
    equiv_margin: float = 0.0005           # |gain| this small (95% CI inside) counts as 'the same OOS performance'
    min_periods: int = 20
    min_folds: int = 3
    min_transfer: float = 0.6              # share of folds where the complex rule is not worse
    robust_t: float = 1.28                 # gain t that must survive dropping the best single fold
    max_gain_cv: float = 2.0               # sd of fold gains / |mean fold gain| above this = unstable
    tail_frac: float = 0.10
    tail_tolerance: float = 0.5            # tail may be worse by this many sd of the simple rule's outcome
    worst_fold_tolerance: float = 1.0      # worst fold loss (in se of the overall gain) allowed
    max_optimism_growth: float = 1.5       # complex IS-OOS gap may exceed simple's by this factor
    nw_lags: int = 4
    weights: ComplexityWeights = DEFAULT_W


DEFAULT_CCFG = ComplexityConfig()


def validate_config(cfg: ComplexityConfig) -> list[str]:
    errs = []
    if cfg.kappa < 0 or cfg.z0 < 0:
        errs.append("kappa and z0 must be non-negative (complexity may never be rewarded)")
    if cfg.min_periods < 8:
        errs.append("min_periods >= 8")
    if not 0 < cfg.min_transfer <= 1 or not 0 < cfg.tail_frac < 0.5:
        errs.append("min_transfer in (0,1] and tail_frac in (0, 0.5)")
    return errs


class Verdict(_StrEnum):
    SIMPLE = "SIMPLE"                      # complexity has not earned its place
    COMPLEX = "COMPLEX"                    # complexity has earned its place
    TIE = "TIE"                            # same OOS performance: simple preferred
    NEED_DATA = "NEED_DATA"                # cannot tell; simple is provisional


@dc.dataclass(frozen=True)
class ComplexityVerdict:
    verdict: Verdict
    simple_id: str
    complex_id: str
    delta_units: float
    gain: float                            # mean paired OOS gain of complex over simple
    gain_se: float
    gain_t: float
    t_required: float
    transfer: float                        # share of folds where complex >= simple
    fold_gains: tuple[float, ...]
    gain_cv: float
    tail_change: float
    worst_fold_gain: float
    optimism_ratio: float
    n: int
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    reasons: tuple[str, ...]
    robust_t: float = float("nan")        # weakest gain t after dropping any single fold

    @property
    def prefer(self) -> str:
        return self.complex_id if self.verdict == Verdict.COMPLEX else self.simple_id


# ------------------------------------------------------------------------------------------------ core comparison
def required_t(delta_units: float, cfg: ComplexityConfig = DEFAULT_CCFG) -> float:
    """The bar a more complex rule must clear.  Non-positive extra complexity needs only the base bar."""
    return cfg.z0 + cfg.kappa * max(delta_units, 0.0)


def _paired(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    j = pd.concat([a.rename("c"), b.rename("s")], axis=1, join="inner").dropna()
    return j.sort_index()


def _tail(x: np.ndarray, frac: float) -> float:
    k = max(1, int(math.ceil(frac * len(x))))
    return float(np.sort(x)[:k].mean())


def compare(simple: Candidate, cmplx: Candidate, cfg: ComplexityConfig = DEFAULT_CCFG) -> ComplexityVerdict:
    """Should the more complex rule be preferred over the simpler one?  Requires the SAME dates for both OOS series."""
    from engine.pattern_reliability import nw_t
    bad = validate_config(cfg)
    if bad:
        raise ComplexityError("; ".join(bad))
    du = cmplx.spec.units(cfg.weights) - simple.spec.units(cfg.weights)
    j = _paired(cmplx.oos.astype(float), simple.oos.astype(float))
    n = len(j)
    if n < cfg.min_periods:
        return ComplexityVerdict(Verdict.NEED_DATA, simple.spec.rule_id, cmplx.spec.rule_id, du, float("nan"), float("nan"),
                                 float("nan"), required_t(du, cfg), float("nan"), (), float("nan"), float("nan"),
                                 float("nan"), float("nan"), n, (), (), (f"only {n} paired OOS periods (< {cfg.min_periods})",))
    d = (j["c"] - j["s"]).values
    gain = float(d.mean())
    t = nw_t(d, lags=min(cfg.nw_lags, n - 1))
    iid = float(d.std(ddof=1) / math.sqrt(n)) or 1e-12
    se = max(abs(gain / t) if math.isfinite(t) and abs(t) > 1e-12 else iid, 0.5 * iid)
    tstat = gain / se
    treq = required_t(du, cfg)
    reasons, passed, failed = [], [], []

    # transfer + stability across folds
    fold_gains: list[float] = []
    if cmplx.folds is not None:
        fl = cmplx.folds.reindex(j.index)
        for _, idx in pd.Series(d, index=j.index).groupby(fl.values):
            if len(idx) >= 3:
                fold_gains.append(float(idx.mean()))
    transfer = float(np.mean([g >= 0 for g in fold_gains])) if fold_gains else float("nan")
    cv = (float(np.std(fold_gains, ddof=1)) / abs(np.mean(fold_gains))) if len(fold_gains) >= 2 and abs(np.mean(fold_gains)) > 1e-12 else float("nan")
    worst_fold = float(min(fold_gains)) if fold_gains else float("nan")
    rob = float("nan")
    if cmplx.folds is not None and len(fold_gains) >= 2:
        fl = cmplx.folds.reindex(j.index).values
        ts_ = []
        for f_ in pd.unique(fl):
            keep = fl != f_
            if keep.sum() >= 5 and d[keep].std(ddof=1) > 0:
                ts_.append(float(d[keep].mean() / (d[keep].std(ddof=1) / math.sqrt(keep.sum()))))
        rob = min(ts_) if ts_ else float("nan")
    # failure safety: tail of the outcome itself
    tail_change = _tail(j["c"].values, cfg.tail_frac) - _tail(j["s"].values, cfg.tail_frac)
    sd_s = float(j["s"].std(ddof=1)) or 1e-12
    # optimism: how much of the complex rule's in-sample score evaporates OOS relative to the simple rule
    opt = float("nan")
    if cmplx.in_sample is not None and simple.in_sample is not None:
        gap_c = cmplx.in_sample - float(j["c"].mean())
        gap_s = simple.in_sample - float(j["s"].mean())
        opt = gap_c / gap_s if abs(gap_s) > 1e-12 else (1.0 if abs(gap_c) < 1e-12 else float("inf"))

    equivalent = (gain - 1.96 * se > -cfg.equiv_margin) and (gain + 1.96 * se < cfg.equiv_margin)
    if equivalent:
        reasons.append(f"same OOS performance (gain {gain:.5f}, 95% CI inside +-{cfg.equiv_margin}): prefer the simpler rule")
        return ComplexityVerdict(Verdict.TIE, simple.spec.rule_id, cmplx.spec.rule_id, du, gain, se, tstat, treq, transfer,
                                 tuple(fold_gains), cv, tail_change, worst_fold, opt, n, (), (), tuple(reasons))

    if tstat >= treq:
        passed.append(f"gain t={tstat:.2f} >= required {treq:.2f} for +{du:.1f} units")
    else:
        failed.append(f"gain t={tstat:.2f} < required {treq:.2f} for +{du:.1f} units")
    if fold_gains and len(fold_gains) >= cfg.min_folds:
        (passed if transfer >= cfg.min_transfer else failed).append(f"transfer {transfer:.0%} of {len(fold_gains)} folds (need {cfg.min_transfer:.0%})")
        if math.isfinite(rob):
            (passed if rob >= cfg.robust_t else failed).append(
                f"transfer: gain t is {rob:.2f} once its best fold is dropped (need {cfg.robust_t})")
        if math.isfinite(cv):
            (passed if cv <= cfg.max_gain_cv else failed).append(f"fold-gain CV {cv:.2f} (max {cfg.max_gain_cv})")
        wf_ok = worst_fold >= -cfg.worst_fold_tolerance * se
        (passed if wf_ok else failed).append(f"worst fold gain {worst_fold:.5f} vs tolerance {-cfg.worst_fold_tolerance * se:.5f}")
    elif du > 0 and cfg.min_folds > 0:
        failed.append(f"only {len(fold_gains)} folds: transfer of the added complexity is untested")
    tail_ok = tail_change >= -cfg.tail_tolerance * sd_s
    (passed if tail_ok else failed).append(f"tail change {tail_change:.5f} ({'safe' if tail_ok else 'worse'})")
    if math.isfinite(opt):
        (passed if opt <= cfg.max_optimism_growth else failed).append(f"optimism ratio {opt:.2f} (max {cfg.max_optimism_growth})")
    if cmplx.spec.identity_smell() and not simple.spec.identity_smell():
        failed.append("complex rule contains hard-coded exceptions (identity smell)")

    if du <= 0 and gain <= 0:
        v = Verdict.SIMPLE
        reasons.append("the alternative is no more complex and no better")
    elif not failed:
        v = Verdict.COMPLEX
        reasons.append("complexity earned its place: " + "; ".join(passed))
    elif any(not f.startswith("gain t") for f in failed):
        v = Verdict.SIMPLE
        reasons.append("complexity has not earned its place: " + "; ".join(failed))
    elif tstat > 0:
        v = Verdict.NEED_DATA
        reasons.append("gain positive but below the bar, everything else clean: simple rule is provisional. " + "; ".join(failed))
    else:
        v = Verdict.SIMPLE
        reasons.append("complexity has not earned its place: " + "; ".join(failed))
    return ComplexityVerdict(v, simple.spec.rule_id, cmplx.spec.rule_id, du, gain, se, tstat, treq, transfer, tuple(fold_gains),
                             cv, tail_change, worst_fold, opt, n, tuple(passed), tuple(failed), tuple(reasons), rob)


# ------------------------------------------------------------------------------------------------ ranking a field of rules
def adjusted_score(score: float, se: float, units: float, kappa: float = DEFAULT_CCFG.kappa) -> float:
    """Score minus a complexity charge expressed in standard errors: score - kappa*units*se."""
    return float(score - kappa * units * se)


def one_se_rule(candidates: Sequence[tuple[str, float, float, float]], k: float = 1.0) -> str:
    """Breiman's one-standard-error rule.  candidates = (id, units, score, se) with higher score better.  Returns the id of the
    simplest candidate whose score is within k se of the best.  Ties on units go to the higher score."""
    if not candidates:
        raise ComplexityError("no candidates")
    best = max(candidates, key=lambda c: c[2])
    floor = best[2] - k * best[3]
    eligible = [c for c in candidates if c[2] >= floor]
    return min(eligible, key=lambda c: (c[1], -c[2]))[0]


def pareto_front(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """(id, units, score): ids not dominated by a rule that is both simpler-or-equal and better-or-equal (one strictly)."""
    out = []
    for i, (pid, u, s) in enumerate(points):
        dominated = any((u2 <= u and s2 >= s) and (u2 < u or s2 > s) for j, (_, u2, s2) in enumerate(points) if j != i)
        if not dominated:
            out.append(pid)
    return sorted(out, key=lambda pid: next(u for p, u, _ in points if p == pid))


def rank_rules(cands: Sequence[Candidate], cfg: ComplexityConfig = DEFAULT_CCFG) -> pd.DataFrame:
    """Table of every rule: units, OOS mean/se, complexity-adjusted score, Pareto flag, one-SE choice."""
    from engine.pattern_reliability import nw_t
    rows = []
    for c in cands:
        x = c.oos.astype(float).dropna().values
        if len(x) < 5:
            rows.append({"rule": c.spec.rule_id, "units": c.spec.units(cfg.weights), "n": len(x), "mean": float("nan"),
                         "se": float("nan"), "adjusted": float("nan")})
            continue
        m = float(x.mean())
        t = nw_t(x, lags=min(cfg.nw_lags, len(x) - 1))
        se = abs(m / t) if math.isfinite(t) and abs(t) > 1e-12 else float(x.std(ddof=1) / math.sqrt(len(x)))
        u = c.spec.units(cfg.weights)
        rows.append({"rule": c.spec.rule_id, "units": u, "n": len(x), "mean": m, "se": se,
                     "adjusted": adjusted_score(m, se, u, cfg.kappa)})
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["mean"])
    if len(ok):
        front = set(pareto_front([(r.rule, r.units, r.mean) for r in ok.itertuples()]))
        pick = one_se_rule([(r.rule, r.units, r.mean, r.se) for r in ok.itertuples()])
        df["pareto"] = df["rule"].isin(front)
        df["one_se_pick"] = df["rule"] == pick
    return df.sort_values(["units", "rule"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ calibration on planted nulls
def null_optimism(n_obs: int, ks: Iterable[int], n_sims: int = 200, seed: int = 0) -> dict[int, float]:
    """Measured in-sample minus out-of-sample R^2 of regressions on k pure-noise features.  Grows with k: the empirical
    reason complexity must be paid for.  Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    out = {}
    half = n_obs // 2
    for k in ks:
        gaps = []
        for _ in range(n_sims):
            X = rng.normal(size=(n_obs, k)) if k else np.zeros((n_obs, 0))
            y = rng.normal(size=n_obs)
            Xa = np.column_stack([np.ones(half), X[:half]])
            beta, *_ = np.linalg.lstsq(Xa, y[:half], rcond=None)
            r2_is = 1 - np.sum((y[:half] - Xa @ beta) ** 2) / np.sum((y[:half] - y[:half].mean()) ** 2)
            Xb = np.column_stack([np.ones(n_obs - half), X[half:]])
            r2_os = 1 - np.sum((y[half:] - Xb @ beta) ** 2) / np.sum((y[half:] - y[:half].mean()) ** 2)
            gaps.append(r2_is - r2_os)
        out[int(k)] = float(np.mean(gaps))
    return out


def calibrate_kappa(n_obs: int = 200, extra_units: Sequence[int] = (1, 2, 4), alpha: float = 0.05, sims: int = 300,
                    z0: float = DEFAULT_CCFG.z0, grid: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5),
                    seed: int = 0) -> dict:
    """Smallest kappa in `grid` for which a complex rule built from pure-noise features (fit on the first half, judged on
    the second) earns its place no more than `alpha` of the time at EVERY tested extra-unit count.  Returns the false-earn
    rate table so the choice is auditable."""
    rng = np.random.default_rng(seed)
    half = n_obs // 2
    ts = {u: [] for u in extra_units}
    for u in extra_units:
        for _ in range(sims):
            X = rng.normal(size=(n_obs, u))
            y = rng.normal(size=n_obs)
            Xa = np.column_stack([np.ones(half), X[:half]])
            beta, *_ = np.linalg.lstsq(Xa, y[:half], rcond=None)
            pred = np.column_stack([np.ones(n_obs - half), X[half:]]) @ beta
            simple_err = (y[half:] - y[:half].mean()) ** 2
            cmplx_err = (y[half:] - pred) ** 2
            d = simple_err - cmplx_err                     # positive = complex better OOS
            se = d.std(ddof=1) / math.sqrt(len(d))
            ts[u].append(d.mean() / se if se > 0 else 0.0)
    table = {}
    chosen = None
    for kap in grid:
        rates = {u: float(np.mean(np.array(ts[u]) >= z0 + kap * u)) for u in extra_units}
        table[kap] = rates
        if chosen is None and all(r <= alpha for r in rates.values()):
            chosen = kap
    return {"kappa": chosen if chosen is not None else grid[-1], "met_target": chosen is not None, "false_earn_rates": table,
            "alpha": alpha, "n_obs": n_obs, "sims": sims, "seed": seed}


def format_verdict(v: ComplexityVerdict) -> str:
    lines = [f"{v.verdict}: prefer {v.prefer}  (+{v.delta_units:.1f} units; gain {v.gain:+.5f}, t={v.gain_t:.2f} vs {v.t_required:.2f})"]
    lines += [f"  ok   {p}" for p in v.passed]
    lines += [f"  FAIL {f}" for f in v.failed]
    lines += [f"  note {r}" for r in v.reasons]
    return "\n".join(lines)


def spec_hash(spec: RuleSpec) -> str:
    return stable_hash(spec)


# ------------------------------------------------------------------------------------------------ structure from existing objects
def spec_from_pattern_text(text: str, rule_id: str | None = None) -> RuleSpec:
    """RuleSpec of an engine pattern from its `key_named` text ('A q1 & B q4 unless C q0'), parsed by
    engine.pattern_identity.Expression.  Each base term is a feature, each conjunction an interaction, each exception a condition."""
    from engine.pattern_identity import Expression
    e = Expression.parse(text)
    nb, nu = len(e.base), len(e.unless)
    return RuleSpec(rule_id or e.text, n_features=len(e.features), n_conditions=nu, n_thresholds=nb + nu,
                    n_interactions=max(nb - 1, 0), tree_depth=1 + (1 if nu else 0), n_free_params=1)


def ridge_effective_df(X: np.ndarray, lam: float) -> float:
    """Effective number of parameters of a ridge fit: trace of the hat matrix.  A penalised rule with 20 features may be as
    simple as 4 - structural counts overstate it; this is the measured version (Hastie-Tibshirani)."""
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    return float(np.sum(s ** 2 / (s ** 2 + lam)))


def optimism_adjusted(in_sample_mse: float, n: int, k: float, sigma2: float | None = None) -> float:
    """Mallows Cp correction: in-sample mean squared error + 2*k*sigma2/n is an unbiased estimate of out-of-sample error for a
    linear fit with k effective parameters.  sigma2 defaults to the in-sample error itself (conservative when k is small)."""
    if n <= 0:
        raise ComplexityError("n must be positive")
    s2 = in_sample_mse if sigma2 is None else sigma2
    return float(in_sample_mse + 2.0 * k * s2 / n)


def max_units(n_eff: float, obs_per_unit: float = 15.0, floor: float = 2.0) -> float:
    """Complexity budget the data can support: roughly one unit per `obs_per_unit` independent observations."""
    return max(floor, float(n_eff) / obs_per_unit)


def within_budget(spec: RuleSpec, n_eff: float, cfg: ComplexityConfig = DEFAULT_CCFG, obs_per_unit: float = 15.0) -> bool:
    return spec.units(cfg.weights) <= max_units(n_eff, obs_per_unit)


# ------------------------------------------------------------------------------------------------ does the verdict survive resampling?
def bootstrap_earn_probability(simple: Candidate, cmplx: Candidate, cfg: ComplexityConfig = DEFAULT_CCFG, n_boot: int = 200,
                               block: int = 4, seed: int = 0) -> dict:
    """Share of block-bootstrap resamples of the paired OOS record in which the complex rule clears its bar.  A rule that
    earns its place in 55% of resamples has not earned it; one that does so in 95% has.  Deterministic in `seed`."""
    j = _paired(cmplx.oos.astype(float), simple.oos.astype(float))
    n = len(j)
    if n < cfg.min_periods:
        return {"p_earn": float("nan"), "n": n, "note": "too few paired periods"}
    d = (j["c"] - j["s"]).values
    du = cmplx.spec.units(cfg.weights) - simple.spec.units(cfg.weights)
    bar = required_t(du, cfg)
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1)
    hits, ts = 0, []
    for _ in range(n_boot):
        idx = np.empty(n, int)
        i, pos = 0, int(rng.integers(0, n))
        while i < n:
            idx[i] = pos % n
            i += 1
            pos = int(rng.integers(0, n)) if rng.random() < p else pos + 1
        x = d[idx]
        sd = x.std(ddof=1)
        t = x.mean() / (sd / math.sqrt(n)) if sd > 0 else 0.0
        ts.append(t)
        hits += int(t >= bar)
    return {"p_earn": hits / n_boot, "n": n, "bar": bar, "t_median": float(np.median(ts)), "t_lo": float(np.quantile(ts, 0.05)),
            "t_hi": float(np.quantile(ts, 0.95))}


def verdict_by_era(simple: Candidate, cmplx: Candidate, cfg: ComplexityConfig = DEFAULT_CCFG, n_eras: int = 3) -> list[dict]:
    """Run the comparison inside each of `n_eras` contiguous eras.  Complexity that earns its place in one era and loses in
    the others is regime luck (or a regime condition worth learning) - the caller must know which."""
    j = _paired(cmplx.oos.astype(float), simple.oos.astype(float))
    out = []
    if len(j) < n_eras * cfg.min_periods:
        return out
    for k, ix in enumerate(np.array_split(np.arange(len(j)), n_eras)):
        sub = j.iloc[ix]
        s = Candidate(simple.spec, sub["s"], None, simple.in_sample)
        c = Candidate(cmplx.spec, sub["c"], None, cmplx.in_sample)
        v = compare(s, c, dc.replace(cfg, min_folds=0))
        out.append({"era": k, "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()), "verdict": str(v.verdict),
                    "gain": v.gain, "t": v.gain_t})
    return out


# ------------------------------------------------------------------------------------------------ growing a model honestly
def sequential_growth(cands: Sequence[Candidate], cfg: ComplexityConfig = DEFAULT_CCFG) -> dict:
    """Forward 'earn your place' selection.  Sort by complexity; start with the simplest; move to the next only if it beats the
    CURRENT choice under `compare`.  Returns the chosen rule and the path of decisions (every rejected step is recorded)."""
    if not cands:
        raise ComplexityError("no candidates")
    order = sorted(cands, key=lambda c: (c.spec.units(cfg.weights), c.spec.rule_id))
    cur = order[0]
    path = []
    for nxt in order[1:]:
        v = compare(cur, nxt, cfg)
        path.append({"from": cur.spec.rule_id, "to": nxt.spec.rule_id, "verdict": str(v.verdict), "gain": v.gain, "t": v.gain_t,
                     "required": v.t_required, "delta_units": v.delta_units})
        if v.verdict == Verdict.COMPLEX:
            cur = nxt
    return {"chosen": cur.spec.rule_id, "path": path, "n_upgrades": sum(1 for p in path if p["verdict"] == "COMPLEX")}


def pairwise_verdicts(cands: Sequence[Candidate], cfg: ComplexityConfig = DEFAULT_CCFG) -> pd.DataFrame:
    """Matrix of verdicts for every (simpler, more complex) pair - which additions earn their place against which baselines."""
    rows = []
    order = sorted(cands, key=lambda c: (c.spec.units(cfg.weights), c.spec.rule_id))
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if b.spec.units(cfg.weights) <= a.spec.units(cfg.weights):
                continue
            v = compare(a, b, cfg)
            rows.append({"simple": a.spec.rule_id, "complex": b.spec.rule_id, "verdict": str(v.verdict), "gain": v.gain,
                         "t": v.gain_t, "required": v.t_required, "transfer": v.transfer})
    return pd.DataFrame(rows)


class ComparisonLedger:
    """Append-only record of complexity decisions, so a rule that was once preferred for its simplicity (or once allowed its
    complexity) can be audited later against what the data showed at the time."""

    def __init__(self):
        self._rows: list[dict] = []

    def record(self, v: ComplexityVerdict, as_of: str) -> int:
        if self._rows and as_of < self._rows[-1]["as_of"]:
            raise ComplexityError("ledger is append-only in time")
        self._rows.append({"as_of": str(as_of), "simple": v.simple_id, "complex": v.complex_id, "verdict": str(v.verdict),
                           "gain": v.gain, "t": v.gain_t, "required": v.t_required, "n": v.n,
                           "id": stable_hash([as_of, v.simple_id, v.complex_id, str(v.verdict)], 12)})
        return len(self._rows)

    def history(self, rule_id: str | None = None) -> list[dict]:
        return [r for r in self._rows if rule_id in (None, r["simple"], r["complex"])]

    def reversals(self) -> list[dict]:
        """Pairs whose verdict changed between comparisons: complexity that was earned and later lost (or the reverse)."""
        last: dict[tuple, dict] = {}
        out = []
        for r in self._rows:
            k = (r["simple"], r["complex"])
            if k in last and last[k]["verdict"] != r["verdict"]:
                out.append({"pair": k, "from": last[k]["verdict"], "to": r["verdict"], "at": r["as_of"]})
            last[k] = r
        return out


# ------------------------------------------------------------------------------------------------ candidates and audits
def unit_breakdown(spec: RuleSpec, w: ComplexityWeights = DEFAULT_W) -> dict[str, float]:
    """Where the complexity units of a rule come from."""
    return {"features": w.feature * spec.n_features, "conditions": w.condition * spec.n_conditions,
            "thresholds": w.threshold * spec.n_thresholds, "interactions": w.interaction * spec.n_interactions,
            "depth": w.depth * (spec.tree_depth - 1), "free_params": w.free_param * spec.n_free_params,
            "exceptions": w.exception * spec.n_exceptions}


def simpler_variants(spec: RuleSpec) -> list[RuleSpec]:
    """One-step simplifications of a rule: drop a feature, a condition (with its threshold), an interaction, an exception, a
    level of depth.  These are the natural baselines a complex rule must beat - generated, not hand-picked."""
    out = []
    if spec.n_features > 1:
        out.append(dc.replace(spec, rule_id=f"{spec.rule_id}-feature", n_features=spec.n_features - 1,
                              n_interactions=max(spec.n_interactions - 1, 0)))
    if spec.n_conditions > 0:
        out.append(dc.replace(spec, rule_id=f"{spec.rule_id}-condition", n_conditions=spec.n_conditions - 1,
                              n_thresholds=max(spec.n_thresholds - 1, 0), tree_depth=max(spec.tree_depth - (spec.n_conditions == 1), 1)))
    if spec.n_interactions > 0:
        out.append(dc.replace(spec, rule_id=f"{spec.rule_id}-interaction", n_interactions=spec.n_interactions - 1))
    if spec.n_exceptions > 0:
        out.append(dc.replace(spec, rule_id=f"{spec.rule_id}-exception", n_exceptions=spec.n_exceptions - 1))
    return out


def audit_specs(specs: Sequence[RuleSpec], w: ComplexityWeights = DEFAULT_W) -> dict:
    """Portfolio view of a set of rules: unit distribution and how many carry an identity smell (hard-coded exceptions)."""
    if not specs:
        return {"n": 0}
    u = np.array([s.units(w) for s in specs])
    return {"n": len(specs), "mean_units": float(u.mean()), "median_units": float(np.median(u)), "max_units": float(u.max()),
            "p90_units": float(np.quantile(u, 0.9)), "identity_smells": [s.rule_id for s in specs if s.identity_smell()],
            "share_over_15": float(np.mean(u > 15))}


def equivalence_pvalue(simple: Candidate, cmplx: Candidate, margin: float, nw_lags: int = 4) -> float:
    """Two one-sided tests (TOST) that the paired OOS difference lies inside +-margin.  Small p = the two rules ARE the same
    out of sample to within `margin`, so the simpler must be kept.  1.0 when the data cannot show equivalence."""
    from engine.pattern_reliability import nw_t
    j = _paired(cmplx.oos.astype(float), simple.oos.astype(float))
    if len(j) < 8 or margin <= 0:
        return 1.0
    d = (j["c"] - j["s"]).values
    m = float(d.mean())
    t = nw_t(d, lags=min(nw_lags, len(d) - 1))
    iid = float(d.std(ddof=1) / math.sqrt(len(d))) or 1e-12
    se = max(abs(m / t) if math.isfinite(t) and abs(t) > 1e-12 else iid, 0.5 * iid)
    p_lo = float(sps.norm.sf((m + margin) / se))          # H0: diff <= -margin
    p_hi = float(sps.norm.cdf((m - margin) / se))         # H0: diff >= +margin
    return max(p_lo, p_hi)


def complexity_report(cands: Sequence[Candidate], cfg: ComplexityConfig = DEFAULT_CCFG, seed: int = 0) -> str:
    """Text report: every rule's units and OOS score, the Pareto/one-SE picks, the earn-your-place path, and how robust each
    upgrade is to resampling.  Everything shown is recomputable from the candidates."""
    if not cands:
        return "(no candidates)"
    tab = rank_rules(cands, cfg)
    grow = sequential_growth(cands, cfg)
    by_id = {c.spec.rule_id: c for c in cands}
    lines = ["RULES (units, OOS mean, se, adjusted):"]
    for r in tab.itertuples():
        flags = ("P" if getattr(r, "pareto", False) else "-") + ("1" if getattr(r, "one_se_pick", False) else "-")
        lines.append(f"  {r.rule:<28}{r.units:>6.1f}{r.mean:>11.5f}{r.se:>10.5f}{r.adjusted:>11.5f}  [{flags}]")
    lines.append(f"EARN-YOUR-PLACE CHOICE: {grow['chosen']}  ({grow['n_upgrades']} upgrade(s) accepted)")
    for step in grow["path"]:
        p = bootstrap_earn_probability(by_id[step["from"]], by_id[step["to"]], cfg, n_boot=100, seed=seed)
        lines.append(f"  {step['from']} -> {step['to']}: {step['verdict']}  gain {step['gain']:+.5f} t={step['t']:.2f} "
                     f"(bar {step['required']:.2f}); earns in {p['p_earn']:.0%} of resamples")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ does the whole procedure work?
def simulate_ladder(true_level: int, n: int = 240, seed: int = 0, levels: int = 4, gain_per_level: float = 0.006,
                    noise: float = 0.01) -> list[Candidate]:
    """A ladder of rules r0..r{levels-1}, each more complex than the last.  Rule k earns +gain_per_level over rule k-1 for k <=
    true_level; beyond that extra complexity adds only noise.  The correct choice is therefore r{true_level}."""
    if not 0 <= true_level < levels:
        raise ComplexityError("true_level must be in [0, levels)")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-04", periods=n, freq="W-MON")
    base = pd.Series(rng.normal(0.004, noise, n), index=idx)
    folds = pd.Series(idx.year, index=idx)
    out = []
    for k in range(levels):
        g = gain_per_level * min(k, true_level)
        spec = RuleSpec(f"r{k}", n_features=1 + k, n_conditions=max(k - 1, 0), n_thresholds=max(k - 1, 0))
        out.append(Candidate(spec, base + g + rng.normal(0, noise * 0.3, n) * (1 if k else 0), folds))
    return out


def selection_accuracy(true_level: int, n: int = 240, sims: int = 40, cfg: ComplexityConfig = DEFAULT_CCFG, **kw) -> dict:
    """How often does sequential_growth pick the right rung, too low (missed real structure) or too high (bought noise)?  The
    honest scorecard of the earn-your-place procedure on a planted ladder."""
    right = low = high = 0
    for s in range(sims):
        lad = simulate_ladder(true_level, n=n, seed=s, **kw)
        pick = int(sequential_growth(lad, cfg)["chosen"][1:])
        right += pick == true_level
        low += pick < true_level
        high += pick > true_level
    return {"true_level": true_level, "n": n, "sims": sims, "right": right / sims, "too_simple": low / sims, "too_complex": high / sims}


def cross_validated_error(X: np.ndarray, y: np.ndarray, ks: Sequence[int], folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold squared error of linear fits that use the first k columns of X, for each k.  The measured price of adding
    parameters: in-sample error always falls with k, out-of-fold error turns up when the extra columns are noise.  Returns k,
    mean error and standard error across folds; feed to one_se_rule to choose the simplest adequate k."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    if X.shape[0] != n or folds < 2 or n < 2 * folds:
        raise ComplexityError("need X and y of equal length with at least 2 observations per fold")
    order = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(order, folds)
    rows = []
    for k in ks:
        errs = []
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            A = np.column_stack([np.ones(len(tr)), X[tr, :k]])
            beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred = np.column_stack([np.ones(len(te)), X[te, :k]]) @ beta
            errs.append(float(np.mean((y[te] - pred) ** 2)))
        rows.append({"k": int(k), "cv_error": float(np.mean(errs)), "se": float(np.std(errs, ddof=1) / math.sqrt(folds))})
    return pd.DataFrame(rows)
