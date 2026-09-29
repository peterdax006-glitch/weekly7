"""Transfer ratio, over-specialisation, memorisation and identity gaps, transfer stability (contract C62 sections 26-27, 47;
checklist E10-E13, L08). Bible: the learning delta must be judged on what the learner has NOT seen.

STATUS: IMPLEMENTED - NOT VALIDATED.

transfer_ratio = cross_context_gain / same_context_gain, but a ratio is only a number when its denominator is a real positive
gain. Every other case (no same-context gain, a negative one, a gain smaller than noise) gets an explicit status instead of a
divide-by-zero, an infinity, or a sign flip that would turn "it got worse in training AND worse elsewhere" into a "good" ratio.

All statistics are cluster bootstraps (units from one week / one year / one ticker are not independent), seeded, and return
honest infinite intervals when there is too little data. Nothing here reads a clock or a file."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .core import ValidationLabel, _StrEnum, clip01

EPS_GAIN = 1e-4            # a per-unit gain under one basis point is indistinguishable from zero
RATIO_CAP = 3.0            # a transfer ratio above +-3 is capped: it means the denominator was tiny, not that transfer is superb
RATIO_FLOOR = 0.30         # cross gain under 30% of same-context gain marks a learner as over-specialised
MAX_DRAWS = 4_000_000      # cap on n_boot * n_clusters so a big panel cannot exhaust RAM


class RatioStatus(_StrEnum):
    OK = "OK"                              # same-context gain is real and positive: ratio is meaningful (may be < 0)
    TRANSFER_ONLY = "TRANSFER_ONLY"        # no same-context gain but a real cross-context gain: ratio undefined, not infinite
    NO_GAIN = "NO_GAIN"                    # neither context gained more than noise
    HARMFUL = "HARMFUL"                    # no same-context gain and cross-context damage
    INSUFFICIENT = "INSUFFICIENT"          # NaN or too few units on a side


class TransferVerdictLabel(_StrEnum):
    GENERALISES = "GENERALISES"
    OVER_SPECIALISED = "OVER_SPECIALISED"
    IDENTITY_DEPENDENT = "IDENTITY_DEPENDENT"
    TRANSFER_ONLY = "TRANSFER_ONLY"
    UNSTABLE = "UNSTABLE"
    NO_LEARNING = "NO_LEARNING"
    HARMFUL = "HARMFUL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------------------------------------------
# cluster bootstrap (shared by transfer.py and portfolio_value.py)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootMean:
    mean: float
    lo: float
    hi: float
    n: int                 # units
    n_clusters: int

    @property
    def excludes_zero_above(self) -> bool:
        return self.n_clusters >= 2 and self.lo > 0

    @property
    def excludes_zero_below(self) -> bool:
        return self.n_clusters >= 2 and self.hi < 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _clean(values, clusters):
    v = np.asarray(values, float).ravel()
    ok = np.isfinite(v)
    if clusters is None:
        cl = np.arange(len(v))
    else:
        cl = np.asarray(clusters).ravel()
        if len(cl) != len(v):
            raise ValueError(f"clusters ({len(cl)}) and values ({len(v)}) differ in length")
    return v[ok], cl[ok]


def bootstrap_draws(values, clusters=None, n_boot=1000, rng=None):
    """Bootstrap distribution of the mean, resampling whole clusters. Returns (draws, point_mean, n_units, n_clusters);
    draws is empty when there are fewer than two clusters (no honest resampling exists)."""
    v, cl = _clean(values, clusters)
    n = len(v)
    if n == 0:
        return np.array([]), float("nan"), 0, 0
    _, inv = np.unique(cl.astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return np.array([]), float(v.mean()), n, g
    rng = rng if rng is not None else np.random.default_rng(0)
    sums, cnt = np.bincount(inv, weights=v, minlength=g), np.bincount(inv, minlength=g).astype(float)
    nb = int(min(n_boot, max(200, MAX_DRAWS // g)))
    idx = rng.integers(0, g, size=(nb, g))
    return sums[idx].sum(axis=1) / cnt[idx].sum(axis=1), float(v.mean()), n, g


def cluster_bootstrap_mean(values, clusters=None, n_boot=1000, level=0.95, seed=0) -> BootMean:
    """Bootstrap mean CI. Independent units (clusters=None) go through learning_delta.boot_ci (the audited implementation, reused
    not copied); with clusters the whole cluster is resampled, which boot_ci cannot do."""
    if clusters is None:
        from .. import learning_delta as LD
        m, lo, hi, n = LD.boot_ci(values, n_boot, level, seed)
        return BootMean(m, lo, hi, n, n)
    draws, m, n, g = bootstrap_draws(values, clusters, n_boot, np.random.default_rng(seed))
    if n == 0:
        return BootMean(float("nan"), float("nan"), float("nan"), 0, 0)
    if len(draws) == 0:
        return BootMean(m, float("-inf"), float("inf"), n, g)
    q = (1 - level) / 2
    return BootMean(m, float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q)), n, g)


def cluster_bootstrap_diff(a, b, clusters_a=None, clusters_b=None, n_boot=1000, level=0.95, seed=0) -> BootMean:
    """mean(a) - mean(b) for two independent unit sets (unpaired), each resampled by its own clusters."""
    rng = np.random.default_rng(seed)
    da, ma, na, ga = bootstrap_draws(a, clusters_a, n_boot, rng)
    db, mb, nb, gb = bootstrap_draws(b, clusters_b, n_boot, rng)
    if na == 0 or nb == 0:
        return BootMean(float("nan"), float("nan"), float("nan"), na + nb, ga + gb)
    if len(da) == 0 or len(db) == 0:
        return BootMean(ma - mb, float("-inf"), float("inf"), na + nb, ga + gb)
    k = min(len(da), len(db))
    d = da[:k] - db[:k]
    q = (1 - level) / 2
    return BootMean(ma - mb, float(np.quantile(d, q)), float(np.quantile(d, 1 - q)), na + nb, ga + gb)


def paired_bootstrap_diff(a, b, clusters=None, n_boot=1000, level=0.95, seed=0) -> BootMean:
    """mean(a - b) for unit-aligned arrays (a unit that is NaN on either side is dropped)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) != len(b):
        raise ValueError("paired arrays differ in length")
    ok = np.isfinite(a) & np.isfinite(b)
    cl = None if clusters is None else np.asarray(clusters).ravel()[ok]
    return cluster_bootstrap_mean((a - b)[ok], cl, n_boot, level, seed)


# ---------------------------------------------------------------------------------------------------------------
# the ratio
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RatioResult:
    value: float | None        # None when the ratio is not defined; NEVER inf, NEVER a silently-flipped sign
    capped: float | None       # value clipped to +-RATIO_CAP
    status: RatioStatus
    cross_gain: float
    same_gain: float
    reason: str

    @property
    def defined(self) -> bool:
        return self.value is not None


def transfer_ratio(cross_gain: float, same_gain: float, *, eps: float = EPS_GAIN, cap: float = RATIO_CAP,
                   n_cross: int | None = None, n_same: int | None = None, min_n: int = 1) -> RatioResult:
    """cross_context_gain / same_context_gain with every degenerate denominator named.
    same > eps            -> ratio (negative when the cross-context gain is negative: the learner hurt elsewhere)
    same <= eps, cross > eps  -> TRANSFER_ONLY (learning showed up elsewhere but not where it was trained)
    same <= eps, cross < -eps -> HARMFUL
    otherwise                 -> NO_GAIN
    NaN inputs or fewer than `min_n` units on either side -> INSUFFICIENT."""
    c, s = float(cross_gain), float(same_gain)
    if math.isnan(c) or math.isnan(s):
        return RatioResult(None, None, RatioStatus.INSUFFICIENT, c, s, "a gain is NaN (never measured)")
    if (n_cross is not None and n_cross < min_n) or (n_same is not None and n_same < min_n):
        return RatioResult(None, None, RatioStatus.INSUFFICIENT, c, s, f"fewer than {min_n} units (cross={n_cross}, same={n_same})")
    if s > eps:
        v = c / s
        return RatioResult(v, float(np.clip(v, -cap, cap)), RatioStatus.OK, c, s, f"cross {c:+.4g} / same {s:+.4g}")
    if c > eps:
        return RatioResult(None, None, RatioStatus.TRANSFER_ONLY, c, s, f"same-context gain {s:+.4g} is not positive but cross-context gain is {c:+.4g}")
    if c < -eps:
        return RatioResult(None, None, RatioStatus.HARMFUL, c, s, f"no same-context gain ({s:+.4g}) and cross-context damage ({c:+.4g})")
    return RatioResult(None, None, RatioStatus.NO_GAIN, c, s, f"both gains within +-{eps:g} of zero")


@dataclass(frozen=True)
class RatioCI:
    point: RatioResult
    lo: float | None
    hi: float | None
    denominator_risk: float    # share of bootstrap draws whose same-context gain fell to <= eps
    bounded: bool              # False: the denominator's interval reaches zero, so no finite ratio interval exists
    n_draws: int


def ratio_ci(cross_values, same_values, cross_clusters=None, same_clusters=None, *, eps: float = EPS_GAIN, n_boot: int = 1500,
             level: float = 0.95, seed: int = 0, max_denominator_risk: float = 0.05) -> RatioCI:
    """Bootstrap interval for the ratio. A ratio of two noisy means explodes when the denominator's distribution reaches zero, so
    the share of draws with a non-positive denominator is reported, and when it exceeds `max_denominator_risk` the interval is
    declared unbounded instead of quoting the (misleading) quantiles of the surviving draws."""
    rng = np.random.default_rng(seed)
    dc, mc, nc, _ = bootstrap_draws(cross_values, cross_clusters, n_boot, rng)
    ds, ms, ns, _ = bootstrap_draws(same_values, same_clusters, n_boot, rng)
    point = transfer_ratio(mc, ms, eps=eps, n_cross=nc, n_same=ns)
    if len(dc) == 0 or len(ds) == 0:
        return RatioCI(point, None, None, 1.0 if point.status != RatioStatus.OK else float("nan"), False, 0)
    k = min(len(dc), len(ds))
    dc, ds = dc[:k], ds[:k]
    good = ds > eps
    risk = float(1 - good.mean())
    if risk > max_denominator_risk or good.sum() < 20:
        return RatioCI(point, None, None, risk, False, int(good.sum()))
    r = np.clip(dc[good] / ds[good], -RATIO_CAP, RATIO_CAP)
    q = (1 - level) / 2
    return RatioCI(point, float(np.quantile(r, q)), float(np.quantile(r, 1 - q)), risk, True, int(good.sum()))


# ---------------------------------------------------------------------------------------------------------------
# over-specialisation
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Specialisation:
    over_specialised: bool
    suspected: bool            # point estimate says so but the intervals do not yet settle it
    severity: float            # 0..1: how much of the same-context gain fails to appear elsewhere
    reasons: tuple

    @property
    def flag(self) -> str:
        return "OVER_SPECIALISED" if self.over_specialised else ("SUSPECTED" if self.suspected else "NONE")


def specialisation(same: BootMean, cross: BootMean, ratio: RatioResult, *, floor: float = RATIO_FLOOR, eps: float = EPS_GAIN) -> Specialisation:
    """A learner that dramatically improves the training context but fails elsewhere (section 27).
    over_specialised: same-context gain is significantly positive AND the cross-context gain's upper bound is under `floor`
    times the same-context gain's lower bound, or the cross gain is significantly negative.
    suspected: same-context gain positive and the point ratio is under `floor` (or the ratio is HARMFUL/NO_GAIN) but the
    intervals overlap."""
    reasons = []
    if same.n == 0 or math.isnan(same.mean):
        return Specialisation(False, False, 0.0, ("no same-context evidence",))
    same_real = same.mean > eps and same.lo > 0
    if not same_real:
        return Specialisation(False, False, 0.0, ("no significant same-context gain to be specialised on",))
    sev = 0.0
    if not math.isnan(cross.mean):
        sev = clip01(1.0 - max(cross.mean, 0.0) / same.mean)
    hard = False
    if cross.n > 0 and cross.n_clusters >= 2 and cross.hi < floor * same.lo:
        hard = True
        reasons.append(f"cross-context gain upper bound {cross.hi:+.4g} < {floor:.0%} of same-context lower bound {same.lo:+.4g}")
    if cross.n > 0 and cross.n_clusters >= 2 and cross.hi < 0:
        hard = True
        reasons.append(f"cross-context gain is significantly negative ({cross.mean:+.4g})")
    soft = False
    if not hard:
        if ratio.status == RatioStatus.OK and ratio.value is not None and ratio.value < floor:
            soft = True
            reasons.append(f"point ratio {ratio.value:.2f} < {floor:.2f} but intervals overlap")
        elif ratio.status in (RatioStatus.NO_GAIN, RatioStatus.HARMFUL):
            soft = True
            reasons.append(f"same-context gain positive, cross-context {ratio.status.value}")
        elif cross.n == 0:
            soft = True
            reasons.append("no cross-context evidence at all (untested is not transferred)")
    return Specialisation(hard, soft, sev, tuple(reasons))


# ---------------------------------------------------------------------------------------------------------------
# memorisation gap and identity gap
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GapResult:
    name: str
    gap: float                 # a - b (a is the condition that could exploit identity/answers)
    lo: float
    hi: float
    n_a: int
    n_b: int
    relative: float            # gap / |a|, NaN when a is ~0
    flagged: bool              # gap significantly above `tol`

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _gap(name, a_vals, b_vals, a_cl, b_cl, paired, tol, n_boot, seed):
    if paired:
        r = paired_bootstrap_diff(a_vals, b_vals, a_cl, n_boot, seed=seed)
        na = nb = r.n
        ma = float(np.nanmean(a_vals)) if len(a_vals) else float("nan")
    else:
        r = cluster_bootstrap_diff(a_vals, b_vals, a_cl, b_cl, n_boot, seed=seed)
        na, nb = int(np.isfinite(np.asarray(a_vals, float)).sum()), int(np.isfinite(np.asarray(b_vals, float)).sum())
        ma = float(np.nanmean(a_vals)) if na else float("nan")
    rel = r.mean / abs(ma) if np.isfinite(ma) and abs(ma) > EPS_GAIN and np.isfinite(r.mean) else float("nan")
    return GapResult(name, r.mean, r.lo, r.hi, na, nb, rel, bool(r.n_clusters >= 2 and r.lo > tol))


def memorization_gap(seen_gains, fresh_gains, seen_clusters=None, fresh_clusters=None, *, tol: float = 0.0, n_boot: int = 1000, seed: int = 0) -> GapResult:
    """Gain on units whose answers the learner could have stored (the situations it trained on, replayed) minus gain on
    equally-structured FRESH units (disguised or later). A learner that learned a rule scores about the same on both; a lookup
    table scores far higher on the first. Unpaired: the two sets need not be the same size."""
    return _gap("memorization_gap", seen_gains, fresh_gains, seen_clusters, fresh_clusters, False, tol, n_boot, seed)


def identity_gap(gains_identity_kept, gains_identity_scrambled, clusters=None, *, tol: float = 0.0, n_boot: int = 1000, seed: int = 0) -> GapResult:
    """Paired gain with real identities minus the gain on the SAME units after identities were scrambled/disguised. Positive and
    significant means the learner's benefit depended on knowing WHICH stock/date it was, not on what the stock was doing."""
    return _gap("identity_gap", gains_identity_kept, gains_identity_scrambled, clusters, clusters, True, tol, n_boot, seed)


@dataclass(frozen=True)
class Concentration:
    n_keys: int
    hhi: float                 # Herfindahl index of positive gain over keys, 1/n_keys .. 1
    effective_n: float         # 1 / hhi: how many keys the gain is really spread over
    top_share: float           # share of all positive gain from the single largest key
    top5_share: float
    gini: float

    @property
    def concentrated(self) -> bool:
        return self.n_keys >= 5 and (self.top_share > 0.5 or self.effective_n < max(2.0, 0.1 * self.n_keys))


def gain_concentration(gains, keys) -> Concentration:
    """How concentrated the learner's gain is over identities (tickers or ticker-dates). A memoriser makes its money on the few
    identities it stored; a rule spreads it. Computed on the positive part of the per-key total gain."""
    g = np.asarray(gains, float).ravel()
    k = np.asarray(keys).ravel().astype(str)
    if len(g) != len(k):
        raise ValueError("gains and keys differ in length")
    ok = np.isfinite(g)
    g, k = g[ok], k[ok]
    if len(g) == 0:
        return Concentration(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    uniq, inv = np.unique(k, return_inverse=True)
    tot = np.bincount(inv, weights=g)
    pos = np.clip(tot, 0.0, None)
    s = pos.sum()
    if s <= 0:
        return Concentration(len(uniq), float("nan"), float("nan"), 0.0, 0.0, 0.0)
    p = np.sort(pos / s)[::-1]
    hhi = float((p ** 2).sum())
    asc = np.sort(pos / s)
    n = len(asc)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(asc) / n) if n > 1 else 0.0
    return Concentration(n, hhi, 1.0 / hhi, float(p[0]), float(p[:5].sum()), gini)


# ---------------------------------------------------------------------------------------------------------------
# transfer stability
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransferStability:
    n_groups: int
    mean: float
    sd: float
    minimum: float
    maximum: float
    share_positive: float
    q20: float                 # 20th percentile of group gains: what a bad-but-not-worst group looks like
    score: float               # 0..1; 0 unless the mean is positive

    @property
    def stable(self) -> bool:
        return self.n_groups >= 3 and self.score >= 0.5 and self.q20 > -abs(self.mean)


def transfer_stability(group_gains: Sequence[float]) -> TransferStability:
    """Stability of transfer across held-out groups (years, regimes, sectors...): a learner that transfers on average but
    fails badly on some groups is not stable. score = share_positive * mean / (mean + sd) for a positive mean, else 0."""
    g = np.asarray([x for x in group_gains if np.isfinite(x)], float)
    if len(g) == 0:
        return TransferStability(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0.0)
    m = float(g.mean())
    sd = float(g.std(ddof=1)) if len(g) > 1 else 0.0
    sp = float((g > 0).mean())
    score = clip01(sp * m / (m + sd)) if m > 0 and (m + sd) > 0 else 0.0
    return TransferStability(len(g), m, sd, float(g.min()), float(g.max()), sp, float(np.quantile(g, 0.2)), score)


@dataclass(frozen=True)
class RollingStability:
    n_windows: int
    share_positive_windows: float
    longest_negative_run: int
    worst_window_mean: float
    sign_flips: int


def rolling_stability(gains_in_time_order: Sequence[float], window: int = 8) -> RollingStability:
    """Does the gain persist through time? Rolling-window means of a time-ordered gain series: how many windows are positive,
    how long the longest run of negative windows is, and how often the sign flips."""
    g = np.asarray([x for x in gains_in_time_order if np.isfinite(x)], float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(g) < window:
        return RollingStability(0, float("nan"), 0, float("nan"), 0)
    rm = np.convolve(g, np.ones(window) / window, mode="valid")
    neg = rm <= 0
    run = best = 0
    for x in neg:
        run = run + 1 if x else 0
        best = max(best, run)
    sgn = np.sign(rm)
    flips = int((np.diff(sgn[sgn != 0]) != 0).sum()) if (sgn != 0).sum() > 1 else 0
    return RollingStability(len(rm), float((rm > 0).mean()), best, float(rm.min()), flips)


# ---------------------------------------------------------------------------------------------------------------
# combined judgement
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransferVerdict:
    label: TransferVerdictLabel
    validation: ValidationLabel
    reasons: tuple
    ratio: RatioResult | None = None

    def summary(self) -> str:
        return f"{self.label.value} [{self.validation.value}]: " + "; ".join(self.reasons)


def classify_transfer(same: BootMean, cross: BootMean, ratio: RatioResult, spec: Specialisation, *, mem: GapResult | None = None,
                      ident: GapResult | None = None, stability: TransferStability | None = None, min_clusters: int = 3) -> TransferVerdict:
    """One honest label from the evidence. Precedence: too little evidence, identity reliance, over-specialisation, harm, no
    gain, instability, then GENERALISES. Nothing here ever returns VALIDATED: the validation label is at best NOT_VALIDATED,
    because validation needs the controls of the scorecard (section 47) and a blind window."""
    nv = ValidationLabel.NOT_VALIDATED
    if cross.n_clusters < min_clusters and same.n_clusters < min_clusters:
        return TransferVerdict(TransferVerdictLabel.INSUFFICIENT_EVIDENCE, ValidationLabel.INSUFFICIENT_EVIDENCE,
                               (f"fewer than {min_clusters} independent clusters on both sides",), ratio)
    if ident is not None and ident.flagged:
        return TransferVerdict(TransferVerdictLabel.IDENTITY_DEPENDENT, ValidationLabel.FAILED_VALIDATION,
                               (f"identity gap {ident.gap:+.4g} (CI lo {ident.lo:+.3g}) > 0: the gain needs real identities",), ratio)
    if mem is not None and mem.flagged and spec.over_specialised:
        return TransferVerdict(TransferVerdictLabel.OVER_SPECIALISED, ValidationLabel.FAILED_VALIDATION,
                               (f"memorisation gap {mem.gap:+.4g} and", *spec.reasons), ratio)
    if spec.over_specialised:
        return TransferVerdict(TransferVerdictLabel.OVER_SPECIALISED, ValidationLabel.FAILED_VALIDATION, spec.reasons, ratio)
    if ratio.status == RatioStatus.HARMFUL or (cross.n_clusters >= min_clusters and cross.excludes_zero_below):
        return TransferVerdict(TransferVerdictLabel.HARMFUL, ValidationLabel.FAILED_VALIDATION, (ratio.reason,), ratio)
    if ratio.status == RatioStatus.INSUFFICIENT:
        return TransferVerdict(TransferVerdictLabel.INSUFFICIENT_EVIDENCE, ValidationLabel.INSUFFICIENT_EVIDENCE, (ratio.reason,), ratio)
    if ratio.status == RatioStatus.NO_GAIN or not (cross.excludes_zero_above or same.excludes_zero_above):
        return TransferVerdict(TransferVerdictLabel.NO_LEARNING, nv, ("neither context shows a gain beyond noise",), ratio)
    if ratio.status == RatioStatus.TRANSFER_ONLY:
        return TransferVerdict(TransferVerdictLabel.TRANSFER_ONLY, nv, (ratio.reason,), ratio)
    if stability is not None and stability.n_groups >= 3 and not stability.stable:
        return TransferVerdict(TransferVerdictLabel.UNSTABLE, nv,
                               (f"gain differs across {stability.n_groups} held-out groups: {stability.share_positive:.0%} positive, "
                                f"worst {stability.minimum:+.4g}",), ratio)
    if cross.excludes_zero_above:
        return TransferVerdict(TransferVerdictLabel.GENERALISES, nv,
                               (f"cross-context gain {cross.mean:+.4g} (CI {cross.lo:+.3g}..{cross.hi:+.3g}); ratio {ratio.reason}",), ratio)
    return TransferVerdict(TransferVerdictLabel.INSUFFICIENT_EVIDENCE, ValidationLabel.INSUFFICIENT_EVIDENCE,
                           ("same-context gain is real but cross-context gain is not distinguishable from zero",), ratio)


# ---------------------------------------------------------------------------------------------------------------
# significance, power, and many-item corrections
# ---------------------------------------------------------------------------------------------------------------
def cluster_signflip_p(values, clusters=None, n_perm: int = 4000, seed: int = 0) -> float:
    """Two-sided p-value that the mean gain is zero, flipping the sign of whole clusters (a cluster's units share one draw of
    luck, so units are not flipped one by one). Exact enumeration up to 12 clusters, seeded Monte Carlo above. NaN with no data;
    1.0 when every cluster mean is zero. Reuses learning_delta.signflip_p on the cluster means."""
    from .. import learning_delta as LD
    v, cl = _clean(values, clusters)
    if len(v) == 0:
        return float("nan")
    _, inv = np.unique(cl.astype(str), return_inverse=True)
    sums, cnt = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    # weight each cluster mean by its size so that a big cluster counts for what it is worth
    return LD.signflip_p(sums / cnt * (cnt / cnt.mean()), n_perm=n_perm, seed=seed)


@dataclass(frozen=True)
class Power:
    n_clusters: int
    between_cluster_sd: float
    se: float                  # standard error of the pooled mean gain
    mde_80: float              # smallest true mean gain detected with 80% power at one-sided 5%
    observed: float
    underpowered: bool         # the observed cross-context gain is below the MDE: 'no significant transfer' is not evidence of none

    def statement(self) -> str:
        return (f"{self.n_clusters} clusters, se {self.se:.4g}: gains below {self.mde_80:.4g} could not have been detected"
                + ("; the observed gain is inside that blind zone, so a non-significant result says little" if self.underpowered else ""))


def power_analysis(values, clusters=None, observed: float | None = None) -> Power:
    """Minimum detectable gain of the cross-context test, from the between-cluster spread of the data itself: (1.645 + 0.842) x se,
    the normal approximation for 80% power at a one-sided 5% level. Prevents 'no transfer' being read from a test that had no
    ability to see transfer. NaNs with fewer than 3 clusters."""
    v, cl = _clean(values, clusters)
    if len(v) == 0:
        return Power(0, float("nan"), float("nan"), float("nan"), float("nan"), True)
    _, inv = np.unique(cl.astype(str), return_inverse=True)
    g = int(inv.max()) + 1
    cm = np.bincount(inv, weights=v) / np.bincount(inv)
    obs = float(v.mean()) if observed is None else float(observed)
    if g < 3:
        return Power(g, float("nan"), float("nan"), float("nan"), obs, True)
    sd = float(cm.std(ddof=1))
    se = sd / math.sqrt(g)
    mde = 2.487 * se
    return Power(g, sd, se, mde, obs, bool(abs(obs) < mde))


def adjust_many(p_values: Sequence[float], method: str = "bh") -> np.ndarray:
    """Adjusted p-values over many knowledge items / axes tested at once. 'bh' = Benjamini-Hochberg q-values via
    pattern_stats.bh_qvalues (reused, not copied); 'holm' = pattern_reliability.holm. NaN p-values stay NaN and are not counted."""
    from .. import pattern_stats as PS, pattern_reliability as PR
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = PS.bh_qvalues(p[ok]) if method == "bh" else np.asarray(PR.holm(p[ok]), float)
    return out


def lower_confidence_gain(values, clusters=None, level: float = 0.9, n_boot: int = 800, seed: int = 0) -> float:
    """One-sided lower bound of the mean gain (cluster bootstrap): the number a cautious consumer should plan on."""
    b = cluster_bootstrap_mean(values, clusters, n_boot=n_boot, level=2 * level - 1, seed=seed)
    return b.lo


def clusters_needed(effect: float, between_cluster_sd: float, *, z_alpha: float = 1.645, z_power: float = 0.842) -> int:
    """Planning: how many independent clusters (months) a cross-context test needs to detect a true mean gain of `effect` with 80%
    power at a one-sided 5% level, given the between-cluster spread. Inverse of power_analysis; use it to say how much unseen data
    a claim of transfer would have to wait for."""
    if not (effect > 0 and between_cluster_sd > 0):
        raise ValueError("effect and between_cluster_sd must be positive")
    return int(math.ceil(((z_alpha + z_power) * between_cluster_sd / effect) ** 2))


def verdict_table(rows: Sequence[Mapping]):
    """Tidy table of per-axis verdicts for reports: one row per mapping with axis, same, cross, ratio, flag, label. Missing keys
    become NaN/'' rather than being dropped, so an untested axis stays visible."""
    import pandas as pd
    cols = ["axis", "same", "cross", "ratio", "specialisation", "stability", "label"]
    text = ("axis", "specialisation", "label")
    return pd.DataFrame([{c: r.get(c, "" if c in text else float("nan")) for c in cols} for r in rows], columns=cols)


@dataclass(frozen=True)
class SpecialisationVerdict:
    """Section 27's verdict with the interval that supports it."""
    label: str                     # OVER_SPECIALISED | SUSPECTED | GENERAL | UNDEFINED
    ratio: RatioResult
    ratio_lo: float | None
    ratio_hi: float | None
    p_below_floor: float           # share of bootstrap draws (with a positive denominator) whose ratio fell under the floor
    denominator_risk: float
    reasons: tuple


def over_specialisation_verdict(cross_values, same_values, cross_clusters=None, same_clusters=None, *, floor: float = RATIO_FLOOR, eps: float = EPS_GAIN,
                                n_boot: int = 1500, seed: int = 0, confident: float = 0.95) -> SpecialisationVerdict:
    """Bootstrap the ratio cross/same (clusters resampled, draws with a denominator <= eps discarded and counted) and decide:
    OVER_SPECIALISED when at least `confident` of the valid draws fall under `floor`; GENERAL when at least `confident` are at or above
    it; SUSPECTED otherwise; UNDEFINED when the point ratio has no meaning (no same-context gain, NaN) or the denominator reaches
    zero too often to say. The interval and the share below the floor are returned, so the verdict can be audited."""
    rng = np.random.default_rng(seed)
    dc, mc, nc, _ = bootstrap_draws(cross_values, cross_clusters, n_boot, rng)
    ds, ms, ns, _ = bootstrap_draws(same_values, same_clusters, n_boot, rng)
    point = transfer_ratio(mc, ms, eps=eps, n_cross=nc, n_same=ns)
    if len(dc) == 0 or len(ds) == 0:
        return SpecialisationVerdict("UNDEFINED", point, None, None, float("nan"), 1.0, ("too few clusters to resample",))
    k = min(len(dc), len(ds))
    good = ds[:k] > eps
    risk = float(1 - good.mean())
    if point.status != RatioStatus.OK or good.sum() < 20 or risk > 0.05:
        return SpecialisationVerdict("UNDEFINED", point, None, None, float("nan"), risk,
                                     (f"ratio {point.status.value}; denominator reached zero in {risk:.0%} of draws",))
    r = np.clip(dc[:k][good] / ds[:k][good], -RATIO_CAP, RATIO_CAP)
    below = float((r < floor).mean())
    lo, hi = float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
    if below >= confident:
        label, why = "OVER_SPECIALISED", f"{below:.0%} of bootstrap ratios are under {floor:.2f}"
    elif below <= 1 - confident:
        label, why = "GENERAL", f"only {below:.0%} of bootstrap ratios are under {floor:.2f}"
    else:
        label, why = "SUSPECTED", f"{below:.0%} of bootstrap ratios are under {floor:.2f}: not settled"
    return SpecialisationVerdict(label, point, lo, hi, below, risk, (why,))
