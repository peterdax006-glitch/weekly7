"""Time-aware memory (contract C62 section 14; checklist B07 temporal index, supports B11/J06). Serves Bible phase 4/12
(pattern lifecycle, decay) at the generic knowledge level.

Every knowledge item gets a LEARNED temporal behaviour - PERSISTENT, SLOW_DECAY, FAST_DECAY, EPISODIC, REGIME_BOUND,
EVENT_BOUND, SEASONAL or UNKNOWN - estimated from its own evidence series, never from an assumed half-life. Classification is
model competition on one weighted-chi-square + BIC scale:

    zero effect | constant | exponential decay to zero | decay to a floor | group means (regime / event / season) | on-off bursts

A structured class wins only if it beats "constant" by a margin AND is not within a hair of a rival of a different class
(then the honest answer is UNKNOWN with the rivals listed). The profile carries the expected useful lifetime, its uncertainty
(posterior over the decay constant on a likelihood grid - deterministic, no draws) and machine-readable recovery conditions
that plug straight into `retirement.RecoveryCondition`. Everything takes `now`; a series containing a date at/after `now`
fails closed."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from scipy import stats as sps

from .core import FirewallBreach, TemporalClass, as_date, require_past, stable_hash
from .retirement import RecoveryCondition

TC = TemporalClass
DECAY = (TC.SLOW_DECAY, TC.FAST_DECAY)


# ------------------------------------------------------------------------------------------------- input series

@dataclasses.dataclass(frozen=True)
class EffectSeries:
    """Per-period effect estimates of one item, oriented as the caller likes (sign is handled internally).
    `ses` are standard errors of each period's estimate; `labels` optional regime/event tags per period."""
    dates: tuple[str, ...]
    values: tuple[float, ...]
    ses: tuple[float, ...] | None = None
    regimes: tuple[str, ...] | None = None
    events: tuple[bool, ...] | None = None

    def validate(self, now) -> list[str]:
        errs = []
        n = len(self.dates)
        if len(self.values) != n:
            errs.append("values/dates length mismatch")
        for name in ("ses", "regimes", "events"):
            v = getattr(self, name)
            if v is not None and len(v) != n:
                errs.append(f"{name} length mismatch")
        if errs:
            return errs
        ds = [as_date(d) for d in self.dates]
        if any(b <= a for a, b in zip(ds, ds[1:])):
            errs.append("dates not strictly increasing")
        if n and ds[-1] >= as_date(now):
            errs.append(f"series reaches {ds[-1]} which is not strictly before now={now}")
        if any(not math.isfinite(v) for v in self.values):
            errs.append("non-finite value")
        if self.ses is not None and any((not math.isfinite(s)) or s < 0 for s in self.ses):
            errs.append("bad standard error")
        return errs

    def require_valid(self, now) -> None:
        errs = self.validate(now)
        if any("not strictly before" in e for e in errs):
            raise FirewallBreach("temporal series: " + "; ".join(errs))
        if errs:
            raise ValueError("bad EffectSeries: " + "; ".join(errs))


def from_outcomes(dates: Sequence[Any], values: Sequence[float], now, period_days: int = 30,
                  regimes: Sequence[str] | None = None, events: Sequence[bool] | None = None) -> EffectSeries:
    """Aggregate raw per-trade / per-day signed outcomes into period bins (mean, standard error). Rows dated at/after `now`
    are dropped (not yet known). A bin's regime is its majority label; a bin is an event bin if most of its rows are."""
    if period_days < 1:
        raise ValueError("period_days must be >= 1")
    cut = as_date(now)
    rows = []
    for i, (d, v) in enumerate(zip(dates, values)):
        dd = as_date(d)
        if dd < cut and math.isfinite(float(v)):
            rows.append((dd.toordinal() // period_days, dd, float(v), regimes[i] if regimes is not None else None,
                         bool(events[i]) if events is not None else None))
    if not rows:
        return EffectSeries((), (), (), None if regimes is None else (), None if events is None else ())
    rows.sort(key=lambda r: (r[0], r[1]))
    out_d, out_v, out_s, out_r, out_e = [], [], [], [], []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        chunk = rows[i:j]
        vals = np.array([c[2] for c in chunk])
        out_d.append(chunk[len(chunk) // 2][1].isoformat())
        out_v.append(float(vals.mean()))
        out_s.append(float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"))
        if regimes is not None:
            labs = cast(list[str], [c[3] for c in chunk])
            out_r.append(max(sorted(set(labs)), key=labs.count))
        if events is not None:
            out_e.append(sum(1 for c in chunk if c[4]) * 2 > len(chunk))
        i = j
    ses = np.array(out_s, float)
    ok = ses[np.isfinite(ses)]
    fill = float(np.median(ok)) if len(ok) else float("nan")        # single-row bins borrow the median se
    ses = np.where(np.isfinite(ses), ses, fill)
    return EffectSeries(tuple(out_d), tuple(out_v), tuple(float(s) for s in ses) if np.isfinite(ses).all() else None,
                        tuple(out_r) if regimes is not None else None, tuple(out_e) if events is not None else None)


# ------------------------------------------------------------------------------------------------- profile

@dataclasses.dataclass(frozen=True)
class ModelFit:
    name: str
    klass: str
    k: int                                   # effective parameters (change-points count for bursts)
    chi2: float                              # noise-scaled weighted residual sum of squares
    bic: float
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TemporalProfile:
    """The learned temporal behaviour of one item as of `as_of` (immutable; a re-estimate is a new profile)."""
    knowledge_id: str
    as_of: str
    klass: str
    n_bins: int
    span_days: int
    reason: str
    tau_days: float | None = None            # decay constant of the winning decay model (median of its posterior)
    half_life_days: float | None = None
    lifetime_days: float | None = None       # expected useful lifetime measured from the series start; None = unbounded/unknown
    lifetime_lo: float | None = None         # 5% posterior quantile (a persistent item reports a lower bound only)
    lifetime_hi: float | None = None
    remaining_days: float | None = None      # lifetime_days minus age at as_of (floored at 0)
    remaining_lo: float | None = None
    p_fast: float | None = None
    uncertain: bool = False
    winner: str = ""
    scores: Mapping[str, float] = dataclasses.field(default_factory=dict)       # model name -> BIC
    alternatives: tuple[str, ...] = ()
    group_effects: Mapping[str, float] = dataclasses.field(default_factory=dict)
    active_groups: tuple[str, ...] = ()
    recovery: tuple[RecoveryCondition, ...] = ()
    burst_days: float | None = None
    gap_days: float | None = None
    fit: Mapping[str, float] = dataclasses.field(default_factory=dict)     # parameters needed to forecast (see predicted_effect)

    def validate(self) -> list[str]:
        errs = []
        try:
            TC(self.klass)
        except ValueError:
            errs.append(f"unknown class {self.klass}")
        if self.klass in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value) and self.tau_days is None:
            errs.append("decay class without tau")
        if self.klass == TC.UNKNOWN.value and not self.reason:
            errs.append("UNKNOWN must say why")
        if self.lifetime_lo is not None and self.lifetime_days is not None and self.lifetime_lo > self.lifetime_days + 1e-9:
            errs.append("lifetime_lo above lifetime")
        for c in self.recovery:
            errs.extend(c.validate())
        return errs

    @property
    def profile_id(self) -> str:
        return stable_hash(dataclasses.asdict(self))


# ------------------------------------------------------------------------------------------------- estimation helpers

def _robust_noise_sd(y: np.ndarray) -> float:
    """Noise sd from first differences: differencing cancels slow structure (trend, regimes, seasons) so what is left is noise."""
    if len(y) < 4:
        return float("nan")
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / math.sqrt(2.0))


def _weights(y: np.ndarray, ses: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Inverse-variance weights and an overdispersion factor phi >= 1 (reported ses are trusted only as far as the
    difference-based noise estimate agrees with them: autocorrelated outcomes make ses too small)."""
    nsd = _robust_noise_sd(y)
    if ses is None or not np.all(np.isfinite(ses)) or np.all(ses <= 0):
        sd = nsd if nsd > 0 else float(np.std(y)) or 1.0
        return np.full(len(y), 1.0), sd * sd
    floor = 0.05 * float(np.median(ses[ses > 0]))
    s = np.maximum(ses, floor)
    phi = 1.0
    if nsd > 0:
        phi = max(1.0, (nsd ** 2) / float(np.mean(s ** 2)))
    return 1.0 / (s * s), phi


def _bic(chi2: float, k: float, n: int, phi: float) -> float:
    return chi2 / phi + k * math.log(max(n, 2))


def _tau_grid(span: float, n: int = 64) -> np.ndarray:
    return np.exp(np.linspace(math.log(10.0), math.log(max(40.0 * span, 400.0)), n))


def _decay_profile(t: np.ndarray, y: np.ndarray, w: np.ndarray, taus: np.ndarray, floor: bool):
    """Closed-form weighted fits over a grid of tau. Returns (chi2[G], a[G], c[G], valid[G]); valid needs a>0 (a decay, not
    a growth) and c>=0 (a floor cannot reverse the effect - reversal is a different story)."""
    X = np.exp(-t[None, :] / taus[:, None])
    S0, Sy = w.sum(), (w * y).sum()
    Sx, Sxx = (X * w).sum(1), (X * X * w).sum(1)
    Sxy = (X * (w * y)).sum(1)
    if floor:
        det = S0 * Sxx - Sx ** 2
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        c = (Sxx * Sy - Sx * Sxy) / det
        a = (S0 * Sxy - Sx * Sy) / det
    else:
        a = Sxy / np.where(Sxx > 0, Sxx, np.nan)
        c = np.zeros_like(a)
    pred = c[:, None] + a[:, None] * X
    chi2 = ((y[None, :] - pred) ** 2 * w[None, :]).sum(1)
    valid = np.isfinite(chi2) & (a > 0) & (c >= -1e-12)
    return chi2, a, c, valid


def _wquantile(x: np.ndarray, p: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cx, cp = x[o], np.cumsum(p[o])
    cp = cp / cp[-1]
    return float(cx[min(int(np.searchsorted(cp, q)), len(cx) - 1)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _group_fit(y, w, labels: Sequence[Any], klass: TC, name: str, n: int, phi: float, min_per: int = 2):
    """Piecewise-constant model with one mean per label. Labels with fewer than `min_per` bins are pooled into '_rare'."""
    lab_s = np.array([str(x) for x in labels])
    uniq0, counts = np.unique(lab_s, return_counts=True)
    keep = {u for u, c in zip(uniq0, counts) if c >= min_per}
    labs = np.array([x if x in keep else "_rare" for x in lab_s])
    uniq = [u for u in np.unique(labs)]
    if len(uniq) < 2 or len(uniq) > max(2, n // 3):
        return None
    chi2, means = 0.0, {}
    for u in uniq:
        m = labs == u
        mu = float((w[m] * y[m]).sum() / w[m].sum())
        chi2 += float((w[m] * (y[m] - mu) ** 2).sum())
        means[u] = mu
    se = {u: 1.0 / math.sqrt(float(w[labs == u].sum() / phi)) for u in uniq}
    return ModelFit(name, klass.value, len(uniq), chi2, _bic(chi2, len(uniq), n, phi),
                    {"means": means, "se": se, "labels": labs.tolist()})


def _phase_labels(dates: Sequence[dt.date]) -> list[str]:
    return [f"{d.month:02d}" for d in dates]


def _month_intervals(months: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous (wrapping) month intervals, e.g. [11, 12, 1] -> [(11, 13)] expressed as lo..hi with hi possibly > 12."""
    ms = sorted(set(months))
    if not ms:
        return []
    out, lo, prev = [], ms[0], ms[0]
    for m in ms[1:]:
        if m == prev + 1:
            prev = m
            continue
        out.append((lo, prev))
        lo = prev = m
    out.append((lo, prev))
    if len(out) > 1 and out[0][0] == 1 and out[-1][1] == 12:
        out[0] = (out[-1][0], out[0][1] + 12)
        out.pop()
    return out


# ------------------------------------------------------------------------------------------------- the classifier

@dataclasses.dataclass(frozen=True)
class TemporalConfig:
    """Labelling conventions and evidence margins. `fast_days` only names the boundary between SLOW and FAST decay in the
    label; the decay constant itself is always estimated. Margins are to be tuned in the validation wave."""
    min_bins: int = 10
    min_span_days: int = 180
    min_delta_bic: float = 10.0              # a structured model must beat "constant" by this much
    rival_delta_bic: float = 2.0             # different-class rivals closer than this make the answer ambiguous
    fast_days: float = 120.0
    useful_fraction: float = 0.25            # useful while fitted effect >= this fraction of its initial value (if no min_effect)
    min_effect: float | None = None          # absolute usefulness floor overriding useful_fraction
    active_t: float = 2.0                    # group is 'active' if its mean effect has |t| >= this and the right sign
    min_burst_runs: int = 2
    lifetime_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)

    def validate(self) -> list[str]:
        errs = []
        if self.min_bins < 6:
            errs.append("min_bins < 6 cannot support model comparison")
        if self.min_delta_bic <= 0 or self.rival_delta_bic < 0:
            errs.append("bic margins must be positive")
        if not 0 < self.useful_fraction < 1:
            errs.append("useful_fraction outside (0,1)")
        return errs


def _unknown(kid, as_of, n, span, reason, scores=None, alternatives=(), winner="") -> TemporalProfile:
    return TemporalProfile(kid, as_date(as_of).isoformat(), TC.UNKNOWN.value, n, int(span), reason, winner=winner,
                           scores=scores or {}, alternatives=tuple(alternatives))


def estimate(knowledge_id: str, series: EffectSeries, now, cfg: TemporalConfig | None = None) -> TemporalProfile:
    """Estimate the temporal class of one item from evidence dated strictly before `now`."""
    cfg = cfg or TemporalConfig()
    errs = cfg.validate()
    if errs:
        raise ValueError("bad TemporalConfig: " + "; ".join(errs))
    series.require_valid(now)
    n = len(series.dates)
    ds = [as_date(d) for d in series.dates]
    span = (ds[-1] - ds[0]).days if n else 0
    if n < cfg.min_bins or span < cfg.min_span_days:
        return _unknown(knowledge_id, now, n, span, f"insufficient evidence: {n} periods over {span} days "
                        f"(need {cfg.min_bins} and {cfg.min_span_days})")
    y = np.array(series.values, float)
    ses = np.array(series.ses, float) if series.ses is not None else None
    wt0 = np.ones(n) if ses is None else 1.0 / np.maximum(ses, 1e-12) ** 2
    s = 1.0 if float((y * wt0).sum()) >= 0 else -1.0                 # orient so that "positive" means the item works
    y = y * s
    w, phi = _weights(y, ses)
    t = np.array([(d - ds[0]).days for d in ds], float)
    fits: list[ModelFit] = []
    fits.append(ModelFit("zero", "ZERO", 0, float((w * y * y).sum()), _bic(float((w * y * y).sum()), 0, n, phi)))
    mu = float((w * y).sum() / w.sum())
    chi_c = float((w * (y - mu) ** 2).sum())
    fits.append(ModelFit("constant", TC.PERSISTENT.value, 1, chi_c, _bic(chi_c, 1, n, phi), {"mean": mu * s}))
    taus = _tau_grid(span)
    prof = {}
    for floor in (False, True):
        chi2, a, c, valid = _decay_profile(t, y, w, taus, floor)
        if valid.any():
            idx = int(np.argmin(np.where(valid, chi2, np.inf)))
            name = "decay_floor" if floor else "decay_zero"
            fits.append(ModelFit(name, "DECAY", 3 if floor else 2, float(chi2[idx]), _bic(float(chi2[idx]), 3 if floor else 2, n, phi),
                                 {"tau": float(taus[idx]), "a": float(a[idx]), "c": float(c[idx])}))
            prof[name] = (chi2, a, c, valid)
    if series.regimes is not None:
        f = _group_fit(y, w, series.regimes, TC.REGIME_BOUND, "regime", n, phi)
        if f:
            fits.append(f)
    if series.events is not None:
        f = _group_fit(y, w, ["event" if e else "none" for e in series.events], TC.EVENT_BOUND, "event", n, phi)
        if f and set(f.detail["means"]) >= {"event", "none"}:
            fits.append(f)
    years = {d.year for d in ds}
    if len(years) >= 2 and span >= 540:
        f = _group_fit(y, w, _phase_labels(ds), TC.SEASONAL, "season", n, phi)
        if f and _season_replicates(y, ds):
            fits.append(f)
    ep = _burst_fit(y, w, t, n, phi, cfg)
    if ep:
        fits.append(ep)
    scores = {f.name: round(f.bic, 3) for f in fits}
    by_name = {f.name: f for f in fits}
    fitinfo = {"s": s, "t0": float(ds[0].toordinal()), "level": mu * s}
    best_any = min(fits, key=lambda f: f.bic)
    if by_name["zero"].bic - best_any.bic < cfg.min_delta_bic:
        return _unknown(knowledge_id, now, n, span, "no detectable effect in the evidence series (zero model not beaten by "
                        f"{cfg.min_delta_bic} BIC)", scores)
    const = by_name["constant"]
    cands = sorted((f for f in fits if f.klass not in ("ZERO", TC.PERSISTENT.value)), key=lambda f: f.bic)
    if not cands or const.bic - cands[0].bic < cfg.min_delta_bic:
        return _with_fit(_persistent_profile(knowledge_id, now, n, span, ds, t, y, w, taus, chi_c, phi, cfg, scores, s), fitinfo)
    win = cands[0]
    rivals = [f for f in cands[1:] if f.klass != win.klass and f.bic - win.bic < cfg.rival_delta_bic]
    if rivals:
        return _unknown(knowledge_id, now, n, span, "ambiguous: " + ", ".join(f.name for f in [win] + rivals)
                        + " explain the series about equally well", scores, [f.name for f in [win] + rivals], win.name)
    if win.klass == "DECAY":
        d = win.detail
        return _with_fit(_decay_out(knowledge_id, now, n, span, ds, phi, cfg, scores, prof[win.name], win.name),
                         dict(fitinfo, tau=d['tau'], a=d['a'] * s, c=d['c'] * s))
    if win.klass == TC.EPISODIC.value:
        return _with_fit(_episodic_profile(knowledge_id, now, n, span, t, win, cfg, scores), dict(fitinfo, on=win.detail['on'] * s,
                                                                                              off=win.detail['off'] * s))
    return _with_fit(_group_profile(knowledge_id, now, n, span, ds, y, w, phi, win, cfg, scores, s), fitinfo)


def _with_fit(p: TemporalProfile, info: Mapping[str, float]) -> TemporalProfile:
    return dataclasses.replace(p, fit=dict(info))


def _season_replicates(y, ds) -> bool:
    """Seasonal means must repeat: phase means from odd and even years must agree in sign pattern (rank correlation > 0.3)."""
    ph = np.array([d.month for d in ds])
    yr = np.array([d.year for d in ds])
    A, B = [], []
    for m in range(1, 13):
        a, b = y[(ph == m) & (yr % 2 == 0)], y[(ph == m) & (yr % 2 == 1)]
        if len(a) and len(b):
            A.append(a.mean()); B.append(b.mean())
    if len(A) < 6:
        return False
    r, _ = sps.spearmanr(A, B)
    return bool(np.isfinite(r) and r > 0.3)


def _burst_fit(y, w, t, n, phi, cfg) -> ModelFit | None:
    sd = 1.0 / np.sqrt(w)
    on = (y / sd) > 1.0
    on = on | (np.roll(on, 1) & np.roll(on, -1))                     # bridge one-bin dropouts inside a burst
    on[0] = (y[0] / sd[0]) > 1.0
    on[-1] = (y[-1] / sd[-1]) > 1.0
    runs = _runs(on)
    if len(runs) < cfg.min_burst_runs or on.all() or not on.any():
        return None
    lvl_on = float((w[on] * y[on]).sum() / w[on].sum())
    lvl_off = float((w[~on] * y[~on]).sum() / w[~on].sum())
    if lvl_on <= 0 or lvl_off > 0.5 * lvl_on:
        return None                                                  # no quiet baseline: that is noise around a level, not bursts
    chi2 = float((w[on] * (y[on] - lvl_on) ** 2).sum() + (w[~on] * (y[~on] - lvl_off) ** 2).sum())
    k = 2 + 3 * len(runs)                                            # levels + (start, end, selection) per burst
    dt_bin = float(np.median(np.diff(t))) if n > 1 else 1.0
    lens = [(b - a + 1) * dt_bin for a, b in runs]
    gaps = [(runs[i + 1][0] - runs[i][1] - 1) * dt_bin for i in range(len(runs) - 1)]
    return ModelFit("bursts", TC.EPISODIC.value, k, chi2, _bic(chi2, k, n, phi),
                    {"runs": runs, "lens": lens, "gaps": gaps, "on": lvl_on, "off": lvl_off})


def _persistent_profile(kid, now, n, span, ds, t, y, w, taus, chi_c, phi, cfg, scores, s) -> TemporalProfile:
    """Constant wins. Lifetime is unbounded; the uncertainty is a LOWER bound: the shortest decay constant the data cannot reject."""
    chi2, a, c, valid = _decay_profile(t, y, w, taus, False)
    ok = valid & ((chi2 - chi_c) / phi <= 3.84)
    lo = float(taus[ok].min()) if ok.any() else float(taus.max())
    return TemporalProfile(kid, as_date(now).isoformat(), TC.PERSISTENT.value, n, span,
                           "constant effect not beaten by any structured model", lifetime_days=None, lifetime_lo=lo,
                           remaining_days=None, remaining_lo=max(0.0, lo - (as_date(now) - ds[0]).days), winner="constant",
                           scores=scores)


def _decay_out(kid, now, n, span, ds, phi, cfg, scores, fitted, best_name) -> TemporalProfile:
    chi2, a, c, valid = fitted
    chi_min = float(np.min(np.where(valid, chi2, np.inf)))
    post = np.where(valid, np.exp(-0.5 * (chi2 - chi_min) / phi), 0.0)
    post = post / post.sum()
    taus = _tau_grid(span)
    thr = abs(cfg.min_effect) if cfg.min_effect is not None else cfg.useful_fraction * (a + c)
    with np.errstate(all="ignore"):
        life = np.where(c >= thr, 1e12, taus * np.log(np.maximum(a / np.maximum(thr - c, 1e-15), 1.0000001)))
    ql, qm, qh = (_wquantile(life[valid], post[valid], q) for q in cfg.lifetime_quantiles)
    tau_med = _wquantile(taus[valid], post[valid], 0.5)
    p_fast = float(post[valid & (taus < cfg.fast_days)].sum())
    klass = TC.FAST_DECAY if tau_med < cfg.fast_days else TC.SLOW_DECAY
    age = (as_date(now) - ds[0]).days
    unb = lambda v: None if v >= 1e11 else float(v)
    return TemporalProfile(kid, as_date(now).isoformat(), klass.value, n, span,
                           f"{best_name} beats a constant by >= {cfg.min_delta_bic} BIC; tau posterior median {tau_med:.0f} days",
                           tau_days=tau_med, half_life_days=tau_med * math.log(2), lifetime_days=unb(qm), lifetime_lo=unb(ql),
                           lifetime_hi=unb(qh), remaining_days=None if qm >= 1e11 else max(0.0, qm - age),
                           remaining_lo=None if ql >= 1e11 else max(0.0, ql - age), p_fast=p_fast,
                           uncertain=0.2 < p_fast < 0.8, winner=best_name, scores=scores)


def _episodic_profile(kid, now, n, span, t, win, cfg, scores) -> TemporalProfile:
    lens, gaps = win.detail["lens"], win.detail["gaps"]
    burst = float(np.mean(lens))
    gap = float(np.mean(gaps)) if gaps else None
    se = float(np.std(lens, ddof=1) / math.sqrt(len(lens))) if len(lens) > 1 else burst
    return TemporalProfile(kid, as_date(now).isoformat(), TC.EPISODIC.value, n, span,
                           f"{len(lens)} bursts over a quiet baseline (bursts beat constant by >= {cfg.min_delta_bic} BIC)",
                           lifetime_days=burst, lifetime_lo=max(0.0, burst - 1.645 * se), lifetime_hi=burst + 1.645 * se,
                           uncertain=len(lens) < 4, winner="bursts", scores=scores, burst_days=burst, gap_days=gap)


def _group_profile(kid, now, n, span, ds, y, w, phi, win, cfg, scores, s) -> TemporalProfile:
    means, ses = win.detail["means"], win.detail["se"]
    labels = np.array(win.detail["labels"])
    active = tuple(sorted(u for u, m in means.items() if u != "_rare" and m > 0 and m / max(ses[u], 1e-12) >= cfg.active_t))
    if not active:
        return _unknown(kid, now, n, span, f"{win.name} groups differ but none has a significant effect of its own", scores,
                        [win.name], win.name)
    klass = TC(win.klass)
    conds: list[RecoveryCondition] = []
    lifetime = None
    if klass is TC.REGIME_BOUND:
        conds = [RecoveryCondition("regime", f"regime {u} returns", equals=u) for u in active]
        lens = _label_run_lengths(labels, ds, set(active))
        lifetime = float(np.mean(lens)) if lens else None
    elif klass is TC.EVENT_BOUND:
        conds = [RecoveryCondition("event", "an event of the learned kind occurs", equals="True")]
        lens = _label_run_lengths(labels, ds, set(active))
        lifetime = float(np.mean(lens)) if lens else None
    else:
        months = [int(u) for u in active if u.isdigit()]
        conds = [RecoveryCondition("month", f"calendar months {lo}-{hi}", lo=float(lo), hi=float(hi))
                 for lo, hi in _month_intervals(months)]
        lifetime = 30.4 * len(months)
    unit = {TC.REGIME_BOUND: "regime", TC.EVENT_BOUND: "event", TC.SEASONAL: "season"}[klass]
    return TemporalProfile(kid, as_date(now).isoformat(), klass.value, n, span,
                           f"{unit} group means beat a constant by >= {cfg.min_delta_bic} BIC; active groups {list(active)}",
                           lifetime_days=lifetime, winner=win.name, scores=scores,
                           group_effects={u: float(m * s) for u, m in means.items()}, active_groups=active,
                           recovery=tuple(conds))


def _label_run_lengths(labels: np.ndarray, ds: Sequence[dt.date], active: set[str]) -> list[float]:
    mask = np.array([str(l) in active for l in labels])
    out = []
    for a, b in _runs(mask):
        nxt = ds[b + 1] if b + 1 < len(ds) else ds[b]
        out.append(float((nxt - ds[a]).days) or 1.0)
    return out


# ------------------------------------------------------------------------------------------------- using a profile

def evidence_weights(profile: TemporalProfile, obs_dates: Sequence[Any], now, obs_regimes: Sequence[str] | None = None,
                     current_regime: str | None = None, floor: float = 0.05) -> np.ndarray:
    """How much should each past observation count today, given what we learned about the item's time behaviour?
    PERSISTENT 1; decays exp(-age/tau); SEASONAL same-phase months only; REGIME_BOUND only same-regime rows;
    EPISODIC and UNKNOWN discount by age with the lifetime (or a year when nothing is known). Never below `floor`,
    because 'no weight' would be a claim of certainty the estimate cannot support."""
    cut = as_date(now)
    ds = [as_date(d) for d in obs_dates]
    if any(d >= cut for d in ds):
        raise FirewallBreach("evidence_weights: observation at/after now")
    age = np.array([(cut - d).days for d in ds], float)
    k = profile.klass
    if k == TC.PERSISTENT.value:
        w = np.ones(len(ds))
    elif k in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value):
        w = np.exp(-age / profile.tau_days)
    elif k == TC.SEASONAL.value:
        act = {int(u) for u in profile.active_groups if u.isdigit()}
        w = np.array([1.0 if d.month in act else floor for d in ds]) if cut.month in act else np.full(len(ds), floor)
    elif k in (TC.REGIME_BOUND.value, TC.EVENT_BOUND.value):
        if obs_regimes is None or current_regime is None:
            w = np.full(len(ds), 0.5)
        else:
            w = np.array([1.0 if str(r) == str(current_regime) else floor for r in obs_regimes])
    else:
        scale = profile.lifetime_days or 365.0
        w = np.exp(-age / max(scale, 30.0))
    return np.maximum(w, floor)


def expected_influence(profile: TemporalProfile, now, context: Mapping[str, Any] | None = None) -> float:
    """Conservative multiplier in [0, 1] for how much of the item's effect to expect today. Decays use the LOWER lifetime
    quantile; conditional classes give 1 only when a recovery condition holds in `context`; UNKNOWN gives 0.5 (unknown is not
    zero and not one)."""
    k, cut = profile.klass, as_date(now)
    if k == TC.PERSISTENT.value:
        return 1.0
    if k in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value):
        if profile.remaining_lo is None:
            return 1.0                                   # even the pessimistic lifetime is unbounded
        over = (cut - as_date(profile.as_of)).days - profile.remaining_lo
        return 1.0 if over <= 0 else float(math.exp(-over / max(profile.tau_days or 1.0, 1.0)))
    if k in (TC.REGIME_BOUND.value, TC.EVENT_BOUND.value, TC.SEASONAL.value):
        ctx = dict(context or {})
        if k == TC.SEASONAL.value:
            ctx.setdefault("month", cut.month)
        return 1.0 if any(c.satisfied_by(ctx) for c in profile.recovery) else 0.0
    if k == TC.EPISODIC.value:
        return 0.5
    return 0.5


def needs_review(profile: TemporalProfile, now) -> bool:
    """A profile older than its own remaining-lifetime lower bound (or than a year when it has none) should be re-estimated."""
    age = (as_date(now) - as_date(profile.as_of)).days
    if profile.klass == TC.UNKNOWN.value:
        return age >= 90
    if profile.remaining_lo is not None:
        return age >= max(30.0, profile.remaining_lo)
    return age >= 365


class TemporalMemory:
    """Versioned store of profiles (B07 temporal index). History is never overwritten."""

    def __init__(self):
        self._p: dict[str, list[TemporalProfile]] = {}

    def add(self, profile: TemporalProfile) -> None:
        errs = profile.validate()
        if errs:
            raise ValueError("invalid profile: " + "; ".join(errs))
        hist = self._p.setdefault(profile.knowledge_id, [])
        if hist and as_date(profile.as_of) < as_date(hist[-1].as_of):
            raise FirewallBreach(f"{profile.knowledge_id}: profile dated {profile.as_of} precedes {hist[-1].as_of}")
        hist.append(profile)

    def get(self, kid: str, as_of) -> TemporalProfile | None:
        """Newest profile estimated strictly before `as_of` (a profile is built from data before its own as_of)."""
        cut = as_date(as_of)
        out = [p for p in self._p.get(kid, ()) if as_date(p.as_of) < cut]
        return out[-1] if out else None

    def history(self, kid: str) -> tuple[TemporalProfile, ...]:
        return tuple(self._p.get(kid, ()))

    def reclassified(self, kid: str) -> list[tuple[str, str, str]]:
        h = self._p.get(kid, [])
        return [(b.as_of, a.klass, b.klass) for a, b in zip(h, h[1:]) if a.klass != b.klass]

    def class_counts(self, as_of) -> dict[str, int]:
        out: dict[str, int] = {}
        for kid in self._p:
            p = self.get(kid, as_of)
            if p:
                out[p.klass] = out.get(p.klass, 0) + 1
        return dict(sorted(out.items()))

    def due(self, as_of) -> list[str]:
        return sorted(k for k in self._p if (p := self.get(k, as_of)) is not None and needs_review(p, as_of))

    def family_lifetimes(self, as_of, family: Callable[[str], str]) -> dict[str, dict[str, float]]:
        """Median observed lifetime by family key - a data-driven prior for new items of that family (never a fixed half-life)."""
        buckets: dict[str, list[float]] = {}
        for kid in self._p:
            p = self.get(kid, as_of)
            if p and p.lifetime_days:
                buckets.setdefault(family(kid), []).append(p.lifetime_days)
        return {f: {"n": len(v), "median": float(np.median(v)), "iqr": float(np.subtract(*np.percentile(v, [75, 25])))}
                for f, v in sorted(buckets.items())}


def describe(p: TemporalProfile) -> str:
    life = "unbounded" if p.lifetime_days is None else f"{p.lifetime_days:.0f}d"
    lo = "" if p.lifetime_lo is None else f" (>= {p.lifetime_lo:.0f}d at 95%)" if p.klass == TC.PERSISTENT.value else \
        f" [{p.lifetime_lo:.0f}, {'inf' if p.lifetime_hi is None else format(p.lifetime_hi, '.0f')}]"
    rec = "; recovery: " + ", ".join(c.description for c in p.recovery) if p.recovery else ""
    return f"{p.knowledge_id}: {p.klass}, useful lifetime {life}{lo}{rec} - {p.reason} - IMPLEMENTED - NOT VALIDATED"


# ------------------------------------------------------------------------------------------------- forecasting and recovery

def predicted_effect(profile: TemporalProfile, on_date, context: Mapping[str, Any] | None = None) -> float | None:
    """The effect the learned temporal model expects on `on_date`, in the item's own sign convention; None when the class
    gives no forecast (EPISODIC bursts are unpredictable by construction, UNKNOWN by definition). Decays extrapolate their
    fitted curve, PERSISTENT its level, conditional classes the mean of the active vs the inactive groups given `context`."""
    k, f = profile.klass, profile.fit
    if not f:
        return None
    if k == TC.PERSISTENT.value:
        return float(f["level"])
    if k in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value):
        age = as_date(on_date).toordinal() - f["t0"]
        return float(f["c"] + f["a"] * math.exp(-max(age, 0.0) / f["tau"]))
    if k in (TC.REGIME_BOUND.value, TC.EVENT_BOUND.value, TC.SEASONAL.value):
        ctx = dict(context or {})
        if k == TC.SEASONAL.value:
            ctx.setdefault("month", as_date(on_date).month)
        active = any(c.satisfied_by(ctx) for c in profile.recovery)
        vals = [v for g, v in profile.group_effects.items() if (g in profile.active_groups) == active and g != "_rare"]
        return float(np.mean(vals)) if vals else None
    return None


@dataclasses.dataclass(frozen=True)
class RecoveryTest:
    n: int
    mean_excess: float                       # mean of (observed - forecast) in the item's own sign
    z_excess: float                          # how many standard errors the recent evidence sits above the decayed forecast
    mean_observed: float
    z_observed: float
    recovered: bool
    reason: str


def recovery_test(profile: TemporalProfile, recent: EffectSeries, now, z_min: float = 2.0,
                  context: Mapping[str, Any] | None = None) -> RecoveryTest:
    """Has an item that the temporal model says should have faded (or gone quiet) come back? Recent evidence must be BOTH
    significantly positive on its own AND significantly above what the fitted decay forecasts - otherwise the 'recovery' is just
    the tail of the same decay. `recent` must contain only periods after `profile.as_of`."""
    recent.require_valid(now)
    keep = [i for i, d in enumerate(recent.dates) if as_date(d) > as_date(profile.as_of)]
    if len(keep) < 3 or recent.ses is None:
        return RecoveryTest(len(keep), 0.0, 0.0, 0.0, 0.0, False, "fewer than 3 periods after the profile date, or no standard errors")
    vals = np.array([recent.values[i] for i in keep], float)
    ses = np.maximum(np.array([recent.ses[i] for i in keep], float), 1e-12)
    pred = [predicted_effect(profile, recent.dates[i], context) for i in keep]
    sign = profile.fit.get("s", 1.0) if profile.fit else 1.0
    base = np.array([0.0 if q is None else q for q in pred], float)
    if not profile.fit:
        base = np.zeros(len(keep))
    w = 1.0 / ses ** 2
    se = 1.0 / math.sqrt(float(w.sum()))
    obs_mean = float((w * vals * sign).sum() / w.sum())
    exc_mean = float((w * (vals - base) * sign).sum() / w.sum())
    zo, ze = obs_mean / se, exc_mean / se
    ok = zo >= z_min and ze >= z_min
    why = ("recent evidence is significantly positive and above the decay forecast" if ok else
           "recent evidence does not beat the decay forecast" if zo >= z_min else "recent evidence is not significantly positive")
    return RecoveryTest(len(keep), exc_mean, ze, obs_mean, zo, bool(ok), why)


def to_evidence(series: EffectSeries, since, now):
    """Bridge to the retirement gates: summarise a temporal series over (since, now) as a retirement.Evidence (each period
    counts as one observation weighted by its own precision)."""
    from .retirement import Evidence
    lo, hi = as_date(since), as_date(now)
    idx = [i for i, d in enumerate(series.dates) if lo < as_date(d) < hi]
    if not idx or series.ses is None:
        return Evidence(0, 0.0, 0.0, str(lo), str(lo), None, "temporal")
    v = np.array([series.values[i] for i in idx], float)
    w = 1.0 / np.maximum(np.array([series.ses[i] for i in idx], float), 1e-12) ** 2
    return Evidence(len(idx), float((w * v).sum() / w.sum()), float(1.0 / math.sqrt(w.sum())), series.dates[idx[0]],
                    series.dates[idx[-1]], None, "temporal")


# ------------------------------------------------------------------------------------------------- lifetimes across items

@dataclasses.dataclass(frozen=True)
class KMCurve:
    times: tuple[float, ...]
    survival: tuple[float, ...]
    se: tuple[float, ...]                    # Greenwood standard errors
    at_risk: tuple[int, ...]
    n: int
    n_events: int

    def at(self, t: float) -> float:
        s = 1.0
        for ti, si in zip(self.times, self.survival):
            if ti <= t:
                s = si
            else:
                break
        return s

    def quantile(self, q: float) -> float | None:
        """Smallest time by which a share q of items has ended; None when the curve never falls that far (censored)."""
        for ti, si in zip(self.times, self.survival):
            if 1.0 - si >= q:
                return ti
        return None

    def median(self) -> float | None:
        return self.quantile(0.5)


def kaplan_meier(durations: Sequence[float], ended: Sequence[bool]) -> KMCurve:
    """Survival of knowledge items: durations in days from birth to failure (ended=True) or to `now` for items still working
    (ended=False, censored). Treating still-alive items as failures, or dropping them, both bias lifetimes short/long; this
    does neither. The curve is the data-driven prior for a new item of the same family - no fixed half-life assumed."""
    d = np.asarray(durations, float)
    e = np.asarray(ended, bool)
    if len(d) != len(e):
        raise ValueError("durations and ended differ in length")
    if len(d) and (d < 0).any():
        raise ValueError("negative duration")
    times, surv, ses, risk = [], [], [], []
    s, gw = 1.0, 0.0
    for t in np.unique(d[e]):
        n_i = int((d >= t).sum())
        d_i = int(((d == t) & e).sum())
        s *= 1.0 - d_i / n_i
        if n_i > d_i:
            gw += d_i / (n_i * (n_i - d_i))
        times.append(float(t)); surv.append(float(s)); ses.append(float(s * math.sqrt(gw))); risk.append(n_i)
    return KMCurve(tuple(times), tuple(surv), tuple(ses), tuple(risk), int(len(d)), int(e.sum()))


def profile_lifetimes(memory: "TemporalMemory", as_of, born: Mapping[str, Any]) -> tuple[list[float], list[bool]]:
    """Durations for `kaplan_meier` from a TemporalMemory: an item whose newest profile has a finite lifetime and whose age
    exceeds it has ENDED at that lifetime; every other item is censored at its current age. `born` maps id -> birth date."""
    dur, ended = [], []
    for kid, b in sorted(born.items()):
        p = memory.get(kid, as_of)
        if p is None:
            continue
        age = float((as_date(as_of) - as_date(b)).days)
        if p.lifetime_days is not None and p.klass in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value) and age >= p.lifetime_days:
            dur.append(float(p.lifetime_days)); ended.append(True)
        else:
            dur.append(age); ended.append(False)
    return dur, ended


# ------------------------------------------------------------------------------------------------- diagnostics

def class_stability(knowledge_id: str, series: EffectSeries, now, cfg: TemporalConfig | None = None, n_cuts: int = 5) -> dict:
    """Walk-forward stability: re-estimate the class with only the data available at each of `n_cuts` earlier cut dates (never
    later data) and report the class sequence. A class that flips as data arrives is a class we do not know yet."""
    series.require_valid(now)
    n = len(series.dates)
    cfg = cfg or TemporalConfig()
    cuts = sorted({int(round(n * f)) for f in np.linspace(0.5, 1.0, n_cuts)})
    seq = []
    for c in cuts:
        if c < cfg.min_bins:
            continue
        sub = EffectSeries(series.dates[:c], series.values[:c], None if series.ses is None else series.ses[:c],
                           None if series.regimes is None else series.regimes[:c], None if series.events is None else series.events[:c])
        cut_date = as_date(series.dates[c - 1]) + dt.timedelta(days=1)
        seq.append((cut_date.isoformat(), estimate(knowledge_id, sub, cut_date, cfg).klass))
    classes = [k for _, k in seq]
    flips = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
    final = classes[-1] if classes else TC.UNKNOWN.value
    agree = sum(1 for k in classes if k == final) / len(classes) if classes else 0.0
    return {"sequence": seq, "flips": flips, "final": final, "agreement_with_final": agree,
            "stable": bool(classes) and flips <= 1 and agree >= 0.6}


def profile_table(profiles: Sequence[TemporalProfile]) -> list[dict[str, Any]]:
    """Flat rows for reports/dashboards, sorted by id so two runs print identically."""
    rows = []
    for p in sorted(profiles, key=lambda q: q.knowledge_id):
        rows.append({"id": p.knowledge_id, "class": p.klass, "n_bins": p.n_bins, "span_days": p.span_days,
                     "tau_days": p.tau_days, "lifetime_days": p.lifetime_days, "lifetime_lo": p.lifetime_lo,
                     "remaining_lo": p.remaining_lo, "uncertain": p.uncertain, "recovery": [c.description for c in p.recovery],
                     "reason": p.reason})
    return rows


# ------------------------------------------------------------------------------------------------- legitimacy and uncertainty checks

def seasonal_legitimacy(series: EffectSeries, active_months: Sequence[int], min_years: int = 3, min_share: float = 0.75,
                        known_artefacts: Sequence[tuple[int, int]] = ()) -> dict[str, Any]:
    """Calendar-leak guard for a SEASONAL verdict. A window that merely coincides with one crisis year or a one-off calendar
    event is not a season. It must (a) recur with the same sign in at least `min_share` of >= `min_years` distinct years,
    (b) survive leaving each year out in turn (Welch t vs the inactive months stays >= 2), (c) not cover ten or more months,
    and (d) still hold after `known_artefacts` (year, month) pairs - index rebalances, crisis months - are removed."""
    act = {int(m) for m in active_months}
    skip = {(int(a), int(b)) for a, b in known_artefacts}
    rows = [(as_date(d), float(v)) for d, v in zip(series.dates, series.values) if (as_date(d).year, as_date(d).month) not in skip]
    ins = [(d, v) for d, v in rows if d.month in act]
    out = [v for d, v in rows if d.month not in act]
    years = sorted({d.year for d, _ in ins})
    res: dict[str, Any] = {"years_seen": len(years), "years_positive": 0, "loo_min_t": float("nan"), "legitimate": False, "reason": ""}
    if len(act) >= 10:
        return dict(res, reason="window covers ten or more months: not a season")
    if len(years) < min_years or len(out) < 3:
        return dict(res, reason=f"only {len(years)} distinct years of in-season data (need {min_years})")
    sign = 1.0 if np.mean([v for _, v in ins]) >= 0 else -1.0
    res["years_positive"] = sum(1 for yr in years if sign * np.mean([v for d, v in ins if d.year == yr]) > 0)
    ts = []
    for yr in years:
        a = np.array([sign * v for d, v in ins if d.year != yr])
        b = np.array([sign * v for v in out])
        ts.append(float(sps.ttest_ind(a, b, equal_var=False).statistic) if len(a) > 2 else 0.0)
    res["loo_min_t"] = min(ts)
    share = res["years_positive"] / len(years)
    if share < min_share:
        return dict(res, reason=f"same sign in only {share:.0%} of years")
    if res["loo_min_t"] < 2.0:
        return dict(res, reason="effect depends on a single year (leave-one-year-out t < 2)")
    return dict(res, legitimate=True, reason="recurs across years and survives leave-one-year-out")


def bootstrap_lifetime(profile: TemporalProfile, series: EffectSeries, now, rng: np.random.Generator, n_boot: int = 100,
                       cfg: TemporalConfig | None = None) -> dict[str, Any]:
    """Uncertainty of the class and lifetime by residual bootstrap: refit on the fitted curve plus resampled residuals. Reports
    how often the class is reproduced and quantiles of the lifetime among replicates that classify as a decay. Complements the
    likelihood-grid interval in the profile: a class that flips often under resampling is not settled."""
    if not profile.fit or profile.klass not in (TC.PERSISTENT.value, TC.SLOW_DECAY.value, TC.FAST_DECAY.value):
        raise ValueError("bootstrap_lifetime needs a fitted PERSISTENT or decay profile")
    series.require_valid(now)
    y = np.array(series.values, float)
    pred = np.array([predicted_effect(profile, d) for d in series.dates], float)
    res = y - pred
    classes: dict[str, int] = {}
    lifes = []
    for _ in range(n_boot):
        ys = pred + rng.choice(res, size=len(res), replace=True)
        p = estimate(profile.knowledge_id, EffectSeries(series.dates, tuple(ys), series.ses), now, cfg)
        classes[p.klass] = classes.get(p.klass, 0) + 1
        if p.klass in (TC.SLOW_DECAY.value, TC.FAST_DECAY.value) and p.lifetime_days is not None:
            lifes.append(p.lifetime_days)
    q = {f"q{int(a * 100):02d}": float(np.quantile(lifes, a)) for a in (0.05, 0.5, 0.95)} if len(lifes) >= 5 else {}
    return {"n_boot": n_boot, "class_freq": {k: v / n_boot for k, v in sorted(classes.items())},
            "p_same_class": classes.get(profile.klass, 0) / n_boot, "n_decay": len(lifes), **q}


def failure_context_profile(series: EffectSeries, context: Mapping[str, Sequence[float]], fail_quantile: float = 0.35) -> dict[str, Any]:
    """Separate REGIME-bound from EVENT-bound failure using the market context (m_ columns) AT the failure times. Failure
    periods are the worst `fail_quantile` of oriented effects. A regime failure lines up with a slow, persistent m_ state and
    comes in long runs; an event failure comes in short runs tied to the event flags. UNKNOWN when neither is significant."""
    n = len(series.values)
    y = np.array(series.values, float)
    y = y * (1.0 if y.mean() >= 0 else -1.0)
    fail = y <= np.quantile(y, fail_quantile)
    closed = fail.copy()                                   # noise splits a regime's bad stretch: bridge gaps of up to 2 bins
    idx = np.flatnonzero(fail)
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= 3:
            closed[a:b + 1] = True
    runs = [b - a + 1 for a, b in _runs(closed)]
    mean_run = float(np.mean(runs)) if runs else 0.0
    best: dict[str, Any] = {"feature": None, "p": 1.0, "corr": 0.0, "autocorr": 0.0}
    for name, vals in sorted(context.items()):
        v = np.asarray(vals, float)
        if not name.startswith("m_") or len(v) != n or v.std() == 0:
            continue
        r, p = sps.pointbiserialr(fail.astype(float), v)
        if np.isfinite(p) and p < best["p"]:
            ac = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if n > 3 else 0.0
            best = {"feature": name, "p": float(p), "corr": float(r), "autocorr": ac}
    ev_p = 1.0
    if series.events is not None and fail.any() and (~fail).any():
        ev = np.array(series.events, bool)
        table = [[int((ev & fail).sum()), int((ev & ~fail).sum())], [int((~ev & fail).sum()), int((~ev & ~fail).sum())]]
        ev_p = float(sps.fisher_exact(table)[1])
    if ev_p < 0.01 and ev_p <= best["p"] and mean_run < 2.5:
        verdict = TC.EVENT_BOUND.value
    elif best["p"] < 0.01 and best["autocorr"] >= 0.5 and mean_run >= 2.5:
        verdict = TC.REGIME_BOUND.value
    else:
        verdict = TC.UNKNOWN.value
    return {"verdict": verdict, "mean_failure_run": mean_run, "event_p": ev_p, **{f"ctx_{k}": v for k, v in best.items()}}
