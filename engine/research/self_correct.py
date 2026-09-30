"""Self-correcting prediction network (C68 checklist Q). IMPLEMENTED - NOT VALIDATED.

When prediction accuracy deteriorates the system must find WHICH part failed and test candidate fixes one at a time; a fix reaches
the production learner only through the existing validation and anti-leak architecture (engine.research.quality_gate, which itself
calls engine.learning.promotion). Nothing here promotes anything by itself.

    detect_deterioration   forward-only: is recent absolute error larger than the reference period's? (Welch t on matured rows,
                           plus engine.learning.calibration.change_scan to date the shift). Rows not matured before `now` are a
                           FirewallBreach, never silently used.
    diagnose               one statistical test per checklist-Q component (feature weighting, missing/obsolete feature, pattern
                           interaction, regime recognition, confidence calibration, candidate selection, volatility, direction,
                           timing, exit, market and sector conditioning, over- and under-fitting). A component whose inputs are
                           absent is UNTESTED - unknown stays unknown. Multiplicity across components is controlled with the
                           repo's Benjamini-Hochberg (engine.learning.surprise.benjamini_hochberg).
    test_fixes             each CandidateFix is trained on rows matured before a split date and compared ALONE with the incumbent
                           prediction on later rows (never stacked with another fix), giving per-week OOS improvements, in-sample
                           improvement, search-adjusted statistics and seeded reruns.
    gate_fixes / step      turn each fix's measured evidence into quality_gate evidence (OOS, statistics, reproducibility,
                           point-in-time inputs), merge it with the caller's audits (leak, identity, replication, risk ...) and let
                           QualityGate decide. Missing audits mean NEEDS_MORE_EVIDENCE, never a pass.
    full_fix_evidence      (F11, W-06) every other gate's evidence MEASURED from the fix's own rows: identity harness on the fix's
                           adjustment, the leak audit (evidence.leakage -> firewalls), replication on the stock half it never saw with
                           a shuffled-outcome control, its derived band probability, the book it would select (risk), complexity
                           against the incumbent, transfer across eras, failure behaviour. `step(..., evidence=FixEvidenceConfig())`.
    slope_fix              a fix for a slope that differs in one stock type (one within-group interaction, searched honestly).
    regime_check           (F13, C68 forward-only regimes) always applied by gate_bundles: a fix's weekly OOS gains must come from the
                           regime in force at `now` (change points from a forward CUSUM on those gains and from the caller's
                           detectors), or be shown to hold across the change; it only ever makes a verdict stricter.
    revalidate             (F13) a change declared after a promotion re-opens it: re-validate on post-change weeks or roll back
                           within `reval_max_weeks`; rollback_delay_curve measures the delay distribution by simulation.

Frame contract (one row per matured prediction; research side): `date` decision date, `matured_at` outcome date, `predicted`,
`realised`; optional f_* model features, u_* available-but-unused features, m_* market context, `regime`, `sector`, `selected`,
`pred_vol`/`real_vol`, `p_up`, `pred_peak_day`/`real_peak_day`, `pred_exit`/`real_exit`, `in_sample_pred`, `p_in_band`/`in_band`,
`q_lo`/`q_hi` (a stated central interval)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.learning import calibration as CAL
from engine.learning import complexity as CX
from engine.learning import promotion as PR
from engine.learning.core import FirewallBreach, Provenance, _StrEnum, as_date, stable_hash
from engine.learning.surprise import benjamini_hochberg
from engine.research import change_points as CP
from engine.research import quality_gate as QG
from engine.research.core import Availability, GateVerdict

SECTION = "C68 checklist Q"
REQUIRED = ("date", "matured_at", "predicted", "realised")


class Component(_StrEnum):
    FEATURE_WEIGHTING = "feature_weighting"
    MISSING_FEATURE = "missing_feature"
    OBSOLETE_FEATURE = "obsolete_feature"
    PATTERN_INTERACTION = "pattern_interaction"
    REGIME_RECOGNITION = "regime_recognition"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    CANDIDATE_SELECTION = "candidate_selection"
    VOLATILITY_ESTIMATION = "volatility_estimation"
    DIRECTION_ESTIMATION = "direction_estimation"
    TIMING = "timing"
    EXIT_ESTIMATION = "exit_estimation"
    MARKET_CONDITIONING = "market_level_conditioning"
    SECTOR_CONDITIONING = "sector_conditioning"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"


class FindingState(_StrEnum):
    IMPLICATED = "IMPLICATED"
    NOT_IMPLICATED = "NOT_IMPLICATED"
    UNTESTED = "UNTESTED"                 # inputs absent or too few rows: we do not know


@dataclass(frozen=True)
class SelfCorrectConfig:
    recent_share: float = 0.3             # most recent share of matured rows compared with the older reference
    min_rows: int = 40                    # per side, below which a component is UNTESTED
    alpha: float = 0.05
    fdr_q: float = 0.10
    ridge: float = 1.0
    max_pairs: int = 15
    split_share: float = 0.4              # fixes train on the earliest share of the timeline, then are tested on the rest
    label_horizon_days: int = 7
    reruns: tuple[int, ...] = (1, 1, 2)   # seeds: one repeated (determinism) and a second seed (sensitivity)

    def validate(self) -> list[str]:
        errs = []
        if not 0.1 <= self.recent_share <= 0.5:
            errs.append("recent_share must be in [0.1, 0.5]")
        if self.min_rows < 20:
            errs.append("min_rows must be >= 20")
        if not 0 < self.alpha < 0.5 or not 0 < self.fdr_q < 0.5:
            errs.append("alpha and fdr_q must be in (0, 0.5)")
        if not 0.2 <= self.split_share <= 0.8:
            errs.append("split_share must be in [0.2, 0.8]")
        if len(set(self.reruns)) < 2 or len(self.reruns) == len(set(self.reruns)):
            errs.append("reruns need two distinct seeds and one repeated seed")
        return errs


def as_of(frame: pd.DataFrame, now) -> pd.DataFrame:
    """Rows whose outcome matured strictly before `now` (the only rows research may study at `now`), in maturity order."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(REQUIRED))
    m = pd.to_datetime(frame["matured_at"]).dt.date < as_date(now)
    return frame[m.to_numpy()].sort_values(["matured_at", "date"], kind="stable").reset_index(drop=True)


def _check(frame: pd.DataFrame, now) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"error frame missing columns {missing}")
    if len(frame):
        late = pd.to_datetime(frame["matured_at"]).dt.date >= as_date(now)
        if late.any():
            raise FirewallBreach(f"{int(late.sum())} row(s) matured at/after now={now}: filter with as_of(frame, now) first")
        bad = pd.to_datetime(frame["matured_at"]) < pd.to_datetime(frame["date"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} row(s) matured before their decision date")
    return frame.sort_values(["matured_at", "date"], kind="stable").reset_index(drop=True)


def _split(frame: pd.DataFrame, share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = int(round(len(frame) * (1 - share)))
    return frame.iloc[:k], frame.iloc[k:]


def _cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(c for c in frame.columns if str(c).startswith(prefix))


# ------------------------------------------------------------------------------------------------ deterioration
@dataclass(frozen=True)
class Deterioration:
    detected: bool
    n_reference: int
    n_recent: int
    reference_mae: float
    recent_mae: float
    t: float
    p: float
    scan_stat: float
    change_at: str | None


def detect_deterioration(frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> Deterioration:
    """Forward-only: the decision at `now` depends on matured rows only (a test scrambles everything after `now` and gets the same
    answer). Detected = recent absolute error significantly above the reference period's."""
    f = _check(frame, now)
    if len(f) < 2 * cfg.min_rows:
        return Deterioration(False, 0, len(f), float("nan"), float("nan"), float("nan"), float("nan"), 0.0, None)
    e = (f["realised"] - f["predicted"]).abs().to_numpy(float)
    ref, rec = _split(f, cfg.recent_share)
    a, b = e[: len(ref)], e[len(ref):]
    t, p2 = sps.ttest_ind(b, a, equal_var=False)
    p = float(p2 / 2 if t > 0 else 1 - p2 / 2)
    z = (e - a.mean()) / (a.std() or 1.0)
    stat, idx = CAL.change_scan(z)
    at = str(as_date(f["matured_at"].iloc[idx])) if idx is not None else None
    return Deterioration(bool(p < cfg.alpha), len(a), len(b), float(a.mean()), float(b.mean()), float(t), p, float(stat), at)


# ------------------------------------------------------------------------------------------------ component tests
@dataclass(frozen=True)
class Finding:
    component: Component
    state: FindingState
    p_value: float
    effect: float
    detail: str
    measures: Mapping[str, Any] = field(default_factory=dict)


def _untested(c: Component, why: str) -> Finding:
    return Finding(c, FindingState.UNTESTED, float("nan"), float("nan"), why)


def _res(c: Component, p: float, effect: float, detail: str, alpha: float, **m) -> Finding:
    st = FindingState.IMPLICATED if math.isfinite(p) and p < alpha else FindingState.NOT_IMPLICATED
    return Finding(c, st, float(p), float(effect), detail, m)


def _ols_ssr(X: np.ndarray, y: np.ndarray) -> float:
    A = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return float(r @ r)


def _corr_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r, p = sps.pearsonr(x, y)
    return float(r), float(p)


def _max_corr(resid: np.ndarray, frame: pd.DataFrame, cols: Sequence[str]) -> tuple[str | None, float, float]:
    best, br, bp = None, 0.0, 1.0
    for c in cols:
        r, p = _corr_p(frame[c].to_numpy(float), resid)
        if math.isfinite(p) and (best is None or p < bp):
            best, br, bp = c, r, p
    return best, br, min(1.0, bp * max(len(cols), 1))          # Bonferroni across the columns searched


def t_feature_weighting(ref, rec, cfg) -> Finding:
    """Chow test: do the realised-on-feature coefficients differ between the reference and recent periods?"""
    C = Component.FEATURE_WEIGHTING
    cols = _cols(rec, "f_")
    if not cols:
        return _untested(C, "no f_* model features supplied")
    Xa, Xb = ref[cols].to_numpy(float), rec[cols].to_numpy(float)
    ya, yb = ref["realised"].to_numpy(float), rec["realised"].to_numpy(float)
    k = len(cols) + 1
    ssr_p = _ols_ssr(np.vstack([Xa, Xb]), np.concatenate([ya, yb]))
    ssr_s = _ols_ssr(Xa, ya) + _ols_ssr(Xb, yb)
    dfd = len(ya) + len(yb) - 2 * k
    if dfd <= 0 or ssr_s <= 0:
        return _untested(C, "too few rows for a Chow test")
    F = ((ssr_p - ssr_s) / k) / (ssr_s / dfd)
    return _res(C, float(sps.f.sf(F, k, dfd)), float(F), f"Chow F={F:.2f} on {len(cols)} features", cfg.alpha, F=F)


def t_missing_feature(ref, rec, cfg) -> Finding:
    C = Component.MISSING_FEATURE
    cols = _cols(rec, "u_")
    if not cols:
        return _untested(C, "no u_* candidate features supplied")
    resid = (rec["realised"] - rec["predicted"]).to_numpy(float)
    best, r, p = _max_corr(resid, rec, cols)
    return _res(C, p, r, f"recent error correlates {r:+.2f} with unused {best}", cfg.alpha, feature=best)


def t_obsolete_feature(ref, rec, cfg) -> Finding:
    """A feature that predicted outcomes in the reference period but whose slope has collapsed recently."""
    C = Component.OBSOLETE_FEATURE
    cols = _cols(rec, "f_")
    if not cols:
        return _untested(C, "no f_* model features supplied")
    best, bp, beff = None, 1.0, 0.0
    for c in cols:
        ra, pa = _corr_p(ref[c].to_numpy(float), ref["realised"].to_numpy(float))
        rb, _ = _corr_p(rec[c].to_numpy(float), rec["realised"].to_numpy(float))
        if not (math.isfinite(pa) and math.isfinite(rb)) or pa >= cfg.alpha:
            continue
        za, zb = np.arctanh(np.clip([ra, rb], -0.999, 0.999))
        se = math.sqrt(1 / (len(ref) - 3) + 1 / (len(rec) - 3))
        zd = (abs(za) - abs(zb) * np.sign(za * zb)) / se            # shrinkage of the effect in its original direction
        p = float(sps.norm.sf(zd))
        if p < bp:
            best, bp, beff = c, p, float(ra - rb)
    if best is None:
        return Finding(C, FindingState.NOT_IMPLICATED, 1.0, 0.0, "no feature was predictive in the reference period")
    p = min(1.0, bp * len(cols))
    return _res(C, p, beff, f"{best} lost {beff:+.2f} of correlation with outcomes", cfg.alpha, feature=best)


def t_pattern_interaction(ref, rec, cfg) -> Finding:
    C = Component.PATTERN_INTERACTION
    cols = _cols(rec, "f_")
    if len(cols) < 2:
        return _untested(C, "fewer than two f_* features: no interaction to test")
    resid = (rec["realised"] - rec["predicted"]).to_numpy(float)
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]][: cfg.max_pairs]
    prod = pd.DataFrame({f"{a}*{b}": rec[a].to_numpy(float) * rec[b].to_numpy(float) for a, b in pairs})
    best, r, p = _max_corr(resid, prod, list(prod.columns))
    return _res(C, p, r, f"recent error correlates {r:+.2f} with interaction {best}", cfg.alpha, interaction=best)


def _anova(resid: np.ndarray, groups: pd.Series, min_n: int = 8) -> tuple[float, float, dict]:
    parts = {str(g): resid[(groups == g).to_numpy()] for g in groups.dropna().unique()}
    parts = {g: v for g, v in parts.items() if len(v) >= min_n}
    if len(parts) < 2:
        return float("nan"), float("nan"), {}
    F, p = sps.f_oneway(*parts.values())
    return float(F), float(p), {g: float(v.mean()) for g, v in parts.items()}


def t_regime(ref, rec, cfg) -> Finding:
    C = Component.REGIME_RECOGNITION
    if "regime" not in rec.columns:
        return _untested(C, "no regime label supplied")
    F, p, means = _anova((rec["realised"] - rec["predicted"]).to_numpy(float), rec["regime"])
    if not math.isfinite(p):
        return _untested(C, "fewer than two regimes with enough rows")
    return _res(C, p, F, f"recent error differs by regime (F={F:.2f}): {means}", cfg.alpha, means=means)


def t_sector(ref, rec, cfg) -> Finding:
    C = Component.SECTOR_CONDITIONING
    if "sector" not in rec.columns:
        return _untested(C, "no sector label supplied")
    F, p, means = _anova((rec["realised"] - rec["predicted"]).to_numpy(float), rec["sector"])
    if not math.isfinite(p):
        return _untested(C, "fewer than two sectors with enough rows")
    return _res(C, p, F, f"recent error differs by sector (F={F:.2f})", cfg.alpha, means=means)


def t_market(ref, rec, cfg) -> Finding:
    C = Component.MARKET_CONDITIONING
    cols = _cols(rec, "m_")
    if not cols:
        return _untested(C, "no m_* market context supplied")
    best, r, p = _max_corr((rec["realised"] - rec["predicted"]).to_numpy(float), rec, cols)
    return _res(C, p, r, f"recent error correlates {r:+.2f} with market context {best}", cfg.alpha, context=best)


def t_confidence(ref, rec, cfg) -> Finding:
    """Band probabilities: Platt slope below 1 = overconfident. Stated interval: coverage vs its nominal share (binomial)."""
    C = Component.CONFIDENCE_CALIBRATION
    if {"p_in_band", "in_band"} <= set(rec.columns):
        s = CAL.platt_slope(rec["p_in_band"].to_numpy(float), rec["in_band"].to_numpy(float))
        if not math.isfinite(s["b"]):
            return _untested(C, "calibration slope not estimable")
        z = (s["b"] - 1.0) / (s["se_b"] or float("inf"))
        return _res(C, float(2 * sps.norm.sf(abs(z))), s["b"], f"Platt slope {s['b']:.2f} (1 = calibrated)", cfg.alpha, slope=s["b"])
    if {"q_lo", "q_hi"} <= set(rec.columns):
        nominal = float(rec.attrs.get("interval_level", 0.8))
        inside = ((rec["realised"] >= rec["q_lo"]) & (rec["realised"] <= rec["q_hi"])).to_numpy()
        p = float(sps.binomtest(int(inside.sum()), len(inside), nominal).pvalue)
        return _res(C, p, float(inside.mean() - nominal), f"interval covers {inside.mean():.0%} vs nominal {nominal:.0%}", cfg.alpha)
    return _untested(C, "no band probabilities or stated intervals supplied")


def t_selection(ref, rec, cfg) -> Finding:
    """Winner's curse: selected names' outcomes fall short of their predictions more than the non-selected names' do."""
    C = Component.CANDIDATE_SELECTION
    if "selected" not in rec.columns:
        return _untested(C, "no selected flag supplied")
    s = rec["selected"].astype(bool).to_numpy()
    r = (rec["realised"] - rec["predicted"]).to_numpy(float)
    if s.sum() < 8 or (~s).sum() < 8:
        return _untested(C, "too few selected or non-selected rows")
    t, p2 = sps.ttest_ind(r[s], r[~s], equal_var=False)
    p = float(p2 / 2 if t < 0 else 1 - p2 / 2)
    return _res(C, p, float(r[s].mean() - r[~s].mean()), f"selected shortfall {r[s].mean() - r[~s].mean():+.4f} vs non-selected", cfg.alpha)


def _worse(ref, rec, a: str, b: str, C: Component, cfg, transform=None) -> Finding:
    if not {a, b} <= set(rec.columns):
        return _untested(C, f"needs {a} and {b}")
    f = transform or (lambda x, y: np.abs(x - y))
    ea = f(ref[a].to_numpy(float), ref[b].to_numpy(float))
    eb = f(rec[a].to_numpy(float), rec[b].to_numpy(float))
    ea, eb = ea[np.isfinite(ea)], eb[np.isfinite(eb)]
    if len(ea) < 8 or len(eb) < 8:
        return _untested(C, "too few rows")
    t, p2 = sps.ttest_ind(eb, ea, equal_var=False)
    p = float(p2 / 2 if t > 0 else 1 - p2 / 2)
    return _res(C, p, float(eb.mean() - ea.mean()), f"error {ea.mean():.4f} -> {eb.mean():.4f}", cfg.alpha)


def t_volatility(ref, rec, cfg) -> Finding:
    return _worse(ref, rec, "pred_vol", "real_vol", Component.VOLATILITY_ESTIMATION, cfg,
                  lambda x, y: np.abs(np.log(np.maximum(y, 1e-9) / np.maximum(x, 1e-9))))


def t_direction(ref, rec, cfg) -> Finding:
    C = Component.DIRECTION_ESTIMATION
    if "p_up" not in rec.columns:
        return _untested(C, "no p_up supplied")
    ha = ((ref["p_up"] >= 0.5) == (ref["realised"] > 0)).to_numpy()
    hb = ((rec["p_up"] >= 0.5) == (rec["realised"] > 0)).to_numpy()
    pa, pb = ha.mean(), hb.mean()
    pp = (ha.sum() + hb.sum()) / (len(ha) + len(hb))
    se = math.sqrt(max(pp * (1 - pp) * (1 / len(ha) + 1 / len(hb)), 1e-12))
    z = (pa - pb) / se
    return _res(C, float(sps.norm.sf(z)), float(pb - pa), f"direction hit rate {pa:.1%} -> {pb:.1%}", cfg.alpha)


def t_timing(ref, rec, cfg) -> Finding:
    return _worse(ref, rec, "pred_peak_day", "real_peak_day", Component.TIMING, cfg)


def t_exit(ref, rec, cfg) -> Finding:
    return _worse(ref, rec, "pred_exit", "real_exit", Component.EXIT_ESTIMATION, cfg)


def t_overfitting(ref, rec, cfg) -> Finding:
    """In-sample fit far better than recent out-of-sample error on the same rows' model."""
    C = Component.OVERFITTING
    if "in_sample_pred" not in rec.columns:
        return _untested(C, "no in_sample_pred supplied")
    ins = (rec["realised"] - rec["in_sample_pred"]).abs().to_numpy(float)
    oos = (rec["realised"] - rec["predicted"]).abs().to_numpy(float)
    d = oos - ins
    t = PR.t_stat(d)
    p = float(sps.norm.sf(t)) if math.isfinite(t) else float("nan")
    ratio = float(oos.mean() / max(ins.mean(), 1e-12))
    return _res(C, p, ratio, f"OOS/in-sample error ratio {ratio:.2f}", cfg.alpha, ratio=ratio)


def t_underfitting(ref, rec, cfg) -> Finding:
    """Recent error is predictable from NONLINEAR transforms of the model's own features: structure the model cannot express."""
    C = Component.UNDERFITTING
    cols = _cols(rec, "f_")
    if not cols:
        return _untested(C, "no f_* model features supplied")
    y = (rec["realised"] - rec["predicted"]).to_numpy(float)
    X = np.hstack([rec[cols].to_numpy(float) ** 2, np.abs(rec[cols].to_numpy(float))])
    ssr0 = float(((y - y.mean()) ** 2).sum())
    ssr1 = _ols_ssr(X, y)
    k, dfd = X.shape[1], len(y) - X.shape[1] - 1
    if dfd <= 0 or ssr1 <= 0:
        return _untested(C, "too few rows")
    F = ((ssr0 - ssr1) / k) / (ssr1 / dfd)
    return _res(C, float(sps.f.sf(F, k, dfd)), float(1 - ssr1 / ssr0), f"nonlinear terms explain {1 - ssr1 / ssr0:.1%} of recent error", cfg.alpha)


TESTS: dict[Component, Callable[[pd.DataFrame, pd.DataFrame, SelfCorrectConfig], Finding]] = {
    Component.FEATURE_WEIGHTING: t_feature_weighting, Component.MISSING_FEATURE: t_missing_feature,
    Component.OBSOLETE_FEATURE: t_obsolete_feature, Component.PATTERN_INTERACTION: t_pattern_interaction,
    Component.REGIME_RECOGNITION: t_regime, Component.CONFIDENCE_CALIBRATION: t_confidence,
    Component.CANDIDATE_SELECTION: t_selection, Component.VOLATILITY_ESTIMATION: t_volatility,
    Component.DIRECTION_ESTIMATION: t_direction, Component.TIMING: t_timing, Component.EXIT_ESTIMATION: t_exit,
    Component.MARKET_CONDITIONING: t_market, Component.SECTOR_CONDITIONING: t_sector, Component.OVERFITTING: t_overfitting,
    Component.UNDERFITTING: t_underfitting,
}


@dataclass(frozen=True)
class Diagnosis:
    now: str
    deterioration: Deterioration
    findings: tuple[Finding, ...]
    implicated: tuple[Component, ...]      # after FDR control, most significant first
    untested: tuple[Component, ...]

    def finding(self, c: Component) -> Finding:
        return next(f for f in self.findings if f.component == c)


def diagnose(frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> Diagnosis:
    """Run every component test on matured rows. A test that crashes is UNTESTED with the error named (never a silent pass)."""
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid SelfCorrectConfig: " + "; ".join(errs))
    f = _check(frame, now)
    det = detect_deterioration(f, now, cfg)
    if len(f) < 2 * cfg.min_rows:
        fs = tuple(_untested(c, f"only {len(f)} matured rows") for c in Component)
        return Diagnosis(str(as_date(now)), det, fs, (), tuple(Component))
    ref, rec = _split(f, cfg.recent_share)
    out = []
    for c, fn in TESTS.items():
        try:
            out.append(fn(ref, rec, cfg))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            out.append(_untested(c, f"test failed: {type(e).__name__}: {e}"))
    tested = [x for x in out if x.state != FindingState.UNTESTED]
    keep = benjamini_hochberg([x.p_value for x in tested], cfg.fdr_q)
    sig = {x.component for x, k in zip(tested, keep) if k and x.state == FindingState.IMPLICATED}
    final = tuple(x if x.state == FindingState.UNTESTED or x.component in sig else dataclasses.replace(x, state=FindingState.NOT_IMPLICATED)
                  for x in out)
    order = tuple(sorted(sig, key=lambda c: next(x.p_value for x in final if x.component == c)))
    return Diagnosis(str(as_date(now)), det, final, order, tuple(x.component for x in final if x.state == FindingState.UNTESTED))


# ------------------------------------------------------------------------------------------------ candidate fixes
@dataclass(frozen=True)
class FixInput:
    name: str
    available_at_lag_days: int = 0        # 0 = known at the decision's prior close; negative = known only AFTER the decision
    availability: Availability = Availability.KNOWN_BEFORE_EVENT


@dataclass(frozen=True)
class CandidateFix:
    """One proposed change, tested ALONE. `build(train, seed)` returns a predictor `frame -> predicted realisable gain`; a predictor
    may carry `.detail` (a mapping of what its own training chose, e.g. the interaction it picked). `spec` is the structure the fix
    ADDS to the incumbent (the complexity gate prices it); `n_variants` is how many variants the fix's own search looked at (a fix
    that picks the best of 20 interactions pays for 20 in the multiplicity correction)."""
    name: str
    component: Component
    inputs: tuple[FixInput, ...]
    build: Callable[[pd.DataFrame, int], Callable[[pd.DataFrame], np.ndarray]]
    spec: CX.RuleSpec | None = None
    n_variants: int = 1

    def rule_spec(self) -> CX.RuleSpec:
        return self.spec or CX.RuleSpec(f"fix_{self.name}", n_features=len(self.inputs), n_free_params=len(self.inputs))


@dataclass(frozen=True)
class FixResult:
    fix: str
    component: Component
    split: str
    train_years: tuple[int, ...]
    n_train: int
    n_oos: int
    oos_periods: tuple[str, ...]
    oos_effects: tuple[float, ...]         # per week: incumbent |error| - fix |error| (positive = the fix helps)
    in_sample_effect: float
    mean_effect: float
    t: float
    reruns: tuple[tuple[int, float], ...]
    data_digest: str
    detail: Mapping[str, Any] = field(default_factory=dict)            # what the fix's own training chose (its `.detail`)
    panel: Any = field(default=None, compare=False, repr=False)       # per-row train/test predictions: the full evidence reads it


def _abs_err(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.abs(frame["realised"].to_numpy(float) - np.asarray(pred, float))


def split_frame(frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """(matured frame, train, test, split): train = rows MATURED before the split date, test = rows DECIDED after split + the label
    horizon (the embargo leaves out rows whose outcome straddles the split)."""
    f = _check(frame, now)
    if len(f) < 2 * cfg.min_rows:
        raise ValueError(f"only {len(f)} matured rows: cannot test a fix")
    dates = pd.to_datetime(f["date"])
    split = dates.quantile(cfg.split_share)
    train = f[(pd.to_datetime(f["matured_at"]) < split).to_numpy()].reset_index(drop=True)
    test = f[(dates > split + pd.Timedelta(days=cfg.label_horizon_days)).to_numpy()].reset_index(drop=True)
    if len(train) < cfg.min_rows or len(test) < cfg.min_rows:
        raise ValueError("split leaves too few training or test rows")
    return f, train, test, split


def weekly(P: pd.DataFrame, col: str) -> pd.Series:
    """Per decision week (W-FRI), the mean of `col`, labelled by the week's LAST maturity date: a week's evidence exists once its last
    outcome has matured - labelling it by the calendar week's end could date it after `now` (a Friday not yet reached), which the
    gate's provenance check rightly refuses (found wiring P06 into the loop)."""
    if not len(P):
        return pd.Series(dtype=float)
    wk = pd.to_datetime(P["date"]).dt.to_period("W-FRI").astype(str).to_numpy()
    g = pd.DataFrame({"v": P[col].to_numpy(float), "m": pd.to_datetime(P["matured_at"]).to_numpy(), "w": wk}).groupby("w", sort=True)
    s = pd.Series(g["v"].mean().to_numpy(float), index=pd.DatetimeIndex(g["m"].max().to_numpy()))
    return s.groupby(level=0).mean().sort_index()


def _panel(train: pd.DataFrame, p_train: np.ndarray, test: pd.DataFrame, p_test: np.ndarray) -> pd.DataFrame:
    keep = [c for c in ("date", "matured_at", "ticker", "sector", "m_vol") if c in train.columns]
    out = []
    for part, fr, p in (("train", train, p_train), ("test", test, p_test)):
        d = fr[keep].copy()
        d["realised"] = fr["realised"].to_numpy(float)
        d["incumbent"] = fr["predicted"].to_numpy(float)
        d["fix"] = np.asarray(p, float)
        d["gain"] = np.abs(d["realised"] - d["incumbent"]) - np.abs(d["realised"] - d["fix"])
        d["part"] = part
        out.append(d)
    return pd.concat(out, ignore_index=True)


def test_fix(fix: CandidateFix, frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> FixResult:
    """Train on rows matured before the split date, test on rows DECIDED after it (the embargo leaves out rows whose outcome
    straddles the split). The incumbent is the recorded `predicted` column; the fix is compared with it and nothing else."""
    f, train, test, split = split_frame(frame, now, cfg)
    dg = stable_hash({"n": len(f), "first": str(f["date"].iloc[0]), "last": str(f["matured_at"].iloc[-1]),
                      "sum": round(float(f["realised"].sum()), 10)})
    reruns = []
    panel, detail, ins = None, {}, float("nan")
    for seed in cfg.reruns:
        pred = fix.build(train, seed)
        p_test = np.asarray(pred(test), float)
        gain = _abs_err(test["predicted"].to_numpy(float), test) - _abs_err(p_test, test)
        reruns.append((int(seed), float(gain.mean())))
        if panel is None:
            p_train = np.asarray(pred(train), float)
            ins = float((_abs_err(train["predicted"].to_numpy(float), train) - _abs_err(p_train, train)).mean())
            panel = _panel(train, p_train, test, p_test)
            detail = dict(getattr(pred, "detail", None) or {})
    eff = weekly(panel[panel["part"] == "test"], "gain")
    e = eff.to_numpy(float)
    per = tuple(str(d.date()) for d in eff.index)
    years = tuple(sorted({int(y) for y in pd.to_datetime(train["date"]).dt.year}))
    return FixResult(fix.name, fix.component, str(split.date()), years, len(train), len(test), per, tuple(float(x) for x in e), ins,
                     float(e.mean()), float(PR.t_stat(e)), tuple(reruns), dg, detail, panel)


def test_fixes(fixes: Sequence[CandidateFix], frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> list[FixResult]:
    """Every fix independently against the SAME incumbent on the SAME split - never cumulatively, so one fix's gain is not credited
    to another and an interaction of two fixes is a separate candidate if wanted."""
    names = [x.name for x in fixes]
    if len(set(names)) != len(names):
        raise ValueError("fix names must be unique")
    return [test_fix(x, frame, now, cfg) for x in fixes]


def fix_evidence(r: FixResult, fix: CandidateFix, n_searched: int, now, base: QG.QualityEvidence | None, code_hash: str) -> QG.QualityEvidence:
    """The measured parts of a fix's case (OOS, statistics, reproducibility, point-in-time inputs), merged into the caller's audit
    bundle. Audits this module cannot run (leak, identity, replication, risk, transfer ...) come only from `base`; absent = MISSING."""
    split = as_date(r.split)
    decision = split + pd.Timedelta(days=1).to_pytimedelta()
    # lag 0 = known at the prior close (one day before the decision); a negative lag = known only after the decision
    feats = tuple(QG.FeatureUse(i.name, str(decision - pd.Timedelta(days=i.available_at_lag_days + 1).to_pytimedelta())
                                if i.availability == Availability.KNOWN_BEFORE_EVENT else None, i.availability) for i in fix.inputs)
    first_test = min(r.oos_periods) if r.oos_periods else str(split)
    pit = QG.PITEvidence(feats, str(decision), 7, str(split), first_test, True, str(split))
    t = r.t
    stat = PR.StatisticalEvidence(r.mean_effect, r.n_oos, t, None, max(1, n_searched), float(len(r.oos_effects)))
    oos = QG.OOSBundle(stat, PR.OOSEvidence(str(split), r.oos_periods, r.oos_effects, r.in_sample_effect), r.train_years)
    repro = PR.ReproEvidence(tuple(PR.RerunRecord(v, s, code_hash, r.data_digest) for s, v in r.reruns), r.data_digest)
    last = max(r.oos_periods) if r.oos_periods else str(split)
    b = base or QG.QualityEvidence()
    prov = b.provenance
    if prov is not None:
        prov = dataclasses.replace(prov, learned_at=last, outcomes_seen_through=last, code_hash=code_hash, data_hash=r.data_digest)
    return dataclasses.replace(b, pit=pit, oos=oos, repro=repro, justification=("oos_transfer",) + tuple(x for x in b.justification if x != "oos_transfer"),
                               provenance=prov)


# ------------------------------------------------------------------------------------------------ the FULL evidence of a fix (W-06)
# P06 found that no fix could ever reach PROMOTE on its own evidence: fix_evidence supplies OOS, statistics, reproducibility and PIT,
# and every other gate stayed MISSING (NEEDS_MORE_EVIDENCE) unless a caller forged it. This section MEASURES the rest from the fix's
# own train/test rows, with the modules that own each kind of evidence (identity_firewall, evidence.leakage -> firewalls,
# replication, calibration, complexity, promotion) - the fix-shaped counterpart of engine.research.evidence.assemble, which does the
# same for a derived-feature ranking finding. Nothing is asserted: a part that cannot be computed stays None (MISSING) and is named.
@dataclass(frozen=True)
class FixEvidenceConfig:
    lo: float = 0.05                      # the band a point forecast selects into (checklist L): risk and calibration read it
    hi: float = 0.10
    era: str = "half"                     # transfer contexts and complexity folds: calendar 'half' years or whole 'year's
    identity_boot: int = 100
    repl_windows: int = 2                 # replication windows cut from the test weeks, on the stock half the discovery never saw
    resid_draws: int = 300                # train residuals behind a fix's P(in band)
    perturb_sd: float = 0.25              # planted input degradation, in each input's own sd
    catastrophic: float = -0.20
    noise_alpha: float = 0.05             # family-wise level of the 'sampling noise' explanation of a failure week
    screen_t: float = 1.64                # evidence ladder (C66 section 18): only a fix whose OOS screen passes gets the costly audits
    retirement_trigger: bool = False      # the CALLER monitors a promoted fix and rolls it back (declared by the loop that does it)
    states_probabilities: bool = False    # True = the fix's own P(in band) is judged by the calibration gate. A point-forecast fix
                                          # states no probability (the band probability stays the gain model's, from its residuals);
                                          # its derived P(in band) is still measured and reported (parts: brier_fix vs incumbent)
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if not 0 <= self.lo < self.hi or self.era not in ("half", "year"):
            errs.append("band lo < hi and era in {half, year} required")
        if self.repl_windows < 2 or self.resid_draws < 50 or self.perturb_sd <= 0 or self.identity_boot < 20:
            errs.append("repl_windows >= 2, resid_draws >= 50, perturb_sd > 0, identity_boot >= 20 required")
        return errs


@dataclass
class FixBundle:
    fix: str
    evidence: QG.QualityEvidence
    parts: dict                           # the numbers behind each part (reports and tests read them)
    missing: dict                         # evidence kind -> why it could not be computed (left None: the gate blocks)
    full: bool = True                     # False = the cheap bundle (the OOS screen failed; the gate fails it on OOS)


def era_of(d, era: str = "half") -> str:
    ts = pd.Timestamp(d)
    return f"{ts.year}H{1 if ts.month <= 6 else 2}" if era == "half" else str(ts.year)


def screen_passes(r: FixResult, ec: FixEvidenceConfig) -> bool:
    return bool(r.mean_effect > 0 and math.isfinite(r.t) and r.t >= ec.screen_t and len(r.oos_effects) >= 2)


def _panel_X(fr: pd.DataFrame) -> pd.DataFrame:
    """The fix's inputs as an identity-firewall panel: (date, ticker) index, every column the fix may read (never the outcome)."""
    cols = [c for c in fr.columns if c not in ("date", "ticker", "realised", "matured_at")]
    X = fr[cols].copy()
    X.index = pd.MultiIndex.from_arrays([pd.to_datetime(fr["date"]).to_numpy(), fr["ticker"].astype(str).to_numpy()], names=["date", "ticker"])
    return X


def _frame_from_X(X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
    fr = X.reset_index()
    fr["date"] = pd.to_datetime(fr["date"]).dt.strftime("%Y-%m-%d")
    fr["ticker"] = fr["ticker"].astype(str)
    if y is not None:
        fr["realised"] = y.to_numpy(float)
    return fr


def fix_identity(fix: CandidateFix, train: pd.DataFrame, test: pd.DataFrame, P: pd.DataFrame, ec: FixEvidenceConfig):
    """engine.learning.identity_firewall.IdentityHarness on the fix's OWN contribution: the score is the fix's adjustment to the
    incumbent and the target the incumbent's error, so the incumbent's skill cannot hide a fix that memorises names (a per-name
    correction collapses once the names are permuted). Plus the per-name share of the out-of-sample gain."""
    from engine.learning import identity_firewall as IDF
    Xt, Xe = _panel_X(train), _panel_X(test)
    yt = pd.Series((train["realised"] - train["predicted"]).to_numpy(float), index=Xt.index)
    ye = pd.Series((test["realised"] - test["predicted"]).to_numpy(float), index=Xe.index)

    def learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
        tr = _frame_from_X(X_train, None)
        tr["realised"] = tr["predicted"].to_numpy(float) + y_train.to_numpy(float)
        ev = _frame_from_X(X_eval, None)
        adj = np.asarray(fix.build(tr, ec.seed)(ev), float) - ev["predicted"].to_numpy(float)
        return pd.Series(adj, index=X_eval.index)
    rep = IDF.IdentityHarness(learner, attacks=("ticker_permutation", "date_permutation", "stock_substitution"), seed=ec.seed + 11,
                              boot=ec.identity_boot).run(Xt, yt, Xe, ye)
    ret = rep.retention_by_kind("eval")
    vals = [float(v) for v in ret.values if np.isfinite(v)]
    shuf = ret.get("ticker_permutation", np.nan)
    T = P[P["part"] == "test"]
    contrib = T.groupby("ticker")["gain"].sum()
    pos = float(contrib[contrib > 0].sum())
    top = float(contrib.max() / pos) if pos > 0 else 1.0
    adj = T["fix"] - T["incumbent"]
    tot = float(adj.var(ddof=0))
    between = float(adj.groupby(T["ticker"]).transform("mean").var(ddof=0))
    lookup = bool(tot > 1e-18 and between / tot >= 0.9)                  # the adjustment is (almost) a per-name constant
    memo = PR.MemorizationEvidence(float(shuf) if np.isfinite(shuf) else None, float(min(vals)) if vals else None,
                                   tuple(i.name for i in fix.inputs), lookup, top, int(contrib.size))
    return rep, memo, {"identity_retention_min": min(vals) if vals else None, "shuffle_retention": float(shuf) if np.isfinite(shuf) else None,
                       "top_identity_share": top, "lookup_share": between / tot if tot > 1e-18 else 0.0, "base_ic": float(rep.base_ic)}


def fix_leakage(fix: CandidateFix, P: pd.DataFrame, now, split, first_test, rep, code_hash: str, prov: Provenance, n_searched: int, seed: int):
    """engine.research.evidence.leakage on the fix: the learning firewall on the fix's own context, the firewall's planted corpus, a
    planted label copy the leak screen must flag, and the count of outcomes at/after now. What is audited is the fix's OWN content:
    its adjustment to the incumbent as the feature and the incumbent's error as the label (a fix that reads the outcome reproduces
    that error exactly and is flagged); the incumbent is already in production and is not what this gate decides. The test period
    starts at the first test DECISION (a label of the training window reaching past it would be an unpurged overlap)."""
    from engine.research import evidence as EV
    G = pd.DataFrame({"close": (P["realised"] - P["incumbent"]).to_numpy(float), "end": pd.to_datetime(P["matured_at"]).to_numpy()},
                     index=pd.MultiIndex.from_arrays([pd.to_datetime(P["date"]).to_numpy(), P["ticker"].astype(str).to_numpy()]))
    score = pd.Series((P["fix"] - P["incumbent"]).to_numpy(float), index=G.index)
    spec = EV.FindingSpec(f"fix:{fix.name}", f"fix_{fix.name}", 1.0, "VOLATILITY", max(1, n_searched), False, f"self_correct.{fix.name}",
                          prov.run_id or "research", seed)
    return EV.leakage(G, score, G["close"], spec, now, split, first_test, rep, code_hash, prov, int((P["part"] == "test").sum()))


def fix_replication(fix: CandidateFix, P: pd.DataFrame, control: np.ndarray, now, code_hash: str, data_hash: str, n_searched: int,
                    ec: FixEvidenceConfig):
    """engine.research.replication: the Discovery is the fix's weekly gain on one half of the names in its training window; the runs
    are its weekly gains on the OTHER half in `repl_windows` later windows (fresh period + fresh stocks), each with a matched control
    on the same rows - the same fix trained on outcomes shuffled within each week (all structure, no signal)."""
    from engine.research import evidence as EV
    from engine.research import replication as RP
    A, B = EV.stock_halves(P["ticker"].astype(str), f"fix:{fix.name}")
    P = P.assign(control=control)
    tr = P[(P["part"] == "train") & P["ticker"].astype(str).isin(A)]
    e0 = weekly(tr, "gain")
    if len(e0) < 3 or float(e0.std()) <= 0:
        return None, {"why": f"{len(e0)} training weeks on the discovery half"}
    did = "F" + stable_hash(fix.name, 8)
    disc = RP.Discovery(did, float(e0.mean()), float(e0.std()), int(len(e0)), (str(tr["date"].min()), str(tr["date"].max())), 7, A,
                        (ec.seed,), frozenset({"train"}), code_hash, data_hash, str(tr["matured_at"].max()), max(1, n_searched))
    te = P[(P["part"] == "test") & P["ticker"].astype(str).isin(B)].copy()
    wk = pd.to_datetime(te["date"]).dt.to_period("W-FRI").astype(str)
    weeks = np.array(sorted(wk.unique()))
    mv = P.groupby(pd.to_datetime(P["date"]).dt.to_period("W-FRI").astype(str))["m_vol"].median() if "m_vol" in P else None
    runs = []
    for i, chunk in enumerate(c for c in np.array_split(weeks, ec.repl_windows) if len(c)):
        R = te[wk.isin(set(chunk)).to_numpy()]
        e = weekly(R, "gain")
        c = weekly(R.assign(gain=R["control"]), "gain").reindex(e.index).fillna(0.0)
        if len(e) < 2:
            continue
        regime = "calm" if mv is None or float(mv.reindex(chunk).median()) <= float(mv.median()) else "high_vol"
        runs.append(RP.ReplicationRun(f"{did}-w{i}", did, (str(R["date"].min()), str(R["date"].max())), B, ec.seed + 1 + i, regime,
                                      tuple(float(v) for v in e.to_numpy()), tuple(float(v) for v in c.to_numpy()), code_hash, data_hash,
                                      str(R["matured_at"].max())))
    a = RP.assess(disc, runs, now)
    return a, {"repl_status": str(a.status), "repl_runs": len(runs), "repl_discovery_weeks": int(len(e0)),
               "repl_supporting": a.n_supporting}


def _in_band_prob(pred: np.ndarray, resid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.asarray(pred, float)[:, None] + np.asarray(resid, float)[None, :]
    return ((d >= lo) & (d <= hi)).mean(1)


def fix_calibration(P: pd.DataFrame, ec: FixEvidenceConfig) -> tuple[QG.CalibrationEvidence | None, str, dict]:
    """The fix's statement about the band: P(realised in [lo, hi]) = its point forecast plus its OWN training residuals (the way the
    gain model turns a point into a band probability), scored on the test rows it never saw."""
    tr, te = P[P["part"] == "train"], P[P["part"] == "test"]
    res = (tr["realised"] - tr["fix"]).to_numpy(float)
    res = res[np.isfinite(res)]
    if len(res) < 50 or len(te) < 50:
        return None, "too few rows to state a band probability", {}
    rng = np.random.default_rng(ec.seed + 5)
    draws = rng.choice(res, size=min(ec.resid_draws, len(res)), replace=False)
    p = np.clip(_in_band_prob(te["fix"].to_numpy(float), draws, ec.lo, ec.hi), 0.001, 0.999)
    y = ((te["realised"] >= ec.lo) & (te["realised"] <= ec.hi)).to_numpy(int)
    pi = np.clip(_in_band_prob(te["incumbent"].to_numpy(float), draws, ec.lo, ec.hi), 0.001, 0.999)
    brier = lambda q: float(np.mean((q - y) ** 2))                                      # noqa: E731
    return QG.CalibrationEvidence(tuple(float(v) for v in p), tuple(int(v) for v in y), ec.seed), "", \
        {"brier_fix": brier(p), "brier_incumbent_same_residuals": brier(pi), "base_in_band": float(y.mean())}


def _book(T: pd.DataFrame, col: str, ec: FixEvidenceConfig) -> pd.Series:
    """Per test week: the equal-weight realised gain of the names whose `col` point forecast lies in the band (0 = no name, cash)."""
    sel = T[(T[col] >= ec.lo) & (T[col] <= ec.hi)]
    all_w = weekly(T.assign(one=0.0), "one")
    return weekly(sel, "realised").reindex(all_w.index).fillna(0.0) if len(sel) else all_w * 0.0


def _risk_numbers(x: pd.Series) -> tuple[float, float, float, int]:
    cum = x.cumsum()
    k = max(1, int(0.05 * len(x)))
    return float(x.min()), float((cum - cum.cummax()).min()), float(x.nsmallest(k).mean()), 0


def fix_risk(P: pd.DataFrame, ec: FixEvidenceConfig):
    """The book the fix would select (its in-band names, weekly) against the incumbent's book on the same weeks."""
    T = P[P["part"] == "test"]
    fb, ib = _book(T, "fix", ec), _book(T, "incumbent", ec)
    if len(fb) < 2:
        return None, "fewer than two test weeks", {}
    w, mdd, cvar, _ = _risk_numbers(fb)
    iw, imdd, _, _ = _risk_numbers(ib)
    return PR.RiskEvidence(int(len(fb)), w, mdd, cvar, int((fb <= ec.catastrophic).sum()), iw, imdd), "", \
        {"book_mean": float(fb.mean()), "incumbent_book_mean": float(ib.mean()), "worst_week": w}


def fix_complexity(fix: CandidateFix, P: pd.DataFrame, ec: FixEvidenceConfig) -> QG.ComplexityEvidence:
    """engine.learning.complexity: each rule's weekly OUTCOME (minus its mean absolute error) on the same test weeks, folds = eras.
    The baseline is the incumbent (no added structure). The baseline's in-sample score is not given: the optimism ratio divides by
    the incumbent's own in-sample-to-test gap, which is market noise here (the incumbent was not fitted on this split); the fix's
    optimism is judged by the OOS gate's retention of its in-sample gain instead."""
    T = P[P["part"] == "test"]
    tr = P[P["part"] == "train"]
    c = -weekly(T.assign(e=(T["realised"] - T["fix"]).abs()), "e")
    s = -weekly(T.assign(e=(T["realised"] - T["incumbent"]).abs()), "e")
    folds = pd.Series([era_of(d, ec.era) for d in c.index], index=c.index)
    cand = CX.Candidate(fix.rule_spec(), c, folds, -float((tr["realised"] - tr["fix"]).abs().mean()))
    base = CX.Candidate(CX.RuleSpec("incumbent"), s, folds, None)
    return QG.ComplexityEvidence(cand, base, float(len(c)))


def fix_transfer(r: FixResult, P: pd.DataFrame, ec: FixEvidenceConfig):
    """Does the fix's gain carry from the era it was fitted in (home = its in-sample gain) to every later era (calendar halves or
    years of the test weeks)? A fix is scoped to what it corrects (a stock type, a pattern), so the contexts are eras, not sectors."""
    eff = pd.Series(r.oos_effects, index=pd.DatetimeIndex(pd.to_datetime(list(r.oos_periods))))
    if not len(eff) or not math.isfinite(r.in_sample_effect):
        return None, "no test weeks or no in-sample effect", {}
    g = eff.groupby([era_of(d, ec.era) for d in eff.index])
    ctx = {k: float(v) for k, v in g.mean().items()}
    n = {k: int(v) for k, v in g.size().items()}
    tr = P[P["part"] == "train"]
    ctx["in_sample"], n["in_sample"] = float(r.in_sample_effect), int(len(weekly(tr, "gain")))
    return PR.TransferEvidence(ctx, n, ("in_sample",), float(r.in_sample_effect)), "", {"eras": {k: round(v, 6) for k, v in ctx.items()}}


def fix_failure(fix: CandidateFix, r: FixResult, P: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, worst: float | None,
                ec: FixEvidenceConfig):
    """Failure behaviour MEASURED: failure weeks = test weeks where the fix did worse than the incumbent; the conditions that explain
    them (high / low market volatility, an era, or sampling noise: a week within z standard errors of the mean gain, family-wise over
    the test weeks); perturbation retention (the fix re-run on inputs degraded by seeded noise); abstention (asked to predict without
    its inputs it must refuse, never fill zeros); the retirement trigger is the caller's declared monitor."""
    T = P[P["part"] == "test"]
    eff = weekly(T, "gain")
    fails = eff[eff < 0]
    conds, explained = [], set()
    if "m_vol" in T and T["m_vol"].notna().any():
        mv = weekly(T, "m_vol")
        hi = mv > mv.median()
        for lab, mask in (("high market volatility", hi), ("low market volatility", ~hi)):
            ds = [d for d in eff.index if bool(mask.get(d, False))]
            if ds and eff.loc[ds].mean() < 0:
                conds.append(f"worse than the incumbent in {lab}")
                explained |= set(ds)
    for er, v in eff.groupby([era_of(d, ec.era) for d in eff.index]):
        if len(v) >= 4 and v.mean() < 0:
            conds.append(f"worse than the incumbent in era {er}")
            explained |= set(v.index)
    if len(fails):
        from statistics import NormalDist
        wk = pd.to_datetime(T["date"]).dt.to_period("W-FRI").astype(str).to_numpy()
        g = pd.DataFrame({"g": T["gain"].to_numpy(float), "m": pd.to_datetime(T["matured_at"]).to_numpy(), "w": wk}).groupby("w")
        se = pd.Series((g["g"].std(ddof=1) / np.sqrt(g["g"].size())).to_numpy(float), index=pd.DatetimeIndex(g["m"].max().to_numpy()))
        se = se.groupby(level=0).mean()
        z = NormalDist().inv_cdf(1.0 - ec.noise_alpha / (2.0 * max(1, len(eff))))
        mean = float(eff.mean())
        noise = [d for d, e in fails.items() if mean > 0 and math.isfinite(float(se.get(d, np.nan))) and (mean - e) <= z * float(se[d])]
        if noise:
            conds.append(f"sampling noise: weekly gain within {z:.2f} standard errors of its mean (family-wise {ec.noise_alpha:g})")
            explained |= set(noise)
    unknown = float(np.mean([d not in explained for d in fails.index])) if len(fails) else 0.0
    pred = fix.build(train, ec.seed)
    rng = np.random.default_rng(ec.seed + 17)
    noisy = test.copy()
    for c in [c for c in test.columns if str(c).startswith(("f_", "m_"))]:
        x = noisy[c].to_numpy(float)
        noisy[c] = x + rng.normal(0, ec.perturb_sd * (float(np.nanstd(x)) or 1.0), len(x))
    g_noisy = float((_abs_err(test["predicted"].to_numpy(float), test) - _abs_err(pred(noisy), test)).mean())
    base = float(T["gain"].mean())
    retention = float(g_noisy / base) if base > 0 else 0.0
    abstains = False
    try:
        out = pred(test.drop(columns=[c for c in test.columns if str(c).startswith(("f_", "m_"))]).iloc[:5])
        abstains = not np.isfinite(np.asarray(out, float)).all()
    except (KeyError, ValueError):
        abstains = True
    fe = QG.FailureEvidence(tuple(conds) or ("worse-than-incumbent weeks with no dominant context",), int(len(fails)),
                            bool(ec.retirement_trigger), abstains, retention, unknown, worst)
    return fe, {"failure_weeks": int(len(fails)), "perturbation_retention": retention, "unknown_cause_share": unknown, "abstains": abstains}


def control_predictions(fix: CandidateFix, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    """The matched control: the same fix trained on the same rows with the outcome shuffled WITHIN each decision week (same model,
    same inputs, same weeks - no signal). Its gain on the test rows is what luck and structure alone deliver."""
    rng = np.random.default_rng(seed + 101)
    sh = train.copy()
    sh["realised"] = sh.groupby("date")["realised"].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    return np.asarray(fix.build(sh, seed)(test), float)


def full_fix_evidence(r: FixResult, fix: CandidateFix, frame: pd.DataFrame, now, n_searched: int, base: QG.QualityEvidence | None,
                      code_hash: str, cfg: SelfCorrectConfig = SelfCorrectConfig(), ec: FixEvidenceConfig = FixEvidenceConfig()) -> FixBundle:
    """PUBLIC. Every QualityEvidence field for one tested fix, measured from its own rows. The cheap part is fix_evidence (OOS,
    statistics, reproducibility, PIT); the rest is built here only when the OOS screen passes (the section-18 ladder: a fix that does
    not beat the incumbent out of sample is FAILED on that alone, and costly audits are not spent on it)."""
    errs = ec.validate()
    if errs:
        raise ValueError("invalid FixEvidenceConfig: " + "; ".join(errs))
    ev = fix_evidence(r, fix, n_searched, now, base, code_hash)
    if not screen_passes(r, ec):
        return FixBundle(fix.name, ev, {"screen": "failed"}, {"all_audits": "the OOS screen failed: FAILED on out_of_sample"}, full=False)
    missing: dict = {}
    parts: dict = {"screen": "passed", "detail": dict(r.detail)}
    f, train, test, split = split_frame(frame, now, cfg)
    P = r.panel if r.panel is not None else test_fix(fix, frame, now, cfg).panel
    have_ticker = "ticker" in P.columns and P["ticker"].notna().all()
    rep = ident = None
    if have_ticker:
        try:
            rep, memo, p = fix_identity(fix, train, test, P, ec)
            ident = QG.IdentityEvidence(rep, memo)
            parts.update(p)
        except Exception as e:                                  # noqa: BLE001 - a harness refusal is recorded; the gate blocks
            missing["identity"] = f"{type(e).__name__}: {str(e)[:120]}"
    else:
        missing["identity"] = "no ticker column: identity dependence cannot be measured"
    prov = ev.provenance or Provenance(str(as_date(now)), str(as_date(now)), code_hash)
    leak = None
    if have_ticker:
        leak, p = fix_leakage(fix, P, now, split, str(test["date"].min()), rep, code_hash, prov, n_searched, ec.seed)
        parts.update(p)
    else:
        missing["leak"] = "no ticker column: the firewall needs a (date, ticker) panel"
    repl = None
    if have_ticker:
        repl, p = fix_replication(fix, P, _control_column(fix, P, train, test, ec), now, code_hash, r.data_digest, n_searched, ec)
        parts.update(p)
        if repl is None:
            missing["replication"] = p.get("why", "not computable")
    else:
        missing["replication"] = "no ticker column: fresh stocks cannot be separated"
    cal, why, p = fix_calibration(P, ec)
    parts.update(p)
    if cal is None:
        missing["calibration"] = why
    rk, why, p = fix_risk(P, ec)
    parts.update(p)
    if rk is None:
        missing["risk"] = why
    tev, why, p = fix_transfer(r, P, ec)
    parts.update(p)
    if tev is None:
        missing["transfer"] = why
    cx = fix_complexity(fix, P, ec)
    fe, p = fix_failure(fix, r, P, train, test, rk.worst_period if rk is not None else None, ec)
    parts.update(p)
    just = ("oos_transfer",) + (("replication",) if repl is not None else ()) + (("calibration",) if cal is not None and ec.states_probabilities else ())
    out = dataclasses.replace(ev, leak=leak, identity=ident, replication=repl, outputs_probabilities=bool(ec.states_probabilities), calibration=cal,
                              changes_risk_decisions=True, risk=rk, complexity=cx, transfer=tev, failure=fe, justification=just)
    return FixBundle(fix.name, out, parts, missing, full=True)


def _control_column(fix: CandidateFix, P: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, ec: FixEvidenceConfig) -> np.ndarray:
    """Per panel row, the matched control's gain over the incumbent (NaN on training rows: the control is scored out of sample)."""
    ctrl = control_predictions(fix, train, test, ec.seed)
    T = P["part"].to_numpy() == "test"
    out = np.full(len(P), np.nan)
    real, inc = P["realised"].to_numpy(float)[T], P["incumbent"].to_numpy(float)[T]
    out[T] = np.abs(real - inc) - np.abs(real - ctrl)
    return out


# ------------------------------------------------------------------------------------------------ forward-only regimes of fix evidence (F13)
# F11 flip seed 0: the planted error stopped at 80% of the sample and the sector-slope fix was PROMOTED eleven weeks later, on ~100
# weeks of OOS gains of which ~90 predated the stop. Nothing in the gate asked whether the evidence came from the regime in force: the
# OOS gate reads the whole-window mean, the loop's change detectors (change_points / regime_memory) were never consulted, and the
# market-level streams they watch do not move when one sector stops giving back its run-up. C68 (forward-only regimes): evidence for
# a promotion must come from the regime in force at the decision or be shown to hold across the change, and a change detected after a
# promotion re-opens it. Both are measured here from the fix's OWN weekly OOS gains, with change_points' calibrated CUSUM run forward.
REGIME_GATE = "regime_in_force"


@dataclass(frozen=True)
class RegimeConfig:
    # weekly gains, ~3 years per horizon. k = 1.0 targets a shift of ~2 weekly sds (a fix that stops working swings its gain from +mu
    # to below zero); warmup 30 because a 20-week reference is so noisy the calibrated threshold rose to h = 24.8 (vs 6.7 here)
    change: CP.ChangeConfig = CP.ChangeConfig(k=1.0, warmup=30, settle=10, horizon=150, scan_window=160)
    min_post_weeks: int = 8               # weeks inside the regime in force before its evidence can stand alone
    screen_t: float = 1.64                # the same bar as the OOS screen (FixEvidenceConfig.screen_t): never lower
    tail_weeks: int = 13                  # newest 3..13 weeks scanned for a turn the CUSUM has not declared yet
    tail_t: float = -3.0                  # tail mean below the older weeks at z <= this (older weeks' sd): the evidence does not hold
                                          # across it. A 6-week Welch t with its own variance false-alarmed on a stationary frame
    reval_min_weeks: int = 6              # post-change weeks before a promoted fix's re-validation can conclude
    reval_max_weeks: int = 13             # ... and by which it must have re-validated, or it is rolled back (the stated bound)

    def validate(self) -> list[str]:
        errs = list(self.change.validate())
        if self.min_post_weeks < 4 or self.tail_weeks < 3 or self.tail_t >= 0 or self.screen_t < 1.64:
            errs.append("min_post_weeks >= 4, tail_weeks >= 3, tail_t < 0 and screen_t >= 1.64 (never below the OOS screen) required")
        if not 3 <= self.reval_min_weeks <= self.reval_max_weeks:
            errs.append("3 <= reval_min_weeks <= reval_max_weeks required")
        return errs


@dataclass(frozen=True)
class ChangeMark:
    """One change point the evidence knows about: where the new level began, when it was declared, and by whom."""
    source: str                           # 'fix_gain' (this module's forward CUSUM) or the caller's (e.g. 'regime_memory:<scope>')
    change_date: str
    alarm_date: str
    direction: int = 0                    # +1 / -1 for a gain stream; 0 = an external change of unstated sign


@dataclass(frozen=True)
class RegimeCheck:
    fix: str
    now: str
    verdict: str                          # NO_CHANGE / HOLDS pass; STALE / TURNING need more evidence; FAILS fails
    reason: str
    n_weeks: int
    changes: tuple[ChangeMark, ...]
    regime_start: str | None
    post_weeks: int
    post_mean: float
    post_t: float
    tail_t: float
    stale_share: float                    # share of the evidence weeks dated before the regime in force began

    @property
    def ok(self) -> bool:
        return self.verdict in ("NO_CHANGE", "HOLDS")

    @property
    def gate_verdict(self) -> GateVerdict:
        return GateVerdict.PROMOTE if self.ok else GateVerdict.FAILED if self.verdict == "FAILS" else GateVerdict.NEEDS_MORE_EVIDENCE

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason, "weeks": self.n_weeks, "regime_start": self.regime_start,
                "post_weeks": self.post_weeks, "post_mean": self.post_mean, "post_t": self.post_t, "tail_t": self.tail_t,
                "stale_share": self.stale_share, "changes": [dataclasses.asdict(c) for c in self.changes]}


def gain_changes(dates: Sequence[str], values: Sequence[float], cfg: RegimeConfig = RegimeConfig(), stream: str = "fix_gain") -> list[ChangeMark]:
    """Change points of one weekly gain series, found by change_points.StreamDetector fed ONE week at a time in date order (a
    detection at week t is a function of weeks <= t only: change_points.check_no_lookahead's property, re-proved in the tests)."""
    det = CP.StreamDetector(stream, CP.Target.ERROR_DISTRIBUTION, cfg.change)
    out = []
    for d, v in zip(dates, values):
        x = det.update(d, v)
        if x is not None:
            out.append(ChangeMark("fix_gain", x.change_date, x.alarm_date, int(x.direction)))
    return out


def _mean_t(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(PR.t_stat(x))


def tail_z(tail: np.ndarray, older: np.ndarray) -> float:
    """z of mean(tail) - mean(older) on the OLDER weeks' sd (a handful of tail weeks cannot estimate their own variance: a Welch t
    on 6 weeks alarmed on a stationary planted frame); NaN when the older side is too short."""
    if len(tail) < 1 or len(older) < 8:
        return float("nan")
    sd = float(np.std(older, ddof=1))
    return float((tail.mean() - older.mean()) / (sd * math.sqrt(1.0 / len(tail) + 1.0 / len(older)))) if sd > 0 else float("nan")


def tail_scan(v: np.ndarray, max_k: int) -> tuple[float, int]:
    """The most negative tail_z over the newest k = 3..max_k weeks (each against all older weeks), and its k; (NaN, 0) when the
    series is too short for any k. Reads only the series handed in (weeks already matured)."""
    best, arg = float("nan"), 0
    for k in range(3, min(max_k, len(v) - 8) + 1):
        z = tail_z(v[-k:], v[:-k])
        if math.isfinite(z) and not z >= best:
            best, arg = z, k
    return best, arg


def regime_check(fix: str, dates: Sequence[str], values: Sequence[float], now, external: Sequence[ChangeMark] = (),
                 cfg: RegimeConfig = RegimeConfig()) -> RegimeCheck:
    """PUBLIC. Is a fix's OOS evidence drawn from the regime in force at `now`? The regime in force begins at the newest change point
    known at `now` - declared by the forward CUSUM on the fix's own weekly gains, or handed in by the caller (the loop's
    change_points / regime_memory) - that lies inside the evidence window. Verdicts:
      NO_CHANGE  no change inside the window, and the newest `tail_weeks` are not significantly below the rest (no turn under way)
      TURNING    no declared change, but the newest weeks are significantly worse than the older ones: not shown to hold across it
      STALE      a change, but fewer than `min_post_weeks` weeks after it: the evidence predates the regime in force
      HOLDS      the post-change weeks ALONE pass the OOS screen (mean > 0, t >= screen_t)
      FAILS      they do not: in the regime in force the fix is not shown to work.
    Weeks dated at/after `now` are a FirewallBreach (the evidence must already exist)."""
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid RegimeConfig: " + "; ".join(errs))
    d = [str(as_date(x)) for x in dates]
    v = np.asarray(values, float)
    if len(d) != len(v):
        raise ValueError("dates and values differ in length")
    nd = str(as_date(now))
    if any(x >= nd for x in d):
        raise FirewallBreach(f"{fix}: evidence week dated at/after now={nd}")
    first = d[0] if d else nd
    marks = gain_changes(d, v, cfg) + [m for m in external if first <= str(m.change_date) and str(m.alarm_date) < nd]
    marks = sorted(marks, key=lambda m: (m.change_date, m.alarm_date, m.source))
    start = marks[-1].change_date if marks else None
    post = v[np.array([x >= start for x in d], bool)] if start is not None and len(d) else np.array([])
    pm, pt = _mean_t(post)
    tt, k = tail_scan(v, cfg.tail_weeks)
    stale = float(np.mean([x < start for x in d])) if start is not None and d else 0.0
    mk = lambda verdict, why: RegimeCheck(fix, nd, verdict, why, len(v), tuple(marks), start, int(len(post)), pm, pt, tt, stale)  # noqa: E731
    if start is None:
        if math.isfinite(tt) and tt <= cfg.tail_t:
            return mk("TURNING", f"newest {k} weeks below the older {len(v) - k} at z={tt:.2f} <= {cfg.tail_t}: no change declared yet, "
                                 f"but the evidence is not shown to hold across the turn")
        return mk("NO_CHANGE", f"no change point in {len(v)} evidence weeks; worst newest-k z={tt:.2f} (k={k})")
    if len(post) < cfg.min_post_weeks:
        return mk("STALE", f"change at {start} ({marks[-1].source}, declared {marks[-1].alarm_date}); only {len(post)} week(s) after it "
                           f"(< {cfg.min_post_weeks}): {stale:.0%} of the evidence predates the regime in force")
    if not (pm > 0 and math.isfinite(pt) and pt >= cfg.screen_t):
        return mk("FAILS", f"after the change at {start}: {len(post)} weeks, mean {pm:+.5f}, t={pt:.2f} (< {cfg.screen_t} or mean <= 0): "
                           f"the fix is not shown to work in the regime in force")
    tt, k = tail_scan(post, cfg.tail_weeks)                                     # a new turn inside the regime in force
    if math.isfinite(tt) and tt <= cfg.tail_t:
        return dataclasses.replace(mk("TURNING", f"after the change at {start} the newest {k} weeks fall below the rest of the regime at "
                                                 f"z={tt:.2f} <= {cfg.tail_t}"), tail_t=tt)
    return dataclasses.replace(mk("HOLDS", f"after the change at {start}: {len(post)} weeks, mean {pm:+.5f}, t={pt:.2f} >= {cfg.screen_t}"),
                               tail_t=tt)


def apply_regime(decision: QG.QualityDecision, chk: RegimeCheck) -> QG.QualityDecision:
    """Merge a regime check into the gate's decision: it can only make it stricter (PROMOTE -> NEEDS_MORE_EVIDENCE / FAILED, a
    NEEDS_MORE_EVIDENCE -> FAILED); QUARANTINED and UNKNOWN stand. The check is added to the decision's gates either way."""
    ok = chk.ok
    state = QG.PASS if ok else (QG.FAIL if chk.verdict == "FAILS" else QG.MISSING)
    gate = QG.GateOutcome(REGIME_GATE, state, True, chk.reason, {"verdict": chk.verdict, "post_weeks": chk.post_weeks,
                                                                   "stale_share": chk.stale_share})
    rank = {GateVerdict.PROMOTE: 0, GateVerdict.NEEDS_MORE_EVIDENCE: 1, GateVerdict.FAILED: 2}
    verdict = decision.verdict
    if not ok and verdict in rank and rank[chk.gate_verdict] > rank[verdict]:
        verdict = chk.gate_verdict
    blocking = decision.blocking + ((REGIME_GATE,) if not ok else ())
    reasons = decision.reasons + ((f"{REGIME_GATE}: {chk.reason}",) if not ok else ())
    return dataclasses.replace(decision, verdict=verdict, gates=decision.gates + (gate,), blocking=blocking, reasons=reasons)


@dataclass(frozen=True)
class Revalidation:
    """What a change point detected AFTER a promotion does to the promoted fix."""
    state: str                            # WATCHING (no new change) / REVALIDATING / REVALIDATED / ROLLBACK
    reason: str
    change: ChangeMark | None
    post_weeks: int
    post_mean: float
    post_t: float

    @property
    def rollback(self) -> bool:
        return self.state == "ROLLBACK"


def revalidate(evidence_dates: Sequence[str], evidence_values: Sequence[float], since, live_dates: Sequence[str],
               live_values: Sequence[float], now, external: Sequence[ChangeMark] = (), cfg: RegimeConfig = RegimeConfig()) -> Revalidation:
    """PUBLIC. Re-validation of a promoted fix. Its promotion evidence (weekly OOS gains up to `since`) and its live weekly gains over
    the incumbent kept in the shadow since then form one series, watched by the same forward CUSUM; any change DECLARED after `since`
    (or an external one declared after it, whose change lies inside the series) means the promotion's evidence predates the regime in
    force, and the fix must re-validate on post-change weeks alone:
      post mean <= 0 after reval_min_weeks                       ROLLBACK (it stopped working)
      post mean > 0 and t >= screen_t after reval_min_weeks      REVALIDATED
      still neither after reval_max_weeks                        ROLLBACK (not re-validated within the stated bound)
    So a fix that stops working is rolled back at most reval_max_weeks post-change weeks after the change was declared (plus the
    caller's cadence). Weeks at/after `now` are a FirewallBreach."""
    nd, sd = str(as_date(now)), str(as_date(since))
    ld = [str(as_date(x)) for x in live_dates]
    lv = np.asarray(live_values, float)
    ed = [str(as_date(x)) for x in evidence_dates]
    ev = np.asarray(evidence_values, float)
    if any(x >= nd for x in ld + ed):
        raise FirewallBreach(f"re-validation at {nd} handed a week dated at/after it")
    cut = ld[0] if ld else nd
    keep = np.array([x < cut for x in ed], bool) if ed else np.array([], bool)
    dates = [x for x, k in zip(ed, keep) if k] + ld
    vals = np.concatenate([ev[keep] if len(ev) else ev, lv])
    first = dates[0] if dates else nd
    marks = [m for m in gain_changes(dates, vals, cfg) if m.alarm_date > sd]
    # no tail test here: tried (F13) against [evidence + live], it rolled back two WORKING fixes (flip seeds 1 and 5, 25 weeks BEFORE
    # the flip) because live gains run below the promotion evidence's level; smaller drops are left to the degradation monitor
    marks += [m for m in external if str(m.alarm_date) > sd and str(m.alarm_date) < nd and str(m.change_date) >= first]
    if not marks:
        return Revalidation("WATCHING", "no change declared since the promotion", None, 0, float("nan"), float("nan"))
    m = max(marks, key=lambda x: (x.change_date, x.alarm_date))
    post = vals[np.array([x >= m.change_date for x in dates], bool)]
    pm, pt = _mean_t(post)
    n = int(len(post))
    head = f"change at {m.change_date} ({m.source}, declared {m.alarm_date}); {n} post-change week(s), mean {pm:+.5f}, t={pt:.2f}"
    if n >= cfg.reval_min_weeks and pm <= 0:
        return Revalidation("ROLLBACK", head + ": the fix stopped working in the regime in force", m, n, pm, pt)
    if n >= cfg.reval_min_weeks and math.isfinite(pt) and pt >= cfg.screen_t:
        return Revalidation("REVALIDATED", head + f": re-validated (t >= {cfg.screen_t})", m, n, pm, pt)
    if n >= cfg.reval_max_weeks:
        return Revalidation("ROLLBACK", head + f": not re-validated within {cfg.reval_max_weeks} post-change weeks", m, n, pm, pt)
    return Revalidation("REVALIDATING", head + ": awaiting post-change evidence", m, n, pm, pt)


@dataclass(frozen=True)
class DelayPoint:
    shift: float                          # post-change gain minus pre-change gain, in weekly-gain sds (negative = the fix got worse)
    rollback_prob: float                  # within max_weeks after the change
    median_weeks: float | None
    p90_weeks: float | None
    max_weeks_seen: float | None
    false_rollback: float                 # the same, with no change at all (shift 0 path): rollbacks per run


def rollback_delay_curve(cfg: RegimeConfig = RegimeConfig(), snr: float = 0.8, shifts: Sequence[float] = (-1.0, -1.5, -2.0, -3.0),
                         evidence_weeks: int = 60, pre_weeks: int = 8, max_weeks: int = 52, n_sim: int = 200, seed: int = 0) -> list[DelayPoint]:
    """The rollback delay distribution by simulation of exactly `revalidate` week by week: a fix promoted on `evidence_weeks` of
    weekly gains ~ N(snr, 1), live for `pre_weeks` weeks, then its gain shifts by `shift` sds. Delay = weeks from the change to the
    first ROLLBACK. `false_rollback` runs the same with shift 0 for pre_weeks + max_weeks weeks."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2010-01-01")
    wk = lambda i: str((base + pd.Timedelta(weeks=int(i))).date())                                       # noqa: E731

    def first_rollback(ev: np.ndarray, live: np.ndarray) -> int | None:
        ed = [wk(i) for i in range(len(ev))]
        since = wk(len(ev))
        for j in range(1, len(live) + 1):
            ld = [wk(len(ev) + 1 + i) for i in range(j)]
            if revalidate(ed, ev, since, ld, live[:j], wk(len(ev) + 2 + j), cfg=cfg).rollback:
                return j
        return None
    out = []
    null = [first_rollback(rng.normal(snr, 1, evidence_weeks), rng.normal(snr, 1, pre_weeks + max_weeks)) for _ in range(max(20, n_sim // 4))]
    fr = float(np.mean([x is not None for x in null]))
    for sh in shifts:
        delays = []
        for _ in range(n_sim):
            live = np.concatenate([rng.normal(snr, 1, pre_weeks), rng.normal(snr + sh, 1, max_weeks)])
            j = first_rollback(rng.normal(snr, 1, evidence_weeks), live)
            if j is not None and j > pre_weeks:
                delays.append(j - pre_weeks)
        out.append(DelayPoint(float(sh), len(delays) / n_sim, float(np.median(delays)) if delays else None,
                              float(np.quantile(delays, 0.9)) if delays else None, float(max(delays)) if delays else None, fr))
    return out


@dataclass(frozen=True)
class CorrectionReport:
    now: str
    diagnosis: Diagnosis
    results: tuple[FixResult, ...]
    decisions: tuple[QG.QualityDecision, ...]
    promoted: tuple[str, ...]
    rejected: Mapping[str, str]             # fix -> verdict and blocking gates
    bundles: tuple = ()                     # FixBundle per fix: the evidence the gate saw, and what was missing
    regime: Mapping[str, RegimeCheck] = field(default_factory=dict)   # fix -> is its evidence from the regime in force (F13)

    def summary(self) -> str:
        d = self.diagnosis
        lines = [f"deterioration: {'YES' if d.deterioration.detected else 'no'} (MAE {d.deterioration.reference_mae:.4f} -> "
                 f"{d.deterioration.recent_mae:.4f}, p={d.deterioration.p:.3g})",
                 "implicated: " + (", ".join(c.value for c in d.implicated) or "none"),
                 "untested: " + (", ".join(c.value for c in d.untested) or "none")]
        for r in self.results:
            lines.append(f"fix {r.fix} [{r.component.value}]: OOS gain {r.mean_effect:+.5f}/prediction (t={r.t:.2f}, {len(r.oos_effects)} weeks)"
                         f" -> {'PROMOTED' if r.fix in self.promoted else self.rejected.get(r.fix, '?')}")
        return "\n".join(lines)

    def bundle(self, fix: str) -> FixBundle | None:
        return next((b for b in self.bundles if b.fix == fix), None)

    def decision(self, fix: str) -> QG.QualityDecision | None:
        return next((d for d in self.decisions if d.subject_id == f"fix:{fix}"), None)


def gate_bundles(fixes: Sequence[CandidateFix], results: Sequence[FixResult], now, base: QG.QualityEvidence | None = None,
                 policy: QG.QualityPolicy | None = None, frame: pd.DataFrame | None = None, cfg: SelfCorrectConfig = SelfCorrectConfig(),
                 evidence: FixEvidenceConfig | None = None, n_searched: int | None = None, regime: RegimeConfig = RegimeConfig(),
                 changes: Sequence[ChangeMark] = ()):
    """Gate every tested fix through engine.research.quality_gate. With `evidence` (and the frame the fixes were tested on) each fix's
    FULL evidence is measured (full_fix_evidence); without it only the cheap measured part plus the caller's `base` audits reach the
    gate (absent audits = MISSING, never a pass). Multiplicity: the search size is the sum of every fix's own variants.
    Every decision then passes the regime check (F13, always on): the fix's weekly OOS gains must come from the regime in force at
    `now` - change points from its own forward CUSUM and the caller's `changes` (the loop's change_points / regime_memory). The
    check can only make a verdict stricter; its result is in each bundle's parts['regime']."""
    pol = policy or QG.QualityPolicy()
    code = pol.code_hash or PR.current_code_hash()
    ns = int(n_searched or sum(max(1, int(x.n_variants)) for x in fixes))
    bundles = []
    for x, r in zip(fixes, results):
        if evidence is not None and frame is not None:
            bundles.append(full_fix_evidence(r, x, frame, now, ns, base, code, cfg, evidence))
        else:
            bundles.append(FixBundle(x.name, fix_evidence(r, x, ns, now, base, code), {}, {}, full=False))
    rep = QG.step([QG.Candidate(f"fix:{b.fix}", b.evidence) for b in bundles], now, pol)
    by_fix = {r.fix: r for r in results}
    decisions = []
    for d in rep.decisions:
        name = d.subject_id.split(":", 1)[1]
        r = by_fix.get(name)
        if r is not None:
            chk = regime_check(name, r.oos_periods, r.oos_effects, now, changes, regime)
            d = apply_regime(d, chk)
            b = next((b for b in bundles if b.fix == name), None)
            if b is not None:
                b.parts["regime"] = chk
        decisions.append(d)
    promoted = [d.subject_id.split(":", 1)[1] for d in decisions if d.promote]
    rejected = {d.subject_id.split(":", 1)[1]: f"{d.verdict.value} ({', '.join(d.blocking)})" for d in decisions if not d.promote}
    return decisions, promoted, rejected, bundles


def gate_fixes(fixes: Sequence[CandidateFix], results: Sequence[FixResult], now, base: QG.QualityEvidence | None = None,
               policy: QG.QualityPolicy | None = None) -> tuple[list[QG.QualityDecision], list[str], dict[str, str]]:
    d, p, r, _ = gate_bundles(fixes, results, now, base, policy)
    return d, p, r


def step(frame: pd.DataFrame, fixes: Sequence[CandidateFix], now, base: QG.QualityEvidence | None = None,
         policy: QG.QualityPolicy | None = None, cfg: SelfCorrectConfig = SelfCorrectConfig(), only_if_deteriorated: bool = False,
         evidence: FixEvidenceConfig | None = None, n_searched: int | None = None, regime: RegimeConfig = RegimeConfig(),
         changes: Sequence[ChangeMark] = ()) -> CorrectionReport:
    """Public entry for the wave-2 loop: diagnose, test each fix alone, gate each through the existing architecture (with `evidence`,
    on its full measured evidence), then through the regime check (evidence from the regime in force; `changes` = the caller's
    detected change points)."""
    diag = diagnose(frame, now, cfg)
    if only_if_deteriorated and not diag.deterioration.detected:
        return CorrectionReport(str(as_date(now)), diag, (), (), (), {})
    results = test_fixes(fixes, frame, now, cfg)
    decisions, promoted, rejected, bundles = gate_bundles(fixes, results, now, base, policy, frame, cfg, evidence, n_searched, regime, changes)
    checks = {b.fix: b.parts["regime"] for b in bundles if "regime" in b.parts}
    return CorrectionReport(str(as_date(now)), diag, tuple(results), tuple(decisions), tuple(promoted), rejected, tuple(bundles), checks)


def slope_fix(name: str, features: Sequence[str], group_col: str = "sector", n_groups: int = 1, lam: float = 1.0, min_rows: int = 30,
              component: Component = Component.SECTOR_CONDITIONING) -> CandidateFix:
    """A checklist-Q fix for a slope that differs in one stock type: the incumbent's error is regressed, within each group, on each
    (centred) feature; the (feature, group) pair with the largest |t| on the TRAINING rows is kept, and the fix adds that one
    within-group slope to the incumbent: predicted + b * 1[group] * (feature - its training mean in the group). It is a pure slope
    change (zero mean in the group on the training rows), so it can never be credited with a level / bias correction it does not
    make. Structure added: one interaction and one condition (the group clause); the search over features x groups is its
    multiplicity (n_variants). A frame without the chosen columns is refused (KeyError), never zero-filled."""
    feats = tuple(features)

    def build(train: pd.DataFrame, seed: int):
        resid = (train["realised"] - train["predicted"]).to_numpy(float)
        grp = train[group_col].astype(str).to_numpy() if group_col in train else np.full(len(train), "", object)
        best = None
        for f in feats:
            if f not in train:
                continue
            x_all = train[f].to_numpy(float)
            for g in sorted(set(grp.tolist())):
                m = (grp == g) & np.isfinite(x_all) & np.isfinite(resid)
                if m.sum() < min_rows:
                    continue
                c = float(x_all[m].mean())
                x, y = x_all[m] - c, resid[m]
                sxx = float(x @ x)
                if sxx <= 1e-18:
                    continue
                b = float(x @ y) / (sxx + lam * float(np.var(x)))
                e = y - b * x
                se = math.sqrt(max(float(e @ e) / max(len(y) - 2, 1), 1e-18) / sxx)
                t = b / se
                if best is None or abs(t) > abs(best[4]):
                    best = (f, g, c, b, t)

        def predict(fr: pd.DataFrame) -> np.ndarray:
            base = fr["predicted"].to_numpy(float)
            if best is None:
                return base
            f, g, c, b, _ = best
            x = fr[f].to_numpy(float)                            # KeyError when the input is absent: the fix abstains
            return base + np.where(fr[group_col].astype(str).to_numpy() == g, b * (x - c), 0.0)
        predict.detail = {} if best is None else {"feature": best[0], "group": best[1], "centre": best[2], "coef": best[3], "t": best[4],
                                                   "group_col": group_col}
        return predict
    return CandidateFix(name, component, tuple(FixInput(f[2:] if f.startswith("f_") else f) for f in feats) + (FixInput(group_col),), build,
                        CX.RuleSpec(f"fix_{name}", n_conditions=1, n_interactions=1), max(1, len(feats) * max(1, int(n_groups))))


def targeted(diag: Diagnosis, fixes: Sequence[CandidateFix]) -> list[CandidateFix]:
    """Fixes aimed at an implicated component first, then the rest (a fix for an unimplicated component is still testable, but it
    is research, not a response to the diagnosis)."""
    rank = {c: i for i, c in enumerate(diag.implicated)}
    return sorted(fixes, key=lambda x: (rank.get(x.component, len(rank)), x.name))
