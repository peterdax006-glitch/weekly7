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
from engine.learning import promotion as PR
from engine.learning.core import FirewallBreach, Provenance, _StrEnum, as_date, stable_hash
from engine.learning.surprise import benjamini_hochberg
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
    """One proposed change, tested ALONE. `build(train, seed)` returns a predictor `frame -> predicted realisable gain`."""
    name: str
    component: Component
    inputs: tuple[FixInput, ...]
    build: Callable[[pd.DataFrame, int], Callable[[pd.DataFrame], np.ndarray]]


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


def _abs_err(pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.abs(frame["realised"].to_numpy(float) - np.asarray(pred, float))


def test_fix(fix: CandidateFix, frame: pd.DataFrame, now, cfg: SelfCorrectConfig = SelfCorrectConfig()) -> FixResult:
    """Train on rows matured before the split date, test on rows DECIDED after it (the embargo leaves out rows whose outcome
    straddles the split). The incumbent is the recorded `predicted` column; the fix is compared with it and nothing else."""
    f = _check(frame, now)
    if len(f) < 2 * cfg.min_rows:
        raise ValueError(f"only {len(f)} matured rows: cannot test a fix")
    dates = pd.to_datetime(f["date"])
    split = dates.quantile(cfg.split_share)
    train = f[(pd.to_datetime(f["matured_at"]) < split).to_numpy()]
    test = f[(dates > split + pd.Timedelta(days=cfg.label_horizon_days)).to_numpy()]
    if len(train) < cfg.min_rows or len(test) < cfg.min_rows:
        raise ValueError("split leaves too few training or test rows")
    dg = stable_hash({"n": len(f), "first": str(f["date"].iloc[0]), "last": str(f["matured_at"].iloc[-1]),
                      "sum": round(float(f["realised"].sum()), 10)})
    reruns = []
    week = pd.to_datetime(test["date"]).dt.to_period("W-FRI").astype(str).to_numpy()
    eff_by_week, ins = None, float("nan")
    for seed in cfg.reruns:
        pred = fix.build(train, seed)
        gain = _abs_err(test["predicted"].to_numpy(float), test) - _abs_err(pred(test), test)
        reruns.append((int(seed), float(gain.mean())))
        if eff_by_week is None:
            eff_by_week = pd.Series(gain).groupby(week).mean()
            ins = float((_abs_err(train["predicted"].to_numpy(float), train) - _abs_err(pred(train), train)).mean())
    e = eff_by_week.to_numpy(float)
    # a week's evidence exists once its LAST outcome has matured - labelling it by the calendar week's end could date it after `now`
    # (a Friday not yet reached), which the gate's provenance check rightly refuses (found wiring P06 into the loop)
    last_mat = pd.Series(pd.to_datetime(test["matured_at"]).to_numpy()).groupby(week).max()
    per = tuple(str(pd.Timestamp(last_mat[p]).date()) for p in eff_by_week.index)
    years = tuple(sorted({int(y) for y in pd.to_datetime(train["date"]).dt.year}))
    return FixResult(fix.name, fix.component, str(split.date()), years, len(train), len(test), per, tuple(float(x) for x in e), ins,
                     float(e.mean()), float(PR.t_stat(e)), tuple(reruns), dg)


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


@dataclass(frozen=True)
class CorrectionReport:
    now: str
    diagnosis: Diagnosis
    results: tuple[FixResult, ...]
    decisions: tuple[QG.QualityDecision, ...]
    promoted: tuple[str, ...]
    rejected: Mapping[str, str]             # fix -> verdict and blocking gates

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


def gate_fixes(fixes: Sequence[CandidateFix], results: Sequence[FixResult], now, base: QG.QualityEvidence | None = None,
               policy: QG.QualityPolicy | None = None) -> tuple[list[QG.QualityDecision], list[str], dict[str, str]]:
    pol = policy or QG.QualityPolicy()
    code = pol.code_hash or PR.current_code_hash()
    cands = [QG.Candidate(f"fix:{x.name}", fix_evidence(r, x, len(fixes), now, base, code)) for x, r in zip(fixes, results)]
    rep = QG.step(cands, now, pol)
    promoted = [d.subject_id.split(":", 1)[1] for d in rep.decisions if d.promote]
    rejected = {d.subject_id.split(":", 1)[1]: f"{d.verdict.value} ({', '.join(d.blocking)})" for d in rep.decisions if not d.promote}
    return list(rep.decisions), promoted, rejected


def step(frame: pd.DataFrame, fixes: Sequence[CandidateFix], now, base: QG.QualityEvidence | None = None,
         policy: QG.QualityPolicy | None = None, cfg: SelfCorrectConfig = SelfCorrectConfig(), only_if_deteriorated: bool = False) -> CorrectionReport:
    """Public entry for the wave-2 loop: diagnose, test each fix alone, gate each through the existing architecture."""
    diag = diagnose(frame, now, cfg)
    if only_if_deteriorated and not diag.deterioration.detected:
        return CorrectionReport(str(as_date(now)), diag, (), (), (), {})
    results = test_fixes(fixes, frame, now, cfg)
    decisions, promoted, rejected = gate_fixes(fixes, results, now, base, policy)
    return CorrectionReport(str(as_date(now)), diag, tuple(results), tuple(decisions), tuple(promoted), rejected)


def targeted(diag: Diagnosis, fixes: Sequence[CandidateFix]) -> list[CandidateFix]:
    """Fixes aimed at an implicated component first, then the rest (a fix for an unimplicated component is still testable, but it
    is research, not a response to the diagnosis)."""
    rank = {c: i for i, c in enumerate(diag.implicated)}
    return sorted(fixes, key=lambda x: (rank.get(x.component, len(rank)), x.name))
