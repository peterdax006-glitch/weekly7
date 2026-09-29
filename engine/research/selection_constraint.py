"""5-10% selection constraint (C68 checklist L, with M's "no historical-statistics selection"). IMPLEMENTED - NOT VALIDATED.

A name is ELIGIBLE only when the system predicts that its eventual REALISABLE gain under the INTENDED exit policy falls in
[+5%, +10%] - not because P(up) is high, not because volatility is high, not because selecting it would improve a historical
statistic. The quantity forecast is therefore fixed: the net return that the intended exit rule (an engine.exits.Rule, e.g. the
learned exit of engine.research.exit_research or a tournament winner from engine.exits.learn) actually realises. A forecast made
under a different exit policy than the one that will trade is STALE and never eligible.

Mechanism (past only, fail closed): `RealisableGainModel.fit` runs the intended rule on matured positions (every position's last bar
strictly before `now`, else FirewallBreach), regresses the realised net return on features known at entry (ridge, standardised,
kind intercepts), and keeps the residuals of a LATER-IN-TIME holdout so the band probability is split-conformal, not the in-sample
optimism. `forecast` returns mean, quantiles and P(in band); `select` applies the constraint and returns a funnel with a reason for
every name. `identification_curve` measures, period by period, how well the constraint finds the band (precision vs base rate with
Wilson intervals) - the "continuously improve" readout. Builds on engine.exits (Paths, Rule, run_exit), engine.stops.wilson and
engine.research.two_stage (DayDecision positions are filtered by `apply_to_positions`)."""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import exits as EX
from engine.learning.core import FirewallBreach, _StrEnum, as_date, stable_hash
from engine.stops import wilson

SECTION = "C68 checklist L"
QUANTITY = "realisable_net_gain_under_intended_exit_policy"
# Inputs that are outcomes or statistics of outcomes: a model fed these "predicts" by reading the answer (checklist L/M).
FORBIDDEN_FEATURE = re.compile(r"(fwd|forward|future|realis|realiz|outcome|label|target|next_|_ahead|hindsight|pnl|hit_rate|win_rate)", re.I)
# Bases that checklist L names explicitly as NOT sufficient for selection.
FORBIDDEN_BASES = frozenset({"p_up", "direction_probability", "volatility", "vol_rank", "historical_statistic", "backtest_improvement"})


class Reason(_StrEnum):
    ELIGIBLE = "ELIGIBLE"
    BELOW_BAND = "BELOW_BAND"
    ABOVE_BAND = "ABOVE_BAND"
    LOW_BAND_PROBABILITY = "LOW_BAND_PROBABILITY"
    STALE_POLICY = "STALE_POLICY"
    WRONG_QUANTITY = "WRONG_QUANTITY"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class SelectionConfig:
    lo: float = EX.BAND_LO
    hi: float = EX.BAND_HI
    min_p_in_band: float = 0.25            # stated, not tuned: the forecast must put real mass inside the band
    point: str = "median"                  # which point forecast must lie in the band: median | mean
    min_support: int = 60                  # matured training positions below which the model abstains
    holdout_share: float = 0.3             # later-in-time share kept for conformal residuals
    ridge: float = 1.0
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 <= self.lo < self.hi):
            errs.append("band must satisfy 0 <= lo < hi")
        if (self.lo, self.hi) != (EX.BAND_LO, EX.BAND_HI):
            errs.append(f"band {self.lo}-{self.hi} differs from the trading constraint {EX.BAND_LO}-{EX.BAND_HI} (checklist L fixes it)")
        if not 0.0 <= self.min_p_in_band <= 1.0:
            errs.append("min_p_in_band must be in [0,1]")
        if self.point not in ("median", "mean"):
            errs.append("point must be median or mean")
        if not 0.1 <= self.holdout_share <= 0.5:
            errs.append("holdout_share must be in [0.1, 0.5]")
        if self.min_support < 20:
            errs.append("min_support below 20 cannot estimate a band probability")
        if self.ridge <= 0:
            errs.append("ridge must be positive")
        if any(not 0 < q < 1 for q in self.quantiles) or 0.5 not in self.quantiles:
            errs.append("quantiles must be in (0,1) and include 0.5")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


def policy_id(rule: Any) -> str:
    """Identity of an exit policy: its declared id, else a hash of its class, name and parameters. Two different exit rules can
    never share an id, so a forecast made under one cannot be used to select for the other."""
    pid = getattr(rule, "policy_id", None)
    if isinstance(pid, str) and pid:
        return pid
    body: dict[str, Any] = {"cls": type(rule).__name__, "name": getattr(rule, "name", "")}
    spec = getattr(rule, "spec", None)
    if spec is not None and dataclasses.is_dataclass(spec):
        body["spec"] = dataclasses.asdict(spec)
    for attr in ("k", "dist_", "threshold", "params"):
        if hasattr(rule, attr):
            v = getattr(rule, attr)
            body[attr] = v.tolist() if isinstance(v, np.ndarray) else v
    return "X" + stable_hash(body, 14)


@dataclass(frozen=True)
class GainForecast:
    """One name's forecast of the realisable gain under ONE exit policy, as of `as_of`. `candidate` is opaque (the trader's
    position key); `quantity` must be QUANTITY and `basis` records what drove it (never a forbidden basis)."""
    candidate: str
    as_of: str
    policy_id: str
    mean: float
    quantiles: Mapping[float, float]
    p_in_band: float
    n_support: int
    model_digest: str
    quantity: str = QUANTITY
    basis: str = "realisable_gain_model"

    @property
    def median(self) -> float:
        return float(self.quantiles.get(0.5, self.mean))

    def validate(self) -> list[str]:
        errs = []
        if not math.isfinite(self.mean):
            errs.append("non-finite mean")
        if not 0.0 <= self.p_in_band <= 1.0:
            errs.append("p_in_band outside [0,1]")
        qs = [self.quantiles[k] for k in sorted(self.quantiles)]
        if any(b < a - 1e-12 for a, b in zip(qs, qs[1:])):
            errs.append("quantiles not monotone")
        if self.quantity != QUANTITY:
            errs.append(f"forecast quantity {self.quantity!r} is not the realisable gain under the intended exit policy")
        if self.basis in FORBIDDEN_BASES:
            errs.append(f"selection basis {self.basis!r} is forbidden by checklist L")
        return errs


@dataclass(frozen=True)
class Eligibility:
    candidate: str
    eligible: bool
    reason: Reason
    detail: str
    point: float | None = None
    p_in_band: float | None = None


def _design(F: np.ndarray, kinds: np.ndarray, kind_levels: Sequence[str]) -> np.ndarray:
    onehot = np.stack([(kinds == k).astype(float) for k in kind_levels], 1) if kind_levels else np.zeros((len(F), 0))
    return np.hstack([F, onehot])


def _check_features(names: Sequence[str]) -> None:
    bad = [n for n in names if FORBIDDEN_FEATURE.search(str(n))]
    if bad:
        raise FirewallBreach(f"outcome-like feature(s) {bad} refused: selection may only use information known at entry")


def _matured_only(p: EX.Paths, now) -> None:
    if len(p) and (p.end >= np.datetime64(as_date(now), "D")).any():
        n = int((p.end >= np.datetime64(as_date(now), "D")).sum())
        raise FirewallBreach(f"{n} training position(s) end at/after now={now}: their realised gain was not known yet")


@dataclass
class RealisableGainModel:
    """Past-only model of the net return the INTENDED exit rule realises, given entry-time features."""
    rule: Any
    feature_names: tuple[str, ...]
    cfg: SelectionConfig = field(default_factory=SelectionConfig)
    fitted_as_of: str | None = None
    kind_levels: tuple[str, ...] = ()
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    beta_: np.ndarray | None = None
    resid_: np.ndarray | None = None
    n_support: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def policy(self) -> str:
        return policy_id(self.rule)

    def digest(self) -> str:
        return stable_hash({"policy": self.policy, "features": self.feature_names, "cfg": self.cfg.digest(), "as_of": self.fitted_as_of,
                            "beta": None if self.beta_ is None else [round(float(b), 10) for b in self.beta_]})

    def fit(self, train: EX.Paths, F: np.ndarray, now) -> "RealisableGainModel":
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid SelectionConfig: " + "; ".join(errs))
        _check_features(self.feature_names)
        _matured_only(train, now)
        F = np.asarray(F, float).reshape(len(train), -1)
        if F.shape[1] != len(self.feature_names):
            raise ValueError(f"F has {F.shape[1]} columns, feature_names has {len(self.feature_names)}")
        self.fitted_as_of = str(as_date(now))
        self.n_support = len(train)
        if len(train) < self.cfg.min_support:
            self.beta_ = None
            self.diagnostics = {"status": "abstain", "n": len(train)}
            return self
        y = self.rule.run(train).net                          # the INTENDED policy's realised outcome, nothing else
        ok = np.isfinite(y) & np.isfinite(F).all(1)
        order = np.argsort(train.week[ok], kind="stable")
        Fo, yo, ko = F[ok][order], y[ok][order], train.kind[ok][order]
        self.kind_levels = tuple(sorted(set(ko.tolist())))
        n_hold = max(10, int(round(self.cfg.holdout_share * len(yo))))
        fit_i, hold_i = np.arange(len(yo) - n_hold), np.arange(len(yo) - n_hold, len(yo))
        self._solve(Fo[fit_i], yo[fit_i], ko[fit_i])
        resid_hold = yo[hold_i] - self._predict_raw(Fo[hold_i], ko[hold_i])
        self._solve(Fo, yo, ko)                                # final coefficients use everything matured
        self.resid_ = np.sort(resid_hold)
        in_band = (yo >= self.cfg.lo) & (yo <= self.cfg.hi)
        self.diagnostics = {"status": "fitted", "n": int(len(yo)), "n_holdout": int(n_hold), "base_in_band": float(in_band.mean()),
                            "resid_sd": float(resid_hold.std()), "mean_realised": float(yo.mean())}
        return self

    def _solve(self, F: np.ndarray, y: np.ndarray, kinds: np.ndarray) -> None:
        self.mu_ = F.mean(0) if len(F) else np.zeros(F.shape[1])
        sd = F.std(0) if len(F) else np.ones(F.shape[1])
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        X = _design((F - self.mu_) / self.sd_, kinds, self.kind_levels)
        X = np.hstack([np.ones((len(X), 1)), X])
        pen = np.eye(X.shape[1]) * self.cfg.ridge
        pen[0, 0] = 0.0
        self.beta_ = np.linalg.solve(X.T @ X + pen, X.T @ y)

    def _predict_raw(self, F: np.ndarray, kinds: np.ndarray) -> np.ndarray:
        X = _design((np.asarray(F, float) - self.mu_) / self.sd_, kinds, self.kind_levels)
        return np.hstack([np.ones((len(X), 1)), X]) @ self.beta_

    def forecast(self, candidates: Sequence[str], F: np.ndarray, kinds: Sequence[str], as_of) -> list[GainForecast]:
        """Forecast each candidate's realisable gain. Refuses to forecast for a date at/before the fit (the model would then
        contain outcomes that were not known on that date)."""
        if self.fitted_as_of is None:
            raise ValueError("model not fitted")
        if as_date(as_of) < as_date(self.fitted_as_of):
            raise FirewallBreach(f"model fitted as of {self.fitted_as_of} cannot forecast for earlier date {as_of}")
        F = np.asarray(F, float).reshape(len(candidates), -1)
        kinds = np.asarray(list(kinds), object)
        dg = self.digest()
        if self.beta_ is None or self.resid_ is None:
            return [GainForecast(str(c), str(as_date(as_of)), self.policy, float("nan"), {}, 0.0, self.n_support, dg, basis="abstain")
                    for c in candidates]
        mean = self._predict_raw(F, kinds)
        out = []
        for c, m in zip(candidates, mean):
            draws = m + self.resid_
            qs = {float(q): float(np.quantile(draws, q)) for q in self.cfg.quantiles}
            p = float(((draws >= self.cfg.lo) & (draws <= self.cfg.hi)).mean())
            out.append(GainForecast(str(c), str(as_date(as_of)), self.policy, float(m), qs, p, self.n_support, dg))
        return out


def eligibility(fc: GainForecast, intended_policy: str, cfg: SelectionConfig = SelectionConfig()) -> Eligibility:
    """Checklist L for ONE name. Order matters: a wrong quantity or a stale policy is refused before any number is read."""
    errs = fc.validate() if math.isfinite(fc.mean) else []
    if fc.quantity != QUANTITY or fc.basis in FORBIDDEN_BASES:
        return Eligibility(fc.candidate, False, Reason.WRONG_QUANTITY, "; ".join(errs) or "not a realisable-gain forecast")
    if fc.policy_id != intended_policy:
        return Eligibility(fc.candidate, False, Reason.STALE_POLICY, f"forecast under {fc.policy_id}, trading under {intended_policy}")
    if fc.basis == "abstain" or not math.isfinite(fc.mean) or fc.n_support < cfg.min_support:
        return Eligibility(fc.candidate, False, Reason.UNSUPPORTED, f"model abstains (support {fc.n_support} < {cfg.min_support})")
    if errs:
        return Eligibility(fc.candidate, False, Reason.INVALID, "; ".join(errs))
    pt = fc.median if cfg.point == "median" else fc.mean
    if pt < cfg.lo:
        return Eligibility(fc.candidate, False, Reason.BELOW_BAND, f"predicted {pt:+.2%} < {cfg.lo:+.0%}", pt, fc.p_in_band)
    if pt > cfg.hi:
        return Eligibility(fc.candidate, False, Reason.ABOVE_BAND, f"predicted {pt:+.2%} > {cfg.hi:+.0%}", pt, fc.p_in_band)
    if fc.p_in_band < cfg.min_p_in_band:
        return Eligibility(fc.candidate, False, Reason.LOW_BAND_PROBABILITY, f"P(in band)={fc.p_in_band:.2f} < {cfg.min_p_in_band}", pt, fc.p_in_band)
    return Eligibility(fc.candidate, True, Reason.ELIGIBLE, f"predicted {pt:+.2%}, P(in band)={fc.p_in_band:.2f}", pt, fc.p_in_band)


@dataclass(frozen=True)
class SelectionReport:
    as_of: str
    intended_policy: str
    eligible: tuple[str, ...]            # ordered: most in-band mass first, then closeness to the weekly target
    decisions: tuple[Eligibility, ...]
    funnel: Mapping[str, int]
    config_digest: str

    def reason_of(self, candidate: str) -> Reason:
        return next(d.reason for d in self.decisions if d.candidate == candidate)


def select(forecasts: Sequence[GainForecast], now, intended_policy: str, cfg: SelectionConfig = SelectionConfig(),
           max_names: int | None = None) -> SelectionReport:
    """The public entry: apply the constraint to one day's forecasts. It takes NO outcome data at all, so no historical
    statistic can steer the choice (checklist L, third bullet). Forecasts dated after `now` are future information."""
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid SelectionConfig: " + "; ".join(errs))
    for fc in forecasts:
        if as_date(fc.as_of) > as_date(now):
            raise FirewallBreach(f"forecast for {fc.candidate} dated {fc.as_of} is after now={now}")
    seen: set[str] = set()
    decs = []
    for fc in forecasts:
        if fc.candidate in seen:
            raise ValueError(f"duplicate forecast for {fc.candidate}")
        seen.add(fc.candidate)
        decs.append(eligibility(fc, intended_policy, cfg))
    ok = [d for d in decs if d.eligible]
    ok.sort(key=lambda d: (-(d.p_in_band or 0.0), abs((d.point or 0.0) - EX.WEEKLY_TARGET), d.candidate))
    if max_names is not None:
        ok = ok[:max_names]
    funnel = {r.value: sum(d.reason == r for d in decs) for r in Reason}
    funnel["selected"] = len(ok)
    return SelectionReport(str(as_date(now)), intended_policy, tuple(d.candidate for d in ok), tuple(decs), funnel, cfg.digest())


def apply_to_positions(positions: pd.DataFrame, report: SelectionReport, key: str = "ticker") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hook for engine.research.two_stage: keep only DayDecision positions the constraint made eligible; return (kept, dropped
    with reasons). A position with no forecast at all is dropped as UNSUPPORTED, never waved through."""
    if positions is None or positions.empty:
        return positions, pd.DataFrame(columns=[key, "reason"])
    keys = positions[key].astype(str) if key in positions.columns else positions.index.astype(str).to_series(index=positions.index)
    elig = set(report.eligible)
    reasons = {d.candidate: d.reason.value for d in report.decisions}
    keep = keys.isin(elig).to_numpy()
    dropped = pd.DataFrame({key: keys[~keep].to_numpy(), "reason": [reasons.get(k, Reason.UNSUPPORTED.value) for k in keys[~keep]]})
    return positions[keep], dropped


# ------------------------------------------------------------------------------------------------ how well is the band found?
@dataclass(frozen=True)
class IdentificationRow:
    period: str
    n_candidates: int
    n_selected: int
    base_rate: float                    # share of ALL candidates whose realised gain landed in band
    precision: float                    # share of SELECTED that landed in band
    precision_lo: float
    precision_hi: float
    recall: float
    lift: float


def identification_curve(forecasts: Sequence[GainForecast], realised: Mapping[str, float], realised_at: Mapping[str, str], now,
                         intended_policy: str, cfg: SelectionConfig = SelectionConfig(), freq: str = "Q") -> pd.DataFrame:
    """Research-side readout (matured outcomes only): per period, does the constraint pick in-band names more often than the
    candidate base rate? Every forecast counts - one whose outcome has not matured by `now` is left out WITH a count, never
    silently. A curve rising over periods is the evidence that identification improves; a flat one says it does not."""
    rows, pending = [], 0
    recs = []
    for fc in forecasts:
        if fc.candidate not in realised or fc.candidate not in realised_at or as_date(realised_at[fc.candidate]) >= as_date(now):
            pending += 1
            continue
        e = eligibility(fc, intended_policy, cfg)
        r = float(realised[fc.candidate])
        recs.append((pd.Timestamp(as_date(fc.as_of)).to_period(freq), e.eligible, cfg.lo <= r <= cfg.hi))
    if not recs:
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(IdentificationRow)] + ["pending"])
    df = pd.DataFrame(recs, columns=["period", "sel", "hit"])
    for per, g in df.groupby("period", sort=True):
        ns, nh = int(g.sel.sum()), int((g.sel & g.hit).sum())
        base = float(g.hit.mean())
        lo, hi = wilson(nh, ns) if ns else (float("nan"), float("nan"))
        prec = nh / ns if ns else float("nan")
        rec = nh / int(g.hit.sum()) if g.hit.sum() else float("nan")
        rows.append(IdentificationRow(str(per), len(g), ns, base, prec, lo, hi, rec, prec / base if ns and base > 0 else float("nan")))
    out = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    out["pending"] = pending
    return out


def improvement_trend(curve: pd.DataFrame) -> dict:
    """Least-squares slope of per-period lift (weighted by selections). Reports the slope with its standard error so a trend
    built on three noisy quarters is not read as progress."""
    c = curve.dropna(subset=["lift"]) if len(curve) else curve
    if len(c) < 3:
        return {"n_periods": int(len(c)), "slope": None, "se": None, "improving": None}
    x = np.arange(len(c), dtype=float)
    w = np.maximum(c["n_selected"].to_numpy(float), 1.0)
    y = c["lift"].to_numpy(float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxx = float((w * (x - xm) ** 2).sum())
    slope = float((w * (x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + slope * (x - xm))
    se = float(math.sqrt(max((w * resid ** 2).sum() / max(len(c) - 2, 1), 0.0) / sxx))
    return {"n_periods": int(len(c)), "slope": slope, "se": se, "improving": bool(slope - 1.64 * se > 0)}


def basis_audit(forecasts: Sequence[GainForecast], p_up: Mapping[str, float] | None = None, vol: Mapping[str, float] | None = None,
                intended_policy: str = "", cfg: SelectionConfig = SelectionConfig()) -> dict:
    """Proves the constraint is not a relabelled P(up) or volatility screen: among names with the highest P(up) / volatility,
    how many are eligible, and what is the rank correlation of eligibility with each? A correlation near 1 means the band
    constraint is doing nothing the forbidden basis would not do."""
    el = {fc.candidate: eligibility(fc, intended_policy or fc.policy_id, cfg).eligible for fc in forecasts}
    out: dict[str, Any] = {"n": len(el), "n_eligible": int(sum(el.values()))}
    for name, m in (("p_up", p_up), ("volatility", vol)):
        if not m:
            out[name] = None
            continue
        keys = [k for k in el if k in m]
        if len(keys) < 3:
            out[name] = None
            continue
        a = pd.Series([float(m[k]) for k in keys]).rank().to_numpy()
        b = np.array([float(el[k]) for k in keys])
        corr = float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 and a.std() > 0 else float("nan")
        top = sorted(keys, key=lambda k: -float(m[k]))[: max(1, len(keys) // 5)]
        out[name] = {"rank_corr_with_eligibility": corr, "top_quintile_eligible_share": float(np.mean([el[k] for k in top]))}
    return out


def entry_features(p: EX.Paths, extra: Mapping[str, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Features known at the deciding close from a Paths block: daily vol and ATR, plus caller columns (e.g. m_* market context).
    The gap into the entry open is deliberately NOT a feature: the decision is taken at the prior close and fills at that open,
    so the gap is unknown when the name is chosen (rule 3)."""
    cols = {"vol": p.vol, "atr": p.atr}
    for k, v in (extra or {}).items():
        cols[k] = np.asarray(v, float).reshape(-1)
    names = tuple(cols)
    _check_features(names)
    return np.stack([cols[k] for k in names], 1) if len(p) else np.zeros((0, len(names))), names
