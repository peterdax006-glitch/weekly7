"""Bible Phase 16 (Stop / loss engine), gap-risk part: the honest answer to "losers never worse than -20%".

The first gap model in stops.py (per-ticker 95% quantile of past gaps) was miscalibrated out of sample: 11-16% of
nights beat its 95% tail. This module replaces it with a CONDITIONAL model:
  * event-aware: an earnings-type filing expected inside the holding week (predicted point-in-time from the ticker's own
    filing rhythm; the realised filing is only used for diagnostics, never as a model input);
  * volatility-regime-aware: gaps are standardised by the position's trailing ATR, and a separate tail is kept for weeks
    when the cross-section is in a high-volatility regime;
  * peaks-over-threshold (generalised Pareto, probability-weighted moments, shrunk shape) for the far tail, where the
    empirical quantile has no data;
  * conformal-style calibration on the most recent training weeks (a scale multiplier), then a walk-forward CHECK of
    coverage per era against the old model.
`loss_cap_analysis` then asks what would make P(one position ends worse than -20%) <= 1% and answers with avoidance
rules (event weeks, model tail probability), reported per era. Position size cannot lower that probability - it only
lowers what a breach costs the portfolio - and the report says so.
No look-ahead: every fit sees only positions whose last bar is on or before its `as_of`; calendar use is point-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .exits import ExitSpec, Paths, _week_codes, run_exit
from .stops import MAX_DIST, MIN_DIST, adverse_gaps, wilson

XI_MAX, XI_PRIOR, N0_SHRINK = 0.7, 0.25, 150


# ------------------------------------------------------------------ event calendar
class EventCalendar:
    """Filing calendar keyed by ticker. A filing accepted before 09:30 ET moves that day's open; any later filing moves
    the NEXT session's open. Only kinds in `kinds` count (default: earnings-release 8-Ks)."""

    def __init__(self, events: pd.DataFrame, kinds=("EARN",)):
        e = events[events["kind"].isin(kinds)]
        if len(e) == 0:
            self.by_ticker = {}
            return
        et = pd.to_datetime(e["accepted"], utc=True).dt.tz_convert("America/New_York")
        day = et.dt.normalize().dt.tz_localize(None).values.astype("datetime64[D]")
        before_open = (et.dt.hour * 60 + et.dt.minute).values < 9 * 60 + 30
        eff = np.where(before_open, np.busday_offset(day, 0, roll="forward"), np.busday_offset(day, 1, roll="forward"))
        df = pd.DataFrame({"t": e["ticker"].values, "d": eff}).drop_duplicates()
        self.by_ticker = {t: np.sort(g["d"].values.astype("datetime64[D]")) for t, g in df.groupby("t")}

    def realized(self, p: Paths) -> np.ndarray:
        """(N, D) True where an event filing moved that session's open. DIAGNOSTIC ONLY: it uses the filing itself."""
        out = np.zeros((len(p), p.D), bool)
        for i, (t, w) in enumerate(zip(p.ticker, p.week)):
            arr = self.by_ticker.get(t)
            if arr is None:
                continue
            days = np.busday_offset(w, np.arange(p.D))
            out[i] = np.isin(days, arr)
        return out

    def expected_in_week(self, p: Paths, tol_days: int = 6, min_events: int = 3) -> np.ndarray:
        """Point-in-time forecast that an event lands in position i's week: extrapolate the ticker's own recent filing
        interval from filings dated strictly BEFORE the entry date."""
        out = np.zeros(len(p), bool)
        for i, (t, w) in enumerate(zip(p.ticker, p.week)):
            arr = self.by_ticker.get(t)
            if arr is None:
                continue
            hist = arr[arr < w][-6:]
            if len(hist) < min_events:
                continue
            I = float(np.median(np.diff(hist).astype(float)))
            if not 30 <= I <= 200:
                continue
            last = hist[-1]
            k = max(1, int(np.ceil(((w - last).astype(float) - tol_days) / I)))
            nxt = float((last - w).astype(float)) + k * I          # days from entry to the predicted event
            out[i] = -tol_days <= nxt <= (p.D - 1) + tol_days
        return out


# ------------------------------------------------------------------ extreme-value tail
def fit_gpd(exc: np.ndarray, xi_prior: float = XI_PRIOR, n0: int = N0_SHRINK) -> tuple[float, float]:
    """Generalised Pareto (xi, beta) for threshold excesses by probability-weighted moments (Hosking-Wallis), with the
    shape shrunk toward `xi_prior` by n/(n+n0) so a thin sample cannot claim a light or absurdly heavy tail."""
    x = np.sort(np.asarray(exc, float))
    n = len(x)
    if n < 10 or x.mean() <= 0:
        return xi_prior, max(float(x.mean()) if n else 0.01, 1e-4)
    a0 = x.mean()
    a1 = np.mean(x * (n - np.arange(1, n + 1)) / (n - 1))
    den = a0 - 2 * a1
    xi_hat = 2 - a0 / den if den > 1e-12 else xi_prior
    w = n / (n + n0)
    xi = float(np.clip(w * xi_hat + (1 - w) * xi_prior, 0.0, XI_MAX))
    beta = a0 * (1 - xi)                                          # keeps the fitted mean excess equal to the sample's
    return xi, float(max(beta, 1e-6))


def gpd_sf(z: np.ndarray, u: float, xi: float, beta: float, lam: float) -> np.ndarray:
    """P(Z > z) for z above the threshold u; lam is P(Z > u). Below u the answer is lam (a floor)."""
    e = np.maximum(np.asarray(z, float) - u, 0.0)
    tail = np.exp(-e / beta) if xi < 1e-9 else np.power(1 + xi * e / beta, -1 / xi)
    return lam * tail


def gpd_quantile(t: float, u: float, xi: float, beta: float, lam: float) -> float:
    """z such that P(Z > z) = t (t < lam)."""
    r = t / lam
    return u + (beta * (-np.log(r)) if xi < 1e-9 else beta / xi * (r ** (-xi) - 1))


# ------------------------------------------------------------------ conditional model
@dataclass
class _Cls:
    u: float
    xi: float
    beta: float
    lam: float
    n: int


@dataclass
class ConditionalGapModel:
    """Adverse overnight gap tail conditional on (expected event in week, high-volatility regime), in ATR units."""
    calendar: EventCalendar | None = None
    q0: float = 0.90              # threshold quantile of standardised gaps
    min_cls: int = 400            # nights needed for a class of its own, else it borrows the pooled tail
    atr_floor: float = 0.005
    hv_quantile: float = 0.75
    calibrate: bool = True
    calib_frac: float = 0.2
    calib_prob: float = 0.95
    kappa: float = 1.0
    hv_cut: float = np.inf
    cls: dict = field(default_factory=dict)
    q: float = 0.95               # GapModel-compatible attribute used by tail()

    # -- features
    def _scale(self, p):
        return np.maximum(p.atr, self.atr_floor)

    def _week_regime(self, p) -> np.ndarray:
        """Cross-sectional median ATR of the position's own week (positions in one week are all known at entry)."""
        codes, W = _week_codes(p)
        med = pd.Series(p.atr).groupby(codes).transform("median").values
        return med

    def _keys(self, p):
        ev = self.calendar.expected_in_week(p) if self.calendar is not None else np.zeros(len(p), bool)
        hv = self._week_regime(p) > self.hv_cut
        return ev, hv

    def _cls_for(self, ev, hv):
        for k in ((ev, hv), (ev, None), (None, None)):
            if k in self.cls:
                return self.cls[k]
        raise RuntimeError("model not fitted")

    # -- fit
    def fit(self, train: Paths) -> "ConditionalGapModel":
        m = ConditionalGapModel(self.calendar, self.q0, self.min_cls, self.atr_floor, self.hv_quantile, self.calibrate,
                                self.calib_frac, self.calib_prob)
        if len(train) == 0:
            m.cls = {(None, None): _Cls(0.01, XI_PRIOR, 0.01, 1 - m.q0, 0)}
            return m
        if m.calibrate and len(np.unique(train.week)) >= 40:
            weeks = np.unique(train.week)
            cut = weeks[int(len(weeks) * (1 - m.calib_frac))]
            early, late = train.take(np.flatnonzero(train.week < cut)), train.take(np.flatnonzero(train.week >= cut))
            base = m._fit_core(early)
            base.kappa = 1.0
            target = 1 - m.calib_prob
            grid = np.linspace(0.5, 4.0, 36)
            g = adverse_gaps(late)
            q = base.night_quantile(late, m.calib_prob)
            rates = np.array([(g > k * q[:, None]).mean() for k in grid])
            m._fit_core(train)
            m.kappa = float(grid[np.argmin(np.abs(rates - target))])
            return m
        return m._fit_core(train)

    def _fit_core(self, train: Paths) -> "ConditionalGapModel":
        wk = self._week_regime(train)
        self.hv_cut = float(np.quantile(wk, self.hv_quantile))
        ev, hv = self._keys(train)
        z = adverse_gaps(train) / self._scale(train)[:, None]
        cls = {}
        pooled = self._fit_class(z.ravel(), None)
        cls[(None, None)] = pooled
        for e in (False, True):
            me = ev == e
            if me.any():
                ce = self._fit_class(z[me].ravel(), pooled)
                cls[(e, None)] = ce
                for h in (False, True):
                    mm = me & (hv == h)
                    if mm.sum() * z.shape[1] >= self.min_cls:
                        cls[(e, h)] = self._fit_class(z[mm].ravel(), ce)
        self.cls = cls
        return self

    def _fit_class(self, z: np.ndarray, parent: _Cls | None) -> _Cls:
        if len(z) < self.min_cls and parent is not None:
            return parent
        u = float(np.quantile(z, self.q0))
        exc = z[z > u] - u
        xi, beta = fit_gpd(exc, parent.xi if parent else XI_PRIOR)
        return _Cls(u, xi, beta, float((z > u).mean()) or (1 - self.q0), int(len(z)))

    # -- use
    def _params(self, p):
        ev, hv = self._keys(p)
        return [self._cls_for(bool(e), bool(h)) for e, h in zip(ev, hv)]

    def p_night(self, p: Paths, level) -> np.ndarray:
        """(N,) P(adverse gap on one night > level), level a scalar or (N,) fraction of the previous close."""
        lv = np.broadcast_to(np.asarray(level, float), (len(p),))
        z = lv / (self._scale(p) * self.kappa)
        return np.array([gpd_sf(zi, c.u, c.xi, c.beta, c.lam) for zi, c in zip(z, self._params(p))], float)

    def p_position(self, p: Paths, level) -> np.ndarray:
        """(N,) P(some night of the holding week gaps beyond `level`)."""
        return 1 - (1 - self.p_night(p, level)) ** p.D

    def night_quantile(self, p: Paths, prob: float) -> np.ndarray:
        """(N,) gap size exceeded on a single night with probability 1 - prob."""
        t = 1 - prob
        out = np.empty(len(p))
        for i, (s, c) in enumerate(zip(self._scale(p), self._params(p))):
            zq = gpd_quantile(t, c.u, c.xi, c.beta, c.lam) if t < c.lam else c.u
            out[i] = s * self.kappa * zq
        return out

    def tail(self, p: Paths) -> np.ndarray:
        """GapModel-compatible: the model's 95% single-night gap, so stop rules/filters can use this model unchanged."""
        return self.night_quantile(p, self.q)


def conditional_gap_factory(calendar: EventCalendar | None, **kw) -> Callable[[Paths], ConditionalGapModel]:
    """`fit_fn` for stops.StopRule.gap_model_fn: fits the conditional model on training positions."""
    return lambda train: ConditionalGapModel(calendar, **kw).fit(train)


# ------------------------------------------------------------------ walk-forward coverage
def _blocks(paths: Paths, min_train_weeks: int, block_weeks: int):
    weeks = np.unique(paths.week)
    for s in range(min_train_weeks, len(weeks), block_weeks):
        blk = weeks[s:s + block_weeks]
        yield blk, weeks[s] - np.timedelta64(1, "D")


def walk_forward_coverage(paths: Paths, factories: dict, probs=(0.95, 0.99), min_train_weeks: int = 104,
                          block_weeks: int = 26, label_fn: Callable[[Paths], np.ndarray] | None = None) -> pd.DataFrame:
    """Per calendar year and model: how often does a night's adverse gap exceed the model's `prob` quantile, fitted only
    on earlier finished weeks? `label_fn(test)` swaps the per-year grouping for any position label (e.g. expected-event
    week or not: an unconditional model can be right on average and wrong in exactly the weeks that matter). A calibrated model shows exceed_rate ~ 1 - prob in every era. `factories[name](train,
    prob)` returns a function p -> (N,) quantile. Nights within a crash week are correlated, so `ci_*` (plain
    binomial) is optimistic; read `ratio` across eras rather than one interval."""
    recs = []
    for blk, as_of in _blocks(paths, min_train_weeks, block_weeks):
        train = paths.until(as_of)
        test = paths.take(np.flatnonzero(np.isin(paths.week, blk)))
        if len(train) == 0 or len(test) == 0:
            continue
        g = adverse_gaps(test)
        yr = test.week.astype("datetime64[Y]").astype(int) + 1970 if label_fn is None else np.asarray(label_fn(test))
        for name, fac in factories.items():
            for pr in probs:
                exc = g > fac(train, pr)(test)[:, None]
                for y in np.unique(yr):
                    m = yr == y
                    recs.append((name, pr, y.item() if hasattr(y, 'item') else y, int(exc[m].sum()), int(exc[m].size)))
    if not recs:
        return pd.DataFrame(columns=["model", "prob", "era", "exceed_rate", "target", "ratio", "nights", "ci_lo", "ci_hi"])
    d = pd.DataFrame(recs, columns=["model", "prob", "era", "k", "n"]).groupby(["model", "prob", "era"], as_index=False).sum()
    d["exceed_rate"] = d.k / d.n
    d["target"] = 1 - d.prob
    d["ratio"] = d.exceed_rate / d.target
    ci = [wilson(int(k), int(n)) for k, n in zip(d.k, d.n)]
    d["ci_lo"], d["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return d.rename(columns={"n": "nights"}).drop(columns="k")


def coverage_summary(cov: pd.DataFrame) -> pd.DataFrame:
    """Per model and probability: night-weighted exceed rate, worst era ratio and share of eras within 1.5x of target."""
    rows = []
    for (m, pr), g in cov.groupby(["model", "prob"]):
        rows.append({"model": m, "prob": pr, "exceed_rate": float((g.exceed_rate * g.nights).sum() / g.nights.sum()), "target": 1 - pr,
                     "worst_era_ratio": float(g.ratio.max()), "eras_within_1.5x": float((g.ratio <= 1.5).mean()), "eras": len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ loss-cap analysis
@dataclass
class LossCap:
    p_model: np.ndarray       # model P(position breaches the cap), fitted only on earlier weeks
    event: np.ndarray         # expected-event-week flag
    realized: np.ndarray      # position actually ended at or below -cap
    era: np.ndarray
    kind: np.ndarray
    net: np.ndarray
    cap: float

    def __len__(self):
        return len(self.realized)


def loss_cap_analysis(paths: Paths, calendar: EventCalendar | None, cap: float = 0.20, stop_k: float = 3.0,
                      min_train_weeks: int = 104, block_weeks: int = 26, **model_kw) -> LossCap:
    """Walk-forward: for each test block fit the conditional model on earlier weeks, give every position the model
    probability that a gap breaches the cap from where its stop sits (gap > cap - stop distance), and record what
    really happened under an ATR stop (fills at the open when a gap jumps the stop)."""
    P, EV, R, ERA, KIND, NET = [], [], [], [], [], []
    for blk, as_of in _blocks(paths, min_train_weeks, block_weeks):
        train = paths.until(as_of)
        test = paths.take(np.flatnonzero(np.isin(paths.week, blk)))
        if len(train) == 0 or len(test) == 0:
            continue
        m = ConditionalGapModel(calendar, **model_kw).fit(train)
        dist = np.clip(stop_k * test.atr, MIN_DIST, MAX_DIST)
        lvl = np.maximum(cap - dist, 0.01)
        res = run_exit(test, ExitSpec(), stop_dist=dist)
        P.append(m.p_position(test, lvl))
        EV.append(m._keys(test)[0])
        R.append(res.net <= -cap)
        ERA.append(test.week.astype("datetime64[Y]").astype(int) + 1970)
        KIND.append(test.kind)
        NET.append(res.net)
    if not P:
        z = np.zeros(0)
        return LossCap(z, z.astype(bool), z.astype(bool), z.astype(int), z.astype(object), z, cap)
    c = np.concatenate
    return LossCap(c(P), c(EV), c(R), c(ERA), c(KIND), c(NET), cap)


def policy_table(lc: LossCap, taus=(0.10, 0.05, 0.03, 0.02, 0.01), target: float = 0.01) -> pd.DataFrame:
    """Realised P(position <= -cap) among the positions each avoidance rule would keep, per era and pooled.
    `meets_target` is judged on the Wilson UPPER bound, not the point estimate."""
    if len(lc) == 0:
        return pd.DataFrame()
    pols = {"no_filter": np.ones(len(lc), bool), "avoid_event_weeks": ~lc.event}
    for t in taus:
        pols[f"model_p<={t}"] = lc.p_model <= t
        pols[f"avoid_events+model_p<={t}"] = (lc.p_model <= t) & ~lc.event
    rows = []
    for name, keep in pols.items():
        for era in [None, *sorted(set(lc.era))]:
            m = keep if era is None else keep & (lc.era == era)
            n = int(m.sum())
            if n == 0:
                rows.append({"policy": name, "era": "ALL" if era is None else int(era), "kept": 0, "kept_share": 0.0, "p_breach": np.nan,
                             "p_hi": np.nan, "model_mean_p": np.nan, "worst": np.nan, "meets_target": False})
                continue
            k = int(lc.realized[m].sum())
            lo, hi = wilson(k, n)
            rows.append({"policy": name, "era": "ALL" if era is None else int(era), "kept": n,
                         "kept_share": float(n / max(1, (lc.era == era).sum() if era is not None else len(lc))),
                         "p_breach": k / n, "p_hi": hi, "model_mean_p": float(lc.p_model[m].mean()),
                         "worst": float(lc.net[m].min()), "meets_target": bool(hi <= target)})
    return pd.DataFrame(rows)


def required_avoidance(lc: LossCap, target: float = 0.01) -> dict:
    """Most permissive model-probability cut whose pooled realised breach rate has upper bound <= target. `feasible`
    False means no cut on the model's ranking reaches the target with the data at hand."""
    order = np.argsort(lc.p_model)
    real = lc.realized[order].astype(float)
    n = np.arange(1, len(real) + 1)
    rate = np.cumsum(real) / n
    ok = [i for i in range(len(n)) if n[i] >= 200 and wilson(int(rate[i] * n[i]), int(n[i]))[1] <= target]
    if not ok:
        return {"feasible": False, "base_rate": float(lc.realized.mean()) if len(lc) else 0.0, "kept_share": 0.0}
    i = ok[-1]
    return {"feasible": True, "p_cut": float(lc.p_model[order][i]), "kept_share": float((i + 1) / len(n)), "p_breach": float(rate[i]),
            "p_hi": wilson(int(rate[i] * n[i]), int(n[i]))[1], "base_rate": float(lc.realized.mean())}


def weight_cap_for_portfolio_hit(max_hit: float = 0.03, cap: float = 0.20) -> float:
    """Sizing cannot lower P(position breach); it bounds the cost: weight * cap <= max_hit of the portfolio."""
    return max_hit / cap


def calibration_of_p(lc: LossCap, bins: int = 5) -> pd.DataFrame:
    """Predicted vs realised breach probability by predicted-risk bin (does the ranking carry information?)."""
    if len(lc) == 0:
        return pd.DataFrame()
    edges = np.unique(np.quantile(lc.p_model, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, lc.p_model, side="right") - 1, 0, max(len(edges) - 2, 0))
    return pd.DataFrame([{"bin": j, "n": int((b == j).sum()), "predicted": float(lc.p_model[b == j].mean()),
                          "realized": float(lc.realized[b == j].mean())} for j in np.unique(b)])
