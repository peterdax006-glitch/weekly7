"""Optimal-exit research and the exit independence rule (C68 checklists N and O; EXTENDS Bible phase 15, engine/exits.py). IMPLEMENTED -
NOT VALIDATED.

Checklist N asks "when should this position be sold?" and wants the answer learned BEFORE the exit point, from the past only. The
existing exit learner (engine.exits) chooses among fixed rule families; this module adds a STATE-BASED learned exit that plugs into
the same tournament as one more engine.exits.Rule (`LearnedExitRule`), so it is compared out of sample by exits.walk_forward with the
identical fill model, costs and tiered objective - never a second exit system.

At every close d the state of each open position is described only from bars 0..d (`state_features`): return so far, maximum
favourable/adverse excursion, drawdown from the peak, time since entry and time remaining, one-day momentum and momentum decay,
realised-vs-entry volatility (volatility decay), pattern-failure flags (pattern decay), caller context (market regime, sector
behaviour; m_* style columns), stock type (sector / stock-specific intercept), entry vol/ATR, and an opportunity-cost term. From
MATURED training positions only (every last bar strictly before `now`) it learns
    E[return from the next open to the planned end | hold]   (ridge; the "expected future return conditional on holding")
    P(continued upside of +u from the next open)            (logistic)
    P(reversal of -r from the next open)                    (logistic)
and a learned exit threshold, chosen on a later-in-time inner split with the exits.compare tiered objective. The policy sells at the
next open whenever E[hold] - opportunity cost < threshold. `TrajectoryModel` gives the expected return path and expected best exit
day AT ENTRY (before any future bar exists).

Checklist O: exits are decided by this learned policy ONLY. No input may name a prediction, target, expectation or calibration
tolerance (`ANCHOR` refuses them), the module never imports the calibration-target code, and `exit_independence_audit` re-runs any
exit procedure under several evaluation targets and reports any difference as a violation - a planted target-aware exit is caught."""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from engine import exits as EX
from engine.learning.core import FirewallBreach, as_date, stable_hash

SECTION = "C68 checklists N, O"
STATE_FEATURES = ("ret", "mfe", "mae", "dd_peak", "t_frac", "remaining", "mom1", "mom_decay", "vol_ratio", "pattern_fail",
                  "fail_seen", "vol", "atr")
ANCHOR = re.compile(r"(predict|expect|target|calib|toleran|within|_pp\b|pp_|forecast|fwd|future)", re.I)
R_DECIDED = EX.R_FAIL           # exits.REASONS code for "decided at a close, filled at the next open"


@dataclass(frozen=True)
class ExitResearchConfig:
    up: float = 0.03                    # continued upside: +3% above the next open before the planned end
    rev: float = 0.03                   # reversal: -3% below the next open before the planned end
    ridge: float = 1.0
    inner_share: float = 0.3            # later-in-time share of training weeks used to choose the threshold
    grid: int = 9                       # candidate thresholds (quantiles of predicted E[hold]) plus "never sell early"
    min_train: int = 60
    redeploy: bool = False              # can freed capital be re-used before the week ends? (weekly system: no)
    stop: float | None = None           # optional protective stop, fixed at entry (engine.stops learns distances)
    cost: EX.CostModel = EX.CostModel()

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.up < 1 or not 0 < self.rev < 1:
            errs.append("up and rev must be in (0,1)")
        if not 0.1 <= self.inner_share <= 0.5:
            errs.append("inner_share must be in [0.1, 0.5]")
        if self.grid < 2:
            errs.append("grid must be >= 2")
        if self.ridge <= 0:
            errs.append("ridge must be positive")
        if self.stop is not None and not 0 < self.stop < 1:
            errs.append("stop must be in (0,1)")
        return errs


def check_inputs(names: Sequence[str]) -> None:
    """Checklist O: an exit input may never be the prediction or its evaluation target - holding 'until the result matches the
    forecast' is exactly the behaviour the rule forbids, and it starts with such an input."""
    bad = [n for n in names if ANCHOR.search(str(n))]
    if bad:
        raise FirewallBreach(f"exit inputs {bad} refer to a prediction/target/calibration: exits must be target-blind")


def _context(p: EX.Paths, context: Mapping[str, np.ndarray] | None) -> tuple[np.ndarray, tuple[str, ...]]:
    ctx = dict(context or {})
    check_inputs(list(ctx))
    names = tuple(sorted(ctx))
    if not names:
        return np.zeros((len(p), 0)), ()
    return np.stack([np.asarray(ctx[k], float).reshape(len(p)) for k in names], 1), names


def state_features(p: EX.Paths, d: int, context: Mapping[str, np.ndarray] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """State of every position at the CLOSE of session d, from bars 0..d only (nothing later is sliced)."""
    if not 0 <= d < p.D:
        raise IndexError(f"session {d} outside 0..{p.D - 1}")
    o, h, l, c = p.o[:, :d + 1], p.h[:, :d + 1], p.l[:, :d + 1], p.c[:, :d + 1]
    entry = o[:, 0]
    cd = c[:, d]
    hi, lo = h.max(1), l.min(1)
    ret = cd / entry - 1
    prev = c[:, d - 1] if d > 0 else entry
    mom1 = cd / prev - 1
    base = np.concatenate([p.prev_close.reshape(-1, 1), c[:, :d]], 1)
    lr = np.log(c / base)
    realised = np.sqrt(np.mean(lr ** 2, 1))
    fail = p.fail[:, :d + 1] if p.fail is not None else np.zeros((len(p), d + 1), bool)
    cols = [ret, hi / entry - 1, lo / entry - 1, cd / hi - 1, np.full(len(p), (d + 1) / p.D), np.full(len(p), (p.D - 1 - d) / p.D),
            mom1, mom1 - ret / (d + 1), realised / np.maximum(p.vol, 1e-6), fail[:, d].astype(float), fail.any(1).astype(float), p.vol, p.atr]
    X = np.stack(cols, 1)
    C, cn = _context(p, context)
    return np.hstack([X, C]), STATE_FEATURES + cn


def hold_labels(p: EX.Paths, d: int, cfg: ExitResearchConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRAINING labels at close d (matured positions only): return from the next open to the planned end, whether +up is reached
    after the next open, whether -rev is reached. Undefined on the last session (nothing left to hold)."""
    if d >= p.D - 1:
        raise IndexError("no holding period after the last session")
    nxt = p.o[:, d + 1]
    hold = p.c[:, -1] / nxt - 1
    up = p.h[:, d + 1:].max(1) / nxt - 1 >= cfg.up
    rev = p.l[:, d + 1:].min(1) / nxt - 1 <= -cfg.rev
    return hold, up, rev


def _matured(p: EX.Paths, now) -> None:
    if now is not None and len(p) and (p.end >= np.datetime64(as_date(now), "D")).any():
        raise FirewallBreach(f"{int((p.end >= np.datetime64(as_date(now), 'D')).sum())} training position(s) had not finished before now={now}")


def _kinds_onehot(kinds: np.ndarray, levels: Sequence[str]) -> np.ndarray:
    return np.stack([(kinds == k).astype(float) for k in levels], 1) if levels else np.zeros((len(kinds), 0))


class _Logit:
    """Logistic probability with a constant fallback when training has one class (deterministic lbfgs)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(int)
        self.const = float(y.mean()) if len(y) else 0.5
        self.m = None
        if len(y) >= 10 and 0 < y.sum() < len(y):
            self.m = LogisticRegression(C=1.0, max_iter=500).fit(X, y)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(X)[:, 1] if self.m is not None else np.full(len(X), self.const)


@dataclass
class ExitValueModel:
    """E[hold], P(upside), P(reversal) as functions of the close-d state, fitted on matured positions at every d < D-1."""
    cfg: ExitResearchConfig = field(default_factory=ExitResearchConfig)
    names: tuple[str, ...] = ()
    kind_levels: tuple[str, ...] = ()
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    beta_: np.ndarray | None = None
    p_up_: Any = None
    p_rev_: Any = None
    opp_rate_: dict = field(default_factory=dict)
    n_rows: int = 0

    def _X(self, X: np.ndarray, kinds: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        return np.hstack([np.ones((len(Z), 1)), Z, _kinds_onehot(kinds, self.kind_levels)])

    def fit(self, train: EX.Paths, context: Mapping[str, np.ndarray] | None = None, now=None) -> "ExitValueModel":
        _matured(train, now)
        if train.D < 2:
            raise ValueError("an exit needs at least two sessions")
        rows: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ups: list[np.ndarray] = []
        revs: list[np.ndarray] = []
        ks: list[np.ndarray] = []
        for d in range(train.D - 1):
            X, self.names = state_features(train, d, context)
            hold, up, rev = hold_labels(train, d, self.cfg)
            rows.append(X)
            ys.append(hold)
            ups.append(up)
            revs.append(rev)
            ks.append(train.kind)
        X, y, U, R, K = np.vstack(rows), np.concatenate(ys), np.concatenate(ups), np.concatenate(revs), np.concatenate(ks)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y, U, R, K = X[ok], y[ok], U[ok], R[ok], K[ok]
        self.n_rows = len(y)
        self.kind_levels = tuple(sorted(set(K.tolist())))
        self.mu_ = X.mean(0)
        sd = X.std(0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        A = self._X(X, K)
        pen = np.eye(A.shape[1]) * self.cfg.ridge
        pen[0, 0] = 0.0
        self.beta_ = np.linalg.solve(A.T @ A + pen, A.T @ y)
        self.p_up_, self.p_rev_ = _Logit(A[:, 1:], U), _Logit(A[:, 1:], R)
        week_net = train.c[:, -1] / train.o[:, 0] - 1
        for k in self.kind_levels:                         # what a fresh slot earns per session, per type (opportunity cost)
            m = train.kind == k
            self.opp_rate_[k] = float(week_net[m].mean() / train.D) if m.any() else 0.0
        return self

    def predict(self, X: np.ndarray, kinds: np.ndarray) -> dict[str, np.ndarray]:
        A = self._X(X, np.asarray(kinds, object))
        return {"ev_hold": A @ self.beta_, "p_up": self.p_up_(A[:, 1:]), "p_rev": self.p_rev_(A[:, 1:])}

    def opportunity_cost(self, kinds: np.ndarray, sessions_left: int) -> np.ndarray:
        if not self.cfg.redeploy:
            return np.zeros(len(kinds))                    # freed cash waits for next week's entries: it earns nothing
        return np.array([self.opp_rate_.get(k, 0.0) for k in kinds]) * sessions_left

    def contributions(self, X: np.ndarray, kinds: np.ndarray, top: int = 3) -> list[list[tuple[str, float]]]:
        """Largest signed feature contributions to E[hold] per position (coefficient x standardised value)."""
        mu, sd, beta = self.mu_, self.sd_, self.beta_
        if mu is None or sd is None or beta is None:
            raise RuntimeError("the hold model is not fitted")
        Z = (X - mu) / sd
        contrib = Z * beta[1:1 + Z.shape[1]]
        out = []
        for row in contrib:
            idx = np.argsort(-np.abs(row))[:top]
            out.append([(self.names[i], float(row[i])) for i in idx])
        return out


@dataclass
class ExitTrace:
    """Every close decision of one simulation: E[hold], probabilities and whether the policy sold (NaN once closed)."""
    ev: np.ndarray
    p_up: np.ndarray
    p_rev: np.ndarray
    sell: np.ndarray
    threshold: float


def simulate(p: EX.Paths, model: ExitValueModel, threshold: float, context: Mapping[str, np.ndarray] | None = None,
             cfg: ExitResearchConfig | None = None) -> tuple[EX.ExitResult, ExitTrace]:
    """Run the learned policy with the exits.run_exit fill model: entry at the first open (+slippage), a close decision fills at the
    NEXT open (+slippage), an optional entry-fixed stop fills at its level or at a gapped open, the last bar exits at the close."""
    cfg = cfg or model.cfg
    n, D = p.o.shape
    if n == 0:
        z = np.zeros(0)
        return EX.ExitResult(z, z, z.astype(int), z.astype(int), z, D), ExitTrace(np.zeros((0, D)), np.zeros((0, D)), np.zeros((0, D)), np.zeros((0, D), bool), threshold)
    slip, fee = cfg.cost.slip_bps / 1e4, cfg.cost.fee_bps / 1e4
    entry = p.o[:, 0] * (1 + slip)
    lvl = entry * (1 - cfg.stop) if cfg.stop is not None else np.full(n, -np.inf)
    alive, pending = np.ones(n, bool), np.zeros(n, bool)
    px, day, reason, over = np.full(n, np.nan), np.full(n, D, int), np.full(n, EX.R_TIME, int), np.zeros(n)
    tr = ExitTrace(np.full((n, D), np.nan), np.full((n, D), np.nan), np.full((n, D), np.nan), np.zeros((n, D), bool), float(threshold))

    def fill(mask, price, d, code):
        px[mask] = price[mask]
        day[mask], reason[mask], alive[mask] = d + 1, code, False

    for d in range(D):
        o, l, c = p.o[:, d], p.l[:, d], p.c[:, d]
        fill(alive & pending, o * (1 - slip), d, R_DECIDED)
        gs = alive & (o <= lvl)
        over[gs] = np.maximum(0.0, (lvl - o * (1 - slip)) / entry)[gs]
        fill(gs, o * (1 - slip), d, EX.R_GAP_STOP)
        st = alive & (l <= lvl)
        fill(st, lvl * (1 - slip), d, EX.R_STOP)
        if d == D - 1:
            fill(alive, c * (1 - slip), d, EX.R_TIME)
            break
        X, _ = state_features(p, d, context)
        pr = model.predict(X, p.kind)
        opp = model.opportunity_cost(p.kind, D - 1 - d)
        sell = alive & (pr["ev_hold"] - opp < threshold)
        for arr, key in ((tr.ev, "ev_hold"), (tr.p_up, "p_up"), (tr.p_rev, "p_rev")):
            arr[alive, d] = pr[key][alive]
        tr.sell[:, d] = sell
        pending = sell
    gross = px / entry - 1
    return EX.ExitResult(gross - 2 * fee, gross, day, reason, over, D), tr


def _weeks(p: EX.Paths) -> tuple[np.ndarray, int]:
    u, codes = np.unique(p.week, return_inverse=True)
    return codes, len(u)


def choose_threshold(model: ExitValueModel, val: EX.Paths, context: Mapping[str, np.ndarray] | None = None) -> tuple[float, pd.DataFrame]:
    """Pick the threshold on a validation block by the exits tiered objective (band first, then risk, then consistency). The
    candidate set always contains -inf (never sell early = the week-end baseline), so the learned exit must beat holding."""
    if len(val) == 0:
        return -math.inf, pd.DataFrame()
    evs = []
    for d in range(val.D - 1):
        X, _ = state_features(val, d, context)
        evs.append(model.predict(X, val.kind)["ev_hold"])
    grid = [-math.inf] + sorted(set(float(q) for q in np.quantile(np.concatenate(evs), np.linspace(0.05, 0.5, model.cfg.grid))))
    codes, W = _weeks(val)
    mets, rows = [], []
    for t in grid:
        res, _ = simulate(val, model, t, context)
        m = EX.band_metrics(EX.week_table(res.net, codes, W))
        mets.append(m)
        rows.append({"threshold": t, **{k: v for k, v in m.items() if k != "weeks"}, "mean_trade": float(res.net.mean()),
                     "days": float(res.days.mean())})
    best = 0
    for i in range(1, len(grid)):
        if EX.compare(mets[i], mets[best]) > 0:
            best = i
    return grid[best], pd.DataFrame(rows)


@dataclass
class LearnedExitRule(EX.Rule):
    """The state-based learned exit as an engine.exits.Rule: exits.select_rule / learn / walk_forward can put it in the same
    tournament as the fixed families. fit() uses training positions only; the threshold comes from their later-in-time part."""
    cfg: ExitResearchConfig = field(default_factory=ExitResearchConfig)
    context_fn: Callable[[EX.Paths], Mapping[str, np.ndarray]] | None = None
    name: str = "learned_ev"
    family: str = "learned"
    model: ExitValueModel | None = None
    threshold: float = -math.inf
    selection_table: pd.DataFrame | None = None

    @property
    def policy_id(self) -> str:
        b = None if self.model is None or self.model.beta_ is None else [round(float(x), 10) for x in self.model.beta_]
        return "L" + stable_hash({"name": self.name, "cfg": dataclasses.asdict(self.cfg), "beta": b, "thr": self.threshold}, 14)

    def _ctx(self, p: EX.Paths):
        return self.context_fn(p) if self.context_fn else None

    def fit(self, train: EX.Paths, now=None) -> "LearnedExitRule":
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid ExitResearchConfig: " + "; ".join(errs))
        _matured(train, now)
        out = dataclasses.replace(self, model=None, threshold=-math.inf, selection_table=None)
        if len(train) < self.cfg.min_train or train.D < 2:
            return out                                     # abstain: behaves exactly like holding to the planned end
        weeks = np.unique(train.week)
        cut = weeks[int(len(weeks) * (1 - self.cfg.inner_share))] if len(weeks) >= 4 else weeks[-1] + np.timedelta64(1, "D")
        inner, val = train.take(np.flatnonzero(train.week < cut)), train.take(np.flatnonzero(train.week >= cut))
        if len(inner) >= self.cfg.min_train // 2 and len(val):
            m0 = ExitValueModel(self.cfg).fit(inner, self._ctx(inner))
            out.threshold, out.selection_table = choose_threshold(m0, val, self._ctx(val))
        out.model = ExitValueModel(self.cfg).fit(train, self._ctx(train))
        return out

    def run(self, p: EX.Paths) -> EX.ExitResult:
        if self.model is None:
            return EX.run_exit(p, EX.ExitSpec(), self.cfg.cost)
        return simulate(p, self.model, self.threshold, self._ctx(p), self.cfg)[0]

    def trace(self, p: EX.Paths) -> ExitTrace:
        if self.model is None:
            raise ValueError("unfitted learned exit has no decisions to trace")
        return simulate(p, self.model, self.threshold, self._ctx(p), self.cfg)[1]


def decide_at_close(rule: LearnedExitRule, p: EX.Paths, d: int, open_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Live-style single decision at the close of session d (fills at the next open): sell?, E[hold], probabilities and the top
    reasons. Uses bars 0..d only - identical whatever happens later (tested by scrambling the future)."""
    if rule.model is None:
        raise ValueError("learned exit not fitted")
    X, _ = state_features(p, d, rule._ctx(p))
    pr = rule.model.predict(X, p.kind)
    opp = rule.model.opportunity_cost(p.kind, p.D - 1 - d)
    sell = pr["ev_hold"] - opp < rule.threshold
    if open_mask is not None:
        sell = sell & np.asarray(open_mask, bool)
    why = rule.model.contributions(X, p.kind)
    return pd.DataFrame({"sell": sell, "ev_hold": pr["ev_hold"], "opportunity_cost": opp, "p_up": pr["p_up"], "p_rev": pr["p_rev"],
                         "threshold": rule.threshold, "why": [", ".join(f"{n}{v:+.3f}" for n, v in w) for w in why]})


# ------------------------------------------------------------------------------------------------ expected path at entry
@dataclass
class TrajectoryModel:
    """Expected cumulative return at every session close, predicted AT ENTRY from entry-known features (vol, ATR, type, context).
    `expected_best_exit` is the argmax of that path: the best exit estimated before any future bar exists (checklist N, last line)."""
    ridge: float = 1.0
    betas_: np.ndarray | None = None
    kind_levels: tuple[str, ...] = ()
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None

    def _X(self, p: EX.Paths, context) -> np.ndarray:
        C, _ = _context(p, context)
        F = np.hstack([np.stack([p.vol, p.atr], 1), C])
        return np.hstack([np.ones((len(p), 1)), (F - self.mu_) / self.sd_, _kinds_onehot(p.kind, self.kind_levels)])

    def fit(self, train: EX.Paths, context: Mapping[str, np.ndarray] | None = None, now=None) -> "TrajectoryModel":
        _matured(train, now)
        C, _ = _context(train, context)
        F = np.hstack([np.stack([train.vol, train.atr], 1), C])
        self.mu_, sd = F.mean(0), F.std(0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        self.kind_levels = tuple(sorted(set(train.kind.tolist())))
        A = self._X(train, context)
        Y = train.c / train.o[:, :1] - 1
        pen = np.eye(A.shape[1]) * self.ridge
        pen[0, 0] = 0.0
        self.betas_ = np.linalg.solve(A.T @ A + pen, A.T @ Y)
        return self

    def path(self, p: EX.Paths, context: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        return self._X(p, context) @ self.betas_

    def expected_best_exit(self, p: EX.Paths, context: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
        P = self.path(p, context)
        best = P.argmax(1)
        return pd.DataFrame({"best_day": best + 1, "expected_at_best": P[np.arange(len(P)), best], "expected_at_end": P[:, -1]})


# ------------------------------------------------------------------------------------------------ path model for checklist-A expectations
TRAJECTORY_FIELDS = ("predicted_return", "distribution", "time_to_peak", "exit_window", "holding_period", "mfe", "mae",
                     "predicted_volatility", "prob_distribution", "uncertainty", "alternatives")   # = expectations.TRAJECTORY_FIELDS
PROB_EDGES = (-1.0, -0.05, 0.0, 0.05, 0.10, 0.20, 10.0)
QUANTS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _ridge(A: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    pen = np.eye(A.shape[1]) * lam
    pen[0, 0] = 0.0
    inv = np.linalg.inv(A.T @ A + pen)
    return inv @ A.T @ y, inv


@dataclass
class PathModel:
    """Past-only forecaster of the trajectory fields a checklist-A expectation needs (engine.research.expectations
    `expectations_from_day(day, path_model, ...)` calls `path_model(row, subject)`). It FORECASTS what the learned exit policy will
    realise - return and its distribution, holding period and exit window, time to peak, MFE/MAE, daily volatility - from
    entry-known features; it never decides an exit, and neither it nor the exit policy sees the +-1pp target.
    Fit on matured positions only; bind today's entry features with `bind(entries, now)`; long positions only (Paths are long)."""
    rule: LearnedExitRule
    ridge: float = 1.0
    fitted_as_of: str | None = None
    kind_levels: tuple[str, ...] = ()
    ctx_names: tuple[str, ...] = ()
    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None
    coef_: dict = field(default_factory=dict)
    inv_: np.ndarray | None = None
    resid_: np.ndarray | None = None
    days_resid_: np.ndarray | None = None
    D: int = 0
    entries: dict = field(default_factory=dict)
    bound_as_of: str | None = None

    def _A(self, F: np.ndarray, kinds: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(F), 1)), (F - self.mu_) / self.sd_, _kinds_onehot(np.asarray(kinds, object), self.kind_levels)])

    def _F(self, vol, atr, ctx: Mapping[str, Any]) -> np.ndarray:
        return np.array([[float(vol), float(atr)] + [float(ctx.get(k, 0.0)) for k in self.ctx_names]])

    def fit(self, train: EX.Paths, now, context: Mapping[str, np.ndarray] | None = None) -> "PathModel":
        _matured(train, now)
        if len(train) < 20:
            raise ValueError(f"only {len(train)} matured positions: too few to forecast a path")
        C, self.ctx_names = _context(train, context)
        F = np.hstack([np.stack([train.vol, train.atr], 1), C])
        self.mu_, sd = F.mean(0), F.std(0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        self.kind_levels = tuple(sorted(set(train.kind.tolist())))
        self.D = train.D
        res = self.rule.run(train)                                   # what the LEARNED exit realised: the forecast target
        entry = train.o[:, :1]
        cum = train.c / entry - 1
        base = np.concatenate([train.prev_close.reshape(-1, 1), train.c[:, :-1]], 1)
        targets = {"ret": res.net, "days": res.days.astype(float), "mfe": train.h.max(1) / entry[:, 0] - 1,
                   "mae": train.l.min(1) / entry[:, 0] - 1, "vol": np.log(np.maximum(np.std(train.c / base - 1, 1), 1e-5))}
        targets.update({f"cum{d}": cum[:, d] for d in range(train.D)})
        A = self._A(F, train.kind)
        for k, y in targets.items():
            self.coef_[k], self.inv_ = _ridge(A, y, self.ridge)
        self.resid_ = np.sort(res.net - A @ self.coef_["ret"])
        self.days_resid_ = np.sort(res.days - A @ self.coef_["days"])
        self.fitted_as_of = str(as_date(now))
        return self

    def bind(self, entries: Mapping[str, Mapping[str, Any]], now) -> "PathModel":
        """Entry features per subject known at the deciding close {subject: {vol, atr, kind, <context...>}}. A model fitted AFTER
        `now` would contain outcomes unknown at `now`: refused."""
        if self.fitted_as_of is None:
            raise ValueError("path model not fitted")
        if as_date(now) < as_date(self.fitted_as_of):
            raise FirewallBreach(f"path model fitted as of {self.fitted_as_of} cannot serve a decision on {now}")
        check_inputs([k for e in entries.values() for k in e if k not in ("vol", "atr", "kind")])
        self.entries = {str(k): dict(v) for k, v in entries.items()}
        self.bound_as_of = str(as_date(now))
        return self

    def forecast(self, vol: float, atr: float, kind: str, ctx: Mapping[str, Any] | None = None) -> dict:
        resid, days_resid, inv = self.resid_, self.days_resid_, self.inv_
        if resid is None or days_resid is None or inv is None:
            raise RuntimeError("the path model is not fitted")
        A = self._A(self._F(vol, atr, ctx or {}), np.array([kind], object))
        pr = {k: float((A @ b)[0]) for k, b in self.coef_.items()}
        ret = pr["ret"]
        draws = ret + resid
        qv = np.maximum.accumulate(np.quantile(draws, QUANTS))
        path = np.array([pr[f"cum{d}"] for d in range(self.D)])
        hold = float(np.clip(pr["days"], 1.0, self.D))
        dq = np.clip(hold + np.quantile(days_resid, (0.1, 0.9)), 1.0, self.D)
        mfe, mae = max(pr["mfe"], ret, 0.0), min(pr["mae"], ret, 0.0)
        clipped = np.clip(draws, PROB_EDGES[0], PROB_EDGES[-1] - 1e-9)
        counts = np.histogram(clipped, bins=PROB_EDGES)[0].astype(float)
        probs = counts / counts.sum()
        probs[-1] = 1.0 - probs[:-1].sum()
        sigma = float(resid.std())
        epi = float(math.sqrt(max(float(A[0] @ inv @ A[0]), 0.0)) * sigma)
        loss_p, over_p = float((draws < 0).mean()), float((draws > EX.BAND_HI).mean())
        alts: list[dict[str, Any]] = []
        if loss_p > 0:
            alts.append({"hypothesis": "reversal_to_loss", "probability": loss_p, "predicted_return": float(draws[draws < 0].mean())})
        if over_p > 0:
            alts.append({"hypothesis": "overshoot_above_band", "probability": over_p, "predicted_return": float(draws[draws > EX.BAND_HI].mean())})
        if not alts:
            alts.append({"hypothesis": "flat", "probability": 0.0, "predicted_return": 0.0})
        return {"predicted_return": ret, "distribution": tuple((float(q), float(v)) for q, v in zip(QUANTS, qv)),
                "time_to_peak": float(int(path.argmax()) + 1), "exit_window": (float(min(dq[0], hold)), float(max(dq[1], hold))),
                "holding_period": hold, "mfe": mfe, "mae": mae, "predicted_volatility": float(math.exp(pr["vol"])),
                "prob_distribution": tuple((PROB_EDGES[i], PROB_EDGES[i + 1], float(probs[i])) for i in range(len(probs))),
                "uncertainty": {"aleatoric": sigma, "epistemic": epi}, "alternatives": tuple(alts)}

    def __call__(self, row: Any, subject: str) -> dict:
        """The `path_model(row, subject)` interface of expectations_from_day. Features come from `bind`, else from the row."""
        if self.bound_as_of is None:
            raise ValueError("bind(entries, now) before serving expectations")
        side = int(row["side"]) if "side" in row else 1
        if side != 1:
            raise ValueError(f"{subject}: the path model forecasts long positions only (exits are long-only Paths)")
        e = self.entries.get(str(subject))
        if e is None:
            e = {k: row[k] for k in ("vol", "atr", "kind") if k in row}
        if "vol" not in e or "atr" not in e:
            raise ValueError(f"{subject}: no entry features bound (vol, atr)")
        ctx = {k: v for k, v in e.items() if k not in ("vol", "atr", "kind")}
        return self.forecast(e["vol"], e["atr"], str(e.get("kind", self.kind_levels[0] if self.kind_levels else "")), ctx)


# ------------------------------------------------------------------------------------------------ research-side diagnostics
def exit_regret(p: EX.Paths, res: EX.ExitResult, now) -> pd.DataFrame:
    """HINDSIGHT (research world only; every position must have matured before `now`): the best achievable next-open-or-final-close
    exit, the realised exit, regret and MFE capture per position. It grades the policy; it is never an input to it."""
    _matured(p, now)
    if len(p) == 0:
        return pd.DataFrame(columns=["best", "best_day", "realised", "regret", "mfe", "mfe_capture"])
    entry = p.o[:, 0]
    cands = np.hstack([p.o[:, 1:] / entry[:, None] - 1, (p.c[:, -1] / entry - 1)[:, None]])
    best = cands.max(1)
    mfe = p.h.max(1) / entry - 1
    return pd.DataFrame({"best": best, "best_day": cands.argmax(1) + 1, "realised": res.gross, "regret": best - res.gross,
                         "mfe": mfe, "mfe_capture": np.where(mfe > 1e-9, res.gross / np.where(mfe > 1e-9, mfe, 1.0), np.nan)})


def ev_calibration(p: EX.Paths, trace: ExitTrace, now, bins: int = 5) -> pd.DataFrame:
    """Did E[hold] mean what it said? Bin every recorded close decision by predicted E[hold] and compare with the realised
    next-open-to-end return (matured positions only). A flat realised column means the value model carries no information."""
    _matured(p, now)
    pe, re_ = [], []
    for d in range(p.D - 1):
        m = np.isfinite(trace.ev[:, d])
        if m.any():
            pe.append(trace.ev[m, d])
            re_.append((p.c[m, -1] / p.o[m, d + 1]) - 1)
    if not pe:
        return pd.DataFrame(columns=["bin", "n", "predicted", "realised"])
    x, y = np.concatenate(pe), np.concatenate(re_)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = [{"bin": int(k), "n": int((b == k).sum()), "predicted": float(x[b == k].mean()), "realised": float(y[b == k].mean())}
            for k in sorted(set(b.tolist()))]
    return pd.DataFrame(rows)


def best_exit_error(p: EX.Paths, traj: TrajectoryModel, now, context: Mapping[str, np.ndarray] | None = None) -> dict:
    """How far the entry-time expected best exit day was from the hindsight best day (matured only)."""
    _matured(p, now)
    if len(p) == 0:
        return {"n": 0}
    pred = traj.expected_best_exit(p, context)["best_day"].to_numpy()
    real = (p.c / p.o[:, :1] - 1).argmax(1) + 1
    err = pred - real
    return {"n": int(len(p)), "mean_abs_days": float(np.abs(err).mean()), "bias_days": float(err.mean()),
            "exact_share": float((err == 0).mean())}


def feature_decay(rule: LearnedExitRule) -> pd.DataFrame:
    """Coefficient table of the value model: which state features currently push toward holding or selling (momentum decay,
    volatility decay and pattern decay each have a signed weight here)."""
    m = rule.model
    if m is None or m.beta_ is None:
        return pd.DataFrame(columns=["feature", "coef"])
    names = list(m.names) + [f"kind={k}" for k in m.kind_levels]
    return pd.DataFrame({"feature": ["intercept"] + names, "coef": m.beta_}).sort_values("coef", key=np.abs, ascending=False)


def tournament(paths: EX.Paths, learned: LearnedExitRule | None = None, min_train_weeks: int = 52, block_weeks: int = 13, seed: int = 0,
               **kw) -> EX.WalkForward:
    """The learned exit enters the EXISTING out-of-sample tournament (engine.exits.walk_forward) beside the week-end baseline and
    the fixed families; it is only used where the week-clustered bootstrap says it wins."""
    rules = EX.default_rules(has_fail=paths.fail is not None) + [learned or LearnedExitRule()]
    return EX.walk_forward(paths, rules, min_train_weeks=min_train_weeks, block_weeks=block_weeks, seed=seed, **kw)


# ------------------------------------------------------------------------------------------------ checklist O
@dataclass(frozen=True)
class IndependenceReport:
    n_targets: int
    identical: bool
    differing: tuple[str, ...]          # targets whose exits differed from the first one
    fields_differing: tuple[str, ...]
    reference_digest: str


def _exit_digest(res: EX.ExitResult) -> str:
    return stable_hash({"net": [round(float(x), 12) for x in res.net], "days": res.days.tolist(), "reason": res.reason.tolist()})


def exit_independence_audit(run_exits: Callable[[Any], EX.ExitResult], targets: Sequence[Any]) -> IndependenceReport:
    """Run the SAME exit procedure once per evaluation target (e.g. +-1pp, +-5pp, a different goal share) and require bit-identical
    exits. `run_exits(target)` is the full pipeline under that target; if any exit day, reason or return moves, the exit is
    reading the target and checklist O is violated."""
    if not targets:
        raise ValueError("need at least one target")
    results = [run_exits(t) for t in targets]
    ref = results[0]
    diff, fields = [], set()
    for t, r in zip(targets[1:], results[1:]):
        bad = [f for f in ("net", "days", "reason") if len(getattr(r, f)) != len(getattr(ref, f)) or not np.array_equal(getattr(r, f), getattr(ref, f))]
        if bad:
            diff.append(repr(t))
            fields.update(bad)
    return IndependenceReport(len(targets), not diff, tuple(diff), tuple(sorted(fields)), _exit_digest(ref))
