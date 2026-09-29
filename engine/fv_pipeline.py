"""Bible Phase 38 checklist V1-V5 (Find volatility), canons C23 / C24 / C33: ONE point-in-time weekly pipeline.

Every week (decision at the LAST session's close, fills at the NEXT session's open, regular hours only, five sessions
held, no crypto - C33):
  1. MOVERS     pick the `n_picks` stocks most likely to touch +-10% next week (LightGBM on cross-sectional ranks, Platt
                calibrated on a later block; the smallest probability whose out-of-sample precision reaches 95% is
                reported as tau95, "sufficient candidates" = how many clear it) - V1;
  2. DIRECTION  engine.direction.DirectionEngine stacks a mover-only direction model and abstains below the calibrated
                80% gate (C24) - V2;
  3. EXITS      a per-stock-type exit rule chosen on the FV objective (share of positions reaching +10%, subject to a
                catastrophic-loss budget) from positions that had ALREADY finished - V3;
  4. STOPS      stop distances from engine.stops plus a gaprisk loss-cap filter that puts names whose learned P(breach
                of -20%) is too high into cash - V4.
Nothing here promises a floor: a gap can jump any stop, so the report states P(loss <= -20%) with a Wilson bound.

Walk-forward only: models and policies are refitted at origins every `refit_every` weeks from rows whose last bar is
on or before the origin close, and used only for the decisions after it. Every stage can be switched off; `ablation`
evaluates M / M+D / M+D+E / M+D+E+S (and any other subset) on the SAME picks with paired week-block bootstrap CIs.
A pick whose entry bar does not exist is left as cash in its slot (it is never replaced - that would be look-ahead).

Panel convention as elsewhere: X indexed (date, ticker); market-context columns start `m_`. Deterministic: every
random draw takes a seed. Shorts are simulated by reflecting the path around the entry price (exact negated return,
unlevered) and are switched off with `allow_short=False`."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from sklearn.linear_model import LogisticRegression

from .direction import DirectionEngine, build_inputs
from .exits import CATASTROPHE, CostModel, ExitResult, ExitSpec, Paths, Rule, run_exit, _week_codes
from .gaprisk import ConditionalGapModel, EventCalendar
from .pattern_movers import auc
from .stops import (MAX_DIST, MIN_DIST, AtrStop, FixedStop, GapAwareStop, HybridStop, StopRule, wilson)

D_BARS = 5                       # sessions held (entry = first bar's open)
HIT = 0.10                       # the +10% objective
STAGES = ("M", "D", "E", "S")
ABLATIONS = {"M": "M", "M+D": "MD", "M+D+E": "MDE", "M+D+S": "MDS", "M+D+E+S": "MDES"}


# ------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class FVConfig:
    n_picks: int = 10
    move: float = 0.10               # a mover touches +-move inside the holding week
    mover_label: str = "touch"       # touch (high/low) or close (week-end close)
    gate: float = 0.80               # direction confidence gate (C24)
    refill: bool = False             # abstained slots are refilled from deeper in the mover list
    strict_movers: bool = False      # only picks above tau95 (fewer than n_picks when candidates are scarce)
    pool: int = 30                   # candidates recorded per week (exit/stop learning history + refill depth)
    allow_short: bool = True
    min_train_weeks: int = 104
    refit_every: int = 13
    min_price: float = 3.0
    min_dollar_vol: float = 2e6
    cal_frac: float = 0.20
    tau_target: float = 0.95
    tau_min_rows: int = 40
    mover_min_rows: int = 2000
    lgb_trees: int = 150
    lgb_leaves: int = 15
    lgb_min_child: int = 80
    lgb_jobs: int = 1
    dir_split: float = 0.50          # share of mover dates used only to fit the direction model itself
    dir_min_rows: int = 300
    dir_min_inputs: int = 1
    dir_horizon_days: int = 8
    cap: float = -0.20               # the worst position loss the owner tolerates (C24)
    max_cat_hi: float = 0.05         # exit/stop feasibility: Wilson upper bound of P(loss <= cap)
    cap_tau: float = 0.03            # loss-cap filter: drop names whose model P(breach) exceeds this
    cap_min_train: int = 300
    pool_train_side: str = "soft"    # exit history is oriented by the (ungated) direction sign
    policy_min_weeks: int = 26
    policy_min_trades: int = 80
    policy_min_win: float = 0.70
    policy_boot: int = 120
    fee_bps: float = 2.0
    slip_bps: float = 5.0
    seed: int = 7

    def cost(self) -> CostModel:
        return CostModel(self.fee_bps, self.slip_bps)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ bars and features
BAR_KEYS = ("Open", "High", "Low", "Close")


def validate_bars(bars: dict) -> list[str]:
    """Problems that would silently corrupt every downstream number (empty list = clean)."""
    bad = []
    for k in BAR_KEYS:
        if k not in bars:
            bad.append(f"missing {k}")
    if bad:
        return bad
    ref = bars["Close"]
    for k in BAR_KEYS + (("Volume",) if "Volume" in bars else ()):
        f = bars[k]
        if f.shape != ref.shape or not f.index.equals(ref.index) or not f.columns.equals(ref.columns):
            bad.append(f"{k} not aligned with Close")
    if not ref.index.is_monotonic_increasing or ref.index.has_duplicates:
        bad.append("sessions not strictly increasing")
    if bad:
        return bad
    o, h, l, c = (bars[k].to_numpy(np.float64) for k in BAR_KEYS)
    with np.errstate(invalid="ignore"):
        if (c[np.isfinite(c)] <= 0).any():
            bad.append("non-positive close")
        wick = np.isfinite(h) & np.isfinite(l) & (h < l - 1e-9)
        if wick.any():
            bad.append(f"high below low in {int(wick.sum())} bars")
    if not np.isfinite(c).any():
        bad.append("no finite closes")
    return bad


def week_end_sessions(sessions: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last session of each ISO week (a decision is taken at that close). The final session of the
    data is never a week end here: its week is not complete."""
    iso = sessions.isocalendar()
    key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    return np.flatnonzero(key[1:] != key[:-1])


def price_features(bars: dict, min_price: float = 3.0, min_dollar_vol: float = 2e6) -> dict:
    """Daily price/volume features from OHLCV only (float32 wide frames, date x ticker) plus the `tradable` mask.
    Every value at row t uses bars up to and including t's close."""
    C = bars["Close"].astype("float32")
    H, L, O = (bars[k].astype("float32") for k in ("High", "Low", "Open"))
    V = bars["Volume"].astype("float32") if "Volume" in bars else None
    r1 = C.pct_change(fill_method=None)
    f = {"r1": r1, "r5": C / C.shift(5) - 1, "r20": C / C.shift(20) - 1, "r60": C / C.shift(60) - 1,
         "vol20": r1.rolling(20, min_periods=15).std(),
         "atr": ((H - L) / C.shift(1)).rolling(14, min_periods=10).mean(),
         "gap": O / C.shift(1) - 1,
         "range20": (H.rolling(20, min_periods=15).max() - L.rolling(20, min_periods=15).min()) / C,
         "dist_hi": C / H.rolling(60, min_periods=40).max() - 1, "dist_lo": C / L.rolling(60, min_periods=40).min() - 1,
         "absr1": r1.abs(), "absr5": (C / C.shift(5) - 1).abs(), "max5": r1.abs().rolling(5, min_periods=4).max(),
         "logp": np.log(C)}
    if V is not None:
        dv = (C * V).rolling(20, min_periods=15).mean()
        f["vol_surge"] = V / V.rolling(20, min_periods=15).mean()
        f["log_dv"] = np.log1p(dv)
        tradable = (C >= min_price) & (dv >= min_dollar_vol) & (V > 0)
    else:
        tradable = C >= min_price
    f["m_r5"] = pd.DataFrame(np.repeat(f["r5"].median(axis=1).to_numpy()[:, None], C.shape[1], 1), C.index, C.columns)
    f["m_vol"] = pd.DataFrame(np.repeat(f["vol20"].median(axis=1).to_numpy()[:, None], C.shape[1], 1), C.index, C.columns)
    f["m_r20"] = pd.DataFrame(np.repeat(f["r20"].median(axis=1).to_numpy()[:, None], C.shape[1], 1), C.index, C.columns)
    f["m_breadth"] = pd.DataFrame(np.repeat((f["r5"] > 0).mean(axis=1).to_numpy()[:, None], C.shape[1], 1), C.index, C.columns)
    f["tradable"] = tradable
    return f


@dataclass
class Panel:
    """Everything the walk-forward needs, aligned: features X and labels `lab` on the same (date, ticker) rows, plus the
    raw OHLC arrays to build position paths from."""
    sessions: pd.DatetimeIndex
    tickers: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    dec: np.ndarray            # session position of each decision date
    last: np.ndarray           # session position of each week's last real bar
    dates: pd.DatetimeIndex    # decision dates
    X: pd.DataFrame            # (date, ticker) features, core columns finite
    lab: pd.DataFrame          # k, j, kind, fill_ok, entry, hi, lo, close, touch, up, end
    XR: pd.DataFrame = None    # cross-sectional rank version of X (m_ columns raw)

    def gather(self, k: np.ndarray, j: np.ndarray):
        """OHLC bars (n, D_BARS) for decision indices k and ticker columns j. Bars after the week's last session, and
        bars missing from the source, are flat at the previous close (a delisting exits at the last price, never
        earlier than it happens). Returns (o, h, l, c, entry_ok, prev_close)."""
        k, j = np.asarray(k), np.asarray(j)
        w, last = self.dec[k], self.last[k]
        idx = w[:, None] + 1 + np.arange(D_BARS)[None, :]
        valid = idx <= last[:, None]
        idc = np.minimum(idx, len(self.sessions) - 1)
        g = lambda A: A[idc, j[:, None]].astype(np.float64)
        o, h, l, c = g(self.O), g(self.H), g(self.L), g(self.C)
        prev0 = self.C[w, j].astype(np.float64)
        finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & valid
        entry_ok = finite[:, 0]
        prev = prev0.copy()
        for d in range(D_BARS):
            bad = ~finite[:, d]
            for a in (o, h, l, c):
                a[bad, d] = prev[bad]
            prev = c[:, d]
        h[:] = np.maximum.reduce([h, o, c])
        l[:] = np.minimum.reduce([l, o, c])
        return o, h, l, c, entry_ok, prev0

    def paths(self, rows: np.ndarray, weight: float, vol: np.ndarray, atr: np.ndarray, kind: np.ndarray) -> Paths:
        """Long-oriented Paths for label-table row positions `rows`."""
        k, j = self.lab["k"].to_numpy()[rows], self.lab["j"].to_numpy()[rows]
        o, h, l, c, ok, prev = self.gather(k, j)
        wk = self.dates.to_numpy()[k].astype("datetime64[D]")
        en = self.sessions.to_numpy()[self.last[k]].astype("datetime64[D]")
        return Paths(o, h, l, c, prev, vol, atr, wk, en, kind, self.tickers[j], None, np.full(len(rows), weight))


def build_panel(bars: dict, cfg: FVConfig = FVConfig(), start=None) -> Panel:
    """Features at week-end decisions and the realised outcome of buying the NEXT open and holding five sessions."""
    bad = validate_bars(bars)
    if bad:
        raise ValueError("invalid bars: " + "; ".join(bad))
    sessions = pd.DatetimeIndex(bars["Close"].index)
    tick = np.asarray(bars["Close"].columns, dtype=object)
    f = price_features(bars, cfg.min_price, cfg.min_dollar_vol)
    weekends = week_end_sessions(sessions)
    nxt = np.append(weekends[1:], len(sessions) - 1)
    keep = nxt > weekends
    if start is not None:
        keep &= sessions[weekends] >= pd.Timestamp(start)
    dec, last = weekends[keep], np.minimum(nxt[keep], weekends[keep] + D_BARS)
    dates = sessions[dec]
    core = [k for k in f if k != "tradable"]
    cube = {k: f[k].to_numpy()[dec] for k in core}
    ok = f["tradable"].to_numpy()[dec].copy()
    for k in core:
        ok &= np.isfinite(cube[k])
    kk, jj = np.nonzero(ok)
    ix = pd.MultiIndex.from_arrays([dates[kk], tick[jj]], names=["date", "ticker"])
    X = pd.DataFrame({k: cube[k][kk, jj].astype("float32") for k in core}, index=ix)
    C = bars["Close"].to_numpy()
    P = Panel(sessions, tick, bars["Open"].to_numpy(), bars["High"].to_numpy(), bars["Low"].to_numpy(), C, dec, last, dates, X,
              pd.DataFrame(), None)
    o, h, l, c, entry_ok, prev = P.gather(kk, jj)
    entry = o[:, 0]
    hi, lo = h.max(1) / entry - 1, l.min(1) / entry - 1
    close = c[:, -1] / entry - 1
    vol_hi = X["vol20"].to_numpy() >= pd.Series(X["vol20"].to_numpy()).groupby(kk).transform("median").to_numpy()
    kind = np.where(vol_hi, "hv", "lv").astype(object) + np.where(prev < 10, "_lp", "_hp").astype(object)
    if cfg.mover_label == "close":
        touch = np.abs(close) >= cfg.move
    else:
        touch = (hi >= cfg.move) | (lo <= -cfg.move)
    lab = pd.DataFrame({"k": kk, "j": jj, "kind": kind, "fill_ok": entry_ok, "entry": entry, "hi": hi, "lo": lo,
                        "close": close, "touch": np.where(entry_ok, touch, np.nan),
                        "up": np.where(entry_ok, (close > 0).astype(float), np.nan),
                        "end": sessions[last[kk]].to_numpy()}, index=ix)
    lab.loc[~entry_ok, ["hi", "lo", "close"]] = np.nan
    P.lab = lab
    P.XR = rank_features(X)
    return P


def rank_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date (era-free); market-context columns stay raw."""
    mcols = [c for c in X.columns if c.startswith("m_")]
    R = X.drop(columns=mcols).groupby(level=0).rank(pct=True).astype("float32")
    for c in mcols:
        R[c] = X[c]
    return R[list(X.columns)]


def synthetic_bars(n_tickers=120, n_weeks=260, seed=0, event_p=0.15, dir_acc=0.95, informative_mover=True,
                   start="2010-01-04") -> dict:
    """Planted market for tests. An event week gives a stock a +-13..22% drift; the decision day's own return is a large
    shock that announces the event (marker) and points the right way with probability `dir_acc`. Everything else is noise.
    informative_mover=False removes the marker; dir_acc=0.5 removes the direction signal (pure noise)."""
    rng = np.random.default_rng(seed)
    sess = pd.bdate_range(start, periods=n_weeks * 5 + 1)
    n = n_weeks * 5 + 1
    ret = rng.normal(0, 0.008, (n, n_tickers))
    ret[0] = 0.0
    for w in range(1, n_weeks):
        ev = rng.random(n_tickers) < event_p
        sign = np.where(rng.random(n_tickers) < 0.5, 1.0, -1.0)
        mag = rng.uniform(0.13, 0.22, n_tickers)
        for d in range(5):
            ret[w * 5 + d] = np.where(ev, sign * mag / 5 + rng.normal(0, 0.003, n_tickers), ret[w * 5 + d])
        hint = np.where(rng.random(n_tickers) < dir_acc, sign, -sign)
        marker = rng.uniform(0.02, 0.04, n_tickers)
        ret[w * 5 - 1] = np.where(ev & informative_mover, hint * marker, np.where(ev, rng.normal(0, 0.008, n_tickers), ret[w * 5 - 1]))
    close = 40 * np.exp(np.cumsum(ret, 0))
    gap = rng.normal(0, 0.002, close.shape)
    op = np.vstack([close[:1], close[:-1]]) * (1 + gap)
    op[0] = close[0]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.003, close.shape)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.003, close.shape)))
    tk = [f"S{i:03d}" for i in range(n_tickers)]
    vol = pd.DataFrame(rng.uniform(5e5, 5e6, close.shape), sess, tk)
    fr = lambda a: pd.DataFrame(a, sess, tk)
    return {"Open": fr(op), "High": fr(hi), "Low": fr(lo), "Close": fr(close), "Volume": vol}


# ------------------------------------------------------------------------------------------------ paths and shorts
def reflect_short(p: Paths, side: np.ndarray) -> Paths:
    """Orient paths for the side of the bet. A short is the path reflected about its entry open (x -> 2e - x), so a long
    simulator run on it returns exactly the negated price return. Floored at 1% of entry (a +100% squeeze is a total
    loss well before the floor matters)."""
    s = np.asarray(side).reshape(-1)
    short = s < 0
    if not short.any():
        return p
    e = p.o[:, :1]
    fl = 0.01 * e
    flip = lambda a: np.maximum(2 * e - a, fl)
    m = short[:, None]
    o2, c2 = np.where(m, flip(p.o), p.o), np.where(m, flip(p.c), p.c)
    h2, l2 = np.where(m, flip(p.l), p.h), np.where(m, flip(p.h), p.l)
    prev2 = np.where(short, np.maximum(2 * e[:, 0] - p.prev_close, fl[:, 0]), p.prev_close)
    return Paths(o2, h2, l2, c2, prev2, p.vol, p.atr, p.week, p.end, p.kind, p.ticker, p.fail, p.weight)


def concat_paths(chunks: Sequence[Paths]) -> Paths:
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        z = np.zeros((0, D_BARS))
        return Paths(z, z, z, z, [], [], [], [], [], [], [])
    cat = lambda name: np.concatenate([getattr(c, name) for c in chunks])
    return Paths(cat("o"), cat("h"), cat("l"), cat("c"), cat("prev_close"), cat("vol"), cat("atr"), cat("week"), cat("end"),
                 cat("kind"), cat("ticker"), None, cat("weight"))


# ------------------------------------------------------------------------------------------------ stage 1: movers
class MoverStage:
    """P(the stock touches +-move next week). Trained on rows that had finished by the origin; the newest `cal_frac` of
    dates (one date purged) calibrate the probability and certify tau95."""

    def __init__(self, cfg: FVConfig):
        self.cfg = cfg
        self.clf = None
        self.platt = None
        self.tau95 = None
        self.reason = "not fitted"
        self.diag: dict = {}

    def fit(self, XR: pd.DataFrame, y: np.ndarray, dates: np.ndarray):
        import lightgbm as lgb
        c = self.cfg
        self.clf = self.platt = self.tau95 = None
        ud = np.unique(dates)
        n = len(y)
        if n < c.mover_min_rows or len(ud) < 40 or y.sum() < 50 or y.sum() > n - 50:
            self.reason = f"insufficient data: {n} rows, {int(y.sum())} movers"
            return self
        ci = int(len(ud) * (1 - c.cal_frac))
        tr, ca = dates < ud[max(ci - 1, 1)], dates >= ud[ci]
        self.clf = lgb.LGBMClassifier(objective="binary", n_estimators=c.lgb_trees, num_leaves=c.lgb_leaves,
                                      min_child_samples=c.lgb_min_child, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.7, random_state=c.seed, verbose=-1, n_jobs=c.lgb_jobs)
        self.clf.fit(XR[tr], y[tr])
        raw = self.clf.predict_proba(XR[ca])[:, 1]
        yc = y[ca]
        if len(np.unique(yc)) > 1 and ca.sum() > 100:
            self.platt = LogisticRegression(C=1e3, max_iter=300).fit(_logit(np.clip(raw, 1e-5, 1 - 1e-5))[:, None], yc)
        pc = self._cal(raw)
        self.tau95 = certify_threshold(pc, yc, c.tau_target, c.tau_min_rows)
        self.diag = dict(n_train=int(tr.sum()), n_cal=int(ca.sum()), base_rate_train=float(y[tr].mean()),
                         base_rate_cal=float(yc.mean()), auc_cal=auc(raw, yc.astype(bool)), tau95=self.tau95,
                         top_decile_precision=float(yc[pc >= np.quantile(pc, 0.9)].mean()))
        self.reason = "ok"
        return self

    def _cal(self, raw):
        if self.platt is None:
            return raw
        return self.platt.predict_proba(_logit(np.clip(raw, 1e-5, 1 - 1e-5))[:, None])[:, 1]

    @property
    def ok(self):
        return self.clf is not None

    def score(self, XR: pd.DataFrame) -> np.ndarray:
        if not self.ok:
            raise RuntimeError("mover stage not fitted: " + self.reason)
        return self._cal(self.clf.predict_proba(XR)[:, 1])


def certify_threshold(p: np.ndarray, y: np.ndarray, target: float = 0.95, min_rows: int = 40):
    """Smallest probability threshold whose precision on (p, y) reaches `target` with at least `min_rows` rows at or above
    it; None when no threshold does. Chosen on the calibration block, so it is optimistic - the walk-forward hit rate of
    the picks is the honest number."""
    if len(p) == 0:
        return None
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum_n = np.arange(1, len(p) + 1)
    prec = np.cumsum(ys) / cum_n
    good = np.flatnonzero((cum_n >= min_rows) & (prec >= target))
    return float(ps[good[-1]]) if len(good) else None


# ------------------------------------------------------------------------------------------------ stage 2: direction
class DirectionStage:
    """A mover-only direction model (fitted on the EARLY mover dates) feeds DirectionEngine, which stacks, calibrates and
    gates on the later dates - so the engine never sees a score the direction model was trained on."""

    def __init__(self, cfg: FVConfig):
        self.cfg = cfg
        self.clf = None
        self.engine: DirectionEngine | None = None
        self.reason = "not fitted"

    def fit(self, XR: pd.DataFrame, up: np.ndarray, mover: np.ndarray, dates: np.ndarray, origin, extra: pd.DataFrame | None = None):
        import lightgbm as lgb
        c = self.cfg
        self.clf = self.engine = None
        m = mover.astype(bool)
        if m.sum() < c.dir_min_rows:
            self.reason = f"only {int(m.sum())} mover rows"
            return self
        R, y, d = XR[m], up[m], dates[m]
        ud = np.unique(d)
        cut = ud[int(len(ud) * c.dir_split)]
        early = d < cut
        if early.sum() < 150 or len(np.unique(y[early])) < 2:
            self.reason = "direction model has too little early data"
            return self
        self.clf = lgb.LGBMClassifier(objective="binary", n_estimators=c.lgb_trees, num_leaves=7, min_child_samples=40,
                                      learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                      random_state=c.seed + 1, verbose=-1, n_jobs=c.lgb_jobs)
        self.clf.fit(R[early], y[early])
        late = ~early
        F = self.inputs(R[late], None if extra is None else extra[m][late])
        self.engine = DirectionEngine(gate=c.gate, min_rows=c.dir_min_rows // 2, min_test=max(60, c.dir_min_rows // 3),
                                      min_inputs=c.dir_min_inputs, horizon_days=c.dir_horizon_days, seed=c.seed)
        self.engine.fit(F, pd.Series(y[late], index=F.index), pd.Timestamp(origin))
        self.reason = self.engine.reason
        return self

    def model_score(self, R: pd.DataFrame) -> pd.Series:
        p = np.clip(self.clf.predict_proba(R)[:, 1], 1e-4, 1 - 1e-4)
        return pd.Series(_logit(p), index=R.index)

    def inputs(self, R: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        kw = {"model": self.model_score(R)}
        if extra is not None:
            for k in ("pattern", "analog", "trust", "mw", "evidence"):
                if k in extra:
                    kw[k] = extra[k]
        return build_inputs(index=R.index, **kw)

    def decide(self, R: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        """p_up, conf, side, bet, reason for each row; p_soft = the ungated direction estimate (used to orient the exit /
        stop training history). A stage that could not be fitted abstains on everything and says why."""
        n = len(R)
        if self.clf is None:
            return pd.DataFrame({"p_up": np.full(n, 0.5), "conf": np.full(n, 0.5), "side": 0, "bet": False,
                                 "reason": "direction_unfitted: " + self.reason, "p_soft": 0.5}, index=R.index)
        F = self.inputs(R, extra)
        out = self.engine.decide(F)
        out["p_soft"] = self.engine.predict(F).values if self.engine.stack is not None else 1 / (1 + np.exp(-F["model"].values))
        return out


# ------------------------------------------------------------------------------------------------ stages 3-4: rules
@dataclass
class ComboRule(Rule):
    """An exit spec (target / trail / time) optionally combined with a learned stop distance from engine.stops. With
    `stop=None` this is a pure exit rule; with the null spec it is a pure stop rule. Element 0 of a rule set is always the
    plain week-end baseline."""
    spec: ExitSpec = field(default_factory=ExitSpec)
    stop: StopRule | None = None
    name: str = "week_end"
    family: str = "week_end"
    cost: CostModel = field(default_factory=CostModel)

    def fit(self, train: Paths):
        r = copy.copy(self)
        if self.stop is not None:
            r.stop = self.stop.fit(train)
        return r

    def dist(self, p: Paths):
        return None if self.stop is None else self.stop.dist(p)

    def run(self, p: Paths) -> ExitResult:
        res = run_exit(p, self.spec, self.cost, stop_dist=self.dist(p))
        st = self.stop
        if st is not None and st.skip_tail is not None and st.gap is not None and len(p):
            skip = st.gap.tail(p) > st.skip_tail          # cash: the position is not opened
            res.net[skip], res.gross[skip], res.days[skip], res.stop_overshoot[skip] = 0.0, 0.0, 0, 0.0
        return res


def exit_specs() -> dict:
    """Exit families of Phase 15 reduced to the ones that matter for a +10% goal (all share one fill simulator)."""
    return {"week_end": ExitSpec(), "target_7": ExitSpec(target=0.07), "target_10": ExitSpec(target=0.10),
            "trail_6": ExitSpec(trail=0.06, arm=0.03), "t10_trail6": ExitSpec(target=0.10, trail=0.06, arm=0.03),
            "time_4": ExitSpec(hold_days=4), "volt_1": ExitSpec(target_vol=1.0)}


def stop_rules(cost: CostModel) -> dict:
    return {"atr_3": AtrStop(name="atr_3", family="atr", cost=cost, k=3.0),
            "fixed_10": FixedStop(name="fixed_10", family="type_specific", cost=cost, d=0.10),
            "gap_3": GapAwareStop(name="gap_3", family="gap_aware", cost=cost, k=3.0),
            "hybrid": HybridStop(name="hybrid", family="hybrid", cost=cost),
            "hybrid_filter": HybridStop(name="hybrid_filter", family="hybrid", cost=cost, skip_tail=0.12)}


def build_rule_set(exits_on: bool, stops_on: bool, cost: CostModel) -> list[Rule]:
    """[baseline] + exit rules (exits_on) + stop rules with the week-end exit (stops_on) + the cross of the two."""
    ex, st = exit_specs(), stop_rules(cost)
    rules: list[Rule] = [ComboRule(ExitSpec(), None, "week_end", "week_end", cost)]
    if exits_on:
        rules += [ComboRule(s, None, n, "exit", cost) for n, s in ex.items() if n != "week_end"]
    if stops_on:
        rules += [ComboRule(ExitSpec(), s, s.name, "stop", cost) for s in st.values()]
    if exits_on and stops_on:
        for n, s in ex.items():
            if n in ("week_end", "time_4"):
                continue
            for sname in ("atr_3", "fixed_10", "gap_3"):
                rules.append(ComboRule(s, st[sname], f"{n}+{sname}", "combo", cost))
    return rules


def _fv_tables(net, gross, codes, W, cap):
    b = lambda w: np.bincount(codes, weights=w, minlength=W)
    return {"n": b(np.ones(len(net))), "hit": b((gross >= HIT - 1e-9).astype(float)), "tot": b(net),
            "cat": b((net <= cap).astype(float)), "low": b(np.minimum(net, 0))}


def _fv_metrics(t, order=None, max_cat_hi=0.05):
    if order is not None:
        t = {k: v[order] for k, v in t.items()}
    n = t["n"].sum()
    if n == 0:
        return {"n": 0, "hit": 0.0, "mean": 0.0, "cat": 0.0, "cat_hi": 1.0, "feasible": False}
    cat = t["cat"].sum() / n
    hi = wilson(int(round(t["cat"].sum())), int(n))[1]
    return {"n": int(n), "hit": float(t["hit"].sum() / n), "mean": float(t["tot"].sum() / n), "cat": float(cat), "cat_hi": hi,
            "feasible": bool(hi <= max_cat_hi)}


def fv_compare(a: dict, b: dict) -> int:
    """Lexicographic FV objective: (1) feasibility of the catastrophic-loss budget, (2) share of positions reaching +10%
    (tolerance 1 pt), (3) mean net return (0.2 pt), (4) catastrophic rate (0.5 pt). +1 a better, -1 b better, 0 tie."""
    if a["n"] == 0 or b["n"] == 0:
        return 0
    if a["feasible"] != b["feasible"]:
        return 1 if a["feasible"] else -1
    for key, sign, tol in (("hit", 1, 0.01), ("mean", 1, 0.002), ("cat", -1, 0.005)):
        d = (a[key] - b[key]) * sign
        if d > tol:
            return 1
        if d < -tol:
            return -1
    return 0


@dataclass
class Selection:
    kind: str
    chosen: str
    reason: str
    n_trades: int
    n_weeks: int
    win: float | None = None
    table: pd.DataFrame | None = None


def fv_select(train: Paths, rules: Sequence[Rule], seed: int, cfg: FVConfig, kind: str = "all"):
    """Choose one rule for one stock type from TRAINING positions only (in-sample tournament, then a week-bootstrap against
    the baseline). Baseline is kept on thin data, ties, or a win that is not week-robust."""
    base = rules[0]
    codes, W = _week_codes(train) if len(train) else (np.zeros(0, int), 0)
    if len(train) < cfg.policy_min_trades or W < cfg.policy_min_weeks:
        return base, Selection(kind, base.name, "neutral:insufficient_data", len(train), W)
    fitted = [r.fit(train) for r in rules]
    res = [f.run(train) for f in fitted]
    tabs = [_fv_tables(r.net, r.gross, codes, W, cfg.cap) for r in res]
    mets = [_fv_metrics(t, None, cfg.max_cat_hi) for t in tabs]
    champ = 0
    for i in range(1, len(rules)):
        if fv_compare(mets[i], mets[champ]) > 0:
            champ = i
    tab = pd.DataFrame([{"rule": rules[i].name, "family": rules[i].family, **mets[i]} for i in range(len(rules))])
    if champ == 0:
        return fitted[0], Selection(kind, base.name, "baseline_best", len(train), W, None, tab)
    rng = np.random.default_rng(seed)
    wins = losses = 0
    for _ in range(cfg.policy_boot):
        order = rng.integers(0, W, W)
        v = fv_compare(_fv_metrics(tabs[champ], order, cfg.max_cat_hi), _fv_metrics(tabs[0], order, cfg.max_cat_hi))
        wins += v > 0
        losses += v < 0
    win = wins / (wins + losses) if wins + losses else 0.5
    if win < cfg.policy_min_win:
        return fitted[0], Selection(kind, base.name, "baseline_kept:not_week_robust", len(train), W, win, tab)
    return fitted[champ], Selection(kind, rules[champ].name, "selected", len(train), W, win, tab)


@dataclass
class FVPolicy:
    """Per-type rules learned as of `as_of`. A type with too little data borrows the pooled rule only if the pooled rule
    itself passed every gate; otherwise it runs the baseline."""
    by_kind: dict
    selections: dict
    baseline: Rule
    pooled: Rule | None = None
    as_of: np.datetime64 | None = None

    def rule_for(self, kind):
        return self.by_kind.get(kind, self.pooled or self.baseline)

    def apply(self, p: Paths) -> ExitResult:
        n = len(p)
        out = ExitResult(np.zeros(n), np.zeros(n), np.full(n, p.D if n else 0), np.zeros(n, int), np.zeros(n), p.D if n else 0)
        for k in set(p.kind):
            m = p.kind == k
            r = self.rule_for(k).run(p.take(np.flatnonzero(m)))
            for a in ("net", "gross", "days", "reason", "stop_overshoot"):
                getattr(out, a)[m] = getattr(r, a)
        return out

    def describe(self) -> dict:
        return {k: s.chosen for k, s in self.selections.items()}


def learn_policy(train: Paths, rules: Sequence[Rule], as_of, seed: int, cfg: FVConfig) -> FVPolicy:
    """Learn a per-type policy using ONLY positions that finished by `as_of`."""
    tr = train.until(as_of)
    by, sels = {}, {}
    for j, k in enumerate(sorted(set(tr.kind))):
        by[k], sels[k] = fv_select(tr.take(np.flatnonzero(tr.kind == k)), rules, seed + j, cfg, k)
    pooled = None
    thin = [k for k, s in sels.items() if s.reason == "neutral:insufficient_data"]
    if len(tr):
        pr, ps = fv_select(tr, rules, seed + 999, cfg, "POOLED")
        sels["POOLED"] = ps
        if ps.reason == "selected":
            pooled = pr
            for k in thin:
                by.pop(k, None)
    return FVPolicy(by, sels, rules[0].fit(tr), pooled, np.datetime64(as_of, "D"))


class LossCapFilter:
    """gaprisk-based cash filter: a name whose model probability of a gap beyond (cap - stop distance) inside the week
    exceeds `tau` is not traded. Fitted on finished positions only; not fitted (keeps everything) on thin history."""

    def __init__(self, cfg: FVConfig, calendar: EventCalendar | None = None):
        self.cfg, self.calendar = cfg, calendar
        self.model: ConditionalGapModel | None = None
        self.reason = "not fitted"

    def fit(self, train: Paths):
        if len(train) < self.cfg.cap_min_train:
            self.model, self.reason = None, f"only {len(train)} training positions"
            return self
        self.model = ConditionalGapModel(self.calendar).fit(train)
        self.reason = "ok"
        return self

    def p_breach(self, p: Paths) -> np.ndarray:
        if self.model is None or len(p) == 0:
            return np.zeros(len(p))
        dist = np.clip(3.0 * p.atr, MIN_DIST, MAX_DIST)
        return self.model.p_position(p, np.maximum(-self.cfg.cap - dist, 0.01))

    def keep(self, p: Paths) -> np.ndarray:
        return self.p_breach(p) <= self.cfg.cap_tau


# ------------------------------------------------------------------------------------------------ the walk-forward
@dataclass
class FVRun:
    cfg: FVConfig
    panel: Panel
    cands: pd.DataFrame                  # one row per (decision week, candidate rank < pool)
    rows: np.ndarray                     # label-table row position of every candidate (for paths)
    paths: Paths                         # long-oriented paths of the candidates (same order)
    blocks: dict                         # block id -> {"origin", "policies": {E, S, ES}, "cap": LossCapFilter}
    origins: pd.DataFrame                # per-origin diagnostics
    week_dates: pd.DatetimeIndex         # every decision week that was traded (including weeks with no bet)


def _extra(hook_out, index):
    return None if hook_out is None else hook_out.reindex(index)


def walk_forward(panel: Panel, cfg: FVConfig = FVConfig(), calendar: EventCalendar | None = None,
                 pattern_factory: Callable | None = None, ckpt: str | Path | None = None, log=None,
                 last_block: int | None = None) -> FVRun:
    """Run the pipeline through time. `pattern_factory()` (optional) returns an object with fit(X, y, now) and
    features(Xday, as_of) -> DataFrame (engine.pattern_movers.PatternMoverModel has that shape): it is fitted on the EARLY
    half of the training dates only, and the mover model trains on the later half so it never sees in-sample pattern scores."""
    lab, XR, X = panel.lab, panel.XR, panel.X
    kk = lab["k"].to_numpy()
    n_dec = len(panel.dates)
    starts = list(range(cfg.min_train_weeks, n_dec, cfg.refit_every))
    n_run = len(starts) if last_block is None else min(last_block, len(starts))   # a truncated run keeps true block edges
    state = {"cands": [], "rows": [], "paths": [], "soft": [], "blocks": {}, "origins": [], "week_dates": [], "done": 0, "cfg": cfg.hash()}
    if ckpt and Path(ckpt).exists():
        with open(ckpt, "rb") as fh:
            st = pickle.load(fh)
        if st.get("cfg") == cfg.hash():
            state = st
    end_ts = lab["end"].to_numpy()
    y_mov = lab["touch"].to_numpy()
    y_up = lab["up"].to_numpy()
    dvals = panel.dates.to_numpy()
    rowdate = dvals[kk]
    vol_all, atr_all = X["vol20"].to_numpy(np.float64), X["atr"].to_numpy(np.float64)
    kind_all = lab["kind"].to_numpy(object)
    cost = cfg.cost()
    rule_sets = {"E": build_rule_set(True, False, cost), "S": build_rule_set(False, True, cost),
                 "ES": build_rule_set(True, True, cost)}

    for bi in range(state["done"], n_run):
        s0 = starts[bi]
        s1 = starts[bi + 1] if bi + 1 < len(starts) else n_dec
        origin = panel.dates[s0]
        o64 = np.datetime64(origin)
        past = (end_ts <= o64) & np.isfinite(y_mov)
        rec = {"block": bi, "origin": origin, "n_past": int(past.sum())}
        ms, ds = MoverStage(cfg), DirectionStage(cfg)
        pat_hook, pat_cols = None, None
        Xp, Rp = X[past], XR[past]
        if pattern_factory is not None and past.sum() > 0:
            ud = np.unique(rowdate[past])
            cut = ud[len(ud) // 2]
            ea = past & (rowdate < cut)
            try:
                pat_hook = pattern_factory()
                pat_hook.fit(X[ea], lab["close"][ea], pd.Timestamp(cut))
                pf = pat_hook.features(X[past], panel.dates[s0])
                Rp = pd.concat([Rp, pf.add_prefix("pat_").astype("float32")], axis=1)
                pat_cols = list(pf.columns)
                rec["pattern_deploy"] = bool(getattr(pat_hook, "decision", {}).get("deploy", True))
            except Exception as e:                     # a failing optional stage must not silently change the others
                pat_hook, pat_cols = None, None
                rec["pattern_error"] = repr(e)[:200]
        ms.fit(Rp, y_mov[past].astype(bool), rowdate[past])
        rec.update({f"mover_{k}": v for k, v in ms.diag.items()}, mover_reason=ms.reason)
        ds.fit(XR[past], y_up[past], (y_mov[past] == 1), rowdate[past], origin)
        rec["dir_reason"] = ds.reason
        if ds.engine is not None:
            rec.update({f"dir_{k}": v for k, v in ds.engine.diag.get("test", {}).items() if np.isscalar(v)})
            rec["dir_open"] = ds.engine.open
        # exit / stop learning history: candidates of earlier weeks that have finished by the origin
        hist_paths = concat_paths(state["soft"])   # oriented by the ungated direction sign
        hist_paths = hist_paths.until(o64)
        pol, capf = {}, LossCapFilter(cfg, calendar)
        if len(hist_paths):
            for key, rs in rule_sets.items():
                pol[key] = learn_policy(hist_paths, rs, o64, cfg.seed + bi, cfg)
            capf.fit(hist_paths)
        rec["policy_E"] = pol["E"].describe() if "E" in pol else {}
        rec["policy_ES"] = pol["ES"].describe() if "ES" in pol else {}
        rec["cap_filter"] = capf.reason
        rec["hist_positions"] = int(len(hist_paths))
        state["blocks"][bi] = {"origin": origin, "policies": pol, "cap": capf}
        state["origins"].append(rec)
        if log:
            log(f"block {bi + 1}/{len(starts)} origin {origin.date()} movers={ms.reason} dir={ds.reason[:60]} hist={len(hist_paths)}")
        if ms.ok:
            for k in range(s0, s1):
                d = panel.dates[k]
                idx = np.flatnonzero(kk == k)
                if len(idx) < cfg.n_picks:
                    continue
                pm = ms.score(_with_pat(XR.iloc[idx], pat_hook, X.iloc[idx], pat_cols, d))
                order = np.argsort(-pm, kind="stable")[:cfg.pool]
                sel = idx[order]
                R = XR.iloc[sel]
                dec = ds.decide(R)
                df = pd.DataFrame({"date": d, "ticker": lab.index.get_level_values(1)[sel], "block": bi, "rank": np.arange(len(sel)),
                                   "p_move": pm[order], "qual": (pm[order] >= ms.tau95) if ms.tau95 is not None else False,
                                   "kind": kind_all[sel], "fill_ok": lab["fill_ok"].to_numpy()[sel],
                                   "touch": y_mov[sel], "close": lab["close"].to_numpy()[sel], "hi": lab["hi"].to_numpy()[sel],
                                   "lo": lab["lo"].to_numpy()[sel], "p_up": dec["p_up"].values, "conf": dec["conf"].values,
                                   "side_eng": dec["side"].values, "eng_bet": dec["bet"].values, "p_soft": dec["p_soft"].values,
                                   "reason": dec["reason"].values, "tau95": ms.tau95 if ms.tau95 is not None else np.nan})
                if not cfg.allow_short:
                    neg = df["side_eng"] < 0
                    df.loc[neg, ["eng_bet"]] = False
                    df.loc[neg, "side_eng"] = 0
                top = df["rank"] < cfg.n_picks
                df["bet_noref"] = df["eng_bet"] & top
                ranked_bets = df["eng_bet"].cumsum()
                df["bet_refill"] = df["eng_bet"] & (ranked_bets <= cfg.n_picks)
                df["side_soft"] = np.where(df["p_soft"] >= 0.5, 1, -1) if cfg.allow_short else 1
                state["cands"].append(df)
                state["rows"].append(sel)
                lp = panel.paths(sel, 1.0 / cfg.n_picks, vol_all[sel], atr_all[sel], kind_all[sel])
                state["paths"].append(lp)
                state["soft"].append(reflect_short(lp, df["side_soft"].to_numpy()))
                state["week_dates"].append(d)
        state["done"] = bi + 1
        if ckpt:
            tmp = str(ckpt) + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(state, fh)
            Path(tmp).replace(ckpt)
    cands = pd.concat(state["cands"], ignore_index=True) if state["cands"] else pd.DataFrame()
    rows = np.concatenate(state["rows"]) if state["rows"] else np.zeros(0, int)
    return FVRun(cfg, panel, cands, rows, concat_paths(state["paths"]), state["blocks"], pd.DataFrame(state["origins"]),
                 pd.DatetimeIndex(state["week_dates"]))


def _with_pat(R, hook, X, cols, as_of):
    if hook is None or cols is None:
        return R
    pf = hook.features(X, as_of).add_prefix("pat_").astype("float32")
    return pd.concat([R, pf.reindex(R.index)], axis=1)


# ------------------------------------------------------------------------------------------------ evaluation
def stage_key(stages: str) -> str:
    """Normalise 'M+D+E+S' / 'MDES' / 'md' to the canonical stage string."""
    s = stages.upper().replace("+", "").replace(" ", "")
    if not s.startswith("M") or any(ch not in "MDES" for ch in s):
        raise ValueError(f"bad stage spec {stages!r}: use letters from M D E S, starting with M")
    if ("E" in s or "S" in s) and "D" not in s:
        raise ValueError("exits and stops need a direction stage (E/S without D is not a defined variant)")
    return "".join(ch for ch in "MDES" if ch in s)


def positions(run: FVRun, stages: str = "MDES") -> pd.DataFrame:
    """Every candidate the given stage combination trades, with its simulated result. Columns include date, ticker, side,
    net, gross, taken (False = left as cash: unfilled entry, skipped by the loss-cap filter) and reason code."""
    st = stage_key(stages)
    c, cfg = run.cands, run.cfg
    if len(c) == 0:
        return pd.DataFrame(columns=["date", "ticker", "side", "net", "gross", "taken", "block", "rank", "close", "touch"])
    if "D" in st:
        flag = c["bet_refill" if cfg.refill else "bet_noref"].to_numpy()
        side = c["side_eng"].to_numpy()
    else:
        flag = (c["rank"] < cfg.n_picks).to_numpy()
        if cfg.strict_movers:
            flag &= c["qual"].to_numpy(bool)
        side = np.ones(len(c), int)
    idx = np.flatnonzero(flag)
    out = c.iloc[idx][["date", "ticker", "block", "rank", "close", "touch", "p_move", "conf", "kind"]].copy()
    out["side"] = side[idx]
    net, gross = np.zeros(len(idx)), np.zeros(len(idx))
    days, code = np.zeros(len(idx), int), np.zeros(len(idx), int)
    taken = c["fill_ok"].to_numpy()[idx].astype(bool).copy()
    key = ("ES" if "E" in st and "S" in st else "E" if "E" in st else "S" if "S" in st else None)
    cost = cfg.cost()
    blocks = c["block"].to_numpy()[idx]
    for b in np.unique(blocks):
        m = np.flatnonzero(blocks == b)
        pth = reflect_short(run.paths.take(idx[m]), side[idx[m]])
        info = run.blocks[b]
        pol = info["policies"].get(key) if key else None
        res = pol.apply(pth) if pol is not None else run_exit(pth, ExitSpec(), cost)
        ok = taken[m].copy()
        if "S" in st:
            ok &= info["cap"].keep(pth)
        ok &= res.days > 0
        net[m], gross[m], days[m], code[m] = np.where(ok, res.net, 0.0), np.where(ok, res.gross, 0.0), res.days, res.reason
        taken[m] = ok
    out["net"], out["gross"], out["days"], out["reason_code"], out["taken"] = net, gross, days, code, taken
    return out.reset_index(drop=True)


def weekly_frame(run: FVRun, pos: pd.DataFrame) -> pd.DataFrame:
    """One row per traded decision week (weeks with no bet are rows of zeros - they are part of the record)."""
    n = run.cfg.n_picks
    wk = pd.DataFrame(index=run.week_dates)
    t = pos[pos["taken"]] if len(pos) else pos
    g = t.groupby("date") if len(t) else None
    wk["port"] = g["net"].sum().div(n).reindex(wk.index).fillna(0.0) if g is not None else 0.0
    wk["n_taken"] = g.size().reindex(wk.index).fillna(0).astype(int) if g is not None else 0
    def cnt(mask):
        return t[mask].groupby("date").size().reindex(wk.index).fillna(0).astype(int) if g is not None else 0
    if g is not None:
        wk["hit_net"] = cnt(t["net"] >= HIT)
        wk["hit_gross"] = cnt(t["gross"] >= HIT - 1e-9)
        wk["correct"] = cnt(t["side"] * t["close"] > 0)
        wk["cat"] = cnt(t["net"] <= run.cfg.cap)
        wk["neg10"] = cnt(t["net"] <= -HIT)
        wk["worst"] = t.groupby("date")["net"].min().reindex(wk.index).fillna(0.0)
    else:
        for k in ("hit_net", "hit_gross", "correct", "cat", "neg10"):
            wk[k] = 0
        wk["worst"] = 0.0
    return wk


def mover_weekly(run: FVRun) -> pd.DataFrame:
    """The mover stage on its own: hit rate of the top-n picks each week (V1)."""
    c = run.cands
    if len(c) == 0:
        return pd.DataFrame(columns=["touch", "close10", "exp", "qualified", "n"])
    top = c[c["rank"] < run.cfg.n_picks]
    g = top.groupby("date")
    wk = pd.DataFrame({"touch": g["touch"].sum(), "close10": g.apply(lambda d: (d["close"].abs() >= run.cfg.move).sum(), include_groups=False),
                       "exp": g["p_move"].mean(), "qualified": c.groupby("date")["qual"].sum(), "n": g.size()})
    return wk


def boot_index(n: int, B: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices (B, n) over weeks; block=1 is the ordinary bootstrap."""
    if n == 0:
        return np.zeros((B, 0), int)
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n, (B, nb))
    return ((st[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, -1)[:, :n]


def ci(stat: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(stat, a)), float(np.quantile(stat, 1 - a))


def summarize(run: FVRun, stages: str, pos: pd.DataFrame | None = None, boot: int = 400, block: int = 4, seed: int = 0) -> dict:
    """The owner's metrics for one stage combination with bootstrap CIs over weeks (ratio metrics are ratios of sums)."""
    pos = positions(run, stages) if pos is None else pos
    wk = weekly_frame(run, pos)
    n, cfg = len(wk), run.cfg
    out = {"stages": stage_key(stages), "weeks": n}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    bi = boot_index(n, boot, block, rng)
    A = {k: wk[k].to_numpy(float) for k in wk.columns}
    ratio = lambda num, den: (lambda i: A[num][i].sum(1) / np.maximum(A[den][i].sum(1), 1))
    pick_slots = np.full(n, float(cfg.n_picks))
    A["slots"] = pick_slots
    stats = {
        "mean_week": lambda i: A["port"][i].mean(1),
        "share10_net": lambda i: A["hit_net"][i].sum(1) / A["slots"][i].sum(1),
        "share10_gross": lambda i: A["hit_gross"][i].sum(1) / A["slots"][i].sum(1),
        "hit10_of_bets": ratio("hit_gross", "n_taken"),
        "dir_acc": ratio("correct", "n_taken"),
        "cat_rate": ratio("cat", "n_taken"),
        "pos_weeks": lambda i: (A["port"][i] > 0).mean(1),
        "goal8_weeks": lambda i: (A["hit_gross"][i] >= 8).mean(1),
        "coverage": lambda i: A["n_taken"][i].sum(1) / A["slots"][i].sum(1),
    }
    point = {"mean_week": A["port"].mean(), "share10_net": A["hit_net"].sum() / A["slots"].sum(),
             "share10_gross": A["hit_gross"].sum() / A["slots"].sum(), "hit10_of_bets": A["hit_gross"].sum() / max(A["n_taken"].sum(), 1),
             "dir_acc": A["correct"].sum() / max(A["n_taken"].sum(), 1), "cat_rate": A["cat"].sum() / max(A["n_taken"].sum(), 1),
             "pos_weeks": float((A["port"] > 0).mean()), "goal8_weeks": float((A["hit_gross"] >= 8).mean()),
             "coverage": A["n_taken"].sum() / A["slots"].sum()}
    for k, f in stats.items():
        lo, hi = ci(f(bi))
        out[k], out[k + "_lo"], out[k + "_hi"] = float(point[k]), lo, hi
    port = A["port"]
    eq = np.cumprod(1 + port)
    k5 = max(1, int(np.ceil(0.05 * n)))
    out.update(median_week=float(np.median(port)), sd_week=float(port.std()), in_band=float(((port >= 0.05) & (port <= 0.10)).mean()),
               cvar5=float(np.sort(port)[:k5].mean()), maxdd=float((1 - eq / np.maximum.accumulate(np.maximum(eq, 1.0))).max()),
               n_bets=int(A["n_taken"].sum()), worst_week=float(port.min()), worst_position=float(pos.loc[pos["taken"], "net"].min()) if pos["taken"].any() else 0.0)
    nc, nb = int(A["cat"].sum()), int(A["n_taken"].sum())
    out["cat_wilson_hi"] = wilson(nc, nb)[1] if nb else 1.0
    t = pos[pos["taken"]]
    out["share_le_m10"] = float((t["net"] <= -HIT).mean()) if len(t) else 0.0
    out["share_le_m15"] = float((t["net"] <= -0.15).mean()) if len(t) else 0.0
    out["ratio_up10_to_down10"] = float((t["net"] >= HIT).sum() / max((t["net"] <= -HIT).sum(), 1)) if len(t) else 0.0
    out["dir_acc_wilson"] = wilson(int(A["correct"].sum()), nb) if nb else (0.0, 1.0)
    return out


def era_table(run: FVRun, stages: str, years_per_era: int = 1) -> pd.DataFrame:
    """Per-era breakdown (calendar years by default) of the same metrics, no CIs (eras are short)."""
    pos = positions(run, stages)
    wk = weekly_frame(run, pos)
    n = run.cfg.n_picks
    mv = mover_weekly(run)
    era = (wk.index.year // years_per_era) * years_per_era
    rows = []
    for e, g in wk.groupby(era):
        m = mv.reindex(g.index)
        slots = n * len(g)
        rows.append({"era": int(e), "weeks": len(g), "mean_week": g["port"].mean(), "pos_weeks": (g["port"] > 0).mean(),
                     "in_band": ((g["port"] >= 0.05) & (g["port"] <= 0.10)).mean(), "mover_hit": m["touch"].sum() / max(m["n"].sum(), 1),
                     "coverage": g["n_taken"].sum() / slots, "dir_acc": g["correct"].sum() / max(g["n_taken"].sum(), 1),
                     "share10_gross": g["hit_gross"].sum() / slots, "goal8_weeks": (g["hit_gross"] >= 8).mean(),
                     "worst_position": g["worst"].min(), "cat": int(g["cat"].sum()), "bets": int(g["n_taken"].sum())})
    return pd.DataFrame(rows)


def type_table(run: FVRun, stages: str) -> pd.DataFrame:
    """Per stock-type breakdown of the bets actually taken."""
    pos = positions(run, stages)
    t = pos[pos["taken"]]
    if len(t) == 0:
        return pd.DataFrame(columns=["kind", "bets"])
    g = t.groupby("kind")
    return pd.DataFrame({"bets": g.size(), "dir_acc": g.apply(lambda d: (d["side"] * d["close"] > 0).mean(), include_groups=False),
                         "hit10_gross": g.apply(lambda d: (d["gross"] >= HIT - 1e-9).mean(), include_groups=False),
                         "mean_net": g["net"].mean(), "worst": g["net"].min(),
                         "cat": g.apply(lambda d: (d["net"] <= run.cfg.cap).sum(), include_groups=False)}).reset_index()


def mover_report(run: FVRun, boot: int = 400, block: int = 4, seed: int = 0) -> dict:
    """V1: hit rate of the mover picks (touch and close) vs the all-candidate base rate, with a CI, tau95 coverage, and the
    calibration of the predicted probability."""
    mv = mover_weekly(run)
    c = run.cands
    if len(mv) == 0:
        return {"weeks": 0}
    rng = np.random.default_rng(seed)
    bi = boot_index(len(mv), boot, block, rng)
    tot = float(run.cfg.n_picks)
    t, cl, ex = mv["touch"].to_numpy(float), mv["close10"].to_numpy(float), mv["exp"].to_numpy(float)
    hit_ci = ci(t[bi].sum(1) / (tot * len(mv)))
    q = mv["qualified"].to_numpy(float)
    tau = c["tau95"].dropna()
    top = c[c["rank"] < run.cfg.n_picks]
    cal = pd.DataFrame({"p": top["p_move"].values, "y": top["touch"].values})
    cal["bin"] = pd.qcut(cal["p"], 5, duplicates="drop") if cal["p"].nunique() > 5 else 0
    calib = cal.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return {"weeks": len(mv), "hit_touch": float(t.sum() / (tot * len(mv))), "hit_touch_lo": hit_ci[0], "hit_touch_hi": hit_ci[1],
            "hit_close10": float(cl.sum() / (tot * len(mv))), "predicted_mean": float(ex.mean()),
            "base_rate_pool": float(c["touch"].mean()),
            "base_rate_all": float(np.nanmean(run.panel.lab.loc[run.panel.lab["k"].isin(np.flatnonzero(run.panel.dates.isin(run.week_dates))), "touch"].to_numpy(float))), "target_95_met": bool(t.sum() / (tot * len(mv)) >= 0.95),
            "weeks_with_10_qualified": float((q >= run.cfg.n_picks).mean()), "mean_qualified": float(q.mean()),
            "tau95_available": float(tau.notna().mean()) if len(tau) else 0.0, "calibration": calib}


def ablation(run: FVRun, variants: dict | None = None, boot: int = 400, block: int = 4, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each named variant's metrics with CIs, and the PAIRED difference of consecutive variants in mean weekly return and
    in share10_gross (same resampled weeks), so a stage's contribution is separated from the luck of the weeks."""
    variants = variants or ABLATIONS
    rows, weekly, names = [], {}, list(variants)
    for name in names:
        pos = positions(run, variants[name])
        weekly[name] = weekly_frame(run, pos)
        rows.append({"variant": name, **summarize(run, variants[name], pos, boot, block, seed)})
    tab = pd.DataFrame(rows)
    n = len(run.week_dates)
    rng = np.random.default_rng(seed + 1)
    bi = boot_index(n, boot, block, rng)
    diffs = []
    for a, b in zip(names[:-1], names[1:]):
        wa, wb = weekly[a], weekly[b]
        if n == 0:
            continue
        d_ret = wb["port"].to_numpy() - wa["port"].to_numpy()
        d_hit = (wb["hit_gross"].to_numpy() - wa["hit_gross"].to_numpy()) / run.cfg.n_picks
        lo_r, hi_r = ci(d_ret[bi].mean(1))
        lo_h, hi_h = ci(d_hit[bi].mean(1))
        diffs.append({"from": a, "to": b, "d_mean_week": float(d_ret.mean()), "d_mean_week_lo": lo_r, "d_mean_week_hi": hi_r,
                      "d_share10": float(d_hit.mean()), "d_share10_lo": lo_h, "d_share10_hi": hi_h,
                      "verdict": "helps" if lo_r > 0 else "hurts" if hi_r < 0 else "not distinguishable"})
    return tab, pd.DataFrame(diffs)


def goal_verdict(summary: dict, goal_share: float = 0.80, cap: float = -0.20) -> dict:
    """V5 in plain terms: did the pipeline reach 8 of 10 finishing +10%, with losers held to the cap? Uses the CI, not the
    point estimate."""
    if summary.get("weeks", 0) == 0:
        return {"goal_met": False, "why": "no traded weeks"}
    ok_share = summary["share10_gross_lo"] >= goal_share
    ok_cap = summary["worst_position"] > cap
    return {"goal_met": bool(ok_share and ok_cap), "share10_gross": summary["share10_gross"],
            "share10_gross_lo": summary["share10_gross_lo"], "worst_position": summary["worst_position"],
            "why": ("goal reached" if ok_share and ok_cap else
                    f"share10 lower bound {summary['share10_gross_lo']:.2f} < {goal_share:.2f}" if not ok_share else
                    f"worst position {summary['worst_position']:.2f} breached {cap:.2f}")}


# ------------------------------------------------------------------------------------------------ audits
def audit_no_lookahead(bars: dict, cfg: FVConfig, cut, seed: int = 11, last_block: int | None = None) -> dict:
    """Future-scramble test: replace every bar AFTER `cut` by noise and rerun. Decisions dated on or before `cut` must be
    identical (same tickers, ranks, sides, bets) - anything else means the pipeline read the future."""
    cut = pd.Timestamp(cut)
    rng = np.random.default_rng(seed)
    b2 = {k: v.copy() for k, v in bars.items()}
    m = b2["Close"].index > cut
    noise = np.exp(rng.normal(0, 0.3, b2["Close"].loc[m].shape))      # one factor per bar keeps high >= low
    for k in BAR_KEYS:
        b2[k].loc[m] = b2[k].loc[m].to_numpy() * noise
    a = walk_forward(build_panel(bars, cfg), cfg, last_block=last_block)
    b = walk_forward(build_panel(b2, cfg), cfg, last_block=last_block)
    ca = a.cands[a.cands["date"] <= cut].reset_index(drop=True)
    cb = b.cands[b.cands["date"] <= cut].reset_index(drop=True)
    cols = ["date", "ticker", "rank", "side_eng", "eng_bet"]
    same = len(ca) == len(cb) and len(ca) > 0 and ca[cols].equals(cb[cols])
    pdiff = float(np.abs(ca["p_move"].to_numpy() - cb["p_move"].to_numpy()).max()) if len(ca) == len(cb) and len(ca) else float("nan")
    return {"identical": bool(same), "n_compared": int(len(ca)), "max_p_move_diff": pdiff}


def audit_fill_timing(run: FVRun) -> dict:
    """C33 check on the recorded paths: entry is the open of the session AFTER the decision date, five sessions at most, and
    no bar falls on a weekend."""
    P, lab = run.panel, run.panel.lab
    k = lab["k"].to_numpy()[run.rows]
    first_bar = P.sessions[np.minimum(P.dec[k] + 1, len(P.sessions) - 1)]
    late_entry = int((first_bar <= P.dates[k]).sum())
    span = (P.last[k] - P.dec[k])
    weekend = int((P.sessions[P.last[k]].dayofweek >= 5).sum() + (first_bar.dayofweek >= 5).sum())
    return {"positions": int(len(k)), "entry_not_after_decision": late_entry, "max_bars": int(span.max()) if len(k) else 0,
            "weekend_bars": weekend, "ok": bool(late_entry == 0 and weekend == 0 and (len(k) == 0 or span.max() <= D_BARS))}


# ------------------------------------------------------------------------------------------------ reporting
def format_report(run: FVRun, tab: pd.DataFrame, diffs: pd.DataFrame, mover: dict, eras: pd.DataFrame, verdict: dict) -> str:
    p = lambda x: f"{100 * x:.1f}%"
    L = ["# Find volatility pipeline (V1-V5)", "",
         f"weeks traded: {len(run.week_dates)}   origins: {len(run.origins)}   picks/week: {run.cfg.n_picks}   gate: {run.cfg.gate}", "",
         "## V1 movers (touch +-10% inside the week)"]
    if mover.get("weeks", 0):
        L += [f"- hit rate {p(mover['hit_touch'])} (90% CI {p(mover['hit_touch_lo'])}-{p(mover['hit_touch_hi'])}) vs base rate of all "
              f"tradable stocks {p(mover['base_rate_all'])}; predicted {p(mover['predicted_mean'])}",
              f"- 95% target {'MET' if mover['target_95_met'] else 'NOT met'}; weeks with >=10 candidates above tau95: "
              f"{p(mover['weeks_with_10_qualified'])} (mean {mover['mean_qualified']:.1f} qualified)"]
    L += ["", "## Ablation (same picks, paired CIs)", "", tab[["variant", "weeks", "mean_week", "mean_week_lo", "mean_week_hi", "share10_gross",
                                                         "share10_gross_lo", "coverage", "dir_acc", "cat_rate", "worst_position", "goal8_weeks"]]
          .round(4).to_string(index=False), "", diffs.round(4).to_string(index=False) if len(diffs) else "", "",
          "## Per era (full pipeline)", "", eras.round(3).to_string(index=False), "", "## V5 verdict", "",
          f"- {verdict['why']}"]
    L += ["", "No stop guarantees a floor: gaps fill through stops. Read cat_rate and cat_wilson_hi, not a promise."]
    return "\n".join(L)


def save_results(out: str | Path, run: FVRun, tab, diffs, mover, eras, types, verdict, extra: dict | None = None) -> Path:
    """Write CSVs + report + provenance (config hash, seed, code hash) so the numbers can be re-derived."""
    from . import provenance
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "ablation.csv", index=False)
    diffs.to_csv(out / "ablation_paired.csv", index=False)
    eras.to_csv(out / "eras.csv", index=False)
    types.to_csv(out / "types.csv", index=False)
    run.origins.drop(columns=[c for c in run.origins.columns if c.startswith("policy_")]).to_csv(out / "origins.csv", index=False)
    run.cands.to_parquet(out / "candidates.parquet")
    (out / "report.md").write_text(format_report(run, tab, diffs, mover, eras, verdict), encoding="utf-8")
    prov = {"config": asdict(run.cfg), "config_hash": run.cfg.hash(), "seed": run.cfg.seed, "verdict": verdict,
            "mover": {k: v for k, v in mover.items() if k != "calibration"}, "policies": {int(b): {k: v.describe() for k, v in i["policies"].items()}
                                                                                            for b, i in run.blocks.items()},
            **(extra or {})}
    try:
        prov["stamp"] = provenance.stamp(asdict(run.cfg), run.cfg.seed)
    except Exception as e:
        prov["stamp_error"] = repr(e)[:200]
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    return out
