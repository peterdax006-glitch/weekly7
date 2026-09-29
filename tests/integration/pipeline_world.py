"""Shared harness for the level-2 integration tests (Bible PHASE 32): a small synthetic market with a KNOWN, planted,
forward-looking effect, and a compact features -> pattern miner -> session-snapshot pipeline built from the real engine
modules (engine.candles, engine.patterns, engine.adaptive). Not a test file: helpers only, synthetic data only.

The planted effect: a stock whose day closed in the top fifth of its own high-low range (cd_pos >= 0.8) earns `eff`
extra return over the next five sessions. The candle feature that carries it is computed by the real feature code, so
the miner has to find it through the same path production uses. Every draw takes an explicit seed.

Pipeline contract (what the tests prove): everything used for a decision on `as_of` is sliced from data <= as_of INSIDE
the pipeline, labels are used only once realised, and the only planted defect (`leak="labels"`) is the one the scramble
test must expose."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import adaptive, candles
from engine.patterns import PatternMiner

HORIZON = 5                                   # sessions; matches the weekly rebalance cadence of the Session
WARMUP = 30                                   # candle features need ~21 sessions of history (monthly candle)
MIN_TRAIN_WEEKS = 40                          # below this the miner is not asked: weekly t-tests need some weeks
FEATURES = ["cd_body", "cd_upwick", "cd_lowwick", "cd_pos", "cd_range", "cw_pos", "cw_body", "streak", "gap",
            "week_vs_month_pos"]
MINER = {"max_pairs": 120, "max_unless": 20, "null_reps": 2, "min_n": 150, "half_life_years": 50, "seed": 7}
CFG = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0,
       "pick": "top", "pool_q": 0.7, "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}


def make_world(seed=0, n_days=900, n_tickers=50, eff=0.03, names=None):
    """{'Open','High','Low','Close'} DataFrames on business days starting on a Monday (weeks are 5 sessions)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n_days)
    tk = names or [f"S{i:03d}" for i in range(n_tickers)]
    k = len(tk)
    u = rng.random((n_days, k))                                    # where the close sits in the day's range
    csum = np.vstack([np.zeros((1, k)), np.cumsum(u >= 0.8, axis=0)]).astype(float)
    t = np.arange(n_days)
    # bump[t] = eff/HORIZON * (# signal days among the previous HORIZON sessions): the effect only ever looks BACK
    bump = eff / HORIZON * (csum[t] - csum[np.maximum(t - HORIZON, 0)])
    gap = rng.normal(0, 0.012, (n_days, k)) + rng.normal(0, 0.006, (n_days, 1))
    a = np.abs(rng.normal(0, 0.008, (n_days, k))) + 0.002
    b = np.abs(rng.normal(0, 0.008, (n_days, k))) + 0.002
    O, H, L, C = (np.empty((n_days, k)) for _ in range(4))
    prev = np.full(k, 30.0)
    for i in range(n_days):
        O[i] = prev * np.exp(gap[i] + bump[i])
        L[i], H[i] = O[i] * np.exp(-a[i]), O[i] * np.exp(b[i])
        C[i] = L[i] + u[i] * (H[i] - L[i])
        prev = C[i]
    return {n: pd.DataFrame(v, index=idx, columns=tk) for n, v in (("Open", O), ("High", H), ("Low", L), ("Close", C))}


def scramble_after(stocks, cut, seed=1):
    """Replace every bar strictly after `cut` with noise (multiplicative, same shape): the future must not matter."""
    rng = np.random.default_rng(seed)
    m = stocks["Close"].index > pd.Timestamp(cut)
    noise = np.exp(rng.normal(0, 0.25, stocks["Close"].loc[m].shape))
    out = {}
    for name, df in stocks.items():
        d = df.copy()
        d.loc[m] = d.loc[m].values * noise
        out[name] = d
    return out


def permute_tickers(stocks, seed=2):
    """Disguise: rename every ticker and shuffle the column order. Returns (stocks, old->new mapping)."""
    rng = np.random.default_rng(seed)
    old = list(stocks["Close"].columns)
    new = [f"Z{j:03d}" for j in rng.permutation(len(old))]
    mp = dict(zip(old, new))
    order = list(rng.permutation(len(old)))
    return {n: df.rename(columns=mp).iloc[:, order] for n, df in stocks.items()}, mp


def decision_days(closes, start):
    """Last session of each calendar week from `start` on (where the Session asks for a snapshot)."""
    idx = closes.index[closes.index >= pd.Timestamp(start)]
    wk = pd.Series(idx.isocalendar().week.to_numpy(), index=idx)
    return list(idx[wk.ne(wk.shift(-1)).to_numpy()])


class Pipeline:
    """features -> miner -> snapshot, every step causal in `as_of`. leak='labels' plants a look-ahead defect: the miner
    is trained on labels read from the FULL price history, so labels not yet realised at as_of leak in (subtle: one
    row). leak='peek' adds the next HORIZON sessions' return to the score (blatant: it changes decisions)."""

    def __init__(self, stocks, miner_params=None, leak=None):
        self.stocks, self.params, self.leak = stocks, {**MINER, **(miner_params or {})}, leak
        self.fit_log = []                              # (as_of, last session any training label depends on, n rows)

    def _hist(self, as_of):
        return {k: v.loc[:pd.Timestamp(as_of)] for k, v in self.stocks.items()}

    def panel(self, as_of):
        """(X on weekly decision dates <= as_of, closes <= as_of), computed from data <= as_of only."""
        h = self._hist(as_of)
        F = candles.build(h)
        idx = h["Close"].index
        dates = idx[WARMUP:][(np.arange(WARMUP, len(idx)) % HORIZON) == HORIZON - 1]
        tk = h["Close"].columns
        X = pd.DataFrame({c: F[c].loc[dates].to_numpy(dtype="float32").ravel() for c in FEATURES},
                         index=pd.MultiIndex.from_product([dates, tk], names=["date", "ticker"]))
        vix = pd.Series(np.sin(np.arange(len(idx)) / 40.0), index=idx)     # a smooth market-context reading
        X["m_vix"] = np.repeat(vix.loc[dates].to_numpy(), len(tk))
        return X, h["Close"]

    def fit(self, as_of):
        """Fit the miner on rows whose label was realised by as_of. Returns None when there is too little history."""
        as_of = pd.Timestamp(as_of)
        X, closes = self.panel(as_of)
        dates = X.index.get_level_values(0)
        label_src = self.stocks["Close"] if self.leak == "labels" else closes     # the defect reads the future
        fwd = (label_src.shift(-HORIZON) / label_src - 1).reindex(dates.unique())
        y = pd.Series(fwd.to_numpy().ravel(), index=X.index)
        y = y - y.groupby(level=0).transform("mean")
        pos = pd.Series(np.arange(len(closes)), index=closes.index)
        keep = (pos.reindex(dates).to_numpy() + HORIZON <= len(closes) - 1) | (self.leak == "labels")
        Xt, yt = X[keep], y[keep]
        n_weeks = Xt.index.get_level_values(0).nunique()
        if n_weeks < MIN_TRAIN_WEEKS:
            self.fit_log.append((as_of, None, 0))
            return None
        last = int(pos.reindex(Xt.index.get_level_values(0)).max()) + HORIZON
        self.fit_log.append((as_of, label_src.index[min(last, len(label_src) - 1)], len(Xt)))
        return PatternMiner(self.params).fit(Xt, yt, now=as_of)

    def feature_row(self, day):
        """(features of `day` as the miner scores them, closes <= day, candle dict), from data <= day only."""
        day = pd.Timestamp(day)
        h = self._hist(day)
        F = candles.build(h)
        row = pd.DataFrame({c: F[c].loc[day].astype(float) for c in FEATURES})
        row["m_vix"] = float(np.sin((len(h["Close"]) - 1) / 40.0))
        return row, h["Close"], F

    def snapshot(self, day, miner):
        """The Session's decision table for `day`: pattern score as mu_raw plus the risk columns policy.pick reads."""
        day = pd.Timestamp(day)
        row, C, F = self.feature_row(day)
        tk = C.columns
        score = miner.score(row) if miner is not None else pd.Series(0.0, index=tk)
        if self.leak == "peek":                                   # planted defect: reads the next HORIZON sessions
            full = self.stocks["Close"]
            j = full.index.get_loc(day)
            ahead = full.iloc[min(j + HORIZON, len(full) - 1)] / full.iloc[j] - 1
            score = score + 10.0 * ahead.reindex(tk)
        r = C.pct_change()
        tiebreak = 1e-6 * F["cd_pos"].loc[day].rank(pct=True).reindex(tk).fillna(0.5)   # name-independent tie break
        return pd.DataFrame({"mu_raw": score.to_numpy() + tiebreak.to_numpy(), "evidence": 0.5,
                             "vol20": r.iloc[-20:].std().reindex(tk).fillna(0.02).to_numpy(),
                             "max20": r.iloc[-20:].abs().max().reindex(tk).fillna(0.05).to_numpy(), "log_dv": 18.0,
                             "ev_red_flag": 0.0, "ev_offering": 0.0,
                             "r5": (C.iloc[-1] / C.iloc[max(len(C) - 6, 0)] - 1).to_numpy(),
                             "m_vix": row["m_vix"].iloc[0], "m_vix_term": 0.9, "m_spy_ma200": 1.05},
                            index=pd.Index(tk, name="ticker"))

    def snapshots(self, days, refit_every=8):
        """{date str: snapshot} for decision `days`, refitting the miner every `refit_every` decisions (walk-forward)."""
        snaps, miner = {}, None
        for i, d in enumerate(days):
            if i % refit_every == 0:
                miner = self.fit(d)
            snaps[str(pd.Timestamp(d).date())] = self.snapshot(d, miner)
        return snaps


START = 480                                   # default first replayed session


def replay_chain(stocks, start=START, end=None, leak=None, refit_every=8, snaps=None, cfg=None):
    """Walk-forward chain: refit the miner every `refit_every` decisions, snapshot each decision day, replay the Session
    from session `start`. `snaps` substitutes ready-made snapshots (controls). Returns (Session, Pipeline, snapshots)."""
    C, O = stocks["Close"], stocks["Open"]
    first = C.index[start]                                       # the first session builds the initial portfolio
    days = [first] + [d for d in decision_days(C, first) if d > first]
    if end is not None:
        days = [d for d in days if d <= pd.Timestamp(end)]
    P = Pipeline(stocks, leak=leak)
    sn = snaps if snaps is not None else P.snapshots(days, refit_every)
    Cs, Os = C.iloc[start:], O.iloc[start:]
    S = adaptive.replay(cfg or CFG, sn, Cs, 0.0, {}, opens=Os)
    return S, P, sn
