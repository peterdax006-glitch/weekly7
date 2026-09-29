"""Analog engine core, feature weighting and ablation (Bible PHASE 8: 8.2, 8.3, 8.4, 8.7, 8.8; serves canon C41).

KNNEngine       one nearest-analog machine over any (fingerprints F, outcomes O) pair - market, sector or stock. It
                enforces the phase-8 rules in one place: point-in-time standardisation (expanding moments, only rows
                up to the query day), analog end >= `gap` sessions before the query (and >= the outcome horizon, so
                every analog's outcome was fully known), analogs >= `sep` sessions apart, one analog per episode.
learn_weights_walk_forward
                feature weights trained walk-forward: at each refit date only rows and outcomes known by then are used,
                with an embargo, and the learned weights are kept only if they beat uniform weights on a later
                held-out block; otherwise that refit reverts to uniform. No full-history optimisation exists here.
predict_series / ablation_table
                the 8.8 comparison: no analog, unweighted, weighted, random analogs, block-shuffled outcomes and a
                nearest-neighbour control with random feature weights, scored on the same days with paired tests."""
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

METRICS = ("euclid", "manhattan", "cosine")
ZCLIP = 5.0                    # winsorise z-scores: one wild feature must not decide a neighbourhood
MAX_NAN_FRAC = 1 / 3           # a pool row missing more than this share of the used features is not comparable


def publication_lag(df, lags, default=1):
    """Shift each column by its publication delay in sessions so a value is only used once public (point-in-time)."""
    out = df.copy()
    for c in df.columns:
        out[c] = df[c].shift(int(lags.get(c, default)))
    return out


def pit_moments(F, min_periods=60):
    """Expanding mean / std / coverage per row: row i sees only rows <= i (8.2 - never normalise with the future)."""
    mu = F.expanding(min_periods=min_periods).mean().values
    sd = F.expanding(min_periods=min_periods).std().values
    sd = np.where(sd > 1e-12, sd, np.nan)
    cov = F.notna().astype(float).expanding().mean().values
    return mu, sd, cov


def weighted_distance(Z, zt, w, metric="euclid"):
    """Distance from every row of Z to the vector zt under feature weights w (same scale for every metric: ~1 is 'typical')."""
    w = np.asarray(w, float)
    ws = w.sum()
    if not np.isfinite(ws) or ws <= 0:
        raise ValueError("feature weights must have a positive sum")
    diff = Z - zt
    if metric == "euclid":
        return np.sqrt((diff ** 2) @ w / ws)
    if metric == "manhattan":
        return np.abs(diff) @ w / ws
    if metric == "cosine":                      # direction of the state, not its size: chord distance on the unit sphere
        sw = np.sqrt(w)
        A, b = Z * sw, zt * sw
        cos = (A @ b) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-12)
        return np.sqrt(np.clip(2 * (1 - cos), 0, None))
    raise ValueError(f"unknown metric {metric!r}; use one of {METRICS}")


def select_episodes(order, k, sep):
    """Walk candidates nearest-first and keep one per episode: a pick must be >= sep sessions from every pick so far."""
    chosen = []
    for j in order:
        if all(abs(int(j) - c) >= sep for c in chosen):
            chosen.append(int(j))
            if len(chosen) == k:
                break
    return chosen


def kernel_weights(d, ref, bw=0.5):
    """Gaussian kernel with a scale-free bandwidth (bw x the median pool distance); uniform if everything underflows."""
    if not np.isfinite(ref) or ref <= 0:
        return np.full(len(d), 1.0 / len(d))
    k = np.exp(-(np.asarray(d) / (bw * ref)) ** 2)
    s = k.sum()
    return k / s if s > 1e-300 else np.full(len(d), 1.0 / len(d))


def price_state(px, mkt=None, min_obs=60):
    """Shared price-only fingerprint for a sector index or a stock (8.1 at sector/stock scale): multi-scale returns,
    acceleration, drawdown, distance from MA200, volatility and its ratio; with a market series also relative strength,
    63-session correlation and 126-session beta. Row t uses data <= t only."""
    px = px.astype(float)
    r = np.log(px / px.shift(1))
    F = pd.DataFrame(index=px.index)
    for n, lab in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        F[f"ret_{lab}"] = np.log(px / px.shift(n))
    F["accel"] = F["ret_6m"] - F["ret_12m"] / 2
    F["drawdown"] = px / px.rolling(252, min_periods=min_obs).max() - 1
    F["dist_ma200"] = px / px.rolling(200, min_periods=100).mean() - 1
    F["vol_1m"] = r.rolling(21, min_periods=15).std() * np.sqrt(252)
    F["vol_ratio"] = r.rolling(21, min_periods=15).std() / r.rolling(252, min_periods=100).std()
    if mkt is not None:
        m = mkt.reindex(px.index).ffill().astype(float)
        mr = np.log(m / m.shift(1))
        for n, lab in ((21, "1m"), (63, "3m"), (126, "6m")):
            F[f"rel_{lab}"] = F[f"ret_{lab}"] - np.log(m / m.shift(n))
        F["corr_mkt"] = r.rolling(63, min_periods=40).corr(mr)
        F["beta_mkt"] = r.rolling(126, min_periods=80).cov(mr) / mr.rolling(126, min_periods=80).var()
    return F


def forward_outcomes(px, h, mkt=None):
    """What followed each day: fwd_ret_{h}d, fwd_vol_{h}d, fwd_maxdd_{h}d (+ fwd_rel_{h}d vs a market series). Outcomes span
    sessions t+1..t+h and are used only for analogs that ended >= gap >= h sessions before the query."""
    px = px.astype(float)
    r = np.log(px / px.shift(1))
    O = pd.DataFrame(index=px.index)
    O[f"fwd_ret_{h}d"] = np.log(px.shift(-h) / px)
    O[f"fwd_vol_{h}d"] = r.rolling(h, min_periods=max(3, h // 2)).std().shift(-h) * np.sqrt(252)
    O[f"fwd_maxdd_{h}d"] = (px.rolling(h).min().shift(-h) / px - 1).clip(upper=0.0)
    if mkt is not None:
        m = mkt.reindex(px.index).ffill().astype(float)
        O[f"fwd_rel_{h}d"] = O[f"fwd_ret_{h}d"] - np.log(m.shift(-h) / m)
    return O


@dataclass
class WeightSchedule:
    """Feature weights by refit date. `at(t)` returns the latest weights fitted on or before t (None = uniform)."""
    dates: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def at(self, t):
        if not self.dates:
            return None
        j = int(np.searchsorted(np.array(self.dates, dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(t)), side="right")) - 1
        return self.weights[j] if j >= 0 else None


class KNNEngine:
    """Nearest-analog search over (F, O). F: fingerprints (date index, feature columns). O: outcomes on the same index
    (columns named fwd_ret_*, fwd_vol_*, fwd_maxdd_*; `target` is the primary one). `horizon` = sessions an outcome spans."""

    def __init__(self, F, O, target, horizon=21, gap=63, sep=21, metric="euclid", min_hist=250, min_features=4,
                 min_moment_rows=60):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        if target not in O.columns:
            raise KeyError(f"target {target!r} not in outcomes")
        if not F.index.equals(O.index):
            raise ValueError("fingerprints and outcomes must share one index")
        if gap < horizon:
            raise ValueError("gap must be >= horizon so every analog outcome is fully known at the query")
        self.F, self.O, self.target = F, O, target
        self.horizon, self.gap, self.sep, self.metric = horizon, gap, sep, metric
        self.min_hist, self.min_features = min_hist, min_features
        self.cols = list(F.columns)
        self.Fv = F.values.astype(float)
        self.Ov = O.values.astype(float)
        self.ti = list(O.columns).index(target)
        self.mu, self.sd, self.cov = pit_moments(F, min_moment_rows)
        self.schedule = WeightSchedule()
        self._base = self._expanding_target_mean()

    def clone(self, F=None, O=None, metric=None):
        """A fresh engine with the same settings (empty weight schedule); optionally on replaced frames or another metric."""
        return KNNEngine(self.F if F is None else F, self.O if O is None else O, self.target, self.horizon, self.gap,
                         self.sep, metric or self.metric, self.min_hist, self.min_features)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _expanding_target_mean(self):
        y = pd.Series(self.Ov[:, self.ti])
        return y.expanding(min_periods=20).mean().shift(self.gap).values     # only outcomes >= gap sessions old

    def baseline(self, i):
        """'No analog' forecast: mean outcome of all sessions eligible at i."""
        return float(self._base[i]) if np.isfinite(self._base[i]) else np.nan

    def _prep(self, i):
        """(cols idx, pool positions, Z pool, z today) for query row i, or None when history/features are short."""
        lim = i - self.gap
        if lim < self.min_hist:
            return None
        x = self.Fv[i]
        ok = np.isfinite(x) & (self.cov[i] >= 0.5) & np.isfinite(self.sd[i]) & np.isfinite(self.mu[i])
        c = np.flatnonzero(ok)
        if len(c) < self.min_features:
            return None
        Zp = np.clip((self.Fv[: lim + 1, c] - self.mu[i, c]) / self.sd[i, c], -ZCLIP, ZCLIP)
        miss = np.isnan(Zp)
        valid = (miss.mean(axis=1) <= MAX_NAN_FRAC) & np.isfinite(self.Ov[: lim + 1, self.ti])
        pos = np.flatnonzero(valid)
        if len(pos) < 3 * self.sep:
            return None
        Z = np.where(miss[pos], 0.0, Zp[pos])
        zt = np.clip((x[c] - self.mu[i, c]) / self.sd[i, c], -ZCLIP, ZCLIP)
        return c, pos, Z, zt

    def _weights(self, cols_idx, w, date):
        if w is None:
            w = self.schedule.at(date)
        if w is None:
            return np.ones(len(cols_idx))
        vec = pd.Series(w).reindex([self.cols[j] for j in cols_idx]).fillna(1.0).values.astype(float)
        return vec if vec.sum() > 0 else np.ones(len(cols_idx))

    # ---- the query -------------------------------------------------------------------------------------------
    def query(self, i, k=5, w=None, uniform=False, mode="nn", rng=None, bw=0.5, precise_uniqueness=True):
        """Analogs for row position i. mode: 'nn' (nearest), 'random' (random eligible days, same episode rules),
        'shuffled' (nearest days, outcomes taken from a block-permuted history: breaks fingerprint->outcome link).
        w: explicit weights (Series by feature name); uniform=True forces equal weights; otherwise the schedule.
        precise_uniqueness=False skips the nearest-neighbour yardstick (25 extra distance scans) and divides by the median
        pool distance instead: what bulk scoring loops want, since they never read uniqueness."""
        prep = self._prep(i)
        if prep is None:
            return None
        c, pos, Z, zt = prep
        date = self.F.index[i]
        wv = np.ones(len(c)) if uniform else self._weights(c, w, date)
        d = weighted_distance(Z, zt, wv, self.metric)
        if mode == "random":
            if rng is None:
                raise ValueError("mode='random' needs an explicit rng")
            order = rng.permutation(len(pos))
        else:
            order = np.argsort(d, kind="stable")
        picked = select_episodes(pos[order], k, self.sep)                  # positions in F
        if not picked:
            return None
        loc = np.searchsorted(pos, picked)                                 # index within the pool
        out_pos = np.array(picked)
        if mode == "shuffled":
            if rng is None:
                raise ValueError("mode='shuffled' needs an explicit rng")
            blocks = [np.arange(s, min(s + self.sep, len(pos))) for s in range(0, len(pos), self.sep)]
            perm = np.concatenate([blocks[b] for b in rng.permutation(len(blocks))])
            out_pos = pos[perm[loc]]
        A_dates = self.F.index[picked]
        return summarise(A_dates, self.F.index[i], d[loc], d, self.Ov[out_pos], list(self.O.columns), self.target, k,
                         self.sep, len(c), len(pos), bw, i - np.array(picked),
                         nn_reference(Z, pos, wv, self.metric, self.gap) if precise_uniqueness else None, uniform_kernel=(mode == "random"))


def weighted_quantile(x, w, qs):
    """Quantiles of x under weights w (interpolated on the weighted empirical CDF midpoints). NaNs in x are ignored."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return [np.nan for _ in qs]
    x, w = x[ok], w[ok]
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, cdf, x)) for q in qs]


def nn_reference(Z, pos, w, metric, gap, m=25, min_cand=20):
    """Median over `m` evenly spaced pool rows of that row's own nearest-neighbour distance among rows >= gap sessions
    before it: the yardstick for 'how close does history usually get to a state it has not seen'. NaN if too little data."""
    n = len(pos)
    out = []
    for q in np.unique(np.linspace(0, n - 1, m).astype(int)):
        cnt = int(np.searchsorted(pos, pos[q] - gap, side="right"))
        if cnt >= min_cand:
            out.append(float(weighted_distance(Z[:cnt], Z[q], w, metric).min()))
    return float(np.median(out)) if out else np.nan


def summarise(dates, today, d_pick, d_all, O, ocols, target, k, sep, n_features, n_pool, bw=0.5, age_sessions=None,
              nn_ref=None, uniform_kernel=False):
    """Turn a chosen analog set into the 8.7 output: dates, distances, ages, forecast return / vol / drawdown, forecast
    spread, uniqueness (nearest distance over the typical nearest-neighbour distance of past days, `nn_ref`; >1 means
    today is farther from history than history usually is from itself; falls back to the median pool distance), analog count,
    close-analog count and a 0-1 confidence = (effective analogs / k) x closeness x directional agreement above 50%."""
    ref = float(np.median(d_all))
    kw = np.full(len(d_pick), 1.0 / len(d_pick)) if uniform_kernel else kernel_weights(d_pick, ref, bw)
    cnt = np.isfinite(O)
    wm = kw[:, None] * cnt
    fc = (kw[:, None] * np.where(cnt, O, 0.0)).sum(axis=0) / np.where(wm.sum(axis=0) > 0, wm.sum(axis=0), np.nan)
    pred = dict(zip(ocols, fc))
    ti = ocols.index(target)
    y, f = O[:, ti], pred[target]
    sd_y = float(np.sqrt((kw * (y - f) ** 2).sum())) if np.isfinite(f) else np.nan
    n_eff = float(1.0 / (kw ** 2).sum())
    agree = float((kw * (np.sign(y) == np.sign(f))).sum()) if np.isfinite(f) else 0.0
    closeness = float(np.exp(-(np.mean(d_pick) / ref) ** 2)) if ref > 0 else 0.0
    conf = float(np.clip((n_eff / k) * closeness * ((2 * agree - 1) if agree > 0.5 else 0.0), 0, 1))
    A = pd.DataFrame({"date": pd.DatetimeIndex(dates), "distance": d_pick, "weight": kw,
                      "age_sessions": age_sessions if age_sessions is not None else np.nan,
                      "age_years": [(today - t).days / 365.25 for t in dates]})
    for j, name in enumerate(ocols):
        A[name] = O[:, j]
    q10, q50, q90 = weighted_quantile(y, kw, (0.1, 0.5, 0.9))
    return {"analogs": A, "forecast_return": f, "forecast_return_sd": sd_y,
            "forecast_quantiles": {"p10": q10, "p50": q50, "p90": q90},
            "prob_up": float((kw * (y > 0)).sum()) if np.isfinite(f) else np.nan,
            "forecast_vol": _pick(pred, "fwd_vol"), "forecast_drawdown": _pick(pred, "fwd_maxdd"),
            "prediction": pred, "nearest_distance": float(np.min(d_all)),
            "uniqueness": float(np.min(d_all) / (nn_ref if nn_ref and nn_ref > 0 else ref)) if ref > 0 else np.nan,
            "n_analogs": len(dates), "n_close": int((np.asarray(d_all) < 1.0).sum()), "n_pool": int(n_pool),
            "confidence": conf, "n_features": int(n_features), "as_of": today}


def _pick(pred, prefix):
    for k, v in pred.items():
        if k.startswith(prefix):
            return v
    return np.nan


# ---- walk-forward weight learning ----------------------------------------------------------------------------
def _train_blocks(eng, targets, c_row, cols, mu, sd, max_pool=3000):
    """Per training target: pool positions, per-feature difference matrix (float32), pool outcomes, target outcome."""
    Zall = np.clip((eng.Fv[: c_row + 1][:, cols] - mu[cols]) / sd[cols], -ZCLIP, ZCLIP)
    miss = np.isnan(Zall)
    valid = (miss.mean(axis=1) <= MAX_NAN_FRAC) & np.isfinite(eng.Ov[: c_row + 1, eng.ti])
    Z = np.where(miss, 0.0, Zall)
    sq = eng.metric != "manhattan"
    blocks = []
    for i in targets:
        pool = np.flatnonzero(valid[: i - eng.gap + 1])
        if len(pool) > max_pool:                              # bound memory: an evenly thinned pool keeps every era represented
            pool = pool[:: int(np.ceil(len(pool) / max_pool))]
        if len(pool) < 3 * eng.sep:
            continue
        D = Z[pool] - Z[i]
        D = (D ** 2 if sq else np.abs(D)).astype(np.float32)
        blocks.append((pool, D, eng.Ov[pool, eng.ti], float(eng.Ov[i, eng.ti])))
    return blocks


def _loss(blocks, w, eng, k, bw, var_y, lam=0.01):
    """Mean squared analog-forecast error / outcome variance, plus an L1 pull toward uniform weights."""
    ws = w.sum()
    if ws <= 0:
        return np.inf
    err, n = 0.0, 0
    for pool, D, yc, yt in blocks:
        d = D @ w.astype(np.float32) / ws
        d = np.sqrt(d) if eng.metric != "manhattan" else d
        m = min(len(d), max(20 * k, 200))
        part = np.argpartition(d, m - 1)[:m] if m < len(d) else np.arange(len(d))
        order = part[np.argsort(d[part], kind="stable")]
        ch = select_episodes(pool[order], k, eng.sep)
        if len(ch) < min(3, k):
            order = np.argsort(d, kind="stable")
            ch = select_episodes(pool[order], k, eng.sep)
            if not ch:
                continue
        loc = np.searchsorted(pool, ch)
        kw = kernel_weights(d[loc], float(np.median(d)), bw)
        err += (float((kw * yc[loc]).sum()) - yt) ** 2
        n += 1
    if n == 0:
        return np.inf
    return err / n / var_y + lam * float(np.mean(np.abs(w - 1.0)))


def learn_weights_walk_forward(eng, refit_positions, k=5, grid=(0.0, 0.5, 1.0, 2.0, 4.0), sweeps=2, max_train=60,
                               max_val=40, val_frac=0.25, min_gain=0.005, val_margin=0.02, bw=0.5, seed=0, max_pool=3000,
                               method="descent", ic_floor=0.25):
    """Train feature weights at each refit row using only rows/outcomes known by then (8.4). Returns (schedule, sets
    eng.schedule). Per refit: targets are rows <= r - horizon (their outcomes are public at r); the last `val_frac` of
    them are held out (with an embargo of `horizon` before them); weights are learned by coordinate descent over
    `grid` on the earlier block and accepted only if they beat uniform weights on the held-out block by `val_margin`.
    method="ic" replaces the search with one-shot weights proportional to each feature's absolute rank correlation with the
    outcome on the earlier block (floored at ic_floor x mean): cheaper, far less able to overfit, same held-out gate."""
    sched = WeightSchedule()
    diag = []
    for r in refit_positions:
        c_row = r - eng.horizon
        date = eng.F.index[r]
        rec = {"date": date, "accepted": False, "n_train": 0, "n_val": 0, "val_uniform": np.nan, "val_learned": np.nan}
        ones = pd.Series(np.nan, index=eng.cols)
        lo = eng.min_hist + eng.gap
        if c_row - lo < 8 * eng.sep:
            sched.dates.append(date); sched.weights.append(None); diag.append(rec)
            continue
        mu, sd = eng.mu[c_row], eng.sd[c_row]
        cols = np.flatnonzero((eng.cov[c_row] >= 0.5) & np.isfinite(sd) & np.isfinite(mu))
        y = eng.Ov[lo: c_row + 1, eng.ti]
        cand = np.arange(lo, c_row + 1)[np.isfinite(y)]
        if len(cols) < eng.min_features or len(cand) < 8 * eng.sep:
            sched.dates.append(date); sched.weights.append(None); diag.append(rec)
            continue
        rng = np.random.default_rng([seed, r])
        cut = int(len(cand) * (1 - val_frac))
        val_all = cand[cut:]
        train_all = cand[: cut][cand[:cut] <= val_all[0] - eng.horizon]        # embargo between train and validation
        tr = np.sort(rng.choice(train_all, min(max_train, len(train_all)), replace=False))
        va = np.sort(rng.choice(val_all, min(max_val, len(val_all)), replace=False))
        tb = _train_blocks(eng, tr, c_row, cols, mu, sd, max_pool)
        vb = _train_blocks(eng, va, c_row, cols, mu, sd, max_pool)
        yv = eng.Ov[cand, eng.ti]
        var_y = float(np.var(yv)) or 1.0
        if len(tb) < 5 or len(vb) < 5:
            sched.dates.append(date); sched.weights.append(None); diag.append(rec)
            continue
        w = np.ones(len(cols))
        best = _loss(tb, w, eng, k, bw, var_y)
        if method == "ic":
            w = _ic_weights(eng, cols, mu, sd, tr, ic_floor)
        elif method != "descent":
            raise ValueError(f"unknown method {method!r}")
        for _ in range(sweeps if method == "descent" else 0):
            improved = False
            for f in rng.permutation(len(cols)):
                for g in grid:
                    if g == w[f]:
                        continue
                    trial = w.copy(); trial[f] = g
                    L = _loss(tb, trial, eng, k, bw, var_y)
                    if L < best - min_gain * max(best, 1e-9):
                        w, best, improved = trial, L, True
            if not improved:
                break
        lv_uni = _loss(vb, np.ones(len(cols)), eng, k, bw, var_y, lam=0.0)
        lv_new = _loss(vb, w, eng, k, bw, var_y, lam=0.0)
        acc = bool(np.isfinite(lv_new) and lv_new < lv_uni * (1 - val_margin))
        rec.update(accepted=acc, n_train=len(tb), n_val=len(vb), val_uniform=lv_uni, val_learned=lv_new)
        if acc:
            ones = pd.Series(np.nan, index=eng.cols)
            ones.iloc[cols] = w / w.mean() if w.mean() > 0 else 1.0
            sched.dates.append(date); sched.weights.append(ones)
        else:
            sched.dates.append(date); sched.weights.append(None)
        diag.append(rec)
    sched.diagnostics = pd.DataFrame(diag)
    eng.schedule = sched
    return sched


# ---- predictions, ablation and tests -----------------------------------------------------------------------------
def eval_positions(eng, start=None, end=None, step=21):
    """Query rows spaced `step` sessions apart (>= horizon keeps outcomes non-overlapping) inside [start, end]."""
    idx = eng.F.index
    lo = eng.min_hist + eng.gap
    if start is not None:
        lo = max(lo, int(idx.searchsorted(pd.Timestamp(start))))
    hi = len(idx) - 1 - eng.horizon if end is None else int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    hi = min(hi, len(idx) - 1 - eng.horizon)
    return list(range(lo, hi + 1, step))


def predict_series(eng, positions, mode, k=5, seed=0, n_random=1):
    """Forecast of the primary outcome per query row. modes: none | unweighted | weighted | random | shuffled |
    nn_random (nearest neighbours under random feature weights). Actuals come from O and are used only for scoring."""
    rows = []
    for i in positions:
        t = eng.F.index[i]
        rng = np.random.default_rng([seed, i, zlib.crc32(mode.encode())])
        if mode == "none":
            p, res = eng.baseline(i), {"confidence": np.nan, "uniqueness": np.nan}
        else:
            preds, res = [], None
            for rep in range(n_random if mode in ("random", "shuffled", "nn_random") else 1):
                if mode == "unweighted":
                    r = eng.query(i, k, uniform=True, precise_uniqueness=False)
                elif mode == "weighted":
                    r = eng.query(i, k, precise_uniqueness=False)
                elif mode == "random":
                    r = eng.query(i, k, mode="random", rng=rng, precise_uniqueness=False)
                elif mode == "shuffled":
                    r = eng.query(i, k, uniform=True, mode="shuffled", rng=rng, precise_uniqueness=False)
                elif mode == "nn_random":
                    pr = eng._prep(i)
                    if pr is None:
                        r = None
                    else:
                        w = pd.Series(rng.exponential(size=len(pr[0])), index=[eng.cols[j] for j in pr[0]])
                        r = eng.query(i, k, w=w, precise_uniqueness=False)
                else:
                    raise ValueError(f"unknown mode {mode!r}")
                if r is None:
                    break
                preds.append(r["forecast_return"])
                res = res or r
            p = float(np.mean(preds)) if preds else np.nan
            res = res or {"confidence": np.nan, "uniqueness": np.nan}
        rows.append({"date": t, "pred": p, "actual": eng.Ov[i, eng.ti], "confidence": res.get("confidence", np.nan),
                     "uniqueness": res.get("uniqueness", np.nan)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["pred", "actual", "confidence", "uniqueness"])


def hac_t(d, lags=4):
    """t-statistic of mean(d) with a Bartlett (Newey-West) variance, so overlapping/serially-linked losses are not over-counted."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return np.nan
    e = d - d.mean()
    v = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * float(e[L:] @ e[:-L]) / n
    return float(d.mean() / np.sqrt(max(v, 1e-18) / n))


def sign_flip_p(d, seed=0, n_perm=2000):
    """One-sided randomisation p that mean(d) > 0: flip the sign of each paired difference at random (seeded)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d))) @ d / len(d)
    return float((1 + (flips >= obs - 1e-15).sum()) / (n_perm + 1))


def ablation_table(preds, actual=None, ref="none", control="random", seed=0):
    """Score named forecast series on their COMMON days. Columns: mse, mae, corr, hit, skill vs `ref`, mean paired loss gain
    vs ref and vs control, HAC t and sign-flip p (one-sided: better than). Rows keep the input order."""
    names = list(preds)
    common = None
    for n in names:
        ix = preds[n].dropna(subset=["pred"]).index if "pred" in preds[n] else preds[n].dropna().index
        common = ix if common is None else common.intersection(ix)
    if common is None or len(common) < 5:
        return pd.DataFrame(index=names, columns=["n", "mse"])
    P = {n: (preds[n]["pred"] if "pred" in preds[n] else preds[n]).loc[common] for n in names}
    y = (actual if actual is not None else preds[names[0]]["actual"]).loc[common]
    se = {n: (P[n] - y) ** 2 for n in names}
    rows = {}
    for n in names:
        row = {"n": len(common), "mse": float(se[n].mean()), "mae": float((P[n] - y).abs().mean()),
               "corr": float(np.corrcoef(P[n], y)[0, 1]) if P[n].std() > 0 and y.std() > 0 else np.nan,
               "hit": float((np.sign(P[n]) == np.sign(y)).mean())}
        for tag, other in (("ref", ref), ("ctl", control)):
            if other in se and other != n:
                gain = se[other] - se[n]
                row[f"skill_{tag}"] = float(1 - se[n].mean() / se[other].mean())
                row[f"t_{tag}"] = hac_t(gain.values)
                row[f"p_{tag}"] = sign_flip_p(gain.values, seed)
        rows[n] = row
    return pd.DataFrame(rows).T


def compare_methods(eng, positions, k=5, seed=0, n_random=10, modes=("none", "unweighted", "weighted", "random", "shuffled", "nn_random")):
    """Run every mode over the same rows and return (ablation table, predictions by mode)."""
    preds = {m: predict_series(eng, positions, m, k, seed, n_random) for m in modes}
    return ablation_table(preds, seed=seed), preds


def weighting_verdict(table, preds, alpha=0.05, seed=0):
    """Does analog weighting earn its keep? Weighted must beat unweighted analogs (paired), random analogs and 'no analog'."""
    need = {"weighted", "unweighted"}
    if not need <= set(table.index) or not need <= set(preds):
        return {"beats_unweighted": False, "beats_random": False, "beats_none": False, "note": "modes missing"}
    w = table.loc["weighted"]
    wu = weighted_vs_unweighted(preds, seed)
    return {"beats_unweighted": bool(wu["skill"] > 0 and wu["p"] < alpha), "p_unweighted": wu["p"],
            "beats_random": bool(w.get("skill_ctl", np.nan) > 0 and w.get("p_ctl", 1) < alpha),
            "beats_none": bool(w.get("skill_ref", np.nan) > 0 and w.get("p_ref", 1) < alpha),
            "mse_weighted": float(w["mse"]), "n": int(w["n"])}


def weighted_vs_unweighted(preds, seed=0):
    """Paired test that the WEIGHTED analog forecast has lower squared error than the UNWEIGHTED one (same days)."""
    tab = ablation_table({"unweighted": preds["unweighted"], "weighted": preds["weighted"]}, ref="unweighted", control="none", seed=seed)
    return {"skill": float(tab.loc["weighted", "skill_ref"]), "t": float(tab.loc["weighted", "t_ref"]),
            "p": float(tab.loc["weighted", "p_ref"]), "n": int(tab.loc["weighted", "n"])}


# ---- market-level fingerprint (8.1), publication lags applied ------------------------------------------------------
MACRO_LAGS = {"DGS10": 1, "BAMLH0A0HYM2": 1, "T10Y2Y": 1}


def market_context(spx, vix=None, vix3m=None, macro=None, lags=None):
    """Compact market fingerprint aligned to the index's sessions: price state of the index, VIX and its term structure,
    10Y-yield change over a year and credit-spread change over a quarter. Every macro column is forward-filled onto trading
    days and THEN shifted by its publication lag, so a value appears only once it was public (8.2)."""
    lags = {**MACRO_LAGS, **(lags or {})}
    spx = spx.dropna().astype(float)
    F = price_state(spx)[["drawdown", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "accel", "dist_ma200", "vol_1m", "vol_ratio"]]
    if vix is not None:
        v = vix.reindex(spx.index).ffill()
        F["vix"] = v
        if vix3m is not None:
            F["vix_term"] = v / vix3m.reindex(spx.index).ffill()
    if macro is not None:
        m = publication_lag(macro.reindex(macro.index.union(spx.index)).ffill().reindex(spx.index), lags)
        if "DGS10" in m:
            F["rate_chg_1y"] = m["DGS10"] - m["DGS10"].shift(252)
        if "BAMLH0A0HYM2" in m:
            F["credit_chg_3m"] = m["BAMLH0A0HYM2"] - m["BAMLH0A0HYM2"].shift(63)
    return F


# ---- diagnostics: audit, calibration, eras, feature stability, settings, blending -------------------------------------
def audit_result(res, i, index, gap=63, sep=21):
    """Rule violations in one analog result (empty list = clean): analog end < gap sessions before the query, analogs closer
    than sep sessions, duplicate dates, weights not summing to 1, negative distances, inconsistent ages."""
    if res is None:
        return []
    A = res["analogs"]
    pos = index.get_indexer(pd.DatetimeIndex(A["date"]))
    if (pos < 0).any():
        return ["analog date not in index"]
    bad = []
    if (i - pos < gap).any():
        bad.append("analog closer than gap")
    if len(set(pos)) != len(pos):
        bad.append("duplicate analog date")
    if any(abs(a - b) < sep for n, a in enumerate(pos) for b in pos[:n]):
        bad.append("analogs within one episode")
    if not np.isclose(A["weight"].sum(), 1.0):
        bad.append("weights do not sum to 1")
    if (A["distance"] < 0).any():
        bad.append("negative distance")
    if "age_sessions" in A and A["age_sessions"].notna().all() and (A["age_sessions"].values != i - pos).any():
        bad.append("age inconsistent with date")
    return bad


def audit_engine(eng, positions, k=5):
    """Run audit_result over many query rows; returns {(row, kind): [violations]} for the dirty ones (empty = all clean)."""
    out = {}
    for i in positions:
        for tag, r in (("nn", eng.query(i, k)), ("uniform", eng.query(i, k, uniform=True))):
            bad = audit_result(r, i, eng.F.index, eng.gap, eng.sep)
            if bad:
                out[(int(i), tag)] = bad
    return out


def confidence_calibration(df, n_bins=4):
    """Does the reported confidence mean anything? Bin rows by confidence; per bin: mean confidence, mean absolute error,
    direction hit rate. Returns (table, spearman(confidence, -|error|)): positive = more confident is more accurate."""
    d = df.dropna(subset=["pred", "actual", "confidence"])
    if len(d) < 4 * n_bins or d["confidence"].nunique() < 2:
        return pd.DataFrame(columns=["n", "confidence", "mae", "hit"]), np.nan
    from scipy.stats import spearmanr
    err = (d["pred"] - d["actual"]).abs()
    bins = pd.qcut(d["confidence"].rank(method="first"), n_bins, labels=False)
    tab = pd.DataFrame({"n": d.groupby(bins).size(), "confidence": d["confidence"].groupby(bins).mean(),
                        "mae": err.groupby(bins).mean(),
                        "hit": (np.sign(d["pred"]) == np.sign(d["actual"])).groupby(bins).mean()})
    return tab, float(spearmanr(d["confidence"], -err).statistic)


def era_breakdown(preds, eras=None, ref="none"):
    """Per-era skill of every mode vs `ref` on the common days. eras: [(label, start, end)] (default: decades present).
    Long frame (era, mode, n, mse, skill_ref, p_ref); eras with < 8 common days are reported with NaNs, not hidden."""
    common = None
    for df in preds.values():
        ix = df.dropna(subset=["pred", "actual"]).index
        common = ix if common is None else common.intersection(ix)
    cols = ["era", "mode", "n", "mse", "skill_ref", "p_ref"]
    if common is None or len(common) == 0:
        return pd.DataFrame(columns=cols)
    if eras is None:
        dec = sorted({(d.year // 10) * 10 for d in common})
        eras = [(f"{y}s", f"{y}-01-01", f"{y + 9}-12-31") for y in dec]
    rows = []
    for label, a, b in eras:
        keep = common[(common >= pd.Timestamp(a)) & (common <= pd.Timestamp(b))]
        sub = {m: df.loc[keep] for m, df in preds.items()}
        if len(keep) < 8:
            rows += [{"era": label, "mode": m, "n": len(keep), "mse": np.nan, "skill_ref": np.nan, "p_ref": np.nan} for m in preds]
            continue
        tab = ablation_table(sub, ref=ref, control=ref)
        rows += [{"era": label, "mode": m, "n": len(keep), "mse": tab.loc[m, "mse"], "skill_ref": tab.loc[m].get("skill_ref", np.nan),
                  "p_ref": tab.loc[m].get("p_ref", np.nan)} for m in preds]
    return pd.DataFrame(rows, columns=cols)


def feature_report(sched, cols):
    """Which features the walk-forward weights lean on: mean weight over accepted refits, share of refits that gave the
    feature more than uniform weight, and how stable the ordering is between consecutive accepted refits (Spearman)."""
    ws = [w.reindex(cols) for w in sched.weights if w is not None]
    if not ws:
        return pd.DataFrame({"mean_weight": np.nan, "share_above_uniform": np.nan, "stability": np.nan}, index=cols)
    M = pd.concat(ws, axis=1)
    stab = np.nan
    if M.shape[1] >= 2:
        from scipy.stats import spearmanr
        rs = [spearmanr(M.iloc[:, j].fillna(1), M.iloc[:, j + 1].fillna(1)).statistic for j in range(M.shape[1] - 1)]
        stab = float(np.nanmean(rs))
    return pd.DataFrame({"mean_weight": M.mean(axis=1), "share_above_uniform": (M.fillna(1) > 1.0).mean(axis=1),
                         "stability": stab}).sort_values("mean_weight", ascending=False)


def sweep_settings(eng, positions, ks=(3, 5, 10), metrics=METRICS, seed=0):
    """Uniform-weight analog error by (metric, k) on the same rows against the no-analog baseline: how sensitive is the
    result to the two settings that cannot be learned? Rows where any setting cannot answer drop out for all."""
    preds = {}
    for m in metrics:
        e = KNNEngine(eng.F, eng.O, eng.target, eng.horizon, eng.gap, eng.sep, m, eng.min_hist, eng.min_features)
        for k in ks:
            preds[f"{m}_k{k}"] = predict_series(e, positions, "unweighted", k, seed)
    preds["none"] = predict_series(eng, positions, "none")
    tab = ablation_table(preds, ref="none", control="none")
    return tab.drop(index="none")[["n", "mse", "corr", "hit", "skill_ref", "p_ref"]]


def blend_forecasts(levels, min_conf=0.0):
    """Confidence-weighted average of level forecasts (market / sector / stock) per date. Levels below min_conf or NaN drop out
    of that date; a date with no surviving level stays NaN. Returns pred, confidence (mean of survivors), n_levels."""
    P = pd.concat({n: df["pred"] for n, df in levels.items()}, axis=1)
    C = pd.concat({n: df["confidence"] for n, df in levels.items()}, axis=1)
    C = C.where(P.notna() & (C >= min_conf) & (C > 0))
    wsum = C.sum(axis=1)
    pred = (P.where(C.notna()) * C).sum(axis=1).div(wsum.where(wsum > 0))
    out = pd.DataFrame({"pred": pred, "confidence": C.mean(axis=1), "n_levels": C.notna().sum(axis=1)})
    if all("actual" in df for df in levels.values()):
        out["actual"] = next(iter(levels.values()))["actual"]
    return out


def levels_table(preds, seed=0):
    """8.8 in one call: rows none / market / sector / stock / shuffled / nn_random (whichever are supplied), scored on the
    common days against 'none', with the nearest-neighbour random-weight control as the bar an analog level must clear."""
    return ablation_table(preds, ref="none", control="nn_random" if "nn_random" in preds else "random", seed=seed)


def render_report(title, tables, notes=()):
    """Markdown report from {heading: DataFrame | str}. Numbers only come from the frames handed in."""
    out = [f"# {title}", ""]
    out += [f"- {n}" for n in notes]
    for head, obj in tables.items():
        out += ["", f"## {head}", ""]
        if isinstance(obj, str):
            out.append(obj)
        elif len(obj):
            out.append(obj.round(4).to_string())
        else:
            out.append("(empty)")
    return "\n".join(out) + "\n"


def cross_sectional_ic(P, Y, min_n=5, seed=0):
    """Rank information coefficient of forecasts against realised outcomes ACROSS names on each date (dates x names frames).
    Does ranking sectors/stocks by their analog forecast order what happened next? Returns (IC per date, summary with mean IC,
    HAC t, sign-flip p that mean IC > 0, share of dates with IC > 0, number of dates)."""
    from scipy.stats import spearmanr
    ic = {}
    for t in P.index.intersection(Y.index):
        a, b = P.loc[t], Y.loc[t]
        ok = a.notna() & b.notna()
        if ok.sum() >= min_n and a[ok].nunique() > 1 and b[ok].nunique() > 1:
            ic[t] = float(spearmanr(a[ok], b[ok]).statistic)
    ic = pd.Series(ic, dtype=float)
    if len(ic) < 5:
        return ic, {"mean_ic": np.nan, "t": np.nan, "p": np.nan, "share_pos": np.nan, "n_dates": int(len(ic))}
    return ic, {"mean_ic": float(ic.mean()), "t": hac_t(ic.values), "p": sign_flip_p(ic.values, seed),
                "share_pos": float((ic > 0).mean()), "n_dates": int(len(ic))}


# ---- IC-based weights, learner null, learner comparison --------------------------------------------------------------------
def _ic_weights(eng, cols, mu, sd, train_targets, floor=0.25):
    """One-shot weights: |Spearman(feature z, outcome)| over the rows up to the last training target, scaled to mean 1 and
    floored at `floor` so no feature is silenced by a noisy estimate. Only rows whose outcome was public at refit are used."""
    from scipy.stats import rankdata
    hi = int(np.max(train_targets))
    Z = np.clip((eng.Fv[: hi + 1][:, cols] - mu[cols]) / sd[cols], -ZCLIP, ZCLIP)
    y = eng.Ov[: hi + 1, eng.ti]
    ok = np.isfinite(y) & (np.isnan(Z).mean(axis=1) <= MAX_NAN_FRAC)
    if ok.sum() < 30:
        return np.ones(len(cols))
    Zr = rankdata(np.where(np.isnan(Z[ok]), 0.0, Z[ok]), axis=0)
    yr = rankdata(y[ok])
    Zr, yr = Zr - Zr.mean(axis=0), yr - yr.mean()
    den = np.sqrt((Zr ** 2).sum(axis=0) * (yr ** 2).sum())
    ic = np.abs((Zr * yr[:, None]).sum(axis=0) / np.where(den > 0, den, np.nan))
    ic = np.nan_to_num(ic, nan=0.0)
    if ic.mean() <= 0:
        return np.ones(len(cols))
    return np.maximum(ic / ic.mean(), floor)


def block_shuffle(values, block, rng):
    """Permute a series in contiguous blocks (keeps within-block autocorrelation, destroys any tie to the fingerprints)."""
    n = len(values)
    blocks = [np.arange(s, min(s + block, n)) for s in range(0, n, block)]
    perm = np.concatenate([blocks[b] for b in rng.permutation(len(blocks))])
    return np.asarray(values)[perm]


def weight_learner_null(eng, refit_positions, n_shuffles=5, seed=0, method="descent", block=63, **kw):
    """False-acceptance rate of the weight learner: re-run it on copies whose OUTCOMES are block-shuffled (no fingerprint ->
    outcome relation exists) and count how often the held-out gate still accepts learned weights. Compare with the real rate:
    a learner that accepts as often on shuffled outcomes as on real ones is fitting noise."""
    def rate(e):
        d = learn_weights_walk_forward(e, refit_positions, seed=seed, method=method, **kw).diagnostics
        ok = d["n_train"] > 0
        return float(d.loc[ok, "accepted"].mean()) if ok.any() else np.nan
    real = rate(eng.clone())
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffles):
        O2 = eng.O.copy()
        O2[:] = np.column_stack([block_shuffle(O2[c].values, block, rng) for c in O2.columns])
        nulls.append(rate(eng.clone(O=O2)))
    nulls = np.array(nulls, float)
    return {"accept_real": real, "accept_null_mean": float(np.nanmean(nulls)) if np.isfinite(nulls).any() else np.nan,
            "accept_null_max": float(np.nanmax(nulls)) if np.isfinite(nulls).any() else np.nan, "n_shuffles": n_shuffles}


def compare_learners(eng, refit_positions, positions, k=5, seed=0, n_random=5, methods=("descent", "ic")):
    """Score uniform weights, each weight learner and the no-analog baseline on the same rows (one engine clone per learner,
    each with its own walk-forward schedule). Returns (table, preds)."""
    preds = {"none": predict_series(eng, positions, "none"), "uniform": predict_series(eng, positions, "unweighted", k, seed)}
    for m in methods:
        e = eng.clone()
        learn_weights_walk_forward(e, refit_positions, k=k, seed=seed, method=m)
        preds[f"learned_{m}"] = predict_series(e, positions, "weighted", k, seed)
    return ablation_table(preds, ref="none", control="uniform", seed=seed), preds


# ---- 8.1 fingerprint coverage and causality audit ------------------------------------------------------------------------
FINGERPRINT_SPEC = {
    "market drawdown": ["drawdown"], "1m return": ["ret_1m"], "3m return": ["ret_3m"], "6m return": ["ret_6m"],
    "12m return": ["ret_12m"], "acceleration": ["accel"], "distance from MA200": ["dist_ma200"], "volatility": ["vol_1m"],
    "volatility ratio": ["vol_ratio"], "VIX": ["vix"], "VIX term structure": ["vix_term"], "breadth": ["breadth"],
    "dispersion": ["dispersion"], "sector crowding": ["sector_crowding"],
    "top-sector concentration change": ["top_sector_share_chg"], "FRED data": ["mac_*"],
    "10Y yield change": ["rate_chg_1y"], "credit-spread change": ["credit_chg_3m"]}


def spec_coverage(F, spec=None):
    """One row per Bible 8.1 item: which fingerprint column(s) supply it, non-NaN share, first/last valid date. Missing items
    are listed (present=False), never silently dropped. A trailing * matches a column prefix."""
    rows = []
    for item, pats in (spec or FINGERPRINT_SPEC).items():
        cols = [c for p in pats for c in F.columns if (c.startswith(p[:-1]) if p.endswith("*") else c == p)]
        if not cols:
            rows.append({"item": item, "columns": "", "present": False, "non_nan": 0.0, "first": pd.NaT, "last": pd.NaT})
            continue
        sub = F[cols]
        any_ok = sub.notna().any(axis=1)
        rows.append({"item": item, "columns": ",".join(cols), "present": True, "non_nan": float(sub.notna().mean().mean()),
                     "first": sub.index[any_ok][0] if any_ok.any() else pd.NaT, "last": sub.index[any_ok][-1] if any_ok.any() else pd.NaT})
    return pd.DataFrame(rows).set_index("item")


def causality_audit(build, data, cut_points, slicer, atol=1e-9):
    """Is a fingerprint builder causal? Build on the full input, then on the input truncated at each cut point, and require the
    two frames to agree on every row up to the cut. Returns a list of (cut, column, max abs difference) for offenders, so a
    builder that peeks at the future (centered windows, whole-history normalisation, look-ahead joins) is named, not guessed at."""
    full = build(data)
    bad = []
    for c in cut_points:
        part = build(slicer(data, c))
        ref = full.loc[part.index]
        for col in part.columns.intersection(ref.columns):
            a, b = part[col].values.astype(float), ref[col].values.astype(float)
            both = np.isfinite(a) & np.isfinite(b)
            miss = np.isfinite(a) != np.isfinite(b)
            diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            if miss.any() or diff > atol:
                bad.append((c, col, float("inf") if miss.any() else diff))
    return bad


# ---- cross-level agreement and result ledger ---------------------------------------------------------------------------
def level_agreement(levels, eps=0.0):
    """How often do analog levels (market / sector / stock, same horizon) point the same way, and does agreement pay?
    Returns sign-agreement and correlation matrices, the share of days with a conflict, and hit rate of the mean forecast on
    agree vs conflict days (a level stack is only worth building if agreement is more accurate than conflict)."""
    P = pd.concat({n: df["pred"] for n, df in levels.items()}, axis=1).dropna()
    y = next(iter(levels.values())).loc[P.index, "actual"]
    sg = np.sign(P.where(P.abs() > eps, 0.0))
    names = list(P.columns)
    agree = pd.DataFrame({a: {b: float((sg[a] == sg[b]).mean()) for b in names} for a in names})
    if len(P) == 0:
        return {"sign_agreement": agree, "corr": P.corr(), "conflict_share": np.nan, "hit_agree": np.nan, "hit_conflict": np.nan, "n": 0}
    conflict = (sg.max(axis=1) > 0) & (sg.min(axis=1) < 0)
    mean = P.mean(axis=1)
    hit = np.sign(mean) == np.sign(y)
    return {"sign_agreement": agree, "corr": P.corr(), "conflict_share": float(conflict.mean()),
            "hit_agree": float(hit[~conflict].mean()) if (~conflict).any() else np.nan,
            "hit_conflict": float(hit[conflict].mean()) if conflict.any() else np.nan, "n": int(len(P))}


def results_to_frame(results, tag, key=None):
    """Flatten analog results (dicts from query) into one row per (query, analog): query date, level tag, forecast fields,
    analog date, distance, age, weight and outcomes. The 8.7 outputs in a shape that can be stored and audited later."""
    rows = []
    for j, r in enumerate(results):
        if r is None:
            continue
        head = {"tag": tag, "key": key[j] if key is not None else "", "as_of": r["as_of"], "forecast_return": r["forecast_return"],
                "forecast_vol": r["forecast_vol"], "forecast_drawdown": r["forecast_drawdown"], "prob_up": r.get("prob_up", np.nan),
                "uniqueness": r["uniqueness"], "confidence": r["confidence"], "n_analogs": r["n_analogs"]}
        for rank, (_, a) in enumerate(r["analogs"].iterrows()):
            rows.append({**head, "rank": rank, "analog_date": a["date"], "distance": a["distance"], "age_years": a["age_years"],
                         "weight": a["weight"], **{f"out_{c}": a[c] for c in r["analogs"].columns if c.startswith("fwd_")}})
    return pd.DataFrame(rows)


def append_ledger(path, frame):
    """Append to a parquet ledger keyed by (tag, key, as_of, rank); a re-run replaces its own rows instead of duplicating them."""
    from pathlib import Path
    path = Path(path)
    if frame.empty:
        return 0
    if path.exists():
        old = pd.read_parquet(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame = frame.drop_duplicates(["tag", "key", "as_of", "rank"], keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return len(frame)
