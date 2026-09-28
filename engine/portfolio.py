"""Portfolio layer (Blueprint Part D + E).

simulate()  joint weekly scenarios for candidates: market + sector factors (regime covariance,
            Ledoit-Wolf-style shrinkage), Student-t idiosyncratic, earnings jumps.
optimise()  choose weights maximising P(week >= target) subject to CVaR and caps,
            by greedy build-up + swap search, exact counting over scenarios.
control()   within-week exposure policy (bank / brake / capped catch-up)."""
import numpy as np
import pandas as pd

from . import config as K

rng_global = np.random.default_rng(7)


def simulate(mu, sigma, beta, sector, fac_ret_hist, earn_jump, n=K.N_SCENARIOS, horizon=5, df=4, rng=None):
    """mu, sigma: per-stock expected horizon log return and daily vol (arrays, len m).
    beta: market beta; sector: integer codes; fac_ret_hist: DataFrame of recent daily
    factor returns (col 'mkt' + one col per sector code) used for the factor covariance.
    earn_jump: per-stock std of an earnings-day jump (0 if no earnings in window)."""
    rng = rng or rng_global
    m = len(mu)
    F = fac_ret_hist.replace([np.inf, -np.inf], np.nan)
    F = F.dropna(axis=1, thresh=int(0.8 * len(F))).fillna(0.0).clip(-0.3, 0.3)
    if "mkt" not in F or len(F) < 30:          # not enough history: market-only with a typical vol
        F = pd.DataFrame({"mkt": np.random.default_rng(0).normal(0, 0.011, 250)})
    cov = np.cov(F.values.T, ddof=1).reshape(len(F.columns), len(F.columns)) * horizon
    cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))            # shrink off-diagonals
    cov = np.nan_to_num(cov)
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)              # PSD repair instead of SVD failure
    root = vecs * np.sqrt(np.clip(vals, 0, None))
    fac = rng.standard_normal((n, len(cov))) @ root.T
    mkt = fac[:, 0]
    cols = list(F.columns)
    sec_idx = np.array([cols.index(s) if s in cols else 0 for s in sector])
    sec_move = np.where(sec_idx > 0, fac[:, sec_idx], 0.0)
    # idiosyncratic: what the factors don't explain of each stock's own variance
    tot_var = (sigma ** 2) * horizon
    sys_var = beta ** 2 * cov[0, 0] + np.where(sec_idx > 0, cov[sec_idx, sec_idx], 0)
    idio_sd = np.sqrt(np.clip(tot_var - sys_var, 0.25 * tot_var, None))
    t = rng.standard_t(df, size=(n, m)) * np.sqrt((df - 2) / df)
    jump = rng.standard_t(3, size=(n, m)) * np.sqrt(1 / 3) * earn_jump
    logret = mu + beta * mkt[:, None] + sec_move + idio_sd * t + jump
    return np.expm1(logret)                                    # simple returns, (n, m)


def _stats(port, target):
    p = (port >= target).mean()
    q = np.quantile(port, 0.05)
    cvar = port[port <= q].mean()
    elog = np.log1p(np.clip(port, -0.95, None)).mean()
    return p, cvar, elog


def optimise(S, names, sectors, mu, earn_flag, need=K.WEEKLY_TARGET, gross=1.0,
             max_names=K.MAX_NAMES, min_names=K.MIN_NAMES, squeeze=None, objective="goal"):
    """S: scenarios (n, m). Returns (weights Series, diagnostics). Weights sum to <= gross.
    Candidates with non-positive expected return are never bought (no volatility without edge)."""
    m = S.shape[1]
    ok = np.where(mu > 0)[0]
    if len(ok) == 0:
        return pd.Series(dtype=float), {"p_target": 0.0, "cvar": 0.0, "elog": 0.0, "n": 0}
    step = 0.05
    w = np.zeros(m)

    def feasible(w):
        if w.max() > K.MAX_WEIGHT + 1e-9 or w.sum() > gross + 1e-9:
            return False
        for s in set(sectors):
            if w[np.array(sectors) == s].sum() > K.MAX_SECTOR + 1e-9:
                return False
        if (w[earn_flag] > 0).sum() > K.MAX_EARNINGS_HOLDS:
            return False
        if squeeze is not None and w[squeeze].sum() > K.MAX_SQUEEZE + 1e-9:
            return False
        return (w > 0).sum() <= max_names

    def score(w):
        p, cvar, elog = _stats(S @ w, need)
        pen = 0 if cvar >= K.CVAR_LIMIT else (K.CVAR_LIMIT - cvar) * 20
        if objective == "growth":          # log-utility (Kelly) under the same tail limit
            return elog * 100 - pen, (p, cvar, elog)
        pen += 0 if elog > 0 else -elog * 50
        return p - pen, (p, cvar, elog)

    best, info = score(w)
    # greedy: add 5% slices to whichever name raises the objective most
    while w.sum() < gross - 1e-9:
        cand = []
        for j in ok:
            w2 = w.copy(); w2[j] += step
            if feasible(w2):
                cand.append((score(w2)[0], j))
        if not cand:
            break
        val, j = max(cand)
        if val < best - 1e-4 and (w > 0).sum() >= min_names:
            break
        w[j] += step; best, info = score(w)
    # swap search: move a slice from one holding to another
    improved = True
    while improved:
        improved = False
        held = np.where(w > 0)[0]
        for a in held:
            for b in ok:
                if a == b:
                    continue
                w2 = w.copy(); w2[a] -= step; w2[b] += step
                if w2[a] < -1e-9 or not feasible(w2):
                    continue
                v, i2 = score(w2)
                if v > best + 1e-4:
                    w, best, info, improved = w2, v, i2, True
                    break
            if improved:
                break
    ws = pd.Series(w, index=names)
    ws = ws[ws > 1e-9]
    return ws, {"p_target": float(info[0]), "cvar": float(info[1]), "elog": float(info[2]), "n": int(len(ws))}


def control(week_ret, days_left, edge_positive, regime):
    """Part E within-week exposure multiplier."""
    base = K.REGIME_GROSS.get(regime, 0.8)
    if week_ret >= K.BANK_LEVEL:
        return min(base, K.BANK_EXPOSURE), "bank"
    if week_ret <= K.BRAKE_LEVEL:
        return base * K.BRAKE_EXPOSURE, "brake"
    behind = K.WEEKLY_TARGET - week_ret
    if behind > 0.03 and days_left <= 3:
        if edge_positive:
            return min(1.0, base * K.MAX_CATCHUP), "catchup"
        return base * 0.6, "no-edge-cut"
    return base, "normal"
