"""The trading policy, shared by the backtest and live trading so that what is tested is what trades.

score()     blended ranking: expected-return model + evidence prior (+ live options tilt)
plan()      candidates -> scenarios -> optimiser -> target weights, for one decision
PARAMS      the champion's settings; promoted challengers change these via state/models/meta.json"""
import numpy as np
import pandas as pd

from . import config as K
from .portfolio import simulate, optimise

PARAMS = {
    "w_model": 0.5,          # weight of the expected-return model vs the evidence prior
    "objective": "goal",     # "goal" = max P(week >= +7%), "growth" = max E[log wealth]
    "vol_filter": True,      # drop top-decile volatility / MAX names (Ang 2006, Bali 2011)
    "min_dv": 2e7,           # liquidity floor, $/day
    "keep_pct": 0.90,        # hysteresis: holdings stay eligible while in the top decile
    "mu_shrink": 0.5,
    "stop_atr": K.STOP_ATR,
    "rebalance": "weekly",
}


def score(P: pd.DataFrame, w_model=None) -> pd.Series:
    """P: one day's (or a panel's) predictions with mu_raw and evidence columns."""
    w = PARAMS["w_model"] if w_model is None else w_model
    if isinstance(P.index, pd.MultiIndex):
        mr = P.groupby(level=0)["mu_raw"].rank(pct=True)
    else:
        mr = P["mu_raw"].rank(pct=True)
    return (w * mr + (1 - w) * P["evidence"]).rename("score")


def eligible(xr: pd.DataFrame, params=None) -> pd.Series:
    p = {**PARAMS, **(params or {})}
    ok = ~((xr["ev_red_flag"] > 0) | ((xr["ev_offering"] > 0) & (xr["log_dv"] < np.log1p(5e7))))
    if p["vol_filter"]:
        ok &= ~((xr["vol20"].rank(pct=True) > 0.9) | (xr["max20"].rank(pct=True) > 0.9))
    if p["min_dv"]:
        ok &= xr["log_dv"] >= np.log1p(p["min_dv"])
    return ok


def plan(xr, pr, s, held, gross, need, horizon, beta, sec_map, fac_hist, rng=None, n_scen=K.N_SCENARIOS,
         params=None, sig_override=None):
    """One decision. xr: features today; pr: predictions today; s: score today (Series);
    held: iterable of current tickers; beta: Series by ticker; sec_map: Series ticker->sector;
    sig_override: optional Series of daily vol (e.g. from options IV) by ticker."""
    p = {**PARAMS, **(params or {})}
    ok = eligible(xr, p).reindex(s.index).fillna(False)
    s_ok = s[ok]
    cand = list(s_ok.sort_values(ascending=False).head(K.N_CANDIDATES).index)
    q = s_ok.rank(pct=True)
    cand += [t for t in held if t in q.index and q[t] >= p["keep_pct"] and t not in cand]
    cand = pd.Index(cand)
    if len(cand) == 0:
        return pd.Series(dtype=float), {"p_target": 0.0, "cvar": 0.0, "n": 0}
    mu = p["mu_shrink"] * pr.loc[cand, "mu_raw"].values * horizon / 5
    mu = mu + np.array([t in set(held) for t in cand]) * 2 * K.COST_BPS_ILLIQUID / 1e4
    sig = xr.loc[cand, "vol20"].fillna(0.03).values
    if sig_override is not None:
        iv = sig_override.reindex(cand)
        sig = np.where(iv.notna(), 0.5 * sig + 0.5 * iv.values, sig)
    b = beta.reindex(cand).fillna(1.0).values
    secs = sec_map.reindex(cand).fillna("9").values
    earn = xr.loc[cand, "earn_in_week"].fillna(0).values > 0
    ej = np.where(earn, 1.5 * sig * np.sqrt(5), 0.0)
    S = simulate(mu, sig, b, secs, fac_hist, ej, n=n_scen, horizon=horizon, rng=rng)
    return optimise(S, list(cand), list(secs), mu, earn, need=max(need, 0.005), gross=gross,
                    objective=p["objective"])


# ---------------- champion constructor (v1.1): concentrated top-k with hysteresis ----------------
# Champion v1.1 (registry: topk_rules, 28 Sep 2026): top-4 equal weight, keep while in top 20%, weekly,
# -8% weekly brake to 1/3 exposure; no bank rule (cost ~$850 in backtest); no per-stock stops (destroyed value).
TOPK = {"k": 4, "exit_q": 0.80, "bank": None, "brake": 0.08, "bank_exposure": 0.4, "brake_exposure": 1 / 3}

NYSE_HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
                 "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}


def last_session_of_week(day) -> bool:
    """True when no later trading session exists in the same ISO week (Friday, or Thursday before a holiday)."""
    d = pd.Timestamp(day).normalize()
    for k in range(1, 7 - d.weekday()):
        n = d + pd.Timedelta(days=k)
        if n.weekday() < 5 and n.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS:
            return False
    return True


def topk_targets(s_ok: pd.Series, held, k=None, exit_q=None) -> pd.Series:
    """Equal-weight k names: keep holdings still in the top (1-exit_q) of eligible names, fill the
    rest from the top of the ranking. Shared by backtest and live (what is tested is what trades)."""
    k = k or TOPK["k"]
    exit_q = TOPK["exit_q"] if exit_q is None else exit_q
    q = s_ok.rank(pct=True)
    keep = [t for t in held if t in q.index and q[t] >= exit_q][:k]
    fill = [t for t in s_ok.sort_values(ascending=False).index if t not in keep][: k - len(keep)]
    names = keep + fill
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)
