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


def crypto_set() -> set:
    """Real tickers that are crypto by canon C11 (hard list plus crypto-named companies in the universe file)."""
    from .universe import CRYPTO_TICKERS, CRYPTO_NAME
    global _CRYPTO
    if "_CRYPTO" not in globals():
        u = pd.read_csv(K.CACHE / "universe.csv")
        _CRYPTO = set(CRYPTO_TICKERS) | set(u.loc[u["name"].str.contains(CRYPTO_NAME, na=False), "ticker"])
    return _CRYPTO


def not_crypto(index) -> pd.Series:
    """Canon C11: crypto is off limits, everywhere a stock can be picked. (Blind runs see code names, so this is inert
    there; the blind Feed removes crypto tickers from the universe BEFORE disguising - see livesim.Feed.)"""
    c = crypto_set()
    return pd.Series([t not in c for t in index], index=index)


def eligible(xr: pd.DataFrame, params=None) -> pd.Series:
    p = {**PARAMS, **(params or {})}
    ok = not_crypto(xr.index) & ~((xr["ev_red_flag"] > 0) | ((xr["ev_offering"] > 0) & (xr["log_dv"] < np.log1p(5e7))))
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
# v1.2 restored (28 Sep 2026). v1.3 (tuning round 1) was rolled back before it traded: over 2017-2026 it made
# $3,579 vs S&P $3,987 with a -73% drawdown; its promotion guards only looked at single months. See registry.
TOPK = {"k": 4, "exit_q": 0.80, "bank": None, "brake": 0.08, "bank_exposure": 0.4, "brake_exposure": 1 / 3,
        "max_per_sector": 2, "pick": "top", "pool_q": 0.95}

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


def sic_division(sic) -> str:
    """SIC division letter (A-J): the 'sector' used for the blueprint's 40% sector cap."""
    try:
        c = int(str(sic)[:2])
    except ValueError:
        return "?"
    for hi, div in ((9, "A"), (14, "B"), (17, "C"), (39, "D"), (49, "E"), (51, "F"), (59, "G"), (67, "H"), (89, "I")):
        if c <= hi:
            return div
    return "J"


def _ranked(primary: pd.Series, tiebreak=None):
    """Names by primary score, descending. With `tiebreak` (content, e.g. volatility) exact ties are ordered by it, so
    the choice does not depend on what the tickers are CALLED (a disguised rerun renamed tickers and moved a year's
    return from +5% to +23% through name-order tie-breaks; learning-delta probe, 2026-09-29). Without it: the old
    behaviour (used by Live, which changes only through a promoted challenger)."""
    if tiebreak is None:
        return list(primary.sort_values(ascending=False).index)
    tb = tiebreak.reindex(primary.index).astype(float).fillna(-np.inf).values
    order = np.lexsort((-tb, -primary.astype(float).values))
    return list(primary.index[order])


def topk_targets(s_ok: pd.Series, held, k=None, exit_q=None, sectors=None, max_per_sector=None,
                 pick="top", vol=None, pool_q=0.95, tiebreak=None) -> pd.Series:
    """Equal-weight k names: keep holdings still in the top (1-exit_q) of eligible names, fill the
    rest from the top of the ranking. Shared by backtest and live (what is tested is what trades)."""
    k = k or TOPK["k"]
    exit_q = TOPK["exit_q"] if exit_q is None else exit_q
    q = s_ok.rank(pct=True)
    keep = [t for t in held if t in q.index and q[t] >= exit_q][:k]
    if pick == "hivol" and vol is not None:
        # among the top (1-pool_q) by score, rank by volatility: more +7% weeks, still only names with an edge
        pool = q[q >= pool_q].index
        if tiebreak is not None:
            tiebreak = s_ok.reindex(pool)                      # hivol: ties in volatility broken by the score
        s_ok = vol.reindex(pool).fillna(0).rename(None)
    if sectors is None or not max_per_sector:
        fill = [t for t in _ranked(s_ok, tiebreak) if t not in keep][: k - len(keep)]
    else:
        count, fill = {}, []
        for t in keep:
            count[sectors.get(t, "?")] = count.get(sectors.get(t, "?"), 0) + 1
        for t in _ranked(s_ok, tiebreak):
            if len(keep) + len(fill) >= k:
                break
            sec = sectors.get(t, "?")
            if t in keep or count.get(sec, 0) >= max_per_sector:
                continue
            fill.append(t); count[sec] = count.get(sec, 0) + 1
    names = keep + fill
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def regime_targets(s_ok, held, cfg, vol, divs, mkt):
    """Regime-aware selection shared by the blind trader and the re-tester (never diverge).
    mkt: dict of market-wide readings for today (m_vix_term, m_spy_ma200, ...).
    - stress-rebound mode: short-term fear above long-term fear -> concentrated high-volatility picks;
    - downtrend cash filter: market below its 200-day average (and no rebound signal) -> invest only part."""
    k, pick = cfg["k"], cfg["pick"]
    thr = cfg.get("stress_thr")
    stress = thr is not None and mkt.get("m_vix_term") is not None and mkt["m_vix_term"] == mkt["m_vix_term"] \
        and mkt["m_vix_term"] > thr
    if stress:
        k, pick = cfg.get("stress_k", 3), "hivol"
    t = topk_targets(s_ok, held, k, cfg["exit_q"], divs if cfg["max_per_sector"] else None, cfg["max_per_sector"],
                     pick=pick, vol=vol, pool_q=cfg["pool_q"], tiebreak=vol)
    tf = cfg.get("trend_filter")
    ma = mkt.get("m_spy_ma200")
    if not stress and tf is not None and ma is not None and ma == ma and ma < tf:
        t = t * cfg.get("trend_gross", 0.5)
    return t


EVIDENCE_FEATS = ["ear", "ins_buyers30", "ins_officer30", "dist_52wh", "ind_mom60", "frog", "mom_12_1",
                  "r5_nonews", "max20", "ev_offering", "ev_shelf", "ev_red_flag", "skew60"]


def default_evidence_weights():
    from .model import EVIDENCE
    return {k: v for k, v in EVIDENCE.items() if k != "ev_activist"}


def evidence_from(p, ew=None):
    """Evidence score for one decision day from its per-indicator ranks (columns e_<feature>).
    Identical to model.evidence_score on that day's ranks; shared by the trader and the re-tester."""
    if ew is None:
        return p["evidence"]
    s = sum(w * p[f"e_{c}"].fillna(0) for c, w in ew.items() if f"e_{c}" in p)
    return s.rank(pct=True) if hasattr(s, "rank") else p["evidence"]
