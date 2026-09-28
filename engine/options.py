"""Live options features for the top candidates (Blueprint B2). Free, 15-minute delayed.

vol_spread   Cremers-Weinbaum: mean(call IV - put IV) at matched strikes near the money (+ bullish)
put_skew     Xing-Zhang-Zhao: OTM put IV - ATM call IV (+ = bearish)
cp_volume    log call/put volume ratio (Pan-Poteshman direction proxy)
iv_atm       at-the-money IV (annualised) -> weekly sigma for the scenario simulator"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime


def _clean(df):
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["impliedVolatility"].between(0.05, 5))]
    return df[(df["openInterest"].fillna(0) >= 10) | (df["volume"].fillna(0) >= 10)]


def features_for(ticker: str, spot: float):
    try:
        tk = yf.Ticker(ticker)
        exps = [e for e in tk.options if (datetime.strptime(e, "%Y-%m-%d").date() - date.today()).days >= 5]
        if not exps:
            return None
        ch = tk.option_chain(exps[0])
        c, p = _clean(ch.calls), _clean(ch.puts)
        if len(c) < 3 or len(p) < 3:
            return None
        both = c.merge(p, on="strike", suffixes=("_c", "_p"))
        near = both[(both["strike"] / spot).between(0.9, 1.1)]
        if near.empty:
            return None
        w = near["openInterest_c"].fillna(0) + near["openInterest_p"].fillna(0) + 1
        vol_spread = float(np.average(near["impliedVolatility_c"] - near["impliedVolatility_p"], weights=w))
        atm_c = c.iloc[(c["strike"] - spot).abs().argsort()[:1]]["impliedVolatility"].iloc[0]
        otm_p = p[(p["strike"] / spot).between(0.85, 0.97)]
        put_skew = float(otm_p["impliedVolatility"].mean() - atm_c) if len(otm_p) else np.nan
        cv, pv = ch.calls["volume"].fillna(0).sum(), ch.puts["volume"].fillna(0).sum()
        return {"ticker": ticker, "vol_spread": vol_spread, "put_skew": put_skew,
                "cp_volume": float(np.log((cv + 1) / (pv + 1))), "iv_atm": float(atm_c), "expiry": exps[0]}
    except Exception:
        return None


def snapshot(tickers, spots) -> pd.DataFrame:
    rows = [features_for(t, spots[t]) for t in tickers if t in spots]
    return pd.DataFrame([r for r in rows if r]).set_index("ticker") if any(rows) else pd.DataFrame()


def evidence_adjustment(o: pd.DataFrame) -> pd.Series:
    """Rank-based tilt added to the live score: bullish vol spread and call volume up,
    steep put skew down. Weights are the Part B priors; Part M re-weights them from live IC."""
    if o.empty:
        return pd.Series(dtype=float)
    r = lambda s: s.rank(pct=True) - 0.5
    return (0.6 * r(o["vol_spread"]) + 0.3 * r(o["cp_volume"]) - 0.5 * r(o["put_skew"].fillna(o["put_skew"].median()))).fillna(0)
