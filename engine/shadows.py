"""Shadow baseline portfolios (Blueprint Part H2), each with its own $1,000:
spy, random_vol (8 random high-vol names), trend_chaser (top 8 by 20d return + volume surge),
evidence_only (top 8 by the literature composite, no ML). Rebalanced at each Friday close
from that day's prediction log; marked to market every close."""
import json
import numpy as np
import pandas as pd

from . import config as K

DIR = K.STATE / "shadows"
DIR.mkdir(parents=True, exist_ok=True)
NAMES = ["spy", "random_vol", "trend_chaser", "evidence_only"]


def _load(n):
    p = DIR / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else {"cash": K.START_CASH, "hold": {}, "equity": []}


def _pick(name, P, rng):
    if name == "spy":
        return ["SPY"]
    if name == "random_vol":
        pool = P[P["x_vol20"].rank(pct=True) > 0.7].index
        return list(rng.choice(pool, size=min(8, len(pool)), replace=False))
    if name == "trend_chaser":
        s = P["x_r20"].rank(pct=True) + P["x_vol_surge5"].rank(pct=True)
        return list(s.nlargest(8).index)
    return list(P["evidence"].nlargest(8).index)


def update_shadows(stocks, market_close=None, today=None):
    from .live import PRED_DIR
    C = stocks["Close"]
    today = today or C.index[-1]
    px = C.loc[today].to_dict()
    if "SPY" not in px:
        import yfinance as yf
        px["SPY"] = float(yf.download("SPY", period="5d", progress=False, auto_adjust=True)["Close"].iloc[-1].squeeze())
    is_friday = pd.Timestamp(today).weekday() == 4
    pf = PRED_DIR / f"{pd.Timestamp(today).date()}.parquet"
    rng = np.random.default_rng(int(pd.Timestamp(today).strftime("%Y%m%d")))
    for n in NAMES:
        s = _load(n)
        val = s["cash"] + sum(q * px.get(t, 0) for t, q in s["hold"].items())
        if (not s["hold"] or is_friday) and (pf.exists() or n == "spy"):
            picks = _pick(n, pd.read_parquet(pf) if pf.exists() else None, rng)
            picks = [t for t in picks if px.get(t) and np.isfinite(px[t])]
            if picks:
                val *= 1 - 2 * K.COST_BPS_LIQUID / 1e4 if s["hold"] else 1
                s["hold"] = {t: val / len(picks) / px[t] for t in picks}
                s["cash"] = 0.0
        s["equity"] = [e for e in s["equity"] if e["date"] != str(pd.Timestamp(today).date())]
        s["equity"].append({"date": str(pd.Timestamp(today).date()), "equity": round(val, 2)})
        (DIR / f"{n}.json").write_text(json.dumps(s, indent=1))
