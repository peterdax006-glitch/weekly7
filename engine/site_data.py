"""Writes site/data.json — everything the dashboard shows (Blueprint Part J)."""
import json
from datetime import datetime, timezone

import pandas as pd

from . import config as K
from .scoring import load_scores


def _j(p, d):
    return json.loads(p.read_text()) if p.exists() else d


def _jsonl(p, n=None):
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[-n:] if n else rows


def write_site(broker, prices):
    from .live import EQUITY, META, DECISIONS, PRED_DIR
    from .shadows import NAMES, DIR as SH
    acct = broker.account(prices)
    pos = broker.positions()
    meta = _j(META, {})
    holdings = []
    for t, p in pos.items():
        px = prices.get(t, p["avg"])
        m = meta.get(t, {})
        holdings.append({"ticker": t, "qty": p["qty"], "avg": p["avg"], "price": px,
                         "value": p["qty"] * px, "weight": p["qty"] * px / acct["equity"],
                         "pnl": px / p["avg"] - 1, "stop": m.get("stop"), "opened": m.get("opened"),
                         "why": m.get("why", []), "p_target": m.get("p_target")})
    preds = sorted(PRED_DIR.glob("*.parquet"))
    lastP = pd.read_parquet(preds[-1]) if preds else None
    from .live import PLAIN
    spaced = {k.replace("_", " "): v for k, v in PLAIN.items()}
    for h in holdings:
        h["why"] = [spaced.get(w, w) for w in h["why"]]       # older records stored raw feature names
        if not h["why"] and lastP is not None and "why" in lastP and h["ticker"] in lastP.index and lastP.loc[h["ticker"], "why"]:
            h["why"] = lastP.loc[h["ticker"], "why"].split(" | ")
        if h["p_target"] is None and lastP is not None and h["ticker"] in lastP.index:
            h["p_target"] = float(lastP.loc[h["ticker"], "p_target"])
    eq = _j(EQUITY, [])
    weekly = []
    if eq:
        s = pd.Series({pd.Timestamp(e["date"]): e["equity"] for e in eq}).sort_index()
        s = pd.concat([pd.Series({s.index[0] - pd.Timedelta(days=3): K.START_CASH}), s])
        w = s.resample("W-FRI").last().pct_change().dropna()
        weekly = [{"week_end": str(d.date()), "ret": float(r)} for d, r in w.items()]
    shadows = {n: _j(SH / f"{n}.json", {}).get("equity", []) for n in NAMES}
    watch = []
    preds = sorted(PRED_DIR.glob("*.parquet"))
    if preds:
        P = pd.read_parquet(preds[-1])
        top = P.nlargest(20, "score")
        watch = [{"ticker": t, "score": float(r["score"]), "p_target": float(r["p_target"]),
                  "evidence": float(r["evidence"])} for t, r in top.iterrows()]
    S = load_scores()
    signal = []
    if not S.empty:
        for _, r in S.tail(60).iterrows():
            signal.append({"date": r["date"], "score_ic": r["ic"].get("score"), "evidence_ic": r["ic"].get("evidence"),
                           "model_ic": r["ic"].get("mu_raw"), "pred_rate": r["pred_rate"], "actual_rate": r["actual_rate"]})
    orders = _j(K.STATE / "ledger.json", {}).get("orders", []) + _jsonl(K.STATE / "orders.jsonl")
    dec = _jsonl(DECISIONS, 1)
    reports = sorted((K.STATE / "reports").glob("*.md")) if (K.STATE / "reports").exists() else []
    from .live import week_state
    wr, days_left, _ = week_state(acct["equity"])
    live = {"week_ret": wr, "need": (1 + K.WEEKLY_TARGET) / (1 + wr) - 1, "days_left": days_left,
            "p_week7": week_probability(holdings, lastP, acct["equity"], wr, days_left)}
    out = {
        "live": live,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "broker": broker.name, "start_cash": K.START_CASH, "target": K.WEEKLY_TARGET,
        "equity": acct["equity"], "cash": acct["cash"],
        "equity_history": eq, "weekly": weekly, "shadows": shadows,
        "holdings": sorted(holdings, key=lambda h: -h["value"]),
        "last_decision": dec[0] if dec else None,
        "orders": orders[-60:][::-1], "watchlist": watch, "signal_health": signal,
        "experiments": _jsonl(K.STATE / "experiments.jsonl", 40)[::-1],
        "challengers": _j(K.STATE / "challengers.json", []),
        "findings": _j(K.STATE / "findings.json", {}),
        "engine_version": _j(K.STATE / "models" / "meta.json", {}).get("version", "1.0"),
        "latest_report": reports[-1].read_text(encoding="utf-8") if reports else None,
        "research": _j(K.STATE / "research" / "backtest_summary.json", None),
        "tuning": _j(K.STATE / "research" / "tuning_summary.json", None),
    }
    (K.SITE / "data.json").write_text(json.dumps(_clean(out), default=float, allow_nan=False))


def _clean(o):
    """Browsers reject NaN/Infinity in JSON; a single one would blank the dashboard."""
    import math
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, str)) or o is None:
        return o
    try:
        f = float(o)
        if isinstance(o, (int, float)) or hasattr(o, "dtype"):
            return None if (math.isnan(f) or math.isinf(f)) else (int(o) if isinstance(o, int) else f)
    except (TypeError, ValueError):
        pass
    return o


def week_probability(holdings, P, equity, week_ret, days_left, n=20000):
    """P(week ends >= +7%) for the book as it stands: each holding's expected return (model, shrunk)
    and volatility, one-factor correlation, Student-t tails; remaining sessions this week."""
    import numpy as np
    if not holdings or P is None:
        return None
    sessions = days_left if days_left > 0 else 5
    need = (1 + K.WEEKLY_TARGET) / (1 + (week_ret if days_left > 0 else 0)) - 1
    w = np.array([h["weight"] for h in holdings])
    mu = np.array([0.5 * float(P.loc[h["ticker"], "mu_raw"]) * sessions / 5 if h["ticker"] in P.index else 0.0 for h in holdings])
    sd = np.array([float(P.loc[h["ticker"], "x_vol20"]) if h["ticker"] in P.index else 0.02 for h in holdings]) * np.sqrt(sessions)
    rng = np.random.default_rng(0)
    rho = 0.35
    mkt = rng.standard_normal((n, 1))
    t = rng.standard_t(4, (n, len(w))) * np.sqrt(0.5)
    r = mu + sd * (np.sqrt(rho) * mkt + np.sqrt(1 - rho) * t)
    port = np.expm1(r) @ w
    return float((port >= need).mean())
