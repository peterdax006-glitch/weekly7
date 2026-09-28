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
    out = {
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
