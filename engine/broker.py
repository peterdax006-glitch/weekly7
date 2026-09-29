"""Execution (Blueprint Part F). Two interchangeable brokers:

AlpacaBroker  paper account via alpaca-py (realistic fills) when ALPACA_KEY/ALPACA_SECRET exist.
LocalBroker   deterministic ledger in state/ledger.json filled at the last price + cost model.

Every run reconciles to the broker's actual positions, so missed or duplicated scheduler
runs cannot corrupt the book (the live state is always re-read, never assumed)."""
import json, os
from datetime import datetime, timezone

from . import config as K

LEDGER = K.STATE / "ledger.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def regular_hours(ts=None) -> bool:
    """Canon C6: orders only during the regular session - NYSE holidays and half-day early closes included.
    The old weekday 9:30-16:00 rule allowed 1,090 of 140,256 half-hour slots in 2019-26 that were closed (988 holiday,
    102 half-day afternoons; B13 audit 2026-09-29), relying on engine.tick alone to screen holidays."""
    from .isolation import strict_session
    return strict_session(ts or datetime.now(timezone.utc))


class OutsideHours(Exception):
    pass


class LocalBroker:
    name = "local"

    def is_session(self, day):
        return None                                               # no calendar source: the caller falls back

    def __init__(self):
        if LEDGER.exists():
            self.s = json.loads(LEDGER.read_text())
        else:
            self.s = {"cash": K.START_CASH, "positions": {}, "orders": [], "created": now()}

    def save(self):
        LEDGER.write_text(json.dumps(self.s, indent=1))

    def account(self, prices):
        eq = self.s["cash"] + sum(p["qty"] * prices.get(t, p["avg"]) for t, p in self.s["positions"].items())
        return {"equity": eq, "cash": self.s["cash"]}

    def positions(self):
        return {t: dict(p) for t, p in self.s["positions"].items()}

    def order(self, ticker, qty, price, reason, cost_bps=K.COST_BPS_LIQUID):
        if not regular_hours():
            raise OutsideHours(f"refused {ticker} {qty:+.4f}: outside regular trading hours (canon C6)")
        if abs(qty * price) < 1.0:
            return None
        fill = price * (1 + cost_bps / 1e4 * (1 if qty > 0 else -1))    # pay the spread + slippage
        self.s["cash"] -= qty * fill
        p = self.s["positions"].get(ticker, {"qty": 0.0, "avg": fill, "opened": now()})
        if qty > 0:
            p["avg"] = (p["avg"] * p["qty"] + qty * fill) / (p["qty"] + qty)
        p["qty"] += qty
        if abs(p["qty"]) * fill < 0.5:
            self.s["positions"].pop(ticker, None)
        else:
            self.s["positions"][ticker] = p
        o = {"t": now(), "ticker": ticker, "qty": round(qty, 6), "fill": round(fill, 4), "reason": reason}
        self.s["orders"].append(o)
        self.save()
        return o


class AlpacaBroker:
    name = "alpaca-paper"

    def __init__(self):
        from alpaca.trading.client import TradingClient
        self.c = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
        self.log = K.STATE / "orders.jsonl"

    def is_session(self, day):
        """Read-only exchange-calendar lookup through the same single paper client."""
        from alpaca.trading.requests import GetCalendarRequest
        return len(self.c.get_calendar(GetCalendarRequest(start=day, end=day))) > 0

    def account(self, prices=None):
        a = self.c.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "daytrades": int(a.daytrade_count or 0)}

    def positions(self):
        return {p.symbol: {"qty": float(p.qty), "avg": float(p.avg_entry_price)} for p in self.c.get_all_positions()}

    def order(self, ticker, qty, price, reason, cost_bps=None):
        if not regular_hours():
            raise OutsideHours(f"refused {ticker} {qty:+.4f}: outside regular trading hours (canon C6)")
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        q = round(abs(qty), 6)
        if q * price < 1.0:
            return None
        # fractional orders must be DAY; use a marketable limit (price +/- 0.3%) to cap slippage
        lim = round(price * (1.003 if qty > 0 else 0.997), 2)
        try:
            req = LimitOrderRequest(symbol=ticker, qty=q, side=side, time_in_force=TimeInForce.DAY, limit_price=lim)
            o = self.c.submit_order(req)
        except Exception:
            req = MarketOrderRequest(symbol=ticker, qty=q, side=side, time_in_force=TimeInForce.DAY)
            o = self.c.submit_order(req)
        rec = {"t": now(), "ticker": ticker, "qty": qty, "limit": lim, "id": str(o.id), "reason": reason}
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def close_all(self, ticker, reason):
        self.c.close_position(ticker)


def get_broker():
    if os.environ.get("ALPACA_KEY") and os.environ.get("ALPACA_SECRET"):
        return AlpacaBroker()
    return LocalBroker()
