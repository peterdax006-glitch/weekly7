"""Paper-only, trading-hours and broker-safety firewalls (Phase 28, L2-L4) against the real broker/live modules."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from engine import broker, isolation as I

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=ET)


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(broker, "LEDGER", tmp_path / "ledger.json")
    return broker.LocalBroker()


def alpaca_fake(tmp_path):
    b = object.__new__(broker.AlpacaBroker)          # skip __init__: it needs alpaca keys and a network client
    b.c, b.log = I.RecordingClient(), tmp_path / "orders.jsonl"
    return b


# ---- L2 paper only
def test_broker_source_is_paper_only():
    r = I.audit_paper_only(Path(broker.__file__).read_text(encoding="utf-8"))
    assert r["ok"], r["problems"]
    assert r["clients"] == 1


def test_paper_audit_catches_planted_live_client():
    assert not I.audit_paper_only("c = TradingClient(k, s, paper=False)\n")["ok"]
    assert not I.audit_paper_only("c = TradingClient(k, s, paper=flag)\n")["ok"]
    assert not I.audit_paper_only("c = TradingClient(k, s)\n")["ok"]
    assert not I.audit_paper_only("u = 'https://api.alpaca.markets/v2'\nTradingClient(k, s, paper=True)\n")["ok"]


def test_paper_audit_on_empty_source_does_not_vouch():
    assert not I.audit_paper_only("x = 1\n")["ok"]


# ---- L3 hours firewall
def test_regular_hours_boundaries():
    assert broker.regular_hours(et(2026, 9, 28, 9, 30))
    assert not broker.regular_hours(et(2026, 9, 28, 9, 29))
    assert broker.regular_hours(et(2026, 9, 28, 15, 59))
    assert not broker.regular_hours(et(2026, 9, 28, 16, 0))
    assert not broker.regular_hours(et(2026, 9, 26, 11))         # Saturday
    assert not broker.regular_hours(et(2026, 9, 27, 11))         # Sunday
    # UTC instants across a DST boundary still resolve by New York time
    assert broker.regular_hours(datetime(2026, 3, 9, 13, 31, tzinfo=timezone.utc))       # EDT: 9:31
    assert not broker.regular_hours(datetime(2026, 11, 2, 14, 29, tzinfo=timezone.utc))  # EST: 9:29


def test_orders_refused_when_closed_and_untouched(local, tmp_path):
    assert I.probe_order(local, False, broker) == {"raised": True, "filled": False}
    assert local.s["orders"] == [] and not local.s["positions"]
    a = alpaca_fake(tmp_path)
    assert I.probe_order(a, False, broker)["raised"]
    assert a.c.submitted == [] and not a.log.exists()


def test_orders_pass_when_open_control(local, tmp_path):
    """Control: the same probe DOES fill when open, so the refusal above is the firewall, not a broken probe."""
    assert I.probe_order(local, True, broker) == {"raised": False, "filled": True}
    a = alpaca_fake(tmp_path)
    pytest.importorskip("alpaca")
    assert I.probe_order(a, True, broker)["filled"]
    assert len(a.c.submitted) == 1


def test_calendar_known_dates():
    h = I.nyse_holidays(2026)
    assert date(2026, 1, 1) in h and date(2026, 4, 3) in h            # New Year, Good Friday
    assert date(2026, 7, 3) in h                                      # Jul 4 is Saturday -> Fri observed
    assert date(2026, 11, 26) in h and date(2026, 12, 25) in h
    assert date(2026, 6, 19) in h and date(2021, 6, 18) not in I.nyse_holidays(2021)
    assert date(2027, 12, 31) not in I.nyse_holidays(2027)            # Sat New Year 2028 is not observed
    assert date(2026, 11, 27) in I.half_days(2026) and date(2026, 12, 24) in I.half_days(2026)
    assert len(h) == 10


def test_hours_firewall_audit_finds_the_holiday_gap():
    """Defect that exists in the repo today: broker.regular_hours ignores holidays and half days."""
    ts = [et(2026, 12, 25, 11), et(2026, 11, 27, 14), et(2026, 9, 28, 11), et(2026, 9, 28, 8)]
    r = I.audit_hours_firewall(broker, ts)
    assert et(2026, 12, 25, 11).isoformat() in r["unsafe"]
    assert et(2026, 11, 27, 14).isoformat() in r["unsafe"]
    assert len(r["unsafe"]) == 2 and r["over_blocked"] == []


def test_hours_audit_clean_on_ordinary_week():
    ts = [et(2026, 9, 28, h, m) + timedelta(days=i) for i in range(5) for h in range(0, 24) for m in (0, 30)]
    r = I.audit_hours_firewall(broker, ts)
    assert r["unsafe"] == [] and r["over_blocked"] == [] and r["n"] == 240


def test_strict_session_rejects_naive():
    with pytest.raises(ValueError):
        I.strict_session(datetime(2026, 9, 28, 11))


# ---- L4 broker safety
def test_check_order_rules():
    ok = dict(equity=1000.0)
    assert I.check_order("AAPL", 1, 100, **ok) == []
    assert "would open a short" in I.check_order("AAPL", -1, 100, held_qty=0.5, **ok)
    assert I.check_order("AAPL", -1, 100, held_qty=1, **ok) == []
    assert "crypto pair not allowed" in I.check_order("BTC/USD", 1, 100, **ok)
    assert "position exceeds concentration cap" in I.check_order("AAPL", 4, 100, **ok)
    assert "below minimum notional" in I.check_order("AAPL", 0.001, 100, **ok)
    assert "bad price" in I.check_order("AAPL", 1, float("nan"), **ok)
    assert "bad quantity" in I.check_order("AAPL", float("nan"), 10, **ok)
    assert "no equity" in I.check_order("AAPL", 1, 10, equity=0)
    assert "bad ticker" in I.check_order("", 1, 10, **ok)


class FakeBroker:
    name = "fake"

    def __init__(self, pos=None):
        self.pos, self.orders = pos or {}, []

    def positions(self):
        return {t: dict(p) for t, p in self.pos.items()}

    def order(self, t, q, p, why, **k):
        self.orders.append((t, q, p))
        return {"ticker": t}


def test_live_execute_never_emits_unsafe_orders(tmp_path, monkeypatch):
    """Drive the REAL live.execute planner through SafeBroker on random targets/holdings."""
    from engine import live
    monkeypatch.setattr(live, "META", tmp_path / "meta.json")
    rng = np.random.default_rng(3)
    for trial in range(15):
        tick = [f"T{i}" for i in range(12)]
        prices = pd.Series(rng.uniform(5, 300, 12), index=tick)
        held = {t: {"qty": float(rng.uniform(0.1, 3)), "avg": prices[t]} for t in rng.choice(tick, 4, replace=False)}
        equity = 1000.0
        w = pd.Series(rng.dirichlet(np.ones(6)) * 0.98, index=rng.choice(tick, 6, replace=False)).clip(upper=0.25)
        fb = FakeBroker(held)
        sb = I.SafeBroker(fb, equity)
        xr = pd.DataFrame({"atr_pct": 0.03}, index=tick)
        P = pd.DataFrame({"p_target": 0.5}, index=tick)
        contrib = pd.DataFrame(0.0, index=tick, columns=["r5"])
        live.execute(sb, w, prices.to_dict(), xr, equity, "test", contrib, P)
        assert sb.refused == [], sb.refused
        assert fb.orders                                     # it did trade (non-vacuous)


def test_safe_broker_refuses_planted_short_and_overspend():
    sb = I.SafeBroker(FakeBroker({"AAPL": {"qty": 1.0, "avg": 100}}), 1000.0)
    with pytest.raises(I.IsolationError):
        sb.order("AAPL", -2.0, 100, "planted short")
    with pytest.raises(I.IsolationError):
        sb.order("MSFT", 5.0, 100, "planted oversize")
    assert sb.inner.orders == [] and len(sb.refused) == 2


def test_safe_broker_budget_across_many_small_buys():
    sb = I.SafeBroker(FakeBroker(), 1000.0, gross_cap=1.0)
    for t in "ABCD":
        sb.order(t, 2.0, 100.0, "x")                          # 200 each = 800
    sb.order("E", 1.9, 100.0, "x")                            # 990
    with pytest.raises(I.IsolationError, match="budget"):
        sb.order("F", 1.0, 100.0, "x")


def test_live_and_broker_modules_never_open_sealed_windows():
    rep = I.audit_imports()
    assert [v for v in rep["violations"] if v["kind"] == "live_reads_sealed"] == []
