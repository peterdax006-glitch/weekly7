"""Canon C33 gate: engine.fill_audit on Session-shaped records. Honest next-open fills pass; a same-close fill, a
close-priced fill, a weekend fill and a Session with no fill records all fail (fail closed)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from engine import fill_audit

IDX = pd.bdate_range("2021-03-01", periods=10)
CLOSES = pd.DataFrame({"A": np.linspace(10, 12, 10), "B": np.linspace(20, 18, 10)}, index=IDX)
OPENS = CLOSES.shift(1).fillna(CLOSES.iloc[0]) * 1.01


def fill(n, t, dec_i, fill_i, price=None, fill_date=None):
    fd = fill_date if fill_date is not None else IDX[fill_i]
    return {"order_id": f"o{n}", "ticker": t, "decision_date": IDX[dec_i], "fill_date": fd,
            "fill_price": OPENS.loc[IDX[fill_i], t] if price is None else price}


def S(fills):
    return SimpleNamespace(fills=fills)


def test_honest_next_open_fills_pass():
    ok, msg = fill_audit.gate(S([fill(1, "A", 2, 3), fill(2, "B", 4, 5)]), OPENS, CLOSES)
    assert ok, msg


def test_same_close_fill_fails():
    ok, msg = fill_audit.gate(S([fill(1, "A", 3, 3)]), OPENS, CLOSES)
    assert not ok


def test_fill_at_close_price_fails():
    ok, msg = fill_audit.gate(S([fill(1, "A", 2, 3, price=CLOSES.loc[IDX[3], "A"])]), OPENS, CLOSES)
    assert not ok, msg


def test_weekend_fill_fails():
    sat = IDX[4] + pd.Timedelta(days=1)                               # 2021-03-06 is a Saturday
    ok, msg = fill_audit.gate(S([fill(1, "A", 4, 5, fill_date=sat)]), OPENS, CLOSES)
    assert not ok


def test_skipping_a_session_fails():
    ok, msg = fill_audit.gate(S([fill(1, "A", 2, 4)]), OPENS, CLOSES)   # decided day 2, filled day 4: not the NEXT open
    assert not ok


def test_session_without_fill_records_is_unauditable_not_pass():
    r = fill_audit.audit_session(SimpleNamespace(), OPENS, CLOSES)
    assert r["status"] == "UNAUDITABLE"
    assert not fill_audit.gate(SimpleNamespace(), OPENS, CLOSES)[0]


def test_missing_opens_cannot_pass():
    ok, _ = fill_audit.gate(S([fill(1, "A", 2, 3, price=CLOSES.loc[IDX[3], "A"])]), None, CLOSES)
    assert not ok


def test_no_trades_is_a_clean_pass():
    r = fill_audit.audit_session(S([]), OPENS, CLOSES)
    assert r["status"] == "PASS" and r["n_fills"] == 0
