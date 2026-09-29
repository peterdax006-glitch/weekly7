"""Canon C6 at the order path: engine.broker.regular_hours refuses NYSE holidays and the afternoon of half days, and
still allows a normal session. Before 2026-09-29 it only checked weekday 9:30-16:00 (B13 audit: 1,090 closed slots
allowed in 2019-26)."""
from datetime import datetime
from zoneinfo import ZoneInfo

from engine import broker

ET = ZoneInfo("America/New_York")


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_normal_session_open_and_closed_edges():
    assert broker.regular_hours(at(2025, 3, 12, 10, 0))
    assert broker.regular_hours(at(2025, 3, 12, 9, 30))
    assert not broker.regular_hours(at(2025, 3, 12, 9, 29))
    assert not broker.regular_hours(at(2025, 3, 12, 16, 0))


def test_weekend_closed():
    assert not broker.regular_hours(at(2025, 3, 15, 11, 0))


def test_holidays_closed():
    for y, m, d in ((2025, 12, 25), (2025, 7, 4), (2025, 1, 1), (2024, 11, 28), (2025, 4, 18)):   # incl. Good Friday
        assert not broker.regular_hours(at(y, m, d, 11, 0)), (y, m, d)


def test_half_day_afternoon_closed_morning_open():
    assert broker.regular_hours(at(2024, 11, 29, 11, 0))            # day after Thanksgiving: open until 13:00
    assert not broker.regular_hours(at(2024, 11, 29, 14, 0))


def test_utc_input_is_converted():
    from datetime import timezone
    assert broker.regular_hours(datetime(2025, 3, 12, 15, 0, tzinfo=timezone.utc))      # 11:00 ET
