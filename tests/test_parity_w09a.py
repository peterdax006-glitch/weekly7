"""F15 (C69 sections 21-22, 27, 28; canon C56, C64; ledger W-13): the w09a parity failure.

Cause: Feed.history() cut the MARKET frames by the stock-session POSITION. The market calendar (SPY + index history) is not
the stock calendar: one market-only row made the live market history end one session before `now` for the rest of the
window, so the live path lost today's SPY (m_spy_*, ear NaN; parity 'max diff 1' = the NaN-mismatch sentinel). The mirror
case - a stock session with no market row - made the positional cut serve the NEXT session's market bar (a future read).
Fix: cut every frame by DATE; parity_test checks the served calendar on every session (calendar_parity)."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import blind_gates as BG, livesim


def seal_file(tmp, run_id="p1", seed=11):
    rec = BG.seal_window([], seed, "2026-01-01", tag=run_id)
    rec["digest"] = BG.seal_digest(rec)
    (tmp / f"sealed_{run_id}.json").write_text(json.dumps(rec))
    return rec


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(livesim, "DIR", tmp_path)
    return tmp_path


def make_data(start, n=10, seed=5, stock_gap=None, market_gap=None):
    """Two years of warm-up plus the 12-month window, business days. `stock_gap`/`market_gap` = offset (in sessions from the
    window start) of one date removed from the stock panel only / the market frames only."""
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=2), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = [f"T{chr(65 + i)}{i}" for i in range(n)]
    close = pd.DataFrame(40 * np.exp(rng.normal(0.0003, 0.02, (len(idx), n)).cumsum(0)), index=idx, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    vol = pd.DataFrame(rng.integers(50_000, 500_000, close.shape).astype(float), index=idx, columns=tick)
    stocks = {"Close": close, "Open": opn, "High": np.maximum(close, opn) * 1.01, "Low": np.minimum(close, opn) * 0.99,
              "Volume": vol}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()),
                       "^VIX": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    market = {f: mk.copy() for f in ("Close", "Open", "High", "Low")}
    market["Volume"] = mk * 1e6
    first = idx.searchsorted(start)
    if stock_gap is not None:
        stocks = {f: v.drop(idx[first + stock_gap]) for f, v in stocks.items()}
    if market_gap is not None:
        market = {f: v.drop(idx[first + market_gap]) for f, v in market.items()}
    at = pd.DatetimeIndex(idx[::11], tz="UTC") + pd.Timedelta(hours=14)
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at,
                       "kind": ["EARN" if i % 3 == 0 else "AGREEMENT" for i in range(len(at))], "form": "8-K"})
    fd = idx[::9]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2),
                        "owner_cik": [i % 4 for i in range(len(fd))], "value": 60_000.0, "relation": "officer", "title": "ceo"})
    sic = pd.DataFrame({"ticker": tick, "sic": 3570})
    return stocks, market, ev, ins, sic


def feed_for(home, cls=None, **gaps):
    rec = seal_file(home)
    cls = cls or livesim.blind_feed_class(True)
    return cls(livesim.SealedYear("p1"), warmup_years=2, data=make_data(rec["start"], **gaps))


def old_positional_history(self, lookback=None):
    """The pre-F15 cut, kept here only to prove the new checks catch it."""
    lo = 0 if lookback is None else max(0, self.i - lookback)
    cut = lambda d: {f: v.iloc[lo:self.i + 1] for f, v in d.items()}
    return cut(self._stocks), cut(self._market)


# ---------------------------------------------------------------- the w09a shape: one market-only row inside the window
def test_market_history_ends_on_now_after_a_market_only_row(home):
    feed = feed_for(home, stock_gap=5)
    s, mi = feed.sessions, feed._market["Close"].index
    assert len(mi.difference(s)) == 1 and len(s.difference(mi)) == 0      # the planted w09a calendar
    feed.i = s.get_loc(feed.first_live) + 40
    stocks, market = feed.history()
    assert stocks["Close"].index[-1] == feed.now
    for f, v in market.items():
        assert v.index[-1] == feed.now, f                                  # old code: the session before now
    assert feed.history(lookback=10)[1]["Close"].index[0] >= s[feed.i - 10]



def test_market_is_served_on_the_stock_calendar_only(home):
    """Main session, 30 Sep: the market-only row itself (an index printing on a day stocks were shut) is an unusual date the
    trader could use to recognise the year (leak channel 6); the live history serves the market on stock sessions only, as the
    fast path does."""
    feed = feed_for(home, stock_gap=5)
    s = feed.sessions
    extra = feed._market["Close"].index.difference(s)
    feed.i = s.get_loc(feed.first_live) + 40
    _, market = feed.history()
    for f, v in market.items():
        assert v.index.equals(s[: feed.i + 1]) and not v.index.isin(extra).any(), f

def test_market_history_never_serves_the_next_session_when_market_lacks_a_stock_day(home):
    feed = feed_for(home, market_gap=5)
    s = feed.sessions
    assert len(s.difference(feed._market["Close"].index)) == 1
    for k in range(3, 60):
        feed.i = s.get_loc(feed.first_live) + k
        for f, v in feed.history()[1].items():
            assert v.index[-1] <= feed.now, (k, f)                          # old code: tomorrow's market bar from k=5 on


def test_calendar_parity_is_clean_on_both_misalignments_and_the_aligned_case(home):
    for gaps in ({}, {"stock_gap": 5}, {"market_gap": 5}):
        assert livesim.calendar_parity(feed_for(home, **gaps)) == [], gaps


def test_calendar_parity_catches_the_old_positional_cut(home, monkeypatch):
    monkeypatch.setattr(livesim.Feed, "history", old_positional_history)
    lagged = livesim.calendar_parity(feed_for(home, stock_gap=5))
    assert lagged and all(got < want for _, key, got, want in lagged if key[0] == "market")
    assert all(key[0] == "market" for _, key, _, _ in lagged)              # stock frames were never misaligned
    ahead = livesim.calendar_parity(feed_for(home, market_gap=5))
    assert ahead and any(got > d for d, _, got, _ in ahead)                # the future read is flagged too
    feed = feed_for(home, stock_gap=5)
    feed.precompute_features()
    with pytest.raises(RuntimeError, match="PARITY FAIL"):
        livesim.parity_test(feed)


def test_features_match_live_after_a_market_only_row(home):
    """The w09a failure itself: fast features == live-computed ones on sessions after the extra market row."""
    feed = feed_for(home, stock_gap=2)
    feed.precompute_features()
    assert livesim.parity_test(feed, n_days=3, seed=4) <= 1e-4


def test_old_live_path_reproduces_the_w09a_nan_mismatch(home, monkeypatch):
    """Plant the old cut WITHOUT the calendar gate: the feature comparison alone must fail with the NaN sentinel (1.0) -
    exactly the 'max diff 1' in w09a's health2.err - and its cause is the market (m_spy_*) columns."""
    monkeypatch.setattr(livesim.Feed, "history", old_positional_history)
    monkeypatch.setattr(livesim, "calendar_parity", lambda feed: [])
    feed = feed_for(home, stock_gap=2)
    feed.precompute_features()
    with pytest.raises(RuntimeError, match=r"max diff 1\b"):
        livesim.parity_test(feed, n_days=2, seed=4)
    feed.i = feed.sessions.get_loc(feed.first_live) + 30
    _, market = feed.history()
    spy = market["Close"]["SPY"].reindex(feed.history()[0]["Close"].index)
    assert np.isnan(spy.iloc[-1])                                          # today's SPY is missing on the live path


def test_default_parity_sample_is_reproducible(home, monkeypatch):
    """No seed -> the sampled days are fixed per window, so a failure reproduces on a rerun (w09a's was unseeded)."""
    feed = feed_for(home)
    seen = []
    real = livesim.features.build

    def spy_build(stocks, *a, **k):
        seen.append(stocks["Close"].index[-1])
        return real(stocks, *a, **k)
    feed.precompute_features()
    monkeypatch.setattr(livesim.features, "build", spy_build)
    livesim.parity_test(feed)
    first = list(seen)
    seen.clear()
    livesim.parity_test(feed)
    assert first == seen and len(first) == 2


def test_information_ledger_sources_are_dated_by_calendar(home):
    feed = feed_for(home, market_gap=5, cls=livesim.Feed)
    feed.i = feed.sessions.get_loc(feed.first_live) + 5                    # the session the market lacks
    src = feed._information_sources()
    assert src["close"] == feed.now and src["market_close"] < feed.now     # never a later market bar
    feed.i = 0
    assert feed._information_sources()["market_close"] == feed.sessions[0]
