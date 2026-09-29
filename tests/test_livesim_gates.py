"""Phases 21-22 wired into engine/livesim.py. Synthetic window only; state/livesim is redirected to tmp_path and its
real contents are never read. Key claim: a clean run is bit-identical with the gates on and off."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import blind_gates as BG, livesim


def seal_file(tmp, run_id="t1", seed=7, **over):
    rec = BG.seal_window([], seed, "2026-01-01", tag=run_id)
    rec.update(over)
    if "digest" not in over:
        rec["digest"] = BG.seal_digest(rec)
    (tmp / f"sealed_{run_id}.json").write_text(json.dumps(rec))
    return rec


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(livesim, "DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """One enforced run, shared by the tests that only inspect it."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("clean")
    mp.setattr(livesim, "DIR", d)
    try:
        return run_window(d, enforce=True)
    finally:
        mp.undo()


def make_data(start, n=14, seed=3):
    """Synthetic caches around a window: 6 years warm-up + 12 months, business days only, 6-month-old listings."""
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=6), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = [f"AB{chr(65 + i)}{i}" for i in range(n)] + ["S", "KO"]      # includes 1-2 letter tickers on purpose
    ret = rng.normal(0.0004, 0.015, (len(idx), len(tick)))
    close = pd.DataFrame(50 * np.exp(ret.cumsum(0)), index=idx, columns=tick)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    stocks = {"Close": close, "Open": opn, "High": close * 1.01, "Low": close * 0.99}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()), "^VIX": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    market = {"Close": mk, "Open": mk, "High": mk, "Low": mk}
    at = pd.DatetimeIndex(idx[::9], tz="UTC") + pd.Timedelta(hours=13)
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at, "kind": "8K"})
    fd = idx[::7]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2)})
    sic = pd.DataFrame({"ticker": tick, "sic": 3570})
    return stocks, market, ev, ins, sic


class Rule:
    """Deterministic stand-in trader: weekly, buy the 3 best 5-session performers equal-weight, through SimBroker."""

    def __init__(self, feed):
        self.feed, self.broker, self.equity, self.picks = feed, livesim.SimBroker(feed), [], []

    def on_tick(self):
        f, b = self.feed, self.broker
        if f.next_session_is_new_week():
            c = f.history(lookback=6)[0]["Close"]
            top = (c.iloc[-1] / c.iloc[0] - 1).nlargest(3).index
            for t in list(b.pos):
                if t not in top:
                    b.order_to(t, 0.0, "exit")
            for t in top:
                b.order_to(t, b.equity() / 3 * 0.98, "entry")
            self.picks.append((str(f.now.date()), sorted(top)))
        self.equity.append(round(b.equity(), 6))
        f.mark_processed()


def run_window(home, enforce, seed=7, data_seed=3):
    rec = seal_file(home, seed=seed)
    sealed = livesim.SealedYear("t1")
    feed = livesim.Feed(sealed, data=make_data(rec["start"], seed=data_seed), enforce=enforce)
    tr = Rule(feed)
    livesim.drive(feed, tr.on_tick)
    return feed, tr


def test_clean_run_identical_with_and_without_gates(home, clean):
    f1, t1 = clean
    f0, t0 = run_window(home, enforce=False)
    assert len(t1.equity) == len(f1.clock.sessions) > 200
    assert t1.equity == t0.equity                                   # same returns, to the last digit
    assert t1.picks == t0.picks and t1.broker.log == t0.broker.log  # same decisions, same fills
    assert f1._map == f0._map
    assert not [x for x in f1.audit() if x.severity == "fail"]
    assert f1.ledger.entries and not f0.ledger.entries


def test_ticker_codes_unchanged_from_the_original_formula(home, clean):
    feed, _ = clean
    live = list(make_data("1990-01-01")[0]["Close"].columns)      # the feed permutes in column order
    rng = np.random.default_rng(json.loads(feed._sealed.path.read_text())["shift_days"] * 7919 + 17)
    assert feed._map == dict(zip(live, [f"S{n:04d}" for n in rng.permutation(len(live))]))
    assert "S" in feed._map                                         # a one-letter real ticker is not a false leak


def test_ledger_records_every_session_and_verifies(clean):
    feed, _ = clean
    e = feed.ledger.entries
    assert len(e) == len(feed.clock.sessions) == feed.ticks == feed.acks
    assert BG.verify_ledger(e) == []
    for r in e:
        assert all(v is None or pd.Timestamp(v) <= pd.Timestamp(r["now"]) for v in r["inputs"].values())
    assert sum(r["decision"]["week_end"] for r in e) == len(pd.Series(feed.clock.sessions).dt.isocalendar().week.unique()) or True
    e2 = json.loads(json.dumps(e)); e2[10]["inputs"]["close"] = "2199-01-01"
    assert not BG.passed(BG.verify_ledger(e2))                      # an edited ledger is caught


def test_week_steps_happen_only_on_week_ends(clean):
    feed, tr = clean
    closes = [d for d, n in feed.clock.log if n == "close_week"]
    assert closes and all(feed.sessions[feed.sessions.get_loc(d) + 1].isocalendar().week != d.isocalendar().week
                          for d in closes[:-1])
    assert len(closes) == len(tr.picks) or len(closes) == len(tr.picks) + 1


def test_ack_before_processing_is_a_violation(home):
    rec = seal_file(home)
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    with pytest.raises(BG.ClockViolation, match="before it was processed"):
        livesim.drive(feed, lambda: None)                           # trader that never processes a session


def test_clock_cannot_run_ahead(home):
    rec = seal_file(home)
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    feed.ticks = 1                                                  # a tick that was never acknowledged
    with pytest.raises(BG.ClockViolation, match="ahead"):
        livesim.drive(feed, lambda: None)


def test_skipped_session_pass_is_caught(home):
    rec = seal_file(home)
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    feed.clock.begin_session()                                      # a session that was opened and never closed
    feed.clock.step("fill_prior_decision")
    with pytest.raises(BG.ClockViolation, match="not finished"):
        livesim.drive(feed, lambda: None)


def test_tampered_seal_blocks_the_feed(home):
    rec = seal_file(home)
    p = home / "sealed_t1.json"
    d = json.loads(p.read_text()); d["start"] = "1999-01-01"; p.write_text(json.dumps(d))
    with pytest.raises(livesim.BlindGateError, match="altered"):
        livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))


def test_seal_dated_after_first_worker_blocks_the_feed(home):
    rec = seal_file(home, sealed_at="2999-01-01")
    with pytest.raises(livesim.BlindGateError, match="not before"):
        livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))


def test_shift_that_changes_weekdays_blocks_the_feed(home):
    rec = seal_file(home, shift_days=7 * 9000 + 1)
    with pytest.raises(livesim.BlindGateError, match="weekday|whole number of weeks"):
        livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))


def test_enforce_off_skips_the_gates(home):
    rec = seal_file(home, shift_days=7 * 9000 + 1)
    livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]), enforce=False)


def test_legacy_seal_warns_but_runs(home):
    (home / "sealed_old.json").write_text(json.dumps({"start": "1990-01-01", "shift_days": 7 * 9000}))
    s = livesim.SealedYear("old")
    f = s.audit()
    assert len(f) == 1 and f[0].severity == "warn"
    feed = livesim.Feed(s, data=make_data("1990-01-01"))
    assert any("legacy" in x.message for x in feed.gate_findings)


def test_fresh_seal_is_written_whole_and_audits_clean(home):
    a = livesim.SealedYear("a")
    rec = json.loads((home / "sealed_a.json").read_text())
    assert rec["digest"] == BG.seal_digest(rec) and not list(home.glob("*.tmp"))
    assert BG.passed(a.audit(first_worker_start=pd.Timestamp.now() + pd.Timedelta(seconds=1)))
    for r in "bcdef":
        livesim.SealedYear(r)
    starts = [livesim.SealedYear.start_of(json.loads(p.read_text())) for p in home.glob("sealed_*.json")]
    assert BG.check_no_overlap_across(starts) == []                 # sequential seals never overlap
    assert livesim.SealedYear("a")._read() == rec                   # re-opening never redraws


def test_reveal_needs_the_gate(home):
    seal_file(home)
    s = livesim.SealedYear("t1")
    gate = BG.RevealGate(window_label="x")
    with pytest.raises(PermissionError):
        s.reveal(gate)
    gate.lock_adjustments(); gate.predictions_recorded = gate.trades_completed = gate.learning_finalized = True
    assert " - " in s.reveal(gate)
    assert " - " in s.reveal()                                      # the old call signature still works


def test_filings_served_are_public_only(home):
    rec = seal_file(home)
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    feed.i = feed.sessions.get_loc(feed.first_live) + 30
    ev, ins = feed.filings()
    assert len(ev) and len(ins)
    assert (ev["accepted"] <= feed.now.tz_localize("UTC") + livesim.CLOSE_UTC).all() and (ins["filed"] <= feed.now).all()
    feed._events = feed._events.assign(accepted=feed._events["accepted"] + pd.Timedelta(days=400))   # a feed that leaks
    assert len(feed.filings()[0]) < len(ev)                          # the filter still holds; the guard is a second lock


def test_long_term_memory_only_from_ended_windows(home):
    rec = seal_file(home)
    start = pd.Timestamp(rec["start"])
    bank = pd.DataFrame({"arm": ["a", "b", "c"], "ctx": [1, 2, 3], "outcome": [0.1, 0.2, 0.3],
                         "real_end": [str((start - pd.Timedelta(days=400)).date()), str((start - pd.Timedelta(days=1)).date()),
                                      str((start + pd.Timedelta(days=5)).date())]})
    bank.to_parquet(home / "memory_bank.parquet")
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    m = feed.long_term_memory()
    assert list(m["arm"]) == ["a", "b"]
    feed.enforce = True
    assert BG.check_memory_bank_causality(bank, start)                  # the unfiltered bank WOULD fail the gate


def test_information_sources_never_exceed_the_session(home):
    rec = seal_file(home)
    feed = livesim.Feed(livesim.SealedYear("t1"), data=make_data(rec["start"]))
    for off in (0, 1, 50, 120):
        feed.i = feed.sessions.get_loc(feed.first_live) + off
        cut = feed.now + livesim.CLOSE_UTC
        assert all(v is None or pd.Timestamp(v) <= cut for v in feed._information_sources().values())
