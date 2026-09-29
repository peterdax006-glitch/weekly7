"""Phases 21-22: each gate must pass a clean run and catch a planted defect."""
import numpy as np
import pandas as pd
import pytest

from engine import blind_gates as G


def _sessions(n=30, start="2101-01-04"):
    return pd.bdate_range(start, periods=n)


# ---------------- seal
def test_seal_is_deterministic_and_clean():
    a = G.seal_window([], 5, "2026-01-01")
    assert a == G.seal_window([], 5, "2026-01-01")
    assert G.passed(G.check_seal(a, first_worker_start="2026-02-01"))
    assert pd.Timestamp(a["start"]).day == 1 and a["shift_days"] % 7 == 0


def test_seal_avoids_played_windows():
    used = [str(m.date()) for m in pd.date_range("1965-01-01", "2020-01-01", freq="6MS")]
    for s in range(20):
        r = G.seal_window(used, s, "2026-01-01")
        assert all(G.overlap_days(r["start"], u) <= G.MAX_OVERLAP_DAYS for u in used)


def test_seal_crowded_calendar_falls_back_without_error():
    used = [str(m.date()) for m in pd.date_range("1965-01-01", "2025-09-01", freq="MS")]
    r = G.seal_window(used, 1, "2026-01-01")
    assert r["start"] in used


def test_tampered_seal_caught():
    r = G.seal_window([], 1, "2026-01-01")
    r["start"] = "1999-01-01"
    assert any("altered" in f.message for f in G.check_seal(r))


def test_seal_after_worker_start_caught():
    r = G.seal_window([], 1, "2026-03-01")
    f = G.check_seal(r, first_worker_start="2026-02-01")
    assert not G.passed(f) and "not before" in f[0].message


def test_seal_overlap_and_non_month_start_caught():
    r = G.seal_window([], 1, "2026-01-01")
    other = str((pd.Timestamp(r["start"]) + pd.DateOffset(months=2)).date())
    assert not G.passed(G.check_seal(r, other_starts=[other]))
    assert not G.passed(G.check_no_overlap_across([r["start"], other]))
    assert G.passed(G.check_no_overlap_across([r["start"], "1990-01-01"]))
    bad = dict(r, start="1980-01-15")
    bad["digest"] = G.seal_digest(bad)
    assert any("month start" in f.message for f in G.check_seal(bad))


def test_seal_missing_fields_and_worker_order():
    assert not G.passed(G.check_seal({"start": "1980-01-01"}))
    f = G.check_sealed_before_workers("2026-01-02", {"w1": "2026-01-03", "w2": "2026-01-01"})
    assert len(f) == 1 and "w2" in f[0].message


# ---------------- disguise
def test_shift_checks():
    real = pd.bdate_range("1980-01-07", periods=10)
    good = 7 * 9000
    assert G.passed(G.check_shift(good, real, real + pd.Timedelta(days=good)))
    assert not G.passed(G.check_shift(good + 1, real, real + pd.Timedelta(days=good + 1)))     # weekdays move
    assert not G.passed(G.check_shift(7 * 100, real, real + pd.Timedelta(days=700)))          # real-era year
    sh = (real + pd.Timedelta(days=good)).to_series().reset_index(drop=True)
    sh.iloc[3] += pd.Timedelta(days=7)                                                        # one date bent
    assert not G.passed(G.check_shift(good, real, pd.DatetimeIndex(sh)))
    assert G.passed(G.check_shift(good, [], []))


def test_ticker_map_properties():
    tick = [f"T{i}X" for i in range(60)] + ["AAPL", "IBM"]
    m = G.make_ticker_map(tick, 8000 * 7)
    assert m == G.make_ticker_map(reversed(tick), 8000 * 7)          # input order irrelevant
    assert m != G.make_ticker_map(tick, 8001 * 7)
    assert G.passed(G.check_ticker_map(m, tick, seed=8000 * 7))
    assert len(set(m.values())) == len(tick)


def test_ticker_map_defects_caught():
    tick = sorted(f"Q{i:02d}" for i in range(40))
    ident = {t: f"S{i:04d}" for i, t in enumerate(tick)}             # sorted-order leak
    assert any("alphabetical" in f.message for f in G.check_ticker_map(ident))
    dup = dict(G.make_ticker_map(tick, 1)); dup[tick[0]] = dup[tick[1]]
    assert any("bijection" in f.message for f in G.check_ticker_map(dup))
    assert any("contains" in f.message for f in G.check_ticker_map({"S0001": "S0001x"}))
    assert any("reproducible" in f.message for f in G.check_ticker_map(G.make_ticker_map(tick, 1), seed=2))


def test_disguised_frame_leak_caught():
    real = ["AAPL", "IBM", "KO"]
    m = G.make_ticker_map(real, 63000)
    idx = pd.bdate_range("2110-01-04", periods=5)
    good = pd.DataFrame(1.0, index=idx, columns=[m[t] for t in real])
    assert G.passed(G.check_disguised_frame(good, m))
    leak = good.rename(columns={m["IBM"]: "IBM"})
    assert any("real tickers" in f.message for f in G.check_disguised_frame(leak, m))
    old = good.copy(); old.index = pd.bdate_range("1999-01-04", periods=5)
    assert any("real-history" in f.message for f in G.check_disguised_frame(old, m))
    assert any("unshifted" in f.message for f in G.check_disguised_frame(good, m, real_dates_known=idx[:2]))
    mi = pd.DataFrame({"a": 1.0}, index=pd.MultiIndex.from_product([idx, ["AAPL"]]))
    assert not G.passed(G.check_disguised_frame(mi, m))


def test_duplicate_price_paths_caught():
    idx = pd.bdate_range("2110-01-04", periods=20)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 3)).cumsum(0) + 50, index=idx, columns=["S1", "S2", "S3"])
    assert G.passed(G.check_disguise_signature(df))
    df["S3"] = df["S1"]
    assert not G.passed(G.check_disguise_signature(df))


# ---------------- warm-up and reveal
def test_warmup():
    assert G.warmup_years_available("1965-01-01") == pytest.approx(2.999, abs=0.01)
    assert G.warmup_years_available("2015-01-01") == 6.0
    assert G.passed(G.check_warmup("2015-01-01", 6.0))
    assert not G.passed(G.check_warmup("2015-01-01", 3.0))
    assert G.passed(G.check_warmup("1964-01-01", G.warmup_years_available("1964-01-01")))
    assert G.check_warmup("2015-01-01", 9.0)[0].severity == "warn"


def test_reveal_gate_refuses_until_all_four():
    g = G.RevealGate("Jan 1987 - Dec 1987")
    with pytest.raises(PermissionError, match="adjustments_locked"):
        g.reveal()
    g.lock_adjustments(); g.predictions_recorded = True; g.trades_completed = True
    with pytest.raises(PermissionError, match="learning_finalized"):
        g.reveal()
    g.learning_finalized = True
    assert g.reveal() == "Jan 1987 - Dec 1987" and g.revealed


def test_reveal_order_in_log():
    ok = [("adjustments_locked", 1), ("predictions_recorded", 2), ("trades_completed", 3),
          ("learning_finalized", 4), ("reveal", 5)]
    assert G.passed(G.check_reveal_order(ok))
    assert not G.passed(G.check_reveal_order([ok[0], ok[4], ok[1], ok[2], ok[3]]))
    assert not G.passed(G.check_reveal_order(ok + [("adjustment", 6)]))
    assert G.passed(G.check_reveal_order([]))                          # nothing revealed, nothing to object to


# ---------------- clock
def _run_session(c):
    c.begin_session()
    for s in G.STEPS:
        if s in G.WEEK_ONLY and not c.is_week_end():
            continue
        c.step(s)


def test_clock_full_run_and_week_steps():
    s = _sessions(12)
    c = G.BlindClock(s)
    for _ in s:
        _run_session(c)
    steps = pd.Series([n for _, n in c.log]).value_counts()
    assert steps["fill_prior_decision"] == 12
    assert steps["close_week"] == 3          # two Fridays plus the final session
    with pytest.raises(G.ClockViolation, match="no more"):
        c.begin_session()


def test_clock_order_violations():
    c = G.BlindClock(_sessions(5))
    with pytest.raises(G.ClockViolation, match="begin_session"):
        c.step("fill_prior_decision")
    with pytest.raises(G.ClockViolation, match="not started"):
        c.now
    c.begin_session()
    with pytest.raises(G.ClockViolation, match="requires fill_prior_decision"):
        c.step("observe_close")
    c.step("fill_prior_decision")
    with pytest.raises(G.ClockViolation, match="requires observe_close"):
        c.step("decide_next_week")
    with pytest.raises(G.ClockViolation, match="unknown"):
        c.step("peek")
    with pytest.raises(G.ClockViolation, match="not finished"):
        c.begin_session()
    with pytest.raises(G.ClockViolation, match="ended before"):
        c.finish_session()


def test_clock_rejects_bad_sessions():
    with pytest.raises(ValueError):
        G.BlindClock(pd.DatetimeIndex(["2101-01-05", "2101-01-04"]))
    c = G.BlindClock(pd.DatetimeIndex([]))
    with pytest.raises(G.ClockViolation):
        c.begin_session()


def test_week_steps_skipped_midweek_but_required_on_friday():
    s = pd.bdate_range("2101-01-04", periods=5)      # Mon-Fri
    c = G.BlindClock(s)
    c.begin_session()
    for n in ("fill_prior_decision", "observe_close", "update_state"):
        c.step(n)
    with pytest.raises(G.ClockViolation, match="record_information"):
        c.step("close_week")                          # Monday is not a week end
    c.step("record_information")
    for _ in range(3):
        _run_session(c)
    c.begin_session()
    for n in G.STEPS[:3]:
        c.step(n)
    with pytest.raises(G.ClockViolation, match="close_week"):
        c.step("record_information")                  # Friday must not skip the week close


def test_scan_information_catches_future_and_labels():
    now = pd.Timestamp("2101-03-05")
    clean = {"now": now, "fields": {"close": now, "vix": now - pd.Timedelta(days=1)},
             "frames": {"px": pd.DataFrame({"a": 1}, index=pd.bdate_range(end=now, periods=3))}}
    assert G.passed(G.scan_information([clean]))
    fut = {"now": now, "fields": {"earnings": now + pd.Timedelta(days=2)}}
    assert not G.passed(G.scan_information([fut]))
    lab = {"now": now, "fields": {"fwd_ret_5d": None}}
    assert any("forbidden" in f.message for f in G.scan_information([lab]))
    fr = {"now": now, "frames": {"px": pd.DataFrame({"a": 1}, index=pd.bdate_range(end=now + pd.Timedelta(days=3), periods=3))}}
    f = G.scan_information([fr])
    assert not G.passed(f) and "future" in f[0].message
    mi = pd.DataFrame({"a": 1}, index=pd.MultiIndex.from_product([[now + pd.Timedelta(days=1)], ["S1"]]))
    assert not G.passed(G.scan_information([{"now": now, "frames": {"m": mi}}]))
    assert G.passed(G.scan_information([]))


def test_fill_timing():
    s = _sessions(10)
    dec = pd.DataFrame({"ticker": ["A", "B"], "decided_at": [s[1], s[4]]})
    good = pd.DataFrame({"ticker": ["A", "B"], "decided_at": [s[1], s[4]], "filled_at": [s[2], s[5]], "price_kind": "open"})
    assert G.passed(G.check_fill_timing(dec, good, s))
    same_day = good.copy(); same_day["filled_at"] = [s[1], s[5]]
    assert any("expected" in f.message for f in G.check_fill_timing(dec, same_day, s))
    close = good.copy(); close["price_kind"] = ["close", "open"]
    assert any("must fill" in f.message or "at the open" in f.message for f in G.check_fill_timing(dec, close, s))
    wk = good.copy(); wk["filled_at"] = [s[2], pd.Timestamp("2101-01-16")]
    assert not G.passed(G.check_fill_timing(dec, wk, s))
    assert not G.passed(G.check_fill_timing(dec, good.iloc[:0], s))
    assert G.passed(G.check_fill_timing(dec.iloc[:0], good.iloc[:0], s))
    last = pd.DataFrame({"ticker": ["A"], "decided_at": [s[-1]], "filled_at": [s[-1]], "price_kind": "open"})
    assert not G.passed(G.check_fill_timing(last, last, s))
    assert G.check_fill_timing(dec, good.iloc[:1], s)[0].severity == "warn"


def _panel(n=40):
    idx = pd.MultiIndex.from_product([pd.bdate_range("2101-01-04", periods=n), ["S1", "S2"]], names=["date", "t"])
    return pd.DataFrame(np.random.default_rng(3).normal(size=(len(idx), 2)), index=idx, columns=["a", "b"])


def test_probe_passes_honest_and_catches_peeking():
    P, now = _panel(), pd.Timestamp("2101-02-05")

    def honest(d):
        return G.truncate(d, now).groupby(level=1).mean()

    def peeks(d):                                   # uses the last row of ALL data: a look-ahead
        return d.groupby(level=1).tail(1)

    def leaks_via_norm(d):                          # cleverer: normalises with a full-sample statistic
        return G.truncate(d, now)["a"] - d["a"].mean()

    assert G.passed(G.lookahead_probe(honest, P, now))
    assert not G.passed(G.lookahead_probe(peeks, P, now))
    assert not G.passed(G.lookahead_probe(leaks_via_norm, P, now))
    s = P["a"]
    assert G.passed(G.lookahead_probe(lambda d: G.truncate(d, now).sum(), s, now))
    assert not G.passed(G.lookahead_probe(lambda d: d.iloc[-1], s, now))
    assert not G.passed(G.lookahead_probe(honest, P, pd.Timestamp("2200-01-01")))   # nothing to scramble = fail


def test_truncate_view():
    P = _panel()
    t = G.truncate(P, "2101-01-08")
    assert t.index.get_level_values(0).max() == pd.Timestamp("2101-01-07")  # the 8th is a Saturday
    assert len(G.truncate(P["a"], "2000-01-01")) == 0


def test_audit_run_and_report(tmp_path):
    s = _sessions(10)
    seal = G.seal_window([], 4, "2026-01-01")
    tick = [f"K{i}" for i in range(35)]
    m = G.make_ticker_map(tick, seal["shift_days"])
    px = pd.DataFrame(1.0, index=s, columns=list(m.values()))
    dec = pd.DataFrame({"ticker": ["S0001"], "decided_at": [s[1]]})
    fills = pd.DataFrame({"ticker": ["S0001"], "decided_at": [s[1]], "filled_at": [s[2]], "price_kind": "open"})
    rec = [{"now": s[3], "fields": {"close": s[3]}}]
    f = G.audit_run(seal, s, rec, dec, fills, m, px, seal["shift_days"], first_worker_start="2026-02-01")
    assert G.passed(f), [str(x) for x in f]
    bad = G.audit_run(seal, s, [{"now": s[3], "fields": {"close": s[5]}}], dec, fills, m, px, seal["shift_days"])
    assert not G.passed(bad)
    p = tmp_path / "r" / "gate.json"
    assert G.save_report(f, p) is True and '"passed": true' in p.read_text()
    assert G.save_report(bad, p) is False


# ---------------- coverage, memory-bank causality, filings, text leaks, calendar
def test_era_and_coverage():
    assert [G.era_of(y) for y in ("1990-01-01", "1999-06-01", "2010-01-01")] == ["pre1997", "1997-2000", "2001+"]
    lop = [f"{y}-01-01" for y in (2016, 2017, 2018, 2019, 2020, 2021, 2022)]
    rep = G.coverage_report(lop)
    assert rep["max_share"] == 1.0 and "1965-1974" in rep["empty_bins"] and rep["per_era"] == {"2001+": 7}
    assert G.check_coverage(lop)[0].severity == "warn"
    spread = [f"{y}-01-01" for y in (1968, 1978, 1988, 1998, 2008, 2018)]
    assert G.check_coverage(spread) == []
    assert G.check_coverage(lop[:3]) == []                      # too few windows to judge
    assert G.coverage_report([])["n"] == 0


def test_memory_bank_causality():
    bank = pd.DataFrame({"arm": ["a", "b"], "real_end": ["1980-12-31", "1990-06-30"]})
    assert G.passed(G.check_memory_bank_causality(bank, "1991-01-01"))
    f = G.check_memory_bank_causality(bank, "1990-01-01")
    assert not G.passed(f) and "1 lessons" in f[0].message
    assert not G.passed(G.check_memory_bank_causality(bank.drop(columns="real_end"), "1990-01-01"))
    assert G.passed(G.check_memory_bank_causality(None, "1990-01-01"))
    assert G.passed(G.check_memory_bank_causality(bank.iloc[:0], "1990-01-01"))


def test_public_release():
    now = pd.Timestamp("2101-03-05 16:00")
    fl = pd.DataFrame({"accepted": pd.to_datetime(["2101-03-05 09:00", "2101-03-05 20:30"])})
    f = G.check_public_release(fl, now)
    assert not G.passed(f) and "1 filings" in f[0].message
    assert G.passed(G.check_public_release(fl.iloc[:1], now))
    tz = fl.copy(); tz["accepted"] = tz["accepted"].dt.tz_localize("UTC")
    assert not G.passed(G.check_public_release(tz, now))
    assert G.passed(G.check_public_release(None, now))


def test_text_leak_scan():
    assert G.passed(G.scan_text_for_leaks("week 12 of the window; S0042 bought", ["AAPL", "KO"], [1987]))
    assert not G.passed(G.scan_text_for_leaks("crash of 1987 hit", ["AAPL"], [1987]))
    assert G.passed(G.scan_text_for_leaks("year 2103 in the sim", ["AAPL"], [1987]))
    assert not G.passed(G.scan_text_for_leaks("held AAPL, MSFT", ["AAPL", "MSFT"], []))
    assert G.passed(G.scan_text_for_leaks("KO is short", ["KO"], [2000]))          # two letters are skipped by design
    assert not G.passed(G.scan_text_for_leaks("in 1987", [], []))                  # no year list = any real-era year


def test_calendar_fingerprint():
    real = pd.bdate_range("1987-01-05", periods=60).delete([10, 11, 30])          # holidays removed
    shifted = real + pd.Timedelta(days=7 * 9000)
    assert G.passed(G.check_calendar(real, shifted))
    tampered = shifted.delete(20)
    assert not G.passed(G.check_calendar(real, tampered))
    assert G.calendar_fingerprint(real)["n"] == 57


# ---------------- ledger
def test_ledger_chain_and_tamper_detection(tmp_path):
    L = G.InformationLedger()
    s = _sessions(5)
    for i, d in enumerate(s):
        L.record(d, {"close": d, "macro": d - pd.Timedelta(days=1), "filing": None}, decision={"n": i})
    assert L.verify() == []
    L.save(tmp_path / "l.json")
    ent = __import__("json").loads((tmp_path / "l.json").read_text())
    assert G.verify_ledger(ent) == []
    edited = __import__("copy").deepcopy(ent); edited[2]["decision"] = {"n": 99}
    assert any("edited" in f.message for f in G.verify_ledger(edited))
    removed = ent[:2] + ent[3:]
    assert any("removed or reordered" in f.message for f in G.verify_ledger(removed))
    swapped = [ent[1], ent[0]] + ent[2:]
    assert not G.passed(G.verify_ledger(swapped))
    assert G.verify_ledger([]) == []


def test_ledger_refuses_future_inputs_and_time_reversal():
    L = G.InformationLedger()
    d = pd.Timestamp("2101-02-01")
    with pytest.raises(G.ClockViolation, match="future"):
        L.record(d, {"earnings": d + pd.Timedelta(days=1)})
    L.record(d, {"close": d})
    with pytest.raises(G.ClockViolation, match="advance"):
        L.record(d, {"close": d})
    # a forged entry with a future input and a valid-looking hash is still caught by verify
    forged = __import__("copy").deepcopy(L.entries)
    forged[0]["inputs"]["x"] = "2101-03-01"
    body = {k: forged[0][k] for k in ("now", "inputs", "decision", "prev")}
    forged[0]["hash"] = __import__("hashlib").sha256(__import__("json").dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    assert any("dated after the session" in f.message for f in G.verify_ledger(forged))


def test_decisions_need_recorded_information():
    L = G.InformationLedger()
    s = _sessions(3)
    for d in s[:2]:
        L.record(d, {"close": d})
    assert G.passed(G.check_decisions_use_recorded_info(s[:2], L.entries))
    assert not G.passed(G.check_decisions_use_recorded_info(s, L.entries))


def test_label_alignment():
    s = _sessions(20)
    ok = G.check_label_alignment([s[0], s[1]], [s[5], s[6]], 5, s)
    assert G.passed(ok)
    early = G.check_label_alignment([s[0], s[1]], [s[3], s[6]], 5, s)
    assert not G.passed(early) and "1 rows" in early[0].message
    assert not G.passed(G.check_label_alignment([s[0]], [pd.Timestamp("2200-01-01")], 5, s))
    assert not G.passed(G.check_label_alignment([s[0]], [], 5, s))
    assert G.passed(G.check_label_alignment([], [], 5, s))
