"""Bible Phase 1 tests: the point-in-time firewall. Every detector is proven against a planted defect, and the clean
case is proven to pass (a check that cannot fail is worthless). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from engine import pit
from engine.pit import (AuditLog, Calendar, FailClosed, FillTimingError, Guard, IntegrityError, LabelLeakError,
                        LookAheadError, PITStore, SurvivorshipError)

SESS = pd.bdate_range("2020-01-01", "2021-06-30")
SESS = SESS[~SESS.isin(pd.to_datetime(["2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25"]))]
CAL = Calendar(SESS)
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def make_prices(seed=0):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(SESS), 4)), axis=0)), index=SESS, columns=TICKERS)
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    return {"Close": close, "Open": opn}


def make_store(seed=0):
    rng = np.random.default_rng(seed)
    px = make_prices(seed)
    st = PITStore(CAL).add_prices("prices", px)
    ev = pd.DataFrame({"ticker": ["AAA", "BBB", "AAA", "CCC"],
                       "date": pd.to_datetime(["2020-03-04", "2020-03-04", "2020-06-10", "2020-09-15"]),
                       "timing": ["bmo", "amc", "amc", "bmo"], "surprise": [0.1, -0.2, 0.05, 0.3]})
    st.add_events("earnings", ev)
    ins = pd.DataFrame({"ticker": ["AAA", "BBB"], "trade_date": pd.to_datetime(["2020-03-02", "2020-03-03"]),
                        "filing_date": pd.to_datetime(["2020-03-04", "2020-03-05"]), "value": [1e5, 2e5]})
    st.add_insider("insider", ins)
    fund = pd.DataFrame({"ticker": ["AAA"] * 3, "period_end": pd.to_datetime(["2019-12-31"] * 2 + ["2020-03-31"]),
                         "filed": pd.to_datetime(["2020-02-20", "2020-06-01", "2020-05-05"]),
                         "eps": [1.00, 0.80, 1.10]})
    st.add_fundamentals("fund", fund)
    mac = pd.DataFrame({"series": ["CPI"] * 4, "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-03-31"]),
                        "value": [1.0, 1.1, 1.2, 1.5]})
    mac["vintage"] = pd.to_datetime(["2020-02-14", "2020-03-13", "2020-04-15", "2020-06-15"])
    st.add_macro("macro", mac, vintage="vintage")
    lst = pd.DataFrame({"ticker": TICKERS, "list_date": pd.Timestamp("2015-01-01"),
                        "delist_date": [pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2020-09-01")], "reason": ["", "", "", "bankrupt"]})
    st.add_listings(lst)
    return st


# ------------------------------------------------------------------ calendar
def test_calendar_shift_and_next_session():
    fri, sat = pd.Timestamp("2020-03-06"), pd.Timestamp("2020-03-07")
    assert CAL.strict_next([fri])[0] == pd.Timestamp("2020-03-09")
    assert CAL.strict_next([sat])[0] == pd.Timestamp("2020-03-09")        # not Tuesday
    assert CAL.on_or_after([sat])[0] == pd.Timestamp("2020-03-09")
    assert CAL.strict_next([pd.Timestamp("2020-01-17")])[0] == pd.Timestamp("2020-01-21")   # holiday skipped
    assert not CAL.is_session([pd.Timestamp("2020-01-20")])[0]
    assert CAL.shift([pd.Timestamp("2020-03-02")], 5)[0] == pd.Timestamp("2020-03-09")
    assert pd.isna(CAL.shift([pd.NaT], 1)[0])


# ------------------------------------------------------------------ fence
def test_guard_hides_future_and_raises_on_future_request():
    st = make_store()
    g = st.view("2020-06-05")
    df = g.wide("prices", "Close")
    assert df.index.max() == pd.Timestamp("2020-06-05")
    assert len(g.wide("prices", "Close", lookback=5)) == 5
    with pytest.raises(LookAheadError):
        g.wide("prices", "Close", end="2020-06-08")
    with pytest.raises(LookAheadError):
        g.wide("prices", "Close", start="2020-07-01")
    with pytest.raises(LookAheadError):
        g.fence("2020-06-06")
    assert g.fence("2020-06-05") == pd.Timestamp("2020-06-05")
    d = st.log.denied()
    assert len(d) == 3 and all(r["verdict"] == "denied" for r in d)


def test_guard_is_immutable_and_has_no_store_widening():
    g = make_store().view("2020-03-10")
    with pytest.raises(AttributeError):
        g.as_of = pd.Timestamp("2021-01-01")
    assert g.as_of == pd.Timestamp("2020-03-10")


def test_live_fields_only_when_permitted():
    st = make_store()
    g = st.view("2020-03-10", live={"last": 101.5, "secret": 5.0}, live_fields=["last"])
    assert g.live("last") == 101.5
    with pytest.raises(LookAheadError):
        g.live("secret")
    with pytest.raises(LookAheadError):
        g.live("missing")


def test_assert_frame_clean_catches_model_built_future_frames():
    g = make_store().view("2020-03-10")
    ok = pd.DataFrame({"a": 1}, index=pd.bdate_range("2020-03-02", "2020-03-10"))
    g.assert_frame_clean(ok)
    bad = pd.DataFrame({"a": 1}, index=pd.bdate_range("2020-03-02", "2020-03-12"))
    with pytest.raises(LookAheadError):
        g.assert_frame_clean(bad)
    panel = pd.DataFrame({"a": 1.0}, index=pd.MultiIndex.from_product([pd.bdate_range("2020-03-09", "2020-03-11"), ["A"]]))
    with pytest.raises(LookAheadError):
        g.assert_frame_clean(panel)


# ------------------------------------------------------------------ publication lags / events
def test_earnings_timing_bmo_amc_and_insider_filing_dates():
    st = make_store()
    on_day = st.view("2020-03-04")
    e = on_day.records("earnings")
    assert list(e["ticker"]) == ["AAA"]                   # bmo known at that close; BBB (amc) is not
    assert list(st.view("2020-03-05").records("earnings")["ticker"].sort_values()) == ["AAA", "BBB"]
    assert list(on_day.records("insider")["ticker"]) == ["AAA"]          # AAA filed 03-04, BBB filed 03-05
    assert len(st.view("2020-03-03").records("insider")) == 0            # traded 03-02 but not yet filed


def test_macro_vintages_and_restatement_as_of():
    st = make_store()
    before = st.view("2020-04-16").macro_wide("macro")
    assert before.loc["2020-03-31", "CPI"] == 1.2          # first print
    after = st.view("2020-06-16").macro_wide("macro")
    assert after.loc["2020-03-31", "CPI"] == 1.5           # revised value visible only after its release
    assert pd.Timestamp("2020-03-31") not in st.view("2020-04-14").macro_wide("macro").index
    src = st.source("macro")
    assert src.versions("2020-06-16", ["value"]).shape[0] == 1
    assert src.versions("2020-04-16", ["value"]).shape[0] == 0


def test_fundamental_restatement_exposure_quantifies_naive_leak():
    st = make_store()
    src = st.source("fund")
    g = st.view("2020-03-01")
    f = g.records("fund", latest=True)
    assert len(f) == 1 and f["eps"].iloc[0] == 1.00                      # original filing, not the June restatement
    exp = src.restatement_exposure("2020-03-01", ["eps"])
    assert exp["restated"] == 1 and exp["mean_abs_rel_diff"] == pytest.approx(0.25)
    assert st.view("2020-06-02").records("fund", latest=True).query("period_end < '2020-01-01'")["eps"].iloc[0] == 0.80


def test_macro_lag_must_be_declared_per_series():
    st = PITStore(CAL)
    df = pd.DataFrame({"series": ["A", "B"], "date": pd.to_datetime(["2020-03-02"] * 2), "value": [1, 2]})
    with pytest.raises(IntegrityError):
        st.add_macro("m", df, lag={"A": 1})
    st.add_macro("m", df, lag={"A": 1, "B": 3})
    assert list(st.view("2020-03-03").records("m")["series"]) == ["A"]


# ------------------------------------------------------------------ integrity validation
def test_validate_clean_store_ok():
    rep = make_store().validate()
    assert rep.ok, rep.summary()


def test_validate_flags_planted_defects():
    st = PITStore(CAL)
    fund = pd.DataFrame({"ticker": ["A", "A", "B"], "period_end": pd.to_datetime(["2020-03-31"] * 3),
                         "filed": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-01"]), "eps": [1, 2, 3]})
    st.add_fundamentals("f", fund)
    ev = pd.DataFrame({"ticker": ["A"], "date": pd.to_datetime(["2020-03-02"])})
    st.add_events("e", ev)
    dup = make_prices()["Close"].iloc[[0, 1, 1, 2]]
    st.add_prices("p", dup)
    rep = st.validate()
    codes = rep.error_codes()
    assert {"available_before_effective", "ambiguous_vintage", "no_publication_lag", "duplicate_dates"} <= codes
    with pytest.raises(pit.PITError):
        rep.require()


def test_validate_nat_availability_is_error_and_row_never_visible():
    st = PITStore(CAL)
    ins = pd.DataFrame({"ticker": ["A", "B"], "trade_date": pd.to_datetime(["2020-03-02", "2020-03-02"]),
                        "filing_date": [pd.Timestamp("2020-03-04"), pd.NaT], "v": [1, 2]})
    st.add_insider("i", ins)
    assert "missing_availability" in st.validate().error_codes()
    assert list(st.view("2021-06-01").records("i")["ticker"]) == ["A"]


# ------------------------------------------------------------------ labels / features
def test_label_close_dates_and_training_verification():
    d = pd.DatetimeIndex(["2020-03-02", "2020-03-06"])
    lc = pit.label_close_dates(d, 5, CAL)
    assert list(lc) == [pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-13")]
    pit.verify_training_rows(d, lc, "2020-03-13")
    with pytest.raises(LabelLeakError):
        pit.verify_training_rows(d, lc, "2020-03-12")            # second label not closed yet
    with pytest.raises(LabelLeakError):
        pit.verify_training_rows(d, pd.DatetimeIndex([pd.NaT, lc[1]]), "2020-12-31")
    with pytest.raises(LabelLeakError):
        pit.verify_training_rows(d, d - pd.Timedelta(days=1), "2020-12-31")
    with pytest.raises(ValueError):
        pit.label_close_dates(d, 5, CAL, entry_lag=0)            # same-close entry is not allowed


def test_purged_training_set_drops_rows_whose_labels_are_still_open():
    dates = SESS[:30]
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    as_of = dates[20]
    Xs, ys = pit.purged_training_set(X, y, as_of, horizon=5, calendar=CAL)
    assert Xs.index.get_level_values(0).max() == dates[15]       # label of dates[15] closes on dates[20]
    Xe, _ = pit.purged_training_set(X, y, as_of, horizon=5, calendar=CAL, embargo=2)
    assert Xe.index.get_level_values(0).max() == dates[13]
    Xn, _ = pit.purged_training_set(X, y, dates[2], horizon=5, calendar=CAL)
    assert len(Xn) == 0                                            # degenerate: nothing has closed


def test_verify_feature_availability():
    rd = pd.DatetimeIndex(["2020-03-04", "2020-03-05"])
    pit.verify_feature_availability(rd, pd.DataFrame({"f": pd.to_datetime(["2020-03-04", "2020-03-02"])}))
    with pytest.raises(LookAheadError):
        pit.verify_feature_availability(rd, pd.DataFrame({"f": pd.to_datetime(["2020-03-04", "2020-03-06"])}))
    with pytest.raises(LookAheadError):
        pit.verify_feature_availability(rd, pd.Series(pd.to_datetime([pd.NaT, "2020-03-01"])))


# ------------------------------------------------------------------ survivorship
def _surv_setup(n=80, drop=True, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    names = [f"S{i:03d}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), n)), axis=0)), index=dates, columns=names)
    dl = pd.Series(pd.NaT, index=names, dtype="datetime64[ns]")
    dead = names[: n // 6]
    for i, t in enumerate(dead):
        dl[t] = pd.Timestamp("2016-06-01") + pd.Timedelta(days=200 * i)
    lst = pd.DataFrame({"ticker": names, "list_date": pd.Timestamp("2010-01-01"), "delist_date": dl.values})
    if drop:
        for t in dead:
            px.loc[px.index > dl[t], t] = np.nan
    return px, pit.Listings(lst), dead, dl


def test_survivorship_clean_panel_passes():
    px, lst, _, _ = _surv_setup()
    rep = pit.survivorship_report(px, lst, "2020-12-31")
    assert rep.ok, rep.summary()
    assert rep.stats["attrition_annual"] > 0.015


def test_survivorship_catches_survivor_only_panel_and_ghosts_and_truncation():
    px, lst, dead, dl = _surv_setup()
    only = px.drop(columns=dead)
    lst_only = pit.Listings(lst.df[~lst.df["ticker"].isin(dead)])
    rep = pit.survivorship_report(only, lst_only, "2020-12-31")
    assert "survivor_only_panel" in rep.error_codes()
    rep2 = pit.survivorship_report(only, lst, "2020-12-31")
    assert "missing_history" in rep2.error_codes()
    ghost = px.copy()
    ghost[dead[0]] = ghost[dead[1]].fillna(100.0).values           # bars continue after the delisting
    assert "ghost_prices" in pit.survivorship_report(ghost, lst, "2020-12-31").error_codes()
    trunc = px.copy()
    trunc.loc[trunc.index > dl[dead[0]] - pd.Timedelta(days=90), dead[0]] = np.nan
    assert "truncated_history" in pit.survivorship_report(trunc, lst, "2020-12-31").error_codes()
    with pytest.raises(SurvivorshipError):
        rep.require(SurvivorshipError)


def test_membership_does_not_know_future_delistings():
    st = make_store()
    assert "DDD" in st.view("2020-08-31").members()
    assert "DDD" not in st.view("2020-09-01").members()
    lst = st.listings
    assert len(lst.known_delisted("2020-08-31")) == 0 and len(lst.known_delisted("2020-09-30")) == 1


# ------------------------------------------------------------------ fills
def _fills_fixture():
    px = make_prices()
    dd = [pd.Timestamp("2020-03-06"), pd.Timestamp("2020-03-13")]          # both Fridays
    dec = pd.DataFrame({"order_id": ["o1", "o2"], "ticker": ["AAA", "BBB"], "decision_date": dd})
    fills = pd.DataFrame({"order_id": ["o1", "o2"], "ticker": ["AAA", "BBB"],
                          "fill_date": [pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-16")],
                          "fill_price": [px["Open"].at["2020-03-09", "AAA"], px["Open"].at["2020-03-16", "BBB"]]})
    return px, dec, fills


def test_audit_fills_clean():
    px, dec, fills = _fills_fixture()
    rep = pit.audit_fills(dec, fills, px["Open"], px["Close"], CAL)
    assert rep.ok, rep.summary()


@pytest.mark.parametrize("mut,code", [
    (lambda d, f, px: f.__setitem__("fill_date", [pd.Timestamp("2020-03-06"), f.fill_date[1]]), "same_close_fill"),
    (lambda d, f, px: f.__setitem__("fill_date", [pd.Timestamp("2020-03-07"), f.fill_date[1]]), "fill_not_session"),
    (lambda d, f, px: f.__setitem__("fill_price", [px["Close"].at["2020-03-06", "AAA"], f.fill_price[1]]), "same_close_price"),
    (lambda d, f, px: f.__setitem__("fill_price", [f.fill_price[0] * 1.03, f.fill_price[1]]), "not_open_price"),
    (lambda d, f, px: f.__setitem__("order_id", ["o1", "ghost"]), "fill_without_decision"),
    (lambda d, f, px: d.__setitem__("decision_time", [15.5, 16.5]), "decision_before_close"),
    (lambda d, f, px: d.__setitem__("inputs_max_avail", [pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-13")]),
     "next_day_information"),
    (lambda d, f, px: d.__setitem__("decision_date", [pd.Timestamp("2020-03-07"), d.decision_date[1]]), "decision_not_session"),
])
def test_audit_fills_catches_planted_violations(mut, code):
    px, dec, fills = _fills_fixture()
    mut(dec, fills, px)
    rep = pit.audit_fills(dec, fills, px["Open"], px["Close"], CAL)
    assert code in rep.error_codes(), rep.summary()


def test_audit_fills_late_fill_is_warning_and_empty_ok():
    px, dec, fills = _fills_fixture()
    fills.loc[0, "fill_date"] = pd.Timestamp("2020-03-10")
    fills.loc[0, "fill_price"] = px["Open"].at["2020-03-10", "AAA"]
    rep = pit.audit_fills(dec, fills, px["Open"], px["Close"], CAL)
    assert rep.ok and "late_fill" in rep.codes()
    empty = pit.audit_fills(dec.iloc[:0], fills.iloc[:0], px["Open"], None, CAL)
    assert empty.ok and empty.stats["fills"] == 0


def test_executor_fills_only_at_next_open_and_refuses_weekends():
    st = make_store()
    ex = st.executor("2020-03-06")
    r = ex.fill_next_open("AAA", 10, order_id="x")
    assert r["fill_date"] == pd.Timestamp("2020-03-09")
    assert r["fill_price"] == st.source("prices").frames["Open"].at["2020-03-09", "AAA"]
    assert r["fill_price"] != st.source("prices").frames["Close"].at["2020-03-06", "AAA"]
    with pytest.raises(FillTimingError):
        st.executor("2020-03-07")                                   # Saturday decision
    with pytest.raises(FillTimingError):
        ex.fill_next_open("ZZZ", 1)                                 # no data -> no fill
    with pytest.raises(FillTimingError):
        st.executor(SESS[-1]).fill_next_open("AAA", 1)              # nothing after the last bar
    # fills are exempt from the leak tripwire but every one is logged
    assert not st.log.leaks() and sum(1 for x in st.log.records if x["op"] == "fill") == 1
    dec = pd.DataFrame({"order_id": ["x"], "ticker": ["AAA"], "decision_date": [pd.Timestamp("2020-03-06")]})
    px = st.source("prices").frames
    assert pit.audit_fills(dec, pd.DataFrame([r]), px["Open"], px["Close"], CAL).ok


# ------------------------------------------------------------------ audit log
def test_audit_log_chain_detects_tampering_and_records_every_access():
    st = make_store()
    g = st.view("2020-06-05")
    g.wide("prices", "Close", lookback=3)
    g.records("earnings")
    g.members()
    try:
        g.wide("prices", end="2021-01-01")
    except LookAheadError:
        pass
    log = st.log
    assert len(log) == 4 and log.verify() and len(log.denied()) == 1 and not log.leaks()
    assert log.sources_touched() == {"prices", "earnings", "listings"}
    assert log.max_avail_touched("2020-06-05") <= pd.Timestamp("2020-06-05")
    log.records[1]["rows"] = 999
    assert not log.verify()
    rep = pit.audit_log_against_decisions(log, pd.DataFrame({"decision_date": ["2020-06-05", "2020-07-01"]}))
    assert "log_tampered" in rep.error_codes() and "decision_without_access" in rep.codes()


def test_leak_tripwire_fires_when_a_filter_is_broken(monkeypatch):
    st = make_store()
    src = st.source("insider")
    monkeypatch.setattr(type(src), "_visible", lambda self, as_of: self.df)     # a broken filter returns everything
    with pytest.raises(LookAheadError):
        st.view("2020-03-03").records("insider")          # traded 03-02 (<= as_of) but filed 03-04
    assert st.log.leaks()                                                        # and the log shows it independently


# ------------------------------------------------------------------ feature invariance
def _panel_frame(seed=3, n=160):
    rng = np.random.default_rng(seed)
    idx = SESS[:n]
    return pd.DataFrame(rng.normal(0, 1, (n, 3)).cumsum(0) + 50, index=idx, columns=list("abc"))


def test_invariance_passes_for_trailing_features():
    df = _panel_frame()
    rep = pit.future_invariance(lambda d: d.rolling(10).mean().pct_change(3).ewm(span=5).mean(), df)
    assert rep.passed and rep.rows_checked > 100, rep.summary()
    rep.require()


@pytest.mark.parametrize("name,fn", [
    ("negative_shift", lambda d: d.shift(-1)),
    ("centered_window", lambda d: d.rolling(5, center=True).mean()),
    ("full_sample_zscore", lambda d: (d - d.mean()) / d.std()),
    ("forward_fill_from_future", lambda d: d.bfill()),
    ("expanding_rank_full", lambda d: d.rank(pct=True)),
])
def test_invariance_catches_planted_lookahead(name, fn):
    df = _panel_frame()
    df = df.mask(np.random.default_rng(5).random(df.shape) < 0.25)       # gives bfill something to peek at
    rep = pit.future_invariance(fn, df)
    assert not rep.passed, name
    with pytest.raises(FailClosed):
        rep.require()


def test_invariance_works_on_dict_input_and_panel_output_and_degenerate():
    df = _panel_frame()
    frames = {"c": df, "v": df * 2}
    ok = pit.future_invariance(lambda f: f["c"].pct_change(2) / (f["v"] + 1), frames)
    assert ok.passed
    bad = pit.future_invariance(lambda f: f["c"] - f["v"].mean(), frames)
    assert not bad.passed
    tiny = pit.future_invariance(lambda d: d.shift(-1), df.iloc[:2])
    assert tiny.passed and tiny.rows_checked == 0                       # too short to say anything
    panel = df.stack().rename("x").to_frame()
    panel.index.names = ["date", "ticker"]
    lag = pit.future_invariance(lambda d: d.groupby(level=1).shift(1), panel)
    assert lag.passed
    lead = pit.future_invariance(lambda d: d.groupby(level=1).shift(-1), panel)
    assert not lead.passed


# ------------------------------------------------------------------ pipeline scramble
def _guarded_pipeline(g: Guard):
    close = g.wide("prices", "Close")
    ret = close.pct_change(5).iloc[-1]
    ev = g.records("earnings")
    mac = g.macro_wide("macro")
    return {"scores": ret.rank(), "n_events": len(ev), "mem": pit.fingerprint(ret.values),
            "macro": mac.iloc[-1] if len(mac) else None, "members": g.members()}


def test_guarded_pipeline_survives_store_scramble():
    st = make_store()
    rep = pit.future_scramble_store(_guarded_pipeline, st, "2020-06-05", seed=4)
    assert rep.passed, rep.summary()
    rep.require()


def test_pipeline_that_reads_the_future_fails_closed():
    st = make_store()

    def leaky(g):
        out = _guarded_pipeline(g)
        raw = g._store.source("prices").frames["Close"]            # reaches around the fence
        out["peek"] = float(raw.iloc[-1, 0])
        return out
    rep = pit.future_scramble_store(leaky, st, "2020-06-05", seed=4)
    assert not rep.passed
    with pytest.raises(FailClosed):
        rep.require()


def test_pipeline_that_crashes_only_under_scramble_is_a_failure():
    st = make_store()
    real_len = len(st.source("prices").frames["Close"])

    def fragile(g):
        if len(g._store.source("prices").frames["Close"]) != real_len:
            raise RuntimeError("shape changed")
        return {"x": 1}
    rep = pit.future_scramble_store(fragile, st, "2020-06-05", seed=1)
    assert not rep.passed and "scrambled" in rep.errors


def test_frame_pipeline_scramble_truncate_and_scramble_variants():
    px = make_prices()
    frames = {"close": px["Close"], "ev": pd.DataFrame({"date": pd.to_datetime(["2020-03-04", "2020-08-01"]),
                                                        "x": [1.0, 2.0]})}
    as_of = pd.Timestamp("2020-06-05")

    def honest(fr, t):
        c = fr["close"]
        return {"m": c[c.index <= t].pct_change(5).iloc[-1]}

    def leaky(fr, t):
        c = fr["close"]
        return {"m": c.pct_change(5).iloc[-1]}               # forgot to clip at t

    assert pit.future_scramble(honest, frames, as_of, seed=2).passed
    r = pit.future_scramble(leaky, frames, as_of, seed=2)
    assert not r.passed and set(r.variants) == {"truncated", "scrambled"}


def test_compare_detects_small_and_structural_differences():
    a = {"df": pd.DataFrame({"x": [1.0, 2.0]}), "arr": np.array([1.0, 2.0]), "s": "k", "l": [1, 2]}
    b = {"df": pd.DataFrame({"x": [1.0, 2.0000001]}), "arr": np.array([1.0, 2.1]), "s": "k", "l": [1, 3]}
    diffs = pit._compare(a, b)
    assert any("arr" in d for d in diffs) and any("l" in d for d in diffs) and not any("df" in d for d in diffs)
    assert pit._compare({"a": 1}, {"b": 1})
    assert pit.fingerprint(a) == pit.fingerprint(dict(a)) and pit.fingerprint(a) != pit.fingerprint(b)


def test_full_check_gate():
    st = make_store()
    rep = pit.full_check(st, "2020-06-05", _guarded_pipeline)
    assert rep.ok, rep.summary()
    st2 = make_store()
    st2.log.records.append({"op": "wide", "source": "prices", "as_of": "2020-01-10", "verdict": "ok", "max_avail": "2020-02-01"})
    st2.log._digests.append(st2.log._digest("", st2.log.records[-1]))
    assert not pit.full_check(st2, "2020-06-05", _guarded_pipeline).ok


# ------------------------------------------------------------------ label-knowing features
def _ic_panel(seed=7, dates=60, names=40):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=dates), range(names)], names=["date", "ticker"])
    X = pd.DataFrame({"honest": rng.normal(size=len(idx)), "noise": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(rng.normal(size=len(idx)), index=idx) + 0.05 * X["honest"]
    X["leak"] = y + rng.normal(0, 0.5, len(idx))                   # contains the label
    return X, y


def test_implausible_ic_flags_only_the_leaking_feature():
    X, y = _ic_panel()
    r = pit.implausible_ic(X, y)
    assert r.loc["leak", "flag"] and not r.loc["honest", "flag"] and not r.loc["noise", "flag"]
    assert r.loc["leak", "mean_ic"] > 0.6


def test_implausible_ic_degenerate_inputs():
    X, y = _ic_panel(dates=5, names=5)                             # too few names per date
    r = pit.implausible_ic(X, y)
    assert not r["flag"].any()
    const = X.copy()
    const["noise"] = 1.0
    assert not pit.implausible_ic(const, y).loc["noise", "flag"]


# ------------------------------------------------------------------ empty / degenerate
def test_empty_store_and_as_of_before_all_data():
    st = PITStore()
    assert st.validate().ok and st.names() == []
    with pytest.raises(KeyError):
        st.view("2020-01-01").wide("nope")
    full = make_store()
    early = full.view("2019-01-01")
    assert early.wide("prices", "Close").empty
    assert early.records("earnings").empty and early.macro_wide("macro").empty
    assert early.last_row("prices", "Close").empty
    assert early.members() == TICKERS
    e = PITStore(CAL).add_events("e", pd.DataFrame({"ticker": [], "date": pd.to_datetime([])}))
    assert e.view("2020-06-01").records("e").empty and e.validate().ok


def test_wrong_source_type_and_duplicate_registration_rejected():
    st = make_store()
    g = st.view("2020-06-05")
    with pytest.raises(TypeError):
        g.records("prices")
    with pytest.raises(TypeError):
        g.wide("earnings")
    with pytest.raises(IntegrityError):
        st.add_events("earnings", pd.DataFrame({"ticker": ["A"], "date": pd.to_datetime(["2020-01-01"])}))
    with pytest.raises(KeyError):
        g.wide("prices", "Bogus")
    assert pit._ts("2020-03-04 15:30:00-05:00") == pd.Timestamp("2020-03-04")


def test_scrambled_store_changes_only_the_future():
    st = make_store()
    sc = st.scrambled("2020-06-05", seed=9)
    a = st.view("2020-06-05").wide("prices", "Close")
    b = sc.view("2020-06-05").wide("prices", "Close")
    assert a.equals(b)
    fa = st.source("prices").frames["Close"].loc["2020-06-08":]
    fb = sc.source("prices").frames["Close"].loc["2020-06-08":fa.index[-1]]
    assert not np.allclose(fa.values, fb.values)                      # the future really was destroyed
    assert len(sc.source("prices").frames["Close"]) == len(st.source("prices").frames["Close"]) + 3
    e1 = st.view("2020-06-05").records("earnings")
    e2 = sc.view("2020-06-05").records("earnings")
    assert e1.reset_index(drop=True).equals(e2.reset_index(drop=True))


# ------------------------------------------------------------------ diagnostics used on real caches
def test_lag_profile_by_group_finds_planted_negative_and_late_filers():
    eff = pd.to_datetime(["2020-03-02"] * 6 + ["2020-03-04"] * 2)
    av = pd.to_datetime(["2020-03-04", "2020-03-04", "2020-03-05", "2020-03-20", "2020-03-01", "2020-03-04",
                         "2020-03-05", "2020-04-30"])
    grp = ["ok", "ok", "ok", "late", "impossible", "ok", "ok", "late"]
    p = pit.lag_profile(eff, av, CAL, by=grp)
    assert p.loc["impossible", "negative"] == 1 and p.loc["ok", "negative"] == 0
    assert p.loc["late", "late_share"] == 1.0 and p.loc["ok", "late_share"] == pytest.approx(0.2)
    assert p.loc["ok", "median"] == 2.0 and p.loc["late", "p95"] > 15
    assert p["n"].sum() == 8
    assert pit.lag_profile(pd.to_datetime([]), pd.to_datetime([]), CAL).empty
    with pytest.raises(ValueError):
        pit.lag_profile(eff, av[:3], CAL)
    nat = pit.lag_profile(pd.to_datetime(["2020-03-02", pd.NaT]), pd.to_datetime(["2020-03-03", "2020-03-03"]), CAL)
    assert nat["n"].iloc[0] == 1


def test_visibility_gap_flags_a_system_that_sees_things_early():
    pit_v = pd.to_datetime(["2020-03-10", "2020-03-10", "2020-03-10", "2020-04-01"])
    sys_v = pd.to_datetime(["2020-03-10", "2020-03-04", "2020-03-12", "2020-03-25"])
    g = pit.visibility_gap(sys_v, pit_v, by=["a", "a", "a", "b"], calendar=CAL)
    assert g.loc["a", "early"] == 1 and g.loc["a", "late"] == 1 and g.loc["a", "early_sessions_mean"] == 4.0
    assert g.loc["b", "early"] == 1 and g.loc["b", "early_days_max"] == 7
    assert pit.visibility_gap(pit_v, pit_v)["early"].sum() == 0
    assert pit.visibility_gap(pd.to_datetime([]), pd.to_datetime([])).empty


def test_entry_gap_profile_measures_the_overnight_return_a_close_entry_ignores():
    px = make_prices()
    c = px["Close"]
    none = pit.entry_gap_profile(c.shift(1).fillna(c), c, by_year=False)     # next open == this close
    assert none["mean_abs"].iloc[0] < 1e-12
    gapped = c.shift(1).fillna(c).copy()
    gapped.iloc[1::2] *= 1.03                                     # every other next-open gaps +3%
    prof = pit.entry_gap_profile(gapped, c)
    assert (prof["share_abs_gt_1pct"] > 0.3).all() and (prof["mean_abs"] > 0.01).all()
    assert pit.entry_gap_profile(c.iloc[:0], c.iloc[:0]).empty


def test_invariance_with_long_frames_needs_mask_fns_and_catches_event_peek():
    idx = SESS[:120]
    px = pd.DataFrame(np.random.default_rng(2).normal(0, 1, (120, 2)).cumsum(0) + 60, index=idx, columns=["A", "B"])
    ev = pd.DataFrame({"ticker": ["A", "B", "A"], "when": [idx[30], idx[60], idx[100]], "v": [1.0, 2.0, 3.0]})
    masks = {"ev": lambda d, t: pd.to_datetime(d["when"]) > t}

    def honest(d):
        flag = pd.DataFrame(0.0, index=d["px"].index, columns=d["px"].columns)
        for _, r in d["ev"].iterrows():
            flag.loc[flag.index >= r["when"], r["ticker"]] += r["v"]
        return flag

    def peeks(d):
        return honest(d) * 0 + d["ev"]["v"].sum()                # uses ALL events, including unfiled ones

    data = {"px": px, "ev": ev}
    assert pit.future_invariance(honest, data, mask_fns=masks).passed
    assert not pit.future_invariance(peeks, data, mask_fns=masks).passed
    with pytest.raises(ValueError):
        pit.future_invariance(honest, data)                      # long frame with no mask: refuses to guess


def test_audit_fills_checks_decisions_even_when_nothing_filled():
    dec = pd.DataFrame({"order_id": ["a", "b"], "ticker": ["AAA", "BBB"],
                        "decision_date": pd.to_datetime(["2020-03-06", "2020-03-07"]), "decision_time": [15.7, 16.2]})
    rep = pit.audit_fills(dec, pd.DataFrame(columns=["order_id", "ticker", "fill_date", "fill_price"]),
                          make_prices()["Open"], None, CAL)
    assert {"decision_before_close", "decision_not_session"} <= rep.error_codes()
    assert "unfilled_decision" in rep.codes()


def test_scramble_keeps_positive_columns_positive_and_keeps_tz_dtype():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"px": np.linspace(5, 50, 40), "chg": np.linspace(-1, 1, 40),
                       "t": pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")}, index=SESS[:40])
    out = pit.scramble_after(df, SESS[19], rng)
    assert (out["px"] > 0).all() and str(out["t"].dtype) == str(df["t"].dtype)
    assert out.iloc[:20].equals(df.iloc[:20]) and not np.allclose(out["px"].iloc[20:], df["px"].iloc[20:])
