"""Data expansion (Phase 27): validation gate, sector ETF adapter, backup reconciliation, delisted registry,
incremental information. Synthetic data and fake fetchers only."""
import numpy as np
import pandas as pd
import pytest

from engine import data_sources as D

AS_OF = "2024-06-28"


def make_prices(tickers=("XLK", "XLF"), n=60, seed=0, start="2024-03-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    rows = []
    for t in tickers:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for d, x in zip(days, c):
            rows.append((d, t, x * 0.999, x * 1.004, x * 0.996, x, 1e6))
    return pd.DataFrame(rows, columns=D.PRICE_COLS)


def test_clean_frame_passes_with_provenance():
    df, rep = D.validate_prices(make_prices(), "fake", AS_OF, fetched_at="2024-06-28T20:00")
    assert rep.ok and rep.rows_out == 120 and not rep.dropped
    p = rep.provenance
    assert p["source"] == "fake" and p["tickers"] == 2 and len(p["content_hash"]) == 16
    # hash is order independent and content sensitive
    assert D.frame_hash(df.sample(frac=1, random_state=1)) == p["content_hash"]
    assert D.frame_hash(df.assign(close=df["close"] * 1.0001)) != p["content_hash"]


def test_schema_and_empty_refused():
    _, r = D.validate_prices(make_prices().drop(columns=["volume"]), "x", AS_OF)
    assert not r.ok and "missing columns" in r.errors[0]
    _, r = D.validate_prices(make_prices().iloc[0:0], "x", AS_OF)
    assert r.errors == ["empty frame"]


def test_planted_defects_each_caught_and_counted():
    df = make_prices(n=40)
    bad = df.copy()
    bad["date"] = bad["date"].astype(object)
    bad.loc[0, "date"] = pd.Timestamp("2024-03-02")                 # Saturday
    bad.loc[1, "close"] = np.nan
    bad.loc[2, ["open", "high", "low", "close"]] = [10, 5, 4, 8]    # high below open
    bad.loc[3, "close"] = -1.0
    bad.loc[4, "date"] = "not a date"
    bad = pd.concat([bad, bad.iloc[[10]]])                          # exact duplicate
    late = df.iloc[[20]].copy()
    late["date"] = pd.Timestamp("2024-07-05")                       # after as_of
    out, rep = D.validate_prices(pd.concat([bad, late]), "x", AS_OF)
    assert rep.ok
    assert rep.dropped["weekend_date"] == 1 and rep.dropped["missing_close"] == 1
    assert rep.dropped["ohlc_inconsistent"] == 1 and rep.dropped["non_positive_price"] == 1
    assert rep.dropped["unparseable_date"] == 1 and rep.dropped["exact_duplicate"] == 1
    assert rep.dropped["after_as_of"] == 1
    assert out["date"].max() <= pd.Timestamp(AS_OF) and not out.duplicated(["date", "ticker"]).any()


def test_conflicting_duplicate_is_an_error():
    df = make_prices(n=10)
    dup = df.iloc[[3]].copy()
    dup["close"] = dup["close"] * 1.01
    dup["high"] = dup["high"] * 1.02
    out, rep = D.validate_prices(pd.concat([df, dup]), "x", AS_OF)
    assert not rep.ok and out.empty and "conflict" in rep.errors[0]


def test_gap_and_split_warnings():
    df = make_prices(("AAA",), n=30)
    df = df[~df["date"].between("2024-03-08", "2024-03-20")]          # 12-day hole
    df.loc[df["date"] >= "2024-03-25", ["open", "high", "low", "close"]] /= 4      # unadjusted 4:1 split
    _, rep = D.validate_prices(df, "x", AS_OF)
    assert any("gaps" in w for w in rep.warnings) and any("unadjusted split" in w for w in rep.warnings)


def test_etf_inception_enforced():
    fake = lambda tk, s, e: make_prices(("XLC", "XLK"), n=40, start="2018-05-01")
    df, rep = D.SectorETFAdapter(fake).load("2018-05-01", "2018-06-30", "2018-06-30")
    assert rep.dropped.get("before_inception:XLC", 0) > 0
    assert df[df["ticker"] == "XLC"]["date"].min() >= D.XLC_START


def test_adapter_never_requests_past_as_of():
    seen = {}

    def fake(tk, s, e):
        seen["end"] = e
        return make_prices(("XLK",), n=10)
    D.SectorETFAdapter(fake, ["XLK"]).load("2024-03-01", "2025-01-01", AS_OF)
    assert seen["end"] == AS_OF


def test_adapter_reports_absent_tickers():
    fake = lambda tk, s, e: make_prices(("XLK",), n=10)
    _, rep = D.SectorETFAdapter(fake, ["XLK", "XLE"]).load("2024-03-01", AS_OF, AS_OF)
    assert any("XLE" in w for w in rep.warnings)


def test_sector_relative_is_trailing_only():
    df = make_prices(("XLK",), n=80)
    bench = D.to_wide(make_prices(("SPY",), n=80, seed=5))["SPY"]
    full = D.sector_relative(df, bench)
    cut = D.sector_relative(df[df["date"] <= full.index[30]], bench)
    pd.testing.assert_series_equal(full["XLK"].loc[cut.index], cut["XLK"])      # no look-ahead: same values


# ---- reconciliation
def test_reconcile_agreeing_sources():
    p = D.validate_prices(make_prices(), "p", AS_OF)[0]
    b = p.copy()
    b["close"] *= 1 + np.random.default_rng(1).normal(0, 0.0005, len(b))
    rec = D.reconcile(p, b)
    assert rec["coverage"] == 1.0 and rec["share_within_tol"] == 1.0
    assert rec["split_suspects"] == [] and len(rec["disagreements"]) == 0


def test_reconcile_finds_planted_split_mismatch_and_bad_prints():
    p = D.validate_prices(make_prices(("AAA", "BBB"), n=40), "p", AS_OF)[0]
    b = p.copy()
    b.loc[b["ticker"] == "AAA", "close"] *= 2.0                       # backup unadjusted for a 2:1 split
    i = b.index[b["ticker"] == "BBB"][5]
    b.loc[i, "close"] *= 1.05                                         # one bad print
    rec = D.reconcile(p, b)
    assert rec["split_suspects"] == ["AAA"]
    assert (rec["disagreements"]["ticker"] == "BBB").sum() == 1
    assert rec["max_rel_diff"] == pytest.approx(1.0)


def test_reconcile_disjoint_and_empty_overlap():
    p = D.validate_prices(make_prices(("AAA",)), "p", AS_OF)[0]
    b = D.validate_prices(make_prices(("ZZZ",)), "b", AS_OF)[0]
    rec = D.reconcile(p, b)
    assert rec["overlap"] == 0 and rec["coverage"] == 0.0 and len(rec["only_backup"]) == 60


def test_merge_fills_gaps_but_never_overrides_primary():
    full = D.validate_prices(make_prices(("AAA", "BBB"), n=60), "b", AS_OF)[0]
    p = full.drop(full.index[[10, 11, 70]])
    b = full.copy()
    b["close"] = b["close"] * 1.0                                     # identical -> fills are exact
    merged, info = D.merge_with_backup(p, b, max_fill_frac=0.1)
    assert info["filled"] == 3 and len(merged) == len(full) and (merged["source"] == "backup").sum() == 3
    prim = merged[merged["source"] == "primary"].sort_values(["ticker", "date"])
    pd.testing.assert_frame_equal(prim.drop(columns="source").reset_index(drop=True),
                                  p.sort_values(["ticker", "date"]).reset_index(drop=True))


def test_merge_refuses_split_suspects_and_mass_fill():
    full = D.validate_prices(make_prices(("AAA", "BBB"), n=60), "b", AS_OF)[0]
    p = full.drop(full.index[[10, 11]])
    b = full.copy()
    b.loc[b["ticker"] == "AAA", "close"] *= 2.0
    _, info = D.merge_with_backup(p, b, max_fill_frac=0.5)
    assert info["refused_split"] == ["AAA"] and info["filled"] == 0   # the gaps are AAA's; its backup is 2x off
    m, info2 = D.merge_with_backup(p, full, max_fill_frac=0.001)      # 2 fills > 0.1%
    assert not info2["applied"] and len(m) == len(p)


def test_parity_detects_single_changed_value():
    a = D.validate_prices(make_prices(), "a", AS_OF)[0]
    assert D.parity(a, a.copy())["ok"]
    b = a.copy()
    b.loc[7, "volume"] += 500
    r = D.parity(a, b)
    assert not r["ok"] and r["volume"] == 1 and r["close"] == 0
    assert D.parity(a, b.iloc[1:])["only_a"] == 1


# ---- delisted registry
def registry():
    return D.DelistedRegistry([
        {"ticker": "ENRN", "delist_date": "2001-12-03", "announced": "2001-11-28", "reason": "bankruptcy",
         "last_close": 0.26, "terminal_return": -1.0},
        {"ticker": "TWX", "delist_date": "2018-06-14", "announced": "2016-10-22", "reason": "acquired",
         "last_close": 99.4, "terminal_return": 0.0},
    ])


def test_registry_is_point_in_time():
    r = registry()
    assert not r.is_delisted("ENRN", "2001-11-30")            # announced but still trading
    assert not r.is_delisted("ENRN", "2001-12-03")            # last trading day itself
    assert r.is_delisted("ENRN", "2001-12-04")
    assert len(r.delistings_known_at("2001-11-27")) == 0
    assert list(r.delistings_known_at("2017-01-01")["ticker"]) == ["ENRN", "TWX"]
    assert r.terminal_return("ENRN", "2001-11-30") is None
    assert r.terminal_return("enrn", "2002-01-01") == -1.0


def test_registry_rejects_bad_records_and_keeps_reasons():
    r = D.DelistedRegistry()
    bad = [{"ticker": "A", "delist_date": "2020-01-10", "announced": "2020-02-01"},
           {"ticker": "B", "delist_date": "2020-01-10", "reason": "aliens"},
           {"ticker": "C", "delist_date": "garbage"}, {"delist_date": "2020-01-01"},
           {"ticker": "D", "delist_date": "2020-01-10", "last_close": -3},
           {"ticker": "E", "delist_date": "2020-01-10", "terminal_return": -1.5}]
    assert not any(r.add(x) for x in bad)
    assert [w for _, w in r.rejected] == ["announced after delist date", "unknown reason", "schema", "schema",
                                          "bad last_close", "terminal return below -100%"]
    assert r.frame().empty
    assert r.add({"ticker": "F", "delist_date": "2020-01-10"}) and not r.add({"ticker": "F", "delist_date": "2020-01-10"})


def test_registry_reused_ticker_episodes():
    r = D.DelistedRegistry([{"ticker": "X", "delist_date": "2005-01-03"}, {"ticker": "X", "delist_date": "2015-01-05"}])
    assert r.is_delisted("X", "2010-01-01") and r.is_delisted("X", "2016-01-01")
    assert len(r.frame()) == 2


def test_universe_at_keeps_future_delisted_names():
    r = registry()
    listed = {"AAPL": ("1980-12-12", None), "ENRN": ("1985-01-01", "2001-12-03"), "TWX": ("2003-01-01", "2018-06-14"),
              "NEW": ("2010-01-01", None), "GHOST": ("1990-01-01", "1999-01-01")}
    assert r.universe_at(listed, "2001-06-01") == ["AAPL", "ENRN"]        # survivors-only would drop ENRN
    assert r.universe_at(listed, "2015-01-01") == ["AAPL", "NEW", "TWX"]
    assert r.universe_at(listed, "2019-01-01") == ["AAPL", "NEW"]         # GHOST: unexplained stop, excluded
    gap = r.survivorship_gap(listed, "2001-01-01", "2002-01-01")
    assert gap["delisted_in_window"] == ["ENRN"] and gap["share_of_start_universe"] == pytest.approx(0.5)


def test_registry_roundtrip_and_missing_file(tmp_path):
    r = registry()
    r.save(tmp_path / "d.json")
    r2 = D.DelistedRegistry.load(tmp_path / "d.json")
    pd.testing.assert_frame_equal(r.frame(), r2.frame())
    assert D.DelistedRegistry.load(tmp_path / "none.json").frame().empty


# ---- point-in-time and information value
def test_point_in_time_filter():
    df = pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                       "available": ["2024-01-03", "2024-01-02", "2024-02-10", "2024-01-06"]})
    out, c = D.point_in_time_filter(df, "2024-01-31")
    assert c == {"future": 1, "impossible": 1} and len(out) == 2
    with pytest.raises(KeyError):
        D.point_in_time_filter(df.drop(columns="available"), "2024-01-31")


def panel(n_dates=120, n_t=30, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2023-01-02", periods=n_dates), [f"T{i}" for i in range(n_t)]],
                                     names=["date", "ticker"])
    base = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    new = pd.DataFrame({"n": rng.normal(size=len(idx))}, index=idx)
    y = 0.3 * base["a"] + signal * new["n"] + rng.normal(size=len(idx))
    return base, new, y


def test_information_value_detects_planted_signal():
    base, new, y = panel(signal=0.6)
    r = D.incremental_information(base, new, y, seed=1, n_perm=20)
    assert r["adds_value"] and r["gain"] > r["null_mean"] and r["p_value"] < 0.05


def test_information_value_rejects_noise_and_is_deterministic():
    base, new, y = panel(signal=0.0, seed=4)
    r1 = D.incremental_information(base, new, y, seed=2, n_perm=20)
    r2 = D.incremental_information(base, new, y, seed=2, n_perm=20)
    assert r1 == r2 and not r1["adds_value"]


def test_information_value_degenerate_no_overlap():
    base, new, y = panel(n_dates=30)
    other = new.copy()
    other.index = other.index.set_levels(other.index.levels[1].str.replace("T", "Q"), level=1)
    with pytest.raises(Exception):
        D.incremental_information(base, other, y, seed=0, n_perm=2)


# ---- calendar coverage, splits, stale feeds, eras, stops, ledger
def test_expected_sessions_skip_holidays():
    s = D.expected_sessions("2026-11-23", "2026-11-30")
    assert pd.Timestamp("2026-11-26") not in s and len(s) == 5


def test_coverage_counts_real_gaps_not_holidays():
    raw = make_prices(("AAA",), n=10, start="2026-11-16")
    sess = D.expected_sessions("2026-11-16", "2026-11-27")           # no Thanksgiving
    df = raw[raw["date"].isin(sess)]
    assert D.coverage(df).loc["AAA", "missing"] == 0                 # the holiday is not counted as missing
    r = D.coverage(df[~df["date"].between("2026-11-17", "2026-11-19")]).loc["AAA"]
    assert r["missing"] == 3 and r["longest_missing_run"] == 3 and r["expected"] == 9
    _, rep = D.validate_prices(raw, "x", "2026-12-31")               # a print ON Thanksgiving is flagged, not dropped
    assert any("NYSE holiday" in w for w in rep.warnings) and rep.rows_out == len(raw)


def test_detect_splits_flags_planted_split_with_volume_support_and_not_a_crash():
    idx = pd.bdate_range("2024-01-01", periods=30)
    close = pd.DataFrame({"S": 100.0, "C": 100.0, "Q": 100.0}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    close.loc[idx[15]:, "S"] = 50.0
    vol.loc[idx[15]:, "S"] = 2e6                                     # 2:1 split, volume doubles
    close.loc[idx[20]:, "C"] = 80.0                                  # -20% crash: no split ratio
    close.loc[idx[10]:, "Q"] = 25.0                                  # 4:1 price drop but volume unchanged
    r = D.detect_splits(close, vol).set_index("ticker")
    assert set(r.index) == {"S", "Q"} and "C" not in r.index
    assert bool(r.loc["S", "supported"]) and not bool(r.loc["Q", "supported"])
    assert D.detect_splits(close.iloc[:0]).empty


def test_stale_runs_finds_flatline_and_zero_volume():
    df = make_prices(("AAA", "BBB"), n=40)
    df.loc[(df["ticker"] == "AAA") & (df["date"] >= "2024-03-20") & (df["date"] <= "2024-03-27"), "close"] = 77.0
    df.loc[(df["ticker"] == "BBB") & (df["date"] >= "2024-04-01") & (df["date"] <= "2024-04-05"), "volume"] = 0
    r = D.stale_runs(df, min_run=5)
    assert set(zip(r["ticker"], r["kind"])) == {("AAA", "same_close"), ("BBB", "zero_volume")}
    assert D.stale_runs(make_prices(("AAA",), n=40), min_run=5).empty


def test_defect_census_per_era():
    a = make_prices(("AAA",), n=30, start="1985-01-01")
    b = make_prices(("AAA",), n=30, start="2015-01-01")
    a.loc[3, "close"] = np.nan
    b.loc[5, ["high"]] = 1.0                                          # high below the close
    b.loc[6, "volume"] = 0
    c = D.defect_census(pd.concat([a, b]))
    assert list(c.index) == ["pre-1990", "2010s"]
    assert c.loc["pre-1990", "missing_close"] == 1 and c.loc["2010s", "ohlc_inconsistent"] == 1
    assert c.loc["2010s", "zero_volume"] == 1 and c.loc["2010s", "defect_rate"] > 0
    assert c.loc["pre-1990", "rows"] == 30


def test_unexplained_stops_flags_only_unregistered_dead_names():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.normal(0, 1, (120, 4)).cumsum(0), index=idx, columns=["LIVE", "GONE", "REG", "BK"])
    close.loc[idx[60]:, "GONE"] = np.nan
    close.loc[idx[60]:, "REG"] = np.nan
    close.loc[idx[60]:, "BK"] = np.nan
    close.loc[idx[40]:idx[59], "BK"] = np.linspace(2, 0.3, 20)
    close["FROZEN"] = close["LIVE"]
    close.loc[idx[70]:, "FROZEN"] = close.loc[idx[70], "FROZEN"]            # forward-filled dead name
    reg = D.DelistedRegistry([{"ticker": "REG", "delist_date": idx[59]}])
    r = D.unexplained_stops(close, idx[-1], reg).set_index("ticker")
    assert set(r.index) == {"GONE", "BK", "FROZEN"} and r.loc["BK", "guess"] == "bankruptcy-like"
    assert r.loc["GONE", "guess"] == "acquired-like" and r.loc["FROZEN", "kind"] == "flat_tail"
    assert r.loc["FROZEN", "last_date"] == idx[70] and r.loc["GONE", "kind"] == "no_data"


def test_provenance_ledger_chain_detects_tamper_and_refuses_failed(tmp_path):
    led = D.ProvenanceLedger(tmp_path / "l.jsonl")
    assert led.verify() == {"ok": True, "entries": 0, "broken_at": []} and led.latest("x") is None
    _, r1 = D.validate_prices(make_prices(), "src1", AS_OF)
    _, r2 = D.validate_prices(make_prices(seed=3), "src2", AS_OF)
    led.append(r1)
    led.append(r2, note="second")
    assert led.verify()["ok"] and led.latest("src2")["note"] == "second"
    text = (tmp_path / "l.jsonl").read_text()
    (tmp_path / "l.jsonl").write_text(text.replace('"rows_out": 120', '"rows_out": 121', 1))
    v = led.verify()
    assert not v["ok"] and v["broken_at"][0] == 0
    _, bad = D.validate_prices(make_prices().drop(columns=["close"]), "x", AS_OF)
    with pytest.raises(ValueError):
        led.append(bad)


# ---- EDGAR-derived delistings (pure logic; network is in scripts/fetch_delisted.py)
IDX = """Description:           Master Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 31, 2009

Form Type   Company Name                                                  CIK         Date Filed  File Name
--------------------------------------------------------------------------------------------------------------------------------------------
25               ADHEREX TECHNOLOGIES INC                                      1211583     2009-01-20  edgar/data/1211583/0001193125-09-008152.txt
25-NSE           AMEN PROPERTIES INC                                           1037599     2009-02-23  edgar/data/1037599/0001157523-09-001490.txt
15-12G           ADHEREX TECHNOLOGIES INC                                      1211583     2009-02-10  edgar/data/1211583/x.txt
10-K             APPLE INC                                                     320193      2009-11-01  edgar/data/320193/y.txt
8-K              APPLE INC                                                     320193      2009-11-02  edgar/data/320193/z.txt
10-K             WEIRD CO                                                      inf         2009-11-01  edgar/data/1/bad.txt
10-K             BADDATE CO                                                    555         not-a-date  edgar/data/1/bad2.txt
"""


def test_parse_form_idx_and_relevant_filter():
    df = D.parse_form_idx(IDX)
    assert len(df) == 5 and df["cik"].dtype == "int64"          # the "inf" CIK and the bad date are dropped
    keep = D.keep_relevant_forms(df)
    assert sorted(keep["form"]) == ["10-K", "15-12G", "25", "25-NSE"]
    assert D.parse_form_idx("nothing here").empty


def ev(rows):
    return pd.DataFrame(rows, columns=["form", "company", "cik", "date"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_classify_delistings_statuses():
    rows = [("25", "DEAD INC", 1, "2010-03-01"), ("15-12G", "DEAD INC", 1, "2010-03-20"),        # deregistered
            ("25-NSE", "MOVER INC", 2, "2010-03-01"), ("10-K", "MOVER INC", 2, "2012-03-01"),     # transfer: keeps filing
            ("25", "DARK INC", 3, "2010-03-01"),                                                  # nothing after
            ("25", "NEW INC", 4, "2024-06-01"),                                                   # too recent
            ("25", "PARTIAL INC", 5, "2005-01-01"), ("10-K", "PARTIAL INC", 5, "2008-01-01"),
            ("25", "PARTIAL INC", 5, "2015-01-01"), ("15-12B", "PARTIAL INC", 5, "2015-01-10")]  # last F25 is terminal
    c = D.classify_delistings(ev(rows), "2024-12-31").set_index("cik")
    assert c["status"].to_dict() == {1: "deregistered", 2: "continuing", 3: "went_dark", 4: "too_recent",
                                     5: "deregistered"}
    assert c["terminal"].to_dict() == {1: True, 2: False, 3: True, 4: False, 5: True}
    assert c.loc[1, "delist_date"] == pd.Timestamp("2010-03-11") and c.loc[1, "announced"] == pd.Timestamp("2010-03-01")


def test_classify_is_point_in_time():
    rows = [("25", "X", 1, "2010-03-01"), ("15-12G", "X", 1, "2010-04-01"), ("10-K", "X", 1, "2013-01-01")]
    early = D.classify_delistings(ev(rows), "2012-01-01").iloc[0]
    late = D.classify_delistings(ev(rows), "2014-01-01").iloc[0]
    assert early["status"] == "deregistered" and late["status"] == "continuing"      # the later 10-K is invisible early
    assert D.classify_delistings(ev(rows).iloc[0:0], "2014-01-01").empty


def test_name_similarity_and_symbol_pick():
    assert D.clean_name("Lehman Brothers Holdings Inc.") == "LEHMAN BROTHERS"
    assert D.name_similarity("LEHMAN BROTHERS HOLDINGS INC", "Lehman Brothers Holdings Capita") > 0.75
    assert D.name_similarity("ENRON CORP", "Enron Oil & Gas") < 0.75
    assert D.name_similarity("", "X") == 0.0
    q = [{"symbol": "LEHKQ", "shortname": "Lehman Brothers Holdings Capita", "quoteType": "EQUITY"},
         {"symbol": "ETF1", "shortname": "Lehman Brothers Holdings Fund", "quoteType": "ETF"}]
    assert D.pick_symbol("LEHMAN BROTHERS HOLDINGS INC", q)["symbol"] == "LEHKQ"
    assert D.pick_symbol("ENRON CORP", [{"symbol": "EOG", "shortname": "EOG Resources"}]) is None   # wrong name: none


def test_to_registry_and_attrition_coverage():
    rows = [("25", f"CO{i}", i, f"{2010 + i % 2}-03-01") for i in range(1, 9)] + \
           [("15-12G", f"CO{i}", i, f"{2010 + i % 2}-03-20") for i in range(1, 9)]
    c = D.classify_delistings(ev(rows), "2020-01-01")
    reg = D.to_registry(c, {1: "AAA", 2: "BBB", 3: "CCC"}, {"AAA": 4.2})
    assert len(reg.frame()) == 3 and reg.rows["AAA"][0]["last_close"] == 4.2
    assert reg.is_delisted("AAA", "2011-06-01") and not reg.is_delisted("AAA", "2010-03-05")   # PIT holds
    alive = pd.Series({2010: 100, 2011: 100})
    cov = D.attrition_coverage(c, resolved={1, 2, 3, 4}, priced={1, 2}, alive_by_year=alive)
    assert cov["terminal_events"].sum() == 8 and cov["resolved"].sum() == 4 and cov["priced"].sum() == 2
    assert cov.loc[2010, "expected_low"] == 3.0 and cov.loc[2010, "found_vs_high"] == pytest.approx(4 / 6.0)
