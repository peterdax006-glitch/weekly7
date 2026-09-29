"""engine/leak_audit.py: every channel has a test that FAILS when the leak exists (a planted defect is caught, the clean
case passes) plus the empty/degenerate case. Synthetic data only; state/livesim is redirected to tmp_path."""
import json
import socket
import threading

import numpy as np
import pandas as pd
import pytest

from engine import blind_gates as BG, features, leak_audit as L, livesim


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def panel(n_days=700, n=30, seed=1, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    r = rng.normal(0.0004, 0.02, (n_days, n))
    close = pd.DataFrame(40 * np.exp(r.cumsum(0)), index=idx, columns=[f"T{i:02d}" for i in range(n)])
    vol = pd.DataFrame(rng.integers(2e5, 2e6, (n_days, n)).astype(float), index=idx, columns=close.columns)
    return close, vol


def seal_file(tmp, run_id="t1", seed=7):
    rec = BG.seal_window([], seed, "2026-01-01", tag=run_id)
    (tmp / f"sealed_{run_id}.json").write_text(json.dumps(rec))
    return rec


def make_data(start, n=40, seed=3, late_ipo=None):
    start = pd.Timestamp(start)
    idx = pd.bdate_range(start - pd.DateOffset(years=6), start + pd.DateOffset(months=12) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    tick = sorted([f"AB{chr(65 + i % 26)}{i}" for i in range(n)] + ["S", "KO"])
    ret = rng.normal(0.0004, 0.015, (len(idx), len(tick)))
    close = pd.DataFrame(50 * np.exp(ret.cumsum(0)), index=idx, columns=tick)
    if late_ipo:
        close.loc[: start + pd.Timedelta(days=90), late_ipo] = np.nan       # lists ~3 months into the hidden window
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    stocks = {"Close": close, "Open": opn, "High": close * 1.01, "Low": close * 0.99, "Volume": close * 0 + 1e6}
    mk = pd.DataFrame({"SPY": 100 * np.exp(rng.normal(0.0003, 0.01, len(idx)).cumsum()) * 3.0, "^VIX": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    market = {"Close": mk, "Open": mk, "High": mk, "Low": mk, "Volume": mk * 0 + 1e6}
    at = pd.DatetimeIndex(idx[::9], tz="UTC") + pd.Timedelta(hours=13)
    ev = pd.DataFrame({"ticker": [tick[i % n] for i in range(len(at))], "accepted": at, "kind": "8K", "form": "8-K"})
    fd = idx[::7]
    ins = pd.DataFrame({"symbol": [tick[i % n] for i in range(len(fd))], "filed": fd, "tdate": fd - pd.Timedelta(days=2)})
    sic = pd.DataFrame({"ticker": tick, "sic": 3570})
    return stocks, market, ev, ins, sic


class Rule:
    """Deterministic stand-in trader: weekly, buy the 3 best 6-session performers equal-weight."""

    def __init__(self, feed):
        self.feed, self.broker, self.equity = feed, livesim.SimBroker(feed), []

    def on_tick(self):
        f, b = self.feed, self.broker
        if f.next_session_is_new_week():
            c = f.history(lookback=6)[0]["Close"]
            top = (c.iloc[-1] / c.iloc[0] - 1).dropna().nlargest(3).index
            for t in list(b.pos):
                if t not in top:
                    b.order_to(t, 0.0, "exit")
            for t in top:
                b.order_to(t, b.equity() / 3, "buy")
        self.equity.append(round(b.equity(), 6))
        f.mark_processed()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(livesim, "DIR", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------------------------------------------------
# report objects
# --------------------------------------------------------------------------------------------------------------------
def test_channel_rejects_unknown_status():
    with pytest.raises(ValueError):
        L.Channel("x", "bad", "MAYBE")


def test_audit_counts_open_leaks_and_saves(tmp_path):
    a = L.Audit()
    a.add(L.Channel("1", "one", L.LEAK, {"n": np.int64(3), "s": pd.Series({"a": 1.0})}, test="t1"))
    a.add(L.Channel("2", "two", L.FIXED, {"share": float("nan")}, fix="done"))
    assert a.counts() == {"LEAK": 1, "CLEAN": 0, "FIXED": 1, "QUARANTINED": 0} and a.open_leaks() == ["1"]
    a.save(tmp_path, stamp={"code_hash": "abc", "seed": 5})
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["open_leaks"] == ["1"] and s["provenance"]["code_hash"] == "abc" and s["channels"]["2"]["evidence"]["share"] is None
    assert "LEAK" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_empty_audit_is_valid():
    a = L.Audit()
    assert a.counts()["LEAK"] == 0 and a.open_leaks() == [] and "0 channels" in a.markdown()


# --------------------------------------------------------------------------------------------------------------------
# 7 network
# --------------------------------------------------------------------------------------------------------------------
def _server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    def accept():
        try:
            srv.accept()
        except OSError:
            pass
    threading.Thread(target=accept, daemon=True).start()
    return srv


def test_network_reachable_without_guard_and_blocked_with_it():
    srv = _server()
    port = srv.getsockname()[1]
    c = socket.create_connection(("127.0.0.1", port), timeout=2)      # proves the connection WOULD work: the guard is what stops it
    c.close()
    srv2 = _server()
    with L.NetworkGuard() as g:
        with pytest.raises(L.NetworkBlocked):
            socket.create_connection(("127.0.0.1", srv2.getsockname()[1]), timeout=2)
        with pytest.raises(L.NetworkBlocked):
            socket.getaddrinfo("query1.finance.yahoo.com", 443)
        assert g.blocked
    srv3 = _server()
    socket.create_connection(("127.0.0.1", srv3.getsockname()[1]), timeout=2).close()   # uninstalled: works again
    for s in (srv, srv2, srv3):
        s.close()


def test_guard_allow_loopback_still_blocks_remote():
    srv = _server()
    with L.NetworkGuard(allow_loopback=True):
        socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2).close()
        with pytest.raises(L.NetworkBlocked):
            socket.getaddrinfo("example.com", 80)
    srv.close()


def test_poison_raises_and_restores():
    import types
    M = types.ModuleType("fakedata")
    M.update = lambda name: "refreshed"
    restore = L.poison(M, ["update", "missing"])
    with pytest.raises(L.NetworkBlocked, match="fakedata.update"):
        M.update("stocks")
    restore()
    assert M.update("stocks") == "refreshed"


def test_network_markers_find_planted_refresh(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import requests\nfrom x import y\n\ndef go():\n    data.update('stocks')\n    return 1\n")
    kinds = {(m["kind"], m["what"]) for m in L.network_markers(f)}
    assert ("import", "requests") in kinds and ("call", "update") in kinds
    g = tmp_path / "clean.py"
    g.write_text("import numpy as np\n\ndef go():\n    return np.zeros(3)\n")
    assert L.network_markers(g) == []


# --------------------------------------------------------------------------------------------------------------------
# 8 reachability
# --------------------------------------------------------------------------------------------------------------------
def test_import_closure_follows_lazy_imports_and_finds_reads(tmp_path):
    (tmp_path / "a.py").write_text("from . import b\n\ndef f():\n    from .c import g\n    return g()\n")
    (tmp_path / "b.py").write_text("X = 1\n")
    (tmp_path / "c.py").write_text("import pandas as pd\n\ndef g():\n    return pd.read_parquet('secret_future.parquet')\n")
    (tmp_path / "d.py").write_text("Z = 1\n")
    cl = L.import_closure([tmp_path / "a.py"], engine_dir=tmp_path)
    assert set(cl) == {"a", "b", "c"} and "d" not in cl
    reads = L.trader_side_reads(cl, engine_dir=tmp_path)
    assert list(reads) == ["c"] and "secret_future" in reads["c"][0]["call"]
    assert reads["c"][0]["function"] == "g"


def test_file_access_recorder_sees_reads_inside_the_block_only(tmp_path):
    from engine import config as K
    inside = K.ROOT / "state" / "_leak_audit_probe.txt"
    (tmp_path / "outside.txt").write_text("x")
    with L.FileAccessRecorder() as rec:
        pd.Series([1]).to_csv(tmp_path / "in_tmp.csv")
        inside.write_text("y")
        inside.read_text()
    inside.unlink()
    open(tmp_path / "outside.txt").read()                              # after the block: not recorded
    assert any(p.endswith("_leak_audit_probe.txt") for p in rec.paths)
    assert not any(p.endswith("outside.txt") for p in rec.paths)
    pf = rec.project_files()
    assert "state/_leak_audit_probe.txt" in pf["state"]
    assert all(not p.startswith("..") for k in pf for p in pf[k])


def test_file_access_recorder_catches_a_planted_cache_read():
    from engine import config as K
    target = K.CACHE / "sic.parquet"
    with L.FileAccessRecorder() as rec:
        pd.read_parquet(target)
    assert "data/cache/sic.parquet" in rec.project_files()["data_cache"]
    with L.FileAccessRecorder() as quiet:
        _ = 1 + 1
    assert quiet.project_files() == {"code": [], "data_cache": [], "state": [], "other": []}


def test_import_closure_empty_entry():
    assert L.import_closure([]) == {}


def test_real_blind_path_reaches_no_pattern_bank_or_lessons():
    from engine import config as K
    cl = L.import_closure([K.ROOT / "engine" / "livesim.py", K.ROOT / "engine" / "adaptive.py", K.ROOT / "scripts" / "livesim_loop2.py"])
    assert {"livesim", "adaptive", "memory", "features", "model"} <= set(cl)
    assert not ({"pattern_bank", "lessons", "analogs", "patterns", "trust_store"} & set(cl))


# --------------------------------------------------------------------------------------------------------------------
# 1 survivorship
# --------------------------------------------------------------------------------------------------------------------
def test_attrition_profile_sees_dead_names_and_flags_survivor_panel():
    close, _ = panel(n_days=1300)
    surv = L.attrition_profile(close)
    assert surv["n_exit"].sum() == 0                               # a survivor panel: nobody ever leaves
    dead = close.copy()
    dead.iloc[400:, :6] = np.nan                                   # six names leave in year 3
    prof = L.attrition_profile(dead)
    assert prof["n_exit"].sum() == 6 and prof.loc[2016, "exit_rate"] > 0


def test_attrition_profile_empty():
    assert L.attrition_profile(pd.DataFrame()).empty


def test_delisting_hazard_and_haircut_arithmetic():
    ev = pd.DataFrame({"delist_date": pd.to_datetime(["2016-03-01", "2016-05-01", "2017-01-01", "2017-02-01"]),
                       "terminal": [True, True, True, False]})
    h = L.delisting_hazard(ev, pd.Series({2016: 100, 2017: 100}))
    assert h.loc[2016, "rate"] == 0.02 and h.loc[2017, "rate"] == 0.01     # the non-terminal event is not counted
    hc = L.survivorship_haircut(0.05, terminal_loss=-0.5, concentration=1.0)
    assert hc["weekly_drag"] == pytest.approx(-0.5 * (1 - 0.95 ** (1 / 52)))
    assert L.survivorship_haircut(0.05, -0.5, concentration=3.0)["weekly_drag"] < hc["weekly_drag"]
    assert L.survivorship_haircut(0.0)["weekly_drag"] == 0.0
    with pytest.raises(ValueError):
        L.survivorship_haircut(1.5)


def test_delisting_hazard_empty_registry():
    h = L.delisting_hazard(pd.DataFrame(), pd.Series({2016: 50}))
    assert h.loc[2016, "rate"] == 0.0


def test_concentration_from_registry():
    reg = pd.DataFrame({"last_close": [1.0, 2.0, 50.0, 2.5]})
    r = L.concentration_from_registry(reg, pd.Series([10.0, 20.0, 30.0, 40.0, 2.0]))
    assert r["dead_below_floor"] == 0.75 and r["live_below_floor"] == 0.2 and r["ratio"] == pytest.approx(3.75)
    assert np.isnan(L.concentration_from_registry(pd.DataFrame(), pd.Series(dtype=float))["ratio"])


def test_inject_dead_names_creates_terminal_prints_and_lowers_vol_basket():
    close, _ = panel(n_days=900, n=80, seed=4)
    dates = close.index[[300, 400, 500, 600, 700] * 4]
    aug = L.inject_dead_names(close, dates, seed=11, terminal_mean=-0.6, terminal_sd=0.1, high_vol_bias=1.0)
    added = [c for c in aug.columns if c.startswith("DEAD")]
    assert len(added) == 20 and aug[added].notna().any(axis=1).sum() > 0
    for c in added:                                                    # each one ends on its terminal print
        last = aug[c].last_valid_index()
        assert aug[c].loc[last] < aug[c].dropna().iloc[-2]
    base = L.vol_basket_weekly(close, top_frac=0.15).mean()
    corrected = L.vol_basket_weekly(aug, top_frac=0.15).mean()
    assert corrected < base                                            # the survivor panel overstates a volatility basket
    again = L.inject_dead_names(close, dates, seed=11, terminal_mean=-0.6, terminal_sd=0.1, high_vol_bias=1.0)
    pd.testing.assert_frame_equal(aug, again)                          # seeded
    assert L.inject_dead_names(close, [], seed=1).equals(close)


def test_vol_basket_clips_a_planted_glitch_and_thin_universe_share_counts_thin_windows():
    close, _ = panel(n_days=400, n=40, seed=6)
    glitch = close.copy()
    glitch.iloc[200:, 3] *= 500.0                                    # a persistent 500x data error in one name
    raw = L.vol_basket_weekly(glitch, top_frac=1.0, clip=(-100.0, 1e9))
    clipped = L.vol_basket_weekly(glitch, top_frac=1.0)
    assert raw.max() > 3 * clipped.max() and clipped.max() < 1.0
    alive = pd.Series({y: (30 if y < 1975 else 800) for y in range(1965, 2027)})
    starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
    r = L.thin_universe_share(alive, starts, min_names=500)
    assert 0.1 < r["share_below"] < 0.2 and r["first_clean_start"] == "1975-01-01" and r["min_names_in_any_window"] == 30
    assert r["share_below_by_decade"][1960] == 1.0 and r["share_below_by_decade"][2000] == 0.0
    assert np.isnan(L.thin_universe_share(pd.Series(dtype=float), starts)["share_below"])


def test_pit_universe_excludes_future_ipo_and_effective_delistings():
    close, _ = panel(n_days=500, n=6)
    close.iloc[:300, 0] = np.nan                                       # T00 lists at session 300
    close.iloc[380:, 1] = np.nan                                       # T01 last trades at 379
    reg = pd.DataFrame({"ticker": ["T01", "T02"], "announced": [close.index[370], close.index[450]],
                        "delist_date": [close.index[379], close.index[460]]})
    u = L.pit_universe(close, close.index[200], reg)
    assert "T00" not in u and "T01" in u and "T02" in u
    u2 = L.pit_universe(close, close.index[400], reg)
    assert "T00" in u2 and "T01" not in u2 and "T02" in u2             # T02's delisting is announced later: still investable
    assert L.pit_universe(close.iloc[0:0], "2020-01-01") == []


# --------------------------------------------------------------------------------------------------------------------
# 2 adjusted prices
# --------------------------------------------------------------------------------------------------------------------
def test_cum_future_split_factor_and_reconstruction():
    idx = pd.bdate_range("2020-01-01", periods=10)
    splits = pd.Series({idx[4]: 4.0, idx[8]: 2.0})
    f = L.cum_future_split_factor(splits, idx)
    assert f.iloc[0] == 8.0 and f.iloc[4] == 2.0 and f.iloc[7] == 2.0 and f.iloc[8] == 1.0 and f.iloc[9] == 1.0
    adj = pd.Series(10.0, index=idx)
    traded = L.reconstruct_as_traded(adj, splits)
    assert traded.iloc[0] == 80.0 and traded.iloc[-1] == 10.0
    div = pd.Series(np.linspace(0.8, 1.0, 10), index=idx)
    assert L.reconstruct_as_traded(adj, splits, div).iloc[0] == pytest.approx(80.0 / 0.8)
    assert L.cum_future_split_factor(pd.Series(dtype=float), idx).eq(1.0).all()


def test_price_level_drift_detects_back_adjustment():
    idx = pd.bdate_range("2010-01-01", "2019-12-31")
    rng = np.random.default_rng(0)
    base = pd.DataFrame(30 * np.exp(rng.normal(0, 0.3, (len(idx), 50))), index=idx)      # as-traded: flat level
    flat = L.price_level_drift(base)["median_close"]
    assert flat.max() / flat.min() < 1.3
    adj = base * np.linspace(0.25, 1.0, len(idx))[:, None]                                # later splits shrink old prices
    d = L.price_level_drift(adj)
    assert d["median_close"].iloc[0] < 0.6 * d["median_close"].iloc[-1]
    assert d["share_below_3"].iloc[0] >= d["share_below_3"].iloc[-1]


def test_level_rule_disagreement_catches_split_and_is_zero_when_unadjusted():
    close, _ = panel(n_days=300, n=20, seed=2)
    assert L.level_rule_disagreement(close, close)["floor_flip_share"] == 0.0
    traded = close.copy()
    traded.iloc[:, :5] *= 8.0                                       # five names split 8:1 later: as-traded were 8x higher
    adj = close
    adj = adj.assign(**{c: adj[c] * 0.05 for c in adj.columns[:3]})       # ... and adjusted prices fall under the $3 floor
    r = L.level_rule_disagreement(adj, traded, floor=3.0)
    assert r["floor_flip_share"] > 0.05 and r["adj_below_floor_but_traded_above"] > 0
    assert L.level_rule_disagreement(adj.iloc[0:0], traded.iloc[0:0]) == {"n_cells": 0}


def test_split_invariant_tradable_ignores_a_later_split_but_price_rank_does_not():
    close, vol = panel(n_days=200, n=40, seed=5)
    base = L.split_invariant_tradable(close, vol)
    c2, v2 = close.copy(), vol.copy()
    c2.iloc[:, :10] /= 20.0                                          # a later 20:1 split rewrites history: price down, volume up
    v2.iloc[:, :10] *= 20.0
    after = L.split_invariant_tradable(c2, v2)
    assert (base == after).all().all()
    price_rank = lambda C: C.rank(axis=1, pct=True) >= 0.2           # the clause features.build uses in relative mode
    assert (price_rank(close) != price_rank(c2)).to_numpy().mean() > 0.05


# --------------------------------------------------------------------------------------------------------------------
# 3 metadata
# --------------------------------------------------------------------------------------------------------------------
def test_crypto_exposure_flags_name_and_list_matches_with_dates():
    first = pd.Series(pd.to_datetime(["1998-01-05", "2021-04-14", "2010-01-04", "2012-03-01"]), index=["MSTR", "COIN", "ABC", "XYZ"])
    last = pd.Series(pd.to_datetime(["2026-09-25"] * 4), index=first.index)
    names = pd.Series({"MSTR": "Strategy Inc", "COIN": "Coinbase Global", "ABC": "Abc Corp", "XYZ": "Xyz Blockchain Holdings"})
    df = L.crypto_exposure(first, last, names)
    assert set(df["ticker"]) == {"MSTR", "COIN", "XYZ"}
    assert bool(df.set_index("ticker").loc["XYZ", "by_name"]) and bool(df.set_index("ticker").loc["MSTR", "by_list"])
    assert L.crypto_exposure(first.iloc[0:0], last.iloc[0:0]).empty


def test_crypto_filter_is_inert_on_disguised_codes():
    from engine import policy
    codes = pd.Index([f"S{n:04d}" for n in range(30)])
    assert policy.not_crypto(codes).all()                            # blind codes never match today's names: the filter cannot leak, it does nothing


def test_coarsen_sic_reduces_codes():
    sic = pd.DataFrame({"ticker": ["A", "B", "C"], "sic": [3571, 3674, 7372]})
    c = L.coarsen_sic(sic, 1)
    assert list(c["sic"]) == ["3000", "3000", "7000"] and sic["sic"].tolist() == [3571, 3674, 7372]


# --------------------------------------------------------------------------------------------------------------------
# 4 learned state
# --------------------------------------------------------------------------------------------------------------------
def _lin():
    lin = L.BasisLineage()
    T = lambda i, a, b: L.TrainedOn(i, pd.Timestamp(a), pd.Timestamp(b))
    lin.register(1, {"k": 2}, {}, [T("w1", "2000-01-01", "2000-12-31")])
    lin.register(2, {"k": 3}, {}, [T("w1", "2000-01-01", "2000-12-31"), T("w2", "2015-01-01", "2015-12-31")])
    return lin


def test_basis_for_picks_newest_version_trained_only_on_the_past():
    lin = _lin()
    assert lin.basis_for("2020-06-01")["version"] == 2               # both windows ended earlier
    assert lin.basis_for("2010-01-01")["version"] == 1               # v2 saw 2015: a future window for a 2010 target
    assert lin.basis_for("1990-01-01") is None                       # v1 saw 2000: nothing clean, use the defaults
    assert lin.basis_for("2000-06-01") is None                       # the overlapping window is not the past either


def test_basis_for_same_window_needs_explicit_opt_in():
    lin = _lin()
    assert lin.basis_for("2000-01-01") is None
    assert lin.basis_for("2000-01-01", allow_same_window=True)["version"] == 1   # C54 reruns: owner must opt in


def test_violations_flags_planted_future_training():
    lin = _lin()
    v = lin.violations([{"id": "wA", "real_start": "2010-01-01", "version": 2}, {"id": "wB", "real_start": "2020-01-01", "version": 2}])
    assert [x["window"] for x in v] == ["wA"] and v[0]["trained_on_late"] == ["w2"]
    assert lin.violations([]) == []


def test_strict_training_windows():
    ws = [{"id": "a", "end": pd.Timestamp("2001-01-01")}, {"id": "b", "end": pd.Timestamp("2011-01-01")}]
    assert [w["id"] for w in L.strict_training_windows(ws, "2005-01-01")] == ["a"]
    assert L.strict_training_windows([], "2005-01-01") == []


def test_current_loop_design_trains_on_windows_from_the_future():
    rounds = L.simulate_window_draws(12, par=3, seed=5)
    r = L.basis_future_share(rounds)
    assert r["n_played"] == 33 and 0.25 < r["mean_future_share"] < 0.75 and r["share_of_windows_touched"] > 0.9
    again = L.basis_future_share(L.simulate_window_draws(12, par=3, seed=5))
    assert again == r                                                # seeded
    assert L.basis_future_share([]) ["n_played"] == 0 and L.basis_future_share(rounds[:1])["n_played"] == 0


def test_lineage_filter_removes_the_measured_leak():
    """Apply BasisLineage to the simulated loop: every window's basis is trained on its past only."""
    rounds = L.simulate_window_draws(10, par=3, seed=9)
    lin = L.BasisLineage()
    seen = []
    T = lambda w: L.TrainedOn(w["id"], w["start"], w["end"])
    played = []
    for r, rnd in enumerate(rounds):
        for w in rnd:
            rec = lin.basis_for(w["start"])
            played.append({"id": w["id"], "real_start": w["start"], "version": rec["version"] if rec else 0})
        seen += rnd
        lin.register(r + 1, {}, {}, [T(w) for w in seen])
    assert lin.violations([p for p in played if p["version"]]) == []
    assert sum(p["version"] == 0 for p in played) > 3                # many early-history targets fall back to the defaults


def test_defaults_contamination_counts_windows_touching_tuned_years():
    starts = pd.date_range("1965-01-01", "2025-09-01", freq="MS")
    r = L.defaults_contamination([1969, 1987, 2016], starts)
    assert 0.05 < r["contaminated_share"] < 0.12 and r["n_tuned_years"] == 3
    assert L.defaults_contamination(range(1965, 2026), starts)["contaminated_share"] == 1.0
    assert L.defaults_contamination([], starts)["contaminated_share"] == 0.0
    assert L.defaults_contamination([2000], pd.DatetimeIndex([]))["n_windows"] == 0
    # a window starting 1 Jan 1970 covers only 1970: clean of 1969; one starting Dec 1969 touches it
    assert L.defaults_contamination([1969], pd.DatetimeIndex(["1970-01-01"]))["contaminated_share"] == 0.0
    assert L.defaults_contamination([1969], pd.DatetimeIndex(["1969-12-01"]))["contaminated_share"] == 1.0


def test_neutral_default_cfg_is_positional_and_ignores_duplicate_weights():
    space = {"k": [1, 1, 2, 2, 3, 4], "brake": [None, None, 0.15], "pick": ["hivol", "hivol", "top"]}
    assert L.neutral_default_cfg(space) == {"k": 3, "brake": 0.15, "pick": "top"}
    assert L.neutral_default_cfg({}) == {}


def test_free_text_columns_finds_a_planted_headline_and_ignores_codes():
    df = pd.DataFrame({"ticker": ["A", "B"], "form": ["8-K", "10-Q"], "kind": ["EARN", "PERIODIC"],
                       "headline": ["Company X announces record quarterly revenue and raises full-year guidance sharply above consensus"] * 2})
    assert L.free_text_columns(df) == ["headline"]
    assert L.free_text_columns(df.drop(columns="headline")) == [] and L.free_text_columns(pd.DataFrame()) == []


# --------------------------------------------------------------------------------------------------------------------
# 5 macro
# --------------------------------------------------------------------------------------------------------------------
def test_macro_revision_risk_and_unknown_series_treated_as_revised():
    r = L.macro_revision_risk(["DGS10", "USREC", "MYSTERY"]).set_index("series")
    assert not r.loc["DGS10", "revised"] and r.loc["USREC", "revised"] and r.loc["MYSTERY", "revised"]
    assert L.macro_revision_risk([]).empty


def test_pit_macro_drops_revised_lags_monthly_and_never_shows_the_future():
    idx = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    M = pd.DataFrame({"DGS10": np.arange(len(idx), dtype=float), "USREC": (idx >= "2020-02-01").astype(float),
                      "UNRATE": np.linspace(3, 9, len(idx))}, index=idx)
    p = L.pit_macro(M, "2020-06-01")
    assert list(p.columns) == ["DGS10"] and p.index.max() <= pd.Timestamp("2020-06-01")
    assert p["DGS10"].iloc[-1] == M.loc["2020-05-31", "DGS10"]      # one day of publication lag
    first_release = M["UNRATE"].shift(3)                             # a supplied vintage series replaces the drop
    p2 = L.pit_macro(M, "2020-06-01", vintages={"UNRATE": first_release})
    assert "UNRATE" in p2 and p2.index.max() <= pd.Timestamp("2020-06-01")
    assert p2["UNRATE"].dropna().index.min() >= idx[3] + pd.Timedelta(days=35)
    with_rec = L.pit_macro(M, "2020-06-01", drop_revised=False)
    assert "USREC" in with_rec                                       # the explicit opt-out is visible, not silent


# --------------------------------------------------------------------------------------------------------------------
# 6 fingerprints
# --------------------------------------------------------------------------------------------------------------------
def synth_daily(scrubbed_levels=False, seed=0):
    """Daily fingerprint series 1962-2025 where the universe grows over time (a real-world fingerprint)."""
    idx = pd.bdate_range("1962-01-02", "2025-12-31")
    rng = np.random.default_rng(seed)
    yrs = (idx.year - 1962).to_numpy()
    n = np.full(len(idx), 800.0) if scrubbed_levels else 300 + 60 * yrs + rng.normal(0, 15, len(idx))
    df = pd.DataFrame({"n_names": n, "med_logp": rng.normal(3, 0.02, len(idx)) + (0 if scrubbed_levels else 0.01 * yrs),
                       "share_lt3": 0.05, "med_logdv": rng.normal(15, 0.05, len(idx)), "zero_vol": 0.01, "hl_equal": 0.02, "open_nan": 0.0,
                       "spy_log": np.cumsum(rng.normal(0.0003, 0.01, len(idx))), "vix": 18 + rng.normal(0, 1, len(idx))}, index=idx)
    return df


def _features_for(daily, step_months=1):
    starts = pd.date_range("1965-01-01", "2025-01-01", freq="MS")[::step_months]
    return pd.DataFrame([window_row for window_row in (L.window_features(daily, s) for s in starts)], index=starts), starts


def test_probe_identifies_year_from_a_planted_level_channel_and_not_after_scrub():
    F, starts = _features_for(synth_daily())
    hit = L.FingerprintProbe(seed=0, n_trees=60).score(F, starts, L.GROUPS["levels_raw"])
    assert L.fingerprint_verdict(hit) == "identifiable" and hit["skill"] > 0.6
    Fs, starts_s = _features_for(synth_daily(scrubbed_levels=True))
    miss = L.FingerprintProbe(seed=0, n_trees=60).score(Fs, starts_s, L.GROUPS["levels_scrubbed"] + L.GROUPS["market_state"])
    assert L.fingerprint_verdict(miss) == "not identifiable" and miss["skill"] < 0.2


def test_regime_series_matches_the_trader_feature_code():
    from engine import features
    close, _ = panel(n_days=320, n=25, seed=8)
    close.iloc[:100, 5:9] = np.nan
    mk = pd.DataFrame({"SPY": 100 * np.exp(np.random.default_rng(1).normal(0.0003, 0.01, len(close)).cumsum()),
                       "^VIX": 20 + np.random.default_rng(2).normal(0, 1, len(close))}, index=close.index)
    got = L.regime_series(close, mk, col_chunk=7)
    want = features.regime_frame({"Close": mk}, close, np.log(close / close.shift(1)))
    for c in ["m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_vix", "m_vix_chg5", "m_breadth", "m_dispersion"]:
        pd.testing.assert_series_equal(got[c].astype("float32"), want[c], check_names=False, rtol=1e-3, atol=1e-5)


def test_probe_degenerate_inputs_do_not_crash():
    empty = pd.DataFrame({c: [] for c in L.CAL_FEATS})
    r = L.FingerprintProbe().score(empty, pd.DatetimeIndex([]), L.CAL_FEATS)
    assert np.isnan(r["mae_years"]) and L.fingerprint_verdict(r) == "not identifiable"
    short = synth_daily().iloc[:10]
    row = L.window_features(short, "1970-01-01")
    assert all(np.isnan(v) for v in row.values())


def test_probe_split_never_overlaps_train_and_test_windows():
    starts = pd.date_range("1965-01-01", "2025-01-01", freq="MS")
    tr, te = L.FingerprintProbe.split(starts)
    last_train_end = starts[tr].max() + pd.DateOffset(months=12)
    assert last_train_end < starts[te].min() + pd.DateOffset(years=100) and len(tr) > len(te) > 0
    for t in te[:50]:
        assert not any(abs((starts[t] - starts[i]).days) < 365 for i in tr)


def test_calendar_features_see_a_midweek_closure_and_the_regular_grid_removes_it():
    d = pd.bdate_range("2001-01-01", "2001-12-31")
    normal = L.calendar_features(d)
    with_closure = L.calendar_features(d.delete(d.get_loc(pd.Timestamp("2001-09-12"))).delete(d.get_loc(pd.Timestamp("2001-09-13"))))
    assert normal["very_short_weeks"] == 0 and with_closure["very_short_weeks"] == 1     # Wed+Thu closed: a 3-session week
    assert with_closure["short_week_rate"] > normal["short_week_rate"]
    real = d.delete([50, 51, 130, 200])
    assert L.calendar_features(real)["gap2_rate"] + L.calendar_features(real)["gap5plus"] > 0
    grid = L.regular_grid_index(len(real))
    g = L.calendar_features(grid)
    assert g["gap2_rate"] == 0 and g["gap5plus"] == 0 and g["max_gap"] == 3 and g["short_week_rate"] == 0 and g["very_short_weeks"] == 0


def test_daily_fingerprint_series_matches_direct_computation_and_handles_empty():
    close, vol = panel(n_days=90, n=12)
    close.iloc[:, 0] = np.nan
    high, low = close * 1.02, close * 0.98
    high.iloc[5, 3] = low.iloc[5, 3] = close.iloc[5, 3]
    mk = pd.DataFrame({"SPY": np.linspace(100, 120, 90), "^VIX": 20.0}, index=close.index)
    df = L.daily_fingerprint_series(close, close, high, low, vol, mk, chunk=25)
    assert (df["n_names"] == 11).all()
    assert df["hl_equal"].iloc[5] == pytest.approx(1 / 11) and df["hl_equal"].iloc[6] == 0
    assert df["med_logp"].iloc[10] == pytest.approx(np.log(close.iloc[10].dropna()).median())
    assert df["spy_log"].iloc[0] == pytest.approx(np.log(100))
    assert len(L.daily_fingerprint_series(close.iloc[0:0], None, None, None, None)) == 0


# --------------------------------------------------------------------------------------------------------------------
# 8 feature causality
# --------------------------------------------------------------------------------------------------------------------
def _synthetic_build_inputs():
    stocks, market, ev, ins, sic = make_data("2012-01-01", n=14)
    stocks = {k: v.loc["2010-06-01":"2012-12-31"] for k, v in stocks.items()}
    market = {k: v.loc["2010-06-01":"2012-12-31"] for k, v in market.items()}
    ins = ins.assign(owner_cik=1, value=50_000.0, relation="officer", title="ceo")
    return stocks, market, ev[ev["accepted"] < "2013-01-01"], ins, sic


def test_real_feature_builder_is_future_invariant_on_synthetic_panel():
    stocks, market, ev, ins, sic = _synthetic_build_inputs()
    res = L.truncation_invariance(features.build, stocks, market, ev, ins, sic, ["2012-03-15", "2012-08-20"], window=10, start="2011-06-01")
    assert res["clean"], res["leaky_features"]


def test_truncation_invariance_catches_a_planted_look_ahead_feature():
    stocks, market, ev, ins, sic = _synthetic_build_inputs()

    def peeking_build(s, m, e, i, sc, start=None, relative=False):
        X, atr = features.build(s, m, e, i, sc, start=start, relative=relative)
        nxt = s["Close"].shift(-1) / s["Close"] - 1                   # tomorrow's return leaked into today's row
        X["peek"] = nxt.stack(future_stack=True).reindex(X.index).values
        return X, atr
    res = L.truncation_invariance(peeking_build, stocks, market, ev, ins, sic, ["2012-03-15"], window=10, start="2011-06-01")
    assert not res["clean"] and "peek" in res["leaky_features"]


# --------------------------------------------------------------------------------------------------------------------
# 8 feed exposure and the hardened feed
# --------------------------------------------------------------------------------------------------------------------
def _feed(home, cls, late_ipo="ABC2"):
    rec = seal_file(home)
    data = make_data(rec["start"], late_ipo=late_ipo)
    return cls(livesim.SealedYear("t1"), data=data), rec


def test_plain_feed_shows_real_alphabetical_order_future_ipo_and_absolute_spy(home):
    feed, _ = _feed(home, livesim.Feed)
    feed.i = feed.sessions.get_loc(feed.first_live)
    e = L.feed_exposure(feed)
    assert e["column_order_vs_real_alpha_rho"] == pytest.approx(1.0)  # positions ARE the real tickers' alphabetical order
    assert e["columns_shown_before_listing"] >= 1                     # the IPO name is on the screen before it lists
    assert e["spy_first_level"] > 200                                 # raw level (300 in this synthetic world)


def test_hardened_feed_closes_the_three_exposures(home):
    feed, _ = _feed(home, L.hardened_feed_class())
    feed.i = feed.sessions.get_loc(feed.first_live)
    e = L.feed_exposure(feed)
    assert abs(e["column_order_vs_real_alpha_rho"]) < 0.5
    assert e["columns_shown_before_listing"] == 0
    assert e["spy_first_level"] == pytest.approx(100.0)
    feed.i = feed.sessions.get_loc(feed.first_live) + 200            # 8+ months in: the IPO name appears the day it lists
    assert len(feed.history()[0]["Close"].columns) == len(feed._stocks["Close"].columns)


def test_hardened_feed_changes_no_decision_on_clean_data(home):
    """No IPO, so the only differences are column order and rebased SPY: same picks, same equity, gates on."""
    f0, _ = _feed(home, livesim.Feed, late_ipo=None)
    t0 = Rule(f0)
    livesim.drive(f0, t0.on_tick)
    f1, _ = _feed(home, L.hardened_feed_class(), late_ipo=None)
    t1 = Rule(f1)
    livesim.drive(f1, t1.on_tick)
    assert len(t0.equity) == len(t1.equity) > 200
    assert t0.equity == t1.equity and t0.broker.log == t1.broker.log
    assert not [x for x in f1.audit() if x.severity == "fail"]


def test_hardened_run_closes_the_network_for_the_whole_window_and_reopens_it_after(home):
    rec = seal_file(home)
    cfg = {"k": 2, "exit_q": 0.8, "rebalance_weeks": 1, "brake": None, "max_per_sector": None, "w_model": 1.0, "pick": "hivol", "pool_q": 0.7,
           "liq_q": 0.0, "vol_filter": False, "stress_thr": None, "stress_k": 2, "trend_filter": None, "trend_gross": 0.0}
    seen = {}
    real_train = livesim.BlindTrader.train
    stocks, market, ev, ins, sic = make_data(rec["start"], n=8)
    data = (stocks, market, ev, ins.assign(owner_cik="1", value=50_000.0, relation="officer", title="ceo"), sic)

    def spying_train(self):
        try:                                                        # a refresh attempted mid-run, as a careless module might do
            socket.getaddrinfo("query1.finance.yahoo.com", 443)
            seen["blocked"] = False
        except L.NetworkBlocked:
            seen["blocked"] = True
        return real_train(self)
    livesim.BlindTrader.train = spying_train
    try:
        feed, trader, sealed, wall = L.hardened_run(cfg, "t1", log=lambda *a: None, check_parity=False, data=data, warmup_years=1)
    finally:
        livesim.BlindTrader.train = real_train
    assert seen["blocked"] is True and len(trader.session.days) > 200
    assert not [f for f in feed.audit() if f.severity == "fail"]
    srv = _server()
    socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2).close()   # the guard is gone after the run
    srv.close()
