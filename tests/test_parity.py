"""Bible Phase 2: the parity firewall. Synthetic bars only. The honest features must reproduce themselves; planted
look-ahead, universe/history dependence, NaN and dtype defects must each be caught."""
import numpy as np
import pandas as pd
import pytest

from engine import parity as P
from engine import features as F

N_DAYS, N_TICK = 520, 10


@pytest.fixture(scope="module")
def inp():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-01", periods=N_DAYS)
    tick = [f"T{i}" for i in range(N_TICK)]
    ret = rng.normal(0.0003, 0.02, (N_DAYS, N_TICK))
    C = pd.DataFrame(50 * np.exp(np.cumsum(ret, 0)), index=dates, columns=tick)
    O = C.shift(1).fillna(C.iloc[0]) * (1 + rng.normal(0, 0.004, C.shape))
    H = np.maximum(O, C) * (1 + np.abs(rng.normal(0, 0.006, C.shape)))
    L = np.minimum(O, C) * (1 - np.abs(rng.normal(0, 0.006, C.shape)))
    V = pd.DataFrame(rng.uniform(2e6, 6e6, C.shape), index=dates, columns=tick)
    stocks = dict(Open=O, High=H, Low=L, Close=C, Volume=V)
    m = pd.DataFrame({"SPY": 300 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, N_DAYS))),
                      "^VIX": 23 + rng.normal(0, 2, N_DAYS).cumsum() * 0.2}, index=dates)
    market = {"Close": m}
    acc = [pd.Timestamp(dates[i]).tz_localize("America/New_York") + pd.Timedelta(hours=h)
           for i, h in ((300, 8), (330, 17), (380, 9))]
    ev = pd.DataFrame({"kind": ["EARN", "SHELF", "EARN"], "accepted": acc, "ticker": ["T1", "T2", "T3"],
                       "form": ["8-K", "S-3", "8-K"]})
    sic = pd.DataFrame({"ticker": tick, "sic": [f"{2000 + 10 * (i % 3)}" for i in range(N_TICK)]})
    cal = pd.date_range(dates[0] - pd.Timedelta(days=200), dates[-1], freq="D")
    mo = pd.date_range(cal[0].replace(day=1), cal[-1], freq="MS")
    macro = pd.concat([
        pd.Series(3 + rng.normal(0, 0.05, len(cal)).cumsum(), index=cal, name="DGS10"),
        pd.Series(250 + rng.normal(0.3, 0.5, len(mo)).cumsum(), index=mo, name="CPIAUCSL"),
        pd.Series(5 + rng.normal(0, 0.1, len(mo)).cumsum(), index=mo, name="UNRATE"),
        pd.Series(rng.normal(0, 1, len(cal[::7])), index=cal[::7], name="NFCI")], axis=1).sort_index()
    return P.Inputs(stocks, market, ev, None, sic, macro)


@pytest.fixture(scope="module")
def fast(inp):
    return P.default_builder(inp, inp.dates[260])


def leaky_builder(kind):
    def b(inp, start):
        X = P.default_builder(inp, start)
        C = inp.stocks["Close"]
        if kind == "future":               # peeks one session ahead
            s = np.log(C.shift(-1) / C)
        else:                              # standardises with statistics of the WHOLE loaded history
            r = np.log(C / C.shift(1))
            s = (r - r.stack().mean()) / r.stack().std()
        s = s.stack(future_stack=True).rename("leak").astype("float32")
        s.index.names = ["date", "ticker"]
        return X.join(s)
    return b


def test_honest_features_pass_strict_and_live(inp, fast):
    r = P.run_parity(inp, fast=fast, n_dates=4, seed=1)
    assert r.passed, r.summary()
    assert r.n_compared > 4 * N_TICK and r.max_abs <= P.GATE_TOL
    rl = P.run_parity(inp, fast=fast, n_dates=3, seed=2, mode="live")
    assert rl.passed, rl.summary()
    P.require_parity(r)


@pytest.mark.parametrize("kind", ["future", "fullsample"])
def test_planted_lookahead_is_caught(inp, kind):
    r = P.run_parity(inp, builder=leaky_builder(kind), n_dates=3, seed=4)
    assert not r.passed
    assert r.failing_features == ["leak"], r.summary()
    with pytest.raises(P.ParityFailure):
        P.require_parity(r, label="miner")


def test_tolerance_cannot_be_loosened(inp, fast):
    r = P.run_parity(inp, fast=fast, n_dates=1, seed=0)
    with pytest.raises(ValueError):
        P.require_parity(r, tol=1e-2)
    loose = P.run_parity(inp, fast=fast, n_dates=1, seed=0, tol=1e-2)
    with pytest.raises(P.ParityFailure):
        P.require_parity(loose)


def test_compare_detects_nan_missing_and_value_drift(fast):
    a = fast.iloc[:60].copy()
    b = a.copy()
    assert P.compare_panels(a, b).passed
    b.iloc[3, 2] = np.nan
    assert P.compare_panels(a, b).per_feature[a.columns[2]].n_nan_mismatch == 1
    b = a.copy()
    b.iloc[5, 4] = b.iloc[5, 4] + 1e-3
    r = P.compare_panels(a, b)
    assert r.failing_features == [a.columns[4]] and r.max_abs == pytest.approx(1e-3, rel=0.1)
    b = a.copy()
    b.iloc[7, 4] = b.iloc[7, 4] + 5e-5                 # inside the gate
    assert P.compare_panels(a, b).passed
    assert not P.compare_panels(a, a.iloc[1:]).passed                 # row missing from one side
    assert not P.compare_panels(a, a.drop(columns=a.columns[0])).passed
    bi = a.copy()
    bi.iloc[2, 1] = np.inf
    assert P.compare_panels(a, bi).per_feature[a.columns[1]].n_inf_mismatch == 1


def test_empty_and_degenerate_fail_closed(inp):
    e = pd.DataFrame()
    r = P.compare_panels(e, e)
    assert not r.passed and "empty" in r.error
    with pytest.raises(P.ParityFailure):
        P.require_parity(r)
    assert P.sample_dates([], 5) == []
    three = pd.bdate_range("2020-01-01", periods=3)
    assert P.sample_dates(three, 10, 0) == list(three)
    crash = P.run_parity(inp, builder=lambda i, s: 1 / 0, n_dates=1)
    assert not crash.passed and "ZeroDivisionError" in crash.error


def test_sampling_is_seeded_and_keeps_last_date():
    d = pd.bdate_range("2020-01-01", periods=300)
    a, b = P.sample_dates(d, 6, 9, warmup=100), P.sample_dates(d, 6, 9, warmup=100)
    assert a == b and len(a) == 6 and d[-1] in a and min(a) >= d[100]
    assert P.sample_dates(d, 6, 10, warmup=100) != a


def test_truncation_hides_the_future_and_late_filings(inp):
    t = inp.dates[310]
    tr = inp.truncate(t)
    assert tr.stocks["Close"].index[-1] == t and tr.market["Close"].index[-1] == t
    assert len(tr.ev) == 1 and tr.ev.iloc[0]["ticker"] == "T1"       # the day-330 and day-380 filings are future
    lb = inp.truncate(t, lookback_days=100)
    assert (t - lb.stocks["Close"].index[0]).days <= 100


def test_report_roundtrip_and_cache_invalidation(inp, fast, tmp_path, monkeypatch):
    r = P.run_parity(inp, fast=fast, n_dates=2)
    p = P.save_report(r, tmp_path / "sub" / "parity.json")
    assert P.cached_pass(p)
    monkeypatch.setattr(P, "feature_code_fingerprint", lambda module=F: "changed")
    assert not P.cached_pass(p)                       # feature code changed -> pass no longer valid
    assert not P.cached_pass(tmp_path / "absent.json")


def test_contracts_infer_and_catch_defects(fast):
    cur = fast.iloc[len(fast) // 2:]
    ct = P.infer_contracts(fast)               # contracts come from the accepted panel; a later slice must satisfy them
    assert P.check_contracts(cur, ct) == []
    bad = cur.copy()
    bad["r5"] = bad["r5"].astype("float64")
    bad["vol20"] = np.inf
    bad.iloc[0:5, bad.columns.get_loc("r20")] = 99.0
    bad["m_vix"] = bad["m_vix"] + np.arange(len(bad), dtype="float32")   # varies within a date
    bad = bad.drop(columns=["r1"])
    v = " | ".join(P.check_contracts(bad, ct))
    for needle in ("r5: dtype", "vol20: ", "r20: range", "m_vix: not constant", "r1: contracted feature missing",
                   "infinite"):
        assert needle in v
    assert P.check_contracts(cur.reset_index(), ct)[0].startswith("index must be")
    dup = pd.concat([cur, cur.iloc[:2]])
    assert any("duplicate" in x for x in P.check_contracts(dup, ct))


def test_drift_flags_a_shifted_feature_only(fast):
    half = len(fast) // 2
    ref, cur = fast.iloc[:half].copy(), fast.iloc[half:].copy()
    cur["r5"] = cur["r5"] + 0.5
    cur["vol20"] = np.nan
    t = P.drift_report(ref, cur).set_index("feature")
    assert t.loc["r5", "status"] == "alarm" and t.loc["vol20", "status"] == "alarm"
    same = P.drift_report(ref, ref.sample(frac=0.7, random_state=0)).set_index("feature")
    assert (same["status"] != "alarm").all() and (same["psi"].dropna() < 0.25).all()   # a resample is not drift
    assert P.drift_report(ref, cur.drop(columns=["r1"])).set_index("feature").loc["r1", "status"] == "missing"
    assert P.psi(np.ones(50), np.ones(50)) == 0.0 and P.psi(np.ones(50), np.zeros(50)) > 1
    assert np.isnan(P.psi(np.array([]), np.ones(3)))


def test_firewall_end_to_end(inp):
    out = P.firewall(inp, n_dates=2, seed=0)
    assert out["passed"], out["strict"].summary()
    assert isinstance(out["drift"], pd.DataFrame)
    assert not P.firewall(inp, builder=leaky_builder("future"), n_dates=2, live=False)["passed"]
