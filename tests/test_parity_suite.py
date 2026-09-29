"""Bible Phase 2, all feature families (engine/parity_suite.py): candles, market context, fingerprints, macro with
publication lags, analog find, and the research-vs-live code path. Each family has a planted defect the suite must
catch, plus the empty/degenerate case. Synthetic data only (shared fixture in test_parity.py)."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import parity as P
from engine import parity_suite as S
from test_parity import inp  # noqa: F401  (module fixture)


@pytest.fixture(scope="module")
def suite_ok(inp):
    return S.run_suite(inp, n_dates=3, seed=0)


def test_every_family_passes_every_check(suite_ok):
    assert suite_ok.passed, suite_ok.summary()
    assert {"features", "candles", "market", "fingerprints", "macro", "analogs", "live-path"} <= {c.family for c in suite_ok.checks}
    assert {"strict-vs-batch", "future-invariance", "gappy-data", "universe-invariance", "publication-lag",
            "find-vs-truncated", "live-window"} <= {c.name for c in suite_ok.checks}
    assert not suite_ok.skipped and all(c.report for c in suite_ok.checks)


def test_suite_report_serialises(suite_ok):
    d = json.loads(json.dumps(suite_ok.to_dict(), default=str))
    assert d["passed"] and d["fingerprint"] == P.feature_code_fingerprint() and len(d["checks"]) == len(suite_ok.checks)


def bad_candles(inp, start):
    """candle feature using TOMORROW's open (a gap measured to the next session)."""
    X = S.candles_builder(inp, start)
    return X.join(S._panelise({"gap_next": inp.stocks["Open"].shift(-1) / inp.stocks["Close"] - 1}, start))


def bad_market(inp, start):
    """market context centred on the mean VIX of the whole loaded history."""
    X = S.market_builder(inp, start)
    X["m_vix_z"] = (X["m_vix"] - inp.market["Close"]["^VIX"].mean()).astype("float32")
    return X


@pytest.mark.parametrize("fam,b,col", [("candles", bad_candles, "gap_next"), ("market", bad_market, "m_vix_z")])
def test_planted_defect_per_family_is_caught(inp, fam, b, col):
    r = S.run_suite(inp, families=[fam], n_dates=3, seed=1, live=False, builders={fam: b})
    assert not r.passed
    by = {c.name: c for c in r.checks}
    assert not by["future-invariance"].passed and col in by["future-invariance"].detail, r.summary()
    if fam == "candles":                                   # t+1 read: strict truncation sees a NaN it should not
        assert not by["strict-vs-batch"].passed and col in by["strict-vs-batch"].detail


def test_future_invariance_catches_what_truncation_misses(inp):
    """Full-history statistic: the batch and the strict run each use the data they were given, so they may agree with
    each other only by luck; junking the future must move the value."""
    def b(i, start):
        X = S.candles_builder(i, start)
        X["cd_z"] = (X["cd_range"] / i.stocks["Close"].pct_change().abs().stack().mean()).astype("float32")
        return X
    fast = b(inp, inp.dates[30])
    days = P.sample_dates(fast.index.get_level_values(0).unique(), 3, 0)
    assert not S.future_invariance(b, inp, fast, days).passed
    good = S.candles_builder(inp, inp.dates[30])
    assert S.future_invariance(S.candles_builder, inp, good, days).passed


def test_macro_used_before_publication_is_caught(inp):
    """Truncation cannot see a missing publication lag (the observation is inside the window); the audit must."""
    from engine import analogs
    ok = S.audit_macro_publication(inp.macro, inp.dates, analogs.LAG, inp.dates[[340, 400, 500]])
    assert ok["lag_table"] == [] and ok["passed"]
    a = S.audit_macro_publication(inp.macro, inp.dates, dict(analogs.LAG, CPIAUCSL=1), inp.dates[[340]])
    assert not a["passed"] and any("CPIAUCSL" in x for x in a["lag_table"])
    missing = {k: v for k, v in analogs.LAG.items() if k != "UNRATE"}
    assert any("UNRATE" in x and "no LAG entry" in x
               for x in S.audit_macro_publication(inp.macro, inp.dates, missing, [])["lag_table"])
    assert S.cadence(inp.macro["CPIAUCSL"]) == "monthly" and S.cadence(inp.macro["NFCI"]) == "weekly" \
        and S.cadence(inp.macro["DGS10"]) == "daily"


def test_macro_value_audit_detects_a_shifted_column(inp):
    from engine import analogs
    with S.analogs_env(inp) as an:
        F, _ = an.fingerprints()
    days = list(F.index[[350, 420, 500]])
    assert S.audit_macro_publication(inp.macro, F.index, analogs.LAG, days, F)["passed"]
    F2 = F.copy()
    F2["mac_CPIAUCSL"] = F2["mac_CPIAUCSL"].shift(-20)                # value from 20 sessions in the future
    a = S.audit_macro_publication(inp.macro, F.index, analogs.LAG, days, F2)
    assert not a["passed"] and any("mac_CPIAUCSL" in x for x in a["values"])


def test_macro_family_fails_when_lag_removed_in_the_real_code(inp, monkeypatch):
    from engine import analogs
    monkeypatch.setitem(analogs.LAG, "CPIAUCSL", 1)
    r = S.run_suite(inp, families=["macro"], n_dates=3, live=False)
    assert any(c.name == "publication-lag" and not c.passed for c in r.checks)


def test_universe_dependence_is_caught(inp):
    """The 28-Sep insider-style defect: a per-stock feature normalised by a statistic of whichever stocks are loaded."""
    def b(i, start):
        X = S.candles_builder(i, start)
        avg = (i.stocks["High"] / i.stocks["Low"] - 1).mean(axis=1)
        X["cd_rel_univ"] = (X["cd_range"] - avg.reindex(X.index.get_level_values(0)).values).astype("float32")
        return X
    bad = S.universe_invariance(b, inp, inp.dates[-1], 0.5, 0)
    assert not bad.passed and "cd_rel_univ" in bad.failing_features
    assert S.universe_invariance(S.candles_builder, inp, inp.dates[-1], 0.5, 0).passed
    assert S.universe_invariance(b, inp, inp.dates[-1], 0.5, 0, ignore=("cd_rel_*",)).passed   # exempt only when named


def test_gappy_inputs_still_reproduce(inp):
    for fam in ("features", "candles"):
        r = S.gappy_parity(S.FAMILIES[fam][0], inp, S.FAMILIES[fam][1], 3, 2)
        assert r.passed, r.summary()
    g = inp.with_gaps(seed=1, frac=0.02, late_ipo=100, delist=400)
    C = g.stocks["Close"]
    assert C.iloc[:100, 0].isna().all() and C.iloc[400:, -1].isna().all() and C.isna().to_numpy().mean() > 0.01


def test_perturb_future_leaves_the_past_alone(inp):
    t = inp.dates[400]
    p = inp.perturb_future(t, 3)
    for k in inp.stocks:
        assert inp.stocks[k].loc[:t].equals(p.stocks[k].loc[:t])
        assert not inp.stocks[k].loc[t:].iloc[1:].equals(p.stocks[k].loc[t:].iloc[1:])
    assert inp.macro.loc[:t].equals(p.macro.loc[:t]) and not inp.macro.loc[t:].iloc[1:].equals(p.macro.loc[t:].iloc[1:])
    assert (p.stocks["High"].to_numpy() >= p.stocks["Low"].to_numpy()).all()


def test_analog_find_parity_catches_an_engine_that_saw_the_future(inp, monkeypatch):
    from engine import analogs
    days = inp.dates[[350, 450]]
    assert S.analog_find_parity(inp, days).passed
    orig = analogs.Analogs.find

    def leaky(self, t, k=5):
        r = orig(self, t, k)
        if r is not None:                                # uniqueness leaks NEXT day's 1-month return
            i = self.F.index.get_loc(t)
            r["uniqueness"] += float(self.F.iloc[i + 1]["ret_1m"]) if i + 1 < len(self.F) else 0.0
        return r
    monkeypatch.setattr(analogs.Analogs, "find", leaky)
    assert not S.analog_find_parity(inp, days).passed


def test_analog_find_short_history_returns_none_consistently(inp):
    r = S.analog_find_parity(inp, [inp.dates[150]])
    assert r.passed and r.per_feature["none"].max_abs == 0.0


def test_live_path_matches_research_and_a_drifting_live_path_is_caught(inp, monkeypatch):
    days = inp.dates[[400, 519]]
    fast = P.default_builder(inp, inp.dates[260])
    out = S.live_vs_research(inp, days, fast)
    assert out["same_inputs"].passed, out["same_inputs"].summary()
    assert out["research_vs_live"].passed, out["research_vs_live"].summary()
    from engine import live, features
    real = features.build

    def drifted(stocks, market, ev, ins, sic, start="2013-01-01", **kw):
        X, atr = real(stocks, market, ev, ins, sic, start=start, **kw)
        X["r5"] = X["r5"] * 1.01                         # the live code path quietly changed
        return X, atr
    import types
    monkeypatch.setattr(live, "features", types.SimpleNamespace(build=drifted))   # live only; research keeps the real one
    bad = S.live_vs_research(inp, days, fast)
    assert not bad["same_inputs"].passed and "r5" in bad["same_inputs"].failing_features
    assert not bad["research_vs_live"].passed


def test_suite_degenerate_inputs_do_not_pass():
    tiny = pd.bdate_range("2020-01-01", periods=5)
    z = pd.DataFrame(1.0, index=tiny, columns=["A", "B"])
    e = P.Inputs(dict(Open=z, High=z, Low=z, Close=z, Volume=z),
                 {"Close": pd.DataFrame({"SPY": 1.0, "^VIX": 1.0}, index=tiny)},
                 pd.DataFrame({"kind": [], "accepted": [], "ticker": [], "form": []}), None,
                 pd.DataFrame({"ticker": ["A", "B"], "sic": ["1", "2"]}))
    assert not S.run_suite(e, families=["candles", "features"], n_dates=2, live=False).passed   # nothing comparable
    assert not S.SuiteReport().passed                                                          # an empty suite is no pass
