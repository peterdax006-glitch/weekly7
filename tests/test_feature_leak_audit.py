"""F09 feature future-leak audit (C69 ledger section 5; canon C56). Synthetic inputs only.

The audit must FAIL on every planted leaking operation (panel-wide z-score, full-sample winsor/min-max/rank, backfill, centred window,
negative shift, a future-dependent row set), pass on the fixed production set, never call an empty or all-NaN comparison CLEAN, and
each fixed feature keeps a truncation test of its own."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("feature_leak_audit", ROOT / "scripts" / "feature_leak_audit.py")
A = importlib.util.module_from_spec(_spec)
sys.modules["feature_leak_audit"] = A
_spec.loader.exec_module(A)


@pytest.fixture(scope="module")
def inp():
    return A.planted_inputs(n_days=320, n_tickers=16, seed=3)


@pytest.fixture(scope="module")
def run(inp):
    return A.run_audit(inp, A.PRODUCTION, n_cuts=4, source="planted")


def _statuses(r):
    return {f.feature: f.status for f in r.features}


# ------------------------------------------------------------------ the canary: planted leaks are caught, honest columns are not
def test_canary_flags_every_planted_leak_and_clears_honest_columns(run):
    assert run.canary_ok
    can = next(r for r in run.results if r.planted)
    st = _statuses(can)
    leaks = [c for c in st if c.startswith("leak_")]
    assert len(leaks) == 7 and all(st[c] == "LEAK" for c in leaks), st
    assert all(st[c] == "CLEAN" for c in st if c.startswith("honest_")), st


def test_blind_canary_fails_the_run(inp):
    honest = A.Builder("blind", lambda i, T, s: A._long({"leak_not_really": i.stocks["Close"].pct_change(fill_method=None)}, s),
                       ("x:y",), "test", planted=True)
    r = A.run_audit(inp, [], n_cuts=3, canary=honest)
    assert not r.canary_ok and not r.passed
    assert "leak_not_really came out CLEAN" in r.results[0].error


@pytest.mark.parametrize("kind", ["zscore", "winsor", "bfill", "centred", "shift", "scaler"])
def test_a_planted_leak_inside_a_production_builder_is_caught_and_located(inp, kind):
    base = A.b_features("default")

    def leaky(i, T, start):
        X = base(i, T, start)
        C = i.stocks["Close"]
        r = np.log(C / C.shift(1))
        if kind == "zscore":
            w = (r - r.stack().mean()) / r.stack().std()
        elif kind == "winsor":
            lo, hi = r.stack().quantile([0.02, 0.98])
            w = r.clip(lo, hi)
        elif kind == "bfill":
            w = r.where(np.repeat((np.arange(len(r)) % 7 == 0)[:, None], r.shape[1], 1)).bfill()
        elif kind == "centred":
            w = r.rolling(3, center=True, min_periods=1).mean()
        elif kind == "shift":
            w = r.shift(-2)
        else:
            w = (r - r.min()) / (r.max() - r.min())
        s = w.stack(future_stack=True).rename("planted").astype("float64")
        s.index.names = ["date", "ticker"]
        return X.join(s)
    b = A.Builder("leaky", leaky, ("engine.features:build", "test_feature_leak_audit:test_a_planted_leak_inside_a_production_builder_is_caught_and_located"),
                  "test")
    res = A.audit_builder(b, inp, A.choose_cuts(inp.dates, 4, A._start(inp)))
    st = _statuses(res)
    assert st["planted"] == "LEAK", (kind, st["planted"])
    assert sum(v == "LEAK" for v in st.values()) == 1, [k for k, v in st.items() if v == "LEAK"]      # nothing honest is dragged along
    assert res.status == "LEAK"
    where = next(f.where for f in res.features if f.feature == "planted")
    assert "tests/test_feature_leak_audit.py:" in where and "suspect" in where, where


def test_a_future_dependent_row_set_is_a_row_leak(inp):
    """Rows kept only for tickers that are still listed at the END of the data: a survivor filter the truncated build cannot reproduce."""
    base = A.b_features("default")

    def survivors(i, T, start):
        X = base(i, T, start)
        alive = i.stocks["Close"].iloc[-1].notna()
        return X[X.index.get_level_values(1).isin(alive[alive].index)]
    res = A.audit_builder(A.Builder("survivor", survivors, ("engine.features:build",), "test"), inp,
                          A.choose_cuts(inp.dates, 4, A._start(inp)))
    assert res.rows_leaked > 0 and res.status == "LEAK"


# ------------------------------------------------------------------ the fixed production set passes
def test_production_builders_are_clean_except_the_open_w02_leak(run):
    bad = {(f.builder, f.feature) for f in run.leaks}
    assert bad <= {("feeds.market_proxy", "^VIX"), ("feeds.market_proxy", "^VIX3M")}, bad
    for r in run.production:
        if r.builder != "feeds.market_proxy":
            assert r.status == "OK", (r.builder, r.status, r.error, r.rows_leaked)
            assert r.rows_leaked == 0
    counts = run.counts()
    assert counts.get("CLEAN", 0) > 350 and counts.get("LABEL", 0) == 6


def test_market_proxy_is_clean(run):
    r = next(r for r in run.results if r.builder == "feeds.market_proxy")
    assert r.status == "OK"


def test_every_owned_builder_is_listed_with_its_source(run):
    names = {r.builder for r in run.production}
    assert {"features.build[default]", "features.build[relative]", "features.build[split_invariant]", "fv_pipeline.build_panel.X",
            "volatility_lab.frame_from_panel", "volatility_lab.event_inputs", "direction_lab.derive_features"} <= names
    for b in A.PRODUCTION:
        for src in b.sources:
            A._resolve(src)                                  # every named source exists (the locator can find it)


# ------------------------------------------------------------------ truncation tests of each fixed feature
def test_filing_n5_before_a_late_first_filing_does_not_see_it():
    from engine.research import volatility_lab as VL
    days = pd.bdate_range("2020-01-01", periods=40)
    idx = pd.MultiIndex.from_product([days, ["AAA", "LATE"]], names=["date", "ticker"])
    acc = pd.to_datetime([days[5] + pd.Timedelta(hours=10), days[30] + pd.Timedelta(hours=10)]).tz_localize("America/New_York")
    ev = pd.DataFrame({"ticker": ["AAA", "LATE"], "accepted": acc, "kind": ["EARN", "EARN"], "form": ["8-K", "8-K"]})
    T = days[20]
    full = VL.event_inputs(idx, ev, None)
    cut = VL.event_inputs(idx[idx.get_level_values(0) <= T], ev[ev["accepted"] < (T + pd.Timedelta(days=1)).tz_localize("America/New_York")], None)
    a = full.loc[full.index.get_level_values(0) <= T, "filing_n5"]
    assert a.equals(cut["filing_n5"]), pd.concat([a, cut["filing_n5"]], axis=1)[a != cut["filing_n5"]]
    assert (a.xs("LATE", level=1) == 0).all()                # no filing yet is 0, exactly as when the table ends before it
    assert full.xs("LATE", level=1).loc[days[31]:days[33], "filing_n5"].eq(1).all()
    assert VL.event_inputs(idx, None, None)["filing_n5"].isna().all()       # no table stays NaN, never 0


def test_direction_lab_fixed_centres_are_truncation_invariant(inp):
    """R08's two fixes (vix_x_r5, path_efficiency: panel medians replaced by fixed centres) stay point-in-time."""
    b = A.Builder("dl", A.b_direction_lab, ("engine.research.direction_lab:derive_features",), "test")
    res = A.audit_builder(b, inp, A.choose_cuts(inp.dates, 3, A._start(inp)))
    st = _statuses(res)
    assert st["vix_x_r5"] == "CLEAN" and st["path_efficiency"] == "CLEAN", st


def test_a_panel_median_centre_would_be_caught(inp):
    """The pre-fix form of vix_x_r5 (centred on the panel-wide median) is a LEAK - the check can fail on exactly that defect."""
    base = A.b_features("relative")

    def old_vix_x_r5(i, T, start):
        X = base(i, T, start)
        return pd.DataFrame({"vix_x_r5_old": (X["m_vix"] - X["m_vix"].median()) * X["r5"]})
    res = A.audit_builder(A.Builder("old", old_vix_x_r5, ("engine.research.direction_lab:derive_features",), "test"), inp,
                          A.choose_cuts(inp.dates, 3, A._start(inp)))
    assert _statuses(res)["vix_x_r5_old"] == "LEAK"


# ------------------------------------------------------------------ empty and degenerate cases
def test_empty_builder_is_never_clean(inp):
    res = A.audit_builder(A.Builder("empty", lambda i, T, s: pd.DataFrame(), ("x:y",), "t"), inp, A.choose_cuts(inp.dates, 2, A._start(inp)))
    assert res.status == "EMPTY" and not res.features
    r = A.AuditRun("planted", [], [res], canary_ok=True, started="x")
    assert not r.passed


def test_all_nan_column_is_untested_not_clean(inp):
    def nanb(i, T, s):
        X = A.b_features("default")(i, T, s)[["r1"]].copy()
        X["always_nan"] = np.nan
        return X
    res = A.audit_builder(A.Builder("nan", nanb, ("x:y",), "t"), inp, A.choose_cuts(inp.dates, 2, A._start(inp)))
    st = _statuses(res)
    assert st["always_nan"] == "UNTESTED" and st["r1"] == "CLEAN"


def test_no_cuts_means_nothing_is_clean(inp):
    res = A.audit_builder(A.Builder("f", A.b_features("default"), ("engine.features:build",), "t"), inp, [])
    assert res.features and all(f.status == "UNTESTED" for f in res.features)
    assert A.choose_cuts(inp.dates[:3], 4, inp.dates[0]) == [] and A.choose_cuts(inp.dates, 0, inp.dates[0]) == []


def test_crashing_builder_is_an_error_and_fails_the_run(inp):
    def boom(i, T, s):
        if len(i.dates) < len(inp.dates):
            raise RuntimeError("cannot build on less data")
        return A.b_features("default")(i, T, s)
    res = A.audit_builder(A.Builder("boom", boom, ("x:y",), "t"), inp, A.choose_cuts(inp.dates, 2, A._start(inp)))
    assert res.status == "ERROR" and "cannot build" in res.error
    assert not A.AuditRun("planted", [], [res], True, "x").passed


def test_weekly_boundary_rows_are_not_leaks(inp):
    fridays = [d for d in inp.dates if d.dayofweek == 4 and d > pd.Timestamp(A._start(inp)) and d < inp.dates[-10]][:2]
    res = A.audit_builder(A.Builder("fv", A.b_fv_X, ("engine.fv_pipeline:build_panel",), "t"), inp, fridays)
    assert res.rows_boundary > 0 and res.rows_leaked == 0 and res.status == "OK"


# ------------------------------------------------------------------ locator, report, CLI
def test_suspect_lines_ignore_docstrings_and_find_operations():
    hits = A.suspect_lines("feature_leak_audit:b_planted")
    ops = {op for _, _, op, _ in hits}
    assert {"backfill", "centred window", "negative shift", "whole-sample statistic"} <= ops
    assert all("canary" not in code.lower() for _, _, _, code in hits)          # the docstring line is not code
    assert A.suspect_lines("no.such.module:fn") == []


def test_market_proxy_no_longer_backfills():
    ops = {op for _, _, op, _ in A.suspect_lines("engine.research.feeds:market_proxy")}
    assert "backfill" not in ops                                    # the F09 leak (.bfill() over the warm-up) stays fixed


def test_report_has_a_row_per_feature_and_the_fix_tables(run, tmp_path):
    md = A.render_report([run], A.FIXED)
    n_feat = sum(len(r.features) for r in run.results)
    assert md.count("\n| ") >= n_feat
    assert "Before / after" in md and "volatility_lab.event_inputs/filing_n5" in md and ("Open leaks" in md) == bool(A.OPEN)
    assert "What truncation cannot see" in md
    p = tmp_path / "r.json"
    import json
    p.write_text(json.dumps(run.as_dict(), default=str))
    back = A.load_run(p)
    assert back.counts() == run.counts() and back.canary_ok == run.canary_ok


def test_cli_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "planted_inputs", lambda seed=0: _small(seed))
    assert A.main(["--only", "candles.build", "--cuts", "2", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists() and (tmp_path / "results_planted.json").exists()
    leaky = A.Builder("planted_test.leaky_proxy", lambda inp:
                      A.b_market_proxy(inp).iloc[::-1].expanding().mean().iloc[::-1],
                      ("engine.research.feeds:market_proxy",), "test")      # a reverse-expanding mean reads the future
    monkeypatch.setattr(A, "PRODUCTION", list(A.PRODUCTION) + [leaky])
    assert A.main(["--only", "leaky_proxy", "--cuts", "3", "--no-report"]) == 1
    assert A.main(["--only", "leaky_proxy", "--cuts", "3", "--no-report", "--fail-on", "never"]) == 0
    assert A.main(["--only", "feeds.market_proxy", "--cuts", "3", "--no-report"]) == 0


_ORIG_PLANTED = A.planted_inputs


def _small(seed):
    return _ORIG_PLANTED(n_days=260, n_tickers=10, seed=seed)
