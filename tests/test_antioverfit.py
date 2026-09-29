"""Bible Phase 26: the battery must pass a clean pipeline and CATCH each planted defect (synthetic panels only)."""
import json

import numpy as np
import pandas as pd
import pytest

from engine import antioverfit as ao


def make_panel(seed=0, n_dates=240, n_tk=40, signal=(0.10, 0.06), n_feat=6, regime_flip=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    tks = [f"T{i:03d}" for i in range(n_tk)]
    idx = pd.MultiIndex.from_product([dates, tks], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), n_feat)), index=idx, columns=[f"f{i}" for i in range(n_feat)])
    vix = pd.Series(rng.standard_normal(n_dates), index=dates)
    X["m_vix"] = vix.reindex(idx.get_level_values(0)).values
    sign = np.where(X["m_vix"].values > np.quantile(vix, 0.66), -1.0, 1.0) if regime_flip else 1.0
    y = sign * (signal[0] * X["f0"] + signal[1] * X["f1"]) * 0.05 + 0.05 * rng.standard_normal(len(idx))
    return X, pd.Series(y.values, index=idx)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


NOW = pd.Timestamp("2015-01-01") + pd.offsets.BDay(239)


def status(rep, name):
    return next(v for v in rep["verdicts"] if v["name"] == name)["status"]


# ------------------------------------------------------------ clean pipeline
def test_clean_pipeline_passes_everything(panel):
    X, y = panel
    rep = ao.run_battery(ao.wf_evaluate, X, y, NOW, seed=3, n_rep=8)
    assert rep["real"]["metric"] > 0.03 and rep["real"]["t"] > 2
    assert rep["overall"] == "pass", ao.render_text(rep)
    for n in ("future_scramble", "label_permutation", "date_disguise", "walk_forward", "randomized_outcomes",
              "dead_feature_injection", "duplicate_feature_injection", "ticker_permutation"):
        assert status(rep, n) == "pass", (n, ao.render_text(rep))
    assert status(rep, "feature_shuffle") in ("pass", "flag")


def test_battery_is_deterministic_and_serialisable(panel, tmp_path):
    X, y = panel
    kw = dict(tests=("label_permutation", "randomized_outcomes", "feature_shuffle"), n_rep=5)
    a = ao.run_battery(ao.wf_evaluate, X, y, NOW, seed=11, **kw)
    b = ao.run_battery(ao.wf_evaluate, X, y, NOW, seed=11, **kw)
    assert ao._jsonable(a["verdicts"]) == ao._jsonable(b["verdicts"])
    p = ao.write_report(a, tmp_path / "r" / "rep.json")
    assert json.loads(p.read_text())["overall"] == a["overall"]
    assert "label_permutation" in ao.render_text(a)


# ------------------------------------------------------------ planted defects
def cherry_evaluator(X, y, now):
    """Defect: picks the best feature by IN-SAMPLE correlation on all rows and reports it (selection on the test set)."""
    cols = ao.feature_cols(X)
    ok = y.notna().values
    cs = {c: float(np.corrcoef(X[c].values[ok], y.values[ok])[0, 1]) for c in cols}
    best = max(cs.values())
    return {"metric": best, "t": best / 0.014, "importance": {c: 1.0 / len(cols) for c in cols}}


def test_label_permutation_catches_selection_on_the_test_set(panel):
    X = panel[0].iloc[:, :8]
    X = pd.concat([X, pd.DataFrame(np.random.default_rng(1).standard_normal((len(X), 12)), index=X.index,
                                   columns=[f"n{i}" for i in range(12)])], axis=1)
    y = panel[1]
    rep = ao.run_battery(cherry_evaluator, X, y, NOW, n_rep=10, tests=("label_permutation", "randomized_outcomes"))
    assert status(rep, "label_permutation") == "fail"
    assert status(rep, "randomized_outcomes") == "fail"
    assert rep["overall"] == "fail"


def test_future_scramble_catches_an_evaluator_that_ignores_now(panel):
    X, y = panel

    def leaky(X, y, now):
        return ao.wf_evaluate(X, y, X.index.get_level_values(0).max())   # reads the future
    rep = ao.run_battery(leaky, X, y, NOW, tests=("future_scramble",))
    assert status(rep, "future_scramble") == "fail"
    assert ao.run_battery(ao.wf_evaluate, X, y, NOW, tests=("future_scramble",))["overall"] == "pass"


def test_future_scramble_catches_outlier_trimming_on_all_rows(panel):
    X, y = panel

    def trimmed(X, y, now):
        """Defect: winsorises labels at a quantile computed over ALL rows, future included, before honouring `now`."""
        return ao.wf_evaluate(X, y.where(y <= y.quantile(0.9)), now)
    rep = ao.run_battery(trimmed, X, y, NOW, tests=("future_scramble",))
    assert status(rep, "future_scramble") == "fail"


def test_date_disguise_catches_calendar_dependence(panel):
    X, y = panel
    ev = lambda X, y, now: {**ao.wf_evaluate(X, y, now),
                            "metric": ao.wf_evaluate(X, y, now)["metric"] + 0.01 * np.sin(pd.Timestamp(now).value / 1e17)}
    assert status(ao.run_battery(ev, X, y, NOW, tests=("date_disguise",)), "date_disguise") == "fail"


def test_ticker_permutation_catches_identity_dependence(panel):
    X, y = panel

    def ident(X, y, now):
        r = ao.wf_evaluate(X, y, now)
        return {**r, "metric": r["metric"] + 0.01 * float(y.xs("T000", level=1).mean())}
    rep = ao.run_battery(ident, X, y, NOW, tests=("ticker_permutation",))
    v = rep["verdicts"][0]
    assert v["status"] == "fail" and "identity" in v["reason"]


def test_duplicate_injection_catches_double_counting(panel):
    X, y = panel

    def naive(X, y, now):
        """Defect: evidence = sum over features of |t|, so every copy of a feature adds its full evidence again."""
        cols = ao.feature_cols(X)
        base = ao.wf_evaluate(X, y, now)
        ts = [abs(float(np.corrcoef(X[c].values, y.fillna(0).values)[0, 1])) * np.sqrt(len(X)) for c in cols]
        return {**base, "evidence": float(sum(ts))}
    rep = ao.run_battery(naive, X, y, NOW, tests=("duplicate_feature_injection",))
    assert rep["verdicts"][0]["status"] == "fail" and "double-count" in rep["verdicts"][0]["reason"]


def test_dead_feature_injection_catches_in_sample_overfit():
    X, y = make_panel(seed=4, n_dates=60, n_tk=30, signal=(0.10, 0.06))

    def insample_ols(X, y, now):
        cols = [c for c in X.columns if not str(c).startswith("m_")]
        A = np.column_stack([X[cols].values, np.ones(len(X))])
        b = np.linalg.lstsq(A, y.values, rcond=None)[0]
        p = pd.Series(A @ b, index=X.index)
        ic = pd.Series({d: np.corrcoef(p[d], y[d])[0, 1] for d in p.index.get_level_values(0).unique()})
        return {"metric": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                "importance": {c: 1 / len(cols) for c in cols}}
    real = insample_ols(X, y, NOW)
    v = ao.dead_feature_injection(insample_ols, X, y, NOW, real, np.random.default_rng(0), n_dead=30, n_rep=4)
    assert v["status"] == "fail"
    good = ao.dead_feature_injection(ao.wf_evaluate, *make_panel(seed=4), NOW, ao.wf_evaluate(*make_panel(seed=4), NOW),
                                     np.random.default_rng(0), n_dead=9, n_rep=4)
    assert good["status"] == "pass", good["reason"]


def test_feature_shuffle_flags_unattributed_signal(panel):
    X, y = panel

    def liar(X, y, now):
        """Defect: says f5 is the important feature while the score actually comes from f0/f1."""
        r = ao.wf_evaluate(X, y, now)
        return {**r, "importance": {"f5": 0.9, "f4": 0.1}}
    v = ao.run_battery(liar, X, y, NOW, tests=("feature_shuffle",), n_rep=4)["verdicts"][0]
    assert v["status"] == "fail" and v["retained"] > 0.5


def test_regime_split_catches_a_sign_flip_regime():
    X, y = make_panel(seed=2, signal=(0.20, 0.12), regime_flip=True)
    rep = ao.run_battery(ao.wf_evaluate, X, y, NOW, tests=("regime_split",))
    v = rep["verdicts"][0]
    assert v["status"] == "fail" and "negative" in v["reason"]
    v2 = ao.run_battery(ao.wf_evaluate, *make_panel(seed=2), NOW, tests=("regime_split",))["verdicts"][0]
    assert v2["status"] == "pass"


def test_no_signal_is_inconclusive_not_a_pass():
    X, y = make_panel(seed=9, signal=(0.0, 0.0))
    rep = ao.run_battery(ao.wf_evaluate, X, y, NOW, tests=("feature_shuffle",))
    assert rep["verdicts"][0]["status"] == "inconclusive"
    lp = ao.run_battery(ao.wf_evaluate, X, y, NOW, tests=("label_permutation",), n_rep=8)["verdicts"][0]
    assert lp["status"] == "not_significant"        # clean pipeline, but the real result is not distinguishable from noise


# ------------------------------------------------------------ walk-forward audit
def test_walk_forward_verifier_flags_overlap_and_missing_gap():
    dates = pd.bdate_range("2020-01-01", periods=200)
    good = ao.walk_forward_splits(dates, 4, horizon=5)
    assert good and ao.verify_walk_forward(good, dates) == []
    for s in good:                               # test starts 5 sessions after training ends
        gap = np.searchsorted(dates.values, np.datetime64(s["test_start"])) - np.searchsorted(dates.values, np.datetime64(s["train_end"])) - 1
        assert gap >= 5
    bad = [dict(good[0], train_end=good[0]["test_start"])]
    assert "not before test start" in ao.verify_walk_forward(bad, dates)[0]
    tight = [dict(good[0], train_end=dates[np.searchsorted(dates.values, np.datetime64(good[0]["test_start"])) - 2])]
    assert "sessions between" in ao.verify_walk_forward(tight, dates)[0]
    assert ao.verify_walk_forward([], dates) == ["no splits"]


def test_walk_forward_test_fails_for_unauditable_or_leaky_splits(panel):
    X, y = panel
    silent = lambda X, y, now: {"metric": 0.1, "t": 3.0}
    assert ao.run_battery(silent, X, y, NOW, tests=("walk_forward",))["verdicts"][0]["status"] == "fail"

    def no_embargo(X, y, now):
        return {**ao.wf_evaluate(X, y, now, horizon=0), "horizon": 5}       # trains right up to the test block
    v = ao.run_battery(no_embargo, X, y, NOW, tests=("walk_forward",))["verdicts"][0]
    assert v["status"] == "fail" and v["violations"]


# ------------------------------------------------------------ empty / degenerate
def test_degenerate_inputs():
    X, y = make_panel(n_dates=10, n_tk=5)
    with pytest.raises(ValueError, match="empty"):
        ao.check_panel(X.iloc[:0], y.iloc[:0])
    with pytest.raises(ValueError, match="same index"):
        ao.check_panel(X, y.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="unknown"):
        ao.run_battery(ao.wf_evaluate, X, y, NOW, tests=("nope",))
    rep = ao.run_battery(ao.wf_evaluate, X, y, NOW)             # 10 dates: too few for any fold
    assert rep["overall"] == "fail" and "no metric" in rep["reason"]
    two = make_panel(n_tk=2, n_dates=120)
    v = ao.run_battery(lambda X, y, now: {"metric": 0.1, "t": 3.0}, *two, NOW, tests=("ticker_permutation",))["verdicts"][0]
    assert v["status"] == "inconclusive"
    rep = ao.run_battery(ao.wf_evaluate, *two, NOW)              # 2 tickers: no cross-section to rank
    assert rep["overall"] == "fail" and not rep["verdicts"]


def test_a_crashing_test_is_an_error_and_fails_the_battery(panel):
    X, y = panel
    calls = {"n": 0}

    def flaky(X, y, now):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return ao.wf_evaluate(X, y, now)
    rep = ao.run_battery(flaky, X, y, NOW, tests=("label_permutation",))
    assert rep["verdicts"][0]["status"] == "error" and rep["overall"] == "fail"


def test_default_evaluator_only_reads_rows_up_to_now(panel):
    X, y = panel
    cut = X.index.get_level_values(0).unique()[200]
    a = ao.wf_evaluate(X, y, cut)
    b = ao.wf_evaluate(X[X.index.get_level_values(0) <= cut], y[y.index.get_level_values(0) <= cut], cut)
    assert a["metric"] == b["metric"]
    assert a["predictions"].index.get_level_values(0).max() <= cut


def test_pattern_miner_adapter_smoke():
    X, y = make_panel(seed=5, n_dates=120, n_tk=30, n_feat=4)
    r = ao.miner_evaluator({"min_n": 100, "max_pairs": 50, "max_unless": 10})(X, y, NOW)
    assert np.isfinite(r["metric"]) and set(r["importance"]) == {"f0", "f1", "f2", "f3"}
